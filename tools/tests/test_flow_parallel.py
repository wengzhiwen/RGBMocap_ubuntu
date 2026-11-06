"""Preparation-only CPU parity tests; these are not GPU performance measurements."""
import copy
from datetime import timedelta
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'third_party/FlowHMR')]
HAS_TORCH = importlib.util.find_spec('torch') is not None and importlib.util.find_spec('einops') is not None


def model_and_inputs(start=False, skip=False):
    import torch
    from flowhmr.network.hymotion_mmdit_for_v2m import HunyuanMotionMMDiT
    torch.manual_seed(12)
    model = HunyuanMotionMMDiT(input_dim=12, feat_dim=32, output_dim=12,
        ctxt_input_dim={'feature': 16, 'camera_R': 9}, vtxt_input_dim=8,
        num_layers=3, num_heads=4, mlp_ratio=2., dropout=0., qkv_bias=True,
        text_refiner_module='flowhmr/network/token_refiner.SingleTokenRefiner',
        text_refiner_cfg={'num_layers': 1}, final_layer_cfg={'act_type': 'silu', 'zero_init': False},
        mask_mode='narrowband_v2m', attention_backend='query_batched', attention_query_chunk_size=4,
        narrowband_length=.1, narrowband_v2m_length=2., insert_start_token=start,
        with_long_skip_connection=skip).eval()
    # Fresh AdaLN gates are zero by default: activate them so attention/MLP
    # mistakes and padding changes actually influence the final prediction.
    from flowhmr.network.mmdit.modulate_layers import ModulateDiT
    generator = torch.Generator().manual_seed(77)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, ModulateDiT):
                for value in (module.linear.weight, module.linear.bias):
                    value.copy_(torch.randn(value.shape, generator=generator) * .05)
    mask = torch.ones(2, 9, dtype=torch.bool)
    context = {'feature': torch.randn(2, 9, 16), 'camera_R': torch.randn(2, 9, 9)}
    args = dict(x=torch.randn(2, 9, 12), ctxt_input=context, vtxt_input=torch.randn(2, 1, 8),
                timesteps=torch.tensor([.2, .9]), x_mask_temporal=mask, ctxt_mask_temporal=mask.clone())
    return model, args


def _tp_child(rank, store, output):
    import torch
    import torch.distributed as dist
    from mocap.experimental.flow_parallel import tensor_parallel
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + store, rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        results = []
        for start, skip, transport in ((False, False, 'device'), (True, True, 'device'), (False, False, 'host'), (True, True, 'host')):
            model, args = model_and_inputs(start, skip)
            masked = {**args, 'x_mask_temporal': args['x_mask_temporal'].clone(),
                      'ctxt_mask_temporal': args['ctxt_mask_temporal'].clone()}
            masked['x_mask_temporal'][:, -2:] = False
            masked['ctxt_mask_temporal'][:, -3:] = False
            with torch.no_grad():
                expected = [model(**batch).clone() for batch in (args, masked)]
                assert not torch.equal(expected[0], expected[1]), 'Padding must affect this fixture.'
                tensor_parallel(model, reduce_transport=transport)
                assert model._experimental_parallel['parameters_local'] < model._experimental_parallel['parameters_before']
                for batch, reference in zip((args, masked), expected):
                    actual = model(**batch)
                    error = float((actual - reference).abs().max())
                    assert torch.allclose(actual, reference, rtol=2e-5, atol=3e-6), error
                    assert torch.isfinite(actual).all()
                    results.append(error)
                # A wrong partial reduction must fail parity. Both ranks make
                # the same mutation so this remains a valid collective order.
                model.double_blocks[0].motion_out_proj.weight.mul_(.5)
                wrong = model(**args)
                assert not torch.allclose(wrong, expected[0], rtol=2e-5, atol=3e-6), 'Inactive sharded-layer control.'
        Path(output + str(rank) + '.json').write_text(json.dumps(results))
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(HAS_TORCH, 'requires a pinned Torch/einops environment')
class FlowParallel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        import torch
        torch.set_num_threads(cls.previous_threads)

    def test_tp2_full_network_matches_replicated_reference_with_gloo(self):
        import torch.multiprocessing as mp
        with tempfile.TemporaryDirectory() as folder:
            base = str(Path(folder) / 'rank')
            mp.spawn(_tp_child, args=(str(Path(folder) / 'store'), base), nprocs=2, join=True)
            results = [json.loads(Path(base + str(rank) + '.json').read_text()) for rank in range(2)]
        self.assertEqual(len(results[0]), 8)
        self.assertEqual(results[0], results[1])

    def test_pp2_full_network_preserves_output_and_padding(self):
        import torch
        from mocap.experimental.flow_parallel import pipeline_parallel
        for start, skip, transport in ((False, False, 'direct'), (True, True, 'direct'), (False, False, 'host'), (True, True, 'host')):
            model, args = model_and_inputs(start, skip)
            args['x_mask_temporal'][:, -2:] = False
            args['ctxt_mask_temporal'][:, -3:] = False
            reference = copy.deepcopy(model)
            pipeline_parallel(model, devices=('cpu', 'cpu'), peer_transport=transport)
            with torch.no_grad():
                expected, actual = reference(**args), model(**args)
            self.assertTrue(torch.equal(expected, actual))
            self.assertFalse(model._experimental_parallel['overlap'])
            self.assertEqual(len(model.single_blocks), len(reference.single_blocks))

    def test_requires_fresh_evaluated_model_and_two_ranks(self):
        from mocap.experimental.flow_parallel import pipeline_parallel, tensor_parallel
        model, _ = model_and_inputs()
        model.train()
        with self.assertRaisesRegex(ValueError, 'evaluated'):
            pipeline_parallel(model, devices=('cpu', 'cpu'))
        model.eval()
        with self.assertRaisesRegex(ValueError, 'two-rank'):
            tensor_parallel(model)
        pipeline_parallel(model, devices=('cpu', 'cpu'))
        with self.assertRaisesRegex(ValueError, 'fresh model'):
            pipeline_parallel(model, devices=('cpu', 'cpu'))


if __name__ == '__main__':
    unittest.main()
