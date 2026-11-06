"""Performance prerequisite checks execute without starting CUDA or a model."""
import copy
import json
import importlib.util
from pathlib import Path
import tempfile
import unittest

from tools.benchmark_flow import idle_devices, require_baselines, require_gate
from unittest.mock import patch


class Prerequisites(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)

    def save(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value), encoding='utf-8')
        return path

    def test_partial_and_preliminary_migration_reports_are_refused(self):
        for value in ({}, dict(schema=1, scope='gb10-migration', passed=False),
                      dict(schema=1, scope='gb10-migration', passed=True, gpu='GB10', checks={'cached_rerun': True})):
            with self.assertRaises(ValueError):
                require_gate(self.save('migration.json', value))

    def test_both_platforms_same_identity_and_five_rounds_are_required(self):
        pro = dict(scope='warm-generator', mode='single', passed=True,
            rounds=[{}] * 5, warmups=1, identity={'input': 'same'},
            hardware={'gpus': [{'name': 'NVIDIA RTX PRO 4000 Blackwell'}]})
        spark = copy.deepcopy(pro)
        spark['hardware']['gpus'][0]['name'] = 'NVIDIA GB10'
        a, b = self.save('pro.json', pro), self.save('spark.json', spark)
        self.assertEqual(require_baselines([a, b]), pro)
        for key, value in (('identity', {'input': 'different'}), ('rounds', [{}] * 4), ('mode', 'tp2'), ('passed', False)):
            changed = {**spark, key: value}
            with self.assertRaises(ValueError):
                require_baselines([a, self.save('changed.json', changed)])
        with self.assertRaises(ValueError):
            require_baselines([a, a])

    def test_contention_check_uses_selected_gpu_without_stopping_work(self):
        devices = '0, GPU-zero, NVIDIA RTX PRO 4000\n1, GPU-one, NVIDIA RTX PRO 4000\n'
        apps = 'GPU-one, 1234, unrelated-worker\n'
        with patch('tools.benchmark_flow.subprocess.check_output', side_effect=[devices, apps]), patch.dict('os.environ', {}, clear=True):
            self.assertEqual(idle_devices()[0][0], '0')
        with patch('tools.benchmark_flow.subprocess.check_output', side_effect=[devices, apps]), patch.dict('os.environ', {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, 'existing compute work'):
                idle_devices('tp2')
        with patch('tools.benchmark_flow.subprocess.check_output', side_effect=[devices, apps]), patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': '1'}, clear=True):
            with self.assertRaises(RuntimeError):
                idle_devices()


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('einops'), 'requires pinned Torch/einops')
class NumericalDisagreement(unittest.TestCase):
    def test_identical_and_small_rotation_and_large_translation(self):
        import sys
        import torch
        import math
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'third_party/FlowHMR'))
        from mocap.experimental.flow_benchmark import differences
        reference = dict(trans=torch.zeros(1, 1, 3), shapes=torch.zeros(1, 1, 16),
            foot_contacts=torch.zeros(1, 1, 4), rot6d=torch.tensor([1., 0., 0., 1., 0., 0.]).reshape(1, 1, 1, 6))
        self.assertEqual(differences(reference, reference)['local_rotation_max_degrees'], 0.)
        actual = {key: value.clone() for key, value in reference.items()}
        radians = math.radians(.001)
        actual['rot6d'][0, 0, 0] = torch.tensor([math.cos(radians), -math.sin(radians), math.sin(radians), math.cos(radians), 0., 0.])
        self.assertAlmostEqual(differences(reference, actual)['local_rotation_max_degrees'], .001, places=5)
        actual['trans'] += 1.
        with self.assertRaisesRegex(ValueError, 'tolerance'):
            differences(reference, actual)


if __name__ == '__main__':
    unittest.main()
