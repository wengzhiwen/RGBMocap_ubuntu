"""Worker implementation for tools/benchmark_flow.py; no production routing hooks."""
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import threading
import time

from mocap.paths import ROOT
from mocap.storage import atomic_json, sha256


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def conditioning_identity(project):
    """Content identities survive copies and differing mtimes between machines."""
    from mocap.models import BY_KEY
    names = ('input.mp4', 'flow_tokens.pt', 'flow_camera.npz')
    code = [ROOT / 'mocap/flow_backend.py', ROOT / 'mocap/flow_camera.py']
    code += sorted((ROOT / 'third_party/FlowHMR/flowhmr').rglob('*.py'))
    digest = hashlib.sha256()
    for path in code:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    assets = ROOT / 'third_party/FlowHMR/assets'
    return dict(files={name: sha256(project / name) for name in names},
        models={key: sha256(BY_KEY[key].path) for key in ('flowhmr', 'flow_config', 'smplh')},
        auxiliary={name: sha256(assets / name) for name in ('motion_stats.json', 'body_models/smpl_neutral_J_regressor.pt')},
        generator_code=digest.hexdigest(), options=dict(seed=0, ode_steps=20, cfg_scale=1.,
        precision='FP32', attention='query_batched MATH', query_chunk=256, full_context=True))


def verify(request):
    import numpy as np
    import torch
    from mocap.flow_backend import validate_motion
    from mocap.flow_hamer import load_hands
    from mocap.motion_tools import selected_export
    from mocap.result_artifacts import file_stat
    project = Path(request['project'])
    n = request['frames']
    state = load(project / 'project.json')
    if state['frames'] != n or state['fps'] != 30 or not state['flow_options']['hamer']:
        raise ValueError('Unexpected migration input/options.')
    required = ('track', 'flow_camera', 'flow_features', 'flowhmr', 'correct', 'flow_export')
    if any(state['stages'].get(stage, {}).get('status') != 'complete' for stage in required):
        raise ValueError('Migration stages are incomplete.')
    if list(project.glob('*pending*.json')):
        raise ValueError('Migration left pending generations.')
    for stage in ('flow_camera', 'flow_features', 'flowhmr'):
        receipt = load(project / (stage + '_receipt.json'))
        if any(file_stat(project / name) != stamp for name, stamp in receipt['outputs'].items()):
            raise ValueError('Migration output receipt mismatch: ' + stage)
    values = [torch.load(project / name, map_location='cpu', weights_only=False) for name in
              ('flow_prediction.pt', 'flow_hamer_prediction.pt', 'hmr4d_results.pt', 'corrected_initial.pt')]
    for value in values:
        validate_motion(value, n)
        for tensor in value.values():
            if isinstance(tensor, torch.Tensor) and not torch.isfinite(tensor).all():
                raise ValueError('Non-finite migration output.')
    original, hands, placed, corrected = values
    if not torch.equal(original['rot6d'][:, :, :22], hands['rot6d'][:, :, :22]):
        raise ValueError('HaMeR changed body/root/wrist rotations.')
    for key in ('trans', 'shapes', 'foot_contacts', 'root_rot6d', 'smooth_root_pos', 'pelvis_world'):
        if key in original and not torch.equal(original[key], hands[key]):
            raise ValueError('HaMeR changed ' + key)
    if not torch.equal(placed['rot6d'][:, :, 22:], corrected['rot6d'][:, :, 22:]):
        raise ValueError('Correction changed finger rotations.')
    _, valid = load_hands(project, n)
    counts = valid.sum(axis=0).astype(int).tolist()
    if min(counts) == 0:
        raise ValueError('Need real valid HaMeR inference for both hands.')
    for side, offset in enumerate((22, 37)):
        unavailable = ~torch.from_numpy(valid[:, side])
        if not torch.equal(original['rot6d'][0, unavailable, offset:offset+15], hands['rot6d'][0, unavailable, offset:offset+15]):
            raise ValueError('Unavailable hand crops changed fallback fingers.')
    for kind in ('initial', 'corrected_initial'):
        source = selected_export(project, kind)
        if source is None or f'Frames: {n}' not in source.read_text(encoding='utf-8'):
            raise ValueError('Selected export is unavailable or has incorrect frames.')
    report = load(project / 'corrected_initial_correction.json')
    if report.get('native_seconds', 0) <= 0 or not report.get('backend_sha256'):
        raise ValueError('No actual native correction evidence.')
    video = project / 'input.mp4'
    import cv2
    capture = cv2.VideoCapture(str(video))
    decoded = hashlib.sha256()
    decoded_count = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok: break
            decoded.update(np.ascontiguousarray(frame).tobytes())
            decoded_count += 1
    finally:
        capture.release()
    if decoded_count != n:
        raise ValueError('Unexpected decoded working-video length.')
    checks = dict(frames=n, finite=True, hamer_valid=counts, hamer_body_preserved=True,
        correction_fingers_preserved=True, selected_results=True, native_correction=True,
        decoded_sha256=decoded.hexdigest(), input_sha256=sha256(video),
        results={name: sha256(project / name) for name in
                 ('flow_prediction.pt', 'flow_hamer_prediction.pt', 'hmr4d_results.pt', 'corrected_initial.pt')})
    checks['runtimes'] = dict(flow=load(project / 'flowhmr_receipt.json')['identity']['runtime'],
        hamer=load(project / 'flow_hamer_receipt.json')['identity']['source']['runtime'])
    checks['generator_identity'] = conditioning_identity(project)
    atomic_json(Path(request['output']) / 'checks.json', checks)


def differences(reference, actual, skeleton=None):
    """Numerical disagreement, not accuracy against ground truth."""
    import torch
    metrics = {}
    for key in ('trans', 'shapes', 'rot6d', 'foot_contacts'):
        a, b = reference[key].cpu(), actual[key].cpu()
        if a.shape != b.shape or not torch.isfinite(b).all():
            raise ValueError('Invalid parallel result: ' + key)
        metrics[key + '_max_abs'] = float((a - b).abs().max())
        if not torch.allclose(a, b, rtol=1e-3, atol=1e-4):
            raise ValueError(f'Parallel parity tolerance exceeded: {metrics}')
    metrics['translation_max_m'] = float((reference['trans'].cpu() - actual['trans'].cpu()).norm(dim=-1).max())
    from flowhmr.core.math.geometry import rot6d_to_rotation_matrix
    a = rot6d_to_rotation_matrix(reference['rot6d'].cpu().double())
    b = rot6d_to_rotation_matrix(actual['rot6d'].cpu().double())
    cosine = (((a.transpose(-1, -2) @ b).diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
    metrics['local_rotation_max_degrees'] = 0. if torch.equal(reference['rot6d'], actual['rot6d']) else float(torch.rad2deg(torch.acos(cosine)).max())
    if skeleton is not None:
        def joints(value):
            return skeleton(dict(rot6d=value['rot6d'][0].cpu(), trans=value['trans'][0].cpu(),
                                 shapes=value['shapes'].reshape(1, 16).cpu()))['keypoints3d']
        with torch.no_grad():
            metrics['world_fk_joint_max_m'] = float((joints(reference) - joints(actual)).norm(dim=-1).max())
    return metrics


def hardware(torch, devices):
    import psutil
    command = ['nvidia-smi', '--query-gpu=index,name,uuid,driver_version,pstate,temperature.gpu,power.draw,clocks.sm,clocks.mem', '--format=csv,noheader']
    try: telemetry = subprocess.check_output(command, text=True, timeout=10).strip()
    except (OSError, subprocess.SubprocessError): telemetry = 'unavailable'
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    return dict(machine=platform.machine(), cpu=platform.processor(), cpu_count=os.cpu_count(),
        ram_available_bytes=psutil.virtual_memory().available, python=platform.python_version(),
        torch=str(torch.__version__), cuda=torch.version.cuda, git_revision=revision,
        tf32_matmul=torch.backends.cuda.matmul.allow_tf32, tf32_cudnn=torch.backends.cudnn.allow_tf32,
        gpus=[dict(device=d, name=torch.cuda.get_device_name(d), capability=torch.cuda.get_device_capability(d)) for d in devices],
        nvidia_smi=telemetry, memory_note='CUDA allocation and RSS may share physical RAM on GB10; do not sum them.')


class RssSampler:
    def __init__(self, include_children=False):
        self.include_children = include_children

    def __enter__(self):
        import psutil
        self.process = psutil.Process()
        self.peak = self.process.memory_info().rss
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()
        return self

    def sample(self):
        while not self.stop.wait(.05):
            self.peak = max(self.peak, self.current())

    def current(self):
        import psutil
        total = self.process.memory_info().rss
        if self.include_children:
            for child in self.process.children(recursive=True):
                try: total += child.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied): pass
        return total

    def __exit__(self, *unused):
        self.peak = max(self.peak, self.current())
        self.stop.set()
        self.thread.join()


def generator_rank(rank, request_path, store=None):
    import torch
    import torch.distributed as dist
    import numpy as np
    from mocap.compute_policy import configure_worker_cpu
    from mocap.flow_backend import activate, validate_motion
    from mocap.flow_camera import relative_camera
    from mocap.model_session import set_memory_budget
    request = load(request_path)
    output = Path(request['output'])
    mode = request['mode']
    devices = [rank] if mode == 'tp2' else [0, 1] if mode == 'pp2' else [0]
    if torch.cuda.device_count() < (2 if mode != 'single' else 1):
        raise ValueError('Insufficient visible CUDA devices.')
    if mode != 'single' and any('PRO 4000' not in torch.cuda.get_device_name(d) for d in range(2)):
        raise ValueError('This dual-card experiment is for the two PRO 4000 GPUs.')
    torch.set_num_threads(configure_worker_cpu('flowhmr'))
    for device in devices:
        with torch.cuda.device(device): set_memory_budget(*torch.cuda.mem_get_info(device))
    torch.cuda.set_device(devices[0])
    initialized = False
    try:
        if mode == 'tp2':
            dist.init_process_group('nccl', init_method='file://' + store, rank=rank, world_size=2, timeout=timedelta(minutes=5))
            initialized = True
        activate()
        from flowhmr.pipeline.v2m_inference import build_pipeline, load_yaml, build_feature_dict
        from mocap.models import BY_KEY
        project = Path(request['project'])
        identity = conditioning_identity(project)
        if request['baseline'] and identity != request['baseline']['identity']:
            raise ValueError('Conditioning/weights/code differs from the established baselines.')
        tokens = torch.load(project / 'flow_tokens.pt', map_location='cpu', weights_only=False)
        n = len(tokens)
        if not 1 <= n <= 5000 or tokens.shape != (n, 3072) or not torch.isfinite(tokens).all():
            raise ValueError('Invalid conditioning tokens.')
        with np.load(project / 'flow_camera.npz', allow_pickle=False) as data:
            rt = torch.from_numpy(data['camera_RT_yup'].copy())
        if rt.shape != (n, 4, 4) or not torch.isfinite(rt).all():
            raise ValueError('Invalid camera conditioning.')
        device = torch.device('cuda', devices[0])
        config = load_yaml(str(BY_KEY['flow_config'].path))
        config['network_module_args'].update(attention_backend='query_batched', attention_query_chunk_size=256)
        started = time.perf_counter()
        pipeline = build_pipeline(config, device=str(device), ckpt=str(BY_KEY['flowhmr'].path), validation_steps=20)
        weights = torch.load(BY_KEY['flowhmr'].path, map_location='cpu', weights_only=False, mmap=True)
        weights = weights.get('model_state_dict', weights.get('state_dict', weights))
        if any(name not in weights or weights[name].shape != value.shape for name, value in pipeline.named_parameters()):
            raise ValueError('Missing required learned checkpoint parameters.')
        del weights
        if mode == 'tp2':
            from .flow_parallel import tensor_parallel
            tensor_parallel(pipeline.motion_transformer)
        elif mode == 'pp2':
            from .flow_parallel import pipeline_parallel
            pipeline_parallel(pipeline.motion_transformer)
        feature = build_feature_dict(tokens, relative_camera(rt), device)
        for d in devices: torch.cuda.synchronize(d)
        load_seconds = time.perf_counter() - started
        info = hardware(torch, devices)
        if request['baseline'] and (info['torch'] != request['baseline']['hardware']['torch'] or info['cuda'] != request['baseline']['hardware']['cuda']):
            raise ValueError('Compare dual-card results in the same runtime as the PRO 4000 baseline.')
        reference = torch.load(request['reference'], map_location='cpu', weights_only=False) if request['reference'] else None
        from flowhmr.core.bodymodels.smpl_skeleton import SMPLSkeleton
        skeleton = SMPLSkeleton(model_path=str(BY_KEY['smplh'].path))
        if request['reference']:
            if sha256(Path(request['reference'])) != request['baseline']['reference_sha256']:
                raise ValueError('Reference file does not belong to the PRO 4000 baseline.')
        results = []
        for repetition in range(-request['warmups'], request['rounds']):
            for d in devices:
                torch.cuda.synchronize(d)
                torch.cuda.reset_peak_memory_stats(d)
            if initialized: dist.barrier()
            with RssSampler() as rss:
                started = time.perf_counter()
                with torch.inference_mode():
                    motion = pipeline.generate(feature=feature, seeds=[0], length=n, cfg_scale=1.)
                for d in devices: torch.cuda.synchronize(d)
                seconds = time.perf_counter() - started
            peaks = [dict(device=d, allocated=torch.cuda.max_memory_allocated(d), reserved=torch.cuda.max_memory_reserved(d)) for d in devices]
            motion = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in motion.items()}
            validate_motion(motion, n)
            if reference is None:
                reference = motion
                if rank == 0: torch.save(reference, output / 'reference.pt')
            metrics = differences(reference, motion, skeleton)
            if initialized:
                rows = [None, None]
                dist.all_gather_object(rows, dict(seconds=seconds, peaks=peaks, rss_bytes=rss.peak, differences=metrics))
            else:
                rows = [dict(seconds=seconds, peaks=peaks, rss_bytes=rss.peak, differences=metrics)]
            item = dict(round=repetition, seconds=max(row['seconds'] for row in rows), ranks=rows)
            if repetition >= 0: results.append(item)
            if rank == 0:
                print(json.dumps(dict(type='benchmark_round', warmup=repetition < 0, **item)), flush=True)
                atomic_json(output / 'partial.json', dict(scope='warm-generator', mode=mode, passed=False, rounds=results))
            del motion
        if conditioning_identity(project) != identity:
            raise ValueError('Conditioning or model source changed during measurement.')
        if rank == 0:
            reference_path = Path(request['reference']) if request['reference'] else output / 'reference.pt'
            atomic_json(output / 'result.json', dict(schema=1, scope='warm-generator', mode=mode, passed=True,
                identity=identity, hardware=info, hardware_after=hardware(torch, devices),
                cpu_threads=torch.get_num_threads(), frames=n, warmups=request['warmups'], rounds=results,
                load_seconds=load_seconds, reference_sha256=sha256(reference_path),
                partition=getattr(pipeline.motion_transformer, '_experimental_parallel', None),
                excludes='SAM, tracking, camera estimation, HaMeR, correction, export; resident generator only'))
    finally:
        if initialized: dist.destroy_process_group()


def main(argv):
    request_path = Path(argv[1])
    if argv[0] == 'verify':
        verify(load(request_path))
    elif argv[0] == 'generator':
        request = load(request_path)
        with RssSampler(include_children=True) as memory:
            if request['mode'] == 'tp2':
                import torch.multiprocessing as mp
                # FileStore avoids a fixed TCP port and is unique to this new experiment.
                mp.spawn(generator_rank, args=(str(request_path), str(Path(request['output']) / 'nccl-store')), nprocs=2, join=True)
            else:
                generator_rank(0, request_path)
        result_path = Path(request['output']) / 'result.json'
        result = load(result_path)
        result.update(whole_job_sampled_tree_rss_bytes=memory.peak,
            tree_rss_note='50 ms samples across loading and rounds; sums process RSS, including shared pages, not physical RAM.')
        atomic_json(result_path, result)
    else:
        raise ValueError('Unknown experiment operation.')
