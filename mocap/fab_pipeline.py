"""Publish one pinned FAB capture into the app's shared SMPL-X motion result.

Ubuntu only. This runs inside the shared CPU worker: FAB itself stays in its own
interpreter (see `mocap/fab_runtime.py` and `mocap/fab_infer_entry.py`), and only
numeric arrays cross the boundary. The published `hmr4d_results.pt` keeps the
canonical SMPL-X layout, so the existing mesh preview, contact correction,
retarget and export stages read a FAB project exactly like a GVHMR one. Nothing
here installs, downloads or mutates anything outside one project folder.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time, traceback, uuid
from pathlib import Path

from .paths import ROOT
from .storage import atomic_json
from .result_artifacts import file_stat

STAGE = 'fab_reconstruct'
KIND = 'initial'
MODEL = 'fab-l'
FPS = 30.0
RECEIPT = 'fab_result.json'
PENDING = 'fab_pending.json'
NUMERIC = 'fab_motion.npz'
MOTION = 'hmr4d_results.pt'
WORK = 'fab_work'
INPUTS = ('input.mp4', 'tracking.npz')
OUTPUTS = (MOTION, NUMERIC)
LOG_BYTES = 4096


def emit(kind, **data):
    print('@@MOCAP@@' + json.dumps(dict(type=kind, **data), ensure_ascii=False), flush=True)


def state(project):
    return json.loads((Path(project) / 'project.json').read_text(encoding='utf-8'))


def is_project(project):
    try:
        return state(project).get('pipeline', {}).get('model') == 'fab'
    except (OSError, ValueError, AttributeError, TypeError):
        return False


def requirements(project):
    """Validate the already committed project inputs; nothing is created here."""
    project = Path(project)
    manifest = state(project)
    if manifest.get('pipeline', {}).get('model') != 'fab':
        raise ValueError('This project was not prepared for FAB. Prepare the source video as a FAB project first.')
    frames = int(manifest['frames'])
    fps = float(manifest.get('fps', FPS))
    if frames < 1:
        raise ValueError('A FAB project needs at least one frame.')
    if abs(fps - FPS) > 1e-3:
        raise ValueError(f'FAB reconstructs {FPS:g} fps projects; this project is {fps:g} fps.')
    for name in INPUTS:
        if not (project / name).is_file():
            raise ValueError(f'Run Find person/actor tracking first: {name} is missing from this project.')
    return project, manifest


def receipt(project):
    path = Path(project) / RECEIPT
    try:
        if not path.is_file() or path.stat().st_size > 262144:
            return None
        data = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def identity(project, manifest):
    """Everything a capture must reproduce to be reusable: pinned model/code/assets,
    the registered target body and the project's own input stamps."""
    from . import fab_runtime
    from .models import BY_KEY
    body = BY_KEY['smplx'].path
    return dict(version=1, model=MODEL, frames=int(manifest['frames']), fps=float(manifest.get('fps', FPS)),
                inputs={name: file_stat(project / name) for name in INPUTS},
                body=(file_stat(body) if body.is_file() else None),
                estimator=fab_runtime.identity())


def current(project):
    """FAB-specific source guard for preview, correction, export and camera reads.

    Stats only: it never opens a motion tensor and never hashes a weight file.
    Missing optional estimator files after a committed capture never invalidate
    reopening that capture.
    """
    project = Path(project)
    try:
        if (project / PENDING).exists():
            return False
        record = receipt(project)
        if not record or record.get('version') != 1 or record.get('model') != MODEL:
            return False
        if record.get('source_model') != 'fab' or record.get('static_camera') is not True:
            return False
        if not is_project(project):
            return False
        frames, fps = int(record['frames']), float(record['fps'])
        manifest = state(project)
        if frames < 1 or abs(fps - FPS) > 1e-3 or int(manifest['frames']) != frames:
            return False
        if abs(float(manifest.get('fps', FPS)) - fps) > 1e-3:
            return False
        outputs = record['outputs']
        if set(outputs) != set(OUTPUTS) or not all(file_stat(project / name) == stamp for name, stamp in outputs.items()):
            return False
        identity = record.get('identity') or {}
        for name, stamp in (identity.get('inputs') or {}).items():
            if not (project / name).is_file() or file_stat(project / name) != stamp:
                return False
        body = identity.get('body')
        if body is not None:
            # A baked capture stays reopenable without its registered body model,
            # exactly as an uncorrected preview does; compare it when available.
            from .models import BY_KEY
            path = BY_KEY['smplx'].path
            if path.is_file() and file_stat(path) != body:
                return False
        from . import fab_runtime
        weights = (record.get('weights') or {}).get('sha256') or {}
        if any(weights.get(name) != digest for name, digest in fab_runtime.WEIGHT_SHA256.items()):
            return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def sources(project):
    """Stamps of the motion and the inputs behind it, in the shared receipt form."""
    project = Path(project)
    names = (MOTION, NUMERIC, *INPUTS)
    return {name: file_stat(project / name) for name in names if (project / name).is_file()}


def camera(project):
    """Saved positive static-camera evidence for the GUI and export, or None."""
    project = Path(project)
    if not current(project):
        return None
    record = receipt(project)
    return dict(model=MODEL, static=True, method=record['method'], frames=int(record['frames']),
                fps=float(record['fps']), K=record['K'], world_from_camera=record['world_from_camera'],
                floor_estimate_m=float(record['floor_estimate_m']), receipt=RECEIPT,
                receipt_stat=file_stat(project / RECEIPT))


def tail(path, limit=LOG_BYTES):
    try:
        text = path.read_text(encoding='utf-8', errors='replace').strip()
    except OSError:
        return ''
    return text[-limit:]


def tracking_intrinsics(project):
    import numpy as np
    try:
        with np.load(Path(project) / 'tracking.npz', allow_pickle=False) as data:
            return np.asarray(data['K'], dtype=float).tolist()
    except (OSError, ValueError, KeyError):
        return None


def load_body():
    from .models import BY_KEY
    path = BY_KEY['smplx'].path
    if not path.is_file():
        raise ValueError('Import the registered SMPL-X neutral body model in Model Manager first: '
                         'the FAB actor is mapped onto it, so its shape basis and pelvis pivot are required.')
    import smplx
    return smplx.SMPLX(str(path), gender='neutral', use_pca=False, flat_hand_mean=False).eval()


def payload(parameters):
    """Numeric-only torch payload in the canonical shared result layout."""
    import torch
    tensor = lambda value: torch.as_tensor(value)
    return dict(smpl_params_incam={key: tensor(value) for key, value in parameters['smpl_params_incam'].items()},
                smpl_params_global={key: tensor(value) for key, value in parameters['smpl_params_global'].items()},
                net_outputs={key: tensor(value) for key, value in parameters['net_outputs'].items()},
                source_model='fab', hand_pose_policy='registered-neutral-untracked-v1',
                frames=int(parameters['frames']), fps=float(parameters['fps']))


def run_child(project, output, log, want):
    """One isolated FAB child; relay its progress events and keep its own log."""
    from . import fab_runtime
    command = fab_runtime.command(project, output)
    emit('progress', stage=STAGE, current=0, total=3, message=f'Estimating {want["model"]} motion in the separate FAB runtime')
    with log.open('wb') as errors:
        process = subprocess.Popen(command, cwd=str(ROOT), env=fab_runtime.child_environment(),
                                   stdout=subprocess.PIPE, stderr=errors, text=True, encoding='utf-8',
                                   errors='backslashreplace', bufsize=1)
        try:
            for line in process.stdout:
                line = line.rstrip('\n')
                if line.startswith('@@MOCAP@@'):
                    print(line, flush=True)  # the child's genuine progress events
                elif line:
                    print(line, file=sys.stderr, flush=True)
            code = process.wait()
        except BaseException:
            # Cancellation must not leave an inference child holding a GPU.
            try:
                process.kill()
            except OSError:
                pass
            try:
                process.wait()
            except OSError:
                pass
            raise
    return code


def stage_capture(project, want, weights, emit):
    """Everything is staged in the work folder; committed files stay untouched."""
    from . import fab_data
    from .engines import save_tensor
    token = uuid.uuid4().hex[:12]
    work = project / WORK
    work.mkdir(parents=True, exist_ok=True)
    capture, staged, log = work / f'capture-{token}.npz', work / f'payload-{token}.pt', work / f'capture-{token}.log'
    code = run_child(project, capture, log, want)
    if code != 0 or not capture.is_file():
        detail = tail(log)
        raise RuntimeError(f'FAB reconstruction stopped before writing a result (exit {code}). '
                           + (f' Last runtime output: {detail}' if detail else '')
                           + ' Your previous motion result and video are unchanged.')
    numeric = fab_data.load(capture, want['frames'])
    emit('progress', stage=STAGE, current=1, total=3, message='Mapping the FAB actor onto the registered SMPL-X body')
    parameters = fab_data.result_parameters(numeric, load_body())
    tracked = tracking_intrinsics(project)
    match = tracked is not None and bool(
        abs(tracked[0][0] - numeric['K'][0][0]) <= 0.01 * max(abs(tracked[0][0]), 1.)
        and abs(tracked[1][1] - numeric['K'][1][1]) <= 0.01 * max(abs(tracked[1][1]), 1.)
        and abs(tracked[0][2] - numeric['K'][0][2]) <= 2. and abs(tracked[1][2] - numeric['K'][1][2]) <= 2.)
    emit('progress', stage=STAGE, current=2, total=3, message='Publishing the shared SMPL-X motion result')
    save_tensor(payload(parameters), staged)
    return dict(capture=capture, staged=staged, log=log, numeric=numeric, parameters=parameters, weights=weights,
                tracked_K=tracked, intrinsics_match=match)


def commit(project, want, staged, elapsed):
    """Atomic publication of both payloads, then the receipts that make them visible."""
    from . import fab_runtime
    for name, stamp in (want.get('inputs') or {}).items():
        if not (project / name).is_file() or file_stat(project / name) != stamp:
            raise RuntimeError('FAB inputs changed before publication')
    body = want.get('body')
    if body is not None:
        from .models import BY_KEY
        path = BY_KEY['smplx'].path
        if path.is_file() and file_stat(path) != body:
            raise RuntimeError('FAB body model changed before publication')
    atomic_json(project / PENDING, dict(stage=STAGE, model=want['model'], started=time.time(), identity=want))
    os.replace(staged['capture'], project / NUMERIC)
    os.replace(staged['staged'], project / MOTION)
    shape = staged['parameters']['shape']
    atomic_json(project / RECEIPT, dict(
        version=1, model=want['model'], source_model='fab', frames=want['frames'], fps=want['fps'],
        static_camera=True, method='One fixed FAB camera: camera-to-world rotation plus one clip-wide floor shift',
        K=staged['numeric']['K'].tolist(), world_from_camera=staged['numeric']['world_from_camera'].tolist(),
        floor_estimate_m=float(staged['numeric']['floor_estimate']),
        tracking_K=staged['tracked_K'], intrinsics_match=staged['intrinsics_match'],
        shape=dict(betas_residual_m=float(shape['residual_m']), height_m=float(shape['height_m']),
                   description=shape['model_shape']),
        outputs={name: file_stat(project / name) for name in OUTPUTS},
        weights=dict(pinned=fab_runtime.WEIGHT_SHA256, sha256=staged['weights']),
        identity=want, elapsed=elapsed))
    atomic_json(project / f'{KIND}_sources.json', sources(project))
    manifest = state(project)
    manifest.setdefault('stages', {})[STAGE] = dict(status='complete', output=str(project / MOTION),
                                                    engine_version=1, elapsed=elapsed, model=want['model'])
    manifest.setdefault('pipeline', {})['result'] = KIND
    atomic_json(project / 'project.json', manifest)
    (project / PENDING).unlink(missing_ok=True)
    # The scratch capture, payload and runtime log are superseded by the
    # committed numeric record; a failed run keeps them for diagnosis.
    for path in (staged['capture'], staged['staged'], staged['log']):
        path.unlink(missing_ok=True)


def reconstruct(project, emit, force=False):
    project, manifest = requirements(project)
    from . import fab_runtime
    problems = fab_runtime.missing()
    if problems:
        emit('blocked', stage=STAGE, message='Prepare the FAB runtime and its files first: '
             + ', '.join(row['name'] for row in problems), models=[row['key'] for row in problems],
             hints=[row['hint'] for row in problems])
        return None, False
    import contextlib
    import fcntl

    class FabBusy(Exception):
        pass

    @contextlib.contextmanager
    def publishing_lock():
        handle = open(project / '.fab.lock', 'a+')
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.close()
            raise FabBusy from exc
        except OSError:
            handle.close()
            raise
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()

    try:
        with publishing_lock():
            want = identity(project, manifest)
            previous = receipt(project)
            if not force and previous and previous.get('identity') == want and current(project):
                emit('progress', stage=STAGE, current=3, total=3, message='Reusing the verified FAB capture.')
                return project / MOTION, True
            weights = fab_runtime.integrity()
            wrong = [name for name, digest in fab_runtime.WEIGHT_SHA256.items() if weights.get(name) != digest]
            if wrong:
                raise RuntimeError('FAB release weights do not match the pinned official manifest: '
                                   + ', '.join(wrong) + '. Re-download them (see the FAB guide); your existing results are unchanged.')
            manifest = state(project)
            manifest.setdefault('stages', {})[STAGE] = dict(status='running', engine_version=1, model=want['model'])
            atomic_json(project / 'project.json', manifest)
            started = time.monotonic()
            staged = stage_capture(project, want, weights, emit)
            commit(project, want, staged, time.monotonic() - started)
            return project / MOTION, False
    except FabBusy:
        emit('blocked', stage=STAGE, message='Another FAB publisher for this project is already running; retry later.')
        return None, False


def parse(argv):
    parser = argparse.ArgumentParser(prog='mocap.worker_progress fab_reconstruct',
                                     description='Reconstruct one fixed-camera FAB capture in this project.')
    parser.add_argument('stage', choices=(STAGE,))
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--force', action='store_true', help='Recompute a completed FAB capture.')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse(sys.argv[1:] if argv is None else list(argv))
    started = time.monotonic()
    try:
        if not sys.platform.startswith('linux'):
            raise RuntimeError('FAB reconstruction is only available on Ubuntu/Linux.')
        output, cached = reconstruct(args.project.resolve(), emit, force=args.force)
        if output is None:
            return 3
        emit('complete', stage=STAGE, output=str(output), cached=cached, elapsed=time.monotonic() - started)
        return 0
    except Exception as error:
        details = traceback.format_exc()
        print(details, file=sys.stderr, flush=True)
        project = args.project
        try:
            (project / f'{STAGE}_error.log').write_text(details, encoding='utf-8')
            state_path = project / 'project.json'
            if state_path.exists():
                state = json.loads(state_path.read_text(encoding='utf-8'))
                state.setdefault('stages', {})[STAGE] = dict(status='failed', error=str(error))
                atomic_json(state_path, state)
        except (OSError, ValueError):
            pass
        emit('failed', stage=STAGE, message=str(error))
        return 1
