"""Optional finger-only FAH pass, isolated in its own pinned `.venv-fah` runtime.

Off by default and Ubuntu-only. FAH replaces only the 15 local hand rotations per
side: root, body, wrist, trajectory, shape and the native contact tensors stay
exactly as the body model published them. Frames FAH cannot use keep the original
fingers, so a partial run never invents hand motion.

The parent side never imports FAH or Torch for its own runtime; the child (`run`,
`run_request`) is the only place FAH loads. Two callers share one inference loop:

- FlowHMR (`refine`/`run`): the project's own per-frame SAM body-derived left/right
  hand boxes (`flow_hand_boxes.npy`) with saved per-clip camera intrinsics; the
  fitted rotations are cached in the project and replace rot6d rows directly.
- FAB (`refine_request`/`run_request`): one numeric request from
  `mocap/fab_pipeline.py`, whose boxes come from FAB's own 2D hand keypoints.

There is no FAH detection pass, no video/world mode and no temporal smoothing.
"""
from pathlib import Path
import json, os, subprocess, uuid, zipfile
from .paths import ROOT
from .process_memory import peak_rss_bytes
from . import fah_runtime

PREDICTION = 'flow_fah_prediction.pt'
HANDS = 'flow_fah_hands.npz'
RECEIPT = 'flow_fah_receipt.json'
BOXES = 'flow_hand_boxes.npy'
CAMERA = 'flow_camera.npz'
EVENT = '@@FLOW_FAH@@'
# Left, then right: the hand slots of the committed 52-joint SMPL-H rot6d layout.
ROT6D_SLOTS = (22, 37)
# Gauss-Newton iterations of the pinned fitter from the closed-form Procrustes start.
FIT_ITERS = 1
FIT_WEIGHT_SURFACE = 0.3
FIT_CENTRE = True
FIT_FD_EPS = 1e-2
# The pinned pipeline's own presence gate; FAH's side is taken from this project, not re-decided.
PRESENCE_MIN = 0.5
MAX_FRAMES = 5000
# The FAB request protocol: its boxes come from FAB's own 2D hand keypoints (mocap/fab_data.py).
REQUEST_HANDS = ('every frame, boxes derived from the FAB predictor\'s own 2D hand keypoints, '
                 'saved per-clip intrinsics, no FAH detection, no temporal smoothing')


def saved_flow_options(project, state=None):
    """Scene takes share the scene project's processing settings, as HaMeR does."""
    project = Path(project)
    state = json.loads((project / 'project.json').read_text(encoding='utf-8')) if state is None else state
    options = state.get('flow_options', {})
    if state.get('people_root'):
        root = (project / state['people_root']).resolve()
        if project.resolve().is_relative_to(root) and (root / 'people.json').is_file():
            options = json.loads((root / 'project.json').read_text(encoding='utf-8')).get('flow_options', options)
    return options if isinstance(options, dict) else {}


def enabled(project, state=None):
    """The saved draft choice. Only a committed receipt says what a result used."""
    value = saved_flow_options(project, state).get('fah', False)
    if type(value) is not bool:
        raise ValueError('Invalid FlowHMR setting: fah; expected on or off')
    if value:
        from .flow_hamer import enabled as hamer_enabled
        if hamer_enabled(project):
            raise ValueError('Choose either HaMeR or FAH finger refinement; both are selected.')
    return value


def committed(project):
    """Did the published generation actually use FAH? Never the current checkbox."""
    try:
        return bool(json.loads((Path(project) / 'flowhmr_receipt.json').read_text(encoding='utf-8'))
                    ['identity']['code'].get('fah'))
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return False


def prediction_name(project, generated=False):
    return PREDICTION if (enabled(project) if generated else committed(project)) else None


def hand_boxes_required(project, state=None):
    """SAM must publish the per-frame boxes for either optional finger pass."""
    from .flow_hamer import enabled as hamer_enabled
    return hamer_enabled(project) or enabled(project, state)


def readiness(root=None):
    return fah_runtime.readiness(ROOT if root is None else root)


def missing_names(root=None):
    return fah_runtime.missing(ROOT if root is None else root)


def intrinsics_binding(project):
    """Bind the exact per-clip intrinsics the pass consumes, not just a file name."""
    import hashlib
    import numpy as np
    path = Path(project) / CAMERA
    with np.load(path, allow_pickle=False) as data:
        values = np.asarray(data['K'], np.float64)
    if values.ndim not in (2, 3) or values.shape[-2:] != (3, 3) or not np.isfinite(values).all():
        raise ValueError('Invalid FlowHMR camera intrinsics for FAH')
    focal = values.reshape(-1, 3, 3)
    if (focal[:, 0, 0] <= 0).any() or (focal[:, 1, 1] <= 0).any():
        raise ValueError('FlowHMR camera intrinsics have a non-positive focal length')
    from .result_artifacts import file_stat
    return dict(file=file_stat(path), frames=int(focal.shape[0]), digest=hashlib.sha256(values.tobytes()).hexdigest())


def source_identity(root=None, hands=None):
    """Code, pinned source, package and gated-asset stamps of the FAH pass itself."""
    from . import fah_runtime as runtime
    return dict(version=1, code=runtime.code_identity(ROOT if root is None else root),
                runtime=runtime.identity(ROOT if root is None else root),
                fit=dict(iters=FIT_ITERS, solver='gn', weight_surface=FIT_WEIGHT_SURFACE, centre=FIT_CENTRE,
                         fd_eps=FIT_FD_EPS, init='closed-form per-part Procrustes', presence_min=PRESENCE_MIN,
                         smoothing='none'),
                hands=hands if hands is not None else
                      'every frame, this project\'s SAM left/right boxes, saved per-clip intrinsics, '
                      'no FAH detection, no temporal smoothing')


def hand_identity(project):
    """Everything that can change this frame-by-frame output: the pass code/pins,
    the working video, the exact box file and the clip intrinsics."""
    from .result_artifacts import file_stat
    project = Path(project)
    inputs = {}
    for name in ('input.mp4', BOXES):
        path = project / name
        inputs[name] = file_stat(path) if path.is_file() else None
    try:
        intrinsics = intrinsics_binding(project)
    except (OSError, ValueError, KeyError):
        intrinsics = None
    return dict(source=source_identity(), inputs=inputs, intrinsics=intrinsics)


def load_hands(path, n):
    """Numeric per-frame validity and local rotations; never a pickled object."""
    import numpy as np
    with np.load(Path(path), allow_pickle=False) as z:
        rotations = z['rotations'].copy()
        valid = z['valid'].copy()
    if rotations.shape != (n, 2, 15, 3, 3) or valid.shape != (n, 2) or valid.dtype != bool or \
            not np.isfinite(rotations).all():
        raise ValueError('Invalid FAH finger cache')
    if np.max(np.abs(rotations.swapaxes(-1, -2) @ rotations - np.eye(3))) > 2e-4 or \
            np.min(np.linalg.det(rotations)) < .999:
        raise ValueError('Invalid FAH finger rotations')
    return rotations, valid


def box_validity(boxes, width, height):
    """The same usable-box rule the HaMeR pass applies to the same SAM boxes."""
    import numpy as np
    size = boxes[:, :, 2:] - boxes[:, :, :2]
    clipped = np.maximum(0, np.minimum(boxes[:, :, 2:], [width, height]) - np.maximum(boxes[:, :, :2], 0))
    return (np.isfinite(boxes).all(-1) & (size.min(-1) > 2) & (clipped.min(-1) >= 4) &
            (clipped.prod(-1) / np.maximum(size.prod(-1), 1) >= .1))


def axis_angle_matrices(values):
    """15 flat-hand axis-angle triples -> local rotation matrices in MANO joint order.

    Written locally (Rodrigues) so the pinned fitter's output is checked with the same
    arithmetic the adapter is validated against, without adding a SciPy requirement.
    """
    import numpy as np
    values = np.asarray(values, np.float64).reshape(15, 3)
    if not np.isfinite(values).all():
        raise ValueError('FAH returned non-finite hand angles')
    angle = np.linalg.norm(values, axis=1)
    safe = np.where(angle > 1e-12, angle, 1.0)
    axis = values / safe[:, None]
    zero = np.zeros_like(axis[:, 0])
    skew = np.stack([-zero, -axis[:, 2], axis[:, 1], axis[:, 2], zero, -axis[:, 0],
                     -axis[:, 1], axis[:, 0], zero], -1).reshape(15, 3, 3)
    sine = np.sin(angle)[:, None, None]
    cosine = np.cos(angle)[:, None, None]
    return np.eye(3) + sine * skew + (1 - cosine) * (skew @ skew)


def mirror_left(rotations):
    """Undo FAH's mirrored right-hand frame: each local rotation is conjugated by the x reflection."""
    import numpy as np
    mirror = np.diag([-1.0, 1.0, 1.0])
    return mirror @ np.asarray(rotations, np.float64) @ mirror


def surface_targets(hand, regressor=None):
    """The 778-point MANO-topology surface FAH fitted or predicted, in camera metres."""
    import numpy as np
    surface = hand.get('vertices_mano') if isinstance(hand, dict) else None
    if surface is None and isinstance(hand, dict) and 'vertices' in hand and regressor is not None:
        weights, index = regressor
        surface = np.asarray(weights, np.float64) @ np.asarray(hand['vertices'], np.float64)[np.asarray(index, np.int64)]
    if surface is None or np.asarray(surface).shape != (778, 3):
        return None
    surface = np.asarray(surface, np.float64)
    return surface if np.isfinite(surface).all() else None


def joint_targets(hand):
    import numpy as np
    values = hand.get('keypoints_3d_mano', hand.get('keypoints_3d')) if isinstance(hand, dict) else None
    if values is None:
        return None
    values = np.asarray(values, np.float64)
    if values.shape != (21, 3) or not np.isfinite(values).all():
        return None
    return values


def fit_rotations(hand, is_right, fitter, procrustes, regressor=None, iters=FIT_ITERS):
    """One hand of the pinned FAH output to 15 SMPL-H-convention local rotations.

    FAH fits a right hand even for a left one: the pinned `MeshFitter` reports the
    flat-hand 45-D angles in that mirrored frame, so the mirror is undone here by
    conjugation. Returns None when this hand keeps the caller's own fingers
    (FlowHMR's, or the relaxed mean pose of a FAB body).
    """
    import numpy as np
    if not isinstance(hand, dict):
        return None
    presence = hand.get('presence')
    if presence is not None and (not np.isfinite(float(presence)) or float(presence) < PRESENCE_MIN):
        return None
    surface, joints = surface_targets(hand, regressor), joint_targets(hand)
    if surface is None or joints is None:
        return None
    flip = 1.0 if is_right else -1.0
    target_surface = surface.copy()
    target_surface[:, 0] *= flip
    target_joints = joints.copy()
    target_joints[:, 0] *= flip
    global_rotations, _ = procrustes.rotations(target_surface)
    start_root, start_pose = procrustes.local_pose(global_rotations)
    result = fitter.fit(target_surface[None], target_joints[None], np.asarray(start_root, np.float64)[None],
                        np.asarray(start_pose, np.float64)[None], np.zeros((1, 10)), iters=iters, params=True)
    try:
        mesh, (fitted, pose, betas) = result
    except (TypeError, ValueError):
        raise RuntimeError('The pinned FAH MeshFitter no longer exposes its fitted flat-hand parameters.')
    pose = np.asarray(pose, np.float64).reshape(-1)
    fitted = np.asarray(fitted, np.float64).reshape(-1, 3, 3)
    if pose.shape != (45,) or not np.isfinite(pose).all() or fitted.shape != (1, 3, 3) or \
            not np.isfinite(fitted).all() or not np.isfinite(np.asarray(mesh, np.float64)).all():
        return None
    if abs(float(np.linalg.det(fitted[0])) - 1.0) > 2e-3:
        return None
    local = axis_angle_matrices(pose.reshape(15, 3))
    if not is_right:
        local = mirror_left(local)
    if np.max(np.abs(local.swapaxes(-1, -2) @ local - np.eye(3))) > 2e-4 or np.min(np.linalg.det(local)) < .999:
        return None
    return local.astype(np.float32)


def refine(project, output, emit):
    """Parent side: publish the staged FAH prediction from the child's numeric cache."""
    import gc
    import numpy as np
    import torch
    from .flow_backend import save_tensor, validate_motion
    from .model_session import current
    from .result_artifacts import file_stat
    project = Path(project)
    n = output['rot6d'].shape[1]
    identity = hand_identity(project)
    receipt, payload = project / RECEIPT, project / HANDS
    cached = False
    try:
        value = json.loads(receipt.read_text(encoding='utf-8'))
        cached = value['identity'] == identity and value['output'] == file_stat(payload)
        if cached:
            load_hands(project / HANDS, n)
    except (OSError, ValueError, KeyError, TypeError, EOFError, zipfile.BadZipFile):
        cached = False
    if not cached:
        # Do not hold FlowHMR weights while the independent FAH environment runs.
        session = current()
        if session is not None:
            session.discard()
        gc.collect()
        torch.cuda.empty_cache()
        emit('info', message='FlowHMR FAH: every-frame fingers from this project\'s SAM hand boxes; '
                             'body, wrists, trajectory and contacts are retained.')
        run_child([str(project)], emit, 'FlowHMR FAH')
        if hand_identity(project) != identity:
            raise RuntimeError('FAH inputs changed during inference. Reconstruct 3D again.')
    rotations, valid = load_hands(project / HANDS, n)
    result = dict(output)
    result['rot6d'] = output['rot6d'].clone()
    for side, offset in enumerate(ROT6D_SLOTS):
        pose = torch.from_numpy(rotations[:, side, :, :, :2].reshape(n, 15, 6).astype(np.float32))
        result['rot6d'][0, valid[:, side], offset:offset + 15] = pose[valid[:, side]]
    result['body_rot6d'] = result['rot6d'][:, :, 1:]
    result.pop('local_joints_positions', None)
    validate_motion(result, n)
    save_tensor(project / PREDICTION, result)
    emit('info', message=f'FlowHMR FAH: {int(valid[:, 0].sum())}/{n} left and {int(valid[:, 1].sum())}/{n} right hand '
                         'frames; unusable crops retain FlowHMR fingers.' + (' Verified hand cache reused.' if cached else ''))
    return result


def run_child(arguments, emit, label):
    """Parent side: one isolated `.venv-fah` child. Its events are relayed and its
    last output lines are kept for the error message."""
    python = fah_runtime.interpreter(ROOT)
    if not python.is_file():
        raise RuntimeError('FAH needs its own pinned environment at ' + fah_runtime.VENV + '. ' + fah_runtime.HINT)
    command = [str(python), '-I', '-X', 'utf8', str(ROOT / 'mocap/flow_fah_entry.py'), *arguments]
    from collections import deque
    tail = deque(maxlen=20)
    with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding='utf-8', errors='replace',
                          creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)) as child:
        try:
            for line in child.stdout:
                tail.append(line[-1000:])
                if line.startswith(EVENT):
                    event = json.loads(line[len(EVENT):])
                    kind = event.pop('type')
                    emit(kind, **event)
            if child.wait():
                raise RuntimeError(f'{label} failed:\n' + ''.join(tail)[-4000:])
        except BaseException:
            child.kill()
            child.wait()
            raise


def write_hands(target, rotations, valid, n):
    """Child side: the numeric answer, revalidated before the parent can see it."""
    import numpy as np
    temporary = target.with_name(target.name + '.' + uuid.uuid4().hex + '.partial')
    try:
        with temporary.open('wb') as stream:
            np.savez(stream, rotations=rotations, valid=valid)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return load_hands(target, n)


def infer(video, boxes, intrinsics, width, height, emit):
    """Child side, only inside the pinned `.venv-fah`: every-frame MANO fits for the
    left/right boxes of each frame. Returns the 15 local rotations per hand
    (frames,2,15,3,3), which of them were fitted (frames,2) and the per-side counts."""
    import cv2, gc, inspect, os, time
    import numpy as np, torch
    from .compute_policy import configure_worker_cpu
    from .model_session import set_memory_budget
    started = time.monotonic()
    n = boxes.shape[0]
    torch.set_num_threads(configure_worker_cpu('flowhmr'))
    if not torch.cuda.is_available():
        raise RuntimeError('FAH requires a compatible NVIDIA GPU')
    free, total = torch.cuda.mem_get_info()
    set_memory_budget(free, total)
    os.environ.update(fah_runtime.configure_environment(ROOT))
    name = fah_runtime.model()
    emit('progress', current=0, total=n, phase='fah', message='Load the FAH hand model')
    from fah import load
    from fah.fast_mano import MeshFitter
    from fah.mano_procrustes import ManoProcrustes
    method = load(name, mode='image', weights=os.environ['FAH_WEIGHTS'], sam3d=os.environ['FAH_SAM3D'],
                  mano_dir=os.environ['MANO_DIR'], device='cuda')
    estimator = method.method.est
    for required in ('boxes', 'is_right', 'scores', 'cam_int'):
        if required not in inspect.signature(estimator.process_image).parameters:
            raise RuntimeError(f'The pinned FAH estimator no longer accepts an explicit {required} box input.')
    if 'params' not in inspect.signature(MeshFitter.fit).parameters:
        raise RuntimeError('The pinned FAH fitter no longer exposes its flat-hand parameters.')
    fitter = MeshFitter(os.environ['MANO_DIR'], device='cuda', iters=FIT_ITERS, solver='gn',
                        wv=FIT_WEIGHT_SURFACE, centre=FIT_CENTRE, fd_eps=FIT_FD_EPS, jac='fd')
    procrustes = ManoProcrustes(os.environ['MANO_DIR'])
    regressor = getattr(method.method, 'vreg', None)
    valid = box_validity(boxes, width, height)
    rotations = np.tile(np.eye(3, dtype=np.float32), (n, 2, 15, 1, 1))
    used = [0, 0]
    emit('info', message=f'FAH fingers: {name} backbone; every frame, both supplied hand boxes per frame; '
                         f'{FIT_ITERS} Gauss-Newton step(s) from the closed-form start; no temporal smoothing.')
    try:
        cap = cv2.VideoCapture(str(video))
        try:
            with torch.inference_mode():
                for index in range(n):
                    ok, frame = cap.read()
                    if not ok:
                        raise ValueError(f'Cannot decode FAH frame {index + 1}')
                    sides = [side for side in (0, 1) if valid[index, side]]
                    if sides:
                        hands = estimator.process_image(frame, boxes=boxes[index, sides],
                                                        is_right=np.array([side == 1 for side in sides]),
                                                        scores=np.ones(len(sides), np.float32), cam_int=intrinsics[index])
                        if len(hands) != len(sides):
                            raise RuntimeError('FAH must return one hand for each supplied box')
                        for side, hand in zip(sides, hands):
                            local = fit_rotations(hand, side == 1, fitter, procrustes, regressor)
                            if local is None:
                                valid[index, side] = False
                            else:
                                rotations[index, side] = local
                                used[side] += 1
                    if index // 10 != (index + 1) // 10 or index + 1 == n:
                        emit('progress', current=index + 1, total=n, phase='fah', message='FAH: infer every-frame fingers')
                if cap.read()[0]:
                    raise ValueError('FAH video frame count differs from the project')
        finally:
            cap.release()
    finally:
        del fitter, procrustes, method, estimator
        gc.collect()
        torch.cuda.empty_cache()
    emit('progress', current=n, total=n, phase='fah', message='FAH finger pass complete')
    emit('info', message=f'FAH finger pass: {used[0]}/{n} left and {used[1]}/{n} right fitted frames; '
                         f'{time.monotonic() - started:.2f} s; peak allocated VRAM '
                         f'{torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB; peak worker RAM '
                         f'{peak_rss_bytes() / 2 ** 30:.2f} GiB.')
    return rotations, valid, used


def run(project, emit):
    """Child side for FlowHMR: called only by the separate pinned `.venv-fah` runtime."""
    import numpy as np
    project = Path(project).resolve()
    state = json.loads((project / 'project.json').read_text(encoding='utf-8'))
    n = state['frames']
    if state.get('pipeline', {}).get('model') != 'flowhmr' or not enabled(project) or not 1 <= n <= MAX_FRAMES:
        raise ValueError('Invalid FlowHMR FAH project')
    if not fah_runtime.supported():
        raise ValueError('FAH finger refinement runs only on the Linux route.')
    identity = hand_identity(project)
    boxes = np.load(project / BOXES, allow_pickle=False)
    if boxes.shape != (n, 2, 4):
        raise ValueError('SAM hand boxes do not match the motion')
    with np.load(project / CAMERA, allow_pickle=False) as data:
        intrinsics = np.asarray(data['K'], np.float32).reshape(n, 3, 3)
    rotations, valid, _ = infer(project / 'input.mp4', boxes, intrinsics, state['width'], state['height'], emit)
    if hand_identity(project) != identity:
        raise RuntimeError('FAH inputs changed before commit')
    from .storage import atomic_json
    from .result_artifacts import file_stat
    target = project / HANDS
    write_hands(target, rotations, valid, n)
    atomic_json(project / RECEIPT, dict(identity=identity, output=file_stat(target)))


def read_request(path):
    """Child side: one caller-written numeric request (boxes, intrinsics, frame size)."""
    import numpy as np
    with np.load(path, allow_pickle=False) as data:
        absent = sorted({'boxes', 'K', 'width', 'height'} - set(data.files))
        if absent:
            raise ValueError('The FAH request is missing: ' + ', '.join(absent))
        boxes = np.array(data['boxes'], np.float32)
        intrinsic = np.asarray(data['K'], np.float64)
        width = int(np.asarray(data['width']).reshape(()))
        height = int(np.asarray(data['height']).reshape(()))
    n = boxes.shape[0] if boxes.ndim == 3 else 0
    if not 1 <= n <= MAX_FRAMES or boxes.shape != (n, 2, 4) or not np.isfinite(boxes).all():
        raise ValueError(f'The FAH request needs finite hand boxes with shape (frames,2,4) and 1..{MAX_FRAMES} frames')
    if intrinsic.shape not in ((3, 3), (1, 3, 3), (n, 3, 3)) or not np.isfinite(intrinsic).all():
        raise ValueError('The FAH request needs one (3,3) or one per-frame (frames,3,3) camera matrix')
    focal = intrinsic.reshape(-1, 3, 3)
    if (focal[:, [0, 1], [0, 1]] <= 0).any() or width < 16 or height < 16:
        raise ValueError('The FAH request needs positive focal lengths and the video frame size')
    intrinsics = np.broadcast_to(np.asarray(intrinsic, np.float32).reshape(-1, 3, 3), (n, 3, 3))
    return boxes, np.ascontiguousarray(intrinsics), width, height


def run_request(request, video, output, receipt, emit):
    """Child side for FAB: the caller owns the request, the work folder and all receipts."""
    from .storage import atomic_json
    from .result_artifacts import file_stat
    if not fah_runtime.supported():
        raise ValueError('FAH finger refinement runs only on the Linux route.')
    request, video, output, receipt = (Path(path) for path in (request, video, output, receipt))
    boxes, intrinsics, width, height = read_request(request)
    source = source_identity(hands=REQUEST_HANDS)
    if output.exists():
        raise ValueError('The FAH output already exists; select a new file.')
    rotations, valid, used = infer(video, boxes, intrinsics, width, height, emit)
    write_hands(output, rotations, valid, boxes.shape[0])
    atomic_json(receipt, dict(identity=dict(version=1, source=source, request=file_stat(request),
                                            video=file_stat(video)),
                              output=file_stat(output), used=used))


def refine_request(work, video, boxes, intrinsics, width, height, emit):
    """Parent side for FAB: stage one numeric request in the caller's work folder, run
    the isolated child and return its checked answer. Nothing is cached here; the
    caller owns receipts and removes `files` after its own commit."""
    import numpy as np
    from .result_artifacts import file_stat
    work = Path(work)
    token = uuid.uuid4().hex[:12]
    request, target, receipt = (work / f'fah-{token}-{part}' for part in ('request.npz', 'hands.npz', 'receipt.json'))
    n = int(boxes.shape[0])
    intrinsic = np.asarray(intrinsics, np.float64)
    K = intrinsic if intrinsic.shape == (3, 3) else intrinsic.reshape(-1, 3, 3)   # one, or one per frame
    with request.open('wb') as stream:
        np.savez(stream, boxes=np.asarray(boxes, np.float32),
                 K=K,
                 width=np.array(int(width), np.int32), height=np.array(int(height), np.int32))
    source = source_identity(hands=REQUEST_HANDS)
    run_child(['--request', str(request), '--video', str(video), '--output', str(target), '--receipt', str(receipt)],
              emit, 'FAB FAH')
    try:
        value = json.loads(receipt.read_text(encoding='utf-8'))
    except (OSError, ValueError) as error:
        raise RuntimeError('The FAH finger pass wrote no receipt.') from error
    if (value.get('identity') or {}).get('source') != source or value.get('output') != file_stat(target):
        raise RuntimeError('FAH code, runtime or files changed during the finger pass; run it again.')
    rotations, valid = load_hands(target, n)
    return dict(rotations=rotations, valid=valid, used=[int(valid[:, side].sum()) for side in (0, 1)], source=source,
                files=(request, target, receipt))
