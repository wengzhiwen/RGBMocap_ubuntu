"""Isolated FAB inference entry: one fixed-camera project actor, 30 fps.

Launch (the caller owns environments, options, project files and receipts):

    .venv-fab/bin/python -I -X utf8 mocap/fab_infer_entry.py \
        --project <app model project> --output <new numeric .npz>

One resident PyTorch CUDA worker loads the pinned upstream FAB predictor once
and runs the project's canonical working video one frame at a time with the
tracking stage's boxes and intrinsics. It writes one atomic numeric NPZ of SMPL
parameters, placement, contacts and calibration and nothing else: no project
file, receipt, render, upstream pickle or mesh cache. ``mocap.fab_data`` reads
that record back in the shared CPU runtime.

FAB's released SMPL fit is used, not MHR: ``smpl_pose``/``smpl_betas`` are the
native SMPL axis angles and shape coefficients in the camera frame. The fitted
mesh is re-derived from the pinned predictor's own SMPL forward with no
translation, so the SMPL camera translation and the shape-dependent camera
pelvis come from the same equations FAB used. ``cam_t`` is deliberately not
used for that: it places the MHR mesh, not the fitted SMPL body.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mocap.compute_policy import configure_worker_cpu  # noqa: E402
from mocap.model_session import ModelSession, acquire, model_key, set_memory_budget  # noqa: E402
from mocap.process_memory import peak_rss_bytes  # noqa: E402
from mocap.result_artifacts import file_stat  # noqa: E402

STAGE = 'fab_infer'
MODELS = ('fab-s', 'fab-l')           # must stay inside mocap.fab_data.MODELS
CONSISTENCY_LIMIT_M = 0.005          # mesh vs the pinned SMPL forward; mm on the float32 path
DEPTH_LIMIT_M = 0.05                 # smallest accepted pelvis camera depth
PROGRESS_EVERY = 15


def emit(kind, **values):
    print('@@MOCAP@@' + json.dumps(dict(type=kind, **values)), flush=True)


def asset_paths(model):
    """Official FAB asset locations, resolved through the pinned upstream paths.

    FAB_HOME / FAB_WEIGHTS / FAB_SAM3D / FAB_SMPL_DIR / FAB_G1 select them. The
    G1 URDF is required because the FAB-L constructor loads the G1 head's joint
    limits, even though this scope never reads a G1 result.
    """
    from fab import paths
    from fab.retarget.g1_files import urdf
    weights = paths.model_file(model)
    fitter = paths.shared('smpl_fitter.pt')
    smpl = paths.smpl_dir()
    required = ((weights, 'FAB weights ($FAB_WEIGHTS or $FAB_HOME/weights)'),
                (paths.sam3d_dir() / 'model_config.yaml', 'the SAM 3D Body config ($FAB_SAM3D)'),
                (smpl / 'SMPL_NEUTRAL.pkl', 'the SMPL neutral body model ($FAB_SMPL_DIR)'),
                (smpl / 'J_regressor_h36m.npy', 'the SMPL H36M joint regressor ($FAB_SMPL_DIR)'),
                (fitter, 'the official FAB SMPL fitter ($FAB_WEIGHTS/shared)'),
                (urdf(), 'the Unitree G1 URDF the FAB-L constructor requires ($FAB_G1)'))
    absent = [f'{what} at {path}' for path, what in required if not Path(path).is_file()]
    if absent:
        raise ValueError('Prepare the official FAB files first: ' + '; '.join(absent))
    return dict(weights=str(weights), fitter=str(fitter), sam3d=str(paths.sam3d_dir()),
                smpl=str(smpl), g1=str(urdf()))


def read_inputs(project, np):
    """Validate the app model project manifest and the track stage output."""
    project = Path(project)
    state = json.loads((project / 'project.json').read_text(encoding='utf-8'))
    frames, fps = int(state['frames']), float(state['fps'])
    if not (project / 'input.mp4').is_file():
        raise ValueError('The project has no canonical 30 fps working video (input.mp4).')
    if frames < 1 or not abs(fps - 30.0) <= 1e-3:
        raise ValueError('FAB fixed-camera processing uses a 30 fps project timeline.')
    with np.load(project / 'tracking.npz', allow_pickle=False) as data:
        absent = sorted({'boxes', 'observed', 'K'} - set(data.files))
        if absent:
            raise ValueError('Project tracking is missing: ' + ', '.join(absent))
        boxes = np.array(data['boxes'], np.float32)
        observed = np.array(data['observed'])
        intrinsic = np.array(data['K'], np.float64)
    if boxes.shape != (frames, 4) or observed.shape != (frames,) or intrinsic.shape != (3, 3):
        raise ValueError('Project tracking does not match the project timeline.')
    if observed.dtype.kind != 'b':
        raise ValueError('Project tracking observations are invalid.')
    if not observed.any():
        raise ValueError('Fixed-camera FAB needs at least one tracked person box.')
    if not np.isfinite(boxes).all():
        raise ValueError('Project tracking boxes are not finite.')
    if (boxes[:, 2] - boxes[:, 0] < 2 or boxes[:, 3] - boxes[:, 1] < 2).any():
        raise ValueError('Project tracking boxes are too small.')
    if not np.isfinite(intrinsic).all() or intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0:
        raise ValueError('Project camera intrinsics are invalid.')
    return state, boxes, intrinsic, observed


def readout(predictor, person, np, torch):
    """One frame's SMPL fit, its recovered camera placement and its mesh sole height."""
    pose = np.asarray(person.get('smpl_pose'), np.float32)
    betas = np.asarray(person.get('smpl_betas'), np.float32)
    mesh = np.asarray(person.get('mesh_smpl_fit'), np.float32)
    contact = np.asarray(person.get('contact'), np.float32)
    if pose.shape != (72,) or betas.shape != (10,) or mesh.shape != (6890, 3) or contact.shape != (16,):
        raise ValueError('FAB did not return the released SMPL fit, its fitted mesh or its contact channels.')
    if not all(np.isfinite(value).all() for value in (pose, betas, mesh, contact)):
        raise ValueError('FAB produced non-finite SMPL parameters, mesh or contacts.')
    if contact.min() < 0 or contact.max() > 1:
        raise ValueError('FAB contact channels are not probabilities in [0,1].')
    # The same pinned body with the same returned pose and shape and no
    # translation: the difference to the fitted mesh is the SMPL translation.
    device = predictor.body.v_template.device

    def tensor(value):
        return torch.as_tensor(np.ascontiguousarray(value), device=device)[None]

    with torch.inference_mode():
        vertices, joints = predictor.body(tensor(pose[:3]), tensor(pose[3:]), tensor(betas))
        forward = vertices.double().cpu().numpy()
        pelvis_rest = joints[0, 0].double().cpu().numpy()
    del vertices, joints
    difference = mesh.astype(np.float64) - forward
    del forward
    transl = difference.mean(0)
    spread = float(np.abs(difference - transl).max())
    del difference
    if spread > CONSISTENCY_LIMIT_M:
        raise ValueError(f'The fitted mesh and the pinned SMPL forward disagree by {spread * 1000:.1f} mm; '
                         'the FAB body model does not match this predictor. Reinstall the pinned FAB assets.')
    pelvis = pelvis_rest + transl
    depth = float(pelvis[2])
    if not np.isfinite(pelvis).all() or depth <= DEPTH_LIMIT_M or mesh[:, 2].min() <= 0:
        raise ValueError(f'FAB produced an invalid camera depth ({depth:.3f} m at the pelvis).')
    # Provisional Y-up for an approximately level fixed camera: up is -camera Y.
    return pose, betas, transl, pelvis, float(-mesh[:, 1].max())


def median_rest_joints(body, betas, np, torch, parents, count):
    """The clip's median-shape SMPL rest joints in the neutral body axes."""
    zeros = torch.zeros(1, 3, device=body.v_template.device), torch.zeros(1, 69, device=body.v_template.device)
    with torch.inference_mode():
        _, joints = body(zeros[0], zeros[1], torch.as_tensor(betas, dtype=torch.float32,
                                                            device=body.v_template.device)[None])
        rest = joints[0, :count].double().cpu().numpy()
    bones = np.linalg.norm(rest[1:] - rest[np.asarray(parents[1:])], axis=-1)
    if not np.isfinite(rest).all() or (bones < 1e-4).any() or rest[:, 1].max() - rest[:, 1].min() < 0.2:
        raise ValueError('The pinned SMPL body did not produce a usable rest skeleton.')
    return rest.astype(np.float32)


def reconstruct(project, output, model):
    threads = configure_worker_cpu(STAGE)
    import numpy as np
    from mocap.fab_data import FLOOR_QUANTILE, JOINT_COUNT, MODELS as RECORD_MODELS, PARENTS, world_transform
    if model not in RECORD_MODELS:
        raise ValueError('Unsupported FAB model: ' + str(model))
    project = Path(project).resolve()
    output = Path(output)
    state, boxes, intrinsic, observed = read_inputs(project, np)
    frames = int(state['frames'])
    if output.suffix != '.npz':
        raise ValueError('The numeric FAB output must be a .npz file.')
    if output.exists():
        raise ValueError('The numeric FAB output already exists; select a new file.')
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < 32 * 1024 ** 2:
        raise OSError('Free more disk space for the FAB numeric result.')
    sources = {str(project / name): file_stat(project / name)
               for name in ('project.json', 'input.mp4', 'tracking.npz')}
    import torch
    import cv2
    if not torch.cuda.is_available():
        raise RuntimeError('FAB needs a working CUDA GPU; check the pinned FAB environment.')
    from mocap.memory_mode import announce, effective_mode
    torch.set_num_threads(threads)
    assets = asset_paths(model)
    free, total = torch.cuda.mem_get_info()
    announce('FAB inference', effective_mode(),
             f'One resident {model} predictor, one box per frame, {frames} frames at 30 fps.')
    rows = {name: [] for name in ('smpl_pose', 'smpl_betas', 'smpl_transl', 'camera_pelvis', 'contact')}
    sole = []
    session = ModelSession()
    with session.activate():
        set_memory_budget(free, total)

        def build():
            from fab.predictor import FABPredictor
            # detector="" keeps the project's own boxes, parts=False skips the
            # part branch, fast=False keeps the plain PyTorch path, body_engine=""
            # refuses a TensorRT engine, and the fitter is the fixed official path.
            return FABPredictor(assets['weights'], detector='', parts=False, fast=False, body_engine='',
                                smpl_fitter=assets['fitter'], device='cuda')

        predictor = acquire(model_key(model, *(assets[key] for key in ('weights', 'fitter', 'sam3d', 'smpl', 'g1'))),
                            build)
        if getattr(predictor, 'body', None) is None:
            raise ValueError('The FAB predictor has no SMPL body, so the released SMPL fit is unavailable.')
        capture = None
        try:
            capture = cv2.VideoCapture(str(project / 'input.mp4'))
            if not capture.isOpened():
                raise RuntimeError('Cannot open the project working video.')
            previous = None
            for index in range(frames):
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(f'Cannot read working video frame {index}.')
                image = np.ascontiguousarray(frame[..., ::-1])  # the decoder gives BGR; FAB reads RGB
                # prev carries the same person's previous output only across a
                # consecutive timeline; a gap must not seed the temporal prior.
                people = predictor.predict(
                    image,
                    K=intrinsic,
                    boxes=boxes[index:index + 1],
                    prev=[previous] if (index > 0 and previous is not None and observed[index] and observed[index - 1]) else None,
                )
                if len(people) != 1:
                    raise ValueError(f'FAB returned {len(people)} people for one tracked box at frame {index}.')
                pose, betas, transl, pelvis, lowest = readout(predictor, people[0], np, torch)
                rows['smpl_pose'].append(pose)
                rows['smpl_betas'].append(betas)
                rows['smpl_transl'].append(transl.astype(np.float32))
                rows['camera_pelvis'].append(pelvis.astype(np.float32))
                rows['contact'].append(np.asarray(people[0]['contact'], np.float32))
                sole.append(lowest)
                previous = people[0]
                if index % PROGRESS_EVERY == 0 or index + 1 == frames:
                    emit('progress', stage=STAGE, current=index + 1, total=frames,
                         message=f'{model}: frame {index + 1} of {frames}')
            median = np.median(np.stack(rows['smpl_betas']), axis=0).astype(np.float32)
            rest = median_rest_joints(predictor.body, median, np, torch, PARENTS, JOINT_COUNT)
        finally:
            if capture is not None:
                capture.release()
            previous = None
    # The estimated floor plane of the provisional Y-up frame. A low quantile of
    # the per-frame lowest mesh point, not the minimum: a few frames with a
    # mesh below the ground or a raised leg must not set the level. This is an
    # estimate for an approximately level fixed camera, not measured ground.
    floor = float(np.quantile(np.asarray(sole, np.float64), FLOOR_QUANTILE))
    arrays = dict(rows, boxes=boxes, K=intrinsic, median_betas=median, rest_joints=rest,
                  world_from_camera=world_transform(floor), floor_estimate=np.array(floor, np.float64),
                  frames=np.array(frames, np.int32), fps=np.array(float(state['fps']), np.float64))
    return arrays, output, frames, floor, sources


def write_record(arrays, output, frames, np):
    """Atomic numeric-only NPZ, revalidated before it becomes visible."""
    from mocap.fab_data import load, validate
    temporary = output.with_name(output.name + '.partial.npz')
    try:
        # Every stored array is numeric, so savez never pickles; load refuses pickles.
        np.savez(temporary, **arrays)
        validate(load(temporary, frames))
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    os.replace(temporary, output)


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='backslashreplace')
    parser = argparse.ArgumentParser(prog='fab_infer_entry', description='FAB fixed-camera numeric inference')
    parser.add_argument('--project', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--model', default='fab-l', choices=MODELS)
    arguments = parser.parse_args(sys.argv[1:] if argv is None else argv)
    started = time.monotonic()
    try:
        from mocap.fab_runtime import validate_interpreter
        validate_interpreter()
        import numpy as np
        import torch
        arrays, output, frames, floor, sources = reconstruct(arguments.project, arguments.output, arguments.model)
        # The record must describe exactly the media and track it was computed from.
        if any(file_stat(Path(name)) != stamp for name, stamp in sources.items()):
            raise ValueError('Project media or tracking changed during FAB inference; '
                             'the numeric result was not written.')
        write_record(arrays, output, frames, np)
        emit('fab_numeric', output=str(output), model=arguments.model, frames=frames,
             fps=float(arrays['fps'].reshape(())), floor_estimate_m=floor, camera='fixed',
             median_betas=arrays['median_betas'].tolist())
        emit('complete', stage=STAGE, output=str(output), elapsed=time.monotonic() - started,
             peak_vram_allocated_bytes=torch.cuda.max_memory_allocated(),
             peak_vram_reserved_bytes=torch.cuda.max_memory_reserved(),
             peak_ram_bytes=peak_rss_bytes())
        return 0
    except KeyboardInterrupt:
        emit('failed', stage=STAGE, message='FAB inference was cancelled.')
        return 130
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        emit('failed', stage=STAGE, message=str(error))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
