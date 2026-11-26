"""Committed FAB numeric results and standard SMPL-X result parameters.

This module is CPU only: it never imports FAB, Torch or CUDA, so the shared
runtime's loader, correction and export paths can read a FAB numeric record.
``mocap/fab_infer_entry.py`` writes that record; phase 2b publishes project
files around it.

Stored arrays are one row per working frame in project timeline order. The
camera frame is FAB's / OpenCV's (X right, Y down, Z forward, metres, camera at
the origin). The app's world frame is Y-up, so one fixed camera maps to world
with the proper rotation ``diag(1,-1,-1)`` plus one clip-wide vertical shift
that puts the mesh-estimated floor at ``Y = 0``. Body-only reconstruction from a
single fixed camera stays an approximation: no scene geometry, ground plane or
ground truth is measured, and the estimated floor can be wrong for sitting,
lying or unusual camera heights.

The pinned FAB fit uses the native SMPL model: ``smpl_pose`` is 24 axis-angle
joints (global orient first), ``smpl_betas`` is 10 SMPL shape coefficients and
``smpl_transl``/``camera_pelvis`` are that model's camera-frame placement. The
published SMPL-X parameters keep the first 22 native local rotations, drop the
two SMPL terminal-hand rotations from the derived body (SMPL has no fingers) and
fit one constant SMPL-X neutral shape from the clip's median-shape rest joints
instead of copying SMPL coefficients into the SMPL-X basis.

The optional FAH finger pass (`fab --fah`) is the only source of finger motion.
It crops its hands from FAB's own predicted 2D hand points (``hand_keypoints``,
stored for that purpose alone) and returns 15 local MANO rotations per hand; the
converted MANO finger slots are in the SMPL-X hand order, so they are published
as SMPL-X hand poses (offsets from the body's own per-hand mean). A hand FAH
could not fit keeps the relaxed mean pose, exactly like a plain FAB result.
"""
from __future__ import annotations
from pathlib import Path
import math

import numpy as np

MODELS = ('fab-s', 'fab-l')
FPS = 30.0
VERSION = 1

# FAB fits the native SMPL mesh (6890 vertices); the published body is SMPL-X
# (10475 vertices). The two counts are never interchangeable.
VERTEX_COUNT = 6890               # native SMPL, i.e. the source fit
SMPLX_VERTEX_COUNT = 10475        # SMPL-X, i.e. the published target body
SMPL_POSE_WIDTH = 72          # 24 SMPL joints; the last two are terminal hands
SMPL_BODY_POSE_WIDTH = 63     # the first 21 SMPL joints, i.e. SMPL-X body_pose
SMPL_BETAS = 10
HAND_POSE_WIDTH = 45
JOINT_COUNT = 22
CONTACT_CHANNELS = 16         # 8 static + 8 "any"; the static four come first
CONTACT_STATIC = ('l_heel', 'l_toe', 'r_heel', 'r_toe')
CORRECTION_CHANNELS = 6       # the existing correction input width
DEPTH_LIMIT_M = 0.05          # smallest accepted camera depth of the pelvis
FLOOR_QUANTILE = 0.05         # robust per-clip floor level of the mesh soles
CONTACT_EPS = 1e-3            # keeps a probability logit finite
SHAPE_RIDGE = 1e-3            # Tikhonov term of the neutral-shape fit
SHAPE_LIMIT = 10.0            # accepted range of one shape coefficient
SHAPE_RESIDUAL_LIMIT_M = 0.08
MAX_FRAMES = 10000
RECORD_LIMIT_BYTES = 32 * 1024 ** 2
HAND_JOINTS = 15              # per hand, in SMPL-X and converted MANO finger order
HAND_KEYPOINTS = 21           # FAB's own 2D points per hand
# Square FAH hand boxes from those points: centred on their bounding box, with
# twice the longer extent (the SAM boxes of the FlowHMR route measure about 1.9x
# it), never below the floor and never larger than a share of the body box.
HAND_BOX_POLICY = 'fab-keypoints-square-boxes-v1'
HAND_BOX_SCALE = 2.0
HAND_BOX_MIN_SIDE = 32.0
HAND_BOX_MAX_SHARE = 0.8
HAND_MIN_EXTENT = 16.0        # a hand smaller than this has no usable pixels
# What the published hand poses contain; saved with the result.
HAND_POSE_POLICY = 'registered-neutral-untracked-v1'
FAH_HAND_POSE_POLICY = 'fah-fingers-on-fab-body-v1'

# Parents of the first 22 SMPL / SMPL-X joints in the native order of the
# numeric record and of the app's (22,3) joint caches. The left and right arm
# joints alternate: 16 left shoulder, 17 right shoulder, 18 left elbow,
# 19 right elbow, 20 left wrist, 21 right wrist, as in both body models' own
# trees and mocap.mixamo_rig.BODY_PARENTS.
PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19)

# name: (shape with -1 = frame count, accepted dtype kind)
ARRAYS = {
    'smpl_pose': ((-1, SMPL_POSE_WIDTH), 'f'),
    'smpl_betas': ((-1, SMPL_BETAS), 'f'),
    'smpl_transl': ((-1, 3), 'f'),
    'camera_pelvis': ((-1, 3), 'f'),
    'contact': ((-1, CONTACT_CHANNELS), 'f'),
    'boxes': ((-1, 4), 'f'),
    'rest_joints': ((JOINT_COUNT, 3), 'f'),
    'median_betas': ((SMPL_BETAS,), 'f'),
    'K': ((3, 3), 'f'),
    'world_from_camera': ((4, 4), 'f'),
    'floor_estimate': ((), 'f'),
    'frames': ((), 'iu'),
    'fps': ((), 'f'),
}
# Written by every new capture, but absent in records from before the FAH pass;
# only that pass reads them, so their absence never invalidates a body result.
OPTIONAL_ARRAYS = {
    'hand_keypoints': ((-1, 2, HAND_KEYPOINTS, 2), 'f'),
}
SPECS = {**ARRAYS, **OPTIONAL_ARRAYS}


def camera_basis():
    """The proper camera-to-world rotation of an approximately level fixed camera."""
    return np.diag(np.array([1., -1., -1.]))


def world_transform(floor_estimate):
    """Static 4x4 camera-to-world transform with the floor plane at world Y = 0."""
    floor = float(floor_estimate)
    if not math.isfinite(floor):
        raise ValueError('FAB floor estimate is not finite')
    out = np.eye(4)
    out[:3, :3] = camera_basis()
    out[:3, 3] = (0., -floor, 0.)
    return out


def _finite(name, values):
    values = np.asarray(values)
    if not np.isfinite(values).all():
        raise ValueError(f'FAB numeric record has a non-finite {name}')
    return values


def _array(record, name, frames):
    spec, kind = SPECS[name]
    values = np.asarray(record[name])
    if values.dtype.kind not in kind:
        raise ValueError(f'FAB numeric record stores {name} as {values.dtype}')
    shape = tuple(spec)
    if -1 in shape:
        shape = (frames,) + shape[1:]
    if values.shape != shape:
        raise ValueError(f'FAB numeric record {name} has shape {values.shape}, expected {shape}')
    return _finite(name, values)


def validate(record, frames=None):
    """Validate one numeric record; every failure rejects the whole clip."""
    if not isinstance(record, dict):
        raise ValueError('FAB numeric record is not a set of arrays')
    unknown = sorted(set(record) - set(SPECS))
    if unknown:
        raise ValueError('FAB numeric record has unexpected arrays: ' + ', '.join(unknown))
    missing = sorted(set(ARRAYS) - set(record))
    if missing:
        raise ValueError('FAB numeric record is missing arrays: ' + ', '.join(missing))
    count = int(_array(record, 'frames', 0).reshape(()))
    fps = float(_array(record, 'fps', 0).reshape(()))
    if frames is not None and int(frames) != count:
        raise ValueError(f'FAB numeric record covers {count} frames, the project has {int(frames)}')
    if not 1 <= count <= MAX_FRAMES:
        raise ValueError(f'FAB numeric record frame count {count} is outside 1..{MAX_FRAMES}')
    if not math.isfinite(fps) or abs(fps - FPS) > 1e-3:
        raise ValueError(f'FAB numeric record frame rate {fps} is not {FPS:g} fps')
    out = {'frames': count, 'fps': fps}
    for name in ARRAYS:
        out[name] = _array(record, name, count)
    for name in OPTIONAL_ARRAYS:
        out[name] = None if record.get(name) is None else _array(record, name, count)
    pose, betas = out['smpl_pose'], out['smpl_betas']
    if np.abs(pose).max() > 4 * math.pi:
        raise ValueError('FAB SMPL pose angles are not plausible')
    if np.abs(betas).max() > SHAPE_LIMIT or np.abs(out['median_betas']).max() > SHAPE_LIMIT:
        raise ValueError('FAB SMPL shape coefficients are not plausible')
    contact = out['contact']
    if contact.min() < -CONTACT_EPS or contact.max() > 1 + CONTACT_EPS:
        raise ValueError('FAB contact probabilities must be probabilities in [0,1]')
    boxes = out['boxes']
    if (boxes[:, 2] - boxes[:, 0] <= 1).any() or (boxes[:, 3] - boxes[:, 1] <= 1).any() \
            or (boxes[:, :2] < -1).any() or (boxes[:, 2:] < 0).any():
        raise ValueError('FAB tracking boxes are invalid')
    pelvis, transl = out['camera_pelvis'], out['smpl_transl']
    if np.any(pelvis[:, 2] <= DEPTH_LIMIT_M) or np.any(transl[:, 2] <= DEPTH_LIMIT_M):
        raise ValueError('FAB camera depth is missing or not positive')
    intrinsic = out['K']
    if (intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0
            or intrinsic[0, 2] < 0 or intrinsic[1, 2] < 0
            or abs(intrinsic[2, 0]) > 1e-6 or abs(intrinsic[2, 1]) > 1e-6
            or not math.isclose(intrinsic[2, 2], 1., abs_tol=1e-6)
            or abs(intrinsic[0, 1]) > 1e-6 or abs(intrinsic[1, 0]) > 1e-6):
        raise ValueError('FAB camera intrinsics need positive focal lengths and a (0,0,1) bottom row')
    rest = out['rest_joints']
    bones = rest[1:] - rest[np.asarray(PARENTS[1:])]
    length = np.linalg.norm(bones, axis=-1)
    if (length < 1e-4).any() or rest[:, 1].max() - rest[:, 1].min() < 0.2:
        raise ValueError('FAB rest skeleton is invalid')
    floor = float(out['floor_estimate'].reshape(()))
    transform = out['world_from_camera']
    if (abs(float(transform[3, 3]) - 1.) > 1e-9 or np.abs(transform[3, :3]).max() > 1e-9
            or np.abs(transform[:3, :3].T @ transform[:3, :3] - np.eye(3)).max() > 1e-9
            or not math.isclose(float(np.linalg.det(transform[:3, :3])), 1., abs_tol=1e-9)
            or np.abs(transform[:3, :3] - camera_basis()).max() > 1e-9
            or np.abs(transform[:3, 3] - np.array([0., -floor, 0.])).max() > 1e-9):
        raise ValueError('FAB camera-to-world transform is not the fixed Y-up basis with the recorded floor shift')
    out['floor_estimate'] = floor
    out['world_from_camera'] = transform.astype(np.float64)
    out['K'] = intrinsic.astype(np.float64)
    return out


def load(path, frames=None):
    """Read one committed numeric record. Pickles are never accepted."""
    path = Path(path)
    size = path.stat().st_size
    if size > RECORD_LIMIT_BYTES:
        raise ValueError(f'FAB numeric record exceeds {RECORD_LIMIT_BYTES} bytes')
    with np.load(path, allow_pickle=False) as data:
        unknown = sorted(set(data.files) - set(SPECS))
        if unknown:
            raise ValueError('FAB numeric record has unexpected arrays: ' + ', '.join(unknown))
        missing = sorted(set(ARRAYS) - set(data.files))
        if missing:
            raise ValueError('FAB numeric record is missing arrays: ' + ', '.join(missing))
        record = {name: np.array(data[name]) for name in (*ARRAYS, *(name for name in OPTIONAL_ARRAYS if name in data.files))}
    return validate(record, frames)


def contact_logits(contact):
    """Clipped correction logits (1,frames,6) from FAB contact probabilities.

    FAB's static channels are already left heel, left toe, right heel, right toe
    first (fab.contact_regions.CONTACT_NAMES order), which is the order the
    existing correction reads. Unused channels keep a neutral zero logit. These
    are image evidence, not ground truth; correction re-checks geometry.
    """
    values = _finite('contact', np.asarray(contact, np.float32))
    if values.ndim != 2 or values.shape[1] != CONTACT_CHANNELS:
        raise ValueError(f'FAB contact array has shape {values.shape}, expected (frames,{CONTACT_CHANNELS})')
    if values.min() < -CONTACT_EPS or values.max() > 1 + CONTACT_EPS:
        raise ValueError('FAB contact probabilities must be probabilities in [0,1]')
    static = np.clip(values[:, :len(CONTACT_STATIC)], CONTACT_EPS, 1 - CONTACT_EPS)
    logits = np.zeros((values.shape[0], CORRECTION_CHANNELS), np.float32)
    logits[:, :static.shape[1]] = np.log(static / (1 - static))
    return logits[None]


def body_array(body, name):
    """One array of an instantiated body model (or a dict exposing it), on the CPU."""
    value = body[name] if hasattr(body, '__getitem__') and not hasattr(body, name) else getattr(body, name)
    for helper in ('detach', 'cpu'):
        if hasattr(value, helper):
            value = getattr(value, helper)()
    values = np.asarray(value, np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f'Target body {name} is not finite')
    return values


def target_anatomy(body):
    """Registered target body arrays needed by the shape fit and the pivot.

    Accepts an instantiated SMPL-X body (or any object/dict exposing
    ``v_template``, ``J_regressor`` and ``shapedirs``), which is what the
    publisher already loads for preview, correction and export.
    """
    array = body_array
    template = array(body, 'v_template')
    regressor = array(body, 'J_regressor')
    directions = array(body, 'shapedirs')
    if template.shape != (SMPLX_VERTEX_COUNT, 3):
        raise ValueError(f'Target body template has shape {template.shape}, expected {(SMPLX_VERTEX_COUNT, 3)}')
    if (regressor.ndim != 2 or regressor.shape != (regressor.shape[0], SMPLX_VERTEX_COUNT)
            or regressor.shape[0] < JOINT_COUNT):
        raise ValueError('Target body joint regressor does not cover the 22 common joints')
    if (directions.ndim != 3 or directions.shape[0] != SMPLX_VERTEX_COUNT
            or directions.shape[1] != 3 or directions.shape[2] < SMPL_BETAS):
        raise ValueError('Target body shape directions are invalid')
    first = regressor[:JOINT_COUNT]
    return dict(v_template=template,
                rest_template=np.einsum('jv,vc->jc', first, template),
                rest_shapes=np.einsum('jv,vcd->jcd', first, directions[..., :SMPL_BETAS]))


def neutral_shape(rest_joints, anatomy):
    """One constant target-model shape reproducing the source rest bone offsets.

    Regularised least squares on the 21 common bone offsets, which are affine in
    the target model's linear shape basis. Ridge-limited and range-limited, so a
    source shape the target basis cannot express stays close to neutral instead
    of producing an exaggerated body.
    """
    source = np.asarray(rest_joints, np.float64)
    parents = np.asarray(PARENTS)
    if source.shape != (JOINT_COUNT, 3) or not np.isfinite(source).all():
        raise ValueError('FAB rest joints are invalid')
    bones = source[1:] - source[parents[1:]]
    if (np.linalg.norm(bones, axis=-1) < 1e-4).any() or source[:, 1].max() - source[:, 1].min() < 0.2:
        raise ValueError('FAB rest skeleton is invalid')
    template, shapes = anatomy['rest_template'], anatomy['rest_shapes']
    offsets = bones.reshape(-1)
    neutral = (template[1:] - template[parents[1:]]).reshape(-1)
    system = (shapes[1:] - shapes[parents[1:]]).reshape(-1, SMPL_BETAS)
    scale = float(np.sqrt((system ** 2).sum(0)).mean()) or 1.
    solved = np.linalg.lstsq(np.vstack((system, SHAPE_RIDGE * scale * np.eye(SMPL_BETAS))),
                             np.concatenate((offsets - neutral, np.zeros(SMPL_BETAS))), rcond=None)[0]
    betas = np.clip(solved, -SHAPE_LIMIT, SHAPE_LIMIT)
    shaped = template + np.einsum('jcd,d->jc', shapes, betas)
    residual = float(np.abs((shaped[1:] - shaped[parents[1:]]) - (source[1:] - source[parents[1:]])).max())
    return dict(betas=betas.astype(np.float32), rest_joints=shaped.astype(np.float32),
                pelvis=shaped[0].copy(), residual_m=residual,
                height_m=float(shaped[:, 1].max() - shaped[:, 1].min()))


def _rotvec_to_matrix(values):
    from scipy.spatial.transform import Rotation
    return Rotation.from_rotvec(np.asarray(values, np.float64).reshape(-1, 3)).as_matrix()


def _matrix_to_rotvec(values):
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(np.asarray(values, np.float64).reshape(-1, 3, 3)).as_rotvec().reshape(len(values), 3)


def hand_boxes(record, width, height):
    """Square left/right hand boxes (frames,2,4) in working-video pixels for the
    optional FAH pass, and which of them are usable.

    They come from FAB's own predicted 2D hand points, never from another
    detector or stage. An unusable hand (no stored points, too small to carry
    finger pixels, mostly outside the frame, or larger than a share of the body
    box) is written as an empty box, which the FAH child's own box rule rejects.
    """
    n = int(record['frames'])
    points = record.get('hand_keypoints') if isinstance(record, dict) else None
    points = np.zeros((n, 2, HAND_KEYPOINTS, 2)) if points is None else np.asarray(points, np.float64)
    if points.shape != (n, 2, HAND_KEYPOINTS, 2):
        raise ValueError('FAB hand points do not match the numeric record')
    low, high = points.min(2), points.max(2)
    extent = (high - low).max(-1)
    side = np.maximum(HAND_BOX_SCALE * extent, HAND_BOX_MIN_SIDE)
    centre = (low + high) / 2
    boxes = np.concatenate([centre - side[..., None] / 2, centre + side[..., None] / 2], -1)
    size = boxes[..., 2:] - boxes[..., :2]
    clipped = np.maximum(0, np.minimum(boxes[..., 2:], [width, height]) - np.maximum(boxes[..., :2], 0))
    body = np.asarray(record['boxes'], np.float64).reshape(n, 4)
    longest = np.maximum(body[:, 2] - body[:, 0], body[:, 3] - body[:, 1])[:, None]
    usable = ((np.isfinite(boxes).all(-1) & (extent >= HAND_MIN_EXTENT) & (clipped.min(-1) >= 4)
              & (clipped.prod(-1) / np.maximum(size.prod(-1), 1) >= .1) & (side <= HAND_BOX_MAX_SHARE * longest)))
    return np.where(usable[..., None], boxes, 0.).astype(np.float32), usable


def finger_poses(fingers, body):
    """(2,frames,45) SMPL-X hand poses, left then right, from the FAH finger pass.

    An SMPL-X hand pose is added to the body's own per-hand mean pose before it
    becomes a rotation, and the converted MANO finger slots are in the SMPL-X
    hand order, so each fitted slot is ``rotvec(local rotation) - hand_mean``.
    A hand FAH did not fit keeps zeros, i.e. the same relaxed mean pose as a
    plain FAB result; nothing is carried over from neighbouring frames.
    """
    rotations = np.asarray(fingers['rotations'], np.float64)
    valid = np.asarray(fingers['valid'])
    n = len(valid)
    if rotations.shape != (n, 2, HAND_JOINTS, 3, 3) or valid.shape != (n, 2) or valid.dtype != bool:
        raise ValueError('FAH finger rotations do not match the FAB frames')
    if not np.isfinite(rotations).all() or np.max(np.abs(rotations.swapaxes(-1, -2) @ rotations - np.eye(3))) > 2e-4 \
            or np.min(np.linalg.det(rotations)) < .999:
        raise ValueError('FAH finger rotations are not rotations')
    poses = np.zeros((2, n, HAND_POSE_WIDTH), np.float32)
    for side, name in enumerate(('left', 'right')):
        if not hasattr(body, name + '_hand_mean'):
            raise ValueError(f'The registered SMPL-X body has no {name} hand mean pose; FAH fingers cannot be published on it.')
        mean = body_array(body, name + '_hand_mean').reshape(HAND_JOINTS, 3)
        rows = np.flatnonzero(valid[:, side])
        if len(rows):      # a hand FAH never fitted in this clip stays in the relaxed pose
            local = _matrix_to_rotvec(rotations[rows, side].reshape(-1, 3, 3)).reshape(len(rows), HAND_JOINTS, 3) - mean
            poses[side, rows] = local.reshape(len(rows), HAND_POSE_WIDTH).astype(np.float32)
    return poses


def result_parameters(record, body, fingers=None):
    """Standard SMPL-X result parameters from one validated numeric record.

    Returns the app's usual camera-space and world-space parameter dictionaries,
    the correction contact channels, the static camera transform and the fitted
    target shape. The SMPL-X translation of each space is the pelvis placement
    minus that space's shape-dependent target pelvis pivot, so the published
    joints land where FAB placed them.

    ``fingers`` (the optional FAH pass: ``rotations`` (frames,2,15,3,3) and
    ``valid`` (frames,2)) is the only thing that adds finger motion.
    """
    record = validate(record)
    n = record['frames']
    anatomy = target_anatomy(body)
    shape = neutral_shape(record['rest_joints'], anatomy)
    if shape['residual_m'] > SHAPE_RESIDUAL_LIMIT_M:
        raise ValueError(f"The FAB actor shape cannot be mapped onto the target body "
                         f"({shape['residual_m'] * 1000:.1f} mm bone error); publish with a matching SMPL-X body.")
    pivot = shape['pelvis']
    pose = record['smpl_pose'].astype(np.float64)
    betas = np.repeat(shape['betas'][None], n, 0)
    incam_pelvis = record['camera_pelvis'].astype(np.float64)
    world = record['world_from_camera']
    rotation, shift = world[:3, :3], world[:3, 3]
    body_pose = pose[:, 3:3 + SMPL_BODY_POSE_WIDTH].astype(np.float32)
    world_orient = _matrix_to_rotvec(np.einsum('ij,tjk->tik', rotation, _rotvec_to_matrix(pose[:, :3])))
    world_pelvis = np.einsum('ij,tj->ti', rotation, incam_pelvis) + shift
    if fingers is None:
        left_hand = right_hand = np.zeros((n, HAND_POSE_WIDTH), np.float32)
        policy, fitted = HAND_POSE_POLICY, dict(source=None, frames=int(n))
    else:
        left_hand, right_hand = finger_poses(fingers, body)
        valid = np.asarray(fingers['valid'])
        left, right = int(valid[:, 0].sum()), int(valid[:, 1].sum())
        # A pass that fitted no hand leaves every hand in the relaxed pose of a plain
        # FAB result, which must not be published or exported as FAH finger motion.
        used = bool(left or right)
        policy = FAH_HAND_POSE_POLICY if used else HAND_POSE_POLICY
        fitted = dict(source='fah' if used else None, frames=int(n), left=left, right=right)
    # Only the root orientation and the pelvis position change between spaces;
    # every local rotation keeps its native convention.
    incam = dict(global_orient=pose[:, :3].astype(np.float32), body_pose=body_pose, betas=betas,
                 transl=(incam_pelvis - pivot).astype(np.float32),
                 left_hand_pose=left_hand, right_hand_pose=right_hand)
    world_params = dict(global_orient=world_orient.astype(np.float32), body_pose=body_pose.copy(), betas=betas.copy(),
                        transl=(world_pelvis - pivot).astype(np.float32),
                        left_hand_pose=left_hand.copy(), right_hand_pose=right_hand.copy())
    inverse = np.eye(4)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ shift
    return dict(version=VERSION, source_model='fab', frames=n, fps=record['fps'],
                hand_pose_policy=policy,
                fingers=fitted,
                smpl_params_incam=incam, smpl_params_global=world_params,
                net_outputs={'static_conf_logits': contact_logits(record['contact'])},
                contact_logits=contact_logits(record['contact']),
                world_from_camera=world.astype(np.float64), camera_from_world=inverse.astype(np.float64),
                K=record['K'].astype(np.float64), boxes=record['boxes'].copy(),
                rest_joints=shape['rest_joints'], pelvis_pivot=pivot.astype(np.float32),
                floor_estimate_m=record['floor_estimate'],
                shape=dict(betas=shape['betas'], residual_m=shape['residual_m'], height_m=shape['height_m'],
                           model_shape='constant clip median shape fitted to the target basis'),
                native=dict(smpl_pose=record['smpl_pose'].copy(), smpl_betas=record['smpl_betas'].copy(),
                            smpl_transl=record['smpl_transl'].copy(), camera_pelvis=record['camera_pelvis'].copy(),
                            median_betas=record['median_betas'].copy()))
