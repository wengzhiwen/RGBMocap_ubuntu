"""Shared math for the FAH -> FlowHMR finger transfer and the FAB body comparison.

Conventions (checked against the production code, not assumed):

* FlowHMR rot6d layout is the upstream one in
  `third_party/FlowHMR/flowhmr/core/math/geometry.py`: a rotation matrix's first
  two **columns**, stored row-major, i.e. `[c0[0], c1[0], c0[1], c1[1], c0[2], c1[2]]`.
  `rot6d_to_matrix` / `matrix_to_rot6d` below reproduce it exactly and are checked
  against the upstream torch functions in `tests/test_hand_math.py`.
* A pose is one rotation matrix per joint, applied in the parent's frame; every
  rest joint frame is canonical (no rest orientation) and neither FlowHMR's SMPL-H
  rot6d nor FAH's MANO fit adds a hand mean pose.
* SMPL-H's and MANO's fifteen finger slots are the same order
  (index, middle, pinky, ring, thumb) and the same chain topology, so MANO local
  joint rotations *are* SMPL-H local joint rotations after the left-hand x
  reflection. `tests/test_hand_math.py` proves the equivalent wrist-relative
  global transfer gives identical results, so there is one mode, not two.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

MIRROR_X = np.diag([-1.0, 1.0, 1.0])          # right-hand model frame -> real left hand
IDENTITY = np.eye(3)

# ------------------------------------------------------------------ FlowHMR rot6d
def rot6d_to_matrix(rot6d):
    """Upstream `rot6d_to_rotation_matrix`: view (..., 3, 2), columns 0 and 1."""
    r = np.asarray(rot6d, dtype=np.float64)
    if r.shape[-1] != 6:
        raise ValueError(f"rot6d must end in 6, got {r.shape}")
    x = r.reshape(r.shape[:-1] + (3, 2))
    a1, a2 = x[..., 0], x[..., 1]
    b1 = a1 / np.maximum(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-12)
    b2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 = b2 / np.maximum(np.linalg.norm(b2, axis=-1, keepdims=True), 1e-12)
    b3 = np.cross(b1, b2)
    return np.stack((b1, b2, b3), axis=-1)


def matrix_to_rot6d(matrices):
    """Upstream `rotation_matrix_to_rot6d`: first two columns, row-major."""
    m = frames(matrices)
    return m[..., :, :2].reshape(m.shape[:-2] + (6,))


def frames(matrices, shape=None, name="rotations"):
    """Validate a float64 rotation array; `shape` is the full expected shape."""
    m = np.asarray(matrices, dtype=np.float64)
    if m.shape[-2:] != (3, 3):
        raise ValueError(f"{name} must have trailing shape (3, 3), got {m.shape}")
    if not np.isfinite(m).all():
        raise ValueError(f"{name} contains non-finite values")
    if shape is not None and m.shape != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}, got {m.shape}")
    return m


def check_proper(matrices, ortho_tol=1e-4, det_tol=1e-3, name="rotations"):
    m = frames(matrices, name=name)
    error = float(np.abs(np.swapaxes(m, -1, -2) @ m - IDENTITY).max())
    det = float(np.linalg.det(m).min())
    if error > ortho_tol or det < 1.0 - det_tol:
        raise ValueError(f"{name}: not proper rotations (orthogonality {error:.3g}, min det {det:.6f})")
    return m


def mirror_frame(matrix, mirror=MIRROR_X):
    """Conjugate a rotation by a reflection: proper -> proper (det stays +1).

    FAH fits the right MANO model to a mirrored left-hand crop, so a left hand's
    model-frame rotation `R` becomes the real-world `M R M`; the same rule is what
    RGB MoCap's HaMeR pass uses (`mocap/flow_hamer.py`).
    """
    return mirror @ frames(matrix) @ mirror


def chain(parents, local, root_frames=None):
    """Local joint rotations -> global rotations along a kinematic tree.

    Rotation FK needs only the tree and the local rotations (no rest offsets).
    `parents[j] < 0` joints take their frame from `root_frames` (identity default).
    """
    local = frames(local)
    single = local.ndim == 2
    if single:
        local = local[None]
    parents = tuple(int(p) for p in parents)
    if len(parents) != local.shape[-3]:
        raise ValueError("parents and local rotations disagree")
    if root_frames is None:
        root_frames = np.broadcast_to(IDENTITY, local.shape)
    root_frames = frames(root_frames)
    if root_frames.shape == (3, 3):
        root_frames = np.broadcast_to(root_frames, local.shape)
    if root_frames.shape != local.shape:
        raise ValueError("root_frames and local rotations disagree")
    flat = local.reshape((-1,) + local.shape[-3:])
    roots = root_frames.reshape((-1,) + root_frames.shape[-3:])
    out = np.empty_like(flat)
    for j, parent in enumerate(parents):
        if not -1 <= parent < j:
            raise ValueError(f"joint {j} has parent {parent}: the tree must be listed parents-first")
        out[:, j] = roots[:, j] if parent < 0 else out[:, parent] @ flat[:, j]
    result = out.reshape(local.shape[:-3] + local.shape[-3:])
    return result[0] if single else result


def local_from_global(parents, global_frames):
    """Inverse of `chain` (global frames -> local frames)."""
    g = frames(global_frames)
    parents = tuple(int(p) for p in parents)
    if len(parents) != g.shape[-3]:
        raise ValueError("global frames and parents disagree")
    out = np.empty_like(g)
    for j, parent in enumerate(parents):
        out[..., j, :, :] = (g[..., parent, :, :].swapaxes(-1, -2) @ g[..., j, :, :]) if parent >= 0 else g[..., j, :, :]
    return out


def angle_between(a, b):
    """Geodesic rotation angle in degrees, elementwise over leading dims."""
    r = Rotation.from_matrix(frames(a).swapaxes(-1, -2) @ frames(b))
    return np.rad2deg(r.magnitude())


# ------------------------------------------------------------------ joint tables
SMPLH_BODY_NAMES = ("MidHip", "LUpLeg", "RUpLeg", "spine", "LLeg", "RLeg", "spine1", "LFoot", "RFoot",
                    "spine2", "LToeBase", "RToeBase", "Neck", "LShoulder", "RShoulder", "Head",
                    "LArm", "RArm", "LForeArm", "RForeArm", "LHand", "RHand")
SMPLH_BODY_PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19)
FINGERS = ("index", "middle", "pinky", "ring", "thumb")          # MANO == SMPL-H order
HAND_SLOT_NAMES = tuple(f"{finger}{part}" for finger in FINGERS for part in (1, 2, 3))
HAND_START = (22, 37)                                            # (left, right)
WRIST = (20, 21)                                                 # LHand / RHand
SMPLH_PARENTS = list(SMPLH_BODY_PARENTS)
for _side, _start in enumerate(HAND_START):
    for _finger in range(5):
        _base = _start + _finger * 3
        SMPLH_PARENTS += [WRIST[_side], _base, _base + 1]
SMPLH_PARENTS = tuple(SMPLH_PARENTS)
assert len(SMPLH_PARENTS) == 52
SMPLH_NAMES = SMPLH_BODY_NAMES + tuple(f"left_{n}" for n in HAND_SLOT_NAMES) \
    + tuple(f"right_{n}" for n in HAND_SLOT_NAMES)
HAND_SLICES = (slice(22, 37), slice(37, 52))                      # (left, right) SMPL-H rows
FOOT_JOINTS = {"LFoot": 7, "RFoot": 8, "LToeBase": 10, "RToeBase": 11}

MANO_PARENTS = (-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14)   # kintree_table[0], joints 0..15
# FAH's own table (fah/fast_mano.MANO21_TO_OPENPOSE); MANO slots 1-3 index, 4-6 middle,
# 7-9 pinky, 10-12 ring, 13-15 thumb, 16-20 the mesh tips thumb..pinky.
MANO21_TO_OPENPOSE = (0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20)

SMPL_BODY_PARENTS = SMPLH_BODY_PARENTS                            # SMPL joints 0..21 == SMPL-H body
SMPL_BODY_COUNT = 22


def side_index(side):
    if side in ("left", 0, False):
        return 0
    if side in ("right", 1, True):
        return 1
    raise ValueError(f"unknown hand side {side!r}")


# ------------------------------------------------------------------ MANO -> SMPL-H
def mano_global(pose45, root_rot, mirror=False):
    """FAH MANO fit -> (wrist frame, fifteen hand-slot global frames).

    `pose45` is FAH's flat-hand axis-angle in MANO's own slot order, `root_rot` the
    fitted MANO global rotation and translation is not needed (rotation FK). A left
    hand is stored by FAH in the mirrored right-hand model frame, so `mirror=True`
    reflects every rotation (`R -> M R M`) into real-world axes first.
    """
    pose = np.asarray(pose45, dtype=np.float64).reshape(-1, 3)
    if pose.shape != (15, 3):
        raise ValueError(f"MANO pose must hold 45 values, got {np.asarray(pose45).shape}")
    local = Rotation.from_rotvec(pose).as_matrix()
    root = frames(root_rot, (3, 3), "MANO root rotation")
    if mirror:
        local = MIRROR_X @ local @ MIRROR_X
        root = mirror_frame(root)
    # `chain` takes a parent-less joint's frame from `root_frames`, so the fitted
    # MANO root goes there (its own local slot stays identity) and every hand slot
    # keeps it: `Gm[i] = root @ (local FK)`, `Gm0 = root`.
    full = np.concatenate((IDENTITY[None], local), 0)
    global_frames = chain(MANO_PARENTS, full, root_frames=root)
    return global_frames[0], global_frames[1:]


def transfer_to_smplh(pose45, root_rot, baseline_global_wrist, mirror=False):
    """MANO fit -> the fifteen SMPL-H local hand rotations, wrist kept from the baseline.

    The baseline SMPL-H wrist frame `Gb` and FAH's MANO wrist frame `Gm0` give the
    rigid alignment `A = Gb @ Gm0^T`; each SMPL-H hand slot then carries FAH's
    wrist-relative orientation `A @ Gm[i]`, and its local rotation follows the
    SMPL-H tree, i.e. it is measured against the aligned frame of that slot's
    SMPL-H parent: `Gb` for a finger base, `A @ Gm[i-1]` for its two descendants.
    The first finger joint's local rotation reduces to
    `Gm0^T @ Gm[i]`, i.e. it does not depend on the baseline wrist at all, which is
    why this transfer and a direct MANO pose-parameter copy coincide here.
    """
    wrist, slots = mano_global(pose45, root_rot, mirror=mirror)
    gb = frames(baseline_global_wrist, (3, 3), "baseline wrist rotation")
    align = gb @ wrist.T
    check_proper(np.stack((align, *slots)), name="transferred frames")
    local = np.empty((15, 3, 3))
    for slot in range(15):
        # Parent and child have to be in the same frame: the child is `A @ Gm[slot]`,
        # so the parent is the aligned frame too, never the raw MANO `Gm[slot - 1]`.
        parent = gb if slot % 3 == 0 else align @ slots[slot - 1]
        local[slot] = parent.T @ (align @ slots[slot])
    return local


def rest_report(mano_rest, smplh_rest_hand, mirror=False, joint_tolerance=0.03):
    """Quantify MANO vs SMPL-H rest-skeleton agreement for one hand.

    `mano_rest` (16,3) mean-shape MANO rest joints (joint 0 = wrist).
    `smplh_rest_hand` (16,3) this actor's SMPL-H rest joints in the same order:
    [LHand or RHand, index1..thumb3]. A mirrored MANO skeleton is compared for the
    left hand. Returns the joint correspondence check and the per-bone
    rest-direction angle, which is the only place where a MANO rotation is not
    already an SMPL-H rotation.
    """
    from scipy.optimize import linear_sum_assignment

    mano = np.asarray(mano_rest, float)
    smplh = np.asarray(smplh_rest_hand, float)
    if mano.shape != (16, 3) or smplh.shape != (16, 3):
        raise ValueError(f"expected (16,3) and (16,3), got {mano.shape} {smplh.shape}")
    if mirror:
        mano = mano * np.array([-1.0, 1.0, 1.0])
    mano = mano - mano[0]
    smplh = smplh - smplh[0]
    hand = mano[1:]
    offset = smplh[1:].mean(0) - hand.mean(0)
    cost = np.linalg.norm(hand[:, None, :] - smplh[None, 1:, :] + offset, axis=-1)
    rows, cols = linear_sum_assignment(cost)
    permutation = np.argsort(cols)
    angles = {}
    for slot, name in enumerate(HAND_SLOT_NAMES):
        parent = mano[0] if slot % 3 == 0 else mano[slot]          # MANO joint 1+slot-1
        child = hand[slot]
        s_parent = smplh[0] if slot % 3 == 0 else smplh[slot]      # SMPL-H row 1+slot-1
        m_dir, s_dir = child - parent, smplh[1 + slot] - s_parent
        norm = np.linalg.norm(m_dir) * np.linalg.norm(s_dir)
        angles[name] = float(np.degrees(np.arccos(np.clip(m_dir @ s_dir / norm, -1.0, 1.0)))) if norm > 1e-12 else 0.0
    identity = float(np.linalg.norm(hand + offset - smplh[1:], axis=-1).max())
    best = float(cost[rows, cols].max())
    return dict(optimal_is_identity=bool((permutation == np.arange(15)).all()),
                permutation=permutation.tolist(),
                match_max_m=best,
                identity_max_m=identity,
                identity_is_optimal=bool(identity <= best + 1e-9),
                joint_tolerance_m=joint_tolerance,
                order_matches=bool(identity <= joint_tolerance),
                rest_bone_angle_deg=angles,
                rest_bone_angle_max_deg=float(max(angles.values())),
                rest_bone_angle_mean_deg=float(np.mean(list(angles.values()))))
