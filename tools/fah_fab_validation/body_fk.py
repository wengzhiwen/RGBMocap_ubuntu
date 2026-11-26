"""SMPL-H forward kinematics through RGB MoCap's own FlowHMR body model.

Read-only: imports `third_party/FlowHMR/flowhmr/core/bodymodels/smpl_skeleton.py`
and the registered SMPL-H neutral model that FlowHMR export already uses. Used to
turn local joint rotations into posed joint positions for previews, metrics and
the take checks.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "third_party" / "FlowHMR"
for _path in (str(ROOT), str(Path(__file__).resolve().parent)):
    if _path not in sys.path:
        sys.path.insert(0, _path)


def smplh_path():
    from mocap.models import BY_KEY
    path = BY_KEY["smplh"].path
    if not path.is_file():
        raise SystemExit(f"registered SMPL-H model missing: {path} (import it in FlowHMR Model Manager)")
    return path


def skeleton(path=None):
    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))
    from flowhmr.core.bodymodels.smpl_skeleton import SMPLSkeleton
    return SMPLSkeleton(model_path=str(path or smplh_path()))


def check_parents(skeleton_obj, expected):
    """Regression check: the hand_math tree equals the registered model's tree."""
    parents = tuple(int(p) for p in np.asarray(skeleton_obj.parents).tolist())
    if parents != tuple(expected):
        raise AssertionError(f"SMPL-H tree changed: model {parents} != hand_math {tuple(expected)}")
    return parents


def rest_joints(skeleton_obj, shapes):
    """(52,3) rest joint positions of this actor's shape (zero pose)."""
    import torch
    with torch.no_grad():
        return skeleton_obj.compute_j_shaped(torch.from_numpy(np.asarray(shapes, np.float32).reshape(1, -1)))[0].numpy().astype(np.float64)


def joints(skeleton_obj, local, shapes, trans, batch=32):
    """Local rotations (n,52,3,3) + shape + root translation -> posed joints (n,52,3)."""
    import torch
    local = np.asarray(local, np.float32)
    shapes = np.asarray(shapes, np.float32).reshape(1, -1)
    trans = np.asarray(trans, np.float32)
    n = len(local)
    out = np.empty((n, 52, 3), np.float64)
    with torch.inference_mode():
        for start in range(0, n, batch):
            stop = min(n, start + batch)
            values = skeleton_obj(dict(rot_mats=torch.from_numpy(local[start:stop]),
                                       shapes=torch.from_numpy(shapes),
                                       trans=torch.from_numpy(trans[start:stop])))["keypoints3d"]
            out[start:stop] = values.numpy().astype(np.float64)
    return out
