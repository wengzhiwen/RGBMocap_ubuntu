#!/usr/bin/env python3
"""Unit regressions for the FAH/FAB research math (no GPU, no model inference).

    .venv/bin/python tools/fah_fab_validation/test_hand_math.py [donor-flowhmr-project]

Synthetic cases always run. When a donor FlowHMR project directory is given (and
its artifacts exist) the camera-convention, fused-output and MANO-tree checks run
against that project's real files. Nothing here starts a model.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

import hand_math as H

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _path in (str(ROOT), str(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

FAILED = []


def case(function):
    try:
        function()
    except Exception as error:                                    # noqa: BLE001 - report and continue
        FAILED.append((function.__name__, repr(error)))
        print(f"FAIL {function.__name__}: {error!r}")
    else:
        print(f"ok   {function.__name__}")
    return function


def random_rotations(count, seed=0):
    return Rotation.random(count, random_state=np.random.default_rng(seed)).as_matrix()


def load_pt(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


@case
def rot6d_matches_flowhmr_upstream():
    """Our rot6d packing is FlowHMR's packing, not a split of our own invention."""
    import torch
    sys.path.insert(0, str(ROOT / "third_party" / "FlowHMR"))
    from flowhmr.core.math.geometry import rotation_matrix_to_rot6d, rot6d_to_rotation_matrix

    matrices = random_rotations(64, seed=1)
    mine = H.matrix_to_rot6d(matrices)
    theirs = rotation_matrix_to_rot6d(torch.tensor(matrices)).numpy()
    assert mine.shape == (64, 6) == theirs.shape
    assert np.abs(mine - theirs).max() < 1e-12, (mine[:2], theirs[:2])
    assert np.abs(H.rot6d_to_matrix(theirs) - matrices).max() < 1e-9
    assert np.abs(rot6d_to_rotation_matrix(torch.tensor(mine)).numpy() - matrices).max() < 1e-9
    # Packing and Gram-Schmidt order: `rot6d.reshape(N,3,2)` holds the first two
    # *columns* row-wise, and the third column is b1 x b2.
    sheared = matrices.copy()
    sheared[:, 1, 0] += 0.37
    six = H.matrix_to_rot6d(sheared).reshape(-1, 3, 2)
    a1, a2 = six[:, :, 0], six[:, :, 1]
    assert np.abs(a1 - sheared[:, :, 0]).max() < 1e-12 and np.abs(a2 - sheared[:, :, 1]).max() < 1e-12
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    u = a2 - np.einsum("ni,ni->n", b1, a2)[:, None] * b1
    b2 = u / np.linalg.norm(u, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    assert np.abs(H.rot6d_to_matrix(H.matrix_to_rot6d(sheared)) - np.stack((b1, b2, b3), -1)).max() < 1e-9


@case
def chain_root_frames_semantics():
    """`chain` takes a parent-less joint's frame from `root_frames`, not from its own
    local slot; `local_from_global` is the inverse once that is respected."""
    local = random_rotations(52, seed=2)
    root = np.repeat(local[None, 0], 52, axis=0)
    assert np.abs(H.chain(H.SMPLH_PARENTS, local)[0] - np.eye(3)).max() < 1e-12
    assert np.abs(H.chain(H.SMPLH_PARENTS, local, root_frames=root)[0] - local[0]).max() < 1e-12
    global_frames = H.chain(H.SMPLH_PARENTS, local, root_frames=root)
    back = H.local_from_global(H.SMPLH_PARENTS, global_frames)
    assert np.abs(back - local).max() < 1e-9
    assert H.angle_between(back, local).max() < 1e-7


@case
def chain_parent_composition():
    local = random_rotations(52, seed=3)
    root = np.broadcast_to(local[:, None, 0], local.shape)
    global_frames = H.chain(H.SMPLH_PARENTS, local, root_frames=root)
    for child in (12, 22, 24, 37, 51):
        parent = int(H.SMPLH_PARENTS[child])
        want = global_frames[parent] @ local[child] if parent >= 0 else local[child]
        assert np.abs(global_frames[child] - want).max() < 1e-12, child


@case
def mirror_is_an_involutive_proper_map():
    right = random_rotations(24, seed=4)
    left = H.mirror_frame(right)
    assert np.abs(H.mirror_frame(left) - right).max() < 1e-12
    assert np.abs(np.linalg.det(left) - 1.0).max() < 1e-9
    assert np.abs(left - H.MIRROR_X @ right @ H.MIRROR_X).max() < 1e-12
    translation = np.array([0.3, -1.2, 2.0])
    assert np.allclose(H.MIRROR_X @ translation, [-0.3, -1.2, 2.0])


@case
def mano_chain_mirroring_is_consistent():
    """FAH fits a right MANO to a mirrored left crop; reflecting the parameters must
    reflect the whole chain, which is the left-hand convention `fah_hands.py` stores."""
    rng = np.random.default_rng(6)
    pose = rng.normal(0, 0.4, 45)
    root = Rotation.from_rotvec(rng.normal(0, 1, 3)).as_matrix()
    wrist_r, slots_r = H.mano_global(pose, root, mirror=False)
    wrist_l, slots_l = H.mano_global(pose, root, mirror=True)
    assert np.abs(wrist_l - H.mirror_frame(wrist_r)).max() < 1e-9
    assert np.abs(slots_l - H.mirror_frame(slots_r)).max() < 1e-9
    # The chain itself: each slot's global frame is its parent's frame times its own
    # local rotation, in MANO's tree.
    local = Rotation.from_rotvec(pose.reshape(15, 3)).as_matrix()
    for slot in range(1, 15):
        parent = int(H.MANO_PARENTS[slot + 1])
        want = slots_r[parent - 1] @ local[slot] if parent else root @ local[slot]
        assert np.abs(slots_r[slot] - want).max() < 1e-9, slot


@case
def transfer_to_smplh_keeps_the_wrist_and_the_tree():
    rng = np.random.default_rng(7)
    pose = rng.normal(0, 0.3, 45)
    root = Rotation.from_rotvec(rng.normal(0, 1, 3)).as_matrix()
    baseline_wrist = Rotation.from_rotvec(rng.normal(0, 1, 3)).as_matrix()
    for mirror in (False, True):
        local = H.transfer_to_smplh(pose, root, baseline_wrist, mirror=mirror)
        assert local.shape == (15, 3, 3)
        H.check_proper(local, name="transferred local")
        wrist, slots = H.mano_global(pose, root, mirror=mirror)
        align = baseline_wrist @ wrist.T
        # First joint of every finger: local == Gm0^T @ Gm[slot], baseline-wrist free.
        for slot in (0, 3, 6, 9, 12):
            assert H.angle_between(local[slot], wrist.T @ slots[slot]).max() < 1e-9, slot
        # Descendants are measured against the aligned parent slot.
        for slot in (1, 2, 4, 5, 7, 8, 10, 11, 13, 14):
            assert H.angle_between(local[slot], (align @ slots[slot - 1]).T @ (align @ slots[slot])).max() < 1e-9, slot
        # Rebuilding the global frames from the returned local rotations, through the
        # SMPL-H hand tree (which equals MANO's tree) with the baseline wrist as root,
        # reproduces the aligned MANO chain.
        full = np.concatenate((np.eye(3)[None], local), 0)
        rebuilt = H.chain(H.MANO_PARENTS, full, root_frames=baseline_wrist)[1:]
        assert H.angle_between(rebuilt, align @ slots).max() < 1e-9, \
            (mirror, float(H.angle_between(rebuilt, align @ slots).max()))


@case
def hand_tree_tables_agree():
    mano_chain = tuple(int(H.MANO_PARENTS[slot + 1]) for slot in range(15))
    left = tuple(int(p) for p in H.SMPLH_PARENTS[22:37])
    right = tuple(int(p) for p in H.SMPLH_PARENTS[37:52])
    relative = lambda parents, wrist: tuple(0 if p == wrist else p - 22 + 1 for p in parents)
    assert relative(left, 20) == mano_chain, (mano_chain, relative(left, 20))
    assert tuple(0 if p == 21 else p - 37 + 1 for p in right) == mano_chain, (mano_chain, right)
    assert H.HAND_SLICES[0] == slice(22, 37) and H.HAND_SLICES[1] == slice(37, 52)
    assert H.WRIST == (20, 21) and len(H.SMPLH_PARENTS) == 52
    assert H.HAND_SLOT_NAMES[0] == "index1" and len(H.HAND_SLOT_NAMES) == 15
    assert [int(p) for p in H.SMPLH_PARENTS[22:37]] == [20, 22, 23, 20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34, 35]


@case
def rest_report_detects_order_and_rest_differences():
    rest = np.array([[0.0, 0.0, 0.0]] + [[0.02 * (i % 3 + 1), 0.05 * (i // 3) + 0.02, 0.01 * (i % 3)]
                                        for i in range(15)])
    good = H.rest_report(rest, rest.copy())
    assert good["order_matches"] and good["identity_max_m"] < 1e-9 and good["identity_is_optimal"], good
    assert good["rest_bone_angle_max_deg"] < 1e-6
    swapped = rest.copy()
    swapped[[1, 4]] = swapped[[4, 1]]                      # index1 <-> middle1
    bad = H.rest_report(rest, swapped)
    assert not bad["order_matches"], bad
    assert bad["rest_bone_angle_max_deg"] > 5.0, bad["rest_bone_angle_max_deg"]
    mirrored = H.rest_report(rest, np.c_[-rest[:, 0], rest[:, 1:]], mirror=True)
    assert mirrored["identity_max_m"] < 1e-9, mirrored


@case
def angle_helpers_match_scipy():
    a, b = random_rotations(32, seed=8), random_rotations(32, seed=9)
    ours = H.angle_between(a, b)
    theirs = np.degrees(np.array([float((Rotation.from_matrix(x).inv() * Rotation.from_matrix(y)).magnitude())
                                  for x, y in zip(a, b)]))
    assert np.abs(ours - theirs).max() < 1e-9, (ours[:3], theirs[:3])
    assert ours.max() <= 180.0 + 1e-9 and ours.min() >= 0.0
    assert H.angle_between(a, a).max() < 1e-10
    assert abs(H.angle_between(np.eye(3), Rotation.from_euler("z", 90, degrees=True).as_matrix()) - 90.0) < 1e-9


@case
def check_proper_rejects_bad_matrices():
    good = random_rotations(4, seed=10)
    assert H.check_proper(good).shape == (4, 3, 3)
    broken = good.copy()
    broken[0, 0, 0] += 0.5
    for bad in (broken, good * 2.0):
        try:
            H.check_proper(bad)
        except ValueError:
            continue
        raise AssertionError(f"check_proper accepted {bad.shape[0]} bad matrices")


@case
def temporal_policy_drops_short_runs():
    from fuse_hands import temporal_policy

    accepted = np.array([[1, 0], [1, 0], [0, 0], [1, 0], [0, 0], [1, 1], [1, 1]], bool)
    keep = temporal_policy(accepted.copy(), accepted.shape[0], min_run=2)
    assert keep[:, 0].tolist() == [True, True, False, False, False, True, True], keep[:, 0]
    assert keep[:, 1].tolist() == [False] * 5 + [True, True], keep[:, 1]
    assert not temporal_policy(np.zeros((5, 2), bool), 5, min_run=1).any()
    assert temporal_policy(np.ones((4, 2), bool), 4, min_run=9).sum() == 0
    single = np.zeros((4, 2), bool)
    single[2, 1] = True
    assert not temporal_policy(single.copy(), 4, min_run=2).any()
    assert temporal_policy(single.copy(), 4, min_run=1)[2, 1]
    after = single.copy()
    dropped = temporal_policy(after, 4, min_run=3)
    assert single[2, 1] and after[2, 1] and not dropped[2, 1], "temporal_policy must not modify its input"
    assert np.array_equal(temporal_policy(accepted.copy(), 7, min_run=1), accepted)


@case
def gate_reason_codes_are_specific():
    import argparse

    from fuse_hands import gate

    frames = 3
    cfg = argparse.Namespace(presence=0.5, side=0.5, residual=0.035, max_joint_deg=100.0,
                             max_finger_deg=220.0, max_hand_deg=700.0)
    clean = dict(accepted=np.ones((frames, 2), bool), fitted=np.ones((frames, 2), bool),
                 presence=np.full((frames, 2), 0.9), side_score=np.full((frames, 2), 0.9),
                 fit_residual_m=np.full((frames, 2), 0.01), mano_pose=np.zeros((frames, 2, 45)))

    def patched(key, value):
        arrays = {name: np.asarray(copy, np.float64 if name != "accepted" and name != "fitted" else bool).copy()
                  for name, copy in clean.items()}
        arrays[key][0, 0] = value
        return arrays

    with tempfile.TemporaryDirectory() as tmp:
        counter = [0]

        def record(**override):
            arrays = {name: np.asarray(copy).copy() for name, copy in clean.items()}
            arrays.update({name: np.asarray(value) for name, value in override.items()})
            counter[0] += 1
            path = Path(tmp) / f"fah{counter[0]}.npz"
            np.savez(path, **arrays)
            return np.load(path)

        ok, why = gate(record(), 0, 0, cfg)
        assert ok and not why, why
        ok, why = gate(record(**{k: v for k, v in patched("presence", 0.1).items()}), 0, 0, cfg)
        assert not ok and why == ["presence<0.5"], why
        ok, why = gate(record(**{k: v for k, v in patched("side_score", 0.2).items()}), 0, 0, cfg)
        assert not ok and why == ["side<0.5"], why
        ok, why = gate(record(**{k: v for k, v in patched("fit_residual_m", 0.09).items()}), 0, 0, cfg)
        assert not ok and why == ["fit>35mm"], why
        ok, why = gate(record(**{k: v for k, v in patched("mano_pose", np.radians(120)).items()}), 0, 0, cfg)
        assert not ok and "joint>100deg" in why, why
        ok, why = gate(record(**{k: v for k, v in patched("presence", np.nan).items()}), 0, 0, cfg)
        assert not ok and "presence<0.5" in why, why
        ok, why = gate(record(accepted=np.zeros((frames, 2), bool), fitted=np.zeros((frames, 2), bool)), 0, 0, cfg)
        assert not ok and set(why) >= {"box", "no-fit"}, why


@case
def fused_schema_passes_the_apps_motion_validator():
    from mocap.flow_backend import validate_motion
    import torch

    matrices = random_rotations(52 * 3, seed=11).reshape(3, 52, 3, 3)
    rot6d = torch.tensor(H.matrix_to_rot6d(matrices)[None], dtype=torch.float32)
    motion = dict(rot6d=rot6d,
                  root_rot6d=rot6d[:, :, 0], body_rot6d=rot6d[:, :, 1:],
                  trans=torch.zeros(1, 3, 3), shapes=torch.zeros(1, 1, 16),
                  scene_offset=torch.zeros(1, 3), source_representation="smplh",
                  flow_version=1, placement_version=2)
    validate_motion(motion, 3)
    broken = dict(motion, rot6d=rot6d[:, :, 1:])
    try:
        validate_motion(broken, 3)
    except ValueError:
        pass
    else:
        raise AssertionError("validate_motion accepted a wrong-shaped rot6d")


def donor_cases(donor, fused_path=None, fah_path=None):
    """Checks that use the donor project's real artifacts (still no inference)."""
    experiment = donor.parents[1]
    fused = fused_path or experiment / "agent-smoke2" / "fused.npz"
    fah = fah_path or experiment / "agent-smoke" / "fah_smoke2.npz"

    @case
    def camera_convention_reproduces_the_recorded_anchor_residual():
        import body_fk
        import preview_hands

        frames = int(np.load(donor / "flow_camera.npz", allow_pickle=False)["camera_RT_yup"].shape[0])
        rotation, offset, k, world = preview_hands.camera_project(donor, frames)
        placed = load_pt(donor / "hmr4d_results.pt")
        joints = body_fk.joints(body_fk.skeleton(), H.rot6d_to_matrix(np.asarray(load_pt(donor / "hmr4d_results.pt")["rot6d"], np.float64)[0]),
                                np.asarray(placed["shapes"], np.float64).reshape(-1),
                                np.asarray(placed["trans"], np.float64)[0])
        check = preview_hands.anchor_check(donor, joints, rotation, offset, k)
        recorded = check["recorded_residual_m"]
        assert recorded, "flow_world.json records no anchor residual to compare with"
        assert abs(check["reproject_residual_m"]["mean"] - recorded["mean"]) < 1e-5, check
        assert abs(check["reproject_residual_m"]["max"] - recorded["maximum"]) < 1e-5, check
        _, camera = preview_hands.project_joints(joints, rotation, offset, k, 0)
        assert (camera[:, 2] > 0).all(), f"negative depth in the camera frame: {camera[:, 2].min()}"

    @case
    def fused_output_keeps_the_body_and_only_moves_fingers():
        import json

        if not fused.is_file():
            raise AssertionError(f"run fuse_hands.py first: {fused} is missing")
        data = np.load(fused, allow_pickle=False)
        n = data["placed_rot6d"].shape[0]
        assert data["fused_rot6d"].shape == (n, 52, 6) and data["baseline_rot6d"].shape == (n, 52, 6)
        assert np.array_equal(data["fused_rot6d"][:, 0:22], data["placed_rot6d"][:, 0:22])
        base, fus = data["baseline_local"], data["fused_local"]
        H.check_proper(fus)
        untouched = np.ones((n, 52), bool)
        for side in range(2):
            untouched[:, H.HAND_SLICES[side]] = ~data["applied"][:, side][:, None]
        delta = np.abs(fus - base).max((-1, -2))
        assert not delta[untouched].any(), "baseline finger rows changed where FAH was not applied"
        for side in range(2):
            rows = H.HAND_SLICES[side]
            changed = (np.abs(fus[:, rows] - base[:, rows]).max(-1) > 0).any(0)
            assert changed.any() == bool(data["applied"][:, side].any()), side
        report = json.loads((fused.parent / "report.json").read_text(encoding="utf-8"))
        assert report["applied"] == int(data["applied"].sum())
        assert report["gated_after_temporal"] == int(data["accepted"].sum()), report
        assert report["gated"] >= report["gated_after_temporal"] and report["dropped_by_temporal"] == \
            report["gated"] - report["gated_after_temporal"], report
        assert report["unchanged_rows_max_delta"] == 0.0, report["unchanged_rows_max_delta"]

    @case
    def mano_tree_and_rest_agreement_from_the_fah_run():
        import body_fk

        if not fah.is_file():
            raise AssertionError(f"run fah_hands.py first: {fah} is missing")
        data = np.load(fah, allow_pickle=False)
        assert tuple(int(p) for p in data["mano_parents"]) == H.MANO_PARENTS, data["mano_parents"]
        assert data["mano_rest_joints"].shape == (16, 3)
        model = body_fk.skeleton()
        body_fk.check_parents(model, H.SMPLH_PARENTS)
        placed = load_pt(donor / "hmr4d_results.pt")
        rest = body_fk.rest_joints(model, np.asarray(placed["shapes"], np.float64).reshape(-1))
        for side, name in enumerate(("left", "right")):
            hand = rest[H.HAND_SLICES[side].start:H.HAND_SLICES[side].stop]
            with_wrist = np.concatenate((rest[H.WRIST[side]][None], hand), 0)
            report = H.rest_report(np.asarray(data["mano_rest_joints"], np.float64), with_wrist, mirror=(side == 0))
            assert report["order_matches"], (name, report)
            assert report["rest_bone_angle_max_deg"] < 45.0, (name, report["rest_bone_angle_deg"])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("donor", nargs="*", help="donor FlowHMR project directory")
    parser.add_argument("--fused", help="fuse_hands.py fused.npz (default: <experiment>/agent-smoke2/fused.npz)")
    parser.add_argument("--fah", help="fah_hands.py output (default: <experiment>/agent-smoke/fah_smoke2.npz)")
    options = parser.parse_args()
    for argument in options.donor:
        donor = Path(argument).resolve()
        if not (donor / "project.json").is_file():
            print(f"skip donor cases: {donor} is not a project directory")
            continue
        donor_cases(donor, Path(options.fused).resolve() if options.fused else None,
                    Path(options.fah).resolve() if options.fah else None)
    if FAILED:
        print(f"\n{len(FAILED)} failed case(s)")
        for name, error in FAILED:
            print(f"  {name}: {error}")
        raise SystemExit(1)
    print("\nall cases passed")
