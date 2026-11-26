#!/usr/bin/env python3
"""Replace only FlowHMR's SMPL-H finger rotations with FAH's MANO fit.

    .venv/bin/python tools/fah_fab_validation/fuse_hands.py \
        --project experiments/fah-fab-20261123/flow-moving-hamer/flowhmr \
        --fah experiments/fah-fab-20261123/run-1/fah_hands.npz \
        --out-dir experiments/fah-fab-20261123/run-1

Controlled comparison, per the donor run:

* the placed body, root trajectory, translation, shape coefficients and scene
  placement come from the donor's `hmr4d_results.pt` and are copied through
  bit-exactly (rows 0..21 of `rot6d`, `trans`, `shapes`, `scene_offset`);
* the **baseline finger slices** (rows 22..51) are restored from the donor's
  unmodified `flow_prediction.pt`, so the HaMeR refinement inside
  `hmr4d_results.pt` does not silently become part of the baseline. `--fingers`
  picks the source: `flow_prediction` (default, plain FlowHMR) or `placed`
  (whatever `hmr4d_results.pt` holds, i.e. the HaMeR result);
* the FAH candidate replaces those slices only where the gates pass. Everywhere
  else the rot6d values are copied through unchanged, so baseline and fused
  differ in exactly the frames/sides reported in `report.json`. A failed or
  occluded hand always keeps its baseline fingers; there is no carry-over of a
  neighbouring frame's fit;
* the FAH npz is checked to hold the donor's own `flow_hand_boxes.npy` boxes
  bit-for-bit, i.e. FAH segmented exactly the crops FlowHMR/HaMeR conditioned on.

Outputs in `--out-dir`:

* `fused_motion.pt`  FlowHMR schema (`rot6d (1,n,52,6)`, `trans`, `shapes`,
  `root_rot6d`, `body_rot6d`, `scene_offset`, `source_representation`,
  `flow_version`, `placement_version`) - what `make_take.py` feeds to the app's
  own BVH/FBX export code;
* `fused.npz`        float64 local rotation matrices of baseline and fused, both
  rot6d variants, per-frame/per-side decisions and gate reasons;
* `report.json`      counts, gate rejections, finger change statistics, and - with
  `--rest-report` - the MANO/SMPL-H rest-skeleton agreement.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import hand_math as H


def load_pt(path):
    import torch
    value = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    return {k: (v.numpy() if hasattr(v, "numpy") else v) for k, v in value.items()}


def gate(fah, index, side, cfg):
    """Return (ok, reasons) for one hand of one frame."""
    reasons = []
    if not bool(fah["accepted"][index, side]):
        reasons.append("box")
    fitted = bool(fah["fitted"][index, side])
    if not fitted:
        reasons.append("no-fit")
    presence = float(fah["presence"][index, side])
    if not np.isfinite(presence) or presence < cfg.presence:
        reasons.append(f"presence<{cfg.presence:g}")
    side_score = float(fah["side_score"][index, side])
    if not np.isfinite(side_score) or side_score < cfg.side:
        reasons.append(f"side<{cfg.side:g}")
    residual = float(fah["fit_residual_m"][index, side])
    if fitted and not np.isfinite(residual):
        reasons.append("fit-nonfinite")          # a fitted row must carry a real residual
    elif residual > cfg.residual:                # NaN never compares greater; `no-fit` covers those rows
        reasons.append(f"fit>{cfg.residual * 1000:g}mm")
    pose = np.asarray(fah["mano_pose"][index, side], np.float64).reshape(15, 3)
    if fitted and not np.isfinite(pose).all():
        reasons.append("pose-nonfinite")
    # MANO/SMPL-H pose parameters are per-slot axis-angle vectors; the angle against
    # the canonical rest frame is the vector norm, not the largest component.
    angles = np.rad2deg(np.linalg.norm(pose, axis=1))
    if angles.max() > cfg.max_joint_deg:
        reasons.append(f"joint>{cfg.max_joint_deg:g}deg")
    chain = [float(angles[f * 3:f * 3 + 3].sum()) for f in range(5)]
    if max(chain) > cfg.max_finger_deg:
        reasons.append(f"finger>{cfg.max_finger_deg:g}deg")
    if sum(chain) > cfg.max_hand_deg:
        reasons.append(f"hand>{cfg.max_hand_deg:g}deg")
    return not reasons, reasons


def temporal_policy(accepted, frames, min_run=2):
    """Runs shorter than `min_run` fall back to the baseline. A rejected frame keeps
    the baseline fingers; an accepted frame is never carried over to it. The input is
    never modified: the pre-policy gate result is what `report.json` counts as gated."""
    keep = np.array(accepted, dtype=bool, copy=True)
    for side in (0, 1):
        column = keep[:, side]
        start = 0
        while start < frames:
            if not column[start]:
                start += 1
                continue
            stop = start
            while stop < frames and column[stop]:
                stop += 1
            if stop - start < min_run:
                column[start:stop] = False
            start = stop
        keep[:, side] = column
    return keep


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project", required=True, help="donor FlowHMR project directory (read-only)")
    p.add_argument("--fah", required=True, help="fah_hands.py output npz")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--fingers", choices=("flow_prediction", "placed"), default="flow_prediction",
                   help="baseline finger source: the unmodified flow_prediction.pt slices, or the placed result's own")
    p.add_argument("--presence", type=float, default=0.5, help="FAH presence head threshold")
    p.add_argument("--side", type=float, default=0.5, help="FAH side (handedness) head threshold")
    p.add_argument("--residual", type=float, default=0.035, help="max MANO fit joint residual in metres")
    p.add_argument("--max-joint-deg", type=float, default=100.0, help="max single finger joint angle vs rest")
    p.add_argument("--max-finger-deg", type=float, default=220.0, help="max summed angle of one finger chain")
    p.add_argument("--max-hand-deg", type=float, default=700.0, help="max summed angle over a hand")
    p.add_argument("--min-run", type=int, default=2, help="drop accepted runs shorter than this")
    p.add_argument("--rest-report", action="store_true", help="also compare MANO and SMPL-H rest skeletons")
    args = p.parse_args(argv)

    project, out = Path(args.project).resolve(), Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    placed = load_pt(project / "hmr4d_results.pt")
    rot6d_placed = np.asarray(placed["rot6d"], np.float64)[0]
    n = rot6d_placed.shape[0]
    if rot6d_placed.shape != (n, 52, 6):
        raise SystemExit(f"{project}/hmr4d_results.pt: unexpected rot6d {rot6d_placed.shape}")
    base_rot6d = rot6d_placed.copy()
    finger_source = "placed"
    if args.fingers == "flow_prediction":
        path = project / "flow_prediction.pt"
        if not path.is_file():
            raise SystemExit(f"{path} is missing; pass --fingers placed")
        raw = load_pt(path)["rot6d"]
        raw = np.asarray(raw, np.float64).reshape(-1, 52, 6)
        if raw.shape[0] != n:
            raise SystemExit(f"{path}: {raw.shape[0]} frames, expected {n}")
        base_rot6d[:, 22:52] = raw[:, 22:52]
        finger_source = "flow_prediction"

    fah = np.load(Path(args.fah).resolve(), allow_pickle=False)
    if fah["mano_pose"].shape[0] != n:
        raise SystemExit(f"{args.fah}: {fah['mano_pose'].shape[0]} frames, expected {n}")
    # FAH must have run on the donor's exact boxes, not on its own detections, or the
    # comparison is not against the crops FlowHMR/HaMeR conditioned on.
    donor_boxes_path = project / "flow_hand_boxes.npy"
    if not donor_boxes_path.is_file():
        raise SystemExit(f"{donor_boxes_path} is missing; cannot confirm the FAH run used the donor boxes")
    donor_boxes = np.load(donor_boxes_path, allow_pickle=False)
    fah_boxes = np.asarray(fah["boxes"], np.float64)
    if not np.array_equal(fah_boxes, np.asarray(donor_boxes, np.float64), equal_nan=True):
        raise SystemExit(f"{args.fah}: boxes {fah_boxes.shape} are not the donor boxes "
                         f"{donor_boxes_path} {np.asarray(donor_boxes).shape} (an exact match is required)")
    cfg = argparse.Namespace(presence=args.presence, side=args.side, residual=args.residual,
                             max_joint_deg=args.max_joint_deg, max_finger_deg=args.max_finger_deg,
                             max_hand_deg=args.max_hand_deg)

    baseline_local = H.rot6d_to_matrix(base_rot6d)
    body_local = H.rot6d_to_matrix(rot6d_placed)                     # placed body -> the parent frames used below
    body_global = H.chain(H.SMPLH_PARENTS, body_local)
    H.check_proper(baseline_local, name="baseline local rotations")

    fused_rot6d = base_rot6d.copy()
    fused_rot6d[:, 0:22] = rot6d_placed[:, 0:22]
    accepted = np.zeros((n, 2), bool)
    applied = np.zeros((n, 2), bool)
    reasons = [[[] for _ in range(2)] for _ in range(n)]
    for index in range(n):
        for side in range(2):
            ok, why = gate(fah, index, side, cfg)
            accepted[index, side] = ok
            reasons[index][side] = why
    gated_before = int(accepted.sum())
    accepted = temporal_policy(accepted, n, min_run=args.min_run)

    change = {side: [] for side in (0, 1)}
    for index in range(n):
        for side in range(2):
            if not accepted[index, side]:
                continue
            local = H.transfer_to_smplh(fah["mano_pose"][index, side], fah["mano_R"][index, side],
                                        body_global[index, H.WRIST[side]], mirror=(side == 0))
            check = H.check_proper(local, name=f"fused hand frame {index} side {side}")
            fused_rot6d[index, H.HAND_SLICES[side]] = H.matrix_to_rot6d(check)
            applied[index, side] = True
            change[side].append(H.angle_between(local, baseline_local[index, H.HAND_SLICES[side]]).ravel())

    fused_local_matrix = H.rot6d_to_matrix(fused_rot6d)
    H.check_proper(fused_local_matrix, name="fused local rotations")
    # Rows that were not replaced must be bit-identical to the baseline.
    untouched = np.ones((n, 52), bool)
    for side in range(2):
        untouched[:, H.HAND_SLICES[side]] = ~applied[:, side][:, None]
    diff = np.abs(fused_local_matrix - baseline_local).max((-1, -2))
    if not np.array_equal(fused_rot6d[:, 0:22], rot6d_placed[:, 0:22]):
        raise AssertionError("the placed body/root rot6d rows changed")
    if np.any(diff[untouched] > 0):
        raise AssertionError("baseline finger rows changed where FAH was not applied")

    motion = dict(placed)
    motion["rot6d"] = fused_rot6d[None].astype(np.float32)
    motion["root_rot6d"] = motion["rot6d"][:, :, 0]
    motion["body_rot6d"] = motion["rot6d"][:, :, 1:]
    motion["source_representation"] = "smplh"
    motion["flow_version"] = int(placed.get("flow_version", 1))
    motion["placement_version"] = int(placed.get("placement_version", 2))
    motion["fah_fingers"] = dict(source=str(Path(args.fah).resolve()), project=str(project),
                                 finger_baseline=finger_source, applied=int(applied.sum()))
    motion.pop("local_joints_positions", None)

    import torch
    temporary = out / "fused_motion.partial.pt"
    try:
        torch.save(motion, temporary)
        temporary.replace(out / "fused_motion.pt")
    finally:
        temporary.unlink(missing_ok=True)

    np.savez_compressed(out / "fused.npz",
                        baseline_local=baseline_local.astype(np.float64), fused_local=fused_local_matrix.astype(np.float64),
                        baseline_rot6d=base_rot6d.astype(np.float32), fused_rot6d=fused_rot6d.astype(np.float32),
                        placed_rot6d=rot6d_placed.astype(np.float32),
                        trans=np.asarray(placed["trans"], np.float32)[0], shapes=np.asarray(placed["shapes"], np.float32).reshape(-1),
                        applied=applied, accepted=accepted, boxes=np.asarray(fah["boxes"], np.float32),
                        mano_pose=np.asarray(fah["mano_pose"], np.float32), mano_R=np.asarray(fah["mano_R"], np.float32),
                        mano_trans=np.asarray(fah["mano_trans"], np.float32), mano_betas=np.asarray(fah["mano_betas"], np.float32),
                        fah_joints3d=np.asarray(fah["joints3d"], np.float32), presence=np.asarray(fah["presence"], np.float32),
                        side_score=np.asarray(fah["side_score"], np.float32),
                        fit_residual_m=np.asarray(fah["fit_residual_m"], np.float32))

    summary = {}
    for side, name in enumerate(("left", "right")):
        values = np.concatenate(change[side]) if change[side] else np.zeros(0)
        summary[name] = dict(frames_applied=int(applied[:, side].sum()), frames_gated=int(accepted[:, side].sum()),
                             frames=int(n),
                             joint_change_deg=(None if not len(values) else
                                               dict(mean=float(values.mean()), p95=float(np.quantile(values, .95)),
                                                    max=float(values.max()),
                                                    per_slot={slot: float(values.reshape(-1, 15)[:, i].mean())
                                                              for i, slot in enumerate(H.HAND_SLOT_NAMES)})))
    rejected = {}
    for side, name in enumerate(("left", "right")):
        tally = {}
        for index in range(n):
            if accepted[index, side] or not reasons[index][side]:
                continue
            for reason in reasons[index][side]:
                tally[reason] = tally.get(reason, 0) + 1
        rejected[name] = tally
    # The two finger baselines are not interchangeable when the run applied HaMeR:
    # `placed` carries the committed result's fingers, `flow_prediction` is FlowHMR's
    # own SMPL-H hand prediction. Report the gap instead of hiding it.
    placed_fingers = H.rot6d_to_matrix(rot6d_placed)[:, 22:52]
    baseline_fingers = baseline_local[:, 22:52]
    gap = np.degrees(np.abs(baseline_fingers - placed_fingers).max((-1, -2)))
    report = dict(project=str(project), fah=str(Path(args.fah).resolve()), frames=n,
                  finger_baseline=finger_source, fah_boxes_donor=str(donor_boxes_path),
                  fah_boxes_exact_match=True,
                  finger_baseline_gap_vs_placed_deg=dict(mean=float(gap.mean()), max=float(gap.max()),
                                                         identical=bool(gap.max() < 1e-9)), policy=dict(presence=args.presence, side=args.side,
                                                             residual_m=args.residual, max_joint_deg=args.max_joint_deg,
                                                             max_finger_deg=args.max_finger_deg,
                                                             max_hand_deg=args.max_hand_deg,
                                                             min_run=args.min_run),
                  gated=gated_before, gated_after_temporal=int(accepted.sum()),
                  dropped_by_temporal=gated_before - int(accepted.sum()),
                  applied=int(applied.sum()),
                  per_side=summary, rejected_for=rejected,
                  unchanged_rows_max_delta=float(diff[untouched].max()))
    if args.rest_report:
        report["rest"] = rest_report(project, fah)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{out}: gated {gated_before}/{2 * n} "
          f"({gated_before - int(accepted.sum())} dropped by the min-run policy), "
          f"applied {applied.sum()}/{2 * n} hands "
          f"(baseline fingers from {finger_source}); report.json, fused.npz, fused_motion.pt")


def rest_report(project, fah):
    """MANO vs SMPL-H rest agreement for this actor (needs the registered SMPL-H).

    The MANO rest joints and kinematic tree come from the FAH run's npz, i.e. from
    the MANO files FAH itself loaded; the tree is checked against `hand_math` so a
    wrong MANO slot order can never slip through silently."""
    import body_fk
    model = body_fk.skeleton()
    body_fk.check_parents(model, H.SMPLH_PARENTS)
    parents = tuple(int(p) for p in np.asarray(fah["mano_parents"]).tolist())
    if parents != H.MANO_PARENTS:
        raise SystemExit(f"MANO kintree in the FAH run is {parents}, hand_math assumes {H.MANO_PARENTS}")
    mano_rest = np.asarray(fah["mano_rest_joints"], np.float64)
    if mano_rest.shape != (16, 3):
        raise SystemExit(f"FAH npz mano_rest_joints has shape {mano_rest.shape}, expected (16, 3)")
    placed = load_pt(project / "hmr4d_results.pt")
    rest = body_fk.rest_joints(model, np.asarray(placed["shapes"], np.float64).reshape(-1))
    result = dict(mano_parents=parents, smplh_parents=H.SMPLH_PARENTS)
    for side, name in enumerate(("left", "right")):
        hand_rest = rest[H.HAND_SLICES[side].start:H.HAND_SLICES[side].stop]
        with_wrist = np.concatenate((rest[H.WRIST[side]][None], hand_rest), 0)   # (16,3), wrist first
        result[name] = H.rest_report(mano_rest, with_wrist, mirror=(side == 0))
    return result


if __name__ == "__main__":
    main()
