# FAH / FAB research validation tools

Research-only tooling for the FAH (hand fingers) and FAB (body) comparison inside
`experiments/fah-fab-20261123/`. Nothing here is wired into the GUI, the installer or
any production pipeline, and nothing here writes outside the research directory you
pass to it. Production code under `mocap/` and `third_party/FlowHMR/` is **imported,
never modified**; the donor FlowHMR project is opened read-only.

| file | what it does |
| --- | --- |
| `fah_hands.py` | Runs FAH on FlowHMR's **own per-frame hand boxes** and records MANO parameters, scores and joints. |
| `hand_math.py` | rot6d packing, kinematic-chain helpers, left-hand mirroring, MANO→SMPL-H finger transfer, MANO/SMPL-H rest comparison, joint tables. |
| `body_fk.py` | Posed SMPL-H joints through RGB MoCap's own FlowHMR body model (`mocap.models` + `flowhmr`), used for previews and gate checks. |
| `fuse_hands.py` | Gating + fusion into FlowHMR's own motion schema: `fused.npz`, `fused_motion.pt`, `report.json`. |
| `preview_hands.py` | Same-frame still comparison (baseline vs fused) drawn on the source frame. |
| `hands_video.py` | Three-column comparison video: plain FlowHMR / FlowHMR+HaMeR / FlowHMR+FAH over the same decoded frames. |
| `make_take.py` | Builds an **independent research take** and exports BVH/FBX by calling the app's own export code. |
| `fab_body.py` | CPU processing of a `fab-video` body result plus a FlowHMR overlay (axes, scale, trajectory, foot-drift evidence). |
| `test_hand_math.py` | Unit regressions for all of the above math (no GPU, no inference). |

## FAH path — commands

The GPU step is run by the operator; the rest is CPU.

```bash
ROOT=$PWD
EXP=$ROOT/experiments/fah-fab-20261123
DONOR=$EXP/flow-moving-hamer/flowhmr           # finished FlowHMR project (read-only)
RUN=$EXP/run-1                                 # new directory for a new experiment
mkdir -p $RUN

# 1. FAH inference on the donor's exact boxes (180,2,4) xyxy px, slot 0 = left, 1 = right
$EXP/.venv-fah/bin/python tools/fah_fab_validation/fah_hands.py \
    --project $DONOR --video $EXP/clip-30fps.mp4 --out $RUN/fah.npz --device cuda

# 2. Fuse: replace only SMPL-H rows 22..51, keep everything else bit-exact
.venv/bin/python tools/fah_fab_validation/fuse_hands.py \
    --project $DONOR --fah $RUN/fah.npz --out-dir $RUN --rest-report

# 3. Same-frame comparison (stills, then the three-column video)
.venv/bin/python tools/fah_fab_validation/preview_hands.py \
    --fused $RUN/fused.npz --project $DONOR --frames 0,60,120 --out-dir $RUN/preview
.venv/bin/python tools/fah_fab_validation/hands_video.py \
    --fused $RUN/fused.npz --project $DONOR --out $RUN/hands_compare.mp4

# 4. Independent take + the app's own BVH/FBX export
.venv/bin/python tools/fah_fab_validation/make_take.py \
    --fused $RUN/fused_motion.pt --out-dir $EXP/takes/fah-fingers \
    --donor $DONOR --fused-npz $RUN/fused.npz --report $RUN/report.json --fbx

# 5. Unit regressions
.venv/bin/python tools/fah_fab_validation/test_hand_math.py $DONOR \
    --fused $RUN/fused.npz --fah $RUN/fah.npz
```

`fuse_hands.py` gates each hand independently and falls back to the baseline fingers
with an explicit reason code (`box`, `no-fit`, `presence<…`, `side<…`, `fit>…mm`,
`joint>…deg`, `finger>…deg`, `hand>…deg`); runs shorter than `--min-run` are dropped
and reported separately (`gated`, `gated_after_temporal`, `dropped_by_temporal`).

## Data contracts

`fah_hands.py` npz (one entry per frame × side, side 0 = left): `mano_R (n,2,3,3)`
MANO global root, `mano_pose (n,2,45)`, `mano_betas (n,2,10)`, `mano_trans (n,2,3)`,
`presence`, `side_score`, `fit_residual_m`, `joints3d (n,2,21,3)` (original OpenCV
camera frame, FAH's own intrinsics), `joints2d`, `boxes`, `accepted`, `fitted`, plus
`mano_rest_joints (16,3)` / `mano_parents` recorded from the MANO files FAH actually
loaded. Left-hand MANO parameters are stored **in the mirrored right-hand frame**
(FAH fits a right MANO to the mirrored left crop).

`fused.npz`: `baseline_local`, `fused_local` (n,52,3,3), `baseline_rot6d`,
`fused_rot6d`, `placed_rot6d` (n,52,6), `trans`, `shapes`, `applied`, `accepted`,
`boxes`, the MANO arrays and the gate scores. `fused_motion.pt` keeps FlowHMR's own
schema (`rot6d (1,n,52,6)`, `trans`, `shapes`, `root_rot6d`, `body_rot6d`,
`scene_offset`, `source_representation="smplh"`, `flow_version`, `placement_version`)
so the export path below works without touching production code.

The take directory written by `make_take.py` contains its own `project.json`,
`hmr4d_results.pt` (the fused tensor), `initial_sources.json` (this take's real
single input), provenance copies and the app's export artifacts. The donor's
`flowhmr_receipt.json`, `flow_world.json`, `flow_tokens.pt` and camera caches are
deliberately **not** copied, so the donor project's identity is never claimed.

## Invariants that are enforced, not assumed

- Rows 0..21 of `rot6d`, plus `trans` and `shapes`, are copied bit-exactly from the
  donor result; `fuse_hands.py` aborts if they change.
- Finger rows where FAH was not applied stay bit-identical to the baseline
  (`report.json: unchanged_rows_max_delta` is 0.0).
- Every stored rotation is checked proper (`hand_math.check_proper`).
- MANO's kinematic tree is taken from FAH's MANO files and compared with the SMPL-H
  hand tree; a wrong slot order stops the run.
- `make_take.py --donor` re-checks, against the donor, that only rows 22..51 differ.
- Previews and videos reproject the SAM placement anchors and compare with the
  residuals recorded in `flow_world.json`, so an axis/camera-convention mistake shows
  up as centimetres instead of as a plausible-looking picture.

## Transfer model

- rot6d is FlowHMR's packing: `rotation_matrix[..., :, :2].reshape(..., 6)` (the
  first two **columns**, stored row-wise), with Gram-Schmidt reconstruction and the
  third column as `b1 × b2`. `test_hand_math.py` compares it bit-for-bit with
  `flowhmr.core.math.geometry`.
- Fingers only: FAH's MANO slot frames are aligned to the baseline SMPL-H wrist
  (`A = Gb · Gm0ᵀ`) and each SMPL-H hand slot takes FAH's wrist-relative orientation;
  the baseline wrist, root and body are never touched. Because MANO's tree equals the
  SMPL-H hand tree, the finger-base local rotation reduces to `Gm0ᵀ · Gm[slot]`.
- Left hand: `R_left = M · R_right_model · M`, `t_left = M · t_right_model`,
  `M = diag(-1,1,1)` (conjugation, so products stay proper rotations).
- No `hands_mean` / `smplh_mean` correction is added anywhere in this path: FlowHMR
  and RGB MoCap store direct SMPL-H joint rotations and FAH's fitter does not add
  MANO's mean pose.
- Measured on this actor (donor shapes, `--rest-report`): MANO↔SMPL-H hand joint
  correspondence is the identity within 3.4 mm (right) / 5.2 mm (left, mirrored), and
  rest bone directions differ by at most 7.8° (right) / 13.4° (left) — worst slot
  `thumb1`, distal joints ≤ 4°. That residual rest offset is the only place a MANO
  rotation is not already an SMPL-H rotation, and it is reported, not corrected.

## Finger baselines

The donor project's committed result carries **HaMeR-refined** fingers: rows 22..51 of
`hmr4d_results.pt` equal `flow_hamer_prediction.pt` (only row 0, the root, differs —
placement), and they differ from `flow_prediction.pt` by up to 0.947 rad. So:

- `--fingers flow_prediction` (default) = pure FlowHMR SMPL-H fingers, the FAH comparison baseline;
- `--fingers placed` = the committed (HaMeR) fingers actually exported by the app.

`report.json` records `finger_baseline` and `finger_baseline_gap_vs_placed_deg`.

## Verification record, 2026-11-23 (what was actually run here)

- `fah_hands.py --limit 3 --device cpu`: 6/360 boxes accepted, 6 MANO fits, FAH
  `MeshStage` parameter regression against the official fitter `2.38e-07 m`.
- `fuse_hands.py` on that smoke record: gated 3, applied 3 (right hand; the left hand
  fell back on `presence<0.5`), `unchanged_rows_max_delta = 0.0`.
- `preview_hands.py`: 3 comparison PNGs; reprojected anchor residual matched
  `flow_world.json` to 1e-8 (mean 0.09255493 vs recorded 0.09255496).
- `make_take.py --donor … --fbx`: `motion.bvh` 380,401 B and FBX 412,876 B written by
  `mocap.flow_export.prepare` and `mocap.fbx_export.run` unchanged; donor identity
  check reported body rows bit-identical and changed rows exactly 22..51.
- `test_hand_math.py`: 16 cases pass — rot6d vs upstream, chain/root-frame semantics,
  chain mirror consistency, transfer invariants, joint-table agreement, rest-report
  sensitivity, gate reason codes, temporal policy (including input immutability),
  `mocap.flow_backend.validate_motion` schema acceptance, donor camera convention,
  fused-output invariants and MANO tree/rest agreement.
- Not run here: any full-clip GPU FAH run (only the operator's single-frame CUDA
  smoke), any FAB run, and any quality judgment about FAH or FAB output.

## Open points

- Gate thresholds are starting values; on this clip FAH's left-hand presence stayed
  below 0.5 in the smoke frames, so left hands currently fall back. Per-side
  thresholds are not tuned and nothing is ranked.
- FAB comparison evidence comes from `fab_body.py` (FAB pickle at
  `experiments/fah-fab-20261123/fab-l-fixed.pkl`, SMPL model at
  `experiments/fah-fab-20261123/fab-home/body_models/smpl/SMPL_NEUTRAL.pkl`); it is
  descriptive (axes, scale, trajectory, foot drift) and makes no accuracy claim.
