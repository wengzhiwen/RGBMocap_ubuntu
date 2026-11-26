#!/usr/bin/env python3
"""Run FAH on FlowHMR's exact per-frame left/right hand boxes and fit MANO.

Research only. Reads a FlowHMR project directory read-only and writes one npz.

    .venv-fah/bin/python tools/fah_fab_validation/fah_hands.py \
        --project experiments/fah-fab-20261123/flow-moving-hamer/flowhmr \
        --video experiments/fah-fab-20261123/clip-30fps.mp4 \
        --out experiments/fah-fab-20261123/run-1/fah_hands.npz \
        --fah-dir experiments/fah-fab-20261123/FAH \
        --weights experiments/fah-fab-20261123/weights-fah \
        --mano-dir experiments/fah-fab-20261123/mano \
        --sam3d models/flowhmr/sam

Boxes come from `flow_hand_boxes.npy` (n,2,4) xyxy px, slot 0 = left, 1 = right -
the same boxes RGB MoCap's HaMeR pass consumes. They are handed to FAH's own
`HandEstimator.process_image(..., boxes=..., is_right=...)`, so FAH's detector and
its side classifier never re-decide the crop; FAH's `verify` head logits are still
recorded as `presence` / `side_score` and the fused step gates on them.

MANO parameters come from FAH's official image-mode mesh stage (`MeshStage`:
closed-form per-part Procrustes init + the released learned fitter init + Gauss-
Newton), read back from the stage's `out_R` / `out_x` buffers - the same buffers
`MeshFitter.fit(..., params=True)` returns. FAH fits the right-hand MANO model to
every hand, so each hand is x-reflected on its own (left hand mirrored, right hand
not): a left hand's stored pose / rotation / translation are in the mirrored
right-hand frame (`mirrored_left=True`) and `fuse_hands.py` applies the x-reflection
back. No MANO/SMPL-H hand mean is added anywhere.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


def load_project(project):
    project = Path(project).resolve()
    state = json.loads((project / "project.json").read_text(encoding="utf-8"))
    if state.get("pipeline", {}).get("model") != "flowhmr":
        raise SystemExit(f"{project}: not a FlowHMR project (pipeline.model={state.get('pipeline', {}).get('model')!r})")
    n, width, height, fps = int(state["frames"]), int(state["width"]), int(state["height"]), float(state["fps"])
    boxes = np.load(project / "flow_hand_boxes.npy", allow_pickle=False)
    if boxes.shape != (n, 2, 4):
        raise SystemExit(f"{project}/flow_hand_boxes.npy: expected ({n}, 2, 4), got {boxes.shape}")
    # FlowHMR conditions its body/hand features on focal = image diagonal; FAH's
    # default intrinsics are the same, so both passes share one camera model.
    focal = float(np.hypot(width, height))
    K = np.array([[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]], np.float32)
    return project, dict(frames=n, width=width, height=height, fps=fps), boxes, K


def box_ok(box, width, height, min_side=2.0, min_clip=4.0, min_inside=0.1):
    """The same crop sanity RGB MoCap's HaMeR pass uses."""
    b = np.asarray(box, np.float64)
    if not np.isfinite(b).all():
        return False
    size = b[2:] - b[:2]
    if size.min() <= min_side:
        return False
    clipped = np.maximum(0.0, np.minimum(b[2:], [width, height]) - np.maximum(b[:2], 0.0))
    if clipped.min() < min_clip:
        return False
    return clipped.prod() / max(size.prod(), 1.0) >= min_inside


# One reflection per hand: FAH's mesh stage only ever runs the right-hand MANO
# model, so the left hand's targets/vertices carry the x-reflection
# (bench/methods/fah.py `_fit_now` mirrors per hand). Never share one mirror across
# a two-hand batch.
MIRROR_LEFT = np.array([-1.0, 1.0, 1.0], np.float64)
MIRROR_RIGHT = np.array([1.0, 1.0, 1.0], np.float64)


def verify_shapes(arrays, n):
    """End-to-end shape guard for the record that is about to be written."""
    expected = {"mano_R": (n, 2, 3, 3), "mano_pose": (n, 2, 45), "mano_betas": (n, 2, 10),
                "mano_trans": (n, 2, 3), "score": (n, 2), "presence": (n, 2), "side_score": (n, 2),
                "fit_residual_m": (n, 2), "joints3d": (n, 2, 21, 3), "joints2d": (n, 2, 21, 2),
                "boxes": (n, 2, 4), "accepted": (n, 2), "fitted": (n, 2), "verts3d": (n, 2, 778, 3)}
    bad = [(k, arrays[k].shape, want) for k, want in expected.items() if k in arrays and arrays[k].shape != want]
    missing = [k for k in expected if k not in arrays and k != "verts3d"]
    if bad or missing:
        raise SystemExit("output shape guard failed: "
                         + ", ".join(f"{k} {got} != {want}" for k, got, want in bad)
                         + ("; missing " + ",".join(missing) if missing else ""))


def build_fah(args):
    sys.path.insert(0, str(Path(args.fah_dir).resolve()))
    os.environ["MANO_DIR"] = str(Path(args.mano_dir).resolve())
    import fah
    from bench.methods.fah import FAHMethod

    method_args = fah.method_args(args.model, "image", weights=str(Path(args.weights).resolve()),
                                  device=args.device, sam3d=str(Path(args.sam3d).resolve()))
    method = FAHMethod(**method_args)
    return method, method_args


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project", required=True, help="FlowHMR project directory (read-only)")
    p.add_argument("--video", help="video to decode (default: <project>/input.mp4, used directly - it must be "
                                   "the same input the project was built from: fps and frame shape are checked)")
    p.add_argument("--out", required=True, help="output npz (research directory); an existing file is reused, "
                                                "never overwritten (pass --force to replace it)")
    p.add_argument("--force", action="store_true", help="overwrite --out if it already exists")
    p.add_argument("--fah-dir", default="experiments/fah-fab-20261123/FAH")
    p.add_argument("--weights", default="experiments/fah-fab-20261123/weights-fah")
    p.add_argument("--mano-dir", default="experiments/fah-fab-20261123/mano")
    p.add_argument("--sam3d", default="models/flowhmr/sam")
    p.add_argument("--model", default="fah", choices=["fah-s", "fah-b", "fah", "fah-h"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="process at most N frames (smoke runs)")
    p.add_argument("--save-verts", action="store_true", help="also store FAH's fitted MANO mesh (778,3) per hand")
    p.add_argument("--no-fit", action="store_true", help="skip the MANO stage (FAH points only)")
    p.add_argument("--allow-fps-mismatch", action="store_true",
                   help="warn instead of stop when the decoded video fps disagrees with the project manifest")
    args = p.parse_args(argv)

    project, meta, hand_boxes, K = load_project(args.project)
    video = Path(args.video) if args.video else project / "input.mp4"
    if not video.is_file():
        raise SystemExit(f"video not found: {video}")
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not args.force:
        raise SystemExit(f"{out} already exists - use it directly, or pass --force to re-run over it")

    n = meta["frames"]
    first, last = args.start, n if not args.limit else min(n, args.start + args.limit)

    pose = np.zeros((n, 2, 45), np.float32)
    betas = np.zeros((n, 2, 10), np.float32)
    root = np.tile(np.eye(3, dtype=np.float32), (n, 2, 1, 1))          # -> (n,2,3,3), checked by verify_shapes
    trans = np.zeros((n, 2, 3), np.float32)
    score = np.full((n, 2), np.nan, np.float32)
    presence = np.full((n, 2), np.nan, np.float32)
    side_score = np.full((n, 2), np.nan, np.float32)
    residual = np.full((n, 2), np.nan, np.float32)
    joints3d = np.full((n, 2, 21, 3), np.nan, np.float32)
    joints2d = np.full((n, 2, 21, 2), np.nan, np.float32)
    accepted = np.zeros((n, 2), bool)
    fitted = np.zeros((n, 2), bool)
    verts = np.zeros((n, 2, 778, 3), np.float32) if args.save_verts else None

    method, method_args = build_fah(args)
    import cv2
    import torch
    from fah.fast_mano import FastMano, MeshStage
    from fah.utils import pick_device

    est = method.est
    stage = None if args.no_fit else MeshStage(device=method_args.get("mesh_fit_device") or method_args["device"],
                                               iters=method_args.get("mesh_fit", 1), wv=method_args.get("mesh_fit_wv", 0.3),
                                               lmf=method_args.get("mesh_fit_lmf") or None,
                                               stride=method_args.get("mesh_fit_stride", 1),
                                               jac=method_args.get("mesh_fit_jac", "ad"),
                                               centre=method_args.get("mesh_fit_centre", False),
                                               fd_eps=method_args.get("mesh_fit_fd_eps", 1e-3))
    fast = None if args.no_fit else FastMano(str(Path(args.mano_dir).resolve()),
                                             device=method_args.get("mesh_fit_device") or method_args["device"])
    import pickle
    with open(Path(args.mano_dir).resolve() / "MANO_RIGHT.pkl", "rb") as stream:
        mano_data = pickle.load(stream, encoding="latin1")
    dense = lambda x: np.asarray(x.todense() if hasattr(x, "todense") else x, np.float64)
    mano_rest = dense(mano_data["J_regressor"]) @ dense(mano_data["v_template"])   # (16,3) mean-shape rest joints
    mano_parents = np.asarray(mano_data["kintree_table"][0], np.int64).copy()
    mano_parents[0] = -1
    print(f"MANO rest joints and kintree recorded (parents {mano_parents.tolist()})", flush=True)
    print(f"FAH {args.model} ready on {pick_device(args.device)}; mesh_fit={method_args.get('mesh_fit')} "
          f"jac={method_args.get('mesh_fit_jac')} lmf={'on' if method_args.get('mesh_fit_lmf') else 'off'}", flush=True)

    cap = cv2.VideoCapture(str(video))
    video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if abs(video_fps - meta["fps"]) > max(0.5, 0.02 * meta["fps"]):
        message = (f"{video} decodes at {video_fps:g} fps but {project} declares {meta['fps']:g} fps - run FAH "
                   f"on the same input the project was built from")
        if not args.allow_fps_mismatch:
            raise SystemExit(message)
        print(f"warning: {message}", flush=True)
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    started = time.time()
    regress_max = 0.0
    for index in range(first, last):
        ok, image = cap.read()
        if not ok:
            raise SystemExit(f"cannot decode frame {index + 1} of {video}")
        if image.shape != (meta["height"], meta["width"], 3):
            raise SystemExit(f"{video} frame {index + 1} decodes to {image.shape} but {project} declares "
                             f"({meta['height']}, {meta['width']}, 3) - the hand boxes would not line up")
        sides = [side for side in (0, 1) if box_ok(hand_boxes[index, side], meta["width"], meta["height"])]
        accepted[index, sides] = True
        if not sides:
            continue
        boxes = np.asarray([hand_boxes[index, s] for s in sides], np.float32)
        is_right = np.array([bool(s) for s in sides])          # slot 0 = left -> False
        hands = est.process_image(image, boxes=boxes, is_right=is_right, cam_int=K)
        for offset, (side, hand) in enumerate(zip(sides, hands)):
            score[index, side] = float(hand.get("score", np.nan))   # 1.0: the box was forced, not detected
            presence[index, side] = float(hand.get("presence", np.nan)) if hand.get("presence") is not None else np.nan
            side_score[index, side] = float(hand.get("side_score", np.nan)) if hand.get("side_score") is not None else np.nan
            joints3d[index, side] = np.asarray(hand["keypoints_3d"], np.float32)
            joints2d[index, side] = np.asarray(hand["keypoints_2d"], np.float32)
        if stage is None:
            continue
        # FAH's released models are decoder-free (`kp_only`): the direct keypoint
        # head gives the MANO-topology dense surface as `vertices_mano`, which is
        # exactly the target FAHMethod._fit_meshes feeds to the mesh stage.
        if any(len(np.asarray(h.get("vertices_mano"), np.float64)) != 778 for h in hands):
            print(f"frame {index + 1}: FAH returned no full 778-vertex surface; MANO fit skipped", flush=True)
            continue
        # Per-hand targets and reflection: left hand x-reflected, right hand not, as a
        # (batch,1,3) broadcast over vertices/joints. A single mirror for the batch is
        # wrong as soon as both hands are fitted together.
        target_v = np.stack([np.asarray(h["vertices_mano"], np.float64) for h in hands])
        target_j = np.stack([np.asarray(h.get("keypoints_3d_mano", h["keypoints_3d"]), np.float64) for h in hands])
        mirrors = np.array([MIRROR_LEFT if s == 0 else MIRROR_RIGHT for s in sides], np.float64)[:, None, :]
        usable = (np.isfinite(target_v).reshape(len(sides), -1).all(axis=1)
                  & np.isfinite(target_j).reshape(len(sides), -1).all(axis=1))
        if not usable.any():
            print(f"frame {index + 1}: FAH surface/keypoints are not finite for either hand; MANO fit skipped", flush=True)
            continue
        if not usable.all():
            print(f"frame {index + 1}: FAH targets not finite for side(s) "
                  f"{[s for s, keep in zip(sides, usable) if not keep]}; fitting the other hand only", flush=True)
        mirrors = mirrors[usable]
        target_v, target_j = target_v[usable] * mirrors, target_j[usable] * mirrors
        fit_sides = [s for s, keep in zip(sides, usable) if keep]
        meshes = np.asarray(stage(target_v, target_j), np.float32)
        batch = len(fit_sides)
        buffers = stage.graphs[batch][0]["fs"]
        params, r_fit = buffers["out_x"], buffers["out_R"]
        with torch.no_grad():
            out_v, out_j = fast(params[:, :45], params[:, 45:55])
        # `out_x[:, 58:61]` is a torch (batch,3) buffer, so the centre-mode offset is
        # each hand's wrist as (batch,3) on that device/dtype - the (batch,1,3) target
        # slice would broadcast into a (batch,batch,3) tensor.
        centre = target_j[:, 0] if stage.centre else np.zeros((batch, 3), np.float64)
        offset_world = params[:, 58:61] + torch.as_tensor(centre, dtype=params.dtype, device=params.device)
        mesh_check = (out_v @ r_fit.mT + offset_world[:, None, :]).cpu().numpy()
        joint_check = (out_j @ r_fit.mT + offset_world[:, None, :]).cpu().numpy()
        for offset, side in enumerate(fit_sides):
            residual_m = float(np.linalg.norm(joint_check[offset] - target_j[offset], axis=-1).mean())
            solved = np.concatenate([params[offset].cpu().numpy().ravel(), r_fit[offset].cpu().numpy().ravel(),
                                     offset_world[offset].cpu().numpy(), joint_check[offset].ravel()])
            if not (np.isfinite(solved).all() and np.isfinite(residual_m) and np.isfinite(mesh_check[offset]).all()):
                print(f"frame {index + 1} side {side}: MANO stage returned non-finite parameters; this hand keeps "
                      f"its baseline record", flush=True)
                continue
            diff = float(np.abs(mesh_check[offset] - meshes[offset]).max())
            if np.isfinite(diff):
                regress_max = max(regress_max, diff)
            pose[index, side] = params[offset, :45].cpu().numpy().astype(np.float32)
            betas[index, side] = params[offset, 45:55].cpu().numpy().astype(np.float32)
            root[index, side] = r_fit[offset].cpu().numpy().astype(np.float32)
            trans[index, side] = offset_world[offset].cpu().numpy().astype(np.float32)
            residual[index, side] = residual_m
            fitted[index, side] = True
            if verts is not None:
                verts[index, side] = (meshes[offset] * mirrors[offset]).astype(np.float32)
        if (index - first) % 10 == 0 or index + 1 == last:
            print(f"frame {index + 1}/{last} in {time.time() - started:.1f} s", flush=True)
    if last == n and cap.read()[0]:
        print(f"warning: {video} holds more frames than the project manifest ({n})", flush=True)
    cap.release()

    meta_out = dict(version=1, kind="fah-hands-on-flowhmr-boxes", frames=n, processed=[first, last],
                    kp_only=bool(getattr(est.model, "kp_only", False)),
                    width=meta["width"], height=meta["height"], fps=meta["fps"], K=K.tolist(),
                    fah_model=args.model, fah_args={k: v for k, v in method_args.items() if isinstance(v, (int, float, str, bool))},
                    mano_dir=str(Path(args.mano_dir).resolve()), mirrored_left=True, video_fps=video_fps,
                    frame_shape=[meta["height"], meta["width"], 3],
                    hand_mean_added=False, joint_order="OpenPose hand (wrist, thumb1-4, index1-4, middle1-4, ring1-4, pinky1-4)",
                    fitter="FAH MeshStage (Procrustes init + released mesh_fitter.pt + Gauss-Newton), params from out_R/out_x",
                    fit_params="mano_R (n,2,3,3) + mano_pose (45 flat-hand axis-angle, MANO/SMPL-H slot order) + mano_betas (10) + mano_trans (m)",
                    fit_targets="FAH kp_only keypoint head: vertices_mano (778,3 MANO topology) + keypoints_3d_mano (21,3 OpenPose order)",
                    handedness="flow_hand_boxes slot 0 = left hand, slot 1 = right hand; FAH side_score confirms this on this clip",
                    regression_max_m=regress_max, torch=[torch.__version__],
                    project=str(project), video=str(video))
    arrays = dict(mano_R=root, mano_pose=pose, mano_betas=betas, mano_trans=trans, score=score, presence=presence,
                  side_score=side_score, fit_residual_m=residual, joints3d=joints3d, joints2d=joints2d,
                  boxes=hand_boxes.astype(np.float32), accepted=accepted, fitted=fitted,
                  mano_rest_joints=mano_rest.astype(np.float32), mano_parents=mano_parents)
    if verts is not None:
        arrays["verts3d"] = verts
    verify_shapes(arrays, n)
    temporary = out.with_name(out.name + ".partial.npz")
    try:
        np.savez(temporary, meta=np.array(json.dumps(meta_out)), **arrays)
        os.replace(temporary, out)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"wrote {out}: accepted {accepted.sum()}/{2 * n} hand boxes, fitted {fitted.sum()}, "
          f"MANO stage params regression {regress_max:.2e} m")


if __name__ == "__main__":
    main()
