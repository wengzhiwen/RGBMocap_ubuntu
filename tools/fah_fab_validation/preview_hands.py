#!/usr/bin/env python3
"""Same-frame preview: FlowHMR baseline fingers vs FAH-fused fingers, one image pair.

    .venv/bin/python tools/fah_fab_validation/preview_hands.py \
        --fused experiments/fah-fab-20261123/run-1/fused.npz \
        --project experiments/fah-fab-20261123/flow-moving-hamer/flowhmr \
        --frames 0,30,60 --out-dir experiments/fah-fab-20261123/run-1/preview

Both skeletons are the *same* committed result: rows 0..21 and the root are
identical by construction, so the picture shows only what the finger substitution
changed. Posed joints come from the registered SMPL-H model through RGB MoCap's
own FlowHMR body model (`body_fk`), projected with the app's own FlowHMR camera:

    p_cam = camera_RT_yup[t,:3,:3] @ scene_basis.T @ (p_world - [0, camera_height, 0])

which is exactly `mocap/flow_placement.py::placement_fit`'s camera convention (and
`mocap/camera_tracks.load`'s fixed-camera branch at t = 0). The reprojected SAM
pelvis anchors from `flow_placement.json` are compared with the anchors recorded in
`flow_world.json`, so a wrong axis/order convention shows up as a centimetre error
instead of a plausible-looking picture.

Blue = baseline (FlowHMR / restored `flow_prediction.pt` fingers), red = fused
(FAH fingers), green crosses = FAH's own 2D joints in original image pixels,
yellow = FlowHMR's hand box.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import body_fk
import hand_math as H

BODY_LINKS = [(int(p), j) for j, p in enumerate(H.SMPLH_PARENTS[:H.WRIST[1] + 1]) if p >= 0]


def hand_links(start):
    links = []
    for finger in range(5):
        base = start + finger * 3
        links.append((H.WRIST[0 if start == 22 else 1], base))
        links += [(base, base + 1), (base + 1, base + 2)]
    return links


def camera_project(donor, frames):
    """Per-frame world -> OpenCV camera transforms and the intrinsics, as the app uses them."""
    world = json.loads((donor / "flow_world.json").read_text(encoding="utf-8"))
    with np.load(donor / "flow_camera.npz", allow_pickle=False) as data:
        rt = np.asarray(data["camera_RT_yup"], np.float64)
    basis = np.asarray(world["scene_basis"], np.float64)
    height = float(world["camera_height"])
    k = np.asarray(world["intrinsics"], np.float64)
    rotation = rt[:frames, :3, :3] @ basis.T
    offset = -np.einsum("tij,j->ti", rotation, [0.0, height, 0.0])
    return rotation, offset, k, world


def anchor_check(donor, joints, rotation, offset, k):
    """Reproject the SAM placement anchors and compare with the recorded residuals.

    `mocap/flow_placement.py` fitted the placement against these anchors, so a wrong
    axis or camera convention shows up here as centimetres rather than as a picture
    that merely looks plausible.
    """
    anchors = np.asarray(json.loads((donor / "flow_placement.json").read_text(encoding="utf-8"))["camera_pelvis"], np.float64)
    recorded = json.loads((donor / "flow_world.json").read_text(encoding="utf-8")).get("anchor_residual_m", {})
    index = anchors[:, 0].astype(int)
    camera = np.einsum("tij,tj->ti", rotation[index], joints[index, 0]) + offset[index]
    residual = np.linalg.norm(camera - anchors[:, 1:], axis=-1)
    return dict(anchor_frames=int(len(anchors)),
                reproject_residual_m=dict(mean=float(residual.mean()), max=float(residual.max())),
                recorded_residual_m=recorded)


def project_joints(points, rotation, offset, k, index):
    """World joints -> OpenCV camera coordinates and image pixels for one frame."""
    camera_points = points[index] @ rotation[index].T + offset[index]
    depth = np.clip(camera_points[:, 2], 1e-3, None)
    return camera_points @ k.T / depth[:, None], camera_points


def draw(image, links, points, color, thickness=2):
    import cv2
    for a, b in links:
        pa, pb = points[a], points[b]
        if not (np.isfinite(pa).all() and np.isfinite(pb).all()):
            continue
        cv2.line(image, (int(round(pa[0])), int(round(pa[1]))), (int(round(pb[0])), int(round(pb[1]))), color, thickness)
    for p in points:
        if np.isfinite(p).all():
            cv2.circle(image, (int(round(p[0])), int(round(p[1]))), 2, color, -1)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fused", required=True, help="fuse_hands.py fused.npz")
    p.add_argument("--project", required=True, help="donor FlowHMR project (read-only, for camera and body model)")
    p.add_argument("--frames", default="0", help="comma separated frame indices")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--video", help="source video for the image background (default: <project>/input.mp4)")
    p.add_argument("--no-image", action="store_true", help="draw on a blank canvas instead of decoding video frames")
    args = p.parse_args(argv)

    fused = Path(args.fused).resolve()
    donor = Path(args.project).resolve()
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    data = np.load(fused, allow_pickle=False)
    frames = int(data["trans"].shape[0])
    wanted = [int(v) for v in str(args.frames).split(",") if v.strip() != ""]
    if any(not 0 <= v < frames for v in wanted):
        raise SystemExit(f"--frames must be within 0..{frames - 1}")

    model = body_fk.skeleton()
    body_fk.check_parents(model, H.SMPLH_PARENTS)
    joints_base = body_fk.joints(model, data["baseline_local"], data["shapes"], data["trans"])
    joints_fused = body_fk.joints(model, data["fused_local"], data["shapes"], data["trans"])
    rotation, offset, k, world = camera_project(donor, frames)
    check = anchor_check(donor, joints_fused, rotation, offset, k)

    import cv2
    cap = None
    if not args.no_image:
        video = Path(args.video) if args.video else donor / "input.mp4"
        if not video.is_file():
            raise SystemExit(f"video not found: {video} (use --no-image for a blank canvas)")
        cap = cv2.VideoCapture(str(video))
    width, height = int(round(2 * k[0, 2])), int(round(2 * k[1, 2]))

    summary = {"camera": dict(anchor_check=check, intrinsics=k.tolist(), camera_height_m=world["camera_height"],
                              fixed_camera=bool(np.allclose(rotation, rotation[0], atol=1e-3))),
               "frames": {}}
    for index in wanted:
        if cap is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, image = cap.read()
            if not ok:
                raise SystemExit(f"cannot decode frame {index}")
        else:
            image = np.full((height, width, 3), 32, np.uint8)
        uv_base, _ = project_joints(joints_base, rotation, offset, k, index)
        uv_fused, _ = project_joints(joints_fused, rotation, offset, k, index)
        panel = image.copy()
        draw(panel, BODY_LINKS, uv_base, (160, 160, 160), 1)
        for side, start in enumerate(H.HAND_START):
            links = hand_links(start)
            draw(panel, links, uv_base, (255, 120, 40), 2)          # blue = baseline
            draw(panel, links, uv_fused, (40, 60, 235), 2)          # red = fused
            box = data["boxes"][index, side]
            cv2.rectangle(panel, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 220, 220), 1)
            fah_joints = np.asarray(data["fah_joints3d"][index, side], np.float64)
            if np.isfinite(fah_joints).all():
                k_fah = np.asarray(json.loads((donor / "flow_placement.json").read_text(encoding="utf-8"))["intrinsics"], np.float64)
                fah_uv = fah_joints @ k_fah.T / np.clip(fah_joints[:, 2:], 1e-3, None)
                for q in fah_uv:
                    cv2.drawMarker(panel, (int(round(q[0])), int(round(q[1]))), (60, 200, 60), cv2.MARKER_CROSS, 8, 1)
            shift = np.linalg.norm(joints_fused[index, start:start + 15] - joints_base[index, start:start + 15], axis=-1)
            cv2.putText(panel, f"f{index} {'L' if side == 0 else 'R'} applied={bool(data['applied'][index, side])} "
                               f"shift_mean={shift.mean() * 1000:.1f}mm max={shift.max() * 1000:.1f}mm",
                        (12, 28 + 22 * side), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            summary["frames"].setdefault(str(index), {})[["left", "right"][side]] = dict(
                applied=bool(data["applied"][index, side]), shift_mm=dict(mean=float(shift.mean() * 1000), max=float(shift.max() * 1000)))
        (out / f"frame_{index:04d}_compare.png").write_bytes(cv2.imencode(".png", panel)[1])
    if cap is not None:
        cap.release()
    (out / "preview_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}: {len(wanted)} frame(s); anchor check {check['reproject_residual_m']}")


if __name__ == "__main__":
    main()
