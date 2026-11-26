#!/usr/bin/env python3
"""Side-by-side hand comparison video: FlowHMR / FlowHMR + HaMeR / FlowHMR + FAH.

    .venv/bin/python tools/fah_fab_validation/hands_video.py \
        --fused experiments/fah-fab-20261123/hands-result/fused.npz \
        --project experiments/fah-fab-20261123/flow-moving-hamer/flowhmr \
        --out experiments/fah-fab-20261123/hands-result/hands_compare.mp4

One output frame = three columns over the **same decoded source frame**, each with
its own left/right hand zoom inset underneath:

| column | finger rotations (SMPL-H rows 22..51) | `fused.npz` key |
| --- | --- | --- |
| 1 | plain FlowHMR (the unmodified `flow_prediction.pt` slices) | `baseline_local` |
| 2 | FlowHMR with the HaMeR-refined fingers that were placed | `placed_rot6d` |
| 3 | FlowHMR with FAH's MANO fingers where the gates passed | `fused_local` |

Rows 0..21, `trans` and `shapes` are shared by all three columns (`fuse_hands.py`
copies them through bit-exactly), so the body overlay and root trajectory are
identical everywhere and only the 15 left-hand + 15 right-hand slots change. That
is re-checked here: any non-zero difference in rows 0..21 stops the run instead of
drawing a misleading comparison.

No checkpoint, model inference or `.pt` motion file is loaded: every number comes
from the fused record plus the donor project's camera. The overlay uses the real
registered SMPL-H model through `body_fk` (RGB MoCap's own FlowHMR body model) and
the app's own placement camera, exactly as `preview_hands.py` does:

    p_cam = (p_world - [0, camera_height, 0]) @ (camera_RT_yup[t, :3, :3] @ scene_basis.T).T
    pixel = K @ p_cam / p_cam.z              with K = flow_world.json["intrinsics"]

`K` belongs to the donor's native pixel size, so joints are projected in the
original decoded pixels and only then multiplied by the panel scale (equivalently
`K` is rescaled); nothing is ever projected in resized pixels. The
`flow_placement.json` pelvis anchors are re-projected as a registration check, so a
wrong axis or order shows up as centimetres instead of as a picture that merely
looks plausible. That is a sanity check, not a quality rating: the video only shows
what differs, and the on-screen text states which pipeline produced each column and
which hand fell back.

Invalid hand boxes, joints behind the camera and undecodable projections never
raise: that tile falls back to a clean wrist-relative drawing and says so. The MP4
is written through FFmpeg (`libx264`, `yuv420p`, `+faststart`) to a partial file
that is only renamed after the `ftyp`/`moov` boxes and `ffprobe` are verified, the
inputs are refused as an output target and re-statted at the end to prove nothing
was touched, and `--out` inside a user-owned folder is rejected.

Temporary research renderer next to `preview_hands.py`, which is left untouched.
Nothing here is wired into the GUI, installer, projects or receipts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np

import body_fk                       # also puts the repository root on sys.path
import hand_math as H
from fah_hands import box_ok         # the crop sanity RGB MoCap's HaMeR pass uses

ROOT = Path(__file__).resolve().parents[2]
COLUMNS = (("1  FlowHMR", "baseline_local", "FlowHMR", (0, 190, 255)),
           ("2  FlowHMR + HaMeR", "placed_rot6d", "HaMeR", (255, 150, 60)),
           ("3  FlowHMR + FAH", "fused_local", "FAH", (60, 60, 235)))
BODY_GRAY, BOX_YELLOW, TEXT, DIM, BG = (130, 130, 130), (0, 215, 215), (235, 235, 235), (185, 185, 185), (22, 22, 22)
HEADER, GAP, LABEL, FOOTER, PAD = 42, 10, 20, 40, 176
MIN_ZOOM, MAX_ZOOM = 0.2, 24.0
PROTECTED = ("projects", "models", "cache", "character_scenes", "runtime", "releases", "build")

BODY_LINKS = tuple((int(p), j) for j, p in enumerate(H.SMPLH_PARENTS[:H.WRIST[1] + 1]) if p >= 0)
HAND_ROWS = tuple(tuple([H.WRIST[side]] + list(range(start, start + 15))) for side, start in enumerate(H.HAND_START))
HAND_LINKS_LOCAL = tuple([(0, 1 + 3 * f) for f in range(5)] + [(1 + 3 * f, 2 + 3 * f) for f in range(5)]
                         + [(2 + 3 * f, 3 + 3 * f) for f in range(5)])


def hand_links(side):
    """SMPL-H row pairs of one hand (wrist -> finger base -> two more joints)."""
    rows = HAND_ROWS[side]
    return tuple((rows[0], rows[1 + 3 * f]) for f in range(5)) \
        + tuple((rows[1 + 3 * f], rows[2 + 3 * f]) for f in range(5)) \
        + tuple((rows[2 + 3 * f], rows[3 + 3 * f]) for f in range(5))


# ----------------------------------------------------------------------- inputs
def identity(path):
    """Read-only fingerprint of an input (hash only for small files)."""
    stat = Path(path).stat()
    record = dict(path=str(Path(path).resolve()), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    if stat.st_size <= 32 * 1024 * 1024:
        record["sha256"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return record


def load_record(path):
    """The three column poses plus the per-hand decision and box record."""
    data = np.load(Path(path), allow_pickle=False)
    missing = [key for key in ("baseline_local", "fused_local", "placed_rot6d", "trans", "shapes",
                               "applied", "accepted", "boxes") if key not in data.files]
    if missing:
        raise SystemExit(f"{path}: missing keys {missing}")
    trans = np.asarray(data["trans"], np.float64)
    shapes = np.asarray(data["shapes"], np.float64).reshape(-1)
    frames = int(trans.shape[0])
    if trans.shape != (frames, 3) or not np.isfinite(trans).all() or shapes.size != 16:
        raise SystemExit(f"{path}: bad trans {trans.shape} or shapes {shapes.shape}")
    poses = (np.asarray(data["baseline_local"], np.float64),
             H.rot6d_to_matrix(np.asarray(data["placed_rot6d"], np.float64)),
             np.asarray(data["fused_local"], np.float64))
    for pose, (title, _, _, _) in zip(poses, COLUMNS):
        H.frames(pose, (frames, 52, 3, 3), f"{title} local rotations")
        H.check_proper(pose, name=f"{title} local rotations")
    body_delta = max(float(np.abs(pose[:, 0:22] - poses[1][:, 0:22]).max()) for pose in poses)
    if body_delta > 0.0:
        raise SystemExit(f"{path}: rows 0..21 differ between the columns by {body_delta}; the "
                         "comparison requires identical body, root, trans and shapes")
    return dict(frames=frames, poses=poses, trans=trans, shapes=shapes,
                applied=np.asarray(data["applied"], bool).reshape(frames, 2),
                accepted=np.asarray(data["accepted"], bool).reshape(frames, 2),
                boxes=np.asarray(data["boxes"], np.float64).reshape(frames, 2, 4),
                body_delta=body_delta)


def finger_deltas(poses):
    """Finger-slot angle differences per frame (what changed, not how good it is)."""
    out = {}
    for first, second, name in ((0, 1, "hamer_vs_flowhmr"), (0, 2, "fah_vs_flowhmr"), (1, 2, "fah_vs_hamer")):
        slots = H.angle_between(poses[second][:, 22:].reshape(-1, 3, 3),
                                poses[first][:, 22:].reshape(-1, 3, 3)).reshape(-1, 30)
        out[name] = dict(per_frame_mean_deg=slots.mean(-1),
                         stats=dict(mean_deg=float(slots.mean()), max_deg=float(slots.max()),
                                    frames_changed=int((slots.max(-1) > 0.5).sum())))
    return out


def camera_maps(project, frames):
    """Per-frame scene world -> OpenCV camera transform, plus the placement intrinsics."""
    world = json.loads((project / "flow_world.json").read_text(encoding="utf-8"))
    with np.load(project / "flow_camera.npz", allow_pickle=False) as data:
        rt = np.asarray(data["camera_RT_yup"], np.float64)
    if rt.shape[0] < frames:
        raise SystemExit(f"{project}/flow_camera.npz: {rt.shape[0]} frames, the record has {frames}")
    basis = np.asarray(world["scene_basis"], np.float64)
    if basis.shape != (3, 3) or "camera_height" not in world:
        raise SystemExit(f"{project}/flow_world.json: bad scene_basis {basis.shape} or missing camera_height")
    rotation = rt[:frames, :3, :3] @ basis.T
    offset = -np.einsum("tij,j->ti", rotation, np.array([0.0, float(world["camera_height"]), 0.0]))
    return rotation, offset, np.asarray(world["intrinsics"], np.float64), world


def match_intrinsics(k, width, height):
    """Rescale `K` when the donor's pixel size differs from the decoded frame size."""
    cx, cy = float(k[0, 2]), float(k[1, 2])
    if cx <= 0 or cy <= 0:
        raise SystemExit(f"unusable intrinsics principal point {cx}, {cy}")
    scale_x, scale_y = width / (2.0 * cx), height / (2.0 * cy)
    if abs(scale_x - 1.0) < 1e-6 and abs(scale_y - 1.0) < 1e-6:
        return k, dict(scale_x=1.0, scale_y=1.0, native_pixels=[2.0 * cx, 2.0 * cy])
    scaled = k.copy()
    scaled[0, 0] *= scale_x
    scaled[1, 1] *= scale_y
    scaled[0, 2] *= scale_x
    scaled[1, 2] *= scale_y
    return scaled, dict(scale_x=scale_x, scale_y=scale_y, native_pixels=[2.0 * cx, 2.0 * cy])


def project_points(points, rotation, offset, k):
    """(n,52,3) scene world joints -> original-frame pixels + camera depth (NaN when invalid)."""
    camera = np.einsum("tpj,tij->tpi", points, rotation) + offset[:, None, :]
    depth = camera[..., 2].astype(np.float64)
    uv = (camera @ k.T)[..., :2] / np.clip(depth, 1e-3, None)[..., None]
    uv[~np.isfinite(uv).all(-1) | (depth <= 1e-3)] = np.nan
    return uv.astype(np.float64), depth


def registration_check(project, joints, rotation, offset):
    """Re-project the SAM placement anchors; a wrong convention shows up in centimetres."""
    placement = project / "flow_placement.json"
    if not placement.is_file():
        return dict(status="flow_placement.json not found")
    anchors = np.asarray(json.loads(placement.read_text(encoding="utf-8"))["camera_pelvis"], np.float64)
    index = anchors[:, 0].astype(int)
    camera = np.einsum("tij,tj->ti", rotation[index], joints[index, 0]) + offset[index]
    residual = np.linalg.norm(camera - anchors[:, 1:], axis=-1)
    recorded = json.loads((project / "flow_world.json").read_text(encoding="utf-8")).get("anchor_residual_m", {})
    return dict(status="ok", anchors=int(len(anchors)),
                residual_m=dict(mean=float(residual.mean()), max=float(residual.max())), recorded_m=recorded)


def guard_output(out, inputs, force):
    """Refuse user-owned output folders, existing files and the inputs themselves."""
    out = Path(out).resolve()
    for name in PROTECTED:
        if out.is_relative_to((ROOT / name).resolve()):
            if not force:
                raise SystemExit(f"--out is inside the user-owned {name}/ directory: {out} "
                                 "(research output belongs under experiments/; --force to override)")
            print(f"warning: writing into {name}/ because --force was given: {out}", file=sys.stderr)
    if out in inputs:
        raise SystemExit(f"--out is one of the inputs: {out}")
    if out.exists() and not force:
        raise SystemExit(f"--out already exists: {out} (pass --force to replace it)")
    return out


# ---------------------------------------------------------------------- drawing
def draw_links(image, uv, links, color, thickness, offset=(0, 0)):
    for a, b in links:
        pa, pb = uv[a], uv[b]
        if not (np.isfinite(pa).all() and np.isfinite(pb).all()):
            continue
        cv2.line(image, (int(round(pa[0])) + offset[0], int(round(pa[1])) + offset[1]),
                 (int(round(pb[0])) + offset[0], int(round(pb[1])) + offset[1]), color, thickness, cv2.LINE_AA)


def draw_points(image, uv, color, radius, offset=(0, 0)):
    for p in uv:
        if np.isfinite(p).all():
            cv2.circle(image, (int(round(p[0])) + offset[0], int(round(p[1])) + offset[1]),
                       radius, color, -1, cv2.LINE_AA)


def put_centered(image, text, x_center, y, scale=0.5, color=TEXT):
    width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
    cv2.putText(image, text, (int(x_center - width / 2), y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def note_in_tile(tile, text):
    """Fallback reason inside the tile, where the label line cannot overflow."""
    for dx, dy, color in ((1, 1, (30, 30, 30)), (0, 0, (240, 240, 240))):
        cv2.putText(tile, text, (6 + dx, tile.shape[0] - 8 + dy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)


def schematic_tile(uv, tile_px, color):
    """Clean wrist-relative drawing: no crop and no photo, so it always has a background."""
    tile = np.full((tile_px, tile_px, 3), 18, np.uint8)
    visible = np.isfinite(uv).all(-1)
    if int(visible.sum()) < 3:
        cv2.putText(tile, "no projection", (8, tile_px // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, DIM, 1, cv2.LINE_AA)
        return tile, "none"
    origin = uv[0] if visible[0] else np.nanmean(uv[visible], axis=0)
    relative = uv - origin
    low, high = np.nanmin(relative[visible], axis=0), np.nanmax(relative[visible], axis=0)
    zoom = float(np.clip(0.72 * tile_px / max(float((high - low).max()), 1e-3), MIN_ZOOM, MAX_ZOOM))
    drawn = relative * zoom - 0.5 * (high + low) * zoom + tile_px / 2.0
    drawn[~visible] = np.nan
    draw_links(tile, drawn, HAND_LINKS_LOCAL, color, 2)
    draw_points(tile, drawn, color, 2)
    return tile, "schematic"


def zoom_tile(frame_padded, uv, box, tile_px, color):
    """Zoom the donor's own hand box out of the full-resolution frame (real pixels)."""
    tile = np.full((tile_px, tile_px, 3), 18, np.uint8)
    centre = 0.5 * (box[:2] + box[2:])
    crop = int(np.clip(1.35 * max(box[2] - box[0], box[3] - box[1]), 80.0, 2.0 * PAD))
    half = crop // 2
    x, y = int(round(centre[0])) + PAD, int(round(centre[1])) + PAD
    tile[:, :] = cv2.resize(frame_padded[y - half:y + half, x - half:x + half], (tile_px, tile_px),
                            interpolation=cv2.INTER_LINEAR)
    zoom = tile_px / float(crop)
    drawn = (uv - centre) * zoom + tile_px / 2.0
    corners = (box[:2] - centre) * zoom + tile_px / 2.0, (box[2:] - centre) * zoom + tile_px / 2.0
    cv2.rectangle(tile, (int(round(corners[0][0])), int(round(corners[0][1]))),
                  (int(round(corners[1][0])), int(round(corners[1][1]))), BOX_YELLOW, 1, cv2.LINE_AA)
    draw_links(tile, drawn, HAND_LINKS_LOCAL, color, 2)
    draw_points(tile, drawn, color, 2)
    return tile, ("zoom" if np.isfinite(drawn).all(-1).sum() else "zoom-empty")


def hand_tile(record, index, side, column, frame_padded, uv, size, tile_px, mode):
    """One inset tile and its status. Invalid boxes / projections fall back, never raise."""
    if mode == "off":
        return None, "off"
    reason = ""
    if mode == "schematic":
        tile, kind = schematic_tile(uv, tile_px, COLUMNS[column][3])
    elif not box_ok(record["boxes"][index, side], size[0], size[1]):
        tile, kind = schematic_tile(uv, tile_px, COLUMNS[column][3])
        kind, reason = "schematic", "invalid box -> schematic"
    else:
        tile, kind = zoom_tile(frame_padded, uv, record["boxes"][index, side], tile_px, COLUMNS[column][3])
    provenance = COLUMNS[column][2]
    if column == 2:
        provenance += "+applied" if record["applied"][index, side] else "+keep"
    if reason:
        note_in_tile(tile, reason)
    return tile, f"{('L', 'R')[side]} {provenance} {kind}"


# ----------------------------------------------------------------------- output
def mp4_boxes(path):
    """Top-level ISO BMFF box list: proves a real `ftyp` + `moov` container."""
    path, offset, found = Path(path), 0, []
    size = path.stat().st_size
    with path.open("rb") as handle:
        while offset + 8 <= size and len(found) < 64:
            handle.seek(offset)
            head = handle.read(8)
            length = int.from_bytes(head[:4], "big")
            name = head[4:8].decode("latin-1")
            if length == 1:
                length = int.from_bytes(handle.read(8), "big")
            elif length == 0:
                length = size - offset
            if length < 8 or offset + length > size:
                break
            found.append((name, offset, length))
            offset += length
    return found


def open_encoder(ffmpeg, partial, width, height, fps, count, crf, preset):
    """Raw BGR frames into a real MP4 (H.264, yuv420p, `moov` moved to the front)."""
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
               "-framerate", format(fps, ".12g"), "-i", "pipe:0",
               "-frames:v", str(count), "-map", "0:v:0", "-an",
               "-c:v", "libx264", "-preset", preset, "-crf", format(crf, ".12g"),
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(partial)]
    return subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fused", required=True, help="fuse_hands.py fused.npz (read-only)")
    parser.add_argument("--project", required=True, help="donor FlowHMR project dir, used for the camera only")
    parser.add_argument("--video", help="source video (default: <project>/input.mp4)")
    parser.add_argument("--out", required=True, help="output MP4 (written atomically)")
    parser.add_argument("--summary", help="JSON sidecar (default: --out with a .json suffix)")
    parser.add_argument("--fps", type=float, help="required fps (default: the donor project's fps)")
    parser.add_argument("--start", type=int, default=0, help="first frame index")
    parser.add_argument("--count", type=int, help="frames to render (default: all of them)")
    parser.add_argument("--panel-width", type=int, default=640, help="source image width per column")
    parser.add_argument("--inset", type=int, default=256, help="hand inset tile side in pixels")
    parser.add_argument("--inset-mode", choices=("zoom", "schematic", "off"), default="zoom")
    parser.add_argument("--crf", type=float, default=20.0)
    parser.add_argument("--preset", default="veryfast")
    parser.add_argument("--force", action="store_true", help="replace --out if it already exists")
    args = parser.parse_args(argv)

    try:
        from mocap.video import executable, probe
    except ImportError:
        raise SystemExit("run this from the repository root: mocap.video provides the FFmpeg tools")
    fused, project = Path(args.fused).resolve(), Path(args.project).resolve()
    video = Path(args.video).resolve() if args.video else (project / "input.mp4").resolve()
    inputs = [fused, video, project / "project.json", project / "flow_world.json", project / "flow_camera.npz"]
    for path, what in ((fused, "fused record"), (project / "flow_world.json", "camera"),
                       (project / "flow_camera.npz", "camera"), (project / "project.json", "project state"),
                       (video, "source video")):
        if not path.is_file():
            raise SystemExit(f"{what} not found: {path}")
    out = guard_output(args.out, {path.resolve() for path in inputs}, args.force)
    out.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = executable("ffmpeg")

    before = {str(path): identity(path) for path in inputs}
    record = load_record(fused)
    n = record["frames"]
    state = json.loads((project / "project.json").read_text(encoding="utf-8"))
    info = probe(video)

    # Same frame indices, count and fps as the take the record came from.
    start = int(args.start)
    count = n - start if args.count is None else int(args.count)
    if start < 0 or count <= 0 or start + count > n:
        raise SystemExit(f"--start/--count must select frames inside 0..{n - 1}")
    fps = float(args.fps if args.fps else state.get("fps", info["fps"]))
    timing = dict(record_frames=n, project_frames=int(state.get("frames", -1)), video_frames=int(info["frames"]),
                  requested_fps=fps, project_fps=float(state.get("fps", 0.0)), video_fps=float(info["fps"]),
                  take_size=[int(state.get("width", 0)), int(state.get("height", 0))],
                  video_size=[int(info["width"]), int(info["height"])])
    if timing["project_frames"] != n:
        raise SystemExit(f"{project}: project.json has {timing['project_frames']} frames, the record has {n}")
    if start + count > min(n, timing["video_frames"]):
        raise SystemExit(f"{video}: {timing['video_frames']} decoded frames, frames {start}..{start + count - 1} asked for")
    if abs(fps - timing["project_fps"]) > 1e-3 or abs(fps - timing["video_fps"]) > 1e-3:
        raise SystemExit(f"fps mismatch: record/project {fps}, video {timing['video_fps']} - the three "
                         "columns need identical frame indices, count and fps")
    if timing["video_size"] != timing["take_size"]:
        raise SystemExit(f"{video} is {timing['video_size']}, the donor take is {timing['take_size']}")

    rotation, offset, k, world = camera_maps(project, n)
    k, k_scale = match_intrinsics(k, timing["video_size"][0], timing["video_size"][1])
    model = body_fk.skeleton()
    body_fk.check_parents(model, H.SMPLH_PARENTS)
    joints, uv = [], []
    for pose in record["poses"]:
        points = body_fk.joints(model, pose, record["shapes"], record["trans"])
        joints.append(points)
        uv.append(project_points(points, rotation, offset, k)[0])
    check = registration_check(project, joints[2], rotation, offset)
    delta = finger_deltas(record["poses"])

    # Layout: the source image per column, then the L/R inset tiles under it.
    scale = float(args.panel_width) / timing["video_size"][0]
    panel_w = int(round(timing["video_size"][0] * scale))
    panel_h = int(round(timing["video_size"][1] * scale))
    panel_w += panel_w % 2
    panel_h += panel_h % 2
    tile_px = abs(int(args.inset)) + abs(int(args.inset)) % 2
    if args.inset_mode != "off" and tile_px < 64:
        raise SystemExit("--inset must be at least 64 pixels (use --inset-mode off for no insets)")
    stack = GAP + LABEL + tile_px if args.inset_mode != "off" else 0
    width = 3 * panel_w
    height = HEADER + panel_h + stack + FOOTER
    panel_uv = [frame_uv * scale for frame_uv in uv]

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    if start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(start))
    partial = out.parent / f"{out.stem}.partial{out.suffix}"
    encoder = open_encoder(ffmpeg, partial, width, height, fps, count, args.crf, args.preset)
    labels, written = {}, 0
    try:
        for index in range(start, start + count):
            ok, frame = cap.read()
            if not ok:
                raise SystemExit(f"{video}: cannot decode frame {index}")
            if (frame.shape[1], frame.shape[0]) != tuple(timing["video_size"]):
                raise SystemExit(f"{video}: frame {index} is {frame.shape[1]}x{frame.shape[0]}")
            small = cv2.resize(frame, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
            padded = cv2.copyMakeBorder(frame, PAD, PAD, PAD, PAD, cv2.BORDER_REPLICATE)
            canvas = np.full((height, width, 3), BG, np.uint8)
            for column, (title, _, _, color) in enumerate(COLUMNS):
                x0 = column * panel_w
                canvas[HEADER:HEADER + panel_h, x0:x0 + panel_w] = small
                put_centered(canvas, title, x0 + panel_w // 2, 26, 0.55, color)
                cv2.rectangle(canvas, (x0, HEADER), (x0 + panel_w - 1, HEADER + panel_h - 1), (70, 70, 70), 1)
                draw_links(canvas, panel_uv[column][index], BODY_LINKS, BODY_GRAY, 1, (x0, HEADER))
                draw_links(canvas, panel_uv[column][index], hand_links(0) + hand_links(1), color, 2, (x0, HEADER))
                rows = HAND_ROWS[0] + HAND_ROWS[1]
                draw_points(canvas, panel_uv[column][index][list(rows)], color, 1, (x0, HEADER))
                for side in (0, 1):                     # FlowHMR's own hand boxes, identical in every column
                    box = record["boxes"][index, side]
                    if box_ok(box, frame.shape[1], frame.shape[0]):
                        scaled = box * scale
                        cv2.rectangle(canvas, (int(scaled[0]) + x0, int(scaled[1]) + HEADER),
                                      (int(scaled[2]) + x0, int(scaled[3]) + HEADER), BOX_YELLOW, 1)
                    tile, label = hand_tile(record, index, side, column, padded,
                                            uv[column][index][list(HAND_ROWS[side])],
                                            (frame.shape[1], frame.shape[0]), tile_px, args.inset_mode)
                    labels[label] = labels.get(label, 0) + 1
                    if tile is None:
                        continue
                    y = HEADER + panel_h + GAP + LABEL
                    x = x0 + (16 if side == 0 else panel_w - 16 - tile_px)
                    canvas[y:y + tile_px, x:x + tile_px] = tile
                    cv2.rectangle(canvas, (x, y), (x + tile_px - 1, y + tile_px - 1), color, 1)
                    put_centered(canvas, label, x + tile_px // 2, y - 6, 0.42, TEXT)
            gates = "  ".join(f"{('L', 'R')[side]} "
                              f"{'applied' if record['applied'][index, side] else 'kept-baseline'}"
                              for side in (0, 1))
            cv2.putText(canvas, f"frame {index + 1}/{n}  t={index / fps:.2f}s  {fps:g} fps  {width}x{height}",
                        (12, height - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT, 1, cv2.LINE_AA)
            cv2.putText(canvas, f"col3 FAH gates: {gates}   finger delta col2 {delta['hamer_vs_flowhmr']['per_frame_mean_deg'][index]:.1f}"
                                f"deg  col3 {delta['fah_vs_flowhmr']['per_frame_mean_deg'][index]:.1f}deg"
                                "   (differences only, no accuracy rating)",
                        (12, height - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, DIM, 1, cv2.LINE_AA)
            try:
                encoder.stdin.write(np.ascontiguousarray(canvas).tobytes())
            except (BrokenPipeError, OSError):
                raise SystemExit(f"ffmpeg stopped early: {encoder.stderr.read().decode('utf-8', 'replace')[:600]}")
            written += 1
            if written % 30 == 0:
                print(f"{written}/{count} frames", flush=True)
    finally:
        cap.release()
        try:
            encoder.stdin.close()
        except OSError:
            pass
    status = encoder.wait()
    if status != 0 or written != count:
        partial.unlink(missing_ok=True)
        raise SystemExit(f"ffmpeg exited {status} after {written}/{count} frames: "
                         f"{encoder.stderr.read().decode('utf-8', 'replace').strip()[:600]}")

    boxes = mp4_boxes(partial)
    names = [name for name, _, _ in boxes]
    encoded = probe(partial)
    if not (names[:1] == ["ftyp"] and "moov" in names and int(encoded["frames"]) == count
            and [int(encoded["width"]), int(encoded["height"])] == [width, height]
            and abs(float(encoded["fps"]) - fps) <= 1e-3):
        partial.unlink(missing_ok=True)
        raise SystemExit(f"output MP4 is not usable: boxes {names}, probe {encoded}")
    partial.replace(out)                       # only a verified container ever reaches --out

    after = {str(path): identity(path) for path in inputs}
    summary = dict(inputs=before, inputs_unchanged=before == after,
                   columns=[dict(index=i + 1, title=title, key=key, tag=tag, color_bgr=list(color))
                            for i, (title, key, tag, color) in enumerate(COLUMNS)],
                   timing=dict(fps=fps, record_frames=n, rendered=written, first_frame=start,
                               last_frame=start + count - 1, cross_checked=timing),
                   invariants=dict(body_rows_0_21_max_abs_delta=record["body_delta"],
                                   rotations="proper rotations checked for all three columns"),
                   finger_delta_deg={name: values["stats"] for name, values in delta.items()},
                   camera=dict(projection="p_cam = (world - [0, camera_height, 0]) @ "
                                          "(camera_RT_yup[t,:3,:3] @ scene_basis.T).T",
                               intrinsics=k.tolist(), intrinsics_scale=k_scale,
                               camera_height_m=float(world["camera_height"]),
                               scene_basis=np.asarray(world["scene_basis"], np.float64).tolist(),
                               fixed_camera=bool(np.allclose(rotation, rotation[0], atol=1e-3)),
                               overlay="registered SMPL-H joints through body_fk, projected in original "
                                       "pixels then scaled to the panel",
                               registration_check=check),
                   tiles=dict(mode=args.inset_mode, side_px=tile_px, panel_scale=scale, label_counts=labels),
                   output=dict(path=str(out), bytes=out.stat().st_size, boxes=names, probe=encoded,
                               ffmpeg=ffmpeg, command_frames=count))
    if before != after:
        raise SystemExit(f"an input changed during the run; {out} was kept - investigate before trusting it")
    summary["output"]["probe"] = probe(out)     # re-probed under its final name
    sidecar = Path(args.summary).resolve() if args.summary else out.with_suffix(".json")
    sidecar.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"{out}: {written} frames at {fps:g} fps, {width}x{height}, MP4 boxes {names}; "
          f"registration {check['status']} {check.get('residual_m', '')}; tile labels {labels}; summary {sidecar}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
