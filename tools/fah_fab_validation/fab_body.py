#!/usr/bin/env python3
"""fab_body: CPU-only processing of a `fab-video` body result + FlowHMR comparison preview.

Research tool for experiments/fah-fab-20261123. Read-only over its inputs; everything it
writes goes to --out-dir:

  fab_smpl.npz                standard SMPL pose / betas / trans + joints, FAB camera frame
                              (`poses` is written as an alias of `pose` for upstream
                              fab.retarget.smpl_input; K stays in original source pixels)
  fab_body_report.json        axes / scale / trajectory / foot-drift evidence (no judgment)
  fab_flow_overlay_30fps.mp4  side-by-side FAB vs FlowHMR overlay from the same source frame
                              (panels are resized from the source frame, so the K used for drawing
                              is rescaled to panel pixels; the reported Ks are not)

FAB input: a `fab-video --results` pickle, exactly as experiments/fah-fab-20261123/FAB/
fab/video.py `run()` writes it:
  {video, fps, size (W,H), K (3,3), model, names, call,
   frames: [{"index": t, "people": [<per-person FABPredictor output>]}]}
and the per-person dict documented by fab/predictor.py: box (4) xyxy px, track, cam_t (3),
smpl_pose (72,) axis-angle, smpl_betas (10,), mesh_smpl_fit (6890,3) metres camera frame,
keypoints2d (70,2), keypoints3d (70,3), contact (16,), plus `<output>_smoothed` copies.
Only the RAW fields of ONE track are used (`--track`; otherwise the largest box per frame).

Camera-space translation: `cam_t` is FAB's MHR camera translation, not the translation of the
neutral SMPL fit, so it is never used here. `trans` comes from a rigid (rotation + translation,
no scale) Kabsch alignment of the upstream neutral SMPL vertices (bench/body_models.SMPL at
transl=0) with `mesh_smpl_fit` — the same relation bench/sonic_motion.py::one_pass_fit uses,
solved rigidly instead of by centroid alone:
  SMPL(global_orient, body_pose, betas, transl) vertices  ~  mesh_smpl_fit   -> (R, t)
`trans = t`, and the same model's 24 kinematic joints ("native joints", authoritative) are
shifted by the same t; SMPL(pose, betas, t) vertices then equal the evaluated vertices + t.
The rotation angle of R and the vertex residual are reported: they say how far the fit is from
a pure translation, they are not an accuracy claim. A `J_regressor @ mesh_smpl_fit` variant is
written next to the joints as a labelled sanity check only (regressing joints off a posed mesh
is not the fit's joint chain).

FlowHMR input (a finished flowhmr project, read-only):
  project.json       pipeline.model must be "flowhmr"; frames / fps / width / height
  initial_joints.npz joints (n,22,3), rotations (n,22,3,3), fps — keys exactly as written by
                     mocap/flow_export.py (`np.savez_compressed(project/(kind+'_joints.npz'),
                     joints=native_points, rotations=native_rotations, fps=np.array(fps))`) and
                     as validated by mocap/preview_data.py (joints (22,3), rotations (22,3,3)).
                     They are the native SMPL-H keypoints3d / world rotations of the committed
                     result, in the committed scene WV world (Y-up, floor at y=0), metres.
  flow_world.json    scene_basis, camera_height, intrinsics (mocap/flow_placement.py)
  flow_camera.npz    camera_RT_yup (frames,4,4)
World(Y-up) -> OpenCV camera conversion reuses the public mocap/camera_tracks.load(); when that
loader refuses (stale receipt etc.) the same fixed-camera branch of that file is applied
verbatim (R = camera_RT_yup[0,:3,:3] @ scene_basis.T, t = -R @ [0, camera_height, 0]) and the
reason is recorded; a take without its own flow_world.json uses whatever saved camera that
loader publishes. Raw camera_RT_yup alone is not a CV camera. Fixed camera only: the camera
track that is used must not rotate by more than 0.5 deg, otherwise this tool stops.

All written 3D quantities use the documented OpenCV camera frame: x right, y down, z forward,
metres, projected with the intrinsics of the model that produced them (both Ks are reported).
Everything reported is descriptive — axes, scale, trajectories, foot drift — no accuracy
judgment is made or implied. Run with the FAB venv Python; CPU only, no GPU, no model weights.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
FAB_DIR = REPO / "experiments/fah-fab-20261123/FAB"
SMPL_MODEL = REPO / "experiments/fah-fab-20261123/fab-home/body_models/smpl/SMPL_NEUTRAL.pkl"
CV = "OpenCV camera frame: x right, y down, z forward, metres"
# SMPL's first 22 joints are SMPL-H's body prefix (same order, same parents), which is what a
# FlowHMR project's initial_joints.npz stores; joints 22/23 are SMPL's hand tips (SMPL-H's hand
# roots), which only exist in the 24-joint SMPL chain and have no FlowHMR counterpart.
NAMES = ("pelvis", "L_hip", "R_hip", "spine1", "L_knee", "R_knee", "spine2", "L_ankle", "R_ankle",
         "spine3", "L_toe", "R_toe", "neck", "L_collar", "R_collar", "head", "L_shoulder",
         "R_shoulder", "L_elbow", "R_elbow", "L_wrist", "R_wrist")
PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19)
LINKS = [(j, p) for j, p in enumerate(PARENTS) if p >= 0]
WATCH = ("pelvis", "L_ankle", "R_ankle", "L_toe", "R_toe", "head")
FEET = ("L_ankle", "R_ankle", "L_toe", "R_toe")
INDEX = {name: i for i, name in enumerate(NAMES)}


def box_area(box):
    """Area of an xyxy box; an inverted extent counts as zero area, not as an absolute span."""
    b = np.asarray(box, float)
    return float(np.prod(np.maximum(0.0, b[2:] - b[:2])))


def rot_deg(rotations):
    """Geodesic angle (deg) of stacked rotation matrices away from identity."""
    return np.rad2deg(np.arccos(np.clip((np.trace(rotations, axis1=1, axis2=2) - 1) * .5, -1, 1)))


def kabsch(src, dst):
    """Rigid (no scale) alignment of stacked point sets: R src + t ~ dst -> R (n,3,3), t (n,3)."""
    cs, cd = src.mean(1), dst.mean(1)
    H = np.einsum("bni,bnj->bij", src - cs[:, None], dst - cd[:, None])
    U, _, Vt = np.linalg.svd(H)
    flip = np.sign(np.linalg.det(Vt.transpose(0, 2, 1) @ U.transpose(0, 2, 1)))
    D = np.broadcast_to(np.eye(3), H.shape).copy()
    D[:, 2, 2] = flip
    R = Vt.transpose(0, 2, 1) @ D @ U.transpose(0, 2, 1)
    return R, cd - np.einsum("bij,bj->bi", R, cs)


def evaluate_smpl(model, pose, betas, batch=8):
    """Upstream neutral SMPL forward at transl=0 -> (verts (n,6890,3), joints (n,24,3))."""
    import torch

    v, j, n = [], [], len(pose)
    with torch.no_grad():
        for i in range(0, n, batch):
            sl = slice(i, i + batch)
            a, b, c = (torch.from_numpy(pose[sl, :3]).float(), torch.from_numpy(pose[sl, 3:]).float(),
                       torch.from_numpy(betas[sl]).float())
            verts, joints = model(a, b, c, None)
            v.append(verts.numpy())
            j.append(joints.numpy())
    return np.concatenate(v).astype(np.float64), np.concatenate(j).astype(np.float64)


def load_fab(path, track):
    """-> (result dict, frames, pose (n,72), betas (n,10), mesh (n,6890,3), skipped, tracks)."""
    with open(path, "rb") as fh:
        res = pickle.load(fh)
    for key in ("fps", "size", "K", "model", "call", "frames"):
        if key not in res:
            raise SystemExit(f"{path} is not a fab-video result pickle: no {key!r}")
    need = {"smpl_pose": (72,), "smpl_betas": (10,), "mesh_smpl_fit": (6890, 3), "box": (4,)}
    frames, pose, betas, mesh, skipped, used = [], [], [], [], [], set()
    for frame in res["frames"]:
        people = [q for q in frame["people"]
                  if all(k in q and np.asarray(q[k]).shape == s for k, s in need.items())]
        want = people if track is None else [q for q in people if q.get("track") == track]
        pick = max(want, key=lambda q: box_area(q["box"]), default=None)
        if pick is None:
            skipped.append(int(frame["index"]))
            continue
        used.add(int(pick.get("track", -1)))
        frames.append(int(frame["index"]))
        pose.append(np.asarray(pick["smpl_pose"], np.float32))
        betas.append(np.asarray(pick["smpl_betas"], np.float32))
        mesh.append(np.asarray(pick["mesh_smpl_fit"], np.float32))
    if not frames:
        raise SystemExit(f"no raw SMPL fit in {path}" + (f" for track {track}" if track is not None else ""))
    return res, np.array(frames), np.stack(pose), np.stack(betas), np.stack(mesh), skipped, sorted(used)


def load_flow(project):
    """Read-only FlowHMR take: joints (n,22,3) in the committed Y-up scene world + CV camera."""
    project = Path(project)
    state = json.loads((project / "project.json").read_text(encoding="utf-8"))
    if state.get("pipeline", {}).get("model") != "flowhmr":
        raise SystemExit(f"{project} is not a flowhmr project")
    archive = project / "initial_joints.npz"
    with np.load(archive, allow_pickle=False) as z:
        keys = sorted(z.files)
        if keys != ["fps", "joints", "rotations"]:
            raise SystemExit(f"{archive} keys {keys} != fps/joints/rotations (mocap/flow_export.py)")
        joints, rotations = np.asarray(z["joints"], float), np.asarray(z["rotations"], float)
        fps = float(np.asarray(z["fps"]).ravel()[0])
    if joints.shape[1:] != (22, 3) or rotations.shape[1:] != (22, 3, 3):
        raise SystemExit(f"{archive} joints {joints.shape} / rotations {rotations.shape} unexpected")
    flat = rotations.reshape(-1, 3, 3)
    check = dict(max_orthonormality_error=float(np.abs(flat.swapaxes(-1, -2) @ flat - np.eye(3)).max()),
                 max_abs_det_minus_one=float(np.abs(np.linalg.det(flat) - 1).max()))
    # Camera. With the take's own flow_world.json placement this is the fixed-camera flow branch
    # of mocap/camera_tracks.load; otherwise whatever that public loader publishes (a shared
    # scene_camera.json). Raw camera_RT_yup alone is never used: scene_basis and camera_height are
    # what place the Y-up world at the camera.
    branch, raw = None, None
    world = json.loads((project / "flow_world.json").read_text(encoding="utf-8")) \
        if (project / "flow_world.json").is_file() else None
    intrinsics = np.asarray(world["intrinsics"], float) if world and world.get("intrinsics") else None
    if world is not None:
        raw = np.load(project / "flow_camera.npz", allow_pickle=False)["camera_RT_yup"].astype(float)
        deviation = rot_deg(raw[:, :3, :3] @ raw[0, :3, :3].T)
        if float(deviation.max()) > 0.5:
            raise SystemExit(f"camera_RT_yup rotates up to {deviation.max():.2f} deg: fixed-camera tool, refusing")
        branch = np.tile(np.eye(4), (len(raw), 1, 1))
        branch[:, :3, :3] = raw[0, :3, :3] @ np.asarray(world["scene_basis"], float).T
        branch[:, :3, 3] = -branch[:, :3, :3] @ np.array([0.0, float(world["camera_height"]), 0.0])
        loader, note = "inline fixed-camera flow branch", "same algebra as the camera_tracks flow branch"
    else:
        loader, note = "mocap.camera_tracks.load", "saved camera of this project (e.g. scene_camera.json)"
    try:
        if str(REPO) not in sys.path:
            sys.path.insert(0, str(REPO))
        from mocap.camera_tracks import load

        track = load(project)["cameras"][0]
        public = np.asarray(track["matrices"], float)
        if intrinsics is None:
            intrinsics = np.asarray(track["K"], float)
        if branch is None:
            branch = public
        elif len(public) == len(raw) and np.allclose(public, branch, atol=1e-6):
            loader, note = "mocap.camera_tracks.load", "public loader output, identical to the flow branch"
        else:
            note = "public loader returned a different track; the flow branch values are used"
    except Exception as exc:  # noqa: BLE001 - any public-loader refusal is recorded, not hidden
        if branch is None:
            raise SystemExit(f"{project} has no usable saved camera: {exc}") from exc
        note = f"mocap.camera_tracks.load refused ({type(exc).__name__}: {exc}); flow branch used"
    if intrinsics is None and (project / "flow_camera.npz").is_file():
        intrinsics = np.load(project / "flow_camera.npz", allow_pickle=False)["K"][0].astype(float)
    if intrinsics is None:
        raise SystemExit(f"{project} publishes no intrinsics")
    stationary = float(rot_deg(branch[:, :3, :3] @ branch[0, :3, :3].T).max())
    if stationary > 0.5:
        raise SystemExit(f"the saved camera track rotates up to {stationary:.2f} deg: fixed-camera tool, refusing")
    return dict(project=str(project), state=state, joints=joints, fps=fps, camera=branch[0], K=intrinsics,
                loader=loader, note=note, camera_stationary_deg=round(stationary, 4), rotation_check=check)


def trajectory(points, fps, names=WATCH):
    """Descriptive camera-frame trajectory per watched joint (no accuracy judgment)."""
    out = {}
    for name in names:
        p = points[:, INDEX[name]]
        step = np.linalg.norm(np.diff(p, axis=0), axis=1) if len(p) > 1 else np.zeros(1)
        out[name] = dict(start=p[0].round(3).tolist(), end=p[-1].round(3).tolist(),
                         net_m=round(float(np.linalg.norm(p[-1] - p[0])), 4),
                         path_m=round(float(step.sum()), 4),
                         x_m=[round(float(v), 4) for v in (p[:, 0].min(), p[:, 0].max())],
                         y_down_m=[round(float(v), 4) for v in (p[:, 1].min(), p[:, 1].max())],
                         z_depth_m=[round(float(v), 4) for v in (p[:, 2].min(), p[:, 2].max())],
                         mean_speed_m_s=round(float(step.mean() * fps), 4))
    return out


def foot_drift(points):
    """Foot drift evidence: horizontal (x, z) movement of each foot while it is the lowest
    (y-down) body point of the frame, plus its height range and total camera-frame path."""
    lowest = points[:, :22, 1].argmin(1)
    out = {}
    for name in FEET:
        p, j = points[:, INDEX[name]], INDEX[name]
        on = np.flatnonzero(lowest == j)
        runs = np.split(on, np.flatnonzero(np.diff(on) != 1))
        out[name] = dict(lowest_frames=int(len(on)),
                         lowest_runs=int(len(runs)),
                         drift_while_lowest_m=[round(float(np.linalg.norm(np.diff(p[run][:, [0, 2]], axis=0), axis=1).sum()), 4) for run in runs[:8] if len(run)],
                         y_down_m=[round(float(v), 4) for v in (p[:, 1].min(), p[:, 1].max())],
                         z_depth_change_m=round(float(p[-1, 2] - p[0, 2]), 4),
                         path_m=round(float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()), 4))
    return out


def scale_summary(points):
    bones = np.linalg.norm(points[:, [b for _, b in LINKS]] - points[:, [a for a, _ in LINKS]], axis=-1)
    tilt = np.arccos(np.clip(-((points[:, INDEX["head"]] - points[:, [7, 8]].mean(1))
                              / np.linalg.norm(points[:, INDEX["head"]] - points[:, [7, 8]].mean(1),
                                               axis=1, keepdims=True))[:, 1], -1, 1))
    mean = (points[:, INDEX["head"]] - points[:, [7, 8]].mean(1)).mean(0)
    mean /= np.linalg.norm(mean)
    return dict(total_bone_length_m=round(float(bones.sum(1).mean()), 4),
                mean_bone_length_m=round(float(bones.mean()), 4),
                head_over_ankles_unit_direction_cv=mean.round(4).tolist(),
                degrees_from_cv_up=dict(mean=round(float(np.rad2deg(tilt.mean())), 2),
                                       min=round(float(np.rad2deg(tilt.min())), 2),
                                       max=round(float(np.rad2deg(tilt.max())), 2)))


def project(K, points):
    z = np.where(np.abs(points[:, 2]) < 1e-6, 1e-6, points[:, 2])
    return np.c_[K[0, 0] * points[:, 0] / z + K[0, 2], K[1, 1] * points[:, 1] / z + K[1, 2]]


def draw(panel, joints, K, color, label):
    px = project(K, joints[:22])
    for a, b in LINKS:
        if joints[a, 2] > 0 and joints[b, 2] > 0:
            cv2.line(panel, tuple(px[a].astype(int)), tuple(px[b].astype(int)), color, 2, cv2.LINE_AA)
    for j in range(22):
        if joints[j, 2] > 0:
            cv2.circle(panel, tuple(px[j].astype(int)), 3, color, -1, cv2.LINE_AA)
    cv2.putText(panel, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1, cv2.LINE_AA)
    return panel


def preview(a, out, frames, joints, K_fab, model_name, flow_cv, flow_ok, K_flow):
    """Side-by-side overlay: both panels are the same source frame, written at 30 fps."""
    video = out / "fab_flow_overlay_30fps.mp4"
    cap = cv2.VideoCapture(str(a.video))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    first, last = int(frames[0]), int(frames[-1])
    writer, held, written, next_frame = None, None, 0, 0
    panel_size = None  # (source W, source H, panel W, panel H) of the last frame drawn
    while True:
        src = first + int(round(written * src_fps / 30.0))
        if src > last:
            break
        while next_frame <= src:
            ok, held = cap.read()
            if not ok:
                held = None
                break
            next_frame += 1
        if held is None:
            break
        row = np.flatnonzero(frames == src)
        panel_h = int(round(held.shape[0] * a.panel_width / held.shape[1]))
        panel_size = [int(held.shape[1]), int(held.shape[0]), int(a.panel_width), panel_h]
        panels = [cv2.resize(held, (a.panel_width, panel_h)) for _ in range(2)]
        # K is in source-frame pixels but the panels are resized; diag(sx, sy, 1) @ K puts both
        # models' intrinsics in panel pixels. Only the drawn K changes: the NPZ/report keep K in
        # original source pixels.
        K_panel = np.diag([a.panel_width / held.shape[1], panel_h / held.shape[0], 1.0])
        if len(row):  # both panels are this one source frame; flow_cv is aligned to frames
            b = row[0]
            draw(panels[0], joints[b], K_panel @ K_fab, (0, 220, 255), f"FAB {model_name} frame {src}")
            if flow_ok[b]:
                draw(panels[1], flow_cv[b], K_panel @ K_flow, (0, 255, 0), f"FlowHMR frame {src + a.flow_offset}")
        strip = np.hstack(panels)
        strip = strip[:strip.shape[0] // 2 * 2, :strip.shape[1] // 2 * 2]
        if writer is None:
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, strip.shape[1::-1])
        writer.write(strip)
        written += 1
    cap.release()
    if writer is not None:
        writer.release()
    return dict(video=str(video), frames=written, fps=30.0, source_fps=float(src_fps),
                panels="left FAB (FAB K) / right FlowHMR (Flow K)", same_source_frame_per_pair=True,
                drawn_K_scaled_to_panel=True, panel_size_source_then_panel=panel_size)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                               formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fab-results", required=True, help="fab-video --results pickle")
    p.add_argument("--smpl-model", default=str(SMPL_MODEL), help="upstream SMPL_NEUTRAL.pkl")
    p.add_argument("--flow-project", required=True, help="finished flowhmr project directory")
    p.add_argument("--video", required=True, help="the same source video fab-video processed")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--track", type=int, help="FAB track id (default: largest box in each frame)")
    p.add_argument("--flow-offset", type=int, default=0, help="Flow row that matches FAB frame 0")
    p.add_argument("--fab-dir", default=str(FAB_DIR), help="FAB checkout to import bench.body_models from")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--panel-width", type=int, default=480)
    p.add_argument("--no-preview", action="store_true", help="skip the overlay mp4")
    a = p.parse_args(argv)

    import torch

    torch.set_num_threads(max(1, a.threads))
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    res, frames, pose, betas, mesh, skipped, used = load_fab(a.fab_results, a.track)
    if str(a.fab_dir) not in sys.path:
        sys.path.insert(0, str(a.fab_dir))
    from bench.body_models import SMPL

    model = SMPL(a.smpl_model, num_betas=10)
    verts, joints0 = evaluate_smpl(model, pose, betas)
    R, trans = kabsch(verts, mesh)
    residual = np.linalg.norm(np.einsum("bij,bvj->bvi", R, verts) + trans[:, None] - mesh, axis=-1)
    del verts
    joints = joints0 + trans[:, None]
    jreg = np.einsum("jv,bvc->bjc", model.J_regressor.numpy().astype(np.float64), mesh)
    joint_gap = np.linalg.norm(joints - jreg, axis=-1)
    K_fab = np.asarray(res["K"], float)
    align = dict(kinematic_joints_minus_j_regressor_m=dict(mean=round(float(joint_gap.mean()), 4),
                                                           p95=round(float(np.quantile(joint_gap, .95)), 4)),
                 rigid_fit=dict(mean_vertex_residual_m=round(float(residual.mean()), 5),
                                p95_vertex_residual_m=round(float(np.quantile(residual, .95)), 5),
                                mean_rotation_from_identity_deg=round(float(rot_deg(R).mean()), 3),
                                max_rotation_from_identity_deg=round(float(rot_deg(R).max()), 3)),
                 note=("trans is the rigid SMPL->mesh_smpl_fit translation; FAB cam_t (MHR) is not used. "
                       "pose/betas are FAB's raw values: any non-identity alignment rotation belongs to "
                       "global_orient and is reported, not folded in."))

    flow = load_flow(a.flow_project)
    flow_pick = np.full(len(frames), -1)
    pick = frames + a.flow_offset
    inside = (pick >= 0) & (pick < len(flow["joints"]))
    flow_pick[inside] = pick[inside]
    have = flow_pick >= 0
    m = flow["camera"]
    flow_cv = np.stack([np.einsum("ij,nj->ni", m[:3, :3], flow["joints"][p]) + m[:3, 3] if p >= 0
                        else np.zeros((22, 3)) for p in flow_pick])
    compared = np.flatnonzero(have)
    pair = {}
    for name in WATCH:
        d = np.linalg.norm(joints[compared, INDEX[name]] - flow_cv[compared, INDEX[name]], axis=-1)
        pair[name] = dict(mean_m=round(float(d.mean()), 4), p50_m=round(float(np.median(d)), 4),
                          p95_m=round(float(np.quantile(d, .95)), 4), max_m=round(float(d.max()), 4))

    # `poses` is the key upstream fab.retarget.smpl_input reads (72D axis-angle, same array as
    # the older `pose` key, which is kept for this tool's own consumers).
    np.savez(out / "fab_smpl.npz", pose=pose, poses=pose, betas=betas, trans=trans, frames=frames,
             fps=float(res["fps"]), K=K_fab, joints_smpl=joints, joints_j_regressor_approx=jreg,
             alignment_rotation=R, source_pickle=str(a.fab_results), source_model=res["model"],
             convention=CV)
    fab_scale, flow_scale = scale_summary(joints), scale_summary(flow_cv[compared])
    report = dict(tool="tools/fah_fab_validation/fab_body.py", convention=CV,
                  fab=dict(pickle=str(a.fab_results), model=res["model"], call=res["call"],
                           fps=float(res["fps"]), size=list(res["size"]), K=K_fab.tolist(),
                           frames_used=len(frames), frame_range=[int(frames[0]), int(frames[-1])],
                           skipped_frames=skipped, tracks_seen=used, requested_track=a.track,
                           pose_shape=list(pose.shape), betas_shape=list(betas.shape),
                           camera_translation_source="rigid Kabsch SMPL(pose,betas,0) -> mesh_smpl_fit",
                           alignment=align, trajectory_camera_frame=trajectory(joints, float(res["fps"])),
                           foot_drift=foot_drift(joints), scale=fab_scale),
                  flow=dict(project=str(a.flow_project), frames=len(flow["joints"]), fps=flow["fps"],
                            initial_joints_keys=["fps", "joints", "rotations"],
                            joints_space="committed scene WV world, Y-up, floor y=0, metres",
                            camera_loader=flow["loader"], camera_note=flow["note"],
                            camera_world_to_cv=m.tolist(), K_flow=flow["K"].tolist(),
                            camera_stationary_deg=flow["camera_stationary_deg"],
                            rotation_check=flow["rotation_check"], flow_offset=a.flow_offset,
                            frames_compared=int(len(compared)),
                            trajectory_camera_frame=trajectory(flow_cv[compared], flow["fps"]),
                            foot_drift=foot_drift(flow_cv[compared]), scale=flow_scale),
                  comparison=dict(frame_pairs=int(len(compared)), same_source_frames=True,
                                  joint_position_difference_fab_minus_flow=pair,
                                  bone_length_ratio_fab_over_flow=round(float(
                                      fab_scale["total_bone_length_m"] / flow_scale["total_bone_length_m"]), 4),
                                  body_axis_angle_between_deg=round(float(np.rad2deg(np.arccos(np.clip(
                                      np.dot(fab_scale["head_over_ankles_unit_direction_cv"],
                                             flow_scale["head_over_ankles_unit_direction_cv"]), -1, 1)))), 2),
                                  note="descriptive only: no accuracy judgment"))
    if not a.no_preview:
        report["preview"] = preview(a, out, frames, joints, K_fab, res["model"], flow_cv, have, flow["K"])
    (out / "fab_body_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(dict(frames_used=len(frames), frame_pairs=len(compared), out_dir=str(out),
                          preview=(report.get("preview") or {}).get("frames")), indent=2))


if __name__ == "__main__":
    main()
