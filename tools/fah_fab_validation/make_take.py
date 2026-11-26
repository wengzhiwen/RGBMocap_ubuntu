#!/usr/bin/env python3
"""Turn a fused take into an independent research project and export it with the app's own code.

    .venv/bin/python tools/fah_fab_validation/make_take.py \
        --fused experiments/fah-fab-20261123/run-1/fused_motion.pt \
        --out-dir experiments/fah-fab-20261123/takes/fah-fingers [--fbx]

No production file, project or receipt is touched and nothing in `mocap/` is
patched: a new research project directory is created under `--out-dir` holding

    project.json          pipeline.model = flowhmr, frames / fps / width / height
    hmr4d_results.pt      the fused SMPL-H motion, in FlowHMR's own schema
    initial_sources.json  this take's own source manifest: the fused tensor above
    fused.npz / report.json / take_manifest.json   provenance copies

and RGB MoCap's own export code is then called on it:

    mocap.flow_export.prepare(take, emit, 'initial')   -> <kind>_motion.bvh, motion.bvh,
                                                         rig / surface / joint artifacts
    mocap.fbx_export.run(request)                      -> generic-profile FBX (--fbx)

`initial_sources.json` is written because this take genuinely has one input, the
fused motion tensor; the donor project's `flowhmr_receipt.json`,
`flow_world.json` and feature caches are deliberately **not** copied, so the
donor project's identity is never claimed or reused.

`--donor <flowhmr project>` additionally re-checks, against the donor, that rows
0..21 of `rot6d`, `trans` and `shapes` are bit-identical in the take (only finger
rows 22..51 may differ).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _path in (str(ROOT), str(Path(__file__).resolve().parent)):
    if _path not in sys.path:
        sys.path.insert(0, _path)


def load_pt(path):
    import torch
    value = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    return {k: (v.numpy() if hasattr(v, "numpy") else v) for k, v in value.items()}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fused", required=True, help="fuse_hands.py fused_motion.pt")
    p.add_argument("--out-dir", required=True, help="independent research take directory (must not exist)")
    p.add_argument("--take-name", default="", help="rig name used in the FBX (default: directory name)")
    p.add_argument("--width", type=int, default=0)
    p.add_argument("--height", type=int, default=0)
    p.add_argument("--fps", type=float, default=0.0)
    p.add_argument("--donor", default="", help="donor FlowHMR project: re-check body rows are unchanged")
    p.add_argument("--fused-npz", default="", help="fuse_hands.py fused.npz (provenance copy)")
    p.add_argument("--report", default="", help="fuse_hands.py report.json (provenance copy)")
    p.add_argument("--fbx", action="store_true", help="also run the app's generic FBX export")
    args = p.parse_args(argv)

    fused = Path(args.fused).resolve()
    take = Path(args.out_dir).resolve()
    if take.exists():
        raise SystemExit(f"{take} already exists; research takes are never overwritten")

    motion = load_pt(fused)
    rot6d = np.asarray(motion["rot6d"], np.float64)
    if rot6d.ndim != 4 or rot6d.shape[0] != 1 or rot6d.shape[2:] != (52, 6):
        raise SystemExit(f"{fused}: unexpected rot6d {rot6d.shape}, expected (1, frames, 52, 6)")
    frames = rot6d.shape[1]
    state = {}
    donor = Path(args.donor).resolve() if args.donor else None
    if donor:
        state = json.loads((donor / "project.json").read_text(encoding="utf-8"))
    width = args.width or int(state.get("width", 0))
    height = args.height or int(state.get("height", 0))
    fps = args.fps or float(state.get("fps", 30.0))
    if width <= 0 or height <= 0:
        raise SystemExit("width/height are unknown: pass --width/--height or --donor")

    # mocap's validator and FlowHMR's own result reader expect tensors, so the numpy
    # view from load_pt is converted before anything is written into the take.
    sys.path.insert(0, str(ROOT / "third_party" / "FlowHMR"))
    import torch
    from mocap.flow_backend import validate_motion
    from mocap.flow_export import prepare
    from mocap.result_artifacts import file_stat

    tensors = {k: (torch.as_tensor(v) if isinstance(v, np.ndarray) else v) for k, v in motion.items()}
    validate_motion(tensors, frames)

    take.mkdir(parents=True)
    for source, name in ((args.fused_npz, "fused.npz"), (args.report, "report.json")):
        if source:
            path = Path(source).resolve()
            if path.is_file():
                shutil.copy2(path, take / name)
    (take / "project.json").write_text(json.dumps(dict(
        version=1, frames=frames, fps=fps, width=width, height=height,
        pipeline=dict(model="flowhmr", result="initial", htd_refine=False, footmr=False, handflow=False),
        research_take=dict(fused_motion=str(fused), donor=str(donor) if donor else None,
                           fah_fingers=motion.get("fah_fingers")),
        source=dict(path=str(donor / "input.mp4") if donor else None, width=width, height=height,
                    fps=fps, frames=frames)), indent=2) + "\n", encoding="utf-8")

    torch.save(tensors, take / "hmr4d_results.pt")
    (take / "initial_sources.json").write_text(json.dumps(
        {"hmr4d_results.pt": file_stat(take / "hmr4d_results.pt")}, indent=2) + "\n", encoding="utf-8")

    def emit(kind, **event):
        line = f"[{kind}] " + " ".join(f"{k}={v}" for k, v in event.items() if k != "message")
        if event.get("message"):
            line += f" {event['message']}"
        print(line, flush=True)

    bvh = prepare(take, emit, "initial")
    checks = dict(frames=frames, fps=fps, bvh=str(bvh), bvh_bytes=int(Path(bvh).stat().st_size),
                  export_json=json.loads((take / "export.json").read_text(encoding="utf-8")))

    if donor:
        from hand_math import rot6d_to_matrix
        donor_motion = load_pt(donor / "hmr4d_results.pt")
        donor_rot = np.asarray(donor_motion["rot6d"], np.float64)[0]
        take_rot = np.asarray(load_pt(take / "hmr4d_results.pt")["rot6d"], np.float64)[0]
        same_body = np.array_equal(take_rot[:, 0:22], donor_rot[:, 0:22])
        same_trans = np.array_equal(np.asarray(motion["trans"]), np.asarray(donor_motion["trans"]))
        same_shapes = np.array_equal(np.asarray(motion["shapes"]), np.asarray(donor_motion["shapes"]))
        matrix_delta = np.abs(rot6d_to_matrix(take_rot) - rot6d_to_matrix(donor_rot)).max((-1, -2))
        checks["donor_identity"] = dict(body_rows_bit_identical=bool(same_body), trans_identical=bool(same_trans),
                                       shapes_identical=bool(same_shapes),
                                       changed_rows=int((matrix_delta > 0).any(0).sum()),
                                       changed_row_indices=[int(i) for i in np.flatnonzero((matrix_delta > 0).any(0))],
                                       max_rotation_delta_rad=float(matrix_delta.max()))
        if not (same_body and same_trans and same_shapes):
            raise SystemExit("the take is not a finger-only change of the donor: " + json.dumps(checks["donor_identity"]))

    outputs = {}
    if args.fbx:
        from mocap.fbx_export import run as run_fbx
        from mocap.motion_tools import selected_export

        source = selected_export(take, "initial")
        if source is None:
            raise SystemExit("the app refuses this take as an export source; see take_report.json checks")
        target = take / f"{args.take_name or take.name}.fbx"
        request = take / "fbx_request.json"
        request.write_text(json.dumps([dict(project=str(take), kind="initial", target=str(target),
                                            source=str(source), fbx_profile="generic",
                                            rig_name=args.take_name or take.name)]), encoding="utf-8")
        count = run_fbx(request, emit)
        outputs["fbx"] = dict(written=count, target=str(target), exists=target.is_file(),
                              bytes=int(target.stat().st_size) if target.is_file() else 0)
    checks["fbx"] = outputs.get("fbx")

    (take / "take_manifest.json").write_text(json.dumps(dict(
        kind="fah-fab research take", fused_motion=str(fused), donor=str(donor) if donor else None,
        frames=frames, fps=fps, width=width, height=height,
        copied=["project.json", "hmr4d_results.pt", "initial_sources.json"]
                + [name for _, name in ((args.fused_npz, "fused.npz"), (args.report, "report.json"))
                   if (take / name).is_file()],
        not_copied=["flowhmr_receipt.json", "flow_world.json", "flow_tokens.pt", "flow_camera.npz",
                    "flow_placement.json", "tracking.npz", "input.mp4"]), indent=2) + "\n", encoding="utf-8")
    (take / "take_report.json").write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")
    print(f"take ready: {take}\n  BVH {checks['bvh']} ({checks['bvh_bytes']} bytes)"
          + (f"\n  FBX {outputs['fbx']['target']} ({outputs['fbx']['bytes']} bytes)" if outputs.get("fbx") else ""))


if __name__ == "__main__":
    main()
