#!/usr/bin/env python3
"""Compare the native correction solver across CPU architectures on fixed synthetic motion.

Run one interpreter per build, then compare the two reports:

    .venv/bin/python -I -B tools/correction_crosscheck.py run --output amd64.json
    .venv/bin/python -I -B tools/correction_crosscheck.py run --output arm64.json   # on ARM
    python3 tools/correction_crosscheck.py compare amd64.json arm64.json

Only NumPy and the built module are needed: no GPU, model, project or user data.
"""
import argparse
import base64
import hashlib
import importlib.util
import json
import math
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
# A generic standing skeleton in meters: pelvis-relative offsets from each parent.
OFFSETS = [(0, 0, 0), (.06, -.09, 0), (-.06, -.09, 0), (0, .11, 0), (.03, -.38, 0), (-.03, -.38, 0), (0, .14, 0), (.01, -.42, -.03),
           (-.01, -.42, -.03), (0, .05, 0), (.02, -.06, .12), (-.02, -.06, .12), (0, .21, 0), (.08, .12, 0), (-.08, .12, 0),
           (0, .09, 0), (.1, .03, 0), (-.1, .03, 0), (.25, 0, 0), (-.25, 0, 0), (.24, 0, 0), (-.24, 0, 0)]
FRAMES = 120
LEFT_FOOT, RIGHT_FOOT = 7, 8


def synthetic_motion(np):
    """Deterministic gait-like input. No randomness, so every build receives identical bytes."""
    time = np.arange(FRAMES, dtype=np.float64) / 30
    roots = np.stack((.6 * time, .9 + .02 * np.sin(4 * np.pi * time), .05 * np.sin(2 * np.pi * time)), axis=1)
    rotations = np.zeros((FRAMES, 22, 4))
    rotations[..., 0] = 1
    for joint, (amplitude, phase) in {1: (.35, 0), 2: (.35, np.pi), 4: (.45, .5), 5: (.45, np.pi + .5), 3: (.08, .2), 16: (.2, 1), 17: (.2, 2)}.items():
        angle = amplitude * np.sin(2 * np.pi * time + phase) + (.25 if joint in (4, 5) else 0)
        rotations[:, joint, 0], rotations[:, joint, 1] = np.cos(angle / 2), np.sin(angle / 2)
    return roots.astype(np.float32), rotations.astype(np.float32)


def forward_kinematics(np, roots, rotations):
    positions = np.zeros((FRAMES, 22, 3))
    world = np.zeros((FRAMES, 22, 3, 3))
    w, x, y, z = [rotations[..., i].astype(np.float64) for i in range(4)]
    local = np.stack((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                      2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                      2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)), axis=-1).reshape(FRAMES, 22, 3, 3)
    positions[:, 0], world[:, 0] = roots, local[:, 0]
    for joint in range(1, 22):
        parent = PARENTS[joint]
        positions[:, joint] = positions[:, parent] + np.einsum('tij,j->ti', world[:, parent], OFFSETS[joint])
        world[:, joint] = world[:, parent] @ local[:, joint]
    return positions


def load_module(path=None):
    if path is None:
        from mocap.linux_runtime import correction_path
        path = correction_path(ROOT)
        if path is None:
            raise SystemExit('No correction module for this interpreter. Build it with tools/build_motion_correction.py, or pass --module.')
    spec = importlib.util.spec_from_file_location('_motion_correction', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, Path(path)


def solve(module, np):
    """Same call shape as mocap/correction.py: anatomical solve with explicit contact goals."""
    roots, rotations = synthetic_motion(np)
    positions = forward_kinematics(np, roots, rotations)
    windows = np.zeros((FRAMES, 2), bool)
    windows[10:50, 0] = windows[70:110, 0] = True
    windows[40:80, 1] = True
    targets = positions[:, [LEFT_FOOT, RIGHT_FOOT]].copy()
    for side, (first, end) in ((0, (10, 50)), (0, (70, 110)), (1, (40, 80))):
        # Hold each foot at one point, with a small floor offset, to give the solver real work.
        targets[first:end, side] = targets[(first + end) // 2, side] + (0, -.01, 0)
    contacts = np.zeros((FRAMES, 4), np.float32)
    contacts[:, [0, 2]] = windows
    zero = np.zeros(FRAMES, np.float32)
    # 'r' and 'q' are views: the solver writes its result into them, as the app relies on.
    r, q = roots.copy(), rotations.copy()
    metrics = module.correct_motion(r.reshape(-1), q.reshape(-1), r.copy().reshape(-1), q.copy().reshape(-1),
                                    zero, zero, zero, zero, zero, zero, contacts.reshape(-1), PARENTS, [list(map(float, offset)) for offset in OFFSETS],
                                    [[0., 0., 0., 1.] for _ in PARENTS], 20, 21, LEFT_FOOT, RIGHT_FOOT, .5, .04, False,
                                    np.ascontiguousarray(targets, np.float32).reshape(-1), np.array([1., 0, 0, 1., 0, 0], np.float64))
    return r, q, dict(metrics), rotations


def report(np, module, path):
    r, q, metrics, source = solve(module, np)
    if not (np.isfinite(r).all() and np.isfinite(q).all()):
        raise SystemExit('The solver returned non-finite motion.')
    return dict(schema=1, machine=platform.machine(), python=platform.python_version(), numpy=np.__version__, module=path.name,
                module_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), frames=FRAMES,
                changed_rotation_values=int((q != source).sum()), metrics={key: float(value) for key, value in metrics.items()},
                roots=base64.b64encode(r.astype('<f4').tobytes()).decode(), rotations=base64.b64encode(q.astype('<f4').tobytes()).decode())


def decode(encoded):
    from array import array
    values = array('f')
    values.frombytes(base64.b64decode(encoded))
    if sys.byteorder != 'little':
        values.byteswap()
    return values


def compare(first, second):
    """Largest differences. Standard library only, so it also works with a system Python."""
    a_roots, b_roots = decode(first['roots']), decode(second['roots'])
    a_rot, b_rot = decode(first['rotations']), decode(second['rotations'])
    if len(a_roots) != len(b_roots) or len(a_rot) != len(b_rot):
        raise SystemExit('The reports have different motion sizes.')
    root_m = max(map(lambda pair: abs(pair[0] - pair[1]), zip(a_roots, b_roots)), default=0.)
    rotation_deg = 0.
    for start in range(0, len(a_rot), 4):
        a, b = a_rot[start:start + 4], b_rot[start:start + 4]
        sign = -1. if sum(x * y for x, y in zip(a, b)) < 0 else 1.
        # atan2 stays exact near zero, where acos of a dot product is dominated by rounding.
        apart = math.sqrt(sum((x - sign * y) ** 2 for x, y in zip(a, b)))
        together = math.sqrt(sum((x + sign * y) ** 2 for x, y in zip(a, b)))
        rotation_deg = max(rotation_deg, math.degrees(4 * math.atan2(apart, together)))
    identical = first['roots'] == second['roots'] and first['rotations'] == second['rotations']
    return dict(identical=identical, max_root_difference_m=root_m, max_rotation_difference_deg=rotation_deg,
                metric_differences={key: abs(first['metrics'][key] - second['metrics'].get(key, math.inf)) for key in first['metrics']})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    run = commands.add_parser('run')
    run.add_argument('--output', required=True, type=Path)
    run.add_argument('--module', type=Path, help='Built module; default is the one matching this interpreter.')
    check = commands.add_parser('compare')
    check.add_argument('first', type=Path)
    check.add_argument('second', type=Path)
    check.add_argument('--max-root-mm', type=float, default=.05)
    check.add_argument('--max-rotation-deg', type=float, default=.05)
    args = parser.parse_args(argv)
    if args.command == 'run':
        import numpy as np
        module, path = load_module(args.module)
        value = report(np, module, path)
        args.output.write_text(json.dumps(value), encoding='utf-8')
        print(f"{value['machine']} {value['module']}: {value['changed_rotation_values']} rotation values changed; metrics {value['metrics']}")
        return 0
    first, second = (json.loads(path.read_text(encoding='utf-8')) for path in (args.first, args.second))
    result = compare(first, second)
    print(json.dumps(dict(machines=[first['machine'], second['machine']], **result), indent=2))
    # A difference within tolerance is expected across compilers and CPUs; report, do not hide it.
    within = result['max_root_difference_m'] * 1000 <= args.max_root_mm and result['max_rotation_difference_deg'] <= args.max_rotation_deg
    print('WITHIN TOLERANCE' if within else 'DIFFERENT BEYOND TOLERANCE')
    return 0 if within else 1


if __name__ == '__main__':
    raise SystemExit(main())
