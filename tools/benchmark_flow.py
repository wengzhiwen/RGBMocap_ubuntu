#!/usr/bin/env python3
"""Gated FlowHMR migration and performance experiments. See docs/BENCHMARK_PROTOCOL.md.

The launcher uses the standard library. Model work stays in validated, isolated
worker environments. It never edits a supplied conditioning project.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import platform
import signal
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    from mocap.storage import atomic_json
    atomic_json(Path(path), value)


def require_gate(path):
    value = read(path)
    if value.get('schema') != 1 or value.get('scope') != 'gb10-migration' or value.get('passed') is not True:
        raise ValueError('A successful real GB10 migration report is required before timing.')
    checks = value.get('checks', {})
    required = ('finite', 'hamer_body_preserved', 'correction_fingers_preserved',
                'selected_results', 'native_correction', 'source_fbx', 'cached_rerun')
    if ('GB10' not in value.get('gpu', '') or any(checks.get(key) is not True for key in required)
            or checks.get('frames', 0) < 60 or len(checks.get('hamer_valid', [])) != 2
            or min(checks['hamer_valid']) <= 0 or not checks.get('decoded_sha256')):
        raise ValueError('The migration report lacks full GB10 inference/export/cache evidence.')
    return value


def require_baselines(paths):
    values = [read(path) for path in paths]
    kinds = set()
    for value in values:
        if value.get('scope') != 'warm-generator' or value.get('mode') != 'single' or value.get('passed') is not True:
            raise ValueError('Dual-GPU trials need successful single-GPU generator baselines.')
        if len(value.get('rounds', [])) < 5 or value.get('warmups', 0) < 1:
            raise ValueError('Establish at least five measured rounds and one warm-up per baseline.')
        gpu = value['hardware']['gpus'][0]['name']
        kinds.add('gb10' if 'GB10' in gpu else 'pro4000' if 'PRO 4000' in gpu else gpu)
    if kinds != {'gb10', 'pro4000'} or len(values) != 2:
        raise ValueError('Supply one GB10 and one PRO 4000 baseline.')
    if values[0]['identity'] != values[1]['identity']:
        raise ValueError('Baseline weights, code and conditioning must be identical.')
    return next(value for value in values if 'PRO 4000' in value['hardware']['gpus'][0]['name'])


def worker_command(profile, operation, *arguments):
    from mocap.linux_runtime import interpreter
    code = ('import sys;sys.path.insert(0,sys.argv[1]);'
            'from mocap.runtime_registry import activate;activate(sys.argv[1],sys.argv[2]);'
            'from mocap.experimental.flow_benchmark import main;main(sys.argv[3:])')
    return [str(interpreter(ROOT, profile)), '-I', '-X', 'utf8', '-c', code, str(ROOT), profile, operation, *map(str, arguments)]


def execute(command, log, env):
    from tools.ubuntu_cli import WorkerSupervisor, stop_group
    events = []
    with Path(log).open('x', encoding='utf-8') as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', start_new_session=True)
        supervisor = WorkerSupervisor(process, env)
        supervisor.thread.start()
        try:
            for line in process.stdout:
                supervisor.guard.output()
                stream.write(line)
                stream.flush()
                print(line, end='', flush=True)
                try: events.append(json.loads(line))
                except ValueError: pass
            code = process.wait()
            if supervisor.message:
                raise RuntimeError(supervisor.message)
            if code:
                raise RuntimeError(f'Experiment worker exited {code}; see {log}.')
        except BaseException:
            stop_group(process)
            raise
        finally:
            supervisor.close()
            process.stdout.close()
    return events


def environment():
    from tools.ubuntu_cli import job_environment
    env = job_environment(argparse.Namespace(resource_safety='on'))
    # Do not inherit a caller's distributed rank or launch address.
    for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'):
        env.pop(key, None)
    return env


def idle_devices(mode='single'):
    """Reject contended baselines without stopping or reconfiguring other work."""
    def query(kind, fields):
        output = subprocess.check_output(['nvidia-smi', f'--query-{kind}={fields}', '--format=csv,noheader'], text=True, timeout=15)
        return [[part.strip() for part in row] for row in csv.reader(output.splitlines()) if row]
    devices = query('gpu', 'index,uuid,name')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible is not None:
        selected = []
        for key in visible.split(','):
            matches = [row for row in devices if row[0] == key.strip() or row[1].startswith(key.strip()) and key.strip()]
            if len(matches) != 1:
                raise ValueError('Cannot map CUDA_VISIBLE_DEVICES to physical GPUs for contention checks.')
            selected.append(matches[0])
        devices = selected
    needed = 1 if mode == 'single' else 2
    if len(devices) < needed:
        raise ValueError('Insufficient visible GPUs for this experiment.')
    devices = devices[:needed]
    busy = [row for row in query('compute-apps', 'gpu_uuid,pid,process_name') if row[0] in {d[1] for d in devices}]
    if busy:
        raise RuntimeError(f'Performance run deferred: selected GPU has existing compute work: {busy}')
    return devices


def migration(args):
    from mocap.linux_runtime import interpreter
    import psutil
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True).strip()
    if platform.machine() not in ('aarch64', 'arm64') or 'GB10' not in gpu:
        raise ValueError('The migration gate must actually run on the GB10 host.')
    available = psutil.virtual_memory().available
    if available < 32 * 1024**3:
        raise RuntimeError(f'GB10 migration deferred: {available / 1024**3:.1f} GiB available; require 32 GiB for this test. Other workloads were preserved.')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = dict(schema=1, scope='gb10-migration', passed=False, gpu=gpu, checks={})
    write(output / 'migration.json', report)
    env = environment()
    command = [str(interpreter(ROOT, 'shared')), '-X', 'utf8', str(ROOT / 'tools/ubuntu_cli.py'),
        'flow', '--video', str(ROOT / 'docs/samples/motion-capture.mp4'), '--project', str(output / 'project'),
        '--end', str(args.frames / 30), '--camera-interval', '-1', '--hamer', '--correct', '--resource-safety', 'on', '--json']
    try:
        fresh = execute(command, output / 'fresh.log', env)
        project = next(Path(e['project']) for e in fresh if isinstance(e, dict) and e.get('type') == 'prepared')
        request = dict(project=str(project), frames=args.frames, output=str(output), gpu=gpu)
        write(output / 'verification-request.json', request)
        execute(worker_command('shared', 'verify', output / 'verification-request.json'), output / 'verify.log', env)
        checks = read(output / 'checks.json')
        execute([str(interpreter(ROOT, 'shared')), '-X', 'utf8', str(ROOT / 'tools/ubuntu_cli.py'), 'export',
            '--project', str(project), '--kind', 'corrected_initial', '--output', str(output / 'motion.fbx'),
            '--resource-safety', 'on'], output / 'export.log', env)
        if (output / 'motion.fbx').stat().st_size < 1024:
            raise ValueError('The source FBX export is unexpectedly empty.')
        cached = execute(command, output / 'cached.log', env)
        cached_stages = {e.get('stage') for e in cached if isinstance(e, dict) and e.get('type') == 'complete' and e.get('cached')}
        if not {'track', 'flow_camera', 'flow_features', 'flowhmr'}.issubset(cached_stages):
            raise ValueError('Repeated CLI run did not confirm all inference caches.')
        execute(worker_command('shared', 'verify', output / 'verification-request.json'), output / 'verify-cached.log', env)
        after = read(output / 'checks.json')
        if checks != after:
            raise ValueError('The cached run changed verified source/output identities.')
        report.update(passed=True, checks={**checks, 'source_fbx': True, 'cached_rerun': True}, project=str(project))
    except BaseException as error:
        report['error'] = str(error)
        raise
    finally:
        write(output / 'migration.json', report)


def generator(args):
    gate = require_gate(args.migration_report)
    baseline = require_baselines(args.baselines) if args.mode != 'single' else None
    if args.mode != 'single' and not args.reference:
        raise ValueError('TP2/PP2 requires the PRO 4000 single-GPU reference.pt.')
    if args.mode == 'single' and args.reference:
        raise ValueError('A single-GPU baseline creates its own reference; omit --reference.')
    idle_devices(args.mode)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    request = dict(mode=args.mode, project=str(args.project.resolve()), output=str(output),
        warmups=args.warmups, rounds=args.rounds, migration=gate, baseline=baseline,
        reference=str(Path(args.reference).resolve()) if args.reference else None)
    write(output / 'request.json', request)
    write(output / 'result.json', dict(schema=1, scope='warm-generator', mode=args.mode, passed=False))
    execute(worker_command('flow', 'generator', output / 'request.json'), output / 'worker.log', environment())


def pipeline(args):
    """Fresh-process end-to-end rounds; every repetition owns a new project."""
    from mocap.linux_runtime import interpreter
    require_gate(args.migration_report)
    devices = idle_devices()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = dict(schema=1, scope='fresh-pipeline-including-prepare-and-preview', mode='single',
        passed=False, hardware=devices, frames=args.frames, warmups=args.warmups, rounds=[],
        source_video=str(args.video.resolve()), camera_interval=args.camera_interval,
        hamer=True, correction=True, resource_safety=True, os_cache='unmanaged; no cache clearing',
        excludes='External source FBX/Blender export; CPU validation and hashing excluded from timer.')
    write(output / 'result.json', report)
    expected = None
    env = environment()
    try:
        for repetition in range(-args.warmups, args.rounds):
            idle_devices()
            folder = output / (f'warmup{-repetition}' if repetition < 0 else f'round{repetition}')
            folder.mkdir()
            command = [str(interpreter(ROOT, 'shared')), '-X', 'utf8', str(ROOT / 'tools/ubuntu_cli.py'),
                'flow', '--video', str(args.video.resolve()), '--project', str(folder / 'project'),
                '--end', str(args.frames / 30), '--camera-interval', str(args.camera_interval),
                '--hamer', '--correct', '--resource-safety', 'on', '--json']
            started = time.perf_counter()
            events = execute(command, folder / 'pipeline.log', env)
            seconds = time.perf_counter() - started
            completed = [e for e in events if isinstance(e, dict) and e.get('type') == 'complete']
            if any(e.get('cached') for e in completed if e.get('stage') in ('track', 'flow_camera', 'flow_features', 'flowhmr')):
                raise ValueError('A fresh pipeline repetition unexpectedly reused inference outputs.')
            project = next(Path(e['project']) for e in events if isinstance(e, dict) and e.get('type') == 'prepared')
            write(folder / 'verification-request.json', dict(project=str(project), frames=args.frames, output=str(folder)))
            execute(worker_command('shared', 'verify', folder / 'verification-request.json'), folder / 'verify.log', env)
            checks = read(folder / 'checks.json')
            identity = {key: checks[key] for key in ('frames', 'decoded_sha256', 'input_sha256')}
            if expected is None: expected = identity
            if identity != expected:
                raise ValueError('Working-video pixels changed between repetitions.')
            if repetition >= 0:
                report['rounds'].append(dict(round=repetition, seconds=seconds, stages=completed, checks=checks))
            report['runtimes'] = checks['runtimes']
            write(output / 'result.json', report)
        report.update(passed=True, identity=expected)
    except BaseException as error:
        report['error'] = str(error)
        raise
    finally:
        write(output / 'result.json', report)


def summary(args):
    for path in args.reports:
        value = read(path)
        rounds = value.get('rounds', [])
        if not value.get('passed') or not rounds:
            print(f'{path}: FAILED / incomplete')
            continue
        seconds = [r['seconds'] for r in rounds]
        print(f"{path}: {value['mode']}, n={len(seconds)}, median={statistics.median(seconds):.4f}s, range={min(seconds):.4f}–{max(seconds):.4f}s")


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest='command', required=True)
    gate = commands.add_parser('migration', help='Fresh public-sample FlowHMR + HaMeR + correction + export on GB10.')
    gate.add_argument('--output', type=Path, required=True)
    gate.add_argument('--frames', type=int, choices=(60, 451, 690), default=60)
    gate.set_defaults(handler=migration)
    gen = commands.add_parser('generator', help='Warm generator scope only, gated by actual GB10 migration.')
    gen.add_argument('--migration-report', type=Path, required=True)
    gen.add_argument('--project', type=Path, required=True, help='Trusted local conditioning project, read only.')
    gen.add_argument('--output', type=Path, required=True, help='New, private experiment folder.')
    gen.add_argument('--mode', choices=('single', 'tp2', 'pp2'), default='single')
    gen.add_argument('--rounds', type=int, default=5)
    gen.add_argument('--warmups', type=int, default=1)
    gen.add_argument('--baselines', type=Path, nargs=2, default=[])
    gen.add_argument('--reference', type=Path, help='PRO 4000 single-GPU reference.pt for numerical comparison.')
    gen.set_defaults(handler=generator)
    whole = commands.add_parser('pipeline', help='Fresh FlowHMR + HaMeR + correction, including preparation and previews.')
    whole.add_argument('--migration-report', type=Path, required=True)
    whole.add_argument('--video', type=Path, default=ROOT / 'docs/samples/motion-capture.mp4')
    whole.add_argument('--output', type=Path, required=True)
    whole.add_argument('--frames', type=int, choices=(60, 451, 690), default=60)
    whole.add_argument('--camera-interval', type=int, choices=(-1, 0, 1, 2, 5, 10, 30), default=-1)
    whole.add_argument('--rounds', type=int, default=5)
    whole.add_argument('--warmups', type=int, default=1)
    whole.set_defaults(handler=pipeline)
    report = commands.add_parser('summary')
    report.add_argument('reports', type=Path, nargs='+')
    report.set_defaults(handler=summary)
    return root


def main():
    args = parser().parse_args()
    if getattr(args, 'rounds', 5) < 5 or getattr(args, 'warmups', 1) < 1:
        raise ValueError('Use at least five rounds and one excluded warm-up.')
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    args.handler(args)


if __name__ == '__main__':
    try: main()
    except (ValueError, RuntimeError, OSError, KeyError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)
