#!/usr/bin/env python3
"""Command-line front end over the same workers, projects and receipts as the GUI.

This module uses only the standard library and imports no AI package: every stage runs
through `mocap.paths.worker_command` in that stage's own validated environment.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
os.environ.pop('PYTHONHOME', None)
os.environ.pop('PYTHONPATH', None)

from mocap.linux_runtime import SETUP_HINT, SUPPORTED_MACHINES, correction_path, effective_profile, interpreter, linux_variant, normalize_machine, runtime_environment  # noqa: E402

# mocap.paths reads RGB_MOCAP_DATA when it is first imported.
os.environ.update(runtime_environment(ROOT))

PREFIX = '@@MOCAP@@'
REQUIRED_PROFILES = ('gui', 'shared', 'flow', 'exmr')
OPTIONAL_PROFILES = ('hand', 'htd')
# worker.py's argparse stages plus the routes worker_progress.py adds before it.
KNOWN_STAGES = ('diagnose', 'prepare', 'download', 'import_model', 'mesh', 'rig', 'save_fbx', 'track', 'pva', 'features',
                'gvhmr', 'refine', 'footmr', 'handflow', 'correct', 'export', 'people', 'person', 'flow_camera',
                'flow_features', 'flowhmr', 'flow_export', 'exmr_preprocess', 'exmr_features', 'exmr', 'flow_batch',
                'exmr_batch', 'gvhmr_batch', 'save_project', 'character', 'save_blend', 'save_blend_scene', 'copy_bvh',
                'camera_background', 'camera_reference')
RESULT_STAGES = {'export', 'flow_export', 'footmr', 'handflow', 'correct'}


# Running workers ----------------------------------------------------------------------

def job_environment(args=None, base=None):
    """The variables gui.py passes to a worker, including FlowHMR's saved option inputs."""
    from mocap import resource_safety
    env = dict(os.environ if base is None else base)
    env.update(runtime_environment(ROOT, env))
    safety = getattr(args, 'resource_safety', None)
    safety = resource_safety.load() if safety is None else safety == 'on'
    env.update(PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1',
               RGB_MOCAP_COMPUTE_MODE=getattr(args, 'compute_mode', None) or 'auto',
               RGB_MOCAP_RESOURCE_SAFETY='1' if safety else '0',
               RGB_MOCAP_FLOW_INTERVAL=str(getattr(args, 'camera_interval', 0)),
               RGB_MOCAP_FLOW_FAST_SAM='1' if getattr(args, 'fast_sam', True) else '0',
               RGB_MOCAP_FLOW_BATCH4='1' if getattr(args, 'sam_batch', True) else '0',
               RGB_MOCAP_FLOW_GPU_CROP='1' if getattr(args, 'gpu_crop', True) else '0',
               RGB_MOCAP_FLOW_HAND_STRIDE4='1' if getattr(args, 'hand_stride4', True) else '0',
               RGB_MOCAP_FLOW_HAMER='1' if getattr(args, 'hamer', False) else '0')
    return env


class EventPrinter:
    """Readable progress, or the raw worker events with --json."""

    def __init__(self, raw=False, stream=None):
        self.raw = raw
        self.stream = sys.stdout if stream is None else stream
        self.last_progress = None

    def __call__(self, stage, event):
        if self.raw:
            print(json.dumps(event, ensure_ascii=False), file=self.stream, flush=True)
            return
        kind = event.get('type')
        if kind == 'progress':
            line = self.progress(event)
            if line != self.last_progress:
                self.last_progress = line
                print(f'[{stage}] {line}', file=self.stream, flush=True)
        elif kind == 'info':
            print(f"[{stage}] {event.get('message', '')}", file=self.stream, flush=True)
        elif kind == 'prepared':
            print(f"[{stage}] project: {event.get('project')}", file=self.stream, flush=True)
        elif kind == 'complete' and not event.get('batch_inner'):
            details = ['cached' if event.get('cached') else f"{float(event.get('elapsed') or 0):.1f} s"]
            if event.get('peak_ram_gib') is not None:
                details.append(f"peak RAM {float(event['peak_ram_gib']):.2f} GiB")
            if event.get('peak_cuda_allocated_gib') is not None:
                details.append(f"peak allocated VRAM {float(event['peak_cuda_allocated_gib']):.2f} GiB")
            print(f"[{stage}] complete: {', '.join(details)}", file=self.stream, flush=True)

    @staticmethod
    def progress(event):
        message = str(event.get('message') or event.get('phase') or '')
        current, total = event.get('current'), event.get('total')
        if isinstance(current, (int, float)) and isinstance(total, (int, float)) and total > 0:
            return f'{message} {100 * current / total:.0f}%'.strip()
        return message


class WorkerSupervisor:
    """Use the GUI's RAM and stall policy while stdout may be blocked."""

    def __init__(self, process, env, guard_factory=None):
        if guard_factory is None:
            from mocap.processing_guard import ProcessingGuard
            guard_factory = ProcessingGuard
        self.process = process
        self.guard = guard_factory(process.pid, resource_safety=env.get('RGB_MOCAP_RESOURCE_SAFETY') != '0')
        self.finished = threading.Event()
        self.message = None
        self.thread = threading.Thread(target=self.watch, name='worker-supervision', daemon=True)

    def watch(self):
        while not self.finished.wait(.5):
            if self.process.poll() is not None:
                return
            try:
                self.message = self.guard.check()
            except Exception as error:
                self.message = 'Worker supervision failed: ' + str(error)
            if self.message:
                stop_group(self.process)
                return

    def close(self):
        self.finished.set()
        if self.thread.ident is not None:
            self.thread.join()


def run_worker(stage, arguments, env, on_event, guard_factory=None):
    """Run one worker to completion. Returns (exit code, events)."""
    from mocap.paths import worker_command
    command = worker_command(stage, *map(str, arguments))
    events = []
    # A separate process group lets cancellation also reach FFmpeg and the HaMeR child.
    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, text=True, encoding='utf-8',
                               errors='replace', stderr=subprocess.STDOUT, start_new_session=True)
    supervisor = None
    try:
        supervisor = WorkerSupervisor(process, env, guard_factory)
        supervisor.thread.start()
        for line in process.stdout:
            supervisor.guard.output()
            if line.startswith(PREFIX):
                try:
                    event = json.loads(line[len(PREFIX):])
                except ValueError:
                    print(line, end='', flush=True)
                    continue
                events.append(event)
                on_event(stage, event)
            else:
                print(line, end='', flush=True)
        code = process.wait()
        supervisor.close()
        if supervisor.message:
            raise RuntimeError(supervisor.message)
        return code, events
    except BaseException:
        stop_group(process)
        raise
    finally:
        if supervisor is not None:
            supervisor.close()
        process.stdout.close()


def stop_group(process, grace=10):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pass
    # A parent can exit while a child ignores SIGTERM and retains stdout.
    # Reap the entire owned group even when waiting for the parent succeeded.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def run_request_worker(stage, arguments, env, on_event):
    """Batch stages receive a request file, exactly as the GUI's start_jobs does."""
    from mocap.queue_groups import BATCH_STAGES, remove_request, write_request
    if stage not in BATCH_STAGES:
        return run_worker(stage, arguments, env, on_event)
    project = arguments[arguments.index('--project') + 1]
    request = write_request(stage, arguments)
    try:
        return run_worker(stage, ['--request', str(request), '--project', project], env, on_event)
    finally:
        remove_request(request)


# FlowHMR pipeline ---------------------------------------------------------------------

def correction_arguments(args):
    values = ['--correction-preset', args.correction_preset, '--correction-mode', args.correction_mode,
              '--foot-lock', str(int(args.foot_lock)), '--hand-lock', str(int(args.hand_lock)),
              '--pelvis-offset', str(args.pelvis_offset)]
    return values if args.floor is None else ['--floor', str(args.floor), *values]


def flow_prepare_arguments(args):
    from mocap.motion_tools import result_kind
    result = result_kind(correction=args.correct)
    values = ['--video', str(Path(args.video).resolve()), '--start', str(args.start), '--motion-tool', 'flowhmr', '--result', result]
    if args.end is not None:
        values += ['--end', str(args.end)]
    if args.project is not None:
        values += ['--project', str(Path(args.project).resolve())]
    return values + correction_arguments(args) if args.correct else values


def flow_jobs(args, project):
    """Same order as gui.py's Reconstruct 3D: optional correction, then the preview/rig."""
    from mocap.motion_tools import MOTION_TOOLS, result_kind
    from mocap.queue_groups import join_single_pipeline
    result = result_kind(correction=args.correct)
    jobs = []
    for stage in MOTION_TOOLS['flowhmr'].stages(correction=args.correct):
        arguments = ['--project', str(project)]
        if stage in RESULT_STAGES:
            arguments += ['--result', result]
        if args.force and stage != 'track':
            arguments.append('--force')
        jobs.append((stage, arguments))
    start = next(i for i, (stage, _) in enumerate(jobs) if stage == 'flow_features')
    # One worker keeps features and generation connected, as the GUI does for one actor.
    return jobs[:start] + join_single_pipeline(jobs[start:], project)


def flow_stage_names(args):
    from mocap.motion_tools import MOTION_TOOLS
    return MOTION_TOOLS['flowhmr'].stages(correction=args.correct)


def missing_models(stages, camera_interval=0, hamer=False):
    """The GUI's readiness rule. Workers still verify files before using them."""
    from mocap.flow_hamer import MODEL_KEYS
    from mocap.models import BY_KEY, missing, ready
    names = []
    for stage in stages:
        if stage == 'flow_camera' and camera_interval == -1:
            continue
        names += [model.name for model in missing('flow_export' if stage in ('correct', 'export') else stage)]
    if hamer:
        names += [BY_KEY[key].name for key in MODEL_KEYS if not ready(BY_KEY[key])]
    return list(dict.fromkeys(names))


def command_flow(args, runner=run_request_worker):
    video = Path(args.video)
    if not video.is_file():
        print(f'Video not found: {video}', file=sys.stderr)
        return 2
    stages = flow_stage_names(args)
    absent = missing_models(stages, args.camera_interval, args.hamer)
    if absent and not args.dry_run:
        print('Prepare these models first (GUI Model Manager, or the import_model stage):\n  ' + '\n  '.join(absent), file=sys.stderr)
        return 2
    prepare = flow_prepare_arguments(args)
    if args.dry_run:
        print(json.dumps(dict(stages=stages, prepare=prepare, flow_options=flow_option_summary(args), missing_models=absent), indent=2))
        return 0
    env = job_environment(args)
    printer = EventPrinter(args.json)
    code, events = runner('prepare', prepare, env, printer)
    if code:
        return report_failure('prepare', code)
    project = next((event['project'] for event in events if event.get('type') == 'prepared'), None)
    if project is None:
        print('The prepare stage did not report a project.', file=sys.stderr)
        return 1
    for stage, arguments in flow_jobs(args, project):
        code, _ = runner(stage, arguments, env, printer)
        if code:
            return report_failure(stage, code, project)
    print(f'Finished. Project: {project}')
    return 0


def flow_option_summary(args):
    return dict(camera_interval=args.camera_interval, fast_sam=args.fast_sam or args.hamer, sam_batch4=args.sam_batch,
                sam_gpu_crop=args.gpu_crop, sam_hand_stride4=args.hand_stride4 or args.hamer, hamer=args.hamer)


def report_failure(stage, code, project=None):
    location = f' Completed stages are kept in {project}; rerun to reuse them.' if project else ''
    print(f'Stage {stage} failed with exit code {code}.{location}', file=sys.stderr)
    return code if isinstance(code, int) and 0 < code < 256 else 1


# Export and the escape hatch ----------------------------------------------------------

EXPORT_STAGES = {'fbx': 'save_fbx', 'bvh': 'copy_bvh', 'blend': 'save_blend'}


class ExportError(ValueError):
    """A request refused before any worker starts."""


def export_target(output, requested='auto'):
    """Resolve (format, worker target). A Blender bundle is a new folder holding <name>.blend."""
    output = Path(output).resolve()
    suffix = output.suffix.lower().lstrip('.')
    fmt = suffix if requested == 'auto' else requested
    if fmt not in EXPORT_STAGES:
        raise ExportError(f'Cannot infer the format of {output.name}: use a .fbx, .bvh or .blend name, or pass --format.')
    if fmt == 'blend':
        name = output.name[:-len('.blend')] if suffix == 'blend' else output.name
        folder = output.with_name(name)
        if folder.exists():
            raise ExportError(f'{folder} already exists. Choose a new Blender bundle folder; existing files are kept.')
        target = folder / (name + '.blend')
    elif suffix and suffix != fmt:
        raise ExportError(f'{output.name} does not match --format {fmt}.')
    else:
        target = output if suffix else output.with_name(f'{output.name}.{fmt}')
    parent = folder.parent if fmt == 'blend' else target.parent
    if not parent.is_dir():
        raise ExportError(f'The destination folder does not exist: {parent}')
    return fmt, target


def export_options(args, fmt):
    """Row keys from source_export_dialog.options(); an absent key keeps the worker default."""
    for flag, value, owners in (('--bvh-profile', args.bvh_profile, ('bvh',)), ('--rig-name', args.rig_name, ('fbx', 'blend')),
                                ('--[no-]animate-camera', args.animate_camera, ('blend',)), ('--background-refine', args.background_refine, ('blend',)),
                                ('--camera-translation-scale', args.camera_translation_scale, ('blend',))):
        if value is not None and fmt not in owners:
            raise ExportError(f"{flag} applies to {' and '.join(owners)} exports, not {fmt}.")
    row = dict(prepend_tpose=True) if args.prepend_tpose else {}
    if args.rig_name is not None:
        row['rig_name'] = args.rig_name
    if fmt == 'bvh':
        row['bvh_profile'] = args.bvh_profile or 'generic'
    if fmt != 'blend':
        return row
    if args.background_refine and args.animate_camera is False:
        raise ExportError('--background-refine needs the exported camera animation.')
    scale = 1. if args.camera_translation_scale is None else args.camera_translation_scale
    if not 0 <= scale <= 100:
        raise ExportError('--camera-translation-scale must be between 0 and 100.')
    return {**row, 'animate_camera': args.animate_camera is not False, 'background_refine': bool(args.background_refine), 'camera_translation_scale': scale}


def require_blender():
    from mocap.retarget.runtime import blender
    try:
        blender()
    except RuntimeError:
        where = 'Blender has no official Linux aarch64 build: set RGBMOCAP_BLENDER to your own build.' if normalize_machine() == 'aarch64' \
            else 'Put it in runtime_character/ or set RGBMOCAP_BLENDER.'
        raise ExportError(f'Blender bundles need Blender 4.5. {where} FBX and BVH export do not use Blender.') from None


def export_plan(project, kind, env):
    """The selected source and shared placement, or the reason the result cannot be exported."""
    python = interpreter(ROOT, 'shared')
    if not python.is_file():
        raise RuntimeError(f'shared runtime is missing ({python}). {SETUP_HINT}')
    # selected_export and the scene context need NumPy, so they run in the export workers' runtime.
    code_text = ('import sys;sys.path.insert(0,sys.argv[1]);from mocap.runtime_registry import activate;activate(sys.argv[1],"shared");'
                 'from mocap.export_plan import main;main(sys.argv[2:])')
    code, output = capture([python, '-I', '-X', 'utf8', '-c', code_text, ROOT, project, kind], env, 300)
    line = next((text for text in reversed(output.splitlines()) if text.startswith('@@PLAN@@')), None)
    value = json.loads(line[len('@@PLAN@@'):]) if line else dict(error=f'Could not check the export: {last_line(output)}')
    if 'error' in value:
        raise ExportError(value['error'])
    return value


def discard_export(request, leftover):
    """As gui.cleanup_fbx_export, after the worker ended: only exact owned paths are removed."""
    from mocap.blend_support import cleanup
    for path in (request, leftover):
        try:
            if path.is_dir():
                cleanup(path, request)
            else:
                path.unlink(missing_ok=True)
        except (OSError, ValueError) as error:
            print(f'Could not remove {path}: {error}', file=sys.stderr)


def command_export(args, runner=run_worker, plan=export_plan):
    """Export a committed result through the existing workers and their generation checks."""
    from mocap.blend_support import partial
    from mocap.paths import HOME
    from mocap.storage import atomic_json
    import uuid
    project = Path(args.project).resolve()
    env = job_environment(args)
    try:
        fmt, target = export_target(args.output, args.format)
        options = export_options(args, fmt)
        if fmt == 'blend':
            require_blender()
        planned = plan(project, args.kind, env)
        if (args.background_refine or args.camera_translation_scale is not None) and planned['model'] not in ('flowhmr', 'exmr'):
            raise ExportError('--background-refine and --camera-translation-scale apply to FlowHMR and ExMR saved cameras.')
    except ExportError as error:
        print(error, file=sys.stderr)
        return 2
    row = dict(project=str(project), kind=args.kind, source=planned['source'], target=str(target), **options)
    if 'scene' in planned:
        row['scene'] = planned['scene']
    folder = HOME / 'cache/worker_jobs'
    folder.mkdir(parents=True, exist_ok=True)
    # The worker consumes this request; discard_export covers a worker that was killed.
    request = folder / (uuid.uuid4().hex + '.json')
    leftover = partial(target.parent, request) if fmt == 'blend' else target.with_name(f'.{target.name}.rgbmocap-{request.stem}.partial')
    atomic_json(request, [row])
    stage = EXPORT_STAGES[fmt]
    try:
        code, _ = runner(stage, ['--project', str(project), '--output', str(request)], env, EventPrinter(args.json))
    finally:
        discard_export(request, leftover)
    if code:
        return report_failure(stage, code, project)
    print(json.dumps(dict(type='exported', format=fmt, path=str(target))) if args.json else f'Exported: {target}')
    return 0


def command_stage(args, runner=run_worker):
    """Run any worker stage with its native arguments; nothing is added or validated here."""
    arguments = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
    code, _ = runner(args.name, arguments, job_environment(args), EventPrinter(args.json))
    return code


def command_stages(args):
    from mocap.motion_tools import MOTION_TOOLS
    from mocap.runtime_registry import PROFILES, profile_for
    routed = {}
    for stage in KNOWN_STAGES:
        key = profile_for(stage, ())
        routed[stage] = dict(profile=key, environment=PROFILES[key]['dev'])
    value = dict(machine=normalize_machine(), variant=(linux_variant() or {}).get('name'),
                 torch={key: effective_profile(key)['packages']['torch'] for key in PROFILES},
                 flowhmr_pipeline=MOTION_TOOLS['flowhmr'].stages(), flowhmr_corrected_pipeline=MOTION_TOOLS['flowhmr'].stages(correction=True),
                 hamer=dict(runs_in='exmr', note='A child of flowhmr; enable it with flow --hamer.'), stages=routed,
                 note='rig/mesh/export/correct/character route by saved project model when --project is given.')
    if args.json:
        print(json.dumps(value, indent=2))
        return 0
    print(f"Machine:          {value['machine']} ({value['variant'] or 'standard pins'})")
    print('FlowHMR:          ' + ' -> '.join(value['flowhmr_pipeline']))
    print('With correction:  ' + ' -> '.join(value['flowhmr_corrected_pipeline']))
    print('HaMeR:            optional child of flowhmr, in .venv-exmr (flow --hamer)')
    print(f"\n{'stage':<20}{'profile':<10}environment")
    for stage, row in routed.items():
        print(f"{stage:<20}{row['profile']:<10}{row['environment']}")
    print('\n' + value['note'])
    return 0


# Doctor -------------------------------------------------------------------------------

def check(name, status, detail=''):
    return dict(name=name, status=status, detail=str(detail))


def capture(command, env, timeout=120):
    try:
        done = subprocess.run([str(part) for part in command], cwd=ROOT, env=env, capture_output=True, text=True,
                              encoding='utf-8', errors='replace', timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as error:
        return 1, str(error)
    return done.returncode, (done.stdout + done.stderr).strip()


def last_line(text):
    lines = [line for line in str(text).splitlines() if line.strip()]
    return lines[-1] if lines else ''


def check_profiles(env):
    rows = []
    for key in (*REQUIRED_PROFILES, *OPTIONAL_PROFILES):
        name = f'runtime {key}'
        python = interpreter(ROOT, key)
        absent = 'fail' if key in REQUIRED_PROFILES else 'warn'
        # A variant may lack GUI wheels on some systems; the CLI remains usable without them.
        if key == 'gui' and linux_variant() is not None:
            absent = 'warn'
        if not python.is_file():
            rows.append(check(name, absent, f'{python.parent.parent.name} is missing'))
            continue
        code, output = capture([python, '-I', '-X', 'utf8', ROOT / 'worker_boot.py', '--runtime-check', key], env)
        rows.append(check(name, 'ok' if code == 0 else absent, python.parent.parent.name if code == 0 else last_line(output)))
    return rows


def check_architecture():
    machine = normalize_machine()
    variant = linux_variant()
    if machine not in SUPPORTED_MACHINES:
        return [check('architecture', 'fail', f'{machine}: no pinned wheels; supported: {", ".join(SUPPORTED_MACHINES)}')]
    if variant is None:
        return [check('architecture', 'ok', f'{machine}: standard pins')]
    rows = [check('architecture', 'ok', f"{machine}: {variant['name']}, Torch {variant['cuda_packages']['torch']['version']}")]
    for name, row in variant.get('unavailable', {}).items():
        rows.append(check(f'not available: {name}', 'warn', row['reason']))
    return rows


def check_torchvision_builds():
    """PyPI's aarch64 torchvision is a CPU build with the same version as the CUDA one."""
    if linux_variant() is None:
        return []
    rows = []
    for folder in dict.fromkeys(effective_profile(key)['dev'] for key in ('shared', 'flow', 'exmr', 'hand', 'htd')):
        for library in sorted((ROOT / folder / 'lib').glob('python*/site-packages/torchvision/_C*.so')):
            cuda = b'libcudart' in library.read_bytes()
            rows.append(check(f'torchvision build {folder}', 'ok' if cuda else 'fail', 'links the CUDA runtime' if cuda else 'CPU build; rerun tools/setup_ubuntu.py'))
    return rows


def check_qt_runtime(env):
    python = interpreter(ROOT, 'gui')
    if not python.is_file():
        return []  # check_profiles already reports the missing environment.
    code, output = capture([python, '-I', '-X', 'utf8', ROOT / 'tools/check_qt_runtime.py'], env, 30)
    return [check('Qt painting bindings', 'ok' if code == 0 else 'warn' if code == 2 else 'fail',
                  last_line(output) if code == 0 else last_line(output) + ' Run tools/setup_ubuntu.py --profiles shared.')]


def check_tools(env):
    from mocap.video import executable
    rows = []
    for name in ('ffmpeg', 'ffprobe'):
        try:
            path = executable(name)
        except RuntimeError as error:
            rows.append(check(name, 'fail', error))
            continue
        code, output = capture([path, '-version'], env, 30)
        rows.append(check(name, 'ok' if code == 0 else 'fail', f'{path}: {output.splitlines()[0] if output else ""}'))
    from mocap.retarget.runtime import blender
    try:
        path = blender()
    except RuntimeError:
        return rows + [check('blender', 'warn', 'Missing. Needed for character import, retargeting and Blender bundles only; see docs/UBUNTU.md.')]
    code, output = capture([path, '--version'], env, 60)
    first = output.splitlines()[0] if output else ''
    rows.append(check('blender', 'ok' if code == 0 and ' 4.5' in first else 'warn', f'{path}: {first}'))
    return rows


def check_libraries(env):
    """dlopen in a fresh process, so .runtime/lib and the system path both count."""
    rows = []
    for library, purpose, status in (('libGLU.so.1', 'HaMeR/pyrender import', 'fail'), ('libEGL.so.1', 'HaMeR/pyrender import', 'fail'),
                                     ('libGL.so.1', 'GUI viewport', 'warn'), ('libxcb-cursor.so.0', 'Qt on X11', 'warn')):
        code, output = capture([sys.executable, '-I', '-c', f'import ctypes;ctypes.CDLL({library!r})'], env, 30)
        rows.append(check(f'library {library}', 'ok' if code == 0 else status, purpose if code == 0 else f'{purpose}: {last_line(output)}'))
    return rows


def check_correction(env):
    # Correction workers use the Python 3.10 profiles.
    python = interpreter(ROOT, 'shared')
    artifacts = sorted((ROOT / 'third_party/KimodoMotionCorrection/bin').glob('_motion_correction*.so'))
    if not artifacts:
        return [check('native correction', 'warn', 'No Linux module. Correction is unavailable until tools/build_motion_correction.py builds it.')]
    if not python.is_file():
        return [check('native correction', 'warn', f'{artifacts[0].name} found; .venv is needed to test it.')]
    code_text = ('import sys;sys.path.insert(0,sys.argv[1]);from mocap.runtime_registry import activate;activate(sys.argv[1],"shared");'
                 'from mocap.linux_runtime import correction_backend;print(correction_backend().__file__)')
    code, output = capture([python, '-I', '-X', 'utf8', '-c', code_text, ROOT], env)
    return [check('native correction', 'ok' if code == 0 else 'fail', last_line(output))]


def check_assets():
    from mocap.storage import sha256
    rows = json.loads((ROOT / 'installer/external-assets.json').read_text(encoding='utf-8'))['files']
    bad = {'fail': [], 'warn': []}
    for row in rows:
        path = ROOT / row['path']
        if not path.is_file() or path.stat().st_size != row['bytes'] or sha256(path) != row['sha256']:
            bad['fail' if row['path'].startswith('third_party/FlowHMR/') else 'warn'].append(row['path'])
    hint = ' Run tools/setup_ubuntu.py --stage-assets.'
    result = []
    if bad['fail']:
        result.append(check('FlowHMR auxiliary files', 'fail', ', '.join(bad['fail']) + hint))
    if bad['warn']:
        result.append(check('other auxiliary files', 'warn', f"{len(bad['warn'])} of {len(rows)} missing or different; GVHMR/add-ons may need them." + hint))
    return result or [check('auxiliary files', 'ok', f'{len(rows)} verified')]


def check_models():
    core = missing_models(['track', 'flow_camera', 'flow_features', 'flowhmr', 'flow_export'])
    hamer = [name for name in missing_models([], hamer=True)]
    return [check('FlowHMR models', 'fail' if core else 'ok', ', '.join(core) or 'ready'),
            check('HaMeR models', 'warn' if hamer else 'ok', ', '.join(hamer) or 'ready')]


def check_gpu(env, real=False):
    smi = shutil.which('nvidia-smi')
    if not smi:
        rows = [check('NVIDIA GPU', 'warn', 'nvidia-smi was not found')]
    else:
        code, output = capture([smi, '-L'], env, 30)
        rows = [check('NVIDIA GPU', 'ok' if code == 0 and output else 'warn', output)]
    if not real:
        return rows
    code_text = ('import json,sys;sys.path.insert(0,sys.argv[1]);from mocap.runtime_registry import activate;activate(sys.argv[1],sys.argv[2]);'
                 'from mocap.diagnostics import collect;value=collect()\n'
                 'try:\n'
                 ' import torch,torchvision;boxes=torch.tensor([[0.,0.,1.,1.],[0.,0.,1.,1.]],device="cuda");'
                 'value["torchvision_nms_verified"]=torchvision.ops.nms(boxes,torch.tensor([.9,.8],device="cuda"),.5).tolist()==[0]\n'
                 'except Exception as error:value["torchvision_nms_error"]=str(error)\n'
                 'print("@@DIAG@@"+json.dumps(value))')
    for key in ('shared', 'flow', 'exmr'):
        python = interpreter(ROOT, key)
        if not python.is_file():
            rows.append(check(f'CUDA {key}', 'fail', 'environment is missing'))
            continue
        code, output = capture([python, '-I', '-X', 'utf8', '-c', code_text, ROOT, key], env, 300)
        line = next((text for text in output.splitlines() if text.startswith('@@DIAG@@')), None)
        value = json.loads(line[len('@@DIAG@@'):]) if line else {}
        passed = code == 0 and value.get('cuda') is True and value.get('cuda_matmul_verified') is True
        # Standard pins carry a tagged CUDA torchvision; a variant's untagged one must prove it.
        if linux_variant() is not None:
            passed = passed and value.get('torchvision_nms_verified') is True
        rows.append(check(f'CUDA {key}', 'ok' if passed else 'fail',
                          f"{value.get('torch')} {value.get('gpu')} sm_{''.join(map(str, value.get('compute_capability') or []))} "
                          f"matmul={value.get('cuda_matmul_verified')} nms={value.get('torchvision_nms_verified', value.get('torchvision_nms_error'))}"
                          if value else last_line(output)))
    return rows


def run_checks(env, gpu=False):
    rows = [check('platform', 'ok' if sys.platform == 'linux' else 'fail', sys.platform)]
    rows += check_architecture()
    rows += check_profiles(env)
    rows += check_qt_runtime(env)
    rows += check_torchvision_builds()
    rows += check_tools(env)
    rows += check_libraries(env)
    rows += check_correction(env)
    rows += check_assets()
    rows += check_models()
    rows += check_gpu(env, gpu)
    free = shutil.disk_usage(ROOT).free / 1024**3
    rows.append(check('free disk', 'ok' if free >= 20 else 'warn', f'{free:.1f} GiB'))
    if not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
        rows.append(check('display', 'warn', 'No display: the GUI needs one, or QT_QPA_PLATFORM=offscreen for smoke tests.'))
    return rows


def command_doctor(args):
    rows = run_checks(job_environment(args), args.gpu)
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        for row in rows:
            print(f"{row['status'].upper():<5} {row['name']:<28} {row['detail']}")
    failures = sum(row['status'] == 'fail' for row in rows)
    if not args.json:
        print(f"\n{failures} failed, {sum(row['status'] == 'warn' for row in rows)} warnings. "
              + ('This checks readiness; only --gpu runs real CUDA work.' if not args.gpu else 'GPU checks ran real CUDA operations.'))
    return 1 if failures else 0


# Arguments ----------------------------------------------------------------------------

def add_common(parser):
    parser.add_argument('--json', action='store_true', help='Print raw machine-readable output.')
    parser.add_argument('--compute-mode', choices=('auto', 'low_memory'), default='auto', help='Resource Mode, as in the GUI.')
    parser.add_argument('--resource-safety', choices=('on', 'off'), help='Override the saved local preference for this run.')


def parser():
    root = argparse.ArgumentParser(prog='rgbmocap.sh', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = root.add_subparsers(dest='command', required=True)
    doctor = commands.add_parser('doctor', help='Check environments, tools, native modules and models.')
    add_common(doctor)
    doctor.add_argument('--gpu', action='store_true', help='Also run a real CUDA matmul in each main environment.')
    doctor.set_defaults(handler=command_doctor)
    stages = commands.add_parser('stages', help='List pipelines and the environment each stage uses.')
    stages.add_argument('--json', action='store_true')
    stages.set_defaults(handler=command_stages)
    flow = commands.add_parser('flow', help='Run FlowHMR on one video, for one person.')
    add_common(flow)
    flow.add_argument('--video', required=True, type=Path)
    flow.add_argument('--project', type=Path, help='Project folder; default is under projects/.')
    flow.add_argument('--start', type=float, default=0.0, help='Start of the processing range, in seconds.')
    flow.add_argument('--end', type=float, help='End of the processing range, in seconds.')
    flow.add_argument('--camera-interval', type=int, choices=(-1, 0, 1, 2, 5, 10, 30), default=0,
                      help='0 Moving/Unknown Auto; -1 Static Camera; otherwise analyze every N frames.')
    flow.add_argument('--hamer', action='store_true', help='Optional HaMeR finger refinement in the ExMR environment; forces Fast SAM and 4-frame hands.')
    for name, flag, label in (('fast_sam', 'fast-sam', 'Fast SAM'), ('sam_batch', 'sam-batch', 'adaptive Fast SAM batching'),
                              ('gpu_crop', 'gpu-crop', 'GPU hand crops'), ('hand_stride4', 'hand-stride4', '4-frame hand sampling')):
        flow.add_argument('--' + flag, dest=name, action=argparse.BooleanOptionalAction, default=True, help=f'{label} (default on).')
    flow.add_argument('--correct', action='store_true', help='Run contact correction and prepare the corrected result.')
    flow.add_argument('--correction-preset', default='normal', help='A preset from mocap/correction_settings.py.')
    flow.add_argument('--correction-mode', choices=('flat', 'multi_level'), default='flat')
    flow.add_argument('--floor', type=float, help='Manual world floor Y; default is automatic.')
    flow.add_argument('--foot-lock', action=argparse.BooleanOptionalAction, default=True)
    flow.add_argument('--hand-lock', action=argparse.BooleanOptionalAction, default=False)
    flow.add_argument('--pelvis-offset', type=float, default=0.0)
    flow.add_argument('--force', action='store_true', help='Recompute completed stages instead of reusing them.')
    flow.add_argument('--dry-run', action='store_true', help='Print the planned stages and readiness; run nothing.')
    flow.set_defaults(handler=command_flow)
    export = commands.add_parser('export', help='Export a prepared result as FBX, BVH or a portable Blender bundle.')
    add_common(export)
    export.add_argument('--project', required=True, type=Path, help='The model project reported by the flow command.')
    export.add_argument('--kind', default='initial', help='Result to export, such as initial or corrected_initial.')
    export.add_argument('--output', required=True, type=Path, help='A .fbx or .bvh file, which is replaced atomically, or a new Blender bundle folder (x or x.blend makes x/x.blend).')
    export.add_argument('--format', choices=('auto', *EXPORT_STAGES), default='auto', help='auto uses the --output suffix.')
    export.add_argument('--prepend-tpose', action='store_true', help='Add one T-pose frame before the motion. Original frames shift by one.')
    export.add_argument('--rig-name', help='FBX and Blender rig name; the default is SourceAnimation.')
    export.add_argument('--bvh-profile', choices=('generic', 'biped'), help='BVH: Generic / Blender (default) or 3ds Max Biped.')
    export.add_argument('--animate-camera', action=argparse.BooleanOptionalAction, default=None, help='Blender: export saved camera tracking (default on).')
    export.add_argument('--background-refine', action='store_true', default=None, help='Blender: Refine Camera Rotation, for FlowHMR and ExMR (default off).')
    export.add_argument('--camera-translation-scale', type=float, help='Blender: camera movement scale from 0 to 100, for FlowHMR and ExMR (default 1).')
    export.set_defaults(handler=command_export)
    stage = commands.add_parser('stage', help='Escape hatch: run one worker stage with its native arguments.',
                                description='Example: stage correct --project P --result corrected_initial')
    add_common(stage)
    stage.add_argument('name', help='Worker stage; see the stages command.')
    stage.add_argument('arguments', nargs=argparse.REMAINDER)
    stage.set_defaults(handler=command_stage)
    return root


def _terminated(*_):
    raise KeyboardInterrupt


def main(argv=None):
    args = parser().parse_args(argv)
    signal.signal(signal.SIGTERM, _terminated)
    try:
        return args.handler(args)
    except KeyboardInterrupt:
        print('Cancelled. Completed stages will be reused next time.', file=sys.stderr)
        return 130
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
