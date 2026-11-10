"""CLI source export: requests, guards and cleanup. CPU only; no models, projects or user data.

Worker tests use a synthetic BVH and are skipped without NumPy and SciPy (present in the pinned
environments). Run with:  python3 -m unittest discover -s tools/tests -v
"""
import contextlib
import functools
import io
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]

from mocap import blend_support, export_plan  # noqa: E402
import ubuntu_cli  # noqa: E402

try:
    import numpy  # noqa: F401
    import scipy  # noqa: F401
    HAS_NUMERICS = True
except ImportError:
    HAS_NUMERICS = False


class TempRoot(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name).resolve()
        self.project = self.root / 'project'
        self.project.mkdir()
        patcher = mock.patch('mocap.paths.HOME', self.root)
        patcher.start()
        self.addCleanup(patcher.stop)

    def args(self, *extra, output='out.fbx'):
        return ubuntu_cli.parser().parse_args(['export', '--project', str(self.project), '--output', str(self.root / output), *map(str, extra)])

    def plan(self, model='flowhmr', **extra):
        return lambda *_: dict(model=model, source=str(self.project / 'initial.bvh'), **extra)


class Recorder:
    """Stands in for a worker that consumes its request, as all three do."""

    def __init__(self, code=0):
        self.code = code
        self.calls = []

    def __call__(self, stage, arguments, env, on_event):
        request = Path(arguments[arguments.index('--output') + 1])
        self.calls.append(dict(stage=stage, arguments=list(arguments), rows=json.loads(request.read_text()), env=dict(env), request=request))
        request.unlink()
        return self.code, []


def quietly(function, *args, **kwargs):
    """Returns (result, combined output)."""
    output = io.StringIO()
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
        return function(*args, **kwargs), output.getvalue()


class Targets(TempRoot):
    def target(self, name, requested='auto'):
        return ubuntu_cli.export_target(self.root / name, requested)

    def test_suffix_selects_the_format(self):
        self.assertEqual(self.target('a.fbx'), ('fbx', self.root / 'a.fbx'))
        self.assertEqual(self.target('a.BVH'), ('bvh', self.root / 'a.BVH'))
        self.assertEqual(self.target('a.blend'), ('blend', self.root / 'a/a.blend'))

    def test_explicit_format_adds_only_a_missing_suffix(self):
        self.assertEqual(self.target('a', 'bvh'), ('bvh', self.root / 'a.bvh'))
        self.assertEqual(self.target('a.fbx', 'fbx'), ('fbx', self.root / 'a.fbx'))

    def test_bundle_is_a_folder_named_like_the_gui_target(self):
        self.assertEqual(self.target('a', 'blend'), ('blend', self.root / 'a/a.blend'))
        self.assertEqual(self.target('take.v2', 'blend'), ('blend', self.root / 'take.v2/take.v2.blend'))

    def test_unclear_or_conflicting_names_are_refused(self):
        for name, requested, message in (('a', 'auto', 'infer'), ('a.txt', 'auto', 'infer'), ('a.fbx', 'bvh', 'does not match'), ('a.bvh', 'blend', None)):
            if message is None:
                self.assertEqual(self.target(name, requested)[1], self.root / 'a.bvh/a.bvh.blend')
                continue
            with self.assertRaisesRegex(ubuntu_cli.ExportError, message):
                self.target(name, requested)

    def test_existing_bundle_folder_is_never_reused(self):
        existing = self.root / 'a'
        existing.mkdir()
        (existing / 'keep.txt').write_text('user data')
        with self.assertRaisesRegex(ubuntu_cli.ExportError, 'already exists'):
            self.target('a.blend')
        self.assertEqual((existing / 'keep.txt').read_text(), 'user data')

    def test_missing_destination_folder_is_reported(self):
        for name in ('none/a.fbx', 'none/a.blend'):
            with self.assertRaisesRegex(ubuntu_cli.ExportError, 'folder does not exist'):
                self.target(name)


class Requests(TempRoot):
    def export(self, *extra, output='out.fbx', plan=None, code=0):
        runner = Recorder(code)
        result, text = quietly(ubuntu_cli.command_export, self.args(*extra, output=output), runner, plan or self.plan())
        return result, runner, text

    def test_existing_fbx_command_keeps_its_exact_request(self):
        result, runner, text = self.export()
        self.assertEqual(result, 0)
        call = runner.calls[0]
        self.assertEqual(call['stage'], 'save_fbx')
        self.assertEqual(call['arguments'][:2], ['--project', str(self.project)])
        self.assertEqual(call['rows'], [dict(project=str(self.project), kind='initial', source=str(self.project / 'initial.bvh'), target=str(self.root / 'out.fbx'))])
        self.assertIn(f"Exported: {self.root / 'out.fbx'}", text)

    def test_fbx_options(self):
        _, runner, _ = self.export('--prepend-tpose', '--rig-name', 'Dancer', '--kind', 'corrected_initial')
        row = runner.calls[0]['rows'][0]
        self.assertEqual((row['prepend_tpose'], row['rig_name'], row['kind']), (True, 'Dancer', 'corrected_initial'))

    def test_bvh_request(self):
        _, runner, _ = self.export(output='out.bvh')
        self.assertEqual(runner.calls[0]['stage'], 'copy_bvh')
        self.assertEqual(runner.calls[0]['rows'][0]['bvh_profile'], 'generic')
        _, runner, _ = self.export('--bvh-profile', 'biped', '--prepend-tpose', output='out.bvh')
        row = runner.calls[0]['rows'][0]
        self.assertEqual((row['bvh_profile'], row['prepend_tpose']), ('biped', True))

    def test_blend_request_defaults_match_the_gui(self):
        with mock.patch.object(ubuntu_cli, 'require_blender'):
            _, runner, _ = self.export(output='bundle.blend')
        self.assertEqual(runner.calls[0]['stage'], 'save_blend')
        row = runner.calls[0]['rows'][0]
        self.assertEqual(row['target'], str(self.root / 'bundle/bundle.blend'))
        self.assertEqual((row['animate_camera'], row['background_refine'], row['camera_translation_scale']), (True, False, 1.))
        self.assertIn("('animate_camera',True),('background_refine',False),('camera_translation_scale',1.)", (ROOT / 'mocap/gui.py').read_text(encoding='utf-8'))

    def test_blend_camera_options(self):
        with mock.patch.object(ubuntu_cli, 'require_blender'):
            _, runner, _ = self.export('--background-refine', '--camera-translation-scale', '.5', '--prepend-tpose', output='bundle.blend')
            row = runner.calls[0]['rows'][0]
            self.assertEqual((row['background_refine'], row['camera_translation_scale'], row['prepend_tpose']), (True, .5, True))
            _, runner, _ = self.export('--no-animate-camera', '--format', 'blend', output='other')
            self.assertIs(runner.calls[0]['rows'][0]['animate_camera'], False)

    def test_scene_placement_reaches_every_format(self):
        scene = dict(root=str(self.root), transform=[[1.]])
        with mock.patch.object(ubuntu_cli, 'require_blender'):
            for name in ('out.fbx', 'out.bvh', 'out.blend'):
                _, runner, _ = self.export(output=name, plan=self.plan(scene=scene))
                self.assertEqual(runner.calls[0]['rows'][0]['scene'], scene, name)

    def test_misplaced_options_are_refused_before_any_runtime_work(self):
        def forbidden(*_):
            self.fail('The runtime check must not run.')
        for extra, output in ((('--bvh-profile', 'biped'), 'a.fbx'), (('--rig-name', 'A'), 'a.bvh'), (('--no-animate-camera',), 'a.bvh'),
                              (('--background-refine',), 'a.fbx'), (('--camera-translation-scale', '2'), 'a.fbx'),
                              (('--no-animate-camera', '--background-refine'), 'a.blend'), (('--camera-translation-scale', '101'), 'a.blend'),
                              (('--camera-translation-scale', 'nan'), 'a.blend')):
            with mock.patch.object(ubuntu_cli, 'require_blender'):
                result, runner, _ = self.export(*extra, output=output, plan=forbidden)
            self.assertEqual((result, runner.calls), (2, []), extra)

    def test_camera_options_need_a_model_with_saved_rotation(self):
        with mock.patch.object(ubuntu_cli, 'require_blender'):
            result, runner, text = self.export('--background-refine', output='a.blend', plan=self.plan(model='gvhmr'))
            self.assertEqual((result, runner.calls), (2, []))
            self.assertIn('FlowHMR and ExMR', text)
            result, _, _ = self.export(output='b.blend', plan=self.plan(model='gvhmr'))
            self.assertEqual(result, 0)

    def test_json_mode_ends_with_a_machine_readable_result(self):
        _, _, text = self.export('--json', output='out.bvh')
        self.assertEqual(json.loads(text.splitlines()[-1]), dict(type='exported', format='bvh', path=str(self.root / 'out.bvh')))

    def test_resource_safety_reaches_the_worker(self):
        _, runner, _ = self.export('--resource-safety', 'off')
        self.assertEqual(runner.calls[0]['env']['RGB_MOCAP_RESOURCE_SAFETY'], '0')


class Guards(TempRoot):
    def run_refused(self, plan):
        runner = Recorder()
        result, text = quietly(ubuntu_cli.command_export, self.args(), runner, plan)
        self.assertEqual((result, runner.calls), (2, []))
        self.assertFalse(list((self.root / 'cache').rglob('*.json')) if (self.root / 'cache').exists() else [])
        return text

    def test_stale_result_starts_no_worker_and_leaves_no_request(self):
        def stale(*_):
            raise ubuntu_cli.ExportError('initial is unavailable or has pending correction edits.')
        self.assertIn('pending correction edits', self.run_refused(stale))

    def test_missing_blender_keeps_fbx_and_bvh_available(self):
        def absent():
            raise RuntimeError('Character runtime is missing.')
        for machine, message in (('aarch64', 'no official Linux aarch64 build'), ('x86_64', 'runtime_character')):
            with mock.patch('mocap.retarget.runtime.blender', absent), mock.patch.object(ubuntu_cli, 'normalize_machine', return_value=machine):
                runner = Recorder()
                result, text = quietly(ubuntu_cli.command_export, self.args(output='a.blend'), runner, self.plan())
                self.assertEqual((result, runner.calls), (2, []))
                self.assertIn(message, text)
                self.assertIn('FBX and BVH export do not use Blender', text)
                for name in ('a.fbx', 'a.bvh'):
                    result, _ = quietly(ubuntu_cli.command_export, self.args(output=name), Recorder(), self.plan())
                    self.assertEqual(result, 0, name)

    def test_found_blender_passes(self):
        with mock.patch('mocap.retarget.runtime.blender', lambda: Path('/opt/blender')):
            ubuntu_cli.require_blender()

    def test_missing_shared_runtime_is_a_clear_error(self):
        with mock.patch.object(ubuntu_cli, 'interpreter', return_value=self.root / 'absent/python'), self.assertRaisesRegex(RuntimeError, 'setup_ubuntu.py'):
            ubuntu_cli.export_plan(self.project, 'initial', {})


class PlanHelper(TempRoot):
    """mocap/export_plan.py, with its numerical dependencies replaced."""

    def setUp(self):
        super().setUp()
        (self.project / 'project.json').write_text(json.dumps(dict(pipeline=dict(model='flowhmr'))))
        for target, value in (('mocap.motion_tools.selected_export', lambda project, kind: project / f'{kind}.bvh'),
                              ('mocap.motion_tools.require_supported_project', lambda project: SimpleNamespace(key='flowhmr'))):
            patcher = mock.patch(target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_single_actor_has_no_scene(self):
        self.assertEqual(export_plan.plan(self.project, 'initial'), dict(model='flowhmr', source=str(self.project / 'initial.bvh')))

    def test_stale_result_is_refused_in_the_guis_words(self):
        with mock.patch('mocap.motion_tools.selected_export', lambda *_: None), self.assertRaisesRegex(ValueError, 'Apply Correction / Prepare Preview'):
            export_plan.plan(self.project, 'corrected_initial')

    def test_unknown_result_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'Unknown result'):
            export_plan.plan(self.project, 'draft')

    def test_people_scene_uses_the_group_root_like_the_gui(self):
        group = self.root / 'group'
        take = group / 'person_1'
        take.mkdir(parents=True)
        (take / 'project.json').write_text(json.dumps(dict(people_root='..', pipeline=dict(model='flowhmr'))))
        seen = {}
        scene_module = ModuleType('mocap.scene_export')
        scene_module.context = lambda root, project: seen.update(root=root, project=project) or dict(root=str(root), transform=[[1.]])
        with mock.patch.dict(sys.modules, {'mocap.scene_export': scene_module}), mock.patch('mocap.people_store.load_people', lambda root: dict(people=[])):
            value = export_plan.plan(take, 'initial')
        self.assertEqual(seen, dict(root=group, project=take))
        self.assertEqual(value['scene']['root'], str(group))

    def test_main_reports_refusals_as_data(self):
        with mock.patch('mocap.motion_tools.selected_export', lambda *_: None):
            _, text = quietly(export_plan.main, [str(self.project), 'initial'])
        self.assertIn('Apply Correction', json.loads(text.split('@@PLAN@@')[1])['error'])

    def test_cli_runs_it_isolated_in_the_shared_runtime_and_reads_only_its_line(self):
        python = self.root / 'python'
        python.write_text('')
        replies = iter([(0, 'warning\n@@PLAN@@' + json.dumps(dict(model='exmr', source='s'))), (0, '@@PLAN@@' + json.dumps(dict(error='Not current'))), (1, 'ImportError: broken\n')])
        seen = []
        with mock.patch.object(ubuntu_cli, 'interpreter', return_value=python) as find, mock.patch.object(ubuntu_cli, 'capture', lambda command, env, timeout: seen.append(command) or next(replies)):
            self.assertEqual(ubuntu_cli.export_plan(self.project, 'initial', {}), dict(model='exmr', source='s'))
            for message in ('Not current', 'Could not check the export: ImportError: broken'):
                with self.assertRaisesRegex(ubuntu_cli.ExportError, message):
                    ubuntu_cli.export_plan(self.project, 'initial', {})
        find.assert_called_with(ubuntu_cli.ROOT, 'shared')
        self.assertEqual(seen[0][:3], [python, '-I', '-X'])
        self.assertIn('activate(sys.argv[1],"shared")', seen[0][5])

    def test_real_process_refuses_the_wrong_interpreter(self):
        other = self.root / 'unrelated-runtime'
        subprocess.run([sys.executable, '-I', '-m', 'venv', '--without-pip', str(other)], check=True, capture_output=True)
        wrong = other / 'bin/python'
        with mock.patch.object(ubuntu_cli, 'interpreter', return_value=wrong), self.assertRaisesRegex(ubuntu_cli.ExportError, 'Could not check the export'):
            ubuntu_cli.export_plan(self.project, 'initial', ubuntu_cli.job_environment())


class Cleanup(TempRoot):
    def fail_after(self, make, code=1, raises=None):
        def runner(stage, arguments, env, on_event):
            make(Path(arguments[arguments.index('--output') + 1]))
            if raises:
                raise raises
            return code, []
        return runner

    def requests(self):
        return list((self.root / 'cache/worker_jobs').glob('*'))

    def test_failed_file_export_removes_its_partial_and_keeps_the_old_output(self):
        target = self.root / 'out.bvh'
        target.write_text('previous export')
        made = []

        def make(request):
            made.append(target.with_name(f'.{target.name}.rgbmocap-{request.stem}.partial'))
            made[0].write_text('half written')
        result, _ = quietly(ubuntu_cli.command_export, self.args(output='out.bvh'), self.fail_after(make, 3), self.plan())
        self.assertEqual(result, 3)
        self.assertFalse(made[0].exists())
        self.assertEqual(target.read_text(), 'previous export')
        self.assertEqual(self.requests(), [])

    def test_failed_bundle_removes_only_its_owned_staging_folder(self):
        made = []

        def make(request):
            made.append(blend_support.Bundle(self.root / 'bundle/bundle.blend', request).work)
            (made[0] / 'video.mp4').write_text('copy')
        with mock.patch.object(ubuntu_cli, 'require_blender'):
            result, _ = quietly(ubuntu_cli.command_export, self.args(output='bundle.blend'), self.fail_after(make), self.plan())
        self.assertEqual(result, 1)
        self.assertFalse(made[0].exists())
        self.assertFalse((self.root / 'bundle').exists())
        self.assertEqual(self.requests(), [])

    def test_unowned_folder_at_the_staging_name_is_preserved(self):
        made = []

        def make(request):
            made.append(blend_support.partial(self.root / 'bundle', request))
            made[0].mkdir()
            (made[0] / 'user.txt').write_text('not ours')
        with mock.patch.object(ubuntu_cli, 'require_blender'):
            result, text = quietly(ubuntu_cli.command_export, self.args(output='bundle.blend'), self.fail_after(make), self.plan())
        self.assertEqual(result, 1)
        self.assertEqual((made[0] / 'user.txt').read_text(), 'not ours')
        self.assertIn('Could not remove', text)

    def test_cancellation_cleans_up_and_still_propagates(self):
        made = []

        def make(request):
            made.append(self.root / f'.out.fbx.rgbmocap-{request.stem}.partial')
            made[0].write_text('half written')
        with self.assertRaises(KeyboardInterrupt):
            quietly(ubuntu_cli.command_export, self.args(), self.fail_after(make, raises=KeyboardInterrupt()), self.plan())
        self.assertFalse(made[0].exists())
        self.assertEqual(self.requests(), [])

    def test_real_process_group_is_stopped_before_cleanup(self):
        class Guard:
            def __init__(self, pid, resource_safety):
                pass

            def check(self):
                return None

            def output(self):
                pass
        marker = self.root / 'child.pid'
        script = ("import json,pathlib,subprocess,sys,time;request=pathlib.Path(sys.argv[1]);target=pathlib.Path(json.loads(request.read_text())[0]['target']);"
                  "target.with_name('.'+target.name+'.rgbmocap-'+request.stem+'.partial').write_text('half');"
                  f"child=subprocess.Popen(['sleep','60']);pathlib.Path({str(marker)!r}).write_text(str(child.pid));"
                  "print('@@MOCAP@@'+json.dumps(dict(type='info',message='go')),flush=True);time.sleep(60)")

        def command(stage, *arguments):
            return [sys.executable, '-I', '-c', script, arguments[arguments.index('--output') + 1]]

        def interrupt(stage, event):
            raise KeyboardInterrupt
        printer = mock.patch.object(ubuntu_cli, 'EventPrinter', lambda raw: interrupt)
        with mock.patch('mocap.paths.worker_command', command), printer, self.assertRaises(KeyboardInterrupt):
            quietly(ubuntu_cli.command_export, self.args(), functools.partial(ubuntu_cli.run_worker, guard_factory=Guard), self.plan())
        self.assertEqual(list(self.root.glob('.out.fbx.*')), [])
        self.assertEqual(self.requests(), [])
        pid = int(marker.read_text())
        for _ in range(100):
            state = Path(f'/proc/{pid}/stat')
            if not state.exists() or state.read_text().split(')')[1].split()[0] == 'Z':
                return
            time.sleep(.05)
        os.kill(pid, 9)
        self.fail('The worker child survived cancellation.')


def synthetic_bvh(path):
    """A valid committed-style 52-bone BVH: only the worker's parser and channels matter here."""
    from mocap.biped_bvh import bone_names
    names = ['Hips', *[name for name in bone_names() if name != 'Hips']]
    lines = ['HIERARCHY']
    for depth, name in enumerate(names):
        indent = '  ' * depth
        channels = 'CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation' if depth == 0 else 'CHANNELS 3 Zrotation Xrotation Yrotation'
        lines += [f"{indent}{'ROOT' if depth == 0 else 'JOINT'} {name}", indent + '{', f'{indent}  OFFSET 0 0.1 0', f'{indent}  {channels}']
    lines += ['  ' * len(names) + 'End Site', '  ' * len(names) + '{', '  ' * len(names) + '  OFFSET 0 0.1 0', '  ' * len(names) + '}']
    lines += ['  ' * depth + '}' for depth in reversed(range(len(names)))]
    lines += ['MOTION', 'Frames: 3', 'Frame Time: 0.033333']
    for frame in range(3):
        lines.append(' '.join(['0.5', str(.9 + frame / 100), '0.1', *['5.0', '-3.0', '2.0'] * 52]))
    path.write_text('\n'.join(lines) + '\n')


@unittest.skipUnless(HAS_NUMERICS, 'Needs NumPy and SciPy from a pinned environment.')
class RealBvhWorker(TempRoot):
    """The CLI's request through the real worker entry and export_copy; only the receipt lookup is stubbed."""

    def setUp(self):
        super().setUp()
        self.source = self.project / 'initial.bvh'
        synthetic_bvh(self.source)
        patcher = mock.patch('mocap.export_copy.selected_export', lambda project, kind: self.source)
        patcher.start()
        self.addCleanup(patcher.stop)

    def export(self, *extra, output='out.bvh', plan=None):
        events = []

        def runner(stage, arguments, env, on_event):
            from mocap import source_export_worker
            saved = sys.argv
            sys.argv = ['mocap.worker', stage, *arguments]
            try:
                with mock.patch('mocap.compute_policy.configure_worker_cpu'):
                    code = source_export_worker.main(lambda kind, **value: events.append(dict(type=kind, **value)))
            finally:
                sys.argv = saved
            return code, events
        result, _ = quietly(ubuntu_cli.command_export, self.args(*extra, output=output), runner, plan or self.plan())
        self.assertEqual(result, 0)
        self.assertEqual(events[-1]['stage'], 'copy_bvh')
        self.assertEqual(list((self.root / 'cache/worker_jobs').glob('*')), [])
        self.assertEqual(list(self.root.glob('.out.bvh.*')), [])
        return (self.root / output).read_text()

    def test_generic_is_an_exact_copy(self):
        self.assertEqual(self.export(), self.source.read_text())

    def test_existing_output_is_replaced_atomically(self):
        (self.root / 'out.bvh').write_text('previous')
        self.assertEqual(self.export(), self.source.read_text())

    def test_biped_renames_the_exported_copy_only(self):
        original = self.source.read_text()
        text = self.export('--bvh-profile', 'biped')
        self.assertIn('JOINT LeftHip', text)
        self.assertNotIn('LeftUpLeg', text)
        self.assertIn('Frames: 3', text)
        self.assertEqual(self.source.read_text(), original)

    def test_prepended_tpose_adds_one_frame(self):
        text = self.export('--prepend-tpose')
        self.assertIn('Frames: 4', text)
        self.assertEqual(len(text[text.index('Frame Time'):].splitlines()) - 1, 4)

    def test_biped_with_tpose(self):
        text = self.export('--bvh-profile', 'biped', '--prepend-tpose')
        self.assertIn('Frames: 4', text)
        self.assertIn('JOINT LeftHip', text)

    def test_worker_refuses_a_source_that_changed_after_the_plan(self):
        moved = self.project / 'other.bvh'
        moved.write_text('newer generation')
        request = self.root / 'request.json'
        request.write_text(json.dumps([dict(project=str(self.project), kind='initial', source=str(moved), target=str(self.root / 'out.bvh'))]))
        from mocap.export_copy import run
        with self.assertRaisesRegex(ValueError, 'Selected target motion changed'):
            run(request)
        self.assertFalse((self.root / 'out.bvh').exists())
        self.assertFalse(request.exists())


class WorkerEntries(TempRoot):
    def test_blend_worker_accepts_the_cli_arguments(self):
        seen = {}
        module = ModuleType('mocap.blend_export')
        module.run = lambda request, emit: seen.update(request=request) or 1
        emitted = []
        saved = sys.argv
        request = self.root / 'r.json'
        sys.argv = ['mocap.worker', 'save_blend', '--project', str(self.project), '--output', str(request)]
        try:
            from mocap import blend_export_worker
            with mock.patch.dict(sys.modules, {'mocap.blend_export': module}), mock.patch('mocap.compute_policy.configure_worker_cpu'):
                self.assertEqual(blend_export_worker.main(lambda kind, **value: emitted.append(dict(type=kind, **value))), 0)
        finally:
            sys.argv = saved
        self.assertEqual(seen['request'], request)
        self.assertEqual((emitted[-1]['stage'], emitted[-1]['files']), ('save_blend', 1))

    def test_every_stage_routes_to_the_shared_runtime_and_is_listed(self):
        from mocap.runtime_registry import profile_for
        for stage in ubuntu_cli.EXPORT_STAGES.values():
            self.assertEqual(profile_for(stage, ('--project', str(self.project))), 'shared')
            self.assertIn(stage, ubuntu_cli.KNOWN_STAGES)
        progress = (ROOT / 'mocap/worker_progress.py').read_text(encoding='utf-8')
        self.assertIn("'copy_bvh'", progress)
        self.assertIn("'save_blend'", progress)

    def test_partial_names_follow_the_workers(self):
        # export_copy inlines the formula; fbx_export delegates to blend_support.partial (upstream 9617dae).
        self.assertIn("temporary=target.with_name('.'+target.name+'.rgbmocap-'+request.stem+'.partial')",
                      (ROOT / 'mocap/export_copy.py').read_text(encoding='utf-8'), 'mocap/export_copy.py')
        self.assertIn("temporary=partial(target,request)",
                      (ROOT / 'mocap/fbx_export.py').read_text(encoding='utf-8'), 'mocap/fbx_export.py')
        self.assertIn("from .blend_support import", (ROOT / 'mocap/fbx_export.py').read_text(encoding='utf-8'), 'mocap/fbx_export.py')
        self.assertEqual(blend_support.partial(self.root / 'b', 'x/abc.json'), self.root / '.b.rgbmocap-abc.partial')

    def test_no_numerical_or_receipt_code_is_imported_by_the_cli(self):
        source = (ROOT / 'tools/ubuntu_cli.py').read_text(encoding='utf-8')
        self.assertFalse(re.search(r'^\s*(import|from)\s+(numpy|scipy|torch|cv2)\b', source, re.M))
        # Stale results must be refused by selected_export, never by the CLI reading receipts itself.
        self.assertNotIn('_mixamo.json', source)


class Documentation(unittest.TestCase):
    def test_guide_covers_the_formats(self):
        text = (ROOT / 'docs/UBUNTU.md').read_text(encoding='utf-8')
        for fragment in ('--output motion.bvh', '--bvh-profile biped', '--output bundle.blend', 'save_blend', 'copy_bvh'):
            self.assertIn(fragment, text)
        self.assertNotIn('`export` writes FBX. BVH and Blender bundles use request files', text)


if __name__ == '__main__':
    unittest.main()
