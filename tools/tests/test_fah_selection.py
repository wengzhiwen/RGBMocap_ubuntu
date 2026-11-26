"""FAH Finger Refinement is one choice per motion model: FlowHMR and FAB each have their own.

Both models select the optional FAH finger pass through their own command flag
and their own settings-page checkbox, and neither model's choice reaches the
other's worker. CPU only: no model, runtime, weights or project data, and the
workers are replaced by a recorder. Run with:
    QT_QPA_PLATFORM=offscreen .venv/bin/python -m unittest discover -s tools/tests -p test_fah_selection.py -v
"""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]

import ubuntu_cli  # noqa: E402

GAP = 'FAH model weights'
LINUX = sys.platform.startswith('linux')


class Recorder:
    """Stands in for every worker: notes each stage and the FAH variable it was given; starts nothing."""

    def __init__(self, project):
        self.project, self.calls = project, []

    def __call__(self, stage, arguments, env, printer):
        self.calls.append((stage, env.get('RGB_MOCAP_FAH')))
        return 0, [dict(type='prepared', project=str(self.project))] if stage == 'prepare' else []


class Quiet:
    """Capture what one command prints."""

    def __enter__(self):
        self.out, self.err = io.StringIO(), io.StringIO()
        self.stack = contextlib.ExitStack()
        self.stack.enter_context(contextlib.redirect_stdout(self.out))
        self.stack.enter_context(contextlib.redirect_stderr(self.err))
        return self

    def __exit__(self, *error):
        return self.stack.__exit__(*error)


class CommandLine(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.project = self.root / 'project'
        self.project.mkdir()
        self.video = self.root / 'clip.mp4'
        self.video.write_bytes(b'')

    def launch(self, command, *extra, gaps=()):
        """One command with every model reported ready and the FAH runtime reporting ``gaps``.

        Returns the exit code, the worker recorder and the printed text. Nothing is started.
        """
        args = ubuntu_cli.parser().parse_args([command, '--video', str(self.video), *map(str, extra)])
        recorder = Recorder(self.project)
        with mock.patch.object(ubuntu_cli, 'fah_gaps', return_value=list(gaps)) as fah, \
                mock.patch('mocap.models.missing', return_value=[]) as models, \
                mock.patch('mocap.fab_runtime.missing', return_value=[]) as fab, Quiet() as quiet:
            code = args.handler(args, recorder)
        self.readiness = dict(fah=fah.call_count, models=models.call_count, fab=fab.call_count)
        return code, recorder, quiet.out.getvalue(), quiet.err.getvalue()

    def test_both_commands_pass_their_own_choice_to_every_worker(self):
        from mocap.motion_tools import MOTION_TOOLS
        for command in ('flow', 'fab'):
            for extra, value in (((), '0'), (('--no-fah',), '0'), (('--fah',), '1')):
                with self.subTest(command=command, extra=extra):
                    code, recorder, out, err = self.launch(command, *extra)
                    self.assertEqual(code, 0, msg=err)
                    self.assertIn('Finished. Project:', out)
                    self.assertTrue(len(recorder.calls) >= 3, msg=str(recorder.calls))
                    # One variable for the whole run of that model: prepare, the motion stages and export.
                    self.assertEqual(recorder.calls[0][0], 'prepare')
                    self.assertEqual([value for _, value in recorder.calls], [value] * len(recorder.calls))
                    self.assertTrue(recorder.calls[-1][0].endswith('export'), msg=str(recorder.calls[-1]))
                    self.assertEqual(self.readiness['fah'] == 0, extra != ('--fah',))
                    self.assertEqual(self.readiness['fab'], 1 if command == 'fab' else 0)

    def parse(self, command, *extra):
        return ubuntu_cli.parser().parse_args([command, '--video', str(self.video), *map(str, extra)])

    def test_an_unready_fah_runtime_stops_before_any_worker(self):
        for command in ('flow', 'fab'):
            with self.subTest(command=command):
                code, recorder, out, err = self.launch(command, '--fah', gaps=[GAP])
                self.assertEqual((code, recorder.calls), (2, []))
                self.assertIn(GAP, err)
                # The plan of a dry run names the missing part and still starts nothing.
                code, recorder, out, err = self.launch(command, '--fah', '--dry-run', gaps=[GAP])
                self.assertEqual((code, recorder.calls), (0, []))
                self.assertEqual(json.loads(out)['missing_fah'], [GAP])
                # Without the pass the same missing runtime is nobody's business.
                code, recorder, out, err = self.launch(command, gaps=[GAP])
                self.assertEqual(code, 0, msg=err)
                self.assertTrue(recorder.calls, msg=command)
                self.assertEqual({value for _, value in recorder.calls}, {'0'})
                self.assertEqual(self.readiness['fah'], 0)

    def test_each_command_runs_its_own_stages_only(self):
        from mocap.motion_tools import MOTION_TOOLS
        for command in ('flow', 'fab'):
            with self.subTest(command=command):
                code, recorder, _, err = self.launch(command, '--fah')
                self.assertEqual(code, 0, msg=err)
                stages = [stage for stage, _ in recorder.calls]
                planned = (ubuntu_cli.flow_jobs(self.parse(command, '--fah'), self.project) if command == 'flow'
                           else [(stage, []) for stage in MOTION_TOOLS['fab'].stages() if stage != 'prepare'])
                self.assertEqual(stages, ['prepare'] + [stage for stage, _ in planned])
                if command == 'flow':
                    self.assertIn('flow_batch', stages)                 # features and generation share one worker
                    self.assertNotIn('fab_reconstruct', stages)
                else:
                    self.assertIn('fab_reconstruct', stages)
                    self.assertFalse(any(stage.startswith('flow') for stage in stages), msg=str(stages))

    def test_flow_refuses_two_finger_passes(self):
        code, recorder, out, err = self.launch('flow', '--fah', '--hamer')
        self.assertEqual((code, recorder.calls), (2, []))
        self.assertIn('not both', err)

    def test_only_flow_and_fab_carry_the_choice(self):
        for command in ('flow', 'fab'):
            for extra, value in (((), False), (('--no-fah',), False), (('--fah',), True)):
                with self.subTest(command=command, extra=extra):
                    args = self.parse(command, *extra)
                    self.assertEqual(bool(args.fah), value)
                    self.assertEqual(ubuntu_cli.job_environment(args, base={})['RGB_MOCAP_FAH'], '1' if value else '0')
        other = ubuntu_cli.parser().parse_args(['export', '--project', 'project', '--output', 'motion.bvh'])
        self.assertFalse(hasattr(other, 'fah'))
        self.assertEqual(ubuntu_cli.job_environment(other, base={})['RGB_MOCAP_FAH'], '0')


try:
    from PySide6.QtWidgets import QApplication
    from mocap.fah_runtime import supported as fah_supported
    from mocap.gui import MainWindow
    HAS_QT = True
except ImportError:
    HAS_QT = False


@unittest.skipUnless(HAS_QT, 'Requires the pinned GUI environment')
class ModelChoice(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        folder = tempfile.TemporaryDirectory()
        cls.addClassCleanup(folder.cleanup)
        cls.root = Path(folder.name)

    def setUp(self):
        self.w = MainWindow()

    def tearDown(self):
        self.w._video_batch = None
        self.w.busy = False
        self.w.close()
        self.app.processEvents()

    def pages(self):
        return {self.w.model_pages.widget(index).property('motion_tool'): index
                for index in range(self.w.model_pages.count())}

    def select(self, key):
        choice = self.w.model_choice
        self.assertNotEqual(choice.findData(key), -1, msg=key)
        choice.blockSignals(True)
        choice.setCurrentIndex(choice.findData(key))
        choice.blockSignals(False)
        self.w.select_model()
        self.app.processEvents()

    def require_fab(self):
        # FAB is registered on Linux; elsewhere its page and checkbox are simply never selected.
        if 'fab' not in self.pages():
            if LINUX:
                self.fail('FAB has no settings page')
            self.skipTest('FAB is registered on Linux only')

    def test_each_model_shows_its_own_settings_page(self):
        pages = self.pages()
        self.assertEqual(len(pages), self.w.model_pages.count())        # one page per model
        self.assertTrue({'gvhmr', 'exmr', 'flowhmr'} <= set(pages), msg=str(sorted(pages)))
        self.require_fab()
        for key in ('exmr', 'flowhmr', 'fab'):
            with self.subTest(model=key):
                self.select(key)
                self.assertEqual(self.w.model_choice.currentData(), key)
                self.assertEqual(self.w.model_pages.currentIndex(), pages[key])
                self.assertEqual(self.w.model_pages.currentWidget().property('motion_tool'), key)
                self.assertFalse(self.w.model_pages.isHidden())

    def test_the_fab_page_holds_the_fab_finger_pass(self):
        self.require_fab()
        self.select('fab')
        self.assertIs(self.w.model_pages.currentWidget(), self.w.fab_page)
        self.assertEqual(self.w.fab_fah.text(), 'FAH Finger Refinement')
        parent = self.w.fab_fah.parentWidget()
        while parent is not None and parent is not self.w.fab_page:
            parent = parent.parentWidget()
        self.assertIs(parent, self.w.fab_page)            # on the FAB page, not on FlowHMR's
        self.assertEqual(self.w.fab_fah.isHidden(), not fah_supported())
        self.assertEqual(self.w.flow_fah.isHidden(), not fah_supported())

    def test_each_model_keeps_its_own_choice(self):
        self.require_fab()
        self.select('flowhmr')
        self.w.flow_fah.setChecked(True)
        self.assertIs(self.w.fah_selected(), True)
        self.assertFalse(self.w.fab_fah.isChecked())
        self.assertIs(self.w.workspace_settings()['flow_options']['fah'], True)
        self.assertEqual(self.w.workspace_settings()['fab_options'], {'fah': False})
        self.select('fab')
        self.assertIs(self.w.fah_selected(), False)                 # FAB does not inherit FlowHMR's choice
        self.w.fab_fah.setChecked(True)
        self.assertIs(self.w.fah_selected(), True)
        self.w.flow_fah.setChecked(False)
        self.assertIs(self.w.fah_selected(), True)                  # and FlowHMR's change does not reach FAB
        self.assertEqual(self.w.workspace_settings()['fab_options'], {'fah': True})
        self.assertIs(self.w.workspace_settings()['flow_options']['fah'], False)
        for other in ('exmr', 'gvhmr', 'mamma', 'prompthmr'):
            self.assertIs(self.w.fah_selected(other), False)

    def test_choices_reset_and_restore_with_the_other_settings(self):
        self.require_fab()
        self.w.flow_fah.setChecked(True)
        self.w.fab_fah.setChecked(True)
        settings = self.w.workspace_settings()
        self.w.restore_workspace_settings({'flow_options': dict(settings['flow_options'], fah=False),
                                           'fab_options': {'fah': False}})
        self.assertFalse(self.w.flow_fah.isChecked() or self.w.fab_fah.isChecked())
        self.w.restore_workspace_settings({})                        # projects saved before the pass existed
        self.assertFalse(self.w.flow_fah.isChecked() or self.w.fab_fah.isChecked())
        self.w.restore_workspace_settings({'flow_options': dict(settings['flow_options'], fah=True),
                                           'fab_options': {'fah': True}})
        self.assertIs(self.w.flow_fah.isChecked(), fah_supported())
        self.assertIs(self.w.fab_fah.isChecked(), fah_supported())
        # Anything but an explicit saved "on" stays off.
        self.w.restore_workspace_settings({'flow_options': dict(settings['flow_options'], fah='1'),
                                           'fab_options': {'fah': 1}})
        self.assertFalse(self.w.flow_fah.isChecked() or self.w.fab_fah.isChecked())

    def test_a_new_video_clears_both_choices(self):
        self.require_fab()
        self.w.flow_fah.setChecked(True)
        self.w.fab_fah.setChecked(True)
        with mock.patch.object(self.w.video_view, 'open'):
            self.w.load_video(self.root / 'clip.mp4')
        self.assertFalse(self.w.flow_fah.isChecked() or self.w.fab_fah.isChecked())


if __name__ == '__main__':
    unittest.main()
