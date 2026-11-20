"""Parallel GPU batch: protocol, GPU parsing, dialog wiring and child orchestration.

QT_QPA_PLATFORM=offscreen .venv/bin/python -m unittest discover -s tools/tests -p test_batch_parallel.py -v
No GPU work: children run tools/tests/batch_child_stub.py scenarios.
"""
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from PySide6.QtCore import QObject, QTimer, Signal
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication, QCheckBox, QComboBox, QLabel, QProgressBar, QWidget
    from mocap import batch_parallel, batch_protocol
    from mocap.batch_child import apply_request_window_state, run_child
    from mocap.video_batch_ui import BatchExportDialog
    HAS_QT = True
except ImportError:
    HAS_QT = False

STUB = Path(__file__).resolve().parent / 'batch_child_stub.py'
SAMPLE_SMI = (
    '0, GPU-aaa-111, NVIDIA RTX PRO 4000, 595.99.02, 00000000:01:00.0, 24467 MiB\n'
    '1, GPU-bbb-222, NVIDIA RTX PRO 4000, 595.99.02, 00000000:03:00.0, 24467 MiB\n'
    '2, GPU-ccc-333, GeForce RTX 4060 Ti, 595.99.02, 00000000:1C:00.0, 8188 MiB\n')


def stub_command(scenario):
    def command(request, request_path):
        return [sys.executable, '-I', '-B', str(STUB),
                '--request', str(request_path), '--scenario', scenario]
    return command


def wait_until(predicate, timeout=20.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        QTest.qWait(50)
    return predicate()


def completed(result, code):
    return result, code


if HAS_QT:
    class FakePreviewController(QObject):
        completed = Signal(bool, str, bool)

        def __init__(self):
            super().__init__()
            self.done = False
            self.path = None

        def cancel(self):
            pass

    class FakeParallelWindow(QWidget):
        """The subset of MainWindow the parallel controller touches."""

        def __init__(self, root):
            super().__init__()
            self.video_path = root / 'first.mp4'
            self.info = dict(fps=30., frames=300)
            self.source_offset = 0
            self.slider = SimpleNamespace(clip_in=30, clip_out=150, set_clip=self.clip)
            self.model_choice = QComboBox()
            self.model_choice.addItem('FlowHMR', 'flowhmr')
            self.model_choice.addItem('GVHMR', 'gvhmr')
            self.compute_mode = QComboBox()
            self.compute_mode.addItem('Automatic', 'auto')
            self.compute_mode.addItem('Low Memory', 'low_memory')
            self.resource_safety = QCheckBox('Resource Safety')
            self.settings = dict(result='corrected_initial', flow_options=dict(hamer=True))
            self.camera_tools = SimpleNamespace(background_options=lambda: {'blur': 2},
                                                restore_background=Mock())
            self.shortcuts = []
            self.busy = False
            self.video_readable = True
            self.people_state = None
            self.people_root = None
            self.status = QLabel()
            self.task_info = QLabel()
            self.progress = QProgressBar()
            self.log = SimpleNamespace(append=Mock())
            self._video_batch = None
            self._preview_export = None
            self.project = None
            self.ready_for_preview = False
            self.last_preview = None
            self.load_project = Mock(side_effect=self.open_project)
            self.start_preview_export = Mock(side_effect=self.preview_start)
            self.update_readiness = Mock()
            self.stop_playback = Mock()

        def open_project(self, path):
            self.project = path
            self.ready_for_preview = True
            self._preview_export = None
            self.last_preview = None

        def preview_start(self, path, **kwargs):
            controller = FakePreviewController()
            controller.path = path
            self.last_preview = controller
            self._preview_export = controller
            return controller

        def preview_export_available(self):
            return self.ready_for_preview and self._preview_export is None

        def workspace_settings(self):
            return self.settings

        def processing_range(self):
            return self.slider.clip_in / 30, self.slider.clip_out / 30

        def clip(self, first, end):
            self.slider.clip_in, self.slider.clip_out = first, end


@unittest.skipUnless(HAS_QT, 'Requires the pinned GUI environment')
class Protocol(unittest.TestCase):
    def test_request_roundtrip_and_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'request.json'
            request = batch_protocol.build_request(
                video='a.mp4', number=2, output_dir=folder, model='flowhmr',
                options=dict(format='fbx', preview=False), settings=dict(result='initial'),
                background=dict(blur=1), trim=[1.5, 9.5], compute_mode='low_memory',
                resource_safety=False, manual_people=None,
                gpu=dict(index=1, uuid='GPU-bbb-222', name='X'), result=str(path) + '.out')
            batch_protocol.write_request(path, request)
            self.assertEqual(batch_protocol.read_request(path), request)

    def test_read_request_rejects_other_versions_and_missing_keys(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'request.json'
            path.write_text(json.dumps(dict(version=99)), encoding='utf-8')
            with self.assertRaises(batch_protocol.ProtocolError):
                batch_protocol.read_request(path)
            path.write_text(json.dumps(dict(version=1, video='a')), encoding='utf-8')
            with self.assertRaises(batch_protocol.ProtocolError):
                batch_protocol.read_request(path)

    def test_result_tolerates_missing_and_garbage(self):
        with tempfile.TemporaryDirectory() as folder:
            missing = Path(folder) / 'nope.json'
            self.assertIsNone(batch_protocol.read_result(missing))
            garbage = Path(folder) / 'bad.json'
            garbage.write_text('not json', encoding='utf-8')
            self.assertIsNone(batch_protocol.read_result(garbage))
            bad_version = Path(folder) / 'old.json'
            bad_version.write_text(json.dumps(dict(version=0)), encoding='utf-8')
            self.assertIsNone(batch_protocol.read_result(bad_version))

    def test_child_command_shape(self):
        command = batch_protocol.child_command(Path('/tmp/req.json'))
        self.assertEqual(command[0], sys.executable)
        for flag in ('-I', '-X', 'utf8', '-u', '-B'):
            self.assertIn(flag, command)
        self.assertTrue(command[-1].endswith('req.json'))
        self.assertTrue(any(part.endswith('app.py') for part in command))
        self.assertIn('--batch-child', command)

    def test_parse_line_filters_noise_and_append_tail_bounds(self):
        self.assertIsNone(batch_protocol.parse_line('Qt warning: fonts'))
        event = batch_protocol.parse_line(batch_protocol.EVENT_PREFIX + '{"type": "phase", "phase": "done"}')
        self.assertEqual(event, dict(type='phase', phase='done'))
        self.assertIsNone(batch_protocol.parse_line(batch_protocol.EVENT_PREFIX + 'not json'))
        tail = batch_protocol.append_tail('', 'x' * (batch_protocol.TAIL_BYTES + 10))
        self.assertLessEqual(len(tail), batch_protocol.TAIL_BYTES + 1)


@unittest.skipUnless(HAS_QT, 'Requires the pinned GUI environment')
class GpuParsing(unittest.TestCase):
    def rows(self):
        with patch.object(batch_parallel.subprocess, 'run') as run:
            run.return_value = SimpleNamespace(stdout=SAMPLE_SMI)
            return batch_parallel.query_gpus()

    def test_query_gpus_parses_rows(self):
        rows = self.rows()
        self.assertEqual([r['index'] for r in rows], [0, 1, 2])
        self.assertEqual(rows[0]['uuid'], 'GPU-aaa-111')
        self.assertEqual(rows[2]['memory_mib'], 8188)
        self.assertIn('8 GB', batch_parallel.gpu_label(rows[2]))

    def test_query_gpus_raises_without_tool(self):
        with patch.object(batch_parallel.subprocess, 'run', side_effect=OSError('missing')):
            with self.assertRaises(batch_parallel.GpuQueryError):
                batch_parallel.query_gpus()

    def test_parent_scope_translation(self):
        rows = self.rows()
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '2'}):
            self.assertEqual(batch_parallel.parent_scope(rows), [rows[2]])
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '1,0'}):
            self.assertEqual(batch_parallel.parent_scope(rows), [rows[1], rows[0]])
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': 'GPU-aaa'}):
            self.assertEqual(batch_parallel.parent_scope(rows), [rows[0]])
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '9'}):
            with self.assertRaises(batch_parallel.GpuQueryError):
                batch_parallel.parent_scope(rows)
        environment = dict(os.environ)
        environment.pop('CUDA_VISIBLE_DEVICES', None)
        with patch.dict(os.environ, environment, clear=True):
            self.assertIsNone(batch_parallel.parent_scope(rows))


@unittest.skipUnless(HAS_QT, 'Requires the pinned GUI environment')
class DialogWiring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def dialog(self, saved, gpus):
        parent = QWidget()
        self.addCleanup(parent.deleteLater)
        with patch.object(batch_parallel, 'query_gpus', return_value=gpus), \
                patch.object(batch_parallel, 'parent_scope', return_value=None):
            dialog = BatchExportDialog(parent, saved)
        self.addCleanup(dialog.deleteLater)
        return dialog

    def test_two_checked_gpus_keep_preview_with_sequential_note(self):
        gpus = [dict(index=0, uuid='GPU-a', name='A', memory_mib=24576),
                dict(index=1, uuid='GPU-b', name='B', memory_mib=24576),
                dict(index=2, uuid='GPU-c', name='C', memory_mib=8188)]
        dialog = self.dialog(dict(preview=True), gpus)
        dialog.preview.setChecked(True)
        boxes = [box for _, box in dialog.gpu_boxes]
        boxes[0].setChecked(True)
        boxes[1].setChecked(True)
        self.assertTrue(dialog.preview.isEnabled())
        self.assertTrue(dialog.preview.isChecked())
        selected = dialog.options()['parallel_gpus']
        self.assertEqual([g['uuid'] for g in selected], ['GPU-a', 'GPU-b'])
        self.assertIn('preview videos sequentially in the main window', dialog.parallel_note.text())
        self.assertNotIn('watch RAM headroom', dialog.parallel_note.text())
        boxes[2].setChecked(True)
        self.assertIn('watch RAM headroom', dialog.parallel_note.text())
        self.assertEqual(len(dialog.options()['parallel_gpus']), 3)

    def test_single_gpu_keeps_sequential_defaults(self):
        gpus = [dict(index=0, uuid='GPU-a', name='A', memory_mib=8188),
                dict(index=1, uuid='GPU-b', name='B', memory_mib=8188)]
        dialog = self.dialog(dict(preview=True), gpus)
        [box for _, box in dialog.gpu_boxes][0].setChecked(True)
        self.assertTrue(dialog.preview.isEnabled())
        self.assertEqual(dialog.options()['parallel_gpus'],
                         [dict(index=0, uuid='GPU-a', name='A')])

    def test_saved_selection_roundtrip(self):
        gpus = [dict(index=0, uuid='GPU-a', name='A', memory_mib=8188),
                dict(index=1, uuid='GPU-b', name='B', memory_mib=8188)]
        dialog = self.dialog(dict(parallel_gpus=[dict(uuid='GPU-b')]), gpus)
        checked = [gpu['uuid'] for gpu, box in dialog.gpu_boxes if box.isChecked()]
        self.assertEqual(checked, ['GPU-b'])

    def test_enumeration_failure_hides_the_group(self):
        parent = QWidget()
        self.addCleanup(parent.deleteLater)
        with patch.object(batch_parallel, 'query_gpus',
                          side_effect=batch_parallel.GpuQueryError('no nvidia-smi')):
            dialog = BatchExportDialog(parent, dict(preview=True))
        self.addCleanup(dialog.deleteLater)
        self.assertEqual(dialog.gpu_boxes, [])
        self.assertTrue(dialog.preview.isEnabled())
        self.assertEqual(dialog.options()['parallel_gpus'], [])


@unittest.skipUnless(HAS_QT, 'Requires the pinned GUI environment')
class ParallelOrchestration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.w = FakeParallelWindow(self.root)
        self.controller = None
        self.videos = [self.root / 'one.mp4', self.root / 'two.mp4', self.root / 'three.mp4']

    def tearDown(self):
        if self.controller is not None and self.controller.phase != 'finished':
            self.controller.cancel()
            wait_until(lambda: self.controller.phase == 'finished', 10)
        self.w.close()
        self.temp.cleanup()

    def make_controller(self, videos, scenarios, gpus=None, preview=False):
        gpus = gpus or [dict(index=0, uuid='GPU-a', name='A'),
                        dict(index=1, uuid='GPU-b', name='B')]
        scenarios = list(scenarios) + ['ok'] * (len(videos) - len(scenarios))

        def command(request, request_path):
            scenario = scenarios[int(request['number']) - 1]
            return [sys.executable, '-I', '-B', str(STUB),
                    '--request', str(request_path), '--scenario', scenario]
        options = dict(folder=str(self.root), format='fbx', preview=preview, auto_people=False)
        self.controller = batch_parallel.ParallelVideoBatch(
            self.w, videos, options, gpus, child_command=command)
        self.w._video_batch = self.controller
        return self.controller

    def drive_previews(self, controller, success=True, message=''):
        """Complete every preview the parent window starts during the pass."""
        handled = 0
        deadline = time.monotonic() + 20
        while controller.phase != 'finished' and time.monotonic() < deadline:
            QTest.qWait(50)
            fake = self.w.last_preview
            if fake is not None and not fake.done:
                fake.done = True
                fake.completed.emit(success, message, False)
                self.w._preview_export = None
                handled += 1
        return handled

    def test_two_gpus_finish_all_videos_and_write_merged_report(self):
        controller = self.make_controller(self.videos[:2], ['ok', 'ok'])
        controller.start()
        self.assertTrue(wait_until(lambda: len(controller.children) == 2)
                        or controller.phase == 'finished')
        # The GPU column and one status row per GPU are visible while running.
        for row, gpu in enumerate(['GPU 0', 'GPU 1']):
            self.assertEqual(controller.dialog.table.item(row, 1).text(), gpu)
        self.assertEqual(sorted(controller.dialog.gpu_rows), [0, 1])
        self.assertTrue(wait_until(lambda: controller.phase == 'finished'))
        self.assertEqual([r['status'] for r in controller.rows], ['done', 'done'])
        self.assertEqual(controller.rows[0]['exports'], [str(controller.folder / '0001-one' / 'motion.fbx')])
        self.assertEqual([r['gpu'] for r in controller.rows], ['GPU 0', 'GPU 1'])
        report = json.loads((controller.folder / 'batch-report.json').read_text(encoding='utf-8'))
        self.assertEqual([r['status'] for r in report['videos']], ['done', 'done'])
        self.assertEqual([g['uuid'] for g in report['parallel']['gpus']], ['GPU-a', 'GPU-b'])
        self.assertFalse(report['export']['preview'])
        self.assertIsNone(self.w._video_batch)

    def test_preview_pass_renders_previews_sequentially_in_the_parent(self):
        controller = self.make_controller(self.videos[:2], ['ok', 'ok'], preview=True)
        controller.start()
        handled = self.drive_previews(controller)
        self.assertEqual(handled, 2)
        self.assertEqual(controller.phase, 'finished')
        self.assertEqual([r['status'] for r in controller.rows], ['done', 'done'])
        for index, name in enumerate(('0001-one', '0002-two')):
            self.assertIn(str(controller.folder / name / 'preview.mp4'),
                          controller.rows[index]['exports'])
        # The parent opened each finished project in turn, in queue order.
        self.assertEqual(self.w.load_project.call_count, 2)
        report = json.loads((controller.folder / 'batch-report.json').read_text(encoding='utf-8'))
        self.assertTrue(report['export']['preview'])
        self.assertIsNone(self.w._video_batch)

    def test_failed_preview_marks_the_row_failed_and_keeps_exports(self):
        controller = self.make_controller(self.videos[:2], ['ok', 'ok'], preview=True)
        controller.start()
        handled = self.drive_previews(controller, success=False, message='encoder failed')
        self.assertEqual(handled, 2)
        self.assertEqual([r['status'] for r in controller.rows], ['failed', 'failed'])
        self.assertEqual(controller.rows[0]['error'], 'encoder failed')
        # The completed motion exports remain attached to the failed rows.
        self.assertIn(str(controller.folder / '0001-one' / 'motion.fbx'),
                      controller.rows[0]['exports'])
        self.assertNotIn(str(controller.folder / '0001-one' / 'preview.mp4'),
                         controller.rows[0]['exports'])

    def test_crashed_child_fails_the_row_and_dispatch_continues(self):
        controller = self.make_controller(self.videos, ['crash', 'ok', 'ok'])
        controller.start()
        self.assertTrue(wait_until(lambda: controller.phase == 'finished'))
        self.assertEqual([r['status'] for r in controller.rows], ['failed', 'done', 'done'])
        self.assertIn('exited with code 1', controller.rows[0]['error'])
        self.assertIn('stub noise', controller.rows[0]['error'])

    def test_failed_result_is_recorded_from_the_result_file(self):
        controller = self.make_controller(self.videos[:2], ['fail', 'ok'])
        controller.start()
        self.assertTrue(wait_until(lambda: controller.phase == 'finished'))
        self.assertEqual([r['status'] for r in controller.rows], ['failed', 'done'])
        self.assertEqual(controller.rows[0]['error'], 'stub failure')

    def test_concurrency_is_bounded_by_selected_gpus(self):
        marker = self.root / 'overlap.log'
        with patch.dict(os.environ, {'BATCH_STUB_OVERLAP': str(marker)}):
            controller = self.make_controller(self.videos, ['slow-ok', 'slow-ok', 'slow-ok'])
            controller.start()
            self.assertTrue(wait_until(lambda: controller.phase == 'finished'))
        lines = [line for line in marker.read_text().splitlines() if line]
        running, peak = 0, 0
        for line in lines:
            running += 1 if line.startswith('+') else -1
            peak = max(peak, running)
        self.assertEqual(peak, 2)
        self.assertEqual([r['status'] for r in controller.rows], ['done', 'done', 'done'])

    def test_cancel_marks_running_cancelled_and_queued_skipped(self):
        controller = self.make_controller(self.videos, ['ignore-term', 'ignore-term', 'ok'])
        controller.start()
        self.assertTrue(wait_until(lambda: len(controller.children) == 2), 'children did not start')
        controller.cancel()
        self.assertTrue(wait_until(lambda: controller.phase == 'finished', 15))
        self.assertEqual([r['status'] for r in controller.rows], ['cancelled', 'cancelled', 'skipped'])

    def test_requires_two_gpus(self):
        with self.assertRaises(ValueError):
            batch_parallel.ParallelVideoBatch(self.w, self.videos, dict(folder=str(self.root)),
                                              [dict(index=0, uuid='GPU-a', name='A')])


@unittest.skipUnless(HAS_QT, 'Requires the pinned GUI environment')
class Routing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        from mocap.gui import MainWindow
        self.w = MainWindow()

    def tearDown(self):
        self.w._video_batch = None
        self.w.busy = False
        self.w.close()
        self.app.processEvents()
        self.temp.cleanup()

    def test_parallel_options_route_to_the_parallel_controller(self):
        self.w._batch_videos = [self.root / 'a.mp4', self.root / 'b.mp4']
        self.w._batch_export_options = dict(folder=str(self.root), format='fbx',
                                            parallel_gpus=[dict(index=0, uuid='GPU-a', name='A'),
                                                           dict(index=1, uuid='GPU-b', name='B')])
        with patch('mocap.batch_parallel.ParallelVideoBatch') as parallel:
            self.w.start_video_batch()
        parallel.assert_called_once()
        parallel.return_value.start.assert_called_once()
        self.assertIs(self.w._video_batch, parallel.return_value)

    def test_sequential_options_keep_the_sequential_controller(self):
        self.w._batch_videos = [self.root / 'a.mp4', self.root / 'b.mp4']
        self.w._batch_export_options = dict(folder=str(self.root), format='fbx')
        import mocap.video_batch_ui as ui
        with patch.object(ui, 'VideoBatch') as sequential:
            self.w.start_video_batch()
        sequential.assert_called_once()
        sequential.return_value.start.assert_called_once()


@unittest.skipUnless(HAS_QT, 'Requires the pinned GUI environment')
class ChildEntry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_apply_request_window_state_without_config_write(self):
        from mocap import resource_safety
        window = FakeParallelWindow(self.root)
        request = dict(model='gvhmr', compute_mode='low_memory', resource_safety=False)
        with patch.object(resource_safety, 'save') as save, \
                patch.dict(os.environ, {resource_safety.ENV: '1'}):
            apply_request_window_state(window, request)
            save.assert_not_called()
            self.assertEqual(window.model_choice.currentData(), 'gvhmr')
            self.assertEqual(window.compute_mode.currentData(), 'low_memory')
            self.assertFalse(window.resource_safety.isChecked())
            self.assertEqual(os.environ[resource_safety.ENV], '0')

    def test_apply_request_rejects_unknown_model(self):
        window = FakeParallelWindow(self.root)
        with self.assertRaises(ValueError):
            apply_request_window_state(window, dict(model='nonexistent', compute_mode='auto',
                                                    resource_safety=True))

    def test_run_child_processes_one_video_and_exits_ok(self):
        from mocap.video_batch_ui import VideoBatch

        class ChildWindow(FakeParallelWindow):
            def __init__(self, root):
                super().__init__(root)
                self.project = root / 'project'
                self.active_result = 'initial'
                self.load_video = Mock(side_effect=self.load)
                self.run_full_single = Mock(side_effect=self.run)
                self.submit_fbx_exports = Mock(side_effect=self.export)
                self.input_jobs = Mock()
                self.cancel_job = Mock()

            def load(self, path):
                self.video_path = path  # decode completes instantly in the fake

            def run(self):
                self.busy = True

            def export(self, files):
                self.busy = True
                self._motion_export_targets = [str(row[3]) for row in files]

            def current_export(self):
                return self.project / 'motion.bvh'

            def restore_workspace_settings(self, value):
                self.settings = value

            def shutdown_preview(self):
                pass

        holder = {}

        def factory():
            window = ChildWindow(self.root)
            holder['window'] = window

            def settle():
                window.busy = False
                window._video_batch.jobs_finished(False, False, 'done')

            QTimer.singleShot(400, settle)
            QTimer.singleShot(450, settle)
            return window

        request_path = self.root / 'child-request.json'
        result_path = self.root / 'child-result.json'
        request = batch_protocol.build_request(
            video=self.root / 'clip.mp4', number=1, output_dir=self.root, model='flowhmr',
            options=dict(folder=str(self.root), format='fbx', preview=True, auto_people=False),
            settings=dict(result='initial'), background={}, trim=None, compute_mode='auto',
            resource_safety=True, manual_people=None,
            gpu=dict(index=2, uuid='GPU-c', name='C'), result=result_path)
        batch_protocol.write_request(request_path, request)
        code = run_child(request_path, window_factory=factory)
        self.assertEqual(code, batch_protocol.EXIT_OK)
        result = batch_protocol.read_result(result_path)
        self.assertTrue(result['ok'])
        self.assertEqual(result['status'], 'done')
        window = holder['window']
        self.assertEqual(len(result['exports']), 1)
        self.assertIn('motion.fbx', result['exports'][0])
        self.assertIsNone(window._video_batch)


if __name__ == '__main__':
    unittest.main()
