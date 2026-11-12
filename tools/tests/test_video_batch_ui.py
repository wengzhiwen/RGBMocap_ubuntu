"""GUI batch lifecycle with synthetic inputs and simulated worker completions.

QT_QPA_PLATFORM=offscreen .venv/bin/python -m unittest discover -s tools/tests -p test_video_batch_ui.py -v
No inference, GPU work or user project writes.
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
    from PySide6.QtCore import QObject, Signal
    from PySide6.QtWidgets import QApplication, QDialog, QLabel, QProgressBar, QWidget
    from mocap.video_batch_ui import BatchExportDialog, VideoBatch, trimmed_frames
    HAS_QT = True
except ImportError:
    HAS_QT = False


if HAS_QT:
    class FakeWindow(QWidget):
        def __init__(self, root):
            super().__init__()
            self.video_path = root / 'first.mp4'
            self.info = dict(fps=30., frames=300)
            self.source_offset = 0
            self.slider = SimpleNamespace(clip_in=0, clip_out=300, set_clip=self.clip)
            self.model_choice = SimpleNamespace(currentData=lambda: 'flowhmr')
            self.settings = dict(result='corrected_initial', flow_options=dict(hamer=True))
            self.camera_tools = SimpleNamespace(background_options=lambda: {}, restore_background=Mock())
            self.shortcuts = []
            self.busy = False
            self.video_readable = True
            self._preview_export = None
            self.status = QLabel()
            self.task_info = QLabel()
            self.progress = QProgressBar()
            self.log = SimpleNamespace(append=Mock())
            self.people_state = None
            self.project = root / 'project'
            self.project.mkdir()
            self.active_result = 'corrected_initial'
            self.load_video = Mock(side_effect=self.load)
            self.run_full_single = Mock(side_effect=self.run)
            self.submit_fbx_exports = Mock(side_effect=self.export)
            self.start_preview_export = Mock(side_effect=self.preview)
            self.update_readiness = Mock()
            self.cancel_job = Mock()
            self.ready_for_preview = False

        def clip(self, first, end):
            self.slider.clip_in, self.slider.clip_out = first, end

        def load(self, path):
            self.video_path = path
            # Loading a source resets options in the real GUI too.
            self.settings = {}
            self.video_readable = False

        def run(self):
            self.busy = True

        def export(self, files):
            self.busy = True
            self._motion_export_targets = [str(row[3]) for row in files]

        def preview(self, path, **kwargs):
            self._preview_export = FakePreview(self)
            self._preview_export.destination = path
            return self._preview_export

        def stop_playback(self):
            pass

        def workspace_settings(self):
            return self.settings

        def restore_workspace_settings(self, value):
            self.settings = value

        def processing_range(self):
            return self.slider.clip_in / 30, self.slider.clip_out / 30

        def current_export(self):
            return self.project / 'motion.bvh'

        def preview_export_available(self):
            return self.ready_for_preview

    class FakePreview(QObject):
        completed = Signal(bool, str, bool)

        def cancel(self):
            pass


@unittest.skipUnless(HAS_QT, 'Requires the pinned GUI environment')
class BatchLifecycle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.w = FakeWindow(self.root)
        self.batch = None

    def tearDown(self):
        if self.batch:
            self.batch.timer.stop()
            self.batch.stopping = True
            self.w.busy = False
            self.w._preview_export = None
            self.batch.finish()
            self.batch.dialog.accept()
        self.app.processEvents()
        self.w.close()
        self.temp.cleanup()

    def make_batch(self, count=2, preview=False, fmt='fbx'):
        videos = [self.w.video_path] + [self.root / f'video{i}.mp4' for i in range(1, count)]
        self.batch = VideoBatch(self.w, videos, dict(folder=str(self.root), format=fmt, preview=preview))
        self.w._video_batch = self.batch
        self.batch.start()
        self.batch.timer.stop()  # tests drive transitions explicitly
        return self.batch

    def settle(self, failed=False, cancelled=False, message='done'):
        self.w.busy = False
        self.batch.jobs_finished(failed, cancelled, message)

    def next_video(self):
        self.app.processEvents()

    def test_reconstruct_export_then_next_and_report(self):
        b = self.make_batch()
        b.tick()
        self.assertEqual(b.phase, 'reconstructing')
        self.assertEqual(b.output.name, '0001-first')
        self.settle()
        self.assertEqual(b.phase, 'exporting')
        self.w.load_video.assert_not_called()
        self.settle()
        self.next_video()
        self.assertEqual(b.phase, 'opening')
        self.assertEqual(self.w.video_path.name, 'video1.mp4')
        self.assertEqual(b.output.name, '0002-video1')
        self.w.video_readable = True
        b.tick()
        self.assertTrue(self.w.settings['flow_options']['hamer'])
        self.settle()
        self.settle()
        self.next_video()
        report = json.loads((b.folder / 'batch-report.json').read_text())
        self.assertEqual([r['status'] for r in report['videos']], ['done', 'done'])
        self.assertEqual(len(report['videos'][0]['exports']), 1)
        self.assertNotEqual(report['videos'][0]['exports'], report['videos'][1]['exports'])
        self.assertIsNone(self.w._video_batch)

    def test_inference_failure_is_recorded_and_skips_export(self):
        b = self.make_batch()
        b.tick()
        self.settle(failed=True, message='CUDA worker failed: sample error')
        self.next_video()
        self.assertEqual(b.index, 1)
        self.assertEqual(b.rows[0]['error'], 'CUDA worker failed: sample error')
        self.w.submit_fbx_exports.assert_not_called()

    def test_export_failure_continues(self):
        b = self.make_batch()
        b.tick()
        self.settle()
        self.settle(failed=True, message='Disk full')
        self.next_video()
        self.assertEqual(b.index, 1)
        self.assertEqual(b.rows[0]['error'], 'Disk full')

    def test_failed_video_waits_for_process_tree_cleanup_before_next(self):
        from PySide6.QtCore import QProcess
        from PySide6.QtTest import QTest
        b = self.make_batch()
        b.tick()
        self.w.cancel_process = SimpleNamespace(state=lambda: QProcess.Running)
        self.settle(failed=True, message='Resource stop')
        self.next_video()
        self.assertEqual(b.index, 0)
        self.w.cancel_process = None
        QTest.qWait(150)
        self.assertEqual(b.index, 1)

    def test_failed_first_video_can_advance(self):
        self.w.info = None
        self.w.video_readable = False
        b = self.make_batch()
        b.fail('Cannot open video')
        self.next_video()
        self.assertEqual(b.index, 1)
        self.assertEqual(b.rows[0]['status'], 'failed')

    def test_cancellation_waits_for_worker_and_skips_remaining(self):
        b = self.make_batch()
        b.tick()
        b.dialog.reject()  # Esc follows the same cancellation path
        self.assertTrue(b.stopping)
        self.w.cancel_job.assert_called_once()
        b.tick()
        self.assertIs(self.w._video_batch, b)
        self.settle(cancelled=True)
        b.tick()
        self.assertIsNone(self.w._video_batch)
        self.assertEqual([r['status'] for r in b.rows], ['cancelled', 'skipped'])
        self.w.load_video.assert_not_called()

    def test_cancel_between_files_preserves_done(self):
        b = self.make_batch()
        b.tick()
        self.settle()
        self.settle()
        b.cancel()
        self.next_video()
        self.assertEqual([r['status'] for r in b.rows], ['done', 'skipped'])

    def test_preview_waits_then_blocks_next_video_until_finished(self):
        b = self.make_batch(preview=True)
        b.tick()
        self.settle()
        self.settle()
        self.assertEqual(b.phase, 'waiting_preview')
        b.tick()
        self.w.start_preview_export.assert_not_called()
        self.w.ready_for_preview = True
        b.tick()
        self.assertEqual(b.phase, 'preview')
        self.w.load_video.assert_not_called()
        self.w._preview_export.completed.emit(True, 'Saved', False)
        self.w._preview_export = None
        self.next_video()
        self.assertEqual(b.index, 1)
        self.assertEqual(len(b.rows[0]['exports']), 2)
        self.assertTrue(b.dialog.isModal())

    def test_preview_failure_keeps_motion_export_and_continues(self):
        b = self.make_batch(preview=True)
        b.tick()
        self.settle()
        self.settle()
        self.w.ready_for_preview = True
        b.tick()
        self.w._preview_export.completed.emit(False, 'Encoder failed', False)
        self.w._preview_export = None
        self.next_video()
        self.assertEqual(b.rows[0]['error'], 'Encoder failed')
        self.assertEqual(len(b.rows[0]['exports']), 1)
        self.assertEqual(b.index, 1)

    def test_cancel_preview_waits_for_encoder_cleanup(self):
        b = self.make_batch(preview=True)
        b.tick()
        self.settle()
        self.settle()
        self.w.ready_for_preview = True
        b.tick()
        preview = self.w._preview_export
        preview.cancel = Mock()
        b.cancel()
        preview.cancel.assert_called_once()
        b.tick()
        self.assertIs(self.w._video_batch, b)
        self.w._preview_export = None
        b.tick()
        self.assertIsNone(self.w._video_batch)
        self.assertEqual(b.rows[1]['status'], 'skipped')

    def test_all_checked_people_and_takes_use_separate_names(self):
        w = self.w
        w.people_state = {'people': []}
        w.people_root = self.root
        w.checked_people = lambda: [dict(id=1, name='Same name', takes=[dict(project='person1/take1'), dict(project='person1/take2')]),
                                   dict(id=2, name='Same name', takes=[dict(project='person2/take1')])]
        b = self.make_batch(count=1)
        b.tick()
        with patch('mocap.motion_tools.selected_export', side_effect=lambda project, kind: project / 'motion.bvh'):
            self.settle()
        files = w.submit_fbx_exports.call_args.args[0]
        self.assertEqual(len(files), 3)
        self.assertEqual(len({row[3] for row in files}), 3)

    def test_timeout_continues(self):
        b = self.make_batch()
        self.w.video_readable = False
        b.wait_started = time.monotonic() - 61
        b.tick()
        self.next_video()
        self.assertEqual(b.index, 1)
        self.assertIn('Timed out', b.rows[0]['error'])

    def test_trim_in_seconds_and_no_overlap(self):
        self.assertEqual(trimmed_frames(dict(fps=24, frames=240), (1, 20)), (24, 240))
        self.assertEqual(trimmed_frames(dict(fps=24, frames=240), None), (0, 240))
        with self.assertRaises(ValueError):
            trimmed_frames(dict(fps=24, frames=24), (2, 3))

    def test_formats_and_options_are_passed_to_guarded_export(self):
        for fmt in ('fbx', 'bvh', 'blend'):
            with self.subTest(fmt=fmt):
                b = self.make_batch(count=1, fmt=fmt)
                b.tick()
                self.settle()
                files = self.w.submit_fbx_exports.call_args.args[0]
                self.assertEqual(files[0][3].suffix, '.' + fmt)
                self.assertEqual(self.w._source_export_options['format'], fmt)
                self.settle()
                self.next_video()
                b.dialog.accept()
                self.batch = None

    def test_existing_outputs_survive_new_runs(self):
        existing = self.root / 'motion.fbx'
        existing.write_text('keep')
        b = self.make_batch()
        self.assertNotEqual(b.folder, self.root)
        self.assertEqual(existing.read_text(), 'keep')


@unittest.skipUnless(HAS_QT, 'Requires the pinned GUI environment')
class RealGuiIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        from mocap.gui import MainWindow
        self.w = MainWindow()

    def tearDown(self):
        self.w._video_batch = None
        self.w.busy = False
        self.w.close()
        self.app.processEvents()

    def test_menu_and_file_count_and_selection_order(self):
        w = self.w
        self.assertIn('Batch Processing', [a.text() for a in w.menuBar().actions()])
        self.assertEqual([a.text() for a in w.batch_menu.actions()], ['Select Videos...', 'Setup Export...'])
        with patch('mocap.video_batch_ui.QFileDialog.getOpenFileNames', return_value=(['/tmp/a.mp4', '/tmp/b.mp4', '/tmp/a.mp4'], '')), patch.object(w, 'load_video') as load:
            w.select_batch_videos()
        load.assert_called_once_with(Path('/tmp/a.mp4'))
        self.assertEqual(w.run_button.text(), 'Reconstruct 3D (2)')
        with patch.object(w, 'start_video_batch') as start:
            w.run_full()
        start.assert_called_once()

    def test_export_dialog_roundtrip(self):
        with tempfile.TemporaryDirectory() as folder:
            saved = dict(folder=folder, format='bvh', bvh_profile='biped', prepend_tpose=True, preview=True)
            dialog = BatchExportDialog(self.w, saved)
            self.assertEqual(dialog.options(), saved)
            dialog.accept()
            self.assertEqual(dialog.result(), QDialog.Accepted)
            dialog.deleteLater()

    def test_actual_job_end_notifies_batch_after_export(self):
        w = self.w
        receiver = Mock()
        w._video_batch = receiver
        w.current_stage = 'copy_bvh'
        w.jobs = []
        w.failed = False
        w.cancelled = False
        w.next_job()
        self.app.processEvents()
        receiver.jobs_finished.assert_called_once()
        self.assertEqual(receiver.jobs_finished.call_args.args[:2], (False, False))
        self.assertFalse(w.busy)

    def test_failed_to_start_notifies_batch(self):
        from PySide6.QtCore import QProcess
        receiver = Mock()
        self.w._video_batch = receiver
        self.w.process_error(QProcess.FailedToStart)
        self.app.processEvents()
        receiver.jobs_finished.assert_called_once()
        self.assertTrue(receiver.jobs_finished.call_args.args[0])

    def test_preview_completion_preserves_shared_modal_and_clears_controller(self):
        from mocap.preview_export_ui import PreviewExport
        from mocap.video_batch_ui import BatchProgress
        w = self.w
        w.video_path = Path('/tmp/synthetic-batch-video.mp4')
        w.video_view.decoder = SimpleNamespace(path=str(w.video_path))
        w.info = dict(fps=30, frames=10, duration=10/30)
        w.slider.setRange(0, 9)
        w.slider.set_clip(0, 10)
        dialog = BatchProgress(w, [w.video_path])
        dialog.show()
        try:
            with patch.object(w, 'seek'):
                controller = PreviewExport(w, '/tmp/synthetic-preview.mp4', progress_dialog=dialog)
                w._preview_export = controller
                controller.writer = SimpleNamespace(isRunning=lambda: False, succeeded=True, error=None)
                observed = Mock()
                controller.completed.connect(observed)
                controller.finish()
                observed.assert_called_once()
                self.assertTrue(observed.call_args.args[0])
                self.assertIsNone(w._preview_export)
                self.assertTrue(dialog.isVisible())
                self.assertFalse(dialog.allow_close)
        finally:
            w.video_view.decoder = None
            dialog.allow_close = True
            dialog.accept()
            dialog.deleteLater()

    def test_real_decoder_skips_broken_first_video_and_opens_next(self):
        import cv2
        import numpy as np
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            good = root / 'next.avi'
            writer = cv2.VideoWriter(str(good), cv2.VideoWriter_fourcc(*'MJPG'), 24, (64, 64))
            self.assertTrue(writer.isOpened())
            for _ in range(12):
                writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
            writer.release()
            w = self.w
            missing = root / 'broken.mp4'
            w.video_path = missing
            w._batch_videos = [missing, good]
            w._batch_export_options = dict(folder=folder, format='fbx', preview=False)
            try:
                with patch.object(w, 'run_full_single', side_effect=lambda: setattr(w, 'busy', True)) as run:
                    w.start_video_batch()
                    b = w._video_batch
                    deadline = time.monotonic() + 5
                    while not run.called and time.monotonic() < deadline:
                        self.app.processEvents()
                        time.sleep(.005)
                    run.assert_called_once()
                    self.assertEqual(b.index, 1)
                    self.assertEqual(b.rows[0]['status'], 'failed')
                    self.assertEqual(w.video_path, good)
                    self.assertEqual(w.info['frames'], 12)
                    self.assertEqual(w.run_button.text(), 'Reconstruct 3D (2)')
            finally:
                w.busy = False
                b.stopping = True
                b.finish()
                b.dialog.accept()


if __name__ == '__main__':
    unittest.main()
