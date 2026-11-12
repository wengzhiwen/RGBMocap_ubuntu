"""Sequential video automation over the ordinary GUI jobs and guarded exports."""
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import time

from PySide6.QtCore import QObject, QTimer, Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QDialog, QFileDialog, QHBoxLayout, QLabel,
    QLineEdit, QMenu, QProgressBar, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from .source_export_dialog import SourceExportDialog
from .storage import atomic_json


class BatchExportDialog(SourceExportDialog):
    def __init__(self, window, saved):
        super().__init__(window)
        self.setWindowTitle('Setup Batch Export')
        self.whole_scene.setChecked(False)
        self.whole_scene.hide()
        panel = QWidget(self)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel('Output folder'))
        row = QHBoxLayout()
        self.folder = QLineEdit(saved.get('folder', ''))
        row.addWidget(self.folder)
        browse = QPushButton('Browse...')
        browse.clicked.connect(self.browse)
        row.addWidget(browse)
        layout.addLayout(row)
        self.preview = QCheckBox('Export Preview Video')
        self.preview.setChecked(saved.get('preview', False))
        layout.addWidget(self.preview)
        note = QLabel('Each run creates a new batch folder with per-video exports and a progress/error report. '
                      'The first video supplies the shared model settings and trim in seconds. '
                      'An untrimmed first video processes every video in full; shorter videos are clamped. '
                      'People are tracked automatically in subsequent videos.')
        note.setWordWrap(True)
        layout.addWidget(note)
        self.layout().insertWidget(self.layout().count() - 1, panel)
        self.format.setCurrentIndex(max(0, self.format.findData(saved.get('format', 'fbx'))))
        self.profile.setCurrentIndex(max(0, self.profile.findData(saved.get('bvh_profile', 'generic'))))
        self.tpose.setChecked(saved.get('prepend_tpose', False))
        self.animate_camera.setChecked(saved.get('animate_camera', True))
        self.background_refine.setChecked(saved.get('background_refine', False))
        self.translation_scale.setValue(saved.get('camera_translation_scale', 1.))

    def browse(self):
        folder = QFileDialog.getExistingDirectory(self, 'Batch Export Folder', self.folder.text())
        if folder:
            self.folder.setText(folder)

    def accept(self):
        if not self.folder.text().strip() or not Path(self.folder.text().strip()).is_dir():
            self.folder.setFocus()
            self.folder.setPlaceholderText('Choose an existing output folder')
            return
        super().accept()

    def options(self):
        return dict(super().options(), folder=self.folder.text().strip(), preview=self.preview.isChecked())


class BatchProgress(QDialog):
    cancel_requested = Signal()

    def __init__(self, window, videos):
        super().__init__(window)
        self.setWindowTitle('Batch Processing')
        self.setWindowModality(Qt.ApplicationModal)
        self.setMinimumSize(620, 400)
        self.allow_close = False
        layout = QVBoxLayout(self)
        self.summary = QLabel()
        layout.addWidget(self.summary)
        self.overall = QProgressBar()
        self.overall.setRange(0, len(videos))
        layout.addWidget(self.overall)
        self.table = QTableWidget(len(videos), 2)
        self.table.setHorizontalHeaderLabels(['Video', 'Status'])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setColumnWidth(0, 280)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        for i, path in enumerate(videos):
            item = QTableWidgetItem(path.name)
            item.setToolTip(str(path))
            self.table.setItem(i, 0, item)
            self.table.setItem(i, 1, QTableWidgetItem('Pending'))
        layout.addWidget(self.table)
        self.label = QLabel('Starting...')
        self.label.setTextFormat(Qt.PlainText)
        self.label.setWordWrap(True)
        layout.addWidget(self.label)
        self.progress = QProgressBar()
        layout.addWidget(self.progress)
        self.cancel_button = QPushButton('Cancel Batch')
        self.cancel_button.clicked.connect(self.request_close)
        layout.addWidget(self.cancel_button)

    def request_close(self):
        if self.allow_close:
            self.accept()
        else:
            self.cancel_requested.emit()

    def reject(self):
        self.request_close()

    def closeEvent(self, event):
        if self.allow_close:
            event.accept()
        else:
            self.cancel_requested.emit()
            event.ignore()


def trimmed_frames(info, trim):
    """A shared source-time interval; full-length input remains full-length."""
    if trim is None:
        return 0, info['frames']
    first = max(0, round(trim[0] * info['fps']))
    end = min(info['frames'], round(trim[1] * info['fps']))
    if first >= end:
        raise ValueError('The shared trim does not overlap this video.')
    return first, end


class VideoBatch(QObject):
    def __init__(self, window, videos, options):
        super().__init__(window)
        self.window = window
        self.videos = list(videos)
        self.options = deepcopy(options)
        self.settings = deepcopy(window.workspace_settings())
        self.background = window.camera_tools.background_options()
        self.trim = None
        if window.info:
            first, end = window.processing_range()
            if not (window.source_offset == 0 and window.slider.clip_in == 0
                    and window.slider.clip_out == window.info['frames']):
                self.trim = (first, end)
        self.model = window.model_choice.currentData()
        self.index = -1
        self.phase = 'starting'
        self.stopping = False
        self.rows = [dict(video=str(p), status='pending', error='', exports=[]) for p in videos]
        # Exclusive folders protect same-stem inputs and all previously saved output.
        self.folder = Path(options['folder']) / ('Batch_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        self.folder.mkdir()
        self.dialog = BatchProgress(window, videos)
        self.dialog.cancel_requested.connect(self.cancel)
        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.tick)
        self.saved_shortcuts = [s.isEnabled() for s in window.shortcuts]
        self.saved_drops = window.acceptDrops()
        self.previous_export_options = getattr(window, '_source_export_options', {})

    def start(self):
        w = self.window
        w.stop_playback()
        for shortcut in w.shortcuts:
            shortcut.setEnabled(False)
        w.setAcceptDrops(False)
        self.dialog.show()
        self.timer.start()
        self.advance()

    def save_report(self):
        try:
            atomic_json(self.folder / 'batch-report.json', dict(version=1, model=self.model,
                settings=self.settings, trim_seconds=self.trim, export=self.options,
                status=self.phase, videos=self.rows))
        except OSError as error:
            self.window.log.append('Cannot save batch report: ' + str(error))

    def set_phase(self, phase):
        self.phase = phase
        self.wait_started = time.monotonic()
        self.rows[self.index]['status'] = phase
        self.dialog.table.item(self.index, 1).setText(phase.replace('_', ' ').title())
        self.dialog.table.scrollToItem(self.dialog.table.item(self.index, 0))
        self.dialog.progress.setRange(0, 0)
        self.dialog.label.setText(phase.replace('_', ' ').title())
        if not self.stopping:
            self.dialog.cancel_button.setEnabled(True)
        self.save_report()

    def advance(self):
        if self.stopping:
            self.finish()
            return
        # On Windows the worker can exit before taskkill finishes reaping its tree.
        # A resource-stop failure must not overlap that cleanup with the next video.
        from PySide6.QtCore import QProcess
        pending = getattr(self.window, 'cancel_process', None)
        if pending is not None and pending.state() != QProcess.NotRunning:
            QTimer.singleShot(100, self.advance)
            return
        self.index += 1
        if self.index == len(self.videos):
            self.finish()
            return
        w = self.window
        self.dialog.summary.setText(f'Video {self.index + 1}/{len(self.videos)} · {self.videos[self.index].name}')
        self.set_phase('opening')
        try:
            self.output = self.folder / f'{self.index + 1:04d}-{self.videos[self.index].stem}'
            self.output.mkdir()
            self.keep_first = self.index == 0 and w.video_readable and Path(w.video_path).resolve() == self.videos[0]
            if not self.keep_first:
                w.load_video(self.videos[self.index])
        except Exception as error:
            self.fail(str(error))

    def tick(self):
        w = self.window
        if self.stopping:
            pending = getattr(w, 'cancel_process', None)
            from PySide6.QtCore import QProcess
            if not w.busy and getattr(w, '_preview_export', None) is None and (pending is None or pending.state() == QProcess.NotRunning):
                self.finish()
            return
        try:
            if self.phase == 'opening' and w.video_readable and w.info:
                w.restore_workspace_settings(deepcopy(self.settings))
                w.camera_tools.restore_background(self.background)
                # Keep the first video's own person/take selection and source trim.
                if not self.keep_first:
                    w.slider.set_clip(*trimmed_frames(w.info, self.trim))
                self.set_phase('reconstructing')
                w.run_full_single()
                if not w.busy:
                    self.fail(w.status.text() or 'Reconstruction did not start.')
            elif self.phase == 'waiting_preview' and w.preview_export_available():
                self.set_phase('preview')
                controller = w.start_preview_export(self.output / 'preview.mp4', resume=False, progress_dialog=self.dialog)
                controller.completed.connect(self.preview_finished)
            elif self.phase in ('opening', 'waiting_preview') and time.monotonic() - self.wait_started > 60:
                self.fail('Timed out waiting for video or motion preview.')
            if self.phase in ('reconstructing', 'exporting'):
                self.dialog.label.setText(w.status.text() + '\n' + w.task_info.text())
                self.dialog.progress.setRange(w.progress.minimum(), w.progress.maximum())
                self.dialog.progress.setValue(w.progress.value())
        except Exception as error:
            # A live worker must settle before advancing, even if a GUI refresh fails.
            if w.busy:
                self.pending_error = str(error)
                w.cancel_job()
            else:
                self.fail(str(error))

    def jobs_finished(self, failed, cancelled, message):
        if self.phase not in ('reconstructing', 'exporting'):
            return
        if self.stopping:
            return  # tick waits for the cancellation helper too
        error = getattr(self, 'pending_error', None)
        self.pending_error = None
        if failed or cancelled or error:
            self.fail(error or message)
            return
        try:
            if self.phase == 'reconstructing':
                self.rows[self.index]['project'] = str(self.window.project)
                self.export_motion()
            else:
                self.rows[self.index]['exports'] = list(self.window._motion_export_targets)
                if self.options.get('preview'):
                    self.set_phase('waiting_preview')
                else:
                    self.complete_item()
        except Exception as error:
            self.fail(str(error))

    def export_motion(self):
        w = self.window
        from .motion_tools import selected_export
        from .people_store import safe_child
        from .person_names import export_stem
        files = []
        fmt = self.options['format']
        if w.people_state:
            for person in w.checked_people():
                if not person.get('takes'):
                    raise ValueError(person['name'] + ': no reconstructed motion.')
                for i, take in enumerate(person['takes'], 1):
                    project = safe_child(w.people_root, take['project'])
                    name = export_stem(person['name']) + f'_{person["id"]}_take_{i:02d}'
                    files.append((project, w.active_result, selected_export(project, w.active_result), self.output / (name + '.' + fmt)))
        else:
            files.append((w.project, w.active_result, w.current_export(), self.output / ('motion.' + fmt)))
        if not files or any(source is None for _, _, source, _ in files):
            raise ValueError('Selected motion is unavailable or correction is pending.')
        w._source_export_options = deepcopy(self.options)
        self.set_phase('exporting')
        w.submit_fbx_exports(files)
        if not w.busy:
            self.fail(w.status.text() or 'Motion export did not start.')

    def preview_finished(self, success, message, cancelled):
        if self.stopping:
            return
        if success:
            self.rows[self.index]['exports'].append(str(self.output / 'preview.mp4'))
            self.complete_item()
        else:
            self.fail(message)

    def complete_item(self):
        self.set_phase('done')
        self.dialog.overall.setValue(self.index + 1)
        QTimer.singleShot(0, self.advance)

    def fail(self, message):
        if self.phase in ('failed', 'done', 'finished', 'cancelled') or self.stopping:
            return
        self.rows[self.index]['error'] = message
        self.window.log.append(f'Batch video {self.index + 1}: {self.videos[self.index]}\n{message}')
        self.set_phase('failed')
        item = self.dialog.table.item(self.index, 1)
        item.setText('Failed: ' + message)
        item.setToolTip(message)
        self.dialog.overall.setValue(self.index + 1)
        QTimer.singleShot(0, self.advance)

    def cancel(self):
        if self.stopping or self.phase == 'finished':
            return
        self.stopping = True
        self.dialog.cancel_button.setEnabled(False)
        self.dialog.label.setText('Stopping batch...')
        preview = getattr(self.window, '_preview_export', None)
        if preview is not None:
            preview.cancel()
        elif self.window.busy:
            self.window.cancel_job()

    def finish(self):
        if self.phase == 'finished':
            return
        self.timer.stop()
        if self.stopping:
            for i, row in enumerate(self.rows):
                if row['status'] not in ('done', 'failed'):
                    row['status'] = 'cancelled' if i == self.index else 'skipped'
                    self.dialog.table.item(i, 1).setText(row['status'].title())
        self.phase = 'finished'
        self.save_report()
        done = sum(r['status'] == 'done' for r in self.rows)
        failed = sum(r['status'] == 'failed' for r in self.rows)
        message = f'Batch {"cancelled" if self.stopping else "complete"}: {done} succeeded, {failed} failed. Report: {self.folder / "batch-report.json"}'
        self.dialog.summary.setText(message)
        self.dialog.summary.setWordWrap(True)
        self.dialog.label.setText('Completed exports and project stages have been kept.')
        self.dialog.progress.setRange(0, 100)
        self.dialog.progress.setValue(100)
        self.dialog.allow_close = True
        self.dialog.cancel_button.setText('Close')
        self.dialog.cancel_button.setEnabled(True)
        w = self.window
        w._video_batch = None
        w._source_export_options = self.previous_export_options
        w.setAcceptDrops(self.saved_drops)
        for shortcut, enabled in zip(w.shortcuts, self.saved_shortcuts):
            shortcut.setEnabled(enabled)
        w.status.setText(message)
        w.log.append(message)
        w.update_readiness()
        # The summary stays modal until reviewed. Release this controller afterwards.
        self.dialog.finished.connect(self.dialog.deleteLater)
        self.dialog.finished.connect(lambda _: self.deleteLater())


class VideoBatchWorkspace:
    def init_video_batch(self):
        self._batch_videos = []
        self._batch_export_options = {}
        self._video_batch = None

    def build_video_batch_menu(self):
        self.batch_menu = QMenu('Batch Processing', self)
        self.menuBar().insertMenu(self.menuBar().actions()[-1], self.batch_menu)
        self.batch_menu.addAction('Select Videos...', self.select_batch_videos).setToolTip(
            'Queue independent videos and open the first to configure shared settings.')
        self.batch_menu.addAction('Setup Export...', self.setup_batch_export).setToolTip(
            'Choose the output folder, motion format and optional Preview Video for every video.')

    def sync_video_batch(self):
        supported = self.model_choice.currentData() not in ('pear', 'mamma')
        idle = not self.busy and not self.contact_saving and self._video_batch is None and not getattr(self, '_preview_export', None)
        self.batch_menu.setEnabled(supported and idle)
        self.run_button.setText(f'Reconstruct 3D ({len(self._batch_videos) if self._batch_videos and supported else int(self.video_path is not None)})')

    def select_batch_videos(self):
        names, _ = QFileDialog.getOpenFileNames(self, 'Select Videos', '', 'Videos (*.mp4 *.mov *.avi *.mkv *.webm)')
        if not names:
            return
        videos = list(dict.fromkeys(Path(name).resolve() for name in names))
        self._batch_videos = []
        self.load_video(videos[0])
        self._batch_videos = videos
        self.sync_video_batch()

    def setup_batch_export(self):
        dialog = BatchExportDialog(self, self._batch_export_options)
        try:
            if dialog.exec() == QDialog.Accepted:
                self._batch_export_options = dialog.options()
                return True
            return False
        finally:
            dialog.deleteLater()

    def start_video_batch(self):
        if self.busy or self.contact_saving or self._video_batch is not None:
            return
        if not self._batch_export_options and not self.setup_batch_export():
            return
        try:
            controller = VideoBatch(self, self._batch_videos, self._batch_export_options)
        except (OSError, ValueError) as error:
            self.status.setText('Cannot start batch: ' + str(error))
            return
        self._video_batch = controller
        self.sync_video_batch()
        controller.start()

    def video_batch_jobs_finished(self):
        controller = self._video_batch
        if controller is not None:
            failed, cancelled, message = self.failed, self.cancelled, self.status.text()
            QTimer.singleShot(0, lambda: controller.jobs_finished(failed, cancelled, message))
