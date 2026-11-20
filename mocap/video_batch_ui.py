"""Sequential video automation over the ordinary GUI jobs and guarded exports."""
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import json
import os
import time

from PySide6.QtCore import QObject, QTimer, Qt, Signal
from PySide6.QtWidgets import (QApplication, QCheckBox, QDialog, QFileDialog, QFrame, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QMenu, QProgressBar, QPushButton, QScrollArea,
    QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from .source_export_dialog import SourceExportDialog
from .storage import atomic_json
from .tooltips import tooltip_text
from .motion_tools import LINUX_UNAVAILABLE

BATCH_MODELS = frozenset(('exmr', 'flowhmr', 'gvhmr', 'prompthmr'))
if os.name != 'nt':  # PromptHMR stays a Windows-only batch model
    BATCH_MODELS = BATCH_MODELS - LINUX_UNAVAILABLE
# Find People discovery applies to the models whose pipeline consumes the
# per-person workspace; PromptHMR tracks its own people inside its pipeline.
PEOPLE_BATCH_MODELS = frozenset(('exmr', 'flowhmr', 'gvhmr'))


def _wrapped_height(label, width):
    """Word-wrapped labels report one-line minimums; use their real wrapped height."""
    height = label.heightForWidth(width)
    hint = label.sizeHint().height()
    return max(height, hint) if height >= 0 else hint


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
        people = QHBoxLayout()
        self.auto_people = QCheckBox('Find people first; check the first')
        self.auto_people.setChecked(saved.get('auto_people', True))
        self.auto_people.setToolTip(tooltip_text('Run Find People on every video before reconstruction and '
                                                 'check the first people automatically, so each is reconstructed '
                                                 'and exported separately. The default tracker is used.'))
        self.people_limit = QSpinBox()
        self.people_limit.setRange(1, 10)
        self.people_limit.setValue(saved.get('people_limit', 3))
        self.people_limit.setToolTip(tooltip_text('How many of the first found people are checked per video.'))
        people.addWidget(self.auto_people)
        people.addWidget(self.people_limit)
        people.addWidget(QLabel('people'))
        people.addStretch()
        layout.addLayout(people)
        note = QLabel('Each run creates a new batch folder with per-video exports and a progress/error report. '
                      'The first video supplies the shared model settings and trim in seconds. '
                      'An untrimmed first video processes every video in full; shorter videos are clamped. '
                      'Find People runs per video with the default tracker and checks the first people '
                      '(ExMR, FlowHMR, GVHMR); a first video with a manual People selection keeps it. '
                      'PromptHMR uses its own people settings.')
        note.setWordWrap(True)
        note.setMinimumHeight(_wrapped_height(note, 588))
        layout.addWidget(note)
        self._build_parallel_group(saved, layout)
        # The batch panel can exceed short screens; keep it scrollable then.
        # The scroll area demands the panel's full height, so tall screens show
        # everything without a scrollbar and only short screens scroll.
        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setMinimumHeight(max(panel.layout().totalMinimumSize().height(), 320))
        self.layout().insertWidget(self.layout().count() - 1, scroll)
        self.format.setCurrentIndex(max(0, self.format.findData(saved.get('format', 'fbx'))))
        self.profile.setCurrentIndex(max(0, self.profile.findData(saved.get('bvh_profile', 'generic'))))
        self.tpose.setChecked(saved.get('prepend_tpose', False))
        self.fbx_compatibility.setChecked(saved.get('fbx_compatibility', False))
        self.place_at_origin.setChecked(saved.get('place_at_origin', False))
        self.symmetric_rest.setChecked(saved.get('symmetric_rest', False))
        self.animate_camera.setChecked(saved.get('animate_camera', True))
        self.background_refine.setChecked(saved.get('background_refine', False))
        self.translation_scale.setValue(saved.get('camera_translation_scale', 1.))
        # The base export dialog is a fixed 660x400; the batch panel needs more room.
        self._grow_for_batch_panel()

    def update_format(self):
        super().update_format()
        # The base note is word-wrapped too; keep its minimum honest after text swaps.
        self.note.setMinimumHeight(_wrapped_height(self.note, 588))

    def _grow_for_batch_panel(self):
        self.layout().activate()
        # Wrapped labels report optimistic minimums; the margin keeps the grid rows
        # from being squeezed flat when the layout settles.
        needed = self.layout().totalMinimumSize().height() + 48
        screen = self.screen() or QApplication.primaryScreen()
        cap = screen.availableGeometry().height() - 60 if screen is not None else needed
        self.setFixedSize(660, max(400, min(needed, cap)))

    def _build_parallel_group(self, saved, layout):
        """One checkbox per physical GPU; hidden when nvidia-smi is unavailable."""
        self.gpu_boxes = []
        self.parallel_note = None
        try:
            from .batch_parallel import GpuQueryError, gpu_label, parent_scope, query_gpus
            try:
                gpus = query_gpus()
                current = parent_scope(gpus) or []
            except GpuQueryError:
                gpus, current = [], []
        except ImportError:
            gpus, current = [], []
        if len(gpus) < 2:
            return
        group = QGroupBox('Parallel GPUs')
        column = QVBoxLayout(group)
        saved_uuids = {entry.get('uuid') for entry in saved.get('parallel_gpus', [])
                       if isinstance(entry, dict)}
        for gpu in gpus:
            box = QCheckBox(gpu_label(gpu, current=gpu in current))
            box.setChecked(gpu['uuid'] in saved_uuids)
            box.toggled.connect(self.sync_parallel)
            # Two columns would exceed the dialog's fixed width; one label per
            # row keeps the panel inside the scroll viewport without a horizontal
            # scrollbar. The explicit minimum keeps wrapped labels elsewhere from
            # squeezing the rows flat.
            box.setMinimumHeight(box.sizeHint().height())
            column.addWidget(box)
            self.gpu_boxes.append((gpu, box))
        note = QLabel()
        note.setWordWrap(True)
        column.addWidget(note)
        self.parallel_note = note
        layout.addWidget(group)
        self.sync_parallel()

    def sync_parallel(self):
        count = sum(box.isChecked() for _, box in self.gpu_boxes)
        parallel = count >= 2
        self.preview.setToolTip(tooltip_text(
            'Also save the input video beside the current 3D viewport as MP4.'
            + (' In parallel runs the preview videos are rendered sequentially in the main window after reconstruction.'
               if parallel else '')))
        if self.parallel_note is not None:
            self.parallel_note.setText(
                'Check two or more to process one video per GPU in parallel. '
                + ('Parallel runs render preview videos sequentially in the main window after reconstruction. '
                   if parallel else '')
                + ('Each parallel copy also loads its own GUI and workers; watch RAM headroom.'
                   if count > 2 else ''))
            self.parallel_note.setMinimumHeight(_wrapped_height(self.parallel_note, 568))

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
        gpus = [dict(index=gpu['index'], uuid=gpu['uuid'], name=gpu['name'])
                for gpu, box in self.gpu_boxes if box.isChecked()]
        return dict(super().options(), folder=self.folder.text().strip(), preview=self.preview.isChecked(),
                    auto_people=self.auto_people.isChecked(), people_limit=self.people_limit.value(),
                    parallel_gpus=gpus)


class BatchProgress(QDialog):
    cancel_requested = Signal()

    def __init__(self, window, videos, gpus=None):
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
        # Parallel batches add the GPU column; the sequential table stays as it was.
        headers = ['Video', 'GPU', 'Status'] if gpus else ['Video', 'Status']
        self.table = QTableWidget(len(videos), len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setColumnWidth(0, 280)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        for i, path in enumerate(videos):
            item = QTableWidgetItem(path.name)
            item.setToolTip(str(path))
            self.table.setItem(i, 0, item)
            for column in range(1, len(headers)):
                self.table.setItem(i, column, QTableWidgetItem(
                    '' if gpus and column == 1 else 'Pending'))
        layout.addWidget(self.table)
        if gpus:
            # One status line and progress bar per GPU; children report
            # independently and a single shared label would flicker between them.
            self.gpu_rows = {}
            rows_layout = QVBoxLayout()
            rows_layout.setSpacing(2)
            for gpu in gpus:
                label = QLabel(f"GPU {gpu['index']} · idle")
                label.setTextFormat(Qt.PlainText)
                bar = QProgressBar()
                bar.setRange(0, 0)
                rows_layout.addWidget(label)
                rows_layout.addWidget(bar)
                self.gpu_rows[gpu['index']] = (label, bar)
            layout.addLayout(rows_layout)
        self.label = QLabel('Starting...')
        self.label.setTextFormat(Qt.PlainText)
        self.label.setWordWrap(True)
        if gpus:
            self.label.hide()
        layout.addWidget(self.label)
        self.progress = QProgressBar()
        if gpus:
            self.progress.hide()
        layout.addWidget(self.progress)
        self.cancel_button = QPushButton('Cancel Batch')
        self.cancel_button.clicked.connect(self.request_close)
        layout.addWidget(self.cancel_button)

    def set_gpu_status(self, index, text, value=None, maximum=None):
        row = getattr(self, 'gpu_rows', {}).get(index)
        if row is None:
            return
        label, bar = row
        label.setText(text)
        if maximum:
            bar.setRange(0, maximum)
            bar.setValue(value or 0)
        else:
            bar.setRange(0, 0)

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
    child_result = Signal(dict)
    phase_changed = Signal(str)

    def __init__(self, window, videos, options, headless=None):
        super().__init__(window)
        self.window = window
        self.videos = list(videos)
        self.options = deepcopy(options)
        # A parallel-batch child runs the same state machine without a dialog or
        # batch folder; it restores the parent's captured state from the request.
        self.headless = headless
        if headless is not None:
            self.settings = deepcopy(headless['settings'])
            self.background = deepcopy(headless['background'])
            self.trim = headless['trim']
            self.model = headless['model']
            self.options['preview'] = False  # preview capture needs the shown window
            self.folder = None
            self.output = Path(headless['output_dir'])
        else:
            self.settings = deepcopy(window.workspace_settings())
            self.background = window.camera_tools.background_options()
            self.trim = None
            if window.info:
                first, end = window.processing_range()
                if not (window.source_offset == 0 and window.slider.clip_in == 0
                        and window.slider.clip_out == window.info['frames']):
                    self.trim = (first, end)
            self.model = window.model_choice.currentData()
            # Exclusive folders protect same-stem inputs and all previously saved output.
            self.folder = Path(options['folder']) / ('Batch_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
            self.folder.mkdir()
        if self.options.get('auto_people') and self.model in PEOPLE_BATCH_MODELS:
            from .models import missing
            unavailable = missing('people')
            if unavailable:
                raise ValueError('Prepare person detection models first: '
                                 + ', '.join(p.name for p in unavailable))
        self.index = -1
        self.phase = 'starting'
        self.stopping = False
        self._manual_restored = False
        self._last_progress = 0.
        self.rows = [dict(video=str(p), status='pending', error='', exports=[]) for p in videos]
        self.dialog = None if headless is not None else BatchProgress(window, videos)
        if self.dialog is not None:
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
        if self.dialog is not None:
            self.dialog.show()
        self.timer.start()
        self.advance()

    def save_report(self):
        if self.headless is not None:
            return  # the parallel parent writes the single merged report
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
        if self.dialog is not None:
            self.dialog.table.item(self.index, 1).setText(phase.replace('_', ' ').title())
            self.dialog.table.scrollToItem(self.dialog.table.item(self.index, 0))
            self.dialog.progress.setRange(0, 0)
            self.dialog.label.setText(phase.replace('_', ' ').title())
            if not self.stopping:
                self.dialog.cancel_button.setEnabled(True)
        else:
            from . import batch_protocol
            batch_protocol.emit(dict(type='phase', phase=phase))
            self.phase_changed.emit(phase)
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
        if self.dialog is not None:
            self.dialog.summary.setText(f'Video {self.index + 1}/{len(self.videos)} · {self.videos[self.index].name}')
        self.set_phase('opening')
        try:
            if self.headless is None:
                self.output = self.folder / f'{self.index + 1:04d}-{self.videos[self.index].stem}'
            self.output.mkdir(parents=True, exist_ok=True)
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
                if self.headless is not None:
                    # The child freshly loaded this video, so the shared trim also
                    # applies to its first video; it carries the same source seconds.
                    w.slider.set_clip(*trimmed_frames(w.info, self.trim))
                    self._restore_manual_people()
                elif not self.keep_first:
                    # Keep the first video's own person/take selection and source trim.
                    w.slider.set_clip(*trimmed_frames(w.info, self.trim))
                if self.needs_find_people():
                    self.set_phase('finding_people')
                    self.start_find_people()
                    if not w.busy:
                        self.fail(w.status.text() or 'Find People did not start.')
                else:
                    self.begin_reconstructing()
            elif self.phase == 'people_ready' and w.video_readable and w.info:
                self.begin_reconstructing()
            elif self.phase == 'waiting_preview' and w.preview_export_available():
                self.set_phase('preview')
                controller = w.start_preview_export(self.output / 'preview.mp4', resume=False, progress_dialog=self.dialog)
                controller.completed.connect(self.preview_finished)
            elif self.phase in ('opening', 'people_ready', 'waiting_preview') and time.monotonic() - self.wait_started > 60:
                self.fail('Timed out waiting for video or motion preview.')
            if self.phase in ('finding_people', 'reconstructing', 'exporting'):
                if self.dialog is not None:
                    self.dialog.label.setText(w.status.text() + '\n' + w.task_info.text())
                    self.dialog.progress.setRange(w.progress.minimum(), w.progress.maximum())
                    self.dialog.progress.setValue(w.progress.value())
                elif time.monotonic() - self._last_progress >= 0.5:
                    self._last_progress = time.monotonic()
                    from . import batch_protocol
                    batch_protocol.emit(dict(type='progress', status=w.status.text(), task=w.task_info.text(),
                                             value=w.progress.value(), maximum=w.progress.maximum()))
        except Exception as error:
            # A live worker must settle before advancing, even if a GUI refresh fails.
            if w.busy:
                self.pending_error = str(error)
                w.cancel_job()
            else:
                self.fail(str(error))

    def needs_find_people(self):
        """Run discovery per video when the option is on; PromptHMR and an
        existing manual first-video People selection keep their own people."""
        if not self.options.get('auto_people') or self.model not in PEOPLE_BATCH_MODELS:
            return False
        if self.index == 0 and self.headless is not None and self._manual_restored:
            return False  # the parent's recorded first-video selection was restored
        if self.index == 0 and self.keep_first and self.window.people_state:
            if any(p.get('checked') for p in self.window.people_state.get('people', [])):
                return False
        return True

    def _restore_manual_people(self):
        """Re-apply the parent's first-video People selection inside a child.

        The people root is content-hash keyed, so it exists on disk exactly as it
        did in the parent. Any mismatch falls back to ordinary per-video discovery.
        """
        manual = self.headless.get('manual_people') if self.headless is not None else None
        if not manual or self.index != 0:
            return False
        root = Path(manual['people_root'])
        state = manual.get('state') or {}
        people = state.get('people') or []
        try:
            if not (root / 'project.json').is_file() or not any(p.get('checked') for p in people):
                raise ValueError('The recorded People selection is unusable.')
        except (OSError, ValueError):
            self.window.log.append('Batch child: the recorded People selection was not reusable; '
                                   'running Find People instead.')
            return False
        self.window.people_root = root
        self.window.people_state = dict(state)
        self._manual_restored = True
        return True

    def start_find_people(self):
        """Start the people discovery stage exactly like the Find person button's
        default tracker, including reuse of an existing matching people root."""
        w = self.window
        from .models import missing
        unavailable = missing('people')
        if unavailable:
            raise RuntimeError('Prepare person detection models first: '
                               + ', '.join(p.name for p in unavailable))
        if getattr(w, 'people_root', None) is not None and w.info is not None and w.video_readable:
            manifest = json.loads((w.people_root / 'project.json').read_text(encoding='utf-8'))
            first, end = w.processing_range()
            root_start = float(manifest['start'])
            root_end = min(float(manifest['end']), root_start + manifest['frames'] / manifest['fps'])
            # Cached discovery belongs to the group's complete input, not to a
            # newly trimmed preview or a shorter selected person's take.
            if abs(first - root_start) < 1e-6 and abs(end - root_end) < 1e-6:
                w.start_jobs([('people', ['--project', str(w.people_root)])])
                return
        w.input_jobs(['people'])

    def people_selected(self):
        """Check the first found people, then start reconstruction."""
        w = self.window
        people = w.people_state['people'] if w.people_state else []
        if not people:
            raise RuntimeError('No people were found in this video.')
        limit = max(1, int(self.options.get('people_limit', 3)))
        wanted = {p['id'] for p in people[:limit]}
        for person in people:
            person['checked'] = person['id'] in wanted
        if getattr(w, 'people_root', None) is not None:
            from .people_store import save_people
            save_people(w.people_root, w.people_state)
        people_list = getattr(w, 'people_list', None)
        if people_list is not None:
            for row in range(people_list.count()):
                item = people_list.item(row)
                ident = item.data(Qt.UserRole)
                if isinstance(ident, str):
                    continue  # character rows are not people
                state = Qt.Checked if ident in wanted else Qt.Unchecked
                if item.checkState() != state:
                    item.setCheckState(state)
        w.log.append(f"Batch: checked {len(wanted)} of {len(people)} found "
                     f"{'person' if len(people) == 1 else 'people'} (limit {limit}).")
        if w.video_readable and w.info:
            self.begin_reconstructing()
        else:
            # Loading the people workspace reopens its preview; wait like 'opening'.
            self.set_phase('people_ready')

    def begin_reconstructing(self):
        self.set_phase('reconstructing')
        w = self.window
        w.run_full_single()
        if not w.busy:
            self.fail(w.status.text() or 'Reconstruction did not start.')

    def jobs_finished(self, failed, cancelled, message):
        if self.phase not in ('finding_people', 'reconstructing', 'exporting'):
            return
        if self.stopping:
            return  # tick waits for the cancellation helper too
        error = getattr(self, 'pending_error', None)
        self.pending_error = None
        if failed or cancelled or error:
            self.fail(error or message)
            return
        try:
            if self.phase == 'finding_people':
                self.people_selected()
            elif self.phase == 'reconstructing':
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
        if self.dialog is not None:
            self.dialog.overall.setValue(self.index + 1)
        QTimer.singleShot(0, self.advance)

    def fail(self, message):
        if self.phase in ('failed', 'done', 'finished', 'cancelled') or self.stopping:
            return
        self.rows[self.index]['error'] = message
        self.window.log.append(f'Batch video {self.index + 1}: {self.videos[self.index]}\n{message}')
        self.set_phase('failed')
        if self.dialog is not None:
            item = self.dialog.table.item(self.index, 1)
            item.setText('Failed: ' + message)
            item.setToolTip(message)
            self.dialog.overall.setValue(self.index + 1)
        QTimer.singleShot(0, self.advance)

    def cancel(self):
        if self.stopping or self.phase == 'finished':
            return
        self.stopping = True
        if self.dialog is not None:
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
                    if self.dialog is not None:
                        self.dialog.table.item(i, 1).setText(row['status'].title())
        self.phase = 'finished'
        if self.headless is not None:
            # After the final advance() the index already points past the last row.
            if 0 <= self.index < len(self.rows):
                row = self.rows[self.index]
            elif self.index >= len(self.rows) and self.rows:
                row = self.rows[-1]
            else:
                row = dict(status='cancelled', error='', exports=[], project='')
            self.child_result.emit(dict(ok=row['status'] == 'done', status=row['status'],
                                        error=row.get('error', ''), exports=list(row.get('exports', [])),
                                        project=str(row.get('project', ''))))
            w = self.window
            w._video_batch = None
            w._source_export_options = self.previous_export_options
            w.setAcceptDrops(self.saved_drops)
            for shortcut, enabled in zip(w.shortcuts, self.saved_shortcuts):
                shortcut.setEnabled(enabled)
            w.update_readiness()
            self.deleteLater()
            return
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
        supported = self.model_choice.currentData() in BATCH_MODELS
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
        if self.model_choice.currentData() not in BATCH_MODELS or self.busy or self.contact_saving or self._video_batch is not None:
            return
        if not self._batch_export_options and not self.setup_batch_export():
            return
        try:
            gpus = self._batch_export_options.get('parallel_gpus') or []
            if len(gpus) >= 2 and len(self._batch_videos) >= 2:
                from .batch_parallel import ParallelVideoBatch
                controller = ParallelVideoBatch(self, self._batch_videos, self._batch_export_options, gpus)
            else:
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
