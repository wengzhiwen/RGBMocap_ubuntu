"""Multi-GPU parallel video batch: one headless child process per selected GPU.

The parent keeps the single batch dialog, report and output folder; each video
runs in its own offscreen copy of the application pinned to one GPU through
CUDA_VISIBLE_DEVICES. Inside every child the ordinary sequential pipeline,
worker sessions and Resource Safety guards are unchanged.
"""
from collections import deque
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import os
import subprocess
import time

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer

PREVIEW_WAIT_SECONDS = 60

from . import batch_protocol
from .paths import ROOT
from .storage import atomic_json
from .video_batch_ui import BatchProgress, PEOPLE_BATCH_MODELS

GPU_QUERY_FIELDS = 'index,uuid,name,driver_version,pci.bus_id,memory.total'


class GpuQueryError(RuntimeError):
    pass


def query_gpus(timeout=5.0):
    """Enumerate physical GPUs with the standard library only; no Torch import."""
    command = ['nvidia-smi', f'--query-gpu={GPU_QUERY_FIELDS}', '--format=csv,noheader']
    try:
        completed = subprocess.run(command, capture_output=True, text=True,
                                   timeout=timeout, check=True)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        raise GpuQueryError(f'Cannot query GPUs: {error}') from error
    rows = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(',')]
        if len(parts) < 6 or not parts[0].isdigit():
            continue
        rows.append(dict(index=int(parts[0]), uuid=parts[1], name=parts[2],
                         driver=parts[3], pci=parts[4], memory_mib=_memory_mib(parts[5])))
    if not rows:
        raise GpuQueryError('nvidia-smi reported no GPUs.')
    return rows


def _memory_mib(text):
    try:
        return int(float(text.split()[0]))
    except (ValueError, IndexError):
        return 0


def gpu_label(gpu, current=False):
    memory = gpu.get('memory_mib') or 0
    size = f' ({memory / 1024:.0f} GB)' if memory else ''
    suffix = ' (current)' if current else ''
    return f"GPU {gpu['index']} — {gpu['name']}{size}{suffix}"


def parent_scope(gpus):
    """Physical rows the parent's CUDA_VISIBLE_DEVICES exposes; None when unset."""
    tokens = [token.strip() for token in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
              if token.strip()]
    if not tokens:
        return None
    rows = []
    for token in tokens:
        if token == '-1':
            break
        if token.isdigit():
            matches = [row for row in gpus if str(row['index']) == token]
        else:
            matches = [row for row in gpus if row['uuid'].startswith(token)]
        if len(matches) != 1:
            raise GpuQueryError(
                f'Cannot map CUDA_VISIBLE_DEVICES entry {token!r} to one physical GPU.')
        rows.append(matches[0])
    return rows


def default_child_command(request, request_path):
    return batch_protocol.child_command(request_path)


class ParallelVideoBatch(QObject):
    """Dispatch videos to at most one child per selected GPU, dynamically."""

    def __init__(self, window, videos, options, gpus, child_command=default_child_command):
        super().__init__(window)
        if len(gpus) < 2:
            raise ValueError('Parallel batch needs at least two GPUs.')
        self.window = window
        self.videos = list(videos)
        self.options = deepcopy(options)
        self.gpus = [dict(gpu) for gpu in gpus]
        self.child_command = child_command
        self.settings = deepcopy(window.workspace_settings())
        self.background = window.camera_tools.background_options()
        self.trim = None
        if window.info:
            first, end = window.processing_range()
            if not (window.source_offset == 0 and window.slider.clip_in == 0
                    and window.slider.clip_out == window.info['frames']):
                self.trim = (first, end)
        self.model = window.model_choice.currentData()
        if self.options.get('auto_people') and self.model in PEOPLE_BATCH_MODELS:
            from .models import missing
            unavailable = missing('people')
            if unavailable:
                raise ValueError('Prepare person detection models first: '
                                 + ', '.join(p.name for p in unavailable))
        self.compute_mode = window.compute_mode.currentData()
        self.resource_safety = window.resource_safety.isChecked()
        # The first video keeps a manual People selection exactly like the
        # sequential batch keeps the open window's selection.
        self.manual_people = None
        if window.video_readable and window.info and getattr(window, 'people_root', None) is not None \
                and window.people_state and Path(window.video_path).resolve() == self.videos[0] \
                and any(p.get('checked') for p in window.people_state.get('people', [])):
            self.manual_people = dict(people_root=str(window.people_root),
                                      state=deepcopy(window.people_state))
        self.rows = [dict(video=str(p), status='pending', error='', exports=[], gpu='')
                     for p in videos]
        # Exclusive folders protect same-stem inputs and all previously saved output.
        self.folder = Path(options['folder']) / ('Batch_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        self.folder.mkdir()
        self.dialog = BatchProgress(window, videos, gpus=self.gpus)
        self.dialog.cancel_requested.connect(self.cancel)
        self.stopping = False
        self.phase = 'running'
        self.next_index = 0
        self.children = {}
        self.idle = deque(self.gpus)
        self.saved_shortcuts = [s.isEnabled() for s in window.shortcuts]
        self.saved_drops = window.acceptDrops()

    def start(self):
        w = self.window
        w.stop_playback()
        for shortcut in w.shortcuts:
            shortcut.setEnabled(False)
        w.setAcceptDrops(False)
        self.dialog.show()
        for _ in range(min(len(self.gpus), len(self.videos))):
            self._dispatch()

    def jobs_finished(self, failed, cancelled, message):
        pass  # the parent window runs no jobs during a parallel batch

    def save_report(self):
        try:
            atomic_json(self.folder / 'batch-report.json', dict(
                version=1, model=self.model, settings=self.settings, trim_seconds=self.trim,
                export=self.options, status=self.phase, videos=self.rows,
                parallel=dict(gpus=[dict(index=g['index'], uuid=g['uuid'], name=g['name'])
                                    for g in self.gpus])))
        except OSError as error:
            self.window.log.append('Cannot save batch report: ' + str(error))

    def _dispatch(self):
        if self.stopping or not self.idle or self.next_index >= len(self.videos):
            return
        index = self.next_index
        self.next_index += 1
        gpu = self.idle.popleft()
        video = self.videos[index]
        output = self.folder / f'{index + 1:04d}-{video.stem}'
        output.mkdir(parents=True, exist_ok=True)
        request_path = output / 'child-request.json'
        request = batch_protocol.build_request(
            video=video, number=index + 1, output_dir=output, model=self.model,
            options=self.options, settings=self.settings, background=self.background,
            trim=self.trim, compute_mode=self.compute_mode,
            resource_safety=self.resource_safety,
            manual_people=self.manual_people if index == 0 else None,
            gpu=gpu, result=output / 'child-result.json')
        batch_protocol.write_request(request_path, request)
        process = QProcess(self)
        process.setWorkingDirectory(str(ROOT))
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert('QT_QPA_PLATFORM', 'offscreen')
        environment.insert('CUDA_VISIBLE_DEVICES', gpu['uuid'])
        environment.insert('PYTHONIOENCODING', 'utf-8')
        environment.insert('PYTHONUNBUFFERED', '1')
        process.setProcessEnvironment(environment)
        process.setProcessChannelMode(QProcess.MergedChannels)
        process.readyReadStandardOutput.connect(lambda i=index: self._read_child(i))
        process.finished.connect(lambda code, status, i=index: self._child_finished(i, code))
        process.errorOccurred.connect(lambda error, i=index: self._child_error(i, error))
        self.children[index] = dict(process=process, gpu=gpu, tail='', pending='',
                                    request=request_path, result=request['result'],
                                    result_event=None)
        self.rows[index]['gpu'] = f"GPU {gpu['index']}"
        gpu_item = self.dialog.table.item(index, 1)
        if gpu_item is not None:
            gpu_item.setText(f"GPU {gpu['index']}")
        self._set_status(index, 'opening')
        self.dialog.set_gpu_status(gpu['index'], f"GPU {gpu['index']} · {video.name}: Opening")
        command = self.child_command(request, request_path)
        process.start(command[0], command[1:])

    def _read_child(self, index):
        record = self.children.get(index)
        if record is None:
            return
        data = bytes(record['process'].readAllStandardOutput()).decode('utf-8', 'replace')
        record['tail'] = batch_protocol.append_tail(record['tail'], data)
        text = record['pending'] + data
        lines = text.split('\n')
        record['pending'] = lines.pop() if len(lines) > 1 else ''
        for line in lines:
            event = batch_protocol.parse_line(line)
            if event is not None:
                self._apply_event(index, event)

    def _apply_event(self, index, event):
        record = self.children.get(index)
        kind = event.get('type')
        if kind == 'phase':
            phase = str(event.get('phase', ''))
            self.rows[index]['status'] = phase
            self._set_status(index, phase)
            if record is not None:
                gpu = record['gpu']
                self.dialog.set_gpu_status(gpu['index'],
                                           f"GPU {gpu['index']} · {self.videos[index].name}: "
                                           + phase.replace('_', ' ').title())
            self.save_report()
        elif kind == 'progress' and record is not None and not self.stopping:
            gpu = record['gpu']
            self.dialog.set_gpu_status(gpu['index'],
                                       f"GPU {gpu['index']} · {self.videos[index].name}: "
                                       + str(event.get('status', '')),
                                       int(event.get('value') or 0),
                                       int(event.get('maximum') or 0))
        elif kind == 'result' and record is not None:
            record['result_event'] = event

    def _set_status(self, index, phase, prefix=''):
        text = (prefix or str(phase).replace('_', ' ').title())
        item = self.dialog.table.item(index, 2)
        if item is not None:
            item.setText(text)

    def _child_error(self, index, error):
        record = self.children.get(index)
        if record is not None and error == QProcess.FailedToStart:
            record['tail'] = batch_protocol.append_tail(
                record['tail'], 'child process failed to start\n')

    def _child_finished(self, index, code):
        record = self.children.pop(index, None)
        if record is None:
            return
        record['process'].deleteLater()
        self.idle.append(record['gpu'])
        self.dialog.set_gpu_status(record['gpu']['index'],
                                   f"GPU {record['gpu']['index']} · idle")
        result = record.get('result_event') or batch_protocol.read_result(record['result'])
        row = self.rows[index]
        if self.stopping:
            if row['status'] not in ('done', 'failed'):
                row['status'] = 'cancelled'
                self._set_status(index, row['status'])
        elif result is not None and result.get('ok'):
            row['status'] = 'done'
            row['exports'] = [str(e) for e in result.get('exports', [])]
            row['project'] = str(result.get('project', ''))
            self._set_status(index, 'done')
        elif result is not None:
            row['status'] = 'failed'
            row['error'] = str(result.get('error', '')) or 'The video failed.'
            self._set_status(index, 'failed', 'Failed: ' + row['error'])
            self.window.log.append(f'Batch video {index + 1}: {self.videos[index]}\n{row["error"]}')
        else:
            row['status'] = 'failed'
            row['error'] = f'Child process exited with code {code}.'
            if record['tail']:
                excerpt = '\n'.join(record['tail'].strip().splitlines()[-5:])
                row['error'] += '\n' + excerpt
            self._set_status(index, 'failed', 'Failed: ' + row['error'])
            self.window.log.append(f'Batch video {index + 1}: {self.videos[index]}\n{row["error"]}')
        completed = sum(r['status'] in ('done', 'failed', 'cancelled', 'skipped') for r in self.rows)
        self.dialog.overall.setValue(completed)
        self.save_report()
        if not self.children and self.next_index >= len(self.videos):
            self._start_preview_pass()
            return
        if not self.children and self.stopping:
            self.finish()
            return
        self._dispatch()

    def _start_preview_pass(self):
        """Render preview videos sequentially in the shown parent window.

        The parallel children are offscreen and cannot capture the OpenGL
        viewport; after inference the parent replays each finished project and
        renders its preview exactly like the sequential batch would.
        """
        self.preview_queue = ([index for index, row in enumerate(self.rows)
                               if row['status'] == 'done' and row.get('project')]
                              if self.options.get('preview') else [])
        self.preview_position = 0
        self.current_preview = None
        self.preview_subphase = 'idle'
        if not self.preview_queue:
            self.finish()
            return
        self.phase = 'preview_pass'
        self.preview_timer = QTimer(self)
        self.preview_timer.setInterval(100)
        self.preview_timer.timeout.connect(self._preview_tick)
        self.preview_timer.start()
        self._next_preview()

    def _next_preview(self):
        if self.preview_position >= len(self.preview_queue):
            self.preview_timer.stop()
            self.finish()
            return
        index = self.preview_queue[self.preview_position]
        self.preview_position += 1
        self.current_preview = index
        self.preview_subphase = 'opening'
        self.preview_started = time.monotonic()
        row = self.rows[index]
        self.dialog.summary.setText(
            f'Preview {self.preview_position}/{len(self.preview_queue)} · {Path(row["video"]).name}')
        self._set_status(index, 'preview', 'Rendering preview')
        try:
            opened = self.window.load_project(Path(row['project']))
        except Exception as error:
            opened = error
        if opened is False or isinstance(opened, Exception):
            self._preview_failed(str(opened))
            return
        self.save_report()

    def _preview_tick(self):
        w = self.window
        if self.stopping:
            preview = getattr(w, '_preview_export', None)
            if preview is not None:
                return  # its cancelled completion settles this pass
            if not w.busy:
                self.preview_timer.stop()
                self.finish()
            return
        index = self.current_preview
        if index is None:
            return
        try:
            if self.preview_subphase == 'opening':
                if w.preview_export_available():
                    self.preview_subphase = 'rendering'
                    controller = w.start_preview_export(
                        self.folder / f'{index + 1:04d}-{self.videos[index].stem}' / 'preview.mp4',
                        resume=False, progress_dialog=self.dialog)
                    controller.completed.connect(self._preview_finished)
                elif time.monotonic() - self.preview_started > PREVIEW_WAIT_SECONDS:
                    self._preview_failed('Timed out waiting for the motion preview.')
            elif self.preview_subphase == 'rendering':
                self.dialog.progress.setRange(w.progress.minimum(), w.progress.maximum())
                self.dialog.progress.setValue(w.progress.value())
        except Exception as error:
            self._preview_failed(str(error))

    def _preview_finished(self, success, message, cancelled):
        index = self.current_preview
        if index is None or self.preview_subphase not in ('rendering', 'opening'):
            return
        if self.stopping:
            return  # the cancelled export settles through the stopping tick
        if success:
            preview_path = self.folder / f'{index + 1:04d}-{self.videos[index].stem}' / 'preview.mp4'
            self.rows[index]['exports'].append(str(preview_path))
            self._set_status(index, 'done')
            self.save_report()
            self.current_preview = None
            QTimer.singleShot(0, self._next_preview)
        else:
            self._preview_failed(message if not cancelled else 'Preview export was cancelled.')

    def _preview_failed(self, message):
        index = self.current_preview
        if index is None:
            return
        if self.stopping:
            self.current_preview = None
            return
        # Same rule as the sequential batch: a failed preview marks the video
        # failed while keeping its completed motion exports.
        row = self.rows[index]
        row['status'] = 'failed'
        row['error'] = message
        self._set_status(index, 'failed', 'Failed: ' + message)
        self.window.log.append(f'Batch preview {index + 1}: {self.videos[index]}\n{message}')
        self.current_preview = None
        self.save_report()
        QTimer.singleShot(0, self._next_preview)

    def cancel(self):
        if self.stopping or self.phase == 'finished':
            return
        self.stopping = True
        self.dialog.cancel_button.setEnabled(False)
        self.dialog.label.setText('Stopping batch...')
        for index in getattr(self.dialog, 'gpu_rows', {}):
            self.dialog.set_gpu_status(index, f'GPU {index} · stopping...')
        preview = getattr(self.window, '_preview_export', None)
        if preview is not None:
            preview.cancel()
        for record in self.children.values():
            process = record['process']
            if process.state() != QProcess.NotRunning:
                from .linux_runtime import terminate_descendants
                terminate_descendants(process.processId())
                process.kill()

    def finish(self):
        if self.phase == 'finished':
            return
        timer = getattr(self, 'preview_timer', None)
        if timer is not None:
            timer.stop()
        if self.stopping:
            for i, row in enumerate(self.rows):
                if row['status'] not in ('done', 'failed', 'cancelled'):
                    row['status'] = 'skipped'
                    self._set_status(i, row['status'])
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
        for index in getattr(self.dialog, 'gpu_rows', {}):
            self.dialog.set_gpu_status(index, f'GPU {index} · finished', 100, 100)
        self.dialog.allow_close = True
        self.dialog.cancel_button.setText('Close')
        self.dialog.cancel_button.setEnabled(True)
        w = self.window
        w._video_batch = None
        w.setAcceptDrops(self.saved_drops)
        for shortcut, enabled in zip(w.shortcuts, self.saved_shortcuts):
            shortcut.setEnabled(enabled)
        w.status.setText(message)
        w.log.append(message)
        w.update_readiness()
        # The summary stays modal until reviewed. Release this controller afterwards.
        self.dialog.finished.connect(self.dialog.deleteLater)
        self.dialog.finished.connect(lambda _: self.deleteLater())
