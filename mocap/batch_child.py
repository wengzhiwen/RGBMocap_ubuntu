"""One headless batch child: a single offscreen window processing one video.

The child is launched by the parallel batch parent through `app.py
--batch-child <request.json>` with CUDA_VISIBLE_DEVICES pinning its GPU. It
reuses the ordinary VideoBatch state machine unchanged; model switching,
compute mode and Resource Safety are restored from the request.
"""
from pathlib import Path
import os
import sys

from . import batch_protocol

# Belt-and-braces bound for a child that stopped reporting phases; the window's
# own worker stall detection remains the first line of defense.
WATCHDOG_SECONDS = 30 * 60


def apply_request_window_state(window, request):
    """Restore the state workspace_settings does not cover, without side writes."""
    model = request.get('model')
    index = window.model_choice.findData(model)
    if index < 0:
        raise ValueError('The batch request names an unavailable model: ' + str(model))
    if index != window.model_choice.currentIndex():
        window.model_choice.setCurrentIndex(index)  # select_model runs through its signal
    window.compute_mode.setCurrentIndex(max(0, window.compute_mode.findData(request.get('compute_mode', 'auto'))))
    # Set without toggled(): resource_safety_changed would write the shared local config.
    from .resource_safety import ENV
    enabled = bool(request.get('resource_safety', True))
    window.resource_safety.blockSignals(True)
    window.resource_safety.setChecked(enabled)
    window.resource_safety.blockSignals(False)
    os.environ[ENV] = '1' if enabled else '0'


class BatchChild:
    def __init__(self, request, app, window):
        self.request = request
        self.app = app
        self.window = window
        self.done = False
        from PySide6.QtCore import QTimer
        self.watchdog = QTimer()
        self.watchdog.setSingleShot(True)
        self.watchdog.setInterval(WATCHDOG_SECONDS * 1000)
        self.watchdog.timeout.connect(self._stalled)

    def attach(self):
        from .video_batch_ui import VideoBatch
        options = dict(self.request['options'])
        options['preview'] = False
        self.controller = VideoBatch(self.window, [Path(self.request['video'])], options,
                                     headless=self.request)
        self.window._video_batch = self.controller
        self.controller.phase_changed.connect(self.watchdog.start)
        self.controller.child_result.connect(self._finished)
        self.watchdog.start()

    def _stop_live_worker(self):
        from PySide6.QtCore import QProcess
        try:
            process = getattr(self.window, 'process', None)
            if process is not None and process.state() != QProcess.NotRunning:
                from .linux_runtime import terminate_descendants
                terminate_descendants(process.processId())
                process.kill()
        except Exception as error:  # teardown best-effort on a stuck child
            print(f'Batch child teardown: {error}', file=sys.stderr, flush=True)

    def _emit(self, result, code):
        if self.done:
            return
        self.done = True
        try:
            batch_protocol.write_result(self.request['result'], result)
        except OSError as error:
            print(f'Cannot write the batch child result: {error}', file=sys.stderr, flush=True)
        batch_protocol.emit(dict(type='result', **{k: v for k, v in result.items() if k != 'version'}))
        self.app.exit(code)

    def _finished(self, result):
        status = result.get('status')
        code = batch_protocol.EXIT_OK if result.get('ok') else (
            batch_protocol.EXIT_CANCELLED if status in ('cancelled', 'skipped') else batch_protocol.EXIT_FAILED)
        self._emit(batch_protocol.build_result(
            request=self.request, ok=result.get('ok', False), status=status,
            error=result.get('error', ''), exports=result.get('exports', []),
            project=result.get('project', '')), code)

    def _stalled(self):
        if self.done:
            return
        self._stop_live_worker()
        self._emit(batch_protocol.build_result(
            request=self.request, ok=False, status='failed',
            error=f'Batch child watchdog: no phase progress for {WATCHDOG_SECONDS // 60} minutes.'),
            batch_protocol.EXIT_FAILED)


def run_child(request_path, window_factory=None):
    try:
        request = batch_protocol.read_request(request_path)
    except (OSError, ValueError) as error:
        print(f'Cannot read the batch child request: {error}', file=sys.stderr, flush=True)
        return batch_protocol.EXIT_CRASH
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        print('Batch child warning: CUDA_VISIBLE_DEVICES is unset; using the default GPU.',
              file=sys.stderr, flush=True)
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv[:1])
    if window_factory is None:
        from .gui import MainWindow
        window_factory = MainWindow
    window = window_factory()
    apply_request_window_state(window, request)
    # The child never shows or closes the window: shutdown_preview is the same
    # pure teardown the smoke path uses; closeEvent's prompts and project-view
    # saves stay untouched.
    app.aboutToQuit.connect(window.shutdown_preview)
    child = BatchChild(request, app, window)
    child.attach()
    from PySide6.QtCore import QTimer
    QTimer.singleShot(0, child.controller.start)
    return app.exec()
