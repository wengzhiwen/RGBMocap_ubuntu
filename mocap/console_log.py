"""Mirror the in-app console to a command window without blocking the UI."""
import os
import queue
import sys
import threading

from PySide6.QtWidgets import QTextEdit


class ConsoleLog(QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._messages = None
        self._last_status = None
        stream = sys.stdout if os.environ.get("RGB_MOCAP_MANAGED") == "1" or os.name != "nt" else None
        if stream is not None:
            self._messages = queue.Queue(maxsize=256)
            threading.Thread(target=self._write_messages, args=(stream,), daemon=True,
                             name="console-output").start()
            self.write_terminal("RGB MoCap started. Processing messages also appear in the app Console.")

    def _write_messages(self, stream):
        while True:
            message = self._messages.get()
            try:
                stream.write(message + "\n")
                stream.flush()
            except (OSError, ValueError, UnicodeError):
                # Closing a terminal must not interrupt processing or the GUI.
                return

    def write_terminal(self, message):
        if self._messages is not None and message and str(message).strip():
            try:
                self._messages.put_nowait(str(message)[:16384])
            except queue.Full:
                # A blocked output stream must not stall the GUI or grow memory.
                pass

    def status_terminal(self, message):
        if message != self._last_status:
            self._last_status = message
            self.write_terminal(message)

    def append(self, text):
        super().append(text)
        self.write_terminal(text)
