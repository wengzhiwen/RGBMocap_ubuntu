"""Request/result protocol between the parallel batch parent and its children.

Pure standard-library module shared by the GUI parent, the headless child and
the tests; it must stay importable without Qt, Torch or a display.
"""
from pathlib import Path
import json
import sys

from .paths import ROOT
from .storage import atomic_json

EVENT_PREFIX = '@@MOCAPBATCH@@'
REQUEST_VERSION = 1
RESULT_VERSION = 1
# 0 done, 3 recorded failure, 4 cancelled; anything else is an unexplained crash.
EXIT_OK, EXIT_CRASH, EXIT_FAILED, EXIT_CANCELLED = 0, 1, 3, 4
# Bounded log tails: parent-side buffer and the result-file excerpt.
TAIL_BYTES = 64 * 1024
RESULT_TAIL_BYTES = 8 * 1024
REQUIRED_REQUEST_KEYS = ('version', 'video', 'number', 'output_dir', 'model', 'options',
                         'settings', 'background', 'trim', 'compute_mode', 'resource_safety',
                         'gpu', 'result')


class ProtocolError(ValueError):
    pass


def child_command(request_path):
    """Launch one headless copy of this application for a single video."""
    return [sys.executable, '-I', '-X', 'utf8', '-u', '-B', str(ROOT / 'app.py'),
            '--batch-child', str(request_path)]


def build_request(*, video, number, output_dir, model, options, settings, background,
                  trim, compute_mode, resource_safety, manual_people, gpu, result):
    return dict(version=REQUEST_VERSION, video=str(video), number=int(number),
                output_dir=str(output_dir), model=model, options=dict(options),
                settings=dict(settings), background=dict(background or {}), trim=trim,
                compute_mode=compute_mode, resource_safety=bool(resource_safety),
                manual_people=manual_people, gpu=dict(gpu), result=str(result))


def write_request(path, request):
    atomic_json(Path(path), request)


def read_request(path):
    request = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(request, dict) or request.get('version') != REQUEST_VERSION:
        raise ProtocolError('Unsupported batch child request version')
    missing = [key for key in REQUIRED_REQUEST_KEYS if key not in request]
    if missing:
        raise ProtocolError('Batch child request is missing ' + ', '.join(missing))
    return request


def write_result(path, result):
    atomic_json(Path(path), result)


def read_result(path):
    """Never raises: a missing or unreadable result is reported as a crash."""
    try:
        result = json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    if not isinstance(result, dict) or result.get('version') != RESULT_VERSION:
        return None
    return result


def build_result(*, request, ok, status, error='', exports=(), project='', tail=''):
    return dict(version=RESULT_VERSION, video=request['video'], number=request['number'],
                ok=bool(ok), status=status, error=str(error), exports=[str(e) for e in exports],
                project=str(project), gpu=dict(request['gpu']),
                tail=str(tail)[-RESULT_TAIL_BYTES:])


def emit(event):
    """One JSON event per stdout line; -u keeps the pipe unbuffered."""
    print(EVENT_PREFIX + json.dumps(event, ensure_ascii=False), flush=True)


def parse_line(line):
    text = line.strip()
    if not text.startswith(EVENT_PREFIX):
        return None
    try:
        event = json.loads(text[len(EVENT_PREFIX):])
    except ValueError:
        return None
    return event if isinstance(event, dict) else None


def append_tail(buffer, line):
    """Keep a bounded tail of non-protocol output for failure reporting."""
    text = line if line.endswith('\n') else line + '\n'
    return (buffer + text)[-TAIL_BYTES:]
