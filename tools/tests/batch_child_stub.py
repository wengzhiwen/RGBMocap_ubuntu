"""Scripted stand-in for the headless batch child; a test fixture, not product code.

Reads the same --request file the real child reads, prints @@MOCAPBATCH@@ events,
writes the result file and exits with the protocol's exit codes. Scenarios:
ok, slow-ok, fail, crash, ignore-term, overlap.
"""
from pathlib import Path
import argparse
import os
import signal
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mocap import batch_protocol  # noqa: E402


def overlap_marker():
    return os.environ.get('BATCH_STUB_OVERLAP')


def note(marker, text):
    if marker:
        with open(marker, 'a', encoding='utf-8') as handle:
            handle.write(text + '\n')


def finish_ok(request, slow):
    batch_protocol.emit(dict(type='phase', phase='reconstructing'))
    batch_protocol.emit(dict(type='progress', status='Working', task='stub', value=1, maximum=4))
    time.sleep(1.0 if slow else 0.05)
    batch_protocol.emit(dict(type='phase', phase='exporting'))
    result = batch_protocol.build_result(
        request=request, ok=True, status='done',
        exports=[str(Path(request['output_dir']) / 'motion.fbx')],
        project=str(Path(request['output_dir'])))
    batch_protocol.write_result(request['result'], result)
    batch_protocol.emit(dict(type='result', **{k: v for k, v in result.items() if k != 'version'}))
    return batch_protocol.EXIT_OK


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True)
    parser.add_argument('--scenario', default='ok')
    args = parser.parse_args()
    request = batch_protocol.read_request(args.request)
    marker = overlap_marker()
    if args.scenario == 'crash':
        print('stub noise before the crash')
        print('another noise line', file=sys.stderr)
        return 1
    if args.scenario == 'ignore-term':
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        print('stub ignoring SIGTERM', flush=True)
        time.sleep(30)
        return 1
    note(marker, f"+{request['number']}")
    try:
        batch_protocol.emit(dict(type='phase', phase='opening'))
        time.sleep(0.02)
        if args.scenario == 'fail':
            result = batch_protocol.build_result(request=request, ok=False, status='failed',
                                                 error='stub failure')
            batch_protocol.write_result(request['result'], result)
            batch_protocol.emit(dict(type='result',
                                     **{k: v for k, v in result.items() if k != 'version'}))
            return batch_protocol.EXIT_FAILED
        return finish_ok(request, args.scenario == 'slow-ok')
    finally:
        note(marker, f"-{request['number']}")


if __name__ == '__main__':
    raise SystemExit(main())
