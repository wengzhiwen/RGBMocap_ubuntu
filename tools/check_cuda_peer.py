#!/usr/bin/env python3
"""Verify real FP32 copies between two CUDA devices. See docs/CUDA_PEER_CHECK.md.

The launcher uses the standard library and never imports Torch. Copies run in the
validated, isolated Flow runtime. This tool changes no OS, driver or service setting,
loads no model and opens no network port.
"""
import argparse
from collections import deque
import csv
from itertools import product
import json
import math
import os
from pathlib import Path
import re
import select
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PREFIX = '@@CUDA_PEER@@'
WORKER_FLAG = '--isolated-worker'
LAYOUTS = ('contiguous', 'noncontiguous')
# Bounds tensor memory and keeps every arange value plus case offset exact in FP32.
MAX_ELEMENTS = 2 ** 23
MAX_TIMEOUT = 3600.
MAX_LINE_BYTES = 64 * 1024
TAIL_LINES = 40
EXIT_CODES = dict(PASS=0, FAIL=1, NOT_APPLICABLE=3, ERROR=4, REFUSED=5, TIMEOUT=124, CANCELLED=130)
CANCEL_SIGNALS = [getattr(signal, name) for name in ('SIGINT', 'SIGTERM', 'SIGHUP') if hasattr(signal, name)]


def plan(transport):
    """Host controls run first, so a direct hang or driver fault cannot cost their evidence."""
    transports = ('host', 'direct') if transport == 'both' else (transport,)
    return [dict(id=index, transport=name, source=source, destination=1 - source, layout=layout)
            for index, (name, source, layout) in enumerate(product(transports, (0, 1), LAYOUTS))]


def compare_values(expected, received):
    """Compare every FP32 value by bit pattern, with no Torch or GPU needed to test it."""
    result = dict(values_checked=min(len(expected), len(received)) // 4, mismatches=0, max_abs_diff=0.,
                  nonfinite_received=0, first_mismatch=None)
    if expected == received:
        return result
    wanted, actual = memoryview(expected).cast('f'), memoryview(received).cast('f')
    largest = None
    for index, (left, right) in enumerate(zip(memoryview(expected).cast('I'), memoryview(received).cast('I'))):
        if left == right:
            continue
        value = actual[index]
        finite = math.isfinite(value)
        if result['first_mismatch'] is None:
            # Corrupt data can hold NaN or infinity, which strict JSON cannot carry as numbers.
            result['first_mismatch'] = dict(index=index, expected=wanted[index], received=value if finite else repr(value),
                                            received_bits=f'{right:#010x}')
        result['mismatches'] += 1
        if finite:
            largest = max(largest or 0., abs(value - wanted[index]))
        else:
            result['nonfinite_received'] += 1
    result['max_abs_diff'] = largest
    return result


# Isolated Flow-runtime worker ---------------------------------------------------------

def emit(**event):
    print(PREFIX + json.dumps(event, allow_nan=False), flush=True)


def readback(tensor):
    return tensor.cpu().contiguous().numpy().tobytes()


def run_case(torch, planned, elements):
    """One real copy. Expected values exist on the CPU and are never trusted from a GPU."""
    source_device, destination_device = f"cuda:{planned['source']}", f"cuda:{planned['destination']}"
    contiguous = planned['layout'] == 'contiguous'
    # A distinct offset per case stops reused allocator memory from passing as a fresh copy.
    values = torch.arange(elements, dtype=torch.float32).add_(planned['id'])
    expected = values.numpy().tobytes()
    if contiguous:
        source = values.to(source_device)
    else:
        # A stepped slice is truly sparse. Transposed dense data could still be a plain memcpy.
        stored = torch.full((2 * elements,), -1., dtype=torch.float32)
        stored[::2] = values
        source = stored.to(source_device)[::2]
    if source.is_contiguous() != contiguous:
        raise RuntimeError(f"Source layout is not {planned['layout']}; the case would not test that path.")
    uploaded = compare_values(expected, readback(source))
    if uploaded['mismatches'] or uploaded['values_checked'] != elements:
        raise RuntimeError(f'Source GPU readback was wrong before any peer transfer, so this is not peer evidence: {uploaded}')
    moved = source.to(destination_device) if planned['transport'] == 'direct' else source.cpu().to(destination_device)
    if moved.device != torch.device(destination_device) or moved.dtype != values.dtype or moved.shape != values.shape:
        raise RuntimeError(f'Unexpected destination tensor: {moved.device}, {moved.dtype}, {tuple(moved.shape)}.')
    result = compare_values(expected, readback(moved))
    return dict(result, exact=result['mismatches'] == 0 and result['values_checked'] == elements)


def worker(encoded):
    request = json.loads(encoded)
    # Default SIGALRM action ends the process, even if the launcher itself was SIGKILLed.
    signal.alarm(math.ceil(request['timeout']) + 30)
    import torch
    from mocap.compute_policy import gpu_memory_fraction
    gpus = request['gpus']
    if torch.cuda.device_count() != 2:
        raise RuntimeError(f'Expected the two selected GPUs, but Torch sees {torch.cuda.device_count()}.')
    devices = []
    for index, gpu in enumerate(gpus):
        free, total = torch.cuda.mem_get_info(index)
        fraction = max(1024 ** 2 / total, gpu_memory_fraction(free, total))
        torch.cuda.set_per_process_memory_fraction(fraction, index)
        properties = torch.cuda.get_device_properties(index)
        devices.append(dict(logical=gpu['logical'], worker_device=f'cuda:{index}', torch_name=properties.name,
                            torch_uuid=str(getattr(properties, 'uuid', '') or '') or None, free_bytes=free,
                            total_bytes=total, vram_budget_fraction=fraction))
    emit(type='environment', python=sys.version.split()[0], torch=torch.__version__, cuda=torch.version.cuda, devices=devices,
         peer_access={f"{gpus[source]['logical']}->{gpus[1 - source]['logical']}": bool(torch.cuda.can_device_access_peer(source, 1 - source))
                      for source in (0, 1)})
    for planned in plan(request['transport']):
        source, destination = gpus[planned['source']], gpus[planned['destination']]
        case = dict(id=planned['id'], transport=planned['transport'], layout=planned['layout'],
                    source_logical=source['logical'], destination_logical=destination['logical'],
                    source_uuid=source['uuid'], destination_uuid=destination['uuid'], exact=False, error=None)
        try:
            case.update(run_case(torch, planned, request['elements']))
        except Exception as error:
            case['error'] = f'{type(error).__name__}: {error}'[:2000]
        emit(type='case', **case)
        torch.cuda.empty_cache()


# Host state ---------------------------------------------------------------------------

def visible_gpus():
    """Visible selection order, using nvidia-smi indices when no UUID list is set."""
    keys = ('index', 'uuid', 'name', 'driver', 'pci')
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,driver_version,pci.bus_id',
                                      '--format=csv,noheader'], text=True, timeout=15)
    gpus = [dict(zip(keys, map(str.strip, row))) for row in csv.reader(output.splitlines()) if row]
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible is None:
        return gpus
    selected = []
    for token in map(str.strip, visible.split(',')):
        if token in ('', '-1'):  # CUDA hides this entry and everything after it.
            break
        matches = [gpu for gpu in gpus if token == gpu['index'] or gpu['uuid'].startswith(token)]
        if len(matches) != 1:
            raise ValueError(f'Cannot map CUDA_VISIBLE_DEVICES entry {token!r} to one physical GPU.')
        selected.append(matches[0])
    return selected


def iommu_state(gpus, root='/'):
    """Readable kernel state only. Nothing here needs privileges or changes the host."""
    root = Path(root)

    def read(path):
        try:
            return path.read_text(encoding='utf-8', errors='replace').strip()
        except OSError:
            return None
    cmdline = read(root / 'proc/cmdline')
    tokens = None if cmdline is None else [token for token in cmdline.split() if token.startswith(('iommu=', 'iommu.', 'intel_iommu', 'amd_iommu'))]
    state = dict(kernel_arguments=tokens, units=sorted(path.name for path in (root / 'sys/class/iommu').glob('*')), gpus=[])
    for gpu in gpus:
        # nvidia-smi prints an eight-digit PCI domain; sysfs uses four.
        group = root / 'sys/bus/pci/devices' / gpu['pci'][-12:].lower() / 'iommu_group'
        state['gpus'].append(dict(logical=gpu['logical'], pci=gpu['pci'], group=group.resolve().name if group.exists() else None,
                                  type=read(group / 'type')))
    return state


# Bounded supervision ------------------------------------------------------------------

def group_members(group):
    """Live process-group members and kernel states; D means stuck in uninterruptible wait."""
    members = {}
    for entry in Path('/proc').iterdir():
        if entry.name.isdigit():
            try:
                fields = (entry / 'stat').read_bytes().rpartition(b')')[2].split()
            except OSError:
                continue
            if int(fields[2]) == group and fields[0] != b'Z':
                members[entry.name] = fields[0].decode('ascii', 'replace')
    return members


def stop_tree(process, grace=5., settle=5.):
    """ubuntu_cli.stop_group with a limit: a driver hang can leave a process unkillable."""
    def send(number):
        try:
            os.killpg(process.pid, number)
        except ProcessLookupError:
            return False
        return True
    if send(signal.SIGTERM):
        try:
            process.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            pass
    # A parent can exit while a child ignores SIGTERM and retains stdout.
    send(signal.SIGKILL)
    deadline = time.monotonic() + settle
    while True:
        process.poll()
        survivors = group_members(process.pid)
        if not survivors or time.monotonic() >= deadline:
            return survivors
        time.sleep(.05)


class LineReader:
    """Bounded line framing over raw reads: an open pipe can never block supervision."""
    def __init__(self, descriptor, on_line):
        self.descriptor, self.on_line = descriptor, on_line
        self.pending, self.closed = b'', False

    def read(self, wait):
        if self.closed:
            time.sleep(wait)
            return False
        if not select.select([self.descriptor], [], [], wait)[0]:
            return False
        chunk = os.read(self.descriptor, 65536)
        if not chunk:
            self.closed = True
            lines, self.pending = [self.pending] if self.pending else [], b''
        else:
            *lines, self.pending = (self.pending + chunk).split(b'\n')
            if len(self.pending) > MAX_LINE_BYTES:
                lines, self.pending = [*lines, self.pending], b''
        for line in lines:
            self.on_line(line.decode('utf-8', 'replace'))
        return True

    def drain(self, limit):
        deadline = time.monotonic() + limit
        while not self.closed and time.monotonic() < deadline:
            self.read(.05)


def load_guard():
    """The GUI's measured RAM and stall policy, confirmed available before a worker starts."""
    from importlib.util import find_spec
    if find_spec('psutil') is None:
        raise RuntimeError('Resource Safety needs psutil. Launch with .venv/bin/python; see docs/CUDA_PEER_CHECK.md.')
    from mocap.processing_guard import ProcessingGuard
    return lambda pid: ProcessingGuard(pid, resource_safety=True)


def supervise(command, env, timeout, guard_factory, on_line, grace=5., settle=5.):
    """Run one owned process group. Total time is bounded even if a member cannot be killed."""
    cancelled = []
    previous = {number: signal.signal(number, lambda *_: cancelled.append(number)) for number in CANCEL_SIGNALS}
    outcome = dict(termination=None, message=None, returncode=None, survivors={})
    process = lines = None
    try:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, bufsize=0, start_new_session=True)
        lines = LineReader(process.stdout.fileno(), on_line)
        guard = guard_factory(process.pid)
        started = time.monotonic()
        deadline, next_check = started + timeout, started + .5
        while process.poll() is None:
            now = time.monotonic()
            if cancelled:
                outcome['termination'] = 'cancelled'
                break
            if now >= deadline:
                outcome.update(termination='timeout', message=f'Stopped after the {timeout:g} s limit.')
                break
            if now >= next_check:
                next_check = now + .5
                try:
                    message = guard.check()
                except Exception as error:
                    message = f'Worker supervision failed: {error}'
                if message:
                    outcome.update(termination='guard', message=message)
                    break
            if lines.read(min(.25, deadline - now)):
                guard.output()
    finally:
        if process is not None:
            outcome['survivors'] = stop_tree(process, grace, settle)
            if lines is not None:
                lines.drain(2.)  # Keep evidence still in the pipe when the worker ended.
            outcome['returncode'] = process.poll()
            process.stdout.close()
        for number, handler in previous.items():
            signal.signal(number, handler)
    return outcome


def worker_command(request):
    from mocap.linux_runtime import SETUP_HINT, interpreter
    python = interpreter(ROOT, 'flow')
    if not python.is_file():
        raise RuntimeError(f'Flow runtime is missing ({python}). {SETUP_HINT}')
    # A script path, not -c: a site-packages "tools" package could shadow this namespace folder.
    return [str(python), '-I', '-B', '-X', 'utf8', str(Path(__file__).resolve()), WORKER_FLAG, json.dumps(request)]


# Report -------------------------------------------------------------------------------

def case_passed(case, elements):
    return (case.get('error') is None and case.get('exact') is True
            and case.get('values_checked') == elements and case.get('mismatches') == 0)


def conclude(report):
    """Direct failure stays FAIL whatever else succeeded or however the run ended."""
    request, worker_state = report['request'], report['worker']
    planned = plan(request['transport'])
    cases = report['cases']
    elements = request['elements']

    def status(name):
        wanted = [row['id'] for row in planned if row['transport'] == name]
        own = [case for case in cases if case.get('transport') == name]
        if not wanted:
            return 'NOT_TESTED'
        if not all(case_passed(case, elements) for case in own):
            return 'FAIL'
        return 'PASS' if sorted(case['id'] for case in own) == wanted else 'INCOMPLETE'
    report['transports'] = dict(direct=status('direct'), host=status('host'))
    direct, host = report['transports']['direct'], report['transports']['host']
    termination, survivors = worker_state['termination'], worker_state['survivors']
    complete = sorted(case['id'] for case in cases) == [row['id'] for row in planned]
    if not all(case_passed(case, elements) for case in cases):
        verdict = 'FAIL'
    elif termination in ('timeout', 'cancelled'):
        verdict = termination.upper()
    elif termination or worker_state['returncode'] != 0 or survivors or not complete:
        verdict = 'ERROR'
    else:
        verdict = 'PASS'
    report['verdict'] = verdict
    notes = report['messages']
    if worker_state['message']:
        notes.append(worker_state['message'])
    if survivors:
        notes.append(f'Process-group members survived SIGKILL (pid: kernel state): {survivors}. '
                     'State D suggests a stuck driver wait; these processes may hold the GPUs.')
    if direct == 'FAIL':
        notes.append('Direct transfer FAILED.' + (' Explicit CPU-staged copies were exact, so staging is a usable workaround; '
                     'that does not clear direct transfer.' if host == 'PASS' else ''))
        if any((report.get('environment') or {}).get('peer_access', {}).values()):
            notes.append('can_device_access_peer=True does not establish correct data transfer.')
        translated = sorted({gpu['type'] for gpu in (report.get('iommu') or {}).get('gpus', []) if str(gpu.get('type')).startswith('DMA')})
        if translated:
            notes.append(f'GPU IOMMU groups use translated DMA domains {translated}. This is a strong diagnostic lead, not a proven exclusive cause.')
    if host == 'FAIL':
        notes.append('CPU-staged copies also failed, so the fault is not limited to direct GPU-to-GPU transfer.')
    if direct == 'PASS':
        notes.append('Direct copies were exact in this one run. This does not show which hardware path carried them, or validate NCCL.')
    elif direct == 'NOT_TESTED' and verdict == 'PASS':
        notes.append('Direct transfer was not tested. This result is not P2P evidence.')


def new_report(args):
    return dict(schema=1, tool='check_cuda_peer', verdict=None, exit_code=None,
                transports=dict(direct='NOT_TESTED', host='NOT_TESTED'),
                request=dict(transport=args.transport, devices=list(args.devices), elements=args.elements,
                             timeout_seconds=args.timeout, resource_safety=True),
                gpus=[], iommu=None, environment=None, cuda_environment=None, cases=[],
                worker=dict(termination=None, message=None, returncode=None, survivors={}, output_tail=[]), messages=[])


def decode_event(line):
    if not line.startswith(PREFIX):
        return None
    try:
        event = json.loads(line[len(PREFIX):])
    except ValueError:
        return None
    usable = isinstance(event, dict) and (event.get('type') == 'environment' or (event.get('type') == 'case' and isinstance(event.get('id'), int)))
    return event if usable else None


def case_line(case):
    state = 'ERROR' if case.get('error') else 'exact' if case.get('exact') is True else 'WRONG'
    text = f"{str(case.get('transport')):6} GPU {case.get('source_logical')} -> GPU {case.get('destination_logical')} {str(case.get('layout')):13} {state}"
    if case.get('error'):
        return f"{text} {case['error']}"
    return f"{text} max_abs_diff={case.get('max_abs_diff')} mismatches={case.get('mismatches')}/{case.get('values_checked')}"


def run(args, report):
    """Refuse unusable hosts before any worker starts; otherwise collect evidence into report."""
    if os.name != 'posix':
        raise RuntimeError('This check supports the Linux route only; see docs/UBUNTU.md.')
    from tools.benchmark_flow import environment, idle_devices
    gpus = visible_gpus()
    if len(gpus) < 2:
        report['verdict'] = 'NOT_APPLICABLE'
        report['messages'].append(f'Not applicable: {len(gpus)} visible GPU(s), and this check needs two. '
                                  'Peer transfer was not tested; this is not a P2P pass.')
        return
    if max(args.devices) >= len(gpus):
        raise ValueError(f'--devices {args.devices[0]},{args.devices[1]} is outside the {len(gpus)} visible GPUs.')
    selected = [dict(gpus[index], logical=index) for index in args.devices]
    if selected[0]['uuid'] == selected[1]['uuid']:
        raise ValueError('--devices selects one physical GPU twice.')
    report.update(gpus=selected, iommu=iommu_state(selected))
    # idle_devices follows this variable, so the contention check covers exactly this pair.
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpu['uuid'] for gpu in selected)
    try:
        idle_devices('pp2')
    except RuntimeError as error:
        report['verdict'] = 'REFUSED'
        report['messages'].append(f'Refused: {error}. No workload was stopped.')
        return
    guard_factory = load_guard()
    command = worker_command(dict(transport=args.transport, elements=args.elements, timeout=args.timeout, gpus=selected))
    env = environment()
    report['cuda_environment'] = {key: value for key, value in env.items() if key.startswith(('CUDA_', 'PYTORCH_CUDA_'))}
    tail = deque(maxlen=TAIL_LINES)

    def on_line(line):
        event = decode_event(line)
        if event is None:
            tail.append(line[:2000])
            print(line, file=sys.stderr, flush=True)
        elif event['type'] == 'environment':
            report['environment'] = event
        else:
            report['cases'].append(event)
            print(case_line(event), file=sys.stderr, flush=True)
    try:
        report['worker'].update(supervise(command, env, args.timeout, guard_factory, on_line))
    finally:
        report['worker']['output_tail'] = list(tail)


def print_summary(report):
    print(f"CUDA peer check: {report['verdict']} (exit {report['exit_code']})")
    print(f"Transports: direct={report['transports']['direct']} host={report['transports']['host']}")
    for gpu in report['gpus']:
        print(f"GPU {gpu['logical']}: {gpu['name']} {gpu['uuid']} PCI {gpu['pci']} driver {gpu['driver']}")
    environment = report['environment']
    if environment:
        print(f"Torch {environment['torch']}, CUDA {environment['cuda']}, can_device_access_peer {environment['peer_access']}")
    if report['iommu']:
        print('IOMMU:', json.dumps(report['iommu']))
    for case in report['cases']:
        print(case_line(case))
    for message in report['messages']:
        print(message)


def device_pair(value):
    if not re.fullmatch(r'\d+,\d+', value):
        raise argparse.ArgumentTypeError('use two logical GPU indices, such as 0,1')
    pair = tuple(map(int, value.split(',')))
    if pair[0] == pair[1]:
        raise argparse.ArgumentTypeError('the two devices must be distinct')
    return pair


def element_count(value):
    count = int(value)
    if not 2 <= count <= MAX_ELEMENTS:
        raise argparse.ArgumentTypeError(f'must be from 2 to {MAX_ELEMENTS}')
    return count


def finite_timeout(value):
    seconds = float(value)
    if not math.isfinite(seconds) or not 0 < seconds <= MAX_TIMEOUT:
        raise argparse.ArgumentTypeError(f'must be finite, above 0 and at most {MAX_TIMEOUT:g}')
    return seconds


class Parser(argparse.ArgumentParser):
    json_output = False

    def error(self, message):
        if self.json_output:
            print(json.dumps(dict(schema=1, tool='check_cuda_peer', verdict='ERROR', exit_code=2,
                                  transports=dict(direct='NOT_TESTED', host='NOT_TESTED'), messages=[message])))
        super().error(message)


def parser():
    root = Parser(description=__doc__, allow_abbrev=False, formatter_class=argparse.RawDescriptionHelpFormatter)
    root.add_argument('--transport', choices=('direct', 'host', 'both'), default='both',
                      help='direct: Tensor.to between GPUs; host: explicit CPU staging; both: compare them (default).')
    root.add_argument('--devices', type=device_pair, default=(0, 1), metavar='A,B',
                      help='Two distinct logical indices within the visible GPUs (default 0,1).')
    root.add_argument('--elements', type=element_count, default=2 ** 20, metavar='N',
                      help=f'FP32 values per copy, at most {MAX_ELEMENTS} (default {2 ** 20}).')
    root.add_argument('--timeout', type=finite_timeout, default=300., metavar='SECONDS',
                      help=f'Worker time limit, at most {MAX_TIMEOUT:g} (default 300).')
    root.add_argument('--json', action='store_true', help='Print one JSON report on stdout; diagnostics go to stderr.')
    return root


def interrupt(*_):
    raise KeyboardInterrupt


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    root = parser()
    root.json_output = '--json' in argv
    args = root.parse_args(argv)
    report = new_report(args)
    previous = {number: signal.signal(number, interrupt) for number in CANCEL_SIGNALS}
    try:
        run(args, report)
    except KeyboardInterrupt:
        report['worker']['termination'] = 'cancelled'
    except Exception as error:  # A report boundary: keep the output contract and all evidence.
        report['worker']['termination'] = 'error'
        report['messages'].append(f'{type(error).__name__}: {error}')
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)
    if report['verdict'] is None:
        conclude(report)
    report['exit_code'] = EXIT_CODES[report['verdict']]
    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
    else:
        print_summary(report)
    return report['exit_code']


if __name__ == '__main__':
    if sys.argv[1:2] == [WORKER_FLAG]:
        from mocap.runtime_registry import activate
        activate(ROOT, 'flow')
        worker(sys.argv[2])
    else:
        raise SystemExit(main())
