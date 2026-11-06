"""CPU checks of validation, comparison, reporting and process cleanup; none is GPU evidence."""
from array import array
import contextlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

if os.name != 'posix':
    raise unittest.SkipTest('The peer check supports the Linux route only.')

from tools import check_cuda_peer as peer  # noqa: E402

ROWS = ('0, GPU-aaa, NVIDIA RTX PRO 4000 Blackwell, 595.99.02, 00000000:01:00.0',
        '1, GPU-bbb, NVIDIA RTX PRO 4000 Blackwell, 595.99.02, 00000000:02:00.0',
        '2, GPU-ccc, NVIDIA RTX PRO 4000 Blackwell, 595.99.02, 00000000:03:00.0')


def fake_smi(rows, apps=''):
    """Answers both this tool's query and the reused idle_devices query."""
    def check_output(command, **_):
        if command[1].startswith('--query-compute-apps'):
            return apps
        if 'driver_version' in command[1]:
            return '\n'.join(rows) + '\n'
        return ''.join(', '.join(row.split(', ')[:3]) + '\n' for row in rows)
    return check_output


def call_main(*argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = peer.main(list(argv))
    return code, out.getvalue(), err.getvalue()


def worker_outcome(**values):
    return dict(dict(termination=None, message=None, returncode=0, survivors={}), **values)


def case(identifier, transport, source=0, layout='contiguous', **values):
    """Hand-written verdict-logic input, never a measurement."""
    base = dict(type='case', id=identifier, transport=transport, layout=layout, source_logical=source,
                destination_logical=1 - source, exact=True, error=None, values_checked=4, mismatches=0, max_abs_diff=0.)
    return dict(base, **values)


def report_for(transport, cases, **worker):
    args = peer.parser().parse_args(['--transport', transport, '--elements', '4'])
    report = peer.new_report(args)
    report['cases'] = cases
    report['worker'].update(worker_outcome(**worker))
    peer.conclude(report)
    return report


def all_cases(transport):
    return [case(row['id'], row['transport'], row['source'], row['layout']) for row in peer.plan(transport)]


class Arguments(unittest.TestCase):
    def test_invalid_values_fail_before_any_gpu_query_with_parseable_json(self):
        invalid = (['--devices', '0,0'], ['--devices', '1'], ['--devices', 'a,b'], ['--devices', '-1,0'],
                   ['--timeout', 'nan'], ['--timeout', 'inf'], ['--timeout', '0'], ['--timeout', '-5'], ['--timeout', '99999'],
                   ['--elements', '1'], ['--elements', str(peer.MAX_ELEMENTS + 1)], ['--transport', 'p2p'])
        for argv in invalid:
            out, err = io.StringIO(), io.StringIO()
            with self.subTest(argv=argv), patch('subprocess.check_output', side_effect=AssertionError), \
                    contextlib.redirect_stdout(out), contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as caught:
                peer.main(['--json', *argv])
            self.assertEqual(caught.exception.code, 2)
            self.assertEqual(json.loads(out.getvalue())['verdict'], 'ERROR')

    def test_plan_covers_each_direction_and_layout_with_host_controls_first(self):
        rows = peer.plan('both')
        self.assertEqual(len(rows), 8)
        self.assertEqual([row['transport'] for row in rows], ['host'] * 4 + ['direct'] * 4)
        self.assertEqual({(row['transport'], row['source'], row['destination'], row['layout']) for row in rows},
                         {(t, s, 1 - s, l) for t in ('host', 'direct') for s in (0, 1) for l in peer.LAYOUTS})
        self.assertEqual([row['id'] for row in rows], list(range(8)))


class Comparison(unittest.TestCase):
    def test_zero_filled_destination_reports_every_wrong_value(self):
        count = 2 ** 20
        result = peer.compare_values(array('f', range(count)).tobytes(), bytes(4 * count))
        self.assertEqual(result['values_checked'], count)
        self.assertEqual(result['mismatches'], count - 1)  # Value zero happens to match.
        self.assertEqual(result['max_abs_diff'], 1048575.)
        self.assertEqual(result['first_mismatch']['index'], 1)

    def test_exact_copy_and_bit_level_differences(self):
        values = array('f', [1., 0., 3.])
        self.assertEqual(peer.compare_values(values.tobytes(), values.tobytes())['mismatches'], 0)
        signed_zero = peer.compare_values(values.tobytes(), array('f', [1., -0., 3.]).tobytes())
        self.assertEqual((signed_zero['mismatches'], signed_zero['max_abs_diff']), (1, 0.))

    def test_nonfinite_data_stays_failed_and_strict_json(self):
        result = peer.compare_values(array('f', [1., 2.]).tobytes(), array('f', [math.nan, math.inf]).tobytes())
        self.assertEqual((result['mismatches'], result['nonfinite_received'], result['max_abs_diff']), (2, 2, None))
        json.dumps(result, allow_nan=False)

    def test_truncated_data_cannot_pass_a_case(self):
        values = array('f', [1., 2., 3., 4.]).tobytes()
        result = peer.compare_values(values, values[:8])
        self.assertEqual(result['values_checked'], 2)
        self.assertFalse(peer.case_passed(case(0, 'direct', **result), 4))


class Verdicts(unittest.TestCase):
    def test_direct_failure_stays_fail_when_host_staging_is_exact(self):
        cases = all_cases('both')
        cases[4].update(exact=False, mismatches=4, max_abs_diff=3.)
        report = report_for('both', cases)
        self.assertEqual(report['verdict'], 'FAIL')
        self.assertEqual(report['transports'], dict(direct='FAIL', host='PASS'))
        self.assertEqual(peer.EXIT_CODES[report['verdict']], 1)
        self.assertIn('does not clear direct transfer', ' '.join(report['messages']))

    def test_direct_exception_is_a_failure_and_iommu_stays_a_lead(self):
        cases = all_cases('direct')
        cases[0].update(exact=False, error='RuntimeError: CUDA error')
        args = peer.parser().parse_args(['--transport', 'direct', '--elements', '4'])
        report = peer.new_report(args)
        report.update(cases=cases, iommu=dict(gpus=[dict(type='DMA-FQ')]), environment=dict(peer_access={'0->1': True}))
        report['worker'].update(worker_outcome())
        peer.conclude(report)
        text = ' '.join(report['messages'])
        self.assertEqual(report['verdict'], 'FAIL')
        self.assertIn('not a proven exclusive cause', text)
        self.assertIn('does not establish correct data transfer', text)

    def test_failure_evidence_survives_timeout_and_cancellation(self):
        for termination in ('timeout', 'cancelled'):
            report = report_for('both', [case(4, 'direct', exact=False, mismatches=4)], termination=termination, returncode=-9)
            self.assertEqual(report['verdict'], 'FAIL')
            self.assertEqual(report['worker']['termination'], termination)

    def test_incomplete_or_unclean_runs_never_pass(self):
        complete = all_cases('both')
        self.assertEqual(report_for('both', complete[:-1])['verdict'], 'ERROR')
        self.assertEqual(report_for('both', complete + complete[-1:])['verdict'], 'ERROR')
        self.assertEqual(report_for('both', complete, returncode=1)['verdict'], 'ERROR')
        self.assertEqual(report_for('both', complete, returncode=None)['verdict'], 'ERROR')
        self.assertEqual(report_for('both', complete, survivors={'7': 'D'})['verdict'], 'ERROR')
        self.assertEqual(report_for('both', complete, termination='guard', message='low RAM')['verdict'], 'ERROR')
        self.assertEqual(report_for('both', complete[:2], termination='timeout')['verdict'], 'TIMEOUT')
        self.assertEqual(report_for('both', [], termination='cancelled')['verdict'], 'CANCELLED')

    def test_short_case_fails_even_when_it_claims_exact(self):
        cases = all_cases('direct')
        cases[0]['values_checked'] = 3
        self.assertEqual(report_for('direct', cases)['verdict'], 'FAIL')

    def test_verdict_logic_for_complete_runs_and_host_only_scope(self):
        self.assertEqual(report_for('both', all_cases('both'))['verdict'], 'PASS')
        host = report_for('host', all_cases('host'))
        self.assertEqual(host['transports'], dict(direct='NOT_TESTED', host='PASS'))
        self.assertIn('not P2P evidence', ' '.join(host['messages']))


class Selection(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)
        os.environ.pop('CUDA_VISIBLE_DEVICES', None)

    def test_too_few_gpus_are_not_applicable_and_start_nothing(self):
        for rows, visible in ((ROWS[:1], None), (ROWS, ''), (ROWS, '-1'), (ROWS, 'GPU-bbb')):
            os.environ.pop('CUDA_VISIBLE_DEVICES', None)
            if visible is not None:
                os.environ['CUDA_VISIBLE_DEVICES'] = visible
            with self.subTest(visible=visible), patch('subprocess.check_output', fake_smi(rows)), \
                    patch.object(peer, 'supervise', side_effect=AssertionError):
                code, out, _ = call_main('--json')
                report = json.loads(out)
                self.assertEqual((code, report['verdict']), (3, 'NOT_APPLICABLE'))
                self.assertEqual(report['transports'], dict(direct='NOT_TESTED', host='NOT_TESTED'))
                self.assertIn('not a P2P pass', report['messages'][0])

    def test_selected_gpu_with_other_compute_work_is_refused_by_idle_devices(self):
        with patch('subprocess.check_output', fake_smi(ROWS, 'GPU-bbb, 4321, other-worker\n')), \
                patch.object(peer, 'supervise', side_effect=AssertionError):
            code, out, _ = call_main('--json', '--devices', '0,1')
        report = json.loads(out)
        self.assertEqual((code, report['verdict']), (5, 'REFUSED'))
        self.assertIn('4321', report['messages'][0])

    def test_work_on_an_unselected_gpu_does_not_block_and_selection_follows_visible_order(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = 'GPU-ccc,GPU-aaa,GPU-bbb'
        requests = []

        def command(request):
            requests.append(request)
            return ['unused']
        with patch('subprocess.check_output', fake_smi(ROWS, 'GPU-bbb, 4321, other-worker\n')), \
                patch.object(peer, 'load_guard'), patch.object(peer, 'worker_command', command), \
                patch.object(peer, 'supervise', return_value=worker_outcome(termination='cancelled')) as supervise:
            code, out, _ = call_main('--json', '--devices', '1,0')
        self.assertEqual((code, json.loads(out)['verdict']), (130, 'CANCELLED'))
        self.assertEqual([(gpu['logical'], gpu['uuid']) for gpu in requests[0]['gpus']], [(1, 'GPU-aaa'), (0, 'GPU-ccc')])
        environment = supervise.call_args.args[1]
        self.assertEqual(environment['CUDA_VISIBLE_DEVICES'], 'GPU-aaa,GPU-ccc')
        self.assertEqual(environment['RGB_MOCAP_RESOURCE_SAFETY'], '1')

    def test_selection_errors_and_missing_tools_report_json(self):
        with patch('subprocess.check_output', fake_smi(ROWS[:2])):
            code, out, _ = call_main('--json', '--devices', '0,5')
        self.assertEqual((code, json.loads(out)['verdict']), (4, 'ERROR'))
        os.environ['CUDA_VISIBLE_DEVICES'] = 'GPU-aaa,GPU-aaa'
        with patch('subprocess.check_output', fake_smi(ROWS[:2])):
            code, out, _ = call_main('--json')
        self.assertEqual(code, 4)
        self.assertIn('twice', json.loads(out)['messages'][0])
        with patch('subprocess.check_output', side_effect=FileNotFoundError('nvidia-smi')):
            code, out, _ = call_main('--json')
        self.assertEqual((code, json.loads(out)['verdict']), (4, 'ERROR'))

    def test_missing_guard_dependency_refuses_before_a_worker_starts(self):
        with patch('subprocess.check_output', fake_smi(ROWS)), patch('importlib.util.find_spec', return_value=None), \
                patch.object(peer, 'supervise', side_effect=AssertionError):
            code, out, _ = call_main('--json')
        self.assertEqual(code, 4)
        self.assertIn('psutil', json.loads(out)['messages'][0])


class IommuState(unittest.TestCase):
    def test_reads_only_iommu_tokens_and_gpu_group_type(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'proc').mkdir()
            (root / 'proc/cmdline').write_text('quiet intel_iommu=on iommu.strict=0 secret=hidden\n')
            (root / 'sys/class/iommu/dmar0').mkdir(parents=True)
            (root / 'groups/21').mkdir(parents=True)
            (root / 'groups/21/type').write_text('DMA-FQ\n')
            device = root / 'sys/bus/pci/devices/0000:01:00.0'
            device.mkdir(parents=True)
            (device / 'iommu_group').symlink_to(root / 'groups/21')
            state = peer.iommu_state([dict(logical=0, pci='00000000:01:00.0'), dict(logical=1, pci='[N/A]')], root)
        self.assertEqual(state['kernel_arguments'], ['intel_iommu=on', 'iommu.strict=0'])
        self.assertEqual(state['units'], ['dmar0'])
        self.assertEqual((state['gpus'][0]['group'], state['gpus'][0]['type']), ('21', 'DMA-FQ'))
        self.assertEqual((state['gpus'][1]['group'], state['gpus'][1]['type']), (None, None))

    def test_unreadable_state_is_reported_as_absent(self):
        with tempfile.TemporaryDirectory() as folder:
            state = peer.iommu_state([], folder)
        self.assertEqual(state, dict(kernel_arguments=None, units=[], gpus=[]))


class StubGuard:
    """Stands in for ProcessingGuard, whose psutil is absent from system Python."""
    def __init__(self, pid):
        self.pid = pid

    def output(self):
        pass

    def check(self):
        return None


# Leader and child both ignore SIGTERM, so only the group SIGKILL can end them.
STUBBORN_TREE = '''
import signal, subprocess, sys, time
child = subprocess.Popen([sys.executable, "-c", "import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);print('ready',flush=True);time.sleep(600)"], stdout=subprocess.PIPE)
child.stdout.readline()
signal.signal(signal.SIGTERM, signal.SIG_IGN)
print("child", child.pid, flush=True)
time.sleep(600)
'''


def alive(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_bytes().rpartition(b')')[2].split()
    except OSError:
        return False
    return fields[0] != b'Z'


class ProcessTree(unittest.TestCase):
    def supervise(self, script, timeout=30., guard=StubGuard):
        lines = []
        started = time.monotonic()
        outcome = peer.supervise([sys.executable, '-c', script], dict(os.environ), timeout, guard, lines.append, grace=.2, settle=5.)
        return outcome, lines, time.monotonic() - started

    def assert_tree_gone(self, outcome, lines):
        child = next(int(line.split()[1]) for line in lines if line.startswith('child'))
        self.assertEqual(outcome['survivors'], {})
        self.assertFalse(alive(child))

    def test_timeout_kills_a_tree_that_ignores_sigterm(self):
        outcome, lines, seconds = self.supervise(STUBBORN_TREE, timeout=1.5)
        self.assertEqual(outcome['termination'], 'timeout')
        self.assert_tree_gone(outcome, lines)
        self.assertLess(seconds, 20)

    def test_cancellation_signal_kills_the_tree_and_restores_handlers(self):
        before = signal.getsignal(signal.SIGTERM)
        lines = []

        def cancel_when_tree_is_up(line):
            lines.append(line)
            if line.startswith('child'):
                os.kill(os.getpid(), signal.SIGTERM)
        outcome = peer.supervise([sys.executable, '-c', STUBBORN_TREE], dict(os.environ), 30., StubGuard,
                                 cancel_when_tree_is_up, grace=.2, settle=5.)
        self.assertEqual(outcome['termination'], 'cancelled')
        self.assert_tree_gone(outcome, lines)
        self.assertEqual(signal.getsignal(signal.SIGTERM), before)

    def test_guard_stop_preserves_its_message_and_kills_the_tree(self):
        class LowMemory(StubGuard):
            def check(self):
                return 'Processing stopped because available system memory is low.'
        outcome, _, _ = self.supervise('import time;time.sleep(600)', guard=LowMemory)
        self.assertEqual((outcome['termination'], outcome['survivors']), ('guard', {}))
        self.assertIn('memory is low', outcome['message'])

    def test_exited_worker_with_a_child_holding_stdout_cannot_block_and_final_output_is_kept(self):
        script = ('import subprocess,sys;'
                  'child=subprocess.Popen([sys.executable,"-c","import time;time.sleep(600)"]);'
                  'print("child",child.pid,flush=True);print("final line",flush=True);sys.exit(7)')
        outcome, lines, seconds = self.supervise(script)
        self.assertEqual((outcome['termination'], outcome['returncode']), (None, 7))
        self.assertIn('final line', lines)
        self.assert_tree_gone(outcome, lines)
        self.assertLess(seconds, 20)

    def test_output_left_in_the_pipe_by_an_exited_worker_is_kept(self):
        def after_worker_exit(pid):
            os.waitid(os.P_PID, pid, os.WEXITED | os.WNOWAIT)  # Nothing has been read yet.
            return StubGuard(pid)
        outcome, lines, _ = self.supervise('print("last evidence", flush=True)', guard=after_worker_exit)
        self.assertEqual(outcome['returncode'], 0)
        self.assertEqual(lines, ['last evidence'])

    def test_guard_policy_always_enables_resource_safety(self):
        with patch('importlib.util.find_spec', return_value=True), patch('mocap.processing_guard.ProcessingGuard') as guard:
            peer.load_guard()(123)
        guard.assert_called_once_with(123, resource_safety=True)

    def test_unfinished_line_and_oversized_output_stay_bounded(self):
        script = f'import sys;sys.stdout.write("x"*{peer.MAX_LINE_BYTES * 3});sys.stdout.flush();sys.stdout.write("end")'
        outcome, lines, _ = self.supervise(script)
        self.assertEqual(outcome['returncode'], 0)
        self.assertEqual(sum(map(len, lines)), peer.MAX_LINE_BYTES * 3 + 3)
        self.assertLess(max(map(len, lines)), peer.MAX_LINE_BYTES + 65537)


# Corrupt-copy evidence, then native noise, then a hang: only failure is ever simulated.
FAILING_WORKER = f'''
import json, time
event = dict(type="case", id=4, transport="direct", layout="contiguous", source_logical=0, destination_logical=1,
             exact=False, error=None, values_checked=4, mismatches=3, max_abs_diff=3.0)
print("native library noise", flush=True)
print({peer.PREFIX!r} + json.dumps(event), flush=True)
time.sleep(600)
'''


class EndToEndFailure(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)
        os.environ.pop('CUDA_VISIBLE_DEVICES', None)

    def run_failure(self, *argv):
        with patch('subprocess.check_output', fake_smi(ROWS)), patch.object(peer, 'load_guard', return_value=StubGuard), \
                patch.object(peer, 'worker_command', return_value=[sys.executable, '-c', FAILING_WORKER]):
            return call_main('--elements', '4', '--timeout', '1', *argv)

    def test_json_stdout_is_one_document_with_noise_routed_to_stderr_and_evidence_kept(self):
        code, out, err = self.run_failure('--json')
        report = json.loads(out)
        self.assertEqual(code, 1)
        self.assertEqual(report['verdict'], 'FAIL')
        self.assertEqual(report['worker']['termination'], 'timeout')
        self.assertEqual(report['cases'][0]['max_abs_diff'], 3.)
        self.assertEqual(report['worker']['output_tail'], ['native library noise'])
        self.assertIn('native library noise', err)

    def test_text_summary_states_the_failure(self):
        code, out, _ = self.run_failure()
        self.assertEqual(code, 1)
        self.assertIn('CUDA peer check: FAIL', out)
        self.assertIn('direct=FAIL', out)
        self.assertIn('WRONG', out)

    def test_missing_flow_runtime_is_reported_without_a_process(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(peer, 'ROOT', Path(folder)):
            with self.assertRaisesRegex(RuntimeError, 'Flow runtime is missing'):
                peer.worker_command({})


@unittest.skipUnless(importlib.util.find_spec('torch'), 'requires Torch')
class TorchCpuHelpers(unittest.TestCase):
    def test_stepped_slice_is_noncontiguous_and_reads_back_only_selected_values(self):
        import torch
        stored = torch.full((8,), -1., dtype=torch.float32)
        values = torch.arange(4, dtype=torch.float32)
        stored[::2] = values
        view = stored[::2]
        self.assertFalse(view.is_contiguous())
        self.assertEqual(peer.compare_values(values.numpy().tobytes(), peer.readback(view))['mismatches'], 0)


if __name__ == '__main__':
    unittest.main()
