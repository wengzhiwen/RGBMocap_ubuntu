"""Startup/update failures use temporary roots, never installed environments."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mocap import linux_runtime
from tools import linux_startup, setup_ubuntu as setup
from tools.session_lock import app_lock


class TemporaryRoot(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)


class PythonReuse(TemporaryRoot):
    def setUp(self):
        super().setUp()
        self.venv = self.root / '.venv'
        self.venv.mkdir()

    def probe(self, version='3.10.22', machine=None, prefix=None, base='/python', allow=False):
        payload = [version, machine or linux_runtime.normalize_machine(), str(prefix or self.venv), base]
        result = SimpleNamespace(stdout=json.dumps(payload))
        with mock.patch.object(setup.subprocess, 'run', return_value=result), mock.patch('builtins.print'):
            return setup.existing_python(self.venv, '3.10.22', allow)

    def test_exact_python_is_reused_without_uv(self):
        self.assertEqual(self.probe(), ('3.10.22', False))

    def test_fresh_environment_needs_bootstrap(self):
        with mock.patch.object(setup.subprocess, 'run') as run:
            self.assertIsNone(setup.existing_python(self.root / 'new', '3.10.22'))
        run.assert_not_called()

    def test_patch_fallback_requires_explicit_acceptance_or_existing_receipt(self):
        with self.assertRaisesRegex(SystemExit, 'incompatible Python'):
            self.probe(version='3.10.23')
        self.assertEqual(self.probe(version='3.10.23', allow=True), ('3.10.23', True))
        (self.venv / linux_runtime.RECEIPT).write_text(json.dumps(
            dict(python='3.10.23', python_patch_fallback=True)))
        self.assertEqual(self.probe(version='3.10.23'), ('3.10.23', True))
        with self.assertRaisesRegex(SystemExit, 'incompatible Python'):
            self.probe(version='3.11.1', allow=True)

    def test_foreign_architecture_wrong_prefix_and_non_venv_are_preserved(self):
        other = 'x86_64' if linux_runtime.normalize_machine() == 'aarch64' else 'aarch64'
        marker = self.venv / 'user-data'
        marker.write_text('preserve')
        for kwargs in ({'machine': other}, {'prefix': '/other'}, {'base': str(self.venv)}):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(SystemExit, 'preserved'):
                self.probe(**kwargs)
        self.assertEqual(marker.read_text(), 'preserve')

    def test_broken_python_is_not_replaced(self):
        with mock.patch.object(setup.subprocess, 'run', side_effect=FileNotFoundError()), \
                self.assertRaisesRegex(SystemExit, 'preserved'):
            setup.existing_python(self.venv, '3.10.22')


class StartupPreflight(TemporaryRoot):
    def test_healthy_start_does_not_install(self):
        with mock.patch.object(linux_startup, 'activate'), \
                mock.patch.object(linux_startup, 'check_gui'), \
                mock.patch.object(setup, 'repair_gui') as repair:
            linux_startup.prepare_gui(self.root)
            linux_startup.prepare_gui(self.root, repair=True)
        repair.assert_not_called()

    def test_gui_drift_reports_repair_command_without_modification(self):
        with mock.patch.object(linux_startup, 'activate'), \
                mock.patch.object(linux_startup, 'check_gui', side_effect=RuntimeError('Qt mismatch')), \
                mock.patch.object(setup, 'repair_gui') as repair, \
                self.assertRaisesRegex(RuntimeError, 'RGBMoCap.sh --repair'):
            linux_startup.prepare_gui(self.root)
        repair.assert_not_called()

    def test_base_failure_cannot_trigger_qt_repair(self):
        with mock.patch.object(linux_startup, 'activate', side_effect=RuntimeError('Torch mismatch')), \
                mock.patch.object(setup, 'repair_gui') as repair, \
                self.assertRaisesRegex(RuntimeError, '--profiles shared'):
            linux_startup.prepare_gui(self.root, repair=True)
        repair.assert_not_called()

    def test_requested_repair_is_checked_again_and_never_loops(self):
        for failure in (None, RuntimeError('still broken')):
            with self.subTest(failure=failure), mock.patch.object(linux_startup, 'activate'), \
                    mock.patch.object(linux_startup, 'check_gui', side_effect=[RuntimeError('old Qt'), failure]), \
                    mock.patch.object(setup, 'find_uv', return_value='uv'), \
                    mock.patch.object(setup, 'repair_gui') as repair, mock.patch('builtins.print'):
                if failure:
                    with self.assertRaisesRegex(RuntimeError, 'still broken'):
                        linux_startup.prepare_gui(self.root, repair=True)
                else:
                    linux_startup.prepare_gui(self.root, repair=True)
                repair.assert_called_once_with('uv')

    def test_painting_failure_and_timeout_block_startup(self):
        for outcome in (SimpleNamespace(returncode=1, stdout='bad binding', stderr=''),
                        subprocess.TimeoutExpired('Qt probe', 15)):
            with self.subTest(outcome=outcome), mock.patch.object(linux_startup, 'activate'), \
                    mock.patch.object(linux_startup, 'gui_requirements', return_value=[]), \
                    mock.patch.object(linux_startup.subprocess, 'run') as run:
                if isinstance(outcome, Exception):
                    run.side_effect = outcome
                else:
                    run.return_value = outcome
                with self.assertRaisesRegex(RuntimeError, 'Qt painting check'):
                    linux_startup.check_gui(self.root)

    def test_all_qt_versions_and_import_origins_are_checked(self):
        for version, origin in (('6.8.3', None), ('6.7.3', '/external/shiboken6/__init__.py')):
            with self.subTest(version=version), mock.patch.object(linux_startup, 'activate'), \
                    mock.patch.object(linux_startup, 'gui_requirements', return_value=['shiboken6==6.7.3']), \
                    mock.patch.object(linux_startup.metadata, 'version', return_value=version), \
                    mock.patch.object(linux_startup.importlib.util, 'find_spec', return_value=SimpleNamespace(origin=origin)), \
                    mock.patch.object(linux_startup.subprocess, 'run') as run:
                with self.assertRaises(RuntimeError):
                    linux_startup.check_gui(self.root)
                run.assert_not_called()


class SetupRepair(TemporaryRoot):
    def test_private_uv_wins_over_stale_path_tool_and_explicit_choice_wins_over_both(self):
        private = self.root / '.bootstrap/bin/uv'
        private.parent.mkdir(parents=True)
        private.write_text('private tool')
        with mock.patch.object(setup, 'ROOT', self.root), \
                mock.patch.object(setup.shutil, 'which', return_value='/old/uv') as search:
            self.assertEqual(setup.find_uv(), private)
            self.assertEqual(setup.find_uv('/explicit/uv'), Path('/explicit/uv'))
        search.assert_not_called()

    def test_install_and_gui_repair_refuse_running_application_before_mutation(self):
        with mock.patch.object(setup, 'ROOT', self.root), mock.patch.object(setup, 'prepared_lock') as prepare, \
                mock.patch.object(setup, 'find_uv') as uv, app_lock(self.root):
            for args in (['--profiles', 'shared'], ['--gui-only']):
                with self.subTest(args=args), self.assertRaisesRegex(RuntimeError, 'Close RGB MoCap'):
                    setup.main(args)
        prepare.assert_not_called()
        uv.assert_not_called()

    def test_gui_only_never_bootstraps_python_or_installs_numerical_packages(self):
        with mock.patch.object(setup, 'ROOT', self.root), \
                mock.patch.object(setup, 'existing_python', return_value=('3.10.22', False)), \
                mock.patch.object(setup, 'check_runtime') as check, \
                mock.patch.object(setup, 'gui_requirements', return_value=['PySide6==6.7.3']), \
                mock.patch.object(setup, 'run') as run, \
                mock.patch.object(setup, 'ensure_python', side_effect=AssertionError('No bootstrap')), \
                mock.patch('builtins.print'):
            setup.repair_gui('uv')
        self.assertEqual([c.args[1] for c in check.call_args_list], ['shared', 'gui'])
        install = list(map(str, run.call_args_list[0].args[0]))
        self.assertEqual(install[:3], ['uv', 'pip', 'install'])
        self.assertIn('--no-deps', install)
        self.assertIn('--only-binary', install)
        self.assertEqual(install[-1], 'PySide6==6.7.3')
        self.assertFalse(any('torch' in value for value in install))
        self.assertTrue(str(run.call_args_list[-1].args[0][-1]).endswith('check_qt_runtime.py'))

    def test_gui_install_cannot_run_if_shared_runtime_check_fails(self):
        with mock.patch.object(setup, 'ROOT', self.root), \
                mock.patch.object(setup, 'existing_python', return_value=('3.10.22', False)), \
                mock.patch.object(setup, 'check_runtime', side_effect=RuntimeError('bad base')), \
                mock.patch.object(setup, 'run') as run, self.assertRaisesRegex(RuntimeError, 'bad base'):
            setup.repair_gui('uv')
        run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
