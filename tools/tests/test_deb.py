"""Online deb packaging and bootstrap checks; no downloads or system changes."""
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
import build_deb

spec = importlib.util.spec_from_file_location('deb_launcher', ROOT / 'packaging/debian/launcher.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class Packaging(unittest.TestCase):
    def test_only_public_source_and_licenses(self):
        paths = {str(p) for p in build_deb.selected_sources()}
        for required in ('mocap/gui.py', 'mocap/video_batch_ui.py', 'mocap/runtime_registry.py',
                         'installer/linux-aarch64.json', 'tools/setup_ubuntu.py',
                         'packaging/debian/launcher.py', 'LICENSE', 'THIRD_PARTY_NOTICES.md'):
            self.assertIn(required, paths)
        self.assertFalse(any(Path(p).suffix in build_deb.FORBIDDEN for p in paths))
        self.assertFalse(any(Path(p).parts[0] in build_deb.PRIVATE for p in paths))

    def test_tracked_private_data_is_excluded_too(self):
        with patch.object(build_deb, 'git', return_value=b'models/private.txt\0projects/user.txt\0config/local.json\0.env\0'):
            paths = {str(p) for p in build_deb.selected_sources()}
        self.assertEqual(paths, set(build_deb.EXTRA))

    def test_payload_contains_exact_source_bytes_and_modes(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'payload.tar.xz'
            build_deb.payload(output, [Path('RGBMoCap.sh'), Path('mocap/runtime_registry.py')], 100)
            with tarfile.open(output) as archive:
                self.assertEqual(archive.getmember('RGBMoCap.sh').mode, 0o755)
                self.assertEqual(archive.extractfile('mocap/runtime_registry.py').read(), (ROOT / 'mocap/runtime_registry.py').read_bytes())


class Bootstrap(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.package = self.folder / 'package'
        self.package.mkdir()
        self.base = self.folder / 'user'
        payload = self.package / 'source.tar.xz'
        with tarfile.open(payload, 'w:xz') as archive:
            info = tarfile.TarInfo('RGBMoCap.sh')
            info.mode = 0o755
            info.size = 4
            archive.addfile(info, io.BytesIO(b'test'))
        self.metadata = dict(version='1.0.1-1', payload_sha256=launcher.digest(payload))
        (self.package / 'package.json').write_text(json.dumps(self.metadata))

    def test_source_is_fresh_and_reusable_without_overwriting(self):
        root = launcher.install_source(self.package, self.base, self.metadata)
        (root / 'models').mkdir()
        keep = root / 'models/user.txt'
        keep.write_text('keep')
        self.assertEqual(launcher.install_source(self.package, self.base, self.metadata), root)
        self.assertEqual(keep.read_text(), 'keep')
        self.assertTrue(os.access(root / 'RGBMoCap.sh', os.X_OK))

    def test_unowned_existing_folder_is_preserved(self):
        root = self.base / 'installations' / self.metadata['payload_sha256'][:20]
        root.mkdir(parents=True)
        keep = root / 'keep.txt'
        keep.write_text('keep')
        with self.assertRaises(RuntimeError):
            launcher.install_source(self.package, self.base, self.metadata)
        self.assertEqual(keep.read_text(), 'keep')

    def test_packaging_revision_reuses_identical_source(self):
        root = launcher.install_source(self.package, self.base, self.metadata)
        revised = dict(self.metadata, version='1.0.1-2')
        self.assertEqual(launcher.install_source(self.package, self.base, revised), root)

    def test_corrupt_archive_is_refused(self):
        (self.package / 'source.tar.xz').write_bytes(b'corrupt')
        with self.assertRaisesRegex(RuntimeError, 'checksum'):
            launcher.install_source(self.package, self.base, self.metadata)

    def test_path_traversal_and_links_are_refused(self):
        payload = self.package / 'source.tar.xz'
        for name, kind in (('../outside', tarfile.REGTYPE), ('link', tarfile.SYMTYPE)):
            with self.subTest(name=name):
                with tarfile.open(payload, 'w:xz') as archive:
                    info = tarfile.TarInfo(name)
                    info.type = kind
                    archive.addfile(info)
                metadata = dict(self.metadata, payload_sha256=launcher.digest(payload))
                with self.assertRaisesRegex(RuntimeError, 'Unsafe'):
                    launcher.install_source(self.package, self.base, metadata)
        self.assertFalse((self.folder / 'outside').exists())

    def test_setup_preserves_profile_isolation_and_architecture_installer(self):
        root = launcher.install_source(self.package, self.base, self.metadata)
        with patch.object(launcher, 'run') as run:
            launcher.setup(root, ['shared', 'flow', 'exmr'])
        command = list(map(str, run.call_args.args[0]))
        self.assertIn(str(root / 'tools/setup_ubuntu.py'), command)
        self.assertEqual(command[command.index('--profiles') + 1:command.index('--stage-assets')], ['shared', 'flow', 'exmr'])
        self.assertIn('--build-correction', command)
        self.assertTrue((root / '.deb-runtime-ready').is_file())

    def test_failed_setup_does_not_mark_ready(self):
        root = launcher.install_source(self.package, self.base, self.metadata)
        with patch.object(launcher, 'run', side_effect=RuntimeError('download failed')):
            with self.assertRaises(RuntimeError):
                launcher.setup(root, ['shared'])
        self.assertFalse((root / '.deb-runtime-ready').exists())

    def test_user_launch_and_doctor_use_stable_data_directory(self):
        with patch.object(launcher, 'PACKAGE', self.package), patch.object(launcher.os, 'geteuid', return_value=1000), \
                patch.object(launcher.platform, 'machine', return_value='aarch64'), \
                patch.dict(os.environ, {'XDG_DATA_HOME': str(self.base)}, clear=True), \
                patch.object(launcher, 'setup') as setup, patch.object(launcher.os, 'execve') as execute:
            launcher.main(['--doctor', '--gpu'])
        setup.assert_called_once()
        command, argv, env = execute.call_args.args
        self.assertEqual(command.name, 'rgbmocap-cli.sh')
        self.assertEqual(argv[1:], ['doctor', '--gpu'])
        self.assertEqual(env['RGB_MOCAP_DATA'], str(self.base / 'rgbmocap/data'))

    def test_root_launch_is_refused(self):
        with patch.object(launcher.os, 'geteuid', return_value=0):
            with self.assertRaisesRegex(RuntimeError, 'without sudo'):
                launcher.main([])


if __name__ == '__main__':
    unittest.main()
