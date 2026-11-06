"""Ubuntu portability checks: standard library only, CPU only, no user data or weights.

Run with:  python3 -m unittest discover -s tools/tests -v   (or pytest tools/tests)
Temporary roots keep every test away from models/, projects/ and real environments.
"""
import ast
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]

from mocap import linux_runtime, process_memory, resource_cache, runtime_registry  # noqa: E402
from mocap.retarget import runtime as blender_runtime  # noqa: E402
from mocap import video  # noqa: E402
import setup_ubuntu  # noqa: E402
import session_lock  # noqa: E402
import ubuntu_cli  # noqa: E402

BASE_SHA = 'ef05127a0dbddb3f3ac672a0722dddeae42ea651'
PROFILES_DIGEST = '6a21b413b26dff5f6f7325c49f0c19047b1d77a705a93942c769197431c2f7c5'
LOCK_GROUPS_DIGEST = 'ddb0fb566ce5476693eaacd5a686d61842d1a3ec203de88a6b2a1b915a549565'
FLOW_ADAPTER_DIGEST = '391623dd5cc17739bae4c4f6cc1250c115b51ceb613b6f10cfccbf595d069557'
# Hashed into receipts with code_hash: Linux edits must still resolve to the baseline identity.
ALIASED_IDENTITIES = {
    'mocap/worker.py': '883d9dde808c',
    'mocap/exmr_backend.py': '77c7162093f5',
    'mocap/flow_hamer.py': 'b82861ea8496',
}
# Plain-hashed or stat-checked in receipts, or part of the Windows start path: stay byte-identical.
UNCHANGED_FILES = {
    'mocap/correction.py': '7b25e1eb57ed810bdf995a918d7b7633e7e558c25f982ba273f7d64783105ec3',
    'mocap/flow_correction.py': '6fb33f433ece03ef19099a86d803a31a2b67ac23c70aa95a892906afc20e4d7a',
    'mocap/mixamo_data.py': 'a6d8e9a4dfd89d9b4f9103b23be2ad558e55e1831094954a6e2a03cb0f62940a',
    'mocap/handflow.py': 'f157fe40e4c65b4853425ea50ed641a3b1a91fde40225aae93e273f912b05568',
    'mocap/handflow_engine.py': '7d1061f7f56e6c47a7a9660979616fb751dd8da36f39f0f4ed72a3b81135e04d',
    'mocap/flow_hamer_entry.py': '587c42743b11046575ddf18209f481293bd4f4bad5c0197c5fa58faf0f774538',
    'mocap/compute_policy.py': '9c8ca149a0d93075353308e93208d8831476f94d9f089a5b0b2b57253b5f1c0c',
    'mocap/flow_placement.py': '4d2df2dc65e8680fc4add9d750cdcabb1d7d09e9d87167cb340de5909777de23',
    'mocap/flow_sam.py': 'ac4bfaa4ce2ca1caed51e2a693a9c3c39f45dc07241ed9c0e2df7a127f52200f',
    'worker_boot.py': 'db78b2b208bfb1920bdd2ff69ad929901ae3a114ed8cf5d5b2ae68e5b4690023',
    'launch_source.py': 'aeca1cbf79a3cf6c071bba5b2e74a611043d3e80aafcdb2db896ad9c779c3f05',
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def executable(path, text='#!/bin/sh\n'):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


class TempRoot(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name).resolve()


class RegistryRouting(TempRoot):
    def test_each_stage_uses_its_own_isolated_venv(self):
        expected = {'prepare': '.venv', 'track': '.venv', 'save_fbx': '.venv', 'flow_features': '.venv-flow', 'flowhmr': '.venv-flow',
                    'flow_batch': '.venv-flow', 'exmr': '.venv-exmr', 'handflow': '.venv-hand', 'refine': '.venv-htd'}
        for stage, folder in expected.items():
            python = executable(self.root / folder / 'bin/python')
            self.assertEqual(runtime_registry.command(self.root, stage, ('--project', 'P'), frozen=False),
                             [str(python), '-I', '-X', 'utf8', str(self.root / 'worker_boot.py'), stage, '--project', 'P'])

    def test_project_model_routes_post_processing(self):
        project = self.root / 'project'
        project.mkdir()
        (project / 'project.json').write_text(json.dumps({'pipeline': {'model': 'flowhmr'}}))
        python = executable(self.root / '.venv-flow/bin/python')
        self.assertEqual(runtime_registry.command(self.root, 'correct', ('--project', project), frozen=False)[0], str(python))

    def test_frozen_or_managed_flags_cannot_select_windows_paths(self):
        python = executable(self.root / '.venv/bin/python')
        with mock.patch.dict(os.environ, RGB_MOCAP_MANAGED='1'):
            self.assertEqual(runtime_registry.command(self.root, 'prepare', frozen=True)[0], str(python))

    def test_missing_environment_names_the_linux_setup(self):
        with self.assertRaisesRegex(RuntimeError, 'setup_ubuntu.py'):
            runtime_registry.command(self.root, 'flowhmr', frozen=False)

    def windows(self, **environ):
        return mock.patch.object(runtime_registry, 'os', SimpleNamespace(name='nt', environ=environ))

    def test_windows_managed_route_is_unchanged(self):
        python = executable(self.root / 'runtime/python.exe')
        (self.root / 'runtime_flow').mkdir()
        with self.windows(RGB_MOCAP_MANAGED='1'):
            self.assertEqual(runtime_registry.command(self.root, 'flowhmr', frozen=False),
                             [str(python), '-I', '-X', 'utf8', str(self.root / 'worker_boot.py'), 'flowhmr'])
            with self.assertRaisesRegex(RuntimeError, 'Run Setup'):
                runtime_registry.command(self.root, 'exmr', frozen=False)

    def test_windows_development_route_is_unchanged(self):
        python = executable(self.root / '.venv-flow/Scripts/python.exe')
        with self.windows():
            self.assertEqual(runtime_registry.command(self.root, 'flowhmr', frozen=False),
                             [str(python), '-X', 'utf8', '-m', 'mocap.worker_progress', 'flowhmr'])

    def test_receipt_visible_profiles_and_pins_are_unchanged(self):
        self.assertEqual(digest(json.dumps(runtime_registry.PROFILES, sort_keys=True).encode()), PROFILES_DIGEST)
        lock = json.loads((ROOT / 'installer/runtime-lock.json').read_bytes())
        self.assertEqual(digest(json.dumps(lock['groups'], sort_keys=True).encode()), LOCK_GROUPS_DIGEST)

    def test_lock_records_the_edited_registry(self):
        lock = json.loads((ROOT / 'installer/runtime-lock.json').read_bytes())
        self.assertEqual(lock['registry_sha256'], digest((ROOT / 'mocap/runtime_registry.py').read_bytes()))
        recorded = lock.pop('id')
        self.assertEqual(recorded, digest(json.dumps(lock, sort_keys=True).encode()))

    def test_worker_boot_reaches_linux_validation(self):
        other = 'exmr' if Path(sys.prefix).name != '.venv-exmr' else 'flow'
        done = subprocess.run([sys.executable, '-I', '-X', 'utf8', '-B', str(ROOT / 'worker_boot.py'), '--runtime-check', other],
                              capture_output=True, text=True)
        self.assertNotEqual(done.returncode, 0)
        self.assertIn('setup_ubuntu.py', done.stderr)
        self.assertNotIn('python.exe', done.stderr)


@unittest.skipUnless(sys.version_info[:2] in ((3, 10), (3, 12)), 'needs a Python minor used by a profile')
class VenvActivation(TempRoot):
    """Real venvs and real -I processes, with small stand-in packages instead of Torch."""
    SCRIPT = ('import json,sys;sys.path.insert(0,sys.argv[1]);from mocap.runtime_registry import activate;'
              'print(json.dumps(activate(sys.argv[2],sys.argv[3])))')

    def setUp(self):
        super().setUp()
        self.key = 'exmr' if sys.version_info[:2] == (3, 12) else 'shared'
        # The pins this host's activation really enforces, including any architecture variant.
        self.spec = linux_runtime.effective_profile(self.key)
        self.venv = self.root / self.spec['dev']
        python = getattr(sys, '_base_executable', sys.executable)
        subprocess.run([python, '-m', 'venv', '--without-pip', str(self.venv)], check=True)
        self.site = linux_runtime.site_packages(self.venv)
        for name, version in self.spec['packages'].items():
            self.package(name, version)
        self.current = '.'.join(map(str, sys.version_info[:3]))
        self.receipt(True)

    def package(self, name, version):
        folder = self.site / name
        folder.mkdir(exist_ok=True)
        (folder / '__init__.py').write_text('')
        info = self.site / f'{name}-{version}.dist-info'
        info.mkdir(exist_ok=True)
        (info / 'METADATA').write_text(f'Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n')

    def receipt(self, fallback):
        (self.venv / linux_runtime.RECEIPT).write_text(json.dumps(dict(python=self.current, python_patch_fallback=fallback)))

    def activate(self, key=None, env=None):
        return subprocess.run([str(self.venv / 'bin/python'), '-I', '-X', 'utf8', '-B', '-c', self.SCRIPT, str(ROOT), str(self.root), key or self.key],
                              capture_output=True, text=True, env=env)

    def test_valid_environment_reports_venv_imports(self):
        done = self.activate()
        self.assertEqual(done.returncode, 0, done.stderr)
        value = json.loads(done.stdout)
        self.assertEqual(value['environment'], str(self.venv))
        for origin in value['imports'].values():
            self.assertTrue(Path(origin).is_relative_to(self.site.resolve()), origin)

    def test_isolated_mode_ignores_a_hostile_pythonpath(self):
        hostile = self.root / 'hostile'
        (hostile / 'torch').mkdir(parents=True)
        (hostile / 'torch/__init__.py').write_text('raise SystemExit("hostile import")')
        done = self.activate(env={**os.environ, 'PYTHONPATH': str(hostile)})
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertNotIn('hostile', done.stdout)

    def test_another_profile_is_rejected(self):
        done = self.activate('flow' if self.key == 'exmr' else 'exmr')
        self.assertIn('Wrong interpreter', done.stderr)

    def test_wrong_python_patch_needs_a_recorded_fallback(self):
        if self.current == self.spec['python']:
            self.skipTest('This Python is exactly the profile pin.')
        self.receipt(False)
        self.assertIn('Unexpected Python', self.activate().stderr)

    def test_wrong_package_version_is_rejected(self):
        shutil.rmtree(self.site / f"torch-{self.spec['packages']['torch']}.dist-info")
        self.package('torch', '0.0.1')
        self.assertIn('Incompatible torch', self.activate().stderr)

    def test_missing_package_is_rejected(self):
        shutil.rmtree(self.site / f"timm-{self.spec['packages']['timm']}.dist-info")
        self.assertIn('timm is missing', self.activate().stderr)

    def test_windows_overlay_chain_is_rejected(self):
        (self.site / 'shared_runtime.pth').write_text('import sys\n')
        self.assertIn('overlay', self.activate().stderr)


class PeakMemory(unittest.TestCase):
    def test_linux_value_is_bytes(self):
        block = bytearray(64 * 1024**2)
        block[::4096] = b'\x01' * len(block[::4096])
        status = Path('/proc/self/status').read_text()
        peak_kib = int(re.search(r'VmHWM:\s+(\d+) kB', status).group(1))
        value = process_memory.peak_rss_bytes()
        self.assertGreaterEqual(value, 64 * 1024**2)
        self.assertLess(abs(value - peak_kib * 1024), 32 * 1024**2)

    def test_reporting_sites_do_not_use_windows_only_fields(self):
        for name in ('mocap/worker.py', 'mocap/flow_backend.py', 'mocap/flow_hamer.py', 'mocap/playback_check.py'):
            self.assertNotIn('.peak_wset', (ROOT / name).read_text(encoding='utf-8'), name)


class VideoTools(TempRoot):
    def test_bundled_tool_has_no_exe_suffix_on_linux(self):
        tool = executable(self.root / 'tools/bin/ffmpeg')
        with mock.patch.object(video, 'ROOT', self.root):
            self.assertEqual(video.executable('ffmpeg'), str(tool))

    def test_path_fallback_and_missing_tool(self):
        tool = executable(self.root / 'path/ffprobe')
        with mock.patch.object(video, 'ROOT', self.root), mock.patch.dict(os.environ, PATH=str(tool.parent)):
            self.assertEqual(video.executable('ffprobe'), str(tool))
            with self.assertRaisesRegex(RuntimeError, 'Video tools are missing'):
                video.executable('ffmpeg')

    def test_no_windows_process_flag(self):
        self.assertEqual(video.NO_WINDOW, 0)

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'needs ffmpeg')
    def test_real_probe_and_command_runner(self):
        clip = self.root / 'clip.mp4'
        command = [video.executable('ffmpeg'), '-hide_banner', '-loglevel', 'error', '-f', 'lavfi', '-i', 'testsrc=size=64x64:rate=30:duration=1',
                   '-pix_fmt', 'yuv420p', '-y', str(clip)]
        video.run_video_command(command)
        self.assertEqual(video.probe(clip)['frames'], 30)
        clip.unlink()
        video.run_video_command(command, cancelled=lambda: False)
        self.assertTrue(clip.is_file())


class BlenderDiscovery(TempRoot):
    def discover(self, **environ):
        path = environ.pop('PATH', str(self.root / 'empty'))
        with mock.patch.object(blender_runtime, 'ROOT', self.root), mock.patch.dict(os.environ, {'PATH': path, **environ}):
            if 'RGBMOCAP_BLENDER' not in environ:
                os.environ.pop('RGBMOCAP_BLENDER', None)
            return blender_runtime.blender()

    def test_managed_folder_uses_linux_name(self):
        tool = executable(self.root / 'runtime_character/blender')
        self.assertEqual(self.discover(), tool)

    def test_explicit_file_or_folder_wins(self):
        executable(self.root / 'runtime_character/blender')
        explicit = executable(self.root / 'custom/blender')
        self.assertEqual(self.discover(RGBMOCAP_BLENDER=str(explicit)), explicit)
        self.assertEqual(self.discover(RGBMOCAP_BLENDER=str(explicit.parent)), explicit)

    def test_path_fallback(self):
        tool = executable(self.root / 'path/blender')
        self.assertEqual(self.discover(PATH=str(tool.parent)), tool)

    def test_non_executable_file_and_missing_runtime(self):
        path = self.root / 'runtime_character/blender'
        path.parent.mkdir()
        path.write_text('')
        path.chmod(0o644)
        with self.assertRaisesRegex(RuntimeError, r'LTS blender\.$'):
            self.discover()


class CorrectionBackend(TempRoot):
    def test_abi_specific_module_is_preferred(self):
        folder = self.root / linux_runtime.CORRECTION_FOLDER
        folder.mkdir(parents=True)
        (folder / '_motion_correction.so').write_text('')
        specific = folder / '_motion_correction.cpython-310-x86_64-linux-gnu.so'
        specific.write_text('')
        self.assertEqual(linux_runtime.correction_path(self.root, ['.cpython-310-x86_64-linux-gnu.so', '.so']), specific)
        self.assertEqual(linux_runtime.correction_path(self.root, ['.so']), folder / '_motion_correction.so')
        self.assertIsNone(linux_runtime.correction_path(self.root, ['.cpython-312-x86_64-linux-gnu.so']))

    def test_missing_module_has_a_linux_message(self):
        with mock.patch('mocap.paths.ROOT', self.root), self.assertRaisesRegex(RuntimeError, 'build_motion_correction'):
            linux_runtime.correction_backend()

    def test_post_import_patch_is_lazy_and_keeps_module_contents(self):
        name = 'rgbmocap_patch_target'
        (self.root / f'{name}.py').write_text('import builtins\nbuiltins._rgbmocap_loaded=True\nvalue=1\ndef backend():return "windows"\n')
        finder = linux_runtime.PostImportPatches({name: lambda module: setattr(module, 'backend', lambda: 'linux')})
        sys.path.insert(0, str(self.root))
        sys.meta_path.insert(0, finder)
        self.addCleanup(sys.path.remove, str(self.root))
        self.addCleanup(sys.meta_path.remove, finder)
        self.addCleanup(sys.modules.pop, name, None)
        import builtins
        self.assertFalse(hasattr(builtins, '_rgbmocap_loaded'))
        module = importlib.import_module(name)
        self.addCleanup(delattr, builtins, '_rgbmocap_loaded')
        self.assertEqual((module.value, module.backend()), (1, 'linux'))
        self.assertTrue(module.__loader__.get_source(name).startswith('import builtins'))

    def test_install_hook_is_idempotent(self):
        before = list(sys.meta_path)
        self.addCleanup(sys.meta_path.__setitem__, slice(None), before)
        linux_runtime.install_import_hooks()
        linux_runtime.install_import_hooks()
        self.assertEqual(sum(isinstance(item, linux_runtime.PostImportPatches) for item in sys.meta_path), 1)

    def test_opengl_guard_only_replaces_the_windows_platform(self):
        finder = linux_runtime.PostImportPatches({})
        with mock.patch.dict(os.environ, PYOPENGL_PLATFORM='win32'):
            self.assertIsNone(finder.find_spec('OpenGL'))
            self.assertEqual(os.environ['PYOPENGL_PLATFORM'], 'egl')
        with mock.patch.dict(os.environ, PYOPENGL_PLATFORM='osmesa'):
            finder.find_spec('OpenGL')
            self.assertEqual(os.environ['PYOPENGL_PLATFORM'], 'osmesa')

    def test_worker_identity_hashes_the_linux_artifact(self):
        source = (ROOT / 'mocap/worker.py').read_text(encoding='utf-8')
        self.assertIn("glob('*.pyd')", source)
        self.assertIn("glob('*.so')", source)


class NativeBuildTool(TempRoot):
    def test_linux_build_targets_the_requested_venv(self):
        log = self.root / 'cmake.log'
        cmake = executable(self.root / 'bin/cmake', '#!/bin/sh\n'
                           f'printf "%s\\n" "$@" >> "{log}"\n'
                           'if [ "$1" = "--build" ]; then touch "$2/_motion_correction.cpython-310-x86_64-linux-gnu.so"; fi\n')
        build, output = self.root / 'build', self.root / 'output'
        python = '/virtual/.venv/bin/python'
        subprocess.run([sys.executable, '-I', '-B', str(ROOT / 'tools/build_motion_correction.py'), '--build-dir', str(build), '--output-dir', str(output),
                        '--deps', str(self.root / 'no-deps'), '--python', python],
                       check=True, env={**os.environ, 'PATH': f"{cmake.parent}{os.pathsep}{os.environ['PATH']}"}, capture_output=True)
        arguments = log.read_text().splitlines()
        self.assertIn('-DPython3_EXECUTABLE=' + python, arguments)
        self.assertIn('-DCMAKE_BUILD_TYPE=Release', arguments)
        self.assertNotIn('Visual Studio 17 2022', arguments)
        self.assertTrue((output / '_motion_correction.cpython-310-x86_64-linux-gnu.so').is_file())


class CacheIdentity(unittest.TestCase):
    def test_edited_receipt_sources_keep_baseline_identities(self):
        for name, expected in ALIASED_IDENTITIES.items():
            self.assertTrue(resource_cache.code_hash(ROOT / name).startswith(expected),
                            f'{name}: a further edit needs a reviewed PLATFORM_ONLY alias update; see docs/UBUNTU.md')

    def test_plain_hashed_sources_are_byte_identical(self):
        for name, expected in UNCHANGED_FILES.items():
            self.assertEqual(digest((ROOT / name).read_bytes()), expected, name)

    def test_flow_adapter_source_identity_is_unchanged(self):
        source = (ROOT / 'mocap/flow_backend.py').read_text(encoding='utf-8')
        tree = ast.parse(source)
        parts = ''
        for name in ('camera', 'camera_interval', 'features', '_features', 'infer', 'validate_motion', 'activate', 'load_tensor', 'save_tensor'):
            node = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == name)
            parts += ast.get_source_segment(source, node)
        self.assertEqual(digest(parts.encode()), FLOW_ADAPTER_DIGEST)

    def test_exmr_keeps_windows_values_and_respects_a_linux_choice(self):
        source = (ROOT / 'mocap/exmr_backend.py').read_text(encoding='utf-8')
        self.assertIn("PYOPENGL_PLATFORM='win32' if os.name=='nt' else os.environ.get('PYOPENGL_PLATFORM','egl')", source)
        self.assertNotIn('0x08000000', source)


class SetupPlan(TempRoot):
    @classmethod
    def setUpClass(cls):
        cls.lock = json.loads((ROOT / 'installer/runtime-lock.json').read_bytes())

    def plan(self, profile, machine='x86_64'):
        return setup_ubuntu.profile_plan(ROOT, self.lock, profile, machine)

    @staticmethod
    def pins(plan):
        return {setup_ubuntu.normalize(pin.split('==')[0]): pin.split('==')[1] for pin in plan['index'] + plan['cuda']}

    def test_readable_pins_match_the_lock(self):
        for group in self.lock['groups']:
            if group['key'] == 'gui':
                continue
            locked = {setup_ubuntu.normalize(p['name']): p['version'] for p in group['packages']}
            readable = setup_ubuntu.read_pins(ROOT / f"requirements/runtime-{group['key']}.txt")
            self.assertEqual({key: version for key, (_, version) in readable.items()}, locked, group['key'])

    def test_overlay_pins_win_and_profiles_stay_independent(self):
        flow = self.pins(self.plan('flow'))
        self.assertEqual((flow['numpy'], flow['scipy'], flow['timm'], flow['torch']), ('1.26.4', '1.15.3', '1.0.30', '2.7.1+cu128'))
        self.assertNotIn('win32-setctime', flow)
        self.assertNotIn('pyside6', flow)
        shared = self.pins(self.plan('shared'))
        self.assertEqual((shared['numpy'], shared['pyside6'], shared['shiboken6']), ('1.23.5', '6.8.3', '6.8.3'))
        exmr = self.pins(self.plan('exmr'))
        self.assertEqual((self.plan('exmr')['python'], exmr['torch'], exmr['numpy']), ('3.12.10', '2.8.0+cu128', '2.3.5'))
        self.assertEqual((self.plan('flow')['folder'], self.plan('exmr')['folder'], self.plan('shared')['folder']), ('.venv-flow', '.venv-exmr', '.venv'))

    def test_every_activation_pin_is_planned(self):
        for profile in setup_ubuntu.FOLDERS:
            pins = self.pins(self.plan(profile))
            for name, version in runtime_registry.PROFILES[profile]['packages'].items():
                self.assertEqual(pins.get(name), version, f'{profile}:{name}')

    def test_iopath_has_its_linux_locking_dependency(self):
        for profile in setup_ubuntu.FOLDERS:
            pins = self.pins(self.plan(profile))
            if 'iopath' in pins:
                self.assertEqual(pins.get('portalocker'), '4.4.1', profile)

    def test_only_cuda_builds_use_the_cuda_index(self):
        plan = self.plan('flow')
        self.assertEqual(plan['cuda'], ['torch==2.7.1+cu128', 'torchvision==0.22.1+cu128'])
        self.assertFalse([pin for pin in plan['index'] if '+cu' in pin])

    def test_bundled_wheels_are_reconstructed_from_source(self):
        lock = setup_ubuntu.prepared_lock(self.root)
        plan = setup_ubuntu.profile_plan(ROOT, lock, 'shared', 'x86_64')
        self.assertEqual(plan['wheels'], ['wheels/antlr4_python3_runtime-4.9.3-py3-none-any.whl', 'wheels/chumpy-0.70-py3-none-any.whl'])
        import zipfile
        with zipfile.ZipFile(self.root / plan['wheels'][1]) as archive:
            names = archive.namelist()
        self.assertIn('chumpy/__init__.py', names)
        self.assertTrue([name for name in names if name.endswith('.dist-info/METADATA')])
        self.assertFalse([pin for pin in plan['index'] if pin.startswith(('chumpy', 'antlr4'))])

    def test_linux_torch_requirements_are_added_exactly(self):
        metadata = '\n'.join((
            'Name: torch', 'Requires-Dist: numpy; extra == "x"', 'Requires-Dist: filelock', 'Requires-Dist: sympy>=1.13.3',
            'Requires-Dist: nvidia-cuda-runtime-cu12==12.8.90; platform_system == "Linux" and platform_machine == "x86_64"',
            'Requires-Dist: triton==3.3.1; platform_system == "Linux"', 'Requires-Dist: setuptools; platform_system == "Linux"',
            'Requires-Dist: intel-only==1; platform_system == "Windows"'))
        self.assertEqual(setup_ubuntu.linux_dependencies(metadata, {'setuptools', 'sympy', 'filelock'}),
                         ['nvidia-cuda-runtime-cu12==12.8.90', 'triton==3.3.1'])

    def test_python_patch_fallback_is_explicit(self):
        with mock.patch.object(setup_ubuntu, 'run', return_value=SimpleNamespace(returncode=1)):
            with self.assertRaisesRegex(SystemExit, 'python-patch-fallback'):
                setup_ubuntu.ensure_python('uv', '3.10.22', False)
            self.assertEqual(setup_ubuntu.ensure_python('uv', '3.10.22', True), ('3.10', True))
        with mock.patch.object(setup_ubuntu, 'run', return_value=SimpleNamespace(returncode=0)):
            self.assertEqual(setup_ubuntu.ensure_python('uv', '3.10.22', False), ('3.10.22', False))

    def test_plan_leaves_the_checkout_unchanged(self):
        before = sorted(path.name for path in (ROOT / 'build').glob('*')) if (ROOT / 'build').is_dir() else None
        with mock.patch('builtins.print') as output:
            self.assertEqual(setup_ubuntu.main(['--plan', '--profiles', 'flow']), 0)
        self.assertEqual(json.loads(output.call_args.args[0])[0]['folder'], '.venv-flow')
        after = sorted(path.name for path in (ROOT / 'build').glob('*')) if (ROOT / 'build').is_dir() else None
        self.assertEqual(before, after)

    def test_uv_state_stays_private(self):
        env = setup_ubuntu.uv_environment()
        self.assertTrue(env['UV_PYTHON_INSTALL_DIR'].startswith(str(ROOT)))
        self.assertTrue(env['UV_CACHE_DIR'].startswith(str(ROOT)))
        self.assertNotIn('VIRTUAL_ENV', env)

    def test_install_sequence_is_no_deps_and_ends_with_validation(self):
        plan = dict(profile='shared', folder='.venv', python='3.10.22', cuda=['torch==2.7.1+cu128'], index=['numpy==1.23.5'], wheels=['wheels/chumpy-0.70-py3-none-any.whl'])
        site = self.root / '.venv/lib/python3.10/site-packages'
        (site / 'torch-2.7.1+cu128.dist-info').mkdir(parents=True)
        (site / 'torch-2.7.1+cu128.dist-info/METADATA').write_text('Requires-Dist: triton==3.3.1; platform_system == "Linux"\n')
        executable(self.root / '.venv/bin/python')
        calls = []
        done = SimpleNamespace(returncode=0, stdout='3.10.22\n')
        with mock.patch.object(setup_ubuntu, 'ROOT', self.root), mock.patch.object(setup_ubuntu, 'run', side_effect=lambda command, check=True: calls.append(list(map(str, command))) or done), \
                mock.patch.object(setup_ubuntu.subprocess, 'run', return_value=done):
            setup_ubuntu.install_profile('uv', plan, self.root / 'wheels-root', self.root / 'work', False)
        python = str(self.root / '.venv/bin/python')
        self.assertEqual(calls[0], ['uv', 'python', 'install', '3.10.22', '--no-bin'])
        installs = [call for call in calls if call[:3] == ['uv', 'pip', 'install']]
        self.assertEqual(len(installs), 4)
        self.assertTrue(all('--no-deps' in call and python in call for call in installs))
        self.assertIn(setup_ubuntu.CUDA_INDEX, installs[0])
        self.assertIn('-r', installs[1])
        self.assertIn(str(self.root / 'wheels-root/wheels/chumpy-0.70-py3-none-any.whl'), installs[2])
        self.assertEqual(installs[3][-1], 'triton==3.3.1')
        self.assertEqual([call[-1] for call in calls if '--runtime-check' in call], ['gui', 'shared'])
        receipt = json.loads((self.root / '.venv' / linux_runtime.RECEIPT).read_text())
        self.assertEqual((receipt['python'], receipt['python_patch_fallback'], receipt['linux_extras']), ('3.10.22', False, ['triton==3.3.1']))


class RuntimeEnvironment(TempRoot):
    def test_optional_local_tools_are_added_once(self):
        (self.root / 'tools/bin').mkdir(parents=True)
        (self.root / '.runtime/lib').mkdir(parents=True)
        env = linux_runtime.runtime_environment(self.root, {'PATH': '/usr/bin'})
        self.assertEqual(env['PATH'], f"{self.root / 'tools/bin'}{os.pathsep}/usr/bin")
        self.assertEqual(env['LD_LIBRARY_PATH'], str(self.root / '.runtime/lib'))
        again = linux_runtime.runtime_environment(self.root, env)
        self.assertEqual((again['PATH'], again['LD_LIBRARY_PATH']), (env['PATH'], env['LD_LIBRARY_PATH']))

    def test_absent_folders_change_nothing_and_data_stays_private(self):
        env = linux_runtime.runtime_environment(self.root, {'PATH': '/usr/bin'})
        self.assertEqual(env['PATH'], '/usr/bin')
        self.assertNotIn('LD_LIBRARY_PATH', env)
        self.assertEqual((env['RGB_MOCAP_DATA'], env['HF_HOME']), (str(self.root), str(self.root / 'cache/huggingface')))
        data = self.root / 'data'
        self.assertEqual(linux_runtime.runtime_environment(self.root, {'RGB_MOCAP_DATA': str(data)})['TORCH_HOME'], str(data / 'cache/torch'))


class SessionLock(TempRoot):
    def test_second_session_is_refused_until_release(self):
        with session_lock.app_lock(self.root):
            with self.assertRaisesRegex(RuntimeError, 'Close RGB MoCap'):
                with session_lock.app_lock(self.root):
                    self.fail('The second lock must not be granted.')
        with session_lock.app_lock(self.root):
            pass


class Launchers(TempRoot):
    def copy(self, name):
        target = self.root / name
        shutil.copy2(ROOT / name, target)
        return target

    def test_shell_syntax_and_executable_bit(self):
        for name in ('RGBMoCap.sh', 'rgbmocap.sh'):
            subprocess.run(['bash', '-n', str(ROOT / name)], check=True)
            self.assertTrue(os.access(ROOT / name, os.X_OK), name)

    def test_gui_launcher_requires_setup(self):
        done = subprocess.run([str(self.copy('RGBMoCap.sh'))], capture_output=True, text=True)
        self.assertEqual(done.returncode, 1)
        self.assertIn('setup_ubuntu.py', done.stderr)

    def test_gui_launcher_environment_and_interpreter_flags(self):
        launcher = self.copy('RGBMoCap.sh')
        (self.root / 'tools/bin').mkdir(parents=True)
        (self.root / '.runtime/lib').mkdir(parents=True)
        executable(self.root / '.venv/bin/python', '#!/bin/sh\nprintf "%s\\n" "$PATH" "$LD_LIBRARY_PATH" "$@"\n')
        environ = {key: value for key, value in os.environ.items() if key != 'LD_LIBRARY_PATH'}
        lines = subprocess.run([str(launcher), '--video', 'a b.mp4'], capture_output=True, text=True, check=True, env=environ).stdout.splitlines()
        self.assertTrue(lines[0].startswith(str(self.root / 'tools/bin') + ':'))
        self.assertEqual(lines[1], str(self.root / '.runtime/lib'))
        self.assertEqual(lines[2:], ['-I', '-X', 'utf8', '-u', '-B', str(self.root / 'launch_linux.py'), '--video', 'a b.mp4'])

    def test_cli_launcher_runs_before_setup(self):
        done = subprocess.run([str(ROOT / 'rgbmocap.sh'), 'stages', '--json'], capture_output=True, text=True, check=True)
        self.assertIn('flow_features', json.loads(done.stdout)['stages'])


class CliRouting(TempRoot):
    def flow_args(self, *extra):
        clip = self.root / 'clip.mp4'
        clip.write_bytes(b'')
        return ubuntu_cli.parser().parse_args(['flow', '--video', str(clip), *extra])

    def fake_runner(self, failure=None):
        calls = []
        project = str(self.root / 'project')

        def runner(stage, arguments, env, on_event):
            calls.append(dict(stage=stage, arguments=list(arguments), env=dict(env)))
            events = [dict(type='prepared', project=project)] if stage == 'prepare' else []
            return (3 if stage == failure else 0), events
        return runner, calls, project

    def test_known_stages_cover_worker_choices(self):
        source = (ROOT / 'mocap/worker.py').read_text(encoding='utf-8')
        choices = ast.literal_eval(re.search(r'choices=(\["diagnose".*?\])\)', source).group(1))
        self.assertFalse(set(choices) - set(ubuntu_cli.KNOWN_STAGES))
        progress = (ROOT / 'mocap/worker_progress.py').read_text(encoding='utf-8')
        for stage in ('save_project', 'character', 'copy_bvh', 'camera_reference', 'flow_batch'):
            self.assertIn(f"'{stage}'", progress)

    def test_flow_runs_the_gui_order_with_joined_features(self):
        runner, calls, project = self.fake_runner()
        with mock.patch.object(ubuntu_cli, 'missing_models', return_value=[]), mock.patch('builtins.print'):
            self.assertEqual(ubuntu_cli.command_flow(self.flow_args(), runner), 0)
        self.assertEqual([call['stage'] for call in calls], ['prepare', 'track', 'flow_camera', 'flow_batch', 'flow_export'])
        self.assertEqual(calls[1]['arguments'], ['--project', project])
        self.assertIn('--motion-tool', calls[0]['arguments'])
        self.assertEqual(calls[0]['arguments'][calls[0]['arguments'].index('--result') + 1], 'initial')
        self.assertEqual(calls[-1]['arguments'], ['--project', project, '--result', 'initial'])
        self.assertEqual(json.loads(calls[3]['arguments'][calls[3]['arguments'].index('--batch-items') + 1]),
                         [dict(project=project, result='auto', force=False)])
        self.assertEqual(calls[0]['env']['RGB_MOCAP_FLOW_HAMER'], '0')

    def test_hamer_is_opt_in_and_reaches_every_stage(self):
        runner, calls, _ = self.fake_runner()
        with mock.patch.object(ubuntu_cli, 'missing_models', return_value=[]), mock.patch('builtins.print'):
            ubuntu_cli.command_flow(self.flow_args('--hamer', '--no-gpu-crop', '--camera-interval', '-1', '--compute-mode', 'low_memory', '--resource-safety', 'off'), runner)
        for call in calls:
            env = call['env']
            self.assertEqual((env['RGB_MOCAP_FLOW_HAMER'], env['RGB_MOCAP_FLOW_GPU_CROP'], env['RGB_MOCAP_FLOW_INTERVAL'],
                              env['RGB_MOCAP_COMPUTE_MODE'], env['RGB_MOCAP_RESOURCE_SAFETY']), ('1', '0', '-1', 'low_memory', '0'))

    def test_correction_selects_the_corrected_result(self):
        runner, calls, project = self.fake_runner()
        with mock.patch.object(ubuntu_cli, 'missing_models', return_value=[]), mock.patch('builtins.print'):
            ubuntu_cli.command_flow(self.flow_args('--correct', '--no-foot-lock', '--floor', '0.1'), runner)
        self.assertEqual([call['stage'] for call in calls], ['prepare', 'track', 'flow_camera', 'flow_batch', 'correct', 'flow_export'])
        prepare = calls[0]['arguments']
        self.assertEqual(prepare[prepare.index('--result') + 1], 'corrected_initial')
        self.assertEqual((prepare[prepare.index('--foot-lock') + 1], prepare[prepare.index('--floor') + 1]), ('0', '0.1'))
        self.assertEqual(calls[-2]['arguments'], ['--project', project, '--result', 'corrected_initial'])

    def test_force_recomputes_without_disturbing_tracking(self):
        runner, calls, _ = self.fake_runner()
        with mock.patch.object(ubuntu_cli, 'missing_models', return_value=[]), mock.patch('builtins.print'):
            ubuntu_cli.command_flow(self.flow_args('--force'), runner)
        self.assertNotIn('--force', calls[1]['arguments'])
        self.assertIn('--force', calls[2]['arguments'])
        self.assertTrue(json.loads(calls[3]['arguments'][calls[3]['arguments'].index('--batch-items') + 1])[0]['force'])

    def test_failure_stops_the_pipeline(self):
        runner, calls, _ = self.fake_runner(failure='flow_camera')
        with mock.patch.object(ubuntu_cli, 'missing_models', return_value=[]), mock.patch('builtins.print'):
            self.assertEqual(ubuntu_cli.command_flow(self.flow_args(), runner), 3)
        self.assertEqual(calls[-1]['stage'], 'flow_camera')

    def test_missing_models_prevent_work_but_not_dry_run(self):
        runner, calls, _ = self.fake_runner()
        with mock.patch.object(ubuntu_cli, 'missing_models', return_value=['Motion / FlowHMR latest']), mock.patch('builtins.print'):
            self.assertEqual(ubuntu_cli.command_flow(self.flow_args(), runner), 2)
            self.assertEqual(ubuntu_cli.command_flow(self.flow_args('--dry-run'), runner), 0)
        self.assertEqual(calls, [])

    def test_missing_video_is_reported(self):
        args = ubuntu_cli.parser().parse_args(['flow', '--video', str(self.root / 'absent.mp4')])
        with mock.patch('builtins.print'):
            self.assertEqual(ubuntu_cli.command_flow(args), 2)

    def test_stage_escape_hatch_passes_native_arguments_verbatim(self):
        seen = {}

        def runner(stage, arguments, env, on_event):
            seen.update(stage=stage, arguments=list(arguments))
            return 7, []
        args = ubuntu_cli.parser().parse_args(['stage', 'correct', '--project', 'P', '--result', 'corrected_initial', '--force'])
        self.assertEqual(ubuntu_cli.command_stage(args, runner), 7)
        self.assertEqual(seen, dict(stage='correct', arguments=['--project', 'P', '--result', 'corrected_initial', '--force']))

    def test_export_builds_a_worker_request(self):
        project = self.root / 'project'
        project.mkdir()
        (project / 'initial_mixamo.json').write_text(json.dumps(dict(bvh_file='initial.bvh')))
        seen = {}

        def runner(stage, arguments, env, on_event):
            request = Path(arguments[arguments.index('--output') + 1])
            seen.update(stage=stage, rows=json.loads(request.read_text()), arguments=list(arguments))
            request.unlink()
            return 0, []
        args = ubuntu_cli.parser().parse_args(['export', '--project', str(project), '--output', str(self.root / 'out.fbx')])
        with mock.patch('mocap.paths.HOME', self.root), mock.patch('builtins.print'):
            self.assertEqual(ubuntu_cli.command_export(args, runner, plan=lambda *_: dict(model='flowhmr', source=str(project / 'initial.bvh'))), 0)
        self.assertEqual(seen['stage'], 'save_fbx')
        self.assertEqual(seen['rows'], [dict(project=str(project), kind='initial', target=str(self.root / 'out.fbx'), source=str(project / 'initial.bvh'))])
        self.assertEqual(seen['arguments'][:2], ['--project', str(project)])


class WorkerProcess(TempRoot):
    def run_script(self, script, on_event):
        command = [sys.executable, '-I', '-c', script]
        with mock.patch('mocap.paths.worker_command', return_value=command):
            return ubuntu_cli.run_worker('flowhmr', [], dict(os.environ), on_event,
                                        guard_factory=lambda *a, **k: SimpleNamespace(check=lambda: None, output=lambda: None))

    def test_events_are_parsed_and_the_exit_code_is_kept(self):
        seen = []
        script = ('print("@@MOCAP@@{\\"type\\":\\"progress\\",\\"current\\":1,\\"total\\":2,\\"message\\":\\"Step\\"}");'
                  'print("@@MOCAP@@not json");print("plain");raise SystemExit(3)')
        with mock.patch('builtins.print') as output:
            code, events = self.run_script(script, lambda stage, event: seen.append((stage, event)))
        self.assertEqual(code, 3)
        self.assertEqual(events, [dict(type='progress', current=1, total=2, message='Step')])
        self.assertEqual(seen[0][0], 'flowhmr')
        self.assertTrue(any(call.args and call.args[0] == 'plain\n' for call in output.call_args_list))

    def test_printer_reports_progress_once_and_hides_batch_inner_completion(self):
        import io
        stream = io.StringIO()
        printer = ubuntu_cli.EventPrinter(stream=stream)
        printer('s', dict(type='progress', current=1, total=4, message='Step'))
        printer('s', dict(type='progress', current=1, total=4, message='Step'))
        printer('s', dict(type='complete', batch_inner=True, elapsed=1))
        printer('s', dict(type='complete', elapsed=2.0, peak_ram_gib=1.5))
        self.assertEqual(stream.getvalue().splitlines(), ['[s] Step 25%', '[s] complete: 2.0 s, peak RAM 1.50 GiB'])

    def test_interruption_stops_the_whole_process_group(self):
        marker = self.root / 'pid'
        child_script = 'import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);print("ready",flush=True);time.sleep(60)'
        script = (f"import subprocess,sys,time;child=subprocess.Popen([sys.executable,'-c',{child_script!r}],stdout=subprocess.PIPE);child.stdout.readline();open({str(marker)!r},'w').write(str(child.pid));"
                  "print('@@MOCAP@@{\"type\":\"info\",\"message\":\"go\"}',flush=True);time.sleep(60)")

        def interrupt(stage, event):
            raise KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            self.run_script(script, interrupt)
        pid = int(marker.read_text())
        import time
        for _ in range(100):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return
            state = Path(f'/proc/{pid}/stat')
            if state.exists() and state.read_text().split(')')[1].split()[0] == 'Z':
                return
            time.sleep(.05)
        os.kill(pid, 9)
        self.fail('The child survived cancellation.')

    def test_supervision_stops_silent_workers_in_both_safety_modes(self):
        for safety, reason in [('1', 'RAM headroom'), ('0', 'Stalled worker')]:
            with self.subTest(safety=safety):
                flags = []
                def factory(pid, resource_safety):
                    flags.append(resource_safety)
                    return SimpleNamespace(check=lambda: reason, output=lambda: None)
                command = [sys.executable, '-I', '-c', 'import time;time.sleep(60)']
                with mock.patch('mocap.paths.worker_command', return_value=command):
                    with self.assertRaisesRegex(RuntimeError, reason):
                        ubuntu_cli.run_worker('flowhmr', [], {'RGB_MOCAP_RESOURCE_SAFETY': safety},
                                              lambda *a: None, guard_factory=factory)
                self.assertEqual(flags, [safety != '0'])


class Doctor(TempRoot):
    def test_runs_before_setup_without_reporting_false_success(self):
        rows = ubuntu_cli.run_checks(ubuntu_cli.job_environment())
        status = {row['name']: row['status'] for row in rows}
        self.assertEqual(status['platform'], 'ok')
        self.assertIn('FlowHMR models', status)
        if not (ROOT / '.venv/bin/python').is_file():
            self.assertEqual(status['runtime flow'], 'fail')
            self.assertEqual(status['runtime shared'], 'fail')
            # The CLI stays usable if a variant system has no GUI wheels.
            self.assertEqual(status['runtime gui'], 'warn' if linux_runtime.linux_variant() else 'fail')

    def test_gpu_is_not_touched_unless_requested(self):
        with mock.patch.object(ubuntu_cli, 'capture', return_value=(0, 'GPU 0')) as run, mock.patch.object(ubuntu_cli.shutil, 'which', return_value='/usr/bin/nvidia-smi'):
            rows = ubuntu_cli.check_gpu({}, False)
        self.assertEqual(rows[0]['status'], 'ok')
        self.assertEqual(run.call_count, 1)

    def test_real_gpu_check_requires_an_actual_operation(self):
        for key in ('shared', 'flow', 'exmr'):
            executable(self.root / runtime_registry.PROFILES[key]['dev'] / 'bin/python')
        replies = [(0, 'GPU 0'), (0, '@@DIAG@@' + json.dumps(dict(cuda=True, cuda_matmul_verified=True, torch='2', gpu='G'))),
                   (0, '@@DIAG@@' + json.dumps(dict(cuda=True, torch='2', gpu='G'))), (1, 'import failed')]
        with mock.patch.object(ubuntu_cli, 'ROOT', self.root), mock.patch.object(ubuntu_cli.shutil, 'which', return_value='nvidia-smi'), \
                mock.patch.object(ubuntu_cli, 'linux_variant', return_value=None), \
                mock.patch.object(ubuntu_cli, 'capture', side_effect=replies):
            rows = ubuntu_cli.check_gpu({}, True)
        self.assertEqual([row['status'] for row in rows], ['ok', 'ok', 'fail', 'fail'])


class Descendants(unittest.TestCase):
    def test_tree_termination(self):
        try:
            import psutil  # noqa: F401
        except ImportError:
            self.skipTest('psutil is part of the pinned environments, not the system Python.')
        parent = subprocess.Popen(['sh', '-c', 'sleep 60 & wait'])
        self.addCleanup(parent.wait)
        self.addCleanup(parent.kill)
        import time
        children = []
        for _ in range(100):
            children = psutil.Process(parent.pid).children(recursive=True)
            if children:
                break
            time.sleep(.02)
        self.assertTrue(children)
        self.assertEqual({child.pid for child in linux_runtime.terminate_descendants(parent.pid)}, {child.pid for child in children})
        psutil.wait_procs(children, timeout=5)


class ArchitectureSelection(unittest.TestCase):
    def test_machine_names_are_normalized(self):
        for value, expected in (('x86_64', 'x86_64'), ('AMD64', 'x86_64'), ('aarch64', 'aarch64'), ('arm64', 'aarch64')):
            self.assertEqual(linux_runtime.normalize_machine(value), expected)
        self.assertEqual(linux_runtime.normalize_machine('riscv64'), 'riscv64')

    def test_amd64_uses_the_unchanged_profile_objects(self):
        self.assertIsNone(linux_runtime.linux_variant('x86_64'))
        self.assertEqual(linux_runtime.variant_identity('x86_64'), {})
        for key, spec in runtime_registry.PROFILES.items():
            self.assertIs(linux_runtime.effective_profile(key, 'x86_64'), spec)

    def test_aarch64_replaces_only_torch_and_torchvision(self):
        for key, base in runtime_registry.PROFILES.items():
            spec = linux_runtime.effective_profile(key, 'aarch64')
            self.assertEqual((spec['packages']['torch'], spec['packages']['torchvision']), ('2.9.1+cu130', '0.24.1'))
            self.assertEqual({k: v for k, v in spec['packages'].items() if k not in ('torch', 'torchvision')},
                             {k: v for k, v in base['packages'].items() if k != 'torch'})
            self.assertEqual((spec['python'], spec['dev']), (base['python'], base['dev']))
        self.assertEqual(digest(json.dumps(runtime_registry.PROFILES, sort_keys=True).encode()), PROFILES_DIGEST)

    def test_variant_identity_tracks_the_exact_wheels(self):
        first = linux_runtime.variant_identity('aarch64')['linux_variant']
        self.assertEqual((first['name'], first['machine']), ('linux-aarch64-cu130', 'aarch64'))
        variant = json.loads(json.dumps(linux_runtime.linux_variant('aarch64')))
        variant['cuda_packages']['torch']['sha256']['3.12'] = '0' * 64
        with mock.patch.object(linux_runtime, 'linux_variant', return_value=variant):
            self.assertNotEqual(linux_runtime.variant_identity('aarch64')['linux_variant']['pins'], first['pins'])

    def test_unavailable_package_import_explains_itself(self):
        finder = linux_runtime.PostImportPatches({}, {'rgbmocap_absent_package': 'No ARM wheel; use Static Camera.'})
        with self.assertRaisesRegex(ModuleNotFoundError, 'No ARM wheel') as caught:
            finder.find_spec('rgbmocap_absent_package')
        self.assertEqual(caught.exception.name, 'rgbmocap_absent_package')
        self.assertIsNone(finder.find_spec('json'))


class ReceiptIdentity(TempRoot):
    """Receipts name the runtime actually used; a machine change must not alias another's cache."""

    def hamer_runtime(self, machine):
        from mocap import flow_hamer
        real = linux_runtime.effective_profile
        with mock.patch.object(linux_runtime, 'effective_profile', lambda key: real(key, machine)), \
                mock.patch('mocap.result_artifacts.file_stat', lambda path: [0, 0]):
            return flow_hamer.source_identity()['runtime']

    def test_hamer_identity_reflects_the_real_runtime(self):
        amd64 = self.hamer_runtime('x86_64')
        self.assertIs(amd64, runtime_registry.PROFILES['exmr'])
        self.assertEqual(amd64['packages']['torch'], '2.8.0+cu128')
        spark = self.hamer_runtime('aarch64')
        self.assertEqual(spark['packages']['torch'], '2.9.1+cu130')
        self.assertEqual(spark['linux_variant']['name'], 'linux-aarch64-cu130')

    def fingerprint(self, machine):
        from mocap import worker
        project = self.root / 'project'
        project.mkdir(exist_ok=True)
        (project / 'project.json').write_text(json.dumps({'pipeline': {'model': 'flowhmr'}}))
        with mock.patch.object(worker, 'variant_identity', lambda: linux_runtime.variant_identity(machine)), \
                mock.patch('mocap.models.STAGE_REQUIREMENTS', {}):
            return worker.fingerprint('track', project)

    def test_stage_fingerprints_only_change_on_the_variant(self):
        amd64 = self.fingerprint('x86_64')
        self.assertEqual(amd64, self.fingerprint('x86_64'))
        self.assertNotEqual(amd64, self.fingerprint('aarch64'))

    def test_receipt_code_adds_nothing_where_profiles_apply(self):
        for name in ('mocap/worker.py', 'mocap/exmr_backend.py', 'mocap/flow_backend.py'):
            self.assertIn('variant_identity()', (ROOT / name).read_text(encoding='utf-8'), name)
        from mocap import runtime_identity
        with mock.patch.object(runtime_identity, 'os', SimpleNamespace(name='nt')):
            self.assertEqual(runtime_identity.variant_identity(), {})
            self.assertIs(runtime_identity.runtime_profile('exmr'), runtime_registry.PROFILES['exmr'])


class AarchPlan(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lock = json.loads((ROOT / 'installer/runtime-lock.json').read_bytes())
        cls.variant = json.loads((ROOT / 'installer/linux-aarch64.json').read_text(encoding='utf-8'))

    def plan(self, profile):
        return setup_ubuntu.profile_plan(ROOT, self.lock, profile, 'aarch64')

    @staticmethod
    def names(pins):
        return {setup_ubuntu.normalize(pin.split('==')[0]): pin.split('==')[1] for pin in pins}

    def test_only_the_cuda_stack_differs_from_amd64(self):
        for profile in setup_ubuntu.FOLDERS:
            arm, amd = self.plan(profile), setup_ubuntu.profile_plan(ROOT, self.lock, profile, 'x86_64')
            self.assertEqual((arm['python'], arm['folder'], arm['variant'], arm['cuda_index']),
                             (amd['python'], amd['folder'], 'linux-aarch64-cu130', 'https://download.pytorch.org/whl/cu130'))
            self.assertEqual(self.names(arm['cuda']), {'torch': '2.9.1+cu130', 'torchvision': '0.24.1'})
            expected = self.names(amd['index'])
            expected.pop('pycolmap', None)
            for name in self.names(arm['gui']):
                expected.pop(name)
            self.assertEqual(self.names(arm['index']), expected, profile)
            self.assertEqual(arm['wheels'], amd['wheels'])

    def test_torchvision_cannot_come_from_a_look_alike_on_pypi(self):
        plan = self.plan('flow')
        self.assertFalse([pin for pin in plan['index'] if setup_ubuntu.normalize(pin.split('==')[0]) in ('torch', 'torchvision')])
        self.assertEqual(plan['cuda_hashes']['torchvision'], self.variant['cuda_packages']['torchvision']['sha256']['3.10'])
        self.assertEqual(self.plan('exmr')['cuda_hashes']['torch'], self.variant['cuda_packages']['torch']['sha256']['3.12'])

    def test_hashes_exist_for_each_python_and_are_well_formed(self):
        minors = {spec['python'].rsplit('.', 1)[0] for spec in runtime_registry.PROFILES.values()}
        for row in self.variant['cuda_packages'].values():
            self.assertEqual(set(row['sha256']), minors)
            for value in row['sha256'].values():
                self.assertRegex(value, r'^[0-9a-f]{64}$')

    def test_unavailable_package_is_reported_only_where_it_is_pinned(self):
        self.assertEqual(list(self.plan('exmr')['unavailable']), ['pycolmap'])
        self.assertEqual(self.plan('flow')['unavailable'], {})
        self.assertNotIn('pycolmap', self.names(self.plan('exmr')['index']))

    def test_gui_is_separate_so_cli_survives_without_it(self):
        self.assertEqual(self.names(self.plan('shared')['gui']), {'pyside6': '6.8.3', 'pyside6-addons': '6.8.3', 'pyside6-essentials': '6.8.3', 'shiboken6': '6.8.3'})
        self.assertEqual(self.plan('flow')['gui'], [])
        amd = setup_ubuntu.profile_plan(ROOT, self.lock, 'shared', 'x86_64')
        self.assertNotIn('gui', amd)
        self.assertIn('pyside6', self.names(amd['index']))

    def test_machine_markers_are_honored(self):
        metadata = '\n'.join(('Requires-Dist: nvidia-a==1; platform_system == "Linux" and platform_machine == "x86_64"',
                              'Requires-Dist: nvidia-b==2; platform_system == "Linux" and platform_machine == "aarch64"',
                              'Requires-Dist: nvidia-c==3; platform_system == "Linux" and platform_machine != "x86_64"',
                              'Requires-Dist: nvidia-d==4; platform_system == "Linux"'))
        self.assertEqual(setup_ubuntu.linux_dependencies(metadata, set(), 'aarch64'), ['nvidia-b==2', 'nvidia-c==3', 'nvidia-d==4'])
        self.assertEqual(setup_ubuntu.linux_dependencies(metadata, set(), 'x86_64'), ['nvidia-a==1', 'nvidia-d==4'])

    def test_other_machines_are_refused_and_cross_install_is_refused(self):
        with mock.patch('builtins.print'), self.assertRaisesRegex(SystemExit, 'No pinned wheels'):
            setup_ubuntu.main(['--plan', '--machine', 'riscv64'])
        other = 'x86_64' if linux_runtime.normalize_machine() == 'aarch64' else 'aarch64'
        with self.assertRaisesRegex(SystemExit, 'only be used with --plan'):
            setup_ubuntu.main(['--machine', other])


class AarchInstall(TempRoot):
    def test_hash_pinned_cuda_install_and_honest_receipt(self):
        plan = dict(profile='shared', folder='.venv', python='3.10.22', machine='aarch64', variant='linux-aarch64-cu130',
                    cuda_index='https://download.pytorch.org/whl/cu130', cuda=['torch==2.9.1+cu130', 'torchvision==0.24.1'],
                    cuda_hashes={'torch': 'a' * 64, 'torchvision': 'b' * 64}, index=['numpy==1.23.5'], wheels=[], gui=['pyside6==6.8.3'],
                    unavailable={'pycolmap': 'No wheel.'})
        site = self.root / '.venv/lib/python3.10/site-packages'
        (site / 'torch-2.9.1+cu130.dist-info').mkdir(parents=True)
        (site / 'torch-2.9.1+cu130.dist-info/METADATA').write_text('Requires-Dist: triton==3.5.1; platform_system == "Linux"\n')
        executable(self.root / '.venv/bin/python')
        calls = []

        def fake_run(command, check=True):
            calls.append(list(map(str, command)))
            # The GUI wheel step fails here: processing must still be prepared.
            return SimpleNamespace(returncode=1 if 'pyside6==6.8.3' in calls[-1] else 0)
        done = SimpleNamespace(returncode=0, stdout='3.10.22\n')
        with mock.patch.object(setup_ubuntu, 'ROOT', self.root), mock.patch.object(setup_ubuntu, 'run', side_effect=fake_run), \
                mock.patch.object(setup_ubuntu.subprocess, 'run', return_value=done), mock.patch('builtins.print') as output:
            setup_ubuntu.install_profile('uv', plan, self.root / 'wheels', self.root / 'work', False)
        first = next(call for call in calls if call[:3] == ['uv', 'pip', 'install'])
        self.assertIn('--require-hashes', first)
        self.assertEqual(first[first.index('--index-url') + 1], plan['cuda_index'])
        requirements = Path(first[first.index('-r') + 1]).read_text()
        self.assertIn(f"torchvision==0.24.1 --hash=sha256:{'b' * 64}", requirements)
        self.assertEqual([call[-1] for call in calls if '--runtime-check' in call], ['shared'])
        receipt = json.loads((self.root / '.venv' / linux_runtime.RECEIPT).read_text())
        self.assertEqual((receipt['machine'], receipt['variant'], receipt['gui_available'], receipt['cuda_verified'], receipt['linux_extras']),
                         ('aarch64', 'linux-aarch64-cu130', False, False, ['triton==3.5.1']))
        text = ' '.join(' '.join(map(str, call.args)) for call in output.call_args_list)
        self.assertIn('CUDA has not been run', text)
        self.assertIn('NOT INSTALLED pycolmap', text)


class NativeBuildFlags(TempRoot):
    @classmethod
    def setUpClass(cls):
        cls.cmake = (ROOT / 'third_party/KimodoMotionCorrection/CMakeLists.txt').read_text(encoding='utf-8')

    def branch(self):
        start = self.cmake.index('if(MSVC)')
        return self.cmake[start:self.cmake.index('# Python bindings')]

    def test_x86_and_msvc_flags_are_preserved_exactly(self):
        branch = self.branch()
        self.assertIn('target_compile_options(motion_correction_cpp_base PRIVATE /W4 /arch:AVX)', branch)
        self.assertIn('else()\n    # GCC/Clang flags (also applies to MinGW on Windows)\n    # Enable SSE4.1 and AVX instructions for SIMD operations\n'
                      '    target_compile_options(motion_correction_cpp_base PRIVATE -Wall -Wextra -msse4.1 -mavx)\nendif()', branch)

    def test_aarch64_branch_uses_no_x86_flags_and_keeps_x86_semantics(self):
        branch = self.branch()
        arm = branch[branch.index('elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")'):branch.index('# GCC/Clang flags')]
        self.assertNotIn('-msse', arm)
        self.assertNotIn('-mavx', arm)
        self.assertIn('-fsigned-char -ffp-contract=off', arm)
        self.assertIn('SSE2NEON_PRECISE_MINMAX=1', arm)
        self.assertNotRegex(arm, r'SSE2NEON_PRECISE_SQRT=|SSE2NEON_PRECISE_DIV=')
        self.assertRegex(arm, r'sse2neon/[0-9a-f]{40}/sse2neon\.h')
        self.assertRegex(arm, r'URL_HASH SHA256=[0-9a-f]{64}')

    def test_numerical_source_uses_the_adapter_not_edits(self):
        self.assertIn('#include <immintrin.h>', (ROOT / 'third_party/KimodoMotionCorrection/src/cpp/Math/SIMD.h').read_text(encoding='utf-8'))
        done = subprocess.run(['git', 'status', '--porcelain', '--', 'third_party/KimodoMotionCorrection/src', 'third_party/KimodoMotionCorrection/python'],
                              cwd=ROOT, capture_output=True, text=True)
        if done.returncode == 0:
            self.assertEqual(done.stdout, '')

    @unittest.skipUnless(linux_runtime.normalize_machine() == 'aarch64' and shutil.which('g++'), 'needs an aarch64 compiler')
    def test_adapter_matches_x86_results_in_a_real_compile(self):
        folder = os.environ.get('RGB_MOCAP_SSE2NEON_DIR') or ROOT / 'build/motion-correction/_deps/sse2neon-src'
        if not (Path(folder) / 'sse2neon.h').is_file():
            self.skipTest('The pinned sse2neon header has not been fetched by a native build.')
        source = self.root / 'semantics.cpp'
        source.write_text(SEMANTICS_SOURCE)
        program = self.root / 'semantics'
        defines = re.findall(r'SSE2NEON_PRECISE_\w+=\d', self.branch())
        subprocess.run(['g++', '-std=c++17', '-O3', '-fsigned-char', '-ffp-contract=off', *[f'-D{item}' for item in defines], '-isystem', str(folder),
                        '-isystem', str(ROOT / 'third_party/KimodoMotionCorrection/compat/aarch64'), str(source), '-o', str(program)], check=True, capture_output=True)
        done = subprocess.run([str(program)], capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stdout)


SEMANTICS_SOURCE = r"""
#include <immintrin.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
static int failures=0;
static uint32_t bits(float v){uint32_t u;std::memcpy(&u,&v,4);return u;}
static void lanes(const char*name,__m128 v,float a,float b,float c,float d){float o[4];_mm_storeu_ps(o,v);float e[4]={a,b,c,d};for(int i=0;i<4;i++)if(!((std::isnan(e[i])&&std::isnan(o[i]))||bits(o[i])==bits(e[i]))){failures++;printf("FAIL %s\\n",name);return;}}
int main(){const float nan=std::numeric_limits<float>::quiet_NaN();__m128 a=_mm_setr_ps(1,2,3,4);uint32_t c[4];
 lanes("permutevar",_mm_permutevar_ps(a,_mm_setr_epi32(3,2,1,0)),4,3,2,1);
 lanes("permutevar high bits",_mm_permutevar_ps(a,_mm_setr_epi32(4,5,0x7ffffff2,-1)),1,2,3,4);
 lanes("min NaN",_mm_min_ps(_mm_setr_ps(nan,1,nan,5),_mm_setr_ps(1,nan,nan,2)),1,nan,nan,2);
 lanes("max NaN",_mm_max_ps(_mm_setr_ps(nan,1,nan,5),_mm_setr_ps(1,nan,nan,2)),1,nan,nan,5);
 _mm_storeu_si128((__m128i*)c,_mm_cvttps_epi32(_mm_setr_ps(3e9f,-3e9f,nan,-1.9f)));
 if(c[0]!=0x80000000||c[1]!=0x80000000||c[2]!=0x80000000||c[3]!=0xffffffff){failures++;printf("FAIL cvttps\\n");}
 uint32_t state=2463534242u;auto next=[&]{state^=state<<13;state^=state>>17;state^=state<<5;return 1e-3f+(float)(state%2000000)/1000.f;};
 for(int i=0;i<50000;i++){volatile float x=next(),y=next();float o[4];
  _mm_storeu_ps(o,_mm_div_ps(_mm_set1_ps(x),_mm_set1_ps(y)));volatile float d=x/y;if(bits(o[0])!=bits(d)){failures++;printf("FAIL div\\n");break;}
  _mm_storeu_ps(o,_mm_sqrt_ps(_mm_set1_ps(x)));if(bits(o[0])!=bits(std::sqrt((float)x))){failures++;printf("FAIL sqrt\\n");break;}
  _mm_store_ss(o,_mm_sqrt_ss(_mm_set_ss(x)));if(bits(o[0])!=bits(std::sqrt((float)x))){failures++;printf("FAIL sqrt_ss\\n");break;}}
 volatile float e=1.f+1.f/4096;volatile float k=-(1.f+1.f/2048);float r[4];
 _mm_storeu_ps(r,_mm_add_ps(_mm_mul_ps(_mm_set1_ps(e),_mm_set1_ps(e)),_mm_set1_ps(k)));if(r[0]!=0.f){failures++;printf("FAIL fused\\n");}
 if(!((char)-1<0)){failures++;printf("FAIL char\\n");}
 return failures!=0;}
"""


class CorrectionCrosscheck(TempRoot):
    def run_tool(self, *arguments):
        return subprocess.run([sys.executable, '-I', '-B', str(ROOT / 'tools/correction_crosscheck.py'), *map(str, arguments)], capture_output=True, text=True)

    def report(self, name, roots, rotations, machine='a'):
        import base64
        from array import array
        path = self.root / name
        path.write_text(json.dumps(dict(machine=machine, metrics=dict(m=1.), roots=base64.b64encode(array('f', roots).tobytes()).decode(),
                                        rotations=base64.b64encode(array('f', rotations).tobytes()).decode())))
        return path

    def test_identical_reports_have_exactly_zero_difference(self):
        first = self.report('a.json', [1, 2, 3], [.5, .5, .5, .5])
        done = self.run_tool('compare', first, first)
        self.assertEqual(done.returncode, 0, done.stderr)
        value = json.loads(done.stdout[:done.stdout.rindex('}') + 1])
        self.assertEqual((value['identical'], value['max_root_difference_m'], value['max_rotation_difference_deg']), (True, 0.0, 0.0))

    def test_real_differences_are_measured_and_fail(self):
        first = self.report('a.json', [1, 2, 3], [1, 0, 0, 0])
        # A 1 degree rotation about x, and a 1 mm translation.
        half = math.radians(.5)
        second = self.report('b.json', [1.001, 2, 3], [math.cos(half), math.sin(half), 0, 0], 'b')
        done = self.run_tool('compare', first, second)
        self.assertEqual(done.returncode, 1)
        value = json.loads(done.stdout[:done.stdout.rindex('}') + 1])
        self.assertAlmostEqual(value['max_root_difference_m'], .001, places=5)
        self.assertAlmostEqual(value['max_rotation_difference_deg'], 1.0, places=3)
        self.assertIn('DIFFERENT BEYOND TOLERANCE', done.stdout)
        self.assertEqual(self.run_tool('compare', first, second, '--max-root-mm', 2, '--max-rotation-deg', 2).returncode, 0)

    def test_opposite_quaternion_signs_are_the_same_rotation(self):
        first = self.report('a.json', [0], [.5, .5, .5, .5])
        second = self.report('b.json', [0], [-.5, -.5, -.5, -.5])
        value = json.loads(self.run_tool('compare', first, second).stdout.rsplit('\n', 2)[0])
        self.assertEqual(value['max_rotation_difference_deg'], 0.0)

    def test_real_module_is_deterministic_when_available(self):
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest('Needs NumPy from a pinned environment.')
        if linux_runtime.correction_path(ROOT) is None:
            self.skipTest('No correction module is built for this interpreter.')
        outputs = [self.root / 'first.json', self.root / 'second.json']
        for output in outputs:
            done = self.run_tool('run', '--output', output)
            self.assertEqual(done.returncode, 0, done.stderr)
        reports = [json.loads(path.read_text()) for path in outputs]
        self.assertEqual(reports[0]['rotations'], reports[1]['rotations'])
        self.assertGreater(reports[0]['changed_rotation_values'], 0)


class DoctorArchitecture(TempRoot):
    def test_architecture_row_names_the_variant(self):
        with mock.patch.object(ubuntu_cli, 'normalize_machine', return_value='aarch64'), \
                mock.patch.object(ubuntu_cli, 'linux_variant', return_value=linux_runtime.linux_variant('aarch64')):
            rows = ubuntu_cli.check_architecture()
        self.assertIn('Torch 2.9.1+cu130', rows[0]['detail'])
        self.assertEqual((rows[1]['name'], rows[1]['status']), ('not available: pycolmap', 'warn'))
        with mock.patch.object(ubuntu_cli, 'normalize_machine', return_value='riscv64'), mock.patch.object(ubuntu_cli, 'linux_variant', return_value=None):
            self.assertEqual(ubuntu_cli.check_architecture()[0]['status'], 'fail')

    def test_cpu_torchvision_is_a_failure_on_the_variant(self):
        library = self.root / '.venv/lib/python3.10/site-packages/torchvision/_C.so'
        library.parent.mkdir(parents=True)
        library.write_bytes(b'libtorch_cpu.so only')
        with mock.patch.object(ubuntu_cli, 'ROOT', self.root), mock.patch.object(ubuntu_cli, 'linux_variant', return_value=linux_runtime.linux_variant('aarch64')), \
                mock.patch.object(ubuntu_cli, 'effective_profile', lambda key: dict(dev='.venv')):
            rows = ubuntu_cli.check_torchvision_builds()
            self.assertEqual([row['status'] for row in rows], ['fail'])
            library.write_bytes(b'needs libcudart.so.13')
            self.assertEqual([row['status'] for row in ubuntu_cli.check_torchvision_builds()], ['ok'])

    def test_variant_real_gpu_check_requires_a_cuda_nms(self):
        for key in ('shared', 'flow', 'exmr'):
            executable(self.root / runtime_registry.PROFILES[key]['dev'] / 'bin/python')
        good = dict(cuda=True, cuda_matmul_verified=True, torchvision_nms_verified=True, torch='2.9.1+cu130', gpu='GB10', compute_capability=[12, 1])
        without = {key: value for key, value in good.items() if key != 'torchvision_nms_verified'}
        replies = [(0, 'GPU 0'), (0, '@@DIAG@@' + json.dumps(good)), (0, '@@DIAG@@' + json.dumps(without)), (0, '@@DIAG@@' + json.dumps({**without, 'torchvision_nms_error': 'no CUDA kernel'}))]
        with mock.patch.object(ubuntu_cli, 'ROOT', self.root), mock.patch.object(ubuntu_cli.shutil, 'which', return_value='nvidia-smi'), \
                mock.patch.object(ubuntu_cli, 'linux_variant', return_value=linux_runtime.linux_variant('aarch64')), mock.patch.object(ubuntu_cli, 'capture', side_effect=replies):
            rows = ubuntu_cli.check_gpu({}, True)
        self.assertEqual([row['status'] for row in rows], ['ok', 'ok', 'fail', 'fail'])
        self.assertIn('sm_121', rows[1]['detail'])
        self.assertIn('no CUDA kernel', rows[3]['detail'])

    def test_missing_gui_is_only_a_warning_on_the_variant(self):
        with mock.patch.object(ubuntu_cli, 'ROOT', self.root), mock.patch.object(ubuntu_cli, 'linux_variant', return_value=linux_runtime.linux_variant('aarch64')):
            rows = {row['name']: row['status'] for row in ubuntu_cli.check_profiles({})}
        self.assertEqual((rows['runtime gui'], rows['runtime flow']), ('warn', 'fail'))


class Documentation(unittest.TestCase):
    def test_upstream_baseline_is_recorded(self):
        text = (ROOT / 'docs/UBUNTU.md').read_text(encoding='utf-8')
        self.assertIn(BASE_SHA, text)
        for command in ('tools/setup_ubuntu.py', 'rgbmocap.sh doctor', 'RGBMoCap.sh', 'PLATFORM_ONLY'):
            self.assertIn(command, text)


if __name__ == '__main__':
    unittest.main()
