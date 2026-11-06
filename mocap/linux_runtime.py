"""Linux routing and validation for the independent pinned virtual environments.

`runtime_registry.py` defers here only when `os.name != 'nt'`; Windows never imports
this module. Every profile uses its own complete `<root>/<dev>/` environment: no
Windows `runtime*/python.exe` path, overlay folder or `.pth` chain is used on Linux.
PROFILES stay the single source of pins and must not gain Linux keys, because inference
receipts embed them.
"""
from __future__ import annotations

import functools
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
import platform
import sys
from pathlib import Path

SETUP_HINT = 'Run tools/setup_ubuntu.py to prepare it; see docs/UBUNTU.md.'
RECEIPT = 'rgbmocap-linux.json'
CORRECTION_FOLDER = 'third_party/KimodoMotionCorrection/bin'
# Checked in addition to PROFILES without changing those receipt-visible dictionaries.
EXTRA_PACKAGES = {'gui': {'PySide6': '6.8.3'}}
# Machines whose official Torch wheels need pins other than PROFILES. Anything absent here
# uses PROFILES unchanged, so amd64 receipts stay identical to the Windows-derived ones.
VARIANT_FILES = {'aarch64': 'installer/linux-aarch64.json'}
SUPPORTED_MACHINES = ('x86_64', 'aarch64')


def profiles():
    from .runtime_registry import PROFILES
    return PROFILES


def normalize_machine(machine=None):
    value = (machine or platform.machine()).lower()
    return {'amd64': 'x86_64', 'x64': 'x86_64', 'arm64': 'aarch64'}.get(value, value)


@functools.lru_cache(maxsize=4)
def _load_variant(machine):
    name = VARIANT_FILES.get(machine)
    if name is None:
        return None
    return json.loads((Path(__file__).resolve().parents[1] / name).read_text(encoding='utf-8'))


def linux_variant(machine=None):
    """The separately pinned variant for this machine, or None to use PROFILES."""
    return _load_variant(normalize_machine(machine))


def variant_identity(machine=None):
    """Receipt keys naming the variant and its exact wheels; empty when PROFILES apply."""
    variant = linux_variant(machine)
    if variant is None:
        return {}
    pins = json.dumps(variant['cuda_packages'], sort_keys=True).encode()
    return {'linux_variant': dict(name=variant['name'], machine=variant['machine'], pins=hashlib.sha256(pins).hexdigest())}


def effective_profile(key, machine=None):
    """PROFILES[key] itself where it applies; otherwise a copy with the variant's pins."""
    base = profiles()[key]
    variant = linux_variant(machine)
    if variant is None:
        return base
    packages = {**base['packages'], **{name: row['version'] for name, row in variant['cuda_packages'].items()}}
    return {**base, 'packages': packages, **variant_identity(machine)}


def venv_folder(root, key):
    return Path(root) / profiles()[key]['dev']


def interpreter(root, key):
    # Do not resolve this path: Python finds pyvenv.cfg from the unresolved symlink.
    return venv_folder(root, key) / 'bin' / 'python'


def command(root, stage, args=()):
    """Same isolated UTF-8 worker command as a managed install, in the profile's own venv."""
    from .runtime_registry import profile_for
    root = Path(root)
    key = profile_for(stage, args)
    python = interpreter(root, key)
    if not python.is_file():
        raise RuntimeError(f'{key} runtime is missing ({python}). {SETUP_HINT}')
    # Isolated mode ignores PYTHONPATH/PYTHONUTF8; -X utf8 stays explicit.
    return [str(python), '-I', '-X', 'utf8', str(root / 'worker_boot.py'), stage, *map(str, args)]


def allowed_python(venv, pinned):
    """The exact pin, or a same-minor patch deliberately recorded by setup."""
    allowed = {pinned}
    try:
        receipt = json.loads((Path(venv) / RECEIPT).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return allowed
    version = str(receipt.get('python', ''))
    if receipt.get('python_patch_fallback') is True and version.split('.')[:2] == pinned.split('.')[:2]:
        allowed.add(version)
    return allowed


def site_packages(venv):
    return Path(venv) / 'lib' / f'python{sys.version_info[0]}.{sys.version_info[1]}' / 'site-packages'


def activate(root, key):
    """Validate effective imports before any model, CUDA or GUI module loads."""
    import importlib.metadata as metadata
    root = Path(root).resolve()
    spec = effective_profile(key)
    venv = root / spec['dev']
    if sys.prefix == sys.base_prefix or Path(sys.prefix).resolve() != venv.resolve():
        raise RuntimeError(f'Wrong interpreter for {key}: {sys.prefix} is not {venv}. {SETUP_HINT}')
    actual = '.'.join(map(str, sys.version_info[:3]))
    if actual not in allowed_python(venv, spec['python']):
        raise RuntimeError(f"Unexpected Python {actual} for {key}; expected {spec['python']}. {SETUP_HINT}")
    site = site_packages(venv)
    if not site.is_dir():
        raise RuntimeError(f'{key} runtime is missing {site}. {SETUP_HINT}')
    if (site / 'shared_runtime.pth').exists():
        raise RuntimeError(f'{key} uses a Windows-style overlay; Linux profiles must be self-contained. {SETUP_HINT}')
    packages = {**spec['packages'], **EXTRA_PACKAGES.get(key, {})}
    expected = site.resolve()
    imports = {}
    for name, version in packages.items():
        try:
            found = metadata.distribution(name).version
        except metadata.PackageNotFoundError:
            raise RuntimeError(f'{name} is missing from {key}. {SETUP_HINT}') from None
        module = importlib.util.find_spec(name)
        if found != version or module is None or not module.origin:
            raise RuntimeError(f'Incompatible {name} {found} in {key}; expected {version}. {SETUP_HINT}')
        origin = Path(module.origin).resolve()
        if not origin.is_relative_to(expected):
            raise RuntimeError(f'Unexpected {name} import path in {key}: {origin}')
        imports[name] = str(origin)
    install_import_hooks()
    return dict(profile=key, python=sys.executable, packages=packages, imports=imports,
                platform=sys.platform, environment=str(venv), machine=normalize_machine(),
                variant=(linux_variant() or {}).get('name'))


def runtime_environment(root, base=None):
    """Private data/cache locations and optional local native tools for a Linux session."""
    base = os.environ if base is None else base
    root = Path(root)
    data = Path(base.get('RGB_MOCAP_DATA') or root)
    env = dict(RGB_MOCAP_DATA=str(data), PYTHONDONTWRITEBYTECODE='1', HF_HOME=str(data / 'cache/huggingface'),
               TORCH_HOME=str(data / 'cache/torch'), XDG_CACHE_HOME=str(data / 'cache'))
    for name, folder in (('PATH', root / 'tools/bin'), ('LD_LIBRARY_PATH', root / '.runtime/lib')):
        current = base.get(name, '')
        if folder.is_dir() and str(folder) not in current.split(os.pathsep):
            env[name] = str(folder) + (os.pathsep + current if current else '')
        elif current:
            env[name] = current
    return env


def correction_path(root, suffixes=None):
    """The native module built for this interpreter's ABI, most specific tag first."""
    folder = Path(root) / CORRECTION_FOLDER
    for suffix in (importlib.machinery.EXTENSION_SUFFIXES if suffixes is None else suffixes):
        candidate = folder / ('_motion_correction' + suffix)
        if candidate.is_file():
            return candidate
    return None


_loaded_correction = {}


def correction_backend():
    """Linux counterpart of correction.backend(), which only recognizes Windows .pyd files."""
    from .paths import ROOT
    path = correction_path(ROOT)
    if path is None:
        raise RuntimeError('Motion Correction runtime is missing. Build it with tools/build_motion_correction.py; see docs/UBUNTU.md.')
    if path not in _loaded_correction:
        spec = importlib.util.spec_from_file_location('_motion_correction', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _loaded_correction[path] = module
    return _loaded_correction[path]


class _PatchingLoader(importlib.abc.Loader):
    def __init__(self, loader, patch):
        self._loader = loader
        self._patch = patch

    def __getattr__(self, name):
        return getattr(self._loader, name)

    def create_module(self, spec):
        return self._loader.create_module(spec)

    def exec_module(self, module):
        self._loader.exec_module(module)
        self._patch(module)


class PostImportPatches(importlib.abc.MetaPathFinder):
    """Adjust a module right after it executes, without loading it any earlier.

    correction.py is byte-hashed into FlowHMR correction and character receipts, so
    editing it for Linux would invalidate saved Windows work. Importing it here
    instead would load NumPy before workers configure their thread limits.
    """

    def __init__(self, patches, unavailable=None):
        self.patches = patches
        self.unavailable = unavailable or {}

    def find_spec(self, fullname, path=None, target=None):
        # Replace a bare "No module named" with the reason this machine lacks the package.
        if fullname in self.unavailable and importlib.machinery.PathFinder.find_spec(fullname, path, target) is None:
            raise ModuleNotFoundError(self.unavailable[fullname], name=fullname)
        # HandFlow's hashed engine still selects PyOpenGL's Windows platform; PyOpenGL
        # reads the value on import. A user-chosen platform is left unchanged.
        if fullname == 'OpenGL' and os.environ.get('PYOPENGL_PLATFORM') == 'win32':
            os.environ['PYOPENGL_PLATFORM'] = 'egl'
        patch = self.patches.get(fullname)
        if patch is None:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is not None and spec.loader is not None:
            spec.loader = _PatchingLoader(spec.loader, patch)
        return spec


def _patch_correction(module):
    module.backend = correction_backend


def install_import_hooks():
    if any(isinstance(finder, PostImportPatches) for finder in sys.meta_path):
        return
    name = __package__ + '.correction'
    if name in sys.modules:
        _patch_correction(sys.modules[name])
    unavailable = {package: row['reason'] for package, row in (linux_variant() or {}).get('unavailable', {}).items()}
    sys.meta_path.insert(0, PostImportPatches({name: _patch_correction}, unavailable))


def terminate_descendants(pid):
    """Stop a cancelled worker's children; Linux has no `taskkill /T` equivalent."""
    import psutil
    try:
        children = psutil.Process(pid).children(recursive=True)
    except psutil.Error:
        return []
    for child in children:
        try:
            child.kill()
        except psutil.Error:
            pass
    return children
