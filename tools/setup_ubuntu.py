#!/usr/bin/env python3
"""Prepare independent, pinned Ubuntu environments; no system or user-level changes.

Each profile is one complete venv. Base and overlay pins are merged by package name, with
the overlay winning, so the imports match the Windows overlay without any shared folder:

    shared -> .venv        shared + gui (PySide6)
    flow   -> .venv-flow   shared + flow
    hand   -> .venv-hand   shared + hand
    htd    -> .venv-htd    shared + htd
    exmr   -> .venv-exmr   exmr only, on its own Python

Packages install with --no-deps from the exact readable pins, like the Windows lock. Linux
CUDA libraries come from the pinned Torch wheel metadata; iopath also receives the
existing pinned portalocker dependency.
Model weights are never downloaded here; --stage-assets fetches auxiliary files only.

amd64 installs exactly these pins. aarch64 (NVIDIA DGX Spark) replaces only Torch and
torchvision, from installer/linux-aarch64.json, because the amd64 CUDA builds have no ARM
wheels or do not support its GPU. Those two are hash-pinned to the CUDA index: PyPI has a
different torchvision build under the same version.

Setup validates versions and import locations. It does not run CUDA; use doctor --gpu.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from runtime_support import ROOT, prepared_lock, sha  # noqa: E402

sys.path.insert(0, str(ROOT))
from mocap.linux_runtime import SUPPORTED_MACHINES, linux_variant, normalize_machine  # noqa: E402

CUDA_INDEX = 'https://download.pytorch.org/whl/cu128'
UV_VERSION = '0.13.1'
FOLDERS = {'shared': '.venv', 'flow': '.venv-flow', 'hand': '.venv-hand', 'htd': '.venv-htd', 'exmr': '.venv-exmr'}
GROUPS = {'shared': ('shared', 'gui'), 'flow': ('shared', 'flow'), 'hand': ('shared', 'hand'),
          'htd': ('shared', 'htd'), 'exmr': ('exmr',)}
# FlowHMR's HaMeR refinement runs in the ExMR environment.
DEFAULT_PROFILES = ('shared', 'flow', 'exmr')
# Windows-only dependencies that exist in the shared readable pins.
WINDOWS_ONLY = {'win32-setctime'}


def normalize(name):
    return re.sub(r'[-_.]+', '-', name).lower()


def read_pins(path):
    pins = {}
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        name, separator, version = line.partition('==')
        if not separator:
            raise ValueError(f'{path}: expected an exact pin, found {line!r}')
        pins[normalize(name)] = (name, version)
    return pins


def group_pins(root, lock, group):
    """Readable runtime-*.txt pins; the GUI group exists only in the runtime lock."""
    path = Path(root) / 'requirements' / f'runtime-{group}.txt'
    if path.is_file():
        return read_pins(path)
    packages = next(g['packages'] for g in lock['groups'] if g['key'] == group)
    return {normalize(p['name']): (p['name'], p['version']) for p in packages}


def profile_plan(root, lock, profile, machine=None):
    """Later groups win by name, matching the import order of a Windows overlay."""
    machine = normalize_machine(machine)
    merged, bundled = {}, {}
    python = next(g['python'] for g in lock['groups'] if g['key'] == GROUPS[profile][0])
    for group in GROUPS[profile]:
        merged.update(group_pins(root, lock, group))
        for package in next(g['packages'] for g in lock['groups'] if g['key'] == group):
            if package.get('bundled'):
                bundled[normalize(package['name'])] = (package['version'], package['bundled'])
    # iopath imports portalocker on Linux. The Windows base lock omits it;
    # reuse the already reviewed ExMR pin instead of resolving a new version.
    if 'iopath' in merged and 'portalocker' not in merged:
        merged['portalocker'] = group_pins(root, lock, 'exmr')['portalocker']
    pins = {key: value for key, value in merged.items() if key not in WINDOWS_ONLY}
    variant = linux_variant(machine)
    extra = {}
    if variant is not None:
        extra = variant_plan(variant, pins, group_pins(root, lock, 'gui') if 'gui' in GROUPS[profile] else {}, profile, python)
    # A bundled pin overridden by a later group must install from the index instead.
    wheels = {key: wheel for key, (version, wheel) in bundled.items() if key in pins and pins[key][1] == version}
    cuda = sorted(f'{name}=={version}' for name, version in pins.values() if '+cu' in version)
    index = sorted(f'{name}=={version}' for key, (name, version) in pins.items() if '+cu' not in version and key not in wheels)
    return dict(profile=profile, folder=FOLDERS[profile], python=python, groups=list(GROUPS[profile]),
                cuda=extra.get('cuda', cuda), index=index, wheels=sorted(wheels.values()), machine=machine,
                variant=variant['name'] if variant else None, cuda_index=variant['cuda_index'] if variant else CUDA_INDEX,
                **{key: value for key, value in extra.items() if key != 'cuda'})


def variant_plan(variant, pins, gui, profile, python):
    """Remove the variant's replacements from the plain pins and return what replaces them."""
    minor = python.rsplit('.', 1)[0]
    rows = variant['cuda_packages']
    for name in rows:
        pins.pop(normalize(name), None)
    # Installed on their own so that a missing GUI wheel cannot prevent CLI processing.
    replacements = {normalize(name): version for name, version in variant.get('gui_packages', {}).items()}
    gui_pins = sorted(f'{name}=={replacements.get(key, version)}' for key, (name, version) in gui.items() if pins.pop(key, None) is not None)
    unavailable = {}
    for name, row in variant.get('unavailable', {}).items():
        if profile in row['profiles'] and pins.pop(normalize(name), None) is not None:
            unavailable[name] = row['reason']
    return dict(cuda=sorted(f"{name}=={row['version']}" for name, row in rows.items()),
                cuda_hashes={name: row['sha256'][minor] for name, row in rows.items()},
                gui=gui_pins, unavailable=unavailable)


LINUX_MARKER = re.compile(r"""(platform_system|sys_platform)\s*==\s*['"](linux|Linux)['"]""")
OTHER_MARKER = re.compile(r"""(platform_system|sys_platform)\s*==\s*['"](Windows|win32|Darwin|darwin)['"]""")
MACHINE_MARKER = re.compile(r"""platform_machine\s*(==|!=)\s*['"]([^'"]+)['"]""")


def linux_dependencies(metadata, pinned, machine='x86_64'):
    """Exact Linux requirements of a Torch wheel that the Windows pins cannot list."""
    result = []
    for line in metadata.splitlines():
        if not line.startswith('Requires-Dist:'):
            continue
        requirement, _, marker = line[len('Requires-Dist:'):].partition(';')
        if 'extra ==' in marker or OTHER_MARKER.search(marker) or not LINUX_MARKER.search(marker):
            continue
        if any((value == machine) != (operator == '==') for operator, value in MACHINE_MARKER.findall(marker)):
            continue
        requirement = requirement.strip().replace(' ', '').replace('(', '').replace(')', '')
        name = re.match(r'[A-Za-z0-9._-]+', requirement).group(0)
        if normalize(name) not in pinned:
            result.append(requirement)
    return sorted(result)


def find_uv(explicit=None):
    if explicit:
        return Path(explicit).resolve()
    found = shutil.which('uv')
    if found:
        return Path(found)
    local = ROOT / '.bootstrap/bin/uv'
    if not local.is_file():
        raise SystemExit(f'uv was not found. Install uv {UV_VERSION} privately, then pass --uv:\n'
                         f'  python3 -m venv {ROOT}/.bootstrap && {ROOT}/.bootstrap/bin/pip install uv=={UV_VERSION}')
    return local


def uv_environment():
    env = os.environ.copy()
    # Keep Python and wheels inside this checkout. The same-filesystem cache hardlinks
    # identical wheels, so repeated Torch files do not multiply across venvs.
    env.update(PYTHONUTF8='1', UV_PYTHON_INSTALL_DIR=str(ROOT / '.runtime/python'), UV_CACHE_DIR=str(ROOT / '.cache/uv'),
               UV_NO_CONFIG='1')
    for name in ('VIRTUAL_ENV', 'PYTHONHOME', 'PYTHONPATH', 'CONDA_PREFIX'):
        env.pop(name, None)
    return env


def run(command, check=True):
    print('+', ' '.join(map(str, command)), flush=True)
    return subprocess.run([str(part) for part in command], cwd=ROOT, env=uv_environment(), check=check)


def ensure_python(uv, version, allow_fallback):
    """Return (requested interpreter, whether a same-minor patch was accepted)."""
    if run([uv, 'python', 'install', version, '--no-bin'], check=False).returncode == 0:
        return version, False
    if not allow_fallback:
        raise SystemExit(f'uv cannot provide Python {version}. Pass --python-patch-fallback to use the newest {version.rsplit(".", 1)[0]} patch.')
    minor = version.rsplit('.', 1)[0]
    run([uv, 'python', 'install', minor, '--no-bin'])
    return minor, True


def install_profile(uv, plan, wheels_root, work, allow_fallback):
    venv = ROOT / plan['folder']
    python = venv / 'bin/python'
    requested, fallback = ensure_python(uv, plan['python'], allow_fallback)
    if not python.is_file():
        run([uv, 'venv', venv, '--python', requested, '--python-preference', 'only-managed'])
    requirements = work / f"{plan['profile']}.txt"
    requirements.parent.mkdir(parents=True, exist_ok=True)
    requirements.write_text('\n'.join(plan['index']) + '\n', encoding='utf-8')
    base = [uv, 'pip', 'install', '--python', python, '--no-deps']
    index_url = plan.get('cuda_index', CUDA_INDEX)
    hashes = plan.get('cuda_hashes')
    if hashes:
        cuda_requirements = work / f"{plan['profile']}-cuda.txt"
        cuda_requirements.write_text(''.join(f"{pin} --hash=sha256:{hashes[pin.split('==')[0]]}\n" for pin in plan['cuda']), encoding='utf-8')
        run([*base, '--require-hashes', '--index-url', index_url, '-r', cuda_requirements])
    else:
        run([*base, '--index-url', index_url, *plan['cuda']])
    run([*base, '-r', requirements])
    gui_available = True
    if plan.get('gui'):
        gui_available = run([*base, *plan['gui']], check=False).returncode == 0
        if not gui_available:
            print('WARNING: PySide6 could not be installed for this system. The GUI is unavailable; the CLI still works.', flush=True)
    if plan['wheels']:
        run([*base, *[wheels_root / wheel for wheel in plan['wheels']]])
    pinned = {normalize(pin.split('==')[0]) for pin in plan['index'] + plan['cuda']}
    pinned |= {normalize(Path(wheel).name.split('-')[0]) for wheel in plan['wheels']}
    extras = []
    site = next((venv / 'lib').glob('python*/site-packages'))
    for metadata in site.glob('torch-*.dist-info/METADATA'):
        extras += linux_dependencies(metadata.read_text(encoding='utf-8'), pinned, plan.get('machine', 'x86_64'))
    if extras:
        run([*base, *extras])
    version = subprocess.run([str(python), '-I', '-c', 'import platform;print(platform.python_version())'],
                             capture_output=True, text=True, check=True).stdout.strip()
    receipt = dict(schema=1, profile=plan['profile'], python=version, python_patch_fallback=fallback,
                   pinned_sha256=sha(requirements), linux_extras=extras, machine=plan.get('machine', 'x86_64'),
                   variant=plan.get('variant'), gui_available=gui_available, unavailable=plan.get('unavailable', {}),
                   # Activation checks versions and paths; only a real operation can establish CUDA.
                   cuda_verified=False)
    (venv / 'rgbmocap-linux.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    # Activation validates the exact interpreter, versions and import locations.
    checks = (['gui', 'shared'] if gui_available else ['shared']) if plan['profile'] == 'shared' else [plan['profile']]
    for key in checks:
        run([python, '-I', '-X', 'utf8', ROOT / 'worker_boot.py', '--runtime-check', key])
    if plan['profile'] == 'shared' and gui_available:
        # Imports alone miss ABI-incompatible wheels that consume None references.
        run([python, '-I', '-X', 'utf8', ROOT / 'tools/check_qt_runtime.py'])
    for name, reason in plan.get('unavailable', {}).items():
        print(f'NOT INSTALLED {name}: {reason}', flush=True)
    print('READY', plan['profile'], '(versions and import paths; CUDA has not been run)', flush=True)


def build_correction():
    if not (ROOT / '.venv/bin/python').is_file():
        raise SystemExit('Prepare the shared profile before building motion correction.')
    if not shutil.which('cmake'):
        raise SystemExit('CMake is required to build motion correction; see docs/UBUNTU.md.')
    run([sys.executable, ROOT / 'tools/build_motion_correction.py', '--python', ROOT / '.venv/bin/python'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--profiles', nargs='+', choices=[*FOLDERS, 'all'], default=list(DEFAULT_PROFILES))
    parser.add_argument('--plan', action='store_true', help='Print exact pins without changing anything.')
    parser.add_argument('--machine', help='Plan for another architecture, such as aarch64 or x86_64; with --plan only.')
    parser.add_argument('--uv', type=Path)
    parser.add_argument('--python-patch-fallback', action='store_true',
                        help='Accept and record the newest same-minor Python if uv lacks the exact pin.')
    parser.add_argument('--build-correction', action='store_true', help='Build the native correction module (needs CMake and a C++17 compiler).')
    parser.add_argument('--stage-assets', action='store_true', help='Download hash-checked auxiliary files from installer/external-assets.json; no model weights.')
    args = parser.parse_args(argv)
    if sys.platform != 'linux' and not args.plan:
        raise SystemExit('This setup is for Linux. Windows uses install.cmd.')
    machine = normalize_machine(args.machine)
    if machine not in SUPPORTED_MACHINES:
        raise SystemExit(f'No pinned wheels exist for {machine}. Supported: {", ".join(SUPPORTED_MACHINES)}.')
    if machine != normalize_machine() and not args.plan:
        raise SystemExit(f'--machine {machine} can only be used with --plan on this {normalize_machine()} system.')
    profiles = list(FOLDERS) if 'all' in args.profiles else list(dict.fromkeys(args.profiles))
    if args.plan:
        with tempfile.TemporaryDirectory() as folder:
            lock = prepared_lock(folder)
        print(json.dumps([profile_plan(ROOT, lock, profile, machine) for profile in profiles], indent=2))
        return 0
    work = ROOT / 'build/dependencies'
    lock = prepared_lock(work)
    plans = [profile_plan(ROOT, lock, profile, machine) for profile in profiles]
    uv = find_uv(args.uv)
    for plan in plans:
        install_profile(uv, plan, work, ROOT / 'build/ubuntu', args.python_patch_fallback)
    if args.stage_assets:
        run([sys.executable, ROOT / 'tools/stage_assets.py', '--download'])
    if args.build_correction:
        build_correction()
    print('Environments installed; versions and import paths were validated, but CUDA has not been run.\n'
          'Import model files through Model Manager, then verify with: ./rgbmocap-cli.sh doctor --gpu')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
