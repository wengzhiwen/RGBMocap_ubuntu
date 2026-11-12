"""Read-only GUI preflight and explicitly requested Qt repair for Linux."""
import importlib.metadata as metadata
import importlib.util
from pathlib import Path
import subprocess

from mocap.linux_runtime import activate, site_packages
from .setup_ubuntu import gui_requirements


def check_gui(root):
    """Check every Qt distribution, its origin, and actual painting in a child."""
    root = Path(root)
    activate(root, 'gui')
    site = site_packages(root / '.venv').resolve()
    for pin in gui_requirements(root):
        name, version = pin.split('==')
        try:
            found = metadata.version(name)
        except metadata.PackageNotFoundError:
            raise RuntimeError(f'{name} is missing; expected {version}.') from None
        module_name = 'PySide6' if name.lower().startswith('pyside6') else name
        module = importlib.util.find_spec(module_name)
        if found != version or module is None or not module.origin:
            raise RuntimeError(f'Incompatible {name} {found}; expected {version}.')
        if not Path(module.origin).resolve().is_relative_to(site):
            raise RuntimeError(f'{name} is imported from outside {site}.')
    try:
        result = subprocess.run([str(root / '.venv/bin/python'), '-I', '-X', 'utf8',
                                 str(root / 'tools/check_qt_runtime.py')],
                                capture_output=True, text=True, encoding='utf-8', timeout=15)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f'Qt painting check could not complete: {error}') from None
    if result.returncode:
        detail = (result.stdout + result.stderr).strip()[-2000:]
        raise RuntimeError(f'Qt painting check failed (exit {result.returncode}): {detail}')


def prepare_gui(root, repair=False):
    """Caller holds app_lock throughout preflight, repair and the GUI session."""
    try:
        activate(root, 'shared')
    except RuntimeError as error:
        raise RuntimeError(f'{error}\nRepair the shared environment with:\n'
                           '  python3 tools/setup_ubuntu.py --profiles shared') from None
    try:
        check_gui(root)
    except RuntimeError as error:
        if not repair:
            raise RuntimeError(f'{error}\nRepair GUI dependencies and start again with:\n'
                               '  ./RGBMoCap.sh --repair\n'
                               'Qt repair may need network access; models and projects are preserved.') from None
        print(f'GUI preflight failed: {error}\nRepairing pinned Qt packages...', flush=True)
        from .setup_ubuntu import find_uv, repair_gui
        repair_gui(find_uv())
        check_gui(root)
