"""Readiness of the optional FAB estimator on Ubuntu; standard library only.

Imported by the GUI, CLI and the shared worker, so it must never import FAB,
Torch or CUDA and must never read a weight file: readiness only stats the pinned
private environment and the released asset layout that pinned FAB itself looks
for (`fab/paths.py` in the pinned checkout). Hashing happens only where an
explicit call asks for it (`integrity()`), never on import or readiness.

The child interpreter re-validates its own environment with
`validate_interpreter()`, called from `mocap/fab_infer_entry.py`.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path

from .paths import ROOT

PACKAGE = 'fab'
PACKAGE_VERSION = '1.0.0'
COMMIT = 'ec65caf40632b4d36eb48f58ab45ea56d3cba21a'
MODEL = 'fab-l'
VENV = '.venv-fab'
ENTRY = Path('mocap/fab_infer_entry.py')
MINIMUM_PYTHON = (3, 10)
# Small sources hashed into every capture receipt, so a changed adapter or
# child entry invalidates reuse of an older capture.
CODE = ('fab_runtime.py', 'fab_pipeline.py', 'fab_data.py', 'fab_infer_entry.py')
# Release-weight digests from the pinned official manifest (`hf download
# noahcao/FAB`; SHA256SUMS ships in the weights folder), relative to $FAB_WEIGHTS.
WEIGHT_SHA256 = {'fab-l/model.pt': '83531ac15212feb72aec8ba520e3e7bb19493e2263867bde1bf06d828a17fc6d',
                 'fab-l/config.json': '1e55b2088740d6c7a4cee343ab2826b1bc8c86be04521d1011c0f9231b5d9197',
                 'shared/smpl_fitter.pt': '2e5459ecabad5edd0e5ec518679f3f51b66cc69d4cd35af8996cdfa3921134e8'}
NOTICE = 'Prepare FAB with the Ubuntu setup command and download its files in Model Manager; see docs/UBUNTU.md.'


def supported():
    """FAB is an optional Ubuntu-only estimator; every caller guards on this."""
    return sys.platform.startswith('linux')


def env_of(base=None):
    return os.environ if base is None else base


def home(env=None):
    env = env_of(env)
    return Path(env.get('FAB_HOME') or (Path.home() / '.cache' / 'fab')).expanduser()


def weights_dir(env=None):
    env = env_of(env)
    return Path(env.get('FAB_WEIGHTS') or home(env) / 'weights').expanduser()


def sam3d_dir(env=None):
    env = env_of(env)
    return Path(env.get('FAB_SAM3D') or home(env) / 'sam-3d-body-dinov3').expanduser()


def smpl_dir(env=None):
    """The folder holding SMPL_NEUTRAL.pkl and J_regressor_h36m.npy.

    Only `FAB_SMPL_DIR` (a folder that already holds those files, i.e. upstream's
    `.../body_models/smpl`) or FAB's own default `$FAB_HOME/body_models/smpl` is
    used; the app's other SMPL copies are never swapped in, because upstream
    resolves the MHR conversion as `smpl_dir().parent / 'mhr'`.
    See docs/models/FAB_FAH.md.
    """
    env = env_of(env)
    configured = env.get('FAB_SMPL_DIR')
    if configured:
        return Path(configured).expanduser()
    return home(env) / 'body_models' / 'smpl'


def mhr_dir(env=None):
    """Upstream's FAB-L predictor loads the MHR conversion next to its SMPL folder."""
    return smpl_dir(env).parent / 'mhr'


def g1_dir(env=None):
    env = env_of(env)
    return Path(env.get('FAB_G1') or home(env) / 'g1').expanduser()


def assets(env=None):
    """The pinned layout: name -> (path, required for FAB-L reconstruction).

    The released predictor loads the MHR conversion unconditionally because its
    body-model directory exists, and the G1 head needs its URDF, so those assets
    are required as well; see docs/models/FAB_FAH.md.
    """
    weights, sam3d, smpl, mhr, g1 = weights_dir(env), sam3d_dir(env), smpl_dir(env), mhr_dir(env), g1_dir(env)
    return {
        'fab_model': (weights / MODEL / 'model.pt', True),
        'fab_config': (weights / MODEL / 'config.json', True),
        'smpl_fitter': (weights / 'shared' / 'smpl_fitter.pt', True),
        'sam3d_config': (sam3d / 'model_config.yaml', True),
        'mhr_model': (sam3d / 'assets' / 'mhr_model.pt', True),
        'smpl_model': (smpl / 'SMPL_NEUTRAL.pkl', True),
        'smpl_regressor': (smpl / 'J_regressor_h36m.npy', True),
        'mhr_mapping': (mhr / 'mhr2smpl_mapping.npz', True),
        'mhr_face_mask': (mhr / 'mhr_face_mask.ply', True),
        'g1_urdf': (g1 / 'g1_29dof.urdf', True),
    }


def asset_paths(env=None):
    return {name: row[0] for name, row in assets(env).items()}


def asset_stats(env=None):
    """Cheap identity of every asset: [size, mtime_ns], or None when absent."""
    out = {}
    for name, path in asset_paths(env).items():
        try:
            out[name] = file_stat(path)
        except OSError:
            out[name] = None
    return out


def interpreter(root=None):
    # Do not resolve: Python finds pyvenv.cfg from the unresolved venv path.
    return Path(ROOT if root is None else root) / VENV / 'bin' / 'python'


def venv(root=None):
    return Path(ROOT if root is None else root) / VENV


def site_packages(root=None):
    folder = venv(root)
    for name in ('lib', 'lib64'):
        for candidate in sorted((folder / name).glob('python*/site-packages')):
            if candidate.is_dir():
                return candidate
    return None


def python_version(root=None):
    """The interpreter version recorded in pyvenv.cfg, or None when unreadable."""
    try:
        text = (venv(root) / 'pyvenv.cfg').read_text(encoding='utf-8', errors='replace')[:16384]
    except OSError:
        return None
    values = {}
    for line in text.splitlines():
        key, _, value = line.partition('=')
        if key.strip() and value.strip():
            values.setdefault(key.strip().lower(), value.strip())
    raw = values.get('version') or values.get('python') or ''
    parts = raw.split('.')
    if len(parts) < 2 or not all(part.isdigit() for part in parts[:2]):
        return None
    return tuple(int(part) for part in parts[:3] if part.isdigit())


def file_stat(path):
    value = Path(path).stat()
    return [value.st_size, value.st_mtime_ns]


def package(root=None):
    """PEP 610 identity of the installed FAB distribution, read from its metadata only."""
    site = site_packages(root)
    if site is None:
        return None
    for info in sorted((p for p in site.iterdir() if p.is_dir() and p.name.endswith('.dist-info')), key=lambda p: p.name):
        # Strip the suffix first: 'fab-1.0.0.dist-info' must reduce to 'fab', not 'fab-1.0.0.dist'.
        if info.name[:-len('.dist-info')].rsplit('-', 1)[0].lower().replace('_', '-') != PACKAGE:
            continue
        found = dict(name=PACKAGE, dist_info=info.name, version=None, url=None, editable=None, vcs=None, commit=None)
        try:
            with (info / 'METADATA').open('r', encoding='utf-8', errors='replace') as stream:
                for line in stream:
                    if not line.strip():
                        break
                    if line.startswith('Name:'):
                        found['name'] = line[5:].strip().lower().replace('_', '-')
                    elif line.startswith('Version:'):
                        found['version'] = line[8:].strip()
                        break
        except OSError:
            continue
        for name in ('direct_url.json', 'DIRECT_URL'):
            try:
                raw = (info / name).read_text(encoding='utf-8', errors='replace')
            except OSError:
                continue
            if len(raw) > 65536:
                break
            try:
                record = json.loads(raw)
            except ValueError:
                break
            found['url'] = record.get('url')
            found['editable'] = (record.get('dir_info') or {}).get('editable')
            vcs = record.get('vcs_info') or {}
            found['vcs'] = vcs.get('vcs')
            found['commit'] = (vcs.get('commit_id') or '').strip().lower() or None
            break
        if found['name'] == PACKAGE:
            return found
    return None


def package_problems(root=None):
    found = package(root)
    if found is None:
        return [f'the {PACKAGE} package is missing from {venv(root)}']
    if found['version'] != PACKAGE_VERSION:
        return [f'{PACKAGE} {found["version"]} is installed, {PACKAGE_VERSION} is required']
    if not found['commit']:
        return [f'{PACKAGE} was not installed from the pinned source checkout (no VCS commit recorded)']
    commit = found['commit']
    if commit != COMMIT and not (len(commit) < len(COMMIT) and COMMIT.startswith(commit)):
        return [f'{PACKAGE} comes from commit {commit}, commit {COMMIT} is required']
    version = python_version(root)
    if version is not None and version < MINIMUM_PYTHON:
        return [f'{".".join(map(str, version))} is installed; FAB needs Python {".".join(map(str, MINIMUM_PYTHON))}+']
    return []


def missing(root=None, env=None):
    """Ordered, GUI-ready reasons why FAB cannot run; empty when it can."""
    rows = []
    python = interpreter(root)
    if not python.is_file():
        rows.append(dict(key='fab_runtime', name='FAB runtime', path=str(python),
                         hint=f'The private FAB environment is missing. {NOTICE}'))
    for problem in package_problems(root) if python.is_file() else []:
        rows.append(dict(key='fab_package', name='FAB package', path=str(venv(root)),
                         hint=f'{problem}. {NOTICE}'))
    entry = Path(ROOT if root is None else root) / ENTRY
    if not entry.is_file():
        rows.append(dict(key='fab_entry', name='FAB inference entry', path=str(entry),
                         hint='This installation has no FAB inference entry point. Update the app.'))
    for name, (path, required) in assets(env).items():
        if required and not path.is_file():
            rows.append(dict(key=name, name=path.name, path=str(path),
                             hint=f'Prepare this FAB file first (see the FAB guide, Model Manager or Setup). Required for {MODEL}.'))
    return rows


def ready(root=None, env=None):
    return not missing(root, env)


def report(root=None, env=None):
    """Everything the GUI/CLI shows for FAB, without importing or hashing anything."""
    found = package(root)
    return dict(available=supported(), platform=sys.platform, model=MODEL, notice=(NOTICE if supported() else
                'FAB is only available on Ubuntu/Linux.'), runtime=str(venv(root)),
                python=str(interpreter(root)), python_exists=interpreter(root).is_file(),
                python_version=python_version(root), entry=str(Path(ROOT if root is None else root) / ENTRY),
                entry_exists=(Path(ROOT if root is None else root) / ENTRY).is_file(),
                package=found, pinned_package=dict(name=PACKAGE, version=PACKAGE_VERSION, commit=COMMIT),
                home=str(home(env)), weights=str(weights_dir(env)), sam3d=str(sam3d_dir(env)),
                smpl=str(smpl_dir(env)), mhr=str(mhr_dir(env)), g1=str(g1_dir(env)),
                assets={name: dict(path=str(path), required=required, present=path.is_file(),
                                   **({'size': path.stat().st_size} if path.is_file() else {}))
                        for name, (path, required) in assets(env).items()},
                blocked=[row['hint'] for row in missing(root, env)])


def identity(root=None, env=None):
    """Receipt identity of the optional estimator: package, code and asset stamps."""
    from .storage import sha256
    found = package(root)
    code = {}
    for name in CODE:
        path = Path(ROOT if root is None else root) / 'mocap' / name
        code[name] = sha256(path) if path.is_file() else None
    return dict(model=MODEL, python=python_version(root),
                package=(None if found is None else dict(name=found['name'], version=found['version'],
                                                         commit=found['commit'], dist_info=found['dist_info'])),
                pinned=dict(version=PACKAGE_VERSION, commit=COMMIT), code=code, assets=asset_stats(env))


def integrity(weights=None, env=None):
    """Verify the release weights before the child loads them.

    Hashes gigabyte-scale files: call it immediately before inference or from an
    explicit check, never from import, GUI painting or readiness.
    """
    from .storage import sha256
    base = Path(weights).expanduser() if weights is not None else weights_dir(env)
    out = {}
    for name, expected in WEIGHT_SHA256.items():
        path = base / name
        try:
            out[name] = sha256(path)
        except OSError:
            out[name] = None
    return out


def integrity_ok(weights=None, env=None):
    measured = integrity(weights, env)
    return all(measured.get(name) == digest for name, digest in WEIGHT_SHA256.items())


def command(project, output, root=None):
    """One isolated child in the private FAB environment; nothing is installed here."""
    python = interpreter(root)
    if not python.is_file():
        raise RuntimeError(f'The FAB runtime is missing ({python}). {NOTICE}')
    entry = Path(ROOT if root is None else root) / ENTRY
    if not entry.is_file():
        raise RuntimeError(f'This installation has no FAB inference entry point ({entry}). Update the app.')
    # Isolated mode ignores PYTHONPATH/PYTHONUTF8, so -X utf8 and -u stay explicit.
    return [str(python), '-I', '-X', 'utf8', '-u', str(entry), '--project', str(Path(project)),
            '--output', str(Path(output))]


def child_environment(root=None, base=None):
    """Private cache/data locations plus the exact FAB asset layout we validated."""
    base = env_of(base)
    env = dict(base)
    if supported():
        from .linux_runtime import runtime_environment
        env.update(runtime_environment(Path(ROOT if root is None else root), base=base))
    env.update(FAB_HOME=str(home(env)), FAB_WEIGHTS=str(weights_dir(env)), FAB_SAM3D=str(sam3d_dir(env)),
               FAB_SMPL_DIR=str(smpl_dir(env)), FAB_MHR=str(mhr_dir(env)), FAB_G1=str(g1_dir(env)))
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    return env


def validate_interpreter(root=None):
    """Child-side check of the pinned environment; `fab_infer_entry.py` calls this."""
    folder = venv(root)
    if sys.prefix == sys.base_prefix or Path(sys.prefix).resolve() != folder.resolve():
        raise RuntimeError(f'FAB inference must run in {folder}; this interpreter is {sys.prefix}. {NOTICE}')
    if sys.version_info[:2] < MINIMUM_PYTHON:
        raise RuntimeError(f'FAB needs Python {".".join(map(str, MINIMUM_PYTHON))}+; '
                           f'this runtime is {sys.version.split()[0]}. {NOTICE}')
    import importlib.metadata as metadata
    try:
        version = metadata.version(PACKAGE)
        record = metadata.distribution(PACKAGE).read_text('direct_url.json')
    except metadata.PackageNotFoundError:
        record, version = None, None
    if version != PACKAGE_VERSION:
        raise RuntimeError(f'{PACKAGE} {version} is installed, {PACKAGE_VERSION} is required. {NOTICE}')
    commit = None
    if record:
        try:
            commit = ((json.loads(record).get('vcs_info') or {}).get('commit_id') or '').strip().lower() or None
        except ValueError:
            commit = None
    if commit is None:
        raise RuntimeError(f'{PACKAGE} was not installed from the pinned source checkout. {NOTICE}')
    if commit != COMMIT and not (len(commit) < len(COMMIT) and COMMIT.startswith(commit)):
        raise RuntimeError(f'{PACKAGE} comes from commit {commit}, commit {COMMIT} is required. {NOTICE}')
    return dict(python=sys.executable, prefix=sys.prefix, version=version, commit=commit, model=MODEL)
