"""Optional FAH runtime readiness, layout and identity stamps.

Standard library only: this module is safe in the GUI, the CLI and any worker, and
it never imports FAH, Torch or NumPy. Nothing here installs, downloads or writes a
package, environment, weight or cache.

Only the optional FAH pass runs in its own separately provisioned pinned
environment, `<ROOT>/.venv-fah`; the FAB pass is a separate environment with its
own recipe and is never provisioned or activated here. FAH keeps upstream's own
asset layout so the user-provided gated files are found exactly where the
upstream release expects them:

    $FAH_HOME (default ~/.cache/fah)
      weights/{fah-s,fah-b,fah,fah-h}/{model.pt,config.json}
      weights/shared/detector.pt      (the learned mesh fitter is not used by this adapter)
      sam-3d-body-dinov3/{model_config.yaml,assets/mhr_model.pt}
      mano/MANO_RIGHT.pkl                 (or $MANO_DIR)

Only `activate()` runs inside the FAH environment itself; it validates the
interpreter and the pinned package revision before any model loads.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT = 'noahcao/FAH'
PROJECT_URL = 'https://github.com/' + PROJECT
# The exact upstream release this adapter was written and checked against.
REVISION = 'da03aabcd99cf61c7ea179c1ddca84f75be14962'
PACKAGE = 'fah'
PACKAGE_VERSION = '1.0.0'
MIN_PYTHON = (3, 10)
VENV = '.venv-fah'
# Written by the user-provisioned pinned-venv recipe inside the environment.
VENV_RECEIPT = 'rgbmocap-fah.json'
# A deliberately registered matching commit id, for an install that cannot record PEP 610.
SOURCE_REGISTRATION = '.setup/fah-source.json'
MODEL_CHOICES = ('fah-s', 'fah-b', 'fah', 'fah-h')
DEFAULT_MODEL = 'fah'
# The modules that define this optional pass; their bytes belong in run identities.
CODE_FILES = ('mocap/fah_runtime.py', 'mocap/flow_fah.py', 'mocap/flow_hands.py', 'mocap/flow_fah_entry.py')
# The pinned upstream modules this adapter actually calls, when they are readable.
UPSTREAM_FILES = ('fah/api.py', 'fah/estimator.py', 'fah/fast_mano.py', 'fah/mano_procrustes.py')
TRACKED_PACKAGES = (PACKAGE, 'torch', 'torchvision', 'numpy', 'scipy', 'smplx', 'timm')
HINT = ('Prepare the separate pinned FAH environment and its own gated files; '
        'see Model Manager for this app\'s FAH row and docs/UBUNTU.md.')


def root_path(root=None) -> Path:
    if root is not None:
        return Path(root)
    from .paths import ROOT
    return ROOT


def supported() -> bool:
    """The Linux route only; the Windows distribution keeps its existing flow."""
    return sys.platform.startswith('linux')


def model(env=None) -> str:
    value = str((os.environ if env is None else env).get('RGB_MOCAP_FAH_MODEL') or DEFAULT_MODEL)
    if value not in MODEL_CHOICES:
        raise ValueError('Unsupported FAH model: ' + value + ' (choose one of ' + ', '.join(MODEL_CHOICES) + ')')
    return value


def venv(root=None) -> Path:
    return root_path(root) / VENV


def interpreter(root=None) -> Path:
    folder = venv(root)
    return folder / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')


def site_packages(root=None) -> list[Path]:
    folder = venv(root)
    found = [folder / 'Lib' / 'site-packages']
    if (folder / 'lib').is_dir():
        found += sorted((folder / 'lib').glob('python*/site-packages'))
    return [path for path in found if path.is_dir()]


def asset_roots(env=None, root=None) -> dict:
    """FAH's own environment settings, resolved exactly as the pinned release does."""
    env = os.environ if env is None else env
    expand = lambda name: Path(os.path.expanduser(name)) if name else None
    home = expand(env.get('FAH_HOME') or os.path.join(os.path.expanduser('~'), '.cache', 'fah'))
    weights = expand(env.get('FAH_WEIGHTS')) or home / 'weights'
    sam3d = expand(env.get('FAH_SAM3D')) or home / 'sam-3d-body-dinov3'
    mano = expand(env.get('MANO_DIR'))
    if mano is None:
        mano = home / 'mano'
    return dict(home=home, weights=weights, sam3d=sam3d, mano=mano)


def configure_environment(root=None, env=None) -> dict:
    """The values a FAH child process must carry. Set them explicitly instead of
    inheriting a desktop shell that may point somewhere else."""
    paths = asset_roots(env, root)
    return dict(FAH_HOME=str(paths['home']), FAH_WEIGHTS=str(paths['weights']),
                FAH_SAM3D=str(paths['sam3d']), MANO_DIR=str(paths['mano']))


def upstream_names(weights: Path, name: str) -> dict:
    """The released config.json names the pose and detector files used by this adapter."""
    names = dict(model='model.pt', detector='../shared/detector.pt')
    config = weights / name / 'config.json'
    if config.is_file():
        try:
            value = json.loads(config.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return names
        for key in names:
            found = value.get(key)
            if isinstance(found, str) and found and not os.path.isabs(found):
                names[key] = found
    return names


def asset_paths(root=None, env=None) -> dict:
    paths = asset_roots(env, root)
    name = model(env)
    weights = paths['weights'] / name
    names = upstream_names(paths['weights'], name)
    here = lambda key: (weights / names[key]).resolve() if key in names else None
    # Upstream requires the detector during init; the learned mesh fitter is not
    # part of this adapter's FAH dense geometry plus MANO Gauss-Newton path.
    required = {'weights_config': weights / 'config.json', 'weights_model': here('model'),
                'detector': here('detector'),
                'sam_config': paths['sam3d'] / 'model_config.yaml',
                'sam_mhr': paths['sam3d'] / 'assets' / 'mhr_model.pt',
                'mano': paths['mano'] / 'MANO_RIGHT.pkl'}
    optional = {}
    return dict(required=required, optional=optional, model=name, roots=paths)


def installed(root=None) -> dict | None:
    """The installed FAH distribution and its PEP 610 source commit, if recorded."""
    for site in site_packages(root):
        for info in sorted(site.glob(PACKAGE + '-*.dist-info')):
            stem = info.name[: -len('.dist-info')] if info.name.endswith('.dist-info') else info.name
            version = stem[len(PACKAGE) + 1:] if stem.startswith(PACKAGE + '-') else ''
            # The site path is used only to hash readable upstream files; it is never stored in an identity.
            record = dict(version=version, commit=None, url=None, distribution=info.name, site=str(site))
            direct = info / 'direct_url.json'
            if direct.is_file():
                try:
                    value = json.loads(direct.read_text(encoding='utf-8'))
                except (OSError, ValueError):
                    return record
                record['url'] = str(value.get('url') or '')
                vcs = value.get('vcs_info') or {}
                record['commit'] = str(vcs.get('commit_id') or value.get('vcs_commit_id') or '') or None
            return record
    return None


def registered_commit(root=None) -> str | None:
    """A matching commit id the user recorded deliberately, e.g. for a copied source tree."""
    try:
        value = json.loads((root_path(root) / SOURCE_REGISTRATION).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    if not isinstance(value, dict) or str(value.get('project') or PROJECT) != PROJECT:
        return None
    if str(value.get('package') or PACKAGE_VERSION) != PACKAGE_VERSION:
        return None
    return str(value.get('commit') or '') or None


def source_match(root=None) -> dict:
    found = installed(root)
    if found is None:
        return dict(ok=False, detail=PACKAGE + ' is not installed in ' + VENV, installed=None,
                    commit=None, registered=registered_commit(root))
    commit = found['commit']
    registered = registered_commit(root)
    if found['version'] != PACKAGE_VERSION:
        return dict(ok=False, detail=f'{PACKAGE} {found["version"]} differs from the supported {PACKAGE_VERSION}',
                    installed=found, commit=commit, registered=registered)
    if commit == REVISION:
        return dict(ok=True, detail=f'{PACKAGE} {found["version"]} at the pinned revision', installed=found,
                    commit=commit, registered=registered)
    # A hand-written registration only speaks for an install that recorded no
    # PEP 610 commit of its own; a conflicting direct_url commit always wins.
    if registered == REVISION and not commit:
        return dict(ok=True, detail=f'{PACKAGE} {found["version"]}; revision registered in ' + SOURCE_REGISTRATION,
                    installed=found, commit=registered, registered=registered)
    return dict(ok=False, detail=f'{PACKAGE} source commit {commit or registered or "unknown"} does not match the pinned {REVISION}',
                installed=found, commit=commit or registered, registered=registered)


def venv_python_version(root=None) -> str:
    try:
        text = (venv(root) / 'pyvenv.cfg').read_text(encoding='utf-8')
    except OSError:
        return ''
    for line in text.splitlines():
        key, _, value = line.partition('=')
        if key.strip().lower() in ('python', 'version'):
            return value.strip()
    return ''


def runtime_stamp(root=None) -> dict:
    """Package versions actually installed in the FAH environment, read without importing them."""
    found = installed(root)
    packages = {}
    for site in site_packages(root):
        for info in sorted(site.glob('*.dist-info')):
            name, _, version = info.name[:-(len('.dist-info'))].partition('-')
            if name.lower().replace('_', '-') in TRACKED_PACKAGES:
                packages.setdefault(name.lower().replace('_', '-'), version)
    return dict(environment=VENV, python=venv_python_version(root), packages=packages,
                package=(found or {}).get('version'), commit=(found or {}).get('commit'))


def code_identity(root=None) -> dict:
    from .resource_cache import code_hash
    root = root_path(root)
    return {name: code_hash(root / name) for name in CODE_FILES}


def upstream_identity(root=None, env=None) -> dict:
    """Digests of the readable pinned upstream modules this adapter calls; absent
    files are reported as such instead of guessed."""
    from .storage import sha256
    found = installed(root)
    if found is None:
        return {}
    roots = [Path(found['site'])] if found.get('site') else []
    out = {}
    for name in UPSTREAM_FILES:
        for base in roots:
            path = base / name
            if path.is_file():
                out[name] = sha256(path)
                break
    return out


def asset_identity(root=None, env=None) -> dict:
    from .result_artifacts import file_stat
    paths = asset_paths(root, env)
    out = {}
    for key, path in paths['required'].items():
        out[key] = file_stat(path) if path is not None and path.is_file() else None
    return out


def identity(root=None, env=None) -> dict:
    """Scoped source, package/runtime and asset stamps for one optional FAH run.

    Never raises: an unready runtime records why, so a saved generation keeps its
    own identity instead of silently reusing another run's prediction.
    """
    root = root_path(root)
    match = source_match(root)
    try:
        name = model(env)
    except ValueError as error:
        return dict(version=1, unavailable=[str(error)])
    rows = readiness(root, env)
    absent = [row['name'] for row in rows if row['required'] and not row['ok']]
    return dict(version=1, supported=supported(), model=name,
                source=dict(project=PROJECT, revision=REVISION, commit=match['commit'], url=PROJECT_URL),
                package=runtime_stamp(root), code=code_identity(root),
                upstream=upstream_identity(root, env), assets=asset_identity(root, env),
                unavailable=absent or None)


def readiness(root=None, env=None) -> list:
    """Bounded, read-only checks in dependency order. Each row names its own fix."""
    root = root_path(root)
    rows = []
    add = lambda name, ok, detail, required=True: rows.append(dict(name=name, ok=bool(ok), detail=str(detail),
                                                                   required=bool(required)))
    if not supported():
        add('FAH runtime', False, 'FAH refinement is available on the Linux route only; Windows keeps HaMeR.', False)
        return rows
    python = interpreter(root)
    add('FAH environment', python.is_file(), str(python) if python.is_file() else f'{VENV} is missing. {HINT}')
    match = source_match(root)
    add('FAH package', match['ok'], match['detail'])
    paths = asset_paths(root, env)
    labels = dict(weights_config='FAH model configuration', weights_model='FAH model weights',
                  detector='FAH detector package (loaded, never used for detection)',
                  sam_config='SAM 3D Body configuration', sam_mhr='SAM 3D Body MHR geometry',
                  mano='MANO right-hand model')
    for key, label in labels.items():
        path = paths['required'][key]
        add(label, path is not None and path.is_file(),
            path if path is not None and path.is_file() else f'{path} is missing. {HINT}')
    for key, path in paths['optional'].items():
        add('FAH ' + key.replace('_', ' ') + ' (optional)', path is not None and path.is_file(), str(path), False)
    return rows


def missing(root=None, env=None) -> list:
    return [row['name'] for row in readiness(root, env) if row['required'] and not row['ok']]


def report(root=None, env=None) -> str:
    return '\n'.join(f"{row['name']}: {'ready' if row['ok'] else row['detail']}" for row in readiness(root, env))


def activate(root=None) -> dict:
    """Run only inside `.venv-fah`. Validates the interpreter and pinned package."""
    root = root_path(root)
    folder = venv(root)
    if sys.prefix == sys.base_prefix or Path(sys.prefix).resolve() != folder.resolve():
        raise RuntimeError(f'FAH must run in {folder.name}; this interpreter is {sys.prefix}. {HINT}')
    if tuple(sys.version_info[:2]) < MIN_PYTHON:
        raise RuntimeError(f'{VENV} uses Python {sys.version.split()[0]}; FAH needs {".".join(map(str, MIN_PYTHON))} or newer. {HINT}')
    match = source_match(root)
    if not match['ok']:
        raise RuntimeError('FAH package check failed: ' + match['detail'] + f' The supported revision is {REVISION}.')
    if not site_packages(root):
        raise RuntimeError(f'{VENV} has no site-packages folder. {HINT}')
    return dict(profile='fah', python=sys.executable, environment=str(folder), package=match['installed']['version'],
                commit=REVISION, source=PROJECT_URL, python_version='.'.join(map(str, sys.version_info[:3])))
