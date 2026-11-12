#!/usr/bin/python3
"""Online deb bootstrap. Runs only as the desktop user, never from dpkg scripts."""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile

PACKAGE = Path(__file__).resolve().parent


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


@contextmanager
def install_lock(base):
    base.mkdir(parents=True, exist_ok=True)
    with (base / '.installer.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another RGB MoCap installation is running.') from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def install_source(package, base, metadata):
    """Extract only this package's source into a fresh, versioned user folder."""
    root = base / 'installations' / metadata['payload_sha256'][:20]
    receipt = root / '.deb-source.json'
    if root.exists():
        if (root.is_symlink() or not receipt.is_file()
                or json.loads(receipt.read_text()).get('payload_sha256') != metadata['payload_sha256']):
            raise RuntimeError(f'Existing folder is not owned by this package: {root}')
        return root
    payload = package / 'source.tar.xz'
    if digest(payload) != metadata['payload_sha256']:
        raise RuntimeError('The installed source archive failed its checksum.')
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.source-', dir=root.parent) as folder:
        stage = Path(folder) / 'app'
        stage.mkdir()
        with tarfile.open(payload, 'r:xz') as archive:
            for entry in archive:
                relative = PurePosixPath(entry.name)
                if relative.is_absolute() or '..' in relative.parts or not entry.isfile():
                    raise RuntimeError('Unsafe member in source archive: ' + entry.name)
                target = stage.joinpath(*relative.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(entry) as source, target.open('xb') as output:
                    shutil.copyfileobj(source, output)
                target.chmod(0o755 if entry.mode & 0o111 else 0o644)
        (stage / '.deb-source.json').write_text(json.dumps(metadata), encoding='utf-8')
        stage.rename(root)
    return root


def run(command, root):
    print('+', ' '.join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=root, check=True)


def setup(root, profiles):
    print('Preparing independent pinned runtimes. Network access and several GB of free space are required.\n'
          'Models remain separate; import them through Model Manager.', flush=True)
    uv = root / '.bootstrap/bin/uv'
    if not uv.is_file():
        run(['/usr/bin/python3', '-I', '-m', 'venv', root / '.bootstrap'], root)
        run([root / '.bootstrap/bin/python', '-I', '-m', 'pip', '--isolated', 'install', 'uv==0.13.1'], root)
    # Reuse the Linux installer and its architecture pins, profile isolation and lock.
    run(['/usr/bin/python3', '-I', '-X', 'utf8', root / 'tools/setup_ubuntu.py',
         '--uv', uv, '--profiles', *profiles, '--stage-assets', '--build-correction'], root)
    (root / '.deb-runtime-ready').write_text('Setup completed; CUDA is not yet verified.\n', encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description='RGB MoCap for arm64 Ubuntu (online runtime setup).')
    parser.add_argument('--setup', action='store_true', help='Install/repair runtimes without opening the GUI.')
    parser.add_argument('--profiles', nargs='+', choices=['shared', 'flow', 'exmr', 'hand', 'htd', 'all'],
                        default=['shared', 'flow', 'exmr'], help='Profiles for explicit --setup; shared is always included.')
    parser.add_argument('--doctor', action='store_true', help='Run the CLI readiness checker; add --gpu for CUDA checks.')
    args, remaining = parser.parse_known_args(argv)
    if os.geteuid() == 0:
        raise RuntimeError('Run rgbmocap as your desktop user, without sudo.')
    if platform.machine().lower() not in ('aarch64', 'arm64'):
        raise RuntimeError('This package requires arm64 / aarch64 Ubuntu.')
    base = Path(os.environ.get('XDG_DATA_HOME') or Path.home() / '.local/share') / 'rgbmocap'
    if not base.is_absolute():
        raise RuntimeError('XDG_DATA_HOME must be an absolute path.')
    metadata = json.loads((PACKAGE / 'package.json').read_text(encoding='utf-8'))
    with install_lock(base):
        root = install_source(PACKAGE, base, metadata)
        profiles = list(dict.fromkeys(['shared', *args.profiles]))
        if args.setup or not (root / '.deb-runtime-ready').is_file():
            setup(root, profiles)
    if args.setup:
        print(f'Setup complete. Start with rgbmocap; check CUDA with rgbmocap --doctor --gpu.\nApplication: {root}')
        return 0
    # Keep models, projects and caches stable across package versions. Never migrate
    # the user's checkout, existing registrations or external projects automatically.
    env = os.environ.copy()
    env.setdefault('RGB_MOCAP_DATA', str(base / 'data'))
    Path(env['RGB_MOCAP_DATA']).mkdir(parents=True, exist_ok=True)
    script = root / ('rgbmocap.sh' if args.doctor else 'RGBMoCap.sh')
    command = [str(script), *(['doctor'] if args.doctor else []), *remaining]
    os.execve(script, command, env)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        print(f'RGB MoCap could not start: {error}\nRetry setup with: rgbmocap --setup', file=sys.stderr)
        raise SystemExit(1)
