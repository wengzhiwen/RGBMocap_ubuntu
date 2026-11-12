#!/usr/bin/env python3
"""Build the arm64 online installer from public, Git-tracked source only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
EXTRA = ('tools/build_deb.py', 'packaging/debian/launcher.py', 'docs/DEB.md', 'tools/tests/test_deb.py')
PRIVATE = {'models', 'projects', 'cache', 'character_scenes', 'config', 'runtime', 'runtime_character',
           '.git', '.agents', '.codex', '.setup', '.bootstrap', '.runtime', '.cache', 'experiments', 'artifacts'}
FORBIDDEN = {'.pt', '.pth', '.ckpt', '.safetensors', '.onnx', '.npz', '.npy', '.pkl', '.mp4', '.avi', '.mov',
             '.mkv', '.webm', '.fbx', '.bvh', '.pyd', '.so', '.exe', '.dll', '.pyc', '.pem', '.key'}
DEPENDS = ('python3 (>= 3.10), python3-venv, ca-certificates, ffmpeg, cmake, build-essential, git, '
           'libglu1-mesa, libegl1, libgl1, libopengl0, libxcb-cursor0, libxkbcommon-x11-0, '
           'libxcb-icccm4, libxcb-image0, libxcb-keysyms1, libxcb-render-util0, libxcb-xinerama0, '
           'libxcb-xkb1, libsm6, libice6, libxrender1, libxi6, libxext6, libx11-6, libfontconfig1, '
           'libfreetype6, libdbus-1-3, libgomp1, libglib2.0-0')


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def selected_sources():
    resources = json.loads((ROOT / 'installer/source-resources.json').read_text(encoding='utf-8'))['files']
    allowed = {row['path']: row for row in resources}
    tracked = {os.fsdecode(name) for name in git('ls-files', '-z').split(b'\0') if name}
    paths = []
    for name in sorted(tracked | set(EXTRA)):
        path = Path(name)
        if path.parts[0] in PRIVATE or path.parts[0].startswith('.venv') or path.parts[0] in ('prebuilt', 'releases', 'build', 'dist'):
            continue
        if path.suffix.lower() in FORBIDDEN or path.name.startswith('.env') or path.name in ('AGENTS.md', 'CLAUDE.md', 'GEMINI.md', 'CODEX.md'):
            continue
        if path.suffix.lower() == '.blend' and name not in allowed:
            continue
        source = ROOT / path
        if source.is_symlink() or not source.is_file():
            raise ValueError('Source is missing, a symlink, or a submodule: ' + name)
        if name in allowed:
            row = allowed[name]
            if source.stat().st_size != row['bytes'] or hashlib.sha256(source.read_bytes()).hexdigest() != row['sha256']:
                raise ValueError('Source resource hash mismatch: ' + name)
        paths.append(path)
    return paths


def payload(target, paths, epoch):
    with tarfile.open(target, 'w:xz') as archive:
        for path in paths:
            source = ROOT / path
            info = tarfile.TarInfo(path.as_posix())
            info.size = source.stat().st_size
            info.mtime = epoch
            info.mode = 0o755 if os.access(source, os.X_OK) else 0o644
            with source.open('rb') as stream:
                archive.addfile(info, stream)


def write(root, name, value, mode=0o644):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding='utf-8')
    path.chmod(mode)


def build(output, version=None):
    if not shutil.which('dpkg-deb'):
        raise RuntimeError('dpkg-deb is required.')
    commit = git('rev-parse', 'HEAD').decode().strip()
    epoch = int(os.environ.get('SOURCE_DATE_EPOCH') or git('show', '-s', '--format=%ct', 'HEAD'))
    app_version = re.search(r'__version__ = "([\d.]+)"', (ROOT / 'mocap/__init__.py').read_text()).group(1)
    version = version or f'{app_version}+git{commit[:8]}-1'
    if not re.fullmatch(r'[0-9][A-Za-z0-9.+~:-]*', version):
        raise ValueError('Invalid Debian package version')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    target = output / f'rgbmocap_{version}_arm64.deb'
    if target.exists():
        raise FileExistsError(f'Output already exists: {target}; select a new version/output folder.')
    paths = selected_sources()
    with tempfile.TemporaryDirectory(prefix='rgbmocap-deb-') as folder:
        stage = Path(folder)
        package = stage / 'usr/share/rgbmocap'
        package.mkdir(parents=True)
        payload(package / 'source.tar.xz', paths, epoch)
        metadata = dict(version=version, architecture='arm64', source_commit=commit,
                        source_worktree_modified=bool(git('status', '--porcelain', '--untracked-files=no').strip()),
                        payload_sha256=hashlib.sha256((package / 'source.tar.xz').read_bytes()).hexdigest(),
                        source_files=len(paths), runtime_setup='online', models_bundled=False)
        write(stage, 'usr/share/rgbmocap/package.json', json.dumps(metadata, indent=2) + '\n')
        shutil.copyfile(ROOT / 'packaging/debian/launcher.py', package / 'launcher.py')
        write(stage, 'usr/bin/rgbmocap', '#!/bin/sh\nexec /usr/bin/python3 -I -X utf8 /usr/share/rgbmocap/launcher.py "$@"\n', 0o755)
        write(stage, 'usr/share/applications/rgbmocap.desktop',
              '[Desktop Entry]\nType=Application\nName=RGB MoCap\nComment=Video motion capture with online runtime setup\n'
              'Exec=rgbmocap\nIcon=rgbmocap\nTerminal=true\nCategories=Graphics;3DGraphics;\nStartupNotify=false\n')
        icon = stage / 'usr/share/pixmaps/rgbmocap.png'
        icon.parent.mkdir(parents=True)
        shutil.copyfile(ROOT / 'assets/rgbmocap-icon.png', icon)
        for name in ('LICENSE', 'THIRD_PARTY_NOTICES.md', 'docs/DEB.md'):
            write(stage, 'usr/share/doc/rgbmocap/' + ('README.Debian' if name == 'docs/DEB.md' else name), (ROOT / name).read_text(encoding='utf-8'))
        write(stage, 'usr/share/doc/rgbmocap/copyright',
              'RGB MoCap original code: MIT; see LICENSE.\nThird-party source retains its own licenses, '
              'included in source.tar.xz; see THIRD_PARTY_NOTICES.md.\n')
        control = (f'Package: rgbmocap\nVersion: {version}\nArchitecture: arm64\nSection: graphics\nPriority: optional\n'
                   'Maintainer: RGB MoCap contributors <noreply@github.com>\n'
                   f'Depends: {DEPENDS}\nHomepage: https://github.com/wengzhiwen/RGBMocap_ubuntu\n'
                   'Description: RGB video motion capture (online runtime installer)\n'
                   ' First launch downloads pinned independent Python and AI environments as\n'
                   ' the desktop user. Requires a compatible NVIDIA GPU and driver for inference.\n'
                   ' Model weights and Blender are managed separately.\n')
        size = sum(p.stat().st_size for p in stage.rglob('*') if p.is_file())
        write(stage, 'DEBIAN/control', control + f'Installed-Size: {(size + 1023) // 1024}\n')
        sums = ''.join(hashlib.md5(p.read_bytes()).hexdigest() + '  ' + p.relative_to(stage).as_posix() + '\n'
                       for p in sorted(stage.rglob('*')) if p.is_file() and 'DEBIAN' not in p.relative_to(stage).parts)
        write(stage, 'DEBIAN/md5sums', sums)
        for path in sorted(stage.rglob('*'), reverse=True):
            os.utime(path, (epoch, epoch))
        env = dict(os.environ, SOURCE_DATE_EPOCH=str(epoch))
        subprocess.run(['dpkg-deb', '--root-owner-group', '-Zxz', '-z6', '--build', stage, target], check=True, env=env)
    checksum = hashlib.sha256(target.read_bytes()).hexdigest()
    target.with_suffix('.deb.sha256').write_text(f'{checksum}  {target.name}\n', encoding='utf-8')
    print(json.dumps(dict(path=str(target), bytes=target.stat().st_size, sha256=checksum, **metadata), indent=2))
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'dist')
    parser.add_argument('--version')
    args = parser.parse_args()
    build(args.output_dir, args.version)
