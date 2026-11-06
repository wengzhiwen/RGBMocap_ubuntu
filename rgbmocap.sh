#!/usr/bin/env bash
# Command-line interface over the same workers as the GUI. See: ./rgbmocap.sh --help
set -u
root="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
python="$root/.venv/bin/python"
# doctor must also work before setup, so fall back to any Python 3 (standard library only).
[ -x "$python" ] || python="$(command -v python3)"
[ -d "$root/tools/bin" ] && export PATH="$root/tools/bin:$PATH"
[ -d "$root/.runtime/lib" ] && export LD_LIBRARY_PATH="$root/.runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$python" -I -X utf8 -u -B "$root/tools/ubuntu_cli.py" "$@"
