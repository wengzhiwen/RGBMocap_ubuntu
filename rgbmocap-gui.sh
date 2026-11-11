#!/usr/bin/env bash
# Linux counterpart of RGBMoCap.cmd. Keep this terminal open: it shows processing messages.
set -u
root="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
python="$root/.venv/bin/python"
if [ ! -x "$python" ]; then
  echo "RGB MoCap is not installed. Run: python3 tools/setup_ubuntu.py" >&2
  exit 1
fi
[ -d "$root/tools/bin" ] && export PATH="$root/tools/bin:$PATH"
[ -d "$root/.runtime/lib" ] && export LD_LIBRARY_PATH="$root/.runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$root" || exit 1
exec "$python" -I -X utf8 -u -B "$root/launch_linux.py" "$@"
