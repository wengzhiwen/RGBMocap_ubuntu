"""Launch the GUI from the private Linux environment.

Unlike launch_source.py this does not read the Windows managed-install state; it validates
the pinned .venv directly, and every worker revalidates its own environment.
"""
import os
from contextlib import ExitStack
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parent
sys.path.insert(0,str(root))
sys.dont_write_bytecode=True
os.environ.pop('PYTHONHOME',None)
os.environ.pop('PYTHONPATH',None)
from mocap.linux_runtime import runtime_environment
os.environ.update(runtime_environment(root))
from tools.session_lock import app_lock


def main():
    repair = '--repair' in sys.argv[1:]
    if repair:
        sys.argv.remove('--repair')
    from tools.linux_startup import prepare_gui
    # Hold the same lock during preflight/repair and the entire GUI session.
    with ExitStack() as stack:
        try:
            stack.enter_context(app_lock(root))
            prepare_gui(root, repair=repair)
        except (RuntimeError, OSError, subprocess.SubprocessError) as error:
            print(f'RGB MoCap startup check failed:\n{error}', file=sys.stderr)
            return 1
        from app import main as gui_main
        return gui_main()


if __name__ == '__main__':
    raise SystemExit(main())
