"""Launch the GUI from the private Linux environment.

Unlike launch_source.py this does not read the Windows managed-install state; it validates
the pinned .venv directly, and every worker revalidates its own environment.
"""
import os
from pathlib import Path
import sys

root=Path(__file__).resolve().parent
sys.path.insert(0,str(root))
sys.dont_write_bytecode=True
os.environ.pop('PYTHONHOME',None)
os.environ.pop('PYTHONPATH',None)
from mocap.linux_runtime import runtime_environment
os.environ.update(runtime_environment(root))
from tools.session_lock import app_lock
with app_lock(root):
    from mocap.runtime_registry import activate
    activate(root,'gui')
    from app import main
    raise SystemExit(main())
