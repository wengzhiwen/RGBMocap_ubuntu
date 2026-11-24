"""Private optional FlowHMR FAH entry point in its own pinned .venv-fah runtime."""
from pathlib import Path
import sys, json
sys.dont_write_bytecode = True
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from mocap.fah_runtime import activate
activate(root)
from mocap.flow_fah import run
def emit(kind, **value): print('@@FLOW_FAH@@' + json.dumps(dict(type=kind, **value)), flush=True)
if len(sys.argv) != 2:
    raise ValueError('An explicit FlowHMR project path is required')
run(Path(sys.argv[1]), emit)
