"""Private optional FAH entry point in its own pinned .venv-fah runtime.

    flow_fah_entry.py <FlowHMR project>                                  the FlowHMR stage
    flow_fah_entry.py --request R --video V --output O --receipt J        the FAB stage
"""
from pathlib import Path
import sys, json
sys.dont_write_bytecode = True
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from mocap.fah_runtime import activate
activate(root)
from mocap.flow_fah import run, run_request
def emit(kind, **value): print('@@FLOW_FAH@@' + json.dumps(dict(type=kind, **value)), flush=True)
if len(sys.argv) == 2:
    run(Path(sys.argv[1]), emit)
elif len(sys.argv) == 9 and sys.argv[1:7:2] == ['--request', '--video', '--output'] and sys.argv[7] == '--receipt':
    run_request(*sys.argv[2:9:2], emit)
else:
    raise ValueError('An explicit FlowHMR project path, or --request, --video, --output and --receipt, is required')
