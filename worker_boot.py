"""Start one validated private runtime; never fall back to another environment."""
import json
import runpy
import sys
from pathlib import Path
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root))
from mocap.runtime_registry import activate
if len(sys.argv)>1 and sys.argv[1]=='--runtime-check':
    print(json.dumps(activate(root,sys.argv[2])))
else:
    if len(sys.argv)<2: raise RuntimeError('A worker stage is required.')
    if sys.platform.startswith('linux'):
        from mocap.linux_runtime import profile_for
    else:
        from mocap.runtime_registry import profile_for
    activate(root,profile_for(sys.argv[1],sys.argv[2:]))
    sys.argv=['mocap.worker',*sys.argv[1:]]
    runpy.run_module('mocap.worker_progress',run_name='__main__')
