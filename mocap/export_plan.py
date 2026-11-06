"""Check a source export in the shared runtime, with the guards of the GUI's Export action.

Printed for tools/ubuntu_cli.py, which imports no AI packages. Export workers repeat these
checks against the source and scene recorded here, so this is never the only guard.
"""
import json
from pathlib import Path


def plan(project,kind):
    from .motion_tools import RESULTS,require_supported_project,selected_export
    from .people_store import load_people,root_for
    project=Path(project).resolve()
    if kind not in RESULTS:raise ValueError('Unknown result: '+str(kind))
    model=require_supported_project(project).key
    # selected_export checks source, correction, contact edits and target receipts itself.
    source=selected_export(project,kind)
    if source is None:
        raise ValueError(f'{kind} is unavailable or has pending correction edits. Apply Correction / Prepare Preview before exporting.')
    value=dict(model=model,source=str(source))
    root=root_for(project)
    # Same condition as the GUI: a take in a people scene exports with its shared placement.
    if load_people(root) is not None:
        from .scene_export import context
        value['scene']=context(root,project)
    return value


def main(argv):
    try:value=plan(*argv)
    except (OSError,ValueError,KeyError,TypeError) as error:value=dict(error=str(error))
    print('@@PLAN@@'+json.dumps(value),flush=True)
