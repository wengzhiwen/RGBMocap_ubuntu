#!/usr/bin/env python3
"""Read-only probe for the separately provisioned FAH environment.

Run it inside that environment only:

    .venv-fah/bin/python -I -X utf8 tools/check_fah_runtime.py

It validates the interpreter, the pinned upstream package revision and the asset
layout, imports the pinned package, and runs one small CUDA operation. It never
installs or downloads anything, opens no weight file, and writes nothing.

Exit status: 0 ready, 2 the package checks out but CUDA is unusable, 1 otherwise.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))


def main() -> int:
    from mocap import fah_runtime
    value = dict(type='fah_probe', ok=False, environment=str(fah_runtime.venv(ROOT)),
                 revision=fah_runtime.REVISION, revision_ok=False, cuda=False)
    try:
        info = fah_runtime.activate(ROOT)
    except (OSError, RuntimeError, ValueError) as error:
        value['error'] = str(error)
        print('@@FAH@@' + json.dumps(value), flush=True)
        return 1
    value.update(ok=True, revision_ok=True, package=info['package'], commit=info['commit'],
                 python=info['python_version'])
    gaps = [row['name'] for row in fah_runtime.readiness(ROOT) if row['required'] and not row['ok']]
    value['asset_gaps'] = gaps
    try:
        import fah  # noqa: F401
        import torch
    except Exception as error:  # a broken environment must report, not traceback, in doctor
        value['error'] = f'{type(error).__name__}: {error}'
        value['ok'] = False
        value['revision_ok'] = True
        print('@@FAH@@' + json.dumps(value), flush=True)
        return 1
    value['import_ok'] = True
    value['fah_source'] = str(Path(fah.__file__).resolve())
    value['torch'] = torch.__version__
    value['torch_cuda_build'] = torch.version.cuda
    if torch.cuda.is_available():
        try:
            tensor = torch.zeros(16, 16, device='cuda')
            value['cuda'] = bool((tensor @ tensor.T).isfinite().all().item())
        except Exception as error:
            value['cuda_error'] = str(error)
    if not value['cuda']:
        print('@@FAH@@' + json.dumps(value), flush=True)
        return 2
    print('@@FAH@@' + json.dumps(value), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
