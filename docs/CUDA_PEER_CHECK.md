# CUDA peer transfer check

`tools/check_cuda_peer.py` copies known FP32 data between two GPUs and verifies every value on the CPU. It answers one question: did these copies arrive intact? It loads no model or weights, opens no network port and changes no OS, driver or service setting.

**Validation status (2026-11-06):** the actual tool ran on the two-PRO-4000 host
under the pinned Python 3.10/Flow runtime. It completed all eight cases and
returned exit 1: all four direct cases failed and all four CPU-staged cases were
exact, with no surviving worker processes. The 30 CPU tests also passed under
the pinned shared runtime. The observations below additionally include earlier
independent copy and real-model experiments.

## Observed problem

Host: two RTX PRO 4000 Blackwell GPUs, PCIe topology PHB, no NVLink, driver 595.99.02.

| Check | Result |
| --- | --- |
| `torch.cuda.can_device_access_peer` | `True` in both directions |
| 1M-element `arange` FP32 copy, CUDA0 to CUDA1 | **Wrong data**, maximum error 1048575 |
| Same copy, CUDA1 to CUDA0 | Exact in that initial probe; the later standalone run failed both directions |
| CPU-staged copies | Exact in both directions |
| Kernel IOMMU state | Default domain Translated, GPU groups DMA-FQ. No host changes were made. |
| Direct PP2 on real models | **Failed parity**, translation 20.2 m |
| Host-staged PP2 on learned models | Exact on 690 frames |
| NCCL with default IPC and shared memory | Fails or hangs |
| NCCL with P2P and SHM disabled, loopback sockets | Passes real models, five rounds, 690 frames |

So the peer flag did not predict correct transfers. The translated IOMMU domain is a strong diagnostic lead, but it is not a proven exclusive cause: no test here changed the IOMMU setting and compared results. See [NVIDIA's NCCL GPU troubleshooting guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/gpu_troubleshooting.html) for its guidance. Any host change is your decision, and this tool never makes one.

The standalone run used 1,048,576 values plus a distinct case offset. Every
direct case had 1,048,576 mismatches; maximum differences ranged from 1,048,579
to 1,048,582. Host-staged cases had zero mismatches in both layouts/directions.
The earlier reverse-direction success does not establish that direction is
reliable. Private evidence: `.setup/bench/peer-tool-both.json` and `.log`.

On the actual single-GPU GB10 host, the tool returned `NOT_APPLICABLE` with exit
3 before starting a worker. That is the intended scope result, not a peer pass
or a failure of the separately validated FlowHMR pipeline.

## Commands

Run from the repository root. The launcher needs `.venv` (for `psutil`) and the copies run in `.venv-flow`; both come from `tools/setup_ubuntu.py`. `nvidia-smi` must be on `PATH`. Under a Python without `psutil`, the launcher refuses before starting a worker.

```bash
# Compare direct and CPU-staged copies on visible GPUs 0 and 1
.venv/bin/python -I -B -X utf8 tools/check_cuda_peer.py --transport both

# One JSON report on stdout, worker diagnostics on stderr
.venv/bin/python -I -B -X utf8 tools/check_cuda_peer.py --transport both --json > peer.json 2> peer.log; echo "exit $?"

# Swap which GPU is cuda:0, to see whether a fault follows the physical direction; larger copy, shorter limit
.venv/bin/python -I -B -X utf8 tools/check_cuda_peer.py --transport direct --devices 1,0 --elements 8388608 --timeout 120

# CPU-staged copies only
.venv/bin/python -I -B -X utf8 tools/check_cuda_peer.py --transport host

# Logical indices count within CUDA_VISIBLE_DEVICES when it is set: this selects GPU-bbb and GPU-aaa
CUDA_VISIBLE_DEVICES=GPU-aaa,GPU-bbb .venv/bin/python -I -B -X utf8 tools/check_cuda_peer.py --devices 1,0

# CPU tests, which need no GPU or Torch
python3 -m unittest discover -s tools/tests -p 'test_cuda_peer.py' -v
```

| Option | Default | Limits |
| --- | --- | --- |
| `--transport direct\|host\|both` | `both` | |
| `--devices A,B` | `0,1` | Two distinct logical indices |
| `--elements N` | 1048576 | 2 to 8388608 FP32 values per copy |
| `--timeout SECONDS` | 300 | Finite, above 0, at most 3600 |
| `--json` | Off | |

## What runs

`both` runs eight cases: each transport, each direction, and each layout. CPU-staged cases run first, so a hang in a direct copy cannot lose them.

| Term | Meaning |
| --- | --- |
| `direct` | `source.to(destination)` |
| `host` | `source.cpu().to(destination)` |
| `contiguous` | A dense tensor |
| `noncontiguous` | A stepped slice, `stored[::2]`, with sentinel values between the elements |

PyTorch's copy code is expected to use different paths for the two layouts: a plain memory copy for dense data, and a copy kernel or temporary buffers otherwise. The tool does not observe which path ran. A dense transpose is deliberately not used, because it could still reduce to a memory copy.

Each case works as follows:

1. **Build the data on the CPU.** The values are `arange(N) + case id`. The offset differs per case, so reused allocator memory holding an earlier case's values cannot pass as a fresh copy. It also means a zero-filled destination reports a maximum difference of `N - 1 + case id`.
2. **Check the source.** The source GPU's copy is read back first. If it is already wrong, the case reports an error, because the fault cannot be attributed to the peer transfer.
3. **Copy, then compare every value.** The destination is read back to the CPU and compared by bit pattern, so `-0.0` against `0.0` also counts as a mismatch.

## Reading the result

The report lists each case's exactness, mismatch count, maximum absolute difference, first mismatch and source and destination logical indices and UUIDs. It also records GPU names, driver, PCI addresses, Torch and CUDA versions, both `can_device_access_peer` flags, CUDA environment variables and readable IOMMU state (the IOMMU kernel arguments, IOMMU units, and each GPU's group and domain type).

| Exit | Verdict | Meaning |
| --- | --- | --- |
| 0 | `PASS` | Every requested case was present and exact, and the worker exited cleanly |
| 1 | `FAIL` | At least one case was wrong or raised an error |
| 2 | `ERROR` | Invalid arguments |
| 3 | `NOT_APPLICABLE` | Fewer than two visible GPUs. Peer transfer was not tested, so this is not a P2P pass |
| 4 | `ERROR` | The check could not finish: `nvidia-smi` or a runtime unavailable, a device index not visible, a worker crash, a Resource Safety stop, or a process that survived SIGKILL |
| 5 | `REFUSED` | A selected GPU has other compute work. Nothing was stopped |
| 124 | `TIMEOUT` | The limit expired with no failure recorded |
| 130 | `CANCELLED` | Cancelled with no failure recorded |

- **`FAIL` takes priority.** A wrong direct copy stays `FAIL` with exit 1 under `both`, even if every CPU-staged case is exact. It also stays `FAIL` if a later case times out. `transports` shows each result separately, and `worker.termination` records how the run ended.
- **`--transport host` is not P2P evidence.** A pass reports `direct: NOT_TESTED`.
- **A direct `PASS` is limited to this run.** Faults can be intermittent, so repeat the command when that matters. It also does not show which hardware path carried the data, and it does not validate NCCL.

## Safety and limits

- **Resource Safety is always on.** The launcher sets `RGB_MOCAP_RESOURCE_SAFETY=1`, whatever the local preference says. It applies the GUI's `ProcessingGuard` every 0.5 seconds for low-RAM and 180-second stall stops. The worker applies the usual PyTorch VRAM allocator budget to both GPUs.
- **Other workloads are refused.** `idle_devices` checks just before the worker starts that neither selected GPU has compute processes. `nvidia-smi` may not list processes from other containers, graphics workloads are not counted, and work that starts after the check is not detected.
- **Tensor memory is bounded.** At the maximum `--elements`, one case holds at most about 96 MiB of tensor data on each GPU and a few hundred MiB of CPU buffers. The CUDA context and allocator overhead are additional.
- **Stopping is bounded.** Timeout or cancellation sends SIGTERM to the worker's process group, then SIGKILL, and the launcher never waits indefinitely. A process stuck in a driver wait can survive SIGKILL. The report then lists its PID and kernel state and the exit is nonzero. A descendant that left the process group is not tracked; the worker starts no children itself.
- **Launcher killed outright.** If the launcher itself receives SIGKILL it cannot clean up, so the worker carries its own alarm at the timeout plus 30 seconds.
- **Platform.** This is Linux only, like the rest of the Ubuntu route. MIG device identifiers in `CUDA_VISIBLE_DEVICES` are refused rather than guessed. Without that variable, selection indices follow `nvidia-smi`; the worker uses explicit UUIDs for the selected pair.
