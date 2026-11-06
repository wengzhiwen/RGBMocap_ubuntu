# NVIDIA GB10 / Spark validation

## Scope — 2026-11-06

The Spark work extends the Ubuntu port to aarch64 while keeping the amd64 and
Windows runtime pins. It prioritizes FlowHMR and its separate HaMeR refinement.
Full motion inference on GB10 is still pending sufficient free unified memory.
The successful amd64 451-frame pipeline is recorded separately in
[Ubuntu validation](UBUNTU_VALIDATION.md).

Actual remote host: NVIDIA GB10 (compute capability 12.1), Ubuntu 24.04.5 LTS
on aarch64, driver 580.178.04. Existing unrelated GPU work was preserved.

## CUDA operation checks

A separate experimental environment used the official aarch64 CUDA 13 wheels:
Torch 2.9.1+cu130 and torchvision 0.24.1. Tests actually executed on GB10:

- FP32 CUDA matrix multiplication, checked against the CPU result.
- Torchvision CUDA non-maximum suppression.
- FP32 and FP16 scaled dot-product attention.
- FP16 cuDNN convolution, checked against its expected value.

The operation probe limited its PyTorch allocator to 128 MiB and peaked at about
8.2 MiB allocated tensors. CUDA context/library memory is additional. It was
executed without stopping or reconfiguring other GPU applications.

This Torch wheel advertises SM 12.0 binaries and emits a capability-range warning
for GB10's SM 12.1. The listed operations succeeded despite that warning. This
establishes those operations, not a native SM 12.1 PyTorch build or complete
model compatibility. The probe environment has not been registered as a model
runtime based on imports alone.

## Native contact correction

The existing C++ correction implementation uses x86 intrinsics. An ARM-only
compile-time adapter maps them through a pinned sse2neon header, preserving the
numerical C++ source and the original x86 build flags. The ARM build disables
floating-point contraction and preserves the required min/max semantics. Hardware
square root and division retain their default exact implementations.

The real extension was built with GCC on GB10 for CPython 3.10.22. The same
451-frame correction input, NumPy 1.26.4 and SciPy 1.15.3 were evaluated on amd64
and ARM64. Maximum differences were:

| Quantity | Maximum difference |
| --- | ---: |
| Root position | 0.00000960 m |
| FK joint position | 0.0000608 m |
| Local joint rotation | 0.0186 degrees |

Outputs were finite. These are architecture-to-architecture disagreements on
one fixture, not errors against ground truth or a universal tolerance guarantee.
The fixture and licensed geometry remain private and excluded from Git.

## Memory and remaining work

At bring-up, an unrelated workload left about 5.5–6.8 GiB of system memory
available. Complete SAM/HaMeR inference has not been attempted under that
constraint. The CUDA probes and CPU correction test completed and released
their resources.

GB10 shares RAM between CPU and GPU; `nvidia-smi` reporting framebuffer usage as
unsupported is expected. Available system memory and CUDA allocation headroom
must both be considered. See [NVIDIA's unified-memory guidance](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html).

## Integrated Spark installation

The actual Ubuntu installer created separate shared/GUI, Flow and ExMR/HaMeR
environments on GB10 with the hash-pinned CUDA 13 wheels. All four activation
checks passed. This was a fresh install of those project environments on the
existing configured Spark host, not a clean operating-system installation.

`doctor --gpu` completed with zero failures: real CUDA matmul and NMS passed in
all three worker environments, and registered FlowHMR/HaMeR assets were ready.
The real HaMeR hand-feature import chain and MANO preparation also succeeded.
The CLI then prepared the bundled public sample and completed GPU actor tracking
for 16 frames. No complete SAM or HaMeR model inference has been claimed.

The 102-test portability suite passed on amd64 and GB10 (two applicable skips on
each initial run). The compiled ARM intrinsic semantics test was then executed
separately on GB10 with the fetched pinned header and passed. A separate synthetic
120-frame solve produced byte-identical root and rotation arrays on amd64 and
GB10; it changed 1,200 rotation values and exercised the real native solver.

The amd64 FlowHMR + HaMeR command continued to reuse cached stages after the
Spark changes. Remaining work is a fresh full Spark FlowHMR + HaMeR run with
sufficient free memory, its export checks and desktop GUI/playback verification.
Optional models remain outside this validation scope. ExMR Moving Camera needs
an unavailable ARM pycolmap build. Character/Blender exports need a compatible
ARM64 Blender 4.5 installation; source FBX export is independent of Blender.

## Performance preparation — 2026-11-06

The CLI source-export additions and gated performance tools were synchronized to
this host. The expanded CPU suite passed 149 tests with two conditional skips;
the corresponding amd64 run passed 152 with two skips, including three additional
two-process/placement correctness tests. No GPU inference is implied by those
CPU test counts.

The actual public `benchmark_flow.py migration` preflight deferred at 7.7 GiB
available RAM, before creating an inference project. Its 32 GiB admission budget
is conservative for this particular validation run, not a measured model minimum.
The unrelated 85,041 MiB compute workload remained running. Complete GB10
inference, formal single-GPU baselines and GPU TP2/PP2 trials are still pending.
See [the experiment protocol](BENCHMARK_PROTOCOL.md) for tools, CPU full-checkpoint
agreement and the required order of the subsequent tests.
