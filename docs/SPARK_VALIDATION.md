# NVIDIA GB10 / Spark validation

## Scope — 2026-11-06

The Spark work extends the Ubuntu port to aarch64 while keeping the amd64 and
Windows runtime pins. It prioritizes FlowHMR and its separate HaMeR refinement.
The complete 60-frame static-camera FlowHMR + HaMeR + correction + source-export
validation has passed on GB10 after the user authorized releasing the existing
GPU service. This is the tested subset, not validation of every model or GUI.
The successful amd64 451-frame pipeline is recorded separately in
[Ubuntu validation](UBUNTU_VALIDATION.md).

Actual remote host: NVIDIA GB10 (compute capability 12.1), Ubuntu 24.04.5 LTS
on aarch64, driver 580.178.04. Existing unrelated GPU work was initially preserved
and later stopped gracefully with explicit user authorization for these tests.

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

At bring-up, an unrelated workload left about 5.5–7.7 GiB of system memory
available. Full SAM/HaMeR inference was deferred under that constraint. After
explicit user authorization, its named systemd user service was stopped
gracefully, leaving about 118 GiB available. No model/data files were deleted,
the service was not disabled, and Resource Safety remained enabled.

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
for 16 frames. These initial checks preceded the complete run below.

The 102-test portability suite passed on amd64 and GB10 (two applicable skips on
each initial run). The compiled ARM intrinsic semantics test was then executed
separately on GB10 with the fetched pinned header and passed. A separate synthetic
120-frame solve produced byte-identical root and rotation arrays on amd64 and
GB10; it changed 1,200 rotation values and exercised the real native solver.

The amd64 FlowHMR + HaMeR command continued to reuse cached stages after the
Spark changes. The subsequent full run passed; Spark desktop GUI/playback
verification remains pending because this host was tested through SSH.
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
The unrelated 85,041 MiB compute workload remained running at that point. Complete
GB10 inference, formal single-GPU baselines and GPU TP2/PP2 trials were then pending.
See [the experiment protocol](BENCHMARK_PROTOCOL.md) for tools, CPU full-checkpoint
agreement and the required order of the subsequent tests.

## Complete FlowHMR and HaMeR validation

The bundled public sample supplied a fresh 60-frame, 1920×1080, 30 fps case with
explicit static camera, Fast SAM, default adaptive batching/cropping, FP32
full-context FlowHMR (20 ODE steps, 256-query MATH attention), every-frame HaMeR
and default contact correction. All stages completed through validated isolated
worker environments with Resource Safety enabled.

Both hands had valid predictions in all 60 frames. CPU inspection confirmed
finite outputs; exact preservation of body/root/wrist rotations, translation,
shape and native contacts by HaMeR; exact preservation of fingers by correction;
current output receipts and selected-result BVH for both original and corrected
motion; and actual native correction execution. Corrected source FBX exported
successfully. A repeated CLI command reused tracking, camera, features, body
generation and correction caches. Tensor hashes remained identical after that
rerun. This establishes that case's functional compatibility, not ground-truth
motion quality or equivalence with Windows.

The repeatable machine-readable gate report is private under
`experiments/bench-gb10-migration/migration.json`; it was produced by the actual
public benchmark launcher, not a manual success marker. Subsequent timing scopes
and longer cases are documented separately in the benchmark protocol/results.

The complete bundled 690-frame (23-second) case then passed with Moving/Unknown
Auto camera estimation, including 116 VGGT samples at interval six, Fast SAM,
every-frame HaMeR, native correction and both previews. Both hands were valid in
all 690 frames, and the artifact-preservation validator passed. Five subsequent
fresh repetitions completed after an excluded warm-up. Source FBX from the 60-
and 690-frame cases imported successfully in amd64 Blender 4.5.4 at 30 fps with
the expected action ranges. This does not supply an ARM64 Blender runtime.

[The performance results](BENCHMARK_RESULTS_UBUNTU.md) contain the actual PRO
4000/GB10 baselines, generator-only TP2/PP2 measurements and decoder differences.
