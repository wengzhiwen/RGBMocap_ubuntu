# Ubuntu validation log

This fork starts from RGB MoCap upstream commit
`ef05127a0dbddb3f3ac672a0722dddeae42ea651`. Windows validation in the original
project does not establish Linux compatibility. This record describes actual
local tests, with private clips/weights/results excluded from Git.

## Initial bring-up — 2026-11-06

Host: Ubuntu amd64, two NVIDIA RTX PRO 4000 Blackwell GPUs (24 GB each),
NVIDIA driver 595.99.02. Tests use one GPU per inference process; they do not
establish multi-GPU inference support.

Independent Linux environments were prepared using the repository's profile
pins: Python 3.10.22 / Torch 2.7.1+cu128 for shared and Flow, Python 3.12.10 /
Torch 2.8.0+cu128 for ExMR/HaMeR. Flow retains NumPy 1.26.4 and timm 1.0.30;
ExMR retains NumPy 2.3.5 and timm 0.9.12. Pure Python wheel sources in
`installer/runtime_sources` avoid rebuilding the legacy chumpy distribution.
No Windows runtime was reused. Licensed assets were reused privately from an
existing authorized installation, with its source files preserved.

Actual preliminary checks:

- CUDA matrix multiplication succeeded on each physical GPU with Flow's Torch.
  ExMR's separate Torch also executed a CUDA matrix operation.
- The Qt GUI opened on the local desktop and produced a startup screenshot.
- A 16-frame, 30 fps working clip completed YOLO actor tracking, Fast SAM body/
  hand feature extraction, and FlowHMR generation with 20 ODE steps, FP32 and
  the existing 256-query attention implementation.
- HaMeR inferred every frame: both hands had 16 valid predictions. Its cached
  refinement preserved body/root/wrist rotations, translation and shape exactly
  in tensor comparisons while replacing fingers. This tests preservation,
  not motion accuracy against ground truth.
- Native SMPL-H surface/bone preview preparation generated BVH and preview
  artifacts. Transactional source FBX export then succeeded; Blender imported
  all 52 source bones (65 with terminal bones), with a 30 fps action on frames
  1–16. This checks import structure/timing, not full per-bone matrix equivalence.
- The existing modified motion-correction C++ source compiled with GCC through
  CMake and its CPython 3.10 Linux module imported. The short clip subsequently
  completed real contact correction using that module, with finite output and
  unchanged finger rotations. This preliminary test explicitly supplied the
  compiled module to the existing backend loader; normal `.so` discovery still
  needs integration verification.
- Blender 4.5.4 LTS's official Linux archive was SHA-256 checked against its
  release checksum and executed in background mode.

Initial integration blockers were Windows-only interpreter paths, `peak_wset`
statistics, process flags, `.exe`/`.pyd` discovery and forced Windows OpenGL
selection. HaMeR's legacy renderer imports also require GLU on Linux.
The first numerical-stage checks above called existing Python worker/back-end
paths directly; they do not by themselves establish a complete CLI or GUI run.
Further end-to-end results will be recorded here after integration.

Not established by these checks: clean Ubuntu installation, long-clip quality,
interactive playback, multiple cameras, optional MAMMA/PromptHMR native CUDA
builds, PEAR live capture, full ExMR/GVHMR inference, character retargeting,
portable Blender scene/VMD export, or equivalence with Windows output. No model equations,
precision, temporal context or licensed model assets were changed to obtain
these preliminary results.

## Normal worker integration

After platform routing and memory-statistics fixes, a fresh 16-frame project was
prepared through the existing worker entry point. All six stages then returned
success through validated isolated worker commands: tracking, static camera,
Fast SAM features, FlowHMR with a fresh every-frame HaMeR child, correction and
original/corrected preview/BVH publication. The actual compiled correction module
was discovered through Linux runtime activation. The Qt startup diagnostic also
returned success and confirmed no inference module had loaded in the GUI.

Opening this project displayed the video, skeleton and native SMPL-H mesh. A
short playback diagnostic observed zero presented frames in this desktop session;
playback smoothness/synchronization therefore remains unverified and requires
further investigation. A screenshot alone does not establish working playback.

A separate 16-frame moving-camera worker run successfully computed VGGT camera
matrices before the original Windows-only reporting failure. All stored per-frame
rotation/translation and intrinsic matrices were finite; a successful complete
moving-camera CLI run is a separate integration check.

## CLI and longer-clip integration

The public `rgbmocap.sh` entry point completed these additional checks:

- A fresh 16-frame static-camera run with HaMeR, using the connected feature/
  generator batch worker. A repeated command reused tracking, camera, feature
  and motion caches, then reused the prepared preview generation.
- A complete 451-frame (about 15 s) portrait video: automatic moving-camera
  sampling (114 VGGT samples), Fast SAM, full-context FlowHMR, every-frame HaMeR,
  contact correction and both preview generations. HaMeR had 451 valid left-hand
  and 408 valid right-hand frames; unavailable crops retain FlowHMR fingers.
- Exact tensor checks on the longer run confirmed that HaMeR preserved body,
  root/wrist rotations, translation, shape and native foot contacts; subsequent
  contact correction preserved finger rotations. This is preservation evidence,
  not a ground-truth quality comparison.
- Corrected FBX exported through the CLI, then imported in Blender with all 52
  source bones and an action spanning frames 1–451 at 30 fps.
- A separate 16-frame static-camera run with Fast SAM disabled and Low Memory
  selected completed through the original SAM path.
- `doctor --gpu` passed real CUDA matrix multiplication in shared, Flow and
  ExMR runtimes. Optional HandFlow/HTD profiles and auxiliary assets for the
  other models were still absent; they are separate readiness warnings.
- The GUI launcher completed its startup check, including worker creation.

The desktop's GNOME ScreenSaver `GetActive` response was true during playback
measurement. The zero-frame presentation report was obtained while the desktop
was locked. Interactive playback remains to be tested on an unlocked desktop;
no screen unlock or source-video modification was performed.

After the user unlocked the desktop, a second real raw-mesh + bone-overlay
playback check on the same 451-frame project presented 227 video frames in
7.60 seconds (29.87 fps). Maximum video/skeleton and mesh/video frame differences
were both zero; p95 frame interval was 34.66 ms and maximum 42.74 ms. It reported
no mesh/bone renderer errors, no inference modules loaded, zero paused mesh
compute dispatches and 0.0% measured paused CPU. The screenshot was inspected
and showed the video, skeleton and native SMPL-H mesh. This validates that local
unlocked-desktop playback interval, not every display/backend or long-session
playback behavior. Private report/screenshot: `.setup/ubuntu/playback-unlocked.*`.

The subsequent public 690-frame moving-camera benchmark completed one excluded
warm-up and five fresh measured pipelines on both amd64 and GB10. See
[the benchmark results](BENCHMARK_RESULTS_UBUNTU.md) for timings, inputs, runtime
differences and the scope of actual dual-GPU validation. The expanded 153-test
CPU suite passed on both hosts with two conditional skips, including active-gate
and intentionally corrupted-reduction checks in the model-parallel tests.
After the standalone peer diagnostic was added, the full CPU suite passed
183 tests on each host with two conditional skips. The real two-GPU diagnostic
returned the expected failed-direct/passed-host result on amd64; its single-GPU
Spark invocation correctly returned Not Applicable.

## Foundation regression checks

- 74 CPU tests ran under the pinned shared Python; 73 passed and the test for a
  mismatched Python patch was skipped because this interpreter matches its pin.
- `tools/check_source.py` reported 2,589 parsed files and zero errors.
- The Ubuntu installer was actually rerun for shared, Flow and ExMR and all
  activation checks passed. This is an existing-environment repair/reuse check,
  not a clean-install test.
- The public native build tool successfully rebuilt/reused and installed the
  real GCC extension for the requested Python 3.10 environment.
- CLI supervision now uses the same `ProcessingGuard` policy as the GUI. Tests
  cover stopping a silent worker in both Resource Safety modes and cancellation
  of a descendant that deliberately ignores SIGTERM. A supervised cached
  FlowHMR + HaMeR rerun also completed successfully.
- Final `doctor --gpu` had zero failures and three readiness warnings: the
  optional HandFlow and HTD environments, and auxiliary assets for other models.

## Additional CLI source exports — 2026-11-06

The CLI now uses the GUI's selected-result and people-scene placement guards for
FBX, BVH and Blender source exports. On the existing 451-frame amd64 project:

- Corrected Generic BVH was byte-identical to its selected source.
- Biped BVH with a prepended T-pose contained 452 frames.
- Blender 4.5.4 opened the actual exported portable bundle: 52 source bones,
  30 fps, action and scene frames 0–451, one camera and a relative `//video.mp4`
  background reference. No interactive playback or numerical camera equivalence
  is established by this structural check.

The source/export CPU regression checks passed, including unavailable/pending
results, shared scene placement, foreign-interpreter refusal, per-format options,
bundle overwrite refusal and cleanup of request-owned temporary paths. These
checks do not establish character retargeting or Blender compatibility on ARM64.

## ARM64 CPU checks

2026-11-06, on a CPU-only Ubuntu 24.04 aarch64 machine (glibc 2.39, 3 GB RAM). It has no NVIDIA GPU, so **nothing here establishes CUDA, GPU inference or DGX Spark behavior**. No weights, projects or real videos were used.

- `uv` 0.13.1 dry runs found aarch64 wheels for every pin in the shared, GUI, Flow, HandFlow, HTD and ExMR groups except `pycolmap==3.11.1`, which also has no cp312 aarch64 wheel in any other version. The Torch 2.9.1 Linux requirements (15 `nvidia-*` packages and `triton`) also resolved.
- The cu130 index's aarch64 `torchvision-0.24.1` (7.8 MB) links `libcudart.so.13` and `libtorch_cuda.so`, and reports CUDA 13.0 after import. PyPI's wheel of the same version is a different 2.4 MB build. This is why setup pins by hash.
- `setup_ubuntu.py` installed shared, Flow and ExMR with those wheels and passed activation for `gui`, `shared`, `flow` and `exmr`. This was a clean install on that machine. HandFlow and HTD were not installed, because of disk limits.
- CPU imports succeeded in `.venv-flow` for the FlowHMR, Fast SAM, VGGT camera and worker modules, and for pinned libraries including Lightning, timm, ultralytics, OpenCV and PyAV. CPU `torchvision.ops.nms` and scaled dot-product attention also ran. `chumpy` does not import under Flow's NumPy 1.26.4, because `np.bool` no longer exists there. This comes from those pins, not ARM.
- CPU imports succeeded in `.venv-exmr` for HaMeR, pyrender with `PYOPENGL_PLATFORM=egl` and the GLU library from the documented no-sudo steps, ExMR's hand features and its pipeline, all without `pycolmap`. Importing moving-camera odometry reports the documented reason.
- `doctor` reported the variant, the unavailable package and CUDA-linked torchvision in each environment. It correctly failed for missing assets and models. `doctor --gpu` could not run here.
- The real `prepare` worker ran through validated routing on a synthetic clip. The GUI started offscreen with software OpenGL, its worker check returned success and no inference module loaded in the GUI. This says nothing about a real display, the GB10 OpenGL driver or playback.
- The native correction module built with GCC 13.3 through the public tool, with the numerical C++ source unedited. A compiled test matched x86 semantics for variable permute, NaN min/max and out-of-range conversion. It also found zero bit differences from IEEE results for 200,000 divisions, square roots and products, and confirmed that multiply-add is not fused. This test found that `SSE2NEON_PRECISE_SQRT=1` changes 80,389 of 200,000 square roots on aarch64, so that flag is deliberately not used.
- `tools/correction_crosscheck.py` was deterministic across runs on synthetic motion and changed 1,200 rotation values. The subsequent GB10-to-amd64 comparison produced identical root/rotation arrays; see [Spark validation](SPARK_VALIDATION.md) for that test and the separate 451-frame comparison.
- Configuring baseline and new `CMakeLists.txt` for an x86_64 target, using this machine's GCC, produced identical compile flags, definitions and include paths for both targets and fetched no `sse2neon`. No x86 module was compiled on this machine.

Subsequent integration on GB10 passed installation, `doctor --gpu`, a 16-frame preparation/tracking run and native correction comparisons. The 102-test suite passed on amd64 and GB10, with platform/pin-specific skips; the compiled ARM intrinsic test was separately run successfully on GB10. A repeated amd64 FlowHMR + HaMeR command reused its caches after the architecture changes. Full GB10 SAM/HaMeR inference and real-display GUI testing remain pending; see [Spark validation](SPARK_VALIDATION.md).
