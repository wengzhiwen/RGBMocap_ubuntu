# RGB MoCap on Ubuntu

This fork adds a Linux route beside the Windows Git/CMD distribution, for amd64 and for aarch64 systems such as NVIDIA DGX Spark. FlowHMR and its optional HaMeR finger refinement are the priority. Windows behavior, inference and receipts stay as they were; see [what changed](#how-the-linux-route-works).

## Status

| Area | State on Linux |
| --- | --- |
| FlowHMR, HaMeR refinement, GUI, CLI, correction, FBX export | Implemented. Check [validation](#validation) for what has actually been run. |
| ExMR, GVHMR and add-ons (HTD, FootMR, HandFlow) | Routed to their own environments; not yet verified end to end. |
| MAMMA, PromptHMR | Not supported. Their setup builds Windows native modules and their adapters remain Windows-only. |
| PEAR live preview | Not supported. Its shared camera frames use Windows named shared memory. |

### Validation

The CPU portability suite covers routing, pinned environments, cancellation, cache identities and setup planning. Local GPU checks additionally completed a 451-frame FlowHMR + HaMeR + moving-camera + correction pipeline and verified its FBX import in Blender. The GUI starts and loads the resulting project. See [the detailed validation record](UBUNTU_VALIDATION.md) for tested settings and limits.

Clean amd64 installation and other models remain unverified. A subsequent unlocked-desktop playback check passed at 29.87 fps with zero observed video/skeleton/mesh frame differences. On aarch64, see [DGX Spark and aarch64](#nvidia-dgx-spark-and-aarch64): the complete 60-frame static-camera FlowHMR + HaMeR + correction + source-FBX and cache-rerun check passed on GB10 after memory was released with user authorization. See [Spark validation](SPARK_VALIDATION.md) for the tested subset and remaining limits.

Repeated PRO 4000/GB10 baselines and experimental dual-PRO-4000 TP2/PP2 follow
[the benchmark protocol](BENCHMARK_PROTOCOL.md). The tools are separate from
production inference; timing began after complete GB10 migration validation.
See [measured results](BENCHMARK_RESULTS_UBUNTU.md) for five-round baselines,
the peer-transfer failure and the tested TP2/PP2 transport workarounds.
The [standalone peer checker](CUDA_PEER_CHECK.md) validates actual copied values;
advertised peer-access flags alone were insufficient on the tested dual-card host.

A [fresh deployment on a second GB10](validation/gb10-fresh-deployment-2026-11-07.md)
subsequently passed from the published branch without application-source fixes:
independent pinned environments, native correction build, actual CUDA checks,
60/690/451-frame FlowHMR + HaMeR cases, source FBX and native GUI project
rendering. Interactive GB10 playback and target-side Blender remain unverified.

ARM64 GUI deployments pin the PySide6/Essentials/Addons/shiboken6 quartet to
6.7.3 in `installer/linux-aarch64.json`. The tested 6.8.3 ARM64 wheels import
successfully on Python 3.10 but consume a `None` reference per ordinary void
painting call, eventually aborting the GUI. The amd64/Windows 6.8.3 pins and
numerical environments remain unchanged. Existing ARM64 installs should run
`python3 tools/setup_ubuntu.py --profiles shared`, with the GUI closed.
Setup and `./rgbmocap-cli.sh doctor` now exercise these bindings using the headless
`tools/check_qt_runtime.py` probe rather than relying only on import success.
See the [crash follow-up and 60-second GB10 playback check](validation/gb10-gui-crash-2026-11-07.md).

## Prerequisites

- 64-bit Ubuntu on x86-64 or aarch64, an NVIDIA GPU with a current driver, and OpenGL 4.3 for the GUI viewport.
- [uv](https://docs.astral.sh/uv/) (0.13.1 is what the Windows lock pins). `setup_ubuntu.py` finds it on `PATH`, at `.bootstrap/bin/uv`, or through `--uv`.
- FFmpeg and FFprobe, either on `PATH` or as `tools/bin/ffmpeg` and `tools/bin/ffprobe` without a suffix.

System packages used by the GUI, HaMeR and the correction build:

```bash
sudo apt install ffmpeg libglu1-mesa libegl1 libgl1 libopengl0 libxcb-cursor0 cmake build-essential git
```

The setup never runs `apt`. Without sudo, unpack a library into the private `.runtime/lib` folder, which the launchers add to `LD_LIBRARY_PATH`:

```bash
mkdir -p .runtime/lib /tmp/glu && cd /tmp/glu && apt-get download libglu1-mesa
dpkg-deb -x libglu1-mesa_*.deb extracted
cp -a extracted/usr/lib/*/libGLU.so.1* "$OLDPWD/.runtime/lib/"
```

## Set up

```bash
python3 tools/setup_ubuntu.py --plan                  # print exact pins; add --machine aarch64 or x86_64 to plan for another system
python3 tools/setup_ubuntu.py                         # .venv, .venv-flow, .venv-exmr
python3 tools/setup_ubuntu.py --stage-assets          # hash-checked auxiliary files, no weights
python3 tools/setup_ubuntu.py --profiles shared --build-correction   # native correction module
./rgbmocap-cli.sh doctor
```

| Profile | Environment | Python | Pins |
| --- | --- | --- | --- |
| shared and gui | `.venv` | 3.10.22 | `runtime-shared.txt` + the lock's PySide6 `gui` group |
| flow | `.venv-flow` | 3.10.22 | `runtime-shared.txt` + `runtime-flow.txt` |
| exmr, also used by HaMeR | `.venv-exmr` | 3.12.10 | `runtime-exmr.txt` |
| hand (`--profiles hand`) | `.venv-hand` | 3.10.22 | `runtime-shared.txt` + `runtime-hand.txt` |
| htd (`--profiles htd`) | `.venv-htd` | 3.10.22 | `runtime-shared.txt` + `runtime-htd.txt` |

- Each environment is complete and independent. Merged pins follow the Windows overlay's import order: the later group wins by package name. For example, `.venv-flow` has NumPy 1.26.4 while `.venv` has 1.23.5.
- Everything installs with `--no-deps`. `+cu128` builds come from the PyTorch CUDA index. `win32-setctime` is skipped. `chumpy`, `antlr4-python3-runtime` and `iopath` are rebuilt from `installer/runtime_sources` by the same code the Windows developer setup uses.
- Linux CUDA requirements declared by the Torch wheel, such as `nvidia-*` and `triton`, are added and recorded in `<venv>/rgbmocap-linux.json`. Profiles using `iopath` also receive `portalocker` from the existing pinned ExMR profile, satisfying its Linux file-lock dependency.
- Python and the uv cache stay in `.runtime/python` and `.cache/uv`. Identical wheels are hardlinked from that cache, so repeated Torch files are not stored once per environment when the cache and venvs share a filesystem.
- Each environment must pass its activation check before setup reports it ready. If uv cannot provide an exact Python, setup stops unless you pass `--python-patch-fallback`; the accepted same-minor patch is then recorded and allowed by activation.
- Setup validates versions and import locations only, and it records `cuda_verified: false`. Importing Torch does not show that CUDA works: run `./rgbmocap-cli.sh doctor --gpu`, then a real clip.

Models are obtained separately, as on Windows: use **Model Manager**, or the `import_model` stage below. Do not copy runtime folders from a Windows install. Model receipts record file size and timestamp, so a copied file that shows as not ready should be imported again rather than having its receipt edited.

## Run the GUI

```bash
./RGBMoCap.sh
```

The original `RGBMoCap.sh` GUI entry point is preserved; `rgbmocap-gui.sh` is an additional GUI entry point with the same behavior. The CLI uses `rgbmocap-cli.sh` (formerly `rgbmocap.sh`) to avoid a filename collision with `RGBMoCap.sh` on case-insensitive filesystems. Existing GUI shortcuts continue to work; update CLI scripts to use `rgbmocap-cli.sh`.

Keep the terminal open; it mirrors processing messages, as the Windows command window does. The launcher adds `tools/bin` and `.runtime/lib` when they exist, then validates `.venv` instead of reading the Windows managed-install state.

## Command line

The CLI runs the same workers with the same project files and receipts, so the GUI can open its projects and completed stages are reused either way.

```bash
./rgbmocap-cli.sh doctor                # readiness; add --gpu for a real CUDA operation in each environment
./rgbmocap-cli.sh stages                # pipelines and the environment each stage uses
./rgbmocap-cli.sh flow --video clip.mp4 --hamer --dry-run
./rgbmocap-cli.sh flow --video clip.mp4 --start 0 --end 5 --hamer
./rgbmocap-cli.sh flow --video clip.mp4 --camera-interval -1 --correct
./rgbmocap-cli.sh export --project <reported project> --kind initial --output motion.fbx
./rgbmocap-cli.sh export --project <reported project> --kind corrected_initial --output motion.bvh
./rgbmocap-cli.sh export --project <reported project> --output biped.bvh --bvh-profile biped --prepend-tpose
./rgbmocap-cli.sh export --project <reported project> --output bundle.blend --animate-camera
./rgbmocap-cli.sh stage correct --project <reported project> --result corrected_initial
./rgbmocap-cli.sh stage import_model --model hand_hamer --video /path/to/downloaded.file
```

- `flow` runs `prepare`, `track`, `flow_camera`, the joined `flow_batch` (`flow_features` + `flowhmr`), optional `correct`, then `flow_export`. It finishes by printing the model project path that `export` needs.
- `--hamer` follows the GUI rule: it forces Fast SAM and 4-frame hand sampling, and the fingers run in `.venv-exmr`. `--camera-interval` matches the GUI's camera choices: `0` is Moving/Unknown Auto and `-1` is Static Camera. Other flags include `--no-fast-sam`, `--no-sam-batch`, `--no-gpu-crop`, `--no-hand-stride4`, `--compute-mode low_memory` and `--force`.
- `stage` is the escape hatch: it passes native worker arguments through unchanged and returns the worker's exit code. Common options such as `--json` go before the stage name.
- `Ctrl+C` stops the worker's whole process group, including FFmpeg and the HaMeR child.

### Reuse a project's processing settings

```bash
./rgbmocap-cli.sh flow --video new_clip.mp4 --template /path/to/old/project.json --dry-run
./rgbmocap-cli.sh flow --video new_clip.mp4 --template /path/to/old/project.json --project /path/to/new_project
```

`--template` accepts a FlowHMR model manifest or a version-2 workspace's
`project.json`. For a workspace it reads the saved FlowHMR model settings,
falling back to its model manifest when settings are absent. The template is
read-only. Only FlowHMR options, the selected initial/corrected result and
correction settings are inherited. Source media, trims, people, manual contact
timings, scene state, results and receipts are not copied.

The template is authoritative: processing flags (`--hamer`/`--no-hamer`, SAM,
hand sampling, camera interval, `--correct` and correction options) are ignored,
with one notice on stderr listing explicitly supplied flags. Missing older
option fields use GUI defaults, never command-line overrides. HaMeR still
forces Fast SAM and four-frame hand sampling, as in the GUI.
`--video`, `--project`, `--start`, `--end`, `--compute-mode`,
`--resource-safety`, `--force`, `--json` and `--dry-run` remain execution inputs.
Resource Safety stays a local/run preference and is never imported from a project.
Dry-run prints the effective FlowHMR options, stages and prepare arguments.

The destination must be a new or empty folder, separate from the template
folder. Omitting `--project` selects a unique folder under `projects/` to avoid
reusing the template's video or cached state. To resume a created project,
rerun without `--template` and provide its project path and processing options,
or open it in the GUI. Other model templates and malformed settings are refused
before starting a worker.

Limits compared with the GUI:

- `flow` handles one person with automatic tracking, and there is no **Find person** selection. The GUI remains the route for multiple people.
- The CLI uses the GUI's `ProcessingGuard` every 0.5 seconds for measured low-RAM stopping and stall detection. `--resource-safety off` retains stall detection, adaptive routing and actual OOM recovery. Cancellation terminates the owned process group, including children that ignore graceful termination.
- `export` selects FBX, BVH or a portable Blender bundle from the output suffix (or `--format`). It uses the same `save_fbx`, `copy_bvh` and `save_blend` workers and generation/scene/contact guards as the GUI. BVH offers `--bvh-profile generic|biped`; `--prepend-tpose` adds one leading animation frame. A Blender target `bundle.blend` creates the new folder `bundle/` containing `bundle.blend`, video and camera metadata; existing bundle folders are refused. Blender exports require Blender 4.5. The selected `--kind` must have a current prepared result.

## Native correction, FFmpeg and Blender

- **Correction.** Build with `python3 tools/build_motion_correction.py --python .venv/bin/python`, which needs CMake, a C++17 compiler and Git to fetch the pinned pybind11 and Eigen sources. The module goes to `third_party/KimodoMotionCorrection/bin/`, where it is ignored by Git. The Python 3.10 environments share it. It is a local build: the Windows prebuilt binary's validation does not transfer to it, so check results before relying on them. Correction, and only correction, needs this module. Each module name carries its Python and CPU tag, so a build from another architecture is ignored.
- **FFmpeg.** Discovery order is `tools/bin/<name>`, then `PATH`.
- **Blender 4.5.** Discovery order is `RGBMOCAP_BLENDER` (a file or a folder), `runtime_character/blender`, then `PATH`. It is needed for character import, retargeting and Blender bundles, not for source FBX export.
- **PyOpenGL.** On Linux, ExMR and HaMeR use `PYOPENGL_PLATFORM=egl` unless you set another value. Importing pyrender then needs `libGLU.so.1` and `libEGL.so.1`, even though inference does not render.

## NVIDIA DGX Spark and aarch64

aarch64 uses a separately pinned variant, chosen automatically from the machine type. Windows and amd64 keep the pins above, and `PROFILES` is unchanged.

| | amd64 | aarch64 variant `linux-aarch64-cu130` |
| --- | --- | --- |
| Torch, shared and Flow | 2.7.1+cu128 | 2.9.1+cu130 |
| Torch, ExMR and HaMeR | 2.8.0+cu128 | 2.9.1+cu130 |
| torchvision | 0.22.1+cu128, 0.23.0+cu128 | 0.24.1 from the cu130 index |
| PySide6 / Essentials / Addons / shiboken6 | 6.8.3 | 6.7.3 (painting ABI check required) |
| Python, NumPy, SciPy, timm and other numerical pins | As above | Identical |
| ExMR Moving Camera | Available | Unavailable: `pycolmap` has no aarch64 wheel |
| Native correction | SSE4.1/AVX | NEON through a compile-time adapter |

Why these choices:

- The cu128 index has no aarch64 Torch 2.8.0, and the CUDA 13 stack is preferred for GB10 (`sm_121`). No code changes model equations, precision or temporal context. Torch itself differs from amd64, however, so results are not expected to be bit-identical.
- `torchvision` has no local `+cu130` tag on aarch64, and PyPI has a different CPU build under the same version. Setup therefore hash-pins Torch and torchvision to the CUDA index. `doctor` reports a CPU build as a failure, and `doctor --gpu` additionally runs a real CUDA NMS on this variant.
- All other required pins have aarch64 wheels on Ubuntu 24.04 (glibc 2.39); the GUI quartet uses the separate versions above. PySide6 installs in its own step: if it cannot be installed on another system, setup warns and CLI processing remains available.
- `pycolmap` is imported only by ExMR's moving-camera odometry. FlowHMR and HaMeR do not use it, and importing it reports this reason instead of a bare missing-module error.
- Blender has no official Linux aarch64 build. Character import, retargeting and Blender bundles need your own compatible Blender 4.5 through `RGBMOCAP_BLENDER`. Source FBX export does not use Blender.

Native correction compiles the same numerical C++ source. `compat/aarch64/immintrin.h` and the pinned `sse2neon` header map its SSE intrinsics to NEON. Flags keep x86's results where defaults differ: no fused multiply-add, signed `char`, and NaN-aware min/max. Hardware square root and division are bit-exact in the included compiled test. The estimate intrinsics (`rcpps`, `rsqrtps`) have no callers in the solver path. Other math, such as Eigen and `libm`, can still round differently, so compare builds with:

```bash
.venv/bin/python -I -B tools/correction_crosscheck.py run --output amd64.json   # on amd64
.venv/bin/python -I -B tools/correction_crosscheck.py run --output arm64.json   # on aarch64
python3 tools/correction_crosscheck.py compare amd64.json arm64.json
```

Spark notes:

- GPU and CPU share memory on this system, so other GPU workloads reduce the RAM that Resource Safety measures. It stops workers when RAM headroom is low, as on other systems. Automatic SAM batching used 16 frames in the measured cases with ample RAM; performance under low shared-memory headroom remains unmeasured.
- `doctor --gpu` allocates a small amount of GPU memory in each environment. Check for other workloads first on a shared machine.

### aarch64 validation status

| Check | Status |
| --- | --- |
| Wheel availability for every pin, Python 3.10 and 3.12 | Checked with uv on Ubuntu 24.04 aarch64 |
| Setup with hash-pinned wheels; activation | Passed on both the ARM CPU host and GB10 for shared, Flow and ExMR |
| FlowHMR and HaMeR import chains; base GUI start offscreen; `prepare` worker | Run on the same machine, with no weights or GPU |
| Native correction build; semantics test; determinism on synthetic motion | Run on the same machine |
| amd64-to-aarch64 correction comparison | Passed synthetic comparison and a private 451-frame input; see [measurements](SPARK_VALIDATION.md) |
| CUDA on GB10 | Matmul and CUDA NMS passed in all three runtimes; FP16 convolution and FP32/FP16 attention also passed |
| GB10 tracking | Video preparation and 16-frame GPU tracking passed |
| Complete GB10 FlowHMR + HaMeR + correction + source FBX | Passed the 60-frame static-camera public sample, including cached rerun |
| GB10 ground-truth quality | Unverified |
| GB10 desktop GUI | Native single-person mesh playback passed after the ARM64 Qt fix; see the [crash follow-up](validation/gb10-gui-crash-2026-11-07.md) |

See the [validation record](UBUNTU_VALIDATION.md#arm64-cpu-checks) for details.

## How the Linux route works

| Concern | Implementation | Why |
| --- | --- | --- |
| Routing | `runtime_registry.command/activate` defer to `mocap/linux_runtime.py` when `os.name != 'nt'` | Windows paths stay untouched and `PROFILES` is unchanged, because receipts embed it |
| Worker start | `.venv-*/bin/python -I -X utf8 worker_boot.py <stage>` | Same isolated UTF-8 launch as a managed install |
| Validation | `sys.prefix` must be the profile's venv; exact Python; pinned versions; imports inside its site-packages; no overlay `.pth` | Comparing resolved interpreters cannot tell venvs apart, because they share a base Python |
| Architecture | `effective_profile` returns `PROFILES[key]` itself on amd64, or a copy with `installer/linux-aarch64.json` applied | Receipts on amd64 stay byte-identical, and the variant can never leak into them |
| Peak memory | `mocap/process_memory.py`: `peak_wset` on Windows, `ru_maxrss` × 1024 on Linux | psutil has no `peak_wset` on Linux |
| Native correction | Linux activation installs a post-import patch for `correction.backend` | `correction.py` is byte-hashed into receipts; see below |
| Process control | `creationflags` falls back to 0; cancel stops descendants through psutil | `taskkill` and Windows flags do not exist on Linux |

### Cache identities

Saved work stays valid because receipts still see baseline code identities:

- `mocap/worker.py`, `mocap/exmr_backend.py` and `mocap/flow_hamer.py` are hashed with `code_hash`. Their Linux routing edits are registered in `PLATFORM_ONLY` in `mocap/resource_cache.py`, which maps the exact new file hash to the baseline identity. This uses the project's mechanism for reviewed non-numerical changes. Only add an alias when no numerical path changed.
- `correction.py`, `flow_correction.py`, `mixamo_data.py`, `handflow.py` and `handflow_engine.py` are hashed directly or stat-checked, so they must stay byte-identical. That is why correction uses an import-time patch and HandFlow's PyOpenGL value is handled by an import guard rather than edits.
- `flow_backend.py` is only identified by selected functions, and the edited reporting code is outside them.
- `mocap/runtime_identity.py` adds a `linux_variant` key (name, machine and a digest of the exact wheels) to stage, FlowHMR and ExMR receipts only on a variant, and adds nothing on Windows and amd64. HaMeR's `source_identity` records the runtime actually used. A project moved between architectures therefore recomputes its stages instead of reusing another build's results.

`tools/tests/` pins all of this, and fails if one of these files changes without a reviewed update.

### Windows impact

`runtime_registry.py` is hash-pinned, so adding the guard also updated `registry_sha256` and the derived `id` in `installer/runtime-lock.json`. The profiles and package groups are unchanged. An existing Windows install of this fork must complete one `update.cmd` or `install.cmd` repair: launching before that reports that dependencies do not match. Matching runtime groups are reused rather than downloaded again, but PEAR support is regenerated because its identity includes the registry file. Upstream Windows installs are not affected unless they take this change.

`SOURCE_MANIFEST.json` is upstream's release snapshot and was deliberately not regenerated.

## Upstream sync

**Baseline:** `1b0384367fc6b63932c82f1c9c5cb4243a91de2e` (upstream `main` merged on 2026-11-10; original port baseline was `ef05127a0dbddb3f3ac672a0722dddeae42ea651`).

Keep the Linux layer small and additive, and merge rather than rebase so the identity reasoning stays reviewable against each upstream version:

```bash
git remote add upstream https://github.com/toyxyz/toyxyz_RGBMocap.git # once, if absent
git fetch upstream
git log --oneline 1b0384367fc6b63932c82f1c9c5cb4243a91de2e..upstream/main
git diff --stat 1b0384367fc6b63932c82f1c9c5cb4243a91de2e..upstream/main -- mocap/runtime_registry.py mocap/worker.py mocap/exmr_backend.py mocap/flow_hamer.py mocap/flow_backend.py mocap/video.py mocap/retarget/runtime.py installer/runtime-lock.json requirements/
git switch -c codex/sync-<date> && git merge upstream/main
```

After each merge:

1. **Line endings.** The repository uses `* -text` and hashes file bytes. Never normalize line endings; keep each file's existing CRLF or LF.
2. **Registry and lock.** Keep the two `os.name` guards and `_linux()`. Set `registry_sha256` to the merged file's hash and recompute `id`, which is the SHA-256 of the sorted JSON without `id`. Review any change to `PROFILES`: it can change pins in `setup_ubuntu.py`, `EXTRA_PACKAGES` and receipts.
3. **`PLATFORM_ONLY`.** Each row maps a merged file's hash to the identity upstream's version of that file has under `code_hash`. Delete rows for files you re-edit, then add the new pair:
   ```bash
   sha256sum mocap/flow_hamer.py                         # key
   git show upstream/main:mocap/flow_hamer.py | sha256sum # value, unless upstream already aliases it
   ```
   Never map onto an older identity when upstream changed numerical code; the point is that real changes invalidate caches.
4. **New Windows-only code.** Search upstream changes for `creationflags=0x`, `.exe`, `Scripts/python`, `peak_wset`, `num_handles` and `win32`. Prefer an `os.name` branch in unhashed files. Check whether a file is hashed before editing it (`code_hash`, `sha256(`, `file_stat(`, `inspect.getsource`).
5. **Pins.** The readable pins must still match the lock; a test checks this. New Windows-only packages go in `WINDOWS_ONLY`. If upstream changes Torch pins, review `installer/linux-aarch64.json` and rerun the aarch64 wheel check: `python3 tools/setup_ubuntu.py --plan --machine aarch64`, then `uv pip install --dry-run --no-deps --only-binary :all:` for its pins.
6. **Test constants.** Update `BASE_SHA`, the digests and the identity tables in `tools/tests/test_ubuntu_portability.py` to the new upstream values, and update the baseline above.
7. **Verify.** Run `python3 -m unittest discover -s tools/tests -v` and `python3 tools/check_source.py`, then `./rgbmocap-cli.sh doctor --gpu` and a real FlowHMR + HaMeR clip. Confirm that re-running it reports cached stages, and only then merge the sync branch.

Tests live in `tools/tests/` because the root `tests/` folder is ignored by Git.
