# RGB MoCap on Ubuntu

This fork adds a Linux route beside the Windows Git/CMD distribution. FlowHMR and its optional HaMeR finger refinement are the priority. Windows behavior, inference and receipts stay as they were; see [what changed](#how-the-linux-route-works).

## Status

| Area | State on Linux |
| --- | --- |
| FlowHMR, HaMeR refinement, GUI, CLI, correction, FBX export | Implemented. Check [validation](#validation) for what has actually been run. |
| ExMR, GVHMR and add-ons (HTD, FootMR, HandFlow) | Routed to their own environments; not yet verified end to end. |
| MAMMA, PromptHMR | Not supported. Their setup builds Windows native modules and their adapters remain Windows-only. |
| PEAR live preview | Not supported. Its shared camera frames use Windows named shared memory. |

### Validation

The CPU portability suite covers routing, pinned environments, cancellation, cache identities and setup planning. Local GPU checks additionally completed a 451-frame FlowHMR + HaMeR + moving-camera + correction pipeline and verified its FBX import in Blender. The GUI starts and loads the resulting project. See [the detailed validation record](UBUNTU_VALIDATION.md) for tested settings and limits.

Clean installation and other models remain unverified. Interactive playback was measured while the desktop was locked and still needs testing on an unlocked desktop.

## Prerequisites

- 64-bit Ubuntu, an NVIDIA GPU with a current driver, and OpenGL 4.3 for the GUI viewport.
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
cp -a extracted/usr/lib/x86_64-linux-gnu/libGLU.so.1* "$OLDPWD/.runtime/lib/"
```

## Set up

```bash
python3 tools/setup_ubuntu.py --plan                  # print exact pins
python3 tools/setup_ubuntu.py                         # .venv, .venv-flow, .venv-exmr
python3 tools/setup_ubuntu.py --stage-assets          # hash-checked auxiliary files, no weights
python3 tools/setup_ubuntu.py --profiles shared --build-correction   # native correction module
./rgbmocap.sh doctor
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

Models are obtained separately, as on Windows: use **Model Manager**, or the `import_model` stage below. Do not copy runtime folders from a Windows install. Model receipts record file size and timestamp, so a copied file that shows as not ready should be imported again rather than having its receipt edited.

## Run the GUI

```bash
./RGBMoCap.sh
```

Keep the terminal open; it mirrors processing messages, as the Windows command window does. The launcher adds `tools/bin` and `.runtime/lib` when they exist, then validates `.venv` instead of reading the Windows managed-install state.

## Command line

The CLI runs the same workers with the same project files and receipts, so the GUI can open its projects and completed stages are reused either way.

```bash
./rgbmocap.sh doctor                # readiness; add --gpu for a real CUDA operation in each environment
./rgbmocap.sh stages                # pipelines and the environment each stage uses
./rgbmocap.sh flow --video clip.mp4 --hamer --dry-run
./rgbmocap.sh flow --video clip.mp4 --start 0 --end 5 --hamer
./rgbmocap.sh flow --video clip.mp4 --camera-interval -1 --correct
./rgbmocap.sh export --project <reported project> --kind initial --output motion.fbx
./rgbmocap.sh stage correct --project <reported project> --result corrected_initial
./rgbmocap.sh stage import_model --model hand_hamer --video /path/to/downloaded.file
```

- `flow` runs `prepare`, `track`, `flow_camera`, the joined `flow_batch` (`flow_features` + `flowhmr`), optional `correct`, then `flow_export`. It finishes by printing the model project path that `export` needs.
- `--hamer` follows the GUI rule: it forces Fast SAM and 4-frame hand sampling, and the fingers run in `.venv-exmr`. `--camera-interval` matches the GUI's camera choices: `0` is Moving/Unknown Auto and `-1` is Static Camera. Other flags include `--no-fast-sam`, `--no-sam-batch`, `--no-gpu-crop`, `--no-hand-stride4`, `--compute-mode low_memory` and `--force`.
- `stage` is the escape hatch: it passes native worker arguments through unchanged and returns the worker's exit code. Common options such as `--json` go before the stage name.
- `Ctrl+C` stops the worker's whole process group, including FFmpeg and the HaMeR child.

Limits compared with the GUI:

- `flow` handles one person with automatic tracking, and there is no **Find person** selection. The GUI remains the route for multiple people.
- The CLI uses the GUI's `ProcessingGuard` every 0.5 seconds for measured low-RAM stopping and stall detection. `--resource-safety off` retains stall detection, adaptive routing and actual OOM recovery. Cancellation terminates the owned process group, including children that ignore graceful termination.
- `export` writes FBX. BVH and Blender bundles use request files from the GUI's export dialog.

## Native correction, FFmpeg and Blender

- **Correction.** Build with `python3 tools/build_motion_correction.py --python .venv/bin/python`, which needs CMake, a C++17 compiler and Git to fetch the pinned pybind11 and Eigen sources. The module goes to `third_party/KimodoMotionCorrection/bin/`, where it is ignored by Git. The Python 3.10 environments share it. It is a local build: the Windows prebuilt binary's validation does not transfer to it, so check results before relying on them. Correction, and only correction, needs this module.
- **FFmpeg.** Discovery order is `tools/bin/<name>`, then `PATH`.
- **Blender 4.5.** Discovery order is `RGBMOCAP_BLENDER` (a file or a folder), `runtime_character/blender`, then `PATH`. It is needed for character import, retargeting and Blender bundles, not for source FBX export.
- **PyOpenGL.** On Linux, ExMR and HaMeR use `PYOPENGL_PLATFORM=egl` unless you set another value. Importing pyrender then needs `libGLU.so.1` and `libEGL.so.1`, even though inference does not render.

## How the Linux route works

| Concern | Implementation | Why |
| --- | --- | --- |
| Routing | `runtime_registry.command/activate` defer to `mocap/linux_runtime.py` when `os.name != 'nt'` | Windows paths stay untouched and `PROFILES` is unchanged, because receipts embed it |
| Worker start | `.venv-*/bin/python -I -X utf8 worker_boot.py <stage>` | Same isolated UTF-8 launch as a managed install |
| Validation | `sys.prefix` must be the profile's venv; exact Python; pinned versions; imports inside its site-packages; no overlay `.pth` | Comparing resolved interpreters cannot tell venvs apart, because they share a base Python |
| Peak memory | `mocap/process_memory.py`: `peak_wset` on Windows, `ru_maxrss` × 1024 on Linux | psutil has no `peak_wset` on Linux |
| Native correction | Linux activation installs a post-import patch for `correction.backend` | `correction.py` is byte-hashed into receipts; see below |
| Process control | `creationflags` falls back to 0; cancel stops descendants through psutil | `taskkill` and Windows flags do not exist on Linux |

### Cache identities

Saved work stays valid because receipts still see baseline code identities:

- `mocap/worker.py`, `mocap/exmr_backend.py` and `mocap/flow_hamer.py` are hashed with `code_hash`. Their Linux routing edits are registered in `PLATFORM_ONLY` in `mocap/resource_cache.py`, which maps the exact new file hash to the baseline identity. This uses the project's mechanism for reviewed non-numerical changes. Only add an alias when no numerical path changed.
- `correction.py`, `flow_correction.py`, `mixamo_data.py`, `handflow.py` and `handflow_engine.py` are hashed directly or stat-checked, so they must stay byte-identical. That is why correction uses an import-time patch and HandFlow's PyOpenGL value is handled by an import guard rather than edits.
- `flow_backend.py` is only identified by selected functions, and the edited reporting code is outside them.

`tools/tests/` pins all of this, and fails if one of these files changes without a reviewed update.

### Windows impact

`runtime_registry.py` is hash-pinned, so adding the guard also updated `registry_sha256` and the derived `id` in `installer/runtime-lock.json`. The profiles and package groups are unchanged. An existing Windows install of this fork must complete one `update.cmd` or `install.cmd` repair: launching before that reports that dependencies do not match. Matching runtime groups are reused rather than downloaded again, but PEAR support is regenerated because its identity includes the registry file. Upstream Windows installs are not affected unless they take this change.

`SOURCE_MANIFEST.json` is upstream's release snapshot and was deliberately not regenerated.

## Upstream sync

**Baseline:** `ef05127a0dbddb3f3ac672a0722dddeae42ea651` (upstream `main` at the time of this port).

Keep the Linux layer small and additive, and merge rather than rebase so the identity reasoning stays reviewable against each upstream version:

```bash
git remote add upstream https://github.com/toyxyz/toyxyz_RGBMocap.git # once, if absent
git fetch upstream
git log --oneline ef05127a0dbddb3f3ac672a0722dddeae42ea651..upstream/main
git diff --stat ef05127a0dbddb3f3ac672a0722dddeae42ea651..upstream/main -- mocap/runtime_registry.py mocap/worker.py mocap/exmr_backend.py mocap/flow_hamer.py mocap/flow_backend.py mocap/video.py mocap/retarget/runtime.py installer/runtime-lock.json requirements/
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
5. **Pins.** The readable pins must still match the lock; a test checks this. New Windows-only packages go in `WINDOWS_ONLY`.
6. **Test constants.** Update `BASE_SHA`, the digests and the identity tables in `tools/tests/test_ubuntu_portability.py` to the new upstream values, and update the baseline above.
7. **Verify.** Run `python3 -m unittest discover -s tools/tests -v` and `python3 tools/check_source.py`, then `./rgbmocap.sh doctor --gpu` and a real FlowHMR + HaMeR clip. Confirm that re-running it reports cached stages, and only then merge the sync branch.

Tests live in `tools/tests/` because the root `tests/` folder is ignored by Git.
