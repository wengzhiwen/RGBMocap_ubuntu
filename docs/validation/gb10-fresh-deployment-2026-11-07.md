# Fresh deployment on a second GB10 — 2026-11-07

The published `codex/ubuntu-port` branch deployed successfully on a second
NVIDIA GB10 running Ubuntu Desktop. Fresh FlowHMR + HaMeR reconstruction,
moving-camera estimation, native correction and source FBX export passed
without changing the application source. This establishes deployment readiness
for that subset, rather than parity with every Windows model and feature.
The [allowlisted results](gb10-fresh-deployment-2026-11-07.json) accompany this
report; the earlier [performance results](../BENCHMARK_RESULTS_UBUNTU.md) remain
separate.

## Installation exercised

The target cloned the published GitHub branch over HTTPS into a new application
directory. Acceptance ran at `a176d61`; its only change after the initial clone
at `6aa2684` was ignoring the private `test_media/` directory. Subsequent report
updates do not change the tested implementation.

- Target: Ubuntu 24.04.5 LTS, aarch64, NVIDIA GB10, driver 580.178.04.
- System dependencies were installed through apt, including FFmpeg, Qt/OpenGL
  libraries, Python venv support, CMake and the C++ compiler. The existing driver
  was retained. This was a fresh application deployment, not a clean OS install.
- A fresh bootstrap installed uv 0.13.1. The public command
  `python3 tools/setup_ubuntu.py --stage-assets --build-correction` created the
  independent pinned shared, Flow and ExMR environments and built the native
  correction extension on the target. No environment or compiled correction
  binary was copied from the earlier machines.
- Shared/Flow use Python 3.10.22; ExMR/HaMeR uses Python 3.12.10. All three use
  the architecture-specific Torch 2.9.1+cu130 pins, keeping the Windows/amd64
  profiles separate. Setup staged and verified 25 auxiliary files.
- Legitimately held model assets were copied separately. CUDA matrix
  multiplication and torchvision CUDA NMS actually passed in all three profiles.
  The GPU doctor reported 24 OK, five warnings and zero failures.
- The CPU suite ran 183 tests: 182 passed, one skipped, zero failures.

## Fresh functional acceptance

Each case reconstructed into a new project instead of reusing transferred
inference outputs. HaMeR ran on every frame; Resource Safety stayed enabled.
The default Fast SAM route retained its four-frame hand-feature sampling and
interpolation; that is separate from the every-frame HaMeR refinement. Automatic
VGGT sampling used 116/690 and 114/451 frames for the two moving-camera cases.

| Case | Camera | Frames | Valid left / right hands | Result |
| --- | --- | ---: | ---: | --- |
| Public short sample | Static | 60 | 60 / 60 | Passed, including cached rerun |
| Public complete sample | Moving | 690 | 690 / 690 | Passed |
| Private video under a Unicode path | Moving | 451 | 451 / 409 | Passed, including missing-hand fallback checks |

The shared validation worker checked completed stages and their output receipts,
absence of pending generations, finite motion tensors and expected frame counts.
HaMeR preserved body/root/wrist motion and the other original body fields exactly;
frames without a valid hand crop retained the original fingers. Correction
preserved finger rotations. Both original and corrected selected BVH results were
available, and correction reports recorded execution of the newly built native
backend. These are functional and preservation checks, not ground-truth motion
accuracy measurements.

The 60-frame case used the public migration command:

```bash
.venv/bin/python tools/benchmark_flow.py migration --output experiments/deployment-acceptance
```

The complete public moving-camera case used:

```bash
./rgbmocap.sh flow --video docs/samples/motion-capture.mp4 \
  --project projects/deployment-moving --end 23 --camera-interval 0 \
  --hamer --correct --resource-safety on --json
./rgbmocap.sh export --project projects/deployment-moving/flowhmr \
  --kind corrected_initial --output experiments/moving.fbx --resource-safety on
```

Source FBX export succeeded for all three cases. The new target's 451- and
690-frame FBX files were additionally imported in Blender 4.5.4 LTS on the amd64
validation host: each contained one armature, 65 bones including terminal bones,
30 fps and the exact 1–451 / 1–690 action range. Blender itself was not installed
or validated on this GB10.

The native desktop GUI started, opened the fresh 60-frame project and rendered
the video and native mesh/fingers in a saved screenshot. The startup worker
exited successfully, resolved the new installation's own directories and did not
import inference into the GUI process. The desktop was locked, so interactive
GB10 playback was not tested. The SSH doctor's missing-DISPLAY warning is
separate from this successful native desktop startup test.

## Private test material transferred

The transfer covered the local migration `projects/` and `experiments/`, bundled
sample videos, and discovered test videos under the original Windows project
tree and additional sample/private-video locations. The inventory contains
89 video files representing 29 unique contents, including duplicate working
videos and previews. An additional 30 model files were copied. All 119 files
matched source sizes and SHA-256 hashes on the target; originals were retained.
The count excludes outputs generated by the new acceptance runs.

A private `test_media/README.md` on the target lists every transferred video,
with `transfer-manifest.json` and `INTEGRITY.json` recording verification.
Archived projects retain their original provenance and may contain old absolute
source/cache paths. New reconstruction should select a transferred video and
create a fresh project. Videos, model assets, results, manifests, raw logs,
network/login details, SSH key details and private filenames are excluded from
Git. SSH public-key authentication was verified separately; no private key was
transferred.

## Remaining limits

- FlowHMR + HaMeR is the verified deployment subset. Full ExMR/GVHMR workflows,
  optional HandFlow/HTD environments and other add-ons are not certified here.
  MAMMA, PromptHMR and PEAR retain the limitations in [UBUNTU.md](../UBUNTU.md).
- ExMR moving-camera SimpleVO is unavailable with these aarch64 pins because
  its pycolmap wheel is unavailable. This does not affect FlowHMR's tested
  moving-camera route.
- Target-side Blender character import, retargeting and Blender bundles remain
  unverified; source FBX export works independently of Blender.
- Torch emits a compatibility warning for sm_121 despite these successful actual
  CUDA operations and end-to-end runs. Import success alone was not used as
  evidence of GPU compatibility.
- These are functional acceptance runs, not new repeated timing baselines.
