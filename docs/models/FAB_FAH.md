# FAH and FAB on Ubuntu

FAH is an optional finger refinement for **FlowHMR** and **FAB**. Each model
has its own **FAH Finger Refinement** choice; selecting it for one model leaves
the other unchanged.

- **FlowHMR + FAH** infers fingers from the selected actor's hand boxes and
  uses FlowHMR's wrist pose. When a hand cannot be fitted, the frame uses the
  FlowHMR finger pose. Selecting FAH enables Fast SAM and four-frame SAM hand
  sampling.
- **FAB + FAH** infers fingers from square hand crops around FAB's own
  predicted left and right hand points, using the saved camera intrinsics.
  Only the 15 local finger rotations of each hand are taken from FAH and
  published on FAB's body; FAH's own hand placement and global orientation are
  not used. The FAB body, wrists, trajectory, shape and contacts are kept. A
  hand without a usable crop (smaller than 16 pixels, mostly outside the
  frame, or larger than a share of the body box) keeps the relaxed default
  fingers.

FAH processes each frame in the clip. It needs its separate runtime and files
for either model, and its accuracy is not guaranteed.

FAB is an optional body model for fixed-camera video with one primary person.
It adapts the fitted SMPL body to the app's SMPL-X skeleton with a relaxed
finger pose. The placement is approximate and assumes a level camera. The
adaptation fits one constant SMPL-X shape to the clip, so the published
proportions and the wrist placement can differ from FAB's own fit; FAH fingers
attach at those wrists.

The released weights are licensed for research and other non-commercial use.
Review the asset terms before obtaining the model files. [FAH weight
licence](https://github.com/noahcao/FAH/blob/da03aabcd99cf61c7ea179c1ddca84f75be14962/WEIGHTS_LICENSE)
and [FAB weight licence](https://github.com/noahcao/FAB/blob/ec65caf40632b4d36eb48f58ab45ea56d3cba21a/WEIGHTS_LICENSE).

## Runtime setup

Complete the [Ubuntu base setup](../UBUNTU.md#set-up), then use a separate
environment for each model you want to try. The example uses PyTorch 2.11.0
with the CUDA 12.8 wheels on x86_64 and CUDA 13.0 wheels on aarch64. Match the
PyTorch wheel to a compatible NVIDIA driver.

```bash
optional_tool=fah                       # use fab for FAB
case "$optional_tool" in
  fah) optional_source=https://github.com/noahcao/FAH.git@da03aabcd99cf61c7ea179c1ddca84f75be14962 ;;
  fab) optional_source=https://github.com/noahcao/FAB.git@ec65caf40632b4d36eb48f58ab45ea56d3cba21a ;;
esac
optional_cuda=cu128
if [ "$(uname -m)" = aarch64 ]; then optional_cuda=cu130; fi
uv venv --python 3.12 ".venv-$optional_tool"
optional_python=".venv-$optional_tool/bin/python"
uv pip install --python "$optional_python" --index-url "https://download.pytorch.org/whl/$optional_cuda" \
  "torch==2.11.0+$optional_cuda" "torchvision==0.26.0+$optional_cuda"
uv pip install --python "$optional_python" "$optional_tool @ git+$optional_source"
```

Use the release files identified by the upstream `SHA256SUMS`. The weight
revisions for this setup are
`c2e67a5779c104f85c33c147e8d4c6a8ebdc28b4` for FAH and
`f2e204f280aa27944b98fbf783f618a4bea2c223` for FAB.

- **FAH** uses the FAH-L bundle in `weights/fah/`, the shared `detector.pt`,
  the SAM 3D Body config and MHR asset, and the licensed `MANO_RIGHT.pkl` and
  `MANO_LEFT.pkl`, each converted as described upstream. Defaults are under
  `~/.cache/fah`.
  [FAH asset setup](https://github.com/noahcao/FAH/blob/da03aabcd99cf61c7ea179c1ddca84f75be14962/README.md#install)
- **FAB** uses the FAB-L bundle in `weights/fab-l/`, the shared SMPL
  fitter, the SAM 3D Body config and MHR asset, the standard SMPL neutral
  model and its conversion assets, and the G1 URDF required by the upstream
  predictor. Defaults are under `~/.cache/fab`. Put `SMPL_NEUTRAL.pkl` and
  `J_regressor_h36m.npy` in `body_models/smpl/`, and `mhr2smpl_mapping.npz`
  and `mhr_face_mask.ply` in `body_models/mhr/`. Prepare the app's SMPL-X
  Neutral and person detector in Model Manager.
  [FAB asset setup](https://github.com/noahcao/FAB/blob/ec65caf40632b4d36eb48f58ab45ea56d3cba21a/README.md#install)

## Import the FAH hand models

Select **FlowHMR** or **FAB**, enable that model's **FAH Finger Refinement**,
and open **Model Manager**. Click **Import FAH MANO…** and select the
`MANO_RIGHT.pkl` produced by the upstream MANO converter, then repeat for
`MANO_LEFT.pkl`. Upstream FAH loads both hands: it fits the right-hand model
to a mirrored crop of the left hand. The import uses `MANO_DIR` when set.
Otherwise it uses the `mano/` folder under `FAH_HOME`, which defaults to
`~/.cache/fah`. The setup panel shows the paths FAH is using and lists each
file as ready or missing.

## Run

In the GUI, select **FlowHMR** or **FAB** and enable that model's **FAH Finger
Refinement** for the finger pass. Select **FAB** without it for body
reconstruction with the relaxed finger pose. Use **Reconstruct 3D** and review
the selected result before exporting. **Model Manager** reports the required
files. Turning FAH off or on again reuses FAB's committed body capture and
repeats only the finger pass.

```bash
./rgbmocap-cli.sh flow --video clip.mp4 --fah --dry-run
./rgbmocap-cli.sh flow --video clip.mp4 --fah
./rgbmocap-cli.sh fab --video clip.mp4 --fah --dry-run
./rgbmocap-cli.sh fab --video clip.mp4              # FAB body, relaxed fingers
./rgbmocap-cli.sh fab --video clip.mp4 --fah        # FAB body with FAH fingers
./rgbmocap-cli.sh export --project /path/to/reported/project --output motion.fbx
```

`--dry-run` prints the planned stages and any missing requirements and runs
nothing. `--no-fah` turns the pass off. Rerunning the same `fab` command with
the other choice reuses the committed body capture.

## What a result used

A FAB project records what its committed result used in `fab_result.json`:
`hand_pose_policy` is `fah-fingers-on-fab-body-v1` when the finger pass fitted
at least one hand and `registered-neutral-untracked-v1` otherwise, including
when no hand had a usable crop or the pass fitted none, and `fingers` counts
the frames fitted per hand. The saved `fab_options` only keeps the draft choice.
`export.json` names the finger motion of the exported result the same way.
