# FAH and FAB on Ubuntu

FAH is an optional FlowHMR finger refinement. It infers fingers from the
selected actor's hand boxes and uses FlowHMR's wrist pose. When a hand cannot
be fitted, the frame uses the FlowHMR finger pose. Selecting FAH enables Fast
SAM and four-frame SAM hand sampling. FAH processes each frame in the clip.

FAB is an optional body model for fixed-camera video with one primary person.
It adapts the fitted SMPL body to the app's SMPL-X skeleton with a relaxed
finger pose. The placement is approximate and assumes a level camera.

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
  the SAM 3D Body config and MHR asset, and a licensed `MANO_RIGHT.pkl`
  converted as described upstream. Defaults are under `~/.cache/fah`.
  [FAH asset setup](https://github.com/noahcao/FAH/blob/da03aabcd99cf61c7ea179c1ddca84f75be14962/README.md#install)
- **FAB** uses the FAB-L bundle in `weights/fab-l/`, the shared SMPL
  fitter, the SAM 3D Body config and MHR asset, the standard SMPL neutral
  model and its conversion assets, and the G1 URDF required by the upstream
  predictor. Defaults are under `~/.cache/fab`. Put `SMPL_NEUTRAL.pkl` and
  `J_regressor_h36m.npy` in `body_models/smpl/`, and `mhr2smpl_mapping.npz`
  and `mhr_face_mask.ply` in `body_models/mhr/`. Prepare the app's SMPL-X
  Neutral and person detector in Model Manager.
  [FAB asset setup](https://github.com/noahcao/FAB/blob/ec65caf40632b4d36eb48f58ab45ea56d3cba21a/README.md#install)

## Run

In the GUI, select **FlowHMR** and enable **FAH Finger Refinement**, or select
**FAB** for body reconstruction. Use **Reconstruct 3D** and review the selected
result before exporting. **Model Manager** reports the required files.

```bash
./rgbmocap-cli.sh flow --video clip.mp4 --fah --dry-run
./rgbmocap-cli.sh flow --video clip.mp4 --fah
./rgbmocap-cli.sh fab --video clip.mp4 --dry-run
./rgbmocap-cli.sh fab --video clip.mp4
./rgbmocap-cli.sh export --project /path/to/reported/project --output motion.fbx
```
