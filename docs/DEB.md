# arm64 Ubuntu online installer

This package targets **Ubuntu 24.04 arm64 / aarch64**, including NVIDIA DGX Spark.
Inference requires a compatible NVIDIA GPU and driver. Other ARM boards without
supported NVIDIA hardware are not validated inference targets.

Install the downloaded file using APT so that system dependencies are resolved:

```bash
sudo apt install ./rgbmocap_<version>_arm64.deb
```

Launch **RGB MoCap** from the application menu, or run `rgbmocap` in a terminal.
The desktop entry opens a terminal to keep installation progress and errors
visible. First launch needs internet access and several GB of download and
installation space. No Python dependencies or models are installed as root, and
the deb's installation does not run network downloads or background services.

The first user launch installs pinned uv 0.13.1 privately, then invokes the
existing `tools/setup_ubuntu.py` for independent shared/GUI, FlowHMR and ExMR
(also used by HaMeR) environments. It stages only the hash-checked public
auxiliary files and compiles the native CPU correction module for that machine.
ARM Torch/torchvision and Qt versions remain those in
`installer/linux-aarch64.json`. See [Ubuntu setup](UBUNTU.md).

To prepare or retry without opening the GUI:

```bash
rgbmocap --setup
rgbmocap --doctor
rgbmocap --doctor --gpu
```

For the optional HandFlow/HTD environments, use
`rgbmocap --setup --profiles all`. Actual GPU validation is a separate step;
dependency installation alone does not establish CUDA correctness.

Models and registered body data are **not included**. Obtain them through Model
Manager under their provider terms. Blender is also separate: install a compatible
Blender 4.5 LTS and set `RGBMOCAP_BLENDER` to its executable when using character
I/O or Blender bundles. FBX/BVH source-motion exports do not require Blender.
Linux support limits for MAMMA, PromptHMR and PEAR remain unchanged.

System files go under `/usr/share/rgbmocap`; `/usr/bin/rgbmocap` is the launcher.
By default, each source payload is extracted into a fresh directory under
`~/.local/share/rgbmocap/installations/<payload-id>`. Its runtimes are installed
there. Models, projects, cache and local preferences use
`~/.local/share/rgbmocap/data`. `XDG_DATA_HOME` changes the parent directory;
an explicit `RGB_MOCAP_DATA` overrides the data location. Existing checkouts,
registered data and projects are never automatically copied or migrated.

A different source payload gets independent runtimes, so an upgrade can require
another download and additional disk space. Old installations and data remain
untouched. Uninstalling the deb removes system files and menu entries; it does
not delete any user's environments, projects, models or exports.

Build from a Git checkout with `dpkg-deb` available:

```bash
python3 tools/build_deb.py
# Optional explicit package version / new output directory:
python3 tools/build_deb.py --version 1.0.1+local.1-1 --output-dir dist
```

The builder includes working-tree bytes for Git-tracked public source plus its
explicit packaging files. It excludes model weights, local environments, caches,
projects, secrets, sample videos and Windows binaries, retains source licenses
and checks importer source resources against their manifest. It creates an
arm64 deb and a SHA-256 file. Existing output files are refused. Build metadata
records the source commit and whether tracked files have local edits.

Validation covers package metadata/content, extraction, source setup planning,
launcher lifecycle and data preservation in temporary folders. A clean APT install,
the complete online runtime download and GPU inference through an installed deb
must be checked separately before claiming a fully tested deployment.
