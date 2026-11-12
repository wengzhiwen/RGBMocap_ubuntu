# Ubuntu arm64 installer

Download the actual installation package:

- [RGB MoCap arm64 online installer](rgbmocap_1.0.1+git74b79bd4-1_arm64.deb)
- [SHA-256 checksum](rgbmocap_1.0.1+git74b79bd4-1_arm64.deb.sha256)

Built from source commit `74b79bd4294c56661eefc6efb4de2f991cbba710`.
For Ubuntu 24.04 arm64 / aarch64, including NVIDIA DGX Spark.

```bash
sha256sum -c rgbmocap_1.0.1+git74b79bd4-1_arm64.deb.sha256
sudo apt install ./rgbmocap_1.0.1+git74b79bd4-1_arm64.deb
rgbmocap
```

First launch downloads independent pinned Python/AI environments. Model weights
are separate. See [installation details](../../docs/DEB.md).

Checked: 12 bootstrap/packaging tests, package and payload checksums, safe source
extraction, current CLI/GUI entry points, arm64 setup planning, desktop entry,
and APT dependency simulation. A clean online installation and GPU inference
through this deb have not been performed.
