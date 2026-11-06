# Published Ubuntu migration evidence

The [migration and benchmark report](../BENCHMARK_RESULTS_UBUNTU.md) records
the completed amd64 and GB10 tests and their limits. The accompanying
[machine-readable measurements](ubuntu-migration-2026-11-06.json) preserve
the measured repetitions, component medians and numerical disagreements.

Publication uses an explicit field allowlist. Hostnames, network addresses,
login names, absolute local paths, device UUIDs/serials, credentials, raw logs,
model weights, projects, generated motion and private test videos are excluded.
The published measurements use the bundled public sample. Private test data
and operational records remain outside Git.

A new-host deployment is evaluated separately: clone the published branch,
install the independent pinned environments, build the native correction module,
register/copy legitimate model assets, execute CUDA checks, and reconstruct/export
a fresh clip. Copying a previously built environment is not a clean deployment
test. Availability of other optional models and ARM64 Blender remains outside
the validated FlowHMR/HaMeR subset.
