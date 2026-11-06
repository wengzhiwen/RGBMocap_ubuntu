# FlowHMR performance experiment protocol

Requested order: complete GB10 migration validation, establish repeated single-GPU
PRO 4000 and GB10 baselines, then evaluate two-PRO-4000 TP2 and PP2 experiments.
Preparation and CPU correctness tests can precede the migration gate; measured
performance rounds and dual-GPU experiments wait for their prerequisite evidence.

## Migration gate

On GB10, complete a fresh FlowHMR + every-frame HaMeR + correction + source-export
run with Resource Safety enabled and adequate free unified RAM. Verify finite
outputs, expected frame counts, actual valid HaMeR predictions, exact retention
of body/wrist/trajectory/shape/native contacts by finger refinement, selected-result
receipts, export and a repeated cached run. Basic CUDA operations, tracking and
native CPU correction alone do not satisfy this gate. No unrelated service is
stopped and no memory cap is disabled to force the test to fit.

Current GB10 status and its external memory constraint are in
[SPARK_VALIDATION.md](SPARK_VALIDATION.md). No formal performance baseline has
been measured yet.

## Fixed inputs and scopes

Use the same canonical 30 fps working video, tracking/conditioning inputs where
applicable, model checkpoints, seed 0, 20 ODE steps, FP32 FlowHMR generation,
256-query MATH attention, full temporal context and saved camera/SAM/HaMeR options.
Retain the original model-native precision of SAM and HaMeR. Save file hashes,
decoded-frame spot checks, resolution, frame count and valid-hand counts.

A public bundled 23-second 1920×1080/60 fps source can supply separate 60-, 451-
and 690-frame 30 fps cases after one canonical preparation. Longer private cases
can be added with their own hashes. Different clip lengths are separate inputs;
no experiment divides a motion sequence into independent temporal windows.

Measure these scopes separately:

- Fresh-process reconstruction, including tracking, camera, SAM, generator,
  optional HaMeR, correction and preview publication. Report preparation and
  export separately if excluded. Fresh inference must not be a cache hit.
- Component loading and cold initialization, separated from resident-model work.
- Warm generation from identical cached conditioning tensors. This isolates the
  generator and must never be presented as full-video throughput.
- Warm SAM/HaMeR component timings only when their actual inputs and work counts
  are held constant. These do not replace the complete reconstruction baseline.

Use one excluded warm-up and five measured repetitions per case/mode by default.
Keep every raw repetition, failures and exclusions; report median and range.
Do not clear system caches, overclock, change power limits or stop other people's
jobs. Label OS-cache state and competing workloads. The primary baseline needs
an otherwise available GPU; contended runs are separate experiments.

## Hardware and software comparability

The validated amd64 installation uses Flow Torch 2.7.1+cu128 and HaMeR Torch
2.8.0+cu128; Spark uses separate Torch 2.9.1+cu130 environments. Their stock
comparison measures the deployed systems, including software and CPU differences.
It cannot isolate GPU hardware speed. A matched Torch 2.9.1/CUDA 13 experiment
may use separate benchmark environments; production pins stay unchanged.

Record GPU names, driver, clocks/temperature/power observations, CPU, available
RAM, Python/Torch/CUDA versions, precision, git/source identities and all options.
Both PRO 4000 GPUs support P2P access; their topology is PHB (PCIe host bridge),
with no NVLink. Measure communication overhead rather than assuming TP2 helps.

## Dual-GPU experiments

TP2 must actually partition model tensor computation (for example attention
heads and MLP widths) with the necessary reductions. PP2 must partition model
blocks/stages between devices while retaining the full sequence. Independent
jobs on two cards are data parallel throughput, a different experiment.

Before timing, compare each experimental mode with its single-GPU reference
using identical weights, inputs, runtime and seeds. Report translation, shape,
contact and rotation differences, plus world-joint disagreement where available.
Float32 reduction ordering can differ; describe observed differences and never
claim ground-truth accuracy or bit equivalence without evidence.

For each repetition synchronize all participating CUDA devices around the timed
scope. Retain per-device allocated and reserved memory peaks, process RSS, elapsed
wall time and the complete input/output identity. Do not add GB10 CPU RSS and
CUDA allocations as if they were disjoint physical memory.

PP2 on one clip can have transfer overhead without useful overlap, because each
ODE step depends on the preceding full output. If separate-clip microbatches are
also tested, report their throughput and single-clip latency independently.

Experimental adapters and outputs remain separate from production inference and
user result generations until correctness and performance evidence support any
promotion. Raw data and generated motion stay under ignored `experiments/`;
only curated methodology and aggregate findings belong in public documentation.

## Reproducible commands

Use new output folders for every invocation. The launcher preserves completed
outputs and refuses an existing experiment folder. Run it through the installed
shared Python; workers activate the separate Flow runtime. Resource Safety and
process-tree cancellation remain enabled.

On Spark, once sufficient memory is available:

```bash
.venv/bin/python tools/benchmark_flow.py migration \
  --output experiments/bench-gb10-migration
```

This runs 60 public-sample frames through fresh tracking, static camera, Fast SAM,
20-step FlowHMR, every-frame HaMeR, correction, selected source FBX and a cached
rerun. The gate checks actual native correction, finite motion, both-hand valid
counts, preserved body/fingers, output receipts and cache reuse. Its conservative
32 GiB free-memory admission threshold is for this validation run, not a measured
minimum system requirement. It defers without creating a project or disturbing
existing work when that budget is unavailable. Moving-camera and longer-clip
validation remain separate from this short static-camera gate.

After it passes, copy `migration.json` and the exact `input.mp4`, `flow_tokens.pt`
and `flow_camera.npz` conditioning bytes into new private experiment folders on
both machines. Do not regenerate those tensors independently for a generator
hardware comparison. Run on each machine:

```bash
.venv/bin/python tools/benchmark_flow.py generator \
  --migration-report experiments/bench-gb10-migration/migration.json \
  --project experiments/bench-conditioning \
  --output experiments/bench-single
.venv/bin/python tools/benchmark_flow.py pipeline \
  --migration-report experiments/bench-gb10-migration/migration.json \
  --frames 60 --output experiments/bench-pipeline
```

The pipeline scope includes video preparation and preview publication, unlike
the generator scope. Each pipeline round uses a new project and refuses cached
inference. CPU verification/hashing is outside its timer. The working-video
bytes and decoded pixels must remain identical between repetitions. Compare
decoded pixels, settings and runtime versions across platforms before interpreting
the complete-pipeline comparison. Longer cases use `--frames 451` or `690`;
`--camera-interval 0` adds moving-camera estimation as a separate case.

Copy both successful **generator** `result.json` files back to the PRO 4000 host.
For each mode `tp2` and `pp2`, run a new experiment:

```bash
.venv/bin/python tools/benchmark_flow.py generator --mode tp2 \
  --migration-report experiments/bench-gb10-migration/migration.json \
  --project experiments/bench-conditioning \
  --baselines experiments/bench-pro-single/result.json experiments/bench-gb10-single/result.json \
  --reference experiments/bench-pro-single/reference.pt \
  --output experiments/bench-tp2
```

The tool requires five measured single-GPU rounds plus an excluded warm-up on
both platforms, identical weight/code/conditioning hashes and the actual PRO 4000
reference artifact. It checks the dual-card runtime against that reference's
runtime. Contention checks refuse other compute jobs on the selected cards;
they never stop those jobs. `CUDA_VISIBLE_DEVICES` can select/reorder cards.
Use `summary` followed by report paths to print medians and ranges. Keep raw
reports and logs, including failures; no production inference route enables these
experimental adapters.

## Preparation evidence — 2026-11-06

There are **no GB10 complete-pipeline passes, GPU TP2/PP2 passes or performance
numbers** yet. Spark currently has about 7.7 GiB available because of unrelated
work, which remains untouched.

Real CPU preparation checks on the amd64 host used the full 460,397,547-parameter
checkpoint, identical cached 16-frame conditioning, seed 0 and 20 ODE steps:

| CPU mode | Maximum translation / shape / rotation-6D / contacts tensor difference |
| --- | --- |
| PP2, two CPU placements | 0 / 0 / 0 / 0 |
| TP2, two Gloo processes | 1.20e-7 / 1.20e-7 / 3.58e-7 / 2.39e-7 |

Each TP2 rank retained 309,316,587 parameters. Conditioning encoders/refiner,
modulation, norms and output head are replicated; main attention heads and MLP
channels are actually partitioned. This establishes this CPU fixture's numerical
agreement, not CUDA/NCCL compatibility, memory reduction or speedup. Additional
small-model tests cover padding, start tokens and long skips. The actual existing
451-frame project also passed the new artifact-preservation validator on CPU;
that is not a new inference run or a GB10 migration pass.
