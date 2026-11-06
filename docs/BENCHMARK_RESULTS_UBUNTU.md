# Ubuntu FlowHMR measurements — 2026-11-06

The GB10 migration gate passed before timing began. Both systems then ran one
excluded warm-up and five measured repetitions per input/scope. These are local
deployment measurements, including different CPUs, operating systems and pinned
Torch/CUDA versions; they do not isolate GPU hardware performance.

## Systems and fixed work

| | PRO 4000 host | GB10 host |
| --- | --- | --- |
| GPU | NVIDIA RTX PRO 4000 Blackwell; one card for baselines, two for TP2/PP2 | NVIDIA GB10 |
| CPU | AMD Ryzen 7 9850X3D, 16 logical processors | 20 ARM cores, Cortex-X925/A725 |
| OS | Ubuntu 26.04.1 LTS, x86_64 | Ubuntu 24.04.5 LTS, aarch64 |
| Driver | 595.99.02 | 580.178.04 |
| Flow environment | Python 3.10.22, Torch 2.7.1+cu128 | Python 3.10.22, Torch 2.9.1+cu130 |
| HaMeR environment | Python 3.12.10, Torch 2.8.0+cu128 | Python 3.12.10, Torch 2.9.1+cu130 |

The PRO cards have a PHB PCIe connection and no NVLink. The existing Spark GPU
service was stopped gracefully with explicit user authorization before the full
validation and measurements. Resource Safety stayed enabled. No power limits,
clocks, kernel/BIOS settings or production runtime pins were changed.

Both cases use the bundled public `docs/samples/motion-capture.mp4`: 60 prepared
frames (2 seconds) and 690 prepared frames (23 seconds), 1920×1080 at 30 fps.
FlowHMR keeps seed 0, CFG 1.0, 20 ODE steps, FP32, 256-query MATH attention and
the entire temporal context. No temporal windows are reconstructed independently.
Fast SAM uses 16-frame batches, GPU hand crops and four-frame hand features;
HaMeR uses its original FP16 network, eight hand crops per batch and every frame.
Default contact correction and preview publication remain enabled.

For generator comparisons the exact same camera and SAM conditioning files were
copied between hosts and content-hashed. Fresh pipeline rounds copy only the
canonical video and preparation metadata into a new project, with all inference
state cleared. Original source-video conversion is excluded; CLI preparation,
worker startup and result previews are included. OS caches were not cleared.

## Complete single-GPU pipelines

Elapsed wall time in seconds; median [minimum–maximum], five measured rounds.

| Input / camera | PRO 4000 | GB10 |
| --- | ---: | ---: |
| 60 frames, explicit static camera; no VGGT estimation | 28.045 [27.923–28.134] | 40.660 [40.343–40.746] |
| 690 frames, moving/unknown Auto; VGGT included | 98.277 [98.223–98.306] | 202.237 [195.556–202.470] |

Every repetition performs fresh tracking, SAM, FlowHMR, every-frame HaMeR,
correction and source previews. Inference cache hits are refused. These timings
exclude subsequent external FBX/Blender export and CPU validation/hash checks.
The static case does not measure camera estimation. The moving case uses Auto's
six-frame sampling interval, with 116 camera samples.

Internal component medians for the 690-frame case, in seconds:

| Component | PRO 4000 | GB10 |
| --- | ---: | ---: |
| Tracking | 8.936 | 16.267 |
| Moving camera | 10.351 | 46.726 |
| SAM features | 54.838 | 103.480 |
| Motion stage, including HaMeR | 14.815 | 23.163 |
| Correction | 0.865 | 1.529 |
| Source preview publication | 3.600 | 5.445 |

The complete wall timer additionally includes CLI preparation and worker/process
overhead. Component medians are measured inside workers; they are not an additive
breakdown of one repetition. SAM and camera estimation dominate this moving case.

## Resident generator only

Elapsed seconds; median [minimum–maximum], five measured rounds. Tracking, SAM,
camera estimation, HaMeR, correction, export and process/model loading are excluded.
The table uses the revised external-memory-monitor measurements (`*-v2` artifacts).

| Mode | 60 frames | 690 frames |
| --- | ---: | ---: |
| PRO 4000 ×1 | 0.2555 [0.2541–0.2560] | 2.0714 [2.0466–2.0915] |
| GB10 ×1 | 0.5032 [0.5010–0.5074] | 2.3705 [2.3535–2.3819] |
| PRO 4000 ×2, TP2, NCCL over local TCP | 0.2819 [0.2765–0.2865] | 1.4961 [1.4939–1.4990] |
| PRO 4000 ×2, PP2, explicit CPU staging | 0.2683 [0.2647–0.2689] | 1.9185 [1.9140–1.9230] |

On the 690-frame case TP2 achieved 1.38× the single-PRO generator speed and PP2
1.08×. Both were slower on the short case. PP2 processes one complete clip with
no microbatch overlap; these results do not establish why its small long-case
gain occurred. An additional single-PRO confirmation run measured 2.0760 seconds
(range 2.0453–2.0942) with the initial monitoring implementation.

These adapters partition the generator, not the complete reconstruction. SAM,
camera estimation, HaMeR and process/loading overhead remain substantial. No
end-to-end speedup using the experimental adapters has been measured, and the
production CLI/GUI still uses its original single-device inference path.

## Numerical agreement and memory

PP2 with CPU staging matched the single-PRO reference exactly for all compared
output tensors and world FK joints on both inputs. For TP2 with NCCL/TCP, the
690-frame maxima across both ranks and every round were:

| Quantity | Maximum disagreement |
| --- | ---: |
| Translation vector norm | 7.38e-7 m |
| World FK joint position | 1.18e-6 m |
| Local joint rotation | 0.0000869 degrees |
| Shape coefficient | 5.07e-7 |
| Native contact value | 1.14e-6 |

These are differences from the single-device reference, not ground-truth errors.
The checks also cover rotation-6D, local predicted joints, smooth root and pelvis.
Full-context execution on 690 frames crosses multiple query chunks.

690-frame steady-state PyTorch peaks in GiB, after an excluded warm-up:

| Mode | Allocated, GPU 0 / GPU 1 | Reserved, GPU 0 / GPU 1 |
| --- | ---: | ---: |
| Single PRO | 1.990 / — | 2.074 / — |
| TP2, NCCL/TCP | 1.381 / 1.381 | 1.527 / 1.527 |
| PP2, CPU staging | 1.073 / 1.013 | 1.117 / 1.086 |

Loading peaks are recorded separately. Each TP2 rank first loads the whole model
before sharding, reaching about 1.81 GiB allocated per card. PP2 similarly loads
the full model on the first card before moving blocks. Lower steady-state usage
does not prove the current loader fits a card with only the steady-state capacity.
CUDA contexts, libraries and NCCL allocations can fall outside PyTorch statistics.
RSS sums can count shared pages more than once; on GB10 RSS and CUDA allocations
can refer to the same physical RAM and must not be added.

Pipeline stage memory events retain the production worker's cumulative scope;
the connected generator worker can retain SAM's earlier peak. HaMeR is a separate
child, so those stage events are not complete simultaneous job-memory figures.

## Local peer-transfer failure and tested workarounds

Although CUDA reports peer access available in both directions, a real FP32
1,048,576-element copy from logical GPU 0 to GPU 1 failed exact comparison,
with maximum absolute difference 1,048,575. The reverse copy and both CPU-staged
directions were exact in that initial probe. The later standalone
[peer-copy diagnostic](CUDA_PEER_CHECK.md) completed all eight cases: all direct
copies in both directions/layouts failed, while all CPU-staged copies were exact.
Direct PP2 consequently failed reference comparison and
was excluded. Default/IPC NCCL probes timed out or reported illegal memory access;
they contribute no performance numbers.

The kernel reports a translated IOMMU domain and both GPU groups report DMA-FQ.
[NVIDIA documents silent corruption risks for Linux bare-metal IOMMU PCIe peer
transfers](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/gpu_troubleshooting.html#linux-bare-metal-iommu-and-pcie-peer-to-peer).
This is a strong diagnostic lead, not proof of the exclusive cause. No kernel,
BIOS, PCI configuration or driver change was performed.

The successful TP2 run explicitly used:

```bash
NCCL_SOCKET_IFNAME=lo NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 \
NCCL_CUMEM_HOST_ENABLE=0 NCCL_CUMEM_ENABLE=0 NCCL_IB_DISABLE=1
```

Those settings are confined to the experiment. PP2 used `--peer-transport host`;
both its weight placement and dynamic cross-device tensors go through CPU memory.
An additional explicit Gloo/CPU reduction path was also GPU-validated, but its
initial timing results are not included in the final table.

## Evidence and limits

The migration checks validate finite results, actual valid HaMeR predictions,
body/root/wrist/shape/trajectory/contact preservation, finger preservation through
correction, native correction execution, current selected results and source FBX.
Both hands were valid for every frame in the public 60- and 690-frame fixtures.
An architecture comparison or successful import does not establish motion quality.

The initial generator measurements used an in-process RSS sampler that could
contend for Python's GIL asymmetrically. An independent Claude Opus 5.5 review
identified that concern. All generator table entries were therefore remeasured
with monitoring in the separate launcher. The raw initial results and failures
were retained. Loading peaks, monitor CPU time, sampled per-round process/tree RSS,
code hashes, physical-device selection and numerical differences accompany the
revised reports. Gate/model/code and reference hashes are checked before timing.

The review also identified inactive zero-initialized gates in the small random
test fixtures. They now explicitly activate the gates, verify that padding changes
the prediction and require an intentionally corrupted reduction to fail parity.
Real learned-weight CPU/GPU checks remain separate evidence.

All six fresh projects per case/host (warm-up plus five measured rounds) were
audited afterward. Working-video bytes and copied previews remained identical
to their canonical templates; no nested re-encoded clip appeared. Actual logs
confirmed matching 16-frame SAM and eight-hand HaMeR batches, Automatic resource
mode and no cached inference or Low Memory retry.

An additional decoder audit found an architecture-dependent limit: the encoded
H.264 files are byte-identical across hosts, but decoded BGR hashes differ. Within
each host, shared OpenCV 4.10 and the Flow/HaMeR OpenCV 4.11 environments produced
identical full-clip decoded hashes. Across hosts, five sampled frames per case
showed maximum channel differences of 3/255 (60 frames) and 2/255 (690 frames),
with mean absolute differences 1.049/255 and 1.064/255. Those sample maxima are
not bounds on every frame. The complete-pipeline comparison therefore includes
decoder/conversion implementation differences. It is not a bit-identical-pixel
hardware benchmark. The resident-generator table uses identical cached tensors
and is unaffected by video decoding.

With those identical cached tensors, single-PRO versus GB10 generator outputs
also passed the numerical comparison. On 690 frames maximum world FK disagreement
was 2.86e-6 m and local rotation disagreement 0.000431 degrees. These are
deployment-to-deployment differences on one fixture, not ground-truth accuracy.

The actual GB10 60- and 690-frame corrected FBX files were imported in Blender
4.5.4 on the amd64 host: 65 bones including terminal bones, 30 fps, and actions
spanning 1–60 and 1–690 respectively. This structural import check does not
establish Blender availability on ARM64 or full pose-matrix equivalence.

Raw reports, logs and generated media stay private under
`experiments/bench-20261106/` and `.setup/bench/`. Reproduction commands and exact
scope definitions are in [BENCHMARK_PROTOCOL.md](BENCHMARK_PROTOCOL.md).
