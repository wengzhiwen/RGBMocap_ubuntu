# Batch processing in the GUI

Use **Batch Processing > Select Videos...** to queue independent video files.
The first video opens in the normal workspace. Configure the model, resource
mode, correction and other reconstruction settings there. The reconstruction
button displays **Reconstruct 3D (N)** with the number of queued videos.

Use **Batch Processing > Setup Export...** to choose an output folder and a
source-motion format: FBX, BVH or a portable Blender project. BVH supports the
existing Generic / Blender and 3ds Max Biped profiles. The existing optional
leading T-pose and Blender camera settings are available. Check **Export Preview
Video** to also save the input video beside the current 3D viewport as MP4.
This uses the existing preview exporter, including its frame pairing and audio.
FBX compatibility for Maya, placement at the origin and symmetrical rest pose
use the same options as individual source exports and are retained by batch setup.

On Linux hosts with `nvidia-smi`, the setup dialog also lists **Parallel GPUs**.
Leave one or none checked for the ordinary sequential batch. Check two or more
to process one video per GPU at the same time; the number of checked GPUs is
the number of videos running concurrently. See *Parallel processing with
multiple GPUs* below. With two or more GPUs checked, **Export Preview Video**
stays available: after the parallel reconstruction finishes, the main window
renders the preview videos one after another.

**Find people first; check the first N people** (default on, limit 3) runs the
ordinary Find People discovery on every video before reconstruction and checks
the first N found people automatically, in first-appearance order. Each checked
person is then reconstructed and exported separately, so a two-actor video
produces one motion file per actor. The default tracker is used; the SAM 3.1
tracker stays interactive and is not used by batch. The option applies to ExMR,
FlowHMR and GVHMR. The first video keeps a manual People selection that already
has a checked person, so you can still pick specific people there by hand.
If the person detection models are not prepared, the batch refuses to start.

Click **Reconstruct 3D (N)** to start. If export has not been configured, its
setup dialog opens first. An application-modal progress window blocks workspace
interaction, shows each video's status and the active task, and offers
**Cancel Batch**. Closing this window or pressing Escape during processing also
requests cancellation. The summary remains open until you close it.

Videos run sequentially unless two or more GPUs are checked for parallel
processing: find people (when enabled), reconstruct, export motion, optionally
export preview, then open the next video. The shared settings are
captured when the batch starts. The first video's selected people/takes are
retained when it already has a checked person; otherwise the first N found
people are checked. Later videos follow the same automatic rule; identities and
manual contact timings are not copied between videos. PromptHMR uses its
configured people and interaction settings.

If the first video is untrimmed, each video is processed in full. Otherwise its
trim is applied in **source seconds**, clamped to each video's end. A video with
no overlap fails and the batch continues. Different frame rates are supported.

Each run creates a new `Batch_<timestamp>` directory inside the chosen folder.
Subdirectories use `0001-<original video stem>` in input selection order
(for example, `dance.mp4` becomes `0001-dance`). They contain
`motion.fbx`, `motion.bvh` or a Blender bundle, plus `preview.mp4` if selected.
For multiple checked people/takes, motion files are separate and include person
and take identifiers. Existing exports and input videos are never replaced by a
new batch run. Reconstruction projects remain in their usual project locations.

`batch-report.json` is updated after transitions and records the inputs, shared
settings, project paths, completed exports, statuses and errors. A reconstruction
or export failure is recorded and processing continues with the next video. A
failed preview leaves any completed motion export in place. The same error also
appears in Console. Cancel stops the current worker/encoder before releasing the
workspace; completed projects and exports are kept, and remaining videos are
marked skipped. Re-running uses the ordinary valid-stage reuse rules and creates
a new export directory. Opening a video or project through File leaves batch mode.

Batch processing supports the offline single-video models (ExMR, FlowHMR,
GVHMR, and PromptHMR where its runtime is supported — the Linux blacklist
hides PromptHMR there). MAMMA's synchronized camera
set is one multi-view performance and keeps its separate input workflow. PEAR
remains a live preview. The batch menu is disabled for both of those tools.
GEM-X keeps its separate single-video reconstruction and export workflow.

## Parallel processing with multiple GPUs

With two or more GPUs checked in the setup dialog, the parent application
launches one headless copy of itself per selected GPU
(`QT_QPA_PLATFORM=offscreen`, `CUDA_VISIBLE_DEVICES=<GPU UUID>`) and hands each
copy the next queued video as GPUs become free. Inside every child the ordinary
sequential pipeline is unchanged: one worker at a time, the same multi-person
batch workers, model sessions, precision and selected-result behavior. A child
loads its video, applies the shared settings and trim, runs Find People when
enabled (the first video keeps a recorded manual People selection), reconstructs
and exports exactly like the sequential batch would for that video.

Consequences of running one child per GPU:

- Preview videos are rendered by the parent after reconstruction: the headless
  children cannot capture the OpenGL viewport, so once every child has
  finished, the shown main window reopens each finished project in turn and
  renders its `preview.mp4` exactly like the sequential batch would. A failed
  preview marks that video failed and keeps its completed motion exports,
  matching the sequential rule; cancelling during this pass keeps every
  rendered preview.
- Each child loads its own GUI copy and workers, so RAM use grows with the
  number of GPUs. Resource Safety stays per process: each child's monitor
  measures the same system RAM and stops only its own workers.
- The progress window adds a GPU column and one status line with its own
  progress bar per selected GPU, and `batch-report.json` records the
  GPU used per video plus the selected GPU list. Output folders, file naming
  and the report layout are otherwise unchanged.
- A crashed or failed child marks its video failed (with the child's log tail)
  and the batch continues on the freed GPU. Cancel terminates the child process
  trees; running videos are marked cancelled and undispatched ones skipped.
- Model and settings requirements are identical to the sequential batch; the
  child skips the startup preflight because the parent already ran it.

Validation: `tools/tests/test_video_batch_ui.py` exercises the lifecycle with
synthetic inputs and simulated worker/encoder completions, plus native Qt menu,
dialog and job-completion integration. It also checks real decoding of a generated
video after skipping a missing first input. `tools/tests/test_batch_parallel.py`
covers the child request/result protocol, GPU enumeration parsing, dialog wiring
and the parallel dispatch, crash, cancel and bounded-concurrency behavior with
scripted child processes. Run them in the pinned GUI environment:

```bash
QT_QPA_PLATFORM=offscreen .venv/bin/python -m unittest discover -s tools/tests -p test_video_batch_ui.py -v
QT_QPA_PLATFORM=offscreen .venv/bin/python -m unittest discover -s tools/tests -p test_batch_parallel.py -v
```

These checks do not establish real-model inference, GPU rendering/MP4 capture,
or Windows end-to-end batch compatibility.
