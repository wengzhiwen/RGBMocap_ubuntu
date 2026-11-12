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

Click **Reconstruct 3D (N)** to start. If export has not been configured, its
setup dialog opens first. An application-modal progress window blocks workspace
interaction, shows each video's status and the active task, and offers
**Cancel Batch**. Closing this window or pressing Escape during processing also
requests cancellation. The summary remains open until you close it.

Videos run sequentially: reconstruct, export motion, optionally export preview,
then open the next video. The shared settings are captured when the batch starts.
The first video's selected people/takes are retained. Later videos track people
automatically; identities and manual contact timings are not copied between
videos. PromptHMR uses its configured people and interaction settings.

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
GVHMR and PromptHMR where its runtime is supported). MAMMA's synchronized camera
set is one multi-view performance and keeps its separate input workflow. PEAR
remains a live preview. The batch menu is disabled for both of those tools.

Validation: `tools/tests/test_video_batch_ui.py` exercises the lifecycle with
synthetic inputs and simulated worker/encoder completions, plus native Qt menu,
dialog and job-completion integration. It also checks real decoding of a generated
video after skipping a missing first input. Run it in the pinned GUI environment:

```bash
QT_QPA_PLATFORM=offscreen .venv/bin/python -m unittest discover -s tools/tests -p test_video_batch_ui.py -v
```

These checks do not establish real-model inference, GPU rendering/MP4 capture,
or Windows end-to-end batch compatibility.
