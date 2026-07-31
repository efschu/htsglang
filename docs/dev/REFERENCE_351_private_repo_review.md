# Task 351 -- Review of three private repos for reuse in the fork

Closeout record for task 351 (bringing `efschu/nvidia-vgpu-consumer`,
`efschu/vs-pipeline`, `efschu/jellyfin-vapoursynth-plugin`, and
`efschu/nvidia-smallbar-p2p` up to publication standard). All four stay
private; none of their visibility changes. This file records what, if
anything, from the first three is worth reusing in this fork. The fourth,
`nvidia-smallbar-p2p`, is out of scope here -- it is being prepared as a
standalone tree for a possible new public repo, not integrated into
htsglang.

Written to stand on its own; no assumption that the reader has this task's
conversation context. Cross-references task 333/339's own review where it
overlaps, rather than repeating it.

---

## 1. `efschu/vs-pipeline` -- already reviewed and substantially absorbed

Task 339's build record (`docs/dev/TASK_333_M2_VIDEO_ENHANCE.md`, section
"Prior art: efschu/vs-pipeline", landing with the rest of #339) already did
the extraction work for this repo against the Class-3 video-enhance stream
server:

- The fp16 export technique (`build/build_esrgan_rtx.py`: convert
  initializers with `numpy.astype`, set graph I/O to fp16, build
  `STRONGLY_TYPED`) was ported to `scripts/video_enhance/export_sr_fp16.py`,
  with provenance recorded in the file header and the artifact sidecar.
  `build_rife_rtx_fp16.py`'s alternative (`--io-cast-only`, fp32 compute
  with fp16 I/O via Cast nodes) was ported alongside it.
- The `tensorrt_rtx` engine builder itself and the checked-in `.engine`
  files were deliberately **not** taken -- the fork's SR path goes through
  ONNX Runtime's TensorRT execution provider, which builds its own engine
  from the ONNX file, so vendoring a second TensorRT distribution would add
  a dependency for no new capability.
- The production GPU mapping recorded in `vs-pipeline`'s README (ESRGAN on
  two RTX 3080s at `cycle=4`, two streams per card, interleave at 4K RGBH,
  RIFE on the RTX 5090, one Bicubic RGB->YUV at the end) reopened the
  Regime B (stage-split-across-cards) question for Class 3, with an
  arithmetic case from the fork's own P2/§9.4 measurements.

Nothing further to extract. The one fact worth restating here because it is
easy to lose in a large task doc: `vs-pipeline`'s prebuilt `.engine` files
were rig- and TensorRT-RTX-version-specific (built against TensorRT-RTX
1.5.0, CUDA 13) and were removed from that repo's own tree during the task
351 cleanup rather than carried anywhere -- they were never a reuse
candidate, only a hygiene problem in the source repo.

---

## 2. `efschu/jellyfin-vapoursynth-plugin` -- one pattern not yet captured

This is the Jellyfin-side counterpart to `vs-pipeline`: a C# plugin
(`Jellyfin.Plugin.VapourSynth`) that runs the same upscale/interpolate chain
as part of Jellyfin's live-transcode and background-job flows. Task 339's
build record already notes it as "prior art for the M2 job API ... rather
than for anything in this task" without detail. Filling in that detail:

### 2.1 What is superseded already, not worth porting

- **`GpuResourceManager.cs`** tracks GPU occupancy by shelling out to
  `nvidia-smi --query-gpu=...` and parsing CSV text, with a semaphore per
  GPU for slot admission. `python/sglang/srt/video_enhance/nvml.py` and
  `reservation.py` already do the equivalent job through NVML directly
  (physical device identity by UUID, not text-parsed `nvidia-smi` output),
  which is strictly more robust. Not a reuse candidate.
- **`VapourSynthController.cs`**'s REST surface (`POST .../jobs`,
  `GET .../jobs/{id}`, `DELETE .../jobs/{id}`, `GET .../status`) is a job
  lifecycle API in the same spirit as `server.py`'s
  `POST/GET /v1/video/enhance`, `GET /v1/video/enhance/{job_id}`,
  `DELETE /v1/video/enhance/{job_id}`. The fork's version already exists and
  is more capable (streaming response bridge, liveness policy, engine/plan
  introspection endpoints). Not a reuse candidate either.

### 2.2 What is genuinely new: the ffmpeg-wrapper interception pattern

Neither `video_enhance/server.py` nor anything else in the fork addresses
*becoming a drop-in replacement inside another application's transcode
pipeline*. `jellyfin-vapoursynth-plugin` does exactly that, and the
mechanism is worth recording because it is the concrete shape a future
"Jellyfin consumer of the Class-3 server" task would need and it does not
exist anywhere else in this fork:

- `vapoursynth-scripts/vs_wrapper.sh` (and `vs-pipeline`'s
  `pipeline/ffmpeg-wrapper.sh`, the same idea in the sibling repo) replaces
  the `ffmpeg` binary Jellyfin calls (`/usr/local/bin/ffmpeg` on the
  container `PATH`, ahead of the real one).
- Non-transcode invocations -- version probing, capability probing, image
  extraction -- are detected by argument shape and passed straight through
  to the real `ffmpeg` binary unmodified. Jellyfin never notices the
  wrapper exists for these calls.
- HLS transcode invocations are parsed for their input file, output file,
  and mid-argument codec/filter options (by scanning the argv for `-i`,
  the output path, and the span between them), then rebuilt as a 3-stage
  pipe chain: decode ffmpeg -> named pipe -> `mpv --vf=vapoursynth=...`
  (upscale/interpolate) -> named pipe -> encode ffmpeg, with the original
  codec/filter args replayed on the encode stage so the output container
  and codec choice Jellyfin asked for is preserved.
  `Services/VapourSynthScriptGen.cs` generates the `.vpy` script per job
  from a template (`vapoursynth-scripts/templates/*.vpy`) rather than
  shipping one fixed script, so per-item settings (scale factor, filter
  choice, target fps from `PluginConfiguration`) become `{{...}}`
  placeholder substitutions.
- A toggle file (`/tmp/vs_pipeline_active`, also checked by the web client
  patch in the plugin's `playersettingsmenu.js`) lets the pipeline be
  switched on/off per-session from the player's quality menu without a
  Jellyfin restart, read by both the wrapper script and the C# controller.

This is the one piece worth keeping as forward-looking design reference: if
a future task wants the Class-3 video-enhance server to be usable as a
transparent drop-in inside an existing transcode pipeline (Jellyfin or
otherwise) rather than only through its own HTTP API, this
argv-sniff-and-repipe approach is the proven mechanism, and the per-call
pass-through/intercept split (by argument shape, not by a separate config
flag) is the detail that makes it safe to install permanently.

Nothing here changes any decision task 339 already made; it is additive
detail for a task that has not been opened yet.

---

## 3. `efschu/nvidia-vgpu-consumer` -- nothing to integrate

This repo is an investigation, not code: whether NVIDIA vGPU's
software-mediated path can be forced on consumer Ampere/Blackwell cards in
place of the hardware-gated SR-IOV path. Verdict of the investigation: no,
for both cards, for hardware and firmware reasons (SR-IOV capability absent
on GA102; GSP firmware fused off on GB202) that have nothing to do with
inference serving.

**Nothing to integrate.** The subject (NVIDIA vGPU virtualization, VM/guest
device assignment) is orthogonal to everything this fork does (tensor
parallelism, speculative decoding, KV-cache placement, collective
communication for LLM serving on bare-metal GPUs). The one methodology
point that generalizes -- resolve physical GPU identity through NVML
(UUID/PCI bus), never through CUDA's own enumeration order, because the two
can diverge -- is already established practice in this fork
(`video_enhance/nvml.py`, and the device-order handling used throughout the
uneven-TP and multi-card work predating task 333). There is no new
technique or finding in that repo this fork does not already apply.
