# Task #333 M2 — Class-3 video-enhance stream server

Implementation record for build stage M2 of `docs/dev/DESIGN_333_multimodal_classes.md`
(§8 in full, §6 for the Class-3 contract, §9.3 for the reuse verdicts).
The design is the specification; this file records what was built, what was
decided differently and why, what was measured, and what M2 does not do.

Everything lives under `python/sglang/srt/video_enhance/`. Nothing in that
package imports `srt` scheduler internals: the tenant is its own process with
its own CUDA context, pinned to one physical GPU through
`CUDA_VISIBLE_DEVICES`.

---

## 1. Module map

| Module | Responsibility | Device needed |
|---|---|---|
| `frame_math.py` | exact frame/stage byte arithmetic, the §6.2 reservation formula, `max_in_flight_for_budget` | no |
| `chain.py` | the chain as a validated graph; stage order, geometry and format continuity | no |
| `frames.py` | `Frame`, the `Stage` protocol, the device-residency check | no |
| `ring.py` | bounded rings, overload policy, stall accounting | no |
| `pipeline.py` | the executor: one task per stage, rings between, arity windows | no (torch-free) |
| `mux.py` | ffprobe inventory, track selection, retiming, ffmpeg remux | ffmpeg |
| `server.py` | the HTTP surface, job registry, response bridge | no |
| `tenant.py` | tenant config, budget, `plan_job`, stage factory | no |
| `engine_cache.py` | engine identity keying and the provenance manifest | no |
| `parity.py` | PSNR/SSIM gate against the fp32 reference | torch |
| `backends.py` | ONNX Runtime CUDA/TensorRT backends, native-TRT seam | torch, onnxruntime |
| `sr.py` | SR stage, pinned artifact, fetch-and-verify | torch, onnxruntime |
| `resize.py` | separable Lanczos-3 | torch |
| `rife.py` + `_vendor/rife/` | RIFE stage, version/padding/scale semantics, vendored IFNets | torch |
| `codec.py` | NVDEC decode, NVENC encode, GPU colour conversion, test-clip generator | PyNvVideoCodec / ffmpeg |
| `nvml.py` | physical device identity by UUID | pynvml |
| `reservation.py` | the Class-3 slice of the §3.3 cross-process ledger | no |
| `shard_plan.py` | capacity-weighted chunk sharding, and the baselines it is measured against | no |
| `probes.py` | measurement posts P1-P5 as a CLI, plus the playback arithmetic | torch |
| `timing.py` | deferred-readout CUDA-event timing | torch optional |

---

## 2. Decisions taken, with the reason

### 2.1 ONNX Runtime's TensorRT EP instead of the vs-mlrt source extraction

§9.3 verdicts the SR execution path as "port by extraction" from
`efschu/vs-mlrt`, on the stated grounds that there is nothing to depend on.
There is: `onnxruntime-gpu`'s `TensorrtExecutionProvider` builds and caches a
real TensorRT engine from an ONNX file, honours a min/opt/max profile, and
binds input and output by device pointer so a torch CUDA tensor goes in and
out with no host copy. The same section's reuse order is dependency before
port, so the dependency wins on the section's own rule.

The extraction is not discarded. `backends.NativeTensorRTBackend` is the
named seam, and it is the route to what the EP does not expose: explicit
execution contexts, per-context workspace accounting (the §6.2
`trt_context_workspace_bytes` post can only be measured directly through it),
and tiling. It is a registered follow-on post, not a gap that was overlooked.

### 2.2 Torch Lanczos-3 instead of `lanczos3_kernel.cu`

Same trade. The separable torch implementation is portable, has no build step,
and accumulates one tap at a time so the 8K-to-4K intermediate never
materialises. Porting the CUDA kernel is a performance post to open once P1
shows the resize stage is a meaningful share of the frame budget — the numbers
in §5 below say whether it is.

### 2.3 The SR allocator overhead is one named constant

§8.3 publishes per-stream budgets (~1.0 GiB at 1080p, ~0.25 GiB at 540p) that
are larger than the exact tensor arithmetic. Rather than writing those numbers
down, `SR_ALLOCATOR_OVERHEAD_FRACTION = 0.45` reproduces both from the
formula, and a unit test asserts it does. Measurement post P3 confirms or
replaces the fraction; it is not a safety margin and the code says so.

### 2.4 RIFE's footprint is refused, not estimated

`rife_footprint` raises `UnprobedFootprintError` unless a measured value is
supplied, and `plan_job` refuses a RIFE chain whose P4 value is unset. §8.3
registers the number as unmeasured and asserts none; an estimator that
invented one would make the ledger confidently wrong.

### 2.5 Parity thresholds are ours, and labelled as such

§8.7 fixes the metrics (PSNR, SSIM against an fp32 reference) but no
thresholds. `parity.py` uses PSNR ≥ 40 dB and SSIM ≥ 0.995 for fp16 against
fp32, with the derivation in the module docstring: at 40 dB the mean squared
error is below the 8-bit quantisation step of the encoded output. The
docstring states these are this build's parameters, not the design's.

### 2.6 Chunk sharding, not modulo-N interleave

VSGAN distributes frames with `SelectEvery(cycle=N)` — a static equal-share
modulo-N round robin. Two reasons that is not copied: it gives a 5090 and a
3080 the same share, and it splits consecutive frame pairs across cards, which
breaks RIFE. `shard_plan.py` assigns contiguous chunks weighted by measured
per-(stage, card) rates, with an explicit one-frame overlap at every boundary
so interpolation can cross the seam. The two baselines it is compared against
(`static_single_card_plan`, `vsgan_style_modulo_plan`) live in the same module
and use the same cost model, so the before/after comparison is not a
hand-rolled apples-to-oranges.

---

## 3. Requirements added during the build

These came from the user during implementation and are part of M2's scope, not
of the original §8 text.

### 3.1 Multi-track discipline (`mux.py`)

*   Every non-video track is stream-copied, never re-encoded: audio,
    subtitles, chapters, metadata, language tags and dispositions pass through
    bit-identically, in source order. Where a source has several video tracks,
    the selected one is enhanced and the others are copied.
*   Cover art is detected via `disposition.attached_pic` and is never a
    candidate for enhancement.
*   The API exposes track selection (`enhance_video_index`, and
    `passthrough_*` switches), and `GET /v1/video/tracks` reports what would
    happen to every track before anything runs.

### 3.2 A/V sync after interpolation

Interpolation changes both the frame count and the frame rate, and the two
must change consistently or audio drifts against video.

*   `retimed_rate` multiplies the rate in exact rational arithmetic:
    24000/1001 at 2x is 48000/1001, never 47.952.
*   `expected_frame_count(n, m) = n + (n-1)(m-1)`. Interpolation fills the
    gaps *between* frames and there are `n-1` of them; the naive `n*m` is
    wrong by `m-1` frames.
*   `duration_drift_s` reports the residual, which is exactly one output frame
    interval — a constant, not something that grows with clip length. A unit
    test asserts the drift for a 10-second and a 10-minute clip is the same
    number.
*   Audio and subtitle timestamps are never touched: they arrive at the muxer
    by stream copy.

### 3.3 Container-aware chunking

Chunk boundaries are an internal transport detail. ffmpeg does the demux,
stream copy and mux — nothing here writes container boxes. MP4 output is
fragmented (`+frag_keyframe+empty_moov+default_base_moof`) so a partial
response is a parseable file rather than a truncated one waiting for a moov
atom that never arrives.

Back-pressure survives the extra process: the remuxer's `feed` awaits a pipe
drain, and its stdout is only read when the bounded response ring has room, so
a stalled client still stalls the decoder.

### 3.4 Per-source chain configuration

The chain is request-level configuration, not a fixed pipeline:

*   `enable_sr`, `sr_scale` (4 for the x4 model, 1 for a same-size denoise or
    restoration model), `enable_resize`, `fps_multiplier`, `rife_scale`,
    `rife_version`.
*   A 4K source is typically RIFE-only — x4 SR on 4K produces 8K, which is
    only done on request. `build_chain` produces exactly
    `decode → colour → rife → colour → encode` for that request, and the
    reservation drops the SR and resize posts with it.
*   A request that asks for no SR, no resize and no interpolation is refused
    rather than run as an expensive copy.

### 3.5 Enhance-by-URL is the primary consumption form

`GET /v1/video/enhance?source_url=...` returns the enhanced stream directly.
Any player that opens an HTTP URL is a client — VLC and mpv need no plugin, no
filter graph, no local install. The VapourSynth shim and the CLI named in the
briefing are convenience wrappers around this URL and are follow-on posts, not
M2.

---

## 4. Back-pressure, the gate that matters most

§10 M2 acceptance gate 3. The mechanism, end to end:

1. One `BoundedRing` per stage boundary. The class has no code path that grows
   past its declared depth.
2. The encode task `await`s the sink. The sink is the remuxer's `feed`, which
   awaits a pipe drain; the remuxer's stdout is drained into a depth-1 bridge
   ring; the response generator yields from that bridge and Starlette awaits
   each yield until the transport accepts it.
3. So a client whose TCP window is full blocks the generator → the bridge
   fills → the muxer blocks → the encode task stops draining → the upstream
   rings fill → the decode task, which is a pull source, stops pulling.
4. Overload policy is per request: `stall` (default) or `drop_frames`. A drop
   increments a counter reported in the progress endpoint and the response
   trailer. Silent dropping does not exist.
5. `max_in_flight` is derived from the reservation, never configured
   independently, so a depth that does not fit cannot be configured.

Proven hermetically in `test/registered/video_enhance/test_backpressure.py`:
with a sink that never accepts a byte, a source willing to produce 200 frames
is stopped after fewer than `boundaries × (depth+1) + 4`, and every ring's
occupancy stays at or below its declared depth.

---

## 5. Measurements

Recorded in §5 of this file as they are taken. Card windows are short, so the
posts are run per card and the JSON is kept.

See `docs/dev/TASK_333_M2_MEASUREMENTS.md` for the raw records.

---

## 6. What M2 does not do

Stated honestly, extending the design's own list:

*   **Single card for the executor.** The shard planner computes multi-card
    plans and predicts their makespan, and the P1 table it consumes is
    measured per card — but the executor runs one chain in one process on one
    card. Distributing a job across cards is the next post, not this one.
*   **No Regime B.** No stage split across cards. P2 is measured so the
    decision has data behind it before any Regime-B code exists.
*   **No int8 compute.** Deferred by §8.7 and by the standing rule that lossy
    features come last.
*   **The native TensorRT driver is a seam, not an implementation.** The
    per-context workspace post (§6.2) is therefore inferred from total peak
    device bytes rather than measured directly.
*   **RIFE runs in torch eager.** The TensorRT path for RIFE is a declared
    seam; `engine_resolution()` returns the padded post-resize size the engine
    would have to be built at.
*   **Three RIFE versions are vendored** (4.6, 4.18, 4.26) out of the 36 in
    upstream's enum. Asking for another known version is refused by name
    rather than silently substituted.
*   **The reservation is a static configured budget, not a registry slot.** M2
    runs ahead of M1, so co-tenancy safety rests on the operator setting
    `budget_mib` correctly. The ledger in `reservation.py` enforces the
    invariant across processes but nothing promotes or demotes.
*   **No per-request SR model selection beyond the pinned artifact.** The
    catalog is a sourcing map; one model is pinned and hash-verified.
*   **Variable denoise strength is not available.** The `wdn` blend is baked
    into the exported ONNX; a request-time knob needs either several
    fixed-blend exports or a pre-export weight interpolation.
*   **4K source at the in-flight depth 1080p supports.** The arithmetic says
    it does not fit alongside a HOT LLM on the 5090, and the code refuses it
    rather than discovering it at runtime.
*   **Audio enhancement is named, not built.** A Demucs-class audio stage over
    the same track inventory is a further Class-3 building block; `mux.py`
    already carries the tracks it would operate on.

---

## 7. Follow-on posts

1. Native TensorRT driver extracted from `efschu/vs-mlrt` (§9.3), unlocking
   per-context workspace measurement and tiling.
2. `lanczos3_kernel.cu` port, if P1 shows resize is a meaningful share.
3. Multi-card execution of `shard_plan`'s output (the planner exists; the
   executor is single-card).
4. RIFE TensorRT backend at the post-resize shape.
5. VapourSynth shim and CLI over the same executor.
6. Audio-enhance stage (Demucs class) on the passthrough track inventory.
7. Registry integration (M1): replace the static budget with a ledger slot.

---

## 8. Live watch + client liveness (user directive 2026-07-31)

Two additions to the M2/#339 scope, recorded here as the persistent decision
(tracked as task #344):

1. **Live watch of running jobs.** While a video is being processed, the web
   frontend must be able to watch BOTH the incoming stream and the outgoing
   (enhanced) stream live. Implementation direction: low-bitrate preview taps
   off the pipeline (NVENC side-encodes at preview resolution), served as
   separate streams; the taps must never stall the main chain (drop-frame
   preview, bounded queues — a slow preview viewer costs preview frames, not
   pipeline throughput).

2. **Universal client liveness with configurable timeouts.** For every kind
   of session — LLM streams, video streams, training jobs, registry leases —
   the server must detect that the other side is gone (bandwidth collapse,
   network loss, silent client death) and clean up QUICKLY: KV, VRAM leases,
   job slots, decoder/encoder pipelines. Reuse the mechanisms the APIs
   already have (TCP/SSE disconnect, heartbeats, ledger lease TTL) rather
   than inventing new ones, but make them interoperate; cleanup never runs
   in the serving hot path and nothing blocks anything else unnecessarily.
   The wait-until-declared-dead duration is user-configurable per endpoint
   class. During a grace window the bound resources are NOT idly pinned:
   they belong to the normal reclamation ladder (idle tenant #341, pressure
   staircase #287, spill/offload) until the client either returns or is
   declared dead. Related prior art in-tree: #312 bounded peer liveness in
   collectives, #305-M1 ledger lease+heartbeat, M2 abort semantics (#338).
