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
| `chain_policy.py` | which chain shape to run: candidates, pricing against the frontier and the reservation, the refusal | no |
| `streaming.py` | streaming-input admission, the seconds-deep output watermark, sustained in/out rate | no |
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
| `multicard.py` | the multi-card executor: chunk specs, the seam convention, ordered stitching, and the two schedulers (pre-weighted, pull queue) | no |
| `chunk_worker.py` | one chunk per process, or a serving worker taking items from a pipe | yes |
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

#### What #338 added to this endpoint

The browser extension in `clients/browser-extension/` is the first client
built on it, and it needed three things the endpoint did not have. All three
are additive: a request that names none of them takes the path it always took.

*   **`start_s` / `duration_s`** — a time range, resolved once at the HTTP
    surface into the `start_frame` / `frame_limit` pair the decode stage
    already had from the multi-card work, plus an input seek on the remuxer's
    source input so the passthrough tracks start at the same point. Seconds
    are converted to frame indices through `Fraction`: at 24000/1001 fps a
    float product lands off by one, and one frame is exactly what is visible
    at a seam. Audio alignment is bounded by one packet of each copied track
    (21.3 ms for 48 kHz AAC), because the kept tracks are `-c copy` and can
    only begin at a packet boundary while the video seek is frame-accurate.

*   **`job_id`** — the caller may name the job. This is what makes
    `DELETE /v1/video/enhance/{id}` reachable at all from the class of client
    this endpoint exists for: a URL handed to a `<video>` element, to mpv or
    to a download never surfaces a response header, so a server-minted id
    cannot be learned. The id is echoed in `X-Enhance-Job` either way, and a
    collision with a *live* job is a 409 rather than a silent reuse that would
    make `DELETE` ambiguous.

*   **`GET /v1/video/capabilities`** — what a client asks before it offers the
    user a chain preset. It keeps two kinds of answer under two keys, because
    conflating them is how a client comes to believe a measurement nobody
    took. `frontier` is measured: rows come from `ProbeReport` JSON on disk
    (§8.6 P1) through `probes.load_frontier`, carrying the card and noise
    floor each row was measured with, and is explicitly `measured: false` with
    a reason when no measurement directory is configured. `budget` is
    arithmetic: `plan_job` is pure, so whether a preset *fits* in the
    configured MiB is answerable with no card at all. Fitting is not the same
    as being fast enough.

The chain presets a client picks between (`rife_only`, `full_chain`) live in
`server.CHAIN_PRESETS` and are mirrored in the extension's `shared.js`; a test
compares the two field by field, so the name a user selects and the name a
measurement row is filed under cannot drift apart.

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

*   ~~**Single card for the executor.**~~ Closed by #339: `multicard.py` runs
    the planner's chunks concurrently, one process per card, and stitches
    them in timeline order. See §9.
*   ~~**No live watch.**~~ Built by #344a — see §13. Client liveness (§8.2)
    is separate and is §11.
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
3. ~~Multi-card execution of `shard_plan`'s output.~~ Done, #339, §9.
   Superseded as the default by pull scheduling, §12.
4. RIFE TensorRT backend at the post-resize shape.
5. VapourSynth shim and CLI over the same executor.
6. Audio-enhance stage (Demucs class) on the passthrough track inventory.
7. Registry integration (M1): replace the static budget with a ledger slot.
8. Bind a torch-owned output buffer to the SR session instead of cloning
   ONNX Runtime's. Needs the output shape threaded from the chain into the
   backend, which the stage knows and the backend does not — see §9.1.
9. Live watch and client liveness (§8), untouched.
10. Regime B, reopened by prior art rather than by argument — see §10.
11. Cut the per-item fixed cost of a pull-queue item so the queue can be
    finer than the balance it currently achieves — §12.7. The encoder
    session is the expensive half.

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

---

## 9. Task #339 — the defects, and multi-card execution

Recorded 2026-07-31. Raw records in `TASK_333_M2_MEASUREMENTS.md` and
`docs/dev/measurements/333-m2/`.

### 9.1 The byte-stability defect was a cross-stream hazard

§ "End-to-end functional proof" recorded "output is not byte-stable across
two runs — not diagnosed". It is diagnosed, and it was not a tolerance
problem: it was corrupting whole frames.

ONNX Runtime's CUDA execution provider runs on its own CUDA stream. The SR
backend binds a torch tensor to it by device pointer, so every inference
crosses a stream boundary twice, and neither crossing was ordered. ORT could
begin reading the input before torch had finished producing it, and torch
could begin reading the output before ORT had finished writing it. Nothing
in the API suggests this; `run_with_iobinding` returns and the output tensor
is there.

The falsifier that settled it needed no chunking and no second card: the
**same whole-clip run at ring depth 1, 2 and 4 produced three different
outputs**, differing in 9 and then 178 of 191 frames. A deeper pipeline
means more frames in flight and a wider window for ORT's stream to write
over a frame another stage is still reading. Cloning the output does not
help — the clone reads the same memory at the same unsynchronised moment.

Fixed with `torch.cuda.current_stream().synchronize()` and
`binding.synchronize_inputs()` before the run, `binding.synchronize_outputs()`
after. Afterwards the chain is ring-depth invariant (0 of 191 frames differ
across depths 1, 2 and 4) and two whole-clip runs produce byte-identical
elementary streams.

**Byte-stability gate: met.** The remaining known nondeterminism source is
named rather than open: convolution results are not bit-identical *across
architectures*, so a frame produced on a 3080 is not the frame the 5090
produces. That is a property of the hardware, it is measured (§9.4), and it
is why the multi-card correctness gate is the same-card control rather than
a cross-card hash.

Two smaller defects found in the same area, both by running rather than by
reading:

*   An fp16 frame was bound to the fp32 SR graph with the element type
    declared as float32, so ONNX Runtime read twice the allocation and
    faulted with `CUDA failure 700: an illegal memory access` from inside the
    session, with nothing in the message pointing at the dtype. The backend
    now casts to the graph's input type and warns once naming both; the SR
    stage restores the chain's declared pixel format on the way out.
*   `cudnn_conv_algo_search` is no longer left at ONNX Runtime's `EXHAUSTIVE`
    default, which picks a convolution algorithm by stopwatch and can
    therefore choose differently in two processes. `HEURISTIC` picks by shape
    and reproduces.

### 9.2 The mov_text subtitle mismatch was an edit-list loss

Also recorded as an open defect, and the smaller half of a larger one.
Bisected entirely on CPU, no card involved.

The muxer is byte-stable: three remuxes of one fixed elementary stream
produce identical bytes, so the container writer was never the instability.
What the bisection found instead is that fragmented MP4 with `+empty_moov`
writes the moov before any packet has been seen, so libavformat cannot emit
an `elst` box for any track. An MP4 whose audio came out of an AAC encoder
carries its 1024-sample priming compensation as exactly such an edit list.

Losing it moved the audio's first packet from -0.021333 s to 0.0 and
dropped its `Skip Samples` side data, so **the copied audio played 21.333 ms
late against the enhanced video**, with the subtitle track dragged along.
The duration check could not see it: all tracks shifted together, so the
durations still matched. Only an inter-track offset comparison sees it, and
`alignment_report()` is now that comparison, gated at 1 ms.

`+delay_moov` holds the initial moov back until the first fragment is cut —
late enough to know the edit lists, early enough to still stream. With it the
`Skip Samples` survives, the cues sit at their source timestamps, and the
second video packet lands at 0.020833 s rather than the 0.042236 s the
flag-less variant produced.

What remains of the subtitle mismatch is one trailing two-byte empty sample.
mov_text has no gap representation, so screen time with no cue on it must be
an explicit empty sample, and an output that runs longer than the source is
obliged to append one or leave the last cue on screen for the extra
duration. That is required muxer output, not source content.
`strip_empty_mov_text()` separates the two so the passthrough gate compares
text rather than padding; a changed cue still fails it.

**Subtitle post: closed.** Audio and subtitle content are bit-identical
across the remux, and inter-track drift is 0.0 s where it was 0.021334 s.

### 9.3 The multi-card executor

`multicard.py` plus `chunk_worker.py`. The planner already decided what each
card should do; this is what carries it out.

The seam convention is the tail: chunk *k* pulls frame `stop` as RIFE's
second input, emits the interpolated frames for the pair that straddles the
boundary, and withholds the trailing original, which chunk *k+1* encodes.
`shard_plan` prices both a lead and a tail overlap because its cost model is
symmetric and cannot know which side will do the work, so the executor's
real seam cost is half what the planner charged — pessimistic, which is the
safe direction for an admission check.

The arithmetic is the gate, not a comment. A chunk with a successor encodes
`n*m` frames, one without encodes `n + (n-1)(m-1)`, and any chunking of `N`
frames at multiplier `m` sums to `N + (N-1)(m-1)` — the same
`expected_frame_count` the muxer retimes against. `verify_chunk_arithmetic()`
refuses a chunking that does not, before a card is touched, and the executor
refuses a chunk whose worker reported a count the arithmetic did not predict.

Two things multi-card gives up, both stated rather than glossed:

*   **Back-pressure is bounded, not immediate.** The single-card chain stalls
    the decoder within one ring depth. Here a finished chunk waiting for an
    earlier one has to be spooled, so a stalled client stops the run after at
    most `spool_chunks` completed chunks — one per card by default, which is
    the smallest bound that still lets every card work at once.
*   **A process launch per chunk.** Roughly 8 s of torch and ONNX Runtime
    import before the first frame moves, which is the whole reason the
    end-to-end speedup below is lower than the compute-only one.

**The device-order trap.** A worker was launched with `CUDA_VISIBLE_DEVICES`
set to an NVML index and nothing else. CUDA enumerates `FASTEST_FIRST` by
default, which on this rig is not NVML's PCI-bus order — the 5090 is NVML
index 1 and CUDA ordinal 0. It does not fail: every card runs, the seam
stays exact, the output is correct. It just measures and schedules the wrong
cards, and it flattered the headline number by nearly a factor of two (2.91x
against a baseline silently on a 3080, 1.44x against the 5090 it was meant
to be compared with). `CUDA_DEVICE_ORDER=PCI_BUS_ID` is now set in the child
environment ahead of anything the caller passes in.

### 9.4 Multi-card measurements

480 source frames, 960x540 → 1920x1080, SR x4 + Lanczos-3 + RIFE 4.6 x2,
fp16 chain, SR on the ONNX Runtime CUDA provider, ffmpeg NVENC encode.
NVML 1 = RTX 5090, NVML 0 and 2 = RTX 3080.

Per-stage ms per invocation, from the run itself:

| Stage | 5090 | 3080 | ratio |
|---|---|---|---|
| SR (960x540, x4) | 35.58 | 90.9 | 2.55 |
| resize (3840x2160 → 1920x1080) | 6.48 | 14.1 | 2.18 |
| RIFE (1920x1080, scale 1.0) | 3.06 | 8.28 | 2.71 |
| encode | 1.56 | 2.5 | 1.60 |

The 5090's SR figure reproduces the single-card P1 table (35.19 ms) to
within the 2.77 percent noise floor, which is the check that the multi-card
harness measures the same thing the earlier post did.

| Arm | wall | notes |
|---|---|---|
| baseline, 5090 alone | 35.80 s | one chunk, same executor |
| three cards | 24.94 s | plan: 5090 [0:243], 3080s [243:361] and [361:480] |
| same-card control | 54.22 s | the identical chunking, all on the 5090 |

*   **End to end: 1.44x.**
*   **Compute only: 1.64x**, excluding the ~8 s worker start from both arms.
*   **Ceiling from the measured rates: 1.78x.** A 3080 is 0.39 of a 5090 on
    this chain, so two of them add 0.78 of a card. The capacity weighting is
    therefore running at about 92 percent of what the rates allow.

The honest framing of the 1.8-2.6x projection: it is not reachable *on this
rig* with this chain, because the cards are that unequal and the job is that
short. Nothing here says it is unreachable on a rig with comparable cards,
and the ceiling calculation is the thing to carry forward, not the 1.44.

Correctness, 480-frame run:

| Check | Result |
|---|---|
| output frame count | 959, matching `expected_frame_count(480, 2)` — pass |
| **seam exact** | **959 of 959 frames bit-identical to the whole-clip run on one card** — pass |
| multi-card vs same-card control | 37.4 dB PSNR — cross-architecture arithmetic, measured, not passed |
| multi-card frames bit-identical to the 5090 baseline | 486 of 959 (exactly the 5090's chunk) |

The seam row is the one that matters, and it is exact rather than
approximate: the comparison is a sha256 per frame taken immediately before
the encoder, so the encoder cannot confound it. That instrument had to be
built — the first attempt compared the two arms by PSNR of the encoded
output and scored 6-25 dB on pixels that were provably identical going in,
because two independently encoded H.264 streams of the same frames are not
the same stream and their PSNR is a statement about rate control.

### 9.5 What is still open from the P-series

*   **P3 (SR allocator overhead)** — still not answered. The corrected probe
    reads the device-wide free-memory delta rather than torch's allocator;
    it did not fit in these windows. `SR_ALLOCATOR_OVERHEAD_FRACTION = 0.45`
    remains unvalidated.
*   **P5 (codec ceiling, concurrent session count)** — not measured.
*   **P6 (co-tenancy jitter against a HOT LLM)** — not measured. The shard
    planner warns when a card declares an LLM co-tenant without a P6 derate,
    so a plan built without it says so.
*   **P7 (8-bit transport matrix)** — not measured.
*   **PyNvVideoCodec encode** — still not working. The `__dlpack__` signature
    problem is worked around, and the wrapped tensor then reaches NVENC and
    is rejected with "incorrect usage of CPU input buffer". Every measurement
    here used the ffmpeg NVENC backend, which is a host round trip per frame
    and is visible in the encode column above. Not diagnosed further.
*   **fp16 TensorRT engine and its parity numbers** — the export and the
    parity harness exist (`scripts/video_enhance/export_sr_fp16.py`); the
    graded numbers are not in yet.

---

## 10. Prior art: `efschu/vs-pipeline`

The reference VapourSynth pipeline M2 was built from became public during
#339. Two things in it bear directly on open posts.

**fp16 export.** `build/build_esrgan_rtx.py` solves the fp16 problem for
*this exact model*: convert the initializers with `numpy.astype`, set the
graph I/O to fp16, build `STRONGLY_TYPED`.
`scripts/video_enhance/export_sr_fp16.py` is that conversion, ported, with
provenance recorded in the file header and in the artifact's sidecar.
`build_rife_rtx_fp16.py`'s alternative — Cast nodes at the I/O so the
compute stays fp32 while the interface is fp16 — is ported alongside it as
`--io-cast-only`, because it is the right answer for a network the parity
gate rejects at full fp16 and having both means the choice is a measurement.

What is deliberately *not* taken is the builder: `vs-pipeline` builds with
`tensorrt_rtx` and ships `.engine` files. This fork's SR path goes through
ONNX Runtime's TensorRT EP (§2.1), which builds its own engine from the
ONNX, so the port produces an ONNX and the engine stays the EP's business.
Adding a second TensorRT distribution to produce an artifact the existing
one can produce would be a dependency for no capability. The prebuilt
`engines/*.engine` are for that toolchain and this rig and are not vendored.

**Regime B, reopened.** The README records the working production mapping:
ESRGAN on the two 3080s (`cycle=4`, two streams per GPU), interleave at 4K
RGBH, RIFE on the 5090, one Bicubic RGB→YUV at the end. That is a *stage*
split across cards — Regime B, which §6 excluded for M2 and which
`shard_plan`'s cost model prices but does not run.

It is excluded on an argument the measurements now partly contradict. P2
measured the host-staged round trip at 1.7 ms for an 11.87 MiB post-resize
boundary, and the reference pipeline's split point is exactly there. And the
per-card table in §9.4 says why the shape is attractive on *this* rig: the
3080s are 0.39 of the 5090 on the whole chain, but SR is 71 percent of the
5090's frame time and 84 percent of a 3080's, so moving SR wholesale onto
two 3080s and leaving RIFE alone on the 5090 balances differently than
chunking does. That is a hypothesis with an arithmetic case, not a result —
but it is now a hypothesis with a working implementation behind it, which is
a better reason to reopen the post than the one that closed it.

`efschu/jellyfin-vapoursynth-plugin` is the other half of the same system: a
job model with `PipelineManager`, `GpuResourceManager` and
`BackgroundJobService` over this chain. It is prior art for the M2 job API
and for the §8 liveness work rather than for anything in this task.

---

## 11. Client liveness (§8.2), built in #339

`liveness.py`, wired into `server.py`'s streaming path. The preview taps of
§8.1 are still open; this is the other half of the directive.

**What it is for.** A client that closes the connection was already handled:
Starlette throws into the response generator and its `finally` tears the job
down. The case that was not handled is the client that neither closes nor
reads. The socket stays open, the TCP window stays full, the sink coroutine
never returns, and back-pressure — working exactly as designed — stalls the
chain and holds a decoder, an encoder, the engines and the reservation for a
viewer who left. From the server's side that is indistinguishable from a
very slow viewer, and the only thing separating them is how long. So the
duration is the policy and it is configured, not constant.

**Progress means bytes the transport accepted**, not bytes the chain
produced. A stalled client makes the chain stop producing, so "the pipeline
is idle" is a consequence of the stall and cannot be its evidence. The
watchdog is stamped after the `yield` returns, which is the one moment in
the process that proves the peer is still there.

**Per endpoint class**, because the right number differs by an order of
magnitude: a paused player is normal and reclaiming its job would be worse
than holding it, while a preview tap that has accepted nothing for ten
seconds has no viewer. Defaults: `video_stream` 300 s, `preview_tap` 15 s,
`control` 60 s. `LivenessConfig.parse("video_stream=120,preview_tap=5")` is
the server-argument form, a non-positive value disables detection for that
class (a batch export nobody watches by design is a real case), and
`GET /v1/video/liveness` reports the resolved policy.

Two things the implementation had to get right that are easy to miss, both
found by the end-to-end test rather than by reading:

*   **A suspended generator never reaches its own `finally`.** The stage
    teardown that runs there on a normal or a cancelled stream does not run
    for a client that simply stopped calling `__anext__`, so the release
    path closes the stages itself. Without that the decoder and encoder
    sessions survive the job.
*   **`executor.cancel()` does not unblock a stalled pipeline.** A stage
    blocked in `ring.put` on a full ring only sees the cancel flag after that
    await returns, and on a stalled pipeline nothing is draining, so it never
    does. The release closes the rings, which is what wakes every blocked
    producer — the same thing `DELETE /v1/video/enhance/{id}` already did,
    now done in both places.

Teardown is itself bounded (`teardown_timeout_s`, default 30 s) and escalates
to a cancel: a release that will not finish must not leave the reservation
held by a job nobody is watching.

**What is not built.** During the grace window the job's resources stay
where they are. The directive asks for them to join the normal reclamation
ladder (idle tenant #341, pressure staircase #287, spill) rather than being
idly pinned, and that ladder is not wired to this tenant. Registered as a
follow-on.

Demonstrated in `test/registered/video_enhance/test_liveness.py`: a consumer
takes one chunk and then stops — no close, no further read — and within the
configured timeout the executor is cancelled, every stage is closed, and the
decoder, which was willing to produce 100000 frames, is stopped within one
ring's worth of the frame it had reached. The control case, a consumer that
keeps reading, runs to completion with `declared_dead` false.

---

## 12. Pull-queue scheduling (user directive 2026-07-31)

§9.3's executor runs the plan `shard_plan` hands it: one chunk per card,
sized by a measured rate table. The directive replaces the assignment step —
the timeline is cut into more items than there are cards, the items name no
card, and each card takes the next one from a shared list whenever it is
free. Both modes now run through the same executor, selected by whether
`cards=` is passed, and the pre-weighted mode is kept rather than deleted
because it is the baseline the new one is measured against.

### 12.1 Why the seam is not at risk, and how that is shown rather than said

Nothing in the seam convention reads the card. `start`, `stop`,
`pulls_successor`, `output_frames` and `encodes()` are functions of the
timeline split alone, so binding a card late cannot move a frame — a chunk
interpolates the pair that straddles its boundary and withholds the trailing
original whichever card takes it, and whether or not its neighbour lands on
the same card.

That argument is one line and would be easy to believe wrongly, so it is a
test rather than a paragraph: every item of an eight-item queue is placed on
each of three cards in turn and the owned, pulled and encoded frame sets are
required to be identical, including `encodes()` over every `(frame, sub)`
pair in range. `verify_chunk_arithmetic` then runs on the queue before a card
is touched, exactly as it does for a weighted plan, and the per-chunk output
count is re-checked against every result.

The card-side re-proof is the same instrument §9.4 used and is the one that
grades this change: a sha256 per frame taken immediately before the encoder,
comparing the pull-queue run against the un-chunked whole-clip run **on the
same card**. A finer queue is more seams — one per item boundary rather than
one per card — so the convention is under more load here than it ever was
under the weighted plan.

### 12.2 What pull scheduling actually buys

Stated as properties, because the throughput number on this rig is bounded by
the same 1.78x card ceiling §9.4 derived and pull scheduling does not move a
ceiling:

*   **No rate table.** `capacity_weighted_plan` refuses to run without a P1
    measurement, and the bench spends a calibration pass per card producing
    one. A pull queue needs none.
*   **Correct under a rate the measurement could not have known.** A thermal
    cap, an LLM co-tenant that arrives mid-job, a clip whose content is
    harder than the calibration clip's. A weighted plan commits before the
    first frame and cannot revise; a queue revises continuously. There is a
    test for exactly this: a card that is fast for its first item and slow
    afterwards ends up with fewer items than its neighbours.
*   **Worst-case imbalance is one item, not the error in a rate estimate.**
    That is what the item count buys, and it is the only reason to want more
    items than cards.

### 12.3 The persistent worker, which is what makes the queue affordable

§9.3 recorded roughly 8 s of torch and ONNX Runtime import per chunk process,
and named it as the reason the end-to-end speedup (1.44x) sat below the
compute-only one (1.64x). A queue of four items per card through
`SubprocessChunkRunner` would pay that eight seconds four times per card and
hand back more than the scheduling could win.

`PersistentChunkRunner` therefore keeps one worker process per card and feeds
it items over a pipe (`chunk_worker --serve`). What is paid once per card
instead of once per item: the import, the CUDA context, the ONNX Runtime
session build and the RIFE weight load. What is necessarily rebuilt per item:
the decoder, which seeks to a different frame, and the encoder, which must
open its own session so each item's elementary stream begins with its own
parameter sets and an IDR — the property §9.3's concatenation argument rests
on. Reuse of the rest is safe because those stages hold no cross-frame state:
the sliding pair RIFE consumes lives in `PipelineExecutor._ArityWindow`,
constructed fresh per `run()`, so no frame of item *k* can reach item *k+1*.

Two things the protocol had to get right, both of which would present as a
hung card rather than as an error:

*   **The report channel has to be a channel, not a convention.** The
    one-shot worker got away with "the parent parses the last line" because
    there was exactly one report and it came last. A serving worker
    interleaves reports with whatever torch, ONNX Runtime and ffmpeg print,
    so the worker dups fd 1 for reports and points fd 1 at fd 2 — the parent's
    stdout pipe then carries reports and nothing else, without asking every
    library in the process to be quiet. Reports also carry a prefix, so both
    ends of the hazard are shut.
*   **A worker's stderr must be drained.** A worker whose stderr pipe fills
    stops writing and therefore stops working, which looks exactly like a
    slow card. It is drained into a bounded tail that becomes the error
    message when a worker dies, and a death mid-item is reported rather than
    waited on.

### 12.4 The spool bound under a queue, and why it cannot deadlock

The bound is unchanged — at most `spool_chunks` completed-but-unforwarded
items — but the argument for it is new, because under a queue a card is not
tied to one item. A card acquires its spool slot *before* it takes an item,
never after. Items therefore leave the queue in exactly the order slots were
acquired, so a card blocked on a slot is always waiting behind items with a
*lower* index than the one it is about to take, and those are precisely the
ones the forwarding loop is working through and releasing. A card can never
be blocked behind work that is itself blocked behind that card.

Written as a test that would hang rather than fail if the order were wrong,
so it runs under a timeout: twelve items, three cards, one spool slot.

### 12.5 Measurements

Raw records in `TASK_333_M2_MEASUREMENTS.md` and
`/spinning/gpu-battery-results/2026-07-31_339_pull/`. The bench
(`scripts/video_enhance/multicard_bench.py --schedule both`) runs the
weighted arm and the pull arm against the same single-card baseline, so the
two are comparable to each other and not only to themselves, and keeps
`--persistent-worker` separable from `--schedule` so a number can be
attributed to the scheduling or to the worker lifetime rather than to both.

**The gate passed, at two queue lengths.** 96 source frames as 9 items with
8 interior seams: 191 of 191 output frames bit-identical to the un-chunked
whole-clip run on the same card. 240 source frames as 6 items with 5
interior seams: 479 of 479. The weighted arm ran in the same two invocations
and also passed, so making the card late-bound did not disturb the path that
was already there. Eight interior seams on a 96-frame clip is four times the
seam density §9.4 measured, which is why it was cut that finely — to load the
convention, not to be fast.

**The balance is real and needed no measurement.** Item counts came out
5090-heaviest in both runs (4/3/2 and 3/1/2) from a scheduler given no P1
rate table at all.

**On throughput the pull queue lost to the weighted plan here — 1.10x
against 1.29x on the 240-frame arm — and the reason is quantisation, not
overhead.** The weighted plan had a calibration pass and can cut at any
frame, so it split 115 / 62 / 63 and finished its three cards within 1.6 s of
each other. Six whole items cannot express that 1.85 : 1 : 1 ratio; the queue
landed on 3 : 1 : 2 and one 3080 carried 80 frames where the plan gave it 63,
which set the makespan. Doubling the item count halves that rounding error;
what stops the item count from rising is the fixed per-item cost, and
reducing it is §12.7 rather than something claimed here.

Two conclusions the numbers do **not** support. First, that pull scheduling
is worse: it lost to a rate table measured minutes earlier on the same clip
on the same idle cards, which is precisely the condition where a
pre-weighted plan is at its best and adaptivity buys nothing — the cases
§12.2 exists for were not what this rig was asked to produce. Second, that
the ratio transfers: these cards differ by ~1.85x on this chain, which is
what makes integer quantisation expensive, and on equal cards the ideal
split is itself an integer one.

### 12.6 A gate that was failing on a threshold the specification disowned

`multicard_matches_same_card_control` was a pass/fail check at 40 dB. It
scored 37.0 dB, so the harness returned non-zero on a run whose frame counts
and seam digests were all green.

40 dB is the floor `parity.py` derives for fp16 against fp32 **on one card**.
The check applies it to a comparison that holds chunk boundaries and GOP
structure fixed and varies the architecture, whose convolutions are not
bit-identical — and §9.4 above had already recorded 37.4 dB for exactly this
comparison and called it "cross-architecture arithmetic, measured, not
passed". The code was failing runs against a number its own specification
had rejected.

It is now `multicard_vs_same_card_control` / `pull_vs_same_card_control`,
recorded with the reasoning attached and not graded. The gate is the digest
comparison, which is exact and which no cross-architecture argument can
soften.

### 12.7 What pull scheduling does not do

*   **The per-item fixed cost is not reduced, only amortised across items on
    one card.** A decode seek, an encoder session and one discarded seam
    frame are still paid per item, and they are what caps the item count —
    which is in turn what caps the balance. The 240-frame arm lost 0.19x to
    integer quantisation for exactly this reason. Registered as the next
    post: the encoder session is the expensive half and is rebuilt only
    because each item's stream must start with its own IDR, which does not
    obviously require a new session.
*   **No work stealing.** An item is taken whole. A card that takes the last
    item and turns out to be slow sets the makespan, and nothing splits that
    item out from under it. The tail is therefore bounded by one item's
    runtime, which is the bound §12.2 claims and no better.
*   **No cross-job queue.** The queue is per job. Two concurrent jobs on the
    same cards do not share one list, so they balance against each other only
    through the reservation, not through the scheduler.
*   **The item count is a configured constant, not derived.**
    `DEFAULT_CHUNKS_PER_CARD = 4` with a `MIN_PULL_CHUNK_FRAMES = 8` floor.
    Deriving it properly needs the per-item fixed cost as a measured number,
    which is the same post as the first bullet.
*   **A worker is not restarted if it dies.** The run fails with the worker's
    stderr tail rather than re-queueing its item on another card. Re-queueing
    is only safe once an item is known to have produced no output, and the
    spool file makes that knowable — but it is not implemented, and a silent
    retry of a half-written item would be worse than the refusal.

---

## 13. Live preview taps (§8.1), built in #344a

`preview.py`, attached in `pipeline.py`, served from `server.py` at
`GET /v1/video/preview/{job_id}/{input|output}`. Off unless the enhance
request passes `preview: true`, so a request that does not ask for one builds
no tap and takes the path it always took.

**The rule is structural, not a tuning goal.** `PreviewTap.offer` is an
ordinary synchronous method with no `await` anywhere in it, so the chain
cannot be suspended by a tap even for one event-loop turn. That is why the
ingress buffer is a plain list with a hand-written drop rule rather than
`BoundedRing`, whose `put` is a coroutine by design — even its drop path takes
an `asyncio.Condition`, which is an await, which is a place a stall can live.
The buffer drops the *oldest* frame, because a preview frame whose moment has
passed is worth less than the one arriving now. Back-pressure exists but is
confined to the lane: a viewer who stops reading fills the byte ring, stalls
the preview encoder, and the tap then drops on ingress.

Frames are dropped, never bytes. An H.264 elementary stream cannot survive
having byte ranges removed from its middle, so decimation happens before the
encoder and a viewer always receives a well-formed stream.

Taps attach by stage kind rather than by position, because the chain is
request-level configuration and a tap pinned to "the third stage" would tap a
different thing per request. Input is the output of `color_to_rgb`; output is
the last RGB stage before `color_to_yuv`.

### 13.1 What a tap costs, measured

**6.84 percent of main-chain throughput against a 3.44 percent noise floor —
measurable.** Both lanes, `fps_divisor` 1, on the 5090, with the output
byte-identical either way. It is a finding, not something to smooth away: the
downscale and the side encode do not *block* the chain, they *compete* with
it for the device, and the structural argument was never going to settle
that. `fps_divisor` is the lever, and the preview lane holds 100 percent
delivery on the input side and 78.5 percent on the output side against a
viewer that never reads a byte. Numbers and method in
`TASK_333_M2_MEASUREMENTS.md`.

### 13.2 The failure mode this feature has, named

A preview that fails is deliberately swallowed so it cannot take a job down —
the job is the product and the preview is a convenience. The cost of that
choice is that **a broken preview presents as a perfectly healthy job with a
silent tap**, and nothing in the pipeline's own statistics shows it. Two
separate defects hid there during this build and both were found by the
throughput measurement reading the preview's counters, not by the tests: a
missing `offer` delegation that raised `AttributeError` on all 861 frames,
and a default encoder backend that selected the PyNvVideoCodec path §9.5
records as broken. Both are fixed, both are pinned by tests, and the general
lesson is recorded here: the preview's own counters are the only place its
health is visible, which is why `GET /v1/video/enhance/{id}` reports them
next to the pipeline's.

### 13.3 What is not built

*   **No preview for a job that did not ask for one.** The tap map is fixed
    when the executor is built, so a viewer cannot attach to a running job
    that started without `preview: true`. The endpoint returns 409 saying so
    rather than an empty stream a client would read as a stalled encoder.
*   **No container, no HLS/DASH.** The body is a bare elementary stream,
    which is what lets a player start on the first IDR with no duration to
    declare. A browser `<video>` element wants a container; the extension
    client would need one, and that is a follow-on.
*   **No adaptive `fps_divisor`.** The cost is measured and the lever is
    exposed, but nothing turns it automatically when the chain falls behind.

---

## 14. Adaptive chain planning (#451)

> Extended by §17: the RIFE version is now chosen by the #460 ladder rather than configured, and §17.4 adds the stage-pipeline regime alongside the Regime-A pricing described here.

`chain_policy.py`. `plan_job` answers "does the chain the caller asked for
fit"; this answers "which chain should the caller have asked for", which a
client cannot answer for itself because it would need the per-stage rate
table, the reservation formula and the geometry rules.

### 14.1 The four shapes and how one is picked

`full`, `rife_only`, `pre_downscale` and `decimate_resynth`. Which of them
*exist* for a request is geometry, not economy: a `rife_only` chain on a
source below the target is not a cheaper option, it is a chain `build_chain`
refuses to build. Which of the existing ones is *chosen* is
`Candidate.quality_cost` -- tier, then detail thrown away, then input frames
thrown away -- with throughput headroom only breaking ties between shapes
that cost the viewer the same. That is what makes `full` automatic rather than
configured: it is tier 0 with no losses, so nothing below it can outrank it
however much faster the cheaper shape is. A test asserts exactly that against
a table where the pre-downscale ladder is eleven times faster and still loses.

`pre_downscale`'s entry point is **solved, not guessed**: the ladder is the SR
input resolutions somebody measured, filtered to those that still reach the
target after the x`sr_scale` upscale, and the boundary moves when one row of
the table moves. `decimate_resynth` is opt-in by name and reports both the
fraction of input discarded and the fraction of output that is synthetic.

### 14.2 Two shapes are recommendable and not runnable, and say so

`build_chain` places resize strictly after SR (§8.1), so a pre-SR resize is
not expressible as a chain at all. The nearest thing that exists is the ffmpeg
decode backend, whose command already carries `scale=W:H` -- but
`codec.DecodeStage._open` refuses outright when the source size differs from
the planned size, the PyNvVideoCodec backend has no scaler, and ffmpeg's
`scale` is not the chain's Lanczos-3. Decimation is worse off: `DecodeStage`
takes a contiguous `start_frame`/`frame_limit` range and has no stride at all.

Both are therefore named requirements on the candidate rather than silently
available, and `PolicyRequest.require_runnable` -- which the HTTP surface sets
unconditionally -- excludes them. The policy is still allowed to *answer* that
a pre-downscale would fit, because that is the answer, and a caller that has
to start a job filters it out rather than discovering at runtime that the
decoder ignored the plan.

### 14.3 Three pricing decisions that would otherwise flatter a mode

*   **A RIFE cell is per interpolated frame, not per pair invocation.**
    `shard_plan.stage_stream_factors` returns one invocation per pair at any
    multiplier, which is right at x2 -- the only multiplier it was written
    for -- and would make every decimated candidate free at the RIFE stage.
    The planner charges `arity_out`.
*   **The decoder is charged for the frames decimation throws away.** A
    decoder cannot skip them, and the decode column is what a high input rate
    makes expensive.
*   **A pre-downscale is charged decode at the source size.** The chain's own
    `source` is the entry resolution, so pricing every stage off the chain
    would credit the mode with a decode saving it does not make.

Each of the three has a test, and each test was shown to fail against the
version of the code without the correction.

### 14.4 What is unpriced, and what is absent

The two colour conversions are in no probe grid and are reported as
`unpriced_stages` on every candidate rather than as free. They are the same
two stages in every candidate, so they cannot reorder a ranking, but an
absolute fps figure from this module is a chain-stage figure and slightly
optimistic.

A stage that *should* have a row and does not makes the candidate
UNPRICEABLE, and the refusal names it. `allow_estimates` prices it instead by
a linear-in-pixels extrapolation off a measured row of the same stage on the
same card, and then the whole decision is labelled `estimate`. An
extrapolation still needs a row to start from: a stage with no measurement at
any resolution stays `absent` with the flag on.

**The shipped P1 report is not enough to price a full chain**, and the test
suite says so with the real files rather than in prose:
`docs/dev/measurements/333-m2/` has SR, resize and RIFE on one card and no
decode or encode row at all, so `choose_chain` against it refuses -- with
`allow_estimates` too.

### 14.5 What the tipping points are worth today

The synthetic tables in `test_chain_policy.py` are anchored on the measured
5090 column and the §9.4 per-stage ratios, with decode and encode invented
because nothing measured them. Against that table and this rig the target
scenario (1080p at 25 fps to 2160p at 50 fps) does **not** reach `full`: the
three cards aggregate 8.9 chain fps against 25 required, and the planner drops
to a 960x540 SR entry point at 27.2 fps -- a shape the executor cannot run.

That is a statement about the fp32-parity-era numbers, not about the feature.
The operating point is the fp16/bf16 TensorRT engine from the pinned ONNX and
RIFE 4.26; neither is measured, so **the production tipping points wait on
that measurement**. Nothing in the module hard-codes a threshold, so the move
is a re-read of the table.

---

## 15. Streaming input, the desk half (#448)

`streaming.py`. Everything upstream of it assumes a finished source, and the
failure when that is not true is not a clean one: a growing file simply ends
at whatever the writer had flushed and the job looks successful over a prefix.

**The admission is the point.** Three refusals, each of which would otherwise
be a job that looked like it worked:

*   A growing or live source on the chunk executor. `verify_chunk_arithmetic`
    checks the whole split against a final frame count before a card is
    touched, and such a source has none; cutting it against the count it
    happens to have right now plans for a prefix. The multi-card seam and the
    scheduler are untouched -- this is the gate in front of them.
*   A chunked run with no frame count at all, for the same reason.
*   A live source under the `stall` overload policy. Stalling is only
    back-pressure when the producer can be slowed down; a live feed cannot be,
    so the frames arriving during a stall are lost either way and the only
    difference is that `drop_frames` counts them. Silent loss is what §8.4
    rule 4 exists to prevent.

**The watermark is a duration.** A file job keeps the depth-1 response bridge,
because anything deeper is a buffer between the socket and the chain that
back-pressure must cross. A streaming job accepts that crossing deliberately
in exchange for not underrunning a player, and what it accepts is stated in
seconds and converted to frames through the output rate.

**"Not yet" is not "no more."** `growing_frames` is the adapter that keeps the
two apart, with an idle timeout measured from the last *frame* rather than the
last call. It is a pull source and buffers nothing, so the executor's rings
remain the only thing bounding memory.

**Sustained rate is sampled off the hot path.** `RateWindow` takes the
pipeline's own cumulative counters at two moments that are already outside the
chain: when the job status is rendered, and after a chunk the transport has
accepted -- the same moment the liveness watchdog uses, and for the same
reason. Fewer than two samples reports `None` rather than 0.0, because a live
watch showing zero for "not measured yet" is reporting a stall that is not
there.

### 15.1 What the desk half does not do

*   **No decode-side streaming backend.** `growing_frames` adapts a producer;
    nothing yet makes `codec.DecodeStage` into one that answers `NOT_YET` on a
    file that is still being written. That is the card-side half of #448.
*   **No cross-chunk streaming.** By construction, per the first refusal.
*   **The watermark is not adaptive.** It is declared and bounded; nothing
    widens it when the measured in/out rates say the chain is falling behind,
    although `RateWindow` is the instrument that would drive it.

---

## 16. Target scenario and admission decisions (2026-08-02/03)

### 16.1 The latency release

The gate this module admits against is **aggregate output fps >= input fps,
summed across cards** — not a per-frame `<= 40 ms` deadline. A few seconds of
steady-state lag behind real-time is acceptable; what is not acceptable is
falling permanently behind, which the aggregate-throughput gate already
catches on its own (a chain that cannot sustain the input rate accumulates
backlog without bound, and the back-pressure policy of §4/§8.4 is what a
sustained-lag job actually hits). This is a release of the framing used
through §14's tipping-point language ("does not reach `full`" reads there as
a per-target-fps question); it is not a change to `chain_policy.py` itself,
which already answers in aggregate fps.

### 16.2 The chunk executor is the primary live design

The full chain per chunk runs on-card: compressed source in, encoded segments
out. This structurally satisfies the NVENC -> NVDEC transport idea from
§10's prior-art comparison without building it as a separate mechanism — no
raw frame ever crosses a card boundary, because a chunk's chain (decode
through encode) stays on the one card it was scheduled to. barlink BAR1 and
lossless-HEVC intermediates are the named escalation tools **if** a stage
ever needs to split across cards (a chunk larger than one card's chain
budget, or a future per-stage placement rather than per-chunk); neither is
needed for the chunk-executor design as it stands, and neither is built for
this scenario.

### 16.3 Target: 1080p@25 -> 2160p@50, and the budget math

> Superseded in part by §17.5, which re-derives this target on the 2026-08-03 measured numbers under the stage-pipeline model. The 1.78x-ceiling arithmetic below is the Regime-A framing and is kept as the record of how the budget was first reasoned about.

Input 25 fps means a 40 ms budget per input frame at 1x. The rig's own
measured aggregate ceiling against a single 5090 is **1.78x**
(`TASK_333_M2_VIDEO_ENHANCE.md` §9.4 / `TASK_333_M2_MEASUREMENTS.md`
§"three-card weighting": "Ceiling from the measured rates: 1.78x. A 3080 is
0.39 of a 5090"), so the per-input-frame budget the rig's aggregate throughput
actually buys is `40 ms x 1.78 ~= 71 ms`.

Two of the three stages in the 2160p-target chain are measured
(`TASK_333_M2_MEASUREMENTS.md` P1/P4, single-card fp32/fp16 figures, 2026-07-31
window, 2.77 % A-vs-A floor):

| stage | measured | source |
|---|---:|---|
| Lanczos-3 resize, 4320p -> 2160p | 24.4 ms/frame (24.37 ± 0.14) | P1 table |
| RIFE pair @ 2160p, `scale=1.0` | 20.7 ms/pair (20.68 ± 0.12) | P1/P4 table |
| RIFE pair @ 2160p, `scale=0.5` | 11.4 ms/pair (11.40 ± 0.10) | P1/P4 table |

That leaves `71 - 24.4 - 20.7 ~= 25.9 ms` (~25 ms) at `scale=1.0`, or
`71 - 24.4 - 11.4 ~= 35.2 ms` (~35 ms) at `scale=0.5`, for the fp16/bf16-TRT
SR step — **the one unmeasured number** in this chain. §14.5 already recorded
that the fp32 ONNX Runtime SR figures (35-146 ms/frame at 960x540-1920x1080,
P1 table) are the wrong operating point for this budget; the TensorRT engine
built under #337 is the number this budget needs, and it has not been probed
against this specific chain yet.

### 16.4 RIFE `scale=0.5` as the 4K-point default candidate

The capability frontier (`TASK_333_M2_MEASUREMENTS.md` §"Capability frontier",
RIFE-only, 5090 alone) is the frontier this default is read off:
**48.4 fps at `scale=1.0` against 87.8 fps at `scale=0.5`** for a 3840x2160
input. §16.3's SR headroom follows directly — 25 ms vs 35 ms — from the same
two rows. `scale=1.0` at 4K/48 sits within 0.8 % of its own measured ceiling
(§"Capability frontier": "well inside the noise floor ... at the limit, not
above it"), so `scale=0.5` is the candidate that leaves the fp16/bf16-TRT SR
step room to be measured at all, not only the candidate with a faster RIFE
pair. Not yet a shipped default — it is a candidate pending the SR engine's
own measured ms/frame at this chain's resolutions, per §16.3.

## 17. The RIFE version ladder (#460) and stage-pipeline pricing (#457 desk)

Both halves land on `feat/video-rife-ladder-460`. §14 (adaptive chain planning)
and §16 (the target scenario) are the sections they change; nothing in §1-§13
moves.

### 17.1 User directives, 2026-08-03

Three, recorded here because they define the scope and each one is answered by
a specific mechanism rather than by a general intention.

1.  **A selectable RIFE version, auto-picking the best that can be computed.**
    The user's belief is that the 4.1x and lite families look better than 4.6
    and that the newest heavy variant is not required to win. So: a ladder,
    with an explicitly-labelled quality order, that climbs as high as the
    *measured* frontier allows and no higher.
2.  **NVENC/NVDEC is optional.** Raw frames may be pushed between cards or
    parked in host RAM. A few seconds of latency is acceptable; the gate stays
    aggregate output fps >= input fps (§16.1).
3.  **Prefetch.** The next frame a card is to work on can be spilled onto that
    card shortly before the current step ends, so there is no waiting time in
    between — the #125 double-buffer pattern applied to video frames.

### 17.2 The ladder: eight rungs off four vendored files

`video_enhance/rife_ladder.py`. Rungs: `4.6`, `4.15`, `4.15.lite`,
`4.16.lite`, `4.17`, `4.17.lite`, `4.18`, `4.26`.

The five new ones cost one new vendored file, because upstream ships several
IFNet files that are **byte-identical to each other** at the commit already
pinned in `_vendor/rife/README.md`:

| upstream sha256 | files | versions |
|---|---|---|
| `96816a3b…` | `IFNet_HDv3_v4_{15,17,18}.py` | 4.15, 4.17, 4.18 |
| `57ec7e07…` | `IFNet_HDv3_v4_{15,16,17}_lite.py` | 4.15.lite, 4.16.lite, 4.17.lite |

`rife._VENDORED` maps the aliases onto one module each. That is not the
substitution `require_supported` refuses — it is the same bytes under two
names — and the *weights* still differ per version and are pinned separately.
All eight rungs were loaded from their real checkpoints on CPU and produce
finite output; the alias mapping is validated by execution, not by reading.

Each rung carries four facts:

* **quality rank** — configurable, default in `DEFAULT_QUALITY_RANK`, and
  labelled `ASSUMPTION` in every report that prints it. Nothing in this tree
  has graded RIFE output. The default order is 4.26 > 4.18 > 4.17 > 4.15 >
  4.17.lite > 4.16.lite > 4.15.lite > 4.6, which is directive 1 turned into a
  total order with two mechanical tie-breaks (newer beats older within a
  family; a full variant beats its own `.lite` sibling). A quality gate — a
  PSNR/SSIM-against-ground-truth harness on held-out frame triples — would
  replace it through `with_quality_ranks` without touching selection code. It
  is a GPU ticket and it does not exist.
* **frontier** — `RifeFrontier`, keyed `(version, card, resolution, scale)`,
  cells are `Rate` with measured/estimate/absent. Separate from the shared
  `StageRateTable` because that table's key has neither version nor scale in
  it.
* **VRAM class** — headless / lite / standard / deep, derivable from the
  architecture. The measured peak bytes live in the frontier alongside the
  rates and are absent wherever P4 did not run.
* **weight state** — present on disk / pinned / unavailable. The registry
  *refuses* an entry that is neither present nor pinned, at construction time,
  because a rung that cannot be fetched reproducibly is a rung missing.

Seeded provenance, `seeded_frontier()`:

| version | 5090 | 3080 | provenance |
|---|---|---|---|
| 4.6 | 1080p s1.0/s0.5, 4K s1.0/s0.5 | same four | measured (ticket V) |
| 4.26 | same four (encode-cache *amortised*) | same four | measured (ticket V) |
| 4.15, 4.15.lite, 4.16.lite, 4.17, 4.17.lite, 4.18 | — | — | **absent** |

Six of eight rungs are absent. That is the state of the rig, and it is why the
"never auto-pick an unmeasured variant" rule has teeth rather than being a
formality.

### 17.3 How a version is chosen

`chain_policy.choose_chain` now nests two decisions. The outer one is the
version, and it deliberately does **not** derive a per-pair budget: splitting a
frame budget across three unequal cards under Regime A has no clean closed
form, and inventing one would be exactly the kind of quiet number this module
refuses. Instead the policy walks the ladder in quality order and asks the
*existing* aggregate-throughput gate whether the whole chain carries that
version. The first version whose chain is feasible wins.

* A variant with no measured frontier is never entered into the walk. It
  surfaces in the decision as `measure_first`, which is where TICKET_460's work
  list comes from.
* `pin_rife_version` overrides everything including an absent frontier — that
  is how a GPU window runs the variant it is about to measure — and the
  selection then reports `provenance=absent` so a pinned unmeasured run cannot
  be mistaken for a priced one.
* `rife_budget_ms` short-circuits the walk for a caller that has done its own
  arithmetic. Judged on the **slowest card in play**, because Regime A runs the
  same chain everywhere and a version too slow on the weakest card drags the
  aggregate.
* `auto_rife_version` is **off by default**, so an existing caller's stream
  does not change model underneath it.

One defect this exposed and fixed: the shared `StageRateTable` is keyed
`(stage, card, resolution)` and cannot tell 4.6 from 4.26. Priced off it, every
version would cost the same and the walk would be theatre. So when a ladder is
supplied *and it has a cell for this card at this geometry*, its version-keyed
frontier is the authority for the interpolation stage. When the ladder has
never heard of the card, it has no opinion and the shared table stands — which
is what keeps deployments whose probe reports use other card keys working.

### 17.4 Stage-pipeline pricing (#457)

`video_enhance/stage_pipeline.py`. `chain_policy` prices Regime A (each card
runs the whole chain over its own stretch; rates add). This prices Regime B:
stages are placed on cards and every frame walks the rig.

* Throughput is `1 / max(card load)`, not `1 / sum(stage costs)`.
* A stage boundary crossing a card is a transfer. No NVLink, no GPUDirect P2P
  (all PHB), so it is a host bounce: D2H over the sender's link, H2D over the
  receiver's, each half charged to the card whose link carries it. barlink BAR1
  is the named alternative and the house default transport wherever a
  combination supports it — but nobody has measured a BAR1 raw-frame move, so
  `barlink_link()` carries an **absence** and a plan that needs it is refused
  by name rather than priced with a guess.
* Directive 3 is `max(0, transfer − window)`, where the window is the receiving
  card's own compute times `prefetch_depth`. `prefetch_depth=0` prices every
  byte and is the pessimistic bound worth keeping next to the optimistic one.
* Deep buffering: `frames_in_flight` buys latency and nothing else
  (`latency_s = frames_in_flight / throughput`), bounded by `max_latency_s`
  when the caller gives one. The smoothness gate stays the aggregate.

Two hard constraints, enforced rather than reported:

* **Co-residency.** SR and the tail resize must share a card. The 8K fp16
  intermediate is 189.84 MiB, ~13.5 ms one way over x8 — more than the entire
  25 ms SR budget.
* **The x4 taboo.** A card may declare `max_transfer_mib`; at or above it, it
  is disqualified as a transfer endpoint. Default for an x4 card is
  `EIGHT_K_FP16_MIB`, derived from the geometry rather than typed in. Expressed
  per card, never as a hard-coded NVML index — enumeration order is not stable
  across boots.

### 17.5 Re-derived verdict: 1080p@25 -> 2160p@50 under the pipeline model

Cards: 5090 (x8), 3080 (x8), 3080 (x4). Stage table is ticket V, 2026-08-03,
per source frame; encode is x2 at 2160p on the **ffmpeg fallback** price
because direct NVENC fails Error 8 on both arches.

| stage | 5090 | 3080 (either) | provenance |
|---|---:|---:|---|
| decode 1080p h264 NVDEC | 4.254 | 7.140 | measured |
| SR 1080p -> 8K, TRT fp16 | 25.424 | 90.343 | measured |
| resize 8K -> 4K, Lanczos-3 | 24.367 | **absent** | — |
| RIFE 4.6 @4K s1.0 | 20.539 | 63.108 | measured |
| RIFE 4.6 @4K s0.5 | 11.359 | 31.999 | measured |
| encode 4K h264 x2 (ffmpeg) | 40.780 | 47.080 | measured |
| host link, one way | 13.70 GiB/s | 13.70 GiB/s (x8) / 6.85 (x4, **estimate**) | |

**At RIFE `scale=0.5`** the exhaustive sweep's best placement is

```
5090     sr + resize                 25.424 + 24.367 = 49.791 ms   <- binds
3080_x8  decode + rife               7.140 + 31.999  = 39.139 ms
3080_x4  encode                                        47.080 ms

crossings, all hidden at prefetch_depth=1:
  decode->sr    1080p NV12   2.97 MiB   0.21 ms each way   (window 39.14 / 49.79)
  resize->rife  4K fp16     47.46 MiB   3.38 ms each way   (window 49.79 / 39.14)
  rife->encode  4K fp16 x2  94.92 MiB   6.77 ms d2h, 13.53 ms h2d over x4
                                                           (window 39.14 / 47.08)

period = max(49.791, 39.139, 47.080) = 49.791 ms  ->  20.08 source-fps
```

**Binding card: the 5090. Binding stage: SR, at 25.42 ms — narrowly, against
resize at 24.37 ms on the same card.** Not encode, which is what bound the
serial figure in ticket V's RESULTS.md.

**At RIFE `scale=1.0`** the best placement is 5090 `decode+sr+resize` (54.05),
3080_x8 `rife` (63.108), 3080_x4 `encode` (47.08): **period 63.108 ms -> 15.85
source-fps, binding card 3080_x8, binding stage RIFE @4K.**

**Verdict: still not-FULL.** 20.08 < 25 at s0.5, 15.85 < 25 at s1.0.

Three things worth reading off this rather than the headline:

1.  **The pipeline verdict is fully measured; the replicated one is not.**
    Every cell in the s0.5 placement above is a ticket-V measurement, because
    resize lands on the only card that has a resize row. The Regime-A figure
    cannot say that: `replicated_throughput` with the default (strict) reading
    drops both 3080s for the absent resize row and reports 8.67 src-fps as a
    **lower** bound; with `omit_absent_stages=True` it drops the absent *term*
    and reproduces RESULTS.md's 18.299 src-fps as an **upper** bound. The true
    replicated figure is somewhere between, and nobody knows where, because the
    3080 resize row was never taken. The pipeline number needs no such caveat.
2.  **Which cells are estimate or absent.** The x4 link rate is an estimate
    (half the measured x8 rate, no transfer benchmark was run on that card) —
    but at `prefetch_depth=1` its crossing is fully hidden and contributes
    exactly zero milliseconds, so the pricer does *not* degrade the verdict's
    provenance to estimate. At `prefetch_depth=0` the same placement yields
    16.50 src-fps and *is* labelled estimate, because then the estimated half
    is actually paid. `color_to_rgb` and `color_to_yuv` remain **absent** at
    these resolutions and are carried as `unpriced_stages` on every report, so
    20.08 is a chain-stage figure and slightly optimistic. The 3080 resize row
    is **absent** and is the reason the replicated comparison has a bound
    rather than a value.
3.  **Fusing the tail resize moves the bind to encode.** `FUSED_TAIL_RESIZE_NOTE`
    in `sr.py` aims at exactly the 24.367 ms term. Setting it to zero:
    5090 `sr` 25.424, 3080_x8 `decode+rife` 39.139, 3080_x4 `encode` 47.080 ->
    **period 47.08 ms, 21.24 src-fps, binding stage encode.** Still short of 25,
    so the fused tail is necessary and not sufficient — the ffmpeg encode
    round-trip is the wall behind it, and that is the NVENC arm, not this one.

Stage-level replication — letting one stage's frames be split across two cards
in proportion to their rates — is **not built**. It would turn the placement
question into an LP, and the honest single-card-per-stage sweep is what is
priced here. It is the most obvious next lever: encode runs twice per source
frame and is the term that binds once resize is fused.

### 17.6 What is BOOT-PENDING

* The frontier for six of the eight rungs (4.15, 4.15.lite, 4.16.lite, 4.17,
  4.17.lite, 4.18) on both arches, 1080p/4K x scale 1.0/0.5. Spec:
  `docs/dev/TICKET_460_rife_frontier.md`.
* The RIFE quality order itself. Everything above treats it as an assumption.
* The host-bounce and prefetch-hiding rates as *observed* rather than modelled:
  the 13.70 GiB/s figure is a round-trip number reused one-way, and the x4 rate
  is halved arithmetic.
* The 3080 resize row, without which the Regime-A comparison stays a bound.
* `color_to_rgb` / `color_to_yuv` at 1080p and 2160p.
