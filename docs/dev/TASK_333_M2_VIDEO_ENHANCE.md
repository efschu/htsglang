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
| `multicard.py` | the multi-card executor: chunk specs, the seam convention, ordered stitching | no |
| `chunk_worker.py` | one chunk, one process, one card | yes |
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

*   ~~**Single card for the executor.**~~ Closed by #339: `multicard.py` runs
    the planner's chunks concurrently, one process per card, and stitches
    them in timeline order. See §9.
*   **No live watch and no client-liveness handling.** Section 8 below is a
    user directive and is *not built*. The preview taps and the
    configurable dead-client timeout are open; a video job today is torn
    down when the response generator's `finally` runs, which covers a clean
    disconnect and does not cover a client that simply stops reading.
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
4. RIFE TensorRT backend at the post-resize shape.
5. VapourSynth shim and CLI over the same executor.
6. Audio-enhance stage (Demucs class) on the passthrough track inventory.
7. Registry integration (M1): replace the static budget with a ledger slot.
8. Bind a torch-owned output buffer to the SR session instead of cloning
   ONNX Runtime's. Needs the output shape threaded from the chain into the
   backend, which the stage knows and the backend does not — see §9.1.
9. Live watch and client liveness (§8), untouched.
10. Regime B, reopened by prior art rather than by argument — see §10.

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
