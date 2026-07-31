# Task #333 M2 — measurement records

Raw JSON in `docs/dev/measurements/333-m2/`. One card window, 2026-07-31,
NVML index 1 = RTX 5090 (`GPU-31d7ef41-f574-4d0e-21ad-e773fd938f6d`,
32607 MiB), driver 595.58.03. Cards 0 and 2 (RTX 3080, 20480 MiB) were not
probed in this window; the P1 table is therefore single-card and the shard
planner has no heterogeneous input yet.

A-versus-A noise floor, measured first on the resize kernel: **2.77 percent**.
Nothing below that is reported as a difference.

## P1 — per-stage ms/frame, fp32, ONNX Runtime CUDA provider

| Stage | Input | Options | ms/frame | ± | measured peak device |
|---|---|---|---|---|---|
| SR | 960x540 | x4 | 35.19 | 0.11 | 5.9 MiB torch-side only |
| SR | 1280x720 | x4 | 64.53 | 0.08 | 10.5 MiB torch-side only |
| SR | 1920x1080 | x4 | 146.29 | 0.07 | 24.0 MiB torch-side only |
| resize | 3840x2160 → 1920x1080 | lanczos3 | 6.27 | 0.11 | 239.2 MiB |
| resize | 5120x2880 → 3840x2160 | lanczos3 | 13.57 | 0.18 | 580.3 MiB |
| resize | 7680x4320 → 3840x2160 | lanczos3 | 24.37 | 0.14 | 950.5 MiB |

The SR peak column is **not** P3. `torch.cuda.max_memory_allocated` only sees
torch's allocator, and ONNX Runtime allocates through its own, so those three
numbers are the input tensor and nothing else. The probe was corrected to read
the device-wide free-memory delta instead, but the corrected run did not fit
in this window. **P3 is therefore not yet answered** and the §8.3 estimator is
unvalidated.

## P1 / P4 — RIFE, fp16, torch eager, version 4.6

| Input | scale | ms/frame pair | ± | per-pair device bytes (P4) |
|---|---|---|---|---|
| 1920x1080 | 1.0 | 5.98 | 0.32 | 1185.4 MiB |
| 1920x1080 | 0.5 | 5.48 | 0.21 | 785.9 MiB |
| 3840x2160 | 1.0 | 20.68 | 0.12 | 4740.7 MiB |
| 3840x2160 | 0.5 | 11.40 | 0.10 | 3140.2 MiB |

P4 has a value for the first time. It is large: a single in-flight 4K frame
pair at `scale=1.0` costs 4.7 GiB, which is why `plan_job` refuses to budget a
RIFE chain without it rather than guessing.

The `scale` arm behaves differently at the two resolutions. At 1080p the
difference between 1.0 and 0.5 is 8.4 percent — above the 2.77 percent floor,
but small. At 4K it is 45 percent and the footprint drops by 1.6 GiB. The
flow-scale knob is a 4K instrument, not a general one.

## Capability frontier (5090 alone, RIFE-only configuration)

Machine-readable via `probes.frontier_from_samples` / `aggregate_frontier`.

| Configuration | Resolution | scale | max output fps |
|---|---|---|---|
| rife_only | 1920x1080 | 0.5 | 182.4 |
| rife_only | 1920x1080 | 1.0 | 167.1 |
| rife_only | 3840x2160 | 0.5 | 87.8 |
| rife_only | 3840x2160 | 1.0 | 48.4 |

**4K interpolation to 48 fps output is sustained by the 5090 alone**, at
`scale=1.0`, with 0.4 fps of margin — and comfortably at `scale=0.5` (87.8
fps). 24 fps source doubled to 48 fps therefore needs no watch-ahead buffer
and no second card. `answer_capability(resolution=3840x2160, target_fps=48,
configuration="rife_only")` returns achievable with the 87.8 fps figure.

The margin at `scale=1.0` is 0.8 percent, well inside the noise floor, so the
honest statement is that `scale=1.0` at 4K/48 is *at* the limit, not above it,
and the flow-scale arm is what buys headroom.

## P2 — host-staged round trip, 5090

| Boundary | ms | GiB/s |
|---|---|---|
| 0.74 MiB | 0.121 | 12.21 |
| 2.97 MiB | 0.447 | 13.29 |
| 11.87 MiB | 1.742 | 13.63 |
| 47.46 MiB | 6.936 | 13.69 |
| 189.84 MiB | 27.722 | 13.70 |

This confirms the §8.2 arithmetic on the fast link: a single 8K fp16 frame
boundary is 27.7 ms of pure transfer per round trip, against a 146 ms SR
stage. Regime B before the resize remains arithmetically excluded. After the
resize the boundary is 11.87 MiB and the round trip 1.7 ms, which is where a
split would have to live if one is ever wanted. The x4-slot 3080 was not
measured in this window.

## End-to-end functional proof

Full chain on the 5090, 48-frame 960x540 source with two audio tracks and one
subtitle track, target 1920x1080, `fps_multiplier=2`, fp32, SR via the CUDA
provider, encode via the ffmpeg NVENC backend, remuxed to fragmented MP4.

| Check | Result |
|---|---|
| chain built | decode → colour → SR(x4, →3840x2160) → resize(→1920x1080) → RIFE → colour → encode |
| output geometry | 1920x1080 — **pass** |
| frame count | 48 in → 95 out, matching `expected_frame_count(48, 2)` — **pass** |
| track count | 4 in, 4 out — **pass** |
| audio track 0 bit-identical | sha256 equal — **pass** |
| audio track 1 bit-identical | sha256 equal — **pass** |
| subtitle track bit-identical | sha256 differs — **FAIL**, see below |
| duration | 2.000 s in, 2.0417 s out, delta 41.7 ms | one output frame interval at 48 fps, exactly the predicted bound |
| byte-stable across two runs | digests differ — **FAIL**, see below |

The chain works end to end: a clip goes in and an enhanced clip comes out,
with the interpolated frame count the retiming arithmetic predicts and the
duration delta bounded at one output frame. Two checks did not pass and are
open defects rather than tolerances:

1. **Subtitle stream hash differs.** Audio passes bit-identical, so the stream
   copy itself is sound; the subtitle track is `mov_text`, whose sample
   payload embeds timing that ffmpeg rewrites when the container timescale
   changes. Whether that is a genuine content change or only a container-level
   re-timing has not been determined. Not a re-encode — `-c copy` covers all
   outputs and the command test asserts no per-stream encoder flag exists.
2. **Output is not byte-stable across two runs.** Source clip generation is
   byte-stable (verified separately by the codec test). The variance is
   downstream: candidates are NVENC rate-control state, the fragmented-MP4
   writer, or non-deterministic kernel selection in the CUDA provider. Not
   diagnosed. This is M2 acceptance gate 1's second half and it is not met.

## Defects found and fixed during the window

*   **NVENC device input.** PyNvVideoCodec 2.2.0 probes its input for
    `__dlpack__` and calls it with the stream as a positional argument;
    torch's `Tensor.__dlpack__` takes it keyword-only, so a torch tensor
    raises `TypeError` inside the encoder and the next use of the session
    faults with an illegal memory access. Fixed by wrapping the tensor in a
    view exposing only `__cuda_array_interface__`. The wrapped form then
    reaches NVENC but is rejected with "incorrect usage of CPU input buffer"
    — **the PyNvVideoCodec encode path is still not working** and the
    end-to-end run used the ffmpeg NVENC backend instead.
*   **Warmup was never called.** The executor now warms every stage before the
    first frame moves, which is also where the capture-lock rule wants engine
    builds and graph capture to happen.
*   **Muxer descriptor passing.** The elementary stream was fed on a passed
    file descriptor addressed as `pipe:3`; `os.pipe()` does not return fd 3, so
    ffmpeg could not open it and died on the first write. Now fed on stdin,
    with the source opened by ffmpeg from its own path.
*   **Arity window.** A single-frame stage retained its previous frame and
    re-submitted every frame twice, multiplying the stream by 2 per stage.
    Caught by the executor test before any GPU run.

## Not measured

P5 (codec ceiling and concurrent session count), P6 (co-tenancy jitter against
a HOT LLM), P7 (the 8-bit transport matrix), the 3080s, the TensorRT-EP fp16
engine build and its parity gate, and the corrected P3. The window was spent
on P1/P2/P4 and on getting the chain to run end to end.

---

# Task #339 window, 2026-07-31

Raw JSON in `docs/dev/measurements/333-m2/multicard_480f.json` and
`multicard_96f_seam.json`. Same rig, NVML 1 = RTX 5090, NVML 0 and 2 = RTX
3080. Every number below was taken *after* the stream-ordering fix in
`backends.py`; numbers taken before it are not comparable and are not kept.

## The two defects that were open

**Byte stability.** Met. The falsifier and the fix are in
`TASK_333_M2_VIDEO_ENHANCE.md` §9.1. Evidence:

| Check | Before | After |
|---|---|---|
| whole clip at ring depth 1 vs 2 | 9 of 191 frames differ | 0 |
| whole clip at ring depth 2 vs 4 | 178 of 191 differ | 0 |
| two whole-clip runs, elementary stream sha256 | recorded as differing | identical |
| chunked vs whole clip, same card | 14 of 191 differ | 0 |

**mov_text subtitle.** Closed, and it was the visible half of a 21.333 ms
A/V lag. `TASK_333_M2_VIDEO_ENHANCE.md` §9.2. Measured on CPU:

| Check | `+empty_moov` alone | with `+delay_moov` |
|---|---|---|
| audio first packet | 0.0, no `Skip Samples` | -0.021333, `Skip Samples 1024` |
| subtitle cue timestamps | 0.021334 / 1.021334 | 0.0 / 1.0 |
| second video packet | 0.042236 | 0.020833 |
| inter-track drift vs source | 0.021334 s | 0.0 s |
| audio tracks bit-identical | yes | yes |
| subtitle content (gap samples stripped) | equal | equal |
| remux of one fixed elementary stream, 3 runs | identical | identical |

## Per-stage rates, both architectures

480-frame run, 960x540 → 1920x1080, SR x4 + Lanczos-3 + RIFE 4.6 x2, fp16
chain, SR on the CUDA provider, ffmpeg NVENC encode. ms per invocation.

| Stage | RTX 5090 | RTX 3080 | ratio |
|---|---|---|---|
| decode | 0.01 | 0.02 | — |
| colour to RGB | 0.22 | 0.48 | 2.18 |
| SR (960x540, x4) | 35.58 | 90.9 | 2.55 |
| resize (3840x2160 → 1920x1080) | 6.48 | 14.1 | 2.18 |
| RIFE (1920x1080, scale 1.0) | 3.06 | 8.28 | 2.71 |
| colour to YUV | 0.46 | 0.98 | 2.13 |
| encode | 1.56 | 2.5 | 1.60 |

The 5090 SR cell reproduces the earlier single-card P1 figure (35.19 ms) to
within the 2.77 percent noise floor. **A 3080 is 0.39 of a 5090 on this
chain.** SR is 71 percent of the 5090's per-frame time and 84 percent of a
3080's.

## Multi-card, 480 frames

| Arm | wall | composition |
|---|---|---|
| baseline, 5090 alone | 35.80 s | one chunk, same executor |
| three cards | 24.94 s | 5090 [0:243], 3080 [243:361], 3080 [361:480] |
| same-card control | 54.22 s | identical chunking, all three chunks on the 5090 |

*   end to end **1.44x**
*   compute only **1.64x** (excluding the ~8 s per-worker torch/ORT import
    from both arms)
*   ceiling from the measured rates **1.78x**; the weighting is running at
    about 92 percent of it

The projection was 1.8-2.6x. It is not reachable on this rig with this
chain — two cards at 0.39 add 0.78 of a card, and that is the arithmetic,
not an implementation shortfall. The ceiling calculation is what carries to
a rig with comparable cards; the 1.44 does not.

The same-card control is slower than the baseline (54.2 s vs 35.8 s) because
it pays three worker starts serially for work one worker did once. It exists
to control the encoder for the correctness gate, not as a performance arm.

## Multi-card correctness

| Check | 96 frames | 480 frames |
|---|---|---|
| output frame count | 191, expected 191 | 959, expected 959 |
| **seam exact** (pre-encode sha256 per frame, chunked vs whole, same card) | **191 of 191** | **959 of 959** |
| multi-card vs same-card control, PSNR | — | 37.4 dB |
| multi-card vs single-card baseline, PSNR | 35.5 dB | 36.5 dB |
| frames bit-identical to the 5090 baseline | 95 of 191 | 486 of 959 |

The seam row is exact, not approximate, and it is the gate. The two PSNR
rows are measurements: 37.4 dB is cross-architecture convolution arithmetic
with the encoder controlled for; 36.5 dB additionally carries the GOP
difference, because a chunk boundary forces an IDR the baseline does not
have.

The count of frames bit-identical to the baseline is exactly the 5090's own
chunk in both runs, which is the expected result and a useful cross-check
that the chunk-to-card assignment is what the plan said.

### The instrument had to be built

The first attempt compared the two arms by PSNR of the encoded output and
scored 6-25 dB on frames whose input pixels were provably identical. Two
independently encoded H.264 streams of the same content are not the same
stream, and at the ffmpeg `h264_nvenc` default rate the two are distorted
differently enough that the number says nothing about the seam. Raising the
rate to 150 Mbps moved the mean to 31.7 dB and did not fix it. Only the
pre-encode per-frame digest answers the question.

## Not measured in this window

P3 (corrected SR allocator overhead), P5, P6, P7, the fp16 TensorRT engine
and its parity numbers, and the PyNvVideoCodec encode path, which is still
rejected with "incorrect usage of CPU input buffer" and was not diagnosed
further. The 3080s were exercised for the first time here, but only as
whole-chain chunk hosts — there is still no per-stage 3080 sweep across
resolutions.

---

# Task #339 pull-queue window, 2026-07-31

Raw records in `/spinning/gpu-battery-results/2026-07-31_339_pull/`. Same
rig, same chain, same clip generator as the #339 window above: 960x540 →
1920x1080, SR x4 + Lanczos-3 + RIFE 4.6 x2, fp16, SR on the ONNX Runtime CUDA
provider, ffmpeg NVENC encode, 150 Mbps. NVML 1 = RTX 5090, NVML 0 and 2 =
RTX 3080. `CUDA_DEVICE_ORDER=PCI_BUS_ID` throughout.

## The seam under a pull queue — the gate

The comparison is a sha256 per frame taken immediately before the encoder,
between the queue run and the un-chunked whole-clip run **on the same card**.
Two runs, at two queue lengths:

| Clip | Queue | Interior seams | Frames | Result |
|---|---|---|---|---|
| 96 source frames | 9 items | 8 | 191 | **191 of 191 bit-identical — pass** |
| 240 source frames | 6 items | 5 | 479 | **479 of 479 bit-identical — pass** |

The weighted arm's `seam_is_exact` ran in the same two invocations and also
passed, 191 of 191 and 479 of 479, so the refactor that made the card
late-bound left the pre-weighted path intact.

Eight interior seams on a 96-frame clip is four times the seam density the
#339 window measured (two interior seams on 480 frames). The convention
holding at that density is the point of running it that way.

## Self-balancing, with no rate table

The pull arms were given no P1 measurement at all.

| Card | 96 frames, 9 items | 240 frames, 6 items |
|---|---|---|
| 1 (RTX 5090) | 4 | 3 |
| 0 (RTX 3080) | 3 | 1 |
| 2 (RTX 3080) | 2 | 2 |

The ordering is right in both — the 5090 takes the most — and it comes purely
from finishing sooner and asking again.

## Throughput: the pull queue loses to a well-calibrated plan here, and why

240 source frames, the arm that is long enough to mean something:

| Arm | wall | vs baseline |
|---|---|---|
| baseline, 5090 alone | 21.95 s | 1.00x |
| three cards, capacity-weighted | 17.04 s | 1.29x |
| three cards, pull queue (6 items) | 19.92 s | 1.10x |

Per-card busy seconds say exactly where the difference went:

| Card | weighted | pull |
|---|---|---|
| 1 (5090) | 15.40 | 15.98 |
| 0 (3080) | 16.98 | 13.27 |
| 2 (3080) | 16.93 | **18.78** |

The weighted plan is balanced to within 1.6 s across three cards, because it
had a calibration pass and could cut the timeline at any frame: it gave the
5090 115 frames and the 3080s 62 and 63, a 1.85 : 1 : 1 split. The queue can
only hand out whole items. Six equal items of 40 frames cannot express
1.85 : 1 : 1 — the nearest integer splits are 3 : 2 : 1 and 4 : 1 : 1 — and it
landed on 3 : 1 : 2, which gave one 3080 eighty frames where the weighted
plan gave it sixty-three. That card is the makespan.

So the gap is quantisation, not scheduling overhead, and it is a direct
function of item count: doubling the items halves the worst-case rounding
error. What stops the item count from rising is the fixed per-item cost — a
decode seek, an encoder session, and one seam frame pulled through the
pre-RIFE prefix and discarded. Reducing that is the lever, and it is
registered as a follow-on rather than claimed.

Two things this does **not** license concluding:

* **That pull scheduling is worse.** It lost to a rate table that was
  measured minutes earlier on the same clip on the same idle cards — the one
  condition under which a pre-weighted plan is at its best and adaptivity is
  worth nothing. The arms where it wins (no calibration available, a rate
  that changes mid-job) are not in this table because this rig was not asked
  to produce them.
* **That the numbers transfer.** This rig's cards differ by ~1.85x on this
  chain, which is what makes integer quantisation expensive. On cards of
  equal speed the ideal split *is* an integer one and the quantisation term
  vanishes.

The 96-frame arm (baseline 13.79 s, weighted 12.62 s / 1.09x, pull 14.59 s /
0.95x) is recorded only so it cannot later be mistaken for evidence. Four
seconds of video is far below where the #339 window could resolve 1.44x, and
nine items on it is deliberately past sensible granularity — that arm was cut
that finely to load the seam, not to be fast.

## Preview taps: what they cost the main chain (2026-07-31 19:57-20:00Z)

`scripts/video_enhance/preview_tap_bench.py --card 1 --frames 96 --reps 4`,
RTX 5090, both lanes on, `fps_divisor` 1, against a never-reading viewer.
A-vs-A noise floor first, then interleaved off/on arms. Raw records in
`/spinning/gpu-battery-results/2026-07-31_339b/taps_v3/`.

| arm | fps mean | stdev |
|---|---|---|
| taps off | 33.615 | 0.125 |
| taps on | 31.317 | 0.297 |

**Taps cost 6.84 percent of main-chain throughput against a 3.44 percent
noise floor — MEASURABLE, and reported as a finding rather than smoothed
away.** The output is byte-identical with and without the taps (same sha256
over the elementary stream), so the cost is contention for the device, not a
change to the deliverable.

Preview delivery in the same run, with a viewer that never reads a byte:

| lane | offered | encoded | dropped | delivered |
|---|---|---|---|---|
| input | 96 | 96 | 0 | 100% |
| output | 191 | 150 | 39 | 78.5% |

The output lane carries the interpolated frames, so it sees twice the rate and
is the one that drops — which is the §8.1 rule working: a viewer who does not
read costs preview frames and the job still finished with a byte-identical
output. `fps_divisor` is the lever that trades that 6.84 percent down.

### Two defects the measurement found that the tests did not

Both had the same shape, and it is worth naming: a preview that fails is
deliberately swallowed so it cannot take a job down, so **a broken preview
presents as a healthy job with a silent tap**. Only a measurement that reads
the preview's own counters can see it.

1.  **The first run measured nothing.** `offered: 0` on both lanes.
    `build_preview_lanes` returns `by_stage` mapping a stage to *lanes*, the
    executor calls `offer` on what it is handed, and `PreviewLane` had no
    `offer` — so all 861 frames raised `AttributeError` into the swallow. The
    unit tests all drove `PreviewTap` directly and could not see it. Fixed by
    delegating, and pinned by a test that drives what the server builds.
2.  **The second run had working taps and a dead encoder.** `offered: 96` but
    `encoded: 0`, with `incorrect usage of CPU input buffer` — the open
    PyNvVideoCodec defect from §9.5. The preview config defaulted its backend
    to `auto`, which selects PyNvVideoCodec; the main chain has always pinned
    ffmpeg for exactly this reason. The preview now pins it too.

The +6.84 percent above is the third run, the first one in which both lanes
actually encoded frames. The two earlier numbers (+1.05 percent with no
frames offered, +5.83 percent with frames offered and no encoder) are void
and are recorded here only so they cannot be mistaken for measurements.

## Per-card, per-stage rates — the Regime-B table (2026-07-31 19:48-19:53Z)

`scripts/video_enhance/stage_rate_sweep.py --cards 1,0,2`, one card at a time
in its own process, two precision passes per card (the chain's fp16 for
resize/RIFE/encode, fp32 for SR because the CUDA provider runs the pinned ONNX
at its exported precision). Raw records in
`/spinning/gpu-battery-results/2026-07-31_339b/stage_rates_v2/`.
NVML 1 = RTX 5090, NVML 0 and 2 = RTX 3080.

**RIFE on a 3080 is measured here for the first time.**

ms per invocation:

| stage @ resolution | 0 (3080) | 1 (5090) | 2 (3080) |
|---|---|---|---|
| sr @ 960x540 | 88.82 | 35.15 | 88.78 |
| sr @ 1280x720 | 157.10 | 64.44 | 156.57 |
| sr @ 1920x1080 | 352.54 | 146.12 | 351.75 |
| resize @ 3840x2160 | 13.97 | 6.22 | 13.92 |
| resize @ 5120x2880 | 25.55 | 13.96 | 25.27 |
| resize @ 7680x4320 | 48.80 | 25.16 | 48.52 |
| rife @ 1920x1080 | 8.97 | 5.16 | 8.76 |
| rife @ 3840x2160 | 30.85 | 11.24 | 30.71 |
| encode @ 1920x1080 | 19.59 | 20.79 | 22.30 |
| encode @ 3840x2160 | 34.86 | 32.64 | 36.86 |

Relative to the fastest card on each row — the comparative-advantage view,
which is the form the assignment question is actually asked in:

| stage @ resolution | 0 | 1 | 2 |
|---|---|---|---|
| sr @ 960x540 | 2.53x | 1.00x | 2.53x |
| sr @ 1920x1080 | 2.41x | 1.00x | 2.41x |
| resize @ 3840x2160 | 2.25x | 1.00x | 2.24x |
| resize @ 7680x4320 | 1.94x | 1.00x | 1.93x |
| rife @ 1920x1080 | 1.74x | 1.00x | 1.70x |
| rife @ 3840x2160 | 2.75x | 1.00x | 2.73x |
| encode @ 1920x1080 | 1.00x | 1.06x | 1.14x |
| encode @ 3840x2160 | 1.07x | 1.00x | 1.13x |

### What the table says

**The 3080 column is not flat, so comparative advantage exists.** Its
disadvantage ranges from ~1.0x (encode) through 1.7x (RIFE at 1080p) to 2.75x
(RIFE at 4K) and 2.5x (SR). Regime B is therefore not ruled out by the data,
which is the question this sweep was run to answer. A flat column would have
closed the post.

**The historical vs-pipeline mapping is contradicted by the numbers.** That
mapping put ESRGAN on the two 3080s and RIFE on the 5090. SR is the 3080's
*worst* stage (2.4-2.5x) and RIFE at 1080p is one of its *best* (1.74x), so
comparative advantage points the other way. The user's framing correction
that the mapping was a convenience of an early development state and not a
template is supported by measurement, not just by argument.

**Two nominally identical 3080s agree to within ~2% on every row**, which is
the independent check that the harness measures the card rather than the
session.

**The 5090's SR figure reproduces prior work**: 35.15 ms at 960x540 against
35.19 ms in the single-card P1 post and 35.58 ms in the §9.4 multi-card run.

### What the table does not say

*   **The encode row is below the measurement's resolution and must not be
    read as "a 3080 encodes faster than a 5090".** The spread across the
    three cards is 6-14%. Five of the six probe passes had noise floors of
    0.28-2.25%, but one — card 2's chain pass — came in at 23.28%, and the
    merged floor takes the worst contributor by design. The defensible
    statement is that encode is at *parity* across all three cards, which is
    still the finding that matters: encode does not scale with the card, so
    the 3080's disadvantage is stage-dependent. NVENC is fixed-function, and
    on this path the measurement is dominated by the ffmpeg host round trip
    (§9.5) rather than by the encoder.
*   **SR was measured on the ONNX Runtime CUDA provider at fp32**, not on the
    TensorRT fp16 path. The ratio may move when that engine exists, which is
    the open fp16-parity post.
*   **RIFE's 5090 figure here (5.16 ms at 1080p) is higher than §9.4's
    in-run 3.06 ms.** The probe times an isolated stage on synthetic tensors;
    §9.4 timed the stage inside a warm chain. Recorded as a difference rather
    than reconciled, because nothing yet says which is the number a planner
    should use.
*   **No optimiser consumes this yet.** The table is data; the assignment
    search over the full space (Regime A replicas, Regime B stage
    assignment, mixed forms) is not written.

## The PSNR gate that was never met and should not have existed

`multicard_matches_same_card_control` was declared a pass/fail gate at 40 dB
and scored 37.0 dB (weighted) and 38.1 dB (pull), so the harness returned
non-zero on a run whose actual gates were all green.

The threshold was a category error rather than a near miss. 40 dB is the
floor `parity.py` derives for **fp16 against fp32 on one card**. This
comparison holds the chunk boundaries and the GOP structure fixed and varies
the *architecture*, whose convolutions are not bit-identical — and §9.4 of
the task document had already recorded 37.4 dB here and described it as
"cross-architecture arithmetic, measured, not passed". The code was failing
runs against a threshold its own specification had disowned.

It is now recorded as a measurement with that reasoning attached, and the
gate is the digest comparison, which is exact. Worth noting in passing that
the pull arm scored *higher* than the weighted one (38.1 vs 37.0 dB), which
is consistent with more of its frames having been produced on the baseline
card.

---

# Task #339 P4 — fp16 SR parity, 2026-07-31 20:47-20:55Z

Raw records in `/spinning/gpu-battery-results/2026-07-31_339_p4/`. RTX 5090,
`CUDA_DEVICE_ORDER=PCI_BUS_ID`, inputs sampled on the CPU and moved to the
device (two architectures do not agree on `torch.rand`, so a device-sampled
input would make these numbers card-specific).

## The gate: both fp16 arms pass, comfortably

fp32 ONNX on the CUDA provider is the reference. Three samples per
resolution, `parity.py`'s fp16-vs-fp32 same-card floor: PSNR >= 40 dB,
SSIM >= 0.995.

| arm | resolution | PSNR dB | SSIM | verdict |
|---|---|---|---|---|
| full fp16 (weights + I/O) | 960x540 | 55.69 - 55.75 | 0.999961 | pass |
| full fp16 (weights + I/O) | 1280x720 | 55.68 - 55.71 | 0.999961 | pass |
| I/O cast only (fp32 compute) | 960x540 | 60.73 - 60.78 | 0.999988 | pass |
| I/O cast only (fp32 compute) | 1280x720 | 60.73 - 60.75 | 0.999988 | pass |

**The full conversion is numerically safe for this model**: ~15 dB of margin
over the bar. The I/O-cast fallback scores ~5 dB better, which is what it
should do — it leaves every computation in fp32 and only changes the
interface — but it therefore buys interface bandwidth and no compute. On this
evidence the full conversion is the one to use, and `--io-cast-only` stays
what it was declared to be: the fallback for a model the full conversion
cannot carry, which this model is not.

## What is NOT measured here, and why

**No TensorRT engine was built. TensorRT is not installed on this rig.**
`libnvinfer.so.10: cannot open shared object file`, no `tensorrt` package in
the venv, no `libnvinfer.so*` anywhere on the filesystem. Both arms above were
therefore graded on the **CUDA execution provider**, which runs the fp16 graph
in fp16 and so is a genuine fp16-vs-fp32 numerical comparison — but it is not
a TensorRT result and is not labelled as one.

Two consequences, stated rather than glossed:

*   The parity table above settles the **conversion's numerics**. It does not
    settle anything about the TensorRT engine.
*   **The SR ratio question is still open.** The per-stage sweep measured SR
    at fp32 on the CUDA provider and found the 3080 at 2.4-2.5x the 5090;
    whether an fp16 TensorRT engine moves that ratio is exactly what an
    engine would answer, and no engine exists. Installing TensorRT is a
    ~2-3 GB change to a shared box and is a rig decision, not one to make
    inside a borrowed card window.

## Three defects, all found by running the gate rather than reading

**1. The full-fp16 artifact had never been loadable.** It was built, hashed
and given a provenance sidecar in an earlier window, and it does not load:

    Type 'tensor(float16)' of input parameter (onnx::Resize_275) of
    operator (Resize) in node (Resize_68) is invalid.

The converter cast every float32 initializer. `Resize` pins `roi` and
`scales` to `tensor(float)` in its schema — they are geometry, not pixels —
so casting them produces a model the spec forbids. `onnx.checker.check_model`
accepted it, which is why nothing caught it at export time.

Fixed generally rather than with a list of op names: `schema_pinned_float_inputs`
asks each operator's own ONNX schema which of its inputs are type variables
and which are pinned, and leaves the pinned ones alone. The re-export converts
101 initializers where the broken one converted 102 — the difference is
exactly the `scales` tensor.

**2. "Built and hashed" did not imply "loadable".** The checker was the only
validation, so an unloadable artifact got a sidecar claiming success and sat
on disk looking finished. `export` now opens the artifact once on the CPU
provider before writing the manifest, and deletes the file rather than leaving
one behind that lies about itself.

**3. A parity record could claim a provider it never ran on.** The SR backend
appends `CUDAExecutionProvider` after the TensorRT EP so a subgraph the EP
cannot take still runs. On a host with no TensorRT that silently turns the
whole session into a CUDA one — it works, it is fast, and every record it
produces was labelled `tensorrt`. This is how a parity table comes to contain
TensorRT numbers taken on a machine where libnvinfer was never installed;
the first io-cast run in this window did exactly that and reported
`PARITY GATE PASSED` under a TensorRT heading.

`BackendInfo` now records `active_providers` and `provider_fell_back` from the
session itself, logs a warning naming both, and the parity rows carry
`candidate_provider` alongside `candidate_provider_requested`. Every row in
the table above is stamped `CUDAExecutionProvider` with `fell_back: true`,
which is why the caveat in this section is verifiable from the artifact rather
than resting on this prose.
