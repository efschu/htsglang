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
