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
