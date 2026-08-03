# TICKET #484 — the NVENC card window

**STATUS: §1, §2, §3 and the ordering diagnosis are DISCHARGED.** The window
was taken on card 0 (RTX 3080) on 2026-08-03 09:28-09:55Z after `window-3`
released the rig; holder + heartbeat per `/spinning/gpu-arb/README.md`,
heartbeat stopped before the holder was removed, card back to 0 MiB.
Results are in `DESIGN_484_inprocess_nvenc.md` §6. What each section produced
is recorded inline below.

**What is still open: §4 at the chain's own geometry, and §5.**

The cards were held by `window-3` (all three, 19-30 GiB used each) for the
whole of the desk build, and `/spinning/gpu-arb/README.md` rule 2 — a card
above 500 MiB is an abort regardless of what the files say — was not
negotiable, so a request was filed at
`/spinning/gpu-arb/requests/2026-08-03T0930Z-484-nvenc-parity.txt`. The rig
freed at 09:28Z and the window was taken immediately.

## Prerequisites

* Worktree on `feat/inprocess-nvenc-484`.
* `/spinning/htsglang-gpu/.venv/bin/python3` — PyNvVideoCodec 2.2.0 is
  installed there; the system python does not have it.
* One card, any. A 3080 is enough and is preferred over the 5090.
* Under 900 MiB: one NVENC session, 60 frames of 720p NV12, plus the ffmpeg
  arm. Corridor rule (free >= 400 MiB) applies.

## §1 — The smoke, and it is the whole diagnosis

```
cd <worktree> && PYTHONPATH=python \
  /spinning/htsglang-gpu/.venv/bin/python3 - <<'PY'
import torch
from sglang.srt.video_enhance import codec
from sglang.srt.video_enhance.frame_math import Resolution
res = Resolution(256, 128)
rgb = torch.rand(3, 128, 256, device="cuda:0")
nv12 = codec.rgb_to_nv12(rgb)
backend = codec._PyNvEncodeBackend(res, device_id=0, bitrate=4_000_000)
print("packets:", [len(p) for p in backend.encode(nv12)])
print("flush:", [len(p) for p in backend.flush()])
backend.close()
PY
```

**RESULT (DISCHARGED).** No error 8: the device gate is passed, so the
DESIGN_484 §1 root cause is confirmed on hardware. Packets come back with a
six-frame lookahead (the first non-empty list is the seventh call) and the
counts balance exactly — 30 frames in, 30 packets out over `Encode` plus
`EndEncode`.

It did NOT work first time, and the second defect is the one worth carrying
forward: the frames were structurally fine and *pictorially wrong* because
NVENC reads the surface without ordering behind torch's stream. See
DESIGN_484 §1 "The second defect". Expect this class of bug from any
zero-copy consumer that is not a torch operator.

Watch for a second, different failure: the binding validates plane geometry
separately (`"Invalid shape: , expected: "`, `"Invalid strides: "` in the
extension's rodata). If one of *those* appears, the gate was passed and only
the plane descriptors are off — the `AppFrame` layout in DESIGN_484 §1 is
what to re-derive against.

## §2 — The parity gate

```
PYTHONPATH=python /spinning/htsglang-gpu/.venv/bin/python3 \
  scripts/video_enhance/nvenc_parity.py --device <idx> --frames 60 \
  --width 1280 --height 720 --out /tmp/nvenc_parity_484.json
```

Bounded by frame count, exits on its own, writes one JSON. Three arms:
`ffmpeg` (baseline), `inprocess`, `wrong-chroma` (the can-fail arm).

**RESULT (DISCHARGED, PASSED).** 15.65 dB against 15.65 dB, 60/60 frames,
`wrong-chroma` rejected at 11.22 dB. Full output in DESIGN_484 §6; raw JSON
`/tmp/nvenc_parity_484.json`.

Two things had to be fixed in the harness itself to get there, and both are
worth knowing before writing the next one:

* The bitstream needs a **container-appropriate suffix** on disk. ffmpeg
  identifies an elementary stream by extension when there is no container to
  read, and a misidentified file decodes to **zero frames rather than to an
  error** — which surfaced as a statistics exception, not as a decode failure.
* The decode side must be **software**. `-hwaccel cuda` fails the round trip
  on this rig with `cuvidCreateDecoder ... CUDA_ERROR_INVALID_VALUE` while an
  NVENC session is open on the same card. NVDEC is not a dependency of the
  thing under test and must not be one of the instrument.

## §3 — the session's own device memory

Measured as device-wide free memory (NVML, not the torch allocator — the #333
P3 lesson) around the session, with the input surface subtracted:

```
free_before = nvmlDeviceGetMemoryInfo(h).free
backend = codec._PyNvEncodeBackend(res, device_id=idx, bitrate=...)
backend.encode(nv12)          # the session allocates lazily; encode once
free_after  = nvmlDeviceGetMemoryInfo(h).free
session_bytes = free_before - free_after - nv12.numel()
```

**RESULT (DISCHARGED).** It moves with resolution, so the conditional fired:
51.8 MiB at 720p, 263.8 MiB at 2160p. `NVENC_SESSION_BYTES` is gone and
`frame_math.nvenc_session_bytes(resolution)` replaces it — an affine fit,
~25.3 MiB fixed plus ~30 B/px. Two points, two unknowns: it reproduces them
rather than being validated by them. **A third geometry is the first real
test of the shape and has not been run.**

## §4 — ms/frame, which is the number the whole task is for

**PARTIALLY DISCHARGED — the ratio is measured, the chain number is not.**

At 720p, three runs per lane in one session: ffmpeg 8.376 ms/frame mean
(range 1.333, 16 % of the min), in-process 4.881 (range 0.049, 1 %).
**1.72x**, with the 3.50 ms gap at 2.6x the noisier arm's whole spread.

**Still open:** the chain's encode column is 47.08 ms per SOURCE frame at
2160p with TWO output frames per source (ticket V,
`TASK_333_M2_VIDEO_ENHANCE.md` §17.7.4), not 8.4 ms at 720p with one. The
720p ratio is the transferable part; the chain verdict needs the measurement
at the chain's own geometry. Do not paste 1.72x into the stage table.

Feed the result into `stage_pipeline` as a new `encode` rate and re-derive the
verdict. §17.7.5's arithmetic says encode at ~0 gives 31.25 src-fps at RIFE
`scale=0.5`; the real number will land between that and 21.24, and where it
lands decides whether the chain clears 25 without stage replication.

## §5 — Only then, the default

`DESIGN_484` §3 lists the conditions. Flipping the `auto` default is a
separate commit with the JSON attached, not a side effect of this ticket.
