# #484 — In-process zero-copy NVENC, and stage-level replication

Two posts, and they meet on the same stage. The enhance chain's encode is
today an ffmpeg subprocess: an architecture violation under the ONE-RUNTIME
law, and — after #457's fused tail — the stage that *binds* the whole
pipeline at 47.08 ms per source frame. Removing the subprocess is therefore
not a tidiness exercise; it is the largest single lever left on the chain.

---

## 1. Why the in-process lane did not work, exactly

`TASK_333_M2_VIDEO_ENHANCE.md` §9.5 records the state this task inherited:

> **PyNvVideoCodec encode** — still not working. The `__dlpack__` signature
> problem is worked around, and the wrapped tensor then reaches NVENC and is
> rejected with "incorrect usage of CPU input buffer". […] Not diagnosed
> further.

It is diagnosed now, and the diagnosis is that **the workaround caused the
second failure**. Both symptoms come from the same wrapper.

`PyNvEncoder::Encode` in PyNvVideoCodec 2.2.0 selects its input path like
this (read off `PyNvVideoCodec_130.cpython-312-x86_64-linux-gnu.so`, the
binary that is installed in the project venv — verified byte-identical to the
`pynvvideocodec-2.2.0-cp312-cp312-manylinux_2_28_x86_64.whl` used for the
analysis):

```
e8639:  call  PyObject_HasAttrString    ; (frame, "cuda")   <- rodata 0x11c5f6
e863e:  cmp   $0x1,%eax
e8641:  je    e8b6a                     ; attribute present -> device path,
                                        ;   which evaluates frame.cuda()
e8647:  mov   0x10(%rsp),%rax           ; this
e864c:  cmpb  $0x0,0x64(%rax)           ; this->useCpuInputBuffer
e8650:  je    40a41                     ; absent AND GPU mode -> throw

40a41:  ...
40aa0:  mov   $0x8,%edx                 ; NVENC error code 8
40aa8:  mov   $0x33b,%r9d               ; PyNvEncoder.cpp:827
        -> "incorrect usage of CPU input buffer"   ; rodata 0x11c628
```

So the device path is gated on **`hasattr(frame, "cuda")`** — not on
`__cuda_array_interface__`, which is what the class docstrings advertise. The
message names the CPU buffer because the check is written from the CPU-mode
side; it says nothing about where the pointer actually lives.

The two failures then follow mechanically:

| input handed to `Encode` | `hasattr(·, "cuda")` | what happens |
|---|---|---|
| bare torch CUDA tensor | **yes** (`Tensor.cuda`) | device path taken, the object it probes exports `__dlpack__`, the binding calls it with the stream POSITIONALLY, torch declares it keyword-only → `TypeError` from inside the encoder, session faults on next use |
| `_CudaArrayView` (`__slots__ = ("_tensor", "__cuda_array_interface__")`) | **no** | falls through to the CPU branch, `usecpuinputbuffer=False` → **error 8** |
| `_NvencDeviceFrame` (this change) | **yes**, and no `__dlpack__` anywhere | device path, `frame.cuda()` returns the plane list |

The `__slots__` that hid `__dlpack__` hid `cuda` with it. Fixing symptom one
created symptom two, and because both were reported as "NVENC rejects our
input" they read as one intractable problem for three GPU windows.

### What the device path consumes

`frame.cuda()` returns a **list of per-plane views**, not one flat array.
NVIDIA's own reference (`samples/utils/Utils.py`, shipped in the wheel) makes
NV12 two planes:

```python
class AppFrame:
    def __init__(self, width, height, format):
        if format == "NV12":
            self.cai = [
                AppCAI((height,      width,      1), (width, 1, 1), "|u1", base),
                AppCAI((height // 2, width // 2, 2), (width, 2, 1), "|u1",
                       base + width * height),
            ]
    def cuda(self):
        return self.cai
```

The chroma row stride is the **full luma width**, because one chroma row
holds `width // 2` interleaved UV pairs. `version` is 3 and `stream` is
**omitted** — the binding rejects a `stream` of 0 by name.

`codec._NvencDeviceFrame` is that shape over the contiguous `(H*3//2, W)`
uint8 tensor `rgb_to_nv12` already produces: two views into one allocation,
no copy, no host round trip, and a reference to the tensor so the allocation
cannot be freed while NVENC holds the pointers.

### The second defect, which only a card could show

The gate fix made NVENC accept the frame. It did not make it encode the right
one. **NVENC reads the input surface on its own engine with no dependency on
the stream that produced it**, and it does not complain about reading a frame
whose kernels have not finished: it encodes what is there. 60 frames at 720p
on card 0, 2026-08-03, graded against an ffmpeg baseline of 15.65 dB:

| ordering | PSNR |
|---|---|
| none (hand the pointer straight over) | **8.59 dB** — silently wrong |
| `current_stream().synchronize()` before `Encode` | **15.65 dB** — parity |
| pass `cudastream=` at session creation | 9.43 dB — not honoured |

The third row is why the fix is a host-side wait rather than a shared stream:
PyNvVideoCodec accepts `cudastream` and does not order the input read behind
it. This is the failure mode a frame count cannot see — every arm delivered
exactly 60 frames — so `test_the_producing_stream_is_synchronised_before_
nvenc_reads` asserts the call order directly.

The same missing ordering is the best explanation for a segmentation fault
seen twice during bring-up: an unsynchronised read of a block torch's
caching allocator had already handed to the next frame. It has not recurred
since the wait was added (300-frame stress with deliberate allocator churn),
but that is absence of recurrence, not a reproduced cause.

### The falsifier

`test_codec._StrictFakeEncoder` is the branch above, in Python: it raises the
real message on a missing `cuda` attribute, raises the real `TypeError` on an
input exporting `__dlpack__`, and asserts the plane geometry. Against the
pre-#484 wrapper it produces

```
--- arm A: current #484 adapter ---
  encoded packets: 1
--- arm B: pre-#484 wrapper restored ---
  FAILED: RuntimeError incorrect usage of CPU input buffer
```

which is the recorded symptom reproduced from its cause. The old fake
encoder accepted anything, which is why three windows of hermetic tests were
green while the lane had never encoded a frame.

---

## 2. Binding decision and licence

**PyNvVideoCodec 2.2.0**, NVIDIA's official binding, from PyPI. **MIT**
(`pynvvideocodec-2.2.0.dist-info/licenses/LICENSE.txt`, SPDX
`MIT`). Already a declared dependency of this chain since #333-M2 and already
installed in the project venv, so this change adds no dependency at all.

A direct Video Codec SDK binding via ctypes/cython was the named alternative
and is **not** taken: the gap it would close does not exist. The binding
accepts foreign device pointers, the plane contract is discoverable, and the
only thing that was missing was knowing what it wanted. Nothing restrictive is
vendored into the repo.

---

## 3. The switch lever

The lane is **built and gated OFF**.

* `SGLANG_VIDEO_INPROCESS_NVENC=1` makes `backend="auto"` prefer the
  in-process lane where the container allows it (`annexb` only — the
  in-process encoder emits an elementary stream, and `mpegts` needs a muxer).
* Unset or `0` — the default — makes `auto` resolve to the ffmpeg subprocess,
  the **named bootstrap fallback**.
* `backend="pynvvideocodec"` ignores the switch. That is how a GPU window
  runs the very path it is about to grade.
* Under `auto`, a session that cannot be opened falls back to the subprocess
  with a logged warning and sets `EncodeStage.fell_back_to_ffmpeg`, so a
  measurement can never be attributed to the lane that did not run it. Under
  an explicit request it raises instead.

This also closes a smaller hole. Before this change, `auto` selected the
in-process lane whenever the *package was importable* — which is how the §9.5
defect reached a preview lane that had never asked for it and silently
delivered zero bytes (`TASK_333_M2_MEASUREMENTS.md`, "the second run had
working taps and a dead encoder"). Installing a package is now not a
deployment decision.

**The default flips when** `scripts/video_enhance/nvenc_parity.py` has been
executed on a card and both of these hold: the in-process arm reconstructs the
source within 1 dB of the ffmpeg arm over 60 frames with an exact frame count,
and the `wrong-chroma` arm is rejected by the same threshold.

Both held on card 0 on 2026-08-03 (§6). The default is nonetheless left on
ffmpeg in this change, and that is deliberate rather than timid: the gate ran
on ONE card, at ONE geometry, on the codec/preset pair the gate pins, and the
chain's own operating point is 2160p with two output frames per source frame.
Flipping the default is a one-line change plus that evidence, and it belongs
in the commit that carries the evidence.

---

## 4. The ledger post

The encoder session is an **asset class in the ledger**, not invisible
overhead. `frame_math.NVENC_SESSION_BYTES` is a named post that
`chain_reservation(..., inprocess_nvenc=True)` adds and `EncodeStage.
session_bytes` reports from the RESOLVED backend, so a fallback at open time
moves the number with it.

It is deliberately **zero on the ffmpeg lane**, and the reason is the point of
the whole task: that memory exists on the same card either way, but under the
subprocess it is held by a process this runtime does not own and cannot see,
park, or evict. Reserving for it here would charge the card twice while
fixing nothing. Moving the encode in-process is what makes it ledgerable —
that is the VRAM half of the ONE-RUNTIME argument, stated as a number instead
of a principle.

The value itself is **UNMEASURED** and labelled as such at the constant.
`TICKET_484` §3 measures it and replaces it. It is not a safety margin.

---

## 5. Stage-level replication

`FEATURE_CATALOG.md` §13 recorded it as *"Stage-level replication (splitting
one stage's frames across cards) is not built"*, and §17.7.5 named it as the
obvious next lever for exactly the stage that binds after the fusion: encode
runs twice per source frame and can be split.

`price_placement` now accepts a tuple of cards for one stage.

### The split is a water-fill, not a halving

Cards differ on the resource the stage is bound by, so an equal split hands
the slow card as many frames as the fast one and lets it set the period. The
shares that minimise the period bring every participating card to the **same
finishing time**, except cards whose fixed load already exceeds it — those
take nothing rather than a negative share. With `fixed_i` the card's load
before this stage and `stage_i` its per-frame cost, the period solves

```
sum_i max(0, (P - fixed_i) / stage_i) = 1
```

`g(P)` is piecewise linear and increasing with breakpoints at the `fixed_i`,
so `split_shares` walks the cards in fixed-load order and takes the first
segment whose closed-form root lies inside it. No search, no tolerance.

This is the per-family × per-phase law applied to one stage: the split is cut
by the resource that determines *this* stage's phase, read per card, and the
result is reported per card rather than as one number.

### Two named limits

1.  **One replicated stage at a time.** With one, the split has the closed
    form above. With two, the shares interact through the cards they share
    and the minimum-period problem becomes a linear program; a water-fill
    applied twice returns *an* answer, not the optimum, and nothing in the
    output would distinguish them. Refused by name with that reason.
2.  **A co-resident stage cannot be replicated.** SR and the tail resize must
    share a card; "share a card" and "spread over cards" are not both
    satisfiable.

The x4 taboo survives replication unchanged: the ceiling is a **per-move**
limit judged on the frame that crosses, and replication makes crossings
*fewer*, not *smaller*.

### What it is worth, on ticket V's fused table

```
                                    single-card encode   encode split 2x3080
5090     decode + sr_fused + rife         41.037 ms            41.037 ms  <- binds
3080_x8  encode                           47.080 ms  <- binds  23.540 ms
3080_x4  (idle) / encode                   0.000 ms            23.540 ms

period                                    47.080 ms            41.037 ms
source-fps                                   21.24                24.37
```

**21.24 → 24.37 src-fps, +14.7 %**, and the bind moves off `encode` onto the
5090's own chain — pinned by `StageReplicationTest.
test_the_encode_split_is_what_ticket_v_predicted`. Both figures inherit
ticket V's ESTIMATE provenance and the `tail_ms = 0.0` optimistic end of the
fused band; this is arithmetic over a measured table, not a new measurement.

Note what this does *not* do. It does not reach 25, and it is the **wrong**
lever to reach for first: §17.7.5's own arithmetic says setting encode to ~0
gives 31.25 src-fps at RIFE `scale=0.5`. The in-process lane removes the host
round trip that costs the 47.08 ms in the first place; replication only
divides it. They compose — a split, zero-copy encode is two cards each doing
half of a much cheaper stage — but the honest ordering is that §1 is the big
lever and §5 is what is left after it.

### What replication costs outside this arithmetic

An encode split across two cards produces **two elementary streams** that the
executor has to interleave back into output order, GOP by GOP. The pricer
does not model that, so `best_placement` only replicates for stages the
caller names in `replicable=` — a pricer must not quietly recommend a shape
the executor cannot run. Wiring the multicard executor for a split encode is
not part of this change.

---

## 6. State

Window: card 0 (RTX 3080), 2026-08-03 09:28-09:55Z, `/spinning/gpu-arb/`
holder + heartbeat, heartbeat stopped before release.

| | |
|---|---|
| root cause (error 8) | **found** at instruction level, on the installed binary |
| second defect (stream ordering) | **found on the card**, 8.59 -> 15.65 dB |
| real encode | **EXECUTED** — 60 frames, 720p, card 0 |
| parity gate | **EXECUTED and PASSED**, can-fail arm rejected |
| session bytes | **MEASURED**, and it is a function of geometry, not a constant |
| ms/frame | **MEASURED**, 1.72x against a same-session A-vs-A floor |
| default lane | **ffmpeg**, still — flipping it is a separate decision |
| stage replication | **built and priced**, executor wiring not built |

### The gate, executed

```
       ffmpeg: 60/60 frames, PSNR 15.65 dB (min 15.13), SSIM 0.70737, 7.979 ms/frame
    inprocess: 60/60 frames, PSNR 15.65 dB (min 15.13), SSIM 0.70738, 4.129 ms/frame
 wrong-chroma: 60/60 frames, PSNR 11.22 dB (min 10.93), SSIM 0.53923, 4.263 ms/frame
PASS: 15.65 dB against 15.65 dB, 60 frames, 1.93x the subprocess lane
can-fail arm REJECTED as required: 11.22 dB
```

Reconstruction is equal to two decimals and SSIM to four; the can-fail arm is
rejected by 4.4 dB, so the threshold is shown to be a threshold. Raw JSON:
`/tmp/nvenc_parity_484.json`.

A single-frame plane-level check at 200 Mbit/s puts the two lanes at the same
place pixel for pixel — luma mean|Δ| 0.12 (max 2), chroma mean|Δ| 0.13 (max 2),
**identical for both arms** — which is the stronger statement: the lanes do
not merely score alike, they produce the same picture.

### ms/frame, against a floor

Three independent runs per lane, same session, same clip:

```
ffmpeg     7.743  8.310  9.076   mean 8.376   range 1.333 ms  (16 % of the min)
inprocess  4.910  4.871  4.861   mean 4.881   range 0.049 ms  ( 1 % of the min)
```

**1.72x**, and the 3.50 ms gap is 2.6x the entire spread of the noisier arm,
so it is not a floor artefact. The subprocess lane is also the *unstable* one,
which is what a host round trip through a pipe looks like.

What this does NOT yet say: the chain's encode column is 47.08 ms per SOURCE
frame at 2160p with two output frames per source (ticket V), not 8.4 ms at
720p with one. The ratio is the transferable part; re-deriving the chain
verdict needs the same measurement at the chain's own geometry, which is
TICKET_484 §4 and did not fit in this window.

### Session bytes, measured — and not a constant

NVML device-wide free delta around session creation plus one encode, input
surface subtracted:

| geometry | session |
|---|---|
| 1280x720 | 51.8 MiB |
| 3840x2160 | 263.8 MiB |

So the ticket's own conditional applies: it moves with geometry and the field
had to become a function. `frame_math.nvenc_session_bytes` is an affine fit,
~25.3 MiB fixed plus ~30 B/px — about twenty NV12 frames' worth, which is what
reconstructed-picture and reference buffers scaling with the frame looks like
next to a context that does not. **Two points, two unknowns**: the fit
reproduces them rather than being validated by them, and a third geometry
would be its first real test. A single 192 MiB constant, which is what this
change originally carried, would have been wrong by 3.7x at 720p.
