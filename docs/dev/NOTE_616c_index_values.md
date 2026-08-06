# NOTE 616c — the index-OOB crash: the VALUES, and what they name

Window: 2026-08-06, agent hunter-5, worktree `/spinning/wt-616c-hunter5`,
branch `fix/accept-index-616c`, base `462012111e`.
Artifacts: `/spinning/616c-hunter5/`.

This continues `NOTE_616b_index_race.md`. That note named the faulting kernel
and left one measurement outstanding: the actual index VALUES and the violated
bound. Section 1 below closes that, **without needing a new reproduction** —
the numbers were recoverable from hunter-4's existing coredump.

## 1. The bound and the values, read out of the EXISTING arm-E coredump

`NOTE_616b` recorded that the arm-E dump was taken with the default
lightweight flags (which include `skip_global_memory`) and concluded the values
were therefore not in it. That is true of the tensor CONTENTS, but not of the
fault itself: **the offending index had already been loaded into a register**
when the bounds check failed, and registers are always present in a coredump.
Kernel PARAMETERS are likewise readable, via cuda-gdb's `@parameter` address
space, even though the flag set nominally names `skip_constbank_memory`.

Method (`/spinning/616c-hunter5/`, cuda-gdb 12.9 against the arm-E dump):

1. Disassemble around the trap PC. The SASS is ATen's bounds check, unrolled:

   ```
   LDG.E.64  R4, [R4.64]                 # index = *(int64*)(index_ptrs[i] + off)
   LDC.64    R2, c[0x0][R29+0x4a0]       # sizes[i]
   IADD3     R7, P0, RZ, -R2, RZ         # -sizes[i]
   ISETP.GE.U32.AND P0, PT, R4, R2, PT   # index >= sizes[i] ?
   ISETP.GE.U32.AND P1, PT, R4, R7, PT   # index >= -sizes[i] ?
   IMAD.MOV.U32 R28, RZ, RZ, R4          # <-- index kept in R28:R27
   IMAD.MOV.U32 R27, RZ, RZ, R5
   @!P0 BRA  P1, <ok>
   MOV       R8, 0x6f                    # 111 == IndexKernel.cu:111
   CALL.ABS.NOINC R2                     # __assertfail
   ```

   So `R28:R27` holds the failing index at the trap, and it is not clobbered
   by the assert-call setup.

2. Read `R28:R27` per trapped lane, and the kernel params.

**Result — the bound:**

| param | value |
|---|---|
| `num_indices` | **1** (a single advanced index, not a multi-index gather) |
| `sizes[0]` | **16** — the violated bound |
| `strides[0]` | 4 bytes — the gathered dim is contiguous, 4-byte elements |
| index dtype | int64, contiguous (byte offset `8 * lane`) |

**Result — the values** (`R27` was 0 for every lane, so all are small positive):

| lane | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| index | 63 | 466 | 1308 | *(in range)* | 42962 | 25 | 561 | 1558 |

| lane | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|
| index | 4918 | *(in range)* | *(in range)* | *(in range)* | 2570 | 11033 | *(in range)* | *(in range)* |

Only the ten out-of-range lanes survive in the dump; the six in-range lanes
passed the check, completed the gather and exited, which is why
`info cuda threads` lists exactly ten.

## 2. This resolves the tension `NOTE_616b` recorded

`NOTE_616b` §1 stated the unresolved point plainly: if the trapped lanes are
`accept_index` gathering from `predict`, row 0's accepted entries should hold
the small chain offsets 0,1,2 — yet they trapped. The dump answers it:
**option 3 of the three the note listed is the true one.** Row 0 holds
63, 466, 1308. `accept_index`'s accepted entries do not carry chain offsets at
the moment of the crash.

The tensors themselves are as `NOTE_616b` guessed. The gather is

```
python/sglang/srt/speculative/eagle_worker_v2.py:2748
    accept_tokens = predict[accept_index]
```

and the boot flagset pins the arithmetic exactly:
`--max-running-requests 4`, `--speculative-num-draft-tokens 4`, so
`predict.numel() == bs * draft_token_num == 4 * 4 == 16` — which is
`sizes[0]` on the nose, and `accept_index` is `(bs, max_tree_depth) == (4, 4)`.

## 3. What the lane pattern now means, quantitatively

`predict` is **zero-initialised** and then written only at accepted positions:

```
python/sglang/srt/speculative/eagle_utils.py:1040
    predict = torch.zeros(predict_shape, dtype=torch.int32, device=device).flatten()
```

Hunter-4 derived per-request accept lengths 3, 4, 1, 2 from the lane geometry.
Those sum to **10** — which is exactly the number of trapped lanes, and exactly
the number of entries the verify kernel writes into `predict`.

So the observed content of `accept_index` at the gather is:
* ten large, scattered, positive values at exactly the accepted positions, and
* six in-range values at exactly the non-accepted positions.

That is the shape of `predict`'s own contents (written entries + zero fill),
not of a valid `accept_index` (chain offsets + `-1` fill). Note this also
revises one detail of `NOTE_616b`: the six in-range lanes are consistent with
**zeros**, not necessarily with the `-1` padding the earlier reading assumed —
both are in range for a bound of 16, so the lane pattern alone cannot tell them
apart, and the earlier note's inference from "`-1` padding intact" is weaker
than it appeared.

**Status of the conclusion: strongly evidenced, not yet proven.** The
consistent reading is that `accept_index`'s buffer is carrying `predict`-like
content. Proving it requires reading both buffers from one dump, which needs a
coredump taken WITH global memory retained — that is what
`/spinning/616c-hunter5/boot_616c.sh` and `analyze_core.gdb` are for. Until
those two arrays are compared element-wise, "it looks like predict" is an
inference from structure and magnitude, not an identification.

## 4. Two circumstantial facts that point at the rank-0 accept broadcast

Recorded as leads, explicitly NOT as conclusions.

1. **Both index-assert reproductions died on a non-root rank.** The arm-E
   coredump's pid (1620168) is TP **rank 1** per the boot log's barlink
   liveness line; arm C's assert is likewise tagged TP1. `spec_accept_broadcast_src()`
   (`eagle_utils.py:880`) returns 0 for this configuration (the weightless-KV
   fast lane is off), so rank 0 is the broadcast SOURCE and its own buffers are
   never written by the broadcast. A fault that only lands on receivers is
   consistent with delivery; n = 2, and both were rank 1 specifically rather
   than "any non-root", so this does not yet separate "non-root" from "rank 1".

2. **`predict` and `accept_index` are broadcast back-to-back, same size, same
   dtype.** `eagle_utils.py:1189` sends the tuple
   `(predict, accept_index, num_correct_drafts)` through
   `capture_safe_tp_broadcast`, which issues one pynccl broadcast per tensor in
   order. `predict` and `accept_index` are both int32 with 16 elements — 64
   bytes each. Any pairing shift of exactly one collective between ranks would
   therefore deliver `predict`'s bytes into `accept_index`'s buffer **with no
   size mismatch for NCCL to reject**.

**Counter-evidence against the simplest version of that story, recorded
because it is inconvenient:** the #583 collective census was armed in every one
of hunter-4's arms (`SGLANG_COLLECTIVE_CENSUS_INTERVAL=1`), and it **never
reported a cross-rank divergence** in arm C or arm E. Its per-family counts are
cumulative, so a one-collective shift should persist and be caught at the next
once-per-iteration comparison. The escape hatch is narrow but real: rank 1 died
inside the crash round and never emitted its own local census dump (only ranks
0 and 2 did, and those two agree exactly — `tp.broadcast 7788x`,
`tp.all_reduce 5672x`, `tp.all_gather 108x`), so the crash-round counts for the
rank that actually faulted are unknown. A count divergence is therefore
**not evidenced**, and a mechanism that needs one has to explain why the census
stayed silent.

## 5. Method note worth keeping

`NOTE_616b` §3 established that any instrument inside the racing window is a
treatment. This window adds the complement: **the coredump had more in it than
the flag set suggested.** `skip_global_memory` removes tensor contents, but
registers and kernel parameters survive, and for a bounds-check assert the
failing value and the violated bound both live there. Before paying for a
reproduction to get a number, check whether the number is already in the dump
you have.

## 6. PROVEN: `accept_index` carries `predict`'s contents

Second reproduction, this window (`/spinning/616c-hunter5/`), rank 2, pid
1673225, coredump taken WITH global memory retained (28.7 GB). This is the
measurement `NOTE_616b` §6 named, and it converts §3's inference into an
identification.

Params: `num_indices=1`, **`sizes[0]=12`** (this batch had bs=3, so
`predict.numel() == 3*4 == 12`), `strides[0]=4`.
Trapped lanes: 0, 4, 5, 6, 8, 9, 10, 11 — eight, matching the eight asserts.

Index tensor (int64, as ATen converted it) at `0x78c729e00800`:

```
21966, 0, 0, 0, 2286, 1098, 1510, 0, 1204, 280, 3173, 6326
```

Gathered source tensor (`in_data`, int32) at `0x78c6db5db200`:

```
21966, 0, 0, 0, 2286, 1098, 1510, 0, 1204, 280, 3173, 6326
```

**Element-wise identical, all twelve.** The tensor being used as the index and
the tensor being gathered from hold the same values.

Two independent details pin which tensor that content belongs to:

1. **The fill value is 0, not -1.** `accept_index` is created with
   `torch.full((bs, max_tree_depth), -1, ...)` (`eagle_utils.py:1041`);
   `predict` with `torch.zeros(...)` (`eagle_utils.py:1040`). The non-accepted
   slots here hold **0**. The buffer carries `predict`'s initialisation, not
   `accept_index`'s.
2. **The magnitudes are token IDs**, and they sit at exactly the accepted
   positions, which is where the verify kernel writes `predicts`.

Reading the rows as `(bs=3) x (width 4)`:

| row | contents | accept_len |
|---|---|---|
| 0 | `21966, 0, 0, 0` | 1 |
| 1 | `2286, 1098, 1510, 0` | 3 |
| 2 | `1204, 280, 3173, 6326` | 4 |

So the crash is `predict[accept_index]` executed when `accept_index` holds
`predict`. The values are out of range because token IDs are unbounded by
`predict.numel()`.

**This falsifies the "corrupted index" framing entirely.** Nothing is
corrupted: a whole, well-formed, correctly-initialised `predict` buffer is
being used where `accept_index` should be. It is a substitution, not damage.

## 7. Rank evidence: receivers only, 3 of 3

| reproduction | pid | TP rank | root? |
|---|---|---|---|
| hunter-4 arm C | — | 1 | no |
| hunter-4 arm E | 1620168 | 1 | no |
| hunter-5 run 1 | 1673225 | **2** | no |

`spec_accept_broadcast_src()` returns 0 for this configuration, so rank 0 is
the broadcast SOURCE — it sends from its buffers and never has them written.
Three faults, three receivers, and now two DIFFERENT receiving ranks, which
removes the "rank 1 specifically" reading left open in §4.

## 8. Mechanism: narrowed to two, not yet decided

`eagle_utils.py:1189` broadcasts the tuple `(predict, accept_index,
num_correct_drafts)` through `capture_safe_tp_broadcast`, which issues **one
pynccl broadcast per tensor, in order**. `predict` and `accept_index` are both
int32 and both `bs*4` elements — byte-identical in size. Two mechanisms both
explain every observation above and are NOT yet separated:

* **M1, pairing shift.** If a rank's broadcast sequence is offset by one
  collective relative to its peers, `predict`'s payload lands in
  `accept_index`'s buffer. Same size means NCCL has nothing to reject, so it is
  silent. Receiver-only by construction.
* **M2, cross-stream block reuse.** The caching allocator hands the block
  freed from one round's `predict` to the next round's `accept_index` while a
  broadcast write to it is still in flight on another stream. Also
  receiver-only (the source rank's broadcast only reads). The tree already
  carries `record_stream_each` / `record_stream_for_v2_verify` helpers in
  `spec_utils.py`, so this bug class is known here.

Evidence bearing on the choice, including the inconvenient part:
the #583 collective census was armed in every hunter-4 arm and in this run, and
**never reported a cross-rank count divergence**. Its counts are cumulative, so
M1 should leave a persistent 1-off that the next once-per-iteration comparison
catches. The escape is that the faulting rank dies inside the crash round and
never emits its own local census line (in arm E only ranks 0 and 2 dumped, and
those two agreed exactly). So M1 is not evidenced and owes an explanation for
the census silence; M2 is not evidenced either. Deciding between them is the
next window's job, and it is now a narrow question rather than an open one.

## 9. Fix direction (designed, NOT implemented or soaked in this window)

The candidate that is robust to the M1/M2 question rather than betting on it:
**fuse the three broadcasts into one packed buffer.** Pack `predict`,
`accept_index` and `num_correct_drafts` into a single contiguous int32 tensor,
broadcast once, unpack. This:

* removes the intra-tuple pairing entirely, so M1 has no shift to make;
* replaces three same-size collectives with one differently-sized one, so any
  residual shift hits a size mismatch and NCCL raises instead of silently
  delivering a wrong-answer payload;
* narrows M2's exposure to one allocation instead of three;
* is strictly fewer collectives on the hot path.

It is a design, not a result. It has not been written, tested, or soaked, and
it should not be treated as a fix until a can-fail falsifier and a soak past
the 3.5-7 min crash envelope exist.

## 10. The wedge family (task B): the briefed target is not what wedged here

Hunter-4 recorded the wedge as all three ranks blocking in
`flashinfer_backend.py:7429` (`self.kv_indptr[:, : bs + 1].cpu()`), a blocking
D2H in the draft path, and the brief for this window was to root-fix that.

This window reproduced a wedge under the same load, and it has a DIFFERENT
shape (`/spinning/616c-hunter5/wedge1_dump_*.txt`):

| rank | frame |
|---|---|
| 0 | `call_begin_forward` (`flashinfer_backend.py:6926`), inside the forward |
| 1 | `alloc` (`memory_pool.py:1525`) via `alloc_req_slots` / `prepare_for_extend` |
| 2 | same as rank 1 |

`memory_pool.py:1525` is `self.req_index_to_mamba_index_mapping[select_index] =
mamba_index_tensor` — a device scatter ENQUEUE, not a D2H. Rank 0 is ahead
inside the forward while ranks 1 and 2 are still preparing the next batch.

So in this instance the host was pinned by **CUDA launch-queue backpressure**
(the device queue not draining), not by a blocking device-to-host copy. That
matters for the fix: replacing the `.cpu()` at 7429 with a staged
non-blocking read would not have cleared this wedge, because no rank was in
that call. A non-blocking copy plus an event wait is also not a deadlock cure
in general — the host still cannot proceed until the queue drains; it only
avoids pulling in more streams than necessary.

The honest position: 7429 is a real blocking D2H on the draft path and is
worth removing on its own merits, but it is **one of several places the host
can get pinned by a stalled queue, not established as the wedge's root**. The
root question is why a device queue stops draining, which points back at the
same collective-pairing question as section 8 rather than at any single
`.cpu()`.

A removal of the 7429 D2H is derivable — the decode path already builds this
indptr host-side from `seq_lens_cpu` (`flashinfer_backend.py:6546-6550`), and
the draft kernel's rule is
`kv_indptr[i][z] = sum(positions[:z]) + z*(i+1)`
(`kernels/ops/speculative/cache_locs.py:189-197`). `ForwardBatch.seq_lens_cpu`
exists, so the host has the raw material. What is NOT established is the exact
relation between the draft `positions` buffer and `seq_lens` (off-by-one, and
the topk>1 / page_size>1 branches). That buffer is device-only under graph
capture, so the relation cannot be confirmed by reading alone. Writing the
host-side formula without confirming it would silently corrupt attention
indices, so the next step is a default-off runtime cross-check that computes
the host candidate and diffs it against the device `kv_indptr` — cheap, and it
either validates the formula or falsifies it in one run. That work is NOT done
here.
