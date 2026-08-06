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

## 11. The wedge's actual blocking site, caught with all three ranks on it

Second wedge this window, ~10.5 min into the post-fix soak (last completion
17:46:26). All THREE ranks were stopped at the same line
(`/spinning/616c-hunter5/wedge2_dump_*.txt`):

```
build_dcp_weighted_kv_indices (layers/dcp/owner.py:529)
call_begin_forward            (flashinfer_backend.py:6926)
update_single_wrapper         (flashinfer_backend.py:6695)
init_forward_metadata         (hybrid_linear_attn_backend.py:933)
_execute_extend               (runner/eager_runner.py:294)
```

`owner.py:529` is:

```python
total = int(full_indptr[bs].item())
```

a **blocking device-to-host `.item()`**, and this is the all-three-ranks-on-one-
line signature Hunter-4 described — but at a site nobody had named. Three
distinct wedge sites are now on record across the two windows:

| window | site | kind |
|---|---|---|
| hunter-4 | `flashinfer_backend.py:7429` `.cpu()` | blocking D2H |
| hunter-5 wedge 1 | `memory_pool.py:1525` scatter | kernel ENQUEUE (queue backpressure) |
| hunter-5 wedge 2 | `dcp/owner.py:529` `.item()` | blocking D2H |

That spread is itself the finding: the host gets pinned wherever it next
touches a device queue that is not draining. Fixing any ONE of these sites
moves the wedge to the next one; it does not cure it. The 7429 fix named in
the brief is therefore necessary-at-best, not sufficient, and this window
should not be read as having fixed the wedge family.

`owner.py:529` is nevertheless the best of the three to remove, because
`total` is pure host-derivable arithmetic -- it is just the sum of
`paged_kernel_lens`, which the caller usually already has on CPU. A safe
change is an optional `total_tokens: int` parameter that callers supply from a
CPU mirror, falling back to the existing `.item()` when they have none. Note
the caller at 6926 passes `prefix_lens` as `paged_kernel_lens`, and whether a
`prefix_lens` CPU mirror exists at that point is NOT established -- that is
the one thing to check before writing the change. Not implemented here.

### Instrument lesson, recorded because it bit this window

The soak probe grepped only for `index out of bounds` and
`Bar1CollectiveAborted`, and so reported "faults=0" while the server had
already been wedged for a minute. The wedge's FIRST observable is neither: it
is `Health check failed ... detokenizer` in the log and a health endpoint that
stops answering, with the BAR1 abort arriving only after the ~300e9-cycle
deadline. Any future soak monitor for this bug must watch health/liveness, not
just the two crash strings, or a wedge reads as a clean soak.

## 12. State of the owner.py:529 fix — channel added, NOT yet wired

`build_dcp_weighted_kv_indices` now accepts `total_tokens: Optional[int]`. When
supplied, the blocking `full_indptr[bs].item()` is skipped entirely; when None
the old read is kept, so this commit changes NO behaviour anywhere yet. It is a
channel, not a fix.

**The remaining step, stated precisely so it is not mistaken for done:** the
call site that actually wedged
(`flashinfer_backend.py:6926`, the `uneven_dcp_weighted` branch) passes
`prefix_lens` as `paged_kernel_lens`, and `call_begin_forward` currently
receives `seq_lens_cpu` but NOT a `prefix_lens` host mirror. Wiring therefore
needs `sum(forward_batch.extend_prefix_lens_cpu)` threaded down through
`init_forward_metadata` -> `update_single_wrapper` -> `call_begin_forward`.
`extend_prefix_lens_cpu` exists (`forward_batch_info.py:473`) and is already
passed into this backend elsewhere (`flashinfer_backend.py:1720`, `5732`), so
the material is there; the threading is what is missing.

Also found while reading, same hazard, NOT addressed: the sibling `else` branch
a few lines below does `int(dcp_lens.sum().item())` — another blocking D2H on
the same path, reached when `uneven_dcp_weighted` is off.

**Do not read this window as having fixed the wedge.** Section 11's three
distinct wedge sites say the host pins wherever it next touches a stalled
queue, so even a wired 529 fix is expected to relocate the wedge rather than
cure it, until the reason a device queue stops draining is itself understood.

## 13. The wedge is NOT a collective desync — falsified two independent ways

Two instruments, both already armed during the wedged soak, agree that the
ranks were in step:

1. **#583 collective census.** No cross-rank divergence was ever reported, and
   the local dumps at abort time are byte-identical between the ranks that
   emitted them (`dcp.all_gather 4480x`, `dcp.all_reduce 320x`,
   `tp.all_gather 504x`, `tp.all_reduce 31809x`, `tp.broadcast 42231x`).
2. **barlink per-rank launch sampler** (`/spinning/wedge-catch-603b/launch_rank*.log`,
   1 Hz, host-side only by design so it cannot sync behind a wedged spin).
   At 17:47:59 all three ranks carry the SAME last-op records, per group:
   `all_gather 192512`, `broadcast 32`, `broadcast 24`, `all_gather 192512`.

So the ranks issued the same collectives, of the same sizes, in the same
order. **The wedge is not a pairing shift and not a count divergence.** Any
future work should stop looking for one; that hypothesis is spent.

### The hypothesis this leaves

If every rank has enqueued the same collective and all three are nonetheless
blocked in the next host sync, the queue is not waiting on a missing peer — it
is busy. `Bar1CollectiveAborted` is then a CONSEQUENCE: a peer's spin kernel
hits its 300e9-cycle deadline because the GPU it waits on is still executing,
not because the matching launch never came.

That points at the device work immediately preceding the sync in
`build_dcp_weighted_kv_indices`: `create_flashinfer_kv_indices_triton`, whose
per-request loop is `cdiv(seq_len, 128)` iterations, and the
`torch.empty(max(total, 1))` sitting right before it. A pathological
`paged_kernel_lens` / `total` would make either of them enormous, and would do
so on ALL ranks at once because the value comes from replicated scheduler
state. That matches the observed "all three ranks on the same line".

This is a hypothesis, not a finding. The measurement that would settle it is
the VALUE of `total` and `max(paged_kernel_lens)` at wedge time. Note that
reading them costs a sync, so the honest instrument is the same one that
worked for the index assert: let the run fault/abort and recover the values
post-mortem, rather than adding a hot-path probe that perturbs the window
(NOTE_616b section 3).

### Why the per-site D2H removal was dropped

`owner.py` contains a SECOND, unavoidable host sync a few lines below 529:
`kv_indices = compact[owned].contiguous()` (line 566) is boolean-mask
indexing, which calls `nonzero()` and must sync to learn the output size.
Removing only the `.item()` at 529 therefore cannot make the function
sync-free, and on the above hypothesis the sync is not the cause anyway --
it is merely where the wait becomes visible.

A `total_tokens` channel is kept on `build_dcp_weighted_kv_indices` (inert by
default, no call site passes it) because it is harmless and documents the
option. The plumbing that would have fed it was written and then **dropped on
purpose**: it derived the total from `extend_prefix_lens_cpu`, which is not
guaranteed on every model/path, so it would have been a model-dependent fix in
what is a model-independent communication and sharding layer -- and a wrong
total silently mis-sizes an attention index buffer. Not a trade worth making
for a sync that is not the root.

## 14. The blocking-D2H theory of the wedge is unsound for THIS transport

The brief for this window (inherited from `NOTE_616b` section 4) reasoned:
all three ranks host-block in a blocking D2H, therefore none can enqueue the
work that would release the peers' BAR1 spin, therefore deadlock. That
reasoning silently assumes the CPU is in the delivery path. Under the BAR1
transport it is not.

From `distributed/device_communicators/barlink_bar1.py`:

> The source card writes via DMA directly into the destination card's BAR1
> aperture. No host memory, no NIC, no NCCL.

and the whole setup (VMM allocation, dma-buf fd passed by `SCM_RIGHTS`,
`dma_buf_attach`, mapping `resource1_wc` + `cudaHostRegister`) runs **once at
startup** -- "nothing is mapped and nothing is registered on the hot path".
The spin side is device-driven too (`barlink_device.py`): "a kernel spin-waits
on the peers' flags with volatile loads -- no CPU involvement, no stream
synchronization."

**Consequence.** A flag is written by an already-enqueued device kernel,
straight into the peer's BAR1 over PCIe. A host thread parked in `.item()` or
`.cpu()` cannot delay that write. So a blocking D2H, on its own, CANNOT
produce this deadlock, and removing the D2H at `flashinfer_backend.py:7429` or
`dcp/owner.py:529` was never going to cure the wedge. That is an independent
confirmation of the decision in section 13 to drop the plumbing.

What a `Bar1CollectiveAborted` therefore means here is narrower and more
useful: the peer's flag did not arrive within the ~3e11-cycle cap, and since
delivery needs no CPU, the peer's flag-writing kernel did not RUN. Only two
things do that:

* the peer's GPU was busy ahead of it (a long-running kernel), or
* the peer had not ENQUEUED it yet, because its host was still upstream of
  the launch.

Both remain open, and the two wedges observed this window look like different
members of that pair: wedge 1 had the ranks at DIFFERENT stages (rank 0 inside
the forward, ranks 1 and 2 still in `prepare_for_extend`), which is the
not-yet-enqueued case; wedge 2 had all three at the same line with identical
barlink records, which is the busy-GPU case. Distinguishing them per-incident
is the next concrete step, and the launch sampler already records enough to do
it if its output is diffed per rank at the moment of abort rather than
afterwards.

## 15. Wedge 3: the GPUs are SPINNING, not busy — my own hypothesis falsified

Soak 2 (post-fix build) ran 18:02:43 -> 18:17:46 = ~15 min, 277 completions,
ZERO index asserts, then wedged. Third wedge of the window, and the first one
measured on the GPUs themselves while it was happening.

`nvidia-smi` DURING the wedge (18:18:52, ~66 s after the last completion):

| GPU | util.gpu | util.memory | power | rated |
|---|---|---|---|---|
| 0 (5090) | 100 % | **0 %** | 176 W | 400 W |
| 1 (3080) | 100 % | **0 %** | 130 W | 200 W |
| 2 (3080) | 100 % | **0 %** | 162 W | 200 W |

100 % SM occupancy with **0 % memory utilisation and power far below limit** is
a spin-wait kernel burning SMs and moving no data. A long-running compute or
index kernel would show the opposite (high memory utilisation, power at cap).

**That falsifies section 13's hypothesis** that the queue is busy with
pathologically long device work. It is not busy; it is waiting. After the
~3e11-cycle cap expired the cards fell to 0 % / ~20 W and
`Bar1CollectiveAborted` fired.

State at the wedge, all three ranks identical:

* Python stack: `build_dcp_weighted_kv_indices` (`dcp/owner.py:548`, which is
  the same `total = int(full_indptr[bs].item())` as before -- the line number
  moved only because a comment was added above it) <- `call_begin_forward`
  (`flashinfer_backend.py:6926`) <- `update_single_wrapper` <-
  `init_forward_metadata` <- `_execute_extend`.
* barlink last ops, per rank: `broadcast:32`, `broadcast:24`,
  `all_gather:192512` -- byte-identical across ranks.

So all three ranks are host-blocked at the same sync, all three GPUs are
spinning, and all three report the same last collective. Every "the ranks
disagree" explanation is now falsified from three directions (census counts,
barlink last-ops, and this).

### Where that leaves it

The remaining explanations are ones this window has NOT been able to
distinguish, and they are all below the Python level:

* the spinning collective and the peers' completed one carry different
  SEQUENCE numbers (a flag is set for a generation the waiter has passed or
  not yet reached), or
* a BAR1 flag write is not visible to the waiting peer (write-combining /
  ordering across the PCIe aperture), rather than never issued.

Distinguishing these needs the DEVICE flag words, which the launch dump
deliberately does not read (reading them would sync behind the wedged spin).
The honest instrument is a post-mortem one: a GPU coredump captured at abort
time would carry the flag region, exactly as it carried the index values in
section 6.

### Scope, stated plainly

This wedge is PRE-EXISTING and not introduced by this branch: hunter-4 saw the
same family, and wedge 1 of this window occurred on the un-fixed build. It is
also a DIFFERENT failure from the #616 index assert this window was sent to
fix. The accept-broadcast fix is validated against its own family (15 min and
277 completions with zero index asserts, against a pre-fix envelope of 192 s);
the wedge is what now truncates every soak, and it is unfixed.

## 16. Load policy, and what it costs the evidence

From this point the rig's test load comes ONLY from real local-model
sub-subagents doing meaningful work (max 2 concurrent), not from synthetic
curl loops. Rationale: a soak that burns the GPU on throwaway prompts produces
no artefact, and the same hours can produce committed code.

**The cost, stated so nobody mistakes a quiet run for a validated fix.** The
`trigger_8way.sh` reproducer drives 8 concurrent long-output completions and
saturates the server (`#running-req: 4`, `#queue-req: 5-6`). Two local-model
workers drive `#running-req: 1, #queue-req: 0`. Both crash families in this
window were only ever observed under SATURATION, and the pre-fix envelope
(192 s to the index assert) was measured there. So:

* the accept-broadcast fix's evidence -- 15 min / 277 completions and
  23.5 min / 273 completions with zero index asserts -- comes from the
  saturated reproducer, and stands;
* any FUTURE clean period under agent-only load is NOT comparable evidence
  and must not be quoted as a soak result.

If a regression soak is needed later, the synthetic load may be used again,
but only: freshly launched, its pgid recorded at launch, killed by that pgid,
and its absence proven (`#running-req` back to the agent-only baseline) before
any soak result is reported. The teardown of soak 4 is the worked example --
before: `#running-req: 3, #queue-req: 5`; after: `#running-req: 1,
#queue-req: 0`, server untouched at 9 processes and health 200.

Trap worth recording, hit twice: `pkill -f <pattern>` and any
`/proc/PID/cmdline` scan match the AGENT'S OWN wrapper shell, because the
pattern appears in the command being run. Both times it killed the calling
shell mid-command. Kill by pgid taken from a listed pid instead.
