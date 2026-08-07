# 616h — the 02:02 crash is #616c, and the unbounded `.item()` is why it was fatal

Crash: 2026-08-07 02:02:41, seven minutes after the first boot of the post-#616-B
tree (`27d2c4f27b`, deployed ~01:55), under light read-only agent load.
Archive: `/spinning/CRASH_20260807_0202_watchdog.log`.

## 1. Verdict: #616c recurrence. Not #616-B, and not a desync.

The wedge began at **02:00:06** (last decode batch; `Health check failed …
detokenizer` from 02:00:26 onward, `last_heartbeat 02:00:06`). The wedge-catcher
fired `WEDGE 20260807_020015` and dumped all three ranks nine seconds in. All
three, in **both** dump rounds, are on one line:

```
build_dcp_weighted_kv_indices (dcp/owner.py:548)      <- total = int(full_indptr[bs].item())
call_begin_forward            (flashinfer_backend.py:6926)
update_single_wrapper         (flashinfer_backend.py:6695)
init_forward_metadata         (hybrid_linear_attn_backend.py:933)
_execute_extend               (runner/eager_runner.py:294)
```

That is section 11's signature verbatim — same function, same caller, same
all-three-ranks-on-one-line shape — at the same site, renumbered 529 -> 548 as
the file grew.

The desync hypothesis is falsified again, independently:

* the collective **census** at abort time is byte-identical between the two
  ranks that emitted one: `dcp.all_gather 5328x, dcp.all_reduce 1376x,
  tp.all_gather 346x, tp.all_reduce 21381x, tp.broadcast 2758x` (rank 0 at
  02:01:49, rank 1 at 02:02:41);
* the **collective history** dumps (4096 entries each) are byte-identical
  between ranks 0 and 1 — every family, every position;
* all three ranks carry the same `bs: 1` and `paged_kernel_lens_sum: 41214`,
  with per-rank `cp_lo/cp_hi/cp_ratio` differing exactly as the weighted split
  requires (`0..30 r30` / `30..47 r17` / `47..64 r17`).

`Bar1CollectiveAborted` is the consequence, not the cause. Back-solving the
`300e9`-cycle deadline from the two abort times gives a kernel start of
**02:00:06 on both ranks** (rank 0 aborts after 103 s => ~2.9 GHz, the 5090;
rank 1 after 155 s => ~1.9 GHz, a 3080) — i.e. the spin kernels started when the
wedge started and simply ran out their deadline.

**The deployed #616-B floor is exonerated for this signature.** Its diff
(`6b41d048a7..27d2c4f27b`) touches `scheduler.py`, `mem_cache/*` and tests only —
no attention-metadata, DCP or collective path — and it adds no collective ("rides
the SAME reduce … NO new"). The identical wedge is on record at `owner.py:529`
from 2026-08-06 16:40 and 17:46, before that fix existed. The 7-minutes-after-
deploy correlation is real but the site is not reachable from the floor's code.

## 2. Section 13's leading hypothesis is dead

Section 13 proposed a pathological `total` / `max(paged_kernel_lens)` making
`torch.empty` or `create_flashinfer_kv_indices_triton` enormous on all ranks at
once, and named the settling measurement: the VALUE of `total` at wedge time.
The catcher dumped locals, so that measurement exists:

```
bs: 1        paged_kernel_lens_sum: 41214        total_tokens: None
```

`bs=1` and `total = sum(prefix_lens) = 41214 - 2048 = 39166`. `torch.empty(39166)`
and a one-program Triton launch of `cdiv(39166,128) = 306` iterations are
trivial. Nothing is enormous. The device work at this site is not the problem —
**the host sync is**, and it is a blocking wait on a queue whose head is a spin
kernel waiting for peers whose hosts are parked in the identical sync.

## 3. Why THIS site is the fatal one, when 82% of wedges are elsewhere

A census of all 239 wedge events the catcher recorded over two days:

| all-three-rank innermost frame | events |
|---|---|
| `_wait_ctl_event` (`barlink_bar1.py:4649`) | 195 (82%) |
| `build_dcp_weighted_kv_indices` (`dcp/owner.py:529`/`:548`) | 5 |
| `synchronize`, `_eagle_prefill_tail_tokens`, `donate_mamba_ping_pong_slot`, `alloc`, … | the rest |

The overwhelming majority of wedges park in `_wait_ctl_event` **and recover**.
That site is a BOUNDED poll (`_ctl_event.query()` + `time.sleep(0.0005)`, breaks
at `sync_deadline_s`, returns False and logs) — the host stays alive, keeps the
GIL cycling, and can still enqueue.

`owner.py:548` is an **UNBOUNDED blocking CUDA sync**. A host parked there cannot
poll, cannot time out, and cannot enqueue the work that would release its peers.
That asymmetry — not the frequency — is why the fatal crash landed on the rare
site rather than the common one.

This corrects section 11's pessimism ("fixing any ONE of these sites moves the
wedge to the next one; it does not cure it"). Moving a pin from an unbounded
sync to barlink's bounded poll is not a null move: it is the difference between
the 5 fatal-capable events and the 195 recoverable ones.

## 4. The fix: section 12's open item, closed

Section 12 left the channel added but unwired, and named the one thing to check
first — whether a `prefix_lens` host mirror exists at `6926`. It does, and the
precise reason it was not reaching the call site:

* `init_forward_metadata` passes `extend_prefix_lens_cpu` into
  `update_single_wrapper` (line 1720), and `prefix_lens` there IS
  `forward_batch.extend_prefix_lens` (line 1681) — so the mirror is exact, not
  an estimate;
* `update_single_wrapper` consumed it **only on the `use_ragged=True` branch**
  and never forwarded it;
* a multimodal model forces `use_ragged=False` (line 1684) while the DCP split
  still runs (the comment at 6949 says so explicitly), so on this rig neither
  path supplied it and `call_begin_forward` always saw `None`;
* `paged_kernel_lens_sum` (41214) was right there but is the sum of `seq_lens`,
  not of `prefix_lens` — using it would have been silently wrong.

Change: forward `extend_prefix_lens_cpu` from `update_single_wrapper` to
`call_begin_forward`, and derive `total_tokens` from it at the weighted-DCP
branch via `_dcp_host_total_tokens`. Every other caller keeps `None` and the old
read, so no other path changes.

Still NOT addressed, same hazard, different config: the sibling `else` branch's
`int(dcp_lens.sum().item())` (reached only when `uneven_dcp_weighted` is off,
which this rig is not).

## 5. Validation

`test/srt/distributed/test_dcp_index_host_sum_616c.py`, hermetic (no CUDA, no
process group). Can-fail proven by running the two seam assertions against the
deployed tree that crashed (`/spinning/wt-530-serving`, `27d2c4f27b`) and against
this branch, same script, same interpreter:

```
BASE   SEAM2 call_begin_forward -> total_tokens = None        FAIL
BASE   SEAM1 update_single_wrapper -> extend_...cpu = None    FAIL
FIXED  SEAM2 call_begin_forward -> total_tokens = 39166       PASS
FIXED  SEAM1 update_single_wrapper -> extend_...cpu = [39166] PASS
```

`39166` is exactly the wedge batch's `41214 - 2048`.

Owner-rule regression, same command on both trees:
`test_dcp_owner_rule_616c` + `test_dcp_weighted_index_math` +
`test_dcp_weighted_owner_rule` + `test_draft_extend_dcp_split` +
`test_triton_weighted_dcp_wiring` => **58 passed / 117 subtests on both**.

ruff on `flashinfer_backend.py`: 30 findings on both trees (all pre-existing).

### Instrument gap found while working

Rank 2 emitted **no** census line at the crash: it never entered the abort
handler, only the bounded-poll warning path. The census is therefore unavailable
on exactly the rank that is most often the late one. If the census is to settle
desync questions by itself, the bounded-poll expiry path needs to dump it too.
