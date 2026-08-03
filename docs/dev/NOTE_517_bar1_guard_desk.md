# #517 — the host-path BAR1 collectives inside the decode loop, and a guard that stops paying for them

Desk half of #517. Tree `/spinning/wt-517-bar1`, branch
`fix/bar1-guard-desk-517`, base `96c7dc5d2b`. **No GPU was touched.** Every
claim below is a file:line in this tree or a number from
`/spinning/gpu-battery-results/2026-08-03_w4_t2_476_bar1_floor/RESULTS.md`,
which is used and not re-derived.

---

## 0 — What the measurement asked, in one line

#476 measured that removing the #431 guard's two seams gives back +2.68 % on
code decode_TPS (reproducing #424's pre-#431 BAR1 advantage of +2.5 %), split
5.26 pp on Seam B (replay boundary) and **6.64 pp on Seam A (host-path
collectives)**. `bench.sh`'s `decode_TPS` is `completion / (wall - TTFT)` and
therefore excludes prefill, so a Seam A cost of 6.64 pp says host-path
(uncaptured) BAR1 collectives run **inside the decode loop**. The window
measured that they exist. This document names them.

---

## 1 — THE NAMED COLLECTIVES

Five BAR1 collectives run on the HOST path per NEXTN decode round on the #476
recipe (TP=3, NEXTN 3/1/4, topk=1, no rejection sampling, barlink active).
They are all `broadcast`, and they are all the **speculative rank-agreement
syncs** — the #50 heterogeneous-GPU determinism machinery.

### 1.1 The plumbing that makes them barlink collectives

`capture_safe_tp_broadcast` (`speculative/spec_utils.py:102-141`) prefers the
pynccl communicator and falls back to `tp_group.broadcast` at `:138`. On a
**barlink boot the pynccl branch is dead**:

* `parallel_state.py:778-781` — `_barlink_active = self.barlink_comm is not
  None`; `self.pynccl_comm = None`; built only `if
  should_build_pynccl(use_pynccl, self.world_size, _barlink_active)`.
* `parallel_state.py:440` — `return use_pynccl and world_size > 1 and not
  barlink_active`.

So `getattr(tp_group, "pynccl_comm", None)` is `None` and every one of these
syncs takes `tp_group.broadcast(t, src=src)`:

```
spec_utils.py:138            tp_group.broadcast(t, src=src)
parallel_state.py:1851-1852  if self.barlink_comm is not None:
                                 return self.barlink_comm.broadcast(input_, src)
barlink.py:1244              self._after_transport(t, "broadcast")   # SEAM A
barlink_bar1.py:3648-3651    self.barlink_all_to_all_single(...)     # BAR1 broadcast
                                                                     # IS an all_to_all
barlink_bar1.py:4022         self._note_launch("all_to_all", int(moved))
```

`_after_transport` → `check_aborted` → `_unchecked_launches == 1 >=
check_every() == 1` → `status()` → `int(self._ctl_dev[0].item())`
(`barlink_bar1.py:4471` pre-#517) = **one 4-byte D2H plus a stream
synchronization per collective**.

An 8-byte broadcast is in range: `_handles_broadcast` covers
`1 .. a2a_slot * bc_max_rounds` with no gaps, and its own docstring notes that
"the standard run sends exactly those 12 bytes".

**This also identifies the §3 crash line.** `Bar1CollectiveAborted ... Last
collective launched: all_to_all (8 bytes, 0 rounds)` on group `tp:0` is a
speculative rank-sync broadcast of an 8-byte tensor (at bs=1, `topk_index` is
one int64 = 8 bytes), because a BAR1 broadcast is issued as an `all_to_all`
and `_note_launch` records it under that name.

### 1.2 The five, with their call path from the decode loop

Decode round, from `EAGLEWorkerV2.forward_batch_generation`
(`speculative/eagle_worker_v2.py:1927`), else-branch at `:2033`:
`draft()` (`:2059`) → `verify()` (`:2062`) → `_draft_extend_for_decode()`
(`:2087`).

| # | site | tensors | call path | per round |
|---|---|---|---|---|
| 1-3 | `speculative/eagle_utils.py:1149` `capture_safe_tp_broadcast(tp_group, (predict, accept_index, num_correct_drafts), src=spec_accept_broadcast_src())` | 3 | `eagle_worker_v2.verify()` `:2565` → target-verify graph replay at `:2628` → `eagle_sample()` `:2680` → `eagle_utils.eagle_sample` → `:1138-1153` | 3 broadcasts |
| 4-5 | `speculative/eagle_worker_v2.py:1583` `_broadcast_draft_picks(ret_topk_index, ret_topk_p, ret_draft_probs)` | 2 (`ret_draft_probs` is `None` without rejection sampling, and `_broadcast_draft_picks` skips `None`, `spec_utils.py:136-138`) | `forward_batch_generation` `:2087` → `_draft_extend_for_decode()` `:1447` → draft-extend graph replay `:1516-1523` → argmax/topk on the SELECTED rows `:1553-1581` → `:1583` | 2 broadcasts |

Site 4-5 is the one the code itself names. `eagle_worker_v2.py:1578-1581`,
verbatim:

```python
# The step-0 pick for the NEXT draft round (runs every decode
# iteration, outside any cuda graph — "selected-row topk is owned by
# the worker"): the per-rank argmax/topk/sample here was the last
# unsynced decision point after the in-loop and verify syncs.
```

Sites 1-3 are host-path for a structural reason: the target-verify CUDA graph
covers the model forward only. `verify()` calls
`self.target_worker.forward_batch_generation(...)` at `:2628` and then does
NaN/Inf probing, grammar bitmask generation and `eagle_sample` at `:2676-2686`
— all host code, after the replay boundary.

### 1.3 What is NOT on the host path (checked, not assumed)

* `_broadcast_draft_picks` at `eagle_worker_v2.py:1251`, inside the multi-step
  draft loop, is **captured**: `eagle_draft_cuda_graph_runner.py:458` records
  `ret = self.eagle_worker.draft_forward(forward_batch)` into the draft-decode
  graph, and `draft()` executes it as ONE
  `self.cuda_graph_runner.execute(forward_batch)` (`:1027-1030`). Its
  broadcasts run on replay with no host code between them — which is exactly
  what arms `_captured_launches` (`barlink_bar1.py:4352-4355`) and is why Seam
  B has to exist.
* `_broadcast_draft_picks` at `:1417` is in `_draft_extend_for_prefill`
  (`:1311`), reached only from the extend branch at `:2020-2027`. Prefill,
  not decode.
* `_broadcast_draft_picks` at `:1726` is in `draft_extend_catchup` (`:1607`),
  the resume/catch-up path, not the steady decode round.
* `speculative/eagle_utils.py:198` and `speculative/cross_algo_worker.py:*` use
  `torch.distributed.*` directly, which never reaches `BarlinkCommunicator`.
* `speculative/adaptive_graph_memory.py:1209` uses `_tp_cpu_group`
  (`all_gather_object`), i.e. gloo, not barlink.
* `eagle_utils.py:975` (weightless-KV worker accept broadcast) is on the #143
  lane, not this recipe.

---

## 2 — THE CORRECTED §1.3 MODEL

`TICKET_476_bar1_decode_floor.md` §1.3 (in `/spinning/wt-476-bar1-floor`, not
in this tree) states the model that the measurement falsifies. Two of its
three points are wrong, and the third is right but incomplete.

| §1.3 said | corrected | evidence |
|---|---|---|
| "Decode is fully captured, in **five graphs per verify round**" (3 draft-decode + 1 target-verify + 1 draft-extend replay) | **Three replay boundaries per decode round.** The draft chain is ONE graph, not one per step. | `eagle_draft_cuda_graph_runner.py:458` captures the whole `draft_forward` loop; `draft()` replays it once (`eagle_worker_v2.py:1027-1030`); `_replay_graph` → `backend.replay` (`eagle_draft_cuda_graph_runner.py:287`) → `full_cuda_graph_backend.py:167` is ONE `graph.replay()` per call. The other two are the target verify and the draft extend. |
| "Prefill is NOT captured ... so prefill runs eager, **on Seam A**", implying Seam A is a prefill-only cost | Seam A also runs **five times per decode round** — the speculative rank-agreement syncs of §1.2. Decode is captured; the syncs BETWEEN the captured pieces are not. | `eagle_worker_v2.py:1578-1583`, `eagle_utils.py:1138-1153`, dispatched through `barlink.py:1244`. |
| "one replay boundary is up to **three** separate `.item()` calls" (three registered transports) | Correct. `barlink_abort_gate.check_aborts` (`:135`) iterates `registered()`, and #476's gate line shows 9 `ACHIEVED=bar1` = 3 groups × 3 ranks. | `barlink_bar1.py:2197` registers once per bring-up. |

**Corrected per-round cost of the pre-#517 guard, on the #476 recipe:**

| seam | events per decode round | blocking device reads |
|---|---|---|
| A — host-path collectives | 5 broadcasts (3 verify-sync + 2 draft-pick) | 5, each a full stream sync |
| B — replay boundaries | 3 boundaries × 3 transports | 9 `.item()`s, of which 3 are real syncs (the first read per boundary; the stream is drained afterwards) |

That ordering — 5 genuine syncs on A against 3 on B — is what makes Seam A
the larger of the two at 6.64 pp vs 5.26 pp, and it is the thing the old model
could not produce. It also **refines RESULTS §4 item 4**: on the DECODE axis
Seam A is not "~150 eager f-strings per forward" (that is the prefill/TTFT
shape) — it is five lost run-aheads. The f-string and the aten round trip are
worth removing, but they cannot explain 6.64 pp of decode.

Caveat, stated rather than smoothed: those counts are the CAPTURED steady
state. Each of the three stages takes its graph only when `can_cuda_graph` is
true (`eagle_worker_v2.py:1027`, `:1516-1523`, and the verify batch's
`decode_cuda_graph_runner.py:2100`); a batch shape outside the captured ladder
runs that stage eagerly, and then its per-layer `all_reduce`s join Seam A as
well. The #476 boot logged all three captures, so the arms this document
prices are the captured ones.

Sanity check against the round: #476's `ms/round` at bs=1 is ~30 ms on code;
5 syncs × 0.4-0.7 ms of lost run-ahead = 2.0-3.5 ms = 6.7-11.7 % — the
measured 6.64 pp sits in that band. (The ticket's own §2.1 fit gives 0.73-1.12
ms per lost sync; the same arithmetic over 8 syncs instead of 5 boundaries
lands on the same total.)

**Consequence for the fix ranking:** RESULTS §4 ranked candidate 1 (async
previous-boundary read) first because Seam B was thought dominant. With Seam A
named, candidate 1 is still first — but only because the same mechanism, a
staged read, is what makes BOTH seams free. A candidate aimed at the replay
boundary alone would have left the larger half untouched, which RESULTS §4
already warned about.

---

## 3 — CANDIDATE TABLE, WITH COSTS

Per NEXTN decode round on the #476 recipe. "sync" = a full stream
synchronization (lost host run-ahead, the expensive unit); "read" = an aten
dispatch that does not synchronize.

| # | candidate | seam | reads/round | syncs/round | where the latency lands | verdict |
|---|---|---|---|---|---|---|
| — | **today (pre-#517)** | A+B | 14 | 8 | none (report is immediate) | 9.22 % of decode |
| 1 | **staged read of the previous check's word, event-gated** | **A+B** | 8 copies + 8 `cudaEventQuery` | **0** in steady state; 1 forced per `MAX_LAG` unresolved checks | report is ≤1 check late, hard-bounded at `MAX_LAG` | **BUILT** |
| 2 | let `CHECK_EVERY` reach the replay boundary | B | 9/K | 3/K | up to K boundaries | **BUILT** (default K=1 ⇒ behaviour-identical; the knob now has reach) |
| 3 | one read for all three transports | A+B | 8 → ~3 | 8 → 3 | none | **absorbed by 1** — with zero syncs there is nothing left to merge, and the three transports keep independent status words |
| 4 | lazy label + skip the aten round trip | A | −8 dispatches | 0 | none | **BUILT** (label); the `_ctl_dev[0]` slice is now a persistent view built once |
| 5 | host-visible status word (mapped pinned page the kernel writes) | A+B | 0 | 0 | none | **NOT BUILT** — needs the ext kernel to take a mapped-host pointer for `ctlStatus` and a GPU to prove the write is visible; keep last, as RESULTS says |

Why 1 rather than 2 as the primary: candidate 2 buys its saving by making the
guard blind for K−1 boundaries, and #476 §3's abort was **intermittent** — an
event you may only get one shot at. Candidate 1 costs the guard at most one
check of latency for a *sticky* bit and keeps every abort. Candidate 2 ships
anyway because its absence was a REACH defect (a documented knob that could
not reach the seam it was documented for), not because it should be turned up.

---

## 4 — WHAT WAS BUILT

Three files, all inside the barlink transport:

**`barlink_abort_gate.py`** — `ENV_DEFER`
(`SGLANG_BARLINK_BAR1_ABORT_DEFER`, default on), `ENV_MAX_LAG`
(`SGLANG_BARLINK_BAR1_ABORT_MAX_LAG`, default 4), `defer_enabled()`,
`max_lag()`, and `should_defer_status(status_is_cuda, defer_on)` — the single
importable definition of the arming decision, the
`parallel_state.should_build_pynccl` pattern. `ENV_EVERY`'s docstring records
its new reach.

**`barlink_bar1.py`**
* `_arm_status_stage()` — called from bring-up right after `_ctl_dev` is
  created. Builds a persistent 1-element view of the status word, a pinned
  host destination and a `torch.cuda.Event`. Degrades to the blocking read
  with a warning if any of that fails.
* `_read_status_for_check()` — issues a non-blocking D2H of the sticky word
  onto the current stream and returns the value staged by an EARLIER check, or
  `None` while it is in flight. After `max_lag()` consecutive unresolved
  checks it forces one `event.synchronize()`.
* `check_aborted()` — uses it; accumulates the unverified window in
  `_deferred_launches` so the raise still names how many collectives are
  implicated; counts replay-boundary entries in `_boundary_checks` so
  `CHECK_EVERY` throttles Seam B too; the message says when the value came
  from the staged copy.
* `close()` drops the staged view/page/event (the view keeps `_ctl_dev` alive
  otherwise).

**`barlink.py`** — `_after_transport` passes the interned op literal instead
of building `f"{op} on group {self.group}"` on every collective. Nothing is
lost: the raised message already opens with `rank r/w group <group>`.

### Why this does not weaken the guard

`ctlStatus` is sticky. The only device writes are `*A.ctlStatus = 1u`
(`barlink_bar1_ext.py:653`, `:714`, `:799`, `:846`, `:1083`) and the only host
write is the `torch.zeros(2, ...)` at bring-up (`barlink_bar1.py:2172`). There
is no clear, so a late read of the bit is the same bit — the deferral trades
reporting LATENCY, never detection.

**Is one check of latency safe?** Verified at the code rather than assumed.
The round's tokens become host-visible at
`managers/scheduler_components/batch_result_processor.py:217`
(`result.copy_done.synchronize()`, and again at `:334`, `:687`, `:699`),
downstream of the async D2H issued on the copy stream
(`managers/scheduler.py:4328-4348`). Under the overlap scheduler that is at
least a full iteration after the graphs of that round ran; there are 8 checks
per decode round, so a one-check-late report is many checks ahead of the
consumption point. In the non-overlap branch (`scheduler.py:4379-4384`) the
copy is in the same iteration, and the report is still ahead of it for
sites 1-4 of a round. The one residual case — an abort inside the LAST
boundary of a round — is what `MAX_LAG` bounds rather than what the design
pretends away.

### The can-fail proof

`test/registered/unit/distributed/test_barlink_bar1_abort_deferred_517.py`,
18 hermetic tests, CPU only. The two that carry the argument:

* `test_the_476_section_3_abort_raises_at_the_next_boundary` — reconstructs
  §3: 8-byte `all_to_all` recorded under capture on `tp:0`, word tripped
  between boundaries, `barlink_abort_gate.check_after_graph_replay()` raises
  with `op="all_to_all"`, `nbytes=8`, `rank 0/3`, "observed at cuda-graph
  replay", "0 collective(s) ran since the previous check" — the crash line,
  reproduced.
* `test_without_the_bound_the_deferred_read_is_blind` /
  `test_with_the_shipped_default_the_bound_forces_the_read` — the naive
  cheapening, shown failing. With a never-ready event and `MAX_LAG` raised out
  of the way, 200 boundaries pass over a tripped word with no raise. With the
  shipped default the 4th unresolved check forces exactly one
  `synchronize()` and raises. That is the binds-proof for the default value.

RED against `96c7dc5d2b`: 13 of the 18 fail (5 pass — they pin behaviour that
was already correct). Two representative failures:

```
E  Bar1CollectiveAborted: barlink-BAR1 rank 0/3 group tp:0: a spin kernel took
   its abort path, observed at cuda-graph replay. ...
   (test_k_greater_than_one_throttles_the_boundary: pre-#517 the FIRST boundary
    reads, because CHECK_EVERY cannot reach Seam B)

E  AssertionError: Lists differ:
   ['all_reduce on group tp:0', 'all_gather on group tp:0', 'broadcast on group tp:0']
   != ['all_reduce', 'all_gather', 'broadcast']
   (test_the_label_is_the_bare_op_from_the_call_site: pre-#517 the label is an
    eager f-string)
```

`test_barlink_bar1_abort_431.py` (the #431 falsifier suite) stays at 21/21:
its `__new__` transports carry a CPU `_ctl_dev`, which
`should_defer_status(False, True) == False` keeps on the blocking read, so
every pre-#517 assertion keeps its exact meaning.

### What stays unmeasured

* **The whole benefit.** Every number in §3's table is a count of syncs, not a
  measured delta. Nothing here has run on a card. The claim is "8 blocking
  syncs per decode round become 0 in the steady state"; whether that recovers
  the 9.22 % is the GPU window's question.
* `MAX_LAG = 4` is a desk-picked bound. It is proven to BIND (the falsifier
  above) and proven not to bind when the event resolves, but the real
  distribution of "how many checks does a staged copy stay in flight under
  overlap" is unmeasured. If the window shows the bound firing often, the
  saving is smaller than modelled and the number should move.
* Candidate 5 (host-visible status word) is unbuilt and unpriced.
* The pinned staging page is one 4-byte pinned allocation per transport (3 per
  rank). Not measured against the corridor; it is far below the 400 MiB rule's
  resolution, but it is a real allocation.

---

## 5 — THE GPU-WINDOW TICKET

Same recipe as #476 so the arms are directly comparable: `#435`'s
`fp8_decode_bar1` verbatim — Qwen3.6-27B-FP8, TP=3, `--rank-gpu-id 0,1,2
--rank-tp-ratio auto --rank-auto-reserve-mib auto --rank-perf-tune
phase-decode`, `--kv-cache-dtype fp8_e4m3 --context-length 131072
--max-running-requests 16`, NEXTN 3/1/4, decode layout only, in the
`htsglang:cu130-nccl2307` image on the PVE host. 4 draws per arm, draw 1
discarded, draws 2-4 as floor and measurement.

| arm | transport | guard | env | expectation |
|---|---|---|---|---|
| **N1** | NCCL | — | — | baseline, redrawn in the same window (a #476 number is not a floor for a new tree) |
| **B0** | BAR1 | pre-#517 | `..._ABORT_DEFER=0` | reproduce #476 A2b: **-9.2 %** vs N1 on code decode_TPS |
| **B1** | BAR1 | #517 staged | default | **≥ +2 %** vs N1, i.e. within the floor of #476's A4 (+2.68 %). This is the arm the ticket is about. |
| **B2** | BAR1 | none | `..._ABORT_CHECK=0` | the ceiling: A4 redrawn. B1 vs B2 is the residual cost of the staged read. |
| **B3** | BAR1 | #517 staged, K=5 | `..._ABORT_CHECK_EVERY=5` | candidate 2 on top of candidate 1. Expected ≈ B1 (nothing left to save) — if it is measurably faster, the staged copy itself is not free and §3's cost model is wrong. |

Also collect, per arm: `grep -c 'a spin kernel took its abort path'` (the §3
event is intermittent — every clean arm is a data point on its rate), the
transport gate (9 × `ACHIEVED=bar1`), the derived memory budgets (device-order
guard), the corridor at 2 s sampling, and `ms/round` at bs=1 and bs=8 —
#476's crossover says the guard-on arms already WIN at bs=8, so B1 should win
at both.

**Falsifier, stated before the run.** B1 must be able to lose. If
B1 ≈ B0 inside the floor, the staged read did not remove the cost and this
document's model of it — 8 blocking syncs per decode round — is wrong.
If B1 ≈ B0 *and* B2 ≫ both, the cost is in the check's mere presence and not
in its synchronization, which would point at candidate 5 and refute §3.

Second falsifier, for the guard itself: the window must show at least one arm
in which `SGLANG_BARLINK_BAR1_ABORT_DEFER=0` and the staged default agree on a
DELIBERATELY tripped word. If no injection harness exists on the card, say so
rather than inferring it from silence — a guard that never fires in a window
proves nothing about a guard that can fire.

---

## 6 — Reading done for this note

`CLAUDE.md` (whole file); `docs/dev/FEATURE_CATALOG.md` §7 (collectives /
transport) and §12 (robustness canon); §17 (combination matrix);
`docs/dev/ANALYSE_431_fp8_bar1_dcp_deadlock.md`;
`/spinning/gpu-battery-results/2026-08-03_w4_t2_476_bar1_floor/RESULTS.md`
§1, §3, §4; `docs/dev/TICKET_476_bar1_decode_floor.md` §1-§2 (read from
`/spinning/wt-476-bar1-floor`, which is where it lives).
