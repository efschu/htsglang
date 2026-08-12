# #631/#656 contradictions register

Every number in this corpus that two shifts disagreed about, in one place, so
a successor does not inherit a retracted claim by reading only the file it was
made in. Built by an audit of HANDOFF_657..674 + PROD_BRINGUP_BENCH.md
(successor 30, 2026-08-11).

**Read this before quoting any capacity number from a handoff.** Seven
capacity headlines in this chain have failed on contact. The pattern is never
sloppy arithmetic; it is a number that was correct for its own boot being
carried across a change that invalidated it.

---

## RESOLVED — do not re-litigate

| # | Quantity | Superseded claim | Standing claim | Closed by |
|---|---|---|---|---|
| C1 | weights-arena tail | 1773/1234/1191, 1773/0/1191 | **319/220/1191 FOR THIS GEOMETRY** | HANDOFF_673 §4 (corrected) |
| C2 | drafter spill payload | 1925 MiB/rank | **439/285/285 MiB** exclusively owned | HANDOFF_672 §1a |
| C3 | PP layer "quantum" | multiples of 4 | **no quantum; 2-layer granularity** | HANDOFF_666 §2b, HANDOFF_673 §1c |
| C4 | `15,9,8` snap | four layers | **two layers** (banker's rounding) | HANDOFF_666 §2b |
| C5 | pool lever | "exhausted", 26 MiB | **works**; 1028-1426 MiB sat in torch's cache | HANDOFF_666 §2a |
| C6 | 5090 context total | 19.58 GiB "driver wall" | **32088 MiB**, only a 519 MiB carve-out | HANDOFF_657 §4 |
| C9 | pool >= 600000 | "structurally unreachable" | **boots and serves**; corridor is the limit | HANDOFF_664, HANDOFF_666 §2b |
| C11 | corridor vs staging bound | "two distinct bounds" | **one bound, same buffers** | HANDOFF_664 §12b |
| C12 | the KV pool's "current" rows | `pool.size` | **`full_pool_backed_rows`**; `size` is a reservation and never moves | HANDOFF_676 §1b |
| C13 | kvso as the host-half destination | "the two halves already compose correctly" | **RESOLVED in HANDOFF_677 §2.** Was: kvso REFUSES FLIP ARMING unconditionally. Now `flip_blocking_guards` asks `kvso_flip_contract.flip_safety_state()` and refuses only `busy` -- presence is no longer a guard. The remaining blocker is host RAM, not config: one full-context region is ~12.9 GB node-wide at ctx 393216 (§2a) | HANDOFF_676 §2 -> HANDOFF_677 §2 |
| C16 | "PP is the larger layout on every rank of this rig" (`_arena_tail_bytes` docstring, `phase_flip_spill` carrier docstring) | a structural property, so the arena tail is committed on tp->pp and released on pp->tp | **falsified on metal**: `--pp-stage-ratio 15,9,8` derives 32,16,16 layers over 64 and puts the middle rank's PP layout BELOW its TP layout. The pp->tp refill then wrote into the released tail -- cudaErrorInvalidValue in the no-return region, all three ranks down. The safe span is `max(both layouts)` on EITHER leg | HANDOFF_677 §1a |
| C17 | the corridor law is enforced | `CorridorGuard.ensure_headroom` implements item 15a's spill-before-alloc, called from ONE site (the flip seam) | **RESOLVED in HANDOFF_678 §1a.** A second caller now sits at prefill admission (`corridor_admission.py`, wired in `_get_new_batch_prefill_raw`). It SPILLS and never REFUSES, because the guard is rank-local while admission must stay rank-uniform. Sibling sweep booked in the same section: decode extend and CUDA capture do not physically grow at runtime, `recover` is already law-bounded, and `vram_dial` grow is C18 | HANDOFF_677 §1a-bis -> HANDOFF_678 §1a |
| C18 | how many floor models are there | one: the corridor law at 1024 MiB, enforced by `CorridorGuard` | **two, and they had never met -- RESOLVED in #658.** `vram_dial._measure_local_floor_bytes` derived its OWN card-level NVML floor (`used - backed`) and every capacity computation spent `budget_bytes - floor_bytes`, so an absolute budget could plan a KV ceiling that consumed the card's last free byte and the guard would have to claw it back at an allocation. The law is now a TERM in that floor (`corridor_law_floor_bytes()`, importing `DEFAULT_FLOOR_MIB` from the guard rather than restating 1024), so the unit cap, the budget rows, the minimum viable budget, the effective ceiling and the below-floor refusal all inherit the reserve. **The boot state is bit-identical** -- the natural budget is `floor + backed`, so raising the floor raises the budget by the same amount and `budget - floor` does not move; the term bites only on an ABSOLUTE budget, which is exactly when an external tenant is taking bytes off the card | HANDOFF_678 §1a (sweep) -> HANDOFF_684 |
| C19 | item 16's rebalance tier is missing an ACTUATOR | the objective is computed and the mover is the gap | **the objective had no CALLER either.** `corridor_guard.water_fill_targets` had exactly one caller in the whole tree and it was a test, so five handoffs of "the rebalance tier is missing" ran while nothing in production ever asked how much to rebalance. **RESOLVED as an instrument in HANDOFF_679 §1e**: `water_fill_transfers` + `describe_water_fill` now render the move wherever the guard already reports the spread, and the measured answer on this rig is ~948 MiB stranded on the wrong card at the binding instant. The actuator is still absent and is now a decision with a price attached, not an open-ended ask | HANDOFF_676 §6.2 -> HANDOFF_679 §1e |
| C20 | where the corridor's DEPTH is made | the allocations the corridor gates price -- that is why the gates exist | **RESOLVED AS A MECHANISM in HANDOFF_681, with its residual named and measured.** The depth is made at seam ENTRY, and the entries are INHERITED: successor 37 measured on s34's own green window that the deepest minima come in PAIRS of cutovers ~2 s apart, the second entering at the first one's trough and drawing nothing (`13:59:11 entry 1499 draw 456 -> min 1043`, then `13:59:13 entry 1043 draw 0 -> min 1043`). The draw is also SELF-LIMITING -- from a low entry it was at most 456 MiB, the 1026 MiB draws all follow entries above 2400 -- so the fix is a condition on ENTRY, not a cap on the draw, and a worst-case entry requirement was rejected because it would arm on 77% of cutovers, which is s36's falsified cache-dumper by another name. The seam gate now asks for `staging + 512 MiB` and grades its verdict (enter / DELAY, budgeted in consecutive GROUP abandons / enter on the law loudly / refuse), riding the existing `_collective_min` with no new collective. **MEASURED over 65 min, 348 flips each way, 0 breaches:** the binding card's IN-CUTOVER minimum 1043 -> **1123 MiB** (+19 -> +99 over the law) and its draw-from-a-low-entry 456 -> **184 MiB**. **THE RESIDUAL, stated so nobody reads this as closed:** a cutover entering HIGH still draws up to 1040 MiB and lands at the same ~1123 floor, so the trough is now bounded by the seam's UNPRICED tail demand rather than by an unguarded entry. That is a staging-pricing problem (`SGLANG_FORWARD_PEAK_PATH`), not an entry-condition one. Price of the margin: the KV rung fired 348 times against s34's 21, every shrink recovered and the pool never left 512552. **RESIDUAL 1 CLOSED in HANDOFF_682**: the rung now keeps an ADMISSION RESERVE above the live high-water mark (the scheduler's own `chunked_prefill_size`, 512 here) and the gate declares its margin DISCRETIONARY, bounded by what the rung can return without crossing that floor. Proven with the same forced 8192 MiB margin that killed the instance: 156 delays, 78 yields (s37: 72/35 then `full_available_size=0` on all three ranks), 0 admission failures, 0 tracebacks, health 200, 0 breaches in 3926 samples. Inert on the shipped margin — floor 593 rows against 512552 backed, 0 bounded asks — and the 30-min confirmation window held every axis (C20 VERDICT: HELD, 0 breaches in 13118 samples, MTP 2.869, YaRN 271237). **RESIDUAL 2 is now C21** | HANDOFF_680 §1a -> HANDOFF_681 §2 -> HANDOFF_682 |
| C21 | the seam's staging price against what the seam HOLDS | HANDOFF_681 §2b: "a cutover entering HIGH still draws ~1040 MiB that `_staging_bytes` does not price", read off the gpu0 corridor column | **THE METAL HALF IS A CARD MIX-UP, THE HERMETIC HALF IS REAL.** `--rank-gpu-id 0,1,2` is in CUDA order, which is FASTEST_FIRST on this rig, so **rank 0 is the 5090 (nvidia-smi index 1)** and ranks 1/2 are the 3080s (smi 0 and 2) -- confirmed three ways: the per-rank MiB vector `31800,14000,15600`, the guard's own free readings (PP1 clears at 1886/2726 MiB match the gpu0 column, PP2's 2968/3236 match gpu2), and the boot's own profile (rank 0 capacity 619990 tokens against rank 2's 308808). Joining `PP0` log lines to the `gpu0` column therefore crosses two different cards. Joined correctly (`scripts/s38_seam_price_vs_draw.py`), the binding card's 115 pp->tp seams were **priced at 1177 MiB p50 (max 1478) against a deepest drawdown of 504 MiB**: the gate OVER-reserves there, it does not under-reserve, and no unpriced tail is visible at the corridor. **What IS real is smaller and hermetic:** with the wave plan matched, `_staging_bytes` prices 2.398 MiB against a measured live set of 3.769 MiB on the three-rank CPU fixture (0.64x) -- the formula models `incoming + max(outgoing, local)` and the probe's high-water holds an outgoing leg beside the local read and its gather window. Not reproduced on metal (torch's allocator cache absorbs it, which is why 348 flips saw 0 breaches), NOT widened on that evidence, and pinned as a one-sided ratchet in `test_the_waved_price_is_short_of_the_measured_live_set` | HANDOFF_681 §2b -> HANDOFF_682 |
| C22 | what item 16's REBALANCE tier can actually steer | HANDOFF_679 §1e and the corridor_guard docstring: the ABSORB half is missing because the DCP owner rule fixes a token's row to a boot-constant vector, i.e. the gap is an ACTUATOR for moving ownership | **the actuator exists and it is the free list; the gap is that ownership does not decide RESIDENCY.** Built in #657: every allocation path takes the HEAD of `free_pages`, so a stable residue-class partition steers all of them at once, places bytes without moving or freeing any, and cannot starve an allocation. Its DECISION works on metal -- UUID permutation `[1,0,2]` agreed over the group, every promoted count identical on all three ranks -- but **its APPLICATION killed the instance**: the bias is re-applied on a rank-local 1 s clock, so three ranks re-sort at three different instants, and at t+18 a pp->tp cutover died on `KvReshardError: payload checksum mismatch -- refusing to scatter` while the steer's own replication check disarmed it in the same second on all three ranks. A pure function applied on a private clock is not a group-uniform mutation of replicated state. **But it cannot move committed VRAM, and the reason is structural:** the scheduler keeps ONE allocator for process life -- the PP stack's (`phase_flip_boot.py:750-756`) -- whose layout has `dcp_size == 1`, so a slot id IS the row on every rank; the TP pools are pre-sized at boot with no growth (`phase_flip_runtime.py:24-26`); and the only lever that returns driver bytes is the PP pool's backing watermark, floored by the MAXIMUM LIVE SLOT ID (`kv_backing_relief.py:406-441`) and released only at the pp->tp gate. So which rank OWNS a row changes nothing a card can feel, and a class bias can only raise the id ceiling (pinned hermetically in `test_a_class_bias_raises_the_maximum_live_id`). **The money is in the ceiling, not the ownership:** on s38's own shipped window the floor sat 55k-210k rows above the live token count (median ~71k = 592 MiB on the binding 3080, 1044 MiB on the 5090) while the traffic returned `#cached-token: 0` on 41952 batches. Instrumented in #657 (the rung's line now names whether the radix tree or a resident request pins the ceiling); NOT actuated | HANDOFF_679 §1e -> HANDOFF_683 |
| ~~C20 (superseded reading)~~ | | | **neither gate can reach it.** s34 logged 232 gate clears, 0 refusals and a 1043 MiB minimum in the same window; the gate frees to `floor + delta` BEFORE the allocation it guards, so no guarded site can leave free below 1792 MiB. The depth is made where no gate looks. Successor 36 localised it: **20/20 of the deepest gpu0 troughs sit within 2 s of a flip cutover** (`evidence-631/s36/TROUGH_VS_CUTOVER.txt`), i.e. INSIDE a code path that does not yield to the scheduler loop, which is why a per-round lender lifts the between-rounds level (measured: 1626 -> 2088 MiB on the binding card) and cannot touch the trough itself. The next margin is in the seam's staging demand, not in the resting level | HANDOFF_680 §1a |
| C14 | recovery's cost | "an idle boundary, where an allocation is affordable" | **falsified**: recovery drove a card to 6 MiB free; it is as large as the relief it undoes | HANDOFF_676 §1c |
| C15 | the fill lever | `--max-total-tokens` | **`--rank-gpu-memory-mib`**; the pool clamps at `_profile_available_bytes` (620000 asked -> 512552 given) | HANDOFF_676 §1d |

### C1 is a correction OF a correction, and the most instructive one

HANDOFF_673 first said its measurement "refutes" 1773/0/1191. It does not.
Rank 2 matches to the decimal and rank 0's TP matches; only the PP figures
move, because the two boots ran **different PP layer splits**. Both were
right about their own geometry.

    tail = max(pp_bytes, tp_bytes) - min(pp_bytes, tp_bytes)

is a function of the split, not a constant of this rig. **Anything that moves
layers between PP stages moves the tail**, including the GDN-only cut A/B.
Re-measure from the `TP stack built` boot line on every boot whose split
differs; never carry a tail figure across a geometry change.

### C7 — the binding phase is not a fact, it is a state

Recorded successively as: TP binds both 3080s (pool 190000) -> PP binds both
(pool 500000) -> TP binds all three (pool 500000, after the drafter spill).
None was wrong. The binding phase is a function of BOTH the pool size and the
residency state, and **installing a spill moves it**, which is a feedback
loop, not a measurement error:

> HANDOFF_672 §4: "the binding phase MOVED to TP, on all three cards. Every
> card now binds where the drafter is resident by design and this rung is
> worth nothing."

Price no spill without re-measuring which phase binds AFTER the last one
landed.

---

## OPEN — flagged, never resolved

**C28 — ROOT-CAUSED BY SUCCESSOR 44, FIXED IN `b2f18010c2` BY SUCCESSOR 45
(2026-08-12). A session that finishes ON HOST deletes its own terminal output
before the streamer reads it, and the caller hangs forever (#659).** This is what stood behind C26. With C26
fixed the instance survives the park, so for the first time a session got
through the whole round trip alive — and the round trip is not the problem.
`rid=s44-sat-3` spilled, parked to the file tier, unparked cleanly
(`identity_miss=0`, zero failures, both ranks identical), then finished on
host two seconds later. **The HTTP client never received a byte and blocked
forever against a scheduler reporting `#running-req: 0`.** 40 further
requests did not release it; `abort_request` on the rid returned 200 and was
a no-op, because the req slot had already been freed.

**THE MECHANISM IS AN ALIAS, AND IT IS EXACT.** `release_finished_spilled_req`
runs from INSIDE the per-request loop of `process_batch_result_decode`
(`batch_result_processor.py:932-938`, gated on `req.kv_spill_state == "host"`),
and it calls `slot.batch.filter_batch()` (`kv_session_offload.py:5868`). For a
spill tick, **`batch` IS `slot.batch` — one object, two names**
(`maybe_take_tick` returns the persistent batch at `kv_session_offload.py:4930,
4974`; the scheduler runs it as `ret = spill_tick_batch`, `scheduler.py:4865`;
under `--disable-overlap-schedule` `event_loop_normal` passes it through
unchanged, `scheduler.py:2232-2233`). `filter_batch` keeps only unfinished
reqs and, when none survive, rebinds `self.reqs = []`
(`schedule_batch.py:3273-3277`). The enclosing `for` loop already holds the
old list so it finishes normally — and then, ~100 lines later,

```python
self.output_streamer.stream_output(batch.reqs, batch.return_logprob)
```

(`batch_result_processor.py:805`) reads the NEW, EMPTY list.
`_stream_output_generation([])` accumulates nothing, `payload is None`,
nothing is sent, the detokenizer never sees a finished chunk, the tokenizer
manager's waiter never resolves. **The request deleted its own completion
from the list that was about to carry it.**

**AND THE FIX FOR THIS ALREADY EXISTS, ON THE OTHER EXIT.** The abort exit
has `_stream_terminal` (`kv_session_spill_destination.py:1537-1546, 1562-1572`),
added by `bcc72dd569 [#552]` with the comment *"Without it the abort frees the
memory and the caller hangs."* That is this bug, written down, on the sibling
path. `kv_session_offload.py` contains **zero** occurrences of
`stream_output`, `output_streamer`, `send_to_detokenizer` or `_stream_terminal`.
The same hole is on the pre-schedule reap (`kv_session_offload.py:4520-4524`).

**WHY EVERY SPILLED SESSION ON THIS BOOT LANDED ON THAT PATH (contributing,
not the wedge).** Restore was unreachable twice over, so finishing on host was
forced rather than chosen:

* an early return before the gate whenever ANY fast-lane request waits
  (`kv_session_offload.py:4707-4712`) — `s44-target` waited throughout;
* the gate itself, `fits_now = restorable >= remaining + restore_margin_tokens`
  (`:4753-4757`): `remaining = 3372 - 3010 = 362` against
  `restore_margin=4096`, so it demands `restorable >= 4458` **while the entire
  pool is `max_total_tokens=4096`** — unsatisfiable by construction. The
  default is an absolute token count (`server_args.py:1870-1877`) validated
  only against `< 0` (`:6983-6984`) and **never sized against the pool**.

A restored session would have finished in the device batch and streamed
normally, which is why `RESTORE complete:` count 0 and the hang co-occur.
**The park is exonerated:** `_commit_unpark` re-inserts the same slot object
and restores every flag (`kv_session_spill_destination.py:1479-1482`), and
`slot.batch` survives the round trip untouched. Had the session finished
WHILE parked, `_release_parked_req` would have streamed it and the client
would have been served.

**THE DISCRIMINATING OBSERVATION, CPU-ONLY, FOR WHOEVER FIXES THIS.** What the
metal run witnesses is that nothing was streamed and that the path has no
emit; it does not directly witness that the finish was processed on the
spill-tick batch rather than via `release_kv_cache` (`mem_cache/common.py:1098`,
equally emit-less but a DIFFERENT fix site). Drive
`process_batch_result_decode` over a one-req spill-tick batch with
`slot.batch is batch` whose req finishes, and assert the streamer receives a
non-empty list. It will receive `[]`. Secondary prediction that discriminates
the alias from a generic missing emit: **the same boot WITHOUT
`--disable-overlap-schedule` should not hang**, because `event_loop_overlap`
snapshots `batch.copy()` (`scheduler.py:2315`, `schedule_batch.py:3474-3479`),
so `filter_batch` cannot empty the list being streamed.

**THE FIX, AND WHY IT IS NOT THE TWIN THAT WAS PROPOSED (successor 45).** The
CPU falsifier above was written
(`test/registered/unit/managers/test_host_finish_stream_659.py`) and it is red
on the pre-fix tree exactly as predicted — verified by reverse-applying the
patch, not by assertion. The fix is NOT a `_stream_terminal` call inside
`release_finished_spilled_req`, which is what the abort-exit twin would have
suggested, and the reason is worth keeping: **the alias exists only when
overlap is off.** Under the overlap loops the result processor runs on
`batch.copy()` — a list the release cannot reach — so an emit inside the
release would DOUBLE-stream at all three overlap dispatch sites
(`scheduler.py:2314-2315` `event_loop_overlap`, `scheduler.py:6071-6080` the
decoupled spill lane, and `pause_generation`'s drain). Instead
`process_batch_result_decode` now binds `reqs` and `return_logprob` ONCE before
its loop and streams from that binding — the same stream-then-filter ordering
`disaggregation/decode.py:2113-2116` already uses. `return_logprob` is
snapshotted too, because `filter_batch` recomputes it from the SURVIVING
requests (`schedule_batch.py:3316`), so a finished request that asked for
logprobs would otherwise be streamed without them.

**CORRECTION TO THE ENTRY ABOVE, small but load-bearing:** the loss is not
confined to the "no request survives" branch. `filter_batch()` keys on
`not finished()`, so a finished spilled request is ALWAYS dropped, and the
survivor branch (`schedule_batch.py:3293`) is equally a rebind. In production
`slot.batch` is a bs=1 batch so the empty branch is the one taken, but a fix
keyed on "the batch became empty" would have been incomplete in principle.

**SIBLING SWEEP (successor 45), so the next shift does not redo it.** Every
`filter_batch()` call site was audited. Two are of the same silent-drop class
and emit nothing: the pre-schedule reap (`kv_session_offload.py:4520-4524`) and
the tick-batch drain (`:4931-4934`). Both are reachable only for a request that
is already `finished()` — hence already streamed — so they are LATENT, not
live, and were deliberately left alone rather than patched with unvalidated
defensive code that could double-emit. The prefill result path, `retract_decode`,
`retract_all` and `pause_generation` are clean (retracted requests are
re-queued; the solo-OOM aborts get a direct `AbortReq` that never reads
`batch.reqs`). The park abort exit (`_release_parked_req` → `_stream_terminal`)
already has the correct stream-then-filter ordering and is the reference shape.

**C29, the restore-margin default is an absolute token count that is never
sized against the KV pool (#659, successor 44 observed it, successor 45
confirms it is independent of C28).** `--kv-session-offload-restore-margin-tokens`
defaults to 4096 (`server_args.py:1870-1877`) and is validated only against
`< 0` (`:6983-6984`). The restore gate is
`fits_now = restorable >= remaining + restore_margin_tokens`
(`kv_session_offload.py:4753-4757`). On any boot whose `max_total_tokens` is at
or below the margin, the gate demands more than the entire pool and **can never
open** — which is why successor 44's boot logged `RESTORE complete: rid=` zero
times and every spilled session finished on the host floor. It is not a
consequence of C28 and C28's fix does not touch it. Successor 45's probe works
around it by CONFIG (`--kv-session-offload-restore-margin-tokens 64` against a
4096-token pool), deliberately not by code, so that the sizing defect stays
visible and gets its own commit rather than being smuggled into a streaming
fix. **The margin should be expressed against the pool** (a fraction, or an
absolute value clamped to a fraction of `max_total_tokens`) with a boot-time
refusal when it exceeds the pool.

**C28 IS CLOSED ON METAL (successor 45, probe v9, 2026-08-12).** The same
pressure band that wedged successor 44's client ran again on the fixed tree
(boot commit `b2f18010c2`). `rid=s45-cohort-3` spilled, PARKED to the file
tier (33 blobs / 3 760 128 B on disk, peak host tier 536 903 680 B), UNPARKED
(`identity_miss=0`, zero `UNPARK ... FAILED`, both ranks logging identically),
finished ON HOST — and **completed, with its output delivered to the HTTP
client** (1744 chars, `round_trip_completed: ['s45-cohort-3']`). The whole
cohort drained with `errors=0` and nothing hung. That is the exact request
shape that blocked forever on the pre-fix tree and could not be released by 40
further requests or by `abort_request`.

**BYTE-IDENTITY IS NOT ATTRIBUTABLE ON THIS BOOT, and the instrument says so
rather than guessing.** Neither the parked arm nor the CONTROL arm matched the
quiescent reference (`parked_arm_identical_to_reference: False`,
`control_arm_identical_to_reference: False`). Per `park_complete_proof2.py`'s
own verdict table that is the "this run separates nothing" outcome: requests
that never parked also diverge, so the divergence is the rig's known
batch-composition nondeterminism (HANDOFF_686's standing rule) and NOT
evidence against the round trip. It cannot be bought with determinism either —
see C30. **Byte-identity across a park therefore remains UNPROVEN, and it is
unproven for a reason that is now understood and booked, not for lack of
trying.**

**C30, `--enable-deterministic-inference` and `--enable-kv-session-offload`
cannot both be on: the instance boots, declares itself ready, and then admits
nothing (#659, successor 45, measured 2026-08-12).** Booked because the next
shift that needs byte-identity will reach for the same flag and lose an hour
to it. The intersection of the two constraints is a single backend and it is
NOT a free choice:

* `--enable-kv-session-offload` REFUSES any backend but flashinfer
  (`ValueError: --enable-kv-session-offload requires the flashinfer attention
  backend; got --attention-backend=triton`), so HANDOFF_688 §1e's prescription
  of triton does not boot at all. The deterministic default (fa3) is
  Hopper-only and does not boot on these SM86 cards either.
* flashinfer IS on `DETERMINISTIC_ATTENTION_BACKEND_CHOICES`
  (`server_args.py:233-239`), so it is the only backend satisfying both — and
  it is not on `RADIX_SUPPORTED_DETERMINISTIC_ATTENTION_BACKEND` (`:241`), so
  `disable_radix_cache = True` is set silently (`:15500-15505`).

**What that combination then does, measured:** the instance boots clean, prints
"The server is fired up and ready to roll!", serves its two warmup prefills —
and never admits another request. A trivial 8-token `/generate` hung for 55 s;
`/health` timed out while `/get_model_info` answered instantly (so the HTTP
loop was healthy and this is not an entrypoint stall); the collective census
froze at 1862 all_reduce; **zero `Decode batch` lines appeared in the entire
boot**; py-spy showed TP1 spinning inside `add_one_req`
(`schedule_policy.py:1255`) while rank 0's metrics reported
`num_queue_reqs 0` — a rank-asymmetric view of the same admission decision.

**Attribution is clean, one variable.** The same boot with the two
deterministic flags removed and everything else byte for byte identical
(`probe_boot_v9.sh` vs `probe_boot_v8.sh`, diff = one line) served a
generation in **0.36 s** and reproduced the park on its first attempt.
Evidence: `/spinning/evidence-631/s45/v8_deterministic_wedge_pyspy.txt`,
`v8_wedge_metrics.txt`, `probe_v8.log`, `probe_v8_triton_refused.log`.

**Consequence for the byte-identity arm:** it cannot be bought with
determinism on this rig while kv-session-offload is on. It must be earned the
way `park_complete_proof2.py` already earns it — a homogeneous cohort whose
CONTROL arm measures whether batch-composition nondeterminism is live in that
boot, so a mismatch can be attributed instead of guessed.

**C27, the corridor law was enforced by nothing at the place it is spent
(#656).** Successor 42's confirmation window breached: 12 samples, gpu0 at
941 MiB for 1.5 s. Root-caused by successor 43 to a `pp_to_tp` cutover on
**rank 1 (= nvidia-smi 0, a 3080)** — not a prefill, not a graph capture; the
instance was quiescent by design and the descent is the arena's
`backing_restore_span` walk. **It is a LATENT DEFECT, not a regression, and
that is shown rather than argued:** s38's GREEN window contains the SAME
event — one excursion per window, 12 samples, 1.54 s vs 1.53 s, at the same
point of the acceptance script (the completion of the 271k YaRN leg) — and
was green by 59 MiB. s34's was green by 19 MiB. The three windows are three
draws from one distribution whose left tail straddles the floor:

| window | entry free | draw | trough | vs law |
|---|---|---|---|---|
| s38 | 3469 | 2386 | 1083 | **+59** |
| s34 | — | — | 1043 | **+19** |
| s42 | 3006 | 2066 | **940** | **−84** |

Three things were wrong at once, and each alone was survivable:

1. **The remedy's trigger was a hardware failure, not the policy floor.**
   `_mem_create_reclaiming` already knew that torch sits on
   reserved-but-unused blocks and that `empty_cache` returns them — but it
   fired on `CUDA_ERROR_OUT_OF_MEMORY`, i.e. free reaching ZERO, which is
   1024 MiB below the law. The census recorded `slack=1054` at the 940 MiB
   trough: **the bytes to stay legal were held throughout and nothing asked
   for them, because nothing had failed yet.** (Now law 16.)
2. **The recogniser held the evidence and never read it.**
   `phase_flip_seam_census` samples the exact NVML observable at every stage,
   names the 1024 MiB floor in its own docstring, and contained no comparison
   — and emits its line only AFTER the flip completes. The process measured
   940 and said nothing; the breach was found hours later in an external CSV.
   Note also that **the 100 ms external sampler UNDERSTATES depth** (it read
   PP0's floor as 2578 where PP0's own census recorded 2375), so the census
   is the tighter instrument and should be the one judged against.
3. **The seam-entry law check was priced on the wrong term.** It subtracts
   `staging_bytes` — what the seam RESERVES — from `verdict.free_after`. The
   census measures the DRAW at 2066 MiB against 1625 MiB staged, and
   `free_after` itself overstates the 3006 the cutover actually entered with.
   s38's own yield was **18 MiB sub-law by that same arithmetic** (free 2190,
   staged 1184) and survived only because its estimate overshot the real draw
   by ~388 MiB. Passing on an estimator's conservatism is not a margin.

**Fixed red-first in `18ff17ec6e`**: the trigger moves to "the next commit
would cross the law" (`kv_vmm_backing._corridor_preempt`), the census compares
and announces at the stage that crosses, and the gate prices a PREDICTION on
the measured draw. **The yield is deliberately left able to yield**: refusing
`pp_to_tp` starves decode outright (411 abandons, 0 completions in 6 minutes,
/health 503, measured 2026-08-10), so the actuator was put where the bytes are
taken rather than in a predicate that can only refuse. Close/reopen trigger: a
>=30 min ship-config window with 0 breaches **and** at least one
`corridor law floor` preemption line proving the mechanism was reachable — a
green window with the mechanism never armed proves nothing.

FALSIFIED ALONG THE WAY, so nobody re-runs them: N41's merged changes are
**inert** (the #657 allocation steering ships OFF and has zero fingerprints in
either log; the registry observer and P2 gate fix are observation/refusal
paths that allocate nothing); graph capture is identical in both windows
(bs=[1,2,3,4], 0.56/0.30/0.28 vs 0.57/0.32/0.32 GB); the environment blocks
are byte-identical. The pool difference (512552 vs 503950) is **a warm page
cache**, not code: weight load took 11.11 s cold and 2.35 s warm, and the
faster load leaves ~0.1 GB more allocator high-water on every rank.

**C26 — ROOT-CAUSED, FIXED IN `bdb2c3db53`, AND NOW PROVEN ON METAL
(successor 44, 2026-08-12).** The probe boot that killed the instance for
successor 42 ran the same pressure band to completion: a session spilled,
parked and unparked with **zero** device-side asserts, zero tracebacks, and
the instance stayed healthy afterwards and served 40 further requests. The
fix holds. It bought exactly what it promised and no more — the crash is
gone, and what stood behind it is now C28.
The mechanism is not a race and not a corner: **PS2 is admitted onto a backend
that has no hook to divert its sentinels.** `spill_extend_alloc` returns
`make_sentinels(...)` AS `out_cache_loc`, and exactly one thing in the tree
diverts that tensor away from the KV write — `_dcp_write_scatter`'s
`_sess_prefill_owner_write` branch, reachable ONLY from the token-sharded DCP
lane (`forward_extend` enters it under `if self.uneven_dcp`). On plain TP the
backend still BUILDS the prefill-spill state and nothing complains;
`forward_extend` falls through to the stock `set_kv_buffer` and the sentinels
reach `store_kvcache`. The arithmetic closes exactly: `host_base=4097` against
a 4096-row allocator, `boundary=2620 L=3012`, so the 392 written indices are
**6717..7108 against a `size_limit` of ~4097** — every row out of bounds,
layer 0, both ranks.

Two premises from HANDOFF_686 are corrected. **The crash is NOT delayed** —
the ~60 s gap is a coincidence of the log; the parks at 01:37:28 completed
cleanly and are unrelated, `fast-3` was admitted at 01:38:26 and the next line
is the exception, so the assert comes from that same extend forward and
`adopt_born_spilled_prefills` is never reached. **The #501 flag-after-last-
decline hypothesis is refuted for this crash** — `req.born_spilled_deep =
False` at the end of `spill_extend_alloc` has no later reader.

Fixed at the admission gate, where the information was missing rather than
wrong: `prefill_spill_deep_gate(backend_write_hook=...)` from
`_sess_mode != "plain"` — replicated boot config, identical on every rank, no
collective, and a decline is a NON-ADMISSION rather than a rank-local skip
around one (law 14). Plus a Python `RuntimeError` in `spill_extend_alloc` as
belt and braces. Close/reopen trigger UNCHANGED and still open: a boot that
drives >=2 concurrent spills to a park and back **with a request COMPLETING
afterwards**. The band is now known — `--chunked-prefill-size 256` removes PS2
entirely (the 392-token tail then chunks and fails `prefill_spill_deep_ok`'s
one-chunk condition) while leaving PS1, the fast-lane spill and the park path
untouched. HANDOFF_687 §2b.

**A SEPARATE, REAL BUG FOUND IN THE SAME TRACE AND DELIBERATELY NOT FIXED
HERE (open):** `_admit_born_spilled_deep` sets `req.born_spilled_deep`,
`prefill_spill_deep_taken` and decrements `prefill_spill_regions`
(`schedule_policy.py:908-910`), and `add_one_req` can still bail out
afterwards (hybrid-SWA `:1290`/`:1322`, dllm `:1352`, truncation
`:1384`/`:1391`/`:1402`) without clearing any of it. The request is never
appended to `can_run_list`, that iteration's region accounting is lost, and a
later adder with `deep_taken=False` can admit the stale-flagged request
WITHOUT re-entering the deep branch — after which it reaches
`prepare_for_extend` and either takes an unintended PS2 allocation or trips
the mixed-batch raise. This IS the #501 flag family, just on the decline side
rather than where HANDOFF_686 looked. It wants its own change with its own
reproducer.

**C26 (ORIGINAL ENTRY, kept for the record), #224's PS2 born-spilled-deep path kills the instance (#659 window).**
Reproduced twice in one shift, both times a device-side assert out of the KV
path, both times with ZERO #659 code involved. The clean signature is PS2:
`admit rid=... BORN-SPILLED DEEP (input=392 does NOT fit device budget=102) --
prefilled straight into a host region, no device KV slots`, immediately
followed by `PREFILL-SPILL (PS2, born-spilled deep) ... NO device KV slots
allocated` and then `torch.AcceleratorError: CUDA error: device-side assert`.
The earlier one (uneven TP=3, --max-total-tokens 8192) asserted inside
`kvcache.cuh:112 store_kvcache: index >= 0 && index < size_limit`. Both look
like a KV row index computed against a device allocation that is not there.
This is what stopped #659's end-to-end proof: the park round trip completed,
then the instance died before any parked session could finish. Reproducers:
`evidence-631/s42/probe_boot_v5.sh` (parks, then PS2-crashes) and
`probe_boot_30040.sh` (the uneven-TP variant). Belongs to the #622/#649 lane.
Close/reopen trigger: a boot that drives >=2 concurrent spills to a park and
back with a request COMPLETING afterwards. HANDOFF_686 §1d-BIS, §1f.


**C25, kv-session-offload cannot run on the Route-A ship recipe at all
(#659).** Five briefs have asked for kvso to be proven "on the Route-A recipe
plus `--enable-kv-session-offload`". That boot does not exist, and the refusal
is at arg-parse, not a runtime surprise: `--enable-kv-session-offload (S1)
supports single-node pure TP/DCP only (pp_size=3, dp_size=1)`. Route A **is**
PP=3 prefill -> flip -> TP=3 decode, so the ship recipe and the feature are
mutually exclusive by construction. Successor 42 ran the metal probe on the
DECODE layout instead (pure TP), which means **no shift has ever exercised the
phase-flip x kvso crossing, and none can until S1 changes.** Any plan that
assumes the two compose is planning against a gate that says otherwise.
Close/reopen trigger: S1 admitting PP, or an explicit decision that kvso is
tested only on the decode layout. Corollary worth carrying separately: the
byte-identity of a kvso restore may NOT be argued from two model generations on
this rig — reference and under-load continuations differ with ZERO parks
(GDN prefill nondeterminism), so the claim has to be made about the transported
BYTES. HANDOFF_686 §1a/§1c.


**C24, the bundled profile's remote rows describe a path this process cannot
route to (#659).** `memtier/profiles/rig1.json` carries `host:rig-2` with
transport `roce-40g` and bandwidth **2.83 GB/s MEASURED**, and
`kv_session_spill_destination.py:29`/`:153` quote **3.43 GB/s RDMA write,
1.47 us** in prose. Measured from inside the serving container by successor 41
(`evidence-631/s41/TIER2_LINK_MEASUREMENTS.md`), the second rig is reachable
only over its 1 GbE `enp7s0`: its 100G/40G NICs sit on `10.10.10.2/30` and
`192.168.40.10/24`, **both unreachable from here**, and `169.254.17.33`
answers ICMP but measures identically to the 1 GbE address. Bulk throughput
**75 MB/s on both paths** (dd 1500 MiB through nc), RTT 0.265 ms avg. The box
also has only **8629 MiB available RAM plus a 64 GiB swapfile**, against a
~12.9 GB full-context kvso region (C13) — so remote "RAM" there is
swap-backed and cannot honour a pinned-residency contract either. Neither
number is wrong about the machine it was taken on; both are numbers carried
across a change in REACHABILITY, which is law 1 in its network form. **Nothing
may size a remote tier against rig1.json without re-measuring the path from
inside the container.** Close/reopen trigger: a boot in which `10.10.10.2` or
`192.168.40.10` is routable from the serving container.


**C23, two independent drivers of one VMM release primitive (#658).**
`pool.runtime_set_backing_rows` -- the only call that hands KV pages back to
the driver -- has TWO callers that do not know about each other.
`KvBackingRelief` drives it from the corridor law floor
(`kv_backing_relief.py:818`, `:941`, floor read at `:921`) at the flip seam,
behind a MAX reduction. `KvCapacityRuntime` drives it from the budget vector
(`vram_dial.py:986`) at a fully-idle consensus boundary, behind a MIN
reduction. #658 wired the DECISIONS together -- the dial's floor now carries
the law, and a budget cut spends the guard's ladder first -- but the two
actuators still race on the same pool, on different cadences, with different
group protocols. Nothing has been measured breaking, and nothing has been
proven safe either: the dial is off in the ship config, so the two have never
run together under load. **The reopen trigger is the first boot that ships
`--enable-vram-dial`**, and the question to answer then is which one owns
`backed_rows` when both want to move it in the same round.


**C10, the per-token slope.** Four values coexist for rank 1 and they do not
reconcile: 10.30 MiB/1000 (total idle, HANDOFF_665), 9.10 (HANDOFF_664),
9.766 resident + 4.517 staging (HANDOFF_667). The decomposition sums to
14.283 against a measured total of 10.30. HANDOFF_667 §3 notices its own
`staging_coeff` of 0.001906 MiB/token "contradicts its OWN measured slope"
and leaves it there.

Anything sized against a slope is sized against an unresolved number. If a
capacity projection matters, measure the two endpoints rather than
extrapolating.

**Contaminated readings.** HANDOFF_666 §2a: every NVML-free reading in this
corpus taken WITHOUT a preceding `/flush_cache` is contaminated, because over
a gibibyte per card of torch allocator cache is held at zero requests. That
retroactively applies to numbers still quoted elsewhere.

---

## SINGLE-SOURCED AND LOAD-BEARING — the next contradictions

Each appears in exactly one place and a decision rests on it. Confirm before
reusing:

| Claim | Only source | Decision it carries |
|---|---|---|
| MambaPool uses `torch.zeros` (no driver-returnable payload) | HANDOFF_673 §1b | the mamba/GDN rung is dismissed as 0-byte |
| ~~kvso never calls `cuMem*`~~ | **CONFIRMED, no longer single-sourced** | see below |
| `staging_coeff` 0.001906 MiB/token | HANDOFF_667 | the 432861-token ceiling |
| staging slope 4.517 MiB/1000 at W=4 | HANDOFF_667/668 | the 601233 restore-first ceiling |
| KV 15.0 / 8.5 / 8.5 KiB per global token | HANDOFF_661 | the 669440 conservation baseline |
| 670803-token fully-streamed seam ceiling | HANDOFF_668 | the 2.1 vs 2.1b design choice |

The 669440 baseline being single-sourced is the uncomfortable one: it is the
conservation check every capacity table is supposed to be judged against.

### "kvso never calls `cuMem*`" is now CONFIRMED (2026-08-11, successor 31)

An independent audit re-ran the search over
`managers/kv_session_offload.py` and reported **zero matches** for
`cuMemUnmap`, `cuMemRelease`, `empty_cache`, `decommit_range`,
`runtime_set_backing`, and `shrink`. Two independent sources now agree, so the
decision it carries — kvso cannot be the guard's byte-returning provider, only
the DESTINATION for bytes the VMM arena releases — no longer rests on one
reading.

What a spill actually frees is device SLOTS, by two calls, and neither reaches
the driver: `allocator.free(over)` (`:3754`, speculative overhang) and
`allocator.free(seg)` (`:3856`, the tail segment whose rows were just copied
to the pinned host pool by `host_pool.backup_from_device_all_layer`, `:3790`).

---

## THE LAWS THIS REGISTER PRODUCES

1. **A number is valid for its geometry, its pool, and its residency state.**
   Say which, or it will be carried somewhere it is false (C1, C7).
2. **A memory-usage delta is not a payload.** Price from bytes the payload
   exclusively owns AND can return to the driver (C2, plus the mamba and kvso
   dismissals).
3. **Flush before believing an idle NVML reading** (C5).
4. **Two data points do not establish a quantum.** C3 and C4 were both an
   exhaustive enumeration away from being avoided.
5. **A fix can move the thing it was measured against** (C7). Re-measure
   after landing, not only before.
6. **Hang relief on a clock that ticks inside the trough** (C20). The
   convenient hot path is not evidence that the pressure is on it, and two
   numbers from the baseline extract will usually say where it is.
7. **Condition the measurement on the state the mechanism will see** (C20,
   HANDOFF_681 §1a). The marginal distribution of the seam's draw said
   "reserve 1026 MiB, arm on 77% of cutovers"; the same data conditioned on a
   low entry said "reserve 456, arm rarely". A quantity that RESPONDS to
   pressure will overstate what a mechanism must reserve if you price it
   marginally.
8. **A budget denominated in a rank-local event cannot bound a group action**
   (HANDOFF_681 §1d). The first cut of C20's delay budget was per-rank while
   the abandon is group-OR, so three ranks taking turns being short refunded
   each other forever: 30 attempts, 0 yields, pp->tp delayed indefinitely --
   the decode wedge, reached through the mechanism built to prevent it. The
   currency has to be something every rank reads identically, and the reduced
   verdict already was one.
9. **A rank id and a card index are two different permutations, and this rig
   proves it.** `--rank-gpu-id 0,1,2` is read in CUDA order; `CUDA_DEVICE_ORDER`
   defaults to FASTEST_FIRST, so CUDA's device 0 is the 5090 while nvidia-smi's
   index 0 is a 3080. Every corridor CSV in this corpus is in nvidia-smi order
   and every log prefix is a rank, so **any join between them needs the
   permutation stated** (`rank 0 -> gpu1`, `rank 1 -> gpu0`, `rank 2 -> gpu2`).
   C21 is what one such join cost: a 1040 MiB "unpriced tail" that was a
   1177 MiB price on a different card. Derive the mapping from the boot -- the
   per-rank MiB vector, the guard's own free readings, or the profiled token
   capacity -- never from the assumption that the two orders agree.
10. **Arithmetic on a value you hold cannot raise and cannot spend.** The same
   cut answered "margin short or law short?" with a second `ensure_headroom`
   call whose `except` returned "proceed", discarding a verdict that had
   already said the law would break. A second query where a subtraction would
   do is not a neutral choice.
11. **Placement is not residency, and a tier can be funded with the wrong
   currency** (C22). Item 16 says "redistribute onto the card with the most
   headroom", and five shifts read that as an ownership problem. Ownership is
   steerable and it buys nothing here: what a card commits is decided by the
   pool's backing watermark, not by which rank owns a row. Before building
   relief, name the quantity the DRIVER sees and check the mechanism moves
   THAT one.
12. **An identity bug shows up as a mechanism that is inert, not as a wrong
   answer** (HANDOFF_683 §1c). Two boots of #657 were spent on them: the bias
   sat on the wrong allocator class, then the rank index came from
   `ps.tp_rank`, which is 0 on every rank in the PP layout. Both were caught
   because the mechanism reports what it resolved and refuses to act on an
   unresolved identity. A mechanism that logs only its intent would have
   shipped twice, silently doing nothing.
13. **Name the LATENCY class of a relief mechanism before calling it relief**
   (C18/#658). The budget dial's answer to a lowered budget was a KV capacity
   cut that flushes the radix cache and commits only at a fully-idle group
   boundary. Both halves disqualify it as relief under load -- unbounded
   latency, and "a smaller pool as the fix" as the FIRST resort -- yet it
   reads like relief because it does return bytes. A mechanism that gives
   memory back EVENTUALLY cannot fund an allocation that is happening NOW,
   and the two must not be registered in the same ladder.
14. **A rank-local `except` around a collective is a deadlock waiting for the
   one rank that took a different branch** (#661, HANDOFF_684 §4b). Measured,
   not theorised: `init_chunked_prefill` wrapped `profile_and_init_predictor`
   in a per-rank `try/except`, and that function ends in a
   `pp_group.broadcast_object_list` every rank enters unconditionally. PP0's
   profile raised, PP0 disabled the feature FOR ITSELF and walked into the
   event loop, and PP1/PP2 blocked in the broadcast until the boot was
   killed. The corridor lane already respects this shape -- it is why
   `KvBackingRelief` is built but deliberately NOT registered as a guard
   provider (`phase_flip_spill.py:1468-1477`). The fix is always the same:
   publish the FAILURE as data through the collective every rank already
   enters, and let every rank decide on it together. **Sibling of law 12:**
   law 12 is an identity bug presenting as inertness; this one is a branch
   bug presenting as a hang, and both are cured by making the mechanism
   report a resolved, group-visible verdict rather than acting on a private
   one.
15. **A per-rank benchmark is not a rank-uniform quantity, and rounding does
   not make it one** (#659, HANDOFF_686 §1b). The #659 park tier shipped with
   bandwidth quantized onto a 0.25 GB/s grid and a docstring asserting "the
   quantum is coarser than the spread between ranks". The first live two-rank
   boot printed **2.41 GB/s on TP0 and 7.00 GB/s on TP1 for the same directory
   at the same instant** — a 2.9x spread, because the two probes contend with
   each other. No absolute grid survives that: any boundary between the two
   splits ONE medium across two buckets, after which two ranks order a
   two-tier ladder differently and park one session to two places, which the
   completion min-reduce turns into a hang rather than an error. **If a group
   decision must consume a measured value, consume a RATIO of two values
   measured on the SAME rank** (contention scales them together, so the ratio
   is stable where the absolute value is not), **or reduce it through a
   collective every rank enters.** Sibling of law 8 (the currency of a group
   decision must read identically on every rank) and law 14; distinct from
   both in that the offending quantity here is not an identity or a branch but
   a *number nobody doubted*. Note the shape of the near-miss: the falsified
   premise was stated confidently in the code's own docstring, and only the
   boot log — which existed because law 12 demanded the mechanism report what
   it RESOLVED — contradicted it.


16. **A relief mechanism triggered by a hardware failure cannot enforce a
   policy floor above that failure** (C27, #656). `_mem_create_reclaiming`
   held exactly the right remedy for exactly the right resource and was
   useless for the corridor, because its trigger was
   `CUDA_ERROR_OUT_OF_MEMORY` — free memory reaching ZERO — while the law it
   needed to protect sits 1024 MiB higher. The gap between the two triggers
   is not a tuning question; it is the entire region in which the policy
   exists. Measured: a cutover marched down in 24 MiB commit steps straight
   through the floor while torch held **1054 MiB of slack** the remedy would
   have released, and nothing asked for it because nothing had failed. **When
   a policy floor is declared, find every mechanism that could fund it and
   check what each one WAITS FOR.** A remedy that waits for the failure the
   policy exists to prevent is not a safety net, and it will pass every
   review because the code is correct — only its trigger is in the wrong
   place. Sibling of law 6 (hang relief on a clock that ticks inside the
   trough): both are mechanisms that are perfectly good except that they
   cannot arrive in time, and both were invisible until someone asked what
   the mechanism was waiting on rather than what it did.

17. **Clarification to law 15, from the fix for C27.** Law 15 forbids a
   per-rank measured number from producing DIVERGENT GROUP ACTION. It does
   not forbid a per-rank number from producing a per-rank OBJECTION that is
   then reduced: the seam entry check already computes `law_ok` from this
   rank's own `verdict.free_after` and rides a reduction where the group
   abandons if ANY rank objects, so feeding a per-rank measured draw into the
   same predicate adds no new divergence — every rank still observes one
   reduced verdict and acts on that. The test is not "is the input
   rank-local" but "can two ranks take different ACTIONS from it". Stated
   because the first draft of the C27 fix was nearly abandoned on a
   misreading of law 15 that would have left the breach unfixed.

18. **An instrument may not NAME the subject of its claim in advance when the
   system chooses that subject** (C28, #659, successor 44). The park-completion
   driver picked a TARGET request, drove pressure at it, and compared that
   request's output to a reference. On metal the server parked a DIFFERENT
   request, because the spill victim is elected by the victim ordering and not
   by which request a test author found interesting. The driver had already
   computed the attribution term (`target_named_in_park_records`) **and did not
   gate on it**, so its verdict would have read PROVEN over a run in which the
   measured session never parked and the parked session was never measured.
   The cure is not a better prediction: make the cohort HOMOGENEOUS (every
   member identical to the reference in prompt and sampling) and assign the
   arms AFTER the fact from the server's own per-rid records. **Then there is
   nothing left to guess, because every possible choice the system makes is
   already a valid measurement.** Sibling of the `proof_driver2` lesson (a
   verdict whose first conjunct must be "the mechanism ran"), one level in:
   there the instrument could pass without the mechanism running, here it could
   pass with the mechanism running *on something else*. Both are the same
   failure — a conjunct that was computed and not enforced.

19. **A state has more than one exit, and a fix applied to one exit is not
   applied to the state** (C28, #659). A parked/spilled session can leave
   through an ABORT or through a normal FINISH. `#552` found that the abort
   exit freed the session's memory without emitting its terminal chunk, fixed
   it with `_stream_terminal`, and left a comment saying in as many words
   *"Without it the abort frees the memory and the caller hangs."* The finish
   exit does the same freeing, was never given the same emit, and hangs the
   caller in exactly the way the comment describes. The bug was documented on
   its sibling for three shifts. **When a fix restores an obligation a cleanup
   path skipped, enumerate every path that performs that cleanup and check
   each one** — the obligation belongs to the STATE being left, not to the
   reason for leaving it. Sibling of law 12 (mechanisms that resolve nothing
   and report intent): here the mechanism reported, correctly and in detail,
   everything it released — `device head + tree lock + mamba + req slot +
   region` — and the log line's very completeness reads as diligence, which is
   why nobody noticed that the client's answer was not on the list.

20. **A cleanup that runs INSIDE an iteration must not mutate the collection
   the iteration's output depends on** (C28, #659). `filter_batch()` on
   `slot.batch` rebinds `reqs` to `[]`; the enclosing loop survives because it
   holds the old list, and the streamer 100 lines later reads the new one and
   sends nothing. The two names (`batch`, `slot.batch`) are one object, and
   nothing in either signature says so. **Membership of the
   Geteilte-Puffer-Familie** (a shared object plus an ordering assumption
   recorded only in a comment): the falsifier is cheap and should be written
   first — assert the streamer's input is non-empty for a request that
   finished. Note the diagnostic trap this creates: the symptom is a CLIENT
   hang with the server idle and healthy, which points every instinct at the
   transport, the scheduler or a deadlock, and none at a list that was
   correctly emptied 100 lines earlier.
