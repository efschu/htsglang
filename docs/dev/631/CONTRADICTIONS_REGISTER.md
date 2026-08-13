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

**C29 IS RESOLVED (successor 46, 2026-08-12, commit `948c53e6da`).**
`resolve_restore_margin_tokens()` sizes the margin against the pool at manager
init — deliberately AFTER the draft-scratch carve-out, which permanently
shrinks `allocator.size`, so the judgement is made against the pool the gate
will actually see rather than the one the boot started with. Who chose the
value decides the outcome, which is the part worth remembering: an EXPLICIT
unsatisfiable margin is REFUSED at startup naming the margin, the pool and
what the gate would have done (the same treatment
`mtp_resident_reservation_error` already gives an unsatisfiable scratch
reservation, for the same reason — both wedge silently); a margin left at the
SHIPPED DEFAULT is clamped to half the pool and logged at ERROR, because the
operator did not choose the default and refusing would turn a shipped constant
into a boot failure on every small-pool instance. `SGLANG_KVSO_RESTORE_MARGIN_
FORCE=1` honours the configured value verbatim, still reporting. The default
is READ from the `ServerArgs` dataclass rather than restated (the C18 rule): a
drifted copy would silently move a boot from the clamp branch to the refusal
branch. Inert on the ship config — `(512552, 4096)` resolves to 4096 with no
log. Red-first was EXECUTED by reverse-applying the patch, and the red is
behavioural rather than an import error: "the manager would run this
4096-token pool with margin 4096, at which no spilled session can EVER be
restored". The test's first half pins the MECHANISM and passes on both trees,
so it survives the fix as a regression guard.

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

**C30 IS RESOLVED, AND THE DIAGNOSIS ABOVE IS WRONG ON EVERY LOAD-BEARING
DETAIL (successor 46, 2026-08-12, commit `6a751b0adb`). Read this before
quoting anything from the C30 paragraphs above.** It is not an exclusion
between those two flags, and the boot-time refusal the entry called for —
one naming both flags — would have REJECTED A WORKING CONFIGURATION while
leaving the real trap armed everywhere else.

**The actual mechanism.** `PrefillAdder.add_one_req`'s chunked branch aligns
the chunk it is about to take and refuses outright when the whole chunk budget
is below one alignment unit:

    trunc_len = self.rem_chunk_tokens // self.page_size * self.page_size
    if truncation_align_size is not None:
        if trunc_len < truncation_align_size:
            return AddReqResult.OTHER

`rem_chunk_tokens` is bounded above by `--chunked-prefill-size`, so a budget
below the alignment refuses EVERY request longer than the budget, forever. The
admission loop `break`s on any non-CONTINUE verdict, so one such request at
the head of the FCFS queue blocks the queue behind it, `can_run_list` stays
empty and no batch is ever built. The wedged boot had `--chunked-prefill-size
256` against an alignment of 4096. **kv-session-offload's only role is that it
forces the flashinfer backend**; the refusing predicate is a THIRD variable,
and the same two flags at `--chunked-prefill-size 4096` serve normally.

**Two claims in the entry above do not survive re-reading its own evidence:**

* **there is NO rank divergence.** `probe_v8.log` prints the #583 collective
  census from both ranks throughout the wedge and the two are IDENTICAL
  (`all_reduce 1862x` on each). It froze because no forward ran, not because a
  rank was stuck in a collective. `num_queue_reqs 0` is a 30 s-cadence idle
  gauge sampled after the client had already disconnected. The py-spy dump
  does not show `schedule_policy.py:1255` at all.
* **the silently disabled radix cache is a red herring.** `ChunkCache`
  inherits `evictable_size() -> 0` (never None, never raising) and
  `_tree_evictable_size()` already tolerates it. Losing the radix cache costs
  restore headroom; it does not gate admission.

**THE ALIGNMENT HAS TWO INDEPENDENT SOURCES and either alone arms the trap**,
which is why the refusal cannot sit behind a deterministic-inference
condition: `--enable-deterministic-inference` on flashinfer/triton (align 4096
by default), and `--mamba-checkpoint-interval`, which sets the alignment on
its own with NO deterministic inference anywhere and is lcm-ed into it when
both are present. `truncation_align_admission_error()` is therefore called
from `init_deterministic_inference_config` AFTER the lcm — the only point
where the final alignment exists.

**Side-finding:** `--mamba-checkpoint-interval`'s own help recommended
"2048", which silently wedges any boot whose `--chunked-prefill-size` is below
2048 — including this rig's ship config at 512. Help corrected, guard
enforces it.

**Inert on the ship config, verified against the live process rather than
assumed** (`/proc/<pid>/cmdline`: `--chunked-prefill-size 512`, no
`--mamba-checkpoint-interval`, no `--enable-deterministic-inference`), and
then proven ON METAL: the ship config booted on commit `c78cc71442` with the
guard executing at every rank's scheduler init and did not refuse.

**Consequence for the byte-identity arm, corrected:** the door is NOT closed.
Deterministic inference and kv-session-offload CAN coexist — what the earlier
attempt lacked was a chunk budget of at least the alignment size. What would
prove byte-identity across a park, for whoever picks this up: boot the probe
with `--enable-deterministic-inference --attention-backend flashinfer
--enable-kv-session-offload` AND `--chunked-prefill-size 4096` (>= the 4096
alignment), then re-run `park_complete_proof2.py`'s cohort. With determinism
live, the CONTROL arm should stop diverging from the quiescent reference — and
once the control holds, a parked-arm mismatch becomes attributable instead of
"NOT ATTRIBUTABLE". Note the cost that made the earlier boot unattractive is
still real: flashinfer under deterministic mode disables the radix cache, so
restore headroom shrinks and C29's margin sizing matters more, not less.

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

**C33 — "THE PP GEOMETRY IS QUANTISED AT FOUR LAYERS" AND "THE RIG CANNOT BE
LEVELLED FROM THE PP SIDE" ARE BOTH FALSE (#485, 2026-08-12).** Desk-proven
on `feat/pp-family-cut-485`; the METAL consequence is unmeasured (no window
this shift), so what is overturned is the REASONING, not yet a boot.

`PROD_BRINGUP_BENCH.md:2430-2448` states the rule as physical — "The model
has 64 layers with one full-attention layer per 4, so a stage boundary can
only fall on a multiple of 4" — and `:2499-2504` draws the conclusion that
"at this layer count the surplus is UNREACHABLE with the layer knob… the
reachable splits are the multiples of 4 layers." Both the ~1.7 GiB overshoot
and the abandonment of the PP levelling lever rest on that.

**The rule is not physical. It is an artifact of deriving two targets from
one number.** `derive_pp_layer_split` (`distributed/utils.py:1481`) computes

```
target_full   = round(n_full   * cum/total)
target_layers = round(n_layers * cum/total)
```

from the SAME fraction, then clamps `target_layers` into the window that
yields `target_full` attention layers on the left. For a period-`P` hybrid
that window starts at `P * target_full`, so whenever the fraction sits near
a multiple of `1/n_full` the layer boundary lands on its bottom edge — a
multiple of 4. **The shipped code already violates the stated rule** where
the fraction does not: `derive_pp_layer_split([15,10,7], …)` returns
`[32,18,14]`, a boundary at 50. That single counter-example was available in
the code the whole time and is enough to falsify the "can only" claim without
any new machinery.

Give the attention target its own vector and the two families separate.
**Sixteen distinct layer splits hold the attention split at `[7,5,4]`**
(`[28,20,16]` through `[31,17,16]`) and sixteen more hold it at `[8,4,4]`
(`[32,16,16]` through `[35,16,13]`). Moving three linear/GDN layers off the
binding card is therefore reachable at **zero** KV cost — roughly 1.1 GiB of
weights against the ~1.3 GiB imbalance sec. 1g called unreachable, and
without the 8th attention layer that drove rank0 to 64 MiB free.

**Second correction, same ticket: the deep-prefill attention term is
COMPUTE-bound, not bandwidth-bound.** Its arithmetic intensity is
`2 * C * q_heads / (kv_heads * dtype_bytes)` — **depth cancels** — which is
24 576 FLOP/byte at a 2048-token chunk against ridge points of 151 (5090)
and 91 (3080). It crosses to bandwidth-bound only below a ~13-token chunk,
i.e. in decode. Any PP cut apportioning attention on the 2.14x
memory-bandwidth spread rather than the 3.54x GEMM spread is under-skewed.

**The law this produces (law 22).** *A lever that looks quantised may be
quantised by the SOLVER, not by the hardware.* Both claims above were read
off the reachable OUTPUTS of one function while treating its input space as
given. The register recorded the output pattern faithfully and promoted it
to a property of the model. Before declaring a knob's granularity physical,
find the line that computes it and ask what it would take to ask for the
value in between — and note the diagnostic tell: a "hardware" quantisation
that exactly equals the model's own period, with no counter-example sought,
is a solver artifact until proven otherwise.

---

**C34 — THE #485 CUT IS REAL, BIGGER THAN PREDICTED, AND LOCKED OUT BY THE
FLIP SEAM (#485/#631, successor 48, 2026-08-12).** C33 overturned the
REASONING on desk. This is the metal, and it splits into a win and a wall.

**The win.** Four-arm same-boot-floor A/B, depth 179200, pool 280000, flip and
spec removed identically from every arm (`server_args` requires that pairing
on PP, so it is not a free choice). Control = the ship cut `[28,20,16]`
attention `[7,5,4]`: **95.436 s**. Planner-optimal `[42,11,11]` attention
`[10,3,3]`: **63.246 s** — **+50.9 %**. Anti-proportional falsifier
`[16,24,24]` attention `[4,6,6]`: 119.271 s, **−20.0 %**, so the instrument
separates in BOTH directions. Floor: A-vs-A spread 0.12-0.97 %, and 0.09 %
across two SEPARATE boots at two different pools. The desk model predicted
+27.6 % and −50 %; it is a usable RANKER and a bad ESTIMATOR — direction right
on both arms, magnitude wrong on both, in opposite directions. Do not quote
its percentages.

**The wall, and it is the finding that matters.** Neither the planner arm nor
the falsifier can BOOT in the configuration this rig ships. With
`--enable-phase-flip` on, every cut that moves the attention split off the
ship's `[7,5,4]` starves the flip's seam staging and wedges the instance:
`corridor gate refused the seam staging: want 5393 MiB, free 5338 -> 5338 MiB,
reserve 1024 MiB, staging needs 4881 MiB` — **short by 55 MiB** — retried
without backoff **528 times**. The instance prints *"The server is fired up
and ready to roll!"* and then never answers `/health` again; the detokenizer
heartbeat stops at that same second. py-spy on all three schedulers and the
detokenizer shows every process IDLE in a normal wait, so this is NOT a
deadlock and will not be found by looking for one — it is an unbounded retry
loop that starves the heartbeat while every stack looks healthy. Arm A, the
swap arm and the restored ship boot ran 0 abandons against 6-72 completed
flips through the same code.

Second, distinct failure at a pool where staging DOES fit: arm C serves, then
dies on the first deep prefill inside `on_idle` with `pool memory leak
detected! [full] total=280000, available=267217, withheld=25566` — available
plus withheld over-counts the pool by 12783 tokens under the skewed token
vector. **Not attributed:** both failures are reached through
`SGLANG_UNEVEN_TOKEN_VECTOR`, set per arm to the arm's attention split. Cut
versus token vector is one boot's work and was not run; recorded as open
rather than guessed.

**Consequence for the ticket: GATE, do not wire — and the reason is new.** The
calibrated memory gate (C35) declared arm C FEASIBLE, predicting rank0 at
~29990 MiB of 32607 with 2617 MiB free, well over the corridor floor. It was
correct about RESIDENCY and useless in practice, because the flip needs
4881 MiB of TRANSIENT staging on that same rank and `planner/pp_cut.py` has no
term for it. A gate that certifies a cut which then wedges the instance is
worse than no gate: it launders an unrunnable configuration through a
calibrated-looking number. `RankResources` needs a `seam_staging_mib` term
before `solve_pp_cut` may reach a boot path — and that work is worth doing,
because it converts a measured +50.9 % from a locked prize into a reachable
one.

---

**C35 — THE UNEXPLAINED RESIDUAL FOLLOWS THE STAGE ROLE, NOT THE CARD; AND
THE RESIDUAL IS CUT-INVARIANT AFTER ALL (#485, successor 48, 2026-08-12).**
Resolves the item HANDOFF_485_PPCUT §1e handed forward, and refutes both the
hypothesis it named and the one the briefing named.

§1e measured residuals of 10171 / 4982 / 7582 MiB, noted that two identical
3080s differed by 2600 MiB, declined to pick a constant because the residual
"is not cut-invariant", and named a co-resident TP weight shard as the leading
suspect. Calibrated against a live at-rest ship boot, with every term named
instead of lumped:

```
fixed_overhead_mib = [4061, 3273, 4275]        # rank0, rank1, rank2
```

**It is cut-invariant.** Every term inside it is sized by the flip's TP vector
`32,16,16` = 0.5/0.25/0.25 — the mamba/GDN state pool (1229/614/614) and the
draft KV (614/307/307) — or is flat (draft weights 2048/1884/1884). Not one
of them is sized by the PP layer count; the mamba pool in particular does NOT
follow the per-stage GDN layer census (21/15/12 would give 0.44/0.31/0.25 and
does not fit the measurement). §1e's residual definition swept these
TP-shaped terms in and then read their non-PP-proportionality as evidence that
the CUT moved them. The suspect was wrong in kind, not just in size.

**And the 3080s are interchangeable.** The briefing asked for per-card
calibration as the default hypothesis and for an escalation if the gap
reproduced. It reproduces in direction (rank2 above rank1) and the hypothesis
is still wrong. Arm `swap` re-runs the arm A cut with `--rank-gpu-id 0,2,1`,
exchanging the two 3080s and changing nothing else:

| | rank1 | rank2 |
|---|---:|---:|
| baseline (rank1=smi0, rank2=smi2) | 1780 | 2592 |
| swap (rank1=smi2, rank2=smi0) | **1780** | **2464** |

The high residual moved from smi2 to smi0. It follows **rank2 — the last
pipeline stage** — and rank1 reads 1780 MiB on either card, to the megabyte.
The extra ~700-1300 MiB is the last stage's allocator high-water (graph pools,
sampler and logits buffers), not a card defect. A per-card table would have
encoded a hardware difference that does not exist, and would have been
confirmed forever by a rig that never swaps the cards. **The cheap
discriminator between "property of the card" and "property of the role" is
one boot with the mapping permuted, and it should be run before any per-card
constant is written down.**

---

**C36 — C34's "WALL" WAS TWO BUGS WEARING ONE ARM, AND THE CUT RUNS UNDER THE
FLIP AFTER ALL (#485/#631, successor 49, 2026-08-12).** C34 recorded that
"every cut that moves the attention split off `[7,5,4]` starves the flip's
seam staging". That is too strong, and the reason it was believable is a
design fault in the arm set, not a fault in the measurement.

N48 set `SGLANG_UNEVEN_TOKEN_VECTOR` per arm to that arm's attention split —
good physics, because the KV arena should follow the attention layers — which
means the cut and the arena moved TOGETHER in every arm. One boot separates
them: arm C's cut at pool 340000, flip on, everything identical except the
vector held at the ship `7,5,4`.

| | vector 10,3,3 (N48) | vector 7,5,4 (this shift) |
|---|---:|---:|
| flips completed | 3 | **6** |
| flips abandoned | 185 (555 lines) | **0** |
| reached /health 200 | never | **yes** |

**The seam wedge follows the ARENA, not the cut.** A desk derivation from
`_staging_bytes` predicted the opposite and was recorded before the boot; it
correctly predicts WHICH RANK refuses in both of N48's wedges (rank0 for arm
C, rank2 for arm D) and still names the wrong variable, because the row counts
in that formula are set by the arena. **A model can get the right answer for
the right mechanism and still be attributing the wrong cause, when the causes
were never separated.**

**The second bug, and it is a correctness bug.** `KvRowCap._apply` accumulates
its withheld ids and was wired as the allocator's on-CLEAR hook as well as its
on-free hook. `clear()` rebuilds `free_pages = arange(1, size+1)`, so the ids
above the cap are taken a SECOND time. Measured twice:

| boot | pool | vector | total−available | withheld | ratio |
|---|---:|---|---:|---:|---:|
| N48 arm C | 280000 | 10,3,3 | 12783 | 25566 | **2.000** |
| successor 49 | 340000 | 7,5,4 | 81640 | 163280 | **2.000** |

`withheld == 2 x (total − available)` exactly, across two pools and two token
vectors. **An exact integer ratio that survives both axes is a duplicate
booking, and that arithmetic is what found the defect** — the share and
rounding hypotheses were refuted by it before any code was read. The free list
stays correct, so only the published counter lies and the idle invariant
reports a leak on an intact pool. Worse half: `release()` cats the doubled
tensor back into `free_pages`, handing one KV row to two requests silently.
Only a configuration that ENGAGES the cap can reach it, which needs a corridor
deficit — so the ship cut never does, and doubling zero is invisible.

**The result.** With both fixed, the planner cut at pool 280000, flip ON:
**66.072 s at depth 179200 against the 98.276 s flip-on control, +48.7 %**,
42 flips, 0 abandoned, five deep prefills survived. Flip-off was +50.9 %, so
**the gain survives the flip world and the honest number is the smaller one.**

**Still GATED, and now for named numbers rather than a category.** The cut
does not hold the corridor: rank0 (the 5090, 10 of 16 attention layers)
bottoms at **976 MiB, 6 samples of 3568 under the 1024 law** at pool 280000,
and at pool 340000 it is 606 MiB and a `cuMemCreate` OOM. And the residency
model itself mispredicts that rank — it called arm C feasible with 2617 MiB
spare. Two calibration gaps, both quantified, neither a mystery.

---

**C37 — A BELT THAT HIDES THE DEFECT FROM ITS OWN TEST IS NOT A BELT
(#485, successor 49, 2026-08-12).** The `KvRowCap` fix above was written as
two changes: the semantic one (give `clear` its own hook that drops the stale
set) and a `torch.unique` "belt" in `_apply` to make a duplicate
unrepresentable. With the belt in place the three new regression tests passed
**whether or not the semantic fix existed** — the can-fail probe is what
revealed it, by reverting the hook and watching the suite stay green.

The belt was REMOVED, not kept. Two reasons, and the second is the general
one: it would have certified the wrong change as the fix, and it would have
silently absorbed the next path that books an id twice instead of failing on
it. **Run the can-fail probe against each candidate fix separately, not
against the patch as a whole** — a patch that contains a masking change and a
real change passes as a unit and teaches nothing about which half worked.
Sibling of the instrument-floor law: an instrument that cannot fail is not
evidence, and that applies to a fix's own tests as much as to a benchmark.

---

**C38 — THE RESIDENCY GATE MISPREDICTS rank0 BY ~3900 MiB, AND TWO OF ITS
ERRORS CANCEL ON THE SHIP CUT (#485, successor 49, 2026-08-12).** C36 left
"the residency model mispredicts rank0" as unexplained. It is now fully
accounted, from logs already on disk, and the account amends C35.

`_price_stage` prices a per-layer census of the TRANSFORMER LAYERS ONLY. The
checkpoint is `Qwen3_5ForConditionalGeneration` — a VL model, INT8-W8A8, with
`lm_head`, the entire visual tower and the embeddings in the quantizer's
`ignore` list, i.e. **bf16 and unpriced**. Fitting the six measured
`Load weight end` values (2 cuts x 3 ranks) closes to the megabyte:

```
embed_tokens = 2425 MiB   stage 0 only   (248320 vocab x 5120 x 2 B, untied)
lm_head      = 2425 MiB   last stage only
vision+loader ~ 1096 MiB  every rank, replicated bf16
```

~3760 MiB missing on rank0, essentially CONSTANT in the cut.

**Why nobody saw it: two errors cancel on the ship cut.** `tp_token_shares`
was fed the flip WEIGHT vector `32,16,16` (0.5/0.25/0.25) while the arena is
actually sized by `SGLANG_UNEVEN_TOKEN_VECTOR` `14,10,8` (0.4375/0.3125/0.25).
On the ship cut the arena term `max(7, 0.5*16)=8` OVERCHARGES KV by 547-664
MiB — almost exactly what the weight term undercharges — so the model looks
calibrated. On the planner cut `n_attn=10 > 8`, the cancellation vanishes and
the full shortfall appears. **A model validated only on the configuration
whose errors cancel is not validated.** Diagnostic tell: a gate that is
accurate on exactly one cut and wrong on every other.

**This partly refutes C35.** C35 concluded `fixed_overhead_mib` is
cut-invariant because "the mamba/GDN pool 1229/614/614 is exactly the TP
vector 0.5/0.25/0.25". There are TWO mamba pools. That figure is the TP-STACK
pool; the PP-STAGE pool is `51.2 MiB x n_linear(stage)` — genuinely PP-cut-
shaped, +563 MiB going ship -> planner on rank0 — and it was never itemized,
so it disappeared into the residual and made the residual LOOK invariant.
C35's card conclusion (the two 3080s are interchangeable, the residual follows
the stage ROLE) STANDS; its cut-invariance conclusion does not. Draft KV also
scales with the pool inside the same bucket (266 -> 328 MiB, 280k -> 340k).

The corrected ledger reproduces NVML free to **within 64 MiB on all three
boots** with a single 577 MiB CUDA-context constant, and gets every verdict
right: planner @340k infeasible by ~900 MiB (metal: OOM), planner @280k 321
MiB headroom (metal: 6 samples 48 MiB under the law), ship @280k 7633 MiB
(metal: 7212). **The planner cut at 340000 was below the corridor AT REST,
before a single token was served** — it was never a load problem.

---

**C39 — C38's TWO ERRORS DO NOT CANCEL; A FITTED RESIDUAL ABSORBED THEM
(#485, successor 50, 2026-08-12).** C38's two defects are real and are now
fixed. Its explanation of why three shifts missed them is wrong, and the
correct one is more dangerous.

The arithmetic, on rank0, ship cut, measured this shift:

| term | direction | @280000 | @620000 |
|---|---|---:|---:|
| unpriced non-layer weights (embed 2425 + vision 930) | under | **3358** | **3358** |
| arena on the flip WEIGHT vector, `max(7, 0.5*16)=8` | over | 547 | 1212 |
| net | under | **−2811** | **−2146** |

The over-charge is at most a third of the under-charge, so "cancel to within
~100 MiB" does not hold at any pool. What hid the error is that
`fixed_overhead_mib` is **a residual fitted on the ship boot**, and a residual
fitted on one cut absorbs every term that is constant on that cut. Measured
four ways on one axis over three boots (`s50/ledger.py`), worst held-out
error on rank0:

```
pre-C38 gate,  calibrated on ship boot   -> 3040.5 MiB  (both planner boots)
pre-C38 gate,  calibrated on planner boot->  109.2 MiB  (the other planner boot)
corrected,     calibrated on ship boot   -> 1266.4 MiB  (both planner boots)
corrected,     calibrated on planner boot->    6.8 MiB  (the other planner boot)
```

Read rows 1 and 2 together: the old gate was already good to ~109 MiB WITHIN
a cut family. **The failure was never "inaccurate", it was "accurate only
where it was fitted".** Diagnostic tell is therefore not "accurate on exactly
one config" (C38's version, which invites you to look for a cancelling pair
that may not exist) but "the model contains a residual, and the residual was
fitted on the config you are trusting it on".

**Still open, and now quantified:** the residual is NOT cut-invariant. rank0
measures **3174 MiB at 28 layers, 4419 at 40, 4440 at 42** — flat across
40 vs 42, 1250 MiB apart across 28 vs 40. It sits in the census's
graphs/workspaces post. Until that is itemized, the gate must be calibrated
in the neighbourhood of the cut it judges, and a cross-family verdict carries
a ~1250 MiB error bar.

**Confirmed by direct measurement, not by fit** (`planner/residency_census.py`,
two cuts, both card types): `embed_tokens` 2425.0 on stage 0 and `lm_head`
2425.0 on the last stage exactly as C38 predicted; the vision tower
replicated at 927.8/927.8/932.8; the PP-stage recurrent pool **51.2 MiB x
n_linear** (C38's figure, against my own mid-shift fit of 158 MiB, which was
wrong — see law 28); and the arena following the TOKEN vector, visible in the
ship boot's own allocation `#tokens 271264/193760/155008` = `14,10,8`.

**One more unpriced term C38 did not name:** the attention layer is 364.88 MiB
resident against the 325.0 MiB the config formula gives, because
`attn_output_gate` adds a second q-sized projection. 482 MiB across 16
layers, and cut-shaped.

---

**C40 — A RANK DIED SILENTLY UNDER THE FLIP WITH 1926 MiB FREE IN HAND
(#485/#622/#649, successor 50, 2026-08-12). CLOSED by C41 — IT WAS NOT
SILENT AND IT WAS NOT A GPU EVENT. Read C41 first; the account below is
preserved because the way it went wrong is the lesson.** During the planner-cut
window, rank0 stopped at 11:48:41 in the middle of an ordinary decode batch:

* no traceback on rank0, and its last log line is a normal `Decode batch`;
* **no OOM anywhere** — zero `out of memory` / `cuMemCreate` /
  `CUDA_ERROR_OUT_OF_MEMORY` lines in the whole boot;
* no kernel OOM-killer record, host RAM at 107 GB available;
* rank0's free VRAM over the preceding 20 s ranged **1926-7726 MiB**, i.e.
  the corridor law was comfortably held at the moment it died;
* ranks 1 and 2 then raised `Bar1CollectiveAborted ... group flip_tp:0`
  via barlink's peer-liveness abort, naming `peer rank gone: rank 0 (pid
  986533)`.

**The barlink abort is the CONSEQUENCE and must not be read as the cause** —
it is the mechanism working correctly, reporting a peer that vanished.
Equally, this is **not** the corridor event that happened in the same window:
that was at 11:44:06, four and a half minutes earlier, and the rank survived
it. Two findings, one window, and conflating them would send the next shift
hunting memory when the memory was there.

This is the #622/#649 silent-wedge family seen under the phase flip on a
planner cut. It is the reason no planner-family cut can be certified on a
20-minute window yet: the window is the instrument that would have to catch
it, and here the window is what it killed.

---

**C41 — C40's "SILENT DEATH" WAS A SIGKILL THE LAUNCHER HAD ALREADY LOGGED,
AND THE LEDGER WAS THE HOST'S (#485/#622/#649, successor 51, 2026-08-12).
RESOLVED.** Line 49494 of the same boot log C40 was written from:

```
[2026-08-12 11:48:55] Subprocess scheduler_0 (pid=986533) crashed with exit
code -9. Triggering SIGQUIT for cleanup...
```

`-9` is `-SIGKILL`. Every clause of C40's evidence list survives except the
one it turned on: there was no traceback (a SIGKILLed process cannot write
one), there was no OOM line (this codebase does not emit one for a kill it
did not perform), the GPU corridor was held (irrelevant -- the kill came from
the host), and the barlink abort was indeed the consequence. What was NOT
true is that nothing recorded how the process ended.

Two reasons it read as absent, and both are now fixed in code:

1. **It was a bare integer.** Nothing decoded `-9` to `SIGKILL`, so the line
   reads as noise. `utils/watchdog.py` now names the signal.
2. **Nothing named a mechanism.** SIGKILL is never sent to a healthy rank by
   this tree. On this rig the sender is the kernel's cgroup OOM killer, whose
   report goes to a ring buffer **a container cannot read**: in CT999 `dmesg`
   is `Operation not permitted`, `/dev/kmsg` does not exist, `journalctl -k`
   is empty. The only in-container trace is `memory.events`'s cumulative
   `oom_kill` counter, which carries neither timestamp nor victim and is
   therefore evidence ONLY against a baseline taken earlier. The watchdog now
   samples one at construction and reports the delta at any SIGKILL --
   including when it is zero, so the instrument can exonerate the OOM killer
   as well as convict it.

**The host ledger, measured this shift.** The container ceiling is ~120 GiB.
With serving up on the ship config the three schedulers hold **75.1 GiB of
unreclaimable shmem** -- their unlinked `/dev/shm/sglang_loads_*.shm` weight
mappings, 33.1 + 17.2 + 25.1 GiB, summing to the cgroup's shmem post exactly
-- and swap is zero. So the real working margin is ~45 GiB, not 120, and the
cgroup carries 9 cumulative OOM kills. It is not a cross-boot leak: stopping
serving returned shmem 75.1 -> 0.0 GiB and MemAvailable 31 -> 118.9 GiB. The
planner cut costs more of this budget than the ship config does: measured on
identical soaks, MemAvailable bottomed at 23.5 GiB on the ship config and at
15.9 GiB on the 40,12,12 arm.

**The general lesson (law 33).** C40 rested on three searches coming back
empty. Two of them could never have come back full. A negative search result
is evidence only once the instrument has been shown able to produce a
positive one.


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

### C31 — the KV spill round trip breaks byte-identity, and it took a working instrument three shifts to say so (#659, successor 47)

**OPEN. Measured, reproduced, attributed; mechanism NOT yet identified.**

The headline is not the defect. It is that **three shifts in a row concluded
"not obtainable" / "separates nothing" from an instrument that could not have
detected the defect if it were there**, and the defect was there the whole
time.

**What was measured**, probe `v10` on port 30042 (`--enable-deterministic-inference
--attention-backend flashinfer --enable-kv-session-offload
--chunked-prefill-size 4096`, boot commit `03b6fb990d`, evidence
`/spinning/evidence-631/s47/`):

| run | pressure | spilled | result |
|---|---|---|---|
| floor x3 | quiescent, sequential | none | all 3 identical, `89f0c7e305c1ceab` |
| `s47c` | 4 concurrent, NO fill | **none** | **all 4 identical**, cohort-0 included |
| `s47` (A) | 4 concurrent + 3 fill | `cohort-0` only | 1/2/3 identical; **cohort-0 `b25a8fc15649a2dc`, 1673 ch** |
| `s47b` (B) | 4 concurrent + 3 fill | `cohort-0` only | 1/2/3 identical; **cohort-0 `81e02f5e659d47d8`, 1706 ch** |
| `s47f` (F) | 8 concurrent + 5 fill | `cohort-0` only | **7 of 8 identical**; **cohort-0 `eb0c3615126dfce2`, 1639 ch** |

Seventeen never-spilled generations of one prompt are byte-identical across
five independent runs, including a run at double the cohort size. The three
generations that spilled are **each** divergent from the reference **and from
each other** (1673 / 1706 / 1639 chars, three distinct digests) — so the spill
path does not shift the output by some fixed amount, it **reintroduces
nondeterminism into a boot whose entire purpose was to remove it**.

**The attribution is closed by the `s47c` falsifier, and that run is the point
of the entry.** `cohort-0` is also the first-submitted request, so "position 0
is special" was a live alternative explanation for A and B. Removing only the
fill pressure — same cohort, same concurrency, same prompt, same sampling —
made `cohort-0` spill-free and byte-identical. Position, concurrency and batch
composition are excluded **by measurement, not by argument**.

The spilled session's own records (`probe_v10.log:380-459`) name what it went
through: `SPILL(partial) rid=s47-cohort-0 L=3073 device_head=0 host_tail=3073`
(the entire context left the device), `first spill tick`, `WAVE-BACK boundary
0->1083 tail_left=2028`, then `finished on host`. It decoded **from host RAM**,
which the boot warned about in advance at `probe_v10.log:166`: *"no
token-sharded DCP active -- the spilled session streams its WHOLE context over
a single PCIe link every decode step"*.

**Why every earlier shift missed it, which is the transferable part.** The
defect needs TWO conditions to be visible and no previous run had both:
determinism live (or every session differs and nothing is attributable — s45
measured exactly that, all four cohort digests distinct, verdict "this run
separates nothing"), AND a spill actually occurring. HANDOFF_689 then
concluded byte-identity was "not obtainable on this rig until deterministic
inference and kv-session-offload can coexist" and told the next shift not to
spend time on flag variations. They coexist (C30). **A null result from an
instrument with no demonstrated floor is not evidence of absence, and this
register now has three shifts of it.** The cheap thing that would have caught
it at any point is the one this shift ran first: three quiescent repeats of
the reference prompt, to show the instrument can register "identical" at all
before asking it to judge "different".

**A SECOND instrument defect, found by the same run and still unfixed.**
`park_complete_proof2.py` exited **3 ("NOTHING PARKED")** on both A and B —
i.e. the run that contains the finding is reported by the driver as a null.
Its arm assignment keys exclusively on file-tier `PARK commit rid=` records
(`PARK_RE`, `:166`), so a session that spilled to the **host** tier and waved
back is not merely missed — it is filed into the **control** arm, where it
sets `control_arm_identical_to_reference = False` and makes a genuinely clean
control look contaminated. The verdict table is therefore wrong in both
directions at once. This is **law 19's shape applied to attribution**: the
state is "this session's KV left the device and came back", and that state has
at least two exits (host tier, file tier) while the instrument was written for
one.

**FIXED this shift** as `park_complete_proof3.py`: arms are assigned from the
UNION of both exits and the tier is reported per rid. Validated by replay
against the recorded `probe_v10.log` rather than by inspection -- on run A's
own digests the corrected logic returns **EXIT 1 DEFECT** ("the parked arm
differs from the reference while the control arm, under the same pressure,
matches it") where v2 returned exit 3 ("nothing parked, no claim can be
made"). The driver reaches C31's conclusion on its own once the arms are
right. The live path is still unexecuted.

**THE MECHANISM, localized from source and verified line by line (successor 47,
same shift).** It is not the round trip. It is that **`--enable-deterministic-
inference` never reaches the spill path at all**, and that a spilled session
does not run the same attention as a resident one.

* **`kv_session_offload.py` contains ZERO references to
  `enable_deterministic_inference`, `fixed_split_size` or `split_tile`**
  (verified: `grep -c` returns 0). Its five "deterministic" mentions are about
  cross-rank agreement, an unrelated property. **The feature was never wired
  for determinism.**
* **The resident path pins the split tile and the spill path drops it.**
  Resident decode passes `fixed_split_size=self.decode_split_tile_size`
  (`flashinfer_backend.py:1668`); the spill path's `w.plan(...)` calls
  (`_sess_plan_block:3731`, `_sess_plan_dev_head:3762`) pass neither
  `fixed_split_size` nor `disable_split_kv` — 0 occurrences in `:3720-3790` —
  so each partial falls back to flashinfer's occupancy heuristic split-k,
  which is exactly what the determinism flag exists to pin.
* **A spilled session's attention is a CHAIN of partial decodes merged in
  software.** `_sess_blockwise_decode_return_lse` (`:3776`) runs one partial
  over the device head plus one per streamed host block and folds them with
  `o_acc, lse_acc = _safe_merge_state(o_acc, lse_acc, o_b, lse_b)` (`:3873`) —
  a **sequential left-fold LSE merge, non-associative in floating point**. A
  different reduction tree from the resident monolithic decode, on
  byte-identical KV.
* **The fold's shape is chosen by a TIMING QUERY, which is what makes the
  three spilled runs differ from EACH OTHER.** The wave-back boundary sets
  both the device-head size and the block count
  (`flashinfer_backend.py:3505,3529`), and it advances only when
  `WaveBackController.plan` allows, whose input is
  `copy_inflight = not self.backend._sess_wave_done.query()`
  (`kv_session_offload.py:4983`) — a **CUDA event progress probe** — plus the
  live allocator free list (`:4918`). Neither is a function of the token
  sequence. PCIe contention and host jitter therefore change the partition,
  the fold order, and the rounding, run to run.
* **The path was never claimed to be bit-exact.** Its own self-test passes at
  `ok = (not nan) and rel < 5e-2` (`:4282`), and the source says so directly
  (`:4140-4142`): *"the ONLY gap T-vs-R is bounded blockwise fp
  reassociation (decode-class)"*. Bounded reassociation is precisely what
  byte-identity forbids.

**Ruled out from source, so nobody re-treads them:** the fp8 round trip (the
host pool inherits `store_dtype`, `uint8` for fp8, and the movers are indexed
byte memcpy — `pool_host/base.py:136`, `memory_pool.py:2012`); recompute (the
target is never re-forwarded, and the draft backfill is gated off at
`spec=False`); and mamba/GDN state (never spilled — the "+ mamba" in the
release line is the finish path freeing a slot).

**Cheapest confirming experiment, no code changes**, both existing CLI knobs:
set `--kv-session-offload-wave-back-min-free-tokens` above the pool size to
freeze the boundary at 0 for the whole episode. If the spilled outputs become
identical **to each other** while still differing from the reference, the
timing-gated boundary is confirmed as the run-to-run variable and the residual
delta is the blockwise decomposition. If they still all differ, the unpinned
split-k is also live. Then additionally raise
`--kv-session-offload-block-size` above the context so the chain collapses to
one partial with no merge.

**What this means for the feature, stated plainly:** kv-session-offload and
deterministic inference are not merely unwired — they are **semantically
incompatible as currently built**, because the spill path's whole design is a
runtime-adaptive decomposition. Making them compose is not a flag fix; it
requires pinning the boundary and the split-k, at a performance cost nobody
has priced. Whether that is worth doing is a product decision, not a bug fix.

**THE REFUSAL THIS PRODUCES, AND THE ONE IT MUST NOT PRODUCE.**

> **REFUSED (guarantee):** a session that has been through a
> kv-session-offload spill is **excluded from any determinism guarantee** this
> engine issues. The **#412 determinism-certificate mode must NAME this
> exclusion** in the certificate itself, not in a footnote — recorded on the
> #412 row of `ROADMAP_456_matrix_execution.md` **before** the mode is built,
> so it cannot ship a claim it cannot honour.
>
> **NOT REFUSED (boot):** the flag pair itself. `--enable-deterministic-
> inference` + `--enable-kv-session-offload` boot and serve correctly together
> at a sufficient chunk budget, and most sessions never spill.

**That split is the entire point, and getting it backwards is a documented
failure mode on this very line.** C30 records a shift that was told to "wire a
LOUD boot-time refusal naming both flags" and would have rejected a working
configuration while leaving the real trap armed. The correct object of refusal
here is **the claim, not the configuration**: a certificate asserting
"deterministic" over a boot where a session may silently spill is false, and
falsity is what gets refused. A boot-time exclusion would destroy a useful
config to protect a guarantee that nothing has yet issued.

The operational consequence for anyone running determinism today: **spilling is
silent from the client's side.** There is no per-response marker saying "this
answer came back through the host tier", so a caller cannot currently tell a
guaranteed answer from an excluded one. If #412 is built, the certificate needs
that per-response signal, and the `PARK/SPILL rid=` records already carry
exactly the information it would need.

The FILE-tier park arm is separately
unproven — the host tier is sized to hold **one** full-context region (proved
by the refusal at `--kv-session-offload-host-ram-gib 0.12`: *"cannot hold even
ONE full-context session ... 3933 tokens < 32770"*), so a single spill fills
it and a second concurrent spiller is required to reach the file tier at all.

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

21. **An instrument that has never been shown to register "identical" cannot
   be believed when it reports "different" — and its NULL results are worth
   nothing at all** (C31, #659, successor 47). Three consecutive shifts drew a
   conclusion from the byte-identity cohort driver while it had no
   demonstrated floor: s45 measured four distinct digests and booked "this run
   separates nothing"; HANDOFF_689 escalated that to "byte-identity is not
   obtainable on this rig" and told the next shift to stop trying flag
   variations. The defect was present in every one of those runs. The check
   that broke it open cost 55 seconds: **run the reference prompt three times
   quiescent and require the digests to match BEFORE running the comparison
   the instrument exists for.** Once the floor held, the same driver on the
   same rig separated a real defect on the first attempt. Sibling of law 18
   (an instrument may not name the subject of its claim in advance) and of the
   `proof_driver2` lesson (a verdict whose first conjunct must be "the
   mechanism ran"), and the same shape as both: a conjunct nobody enforced.
   The distinguishing feature here is the DIRECTION of the error — 18 and the
   proof_driver2 case are instruments that could pass wrongly, this is an
   instrument that FAILED wrongly, three times, and each failure was recorded
   as a fact about the SYSTEM rather than a fact about the instrument. A
   "cannot be done" that rests on a null is a claim about your measurement
   until the measurement has a floor.

---

### C32 — the #330 dial's first metal boot: the join is real, the ladder is empty, and one card's budget moves the WHOLE ceiling (#330/#656, successor 47)

**Two sub-findings RESOLVED (they were false premises), one NEW risk OPEN.**

This was the **first `--enable-vram-dial` boot in this project's history** — no
file under `/spinning/evidence-631/` had ever contained the string `VRAM-DIAL`
before this shift. Boot script `/spinning/evidence-631/s47/boot_dial_tp3.sh`,
evidence `dial_metal_result.json`, `dial_metal_proof.log`.

**What was proven on metal** (TP=3, weighted uneven DCP, vector `[30, 17, 17]`):

| axis | measurement |
|---|---|
| dial-down under load, HTTP latency | **2.0 ms** — arming is synchronous |
| ladder asked before capacity arithmetic | **YES**, one line, at the reduction |
| capacity committed | `VRAM-DIAL DONE SHRINK ... 327760 -> 69824` |
| pages returned to the DRIVER | rank 0 `released 4000.0 MiB`, ranks 1/2 `128.0 MiB` each |
| NVML cross-check (out of band) | target card free **8132 -> 11584 MiB** |
| raise restores capacity | C `69824 -> 327744`, real generation OK after it |
| corridor | **202 samples, 0 breaches**; min free 3485 / 7584 / 4239 vs law 1024 |

**FALSE PREMISE 1, corrected: "the dial refuses PP=3 boots."** There is no
`pp_size` check anywhere in `vram_dial.py`. The refusal is
`if not uneven_dcp_active(dcp_size)` (`vram_dial.py:1291`), and DCP
auto-engages only as `dcp_size = tp_size` (`server_args.py:10702`). The ship
config is PP=3 with **TP=1 per stage**, so `tp_size=1 -> dcp_size=1 ->
uneven_dcp_active(1) is False`. **The dial refuses the ship config because
TP=1 leaves no DCP axis, not because PP is PP.** The distinction matters for
the composition question: a PP boot with TP>=2 per stage would pass this
check, and nothing else in the dial refuses PP.

**FALSE PREMISE 2, corrected: "prove sessions return to CUDA graphs after a
budget raise."** They never leave. The dial commits physical pages behind a
**stable VA reservation** — *"No tensor moves, no CUDA-graph re-capture"*
(`vram_dial.py:23-25`, `rig-runbook.md:882-883`). No such log line exists and
none can. Any future acceptance criterion phrased that way is unmeasurable and
should be rewritten to the converse actually checked here: decode keeps
reporting `cuda graph: True` across the dial and the post-raise generation
succeeds.

**Also corrected: the C18 join runs the other way.** The brief for this shift
said "the guard calls the dial". Source and metal both say the **dial calls
the guard**: `apply_budget_request` -> `_relieve_for_reduction`
(`vram_dial.py:598-599`) -> `_corridor_relief._relief` (`:1106-1108`) ->
`CorridorGuard.ensure_headroom`. `HANDOFF_684:4-5` already said so: *"the dial
cannot be the guard's actuator, so the guard became the dial's."*

**THE LADDER IS ORDERED CORRECTLY AND YIELDS NOTHING.** The one line it
emitted is the whole story:

    VRAM-DIAL budget reduction of 4096 MiB on rank 0: the corridor relief
    ladder returned 0 MiB now; the residual is funded by the capacity
    arithmetic at the next group-idle boundary.

Both providers registered (`CORRIDOR-GUARD registered provider
'allocator-cache' in tier local at cost 10`, then `'draft-weights' in tier
rebalance at cost 20`), so the guard was built and the spend order was right —
and **the entire 4096 MiB still came from the capacity arithmetic.**
`draft-weights` returns 0 unless `phase_flip_active_stack == PHASE_PP`
(`phase_flip_spill.py:1344-1346`), and `allocator-cache` found nothing to
give under load. **No `RELIEF_PARK` or `RELIEF_HOST` provider is registered
anywhere in the tree.** So "the ladder spends before the capacity arithmetic"
is proven as an ORDERING and unproven as an EFFECT; on any boot without the
phase flip in a PP phase the ladder is decorative. Do not quote the dial's own
docstring here (`vram_dial.py:626-631`) — it claims the ladder runs
"local -> park -> host", and two of those three tiers have no provider at all.

**NEW RISK, OPEN — the min-reduce makes a per-card dial a GLOBAL lever.**
Cutting **one** rank's budget by 4096 MiB collapsed the global token ceiling
from **327760 to 69824 — 79% of the KV capacity for 12% of one card's
budget.** The cause is the uneven-DCP sizing path the dial requires: global
`max_total_num_tokens` is a **min-reduce** over per-rank (capacity / ratio)
units, so the dialed rank becomes the binding constraint for every rank.
Rank 0 gave up 4000 MiB; ranks 1 and 2 were forced to release 128 MiB each and
lost the same 79% of ceiling **without being dialed at all**. Nothing is
broken — the arithmetic is correct and the raise restored it — but any
operator model of "dial one card to lend a co-tenant some VRAM" is wrong by a
factor of six here, and a scheduler admitting against the old ceiling during
the gap is the obvious next thing to check. **This is the composition question
for #330 x uneven-DCP and it is not in DESIGN_330.**

**Still unproven, with its exact precondition:** the dial x phase-flip
composition. The ship config cannot host the dial (TP=1 per stage, above), so
the C23 two-actuator race (`KvBackingRelief` MAX-reduction at the flip seam vs
`KvCapacityRuntime` MIN-reduction at the idle boundary, both driving
`pool.runtime_set_backing_rows`) remains unobserved. Its precondition is a
boot with **TP>=2 per PP stage** plus `--enable-phase-flip` plus
`--enable-vram-dial`, which this rig's three cards can host only as PP=1.

22. **A lever that looks quantised may be quantised by the SOLVER, not by
   the hardware** (C33, #485). The PP layer knob was recorded as steppable
   only in multiples of 4 because every split anyone tried came back that
   way — and the cause was that `derive_pp_layer_split` derives its layer
   target and its attention target from ONE fraction, so the layer boundary
   lands on the bottom edge of the attention snap window. The shipped code
   already produced a counter-example (`[15,10,7]` -> `[32,18,14]`, boundary
   50) which nobody looked for. Two conclusions were built on the artifact:
   a ~1.7 GiB "overshoot" and the abandonment of PP-side levelling. **Before
   calling a granularity physical, find the line that computes it and ask
   what it would take to request the value in between.** Diagnostic tell: a
   "hardware" quantisation that exactly equals the model's own period, with
   no counter-example sought, is a solver artifact until proven otherwise.
   Sibling of law 12 (mechanisms that resolve nothing and report intent) —
   here the mechanism reported its reachable outputs perfectly, and their
   very regularity is what read as a physical law.

23. **A residency gate cannot certify runnability: "fits at rest" and "can
   run" are different predicates** (C34, #485). The calibrated memory gate
   priced weights + KV + a measured fixed overhead, declared the planner cut
   feasible with 2617 MiB to spare, and was correct — the configuration does
   fit at rest. It wedged anyway, because the phase flip needs 4881 MiB of
   TRANSIENT staging on that rank at cutover and the model had no term for
   transient demand at all. The gate was not wrong about what it modelled; it
   was wrong about what it was being ASKED. Before a feasibility check is
   allowed to gate a boot, enumerate every mechanism that needs headroom the
   check does not model — peak is not residency, and the mechanisms that need
   peak (seam staging, graph capture, spill arenas) are exactly the ones whose
   demand does not appear in any at-rest measurement. Sibling of law 13 (name
   the LATENCY class of a relief mechanism): same error one axis over — there
   the missing dimension was time, here it is transience.

24. **A setting that is DERIVED from the variable under test is a second
   variable** (C36, #485). The four-arm cut A/B set
   `SGLANG_UNEVEN_TOKEN_VECTOR` per arm to that arm's attention split,
   because the KV arena should follow the attention layers. The physics is
   right and the experiment had two variables in it, so one shift attributed
   two independent failures to a single "wall" and wrote it into the register
   as a property of the cut. One boot holding the derived setting at its ship
   value separated them, and the answer was the opposite of the desk
   prediction on the failure the prediction was aimed at. **Before an arm
   set runs, list every setting that MOVES WITH the variable and ask what a
   result would mean if the rider carried it.** Diagnostic tell: an arm
   differs from the control in "one thing" that is described in two clauses
   joined by "and therefore".

25. **Consumed law 23's other half: the transient term is in the model now,
   and an uncalibrated term is still a zero** (C36, #485). `RankResources`
   gained `seam_staging_mib` and the solver maximizes the tightest RUNNABLE
   headroom, so "fits at rest" and "can run" are finally different
   predicates in code. The field defaults to 0, which means the gate is
   unchanged until someone measures the value — and a gate modelling
   transient demand as zero is exactly the gate that certified the cut that
   wedged. **Adding the TERM is not the same as closing the hole; the hole
   closes when the number is measured.** Two measured demands exist and do
   not separate the term's shape, and no formula was fitted to them,
   because the mechanism a formula would encode was refuted the same week.

26. **A model validated only where its errors cancel is not validated**
   (C38, #485). The residency gate under-priced rank0's weights by ~3760 MiB
   (unquantized embeddings, lm_head and vision tower, all outside the
   per-layer census) and over-priced its KV arena by 547-664 MiB (fed the
   flip WEIGHT vector where the arena uses the TOKEN vector). On the ship
   cut those two cancel to within ~100 MiB and the gate reads calibrated.
   On any cut where the attention count exceeds the token share the
   cancellation disappears and the gate is wrong by ~3900 MiB. **Check a
   model on a configuration it was NOT tuned on before trusting it, and
   when a model is accurate on exactly one config, suspect cancellation
   rather than calibration.** Sibling of law 23: there the gate modelled the
   wrong PREDICATE, here it models the right one with two compensating
   errors.

27. **Two minima from two differently-loaded boots do not make a derivative**
   (successor 49's own retraction, C7). I took rank0's load MINIMUM at two
   pools, divided, got a slope 3.2x shallower than the layer-count model,
   published it as "the measurement beats the theory" and aimed the next
   shift's boot at the wrong pool. The at-rest ledger gives 0.021 MiB/token,
   i.e. the layer-count figure was right all along; the load minima differ
   by load state, which is exactly what C7 says they read. **Before
   subtracting two measurements, ask whether they are the same KIND of
   quantity** — at-rest and under-load are different instruments, and a
   ratio between them is not a slope.

28. **A fitted residual makes any model look calibrated on the config it was
   fitted on** (C39, #485). Not "check a model where its errors cancel" —
   check whether the model has a free parameter absorbing them. The #485 gate
   carried one (`fixed_overhead_mib`, defined as
   `resident_at_rest − weights − kv`), and with it the pre-fix gate predicted
   a held-out boot of the SAME cut family to 109 MiB while being wrong by
   3040 MiB the moment the cut moved. **Report a model's held-out error
   across the axis it will be USED on** — for a cut gate that axis is the
   cut, so a residual calibrated on one cut is evidence about that cut only.
   Corollary: name the free parameters in any ledger, and state which boot
   each was fitted on.

29. **A measurement refutes a fit even when the fit predicts** (successor
   50's own retraction). I fitted 158 MiB/linear-layer from two boots; it
   predicted a third to 88 MiB, and it was still the wrong mechanism — the
   census measured 51.2 MiB/linear-layer directly, on two cuts and both card
   types. Predictive success over a narrow range is not mechanism. **When a
   term can be measured directly, measuring it is not optional because the
   fit happens to work** (sibling of law 28: the fit worked because another
   free parameter moved to accommodate it).

30. **An instrument that must not perturb what it measures has to be
   default-off AND read-only** (#485 residency census). The component balance
   this shift needed already existed in
   `note_post_capture_leftover`, but it is gated on
   `SGLANG_MEASURED_KV_BUDGET`, which also PERSISTS a budget correction and
   changes the next boot's pool. Reusing it would have made every corridor
   window it rode along on worthless. The new census reuses the same
   allocator checkpoints and writes nothing. **Before reusing an existing
   probe, check what else its gate turns on.**

31. **A transient is a property of the LOAD STATE, and "the load" is not one
   state** (successor 50, law 27 repeated). Law 27 says two minima from
   differently-loaded boots are not a derivative. Its corollary, learned the
   expensive way one shift later: a single measured DRAW is not transferable
   either. Measured on rank0 of the reference rig, same rank, same law:

   ```
   deep-prefill A/B load   ->  956 MiB drawn below at-rest
   22-min mixed soak       -> 1989 MiB (planner cut) / 3148 MiB (ship config)
   ```

   I aimed two boots with the 956, and **the refuting number was already on
   my own disk** — I had measured the 3148 myself, four hours earlier, in
   this shift's ship window. **Before reusing a transient, name the load
   state that produced it and check it against every load state you have
   already measured.** The same applies to `RankResources.transient_mib`,
   which is fed a prefill-trigger figure and is therefore optimistic by
   ~650 MiB under the shipping soak.

32. **Execute the path before believing the reading** (successor 50,
   `--pp-solve-cut`). My first card-rate lookup used `--rank-gpu-id` as an
   NVML index. It reads correctly and it is wrong: on this rig
   `--rank-gpu-id 0,1,2` puts stage 0 on NVML index 1, because
   `CUDA_VISIBLE_DEVICES` is set by UUID and torch's device order is not
   NVML's. It priced the 5090 as a 3080 and no unit test would have caught
   it, because the mapping is a property of the rig. Running the handler once
   against a real census found it in seconds. **A per-card join must carry
   its IdentityMap in the artifact** — each rank now records its own device
   in the census — **and desk-written code gets executed before it is
   trusted** (the desk-written-never-executed law, applied to a refusal path
   nobody expected to reach).


33. **A search that comes back empty is evidence only if it could have come
   back full** (successor 51, C41). C40 concluded "silent death" from three
   absent strings: no traceback, no OOM line, no kernel record. A SIGKILLed
   process cannot write a traceback; this tree does not log an OOM it did not
   perform; and the kernel's OOM report goes to a ring buffer that an LXC
   container cannot read at all (`dmesg` denied, no `/dev/kmsg`, empty
   `journalctl -k`). Two of the three searches were incapable of a positive
   result on this rig, and the one line that WAS present -- `crashed with
   exit code -9` -- was passed over because nothing decoded it. **Before
   reporting an absence, name the instrument that would have shown the
   presence, and show it works here.** Sibling of the can-fail proof required
   of a gate, applied to forensics.

34. **Price the term on the path that BOOTS, not the path you reason with**
   (successor 51, #485). The published cut-gate verdicts came from a desk
   script that hand-fed a per-rank transient. The wired `--pp-solve-cut`
   handler never set the field at all, so it took its `0.0` default and
   priced a demand measured between 1346 and 3148 MiB as free memory --
   admitting the very cut metal had measured breaching the corridor. A
   default of zero on an unmeasured term is not a conservative
   simplification; it is the most optimistic answer available, applied to the
   term most likely to bind. **When a model has a desk harness and a wired
   path, report which one produced the number, and make the wired path refuse
   what the desk one merely guesses at.**

35. **A difference is only meaningful against the baseline the rest of the
   model uses** (successor 51, a claim I made and refuted inside one shift).
   I armed the transient census at the residency census's post-capture point,
   then noticed that a phase-flip boot's boot-time backing swap releases the
   non-resident layout AFTER that point -- 3776 MiB on rank0 -- so free at
   rest sits GiB higher than the value I had baselined against. I "fixed" it
   to a running maximum of the free column and wrote that up as a law.

   It was wrong, and the arm boot's own data showed it: rank0's free
   oscillates between 1355 and 7820 MiB with the flip phase, so the maximum
   is not a resting state, it is the moment a layout is released. Worse, the
   gate's `fixed_overhead_mib` is calibrated as `nvml_used - params - pools`
   AT THE POST-CAPTURE POINT, so the residual already counts the layout that
   is later released. Referencing the transient to the higher level charges
   the same bytes twice, on the one rank where the constraint binds.

   Reverted to the post-capture reference, with the maximum recorded
   alongside and the raw per-state minima written into the artifact so any
   reader can re-reference without trusting mine. **Two consequences worth
   keeping: (a) before changing a baseline, ask which other terms were
   calibrated against the old one; (b) under a running phase flip there is no
   single "at rest" -- which is exactly why the corridor law is judged on an
   observed MINIMUM by an independent instrument rather than on a modelled
   at-rest level.**

### C42 — a curated suite reported health while a neighbouring directory was a third red (#656, merge-r4 shift)

Every shift on this line runs `scripts/run_631_flip_family.sh`. It has read
1116 passed since N49, it read 1116 passed after each of the seven merges in
the r4 batch, and it is the number each handoff quotes as the line's health.

`test/registered/unit/planner/` is not in that family. At `0ae49fafb4` — the
tip N51 handed over, with a clean 22-minute window behind it — that directory
read **109 failed, 2230 passed, 3 errors**. The briefing for this merge named
exactly two known-red tests in it. The real number was fifty times that, and
it had been there across at least four shifts without appearing in any
handoff.

It was not caused by any of the seven merges: the identical suite, re-run in a
separate worktree checked out at `0ae49fafb4`, produced the same 109 and a
`diff` of the two failure sets is empty. Step 7 (`feat/rig-advisor-413`) then
repaired 107 of them, because its two "wizard-blocking planner fixes"
(`runtime_reserve_mib` binding, `library=` passthrough) were also the root of
a hundred failures that did not look related to the wizard at all.

**The contradiction is not the red count, it is that the line believed a
number that was never about the thing being claimed.** "The suite is green"
meant "the curated list is green", and the curated list was assembled to cover
the phase-flip family. Its header already warns about three separate
under-collections *within* its own scope (successors 25 and 45 both had to
extend it); this is the same failure one level up, where the missing tests are
not absent from the list but absent from the concept of the list.

Related and deliberately not duplicated: law 33 already says an empty search
is evidence only if it could have come back full. This shift violated it
again, in the cheapest possible way — searching file *contents* for
`test_rejected_evidence_pins` and reporting the test absent, when the string
lives in the *filename*. Recorded here as a repeat offence rather than a new
law, because the law was already correct.

36. **A green suite bounds its own list, not the tree.** Before quoting a
   suite as evidence of health, ask what it was assembled to cover and what
   sits immediately outside it. A directory nobody runs is not passing; it is
   unobserved, and the two are indistinguishable from the outside. When a
   number has been stable across many shifts (1116 here), that stability is
   evidence about the list, not about the code around it.

---

## MERGE-R5 ADDITIONS (2026-08-12)

### C23 — `SGLANG_UNEVEN_TOKEN_VECTOR` is NOT the PP stage ratio

| | |
|---|---|
| **Superseded claim** | HANDOFF_VAL_R4 §4b: "`14,10,8` matches `--pp-stage-ratio 14,10,8`, while the script's `28,26,20` does not, and a per-stage token split inconsistent with the stage ratio is exactly the shape that would stall a PP chain." |
| **Standing claim** | It is the **uneven-DCP KV-token ownership vector** (`planner/placement.py:117`, `planner/feasibility.py:699`), coupled to each rank's remaining memory via `--rank-gpu-memory-mib`. Its agreement with the stage ratio on this rig is a **numeric coincidence**, not a mechanism. |
| **Closed by** | MERGE-R5, by reading the consumers. |

VAL-R4 was explicit that it had *not* bisected the six divergent keys and
named this one as "the first thing to test" — so this corrects a stated
hypothesis, not a stated finding. The wedge itself remains **unattributed**:
six env keys AND, as MERGE-R5 also found, **seven argv flags** differed at
once (`--model-path` differs by the `yarn1.5` suffix, `--pp-stage-ratio`
14,10,8 vs 2,1,1, `--context-length` 393216 vs 262144, and four more).

The operational lesson survives the correction intact, and is stronger than
the hypothesis was: **the drift, not any one key, was the defect.**

### C24 — the #695 recipe's rank discovery could never match

`host_shmem_695.py` tested `comm.startswith("sglang::scheduler")` (17 chars)
against a `comm` the kernel truncates to 15 (`TASK_COMM_LEN`), so it returned
an empty list on a healthy three-rank boot. `scripts/hostmem_sample.sh`
already used the truncated form `sglang::schedul`. **Two sibling instruments
in this tree disagreed about the same kernel fact**, and the newer one was
wrong. Closed by MERGE-R5, commit `1798cbfb6b`.

---

## LAWS, continued

37. **A hermetic test suite cannot see the shape of the world it abstracts.**
   The #695 branch's desk work was hermetic and correct, and it still shipped a
   discovery function that could not match a real `/proc/<pid>/comm`, because
   no hermetic test ever passed it one. Where a function's entire job is to
   recognise something the kernel produces, at least one test must be fed the
   **literal bytes the kernel produces**, copied from a live system, not the
   string the code was written against. "Hermetic" is a property of the test
   environment, never a warrant about the interface.

38. **Extract a suite's verdict from the whole file, not its tail.** MERGE-R5
   twice concluded that a `model_executor` run had been killed mid-flight,
   and twice attributed it to the known "this box kills long pytest runs"
   behaviour. Both runs had completed normally: the summary line was simply
   followed by a flood of trailing stderr, and a `tail -c 300` window never
   reached it. A missing summary is evidence about your extraction before it
   is evidence about the run — and a known flaky failure mode is exactly the
   explanation that will be accepted too readily.

39. **A saving measured at rest must be re-measured under load, and the
   instrument that matters is the one that was binding.** #695's 14.16 GiB was
   first shown on two at-rest snapshots. What makes it a *result* is the
   21-minute soak: cgroup `shmem` peak 76872 -> 62367 MiB and `MemAvailable`
   min 30827 -> 44635 MiB, against a defect whose observed consequence was
   nine cumulative cgroup OOM kills. The at-rest number is the mechanism; the
   under-load number is the claim.

---

## RES-R5 ADDITIONS (2026-08-12)

### C25 — the #695 exact-size pin does not cost flip latency; it saves it

| | |
|---|---|
| **Superseded claim** | HANDOFF_MERGE_R5 §6: flip p50 `2036 -> 2153 ms (+5.7 %)`, "not called a regression, not called clean", with the settling A/B specified. |
| **Standing claim** | Same-harness, one md5-frozen tree, arms differing by `SGLANG_PHASE_FLIP_EXACT_PIN` alone: **fix p50 2200.1 ms (n=540) vs revert p50 2257.5 ms (n=534)** — the exact-size pin is **57.4 ms FASTER at p50, 314.6 ms (18.5 %) faster at the minimum**, and faster at p90, p95 and the mean. |
| **Closed by** | RES-R5, `evidence-631/res-r5/AB_FLIP_LATENCY.json`. |

MERGE-R5's number was never a measurement of the allocator: n=18 against
n=546, two load profiles, and a baseline log holding 18 `FLIP DONE` lines
while its own window text claimed 186. MERGE-R5 said so and declined to quote
it. The replacement holds the tree, the argv, the load and the box constant.

The arms are provably the two allocators rather than two labels: the revert
arm reproduced merge-r4's host profile (shmem peak 76919 vs 76872 MiB) and the
fix arm reproduced merge-r5's to the megabyte (62367 vs 62367 MiB).

### C26 — #644's residual ~16 GB is untrimmed allocator arena, not retention

| | |
|---|---|
| **Open question** | HANDOFF_VAL_R4 §2: ~16 GB of host `RssAnon` survives load on BOTH sides of the #644 fix; RSS cannot say whether it is referenced or merely untrimmed, and gdb is not installed. |
| **Standing claim** | **ALLOCATOR.** In-process at end of load: live CPU tensor storage **20.1 MiB**, both named holders empty, and `malloc_trim(0)` returns **14714.2 MiB — 91.6 % of the residue**. The outside sampler sees the same release at constant process count (17.713 → 3.339 GB). |
| **Closed by** | RES-R5, `evidence-631/res-r5/GGUF_644_VERDICT.txt`. |

There is no holder to fix. #644's own fix is confirmed at the object level —
`data_container` and `expert_data_map` are empty on every parameter, which is
exactly what VAL-R4's RSS instrument could not see.

Not closed: whether to trim automatically at end of load. The residue is real
`MemAvailable` on a swapless box with nine recorded OOM kills, but a trim
costs the next allocations a fault-in and deserves its own measurement.

---

## LAWS, continued

40. **`pkill -f` will eventually kill the process that runs it.** RES-R5 used
   it on its own sampler and the pattern matched its own shell, which died
   mid-statement (exit 144). The prohibition is not about style: `-f` matches
   full command lines, so the killer, the router, and any agent's shell that
   merely *mentions* the pattern are all candidates. Stop processes by PID
   after a `py-spy` dump, and confirm death by PID.

41. **A budget with no slack turns any co-tenant into a boot blocker, and the
   preflight threshold will not catch it.** The ship config asks for 31800 MiB
   of a 32607 MiB card; after the 518 MiB NVML carve-out that leaves **under
   300 MiB**. `s33_boot_from_capture.sh` waits only while a card holds more
   than **2000 MiB**, so a foreign 500 MiB pytest passes the gate and the boot
   dies three minutes later at the memory-pool profile. A preflight threshold
   must be derived from the per-rank budgets, not from a constant.

42. **A drop in a resource trace is only a release if the process count held
   still.** The largest single drop in RES-R5's GGUF trace was 37 loader
   workers exiting at once (19.09 → 4.99 GB), not the `malloc_trim` it was
   looking for (17.71 → 3.34 GB at constant `nproc`). Memory leaving with its
   processes says nothing about whether memory was reclaimable. Constant
   `nproc` is part of the instrument, not a sanity print beside it.

43. **A cap that never binds still does damage: it masks the imbalance that
   would tell you where the capacity went.** `--max-total-tokens 620000` did
   not lower the ship pool -- the min across ranks was 503950, well under it.
   But it made the two non-binding ranks each report their capacity as exactly
   `620000`, so the sizing log read as "three ranks near the cap" when the
   truth was PP0 1252026 / PP1 503950 / PP2 727592 with **971718 token-slots
   stranded**. Lifting a cap is a MEASUREMENT before it is an optimisation:
   read the uncapped per-rank capacities before reasoning about the layout.

44. **Removing the pool cap OOMs the boot, so "the cap is soft" and "the cap
   can be deleted" are different claims.** The number lives in no source file
   (`--max-total-tokens`, `server_args.py:1151`, pass-through to
   `memory_pool.py:2481`) and `req_to_token` is `int32`, so nothing structural
   bounds it -- and yet `--max-total-tokens 100000000` died with
   `cuMemCreate ... CUDA_ERROR_OUT_OF_MEMORY` after the corridor guard counted
   free from 116 to -4 MiB. The TP layout can only address ids the PP
   allocator issues; uncapped, the TP pool allocates VRAM it can never address
   and starves the PP pool. A cap is removable only once the TP pool is sized
   to the PP id space.

45. **torch's device 0 is not NVML's device 0, and a per-rank MiB budget is
   read against the physical card.** torch orders FASTEST_FIRST, so gpu0 is
   the 5090 while `nvidia-smi` index 1 is. `--pp-stage-ratio 18,7,7` was
   costed as "give the big card more layers" and OOMed on it:
   `GPU 0 has a total capacity of 31.34 GiB of which 421.75 MiB is free`.
   rank0's budget was already 31800 of 32607 MiB -- 807 MiB, **below the 1024
   corridor floor**. Resolve the mapping from the boot's own OOM/geometry
   lines before attributing a per-rank number to a card.

46. **The phase-flip weight arena is a max over layouts, not a sum, so a moved
   layer can be free.** `phase_flip_boot.py`: `arena_total = max(
   layout_pp.total_bytes, layout_tp.total_bytes)`. HANDOFF_662 §5 costed
   `[24,23,17]` additively ("frees ~1.77 GiB on rank0 but adds ~1.31 GiB on
   rank1") and rejected it. A rank whose TP weight share already exceeds its
   new PP layer share pays nothing for received layers -- measured PP
   13482.18/8144.00/9114.95 MiB against TP 13692.29/7659.52/7659.52. 662's
   verdict survives for rank0 but on the ceiling argument of entry 45, not on
   its own arithmetic.

47. **A scaled RoPE cache that GROWS is not the cache it was built with.**
   The YaRN family builds from `_compute_inv_freq(self.scaling_factor)` and
   `* self.mscale`, but inherited a `_ensure_cos_sin_cache_length` that used
   `_compute_inv_freq(self.base)` -- the RoPE theta passed where a scaling
   factor belongs -- and no mscale. Appended rows carried different
   frequencies and the wrong amplitude, silently. It already fires at every
   boot (cache 393216 rows, reserve asks 393600) and is invisible only because
   `context_len` caps positions below the appended region. **Raising the
   context ceiling is precisely the change that makes the corrupt rows
   reachable.** Fixed via an overridable hook pair; guarded by
   `test/srt/test_yarn_rope_cache_growth.py`.

48. **A pool that boots, holds corridor and answers /health is not a pool that
   serves.** Boot E sized to 683150 with no cap flag, 0 tracebacks, corridor
   minimum 1469/2912/2975 and 0 breaches of the 1024 floor -- and every
   `/generate` timed out at 120 s. The log repeated `POLICY holding in tp:
   min dwell: 3.0s since last flip < 3s (pending prefill 1 tok, running bs 0)`
   against 362 flip events. Corridor-green is a MEMORY verdict; it says
   nothing about whether tokens come out. Every capacity number must carry a
   real generation beside it or it is a sizing result, not a serving result.

49. **Raising one rank's budget charges every rank, because the pool is a
   min-reduction.** Each rank's usage is `fixed + share_r x 32 KiB x pool`,
   and the pool is `min_r(capacity_r)`, so a budget raise on a non-binding
   rank lifts the global pool and spends VRAM on the binding one. Boot D
   raised rank1 and rank2 together, the pool went to 793844, and rank1 -- the
   rank whose own budget rose only 831 MiB -- died at 21 MiB free. Solve for
   the POOL that leaves every card at the floor, then derive budgets from it;
   do not tune budgets card by card.

50. **CUDA-graph capture peaks above the idle steady state, so budgets solved
   from idle free overshoot.** Boot C's idle free (1855/3468/3229) was
   measured after capture and read as available headroom. Budgets derived
   from it put boot D 867 MiB past rank1's floor and it died inside the full
   cuda-graph capture warmup, not at steady state. The corridor law is a
   continuous minimum; the capture transient is part of the series.

51. **A /proc scan matches the shell that runs it.** Scanning
   `/proc/*/cmdline` for `sglang.launch_server` returned the scanning
   `bash -c` process, whose command line contains the pattern, and killing the
   result took the shell down mid-statement (exit 144). Law 40 is usually
   filed under `pkill -f`; the actual hazard is any full-command-line match,
   including one built by hand. Exclude shells and the caller's own PID.

52. **The KV sizer fills to the corridor floor and leaves nothing for the
   phase-flip seam, which needs ~464 MiB to stage live rows across the layout
   change.** At pool 683150 the wedged boot logged 121x
   `staging 464 MiB needed but only 444 MiB is spendable` -- short by 20 MiB --
   then 336x `phase flip refused (guards): seam unfundable: tp_to_pp abandoned
   8 times consecutively`. Under strict purity a prefill cannot be built in
   the TP phase, so the queued token waited forever: health 200, zero tokens.
   A pool sized to "VRAM minus corridor" is NOT a pool the flip can operate;
   the seam's staging cost is a fixed post like any other and must be
   subtracted before the pool is sized.

53. **The flip policy commits its dwell clock before knowing whether the arm
   succeeded, so a refusable seam becomes an unbounded silent retry.**
   `note_flip_armed(state, decision, inp.now)` resets `last_flip_at` and
   increments `flips_armed` the instant `decide()` wants a flip, and
   `handle_phase_flip` drops the outcome for internally generated requests
   (`if internal: return None`). The runtime's own backoff -- seam_backoff and
   SEAM_ABANDON_CAP, built exactly for this -- is defeated by a policy layer
   that re-arms every `min_dwell_s` regardless. 179 arms, 0 completed
   cutovers. "362 flip events" in a log means arms attempted, NOT phases
   changed; counting them as flips is how this hid.

## 52. A requirement measured in the phase that does not pay it reads zero.

`pending_tail_bytes` is `want - committed` and `pending_restore_bytes` returns
0 unless the drafter is CURRENTLY spilled. Both are STATE, not requirement.
Sampled at the first round -- PP, arena committed, nothing spilled -- the
flip seam's fixed cost measured 0 MiB on all three ranks, against a runtime
refusal on the same rig naming 464 MiB (boot F, 2026-08-13). Price the commit
the seam faces FROM THE OTHER PHASE, which is a static layout quantity and
therefore readable in any phase.

## 53. `num_rows` is the rank's share, not the id space.

`KvPoolView.num_rows` is physical rows, and under the TP layout that is the
rank's TOKEN SHARE. Dividing a per-pool quantity by it gives a coefficient
inflated by 1/share: the same 1396 MiB of wave slack read 2360.7 B/row on
pp_to_tp and 5393.8 on tp_to_pp (boot F). Anything the sizer multiplies by
`T` must be normalised by the GLOBAL id space.

## 54. There are three KV sizing paths, and the ship config takes the one
without a headroom term.

A seam correction hooked at the post-capture measuring point applied nothing
at all on the production configuration -- the pool came back unchanged and
nothing logged (boot H). `_config_from_budget` is the single funnel. And the
pre-capture path has no headroom quantity to subtract a reserve from, so a
correction shaped as "headroom minus X" cannot be expressed there. Anchor on
a MEASURED position instead (bytes spendable above the law at a known id
space): everything unnamed -- activation reserve, capture peak, arena, TP
stack, carve-out -- is already inside a number that was measured with all of
it resident.

## 55. Unit tests that cover the arithmetic do not cover the module's
existence.

An edit spliced a file between two function names and removed four functions
including the one the scheduler calls; a follow-up edit to one of them
matched nothing and silently did nothing; every unit test still passed; the
next boot died at the first round on ImportError (boot I). Tests exercised
the maths, which survived. Pin the IMPORT SURFACE by the names the callers
use, one assertion per call site.

## 56. Corridor-green plus completed flips is still not a capacity claim
about a NUMBER.

The staging-aware sizer produced a boot that serves, completes 24 cutovers,
holds 1634/2845/1804 MiB continuous minimum free with 0 breaches, and
prefills 64001 tokens -- at a pool of 563974, which is BELOW the 620000 that
was already serving-proven. The mechanism being right does not make the
derived number bigger; it makes it honest. Boot E's extra 119176 tokens were
being paid for with a wedge. When a physics-derived pool comes out smaller
than an operator number, the budget vector is what to re-solve -- it was
solved against the corridor law alone and must be solved against
`corridor + seam floor` (measured per rank: 455 / 484 / 1455 MiB).

## 57. #364 idle-vacate is BUILT AND NOT ENGAGED on the phase-flip path.

Checked 2026-08-13 against three flip boots of this shift (boot_f, boot_g,
boot_j in /spinning/evidence-631/kvuniverse-r2): **zero occurrences of
"vacat" and zero of "resident state slots" in any of them.** The machinery
is real -- `managers/gdn_slot_runtime.py`, `mem_cache/gdn_slot_ladder.py`,
`gdn_slot_executor.py` -- and the scheduler calls it at the between-tick
boundary, but behind

    if self.server_args.gdn_resident_state_slots is not None:

and `--gdn-resident-state-slots` appears in NEITHER the ship argv capture
(`/spinning/evidence-631/s485/ship_argv.txt`) nor the uncapped flip argv.
Flag unset -> `gdn_slot_executor` stays None -> the ladder never runs. So
every claim of the form "unused Mamba states are spilled during bs1 time"
is, on this configuration, unbacked: the states sit as dead reservation.

WHY IT MATTERS TO THE SIZER, with numbers of the same order as the defect it
would relieve: idle GDN state is ~147 MiB per request slot (48 GDN layers,
fp32 SSM), so with bs1 running and slots 2-4 idle that is ~440 MiB parked --
against a seam that starved 20 MiB short on rank1 and 931 MiB short on rank2
(boot G). The mamba reservation is charged to the KV budget as its own post
(`MAMBA_BUDGET_POST`, model_runner_kv_cache_mixin) BEFORE KV is sized, and
the seam reserve's measured anchor is taken with every slot reserved -- so
vacatable bytes are counted as unavailable twice over and as reclaimable
never.

The rule this establishes: **a reservation that a built mechanism can
release is not a fixed post, but only if the mechanism is switched on.**
Neither the sizer nor the flip spec may assume the vacate; the flag has to
be in the boot argv and the vacate has to be COUNTED in the log before any
byte of it is spent in an arithmetic.

## KVUNIVERSE-R4 ADDITIONS (2026-08-13)

## 58. Engaging #364 is necessary but NOT sufficient: the ladder's standing
population is refused on the very path that needs it.

R3 (entry 57) found `--gdn-resident-state-slots` absent from every flip argv
and concluded the fix was to add it. Adding it works -- the cap fires, one
line per rank, and the bytes are real -- but the RUNTIME vacate still never
fires, for a reason no argv can fix: `build_gdn_slot_executor` sources idle
holders from `scheduler.kv_session_offload.live_offload_reqs()`, and
`--enable-kv-session-offload` raises at parse time with "(S1) supports
single-node pure TP/DCP only (pp_size=3, dp_size=1)". There is no opt-in env,
though `KVSO_ALLOW_SPEC` and `KVSO_ALLOW_HICACHE` sit within 30 lines of it.
A phase-flip instance is PP=3 by construction. So on the flip path the ladder
can be ARMED and can never have a victim.

The decomposition this forces, and it is the useful part: **slice 1 (boot-time
pool cap) delivers the bytes and needs no runtime vacate at all; slice 3
(admission decoupled above the cap) is the part that depends on the vacate,
and it is the part that is unbacked here.** Banking the bytes is therefore
legitimate; admitting past the cap is not.

## 59. The mamba cap does not touch the largest mamba component, and should
not.

Measured on this rig at `--gdn-resident-state-slots 4` against a profiled 12:
banked 0.26/0.18/0.15 GB per stack, roughly a quarter of R3's ~440 MiB
estimate. `ssm_state` and `conv_state` scale with resident slots and shrink;
`intermediate_ssm_state_cache` (0.62/0.44/0.35 GB -- larger than both) is
`per_req x capped_reqs x max_speculative_num_draft_tokens`, i.e. sized from
ADMISSION, which #364 slice 3 deliberately holds at the pre-cap profiled
count. A cap that shrank it would be capping the thing slice 3 exists to
preserve. **The vacatable mass is the resident state only, and it is about a
quarter of "the mamba memory".** Credit measured at an identical pinned pool
(563974): have_m 3744->4306 / 1224->1574 / 1514->1822 MiB, i.e. ~2x the
per-stack cap line because both the PP and the TP stack are capped.

## 60. `have_bytes` is PHYSICAL free VRAM, so the budget vector is inert for
any rank whose pool is seam-capped.

`measure_and_record` takes `have = torch.cuda.mem_get_info()[0] - law`. It is
not budget-relative and it does not care what `--rank-gpu-memory-mib` says.
On boot K3 rank2 spent 4406 MiB of an 8438 MiB KV budget -- its pool was
stopped by its seam floor, not by its ceiling -- so raising that ceiling
allocates nothing, frees nothing, and moves `have` by exactly zero. The pool
falls straight out of physical arithmetic on the binding rank:

    free at rest 2846 - seam floor 1455 - corridor law 1024 = 367 MiB
    367 MiB / 8192 B per token = 46976 tokens ; 563974 + 46976 = 610950

and the sizer derived **610942**. R3's carry-forward ("the next step is a
BUDGET re-solve, not a cap change") is therefore wrong in its noun: the vector
was never the constraint. **The lever is rank2's 1455 MiB arena tail --
`max(0, pp_bytes - tp_bytes)` over its two layouts, i.e. `--pp-stage-ratio`
against `--phase-flip-tp-vector`.** Reaching 620000 needs 438 MiB more free
memory on rank2; the mamba cap at its most aggressive plus halving
`max_running_requests` does not get there. **The honest optimum at this layout
is 610942, and the quarantine stays.**

## 61. A solver that targets equality ships a boot with zero margin, and its
own gate then reads red.

`seam_allowed_tokens` returns the largest T with `have(T) >= need(T)`, so the
boot it sizes lands, by construction, exactly ON the floor. Boot K3 derived
610942 and re-measured rank2 at **1430 MiB against a 1455 MiB floor** -- 25
MiB the wrong side, logging `THIS BOOT CANNOT FUND ITS OWN FLIP` -- while
completing all 30 cutovers with 0 refusals and 0 abandons. Two things follow.
The 25 MiB is second-order error in the slope (allocator granularity, arena)
against a term with no margin to absorb it. And the flip gate separately
demands a 512 MiB C20 entry margin that the SIZER never books, so "fundable
at rest" and "fundable at the gate" are different questions sized by different
code. A margin term belongs in the solver, and the honest one is the entry
margin the gate will actually apply.

## 62. The seam record fingerprint omits the flags that move the memory
balance by hundreds of MiB.

`measured_kv_budget_fingerprint_fields` carries `rank_gpu_memory_mib`,
`context_length` and `max_running_requests`, but NOT
`gdn_resident_state_slots`, NOT `enable_kv_session_offload`, NOT
`pp_stage_ratio`, NOT `phase_flip_tp_vector`, NOT `max_total_tokens`. Boot K1
consumed boot J's UNCAPPED record while running capped, which under-sizes and
is harmless; the reverse direction over-sizes and is precisely the boot-E
wedge. Meanwhile changing the budget vector DOES orphan the record and costs a
cold boot. So the fingerprint is simultaneously too coarse for the flags that
matter to the seam and fine enough to be expensive for the one that does not.
Any layout experiment must pin or invalidate the record deliberately.

## 63. The phase-flip seam floor is a WEIGHTS term, not a KV term, and the
cheap knob is the TP vector.

Two shifts described rank2's 1455 MiB floor as "the arena tail" without
naming what fills it, and the natural reading -- a KV arena, so reshape the
KV budget -- is wrong. It is `max(0, PP_weights - TP_weights)` on that rank's
two layouts. The runtime prints it (boot L3): `rung 3 released 924.0 MiB of
weights-arena tail to the driver (TP layout needs 8188.4 of 9115.0 MiB)`, and
it reproduces register 46's measured PP/TP weight vectors
(13482.18/8144.00/9114.95 against 13692.29/7659.52/7659.52) to the MiB on all
three ranks.

The consequence is a lever nobody had used: **`--phase-flip-tp-vector` moves
the floor for free, `--pp-stage-ratio` does not.** Raising the binding rank's
TP share shrinks the subtraction without touching `have`, because `have` is
measured at rest in the PP phase, where rung 3 has already released the TP
arena. Moving PP layers instead charges the receiving rank ~1304.9 MiB of
`have` per stage unit -- its PP weights and its KV both grow -- which is why
every layer-shifting candidate modelled WORSE than the status quo while the
TP-vector change modelled better and then measured better:
`32,16,16 -> 30,16,18` dropped rank2's floor 1455 -> 927 MiB.

## 64. The quarantine constant is deleted, and the thing that replaced it is
a mechanism, not a bigger number.

`PHASE_FLIP_SERVING_PROVEN_TOKENS = 620000` carried its own deletion
instruction ("deleted, not raised, when the livelock is fixed"). Boot L3, with
the seam reserve, the margin term and the fixed policy loop, derived **648388
tokens** -- +28388 above the constant -- and served it: 24 completed cutovers
in both directions, 0 abandons, 0 refusals, 0 tracebacks, 64001-token prefill,
real generations, corridor 1128/2567/1664 MiB with 0 breaches over 5991
samples. The net that stays is `seam_reserve_enabled()` defaulting True, so a
pool sized as "VRAM minus corridor" with nothing left to stage the seam cannot
be built by accident; the gate test asserts THAT instead of a token count.

The trap to avoid on the way: **boot L2 derived a seam ceiling of 651498 and
still ran at 620000**, because the constant was still clamping it. A seam
"allowed" number in a log is not the pool. Check for the capping warning, or
read the id space the seam re-measure reports, before believing a capacity.

## 65. A solver that targets equality needs a margin, and the margin belongs
on the measured position, not on the floor.

Adding it to `F` looks equivalent and is not: in the slack-bound regime `F`
appears ONLY in the regime test, so a floor-side margin leaves that whole
branch unmargined. Subtracting from `have_m` is identical in the floor-bound
regime and correct in both. Default 192 MiB, and deliberately NOT the flip
gate's 512 MiB C20 entry margin -- the gate satisfies that at flip time from
transient reclaim (`CORRIDOR-GUARD cleared ... reclaimed 136 MiB from
[allocator-cache]`), so reserving it in the sizer would charge one requirement
twice, about 65k tokens on the binding rank. Result: every rank re-measured
ABOVE its floor at the derived pool (+2873/+224/+173 MiB) against boot K3's
-25 MiB and its `CANNOT FUND ITS OWN FLIP`.

## 66. #364's admission optimism is real but BENIGN: the scheduler's own slot
check gates admission before the allocator is asked.

`session_admission_slots(..., vacate_available=_resident_cap is not None)`
keys "the overflow has a backer" on the FLAG, and on a PP flip boot the
backer cannot exist (entry 58). Falsified directly: four concurrent sessions
against a 4-slot pool (two requests' worth) **all completed correctly**, 200
tokens each, accept 2.857-3.571. The pressure surfaced as max `#running-req`
3 against 4 requested, max `#queue-req` 1, mamba usage 1.00, and 6x `mamba
slot pool exhausted and nothing evictable ... skipping this cache insert` --
with **0 OOM, 0 crashes, 0 wrong answers**. So the hazard is throughput and
lost state caching, not the admit-into-OOM the code guards against, and the
correction is documentation: the flag's help text claims sessions beyond the
cap "run with a vacated (host-blob) state", which on this path is false --
they run by WAITING for a slot.

## 67. "kvso is refused under PP" does NOT mean the bs1 spill requirement is
unmet, and the two must not be conflated.

The spec's "bs2-4 reserves incl. unused Mamba states spilled in bs1 time" is
carried on flip boots by the PHASE-FLIP SPILL LADDER, which fires 18 times per
direction on boot L3: `rung 1 (cache) at pp_to_tp / tp_to_pp`, `rung 2
SPILLED / RESTORED the draft weights`, `rung 3 released ... weights-arena
tail`. Those rungs are what make the seam affordable at all. What is refused
under `pp_size>1` is only kvso's SESSION KV tail on the host, and with it
#549's GDN-vacate-x-kvso fixes. Reading the parse-time refusal as "spilling
does not happen here" would retire a mechanism that is live and load-bearing.

## 68. `--json-model-override-args` is a SHALLOW merge, so a nested override
silently deletes its siblings.

Passing `{"text_config": {"rope_parameters": {...}}}` to raise the YaRN factor
replaced the ENTIRE `text_config` -- 25 keys -- and the boot died on
`'PreTrainedConfig' object has no attribute 'max_position_embeddings'`, an
error that reads like a broken checkpoint rather than a lost sibling key. The
follow-on trap: adding `original_max_position_embeddings`, which the
transformers YaRN validator demands on that path, made the derived ceiling
COLLAPSE from 393216 to 262144, because that key marks `max_position_
embeddings` as already-scaled and the runtime's convention here is
`derived = max_pos x factor`. Two opposite failures from one flag. For a
nested model-config change, edit a config.json in a symlink checkpoint
(7.5 K, weights not copied) instead of overriding through argv.

## 69. The 1M context ceiling is NOT free: ~440 MiB per rank, eager, and
worth 9% of the pool.

Measured at an identical pinned pool with nothing using the context, `have`
fell by exactly 440 MiB on EVERY rank (3312->2872 / 708->268 / 1100->660)
when the ceiling went 393216 -> 1048576. Equal across ranks is the signature
of a replicated per-position structure: `head_dim 256 x partial_rotary 0.25`
-> 64 rotary dims, cos+sin 128 values/row, fp32 512 B/row x 655360 new rows =
320 MiB, with ~120 MiB more not separately attributed. The user law's "every
session grantable ~1M ctx with zero upfront cost" therefore does not hold
today -- the reserve is eager. It IS priced, because `have` is a physical
free reading and the sizer lowers the pool on its own: derived 589736 at
1048576 against 648388 at 393216, **-58652 tokens, -9.0%**. Nothing is broken
without a lazy reserve; 9% of the pool is simply the standing price.

## 70. Positions beyond the old ceiling decode CORRECTLY -- register 47's fix,
proven on metal instead of in a unit test.

Entry 47 established that the YaRN family grew its cos/sin cache with
`_compute_inv_freq(self.base)` and no mscale, so appended rows carried wrong
frequencies, and that **raising the context ceiling is precisely the change
that makes those rows reachable**. Raised to 1048576 and reached: a
determined-answer probe with **400030 prompt tokens** -- 6814 past the old
393216 -- returned the planted `BANANA47`, finish=stop, in 330 s. The guard
test `test_yarn_rope_cache_growth.py` now has a metal counterpart.

## 71. A pinned pool that the seam cannot fund reproduces the wedge exactly,
which is why the pin -- not the pool -- is the defect.

Boot M1 carried the 393216-era pool (648388) into a 1048576 context, where
the eager RoPE reserve had taken 440 MiB per rank. Both small ranks logged
`CANNOT FUND ITS OWN FLIP` AT BOOT (268 vs 484, 660 vs 927); the instance
served short prompts and the 400030-token probe, then held in TP with
`tp_to_pp refused 14 times and treated as unfundable (seam abandoned: a peer
refused); re-probing in 13.8s`, 0 tracebacks, all processes alive. The same
configuration UNPINNED derived 589736 and ran 132 completed cutovers (66/66)
with 0 abandons and 0 refusals. Two lessons: an operator pool number is a
liability whenever anything else about the memory balance moves, and the R3
control loop has converted this failure from a silent livelock into a bounded
one that announces itself twice before it happens.

## 72. A-vs-A is not byte-identical at 1M, and this is recorded WITHOUT an
attribution on purpose.

Two identical temperature-0 requests on boot M2 returned different text and
took 2.79 s and 9.66 s -- i.e. they were served in DIFFERENT PHASES, and the
rig separately carries a documented upstream GDN prefill nondeterminism
beyond ~109 tokens. The control that would attribute it -- the same A-vs-A on
the 393216 configuration -- was NOT run this shift. So "YaRN 4.0 broke
determinism" is unsupported by this evidence, and so is its denial. Recorded
as an open observation with its missing control named, rather than as a
finding; law: a difference measured under two uncontrolled variables is a
question, not a result.

## KVUNIVERSE-R6 ADDITIONS (2026-08-13)

## 73. Serving was never crashing on this edge: it was a cgroup member of the
agent session that started it.

`setsid` detaches the SESSION, not the CGROUP. Every agent shell runs in
`/system.slice/claude.service`, so a server booted from one joins that unit
and every restart of it SIGTERMs the server as collateral. The instance that
drained at 2026-08-13 09:04:59 mid-measurement died exactly that way:
`claude.service` came up at **09:05:07**, eight seconds later, restart counter
11. Checked directly rather than inferred -- the peer shift's fresh restore
and all three of its schedulers read `0::/system.slice/claude.service`; the
router at 30099 survives the same restarts only because it has its own unit.
Fix: capture-replay boots launch inside a transient `systemd-run --scope`, and
the boot prints the membership check that proves it. The law: a process
outlives the session only if it leaves the session's CGROUP, and the proof of
that is `/proc/<pid>/cgroup`, never an absence of crashes.

## 74. An acceptance check that cannot pass is worse than no check: `[ -s ]`
on a cgroup file is always false.

`cgroup.procs` is kernfs and stats as size 0 even when it lists pids, so the
boot's own escape check reported "scope has no pids" for a scope with a live
server in it. The mechanism had worked from the first boot; only its acceptance
print was wrong. A check with a false negative invites exactly the wrong
repair -- undoing a working fix -- so a new check has to be proven able to
PASS as well as to fail.

## 75. A per-batch hook in ModelRunner.forward runs inside CUDA graph capture.

The eagle draft worker enters `ModelRunner.forward` from within
`torch.cuda.graph` capture (`eagle_draft_cuda_graph_runner.run_once ->
draft_runner.forward`). Any device read there raises
`cudaErrorStreamCaptureUnsupported`, and the boot dies later and elsewhere, in
`capture_end`, which points at the graph runner instead of at the read. So a
capture guard belongs at the TOP of any such hook, not inside the function it
calls -- and "the graph runner bypasses ModelRunner.forward on replay" (true)
does not imply it bypasses it on CAPTURE (false).

## 76. The class a rig actually builds is not the class the code reads like.

The lazy RoPE allowlist was written for `YaRNScalingRotaryEmbedding` after a
code survey. The rig builds `YaRNScalingMRotaryEmbedding`, the M-RoPE variant,
and the first lazy boot ran fully EAGER with one log line per rank saying so.
That is the allowlist doing its job -- register 47's failure mode is silent
wrong attention, so an unverified growth hook must not be trusted -- but it
cost a boot, and the lesson is that a class-identity assumption is worth one
`grep` of a real boot log before it is worth a design. It also carried a real
consequence: M-RoPE position ids are NOT bounded by the sequence length, so a
host-side `seq_lens` bound is wrong for a multimodal batch and needs its own
branch.

## 77. The lazy RoPE reserve is CORRECT UNDER TEST AND WRONG ON METAL past a
depth its own guard does not see.

Unit suite: 22 green, including a can-fail proof that reintroducing register
47's bug turns 10 of them red. Metal, same tree, same argv, same prompt, one
variable (`SGLANG_ROPE_LAZY_CACHE`):

| depth | lazy | eager |
|---|---|---|
| 100026 tok | EXACT | -- |
| 250026 tok | EXACT | -- |
| 390026 tok | **1 token, empty** | -- |
| 400026 tok | **1 token, empty** | **EXACT `BANANA47`, 12 tok** |

and the corridor under the 400k probe: lazy 692/1785/1174 MiB with **19
samples under the 1024 MiB law**, eager 1494/3401/1766 with **0**.

`SGLANG_ROPE_LAZY_VERIFY=1` -- which asserts `positions.max() < filled` on
every batch -- stayed SILENT through the failing probe. So the rows the
runtime asked for were inside the region the bookkeeping believed written, and
the defect is in the bytes or their visibility, not in the accounting. Two
candidates, neither eliminated: a fill/read ordering gap across this runtime's
streams and captured graphs, and UVM eviction under the device-memory pressure
this same feature creates by moving its cost from boot time to run time.

Three laws restated by this:
* A host-side bookkeeping assertion cannot witness a device-side visibility
  bug. The guard that would have caught this compares CONTENT (a row read back
  against the formula), not indices.
* Register 55 again, on my own work: unit tests that cover the arithmetic do
  not cover the module's integration.
* A feature that moves a cost from a priced moment (boot, where the sizer
  reads `have`) to an unpriced one (serving) has not saved the cost; it has
  moved it somewhere with no accounting, and the corridor is where that shows.
