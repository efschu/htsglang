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
| C18 | how many floor models are there | one: the corridor law at 1024 MiB, enforced by `CorridorGuard` | **two, and they have never met.** `vram_dial._measure_local_floor_bytes` derives its OWN card-level NVML floor and spends against `budget_bytes - floor_bytes`, with no reference to `CorridorGuard` or `law_floor_bytes`. It is inert on this configuration (`--enable-vram-dial` absent) so nothing is wrong today, but a physically-growing allocator that answers to a different floor than the law is the same shape as C17 and will diverge the moment the dial is enabled | HANDOFF_678 §1a (sweep) |
| C19 | item 16's rebalance tier is missing an ACTUATOR | the objective is computed and the mover is the gap | **the objective had no CALLER either.** `corridor_guard.water_fill_targets` had exactly one caller in the whole tree and it was a test, so five handoffs of "the rebalance tier is missing" ran while nothing in production ever asked how much to rebalance. **RESOLVED as an instrument in HANDOFF_679 §1e**: `water_fill_transfers` + `describe_water_fill` now render the move wherever the guard already reports the spread, and the measured answer on this rig is ~948 MiB stranded on the wrong card at the binding instant. The actuator is still absent and is now a decision with a price attached, not an open-ended ask | HANDOFF_676 §6.2 -> HANDOFF_679 §1e |
| C20 | where the corridor's DEPTH is made | the allocations the corridor gates price -- that is why the gates exist | **RESOLVED AS A MECHANISM in HANDOFF_681, with its residual named and measured.** The depth is made at seam ENTRY, and the entries are INHERITED: successor 37 measured on s34's own green window that the deepest minima come in PAIRS of cutovers ~2 s apart, the second entering at the first one's trough and drawing nothing (`13:59:11 entry 1499 draw 456 -> min 1043`, then `13:59:13 entry 1043 draw 0 -> min 1043`). The draw is also SELF-LIMITING -- from a low entry it was at most 456 MiB, the 1026 MiB draws all follow entries above 2400 -- so the fix is a condition on ENTRY, not a cap on the draw, and a worst-case entry requirement was rejected because it would arm on 77% of cutovers, which is s36's falsified cache-dumper by another name. The seam gate now asks for `staging + 512 MiB` and grades its verdict (enter / DELAY, budgeted in consecutive GROUP abandons / enter on the law loudly / refuse), riding the existing `_collective_min` with no new collective. **MEASURED over 65 min, 348 flips each way, 0 breaches:** the binding card's IN-CUTOVER minimum 1043 -> **1123 MiB** (+19 -> +99 over the law) and its draw-from-a-low-entry 456 -> **184 MiB**. **THE RESIDUAL, stated so nobody reads this as closed:** a cutover entering HIGH still draws up to 1040 MiB and lands at the same ~1123 floor, so the trough is now bounded by the seam's UNPRICED tail demand rather than by an unguarded entry. That is a staging-pricing problem (`SGLANG_FORWARD_PEAK_PATH`), not an entry-condition one. Price of the margin: the KV rung fired 348 times against s34's 21, every shrink recovered and the pool never left 512552. **RESIDUAL 1 CLOSED in HANDOFF_682**: the rung now keeps an ADMISSION RESERVE above the live high-water mark (the scheduler's own `chunked_prefill_size`, 512 here) and the gate declares its margin DISCRETIONARY, bounded by what the rung can return without crossing that floor. Proven with the same forced 8192 MiB margin that killed the instance: 156 delays, 78 yields (s37: 72/35 then `full_available_size=0` on all three ranks), 0 admission failures, 0 tracebacks, health 200, 0 breaches in 3926 samples. Inert on the shipped margin — floor 593 rows against 512552 backed, 0 bounded asks — and the 30-min confirmation window held every axis (C20 VERDICT: HELD, 0 breaches in 13118 samples, MTP 2.869, YaRN 271237). **RESIDUAL 2 is now C21** | HANDOFF_680 §1a -> HANDOFF_681 §2 -> HANDOFF_682 |
| C21 | the seam's staging price against what the seam HOLDS | HANDOFF_681 §2b: "a cutover entering HIGH still draws ~1040 MiB that `_staging_bytes` does not price", read off the gpu0 corridor column | **THE METAL HALF IS A CARD MIX-UP, THE HERMETIC HALF IS REAL.** `--rank-gpu-id 0,1,2` is in CUDA order, which is FASTEST_FIRST on this rig, so **rank 0 is the 5090 (nvidia-smi index 1)** and ranks 1/2 are the 3080s (smi 0 and 2) -- confirmed three ways: the per-rank MiB vector `31800,14000,15600`, the guard's own free readings (PP1 clears at 1886/2726 MiB match the gpu0 column, PP2's 2968/3236 match gpu2), and the boot's own profile (rank 0 capacity 619990 tokens against rank 2's 308808). Joining `PP0` log lines to the `gpu0` column therefore crosses two different cards. Joined correctly (`scripts/s38_seam_price_vs_draw.py`), the binding card's 115 pp->tp seams were **priced at 1177 MiB p50 (max 1478) against a deepest drawdown of 504 MiB**: the gate OVER-reserves there, it does not under-reserve, and no unpriced tail is visible at the corridor. **What IS real is smaller and hermetic:** with the wave plan matched, `_staging_bytes` prices 2.398 MiB against a measured live set of 3.769 MiB on the three-rank CPU fixture (0.64x) -- the formula models `incoming + max(outgoing, local)` and the probe's high-water holds an outgoing leg beside the local read and its gather window. Not reproduced on metal (torch's allocator cache absorbs it, which is why 348 flips saw 0 breaches), NOT widened on that evidence, and pinned as a one-sided ratchet in `test_the_waved_price_is_short_of_the_measured_live_set` | HANDOFF_681 §2b -> HANDOFF_682 |
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
