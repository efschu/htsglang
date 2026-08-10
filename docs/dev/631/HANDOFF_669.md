# HANDOFF 669 — #656 / #631 Route A, successor 26

Predecessor: HANDOFF_668 (successor 25). Read that for the ceiling
confirmation and the 2.1b design note; this file corrects three of its
claims and prices the route honestly.

---

## 0. THE ONE-PARAGRAPH VERSION

Section 2.1b is IMPLEMENTED, tested and pushed. It works, and it raises
the anti-wedge ceiling — but **not to 600,000**. Priced on the rig's own
measured numbers, restore-first + W=16 + ordered waves reaches about
**473,000 tokens**, against the confirmed ~435,000 for the old seam. The
601,233 in HANDOFF_668 rests on a staging slope that does not hold: the
payload leg stops shrinking around W=8, and the backing transient is
pool-proportional and large rather than a 60 MiB constant. The term that
actually blocks 600,000 is the **backing transient (1821 MiB of the
2508 MiB needed at full occupancy, against a 753 MiB budget)** — so the
next move is section 2.1, which removes that term, not more wave tuning.

---

## 1. INHERITED STATE, VERIFIED AND BOOKED

Successor 25's 65-minute acceptance run at pool 430000 completed at
16:45Z. Its detached finisher did what it promised: wrote
`/spinning/evidence-631/s25/acceptance/extract.txt`, folded it into
`PROD_BRINGUP_BENCH` section 2h, committed `06ef108394`, pushed, and
removed its own heartbeat.

**Verdict: ACCEPTANCE: GREEN.**

| axis | result |
|---|---|
| corridor | 0 breaches; min free 2231 / 5098 / 2487 MiB (floor 1024) |
| flips | 348 pp_to_tp + 348 tp_to_pp, both layouts visited |
| abandons / tracebacks | 0 / 0 |
| strict purity | true — 11313 prefill batches, 0 with a graph |
| decode graphs | 1341 of 1344 = 99.8% |
| MTP | accept length 2.788 over 1344 |
| traffic | real, via router 30099 (240 completions + 26 messages) |
| host RAM | peak 112.1 GiB |

**TWO CAVEATS THAT MUST TRAVEL WITH THAT GREEN**, because the extract
alone reads stronger than the evidence is:

1. **Occupancy peaked at 38.1% of pool** (163626 of 430000 live slots).
   By the occupancy law this is a FUNCTIONAL acceptance and says nothing
   about the ceiling. It is not a capacity result.
2. **The corridor law's second half is not met.** Margins of 1207 / 4074
   / 1463 MiB above the floor are "never breached" but not "well filled".
   This is exactly the tension HANDOFF_668 section 2.3 describes, and it
   is the reason the seam work is not optional.

`memory.events` shows `oom_kill 9` with 0 tracebacks. Those counters are
cumulative for the cgroup and predate this run; I did not attribute them
to it and neither should anyone else without a timestamped check.

---

## 2. WHAT I BUILT (0ee52e7dac, f01c17e03a)

The 2.1b `_execute` loop, which HANDOFF_668 left unwritten. Three moves,
all behind one switch.

1. **Per-wave order is now `reclaim -> restore -> release`.**
2. **Wave cap lifted** from the smallest PP stage (4) to one layer per
   wave (16), via `restore_first_wave_count`.
3. **Wave ORDER chosen per direction** by `ordered_layer_waves`, so the
   unavoidable transient lands on the largest-share rank (rank 0, the
   5090 — derived as `argmax(tp_vector)`, not hardcoded).

`SGLANG_FLIP_SEAM_RESTORE_FIRST=0` rolls back all three together.

### 2.1 Three corrections to HANDOFF_668's recipe

Its section 2.1c was flagged UNVERIFIED. I checked it against the code.
The file:line map is broadly right; three of its conclusions are not.

* **`SeamOrderingTest` must NOT go red, and must not be touched.** 668
  lists its three methods as order pins to rewrite. They drive
  `WavedBackingSwap.__call__` — the WHOLE-POOL path, not the wave loop.
  That path holds both layouts for the width of the swap and therefore
  keeps release-first; its docstring already says so. Rewriting it to
  restore-first would reintroduce the residency the waved seam exists to
  remove.

* **"Both binding cards pay approximately ZERO transient" is
  unreachable.** 668's rule (delay each card's first owned layer past
  `1/share_r`) fixes only the FIRST layer and ignores the catch-up the
  rest must do. Rank 1 holds 5 of 16 layers at a 0.3125 share, so even
  placed dead last it ends at `5 - 0.3125*15 = 0.3125`; rank 2 floors at
  `4 - 0.25*15 = 0.25`; both floors need the same final position, so no
  order attains both. Exhaustive search over the real geometry puts the
  optimum at **0.5 layer spans against 0.688** for the proportional W=4
  split — a 27% cut, not a cut to zero. (In `pp_to_tp` the gain is
  larger, 1.375 -> 0.5, because the naive order is much worse there.)

* **Aliased pools needed an explicit ORDER gate.** 668 says
  "`_pools_alias()` already gates that". It gates the wave COUNT only. A
  single wave still runs the same seam branch, and on aliased pools the
  destination's pages ARE the source's — restore-then-release hands back
  the mapping just committed and leaves the destination unbacked. Gated
  now, and pinned on ORDER rather than on bytes, because the existing
  aliased test checks byte identity only and would have stayed green.

### 2.2 A fourth thing, found by my own arithmetic after the fact

The first version of the rollback switch moved only the order, leaving
W at `n_layers` and the slack accounting on the restore-first term. That
is not the old design and is worse than either: a one-layer wave under
RELEASE-first has no release of its own to pay for its commit. Priced at
**354,868 tokens**, an 18% regression against the thing it was meant to
roll back to. Order, wave-count default and `_backing_slack_bytes` now
move as one switch (`f01c17e03a`), with tests.

`_backing_slack_bytes` itself had to be re-derived: it credited a wave's
own release against its own commit, correct under release-first and an
under-reservation now. It weighs commits through `j` against releases
through `j-1` when restore-first is active, and the old way when it is
not.

### 2.3 Tests

Red-first throughout. `TestWavedSeamOrdering` drives the REAL wave loop
by duck-typing `release_wave` onto a recorder — note that
`TestPreWriteSeamOrdering` passes a plain callable and therefore takes
the `swap is None` branch, which is why it could never have seen this
order. The planner is verified against BRUTE FORCE on a map small enough
to enumerate (210 sequences), because a DP optimising the wrong
recurrence still returns a confident answer.

Flip family: **717 passed / 1 failed**, against 702 / 1 inherited. The
single red is the pre-existing constant-dominated `_staging_bytes`
over-reservation in `test_phase_flip_mover_streaming_631`; it calls
`_staging_bytes` without waves, which takes the `len(waves) <= 1` guard
and returns zero backing slack, so nothing in these commits can reach it.
ruff clean on every touched file.

---

## 3. THE CEILING, PRICED HONESTLY

Method: `staging reserved` on the flip DONE line IS `_staging_bytes()`, a
pure function of the plan — so the design comparison is exact arithmetic
and needs no boot. I built the rig geometry (map `[7,5,4]` over 16
full-attention ordinals, vector `(14,10,8)`, rank shares
0.4375/0.3125/0.25) and calibrated one free parameter, the row width,
against a MEASURED point: 1132.0 MiB at 163626 live slots.

Calibration returned **2036.4 B/row**. Independent check: the DONE lines
give 545.38 MiB per 279237 cells = **2048 B/cell**. The model reproduces
a number it was not fitted to, within 0.6%.

**SECOND VALIDATION, on metal, against the NEW code.** The 16:47Z boot at
pool 430000 running restore-first W=16 reported `staging reserved
1312.50 MiB` at 90 live slots. The model — calibrated only against the
OLD accounting at 163626 slots — predicts **1305 MiB** for that point.
0.6% error on a prediction it was not fitted to, in a different accounting
regime and at the opposite end of the occupancy range.

That single number also settles the disagreement with HANDOFF_668
directly, with no modelling in between: **the seam's pool-proportional
constant is ~1305 MiB, not the 60 MiB the 601,233 estimate assumed.** It
is measurable at idle, because it does not depend on the live set.

**Metal also cleared the one risk W=16 introduced.** Four times the waves
means four times the exchange round trips, and the flip sits inside the
no-return region. Measured: **1214–1859 ms over 16 waves**, against
1336–2074 ms over 4 waves on the previous boot. No penalty; the per-wave
fixed cost is not what dominates a flip.

### Staging demand, worst rank, by design

| pool (full occupancy) | old W=4 | 2.1b W=16 |
|---|---|---|
| 430,000 | 2328 MiB | 1806 MiB |

Note the crossover: below ~250k live slots the NEW scheme is *more*
expensive, because it trades a live-set-proportional slope for a
pool-proportional constant. It only pays at high occupancy — which is
the regime that decides the ceiling, so this is the right trade, but it
means a low-occupancy run will make 2.1b look like a regression.

### Ceilings, solved at full occupancy against the binding-card budget implied by the confirmed 435,000

| design | ceiling |
|---|---|
| release-first W=4 (old, confirmed four ways) | ~435,000 |
| restore-first W=4 | 354,868 (regression — see 2.2) |
| restore-first W=8 | 414,923 |
| **restore-first W=16 (2.1b as shipped)** | **473,157** |
| user floor, spec item 6 | 600,000 |

### Why 600,000 does not follow, and what would get there

At 600,000 the binding card's KV alone is 5826 MiB of a 6579 MiB budget,
leaving **753 MiB** for staging. 2.1b needs **2508 MiB**, and the split
is the whole story:

    payload leg          687 MiB
    backing transient   1821 MiB

The payload leg would very nearly fit inside the 753 MiB budget on its
own. **It is the backing transient that blocks 600,000**, and no wave
tuning removes it — it is pool-proportional by construction. That is
section 2.1, the fully streamed seam, which drives the transient toward
zero; HANDOFF_668 priced it at 670,803 and my decomposition is consistent
with that being the design that clears the floor.

So the route to spec item 6 is **2.1, not more of 2.1b**.

### Why 2.1 removes the term, stated as a granularity argument

Worth having in one line, because it says exactly what to build. The
transient is one LAYER span because a whole layer is the unit that gets
committed before anything is released. Nothing about the seam requires
that: the unit is a layer only because `restore_backing(layers)` takes a
layer list. Commit in ROW BLOCKS instead and the transient becomes one
BLOCK span, which is a tuning knob rather than a geometry constant — and
it goes to ~0 as the block shrinks.

That is precisely what the built-but-unused substrate does:
`KvVmmArena.commit_span`/`decommit_span`,
`KvVmmBufferOwner.back_token_span`/`release_token_span`,
`HostKvPool.release_backing_span`/`restore_backing_span`. They are
tested. The missing piece is the loop that drives them, plus the global
round count for the collective. So section 2.1 is not a new design to
invent — it is wiring an existing, tested substrate into the seam.

### Where HANDOFF_668's 601,233 came from

Two terms. It modelled the payload slope as `18 / W` MiB per 1000 live
slots, giving 1.13 at W=16; the payload actually saturates near W=8
because the widest wave still carries a layer of one's own plus one from
each peer, a W-independent floor. And it modelled the transient as a
60 MiB constant; it is pool-proportional and reaches 1821 MiB at 600k.
Neither error is visible from a single measurement, which is why the
number survived to be inherited.

---

## 3b. METAL: THE 430000 STEP UNDER THE 2.1b SEAM

Boot 16:47Z, pool 430000, restore-first W=16, 12-minute step.
`s25_step_verdict.py`: **PASS**.

| | |
|---|---|
| corridor | 0 breaches; min free 2763 / 5506 / 2829 MiB |
| flips | 72, 0 abandoned, `seam waves=16` |
| staging reserved | min 420.1, max 1601.0, mean 921.6 MiB |
| occupancy | 131455 slots = **30.6%** |

The verdict tool refuses this as capacity evidence on occupancy grounds
and it is right to. But its own anti-wedge extrapolation, computed from
the binding card's baseline excluding staging (2763 + 1601 = 4364 MiB),
gives **502,863** as the largest pool whose full-occupancy flip still
clears 1024 MiB.

**So two independent methods now bracket the 2.1b ceiling:**

| method | ceiling |
|---|---|
| calibrated staging model (section 3) | 473,157 |
| verdict-tool extrapolation from the metal step | 502,863 |
| confirmed old seam, four methods | ~435,000 |
| user floor | 600,000 |

They disagree by 6% and agree on the thing that matters: 2.1b is a real
gain of roughly 40–70k tokens, and it does not reach 600,000.

## 3c. THE OCCUPANCY BUG WAS IN THE LOAD, NOT THE POOL

Worth its own section because it has cost this chain four capacity steps.
Every one has been refused for the same reason — 26%, 38.1%, 30.6% — and
each successor has treated it as a load-tuning problem to be nudged.

It is not tunable, it is structural. `soak_631_mixed_load.py` drives
occupancy from its PREFILL worker, whose requests retire almost as fast
as they arrive, while its DECODE workers — the ones whose docstring says
they are "what must be RESIDENT across a cutover" — carry a
one-sentence prompt and `max_tokens=512`. A request that has retired
cannot hold a slot, so no cadence on the prefill side can raise the
resident set. The ceiling of that recipe is a few thousand resident slots
plus whatever prefill happens to be in flight.

`scripts/s26_fill_load.py` holds K concurrent streams of long UNIQUE
context (unique high-entropy prefix, or `--enable-prefix-caching` serves
the filler from cache and the prefill collapses — the same trap
`soak_631_mixed_load` documents). Occupancy becomes `K * context_tokens`,
chosen rather than hoped for. A first probe against the live 430000 boot
took occupancy from 30.6% to 61% within two minutes at
`--context-tokens 100000`; note the char-to-token estimate runs light,
about 66k actual tokens per 100k requested, so scale the request up.

**Any future capacity claim should use this driver.** With
`max_running_requests=4` on this rig, four streams of ~110k tokens hold
~440k slots.

## 3d. SECTION 2.1 STEP A IS WRITTEN (ab3f3e6460), SHIPPED DARK

Since the transient is what blocks 600,000, I wrote the change that
attacks it rather than leaving it as a note. `_stream_wave` restores,
writes and releases ONE ROW BLOCK at a time, so the commit unit stops
being a layer and the term shrinks roughly as `1/blocks`.

* Drives the span substrate that was already built and tested.
* The enabling trick: a contiguous pool row RANGE selects scattered
  writes cheaply, because the plan enumerates slots ascending, so the
  rows inside a range are a contiguous SLICE and the payload slice comes
  along at the same offsets. Asserted, not assumed — an unsorted row
  tensor would pair the wrong payload with the wrong rows.
* The EXCHANGE is deliberately not blocked: that needs the global round
  count, and a rank-local one deadlocks the group while looking like a
  hang. This change touches only local backing and local writes, so no
  rank can diverge.
* Byte identity is structural (one job list, one order) and pinned by
  comparing the same flip at 1 and 4 blocks tensor by tensor.
* `SGLANG_FLIP_SEAM_ROW_BLOCKS` **defaults to 1**. It is unmeasured on
  metal; do not enable it in an acceptance run before an A/B.

## 4. OPEN RISKS AND TRAPS

* **`_pools_alias()` is rank-local**, and so is `SGLANG_FLIP_SEAM_WAVES`.
  If they ever disagree across ranks, the ranks call `_exchange` a
  different number of times. Bounded rather than silent (the liveness
  poll aborts the flip) but unchecked. Gets sharper as W rises.
  Documented in `_flip_waves`; not fixed.
* **Latent NCCL comm-init asymmetry.** `_exchange` is pairwise, with an
  early return when a rank has no ops, and there is no eager warmup on
  `flip_tp.device_group`. Today the first non-empty wave involves all
  three ranks so the communicator gets created there. A rank owning zero
  live slots could skip the first `batch_isend_irecv` while its peers
  enter it. One warmup collective would close this.
* **A detached finisher must not `git add -A`.** It swept my uncommitted
  tests into `06ef108394`. See `COMMIT_MISLABEL_NOTE.md`; second
  occurrence of the same shape.
* **Boot provenance.** `seam_scaling_reboot.py` replays the live
  process's env, so `SGLANG_BOOT_COMMIT` carries the OLD commit forward
  across reboots. Any boot after a code change reports a stale commit
  unless `--set-env SGLANG_BOOT_COMMIT <sha>` is passed. Do that.

---

## 5. WHAT I WOULD DO NEXT, IN ORDER

1. **Do not spend boots chasing 600,000 with 2.1b.** The arithmetic in
   section 3 is calibrated to a measured point and cross-checked; the
   ceiling is ~473,000. Confirm it with ONE high-occupancy step near
   465,000 and then stop.
2. **Section 2.1, the streamed seam** — it is the only route to the user
   floor. Its substrate (`commit_span`/`decommit_span`,
   `back_token_span`/`release_token_span`, `release_backing_span`/
   `restore_backing_span`) is built and tested. What is missing is the
   streamed loop, the GLOBAL round count for row-blocking (derive from
   the replicated plan, never rank-local — a rank-local count deadlocks
   and looks exactly like a hang), and decoupling
   `SGLANG_FLIP_SEAM_CHUNK_MIB` from `retain_handles`, which currently
   PARKS unmapped pages per-arena and defeats exclusive backing outright.
3. **Graph A/Bs (spec item 8)** remain untouched through ten successors.
4. Only then the final all-axes acceptance at whatever pool the seam
   supports.
