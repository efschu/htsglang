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

### Where HANDOFF_668's 601,233 came from

Two terms. It modelled the payload slope as `18 / W` MiB per 1000 live
slots, giving 1.13 at W=16; the payload actually saturates near W=8
because the widest wave still carries a layer of one's own plus one from
each peer, a W-independent floor. And it modelled the transient as a
60 MiB constant; it is pool-proportional and reaches 1821 MiB at 600k.
Neither error is visible from a single measurement, which is why the
number survived to be inherited.

---

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
