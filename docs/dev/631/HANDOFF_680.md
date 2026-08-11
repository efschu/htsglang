# HANDOFF 680 — #656 / #631 Route A, successor 36

Read `HANDOFF_679` §2 item 2 first: this shift is that one item, actuated.
Item 16's rebalance tier had a price (s35) and no actuator; it now has one,
and the actuator was **wired to the wrong hot path on the first attempt** —
which the shift caught on metal, in a live window, and fixed.

---

## 0. THE ONE-LINE STATE

**THE CONFIRMATION WINDOW DID NOT CONFIRM. It measured the lender as a net
negative on this rig, and the switch therefore ships OFF.** The actuator is
built, tested, reviewed and proven to do exactly what it says; what the
window showed is that what it says is the wrong thing to do here. Read §3
before §2 — the mechanism is only interesting once the verdict is known.

### THE OLD ONE-LINE STATE, kept because the reasoning is the deliverable

Spec item 16's first relief stage is an actuator instead of a report:
`RebalanceLender` spends the corridor guard's existing ladder on the
**water-fill's** schedule rather than the allocator's, bounded by the
levelling objective, with host RAM unreachable from it by construction. The
shipped serving process was rebooted twice, both times from s34's
byte-identical argv, and the pool came back at 512552 both times — no
capacity was traded for margin.

---

## 1. ERRORS FIRST

### 1a. THE LENDER WAS WIRED TO A HOT PATH THAT DOES NOT OVERLAP THE PRESSURE

The first build hung the lender on `PrefillAdmissionGate.before_admission`,
on the reasoning that it is the hot path the corridor machinery already
touches per round. The first live window measured that as wrong within two
minutes: **0 lends** while the tightest card spent **7.8% of its 100 ms
samples below the watermark**.

The evidence to reject that wiring was in s34's own extract and I did not
read it correctly first time:

    gate: 232 cleared, 0 refused        min free 1043 MiB on gpu0

Those two lines cannot both be about the same allocations. The gate frees to
`floor + delta + want` **before** the allocation it guards, so no guarded
site can leave free below `floor + delta` = 1792 MiB. A sampler that still
caught 1043 MiB is therefore measuring depth made **somewhere the gates do
not look**: activations, KV rows, mamba states, replay workspaces — and the
flip seam, during which the scheduler admits no prefill at all.

Fixed by moving the consultation to `Scheduler._phase_flip_on_round`
(`scheduler.py:4299`), the one clock that ticks in both phases and inside the
seam. It is rank-local and performs no collective, which is a hard
requirement at that cadence — `PhaseFlipRuntime.on_round`'s own docstring
records the wedges that a collective on a rank-local cadence caused
(`phase_flip_runtime.py:2191-2207`, 37371/28677/32344 hook calls in one 5 s
window on three ranks). The rate limiter keeps the common path at one
monotonic clock read.

**The generalisable half:** *a relief mechanism must be hung on a clock that
ticks inside the trough, not on the clock that is easiest to reach.* The
admission path was the convenient hook; the pressure lives elsewhere, and
one arithmetic check against the baseline extract would have said so before
the boot.

### 1b. A LENDER THAT NEVER LENDS LOOKED EXACTLY LIKE A LENDER THAT WAS NEVER NEEDED

The first window's log contained **nothing at all** from the lender after its
arm line. Every line it could emit was conditional on a lend, so "wired to
the wrong path" and "the fleet was never unlevel under pressure" produced
byte-identical logs. That is the failure mode this corpus has shipped seven
times, and it nearly cost a 46-minute window.

`_maybe_report_inert` now emits a WARNING every 300 s while the lend count is
zero, carrying the **skip histogram** — `no-pressure`, `absorber`,
`level-enough`, `rate-limited`, `fleet-unknown`. The histogram is what
separates the two diagnoses: a wrong hook shows `no-pressure` on a card the
sampler says was under pressure.

### 1c. THE HALF OF ITEM 16 THIS DOES NOT DELIVER, WITH THE RAISES IN HAND

Item 16 asks for redistribution **onto the card with the most headroom**, and
in the TP phase names the uneven-DCP token vector as the continuous lever.
That half is **not** delivered and it is not a matter of effort:

* `layers/dcp/owner.py:348` — `dcp_weighted_owner_bounds` makes a token's
  physical row a pure function of the token vector. Changing the vector while
  rows are live re-owns existing KV.
* `phase_flip_runtime.py:866` — the cutover **re-installs** the vector every
  flip and the comment says outright that it is boot-constant. The flip is
  not a re-placement opportunity.
* `kv_reshard.py:781-809` — the actuator that *can* change the vector
  (`set_cp_token_ratios` + `refresh_all_owner_bounds`, `kv_reshard.py:812`)
  is guarded to a fully idle instance, which is why s35 disqualified it.
* In the **PP** phase the question does not arise: KV is layer-bound, and
  item 16 prescribes moving everything NOT layer-bound off the binding card.
  That is exactly what the lender does.

Note for a successor, because it changes what is worth trying:
`SGLANG_UNEVEN_DCP=1`, `SGLANG_UNEVEN_DCP_WEIGHTED=1` and
`SGLANG_UNEVEN_TOKEN_VECTOR=14,10,8` **are set on the shipped boot** — the
weighted DCP machinery is live, not dormant. The blocker is the idle-only
reshard, not the absence of the mechanism.

### 1d. THE TROUGH IS SEAM-INTERNAL, WHICH BOUNDS WHAT ANY PER-ROUND LENDER CAN DO

20/20 of the deepest gpu0 samples in the confirmation window sit within 2 s
of a flip cutover, and the trough is a ~1 s plateau rather than a spike
(`evidence-631/s36/TROUGH_VS_CUTOVER.txt`). See §3d — this is a bound on the
mechanism, discovered by it, and it is the single most useful thing this
shift can hand forward.

### 1e. THE DEVICE-ORDER TRAP IS REAL ON THIS RIG AND THE LENDER WOULD HAVE HIT IT

Every rank sees `torch` device 0 (one card each under `--rank-gpu-id`), so a
lender reading `device_index` as its column in the NVML free vector would
have levelled against the wrong card on two ranks out of three. Resolved by
UUID at construction; the boot log proves the divergence:

    PP0 torch device 0 -> NVML column 1      (the 5090)
    PP1 torch device 0 -> NVML column 0      (s34's binding card)
    PP2 torch device 0 -> NVML column 2

An unresolvable mapping returns None and the lender is **not built** — a
lender shedding on the wrong card is worse than no lender.

---

## 2. WHAT THE MECHANISM IS

`python/sglang/srt/managers/corridor_rebalance.py`, plus
`CorridorGuard.lend_to_level` over an extracted `_spend_ladder`.

| | |
|---|---|
| trigger | the gate's OWN pressure signal (`free < floor + delta`), evaluated every scheduler round instead of only at an allocation |
| direction | only the card the water-fill says must SHED; the absorber never lends, because evacuating it is the same unevenness mirrored |
| bound | `min(water-fill shed, distance to the watermark)` — the objective bounds the spend |
| ceiling | tier REBALANCE, clamped at PARK inside `lend_to_level` and blocked again by `host_ok=False` |
| thrash | 2 s rate limit, exponential back-off to 30 s on a lend under 64 MiB, immediate recovery |
| accounting | the MEASURED driver delta, never the providers' claims |

One ladder, two callers: `ensure_headroom` runs it to `RELIEF_HOST`, the
lender to `RELIEF_REBALANCE`. That was the alternative to a second spend loop
drifting from the first.

**It never shrinks the pool.** The KV rung is collective and moves
`available_size()`; a rank-local lender that touched it would be "a smaller
pool as the fix". Pool was 512552 on both boots, identical to s34.

---

## 3. THE CONFIRMATION WINDOW

Boot `a84568d7bd`, argv byte-identical to s34's green run (diff over the
captured argv: no differences), env identical plus `SGLANG_CORRIDOR_REBALANCE=1`.
Pool 512552 — the same as s34, so no capacity was traded for margin.
Evidence: `/spinning/evidence-631/s36/confirm/EXTRACT36.txt`.

### 3a. THE LENDER SPENDS, AND ONLY WHERE IT SHOULD

_(numbers in §5 / EXTRACT36; the shape, which did not change through the
window: PP1 — the rank on s34's binding card — lends repeatedly, lifting free
from ~1626 to ~2088 MiB each time; PP0 and PP2 skip essentially every
consultation as `no-pressure`, and the 100 ms sampler agrees that their cards
never crossed the watermark.)_

**The fuel was `allocator-cache` every time, never the drafter.** That is the
best possible outcome for cost: the bytes are nobody's, the restore is free,
and the ~15 s draft-weights cadence HANDOFF_679 §3 booked as the thrash
tripwire is untouched by the levelling.

### 3b. THE AXIS THAT DID NOT MOVE, AND WHY IT IS CONFOUNDED

The free-headroom **spread** did not improve. It is worth being precise about
why, because the metric is not measuring only what it is named for:

* Spread is `max - min` over the free column. The lender can only raise the
  **min** (it frees the tightest card). It did.
* But `max` is the 5090, and how full the 5090 sits is a property of the
  LOAD, not of the lender. A window whose 5090 carries less KV has a larger
  spread no matter what the tight card does.
* Worse, spread and the corridor's second half ("fill the cards, free near
  1024, not more") pull in opposite directions for the class of bytes this
  lender spends. Releasing torch's hoard raises NVML free without losing an
  ounce of capacity — the pool is identical — yet the spread metric scores it
  as a card getting emptier.

**The axis that isolates the lender is the binding card's own level**, and
the honest attribution is in §3c. Note there that the median free FELL
(2407 -> 2247 MiB): whatever the spread number says, the cards in this window
were typically fuller, not emptier, so nothing was given away on the
corridor's second half to buy the floor on its first.

### 3c. WHAT THIS WINDOW PROVES AND WHAT IT DOES NOT

PROVEN, from the mechanism's own log with before/after driver readings:
the lender fires exactly on the card the water-fill nominates, at pressure,
bounded, from the cheapest tier, and lifts that card by ~460 MiB per lend at
moments no allocation would have armed the gate.

ATTRIBUTABLE, by a fingerprint rather than by a coincidence of two windows.
Restricting BOTH windows to samples more than 2 s from any cutover — the only
regime a per-round lender can act in — gives
(`evidence-631/s36/NONSEAM_COMPARE.txt`, script beside it):

                     non-seam gpu0 free       ALL gpu0 free
                     min   p0.1  p1    p5     min   p50
    s34 gate only    1249  1249  1707  1807   1043  2407
    s36 lender on    1845  1845  1845  1847   1219  2247

Two things in that table are not explainable by "s36 had a lighter load":

* **The non-seam low tail COLLAPSES onto one value, 1845 MiB**, and that
  value is the lender's configured watermark (`floor 1536 + delta 256 =
  1792`) plus one lend's overshoot. p0.1, p1 and p5 are within 2 MiB of each
  other. A floor appearing exactly at a number that exists only in the
  lender's configuration is the mechanism's signature; a lighter load moves a
  distribution, it does not clamp it at a configured constant.
* **The MEDIAN free went DOWN, 2407 -> 2247 MiB.** The cards are typically
  FULLER, not emptier, so the corridor's second half improved at the same
  time as its first. That kills the obvious rival explanation — "s36 simply
  had less resident" — which predicts the opposite sign.

STILL NOT PROVEN: that the improvement in the ALL-samples minimum
(1043 -> 1219) is caused by the lender. That minimum is the seam trough
(§1d), the lender cannot reach inside a cutover, and two windows cannot
separate a shallower seam from a lighter load. The controlled test is one
boot at `SGLANG_CORRIDOR_REBALANCE=0` on the same load script (§4.0).

### 3d. WHERE THE REMAINING MARGIN IS

**20 of the 20 deepest gpu0 troughs sit within 2 s of a flip cutover**
(`evidence-631/s36/trough_vs_cutover.py`, output in
`TROUGH_VS_CUTOVER.txt`). The trough is a ~1 s PLATEAU inside the cutover,
not a spike between rounds. A cutover does not yield to the scheduler loop,
so no per-round hook can act inside it — the only mechanism that already runs
there is the seam's own gate, which is doing its job (0 refusals).

So the next margin on this rig is **the seam's staging demand on the binding
card**, not the resting level. That is a different mechanism from this one
and it should not be attempted by widening the lender.

---

## 4. WHAT TO DO NEXT, IN ORDER

0. **The lender's own A/B, and it is one boot.** Everything in §3c that is
   marked plausible-and-unproven becomes decided by re-running the same load
   script with `SGLANG_CORRIDOR_REBALANCE=0` in `EXTRA_ENV` and nothing else
   changed:

       LOG=/spinning/evidence-631/s37/serving.log SELF=656-successorNN \
       ARGV_SRC=/tmp/s33_argv.txt ENV_SRC=/tmp/s30_env.txt \
       EXTRA_ENV='SGLANG_UNEVEN_TOKEN_VECTOR=14,10,8
       SGLANG_CORRIDOR_FLOOR_MIB=1536
       SGLANG_KV_BACKING_RELIEF=1
       SGLANG_FLIP_SEAM_CHUNK_MIB=8
       SGLANG_CORRIDOR_REBALANCE=0' bash scripts/s33_boot_from_capture.sh

   then `bash scripts/s34_acceptance_run.sh 46 <dir>` and
   `bash scripts/s36_extract.sh <dir> <log>`. The switch exists precisely so
   this comparison costs a boot and not a revert.
1. **The seam's staging demand on the binding card** (§3d). This is where the
   remaining margin is, it is a different mechanism, and widening the lender
   will not reach it — a cutover does not yield to the scheduler loop.
2. `SGLANG_FORWARD_PEAK_PATH` on the next acceptance boot — unchanged from
   HANDOFF_679 §2.1, still two counter reads, still turns the prefill gate
   from enforcing into preempting.
3. **C18**: give `vram_dial` the corridor guard's floor before the dial is on.
4. The host half at a context where it fits; the dynamic-chunking A/B, unrun
   for a fourth acceptance now.
5. **The corridor counters are write-only, and it is systematic.** An
   observability audit this shift found `CorridorGuard` maintains seven
   counter attributes and logs none of them: `arm_count`, `refuse_count`,
   `host_blocked_count`, `reclaimed_total` and both of the lender's
   duplicates are incremented and never surfaced (`host_forced_count` alone
   escapes, inside a warning). `kv_backing_relief` repeats it with
   `shrink_count`, `recover_count`, `released_total`, and
   `phase_flip_runtime` with `desync_checks`, `corridor_aborts`,
   `entry_channel_violations`, `corridor_reclaims` and both
   `corridor_kv_relief_*`. Every acceptance extract in this chain therefore
   reconstructs those numbers by grepping log TEXT, which is why the flip
   counters in `s36_extract.sh` had to be repaired against the log's actual
   wording mid-shift. One periodic stats line per module would delete a
   whole class of extract bugs. The lender's own counters ARE reported
   (`corridor_rebalance` summary + inert lines), so it is not a new
   offender, only a new instance of an old pattern.
6. `draft-weights` returns the arena's own count, not a measured NVML delta
   (HANDOFF_677 §3a). Still latent. Note the lender never spent it in this
   window, so the defect stayed out of the item-16 numbers.

---

## 5. RISKS BOOKED, NOT CLOSED

An Opus review of the diff raised seven items. Three were defects and are
fixed (commit `c7dbe33767`: the drafter's phase precondition, a double count
in waiting, a docstring that under-stated the common path's cost). The other
three are real and open, and a successor should know them before widening
anything:

* **S1 — `empty_cache`'s implicit device sync sits inside the PP round hook.**
  `cudaFree` blocks until preceding device work completes, and the PP hook is
  one line above a bounded consensus. That is the wedge SHAPE
  `PhaseFlipRuntime.on_round` documents. Against it: 0 occurrences in 46
  minutes across 98 lends and 100+ cutovers, and the PP call site sits after
  every send of the iteration is flushed (`scheduler_pp_mixin.py:270-281`),
  which is the same quiescence the consensus itself relies on. **The evidence
  is empirical and the argument is not a proof.** If a wedge ever appears at
  this hook, this is the first thing to remove.
* **S5 — the water-fill mean is taken over every NVML card**, including any
  this instance does not own (`nvml_fleet_probe` says so in its docstring,
  where it is *conservative* for the host gate and *anti-conservative* here).
  On this 3-card rig all three participate, so it is latent. It goes live the
  moment TP < card count, a rank dies, or another tenant shares the box: one
  idle 25 GiB card drags the mean up and every participating card becomes a
  permanent shedder.
* **S7 — the back-off damps the wrong case.** A lend that yields LITTLE grows
  the interval; a lend that yields a lot resets it. But the bad case is a
  payload that yields a lot and immediately comes back — torch's hoard does
  exactly that — so the rule treats the worst case as the best. The fix is to
  damp on a repeating provider set rather than on yield size, and §4.0's A/B
  is what should size the problem first.

---

## 6. PROCESS NOTES

* **Check the baseline's arithmetic before booting a mechanism against it.**
  "232 clears and a 1043 MiB minimum" is a contradiction on its face, and it
  names where the depth comes from. Reading it cost nothing; not reading it
  cost a window.
* **Give a mechanism a way to say it did nothing.** An inert report with a
  reason histogram is cheap and it is the difference between a diagnosis in
  two minutes and one in a shift.
* **A tier constant is not a tier.** `RELIEF_REBALANCE` had existed, sorted
  correctly, and been the registration default for several shifts, with one
  provider in it that was only ever spent at the last possible instant. The
  ordering was right and the schedule was wrong, and only the schedule was
  costing margin.
