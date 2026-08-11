# HANDOFF 680 — #656 / #631 Route A, successor 36

Read `HANDOFF_679` §2 item 2 first: this shift is that one item, actuated.
Item 16's rebalance tier had a price (s35) and no actuator; it now has one,
and the actuator was **wired to the wrong hot path on the first attempt** —
which the shift caught on metal, in a live window, and fixed.

---

## 0. THE ONE-LINE STATE

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
the honest attribution is in §3c.

### 3c. WHAT THIS WINDOW PROVES AND WHAT IT DOES NOT

PROVEN, from the mechanism's own log with before/after driver readings:
the lender fires exactly on the card the water-fill nominates, at pressure,
bounded, from the cheapest tier, and lifts that card by ~460 MiB per lend at
moments no allocation would have armed the gate.

NOT PROVEN: that the improvement in the corridor MINIMUM is caused by it.
The trough is seam-internal (§1d) and a single window cannot separate a
shallower seam from a lighter load. The causal story is plausible — the seam
begins from a level the lender has kept higher, so it dips less far — and it
is exactly the kind of story this corpus has been burned by. Booked as
plausible-and-unproven; the falsifier is an A/B at
`SGLANG_CORRIDOR_REBALANCE=0` on the same load script, which is one boot.

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
5. `draft-weights` returns the arena's own count, not a measured NVML delta
   (HANDOFF_677 §3a). Still latent. Note the lender never spent it in this
   window, so the defect stayed out of the item-16 numbers.

---

## 5. PROCESS NOTES

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
