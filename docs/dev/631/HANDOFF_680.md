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

### 1f. THE CONTROL ARM CRASHED, AND IT WAS NOT THIS FEATURE

14.5 minutes into the lender-OFF arm, every rank aborted inside the cutover:

    phase_flip_resident_carry.py:674  <- _cutover:1176 <- _execute:4349
    TypeError: 'NoneType' object cannot be interpreted as an integer

`reseed_decode_input_relay` selects carried requests on `output_ids` alone,
then builds an int64 index tensor from `req_pool_idx`. A request whose pool
slot was released still has output, so `None` reaches the tensor build --
inside the no-return region, so all three ranks go down together. Health 000,
179 soak errors.

**Fixed** (`bd34ac2b0c`): those requests are skipped, because the relay is
slot-indexed and a request with no slot has nothing to reseed, and they are
logged rather than dropped quietly. The regression test is mutation-checked
against the exact production TypeError.

It fired with the lender OFF and never fired in the 46-minute lender-ON
window, so it is a latent defect of the flip path that this A/B exposed, not
a cost of the feature. Serving was rebooted on the fix and is up.

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

## 3. THE CONFIRMATION WINDOW: IT FALSIFIED THE FEATURE

Boot `a84568d7bd`, argv byte-identical to s34's green run, env identical plus
`SGLANG_CORRIDOR_REBALANCE=1`, pool 512552 (same as s34 -- no capacity was
traded). 46 minutes, real agent load, both occupancy legs and a YaRN leg.
Evidence: `/spinning/evidence-631/s36/confirm/`.

### 3a. THE SCORE AGAINST THE BRIEF'S OWN CRITERIA

| criterion | s34 baseline | s36 lender on | verdict |
|---|---|---|---|
| (a) 0 corridor breaches | 0 | **12** | **FAILED** |
| (b) binding margin above +19 MiB | +19 | **-23** | **FAILED** |
| (c) spread improved | 2409 mean | 2398 mean | NOT MET (unmoved) |
| (d) flips both directions | 321/321 | 390/384 | held |
| (d) gate refusals / tracebacks | 0 / 0 | 0 / 0 | held |
| (d) KV rung able to fire | 21 shrinks | 114 shrinks | fires, 5x more |

The 12 breaches are ONE 1.5-second event at 16:07:14, on gpu0, inside a
`pp_to_tp` cutover -- the same seam-internal trough §1d localised. The gate
had done its job two seconds earlier (`want 1548, free 2402 -> 3050`); the
cutover then consumed about 500 MiB more than the gate had priced.

**s34's identical trough landed at +19 MiB and this one at -23.** The margin
at that instant was always approximately zero, and the green run passed on
luck rather than on headroom. That finding outlives this feature.

### 3b. WHY IT HURT, WHICH IS THE PART THAT GENERALISES

The only provider the lender ever spent was `allocator-cache`: 98 lends,
every one of them, never the drafter.

**Torch's allocator cache is the reason most allocations are invisible to the
corridor law.** An allocation served from cache does not move NVML's free
column at all. Dumping the hoard converts those into driver allocations, so
the very column the law is written on begins to move for work that used to
be free. The same dump shrinks `cheap_relief` at the seam, and the KV rung's
deficit discounts against exactly that term -- which is why the rung fired
five times as often for a pool that did not change size.

So the lender was not spending slack. It was destroying the buffer that made
the corridor look calm, and then reporting the resulting driver-side free as
an improvement.

### 3c. THE GAIN WAS AN ARTEFACT, AND I REPORTED IT AS A FINGERPRINT

Mid-shift I found the non-seam floor collapsing onto 1845 MiB where s34 sat
at 1249, argued that a floor appearing at exactly the lender's configured
watermark could not be load variance, and called it attribution.

The reasoning was right and the conclusion was wrong. It IS the lender's
fingerprint -- of the instrument clamping a number, not of the rig gaining
usable memory. A metric that improves because a mechanism holds it up is the
oldest trap in this corpus, and I walked into it with the argument for why I
had not. The median free falling at the same time (2407 -> 2247) genuinely
did rule out "lighter load"; it did not rule out "the number is being held".

**The lesson is narrower than "be careful": a free-memory metric cannot
validate a mechanism whose action is to free memory.** The axes that could
falsify it were the ones with independent meaning -- breaches, completions,
decode batches -- and two of the three were already in the extract.

### 3d-i. THE CONTROLLED ARM SETTLED IT: THE LENDER CAUSED THE STARVATION

`scripts/s36_ab_lender_off.sh` booted the same commit with the switch off and
refused to proceed unless the arm line was absent. Judged at matched elapsed
time by `scripts/s36_ab_judge.sh` (s34's own counters accelerate through its
window, so totals across unequal runs say nothing):

                          s34 gate only   s36 lender ON   s36ab lender OFF
    decode batches/min         15.5             7.0             14.1
    median PP dwell            17.0s           26.0s            17.0s
    median TP dwell             7.0s            4.0s             4.5s
    soak ok at t+14              30              15               21
    corridor breaches             0              12                0

**PP dwell returns to exactly s34's 17.0 s and the decode rate to within 10%
of it.** The instance was not sitting in prefill because of its load; it was
sitting in prefill because the lender was dumping the allocator cache under
it. Everything §3b predicted from the mechanism, the control arm measured.

The control arm ran 14.5 of its 25 minutes before crashing (§1f) -- past the
t+14 comparison, so the verdict stands on complete data, and its 25-minute
column is a shorter sample than the others by construction.

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

0. **DONE THIS SHIFT -- the A/B ran and decided it (§3d-i).** Kept here only
   as the recipe, because it is the cheapest instrument in this corpus and
   the next feature with an off switch should use it on day one:

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
0b. **Re-run the acceptance on the shipped config to restore a GREEN stamp.**
   The last full acceptance is s34's. This shift's two windows were a feature
   trial and its control, and neither is an acceptance: one breached, the
   other crashed at 14.5 min on a defect now fixed. Serving is up on
   `bd34ac2b0c` with the lender off, which IS s34's behaviour plus that fix,
   so the stamp should be cheap -- but it is not yet taken, and nobody should
   report #656 as green until it is.
0c. **Consider loosening the drafter's phase precondition (§1f-adjacent).**
   It refuses outside PP, which also blocks the `tp_to_pp` seam where s34
   spent `draft-weights` 212 times and where the spill is safe (the drafter
   goes idle immediately after). The control arm ran 14.5 min with 0 gate
   refusals and 0 drafter spends, so the relief was not missed there -- but
   that is 14 minutes of evidence, not a proof.
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

* **A free-memory metric cannot validate a mechanism whose action is to free
  memory.** §3c. The corollary is cheap and I should have applied it before
  the window, not after: pick the falsifying axis from the set the mechanism
  does NOT touch. Here that was breaches, completions and decode batches --
  all three already produced by the existing extract.
* **The control arm cost 30 minutes and changed the answer.** Two windows on
  two loads had me attributing a gain with a clever argument; one boot with
  the switch off settled it. When a feature has an off switch, running it is
  never the expensive option. Build the switch first, for that reason.
* **Report the mechanism's own log, not only the outcome metric.** "98 lends,
  all from allocator-cache, never the drafter" is what identified the harm.
  The outcome metrics said the feature worked.

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
