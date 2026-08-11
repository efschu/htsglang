# HANDOFF 681 — #656 / #631 Route A, successor 37

Narrow shift, two objectives, both closed: **register C20** (the seam-internal
trough that made s34's green margin luck) and a **fresh acceptance stamp** on
the tree that carries the fix.

---

## 0. THE ONE-LINE STATE

**#656 HAS A FRESH GREEN ACCEPTANCE STAMP AGAIN, on `bde9d01c51`, and the
seam now has a DESIGNED entry margin instead of a lucky one: the binding
card's IN-CUTOVER minimum is +99 MiB over the law where s34's was +19, with
0 breaches in 28568 samples, 348 flips each way, strict purity, 99.7% decode
graphs, MTP at 2.797, both YaRN legs above 262144, and the pool back at its
boot 512552.**

Read §1d before §3: the first cut of this feature shipped a wedge and a
breach path, a review caught both, and the stamp is on the FIXED tree.

Read §2b before quoting +99 as a win. The entry condition works — the draw
from a low entry more than halved, 456 -> 184 MiB — but a cutover entering
HIGH still draws ~1040 MiB that `_staging_bytes` does not price, and that is
what now sets the floor. C20 therefore moves from "the depth is made where no
gate looks" to "the depth is bounded by the seam's UNPRICED TAIL DEMAND",
which is a different mechanism (§4.5), not a closed one.

---

## 1. ERRORS FIRST

### 1a. THE MARGIN FLOWS UNBOUNDED INTO THE COLLECTIVE KV RUNG, AND AT A LARGE VALUE THAT KILLS THE INSTANCE

`scripts/s37_delay_probe.sh` boots with `SGLANG_SEAM_ENTRY_MARGIN_MIB=8192` —
a margin no ladder on this rig can fund — because the acceptance window funds
the margin on every seam and therefore never executes the two branches that
make the mechanism SAFE. It proved them, and it found this:

    C20 entry margin (armings)   452
    seam entry DELAYED            72     <- the branch runs on metal
    seam entry margin YIELDED     35     <- the budget bounds it, on metal
    corridor gate refused          0
    corridor breaches              0     <- the falsifier holds
    health                       000     <- AND THE INSTANCE IS DEAD

    RuntimeError: Out of memory. Try to allocate 512 tokens.
    Available full tokens: 0 (full_available_size=0 + full_evictable_size=0)
    scheduler.py:5429 _get_new_batch_prefill_raw -> alloc_for_extend
    all three ranks, 19:25:15

**The delay and yield branches are not the cause and the timeline proves it:**
first delay 19:23:03, first yield 19:23:09, first traceback **19:25:15** —
two minutes later, after 42 cutovers.

The cause is that `_corridor_gate` passes `staging + margin` to
`collective_kv_backing_relief` as `want_bytes`. The rung's deficit is
`floor + delta + want - free - cheap_relief`, so an 8 GiB margin makes the
deficit enormous on every seam and the rung shrinks KV backing to its floor.
That floor is `max_live` rows — it protects data that EXISTS and reserves
nothing for admission, so `available_size()` reaches 0 and the next prefill
raises inside the scheduler loop.

**At the shipped 512 MiB none of this happens**: 348 shrinks, every one
recovered, the pool never left 512552, and the acceptance is green. But three
things follow and a successor should not have to rediscover them:

1. **The margin is an unbounded lever on a collective that spends admission
   capacity.** An operator typo of one digit takes the instance down. A cap on
   what the margin may add to the RUNG's ask (the guard's own ladder can stay
   uncapped — it only spends local tiers and can refuse safely) is the obvious
   guard, and it is deliberately NOT in this commit: it would need its own
   window to validate, and shipping an unvalidated fix at the end of a shift
   is how this corpus got most of its retractions.
2. **The KV rung's floor is the wrong floor.** `max_live` keeps live rows and
   leaves nothing to admit with. That is a pre-existing property C20 merely
   made reachable, and it is a better fix than capping the margin.
3. **A pathological configuration produced ZERO corridor breaches.** The gate
   delayed, then yielded, and never walked a seam into a breach — which is
   exactly what `TestNoPathProceedsIntoABreach` pins on CPU, now observed on
   metal under the worst input the switch accepts.

### 1a. THE WORST-CASE SIZING WAS THE OBVIOUS DESIGN AND IT WAS s36's MISTAKE

The first sizing I computed for the seam-entry requirement was
`law + margin + observed draw`, with the draw taken at the p90 of what a
cutover consumes on the binding card (786 MiB, max 1026). It is the
requirement the brief's words point at and it is wrong on this rig:

    gpu0 need 1910 MiB at entry -> would arm on 348/450 cutovers (77.3%)

A relief that arms on three cutovers out of four, at ~10 cutovers a minute,
is successor 36's continuously-dumping lender with a different call site.
His measurement already priced that shape: 98 lends in 46 minutes halved the
decode rate. Mine would have been three times as frequent.

What made a cheap design possible was measuring the draw **conditioned on the
entry level** instead of marginally:

    gpu0 entry<1524 (n=19):  draw p50=0  MAX=456
    gpu0 entry<1824 (n=252): draw p50=0  MAX=456
    (the 1026 MiB draws all follow entries above 2400)

**The draw is self-limiting**, because the gate at the seam frees to
`floor + delta + want` before staging: from a high entry the cutover consumes
what it had, from a low entry it consumes almost nothing. So the requirement
only has to cover 456 MiB, not 1026, and it arms rarely.

The generalisable half: *a marginal distribution of a quantity that responds
to pressure will overstate what a mechanism has to reserve.* Condition on the
state the mechanism will actually see.

### 1b. THE DEPTH IS INHERITED, WHICH IS NOT WHAT "SEAM-INTERNAL" IMPLIED

s36 established that the deepest samples sit inside a cutover. The natural
reading — the cutover digs the hole — is not what the data says. The 15
deepest gpu0 minima come in PAIRS about 2 s apart:

    13:59:11  entry 1499  draw 456  -> min 1043
    13:59:13  entry 1043  draw   0  -> min 1043
    13:42:04  entry 1367  draw 304  -> min 1063
    13:42:05  entry 1063  draw   0  -> min 1063

The second cutover of each pair **enters at the first one's trough** and
draws nothing at all. The memory has not come back yet and nothing was
checking. That is why the fix is a condition on ENTRY rather than a cap on
the draw, and it is why a DELAY is an actuator at all: the level recovers
(median entry 1807 MiB), it just had not recovered two seconds later.

### 1c. THE KV RUNG'S FIRING RATE CHANGES BY AN ORDER OF MAGNITUDE

Asking the rung to fund the margin as well as the staging (which it must, or
the funder of last resort declines exactly the gap the gate is about to delay
for) multiplies its work:

    s34   21 shrinks in 65 min   (0.32/min)
    s37  348 shrinks in 65 min   (5.4/min, 17x)

Every one recovers — the `instead of 512552` figure never moves off the boot
pool, which is the line that separates a spill from a capacity loss — and
spec item 12 asks for exactly this ("KV itself is a spill class"). But it is
a real behavioural change and the next shift should know it was chosen, not
stumbled into. If it ever needs undoing, the one-line version is to price the
rung at `staging_bytes` while the guard is priced at `staging + margin`; the
cost of that is that the pp->tp leg loses its most capable funder and starts
spending its delay budget.

### 1d. THE FIRST CUT OF THIS FEATURE HAD A WEDGE AND A BREACH IN IT

An adversarial review of the first commit (`2f3edccaf7`) found two defects
that falsified its two load-bearing claims, and a live window was already
running on that code when it landed. Both are fixed in `bde9d01c51`, both
have mutation-checked regression tests, and both are worth carrying forward
because neither is specific to this feature.

**The delay budget was per-rank; the abandon is group-OR.** The counter was
incremented on the rank that was short and reset whenever that rank's own ask
cleared. But the group abandons if ANY rank objects, so three ranks taking
turns being short refund each other's budgets and nobody ever reaches a
yield. Reproduced on stubs: 30 attempts, 10 delays per rank, **zero yields**
— pp->tp delayed indefinitely, which under strict purity is exactly the
411-abandon decode wedge the budget existed to prevent.

> *A budget denominated in a rank-local event cannot bound a group action.*
> The currency has to be something every rank reads the same. Here that was
> free: the reduced fit verdict is already identical on all three ranks, so
> booking consecutive GROUP abandons costs no collective at all.

**An exception on the law-only re-ask turned a refusal into a proceed.** The
first cut answered "margin short or law short?" by calling `ensure_headroom`
a second time, with `except: return ""`. That discards a verdict which had
ALREADY said the law would break — the one path in this gate that could walk
a seam into a corridor breach. It also re-ran the whole ladder on the refusal
path: a second `empty_cache`, a second forced host spill, `refuse_count` and
`host_forced_count` double-booked per seam.

The re-ask is gone. The guard's contract is
`ok = (free_after - want) >= law_floor`, so subtracting the *staging* from the
free the ladder actually reached answers the same question from a value
already in hand. **Arithmetic on a value you hold cannot raise and cannot
spend.** A second query where a subtraction would do is not a neutral choice.

Two smaller ones from the same review, both about being able to READ the run:

* a margin delay was logged by the group as `FLIP ABANDONED`, the string every
  acceptance harness in this corpus counts and the one the wedge was measured
  with. A `margin_only` vote now rides the same reduction so the group can say
  `FLIP DELAYED` when no rank objected for any other reason;
* liveness was counted by grepping the margin's reason string, which only
  reaches the log when the guard ARMS — so a run that funded the margin from
  cache every time would have been reported **inert**. That is this corpus's
  favourite false negative, inverted. The gate now announces itself once per
  process on a path that always runs, and the ask count is reported as the
  margin's COST next to the KV-rung shrink count.

### 1e. THE FIXTURE IS 16 LAYERS, NOT 20

The brief (and therefore probably the next one) says "the 20-layer fixture".
There isn't one. The canonical CPU flip fixture is `N_LAYERS = 16` with
`MAP_625 = ((0..7), (8..11), (12..15))`, defined in
`test/registered/scheduler/test_phase_flip_runtime.py:52` and two siblings.
"20" is `#631 boot 20`, cited at `test_phase_flip_runtime.py:1247`.

---

## 2. THE ACCEPTANCE WINDOW

Boot `bde9d01c51`, argv byte-identical to s34's green run, env identical plus
`SGLANG_SEAM_ENTRY_MARGIN_MIB=512` and `SGLANG_SEAM_ENTRY_DELAY_BUDGET=2`.
65 minutes, pool **512552** — the same as s34 and s36, so no capacity was
traded for the margin. Evidence: `/spinning/evidence-631/s37/accept2/`,
log `serving2.log`.

### 2a. THE VERDICT

**ACCEPTANCE: GREEN. Corridor HELD, 0 breaching samples in 28568.**

| | s34 (the standing green) | s37 (C20) |
|---|---|---|
| corridor breaches | 0 | **0** |
| per-card minima | 1043 / 1922 / 1541 | **1123 / 2118 / 1641** |
| **IN-CUTOVER minimum, binding card** | **1043 (+19)** | **1123 (+99)** |
| draw from a LOW entry, binding card | 456 MiB | **184 MiB** |
| flips | 321 / 321 | **348 / 348** |
| FLIP ABANDONED / tracebacks | 0 / 0 | **0 / 0** |
| strict purity (no prefill graph) | True | **True** |
| decode graph share | 99.2% | **99.7%** |
| MTP accept length | 2.850 | **2.797** |
| YaRN legs above 262144 | 271237 x2 | **271237 x2** |
| pool at the last KV proposal | 512552 | **512552** |
| KV rung shrinks | 21 | **348** |
| gate clears / refusals | 232 / 0 | **454 / 0** |

### 2b. THE HONEST READING OF +99

The brief asked for the +19 MiB luck margin to become a designed margin of
order 100 MiB, and that is exactly what it is: **+99, not +500.** Two things
moved and they are different claims:

* **The entry condition works.** The draw from a low entry fell from 456 MiB
  to 184, because the seam now enters with the ladder already spent for the
  margin. The trough is no longer made by a cutover walking into an
  unguarded low entry.
* **The tail draw is unchanged and now sets the floor.** The deepest event of
  this window is `18:29:48 entry 2221 draw 1040 -> min 1181`, followed 2 s
  later by an entry at 1181 that draws nothing. A cutover entering HIGH still
  consumes ~1040 MiB, and `_staging_bytes` does not price that.

So C20 moves from "the depth is made where no gate looks" to "the depth is
bounded by the seam's UNPRICED TAIL DEMAND". That is a different mechanism
— staging pricing, `SGLANG_FORWARD_PEAK_PATH` — and §4 carries it. Anyone
reporting C20 as fully closed is over-reading this run.

### 2c. THE AXES THE MARGIN DOES NOT TOUCH (four arms, matched 25 min)

| | s34 gate only | s36 lender ON | s36ab lender OFF | **s37 C20** |
|---|---|---|---|---|
| prefill batches | 14830 | 16629 | 9012 | **14764** |
| decode batches | 387 | 174 | 204 | **345** |
| pp->tp cutovers | 252 | 189 | 156 | **270** |
| median PP dwell | 17.0 s | 26.0 s | 17.0 s | **17.0 s** |
| soak ok at t+14 | 30 | 15 | 18 | **27** |
| soak ok at t+24 | 59 | 21 | 25 | **41** |

**s37 sits in s34's family, not s36's.** PP dwell is identical to the digit,
prefill is at parity, decode is 89% of s34 and it ran MORE flips. The margin
is not the lender's mistake at a different call site.

The one number that is genuinely down is soak completions at t+24 (41 vs 59,
69%), and the full window closed at 196 ok / 0 err. Do not launder that: the
KV rung fired 348 times against s34's 21, and §4.1 is the A/B that would
price it properly rather than argue about it. Every shrink recovered, no
recovery came up short or was deferred, and the pool never left 512552 --
so it is residency, not capacity loss, which is what spec item 12 asks for.

### 2d. THE GATE'S OWN ACCOUNT

    ARMED announcements (one per rank per boot):  3
    armings that had to spend for the margin:     911
    seams DELAYED for the margin:                 0
    seams entered on the law (budget spent):      0
    seams REFUSED (below the law):                0

**The margin was funded on every one of 696 cutovers, so the DELAY and YIELD
branches never executed in this window.** That is the outcome the feature
wants and the worst possible coverage of the two branches that make it safe,
which is why `scripts/s37_delay_probe.sh` exists and was run separately.

### 2e. THE PROBE: BOTH SAFETY BRANCHES, ON METAL

72 delays and 35 yields under a deliberately unfundable 8 GiB margin, with 0
gate refusals and **0 corridor breaches** — the CPU falsifier reproduced on
the rig under the worst input the switch accepts. The same probe killed the
instance by a different route two minutes later; that is §1a and it is the
most useful thing this shift found.

Evidence: `/spinning/evidence-631/s37/delay-probe/` and `delay-probe.out`.
It is a PROBE, not an acceptance: an impossible margin is not a shipping
configuration and its corridor numbers describe that margin, not this one.

---

## 3. WHAT THE MECHANISM IS

`PhaseFlipRuntime._corridor_gate`, one extra term and a graded verdict. No new
module, no new collective, no new call site.

| | |
|---|---|
| the ask | `staging_bytes + seam_entry_margin_bytes()` (512 MiB default) |
| who funds it | the ladder that already stood there, plus the KV rung, which is asked for the same total |
| margin met | the seam enters |
| margin short, law met | the seam is DELAYED, up to 2 consecutive GROUP abandons per direction |
| budget spent, law met | the seam enters on the law, at WARNING. This is s34's shipped behaviour, so the worst case of the term is the behaviour it replaces |
| law short | refused exactly as before, however spent the budget is |
| which of the two | decided by ARITHMETIC on the verdict already in hand — `free_after - staging >= law_floor` — never by asking the guard again |
| uniformity | the delay joins `too_small` and rides the `_collective_min([fits, -fits, margin_only])` that already made the abandon unanimous |
| the budget's currency | consecutive GROUP abandons, booked in `_execute` from the reduced verdict and reset there when an attempt goes through |

**Why 512.** It covers the measured 456 MiB draw-from-a-low-entry with room
over. It is deliberately not the 1026 MiB worst case (§1a).

**Why the budget is bounded and per-direction.** An unbounded margin refusal
of pp->tp starves decode under strict purity — 411 abandons, 0 requests in 6
minutes, health 503, measured 2026-08-10. All seven of the deepest troughs
are pp->tp, so that leg had to be *fundable* rather than merely refusable;
the guard already unlocks the host tier there via `refusal_is_fatal`.
Delaying tp->pp only defers prefill and is safe, so the two legs keep
separate counters — one shared counter would let the safe leg spend the
dangerous leg's budget.

**Off switch.** `SGLANG_SEAM_ENTRY_MARGIN_MIB=0` restores the single pre-C20
ask exactly, as a value of the same term rather than a second code path.
`SGLANG_SEAM_ENTRY_DELAY_BUDGET` tunes the budget.

### 3a. WHAT IT IS NOT

It is not a lender and it must not grow into one. It acts once per cutover,
at a point where the ladder was already being spent (s34 paid
`allocator-cache` 464 times at this same gate and stayed green), and it is
bounded by an ask that a measurement sized. The mechanism s36 falsified ran
on the scheduler round and had no bound at all except a watermark it set
itself.

---

## 4. WHAT TO DO NEXT, IN ORDER

0. **BOUND WHAT THE MARGIN MAY ASK OF THE KV RUNG, or give the rung an
   admission-aware floor** (§1a). An 8 GiB margin takes the instance down
   through `available_size() == 0`; 512 is fine and green. This is the one
   item with a demonstrated failure attached, and it was left undone
   deliberately rather than shipped unvalidated at the end of a shift.
1. **THE KV RUNG IS THE MARGIN'S PRICE AND NOBODY HAS PRICED IT.** Asking the
   rung to fund `staging + margin` multiplies its firing rate (§1c, §2).
   Every shrink recovered on this run and the pool never left 512552, so it
   is residency and not capacity loss — but the honest next measurement is a
   one-boot A/B of `SGLANG_SEAM_ENTRY_MARGIN_MIB=512` against `0` scored on
   admission throughput, not on the corridor. The switch exists so that costs
   a boot rather than a revert; s36's §4.0 recipe is the template and it is
   the cheapest instrument in this corpus.
2. **The margin's own sizing is measured on PRE-margin data.** `C20_SIZING`
   comes from s34, where no margin existed. Now that the seam enters higher,
   re-run `scripts/s37_c20_proof.py` and `seam_drawdown.py` against THIS
   window: if the draw-from-a-low-entry has moved, 512 should move with it.
   A constant sized once and never re-measured is C1 waiting to happen.
3. **An abandoned pp->tp leaves the KV pool capped.** `collective_kv_backing_
   relief` applies its shrink BEFORE the gate's verdict, by design, so the
   bytes are visible to the guard. If the gate then delays or refuses, the
   shrink stands, and its own justification ("the PP layout goes inactive for
   the whole TP phase") assumes the flip happened. Recovery only runs at a
   tp->pp cutover, which cannot occur until a pp->tp succeeds. Pre-existing;
   C20 matters because it adds a new abandon class that fires exactly when
   memory is tight. Found by the review, not by a window.
4. **The abandon path does not drain the abort deferral window.**
   `_execute`'s abandon returns without `window.deactivate_and_drain()`, so
   deferred client aborts sit queued across every abandoned flip, while
   `AbortDeferralWindow`'s docstring promises a drain after disarm. Also
   pre-existing, also amplified by a new abandon class.
5. **`SGLANG_FORWARD_PEAK_PATH` on the next acceptance boot** — unchanged
   from HANDOFF_679 §2.1 and HANDOFF_680 §4.2, still two counter reads, still
   turns the prefill gate from enforcing into preempting. Note it would also
   give the seam-entry margin a MEASURED price to sit on instead of a
   constant.
6. **C18**: give `vram_dial` the corridor guard's floor before the dial is on.
7. **The corridor counters are still write-only** (HANDOFF_680 §4.5). C20
   added `seam_margin_delays` / `seam_margin_yields` and they are surfaced
   only through log text, like every counter before them. One periodic stats
   line per module still deletes a whole class of extract bugs.

---

## 5. PROCESS NOTES

* **Condition the measurement on the state the mechanism will see** (§1a).
  The marginal draw distribution and the conditional one differ by a factor
  of two here, and only the conditional one is about the case that matters.
* **An instrument that cannot fail is not evidence.** `s37_c20_proof.py` was
  run against s34's own accept2 data first, where it reproduces the +19 MiB
  binding margin exactly and returns FAILED. That is why its PASS on this
  window means something.
* **Pick the falsifying axis from the set the mechanism does not touch.**
  Inherited from HANDOFF_680 §6 and applied here before the boot rather than
  after: the margin makes the gate free more memory, so every corridor number
  it improves is a number it manipulates. `s37_judge.sh` therefore scores
  decode batches, dwell and soak completions against all four arms.
