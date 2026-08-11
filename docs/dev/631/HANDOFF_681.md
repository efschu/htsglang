# HANDOFF 681 — #656 / #631 Route A, successor 37

Narrow shift, two objectives, both closed: **register C20** (the seam-internal
trough that made s34's green margin luck) and a **fresh acceptance stamp** on
the tree that carries the fix.

---

## 0. THE ONE-LINE STATE

_(filled at the close of the window — see §2)_

---

## 1. ERRORS FIRST

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

    s34   21 shrinks in 65 min
    s37   _(see §2)_

Every one recovers — the `instead of 512552` figure never moves off the boot
pool, which is the line that separates a spill from a capacity loss — and
spec item 12 asks for exactly this ("KV itself is a spill class"). But it is
a real behavioural change and the next shift should know it was chosen, not
stumbled into. If it ever needs undoing, the one-line version is to price the
rung at `staging_bytes` while the guard is priced at `staging + margin`; the
cost of that is that the pp->tp leg loses its most capable funder and starts
spending its delay budget.

### 1d. THE FIXTURE IS 16 LAYERS, NOT 20

The brief (and therefore probably the next one) says "the 20-layer fixture".
There isn't one. The canonical CPU flip fixture is `N_LAYERS = 16` with
`MAP_625 = ((0..7), (8..11), (12..15))`, defined in
`test/registered/scheduler/test_phase_flip_runtime.py:52` and two siblings.
"20" is `#631 boot 20`, cited at `test_phase_flip_runtime.py:1247`.

---

## 2. THE ACCEPTANCE WINDOW

_(filled at the close)_

---

## 3. WHAT THE MECHANISM IS

`PhaseFlipRuntime._corridor_gate`, one extra term and a graded verdict. No new
module, no new collective, no new call site.

| | |
|---|---|
| the ask | `staging_bytes + seam_entry_margin_bytes()` (512 MiB default) |
| who funds it | the ladder that already stood there, plus the KV rung, which is asked for the same total |
| margin met | the seam enters; the direction's delay budget is restored |
| margin short, law met | the seam is DELAYED, up to 2 consecutive per direction |
| budget spent, law met | the seam enters on the law, at WARNING. This is s34's shipped behaviour, so the worst case of the term is the behaviour it replaces |
| law short | refused exactly as before, however spent the budget is |
| uniformity | the delay joins `too_small` and rides the `_collective_min([fits, -fits])` that already made the abandon unanimous |

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

_(filled at the close)_

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
