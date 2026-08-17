# #702 decision table — incumbent vs [30,18,16] vs [33,15,16]

**2026-08-17, Slot-3. Decision support, NOT a recommendation.** The
user-facing copy lives in the morning plan file
(`/spinning/evidence-665-f1/PLAN_PERF_PIPELINE_2026-08-16.md`, dated block
appended 2026-08-17, nothing above it edited — head-298 sha256 verified
identical before and after: `81560e94b7628e6c8...`). That file is not under
version control, which is why this copy exists.

Numbers re-derived on the #707 closed form with the calibrated fixture:
`available_bytes` 8,524,386,304 / 4,740,280,320 / 3,573,989,376; cell 2048 B;
ms/layer 1.7571 / 7.740 / 7.275; links 13/13/6.4 GB/s with the 5090 on x8.
Incumbent pool reads **436,275** here against the 436,766 quoted in older
notes; all percentages below use 436,275 so they are internally consistent.

| | **[28,20,16]** incumbent | **[30,18,16]** | **[33,15,16]** |
|---|---|---|---|
| attention per rank | (7, 5, 4) | (7, 5, 4) | (8, 4, 4) |
| coupled pool tokens | 436,275 | **436,275 (+0.0%)** | 415,859 (**-4.7%**) |
| binder | PP2 | PP2 | **PP0** |
| pipelined prefill factor | 1.0000 | **1.1111** | **1.3299** |
| vs A-vs-A noise floor (14.1%) | — | **11.1% — BELOW IT** | 33.0% — above |
| arming floors MiB/rank | **1728 / 1825 / 2467 (MEASURED)** | unbooted, proxy ±500 | unbooted, proxy ±500 |
| switch cost from incumbent | — | **1575 ms** | **1575 ms** |
| payback (prefill-seconds) | — | 15.75 s | 6.35 s |
| stage-1 "pool >= incumbent" gate | passes (is the gate) | **PASSES** | **FAILS (needs waiver)** |

## The two candidates fail in opposite places

**[30,18,16] passes the pool gate and cannot be validated by measurement.** Its
pool is identical to the incumbent's structurally, not coincidentally: it keeps
the attention split (7,5,4) and leaves PP2 (layers 48-63) untouched, and PP2 is
the binder. Per-rank capacity is `[594,615 / 462,920 / 436,275]` at the
incumbent and `[517,464 / 570,931 / 436,275]` here — PP2's column is byte-for-
byte the same, so moving two layers between PP0 and PP1 cannot move a pool PP2
sets. But its 11.1% gain sits BELOW this rig's measured 14.1% A-vs-A noise
floor: **no boot on this rig can distinguish it from the incumbent.**

**[33,15,16] is clearly measurable and needs a capacity waiver.** Its 33.0%
gain clears the noise floor. It costs 4.7% of context and moves the binder from
PP2 to PP0, because rank0 picks up an 8th attention layer and its cell grows.
This is the "USER DECISION PENDING" already on the morning list, unchanged.

## Correction: [30,18,16] is not "+20% pool"

No +20% pool figure for this cut is reproducible. Its coupled pool is +0.0%.
The nearest figure on the list is the **decoupled** pool at +17.7%, which is
**cut-independent** — the same 513,875 tokens for every cut including the
incumbent (this is the #704b R6 sum-rule result) — and therefore not
attributable to [30,18,16] or any other cut.

## Solver defect: the frontier cannot see [30,18,16]

`planner/prefill_frontier.py:144-156` enumerates, for each lead depth `n0`,
**only the tail split that minimises pipelined time**. At `n0=30` that is
`[30,16,18]` (130.95 ms), so `[30,18,16]` (139.32 ms) is never generated. But
`[30,16,18]` holds 343,951 tokens (-21.2%) against `[30,18,16]`'s 436,275
(+0.0%): the enumeration gave up 21 points of pool for 6% of speed, then
reported only the survivor.

**The frontier is Pareto-optimal in speed but not complete in pool.** A cut
that is roomier and slightly slower at the same lead depth cannot appear on it.
`[30,18,16]` strictly dominates the incumbent — equal pool, more speed — and
was invisible for exactly this reason, which is why it never appeared in the
#702 block. NOT FIXED HERE, flagged: the frontier may be hiding other
pool-preserving cuts, and the fix is to keep the Pareto set over
`(pool, speed)` per lead depth rather than the speed-argmin.

## What a post-switch validation boot must show

Ordered by what would falsify the model, not by convenience.

1. **The predicted BINDER, by name.** [30,18,16] must still bind on PP2;
   [33,15,16] must bind on PP0. A different binder voids every pool number here.
2. **Measured arming floors per rank** against the ±500 MiB proxy. At 8
   attention layers ±500 MiB is ±32,000 tokens (~7%) — larger than
   [33,15,16]'s entire 4.7% price, so this term decides whether that trade was
   real.
3. **Measured pool tokens** against 436,275 / 415,859.
4. **Prefill factor against the 14.1% noise floor, A-vs-A first.** For
   [30,18,16] this is expected to be INCONCLUSIVE by construction: a boot
   reporting a clean 1.11x has measured its own noise, and that result must be
   rejected rather than celebrated.
5. **PP2's bracket** (`free_at_measure - arming_floor - margin`), 0.0 MiB at
   the incumbent. If it stays 0.0 under [30,18,16], pool-neutrality is
   confirmed at its source.

## Caveats carried

- Non-incumbent pool rows are EXTRAPOLATED across the `free_at_measure` shift
  only; the incumbent is measured and reproduces its own binder.
- 1575 ms is arithmetic from measured bandwidth (9614.9 MiB/rank arena refill
  over 13/13/6.4 GB/s, concurrent, x4 card gating). **No rung change has ever
  been performed on metal.**
- Speedups are UPPER BOUNDS (`fixed_ms = 0`).
- No recommendation is made or implied; both rows are presented so the pick is
  the user's.
