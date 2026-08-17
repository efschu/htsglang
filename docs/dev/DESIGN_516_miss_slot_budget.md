# DESIGN #516 — the longer-horizon miss budget

Status: **BUILT, DEFAULT-OFF, simulation-positive, unproven on metal.**
(Unparked 2026-08-17; the PARKED note that preceded this is preserved in §5
because its finding is what narrowed the scope.)

## 1. Semantics, settled before code

**What is capped.** The window MISS RATE — misses as a fraction of activations
over one heat window. Not per-wave slots: that cap already exists and is not
this (§5).

**What happens at exhaustion — and why this is the only lossless option.**
Nothing is refused. The budget does not gate ADMISSION, it gates the PLACEMENT
CHANGE. That distinction is forced, not chosen:

* a miss cannot be declined — the wave's grouped GEMM needs those weights;
* deferring it to a later round does not exist inside a wave;
* substituting a resident expert is a DIFFERENT COMPUTATION, i.e. lossy, which
  the quality-last rule puts out of scope for anything default.

So the only lossless lever left is *whether to pay for a re-rank*, and that is
what the budget spends. **Losslessness is therefore structural**: this policy
can only decide whether weights MOVE between tiers, never which expert computes
a token. A test pins that its decision function contains no routing vocabulary
at all, so the property cannot quietly decay into a lossy one.

**Who owns the counter — one authority.** `HeatWindow` (#302a) already
maintains `window_hit_activations` / `window_miss_activations` and already owns
the re-rank decision. The budget is a method on it (`budget_holds`) consulted
by `plan()`. **No new module, no second authority, no new counter**: the
numbers it reads are the ones `close_round` was already folding into
`last_window_hit_rate`.

## 2. The mechanism, which the module had already named

`HeatMigrationConfig.period_forwards` carries this comment, written before
this work: *"Small values re-rank on noise and pay H2D for it."*

That is exactly the effect the budget exploits. A window whose miss rate is
already inside budget is a window whose top-R movement is **noise**, and
re-ranking to it chases that noise — paying H2D to make placement no better,
sometimes worse. The budget declines those swaps and keeps the ones where the
miss rate says placement is genuinely wrong.

## 3. Evidence — simulation, and how it was made possible

`scripts/dev/302a_heat_desk/simulate_miss_budget.py`.

The single-shot `expert_stats_*.json` dumps are aggregate histograms with **no
time axis**, and a longer-horizon budget is defined over time — so at first this
looked unanswerable. It is answerable because the `stats_series_*` directories
hold **12 CUMULATIVE snapshots per rank at 45 s intervals**; differencing
consecutive snapshots yields 11 real per-window activation deltas. That
differencing is the whole reason the question could be asked.

Three causal arms (a policy may only use windows it has already seen):
**A** static (never re-rank, the floor), **B** periodic (re-rank every window —
the equal-count re-rank #302a ships, and the bar), **C** budget.

At budget **0.04**, over all nine recorded rank/series combinations:

| | result |
| --- | --- |
| wins over B | **9 / 9** |
| mean hit-rate delta | **+0.0052** |
| worst case | **+0.0021** (still a gain) |
| swaps vs B | **15–54 %** |

The sweep shows the parameter is real and has a wrong side: 0.02 also wins 9/9
but gains less (+0.0032); 0.06 already loses one combination; by 0.10 it is
6/9 and mean-negative (−0.0009). **0.04 is not a tuned-to-fit choice — it is
the peak of a curve whose shape is visible in both directions.**

The bar the determination set — *beat the equal-count re-rank in simulation or
ship the negative* — is met strictly, and at a fraction of the transfer cost.

## 4. Default-inert, and what is still unproven

`miss_budget` defaults to **0.0**, and 0.0 means "the budget has nothing to
say" rather than "always skip": with it off, every swap the periodic policy
would have made is still made. Flag: `SGLANG_MOE_HEAT_MISS_BUDGET`.

**Everything above is simulation.** It replays recorded routing from one
2026-08-03 battery; it does not measure H2D time saved, does not run under a
live router, and does not interact with graph capture (#302a refuses under
capture, and that refusal is untouched). Activation is a window decision. The
honest summary is: the direction is favourable on every recorded sample, and
the size of the win on metal is unknown.

## 5. What the PARKED pass established, kept because it narrowed the scope

A per-wave miss budget **already exists under another name** and is not this
work: `ExpertOffload.resolve()` (`layers/moe/expert_offload.py:528-586`) raises
when `len(spill) > self.scratch` and directs the caller to wave-split;
`plan_token_waves` (`:262-284`) partitions TOKENS so that "every token is still
computed exactly once with all its experts resident -> byte-identical
regardless of which set is chosen". So: cap = `scratch`, exhaustion =
wave-split, losslessness proven by construction.

That is why `ANALYSE_516_determination.md`'s "NOT BUILT" verdict is **narrowed,
not overturned**: the per-wave cap is built (and is the one that matters for
the graphs half, since a captured graph needs a fixed slot shape and `scratch`
IS that shape); what was genuinely absent, and is what this document builds, is
the longer-horizon one.
