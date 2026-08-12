# HANDOFF #363 — intra-phase stage actuator (errors first)

Branch `feat/regime-stage-actuator-363`, worktree `/spinning/wt-363-stages`,
base `598f570ba4`. `PYTHONPATH=/spinning/wt-363-stages/python`.
No merges performed — the operator sequences.

---

## 1. ERRORS AND CORRECTIONS FIRST

### 1.1 The brief's premise was wrong in two places (verified at code)

**"The table now SOLVES but nothing ACTUATES from it" — FALSE.** The actuation
path exists and is wired:

- `python/sglang/srt/managers/regime_act.py:280` `build_regime_actuator`,
  bound at `regime_runtime.py:726` as `commit_fn` in `MODE_ACT`;
- it arms all three axes already — #297 `KvReshardRuntime.arm`, #330
  `apply_budget_request` (grow only), and the #631 phase flip.

So #578 did not leave a dead table. What it left is narrower and is what this
branch fixes. Anyone re-briefed on the old premise will look for the wrong gap.

**C18's direction is the reverse of the brief.** The brief says the guard
CALLS the dial. In this tree the DIAL calls the GUARD:

- `vram_dial.py:1092-1107` builds `_relief` from
  `get_corridor_guard(scheduler).ensure_headroom`;
- `vram_dial.py:78` imports `DEFAULT_FLOOR_MIB` **from** `corridor_guard`;
- `grep -rn "dial" corridor_guard.py` → **zero hits**. The guard has no
  import path to #330 in either direction.

This branch follows the tree, not the brief: the CALLER prices its move and
asks the guard, which is also the shape `phase_flip_runtime._execute` already
uses (price → gate → group-reduce → cutover).

### 1.2 The two real gaps (both now closed)

**GAP A — ms/round never reached the decision.** `rank_forward_ms` is
accumulated (`regime_runtime.py:340`) and averaged (`:372`), then packed into
the consensus payload (`regime_classifier.pack_proposal`) for ONE purpose:
deriving `rank_ms_spread_pct`, the cross-rank SKEW that act-mode interlock 4
vetoes on. The stage SELECTION was `table.select(regime, sample, ...)` — a
regime LABEL and occupancy. The controller knew the round time and had never
used it to choose a stage.

**GAP B — a stage flip never passed the corridor.** Three facts:
- `RegimeObserver._act_interlocks` ran four interlocks; none was a memory
  admission (`regime_runtime.py:443+`);
- `RegimeActuator.apply` calls the #330 dial directly (`regime_act.py:216`);
- the dial's GROW path validates only `min_viable_budget_bytes` and
  `effective_budget_ceiling_bytes`, and spends the corridor relief ladder
  ONLY when `my_reduction > 0`, i.e. only on a SHRINK (`vram_dial.py:565-580`).

So the one direction that CONSUMES free VRAM was the one direction never
priced against the corridor law. `grep -c "corridor" regime_*.py` was 0.

### 1.3 A design error I made and corrected mid-build

The first version of `regime_ms_clock` scaled the WHOLE round by the ratio of
the two stages' measured gains. That formulation cancels the measured round
out of the verdict algebraically:

    100 * (t - t*cur/cand) / t  ==  100 * (1 - cur/cand)

— an "ms/round-driven" controller returning the identical answer on a
wait-bound and a compute-bound rig, i.e. not ms-driven at all. Corrected to
rescale ONLY the wait term, which is also the honest mechanism claim (the
stage axes move how work is DIVIDED; they do not make a GEMM faster).
`test_improvement_depends_on_the_measured_split` is what keeps it corrected —
if that test is ever deleted the regression is invisible.

### 1.4 Known-unvalidated

**All of it is desk code.** Nothing here has run on a GPU. The modules are
labelled `STATUS: desk code` in their own docstrings. The measurement window
that would validate them is `docs/dev/363/TICKET_363_STAGE_CLOCK.md`, written
and NOT run.

Specifically unvalidated: that the wait term is really what a stage flip
moves (mechanism argument, ticket acceptance A3), and that the 5 % enter
watermark sits above this rig's A-vs-A band (policy number, ticket pre-step
P2). If P2 comes back above 5 %, the axis will simply never flip — the code
refuses any signal inside its measured band — which is the safe direction.

---

## 2. WHAT WAS BUILT

### New modules

`python/sglang/srt/managers/regime_ms_clock.py` — the decision loop.
- `MsRoundWindow` — sliding window, readiness floor, compute/wait means.
- `predicted_round_ms` / `improvement_pct` — rescale the WAIT term only.
- `combined_band_pct` — quadrature sum of both stages' A-vs-A bands; a
  difference of two measured gains carries both, so the floor is larger than
  either and a candidate cannot clear it by having been measured on a quieter
  card.
- `MsStageDecider` — two watermarks, two windows, enforced at construction
  (`exit < enter`, `exit_window > enter_window`) in house style. Does NOT
  carry a dwell: `regime_classifier.DwellGate` owns that and the separation
  is deliberate.
- `pack_ms_sample` / `unpack_ms_sample` — the group reduction. **This is the
  group-uniformity mechanism**: the clock consumes a GROUP statistic, never
  this rank's own, so its verdict is identical on every rank by construction.
  Round length = MAX total (slowest rank sets the barrier — pacemaker law);
  wait = MIN wait (the barrier time every rank was at least paying).
  Conservative at both ends, which is the right bias for a number that goes
  on to authorize spending memory.

`python/sglang/srt/managers/regime_admission.py` — pricing + the gate.
- `price_stage_flip` — residency delta (from the stages' own
  `vram_budget_mib`, the same vector the dial is about to be handed) plus the
  TARGET stage's transient UNDER THE CURRENT LOAD STATE. Eight named
  refusals; **none of them returns a zero**.
- `CorridorAdmission.admit` — asks `guard.ensure_headroom`, then reduces the
  verdict group-unanimously with the packed-pair MIN. A rank with no guard
  ABSTAINS (refuses) rather than voting yes. Never raises into the scheduler
  loop.

### Wiring (all behind `--regime-stage-clock`, default False)

- `regime_runtime.py`: observer takes `stage_clock` / `admission`;
  `_intra_phase_decide` runs the extra collective and the clock;
  **interlock 5** is the corridor admission, placed last of the five because
  it is the only one that touches the driver.
- `regime_runtime.rank_split_ms_from` — new accessor returning
  `(compute_ms, wait_ms)`, `(None, None)` on a graph-covered forward (same
  honesty rule as `rank_forward_ms_from`: no fast zeros).
- `scheduler.py`: passes `rank_compute_ms` / `rank_wait_ms`, one accessor read.
- `server_args.py`: `regime_stage_clock: bool = False`. Only has effect with
  `--regime-controller act`.

---

## 3. TEST RESULTS (recorded before commit, per the rule)

Run: `PYTHONPATH=/spinning/wt-363-stages/python .venv/bin/python -m pytest --color=no`

| Suite | Result |
|---|---|
| `test_regime_ms_clock_363.py` | **35 passed** |
| `test_regime_admission_363.py` | **25 passed** |
| `test_regime_intra_phase_wiring_363.py` | **11 passed** |
| `managers/ -k "regime or corridor or phase_flip or phase_policy"` + `planner/` | **828 passed, 1 skipped** |
| `test_regime_observe.py` (ratchet updated) | **49 passed** |

ruff on all new/changed-by-me files: **clean**. (`server_args.py` carries 357
pre-existing ruff errors on the BASE commit — verified against a pristine
worktree; not introduced here.)

### Pre-existing failures, verified NOT mine

A full `test/registered/unit/managers/` run shows 13 failures. I checked them
against a pristine `598f570ba4` worktree:

- **8 are pre-existing on the untouched base**: `test_first_chunk_dynamic_chunking`
  (2), `test_rank_prefill_log` (1), `test_scheduler_chunked_req_gate` (3),
  `test_scheduler_pp_request_order_633` (1 + 1 subtest).
- **`test_hisparse_unit` (2 + 1 error) is test-ORDER pollution**: it passes
  cleanly on its own in this worktree (`9 passed, 2 skipped`).

### Can-fail proof (a test that cannot fail proves nothing)

Both suites were mutation-checked, mutations reverted afterwards:

1. Made the transient silently default to `0.0` instead of refusing →
   **5 tests went red**.
2. Reverted the arithmetic to whole-round scaling → **6 tests went red**,
   including `test_improvement_depends_on_the_measured_split`.

### Deliberate ratchet update

`test_regime_observe.py::test_the_hook_passes_only_replicated_state_plus_the_named_rank_local`
enumerates the observer hook's keywords and demands a deliberate tier
decision for each new one. Updated with the decision recorded in the
docstring: `rank_compute_ms` / `rank_wait_ms` are TIER-L, same discipline as
`rank_forward_ms` — accumulated rank-locally, never branched on locally,
released only through the MIN reduction.

---

## 4. NEXT STEPS, IN ORDER

1. **Do not merge on green tests alone.** Every claim is desk-validated.
2. Run ticket pre-steps P1-P3 (cheap, no load). **P1 is the likely blocker**:
   if the stage table is all-unmeasured, the window cannot pass and the
   missing work is #584's measurement pass, not this branch.
3. Run the window in `docs/dev/363/TICKET_363_STAGE_CLOCK.md`. Acceptance is
   five items; four of five is not a pass.
4. Only then consider defaulting the flag on — and that is a separate
   decision from merging it off.

## 5. STOPPING POINT

Complete as scoped: both gaps closed, flag-gated, suites green, ruff clean,
ticket and handoff written, pushed. Nothing left half-done. The single
outstanding thing is that none of it has touched a GPU, which is the ticket's
job and was deliberately out of scope for this desk strand.
