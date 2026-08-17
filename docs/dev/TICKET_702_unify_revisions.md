# TICKET #702 — unify the two revisions: `WorldMemory` is canonical

Filed by merge train 2 (2026-08-17) after `fix/602-fill-side` was stopped on
this exact hunk pair. It is deliberate work, not a merge resolution, and it
must not be attempted again as one.

## The situation

`python/sglang/srt/planner/pp_cut.py` exists in two incompatible revisions:

| | revision | model | lineage |
| --- | --- | --- | --- |
| A | "#702 revision — CO-SOLVING the KV/mamba token vector with the layer cut" | `class WorldMemory` (96 lines) | shipping / boot-validated |
| B | "#702 revision 4 — the PP pool divides by ATTENTION layers" | `class PhasePoolModel` (183 lines) | `fix/602-fill-side`, cluster-b |

Merging them textually is not possible: they are two different models of the
same question, not two halves of one.

## The decision

**A (`WorldMemory`) is CANONICAL**, because it is what boot 2 validated on
metal — the co-solved pool plus token vector. B lives on the excluded cluster-b
lineage and has no boot behind it.

**B is not discarded.** It carries two things A does not, and both are to be
ported as an EXTENSION of `WorldMemory`, never as a parallel model:

1. **The pool divides by ATTENTION layers**, not by all layers. On a hybrid
   checkpoint those differ, and A's arithmetic does not distinguish them.
2. **#723 frontier completeness** — the Pareto-set property B's revision was
   built around.

## The rule this ticket exists to enforce

One model. A second class modelling the same pool is how the two revisions
happened in the first place. Any port lands as methods or fields on
`WorldMemory` with the boot evidence for the changed arithmetic named in the
commit, or it does not land.

## Why it is not in train 2

`fix/602-fill-side` had five conflicts against the train. Four were mechanical
(a strict `__all__` superset, two add/add test files with one side empty, a
constant plus its use). Only this one was ambiguous, and choosing between two
design revisions inside a conflict marker is semantic invention — so the whole
branch was stopped rather than half-resolved. The other four resolutions are
recorded here so the next attempt does not re-derive them:

* `managers/seam_slope.py` — take the incoming side: `__all__` gains
  `derive_seam_slope_for_rank` and the function is added; HEAD is a subset.
* `test/registered/unit/model_executor/test_mamba_post_decomposition_704.py`
  and `test/registered/unit/planner/test_pp_cut_prefill_speed_702.py` — add/add
  where one side contributes nothing; union.
* `model_executor/model_runner_kv_cache_mixin.py` — take the incoming side:
  it introduces `MAMBA_POST_PART_NAMES` and uses `MAMBA_POST_PART_NAMES[0]`
  where HEAD had the literal. Verify the literal and the constant agree before
  committing.

## Done when

`fix/602-fill-side` merges with only mechanical conflicts, `pp_cut.py` has ONE
model, and the attention-layer divisor carries a boot number.
