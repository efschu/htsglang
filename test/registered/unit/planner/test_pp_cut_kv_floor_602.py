"""#602 term 2: solve the PP cut for the KV FLOOR, not only the makespan.

WHAT THIS IS ABOUT. On the 2026-08-16 live boot the three stages reported

    PP0 5090   local capacity 550000 tokens   (user cap; seam permits 590281)
    PP1 3080   local capacity 471638 tokens   <- BINDS
    PP2 3080   local capacity 526893 tokens

and ``model_runner_kv_cache_mixin.py`` min-reduces them to one world value,
because under PP a request's tokens occupy KV on EVERY stage. 78362 tokens
were stranded on PP0 and 55255 on PP2 -- about 1.4 GiB of VRAM that no post
holds and no rank can spend.

That surplus can never become KV on the stage that holds it: the token count
is necessarily uniform across the pipeline. The ONLY lever that converts it
is the cut itself -- move layers onto the stage with headroom and the binding
stage's per-token cost falls, so the world MIN rises.

WHY A NEW OBJECTIVE AND NOT ``solve_pp_cut``. That solver minimizes the
lockstep MAKESPAN and, among near-optimal cuts, maximizes the tightest
``runnable_headroom_mib``. Headroom is not capacity: a stage converts headroom
into tokens at a rate set by its OWN attention-layer count, so the cut that
leaves the most MiB on the tightest card is not the cut that lets the pipeline
address the most tokens. This module's objective is

    maximize  min_r ( tokens rank r could hold )

over the same contiguous cuts, under the same hard constraints.

THE CONSTRAINTS STAY HARD, AND THEY ARE THE ONES ALREADY MODELLED.
``corridor_mib`` (the 1024 MiB law) is subtracted from every budget before any
comparison, and ``seam_staging_mib`` is the per-rank phase-flip peak that must
remain reachable on top of residency. Term 1 of the #602 attribution -- the
1728 MiB/rank arming floor -- is BOOKED, not reclaimable, and is expressed
here as corridor 1024 + the rank's measured seam staging. A cut that funds
more tokens by eating either of them is not admissible.
"""

import itertools
import os
import sys
import unittest

from sglang.srt.planner import pp_cut
from sglang.test.ci.ci_register import register_cpu_ci

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_pp_family_cut_485 as ref  # noqa: E402  (reference rig fixture)

register_cpu_ci(est_time=15)

#: The live boot's per-rank budgets (--rank-gpu-memory-mib), 2026-08-16.
LIVE_BUDGET_MIB = (31800.0, 18800.0, 19800.0)

#: The live boot's per-rank phase-flip seam demand, MiB: the measured fixed
#: floor plus the staging drawn at the boot's own arena (227+1062, 138+191,
#: 138+246). Term 1 of the attribution, booked and not reclaimable.
LIVE_SEAM_STAGING_MIB = (1289.0, 329.0, 384.0)

#: The arena the live boot actually sized, after the world MIN.
LIVE_ARENA_TOKENS = 471638

#: The hand-set cut in the live flagset (--pp-layer-ratio 28,20,16).
LIVE_CUT = [28, 20, 16]


def _live_inputs(**over):
    kw = dict(
        budgets=LIVE_BUDGET_MIB,
        seam_staging=LIVE_SEAM_STAGING_MIB,
        pool=LIVE_ARENA_TOKENS,
        corridor=1024.0,
    )
    kw.update(over)
    return ref._inputs(**kw)


class TheFloorOfAGivenCutIsComputable(unittest.TestCase):
    """Scoring must exist before optimising: a solver whose objective cannot
    be evaluated on the INCUMBENT cut cannot be shown to beat it."""

    def test_the_live_cut_scores_a_finite_floor(self):
        floor = pp_cut.world_kv_floor(LIVE_CUT, _live_inputs())
        self.assertIsNotNone(floor, "the shipping cut priced as infeasible")
        self.assertGreater(floor, 0)

    def test_the_floor_is_the_minimum_over_stages_not_the_sum(self):
        inputs = _live_inputs()
        floor = pp_cut.world_kv_floor(LIVE_CUT, inputs)
        per_stage = pp_cut.stage_kv_capacities(LIVE_CUT, inputs)
        self.assertEqual(len(per_stage), 3)
        self.assertAlmostEqual(floor, min(per_stage), places=6)

    def test_an_infeasible_cut_scores_None_rather_than_a_number(self):
        """A cut that cannot fund corridor+seam has no capacity to report;
        returning a number would let the solver rank it."""
        starved = _live_inputs(budgets=(31800.0, 1000.0, 19800.0))
        self.assertIsNone(pp_cut.world_kv_floor(LIVE_CUT, starved))


class TheSolvedCutBeatsTheShippingCut(unittest.TestCase):
    """The claim this whole term rests on."""

    def test_the_solved_cut_raises_the_world_minimum(self):
        inputs = _live_inputs()
        incumbent = pp_cut.world_kv_floor(LIVE_CUT, inputs)
        solved = pp_cut.solve_pp_cut_for_kv_floor(inputs)
        self.assertTrue(solved.feasible, solved.refusals)
        self.assertGreater(
            solved.floor_tokens,
            incumbent,
            "the solved cut does not address more tokens than the hand-set "
            f"{LIVE_CUT}; term 2 of the #602 attribution is not reclaimable "
            "by re-cutting and the premise is wrong",
        )

    def test_the_shipping_cut_is_not_already_optimal(self):
        solved = pp_cut.solve_pp_cut_for_kv_floor(_live_inputs())
        self.assertNotEqual(list(solved.counts), LIVE_CUT)

    def test_the_solved_cut_keeps_every_layer(self):
        inputs = _live_inputs()
        solved = pp_cut.solve_pp_cut_for_kv_floor(inputs)
        self.assertEqual(sum(solved.counts), inputs.n_layers)
        self.assertTrue(all(c >= 1 for c in solved.counts))


class TheHardConstraintsAreNotTradedForTokens(unittest.TestCase):
    """Term 1 stays booked. A cut that buys tokens with the corridor or the
    seam is not a better cut, it is an unrunnable one."""

    def test_every_stage_of_the_solution_is_feasible(self):
        solved = pp_cut.solve_pp_cut_for_kv_floor(_live_inputs())
        for cost in solved.stages:
            self.assertGreaterEqual(cost.runnable_headroom_mib, 0.0)

    def test_raising_the_corridor_lowers_the_floor(self):
        """The corridor must actually bind the objective; if it does not, it
        is not being subtracted and the solution is over-claiming."""
        base = pp_cut.solve_pp_cut_for_kv_floor(_live_inputs(corridor=1024.0))
        tighter = pp_cut.solve_pp_cut_for_kv_floor(_live_inputs(corridor=3072.0))
        self.assertLess(tighter.floor_tokens, base.floor_tokens)

    def test_raising_the_seam_demand_lowers_the_floor(self):
        base = pp_cut.solve_pp_cut_for_kv_floor(_live_inputs())
        tighter = pp_cut.solve_pp_cut_for_kv_floor(
            _live_inputs(seam_staging=(2289.0, 1329.0, 1384.0))
        )
        self.assertLess(tighter.floor_tokens, base.floor_tokens)

    def test_a_budget_that_cannot_fund_the_floor_is_refused_not_shrunk(self):
        solved = pp_cut.solve_pp_cut_for_kv_floor(
            _live_inputs(budgets=(31800.0, 900.0, 19800.0))
        )
        self.assertFalse(solved.feasible)
        self.assertTrue(solved.refusals)


class TheSolverIsExactAndDeterministic(unittest.TestCase):
    """A DP that only usually finds the optimum is a heuristic with a proof
    voice. Check it against the full enumeration."""

    def _brute_force_floor(self, inputs):
        n, k = inputs.n_layers, inputs.pp_size
        best, best_counts = None, None
        for cuts in itertools.combinations(range(1, n), k - 1):
            bounds = list(cuts) + [n]
            counts = [bounds[0]] + [bounds[i] - bounds[i - 1] for i in range(1, k)]
            floor = pp_cut.world_kv_floor(counts, inputs)
            if floor is None:
                continue
            if best is None or floor > best:
                best, best_counts = floor, counts
        return best, best_counts

    def test_the_dp_matches_the_full_enumeration(self):
        inputs = _live_inputs()
        brute, _ = self._brute_force_floor(inputs)
        solved = pp_cut.solve_pp_cut_for_kv_floor(inputs)
        self.assertAlmostEqual(solved.floor_tokens, brute, places=6)

    def test_the_same_inputs_give_the_same_cut(self):
        a = pp_cut.solve_pp_cut_for_kv_floor(_live_inputs())
        b = pp_cut.solve_pp_cut_for_kv_floor(_live_inputs())
        self.assertEqual(a.counts, b.counts)


class TheEmptyKvStageIsRefused(unittest.TestCase):
    """A stage owning no full-attention layer has an empty KV pool, so its
    'capacity' is unbounded and would win every maximin. Same refusal the
    makespan solver already makes."""

    def test_a_zero_attention_stage_never_wins(self):
        solved = pp_cut.solve_pp_cut_for_kv_floor(_live_inputs())
        self.assertTrue(all(a >= 1 for a in solved.attention_counts))


if __name__ == "__main__":
    unittest.main()
