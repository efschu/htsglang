"""#602: the sizer's pool and the corridor-safe floor are different questions.

WHAT WENT WRONG, AND IT WAS CAUGHT ON METAL. The KV-floor solver's prediction
was compared against the pool a live boot actually sized, under a +/-5 % gate.
On F4-r4's census boots that comparison came out **-23.3 %** and he refused to
boot the cut. He was right, and the defect is not a mis-calibrated constant: the
two numbers answer different questions.

    ``world_kv_floor``  funds the WORST measured load transient (law 31).
    In this regime the worst state is a SEAM on EVERY rank --
    SEAM_TP_TO_PP 2168 MiB on rank 0, SEAM_PP_TO_TP 700 / 932 on ranks 1 / 2,
    two to three times the prefill-triggered scalars.

    The live pool sizer charges NO seam transient at all, because it sizes the
    pool BEFORE any seam has ever run.

So one says "the largest pool that stays corridor-safe while a cutover is in
flight" and the other says "the pool the sizer will produce". On this rig they
are ~29 % apart. Comparing either against the other is a category error, and
the +/-5 % gate was doing exactly that.

MY OWN 1.8 % AGREEMENT WAS PARTLY LUCK. On my boot the charged transient came
from the #485 prefill bench (1346/1120/982) rather than from a seam census, so
it was small enough that the category error stayed inside the gate. A regime
whose worst state is a seam exposes it immediately.

THE FIX IS TO SEPARATE THEM, NOT TO RETUNE. ``world_predicted_pool`` excludes
the worst-load transient and is what the gate compares against a measured pool.
``world_kv_floor`` keeps funding it and is what a corridor-safety argument
stands on. Neither is wrong; conflating them is.

CALIBRATION SOURCE: F4-r4's census boots, /spinning/evidence-665-f1/census-602/
(``census_pp*.json``, ``transient_pp*.json``), measured pool 471303 tokens on
the incumbent cut 28,20,16.
"""

import dataclasses
import os
import sys
import unittest

from sglang.srt.planner import pp_cut
from sglang.test.ci.ci_register import register_cpu_ci

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_pp_family_cut_485 as ref  # noqa: E402

register_cpu_ci(est_time=10)

MIB = 1 << 20

# --- F4-r4's regime, from the census JSONs -------------------------------
MEASURED_POOL_TOKENS = 471303
NVML_TOTAL_MIB = (32088.5, 20054.9375, 20054.9375)
#: nvml_used - params - pools, per rank.
RESIDUAL_MIB = (4874.59, 3001.45, 3005.19)
#: Free above the 1024 MiB corridor at rest: what the seam reserve held.
SEAM_HELD_MIB = (2711.6875, 702.375, 1134.375)
#: worst_transient_mib, and it is a SEAM state on every rank.
WORST_TRANSIENT_MIB = (2168.0, 700.0, 932.0)
#: params_mib / n_layers, averaged across the three stages.
ATTN_LAYER_MIB = 374.24
LINEAR_LAYER_MIB = 476.21
#: The census reports a `visual` tower on every stage and no separate MTP line.
REPLICATED_MIB = 920.45
EMBED_MIB = 2425.0
LM_HEAD_MIB = 2425.0
#: pools_mib / (tokens * n_attn), averaged.
KV_CELL_B_PER_TOKEN_PER_ATTN = 2399.6

INCUMBENT_CUT = [28, 20, 16]


def _f4_inputs(**over):
    kw = dict(
        budgets=NVML_TOTAL_MIB,
        seam_staging=SEAM_HELD_MIB,
        overheads=RESIDUAL_MIB,
        transients=WORST_TRANSIENT_MIB,
        pool=MEASURED_POOL_TOKENS,
        corridor=1024.0,
    )
    kw.update(over)
    base = ref._inputs(**kw)
    return dataclasses.replace(
        base,
        attn_layer_weight_bytes=ATTN_LAYER_MIB * MIB,
        linear_layer_weight_bytes=LINEAR_LAYER_MIB * MIB,
        embedding_weight_bytes=EMBED_MIB * MIB,
        lm_head_weight_bytes=LM_HEAD_MIB * MIB,
        replicated_weight_bytes=REPLICATED_MIB * MIB,
        state_bytes_per_linear_layer=0.0,
        kv_bytes_per_token_per_attn_layer=KV_CELL_B_PER_TOKEN_PER_ATTN,
    )


class TheTwoPredictionsAreDistinct(unittest.TestCase):
    """The core of the defect: one output cannot answer both questions."""

    def test_the_pool_exceeds_the_corridor_safe_floor_under_a_seam_transient(self):
        inputs = _f4_inputs()
        pool = pp_cut.world_predicted_pool(INCUMBENT_CUT, inputs)
        floor = pp_cut.world_corridor_safe_floor(INCUMBENT_CUT, inputs)
        self.assertIsNotNone(pool)
        self.assertIsNotNone(floor)
        self.assertGreater(
            pool,
            floor,
            "the sizer-equivalent pool is not above the corridor-safe floor, "
            "so the worst-load transient is not actually being separated",
        )

    def test_they_are_far_apart_not_a_rounding_difference(self):
        inputs = _f4_inputs()
        pool = pp_cut.world_predicted_pool(INCUMBENT_CUT, inputs)
        floor = pp_cut.world_corridor_safe_floor(INCUMBENT_CUT, inputs)
        self.assertGreater((pool - floor) / pool, 0.10)

    def test_they_coincide_when_no_transient_is_charged(self):
        """The only case where conflating them was harmless -- and the reason
        the defect survived a boot whose worst state was a prefill."""
        inputs = _f4_inputs(transients=(0.0, 0.0, 0.0))
        self.assertAlmostEqual(
            pp_cut.world_predicted_pool(INCUMBENT_CUT, inputs),
            pp_cut.world_corridor_safe_floor(INCUMBENT_CUT, inputs),
            places=6,
        )


class ThePoolPredictionMatchesTheMeasuredBoot(unittest.TestCase):
    """The gate, comparing like with like at last.

    F4-r4's boots sized 471303 tokens on the incumbent cut. Only the
    sizer-equivalent prediction may be held to that number.
    """

    GATE = 0.05

    def test_predicted_pool_is_inside_the_gate(self):
        got = pp_cut.world_predicted_pool(INCUMBENT_CUT, _f4_inputs())
        self.assertIsNotNone(got)
        err = (got - MEASURED_POOL_TOKENS) / MEASURED_POOL_TOKENS
        self.assertLess(
            abs(err),
            self.GATE,
            f"predicted pool {got:.0f} against measured "
            f"{MEASURED_POOL_TOKENS} is {err:+.1%}",
        )

    def test_the_corridor_safe_floor_would_have_failed_that_gate(self):
        """Pinned so the category error cannot come back quietly: the number
        that was being compared is far outside the gate, by construction."""
        got = pp_cut.world_corridor_safe_floor(INCUMBENT_CUT, _f4_inputs())
        err = (got - MEASURED_POOL_TOKENS) / MEASURED_POOL_TOKENS
        self.assertGreater(
            abs(err),
            self.GATE,
            "the corridor-safe floor now sits inside the pool gate, which "
            "means the two have stopped being different questions and this "
            "whole separation needs revisiting",
        )


class TheSeparationIsInTheTransientOnly(unittest.TestCase):
    """Everything else the sizer charges must still be charged by BOTH."""

    def test_the_corridor_binds_both(self):
        tight = _f4_inputs(corridor=4096.0)
        base = _f4_inputs()
        self.assertLess(
            pp_cut.world_predicted_pool(INCUMBENT_CUT, tight),
            pp_cut.world_predicted_pool(INCUMBENT_CUT, base),
        )
        self.assertLess(
            pp_cut.world_corridor_safe_floor(INCUMBENT_CUT, tight),
            pp_cut.world_corridor_safe_floor(INCUMBENT_CUT, base),
        )

    def test_the_seam_reserve_binds_both(self):
        """The seam RESERVE is charged by the sizer (it is held at rest); only
        the seam TRANSIENT is not."""
        more = _f4_inputs(seam_staging=tuple(s + 500.0 for s in SEAM_HELD_MIB))
        base = _f4_inputs()
        self.assertLess(
            pp_cut.world_predicted_pool(INCUMBENT_CUT, more),
            pp_cut.world_predicted_pool(INCUMBENT_CUT, base),
        )
        self.assertLess(
            pp_cut.world_corridor_safe_floor(INCUMBENT_CUT, more),
            pp_cut.world_corridor_safe_floor(INCUMBENT_CUT, base),
        )

    def test_the_transient_binds_only_the_corridor_safe_floor(self):
        more = _f4_inputs(transients=tuple(t + 500.0 for t in WORST_TRANSIENT_MIB))
        base = _f4_inputs()
        self.assertAlmostEqual(
            pp_cut.world_predicted_pool(INCUMBENT_CUT, more),
            pp_cut.world_predicted_pool(INCUMBENT_CUT, base),
            places=6,
        )
        self.assertLess(
            pp_cut.world_corridor_safe_floor(INCUMBENT_CUT, more),
            pp_cut.world_corridor_safe_floor(INCUMBENT_CUT, base),
        )


class TheCostKeepsTheTermsApart(unittest.TestCase):
    """The conflation was at the source: ``_price_stage`` summed the worst
    transient into ``transient_mib`` together with the cut-invariant fixed
    overhead, so no consumer could tell them apart."""

    def test_the_stage_cost_reports_them_separately(self):
        inputs = _f4_inputs()
        costs = pp_cut.stage_costs(INCUMBENT_CUT, inputs)
        for cost, transient, overhead in zip(costs, WORST_TRANSIENT_MIB, RESIDUAL_MIB):
            self.assertAlmostEqual(cost.transient_mib, transient, places=3)
            self.assertAlmostEqual(cost.fixed_overhead_mib, overhead, places=3)

    def test_residency_still_counts_both(self):
        """Byte-neutrality for the corridor-safe family: splitting the field
        must not change what a stage is held to occupy."""
        inputs = _f4_inputs()
        for cost in pp_cut.stage_costs(INCUMBENT_CUT, inputs):
            self.assertAlmostEqual(
                cost.resident_mib,
                cost.weight_mib
                + cost.nonlayer_weight_mib
                + cost.state_mib
                + cost.kv_mib
                + cost.transient_mib
                + cost.fixed_overhead_mib
                + cost.draft_mib,
                places=6,
            )


if __name__ == "__main__":
    unittest.main()


class TheSolverGateIsNotTheScoringGate(unittest.TestCase):
    """Why ``world_corridor_safe_floor`` exists beside ``world_kv_floor``.

    ``world_kv_floor`` also refuses a stage whose headroom is negative at the
    INPUT arena. That is right for the solver -- a cut that does not fit the
    arena being searched is not a candidate -- and wrong for scoring an
    incumbent, because a MEASURED arena is full by construction: headroom sits
    at ~0 and a model error of either sign flips it to "infeasible", so the
    capacity cannot be reported at all.

    Found the hard way: every comparison in this file returned ``None`` until
    the two gates were separated.
    """

    def test_the_solver_form_refuses_a_measured_arena(self):
        self.assertIsNone(
            pp_cut.world_kv_floor(INCUMBENT_CUT, _f4_inputs()),
            "the solver form now scores a full arena, so the scoring "
            "counterpart may be redundant -- check before deleting it",
        )

    def test_the_scoring_form_reports_a_number_there(self):
        self.assertIsNotNone(
            pp_cut.world_corridor_safe_floor(INCUMBENT_CUT, _f4_inputs())
        )

    def test_the_solver_form_still_works_below_the_arena(self):
        """It must keep refusing only what genuinely does not fit."""
        roomy = _f4_inputs(pool=int(MEASURED_POOL_TOKENS * 0.5))
        self.assertIsNotNone(pp_cut.world_kv_floor(INCUMBENT_CUT, roomy))
