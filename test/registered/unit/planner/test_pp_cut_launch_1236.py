"""The launcher-side PP cut: makespan under a pool FLOOR, on both axes.

User decision 1 (2026-09-07): the stage cut is not a hand number. The solver
derives it, minimising the pipelined makespan among the cuts whose PP-phase
pool still holds ONE full-context prompt, and prints both objectives so the
trade cannot be paid by accident.

The synthetic table below is a three-card rig with one fast card and two slow
ones -- the shape of the real rig (5090 + 2x 3080, per-layer ms 8.10 vs
35.16/33.59, BSSCALE_0907.md) with round numbers so the expected direction is
arithmetic rather than a re-run of the solver.
"""

import unittest

import pytest

try:
    from sglang.srt.planner.pp_cut import (
        LAYER_FAMILY_ATTENTION,
        LAYER_FAMILY_LINEAR,
        PhasePoolModel,
        attention_counts,
        attention_split_is_realizable,
    )
    from sglang.srt.planner.pp_cut_launch import PPCutRefused, solve_launch_cut
    from sglang.test.ci.ci_register import register_cpu_ci
except RuntimeError as _import_err:  # pragma: no cover - leak-dependent
    pytest.skip(
        f"#249 default-device collection leak broke the import chain: {_import_err}",
        allow_module_level=True,
    )

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

#: 32 layers, every 4th one full attention -- the reference checkpoint's
#: period, scaled down so the enumeration is instant.
FAMILIES = tuple(
    LAYER_FAMILY_ATTENTION if (i % 4) == 3 else LAYER_FAMILY_LINEAR for i in range(32)
)
#: One fast card, two slow: 1 ms/layer against 4 ms/layer.
MS_PER_LAYER = (1.0, 4.0, 4.0)
INCUMBENT = (16, 8, 8)
#: Names that are NOT in any card-rate library, so the library path is
#: skipped and the measured list is used verbatim. Keeping the test off the
#: rig's own artifact is deliberate: a test that reads ~/.cache is not
#: hermetic.
CARDS = ("SYNTHETIC FAST", "SYNTHETIC SLOW", "SYNTHETIC SLOW")


def pool_model(free_mib=(20000.0, 12000.0, 12000.0)) -> PhasePoolModel:
    return PhasePoolModel(
        free_mib=free_mib,
        weight_mib_per_layer=100.0,
        # 1 MiB per token per attention layer: a token costs a whole MiB, so
        # the pool numbers below are small and readable by hand.
        kv_mib_per_token_per_attn_layer=1.0,
        arming_floor_mib=(1000.0, 1000.0, 1000.0),
    )


def solve(cap_tokens, free_mib=(20000.0, 12000.0, 12000.0), **kw):
    return solve_launch_cut(
        layer_families=FAMILIES,
        incumbent_layers=INCUMBENT,
        measured_ms_per_layer=MS_PER_LAYER,
        measured_provenance="SYNTHETIC (unit test)",
        card_names=CARDS,
        pool_model=pool_model(free_mib),
        cap_tokens=cap_tokens,
        **kw,
    )


class TestMakespanMovesLayersUntilThePoolBinds(unittest.TestCase):
    def test_loose_floor_puts_layers_on_the_fast_card(self):
        """With capacity to spare, makespan is the only term that moves."""
        decision = solve(cap_tokens=1)
        chosen = decision.chosen
        self.assertGreater(
            chosen.layers[0],
            INCUMBENT[0],
            f"the fast card must gain layers, got {chosen.layers}",
        )
        # And the balance is real: the slowest stage is faster than the
        # incumbent's slowest stage (8 layers x 4 ms = 32 ms).
        self.assertLess(chosen.makespan_ms, 32.0)
        self.assertFalse(decision.pinned)

    def test_a_binding_floor_pulls_layers_back_off_the_fast_card(self):
        """The floor is a CONSTRAINT, not a term traded against makespan.

        Stage 0's free bytes fall as it takes layers (weights) while its
        attention count rises, so its capacity falls twice over. Raising the
        floor must therefore move the chosen cut back toward the incumbent --
        and the makespan must get WORSE, which is what a constraint doing
        work looks like.
        """
        loose = solve(cap_tokens=1).chosen
        tight = solve(cap_tokens=4000).chosen
        self.assertLess(
            tight.layers[0],
            loose.layers[0],
            f"the floor did not bind: loose={loose.layers} tight={tight.layers}",
        )
        self.assertGreater(tight.makespan_ms, loose.makespan_ms)
        self.assertGreaterEqual(tight.pool_tokens, 4000)

    def test_no_feasible_cut_is_refused_with_the_best_pool_and_the_cap(self):
        with self.assertRaises(PPCutRefused) as caught:
            solve(cap_tokens=5000)
        message = str(caught.exception)
        self.assertIn("W40", message)
        self.assertIn("5000 tokens", message)  # the cap, printed
        # "SERVABLE" since FOLLOW FIX 2 / finding 3: the row is the best pool
        # among the layouts that can actually be RUN, and a priced-but-
        # excluded gapped map holding more is now named beside it rather than
        # dropped.
        self.assertIn("best servable cut", message)
        self.assertIn("short by", message)

    def test_both_objectives_are_printed(self):
        decision = solve(cap_tokens=1)
        line = decision.provenance_line()
        self.assertTrue(line.startswith("PP-CUT solver:"), line)
        self.assertIn("makespan_ms=", line)
        self.assertIn("pool_tokens=", line)
        self.assertIn("constraint pool >= 1", line)
        self.assertIn("alternatives: kv-floor cut", line)
        # The kv-floor alternative really is the pool-maximal one, so the two
        # objectives can be seen to disagree.
        self.assertGreaterEqual(
            decision.kv_floor.pool_tokens, decision.chosen.pool_tokens
        )


class TestOverride(unittest.TestCase):
    def test_explicit_override_wins_and_says_PINNED(self):
        decision = solve(cap_tokens=1, pinned_layers=INCUMBENT)
        self.assertEqual(decision.chosen.layers, tuple(INCUMBENT))
        self.assertTrue(decision.pinned)
        line = decision.provenance_line()
        self.assertIn("PINNED (user override)", line)
        self.assertIn("layers=16,8,8", line)
        # The override is still PRICED on both axes, never merely accepted.
        self.assertAlmostEqual(decision.chosen.makespan_ms, 32.0, places=6)
        self.assertGreater(decision.chosen.pool_tokens, 0.0)

    def test_a_solved_cut_carries_no_PINNED_word(self):
        self.assertNotIn("PINNED", solve(cap_tokens=1).provenance_line())

    def test_an_unrealizable_pinned_pair_is_refused(self):
        with self.assertRaises(PPCutRefused) as caught:
            solve(cap_tokens=1, pinned_layers=(16, 8, 8), pinned_attn=(2, 3, 3))
        self.assertIn("not realizable", str(caught.exception))


class TestTheAttentionAxisIsNotAFreeAllocation(unittest.TestCase):
    """The premise a naive solver would encode, refuted on the real shape.

    MEASURED against the runtime's own derivation (see the launcher's round
    trip): asking for 43,10,11 layers with 5,6,5 attention on the 64-layer
    period-4 checkpoint comes back as 23,24,17 -- snapped, not refused.
    """

    def test_the_contiguous_split_determines_the_attention_counts(self):
        self.assertTrue(
            attention_split_is_realizable(
                FAMILIES, (16, 8, 8), attention_counts(FAMILIES, (16, 8, 8))
            )
        )

    def test_a_free_allocation_is_not_realizable(self):
        # 24 layers cannot hold only 2 attention layers when every 4th layer
        # is one.
        self.assertFalse(
            attention_split_is_realizable(FAMILIES, (24, 4, 4), (2, 3, 3))
        )

    def test_the_solver_only_ever_returns_realizable_pairs(self):
        decision = solve(cap_tokens=1)
        for cand in (decision.chosen, decision.kv_floor):
            self.assertTrue(
                attention_split_is_realizable(FAMILIES, cand.layers, cand.attn),
                f"{cand.layers} / {cand.attn} is not realizable",
            )


if __name__ == "__main__":
    unittest.main()
