"""#1254: the P cut objective defaults to MAKESPAN (user order 2026-09-08).

USER ORDER, verbatim: "nimm als default ab jetzt makespan".

WHAT THE DEFAULT COSTS, pinned so it stays SELECTED and not inherited: makespan
is the solver's SPEED cut (42,11,11 / attn 10,3,3 in the perf boots) and it buys
TTFT with KV -- P pool 587k solver-priced against the incumbent's 714k measured,
for +33.5 % prefill measured in pp1/pp2 only. #1254 originally took makespan
away from the default because it was being paid without anyone selecting it; it
is back because it was selected.

THE CONSUMERS FOLLOW BY CONSTRUCTION, and that is what this file's second half
pins. The chosen candidate fans out through ONE seam --
`chosen.layers/attn` -> `stage_ratio`/`attn_ratio` -> `PCutFacts` -- and the
five argv sites, the W46 split==map guard, the flip-order map, the ring form
key / checkpoint-derived spans and W55/ledger all read that one object. So a
different objective moves them together or not at all; no consumer re-derives
the incumbent for itself. A test that a consumer READ the facts is worth more
than five tests that each recompute 42,11,11.
"""

import unittest

from sglang.srt.weg2 import launcher
from sglang.test.test_utils import CustomTestCase


class TheDefaultIsMakespan(CustomTestCase):
    def test_the_parser_default(self):
        ns = launcher.build_parser().parse_args(["--tree", "/t", "--tag", "x"])
        self.assertEqual(ns.pp_solve_objective, "makespan")

    def test_the_user_order_is_quoted_at_the_flag(self):
        h = launcher.build_parser().format_help()
        self.assertIn("nimm als default ab jetzt makespan", h)

    def test_all_three_arms_stay_selectable(self):
        act = next(a for a in launcher.build_parser()._actions
                   if a.dest == "pp_solve_objective")
        self.assertEqual(set(act.choices), {"maxkv", "makespan", "incumbent"})

    def test_makespan_maps_to_the_solvers_makespan_ranking(self):
        self.assertEqual(launcher.P_SOLVER_OBJECTIVE_OF["makespan"], "makespan")
        # incumbent is a CANDIDATE, not a ranking -- unchanged by this order
        self.assertEqual(launcher.P_SOLVER_OBJECTIVE_OF["incumbent"], "maxkv")


class ThePickerShipsTheMakespanRow(CustomTestCase):
    class _Row:
        layers = (42, 11, 11)
        attn = (10, 3, 3)
        pool_tokens = 587000
        makespan_ms = 367.3

        def fmt(self):
            return "42,11,11 / 10,3,3"

    class _Decision:
        pinned = False
        ranked = ()

        def __init__(self, row):
            self.makespan = row
            self.chosen = row
            self.kv_floor = row

    def test_the_default_objective_returns_the_makespan_row(self):
        row = self._Row()
        got, why = launcher.pick_shipped_cut(
            self._Decision(row), (32, 18, 14), (8, 4, 4), "makespan")
        self.assertIs(got, row)
        self.assertIn("makespan-optimal", why)

    def test_the_reason_names_the_order_and_the_cost(self):
        row = self._Row()
        _got, why = launcher.pick_shipped_cut(
            self._Decision(row), (32, 18, 14), (8, 4, 4), "makespan")
        self.assertIn("nimm als default ab jetzt makespan", why)
        self.assertIn("BUYS TTFT WITH KV", why)

    def test_an_operator_pin_still_outranks_the_objective(self):
        """--pp-stage-ratio must remain an order, not a suggestion."""
        import inspect

        src = inspect.getsource(launcher.solve_p_cut)
        i = src.index("if decision.pinned:")
        self.assertIn("PINNED by --pp-stage-ratio", src[i:i + 400])


class TheConsumersReadTheOneSeam(CustomTestCase):
    """Requirement (2): every consumer follows the chosen cut automatically."""

    def test_the_argv_split_comes_from_the_chosen_candidate(self):
        import inspect

        src = inspect.getsource(launcher.solve_p_cut)
        self.assertIn("stage_ratio = _csv(chosen.layers)", src)
        self.assertIn("attn_ratio = _csv(chosen.attn)", src)

    def test_the_round_trip_refuses_a_cut_that_would_not_survive(self):
        """W40: the layout the provenance line describes must be the one that
        runs -- this is what makes 'the consumers follow' checkable."""
        import inspect

        src = inspect.getsource(launcher.solve_p_cut)
        self.assertIn("does not survive", src)
        self.assertIn("derive_pp_layer_split", src)

    def test_downstream_reads_pcutfacts_not_a_recomputed_incumbent(self):
        import inspect

        self.assertIn("stage_ratio, attn_stage_ratio = cut.stage_ratio, cut.attn_stage_ratio",
                      inspect.getsource(launcher.main))
        # and the incumbent constant is NOT what argv is built from
        solve = inspect.getsource(launcher.solve_p_cut)
        i = solve.index("stage_ratio = _csv(chosen.layers)")
        self.assertNotIn("P_PP_STAGE_RATIO_SCORES", solve[i:i + 600])


class ThePriceOfBothAlternativesIsPrintedEveryBoot(CustomTestCase):
    def test_the_shipped_line_prices_all_three_arms(self):
        """#1286b MOVED THE LINE AND RENAMED ONE FIELD; this guard follows it.

        The line was extracted from ``solve_p_cut`` into the pure
        ``launcher.shipped_line`` so a desk test can RENDER it -- twelve
        substitutions across one %-format whose only proof of rendering used
        to be a boot -- and ``pool_tokens=`` became ``chosen_pool=`` beside a
        new ``pool_floor=``.  This guard is about the INTENT (all three arms
        priced on the shipped line, every boot) and not about the address, so
        it moves with the line rather than being deleted; a guard left pointing
        at the old location is how a suite starts lying about what it pins,
        which is the same reason the makespan-default test above was renamed
        with its value.
        """
        import inspect

        src = inspect.getsource(launcher.shipped_line)
        # ANCHOR ON THE FORMAT LITERAL, not on any mention of the marker: the
        # extracted function names it in its own docstring one screen above,
        # and a bare `index("PP-CUT SHIPPED")` lands there and then reports the
        # format's fields missing from a window of prose.
        i = src.index('"PP-CUT SHIPPED: layers=')
        window = src[i:i + 900]
        for token in ("incumbent %s pool %s", "pool-maximal (kv-floor)",
                      "makespan-optimal", "chosen_pool=%d", "pool_floor=%s",
                      "makespan_ms=%.1f"):
            self.assertIn(token, window)
        # and it is still the line the boot emits, not an orphaned formatter
        self.assertIn("shipped_line(", inspect.getsource(launcher.solve_p_cut))


if __name__ == "__main__":
    unittest.main()
