# SPDX-License-Identifier: Apache-2.0
"""#485 rescoped: the boot-time layout chooser, driven by the REAL solver.

The chooser's whole contract is that it ADDS INFORMATION AND CHANGES NOTHING:
the cut it reports is the cut ``solve_pp_cut`` produced, and the flags it emits
are byte-identical to what the incumbent path already produces. So these tests
drive the real solver with the real #485 fixture rather than a stand-in -- a
stub would let the chooser and the solver drift, which is the one failure this
design cannot tolerate (#624 stub-drift).
"""

import sys
import unittest
from os.path import dirname

sys.path.insert(0, dirname(__file__))

from test_pp_family_cut_485 import _inputs  # noqa: E402  (real fixture, reused)

from sglang.srt.planner import pp_cut  # noqa: E402
from sglang.srt.planner.boot_layout import (  # noqa: E402
    PHASE_PP_PREFILL,
    PRICED_FAMILIES,
    UNFUNDED_FAMILIES,
    BootLayoutError,
    choose_boot_layout,
    describe,
    family_placement,
)


def _solved():
    inp = _inputs()
    return pp_cut.solve_pp_cut(inp), inp


class TestItChangesNothing(unittest.TestCase):
    """RED-FIRST: the chooser must reproduce the solved cut exactly."""

    def test_the_counts_are_the_solutions_counts_verbatim(self):
        sol, inp = _solved()
        layout = choose_boot_layout(sol, inp)
        self.assertEqual(layout.counts, tuple(sol.counts))

    def test_the_layer_ratio_flag_is_byte_identical_to_as_layer_ratio(self):
        sol, inp = _solved()
        layout = choose_boot_layout(sol, inp)
        expected = "--pp-layer-ratio " + ",".join(
            str(c) for c in sol.as_layer_ratio()
        )
        self.assertIn(expected, layout.flags)

    def test_the_attn_ratio_is_the_SAME_QUANTITY_server_args_prints(self):
        """One authority: #713's paste-ready remedy is the full-attention
        counts of the ranges, which is exactly `attention_counts`. If these
        ever diverge, the printed remedy and the solved layout have drifted."""
        sol, inp = _solved()
        layout = choose_boot_layout(sol, inp)
        expected = "--pp-attn-stage-ratio " + ",".join(
            str(c) for c in sol.attention_counts
        )
        self.assertIn(expected, layout.flags)

    def test_it_does_not_re_solve(self):
        """No second solver -- the rescope's acceptance condition."""
        import inspect

        from sglang.srt.planner import boot_layout

        src = inspect.getsource(boot_layout)
        self.assertNotIn("solve_pp_cut(", src)
        self.assertNotIn("_price_stage", src)


class TestAbsentIsNotZero(unittest.TestCase):
    """#606 canon: a family the model does not have is ABSENT, not zero."""

    def test_a_pure_attention_model_has_NO_linear_placement(self):
        fams = tuple([pp_cut.LAYER_FAMILY_ATTENTION] * 12)
        placements = family_placement(fams, (4, 4, 4))
        names = [p.family for p in placements]
        self.assertEqual(names, [pp_cut.LAYER_FAMILY_ATTENTION])
        self.assertNotIn(pp_cut.LAYER_FAMILY_LINEAR, names)

    def test_asking_for_an_absent_family_RAISES_rather_than_returning_zero(self):
        sol, inp = _solved()
        layout = choose_boot_layout(sol, inp)
        with self.assertRaises(BootLayoutError) as e:
            layout.family("experts")
        self.assertIn("ABSENT, not zero", str(e.exception))

    def test_a_hybrid_model_carries_BOTH_families(self):
        sol, inp = _solved()
        layout = choose_boot_layout(sol, inp)
        names = {f.family for f in layout.families}
        self.assertIn(pp_cut.LAYER_FAMILY_ATTENTION, names)
        self.assertIn(pp_cut.LAYER_FAMILY_LINEAR, names)


class TestThePlacementReconciles(unittest.TestCase):
    def test_every_family_totals_its_tagged_layers(self):
        sol, inp = _solved()
        layout = choose_boot_layout(sol, inp)
        for f in layout.families:
            self.assertEqual(
                f.total, sum(1 for t in inp.layer_families if t == f.family)
            )

    def test_the_families_sum_to_the_depth(self):
        sol, inp = _solved()
        layout = choose_boot_layout(sol, inp)
        self.assertEqual(
            sum(f.total for f in layout.families), len(inp.layer_families)
        )

    def test_attention_placement_matches_the_solutions_own_vector(self):
        sol, inp = _solved()
        layout = choose_boot_layout(sol, inp)
        self.assertEqual(
            layout.family(pp_cut.LAYER_FAMILY_ATTENTION).per_stage,
            tuple(sol.attention_counts),
        )


class TestTheEnumerationGapIsNamedNotInvented(unittest.TestCase):
    """#253: no speculative cost terms. The gap is DATA, not a fabricated
    number."""

    def test_only_the_two_measured_families_are_priced(self):
        self.assertEqual(
            set(PRICED_FAMILIES),
            {pp_cut.LAYER_FAMILY_ATTENTION, pp_cut.LAYER_FAMILY_LINEAR},
        )

    def test_every_unfunded_family_says_what_would_fund_it(self):
        """Caught vacuous on the first pass: the assertion concatenated the
        needle onto the haystack, so it could never fail. A gap entry that
        does not say what would close it is just a shrug with a name."""
        for name, why in UNFUNDED_FAMILIES.items():
            self.assertTrue(why.strip(), f"{name} has no funding statement")
            self.assertIn(
                "would need",
                why.lower(),
                f"{name} names no measurement that would fund it",
            )

    def test_the_law_s_families_are_all_accounted_for(self):
        """Either priced, or named as a gap -- nothing silently dropped."""
        for fam in ("kv_heads", "vocab", "experts", "nonlinear_kernels",
                    "linear_per_quant_lane"):
            self.assertIn(fam, UNFUNDED_FAMILIES)

    def test_no_cost_term_was_invented_for_an_unfunded_family(self):
        import inspect

        from sglang.srt.planner import boot_layout

        src = inspect.getsource(boot_layout)
        for bad in ("expert_layer_weight_bytes", "vocab_weight_bytes",
                    "kv_head_flops", "nonlinear_flops"):
            self.assertNotIn(bad, src)

    def test_the_boot_record_NAMES_the_gap(self):
        sol, inp = _solved()
        text = describe(choose_boot_layout(sol, inp))
        self.assertIn("no cost term in this solver", text)
        self.assertIn("experts", text)


class TestItRefusesRatherThanMisleading(unittest.TestCase):
    def test_an_infeasible_solution_emits_NO_flags(self):
        sol, inp = _solved()
        bad = type(sol)(**{**sol.__dict__, "feasible": False,
                           "refusals": ("rank2: out of budget",)})
        with self.assertRaises(BootLayoutError) as e:
            choose_boot_layout(bad, inp)
        self.assertIn("do not paste this layout", str(e.exception))

    def test_a_mismatched_model_is_refused(self):
        sol, inp = _solved()
        short = type(inp)(**{**inp.__dict__,
                             "layer_families": inp.layer_families[:-1]})
        with self.assertRaises(BootLayoutError) as e:
            choose_boot_layout(sol, short)
        self.assertIn("same model", str(e.exception))

    def test_the_phase_is_NAMED_not_assumed(self):
        """#485's matrix has a decode column; this solver does not solve it."""
        sol, inp = _solved()
        self.assertEqual(choose_boot_layout(sol, inp).phase, PHASE_PP_PREFILL)


if __name__ == "__main__":
    unittest.main()
