# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0
"""#735 step 1: the boot ticket derives its split and fires its own guards.

The ticket is a turnkey script that goes on a GPU window list, which means
nobody will be watching it closely when it runs. Two things therefore have to
hold before it gets there.

**The split must be derived, not typed.** DESIGN 9.1's example is two-stage
(``"0-31;32-63"``) while the live cut is three stages from
``--pp-stage-ratio 31,17,16`` plus ``--pp-attn-stage-ratio 7,5,4``. The ratio
is not trivially the boundary set either: ``derive_pp_layer_split`` snaps each
boundary so the requested number of FULL-ATTENTION layers lands on each side. A
snap that moved a boundary would produce two arms that are not comparable, and
byte-identity would fail for a reason that has nothing to do with the code
under test. So the script calls the planner and this file checks the result.

**The 9.2 guards must actually fire.** They are parser- and guard-level, need
no GPU, and are asserted here rather than left for the window.
"""

import importlib.util
import sys
import unittest
from pathlib import Path

from sglang.srt.distributed.utils import PPLayerSetError, parse_pp_layer_sets
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_SCRIPT = Path(__file__).resolve().parents[4] / "scripts" / "boot_735_step1.py"
assert _SCRIPT.exists(), f"boot ticket not found at {_SCRIPT}"
_spec = importlib.util.spec_from_file_location("boot_735_step1", _SCRIPT)
ticket = importlib.util.module_from_spec(_spec)
# Register BEFORE exec: the script's dataclasses use postponed annotations, and
# resolving those needs `sys.modules[cls.__module__]` to exist.
sys.modules[_spec.name] = ticket
_spec.loader.exec_module(ticket)


class TestTheSplitIsDerived(CustomTestCase):
    def test_the_live_ratio_yields_the_three_stage_set(self):
        layer_set, ranges, counts = ticket.derive_layer_set()
        self.assertEqual(layer_set, "0-30;31-47;48-63")
        self.assertEqual(counts, [31, 17, 16])
        self.assertEqual(ranges, [(0, 30), (31, 47), (48, 63)])

    def test_it_is_not_the_designs_two_stage_example(self):
        """CAN-FAIL guard: copying DESIGN 9.1 verbatim would be wrong here."""
        layer_set, _, counts = ticket.derive_layer_set()
        self.assertNotEqual(layer_set, "0-31;32-63")
        self.assertEqual(len(counts), 3)

    def test_the_attention_split_matches_the_attn_ratio(self):
        """7/5/4 FA layers -- the snap is what guarantees this, so check it."""
        _, ranges, _ = ticket.derive_layer_set()
        fa = ticket.is_full_attention_mask()
        per_stage = [sum(1 for L in range(lo, hi + 1) if fa[L]) for lo, hi in ranges]
        self.assertEqual(per_stage, ticket.PP_ATTN_STAGE_RATIO)

    def test_the_derived_set_is_accepted_by_the_real_parser(self):
        """The two halves must agree: what the planner derives, the parser eats."""
        layer_set, _, _ = ticket.derive_layer_set()
        owned = parse_pp_layer_sets(layer_set, ticket.NUM_HIDDEN_LAYERS, 3)
        self.assertEqual(len(owned), 3)
        self.assertEqual(
            sorted(L for s in owned for L in s), list(range(ticket.NUM_HIDDEN_LAYERS))
        )

    def test_ownership_is_contiguous_which_is_step_ones_premise(self):
        _, ranges, _ = ticket.derive_layer_set()
        ticket.verify_split_is_contiguous_and_complete(ranges)

    def test_a_gapped_set_is_refused_as_not_step_one(self):
        """CAN-FAIL PROOF: the contiguity check must be able to refuse."""
        with self.assertRaises(SystemExit) as ctx:
            ticket.verify_split_is_contiguous_and_complete([(0, 30), (32, 63)])
        self.assertIn("REFUSING", str(ctx.exception))

    def test_an_incomplete_set_is_refused(self):
        with self.assertRaises(SystemExit) as ctx:
            ticket.verify_split_is_contiguous_and_complete([(0, 30), (31, 62)])
        self.assertIn("exactly once", str(ctx.exception))


class TestTheGuardsFire(CustomTestCase):
    def test_all_three_designs_92_refusals_fire(self):
        results = ticket.fire_refusals()
        self.assertEqual(len(results), 3)
        for r in results:
            self.assertTrue(r.fired, msg=f"{r.name} did not fire: {r.message}")

    def test_each_refusal_names_what_it_refused(self):
        by_name = {r.name: r for r in ticket.fire_refusals()}
        self.assertIn("[3, 7]", by_name["layer-set + PD disagg"].message)
        self.assertIn("42", by_name["unconverted forward loop"].message)
        self.assertIn("[30]", by_name["malformed layer set"].message)

    def test_the_omitted_layer_probe_really_omits_one(self):
        """CAN-FAIL guard: the malformed probe must be malformed."""
        with self.assertRaises(PPLayerSetError):
            parse_pp_layer_sets("0-29;31-47;48-63", ticket.NUM_HIDDEN_LAYERS, 3)
        # ...and the corrected form must be accepted, so the probe differs from
        # the real set by exactly the omission.
        parse_pp_layer_sets("0-30;31-47;48-63", ticket.NUM_HIDDEN_LAYERS, 3)


class TestTheAcceptanceScan(CustomTestCase):
    def test_a_firing_guard_in_the_log_is_fatal(self):
        hits = ticket.scan_log_for_silent_guard(
            "... RuntimeError: PPMissingLayer for layer 42 was called ..."
        )
        self.assertIn("PPMissingLayer", hits)

    def test_a_clean_log_scans_clean(self):
        self.assertEqual(ticket.scan_log_for_silent_guard("KV Cache is allocated."), [])

    def test_byte_identity_comparison_reports_the_first_divergence(self):
        ok, msg = ticket.compare_arms("a\nb\nc", "a\nb\nc")
        self.assertTrue(ok)
        ok, msg = ticket.compare_arms("a\nb\nc", "a\nX\nc")
        self.assertFalse(ok)
        self.assertIn("line 1", msg)

    def test_the_two_arms_differ_only_in_the_env_var(self):
        layer_set, _, _ = ticket.derive_layer_set()
        base, setarm = ticket.arms(layer_set)
        base_env = base.env({"KEEP": "1"})
        set_env = setarm.env({"KEEP": "1"})
        self.assertNotIn(ticket.LAYER_SET_ENV, base_env)
        self.assertEqual(set_env[ticket.LAYER_SET_ENV], layer_set)
        self.assertEqual(
            {k: v for k, v in base_env.items() if k != ticket.LAYER_SET_ENV},
            {k: v for k, v in set_env.items() if k != ticket.LAYER_SET_ENV},
        )


if __name__ == "__main__":
    unittest.main()
