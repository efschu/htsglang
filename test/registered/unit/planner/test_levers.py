"""CPU unit tests for the five levers."""

import unittest

from sglang.srt.planner.levers import (
    LEVERS,
    RATES_KNOWN,
    STRUCTURE_ONLY,
    Confidence,
    FlagSpec,
    Lever,
    flag_available,
    missing_flags,
    render_levers_text,
    suggest_levers,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

PROBE = {
    "created": "2026-07-21 07:06:43",
    "gpus": {"GPU-a": {"gemm_tflops": 233.9, "membw_gbs": 1664.1}},
}


class TestLeverDefinitions(CustomTestCase):
    def test_the_five_directions_exist(self):
        self.assertEqual(
            set(LEVERS),
            {"context", "decode_speed", "prefill_speed", "ttft_loaded", "energy"},
        )

    def test_every_lever_states_a_price(self):
        """A lever that lists only what it improves is a sales pitch."""
        for key, p in LEVERS.items():
            self.assertTrue(p.gains, key)
            self.assertTrue(p.costs, f"{key} names no cost")
            self.assertTrue(p.maximises, key)
            self.assertTrue(p.vague_statement, f"{key} has no pre-probe statement")

    def test_opposing_levers_name_each_other(self):
        self.assertIn("decode_speed", LEVERS["context"].tradeoff_against)
        self.assertIn("context", LEVERS["decode_speed"].tradeoff_against)
        self.assertIn("decode_speed", LEVERS["energy"].tradeoff_against)

    def test_every_flag_states_why(self):
        for key, p in LEVERS.items():
            for f in p.flags:
                self.assertTrue(f.why, f"{key}: {f.flag} has no rationale")


class TestBuildGate(CustomTestCase):
    def test_flag_availability_is_read_from_server_args(self):
        self.assertTrue(flag_available("rank_tp_ratio"))
        self.assertFalse(flag_available("definitely_not_a_field"))

    def test_a_lever_whose_knob_is_absent_says_so(self):
        """A lever must never emit a flag the running build cannot parse.

        Exercised on a synthetic lever rather than on a real one: which flags
        exist is a property of the branch under test, so asserting a specific
        absence would make the test a claim about the branch instead of about
        the gate.
        """
        synthetic = Lever(
            key="synthetic",
            label="Synthetic",
            maximises="nothing",
            flags=[
                FlagSpec(
                    "--definitely-not-a-flag",
                    "1",
                    "definitely_not_a_field",
                    why="exists only to be missing",
                )
            ],
            gains=["none"],
            costs=["none"],
            vague_statement="none",
        )
        missing = missing_flags(synthetic)
        self.assertEqual([f.flag for f in missing], ["--definitely-not-a-flag"])

    def test_the_decode_lever_has_its_knob_on_this_build(self):
        """#210 shipped --rank-kv-ratio; the decode lever sits in the KV-token
        split, so on this build it must resolve to a real flag rather than to
        an unavailability notice."""
        self.assertTrue(flag_available("rank_kv_ratio"))
        self.assertEqual(missing_flags(LEVERS["decode_speed"]), [])
        s = [x for x in suggest_levers(probe=PROBE) if x.lever.key == "decode_speed"][0]
        self.assertEqual(s.unavailable_flags, [])
        self.assertIn("--rank-kv-ratio speed", " ".join(s.command_flags))
        # 'speed' reads the per-rank bandwidth scores, which only the
        # auto-performance plan resolves; without it the mode degrades to the
        # opposite direction, so the lever must carry it.
        self.assertIn("--rank-tp-ratio auto-performance", " ".join(s.command_flags))

    def test_the_context_lever_asks_for_capacity_ownership(self):
        s = [x for x in suggest_levers(probe=PROBE) if x.lever.key == "context"][0]
        self.assertIn("--rank-kv-ratio capacity", " ".join(s.command_flags))

    def test_available_flags_are_still_emitted(self):
        s = [x for x in suggest_levers(probe=PROBE) if x.lever.key == "context"][0]
        self.assertIn("--rank-tp-ratio auto", " ".join(s.command_flags))


class TestStaging(CustomTestCase):
    def test_without_a_probe_suggestions_are_vague(self):
        for s in suggest_levers(probe=None):
            if s.confidence == Confidence.BLOCKED:
                continue
            self.assertEqual(s.stage, STRUCTURE_ONLY)
            self.assertEqual(s.confidence, Confidence.VAGUE)
            self.assertEqual(s.statement, s.lever.vague_statement)

    def test_rate_dependent_flags_are_withheld_before_a_probe(self):
        s = [x for x in suggest_levers(probe=None) if x.lever.key == "prefill_speed"][0]
        self.assertNotIn("--rank-perf-tune", " ".join(s.command_flags))

    def test_with_a_probe_suggestions_become_concrete(self):
        s = [x for x in suggest_levers(probe=PROBE) if x.lever.key == "prefill_speed"][0]
        self.assertEqual(s.stage, RATES_KNOWN)
        self.assertEqual(s.confidence, Confidence.DETAILED)
        self.assertIn("--rank-perf-tune enc", " ".join(s.command_flags))
        self.assertIn("Price:", s.statement)

    def test_fit_levers_do_not_need_a_probe_for_their_main_flag(self):
        """Fit questions are answerable without measurement; speed questions
        are not."""
        s = [x for x in suggest_levers(probe=None) if x.lever.key == "context"][0]
        self.assertIn("--rank-tp-ratio auto", " ".join(s.command_flags))


class TestHomogeneousRig(CustomTestCase):
    def test_heterogeneous_only_flags_are_dropped_on_a_uniform_rig(self):
        s = [
            x
            for x in suggest_levers(heterogeneous=False, probe=PROBE)
            if x.lever.key == "context"
        ][0]
        self.assertNotIn("--rank-tp-ratio", " ".join(s.command_flags))


class TestPreconditions(CustomTestCase):
    def test_ttft_lever_needs_a_second_node(self):
        s = [x for x in suggest_levers(probe=PROBE, node_count=1)
             if x.lever.key == "ttft_loaded"][0]
        self.assertTrue(s.unmet_preconditions)
        self.assertIn("second node", " ".join(s.unmet_preconditions))
        s2 = [x for x in suggest_levers(probe=PROBE, node_count=2)
              if x.lever.key == "ttft_loaded"][0]
        self.assertNotIn("second node", " ".join(s2.unmet_preconditions))

    def test_energy_lever_needs_power_control(self):
        s = [x for x in suggest_levers(probe=PROBE, facility_keys_available=[])
             if x.lever.key == "energy"][0]
        self.assertIn("Power-target", " ".join(s.unmet_preconditions))
        s2 = [
            x
            for x in suggest_levers(
                probe=PROBE, facility_keys_available=["power_target"]
            )
            if x.lever.key == "energy"
        ][0]
        self.assertEqual(s2.unmet_preconditions, [])


class TestRendering(CustomTestCase):
    def test_json_carries_gains_costs_and_counter_reckoning(self):
        d = suggest_levers(probe=PROBE)[0].to_json()
        for k in ("gains", "costs", "counter_reckoning", "statement", "confidence"):
            self.assertIn(k, d)

    def test_text_states_the_evidence_stage(self):
        txt = render_levers_text(suggest_levers(probe=None))
        self.assertIn("structure only", txt)
        txt2 = render_levers_text(suggest_levers(probe=PROBE))
        self.assertIn("measured rates", txt2)

    def test_text_shows_the_counter_reckoning(self):
        txt = render_levers_text(
            suggest_levers(probe=PROBE, keys=["context"])
        )
        self.assertIn("costs decode_speed", txt)

    def test_filtering_by_key(self):
        out = suggest_levers(probe=PROBE, keys=["energy"])
        self.assertEqual([s.lever.key for s in out], ["energy"])


if __name__ == "__main__":
    unittest.main()
