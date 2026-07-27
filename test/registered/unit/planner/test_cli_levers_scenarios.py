"""CPU unit tests for the unified planner CLI (#220): levers and scenarios."""

import io
import unittest
from contextlib import redirect_stderr, redirect_stdout

from sglang.srt.planner.cli import main
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def run(argv):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = main(argv)
    return rc, out.getvalue(), err.getvalue()


class TestLeversThroughThePlannerCli(CustomTestCase):
    def test_levers_print_all_five_directions(self):
        rc, out, _ = run(["--levers"])
        self.assertEqual(rc, 0)
        for label in ("Context", "Decode speed", "Prefill speed", "TTFT", "Energy"):
            self.assertIn(label, out)

    def test_levers_state_the_evidence_stage(self):
        rc, out, _ = run(["--levers"])
        self.assertTrue("structure only" in out or "measured rates" in out)

    def test_a_subset_can_be_selected(self):
        rc, out, _ = run(["--levers", "--lever", "energy"])
        self.assertEqual(rc, 0)
        self.assertIn("Energy per token", out)
        self.assertNotIn("max_total_num_tokens", out)

    def test_json_carries_the_counter_reckoning(self):
        import json

        rc, out, _ = run(["--levers", "--json"])
        data = json.loads(out)
        self.assertEqual(len(data), 5)
        for entry in data:
            self.assertIn("counter_reckoning", entry)
            self.assertTrue(entry["costs"], entry["key"])

    def test_a_uniform_rig_drops_the_heterogeneous_flags(self):
        rc, out, _ = run(["--levers", "--homogeneous", "--lever", "context"])
        self.assertNotIn("--rank-tp-ratio", out)

    def test_the_ttft_lever_reports_its_unmet_precondition(self):
        rc, out, _ = run(["--levers", "--lever", "ttft_loaded", "--nodes", "1"])
        self.assertIn("second node", out)


class TestScenariosThroughThePlannerCli(CustomTestCase):
    def test_listing_names_every_scenario_with_its_question(self):
        rc, out, _ = run(["--list-scenarios"])
        self.assertEqual(rc, 0)
        self.assertIn("noise_floor", out)
        self.assertIn("?", out)

    def test_one_scenario_renders_question_space_metric_and_stop_rule(self):
        rc, out, _ = run(["--scenario", "power_target_sweep"])
        self.assertEqual(rc, 0)
        for section in ("Question:", "Parameter space:", "Measured:", "Stop when:"):
            self.assertIn(section, out)

    def test_the_primary_metric_is_a_round_time_not_throughput(self):
        rc, out, _ = run(["--scenario", "power_target_sweep"])
        self.assertIn("time per verify round [ms, lower_better] (primary)", out)

    def test_throughput_is_shown_with_the_reason_it_is_not_decisive(self):
        rc, out, _ = run(["--scenario", "power_target_sweep"])
        self.assertIn("context only:", out)

    def test_it_prints_the_command_that_drives_the_existing_harness(self):
        rc, out, _ = run(["--scenario", "noise_floor"])
        self.assertIn("python -m sglang.benchmark.serving", out)

    def test_an_unknown_key_fails_loudly(self):
        rc, _, err = run(["--scenario", "not_a_scenario"])
        self.assertEqual(rc, 2)
        self.assertIn("no scenario", err)

    def test_a_free_text_question_picks_one(self):
        rc, out, _ = run(["--scenario-question", "what does the power cap do?"])
        self.assertEqual(rc, 0)
        self.assertIn("power_target_sweep", out)

    def test_a_question_that_matches_nothing_returns_nothing(self):
        """A confidently wrong scenario costs more than none."""
        rc, _, err = run(["--scenario-question", "zzz qqq"])
        self.assertEqual(rc, 1)
        self.assertIn("No scenario matches", err)


class TestTheOtherDoorIsClosed(CustomTestCase):
    def test_the_rigmon_cli_no_longer_offers_levers_or_scenarios(self):
        """One concept, one front door: the rigmon CLI is host telemetry."""
        from sglang.srt.rigmon.__main__ import build_parser

        actions = build_parser()._subparsers._group_actions[0].choices
        self.assertNotIn("levers", actions)
        self.assertNotIn("scenario", actions)
        self.assertIn("collect", actions)
        self.assertIn("serve", actions)


if __name__ == "__main__":
    unittest.main()
