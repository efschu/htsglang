"""CPU unit tests for the benchmark scenario model."""

import json
import os
import tempfile
import unittest

from sglang.srt.planner.scenarios import (
    SCENARIOS,
    Scenario,
    load_scenarios,
    plan_scenario,
    render_scenario_text,
    suggest,
)
from sglang.srt.rigmon.facilities import HostEnvironment, facilities
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

LXC = HostEnvironment(
    container="lxc", is_root=True, cap_eff=0x1FFFFFFFFFF, full_caps=True,
    driver_version="595.58.03",
)
HOST = HostEnvironment(container="", is_root=True)


class TestRegistry(CustomTestCase):
    def test_the_four_named_questions_exist(self):
        for key in (
            "power_target_sweep",
            "ram_clock_spill",
            "concurrent_prefill_capacity",
            "spill_latency_under_concurrency",
        ):
            self.assertIn(key, SCENARIOS)

    def test_every_scenario_is_complete(self):
        for key, s in SCENARIOS.items():
            self.assertTrue(s.question, key)
            self.assertTrue(s.hypothesis, key)
            self.assertTrue(s.falsifier, f"{key} has no falsifier")
            self.assertTrue(s.axes, f"{key} has no parameter axis")
            self.assertTrue(s.metrics, f"{key} measures nothing")
            self.assertTrue(s.stop_rules, f"{key} has no stopping rule")
            self.assertIsNotNone(s.primary_metric, key)

    def test_scenarios_measure_a_single_factor_not_aggregate_throughput(self):
        """Each scenario's primary metric must be the factor under study, not
        a bare total-throughput number — except where throughput IS the
        quantity in question."""
        prim = SCENARIOS["concurrent_prefill_capacity"].primary_metric
        self.assertEqual(prim.key, "ttft_p95")
        self.assertEqual(prim.direction, "lower_better")
        self.assertEqual(SCENARIOS["ram_clock_spill"].primary_metric.key, "restore_gbs")

    def test_noise_floor_is_required_by_default(self):
        self.assertTrue(SCENARIOS["power_target_sweep"].noise_floor_required)
        # ...and the floor scenario itself does not require itself.
        self.assertFalse(SCENARIOS["noise_floor"].noise_floor_required)


class TestWindows(CustomTestCase):
    def test_spill_scenarios_separate_the_restore_transient(self):
        s = SCENARIOS["spill_latency_under_concurrency"]
        keys = [w.key for w in s.windows]
        self.assertEqual(
            keys, ["pre_spill", "during_spill", "restore_transient", "steady_after"]
        )

    def test_restore_transient_is_excluded_from_headline(self):
        """The window after a restore carries the resume backfill, so a figure
        taken there is a catch-up rate, not a serving rate."""
        s = SCENARIOS["spill_latency_under_concurrency"]
        tr = [w for w in s.windows if w.key == "restore_transient"][0]
        self.assertTrue(tr.exclude_from_headline)
        self.assertIn("backfill", tr.description)
        steady = [w for w in s.windows if w.key == "steady_after"][0]
        self.assertFalse(steady.exclude_from_headline)

    def test_plan_warns_about_non_poolable_windows(self):
        plan = plan_scenario(SCENARIOS["spill_latency_under_concurrency"], facilities(HOST))
        joined = " ".join(plan.preflight)
        self.assertIn("restore_transient", joined)
        self.assertIn("headline", joined)


class TestSuggestion(CustomTestCase):
    def test_suggestion_follows_the_question(self):
        self.assertEqual(
            suggest("what does the power target do to efficiency?").key,
            "power_target_sweep",
        )
        self.assertEqual(
            suggest("how many concurrent prefills can the second rig take?").key,
            "concurrent_prefill_capacity",
        )
        self.assertEqual(
            suggest("does a faster ram clock help the spill offload?").key,
            "ram_clock_spill",
        )

    def test_unmatched_question_returns_nothing_rather_than_a_wrong_guess(self):
        self.assertIsNone(suggest("what colour is the case"))
        self.assertIsNone(suggest(""))


class TestPlanning(CustomTestCase):
    def test_power_sweep_is_blocked_in_a_container_with_a_remedy(self):
        plan = plan_scenario(
            SCENARIOS["power_target_sweep"], facilities(LXC, exists=lambda p: True)
        )
        self.assertFalse(plan.runnable)
        self.assertEqual(plan.blocked_axes[0]["axis"], "power_limit_w")
        missing = {m["key"]: m for m in plan.missing_facilities}
        self.assertIn("power_target", missing)
        self.assertTrue(missing["power_target"]["remedy"])
        self.assertTrue(missing["power_target"]["impossible_in_container"])

    def test_blocked_axis_is_kept_not_dropped(self):
        """A sweep silently reduced to one point looks like a finished
        measurement; the axis must stay visible as blocked."""
        plan = plan_scenario(
            SCENARIOS["power_target_sweep"], facilities(LXC, exists=lambda p: True)
        )
        axis_keys = [a.key for a in plan.scenario.axes]
        self.assertIn("power_limit_w", axis_keys)

    def test_scenario_without_controls_is_runnable_anywhere(self):
        plan = plan_scenario(
            SCENARIOS["concurrent_prefill_capacity"], facilities(LXC, exists=lambda p: True)
        )
        self.assertTrue(plan.runnable)

    def test_preflight_demands_the_noise_floor_first(self):
        plan = plan_scenario(SCENARIOS["power_target_sweep"], facilities(HOST))
        self.assertIn("noise-floor", " ".join(plan.preflight))

    def test_render_text_mentions_the_blocker(self):
        plan = plan_scenario(
            SCENARIOS["power_target_sweep"], facilities(LXC, exists=lambda p: True)
        )
        txt = render_scenario_text(plan)
        self.assertIn("NOT RUNNABLE HERE", txt)
        self.assertIn("remedy", txt)


class TestExtensibility(CustomTestCase):
    def test_a_new_question_needs_no_new_code(self):
        doc = {
            "scenarios": [
                {
                    "key": "pcie_width",
                    "question": "What does the x4 link on GPU0 cost?",
                    "hypothesis": "Host-staged collectives are limited by it.",
                    "falsifier": "Flat across widths means it is not the limit.",
                    "axes": [
                        {"key": "width", "label": "PCIe width", "values": [4, 8, 16]}
                    ],
                    "metrics": [
                        {"key": "ar_us", "label": "allreduce", "unit": "us",
                         "direction": "lower_better", "primary": True}
                    ],
                    "stop_rules": [
                        {"key": "flat", "label": "flat", "condition": "no change"}
                    ],
                    "keywords": ["pcie", "width", "link"],
                }
            ]
        }
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "extra.json")
            with open(path, "w") as f:
                json.dump(doc, f)
            reg = {}
            load_scenarios(path, into=reg)
        self.assertIn("pcie_width", reg)
        self.assertEqual(suggest("how much does pcie width matter", reg).key,
                         "pcie_width")

    def test_roundtrip(self):
        s = SCENARIOS["power_target_sweep"]
        again = Scenario.from_json(json.loads(json.dumps(s.to_json())))
        self.assertEqual(again.key, s.key)
        self.assertEqual([a.key for a in again.axes], [a.key for a in s.axes])
        self.assertEqual(
            [w.exclude_from_headline for w in again.windows],
            [w.exclude_from_headline for w in s.windows],
        )


if __name__ == "__main__":
    unittest.main()
