"""CPU unit tests for the benchmark scenario model."""

import json
import os
import tempfile
import dataclasses
import unittest

from sglang.srt.planner.scenarios import (
    SCENARIOS,
    build_harness_command,
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


class TestHarnessBinding(CustomTestCase):
    """Scenarios DRIVE the existing harnesses; they do not reimplement them."""

    def test_every_scenario_binds_to_an_existing_harness(self):
        for key, s in SCENARIOS.items():
            self.assertIsNotNone(s.harness, f"{key} has no harness binding")
            self.assertTrue(s.harness.module.startswith("sglang."), key)

    def test_the_bound_module_actually_exists(self):
        import importlib

        for s in SCENARIOS.values():
            importlib.import_module(s.harness.module)

    def test_axis_flags_are_real_harness_flags(self):
        import pathlib

        src = pathlib.Path("python/sglang/benchmark/serving.py")
        if not src.is_file():
            self.skipTest("harness source not in this checkout")
        text = src.read_text()
        for s in SCENARIOS.values():
            for flag in s.harness.axis_flags.values():
                if flag.startswith("--"):
                    self.assertIn(f'"{flag}"', text, f"{flag} is not a harness flag")

    def test_command_is_built_for_a_sweep_point(self):
        s = SCENARIOS["concurrent_prefill_capacity"]
        out = build_harness_command(s, {"concurrency": 8, "prompt_len": 8192})
        self.assertTrue(out["runnable"])
        self.assertIn("--max-concurrency 8", out["command"])
        self.assertIn("--random-input-len 8192", out["command"])
        self.assertIn("sglang.benchmark.serving", out["command"])

    def test_axes_the_harness_cannot_set_surface_as_manual_steps(self):
        """A control axis silently dropped would make the sweep run the same
        point repeatedly and still look complete."""
        s = SCENARIOS["power_target_sweep"]
        out = build_harness_command(s, {"power_limit_w": "70%"})
        self.assertEqual(out["external"][0]["axis"], "power_limit_w")
        self.assertIn("host", out["external"][0]["apply"])
        self.assertEqual(out["unmapped_axes"], [])

    def test_server_side_axis_is_marked_as_needing_a_restart(self):
        s = SCENARIOS["spill_latency_under_concurrency"]
        out = build_harness_command(s, {"spill_enabled": True, "sessions": 3})
        ext = {e["axis"]: e for e in out["external"]}
        self.assertIn("restart", ext["spill_enabled"]["apply"])
        self.assertIn("--max-concurrency 3", out["command"])

    def test_reproducibility_flags_are_fixed(self):
        out = build_harness_command(SCENARIOS["noise_floor"], {})
        self.assertIn("--seed", out["command"])
        self.assertIn("--flush-cache", out["command"])
        self.assertIn("--warmup-requests", out["command"])

    def test_windows_travel_with_the_command(self):
        out = build_harness_command(
            SCENARIOS["spill_latency_under_concurrency"], {"sessions": 2}
        )
        self.assertIn("restore_transient", out["windows"])


class TestTheYardstick(CustomTestCase):
    """ms per round is the comparison basis; throughput rides along, demoted.

    The noise floors that force this were measured boot-to-boot on this rig:
    0.08-0.37 % (device) and 0.98-1.46 % (host) for the round time against
    2.7-4.2 % for raw tok/s.
    """

    def test_throughput_is_never_a_primary_metric(self):
        for key, s in SCENARIOS.items():
            pm = s.primary_metric
            self.assertIsNotNone(pm, key)
            self.assertNotEqual(pm.unit, "tok/s", f"{key} decides on throughput")

    def test_throughput_metrics_carry_the_reason_they_are_not_decisive(self):
        for key, s in SCENARIOS.items():
            for m in s.metrics:
                if m.unit == "tok/s":
                    self.assertTrue(
                        m.context_only, f"{key}: {m.key} is shown without a caveat"
                    )

    def test_a_round_time_never_travels_without_its_accept_length(self):
        """Under speculation a round time alone describes an unknown amount of
        work: tok/s = accept length / round time."""
        for key, s in SCENARIOS.items():
            keys = {m.key for m in s.metrics}
            if any("verify_round" in k for k in keys):
                self.assertIn("accept_length", keys, f"{key} has no accept length")

    def test_generating_scenarios_measure_a_round_time(self):
        for key in ("noise_floor", "power_target_sweep"):
            keys = {m.key for m in SCENARIOS[key].metrics}
            self.assertIn("ms_per_verify_round", keys)

    def test_every_harness_field_maps_to_a_declared_metric(self):
        for key, s in SCENARIOS.items():
            if not s.harness:
                continue
            declared = {m.key for m in s.metrics}
            for mk in s.harness.metric_fields:
                self.assertIn(mk, declared, f"{key}: {mk} is not a declared metric")

    def test_the_noise_floor_is_established_per_metric_not_once(self):
        rule = SCENARIOS["noise_floor"].stop_rules[0].condition
        self.assertIn("round time", rule)
        self.assertIn("throughput", rule)


class TestHarnessBindingsAreReal(CustomTestCase):
    """A scenario must not emit a command the shipped harness cannot parse.

    The whole point of binding to sglang's own load generators instead of
    writing another one is that the flags and result fields already exist; a
    binding that has drifted turns that saving into a trap, so it is checked
    against the harness source and the harness result type rather than against
    a copy of them.
    """

    def _harness_source(self, module: str) -> str:
        import importlib

        mod = importlib.import_module(module)
        with open(mod.__file__) as f:
            return f.read()

    def test_every_bound_module_imports(self):
        import importlib

        for key, s in SCENARIOS.items():
            if s.harness:
                importlib.import_module(s.harness.module)

    def test_every_emitted_flag_exists_in_the_harness(self):
        for key, s in SCENARIOS.items():
            if not s.harness:
                continue
            src = self._harness_source(s.harness.module)
            flags = [f.split()[0] for f in s.harness.fixed_flags]
            flags += [
                v for v in s.harness.axis_flags.values() if v.startswith("--")
            ]
            for flag in flags:
                self.assertIn(
                    f'"{flag}"', src, f"{key}: {flag} is not a flag of {s.harness.module}"
                )

    def test_every_harness_result_field_exists_on_the_result_type(self):
        from sglang.benchmark.serving import BenchmarkMetrics

        fields = {f.name for f in dataclasses.fields(BenchmarkMetrics)}
        for key, s in SCENARIOS.items():
            if not s.harness or s.harness.module != "sglang.benchmark.serving":
                continue
            for metric_key, field in s.harness.metric_fields.items():
                # A parenthesised entry names a source the harness does not
                # have -- the collector, or the engine's own exposition.
                if field.startswith("("):
                    continue
                self.assertIn(
                    field, fields, f"{key}: {metric_key} reads a missing field"
                )

    def test_the_command_it_prints_is_the_module_that_exists(self):
        from sglang.srt.planner.scenarios import build_harness_command

        s = SCENARIOS["noise_floor"]
        cmd = build_harness_command(s, {}, base_url="http://x:1")
        self.assertTrue(cmd["command"].startswith("python -m sglang.benchmark.serving"))
        self.assertTrue(cmd["runnable"])


if __name__ == "__main__":
    unittest.main()
