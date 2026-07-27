"""CPU unit tests for the scenario executor (#218 follow-up).

No GPU, no server boot, no benchmark: every external dependency of
:class:`sglang.srt.planner.runner.Study` is injected, which is the property
that makes the executor testable at all.
"""

import types
import unittest

from sglang.srt.planner.comparison import UNKNOWN, VERDICTS, NoiseFloor, WindowResult
from sglang.srt.planner.runner import (
    DEFAULT_WINDOW,
    Arm,
    HarnessOutcome,
    KvBudgetUnpinned,
    PointResult,
    RunPolicy,
    ScheduledBoot,
    Study,
    TimeBudget,
    WithinBootRefused,
    arm_result_from_points,
    build_schedule,
    card_state,
    neutralise_kv_budget,
    noise_floor_from_points,
    own_vram_gate,
    render_study_text,
    suggest_num_prompts,
    window_metrics,
    window_plan,
)
from sglang.srt.planner.scenarios import SCENARIOS
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

NOISE = SCENARIOS["noise_floor"]
POWER = SCENARIOS["power_target_sweep"]
SPILL = SCENARIOS["ram_clock_spill"]


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def settings(port=30000, gpus=(0, 1)):
    return types.SimpleNamespace(
        host="127.0.0.1",
        port=port,
        rank_gpu_id=list(gpus),
        python_exe="/usr/bin/python3",
        extra_env={},
    )


class FakeEngineSample:
    def __init__(self, metrics=None, phase=None):
        self.metrics = dict(metrics or {})
        self.per_rank_phase = dict(phase or {})


class FakeScraper:
    """Emits a rising pair of engine samples per window."""

    def __init__(self, verify_s=1.0, gen=1000.0, accept=2.0):
        self.calls = 0
        self.verify_s = verify_s
        self.gen = gen
        self.accept = accept

    def scrape(self):
        self.calls += 1
        n = self.calls - 1
        return FakeEngineSample(
            metrics={
                "generation_tokens_total": self.gen * n,
                "prompt_tokens_total": 5000.0 * n,
                "spec_accept_length": self.accept,
            },
            phase={0: {"target_verify": self.verify_s * n, "extend": 0.5 * n}},
        )


class FakeSupervisor:
    def __init__(self):
        self.booted = []
        self.stopped = 0
        self.proc = types.SimpleNamespace(pid=0)

    def start(self, s, **kw):
        self.booted.append(s.port)
        return {}

    def stop(self, **kw):
        self.stopped += 1
        return {}


class FakeHarness:
    def __init__(self, duration_s=14.0, ok=True, throughput=100.0):
        self.duration_s = duration_s
        self.ok = ok
        self.throughput = throughput
        self.commands = []

    def run(self, command, timeout_s):
        self.commands.append(command)
        return HarnessOutcome(
            ok=self.ok,
            duration_s=self.duration_s,
            returncode=0 if self.ok else 1,
            reason="" if self.ok else "harness said no",
            command=command,
            result={
                "completed": 64,
                "output_throughput": self.throughput,
                "median_ttft_ms": 42.0,
                "max_concurrency": 8,
                "dataset_name": "random",
                "total_input_tokens": 4096,
            },
        )


class FakeCard:
    def __init__(self, index=0, throttle=(), temp=60.0, clock=1900, clock_max=1900):
        self.index = index
        self.name = f"card{index}"
        self.sm_clock_mhz = clock
        self.sm_clock_max_mhz = clock_max
        self.temp_c = temp
        self.power_w = 200.0
        self.throttle = list(throttle)

    def performance_throttles(self):
        return list(self.throttle)

    def clock_ratio(self):
        return self.sm_clock_mhz / self.sm_clock_max_mhz


class FakeSampler:
    def __init__(self, cards=None):
        self.cards = cards or [FakeCard(0), FakeCard(1)]

    def sample(self):
        return list(self.cards)


class FakeNvml:
    def __init__(self, procs):
        # gpu index -> list of (pid, used bytes)
        self.procs = procs

    def nvmlDeviceGetCount(self):
        return len(self.procs)

    def nvmlDeviceGetHandleByIndex(self, i):
        return i

    def nvmlDeviceGetComputeRunningProcesses(self, handle):
        return [
            types.SimpleNamespace(pid=pid, usedGpuMemory=used)
            for pid, used in self.procs.get(handle, [])
        ]


def a_study(arms=("A", "B"), policy=None, harness=None, scraper=None, **kw):
    policy = policy or RunPolicy(noise_floor_boots=3, comparison_repeats=2, settle_s=0)
    return Study(
        NOISE,
        [Arm(label, settings(port=30000 + i)) for i, label in enumerate(arms)],
        policy=policy,
        supervisor=FakeSupervisor(),
        sampler=FakeSampler(),
        scraper_factory=lambda url: (scraper or FakeScraper()),
        harness=harness or FakeHarness(),
        sleep=lambda s: None,
        **kw,
    )


# ---------------------------------------------------------------------------
# The policy refuses the shortcuts
# ---------------------------------------------------------------------------


class TestPolicyRefusals(CustomTestCase):
    def test_within_boot_noise_floor_is_not_offered(self):
        with self.assertRaises(WithinBootRefused) as cm:
            RunPolicy(noise_floor_mode="within_boot")
        self.assertIn("boot-to-boot", str(cm.exception))
        self.assertIn("2.80", str(cm.exception))

    def test_one_repeat_is_not_a_floor(self):
        with self.assertRaises(ValueError):
            RunPolicy(noise_floor_boots=1)

    def test_kv_budget_must_be_reset_or_pinned(self):
        with self.assertRaises(KvBudgetUnpinned) as cm:
            RunPolicy(reset_kv_budget=False, pin_token_vector=None)
        self.assertIn("kv_budget", str(cm.exception))
        self.assertIn("4x", str(cm.exception))

    def test_pinning_is_the_other_accepted_answer(self):
        RunPolicy(reset_kv_budget=False, pin_token_vector="2,1,1")

    def test_time_budget_band_must_not_be_inverted(self):
        with self.assertRaises(ValueError):
            TimeBudget(target_low_s=30, target_high_s=10)

    def test_ceiling_below_the_band_is_refused(self):
        with self.assertRaises(ValueError):
            TimeBudget(target_low_s=10, target_high_s=20, ceiling_s=15)

    def test_budget_verdicts(self):
        b = TimeBudget()
        self.assertEqual(b.verdict(15.0), "")
        self.assertEqual(b.verdict(3.0), "short")
        self.assertEqual(b.verdict(30.0), "over_target")
        self.assertEqual(b.verdict(90.0), "ceiling")


# ---------------------------------------------------------------------------
# The schedule
# ---------------------------------------------------------------------------


class TestSchedule(CustomTestCase):
    def test_floor_boots_come_first(self):
        plan = build_schedule(["A", "B"], RunPolicy(noise_floor_boots=3))
        self.assertEqual([b.role for b in plan[:3]], ["noise_floor"] * 3)
        self.assertTrue(all(b.arm == "A" for b in plan[:3]))
        self.assertTrue(all(b.role == "comparison" for b in plan[3:]))

    def test_comparison_boots_are_interleaved_not_blocked(self):
        plan = build_schedule(
            ["A", "B"], RunPolicy(noise_floor_boots=2, comparison_repeats=3)
        )
        arms = [b.arm for b in plan if b.role == "comparison"]
        self.assertEqual(arms, ["A", "B", "A", "B", "A", "B"])

    def test_every_repeat_is_its_own_boot(self):
        plan = build_schedule(["A", "B"], RunPolicy(noise_floor_boots=4))
        orders = [b.order for b in plan]
        self.assertEqual(orders, sorted(set(orders)))
        self.assertEqual(len(orders), len(plan))

    def test_single_arm_gets_only_the_floor(self):
        plan = build_schedule(["A"], RunPolicy(noise_floor_boots=2))
        self.assertEqual([b.role for b in plan], ["noise_floor"] * 2)

    def test_no_arms_is_an_error(self):
        with self.assertRaises(ValueError):
            build_schedule([])


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------


class TestWindowPlan(CustomTestCase):
    def test_scenario_without_windows_gets_one_named_window(self):
        steps = window_plan(NOISE)
        self.assertEqual([s.window for s in steps], [DEFAULT_WINDOW])
        self.assertTrue(steps[0].drives_harness)

    def test_declared_windows_are_all_kept(self):
        steps = window_plan(SPILL)
        self.assertEqual(
            [s.window for s in steps], [w.key for w in SPILL.windows]
        )

    def test_the_transient_window_stays_excluded(self):
        steps = {s.window: s for s in window_plan(SPILL)}
        self.assertTrue(steps["restore_transient"].excluded_from_headline)

    def test_an_undrivable_window_is_emitted_with_the_reason(self):
        steps = {s.window: s for s in window_plan(SPILL)}
        self.assertTrue(steps["during_spill"].undrivable_reason)
        self.assertIn("not measured", steps["during_spill"].undrivable_reason)

    def test_a_supplied_driver_removes_the_reason(self):
        steps = {
            s.window: s
            for s in window_plan(SPILL, {"during_spill": lambda *a: None})
        }
        self.assertEqual(steps["during_spill"].undrivable_reason, "")

    def test_exactly_one_window_drives_the_harness(self):
        steps = window_plan(SPILL)
        self.assertEqual(sum(1 for s in steps if s.drives_harness), 1)


# ---------------------------------------------------------------------------
# The #188 trap
# ---------------------------------------------------------------------------


class TestKvBudgetNeutralisation(CustomTestCase):
    def test_reset_clears_every_budget_file(self):
        removed = []
        out = neutralise_kv_budget(
            RunPolicy(),
            cache_dir="/nowhere",
            lister=lambda d: ["/nowhere/kv_budget-a.json", "/nowhere/kv_budget-b.json"],
            resetter=lambda p: removed.append(p) or {"removed": True, "path": p},
        )
        self.assertEqual(out["strategy"], "reset")
        self.assertEqual(len(removed), 2)

    def test_pinning_leaves_the_files_alone_and_sets_the_env(self):
        touched = []
        out = neutralise_kv_budget(
            RunPolicy(reset_kv_budget=False, pin_token_vector="3,2,2"),
            cache_dir="/nowhere",
            lister=lambda d: ["/nowhere/kv_budget-a.json"],
            resetter=lambda p: touched.append(p),
        )
        self.assertEqual(out["strategy"], "pinned")
        self.assertEqual(touched, [])
        self.assertEqual(out["env"]["SGLANG_UNEVEN_TOKEN_VECTOR"], "3,2,2")

    def test_a_failing_reset_is_recorded_not_swallowed(self):
        def boom(path):
            raise OSError("read-only")

        out = neutralise_kv_budget(
            RunPolicy(),
            cache_dir="/nowhere",
            lister=lambda d: ["/nowhere/kv_budget-a.json"],
            resetter=boom,
        )
        self.assertFalse(out["removed"][0]["removed"])
        self.assertIn("read-only", out["removed"][0]["reason"])

    def test_it_runs_before_every_point_not_once(self):
        calls = []
        study = a_study()
        study.kv_cache_dir = "/nowhere"
        import sglang.srt.planner.runner as mod

        original = mod.neutralise_kv_budget
        try:
            mod.neutralise_kv_budget = lambda p, d=None: (
                calls.append(d) or {"strategy": "reset", "env": {}, "removed": []}
            )
            study.run()
        finally:
            mod.neutralise_kv_budget = original
        self.assertEqual(len(calls), len(study.plan()))


# ---------------------------------------------------------------------------
# The VRAM gate
# ---------------------------------------------------------------------------


class TestOwnVramGate(CustomTestCase):
    def test_a_foreign_process_never_blocks(self):
        nvml = FakeNvml({0: [(4242, 8 * 2**30)]})
        out = own_vram_gate([111], [0], nvml=nvml, timeout_s=5, sleep=lambda s: None)
        self.assertTrue(out["clear"])
        self.assertEqual(out["foreign"][0]["pid"], 4242)
        self.assertIn("never opens", out["reason"])

    def test_our_own_process_blocks_and_then_names_itself(self):
        nvml = FakeNvml({0: [(111, 4 * 2**30)]})
        ticks = iter([0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0])
        out = own_vram_gate(
            [111],
            [0],
            nvml=nvml,
            timeout_s=2.0,
            clock=lambda: next(ticks),
            sleep=lambda s: None,
        )
        self.assertFalse(out["clear"])
        self.assertIn("111", out["reason"])

    def test_a_released_card_opens_the_gate(self):
        nvml = FakeNvml({0: []})
        out = own_vram_gate([111], [0], nvml=nvml, sleep=lambda s: None)
        self.assertTrue(out["clear"])
        self.assertEqual(out["own_holding"], [])

    def test_without_nvml_the_gate_says_so_instead_of_blocking(self):
        out = own_vram_gate([111], [0], nvml=None)
        self.assertTrue(out["clear"])
        self.assertIn("NVML unavailable", out["reason"])


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class TestCardState(CustomTestCase):
    def test_throttle_reasons_travel_with_the_point(self):
        rows = card_state([FakeCard(0, throttle=("sw_thermal_slowdown",), temp=88)])
        self.assertTrue(rows[0]["throttled"])
        self.assertEqual(rows[0]["temp_c"], 88)
        self.assertEqual(rows[0]["throttle_reasons"], ["sw_thermal_slowdown"])

    def test_a_healthy_card_is_not_marked(self):
        self.assertFalse(card_state([FakeCard(0)])[0]["throttled"])

    def test_clock_ratio_is_carried(self):
        rows = card_state([FakeCard(0, clock=1695, clock_max=1905)])
        self.assertAlmostEqual(rows[0]["clock_ratio"], 1695 / 1905)


# ---------------------------------------------------------------------------
# Window metrics
# ---------------------------------------------------------------------------


class TestWindowMetrics(CustomTestCase):
    def test_round_time_comes_out_of_the_phase_counter(self):
        before = FakeEngineSample(
            {"generation_tokens_total": 0.0, "spec_accept_length": 2.0},
            {0: {"target_verify": 0.0}},
        )
        after = FakeEngineSample(
            {"generation_tokens_total": 1000.0, "spec_accept_length": 2.0},
            {0: {"target_verify": 1.0}},
        )
        metrics, samples, _ = window_metrics(before, after, 10.0)
        self.assertAlmostEqual(metrics["ms_per_verify_round"], 1.0 * 1000 / 500)
        self.assertEqual(metrics["accept_length"], 2.0)
        self.assertEqual(metrics["verify_ct"], 500.0)

    def test_a_missing_device_timer_names_the_env_var(self):
        before = FakeEngineSample({"generation_tokens_total": 0.0}, {})
        after = FakeEngineSample({"generation_tokens_total": 100.0}, {})
        metrics, _, notes = window_metrics(before, after, 10.0)
        self.assertNotIn("ms_per_verify_round", metrics)
        self.assertTrue(
            any("SGLANG_ENABLE_METRICS_DEVICE_TIMER" in n for n in notes)
        )

    def test_harness_fields_are_bound_by_the_scenario_mapping(self):
        before = FakeEngineSample({}, {})
        after = FakeEngineSample({}, {})
        metrics, samples, _ = window_metrics(
            before,
            after,
            10.0,
            harness_result={"median_ttft_ms": 41.5, "completed": 64},
            metric_fields=NOISE.harness.metric_fields,
        )
        self.assertEqual(metrics["ttft_p50"], 41.5)
        self.assertEqual(samples["ttft_p50"], 64)

    def test_the_collector_placeholders_are_not_read_as_harness_fields(self):
        before = FakeEngineSample({}, {})
        after = FakeEngineSample({}, {})
        metrics, _, _ = window_metrics(
            before,
            after,
            10.0,
            harness_result={"(collector: round_time over the same wall clock)": 9},
            metric_fields=NOISE.harness.metric_fields,
        )
        self.assertNotIn("ms_per_verify_round", metrics)


# ---------------------------------------------------------------------------
# Reducing points
# ---------------------------------------------------------------------------


def point(arm="A", repeat=0, role="noise_floor", aborted="", throttled=False, **metrics):
    return PointResult(
        arm=arm,
        repeat=repeat,
        role=role,
        aborted=aborted,
        throttled=throttled,
        started_at=1.0,
        windows=[WindowResult(window=DEFAULT_WINDOW, metrics=dict(metrics))],
    )


class TestNoiseFloor(CustomTestCase):
    def test_the_floor_is_the_relative_range(self):
        floor = noise_floor_from_points(
            [
                point(ms_per_verify_round=2.00),
                point(ms_per_verify_round=2.02),
                point(ms_per_verify_round=1.98),
            ]
        )
        self.assertAlmostEqual(floor.for_metric("ms_per_verify_round"), 0.04 / 2.00)

    def test_a_metric_seen_once_gets_no_floor(self):
        floor = noise_floor_from_points(
            [point(ms_per_verify_round=2.0, tok_s=100.0), point(ms_per_verify_round=2.1)]
        )
        self.assertIsNone(floor.for_metric("tok_s"))
        self.assertIsNotNone(floor.for_metric("ms_per_verify_round"))

    def test_a_metric_without_a_floor_makes_the_comparison_unknown(self):
        from sglang.srt.planner.comparison import ArmResult, compare_metric

        floor = noise_floor_from_points([point(tok_s=100.0)])

        def arm(label, value):
            return ArmResult(
                label=label,
                windows=[WindowResult(window="steady", metrics={"tok_s": value})],
            )

        c = compare_metric(arm("A", 100.0), arm("B", 140.0), "tok_s", "steady", floor)
        self.assertEqual(c.verdict, UNKNOWN)

    def test_aborted_points_do_not_widen_the_floor(self):
        floor = noise_floor_from_points(
            [
                point(ms_per_verify_round=2.00),
                point(ms_per_verify_round=2.02),
                point(aborted="boot failed", ms_per_verify_round=9.0),
            ]
        )
        self.assertLess(floor.for_metric("ms_per_verify_round"), 0.02)

    def test_throttled_repeats_are_counted_in_the_source(self):
        floor = noise_floor_from_points(
            [
                point(ms_per_verify_round=2.0),
                point(ms_per_verify_round=2.4, throttled=True),
            ]
        )
        self.assertIn("throttled", floor.source)
        self.assertIn("boot-to-boot", floor.source)


class TestArmResults(CustomTestCase):
    def test_repeats_fold_to_the_median(self):
        arm = arm_result_from_points(
            "A",
            [
                point(ms_per_verify_round=2.0),
                point(ms_per_verify_round=2.2),
                point(ms_per_verify_round=9.0),
            ],
        )
        self.assertEqual(arm.window(DEFAULT_WINDOW).metrics["ms_per_verify_round"], 2.2)

    def test_aborted_points_are_excluded_and_counted(self):
        arm = arm_result_from_points(
            "A",
            [point(ms_per_verify_round=2.0), point(aborted="boot failed")],
        )
        self.assertEqual(arm.provenance["boots_used"], 1)
        self.assertEqual(arm.provenance["boots_aborted"][0]["reason"], "boot failed")

    def test_throttled_points_are_kept_and_marked(self):
        arm = arm_result_from_points(
            "A",
            [
                point(ms_per_verify_round=2.0),
                point(ms_per_verify_round=2.6, throttled=True),
            ],
        )
        self.assertEqual(arm.provenance["boots_used"], 2)
        self.assertEqual(arm.provenance["boots_throttled"], 1)
        self.assertIn("kept and", arm.provenance["state_note"])

    def test_accept_length_lands_in_the_conditions(self):
        arm = arm_result_from_points(
            "A", [point(accept_length=2.1), point(accept_length=2.3)]
        )
        self.assertAlmostEqual(arm.conditions["accept_length"], 2.2)

    def test_a_result_carries_its_age(self):
        arm = arm_result_from_points("A", [point(ms_per_verify_round=2.0)])
        self.assertIsNotNone(arm.provenance["newest_point_age_s"])


# ---------------------------------------------------------------------------
# The study end to end
# ---------------------------------------------------------------------------


class TestStudy(CustomTestCase):
    def test_arms_must_have_unique_labels(self):
        with self.assertRaises(ValueError):
            Study(NOISE, [Arm("A", settings()), Arm("A", settings())])

    def test_the_floor_boots_run_before_any_comparison_boot(self):
        study = a_study()
        out = study.run()
        roles = [p.role for p in out.points]
        self.assertEqual(roles[:3], ["noise_floor"] * 3)
        self.assertNotIn("noise_floor", roles[3:])

    def test_the_boot_order_is_floor_then_interleaved(self):
        study = a_study()
        study.run()
        # Port identifies the arm: A is 30000, B is 30001.
        self.assertEqual(
            study.supervisor.booted,
            [30000, 30000, 30000, 30000, 30001, 30000, 30001],
        )

    def test_every_boot_is_torn_down(self):
        study = a_study()
        out = study.run()
        self.assertEqual(study.supervisor.stopped, len(out.points))

    def test_the_floor_is_derived_only_from_the_a_vs_a_boots(self):
        study = a_study()
        out = study.run()
        self.assertIn("3 boot-to-boot repeats", out.noise.source)

    def test_comparisons_use_the_closed_vocabulary(self):
        out = a_study().run()
        self.assertTrue(out.comparisons)
        for c in out.comparisons:
            self.assertIn(c.verdict, VERDICTS)

    def test_a_supplied_floor_that_covers_the_primary_metric_skips_the_boots(self):
        study = a_study()
        out = study.run(
            noise=NoiseFloor(relative={"ms_per_verify_round": 0.004}, source="earlier")
        )
        self.assertEqual([p.role for p in out.points], ["comparison"] * 4)
        self.assertEqual(out.noise.source, "earlier")

    def test_a_supplied_floor_missing_the_primary_metric_runs_them_anyway(self):
        study = a_study()
        out = study.run(noise=NoiseFloor(relative={"tok_s": 0.04}, source="partial"))
        self.assertIn("noise_floor", [p.role for p in out.points])
        self.assertTrue(any("does not cover it" in n for n in out.notes))

    def test_a_point_carries_its_card_state(self):
        study = a_study()
        out = study.run()
        self.assertTrue(out.points[0].state_before)
        self.assertTrue(out.points[0].state_after)

    def test_a_throttled_point_is_kept_and_marked(self):
        study = a_study()
        study.sampler = FakeSampler([FakeCard(0, throttle=("sw_thermal_slowdown",))])
        out = study.run()
        self.assertTrue(out.points[0].throttled)
        self.assertTrue(out.points[0].usable)
        self.assertTrue(any("kept and marked" in n for n in out.points[0].notes))

    def test_a_point_over_the_ceiling_is_aborted_with_the_reason(self):
        study = a_study(harness=FakeHarness(duration_s=95.0))
        out = study.run()
        verdicts = [
            v for p in out.points for v in p.budget_verdicts.values()
        ]
        self.assertIn("ceiling", verdicts)

    def test_the_measured_duration_is_recorded(self):
        study = a_study(harness=FakeHarness(duration_s=13.5))
        out = study.run()
        self.assertEqual(out.points[0].durations_s[DEFAULT_WINDOW], 13.5)

    def test_a_short_point_is_advised_not_silently_resized(self):
        study = a_study(harness=FakeHarness(duration_s=2.0))
        out = study.run()
        self.assertTrue(any("not this one" in n for n in out.notes))
        # every point ran at the load the study started with
        self.assertTrue(
            all("--num-prompts 64" in c for c in study.harness.commands)
        )

    def test_a_failing_harness_aborts_the_point_with_its_reason(self):
        study = a_study(harness=FakeHarness(ok=False))
        out = study.run()
        self.assertTrue(all(p.aborted for p in out.points))
        self.assertIn("harness said no", out.aborted)

    def test_a_blocked_own_pid_aborts_the_point(self):
        study = a_study()
        study.nvml = FakeNvml({0: [(999, 2**30)], 1: []})
        study._own_pids = [999]
        study.policy.own_vram_timeout_s = 0.0
        out = study.run()
        self.assertTrue(all("still hold VRAM" in p.aborted for p in out.points))

    def test_an_external_axis_refuses_to_run_silently(self):
        study = Study(
            POWER,
            [Arm("A", settings())],
            policy=RunPolicy(settle_s=0),
            supervisor=FakeSupervisor(),
            sampler=FakeSampler(),
            scraper_factory=lambda url: FakeScraper(),
            harness=FakeHarness(),
            point={"power_limit_w": "60%"},
            sleep=lambda s: None,
        )
        out = study.run()
        self.assertTrue(all("host or server controls" in p.aborted for p in out.points))

    def test_preflight_names_the_device_timer_switch(self):
        study = a_study()
        self.assertTrue(
            any("SGLANG_ENABLE_METRICS_DEVICE_TIMER" in p for p in study.preflight())
        )

    def test_preflight_is_quiet_about_it_once_it_is_set(self):
        study = a_study(
            policy=RunPolicy(
                settle_s=0, env={"SGLANG_ENABLE_METRICS_DEVICE_TIMER": "1"}
            )
        )
        self.assertFalse(
            any("will be ABSENT" in p for p in study.preflight())
        )

    def test_undrivable_windows_are_reported_empty_not_dropped(self):
        study = Study(
            SPILL,
            [Arm("A", settings())],
            policy=RunPolicy(settle_s=0, noise_floor_boots=2),
            supervisor=FakeSupervisor(),
            sampler=FakeSampler(),
            scraper_factory=lambda url: FakeScraper(),
            harness=FakeHarness(),
            sleep=lambda s: None,
        )
        out = study.run()
        keys = [w.window for w in out.points[0].windows]
        self.assertEqual(keys, [w.key for w in SPILL.windows])
        transient = out.points[0].window("restore_transient")
        self.assertTrue(transient.excluded_from_headline)
        self.assertEqual(transient.metrics, {})
        self.assertTrue(transient.note)

    def test_the_transient_window_never_becomes_a_headline(self):
        from sglang.srt.planner.comparison import HeadlineRefused, headline

        study = Study(
            SPILL,
            [Arm("A", settings())],
            policy=RunPolicy(settle_s=0, noise_floor_boots=2),
            supervisor=FakeSupervisor(),
            sampler=FakeSampler(),
            scraper_factory=lambda url: FakeScraper(),
            harness=FakeHarness(),
            sleep=lambda s: None,
        )
        out = study.run()
        with self.assertRaises(HeadlineRefused):
            headline(out.arms[0], "restore_transient", "tok_s")

    def test_rendering_a_study_mentions_the_floor_and_the_verdicts(self):
        text = render_study_text(a_study().run())
        self.assertIn("Noise floor", text)
        self.assertIn("Comparisons", text)


class TestSuggestNumPrompts(CustomTestCase):
    def test_an_in_band_point_needs_no_advice(self):
        out = suggest_num_prompts(15.0, 64)
        self.assertEqual(out["advice"], "")

    def test_a_long_point_suggests_a_smaller_load(self):
        out = suggest_num_prompts(60.0, 64)
        self.assertLess(out["suggested"], 64)

    def test_the_advice_says_it_is_for_the_next_study(self):
        self.assertIn("NEXT study", suggest_num_prompts(2.0, 64)["advice"])


if __name__ == "__main__":
    unittest.main()
