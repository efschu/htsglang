"""Unit tests for startup_func_log_and_timer.py — no server, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")
register_cpu_ci(est_time=7, suite="base-c-test-cpu")

import ast
import os
import unittest
from unittest.mock import MagicMock, patch

import sglang.srt.observability.startup_func_log_and_timer as mod
from sglang.srt.observability.startup_func_log_and_timer import (
    enable_startup_timer,
    get_max_duration,
    reset_startup_timers,
    set_startup_metric,
    startup_timer,
    time_startup_latency,
)


class TestStartupFuncLogAndTimer(unittest.TestCase):
    def setUp(self):
        self.orig_enable = mod.enable_startup_metrics
        self.orig_gauge = mod.STARTUP_LATENCY_SECONDS
        mod._max_durations.clear()

    def tearDown(self):
        mod.enable_startup_metrics = self.orig_enable
        mod.STARTUP_LATENCY_SECONDS = self.orig_gauge
        mod._max_durations.clear()

    @patch("prometheus_client.Gauge")
    def test_enable_startup_timer(self, MockGauge):
        mod.STARTUP_LATENCY_SECONDS = None
        enable_startup_timer()
        self.assertTrue(mod.enable_startup_metrics)
        self.assertIs(mod.STARTUP_LATENCY_SECONDS, MockGauge.return_value)
        MockGauge.assert_called_once()

    @patch("prometheus_client.Gauge")
    def test_enable_startup_timer_is_idempotent(self, MockGauge):
        """Two enable calls in one process must not re-register the Gauge.

        There is one call per process that emits startup phases -- the parent
        (launch_server) and every scheduler subprocess (run_scheduler_process)
        -- but a process that constructs an Engine in-process can reach both.
        prometheus_client raises "Duplicated timeseries in CollectorRegistry"
        on a second registration of the same metric name, which would turn an
        observability nicety into a boot crash.
        """
        mod.STARTUP_LATENCY_SECONDS = None
        enable_startup_timer()
        first_gauge = mod.STARTUP_LATENCY_SECONDS

        mod.enable_startup_metrics = False  # simulate a fresh-looking flag
        enable_startup_timer()

        self.assertIs(mod.STARTUP_LATENCY_SECONDS, first_gauge)
        self.assertTrue(mod.enable_startup_metrics)
        MockGauge.assert_called_once()

    def test_reset_and_get_max_duration(self):
        mod._max_durations["ctx"] = 5.0
        self.assertAlmostEqual(get_max_duration("ctx"), 5.0)
        self.assertIsNone(get_max_duration("nonexistent"))
        reset_startup_timers()
        self.assertIsNone(get_max_duration("ctx"))

    def test_set_startup_metric_disabled(self):
        """When metrics disabled, returns early without tracking max."""
        mod.enable_startup_metrics = False
        set_startup_metric("ctx", 1.0)
        self.assertIsNone(get_max_duration("ctx"))

    def test_set_startup_metric_enabled(self):
        """Tracks max and updates gauge when enabled."""
        mock_gauge = MagicMock()
        mod.enable_startup_metrics = True
        mod.STARTUP_LATENCY_SECONDS = mock_gauge

        set_startup_metric("ctx", 1.0)
        self.assertAlmostEqual(get_max_duration("ctx"), 1.0)
        mock_gauge.labels.assert_called_with(context="ctx")

        # Lower value → not updated
        mock_gauge.reset_mock()
        set_startup_metric("ctx", 0.5)
        self.assertAlmostEqual(get_max_duration("ctx"), 1.0)
        mock_gauge.labels().set.assert_not_called()

    def test_set_startup_metric_no_log(self):
        mod.enable_startup_metrics = False
        with patch.object(mod.logger, "info") as mock_log:
            set_startup_metric("ctx", 1.0, should_log=False)
            mock_log.assert_not_called()

    def test_startup_timer_basic(self):
        with startup_timer("block"):
            pass
        self.assertGreaterEqual(get_max_duration("block"), 0.0)

    def test_startup_timer_with_gauge(self):
        """Gauge updated when metrics enabled and log_only=False."""
        mock_gauge = MagicMock()
        mod.enable_startup_metrics = True
        mod.STARTUP_LATENCY_SECONDS = mock_gauge

        with startup_timer("block"):
            pass
        mock_gauge.labels.assert_called_with(context="block")
        mock_gauge.labels().set.assert_called_once()

    def test_startup_timer_log_only(self):
        """log_only=True skips gauge but still tracks max."""
        mock_gauge = MagicMock()
        mod.enable_startup_metrics = True
        mod.STARTUP_LATENCY_SECONDS = mock_gauge

        with startup_timer("block", log_only=True):
            pass
        mock_gauge.labels.assert_not_called()
        self.assertIsNotNone(get_max_duration("block"))

    def test_decorator_direct(self):
        """Direct decorator @time_startup_latency preserves return value."""

        @time_startup_latency
        def add(a, b):
            return a + b

        self.assertEqual(add(2, 3), 5)
        self.assertIsNotNone(get_max_duration("add"))

    def test_decorator_factory_with_gauge(self):
        """Factory decorator with custom name, gauge updated."""
        mock_gauge = MagicMock()
        mod.enable_startup_metrics = True
        mod.STARTUP_LATENCY_SECONDS = mock_gauge

        @time_startup_latency(name="custom_op")
        def add(a, b):
            return a + b

        self.assertEqual(add(2, 3), 5)
        mock_gauge.labels.assert_called_with(context="custom_op")

    def test_decorator_log_only(self):
        """log_only=True skips gauge but still tracks max."""
        mock_gauge = MagicMock()
        mod.enable_startup_metrics = True
        mod.STARTUP_LATENCY_SECONDS = mock_gauge

        @time_startup_latency(log_only=True)
        def add(a, b):
            return a + b

        self.assertEqual(add(2, 3), 5)
        mock_gauge.labels.assert_not_called()
        self.assertIsNotNone(get_max_duration("add"))

    def test_startup_timer_noop_without_enable(self):
        """Using startup_timer without calling enable_startup_timer() must
        remain silent (no gauge access, no exceptions). This is the safety
        guarantee that allows call sites to use startup_timer unconditionally
        behind an enable_metrics gate at the caller level."""
        # Ensure metrics are disabled and gauge is None (fresh state).
        mod.enable_startup_metrics = False
        mod.STARTUP_LATENCY_SECONDS = None
        mod._max_durations.clear()

        # Using the timer without enable should not raise and should not
        # touch STARTUP_LATENCY_SECONDS (which is None).
        with startup_timer("noop_test"):
            pass

        # Duration is tracked in-memory (harmless), but gauge must NOT be hit.
        self.assertIsNotNone(get_max_duration("noop_test"))
        # Verify gauge was not accessed (it is None and accessing .labels()
        # would raise AttributeError).
        mod.STARTUP_LATENCY_SECONDS = self.orig_gauge  # restore for tearDown

    def test_time_startup_latency_noop_without_enable(self):
        """The decorator must also be a safe no-op without enable."""
        mod.enable_startup_metrics = False
        mod.STARTUP_LATENCY_SECONDS = None
        mod._max_durations.clear()

        @time_startup_latency(name="noop_decorator_test")
        def sample_work():
            return 42

        result = sample_work()
        self.assertEqual(result, 42)
        self.assertIsNotNone(get_max_duration("noop_decorator_test"))


class TestBootPhaseEmission(unittest.TestCase):
    """Fake a whole boot out of the phase names the real call sites use and
    assert the emitted structure: one INFO line per phase, one gauge label per
    phase, and no extra lines. Boot-time attribution is only usable if every
    phase reports exactly once per boot -- a phase that logs twice makes the
    breakdown unreadable, and one that logs N times per boot is the log-spam
    regression this suite exists to catch."""

    # The phases wired into the real boot path. Kept here as data so the
    # contract test below can assert the code actually emits these names.
    BOOT_PHASES = (
        "weight_loading",
        "memory_pool_init",
        "attention_backend_init",
        "cuda_graph_capture",
    )

    def setUp(self):
        self.orig_enable = mod.enable_startup_metrics
        self.orig_gauge = mod.STARTUP_LATENCY_SECONDS
        mod._max_durations.clear()

    def tearDown(self):
        mod.enable_startup_metrics = self.orig_enable
        mod.STARTUP_LATENCY_SECONDS = self.orig_gauge
        mod._max_durations.clear()

    def test_fake_boot_emits_one_line_and_one_gauge_point_per_phase(self):
        mock_gauge = MagicMock()
        mod.enable_startup_metrics = True
        mod.STARTUP_LATENCY_SECONDS = mock_gauge

        with self.assertLogs(mod.logger, level="INFO") as captured:
            for phase in self.BOOT_PHASES:

                @time_startup_latency(name=phase)
                def _phase_body():
                    return phase

                _phase_body()

        # Exactly one INFO line per phase -- no spam.
        self.assertEqual(len(captured.records), len(self.BOOT_PHASES))

        # Each line is structured and names its phase and a duration.
        for phase, record in zip(self.BOOT_PHASES, captured.records):
            self.assertEqual(record.levelname, "INFO")
            self.assertIn(phase, record.getMessage())
            self.assertRegex(record.getMessage(), rf"^Startup timing: {phase} took ")
            self.assertRegex(record.getMessage(), r"took \d+\.\d{3}s$")

        # One gauge point per phase, under the phase's own context label.
        gauge_contexts = [
            call.kwargs["context"]
            for call in mock_gauge.labels.call_args_list
            if "context" in call.kwargs
        ]
        self.assertEqual(gauge_contexts, list(self.BOOT_PHASES))

        # Every phase is retrievable for boot-time attribution (#539).
        for phase in self.BOOT_PHASES:
            self.assertIsNotNone(get_max_duration(phase))

    def test_repeated_phase_does_not_multiply_gauge_writes(self):
        """A phase that runs again (draft runner, lane bring-up) logs again --
        that is informative -- but only a NEW maximum reaches the gauge, so a
        re-run cannot make the breakdown drift downward."""
        mock_gauge = MagicMock()
        mod.enable_startup_metrics = True
        mod.STARTUP_LATENCY_SECONDS = mock_gauge

        mod._max_durations["cuda_graph_capture"] = 999.0

        with self.assertLogs(mod.logger, level="INFO") as captured:
            with startup_timer("cuda_graph_capture"):
                pass

        self.assertEqual(len(captured.records), 1)
        mock_gauge.labels.assert_not_called()
        self.assertAlmostEqual(get_max_duration("cuda_graph_capture"), 999.0)


class TestStartupTimerWiringContract(unittest.TestCase):
    """Bind the phase names to the real call sites.

    The unit tests above pass just as well against a module with zero
    consumers -- which is exactly the state #560 was opened for. These read
    the shipping source with ast (no torch import, no CUDA, no model) and
    fail if a call site is removed or renamed.
    """

    @staticmethod
    def _srt_path(*parts):
        # sglang.srt is a namespace package (__file__ is None), so anchor on
        # the module under test: .../srt/observability/startup_func_log_and_timer.py
        srt_root = os.path.dirname(os.path.dirname(os.path.abspath(mod.__file__)))
        return os.path.join(srt_root, *parts)

    @staticmethod
    def _parse(path):
        with open(path, "r", encoding="utf-8") as f:
            return ast.parse(f.read())

    def _find_function(self, tree, name):
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return node
        self.fail(f"function {name} not found")

    def test_model_runner_boot_phases_are_decorated(self):
        """weight load / pool init / backend init / graph capture carry timers."""
        tree = self._parse(self._srt_path("model_executor", "model_runner.py"))

        expected = {
            "alloc_memory_pool": "memory_pool_init",
            "init_attention_backends": "attention_backend_init",
            "init_cuda_graphs": "cuda_graph_capture",
        }

        for func_name, phase_name in expected.items():
            node = self._find_function(tree, func_name)
            names = []
            for deco in node.decorator_list:
                if (
                    isinstance(deco, ast.Call)
                    and getattr(deco.func, "id", None) == "time_startup_latency"
                ):
                    for kw in deco.keywords:
                        if kw.arg == "name" and isinstance(kw.value, ast.Constant):
                            names.append(kw.value.value)
            self.assertIn(
                phase_name,
                names,
                f"{func_name} lost its @time_startup_latency(name={phase_name!r})",
            )

    def test_weight_loading_phase_still_wrapped(self):
        """load_model keeps its startup_timer("weight_loading") block."""
        src = open(
            self._srt_path("model_executor", "model_runner.py"), encoding="utf-8"
        ).read()
        self.assertIn('startup_timer("weight_loading")', src)

    def test_scheduler_subprocess_enables_the_timer(self):
        """Without this the scheduler-side phases log but never reach the gauge
        -- the half-wired state #560 inherited."""
        tree = self._parse(self._srt_path("managers", "scheduler.py"))
        node = self._find_function(tree, "run_scheduler_process")
        called = {
            n.func.id
            for n in ast.walk(node)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        }
        self.assertIn("enable_startup_timer", called)

    def test_server_warmup_phase_is_timed(self):
        tree = self._parse(self._srt_path("entrypoints", "http_server.py"))
        node = self._find_function(tree, "_wait_and_warmup")
        timed = [
            n.args[0].value
            for n in ast.walk(node)
            if isinstance(n, ast.Call)
            and getattr(n.func, "id", None) == "startup_timer"
            and n.args
            and isinstance(n.args[0], ast.Constant)
        ]
        self.assertIn("server_warmup", timed)


if __name__ == "__main__":
    unittest.main()
