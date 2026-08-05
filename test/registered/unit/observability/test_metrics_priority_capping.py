"""Unit tests for priority label capping in SchedulerMetricsCollector.

Verifies that transform_priority is applied at construction time
(QueueCount.from_reqs) so that unbounded client-supplied priority
values never reach Prometheus labels.

This is a can-fail proof: with the old str(priority) path,
priority=999999 would emit "999999" and priority=None would
emit "None" — both of which break cardinality guarantees.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")
register_cpu_ci(est_time=7, suite="base-c-test-cpu")

import unittest
from unittest.mock import MagicMock

from sglang.srt.observability.metrics_collector import QueueCount


class FakeReq:
    """Minimal request mock carrying only a priority attribute."""

    def __init__(self, priority):
        self.priority = priority


class TestQueueCountPriorityCapping(unittest.TestCase):
    """Verify QueueCount.from_reqs applies transform_priority to keys."""

    def test_extreme_priority_capped_to_high(self):
        """priority=999999 must be capped to 'HIGH', not leaked as '999999'."""
        reqs = [FakeReq(999999)]
        qc = QueueCount.from_reqs(reqs, enable_priority_scheduling=True)
        self.assertIsNotNone(qc.by_priority)
        # Key must be the capped string, not the raw integer.
        self.assertIn("HIGH", qc.by_priority)
        self.assertNotIn(999999, qc.by_priority)
        self.assertNotIn("999999", qc.by_priority)

    def test_none_priority_capped_to_unknown(self):
        """priority=None must be capped to 'UNKNOWN', not leaked as 'None'."""
        reqs = [FakeReq(None)]
        qc = QueueCount.from_reqs(reqs, enable_priority_scheduling=True)
        self.assertIsNotNone(qc.by_priority)
        self.assertIn("UNKNOWN", qc.by_priority)
        self.assertNotIn(None, qc.by_priority)

    def test_normal_priority_passthrough(self):
        """priority in [0, 31) passes through as its string form."""
        reqs = [FakeReq(3)]
        qc = QueueCount.from_reqs(reqs, enable_priority_scheduling=True)
        self.assertIsNotNone(qc.by_priority)
        self.assertIn("3", qc.by_priority)

    def test_negative_priority_capped_to_low(self):
        """priority < 0 must be capped to 'LOW'."""
        reqs = [FakeReq(-5)]
        qc = QueueCount.from_reqs(reqs, enable_priority_scheduling=True)
        self.assertIsNotNone(qc.by_priority)
        self.assertIn("LOW", qc.by_priority)

    def test_mixed_priorities_capped(self):
        """Multiple requests with different priorities all get capped correctly."""
        reqs = [FakeReq(999999), FakeReq(None), FakeReq(2), FakeReq(-1)]
        qc = QueueCount.from_reqs(reqs, enable_priority_scheduling=True)
        self.assertEqual(qc.total, 4)
        self.assertEqual(qc.by_priority["HIGH"], 1)
        self.assertEqual(qc.by_priority["UNKNOWN"], 1)
        self.assertEqual(qc.by_priority["2"], 1)
        self.assertEqual(qc.by_priority["LOW"], 1)

    def test_by_priority_keys_are_strings_not_ints(self):
        """After transform_priority, keys must be strings, not raw ints."""
        reqs = [FakeReq(5), FakeReq(10)]
        qc = QueueCount.from_reqs(reqs, enable_priority_scheduling=True)
        for key in qc.by_priority:
            self.assertIsInstance(key, str)

    def test_disable_priority_scheduling(self):
        """When priority scheduling is disabled, by_priority is None."""
        reqs = [FakeReq(999999)]
        qc = QueueCount.from_reqs(reqs, enable_priority_scheduling=False)
        self.assertIsNone(qc.by_priority)


class TestPriorityLabelEmission(unittest.TestCase):
    """Verify _log_gauge_queue_count emits the capped string labels.

    Uses a minimal mock collector to avoid constructing the full
    SchedulerMetricsCollector (which requires prometheus_client +
    server_args with prefill_delayer config).
    """

    def test_emitted_label_is_capped_string(self):
        """The gauge labels() call must receive the capped string, not the raw value."""
        # Import the method implementation directly via the module to avoid
        # constructing the full collector.
        from sglang.srt.observability.metrics_collector import (
            SchedulerMetricsCollector,
        )

        # Build a minimal mock object that carries the labels and
        # _known_priorities attributes needed by _log_gauge_queue_count.
        mock_collector = MagicMock(spec=SchedulerMetricsCollector)
        mock_collector.labels = {"model_name": "test", "priority": ""}
        mock_collector._known_priorities = set()

        # Call the real method with a capped QueueCount.
        reqs = [FakeReq(999999)]
        qc = QueueCount.from_reqs(reqs, enable_priority_scheduling=True)

        gauge_mock = MagicMock()
        # Bind the real method and call it on our mock.
        SchedulerMetricsCollector._log_gauge_queue_count(
            mock_collector, gauge_mock, qc
        )

        # Collect all labels() calls that include a priority key.
        calls_with_priority = [
            c for c in gauge_mock.labels.call_args_list
            if c.kwargs.get("priority", "").startswith(("HIGH", "LOW", "UNKNOWN", "0", "1", "2", "3"))
        ]
        self.assertTrue(
            len(calls_with_priority) > 0,
            "Expected at least one label call with a capped priority string",
        )
        for call in calls_with_priority:
            p = call.kwargs["priority"]
            self.assertNotEqual(p, "999999", "Raw priority leaked into label")
            self.assertNotEqual(p, "None", "None priority leaked into label")
            # Must be one of the known capped values
            self.assertIn(p, ("HIGH", "LOW", "UNKNOWN", "0", "1", "2", "3", "4",
                              "5", "6", "7", "8", "9", "10", "11", "12", "13", "14",
                              "15", "16", "17", "18", "19", "20", "21", "22", "23",
                              "24", "25", "26", "27", "28", "29", "30"))

    def test_emitted_label_for_none_priority(self):
        """Verify None priority produces 'UNKNOWN' label, not 'None'."""
        from sglang.srt.observability.metrics_collector import (
            SchedulerMetricsCollector,
        )

        mock_collector = MagicMock(spec=SchedulerMetricsCollector)
        mock_collector.labels = {"model_name": "test", "priority": ""}
        mock_collector._known_priorities = set()

        reqs = [FakeReq(None)]
        qc = QueueCount.from_reqs(reqs, enable_priority_scheduling=True)

        gauge_mock = MagicMock()
        SchedulerMetricsCollector._log_gauge_queue_count(
            mock_collector, gauge_mock, qc
        )

        # Find calls with priority set to something other than the empty string.
        for call in gauge_mock.labels.call_args_list:
            p = call.kwargs.get("priority", "")
            if p:
                self.assertEqual(p, "UNKNOWN")


if __name__ == "__main__":
    unittest.main()
