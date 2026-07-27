"""Unit tests for the per-rank prefill log (RankPrefillLog).

The "Prefill batch" summary is emitted only on the stats-logging rank; the
per-rank line ("Prefill rank batch, ...") must run on EVERY rank, ahead of the
logging-rank gate, and must carry #cached-token as rank-local evidence that a
prefill was computed rather than served from the prefix cache. GPU durations
attach deferred (FIFO pairing with CUDA-event completions); no host sync is
involved anywhere. Pure CPU: the timer is faked, no CUDA events are created.
"""

import logging
import unittest
from types import SimpleNamespace

from sglang.srt.managers.scheduler_components.metrics_reporter import (
    RankPrefillLog,
    SchedulerMetricsReporter,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

LOGGER_NAME = "sglang.srt.managers.scheduler_components.metrics_reporter"


class FakeTimer:
    """Stands in for DeviceTimer: durations 'complete' when the test says so."""

    def __init__(self, log: RankPrefillLog):
        self._log = log
        self.completed = []

    def _report(self):
        while self.completed:
            self._log._on_duration(self.completed.pop(0))


class TestForwardModePlainPrefill(unittest.TestCase):
    def test_truth_table(self):
        self.assertTrue(ForwardMode.EXTEND.is_plain_prefill())
        self.assertTrue(ForwardMode.MIXED.is_plain_prefill())
        for mode in (
            ForwardMode.DECODE,
            ForwardMode.IDLE,
            ForwardMode.TARGET_VERIFY,
            ForwardMode.DRAFT_EXTEND_V2,
            ForwardMode.PREBUILT,
            ForwardMode.SPLIT_PREFILL,
            ForwardMode.DLLM_EXTEND,
        ):
            self.assertFalse(mode.is_plain_prefill(), mode)


class TestRankPrefillLog(unittest.TestCase):
    def _make(self, with_timer: bool):
        log = RankPrefillLog()
        timer = None
        if with_timer:
            timer = FakeTimer(log)
            log.timer = timer
        return log, timer

    def test_untimed_record_emits_immediately(self):
        log, _ = self._make(with_timer=False)
        with self.assertLogs(LOGGER_NAME, level=logging.INFO) as cm:
            log.record(new_tokens=100, cached_tokens=7, timed=True)
        self.assertEqual(len(cm.output), 1)
        line = cm.output[0]
        self.assertIn("Prefill rank batch", line)
        self.assertIn("#new-token: 100", line)
        self.assertIn("#cached-token: 7", line)
        self.assertIn("#chunks: 1", line)
        self.assertNotIn("gpu-ms", line)
        self.assertFalse(log.has_pending)

    def test_untimed_mode_with_timer_emits_immediately(self):
        # timed=False (e.g. SPLIT_PREFILL) must not enter the FIFO pairing.
        log, timer = self._make(with_timer=True)
        with self.assertLogs(LOGGER_NAME, level=logging.INFO) as cm:
            log.record(new_tokens=5, cached_tokens=0, timed=False)
        self.assertEqual(len(cm.output), 1)
        self.assertNotIn("gpu-ms", cm.output[0])
        self.assertFalse(log.has_pending)

    def test_timed_record_waits_for_duration(self):
        log, timer = self._make(with_timer=True)
        log.record(new_tokens=64, cached_tokens=3, timed=True)
        self.assertTrue(log.has_pending)
        # events not complete yet: flush emits nothing
        with self.assertNoLogs(LOGGER_NAME, level=logging.INFO):
            log.flush()
        self.assertTrue(log.has_pending)
        # events complete: line carries the duration
        timer.completed.append(0.25)
        with self.assertLogs(LOGGER_NAME, level=logging.INFO) as cm:
            log.flush()
        self.assertEqual(len(cm.output), 1)
        line = cm.output[0]
        self.assertIn("#new-token: 64", line)
        self.assertIn("#cached-token: 3", line)
        self.assertIn("#chunks: 1", line)
        self.assertIn("gpu-ms: 250.0", line)
        self.assertFalse(log.has_pending)

    def test_flush_aggregates_completed_chunks(self):
        log, timer = self._make(with_timer=True)
        log.record(new_tokens=10, cached_tokens=1, timed=True)
        log.record(new_tokens=20, cached_tokens=2, timed=True)
        timer.completed.extend([0.1, 0.2])
        with self.assertLogs(LOGGER_NAME, level=logging.INFO) as cm:
            log.flush()
        self.assertEqual(len(cm.output), 1)
        line = cm.output[0]
        self.assertIn("#new-token: 30", line)
        self.assertIn("#cached-token: 3", line)
        self.assertIn("#chunks: 2", line)
        self.assertIn("gpu-ms: 300.0", line)

    def test_duration_before_record_pairs_up(self):
        # Overlap schedule: the forward's events can complete before the
        # result (and thus the record) arrives. The orphan duration must wait
        # for its record, not be dropped.
        log, timer = self._make(with_timer=True)
        timer.completed.append(0.05)
        log.flush()  # harvests the duration; no record yet -> no line
        with self.assertLogs(LOGGER_NAME, level=logging.INFO) as cm:
            log.record(new_tokens=8, cached_tokens=0, timed=True)
            log.flush()
        self.assertEqual(len(cm.output), 1)
        self.assertIn("gpu-ms: 50.0", cm.output[0])


class TestReporterEmitsAheadOfLoggingGate(unittest.TestCase):
    def _stub_reporter(self):
        return SimpleNamespace(
            rank_prefill_log=RankPrefillLog(),
            is_stats_logging_rank=False,
            current_scheduler_metrics_enabled=False,
        )

    def test_prefill_stats_on_silent_rank_emits_rank_line(self):
        # A non-logging rank without metrics used to return without ANY
        # output; the per-rank line must be emitted before that gate.
        stub = self._stub_reporter()
        prefill_stats = SimpleNamespace(log_input_tokens=42, log_hit_tokens=17)
        with self.assertLogs(LOGGER_NAME, level=logging.INFO) as cm:
            SchedulerMetricsReporter.report_prefill_stats(
                stub, batch=None, prefill_stats=prefill_stats, can_run_cuda_graph=False
            )
        self.assertEqual(len(cm.output), 1)
        line = cm.output[0]
        self.assertIn("Prefill rank batch", line)
        self.assertIn("#new-token: 42", line)
        self.assertIn("#cached-token: 17", line)

    def test_idle_metrics_flushes_pending(self):
        stub = self._stub_reporter()
        timer = FakeTimer(stub.rank_prefill_log)
        stub.rank_prefill_log.timer = timer
        stub.rank_prefill_log.record(new_tokens=9, cached_tokens=4, timed=True)
        timer.completed.append(0.01)
        with self.assertLogs(LOGGER_NAME, level=logging.INFO) as cm:
            SchedulerMetricsReporter._maybe_log_idle_metrics(stub)
        self.assertEqual(len(cm.output), 1)
        self.assertIn("#cached-token: 4", cm.output[0])


if __name__ == "__main__":
    unittest.main()
