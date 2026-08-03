"""Unit tests for the per-rank prefill log (RankPrefillLog).

The "Prefill batch" summary is emitted only on the stats-logging rank; the
per-rank line ("Prefill rank batch, ...") must run on EVERY rank, ahead of the
logging-rank gate, and must carry #cached-token as rank-local evidence that a
prefill was computed rather than served from the prefix cache. GPU durations
attach deferred (FIFO pairing with CUDA-event completions); no host sync is
involved anywhere. Pure CPU: the timer is faked, no CUDA events are created.

The second half covers the compute/wait split: gpu-ms alone folds the
collective barrier into the measurement, so the line also reports the device
time spent inside collectives (``wait``) and the remainder (``compute``), and
suppresses the split rather than reporting a fake zero when it cannot be
observed (graph replay, unreadable events).
"""

import logging
import unittest

import torch
from types import SimpleNamespace

import pytest

# Import shim for the #249 default-device collection leak: an earlier
# collected module may leave ``torch.set_default_device(<accelerator>)``
# active, which crashes this module's sglang import chain (it reaches the
# ``compressed_tensors`` site package, whose import constructs tensors) on a
# box without that accelerator. Skip the module instead of erroring; the
# side effects on the process are identical to the crash, so the fate of
# every other collected module is unchanged. On a machine where the
# accelerator exists (the registered CI runners) the import succeeds and
# the tests run normally.
try:
    from sglang.srt.managers.scheduler_components.metrics_reporter import (
        RankPrefillLog,
        SchedulerMetricsReporter,
    )
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.utils.collective_clock import CollectiveClock, Slot
    from sglang.test.ci.ci_register import register_cpu_ci
except RuntimeError as _import_err:  # pragma: no cover - leak-dependent
    pytest.skip(
        f"#249 default-device collection leak broke the import chain: "
        f"{_import_err}",
        allow_module_level=True,
    )

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

LOGGER_NAME = "sglang.srt.managers.scheduler_components.metrics_reporter"


class FakeTimer:
    """Stands in for DeviceTimer: durations 'complete' when the test says so."""

    def __init__(self, log: RankPrefillLog):
        self._log = log
        self.completed = []

    def _report(self):
        while self.completed:
            item = self.completed.pop(0)
            if isinstance(item, tuple):
                t, slot = item
                self._log._on_duration(t, collective_slot=slot)
            else:
                self._log._on_duration(item)


class FakeEvent:
    """CUDA event stand-in: fixed timestamp, controllable completion."""

    def __init__(self, ts: float, ready: bool = True):
        self.ts = ts
        self.ready = ready
        self.recorded = 0

    def record(self):
        self.recorded += 1

    def query(self):
        return self.ready

    def elapsed_time(self, other):
        return other.ts - self.ts


class FakeClock:
    """Returns a scripted collective total for whatever slot it is handed."""

    def __init__(self, values):
        self.values = list(values)

    def harvest(self, slot):
        return self.values.pop(0)


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


class TestComputeWaitSplit(unittest.TestCase):
    """The line must separate rank-local compute from collective wait."""

    def _make(self, wait_values):
        log = RankPrefillLog()
        timer = FakeTimer(log)
        log.timer = timer
        log.clock = FakeClock(wait_values)
        return log, timer

    def _emit(self, log, timer, records, completions):
        for new_tokens, cached_tokens, graphed in records:
            log.record(
                new_tokens=new_tokens,
                cached_tokens=cached_tokens,
                timed=True,
                graphed=graphed,
            )
        timer.completed.extend(completions)
        with self.assertLogs(LOGGER_NAME, level=logging.INFO) as cm:
            log.flush()
        self.assertEqual(len(cm.output), 1)
        return cm.output[0]

    def test_split_is_appended_after_gpu_ms(self):
        log, timer = self._make([0.05])
        line = self._emit(log, timer, [(64, 3, False)], [(0.25, Slot())])
        # Prefix-compatible: everything the old line carried is unchanged and
        # in the same order, the split is appended.
        self.assertIn("#chunks: 1, gpu-ms: 250.0 (compute 200.0, wait 50.0)", line)

    def test_split_sums_over_chunks(self):
        log, timer = self._make([0.02, 0.03])
        line = self._emit(
            log,
            timer,
            [(10, 1, False), (20, 2, False)],
            [(0.1, Slot()), (0.2, Slot())],
        )
        self.assertIn("#chunks: 2", line)
        self.assertIn("gpu-ms: 300.0 (compute 250.0, wait 50.0)", line)

    def test_graphed_forward_reports_no_split(self):
        # Collectives inside a replayed CUDA graph never run the Python that
        # records the events; a 0.0 wait would be a lie, so the split is
        # dropped and the line falls back to its previous shape.
        log, timer = self._make([0.0])
        line = self._emit(log, timer, [(64, 0, True)], [(0.25, Slot())])
        self.assertIn("gpu-ms: 250.0", line)
        self.assertNotIn("compute", line)

    def test_one_graphed_chunk_drops_the_whole_line_split(self):
        log, timer = self._make([0.02, 0.0])
        line = self._emit(
            log,
            timer,
            [(10, 0, False), (20, 0, True)],
            [(0.1, Slot()), (0.2, Slot())],
        )
        self.assertIn("gpu-ms: 300.0", line)
        self.assertNotIn("compute", line)

    def test_unreadable_slot_reports_no_split(self):
        log, timer = self._make([None])
        line = self._emit(log, timer, [(64, 0, False)], [(0.25, Slot())])
        self.assertIn("gpu-ms: 250.0", line)
        self.assertNotIn("compute", line)

    def test_capture_skipped_slot_reports_no_split(self):
        slot = Slot()
        slot.graph_capture_skipped = True
        log, timer = self._make([0.05])
        line = self._emit(log, timer, [(64, 0, False)], [(0.25, slot)])
        self.assertIn("gpu-ms: 250.0", line)
        self.assertNotIn("compute", line)

    def test_compute_is_clamped_at_zero(self):
        # Event granularity can make the collective sum marginally exceed the
        # enclosing window; compute must not go negative.
        log, timer = self._make([0.26])
        line = self._emit(log, timer, [(64, 0, False)], [(0.25, Slot())])
        self.assertIn("(compute 0.0, wait 260.0)", line)

    def test_no_clock_keeps_the_old_line(self):
        # Non-CUDA / not installed: unchanged behaviour.
        log = RankPrefillLog()
        timer = FakeTimer(log)
        log.timer = timer
        line = self._emit(log, timer, [(64, 0, False)], [0.25])
        self.assertIn("gpu-ms: 250.0", line)
        self.assertNotIn("compute", line)


class TestCollectiveClock(unittest.TestCase):
    """Arming, re-entrancy and deferred harvesting, with faked events."""

    def _clock(self, timestamps, ready=True):
        clock = CollectiveClock()
        events = [FakeEvent(ts, ready) for ts in timestamps]
        clock._acquire = lambda: events.pop(0)
        return clock

    def test_disarmed_span_records_nothing(self):
        clock = self._clock([0.0, 1.0])
        self.assertFalse(clock.armed)
        with clock.span():
            pass
        self.assertIsNone(clock.disarm())

    @unittest.skipUnless(
        torch.cuda.is_available(), "CollectiveClock spans query CUDA capture state"
    )
    def test_armed_span_accumulates(self):
        clock = self._clock([0.0, 2.0, 5.0, 8.0])
        clock.arm()
        with clock.span():
            pass
        with clock.span():
            pass
        slot = clock.disarm()
        self.assertEqual(len(slot.pairs), 2)
        self.assertAlmostEqual(clock.harvest(slot), 0.005)

    @unittest.skipUnless(
        torch.cuda.is_available(), "CollectiveClock spans query CUDA capture state"
    )
    def test_nested_span_is_counted_once(self):
        # A collective implemented via other collectives must not be charged
        # once per level.
        clock = self._clock([0.0, 10.0])
        clock.arm()
        with clock.span():
            self.assertFalse(clock.armed)
            with clock.span():
                pass
        slot = clock.disarm()
        self.assertEqual(len(slot.pairs), 1)
        self.assertAlmostEqual(clock.harvest(slot), 0.010)

    @unittest.skipUnless(
        torch.cuda.is_available(), "CollectiveClock spans query CUDA capture state"
    )
    def test_harvest_is_query_only_and_defers(self):
        clock = self._clock([0.0, 4.0], ready=False)
        clock.arm()
        with clock.span():
            pass
        slot = clock.disarm()
        self.assertIsNone(clock.harvest(slot))
        for _, end in slot.pairs:
            end.ready = True
        for start, _ in slot.pairs:
            start.ready = True
        self.assertAlmostEqual(clock.harvest(slot), 0.004)

    @unittest.skipUnless(
        torch.cuda.is_available(), "CollectiveClock spans query CUDA capture state"
    )
    def test_harvest_returns_events_to_the_pool(self):
        clock = CollectiveClock()
        events = [FakeEvent(0.0), FakeEvent(3.0)]
        clock._acquire = lambda: events.pop(0)
        clock.arm()
        with clock.span():
            pass
        slot = clock.disarm()
        clock.harvest(slot)
        self.assertEqual(len(clock._pool), 2)
        self.assertEqual(slot.pairs, [])

    def test_empty_slot_is_zero_not_unknown(self):
        clock = CollectiveClock()
        clock.arm()
        self.assertEqual(clock.harvest(clock.disarm()), 0.0)
        self.assertIsNone(clock.harvest(None))


class TestReporterEmitsAheadOfLoggingGate(unittest.TestCase):
    def _stub_reporter(self):
        return SimpleNamespace(
            rank_prefill_log=RankPrefillLog(),
            is_stats_logging_rank=False,
            current_scheduler_metrics_enabled=False,
            # Fixture drift, 2026-07-29: #274 slice D R1 (9d4c495fcf) added a
            # MONOTONE prefill-token counter and deliberately placed the
            # increment AHEAD of the logging-rank gate -- i.e. in exactly the
            # region this stub exercises. The real reporter initialises it
            # unconditionally in __init__ (metrics_reporter.py:277), so the
            # stub simply has to carry it too.
            prefill_tokens_total=0,
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
