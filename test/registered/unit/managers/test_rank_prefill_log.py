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
import os
import unittest
import unittest.mock

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
        f"#249 default-device collection leak broke the import chain: {_import_err}",
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
    """Returns a scripted collective total for whatever slot it is handed.

    ``values`` entries are either a plain total (seconds, no family
    decomposition -- what an unlabelled region looks like) or a
    ``(total_s, {family: FamilyStat})`` tuple.
    """

    def __init__(self, values):
        self.values = list(values)

    def harvest_detail(self, slot):
        from sglang.srt.utils.collective_clock import HarvestResult

        v = self.values.pop(0)
        if v is None:
            return None
        if isinstance(v, tuple):
            total_s, families = v
            return HarvestResult(total_s=total_s, families=families)
        return HarvestResult(total_s=v, families={})

    def harvest(self, slot):
        result = self.harvest_detail(slot)
        return None if result is None else result.total_s


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


class TestCollectiveFamilies(unittest.TestCase):
    """#588: the wait number decomposes by collective family.

    One summed ``wait`` cannot arbitrate between "TP all-reduce payload grows
    with new tokens" and "DCP attention traffic grows with context" -- the two
    have the same units and different fixes. These pin the decomposition.
    """

    def setUp(self):
        # span() asks CUDA whether a graph is capturing. Patch that single
        # query rather than skipping without a GPU: the family accounting is
        # pure arithmetic over faked events and deserves to be pinned on a
        # CPU box, which is where this suite runs.
        patcher = unittest.mock.patch.object(
            torch.cuda, "is_current_stream_capturing", return_value=False
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _clock(self, timestamps, ready=True):
        clock = CollectiveClock()
        events = [FakeEvent(ts, ready) for ts in timestamps]
        clock._acquire = lambda: events.pop(0)
        return clock

    def test_per_family_totals_counts_and_max(self):
        # tp.all_reduce: spans of 2 ms and 6 ms. dcp.all_gather: one 3 ms.
        clock = self._clock([0.0, 2.0, 10.0, 16.0, 20.0, 23.0])
        clock.arm()
        with clock.span("tp.all_reduce"):
            pass
        with clock.span("tp.all_reduce"):
            pass
        with clock.span("dcp.all_gather"):
            pass
        result = clock.harvest_detail(clock.disarm())
        self.assertAlmostEqual(result.total_s, 0.011)
        self.assertEqual(set(result.families), {"tp.all_reduce", "dcp.all_gather"})
        tp = result.families["tp.all_reduce"]
        self.assertAlmostEqual(tp.total_ms, 8.0)
        self.assertEqual(tp.count, 2)
        # max is NOT the total: same 8 ms could be one stall or two transfers.
        self.assertAlmostEqual(tp.max_ms, 6.0)
        ag = result.families["dcp.all_gather"]
        self.assertAlmostEqual(ag.total_ms, 3.0)
        self.assertEqual(ag.count, 1)
        self.assertAlmostEqual(ag.max_ms, 3.0)

    def test_family_totals_sum_to_the_grand_total(self):
        clock = self._clock([0.0, 2.0, 10.0, 16.0, 20.0, 23.0])
        clock.arm()
        for fam in ("tp.all_reduce", "tp.all_reduce", "dcp.all_gather"):
            with clock.span(fam):
                pass
        result = clock.harvest_detail(clock.disarm())
        self.assertAlmostEqual(
            sum(f.total_ms for f in result.families.values()),
            result.total_s * 1000.0,
        )

    def test_unlabelled_span_lands_in_other_and_is_still_counted(self):
        """An unlabelled collective must never be dropped from the total."""
        clock = self._clock([0.0, 4.0])
        clock.arm()
        with clock.span():
            pass
        result = clock.harvest_detail(clock.disarm())
        self.assertAlmostEqual(result.total_s, 0.004)
        self.assertEqual(list(result.families), ["other"])
        self.assertAlmostEqual(result.families["other"].total_ms, 4.0)

    def test_label_scope_overrides_the_dispatch_site_default(self):
        """The caller knows more than the generic dispatch site."""
        clock = self._clock([0.0, 5.0])
        clock.arm()
        with clock.label_scope("cp.lse_ag"):
            with clock.span("dcp.all_gather"):
                pass
        result = clock.harvest_detail(clock.disarm())
        self.assertEqual(list(result.families), ["cp.lse_ag"])

    def test_label_scope_restores_the_previous_label(self):
        clock = self._clock([0.0, 1.0, 5.0, 9.0])
        clock.arm()
        with clock.label_scope("cp.lse_ag"):
            with clock.span("dcp.all_gather"):
                pass
        # Outside the scope the dispatch-site default applies again.
        with clock.span("dcp.all_gather"):
            pass
        result = clock.harvest_detail(clock.disarm())
        self.assertEqual(set(result.families), {"cp.lse_ag", "dcp.all_gather"})

    def test_harvest_float_view_matches_detail_total(self):
        clock = self._clock([0.0, 7.0])
        clock.arm()
        with clock.span("tp.all_reduce"):
            pass
        self.assertAlmostEqual(clock.harvest(clock.disarm()), 0.007)

    def test_unreadable_slot_yields_no_detail(self):
        clock = self._clock([0.0, 7.0], ready=False)
        clock.arm()
        with clock.span("tp.all_reduce"):
            pass
        self.assertIsNone(clock.harvest_detail(clock.disarm()))


class TestFamilyLabelDerivation(unittest.TestCase):
    """The four dispatch sites must carry the family of THEIR group."""

    def test_families_are_derived_from_the_group_name(self):
        from sglang.srt.distributed.parallel_state import collective_clock_families

        self.assertEqual(
            collective_clock_families("tp"),
            ("tp.all_reduce", "tp.all_gather", "tp.all_gatherv", "tp.reduce_scatterv"),
        )
        self.assertEqual(collective_clock_families("dcp")[1], "dcp.all_gather")

    def test_each_dispatch_site_passes_its_own_family(self):
        """Pin the wiring: a site that passes the wrong family would report
        all_gather traffic as all_reduce, which is worse than no label."""

        src = open(
            os.path.join(
                os.path.dirname(
                    os.path.dirname(
                        os.path.dirname(
                            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                        )
                    )
                ),
                "python",
                "sglang",
                "srt",
                "distributed",
                "parallel_state.py",
            )
        ).read()
        for op in ("all_reduce", "all_gather", "all_gatherv", "reduce_scatterv"):
            pattern = (
                r"_COLLECTIVE_CLOCK\.span\(self\._clock_family_%s\):\s*\n\s*"
                r"return self\.%s\(" % (op, op)
            )
            self.assertRegex(
                src,
                pattern,
                f"dispatch site for {op} does not pass its matching family",
            )


class TestWaitByFamilyLogLine(unittest.TestCase):
    """The log-line extension: armed-only, compact, biggest family first."""

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

    def test_families_are_appended_after_the_split(self):
        from sglang.srt.utils.collective_clock import FamilyStat

        log, timer = self._make(
            [
                (
                    0.05,
                    {
                        "dcp.all_gather": FamilyStat(10.0, 8, 4.0),
                        "tp.all_reduce": FamilyStat(40.0, 2, 30.0),
                    },
                )
            ]
        )
        line = self._emit(log, timer, [(64, 3, False)], [(0.25, Slot())])
        self.assertIn("gpu-ms: 250.0 (compute 200.0, wait 50.0)", line)
        # Biggest contributor first, so the owner of the wait is readable.
        self.assertIn(
            "(wait by family: tp.all_reduce 40.0/2x, dcp.all_gather 10.0/8x)", line
        )
        self.assertLess(line.index("wait 50.0"), line.index("wait by family"))

    def test_no_families_means_no_extension(self):
        """Byte-neutral when nothing was labelled: the old line, unchanged."""
        log, timer = self._make([0.05])
        line = self._emit(log, timer, [(64, 3, False)], [(0.25, Slot())])
        self.assertIn("#chunks: 1, gpu-ms: 250.0 (compute 200.0, wait 50.0)", line)
        self.assertNotIn("wait by family", line)

    def test_unclocked_line_is_byte_identical_to_before(self):
        """The off path must not gain a character."""
        log = RankPrefillLog()
        timer = FakeTimer(log)
        log.timer = timer
        log.clock = None
        line = self._emit(log, timer, [(64, 3, False)], [0.25])
        self.assertTrue(
            line.endswith(
                "Prefill rank batch, #new-token: 64, #cached-token: 3, "
                "#chunks: 1, gpu-ms: 250.0"
            ),
            line,
        )

    def test_graph_covered_forward_reports_no_families(self):
        """No split means no decomposition either -- never a fake zero."""
        from sglang.srt.utils.collective_clock import FamilyStat

        log, timer = self._make([(0.05, {"tp.all_reduce": FamilyStat(40.0, 2, 30.0)})])
        line = self._emit(log, timer, [(64, 3, True)], [(0.25, Slot())])
        self.assertNotIn("wait by family", line)
        self.assertNotIn("compute", line)

    def test_families_merge_across_chunks_folded_into_one_line(self):
        from sglang.srt.utils.collective_clock import FamilyStat

        log, timer = self._make(
            [
                (0.04, {"tp.all_reduce": FamilyStat(40.0, 2, 30.0)}),
                (0.06, {"tp.all_reduce": FamilyStat(60.0, 3, 35.0)}),
            ]
        )
        line = self._emit(
            log, timer, [(5, 0, False), (5, 0, False)], [(0.10, Slot()), (0.10, Slot())]
        )
        self.assertIn("tp.all_reduce 100.0/5x", line)


if __name__ == "__main__":
    unittest.main()
