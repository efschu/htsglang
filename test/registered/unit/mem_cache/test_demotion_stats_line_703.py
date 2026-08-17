"""#703 remainder: the demotion counters become readable.

194f1280e4 built eviction-time demotion and counted every outcome --
``demoted``, ``dropped_backpressure``, ``skipped_not_persistable``,
``skipped_no_storage``, ``failed``. Nothing ever read them. The only reference
to ``hicache_demotion`` outside the module was the import in hiradix_cache.py
used to call the two demote hooks; ``stats()`` had no consumer in any metrics
collector, endpoint or log line. So a deployment could not see demotions
failing to reach disk, and the Flip+HiCache boot could not answer whether
retention did anything -- the question the user's decode-bs finding turns on.

That is the same shape as the #410 pin ledger (a mechanism with no caller) and
the #699 detector (a verdict nobody polled), one layer over: numbers nobody
reads. So the test that matters here is not "the line formats correctly" --
it is ``test_the_scheduler_event_drain_actually_emits_the_line``, which drives
the REAL HiRadixCache.check_hicache_events and fails if the call site is
removed. The formatting tests cannot catch that, which is exactly why the
counters stayed orphaned through a green suite.
"""

import logging
import unittest
from unittest import mock

from sglang.srt.mem_cache import hicache_demotion as demotion
from sglang.test.test_utils import CustomTestCase


class _StatsBase(CustomTestCase):
    def setUp(self):
        demotion.reset_stats()
        self.addCleanup(demotion.reset_stats)

    def _enable(self, cap="4"):
        return mock.patch.dict(
            "os.environ", {"SGLANG_HICACHE_DEMOTE_ON_EVICT": cap}, clear=False
        )


class TestTheStatsLine(_StatsBase):
    def test_the_line_carries_every_counter_and_the_ratio(self):
        s = demotion.stats()
        s.demoted, s.dropped_backpressure, s.failed = 7, 2, 1
        s.skipped_not_persistable, s.skipped_no_storage = 5, 3

        line = demotion.stats_line()

        self.assertIn(demotion.STATS_MARKER, line)
        for expected in (
            "demoted=7",
            "dropped_backpressure=2",
            "failed=1",
            "skipped_not_persistable=5",
            "skipped_no_storage=3",
            "attempted=10",
            "landed_pct=70.0",
        ):
            self.assertIn(expected, line, line)

    def test_skips_are_not_counted_as_attempts(self):
        """A skip is an eviction that was never a candidate, not a failure.

        Folding skips into ``attempted`` would depress the ratio that answers
        "is the disk tier keeping up" with evictions the disk tier was never
        asked about.
        """
        s = demotion.stats()
        s.demoted, s.skipped_not_persistable, s.skipped_no_storage = 4, 100, 100

        line = demotion.stats_line()

        self.assertIn("attempted=4", line)
        self.assertIn("landed_pct=100.0", line)

    def test_an_idle_store_reports_a_full_ratio_not_a_division_error(self):
        self.assertIn("landed_pct=100.0", demotion.stats_line())
        self.assertIn("attempted=0", demotion.stats_line())


class TestTheEmissionPolicy(_StatsBase):
    def test_nothing_is_emitted_when_demotion_is_off(self):
        """The default path's log stays byte-identical."""
        with mock.patch.dict(
            "os.environ", {"SGLANG_HICACHE_DEMOTE_ON_EVICT": "0"}, clear=False
        ):
            self.assertFalse(demotion.maybe_log_stats(now=1000.0))

    def test_the_first_line_is_emitted_immediately(self):
        """A boot must be able to see demotion is on before a counter moves."""
        with self._enable():
            with self.assertLogs(demotion.logger, level=logging.INFO) as cm:
                self.assertTrue(demotion.maybe_log_stats(now=1000.0))
        self.assertTrue(any(demotion.STATS_MARKER in m for m in cm.output))

    def test_a_second_call_inside_the_interval_is_silent(self):
        with self._enable():
            demotion.maybe_log_stats(now=1000.0)
            demotion.stats().demoted += 1
            self.assertFalse(demotion.maybe_log_stats(now=1030.0, interval_s=60.0))

    def test_after_the_interval_a_changed_counter_is_reported(self):
        with self._enable():
            demotion.maybe_log_stats(now=1000.0)
            demotion.stats().dropped_backpressure += 3
            with self.assertLogs(demotion.logger, level=logging.INFO) as cm:
                self.assertTrue(demotion.maybe_log_stats(now=1100.0, interval_s=60.0))
        self.assertTrue(any("dropped_backpressure=3" in m for m in cm.output))

    def test_a_steady_state_does_not_repeat_the_same_numbers(self):
        """Repeating an unchanged line every minute for hours trains the reader
        to skip it, which is how a stat meant to be watched stops being read."""
        with self._enable():
            demotion.maybe_log_stats(now=1000.0)
            self.assertFalse(demotion.maybe_log_stats(now=2000.0, interval_s=60.0))
            self.assertFalse(demotion.maybe_log_stats(now=3000.0, interval_s=60.0))

    def test_an_unchanged_check_does_not_delay_the_next_report(self):
        """The silent path must not restamp the clock: a change landing just
        after a no-op check would otherwise wait a full extra interval."""
        with self._enable():
            demotion.maybe_log_stats(now=1000.0)
            self.assertFalse(demotion.maybe_log_stats(now=2000.0, interval_s=60.0))
            demotion.stats().demoted += 1
            self.assertTrue(demotion.maybe_log_stats(now=2001.0, interval_s=60.0))


class _FakeController:
    def __init__(self):
        self.storage_backend = None


class TestTheCallSiteExists(_StatsBase):
    """The anti-orphan case. Everything above passes against a module nobody
    calls -- which is precisely how these counters shipped unread."""

    def test_the_scheduler_event_drain_actually_emits_the_line(self):
        from sglang.srt.mem_cache.hiradix_cache import HiRadixCache

        cache = object.__new__(HiRadixCache)  # no pools, no device, no boot
        cache.cache_controller = _FakeController()
        cache.enable_storage = False
        cache.enable_storage_metrics = False
        cache._drain_async_work = lambda: None
        cache.writing_check = lambda: None
        cache.loading_check = lambda: None

        with self._enable():
            with self.assertLogs(demotion.logger, level=logging.INFO) as cm:
                HiRadixCache.check_hicache_events(cache)

        self.assertTrue(
            any(demotion.STATS_MARKER in m for m in cm.output),
            "check_hicache_events did not emit the demotion stats line",
        )

    def test_the_event_drain_stays_silent_when_demotion_is_off(self):
        from sglang.srt.mem_cache.hiradix_cache import HiRadixCache

        cache = object.__new__(HiRadixCache)
        cache.cache_controller = _FakeController()
        cache.enable_storage = False
        cache.enable_storage_metrics = False
        cache._drain_async_work = lambda: None
        cache.writing_check = lambda: None
        cache.loading_check = lambda: None

        with mock.patch.dict(
            "os.environ", {"SGLANG_HICACHE_DEMOTE_ON_EVICT": "0"}, clear=False
        ):
            with mock.patch.object(demotion.logger, "info") as info:
                HiRadixCache.check_hicache_events(cache)
        info.assert_not_called()


if __name__ == "__main__":
    unittest.main()
