# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""`--enable-metrics` for the translator tenant.

The standing order is that every server boot carries `--enable-metrics`. The
translator was the boot that had no such flag, which is why the §17.8.10
backlog (a turn that produced nothing for 90 s and then flushed 8188 frames
at once) could be described but not localised.

What is pinned here is the part that would otherwise ship looking correct:
that the endpoint REFUSES rather than answers emptily when disabled, that a
recorded turn actually moves the series, and that the live depth gauges read
their sources at scrape time instead of reporting a value cached when nothing
was queued.
"""

import unittest

from sglang.srt.translator import metrics


class TestMetrics(unittest.TestCase):
    def setUp(self):
        metrics.reset_for_test()

    def tearDown(self):
        metrics.reset_for_test()

    def test_disabled_records_nothing(self):
        """Off by default, and off means off -- not "on but empty"."""
        self.assertFalse(metrics.enabled())
        metrics.record_turn({"asr_ms": 120.0, "total_ms": 6000.0})
        self.assertIn("translator_turns_total 0", metrics.render())

    def test_a_turn_moves_the_series(self):
        metrics.enable()
        metrics.record_turn(
            {"asr_ms": 120.0, "mt_total_ms": 400.0, "total_ms": 6000.0}
        )
        out = metrics.render()
        self.assertIn("translator_turns_total 1", out)
        self.assertIn("translator_asr_seconds_sum 0.120000", out)
        self.assertIn("translator_mt_total_seconds_sum 0.400000", out)
        self.assertIn("translator_total_seconds_max 6.000000", out)

    def test_the_max_keeps_the_worst_turn(self):
        """The worst turn is the one the user complains about."""
        metrics.enable()
        metrics.record_turn({"total_ms": 6000.0})
        metrics.record_turn({"total_ms": 84000.0})
        metrics.record_turn({"total_ms": 5000.0})
        out = metrics.render()
        self.assertIn("translator_total_seconds_max 84.000000", out)
        self.assertIn("translator_turns_total 3", out)

    def test_a_depth_is_read_at_scrape_time(self):
        """THE falsifier for the gauges.

        A depth captured at registration would report the idle value forever,
        which is wrong exactly when a queue is building -- the only moment the
        number is worth having.
        """
        metrics.enable()
        depth = {"value": 0.0}
        metrics.register_depths(
            "translator_test_depth", "test", lambda: depth["value"]
        )
        self.assertIn("translator_test_depth 0.000000", metrics.render())
        depth["value"] = 7.0
        self.assertIn("translator_test_depth 7.000000", metrics.render())

    def test_a_broken_gauge_does_not_kill_the_scrape(self):
        """A metric surface that dies takes the diagnosis with it."""
        metrics.enable()

        def _boom():
            raise RuntimeError("gauge exploded")

        metrics.register_depths("translator_bad", "test", _boom)
        metrics.register_depths("translator_good", "test", lambda: 1.0)
        out = metrics.render()
        self.assertNotIn("translator_bad", out)
        self.assertIn("translator_good 1.000000", out)


if __name__ == "__main__":
    unittest.main()
