"""#656: the dynamic-chunking arm must be able to PROVE it engaged.

The only line reporting a predicted chunk width was DEBUG-level, so at the
default log level an A/B of the arm produced a throughput number with no
evidence the mechanism ever moved. A throughput delta with no engagement
proof measures the run, not the feature.

These tests pin the three properties that make the new line usable as
evidence: it is INFO, it fires on the first deviation, and it does NOT fire
once per scheduling iteration (which is thousands of times a minute and
would perturb the thing being measured).
"""

import logging
import types
import unittest

from sglang.srt.managers.scheduler import Scheduler


class _Sched:
    """The narrowest object the logging helper touches."""

    _log_dynamic_chunk_engagement = Scheduler._log_dynamic_chunk_engagement

    def __init__(self, static=512):
        self.chunked_prefill_size = static
        self.ps = types.SimpleNamespace(pp_rank=1)


class TestEngagementLine(unittest.TestCase):
    def setUp(self):
        self.s = _Sched()

    def test_it_is_info_not_debug(self):
        with self.assertLogs("sglang.srt.managers.scheduler", level="INFO") as cm:
            self.s._log_dynamic_chunk_engagement(128, 0)
        self.assertTrue(any("ENGAGED" in m for m in cm.output))
        self.assertTrue(any(r.levelno == logging.INFO for r in cm.records))

    def test_it_reports_the_delta_against_the_static_size(self):
        with self.assertLogs("sglang.srt.managers.scheduler", level="INFO") as cm:
            self.s._log_dynamic_chunk_engagement(128, 4096)
        line = cm.output[0]
        self.assertIn("chunk width 128", line)
        self.assertIn("static --chunked-prefill-size is 512", line)
        self.assertIn("-384", line)
        self.assertIn("history_len=4096", line)

    def test_repeats_of_the_same_width_are_silent(self):
        # Edge-triggered: this helper is on a per-iteration path.
        with self.assertLogs("sglang.srt.managers.scheduler", level="INFO") as cm:
            self.s._log_dynamic_chunk_engagement(128, 0)
            for _ in range(500):
                self.s._log_dynamic_chunk_engagement(128, 0)
        self.assertEqual(len(cm.output), 1)

    def test_each_new_width_is_reported(self):
        with self.assertLogs("sglang.srt.managers.scheduler", level="INFO") as cm:
            self.s._log_dynamic_chunk_engagement(128, 0)
            self.s._log_dynamic_chunk_engagement(256, 0)
            self.s._log_dynamic_chunk_engagement(128, 0)
        self.assertEqual(len(cm.output), 3)

    def test_the_upward_half_is_reported_too(self):
        # The predictor moves in BOTH directions (down to base//4, up via the
        # raised prefill ceiling); an engagement proof that only saw the
        # downward half would miss the risky one.
        with self.assertLogs("sglang.srt.managers.scheduler", level="INFO") as cm:
            self.s._log_dynamic_chunk_engagement(640, 0)
        self.assertIn("+128", cm.output[0])


if __name__ == "__main__":
    unittest.main()
