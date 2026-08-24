"""#856: the HiCache fence's cost is readable from the flip's own stats.

WHY. Once the flip carries no KV, cutover-blocking time is FENCE + WEIGHTS
REFILL -- those two and nothing else. The weights refill has been in the stats
dict since #690 (`movers_ms`) and now says which half bound it (#856 a). The
fence had no entry at all: its cost was visible only as a census SEGMENT, the
delta between the `flip_writeback` and `hicache_quiesce` marks (74.8 ms in
W25), which nothing reading `last_stats` can see.

`maybe_flip_writeback` already returns a `FlipWritebackReport` carrying
`elapsed_s`; the seam was using it as a bare truthiness test and throwing the
number away.

THE ONE RULE THIS FILE EXISTS TO PIN. `None` means NO FENCE RAN and must never
become `0.0`. The fence is skipped outright when there is no canonical store
(`require_canonical_store` refuses), and a defaulted zero would report such a
flip as fully fenced while nothing was persisted -- the #606 defaulted-
measurement shape, in the one place where believing it means losing KV.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

import types
import unittest

from sglang.srt.managers.phase_flip_runtime import _writeback_fence_ms
from sglang.test.test_utils import CustomTestCase


def _report(elapsed_s):
    return types.SimpleNamespace(elapsed_s=elapsed_s)


class TestTheFenceCostIsReported(CustomTestCase):
    def test_a_report_becomes_milliseconds(self):
        self.assertAlmostEqual(_writeback_fence_ms(_report(0.0748)), 74.8, places=6)

    def test_a_two_second_deadline_hit_is_reported_as_such(self):
        # The deadline default is 2.0 s; a fence that spends it is the single
        # most important number this instrument can carry, because it is
        # cutover-blocking time the user feels directly.
        self.assertAlmostEqual(_writeback_fence_ms(_report(2.0)), 2000.0, places=6)


class TestNoneIsNotZero(CustomTestCase):
    """The whole point. Both readings must survive."""

    def test_no_fence_reads_None(self):
        self.assertIsNone(_writeback_fence_ms(None))

    def test_a_genuinely_free_fence_reads_zero_not_None(self):
        # THE CAN-FAIL DIRECTION. An implementation that returned None for
        # anything falsy would satisfy `test_no_fence_reads_None` while
        # erasing a real measurement of zero.
        got = _writeback_fence_ms(_report(0.0))
        self.assertIsNotNone(got)
        self.assertEqual(got, 0.0)

    def test_an_unreadable_report_abstains_rather_than_lying(self):
        for bad in (
            types.SimpleNamespace(),
            types.SimpleNamespace(elapsed_s=None),
            types.SimpleNamespace(elapsed_s="n/a"),
            object(),
        ):
            with self.subTest(report=bad):
                self.assertIsNone(_writeback_fence_ms(bad))

    def test_an_instrument_never_raises_into_the_seam(self):
        # It runs on the flip path with requests parked; the standing rule in
        # this module is that an instrument may cost a missing line, never a
        # cutover.
        class _Angry:
            @property
            def elapsed_s(self):
                raise RuntimeError("probe exploded")

        self.assertIsNone(_writeback_fence_ms(_Angry()))


if __name__ == "__main__":
    unittest.main()
