"""X-SUM-PRICE (Nutzer 05.10.2026): X bounds what D prefills IN TOTAL.

Boot 1005_063507, 06:42Z: the P phase's seat cap (``p-cap`` = free D seats)
left the 12.2k SHORTs weg2-8-32 / 8-34 / 8-37 in the queue. Back on D,
``_asr_queued_short_to_d`` tested each of them ALONE against X=12288 (all
passed) and D prefilled 36.7k fresh tokens in ~40 s with cached=0 -- the sum
DECODE-COLLECT sends to P (``tokens > X`` -> route P). The queued path now
takes the oldest SHORTs only while their SUM (plus what already waits for D)
stays within X; the rest stays queued for P.
"""

import collections
import types
import unittest

from sglang.srt.weg2 import front


def _p(rid, uncached, t):
    return types.SimpleNamespace(
        rid=rid, est_uncached=uncached, t_arrive=t, d_eligible=True, intake_stalled=False,
        leg1_done=False, reroutes=0, x_requeues=0, p_only=False, x_deferred=False,
        d_direct=False)


class TheSumPricedTake(unittest.TestCase):
    def test_three_12k_shorts_do_not_fit_one_x_of_12288(self):
        es = [_p("8-32", 12177, 1.0), _p("8-34", 12281, 2.0), _p("8-37", 12213, 3.0)]
        taken, kept = front._sum_priced_take(es, carried=0, limit=12288)
        self.assertEqual([p.rid for p in taken], ["8-32"])  # the oldest, alone
        self.assertEqual([p.rid for p in kept], ["8-34", "8-37"])

    def test_small_shorts_all_fit(self):
        es = [_p("a", 25, 1.0), _p("b", 400, 2.0), _p("c", 3000, 3.0)]
        taken, kept = front._sum_priced_take(es, carried=0, limit=12288)
        self.assertEqual([p.rid for p in taken], ["a", "b", "c"])
        self.assertEqual(kept, [])

    def test_what_already_waits_for_d_counts(self):
        es = [_p("a", 3000, 1.0)]
        taken, kept = front._sum_priced_take(es, carried=10000, limit=12288)
        self.assertEqual(taken, [])
        self.assertEqual([p.rid for p in kept], ["a"])

    def test_oldest_first_and_a_younger_small_one_backfills(self):
        es = [_p("young-small", 500, 5.0), _p("old-big", 11000, 1.0), _p("mid-big", 11500, 3.0)]
        taken, kept = front._sum_priced_take(es, carried=0, limit=12288)
        self.assertEqual([p.rid for p in taken], ["old-big", "young-small"])
        self.assertEqual([p.rid for p in kept], ["mid-big"])


class _Fake:
    """The slice of Front the queued-SHORT hand-over reads."""

    def __init__(self, queue, ready=()):
        self.admit_d, self.state, self.awake = True, "serving", "D"
        self.x_split = False
        self.tp_prefill_max_tokens = 12288
        self.queue = collections.deque(queue)
        self._ready_for_d = list(ready)
        self.counters = collections.Counter()
        self.synced = 0

    def _sync_batch_gate(self):
        self.synced += 1

    def _x_band_floor(self):
        return 4096

    def _d_holds_work(self):
        return False


class TheQueuedShortHandOverUsesTheSum(unittest.TestCase):
    def test_boot_063507_minute_hands_one_not_three_to_d(self):
        qs = [_p("8-32", 12177, 1.0), _p("8-34", 12281, 2.0), _p("8-37", 12213, 3.0)]
        fake = _Fake(qs)
        moved = front.Front._asr_queued_short_to_d(fake, list(qs), 10.0)
        self.assertEqual([p.rid for p in moved], ["8-32"])
        self.assertEqual([p.rid for p in fake.queue], ["8-34", "8-37"])  # stay queued: P takes them
        self.assertEqual(fake.counters["arrival_seat_queue_sum_kept"], 2)
        self.assertEqual(fake.counters["arrival_seat_queue_to_d"], 1)
        self.assertTrue(moved[0].d_direct)

    def test_a_short_already_waiting_for_d_keeps_the_next_one_on_the_queue(self):
        waiting = _p("w", 9000, 0.5)
        waiting.d_direct = True
        nxt = _p("n", 5000, 1.0)
        fake = _Fake([nxt], ready=[waiting])
        moved = front.Front._asr_queued_short_to_d(fake, [nxt], 10.0)
        self.assertEqual(moved, [])
        self.assertEqual([p.rid for p in fake.queue], ["n"])

    def test_small_shorts_still_all_go_to_d(self):
        qs = [_p("a", 25, 1.0), _p("b", 40, 2.0)]
        fake = _Fake(qs)
        moved = front.Front._asr_queued_short_to_d(fake, list(qs), 10.0)
        self.assertEqual([p.rid for p in moved], ["a", "b"])
        self.assertEqual(len(fake.queue), 0)
        self.assertEqual(fake.counters["arrival_seat_queue_sum_kept"], 0)


if __name__ == "__main__":
    unittest.main()
