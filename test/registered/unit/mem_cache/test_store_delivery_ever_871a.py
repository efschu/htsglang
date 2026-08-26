"""#871a: can any instrument answer "has a byte EVER reached the store"?

Before this, no. Every number the writeback fence reports is per-cutover, and
each one is legitimately zero on a quiet instance:

    eligible=0 staged=0 already_staged=0 acked=0 outstanding=0 elapsed=0.000s

That is what a healthy store looks like when there is nothing to persist. It is
ALSO, byte for byte, what a store that has never taken a single write looks
like -- misconfigured, unreachable, or fed by a writer that never fires.
``report.complete`` calls both of them complete, because ``outstanding == 0`` is
trivially true when nothing was sent.

So the fence could not fail to look healthy, which is the INDICATOR shape: an
instrument is only a finding once it has been checked that it measures what it
claims. "Is the mechanism wired?" was answerable. "Has it ever delivered?" was
not, and that is the one that decides whether a read-through can hit.

WHY A LIFETIME COUNT IS THE ANSWER. The two states are identical at every
individual sample and differ only over TIME. "This fence acked nothing" is a
statement about one cutover; "no fence has ever acked anything, and fences have
been running" is a statement about the store's existence. No per-fence field can
carry the second one, however many are added.

WHAT THIS SUITE PINS
* a delivering store never trips the alarm, however many fences run;
* a store that never delivers trips it, ONCE, at the threshold;
* the threshold is not 1 -- an idle instance's first quiet fences are
  legitimate, and an alarm that fires on them gets muted, which is how the
  original condition survived;
* the counter is a LOWER BOUND and is honest about it: nonzero proves
  delivery, zero is evidence and is worded as evidence.

Hermetic: no CUDA, no server, no boot.
"""

import unittest

from sglang.srt.mem_cache import hicache_flip_writeback as W
from sglang.srt.mem_cache.hicache_flip_writeback import (
    STORE_NEVER_DELIVERED_AFTER,
    FlipWritebackReport,
    reset_store_delivery_counters,
    store_delivery_counters,
)
from sglang.test.test_utils import CustomTestCase


def _report(acked: int, *, eligible: int = 0, outstanding: int = 0):
    return FlipWritebackReport(
        eligible=eligible,
        staged=0,
        already_staged=0,
        acknowledged=acked,
        outstanding=outstanding,
        elapsed_s=0.0,
        deadline_s=2.0,
    )


class TestStoreDeliveryEver(CustomTestCase):
    def setUp(self):
        reset_store_delivery_counters()
        self.addCleanup(reset_store_delivery_counters)

    def test_a_store_that_never_delivers_is_reported(self):
        """The state no per-fence number could express."""
        with self.assertLogs(W.__name__, level="ERROR") as cm:
            for _ in range(STORE_NEVER_DELIVERED_AFTER):
                W._observe_store_delivery(_report(0))
        joined = "\n".join(cm.output)
        self.assertIn("#871a", joined)
        self.assertIn("STORE NEVER DELIVERED", joined)

    def test_a_delivering_store_is_never_alarmed(self):
        """One acknowledged byte, ever, is proof -- and it must silence this."""
        with self.assertNoLogs(W.__name__, level="ERROR"):
            W._observe_store_delivery(_report(1))
            for _ in range(STORE_NEVER_DELIVERED_AFTER * 3):
                W._observe_store_delivery(_report(0))
        fences, acked = store_delivery_counters()
        self.assertEqual(acked, 1)
        self.assertGreater(fences, STORE_NEVER_DELIVERED_AFTER)

    def test_delivery_later_still_counts(self):
        """Proof arriving after some quiet fences still counts as proof."""
        with self.assertNoLogs(W.__name__, level="ERROR"):
            for _ in range(STORE_NEVER_DELIVERED_AFTER - 1):
                W._observe_store_delivery(_report(0))
            W._observe_store_delivery(_report(3))
            W._observe_store_delivery(_report(0))
        self.assertEqual(store_delivery_counters()[1], 3)

    def test_the_threshold_is_not_one(self):
        """An idle instance's first quiet fences are LEGITIMATE.

        A threshold of 1 would fire on every boot before the first request and
        be muted within a day, which is precisely how the silent condition it
        replaces stayed invisible.
        """
        self.assertGreater(STORE_NEVER_DELIVERED_AFTER, 2)
        with self.assertNoLogs(W.__name__, level="ERROR"):
            for _ in range(STORE_NEVER_DELIVERED_AFTER - 1):
                W._observe_store_delivery(_report(0))

    def test_it_fires_once_not_on_every_fence_afterwards(self):
        """A line per cutover would bury the log it exists to inform."""
        with self.assertLogs(W.__name__, level="ERROR") as cm:
            for _ in range(STORE_NEVER_DELIVERED_AFTER * 4):
                W._observe_store_delivery(_report(0))
        self.assertEqual(len(cm.output), 1, cm.output)

    def test_the_zero_is_worded_as_evidence_not_as_proof(self):
        """The counter is a LOWER BOUND: it sees only fence-drained acks.

        A nonzero value proves delivery. A zero does not prove absence, and the
        message must not claim it does -- overstating an instrument is how the
        next reader stops believing it.
        """
        with self.assertLogs(W.__name__, level="ERROR") as cm:
            for _ in range(STORE_NEVER_DELIVERED_AFTER):
                W._observe_store_delivery(_report(0))
        msg = "\n".join(cm.output)
        self.assertIn("If this instance has served any request", msg)

    def test_the_fence_feeds_the_counter(self):
        """Wired to the real fence, not only callable in isolation."""
        import inspect

        src = inspect.getsource(W.flip_writeback)
        self.assertIn("_observe_store_delivery(report)", src)


if __name__ == "__main__":
    unittest.main()
