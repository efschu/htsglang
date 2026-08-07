"""Unit test for HiRadixCache._drain_async_work PP-sync backpressure."""

import types
import unittest

from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _FakeWork:
    def __init__(self):
        self.waited = False

    def is_completed(self) -> bool:
        # #630: the drain now polls before it waits, so the stub has to answer
        # the same question a real torch.distributed.Work does.
        return True

    def wait(self):
        self.waited = True


class _Holder:
    """Minimal carrier exposing only what _drain_async_work touches."""


class TestPPSyncDrain(unittest.TestCase):
    def _holders(self):
        """One holder per cache class, carrying the bounded-wait collaborator.

        Since #630 the drain routes each work through the class's own
        ``_wait_bounded``, so the carrier has to supply it plus the deadline
        and the pp_rank that the timeout label names.
        """
        for cls in (HiRadixCache, UnifiedRadixCache):
            holder = _Holder()
            holder.pp_rank = 0
            holder.pp_size = 2
            holder.collective_timeout_s = 5.0
            holder._wait_bounded = types.MethodType(cls._wait_bounded, holder)
            yield holder, types.MethodType(cls._drain_async_work, holder)

    def test_drain_waits_all_and_clears(self):
        for holder, drain in self._holders():
            works = [_FakeWork(), _FakeWork(), _FakeWork()]
            holder.work_list = list(works)

            drain()

            self.assertTrue(all(w.waited for w in works))
            self.assertEqual(holder.work_list, [])

    def test_drain_empty_is_noop(self):
        for holder, drain in self._holders():
            holder.work_list = []

            drain()

            self.assertEqual(holder.work_list, [])


if __name__ == "__main__":
    unittest.main()
