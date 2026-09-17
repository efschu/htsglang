# SPDX-License-Identifier: Apache-2.0
"""#1459: the shared L3 stem index (arena.c l3idx_*) -- add/has/remove/clear
across two mappings, capacity refusal, env off; and the wiring: the evictor
keeps it (commit / evict / clear), _stat_stems asks it first, the front
queues one request ahead on P.  Hermetic (gcc build of arena.c, tmp dir)."""
import inspect
import os
import tempfile
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.mem_cache.storage.file import l3_index as l3
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class Table(CustomTestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="l3idx-")
        self.path = os.path.join(self.tmp, "l3idx.bin")

    def test_add_has_remove_clear_across_two_mappings(self):
        a = l3.L3Index(self.path, cap=1024)
        self.assertTrue(a.created)
        stems = [f"page-{i}.kv" for i in range(300)]
        self.assertEqual(a.add(stems), 300)
        self.assertEqual(a.add(stems[:10]), 0)          # already present
        self.assertEqual(a.count(), 300)
        b = l3.L3Index(self.path, cap=1024)             # a second process
        self.assertFalse(b.created)
        self.assertEqual(b.has(stems[:5] + ["absent-1", "absent-2"]), [True] * 5 + [False, False])
        self.assertEqual(b.remove(["page-7.kv", "absent-1"]), 1)
        self.assertEqual(a.has(["page-7.kv", "page-8.kv"]), [False, True])
        self.assertEqual(a.count(), 299)
        a.clear()
        self.assertEqual(b.count(), 0)
        self.assertEqual(b.has(["page-8.kv"]), [False])
        a.close(); b.close()

    def test_full_table_refuses_and_stat_falls_back(self):
        a = l3.L3Index(self.path, cap=1024)
        self.assertEqual(a.add([f"s{i}" for i in range(512)]), 512)    # 50 % load reached
        self.assertEqual(a.add(["one-more"]), 0)                        # refused (-2 -> 0), logged once
        self.assertEqual(a.has(["one-more"]), [False])
        a.close()

    def test_bad_cap_and_env_off(self):
        with self.assertRaises(ValueError):
            l3.L3Index(self.path, cap=1000)
        os.environ["SGLANG_HICACHE_L3_INDEX"] = "0"
        try:
            self.assertIsNone(l3.open_index(self.path))
        finally:
            os.environ.pop("SGLANG_HICACHE_L3_INDEX", None)
        self.assertIsNone(l3.open_index(None))


class Wiring(CustomTestCase):
    def test_store_evictor_and_front(self):
        from sglang.srt.mem_cache import hicache_storage as hs
        from sglang.srt.mem_cache.storage.file import lru_file_evictor as ev
        from sglang.srt.weg2 import front
        stat = inspect.getsource(hs.HiCacheFile._stat_stems)
        self.assertIn("_idx = self._l3_index()", stat)
        self.assertLess(stat.index("_idx.has(stems)"), stat.index("pio.stat_sizes"))
        self.assertIn("_idx.clear()  # #1459", inspect.getsource(hs.HiCacheFile.clear))
        evs = inspect.getsource(ev.LRUFileEvictor)
        self.assertIn("_idx.add([suffixed_key])", inspect.getsource(ev.LRUFileEvictor.commit))
        self.assertIn("_idx.remove([evict_stem])  # #1459", evs)
        fsrc = inspect.getsource(front)
        self.assertIn('os.environ.get("SGLANG_WEG2_P_QUEUE_AHEAD", "1")', fsrc)
        self.assertIn("asyncio.Semaphore(self.p_concurrency + _ahead)", fsrc)
        # #1459c: the pool form, not paired batches
        self.assertIn("passes = await _p_drain_pool(", fsrc)
        self.assertNotIn("range(min(self.p_concurrency + _ahead, len(self.queue)))", fsrc)


class Test1459cDrainPool(unittest.TestCase):
    """#1459c: the leg-1 pool refills the moment ONE call finishes."""

    def _run(self, durations, limit, stop_after=None):
        import asyncio
        import collections
        from sglang.srt.weg2 import front

        async def main():
            q = collections.deque(range(len(durations)))
            t0 = loop.time()
            starts, ends, done_order, live = {}, {}, [], [0]
            peak = [0]
            dispatch_ok = [True]

            async def one(i):
                starts[i] = loop.time() - t0
                live[0] += 1
                peak[0] = max(peak[0], live[0])
                await asyncio.sleep(durations[i])
                live[0] -= 1
                ends[i] = loop.time() - t0
                if stop_after is not None and i == stop_after:
                    dispatch_ok[0] = False
                return i

            rounds = await front._p_drain_pool(q, limit, one, done_order.append, lambda: dispatch_ok[0])
            return starts, ends, done_order, peak[0], rounds, list(q)

        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(main())
        finally:
            loop.close()

    def test_third_item_starts_when_first_finishes_not_when_pair_finishes(self):
        # item 0 short, item 1 long: with paired batches item 2 would wait
        # for item 1; the pool starts it right after item 0.
        starts, ends, order, peak, rounds, left = self._run([0.05, 0.30, 0.05, 0.05], limit=2)
        self.assertLessEqual(peak, 2)
        self.assertLess(starts[2], ends[1])
        self.assertAlmostEqual(starts[2], ends[0], delta=0.03)
        self.assertEqual(order, [0, 2, 3, 1])
        self.assertEqual(left, [])
        self.assertGreaterEqual(rounds, 2)

    def test_stop_dispatch_drains_inflight_and_keeps_the_rest_queued(self):
        starts, ends, order, peak, rounds, left = self._run([0.05, 0.05, 0.05, 0.05], limit=2, stop_after=0)
        # item 0 finishing flips may_dispatch False: item 1 (already in
        # flight) is awaited, items 2/3 stay in the queue.
        self.assertEqual(sorted(order), [0, 1])
        self.assertEqual(left, [2, 3])

    def test_empty_queue_returns_zero_rounds(self):
        starts, ends, order, peak, rounds, left = self._run([], limit=2)
        self.assertEqual((order, rounds, left), ([], 0, []))


if __name__ == "__main__":
    unittest.main()
