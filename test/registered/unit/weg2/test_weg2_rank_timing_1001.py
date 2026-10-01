"""RANK-TIMING (user 01.10. ~08:45Z, dashboard expert view): PLE latency and the
L2 / L3 fetch-back times as rankstats fields, per rank (P and D are separate
processes), stage names as in request_done.cached (device, host = L2,
storage = L3, l15).

BUG (coordinator 01.10.): rankstats ``cache.prefetch.landed`` was 0 on every
boot while ``issued`` > 0. It read ``PREFETCH_GATE_COUNTS["landed"]``, which
only the #1068 deferral increments (a DEFERRED prefetch that registered on a
later pass) -- not a store read landing. Now ``landed`` counts the store
reads that landed >= 1 page (the aux IO thread, at the read's end); the
deferral count stays as ``deferred_landed``.

RED on 7b8a2a41a1: weg2/rank_timing.py and cache_controller._note_l3_read do
not exist; ``landed`` stays 0 after a landed read.
"""
from __future__ import annotations

import json
import os
import time
import unittest
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers import cache_controller as cc  # noqa: E402
from sglang.srt.mem_cache import match_refusal_census as mrc  # noqa: E402
from sglang.srt.weg2 import rank_timing as rt  # noqa: E402
from sglang.srt.weg2 import rankstats  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402
from sglang.test.test_utils import CustomTestCase  # noqa: E402

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _sched():
    return SimpleNamespace(tree_cache=None, _weg2_store_short_seen=0)


class TestPrefetchLanded(CustomTestCase):
    def setUp(self):
        rt.reset()
        self._counts = dict(mrc.PREFETCH_GATE_COUNTS)
        mrc.PREFETCH_GATE_COUNTS.clear()
        mrc.PREFETCH_GATE_COUNTS.update({"attempted": 3, "issued": 2})

    def tearDown(self):
        mrc.PREFETCH_GATE_COUNTS.clear()
        mrc.PREFETCH_GATE_COUNTS.update(self._counts)
        rt.reset()

    def test_a_landed_store_read_counts_as_landed(self):
        ctrl = SimpleNamespace(page_size=64, mem_pool_host=SimpleNamespace(size_per_token=2048))
        t0 = time.monotonic()
        op = SimpleNamespace(completed_tokens=640, start_time=t0 - 0.5,
                             read_start_time=t0 - 0.2, read_end_time=t0)
        cc._note_l3_read(ctrl, op)                         # the aux thread, at the read's end
        empty = SimpleNamespace(completed_tokens=0, start_time=t0, read_start_time=t0, read_end_time=t0)
        cc._note_l3_read(ctrl, empty)                      # a read that got nothing
        pf = rankstats._cache_block(_sched())["prefetch"]
        self.assertEqual((pf["issued"], pf["landed"], pf["landed_pages"], pf["empty"], pf["reads"]),
                         (2, 1, 10, 1, 2))
        self.assertEqual(pf["deferred_landed"], 0)
        self.assertAlmostEqual(pf["ms_max"], 500.0, delta=1.0)  # issue -> read end
        self.assertEqual(pf["bytes"], 640 * 2048)

    def test_l3_block(self):
        ctrl = SimpleNamespace(page_size=64, mem_pool_host=SimpleNamespace(size_per_token=1000))
        op = SimpleNamespace(completed_tokens=128, start_time=10.0, read_start_time=10.1, read_end_time=10.35)
        cc._note_l3_read(ctrl, op)
        l3 = rankstats._cache_block(_sched())["l3"]
        self.assertEqual((l3["read_n"], l3["read_pages"], l3["read_bytes"]), (1, 2, 128000))
        self.assertAlmostEqual(l3["read_ms_sum"], 250.0, delta=0.01)
        self.assertEqual((l3["last"]["pages"], len(l3["recent"])), (2, 1))

    def test_an_instrument_never_breaks_the_read(self):
        cc._note_l3_read(SimpleNamespace(), SimpleNamespace())   # nothing raises
        self.assertNotIn("l3", rankstats._cache_block(_sched()))


class TestLoadbackAndPle(CustomTestCase):
    def setUp(self):
        rt.reset()

    def tearDown(self):
        rt.reset()

    def test_loadback_fields(self):
        self.assertNotIn("loadback_ms_sum", rankstats._cache_block(_sched()))   # none yet: no key
        rt.note_loadback(12.5, pages=4, nbytes=4 << 20, t=100.0)
        rt.note_loadback(30.0, pages=8, nbytes=8 << 20, t=101.0)
        c = rankstats._cache_block(_sched())
        self.assertEqual((c["loadback_ms_sum"], c["loadback_ms_max"], c["loadback_pages"],
                          c["loadback_bytes"], c["loadback_count"]), (42.5, 30.0, 12, 12 << 20, 2))
        self.assertEqual(c["loadback_last"], {"t": 101.0, "ms": 30.0, "pages": 8, "bytes": 8 << 20})
        self.assertEqual(c["loadback_recent"][0], [100.0, 12.5, 4, 4 << 20])

    def test_ple_per_phase_and_recent(self):
        self.assertNotIn("ple", rankstats.scheduler_counters(_sched()))   # no PLE gathered: no key
        rt.note_ple("prefill", 41.0, hit=3000, miss=1096, wait_ms=2.0, nbytes=1096 * 320, t=5.0)
        rt.note_ple("decode", 0.4, hit=60, miss=4, wait_ms=0.4, nbytes=60 * 320, t=6.0)
        rt.note_ple("decode", 1.2, hit=64, miss=0, wait_ms=1.2, nbytes=64 * 320, t=7.0)
        ple = rankstats.scheduler_counters(_sched())["ple"]
        self.assertEqual((ple["prefill"]["n"], ple["prefill"]["hit_n"], ple["prefill"]["miss_n"],
                          ple["prefill"]["bytes"]), (1, 3000, 1096, 1096 * 320))
        d = ple["decode"]
        self.assertEqual((d["n"], d["ms_sum"], d["ms_max"], d["last_ms"], d["last_t"], d["hit_n"], d["miss_n"]),
                         (2, 1.6, 1.2, 1.2, 7.0, 124, 4))
        self.assertEqual(ple["recent"][-1], [7.0, 1.2, 64, 0, "decode"])
        json.dumps(ple)

    def test_recent_is_bounded(self):
        for i in range(rt.RECENT_KEEP + 10):
            rt.note_ple("decode", 1.0, t=float(i))
        self.assertEqual(len(rt.ple_block()["recent"]), rt.RECENT_KEEP)

    def test_l15_only_once_written_and_schema_checked(self):
        self.assertNotIn("l15", rankstats._cache_block(_sched()))
        rt.note_l15(kv_pages=100, anchors=3, bytes=1 << 30)
        rt.note_l15(hit_n=1, hit_tokens=4096, p2p_read_n=1, p2p_read_ms_sum=12.0, p2p_bytes=1 << 20)
        rt.note_l15(hit_n=1, hit_tokens=1024)
        l15 = rankstats._cache_block(_sched())["l15"]
        self.assertEqual((l15["kv_pages"], l15["hit_n"], l15["hit_tokens"], l15["evict_n"]), (100, 2, 5120, 0))
        with self.assertRaises(KeyError):
            rt.note_l15(bogus=1)


if __name__ == "__main__":
    unittest.main()
