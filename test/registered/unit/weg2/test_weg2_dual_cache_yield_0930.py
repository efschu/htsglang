# SPDX-License-Identifier: Apache-2.0
"""DUAL-TP3PP3 metal replay dual13 (3q33cu, ...09301054 @b3319f3baf).

From 11:00:04Z everything stalled. PP0 logged
"P-KV PP0 WAIT rid=weg2-0-23/24/25 tokens=62343" many times a second, about
497k lines. D showed token usage 1.00 with 0 running and 0 queued.

The 5090 card ledger in bytes:
* budget: D 2415919104 + P 2122317824 = 4538236928.
* P committed 2952790016. Two grants (weg2-0-21 and weg2-0-22, 10:59:20) sat
  on ONE 65536-token mapping, and each counted as its own span.
* D held 335544320. That is its cache: 65536 mapped, 0 running, 0 queued.
* need 1476395008, free 1249902592. The grant is short and never retried
  successfully.

DANGER DIRECTIONS guarded here:
* a second grant on the same mapping is not a second span. Committed returns
  to the mapped level, and the replayed third grant is taken;
* D's cached prefix is not demand. P waits and D has no running or waiting
  request, so every D rank evicts its cache (L2 keeps the backed prefix) and
  the next tick shrinks. The cache is never taken from under a live seat;
* at most 1 WAIT line per second per rid, with backoff, plus a census line.
  The log must not keep a stalled boot "alive" for the silence watchdog.
"""
from __future__ import annotations

import os
import tempfile
import unittest.mock as mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch

from sglang.srt.weg2 import card_kv_ledger as K
from sglang.srt.weg2 import dual_d_kv_stage as D
from sglang.srt.weg2 import dual_p_kv_stage as S
from sglang.srt.weg2.d_seat_vram import AllocInfo
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

G = 2 << 20
ROW = 2048
ALLOC = (16384 + 64) * ROW
MIB = 1 << 20


class FakeSpans:
    available = True

    def info(self, ptr):
        return AllocInfo(size=ALLOC, mapped=0, planned=0, active=True)

    def set_spans(self, ptr, spans, now):
        return 0


def _p_stage(led):
    return S.PKvStage([(1, S._geom_for(torch.zeros(16448, 512), 16384, 64, "k", ALLOC))], led,
                      allocator=object(), pools=[], page_size=64, granule=G, top_tokens=16384,
                      spans=FakeSpans(), engage_cap=lambda *a: None)


class MetalReplayOf3q33cuP(CustomTestCase):
    """Scaled replay (unit u = one 12288-token grant = 24 MiB). The metal ratios
    are kept: budget/u = 3.07, D/u = 0.23. That makes 2u + d <= budget < 3u + d:
    the third grant fits only when the second grant was not counted twice."""

    def test_a_second_grant_on_the_same_mapping_is_not_a_second_span(self):
        path = os.path.join(tempfile.mkdtemp(prefix="wkvy"), "card")
        d = K.CardKvLedger(path, "D")
        d.contribute(40 * MIB, committed=40 * MIB)
        d.release(35 * MIB)                                     # D keeps 5 MiB (its cache on metal)
        led = K.CardKvLedger(path, "P")
        led.contribute(34 * MIB)                                # budget 74 MiB
        st = _p_stage(led)
        u = st.bytes_for(12288) - st.bytes_for(0)
        self.assertEqual(u, 24 * MIB)
        stages = [{"ledger": path, "step": 4096, "top": 16384, "bytes": st.table()}]
        open_p = lambda pth: K.CardKvLedger(pth, "P")
        st.map_granted(S.group_grant(stages, 12000, open_p))    # weg2-0-22
        st.map_granted(S.group_grant(stages, 12000, open_p))    # weg2-0-21, same second, same mapping
        self.assertEqual(st.mapped_tokens, 12288)
        self.assertEqual(K.peek(path).committed["P"], u, "the second grant stayed counted as a second span")
        self.assertEqual(S.group_grant(stages, 12000, open_p), 12288, "weg2-0-23 starves on P's own count")

    def test_the_mapping_stays_covered(self):
        path = os.path.join(tempfile.mkdtemp(prefix="wkvy"), "card")
        led = K.CardKvLedger(path, "P")
        led.contribute(100 * MIB)
        st = _p_stage(led)
        stages = [{"ledger": path, "step": 4096, "top": 16384, "bytes": st.table()}]
        open_p = lambda pth: K.CardKvLedger(pth, "P")
        st.map_granted(S.group_grant(stages, 3000, open_p))     # 4096
        st.map_granted(S.group_grant(stages, 12000, open_p))    # 12288: grows
        st.map_granted(S.group_grant(stages, 3000, open_p))     # smaller: the mapping never shrinks
        self.assertEqual(st.mapped_tokens, 12288)
        self.assertEqual(K.peek(path).committed["P"], st.bytes_for(12288) - st.bytes_for(0))
        self.assertEqual(st.release_all(), st.bytes_for(12288) - st.bytes_for(0))
        self.assertEqual(K.peek(path).committed["P"], 0)


class _Req:
    def __init__(self, rid, n):
        self.rid, self.origin_input_ids, self.output_ids = rid, list(range(n)), []


class FakeTree:
    """D's device radix: ``cached`` ids are unlocked cache, ``locked`` belong to
    a live seat. evict frees unlocked ids only (backed prefixes stay in L2)."""

    def __init__(self, alloc, cached, locked=None):
        self.alloc, self.cached, self.locked = alloc, cached, locked
        self.evicted = 0

    def evictable_size(self):
        return 0 if self.cached is None else int(self.cached.numel())

    def evict(self, params):
        if self.cached is not None:
            self.alloc.free(self.cached)
            self.evicted += int(self.cached.numel())
            self.cached = None


class _Sched:
    def __init__(self, actor, tree, running=(), waiting=()):
        import types

        self.running_batch = types.SimpleNamespace(reqs=list(running))
        self.chunked_req = None
        self.waiting_queue = list(waiting)
        self.tree_cache = tree
        self.server_args = types.SimpleNamespace(chunked_prefill_size=16, speculative_num_draft_tokens=1)
        self.tp_worker = types.SimpleNamespace(model_runner=types.SimpleNamespace(dual_d_kv=actor))
        self._weg2_group_min_ints = lambda v: list(v)


class MetalReplayOf3q33cuD(CustomTestCase):
    def _d(self, mapped=96):
        from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator

        alloc = TokenToKVPoolAllocator(200, torch.float16, "cpu", None, False)
        path = os.path.join(tempfile.mkdtemp(prefix="wkvyd"), "card")
        led = K.CardKvLedger(path, "D")
        geom = S._geom_for(torch.zeros(264, 8), 256, 1, "k", 264 * 32)
        a = D.DKvStage([(1, geom)], led, allocator=alloc, pools=[], page_size=1, granule=32, top_tokens=192,
                       spans=FakeSpans(), step=16, gmin=lambda v: v)
        b = a.bytes_for(mapped) - a.bytes_for(0)
        led.contribute(b, committed=b)
        a.mapped_tokens, a._committed = mapped, b
        a._engage_cap(alloc, mapped, 1)
        p = K.CardKvLedger(path, "P")
        p.contribute(0)
        return a, alloc, path, p

    def test_cache_without_demand_yields_to_a_waiting_p(self):
        a, alloc, path, p = self._d()
        tree = FakeTree(alloc, alloc.alloc(90))                 # D full of cache, 0 running, 0 queued
        p.request(1 << 30)                                      # P's grant is short -> demand["P"] > 0
        self.assertGreater(K.peek(path).demand["P"], 0)
        sched = _Sched(a, tree)
        for _ in range(4):
            D.tick(sched)
        self.assertEqual(tree.evicted, 90, "D kept a cache nobody runs while P waits")
        self.assertLessEqual(a.mapped_tokens, 32, "D did not shrink after giving up its cache")
        self.assertLess(K.peek(path).committed["D"], a.bytes_for(96) - a.bytes_for(0))

    def test_no_yield_when_p_does_not_wait(self):
        a, alloc, path, _p = self._d()
        tree = FakeTree(alloc, alloc.alloc(90))
        sched = _Sched(a, tree)
        for _ in range(4):
            D.tick(sched)
        self.assertEqual(tree.evicted, 0)                       # D's cache is its own while nobody asks

    def test_no_yield_under_a_live_seat(self):
        a, alloc, path, p = self._d()
        tree = FakeTree(alloc, alloc.alloc(40))
        p.request(1 << 30)
        sched = _Sched(a, tree, running=[_Req("seat", 30)])
        for _ in range(4):
            D.tick(sched)
        self.assertEqual(tree.evicted, 0, "the cache went from under a running D request")

    def test_the_decision_is_the_groups(self):
        # this rank sees no demand, another rank still has a waiting request -> nobody evicts
        a, alloc, path, p = self._d()
        tree = FakeTree(alloc, alloc.alloc(90))
        p.request(1 << 30)
        sched = _Sched(a, tree)
        sched._weg2_group_min_ints = lambda v: [min(x, -40) for x in v]
        for _ in range(4):
            D.tick(sched)
        self.assertEqual(tree.evicted, 0)


class WaitLogRate(CustomTestCase):
    def test_at_most_one_line_per_second_per_rid_with_backoff_and_a_census(self):
        S._reset_wait_log()
        clock = [1000.0]
        lines = []
        with mock.patch.object(S, "_now", lambda: clock[0]), \
                mock.patch.object(S.logger, "info", lambda msg, *a: lines.append(msg % a)):
            for i in range(20000):                             # 200 s at 100 retries per second
                clock[0] = 1000.0 + i * 0.01
                for rid in ("weg2-0-23", "weg2-0-24", "weg2-0-25"):
                    S._log_wait(rid, 62343)
        per_rid = [l for l in lines if "PP0 WAIT rid=weg2-0-23" in l]
        census = [l for l in lines if "PP0 WAIT census" in l]
        self.assertLessEqual(len(per_rid), 9, per_rid)         # 0,1,3,7,15,31,63,127 s (+slack)
        self.assertGreaterEqual(len(per_rid), 5)
        self.assertTrue(census)
        self.assertLessEqual(len(census), 9)
        self.assertIn("waits=", census[-1])
        # a grant resets the rid: the next WAIT of the same rid logs at once
        S._wait_granted("weg2-0-23")
        n = len(lines)
        with mock.patch.object(S, "_now", lambda: clock[0]), \
                mock.patch.object(S.logger, "info", lambda msg, *a: lines.append(msg % a)):
            S._log_wait("weg2-0-23", 62343)
        self.assertEqual(len(lines), n + 1)


if __name__ == "__main__":
    import unittest

    unittest.main()
