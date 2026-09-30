# SPDX-License-Identifier: Apache-2.0
"""DUAL-TP3PP3 unified KV (C): group D's KV actor.

DANGER DIRECTIONS guarded here:
* off by default (born passes tensors through, pool rows unchanged);
* the decision is replicated and pure: grow at once, shrink only with two
  steps of slack after the hold -- or at once when a P prompt waits (D's idle
  cache must not starve P);
* a group grow is all-or-none over the D ranks (MIN); a short card leaves
  pressure on P and nothing mapped;
* a shrink never goes below the group's highest live page.
"""
from __future__ import annotations

import os
import tempfile
import unittest.mock as mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch

from sglang.srt.weg2 import card_kv_ledger as K
from sglang.srt.weg2 import dual_d_kv_stage as D
from sglang.srt.weg2 import dual_p_kv_stage as P
from sglang.srt.weg2.d_seat_vram import AllocInfo
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

G = 2 << 20
ALLOC = (65536 + 64) * 2048


class FakeSpans:
    available = True

    def info(self, ptr):
        return AllocInfo(size=ALLOC, mapped=0, planned=0, active=True)

    def set_spans(self, ptr, spans, now):
        return 0


def _actor(ledger, gmin):
    geom = P._geom_for(torch.zeros(65536 + 64, 512), 65536, 64, "k", ALLOC)
    return D.DKvStage([(1, geom)], ledger, allocator=object(), pools=[], page_size=64, granule=G,
                      top_tokens=65536, spans=FakeSpans(), step=4096, engage_cap=lambda *a: None,
                      gmin=gmin)


class DualDKvStage(CustomTestCase):
    def test_off_by_default(self):
        t = torch.zeros(4)
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIs(D.born(object(), t, "k", boot_tokens=0, spans=FakeSpans(), granule=G), t)
            self.assertEqual(D.pool_tokens(1000), 1000)

    def test_decide(self):
        self.assertEqual(D.decide(8192, 12288, False, 0, 4096), ("grow", 12288))
        self.assertEqual(D.decide(16384, 8192, False, 10, 4096), ("hold", 16384))     # hold not reached
        self.assertEqual(D.decide(16384, 8192, False, 64, 4096), ("shrink", 8192))
        self.assertEqual(D.decide(16384, 12288, True, 0, 4096), ("shrink", 12288))    # P waits: at once
        self.assertEqual(D.decide(16384, 16384, True, 0, 4096), ("hold", 16384))

    def test_group_grow_all_or_none_and_pressure_on_p(self):
        path = os.path.join(tempfile.mkdtemp(prefix="wkvd"), "card")
        led_d, led_p = K.CardKvLedger(path, "D"), K.CardKvLedger(path, "P")
        a = _actor(led_d, gmin=lambda v: v)
        base = a.bytes_for(0)
        led_d.contribute(a.bytes_for(8192) - base, committed=a.bytes_for(8192) - base)
        led_p.contribute(a.bytes_for(8192) - base)                  # P's boot KV joins the pool
        a.mapped_tokens, a._committed = 8192, a.bytes_for(8192) - base
        led_p.request(a.bytes_for(4096) - base)                     # P prefills something
        self.assertFalse(a.group_grow(16384))                       # short: waits, presses P
        self.assertEqual(a.mapped_tokens, 8192)
        self.assertGreater(K.peek(path).pressure["P"], 0)
        led_p.release(a.bytes_for(4096) - base)                     # P paused and released
        self.assertTrue(a.group_grow(16384))
        self.assertEqual(a.mapped_tokens, 16384)
        other_rank_short = _actor(led_d, gmin=lambda v: [0])        # another D rank's card was short
        other_rank_short.mapped_tokens = 16384
        self.assertFalse(other_rank_short.group_grow(20480))
        self.assertEqual(other_rank_short.mapped_tokens, 16384)

    def test_shrink_respects_live_floor(self):
        path = os.path.join(tempfile.mkdtemp(prefix="wkvd"), "card")
        led = K.CardKvLedger(path, "D")
        a = _actor(led, gmin=lambda v: v)
        base = a.bytes_for(0)
        led.contribute(a.bytes_for(32768) - base, committed=a.bytes_for(32768) - base)
        a.mapped_tokens, a._committed = 32768, a.bytes_for(32768) - base
        n = a.group_shrink(4096, live_floor_tokens=20000)           # a live page at 20000
        self.assertEqual(a.mapped_tokens, 20480)
        self.assertEqual(n, a.bytes_for(32768) - a.bytes_for(20480))
        self.assertEqual(K.peek(path).committed["D"], a.bytes_for(20480) - base)

    def test_hooks_and_launcher(self):
        import inspect

        from sglang.srt.managers import scheduler as SC
        from sglang.srt.mem_cache import memory_pool as MP
        from sglang.srt.model_executor import model_runner_kv_cache_mixin as MX
        from sglang.srt.weg2 import launcher as L

        self.assertIn("_ddk.born(pool, _dpk.born(pool, t, name), name)", inspect.getsource(MP._kv_stage_born))
        src = inspect.getsource(MX)
        self.assertIn("_ddk.attach(self)", src)
        self.assertIn("max_tokens = _ddk.pool_tokens(max_tokens)", src)
        self.assertIn("_ddk.tick(self)", inspect.getsource(SC))
        ns = L.build_parser().parse_args(["--tree", "/x", "--tag", "t", "--dual-share", "--dual-unified-kv", "on"])
        L.resolve_dual_layout(ns)
        self.assertEqual(L.dual_share_env(ns, "D")[D.MAX_TOKENS_ENV], "1048576")
        self.assertNotIn(D.MAX_TOKENS_ENV, L.dual_share_env(ns, "P"))

    def test_draft_pool_untouched(self):
        t = torch.zeros(4, 2)
        pool = type("Pool", (), {"size": 1000, "page_size": 64})()
        env = {"SGLANG_WEG2_DUAL_LAYOUT": "1", "SGLANG_WEG2_GROUP": "D", D.MAX_TOKENS_ENV: "65536"}
        with mock.patch.dict(os.environ, env):
            D._BOOT_TOKENS = None
            self.assertIs(D.born(pool, t, "k", spans=FakeSpans(), granule=G), t)   # no boot level set
            D._BOOT_TOKENS = 8192
            self.assertIs(D.born(pool, t, "k", spans=FakeSpans(), granule=G), t)   # not a top-sized pool
            D._BOOT_TOKENS = None

    def test_shrink_on_the_token_allocator_never_unmaps_a_live_row(self):
        # order point 3: rows above the shrunk span must not stay occupied
        from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator

        alloc = TokenToKVPoolAllocator(200, torch.float16, "cpu", None, False)
        path = os.path.join(tempfile.mkdtemp(prefix="wkvd"), "card")
        led = K.CardKvLedger(path, "D")
        geom = P._geom_for(torch.zeros(264, 8), 256, 1, "k", 264 * 32)
        a = D.DKvStage([(1, geom)], led, allocator=alloc, pools=[], page_size=1, granule=32, top_tokens=192,
                       spans=FakeSpans(), step=16, gmin=lambda v: v)
        led.contribute(a.bytes_for(96) - a.bytes_for(0), committed=a.bytes_for(96) - a.bytes_for(0))
        a.mapped_tokens, a._committed = 96, a.bytes_for(96) - a.bytes_for(0)
        a._engage_cap(alloc, 96, 1)
        ids = alloc.alloc(80)                               # ids 1..80 live
        self.assertEqual(P.max_live_id(alloc, 1), 80)
        self.assertGreater(a.group_shrink(16, live_floor_tokens=80), 0)  # shrinks only down to the live floor
        self.assertEqual(a.mapped_tokens, 80)
        with self.assertRaises(P.Weg2DualKvCapBreach):                 # a floor that lies is a named stop
            a.group_shrink(16, live_floor_tokens=0)
        alloc.free(ids)
        self.assertGreater(a.group_shrink(16, live_floor_tokens=0), 0)
        self.assertEqual(a.mapped_tokens, 16)
        P.check_cap(alloc, 16, 1, "test")                   # no free id above the new end
        self.assertLessEqual(int(alloc.alloc(10).max()), 17)



class _Req:
    def __init__(self, rid, n):
        self.rid, self.origin_input_ids, self.output_ids = rid, list(range(n)), []


class _Sched:
    def __init__(self, actor, running, waiting, group):
        import types

        self.running_batch = types.SimpleNamespace(reqs=running)
        self.chunked_req = None
        self.waiting_queue = waiting
        self.server_args = types.SimpleNamespace(chunked_prefill_size=4096, speculative_num_draft_tokens=8)
        self.tp_worker = types.SimpleNamespace(model_runner=types.SimpleNamespace(dual_d_kv=actor))
        self._weg2_group_min_ints = group


class DemandReplayOfBsffsv(CustomTestCase):
    """Metal bsffsv (...09301000): D grew to 86016 for weg2-0-6, its store read
    landed short (X-DEFER prefetch_pending), D shrank while the request still
    waited, the ranks shrank apart (TP0 kept 45056, TP1/TP2 40960) and TP2's
    re-issued loadback never found rows (queue=36863 on TP2 only)."""

    def _actor(self, mapped):
        path = os.path.join(tempfile.mkdtemp(prefix="wkvb"), "card")
        led = K.CardKvLedger(path, "D")
        geom = P._geom_for(torch.zeros(131072 + 64, 512), 131072, 64, "k", (131072 + 64) * 2048)
        a = D.DKvStage([(1, geom)], led, allocator=object(), pools=[], page_size=64, granule=G,
                       top_tokens=131072, spans=FakeSpans(), step=4096, engage_cap=lambda *a: None,
                       gmin=lambda v: v)
        b = a.bytes_for(mapped) - a.bytes_for(0)
        led.contribute(b, committed=b)
        a.mapped_tokens, a._committed = mapped, b
        return a

    def test_waiting_deferred_request_holds_the_level(self):
        a = self._actor(86016)
        deferred = _Req("weg2-0-6", 40767)                       # X-DEFER: stays in the waiting queue
        small = _Req("weg2-0-9", 72)
        sched = _Sched(a, running=[], waiting=[small, deferred], group=lambda v: v)
        with mock.patch.object(D._pk, "max_live_id", lambda *x: 0):
            for _ in range(200):                                # far past the hold
                D.tick(sched)
        self.assertGreaterEqual(a.mapped_tokens, 40767 + 72, "D shrank below its waiting requests")

    def test_ranks_decide_on_the_group_demand(self):
        # rank B has not seen the deferred request yet (one iteration late): both must hold alike
        seen = {}

        def group(vals):                                        # the other rank reports the big demand
            seen["vals"] = list(vals)
            if len(vals) == 3:                                  # [-want, -p_wait, -live]: MAX via MIN of negatives
                return [min(vals[0], -(40767 + 72 + 8192)), vals[1], vals[2]]
            return list(vals)

        a = self._actor(86016)
        sched = _Sched(a, running=[], waiting=[], group=group)
        with mock.patch.object(D._pk, "max_live_id", lambda *x: 0):
            for _ in range(200):
                D.tick(sched)
        self.assertGreaterEqual(a.mapped_tokens, 40767, "a rank shrank on its local view")
