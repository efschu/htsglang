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

