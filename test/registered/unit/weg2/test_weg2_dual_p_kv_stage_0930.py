# SPDX-License-Identifier: Apache-2.0
"""DUAL-TP3PP3 unified KV (B): group P's KV actor.

DANGER DIRECTIONS guarded here:
* off unless dual layout + group P + a max token count: born() leaves every
  tensor untouched (default path byte-identical);
* born trims to 0 tokens on the fixed lattice and registers the tensor;
* ensure() maps only what the card ledger granted -- a short grant maps
  nothing and returns the partial grant (P waits, never presses D);
* grow keeps the lattice cuts (a live move never remaps a kept extent);
* release_all() caps the allocator to 0 BEFORE unmapping and returns exactly
  the grown bytes to the ledger.
"""
from __future__ import annotations

import os
import tempfile
import unittest.mock as mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch

from sglang.srt.weg2 import card_kv_ledger as K
from sglang.srt.weg2 import dual_p_kv_stage as S
from sglang.srt.weg2.d_seat_vram import AllocInfo
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

G = 2 << 20
ENV = {"SGLANG_WEG2_DUAL_LAYOUT": "1", "SGLANG_WEG2_GROUP": "P", S.MAX_TOKENS_ENV: "65536",
       S.STEP_TOKENS_ENV: "4096"}


class FakeSpans:
    available = True

    def __init__(self):
        self.calls = []

    def info(self, ptr):
        return AllocInfo(size=ALLOC, mapped=0, planned=0, active=True)

    def set_spans(self, ptr, spans, now):
        self.calls.append((ptr, tuple(spans), now))
        return 0


class FakePool:
    size, page_size = 65536, 64

    def __init__(self):
        self.rows = []

    def set_stage_backed_rows(self, r):
        self.rows.append(r)


ROW = 2048  # bytes per token of one K buffer (11 FA layers would be 11 such tensors)
ALLOC = (65536 + 64) * ROW


class DualPKvStage(CustomTestCase):
    def test_born_off_by_default(self):
        t = torch.zeros(8)
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIs(S.born(FakePool(), t, "k", spans=FakeSpans(), granule=G), t)
            self.assertEqual(S.pool_tokens(1000), 1000)

    def test_born_trims_and_actor_grows_and_releases(self):
        S._P_BORN.clear()
        spans = FakeSpans()
        pool = FakePool()
        t = torch.zeros(65536 + 64, ROW // 4, dtype=torch.float32)
        with mock.patch.dict(os.environ, ENV):
            self.assertEqual(S.pool_tokens(20000), 65536)
            S.born(pool, t, "k0", spans=spans, granule=G)
        self.assertEqual(len(S._P_BORN), 1)
        self.assertEqual(pool.rows[-1], 64)                     # 0 tokens + the page
        path = os.path.join(tempfile.mkdtemp(prefix="wkvp"), "card")
        d = K.CardKvLedger(path, "D")
        p = K.CardKvLedger(path, "P")
        d.contribute(64 << 20, committed=40 << 20)
        p.contribute(40 << 20)
        caps = []
        with mock.patch.dict(os.environ, ENV):
            st = S.PKvStage(list(S._P_BORN), p, allocator=object(), pools=[pool], page_size=64,
                            granule=G, top_tokens=65536, spans=spans,
                            engage_cap=lambda a, tok, pg: caps.append(tok))
        base = st.bytes_for(0)
        self.assertTrue(st.ensure(10000))                        # -> 12288 on the lattice
        self.assertEqual(st.mapped_tokens, 12288)
        self.assertEqual(caps[-1], 12288)
        got = p.state().committed["P"]
        self.assertEqual(got, st.bytes_for(12288) - base)
        self.assertFalse(st.ensure(65536))                       # more than the pool has free: waits
        self.assertEqual(st.mapped_tokens, 12288)
        self.assertEqual(p.state().committed["P"], got)          # the partial grant went back
        n = st.release_all()
        self.assertEqual(n, got)
        self.assertEqual(caps[-2:], [0, 0])                      # cap to 0 before the unmap
        self.assertEqual(p.state().committed["P"], 0)
        self.assertEqual(st.mapped_tokens, 0)

    def test_lattice_and_rounding(self):
        self.assertEqual(S.round_up(1, 4096), 4096)
        self.assertEqual(S.round_up(4096, 4096), 4096)
        self.assertEqual(S.lattice(8192, 4096), [0, 4096, 8192])

    def test_launcher_env_and_refusal(self):
        from sglang.srt.weg2 import launcher as L

        ns = L.build_parser().parse_args(["--tree", "/x", "--tag", "t", "--dual-share", "--dual-unified-kv", "on"])
        L.resolve_dual_layout(ns)
        self.assertEqual(L.dual_share_env(ns, "P")[S.MAX_TOKENS_ENV], "196608")
        self.assertNotIn(S.MAX_TOKENS_ENV, L.dual_share_env(ns, "D"))
        off = L.build_parser().parse_args(["--tree", "/x", "--tag", "t", "--dual-share"])
        L.resolve_dual_layout(off)
        self.assertNotIn(S.MAX_TOKENS_ENV, L.dual_share_env(off, "P"))
        with self.assertRaises(L.Weg2DualLayoutRefused):
            L.resolve_dual_layout(L.build_parser().parse_args(
                ["--tree", "/x", "--tag", "t", "--dual-unified-kv", "on"]))

    def test_hooks_are_wired(self):
        import inspect

        from sglang.srt.managers import weg2_store_told as T
        from sglang.srt.mem_cache import memory_pool as MP
        from sglang.srt.model_executor import model_runner_kv_cache_mixin as MX

        self.assertIn("_dpk.pp0_grant(scheduler, req) == 0", inspect.getsource(T.intake))
        self.assertIn("_dpk.on_told(scheduler, item)", inspect.getsource(T._follower_absorb_impl))
        self.assertIn("_dual_kv_retry(scheduler)", inspect.getsource(T.pp0_publish))
        self.assertIn("_dpk.born(pool, t, name)", inspect.getsource(MP._kv_stage_born))
        src = inspect.getsource(MX)
        self.assertIn("_dpk.attach(self)", src)
        self.assertIn("max_tokens = _dpk.pool_tokens(max_tokens)", src)
        from sglang.srt.managers import scheduler as SC

        self.assertIn("_dpk.on_idle(self)", inspect.getsource(SC.Scheduler.on_idle))

    def _cards(self, budgets_mib):
        root = tempfile.mkdtemp(prefix="wkvg")
        paths, ds = [], []
        for i, b in enumerate(budgets_mib):
            pth = os.path.join(root, "card%d" % i)
            d = K.CardKvLedger(pth, "D")
            d.contribute(b << 20, committed=0)
            paths.append(pth)
            ds.append(d)
        return paths, ds

    def test_group_grant_is_all_or_none(self):
        paths, ds = self._cards([100, 100, 100])
        stages = [{"ledger": p, "step": 4096, "top": 16384, "bytes": [0, 10 << 20, 20 << 20, 30 << 20, 40 << 20]}
                  for p in paths]
        ds[1].request(95 << 20)                                # card 1 has only 5 MiB free
        open_p = lambda pth: K.CardKvLedger(pth, "P")
        self.assertEqual(S.group_grant(stages, 4000, open_p), 0)
        for p in paths:                                        # nothing held anywhere
            self.assertEqual(K.peek(p).committed["P"], 0)
        ds[1].release(95 << 20)
        self.assertEqual(S.group_grant(stages, 4000, open_p), 4096)
        for p in paths:
            self.assertEqual(K.peek(p).committed["P"], 10 << 20)

    def test_d_growth_between_pp0_grant_and_follower_cannot_break_the_follower(self):
        # operator order: D grows between PP0's grant and the follower's intake -> no alloc failure
        paths, ds = self._cards([100, 100, 100])
        stages = [{"ledger": p, "step": 4096, "top": 16384, "bytes": [0, 10 << 20, 20 << 20, 30 << 20, 40 << 20]}
                  for p in paths]
        led = K.CardKvLedger(paths[2], "P")
        led.contribute(0)                                      # the follower joined at its boot
        lvl = S.group_grant(stages, 8000, lambda pth: K.CardKvLedger(pth, "P"))
        self.assertEqual(lvl, 8192)
        for d in ds:                                           # D takes everything that is left
            d.request(1 << 30)
        spans = FakeSpans()
        follower = S.PKvStage([(1, S._geom_for(torch.zeros(16448, 512), 16384, 64, "k", ALLOC))], led,
                              allocator=object(), pools=[], page_size=64, granule=G, top_tokens=16384,
                              spans=spans, engage_cap=lambda *a: None)
        follower.map_granted(lvl)                              # maps, requests nothing -> cannot fail
        self.assertEqual(follower.mapped_tokens, 8192)
        self.assertEqual(K.peek(paths[2]).committed["P"], 20 << 20)   # PP0's commitment stands
        self.assertLessEqual(sum(K.peek(paths[2]).committed.values()), K.peek(paths[2]).budget)

    def test_told_carries_the_grant(self):
        import types

        from sglang.srt.managers import weg2_store_told as T

        req = types.SimpleNamespace(_dual_kv_tokens=8192)
        told = T._dual_kv_wire(T.Weg2StoreTold(rid="r", told=0), req)
        self.assertEqual(getattr(told, S.WIRE_DUAL_KV), 8192)
        plain = T._dual_kv_wire(T.Weg2StoreTold(rid="r", told=0), types.SimpleNamespace())
        self.assertFalse(hasattr(plain, S.WIRE_DUAL_KV))

    def test_grants_accumulate_until_idle_release(self):
        paths, ds = self._cards([100])
        led = K.CardKvLedger(paths[0], "P")
        led.contribute(0)
        st = S.PKvStage([(1, S._geom_for(torch.zeros(16448, 512), 16384, 64, "k", ALLOC))], led,
                        allocator=object(), pools=[], page_size=64, granule=G, top_tokens=16384,
                        spans=FakeSpans(), engage_cap=lambda *a: None)
        stages = [{"ledger": paths[0], "step": 4096, "top": 16384, "bytes": st.table()}]
        open_p = lambda pth: K.CardKvLedger(pth, "P")
        st.map_granted(S.group_grant(stages, 12000, open_p))   # request 1: 12288 tokens
        st.map_granted(S.group_grant(stages, 3000, open_p))    # request 2 while 1 still holds pages
        self.assertEqual(st.mapped_tokens, 12288)             # the mapping never shrinks under a request
        committed = K.peek(paths[0]).committed["P"]
        self.assertGreaterEqual(committed, st.bytes_for(12288) - st.bytes_for(0))   # covers the mapping
        self.assertEqual(st.release_all(), committed)
        self.assertEqual(K.peek(paths[0]).committed["P"], 0)



class HybridPoolResolution(CustomTestCase):
    """Metal dgkpwa (...09300754): 'HybridLinearKVPool' object has no attribute
    'set_stage_backed_rows' on every rank -- the 27B pool is the hybrid wrapper,
    the stage rows belong to its inner FA pool (MHATokenToKVPool)."""

    def test_stage_pools_resolve_the_inner_fa_pool(self):
        from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, MHATokenToKVPool

        mha = object.__new__(MHATokenToKVPool)
        hybrid = object.__new__(HybridLinearKVPool)
        hybrid.full_kv_pool = mha
        self.assertEqual(S.stage_pools(hybrid), [mha])
        self.assertEqual(S.stage_pools(mha), [mha])

    def test_move_on_a_hybrid_wrapped_pool_sets_the_inner_rows(self):
        from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, MHATokenToKVPool

        mha = object.__new__(MHATokenToKVPool)
        hybrid = object.__new__(HybridLinearKVPool)
        hybrid.full_kv_pool = mha
        geom = S._geom_for(torch.zeros(16448, 512), 16384, 64, "k", ALLOC)
        st = S.PKvStage([(1, geom)], object(), allocator=object(), pools=S.stage_pools(hybrid), page_size=64,
                        granule=G, top_tokens=16384, spans=FakeSpans(), engage_cap=lambda *a: None)
        st._move(4096)
        self.assertEqual(mha._stage_backed_rows, geom.slots_for(4096))
