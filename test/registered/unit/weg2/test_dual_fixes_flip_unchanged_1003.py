# SPDX-License-Identifier: Apache-2.0
"""Item 700 DUAL-FIX-GATE AUDIT: every dual fix since the flip base 673cc89f6a (27B INT8 y8v)
is INERT in the flip form (27B INT8 row authority, NF, anything without the dual layout).

User order 03.10. ("WEHE ... normales flip/27b/nf kaputtgemacht?"): dual fixes live only behind
the dual gate, proven by a test "flip unchanged". One class per fix; each drives the function
the fix touched with the gate OFF (no SGLANG_WEG2_DUAL_LAYOUT, or the wrong group) and shows
the pre-fix behaviour: no state written, no collective, no ledger touched, no reorder.

  Q-610  dual_anchor_release.armed / UnifiedRadixCache claim retry, retain release, END registry
  Q-630  return_untold_grant / _return_untold_dual_grant (fakes without the dual attribute)
  Q-640  pp_slot_fidelity.local_pp_room -- DIFFERENTIAL against the pre-Q-640 body (the one
         ungated fix of the audit: NF runs TP=1/PP>1 and went through the re-read rounds)

INT8 flip port (Q-694 branch, base b2744f5043 = 673cc89f6a + #968): only the classes of the
fixes this base carries (Q-610/630/640). Q-650..Q-691 are not on this line; their classes stay
on desk/27b-dual-gate-audit-1003 (ddc168a999).
"""
from __future__ import annotations

import asyncio
import collections
import os
import tempfile
import time
import types
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from sglang.srt.managers import weg2_store_told as ST  # noqa: E402
from sglang.srt.weg2 import card_kv_ledger as K  # noqa: E402
from sglang.srt.weg2 import dual_anchor_release as DAR  # noqa: E402
from sglang.srt.weg2 import dual_d_kv_stage as DK  # noqa: E402
from sglang.srt.weg2 import dual_p_kv_stage as PK  # noqa: E402
from sglang.srt.weg2 import front as F  # noqa: E402
from sglang.srt.weg2 import p_twin_defer as TW  # noqa: E402
from sglang.srt.weg2 import pp_slot_fidelity as SF  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

MIB = 1 << 20
DUAL_KEYS = ("SGLANG_WEG2_DUAL_LAYOUT", "SGLANG_WEG2_GROUP")


@pytest.fixture(autouse=True)
def flip_env(monkeypatch):
    """The flip form: neither the dual layout nor a group is named."""
    for k in DUAL_KEYS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.delenv(SF.ENV, raising=False)


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()
        asyncio.set_event_loop(asyncio.new_event_loop())


def _front(dual=False):
    f = F.Front(prefill="http://p", decode="http://d", awake="D" if dual else "P", tag="flip",
                store_dir="/tmp", prefill_sid=0, decode_sid=0, dc_reserve={}, w_s=45.0, weight_chunks=2,
                flip_min_work_tokens=1, dual_layout=dual)

    async def rpc(g, path, body, timeout):
        raise AssertionError("the flip form made an RPC: %s" % path)

    f.rpc = rpc
    return f


def _pending(rid, uncached, paused=0):
    fut = asyncio.get_event_loop().create_future()
    p = F.Pending(rid, "/generate", {}, "x", time.time(), fut, est_prompt=uncached, est_uncached=uncached)
    p.dual_paused_n = paused
    return p


def _wrong_gates():
    """Env combinations that must NOT arm group-P dual fixes."""
    return [{}, {"SGLANG_WEG2_GROUP": "P"}, {"SGLANG_WEG2_DUAL_LAYOUT": "1"},
            {"SGLANG_WEG2_DUAL_LAYOUT": "1", "SGLANG_WEG2_GROUP": "D"},
            {"SGLANG_WEG2_DUAL_LAYOUT": "0", "SGLANG_WEG2_GROUP": "P"}]


# ---------------------------------------------------------------------------------- Q-610

class TestQ610FlipUnchanged:
    def test_armed_only_on_dual_group_p(self):
        for env in _wrong_gates():
            assert DAR.armed(env) is False, env
        assert DAR.armed({"SGLANG_WEG2_DUAL_LAYOUT": "1", "SGLANG_WEG2_GROUP": "P"}) is True

    def test_tree_hooks_are_noops_in_the_flip_form(self):
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache as U

        # a bare object: any attribute read past the gate would raise
        tree = SimpleNamespace()
        node = object()
        assert U._weg2_dual_claim_retry(tree, node, object(), "h") is None
        assert U.weg2_dual_release_ended(tree, at="retain") == 0
        assert U.weg2_dual_release_ended(tree, at="claim", claimer=node) == 0
        U._weg2_dual_note_end_anchor(tree, "weg2-0-1", node)
        assert vars(tree) == {}, "the flip form wrote Q-610 state on the tree: %r" % vars(tree)


# ---------------------------------------------------------------------------------- Q-630

class TestQ630FlipUnchanged:
    def test_request_without_a_dual_grant_returns_nothing(self):
        for req in (SimpleNamespace(rid="weg2-0-1"),                       # a fake without the attribute
                    SimpleNamespace(rid="weg2-0-1", _dual_grant_untold=None)):
            assert PK.return_untold_grant(SimpleNamespace(), req, "x") == 0
            assert PK.return_untold_grant(SimpleNamespace(), req, "regrant") == 0

    def test_pp0_drop_hook_is_inert_for_flip_requests(self):
        sched = SimpleNamespace(ps=SimpleNamespace(pp_rank=0))
        ST._return_untold_dual_grant(sched, SimpleNamespace(rid="weg2-0-1"), "left_queue")
        ST._return_untold_dual_grant(sched, SimpleNamespace(rid="weg2-0-1", _dual_grant_untold=None), "abort")

    def test_no_actor_no_grant_and_the_told_is_not_marked(self):
        sched = SimpleNamespace(tp_worker=None, ps=SimpleNamespace(pp_rank=0, pp_size=3))
        req = SimpleNamespace(rid="weg2-0-1", origin_input_ids=[1] * 100)
        assert PK.pp0_grant(sched, req) is None
        told = SimpleNamespace()
        assert PK.with_dual_kv(told, req) is told
        assert not hasattr(req, "_dual_grant_untold")
        assert not hasattr(told, PK.WIRE_DUAL_KV)


# ---------------------------------------------------------------------------------- Q-640

def _base_local_pp_room(tree, kv_tokens, floor, rid=None):
    """local_pp_room exactly as on 673cc89f6a~ (before Q-640): ONE eviction round, the
    verdict from the single read before it. Kept here as the reference."""
    if not SF.enabled() or not getattr(tree, SF.FLOOR_LOCAL_PP_ATTR, False):
        return None
    alloc = getattr(tree, "token_to_kv_pool_allocator", None)
    if alloc is None:
        return None
    kv_tokens = int(kv_tokens)
    avail0 = int(alloc.available_size())
    evictable = int(tree.evictable_size())
    short = kv_tokens - avail0
    if short > 0 and evictable > 0:
        from sglang.srt.mem_cache.base_prefix_cache import EvictParams

        tree.evict(EvictParams(num_tokens=min(short, evictable)))
    return int(alloc.available_size()) >= kv_tokens


class _Alloc:
    def __init__(self, avail):
        self.avail = avail

    def available_size(self):
        return self.avail


class _Tree:
    """Evicting frees ``n``; the eviction also drains an in-flight write-back (#1465) that
    makes ``held`` more tokens evictable -- the y8u PP2 shape."""

    def __init__(self, avail, evictable, held=0, local=True):
        self.token_to_kv_pool_allocator = _Alloc(avail)
        self._evictable, self._held = evictable, held
        self.calls = []
        setattr(self, SF.FLOOR_LOCAL_PP_ATTR, local)

    def evictable_size(self):
        return self._evictable

    def evict(self, params):
        n = min(int(params.num_tokens), self._evictable)
        self._evictable -= n
        self.token_to_kv_pool_allocator.avail += n
        self.calls.append(int(params.num_tokens))
        self._evictable += self._held
        self._held = 0
        return SimpleNamespace(num_tokens_evicted=n)


class TestQ640FlipUnchanged:
    SCENARIOS = [
        # (avail, evictable, held, kv_tokens)
        (991, 2081, 82788, 42084),     # the y8u PP2 shape: a drain makes more evictable
        (991, 2081, 0, 42084),         # a true residual
        (991, 84869, 0, 42084),        # the peers: one round is enough
        (50000, 10, 0, 42084),         # already room
        (0, 0, 0, 1),                  # nothing to evict
        (100, 500, 40000, 700),        # second round would be needed after the first drained
        (3072, 100, 100, 4096),        # 1024 short, 100 evictable + 100 drained: still short
    ]

    @pytest.mark.parametrize("scn", SCENARIOS)
    @pytest.mark.parametrize("env", [{}, {"SGLANG_WEG2_DUAL_LAYOUT": "0"}, {"SGLANG_WEG2_GROUP": "P"},
                                     {"SGLANG_WEG2_GROUP": "D"}])
    def test_flip_verdict_and_evictions_equal_the_pre_q640_body(self, scn, env):
        avail, ev, held, kv = scn
        with mock.patch.dict(os.environ, env):
            a, b = _Tree(avail, ev, held), _Tree(avail, ev, held)
            new = SF.local_pp_room(a, kv, avail, "weg2-0-51")
            old = _base_local_pp_room(b, kv, avail, "weg2-0-51")
        assert new == old
        assert a.calls == b.calls, "the flip form evicted differently from the pre-Q-640 body"
        assert a.token_to_kv_pool_allocator.avail == b.token_to_kv_pool_allocator.avail
        assert len(a.calls) <= 1, "more than one eviction round outside the dual layout"

    def test_not_this_form_stays_none(self):
        assert SF.local_pp_room(_Tree(0, 10, local=False), 5, 0) is None
        assert _base_local_pp_room(_Tree(0, 10, local=False), 5, 0) is None

    def test_the_y8u_shape_is_a_residual_in_flip_and_admitted_in_dual(self):
        # control: the gate really switches the behaviour the flip comparison pins down
        with mock.patch.dict(os.environ, {}):
            assert SF.local_pp_room(_Tree(991, 2081, 82788), 42084, 991) is False
        with mock.patch.dict(os.environ, {"SGLANG_WEG2_DUAL_LAYOUT": "1"}):
            assert SF.local_pp_room(_Tree(991, 2081, 82788), 42084, 991) is True

    def test_max_rounds_by_gate(self):
        assert SF._room_max_rounds({}) == 1
        assert SF._room_max_rounds({"SGLANG_WEG2_DUAL_LAYOUT": "0"}) == 1
        assert SF._room_max_rounds({"SGLANG_WEG2_DUAL_LAYOUT": "1"}) == SF._ROOM_MAX_ROUNDS
