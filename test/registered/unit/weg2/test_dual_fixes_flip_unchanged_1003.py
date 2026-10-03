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
  Q-650  dual_anchor_release.armed_d / D tick / front ANCHOR-OWED hold
  Q-660  _lend_armed / awake_lend / awake_reclaim / wake_reclaim / D tick signal / pump stages
  Q-670  _p_drain_pool wake, _dual_short_reorder (the PP0 GRANT-BYPASS flip test is
         test_weg2_q670_dual_parallel_1003.FlipPp0GrantUnchanged)
  Q-680  flush_acks_when_idle / front RESUME-STALE (the audit made the front gate explicit)
  Q-690  p_twin_defer rule 6 (gate and release_due)
  Q-691  _dual_resume_held unstarve / bypass count
  Q-692  dual_handback_defer.* / d_seats.admission_gate exemption / front pause requeue
  Q-693  follower untold waiting abort / PF ACK-ROOM HOLD / PP0 intake stamp / RESUME-OWN-HELD
  Q-695  #791T probe old-instance chunk / old chunk abort store release / #791C new-instance named
  Q-696  D live cache yield + regrow hold (D tick without the dual actor) / front INTAKE-STALL
         card WAIT (drain ends as before) and STALL-BYPASS / wedge class P-KV-WAIT (no post skip)
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


# ---------------------------------------------------------------------------------- Q-650

class TestQ650FlipUnchanged:
    def test_armed_d_only_on_dual_group_d(self):
        for env in [{}, {"SGLANG_WEG2_GROUP": "D"}, {"SGLANG_WEG2_DUAL_LAYOUT": "1"},
                    {"SGLANG_WEG2_DUAL_LAYOUT": "1", "SGLANG_WEG2_GROUP": "P"}]:
            assert DAR.armed_d(env) is False, env
        assert DAR.armed_d({"SGLANG_WEG2_DUAL_LAYOUT": "1", "SGLANG_WEG2_GROUP": "D"}) is True

    def test_d_tick_touches_nothing_in_the_flip_form(self):
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache as U

        tree = SimpleNamespace()
        assert U._weg2_dual_d_tick(tree) is None
        assert vars(tree) == {}

    def test_anchor_owed_hold_does_not_wait_in_the_flip_form(self):
        async def run():
            f = _front(dual=False)
            with mock.patch.object(DAR, "wait_anchor_room", side_effect=AssertionError("waited")), \
                    mock.patch.object(DAR, "read_room", side_effect=AssertionError("read")):
                await f._q650_anchor_owed_hold("weg2-0-121", 1)
            return f

        f = _run(run())
        assert not any(k.startswith("q650_anchor_owed") for k in f.counters)


# ---------------------------------------------------------------------------------- Q-660

class TestQ660FlipUnchanged:
    def test_lend_is_armed_only_on_dual_group_p(self):
        sched = SimpleNamespace(tp_worker=SimpleNamespace(model_runner=SimpleNamespace(
            dual_p_kv=object())))                     # even with an actor attached
        for env in _wrong_gates():
            with mock.patch.dict(os.environ, env):
                assert PK._lend_armed(sched) is None, env
        with mock.patch.dict(os.environ, {"SGLANG_WEG2_DUAL_LAYOUT": "1", "SGLANG_WEG2_GROUP": "P"}):
            assert PK._lend_armed(sched) is sched.tp_worker.model_runner.dual_p_kv   # control

    def test_loans_do_nothing_without_the_dual_p_actor(self):
        sched = SimpleNamespace()
        boom = mock.Mock(side_effect=AssertionError("the flip form measured the card"))
        assert PK.awake_lend(sched, "x", phys=boom, empty_cache=boom) == 0
        assert PK.awake_reclaim(sched, "x") == 0
        assert PK.wake_reclaim(sched) == 0
        assert PK.lent_bytes(SimpleNamespace(_sleep_lent=3, _awake_lent=4)) == 7
        assert PK.lent_bytes(SimpleNamespace()) == 0

    def test_scheduler_order_handler_is_inert(self):
        from sglang.srt.managers import io_struct as IO
        from sglang.srt.managers import scheduler as SC

        for action in ("lend", "reclaim"):
            assert SC.Scheduler.handle_weg2_dual_p_lend(
                SimpleNamespace(), IO.Weg2DualPLendReqInput(action=action)) is None

    def test_d_tick_without_the_dual_d_actor_publishes_no_signal(self):
        sched = SimpleNamespace(tp_worker=SimpleNamespace(model_runner=SimpleNamespace()))
        with mock.patch.object(DK, "publish_d_signal", side_effect=AssertionError("published")):
            assert DK.tick(sched) is None

    def test_pump_runs_no_pressure_or_stage_tick_without_dual_ledgers(self):
        async def run():
            f = _front(dual=False)
            assert not f.dual_kv_ledgers
            with mock.patch.object(F.Front, "_dual_pressure_tick", side_effect=AssertionError("pressure")), \
                    mock.patch.object(F.Front, "_dual_stage_tick", side_effect=AssertionError("stage")), \
                    mock.patch.object(F.Front, "_dual_pause_inflight", side_effect=AssertionError("pause")):
                f.queue = collections.deque()
                f._dual_pump(lambda: None)
            return f

        f = _run(run())
        assert f.counters.get("dual_passes", 0) == 0


# ---------------------------------------------------------------------------------- Q-670

class TestQ670FlipUnchanged:
    def test_drain_pool_sleeps_on_the_long_leg_as_on_base(self):
        async def run():
            q = collections.deque(["long"])
            started = {}
            t0 = time.monotonic()

            async def one(item):
                started[item] = time.monotonic() - t0
                await asyncio.sleep(0.5 if item == "long" else 0.0)
                return item

            async def arrive():
                await asyncio.sleep(0.05)
                q.append("short")

            asyncio.ensure_future(arrive())
            await F._p_drain_pool(q, 2, one, lambda p: None, lambda: True, poll_s=0.05)
            return started

        started = _run(run())
        assert started["short"] >= 0.45, "the flip form woke for an arrival (base sleeps on the leg)"

    def test_default_of_the_wake_switch_is_off(self):
        import inspect

        assert inspect.signature(F._p_drain_pool).parameters["dual_wake"].default is False

    def test_front_never_reorders_in_the_flip_form(self):
        async def run():
            f = _front(dual=False)
            f.queue = collections.deque([_pending("a", 91477), _pending("b", 25, paused=1),
                                         _pending("c", 25)])
            moved = [f._dual_short_reorder(head_blocked=hb) for hb in (True, False)]
            return moved, [p.rid for p in f.queue], f.counters

        moved, order, counters = _run(run())
        assert moved == [False, False] and order == ["a", "b", "c"]
        assert not any(k.startswith("dual_short") for k in counters)


# ---------------------------------------------------------------------------------- Q-680 / Q-691

def _metal_ledgers():
    root = tempfile.mkdtemp(prefix="wkvaudit")
    paths = [os.path.join(root, "card%d" % i) for i in range(3)]
    for pth in paths:
        K.CardKvLedger(pth, "D").contribute(4000 * MIB, committed=0)
        K.CardKvLedger(pth, "P").contribute(0)
    for pth, c in zip(paths, (1107296256, 201326592, 301989888)):
        K.CardKvLedger(pth, "P").request(c)
    return paths


class _Tree680:
    def __init__(self):
        self.ongoing_load_back = [540]
        self.flushed = 0
        self.checked = 0

    def flush_write_through_acks(self):
        self.flushed += 1

    def loading_check(self):
        self.checked += 1
        self.ongoing_load_back.clear()


class TestQ680Q691FlipUnchanged:
    def test_idle_ack_flush_is_not_taken_without_the_dual_group_p(self):
        for env in _wrong_gates():
            with mock.patch.dict(os.environ, env):
                if env.get("SGLANG_WEG2_DUAL_LAYOUT") == "1" and env.get("SGLANG_WEG2_GROUP") == "P":
                    continue
                tree = _Tree680()
                sched = SimpleNamespace(enable_hierarchical_cache=True, tree_cache=tree)
                assert PK.flush_acks_when_idle(sched) is False, env
                assert (tree.flushed, tree.checked, tree.ongoing_load_back) == (0, 0, [540]), env

    def test_on_idle_without_an_actor_names_nothing(self):
        sched = SimpleNamespace(tp_worker=None)
        with mock.patch.object(PK, "_note_idle_held", side_effect=AssertionError("named")):
            assert PK.on_idle(sched) == 0

    def _held_for_ever(self, dual):
        async def run():
            f = _front(dual=dual)
            f.dual_kv_ledgers = _metal_ledgers()
            f.queue = collections.deque([_pending("weg2-0-309", 91477, paused=1)])
            long_ago = time.time() - 600.0
            f._dual_resume_wait_since = long_ago
            f._dual_resume_stale_since = long_ago
            return f, f._dual_resume_held()

        return _run(run())

    def test_flip_front_does_not_resume_on_a_stale_ledger(self):
        f, held = self._held_for_ever(dual=False)
        assert held is True
        assert f.counters.get("dual_resume_stale", 0) == 0
        assert f.counters.get("dual_resume_unstarve", 0) == 0

    def test_control_the_dual_front_does_resume_on_the_same_rows(self):
        f, held = self._held_for_ever(dual=True)
        assert held is False and f.counters.get("dual_resume_stale", 0) == 1

    def test_flip_short_paused_head_is_not_unstarved_or_counted(self):
        async def run():
            f = _front(dual=False)
            f.dual_kv_ledgers = _metal_ledgers()
            f.queue = collections.deque([_pending("weg2-0-309", 173, paused=1), _pending("x", 600)])
            held = f._dual_resume_held()
            moved = f._dual_short_reorder(head_blocked=True)
            return f, held, moved

        f, held, moved = _run(run())
        assert held is True and moved is False
        assert getattr(f.queue[0], "dual_bypassed_n", 0) == 0
        assert f.counters.get("dual_resume_unstarve", 0) == 0

    def test_flip_dispatch_gate_is_the_base_expression(self):
        async def run():
            f = _front(dual=False)
            f.queue = collections.deque([_pending("a", 91477), _pending("b", 25)])
            return f._dual_dispatch_held(), [p.rid for p in f.queue]

        held, order = _run(run())
        assert held is False and order == ["a", "b"]


# ---------------------------------------------------------------------------------- Q-690

class _Src:
    def __init__(self, rid):
        self.rid = rid

    def finished(self):
        return False


class TestQ690FlipUnchanged:
    def test_gate_needs_layout_and_group_p(self):
        for env in _wrong_gates():
            if env.get("SGLANG_WEG2_DUAL_LAYOUT") == "1" and env.get("SGLANG_WEG2_GROUP") == "P":
                continue
            assert TW._dual_p_env(env) is False, env
        assert TW._dual_p_env({"SGLANG_WEG2_DUAL_LAYOUT": "1", "SGLANG_WEG2_GROUP": "P"}) is True
        assert TW._State(1, 1.0, 0.0, 1).dual is False

    def _release(self, dual):
        src = _Src("weg2-0-42")                                 # left P unfinished: not in flight
        req = SimpleNamespace(rid="weg2-0-45")
        st = TW._State(1, 120.0, 0.0, 1, dual=dual)
        st.waits["weg2-0-45"] = TW._Wait(req=req, sources=[src], since=100.0, shared=60000)
        sched = SimpleNamespace()
        setattr(sched, TW._ATTR, st)
        with mock.patch.object(TW, "inflight", return_value=[]), mock.patch.object(TW, "_now", return_value=101.0), \
                mock.patch.object(TW, "_keeps", return_value=True):
            return TW.release_due(sched, {"weg2-0-45"}), st

    def test_flip_keeps_holding_for_the_vanished_source(self):
        out, st = self._release(dual=False)
        assert out == [] and "weg2-0-45" in st.waits and st.n_source_gone == 0

    def test_control_dual_releases_it_at_once(self):
        out, st = self._release(dual=True)
        assert [r.rid for r, twin in out] == ["weg2-0-45"] and out[0][1] is False
        assert st.n_source_gone == 1


# ---------------------------------------------------------------------------------- Q-692

class TestQ692FlipUnchanged:
    def test_handback_defer_helpers_are_inert_outside_dual_d(self):
        from sglang.srt.weg2 import dual_handback_defer as HBD

        req = SimpleNamespace(rid="weg2-0-373", output_ids=[1] * 25, _weg2_x_deferring=False)
        for env in [{}, {"SGLANG_WEG2_GROUP": "D"}, {"SGLANG_WEG2_DUAL_LAYOUT": "1"},
                    {"SGLANG_WEG2_DUAL_LAYOUT": "1", "SGLANG_WEG2_GROUP": "P"}]:
            assert HBD.armed(env) is False, env
            assert HBD.d_own_tail(req, env) == 0
            assert HBD.d_owned(req, env) is False
            assert HBD.defer_exempt(req, env) is False
            assert HBD.begin(req, 26, env=env) is False
            HBD.note_x_defer(req, True, env)
        assert req._weg2_x_deferring is False, "the flip form recorded an X-DEFER verdict"
        assert not hasattr(req, "_weg2_hb_d_own_said")
        denv = {"SGLANG_WEG2_DUAL_LAYOUT": "1", "SGLANG_WEG2_GROUP": "D"}
        assert HBD.d_own_tail(req, denv) == 26 and HBD.d_owned(req, denv) is True   # control

    def test_admission_gate_exempts_nobody_in_the_flip_form(self):
        from sglang.srt.weg2 import d_seats

        parked = [SimpleNamespace(rid="p1", _weg2_x_deferring=True)]
        assert d_seats._dual_defer_exempt(parked) == frozenset()
        gate = d_seats.AdmissionGate()
        assert gate.defer_exempt == frozenset() and gate.newcomers_free is False

    def test_pause_requeue_keeps_intake_stalled_in_the_flip_form(self):
        async def run():
            f = _front(dual=False)
            p = _pending("weg2-0-321", 1000)
            p.intake_stalled = True
            p.dual_pause = True
            f._dual_requeue_paused(p)
            return p

        p = _run(run())
        assert p.intake_stalled is True


# ---------------------------------------------------------------------------------- Q-693

_Q693_KV = {"SGLANG_WEG2_DUAL_P_KV_MAX_TOKENS": "196608"}


def _q693_wrong_gates():
    """Every wrong gate WITH the P KV size set: the size alone arms nothing."""
    return [dict(env, **_Q693_KV) for env in _wrong_gates()]


class TestQ693FlipUnchanged:
    def test_untold_waiting_abort_is_held_as_before(self):
        from sglang.srt.managers import scheduler as SC
        from sglang.srt.weg2 import p_row_authority

        for env in _q693_wrong_gates():
            with mock.patch.dict(os.environ, env), \
                    mock.patch.object(p_row_authority, "applies", lambda s: True):
                f = SimpleNamespace(ps=SimpleNamespace(pp_rank=1, pp_size=3), chunked_req=None,
                                    _weg2_store_told_armed=True, _weg2_store_told={},
                                    waiting_queue=[SimpleNamespace(rid="weg2-0-152", req_pool_idx=None,
                                                                   is_retracted=False)])
                assert SC.Scheduler._weg2_defer_waiting_abort(f, SC.AbortReq(rid="weg2-0-152")) is True, env
                assert sorted(f._weg2_pending_waiting_aborts) == ["weg2-0-152"], env

    def test_no_room_ack_is_zero_at_once(self):
        from sglang.srt.managers import weg2_told_fallback as FB

        pred = SimpleNamespace(rid="weg2-0-182", fill_ids=[0] * 86561, origin_input_ids=[0] * 86561)
        req = SimpleNamespace(rid="weg2-0-183")
        for env in _q693_wrong_gates():
            with mock.patch.dict(os.environ, env), \
                    mock.patch.object(FB, "_loadback_rows", lambda s, r, t: int(t)):
                tree = SimpleNamespace(token_to_kv_pool_allocator=SimpleNamespace(available_size=lambda: 3735),
                                       evictable_size=lambda: 0)
                s = SimpleNamespace(tree_cache=tree, ps=SimpleNamespace(pp_rank=1), waiting_queue=[req],
                                    mbs=[SimpleNamespace(reqs=[pred])], max_running_requests=1)
                assert FB._room_own(s, req, "weg2-0-183", 68096) == 0, env
                assert not hasattr(s, FB._ROOM_HOLD_ATTR), env

    def test_intake_stamp_survives_forget_left_queue(self):
        for env in _q693_wrong_gates():
            with mock.patch.dict(os.environ, env):
                req = SimpleNamespace(rid="weg2-0-152", _dual_grant_untold=None)
                s = SimpleNamespace(ps=SimpleNamespace(pp_rank=0), _weg2_store_told_armed=True,
                                    _weg2_store_told={}, _weg2_store_held={"weg2-0-152": req},
                                    _weg2_told_intake_t={"weg2-0-152": 1.0}, waiting_queue=[])
                assert "intake_t" not in ST.forget_left_queue(s, req, "abort"), env
                assert s._weg2_told_intake_t == {"weg2-0-152": 1.0}, env

    def test_flip_front_names_no_own_held_head(self):
        async def run():
            f = _front(dual=False)
            f.dual_kv_ledgers = _metal_ledgers()
            f.queue = collections.deque([_pending("weg2-0-152", 169, paused=1)])
            return f, f._dual_resume_held()

        f, held = _run(run())
        assert held is True
        assert f.counters.get("dual_resume_own_held", 0) == 0
        assert getattr(f, "_q693_own_held_rid", None) is None


# ---------------------------------------------------------------------------------- Q-695

def _q695_pp1(told=None, old_end=5120):
    """y8z PP1 at 19:26:05.98: instance 2 chunked at old_end, abort recorded; instance 3 queued."""
    rid = "weg2-0-235"
    old = SimpleNamespace(rid=rid, req_pool_idx=None, is_retracted=False,
                          extend_range=SimpleNamespace(start=old_end - 1024, end=old_end))
    new = SimpleNamespace(rid=rid, req_pool_idx=None, is_retracted=False)
    s = SimpleNamespace(pp_group=object(), ps=SimpleNamespace(pp_rank=1, pp_size=3, tp_size=1),
                        waiting_queue=[new], chunked_req=old, _pending_chunked_abort_req=old,
                        running_batch=None, _pp_row_chain_owed=False, _pp_flip_epoch=lambda: -1,
                        _weg2_store_told_armed=True,
                        _weg2_store_told={} if told is None else {rid: told})
    return s, old, new


class TestQ695FlipUnchanged:
    """Q-695 OLD-INSTANCE CHUNK / ABORT KEEPS THE NEW READ / NEW-INSTANCE NAMED: on every wrong
    gate (P KV size set) the #791T probe, the old chunk's store release and the #791C keep
    answer exactly as on ceff4aae7b."""

    def test_791t_probe_defers_and_stops_as_before(self, monkeypatch, caplog):
        from sglang.srt.managers import scheduler_pp_mixin as ppm
        from sglang.srt.managers.pp_row_defer_cap import ROW_DEFER_LAP_CAP, PpRowDeferCapExceeded

        row = ("weg2-0-235", 5120, 1024, True, False, None, None, False, 13449, (), None)
        frame = {"__stamp__": (1, 1560, 1024, -1, 1560, ("weg2-0-235", 5120, 6144)),
                 ppm._ADMISSION_DECISION_PAYLOAD_KEY: (0, (row,))}
        for env in _q693_wrong_gates():
            q = [frame]
            monkeypatch.setattr(ppm, "resolve_src", lambda group, x: 0)
            monkeypatch.setattr(ppm, "typed_inbox", lambda group, q=q: {(0, "proxy"): q})
            with mock.patch.dict(os.environ, env):
                s, old, new = _q695_pp1()
                with pytest.raises(PpRowDeferCapExceeded, match="#791T STORE-TOLD HOP OVERDUE"):
                    for _ in range(ROW_DEFER_LAP_CAP + 2):
                        assert ppm.SchedulerPPMixin._pp_proxy_frame_pending(s, 1) is False
                assert not hasattr(s, "_q695_continued_n"), env
                assert len(q) == 1, env
        assert "Q-695" not in caplog.text

    def _abort(self, monkeypatch, env, schedules, told=None):
        from sglang.srt.managers import scheduler as SC
        from sglang.srt.weg2 import p_row_authority

        monkeypatch.setattr(SC, "prepare_abort", lambda req, why: setattr(req, "aborted_why", why))
        monkeypatch.setattr(SC, "release_kv_cache", lambda *a, **k: None)
        monkeypatch.setattr(p_row_authority, "applies", lambda s: True)
        s, old, new = _q695_pp1(told=told, old_end=6144)
        old.kv_committed_freed, old.to_finish, old.finished = True, None, (lambda: False)
        old.time_stats = SimpleNamespace(trace_ctx=SimpleNamespace(abort=lambda **k: None))
        released, cur, kept = [], {"s": None}, []
        s._pending_chunked_abort_delay = 0
        s._pp_scheduled_extents = lambda: cur["s"]
        s.disaggregation_mode = None
        s.enable_hicache_storage = True
        s.tree_cache = SimpleNamespace(supports_mamba=lambda: False,
                                       release_aborted_request=released.append)
        s.ipc_channels = SimpleNamespace(send_to_tokenizer=SimpleNamespace(send_output=lambda o, r: None))
        with mock.patch.dict(os.environ, env):
            for sch in schedules:
                cur["s"] = sch
                SC.Scheduler.process_pending_chunked_abort(s)
                kept.append(s.chunked_req is old)
        return s, kept, released

    def test_old_chunk_abort_releases_the_rid_as_before(self, monkeypatch):
        for env in _q693_wrong_gates():
            s, kept, released = self._abort(monkeypatch, env,
                                            [{"weg2-0-235": (5120, 1024)}, {"weg2-0-236": (0, 512)}])
            assert kept == [True, False], env
            assert released == ["weg2-0-235"], env
            assert not hasattr(s, "_q695_keep_read_n"), env

    def test_a_schedule_naming_the_rid_keeps_the_old_chunk_as_before(self, monkeypatch):
        for env in _q693_wrong_gates():
            s, kept, released = self._abort(monkeypatch, env, [{"weg2-0-235": (0, 1024)}], told=0)
            assert kept == [True], env
            assert not hasattr(s, "_q695_new_named_n"), env


# ---------------------------------------------------------------------------------- Q-696

class TestQ696FlipUnchanged:
    def test_d_tick_without_the_dual_actor_reads_no_clock_and_no_collective(self):
        sched = SimpleNamespace(tp_worker=SimpleNamespace(model_runner=SimpleNamespace()),
                                _weg2_group_min_ints=mock.Mock(side_effect=AssertionError("collective")))
        with mock.patch.object(PK, "_now", side_effect=AssertionError("clock")), \
                mock.patch.object(DK, "cache_yield", side_effect=AssertionError("yield")):
            assert DK.tick(sched) is None

    def test_d_actor_is_never_armed_outside_dual_group_d(self):
        for env in [{}, {"SGLANG_WEG2_GROUP": "D"}, {"SGLANG_WEG2_DUAL_LAYOUT": "1"},
                    {"SGLANG_WEG2_DUAL_LAYOUT": "1", "SGLANG_WEG2_GROUP": "P"}]:
            env = dict(env, SGLANG_WEG2_DUAL_D_KV_MAX_TOKENS="196608")
            assert DK.armed(env) is False, env

    def test_intake_stall_ends_the_drain_and_names_the_flip(self, caplog):
        async def run():
            f = _front(dual=False)
            f.dual_kv_ledgers = _metal_ledgers()               # even with ledgers that show P waiting
            for pth in f.dual_kv_ledgers:
                K.CardKvLedger(pth, "P").request(8000 * MIB)

            async def rpc(g, path, body, timeout):
                return 200, {}

            f.rpc = rpc
            f._p_intake_stalled = False
            f.queue = collections.deque([_pending("weg2-0-234", 24)])
            p = _pending("weg2-0-218", 94564)
            f._dual_inflight["weg2-0-222"] = _pending("weg2-0-222", 94720)
            with caplog.at_level("WARNING", logger="weg2.front"):
                await f._requeue_intake_stalled(p, "WEG2-INTAKE-STALL rid=weg2-0-218")
            return f, f._dual_dispatch_held()

        f, held = _run(run())
        assert f._p_intake_stalled is True
        assert [p.rid for p in f.queue] == ["weg2-0-218", "weg2-0-234"]     # no STALL-BYPASS reorder
        assert held is False
        assert f.counters.get("dual_card_stalls", 0) == 0 and f.counters.get("dual_stall_bypass", 0) == 0
        assert not hasattr(f.queue[0], "q696_card_stall")
        assert any("drain ends, flip to D follows: WEG2-INTAKE-STALL" in r.getMessage() for r in caplog.records)

    def test_wedge_class_is_never_p_kv_wait_outside_dual_group_p(self):
        from sglang.srt.weg2 import dual_card_stall as DCS

        boom = mock.Mock(side_effect=AssertionError("read a stage file"))
        sched = SimpleNamespace(ps=SimpleNamespace(pp_size=3))
        for env in _wrong_gates():
            if env.get("SGLANG_WEG2_DUAL_LAYOUT") == "1" and env.get("SGLANG_WEG2_GROUP") == "P":
                continue
            with mock.patch.object(PK, "stage_file", boom):
                assert DCS.p_kv_wait_class(sched, env) is None, env
