# SPDX-License-Identifier: Apache-2.0
"""Q-696 DUAL D-CACHE-HOLDS-CARD -- metal replay of dual y8z (27B NVFP4,
boot dkr27bnvfp4dual1mpsleepbar1fs10031909, image ceff4aae7b).

19:24:40 P's leg pdflip-0-222 (94720 tokens) waits at PP0 for its card grant; D
holds 1342177280 B on PP0's card -- 225280 mapped rows for ONE running request
(pdflip-0-225, 29112 + 5109 tokens), 163929 of them evictable cache. D's
cache_yield ran only with the whole D group empty: no D-KV line for 91 s, the
grant came at 19:25:54 "after 32021 waits over 73.9 s", right after D ran
empty (CACHE-YIELD evicted=163929, SHRINK 225280 -> 8192).

  A  a RUNNING D yields its unlocked cache once P has waited past the bound
     and shrinks to its live floor (DUAL D-CACHE-YIELD-LIVE); a live row high
     up is named (SHRINK-BLOCKED reason=live_floor); a shrink right after a grow
     is held (19:26:04 GROW 110592->196608 + SHRINK back in the same second).
  B  the front: 19:25:16 P-INTAKE-STALL of pdflip-0-218 (held in the same card
     WAIT) ended the drain ("drain ends, flip to D follows") -- pdflip-0-234
     (24 tokens) waited 31 s in the front behind the in-flight pdflip-0-222.
  C  the wedge watcher: CLASS=UNCLEAR + a corridor-relief post that came back
     NOT APPLICABLE after 10 s; now CLASS=P-KV-WAIT and no post.

Scale: one row = 2346.7 metal tokens (225280 / 96).
"""
from __future__ import annotations

import asyncio
import collections
import json
import os
import tempfile
import time
import types
import unittest.mock as mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

from flliper.srt.pdflip import card_kv_ledger as K  # noqa: E402
from flliper.srt.pdflip import dual_card_stall as DCS  # noqa: E402
from flliper.srt.pdflip import dual_d_kv_stage as D  # noqa: E402
from flliper.srt.pdflip import dual_p_kv_stage as S  # noqa: E402
from flliper.srt.pdflip import front as F  # noqa: E402
from flliper.srt.pdflip.d_seat_vram import AllocInfo  # noqa: E402
from flliper.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

MIB = 1 << 20
ROW = 2048
ALLOC = (16384 + 64) * ROW


class FakeSpans:
    available = True

    def info(self, ptr):
        return AllocInfo(size=ALLOC, mapped=0, planned=0, active=True)

    def set_spans(self, ptr, spans, now):
        return 0


class _Req:
    def __init__(self, rid, n):
        self.rid, self.origin_input_ids, self.output_ids = rid, list(range(n)), []


class FakeTree:
    """D's device radix: ``cached`` ids are unlocked cache; a running seat's ids
    are locked (not here). evict frees the unlocked ids only."""

    def __init__(self, alloc, cached):
        self.alloc, self.cached, self.evicted = alloc, cached, 0

    def evictable_size(self):
        return 0 if self.cached is None else int(self.cached.numel())

    def evict(self, params):
        if self.cached is not None:
            self.alloc.free(self.cached)
            self.evicted += int(self.cached.numel())
            self.cached = None


class _Sched:
    def __init__(self, actor, tree, running=(), waiting=(), parked=()):
        self.running_batch = types.SimpleNamespace(reqs=list(running))
        self.chunked_req = None
        self.waiting_queue = list(waiting)
        self.tree_cache = tree
        self.pdflip_d_parked = list(parked)
        self.server_args = types.SimpleNamespace(chunked_prefill_size=4, speculative_num_draft_tokens=1)
        self.tp_worker = types.SimpleNamespace(model_runner=types.SimpleNamespace(dual_d_kv=actor))
        self._pdflip_group_min_ints = lambda v: list(v)


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def _d(mapped=96, budget_extra=0):
    from flliper.srt.mem_cache.allocator.token import TokenToKVPoolAllocator

    alloc = TokenToKVPoolAllocator(200, torch.float16, "cpu", None, False)
    path = os.path.join(tempfile.mkdtemp(prefix="wq696"), "card")
    led = K.CardKvLedger(path, "D")
    geom = S._geom_for(torch.zeros(264, 8), 256, 1, "k", 264 * 32)
    a = D.DKvStage([(1, geom)], led, allocator=alloc, pools=[], page_size=1, granule=32, top_tokens=192,
                   spans=FakeSpans(), step=16, gmin=lambda v: v)
    b = a.bytes_for(mapped) - a.bytes_for(0)
    led.contribute(b, committed=b)
    a.mapped_tokens, a._committed = mapped, b
    a._engage_cap(alloc, mapped, 1)
    p = K.CardKvLedger(path, "P")
    p.contribute(int(budget_extra))
    return a, alloc, path, p


def _metal_y8z(order="seat_low"):
    """D on PP0's card at 19:24:40: 96 rows mapped, ONE running seat (15 rows,
    pdflip-0-225's 34221 tokens), 70 rows of unlocked cache (163929 tokens), P's
    grant short on this card."""
    a, alloc, path, p = _d()
    if order == "seat_low":
        seat_ids = alloc.alloc(15)
        cache = alloc.alloc(70)
    else:                                                     # the live seat sits ABOVE the cache
        cache = alloc.alloc(70)
        seat_ids = alloc.alloc(15)
    assert seat_ids is not None and cache is not None
    tree = FakeTree(alloc, cache)
    p.request(1 << 30)                                        # pdflip-0-222's grant is short: demand["P"] > 0
    assert K.peek(path).demand["P"] > 0
    sched = _Sched(a, tree, running=[_Req("pdflip-0-225", 15)])
    return a, alloc, path, p, tree, sched


def _ticks(sched, clock, seconds, step_s=0.5):
    out = []
    end = clock.t + seconds
    while clock.t < end:
        out.append(D.tick(sched))
        clock.t += step_s
    return out


# ------------------------------------------------------------------------------- A

class TestALiveCacheYield:
    def test_metal_y8z_running_d_gives_its_cache_to_a_waiting_p(self, caplog):
        a, alloc, path, p, tree, sched = _metal_y8z()
        clock = Clock()
        d_before = K.peek(path).committed["D"]
        with mock.patch.object(S, "_now", clock), caplog.at_level("INFO", logger=D.logger.name):
            _ticks(sched, clock, 30.0)                        # 52 s on metal with no D-KV line
        assert tree.evicted == 70, "D kept 163929 tokens of cache under one running request while P waited"
        assert a.mapped_tokens < 96, "D did not shrink to its live floor after the yield"
        assert a.mapped_tokens >= 15, "D shrank under its running seat"
        assert K.peek(path).committed["D"] < d_before
        lines = [r.getMessage() for r in caplog.records if D.LIVE_YIELD_MARK in r.getMessage()]
        assert lines and "freed=70" in lines[0] and "p_wait_s=" in lines[0] and "floor=" in lines[0]

    def test_no_live_yield_before_the_wait_bound(self):
        a, alloc, path, p, tree, sched = _metal_y8z()
        clock = Clock()
        with mock.patch.object(S, "_now", clock):
            _ticks(sched, clock, 3.0)                         # under the 4 s default
        assert tree.evicted == 0

    def test_the_bound_is_the_env(self, monkeypatch):
        monkeypatch.setenv("FLLIPER_PDFLIP_DUAL_D_LIVE_YIELD_WAIT_S", "1")
        a, alloc, path, p, tree, sched = _metal_y8z()
        clock = Clock()
        with mock.patch.object(S, "_now", clock):
            _ticks(sched, clock, 1.6)
        assert tree.evicted == 70

    def test_off_switch_keeps_the_idle_only_rule(self, monkeypatch):
        monkeypatch.setenv("FLLIPER_PDFLIP_DUAL_D_LIVE_YIELD_WAIT_S", "0")
        a, alloc, path, p, tree, sched = _metal_y8z()
        clock = Clock()
        with mock.patch.object(S, "_now", clock):
            _ticks(sched, clock, 30.0)
        assert tree.evicted == 0

    def test_no_live_yield_while_p_does_not_wait(self):
        a, alloc, path, p, tree, sched = _metal_y8z()
        p.request(0)                                          # P's next grant fitted: demand back to 0
        assert K.peek(path).demand["P"] == 0
        clock = Clock()
        with mock.patch.object(S, "_now", clock):
            _ticks(sched, clock, 30.0)
        assert tree.evicted == 0

    def test_a_held_context_keeps_the_cache(self):
        a, alloc, path, p, tree, sched = _metal_y8z()
        sched.pdflip_d_parked = [_Req("pdflip-0-200", 10)]        # D-PARK / W50 hold: its context comes back here
        clock = Clock()
        with mock.patch.object(S, "_now", clock):
            _ticks(sched, clock, 30.0)
        assert tree.evicted == 0

    def test_the_wait_is_the_groups(self):
        # THIS rank's card shows no P demand; another D rank's card has shown it for 5 s
        a, alloc, path, p, tree, sched = _metal_y8z()
        p.request(0)
        seen = []

        def g(vals):
            vals = list(vals)
            seen.append(len(vals))
            vals[1] = min(vals[1], -1)                        # P waits on another rank's card
            if len(vals) > 7:
                vals[7] = min(vals[7], -5000)                 # ... for 5000 ms
            return vals

        sched._pdflip_group_min_ints = g
        clock = Clock()
        with mock.patch.object(S, "_now", clock):
            D.tick(sched)
        assert tree.evicted == 70, "the rank without P demand did not yield with its group"
        assert seen and seen[0] >= 9, "the wait did not ride the tick's one collective"

    def test_a_live_row_high_up_is_named(self, caplog):
        a, alloc, path, p, tree, sched = _metal_y8z(order="seat_high")
        clock = Clock()
        with mock.patch.object(S, "_now", clock), caplog.at_level("INFO", logger=D.logger.name):
            _ticks(sched, clock, 30.0)
        assert tree.evicted == 70
        assert a.mapped_tokens == 96, "D shrank under a live row"
        lines = [r.getMessage() for r in caplog.records if D.SHRINK_BLOCKED_MARK in r.getMessage()]
        assert lines and "reason=live_floor" in lines[0] and "live_floor=" in lines[0]


class TestARegrowHold:
    def _grown(self):
        """D grows for a waiting leg 2 (metal 19:26:04 GROW 110592 -> 196608),
        then P's grant runs short on the card."""
        a, alloc, path, p = _d(mapped=48, budget_extra=4 * MIB)
        big = _Req("pdflip-0-218", 70)
        tree = FakeTree(alloc, None)
        sched = _Sched(a, tree, waiting=[big])
        return a, alloc, path, p, tree, sched, big

    def test_no_shrink_right_after_a_grow(self):
        a, alloc, path, p, tree, sched, big = self._grown()
        clock = Clock()
        with mock.patch.object(S, "_now", clock):
            assert D.tick(sched) == "grow"
            grown = a.mapped_tokens
            assert grown > 48
            sched.waiting_queue = []                          # the request's demand left the bookkeeping
            p.request(1 << 30)                                # P waits for the card
            clock.t += 0.3
            verdicts = _ticks(sched, clock, 1.0, step_s=0.1)  # the same second on metal
        assert "shrink" not in verdicts, "D shrank right after its grow (the y8z flap)"
        assert a.mapped_tokens == grown

    def test_the_shrink_comes_after_the_hold(self):
        a, alloc, path, p, tree, sched, big = self._grown()
        clock = Clock()
        with mock.patch.object(S, "_now", clock):
            D.tick(sched)
            grown = a.mapped_tokens
            sched.waiting_queue = []
            p.request(1 << 30)
            clock.t += 6.0                                    # past the 5 s hold
            _ticks(sched, clock, 1.0, step_s=0.1)
        assert a.mapped_tokens < grown


# ------------------------------------------------------------------------------- B

def _ledgers(p_demand_card=0):
    root = tempfile.mkdtemp(prefix="wq696f")
    paths = [os.path.join(root, "card%d" % i) for i in range(3)]
    for i, pth in enumerate(paths):
        K.CardKvLedger(pth, "D").contribute(1342177280, committed=1342177280)
        P = K.CardKvLedger(pth, "P")
        P.contribute(1291845632)
        if i == p_demand_card:
            P.request(2214592512)                             # pdflip-0-222: 98304 tokens on PP0's card
    return paths


def _front(dual=True):
    f = F.Front(prefill="http://p", decode="http://d", awake="D" if dual else "P", tag="q696",
                store_dir="/tmp", prefill_sid=0, decode_sid=0, dc_reserve={}, w_s=45.0, weight_chunks=2,
                flip_min_work_tokens=1, dual_layout=dual)
    calls = []

    async def rpc(g, path, body, timeout):
        calls.append(path)
        return 200, {}

    f.rpc = rpc
    f._rpc_calls = calls
    return f


def _pending(rid, uncached):
    fut = asyncio.get_event_loop().create_future()
    return F.Pending(rid, "/generate", {}, "x", time.time(), fut, est_prompt=uncached, est_uncached=uncached)


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()
        asyncio.set_event_loop(asyncio.new_event_loop())


STALL = ("leg1 on P returned 503: PDFLIP-INTAKE-STALL rid=pdflip-0-218 need_tokens=0 rem_total_tokens=-1 "
         "cur_rem_tokens=-1 held_s=16.6 running=empty gate=seats allocatable_reqs=0 req_slots_free=0 waiting=2")


class TestBFrontCardStall:
    def _y8z_1925(self, dual=True, ledgers=True):
        """19:25:16: pdflip-0-222 in flight (P's card WAIT), pdflip-0-218 comes back
        INTAKE-STALL; behind it 234 (24 tok), 235 (13450), 236 (27704)."""
        async def run():
            f = _front(dual)
            if ledgers:
                f.dual_kv_ledgers = _ledgers()
            f._p_intake_stalled = False                       # the drain's start
            f._dual_inflight["pdflip-0-222"] = _pending("pdflip-0-222", 3209 + 91511)
            f.queue = collections.deque([_pending("pdflip-0-234", 24), _pending("pdflip-0-235", 13450),
                                         _pending("pdflip-0-236", 27704)])
            p218 = _pending("pdflip-0-218", 94564)
            f._dual_inflight["pdflip-0-218"] = p218
            await f._requeue_intake_stalled(p218, STALL)
            f._dual_inflight.pop("pdflip-0-218", None)          # the leg's finally
            gone = []
            # the drain's gate, as the dual drain reads it (front._p_drain one -> may_dispatch)
            while f.queue and not f._p_intake_stalled and not f._dual_dispatch_held():
                gone.append(f.queue.popleft().rid)
            return f, gone

        return _run(run())

    def test_the_short_leg_goes_while_the_long_leg_waits_for_the_card(self):
        f, gone = self._y8z_1925()
        assert f._p_intake_stalled is False, "the dual drain ended on a card WAIT (flip semantics)"
        assert gone == ["pdflip-0-234"], gone                    # 24 tokens: P's GRANT-BYPASS serves it at once
        assert [p.rid for p in f.queue] == ["pdflip-0-218", "pdflip-0-235", "pdflip-0-236"]
        assert f.counters["dual_stall_bypass"] == 1 and f.counters["dual_card_stalls"] == 1
        assert f._rpc_calls == ["/abort_request"]              # the stalled leg is still dropped on every rank

    def test_the_head_goes_again_once_no_long_leg_is_in_flight(self):
        f, gone = self._y8z_1925()
        f._dual_inflight.pop("pdflip-0-222")                     # 19:25:57: pdflip-0-222's leg 1 came back
        assert f._dual_dispatch_held() is False
        assert f.queue[0].rid == "pdflip-0-218"

    def test_without_a_card_wait_the_stall_ends_the_drain_as_before(self):
        f, gone = self._y8z_1925(ledgers=False)
        assert f._p_intake_stalled is True and gone == []

    def test_flip_form_is_unchanged(self):
        f, gone = self._y8z_1925(dual=False)
        assert f._p_intake_stalled is True and gone == []
        assert f.counters.get("dual_card_stalls", 0) == 0
        assert not getattr(f.queue[0], "q696_card_stall", False)


# ------------------------------------------------------------------------------- C

class TestCWedgeClass:
    def _stage_files(self, tmp, tag, paths):
        for r, pth in enumerate(paths):
            with open(os.path.join(tmp, "st%d" % r), "w") as fh:
                json.dump({"ledger": pth, "step": 4096, "top": 196608, "bytes": [0]}, fh)
        return lambda t, r: os.path.join(tmp, "st%d" % r)

    def _sched(self):
        return types.SimpleNamespace(
            ps=types.SimpleNamespace(pp_size=3, pp_rank=2), is_initializing=False,
            waiting_queue=[object()], running_batch=types.SimpleNamespace(reqs=[]),
            last_first_token_progress_time=time.perf_counter() - 25.7,
            last_prefill_progress_time=time.perf_counter() - 25.7, forward_ct=0)

    def _alarm(self, env, demand=True):
        from flliper.srt.managers.scheduler_components import invariant_checker as IC

        tmp = tempfile.mkdtemp(prefix="wq696c")
        paths = _ledgers(p_demand_card=0 if demand else 99)
        sched = self._sched()
        posts = []
        with mock.patch.dict(os.environ, env), \
                mock.patch.object(S, "stage_file", self._stage_files(tmp, "x", paths)), \
                mock.patch.object(IC, "_admission_wedge_recovery_threshold", lambda: 2.0), \
                mock.patch.object(IC, "get_recovery_channel",
                                  lambda s: types.SimpleNamespace(post=lambda now, tokens=0: posts.append(now) or 1,
                                                                  last_outcome=None)):
            alarm, detail = IC.check_admission_wedge_once(sched)
            IC.AdmissionWedgeRecovery(sched).step(alarm)
        return alarm, detail, posts, sched

    DUAL_P = {"FLLIPER_PDFLIP_DUAL_LAYOUT": "1", "FLLIPER_PDFLIP_GROUP": "P"}

    def test_dual_p_card_wait_is_named_and_not_posted(self):
        alarm, detail, posts, sched = self._alarm(self.DUAL_P)
        assert alarm
        assert "CLASS=P-KV-WAIT" in detail and "CLASS=UNCLEAR" not in detail
        assert "P demand=" in detail
        assert posts == [], "the dual P card wait posted a corridor relief (NOT APPLICABLE by construction)"

    def test_dual_p_without_a_card_wait_is_classified_as_before(self):
        alarm, detail, posts, sched = self._alarm(self.DUAL_P, demand=False)
        assert alarm and "CLASS=P-KV-WAIT" not in detail and len(posts) == 1

    @pytest.mark.parametrize("env", [{}, {"FLLIPER_PDFLIP_GROUP": "P"}, {"FLLIPER_PDFLIP_DUAL_LAYOUT": "1"},
                                     {"FLLIPER_PDFLIP_DUAL_LAYOUT": "1", "FLLIPER_PDFLIP_GROUP": "D"}])
    def test_other_forms_classify_and_post_as_before(self, env):
        alarm, detail, posts, sched = self._alarm(env)
        assert alarm and "CLASS=P-KV-WAIT" not in detail and len(posts) == 1
        assert not hasattr(sched, "_wedge_p_kv_wait")


class TestPure:
    def test_stall_held(self):
        head = types.SimpleNamespace(rid="h", q696_card_stall=True, intake_stalled=True)
        long_ = types.SimpleNamespace(est_uncached=94720)
        short = types.SimpleNamespace(est_uncached=24)
        assert DCS.stall_held(head, {"x": long_}, short_limit=8192) == "x"
        assert DCS.stall_held(head, {"x": short}, short_limit=8192) is None
        assert DCS.stall_held(head, {"h": long_}, short_limit=8192) is None
        head.q696_card_stall = False
        assert DCS.stall_held(head, {"x": long_}, short_limit=8192) is None

    def test_shrink_blocked_reason(self):
        kw = dict(mapped=96, need=32, floor=20, step=16, holds=False, p_missing=False, regrow_hold=False)
        assert D.shrink_blocked_reason(**kw) == "d_air"
        assert D.shrink_blocked_reason(**dict(kw, floor=90)) == "live_floor"
        assert D.shrink_blocked_reason(**dict(kw, regrow_hold=True)) == "regrow_hold"
        assert D.shrink_blocked_reason(**dict(kw, holds=True)) == "holds"
        assert D.shrink_blocked_reason(**dict(kw, need=90)) is None
