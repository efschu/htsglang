# SPDX-License-Identifier: Apache-2.0
"""DOUBLE FLIP 1015 (user reprimand 15.10.): the P phase ends with a request over X still queued.

Boots h16r 39fa0976ce (110016) and h16p 226aad7105 (115748, 124245, 133855), 15.10.2026: 22 P->D flips were
followed within 15 s by a D->P flip (PARK-IMMEDIATE cause=over-x). Read off the front logs (rid in the queue at the
P->D flip begin or not, how the P phase ended):

* 10 stranded a request in the P queue that the park fires for the moment D wakes (``awake_s=0.0``: D decoded not
  one token): 2 at the phase cap (this file), 6 at a P-INTAKE-STALL (defect 1: ``need <= pool`` but
  ``rem_total < need``, evictable KV not counted free; K9b is in this base), 2 at the DECODE-COLLECT seat gate
  (``p-cap`` = free D seats; the seat count, not the phase cap);
* 11 were arrivals AFTER the flip (a new over-X request, legitimate), 1 a lost hand-off (HANDOFF-LOST-REROUTE).

THE CAP CASES (this file): the P phase ends at ``--p-phase-max-requests`` and FLIPCYCLE H5 lets SHORTs ride past
the cap -- but not a request that NEEDS P. Real queues:

* 11:04:50Z epoch 1, ``dispatched=9 cap=6 end=cap queue_left=2``: weg2-0-1..-6 under the cap, then the SHORTs
  weg2-0-8 (40), weg2-0-10 (98) and weg2-1-11 (25) rode; the LONGs weg2-0-7 (7199 > X 5713, waited 36 s) and
  weg2-0-9 (7333) stayed. PARK-IMMEDIATE for weg2-0-7 4.2 s later, awake_s=0.0.
* 12:16:23Z epoch 31, ``dispatched=9 cap=6 end=cap queue_left=7``: weg2-28-76 (uncached 1258, routed LONG on the
  X_busy band floor 1205, so ``d_eligible`` False: no SHORT) stayed, PARK-IMMEDIATE 2.2 s later, waited 47.9 s.

Fix: one predicate for both readers -- the park trigger (:func:`phase_policy.immediate_park_trigger`) and the
cap exemption (``Front._p_phase_cap_exempt``); the exempt ones leave the queue oldest first (PRIO-AS-AGE).
"""
from __future__ import annotations

import asyncio
import collections
import inspect
import types

import pytest

from sglang.srt.weg2 import front as F
from sglang.srt.weg2 import phase_policy


def _real_pending(loop, rid, uncached, *, d_eligible=False, x_routed=0, p_only=False, x_requeues=0, **kw):
    """The REAL Pending dataclass (its field names are what the predicates read)."""
    p = F.Pending(rid=rid, path="/v1/messages", payload={}, text="", t_arrive=0.0, fut=loop.create_future(),
                  est_uncached=int(uncached), d_eligible=d_eligible, x_routed=x_routed, p_only=p_only,
                  x_requeues=x_requeues)
    for k, v in kw.items():
        setattr(p, k, v)
    return p


def _front(x):
    return types.SimpleNamespace(tp_prefill_max_tokens=x)


def _drain(items, cap, exempt, limit=2):
    async def main():
        q = collections.deque(items)
        done = []

        async def one(p):
            await asyncio.sleep(0)
            return p

        stats = {}
        await F._p_drain_pool(q, limit, one, done.append, lambda: True, max_dispatch=cap, stats=stats,
                              cap_exempt=exempt)
        return [p.rid for p in done], [p.rid for p in q], stats

    return asyncio.run(main())


@pytest.fixture
def loop():
    lp = asyncio.new_event_loop()
    yield lp
    lp.close()


def _queue_1104(loop):
    """The 11:04:50Z queue in arrival order (X=5713 live; routed on X=4096). The six under the cap carry the
    uncached the SERVED lines imply (prompt - cached); only 0-7, 0-9 and the three SHORTs matter."""
    L, S = dict(d_eligible=False), dict(d_eligible=True)
    return [
        _real_pending(loop, "weg2-0-1", 7234, **L), _real_pending(loop, "weg2-0-2", 14029, **L),
        _real_pending(loop, "weg2-0-3", 7100, **L), _real_pending(loop, "weg2-0-4", 323, **S),
        _real_pending(loop, "weg2-0-5", 10823, **L), _real_pending(loop, "weg2-0-6", 189, **S),
        _real_pending(loop, "weg2-0-7", 7199, **L), _real_pending(loop, "weg2-0-8", 40, **S),
        _real_pending(loop, "weg2-0-9", 7333, **L), _real_pending(loop, "weg2-0-10", 98, **S),
        _real_pending(loop, "weg2-1-11", 25, **S),
    ]


class TestRealDoubleFlips:
    def test_1104_old_rule_strands_the_older_long_behind_younger_shorts(self, loop):
        """RED (the h16r reading): short-rides only -- the SHORTs pass the cap, the LONGs stay."""
        f = _front(5713)
        done, left, stats = _drain(_queue_1104(loop), 6, lambda p: F.Front._p_phase_short_rides(f, p), limit=6)
        assert left == ["weg2-0-7", "weg2-0-9"]          # the log's queue_left=2
        assert stats["short_rides"] == 3 and stats["dispatched"] == 9  # the log's dispatched=9
        # and the park fires for the head of exactly this queue:
        assert phase_policy.immediate_park_trigger(
            [p for p in _queue_1104(loop) if p.rid in left], 5713).rid == "weg2-0-7"

    def test_1104_new_rule_takes_every_request_that_needs_p(self, loop):
        """GREEN: nothing the park would fire for is left, and the order after the cap is the age order."""
        f = _front(5713)
        order = []
        async def main():
            q = collections.deque(_queue_1104(loop))
            async def one(p):
                order.append(p.rid)
                await asyncio.sleep(0)
                return p
            await F._p_drain_pool(q, 1, one, lambda p: None, lambda: True, max_dispatch=6, stats={},
                                  cap_exempt=lambda p: F.Front._p_phase_cap_exempt(f, p))
            return list(q)
        left = asyncio.run(main())
        assert left == []
        # past the cap the OLDER LONG 0-7 goes before the younger SHORT 0-8 (PRIO-AS-AGE), 0-9 before 0-10, 1-11
        assert order == ["weg2-0-1", "weg2-0-2", "weg2-0-3", "weg2-0-4", "weg2-0-5", "weg2-0-6",
                         "weg2-0-7", "weg2-0-8", "weg2-0-9", "weg2-0-10", "weg2-1-11"]

    def test_1216_routed_long_on_the_band_floor_is_not_a_short_but_needs_p(self, loop):
        """12:16:23Z weg2-28-76: uncached 1258 <= live X 7036 (so below X), but routed LONG on X_busy 1205 -- the
        park reads it as needing P (#246b), the SHORT ride predicate does not (d_eligible False)."""
        f = _front(7036)
        p = _real_pending(loop, "weg2-28-76", 1258, d_eligible=False, x_routed=1205)
        assert not F.Front._p_phase_short_rides(f, p)
        assert phase_policy.immediate_park_trigger([p], 7036) is p
        assert F.Front._p_phase_needs_p(f, p)
        queue = [_real_pending(loop, f"weg2-x-{i}", 90_000, d_eligible=False) for i in range(6)]
        queue += [p, _real_pending(loop, "weg2-31-78", 12, d_eligible=True)]
        done, left, _ = _drain(queue, 6, lambda q: F.Front._p_phase_cap_exempt(f, q), limit=6)
        assert "weg2-28-76" in done and left == []
        done_old, left_old, _ = _drain(queue, 6, lambda q: F.Front._p_phase_short_rides(f, q), limit=6)
        assert left_old == ["weg2-28-76"]


class TestThePredicate:
    def test_exempt_is_short_or_needs_p(self, loop):
        f = _front(5713)
        short = _real_pending(loop, "s", 25, d_eligible=True)
        long_ = _real_pending(loop, "l", 7199)
        mid_not_p = _real_pending(loop, "m", 3000, d_eligible=False)   # below X, not routed to P: neither
        assert F.Front._p_phase_cap_exempt(f, short)
        assert F.Front._p_phase_cap_exempt(f, long_)
        assert not F.Front._p_phase_cap_exempt(f, mid_not_p)

    @pytest.mark.parametrize("kw", [dict(client_gone=True), dict(leg1_done=True), dict(skip_leg1=True)])
    def test_what_does_not_need_p_stays_out(self, loop, kw):
        f = _front(5713)
        assert not F.Front._p_phase_needs_p(f, _real_pending(loop, "l", 20_000, **kw))

    def test_done_future_stays_out(self, loop):
        f = _front(5713)
        p = _real_pending(loop, "l", 20_000)
        p.fut.set_result(None)
        assert not F.Front._p_phase_needs_p(f, p)

    def test_p_only_and_refused_over_x_need_p(self, loop):
        f = _front(5713)
        assert F.Front._p_phase_needs_p(f, _real_pending(loop, "v", 10, p_only=True))
        assert F.Front._p_phase_needs_p(f, _real_pending(loop, "w31", 10, x_requeues=1))

    def test_same_reader_as_the_park(self, loop):
        """No second rule: for every shape the exemption's needs_p equals the park trigger."""
        f = _front(6000)
        for unc, kw in [(5999, {}), (6000, {}), (6001, {}), (1258, dict(x_routed=1205)), (5, dict(p_only=True)),
                        (5, dict(x_requeues=2)), (5, dict(x_deferred=True)), (7000, dict(skip_leg1=True)),
                        (7000, dict(leg1_done=True))]:
            p = _real_pending(loop, "r", unc, **kw)
            assert F.Front._p_phase_needs_p(f, p) == (phase_policy.immediate_park_trigger([p], 6000) is not None)


class TestPoolPlanStillHoldsTheHead:
    def test_the_oldest_exempt_that_does_not_fit_is_not_overtaken(self, loop):
        """Past the cap an exempt LONG that does not fit P's pool waits as the head; no younger SHORT rides past
        it (Prio = age), and it goes once a leg finished. ``cost`` is the est_prompt, ``budget`` P's pool."""
        f = _front(5713)
        q = collections.deque([
            _real_pending(loop, "a", 100, d_eligible=True, est_prompt=200_000),
            _real_pending(loop, "b-long", 9000, est_prompt=150_000),
            _real_pending(loop, "c-short", 20, d_eligible=True, est_prompt=1_000),
        ])
        order = []

        async def main():
            async def one(p):
                order.append(("start", p.rid))
                await asyncio.sleep(0.01)
                order.append(("end", p.rid))
                return p
            await F._p_drain_pool(q, 3, one, lambda p: None, lambda: True, max_dispatch=1,
                                  cost=lambda p: p.est_prompt, budget=262_144, stats={},
                                  cap_exempt=lambda p: F.Front._p_phase_cap_exempt(f, p))
        asyncio.run(main())
        assert not q
        # b-long (150k) does not fit beside a (200k): it holds the head, c-short does NOT overtake it
        assert order.index(("start", "b-long")) < order.index(("start", "c-short"))
        assert order.index(("end", "a")) < order.index(("start", "b-long"))


class TestTheInstrument:
    def _f(self, queue, state="serving"):
        return types.SimpleNamespace(queue=collections.deque(queue), state=state, epoch=7, counters={},
                                     tp_prefill_max_tokens=5713)

    @pytest.mark.parametrize("stalled,dc,k10,state,cause", [
        (True, 0, False, "serving", "stall"), (False, 3, False, "serving", "seat_gate"),
        (False, 0, True, "serving", "k10_rvp_only"), (False, 0, False, "flipping", "state"),
        (False, 0, False, "serving", "other")])
    def test_causes(self, loop, stalled, dc, k10, state, cause):
        f = self._f([_real_pending(loop, "l", 9000)], state=state)
        F.Front._p_end_over_x_left(f, stalled, dc, k10)
        assert f.counters == {"p_end_over_x_left_" + cause: 1}

    @pytest.mark.parametrize("stalled,dc,k10,state,cause", [
        (True, 3, False, "serving", "stall"),            # stall + seat gate: the stall wins (documented precedence)
        (True, 0, True, "flipping", "stall"), (False, 3, True, "flipping", "seat_gate"),
        (False, 0, True, "flipping", "k10_rvp_only")])
    def test_precedence(self, loop, stalled, dc, k10, state, cause):
        f = self._f([_real_pending(loop, "l", 9000)], state=state)
        F.Front._p_end_over_x_left(f, stalled, dc, k10)
        assert f.counters == {"p_end_over_x_left_" + cause: 1}

    def test_silent_when_nothing_needs_p(self, loop):
        f = self._f([_real_pending(loop, "s", 20, d_eligible=True)])
        F.Front._p_end_over_x_left(f, False, 0, False)
        assert f.counters == {}


class TestSeatGateStaysHard:
    """Review A1: the seat gate's cap (``max_dispatch=_dc_cap``) is hard -- no exemption rides past it."""

    def test_for_dc_cap_positive_there_is_no_exemption(self, loop):
        f = _front(5713)
        assert F.Front._p_phase_cap_exempt_for(f, 3) is None
        assert callable(F.Front._p_phase_cap_exempt_for(f, 0))

    def test_pool_leaves_the_over_x_rest_under_the_seat_gate(self, loop):
        f = _front(5713)
        mk = lambda: [_real_pending(loop, f"l{i}", 9000) for i in range(4)] + [_real_pending(loop, "s", 20, d_eligible=True)]
        done, left, _ = _drain(mk(), 2, F.Front._p_phase_cap_exempt_for(f, 2), limit=2)   # _dc_cap = 2 seats
        assert done == ["l0", "l1"] and left == ["l2", "l3", "s"]
        done, left, _ = _drain(mk(), 2, F.Front._p_phase_cap_exempt_for(f, 0), limit=2)   # no seat gate
        assert left == [] and len(done) == 5

    def test_controller_asks_the_one_method_with_the_seat_cap(self):
        assert "cap_exempt=Front._p_phase_cap_exempt_for(self, _dc_cap)" in inspect.getsource(F)


class TestSwitchDoesNotHoldTheDefectFix:
    """Review A3: with FLIPCYCLE H5 (SHORT_RIDES) off the needs-P exemption stays, only the SHORT ride goes."""

    def test_switch_off_over_x_rides_short_does_not(self, loop, monkeypatch):
        from sglang.srt.environ import envs
        f = _front(5713)
        mk = lambda: ([_real_pending(loop, f"l{i}", 9000) for i in range(6)]
                      + [_real_pending(loop, "over-x", 7199), _real_pending(loop, "short", 25, d_eligible=True)])
        monkeypatch.setattr(envs.SGLANG_WEG2_ENABLE_P_PHASE_SHORT_RIDES, "get", lambda: False)
        done, left, _ = _drain(mk(), 6, F.Front._p_phase_cap_exempt_for(f, 0), limit=6)
        assert "over-x" in done and left == ["short"]
        monkeypatch.setattr(envs.SGLANG_WEG2_ENABLE_P_PHASE_SHORT_RIDES, "get", lambda: True)
        done, left, _ = _drain(mk(), 6, F.Front._p_phase_cap_exempt_for(f, 0), limit=6)
        assert left == [] and "short" in done

    def test_switch_off_still_holds_the_seat_gate(self, monkeypatch):
        from sglang.srt.environ import envs
        monkeypatch.setattr(envs.SGLANG_WEG2_ENABLE_P_PHASE_SHORT_RIDES, "get", lambda: False)
        assert F.Front._p_phase_cap_exempt_for(_front(5713), 2) is None


class TestXReadersFollowThePhase:
    """Review A4: riders that need P are not clamped to p_phase_max_requests in the X readers."""

    def _ns(self, k_flip_hist=None):
        ns = types.SimpleNamespace(
            tp_prefill_max_tokens=8000, p_phase_max_requests=6, counters=collections.Counter(),
            _x_k_hist=(1.0, "none"),
            _x_solve_ctx=dict(price=12.0, line={"a_ms": 1728.0, "b_ms": 1.018, "c_ms": 0.0, "n": 40, "n_lo": 182,
                                              "n_hi": 2353, "prefix_med": 0},
                              r_p=None, prefix=0, line_p={"a_ms": 1403.0, "b_ms": 0.068, "c_ms": 0.0, "n": 60,
                                                          "n_lo": 512, "n_hi": 16384}, floor=512, ceiling=12288))
        return ns

    def test_x_for_flip_uses_the_real_rider_count(self):
        ns = self._ns()
        x9 = F.Front._x_for_flip(ns, 9)
        x6 = F.Front._x_for_flip(ns, 6)
        assert 512 < x9 < x6 < 8000          # 133855: k_flip 7..9 used to be clamped to 6 (the same X)

    def test_x_cost_inputs_mean_not_clamped(self):
        ns = types.SimpleNamespace(
            p_phase_max_requests=6, _p_phase_k=collections.deque([9, 9, 9], maxlen=8), _x_cost_seed=None,
            _d_cost_rows=(), _d_cost_all=(), _x_cost_line_of=lambda *a: (None, "none"))
        ns._park_warm_legs_ms = lambda: (None, None, None, "t")
        ns._park_rt_seed = None
        F.Front._x_cost_inputs(ns)
        assert ns._x_k_hist[0] == 9.0            # was min(6, 9) = 6

    def test_curve_verdict_gets_no_max_k(self):
        src = inspect.getsource(F.Front._x_route_of)
        assert "max_k=None" in src and "max_k=float(self.p_phase_max_requests" not in src
        code = lambda fn: "\n".join(l for l in inspect.getsource(fn).splitlines() if not l.lstrip().startswith("#"))
        assert "self.p_phase_max_requests" not in code(F.Front._x_for_flip)
        assert "self.p_phase_max_requests" not in code(F.Front._x_cost_inputs)


class TestWiring:
    def test_controller_passes_the_combined_predicate_and_marks_the_end(self):
        src = inspect.getsource(F)
        assert "cap_exempt=Front._p_phase_cap_exempt_for(self, _dc_cap)" in src
        assert "cap_exempt=(self._p_phase_short_rides" not in src
        i = src.index("Front._p_end_over_x_left(self, bool(self._p_intake_stalled)")
        assert 'await self.flip("P", "D")' in src[i:i + 400]  # the marker sits right before the P->D flip

    def test_marker_is_unconditional_before_the_flip(self):
        """Review A2: for EVERY non-empty queue, not only a stall -- same indentation as the flip, no ``if`` ahead."""
        src = inspect.getsource(F)
        i = src.index("Front._p_end_over_x_left(self, bool(self._p_intake_stalled)")
        line_start = src.rfind("\n", 0, i) + 1
        indent = src[line_start:i]
        assert indent.strip() == ""
        flip = src.index('await self.flip("P", "D")', i)
        assert src[src.rfind("\n", 0, flip) + 1:flip] == indent
        assert "_p_intake_stalled:" not in src[line_start - 120:line_start]

    def test_short_rides_switch_default_on(self):
        from sglang.srt.environ import envs
        assert envs.SGLANG_WEG2_ENABLE_P_PHASE_SHORT_RIDES.get()
