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

* 11:04:50Z epoch 1, ``dispatched=9 cap=6 end=cap queue_left=2``: pdflip-0-1..-6 under the cap, then the SHORTs
  pdflip-0-8 (40), pdflip-0-10 (98) and pdflip-1-11 (25) rode; the LONGs pdflip-0-7 (7199 > X 5713, waited 36 s) and
  pdflip-0-9 (7333) stayed. PARK-IMMEDIATE for pdflip-0-7 4.2 s later, awake_s=0.0.
* 12:16:23Z epoch 31, ``dispatched=9 cap=6 end=cap queue_left=7``: pdflip-28-76 (uncached 1258, routed LONG on the
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

from flliper.srt.pdflip import front as F
from flliper.srt.pdflip import phase_policy


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
        _real_pending(loop, "pdflip-0-1", 7234, **L), _real_pending(loop, "pdflip-0-2", 14029, **L),
        _real_pending(loop, "pdflip-0-3", 7100, **L), _real_pending(loop, "pdflip-0-4", 323, **S),
        _real_pending(loop, "pdflip-0-5", 10823, **L), _real_pending(loop, "pdflip-0-6", 189, **S),
        _real_pending(loop, "pdflip-0-7", 7199, **L), _real_pending(loop, "pdflip-0-8", 40, **S),
        _real_pending(loop, "pdflip-0-9", 7333, **L), _real_pending(loop, "pdflip-0-10", 98, **S),
        _real_pending(loop, "pdflip-1-11", 25, **S),
    ]


class TestRealDoubleFlips:
    def test_1104_old_rule_strands_the_older_long_behind_younger_shorts(self, loop):
        """RED (the h16r reading): short-rides only -- the SHORTs pass the cap, the LONGs stay."""
        f = _front(5713)
        done, left, stats = _drain(_queue_1104(loop), 6, lambda p: F.Front._p_phase_short_rides(f, p), limit=6)
        assert left == ["pdflip-0-7", "pdflip-0-9"]          # the log's queue_left=2
        assert stats["short_rides"] == 3 and stats["dispatched"] == 9  # the log's dispatched=9
        # and the park fires for the head of exactly this queue:
        assert phase_policy.immediate_park_trigger(
            [p for p in _queue_1104(loop) if p.rid in left], 5713).rid == "pdflip-0-7"

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
        assert order == ["pdflip-0-1", "pdflip-0-2", "pdflip-0-3", "pdflip-0-4", "pdflip-0-5", "pdflip-0-6",
                         "pdflip-0-7", "pdflip-0-8", "pdflip-0-9", "pdflip-0-10", "pdflip-1-11"]

    def test_1216_routed_long_on_the_band_floor_is_not_a_short_but_needs_p(self, loop):
        """12:16:23Z pdflip-28-76: uncached 1258 <= live X 7036 (so below X), but routed LONG on X_busy 1205 -- the
        park reads it as needing P (#246b), the SHORT ride predicate does not (d_eligible False)."""
        f = _front(7036)
        p = _real_pending(loop, "pdflip-28-76", 1258, d_eligible=False, x_routed=1205)
        assert not F.Front._p_phase_short_rides(f, p)
        assert phase_policy.immediate_park_trigger([p], 7036) is p
        assert F.Front._p_phase_needs_p(f, p)
        queue = [_real_pending(loop, f"pdflip-x-{i}", 90_000, d_eligible=False) for i in range(6)]
        queue += [p, _real_pending(loop, "pdflip-31-78", 12, d_eligible=True)]
        done, left, _ = _drain(queue, 6, lambda q: F.Front._p_phase_cap_exempt(f, q), limit=6)
        assert "pdflip-28-76" in done and left == []
        done_old, left_old, _ = _drain(queue, 6, lambda q: F.Front._p_phase_short_rides(f, q), limit=6)
        assert left_old == ["pdflip-28-76"]


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
    """After SEAT-AT-HAND-OFF (seatgate dbecd49737) the causes ``seat-gate`` and ``k10-rvp-only`` have no object:
    the gate and K10 are removed, so the instrument names ``stall`` > ``state`` > ``other``."""

    def _f(self, queue, state="serving"):
        return types.SimpleNamespace(queue=collections.deque(queue), state=state, epoch=7, counters={},
                                     tp_prefill_max_tokens=5713)

    @pytest.mark.parametrize("stalled,state,cause", [
        (True, "serving", "stall"), (False, "flipping", "state"), (False, "serving", "other")])
    def test_causes(self, loop, stalled, state, cause):
        f = self._f([_real_pending(loop, "l", 9000)], state=state)
        F.Front._p_end_over_x_left(f, stalled)
        assert f.counters == {"p_end_over_x_left_" + cause: 1}

    def test_precedence_stall_over_state(self, loop):
        """Review A2 (Md): stall and a front leaving "serving" at once -> ``stall`` (P's own refusal comes first)."""
        f = self._f([_real_pending(loop, "l", 9000)], state="flipping")
        F.Front._p_end_over_x_left(f, True)
        assert f.counters == {"p_end_over_x_left_stall": 1}

    def test_silent_when_nothing_needs_p(self, loop):
        f = self._f([_real_pending(loop, "s", 20, d_eligible=True)])
        F.Front._p_end_over_x_left(f, False)
        assert f.counters == {}


class TestOnSeatAtHandOff:
    """Merge with desk/nf-h16q-seatgate-1015 @ dbecd49737: the P phase carries no D seat cap any more, so review A1
    ("the seat gate's cap stays hard against the exemption") has no object -- there is no seat cap to ride past.
    What holds instead: the exemption never depends on D's seats, and is never switched off."""

    def test_the_exemption_is_never_none(self, monkeypatch):
        from flliper.srt.environ import envs
        for on in (True, False):
            monkeypatch.setattr(envs.FLLIPER_PDFLIP_ENABLE_P_PHASE_SHORT_RIDES, "get", lambda on=on: on)
            assert callable(F.Front._p_phase_cap_exempt_pred(_front(5713)))

    def test_no_seat_condition_left_in_the_p_phase(self):
        src = inspect.getsource(F)
        assert "_dc_cap" not in src and "_k10_only" not in src and "_p_phase_cap_exempt_for" not in src
        assert "cap_exempt=Front._p_phase_cap_exempt_pred(self))" in src
        assert "cap_exempt=(self._p_phase_short_rides" not in src  # seatgate's side of the conflict lost


class TestSwitchDoesNotHoldTheDefectFix:
    """Review A3: with FLIPCYCLE H5 (SHORT_RIDES) off the needs-P exemption stays, only the SHORT ride goes."""

    def test_switch_off_over_x_rides_short_does_not(self, loop, monkeypatch):
        from flliper.srt.environ import envs
        f = _front(5713)
        mk = lambda: ([_real_pending(loop, f"l{i}", 9000) for i in range(6)]
                      + [_real_pending(loop, "over-x", 7199), _real_pending(loop, "short", 25, d_eligible=True)])
        monkeypatch.setattr(envs.FLLIPER_PDFLIP_ENABLE_P_PHASE_SHORT_RIDES, "get", lambda: False)
        done, left, _ = _drain(mk(), 6, F.Front._p_phase_cap_exempt_pred(f), limit=6)
        assert "over-x" in done and left == ["short"]
        monkeypatch.setattr(envs.FLLIPER_PDFLIP_ENABLE_P_PHASE_SHORT_RIDES, "get", lambda: True)
        done, left, _ = _drain(mk(), 6, F.Front._p_phase_cap_exempt_pred(f), limit=6)
        assert left == [] and "short" in done

    def test_the_cap_still_binds_what_neither_rides_nor_needs_p(self, loop):
        """The H91 cap is not dead: a request below X that is no SHORT and not routed to P stays for the next
        phase (it does not make the park fire, so leaving it is no double flip)."""
        f = _front(5713)
        q = [_real_pending(loop, f"l{i}", 9000) for i in range(2)] + [_real_pending(loop, "mid", 3000)]
        done, left, _ = _drain(q, 1, F.Front._p_phase_cap_exempt_pred(f), limit=1)
        assert done == ["l0", "l1"] and left == ["mid"]
        assert phase_policy.immediate_park_trigger([q[2]], 5713) is None


class TestReenter:
    """DOUBLE-FLIP 1015 on SEAT-AT-HAND-OFF: a request that needs P and reached the queue after the pool's last
    dispatch keeps P for one more pass instead of a P->D flip the immediate park undoes. One re-entry per rid
    (no timer); a stall or a front leaving "serving" never re-enters."""

    def _f(self, queue, state="serving"):
        return types.SimpleNamespace(queue=collections.deque(queue), state=state, epoch=7, counters={},
                                     tp_prefill_max_tokens=5713)

    def test_a_late_needs_p_arrival_reenters_once(self, loop):
        f = self._f([_real_pending(loop, "late", 9000)])
        assert F.Front._p_phase_reenter(f, False) is True
        assert f.counters == {"p_phase_reenter": 1}
        # the same rid still left after its pass: refused by something else -> flip, the instrument names it
        assert F.Front._p_phase_reenter(f, False) is False
        F.Front._p_end_over_x_left(f, False)
        assert f.counters["p_end_over_x_left_other"] == 1

    def test_a_second_late_arrival_earns_its_own_pass(self, loop):
        f = self._f([_real_pending(loop, "a", 9000)])
        assert F.Front._p_phase_reenter(f, False)
        f.queue.append(_real_pending(loop, "b", 9000))
        assert F.Front._p_phase_reenter(f, False)
        assert not F.Front._p_phase_reenter(f, False)

    @pytest.mark.parametrize("stalled,state", [(True, "serving"), (False, "flipping"), (False, "stopping")])
    def test_stall_or_not_serving_never_reenters(self, loop, stalled, state):
        f = self._f([_real_pending(loop, "l", 9000)], state=state)
        assert F.Front._p_phase_reenter(f, stalled) is False
        assert f.counters == {}

    def test_nothing_that_needs_p_never_reenters(self, loop):
        f = self._f([_real_pending(loop, "s", 20, d_eligible=True), _real_pending(loop, "mid", 3000)])
        assert F.Front._p_phase_reenter(f, False) is False

    def test_wiring_reenter_before_the_marker_and_reset_at_the_flip(self):
        src = inspect.getsource(F)
        i = src.index("if Front._p_phase_reenter(self, bool(self._p_intake_stalled)):")
        blk = src[i:i + 400]
        assert blk.index("continue") < blk.index("Front._p_end_over_x_left(self, bool(self._p_intake_stalled))")
        assert blk.index("self._p_reenter_rids = set()") < blk.index('await self.flip("P", "D")')


class TestCombinationWithSeatAtHandOff:
    """The merge flipqueue x seatgate on the REAL controller (harness of test_pdflip_phase_policy_h91c, D full the
    13:05:10 way of test_pdflip_seat_at_handoff_1015): cap 2, three LONG over X and one SHORT queued while every D
    seat is held and 5 prefilled requests wait for D.

    * seatgate: no seat cap on the P phase -- P prefills although D is full;
    * flipqueue: the H91 cap strands no request that needs P -- all 4 legs before the P->D flip;
    * no double flip: after the P->D flip D is not released again (no D->P) while nothing new arrives, and the
      instrument stays silent (no P-END-OVER-X-LEFT of any cause).
    RED on 3b1e746e0e (seat gate + cap: L3 left), RED on seatgate dbecd49737 alone (cap: L3 left), GREEN merged."""

    def test_full_d_cap_2_all_needs_p_ride_and_no_flip_back(self):
        from test_pdflip_phase_policy_h91c import Harness, _until
        from test_pdflip_seat_at_handoff_1015 import (_fill_d_and_queue_ready, _full_d_harness_kwargs,
                                                     _gens_before_first_release)

        async def body():
            async with Harness(**_full_d_harness_kwargs(p_phase_max_requests=2)) as h:
                h.p.delay = 0.3
                await _fill_d_and_queue_ready(h)
                tasks = []
                for mark, chars in (("L1", 15000), ("L2", 15000), ("L3", 15000), ("S1", 20)):
                    tasks.append(h.post(mark, chars))
                    await asyncio.sleep(0.05)  # arrival order = age order
                assert await _until(lambda: "rpc:release_memory_occupation" in h.p.timeline, 30)
                assert sorted(_gens_before_first_release(h)) == ["gen:L1", "gen:L2", "gen:L3", "gen:S1"], \
                    h.p.timeline
                assert await _until(lambda: h.front.awake == "D" and h.front.state == "serving", 20)
                await asyncio.sleep(1.5)
                assert "rpc:release_memory_occupation" not in h.d.timeline, h.d.timeline  # no D->P flip back
                assert not [k for k in h.front.counters if str(k).startswith("p_end_over_x_left_")], \
                    dict(h.front.counters)
                for t in tasks:
                    t.cancel()

        asyncio.run(body())


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
        assert "cap_exempt=Front._p_phase_cap_exempt_pred(self))" in src
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
        from flliper.srt.environ import envs
        assert envs.FLLIPER_PDFLIP_ENABLE_P_PHASE_SHORT_RIDES.get()
