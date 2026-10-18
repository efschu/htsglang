"""PRIO-AS-AGE 1014 (user decision 14.10.2026 18:27Z): ``priority`` acts as AGE, it no longer parks lower requests.

User, verbatim: "priority 1 immer älter ... in einer priority class ist es aber natürlich wieder altersabhängig wer
bedient wird".  The order key everywhere is (priority desc, arrival asc) -- ``pdflip/prio_age.py``; every wait clock keeps
the REAL arrival; the lane FLOOR of 1008 (LANE-PREEMPT, LANE-RAISE/RESUME, LANE-WAITER-HOLD, the P-leg hold, D's lane
park) is never entered while ``FLLIPER_PDFLIP_PRIO_AS_AGE`` is on (default); a P-bound request with priority > 0 ends a
DECODE-COLLECT window at once.

Pinned here (hermetic: no GPU, fake groups, every async body under ``asyncio.run``):

 1. the key: p1 before p0 despite a younger arrival; inside a priority by age; p1000 before p10; the switch and its
    dependency on FLLIPER_PDFLIP_LANES;
 2. D: the queue / park / barrier / retraction order; SEAT-AGE displacement parks EXACTLY ONE -- the youngest of the
    LOWEST priority -- and a p0 waiter never displaces a p1 decode; the deferred park placed by the key;
 3. the KV rule "oldest first, a younger shorter one moves up": D's ``kv_displace_would_fit`` counts only the
    runners younger by the key; the front's AGE PLAN / KV-DISPLACE victims likewise; YOUNGEST-PARK never a higher
    priority than the oldest waiter;
 4. the front: the P drain pool takes the highest priority first while the queue (and so every clock reading its
    head) stays in arrival order; the D line / the immediate-park head / PBOUND-FLIP-NOW by the key;
 5. clocks unchanged: ``t_arrive`` / the waiter stamps are never rewritten, DECODE-COLLECT's ``t_open`` and the
    drain's oldest SHORT read the real oldest arrival;
 6. mode ON: no LANE-PREEMPT, no floor RPC, no park RPC, floor 0, the D and P lane endpoints refuse; mode OFF: the
    old LANE-PREEMPT;
 7. flip urgency: a P-bound p1 ends the DECODE-COLLECT window and satisfies FLIP-ECONOMICS; a p0 does not;
 8. RANK AGREEMENT (raenge-nie-uneins): every rank forms the same order from the same queue; a MUTANT rank that
    ignores the priority forms another order, and the #791b ballot digest of the queue head STOPS the group.
"""
from __future__ import annotations

import asyncio
import collections
import contextlib
import logging
import os
import sys
import types
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest  # noqa: E402

from flliper.srt.managers.scheduler_components import weight_updater as _wu  # noqa: E402,F401
from flliper.srt.pdflip import arrival_seat_rule as ASR  # noqa: E402
from flliper.srt.pdflip import d_park_runtime as DPR  # noqa: E402
from flliper.srt.pdflip import d_seats as DS  # noqa: E402
from flliper.srt.pdflip import front as F  # noqa: E402
from flliper.srt.pdflip import lane_ctl as LC  # noqa: E402
from flliper.srt.pdflip import lanes as LN  # noqa: E402
from flliper.srt.pdflip import lanes_p  # noqa: E402
from flliper.srt.pdflip import phase_policy as PP  # noqa: E402
from flliper.srt.pdflip import prio_age as PA  # noqa: E402
from flliper.srt.pdflip import seat_age as SA  # noqa: E402
from flliper.srt.pdflip.front import Front  # noqa: E402
from test_pdflip_lanes_l4_1008 import _front, _marks, _note  # noqa: E402

X = 4096
NOW = 1_000_000.0


def R(n, e=12):
    return f"pdflip-{e}-{n}"


@contextlib.contextmanager
def _mode(prio_as_age: bool, lanes: bool = True):
    """The two switches, restored on every exit."""
    keys = ("FLLIPER_PDFLIP_LANES", PA.ENV)
    old = {k: os.environ.get(k) for k in keys}
    os.environ["FLLIPER_PDFLIP_LANES"] = "1" if lanes else "0"
    os.environ[PA.ENV] = "1" if prio_as_age else "0"
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@pytest.fixture(autouse=True)
def _default_mode():
    """Every test runs with the mode at its production default (lanes on, prio-as-age on) unless it says otherwise."""
    with _mode(True):
        yield


def _req(rid, prio=None, site=None, kv=10, seq=None):
    r = types.SimpleNamespace(rid=rid, priority=prio, origin_input_ids=[0] * kv, output_ids=[])
    if seq is not None:
        r.kv_arrival_seq = seq
    if site:
        DS.mark_parked(r, site, now=1.0)
    return r


# ====================================================================== 1. the key
def test_p1_is_older_than_p0_despite_a_younger_arrival_and_inside_a_priority_age_decides():
    assert PA.key(1, 50.0) < PA.key(0, 10.0)        # p1 arrived later, still older
    assert PA.key(0, 10.0) < PA.key(0, 11.0)        # inside p0: real age
    assert PA.key(3, 10.0) < PA.key(3, 11.0)        # inside p3: real age
    assert PA.key(1000, 99.0) < PA.key(10, 1.0)     # p1000 before p10
    assert PA.key(10, 99.0) < PA.key(0, 1.0)
    reqs = [_req("a", 0, seq=1), _req("b", 1000, seq=5), _req("c", 10, seq=2), _req("d", 0, seq=0),
            _req("e", 10, seq=9), _req("f", None, seq=3)]
    got = sorted(reqs, key=lambda r: PA.key(PA.lane_of_req(r), r.kv_arrival_seq))
    assert [r.rid for r in got] == ["b", "c", "e", "d", "a", "f"]


def test_lane_sorted_is_stable_inside_a_priority_and_identity_without_priorities():
    reqs = [_req("a", 0), _req("b", 1), _req("c", 0), _req("d", 1), _req("e", 2)]
    assert [r.rid for r in PA.lane_sorted(reqs)] == ["e", "b", "d", "a", "c"]
    plain = [_req("x"), _req("y"), _req("z")]
    assert PA.lane_sorted(plain) == plain
    assert PA.head_index(reqs) == 4 and PA.head_index(plain) == 0


def test_the_switch_defaults_on_needs_lanes_and_off_restores_the_floor_mode():
    for k in ("FLLIPER_PDFLIP_LANES", PA.ENV):
        os.environ.pop(k, None)
    assert PA.enabled() and not PA.floor_mode() and not LC.enabled()
    with _mode(False):
        assert not PA.enabled() and PA.floor_mode() and LC.enabled()
        assert PA.key(5, 1.0) == (0, 1.0)            # off: the key is the arrival alone
    with _mode(True, lanes=False):
        assert not PA.enabled() and not PA.floor_mode() and not LC.enabled()
    assert PA.enabled({"FLLIPER_PDFLIP_LANES": "1"}) and not PA.enabled({PA.ENV: "0"})
    assert not PA.enabled({"FLLIPER_PDFLIP_LANES": "0"})


def test_malformed_priorities_are_priority_zero():
    for raw in (None, -3, "x", 1.5, True, [1]):
        assert PA.lane(raw) == 0
    assert PA.lane("7") == 7 and PA.lane(7.0) == 7 and PA.lane(1000) == 1000


# ====================================================================== 2. D: order, barrier, retraction
def test_seat_age_by_age_plan_seats_and_is_older_follow_the_key():
    lanes = {R(9): 1, R(2): 0, R(5): 0, R(7): 2}
    lf = lanes.get
    assert SA.by_age([R(2), R(9), R(5), R(7)], lane_of=lf) == [R(7), R(9), R(2), R(5)]
    assert SA.by_age([R(2), R(9), R(5)]) == [R(2), R(5), R(9)]                      # no lane_of: arrival
    assert SA.plan_seats([(R(2), 0), (R(9), 0), (R(5), 0)], 2, lane_of=lf) == [R(9), R(2)]
    assert SA.is_older(R(9), R(2), lane_of=lf) and not SA.is_older(R(2), R(9), lane_of=lf)
    with _mode(False):
        assert SA.by_age([R(2), R(9), R(5)], lane_of=lf) == [R(2), R(5), R(9)]       # mode off: arrival


def test_order_waiting_puts_a_higher_priority_before_a_parked_lower_one_and_keeps_the_rest_stable():
    parked0 = _req(R(3), 0, DS.SITE_PRESSURE)
    a0, b1, c0, d1 = _req(R(4), 0), _req(R(8), 1), _req(R(5), 0), _req(R(9), 1)
    got = DS.order_waiting([a0, b1, parked0, c0, d1])
    assert [r.rid for r in got] == [R(8), R(9), R(3), R(4), R(5)]
    with _mode(False):
        assert [r.rid for r in DS.order_waiting([a0, b1, parked0, c0, d1])] == [R(3), R(4), R(8), R(5), R(9)]


def test_park_running_order_and_the_barrier_pass_a_higher_priority_newcomer():
    assert [r.rid for r in DS.park_running_order([_req(R(2), 0), _req(R(9), 1), _req(R(4), 0)])] == [R(9), R(2), R(4)]
    gate = DS.admission_gate([_req(R(7), 0, DS.SITE_PRESSURE), _req(R(3), 0), _req(R(9), 0), _req(R(11), 1)],
                             running=[_req(R(5), 0)])
    assert gate.oldest_parked_age == 7.0
    assert gate.skip(_req(R(3), 0)) is None                    # older than the parked one
    assert gate.skip(_req(R(9), 0)) == "pdflip_d_park_first"     # younger, same priority: waits
    assert gate.skip(_req(R(11), 1)) is None                   # p1 is older than every p0: passes
    with _mode(False):
        g2 = DS.admission_gate([_req(R(7), 0, DS.SITE_PRESSURE)], running=[_req(R(5), 0)])
        assert g2.skip(_req(R(11), 1)) == "pdflip_d_park_first"


def test_a_parked_p0_does_not_resume_past_a_live_p1_but_a_parked_p1_does():
    # pressure-parked p0 (age 3) while an OLDER-by-key p1 (age 9) runs: blocked ("the older one is live")
    gate = DS.admission_gate([_req(R(3), 0, DS.SITE_PRESSURE)], running=[_req(R(9), 1)])
    assert R(3) in gate.blocked
    gate = DS.admission_gate([_req(R(9), 1, DS.SITE_PRESSURE)], running=[_req(R(3), 0)])
    assert R(9) not in gate.blocked
    with _mode(False):
        assert R(3) not in DS.admission_gate([_req(R(3), 0, DS.SITE_PRESSURE)], running=[_req(R(9), 1)]).blocked


def test_retraction_victim_is_the_youngest_of_the_lowest_priority():
    reqs = [_req(R(9), 1), _req(R(2), 0), _req(R(5), 0), _req(R(12), 1)]
    order = DS.retraction_order(reqs, spec_active=False)
    assert reqs[order[-1]].rid == R(5)                          # youngest p0, not the younger p1s
    assert [reqs[i].rid for i in order] == [R(9), R(12), R(2), R(5)]
    with _mode(False):
        assert reqs[DS.retraction_order(reqs, spec_active=False)[-1]].rid == R(12)


class _DSide:
    """The D-side harness of test_pdflip_seat_age_244_0927 (fake batch + scheduler)."""

    @staticmethod
    def sched(running, waiting, cap=3, avail=1 << 30):
        batch = types.SimpleNamespace(reqs=list(running), released=[], spec_algorithm=None)
        batch.release_req = lambda idx, rem, sa, retain=False: batch.released.append((batch.reqs[idx].rid, retain))

        def filt(keep_indices):
            batch.reqs = [batch.reqs[i] for i in keep_indices]

        batch.filter_batch = filt
        sched = types.SimpleNamespace(waiting_queue=list(waiting), server_args=types.SimpleNamespace(
            max_running_requests=cap), running_batch=batch,
            token_to_kv_pool_allocator=types.SimpleNamespace(available_size=lambda: avail))
        sched._add_request_to_queue = lambda req, is_retracted=False: sched.waiting_queue.append(req)
        return sched, batch

    @staticmethod
    def run(fn):
        with mock.patch.object(DS, "d_flip_park_active", lambda: True), \
             mock.patch.object(DPR, "seat_cap", lambda sched: None):
            return fn()


def test_seat_age_displacement_parks_exactly_one_the_youngest_of_the_lowest_priority():
    # seats full: p1 R(4), p0 R(5), p0 R(6) run; a p1 arrives (R(20), the youngest ARRIVAL of all)
    sched, batch = _DSide.sched([_req(R(4), 1), _req(R(5), 0), _req(R(6), 0)], [_req(R(20), 1)])
    got = _DSide.run(lambda: DPR.displace_for_age(sched, batch))
    assert got == R(6)                                           # youngest p0, never the p1 runner
    assert batch.released == [(R(6), True)]                      # exactly ONE
    assert [r.rid for r in batch.reqs] == [R(4), R(5)]
    # the next pass with the seat free does not displace a second one
    assert _DSide.run(lambda: DPR.displace_for_age(sched, batch)) is None
    assert batch.released == [(R(6), True)]


def test_a_p0_waiter_never_displaces_a_p1_decode_however_old_it_is():
    sched, batch = _DSide.sched([_req(R(7), 1), _req(R(8), 1), _req(R(9), 1)], [_req(R(1), 0)])
    assert _DSide.run(lambda: DPR.displace_for_age(sched, batch)) is None
    assert batch.released == []
    with _mode(False):  # floor mode (1008): the lane pool -- a lower lane's waiter takes no higher lane's seat either
        sched, batch = _DSide.sched([_req(R(7), 1), _req(R(8), 1), _req(R(9), 1)], [_req(R(1), 0)])
        assert _DSide.run(lambda: DPR.displace_for_age(sched, batch)) is None
    with _mode(True, lanes=False):  # lanes off: priority unread, arrival only -> the youngest arrival goes
        sched, batch = _DSide.sched([_req(R(7), 1), _req(R(8), 1), _req(R(9), 1)], [_req(R(1), 0)])
        assert _DSide.run(lambda: DPR.displace_for_age(sched, batch)) == R(9)


def test_deferred_parked_placed_by_the_key():
    a, b = _req(R(4), 0, DS.SITE_FLIP), _req(R(8), 0)
    new_old, new_hi = _req(R(3), 0), _req(R(11), 1)
    sched = types.SimpleNamespace(waiting_queue=[new_hi, a, new_old, b], pdflip_dormant_hold=[], pdflip_park_defer={R(4)})
    DPR._apply_park_defer(sched)
    assert [r.rid for r in sched.waiting_queue] == [R(11), R(3), R(4), R(8)]


# ====================================================================== 3. the KV rule
def test_d_kv_rule_counts_only_runners_younger_by_the_key():
    # the older waiter W (p0, age 2) needs 100 tokens; free KV 0; runners: A p1 age 9 (1000 tokens, OLDER by the
    # key), B p0 age 5 (10 tokens, younger). Mode on: only B may leave -> 10 < 100 -> no fit. Mode off: A+B -> fit.
    W = _req(R(2), 0, kv=100)
    A, B = _req(R(9), 1, kv=1000), _req(R(5), 0, kv=10)
    sched, _b = _DSide.sched([A, B], [W], avail=0)
    sched.tree_cache = None
    with mock.patch.object(DPR, "_evictable_reading", lambda t: 0), \
         mock.patch.object(DPR, "backup_wall_up", lambda t: False):
        assert DPR.kv_displace_would_fit(sched, R(2), [A, B]) is False
        with _mode(False):
            assert DPR.kv_displace_would_fit(sched, R(2), [A, B]) is True


def test_front_age_plan_and_kv_victim_take_the_youngest_of_the_lowest_priority():
    running = ["hi-old", "lo-a", "lo-b"]
    arrivals = {"hi-old": 50.0, "lo-a": 10.0, "lo-b": 20.0}
    tokens = {"hi-old": 5000, "lo-a": 100, "lo-b": 100}
    lanes = {"hi-old": 1, "lo-a": 0, "lo-b": 0}
    # a p1 head that arrived at 60: every p0 decode is younger by the key, the p1 one (arrived 50) is older
    assert ASR.displace_plan(60.0, False, 0, running, arrivals, tokens, lane_of=lanes, head_lane=1) == ["lo-b"]
    assert ASR.kv_displace_victim(60.0, running, arrivals, tokens, 150, lane_of=lanes, head_lane=1) == "lo-b"
    # a p0 head that arrived at 5: the p1 decode is OLDER by the key -- never its victim, even for the deficit
    assert ASR.displace_plan(5.0, True, 300, running, arrivals, tokens, lane_of=lanes, head_lane=0) is None
    assert ASR.kv_displace_victim(5.0, running, arrivals, tokens, 300, lane_of=lanes, head_lane=0) is None
    # no lanes given: the old arrival rule (the p1 decode arrived after 5, it is younger)
    assert ASR.kv_displace_victim(5.0, running, arrivals, tokens, 300) == "hi-old"


def test_youngest_park_never_parks_a_higher_priority_than_the_oldest_waiter():
    admit = {"hi": 30.0, "lo": 10.0}
    lanes = {"hi": 1, "lo": 0}
    assert ASR.youngest_running(admit, ["hi", "lo"]) == "hi"                               # old rule
    assert ASR.youngest_running(admit, ["hi", "lo"], lane_of=lanes) == "lo"               # lowest priority
    assert ASR.youngest_running(admit, ["hi"], lane_of=lanes, than=PA.key(0, 1.0)) is None  # p0 waiter
    assert ASR.youngest_running(admit, ["hi"], lane_of=lanes, than=PA.key(1, 1.0)) == "hi"  # p1 waiter


# ====================================================================== 4./5. front order, clocks unchanged
def _p(f, rid, lane, t, est_prompt=5000, est_uncached=5000, **kw):
    """A front Pending (default LONG: 5000 > X uncached) with its lane noted."""
    loop = asyncio.get_event_loop()
    p = F.Pending(rid, "/generate", {"rid": rid}, "x", t, loop.create_future(), est_prompt=est_prompt,
                  est_uncached=est_uncached, lane=lane, **kw)
    _note(f, rid, lane)
    return p


def test_the_p_drain_pool_takes_the_highest_priority_first_and_the_queue_head_stays_the_real_oldest():
    async def main():
        f = _front("P")
        q = collections.deque([_p(f, "a0", 0, NOW - 30), _p(f, "b0", 0, NOW - 20), _p(f, "c1", 1, NOW - 1),
                               _p(f, "d0", 0, NOW - 10), _p(f, "e2", 2, NOW - 0.5)])
        heads, order = [], []

        async def one(p):
            heads.append(LC.clock_t(q[0]) if q else None)    # the clock that reads the queue head
            return p

        await F._p_drain_pool(q, 1, one, lambda p: order.append(p.rid), lambda: True,
                              pick=lambda qq, pred: Front._prio_pick(f, qq, pred))
        return order, heads

    order, heads = asyncio.run(main())
    assert order == ["e2", "c1", "a0", "b0", "d0"]
    # while e2 and c1 went first, the head of the queue -- what every wait clock reads -- was a0, the real oldest
    assert heads[0] == NOW - 30 and heads[1] == NOW - 30


def test_the_p_drain_pool_without_pick_is_the_old_pool():
    async def main():
        f = _front("P")
        q = collections.deque([_p(f, "a0", 0, NOW - 30), _p(f, "c1", 1, NOW - 1)])
        order = []

        async def one(p):
            return p

        await F._p_drain_pool(q, 1, one, lambda p: order.append(p.rid), lambda: True)
        return order

    assert asyncio.run(main()) == ["a0", "c1"]


def test_prio_pick_respects_the_cap_exempt_predicate_and_keeps_queue_position_inside_a_priority():
    async def main():
        f = _front("P")
        q = collections.deque([_p(f, "a0", 0, NOW - 9), _p(f, "s0", 0, NOW - 99), _p(f, "s1", 1, NOW - 1),
                               _p(f, "t1", 1, NOW - 50)])
        return (Front._prio_pick(f, q, None), Front._prio_pick(f, q, lambda p: p.rid.startswith("s")),
                Front._prio_pick(f, q, lambda p: p.rid == "zz"))

    head, exempt, none = asyncio.run(main())
    assert head == 2            # s1: the FIRST p1 in queue order (t1 is older but stands behind it -- queue position)
    assert exempt == 2 and none is None


def test_front_sorts_keep_t_arrive_and_the_waiter_stamps():
    async def main():
        f = _front("D")
        ps = [_p(f, "a0", 0, NOW - 30), _p(f, "c1", 1, NOW - 1), _p(f, "b0", 0, NOW - 20)]
        before = [p.t_arrive for p in ps]
        got = Front._prio_sorted(f, ps)
        st = f._asr_st()
        st["waiters"].update({"a0": NOW - 30, "c1": NOW - 1})
        head = min(st["waiters"].items(), key=lambda kv: Front._prio_key(f, kv[0], kv[1]))[0]
        return [p.rid for p in got], [p.t_arrive for p in ps] == before, head, Front._asr_waiter_clocks(f, st)

    order, unchanged, head, clocks = asyncio.run(main())
    assert order == ["c1", "a0", "b0"] and unchanged
    assert head == "c1"
    assert sorted(clocks) == [NOW - 30, NOW - 1]                  # the stamps are the real arrivals


def test_set_wait_opens_at_the_members_arrival_and_a_p1_long_waits_t_halb_too(caplog, monkeypatch):
    # SET SCHEDULER (SOLL v2 R4): the wait is the set's T_halb at EVERY priority (S45 18.10.: the old "priority > 0
    # needing P ends it at once" was a planner derivation, not the user's word)
    monkeypatch.setenv("FLLIPER_PDFLIP_HALF_FLIP_S", "4.5")
    caplog.set_level(logging.INFO, logger="pdflip.front")

    async def main(lane_of_long):
        f = _front("D")
        f.awake = "D"
        now = NOW
        f.t_awake = NOW - 100.0
        with mock.patch.object(Front, "_dc_d_busy", lambda self: True), \
             mock.patch.object(Front, "_dc_decoding", lambda self: True), \
             mock.patch.object(Front, "_dc_carried", lambda self: 0), \
             mock.patch.object(Front, "_set_seats_free", lambda self: 6):
            st = f._set_st()
            old_short = _p(f, "s0", 0, now - 3, est_uncached=10)
            big = _p(f, "L", lane_of_long, now - 1, est_uncached=3 * X, est_prompt=3 * X)
            route = f._decode_collect(now, [old_short, big])
            return route, st.get("t_open"), f.counters["decode_collect_prio_seen"]

    route, t_open, n = asyncio.run(main(1))
    assert route == "hold" and n == 1                             # seen and logged, but it waits (R4)
    assert t_open == NOW - 1
    route0, t_open0, n0 = asyncio.run(main(0))
    assert route0 == "hold" and n0 == 0
    assert t_open0 == NOW - 1                                     # the set's clock: its first member's arrival
    with _mode(False):
        assert asyncio.run(main(1))[0] == "hold"


def test_flip_economics_latch_is_satisfied_neither_by_a_p1_nor_by_a_p0():
    # S45 (SOLL v2 R4/R7): priority orders, it is no flip reason of its own
    async def main(lane):
        f = _front("D")
        f.flip_min_work_tokens = 10 ** 9                          # far above the queue: the latch would hold
        f.queue.append(_p(f, "L", lane, NOW - 1, est_uncached=2 * X, est_prompt=2 * X))
        return f._flip_economics_ok(False)

    assert asyncio.run(main(1)) is False
    assert asyncio.run(main(0)) is False
    with _mode(False):
        assert asyncio.run(main(1)) is False


def test_the_d_line_and_the_seat_rotate_plan_use_the_key():
    async def main():
        f = _front("D")
        ps = [_p(f, R(2), 0, NOW - 30), _p(f, R(9), 1, NOW - 1), _p(f, R(5), 0, NOW - 20)]
        line = SA.by_age(ps, lane_of=lambda q: Front._prio_lane(f, q))
        return [p.rid for p in line]

    assert asyncio.run(main()) == [R(9), R(2), R(5)]


# ====================================================================== 6. mode ON: no lane floor machinery
def test_mode_on_a_p1_during_p0_decodes_parks_nothing_sends_no_rpc_and_keeps_the_floor_at_0(caplog):
    caplog.set_level(logging.INFO, logger="pdflip.front")

    async def main():
        f = _front("D")
        D = f.groups["D"]
        for i in range(3):
            D.outstanding[f"d{i}"] = NOW
            _note(f, f"d{i}", 0)
        payload = {"text": "x", "priority": 1}
        Front._lane_note(f, "hi", payload)                     # the field is still READ and normalised
        out = await Front._lane_arrive(f, SimpleNamespace(), "hi", payload)
        await Front._lane_reconcile(f, "tick")
        return f, out, payload

    f, out, payload = asyncio.run(main())
    assert out is None and payload["priority"] == 1
    assert f._lane_state().lane_for("hi") == 1                  # carried: the front knows the lane
    ls = f._lane_state()
    assert (ls.lane_floor, ls.lane_epoch) == (0, 0)
    assert f.rpc.calls == []                                     # no floor RPC, no park RPC
    for mark in (LN.MARK_PREEMPT, LC.MARK_RAISE, LN.MARK_D_PARK_PARK, "PDFLIP LANE-WAITER-HOLD"):
        assert _marks(caplog, mark) == []
    assert "_lane_ctl_obj" not in f.__dict__ or not f._lane_ctl_obj.held
    assert f.counters["lane_preempt"] == 0


def test_mode_off_the_same_arrival_is_the_old_lane_preempt(caplog):
    caplog.set_level(logging.INFO, logger="pdflip.front")

    async def main():
        f = _front("D")
        D = f.groups["D"]
        for i in range(2):
            D.outstanding[f"d{i}"] = NOW
            _note(f, f"d{i}", 0)
        _note(f, "hi", 1)
        await LC.arrive(f, SimpleNamespace(), "hi", lambda r: False, None)
        return f

    with _mode(False):
        f = asyncio.run(main())
    assert f._lane_state().lane_floor == 1
    assert _marks(caplog, LN.MARK_PREEMPT)
    assert f.rpc.paths(PP.PARK_PATH)


def test_mode_on_the_d_and_p_lane_endpoints_refuse_and_the_p_side_takes_no_floor():
    sched = SimpleNamespace(waiting_queue=[], pdflip_d_parked=[])
    out = DPR.lane_floor(sched, SimpleNamespace(floor=1, epoch=1))
    assert out.success is False and "PRIO_AS_AGE" in out.message
    assert lanes_p.active() is False
    lanes_p.on_rpc(sched, SimpleNamespace(floor=1, epoch=1))
    assert getattr(sched, lanes_p.STATE_ATTR, None) is None
    assert DPR.hold_queued(sched) == []


def test_mode_on_the_waiter_hold_and_the_p_leg_hold_are_never_entered():
    async def main():
        f = _front("D")
        _note(f, "lo", 0)
        f._lane_state().lane_floor = 1          # even a stale floor is not read: the hooks are not entered
        st = f._asr_st()
        st["waiters"]["lo"] = NOW
        held = LC.waiter_hold(f, st, "lo") if LC.enabled() else False
        p = _p(f, "leg", 0, NOW)
        return held, LC.lane_held_leg(f, p), "lo" in st["waiters"]

    held, leg_held, still = asyncio.run(main())
    assert held is False and leg_held is False and still


# ====================================================================== 8. rank agreement + mutant
def _sched_rank(queue):
    return SimpleNamespace(waiting_queue=list(queue), enable_priority_scheduling=False)


def _rank_queue(prio_field=True):
    out = []
    for i, p in enumerate([0, 0, 1, 0, 1000, 10, 0]):
        out.append(_req(R(i + 1), p if prio_field else None, seq=i))
    return out


def _form(sched):
    from flliper.srt.managers.scheduler import Scheduler

    Scheduler._pdflip_prio_age_order(sched)
    return [r.rid for r in sched.waiting_queue]


def _ballot(order_per_rank):
    from flliper.srt.managers import prefetch_ballot as PB

    payloads = [PB.build_prefetch_ballot_payload(o, {}) for o in order_per_rank]
    reduced = [min(col) for col in zip(*payloads)]
    return PB, reduced


def test_every_rank_forms_the_same_order_from_the_same_queue_and_the_ballot_agrees(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "D")
    orders = [_form(_sched_rank(_rank_queue())) for _ in range(3)]
    assert orders[0] == [R(5), R(6), R(3), R(1), R(2), R(4), R(7)]   # p1000, p10, p1, then p0 by arrival
    assert orders[0] == orders[1] == orders[2]
    PB, reduced = _ballot(orders)
    assert PB.unpack_prefetch_ballot(reduced[: PB.PREFETCH_BALLOT_SLOTS + 2], orders[0]) is not None


def test_mutant_a_rank_that_ignores_the_priority_forms_another_order_and_the_group_stops(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "D")
    honest = [_form(_sched_rank(_rank_queue())) for _ in range(2)]
    # mutant 1: one rank never received the field (priority stripped on the way to it)
    mutant = _form(_sched_rank(_rank_queue(prio_field=False)))
    assert mutant != honest[0]
    PB, reduced = _ballot(honest + [mutant])
    with pytest.raises(PB.PrefetchBallotDigestMismatch):
        PB.unpack_prefetch_ballot(reduced[: PB.PREFETCH_BALLOT_SLOTS + 2], honest[0], rank=0)
    # mutant 2: one rank runs with the mode off (config drift)
    monkeypatch.setenv(PA.ENV, "0")
    mutant2 = _form(_sched_rank(_rank_queue()))
    monkeypatch.setenv(PA.ENV, "1")
    assert mutant2 != honest[0]
    PB, reduced = _ballot(honest + [mutant2])
    with pytest.raises(PB.PrefetchBallotDigestMismatch):
        PB.unpack_prefetch_ballot(reduced[: PB.PREFETCH_BALLOT_SLOTS + 2], honest[0], rank=1)


def test_the_scheduler_sort_is_off_outside_a_pdflip_group_and_under_upstream_priority_scheduling(monkeypatch):
    monkeypatch.delenv("FLLIPER_PDFLIP_GROUP", raising=False)
    q = _rank_queue()
    assert _form(_sched_rank(q)) == [r.rid for r in q]
    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "P")
    s = _sched_rank(q)
    s.enable_priority_scheduling = True
    assert _form(s) == [r.rid for r in q]
    s = _sched_rank(q)
    assert _form(s)[0] == R(5)                                     # group P: the P batch order follows it too


def test_the_scheduler_pass_calls_the_sort_right_after_the_group_head_order():
    import inspect

    from flliper.srt.managers.scheduler import Scheduler

    src = inspect.getsource(Scheduler._get_new_batch_prefill_raw)
    i = src.index("self._apply_uniform_head_order(_head_inputs)")
    j = src.index("self._pdflip_prio_age_order()")
    k = src.index("self._pdflip_d_park_admission(running_batch)")
    assert i < j < k
