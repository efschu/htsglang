"""X-K-FLIP (30.09., NF y5a 0110f8e132 front): the flip price is amortised over
THIS flip's riders, not over the boot's mean P phase.

Metal (y5a front.log): ``X COST-LINE RE-SOLVE ... k=2.43 [live:mean-of-19]``
took the live X from 4096 down to 1440-1900 -- the mean requests per P phase
is inflated by the dmatrix 6-request bursts -- and 8 flips were fired by ONE
agent request of 2087-2940 new tokens (ARRIVAL-SEAT flip_now / PARK-IMMEDIATE
cause=over-x: weg2-8-13, 30-46, 33-48, 40-60, 42-62, 48-74, 60-88, 62-89),
each carrying k=1-2 in reality. On y5a's lines (D a=1728 ms b=1.018 ms/tok,
P a=1403 ms b=0.068 ms/tok, round trip ~4.3 s) k=1 puts X near 4200.

Pinned: the X in force is the lone request's (k=1) under the switch; the
boot's mean is printed display-only; an arrival is routed on the X of the
flip it would take (1 + requests queued for P now); off = the mean, exactly.
"""

from __future__ import annotations

import asyncio
import collections
import inspect
import logging
import pathlib
import types

import pytest

from sglang.srt.environ import envs
from sglang.srt.weg2 import front as F
from sglang.srt.weg2 import phase_policy as pp

REPO = pathlib.Path(__file__).resolve().parents[4]

LINE_D = {"a_ms": 1728.0, "b_ms": 1.018, "c_ms": 0.0, "n": 40, "n_lo": 182, "n_hi": 2353, "prefix_med": 0}
LINE_P = {"a_ms": 1403.0, "b_ms": 0.068, "c_ms": 0.0, "n": 60, "n_lo": 512, "n_hi": 16384}
K_HIST = [2, 3, 2, 3, 2, 3, 2, 3, 2, 3]            # mean 2.5, the dmatrix-inflated k


def _front(queue=()):
    seed_d = object()
    ns = types.SimpleNamespace(
        tp_prefill_max_tokens=4096, flip_min_work_tokens=4096, _x_min_work_follows=True,
        x_ceiling_tokens=12288, counters=collections.Counter(), _x_cost_last_missing=[],
        _d_cost_rows=collections.deque(), _d_cost_all=collections.deque(), _x_cost_seed=seed_d,
        _p_cost_rows=collections.deque(), _p_cost_all=collections.deque(), _x_cost_seed_p=None,
        _p_phase_k=collections.deque(K_HIST, maxlen=32), _x_samples={"r_p": collections.deque()},
        _park_rt_seed=None, p_phase_max_requests=6, queue=list(queue),
    )
    ns._park_warm_legs_ms = lambda: (1800.0, 2300.0, 200.0, "test")        # 4.3 s round trip
    ns._x_cost_line_of = lambda rows, every, seed: ((LINE_D, "live:test") if seed is seed_d
                                                    else (LINE_P, "live:test-p"))
    ns._note_x_cost_line = lambda *a: None
    ns._x_cost_inputs = lambda: F.Front._x_cost_inputs(ns)
    ns._x_for_flip = lambda k, rid="", site="": F.Front._x_for_flip(ns, k, rid, site)
    ns._x_riders = lambda: F.Front._x_riders(ns)
    return ns


def _queued(rid, uncached, done=False):
    fut = asyncio.get_event_loop_policy().new_event_loop().create_future()
    if done:
        fut.set_result(None)
    return types.SimpleNamespace(rid=rid, est_uncached=uncached, d_direct=False, fut=fut)


def _route(ns, uncached):
    # the route's own rule (RELEASE-INTEG role split): riders count only when the flip is decided
    rides = F.Front._x_flip_decided(ns)
    x = ns._x_for_flip(1 + ns._x_riders() if rides else 1, "weg2-x", "route")
    return F.serviceable_route(uncached, uncached, x, 373536), x


def test_the_model_reproduces_y5a():
    x1, _ = pp.solve_x_cost_line(price_s=4.3, k=1.0, line=LINE_D, r_p=None, prefix_tokens=0, line_p=LINE_P)
    x25, _ = pp.solve_x_cost_line(price_s=4.3, k=2.5, line=LINE_D, r_p=None, prefix_tokens=0, line_p=LINE_P)
    assert 4100 < x1 < 4300                      # the lone request's break-even
    assert 1400 < x25 < 1600                     # the live X y5a ran on (1440-1900)


def test_switch_default_on():
    assert envs.SGLANG_WEG2_X_K_FLIP.get() is True


def test_switch_off_keeps_the_mean(caplog):
    ns = _front()
    # RELEASE-INTEG: the boot-mean form is PDFLIP-X's excursion price off as well (pd-flip: on)
    with envs.SGLANG_WEG2_X_K_FLIP.override(False), envs.SGLANG_WEG2_X_EXCURSION_PRICE.override(False), \
            envs.SGLANG_WEG2_ENABLE_X_COST_LINE.override(True), envs.SGLANG_WEG2_X_COST_MAX_STEP.override(10.0), \
            caplog.at_level(logging.INFO):
        x = F.Front._resolve_x_cost_line(ns)
    assert 1400 < x < 1600                       # y5a's live X: a 2.9k request goes to P
    assert "k=2.50 [live:mean-of-10]" in next(m for m in caplog.messages if "RE-SOLVE" in m)
    assert _route(ns, 2900)[0] == "long"


def test_a_lone_2_9k_request_goes_to_d(caplog):
    ns = _front()
    # RELEASE-INTEG: the X-K-FLIP k_src line is pinned with PDFLIP-X's excursion price off (on, the
    # same lone X is named 'not-divided(...)' -- test_pdflip_x_excursion_1002)
    with envs.SGLANG_WEG2_X_K_FLIP.override(True), envs.SGLANG_WEG2_X_EXCURSION_PRICE.override(False), \
            envs.SGLANG_WEG2_ENABLE_X_COST_LINE.override(True), \
            envs.SGLANG_WEG2_X_COST_MAX_STEP.override(10.0), caplog.at_level(logging.INFO):
        x = F.Front._resolve_x_cost_line(ns)
        route, x_route = _route(ns, 2900)
    assert 4100 < x < 4300 and x_route == x
    assert route == "short"                       # base: long, a flip for one request
    msg = next(m for m in caplog.messages if "RE-SOLVE" in m)
    assert "k=1.00 [flip:1(lone) hist=2.50[live:mean-of-10] display-only]" in msg


def test_the_same_request_with_three_waiting_for_p_may_flip(caplog):
    riders = [_queued(f"weg2-9-{i}", 6000) for i in range(3)]
    ns = _front(queue=riders)
    with envs.SGLANG_WEG2_X_K_FLIP.override(True), envs.SGLANG_WEG2_ENABLE_X_COST_LINE.override(True), \
            envs.SGLANG_WEG2_X_COST_MAX_STEP.override(10.0), caplog.at_level(logging.INFO):
        F.Front._resolve_x_cost_line(ns)
        route, x_route = _route(ns, 2900)
    assert route == "long" and x_route < 2900     # k_flip = 4: the round trip is shared
    line = next(m for m in caplog.messages if "X-FLIP-K" in m)
    assert "k_flip=4 (1 + 3 queued for P)" in line and "site=route" in line
    assert ns.counters["x_k_flip_lowered"] == 1


def test_riders_count_only_live_queued_work():
    ns = _front(queue=[_queued("a", 5000), _queued("b", 5000, done=True), _queued("c", 0)])
    handed = _queued("d", 5000)
    handed.d_direct = True
    ns.queue.append(handed)
    assert ns._x_riders() == 1


def test_riders_never_raise_x_above_the_lone_request():
    ns = _front()
    with envs.SGLANG_WEG2_X_K_FLIP.override(True), envs.SGLANG_WEG2_ENABLE_X_COST_LINE.override(True), \
            envs.SGLANG_WEG2_X_COST_MAX_STEP.override(10.0):
        x1 = F.Front._resolve_x_cost_line(ns)
        assert ns._x_for_flip(1) == x1
        assert ns._x_for_flip(6) <= ns._x_for_flip(2) <= x1


def test_the_arrival_is_routed_on_the_flip_x():
    src = inspect.getsource(F.Front)
    # X-CURVES 1006: the arrival goes through _x_route_of, whose non-curve path
    # is exactly X-K-FLIP's _x_for_flip(k_flip, rid, "route") with the role-split k_flip
    assert 'return self._x_for_flip(k_flip, rid, "route")' in inspect.getsource(F.Front._x_route_of)
    i = src.index("_x_arrival = Front._x_route_of(\n"
                  "                self, rid=rid, depth=store_span, k_flip=1 + self._x_riders() if _x_rides else 1)")
    j = src.index("route = serviceable_route(remainder, carrier_est,\n                                  _x_arrival,", i)
    k = src.index("x_route = _x_arrival", j)
    assert i < j < k


# ---- RELEASE-INTEG 1002: X-K-FLIP x PDFLIP-X split by role (coordinator decision 02.10.) --------

def test_a_trigger_with_k_riders_gets_the_same_x_as_alone():
    """(a) Nothing queued needs P (k waiting SHORTs, each <= X): this arrival would TRIGGER the
    flip and pays the whole round trip -- the X in force, the same as alone."""
    shorts = [_queued(f"weg2-s-{i}", 1500) for i in range(3)]
    ns_k, ns_1 = _front(queue=shorts), _front()
    with envs.SGLANG_WEG2_X_K_FLIP.override(True), envs.SGLANG_WEG2_ENABLE_X_COST_LINE.override(True), \
            envs.SGLANG_WEG2_X_COST_MAX_STEP.override(10.0):
        F.Front._resolve_x_cost_line(ns_k)
        F.Front._resolve_x_cost_line(ns_1)
        assert not F.Front._x_flip_decided(ns_k) and ns_k._x_riders() == 3
        assert _route(ns_k, 2900)[1] == _route(ns_1, 2900)[1] == int(ns_1.tp_prefill_max_tokens)


def test_a_rider_of_a_decided_flip_gets_the_amortised_x():
    """(b) A queued request already needs P (the flip is decided): the arrival rides it, the flip
    cost is sunk -- X-K-FLIP's amortised X, below the lone X."""
    riders = [_queued(f"weg2-9-{i}", 6000) for i in range(3)]
    ns = _front(queue=riders)
    with envs.SGLANG_WEG2_X_K_FLIP.override(True), envs.SGLANG_WEG2_ENABLE_X_COST_LINE.override(True), \
            envs.SGLANG_WEG2_X_COST_MAX_STEP.override(10.0):
        x1 = F.Front._resolve_x_cost_line(ns)
        assert F.Front._x_flip_decided(ns)
        assert _route(ns, 2900)[1] == ns._x_for_flip(4) < x1


def test_a_running_flip_to_p_makes_every_arrival_a_rider():
    ns = _front(queue=[_queued("q", 1500)])
    ns.state, ns._flip_dst = "flipping", "P"
    assert F.Front._x_flip_decided(ns)
    ns.state, ns._flip_dst = "flipping", "D"
    assert not F.Front._x_flip_decided(ns)
