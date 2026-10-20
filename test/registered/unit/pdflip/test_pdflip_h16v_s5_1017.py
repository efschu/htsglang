"""H16V S5 (17.10. evening): four metal findings of the set scheduler (NF h16u2 boot 1017_133812), each pinned at the
front's call edge (Front._decode_collect / _set_flip_justified / _acquire_short_seat / _arrival_seat_wait) and, for F3,
at the scheduler's settle-release gate (d_norecompute.d_direct_extends):

F1 (SOLL R1)  S counts a session's shared prefix once (15:36:19-25: pdflip-80-912 u=5396 + pdflip-80-913 u=5534, common 24064
              of a 20032 credit -> real new tokens 5396 + 1502, the log summed 11107 -> REROUTE_P);
F2 (R2a/R3)   the X of a set of k members carries the k-1 further P fixed costs (read from the curve file);
F3 (R2/R7)    a resumed tail under halbX is D's own extend, no flip (W50-REROUTE store_short_after_flip d_extent 77..2054);
F4 (R5)       every wait for a seat/KV is named seat_wait (never R2); three waiters of one tick do not take ONE seat.
Mutants: test/registered/unit/pdflip/mutants_h16v_s5_1017.sh -- each must kill at least one test here."""
from __future__ import annotations

import asyncio
import logging
import os
import re
import time
import types

import numpy as np
import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from test_pdflip_set_scheduler_1017 import X, _d_began, _fx, _on, _verdicts  # noqa: E402
from test_pdflip_arrival_seat_rule_0929 import _front, _pending  # noqa: E402

from flliper.srt.pdflip import phase_policy as pp  # noqa: E402
from flliper.srt.pdflip import x_curves as xc  # noqa: E402

M = pp.SetMember


def _session(f, rid, sess, ids, depth):
    f.__dict__.setdefault("_sess_by_rid", {})[rid] = sess
    f.__dict__.setdefault("_set_ids", {})[rid] = np.asarray(ids, dtype=np.int32)
    f.__dict__.setdefault("_set_depth", {})[rid] = depth


# ------------------------------------------------------------------------------------------------ F1

def test_f1_pure_shared_prefix_counts_once():
    common = lambda a, b: {("912", "913"): 24064}.get((a, b), 0)  # noqa: E731
    got = pp.shared_prefix_discount([("912", 5396, 20032, 1.0, "s"), ("913", 5534, 20032, 2.0, "s")], common)
    assert got == {"912": 5396, "913": 5534 - (24064 - 20032)}
    # another session / no session: no discount
    assert pp.shared_prefix_discount([("912", 5396, 20032, 1.0, "s"), ("913", 5534, 20032, 2.0, "t")], common)["913"] == 5534
    assert pp.shared_prefix_discount([("912", 5396, 20032, 1.0, ""), ("913", 5534, 20032, 2.0, "")], common)["913"] == 5534
    # the overlap never exceeds what is uncached for B; a later member is judged against EVERY earlier one
    c2 = lambda a, b: 10 ** 9  # noqa: E731
    assert pp.shared_prefix_discount([("a", 100, 0, 1.0, "s"), ("b", 80, 0, 2.0, "s")], c2)["b"] == 0


def test_f1_the_log_case_of_15_36_s_is_counted_once_at_the_front_edge(monkeypatch, caplog):
    """pdflip-80-912 (5396) + pdflip-80-913 (5534), both credit 20032, common prefix 24064: P brings 5396 + 1502 = 6898
    (S_P). H16V review A5: the DECISION S is the D-side sum of the own figures (10930), D's in-batch dedupe is not
    guaranteed; the deduplicated figure is reported as S_P only."""
    _on(monkeypatch)
    f = _fx()
    f.tp_prefill_max_tokens = 6185
    ids912 = list(range(25428))
    ids913 = list(range(24064)) + [-i - 1 for i in range(25566 - 24064)]
    _session(f, "pdflip-80-912", "a4c8b097ac", ids912, 20032)
    _session(f, "pdflip-80-913", "a4c8b097ac", ids913, 20032)
    t0 = time.time()
    st = f._set_st()
    st["shorts"]["pdflip-80-912"] = (t0, 5396, 6185)
    st["shorts"]["pdflip-80-913"] = (t0 + 1.5, 5534, 6185)
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        f._decode_collect(t0 + 1.5)
    (line,) = _verdicts(caplog)
    assert "S=10930 " in line and "members=pdflip-80-912:5396,pdflip-80-913:5534" in line, line
    assert "classes=halb,halb" in line and "prefix-shared=4032" in line and "S_P=6898" in line, line


def test_f1_a_shared_prefix_does_not_lower_the_deciding_s_a5(monkeypatch, caplog):
    """H16V review A5: X=6185, raw 4000 + 4100 = 8100 flips; the second brings 2000 shared tokens (S_P = 6100 < X) --
    P counts them once, but D is not shown to (schedule_policy's in-batch dedupe only runs for lpm/dfs-weight and a
    cached match <= 32 tokens), so the D-side S decides: the same session flips like another one."""
    _on(monkeypatch)
    f = _fx()
    f.tp_prefill_max_tokens = 6185
    _session(f, "a", "s", list(range(30000)), 20000)
    _session(f, "b", "s", list(range(22000)) + [-1] * 2100, 20000)
    t0 = time.time()
    st = f._set_st()
    st["shorts"]["a"] = (t0, 4000, 6185)
    st["shorts"]["b"] = (t0 + 0.5, 4100, 6185)
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        assert f._decode_collect(t0 + 0.6) == "P"
    (line,) = _verdicts(caplog)
    assert "S=8100 " in line and "S_P=6100" in line, line
    # another session: no shared prefix, S = 8100 >= X -> flip
    f2 = _fx()
    f2.tp_prefill_max_tokens = 6185
    _session(f2, "a", "s", list(range(30000)), 20000)
    _session(f2, "b", "t", list(range(22000)) + [-1] * 2100, 20000)
    st2 = f2._set_st()
    st2["shorts"]["a"] = (t0, 4000, 6185)
    st2["shorts"]["b"] = (t0 + 0.5, 4100, 6185)
    assert f2._decode_collect(t0 + 0.6) == "P"


@pytest.mark.usefixtures("decider_off")  # EZ-S2a: describes the S1 code (reachable until S4)


def test_f1_the_queue_sum_flip_decides_on_the_d_side_sum_too_a5(monkeypatch):
    _on(monkeypatch)
    f = _fx(running=())
    f.tp_prefill_max_tokens = 6185
    _session(f, "a", "s", list(range(30000)), 20000)
    _session(f, "b", "s", list(range(22000)) + [-1] * 2100, 20000)
    now = time.time()
    f.queue = [_pending("a", 4000, now - 9.0), _pending("b", 4100, now - 8.0)]
    assert f._set_flip_justified(now) is True and f.counters["flip_held_under_x"] == 0


# ------------------------------------------------------------------------------------------------ F2

def _curves(p_fixed_ms=900.0):
    d = xc.PrefillCurve(rows=tuple(xc.CurveRow(depth=dd, n=(256, 65536), ms=(200.0 + 256.0, 200.0 + 65536.0))
                                   for dd in (0, 262144)), chunk_tokens=0, instrument="synthetic")
    p = xc.PrefillCurve(rows=tuple(xc.CurveRow(depth=dd, n=(256, 65536),
                                               ms=(p_fixed_ms + 25.6, p_fixed_ms + 6553.6)) for dd in (0, 262144)),
                        chunk_tokens=0, instrument="synthetic")
    ident = xc.CurveIdentity(model="m", form="f", hardware="h", created="2026-10-17T00:00:00Z", source="synthetic")
    return xc.XCurves(format=xc.X_CURVES_FORMAT, identity=ident, p_curve=p, d_curve=d,
                      flip_price=xc.FlipPrice(depth=(0,), seconds=(4.0,), attribution="none"))


def test_f2_p_fixed_cost_is_read_from_the_curve_file():
    assert xc.p_fixed_ms(_curves(900.0), 1000.0) == pytest.approx(900.0, abs=0.5)
    assert xc.p_fixed_ms(_curves(0.0), 1000.0) == pytest.approx(0.0, abs=0.5)


def test_f2_x_of_a_set_grows_by_the_further_members_fixed_costs():
    """D = 200 + n, P = 900 + 0.1 n, price 4 s: X(k=1) = 4700/0.9 = 5222; with one more P forward (900 ms) 6222."""
    c = _curves(900.0)
    x1 = xc.x_verdict_from_curves(depth=1000, open_requests=1, curves=c).x
    x2 = xc.x_verdict_from_curves(depth=1000, open_requests=1, curves=c, p_extra_ms=xc.p_fixed_ms(c, 1000.0)).x
    assert x1 == 5222 and 6215 <= x2 <= 6225


def test_f2_x_of_k_is_info_only_the_decision_stays_on_the_own_x_a2(monkeypatch, caplog):
    """H16V review A2 (SOLL R1/R3/R2a): two 2700 members, S = 5400 >= X(k=1) = 5222 -> flip. X(k=2) = 6222 (P fixed
    cost of the second forward) is NOT a threshold of the SOLL; it is reported in the why (``xk=``) only."""
    _on(monkeypatch)
    f = _fx()
    f.tp_prefill_max_tokens = 12000
    f._x_setup = types.SimpleNamespace(curves=_curves(900.0), beyond="clamp")
    x1 = f._x_set_of("a", 1000)
    assert x1 == 5222
    f._x_set_of("b", 1000)
    t0 = time.time()
    st = f._set_st()
    st["shorts"]["a"] = (t0, 2700, x1)
    st["shorts"]["b"] = (t0 + 0.3, 2700, x1)
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        assert f._decode_collect(t0 + 0.4) == "P"
    (line,) = _verdicts(caplog)
    assert re.search(r"S=5400 X=5222 ", line) and "classes=halb,halb" in line, line
    assert re.search(r"xk=62\d\d", line) and "verdict=FLIP_P" in line, line


def test_f2_a_set_over_the_set_x_still_flips(monkeypatch):
    _on(monkeypatch)
    f = _fx()
    f.tp_prefill_max_tokens = 12000
    f._x_setup = types.SimpleNamespace(curves=_curves(900.0), beyond="clamp")
    x1 = f._x_set_of("a", 1000)
    f._x_set_of("b", 1000)
    t0 = time.time()
    st = f._set_st()
    st["shorts"]["a"] = (t0, 3300, x1)
    st["shorts"]["b"] = (t0 + 0.3, 3300, x1)      # S = 6600 >= 6222
    assert f._decode_collect(t0 + 0.4) == "P"


# ------------------------------------------------------------------------------------------------ F3

def _sched(static_x=12288):
    return types.SimpleNamespace(server_args=types.SimpleNamespace(tp_prefill_max_tokens=static_x), page_size=64,
                                 ps=types.SimpleNamespace(tp_rank=0), pdflip_d_parked=[])


def _resumed(rem, front_x=6265, halfx=3132):
    """A resumed request (it decoded). H16V review A3: the front carries X AND halbX; None = nothing carried."""
    n = 75149
    return types.SimpleNamespace(rid="pdflip-18-250", full_untruncated_fill_ids=list(range(n)),
                                 _pdflip_store_delivered=n - rem, stream=True, multimodal_inputs=None,
                                 output_ids=[1, 2, 3], weg2_front_x=front_x, weg2_front_priced=3427,
                                 weg2_front_halfx=halfx)


@pytest.mark.parametrize("rem,expect", [(77, True), (693, True), (1563, True), (2054, True), (2931, True),
                                        (3133, False), (6000, False)])
def test_f3_the_h16u2_resumed_tails_are_d_s_own_extend_not_a_flip(rem, expect):
    """The five p_only flips under X of h16u2 (d_extent 77 / 693 / 1563 / 1926 / 2054, plus 2931): kurz = under half of
    the carried front X 6265 (halbX 3132, carried by the front); from halbX up the tail is RESUME-VIA-P (R3/R4 decide
    that flip)."""
    from flliper.srt.pdflip import d_norecompute as dn

    assert dn.d_direct_extends(_sched(), _resumed(rem), rem) is expect


def test_a3_without_a_carried_front_x_d_never_extends_a_resumed_tail(monkeypatch):
    """H16V review A3 (SOLL section 2 "X nie ein fester Wert"): no carried X / halbX -> RESUME-VIA-P for every tail,
    there is no fixed substitute bound (the former 0.5 x static X = 6144)."""
    from flliper.srt.pdflip import d_norecompute as dn

    for rem in (77, 693, 2054, 3000, 6000):
        assert dn.d_direct_extends(_sched(), _resumed(rem, front_x=None, halfx=None), rem) is False
        assert dn.d_direct_extends(_sched(), _resumed(rem, front_x=6265, halfx=None), rem) is False   # X alone: no halbX
        assert dn.d_direct_extends(_sched(), _resumed(rem, front_x=None, halfx=3132), rem) is False


def test_a3_with_a_carried_halfx_d_never_extends_over_it_and_ignores_the_rank_env(monkeypatch):
    """The front's halbX is the bound -- whatever F_halb a rank's environment says (R11: no rank-local decision)."""
    from flliper.srt.pdflip import d_norecompute as dn

    monkeypatch.setenv("FLLIPER_PDFLIP_HALF_X_FRAC", "0.99")
    for carried in (1000, 2500, 3132):
        for rem in range(1, 4200, 37):
            got = dn.d_direct_extends(_sched(), _resumed(rem, front_x=6265, halfx=carried), rem)
            assert got is (rem < carried), (carried, rem)


def test_f3_the_scheduler_gate_calls_the_extend_before_the_divert(monkeypatch):
    """The settle-release gate, statement for statement: a kurz tail stays in D's release, no W50-REROUTE record."""
    from flliper.srt.pdflip import d_norecompute as dn

    diverted = []
    req = _resumed(77)
    rem = dn.d_would_compute(req)
    assert rem == 77
    keep = rem > 0 and not dn.d_direct_extends(_sched(), req, rem) and diverted.append(req)
    assert not keep and diverted == []


# ------------------------------------------------------------------------------------------------ F4

def test_f4_a_kv_wait_is_named_seat_wait_not_r2(monkeypatch, caplog):
    """16:03:17-38 pdflip-96-1180: a free seat (seats_free>0 at entry), KV short for ~0.7 s, then granted: the wait is
    named seat_wait (it was why=R2 with waited_s=20.6)."""
    _on(monkeypatch)
    f = _front(running=["a"], n=6, kv={"available": 1000, "evictable": 0})
    f.tp_prefill_max_tokens = X
    f.t_awake = time.time() - 60.0
    box = {"kv": {"available": 1000, "evictable": 0}}

    async def kv():
        return box["kv"]

    f._arrival_seat_kv_reading = kv

    async def run():
        t = asyncio.ensure_future(f._acquire_short_seat("k", 400, uncached=400))
        await asyncio.sleep(0.8)
        box["kv"] = {"available": 400000, "evictable": 0}
        return await asyncio.wait_for(t, 4.0)

    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        assert asyncio.run(run()) is not None
    (line,) = _verdicts(caplog)
    assert "verdict=D_NOW why=seat_wait" in line, line


def test_f4_three_waiters_one_free_seat_take_one_seat_not_three(monkeypatch):
    """16:03:32,957: pdflip-96-1178/1179/1180 were all granted at taken=5 n=6 (one free seat). The seat count is read
    again after the KV await: one is granted, the others wait for the next seat."""
    _on(monkeypatch)
    f = _front(running=["r1", "r2", "r3", "r4", "r5"], n=6, kv={"available": 400000, "evictable": 0})
    f.tp_prefill_max_tokens = X
    f.t_awake = time.time() - 60.0

    async def slow_kv():
        await asyncio.sleep(0.2)             # the await between "free" and the grant
        return {"available": 400000, "evictable": 0}

    f._arrival_seat_kv_reading = slow_kv

    async def run():
        ts = [asyncio.ensure_future(f._arrival_seat_wait(r, 100, None, uncached=300, t_arrive=time.time() + i * 0.001))
              for i, r in enumerate(("w1", "w2", "w3"))]
        await asyncio.sleep(1.2)
        granted = sorted(r for r, t in zip(("w1", "w2", "w3"), ts) if t.done() and t.result() is True)
        for t in ts:
            t.cancel()
        return granted

    assert len(asyncio.run(run())) == 1


def test_f4_a_semaphore_wait_after_the_grant_is_named_too(monkeypatch, caplog):
    _on(monkeypatch)
    f = _fx(running=())
    f.tp_prefill_max_tokens = X

    async def run():
        for _ in range(6):
            await f._d_seat.acquire()        # the front's Seats are all held although D's ledger is empty
        t = asyncio.ensure_future(f._acquire_short_seat("k", 400, uncached=400))
        await asyncio.sleep(0.7)
        f._d_seat.release()
        return await asyncio.wait_for(t, 3.0)

    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        assert asyncio.run(run()) is not None
    (line,) = _verdicts(caplog)
    assert "verdict=D_NOW why=seat_wait" in line, line


# ------------------------------------------------------------------------- the user case 15:36Z with S6's curves

def test_user_case_15_36z_is_reroute_p_by_the_soll_not_by_a_moved_threshold(monkeypatch, caplog):
    """NF h16u2 15:36:19-25 with the h16v curve file S6 measured from that boot: pdflip-80-912 (u=5396) and pdflip-80-913
    (u=5534, same session, common prefix 24064 of the 20032 credit) + the running D prefill of 177 tokens (pdflip-80-910).
    H16V review A2: X(k) is no threshold; the verdict is decided against the real X of the members (curve X(k=1) 5586
    at that depth). The log summed S=11107 > X -> REROUTE_P (R6); with the shared prefix counted once (S_P = 5396 +
    1502 + 177 = 7075) it is > X as well -> REROUTE_P either way. That is the result the SOLL gives; the P-side cost
    (P batches the members one by one, 2-3 s gaps) is an engine defect for its own ticket, not a reason to move X."""
    from flliper.srt.pdflip import x_curves as _xc

    _on(monkeypatch)
    f = _fx()
    f.tp_prefill_max_tokens = 12288
    curves = _xc.load(os.path.join(os.path.dirname(__file__), "fixtures", "x_curves.nf-h16v_s6_1017.json"))
    f._x_setup = types.SimpleNamespace(curves=curves, beyond="clamp")
    x1 = f._x_set_of("pdflip-80-912", 20032)
    f._x_set_of("pdflip-80-913", 20032)
    assert 5500 <= x1 <= 5700
    ids912 = list(range(25428))
    ids913 = list(range(24064)) + [-i - 1 for i in range(25566 - 24064)]
    _session(f, "pdflip-80-912", "a4c8b097ac", ids912, 20032)
    _session(f, "pdflip-80-913", "a4c8b097ac", ids913, 20032)
    f._d_pf_book().grant(rid="pdflip-80-910", tokens=177, now=time.time())
    f._d_pf_book().enter_leg2(rid="pdflip-80-910")
    f.groups["D"].outstanding["pdflip-80-910"] = 1.0
    _d_began(f, "pdflip-80-910")   # S49 (SOLL R6, user 18.10.): the case's D prefill was RUNNING
    t0 = time.time()
    st = f._set_st()
    st["shorts"]["pdflip-80-912"] = (t0, 5396, x1)
    st["shorts"]["pdflip-80-913"] = (t0 + 1.5, 5534, x1)
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        assert f._decode_collect(t0 + 1.6) == "P"
    (line,) = _verdicts(caplog)
    m = re.search(r"S=(\d+) X=(\d+) ", line)
    assert m and int(m.group(1)) == 5396 + 5534 + 177 and int(m.group(2)) == x1, line   # X = the members' own X
    assert "verdict=REROUTE_P" in line and "S_P=7075" in line, line
    assert re.search(r"xk=7[56]\d\d", line), line   # X(k=2) ~7620 is reported, never decided on
    # F1 alone: with only the deduplicated S_P = 7075 against the same real X the SOLL gives REROUTE_P as well
    assert pp.set_verdict([M("a", 5396, 0.0), M("b", 1502, 1.5)], x_tokens=x1, half_frac=0.5, t_half_s=4.5, now=1.6,
                          t_open=0.0, d_decoding=True, seats_free=6, d_prefill_running_uncached=177).verdict == "REROUTE_P"
