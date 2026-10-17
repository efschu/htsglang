"""SET SCHEDULER (SCHEDULER-SOLL v2, 17.10.): R1-R11 -- the pure verdict (phase_policy.set_verdict) and, above all, the
front's call edge (Front._decode_collect / _decode_collect_short / _acquire_short_seat / _arrival_seat_step /
_p_batch_takes_d_prefill). X = 6000, halbX = 3000, T_halb = 4.5 s are the SOLL section 5 numbers.

Every rule has a test that goes through the front (the mutants in test/registered/unit/weg2/mutants_set_scheduler_1017.sh
break the call edge, not the helper, and each must kill at least one test here)."""
from __future__ import annotations

import asyncio
import logging
import os
import re
import time

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from test_weg2_arrival_seat_rule_0929 import _front, _pending  # noqa: E402

from sglang.srt.managers.scheduler_components import weight_updater as _wu  # noqa: E402,F401
from sglang.srt.weg2 import phase_policy as pp  # noqa: E402

X, HALF, T = 6000, 0.5, 4.5
M = pp.SetMember


def _v(members, *, t_open=0.0, now=0.0, dec=True, seats=6, run=0, new=0, frac=HALF, t_half=T, x=X):
    return pp.set_verdict(members, x_tokens=x, half_frac=frac, t_half_s=t_half, now=now, t_open=t_open,
                          d_decoding=dec, seats_free=seats, d_prefill_running_uncached=run, new_since_last=new)


# ------------------------------------------------------------------------------------------- pure: SOLL section 5

def test_classes_at_arrival():
    c = lambda u, p=False: pp.classify_arrival(u, X, HALF, p)  # noqa: E731
    assert c(300) == "kurz" and c(2999) == "kurz"
    assert c(3000) == "halb" and c(5999) == "halb"          # halbX <= u < X
    assert c(6000) == "gross" and c(9000) == "gross"        # u >= X
    assert c(300, True) == "p_only" and c(9000, True) == "p_only"   # R9: vision wins over the length
    assert pp.classify_arrival(2500, X, 0.4, False) == "halb"       # F_halb is a flag


def test_ex1_kurz_never_reaches_the_set():
    assert pp.classify_arrival(300, X, HALF, False) == "kurz"
    assert _v([]).verdict == "COLLECT" and _v([]).why == "empty"


def test_ex2_halb_alone_goes_to_d_after_t_halb():
    m = [M("a", 3500, 0.0)]
    assert _v(m, now=0.0).verdict == "COLLECT"
    assert _v(m, now=4.4).verdict == "COLLECT"
    v = _v(m, now=4.5)
    assert v.verdict == "D_AFTER_TIMER" and v.s_tokens == 3500 and v.half_x == 3000


def test_ex3_two_halb_inside_t_halb_flip_at_once():
    v = _v([M("a", 3500, 0.0), M("b", 3200, 2.0)], now=2.0)
    assert v.verdict == "FLIP_P" and v.s_tokens == 6700


def test_ex4_kurz_between_is_no_member_and_the_sum_flips():
    # 300 went to D at once (R2); the set is 3500 + 3100 (planner's correction 17.10.: 2800 would be kurz, under halbX)
    v = _v([M("a", 3500, 0.0), M("c", 3100, 3.0)], now=3.0)
    assert v.verdict == "FLIP_P" and v.s_tokens == 6600
    assert pp.classify_arrival(2800, X, HALF, False) == "kurz"


def test_ex5_gross_waits_for_company_or_t_half():
    g = M("g", 9000, 0.0)
    assert _v([g], now=0.0).verdict == "COLLECT"
    assert _v([g], now=4.4).verdict == "COLLECT"
    assert _v([g], now=4.5).verdict == "FLIP_P"                       # R4: the timer ends
    assert _v([g, M("h", 3100, 1.0)], now=1.0).verdict == "FLIP_P"    # R4: company inside T_halb


def test_ex6_no_decode_decides_at_once():
    assert _v([M("g", 9000, 0.0)], dec=False).verdict == "FLIP_P"       # R8
    assert _v([M("a", 3500, 0.0)], dec=False).verdict == "D_AFTER_TIMER"


def test_ex7_running_d_prefill_is_rerouted_when_s_passes_x():
    v = _v([M("a", 5000, 0.0)], run=4000)
    assert v.verdict == "REROUTE_P" and v.s_tokens == 9000
    # S == X exactly is no reroute (R6 says "über X"): R3's S >= X flips by the normal rule
    assert _v([M("a", 3000, 0.0)], run=3000).verdict == "FLIP_P"
    assert _v([M("a", 3000, 0.0)], run=2900).verdict == "COLLECT"


def test_ex8_no_free_seats_is_park_no_seat():
    two = [M("a", 3500, 0.0), M("b", 3200, 1.0)]
    assert _v(two, now=1.0, seats=0).verdict == "PARK_NO_SEAT"
    assert _v(two, now=1.0, seats=1).verdict == "PARK_NO_SEAT"        # fewer seats than members
    assert _v(two, now=1.0, seats=2).verdict == "FLIP_P"


# ------------------------------------------------------------------------------------------- pure: borders

def test_timer_starts_at_the_first_member_only():
    # F_halb 0.4: two halb members stay under X (2500 + 2500). t_open is the FIRST member's stamp: the second
    # arrival at 4.0 does not move it, so the timer is over at 4.6
    two = [M("a", 2500, 0.0), M("b", 2500, 4.0)]
    assert _v(two, t_open=0.0, now=4.6, frac=0.4).verdict == "D_AFTER_TIMER"
    assert _v(two, t_open=0.0, now=4.4, frac=0.4).verdict == "COLLECT"
    # t_open None: the oldest member's arrival
    assert _v(two, t_open=None, now=4.6, frac=0.4).verdict == "D_AFTER_TIMER"


def test_a_member_that_dies_leaves_the_rest_to_the_same_clock():
    both = [M("g", 9000, 0.0), M("h", 3100, 1.0)]
    assert _v(both, now=1.0).verdict == "FLIP_P"
    assert _v([both[0]], now=1.5).verdict == "COLLECT"                 # h cancelled: g waits on, the timer is g's
    assert _v([both[0]], now=4.5).verdict == "FLIP_P"


def test_a_seat_that_frees_during_t_halb_turns_park_into_flip():
    two = [M("a", 3500, 0.0), M("b", 3200, 1.0)]
    assert _v(two, now=1.0, seats=0).verdict == "PARK_NO_SEAT"
    assert _v(two, now=1.5, seats=2).verdict == "FLIP_P"


def test_decode_ending_during_the_timer_decides_now():
    a = [M("a", 3500, 0.0)]
    assert _v(a, now=1.0, dec=True).verdict == "COLLECT"
    assert _v(a, now=1.0, dec=False).verdict == "D_AFTER_TIMER"        # R8: D has nothing to protect
    assert _v([M("g", 9000, 0.0)], now=1.0, dec=False).verdict == "FLIP_P"


def test_p_only_counts_as_gross_even_when_small():
    p = [M("v", 400, 0.0, p_only=True)]
    assert _v(p, now=1.0).verdict == "COLLECT"
    assert _v(p, now=4.5).verdict == "FLIP_P"
    assert _v(p, now=0.0, dec=False).verdict == "FLIP_P"
    assert _v(p + [M("a", 3500, 0.5)], now=0.5).verdict == "FLIP_P"


def test_t_half_zero_decides_at_once():
    assert _v([M("a", 3500, 0.0)], t_half=0.0).verdict == "D_AFTER_TIMER"
    assert _v([M("g", 9000, 0.0)], t_half=0.0).verdict == "FLIP_P"


def test_set_sum_not_single_request_decides():
    # R1: two halb requests whose sum is X flip, one 5999 alone does not
    assert _v([M("a", 3000, 0.0), M("b", 3000, 0.1)], now=0.1).verdict == "FLIP_P"
    assert _v([M("a", 5999, 0.0)], now=0.1).verdict == "COLLECT"


# ------------------------------------------------------------------------------------------- the front edge

def _on(monkeypatch, t_half="4.5", frac="0.5"):
    monkeypatch.setenv("SGLANG_WEG2_ENABLE_ARRIVAL_SEAT_RULE", "1")
    monkeypatch.delenv("SGLANG_WEG2_ENABLE_ARRIVAL_SEAT_AGE_PLAN", raising=False)
    monkeypatch.delenv("SGLANG_WEG2_PBOUND_FLIP_NOW", raising=False)
    monkeypatch.delenv("SGLANG_WEG2_ENABLE_DECODE_COLLECT_PREFILL_BUSY", raising=False)
    monkeypatch.setenv("SGLANG_WEG2_HALF_FLIP_S", t_half)
    monkeypatch.setenv("SGLANG_WEG2_HALF_X_FRAC", frac)


def _fx(running=("dec",), n=6):
    f = _front(running=list(running), n=n, kv={"available": 400000, "evictable": 0})
    f.tp_prefill_max_tokens = X
    f.t_awake = time.time() - 60.0
    return f


def _verdicts(caplog):
    return [r.getMessage() for r in caplog.records if "WEG2 SET-VERDICT" in r.getMessage()]


def test_envs_defaults_and_the_old_window_is_off():
    from sglang.srt.environ import envs
    assert envs.SGLANG_WEG2_HALF_X_FRAC.default == 0.5
    assert envs.SGLANG_WEG2_HALF_FLIP_S.default == 4.5
    assert envs.SGLANG_WEG2_DECODE_COLLECT_WINDOW_S.default == 0.0


def test_r2_kurz_goes_to_d_at_once_while_a_set_collects(monkeypatch, caplog):
    _on(monkeypatch)
    f = _fx()
    f.queue = [_pending("big", 9000, time.time())]  # a set is open: a big request waits T_halb
    assert f._decode_collect(time.time()) == "hold"
    t0 = time.time()
    with caplog.at_level(logging.INFO, logger="weg2.front"):
        seat = asyncio.run(f._acquire_short_seat("k", 300, uncached=300))
    assert seat is not None and time.time() - t0 < 1.0, "R2: no wait for a kurz request"
    (line,) = [m for m in _verdicts(caplog) if "classes=kurz" in m]
    assert "verdict=D_NOW why=R2" in line and "n=1" in line and "members=k:300" in line
    assert f.counters["set_d_now"] == 1 and f._set_st()["shorts"] == {}


def test_r3_halb_pair_flips_and_both_ride_p(monkeypatch, caplog):
    _on(monkeypatch)
    f = _fx()

    async def run():
        a = asyncio.ensure_future(f._decode_collect_short("a", 3500))
        await asyncio.sleep(0.02)
        assert f._decode_collect(time.time()) == "hold"
        b = asyncio.ensure_future(f._decode_collect_short("b", 3200))
        return await asyncio.gather(a, b)

    t0 = time.time()
    with caplog.at_level(logging.INFO, logger="weg2.front"):
        ra, rb = asyncio.run(run())
    assert (ra, rb) == ("P", "P") and time.time() - t0 < 2.0
    assert f.counters["set_flip_p"] == 1
    assert any("S=6700" in m and "verdict=FLIP_P" in m for m in _verdicts(caplog)), _verdicts(caplog)


def test_r3_halb_alone_goes_to_d_after_t_half_never_a_flip(monkeypatch):
    _on(monkeypatch, t_half="0.3")
    f = _fx()
    t0 = time.time()
    assert asyncio.run(f._decode_collect_short("a", 3500)) == "D"
    assert time.time() - t0 >= 0.25, "it waited T_halb"
    assert f.counters["set_flip_p"] == 0 and f.counters["set_park_no_seat"] == 0 and f.counters["set_d_after_timer"] == 1
    assert f._set_st()["go_p"] == set(), "no flip was committed"


def test_r3_the_timer_starts_at_the_first_member_not_at_each(monkeypatch):
    _on(monkeypatch, t_half="1.0", frac="0.4")      # halbX 2400: 2500 + 2500 stays under X
    f = _fx()
    t0 = time.time()
    f._set_st()["shorts"]["a"] = (t0, 2500)
    assert f._decode_collect(t0 + 0.1) == "hold"
    f._set_st()["shorts"]["b"] = (t0 + 0.8, 2500)
    assert f._decode_collect(t0 + 1.05) == "D", "a's timer ran out at 1.0; b's arrival at 0.8 must not restart it"
    assert f._set_st()["released"] == {"a": "D", "b": "D"}


def test_r4_gross_waits_for_company_with_decode_then_flips(monkeypatch):
    _on(monkeypatch)
    f = _fx()
    now = time.time()
    f.queue = [_pending("g", 9000, now)]
    wf, _fair, imm = asyncio.run(f._arrival_seat_step(f.groups["D"], now))
    assert (wf, imm) == (False, None), "R4: D decodes -> the big request waits for company"
    f._set_st()["shorts"]["h"] = (now + 1.0, 3100)         # company inside T_halb
    wf, _fair, imm = asyncio.run(f._arrival_seat_step(f.groups["D"], now + 1.0))
    assert wf is True and imm.rid == "g"
    assert f.counters["set_flip_p"] == 1


def test_r4_gross_flips_when_t_half_is_over(monkeypatch):
    _on(monkeypatch)
    f = _fx()
    now = time.time()
    f.queue = [_pending("g", 9000, now - 4.6)]
    wf, _fair, imm = asyncio.run(f._arrival_seat_step(f.groups["D"], now))
    assert wf is True and imm.rid == "g" and f.counters["set_flip_p"] == 1


def test_r8_no_decode_a_gross_request_flips_at_once(monkeypatch, caplog):
    _on(monkeypatch)
    f = _fx(running=())
    now = time.time()
    f.queue = [_pending("g", 9000, now)]
    with caplog.at_level(logging.INFO, logger="weg2.front"):
        wf, _fair, imm = asyncio.run(f._arrival_seat_step(f.groups["D"], now))
    assert wf is True and imm.rid == "g"
    (line,) = _verdicts(caplog)
    assert "d_decoding=0" in line and "verdict=FLIP_P" in line and "waited_s=0.0" in line


def test_r5_no_free_seat_is_park_no_seat_and_still_prefills_on_p(monkeypatch, caplog):
    _on(monkeypatch)
    f = _fx(running=[f"d{i}" for i in range(6)], n=6)    # six decodes on six seats
    now = time.time()
    f.queue = [_pending("g", 9000, now - 5.0)]
    with caplog.at_level(logging.INFO, logger="weg2.front"):
        wf, _fair, imm = asyncio.run(f._arrival_seat_step(f.groups["D"], now))
    assert wf is True and imm.rid == "g", "prefilled on P, parked there; the decode starts when a seat frees"
    (line,) = _verdicts(caplog)
    assert "seats_free=0" in line and "verdict=PARK_NO_SEAT" in line
    assert f.counters["set_park_no_seat"] == 1 and f.counters["set_flip_p"] == 0


def test_r6_a_running_d_prefill_is_rerouted_when_the_set_passes_x(monkeypatch, caplog):
    _on(monkeypatch)
    f = _fx()
    f._d_pf_book().grant(rid="run", tokens=4000, now=time.time())
    f._d_pf_book().enter_leg2(rid="run")
    f.groups["D"].outstanding["run"] = 1.0       # D holds it: its prefill is running
    t0 = time.time()
    with caplog.at_level(logging.INFO, logger="weg2.front"):
        assert asyncio.run(f._decode_collect_short("a", 3500)) == "P", "S = 4000 + 3500 > X"
    assert time.time() - t0 < 1.0, "R6: at once, no T_halb"
    assert f.counters["set_reroute_p"] == 1
    assert any("verdict=REROUTE_P" in m and "S=7500" in m for m in _verdicts(caplog))


def test_r7_no_epoch_rule_a_flip_leaves_nothing_behind(monkeypatch):
    _on(monkeypatch)
    f = _fx()
    now = time.time()
    f.queue = [_pending("g", 9000, now - 5.0)]
    assert f._decode_collect(now) == "P"
    assert f._set_st()["go_p"] == {"g"}
    # the flip happened (new epoch); "g" is still queued (P's cap). Nothing of the old epoch's verdict carries over:
    # it is a fresh candidate and waits its own T_halb like any big request
    f.epoch += 1
    f.queue = [_pending("g", 9000, time.time())]
    assert f._set_st()["go_p"] == set()
    assert f._decode_collect(time.time()) == "hold"


def test_r7_after_a_decided_flip_a_kurz_arrival_rides_the_foreseeable_flip_then_is_d_s_again(monkeypatch):
    """H5b (no D-direct pass behind a park the flip needs) still holds while a flip is DECIDED; it is no epoch
    rule: in the next epoch a kurz arrival goes to D at once."""
    _on(monkeypatch)
    f = _fx()
    now = time.time()
    f.queue = [_pending("g", 9000, now - 5.0)]
    assert f._decode_collect(now) == "P"
    assert asyncio.run(f._acquire_short_seat("k", 300, uncached=300)) is None
    assert f.counters["d_direct_yield"] == 1
    f.epoch += 1
    f.queue = []
    t0 = time.time()
    seat = asyncio.run(f._acquire_short_seat("k2", 300, uncached=300))
    assert seat is not None and time.time() - t0 < 1.0


def test_a_committed_flip_is_carried_out_without_a_second_timer(monkeypatch):
    _on(monkeypatch)
    f = _fx()
    now = time.time()
    f.queue = [_pending("g", 9000, now - 5.0)]
    assert f._decode_collect(now) == "P"
    # the controller could not start the flip this tick; the next tick must not collect again
    assert f._decode_collect(now + 0.2) == "P"
    wf, _fair, imm = asyncio.run(f._arrival_seat_step(f.groups["D"], now + 0.2))
    assert wf is True and imm.rid == "g"


def test_t_half_zero_keeps_the_immediate_paths(monkeypatch):
    _on(monkeypatch, t_half="0")
    f = _fx()
    now = time.time()
    f.queue = [_pending("g", 9000, now)]
    assert f._decode_collect(now) == "P"


def test_set_verdict_line_has_the_exact_format(monkeypatch, caplog):
    _on(monkeypatch)
    f = _fx()
    now = time.time()
    f.queue = [_pending("g", 9000, now - 5.0), _pending("h", 3100, now - 4.9)]
    f.queue[1].x_requeues = 1                      # needs P although under X: P-only by state (R9)
    with caplog.at_level(logging.INFO, logger="weg2.front"):
        f._decode_collect(now)
    (line,) = _verdicts(caplog)
    pat = (r"WEG2 SET-VERDICT epoch=\d+ set=\d+ n=\d+ members=(\S+:\d+,?)+ S=\d+ X=\d+ halfX=\d+ "
           r"classes=(kurz|halb|gross|p_only)(,(kurz|halb|gross|p_only))* d_decoding=[01] seats_free=\d+ "
           r"t_half_s=\d+\.\d waited_s=\d+\.\d verdict=[A-Z_]+ why=.+")
    assert re.fullmatch(pat, line), line
    assert "members=g:9000,h:3100" in line and "classes=gross,p_only" in line and "S=12100 X=6000 halfX=3000" in line


def test_p_batch_takes_only_sets_of_x_or_more(monkeypatch):
    f = _fx()
    mk = lambda rid, u: _pending(rid, u, time.time())  # noqa: E731
    a, b = mk("a", 2000), mk("b", 2500)
    for p in (a, b):
        p.d_direct = True
    f._ready_for_d = [a, b]
    f.counters["p_batch_from_d"] = 0
    assert f._p_batch_takes_d_prefill() == 0, "S=4500 < X: the set stays with D (R7)"
    assert f._ready_for_d == [a, b] and a.d_direct is True
    c = mk("c", 2000)
    c.d_direct = True
    f._ready_for_d = [a, b, c]
    f._sync_batch_gate = lambda: None
    assert f._p_batch_takes_d_prefill() == 3, "S=6500 >= X: P takes the set"
    assert f._ready_for_d == [] and a.d_direct is False


def test_half_flip_check_reports_every_10th_flip_and_every_30_min(monkeypatch, caplog):
    _on(monkeypatch)
    f = _fx()
    # 5 flips each way: D>P 9.0 s, P>D 10.0 s -> half of the p90 sum = 9.5; T_halb 4.5: 52.6 % off -> abweichung
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="weg2.front"):
        for i in range(10):
            f._half_flip_note("D>P" if i % 2 == 0 else "P>D", 9000.0 if i % 2 == 0 else 10000.0)
        msgs = [r.getMessage() for r in caplog.records if "HALF-FLIP-CHECK" in r.getMessage()]
        assert len(msgs) == 1 and "flip10" in msgs[0]
        assert re.search(r"t_half_s=4\.5 p90_dp_s=9\.0 p90_pd_s=10\.0 half_p90_s=9\.5 dev_pct=52\.6 verdict=abweichung", msgs[0])
        caplog.clear()
        f._half_flip_tick(f._half_flip_t + 100)
        assert not [r for r in caplog.records if "HALF-FLIP-CHECK" in r.getMessage()]
        f._half_flip_tick(f._half_flip_t + 1801)
        assert [r for r in caplog.records if "HALF-FLIP-CHECK" in r.getMessage() and "30min" in r.getMessage()]


def test_half_flip_check_within_30_percent_is_ok():
    r = pp.half_flip_check(4.5, [4.0, 4.0], [5.0, 5.0])      # half of (4 + 5) = 4.5
    assert r["verdict"] == "ok" and r["dev_pct"] == 0.0 and r["half_p90_s"] == 4.5
    assert pp.half_flip_check(4.5, [4.0], [5.9])["verdict"] == "ok"        # 4.95: 10 % off
    assert pp.half_flip_check(4.5, [8.0], [9.0])["verdict"] == "abweichung"  # 8.5: 47 % off
    assert pp.half_flip_check(4.5, [], [1.0]) is None
