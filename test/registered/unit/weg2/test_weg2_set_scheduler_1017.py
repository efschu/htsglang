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


def test_a1_r1a_a_decided_flip_takes_everything_takable_also_a_small_set(monkeypatch):
    """R1a: P-BATCH-ALL -- the flip a set decided takes ALL D-prefill entries waiting in D's line, also under X."""
    f = _fx()
    mk = lambda rid, u: _pending(rid, u, time.time())  # noqa: E731
    a, b = mk("a", 200), mk("b", 300)
    for p in (a, b):
        p.d_direct = True
    f._ready_for_d = [a, b]
    f._sync_batch_gate = lambda: None
    assert f._p_batch_takes_d_prefill() == 2
    assert f._ready_for_d == [] and a.d_direct is False and f.queue == [a, b] or len(f.queue) == 2


def test_a1_yielded_kurz_is_taken_by_the_p_phase_after_a_decided_flip(monkeypatch):
    """Review A repro: kurz yields to a decided flip (H5b), lands in D's line; the P phase takes it (not P-BATCH-STAYS)."""
    _on(monkeypatch)
    f = _fx()
    now = time.time()
    f.queue = [_pending("g", 9000, now - 5.0)]
    assert f._decode_collect(now) == "P"
    k = _pending("k", 300, time.time())
    k.d_direct = True
    f._ready_for_d.append(k)
    f._sync_batch_gate = lambda: None
    f.awake = "P"
    assert f._p_batch_takes_d_prefill() == 1 and f._ready_for_d == []


def test_a1_r6_counts_only_a_really_running_d_prefill_not_waiting_ready_entries(monkeypatch):
    _on(monkeypatch)
    f = _fx()
    w = _pending("w", 4000, time.time())
    w.d_direct = True
    f._ready_for_d.append(w)       # waits in D's line: no D prefill runs
    assert f._dc_carried_running() == 0
    f._d_pf_book().grant(rid="run", tokens=4000, now=time.time())
    f._d_pf_book().enter_leg2(rid="run")
    f.groups["D"].outstanding["run"] = 1.0
    assert f._dc_carried_running() == 4000


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


# ------------------------------------------------------------------------------------- Review A Auflagen (A2-A8)

def test_a2_price_barrier_no_longer_decides_a_kurz_request_waits_for_no_price(monkeypatch):
    """A2/R2: a LONG candidate no longer registers a price barrier (the to_p outcome was a second decider)."""
    from sglang.srt.environ import envs
    f = _fx()
    f.x_exact = True
    assert envs.SGLANG_WEG2_PRICE_BARRIER.get() is True            # readable ...
    assert type(f)._pb_register(f, "a", 9000) is None              # ... decides nothing
    assert not f.__dict__.get("_pb_inflight")


def test_a2_burst_long_candidate_plus_kurz_kurz_gets_its_seat_without_t_half(monkeypatch):
    _on(monkeypatch)
    f = _fx()
    f.queue = [_pending("big", 9000, time.time())]
    f.__dict__["_pb_inflight"] = {}
    t0 = time.time()
    seat = asyncio.run(f._acquire_short_seat("k", 400, uncached=400))
    assert seat is not None and time.time() - t0 < 1.0


def test_a3_each_member_carries_its_own_x_and_s_is_checked_against_the_smallest(monkeypatch, caplog):
    """R2a: X(depth)=4100 < envelope 6000. Two 2600 SHORTs: halfX(own)=2050 -> halb (not kurz as with the envelope's
    3000), S=5200 >= 4100 -> FLIP_P, and the SET-VERDICT line reports that X."""
    _on(monkeypatch)
    f = _fx()
    kurz_by_envelope = pp.classify_arrival(2600, X, HALF, False)
    assert kurz_by_envelope == "kurz" and pp.classify_arrival(2600, 4100, HALF, False) == "halb"

    async def run():
        a = asyncio.ensure_future(f._acquire_short_seat("a", 2700, uncached=2600, x_set=4100))
        await asyncio.sleep(0.02)
        b = asyncio.ensure_future(f._acquire_short_seat("b", 2700, uncached=2600, x_set=4100))
        return await asyncio.gather(a, b)

    t0 = time.time()
    with caplog.at_level(logging.INFO, logger="weg2.front"):
        ra, rb = asyncio.run(run())
    assert (ra, rb) == (None, None) and time.time() - t0 < 2.0
    lines = [m for m in _verdicts(caplog) if "verdict=FLIP_P" in m]
    assert lines and "X=4100" in lines[0] and "S=5200" in lines[0] and "classes=halb,halb" in lines[0], lines


def test_a3_no_p_only_label_between_depth_x_and_the_envelope(monkeypatch, caplog):
    _on(monkeypatch)
    f = _fx()
    q = _pending("g", 5000, time.time() - 5.0)   # over its own X 4100, under the envelope 6000
    q.x_set = 4100
    q.x_routed = 4100
    f.queue = [q]
    with caplog.at_level(logging.INFO, logger="weg2.front"):
        assert f._decode_collect(time.time()) == "P"
    (line,) = _verdicts(caplog)
    assert "classes=gross" in line and "p_only" not in line and "X=4100" in line, line


def test_a4_a_set_returnee_keeps_its_real_arrival_as_the_seat_order_stamp(monkeypatch):
    _on(monkeypatch)
    f = _front(running=["a"], n=1, kv={"available": 400000, "evictable": 0})
    f.tp_prefill_max_tokens = X
    t_real = time.time() - 4.5

    async def run():
        t = asyncio.ensure_future(f._arrival_seat_wait("h", 100, None, uncached=3500, t_arrive=t_real))
        await asyncio.sleep(0.1)
        stamp = f._asr_st()["waiters"]["h"]
        t.cancel()
        return stamp

    assert asyncio.run(run()) == t_real


def test_a5_t_open_never_before_this_d_phase_woke(monkeypatch):
    _on(monkeypatch)
    f = _fx()
    now = time.time()
    f.t_awake = now - 1.0                              # D woke 1 s ago (P->D flip)
    f.queue = [_pending("g", 9000, now - 10.0)]        # arrived during the flip
    assert f._decode_collect(now) == "hold", "the big request waits its T_halb in the NEW D phase (no double flip)"
    assert f._decode_collect(now + 3.6) == "P"


def test_a6_d_now_is_logged_when_really_admitted_with_its_wait(monkeypatch, caplog):
    _on(monkeypatch)
    f = _front(running=["a"], n=1, kv={"available": 400000, "evictable": 0})
    f.tp_prefill_max_tokens = X
    f.t_awake = time.time() - 60.0

    async def run():
        t = asyncio.ensure_future(f._acquire_short_seat("k", 400, uncached=400))
        await asyncio.sleep(0.3)
        assert not _verdicts(caplog), "no D_NOW while the kurz request still waits for its seat"
        f.groups["D"].outstanding.pop("a")
        return await asyncio.wait_for(t, 3.0)

    with caplog.at_level(logging.INFO, logger="weg2.front"):
        assert asyncio.run(run()) is not None
    (line,) = _verdicts(caplog)
    m = re.search(r"waited_s=(\d+\.\d) verdict=D_NOW why=R2", line)
    assert m and float(m.group(1)) >= 0.2, line


def test_a6_a_kurz_that_is_not_admitted_logs_the_reason(monkeypatch, caplog):
    _on(monkeypatch)
    f = _fx()
    f._d_token_budget_blocks = lambda *a, **k: True
    with caplog.at_level(logging.INFO, logger="weg2.front"):
        assert asyncio.run(f._acquire_short_seat("k", 400, uncached=400)) is None
    (line,) = _verdicts(caplog)
    assert "verdict=NOT_D why=d_budget" in line and "waited_s=" in line, line


def test_a8_a_queue_under_x_is_no_flip_reason(monkeypatch, caplog):  # (a P-by-state orphan is N1, below)
    """R7/A8: small work D could not take; the work-exhausted path does not flip for it (no WAIT-BOUND-ANY flip either)."""
    _on(monkeypatch)
    f = _fx(running=())
    now = time.time()
    s1, s2 = _pending("s1", 1000, now - 90.0), _pending("s2", 800, now - 80.0)
    f.queue = [s1, s2]
    with caplog.at_level(logging.INFO, logger="weg2.front"):
        assert f._set_flip_justified(now) is False
        wf, _fair, imm = asyncio.run(f._arrival_seat_step(f.groups["D"], now))
    assert wf is False and imm is None, "WAIT-BOUND-ANY must not flip a queue under X"
    assert f.counters["flip_held_under_x"] == 1


def test_a8_a_queue_sum_over_x_flips_through_set_verdict_and_is_logged(monkeypatch, caplog):
    _on(monkeypatch)
    f = _fx(running=())
    now = time.time()
    f.queue = [_pending("s1", 3000, now - 9.0), _pending("s2", 3100, now - 8.0)]
    with caplog.at_level(logging.INFO, logger="weg2.front"):
        assert f._set_flip_justified(now) is True
    assert any("verdict=FLIP_P" in m and "queue-sum" in m for m in _verdicts(caplog))


def test_b3_reroute_takes_the_waiting_d_line_entries_along_to_p(monkeypatch, caplog):
    """R6/B3 (example 7): a D prefill of 4000 runs, 5000 arrives -> REROUTE_P; what waits in D's line rides along."""
    _on(monkeypatch)
    f = _fx()
    f._sync_batch_gate = lambda: None
    f._d_pf_book().grant(rid="run", tokens=4000, now=time.time())
    f._d_pf_book().enter_leg2(rid="run")
    f.groups["D"].outstanding["run"] = 1.0
    w = _pending("w", 900, time.time() - 1.0)
    w.d_direct = True
    f._ready_for_d.append(w)
    t0 = time.time()
    f.queue = [_pending("g", 7000, t0 - 0.5)]
    with caplog.at_level(logging.INFO, logger="weg2.front"):
        assert f._decode_collect(t0) == "P"
    assert any("verdict=REROUTE_P" in m for m in _verdicts(caplog))
    assert f._ready_for_d == [] and {p.rid for p in f.queue} == {"g", "w"}, "the waiting entry is on P's queue"
    assert f._set_st()["go_p"] >= {"g", "w"} and w.d_direct is False


def test_b4_leftovers_from_earlier_epochs_wait_their_t_half_in_the_new_d_phase(monkeypatch):
    _on(monkeypatch)
    f = _fx()
    now = time.time()
    f.epoch = 9
    f.t_awake = now - 0.5
    q = _pending("old", 9000, now - 200.0)        # left behind by P's cap two phases ago
    f.queue = [q]
    assert f._decode_collect(now) == "hold"
    assert f._set_st()["t_open"] >= f.t_awake


def test_b6_a_short_the_d_budget_refused_goes_back_to_d_when_the_budget_frees_never_a_flip(monkeypatch):
    """27B 7a19cae42a (restored): budget-refused kurz is queued; as soon as D's budget frees it is D's again."""
    _on(monkeypatch)
    f = _fx()
    now = time.time()
    p = _pending("s1", 800, now - 8.0)
    p.d_eligible, p.d_budget_refused = False, True
    f.queue = [p]
    blocked = {"v": True}
    f._d_token_budget_blocks = lambda *a, **k: blocked["v"]
    wf, _fair, imm = asyncio.run(f._arrival_seat_step(f.groups["D"], now))
    assert (wf, imm) == (False, None) and f._ready_for_d == [] and p in f.queue, "no flip, no move while blocked"
    blocked["v"] = False
    wf, _fair, imm = asyncio.run(f._arrival_seat_step(f.groups["D"], now + 1.0))
    assert (wf, imm) == (False, None)
    assert f._ready_for_d == [p] and p.d_direct is True and p not in f.queue
    assert f.counters["d_budget_refused_back_to_d"] == 1


def _orphan(now, age):
    o = _pending("lost", 2000, now - age)
    o.d_eligible = o.d_direct = o.leg1_done = o.intake_stalled = False
    o.reroutes, o.handoff_lost_reroutes = 1, 0       # leg-2 reroute: leg 1 runs again on P (P-by-state, R9/N1)
    return o


def test_n1_an_orphan_p_by_state_under_x_flips_at_once_on_an_idle_d(monkeypatch, caplog):
    """R9/R8: a request that must run on P by state is p_only = big; D idle -> flip now (was FLIP-HELD for ever)."""
    _on(monkeypatch)
    f = _fx(running=())
    now = time.time()
    f.queue = [_orphan(now, 70.0)]
    with caplog.at_level(logging.INFO, logger="weg2.front"):
        wf, _fair, imm = asyncio.run(f._arrival_seat_step(f.groups["D"], now))
    assert wf is True and imm is not None and imm.rid == "lost"
    assert any("classes=p_only" in m and "verdict=FLIP_P" in m for m in _verdicts(caplog))
    assert f.counters["flip_held_under_x"] == 0


def test_n1_an_orphan_p_by_state_waits_t_half_on_a_decoding_d_then_flips(monkeypatch):
    _on(monkeypatch)
    f = _fx()
    now = time.time()
    f.queue = [_orphan(now, 0.5)]
    wf, _fair, imm = asyncio.run(f._arrival_seat_step(f.groups["D"], now))
    assert (wf, imm) == (False, None), "R4: waits for company"
    wf, _fair, imm = asyncio.run(f._arrival_seat_step(f.groups["D"], now + 4.6))
    assert wf is True and imm.rid == "lost"
    assert f._set_flip_justified(now + 4.7) is True


def test_n1_seat_wait_is_named_in_the_d_now_line(monkeypatch, caplog):
    _on(monkeypatch)
    f = _front(running=["a"], n=1, kv={"available": 400000, "evictable": 0})
    f.tp_prefill_max_tokens = X
    f.t_awake = time.time() - 60.0

    async def run():
        t = asyncio.ensure_future(f._acquire_short_seat("k", 400, uncached=400))
        await asyncio.sleep(0.7)
        f.groups["D"].outstanding.pop("a")
        return await asyncio.wait_for(t, 3.0)

    with caplog.at_level(logging.INFO, logger="weg2.front"):
        assert asyncio.run(run()) is not None
    (line,) = _verdicts(caplog)
    assert "verdict=D_NOW why=seat_wait" in line, line


# ------------------------------------------------------------------------------------------- S28 (18.10.): no flip on a stale price

class _Spans:
    """tspans stand-in: ``pending(ids, epoch)`` -> (uncached, credit, known, src); ``price`` is what the store holds NOW."""

    def __init__(self, price):
        self.price = price
        self.calls = 0

    def pending(self, ids, epoch=None, since_seq=None):
        self.calls += 1
        return self.price, 100000, True, "store_anchor"


class _Tok:
    def ids_for(self, text):
        import numpy as np
        return np.arange(120000, dtype=np.int32)


def _exact(f, price):
    f.x_exact, f.ftok, f.tspans = True, _Tok(), _Spans(price)
    f._d_page_size = lambda: 64
    return f


def _big(rid, u, age):
    q = _pending(rid, u, time.time() - age)
    q.text, q.d_eligible, q.reroutes, q.handoff_lost_reroutes, q.intake_stalled = "t", False, 0, 0, False
    q.x_routed, q.p_only, q.mm_image_end = 0, False, 0
    return q


def test_s28_flip_verdict_is_made_on_the_current_price_not_the_stale_one(monkeypatch, caplog):
    """R1/R7: the queue holds 52002 (stale), the store/anchor presence says 34. T_halb is over: without the pre-verdict
    re-price the set says FLIP_P for 34 real tokens (fLLiper rc8 pdflip-17-189); with it, no flip (S=34 < X)."""
    _on(monkeypatch)
    f = _exact(_fx(), 34)
    q = _big("stale", 52002, 40.0)
    f.queue = [q]
    with caplog.at_level(logging.INFO, logger="weg2.front"):
        assert f._decode_collect(time.time()) != "P"
    assert q.est_uncached == 34 and f.counters["flip_verdict_repriced"] == 1
    assert not any("verdict=FLIP_P" in m or "verdict=PARK_NO_SEAT" in m for m in _verdicts(caplog))
    assert f._set_flip_justified(time.time()) is False


def test_s28_a_really_big_price_still_flips(monkeypatch):
    """The re-price does not weaken R4: the presence confirms 52002 > X -> the verdict stands (and the price is unchanged)."""
    _on(monkeypatch)
    f = _exact(_fx(), 52002)
    f.queue = [_big("big", 52002, 40.0)]
    assert f._decode_collect(time.time()) == "P"
    assert f.counters["flip_verdict_repriced"] == 0


def test_s28_lost_handoff_priced_on_the_store_stays_on_d(monkeypatch, caplog):
    """R1/R2/R7 (item 2): the hand-off credit stops at first_lost_page (uncached 52002 > X) but the store holds the span:
    the rid is priced 34 and is NOT rerouted to P (no P leg, no flip)."""
    import sys
    import types
    from sglang.srt.weg2 import handoff_seam as HS
    mod = types.ModuleType("_fake_hp_s28")
    mod.status = lambda rid: {"state": "lost", "first_lost_page": 1016, "pages": 1828, "page_size": 64}
    mod.drop = lambda rid, reason: None
    sys.modules["_fake_hp_s28"] = mod
    old = HS.MODULE
    HS.MODULE = "_fake_hp_s28"
    HS.reset_module_cache()
    try:
        f = _exact(_fx(), 34)
        f.queue = __import__("collections").deque()
        p = _big("lost", 34, 120.0)
        p.leg1_done, p.skip_leg1, p.d_direct, p.resume_via_p = True, False, False, False
        p.leg1_prompt_tokens, p.fut = 117026, types.SimpleNamespace(done=lambda: False)
        f.tp_prefill_max_tokens = 4921
        f._dp_mark = lambda *a: None
        f._kick_controller = lambda *a: None
        with caplog.at_level(logging.INFO, logger="weg2.front"):
            assert f._hl_check(p, "sweep") is False
        assert p.leg1_done is True and p.handoff_lost_reroutes == 0 and f.queue == []
        assert f.counters["handoff_lost_store_priced"] == 1 and f.counters["handoff_lost_reroutes"] == 0
        assert any("HANDOFF-LOST-STORE-PRICE" in r.getMessage() for r in caplog.records)
        # the store holds nothing: the hand-off's own price (52002 > X) stands -> fresh P (R4 path unchanged)
        f2 = _exact(_fx(), 52002)
        f2.queue = __import__("collections").deque()
        f2.tp_prefill_max_tokens = 4921
        f2._dp_mark, f2._kick_controller = (lambda *a: None), (lambda *a: None)
        p2 = _big("lost2", 52002, 120.0)
        p2.leg1_done, p2.skip_leg1, p2.d_direct, p2.resume_via_p = True, False, False, False
        p2.leg1_prompt_tokens, p2.fut = 117026, types.SimpleNamespace(done=lambda: False)
        assert f2._hl_check(p2, "sweep") is True and p2.handoff_lost_reroutes == 1
    finally:
        HS.MODULE = old
        HS.reset_module_cache()
        sys.modules.pop("_fake_hp_s28", None)


def test_s28_needs_p_by_state_of_a_lost_handoff_only_while_over_x():
    """R7: a lost hand-off is P-by-state only on a price over X; a leg-2 reroute stays P-by-state (R9/N1)."""
    assert pp.needs_p(52002, X, handoff_lost_reroutes=1) is True
    assert pp.needs_p(34, X, handoff_lost_reroutes=1) is False
    assert pp.needs_p(34, X, reroutes=1) is True


def test_s28_flip_price_stale_instrument(monkeypatch, caplog):
    """Item 5: a flip trigger re-priced under X within 2 s of its verdict -> ``WEG2 FLIP-PRICE-STALE`` + counter."""
    _on(monkeypatch)
    f = _exact(_fx(), 34)
    q = _big("trig", 52002, 40.0)
    f.queue = [q]
    f._flip_trig_note("trig", 52002, "park-immediate")
    with caplog.at_level(logging.INFO, logger="weg2.front"):
        f._x_exact_reprice_queue("p_anchor")
    (line,) = [r.getMessage() for r in caplog.records if "FLIP-PRICE-STALE" in r.getMessage()]
    assert "rid=trig site=park-immediate est_at_flip=52002 est_now=34 X=6000" in line
    assert f.counters["flip_price_stale"] == 1
    # later than 2 s, or a price still over X: no signal
    f.counters.clear()
    q2 = _big("late", 52002, 40.0)
    f.queue = [q2]
    f._flip_trig_note("late", 52002, "set-verdict")
    f._flip_trig["late"] = (time.time() - 3.0, 52002, "set-verdict")
    f._x_exact_reprice_queue("p_anchor")
    assert f.counters["flip_price_stale"] == 0
