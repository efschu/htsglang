"""rc30 R8 (NF 21.10.): SCHEDULER-SOLL v2 R8 "Ohne Decode auf D wird sofort entschieden: D bei S < X, P bei S >= X" --
no COLLECT at d_decoding=0; and the S49 X-SUM term never carries a kurz grant (R2 "sofort auf D", R6k "sein Rest zählt nicht
zu S"; R1: the set sum S decides).

Belegfall (NF rc30 boot ..._1d822bd150_1021_063252.front.log, first requests after the boot):
  06:36:19,450 PDFLIP R6-NOT-STARTED granted=3868 running=0             (0-1 1642 running kurz, 0-2 1013 + 0-3 66 + 0-6 1147
                                                                         granted kurz, not begun = 2226)
  06:36:19,451 PDFLIP SET-VERDICT epoch=0 set=0 n=1 members=pdflip-0-4:3821 S=3821 X=4096 halfX=2048 classes=halb
               d_decoding=0 ... verdict=COLLECT why=NS:S49 X-SUM S=3821 + 2226 granted on D not started > X=4096
  06:36:19,458 the next member (0-5, 4004) arrives: R8 FLIP_P S=7825 -- the COLLECT stood 7 ms only because a second halb
               member came; alone, 0-4 would have waited for D to begin or finish three kurz prefills.
The fix: ``set_verdict``'s R8 branch returns D_AFTER_TIMER directly (the S49 hold only at the R3 timer end while D decodes);
the front leaves kurz rows out of ``d_granted_unstarted`` (``_d_pf_pending_tokens(skip_kurz=True)``).

Tests: (a) pure ``set_verdict``/``decide``: R8 never COLLECT (Belegfall numbers + a grid), R8 S >= X flips, the Gegenfälle
with D decoding (R3 collects inside T_halb, R4 waits, the S49 hold at the R3 timer end for a halb/big grant stays);
(b) front ``_decode_collect`` on the Belegfall (idle D, kurz grants), and with D decoding: kurz grants never hold (R2/R6k), a
halb grant still holds (NS:S49) and R6 reroutes once D begins it. Mutants: ``mutants_nf_rc30_r8_1021.py``.
"""
from __future__ import annotations

import itertools
import logging
import os
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from test_pdflip_arrival_seat_rule_0929 import _front  # noqa: E402
from test_pdflip_set_scheduler_1017 import _on  # noqa: E402  (T_halb 4.5 s, F_halb 0.5: SOLL section 5)

from flliper.srt.managers.scheduler_components import weight_updater as _wu  # noqa: E402,F401
from flliper.srt.pdflip import d_extend_beacon as dxb  # noqa: E402
from flliper.srt.pdflip import front as F  # noqa: E402
from flliper.srt.pdflip import phase_policy as pp  # noqa: E402
from flliper.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=5, suite="stage-a-pdflip-unit")

X_RC30 = 4096          # the Belegfall's X (halfX 2048)
T_HALF = 4.5


def M(rid, u, t=0.0, p_only=False):
    return pp.SetMember(rid=rid, uncached=u, t_arrive=t, p_only=p_only)


def V(members, *, now=0.0, t_open=0.0, decoding=False, seats=6, running=0, unstarted=0, x=X_RC30):
    return pp.decide("set", members=members, x_tokens=x, half_frac=0.5, t_half_s=T_HALF, now=now, t_open=t_open,
                     d_decoding=decoding, seats_free=seats, d_prefill_running_uncached=running, new_since_last=0,
                     d_granted_unstarted=unstarted)


# ------------------------------------------------------------------------------------------------------ (a) pure verdict

def test_r8_belegfall_rc30_no_collect_without_decode_d_now():
    """06:36:19,451: 0-4 halb 3821, X=4096, d_decoding=0, 2226 granted on D not begun -> R8: D at once (S < X)."""
    d = V([M("pdflip-0-4", 3821)], unstarted=2226)
    assert d.verdict.verdict == pp.V_D_AFTER_TIMER, d.verdict.why
    assert d.rule == "R8" and d.verdict.s_tokens == 3821 and d.verdict.left_s == 0.0
    assert not d.verdict.why.startswith("NS:S49")


def test_r8_s_at_or_over_x_flips_now_and_a_big_member_flips_now():
    """R8 P side: S >= X (0-4 + 0-5 = 7825) or a big member alone -> FLIP_P at once, grant or not (SOLL example 6)."""
    assert V([M("pdflip-0-4", 3821), M("pdflip-0-5", 4004)], unstarted=2226).verdict.verdict == pp.V_FLIP_P
    assert V([M("g", 9000)], unstarted=9000).verdict.verdict == pp.V_FLIP_P
    assert V([M("v", 100, p_only=True)]).verdict.verdict == pp.V_FLIP_P             # R9


def test_r8_never_collect_on_a_grid():
    """Whatever the members, the timer, the grant and the running rest: nothing decodes on D -> never COLLECT."""
    us = (2048, 3000, 3821, 4095, 4096, 9000)
    for (u1, u2, unstarted, now, n2) in itertools.product(us, us, (0, 1, 2226, 9000), (0.0, 1.0, 9.0), (False, True)):
        mem = [M("a", u1)] + ([M("b", u2, 0.5)] if n2 else [])
        d = V(mem, now=now, unstarted=unstarted)
        assert d.verdict.verdict != pp.V_COLLECT, (u1, u2, unstarted, now, n2, d.verdict.why)
        s = u1 + (u2 if n2 else 0)
        big = u1 >= X_RC30 or (n2 and u2 >= X_RC30)
        want = pp.V_FLIP_P if (big or s >= X_RC30) else pp.V_D_AFTER_TIMER
        assert d.verdict.verdict == want, (u1, u2, unstarted, now, n2, d.verdict.why)


def test_gegenfall_d_decodes_r3_collects_inside_t_half():
    d = V([M("pdflip-0-4", 3821)], now=1.0, decoding=True, unstarted=2226)
    assert (d.verdict.verdict, d.rule) == (pp.V_COLLECT, "R3") and d.verdict.left_s > 0


def test_gegenfall_d_decodes_r4_big_waits_for_company():
    d = V([M("g", 9000)], now=1.0, decoding=True)
    assert (d.verdict.verdict, d.rule) == (pp.V_COLLECT, "R4")
    assert V([M("g", 9000)], now=5.0, decoding=True).verdict.verdict == pp.V_FLIP_P


def test_gegenfall_d_decodes_r3_timer_end_the_s49_hold_for_a_halb_grant_stays():
    """D decodes, T_halb over, S < X: the set goes to D (R3) -- unless a halb/big grant D has not begun passes X with it
    (NS:S49: wait for D to begin it, then R6 counts it running). The front never passes a kurz grant here."""
    d = V([M("a", 3821)], now=5.0, decoding=True, unstarted=2226)
    assert (d.verdict.verdict, d.rule) == (pp.V_COLLECT, "NS:S49")
    assert V([M("a", 3821)], now=5.0, decoding=True, unstarted=0).verdict.verdict == pp.V_D_AFTER_TIMER


# ------------------------------------------------------------------------------------------------------ (b) front glue

def _verdicts(caplog):
    return [r.getMessage() for r in caplog.records if "PDFLIP SET-VERDICT" in r.getMessage()]


def _fx(now, decoding):
    f = _front(running=(["dec"] if decoding else []), n=6, kv={"available": 400000, "evictable": 0})
    f.tp_prefill_max_tokens = X_RC30
    f.t_awake = now - 60.0
    f._sync_batch_gate = lambda: None
    f._d_ext_reading = lambda: {}                      # D has begun none of the grants
    return f


def _grant(f, rid, tokens, klass, now):
    f._d_pf_book().grant(rid=rid, tokens=tokens, now=now - 0.05)
    F.Front._r6k_note_class(f, rid, klass, "arrival", X_RC30 // 2)


def _halb_set(f, rid, u, t):
    f._set_st()["shorts"][rid] = (t, u, int(f.tp_prefill_max_tokens))


def _belegfall_grants(f, now):
    for rid, u in (("pdflip-0-2", 1013), ("pdflip-0-3", 66), ("pdflip-0-6", 1147)):   # kurz, granted, not begun
        _grant(f, rid, u, pp.CLASS_KURZ, now)


def test_front_belegfall_idle_d_goes_to_d_at_once_no_collect(monkeypatch, caplog):
    _on(monkeypatch)
    now = time.time()
    f = _fx(now, decoding=False)
    _belegfall_grants(f, now)
    _halb_set(f, "pdflip-0-4", 3821, now)
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        assert f._decode_collect(now) == "D"
    (line,) = _verdicts(caplog)
    assert "d_decoding=0" in line and "verdict=D_AFTER_TIMER" in line and "R8 no decode on D" in line, line
    assert "verdict=COLLECT" not in line and "NS:S49" not in line
    assert f.counters["set_xsum_hold_unstarted"] == 0 and f.counters["set_reroute_p"] == 0
    assert f._set_st()["released"].get("pdflip-0-4") == "D"


def test_front_idle_d_halb_grant_is_no_hold_either(monkeypatch, caplog):
    """R8 holds nothing: a halb grant (5200, not begun) on an idle D with a second halb arrival -> D at once (S < X)."""
    _on(monkeypatch)
    now = time.time()
    f = _fx(now, decoding=False)
    f.tp_prefill_max_tokens = 6008
    _grant(f, "s0", 5200, pp.CLASS_HALB, now)
    _halb_set(f, "s1", 5200, now)
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        assert f._decode_collect(now) == "D"
    assert f.counters["set_xsum_hold_unstarted"] == 0


def test_front_d_decodes_kurz_grants_never_hold_the_r3_timer_end(monkeypatch, caplog):
    """D decodes, T_halb over, S=3821 < X: the three kurz grants (2226) are no part of the X-SUM term (R2/R6k) -> D."""
    _on(monkeypatch)
    now = time.time()
    f = _fx(now, decoding=True)
    _belegfall_grants(f, now)
    _halb_set(f, "pdflip-0-4", 3821, now - 5.0)
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        assert f._decode_collect(now) == "D"
    (line,) = _verdicts(caplog)
    assert "verdict=D_AFTER_TIMER" in line and "R3 T_halb over" in line, line
    assert f.counters["set_xsum_hold_unstarted"] == 0 and f.counters["xsum_kurz_grant_skipped"] >= 3


def test_front_d_decodes_halb_grant_holds_then_r6_reroutes_once_begun(monkeypatch, caplog):
    """Gegenfall (NS:S49 at the R3 timer end while D decodes): a halb grant 2226 not begun + S 3821 > X -> COLLECT; D begins
    it (D-EXTEND-BEACON, redirectable) -> R6 counts it running -> REROUTE_P."""
    _on(monkeypatch)
    now = time.time()
    f = _fx(now, decoding=True)
    _grant(f, "h0", 2226, pp.CLASS_HALB, now)
    _halb_set(f, "pdflip-0-4", 3821, now - 5.0)
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        assert f._decode_collect(now) == "hold"
    (line,) = _verdicts(caplog)
    assert "verdict=COLLECT" in line and "NS:S49" in line and "d_decoding=1" in line, line
    assert f.counters["set_xsum_hold_unstarted"] == 1
    f._rvp_state()
    f.__dict__.setdefault("_d_prefill_legs", {})["h0"] = time.time()
    f._leg2_lookahead.add("h0")
    f.groups["D"].outstanding["h0"] = now - 0.05
    f._d_pf_book().enter_leg2(rid="h0")
    f._d_ext_reading = lambda: {dxb.rid_key("h0"): (time.time_ns(), 2226)}
    assert f._decode_collect(time.time()) == "P" and f.counters["set_reroute_p"] == 1


def test_fallback_without_decider_r8_is_not_held(monkeypatch, caplog):
    """FLLIPER_PDFLIP_DECIDER off (the S1 path, ``_set_xsum_hold``): R8 is not held either."""
    from flliper.srt.environ import envs
    _on(monkeypatch)
    now = time.time()
    f = _fx(now, decoding=False)
    _grant(f, "s0", 5200, pp.CLASS_HALB, now)
    f.tp_prefill_max_tokens = 6008
    _halb_set(f, "s1", 5200, now)
    with envs.FLLIPER_PDFLIP_DECIDER.override(False):
        assert f._decode_collect(now) == "D"
    assert f.counters["set_xsum_hold_unstarted"] == 0
