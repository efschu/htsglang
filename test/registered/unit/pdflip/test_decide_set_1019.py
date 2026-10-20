"""EZ-S2a (plan nf-entscheidungszentrale-plan-1019, S2a; SOLL R1 R2 R2a R3 R4 R5 R6 R7 R8 R9 R11): ``phase_policy.decide(kind="set")``
and the front glue -- ONE decision for the set/controller core.

(a) one case per rule on ``decide`` (the section-5 examples live in ``test_pdflip_scheduler_soll_1017``, byte-compat in
    ``test_decide_s4_compat_1019``);
(b) front: one X per member (x_own, M12), S from the members' OWN u (M8), a kurz member never in a set (M11), p_only only by
    R9 (A5, M16), the S49 hold inside the verdict (M6), the queue-sum second set gone (FLIP-HELD = DECISION rule R7),
    FLIP-ECONOMICS / VISION-FLIP-URGENT / ``_set_flip_foreseeable`` no verdicts, ``_leg2_verdict`` on x_own;
(c) switch ``FLLIPER_PDFLIP_DECIDER=0``: the S1 code (p_only by state, the queue-sum set, the economics verdict).
"""
import asyncio
import logging
import os
import sys
import time
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flliper.srt.environ import envs  # noqa: E402
from flliper.srt.pdflip import decision_record as dr  # noqa: E402
from flliper.srt.pdflip import front as F  # noqa: E402
from flliper.srt.pdflip import front_decider as fd  # noqa: E402
from flliper.srt.pdflip import phase_policy as pp  # noqa: E402
from test_ez_s1_review_1019 import _fresh, _members, _pend, X, HALF_X  # noqa: E402
from test_decision_numbers_1019 import _parsed  # noqa: E402

T_HALF = 4.5


def M(rid, u, t=0.0, p_only=False, own=0, x_own=0):
    return pp.SetMember(rid=rid, uncached=u, t_arrive=t, p_only=p_only, own_uncached=own, x_own=x_own)


def D(members, *, now=1.0, t_open=0.0, decoding=True, seats=8, running=0, unstarted=0, x=6000):
    return pp.decide("set", members=members, x_tokens=x, half_frac=0.5, t_half_s=T_HALF, now=now, t_open=t_open,
                     d_decoding=decoding, seats_free=seats, d_prefill_running_uncached=running, new_since_last=0,
                     d_granted_unstarted=unstarted)


# ------------------------------------------------------------------------------------------------------ (a) rules on decide

def test_decide_kinds_are_set_admit_arrival_since_s2b():
    # EZ-S2b: 'admit' and 'arrival' now exist (test_ez_s2b_1020); anything else is still refused
    with pytest.raises(ValueError):
        pp.decide("park", members=[])


def test_r1_set_sum_decides():
    assert D([M("a", 3500)], now=2.0).verdict.verdict != pp.V_FLIP_P
    d = D([M("a", 3500), M("b", 3200, 1.0)], now=2.0)
    assert (d.verdict.verdict, d.verdict.s_tokens, d.rule) == (pp.V_FLIP_P, 6700, "R3")


def test_r2_kurz_never_in_the_set_never_in_s_m11():
    """R2/M11: 3500 (halb), 300 (kurz), 2800 (kurz at X 6000): S = 3500, no flip; the kurz ones are named, not counted."""
    d = D([M("a", 3500), M("k", 300, 0.5), M("c", 2800, 2.0)], now=2.5)
    assert d.verdict.s_tokens == 3500 and d.verdict.verdict == pp.V_COLLECT
    assert d.in_set == ("a",) and d.kurz_left_out == ("k", "c") and d.classes == ("halb",)


def test_r2a_class_follows_the_members_own_x_m12():
    """R2a/M12: u = 2000 is kurz at X 6000 (halbX 3000) but halb at its OWN X 3000 (halbX 1500): it is a member."""
    stat = D([M("a", 2000)], now=1.0)
    own = D([M("a", 2000, x_own=3000)], now=1.0)               # the set X stays 6000: only the member knows its own
    assert stat.in_set == () and own.in_set == ("a",) and own.classes == ("halb",)


def test_r3_timer_and_s_reaching_x():
    assert D([M("a", 3500)], now=2.0).verdict.verdict == pp.V_COLLECT
    d = D([M("a", 3500)], now=5.0)
    assert (d.verdict.verdict, d.rule) == (pp.V_D_AFTER_TIMER, "R3")


def test_r4_big_waits_t_halb_for_company():
    assert D([M("g", 9000)], now=1.0).verdict.verdict == pp.V_COLLECT
    assert D([M("g", 9000), M("h", 3100, 1.0)], now=1.0).verdict.verdict == pp.V_FLIP_P


def test_r5_no_seat_parks():
    d = D([M("a", 3500), M("b", 3200, 1.0)], now=2.0, seats=0)
    assert d.verdict.verdict == pp.V_PARK_NO_SEAT


def test_r6_running_d_prefill_redirects():
    d = D([M("a", 5000)], now=1.0, running=4000)
    assert (d.verdict.verdict, d.rule) == (pp.V_REROUTE_P, "R6") and d.verdict.s_tokens == 9000


def test_r8_no_decode_decides_now():
    assert D([M("g", 9000)], decoding=False).verdict.verdict == pp.V_FLIP_P
    assert D([M("a", 3500)], decoding=False).verdict.verdict == pp.V_D_AFTER_TIMER


def test_r9_p_only_counts_big():
    d = D([M("v", 100, p_only=True)], now=1.0)
    assert d.classes == ("p_only",) and d.verdict.verdict == pp.V_COLLECT      # big waits T_halb (R4)
    assert D([M("v", 100, p_only=True)], decoding=False).verdict.verdict == pp.V_FLIP_P


def test_m8_s_is_the_union_figure_r1u_not_the_sum_of_own_u():
    """SOLL R1u (user decision 20.10., supersedes A4/M8): a shared uncached piece counts ONCE in S (uncached 1000 each after the
    discount, own 3500 + 3200 only class the members): S = 2000 < X, no flip."""
    d = D([M("a", 1000, own=3500), M("b", 1000, 1.0, own=3200)], now=2.0)
    assert d.verdict.s_tokens == 2000 and d.verdict.verdict != pp.V_FLIP_P


def test_s49_granted_unstarted_is_no_r6_reason_but_holds_a_d_verdict_m6():
    """Plan 2.6/M6: 3500 in the set, 5000 GRANTED on D but not begun: no R6 redirect (nothing runs), the D verdict is held
    (NS:S49, COLLECT, no flip); begun (running) the same 5000 redirects (R6)."""
    d = D([M("a", 3500)], now=5.0, unstarted=5000)
    assert (d.verdict.verdict, d.rule) == (pp.V_COLLECT, "NS:S49") and d.verdict.s_tokens == 3500
    assert dr.rule_of_set_why(d.verdict.why, d.verdict.verdict) == "NS:S49"
    assert D([M("a", 3500)], now=5.0, unstarted=2000).verdict.verdict == pp.V_D_AFTER_TIMER      # 3500 + 2000 <= X
    assert D([M("a", 3500)], now=5.0, running=5000).verdict.verdict == pp.V_REROUTE_P


def test_s49_a_flip_verdict_is_never_held():
    assert D([M("g", 9000), M("h", 3100, 1.0)], now=1.0, unstarted=9000).verdict.verdict == pp.V_FLIP_P


# ------------------------------------------------------------------------------------------------------ (b) front glue

def _queue(f, rid, u, **attrs):
    p = _pend(rid)
    p.est_uncached = u
    p.x_set = attrs.pop("x_set", X)
    p.d_direct = False
    for k, v in attrs.items():
        setattr(p, k, v)
    f.queue.append(p)
    return p


def test_set_state_is_the_decider_state_and_resets_on_a_new_epoch():
    f = _fresh()
    st = f._set_st()
    assert isinstance(st, fd.DeciderState)
    st["go_p"].add("x")
    f.epoch += 1
    assert f._set_st()["go_p"] == set()


def test_a5_requeued_halb_is_a_member_by_class_not_p_only_m16():
    f = _fresh()
    p = _queue(f, "pdflip-0-1", 3500, x_requeues=1)
    cands, mem = _members(f)
    assert cands == [p] and [m.p_only for m in mem] == [False]
    assert pp.classify_arrival(3500, X, 0.5, mem[0].p_only) == "halb"


def test_a5_requeued_under_halbx_is_no_member_no_flip_and_an_r7_record(caplog):
    f = _fresh()
    _queue(f, "pdflip-0-1", HALF_X - 1, x_requeues=1)
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        cands, mem = _members(f)
    assert cands == [] and mem == []
    r7 = [d for d in _parsed(caplog) if d["rule"] == "R7" and d["action"] == "WAIT_SEAT"]
    assert len(r7) == 1 and r7[0]["rid"] == "pdflip-0-1" and "d_refused_under_x" in r7[0]["why"]


def test_a5_requeued_gross_is_gross_not_p_only():
    f = _fresh()
    _queue(f, "pdflip-0-1", X + 500, x_requeues=1)
    _cands, mem = _members(f)
    assert [m.p_only for m in mem] == [False]
    assert pp.classify_arrival(X + 500, X, 0.5, mem[0].p_only) == "gross"


def test_r9_p_only_entry_stays_p_only():
    f = _fresh()
    _queue(f, "pdflip-0-1", 35, p_only=True)
    _cands, mem = _members(f)
    assert [m.p_only for m in mem] == [True]


def test_m12_the_member_carries_its_own_x():
    f = _fresh()
    _queue(f, "pdflip-0-1", 2000, x_set=3000, x_requeues=1)       # halb at its own X (halbX 1500), kurz at the static 6000
    cands, mem = _members(f)
    assert [c.rid for c in cands] == ["pdflip-0-1"] and mem[0].x_own == 3000
    assert pp.decide("set", members=mem, x_tokens=3000, half_frac=0.5, t_half_s=4.5, now=time.time(), t_open=time.time(),
                     d_decoding=True, seats_free=6).in_set == ("pdflip-0-1",)


def test_needs_p_reads_the_class_on_the_entry_x_not_the_state():
    p = SimpleNamespace(est_uncached=2000, x_requeues=1, x_set=X, skip_leg1=False, leg1_done=False, p_only=False)
    with envs.FLLIPER_PDFLIP_DECIDER.override(True):
        assert pp.immediate_park_trigger([p], X) is None, "D refused it under X: a defect (R7), not a flip reason"
        assert pp.immediate_park_trigger([SimpleNamespace(**{**vars(p), "est_uncached": X})], X) is not None     # u >= X: gross
        assert pp.immediate_park_trigger([SimpleNamespace(**{**vars(p), "est_uncached": 100, "x_requeues": 0, "p_only": True})],
                                         X) is not None
    with envs.FLLIPER_PDFLIP_DECIDER.override(False):
        assert pp.immediate_park_trigger([p], X) is p, "switch off: the S1 rule (requeued = P by state)"


def test_one_set_queue_sum_of_kurz_entries_is_no_flip_and_flip_held_is_an_r7_record(caplog):
    f = _fresh()
    for i in range(3):
        _queue(f, "pdflip-0-%d" % i, 2900)                       # three kurz entries, sum 8700 >= X: the old second set flipped
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        assert F.Front._set_flip_justified(f, time.time()) is False
    assert f._set_st().get("go_p") == set()
    held = [d for d in _parsed(caplog) if d["rule"] == "R7" and d["action"] == "FLIP_HELD"]
    assert len(held) == 1 and held[0]["S"] == 8700
    assert any("FLIP-HELD" in m for m in caplog.messages), "the alias line stays"


def test_switch_off_the_queue_sum_second_set_still_flips():
    f = _fresh()
    for i in range(3):
        _queue(f, "pdflip-0-%d" % i, 2900)
    with envs.FLLIPER_PDFLIP_DECIDER.override(False):
        assert F.Front._set_flip_justified(f, time.time()) is True


def test_flip_foreseeable_reads_the_one_decision_collecting_is_no_flip(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_HALF_FLIP_S", "4.5")             # other test modules set T_halb at import time
    f = _fresh()
    p = _queue(f, "pdflip-0-1", 9000)                                  # gross, D decoding, T_halb 4.5 s: the set COLLECTS
    f._dc_decoding = lambda: True
    assert F.Front._set_flip_foreseeable(f, time.time()) is False
    f._set_st()["go_p"].add(p.rid)                                      # the decision committed
    assert F.Front._set_flip_foreseeable(f, time.time()) is True
    f2 = _fresh()
    _queue(f2, "pdflip-0-2", 9000)
    f2._dc_decoding = lambda: False                                     # R8: nothing decodes -> decided now
    assert F.Front._set_flip_foreseeable(f2, time.time()) is True


def test_flip_economics_is_a_report_not_a_verdict(caplog):
    f = _fresh()
    _queue(f, "pdflip-0-1", 100)
    f.admit_d = True
    with envs.FLLIPER_PDFLIP_DECIDER.override(False), caplog.at_level(logging.INFO, logger="pdflip.front"):
        off = f._flip_economics_ok(False)
    with envs.FLLIPER_PDFLIP_DECIDER.override(True), caplog.at_level(logging.INFO, logger="pdflip.front"):
        on = f._flip_economics_ok(False)
    assert off is False and on is True
    assert any("FLIP-ECONOMICS" in m and "verdict=report" in m for m in caplog.messages)


def test_leg2_verdict_measures_against_x_own(monkeypatch):
    f = _fresh()
    monkeypatch.setattr(F.Front, "_x_own", lambda self, depth: 1000)
    with envs.FLLIPER_PDFLIP_DECIDER.override(True):
        assert f._leg2_verdict(3000, 1500, True, None, False, False, "r") == "short_mispriced"
    with envs.FLLIPER_PDFLIP_DECIDER.override(False):
        assert f._leg2_verdict(3000, 1500, True, None, False, False, "r") == "serve"


def test_x_own_is_the_one_x_function():
    f = _fresh()
    assert F.Front._x_own(f, 12345) == F.Front._x_at_depth(f, 12345)


def test_set_record_names_unstarted_and_the_s49_tag(caplog, monkeypatch):
    """The DECISION record of a held set carries ``rule=NS:S49`` and ``unstarted=``; Vorgabe 0 elsewhere."""
    f = _fresh()
    p = _queue(f, "pdflip-0-1", 3500, x_requeues=1)
    monkeypatch.setattr(fd, "granted_unstarted", lambda *a, **k: 5000)
    now = time.time()
    f._set_st()["t_open"] = now - 10.0                           # the timer is over: a D verdict would stand
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        assert f._decode_collect(now) == "hold"
    rec = [d for d in _parsed(caplog) if d["kind"] == "set" and d["rule"] == "NS:S49"]
    assert rec and rec[-1]["unstarted"] == 5000 and p.rid not in f._set_st()["go_p"]


def test_r9_n1_a_leg2_reroute_is_p_by_state_big_even_under_x():
    """R9/N1 stays: a request that must run on P by state (leg-2 reroute) is p_only = big under X (A5 drops only the
    requeued / carrier / deferred state)."""
    f = _fresh()
    _queue(f, "pdflip-0-1", 200, reroutes=1)
    cands, mem = _members(f)
    assert [c.rid for c in cands] == ["pdflip-0-1"] and [m.p_only for m in mem] == [True]


def test_refused_without_an_extent_is_p_by_state_not_the_stale_estimate_g8():
    """D refused (W50) and named NO extent: there is no number, D's word stands -- P-bound and a big member, never a kurz class
    on the front's estimate (no wedge: the entry would be neither D's nor P's)."""
    f = _fresh()
    p = _queue(f, "pdflip-0-1", 50, x_requeues=1, x_ref_nonum=True)
    assert pp.immediate_park_trigger([p], X) is p
    cands, mem = _members(f)
    assert [c.rid for c in cands] == ["pdflip-0-1"] and [m.p_only for m in mem] == [True]
    q = _queue(f, "pdflip-0-2", 50, x_requeues=1, x_ref_nonum=False)          # a number named: under halbX = defect, no member
    assert pp.immediate_park_trigger([q], X) is None


def test_g9_a_set_of_kurz_only_shorts_releases_to_d_never_holds(caplog):
    """R2: the SHORTs on the seat path whose exact u reads kurz are not a set -- ``_decode_collect`` answers D at once."""
    f = _fresh()
    st = f._set_st()
    st["shorts"]["a"] = (time.time(), 500, 0)
    st["shorts"]["b"] = (time.time(), 500, 0)
    assert f._decode_collect(time.time()) == "D"
    assert st["released"] == {"a": "D", "b": "D"}
