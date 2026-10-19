"""EZ-S1 review Nachbesserung (nf-review-ez-s1-1019.md, Auflagen 1-6; SOLL R2 R3 R4 R7 R8 R9 R11).

B1  a MEASURE returnee is EXECUTED by the class D measured (halb / gross / kurz), never by the state "requeued":
    the real path ``_requeue_after_x_refusal`` -> ``_set_cands`` -> ``_set_members`` -> ``set_verdict`` ->
    ``_decode_collect`` -> ``_asr_queued_short_to_d`` (halb alone, halb + halb S >= X, gross, under halbX, re-probe member);
    the DECISION record and the verdict name the same action;
B2  the fallback ``_x_exact_price -> None`` is MEASURE, never a class on chars/3;
B3  the report's ``under`` is ``u < hx`` of the BARRIER that went with the leg; a curve whose X falls with the depth makes
    the barrier from the smallest X over ``[0, n]``;
B4  the barrier is armed outside the instrument ``except``; a failure is loud and fails closed (front_x=1);
B5  ``KIND_P_UPPER`` (kurz on u_p) carries the barrier too: u_d > halbX > u_p is a report, no silent D prefill;
B7  the ALT shadow shows the decision of a ``(bool, ...)`` tuple and of a ``SetVerdict``.
"""
import asyncio
import logging
import os
import sys
import time
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("FLLIPER_PDFLIP_D_MATCH_PROBE_DEADLINE_MS", "150")
os.environ.setdefault("FLLIPER_PDFLIP_D_MATCH_PROBE_RETRY_MS", "10")

from flliper.srt.environ import envs  # noqa: E402
from flliper.srt.pdflip import decision_numbers as dn  # noqa: E402
from flliper.srt.pdflip import decision_record as dr  # noqa: E402
from flliper.srt.pdflip import front as F  # noqa: E402
from flliper.srt.pdflip import front_x  # noqa: E402
from flliper.srt.pdflip import phase_policy as pp  # noqa: E402
from test_pdflip_front_x_exact import X, _Req, _front  # noqa: E402
from test_decision_numbers_1019 import (HALF_X, N, PAGE, U_P, _drive, _img_payload, _mk, _parsed,  # noqa: E402
                                        _text_payload)


def _fresh(est_prompt=N):
    return _front(True)


def _arm(f, rid, u_p=-1, n=N):
    """The arrival side: the barrier armed and the leg 'sent' (what ``leg2`` does through ``_front_x_headers``)."""
    f._store_probe_info = {"page_size": PAGE}
    F.Front._measure_start(f, rid, X, "test", u_p=u_p, n_prompt=n)
    hdr = F.Front._front_x_headers(f, rid, None, False)
    assert hdr[front_x.HEADER_ACTION] == "MEASURE_ON_D"
    return hdr


def _pend(rid, est_prompt=N):
    p = F.Pending(rid, "/v1/messages", {}, "t", time.time(), asyncio.new_event_loop().create_future(),
                  est_prompt=est_prompt, est_uncached=0)
    p.x_set = X
    p.d_direct = True
    return p


def _refuse(f, rid, u, p=None, est_prompt=N):
    """D's refusal comes back through the REAL ``_requeue_after_x_refusal`` (the entry lands in ``f.queue``)."""
    p = p or _pend(rid, est_prompt)
    body = b'{"error": "W50: this request\'s extent after prefix matching is %d. Re-routed"}' % u

    async def go():
        t = asyncio.create_task(f._requeue_after_x_refusal(_Req({}, "/v1/messages"), rid, {}, "t", False, p, None, body))
        await asyncio.sleep(0.05)
        t.cancel()
        try:
            await t
        except BaseException:  # noqa: BLE001
            pass
    asyncio.run(go())
    p.fut = asyncio.new_event_loop().create_future()      # the cancelled await took the old one with it
    return p


def _returnee(f, rid, u, u_p=-1, est_prompt=N):
    _arm(f, rid, u_p, est_prompt)
    return _refuse(f, rid, u, est_prompt=est_prompt)


def _members(f):
    st = {"go_p": set(), "shorts": {}}
    cands = F.Front._set_cands(f, [q for q in f.queue if not q.fut.done()], st)
    return cands, F.Front._set_members(f, time.time(), cands, st)


def _verdict(f, mem, decoding=False):
    t = time.time()
    return pp.set_verdict(mem, x_tokens=X, half_frac=0.5, t_half_s=4.5, now=t, t_open=t, d_decoding=decoding,
                          seats_free=6)


def _set_records(caplog):
    return [d for d in _parsed(caplog) if d["kind"] == "set" and not d["rule"].startswith("ALT:")]


# =============================================================================================== B1: by the measured class

def test_b1_halb_alone_is_a_member_not_p_only_and_ends_on_d_with_no_flip(caplog):
    f = _fresh()
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        p = _returnee(f, "pdflip-0-1", 3000, u_p=3000)
    rec = [d for d in _parsed(caplog) if d["kind"] == "measure" and d["action"] != "MEASURE_ON_D"]
    assert [(d["action"], d["rule"], d["classes"]) for d in rec] == [("COLLECT", "R3", ["halb"])]
    assert p.measure_class == "halb" and p in f.queue
    cands, mem = _members(f)
    assert [c.rid for c in cands] == ["pdflip-0-1"]
    assert [m.p_only for m in mem] == [False], "R3: a halb member is not P-only by the state 'requeued'"
    sv = _verdict(f, mem)
    assert sv.verdict == pp.V_D_AFTER_TIMER and sv.s_tokens == 3000 < X, sv      # the verdict names what the record named
    now = time.time()
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        route = f._decode_collect(now)
    assert route == "D" and p.d_eligible and p.rid not in f._set_st()["go_p"], "no flip for one half request"
    assert [d["action"] for d in _set_records(caplog)] == [pp.V_D_AFTER_TIMER]
    moved = f._asr_queued_short_to_d([p], now)
    assert moved == [p] and p in f._ready_for_d
    assert p.x_decision[0] == X >= 3000, "D prefills it under the DECISION X, not the barrier halfX-1"


def test_b1_halb_plus_halb_reaching_x_flips_p_under_r3(caplog):
    f = _fresh()
    a = _returnee(f, "pdflip-0-1", 3000, u_p=3000)
    b = _returnee(f, "pdflip-0-2", 3000, u_p=3000)
    cands, mem = _members(f)
    assert len(cands) == 2 and [m.p_only for m in mem] == [False, False]
    sv = _verdict(f, mem)
    assert sv.verdict == pp.V_FLIP_P and sv.s_tokens == 6000 >= X, sv
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        assert f._decode_collect(time.time()) == "P"
    assert {a.rid, b.rid} <= f._set_st()["go_p"]
    assert not a.d_eligible and not b.d_eligible
    assert [d["action"] for d in _set_records(caplog)] == [pp.V_FLIP_P]


def test_b1_gross_is_a_big_member_and_flips_under_r8(caplog):
    f = _fresh()
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        p = _returnee(f, "pdflip-0-1", U_P, u_p=U_P)
    rec = [d for d in _parsed(caplog) if d["kind"] == "measure" and d["action"] != "MEASURE_ON_D"]
    assert [(d["action"], d["rule"], d["classes"]) for d in rec] == [("COLLECT", "R4", ["gross"])]
    cands, mem = _members(f)
    assert [c.rid for c in cands] == [p.rid] and not p.d_eligible
    sv = _verdict(f, mem)
    assert sv.verdict == pp.V_FLIP_P, "R8: big, no decode on D -> decided now"
    assert _verdict(f, mem, decoding=True).verdict == pp.V_COLLECT, "R4: with decode it waits T_halb for company"
    assert f._decode_collect(time.time()) == "P" and p.rid in f._set_st()["go_p"]


def test_b1_under_halfx_is_no_member_and_goes_to_d_with_an_r7_defect_line(caplog):
    f = _fresh()
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        p = _returnee(f, "pdflip-0-1", HALF_X - 1, u_p=HALF_X - 1)
    rec = [d for d in _parsed(caplog) if d["kind"] == "measure" and d["action"] != "MEASURE_ON_D"]
    assert [(d["action"], d["rule"], d["classes"]) for d in rec] == [("TO_D", "R7", ["kurz"])]
    assert "d_refused_under_x" in rec[0]["why"]
    cands, mem = _members(f)
    assert cands == [] and mem == [], "no set member, so never a flip reason (R7)"
    assert p.d_eligible and p.measure_class == "kurz"
    assert pp.immediate_park_trigger([p], X) is None, "not P-bound by the state 'requeued'"
    now = time.time()
    assert f._decode_collect(now) is None and not f._set_st()["go_p"]
    assert f._asr_queued_short_to_d([p], now) == [p] and p in f._ready_for_d
    assert _parsed(caplog, "DECISION-DONE")[-1]["outcome"] == "mismatch"


def test_b1_reprobe_member_runs_the_same_way(caplog):
    f = _fresh()
    f._store_probe_info = {"page_size": PAGE}
    p = _pend("pdflip-3-5", est_prompt=9000)
    p.est_uncached = 7000
    f.queue.append(p)
    f._set_depth = {"pdflip-3-5": 2000}
    f._set_reprobe_unpriced("pdflip-3-5", SimpleNamespace(t=time.monotonic()), "no exact D number", "set_verdict")
    assert F.Front._front_x_headers(f, "pdflip-3-5", None, False)[front_x.HEADER_ACTION] == "MEASURE_ON_D"
    f.queue.remove(p)
    _refuse(f, "pdflip-3-5", 3000, p=p, est_prompt=9000)
    assert p.measure_class == "halb"
    cands, mem = _members(f)
    assert [m.p_only for m in mem] == [False]
    assert _verdict(f, mem).verdict == pp.V_D_AFTER_TIMER
    assert f._decode_collect(time.time()) == "D" and p.d_eligible


def test_b1_w123_p_only_stays_p_only(caplog):
    """R9: an image D refused by name (W123) is P-only whatever the measured class -- the exemption is for the class only."""
    f = _fresh()
    _arm(f, "pdflip-0-1", -1)
    p = _pend("pdflip-0-1")
    body = (b'{"error": "W50: this request\'s extent after prefix matching is 3000. Re-routed (vision reroute) W123"}')

    async def go():
        t = asyncio.create_task(f._requeue_after_x_refusal(_Req({}, "/v1/messages"), "pdflip-0-1", {}, "t", False, p, None, body))
        await asyncio.sleep(0.05)
        t.cancel()
        try:
            await t
        except BaseException:  # noqa: BLE001
            pass
    asyncio.run(go())
    p.fut = asyncio.new_event_loop().create_future()
    assert p.p_only and not p.d_eligible
    _, mem = _members(f)
    assert [m.p_only for m in mem] == [True]


def test_b1_a_non_measure_requeue_is_unchanged():
    f = _fresh()
    p = _refuse(f, "pdflip-0-9", 12000)
    assert p.measure_class == "" and not p.d_eligible
    _, mem = _members(f)
    assert [m.p_only for m in mem] == [False] or mem == []     # u=12000 > X: big by size, as before
    assert pp.immediate_park_trigger([p], X) is p


# =============================================================================================== B2: the fallback is MEASURE

def _fallback_sig(monkeypatch, payload, chars3, caplog):
    f = _mk(monkeypatch, store_u_p=U_P)

    async def none(*a, **k):
        return None
    f._x_exact_price = none
    monkeypatch.setattr(F, "price_remainder", lambda text, spans, epoch=None: (chars3, chars3, False))
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        _drive(f, payload())
    sig = [(d["kind"], d["action"], d["rule"], tuple(d["classes"]), d["X"], d["halfX"],
            tuple((m.u_own, m.src) for m in d["members"])) for d in _parsed(caplog) if d["kind"] in ("measure", "arrival")]
    return f, sig, [r[0] for r in f.routed]


@pytest.mark.parametrize("payload", [_text_payload, _img_payload])
def test_b2_x_exact_none_is_measure_and_chars3_changes_nothing(monkeypatch, caplog, payload):
    f1, s1, r1 = _fallback_sig(monkeypatch, payload, 5, caplog)
    f2, s2, r2 = _fallback_sig(monkeypatch, payload, 10 ** 9, caplog)
    assert s1 and s1 == s2 and r1 == r2, (s1, s2)
    assert [s[:3] for s in s1 if s[0] == "measure"] == [("measure", "MEASURE_ON_D", "NS:MEASURE")]
    assert not [s for s in s1 if s[0] == "arrival" and s[6] and s[6][0][1] == "n_est"], "no class with src=n_est"
    assert f1.counters["x_exact_fallback_measure"] == 1 and f1.counters["measure_on_d"] == 1


def test_b2_switch_off_keeps_the_rc17_fallback(monkeypatch, caplog):
    with envs.FLLIPER_PDFLIP_EXACT_CLASS.override(False):
        f, sig, _ = _fallback_sig(monkeypatch, _text_payload, 5, caplog)
        assert not [s for s in sig if s[0] == "measure"] and f.counters["x_exact_fallback_measure"] == 0


# =============================================================================================== B3: the barrier's own halbX

def test_b3_under_is_judged_by_the_barrier_that_went_with_the_leg(caplog, monkeypatch):
    """Live X rose after the arrival: halbX 2048 at the barrier, 4096 now. u=3000 is kurz at the X now but D refused it
    for the barrier it had -- that is the class at the valid number, no R7 defect, no mismatch."""
    f = _fresh()
    _arm(f, "pdflip-0-1", -1)
    live = {"x": X * 2}
    monkeypatch.setattr(F.Front, "_x_at_depth", lambda self, depth, k=1: live["x"])
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        p = _refuse(f, "pdflip-0-1", 3000)
    rec = [d for d in _parsed(caplog) if d["kind"] == "measure" and d["action"] != "MEASURE_ON_D"]
    assert [(d["action"], d["rule"], d["classes"]) for d in rec] == [("TO_D", "R2", ["kurz"])]
    assert "defect" not in rec[0]["why"] and _parsed(caplog, "DECISION-DONE")[-1]["outcome"] == "ok"
    assert p.d_eligible and _members(f)[1] == []


def test_b3_the_sent_record_carries_the_barriers_halfx():
    f = _fresh()
    _arm(f, "pdflip-0-1", -1)
    assert f._measure_sent["pdflip-0-1"][2] == HALF_X


class _Rows:
    def __init__(self, depths):
        self.rows = [SimpleNamespace(depth=d) for d in depths]


def test_b3_curve_mode_barrier_is_the_smallest_x_over_zero_to_n(monkeypatch):
    f = _fresh()
    f._store_probe_info = {"page_size": PAGE}
    curve = {0: 4096, 5000: 3000, 10000: 2000, 20000: 1000, 30000: 600}      # X FALLS with the depth
    f._x_setup = SimpleNamespace(curves=SimpleNamespace(p_curve=_Rows([0, 10000, 30000]), d_curve=_Rows([5000, 20000])))
    monkeypatch.setattr(F.Front, "_x_at_depth", lambda self, depth, k=1: curve[max(d for d in curve if d <= depth)])
    assert F.Front._x_min_over_depths(f, 25000) == 1000      # the row at 20000 is inside [0, 25000]; 30000 is not
    assert F.Front._x_min_over_depths(f, 100) == 4096
    assert F.Front._x_min_over_depths(f, 30000) == 600
    F.Front._measure_start(f, "pdflip-0-1", curve[0], "test", u_p=-1, n_prompt=25000)
    hdr = F.Front._front_x_headers(f, "pdflip-0-1", None, False)
    hx = pp.half_x_of(1000, 0.5)
    assert hdr[front_x.HEADER_HALFX] == str(hx) and int(hdr[front_x.HEADER_X]) + PAGE == hx - 1, hdr
    # the old behaviour (X at depth 0) would have let D admit u up to 2047 although the class at 20000 deep is halb from 500
    assert front_x.limit_for(X, int(hdr[front_x.HEADER_X]), PAGE) == hx - 1 < pp.half_x_of(4096, 0.5) - 1


def test_b3_without_curves_x_is_the_x_in_force():
    f = _fresh()
    assert F.Front._x_min_over_depths(f, 100000) == X


# =============================================================================================== B4: the barrier outside the except

def test_b4_a_failing_instrument_does_not_take_the_barrier_along(monkeypatch):
    f = _fresh()
    f._store_probe_info = {"page_size": PAGE}
    monkeypatch.setattr(F.Front, "_dec_group", lambda self: (_ for _ in ()).throw(RuntimeError("instrument down")))
    xx = F.Front._x_unpriced(f, "pdflip-0-1", None, 777, -1, True)
    assert xx.measure
    F.Front._dec_arrival(f, "pdflip-0-1", xx, remainder=0, x_set=X, route="short", p_only=False,
                         t_arrive_m=time.monotonic(), n_prompt=N)
    hdr = F.Front._front_x_headers(f, "pdflip-0-1", None, False)
    assert hdr[front_x.HEADER_ACTION] == "MEASURE_ON_D" and int(hdr[front_x.HEADER_X]) + PAGE == HALF_X - 1


def test_b4_a_failing_barrier_is_loud_and_fails_closed(monkeypatch, caplog):
    f = _fresh()
    f._store_probe_info = {"page_size": PAGE}

    def boom(self, rid, x_set, seq=0, u_p=-1):
        raise RuntimeError("no barrier")
    monkeypatch.setattr(F.Front, "_measure_register", boom)
    xx = F.Front._x_unpriced(f, "pdflip-0-1", None, 777, -1, True)
    with caplog.at_level(logging.WARNING, logger="pdflip.front"):
        F.Front._dec_arrival(f, "pdflip-0-1", xx, remainder=0, x_set=X, route="short", p_only=False,
                             t_arrive_m=time.monotonic(), n_prompt=N)
    assert f.counters["measure_barrier_failed"] == 1
    assert any("MEASURE-BARRIER-FAILED" in m for m in caplog.messages)
    hdr = F.Front._front_x_headers(f, "pdflip-0-1", None, False)
    assert hdr[front_x.HEADER_X] == "1" and hdr[front_x.HEADER_ACTION] == "MEASURE_ON_D", "never the decision X"
    assert front_x.limit_for(X, 1, PAGE) == 1 + PAGE < HALF_X
    # D's refusal is still a report
    assert f._measure_sent["pdflip-0-1"][1] == -1


# =============================================================================================== B5: kurz on u_p carries the barrier

def test_b5_p_upper_carries_the_barrier_and_a_halb_u_d_is_reported_not_prefilled(caplog):
    f = _fresh()
    f._store_probe_info = {"page_size": PAGE}
    xx = F.Front._x_unpriced(f, "pdflip-0-1", SimpleNamespace(n=N), 777, 202, False)
    assert xx.class_basis.kind == dn.KIND_P_UPPER and not xx.measure and xx.pending == 202
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        F.Front._dec_arrival(f, "pdflip-0-1", xx, remainder=202, x_set=X, route="short", p_only=False,
                             t_arrive_m=time.monotonic(), n_prompt=N)
    arr = [d for d in _parsed(caplog) if d["kind"] == "arrival" and not d["rule"].startswith("ALT:")]
    assert [(d["action"], d["classes"]) for d in arr] == [("CLASSIFY", ["kurz"])], "the class stays kurz on u_p"
    hdr = F.Front._front_x_headers(f, "pdflip-0-1", None, False)
    assert hdr[front_x.HEADER_ACTION] == "MEASURE_ON_D"
    u_d = 3000                                   # u_d > halbX > u_p: the store did not reach D
    assert 202 < HALF_X <= u_d
    assert u_d > front_x.limit_for(X, int(hdr[front_x.HEADER_X]), PAGE), "D refuses it: a report, no silent D prefill"
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        p = _refuse(f, "pdflip-0-1", u_d)
    assert p.measure_class == "halb"
    assert _parsed(caplog, "DECISION-DONE")[-1]["outcome"] == "mismatch", "u_d > u_p + page names itself"


def test_b5_e2e_a_p_upper_arrival_arms_the_barrier(monkeypatch, caplog):
    f = _mk(monkeypatch, store_u_p=202)
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        _drive(f, _text_payload())
    assert [d["classes"] for d in _parsed(caplog) if d["kind"] == "arrival" and not d["rule"].startswith("ALT:")] == [["kurz"]]
    rid = f.routed[0][1]
    assert rid in f._measure_on_d and f._measure_on_d[rid][0] == HALF_X


def test_b5_switch_off_arms_nothing():
    with envs.FLLIPER_PDFLIP_EXACT_CLASS.override(False):
        f = _fresh()
        xx = F.Front._x_unpriced(f, "pdflip-0-1", SimpleNamespace(n=N), 777, 202, False)
        F.Front._dec_arrival(f, "pdflip-0-1", xx, remainder=0, x_set=X, route="short", p_only=False,
                             t_arrive_m=time.monotonic(), n_prompt=N)
        assert not f.__dict__.get("_measure_on_d")


# =============================================================================================== B7: the shadow shows the decision

def test_b7_summarize_shows_the_bool_of_a_tuple_and_the_verdict_of_a_set_verdict():
    assert dr.summarize((True, "seat fits")) == "true" and dr.summarize((False, "no seat", 3)) == "false"
    sv = pp.SetVerdict(pp.V_FLIP_P, "why", 0.0, 5000, 2048)
    assert dr.summarize(sv) == pp.V_FLIP_P
    assert dr.summarize((1, 2, 3)) == "n3" and dr.summarize(None) == "none" and dr.summarize(True) == "true"
