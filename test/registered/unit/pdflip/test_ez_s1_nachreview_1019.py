"""EZ-S1 Nach-Review 2 (nf-review2-ez-s1-1019.md, N1-N3; SOLL R2 R3 R4 R7 R8 R9 R11).

N1  a measured entry that D refuses a SECOND time is never an orphan: the class is re-formed on the extent D names now
    (kurz -> D again, halb -> the set, gross -> R4/R8), without an extent the class is deleted; exactly ONE owner takes it;
N2  ``_x_min_over_depths`` is the EXACT minimum of X(depth): the kinks at ``row - k*chunk`` and at the flip price's depths are
    candidates (the two cases of the review: 4207 at a chunk edge, 3074 at the flip-price support point); an estimated size
    (media / chars/3 fallback) spans the whole measured range; ``_measure_return`` takes the PROMPT length, not the counter;
N3  a kurz-on-u_p leg (KIND_P_UPPER) has its own action ``D_NOW_BARRIER`` and counter, not ``MEASURE_ON_D``.
"""
import asyncio
import logging
import os
import sys
import time
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flliper.srt.pdflip import decision_numbers as dn  # noqa: E402
from flliper.srt.pdflip import front as F  # noqa: E402
from flliper.srt.pdflip import front_x  # noqa: E402
from flliper.srt.pdflip import phase_policy as pp  # noqa: E402
from flliper.srt.pdflip import x_curves as xc  # noqa: E402
from test_ez_s1_review_1019 import (_fresh, _members, _pend, _refuse, _returnee, _verdict, X, HALF_X, N, PAGE,  # noqa: E402
                                    _arm)
from test_decision_numbers_1019 import _parsed  # noqa: E402
import test_pdflip_x_curves_1006 as T  # noqa: E402


def _owners(f, p):
    """Which of the three ways takes the queued entry: D (``_asr_queued_short_to_d``), the set (a member of ``_set_cands``),
    P by state (``needs_p`` / immediate park)."""
    now = time.time()
    own = []
    cands, _mem = _members(f)
    if p in cands:
        own.append("set")
    if f._asr_queued_short_to_d([p], now) == [p]:
        own.append("D")
    if pp.immediate_park_trigger([p], X):
        own.append("P")
    return own


def _second_leg(f, p, rid, u, body=None):
    """The leg to D went out (taken by ``_asr_queued_short_to_d`` or the set) and D refuses AGAIN."""
    for q in (f._ready_for_d,):
        while p in q:
            q.remove(p)
    while p in f.queue:
        f.queue.remove(p)
    p.d_direct = True
    if body is None:
        return _refuse(f, rid, u, p=p)
    return _refuse_body(f, rid, body, p)


def _refuse_body(f, rid, body, p):
    async def go():
        t = asyncio.create_task(f._requeue_after_x_refusal(F._Req({}, "/v1/messages") if hasattr(F, "_Req") else _req(),
                                                           rid, {}, "t", False, p, None, body))
        await asyncio.sleep(0.05)
        t.cancel()
        try:
            await t
        except BaseException:  # noqa: BLE001
            pass
    asyncio.run(go())
    p.fut = asyncio.new_event_loop().create_future()
    return p


def _req():
    from test_pdflip_front_x_exact import _Req
    return _Req({}, "/v1/messages")


# ======================================================================================================== N1

def test_n1_kurz_refused_twice_is_taken_by_exactly_one_owner_d():
    f = _fresh()
    p = _returnee(f, "pdflip-0-1", HALF_X - 1, u_p=HALF_X - 1)
    assert _owners(f, p) == ["D"]
    p = _second_leg(f, p, "pdflip-0-1", HALF_X - 1)
    assert p.x_requeues == 2 and p.measure_class == "kurz" and p.d_eligible and p in f.queue
    assert _owners(f, p) == ["D"], "the review's orphan: no D, no set, no P"
    assert f.counters["measure_reclass_kurz"] == 1


def test_n1_kurz_then_halb_extent_is_a_set_member_not_stale_kurz(caplog):
    f = _fresh()
    p = _returnee(f, "pdflip-0-1", HALF_X - 1, u_p=HALF_X - 1)
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        p = _second_leg(f, p, "pdflip-0-1", 3000)           # D's extent grew between the legs (3000 in [halbX, X))
    assert p.measure_class == "halb" and p.measured_u == 3000
    cands, mem = _members(f)
    assert [c.rid for c in cands] == ["pdflip-0-1"] and [m.p_only for m in mem] == [False]
    sv = _verdict(f, mem)
    assert sv.verdict == pp.V_D_AFTER_TIMER and sv.s_tokens == 3000
    rec = [d for d in _parsed(caplog) if d["kind"] == "measure" and d["action"] != "MEASURE_ON_D"]
    assert [(d["action"], d["rule"], d["classes"]) for d in rec] == [("COLLECT", "R3", ["halb"])]
    assert "AGAIN" in rec[0]["why"]
    assert f._decode_collect(time.time()) == "D" and p.d_eligible, "alone: D after the timer, no flip"


def test_n1_kurz_then_halb_plus_halb_reaches_x_and_flips():
    f = _fresh()
    p = _returnee(f, "pdflip-0-1", HALF_X - 1, u_p=HALF_X - 1)
    p = _second_leg(f, p, "pdflip-0-1", 3000)
    q = _returnee(f, "pdflip-0-2", 3000, u_p=3000)
    cands, mem = _members(f)
    assert len(cands) == 2 and _verdict(f, mem).verdict == pp.V_FLIP_P


def test_n1_kurz_then_gross_extent_is_the_set_under_r4_r8(caplog):
    f = _fresh()
    p = _returnee(f, "pdflip-0-1", HALF_X - 1, u_p=HALF_X - 1)
    p = _second_leg(f, p, "pdflip-0-1", 3 * X)
    assert p.measure_class == "gross" and not p.d_eligible
    cands, mem = _members(f)
    assert [c.rid for c in cands] == ["pdflip-0-1"]
    assert _verdict(f, mem).verdict == pp.V_FLIP_P, "R8: a big member with D not decoding flips"
    assert _owners(f, p).count("D") == 0


def test_n1_second_refusal_without_extent_deletes_the_class_and_the_state_path_takes_it():
    f = _fresh()
    p = _returnee(f, "pdflip-0-1", HALF_X - 1, u_p=HALF_X - 1)
    assert p.measure_class == "kurz"
    p = _second_leg(f, p, "pdflip-0-1", 0, body=b'{"error": "W50: refused, no extent named"}')
    assert p.measure_class == "" and p.measure_src == "" and p.measured_u is None
    assert f.counters["measure_reclass_cleared"] == 1 and p in f.queue
    assert not p.d_eligible, "no class: the pre-EZ state path (P) as before, not a stale kurz"


def test_n1_a_non_measured_entry_is_untouched():
    f = _fresh()
    p = _refuse(f, "pdflip-0-5", 3000)               # never armed: no measuring request, no class
    assert p.measure_class == "" and f.counters["measure_reclass"] == 0 and not p.d_eligible


def test_n1_switch_off_changes_nothing(monkeypatch):
    from flliper.srt.environ import envs
    f = _fresh()
    p = _returnee(f, "pdflip-0-1", HALF_X - 1, u_p=HALF_X - 1)
    with envs.FLLIPER_PDFLIP_EXACT_CLASS.override(False):
        assert F.Front._measure_return(f, "pdflip-0-1", p, 3000, N) is None
    assert p.measure_class == "kurz"


# ======================================================================================================== N2

def _setup(curves, cap=12288):
    return SimpleNamespace(tp_prefill_max_tokens=cap, _x_setup=SimpleNamespace(curves=curves, beyond=xc.BEYOND_CLAMP))


KINK_PRICE = xc.FlipPrice(depth=(0, 60000, 262144), seconds=(4.0, 3.0, 7.0), attribution="calib_prefix", pairs=9)
KINK_PRICE2 = xc.FlipPrice(depth=(0, 20000, 90000, 262144), seconds=(4.0, 2.0, 6.0, 7.0), attribution="calib_prefix", pairs=9)


def test_n2_chunk_edge_case_of_the_review_4207():
    f = _setup(T._curves())
    assert F.Front._x_min_over_depths(f, 200000) <= 4207      # the review's dense-scan minimum at depth 126976 = 131072-4096
    assert F.Front._x_min_over_depths(f, 200000) != 4213, "the old rows-only candidates gave 4213"


def test_n2_flip_price_support_point_case_of_the_review_3074():
    f = _setup(T._curves(price=KINK_PRICE))
    got = F.Front._x_min_over_depths(f, 200000)
    assert got <= 3074 and got != 3678, "the old rows-only candidates gave 3678"
    window = min(F.Front._x_at_depth(f, d) for d in range(56000, 58600))        # step 1 around the review's minimum
    assert got <= window


def test_n2_kink_table_holds_the_chunk_and_row_kinks_and_the_flip_crossing():
    f = _setup(T._curves(price=KINK_PRICE))
    depths, vals = F.Front._x_kink_table(f)
    assert depths == sorted(set(depths)) and len(depths) == len(vals)
    for want in (131072 - 4096, 131072 - 2 * 4096, 262144 - 4096, 32768 - 4096, 131072, 131071, 131073):
        assert want in depths, want                       # row - k*chunk, reach - j*chunk, rows and both sides
    cross = [d for d in depths if abs(d + F.Front._x_at_depth(f, d) - 60000) <= 1]
    assert cross, "the depth where depth + X(depth) meets the flip price's support point 60000 is a candidate"
    assert F.Front._x_kink_table(f)[0] is depths, "cached per curve set"


@pytest.mark.parametrize("price", [None, KINK_PRICE, KINK_PRICE2])
@pytest.mark.parametrize("n", [5000, 70000, 131072, 200000, 262144])
def test_n2_never_above_any_depth_in_range(price, n):
    f = _setup(T._curves(price=price) if price is not None else T._curves())
    got = F.Front._x_min_over_depths(f, n)
    scan = min(F.Front._x_at_depth(f, d) for d in list(range(0, n + 1, 97)) + [n])
    assert got <= scan, (got, scan)


def test_n2_estimated_size_spans_the_whole_range_never_a_depth_from_the_estimate():
    f = _setup(T._curves(price=KINK_PRICE))
    full = F.Front._x_min_over_depths(f, None)
    assert full <= F.Front._x_min_over_depths(f, 200000) and full < F.Front._x_min_over_depths(f, 100)
    # through the arrival: a media / chars/3-fallback request (n_src n_est) arms the barrier on the whole range
    g = _fresh()
    g._store_probe_info = {"page_size": PAGE}
    g._x_setup = f._x_setup
    xx = SimpleNamespace(measure=True, n_src="n_est", uncached_p=-1, class_basis=None, unpriced=True)
    F.Front._dec_arrival(g, "pdflip-0-1", xx, remainder=0, x_set=X, route="d", p_only=False,
                         t_arrive_m=time.monotonic(), n_prompt=100)    # n_est = 100: a tiny estimate
    assert full < X and g._measure_on_d["pdflip-0-1"][2] == full < F.Front._x_min_over_depths(g, 100)


def test_n2_text_ids_are_an_exact_size_and_bound_the_range():
    g = _fresh()
    g._store_probe_info = {"page_size": PAGE}
    calls = []
    orig = F.Front._x_min_over_depths
    g._x_setup = SimpleNamespace(curves=object(), beyond="clamp")
    F.Front._x_min_over_depths = lambda self, n: calls.append(n) or X
    try:
        for src, want in (("front_ids", 777), ("n_est", None)):
            xx = SimpleNamespace(measure=True, n_src=src, uncached_p=-1, class_basis=None, unpriced=True)
            F.Front._dec_arrival(g, "pdflip-0-%s" % src, xx, remainder=0, x_set=X, route="d", p_only=False,
                                 t_arrive_m=time.monotonic(), n_prompt=777)
    finally:
        F.Front._x_min_over_depths = orig
    assert calls == [777, None]


def test_n2_measure_return_uses_the_prompt_length_not_the_requeue_counter(monkeypatch):
    f = _fresh()
    seen = []
    monkeypatch.setattr(F.Front, "_x_at_depth", lambda self, depth, k=1: seen.append(depth) or X)
    p = _returnee(f, "pdflip-0-1", 3000, u_p=3000)
    assert p.est_prompt == N and (N - 3000) in seen, seen          # depth = prompt - u (not counter - u)
    # an entry without a prompt length: the X in force, never a depth made of the counter
    seen.clear()
    _arm(f, "pdflip-0-2", 3000)
    q = _pend("pdflip-0-2")
    q.est_prompt = 0
    f._x_requeues["pdflip-0-2"] = 1          # n = 2: not terminal
    _refuse(f, "pdflip-0-2", 3000, p=q)
    assert 0 not in seen and q.measure_class == "halb", seen      # the old fallback made depth max(0, 2 - 3000) = 0


# ======================================================================================================== N3

def test_n3_p_upper_leg_has_its_own_action_and_counter(caplog):
    f = _fresh()
    f._store_probe_info = {"page_size": PAGE}
    xx = F.Front._x_unpriced(f, "pdflip-0-1", SimpleNamespace(n=N), 777, 202, False)
    assert xx.class_basis.kind == dn.KIND_P_UPPER
    F.Front._dec_arrival(f, "pdflip-0-1", xx, remainder=202, x_set=X, route="short", p_only=False,
                         t_arrive_m=time.monotonic(), n_prompt=N)
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        hdr = F.Front._front_x_headers(f, "pdflip-0-1", None, False)
    assert hdr[front_x.HEADER_ACTION] == "D_NOW_BARRIER"
    assert f.counters["d_now_barrier_headers"] == 1 and f.counters["measure_on_d_headers"] == 0
    assert any("action=D_NOW_BARRIER" in m for m in caplog.messages)
    assert "pdflip-0-1" in f._measure_sent, "the report still comes back through _measure_return"


def test_n3_a_measuring_request_keeps_measure_on_d():
    f = _fresh()
    hdr = _arm(f, "pdflip-0-1", -1)
    assert hdr[front_x.HEADER_ACTION] == "MEASURE_ON_D"
    assert f.counters["measure_on_d_headers"] == 1 and f.counters["d_now_barrier_headers"] == 0
