"""H16V S8 (17.10. evening): the review's conditions A2-A5 on desk/nf-h16v-int-1017 (77d9fe7b08), each pinned at the
call edge:

A2 (SOLL R1/R3/R2a)  S is decided against the members' OWN X (the smallest); X(k) = X + (k-1) x P fixed cost is a
                     report (``xk=`` in the SET-VERDICT why), never an input of a verdict;
A3 (R2/R7/R11)       the FRONT carries X and halbX on EVERY hand-over to D (also the one after P); D's resumed-tail
                     extend reads only that: nothing carried -> RESUME-VIA-P (no fixed 6144), never over the carried halbX;
A4 (R5)              the seat count is read again before the AGE-PLAN to_d grant, and one younger decode is planned
                     for ONE waiter: three waiters, one seat, one younger decode -> at most one to_d grant;
A5 (R1)              the deciding S is the D-side sum of the members' own figures (D's in-batch prefix dedupe is not
                     guaranteed), the deduplicated figure is P's (``S_P``). The cases of A5/A2 at the front's edge are
                     in test_pdflip_h16v_s5_1017.py (updated); here: the pure function and the call edge.
Mutants: mutants_h16v_s8_1017.sh."""
from __future__ import annotations

import asyncio
import inspect
import logging
import os
import re
import time
import types

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from test_pdflip_set_scheduler_1017 import X, _fx, _on, _verdicts  # noqa: E402
from test_pdflip_arrival_seat_rule_0929 import _front  # noqa: E402
from test_pdflip_front_x_carry_1015 import _tokenized, _scheduler_req, _published_server_args  # noqa: E402,F401
from test_pdflip_h16v_s5_1017 import _curves  # noqa: E402

from flliper.srt.pdflip import front as F  # noqa: E402
from flliper.srt.pdflip import phase_policy as pp  # noqa: E402

M = pp.SetMember


# ------------------------------------------------------------------------------------------------ A2

def test_a2_the_decision_x_is_the_smallest_own_x_not_x_of_k():
    f = _fx()
    f.tp_prefill_max_tokens = 20000
    f._x_setup = types.SimpleNamespace(curves=_curves(900.0), beyond="clamp")
    rd = [("a", 1000, 5222), ("b", 1000, 5300), ("c", 1000, 5222)]
    assert F.Front._set_x_for(f, rd) == 5222
    assert F.Front._set_xk_for(f, rd) > 5222 + 900        # info: three members, two more P forwards
    assert F.Front._set_xk_for(f, rd[:1]) == 0            # k = 1: nothing to report


def test_a2_x_of_k_never_moves_a_verdict_and_is_named_in_the_why(monkeypatch, caplog):
    """Two 2700 members at curve X=5222 (S=5400): flip; the set-X with a second P forward (6222) is reported only."""
    _on(monkeypatch)
    f = _fx()
    f.tp_prefill_max_tokens = 12000
    f._x_setup = types.SimpleNamespace(curves=_curves(900.0), beyond="clamp")
    x1 = f._x_set_of("a", 1000)
    f._x_set_of("b", 1000)
    t0 = time.time()
    st = f._set_st()
    st["shorts"]["a"] = (t0, 2700, x1)
    st["shorts"]["b"] = (t0 + 0.3, 2700, x1)
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        assert f._decode_collect(t0 + 0.4) == "P"
    (line,) = _verdicts(caplog)
    assert re.search(r"S=5400 X=5222 ", line) and re.search(r" xk=62\d\d", line), line


@pytest.mark.usefixtures("decider_off")  # EZ-S2a: describes the S1 code (reachable until S4)


def test_a2_the_queue_sum_path_decides_on_the_own_x_too(monkeypatch, caplog):
    from test_pdflip_arrival_seat_rule_0929 import _pending

    _on(monkeypatch)
    f = _fx(running=())
    f.tp_prefill_max_tokens = 12000
    f._x_setup = types.SimpleNamespace(curves=_curves(900.0), beyond="clamp")
    x1 = f._x_set_of("a", 1000)
    f._x_set_of("b", 1000)
    now = time.time()
    qa, qb = _pending("a", 2700, now - 9.0), _pending("b", 2700, now - 8.0)
    qa.x_set = qb.x_set = x1
    f.queue = [qa, qb]
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        assert f._set_flip_justified(now) is True         # S = 5400 >= X = 5222; X(k=2) = 6222 would have held it
    assert f.counters["flip_held_under_x"] == 0
    assert any(re.search(r"S=5400 X=5222 ", m) and re.search(r" xk=62\d\d", m) for m in _verdicts(caplog)), _verdicts(caplog)


# ------------------------------------------------------------------------------------------------ A3

def test_a3_a_leg_after_p_carries_x_and_halfx_at_the_depth_d_continues_from():
    f = types.SimpleNamespace(counters=__import__("collections").Counter(), tp_prefill_max_tokens=6733,
                              _front_price={})
    f._x_setup = types.SimpleNamespace(curves=_curves(900.0), beyond="clamp")
    pend = types.SimpleNamespace(rid="pdflip-18-250", est_uncached=0, est_prompt=75149, d_direct=False)
    h = F.Front._front_x_headers(f, "pdflip-18-250", pend, False)
    x_curve = F.Front._x_at_depth(f, 75149)
    assert int(h["x-pdflip-front-x"]) == x_curve != 6733, "the X of the curve at the prompt depth, not the X in force"
    assert int(h["x-pdflip-front-halfx"]) == x_curve // 2, h


def test_a3_halfx_is_computed_with_the_fronts_flag_not_a_rank_env(monkeypatch):
    f = types.SimpleNamespace(counters=__import__("collections").Counter(), tp_prefill_max_tokens=6000, _front_price={})
    monkeypatch.setenv("FLLIPER_PDFLIP_HALF_X_FRAC", "0.25")
    pend = types.SimpleNamespace(rid="r", est_uncached=0, est_prompt=100, d_direct=True)
    F.Front._note_x_decision(f, "r", 6000, 10)
    assert F.Front._front_x_headers(f, "r", pend, False)["x-pdflip-front-halfx"] == "1500"


def test_a3_halfx_survives_every_hop_to_the_req(_published_server_args):
    from flliper.srt.managers.io_struct import GenerateReqInput, TokenizedGenerateReqInput
    from flliper.srt.managers.schedule_batch import Req

    for cls in (GenerateReqInput, TokenizedGenerateReqInput):
        fields = set(getattr(cls, "__dataclass_fields__", None) or getattr(cls, "__struct_fields__", ()) or ())
        assert "weg2_front_halfx" in fields, cls.__name__
    assert "self.weg2_front_halfx = None" in inspect.getsource(Req.__init__)
    t = _tokenized(1010, 237, halfx=505)
    assert (t.weg2_front_x, t.weg2_front_halfx) == (1010, 505)
    assert _tokenized(None, None).weg2_front_halfx is None
    req = _scheduler_req(t)
    assert (req.weg2_front_x, req.weg2_front_halfx) == (1010, 505)


def test_a3_the_halfx_header_is_stamped_and_garbage_is_absent():
    from flliper.srt.pdflip import front_x

    obj = types.SimpleNamespace(weg2_front_x=None, weg2_front_priced=None, weg2_front_halfx=None)
    front_x.stamp_request(obj, types.SimpleNamespace(headers={"x-pdflip-front-x": "6000", "x-pdflip-front-halfx": "3000"}))
    assert (obj.weg2_front_x, obj.weg2_front_halfx) == (6000, 3000)
    bad = types.SimpleNamespace(weg2_front_x=None, weg2_front_priced=None, weg2_front_halfx=None)
    front_x.stamp_request(bad, types.SimpleNamespace(headers={"x-pdflip-front-x": "6000", "x-pdflip-front-halfx": "zz"}))
    assert (bad.weg2_front_x, bad.weg2_front_halfx) == (6000, None)


def test_a3_no_carried_x_d_never_extends_even_at_the_old_fixed_bound():
    """The mutant target: the former substitute bound was 0.5 x the static X (6144) -- a tail of 2000..6143 must NOT
    be D's work without a carried X."""
    from flliper.srt.pdflip import d_norecompute as dn
    from test_pdflip_h16v_s5_1017 import _resumed, _sched

    for rem in (2000, 3500, 6143):
        assert dn.d_direct_extends(_sched(12288), _resumed(rem, front_x=None, halfx=None), rem) is False


def test_a3_a_carried_halfx_is_never_exceeded_in_the_released_tail():
    from flliper.srt.pdflip import d_norecompute as dn
    from test_pdflip_h16v_s5_1017 import _resumed, _sched

    for rem in range(1, 7000, 53):
        for halfx in (900, 2750, 3300):
            assert dn.d_direct_extends(_sched(12288), _resumed(rem, 2 * halfx, halfx), rem) is (rem < halfx)


# ------------------------------------------------------------------------------------------------ A4

def test_a4_three_waiters_one_seat_one_younger_decode_make_at_most_one_to_d_grant(monkeypatch):
    """AGE PLAN: a KV deficit, one free seat, ONE younger decode to displace, three older waiters; the seat count is read
    again (no await between it and the grant) and the one decode is spoken for by the first grant."""
    monkeypatch.setenv("FLLIPER_PDFLIP_ENABLE_ARRIVAL_SEAT_RULE", "1")
    monkeypatch.setenv("FLLIPER_PDFLIP_PBOUND_FLIP_NOW", "0")
    monkeypatch.delenv("FLLIPER_PDFLIP_ENABLE_ARRIVAL_SEAT_AGE_PLAN", raising=False)
    f = _front(running=["young"], n=2, kv={"available": 100, "evictable": 0, "decode_clip": 4096})
    f.tp_prefill_max_tokens = X
    now = time.time()
    f.t_awake = now - 60.0
    f.__dict__.setdefault("_asr_arrive", {})["young"] = now + 5.0
    F.Seat(f, "young", "short", tokens=90000)

    async def slow_kv():
        await asyncio.sleep(0.2)                     # the await between "free" and the grant
        return {"available": 100, "evictable": 0, "decode_clip": 4096}

    f._arrival_seat_kv_reading = slow_kv

    async def run():
        ws = ("w1", "w2", "w3")
        for i, r in enumerate(ws):
            f._asr_arrive[r] = now + i * 0.001
        ts = [asyncio.ensure_future(f._arrival_seat_wait(r, 2000, None, uncached=2000, t_arrive=now + i * 0.001))
              for i, r in enumerate(ws)]
        await asyncio.sleep(1.5)
        got = [r for r, t in zip(ws, ts) if t.done() and t.result() is True]
        for t in ts:
            t.cancel()
        return got

    assert asyncio.run(run()) == ["w1"]
    assert f.counters["arrival_seat_age_to_d"] == 1


def test_a4_the_age_plan_reads_the_seat_count_again_before_the_plan():
    src = inspect.getsource(F.Front._arrival_seat_wait_age)
    # S60 F3: the waiter's seat count is ``_arrival_seat_taken(for_rid=rid)`` (only OLDER D-line entries hold a seat)
    i, j = src.index("self._arrival_seat_taken("), src.index("self._arrival_seat_age_verdict(")
    assert i < j and "age_victims" in src


# ------------------------------------------------------------------------------------------------ A5

def test_a5_set_verdict_decides_on_the_union_figure_r1u():
    a, b = M("a", 4000, 0.0), M("b", 2100, 0.5, own_uncached=4100)       # b's 2000 shared tokens P counts once
    v = pp.set_verdict([a, b], x_tokens=6185, half_frac=0.5, t_half_s=4.5, now=0.6, t_open=0.0, d_decoding=True,
                       seats_free=4)
    # SOLL R1u (user decision 20.10., supersedes the A5 own-figure sum): S counts the shared piece ONCE
    assert v.s_tokens == 6100 and v.verdict == pp.V_COLLECT


def test_a5_the_docstring_states_that_the_discount_is_p_side_only():
    d = " ".join(inspect.getdoc(pp.shared_prefix_discount).split())
    assert "NOT assumed for D" in d and "P prefills" in d and "safe side" in d
