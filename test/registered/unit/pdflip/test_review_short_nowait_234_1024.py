"""Review probes #234 SHORT-NOWAIT-1024 (reviewer, not a fix). Each probe states the SOLL expectation; a red probe is a finding.

P1/P2 (SOLL R6k + R2, the author's "Offen 1"): a SHORT released to D by a D VERDICT while kurz on its own (re-probed) figure must
carry the same R6k class "kurz" (and D_NOW number) as one released by the #234 COLLECT path -- one situation, one class.
P3 (SOLL R1a, R4OVER-1024, check item 1 "R1a-Mitfahrt bei 27B-R2-Sperre"): a COLLECT that is the R2 hold of a decided R3 flip
whose members are SHORTs only must not release the kurz SHORT to D (it rides the flip), as with a queue member (P3c control).
"""
from __future__ import annotations

import logging
import os
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from test_pdflip_set_scheduler_1017 import _fx, _on, _verdicts  # noqa: E402
from test_pdflip_arrival_seat_rule_0929 import _pending  # noqa: E402
from test_review_sibling_u_1024 import A10, B10, _anchor_done, _pair_s10, _reprobe_stub  # noqa: E402

from flliper.srt.pdflip import front as front_mod  # noqa: E402
from flliper.srt.pdflip import phase_policy as pp  # noqa: E402

X6 = 6185
K = "pdflip-rv-k"


def test_p1_r6k_lone_reprobed_kurz_short_released_by_the_r2_kurz_only_verdict_is_kurz(monkeypatch, caplog):
    t0 = time.time()
    _on(monkeypatch)
    f = _fx()
    f.tp_prefill_max_tokens = X6
    front_mod.Front._r6k_note_class(f, K, pp.CLASS_HALB, "arrival", 3092)   # it ARRIVED halb (3500)
    st = f._set_st()
    st["shorts"][K] = (t0, 1500, X6)                                            # the S63 re-probe moved it to 1500 (kurz)
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        assert f._decode_collect(t0 + 0.3) == "D"
    assert "R2 kurz only" in _verdicts(caplog)[-1], _verdicts(caplog)
    assert st["released"].get(K) == "D"
    assert f.__dict__["_r6k_class"][K][0] == pp.CLASS_KURZ, f.__dict__["_r6k_class"][K]   # R6k: went to D as kurz (R2)
    assert K in (st.get("kurz_now") or {}), st.get("kurz_now")                             # its D_NOW on the kurz number


def test_p2_r6k_rider_release_rest_then_r2_kurz_only_is_kurz(monkeypatch, caplog):
    t0 = time.time()
    f, st = _pair_s10(monkeypatch, t0)
    front_mod.Front._r6k_note_class(f, B10, pp.CLASS_HALB, "arrival", 3092)  # B10 arrived halb on its own 4000
    _reprobe_stub(monkeypatch, apply=lambda r: st["shorts"].__setitem__(r, (t0 + 0.2, 2500, X6)))
    assert f._decode_collect(t0 + 0.3) == "D" and B10 in st["rider_hold"]
    _anchor_done(f, st, A10)
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        assert f._decode_collect(t0 + 0.8) == "D"
    assert st["released"].get(B10) == "D"
    assert f.__dict__["_r6k_class"][B10][0] == pp.CLASS_KURZ, f.__dict__["_r6k_class"][B10]


def _r2_hold_scene(monkeypatch, now):
    _on(monkeypatch)
    f = _fx(running=("dec", "k1"))
    f.tp_prefill_max_tokens = X6
    f._d_pf_book().grant(rid="k1", tokens=181, now=now - 0.2)   # a kurz D grant opened BEFORE the flip verdict (R2 hold)
    f._d_pf_book().enter_leg2(rid="k1")
    f.groups["D"].outstanding["k1"] = now - 0.2
    f._r6k_mark_kurz("k1")
    f._d_ext_reading = lambda: {}
    _reprobe_stub(monkeypatch)
    return f, f._set_st()


def test_p3_r1a_shorts_only_r3_flip_held_by_r2_is_decided_the_kurz_short_rides(monkeypatch, caplog):
    now = time.time()
    f, st = _r2_hold_scene(monkeypatch, now)
    st["shorts"]["pdflip-rv-h1"] = (now - 1.0, 3500, X6)
    st["shorts"]["pdflip-rv-h2"] = (now - 0.9, 3200, X6)        # S = 6700 >= X: R3 flip (SOLL section 5, example 3)
    st["shorts"][K] = (now - 0.8, 1500, X6)                     # kurz on its own (re-probed) figure
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        assert f._decode_collect(now) == "hold"
    assert f.counters["r2_kurz_grant_hold"] == 1, dict(f.counters)            # the R3 FLIP_P stands, held by R2
    assert f._r2_flip_held(), "R4OVER: a held FLIP_P of SHORT members is decided"
    assert K not in st["released"] and K in st["shorts"], st["released"]       # R1a: it rides, not released to D


def test_p3c_control_queue_member_r4_flip_held_by_r2_the_kurz_short_rides(monkeypatch, caplog):
    now = time.time()
    f, st = _r2_hold_scene(monkeypatch, now)
    f.queue = [_pending("pdflip-rv-g", 9000, now - 4.8, est_prompt=50000)]   # big, T_halb over: R4 flip
    st["shorts"][K] = (now - 0.8, 1500, X6)
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        assert f._decode_collect(now) == "hold"
    assert f.counters["r2_kurz_grant_hold"] == 1, dict(f.counters)
    assert f._r2_flip_held()
    assert K not in st["released"] and K in st["shorts"], st["released"]
