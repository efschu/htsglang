"""REVIEW #235 KALIB-LIVE-X probes (danger direction 2: the switch while a set has collected).

Green = the asserted behaviour is what the reviewed head does. P1 asserts the TRANSITIONAL danger (SOLL R7 direction): a halb
member priced at the old live X keeps that X across the reload, so the timer verdict sends it to D although its u is above the
newly loaded curve X; P2 asserts that the D leg then carries the OLD decision X (D's W50 limit admits it, no W31/W50 bounce);
P3 asserts the bound: nothing above D's static riegel (the old X is <= x_ceiling_tokens)."""
from __future__ import annotations

import logging
import os
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import sys  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_pdflip_set_scheduler_1017 import _fx, _on, _verdicts  # noqa: E402
from test_pdflip_kalib_live_x_235 import CEILING, COMMIT, FORM_ENV_VALUE, TAG, _curves, _write  # noqa: E402

from flliper.srt.pdflip import front_x as _fxm  # noqa: E402
from flliper.srt.pdflip import x_curves as xc  # noqa: E402
from flliper.srt.pdflip.front import Front  # noqa: E402


def _live_front_with_open_set(monkeypatch, tmp_path, t0):
    _on(monkeypatch)
    monkeypatch.setenv("FLLIPER_PDFLIP_FORM", FORM_ENV_VALUE)
    f = _fx()
    f.tag, f.commit, f.x_ceiling_tokens = TAG, COMMIT, CEILING
    f.tp_prefill_max_tokens = CEILING                 # the live re-solve stood at the ceiling (metal: X=12288)
    curves = _curves()
    env = xc.x_envelope(curves, cap=CEILING)
    u = min(CEILING - 1, env + 500)
    assert CEILING // 2 <= u < CEILING and u > env    # halb under the old X, above the new envelope
    st = f._set_st()
    st["shorts"]["h"] = (t0, u, CEILING)              # priced at arrival on the old X (tup[2])
    Front._note_x_decision(f, "h", CEILING, u)        # the arrival's FRONT-X record (old X)
    res = Front.reload_x_curves(f, _write(tmp_path, curves), who="review probe")
    assert res["switched"] and f.tp_prefill_max_tokens == env
    return f, st, u, env


def test_p1_old_member_keeps_its_arrival_x_and_goes_to_d_above_the_new_x(monkeypatch, tmp_path, caplog):
    t0 = time.time() - 5.0                            # T_halb 4.5 s has run out, D decodes
    f, st, u, env = _live_front_with_open_set(monkeypatch, tmp_path, t0)
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        route = f._decode_collect(time.time())
    (line,) = _verdicts(caplog)
    assert route == "D" and "verdict=D_AFTER_TIMER" in line and f"X={CEILING} " in line, line
    assert f"S={u} " in line and u > env


def test_p2_the_d_leg_carries_the_old_decision_x(monkeypatch, tmp_path):
    f, st, u, env = _live_front_with_open_set(monkeypatch, tmp_path, time.time())
    hdr = Front._front_x_headers(f, "h", None, False)
    assert int(hdr[_fxm.HEADER_X]) == CEILING and int(hdr[_fxm.HEADER_PRICED]) == u
    assert _fxm.limit_for(CEILING, CEILING, 64) == CEILING >= u   # D's limit = min(static riegel, front X + page)


def test_p3_a_new_arrival_after_the_reload_is_priced_on_the_curve(monkeypatch, tmp_path):
    f, st, u, env = _live_front_with_open_set(monkeypatch, tmp_path, time.time())
    assert Front._x_at_depth(f, 0) <= env < CEILING
    assert Front._front_x_headers(f, "new", None, False)[_fxm.HEADER_X] == str(env)   # no record: the X in force
