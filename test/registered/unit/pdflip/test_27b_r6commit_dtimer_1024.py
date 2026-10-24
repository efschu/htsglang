"""#200 on the 27B line (fLLiper move strand A, Unify-RISK (a); SOLL v2 R3, R6, R6k, R8, R11 -- /spinning/gpu-arb/SCHEDULER-SOLL.md).

The 27B front has no ``_r6_halt_hold`` (NF-RC31-R6REST is NF only, checked on 5b20670b47), but the same defect class in its sibling
``_r6_commit_hold`` (NF-RC30-R6STALE review A1, ported): a D verdict (``D_AFTER_TIMER``, S < X) made while a counted D prefill's
next chunk is COMMITTED was held, also when the most that chunk can add (D's rest; free <= rest) cannot lift S to X -- then R6
("wächst S (inkl. des laufenden Rests) über X") cannot apply and the hold violates R3 ("Läuft T_halb ab mit S < X: das Set wird
auf D prefillt") / R8. Fix: ``_r6_hold_d_free`` (byte-identical to the NF line). Live numbers 48-687: S=4584, rest 784, X=6114.
"""
from __future__ import annotations

import collections
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.environ import envs  # noqa: E402
from flliper.srt.pdflip import front as F  # noqa: E402
from flliper.srt.pdflip import phase_policy as PP  # noqa: E402
from test_nf_r6_rest_exakt_1020 import _front_live  # noqa: E402
from test_pdflip_set_scheduler_1017 import _on  # noqa: E402

RUN48, SET48 = "pdflip-48-687", "pdflip-48-689"
X48, HX48, U48 = 6114, 3057, 4584


def _verdicts(caplog):
    return [m for m in caplog.messages if "PDFLIP SET-VERDICT" in m]


def _commit_front(monkeypatch, tmp_path, *, rest):
    """Live 48-687 shape: the running halb D prefill's chunk ended with ``rest`` left, D's recv step COMMITTED the next chunk
    (free unknown); the halb member 48-689 (4584) waited 6.8 s > T_halb."""
    _on(monkeypatch)
    now = time.time()
    f = _front_live(now, tmp_path, member_u=U48, ended=True, rest_start=rest + 1000, ext_len=1000, x=X48, klass="halb",
                    waited=6.8, rid_run=RUN48, half_x=HX48)
    ws, _ev = f._test_running[RUN48]
    for w in ws:
        w.commit_next(rids=[RUN48])
    return now, f, ws


def test_r3_commit_pending_rest_784_cannot_lift_s_to_x_d_now(monkeypatch, caplog, tmp_path):
    """REPRO commit side (red before #200: COLLECT "R6 commit pending"). Live 48-687 numbers: S = 4584, COMMITTED rest 784 ->
    at most 5368 < X 6114: the D verdict (T_halb over) is final at once (R3); nothing is redirected (R7)."""
    now, f, _ws = _commit_front(monkeypatch, tmp_path, rest=784)
    with envs.FLLIPER_PDFLIP_DECIDER.override(True), caplog.at_level(logging.INFO, logger="pdflip.front"):
        assert f._dc_carried_running() == 0
        assert f._decode_collect(now) == "D"
    sv = _verdicts(caplog)
    assert any("verdict=D_AFTER_TIMER" in m and "S=%d" % U48 in m for m in sv), sv
    assert not any("R6 commit pending" in m for m in sv), sv
    assert f.counters["r6_commit_pending"] == 1 and f.counters["set_r6_commit_hold"] == 0
    assert f.counters["set_r6_hold_d_free"] == 1 and f.counters["r6_redirected"] == 0
    assert any("R6-HOLD-D-FREE site=set S=4584 commit_bound=784 halt_bound=0 X=6114" in m for m in caplog.messages)


def test_r6_commit_pending_rest_that_can_lift_s_to_x_holds(monkeypatch, caplog, tmp_path):
    """R6: COMMITTED rest 3000 -> S could reach 7584 >= X 6114: the D verdict waits for the chunk's start record (unchanged)."""
    now, f, _ws = _commit_front(monkeypatch, tmp_path, rest=3000)
    with envs.FLLIPER_PDFLIP_DECIDER.override(True), caplog.at_level(logging.INFO, logger="pdflip.front"):
        assert f._dc_carried_running() == 0
        assert f._decode_collect(now) == "hold"
    sv = _verdicts(caplog)
    assert sv and all("verdict=COLLECT" in m and "R6 commit pending" in m and "was D_AFTER_TIMER" in m for m in sv), sv
    assert f.counters["set_r6_commit_hold"] >= 1 and f.counters["set_r6_hold_d_free"] == 0


def _bare(**dct):
    o = type("O", (), {})()
    o.counters = collections.Counter()
    o.epoch = 3
    o.__dict__.update(dct)
    return o


def test_gate_changes_d_verdicts_only_and_needs_every_bound():
    """Flip verdicts are never touched (R1a/R6 carry rules); no X, a legacy set of rids (no rest known) or a stop without its
    counted free keep the hold; an empty pending state is no gate."""
    sv_d = PP.SetVerdict(PP.V_D_AFTER_TIMER, "R3 T_halb over", 0.0, 1000, 3000)
    for v in (PP.V_FLIP_P, PP.V_REROUTE_P, PP.V_PARK_NO_SEAT, PP.V_COLLECT):
        o = _bare(_r6_commit_now={"a": 1})
        assert not F.Front._r6_hold_d_free(o, PP.SetVerdict(v, "w", 0.0, 1000, 3000), 6000, "t")
    assert F.Front._r6_hold_d_free(_bare(_r6_commit_now={"a": 1}), sv_d, 6000, "t")
    assert not F.Front._r6_hold_d_free(_bare(_r6_commit_now={"a": 1}), sv_d, 0, "t")
    assert not F.Front._r6_hold_d_free(_bare(_r6_commit_now={"a"}), sv_d, 6000, "t")
    assert not F.Front._r6_hold_d_free(_bare(), sv_d, 6000, "t")
    o = _bare(_r6_halt_pending={"h"}, _r6_halt={"h": {"counted": 10}})       # review A1: no bound (only counted) -> hold, named
    assert not F.Front._r6_hold_d_free(o, sv_d, 6000, "t") and o.counters["r6_hold_d_free_nobound"] == 1
    o = _bare(_r6_commit_now={"a": 2000}, _r6_halt_pending={"h"}, _r6_halt={"h": {"bound": 3000, "counted": 10}})
    assert not F.Front._r6_hold_d_free(o, sv_d, 6000, "t")                 # 1000 + 2000 + 3000 = X: S may reach X (R6/R3)
    o = _bare(_r6_commit_now={"a": 2000}, _r6_halt_pending={"h"}, _r6_halt={"h": {"bound": 2999, "counted": 10}})
    assert F.Front._r6_hold_d_free(o, sv_d, 6000, "t") and o.counters["set_r6_hold_d_free"] == 1


def test_review_a2_s49_term_in_the_gate():
    """Review A2 (S49 X-SUM, phase_policy ``_d_verdict``): released only when S + bounds + gu <= X while D decodes; R8 (no decode)
    and gu = 0 ignore the term. 48-687: 4584 + 784 = 5368, gu 1000 -> 6368 > 6114 held; gu 746 -> 6114 <= X released."""
    sv_d = PP.SetVerdict(PP.V_D_AFTER_TIMER, "R3 T_halb over", 0.0, U48, 3057)
    assert not F.Front._r6_hold_d_free(_bare(_r6_commit_now={"c": 784}), sv_d, X48, "t", gu=1000, decoding=True)
    o = _bare(_r6_commit_now={"c": 784})
    assert not F.Front._r6_hold_d_free(o, sv_d, X48, "t", gu=1000) and o.counters["r6_hold_d_free_xsum"] == 1
    assert F.Front._r6_hold_d_free(_bare(_r6_commit_now={"c": 784}), sv_d, X48, "t", gu=746, decoding=True)
    assert F.Front._r6_hold_d_free(_bare(_r6_commit_now={"c": 784}), sv_d, X48, "t", gu=1000, decoding=False)   # R8
    assert F.Front._r6_hold_d_free(_bare(_r6_commit_now={"c": 784}), sv_d, X48, "t", gu=0, decoding=True)


def test_review_s1_r2_kurz_only_d_verdict_is_released_unconditionally():
    """Review S1 (SOLL R2): "R2 kurz only: no set member" -- S stays 0 whatever D records, R6 can never apply: released even
    when the bounds alone reach X."""
    sv_r2 = PP.SetVerdict(PP.V_D_AFTER_TIMER, "R2 kurz only: no set member, D now (S=0)", 0.0, 0, 3057)
    o = _bare(_r6_commit_now={"c": 9000})
    assert not F.Front._r6_hold_d_free(o, sv_r2, X48, "t")
    o = _bare(_r6_commit_now={"c": 9000})
    assert F.Front._r6_hold_d_free(o, sv_r2, X48, "t", r2_only=True) and o.counters["set_r6_hold_d_free_r2"] == 1


def test_review_a2_live_48_687_plus_1000_granted_unstarted_holds_then_s49(monkeypatch, caplog, tmp_path):
    """Review A2 REPRO (red on the first #200 head: "D", D would get 4584 + 784 + 1000 = 6368 > X 6114). A halb D prefill of 1000
    granted and NOT begun (S49 term); 48-687's next chunk COMMITTED (rest 784). The gate must hold; once the chunk's start record
    gives free 784 the verdict made again is S49's COLLECT (5368 + 1000 > X), never D over X, no flip (R7)."""
    now, f, ws = _commit_front(monkeypatch, tmp_path, rest=784)
    f._d_pf_book().grant(rid="pdflip-48-690", tokens=1000, now=now - 0.5)
    f.groups["D"].outstanding["pdflip-48-690"] = now - 0.5
    F.Front._r6k_note_class(f, "pdflip-48-690", "halb", "arrival", HX48)
    with envs.FLLIPER_PDFLIP_DECIDER.override(True), caplog.at_level(logging.INFO, logger="pdflip.front"):
        assert f._decode_collect(now) == "hold"
        assert f.counters["r6_hold_d_free_xsum"] >= 1 and f.counters["set_r6_hold_d_free"] == 0
        assert f.counters["set_d_after_timer"] == 0
        for w in ws:                                                    # D's built batch did not take it: free 784 known
            w.resolve_commit(launched=[])
        assert f._decode_collect(now + 0.05) == "hold"
    # the second COLLECT has the same (set, members) key: its SET-VERDICT line is deduplicated -- S49's own counter names it
    assert f.counters["set_xsum_hold_unstarted"] >= 1, dict(f.counters)
    assert f.counters["set_d_after_timer"] == 0 and f.counters["set_flip_p"] == 0 and f.counters["set_reroute_p"] == 0
