"""DECODE-COLLECT "D busy" law, pinned with the real numbers of 15.10. 13:12:55-13:13:02Z (task #119).

User rule: "wenn auf D kein Decode ansteht, wird sofort geprefillt -- sonst gesammelt". The monitor alarm of 13:13:02Z (release
n=1 tokens=5302 X=6057 waited_s=7.5 route=D stage=dcheck) claimed D idle; the logs prove otherwise: D TP0 decoded bs=5-6 and
prefilled pdflip-40-128 (5089 uncached, booked on D by the front: ``carried``) -- the window held for a reason, and its sum
(213 + 5089 = 5302 <= X=6057) went to D at d_check. No front code was wrong, so these are PIN tests (green on the base 226aad7105);
each is killed by a named mutant (see the report). Idle D (nothing decoding, nothing booked, or every decode parked) = no window.
"""
from __future__ import annotations

import logging
import os
import time
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from test_pdflip_arrival_seat_rule_0929 import _front, _pending  # noqa: E402

from flliper.srt.managers.scheduler_components import weight_updater as _wu  # noqa: E402,F401
from flliper.srt.pdflip import phase_policy  # noqa: E402

X_LIVE = 6057  # the X of the 13:12:55 release


def _on(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_ENABLE_ARRIVAL_SEAT_RULE", "1")
    monkeypatch.delenv("FLLIPER_PDFLIP_DECODE_COLLECT_D_CHECK_S", raising=False)  # default 7.5
    monkeypatch.delenv("FLLIPER_PDFLIP_ENABLE_ARRIVAL_SEAT_AGE_PLAN", raising=False)
    monkeypatch.delenv("FLLIPER_PDFLIP_PBOUND_FLIP_NOW", raising=False)
    monkeypatch.setenv("FLLIPER_PDFLIP_DECODE_COLLECT_WINDOW_S", "15")
    monkeypatch.delenv("FLLIPER_PDFLIP_ENABLE_DECODE_COLLECT_PREFILL_BUSY", raising=False)


def _front_1313(monkeypatch):
    """D at 13:12:55.05: five decodes running (bs=5, TP0 Decode rank batch 9055) and pdflip-40-128 handed to D (5089 uncached)."""
    _on(monkeypatch)
    f = _front(running=["r1", "r2", "r3", "r4", "r5"], n=6, kv={"available": 400000, "evictable": 0})
    f.tp_prefill_max_tokens = X_LIVE
    handed = types.SimpleNamespace(rid="pdflip-40-128", est_uncached=5089, d_direct=True, fut=None)
    f._ready_for_d = [handed]
    return f


def _release_lines(caplog):
    return [r.getMessage() for r in caplog.records if "DECODE-COLLECT release" in r.getMessage()]


def test_the_real_1313_set_holds_then_goes_to_d_at_d_check(monkeypatch, caplog):
    f = _front_1313(monkeypatch)
    t0 = time.time()
    f.t_awake = t0 - 60.0
    f.queue = [_pending("pdflip-42-129", 213, t0)]
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        assert f._decode_collect(t0 + 0.1) == "hold", "D decodes (bs=5) and prefills 5089: the arrival collects"
        assert f._decode_collect(t0 + 7.0) == "hold", "before d_check_s (7.5 s) a set that fits still waits"
        assert f._decode_collect(t0 + 7.5) == "D"
    (line,) = _release_lines(caplog)
    assert "n=1 tokens=5302 X=6057 waited_s=7.5 route=D stage=dcheck p_bound=False" in line, line
    assert f.counters["decode_collect_release_D"] == 1 and f.counters["decode_collect_hold"] == 1


def test_the_booked_prefill_counts_into_the_sum(monkeypatch, caplog):
    """5089 on D + 3000 arriving = 8089 > X=6057: the set does not fit -> collects on (dcheck hold), later P. Without ``carried``
    (mutant) the 3000 alone would pass X and go to D."""
    f = _front_1313(monkeypatch)
    t0 = time.time()
    f.t_awake = t0 - 60.0
    f.queue = [_pending("big", 3000, t0)]
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        assert f._decode_collect(t0 + 0.1) == "hold"
        assert f._decode_collect(t0 + 7.6) == "hold"
        assert f.counters["decode_collect_dcheck_hold"] == 1
        assert f._decode_collect(t0 + 15.1) == "P"
    assert "tokens=8089" in _release_lines(caplog)[-1]


def test_an_idle_d_opens_no_window(monkeypatch):
    """Nothing decodes, nothing is booked on D: the arrival is served at once -- no hold, no 7.5 s."""
    _on(monkeypatch)
    f = _front(running=[], n=6, kv={"available": 400000, "evictable": 0})
    f.tp_prefill_max_tokens = X_LIVE
    t0 = time.time()
    f.t_awake = t0 - 60.0
    f.queue = [_pending("pdflip-42-129", 213, t0)]
    assert not f._dc_d_busy()
    assert f._decode_collect(t0 + 0.1) is None
    assert f.counters["decode_collect_hold"] == 0


def test_a_d_whose_decodes_are_all_parked_is_idle(monkeypatch):
    """14.10. 13:33 (LANE-PREEMPT parked D's decodes): outstanding stays, but nothing runs -> no window (LANE_PARK_STAMP never
    lapses, ``_flip_ledger`` excludes it)."""
    _on(monkeypatch)
    f = _front(running=["r1", "r2", "r3"], n=6, kv={"available": 400000, "evictable": 0})
    f.tp_prefill_max_tokens = X_LIVE
    f._d_parked = {r: phase_policy.LANE_PARK_STAMP for r in ("r1", "r2", "r3")}
    t0 = time.time()
    f.t_awake = t0 - 60.0
    f.queue = [_pending("pdflip-42-129", 213, t0)]
    assert not f._dc_d_busy()
    assert f._decode_collect(t0 + 0.1) is None


def test_d_going_idle_inside_the_window_releases_at_once(monkeypatch, caplog):
    """The window opened on a decoding D; the last decode ends 2 s in -> the next look releases (stage=d-idle), it does not wait
    out the 7.5 s / 15 s."""
    _on(monkeypatch)
    f = _front(running=["r1"], n=6, kv={"available": 400000, "evictable": 0})
    f.tp_prefill_max_tokens = X_LIVE
    t0 = time.time()
    f.t_awake = t0 - 60.0
    f.queue = [_pending("pdflip-42-129", 213, t0)]
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        assert f._decode_collect(t0 + 0.1) == "hold"
        f.groups["D"].outstanding.clear()  # r1 finished
        assert f._decode_collect(t0 + 2.0) == "D"
    (line,) = _release_lines(caplog)
    assert "stage=d-idle" in line and "waited_s=2.0" in line, line


# ---- LANES FIX 1014 (95914428c2): a GRANTED SHORT that D does not run is not "D prefills" ---------------------------------------
# NF 14.10. boot bc35462240, 10:29:36-10:30:15Z: pdflip-54-149 (31 uncached) and pdflip-54-150 (222) were granted as SHORT, then D parked them
# (lane park, floor 1->0 at 10:29:36) before their first content. ``_d_pf_pending_tokens`` counted them -> ``_dc_d_busy()`` True on an
# idle D -> the arrival 54-154 (647) sat the full d_check_s=7.5 s (front :10760 hold 10:29:38,412 -> :10782 release 10:29:45,928
# route=D stage=dcheck, 82 such waits in the boot). Mutant M4 (``idle`` filter ignored in DPrefillInflight.pending_tokens) survived
# 70/70 collect and 389/389 lanes tests (review 15.10.) -- these tests kill it.

def _grant(f, rid, tokens, *, leg2=True):
    book = f._d_pf_book()
    book.grant(rid=rid, tokens=tokens, now=time.time())
    if leg2:
        book.enter_leg2(rid=rid)  # the row then lives while D still holds the rid (D.outstanding)


def test_a_lane_parked_granted_short_is_not_d_prefilling(monkeypatch):
    """54-149/54-150 granted, D holds them (outstanding) but parked them (LANE_PARK_STAMP, out of the flip ledger): D runs nothing."""
    _on(monkeypatch)
    f = _front(running=["pdflip-54-149", "pdflip-54-150"], n=6, kv={"available": 400000, "evictable": 0})
    f.tp_prefill_max_tokens = 6283
    f._d_parked = {r: phase_policy.LANE_PARK_STAMP for r in ("pdflip-54-149", "pdflip-54-150")}
    _grant(f, "pdflip-54-149", 31)
    _grant(f, "pdflip-54-150", 222)
    t0 = time.time()
    f.t_awake = t0 - 60.0
    f.queue = [_pending("pdflip-54-154", 647, t0)]
    assert f._d_pf_pending_tokens() == 0, "a parked granted SHORT is kept in the book but counts for nothing"
    assert not f._dc_d_busy()
    assert f._decode_collect(t0 + 0.1) is None, "idle D: the arrival is served at once, no 7.5 s d_check"
    assert f.counters["decode_collect_hold"] == 0


def test_a_granted_short_below_a_raised_lane_floor_is_not_d_prefilling(monkeypatch):
    """D's lane floor is 1 and the granted SHORT is lane 0: it waits below the floor, nothing prefills it."""
    _on(monkeypatch)
    f = _front(running=[], n=6, kv={"available": 400000, "evictable": 0})
    f.tp_prefill_max_tokens = 6283
    f.__dict__["_lane_state_obj"] = types.SimpleNamespace(lane_floor=1, lane_for=lambda rid: 0)
    _grant(f, "pdflip-54-150", 222, leg2=False)  # granted, not yet in leg 2: alive by its grant TTL
    t0 = time.time()
    f.t_awake = t0 - 60.0
    f.queue = [_pending("pdflip-54-154", 647, t0)]
    assert f._d_pf_pending_tokens() == 0
    assert not f._dc_d_busy()
    assert f._decode_collect(t0 + 0.1) is None


def test_the_same_granted_shorts_do_count_when_d_runs_them(monkeypatch):
    """Control (both directions): running (not parked) or at/above the floor, the granted SHORTs ARE D's prefill -> they price the
    set (sum counts). NODECODE 1015 (user law: collecting only while a DECODE stands on D): they no longer HOLD it -- D runs only
    these two prefills, no decode, so the set is decided at once: 253 + 647 = 900 <= X -> D now (was: "hold"); and the sum is
    what decides -- 253 + 6100 = 6353 > X -> P now, although 6100 alone fits."""
    _on(monkeypatch)
    for arrival, route in ((647, "D"), (6100, "P")):
        f = _front(running=["pdflip-54-149", "pdflip-54-150"], n=6, kv={"available": 400000, "evictable": 0})
        f.tp_prefill_max_tokens = 6283
        f.__dict__["_lane_state_obj"] = types.SimpleNamespace(lane_floor=1, lane_for=lambda rid: 1)
        _grant(f, "pdflip-54-149", 31)
        _grant(f, "pdflip-54-150", 222)
        t0 = time.time()
        f.t_awake = t0 - 60.0
        f.queue = [_pending("pdflip-54-154", arrival, t0)]
        assert f._d_pf_pending_tokens() == 253
        assert f._dc_d_busy() and not f._dc_decoding()
        assert f._decode_collect(t0 + 0.1) == route, arrival
        assert f.counters["decode_collect_hold"] == 0


def test_only_the_parked_short_of_a_mixed_book_drops_out(monkeypatch):
    """54-149 parked, 54-150 running: only 54-150's 222 counts (the filter is per rid, not all-or-nothing)."""
    _on(monkeypatch)
    f = _front(running=["pdflip-54-149", "pdflip-54-150"], n=6, kv={"available": 400000, "evictable": 0})
    f.tp_prefill_max_tokens = 6283
    f._d_parked = {"pdflip-54-149": phase_policy.LANE_PARK_STAMP}
    _grant(f, "pdflip-54-149", 31)
    _grant(f, "pdflip-54-150", 222)
    assert f._d_pf_pending_tokens() == 222
    assert f._d_carried_tokens() == 222
