"""DECODE-COLLECT "D busy" law, pinned with the real numbers of 15.10. 13:12:55-13:13:02Z (task #119).

User rule: "wenn auf D kein Decode ansteht, wird sofort geprefillt -- sonst gesammelt". The monitor alarm of 13:13:02Z (release
n=1 tokens=5302 X=6057 waited_s=7.5 route=D stage=dcheck) claimed D idle; the logs prove otherwise: D TP0 decoded bs=5-6 and
prefilled weg2-40-128 (5089 uncached, booked on D by the front: ``carried``) -- the window held for a reason, and its sum
(213 + 5089 = 5302 <= X=6057) went to D at d_check. No front code was wrong, so these are PIN tests (green on the base 226aad7105);
each is killed by a named mutant (see the report). Idle D (nothing decoding, nothing booked, or every decode parked) = no window.
"""
from __future__ import annotations

import logging
import os
import time
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from test_weg2_arrival_seat_rule_0929 import _front, _pending  # noqa: E402

from sglang.srt.managers.scheduler_components import weight_updater as _wu  # noqa: E402,F401
from sglang.srt.weg2 import phase_policy  # noqa: E402

X_LIVE = 6057  # the X of the 13:12:55 release


def _on(monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_ENABLE_ARRIVAL_SEAT_RULE", "1")
    monkeypatch.delenv("SGLANG_WEG2_DECODE_COLLECT_D_CHECK_S", raising=False)  # default 7.5
    monkeypatch.delenv("SGLANG_WEG2_ENABLE_ARRIVAL_SEAT_AGE_PLAN", raising=False)
    monkeypatch.delenv("SGLANG_WEG2_PBOUND_FLIP_NOW", raising=False)
    monkeypatch.setenv("SGLANG_WEG2_DECODE_COLLECT_WINDOW_S", "15")
    monkeypatch.delenv("SGLANG_WEG2_ENABLE_DECODE_COLLECT_PREFILL_BUSY", raising=False)


def _front_1313(monkeypatch):
    """D at 13:12:55.05: five decodes running (bs=5, TP0 Decode rank batch 9055) and weg2-40-128 handed to D (5089 uncached)."""
    _on(monkeypatch)
    f = _front(running=["r1", "r2", "r3", "r4", "r5"], n=6, kv={"available": 400000, "evictable": 0})
    f.tp_prefill_max_tokens = X_LIVE
    handed = types.SimpleNamespace(rid="weg2-40-128", est_uncached=5089, d_direct=True, fut=None)
    f._ready_for_d = [handed]
    return f


def _release_lines(caplog):
    return [r.getMessage() for r in caplog.records if "DECODE-COLLECT release" in r.getMessage()]


def test_the_real_1313_set_holds_then_goes_to_d_at_d_check(monkeypatch, caplog):
    f = _front_1313(monkeypatch)
    t0 = time.time()
    f.t_awake = t0 - 60.0
    f.queue = [_pending("weg2-42-129", 213, t0)]
    with caplog.at_level(logging.INFO, logger="weg2.front"):
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
    with caplog.at_level(logging.INFO, logger="weg2.front"):
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
    f.queue = [_pending("weg2-42-129", 213, t0)]
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
    f.queue = [_pending("weg2-42-129", 213, t0)]
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
    f.queue = [_pending("weg2-42-129", 213, t0)]
    with caplog.at_level(logging.INFO, logger="weg2.front"):
        assert f._decode_collect(t0 + 0.1) == "hold"
        f.groups["D"].outstanding.clear()  # r1 finished
        assert f._decode_collect(t0 + 2.0) == "D"
    (line,) = _release_lines(caplog)
    assert "stage=d-idle" in line and "waited_s=2.0" in line, line
