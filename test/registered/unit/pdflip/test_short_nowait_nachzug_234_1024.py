"""#234 SHORT-NOWAIT-1024 Nachzug Restrisiken (SOLL R1a, R2, R3, R11).

Risk 1 (R1a): the queue-sum site of ``_set_flip_justified`` calls the R2 hold without ``x_class`` and used to overwrite the
frozen ``short_trig`` with [] -- ``_r2_flip_held`` then read "not decided" for a SHORT-only R3 flip until the next
``_decode_collect`` tick and a kurz SHORT / kurz arrival left for D instead of riding the flip.

Risk 2: the #234 helpers' error paths are NAMED (``#234 <NAME> FAILED``) and counted; behaviour unchanged (the two read helpers
answer "no SHORT" as before, ``_set_kurz_shorts_now`` raises as before).

Mutants (each must turn a test here red):
  (1) the R2 hold overwrites ``short_trig`` also without ``x_class`` -> ``test_r1a_queue_sum_site_keeps_the_frozen_short_triggers``;
  (2) a helper's except block without the named line / counter -> the matching ``test_risk2_*``.
"""
from __future__ import annotations

import logging
import os
import time

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from test_review_short_nowait_234_1024 import _r2_hold_scene  # noqa: E402
from test_short_nowait_234_1024 import H, H2, K, X6  # noqa: E402

from flliper.srt.pdflip import front as front_mod  # noqa: E402
from flliper.srt.pdflip import phase_policy as pp  # noqa: E402


def _failed_lines(caplog, name):
    return [r for r in caplog.records if "#234 %s FAILED" % name in r.getMessage()]


def test_r1a_queue_sum_site_keeps_the_frozen_short_triggers(monkeypatch, caplog):
    """Two halb SHORTs (S=6700 >= X) make an R3 flip held by R2 (a kurz D grant k1 is open): the hold freezes them as triggers.
    The queue-sum site then calls the hold with QUEUE members and no x_class: the SHORT triggers stay, the flip stays decided
    (no queue member left in ``f.queue`` -- only ``short_trig`` can say so), K / a fresh kurz arrival ride it."""
    now = time.time()
    f, st = _r2_hold_scene(monkeypatch, now)
    st["shorts"][H] = (now - 1.0, 3500, X6)
    st["shorts"][H2] = (now - 0.9, 3200, X6)
    st["shorts"][K] = (now - 0.8, 1500, X6)                      # kurz on its own figure: rides the flip (R1a)
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        assert f._decode_collect(now) == "hold"
    hold = f.__dict__["_r2_kurz_hold"]
    assert f.counters["r2_kurz_grant_hold"] == 1 and set(hold["short_trig"]) == {H, H2}, hold
    assert f._r2_flip_held() and not f.queue
    qm = [pp.SetMember("pdflip-nz-qa", 3000, now - 1.0, False), pp.SetMember("pdflip-nz-qb", 3100, now - 0.9, False)]
    sv = pp.SetVerdict(pp.V_FLIP_P, "queue-sum", 0.0, 6100, 3092)
    out = front_mod.Front._r2_kurz_grant_hold(f, sv, now + 0.1, members=qm)          # the queue-sum call: no x_class
    assert out.verdict == pp.V_COLLECT
    assert hold["members"] == ["pdflip-nz-qa", "pdflip-nz-qb"]                        # members: as before, the queue's
    assert set(hold["short_trig"]) == {H, H2}, hold                                  # the frozen SHORT triggers stay
    assert f._r2_flip_held(), "R1a: the SHORT-only flip stays decided between the queue-sum call and the next collect tick"
    assert f._set_flip_foreseeable(now + 0.15)                                       # a fresh kurz arrival rides it
    assert K in st["shorts"] and K not in st["released"]


def test_r1a_queue_sum_site_does_not_resurrect_a_released_trigger(monkeypatch, caplog):
    """The kept triggers are filtered by the LIVE state: a trigger SHORT re-probed kurz holds no flip even after the
    queue-sum call (review A1 of R4OVER-1024 is not undone by keeping the frozen list)."""
    now = time.time()
    f, st = _r2_hold_scene(monkeypatch, now)
    st["shorts"][H] = (now - 1.0, 3500, X6)
    st["shorts"][H2] = (now - 0.9, 3200, X6)
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        assert f._decode_collect(now) == "hold"
    sv = pp.SetVerdict(pp.V_FLIP_P, "queue-sum", 0.0, 6100, 3092)
    front_mod.Front._r2_kurz_grant_hold(f, sv, now + 0.1, members=[pp.SetMember("pdflip-nz-qa", 6100, now - 1.0, False)])
    assert f._r2_flip_held()
    st["shorts"][H] = (now - 1.0, 1100, X6)
    st["shorts"][H2] = (now - 0.9, 1200, X6)                                         # both re-probed kurz
    assert not f._r2_flip_held()


def _members_for(rid, u=3500):
    return [pp.SetMember(rid, u, time.time() - 1.0, False)]


def test_risk2_set_short_kurz_now_failure_is_named_counted_and_raises_as_before(monkeypatch, caplog):
    now = time.time()
    f, st = _r2_hold_scene(monkeypatch, now)
    st["shorts"][K] = (now - 0.8, 1500, X6)

    def boom(*a, **k):
        raise RuntimeError("forced #234 failure")

    monkeypatch.setattr(pp, "set_riders", boom)
    kw = {"frac": 0.5, "x_cls": X6, "x_tok": X6}
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        with pytest.raises(RuntimeError, match="forced #234 failure"):
            front_mod.Front._set_kurz_shorts_now(f, st, _members_for(K, 1500), kw, [], now, "collect")
    (rec,) = _failed_lines(caplog, "SET-SHORT-KURZ-NOW")
    assert "site=collect" in rec.getMessage() and rec.exc_info is not None and rec.levelno == logging.ERROR, rec.getMessage()
    assert f.counters["short_nowait_helper_failed_set_short_kurz_now"] == 1 and f.counters["short_nowait_helper_failed"] == 1


def test_risk2_r2_short_p_members_failure_is_named_and_answers_empty_as_before(monkeypatch, caplog):
    now = time.time()
    f, st = _r2_hold_scene(monkeypatch, now)
    st["shorts"][H] = (now - 1.0, 3500, X6)

    def boom(*a, **k):
        raise RuntimeError("forced #234 failure")

    monkeypatch.setattr(pp, "classify_arrival", boom)
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        assert front_mod.Front._r2_short_p_members(f) == frozenset()                 # behaviour unchanged (R2 direction)
    (rec,) = _failed_lines(caplog, "R2-SHORT-P-MEMBERS")
    assert rec.exc_info is not None and "n=1" in rec.getMessage(), rec.getMessage()
    assert f.counters["short_nowait_helper_failed_r2_short_p_members"] == 1


def test_risk2_r2_short_triggers_failure_is_named_and_answers_empty_as_before(monkeypatch, caplog):
    now = time.time()
    f, st = _r2_hold_scene(monkeypatch, now)
    st["shorts"][H] = (now - 1.0, 3500, X6)

    def boom(*a, **k):
        raise RuntimeError("forced #234 failure")

    monkeypatch.setattr(pp, "set_riders", boom)
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        assert front_mod.Front._r2_short_triggers(f, _members_for(H), X6) == []
    (rec,) = _failed_lines(caplog, "R2-SHORT-TRIGGERS")
    assert rec.exc_info is not None, rec.getMessage()
    assert f.counters["short_nowait_helper_failed_r2_short_triggers"] == 1


def test_risk2_read_helper_logs_are_bounded_but_every_failure_is_counted(monkeypatch, caplog):
    """The read helpers run per tick: 5 logged tracebacks, then silence in the log but the counter keeps counting."""
    now = time.time()
    f, st = _r2_hold_scene(monkeypatch, now)
    st["shorts"][H] = (now - 1.0, 3500, X6)
    monkeypatch.setattr(pp, "classify_arrival", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("forced")))
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        for _ in range(8):
            assert front_mod.Front._r2_short_p_members(f) == frozenset()
    assert len(_failed_lines(caplog, "R2-SHORT-P-MEMBERS")) == 5
    assert f.counters["short_nowait_helper_failed_r2_short_p_members"] == 8
