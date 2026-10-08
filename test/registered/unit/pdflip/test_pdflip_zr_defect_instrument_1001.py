"""ZR instrument (01.10., user: a P hand-off or a D resume that computes tokens
again is a bug; anything below 100 % path=skip is a defect).

Pins the PDFLIP-HANDBACK-DEFECT line: written for every hand-off/resume that
computes tokens again (an agreed tail taken only in part, or an origin request
admitted with no tail at all), never for a request D computes for the first
time (an X-route turn without origin), never a gate.
"""

import inspect
import logging
from types import SimpleNamespace

import pytest

from flliper.srt.pdflip import handback_claim as hc
from flliper.srt.pdflip import tail_adopt as ta


@pytest.fixture(autouse=True)
def _clean():
    hc._ORIGIN.clear()
    yield
    hc._ORIGIN.clear()


def _defects(caplog):
    return [r.getMessage() for r in caplog.records if hc.DEFECT_MARK in r.getMessage()]


def test_skip_writes_no_defect(caplog):
    hc.note_origin("pdflip-1-1", hc.ORIGIN_HANDOFF)
    with caplog.at_level(logging.INFO):
        hc.handback_line("pdflip-1-1", 41464, 41464, 0, "skip")
    assert _defects(caplog) == []
    assert hc.origin("pdflip-1-1") is None  # one verdict per resume


def test_extend_from_page_anchor_is_a_defect(caplog):
    # metal y6h pdflip-4-15: N=41464, END state dropped in a non-empty batch
    hc.note_origin("pdflip-4-15", hc.ORIGIN_PARK)
    with caplog.at_level(logging.INFO):
        hc.handback_line("pdflip-4-15", 41464, 41408, 56, "extend:end_only:batch_not_empty")
    (line,) = _defects(caplog)
    assert "rid=pdflip-4-15" in line and "origin=park" in line
    assert "why=end_only:batch_not_empty" in line and "d_compute=56" in line


def test_e1_is_a_defect_too(caplog):
    with caplog.at_level(logging.INFO):
        hc.handback_line("pdflip-2-2", 4446, 4443, 3, "e1")
    (line,) = _defects(caplog)
    assert "path=e1" in line and "origin=tail" in line


def test_origin_without_tail_names_the_staging_reason(caplog):
    # metal y6h pdflip-5-17: D's park, no parts at the wake, D extended 3744
    hc.note_origin("pdflip-5-17", hc.ORIGIN_PARK)
    hc.note_why("pdflip-5-17", "no_parts")
    with caplog.at_level(logging.INFO):
        hc.admission_without_tail("pdflip-5-17", 71584, 67840)
    (line,) = _defects(caplog)
    assert "why=no_parts" in line and "d_compute=3744" in line and "origin=park" in line


def test_first_compute_is_silent(caplog):
    # an X-route turn D computes for the first time has no origin
    with caplog.at_level(logging.INFO):
        assert hc.admission_without_tail("pdflip-9-9", 5000, 3000) is None
    assert _defects(caplog) == []


def test_nothing_computed_is_silent(caplog):
    hc.note_origin("pdflip-3-3", hc.ORIGIN_HANDOFF)
    with caplog.at_level(logging.INFO):
        assert hc.admission_without_tail("pdflip-3-3", 2048, 2048) is None
    assert _defects(caplog) == []


def test_plan_adopt_without_entry_reports(monkeypatch, caplog):
    monkeypatch.setattr(ta, "adopt_enabled", lambda: True)
    ta._AGREED.pop("pdflip-7-7", None)
    hc.note_origin("pdflip-7-7", hc.ORIGIN_HANDOFF)
    req = SimpleNamespace(rid="pdflip-7-7", full_untruncated_fill_ids=list(range(300)))
    with caplog.at_level(logging.INFO):
        assert ta.plan_adopt(req, 256) is None
    (line,) = _defects(caplog)
    assert "why=no_staging" in line and "d_compute=44" in line


def test_agree_no_parts_names_why():
    hc.note_origin("pdflip-8-8", hc.ORIGIN_PARK)
    ta._JOBS["pdflip-8-8"] = SimpleNamespace(
        thread=None, headers=[], state="none", have=0, want=0, box=[], held_since=-1.0
    )
    ta.agree("pdflip-8-8", 0)
    assert hc.origin("pdflip-8-8")["why"] == "no_parts"


def test_origin_wiring():
    from flliper.srt.managers.scheduler import Scheduler
    from flliper.srt.pdflip import d_park_runtime

    assert "note_origin(req.rid, ORIGIN_HANDOFF)" in inspect.getsource(Scheduler._prefetch_kvcache)
    assert "_hb.note_origin(req.rid, _hb.ORIGIN_PARK)" in inspect.getsource(d_park_runtime.park_running)
