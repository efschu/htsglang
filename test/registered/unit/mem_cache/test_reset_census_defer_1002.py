# SPDX-License-Identifier: Apache-2.0
"""CENSUS-DEFER (02.10.): the at=reset holder census leaves the tree reset.

N6d (..._ec4d492f58_1002_170955) P sleeps, 'WEG2-TREE-RESET-SUB': census
19-24 ms of a 34-48 ms reset on PP0, 23.5 of 37 ms on PP1/PP2 -- inside the P>D
quiesce. The census is an instrument (no branch reads it) and already runs
concurrently with the scheduler (the 60 s ARENA-REF-CENSUS thread; a reset
under it is named snapshot=torn). Switch SGLANG_WEG2_RESET_CENSUS_DEFER_S
(default 5 s; 0 = in the reset, as before). Real ``_reset_full`` on the #1424g
arena shell.
"""
from __future__ import annotations

import importlib.util
import logging
import time
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "_h1424g_census_defer", Path(__file__).with_name("test_arena_reset_orphans_1424g.py"))
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)

arena = H.arena  # the fixture
LOG = "sglang.srt.mem_cache.unified_radix_cache"


def _census_lines(caplog):
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith("ARENA-REF-HOLDERS at=reset")]


def test_red_the_reset_returns_before_the_census_and_the_census_lands_later(arena, caplog, monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_RESET_CENSUS_DEFER_S", "0.2")
    t, _pool, _slots = H._parked_two_rids_same_prefix(arena)
    with caplog.at_level(logging.INFO, logger=LOG):
        t._reset_full()
        assert _census_lines(caplog) == [], "the census ran inside the reset"
        time.sleep(0.6)
    lines = _census_lines(caplog)
    assert len(lines) == 1 and "deferred_s=0.2" in lines[0], lines
    assert "tree=0" in lines[0] and "gap=" in lines[0], lines


def test_the_default_defers_by_five_seconds(arena, monkeypatch):
    monkeypatch.delenv("SGLANG_WEG2_RESET_CENSUS_DEFER_S", raising=False)
    from sglang.srt.mem_cache import unified_radix_cache as urc

    assert urc._weg2_reset_census_defer_s() == 5.0
    t, _pool, _slots = H._parked_two_rids_same_prefix(arena)
    t._reset_full()
    timer = t._weg2_census_timer
    assert timer.is_alive() and timer.interval == 5.0
    timer.cancel()


def test_a_newer_reset_replaces_the_pending_census(arena, caplog, monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_RESET_CENSUS_DEFER_S", "0.3")
    t, _pool, _slots = H._parked_two_rids_same_prefix(arena)
    with caplog.at_level(logging.INFO, logger=LOG):
        t._reset_full()
        first = t._weg2_census_timer
        t._reset_full()
        assert t._weg2_census_timer is not first
        time.sleep(0.8)
    assert len(_census_lines(caplog)) == 1, _census_lines(caplog)


def test_zero_is_the_census_in_the_reset_as_before(arena, caplog, monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_RESET_CENSUS_DEFER_S", "0")
    t, _pool, _slots = H._parked_two_rids_same_prefix(arena)
    with caplog.at_level(logging.INFO, logger=LOG):
        t._reset_full()
    lines = _census_lines(caplog)
    assert len(lines) == 1 and "deferred_s" not in lines[0], lines


def test_other_census_sites_are_never_deferred(arena, caplog, monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_RESET_CENSUS_DEFER_S", "5")
    t, _pool, _slots = H._parked_two_rids_same_prefix(arena)
    with caplog.at_level(logging.INFO, logger=LOG):
        t._weg2_log_holder_census("shrink")
    assert [r for r in caplog.records if r.getMessage().startswith("ARENA-REF-HOLDERS at=shrink")]
