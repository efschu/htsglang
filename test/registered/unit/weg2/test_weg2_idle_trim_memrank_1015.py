"""MEMRANK-1015 leak fix 1: the bounded idle malloc_trim of IdleCensus could never fire.

Metal: boot ...042125 (h16n 17c2563647), 378 WEG2-IDLE-CENSUS lines (358 P, 20 D),
``trimmed=0`` in every one, while the P ranks grew +0.8 / +1.8 / +0.9 MiB/min
(PP0 / PP1 / PP2, median over ~95 idle minutes each) and each sleep's
malloc_trim handed 100-250 MiB back (``WEG2-SLEEP-HOST-HEAP`` returned=...).
Cause: ``base = self._anon_at_trim if not None else anon_kib`` -- the attribute
stayed None until a trim had run, so base was always the current reading and
the growth always 0. Red on 364ab5de5d, green with the fix.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import idle_census as ic  # noqa: E402


@pytest.fixture
def census(monkeypatch):
    cur = {"v": 3000 * 1024}
    trims = []
    monkeypatch.setattr(ic, "_read_status", lambda: {"RssAnon": cur["v"], "Threads": 7})

    def _trim():
        trims.append(cur["v"])
        return True

    monkeypatch.setattr(ic, "_malloc_trim", _trim)
    c = ic.IdleCensus(group="P", rank=1, period_s=0.0)
    c.maybe_emit()  # the first call only opens the window
    return c, cur, trims


def _mib(n):
    return int(n * 1024)


def test_idle_growth_over_the_floor_triggers_the_trim(census):
    c, cur, trims = census
    cur["v"] += _mib(9)
    line = c.maybe_emit()
    assert len(trims) == 1 and "trimmed=1" in line


def test_small_growth_does_not_trim(census):
    c, cur, trims = census
    for _ in range(3):
        cur["v"] += _mib(2)  # +6 MiB over three periods, each below 8
        c.maybe_emit()
    assert trims == []
    cur["v"] += _mib(3)      # now 9 MiB above the floor
    c.maybe_emit()
    assert len(trims) == 1


def test_floor_follows_a_drop_such_as_a_sleeps_own_trim(census):
    c, cur, trims = census
    cur["v"] -= _mib(200)    # a sleep's malloc_trim took the rank down
    c.maybe_emit()
    assert trims == []
    cur["v"] += _mib(7)      # +7 over the NEW floor: not yet
    c.maybe_emit()
    assert trims == []
    cur["v"] += _mib(2)      # +9 over the new floor
    c.maybe_emit()
    assert len(trims) == 1


def test_baseline_resets_after_a_trim(census):
    c, cur, trims = census
    cur["v"] += _mib(10)
    c.maybe_emit()
    assert len(trims) == 1
    cur["v"] += _mib(1)      # +1 over the post-trim reading
    c.maybe_emit()
    assert len(trims) == 1


def test_idle_trim_switch_off(census, monkeypatch):
    c, cur, trims = census
    monkeypatch.setenv("SGLANG_WEG2_IDLE_TRIM", "0")
    cur["v"] += _mib(50)
    line = c.maybe_emit()
    assert trims == [] and "trimmed=0" in line


def test_idle_trim_default_on():
    from sglang.srt.environ import envs

    assert envs.SGLANG_WEG2_IDLE_TRIM.get() is True
