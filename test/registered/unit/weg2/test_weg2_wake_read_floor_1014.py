"""WAKE-READ-FLOOR-1014 (NF h17 38c6d40239, boot ..._1014_153824, D TP0).

15:48:07 D took lane floor 1 (``WEG2-D-LANE floor 0->1``, the lane-1 image turn weg2-2-55). 15:48:09, at the
P->D wake: ``#248 WAKE-READ issued=5 ['weg2-2-30', 'weg2-2-31', 'weg2-2-44', 'weg2-2-45', 'weg2-2-47']`` and
``#248f WAKE-READ CAPACITY-WAIT ... in_flight_pages=6309 arena_slots=6485`` -- five lane-0 hold reads that the
floor would not seat took the arena. 15:48:31 ``ARENA-REF-HOLDERS ... tree_in_use=5848 arena_pinned=6323`` of 6323.
weg2-2-55's own read delivered 0 / 128 / 24640 of 192512 in 43 s (L3-FILL ROOM-WAIT / JOIN-WAIT on a full arena),
15:48:54 W31 -> W50 -> flip D->P; P read all 192512 in 8.9 s.

A hold read below D's floor now waits (capacity waiter, parked in the settle) until the floor falls to its lane.
"""

import logging
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_weg2_wake_read_seat_gate_1014 import _req, _sched  # noqa: E402

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.weg2 import d_lane, park_l3  # noqa: E402


@pytest.fixture(autouse=True)
def _group_d(monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_GROUP", "D")
    monkeypatch.delenv(park_l3.FLOOR_ENV, raising=False)
    with envs.SGLANG_WEG2_ENABLE_PARK_L3.override(True):
        yield


def _lane(r, lane):
    r.priority = lane
    return r


def _h17_hold():
    """The five lane-0 hold reads of 15:48:09 (pages as in the CAPACITY-WAIT line: 6309 together)."""
    pages = [1300, 1280, 1250, 1240, 1239]
    rids = ["weg2-2-30", "weg2-2-31", "weg2-2-44", "weg2-2-45", "weg2-2-47"]
    return [_lane(_req(r, p), 0) for r, p in zip(rids, pages)]


def test_default_on():
    assert park_l3._floor_on()


def test_h17_no_lane0_hold_read_under_floor_1(caplog):
    h = _sched()
    setattr(h, d_lane.FLOOR_ATTR, 1)
    hold = _h17_hold()
    with caplog.at_level(logging.INFO):
        out = park_l3.issue_deferred_reads(h, hold)
    assert out == [] and h.issued == [], "15:48:09 issued 5 lane-0 reads and pinned 6309 of 6485 slots"
    assert all(park_l3.capacity_waiting(r) for r in hold)
    assert any(park_l3.FLOOR_MARK in m for m in caplog.messages)


def test_a_floor_lane_hold_read_still_reads():
    h = _sched()
    setattr(h, d_lane.FLOOR_ATTR, 1)
    hold = _h17_hold() + [_lane(_req("weg2-2-55", 3008), 1)]
    park_l3.issue_deferred_reads(h, hold)
    assert h.issued == ["weg2-2-55"]


def test_the_waiters_read_when_the_floor_falls():
    h = _sched()
    setattr(h, d_lane.FLOOR_ATTR, 1)
    hold = _h17_hold()
    park_l3.issue_deferred_reads(h, hold)
    assert park_l3.issue_capacity_waiters(h, hold) == []        # floor still 1: nothing
    setattr(h, d_lane.FLOOR_ATTR, 0)
    out = park_l3.issue_capacity_waiters(h, hold)
    assert [r.rid for r in out] == h.issued and h.issued, "the floor fell: they read, oldest first"


def test_no_floor_is_the_old_issue():
    h = _sched()
    hold = _h17_hold()
    park_l3.issue_deferred_reads(h, hold)
    assert h.issued, "floor 0: the seat/page gates alone decide, as before"


def test_switch_off(monkeypatch):
    monkeypatch.setenv(park_l3.FLOOR_ENV, "0")
    h = _sched()
    setattr(h, d_lane.FLOOR_ATTR, 1)
    park_l3.issue_deferred_reads(h, _h17_hold())
    assert h.issued
