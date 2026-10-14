"""LANEHOLD-FLIP-1014 (NF h13, boot ..._84e787ec96_1014_135753, front.log 14:25:46-49).

weg2-20-119 (lane 0) was parked by the lane-1 preempt of weg2-20-129 (LANE-RAISE floor=1 14:25:46,365; LANE-PREEMPT
14:25:49,491). D refused its resume mid-stream (prefix gone) and filed a needs-P record. The controller pass:

  _lane_reconcile -> sweep (nothing below the floor in queue yet)
  _rvp_take       -> 14:25:49,541 LANE-REPREFILL + W50-REROUTE: the P-only Pending is appended to ``queue``
  ARRIVAL-SEAT    -> 14:25:49,543 PBOUND-FLIP-NOW rid=weg2-20-119 + PARK-IMMEDIATE -> 14:25:49,583 WEG2-FLIP begin D->P

The lane loop's tick sweep held it only afterwards (LANE-EARLY-TAKE 14:25:49,699). A flip for a request held under
the floor. The invariant ("nothing below the floor in queue/_ready_for_d", lane_ctl.sweep) is restored in the same
synchronous step that queues the leg -- the LANES FIX 2 form.
"""
from __future__ import annotations

import asyncio
import inspect
import os
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from sglang.srt.weg2 import front as F  # noqa: E402
from sglang.srt.weg2 import lane_ctl as LC  # noqa: E402
from sglang.srt.weg2 import lanes as LN  # noqa: E402
from sglang.srt.weg2.front import Front  # noqa: E402

X = 7733


def _front(monkeypatch, records, lanes="1"):
    monkeypatch.setenv(LN.ENV_LANES, lanes)
    f = Front("http://p", "http://d", "D", "t", "", 0, 0, {}, 45.0, tp_prefill_max_tokens=X)
    f.state = "serving"
    f.admit_d = True
    f._rvp_state()
    f._rvp_dir = "/nonexistent-rvp-dir"
    monkeypatch.setattr(F._rvp, "take_requests", lambda d: list(records))
    return f


def _rec(rid):
    return {"rid": rid, "input_ids": list(range(64)), "d_extent": 45944,
            "reason": "x_refusal_midstream", "attempt": 1}


def test_weg2_20_119_a_needs_p_leg_below_the_floor_is_held_at_once(monkeypatch):
    async def go():
        f = _front(monkeypatch, [_rec("weg2-20-119")])
        ls = f._lane_state()
        ls.note("weg2-20-119", 0)
        ls.note("weg2-20-129", 1)
        ls.set_floor(1)
        assert f._rvp_take() == 1
        assert list(f.queue) == [], "no ARRIVAL-SEAT candidate: no PBOUND-FLIP-NOW, no PARK-IMMEDIATE"
        lc = LC.ctl(f)
        held = [h.p for h in lc.held.values()]
        assert [p.rid for p in held] == ["weg2-20-119"] and held[0].resume_via_p and held[0].p_only
        # the floor falls: it counts again, with its original arrival
        ls.set_floor(0)
        LC._restore_held(f, lc, 0, time.time())
        assert [p.rid for p in f.queue] == ["weg2-20-119"]
    asyncio.run(go())


def test_a_needs_p_leg_of_the_floor_lane_is_queued_as_before(monkeypatch):
    async def go():
        f = _front(monkeypatch, [_rec("weg2-20-129")])
        ls = f._lane_state()
        ls.note("weg2-20-129", 1)
        ls.set_floor(1)
        f._rvp_take()
        assert [p.rid for p in f.queue] == ["weg2-20-129"]
        assert not LC.ctl(f).held
    asyncio.run(go())


def test_no_floor_no_hold(monkeypatch):
    async def go():
        f = _front(monkeypatch, [_rec("weg2-20-119")])
        f._lane_state().note("weg2-20-119", 0)
        f._rvp_take()
        assert [p.rid for p in f.queue] == ["weg2-20-119"]
    asyncio.run(go())


def test_lanes_off_is_the_old_path(monkeypatch):
    async def go():
        f = _front(monkeypatch, [_rec("weg2-20-119")], lanes="0")
        f._lane_state().set_floor(1)  # a stale floor cannot hold anything with the switch off
        f._rvp_take()
        assert [p.rid for p in f.queue] == ["weg2-20-119"]
    asyncio.run(go())


def test_the_controller_sweeps_after_its_takes_before_it_reads_the_queue():
    src = inspect.getsource(Front)
    i = src.index("self._hl_sweep()  # #243 seam: lost hand-offs of waiting rids -> P")
    blk = src[i:i + 900]
    j_take = blk.index("self._rvp_take()")
    j_sweep = blk.index("_lctl.sweep(self)")
    j_d = blk.index('if self.awake == "D":')
    assert j_take < j_sweep < j_d, "the sweep stands between the takes and the pass's flip/park/batch reads"


def test_lanes_default_is_on(monkeypatch):
    monkeypatch.delenv(LN.ENV_LANES, raising=False)
    assert LN.enabled()
