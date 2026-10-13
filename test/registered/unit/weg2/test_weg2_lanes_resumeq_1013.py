"""PRIORITY LANES x LEDGER, fix "resumeq" (metal int24f6 @ 0ea6d17efd, boot 12:21:03Z, 13.10.2026).

Two losses of one boot, one family: a request that stood in a lane hold across a flip and whose RESUME-VIA-P record was still
in the front's books when the floor fell.

CASE 1 -- the 50.5-s quiesce (D->P flip epoch 73, 13:38:51Z .. 13:39:44Z, flip_total 53 235 ms, quiesce 50 472 ms).
  front 13:38:51,032 ``LANE-RESUME floor=1->0 epoch=8 resumed=13 parked_d=3``; D 13:38:51 ``WEG2-D-LANE floor 1->0 epoch=7->8
  requeued=2`` (``requeue(lane) rids=['weg2-60-548', 'weg2-64-555']``: two lane holds D had taken ITSELF at 13:38:50 --
  ``hold-queued ... via=pass`` -- and that the front never booked).  Both were RESUME-VIA-P rids (``_rvp_inflight``,
  W50-REROUTE 13:38:06): ``_flip_ledger`` leaves such a rid out as "held by D, nothing running".  D ran them again at once (bs 1,
  4 rows, 3 600 decode lines): the flip's park read ``running=0`` (13:38:51,405 PARK-IMMEDIATE "nothing running"), no park RPC,
  the drain was skipped (``ledger-empty``), and the quiesce polled ``/flush_cache`` -> 400 until the decodes ended
  (weg2-64-555 at 13:38:59 after 171 tokens, weg2-60-548 at 13:39:41 after 4 606 tokens).
CASE 2 -- the server stop (P->D flip epoch 73, begin 13:40:11Z, STOP W3 13:41:41Z).
  The same two rids: when their streams ended on D, ``_rvp_stream_end`` dropped their queued P-only Pendings ("1 orphan(s)
  dropped") but not the LEG1-EARLY posted for them at the D->P flip's begin (13:38:51,439).  The legs stood on P
  (``outstanding ['weg2-60-548', 'weg2-64-555']``), P's lane floor 1 skipped them (``decline=loop_skips(weg2_lane_held=2)``),
  they sat in P's waiting queue (``hicache_prefetch(2: ...)``, #1268 PENDING), and the front's P ledger left them out
  (``lane_for`` of an ended rid is 0 < floor): ``front ledger [] (outstanding [...]) vs group(P) flush_cache`` -> W3.

The fix (all behind ``SGLANG_WEG2_LANES``):  the book follows D's answer to the floor RPC (``lane_ctl.d_requeued``), a flip's park
waits for that answer (``settle``), a RESUME-VIA-P Pending that leaves the books takes its early leg off P (``_rvp_drop_leg``),
and the quiesce has a net for both shapes (``quiesce_net``: one park for running work the ledger does not count, one abort for
legs the ledger leaves out).  No timer anywhere.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from sglang.srt.weg2 import front as front_mod  # noqa: E402
from sglang.srt.weg2 import lane_ctl as LC  # noqa: E402
from sglang.srt.weg2 import lanes as LN  # noqa: E402
from sglang.srt.weg2 import phase_policy as PP  # noqa: E402
from sglang.srt.weg2.front import Front  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_weg2_lanes_l4_1008 import NOW, _front, _marks, _note, _on, _pend  # noqa: E402


@pytest.fixture(autouse=True)
def _lanes_switch_pinned_off(monkeypatch):
    """The helpers of the L4 file name the switch themselves (``_on``); the unset state is pinned to off like there."""
    monkeypatch.setenv("SGLANG_WEG2_LANES", "0")


class _D:
    """What the front sees of group D here: the answers of the two RPCs the cases are about, and what D 'runs'."""

    def __init__(self, f, requeued=(), running=0):
        self.f = f
        self.requeued = list(requeued)
        self.running = running
        self.calls = []
        self.floor_gate = None  # an Event: the floor RPC answers when it is set
        self.flush_busy_until_park = False
        self.parked_rids = []
        self.abort_gone = set()

    async def __call__(self, g, path, body, timeout):
        self.calls.append((g.name, path, body))
        if path == LN.RPC_LANE_FLOOR:
            if g.name == "D" and self.floor_gate is not None:
                await self.floor_gate.wait()
            if g.name == "D":
                return 200, json.dumps({"success": True, "floor": body["floor"], "epoch": body["epoch"],
                                        "requeued": self.requeued, "held": 0, "message": "ok"})
            return 200, "{}"
        if path == PP.PARK_PATH:
            asked = list((body or {}).get("rids") or [])
            if asked:
                return 200, json.dumps({"parked": asked, "held": []})
            got = [r for r in self.f.groups["D"].outstanding if r in self.requeued]
            self.parked_rids = got
            self.running = 0
            return 200, json.dumps({"parked": got, "held": []})
        if path == "/flush_cache":
            if g.name == "D":
                return (400 if self.running else 200), "not-idle because: running"
            P = self.f.groups["P"]
            return (400 if P.outstanding else 200), "W-FLUSH GROUP VERDICT PENDING"
        if path == "/abort_request":
            P = self.f.groups["P"]
            P.outstanding.pop(body.get("rid"), None)
            return 200, "{}"
        return 200, "{}"


def _rvp_pending(f, rid, lane, t_arrive):
    """The P-only Pending ``_rvp_take`` queues for a RESUME-VIA-P rid."""
    p = _pend(f, rid, lane, t_arrive, p_only=True, resume_via_p=True)
    f._rvp_inflight.add(rid)
    return p


def _rvp_front(rids, awake="D"):
    f = _front(awake)
    f._rvp_state()
    d = _D(f, requeued=[r for r in rids][:2], running=len(rids))
    f.rpc = d
    return f, d


async def _early_leg(f, rid):
    """What ``Front.leg1`` does around its POST: the rid stands in P's ledger for as long as the task lives."""
    P = f.groups["P"]
    P.outstanding[rid] = time.time() - 3.0
    try:
        await asyncio.sleep(60)
    finally:
        P.outstanding.pop(rid, None)


def _lane_hold_state(f, rids, floor=1):
    """The front's lane controller after a LANE-PREEMPT: ``floor`` stands, the RVP Pendings of ``rids`` are held (out of the
    queue), D's own hold-queued rids are NOT booked as lane parks (D took them itself)."""
    ls = f._lane_state()
    ls.set_floor(floor)
    lc = LC.ctl(f)
    held = []
    for i, r in enumerate(rids):
        p = _rvp_pending(f, r, 0, NOW + i)
        lc.held[id(p)] = LC.Held(p, "queue", NOW + 10)
        lc.mark_held(r)
        held.append(p)
    return lc, held


# ============================================================================ CASE 1: the book follows D's resume
@_on
async def test_case1_the_resume_books_what_d_requeued_and_the_flips_park_sees_it_running(caplog):
    """The metal state: three RESUME-VIA-P rids on D's ledger (all in ``_rvp_inflight``), D re-queues two of them when the floor
    falls (the third stays held: nothing re-queued it).  Before: ``_flip_ledger`` = [] -> "nothing-running", no park RPC.  Now: the
    two are D work, the park RPC goes out, the third stays out of the ledger (D really holds it)."""
    caplog.set_level(logging.INFO, logger="weg2.front")
    rids = ["weg2-60-548", "weg2-64-555", "weg2-68-562"]
    f, d = _rvp_front(rids)
    D = f.groups["D"]
    for r in rids:
        D.outstanding[r] = NOW
        _note(f, r, 0)
    lc, held = _lane_hold_state(f, rids[:2])
    f._rvp_inflight.add(rids[2])
    assert f._flip_ledger(D) == []                                        # the book of the metal run: "nothing running"
    await LC.reconcile(f, "tick")                                         # the lane ends: floor 1 -> 0, floor RPC to P and D
    ls = f._lane_state()
    assert ls.lane_floor == 0
    # D's answer is in the book: the two it re-queued are D work, the one it still holds is not
    assert f._flip_ledger(D) == ["weg2-60-548", "weg2-64-555"]
    assert "weg2-60-548" not in f._rvp_inflight and "weg2-68-562" in f._rvp_inflight
    assert list(f.queue) == [] and lc.held == {}                          # their P-only legs are withdrawn, nothing left to prefill
    assert all(p.fut.done() for p in held)
    line = _marks(caplog, LC.MARK_D_RESUMED)
    assert len(line) == 1 and "n=2 rids=['weg2-60-548', 'weg2-64-555'] rvp_records_ended=2 p_legs_withdrawn=2" in line[0]
    assert f.counters["lane_d_resumed"] == 2 and f.counters["lane_d_resumed_rvp"] == 2
    # the flip's park decides on that book: D runs two decodes -> the park RPC goes out
    f.t_awake = time.time() - 30
    verdict = await f._wait_bound_park(None, cause="over-x")
    assert verdict == "parked"
    parks = [b for g, p, b in d.calls if p == PP.PARK_PATH]
    assert len(parks) == 1 and "rids" not in parks[0]                     # the flip's park: whatever D runs
    assert sorted(f._d_parked) == ["weg2-60-548", "weg2-64-555"]
    assert f._flip_ledger(D) == []                                        # parked: the drain has nothing to wait for


@_on
async def test_case1_the_flips_park_waits_for_the_floor_answer_that_is_on_the_wire():
    """Race: the park decides while D's answer to the resume's floor RPC is still on the wire.  Without ``settle`` it reads the
    old book (running=0) and sends nothing."""
    rids = ["a", "b"]
    f, d = _rvp_front(rids)
    D = f.groups["D"]
    for r in rids:
        D.outstanding[r] = NOW
        _note(f, r, 0)
    lc, _held = _lane_hold_state(f, rids)
    d.floor_gate = asyncio.Event()
    resume = asyncio.ensure_future(LC.reconcile(f, "tick"))
    for _ in range(100):
        await asyncio.sleep(0.01)
        if any(g == "D" and p == LN.RPC_LANE_FLOOR for g, p, _b in d.calls):   # D's floor RPC is on the wire, unanswered
            break
    assert any(g == "D" and p == LN.RPC_LANE_FLOOR for g, p, _b in d.calls) and not resume.done()
    f.t_awake = time.time() - 30
    park = asyncio.ensure_future(f._wait_bound_park(None, cause="over-x"))
    await asyncio.sleep(0.1)
    assert not park.done() and not [1 for _g, p, _b in d.calls if p == PP.PARK_PATH]   # waits for D's answer
    d.floor_gate.set()
    assert await asyncio.wait_for(park, 5) == "parked"
    await asyncio.wait_for(resume, 5)
    assert sorted(f._d_parked) == ["a", "b"]


@_on
async def test_case1_without_a_lane_resume_nothing_waits_and_the_rvp_book_is_the_old_one():
    """A boot that never moved a floor: ``settle`` returns at once, an RVP rid stays out of the ledger."""
    f, d = _rvp_front(["x"])
    D = f.groups["D"]
    D.outstanding["x"] = NOW
    f._rvp_inflight.add("x")
    assert f._flip_ledger(D) == []
    await asyncio.wait_for(LC.settle(f), 1)
    f.t_awake = time.time() - 30
    assert await f._wait_bound_park(None, cause="over-x") == "nothing-running"
    assert not [1 for _g, p, _b in d.calls if p == PP.PARK_PATH]


@_on
async def test_case1_a_d_answer_without_requeued_books_nothing_and_an_old_d_is_no_error():
    rids = ["a"]
    f, d = _rvp_front(rids)
    d.requeued = []
    D = f.groups["D"]
    D.outstanding["a"] = NOW
    _note(f, "a", 0)
    _lane_hold_state(f, rids)
    await LC.reconcile(f, "tick")
    assert f._flip_ledger(D) == [] and "a" in f._rvp_inflight             # D re-queued nothing: still held by D
    assert f.counters["lane_d_resumed"] == 0
    assert LC._requeued_of("not json") == [] and LC._requeued_of("") == [] and LC._requeued_of(None) == []


@_on
async def test_case1_the_quiesce_parks_running_work_the_ledger_does_not_count(caplog):
    """The net: the book missed it anyway (a lost answer, a D that restarted its hold) -- D answers 400 to the flush, the ledger is
    empty, ``weg2_decode_progress.running`` > 0: ONE park RPC, then the flush says idle.  The 50 s of the metal run are not waited."""
    caplog.set_level(logging.INFO, logger="weg2.front")
    f, d = _rvp_front(["a", "b"])
    D = f.groups["D"]
    for r in ("a", "b"):
        D.outstanding[r] = NOW
        _note(f, r, 0)
        f._rvp_inflight.add(r)                     # the old book: held by D
    LC.ctl(f).mark_held("a")                       # a lane acted on this boot
    assert f._flip_ledger(D) == []

    async def _prog(g):
        return {"running": d.running, "gen_tokens_total": 1, "forward_ct": 1}

    f._weg2_decode_progress = _prog
    t0 = time.time()
    idle, msg = await f.quiesce(D)
    assert idle is True and time.time() - t0 < 5
    parks = [b for g, p, b in d.calls if p == PP.PARK_PATH]
    assert len(parks) == 1 and parks[0]["reason"] == "quiesce-unbooked" and "rids" not in parks[0]
    assert f.counters["lane_quiesce_park"] == 1 and sorted(f._d_parked) == ["a", "b"]
    assert len(_marks(caplog, LC.MARK_QUIESCE_PARK)) == 1


@_on
async def test_case1_the_quiesce_net_is_silent_without_a_lane_and_without_running_work():
    f, d = _rvp_front(["a"])
    D = f.groups["D"]
    D.outstanding["a"] = NOW
    f._rvp_inflight.add("a")

    async def _prog(g):
        return {"running": d.running}

    f._weg2_decode_progress = _prog
    d.running = 1
    st = {}
    await LC.quiesce_net(f, D, st)                 # no lane ever acted on this boot: no park, no read
    assert not [1 for _g, p, _b in d.calls if p == PP.PARK_PATH]
    LC.ctl(f).mark_held("a")
    d.running = 0
    await LC.quiesce_net(f, D, {})                 # a lane acted, but D runs nothing: no park
    assert not [1 for _g, p, _b in d.calls if p == PP.PARK_PATH]


# ============================================================================ CASE 2: the ghost leg on P
@_on
async def test_case2_a_rvp_stream_end_takes_the_early_leg_off_p_and_aborts_the_rid(caplog):
    """The metal state: the P-only Pending of a RESUME-VIA-P rid is in the queue with its LEG1-EARLY on P; D finishes the stream
    (``_rvp_stream_end``).  Before: the Pending is dropped, the early leg stays on P (a ghost: P.outstanding keeps the rid, no abort)."""
    caplog.set_level(logging.INFO, logger="weg2.front")
    f, d = _rvp_front(["weg2-64-555"], awake="P")
    P = f.groups["P"]
    p = _rvp_pending(f, "weg2-64-555", 0, NOW)
    f.queue.append(p)
    p._leg1_early = asyncio.ensure_future(_early_leg(f, "weg2-64-555"))
    Front._leg1_early_watch(f, p)
    early = p._leg1_early
    await asyncio.sleep(0.02)
    assert "weg2-64-555" in P.outstanding
    f._rvp_stream_end("weg2-64-555")
    await asyncio.sleep(0.05)
    assert list(f.queue) == [] and "weg2-64-555" not in f._rvp_inflight
    assert early.cancelled() and "weg2-64-555" not in P.outstanding
    assert [(g, b["rid"]) for g, pth, b in d.calls if pth == "/abort_request"] == [("P", "weg2-64-555")]
    assert f.counters["rvp_orphan_early_cancel"] == 1
    assert len(_marks(caplog, "WEG2 RVP-ORPHAN-LEG")) == 1


@_on
async def test_case2_a_lane_held_rvp_pending_of_an_ended_stream_is_not_put_back_by_the_resume():
    f, d = _rvp_front(["r"], awake="D")
    lc, held = _lane_hold_state(f, ["r"])
    assert len(lc.held) == 1
    f._rvp_stream_end("r")
    assert lc.held == {} and held[0].fut.done() and "r" not in f._rvp_inflight
    await LC.reconcile(f, "tick")                                           # the floor falls: nothing comes back
    assert list(f.queue) == []


@_on
async def test_case2_the_p_quiesce_aborts_legs_the_ledger_leaves_out_instead_of_a_w3(caplog):
    """The net for the same shape from any other source: floor 1, ghost rids in P's ``outstanding`` (lane 0 < floor: out of the
    ledger), P answers 400 until they are aborted.  Before: 90 s of polls, then ``W3 Weg2DrainWitnessDisagreement`` (the metal
    stop).  Now: one abort per rid, the next flush says idle."""
    caplog.set_level(logging.INFO, logger="weg2.front")
    f, d = _rvp_front([], awake="P")
    P = f.groups["P"]
    f._lane_state().set_floor(1)
    LC.ctl(f).mark_held("weg2-60-548")
    for r in ("weg2-60-548", "weg2-64-555"):
        P.outstanding[r] = NOW                                              # ended on D, never noted: lane 0
    assert f._flip_ledger(P) == []
    old = front_mod.QUIESCE_DEADLINE_S
    front_mod.QUIESCE_DEADLINE_S = 3.0
    try:
        idle, msg = await f.quiesce(P)
    finally:
        front_mod.QUIESCE_DEADLINE_S = old
    assert idle is True and not P.outstanding
    assert sorted(b["rid"] for g, pth, b in d.calls if pth == "/abort_request") == ["weg2-60-548", "weg2-64-555"]
    assert f.counters["lane_quiesce_abort"] == 2 and len(_marks(caplog, LC.MARK_QUIESCE_ABORT)) == 1


@_on
async def test_case2_the_p_quiesce_net_never_touches_a_leg_the_ledger_counts():
    f, d = _rvp_front([], awake="P")
    P = f.groups["P"]
    f._lane_state().set_floor(1)
    LC.ctl(f).mark_held("x")
    P.outstanding["hi"] = NOW
    _note(f, "hi", 1)                                                        # the floor lane's own leg: in the ledger
    assert f._flip_ledger(P) == ["hi"]
    await LC.quiesce_net(f, P, {})
    assert "hi" in P.outstanding and not [1 for _g, pth, _b in d.calls if pth == "/abort_request"]


# ============================================================================ the switch off / lane 0 only: nothing changes
def test_switch_off_the_stream_end_and_the_quiesce_are_the_old_ones():
    async def body():
        f = _front("P")
        f._rvp_state()
        calls = []

        async def rpc(g, path, b, t):
            calls.append((g.name, path))
            return (200 if path != "/flush_cache" else 200), "{}"

        f.rpc = rpc
        P = f.groups["P"]
        p = _pend(f, "r", 0, NOW, p_only=True, resume_via_p=True)
        f._rvp_inflight.add("r")
        f.queue.append(p)
        p._leg1_early = asyncio.ensure_future(_early_leg(f, "r"))
        await asyncio.sleep(0.02)
        f._rvp_stream_end("r")
        await asyncio.sleep(0.02)
        assert list(f.queue) == [] and not p._leg1_early.cancelled()          # the old behaviour: the early leg is left alone
        assert not [1 for _g, pth in calls if pth == "/abort_request"] and f.counters["rvp_orphan_early_cancel"] == 0
        idle, _m = await f.quiesce(P)
        assert idle and f.counters["lane_quiesce_abort"] == 0 and f.counters["lane_quiesce_park"] == 0
        p._leg1_early.cancel()

    asyncio.run(body())


@_on
async def test_lane_0_only_the_quiesce_and_the_park_are_the_old_ones():
    """Lane 0 only, no lane ever acted: ``touched`` is False, ``settle`` returns at once, the net never reads D's progress."""
    f, d = _rvp_front(["a"])
    D = f.groups["D"]
    D.outstanding["a"] = NOW
    _note(f, "a", 0)
    f._rvp_inflight.add("a")
    assert not LC.touched(f)
    reads = []

    async def _prog(g):
        reads.append(1)
        return {"running": 5}

    f._weg2_decode_progress = _prog
    d.running = 0
    idle, _m = await f.quiesce(D)
    assert idle and not reads and f.counters["lane_quiesce_park"] == 0
