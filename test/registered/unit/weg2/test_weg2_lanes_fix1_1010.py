"""PRIORITY LANES fix round 1 (metal jjbbrx, boot 03:04:42Z, image int23lanes @ b99a895d9b), 10.10.2026.

A. BLOCKER probe 3 -- ``park_running(rids, hold="lane")`` of a CHUNKED PREFILL on D.  The RPC runs on the scheduler
   thread between two passes: the in-flight chunk lands first (``_land_inflight``), that landing IS the chunk
   boundary; the request is retracted like a decode seat with its span retained, and ``sched.chunked_req`` is
   released with it.  Before the fix ``chunked_req`` kept pointing at the retracted request and the next
   ``get_next_batch_to_run`` read ``chunked_req.extend_range.end`` (None after the retract): ``AttributeError:
   'NoneType' object has no attribute 'end'`` on all three D ranks, W17 Weg2GroupDead.  A chunked request of ANOTHER
   lane stays the chunked request and is kept out of the merge.  Every TP rank parks the same request the same way.
B. MAJOR probe 4 -- the flip end is an EVENT for the lane controller (``LC.flip_done``: kick + timestamp), the
   marker names the latency flip end -> begin of the preempt (``LANE-RAISE``, ``since_flip_done_ms=``), and a lower
   lane's speculative early leg 1 is cancelled and aborted when the DEFER is decided (during the flip), not at its
   end (the abort at the end lost the race against P's first chunk by 1 ms).
C. Probe 2b -- the P drain pool postpones ``lane_take`` while a floor-lane request has arrived but has no place yet
   (``LC.take_wait``): no P->D->P flip pair for a lane-1 leg that is itself P-bound; the L3 chunk park can fire.

``scheduler.py`` itself is READ, not imported (its import pulls transformers -> torchao); the pass-head read that
killed D is reproduced by ``_next_pass_head`` and pinned to the source by ``test_the_scheduler_line_...``.
"""
from __future__ import annotations

import asyncio
import collections
import logging
import os
import sys
import time
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from sglang.srt.mem_cache.base_prefix_cache import FORCE_HOST_WRITE_THROUGH_ATTR  # noqa: E402
from sglang.srt.weg2 import d_lane  # noqa: E402
from sglang.srt.weg2 import d_park_runtime as rt  # noqa: E402
from sglang.srt.weg2 import d_seats as ds  # noqa: E402
from sglang.srt.weg2 import lane_ctl as LC  # noqa: E402
from sglang.srt.weg2 import lanes as LN  # noqa: E402
from sglang.srt.weg2.front import Front, _p_drain_pool  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_weg2_lanes_l2_1008 import (  # noqa: E402,F401  (fixtures + stand-ins of the L2 file)
    _Batch, _Sched, _floor, _group_d, _park_lane, _req, clock, lanes_on,
)
from test_weg2_lanes_l4_1008 import (  # noqa: E402
    NOW, _arrive, _early_leg, _front, _marks, _note, _on, _pend,
)
from test_weg2_phase_policy_h91c import _until  # noqa: E402

_SRT = os.path.dirname(os.path.dirname(ds.__file__))


# ================================================================== A. the chunked prefill on D
class _Rng:
    def __init__(self, end):
        self.end = end


class _MBatch(_Batch):
    """The running batch of the L2 stand-in that also merges an extend batch and RESETS what a retract resets
    (``Req.reset_for_retract``: ``extend_range = None``)."""

    def release_req(self, idx, remaining, server_args, retain=False):
        super().release_req(idx, remaining, server_args, retain=retain)
        self.reqs[idx].extend_range = None

    def merge_batch(self, other):
        self.reqs.extend(other.reqs)

    def filter_batch(self, chunked_req_to_exclude=None, keep_indices=None):
        if keep_indices is None and chunked_req_to_exclude:
            ex = list(chunked_req_to_exclude)
            self.reqs = [r for r in self.reqs if not any(r is e for e in ex)]
            return
        super().filter_batch(chunked_req_to_exclude=chunked_req_to_exclude, keep_indices=keep_indices)


class _Extend(_MBatch):
    """``sched.last_batch`` after a prefill pass: the extend batch that carried the chunked request.  When the running
    batch is empty ``_land_inflight`` makes THIS batch the running batch (as the real code does), so it is a full
    batch stand-in."""

    def __init__(self, reqs):
        super().__init__(reqs)
        self.excluded = None
        self.forward_mode = types.SimpleNamespace(is_extend=lambda: True)

    def filter_batch(self, chunked_req_to_exclude=None, keep_indices=None):
        if keep_indices is not None:
            return super().filter_batch(keep_indices=keep_indices)
        ex = list(chunked_req_to_exclude or [])
        if chunked_req_to_exclude is not None:
            self.excluded = ex
        self.reqs = [r for r in self.reqs if not any(r is e for e in ex)]


def _chunked(n, lane, done=3000, chunk=1600):
    r = _req(n, lane)
    r.prefix_indices = list(range(done))
    r.extend_range = _Rng(done + chunk)
    r.inflight_middle_chunks = 0
    r.req_pool_idx = 7
    return r


def _mid_prefill(running, chunk, finished=(), waiting=()):
    """D mid chunked prefill: ``running`` decode seats, ``chunk`` the chunked request, the pass's extend batch
    (``finished`` = requests whose prefill ended in it) is ``last_batch``."""
    s = _Sched(waiting=waiting)
    s.running_batch = _MBatch(running)
    s.last_batch = _Extend(list(finished) + [chunk])
    s.chunked_req = chunk
    return s


def _next_pass_head(s):
    """scheduler.py ``get_next_batch_to_run``: ``if self.chunked_req is not None: ... if
    self.chunked_req.extend_range.end > len(self.chunked_req.prefix_indices): stash`` (the line that killed D)."""
    if s.chunked_req is not None:
        if s.chunked_req.extend_range.end > len(s.chunked_req.prefix_indices):
            return "stash"
    return "-"


def test_the_scheduler_line_that_killed_d_is_the_one_the_stand_in_reproduces():
    src = open(os.path.join(_SRT, "managers", "scheduler.py")).read()
    assert "if self.chunked_req.extend_range.end > len(self.chunked_req.prefix_indices):" in src
    assert "chunked_req_to_exclude.add(self.chunked_req)" in src       # the merge rule _land_inflight follows


def test_the_lane_park_of_a_chunked_prefill_parks_at_the_chunk_border_and_releases_chunked_req(lanes_on, caplog):
    dec, ch = _req(1, 0), _chunked(2, 0)
    s = _mid_prefill([dec], ch)
    with caplog.at_level("INFO"):
        out = _park_lane(s, [dec.rid, ch.rid])
    assert out.success and out.lane is True and out.parked == [dec.rid, ch.rid] and out.lane_skipped == {}
    assert s.chunked_req is None                                   # released with the retract (the fix)
    assert _next_pass_head(s) == "-"                               # the pass head no longer reads a retracted req
    assert ch.extend_range is None                                 # ... it was retracted (what killed D)
    # span retained: release_req(retain=True) + the forced host write-through, exactly the decode seat's park
    assert s.running_batch.released == [(dec.rid, True), (ch.rid, True)]
    assert getattr(ch, FORCE_HOST_WRITE_THROUGH_ATTR) is True
    assert s.last_batch is None and s.running_batch.reqs == []
    assert [r.rid for r in s.weg2_d_parked] == [dec.rid, ch.rid]
    assert all(d_lane.lane_held(r) and ds.park_site(r) == ds.SITE_FLIP for r in (dec, ch))
    line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith(LN.MARK_D_PARK_PARK))
    assert "chunk_boundary=1" in line and "hold=lane" in line


def test_a_lane_park_without_a_chunked_prefill_says_chunk_boundary_0(lanes_on, caplog):
    a = _req(1, 0)
    s = _Sched(running=[a])
    with caplog.at_level("INFO"):
        out = _park_lane(s, [a.rid])
    assert out.parked == [a.rid]
    line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith(LN.MARK_D_PARK_PARK))
    assert "chunk_boundary=0" in line


def test_the_parked_chunked_prefill_resumes_at_the_head_of_the_queue_when_the_floor_falls(lanes_on):
    """Plan 0/1: the parked request keeps its age and its retained span; the resume is the ordinary re-admission
    (prefix match), no re-prefill instruction anywhere on D's side (LANE-REPREFILL is the front's counter)."""
    ch = _chunked(2, 0)
    s = _mid_prefill([], ch, waiting=[_req(9, 0)])
    assert _floor(s, 1, 1).success
    assert _park_lane(s, [ch.rid]).parked == [ch.rid]
    out = _floor(s, 0, 2)
    assert out.success and out.requeued == [ch.rid]
    assert s.waiting_queue[0] is ch and not d_lane.lane_held(ch)
    assert s.chunked_req is None and ch.extend_range is None       # re-admission builds the next range
    assert getattr(ch, FORCE_HOST_WRITE_THROUGH_ATTR) is True      # the span went to the store with the park


def test_a_chunked_prefill_of_another_lane_stays_the_chunked_request_and_is_not_merged(lanes_on):
    dec, hi_dec = _req(1, 0), _req(2, 1)
    fin = _req(4, 1)                                                # a request whose prefill ended in the same pass
    ch = _chunked(3, 1)
    s = _mid_prefill([dec, hi_dec], ch, finished=[fin])
    out = _park_lane(s, [dec.rid])
    assert out.success and out.parked == [dec.rid] and out.lane_skipped == {}
    assert s.chunked_req is ch and ch.extend_range.end == 4600     # untouched
    assert s.last_batch is None
    assert [r.rid for r in s.running_batch.reqs] == [hi_dec.rid, fin.rid]   # fin joined the decode set, ch did not
    assert all(r is not ch for r in s.running_batch.reqs)
    assert _next_pass_head(s) == "stash"                            # the pass head reads it fine, as before


def test_a_chunked_prefill_in_no_batch_is_released_directly_and_parked(lanes_on, monkeypatch):
    """The pass before the RPC ran a decode batch: ``last_batch`` carries no extend batch, the chunked request is
    in no batch at all."""
    ch = _chunked(2, 0)
    seen = []

    def fake_release(sched, req):
        seen.append(req.rid)
        req.extend_range = None
        return req

    monkeypatch.setattr(rt, "_release_outside_chunk", fake_release)
    s = _Sched(running=[_req(1, 1)])
    s.chunked_req = ch
    out = _park_lane(s, [ch.rid])
    assert out.success and out.parked == [ch.rid] and out.lane_skipped == {}
    assert seen == [ch.rid] and s.chunked_req is None and _next_pass_head(s) == "-"
    assert d_lane.lane_held(ch) and [r.rid for r in s.weg2_d_parked] == [ch.rid]
    assert [r.rid for r in s.running_batch.reqs] == ["weg2-1-1"]   # the higher lane's decode seat is untouched


def test_a_chunked_prefill_with_a_chunk_in_flight_is_named_and_stays(lanes_on, monkeypatch):
    ch = _chunked(2, 0)
    ch.inflight_middle_chunks = 1
    monkeypatch.setattr(rt, "_release_outside_chunk", lambda s, r: pytest.fail("must not release"))
    s = _Sched(running=[])
    s.chunked_req = ch
    out = _park_lane(s, [ch.rid])
    assert out.success and out.parked == [] and set(out.lane_skipped) == {ch.rid}
    assert s.chunked_req is ch and ch.extend_range.end == 4600


def test_every_tp_rank_parks_the_same_request_the_same_way(lanes_on):
    """Ranks never disagree: the decision reads replicated state only (rids, chunked_req.rid, in-flight count)."""
    def one_rank():
        dec, ch = _req(1, 0), _chunked(2, 0)
        s = _mid_prefill([dec, _req(3, 1)], ch, waiting=[_req(5, 0)])
        out = _park_lane(s, [dec.rid, ch.rid, "weg2-1-5"])
        return (out.parked, out.held, out.lane_skipped, s.chunked_req is None,
                [r.rid for r in s.weg2_d_parked], [r.rid for r in s.running_batch.reqs],
                list(s.running_batch.released), [r.rid for r in s.waiting_queue])

    ranks = [one_rank() for _ in range(3)]
    assert ranks[0] == ranks[1] == ranks[2]
    assert ranks[0][0] == ["weg2-1-1", "weg2-1-2"] and ranks[0][3] is True


# ---- the mutants: the fix removed -> red with exactly the error of the metal log
def _mutant(old, new):
    src = open(rt.__file__).read()
    assert old in src, "mutation anchor moved"
    mod = types.ModuleType("d_park_runtime_mutant")
    mod.__file__ = rt.__file__
    mod.__package__ = rt.__package__
    exec(compile(src.replace(old, new, 1), rt.__file__, "exec"), mod.__dict__)  # noqa: S102 - test mutant
    return mod


def test_mutant_without_the_release_of_chunked_req_dies_with_the_metal_attribute_error(lanes_on):
    mod = _mutant("    if chunk_parked:\n        sched.chunked_req = None\n", "    if chunk_parked:\n        pass\n")
    from sglang.srt.managers.io_struct import Weg2ParkRunningReqInput

    ch = _chunked(2, 0)
    s = _mid_prefill([], ch)
    out = mod.park_running(s, Weg2ParkRunningReqInput(epoch=7, reason="lane", rids=[ch.rid], hold="lane"))
    assert out.parked == [ch.rid] and s.chunked_req is ch          # still pointing at the retracted request
    with pytest.raises(AttributeError, match="'NoneType' object has no attribute 'end'"):
        _next_pass_head(s)


def test_mutant_that_merges_another_lanes_chunked_request_decodes_it_unfinished(lanes_on):
    mod = _mutant("keep_chunked=None if (park_chunk or chunk is None) else chunk)", "keep_chunked=None)")
    from sglang.srt.managers.io_struct import Weg2ParkRunningReqInput

    dec, ch = _req(1, 0), _chunked(3, 1)
    s = _mid_prefill([dec], ch)
    mod.park_running(s, Weg2ParkRunningReqInput(epoch=7, reason="lane", rids=[dec.rid], hold="lane"))
    assert any(r is ch for r in s.running_batch.reqs) and s.chunked_req is ch   # in the decode set AND chunked


# ---- review round 1 (major): the spec back-only refusal must not leave the chunk in the decode set
def _spec(s):
    s.running_batch.spec_algorithm = types.SimpleNamespace(is_none=lambda: False)
    return s


def test_spec_refusal_of_a_chunked_park_takes_the_chunk_out_of_the_decode_set_again(lanes_on):
    lo_dec, hi_dec = _req(1, 0), _req(2, 1)
    ch = _chunked(3, 0)
    s = _spec(_mid_prefill([lo_dec, hi_dec], ch))
    out = _park_lane(s, [lo_dec.rid, ch.rid])
    assert out.success is False and "not the back of the batch" in out.message
    assert not any(r is ch for r in s.running_batch.reqs)               # not decoding unfinished
    assert [r.rid for r in s.running_batch.reqs] == [lo_dec.rid, hi_dec.rid]
    assert s.chunked_req is ch and _next_pass_head(s) == "stash"        # still only chunked_req, consistent
    assert not out.parked and not s.running_batch.released               # nothing retracted


def test_spec_back_only_park_of_the_chunk_and_the_decode_tail_still_parks_both(lanes_on):
    hi_dec, lo_dec = _req(2, 1), _req(1, 0)
    ch = _chunked(3, 0)
    s = _spec(_mid_prefill([hi_dec, lo_dec], ch))
    out = _park_lane(s, [lo_dec.rid, ch.rid])
    assert out.success and sorted(out.parked) == sorted([lo_dec.rid, ch.rid])
    assert s.chunked_req is None and [r.rid for r in s.running_batch.reqs] == [hi_dec.rid]


def test_every_tp_rank_refuses_the_spec_chunk_park_the_same_way(lanes_on):
    def one_rank():
        lo_dec, hi_dec = _req(1, 0), _req(2, 1)
        ch = _chunked(3, 0)
        s = _spec(_mid_prefill([lo_dec, hi_dec], ch))
        out = _park_lane(s, [lo_dec.rid, ch.rid])
        return (out.success, [r.rid for r in s.running_batch.reqs], s.chunked_req is ch)

    ranks = [one_rank() for _ in range(3)]
    assert ranks[0] == ranks[1] == ranks[2] and ranks[0][0] is False


def test_mutant_without_the_unmerge_leaves_the_chunk_in_both_sets_under_spec_refusal(lanes_on):
    mod = _mutant("        if park_chunk:\n            _unmerge_chunk(sched, chunk)  # nothing parked",
                  "        if False:\n            _unmerge_chunk(sched, chunk)  # nothing parked")
    from sglang.srt.managers.io_struct import Weg2ParkRunningReqInput

    lo_dec, hi_dec = _req(1, 0), _req(2, 1)
    ch = _chunked(3, 0)
    s = _spec(_mid_prefill([lo_dec, hi_dec], ch))
    out = mod.park_running(s, Weg2ParkRunningReqInput(epoch=7, reason="lane",
                                                      rids=[lo_dec.rid, ch.rid], hold="lane"))
    assert out.success is False
    assert any(r is ch for r in s.running_batch.reqs) and s.chunked_req is ch   # double membership


def test_the_flip_park_and_park_youngest_paths_are_untouched():
    """Default path: only ``park_rids`` and ``_land_inflight`` (new optional argument) changed; the flip park keeps its
    own landing block and ``chunked_req = None``, ``park_youngest`` its own (default-path, not a lane path)."""
    src = open(rt.__file__).read()
    i = src.index("def park_running(")
    j = src.index("def _land_inflight(")
    body = src[i:j]
    assert "sched.chunked_req = None" in body and "last.filter_batch(chunked_req_to_exclude=[])" in body


# ================================================================== B. the flip end is an event
@_on
async def test_flip_done_ends_the_deferral_at_once_and_the_marker_names_the_latency(caplog, monkeypatch):
    monkeypatch.setattr(LC, "TICK_S", 5.0)                          # a tick would take 5 s: only the event can be fast
    caplog.set_level(logging.INFO, logger="weg2.front")
    f = _front("D")
    f.state = "flipping"
    f._flip_dst = "P"
    f.groups["D"].outstanding["d0"] = NOW
    _note(f, "d0", 0)
    loop = asyncio.ensure_future(LC.loop(f))
    try:
        await asyncio.sleep(0.05)
        assert await _arrive(f, "hi", 1) is None                    # DEFER
        await asyncio.sleep(0.05)
        ls = f._lane_state()
        assert ls.lane_floor == 0 and len(_marks(caplog, LN.MARK_DEFER)) == 1
        f.state = "serving"                                         # WEG2-FLIP done ...
        t0 = time.time()
        LC.flip_done(f)                                             # ... and its event
        assert await _until(lambda: ls.lane_floor == 1, 1.0)
        assert time.time() - t0 < 1.0                               # not the 5-s tick
    finally:
        loop.cancel()
    raise_line = _marks(caplog, LC.MARK_RAISE)[0]
    assert raise_line.startswith("WEG2 LANE-RAISE floor=1 epoch=") and "deferred=1" in raise_line
    since = float(raise_line.split("since_flip_done_ms=")[1].split()[0])
    assert 0.0 <= since < 500.0
    pre = _marks(caplog, LN.MARK_PREEMPT)
    assert len(pre) == 1 and pre[0].startswith("WEG2 LANE-PREEMPT floor=1 epoch=")      # the old greps still count 1
    assert f"since_flip_done_ms={raise_line.split('since_flip_done_ms=')[1].split()[0]} " in pre[0]
    assert " ms=" in pre[0]
    # LANE-RAISE is written when the floor rises (before the RPCs), LANE-PREEMPT after them
    order = [r.getMessage()[:20] for r in caplog.records if r.getMessage().startswith(("WEG2 LANE-RAISE", LN.MARK_PREEMPT))]
    assert order[0].startswith("WEG2 LANE-RAISE") and order[1].startswith("WEG2 LANE-PREEMPT")


@_on
async def test_mutant_without_the_flip_done_kick_the_deferral_waits_for_the_tick(monkeypatch):
    monkeypatch.setattr(LC, "TICK_S", 5.0)
    f = _front("D")
    f.state = "flipping"
    f._flip_dst = "P"
    f.groups["D"].outstanding["d0"] = NOW
    _note(f, "d0", 0)
    loop = asyncio.ensure_future(LC.loop(f))
    try:
        await asyncio.sleep(0.05)
        assert await _arrive(f, "hi", 1) is None
        f.state = "serving"                                         # the event is lost (no flip_done / no kick)
        assert not await _until(lambda: f._lane_state().lane_floor == 1, 0.5)
    finally:
        loop.cancel()


def test_the_front_calls_flip_done_in_the_flip_done_block_after_the_state_is_serving():
    src = open(os.path.join(_SRT, "weg2", "front.py")).read()
    i = src.index('if not self._enter_state("serving"):\n            # W98-STOP-FINAL')
    j = src.index("_lctl.flip_done(self)", i)
    assert 0 < j - i < 12000 and "WEG2-FLIP done epoch=%d slept=%s woke=%s" in src[j:j + 40000]


@_on
async def test_the_defer_cancels_only_the_lower_lanes_early_legs_and_only_once():
    f = _front("P")
    f.state = "flipping"
    f._flip_dst = "P"
    lo = _pend(f, "lo0", 0, NOW)
    same = _pend(f, "hi0", 1, NOW + 1)
    plain = _pend(f, "lo1", 0, NOW + 2)
    f.queue.extend([lo, same, plain])
    for p in (lo, same):
        p._leg1_early = asyncio.ensure_future(_early_leg(f, p.rid))
        Front._leg1_early_watch(f, p)
    await asyncio.sleep(0.02)
    assert await _arrive(f, "hi", 1) is None                        # DEFER (target 1)
    await LC.reconcile(f, "tick")
    await LC.reconcile(f, "tick")                                   # still flipping: idempotent
    await asyncio.sleep(0.02)
    assert [(g, b["rid"]) for g, b in f.rpc.paths("/abort_request")] == [("P", "lo0")]
    assert same._leg1_early is not None and not same._leg1_early.done()    # lane 1 itself: its early leg flies on
    assert plain.lane_retake is False and f.counters["lane_early_taken"] == 1
    assert list(f.queue) == [lo, same, plain]                       # nothing held before the flip ends
    # nothing ran on P (it was asleep): the fresh leg finding nothing cached is NOT a LANE-REPREFILL
    assert lo.lane_p_ran_s == 0.0 and LC.retake_check(f, lo, 16000, 0) is False
    assert f.counters["lane_reprefill"] == 0


@_on
async def test_a_deferral_whose_lane_ended_during_the_flip_leaves_no_stale_latency_for_the_next_preempt(caplog):
    caplog.set_level(logging.INFO, logger="weg2.front")
    f = _front("D")
    f.state = "flipping"
    f._flip_dst = "P"
    assert await _arrive(f, "hi", 1) is None                        # DEFER
    assert f._lane_ctl_obj.defer_logged is not None and f._lane_ctl_obj.defer_t0 is not None
    f._lane_end("hi")                                               # the lane ends before the flip does
    await LC.reconcile(f, "end")
    assert f._lane_ctl_obj.defer_logged is None and f._lane_ctl_obj.defer_t0 is None
    f.state = "serving"
    LC.flip_done(f)
    assert await _arrive(f, "hi2", 1) is None                       # a fresh arrival while serving: not deferred
    line = _marks(caplog, LC.MARK_RAISE)[0]
    assert "deferred=0" in line and "since_flip_done_ms=-" in line


# ================================================================== C. no take while a floor-lane arrival has no place
def _pool(f, queue, gates, done_order, with_wait=True):
    ls = f._lane_state()

    async def one(p):
        ev = gates.setdefault(p.rid, asyncio.Event())
        try:
            await ev.wait()
        except asyncio.CancelledError:
            if p.lane_p_taken:
                return p
            raise
        return p

    def on_done(p):
        if not p.lane_p_taken:
            done_order.append(p.rid)

    return asyncio.ensure_future(_p_drain_pool(
        queue, 2, one, on_done, lambda: True, lane_held=lambda p: p.lane < ls.lane_floor,
        lane_take=lambda its, cancel: LC.p_take(f, its, cancel),
        lane_take_wait=(lambda: LC.take_wait(f)) if with_wait else None))


@_on
async def test_the_pool_does_not_take_the_held_leg_while_a_floor_lane_arrival_has_no_place():
    """Probe 2b: lane 1 (16 000 tokens, P-bound) arrives while the 98 000-token lane-0 leg runs on P.  The floor
    rises in the arrival's reconcile; the pool then sees only the held leg.  It must wait for the arrival's own leg."""
    f = _front("P")
    queue, gates, done = collections.deque(), {}, []
    lo = _pend(f, "lo0", 0, NOW)
    queue.append(lo)
    task = _pool(f, queue, gates, done)
    await asyncio.sleep(0.05)
    assert "lo0" in gates
    ls = f._lane_state()
    _note(f, "hi", 1)
    LC.ctl(f).arriving["hi"] = time.time()                          # arrive(): registered, reconcile running
    ls.set_floor(1)                                                 # ... the floor rose
    await asyncio.sleep(0.4)                                        # several 0.1-s rounds
    assert f.rpc.paths("/abort_request") == [] and f.counters["lane_p_taken"] == 0 and not task.done()
    hi = _pend(f, "hi", 1, NOW + 5)
    LC.stamp(f, hi)                                                 # its Pending exists ...
    queue.append(hi)                                                # ... and is queued
    await asyncio.sleep(0.25)
    assert "hi" in gates and f.rpc.paths("/abort_request") == []    # dispatched beside the held leg, no take yet
    gates["hi"].set()
    assert await asyncio.wait_for(task, 3) >= 1                     # only the held leg left now: taken, drain ends
    assert done == ["hi"] and [b["rid"] for _g, b in f.rpc.paths("/abort_request")] == ["lo0"]
    assert f.counters["lane_p_taken"] == 1


@_on
async def test_mutant_without_lane_take_wait_the_held_leg_is_taken_the_moment_the_floor_rises():
    f = _front("P")
    queue, gates, done = collections.deque(), {}, []
    queue.append(_pend(f, "lo0", 0, NOW))
    task = _pool(f, queue, gates, done, with_wait=False)
    await asyncio.sleep(0.05)
    _note(f, "hi", 1)
    LC.ctl(f).arriving["hi"] = time.time()
    f._lane_state().set_floor(1)
    await asyncio.sleep(0.4)
    assert [b["rid"] for _g, b in f.rpc.paths("/abort_request")] == ["lo0"]       # the probe-2b take
    task.cancel()


@_on
async def test_take_wait_reads_the_arrival_register_and_nothing_else():
    f = _front("P")
    ls = f._lane_state()
    assert LC.take_wait(f) is False                                 # no register
    lc = LC.ctl(f)
    _note(f, "hi", 1)
    lc.arriving["hi"] = NOW
    assert LC.take_wait(f) is True                                  # lane 1 >= floor 0
    ls.set_floor(1)
    assert LC.take_wait(f) is True                                  # lane 1 >= floor 1
    f.groups["P"].outstanding["hi"] = NOW
    assert LC.take_wait(f) is False                                 # it has a place (its leg stands on P)
    del f.groups["P"].outstanding["hi"]
    f.groups["D"].outstanding["hi"] = NOW
    assert LC.take_wait(f) is False                                 # ... or on D
    del f.groups["D"].outstanding["hi"]
    _note(f, "lo", 0)
    lc.arriving.clear()
    lc.arriving["lo"] = NOW
    assert LC.take_wait(f) is False                                 # a lane below the floor waits at the gate: not floor work
    lc.arriving.clear()
    lc.arriving["hi"] = NOW
    lc.forget("hi")                                                 # the handler ended before it had a place
    assert LC.take_wait(f) is False and "hi" not in lc.arriving


@_on
async def test_arrive_registers_until_the_pending_is_stamped_and_holds_the_take_meanwhile():
    f = _front("P")
    gate = asyncio.Event()
    calls = []

    async def slow_rpc(g, path, body, timeout):                     # P answers the floor RPC after its chunk
        calls.append((g.name, path))
        if path == LN.RPC_LANE_FLOOR and g.name == "P":
            await gate.wait()
        return 200, "{}"
    f.rpc = slow_rpc
    t = asyncio.ensure_future(_arrive(f, "hi", 1))
    assert await _until(lambda: ("P", LN.RPC_LANE_FLOOR) in calls, 2)
    assert f._lane_state().lane_floor == 1 and "hi" in LC.ctl(f).arriving
    assert LC.take_wait(f) is True                                  # the preempt is still waiting for P
    gate.set()
    assert await asyncio.wait_for(t, 2) is None
    assert LC.take_wait(f) is True                                  # routed on, but no Pending yet
    p = _pend(f, "hi", 1, NOW)
    LC.stamp(f, p)
    assert LC.take_wait(f) is False and "hi" not in LC.ctl(f).arriving


def test_the_switch_off_pool_call_passes_no_wait_and_the_default_pool_is_unchanged():
    src = open(os.path.join(_SRT, "weg2", "front.py")).read()
    assert "lane_take_wait=((lambda: _lctl.take_wait(self)) if _lanes.enabled() else None)," in src

    async def run():
        queue = collections.deque()
        seen = []

        class P:
            def __init__(self, rid):
                self.rid = rid
        queue.extend([P("a"), P("b")])

        async def one(p):
            await asyncio.sleep(0.01)
            return p
        await _p_drain_pool(queue, 2, one, lambda p: seen.append(p.rid), lambda: True)
        return seen

    assert asyncio.run(run()) == ["a", "b"]
