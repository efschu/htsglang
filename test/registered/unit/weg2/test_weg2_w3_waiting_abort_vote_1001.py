"""W3-WAITABORT (27B N3 #2, boot dkr27browauthoritybar1fs10011814, e9736b85c0).

18:18:13Z an H102 burst aborted weg2-0-3 / weg2-0-4 while both still sat in
the FOLLOWERS' waiting queues (#TW TWIN-DEFER). PP0 applied the aborts; PP1 and
PP2 held them under #1180-W ("WEG2-PP-WAITING-ABORT held ... applied when PP0's
forwarded schedule decides it"). PP0 then ran one short request and went idle:
no further frame, so the followers skipped the plan (PLAN BYPASS) and
``_weg2_process_waiting_aborts`` never ran again. The #791C idle-vote release
looked at ``_pending_chunked_abort_req`` only and returned. Result: 6643x
"WEG2-P-IDLE-VERDICT idle=False blocking_rank=1 blockers=[waiting_queue]",
every /flush_cache 400, W3 Weg2DrainWitnessDisagreement at 18:19:44Z.

The fix lets the same idle-PP0 vote release the waiting holds: with
``_791c_pp0_drained`` set, ``follower_waiting_abort_verdict`` answers ``pop``.
Driven here on the REAL hook and the REAL Scheduler methods
(process_pending_chunked_abort -> _weg2_process_waiting_aborts ->
_abort_request_now's #1180-W defer) on a stand-in follower."""
from __future__ import annotations

import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from sglang.srt.managers import scheduler as S  # noqa: E402
from sglang.srt.weg2 import p_row_authority  # noqa: E402

RIDS = ("weg2-0-3", "weg2-0-4")


class _Follower:
    def __init__(self, pp_rank=1):
        self.ps = types.SimpleNamespace(pp_rank=pp_rank, pp_size=3)
        self.waiting_queue = [types.SimpleNamespace(rid=r) for r in RIDS]
        self.chunked_req = None
        self._pending_chunked_abort_req = None
        self._pending_chunked_abort_delay = 0
        self._791c_pp0_drained = False
        self.frame = None                  # frameless: PP0 idle, PLAN BYPASS
        self.aborted = []
        self.drained = True

    def _pp_scheduled_extents(self):
        return self.frame

    def _pp_microbatches_drained(self):
        return self.drained

    def _abort_request_now(self, recv_req):
        if S.Scheduler._weg2_defer_waiting_abort(self, recv_req):
            return
        self.aborted.append(recv_req.rid)
        self.waiting_queue = [r for r in self.waiting_queue if r.rid != recv_req.rid]

    def process_pending_chunked_abort(self):
        return S.Scheduler.process_pending_chunked_abort(self)


for _m in ("_weg2_defer_waiting_abort", "_weg2_process_waiting_aborts"):
    setattr(_Follower, _m, getattr(S.Scheduler, _m))


def _vote(pp0_idle):
    from sglang.srt.managers.weg2_idle_vote import Weg2IdleVoteReq

    v = Weg2IdleVoteReq(epoch=55, origin=0, world=3)
    v.slots.append((0, 1 if pp0_idle else 0, "none" if pp0_idle else "chunked_req"))
    return v


def _held(monkeypatch, pp_rank=1, row_authority=True):
    monkeypatch.setattr(p_row_authority, "applies", lambda s: row_authority)
    f = _Follower(pp_rank)
    for rid in RIDS:                       # the H102 burst: AbortReq at receipt
        f._abort_request_now(S.AbortReq(rid=rid))
    return f


def _release(f, vote):
    from sglang.srt.managers.scheduler_pp_mixin import weg2_791c_release_on_idle_vote

    weg2_791c_release_on_idle_vote(f, vote)


def test_a_held_waiting_abort_is_released_by_an_idle_pp0_vote(monkeypatch):
    """(a) RED on d479845aa0: the hook returns at once (no chunked abort) and
    both rids stay in the waiting queue for good -- blockers=[waiting_queue]."""
    f = _held(monkeypatch)
    assert sorted(f._weg2_pending_waiting_aborts) == sorted(RIDS)   # #1180-W held them
    assert [r.rid for r in f.waiting_queue] == list(RIDS)
    _release(f, _vote(pp0_idle=True))
    assert f.waiting_queue == [], "the idle-PP0 vote must pop the held waiting aborts"
    assert sorted(f.aborted) == sorted(RIDS)
    assert not f._weg2_pending_waiting_aborts
    assert f._791c_pp0_drained is False


def test_the_hold_stays_without_an_idle_pp0_or_with_undrained_microbatches(monkeypatch):
    """(b) PP0 not idle (it may still send a frame naming the rid) or this
    follower still has microbatches in flight: keep, next lap."""
    f = _held(monkeypatch)
    _release(f, _vote(pp0_idle=False))
    assert [r.rid for r in f.waiting_queue] == list(RIDS) and not f.aborted
    f.drained = False
    _release(f, _vote(pp0_idle=True))
    assert [r.rid for r in f.waiting_queue] == list(RIDS) and not f.aborted
    f.drained = True
    _release(f, _vote(pp0_idle=True))
    assert f.waiting_queue == []


def test_pp0_and_the_pass_aligned_form_are_untouched(monkeypatch):
    f = _held(monkeypatch, pp_rank=0)       # PP0 pops at receipt, holds nothing
    assert f.waiting_queue == [] and not getattr(f, "_weg2_pending_waiting_aborts", None)
    f = _held(monkeypatch, pp_rank=2, row_authority=False)   # pass-aligned wire: applied at receipt
    assert f.waiting_queue == []


def test_the_chunked_abort_release_is_unchanged(monkeypatch):
    """(c) the #791C path: a follower with a held CHUNKED abort and no waiting
    hold is released exactly as before (process_pending_chunked_abort runs
    under _791c_pp0_drained)."""
    monkeypatch.setattr(p_row_authority, "applies", lambda s: True)
    f = _Follower(pp_rank=2)
    f.waiting_queue = []
    seen = []
    f._pending_chunked_abort_req = types.SimpleNamespace(rid="weg2-9-9")
    f.process_pending_chunked_abort = lambda: seen.append(f._791c_pp0_drained)
    _release(f, _vote(pp0_idle=True))
    assert seen == [True] and f._791c_pp0_drained is False
    seen.clear()
    f._pending_chunked_abort_req = None
    _release(f, _vote(pp0_idle=True))
    assert seen == [], "nothing pending: the hook does not run the abort pass"
