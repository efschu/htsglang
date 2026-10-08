# SPDX-License-Identifier: Apache-2.0
"""Item 120 (boot dkr27bnvfp4dual1mpsleepbar1fs10020527, bf1175a0dd, 02.10. 05:35-05:41Z).

The "cuMemCreate FAILED rc=2" line on D (05:35:45) was a handled GROW-SHORT (rolled
back, ledger reconciled); D lived until the operator teardown at 05:40:59, and the
W17 at 05:41:09 is the health check seeing the torn-down groups. What stalled the
boot was P:

05:36:21 the front's P-PAUSE aborted pdflip-0-72/-73. PP0 applied the aborts and
released its grant; PP1/PP2 still held pdflip-0-73 in their WAITING queue and kept
the abort under #1180-W ("PDFLIP-PP-WAITING-ABORT held ... applied when PP0's
forwarded schedule decides it"). PP0 sent no further frame, the followers skipped
the plan, and the dual layout's liveness release (``follower_release_aborted_chunk``,
dual20) looked at ``_pending_chunked_abort_req`` only: the 352321536 B / 528482304 B
grants stayed committed on two cards, the front's RESUME-WAIT ("resumes at all
zeros") waited 271 s and never saw zero.

Driven on the REAL release and the REAL Scheduler methods
(process_pending_chunked_abort -> _pdflip_process_waiting_aborts) on a stand-in follower.
"""
from __future__ import annotations

import os
import types
import unittest.mock as mock
import uuid

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from flliper.srt.managers import scheduler as SC  # noqa: E402
from flliper.srt.pdflip import dual_p_kv_stage as S  # noqa: E402
from flliper.srt.pdflip import p_row_authority  # noqa: E402

RID = "pdflip-0-73"


class _Follower:
    def __init__(self):
        self.ps = types.SimpleNamespace(pp_rank=1, pp_size=3)
        self.waiting_queue = [types.SimpleNamespace(rid=RID)]
        self.chunked_req = None
        self._pending_chunked_abort_req = None
        self._pending_chunked_abort_delay = 0
        self._791c_pp0_drained = False
        self.forward_ct = 10
        self.frame = None                  # frameless: PP0 idle, PLAN BYPASS
        self.aborted = []

    def _pp_scheduled_extents(self):
        return self.frame

    def _pp_microbatches_drained(self):
        return True

    def _abort_request_now(self, recv_req):
        if SC.Scheduler._pdflip_defer_waiting_abort(self, recv_req):
            return
        self.aborted.append(recv_req.rid)
        self.waiting_queue = [r for r in self.waiting_queue if r.rid != recv_req.rid]

    def process_pending_chunked_abort(self):
        return SC.Scheduler.process_pending_chunked_abort(self)


for _m in ("_pdflip_defer_waiting_abort", "_pdflip_process_waiting_aborts"):
    setattr(_Follower, _m, getattr(SC.Scheduler, _m))


@pytest.fixture
def dual_p(monkeypatch):
    tag = "t-%s" % uuid.uuid4().hex[:8]
    env = {"FLLIPER_PDFLIP_DUAL_LAYOUT": "1", "FLLIPER_PDFLIP_GROUP": "P", "FLLIPER_PDFLIP_DUAL_KV_TAG": tag}
    monkeypatch.setattr(p_row_authority, "applies", lambda s: True)
    with mock.patch.dict(os.environ, env):
        try:
            yield types.SimpleNamespace(ps=types.SimpleNamespace(pp_rank=0), forward_ct=10)
        finally:
            try:
                os.unlink(S._idle_marker(tag))
            except OSError:
                pass


def _held():
    f = _Follower()
    f._abort_request_now(SC.AbortReq(rid=RID))       # the front's P-PAUSE abort at receipt
    assert sorted(f._pdflip_pending_waiting_aborts) == [RID]
    assert [r.rid for r in f.waiting_queue] == [RID]
    return f


def test_a_held_waiting_abort_is_released_once_pp0_is_idle_after_it(dual_p):
    """RED on f77461846b: follower_release_aborted_chunk returns False (no chunked
    abort pending) and pdflip-0-73 stays in the waiting queue for good."""
    f = _held()
    assert S.follower_release_aborted_chunk(f, now=100.0) is False      # sees the held abort
    S.mark_pp0_idle(dual_p, now=99.0)                                   # idle BEFORE it: no
    assert S.follower_release_aborted_chunk(f, now=101.0) is False
    assert [r.rid for r in f.waiting_queue] == [RID]
    S.mark_pp0_idle(dual_p, now=102.0)                                  # PP0 idle after it
    assert S.follower_release_aborted_chunk(f, now=103.0) is True
    assert f.waiting_queue == [] and f.aborted == [RID]
    assert not f._pdflip_pending_waiting_aborts
    assert f._791c_pp0_drained is False


def test_the_ring_order_still_holds_for_a_waiting_hold(dual_p):
    """dual22: PP0 idle alone is not ordered with the ring -- a pass PP0 launched
    (fwd 12) that has not run here yet keeps the hold."""
    f = _held()
    S.follower_release_aborted_chunk(f, now=100.0)
    S.mark_pp0_idle(types.SimpleNamespace(ps=types.SimpleNamespace(pp_rank=0), forward_ct=12), now=102.0)
    assert S.follower_release_aborted_chunk(f, now=103.0) is False
    assert [r.rid for r in f.waiting_queue] == [RID]
    f.forward_ct = 12
    assert S.follower_release_aborted_chunk(f, now=104.0) is True
    assert f.waiting_queue == []


def test_nothing_held_nothing_pending_is_untouched(dual_p):
    f = _Follower()
    f.waiting_queue = []
    assert S.follower_release_aborted_chunk(f, now=100.0) is False
    assert f._dual_abort_seen is None
    S.mark_pp0_idle(dual_p, now=102.0)
    assert S.follower_release_aborted_chunk(f, now=103.0) is False


def test_a_new_hold_restarts_the_clock(dual_p):
    """An idle stamp older than a NEW hold proves nothing about that hold."""
    f = _held()
    S.follower_release_aborted_chunk(f, now=100.0)
    f.waiting_queue.append(types.SimpleNamespace(rid="pdflip-0-74"))
    f._abort_request_now(SC.AbortReq(rid="pdflip-0-74"))
    S.mark_pp0_idle(dual_p, now=100.5)
    assert S.follower_release_aborted_chunk(f, now=101.0) is False      # clock restarts at 101
    S.mark_pp0_idle(dual_p, now=102.0)
    assert S.follower_release_aborted_chunk(f, now=103.0) is True
    assert sorted(f.aborted) == ["pdflip-0-73", "pdflip-0-74"]


def test_pp0_and_off_the_dual_p_nothing(monkeypatch):
    monkeypatch.setattr(p_row_authority, "applies", lambda s: True)
    g = _Follower()
    g._pdflip_pending_waiting_aborts = {RID: [None, 0]}
    with mock.patch.dict(os.environ, {"FLLIPER_PDFLIP_DUAL_LAYOUT": "", "FLLIPER_PDFLIP_GROUP": "P"}):
        assert S.follower_release_aborted_chunk(g) is False
    g.ps.pp_rank = 0
    with mock.patch.dict(os.environ, {"FLLIPER_PDFLIP_DUAL_LAYOUT": "1", "FLLIPER_PDFLIP_GROUP": "P"}):
        assert S.follower_release_aborted_chunk(g) is False
