# SPDX-License-Identifier: Apache-2.0
"""Q-693 RID-REUSE (27B NVFP4 dual y8y, boot dkr27bnvfp4dual1mpsleepbar1fs10031814,
36b5a4d3e9): P PP1 18:25:45Z 'PpRowDeferCapExceeded: #791T STORE-TOLD HOP
OVERDUE ... pdflip-0-1' (the log truncates rids to 8 characters; the debug-hold
locals name pdflip-0-152) -> #1223 DEBUG-HOLD -> front W17.

METAL CHAIN (front + P log):
  18:25:43.27  DUAL P-PAUSE pdflip-0-152 (instance 1 had three chunks on P)
  18:25:43.89  DUAL RESUME-UNSTARVE reason=short wait_s=0.0 bypassed=0
               per=[(0,2306867200,0),(0,419430400,0),(0,629145600,0)] -- no other
               leg in flight: 'P committed' was the head's OWN instance 1
  18:25:44     instance 2 paused at once: PP0 GRANT-RETURN 'told never left',
               Q-580 TOLD-FORGET dropped=held; PP1 'WAITING-ABORT held' (#1180-W)
  18:25:44.29  RESUME-UNSTARVE again -> instance 3; PP0 frames name the rid,
               the #1180-W verdict 'keep' for ever; PP1 admits instance 3, then
               #791T maps the rid to the zombie instance 2 (told never comes).

THE FIXES (dual P / dual front only; every test below has a gate-off twin):
  A. front: RESUME-UNSTARVE only while ANOTHER leg 1 is in flight on P.
  B. follower: an abort of a waiting request whose told never reached this
     rank is applied at receipt (PP0 never admitted it) -- no zombie.
"""
from __future__ import annotations

import asyncio
import collections
import logging
import os
import tempfile
import time
import types
import unittest.mock as mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from flliper.srt.managers import scheduler as SC  # noqa: E402
from flliper.srt.pdflip import card_kv_ledger as K  # noqa: E402
from flliper.srt.pdflip import dual_parallel as DP  # noqa: E402
from flliper.srt.pdflip import dual_untold_abort as DU  # noqa: E402
from flliper.srt.pdflip import front as F  # noqa: E402
from flliper.srt.pdflip import p_intake  # noqa: E402
from flliper.srt.pdflip import p_row_authority  # noqa: E402
from flliper.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

RID = "pdflip-0-152"
MIB = 1 << 20
Y8Y_COMMITTED = (2306867200, 419430400, 629145600)
DUAL_P_ENV = {"FLLIPER_PDFLIP_DUAL_LAYOUT": "1", "FLLIPER_PDFLIP_GROUP": "P",
              "FLLIPER_PDFLIP_DUAL_P_KV_MAX_TOKENS": "196608"}
OFF_ENV = {"FLLIPER_PDFLIP_DUAL_LAYOUT": "", "FLLIPER_PDFLIP_GROUP": "P",
           "FLLIPER_PDFLIP_DUAL_P_KV_MAX_TOKENS": ""}


@pytest.fixture
def dual_p():
    with mock.patch.dict(os.environ, DUAL_P_ENV):
        yield


@pytest.fixture
def gate_off():
    with mock.patch.dict(os.environ, OFF_ENV):
        yield


# --- A. front: RESUME-UNSTARVE needs another leg in flight --------------------------------


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()
        asyncio.set_event_loop(asyncio.new_event_loop())


def _front(dual=True):
    f = F.Front(prefill="http://p", decode="http://d", awake="D" if dual else "P", tag="dual",
                store_dir="/tmp", prefill_sid=0, decode_sid=0, dc_reserve={}, w_s=45.0, weight_chunks=2,
                flip_min_work_tokens=1, dual_layout=dual)

    async def rpc(g, path, body, timeout):
        return 200, "{}"

    f.rpc = rpc
    return f


def _pending(rid, uncached, paused=0):
    fut = asyncio.get_event_loop().create_future()
    p = F.Pending(rid, "/generate", {}, "x", time.time(), fut, est_prompt=uncached, est_uncached=uncached)
    p.dual_paused_n = paused
    return p


def _y8y_ledgers():
    root = tempfile.mkdtemp(prefix="wkv693")
    paths = [os.path.join(root, "card%d" % i) for i in range(3)]
    for pth, c in zip(paths, Y8Y_COMMITTED):
        K.CardKvLedger(pth, "D").contribute(4000 * MIB, committed=0)
        K.CardKvLedger(pth, "P").contribute(0)
        K.CardKvLedger(pth, "P").request(c)
    return paths


def _y8y_front(dual=True):
    f = _front(dual)
    f.dual_kv_ledgers = _y8y_ledgers()
    f.queue = collections.deque([_pending(RID, 169, paused=1)])
    return f


def test_y8y_head_with_no_other_leg_in_flight_is_not_unstarved(caplog):
    """RED on 36b5a4d3e9 / f93a680f6b: RESUME-UNSTARVE reason=short wait_s=0.0."""
    async def run():
        f = _y8y_front()
        with caplog.at_level(logging.INFO, logger=F.logger.name):
            held = f._dual_resume_held()
        return f, held

    f, held = _run(run())
    assert held, "the y8y head went back to P on its own unreleased instance"
    assert f.counters.get("dual_resume_unstarve", 0) == 0
    assert f.counters.get("dual_resume_own_held", 0) == 1
    msgs = [r.getMessage() for r in caplog.records]
    assert any("%s rid=%s would-be=short" % (DP.OWN_HELD_MARK, RID) in m for m in msgs), msgs
    assert not any(DP.UNSTARVE_MARK in m for m in msgs), msgs


def test_y8y_head_resumes_at_all_zeros_once_the_stages_released():
    async def run():
        f = _y8y_front()
        r = [f._dual_resume_held()]
        for pth in f.dual_kv_ledgers:                       # PP0/PP1/PP2 applied the pause abort
            led = K.CardKvLedger(pth, "P")
            led.release(led.state().committed["P"])
        r.append(f._dual_resume_held())
        return f, r

    f, r = _run(run())
    assert r == [True, False]
    assert f.counters.get("dual_resume_unstarve", 0) == 0
    assert f._q693_own_held_rid is None


def test_another_leg_in_flight_keeps_q691(caplog):
    async def run():
        f = _y8y_front()
        f._dual_inflight["pdflip-0-153"] = _pending("pdflip-0-153", 600)
        with caplog.at_level(logging.WARNING, logger=F.logger.name):
            held = f._dual_resume_held()
        return f, held

    f, held = _run(run())
    assert not held
    assert f.counters["dual_resume_unstarve"] == 1
    assert any("%s rid=%s reason=short" % (DP.UNSTARVE_MARK, RID) in r.getMessage() for r in caplog.records)


def test_the_heads_own_rid_in_flight_is_not_another_leg():
    async def run():
        f = _y8y_front()
        f._dual_inflight[RID] = f.queue[0]
        return f._dual_resume_held()

    assert _run(run())


def test_pure_rule():
    rows = [(0, c, 0) for c in Y8Y_COMMITTED]
    kw = dict(head_uncached=169, short_limit=8192, wait_s=0.0, stale_s=10.0, bypassed=0)
    assert DP.resume_unstarve(rows, p_busy=False, **kw) is None
    assert DP.resume_unstarve(rows, p_busy=True, **kw) == DP.UNSTARVE_SHORT
    assert DP.resume_unstarve(rows, **kw) == DP.UNSTARVE_SHORT          # default: the Q-691 rule


def test_flip_form_front_unchanged():
    """No dual layout: the Q-691/Q-693 block is never entered."""
    async def run():
        f = _y8y_front(dual=False)
        return f, f._dual_resume_held()

    f, held = _run(run())
    assert held
    assert f.counters.get("dual_resume_unstarve", 0) == 0
    assert f.counters.get("dual_resume_own_held", 0) == 0


# --- B. follower: an untold waiting abort is applied at receipt ---------------------------


class _Follower:
    def __init__(self):
        self.ps = types.SimpleNamespace(pp_rank=1, pp_size=3)
        self.waiting_queue = []
        self.chunked_req = None
        self._pdflip_store_told_armed = True
        self._pdflip_store_told = {}
        self.aborted = []

    def _abort_request_now(self, recv_req):
        if SC.Scheduler._pdflip_defer_waiting_abort(self, recv_req):
            return
        self.aborted.append(recv_req.rid)
        self.waiting_queue = [r for r in self.waiting_queue if not r.rid.startswith(recv_req.rid)]


def _req(rid=RID, **kw):
    r = types.SimpleNamespace(rid=rid, req_pool_idx=None, is_retracted=False)
    for k, v in kw.items():
        setattr(r, k, v)
    return r


@pytest.fixture
def row_authority(monkeypatch):
    monkeypatch.setattr(p_row_authority, "applies", lambda s: True)


def test_y8y_untold_waiting_abort_is_applied_at_receipt(dual_p, row_authority, caplog):
    """RED on 36b5a4d3e9 / f93a680f6b: PP1 holds instance 2 (#1180-W)."""
    f = _Follower()
    inst2 = _req()
    f.waiting_queue = [inst2]
    with caplog.at_level(logging.WARNING, logger=DU.logger.name):
        f._abort_request_now(SC.AbortReq(rid=RID))
    assert f.aborted == [RID] and f.waiting_queue == []
    assert not getattr(f, "_pdflip_pending_waiting_aborts", None)
    assert any(DU.MARK in r.getMessage() and RID in r.getMessage() for r in caplog.records)


def test_y8y_no_zombie_meets_instance_3(dual_p, row_authority):
    """The death condition: after instance 3 is admitted, a queued object under the
    rid whose told never comes (what #791T maps the frame's rid to)."""
    f = _Follower()
    f.waiting_queue = [_req()]
    f._abort_request_now(SC.AbortReq(rid=RID))          # instance 2's pause abort
    inst3 = _req()
    f.waiting_queue.append(inst3)                       # instance 3 behind it on the chain
    f._pdflip_store_told[RID] = 0                         # its own told (PF fallback told=0)
    f.waiting_queue.remove(inst3)                       # admitted at slot 0, told consumed
    f._pdflip_store_told.pop(RID)
    f.chunked_req = inst3
    zombies = [r for r in f.waiting_queue if r.rid == RID and p_intake.told_pending(f, r)]
    assert zombies == []


def test_a_told_here_keeps_the_hold(dual_p, row_authority):
    """PP0 may have admitted it (its told preceded the abort on the wire)."""
    f = _Follower()
    f.waiting_queue = [_req()]
    f._pdflip_store_told[RID] = 58018
    f._abort_request_now(SC.AbortReq(rid=RID))
    assert f.aborted == [] and sorted(f._pdflip_pending_waiting_aborts) == [RID]


@pytest.mark.parametrize("kw", [dict(is_retracted=True), dict(pdflip_parked_span=4096), dict(req_pool_idx=3)])
def test_a_once_admitted_request_keeps_the_hold(dual_p, row_authority, kw):
    f = _Follower()
    f.waiting_queue = [_req(**kw)]
    f._abort_request_now(SC.AbortReq(rid=RID))
    assert f.aborted == [] and sorted(f._pdflip_pending_waiting_aborts) == [RID]


def test_flip_form_follower_unchanged(gate_off, row_authority):
    """Gate off: the #1180-W hold exactly as before."""
    f = _Follower()
    f.waiting_queue = [_req()]
    f._abort_request_now(SC.AbortReq(rid=RID))
    assert f.aborted == [] and sorted(f._pdflip_pending_waiting_aborts) == [RID]
    assert not hasattr(f, "_q693_at_receipt_n")
