# SPDX-License-Identifier: Apache-2.0
"""#1470 POP KEEPS TWIN (27B NVFP4 dual B9b, boot ...fs10041535_df1fa730cf, P PP1 death 15:45:30Z, rid
weg2-0-89; desk/done/1470-b9b-tod-row-defer.md).

Chain: front P-PAUSE aborts rid 89, PP1 holds the abort (#1180-W), the front RESUMES the rid 550 ms later
(instance 2), instance 2 + its told reach PP1 in the pass whose plan reaches ``misses=3`` (frames of another
rid) -> verdict 'pop' -> AbortReq(rid) matches by prefix and takes instance 2 and its told along -> PP0's
frame for instance 2 names a rid PP1 cannot locate -> PpRowDeferCapExceeded. With
SGLANG_WEG2_DUAL_POP_KEEPS_TWIN=1 only the held objects leave. Default OFF = old behaviour.
"""
from __future__ import annotations

import os
import types
import unittest.mock as mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from sglang.srt.managers import scheduler as SC  # noqa: E402
from sglang.srt.managers import weg2_store_told as ST  # noqa: E402
from sglang.srt.weg2 import dual_untold_abort as DU  # noqa: E402
from sglang.srt.weg2 import p_row_authority  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

RID = "weg2-0-89"
OTHER = "weg2-0-82"
ENV_ON = "SGLANG_WEG2_DUAL_POP_KEEPS_TWIN"
Q698 = "SGLANG_WEG2_DUAL_LATER_TOLD_DROP"
DUAL_P_ENV = {"SGLANG_WEG2_DUAL_LAYOUT": "1", "SGLANG_WEG2_GROUP": "P",
              "SGLANG_WEG2_DUAL_P_KV_MAX_TOKENS": "196608"}
ON = dict(DUAL_P_ENV, **{ENV_ON: "1"})
OFF_ENVS = [
    {"SGLANG_WEG2_DUAL_LAYOUT": "", "SGLANG_WEG2_GROUP": "P", "SGLANG_WEG2_DUAL_P_KV_MAX_TOKENS": "196608", ENV_ON: "1"},
    {"SGLANG_WEG2_DUAL_LAYOUT": "1", "SGLANG_WEG2_GROUP": "D", "SGLANG_WEG2_DUAL_P_KV_MAX_TOKENS": "196608", ENV_ON: "1"},
    {"SGLANG_WEG2_DUAL_LAYOUT": "", "SGLANG_WEG2_GROUP": "", "SGLANG_WEG2_DUAL_P_KV_MAX_TOKENS": "", ENV_ON: "1"},
]


class _Tree:
    def release_aborted_request(self, rid):
        pass


class _Follower:
    """A follower of the dual P group (pp_size 3): absorb + abort dispatch + the per-pass hold verdict."""

    def __init__(self, pp_rank=1):
        self.ps = types.SimpleNamespace(pp_rank=pp_rank, pp_size=3)
        self.waiting_queue = []
        self.chunked_req = None
        self.tree_cache = _Tree()
        self._weg2_store_told_armed = True
        self._weg2_store_told = {}
        self._weg2_store_held = {}
        self.aborted = []
        self.frame = {OTHER: 5120}   # PP0's forwarded schedule of the pass: names ANOTHER rid, never the held one

    def _pp_scheduled_extents(self):
        return self.frame

    def _abort_request_now(self, recv_req):
        if SC.Scheduler._weg2_defer_waiting_abort(self, recv_req):
            return
        left = [r for r in self.waiting_queue if r.rid.startswith(recv_req.rid)]
        if left:
            self.aborted.append(recv_req.rid)
        self.waiting_queue = [r for r in self.waiting_queue if not r.rid.startswith(recv_req.rid)]
        for r in left:
            ST.forget_left_queue(self, r, "abort")

    def dispatch(self, recv_reqs):
        rest = ST.follower_absorb(self, recv_reqs)
        for item in rest:
            if isinstance(item, SC.AbortReq):
                self._abort_request_now(item)
        return rest

    def plan(self):
        SC.Scheduler._weg2_process_waiting_aborts(self)


def _req(rid=RID):
    return types.SimpleNamespace(rid=rid, req_pool_idx=None, is_retracted=False)


def _admit(told=0):
    return ST.Weg2StoreAdmit(rid=RID, told=told)


def _filler():
    return types.SimpleNamespace(rid="weg2-0-7")


@pytest.fixture
def row_auth(monkeypatch):
    monkeypatch.setattr(p_row_authority, "applies", lambda s: True)


def _held(pp_rank=1):
    """Pass 0: PP1 holds the abort of instance 1 (a told of an earlier list stands); the plan of the same
    pass counts miss 1 (frame of rid 82)."""
    f = _Follower(pp_rank)
    inst1 = _req()
    f.waiting_queue = [inst1]
    f._weg2_store_told[RID] = 0
    f.dispatch([SC.AbortReq(rid=RID)])
    assert f.aborted == [] and sorted(f._weg2_pending_waiting_aborts) == [RID]
    f.plan()
    return f, inst1


def _b9b(f):
    """Pass A (miss 2); pass B: instance 2 arrives with its told, the plan reaches miss 3 = 'pop'."""
    f.dispatch([_filler()])
    f.plan()
    inst2 = _req()
    f.waiting_queue.append(inst2)
    f._weg2_store_told.pop(RID, None)
    f.dispatch([_admit(told=0)])
    f.plan()
    return inst2


def test_b9b_pop_keeps_instance_2_and_its_told(row_auth, caplog):
    import logging
    with mock.patch.dict(os.environ, ON):
        f, inst1 = _held()
        with caplog.at_level(logging.WARNING, logger=DU.logger.name):
            inst2 = _b9b(f)
        assert [r for r in f.waiting_queue if r is inst1] == []
        assert [r for r in f.waiting_queue if r is inst2] == [inst2]     # PP0's frame for instance 2 finds it
        assert RID in f._weg2_store_told                                 # its told was not forgotten
        assert f.aborted == []                                           # no AbortReq(rid) (prefix: hits both)
        assert not getattr(f, "_weg2_pending_waiting_aborts", None)
        assert not getattr(f, DU.HOLDS_ATTR, None)
        assert any(DU.MARK_POP_TWIN in r.getMessage() and RID in r.getMessage() for r in caplog.records)


def test_default_off_is_unchanged_old_pop_takes_both():
    """No env: the old rid-wide pop (the B9b death shape) byte for byte, no record written."""
    with mock.patch.dict(os.environ, DUAL_P_ENV), mock.patch.object(p_row_authority, "applies", lambda s: True):
        os.environ.pop(ENV_ON, None)
        os.environ.pop(Q698, None)
        f, inst1 = _held()
        inst2 = _b9b(f)
        assert f.waiting_queue == []                     # instance 2 gone with instance 1: the bug, unchanged
        assert f.aborted == [RID]
        assert RID not in f._weg2_store_told             # and its told forgotten (Q-580)
        assert not hasattr(f, DU.HOLDS_ATTR)


def test_mutant_env_ignored_would_spare_twin_when_off():
    """MUTANT: if the env gate were ignored the OFF scenario would diverge -- the default-off test is sensitive."""
    with mock.patch.dict(os.environ, DUAL_P_ENV), mock.patch.object(p_row_authority, "applies", lambda s: True):
        os.environ.pop(ENV_ON, None)
        with mock.patch.object(DU, "pop_keeps_twin_on", lambda env=None: True):
            f, inst1 = _held()
            inst2 = _b9b(f)
            assert [r for r in f.waiting_queue if r is inst2]   # diverges from OFF


def test_mutant_rid_wide_removal_would_lose_instance_2(row_auth):
    """MUTANT: a rid-wide removal (what AbortReq(rid) does) loses instance 2 -- the object test is sensitive."""
    def by_rid(rec, wq):
        return [r for r in wq if str(r.rid) == rec["objs"][0].rid]
    with mock.patch.dict(os.environ, ON), mock.patch.object(DU, "_zombies_of", by_rid):
        f, inst1 = _held()
        inst2 = _b9b(f)
        assert not [r for r in f.waiting_queue if r is inst2]   # twins list is empty -> old path -> both gone


def test_no_twin_keeps_old_path(row_auth):
    """No newer instance queued: nobody to spare, the old AbortReq(rid) pop (and told forget) runs."""
    with mock.patch.dict(os.environ, ON):
        f, inst1 = _held()
        f.dispatch([_filler()])
        f.plan()
        f.dispatch([_filler()])
        f.plan()
        assert f.waiting_queue == [] and f.aborted == [RID]
        assert not getattr(f, DU.HOLDS_ATTR, None)


def test_instance_2_arriving_after_the_pop_same_end_state_on_both_followers(row_auth):
    """Rank agreement: PP1 sees instance 2 before the pop (spared), PP2 one pass later (old path pop, no twin
    yet); both end with instance 2 queued and its told -- the population PP0 holds."""
    with mock.patch.dict(os.environ, ON):
        f1, i1 = _held(1)
        inst2_1 = _b9b(f1)
        f2, i2 = _held(2)
        f2.dispatch([_filler()])
        f2.plan()
        f2.dispatch([_filler()])
        f2.plan()                                   # miss 3, no twin yet: old pop
        assert f2.waiting_queue == []
        inst2_2 = _req()
        f2.waiting_queue.append(inst2_2)
        f2.dispatch([_admit(told=0)])
        end1 = ([r is inst2_1 for r in f1.waiting_queue], RID in f1._weg2_store_told)
        end2 = ([r is inst2_2 for r in f2.waiting_queue], RID in f2._weg2_store_told)
        assert end1 == end2 == ([True], True)


def test_same_decision_on_three_ranks_and_pp0_never_decides(row_auth):
    with mock.patch.dict(os.environ, ON):
        out = {}
        for rank in (1, 2):
            f, inst1 = _held(rank)
            inst2 = _b9b(f)
            out[rank] = ([r is inst2 for r in f.waiting_queue], sorted(getattr(f, "_weg2_pending_waiting_aborts", {})),
                         getattr(f, "_q1470_pop_n", 0))
        assert out[1] == out[2] == ([True], [], 1)
        p0 = _Follower(pp_rank=0)
        p0.waiting_queue = [_req()]
        DU.note_hold(p0, SC.AbortReq(rid=RID), p0.waiting_queue)
        assert not hasattr(p0, DU.HOLDS_ATTR)
        assert DU.settle_hold(p0, RID, "pop") is False


def test_q698_early_skip_then_pop_keeps_twin(row_auth, caplog):
    """Both envs on = the real B9b chain: Q-698 sees the told list with lists_since_hold=2 < 3 (skip=early),
    the same pass's pop then spares instance 2."""
    import logging
    with mock.patch.dict(os.environ, dict(ON, **{Q698: "1"})):
        f, inst1 = _held()
        with caplog.at_level(logging.WARNING, logger=DU.logger.name):
            inst2 = _b9b(f)
        assert any("skip=early" in r.getMessage() for r in caplog.records)
        assert [r for r in f.waiting_queue if r is inst2] == [inst2] and not [r for r in f.waiting_queue if r is inst1]
        assert RID in f._weg2_store_told and f.aborted == []


def test_admitted_zombie_takes_old_path(row_auth):
    """A zombie that already runs is a chunked request: the old path owns it."""
    with mock.patch.dict(os.environ, ON):
        f, inst1 = _held()
        inst1.req_pool_idx = 5
        _b9b(f)
        assert f.aborted == [RID]


def test_record_discarded_on_every_verdict(row_auth):
    """'gone' (not in the waiting queue any more): the hold record leaves with the hold."""
    with mock.patch.dict(os.environ, ON):
        f, inst1 = _held()
        f.waiting_queue = []
        f.plan()
        assert not getattr(f, DU.HOLDS_ATTR, None) and not f._weg2_pending_waiting_aborts


@pytest.mark.parametrize("env", OFF_ENVS)
def test_gate_flip_nf_int8_dual_d_never_spare(monkeypatch, env):
    """Flip / NF / 27B INT8 / dual D with the env ON: no record, old pop (both leave)."""
    monkeypatch.setattr(p_row_authority, "applies", lambda s: True)
    with mock.patch.dict(os.environ, env):
        f, inst1 = _held()
        _b9b(f)
        assert f.waiting_queue == [] and f.aborted == [RID]
        assert not hasattr(f, DU.HOLDS_ATTR)
