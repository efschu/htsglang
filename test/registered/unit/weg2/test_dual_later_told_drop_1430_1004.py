# SPDX-License-Identifier: Apache-2.0
"""#1430q LATER-TOLD DROPS HELD ZOMBIE (27B NVFP4 dual B9, boot ...fs10041420_662e99d96c, P PP1 death
14:49:55Z, rid weg2-0-103; desk/done/1430-b9-tod-told-hop.md).

Chain: PP1 holds instance 1 (#1180-W, a told of an EARLIER list stands), instance 2 of the same rid
arrives, PP0's told verdict for instance 2 rides a LATER list; the zombie takes the told and the #791T
probe maps the rid to it -> PpRowDeferCapExceeded. With SGLANG_WEG2_DUAL_LATER_TOLD_DROP=1 the zombie is
taken out of the waiting queue object-exactly at that later list. Default OFF = old behaviour.
"""
from __future__ import annotations

import logging
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

RID = "weg2-0-103"
ENV_ON = "SGLANG_WEG2_DUAL_LATER_TOLD_DROP"
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
    """A follower of the dual P group (pp_size 3), the parts absorb + abort dispatch touch."""

    def __init__(self, pp_rank=1):
        self.ps = types.SimpleNamespace(pp_rank=pp_rank, pp_size=3)
        self.waiting_queue = []
        self.chunked_req = None
        self.tree_cache = _Tree()
        self._weg2_store_told_armed = True
        self._weg2_store_told = {}
        self._weg2_store_held = {}
        self.aborted = []

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


def _req(rid=RID):
    return types.SimpleNamespace(rid=rid, req_pool_idx=None, is_retracted=False)


def _admit(rid=RID, told=0):
    return ST.Weg2StoreAdmit(rid=rid, told=told)


def _filler():
    return types.SimpleNamespace(rid="weg2-0-7")


@pytest.fixture
def row_auth(monkeypatch):
    monkeypatch.setattr(p_row_authority, "applies", lambda s: True)


def _held_zombie(pp_rank=1):
    """14:46:59 told of instance 1 absorbed; 14:48:44 the abort finds that told -> #1180-W hold;
    14:48:40..46 instance 2 of the same rid joined the queue behind it."""
    f = _Follower(pp_rank)
    zombie = _req()
    f.waiting_queue = [zombie]
    f._weg2_store_told[RID] = 94720
    f.dispatch([SC.AbortReq(rid=RID)])
    assert f.aborted == [] and sorted(f._weg2_pending_waiting_aborts) == [RID]   # the hold, as before
    inst2 = _req()
    f.waiting_queue.append(inst2)
    return f, zombie, inst2


def _run_to_later_told(f, quiet_lists=2):
    for _ in range(quiet_lists):
        f.dispatch([_filler()])
    f.dispatch([_admit()])


def test_zombie_case_later_told_drops_zombie_keeps_instance_2(row_auth, caplog):
    with mock.patch.dict(os.environ, ON):
        f, zombie, inst2 = _held_zombie()
        with caplog.at_level(logging.WARNING, logger=DU.logger.name):
            _run_to_later_told(f)
        assert [r for r in f.waiting_queue if r is zombie] == []
        assert [r for r in f.waiting_queue if r is inst2] == [inst2]
        assert not getattr(f, "_weg2_pending_waiting_aborts", None)
        assert f._weg2_store_told[RID] == 0                    # the new instance's verdict was absorbed after
        assert f.aborted == []                                 # no AbortReq(rid) was applied (it hits both)
        assert any(DU.MARK_LATER in r.getMessage() and RID in r.getMessage() for r in caplog.records)


def test_default_off_is_unchanged():
    """No env: zombie stays, no record written (old behaviour byte for byte)."""
    with mock.patch.dict(os.environ, DUAL_P_ENV), mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop(ENV_ON, None)
        with mock.patch.object(p_row_authority, "applies", lambda s: True):
            f, zombie, inst2 = _held_zombie()
            _run_to_later_told(f)
            assert [r for r in f.waiting_queue if r is zombie] == [zombie]
            assert sorted(f._weg2_pending_waiting_aborts) == [RID]
            assert not hasattr(f, DU.HOLDS_ATTR)


def test_mutant_env_ignored_would_drop_when_off():
    """MUTANT: if the env gate were ignored the OFF scenario would drop -- the default-off test above is sensitive."""
    with mock.patch.dict(os.environ, DUAL_P_ENV), mock.patch.object(p_row_authority, "applies", lambda s: True):
        os.environ.pop(ENV_ON, None)
        with mock.patch.object(DU, "later_told_drop_on", lambda env=None: True):
            f, zombie, inst2 = _held_zombie()
            _run_to_later_told(f)
            assert [r for r in f.waiting_queue if r is zombie] == []     # the mutant diverges from OFF


def test_mutant_rid_match_would_take_instance_2_along(row_auth):
    """MUTANT: a rid-wide drop (what AbortReq(rid) does) removes instance 2 too -- the object test is sensitive."""
    def by_rid(rec, wq):
        return [r for r in wq if str(r.rid) == rec["objs"][0].rid]
    with mock.patch.dict(os.environ, ON), mock.patch.object(DU, "_zombies_of", by_rid):
        f, zombie, inst2 = _held_zombie()
        _run_to_later_told(f)
        assert [r for r in f.waiting_queue if r is inst2] == []          # instance 2 lost: the real code must not do this


def test_same_list_told_is_still_q697(row_auth):
    """The told rides the abort's own list: Q-697 applies at receipt, no hold, no record."""
    with mock.patch.dict(os.environ, ON):
        f = _Follower()
        f.waiting_queue = [_req()]
        f.dispatch([SC.AbortReq(rid=RID), _admit()])
        assert f.aborted == [RID]
        assert not getattr(f, DU.HOLDS_ATTR, None)


def test_too_early_keeps_hold(row_auth, caplog):
    """Fewer than pp_size PP0 lists since the hold: frames PP0 launched for the old instance may still be in flight."""
    with mock.patch.dict(os.environ, ON):
        f, zombie, inst2 = _held_zombie()
        with caplog.at_level(logging.WARNING, logger=DU.logger.name):
            _run_to_later_told(f, quiet_lists=0)               # B is the first list after the hold
        assert [r for r in f.waiting_queue if r is zombie] == [zombie]
        assert sorted(f._weg2_pending_waiting_aborts) == [RID]
        assert any("skip=early" in r.getMessage() for r in caplog.records)


def test_admitted_zombie_keeps_hold(row_auth):
    """A zombie that already runs is a chunked request: the #1180-W path owns it."""
    with mock.patch.dict(os.environ, ON):
        f, zombie, inst2 = _held_zombie()
        zombie.req_pool_idx = 5
        _run_to_later_told(f)
        assert [r for r in f.waiting_queue if r is zombie] == [zombie]
        assert sorted(f._weg2_pending_waiting_aborts) == [RID]


def test_other_rid_or_paced_readahead_is_no_trigger(row_auth):
    with mock.patch.dict(os.environ, ON):
        f, zombie, inst2 = _held_zombie()
        for _ in range(3):
            f.dispatch([_filler()])
        f.dispatch([_admit(rid="weg2-0-70"), ST.Weg2StoreTold(rid=RID, told=1, paced=True)])
        assert [r for r in f.waiting_queue if r is zombie] == [zombie]
        assert sorted(f._weg2_pending_waiting_aborts) == [RID]


def test_released_hold_leaves_no_drop(row_auth):
    """The hold was released elsewhere (verdict / idle vote): the record is discarded, nothing is dropped."""
    with mock.patch.dict(os.environ, ON):
        f, zombie, inst2 = _held_zombie()
        f._weg2_pending_waiting_aborts.pop(RID)
        _run_to_later_told(f)
        assert zombie in f.waiting_queue and inst2 in f.waiting_queue
        assert not getattr(f, DU.HOLDS_ATTR, None)


def test_rank_agreement_same_list_same_decision_on_three_ranks(row_auth):
    """PP0 / PP1 / PP2 see the same list sequence: both followers decide alike; PP0 never decides (no record)."""
    with mock.patch.dict(os.environ, ON):
        out = {}
        for rank in (1, 2):
            f, zombie, inst2 = _held_zombie(rank)
            _run_to_later_told(f)
            out[rank] = ([r is inst2 for r in f.waiting_queue], sorted(getattr(f, "_weg2_pending_waiting_aborts", {})),
                         getattr(f, "_q1430q_drop_n", 0))
        assert out[1] == out[2] == ([True], [], 1)
        p0 = _Follower(pp_rank=0)
        p0.waiting_queue = [_req()]
        DU.note_hold(p0, SC.AbortReq(rid=RID), p0.waiting_queue)
        assert not hasattr(p0, DU.HOLDS_ATTR)
        DU.note_list(p0, [_admit()])
        assert len(p0.waiting_queue) == 1


@pytest.mark.parametrize("env", OFF_ENVS)
def test_gate_flip_nf_int8_dual_d_never_drop(monkeypatch, env):
    """Flip / NF / 27B INT8 / dual D with the env ON: no record, no drop (the hold is whatever it was)."""
    monkeypatch.setattr(p_row_authority, "applies", lambda s: True)
    with mock.patch.dict(os.environ, env):
        f = _Follower()
        zombie = _req()
        f.waiting_queue = [zombie]
        f._weg2_store_told[RID] = 94720
        f.dispatch([SC.AbortReq(rid=RID)])
        f.waiting_queue.append(_req())
        _run_to_later_told(f)
        assert zombie in f.waiting_queue and len(f.waiting_queue) == 2
        assert not hasattr(f, DU.HOLDS_ATTR)
