# SPDX-License-Identifier: Apache-2.0
"""Q-697 SAME-LIST TOLD ABORT (27B NVFP4 dual, boot
dkr27bnvfp4dual1mpsleepsharebar1fs10032157, image fda96a5338): P PP1 22:07:08Z
'PpRowDeferCapExceeded: #791T STORE-TOLD HOP OVERDUE ... weg2-0-6' -> #1223
DEBUG-HOLD -> front W17 after 6.5 min of serving. The full rid is weg2-0-63
('Q-695 #791T OVERDUE FULL ... rids=weg2-0-63@1024[chunked(end=1024,
abort_pending=False)+queued(told=-)]'); weg2-0-6 itself had finished at 22:01:21.

METAL CHAIN:
  22:06:08  PP0 r101 = [Tok 64, Abort 63, Abort 54, Abort 64]; pp0_publish decides the
            PF Frist of 63 first ('PF TOLD-FALLBACK ... told=94720 -> 0') and appends
            Weg2StoreAdmit(63, 0, fallback) to that list, then dispatches the abort:
            instance 1 of 63 is popped at receipt, never admitted.
  22:06:14  PP1 r152 (n=5: the same four + PP0's Admit). Absorb first ('PF TOLD-FALLBACK
            ABSORBED (n=2)'), then the abort finds a told -> Q-693 does not apply ->
            'WEG2-PP-WAITING-ABORT held rid=weg2-0-63' (#1180-W).
  22:06:29  PP1 admits instance 2 at [0,1024); slot 1 [1024,2048): #791T maps the rid to
            the queued zombie (told consumed) -> 4 laps -> death.

Every test below that needs the dual P layout has a gate-off twin.
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
from sglang.srt.managers import weg2_told_fallback as FB  # noqa: E402
from sglang.srt.weg2 import dual_untold_abort as DU  # noqa: E402
from sglang.srt.weg2 import p_intake  # noqa: E402
from sglang.srt.weg2 import p_row_authority  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

RID = "weg2-0-63"
DUAL_P_ENV = {"SGLANG_WEG2_DUAL_LAYOUT": "1", "SGLANG_WEG2_GROUP": "P",
              "SGLANG_WEG2_DUAL_P_KV_MAX_TOKENS": "196608"}
OFF_ENVS = [
    {"SGLANG_WEG2_DUAL_LAYOUT": "", "SGLANG_WEG2_GROUP": "P", "SGLANG_WEG2_DUAL_P_KV_MAX_TOKENS": "196608"},
    {"SGLANG_WEG2_DUAL_LAYOUT": "1", "SGLANG_WEG2_GROUP": "D", "SGLANG_WEG2_DUAL_P_KV_MAX_TOKENS": "196608"},
    {"SGLANG_WEG2_DUAL_LAYOUT": "", "SGLANG_WEG2_GROUP": "", "SGLANG_WEG2_DUAL_P_KV_MAX_TOKENS": ""},
]


class _Tree:
    def __init__(self):
        self.released = []

    def release_aborted_request(self, rid):
        self.released.append(rid)


class _Follower:
    """PP1 of the dual P group, the parts the absorb + the abort dispatch touch."""

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
        if left:                                         # an abort of a rid not queued here changes nothing
            self.aborted.append(recv_req.rid)
        self.waiting_queue = [r for r in self.waiting_queue if not r.rid.startswith(recv_req.rid)]
        for r in left:                                   # the abort path's Q-580 forget
            ST.forget_left_queue(self, r, "abort")

    def dispatch(self, recv_reqs):
        """The follower's pass: absorb PP0's told objects, then dispatch the rest."""
        rest = ST.follower_absorb(self, recv_reqs)
        for item in rest:
            if isinstance(item, SC.AbortReq):
                self._abort_request_now(item)
        return rest


def _req(rid=RID):
    return types.SimpleNamespace(rid=rid, req_pool_idx=None, is_retracted=False)


def _fallback_admit(rid=RID, told=0):
    a = ST.Weg2StoreAdmit(rid=rid, told=told)
    setattr(a, FB.WIRE_FALLBACK, 1)
    return a


def _r152():
    """PP1's list of 22:06:14: PP0's received four (the abort of 63 among them) plus
    the Admit PP0's pp0_publish appended in the same pass."""
    return [types.SimpleNamespace(rid="weg2-0-64"), SC.AbortReq(rid=RID), SC.AbortReq(rid="weg2-0-54"),
            SC.AbortReq(rid="weg2-0-64"), _fallback_admit()]


@pytest.fixture
def dual_p(monkeypatch):
    monkeypatch.setattr(p_row_authority, "applies", lambda s: True)
    with mock.patch.dict(os.environ, DUAL_P_ENV):
        yield


def _instance1_with_read_ahead():
    f = _Follower()
    f.waiting_queue = [_req()]
    ST._early(f)[RID] = 94720              # 22:06:01: the paced read-ahead reached PP1 (no verdict)
    assert p_intake.told_pending(f, f.waiting_queue[0])
    return f


def test_y9_same_list_admit_abort_is_applied_at_receipt(dual_p, caplog):
    """RED on fda96a5338: PP1 'WEG2-PP-WAITING-ABORT held rid=weg2-0-63' (#1180-W)."""
    f = _instance1_with_read_ahead()
    with caplog.at_level(logging.WARNING, logger=DU.logger.name):
        f.dispatch(_r152())
    assert f.aborted == [RID], f.aborted
    assert f.waiting_queue == []
    assert not getattr(f, "_weg2_pending_waiting_aborts", None)
    assert RID not in f._weg2_store_told                  # the fallback told left with it (Q-580)
    assert any(DU.MARK_SAME_LIST in r.getMessage() and RID in r.getMessage() for r in caplog.records)


def test_y9_no_zombie_meets_instance_2(dual_p):
    """The death condition: once instance 2 is admitted on PP1, no queued object under the
    rid whose told never comes (what the #791T probe maps the next frame's rid to)."""
    f = _instance1_with_read_ahead()
    f.dispatch(_r152())
    inst2 = _req()
    f.waiting_queue.append(inst2)                         # 22:06:19 #1037 instance=2
    f.dispatch([_fallback_admit()])                       # 22:06:29 its own told (PF fallback 0)
    assert f._weg2_store_told[RID] == 0
    f.waiting_queue.remove(inst2)                         # admitted at slot 0, told consumed
    f._weg2_store_told.pop(RID)
    f.chunked_req = inst2
    zombies = [r for r in f.waiting_queue if r.rid == RID and p_intake.told_pending(f, r)]
    assert zombies == []


def test_a_told_from_an_earlier_list_keeps_the_hold(dual_p):
    """PP0 admitted on that told in its earlier pass -- the frame may be in flight."""
    f = _instance1_with_read_ahead()
    f.dispatch([_fallback_admit()])                       # the Admit, one pass earlier
    f.dispatch([types.SimpleNamespace(rid="weg2-0-64"), SC.AbortReq(rid=RID)])
    assert f.aborted == []
    assert sorted(f._weg2_pending_waiting_aborts) == [RID]


def test_a_told_for_another_rid_in_the_list_keeps_the_hold(dual_p):
    f = _instance1_with_read_ahead()
    f.dispatch([_fallback_admit()])
    f.dispatch([SC.AbortReq(rid=RID), _fallback_admit(rid="weg2-0-70")])
    assert f.aborted == [] and sorted(f._weg2_pending_waiting_aborts) == [RID]


def test_a_paced_read_ahead_in_the_list_is_no_verdict(dual_p):
    """A read-ahead sets no told: the Q-693 rule (untold) answers, unchanged."""
    f = _Follower()
    f.waiting_queue = [_req()]
    ahead = ST.Weg2StoreTold(rid=RID, told=94720, paced=True)
    f.dispatch([SC.AbortReq(rid=RID), ahead])
    assert f.aborted == [RID]
    assert getattr(f, DU.LIST_ATTR) is None


def test_single_phase_told_in_the_same_list_counts(dual_p):
    f = _Follower()
    f.waiting_queue = [_req()]
    f.dispatch([SC.AbortReq(rid=RID), ST.Weg2StoreTold(rid=RID, told=0)])
    assert f.aborted == [RID]


def test_admitted_or_retracted_request_keeps_the_hold(dual_p):
    f = _instance1_with_read_ahead()
    f.waiting_queue[0].is_retracted = True
    f.dispatch(_r152())
    assert f.aborted == [] and sorted(f._weg2_pending_waiting_aborts) == [RID]


def test_the_record_is_per_list(dual_p):
    """A later list without the abort object never matches the earlier record."""
    f = _instance1_with_read_ahead()
    f.dispatch([_fallback_admit()])
    stale = SC.AbortReq(rid=RID)
    DU.note_list(f, [stale, _fallback_admit()])
    f.dispatch([types.SimpleNamespace(rid="x")])          # next pass: nothing noted
    assert getattr(f, DU.LIST_ATTR) is None
    assert not DU.told_in_same_list(f, stale, RID)


@pytest.mark.parametrize("env", OFF_ENVS)
def test_gate_off_holds_as_before(monkeypatch, env):
    """Flip / NF / 27B INT8 / dual D: the y9 list keeps the #1180-W hold, no record written."""
    monkeypatch.setattr(p_row_authority, "applies", lambda s: True)
    with mock.patch.dict(os.environ, env):
        f = _instance1_with_read_ahead()
        f.dispatch(_r152())
        assert f.aborted == []
        assert sorted(f._weg2_pending_waiting_aborts) == [RID]
        assert not hasattr(f, DU.LIST_ATTR)


def test_pp0_writes_no_record(dual_p):
    f = _Follower(pp_rank=0)
    DU.note_list(f, _r152())
    assert not hasattr(f, DU.LIST_ATTR)
