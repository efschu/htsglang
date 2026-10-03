# SPDX-License-Identifier: Apache-2.0
"""Q-693 ACK-ROOM HOLD + PP0 INTAKE STAMP (27B NVFP4 dual, P followers / PP0).

C. NO-ROOM (boot dkr27bnvfp4dual1mpsleepbar1fs10031727, bc2bd121c0; 37 'PF TOLD-ACK
   NO-ROOM' lines): PP1 17:42:58 weg2-0-183 told=68096 loadback_rows=68096 room=3735.
   The pool was full of the PREDECESSOR weg2-0-182, whose last chunk (extend=147)
   PP1 had admitted in the same second and which released its rows one pass later
   (full token usage 0.98 -> 0.51). The ack said 0, PP0 answered 'PF TOLD-FALLBACK
   ... told=68096 -> 0 reason=mismatch' and P re-prefilled 68664 tokens (~15 s).
   Fix: the ack is HELD while a predecessor in flight covers the shortfall and
   re-read every pump; 0 only on a stable shortage. The dual1k concurrent-prefill
   case (mem_cache/test_dual_told_room_dual1k_1001.py) still acks 0.
D. ``_weg2_told_intake_t`` (PP0's paced-intake stamp) survived Q-580's
   forget_left_queue; the next instance's setdefault inherited it, so its PF deadline
   (intake + TOTAL_S) lay in the past: 'reason=frist after<0.3s', a full re-prefill.

Every fix is dual-P only; each has a gate-off twin ("flip unchanged").
"""
from __future__ import annotations

import logging
import os
import types
import unittest.mock as mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from sglang.srt.managers import weg2_store_told as ST  # noqa: E402
from sglang.srt.managers import weg2_told_fallback as FB  # noqa: E402
from sglang.srt.managers import weg2_told_fidelity as TF  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

RID = "weg2-0-152"
DUAL_P_ENV = {"SGLANG_WEG2_DUAL_LAYOUT": "1", "SGLANG_WEG2_GROUP": "P",
              "SGLANG_WEG2_DUAL_P_KV_MAX_TOKENS": "196608"}
OFF_ENV = {"SGLANG_WEG2_DUAL_LAYOUT": "", "SGLANG_WEG2_GROUP": "P",
           "SGLANG_WEG2_DUAL_P_KV_MAX_TOKENS": ""}


@pytest.fixture
def dual_p():
    with mock.patch.dict(os.environ, DUAL_P_ENV):
        yield


@pytest.fixture
def gate_off():
    with mock.patch.dict(os.environ, OFF_ENV):
        yield


def _req(rid=RID, **kw):
    r = types.SimpleNamespace(rid=rid, req_pool_idx=None, is_retracted=False)
    for k, v in kw.items():
        setattr(r, k, v)
    return r


# --- C. ACK-ROOM HOLD ----------------------------------------------------------------------

TOLD = 68096
PRED_TOKENS = 86561


class _Alloc:
    def __init__(self, avail):
        self.avail = avail

    def available_size(self):
        return self.avail


class _Tree:
    def __init__(self, avail, evictable=0):
        self.token_to_kv_pool_allocator = _Alloc(avail)
        self.evictable = evictable

    def evictable_size(self):
        return self.evictable

    def check_prefetch_progress(self, rid):
        return True

    def completed_prefetch_tokens(self, rid):
        return TOLD


def _pred(fill=PRED_TOKENS, origin=PRED_TOKENS):
    return types.SimpleNamespace(rid="weg2-0-182", fill_ids=list(range(fill)), origin_input_ids=list(range(origin)))


def _sched(pred=None, *, avail=3551, chunked=None, serial=True):
    req = types.SimpleNamespace(rid=RID, _weg2_early_told=None)
    return types.SimpleNamespace(
        tree_cache=_Tree(avail), ps=types.SimpleNamespace(pp_rank=1, pp_size=3), waiting_queue=[req],
        mbs=[types.SimpleNamespace(reqs=[pred]) if pred is not None else None, None, None],
        running_mbs=[], running_batch=types.SimpleNamespace(reqs=[]), chunked_req=chunked,
        max_running_requests=1 if serial else 2), req


@pytest.fixture
def rows(monkeypatch):
    monkeypatch.setattr(FB, "_loadback_rows", lambda s, r, t: int(t))
    monkeypatch.setattr(TF, "pp0_admissible", lambda s, r, own: None)


def test_bc2bd121c0_predecessor_in_flight_holds_the_ack(dual_p, rows, caplog):
    """RED on 36b5a4d3e9 / f93a680f6b: ack 0 -> PF told=0 -> 68664-token re-prefill."""
    s, req = _sched(_pred())
    with caplog.at_level(logging.WARNING, logger=FB.logger.name):
        assert FB._room_own(s, req, RID, TOLD) is None
    assert any(FB.ROOM_HOLD_MARK in r.getMessage() and "predecessor_rows=%d" % PRED_TOKENS in r.getMessage()
               for r in caplog.records)
    s.mbs = [None, None, None]                                  # the predecessor finished
    s.tree_cache.token_to_kv_pool_allocator.avail = 3551 + PRED_TOKENS
    with caplog.at_level(logging.WARNING, logger=FB.logger.name):
        assert FB._room_own(s, req, RID, TOLD) == TOLD
    assert any("END" in r.getMessage() and "how=room" in r.getMessage() for r in caplog.records)


def test_dual1k_concurrent_hog_without_predecessor_still_acks_zero(dual_p, rows):
    s, req = _sched(None)
    assert FB._room_own(s, req, RID, TOLD) == 0


def test_concurrent_prefill_on_a_parallel_p_is_not_a_predecessor(dual_p, rows):
    """dual1k: a prefill with chunks still to admit, on a P running >1 request."""
    s, req = _sched(None, chunked=_pred(fill=20480, origin=61440), serial=False)
    assert FB._room_own(s, req, RID, TOLD) == 0


def test_serial_p_counts_the_chunked_predecessor(dual_p, rows):
    s, req = _sched(None, chunked=_pred(fill=PRED_TOKENS - 1024), serial=True)
    assert FB._room_own(s, req, RID, TOLD) is None


def test_a_predecessor_too_small_is_a_stable_shortage(dual_p, rows):
    s, req = _sched(_pred(fill=1000, origin=1000), avail=1000)
    assert FB._room_own(s, req, RID, TOLD) == 0


def test_an_aborted_request_is_never_held(dual_p, rows):
    s, req = _sched(_pred())
    s.waiting_queue = []
    assert FB._room_own(s, req, RID, TOLD) == 0


def test_flip_form_room_ack_unchanged(gate_off, rows):
    s, req = _sched(_pred())
    assert FB._room_own(s, req, RID, TOLD) == 0
    assert not hasattr(s, FB._ROOM_HOLD_ATTR)


class _Chan:
    def __init__(self):
        self.sent = []

    def pump(self):
        return True

    def send_nowait(self, ack):
        self.sent.append(list(ack.reads))
        return True


def _pump_setup(monkeypatch, pred):
    s, req = _sched(pred)
    ch = _Chan()
    monkeypatch.setattr(FB, "_channel", lambda sched: ch)
    st = FB._fstate(s, create=True)
    st.expect[RID] = TOLD
    st.registered[RID] = req
    return s, req, ch, st


def test_follower_pump_holds_then_acks_told(dual_p, rows, monkeypatch):
    """RED on 36b5a4d3e9 / f93a680f6b: the first pump sends (rid, 0)."""
    s, req, ch, st = _pump_setup(monkeypatch, _pred())
    FB.follower_pump(s)
    assert ch.sent == [] and RID in st.registered and st.expect[RID] == TOLD
    s.mbs = [None, None, None]
    s.tree_cache.token_to_kv_pool_allocator.avail = 3551 + PRED_TOKENS
    FB.follower_pump(s)
    assert ch.sent == [[(RID, TOLD)]] and RID not in st.registered


def test_pp0_verdict_ends_the_hold(dual_p, rows, monkeypatch, caplog):
    s, req, ch, st = _pump_setup(monkeypatch, _pred())
    FB.follower_pump(s)
    with caplog.at_level(logging.WARNING, logger=FB.logger.name):
        FB.follower_forget(s, RID)                    # PP0's Admit (Frist / fallback) arrived
    assert not getattr(s, FB._ROOM_HOLD_ATTR)
    assert any("how=verdict" in r.getMessage() for r in caplog.records)
    FB.follower_pump(s)
    assert ch.sent == []


def test_flip_form_pump_acks_zero_at_once(gate_off, rows, monkeypatch):
    s, req, ch, st = _pump_setup(monkeypatch, _pred())
    FB.follower_pump(s)
    assert ch.sent == [[(RID, 0)]] and RID not in st.registered


# --- D. PP0's paced intake stamp leaves with the request -----------------------------------


def _pp0(req, stamps):
    return types.SimpleNamespace(
        ps=types.SimpleNamespace(pp_rank=0, pp_size=3), _weg2_store_told_armed=True,
        _weg2_store_told={}, _weg2_store_held={RID: req}, _weg2_told_intake_t=dict(stamps),
        waiting_queue=[])


def test_forget_left_queue_drops_the_intake_stamp(dual_p, monkeypatch):
    """RED on 36b5a4d3e9 / f93a680f6b: the stamp survives; the next instance's
    setdefault inherits it and its PF deadline lies in the past (frist after<0.3s)."""
    monkeypatch.setattr(ST, "_return_untold_dual_grant", lambda s, r, w: None)
    req = _req()
    s = _pp0(req, {RID: 1.0})
    dropped = ST.forget_left_queue(s, req, "abort")
    assert "intake_t" in dropped and RID not in s._weg2_told_intake_t
    ST._pace_intake_t(s).setdefault(RID, 99.0)                  # instance 3's intake
    assert s._weg2_told_intake_t[RID] == 99.0


def test_another_queued_holder_keeps_its_stamp(dual_p, monkeypatch):
    monkeypatch.setattr(ST, "_return_untold_dual_grant", lambda s, r, w: None)
    gone, live = _req(), _req()
    s = _pp0(live, {RID: 5.0})
    s.waiting_queue = [live]
    ST.forget_left_queue(s, gone, "abort")
    assert s._weg2_told_intake_t[RID] == 5.0


def test_flip_form_intake_stamp_unchanged(gate_off, monkeypatch):
    monkeypatch.setattr(ST, "_return_untold_dual_grant", lambda s, r, w: None)
    req = _req()
    s = _pp0(req, {RID: 1.0})
    dropped = ST.forget_left_queue(s, req, "abort")
    assert "intake_t" not in dropped and s._weg2_told_intake_t[RID] == 1.0
