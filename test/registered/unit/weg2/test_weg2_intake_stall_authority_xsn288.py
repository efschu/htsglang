"""weg2xsn288 (18.09.2026): PP2 named an intake stall 2 s after the wake --
its hold (270.9 s) had ridden across sleep, wake and the front's abort --
and dropped weg2-6-4 from ITS queue alone while PP0/PP1 admitted it; the
PP ring stood (PP0 on PP2's output, PP1 on PP0's proxy, PP2 on requests).

Now: only PP0 refuses (the front's /abort_request is the propagation path
to the followers), an abort forgets the rid's hold and reported mark, and
the wake resets the watch.
"""
from __future__ import annotations

import os
from http import HTTPStatus
from types import SimpleNamespace

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import intake_stall as st  # noqa: E402


def _obs(w, rid, now, **kw):
    kw.setdefault("need_tokens", 95476)
    kw.setdefault("rem_total_tokens", 23000)
    kw.setdefault("cur_rem_tokens", 23000)
    kw.setdefault("running_empty", True)
    return w.observe(rid=rid, now=now, **kw)


def test_forget_drops_the_hold_and_the_reported_mark_for_that_rid_only():
    w = st.IntakeStallWatch(hold_s=1.0)
    assert _obs(w, "weg2-6-4", 0.0, immediate=True)          # reported
    assert _obs(w, "weg2-6-4", 5.0, immediate=True) is None  # once per rid
    w.forget("weg2-6-4")
    assert _obs(w, "weg2-6-4", 300.0, immediate=True)        # refusable again next phase
    # a hold that began before the abort does not ride on
    w2 = st.IntakeStallWatch(hold_s=1.0)
    assert _obs(w2, "weg2-6-4", 0.0) is None                 # hold starts
    w2.forget("weg2-6-4")
    assert _obs(w2, "weg2-6-4", 270.9) is None               # hold restarts, not 270.9 s old
    assert "held_s=1.0" in (_obs(w2, "weg2-6-4", 271.9) or "")
    # other rids untouched; abort_all (None) forgets everything
    w3 = st.IntakeStallWatch(hold_s=1.0)
    assert _obs(w3, "a", 0.0, immediate=True) and _obs(w3, "b", 0.0, immediate=True)
    w3.forget("a")
    assert _obs(w3, "b", 1.0, immediate=True) is None and _obs(w3, "a", 1.0, immediate=True)
    w3.forget(None)
    assert _obs(w3, "b", 2.0, immediate=True)


def test_reset_is_a_new_phase():
    w = st.IntakeStallWatch(hold_s=1.0)
    assert _obs(w, "weg2-6-4", 0.0, immediate=True)
    w.reset()
    assert w._rid is None and w._reported == set()
    assert _obs(w, "weg2-6-4", 1.0, immediate=True)


def _scheduler_double(pp_rank, watch, sent):
    from sglang.srt.managers import scheduler as sch
    obj = sch.Scheduler.__new__(sch.Scheduler)
    obj.ps = SimpleNamespace(pp_rank=pp_rank)
    obj._weg2_intake_watch = watch
    req = SimpleNamespace(rid="weg2-6-4", full_untruncated_fill_ids=[0] * 10, prefix_indices=[])
    obj.waiting_queue = [req]
    obj.enable_hicache_storage = False
    obj.enable_hierarchical_cache = False
    obj.ipc_channels = SimpleNamespace(
        send_to_tokenizer=SimpleNamespace(send_output=lambda a, r: sent.append(a)))
    return obj, req


def test_only_pp0_refuses_a_follower_keeps_its_queue(monkeypatch):
    from sglang.srt.managers import scheduler as sch
    from sglang.srt.managers.corridor_guard import GROUP_ENV
    monkeypatch.setenv(GROUP_ENV, "P")
    for rank in (1, 2):
        w = st.IntakeStallWatch(hold_s=1.0)
        sent = []
        obj, req = _scheduler_double(rank, w, sent)
        sch.Scheduler._weg2_intake_stall_observe(obj, req, None, note="gate=admission-wedge", immediate=True)
        assert obj.waiting_queue == [req] and sent == [] and w.stalls == 0, rank
    w = st.IntakeStallWatch(hold_s=1.0)
    sent = []
    obj, req = _scheduler_double(0, w, sent)
    sch.Scheduler._weg2_intake_stall_observe(obj, req, None, note="gate=admission-wedge", immediate=True)
    assert obj.waiting_queue == [] and w.stalls == 1
    assert len(sent) == 1 and sent[0].finished_reason["status_code"] == HTTPStatus.SERVICE_UNAVAILABLE
    assert st.is_intake_stall(sent[0].finished_reason["message"])


def test_abort_forgets_and_wake_resets_source_ratchet():
    from sglang.srt.managers import scheduler as sch
    from sglang.srt.managers.scheduler_components import weight_updater as wu
    src = open(sch.__file__).read()
    i = src.index("def _abort_request_now(self, recv_req: AbortReq):")
    blk = src[i:i + 900]
    assert "_watch.forget(None if recv_req.abort_all else recv_req.rid)" in blk
    j = src.index("def _weg2_intake_stall_observe")
    assert 'getattr(_ps, "pp_rank", 0) or 0) != 0' in src[j:j + 2500]
    wsrc = open(wu.__file__).read()
    k = wsrc.index('"WEG2-DORMANT cleared: kv_cache resumed, admission seams admit"')
    # the wake block up to the store rescan (5890ca792d put the wedge-clock
    # restart between the anchor and the reset: a fixed 600-char window read
    # past neither)
    assert "_iw.reset()" in wsrc[k:wsrc.index("self._weg2_rescan_store_index()", k)]


def test_xsn291_a_request_larger_than_the_pool_is_named_too_large_and_the_front_refuses(monkeypatch):
    """xsn291: weg2-6-4 (133,837 tokens) against PP0's pool of 121,190 -- stalled
    twice, requeued twice, the fairness switch flipped D->P 200 ms after the
    wake, W35/W68, both groups dead. The refusal names it; the front does
    not requeue."""
    from sglang.srt.managers import scheduler as sch
    from sglang.srt.managers.corridor_guard import GROUP_ENV
    from sglang.srt.weg2 import front as fr
    monkeypatch.setenv(GROUP_ENV, "P")
    w = st.IntakeStallWatch(hold_s=1.0)
    sent = []
    obj, req = _scheduler_double(0, w, sent)
    obj.max_total_num_tokens = 121190
    req.full_untruncated_fill_ids = [0] * 133837
    sch.Scheduler._weg2_intake_stall_observe(obj, req, None)       # not immediate: too-large forces it
    assert len(sent) == 1
    msg = sent[0].finished_reason["message"]
    assert st.is_intake_stall(msg) and st.is_too_large(msg) and "pool_tokens=121190" in msg
    # a request that fits the empty pool is NOT too large (ordinary hold applies)
    w2 = st.IntakeStallWatch(hold_s=1.0)
    sent2 = []
    obj2, req2 = _scheduler_double(0, w2, sent2)
    obj2.max_total_num_tokens = 121190
    req2.full_untruncated_fill_ids = [0] * 97872
    sch.Scheduler._weg2_intake_stall_observe(obj2, req2, None)
    assert sent2 == []                                              # hold not yet elapsed
    # the front: too-large is refused, never requeued
    src = open(fr.__file__).read()
    i = src.index("if is_intake_stall(e) and not is_too_large(e):")
    assert "_requeue_intake_stalled(p, e)" in src[i:i + 200]
    assert "WEG2 P-INTAKE-TOO-LARGE" in src[i:i + 900]
    assert "p.fut.set_exception(e)" in src[i:i + 1400]
