# SPDX-License-Identifier: Apache-2.0
"""Q-695 DUAL OLD-INSTANCE CHUNK: the y8z PP1 death replayed in its exact metal order.

Boot dkr27bnvfp4dual1mpsleepbar1fs10031909 (image ceff4aae7b), P log
/spinning/docker-acceptance/27b/evidence/
boot_weg2_dkr27bnvfp4dual1mpsleepbar1fs10031909_ceff4aae7b_1003_190930.P.log 70321-70510,
debug hold rc12gbf10031907_rank1_pid805_port5001_20261003T192606Z.txt:

  19:26:05.24  PP0/PP1/PP2 '#801 ABORT RECEIVED: rid=weg2-0-235' (DUAL P-PAUSE, 2nd);
               PP0 'applied in 2 pass(es)', PP1 'applied when PP0's forwarded schedule
               stops naming it (#791C)' -> instance 2 stays PP1's chunked request
  19:26:05     PP1 runs 1559 [4096,5120) of instance 2 (extend_range.end -> 5120)
  19:26:05.9   PP1 '#1037 REQ RE-CONSTRUCTED rid=weg2-0-2 instance=3' (front RESUME-UNSTARVE)
  19:26:05.98  PP1 probes frame 1560, stamp (1, 1560, 1024, -1, 1560, ('weg2-0-235', 5120,
               6144)), row ('weg2-0-235', 5120, 1024, True, ...): '#791T ROW-PROBE DEFER'
               x4 -> PpRowDeferCapExceeded '#791T STORE-TOLD HOP OVERDUE' (locals:
               _told_missing=['weg2-0-235'], _known={'weg2-0-235','weg2-0-236','weg2-0-237'})

RED ON THE PARENT ceff4aae7b: the probe defers the frame of the old instance's own chunk on
the new instance's missing told and raises on lap 4; the old chunk's abort cuts the new
instance's store read (release_aborted_request by rid); a schedule naming the new instance
keeps the dead old chunk. GREEN: the frame is planned, the new read stays, the old chunk
goes when PP0 names the new instance. The flip form (no dual gate) keeps every base answer
(test_dual_fixes_flip_unchanged_1003.TestQ695FlipUnchanged).
"""
from __future__ import annotations

import os
import types
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from sglang.srt.managers import scheduler_pp_mixin as ppm  # noqa: E402
from sglang.srt.managers.pp_row_defer_cap import ROW_DEFER_LAP_CAP, PpRowDeferCapExceeded  # noqa: E402
from sglang.srt.weg2 import dual_old_instance as Q  # noqa: E402

RID = "weg2-0-235"
SLOT = 1
DUAL = {"SGLANG_WEG2_DUAL_LAYOUT": "1", "SGLANG_WEG2_GROUP": "P",
        "SGLANG_WEG2_DUAL_P_KV_MAX_TOKENS": "196608"}


def _row(rid=RID, prefix=5120, ext=1024):
    return (rid, prefix, ext, True, False, None, None, False, 13449, (), None)


def _frame(prefix=5120, rid=RID, pass_ct=1560):
    return {
        "__stamp__": (SLOT, pass_ct, 1024, -1, pass_ct, (rid, prefix, prefix + 1024)),
        ppm._ADMISSION_DECISION_PAYLOAD_KEY: (0, (_row(rid, prefix),)),
    }


def _req(rid, end=None):
    r = types.SimpleNamespace(rid=rid, req_pool_idx=None, is_retracted=False)
    if end is not None:
        r.extend_range = types.SimpleNamespace(start=end - 1024, end=end)
    return r


@pytest.fixture
def dual_env(monkeypatch):
    for k, v in DUAL.items():
        monkeypatch.setenv(k, v)


@pytest.fixture
def inbox(monkeypatch):
    q = []
    monkeypatch.setattr(ppm, "resolve_src", lambda group, x: 0)
    monkeypatch.setattr(ppm, "typed_inbox", lambda group: {(0, "proxy"): q})
    return q


def _pp1(*, pending=True, told=None, old_end=5120):
    """PP1 at 19:26:05.98: instance 2 chunked at 5120 (abort recorded), instance 3 queued."""
    old = _req(RID, end=old_end)
    new = _req(RID)
    s = types.SimpleNamespace()
    s.pp_group = object()
    s.ps = types.SimpleNamespace(pp_rank=1, pp_size=3, tp_size=1)
    s.waiting_queue = [new, _req("weg2-0-236"), _req("weg2-0-237")]
    s.chunked_req = old
    s._pending_chunked_abort_req = old if pending else None
    s.running_batch = None
    s._pp_row_chain_owed = False
    s._pp_flip_epoch = lambda: -1
    s._weg2_store_told_armed = True
    s._weg2_store_told = {} if told is None else {RID: told}
    return s, old, new


def _probe(s):
    return ppm.SchedulerPPMixin._pp_proxy_frame_pending(s, SLOT)


# ------------------------------------------------------------------ rule 1: the probe

def test_metal_frame_1560_of_the_old_instance_is_planned_not_deferred(dual_env, inbox):
    inbox.append(_frame())
    s, old, new = _pp1()
    for _ in range(ROW_DEFER_LAP_CAP + 2):          # base: four defers, then the raise
        assert _probe(s) is True
    assert len(inbox) == 1, "the frame stays for the slot's own recv"
    assert "defer_told" not in s._pp_row_probe_stats
    assert s._pp_row_chain_owed is False
    assert s._q695_continued_n >= 1


def test_a_row_for_the_new_instance_still_waits_for_its_told(dual_env, inbox, caplog):
    """PP0 naming instance 3 (told=0 -> start 0) while its told is in flight: #791T as before,
    and the stop names the FULL rid with every instance this rank holds under it."""
    inbox.append(_frame(prefix=0))
    s, old, new = _pp1()
    assert _probe(s) is False
    assert s._pp_row_probe_stats.get("defer_told") == 1
    with pytest.raises(PpRowDeferCapExceeded, match="#791T STORE-TOLD HOP OVERDUE"):
        for _ in range(ROW_DEFER_LAP_CAP + 2):
            _probe(s)
    assert ("Q-695 #791T OVERDUE FULL pp_rank=1 slot=1 rids=weg2-0-235@0"
            "[chunked(end=5120,abort_pending=True)+queued(told=-)]") in caplog.text


def test_no_recorded_abort_means_no_exemption(dual_env, inbox):
    inbox.append(_frame())
    s, old, new = _pp1(pending=False)
    assert _probe(s) is False
    assert s._pp_row_probe_stats.get("defer_told") == 1


def test_a_row_off_the_old_instances_end_is_not_its_continuation(dual_env, inbox):
    inbox.append(_frame(prefix=6144))
    s, old, new = _pp1(old_end=5120)
    assert _probe(s) is False


def test_the_new_instances_told_landing_is_the_base_answer(dual_env, inbox):
    inbox.append(_frame())
    s, old, new = _pp1(told=0)
    assert _probe(s) is True
    assert getattr(s, "_q695_continued_n", 0) == 0, "no exemption needed, none taken"


# ---------------------------------------------- rules 2/3: the old chunk's abort, #791C

def _abort_fake(monkeypatch, *, told=None, old_end=6144):
    from sglang.srt.managers import scheduler as SC
    from sglang.srt.weg2 import p_row_authority

    monkeypatch.setattr(SC, "prepare_abort", lambda req, why: setattr(req, "aborted_why", why))
    monkeypatch.setattr(SC, "release_kv_cache", lambda *a, **k: None)
    monkeypatch.setattr(p_row_authority, "applies", lambda s: True)
    s, old, new = _pp1(told=told, old_end=old_end)
    old.kv_committed_freed = True
    old.to_finish = None
    old.finished = lambda: False
    old.time_stats = types.SimpleNamespace(trace_ctx=types.SimpleNamespace(abort=lambda **k: None))
    released = []
    cur = {"s": None}
    s._pending_chunked_abort_delay = 0
    s._pp_scheduled_extents = lambda: cur["s"]
    s.disaggregation_mode = None
    s.enable_hicache_storage = True
    s.tree_cache = types.SimpleNamespace(supports_mamba=lambda: False,
                                         release_aborted_request=released.append)
    s.ipc_channels = types.SimpleNamespace(send_to_tokenizer=types.SimpleNamespace(
        send_output=lambda obj, r: None))

    def step(schedule):
        cur["s"] = schedule
        SC.Scheduler.process_pending_chunked_abort(s)
        return s.chunked_req is old

    return step, s, old, released


def test_metal_old_chunk_runs_1560_then_leaves_without_cutting_the_new_read(dual_env, monkeypatch):
    step, s, old, released = _abort_fake(monkeypatch)
    assert step({RID: (5120, 1024)}), "frame 1560 still names the old chunk: kept (#791C)"
    assert not step({"weg2-0-236": (26157, 512)}), "PP0's next pass (fwd 1561) names 236 only"
    assert old.aborted_why == "Aborted"
    assert released == [], "the rid-keyed release would cut instance 3's store read"
    assert s._q695_keep_read_n == 1


def test_pp0_naming_the_new_instance_releases_the_dead_old_chunk(dual_env, monkeypatch):
    step, s, old, released = _abort_fake(monkeypatch, told=0)
    assert not step({RID: (0, 1024)}), "start 0 != old end 6144 and instance 3's told is here"
    assert s._q695_new_named_n == 1
    assert released == []


def test_new_instance_named_but_told_in_flight_keeps_the_old_chunk(dual_env, monkeypatch):
    """The #791T probe holds such a frame until the told lands; #791C keeps meanwhile."""
    step, s, old, released = _abort_fake(monkeypatch, told=None)
    assert step({RID: (0, 1024)})


def test_helpers_are_inert_on_pp0(dual_env):
    s, old, new = _pp1()
    s.ps.pp_rank = 0
    assert Q.old_chunk_continued(s, RID, 5120) is False
    assert Q.newer_instance_queued(s, old) is False
    assert Q.schedule_names_new_instance(s, old, {RID: (0, 1024)}) is False


def test_helpers_are_inert_without_the_p_kv_size(monkeypatch):
    with mock.patch.dict(os.environ, {"SGLANG_WEG2_DUAL_LAYOUT": "1", "SGLANG_WEG2_GROUP": "P"},
                         clear=False):
        os.environ.pop("SGLANG_WEG2_DUAL_P_KV_MAX_TOKENS", None)
        s, old, new = _pp1()
        assert Q.old_chunk_continued(s, RID, 5120) is False
