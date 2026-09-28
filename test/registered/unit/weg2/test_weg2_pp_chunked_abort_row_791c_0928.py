# SPDX-License-Identifier: Apache-2.0
"""#791C: under the #631 row authority a follower applies a chunked abort off
PP0's forwarded schedule, not off the xsn324 local pass count.

THE SPECIMEN (27B proof boot rc12z24 bb84760576, Fix B armed,
/spinning/docker-acceptance/27b/evidence/
boot_weg2_dkr27browauthoritybar1w109281556_bb84760576_0928_155617.P.log 23300-23440):

    front  WEG2-CLIENT-GONE rid=weg2-0-6 state=p-leg1 action=abort-p (H102, wait_s=61.4)
    PP0  ABORT RECEIVED weg2-0-6 -> "applied in 2 pass(es)"; launches 193 (66097-67121)
         and 194 (67121-68145)
    PP1  ABORT RECEIVED (r37) -> "applied in 1 pass(es)"; runs frame 193, drops before 194
    PP2  ABORT RECEIVED (r37) -> delay 0, drops at once; then drains frame 192
    PP2  #791 FORWARDED SCHEDULE UNEXECUTABLE ... missing rid(s)=weg2-0-6
    PP1  #791 FORWARDED SCHEDULE UNEXECUTABLE ... missing rid(s)=weg2-0-6

The AbortReq reached all three ranks in the same second while the stages were
0/1/2 frames behind PP0; the static pp_size-1-r delay assumes plan lag, which
the row authority removed. RED ON THE PARENT (bb84760576): PP2 drops at its
first pass with frame 192 still naming the rid; PP1 one pass later.
"""

from __future__ import annotations

import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import pp_abort  # noqa: E402

RID = "weg2-0-6"


def _frame(start, other=None):
    """A forwarded schedule naming weg2-0-6's chunk [start, start+1024)."""
    s = {RID: (start, 1024)}
    if other:
        s[other] = (0, 1024)
    return s


# ---- the pure verdict -----------------------------------------------------------

def test_verdict_is_only_for_followers_under_row_authority():
    assert pp_abort.follower_row_verdict(0, True, _frame(66097), RID) is None
    assert pp_abort.follower_row_verdict(2, False, _frame(66097), RID) is None


def test_verdict_keeps_while_named_or_frameless_and_applies_when_dropped():
    assert pp_abort.follower_row_verdict(2, True, _frame(65073), RID) is False
    assert pp_abort.follower_row_verdict(2, True, None, RID) is False
    assert pp_abort.follower_row_verdict(2, True, {}, RID) is False
    assert pp_abort.follower_row_verdict(2, True, {"weg2-1-1": (0, 1024)}, RID) is True


# ---- the scheduler method, replaying the 15:59:23 order -------------------------

def _fake(monkeypatch, pp_rank, row_authority=True):
    from sglang.srt.managers import scheduler as sched_mod
    from sglang.srt.weg2 import p_row_authority

    sent = []
    monkeypatch.setattr(sched_mod, "prepare_abort", lambda req, why: setattr(req, "aborted_why", why))
    monkeypatch.setattr(sched_mod, "release_kv_cache", lambda *a, **k: None)
    monkeypatch.setattr(p_row_authority, "applies", lambda s: row_authority)
    req = types.SimpleNamespace(
        rid=RID, req_pool_idx=None, kv_committed_freed=True, to_finish=1,
        finished=lambda: False,
        time_stats=types.SimpleNamespace(trace_ctx=types.SimpleNamespace(abort=lambda **k: None)),
    )
    cur = {"s": None}
    fake = types.SimpleNamespace(
        ps=types.SimpleNamespace(pp_rank=pp_rank, pp_size=3),
        _pending_chunked_abort_req=req, chunked_req=req,
        _pending_chunked_abort_delay=pp_abort.chunked_abort_delay(3, pp_rank),
        _pp_scheduled_extents=lambda: cur["s"],
        disaggregation_mode=None, enable_hicache_storage=False,
        tree_cache=types.SimpleNamespace(supports_mamba=lambda: False),
        ipc_channels=types.SimpleNamespace(send_to_tokenizer=types.SimpleNamespace(
            send_output=lambda obj, r: sent.append(obj.rid))),
    )
    return sched_mod.Scheduler.process_pending_chunked_abort, fake, cur, req, sent


def _pass(f, fake, cur, schedule):
    cur["s"] = schedule
    f(fake)
    return fake.chunked_req is not None


def test_pp2_keeps_the_chunk_through_every_frame_pp0_still_launched(monkeypatch):
    f, fake, cur, req, sent = _fake(monkeypatch, pp_rank=2)
    for start in (65073, 66097, 67121):          # frames 192, 193, 194 (PP0's last)
        assert _pass(f, fake, cur, _frame(start)), start
    assert _pass(f, fake, cur, None), "a frameless cycle keeps it (plan bypass)"
    assert _pass(f, fake, cur, {}), "and so does an empty frame"
    assert not sent
    assert not _pass(f, fake, cur, {"weg2-1-1": (0, 1024)}), "PP0's next pass names other rids only"
    assert sent == [RID] and req.aborted_why == "Aborted"
    assert fake._pending_chunked_abort_req is None


def test_pp1_keeps_frame_194_that_the_static_delay_dropped(monkeypatch):
    f, fake, cur, req, sent = _fake(monkeypatch, pp_rank=1)
    assert _pass(f, fake, cur, _frame(66097))    # frame 193
    assert _pass(f, fake, cur, _frame(67121))    # frame 194: xsn324's delay 1 dropped it here
    assert not _pass(f, fake, cur, {"weg2-1-1": (0, 1024)})
    assert sent == [RID]


def test_pp0_keeps_the_xsn324_countdown(monkeypatch):
    f, fake, cur, req, sent = _fake(monkeypatch, pp_rank=0)
    assert fake._pending_chunked_abort_delay == 2
    assert _pass(f, fake, cur, None) and _pass(f, fake, cur, None)
    assert not _pass(f, fake, cur, None) and sent == [RID]


def test_without_row_authority_a_follower_keeps_the_xsn324_countdown(monkeypatch):
    f, fake, cur, req, sent = _fake(monkeypatch, pp_rank=1, row_authority=False)
    assert _pass(f, fake, cur, _frame(66097))    # delay 1: one more pass
    assert not _pass(f, fake, cur, _frame(67121)), "the plan-lag form is unchanged"
    assert sent == [RID]
