# SPDX-License-Identifier: Apache-2.0
"""#791C on the form WITHOUT the row authority (NF, z30w-park epoch 113, 29.09. 09:11Z).

THE SPECIMEN (/spinning/docker-acceptance/nf/evidence/
boot_weg2_dkrnfh91dprsavisadoptstcutvsyncodx2bswre2cutz30wparkbar1dauer09290827_50ae2014b0_0929_082743.P.log
215490-215660):

    front  09:11:07 WEG2-CLIENT-GONE rid=weg2-112-205 state=p-leg1 action=abort-p (H102)
    PP0  09:11:10 ABORT RECEIVED -> "applied in 2 pass(es)"; plans 69888 (slot 2, fwd 103)
         and 86272 (slot 0, fwd 104, the chunk that leaves 41122)
    PP1  09:11:13 ABORT RECEIVED -> "applied in 1 pass(es)"; plans 69888 (slot 2, fwd 103), drops 86272
    PP2  09:11:16 ABORT RECEIVED -> delay 0; drops before 69888
    then PP0 in pp:0/recv_object[src=2] awaiting_size, PP1 in recv_object[src=0], PP2 silent.

Every stage read the AbortReq in the SAME pass -- slot 2 of the lap, right before it planned
69888. Without the row authority a follower receives PP0's request list once per pass, in the
pass of the same index (blocking recv at the top of the pass, PP0 forwards before it plans), and
continues the chunk rank-locally in lockstep. The wire position of the AbortReq IS PP0's
decision: xsn324's premise "stage r reads the abort r passes later" holds in wall time only, and
the static delays 2/1/0 turned it into three different launch sets.

The ring below drives the real ``Scheduler._abort_request_now`` and
``Scheduler.process_pending_chunked_abort`` on three stand-in stages: request wire pass-aligned,
chunk continuation rank-local, hidden states stage r-1 -> r per pass, the last stage's output
back to PP0 per pass. RED ON THE BASE (424346f693): PP0 launches two chunks after the abort that
PP2 never runs -- PP0's output receive from src=2 stays open, frames stay unconsumed.
"""

from __future__ import annotations

import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from sglang.srt.weg2 import pp_abort  # noqa: E402

RID = "weg2-112-205"
PP = 3


def _stage(monkeypatch, pp_rank, row_authority):
    from sglang.srt.managers import scheduler as sched_mod
    from sglang.srt.weg2 import p_row_authority

    monkeypatch.setattr(sched_mod, "prepare_abort", lambda req, why: setattr(req, "aborted_why", why))
    monkeypatch.setattr(sched_mod, "release_kv_cache", lambda *a, **k: None)
    monkeypatch.setattr(p_row_authority, "applies", lambda s: row_authority)
    sent = []
    req = types.SimpleNamespace(
        rid=RID, req_pool_idx=0, kv_committed_freed=True, to_finish=None,
        mamba_pool_idx=None, finished=lambda: False,
        time_stats=types.SimpleNamespace(trace_ctx=types.SimpleNamespace(abort=lambda **k: None)),
    )
    st = types.SimpleNamespace(
        ps=types.SimpleNamespace(pp_rank=pp_rank, pp_size=PP),
        chunked_req=req, _pending_chunked_abort_req=None, _pending_chunked_abort_delay=0,
        _pp_scheduled_extents=lambda: None,
        waiting_queue=[], running_mbs=[None] * PP, mbs=[None] * PP, kv_session_offload=None,
        disaggregation_mode=None, enable_hicache_storage=False,
        grammar_manager=types.SimpleNamespace(abort_requests=lambda r: None),
        _weg2_abort_dormant_hold=lambda r: 0, _weg2_d_park_abort=lambda r: 0,
        tree_cache=types.SimpleNamespace(supports_mamba=lambda: False),
        ipc_channels=types.SimpleNamespace(send_to_tokenizer=types.SimpleNamespace(
            send_output=lambda obj, r: sent.append(obj.rid))),
        sent=sent, req=req, launched={}, next_chunk=0,
    )
    return st


def _abort():
    from sglang.srt.managers.io_struct import AbortReq

    return AbortReq(rid=RID)


def _pass(st, p, reqs):
    """One pass of index ``p`` on one stage: intake (the abort is recorded here),
    then the plan (``process_pending_chunked_abort`` first, then the rank-local
    chunk continuation), then the launch."""
    from sglang.srt.managers import scheduler as sched_mod

    for r in reqs:
        sched_mod.Scheduler._abort_request_now(st, r)
    sched_mod.Scheduler.process_pending_chunked_abort(st)
    if st.chunked_req is not None:
        st.launched[p] = st.next_chunk
        st.next_chunk += 1


def _run_ring(monkeypatch, abort_pass, passes=10, row_authority=False):
    """Wall order: stage r runs pass p at tick p + r (the pipeline stagger). The
    request wire is pass-aligned: stage r's pass p reads what PP0 read (and
    forwarded) in its pass p. Returns the stages and the ring's open ends."""
    stages = [_stage(monkeypatch, r, row_authority) for r in range(PP)]
    wire = {abort_pass: [_abort()]}
    for tick in range(passes + PP):
        for r, st in enumerate(stages):
            p = tick - r
            if 0 <= p < passes:
                _pass(st, p, wire.get(p, []))
    open_recv, orphan = [], []
    for p in range(passes):
        for r in range(1, PP):
            up, me = p in stages[r - 1].launched, p in stages[r].launched
            if me and not up:
                open_recv.append(f"PP{r} recv_object[src={r - 1}] pass {p}")
            if up and not me:
                orphan.append(f"PP{r - 1}->PP{r} frame pass {p}")
        if p in stages[0].launched and p not in stages[PP - 1].launched:
            open_recv.append(f"PP0 recv_object[src={PP - 1}] pass {p} (output)")
    return stages, open_recv, orphan


@pytest.mark.parametrize("abort_pass", [1, 3, 6])
def test_ring_stops_at_the_same_chunk_without_row_authority(monkeypatch, abort_pass):
    stages, open_recv, orphan = _run_ring(monkeypatch, abort_pass)
    assert open_recv == [], open_recv
    assert orphan == [], orphan
    chunks = [sorted(st.launched.values()) for st in stages]
    assert chunks[0] == chunks[1] == chunks[2] == list(range(abort_pass)), chunks
    for st in stages:
        assert st.chunked_req is None and st._pending_chunked_abort_req is None
        assert st._pending_chunked_abort_delay == 0
        assert st.sent == [RID] and st.req.aborted_why == "Aborted"


def test_the_specimen_order_is_what_the_base_produced(monkeypatch):
    """The 09:11 launch sets, reproduced by the xsn324 delays (the legacy
    ``row_authority=None`` reading): PP0 two chunks past the abort pass, PP1
    one, PP2 none -- PP0's output receive from src=2 stays open."""
    stages = [_stage(monkeypatch, r, False) for r in range(PP)]
    for r, st in enumerate(stages):
        st._pending_chunked_abort_req = st.req
        st._pending_chunked_abort_delay = pp_abort.chunked_abort_delay(PP, r)
        _pass(st, 5, [])
        _pass(st, 6, [])
    assert [sorted(st.launched) for st in stages] == [[5, 6], [5], []]


def test_delay_is_zero_on_every_stage_without_row_authority():
    assert [pp_abort.chunked_abort_delay(PP, r, row_authority=False) for r in range(PP)] == [0, 0, 0]


def test_row_authority_form_keeps_the_xsn324_delays_byte_for_byte():
    """27B (Fix B): PP0 keeps its countdown, the followers' #791C verdict
    follows PP0's forwarded schedule -- the recorded delays are unchanged."""
    assert [pp_abort.chunked_abort_delay(PP, r, row_authority=True) for r in range(PP)] == [2, 1, 0]
    assert [pp_abort.chunked_abort_delay(PP, r) for r in range(PP)] == [2, 1, 0]
    assert pp_abort.chunked_abort_delay(1, 0, row_authority=False) == 0


def test_abort_request_now_records_the_row_authority_delays(monkeypatch):
    from sglang.srt.managers import scheduler as sched_mod

    for r, want in enumerate((2, 1, 0)):
        st = _stage(monkeypatch, r, True)
        sched_mod.Scheduler._abort_request_now(st, _abort())
        assert st._pending_chunked_abort_req is st.req
        assert st._pending_chunked_abort_delay == want


def test_an_unreadable_row_authority_keeps_the_xsn324_rule(monkeypatch):
    from sglang.srt.managers import scheduler as sched_mod
    from sglang.srt.weg2 import p_row_authority

    st = _stage(monkeypatch, 0, True)

    def _boom(s):
        raise RuntimeError("no form")

    monkeypatch.setattr(p_row_authority, "applies", _boom)
    sched_mod.Scheduler._abort_request_now(st, _abort())
    assert st._pending_chunked_abort_delay == 2


def test_row_authority_of_names_no_form_on_a_non_pp_engine(monkeypatch):
    from sglang.srt.weg2 import p_row_authority

    monkeypatch.setattr(p_row_authority, "applies", lambda s: True)
    one = types.SimpleNamespace(ps=types.SimpleNamespace(pp_rank=0, pp_size=1))
    assert pp_abort.row_authority_of(one) is None
    three = types.SimpleNamespace(ps=types.SimpleNamespace(pp_rank=1, pp_size=3))
    assert pp_abort.row_authority_of(three) is True
    monkeypatch.setattr(p_row_authority, "applies", lambda s: False)
    assert pp_abort.row_authority_of(three) is False
