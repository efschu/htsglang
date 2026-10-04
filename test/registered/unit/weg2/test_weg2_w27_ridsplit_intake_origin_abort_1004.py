"""W27 RID-SPLIT (NF nf9, 04.10.2026, image rc12z30y9nf9, 11:31:33Z after 35 min serving).

PP0 named ``WEG2-INTAKE-STALL rid=weg2-84-299`` (11:31:29) and dropped it from ITS waiting
queue; the front's ``/abort_request`` -- the only path to the followers (xsn288) -- never came
(the leg was an un-awaited LEG1-EARLY future). PP1 kept the rid: four seconds later PP0 planned
``[weg2-79-295 +17]`` and PP1 ``[weg2-84-299 +4643, weg2-79-295 +17]`` for the same pass ->
``#1233 W27 PP WIDTH DIVERGENCE REFUSED`` (17 rows for 4660 tokens), group P dead.

Pinned here: PP0's drop is relayed on the origin's next intake as an AbortReq, every follower
drops the same rid before planning, PP0 answers the tokenizer exactly once, and every form other
than group P's PP origin keeps the hook it had.
"""
from __future__ import annotations

import os
from http import HTTPStatus
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import intake_stall as st  # noqa: E402


def _req(rid, n):
    return SimpleNamespace(rid=rid, full_untruncated_fill_ids=[0] * n, prefix_indices=[],
                           mamba_pool_idx=None)


def _rank(pp_rank, sent):
    from sglang.srt.managers import scheduler as sch
    from sglang.srt.managers.schedule_batch import ScheduleBatch  # noqa: F401  (import check)
    from sglang.srt.disaggregation.utils import DisaggregationMode

    obj = sch.Scheduler.__new__(sch.Scheduler)
    obj.ps = SimpleNamespace(pp_size=3, pp_rank=pp_rank, attn_tp_rank=0, attn_cp_rank=0)
    obj._weg2_intake_watch = st.IntakeStallWatch(hold_s=1.0)
    # each rank holds its OWN copy of the replicated queue (no row authority)
    obj.waiting_queue = [_req("weg2-84-299", 25379), _req("weg2-79-295", 113425)]
    obj.enable_hicache_storage = False
    obj.enable_hierarchical_cache = False
    obj.ipc_channels = SimpleNamespace(
        send_to_tokenizer=SimpleNamespace(send_output=lambda a, r: sent.append(a)))
    obj.chunked_req = None
    obj.anchor_tails = None
    obj.disaggregation_mode = DisaggregationMode.NULL
    obj.grammar_manager = SimpleNamespace(abort_requests=lambda r: None)
    obj.running_mbs, obj.mbs = [], []
    obj.kv_session_offload = None
    return obj


def _origin_wiring(obj, monkeypatch, group="P"):
    """What ``init_request_receiver`` builds for this rank (the seam under test)."""
    from sglang.srt.weg2 import intake_origin_abort as ioa

    obj._weg2_intake_origin_aborts = ioa.IntakeOriginAborts(relay=ioa.relay_applies(
        group=group, pp_size=obj.ps.pp_size, pp_rank=obj.ps.pp_rank,
        attn_tp_rank=obj.ps.attn_tp_rank, attn_cp_rank=obj.ps.attn_cp_rank))
    return ioa.origin_hook(obj._weg2_intake_origin_aborts, None)


def _rids(obj):
    return [r.rid for r in obj.waiting_queue]


def test_pp0_stall_drop_reaches_every_follower_before_the_next_plan(monkeypatch):
    from sglang.srt.managers import scheduler as sch
    from sglang.srt.managers.corridor_guard import GROUP_ENV

    monkeypatch.setenv(GROUP_ENV, "P")
    sent0, sent1, sent2 = [], [], []
    pp0, pp1, pp2 = _rank(0, sent0), _rank(1, sent1), _rank(2, sent2)
    hook = _origin_wiring(pp0, monkeypatch)
    for f in (pp1, pp2):
        _origin_wiring(f, monkeypatch)

    # the stall pass: PP0 refuses weg2-84-299 (nf9 11:31:29), the followers keep theirs
    stalled = pp0.waiting_queue[0]
    sch.Scheduler._weg2_intake_stall_observe(pp0, stalled, None, immediate=True)
    for f in (pp1, pp2):
        sch.Scheduler._weg2_intake_stall_observe(f, f.waiting_queue[0], None, immediate=True)
    assert _rids(pp0) == ["weg2-79-295"]
    assert len(sent0) == 1 and st.is_intake_stall(sent0[0].finished_reason["message"])

    # the next pass: the origin's intake list carries the drop down the chain
    assert hook is not None
    relayed = list(hook())
    assert [a.rid for a in relayed] == ["weg2-84-299"]
    assert relayed[0].finished_reason["status_code"] == HTTPStatus.SERVICE_UNAVAILABLE
    for rank in (pp0, pp1, pp2):
        for a in relayed:
            sch.Scheduler._abort_request_now(rank, a)

    # nf9: PP0 queue-req 2 vs PP1 3 -> W27. Now every stage plans from the same queue.
    assert _rids(pp0) == _rids(pp1) == _rids(pp2) == ["weg2-79-295"]
    assert len(sent0) == 1          # PP0 answered once; its own relayed copy found nothing
    assert list(hook()) == []       # drained: one relay per drop


def test_only_group_p_pp_origin_relays_every_other_form_keeps_its_hook():
    from sglang.srt.weg2 import intake_origin_abort as ioa

    base = dict(pp_size=3, pp_rank=0, attn_tp_rank=0, attn_cp_rank=0)
    assert ioa.relay_applies(group="P", **base)
    assert not ioa.relay_applies(group="D", **base)
    assert not ioa.relay_applies(group="", **base)
    assert not ioa.relay_applies(group="P", **{**base, "pp_size": 1})
    assert not ioa.relay_applies(group="P", **{**base, "pp_rank": 1})
    assert not ioa.relay_applies(group="P", **{**base, "attn_tp_rank": 1})

    def vision():
        return ["vision-verdict"]

    off = ioa.IntakeOriginAborts(relay=False)
    assert off.note("x", "m") is False
    assert ioa.origin_hook(off, vision) is vision          # byte-for-byte the old hook
    assert ioa.origin_hook(off, None) is None
    assert ioa.origin_hook(None, vision) is vision

    on = ioa.IntakeOriginAborts(relay=True)
    hook = ioa.origin_hook(on, vision)
    assert hook() == ["vision-verdict"]                     # nothing noted: nothing added
    on.note("weg2-84-299", "WEG2-INTAKE-STALL rid=weg2-84-299")
    out = hook()
    assert out[0] == "vision-verdict" and out[1].rid == "weg2-84-299"   # vision leads


def test_wiring_source_ratchet():
    from sglang.srt.managers import scheduler as sch

    src = open(sch.__file__).read()
    i = src.index("def init_request_receiver(self) -> None:")
    blk = src[i:src.index("def _health_check_gate", i)]
    assert "_ioa.origin_hook(" in blk and "self._weg2_intake_origin_aborts," in blk
    j = src.index("def _weg2_intake_stall_observe")
    k = src.index("def _abort_request_now", j)
    assert "_ioa.note(req.rid, message)" in src[j:k]
