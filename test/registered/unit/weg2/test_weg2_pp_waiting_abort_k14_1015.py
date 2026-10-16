# SPDX-License-Identifier: Apache-2.0
"""K14: a client abort never kills a PP follower under the #631 row authority.

THE SPECIMEN (NF h16p, death 15.10.2026 21:52:38Z, /spinning/gpu-arb/deskq/ausfaelle/
20261015T215242Z_tot/*.P.log.tail4000 lines ~3940-3999, front.log 21:52:38):

    PP0  21:52:37 #969N ADMIT slot=2 fwd_ct=346 rids=[weg2-164-2265]   (launched)
    front 21:52:38,176 WEG2-CLIENT-GONE rid=weg2-164-2265 state=p-leg1 action=abort-p (H102)
    PP0  #1460 CTRL-FWD kinds=['AbortReq'];  #801 ABORT RECEIVED ... deferred=False
    PP1  CHAIN-RECV AbortReq (before its slot-2 frame) -> #801 ABORT RECEIVED deferred=False
    PP2  idem
    PP1  #631 ROW-PROBE DEFER slot=2 ... not yet locatable (first=weg2-164-2265) laps 1..3
    PP1  PpRowDeferCapExceeded '#1180 PP ROW DEFER PAST ITS LAP CAP' -> RANK_EXCEPTION

PP0 had admitted the rid (it was in a launched microbatch); PP1/PP2 still queued it (the
slot-2 frame not yet consumed) and popped it at receipt. The frame then named a rid the
followers no longer had. RED ON THE BASE (1cfbf32584): the follower pops at receipt and the
real probe defers the frame until the #1180 cap raises.

DANGER DIRECTIONS (mutants below): the abort applied rank-locally again (pop at receipt ->
#1180); the aborted rid computed on as a live request on a follower (PP0 finishes it with
abort, the follower would finish it normally -- KV/slot divergence, the leak side); the abort
applied twice (receipt AND after the plan -- the double-release side).
"""

from __future__ import annotations

import inspect
import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from sglang.srt.managers import scheduler as sched_mod  # noqa: E402
from sglang.srt.managers import scheduler_pp_mixin as ppm  # noqa: E402
from sglang.srt.managers.io_struct import AbortReq  # noqa: E402
from sglang.srt.managers.pp_row_defer_cap import (  # noqa: E402
    ROW_DEFER_LAP_CAP,
    PpRowDeferCapExceeded,
)
from sglang.srt.weg2 import pp_abort  # noqa: E402

S = sched_mod.Scheduler
RID = "weg2-164-2265"
OTHER = "weg2-164-2266"
PP = 3
SLOT = 2
EPOCH = 165


# ----------------------------------------------------------------------------- stand-ins


class _Req(types.SimpleNamespace):
    """Hashable by identity, like ``Req`` (the in-flight set is a set of requests)."""

    __hash__ = object.__hash__
    __eq__ = object.__eq__


def _req(rid=RID):
    return _Req(
        rid=rid, to_finish=None, finished_reason=None, mamba_pool_idx=None,
        req_pool_idx=0, kv_committed_freed=True, finished=lambda: False,
        time_stats=types.SimpleNamespace(trace_ctx=types.SimpleNamespace(abort=lambda **k: None)),
    )


def _batch(*reqs):
    return types.SimpleNamespace(reqs=list(reqs))


def _stage(monkeypatch, pp_rank, *, row_authority=True, pp_size=PP):
    from sglang.srt.weg2 import p_row_authority

    monkeypatch.setattr(sched_mod, "prepare_abort", lambda req, why: setattr(req, "aborted_why", why))
    monkeypatch.setattr(sched_mod, "release_kv_cache", lambda *a, **k: None)
    monkeypatch.setattr(p_row_authority, "applies", lambda s: row_authority)
    sent = []
    st = types.SimpleNamespace(
        ps=types.SimpleNamespace(pp_rank=pp_rank, pp_size=pp_size, tp_size=1),
        chunked_req=None, _pending_chunked_abort_req=None, _pending_chunked_abort_delay=0,
        _pp_scheduled_extents=lambda: None,
        waiting_queue=[], running_mbs=[None] * pp_size, mbs=[None] * pp_size,
        running_batch=None, last_batch=None, kv_session_offload=None,
        disaggregation_mode=None, enable_hicache_storage=False,
        grammar_manager=types.SimpleNamespace(abort_requests=lambda r: None),
        _weg2_abort_dormant_hold=lambda r: 0, _weg2_d_park_abort=lambda r: 0,
        tree_cache=types.SimpleNamespace(supports_mamba=lambda: False),
        ipc_channels=types.SimpleNamespace(send_to_tokenizer=types.SimpleNamespace(
            send_output=lambda obj, r: sent.append(obj.rid))),
        # the #631 probe's surface
        pp_group=object(), _pp_row_chain_owed=False, _pp_flip_epoch=lambda: EPOCH,
        _weg2_store_told_armed=False, _weg2_store_told={}, _weg2_store_held={},
        sent=sent,
    )
    # `abort_request` reaches these through `self.` on a real Scheduler
    st._abort_request_now = lambda r: S._abort_request_now(st, r)
    st._weg2_resolve_waiting_aborts = (
        lambda pp0_drained=False: S._weg2_resolve_waiting_aborts(st, pp0_drained=pp0_drained)
        if hasattr(S, "_weg2_resolve_waiting_aborts") else 0)
    st.process_pending_chunked_abort = lambda: S.process_pending_chunked_abort(st)
    return st


def _wire_row(rid):
    return (rid, 0, 1, True, False, None, None, False, None, (), None)


def _frame(rids, slot=SLOT, pass_ct=347):
    return {
        "__stamp__": (slot, pass_ct, 265, EPOCH, pass_ct, (rids[0], 60736, 61001)),
        ppm._ADMISSION_DECISION_PAYLOAD_KEY: (slot, tuple(_wire_row(r) for r in rids)),
    }


def _probe(monkeypatch, st, inbox):
    monkeypatch.setattr(ppm, "resolve_src", lambda group, x: 0)
    monkeypatch.setattr(ppm, "typed_inbox", lambda group: {(0, "proxy"): inbox})
    return ppm.SchedulerPPMixin._pp_proxy_frame_pending(st, SLOT)


def _stamp(pp0, wire):
    """PP0's CTRL-FWD stamp; absent on the base (1cfbf32584), where nothing is stamped."""
    fn = getattr(pp_abort, "pp0_stamp_aborts", None)
    return fn(pp0, wire) if fn is not None else wire


def _plan_admits(st, req, *, chunked=False):
    """What the follower's plan does with PP0's slot-2 frame: the rid leaves the
    waiting queue into the planned microbatch; then the K14 after-plan hook."""
    st.waiting_queue = [r for r in st.waiting_queue if r is not req]
    st.mbs[SLOT] = _batch(req)
    if chunked:
        st.chunked_req = req
    st._weg2_resolve_waiting_aborts()


def _specimen(monkeypatch, *, chunked=False):
    """The 21:52:38 order on the real methods. Returns (pp0, followers, reqs)."""
    pp0 = _stage(monkeypatch, 0)
    r0 = _req()
    pp0.mbs[SLOT] = _batch(r0)                         # #969N ADMIT slot=2, launched
    if chunked:
        pp0.chunked_req = r0
    fol = [_stage(monkeypatch, r) for r in (1, 2)]
    reqs = [_req() for _ in fol]
    for st, r in zip(fol, reqs):
        st.waiting_queue = [r]                        # chain hop landed, frame not yet consumed
    abort = AbortReq(rid=RID)                         # H102 client-gone
    wire = _stamp(pp0, [abort])                       # #1460 CTRL-FWD
    S.abort_request(pp0, abort)                       # PP0 applies its own
    for st in fol:
        S.abort_request(st, wire[0])                  # CHAIN-RECV before the slot-2 frame
    return pp0, fol, [r0] + reqs


# ----------------------------------------------------------------------------- the repro


def test_specimen_follower_keeps_the_frame_locatable_and_never_dies(monkeypatch):
    pp0, fol, (r0, r1, r2) = _specimen(monkeypatch)
    assert r0.to_finish is not None, "PP0: its in-flight request is finished with abort"
    for st, r in zip(fol, (r1, r2)):
        inbox = [_frame([RID])]
        for lap in range(ROW_DEFER_LAP_CAP + 2):      # the base raised #1180 on lap 4
            assert _probe(monkeypatch, st, inbox) is True, (
                f"pp{st.ps.pp_rank} lap {lap}: the frame naming {RID} must be executable")
        assert st.waiting_queue == [r], "not popped at receipt"
        assert st.sent == [], "no side effect at receipt"
        _plan_admits(st, r)
        assert r.to_finish is not None, "after the admitting plan: PP0's to_finish, here too"
        assert not getattr(st, pp_abort.HELD_ATTR, None)


def test_specimen_red_on_base_shape_is_the_1180_death(monkeypatch):
    """The death itself, reproduced with the K14 hold neutered (the base's behaviour):
    the follower pops at receipt and the probe's cap raises #1180."""
    monkeypatch.setattr(pp_abort, "hold_waiting_abort", lambda s, r: False)
    pp0, fol, _ = _specimen(monkeypatch)
    st = fol[0]
    assert st.waiting_queue == []
    inbox = [_frame([RID])]
    with pytest.raises(PpRowDeferCapExceeded) as cm:
        for _ in range(ROW_DEFER_LAP_CAP + 2):
            assert _probe(monkeypatch, st, inbox) is False
    assert "#1180 PP ROW DEFER PAST ITS LAP CAP" in str(cm.value) and RID in str(cm.value)


def test_specimen_chunked_admission_goes_to_the_791c_verdict(monkeypatch):
    pp0, fol, (r0, r1, r2) = _specimen(monkeypatch, chunked=True)
    assert pp0._pending_chunked_abort_req is r0
    for st, r in zip(fol, (r1, r2)):
        assert _probe(monkeypatch, st, [_frame([RID])]) is True
        _plan_admits(st, r, chunked=True)
        assert st._pending_chunked_abort_req is r, "registered as the chunked abort"
        sched = {"s": {RID: (60736, 265)}}
        st._pp_scheduled_extents = lambda: sched["s"]
        st.process_pending_chunked_abort()
        assert st.chunked_req is r, "#791C keeps the chunk while PP0's schedule names it"
        sched["s"] = {OTHER: (0, 1024)}
        st.process_pending_chunked_abort()
        assert st.chunked_req is None and st.sent == [RID], "released once PP0 stops naming it"


def test_frameless_passes_keep_the_hold(monkeypatch):
    _, fol, (_, r1, _) = _specimen(monkeypatch)
    st = fol[0]
    for _ in range(5):
        st._weg2_resolve_waiting_aborts()
        st.process_pending_chunked_abort()              # top-of-plan fallback
    assert st.waiting_queue == [r1] and st.sent == []
    assert RID in getattr(st, pp_abort.HELD_ATTR)


def test_idle_pp0_vote_releases_a_hold_that_no_frame_can_end(monkeypatch):
    _, fol, (_, r1, _) = _specimen(monkeypatch)
    st = fol[0]
    st._pp_microbatches_drained = lambda: True
    st._791c_pp0_drained = False
    vote = types.SimpleNamespace(slots=[(0, 1, "none"), (1, 0, "queue")])
    ppm.weg2_791c_release_on_idle_vote(st, vote)
    assert st.waiting_queue == [] and st.sent == [RID]
    assert not getattr(st, pp_abort.HELD_ATTR)


# ----------------------------------------------------------------------------- Gegenprobe


def test_pp0_had_not_admitted_followers_pop_at_receipt(monkeypatch):
    pp0 = _stage(monkeypatch, 0)
    pp0.waiting_queue = [_req()]                      # still queued on PP0: no frame names it
    st = _stage(monkeypatch, 1)
    st.waiting_queue = [_req()]
    abort = _stamp(pp0, [AbortReq(rid=RID)])[0]
    assert abort.weg2_pp0_admitted is False
    S.abort_request(pp0, abort)
    S.abort_request(st, abort)
    assert pp0.waiting_queue == [] and st.waiting_queue == [] and st.sent == [RID]
    assert not getattr(st, pp_abort.HELD_ATTR, None)


def test_unstamped_abort_is_todays_path(monkeypatch):
    st = _stage(monkeypatch, 1)
    st.waiting_queue = [_req()]
    S.abort_request(st, AbortReq(rid=RID))            # weg2_pp0_admitted None
    assert st.waiting_queue == [] and st.sent == [RID]


def test_abort_all_never_held(monkeypatch):
    st = _stage(monkeypatch, 1)
    st.waiting_queue = [_req(), _req(OTHER)]
    S.abort_request(st, AbortReq(rid="", abort_all=True, weg2_pp0_admitted=True))
    assert st.waiting_queue == [] and sorted(st.sent) == sorted([RID, OTHER])


def test_pp0_and_non_pp_never_hold(monkeypatch):
    for st in (_stage(monkeypatch, 0), _stage(monkeypatch, 0, pp_size=1)):
        st.waiting_queue = [_req()]
        S.abort_request(st, AbortReq(rid=RID, weg2_pp0_admitted=True))
        assert st.waiting_queue == [] and st.sent == [RID]


def test_abort_on_d_tp_only_unchanged(monkeypatch):
    """Group D on NF is TP3, no PP: nothing stamped, nothing held, popped at receipt."""
    d = _stage(monkeypatch, 0, pp_size=1, row_authority=False)
    d.waiting_queue = [_req()]
    abort = _stamp(d, [AbortReq(rid=RID)])[0]
    assert abort.weg2_pp0_admitted is None
    S.abort_request(d, abort)
    assert d.waiting_queue == [] and d.sent == [RID]


def test_pass_aligned_form_neither_stamps_nor_holds(monkeypatch):
    pp0 = _stage(monkeypatch, 0, row_authority=False)
    pp0.mbs[SLOT] = _batch(_req())
    abort = _stamp(pp0, [AbortReq(rid=RID)])[0]
    assert abort.weg2_pp0_admitted is None, "no row authority: PP0 stamps nothing"
    st = _stage(monkeypatch, 1, row_authority=False)
    st.waiting_queue = [_req()]
    S.abort_request(st, AbortReq(rid=RID, weg2_pp0_admitted=True))
    assert st.waiting_queue == [], "#791C-NF: the pass-aligned wire applies at receipt"


def test_follower_that_already_ran_the_frame_applies_at_receipt(monkeypatch):
    st = _stage(monkeypatch, 1)
    r = _req()
    st.mbs[SLOT] = _batch(r)                          # frame consumed before the abort
    S.abort_request(st, AbortReq(rid=RID, weg2_pp0_admitted=True))
    assert r.to_finish is not None and not getattr(st, pp_abort.HELD_ATTR, None)


def test_other_queued_requests_untouched_by_a_held_abort(monkeypatch):
    _, fol, _ = _specimen(monkeypatch)
    st = fol[0]
    other = _req(OTHER)
    st.waiting_queue.append(other)
    st._weg2_resolve_waiting_aborts()
    assert other in st.waiting_queue and other.to_finish is None


def test_pp0_admitted_reads_the_inflight_set():
    a = _req()
    s = types.SimpleNamespace(chunked_req=None, mbs=[None, _batch(a), None], running_mbs=[],
                              running_batch=None, last_batch=None, waiting_queue=[_req(OTHER)])
    assert pp_abort.pp0_admitted(s, RID) is True
    assert pp_abort.pp0_admitted(s, OTHER) is False, "queued on PP0 = not admitted"
    s2 = types.SimpleNamespace(chunked_req=a, mbs=[], running_mbs=[], running_batch=_batch(),
                               last_batch=None)
    assert pp_abort.pp0_admitted(s2, RID) is True


def test_verdict_table():
    V = pp_abort.waiting_abort_verdict
    assert V(True, False) == "keep"
    assert V(True, True) == "pop"
    assert V(False, False) == "apply" and V(False, True) == "apply"


# ----------------------------------------------------------------------------- mutants


def _specimen_outcome(monkeypatch):
    """Run the specimen end to end; return (died, pp0_finished, follower_finished, sends)."""
    pp0, fol, (r0, r1, r2) = _specimen(monkeypatch)
    died = False
    for st, r in zip(fol, (r1, r2)):
        inbox = [_frame([RID])]
        try:
            for _ in range(ROW_DEFER_LAP_CAP + 2):
                if not _probe(monkeypatch, st, inbox):
                    continue
                break
            else:
                died = True
        except PpRowDeferCapExceeded:
            died = True
        if r in st.waiting_queue:
            _plan_admits(st, r)
        st._weg2_resolve_waiting_aborts()             # a second settle: must be a no-op
    return died, r0.to_finish is not None, [r.to_finish is not None for r in (r1, r2)], [
        list(st.sent) for st in fol]


def test_mutant_none_is_clean(monkeypatch):
    died, pp0_fin, fol_fin, sends = _specimen_outcome(monkeypatch)
    assert not died and pp0_fin and fol_fin == [True, True] and sends == [[], []]


def test_mutant_rank_local_abort_again_is_caught(monkeypatch):
    monkeypatch.setattr(pp_abort, "hold_waiting_abort", lambda s, r: False)
    died, *_ = _specimen_outcome(monkeypatch)
    assert died, "MUTANT survived: a rank-local pop at receipt must reproduce #1180"


def test_mutant_stamp_lies_not_admitted_is_caught(monkeypatch):
    monkeypatch.setattr(pp_abort, "pp0_admitted", lambda s, rid, abort_all=False: False)
    died, *_ = _specimen_outcome(monkeypatch)
    assert died, "MUTANT survived: PP0's verdict must decide the hold"


def test_mutant_aborted_rid_computed_on_is_caught(monkeypatch):
    """Leak side: the hold is never applied -- the follower would finish a request
    PP0 aborted (KV/slot kept against PP0's state)."""
    monkeypatch.setattr(pp_abort, "waiting_abort_verdict", lambda still, drained: "keep")
    died, pp0_fin, fol_fin, _ = _specimen_outcome(monkeypatch)
    assert not died and pp0_fin
    assert fol_fin != [True, True], "MUTANT survived: follower must carry PP0's to_finish"


def test_mutant_applied_twice_is_caught(monkeypatch):
    """Double-release side: the abort runs at receipt AND after the plan."""
    real = pp_abort.hold_waiting_abort

    def _hold_and_apply(s, r):
        real(s, r)
        return False                                  # held AND applied at receipt

    monkeypatch.setattr(pp_abort, "hold_waiting_abort", _hold_and_apply)
    died, _, _, sends = _specimen_outcome(monkeypatch)
    assert died or any(len(x) > 1 for x in sends), "MUTANT survived: abort applied twice"


# ----------------------------------------------------------------------------- wiring


def test_wiring_stamp_before_the_send_on_pp0():
    src = inspect.getsource(ppm.SchedulerPPMixin._pp_forward_and_process_input_requests)
    i_stamp = src.index("pp0_stamp_aborts")
    i_send = src.index("self.send_req_work = self._pp_send_pyobj_to_next_stage(")
    assert i_stamp < i_send
    assert "is_first_rank" in src[max(0, i_stamp - 800):i_stamp]


def test_wiring_settle_after_the_plan_before_the_launch():
    src = inspect.getsource(ppm.SchedulerPPMixin._event_loop_pp_body)
    i_after = src.index("_pp_void_pass.after_plan(self, mb_id)")
    i_hook = src.index("self._weg2_resolve_waiting_aborts()")
    i_launch = src.index("_pp_launch_batch(", i_after)
    assert i_after < i_hook < i_launch


def test_wiring_abort_request_holds_before_any_side_effect():
    src = inspect.getsource(S.abort_request)
    assert src.index("hold_waiting_abort") < src.index("if deferred:")
    assert "drop_ple_admission" not in src, "every side effect lives in _abort_request_apply"
    assert "_abort_request_now" in inspect.getsource(S._abort_request_apply)
    assert "_weg2_resolve_waiting_aborts" in inspect.getsource(S.process_pending_chunked_abort)
    assert "_weg2_pp_held_waiting_aborts" in inspect.getsource(ppm.weg2_791c_release_on_idle_vote)


def test_abortreq_field_roundtrips_the_wire():
    import pickle

    import msgspec

    a = AbortReq(rid=RID)
    assert a.weg2_pp0_admitted is None
    a.weg2_pp0_admitted = True
    assert pickle.loads(pickle.dumps(a)).weg2_pp0_admitted is True
    assert msgspec.msgpack.decode(msgspec.msgpack.encode(a), type=AbortReq).weg2_pp0_admitted is True
