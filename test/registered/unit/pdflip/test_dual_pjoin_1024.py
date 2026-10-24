"""#DPJ (24.10.): two dual-layout P defects of 27B NVFP4 dual rc39a (b9cddabda1),
boot dkr27bnvfp4dualbar1drei10240856, P log ..._1024_085659.

A. TWIN BEHIND AN UNGRANTED SOURCE (p_twin_defer rule 7). 09:02:51 PP0 'P-KV PP0
   GRANT rid=pdflip-0-10 tokens=86016' (GRANT-BYPASS past the head pdflip-0-6, which
   waited for its own card grant), then '#TW TWIN-DEFER rid=pdflip-0-10 ...
   sources=['pdflip-0-6']'. The twin's follower pre-charge (held until its told
   leaves, Q-630) kept 0-6 short ('#1640 GRANT-INFEASIBLE ... waits for the return of
   the followers' pre-charge'); 0-6 waited for the twin, the twin for 0-6 -- until
   the Frist ('TWIN-RELEASE reason=deadline waited_s=120.00' 09:04:51) and 0-6's
   grant one second later ('after 5164 waits over 122.1 s'): ADMISSION-WEDGE
   P-KV-WAIT, 4 queued, 0 running.

B. #1180-W HOLD AGAINST PP0's pp0_admitted=False (scheduler._pdflip_defer_waiting_abort).
   09:09:36 PP0 'PDFLIP-PP-ABORT-VERDICT rid=pdflip-0-28 pp0_admitted=False ... every
   stage applies it at receipt (K14)'; PP2 'PDFLIP-PP-WAITING-ABORT held rid=pdflip-0-28'
   (a told of an earlier list stood there); the front's RESUME sent the next instance
   under the same rid; 09:09:46 PP2 '#996 SECOND CHUNKED MINT ... rid=pdflip-0-28' --
   group death.

Danger directions: (A) the twin held behind a source that computes nothing (the
specimen); a twin released while its source is granted and computing (lost share);
the flip form changed. (B) the stamped False held again (the specimen); an
unstamped / True abort applied at receipt (reopens #1180).
"""

from __future__ import annotations

import logging
import os
import types
import unittest.mock as mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from flliper.srt.managers import scheduler as SC  # noqa: E402
from flliper.srt.pdflip import p_row_authority  # noqa: E402
from flliper.srt.pdflip import p_twin_defer as tw  # noqa: E402
from flliper.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

N = 8192
DUAL_ENV = {"FLLIPER_PDFLIP_DUAL_LAYOUT": "1", "FLLIPER_PDFLIP_GROUP": "P"}


# ---------------------------------------------------------------- A. twin rule 7


class _Req:
    def __init__(self, rid, ids, wait=False):
        self.rid = rid
        self.origin_input_ids = list(ids)
        self._dual_kv_wait = wait
        self.done = False

    def finished(self):
        return self.done


class _Sched:
    def __init__(self):
        self.ps = types.SimpleNamespace(pp_rank=0, pp_size=3, tp_size=1)
        self.waiting_queue = []
        self.chunked_req = None
        self.running_batch = None
        self.page_size = 1
        self.chunked_prefill_size = 1024


def _ids(shared, tail, salt):
    return list(range(shared)) + [10_000_000 + salt * 100_000 + i for i in range(tail)]


class _Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


@pytest.fixture
def clock(monkeypatch):
    c = _Clock()
    monkeypatch.setattr(tw, "_now", c)
    return c


def _arm(monkeypatch, dual):
    monkeypatch.setenv(tw.ENV, "1")
    monkeypatch.setenv(tw.ENV_MIN_TOKENS, str(N))
    monkeypatch.setenv(tw.ENV_WAIT_S, "120")
    monkeypatch.setenv(tw.ENV_SETTLE_MS, "500")
    monkeypatch.setenv("FLLIPER_PDFLIP_P_TWIN_CHAIN", "0")
    monkeypatch.setattr(tw, "_twin_anchor_armed", lambda: False)
    monkeypatch.setattr(tw, "_gain", lambda st, src, shared: True)
    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "P")
    if dual:
        monkeypatch.setenv("FLLIPER_PDFLIP_DUAL_LAYOUT", "1")
    else:
        monkeypatch.delenv("FLLIPER_PDFLIP_DUAL_LAYOUT", raising=False)


@pytest.fixture
def dual(monkeypatch):
    _arm(monkeypatch, True)


@pytest.fixture
def flip(monkeypatch):
    _arm(monkeypatch, False)


def _specimen(head_waits=True):
    """pdflip-0-6 (the head, 250854 tokens) and its fork twin pdflip-0-10 (82000,
    shared 18227) -- scaled: shared 9000 >= N."""
    s = _Sched()
    head = _Req("pdflip-0-6", _ids(9000, 4000, 6), wait=head_waits)
    twin = _Req("pdflip-0-10", _ids(9000, 2000, 10))
    s.waiting_queue = [head, twin]
    return s, head, twin


def test_specimen_twin_not_held_behind_a_grant_waiting_source(dual, clock, caplog):
    """RED on 1ecda02a58: TWIN-DEFER sources=['pdflip-0-6'] -> held to the Frist."""
    s, head, twin = _specimen(head_waits=True)
    with caplog.at_level(logging.INFO, logger=tw.logger.name):
        assert tw.intake_defer(s, twin) is False
    assert not tw.is_deferred(s, twin.rid)
    assert any("TWIN-SOURCE-UNGRANTED" in r.getMessage() and "at=intake" in r.getMessage()
               for r in caplog.records)


def test_held_twin_released_once_its_source_waits_for_its_grant(dual, clock, caplog):
    """A source granted at intake that goes back to a grant wait (P-PAUSE requeue,
    regrant) releases the held twin at once -- not after 120 s."""
    s, head, twin = _specimen(head_waits=False)
    assert tw.intake_defer(s, twin) is True
    head._dual_kv_wait = True
    clock.t += 1.0
    with caplog.at_level(logging.INFO, logger=tw.logger.name):
        out = tw.release_due(s, {twin.rid: twin})
    assert out == [(twin, False)]
    assert not tw.is_deferred(s, twin.rid)
    assert any(tw.REL_SOURCE_UNGRANTED in r.getMessage() for r in caplog.records)


def test_granted_source_still_holds_the_twin(dual, clock):
    """No lost prefix share: a granted, computing source keeps the hold."""
    s, head, twin = _specimen(head_waits=False)
    assert tw.intake_defer(s, twin) is True
    clock.t += 5.0
    assert tw.release_due(s, {twin.rid: twin}) == []
    assert tw.is_deferred(s, twin.rid)


def test_flip_form_unchanged(flip, clock):
    """No dual layout: _dual_kv_wait is never read, the hold as before."""
    s, head, twin = _specimen(head_waits=True)
    assert tw.intake_defer(s, twin) is True
    clock.t += 5.0
    assert tw.release_due(s, {twin.rid: twin}) == []


# ------------------------------------------------- B. follower #1180-W vs K14 stamp
# Review A1: through the REAL entry -- Scheduler.abort_request (class) -> pp_abort.hold_waiting_abort
# -> the real _abort_request_now (PP-SKIPOUT mark_aborts, #1180-W gate, waiting pop) -- on a follower
# stand-in, with PP0's real stamp (pp_abort.pp0_stamp_aborts) and a told of an EARLIER list here.

from flliper.srt.managers.io_struct import AbortReq  # noqa: E402
from flliper.srt.pdflip import abort_match  # noqa: E402
from flliper.srt.pdflip import pp_abort  # noqa: E402

S = SC.Scheduler
RID = "pdflip-0-28"
PP = 3
DUAL_P_ENV = {"FLLIPER_PDFLIP_DUAL_LAYOUT": "1", "FLLIPER_PDFLIP_GROUP": "P",
              "FLLIPER_PDFLIP_DUAL_P_KV_MAX_TOKENS": "196608",
              # the stamp case is under test, not the #1430q/#1470 nets
              "FLLIPER_PDFLIP_DUAL_LATER_TOLD_DROP": "0", "FLLIPER_PDFLIP_DUAL_POP_KEEPS_TWIN": "0"}


class _FReq(types.SimpleNamespace):
    __hash__ = object.__hash__
    __eq__ = object.__eq__


def _freq(rid=RID):
    return _FReq(
        rid=rid, to_finish=None, finished_reason=None, mamba_pool_idx=None,
        req_pool_idx=None, is_retracted=False, kv_committed_freed=True, finished=lambda: False,
        time_stats=types.SimpleNamespace(trace_ctx=types.SimpleNamespace(abort=lambda **k: None)),
    )


def _stage(monkeypatch, pp_rank):
    monkeypatch.setattr(SC, "prepare_abort", lambda req, why: setattr(req, "aborted_why", why))
    monkeypatch.setattr(SC, "release_kv_cache", lambda *a, **k: None)
    monkeypatch.setattr(p_row_authority, "applies", lambda s: True)
    sent = []
    st = types.SimpleNamespace(
        ps=types.SimpleNamespace(pp_rank=pp_rank, pp_size=PP, tp_size=1),
        chunked_req=None, _pending_chunked_abort_req=None, _pending_chunked_abort_delay=0,
        _pp_scheduled_extents=lambda: None, _791c_pp0_drained=False,
        waiting_queue=[], running_mbs=[None] * PP, mbs=[None] * PP,
        running_batch=None, last_batch=None, kv_session_offload=None,
        disaggregation_mode=None, enable_hicache_storage=False,
        grammar_manager=types.SimpleNamespace(abort_requests=lambda r: None),
        _pdflip_abort_dormant_hold=lambda r: 0, _pdflip_d_park_abort=lambda r: 0,
        tree_cache=types.SimpleNamespace(supports_mamba=lambda: False),
        ipc_channels=types.SimpleNamespace(send_to_tokenizer=types.SimpleNamespace(
            send_output=lambda obj, r: sent.append(obj.rid))),
        pp_group=object(), _pp_row_chain_owed=False, _pp_flip_epoch=lambda: 1,
        _pdflip_store_told_armed=True, _pdflip_store_told={}, _pdflip_store_held={},
        _pdflip_force_waiting_abort=set(), sent=sent,
    )
    st._abort_request_now = lambda r: S._abort_request_now(st, r)
    st._pdflip_resolve_waiting_aborts = (
        lambda pp0_drained=False: S._pdflip_resolve_waiting_aborts(st, pp0_drained=pp0_drained)
        if hasattr(S, "_pdflip_resolve_waiting_aborts") else 0)
    st.process_pending_chunked_abort = lambda: S.process_pending_chunked_abort(st)
    return st


@pytest.fixture
def dual_p_env():
    with mock.patch.dict(os.environ, DUAL_P_ENV):
        yield


def _specimen(monkeypatch, stamp):
    """09:09:36: PP0 reads the P-PAUSE abort of pdflip-0-28 (granted, told on the wire at
    09:09:34, NOT admitted -> stamp False; admitted in a launched frame -> True; no stamp ->
    None). PP2 still queues instance 1 and holds its told from that EARLIER list."""
    pp0 = _stage(monkeypatch, 0)
    if stamp is True:
        pp0._pp_launched_batches = {1: types.SimpleNamespace(reqs=[_freq()])}
    fol = _stage(monkeypatch, 2)
    inst1 = _freq()
    fol.waiting_queue = [inst1]
    fol._pdflip_store_told[RID] = 36352
    abort = AbortReq(rid=RID)
    wire = pp_abort.pp0_stamp_aborts(pp0, [abort]) if stamp is not None else [abort]
    assert wire[0].pdflip_pp0_admitted is stamp
    S.abort_request(fol, wire[0])
    return fol, inst1


def _resume(fol):
    """09:09:38 RESUME-UNSTARVE: instance 2 under the SAME rid, its own told; PP0's next frames
    name the rid (for instance 2); the follower's per-pass #1180-W step runs."""
    inst2 = _freq()
    fol.waiting_queue.append(inst2)
    fol._pdflip_store_told[RID] = 36352
    fol._pp_scheduled_extents = lambda: {RID: (36352, 1024)}
    S._pdflip_process_waiting_aborts(fol)
    return inst2


def test_a1_stamp_false_pops_at_receipt_no_hold(monkeypatch, dual_p_env):
    """RED on 1ecda02a58 (PP2 'PDFLIP-PP-WAITING-ABORT held rid=pdflip-0-28')."""
    fol, inst1 = _specimen(monkeypatch, False)
    assert inst1 not in fol.waiting_queue
    assert not getattr(fol, "_pdflip_pending_waiting_aborts", None)
    assert not getattr(fol, pp_abort.HELD_ATTR, None)
    inst2 = _resume(fol)
    assert [r for r in fol.waiting_queue if r.rid == RID] == [inst2], "exactly the new instance"
    assert inst2.to_finish is None


def test_a1_stamp_true_goes_to_the_k14_hold(monkeypatch, dual_p_env):
    fol, inst1 = _specimen(monkeypatch, True)
    assert fol.waiting_queue == [inst1]
    assert RID in (getattr(fol, pp_abort.HELD_ATTR, None) or {})
    assert not getattr(fol, "_pdflip_pending_waiting_aborts", None), "K14, not #1180-W"


def test_a1_unstamped_keeps_the_1180w_hold(monkeypatch, dual_p_env):
    fol, inst1 = _specimen(monkeypatch, None)
    assert fol.waiting_queue == [inst1]
    assert sorted(fol._pdflip_pending_waiting_aborts) == [RID]
    assert not getattr(fol, pp_abort.HELD_ATTR, None)


# ------------------------------------------------- S6. exact rid on abort (#1580 port)


@pytest.fixture
def exact_default(monkeypatch):
    monkeypatch.delenv(abort_match.ENV, raising=False)
    abort_match._reset_for_tests()
    yield
    abort_match._reset_for_tests()


def test_s6_rule(exact_default):
    assert abort_match.rid_hit("pdflip-0-2", "pdflip-0-2")
    assert not abort_match.rid_hit("pdflip-0-20", "pdflip-0-2")
    assert not abort_match.rid_hit("pdflip-0-200", "pdflip-0-2")
    assert abort_match.rid_hit("pdflip-0-2_1", "pdflip-0-2"), "n>1 child rid (io_struct _normalize_rid)"
    assert not abort_match.rid_hit("pdflip-0-3", "pdflip-0-2")
    assert abort_match.rid_hit("abcdef12", "abcd"), "other rid shapes: upstream prefix rule"
    assert abort_match.rid_hit("anything", "")
    assert pp_abort._matches("pdflip-0-2", "pdflip-0-2", False)
    assert not pp_abort._matches("pdflip-0-20", "pdflip-0-2", False)


def test_s6_waiting_abort_spares_the_foreign_rid(monkeypatch, exact_default):
    """RED on 1ecda02a58: the abort of pdflip-0-2 also took the live pdflip-0-20."""
    st = _stage(monkeypatch, 0)
    mine, child, foreign = _freq("pdflip-0-2"), _freq("pdflip-0-2_1"), _freq("pdflip-0-20")
    st.waiting_queue = [mine, child, foreign]
    S.abort_request(st, AbortReq(rid="pdflip-0-2"))
    assert st.waiting_queue == [foreign]


def test_s6_mark_aborts_spares_the_foreign_rid(exact_default):
    from flliper.srt.managers import pp_continuation_seat as seat

    mine, foreign = _freq("pdflip-0-2"), _freq("pdflip-0-20")
    sched = types.SimpleNamespace()
    setattr(sched, seat.DISPLACED_ATTR, [mine, foreign])
    marked = seat.mark_aborts(sched, "pdflip-0-2", False)
    assert foreign not in (marked or [])


def test_s6_switch_off_is_the_prefix_rule(monkeypatch):
    monkeypatch.setenv(abort_match.ENV, "0")
    abort_match._reset_for_tests()
    try:
        assert abort_match.rid_hit("pdflip-0-20", "pdflip-0-2")
    finally:
        abort_match._reset_for_tests()
