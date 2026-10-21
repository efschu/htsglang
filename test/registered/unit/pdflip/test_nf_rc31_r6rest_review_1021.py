"""Review NF-RC31-R6REST (21.10., commit c9616da716; SOLL v2 R1/R2/R3/R6/R6k/R7/R11, /spinning/gpu-arb/SCHEDULER-SOLL.md).

Probes and pins of the independent review (report /spinning/gpu-arb/deskq/done/nf-rc31-r6rest-review-1021.md):
- A3 (R11 rank unanimity): ``halted_free`` counts a halt only once EVERY rank that holds the grant's extend record has halted.
- A2 (R2): a stopped prefill that goes back to D kurz by its halted rest goes to D's admission line at once, also while a set
  collects; a halb rest stays queued for the set's verdict.
- Hold cases the author's tests do not cover: the set's T_halb ends while the stop is pending (the timer verdict waits), a second
  halb member arrives while the stop is pending (REROUTE_P on the halted rest, the stopped prefill rides it), the prefill had no
  rest left at the halt (rest 0: back to D, no hang).
- A1 (open, boot-blocking): ``process_pending_chunked_abort`` releases the aborted chunked request with ``is_insert=False`` -- a
  chunk in flight at the halt is NOT retained in D's tree (its result is processed after the abort in the overlap loop), so the
  halt's ``free`` ("a chunk in flight ends on D") understates what P or D computes afterwards. FIXED (planner follow-up): the R6
  stop of a TP chunked prefill is applied in the recv step RETAINING (``Scheduler._pdflip_halt_retain``: the in-flight chunk lands,
  ``release_req(retain=True)``), the halt record = length minus tree-held on every rank; the discarding fallback counts the
  tree-held prefix. The strict xfail is now a passing test plus a behavioural one on three ranks.
- A4 (planner decision after SOLL R1 "außer der Einstufung bei Ankunft"): the stopped prefill keeps its arrival class and goes
  straight back to D's admission as the continuation of its D verdict, whatever its halted rest."""
from __future__ import annotations

import inspect
import logging
import os
import sys
import time
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.environ import envs  # noqa: E402
from flliper.srt.pdflip import d_extend_beacon as dxb  # noqa: E402
from flliper.srt.pdflip import front as F  # noqa: E402

RID = "pdflip-6-401"
SET1 = "pdflip-6-403"
SID = 7
X, HX = 7370, 3685


def _read(d):
    return dxb.read_group_full(str(d), "D", SID, session_of=lambda pid: SID)


# ------------------------------------------------------------------------------------------------ A3 rank unanimity (R11)
def test_a3_a_halt_of_one_rank_of_three_is_no_halt_yet(tmp_path):
    ws = [dxb.DExtendBeacon(str(tmp_path / ("D-pid%d.bin" % (r + 1))), clock_ns=lambda: 2 * 10**9) for r in range(3)]
    for w in ws:
        w.announce_halt()
        w.note_recv(rid=RID)
        w.note_extend(rids=[RID], rests=[6550], extend_lens=[2240])
    assert ws[0].note_halt(rid=RID)
    assert dxb.halted_free(_read(tmp_path), RID, 1.0) is None          # ranks 1, 2 may still launch a chunk
    assert ws[1].note_halt(rid=RID)
    assert dxb.halted_free(_read(tmp_path), RID, 1.0) is None
    assert ws[2].note_halt(rid=RID)
    assert dxb.halted_free(_read(tmp_path), RID, 1.0) == 4310          # every rank applied the abort


def test_a3_a_stub_reading_without_per_rank_values_keeps_the_merged_entry():
    reading = {dxb.rid_key(RID): (2 * 10**9, 4310, 4310), dxb.halt_key(RID): (2 * 10**9, 4310, 2010)}
    assert dxb.halted_free(reading, RID, 1.0) == 2010


# ------------------------------------------------------------------------------------------------ front harness (author's chain)
def _chain(monkeypatch, tmp_path, *, waited=0.1):
    from test_pdflip_scheduler_soll_1017 import _r6rest_chain

    now, f, ws = _r6rest_chain(monkeypatch, tmp_path)
    if waited != 0.1:
        t, u, x = f._set_st()["shorts"][SET1]
        f._set_st()["shorts"][SET1] = (now - waited, u, x)
    return now, f, ws


def test_hold_the_set_timer_ends_while_the_stop_is_pending_the_d_verdict_waits_then_d(monkeypatch, caplog, tmp_path):
    """R3 + R6: T_halb (4.5 s) ends while D has not applied the stop -- no D_AFTER_TIMER on an unknown rest (it could still be
    REROUTE_P); after the halt (2010, S = 5796 < X): the set goes to D (R3), the stopped prefill back to D (R7), no flip."""
    now, f, ws = _chain(monkeypatch, tmp_path, waited=4.0)
    with envs.FLLIPER_PDFLIP_DECIDER.override(True), caplog.at_level(logging.INFO, logger="pdflip.front"):
        assert f._decode_collect(now) == "hold"                        # R6-HALT-SENT
        assert f._decode_collect(now + 1.0) == "hold"                  # timer over, stop pending: held
        assert f.counters["set_d_after_timer"] == 0
        for w in ws:
            w.note_extend(rids=[RID], rests=[4310], extend_lens=[2300])
            assert w.note_halt(rid=RID)
        assert f._decode_collect(now + 1.1) == "D"
    assert f.counters["set_d_after_timer"] == 1 and f.counters["set_reroute_p"] == 0 and f.counters["r6_halt_to_d"] == 1
    assert any("R6-HALT pending" in m for m in caplog.messages if "SET-VERDICT" in m) or f.counters["set_r6_halt_hold"] >= 2


def test_hold_a_second_halb_member_arrives_while_the_stop_is_pending_reroute_on_the_halted_rest(monkeypatch, caplog, tmp_path):
    """R1 + R3 + R6 + R1a: a second halb member (3700) joins while the stop is pending -> still held; after the halt (2010):
    S = 3786 + 3700 + 2010 = 9496 > X -> REROUTE_P, both members AND the stopped prefill in go_p."""
    now, f, ws = _chain(monkeypatch, tmp_path)
    with envs.FLLIPER_PDFLIP_DECIDER.override(True), caplog.at_level(logging.INFO, logger="pdflip.front"):
        assert f._decode_collect(now) == "hold"
        f._set_st()["shorts"]["pdflip-6-404"] = (now + 0.02, 3700, X)
        assert f._decode_collect(now + 0.05) == "hold"
        assert f.counters["set_reroute_p"] == 0 and f.counters["set_flip_p"] == 0
        for w in ws:
            w.note_extend(rids=[RID], rests=[4310], extend_lens=[2300])
            assert w.note_halt(rid=RID)
        assert f._decode_collect(now + 0.1) == "P"
    assert f.counters["set_reroute_p"] == 1 and f.counters["r6_halt_to_p"] == 1
    assert {RID, SET1, "pdflip-6-404"} <= set(f._set_st()["go_p"])
    sv = [m for m in caplog.messages if "SET-VERDICT" in m and "verdict=REROUTE_P" in m]
    assert sv and "S=9496" in sv[-1], sv


def test_rest_zero_the_prefill_had_launched_its_last_chunk_back_to_d_at_u0_no_hang(monkeypatch, caplog, tmp_path):
    """The prefill launched its LAST chunk (4310) before the abort landed: halt free 0 -> S = 3786 < X, no flip; the request
    goes back to D at u = 0 (kurz) -- decided at once, no liveness wait."""
    now, f, ws = _chain(monkeypatch, tmp_path)
    with envs.FLLIPER_PDFLIP_DECIDER.override(True), caplog.at_level(logging.INFO, logger="pdflip.front"):
        assert f._decode_collect(now) == "hold"
        for w in ws:
            w.note_extend(rids=[RID], rests=[4310], extend_lens=[4310])
            assert w.note_halt(rid=RID)
        assert f._decode_collect(now + 0.1) == "hold"                  # R3: the set collects on, S = 3786
    assert f.counters["r6_halt_to_d"] == 1 and f.counters["r6_halt_stale"] == 0 and f.counters["set_reroute_p"] == 0
    assert any("R6-HALT-STAYS-ON-D rid=%s halted_free=0" % RID in m for m in caplog.messages)


# ------------------------------------------------------------------------------------------------ A2 (R2) return path
def _bare_front_for_enqueue():
    from test_pdflip_arrival_seat_rule_0929 import _front as _fake_front

    f = _fake_front(running=["dec"], n=6, kv={"available": 400000, "evictable": 0})
    f.tp_prefill_max_tokens = X
    f._sync_batch_gate = lambda: None
    f._kick_controller = lambda *a, **k: None
    return f


def _pending(rid, now):
    import asyncio as _a

    return F.Pending(rid, "/v1/messages", {}, "t", now, _a.new_event_loop().create_future(), est_prompt=157590,
                     est_uncached=6550)


@pytest.mark.parametrize("free", [2010, 4000])
def test_a2_a4_the_stopped_prefill_continues_on_d_at_once_with_its_arrival_class(monkeypatch, free):
    """A2 + A4 (planner 21.10., SOLL R1 "außer der Einstufung bei Ankunft", R6k "bei Ankunft", R6, R7): whatever its halted rest
    (kurz-sized 2010 or halb-sized 4000), the stopped prefill is no new arrival and joins no set -- it goes straight back into
    D's admission line (QUEUED-SHORT: seat, X-SUM price) as the continuation of its earlier D verdict; its class is NOT derived
    from the halted rest (the arrival's class stays)."""
    from test_pdflip_set_scheduler_1017 import _on

    _on(monkeypatch)
    f = _bare_front_for_enqueue()
    F.Front._r6k_note_class(f, RID, "halb", "arrival", HX)
    now = time.time()
    p = _pending(RID, now)
    F.Front._r6_halt_enqueue(f, p, "D", {"free": free})
    assert int(p.est_uncached) == free and p.d_eligible
    assert p in f._ready_for_d and p not in f.queue and p.d_direct
    assert f.counters["r6_halt_to_d_now"] == 1
    assert f._r6k_class[RID][0] == "halb"


def test_a2_while_p_is_awake_nothing_moves_to_d():
    f = _bare_front_for_enqueue()
    f.awake = "P"
    p = _pending(RID, time.time())
    F.Front._r6_halt_enqueue(f, p, "D", {"free": 2010})
    assert p in f.queue and p not in f._ready_for_d


# ------------------------------------------------------------------------------------------------ A1: the R6 stop RETAINS what D computed
def test_a1_the_applied_r6_stop_retains_what_d_computed():
    """A1 (structure): the R6 stop of the running chunked prefill is applied in the recv step by ``_pdflip_halt_retain`` -- the
    in-flight batch LANDS first, then the release RETAINS (``_release_outside_chunk`` = ``release_req(retain=True)``,
    is_insert=True), never upstream's ``is_insert=False`` before the in-flight result. Upstream's discarding path stays only
    for PP / a reasoned abort / the switch off, and its halt counts the tree-held prefix, not the launch-based free."""
    from flliper.srt.managers import scheduler as S

    ab = inspect.getsource(S.Scheduler._abort_request_now)
    assert ab.index("Scheduler._pdflip_halt_retain(self, chunked_req, recv_req)") < ab.index(
        "self._pending_chunked_abort_req = chunked_req")
    hr = inspect.getsource(S.Scheduler._pdflip_halt_retain)
    assert hr.index("d_park_runtime._land_inflight(self, keep_chunked=req)") < hr.index(
        "d_park_runtime._release_outside_chunk(self, req)") < hr.index("Scheduler._pdflip_d_ext_halt(")
    assert "is_insert=False" not in hr.split('"""', 2)[2]
    pc = inspect.getsource(S.Scheduler.process_pending_chunked_abort)
    assert "_held_a1 = len(req.prefix_indices)" in pc and "held_free=(max(0, _ntok_a1 - _held_a1)" in pc
    import inspect as _i
    from flliper.srt.pdflip import d_park_runtime as dpr

    assert "retain=True" in _i.getsource(dpr._release_outside_chunk)


N_6401, RET_6401 = 157590, 155521        # live: PDFLIP-D-PARK RETAINED retained=155521 of 157590 (151040 + 2240 + 2300, page/bigram)


def _rank_sched(tmp_path, r, order):
    """One TP rank of D at 21:15:36,9..: chunk 1 (2240) done, chunk 2 (2300) launched and IN FLIGHT (overlap: its result is
    still in ``result_queue``), 6-401 the chunked request."""
    import collections

    w = dxb.DExtendBeacon(str(tmp_path / ("D-pid%d.bin" % (r + 1))), clock_ns=lambda: 2 * 10**9)
    w.announce_halt()
    w.note_recv(rid=RID)
    w.note_extend(rids=[RID], rests=[6550], extend_lens=[2240])
    w.note_extend_done(rids=[RID])
    w.note_extend(rids=[RID], rests=[4310], extend_lens=[2300])
    req = types.SimpleNamespace(rid=RID, origin_input_ids=list(range(N_6401)), output_ids=[], inflight_middle_chunks=1,
                                req_pool_idx=7, to_finish=None,
                                time_stats=types.SimpleNamespace(trace_ctx=types.SimpleNamespace(abort=lambda **k: None)))
    last = types.SimpleNamespace(forward_mode=types.SimpleNamespace(is_extend=lambda: True), reqs=[req])
    last.filter_batch = lambda chunked_req_to_exclude=(): last.reqs.__setitem__(
        slice(None), [q for q in last.reqs if q not in chunked_req_to_exclude])
    last.is_empty = lambda: not last.reqs
    sent = []

    def _process(batch, result):
        order.append(("land", r))
        req.inflight_middle_chunks -= 1             # the middle chunk's result: its accounting drained
        w.note_extend_done(rids=[RID])              # chunk 2's forward ended: the beacon's after-rest

    sched = types.SimpleNamespace(
        pdflip_d_ext_beacon=w, ps=types.SimpleNamespace(pp_size=1, pp_rank=0), enable_overlap=True, last_batch=last,
        result_queue=collections.deque([(last, object())]), process_batch_result=_process,
        running_batch=types.SimpleNamespace(is_empty=lambda: True), chunked_req=req, _pending_chunked_abort_req=None,
        ipc_channels=types.SimpleNamespace(send_to_tokenizer=types.SimpleNamespace(send_output=lambda o, q: sent.append(o))))
    return sched, req, sent


def test_a1_live_6401_the_halt_lands_chunk2_retains_it_and_reports_length_minus_held_on_every_rank(monkeypatch, tmp_path):
    """A1 (behaviour) + A3 + R11, live 6-401: on every TP rank the abort lands the in-flight chunk 2 BEFORE the release, the
    release RETAINS (the tree then holds 155521 = 151040 + 2240 + 2300 page/bigram-cut), and the halt record says the length
    minus what the tree holds: 157590 - 155521 = 2069 -- what P computed live (2070 from the page 155520). Until every rank has
    halted the front reads None (A3); then the same number from all three."""
    from flliper.srt.managers import scheduler as S
    from flliper.srt.managers.io_struct import AbortReq
    from flliper.srt.pdflip import d_park_read, d_park_runtime as dpr

    order = []

    def _release(sched, req):
        assert req.inflight_middle_chunks == 0, "released before the in-flight chunk landed"
        order.append(("retain", id(sched)))
        setattr(req, d_park_read.RETAINED_ATTR, RET_6401)
        return req

    monkeypatch.setattr(dpr, "_release_outside_chunk", _release)
    monkeypatch.setattr(S, "prepare_abort", lambda req, msg, **k: setattr(req, "aborted", msg))
    ranks = [_rank_sched(tmp_path, r, order) for r in range(3)]
    read = lambda: dxb.read_group_full(str(tmp_path), "D", 7, session_of=lambda pid: 7)  # noqa: E731
    for i, (sched, req, sent) in enumerate(ranks):
        assert S.Scheduler._pdflip_halt_retain(sched, req, AbortReq(rid=RID))
        assert sched.chunked_req is None and sched.last_batch is None and not sched.result_queue
        assert getattr(req, "aborted", None) == "Aborted" and len(sent) == 1 and sent[0].rid == RID
        got = dxb.halted_free(read(), RID, 1.0)
        assert got == (N_6401 - RET_6401 if i == 2 else None), (i, got)        # A3: only once all three halted
    assert [o[0] for o in order] == ["land", "retain"] * 3
    entry = read()[dxb.halt_key(RID)]
    assert [int(rk[2]) for rk in entry.ranks] == [N_6401 - RET_6401] * 3           # R11: the same number on every rank


def test_a1_not_applicable_falls_back_to_upstreams_pending_abort(monkeypatch, tmp_path):
    """Not group D (no beacon), PP, an abort with a reason, a prefix-matched rid, the switch off: nothing lands, nothing is
    released -- upstream's pending chunked abort applies (its halt counts the tree-held prefix)."""
    from flliper.srt.managers import scheduler as S
    from flliper.srt.managers.io_struct import AbortReq

    order = []
    sched, req, _sent = _rank_sched(tmp_path, 0, order)
    assert not S.Scheduler._pdflip_halt_retain(sched, req, AbortReq(rid="pdflip-6-40"))
    assert not S.Scheduler._pdflip_halt_retain(sched, req, AbortReq(rid=RID, finished_reason={"type": "abort"}))
    sched.ps.pp_size = 3
    assert not S.Scheduler._pdflip_halt_retain(sched, req, AbortReq(rid=RID))
    sched.ps.pp_size = 1
    with envs.FLLIPER_PDFLIP_D_HALT_RETAIN.override(False):
        assert not S.Scheduler._pdflip_halt_retain(sched, req, AbortReq(rid=RID))
    sched.pdflip_d_ext_beacon = None
    assert not S.Scheduler._pdflip_halt_retain(sched, req, AbortReq(rid=RID))
    assert order == [] and sched.chunked_req is req
