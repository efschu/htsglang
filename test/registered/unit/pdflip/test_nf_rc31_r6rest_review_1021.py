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
  halt's ``free`` ("a chunk in flight ends on D") understates what P or D computes afterwards. Strict xfail until fixed."""
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


@pytest.mark.parametrize("free, to_d_now", [(2010, True), (4000, False)])
def test_a2_kurz_rest_goes_to_d_now_halb_rest_waits_for_the_set(monkeypatch, free, to_d_now):
    from test_pdflip_set_scheduler_1017 import _on

    _on(monkeypatch)
    f = _bare_front_for_enqueue()
    now = time.time()
    p = _pending(RID, now)
    F.Front._r6_halt_enqueue(f, p, "D", {"free": free})
    assert int(p.est_uncached) == free and p.d_eligible
    if to_d_now:
        assert p in f._ready_for_d and p not in f.queue and p.d_direct
        assert f.counters["r6_halt_kurz_to_d_now"] == 1
    else:
        assert p in f.queue and p not in f._ready_for_d and not p.d_direct
        assert f.counters["r6_halt_kurz_to_d_now"] == 0


def test_a2_while_p_is_awake_nothing_moves_to_d():
    f = _bare_front_for_enqueue()
    f.awake = "P"
    p = _pending(RID, time.time())
    F.Front._r6_halt_enqueue(f, p, "D", {"free": 2010})
    assert p in f.queue and p not in f._ready_for_d


# ------------------------------------------------------------------------------------------------ A1 (open): D discards the in-flight chunk
@pytest.mark.xfail(strict=True, reason="review A1 open: the R6 halt applies through the plain chunked abort, which releases the "
                                       "request with is_insert=False before the in-flight chunk's result is processed (overlap "
                                       "loop) -- that chunk is not retained, the halt's free understates what P/D computes")
def test_a1_the_applied_r6_stop_retains_what_d_computed():
    from flliper.srt.managers import scheduler as S

    src = inspect.getsource(S.Scheduler.process_pending_chunked_abort)
    halt_at = src.index("Scheduler._pdflip_d_ext_halt(self, req.rid)")
    release = src.rfind("release_kv_cache(", 0, halt_at)
    assert release >= 0
    assert "is_insert=False" not in src[release:src.index(")", release) + 1]
