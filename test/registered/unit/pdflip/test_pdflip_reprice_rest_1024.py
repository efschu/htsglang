"""REPRICE-REST-1024 (SOLL R2 R7; Auftrag #224): the two misprices SHORT-FLIP-REPRICE-1024 (#220) left open.

CASE 1 -- pdflip-26-232 (27B nvfp4duo 93eae72472, boot boot_weg2_dkr27bnvfp4duorebarbar1duo10240126_93eae72472_1024_012625):
a parked decode, d_extent 26844, P computed 220.

  D 02:05:59 TP0  #248 WAKE-READ issued=1 ['pdflip-26-232']                     (settle clock ~02:06:00.7)
  D 02:06:20 TP0  #1157 PREFETCH REAPED req=pdflip-26-232 requested_pages=146176 hit_pages=146176 completed=119552
                  elapsed=21.30s
  D 02:06:20 TP0  PDFLIP-SHORT-READ HELD rid=pdflip-26-232 completed=119552 of 146176 pages hit=146176
  D 02:06:20 TP0  HiCache prefetch INCOMPLETE ... matched=18138 loaded=101414 ... deliverable=146176 shortfall=26624
  D 02:06:20 TP0  #1324 STORE READ HELD-CUT rid=pdflip-26-232 delivered=119552 deliverable=146176 hit_tokens=146176
                  cut_cause=late_complete
  D 02:06:20 TP0  #1456 HOLD-REFETCH rid=pdflip-26-232 n=1 reason=deferred verdict=issued
  D 02:06:21 TP0  #1471 SETTLE-RELEASE rid=pdflip-28-250 state=reading lapsed=True held_after_wake_s=20.5 (same tick)
  D 02:06:21 TP0  PDFLIP W50-REROUTE rid=pdflip-26-232 d_extent=26844 X=12288 reason=store_short_after_flip
  D 02:06:42 TP0  PDFLIP-READ-STAGES req=pdflip-26-232 pages=26624 ...           (the re-read landed the rest)
  front           RESUME-VIA-P CLASS u=26844 class=gross -> SET-VERDICT FLIP_P R4; USAGE-TRUE p_computed=220

THE ROOT: the #1471 settle bound (20 s) lapsed while the re-read of a span the store HELD was in flight, and
D-NORECOMPUTE priced the request off the superseded first record: 146396 - 119552 = 26844 instead of
146396 - 146176 = 220 (D-RESUME-EXTEND under the front's halbX). Not #220's class: that record's span starts at 0.
THE FIX: settle_writer.held_read_holds_lapse -- the bound waits for that re-read once.

CASE 2 -- pdflip-22-194 (same boot, x_refusal_midstream): D-PROBE device=124928 (the finish-track Mamba state of
pdflip-22-176, finished 3 s earlier, never backed to the host), admission 1 s later found no state at 124928
(``ANCHOR-ONLY-READ ... how=store_miss_unsecured``), the X gate priced 125811 - 119552 = 6259 > front_x 4097 (W31 by
name, SOLL R7 backstop). Not a pricing arithmetic defect: the probe's number was exact at its read, the anchor was lost
between probe and admission -- the X gate refused it by name (R7 backstop) and it rode a flip pdflip-22-196 (u 6413 >= X)
had already earned alone. No code fix here (Bericht: deskq/done/reprice-rest-1024.md).
"""
from __future__ import annotations

import logging
import os
import sys
import time
import types

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_pdflip_store_short_fallback_rc12y as rc12y  # noqa: E402

from flliper.srt.managers import scheduler as sched_mod  # noqa: E402
from flliper.srt.mem_cache.hicache_storage import PrefetchOutcome  # noqa: E402
from flliper.srt.pdflip import d_norecompute as dn  # noqa: E402
from flliper.srt.pdflip import front_x as fx  # noqa: E402
from flliper.srt.pdflip import resume_via_p as rvp  # noqa: E402
from flliper.srt.pdflip import settle_writer as sw  # noqa: E402
from flliper.srt.pdflip import short_read as sr  # noqa: E402
from flliper.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=10, suite="stage-a-pdflip-unit")

#: pdflip-26-232, the metal numbers
N = 146396            # prompt 138228 + 8168 decoded (RESUME-VIA-P tokens=146396)
FIRST = 119552        # the wake read's materialized (matched 18138 + loaded 101414), anchor-cut
HELD = 146176         # deliverable = hit: the store held the whole park span
REST = HELD - FIRST   # 26624: what the in-flight re-read landed (READ-STAGES pages=26624)
X = 12288             # D's static X (W50-REROUTE X=12288)
FRONT_X, HALFX = 4096, 2048
RID = "pdflip-16-42"  # the rc12y harness's rid


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("FLLIPER_PDFLIP_STORE_SHORT_TAIL", "0")
    monkeypatch.delenv(rc12y.MAX_CYCLES_ENV, raising=False)
    monkeypatch.setenv("FLLIPER_HICACHE_ARENA_DIR", str(tmp_path))
    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "D")
    (tmp_path / "handoff").mkdir()
    rvp.take_requests(rvp.needs_p_dir())
    sr.forget(RID)
    yield
    rvp.take_requests(rvp.needs_p_dir())
    sr.forget(RID)


def _first_record():
    # HiCache prefetch INCOMPLETE ... completed_synced=119552 matched=18138 loaded=101414 deliverable=146176
    rec = PrefetchOutcome(FIRST - 18138, hit_tokens=HELD, probed=True, matched=18138, deliverable=HELD,
                          synced=FIRST, cut_cause="late_complete")
    rec.span_start = 0    # the park read [0, 146177): #220's span start is 0 here (base 0, review (6))
    return rec


def _second_record():
    # the re-read from the anchor 119552 (FA PREFETCH-FROM-ANCHOR anchor=119552) landed all 26624 pages
    rec = PrefetchOutcome(REST, hit_tokens=REST, probed=True, matched=0, deliverable=REST, synced=REST)
    rec.span_start = FIRST
    return rec


def _settle_sched():
    """The real ``Scheduler._pdflip_post_wake_settle_tick`` / ``_pdflip_refetch_one`` / ``_pdflip_note_store_shortfall``
    on the rc12y stand-in, 26-232 parked after the wake with its first (short, held) record."""
    S = sched_mod.Scheduler
    base, r = rc12y._sched(N, 0, 1, X)
    s = S.__new__(S)
    for k, v in vars(base).items():
        if not callable(v) or k in ("tree_cache", "server_args", "ipc_channels"):
            setattr(s, k, v)
    s.ps = types.SimpleNamespace(tp_size=1, pp_size=1, tp_rank=0)
    s.page_size = 64
    s.pdflip_dormant_hold = []
    s._pdflip_wake_seq = 1
    s.pdflip_d_parked = []
    s.tree_cache.prefetch_loaded_tokens_by_reqid = {r.rid: _first_record()}
    s.tree_cache.ongoing_prefetch = {}
    s.tree_cache.check_prefetch_progress = lambda rid: rid not in s.tree_cache.ongoing_prefetch
    s.tree_cache.cache_controller = types.SimpleNamespace(pdflip_hold_rids=set(), prefetch_rate_limited=lambda: False,
                                                          host_role="staging")
    calls = []

    def _issue(req, limit_tokens=None):
        # #1456 HOLD-REFETCH ... verdict=issued: the re-read registers from the anchor and runs
        req._prefetch_registered_prefix_len = FIRST
        req._prefetch_span_tokens = REST
        s.tree_cache.ongoing_prefetch[req.rid] = object()
        calls.append(req.rid)
        return "issued"

    s._prefetch_kvcache = _issue
    r._prefetch_span_tokens = HELD + 1          # #1068 PREFETCH DEFERRED ... span=146177
    r.output_ids = list(range(N - 138228))      # a parked decode: 8168 tokens produced
    r.stream, r.multimodal_inputs = True, None
    setattr(r, fx.ATTR_X, FRONT_X)              # FRONT-X 02:02:20.478 front_x=4096 (carried, H16V A3)
    setattr(r, fx.ATTR_HALFX, HALFX)
    r._1471_since = time.monotonic() - 1.0
    r._1471_short = False
    r._1456_n, r._1456_last = 0, 0.0
    s.waiting_queue = []
    s.pdflip_post_wake_settle = [r]
    # PDFLIP-SHORT-READ HELD: the reap's group-MIN hit covered the span (page_size 1 on that line)
    sr.note_reap(r.rid, HELD, HELD, FIRST, 1)
    return s, r, calls


def _tick(s, r):
    r._1471w_t = 0.0
    return s._pdflip_post_wake_settle_tick()


def _to_the_lapse(s, r):
    """02:06:20: the first record is seen short (held) and the re-read is issued; 02:06:21: the bound lapses (20.5 s)."""
    _tick(s, r)
    assert r._pdflip_store_delivered == FIRST and r._1456_n == 1 and r.rid in s.tree_cache.ongoing_prefetch
    r._1471_since = time.monotonic() - 20.5
    return _tick(s, r)


def _land(s, r):
    """02:06:42: the re-read terminates whole."""
    s.tree_cache.ongoing_prefetch.pop(r.rid, None)
    s.tree_cache.prefetch_loaded_tokens_by_reqid[r.rid] = _second_record()


# ---------------------------------------------------------------------------------------------- case 1: 26-232


def test_26_232_the_lapse_waits_for_the_held_reread_and_d_extends_the_220_token_tail(caplog):
    s, r, calls = _settle_sched()
    with caplog.at_level(logging.INFO):
        released = _to_the_lapse(s, r)
        assert released == 0 and s.pdflip_post_wake_settle == [r] and s.waiting_queue == []
        assert not rvp.take_requests(rvp.needs_p_dir()), "no P leg priced off the superseded record"
        assert any("#1471h SETTLE-HELD-READ rid=" + r.rid in m for m in caplog.messages)
        _land(s, r)
        released = _tick(s, r)
    assert calls == [r.rid], "exactly the metal's one re-read"
    assert released == 1 and s.waiting_queue == [r]
    assert r._pdflip_store_delivered == HELD and dn.d_would_compute(r) == N - HELD == 220
    assert any("D-RESUME-EXTEND rid=" + r.rid in m for m in caplog.messages), "SOLL R2/R7: D takes the kurz tail"
    assert not rvp.take_requests(rvp.needs_p_dir()), "no flip for 220 tokens"


def test_mutant_without_the_hold_the_metal_26844_reappears():
    """Danger direction OVER-pricing -> an unneeded flip: the bound lapses on the in-flight read, D-NORECOMPUTE prices the
    first record, writes a needs-p with d_extent 26844 (> X: the front classes it gross, R4 FLIP_P on the metal)."""
    orig = sw.held_read_holds_lapse
    sw.held_read_holds_lapse = lambda req, state: False
    try:
        s, r, _calls = _settle_sched()
        _to_the_lapse(s, r)
    finally:
        sw.held_read_holds_lapse = orig
    recs = rvp.take_requests(rvp.needs_p_dir())
    assert s.waiting_queue == [] and len(recs) == 1 and recs[0]["d_extent"] == N - FIRST == 26844


def test_the_hold_is_spent_once_the_awaited_read_answered():
    """Bounded: a further re-issue counts past the armed read -> the next lapse releases (on the newer record)."""
    s, r, _calls = _settle_sched()
    _to_the_lapse(s, r)
    assert r._1471h_n == 1
    r._1456_n = 2                      # the awaited read answered short and the settle re-issued
    assert sw.held_read_holds_lapse(r, "reading") is False


@pytest.mark.parametrize("held,state,expect", [
    (True, "reading", True), (True, "reissued", True),
    (True, "wait", False), (True, "complete", False), (True, "budget", False),
    (False, "reading", False),          # store did not hold the span: a writer may be missing, the bound stands
])
def test_the_hold_is_narrow(held, state, expect):
    r = types.SimpleNamespace(rid="pdflip-x-1", _1456_n=1)
    sr.forget(r.rid)
    if held:
        sr.note_reap(r.rid, 100, 100, 40, 1)
    else:
        sr.note_reap(r.rid, 100, 60, 40, 1)
    try:
        assert sw.held_read_holds_lapse(r, state) is expect
    finally:
        sr.forget(r.rid)


def test_a_new_wake_rearms_the_hold():
    r = types.SimpleNamespace(rid="pdflip-x-2", _1456_n=3)
    sr.note_reap(r.rid, 100, 100, 40, 1)
    try:
        assert sw.held_read_holds_lapse(r, "reading") and r._1471h_n == 3
        r._1456_n = 4
        assert not sw.held_read_holds_lapse(r, "reading")
        sw.reset_for_wake(r)
        assert sw.held_read_holds_lapse(r, "reading") and r._1471h_n == 4
    finally:
        sr.forget(r.rid)


def test_a_held_read_that_lands_inside_the_bound_is_untouched(caplog):
    """No lapse, no hold: the 'complete' release prices the landed record exactly as before."""
    s, r, _calls = _settle_sched()
    _tick(s, r)
    _land(s, r)
    with caplog.at_level(logging.INFO):
        assert _tick(s, r) == 1
    assert not any("SETTLE-HELD-READ" in m for m in caplog.messages)
    assert dn.d_would_compute(r) == 220 and not rvp.take_requests(rvp.needs_p_dir())
