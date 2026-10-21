"""Second review of NF-RC30-R6STALE: the Auflagen fix d6019301de (A1 COMMITTED marker + front hold, A2 grant-filtered rank merge,
A3 R1u member share of a launched chunk, A4 R6-NOT-STARTED). SOLL v2 R3 R4 R6 R7 R8 R11. Report: deskq/done/
nf-rc30-r6stale-review-1021.md, section "Nachtrag 2".

Green: the hold delays a state-based flip by one tick and never loses it (R4 "company" is a state, not an edge); the overlap
order (done record written after the commit keeps the marker; the previous batch's result after the next launch keeps the exact
free); uneven n_launch over ranks with a lagging TP0.
Strict xfail: B1 -- the stale guard keys its first sighting on (rid, epoch) and never forgets it, so a LATER commit of the same
rid in the same epoch (next chunk boundary, a #679 park-in-place, >5 s after the first sighting) is named R6-COMMIT-STALE at
once, holds nothing and counts nothing (danger direction "flip missing" + a false alarm of a line that "darf nie kommen").
"""
from __future__ import annotations

import logging
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.environ import envs  # noqa: E402
from flliper.srt.pdflip import d_extend_beacon as dxb  # noqa: E402
from flliper.srt.pdflip import front as F  # noqa: E402
from test_nf_r6_rest_exakt_1020 import _front_live, RID_K  # noqa: E402
from test_pdflip_set_scheduler_1017 import _on  # noqa: E402

SID = 7


def _end_and_commit(f, rid=RID_K):
    ws, ev = f._test_running[rid]
    ev["done"] = True
    for w in ws:
        w.poll_done()
        w.commit_next(rids=[rid])
    return ws


def _launch(ws, rid, rest, ext):
    for w in ws:
        w.note_extend(rids=[rid], rests=[rest], extend_lens=[ext])


def test_hold_delays_an_r4_company_flip_by_one_tick_never_loses_it(monkeypatch, caplog, tmp_path):
    """R4: a big member (8000 at X 7016, waited 1 s) and a halb arrival (3600) -> flip now. In that tick a running halb D prefill's
    next chunk is COMMITTED -> the verdict is held (COLLECT) and ``seen`` advances. R4's company rule is a state (n >= 2), so the
    next tick (start record in: free 1904) flips -- delay = one tick, no lost trigger."""
    _on(monkeypatch)
    now = time.time()
    f = _front_live(now, tmp_path, member_u=8000, ended=False, rest_start=6000, ext_len=2048, x=7016, klass="halb", waited=1.0)
    f._set_st()["shorts"]["pdflip-12-280"] = (now - 0.05, 3600, 7016)
    ws = _end_and_commit(f)
    with envs.FLLIPER_PDFLIP_DECIDER.override(True), caplog.at_level(logging.INFO, logger="pdflip.front"):
        assert f._decode_collect(now) == "hold"
        assert f.counters["set_r6_commit_hold"] >= 1
        _launch(ws, RID_K, 6000 - 2048, 2048)
        assert f._decode_collect(now + 0.2) == "P"


def test_overlap_order_done_after_commit_keeps_marker_result_after_launch_keeps_exact_free(tmp_path):
    """event_loop_overlap: recv -> commit -> poll (chunk k's done) -> build -> resolve -> poll -> run_batch (chunk k+1 start)
    -> pop_and_process (chunk k's result, already polled or not). The done record between commit and launch must stay COMMITTED;
    the previous batch's result after the launch must not overwrite the exact free of chunk k+1."""
    w = dxb.DExtendBeacon(str(tmp_path / "D-pid1.bin"), clock_ns=lambda: 10**9)
    read = lambda: dxb.read_group_full(str(tmp_path), "D", SID, session_of=lambda pid: SID)[dxb.rid_key("r")]  # noqa
    w.note_recv(rid="r")
    w.note_extend(rids=["r"], rests=[6000], extend_lens=[2000])
    w.arm_done(key=1, rids=["r"], ready=lambda: False)          # chunk 1's event not polled yet
    w.commit_next(rids=["r"])
    assert read()[2] == dxb.COMMITTED
    w.resolve_commit(launched=["r"])
    w.note_extend(rids=["r"], rests=[4000], extend_lens=[2000])  # chunk 2 launched before chunk 1's done
    assert read()[2] == 2000
    w.note_extend_done(rids=["r"], key=1)                       # chunk 1's result after the launch
    assert read()[1:] == (4000, 2000)


def test_uneven_n_launch_lagging_tp0_the_most_launched_rank_decides(tmp_path):
    """Measured skew: TP0 ends its chunk 110-130 ms after TP1/TP2 (host). The launch of chunk k+1 is one iteration on all ranks
    (collective recv). TP1 launched chunk 2 (n=2, free 0), TP0 still shows chunk 1 done (n=1, free 784), TP2 committed (n=1):
    the rank with the most launches decides -> 0 (chunk 2 is fixed on all ranks)."""
    ws = [dxb.DExtendBeacon(str(tmp_path / ("D-pid%d.bin" % (r + 1))), clock_ns=lambda: 10**9) for r in range(3)]
    for w in ws:
        w.note_recv(rid="r")
        w.note_extend(rids=["r"], rests=[3280], extend_lens=[2496])
        w.note_extend_done(rids=["r"])
    ws[1].commit_next(rids=["r"])
    ws[1].note_extend(rids=["r"], rests=[784], extend_lens=[784])
    ws[2].commit_next(rids=["r"])
    reading = dxb.read_group_full(str(tmp_path), "D", SID, session_of=lambda pid: SID)
    assert dxb.free_of(reading, "r", 0.5) == 0 and dxb.rest_of(reading, "r", 0.5) == 784


@pytest.mark.xfail(strict=True, reason="B1 (review 2): R6-COMMIT-STALE keyed on the first sighting of (rid, epoch), never reset "
                                       "-- a later commit of the same rid >5 s later is stale at once")
def test_b1_a_later_commit_of_the_same_rid_is_not_stale(monkeypatch, caplog, tmp_path):
    """Chunk boundary 1->2 seen COMMITTED at t (resolved by chunk 2's start), boundary 2->3 seen COMMITTED at t+6 s (narrowed
    chunks after a wake: '#794 GROUP-NARROWED ... to 64'; or a #679 park-in-place that re-commits every D iteration). The second
    marker is fresh (one D iteration old): it must hold, not be named stale."""
    _on(monkeypatch)
    now = time.time()
    f = _front_live(now, tmp_path, member_u=3600, ended=False, rest_start=9000, ext_len=2048, x=7016, klass="halb", waited=1.0)
    ws = _end_and_commit(f)
    with envs.FLLIPER_PDFLIP_DECIDER.override(True):
        assert f._dc_carried_running() == 0 and f.counters["r6_commit_pending"] == 1
        _launch(ws, RID_K, 9000 - 2048, 2048)                    # chunk 2 starts: marker resolved
        assert f._dc_carried_running() == 9000 - 4096
    t6 = now + 6.0
    monkeypatch.setattr(F.time, "time", lambda: t6)
    ws_ev = f._test_running[RID_K][0]
    for w in ws_ev:
        w.note_extend_done(rids=[RID_K])                         # chunk 2 ends
        w.commit_next(rids=[RID_K])                              # boundary 2->3: fresh marker
    with envs.FLLIPER_PDFLIP_DECIDER.override(True), caplog.at_level(logging.INFO, logger="pdflip.front"):
        assert f._dc_carried_running() == 0
    assert f.counters["r6_commit_stale"] == 0, [m for m in caplog.messages if "STALE" in m]
    assert RID_K in (f.__dict__.get("_r6_commit_now") or set())
