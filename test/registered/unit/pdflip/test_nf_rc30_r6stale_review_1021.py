"""Review of NF-RC30-R6STALE (07eee5cefa, desk/nf-rc30-r6stale-1021), SOLL v2 R1u R3 R4 R6 R6k R7 R11.

Report: /spinning/gpu-arb/deskq/done/nf-rc30-r6stale-review-1021.md. Live boot (measured, not estimated):
``boot_weg2_dkrnfint4h6ablxcw3spmlrhlvwflbar1dauer10210632_1d822bd150_1021_063252`` -- four R6-REDIRECTs (42-671, 46-680,
48-687, 50-698; the author's report names three). Verdict -> next chunk start on D (device start = PDFLIP-VRAM-PEAK chunk end
minus the batch's gpu-ms): +6 ms, none (abort landed during chunk 1), +1 ms, +36 ms; the abort POST completed +6/+80/+271/+330 ms
after the verdict and D took every abort only after its next forward.

Green tests pin what the fix guarantees (R6 danger direction "rest too small -> flip missing", SOLL example 7 both ways).
The tests named after Auflagen A1-A4 were ``xfail(strict=True)`` at f79f763480; the author's fix (21.10.) turned them green and
removed the markers (A1: the test now also models D's COMMITTED marker of its recv step, the fix's D signal).
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
from test_nf_r6_rest_exakt_1020 import _Ev, _front_live, end_forward, RID_K, RID_G  # noqa: E402
from test_pdflip_arrival_seat_rule_0929 import _front as _fake_front  # noqa: E402
from test_ez_s2b_1020 import _beacon_reading  # noqa: E402
from test_pdflip_set_scheduler_1017 import _on  # noqa: E402

SID = 7


def _launch_next_chunk(f, rid, *, rest, ext_len, key=37871):
    """D launches the next chunk of ``rid`` on every rank (its prefix already covers the earlier chunks)."""
    ws, _ev = f._test_running[rid]
    ev = _Ev()
    for w in ws:
        w.note_extend(rids=[rid], rests=[rest], extend_lens=[ext_len])
        w.arm_done(key=key, rids=[rid], ready=ev.query)
    return ev


def _sv(caplog):
    return [m for m in caplog.messages if "PDFLIP SET-VERDICT" in m]


# ------------------------------------------------------------------------------------------- green: what the fix guarantees
def test_r6_three_chunks_second_in_flight_the_third_counts_and_flips(monkeypatch, caplog, tmp_path):
    """SOLL R6 danger direction ("rest too small -> flip missing"): a halb D prefill of 6000 in three chunks of 2000; chunk 1 ended
    (polled), chunk 2 in flight -> free 2000 is what P can still get. Member 5100 (halb at X 7016): S = 7100 > X -> REROUTE_P,
    the redirect counts 2000 and names the in-flight 2000."""
    _on(monkeypatch)
    now = time.time()
    f = _front_live(now, tmp_path, member_u=5100, ended=False, rest_start=6000, ext_len=2000, klass="halb")
    end_forward(f, RID_K)
    _launch_next_chunk(f, RID_K, rest=4000, ext_len=2000)
    with envs.FLLIPER_PDFLIP_DECIDER.override(True), caplog.at_level(logging.INFO, logger="pdflip.front"):
        assert f._dc_carried_running() == 2000
        assert f._decode_collect(now) == "P"
    assert f.counters["r6_redirected"] == 1
    assert any("R6-REDIRECT" in m and "counted=%s:2000" % RID_K in m and "in_flight=%s:2000" % RID_K in m
               for m in caplog.messages)


@pytest.mark.parametrize("ext_len, want_running, want", [(2000, 2000, "P"), (4000, 0, "hold")])
def test_soll_example_7_running_4000_and_5000_arrives(monkeypatch, caplog, tmp_path, ext_len, want_running, want):
    """SOLL section 5 example 7 (X=6000, halbX=3000): D prefills 4000, 5000 arrives -> S = 9000 > X -> R6 redirect. With the 4000 in
    chunks (first chunk 2000 in flight) R6 still flips (S = 5000 + 2000 = 7000 > X). With the whole 4000 in ONE launched forward
    nothing of it can reach P (it ends on D): S = 5000 < X, no flip under X (R7) -- the reading the fix takes (R1u "neu zu
    rechnende Token", the R6k evidence "P rechnete 5836 von gezaehlten 76136")."""
    _on(monkeypatch)
    now = time.time()
    f = _front_live(now, tmp_path, member_u=5000, ended=False, rest_start=4000, ext_len=ext_len, x=6000, half_x=3000,
                    klass="halb")
    with envs.FLLIPER_PDFLIP_DECIDER.override(True), caplog.at_level(logging.INFO, logger="pdflip.front"):
        assert f._dc_carried_running() == want_running
        assert f._decode_collect(now) == want
    assert f.counters["r6_redirected"] == (1 if want == "P" else 0)


def test_r6k_kurz_running_with_free_rest_still_not_counted(monkeypatch, tmp_path):
    """SOLL R6k unchanged by v4: a kurz running prefill with an unlaunched rest (free 1000 > 0) is not in S, not redirected, and is
    not named as in-flight (R6k decides before ``free`` is read)."""
    _on(monkeypatch)
    now = time.time()
    f = _front_live(now, tmp_path, member_u=3600, ended=False, rest_start=2000, ext_len=1000, klass="kurz")
    with envs.FLLIPER_PDFLIP_DECIDER.override(True):
        assert f._dc_carried_running() == 0
        assert f._decode_collect(now) == "hold"
    assert f.counters["r6_redirected"] == 0 and f.counters["r6_rest_in_flight"] == 0
    assert f.counters["r6k_kurz_stays_on_d"] >= 1


def test_ranks_launch_skew_min_free_is_the_launched_chunk(tmp_path):
    """R11, live skew: rank chunk ends differ by 110-130 ms (TP0 later than TP1/TP2 in all four cases), the launch of the next
    chunk happens in the same iteration on every rank (the recv broadcast is collective). While one rank has noted chunk 2 and the
    others not, free = MIN = 0 (committed on all ranks), rest = MAX (D still computes)."""
    ws = [dxb.DExtendBeacon(str(tmp_path / ("D-pid%d.bin" % (r + 1))), clock_ns=lambda: 10**9) for r in range(3)]
    for w in ws:
        w.note_recv(rid="r")
        w.note_extend(rids=["r"], rests=[4681], extend_lens=[2496])
        w.note_extend_done(rids=["r"])                         # chunk 1 ended on every rank
    ws[2].note_extend(rids=["r"], rests=[2185], extend_lens=[2185])
    got = dxb.read_group_full(str(tmp_path), "D", SID, session_of=lambda pid: SID)[dxb.rid_key("r")]
    assert got[1:] == (2185, 0)


# ------------------------------------------------------------------------------------------- Auflagen A1-A4 (fixed 21.10., markers removed)
def test_a1_chunk_boundary_last_chunk_counts_nothing(monkeypatch, caplog, tmp_path):
    """Live 42-671 (07:51:27,475): u=4681 halb, chunk 1 = 2496 ended (TP0 87.424, polled), chunk 2 = 2185 = the whole remainder,
    not yet noted; the gross member 42-673 (6792 at X 6247) arrived on a D-probe answered at D's pass end. D's next pass launches
    chunk 2 (device start 87.483, +6 ms after the verdict); the abort POST completed 87.483 and D took it after chunk 2 -> P
    computed 9 of the counted 4681. With v4 the reading at the boundary gives free = 2185, S = 8977 -> REROUTE_P + redirect of
    2185 tokens D computes itself. Target: nothing of a remainder D launches before an abort can land is counted or redirected."""
    _on(monkeypatch)
    now = time.time()
    f = _front_live(now, tmp_path, member_u=6792, ended=False, rest_start=4681, ext_len=2496, x=6247, half_x=3123,
                    klass="halb")
    end_forward(f, RID_K)                                       # chunk 1 done, chunk 2 not launched yet: the pass boundary
    # Fix A1 (author, 21.10.): D's recv step is over -- its running chunked_req's next chunk is fixed for this iteration, the
    # rank marks it COMMITTED (Scheduler._pdflip_d_ext_commit) before the probe answer can lead to a verdict
    ws, _ev = f._test_running[RID_K]
    for w in ws:
        w.commit_next(rids=[RID_K])
    with envs.FLLIPER_PDFLIP_DECIDER.override(True), caplog.at_level(logging.INFO, logger="pdflip.front"):
        running = f._dc_carried_running()
        got = f._decode_collect(now)
    assert running == 0, running
    assert got == "hold", (got, _sv(caplog))                    # the final verdict waits for the start record (one D iteration)
    assert not any("R6-REDIRECT" in m for m in caplog.messages)
    assert f.counters["r6_commit_pending"] == 1          # (the gross member's R4 wait is COLLECT anyway: nothing to hold)
    # D launches chunk 2 (= the whole remainder): its start record gives free 0 -> nothing of it is counted, no redirect;
    # the gross member alone waits its T_halb (R4, 3.9 s < 4.5 s)
    _launch_next_chunk(f, RID_K, rest=2185, ext_len=2185)
    caplog.clear()
    with envs.FLLIPER_PDFLIP_DECIDER.override(True), caplog.at_level(logging.INFO, logger="pdflip.front"):
        assert f._dc_carried_running() == 0
        assert f._decode_collect(now) == "hold"
    assert not any("R6-REDIRECT" in m for m in caplog.messages) and f.counters["r6_redirected"] == 0
    assert f.counters["r6_rest_in_flight"] >= 1


def test_a2_free_min_ignores_a_rank_record_from_before_the_grant(tmp_path):
    """Park-handback / resume re-grant of the same rid: leg 1 ended on both ranks (free 0). Leg 2 granted at t=2 s; rank A noted its
    first extend (rest 3000, chunk 1000 -> free 2000), rank B (ms behind in the same iteration) still holds its leg-1 record.
    ``started``/``rest_of`` take the newer start; ``free`` must only take ranks whose record is of THIS grant: 2000, not 0."""
    clock = {"t": 10**9}
    ws = [dxb.DExtendBeacon(str(tmp_path / ("D-pid%d.bin" % (r + 1))), clock_ns=lambda: clock["t"]) for r in range(2)]
    for w in ws:
        w.note_recv(rid="r")
        w.note_extend(rids=["r"], rests=[1500], extend_lens=[1500])
        w.note_extend_done(rids=["r"])
        w.note_done(rid="r")
    clock["t"] = 3 * 10**9
    ws[0].note_recv(rid="r")
    ws[0].note_extend(rids=["r"], rests=[3000], extend_lens=[1000])
    reading = dxb.read_group_full(str(tmp_path), "D", SID, session_of=lambda pid: SID)
    assert dxb.rest_of(reading, "r", 2.0) == 3000
    assert dxb.free_of(reading, "r", 2.0) == 2000


def test_a3_r1u_member_share_of_the_in_flight_chunk_is_not_new(monkeypatch, caplog, tmp_path):
    """R1u ("S = die neu zu rechnenden Token"): run A (N=10000, D's rest 3911: chunk [6089,7089) in flight, free [7089,10000) =
    2911) and halb member B of the same session probed at depth 6089 (before A's chunk), common prefix 7089, N_B 9089 -> u_B 3000.
    P computes A's 2911 + B's own [7089,9089) = 2000 (B's [6089,7089) is A's in-flight chunk: it ends on D, P reads it from the
    store) = 4911 < X 5229 -> no flip (R7). The fix counts S = 2911 + 3000 = 5911 (the span is taken from depth N-free, so the
    in-flight piece is not discounted from u_B) -> REROUTE_P."""
    import numpy as np
    _on(monkeypatch)
    now = time.time()
    xx = 5229
    f = _fake_front(running=["dec"], n=6, kv={"available": 400000, "evictable": 0})
    f.tp_prefill_max_tokens = xx
    f.t_awake = now - 60.0
    f._sync_batch_gate = lambda: None
    ra, rb = "pdflip-10-80", "pdflip-10-81"
    f._d_pf_book().grant(rid=ra, tokens=3911, now=now - 2.0)
    f._d_pf_book().enter_leg2(rid=ra)
    f.groups["D"].outstanding[ra] = now - 2.0
    f._rvp_state()
    f.__dict__.setdefault("_d_prefill_legs", {})[ra] = now - 2.0
    f._leg2_lookahead.add(ra)
    _beacon_reading(f, tmp_path, ra, now - 1.5, 3911, 1000, done=False)
    ids_a = np.arange(10000, dtype=np.int32)
    ids_b = np.concatenate([ids_a[:7089], np.arange(50000, 52000, dtype=np.int32)])
    f.__dict__["_set_ids"] = {ra: ids_a, rb: ids_b}
    f.__dict__["_sess_by_rid"] = {ra: "s1", rb: "s1"}
    f.__dict__["_set_depth"] = {ra: 6089, rb: 6089}
    f._set_st()["shorts"][rb] = (now - 1.0, 3000, xx)
    with envs.FLLIPER_PDFLIP_DECIDER.override(True), caplog.at_level(logging.INFO, logger="pdflip.front"):
        assert f._dc_carried_running() == 2911
        dc = f._decode_collect(now)
    assert dc == "hold", (dc, _sv(caplog))


def test_a4_r6_not_started_is_not_logged_for_a_started_prefill(monkeypatch, caplog, tmp_path):
    """Live 48-687 at 11,134: granted 3280, chunk 1 (2496) in flight, free 784. The row HAS begun on D; v4 makes running (784) <
    granted (3280), so ``_dc_carried_running`` names it ``R6-NOT-STARTED ... D has not begun`` (counter r6_not_started_skipped) --
    a false line, and a new class for the NUR-NEUES monitor in every in-flight case."""
    _on(monkeypatch)
    now = time.time()
    f = _front_live(now, tmp_path, member_u=4584, ended=False, rest_start=3280, ext_len=2496, x=6114, half_x=3057,
                    klass="halb")
    with envs.FLLIPER_PDFLIP_DECIDER.override(True), caplog.at_level(logging.INFO, logger="pdflip.front"):
        assert f._dc_carried_running() == 784
    assert not any("R6-NOT-STARTED" in m for m in caplog.messages)
    assert f.counters["r6_not_started_skipped"] == 0
