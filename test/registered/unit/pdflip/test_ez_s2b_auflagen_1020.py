"""EZ S2b review Auflagen A1-A7 (nf-review-ez-s2b-1020.md; SOLL R1a R2 R4 R5 R6 R7 R9). Real paths, live cases of boot
``..._10e9b2b492_1019_230333``; switch ``FLLIPER_PDFLIP_DECIDER`` ON."""
import asyncio
import logging
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flliper.srt.environ import envs  # noqa: E402
from flliper.srt.pdflip import d_extend_beacon as dxb  # noqa: E402
from flliper.srt.pdflip import front as F  # noqa: E402
from flliper.srt.pdflip import phase_policy as pp  # noqa: E402
from test_ez_s2b_1020 import _beacon_reading  # noqa: E402
from test_pdflip_idle_drain_budget_rc2_0924 import X as RC2_X  # noqa: E402
from test_pdflip_idle_drain_budget_rc2_0924 import Seat, _front as _real_front  # noqa: E402
from test_pdflip_arrival_seat_rule_0929 import _front as _fake_front  # noqa: E402
from test_pdflip_set_scheduler_1017 import _on  # noqa: E402

X_A = 4096
RID_0_1, RID_0_2, RID_0_5 = "pdflip-0-1", "pdflip-0-2", "pdflip-0-5"


# ---------------------------------------------------------------------------------------------------- A2 (a) pdflip-0-1/0-2/0-5
def _front_a(now, tmp_path, *, done):
    """rc19 23:06:43: pdflip-0-1 (D prefill, 1517 tokens, extend began 23:06:37.720, forward ended .498) and the halb member
    pdflip-0-2 (3596) waiting; the verdict fell .576 (78 ms after the forward end)."""
    f = _fake_front(running=["dec"], n=6, kv={"available": 400000, "evictable": 0})
    f.tp_prefill_max_tokens = X_A
    f.t_awake = now - 60.0
    f._sync_batch_gate = lambda: None
    f._d_pf_book().grant(rid=RID_0_1, tokens=1517, now=now - 6.0)
    f._d_pf_book().enter_leg2(rid=RID_0_1)
    f.groups["D"].outstanding[RID_0_1] = now - 6.0
    f._rvp_state()
    f.__dict__.setdefault("_d_prefill_legs", {})[RID_0_1] = now - 6.0
    f._leg2_lookahead.add(RID_0_1)
    F.Front._note_x_decision(f, RID_0_1, X_A, 1517)
    F.Front._front_x_headers(f, RID_0_1, None, False)               # leg2's POST pops the record
    _beacon_reading(f, tmp_path, RID_0_1, now - 5.85, 1517, 1517, done=done)
    f._set_st()["shorts"][RID_0_2] = (now - 1.0, 3596, X_A)         # halb member, S alone 3596 < X
    return f


def test_a_the_rest_of_a_computed_prefill_makes_no_flip_below_x_and_the_kurz_arrival_is_d_now(monkeypatch, caplog, tmp_path):
    """(a) 23:06:43.576 REROUTE_P why=R6 S=5113 > X=4096 with a D prefill of 1517 'running' -- its forward had ended .498. Without
    the computed rest S = 3596 < X: COLLECT, no flip (R7); pdflip-0-5 (25 tokens, .825) is then D_NOW (R2), not a rider."""
    _on(monkeypatch)
    now = time.time()
    f = _front_a(now, tmp_path, done=True)
    with envs.FLLIPER_PDFLIP_DECIDER.override(True), caplog.at_level(logging.INFO, logger="pdflip.front"):
        dc = f._decode_collect(now)
    sv = [m for m in caplog.messages if "PDFLIP SET-VERDICT" in m]
    assert dc == "hold", (dc, sv)
    assert not any("verdict=REROUTE_P" in m for m in sv), sv
    assert f.counters["set_reroute_p"] == 0 and f.admit_d is True and not f._set_st()["go_p"]
    # pdflip-0-5 arrives: the door is open, no decided flip -> D_NOW (R2), the real arrival decision
    flip_decided = (not f.admit_d) or bool(f._set_st()["go_p"])
    ad = pp.decide("arrival", uncached=25, x_own=X_A, half_frac=0.5, d_open=True, flip_decided=flip_decided)
    assert (ad.action, ad.rule) == (pp.ACT_D_NOW, "R2")
    with envs.FLLIPER_PDFLIP_DECIDER.override(True):
        assert f._d_direct_yields(RID_0_5, uncached=25) is False


def test_a_while_the_forward_runs_r6_counts_and_flips(monkeypatch, tmp_path):
    """SOLL R6 side of the same fixture: the 1517 are being computed -> S = 5113 > X -> REROUTE_P (the SOLL fixtures stay)."""
    _on(monkeypatch)
    now = time.time()
    f = _front_a(now, tmp_path, done=False)
    with envs.FLLIPER_PDFLIP_DECIDER.override(True):
        assert f._decode_collect(now) == "P" and f.counters["set_reroute_p"] == 1


# ---------------------------------------------------------------------------------------------------- A3 pool reading, short bound
def _kurz_with_stale_pool(*, stale_available, answer, answer_after_s, retry_blocked=False):
    """A pool reading 1.5 s old (``stale_available`` rows) and a D that answers ``answer`` (rows, None = never) after
    ``answer_after_s``. The REAL ``_acquire_short_seat`` for a kurz request; returns (seat, why, elapsed_s, front)."""
    async def body():
        f = _real_front(awake="D")
        await f._d_seat.acquire()
        running = Seat(f, "running", "short", tokens=2000)
        f.groups["D"].outstanding["running"] = time.time()
        f._d_pool = {"t": time.time() - 1.5, "available": stale_available, "limit": 27466, "occupied": 0}
        if retry_blocked:
            f._d_pool_retry_after = time.time() + 5.0
        calls = []

        async def read():
            calls.append(1)
            await asyncio.sleep(answer_after_s)
            if answer is None:
                return None
            f._d_pool = {"t": time.time(), "available": answer, "limit": 27466, "occupied": 0}
            return f._d_pool
        f._d_pool_reading = read
        why = []
        t0 = time.monotonic()
        with envs.FLLIPER_PDFLIP_DECIDER.override(True):
            seat = await f._acquire_short_seat("k", 500, why, uncached=500, x_set=RC2_X)
        el = time.monotonic() - t0
        if seat is not None:
            seat.release("t")
        running.release("t")
        return seat, why, el, f, calls
    return asyncio.run(body())


def test_a3_a_stale_reading_and_a_d_that_answers_fast_is_gated_on_the_new_reading():
    """The reviewer's probe (N5): reading 1.5 s old, D answers in 5 ms with 'no room' -> the FIX 4a gate holds (d_budget). The old
    S2b code admitted at once without any gate. (MR2: a stale reading counts as fresh -> admits on the OLD number: red here.)"""
    seat, why, el, f, calls = _kurz_with_stale_pool(stale_available=100000, answer=10, answer_after_s=0.005)
    assert seat is None and why == ["d_budget"], (seat, why)
    assert el < 0.3 and calls == [1] and f.counters["r2_kurz_pool_waited_ok"] == 1


def test_a3_a_stale_reading_and_a_fast_d_with_room_admits_on_the_new_reading():
    seat, why, el, f, calls = _kurz_with_stale_pool(stale_available=10, answer=100000, answer_after_s=0.005)
    assert seat is not None and why == [] and el < 0.3          # the OLD reading said no room: the NEW one decides


def test_a3_a_d_that_does_not_answer_goes_on_without_the_gate_after_the_short_bound_only():
    seat, why, el, f, calls = _kurz_with_stale_pool(stale_available=10, answer=None, answer_after_s=1.0)
    assert seat is not None and why == []                       # no reading: gate off by name, as with an unreadable pool
    assert 0.04 <= el < 0.3, el                                 # the bound (50 ms), never the 1.0 s read timeout
    assert f.counters["r2_kurz_pool_nowait"] == 1


def test_a3_a_read_that_failed_lately_costs_no_wait_at_all():
    """``_d_pool_retry_after`` holds (one failed read per age window): _d_pool_reading answers None at once, no round trip."""
    async def body():
        f = _real_front(awake="D")
        await f._d_seat.acquire()
        running = Seat(f, "running", "short", tokens=2000)
        f.groups["D"].outstanding["running"] = time.time()
        f._d_pool = {"t": time.time() - 1.5, "available": 10, "limit": 27466, "occupied": 0}
        f._d_pool_retry_after = time.time() + 5.0
        t0 = time.monotonic()
        with envs.FLLIPER_PDFLIP_DECIDER.override(True):
            seat = await f._acquire_short_seat("k", 500, [], uncached=500, x_set=RC2_X)
        el = time.monotonic() - t0
        running.release("t")
        if seat is not None:
            seat.release("t")
        return seat, el, f
    seat, el, f = asyncio.run(body())
    assert seat is not None and el < 0.04 and f.counters["d_pool_read_suppressed"] == 1


# ---------------------------------------------------------------------------------------------------- A4 rvp text is not p_only (07:20Z)
def _rvp_text_member(monkeypatch, tmp_path):
    """The REAL intake: D's needs-P record of a midstream x-refusal (text, d_extent 875, rc19 epoch 238 pdflip-238-3261) goes
    through ``Front._rvp_take`` and the set's ``_set_members``; returns the member of that rid."""
    from test_ez_s1_review_1019 import _fresh
    from flliper.srt.pdflip import resume_via_p as rvp

    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "D")
    monkeypatch.setenv("FLLIPER_HICACHE_ARENA_DIR", str(tmp_path))
    monkeypatch.delenv(rvp.ENV, raising=False)
    f = _fresh()
    f.tag = "dkrtest"
    rvp.write_request("pdflip-238-3261", list(range(120000)), 875, 7647, "x_refusal_midstream")

    async def body():
        assert f._rvp_take() == 1
    asyncio.run(body())
    p = f.queue[-1]
    assert p.resume_via_p and p.est_uncached == 875
    mem = F.Front._set_members(f, time.time(), [p], f._set_st())
    return p, mem[0]


def test_a4_a_text_rvp_pending_is_not_p_only_by_state(monkeypatch, tmp_path):
    """S107 (was the strict-xfail pin of EZ-S2b review A4 / N6, 07:20Z KURZ-NACH-P-R5): a TEXT rvp leg classes by its
    exact u (d_extent 875 < X 7647: kurz), never p_only by state (R9 is vision only, R7 no flip under X). Its owner: it
    rides a decided flip (R1a) and D's own park re-offer serves the held stream -- the leg is never handed to D's line."""
    p, m = _rvp_text_member(monkeypatch, tmp_path)
    assert m.p_only is False
    assert p.p_only is False and p.est_uncached == 875
    assert pp.classify_arrival(875, 7647, 0.5, False) == pp.CLASS_KURZ
    p.d_eligible = True
    assert p.d_eligible is False, "a RESUME-VIA-P leg is P's by routing: no D line may take it"


def test_a4_a_vision_rvp_pending_is_p_only_r9(monkeypatch, tmp_path):
    """The R9 half stays: with REASON_VISION the request is p_only (D has no tower)."""
    from flliper.srt.pdflip import resume_via_p as rvp
    from test_ez_s1_review_1019 import _fresh

    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "D")
    monkeypatch.setenv("FLLIPER_HICACHE_ARENA_DIR", str(tmp_path))
    monkeypatch.delenv(rvp.ENV, raising=False)
    f = _fresh()
    f.tag = "dkrtest"
    f._rvp_state()
    f._rvp_bodies = {"pdflip-238-3262": ("/v1/messages", {"messages": []})}
    rvp.write_request("pdflip-238-3262", list(range(100)), 50, 7647, rvp.REASON_VISION)

    async def body():
        assert f._rvp_take() == 1
    asyncio.run(body())
    p = f.queue[-1]
    assert p.p_only is True and F.Front._set_members(f, time.time(), [p], f._set_st())[0].p_only is True


# ---------------------------------------------------------------------------------------------------- A5/A6 ONE "decided flip"
def _fx_flip(committed="FLIP"):
    """A set committed to a flip (``go_p`` = {g}), D still OPEN; ``committed`` = FLIP (FLIP_P/REROUTE_P) or PARK_NO_SEAT (R5)."""
    from test_pdflip_set_scheduler_1017 import _fx, _pending
    f = _fx()
    now = time.time()
    f.queue = [_pending("g", 9000, now - 5.0)]
    assert f._decode_collect(now) == "P"
    assert f._set_st()["go_p"] == {"g"}
    if committed == "PARK_NO_SEAT":
        f._set_st().setdefault("park_no_seat", set()).add("g")
    return f


def test_a5_one_meaning_of_decided_the_record_and_the_yield_agree_for_go_p_at_an_open_door(monkeypatch):
    """go_p committed (FLIP_P) with D open and NO seat for the arrival: the record says RIDE_P R1a AND ``_d_direct_yields`` yields
    (rides) -- the same state, the same answer. MR1 (go_p alone is no decided flip) turns this red. With a free D seat the request
    is D's (R2, no epoch rule) in both."""
    _on(monkeypatch)
    f = _fx_flip()
    assert f.admit_d and F.Front._flip_decided(f) is True
    f._arrival_seat_taken = lambda for_rid=None: (6, 6)                # no seat free
    p = F.Pending("k", "/generate", {}, "t", time.time(), asyncio.new_event_loop().create_future(), est_prompt=300, est_uncached=300)
    p.x_set = X_A
    with envs.FLLIPER_PDFLIP_DECIDER.override(True):
        assert f._d_direct_yields("k", uncached=300) is True
        F.Front._arrival_placement_decision(f, p, "k", 300, X_A, route="b", p_phase=False, collect_p=False, unpriced=False, pb_to_p=None)
    assert p.ride_p is True and f.counters["arrival_ride_p_r1a"] == 1
    f._arrival_seat_taken = lambda for_rid=None: (1, 6)                # a seat is free: D's (the SOLL collect_p_not_kept pin)
    with envs.FLLIPER_PDFLIP_DECIDER.override(True):
        assert f._d_direct_yields("k", uncached=300) is False


def test_a5_a_park_no_seat_commitment_is_no_decided_flip(monkeypatch):
    """R5: no seat for the set = prefill and park, no flip begins: the kurz arrival keeps waiting for its D seat (SOLL pin
    ``test_on_no_seat_requests_no_flip_and_the_short_waits_on_d``), in the record and in the yield alike."""
    _on(monkeypatch)
    f = _fx_flip("PARK_NO_SEAT")
    f._arrival_seat_taken = lambda for_rid=None: (6, 6)
    assert F.Front._flip_decided(f) is False
    with envs.FLLIPER_PDFLIP_DECIDER.override(True):
        assert f._d_direct_yields("k", uncached=300) is False


def test_a6_a_kurz_arrival_behind_a_committed_flip_never_waits_for_ever(monkeypatch):
    """The N9 hang (set_scheduler_1017 ``test_r7_after_a_decided_flip...``): go_p committed, an older P member in the queue keeps
    the seat order ('order' wait) -- the kurz request rides the decided flip (R1a/R5) instead of waiting. Bounded: 2 s."""
    _on(monkeypatch)
    f = _fx_flip()

    async def body():
        return await asyncio.wait_for(f._acquire_short_seat("k", 300, uncached=300), 2.0)
    with envs.FLLIPER_PDFLIP_DECIDER.override(True):
        assert asyncio.run(body()) is None
    assert f.counters["d_direct_yield"] + f.counters["arrival_seat_rides_decided_flip"] >= 1


# ---------------------------------------------------------------------------------------------------- A2 wiring (rank side)
def test_a2_the_scheduler_reports_the_rest_after_the_forward_ends():
    """The rank hook: ``process_batch_result`` calls ``note_extend_done`` for an extend batch (never a skip batch) and the extend
    start hands the beacon the batch's ``extend_len``; the file layout version is 3 and v1/v2 readers' files stay readable."""
    import inspect
    from flliper.srt.managers import scheduler as S

    src = inspect.getsource(S.Scheduler.process_batch_result)
    assert "note_extend_done(rids=" in src and "pdflip_skip_extend" in src
    assert "extend_lens=[_pdflip_d_ext.req_extend_len(req)" in inspect.getsource(S.Scheduler._run_batch_forward)
    assert dxb._VERSION == 3 and dxb.req_extend_len(type("R", (), {"extend_input_len": 7})()) == 7


# ---------------------------------------------------------------------------------------------------- A2a / A2b (review of the Auflagen)
_RID9, _SID9 = "pdflip-9-1", 7


def _read9(d):
    return dxb.read_group_full(str(d), "D", _SID9, session_of=lambda pid: _SID9)[dxb.rid_key(_RID9)][1]


def test_a2a_overlap_chunked_done_of_chunk1_after_chunk2_started_keeps_chunk2_rest(tmp_path):
    """SOLL R6: overlap scheduling launches chunk 2 before chunk 1's result is processed; the done of chunk 1 must report
    chunk 1's after-rest (2904), never the 0 of the last chunk (FIFO per rid)."""
    bc = dxb.DExtendBeacon(str(tmp_path / "D-pid1.bin"), clock_ns=lambda: 10**9)
    bc.note_recv(rid=_RID9)
    bc.note_extend(rids=[_RID9], rests=[7000], extend_lens=[4096])
    bc.note_extend(rids=[_RID9], rests=[2904], extend_lens=[2904])
    assert _read9(tmp_path) == 2904
    bc.note_extend_done(rids=[_RID9])
    assert _read9(tmp_path) == 2904
    bc.note_extend_done(rids=[_RID9])
    assert _read9(tmp_path) == 0


def test_a2b_ranks_merge_by_max_rest_not_by_the_later_start_stamp(tmp_path):
    """SOLL R6/R11: TP0 stamped 100 ns later and finished first (0); TP1 still computes (182): the group's rest is 182."""
    a = dxb.DExtendBeacon(str(tmp_path / "D-pid1.bin"), clock_ns=lambda: 1_000_000_100)
    b = dxb.DExtendBeacon(str(tmp_path / "D-pid2.bin"), clock_ns=lambda: 1_000_000_000)
    for bc in (a, b):
        bc.note_recv(rid=_RID9)
        bc.note_extend(rids=[_RID9], rests=[182], extend_lens=[182])
    a.note_extend_done(rids=[_RID9])
    assert _read9(tmp_path) == 182
    b.note_extend_done(rids=[_RID9])
    assert _read9(tmp_path) == 0
