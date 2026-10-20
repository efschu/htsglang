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


@pytest.mark.xfail(strict=True, reason="EZ-S2b review A4 / N6 (07:20Z KURZ-NACH-P-R5): a TEXT rvp pending is p_only by state (front.py "
                   "_rvp_take, text branch), so a kurz u=875 is classed p_only (R9 is vision only) and PBOUND-FLIP-NOW fired at "
                   "S_eff 7361 < X. NOT fixed here: its P leg is the only way D's PARKED stream gets its context (resume_via_p); "
                   "class kurz alone would make it no flip trigger (R7) and leave D's stream waiting for a P phase nobody "
                   "starts -- a hang risk that needs the planner's decision on who serves a midstream-refused kurz request "
                   "(S3a). The pin turns green when the class comes from d_extent and a serving owner exists.")
def test_a4_a_text_rvp_pending_is_not_p_only_by_state(monkeypatch, tmp_path):
    p, m = _rvp_text_member(monkeypatch, tmp_path)
    assert m.p_only is False


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
    assert "note_extend_done(rids=" in src       # review A2c: skip batches report too
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


# ---------------------------------------------------------------------------------------------------- R1u (user decision 20.10.)
def _front_r1u(now, tmp_path, *, sess_a="s1", sess_b="s1", common_len=10000, depth_b=7000):
    """rc26 13:21:56Z: pdflip-10-80 runs on D (N=10000, rest 2911 -> uncached [7089, 10000)), the halb member pdflip-10-81 (u=3068,
    ids = A's 10000 + its own tail, depth 7000, same session) waits; X=5229. Naive S = 3068 + 2911 = 5979 > X -> REROUTE_P;
    the union is S = 3068 (the 2911 are the member's own tail's front)."""
    import numpy as np
    xx = 5229
    f = _fake_front(running=["dec"], n=6, kv={"available": 400000, "evictable": 0})
    f.tp_prefill_max_tokens = xx
    f.t_awake = now - 60.0
    f._sync_batch_gate = lambda: None
    ra, rb = "pdflip-10-80", "pdflip-10-81"
    f._d_pf_book().grant(rid=ra, tokens=2911, now=now - 2.0)
    f._d_pf_book().enter_leg2(rid=ra)
    f.groups["D"].outstanding[ra] = now - 2.0
    f._rvp_state()
    f.__dict__.setdefault("_d_prefill_legs", {})[ra] = now - 2.0
    f._leg2_lookahead.add(ra)
    _beacon_reading(f, tmp_path, ra, now - 1.5, 2911, 1000, done=False)       # chunk running: rest 2911 counts
    ids_a = np.arange(10000, dtype=np.int32)
    ids_b = np.concatenate([ids_a[:common_len], np.arange(50000, 50000 + (7000 + 3068 - common_len), dtype=np.int32)])
    f.__dict__["_set_ids"] = {ra: ids_a, rb: ids_b}
    f.__dict__["_sess_by_rid"] = {ra: sess_a, rb: sess_b}
    f.__dict__["_set_depth"] = {ra: 7089, rb: depth_b}
    f._set_st()["shorts"][rb] = (now - 1.0, 3068, xx)
    return f


def test_r1u_a_shared_piece_of_the_running_rest_and_a_member_counts_once_no_flip(monkeypatch, caplog, tmp_path):
    """SOLL R1u/R1/R6, the live case: S_union = 3068 < X = 5229 -> COLLECT (R3), no REROUTE_P (naive S = 5979)."""
    _on(monkeypatch)
    now = time.time()
    f = _front_r1u(now, tmp_path)
    with envs.FLLIPER_PDFLIP_DECIDER.override(True), caplog.at_level(logging.INFO, logger="pdflip.front"):
        dc = f._decode_collect(now)
    sv = [m for m in caplog.messages if "PDFLIP SET-VERDICT" in m]
    assert dc == "hold", (dc, sv)
    assert not any("verdict=REROUTE_P" in m for m in sv) and f.counters["set_reroute_p"] == 0
    assert any(" S=3068 " in m for m in sv), sv
    assert f.counters["r1u_overlap_tokens"] == 2911 and any("PDFLIP R1U" in m for m in caplog.messages)


def test_r1u_siblings_without_overlap_keep_the_sum(monkeypatch, tmp_path):
    """Another session, and the same session whose prompts diverge before the uncached range: no overlap, S = 5979 > X -> flip."""
    _on(monkeypatch)
    now = time.time()
    for kw in ({"sess_b": "s2"}, {"common_len": 100}):
        f = _front_r1u(now, tmp_path, **kw)
        with envs.FLLIPER_PDFLIP_DECIDER.override(True):
            assert f._decode_collect(now) == "P" and f.counters["set_reroute_p"] == 1
        assert f.counters["r1u_overlap_tokens"] == 0


def test_r1u_pure_union_of_member_pieces_counts_each_token_once():
    """Two members covering overlapping parts of A's rest: the union, not the sum, is taken out."""
    ids = {("A", "B"): 9000, ("A", "C"): 9500}
    ov = pp.running_overlap([("A", 7000, 3000, "s")], [("B", 7000, "s"), ("C", 8000, "s")], lambda a, b: ids[(a, b)])
    assert ov == 2500          # B: [7000,9000), C: [8000,9500) -> union [7000,9500)
    assert pp.running_overlap([("A", 7000, 3000, "")], [("B", 7000, "")], lambda a, b: 9000) == 0     # unknown session


def test_r1u_two_running_prefills_share_a_piece_counted_once_also_with_a_member():
    """R1u, run against run: A (rest 3000 at [7000,10000)) and B (rest 2000 at [8000,10000)), same session, common 10000: the
    piece [8000,10000) is in both rests -> once. With a member that brings the same piece too, the piece is taken out of each
    rest once (union per rest), so S = what is really new."""
    com = lambda a, b: 10000      # noqa: E731
    runs = [("A", 7000, 3000, "s"), ("B", 8000, 2000, "s")]
    assert pp.running_overlap(runs, [], com) == 2000                                  # B's whole rest is A's tail
    # with member M (depth 8000): A loses [8000,10000)=2000, B loses its 2000 (M and A spans coincide: once) -> 4000 out of 5000
    assert pp.running_overlap(runs, [("M", 8000, "s")], com) == 4000
    assert pp.running_overlap(runs, [], lambda a, b: 100) == 0                        # diverging prompts: nothing shared
    assert pp.running_overlap([("A", 7000, 3000, ""), ("B", 8000, 2000, "")], [], com) == 0   # session unknown: counts in full


def test_r1u_two_running_prefills_on_the_real_path(monkeypatch, tmp_path):
    """Front: two counted running prefills of one session whose rests share a piece -> S counts the piece once (R6 rest 3000 +
    2000 -> 3000), so a halb member of 2400 stays under X=5229 (naive 7400 would reroute)."""
    import numpy as np
    _on(monkeypatch)
    now = time.time()
    f = _front_r1u(now, tmp_path)
    ra, rb, rc = "pdflip-10-80", "pdflip-10-79", "pdflip-10-81"
    # pdflip-10-80 is run A (rest 2911); add run B = pdflip-10-79: same session, ids = A's first 10000, depth 8500 (rest 1500)
    ids = f.__dict__["_set_ids"]
    ids[rb] = ids[ra]
    f.__dict__["_sess_by_rid"][rb] = "s1"
    f._d_pf_book().grant(rid=rb, tokens=1500, now=now - 2.0)
    f._d_pf_book().enter_leg2(rid=rb)
    f.groups["D"].outstanding[rb] = now - 2.0
    f.__dict__["_d_prefill_legs"][rb] = now - 2.0
    f._leg2_lookahead.add(rb)
    base = f._d_ext_reading
    import flliper.srt.pdflip.d_extend_beacon as dxb2
    clock = {"t": 0}
    bc = dxb2.DExtendBeacon(str(tmp_path / "D-pid9.bin"), clock_ns=lambda: clock["t"])
    bc.note_recv(rid=rb)
    clock["t"] = int((now - 1.4) * 1e9)
    bc.note_extend(rids=[rb], rests=[1500], extend_lens=[100])
    f._d_ext_reading = lambda: dxb2.read_group_full(str(tmp_path), "D", 7, session_of=lambda pid: 7)
    f._set_st()["shorts"].clear()
    f._set_st()["shorts"]["pdflip-10-90"] = (now - 1.0, 2400, 5229)
    f.__dict__["_set_depth"]["pdflip-10-90"] = 0
    with envs.FLLIPER_PDFLIP_DECIDER.override(True):
        running = f._dc_carried_running()
        eff = F.Front._r1u_running(f, running, [])
    assert running == 2911 + 1500 and eff == 2911, (running, eff)
    assert f.counters["r1u_overlap_tokens"] == 1500


# ---------------------------------------------------------------------------------------------------- rc26 P-PREFILL-GESCHWISTER
def _front_gesch(now, tmp_path, *, rid, x, u, rest_start, ext_len, big_rid, big_u, big_waited, started_ago):
    f = _fake_front(running=["dec"], n=6, kv={"available": 400000, "evictable": 0})
    f.tp_prefill_max_tokens = x
    f.t_awake = now - 60.0
    f._sync_batch_gate = lambda: None
    f._d_pf_book().grant(rid=rid, tokens=u, now=now - started_ago - 0.3)
    f._d_pf_book().enter_leg2(rid=rid)
    f.groups["D"].outstanding[rid] = now - started_ago - 0.3
    f._rvp_state()
    f.__dict__.setdefault("_d_prefill_legs", {})[rid] = now - started_ago - 0.3
    f._leg2_lookahead.add(rid)
    F.Front._note_x_decision(f, rid, x, u)
    F.Front._front_x_headers(f, rid, None, False)
    _beacon_reading(f, tmp_path, rid, now - started_ago, rest_start, ext_len, done=True)
    big = F.Pending(big_rid, "/v1/messages", {}, "t", now - big_waited, asyncio.new_event_loop().create_future(),
                    est_prompt=big_u + 40000, est_uncached=big_u)
    big.x_set = x
    f.queue.append(big)
    return f


def test_rc26_pdflip_48_456_a_finished_single_chunk_prefill_is_no_r6_reason(monkeypatch, tmp_path):
    """rc26 14:15:13: pdflip-48-456 (decision u=1797, X=5397): D's ONE forward ended 13.417 (`Prefill batch #new-token 1797`), the
    verdict fell 13.527 as R6 `S=7376 > X with a D prefill of 1797 running` and P then realised 773. v3 reports rest_after = 0:
    the big member pdflip-48-455 (5579, waited 3.6 s < T_halb) holds (R4) instead of flipping 0.9 s early on a computed rest."""
    _on(monkeypatch)
    now = time.time()
    f = _front_gesch(now, tmp_path, rid="pdflip-48-456", x=5397, u=1797, rest_start=1797, ext_len=1797,
                     big_rid="pdflip-48-455", big_u=5579, big_waited=3.6, started_ago=2.3)
    with envs.FLLIPER_PDFLIP_DECIDER.override(True):
        assert f._dc_carried_running() == 0
        assert f._decode_collect(now) == "hold" and f.counters["set_reroute_p"] == 0


def test_rc26_pdflip_46_451_a_chunked_prefill_counts_only_what_is_left(monkeypatch, tmp_path):
    """rc26 14:13:59: pdflip-46-451 (decision u=2716, chunks 2176 + 540): chunk 1 ended 59.799, the verdict fell 59.832 counting
    the whole 2716. v3 counts the 540 D has left (P then realised 28: D's finished chunks stay in its tree). The flip itself is
    legitimate (the big member pdflip-46-450 waited 5.0 s > T_halb, R4/R6) -- only the figure was wrong."""
    _on(monkeypatch)
    now = time.time()
    f = _front_gesch(now, tmp_path, rid="pdflip-46-451", x=5504, u=2716, rest_start=2716, ext_len=2176,
                     big_rid="pdflip-46-450", big_u=45032, big_waited=5.0, started_ago=3.0)
    with envs.FLLIPER_PDFLIP_DECIDER.override(True):
        assert f._dc_carried_running() == 540
        assert f._decode_collect(now) == "P" and f.__dict__["_r6_counted"] == {"pdflip-46-451": 540}


# ---------------------------------------------------------------------------------------------------- A2c / A3 (review R1u)
def test_a2c_a_skip_batch_reports_its_done_too():
    """The scheduler reports note_extend_done for every extend batch, a skip batch included (FIFO stays aligned)."""
    import inspect
    from flliper.srt.managers import scheduler as S
    src = inspect.getsource(S.Scheduler.process_batch_result)
    i = src.index("note_extend_done")
    assert "pdflip_skip_extend" not in src[src.rindex("if self.pdflip_d_ext_beacon", 0, i):i]


def test_a2c_fifo_stays_aligned_when_a_skip_batch_is_followed_by_a_real_chunk(tmp_path):
    """A skip resume (note_extend + done, nothing computed in the rest after it: reported as sent) then a real chunk: the
    done of the real chunk reads ITS rest, not the skip batch's."""
    bc = dxb.DExtendBeacon(str(tmp_path / "D-pid1.bin"), clock_ns=lambda: 10**9)
    bc.note_recv(rid=_RID9)
    bc.note_extend(rids=[_RID9], rests=[5000], extend_lens=[1000])      # skip batch
    bc.note_extend_done(rids=[_RID9])                                   # now reported (A2c)
    assert _read9(tmp_path) == 4000
    bc.note_extend(rids=[_RID9], rests=[4000], extend_lens=[4000])      # real chunk
    bc.note_extend_done(rids=[_RID9])
    assert _read9(tmp_path) == 0


def test_a3_r1u_unknown_member_depth_takes_nothing_off_and_is_named(monkeypatch, tmp_path):
    """User order "nichts schaetzen": a member whose cached depth is not known is not assumed at depth 0 (that would count its
    whole prefix as shared): no overlap is taken off, ``r1u_depth_unknown`` names it. With the depth known the live case
    still counts once."""
    _on(monkeypatch)
    now = time.time()
    f = _front_r1u(now, tmp_path)
    rb = "pdflip-10-81"
    f.__dict__["_set_depth"].pop(rb)
    f._set_st()["shorts"][rb] = (now - 1.0, 3068, 5229)
    members = [pp.SetMember(rb, 3068, now - 1.0, False)]
    with envs.FLLIPER_PDFLIP_DECIDER.override(True):
        running = f._dc_carried_running()
        assert F.Front._r1u_running(f, running, members) == running == 2911
    assert f.counters["r1u_depth_unknown"] == 1 and f.counters["r1u_overlap_tokens"] == 0
    f.__dict__["_set_depth"][rb] = 7000
    with envs.FLLIPER_PDFLIP_DECIDER.override(True):
        assert F.Front._r1u_running(f, f._dc_carried_running(), members) == 0
