"""EZ S2b (plan nf-entscheidungszentrale-plan-1019, step S2b; SOLL R1a R2 R2a R4 R5 R6 R7 R9): ONE decision at the arrival,
``decide(kind=admit|arrival)``. Live cases of boot ``..._10e9b2b492_1019_230333`` (logs /spinning/docker-acceptance/nf/evidence)
as tests on the REAL paths (real Front, real ``handle_generate`` / ``_acquire_short_seat`` / ``_requeue_after_x_refusal``);
switch ``FLLIPER_PDFLIP_DECIDER`` ON = the new path, OFF = the S2a code (the way back).

(a) P-PREFILL-UNTER-X  pdflip-0-5 (25 tokens, 23:06:43,825 ``BATCH queued awake=D admit_d=False``)
(b) R2-WARTEN          pdflip-2-115 (u=2733 < halbX 4000, 1.0 s D-POOL UNREADABLE before D_NOW)
(c) R6                 pdflip-104-2646 (REROUTE_P why=R6 S=198182 at 2.4 s: the rest 182 of a prefill D had already COMPUTED; review A1/A2)
(d) open points of S2a EARLY-FLIP gate, x_ref_nonum -> MEASURE (review H1)
"""
import asyncio
import logging
import os
import sys
import time
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flliper.srt.environ import envs  # noqa: E402
from flliper.srt.pdflip import d_extend_beacon as dxb  # noqa: E402
from flliper.srt.pdflip import front as F  # noqa: E402
from flliper.srt.pdflip import phase_policy as pp  # noqa: E402
from test_pdflip_idle_drain_budget_rc2_0924 import X as RC2_X  # noqa: E402
from test_pdflip_idle_drain_budget_rc2_0924 import Seat, _drop, _front as _real_front, _route  # noqa: E402
from test_ez_s1_review_1019 import _fresh, _pend, X, HALF_X  # noqa: E402
from test_pdflip_arrival_seat_rule_0929 import _front as _fake_front  # noqa: E402
from test_pdflip_set_scheduler_1017 import _on  # noqa: E402
from test_pdflip_front_x_exact import _Req  # noqa: E402


@pytest.fixture
def pdflip_band_off_fx(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_X_BAND_FOLLOWS_PRICE", "0")      # the same pin the original t9c uses
    with envs.FLLIPER_PDFLIP_DECIDER.override(True):
        yield


@pytest.fixture(params=["on"])
def sw(request):
    with envs.FLLIPER_PDFLIP_DECIDER.override(True):
        yield


def _lines(caplog, token):
    return [m for m in caplog.messages if token in m]


# ------------------------------------------------------------------------------------------------ pure decide(kind=arrival)
def test_decide_arrival_rule_table():
    d = lambda **kw: pp.decide("arrival", half_frac=0.5, x_own=4096, **kw)  # noqa: E731
    assert (d(uncached=25).action, d(uncached=25).rule) == (pp.ACT_D_NOW, "R2")
    assert (d(uncached=25, d_open=False, flip_decided=True).action, d(uncached=25, d_open=False, flip_decided=True).rule) \
        == (pp.ACT_RIDE_P, "R1a")
    assert d(uncached=25, d_open=False, p_phase=True).action == pp.ACT_RIDE_P
    assert (d(uncached=25, d_open=False).action, d(uncached=25, d_open=False).rule) == (pp.ACT_WAIT_SEAT, "R5")   # never P
    assert d(uncached=2048).action == pp.ACT_COLLECT and d(uncached=2048).rule == "R3"          # halb at the boundary
    assert d(uncached=2047).action == pp.ACT_D_NOW
    assert d(uncached=4096).action == pp.ACT_P_CAND and d(uncached=4096).rule == "R4"
    assert d(uncached=10, p_only=True).rule == "R9"
    assert d(uncached=None).action == pp.ACT_MEASURE and d(uncached=100, exact=False).action == pp.ACT_MEASURE
    assert pp.decide("admit", uncached=25, x_own=4096, half_frac=0.5).action == pp.ACT_D_NOW


# ------------------------------------------------------------------------------------------------ (a) pdflip-0-5
def test_a_kurz_arrival_at_a_closing_d_rides_the_decided_flip_with_a_record(sw, caplog):
    """23:06:43,576 REROUTE_P R6 decided the flip, D closes (admit_d False); 23:06:43,825 pdflip-0-5 (25 tokens) arrives:
    BATCH queued. SOLL R1a: a kurz request rides a DECIDED flip -- named ``DECISION kind=arrival action=RIDE_P rule=R1a``, no
    ``ALT:arrival_placement`` shadow, and it is no flip trigger (R2)."""
    async def body():
        f = _real_front(awake="D")
        f.admit_d = False
        with caplog.at_level(logging.INFO, logger="pdflip.front"):
            p, task = await _route(f, 25)
            await _drop(task)
        return f, p
    f, p = asyncio.run(body())
    assert p is not None and p.est_uncached <= 25 + 1
    dl = _lines(caplog, "WEG2 DECISION ")
    assert any("kind=arrival action=RIDE_P rule=R1a" in m and "rid=" in m for m in dl), dl
    assert not any("ALT:arrival_placement" in m for m in dl)
    assert p.ride_p is True and pp.immediate_park_trigger([p], RC2_X) is None      # a rider, never a trigger
    assert f.counters["arrival_ride_p_r1a"] == 1


def test_a_off_is_the_s2a_code_alt_record_and_no_ride_p(caplog):
    async def body():
        f = _real_front(awake="D")
        f.admit_d = False
        with envs.FLLIPER_PDFLIP_DECIDER.override(False), caplog.at_level(logging.INFO, logger="pdflip.front"):
            p, task = await _route(f, 25)
            await _drop(task)
        return p
    p = asyncio.run(body())
    dl = _lines(caplog, "WEG2 DECISION ")
    assert any("ALT:arrival_placement" in m for m in dl) and not any("action=RIDE_P" in m for m in dl)
    assert p.ride_p is False


def test_a_kurz_that_waits_for_a_seat_on_an_open_d_is_wait_seat_not_p(sw, caplog):
    """D open, the seat path fell through (gate/seat/budget): the kurz request is D's (WAIT_SEAT R5), never a P ride."""
    async def body():
        f = _real_front(awake="D")
        await f._d_seat.acquire()
        f.groups["D"].outstanding["running"] = time.time()
        f._d_token_budget_blocks = lambda *a, **k: True          # the pool refuses it now (FIX 4a)
        with caplog.at_level(logging.INFO, logger="pdflip.front"):
            p, task = await _route(f, 25)
            await _drop(task)
        return p
    p = asyncio.run(body())
    dl = _lines(caplog, "WEG2 DECISION ")
    assert any("kind=arrival action=WAIT_SEAT rule=R5" in m for m in dl), dl
    assert p.ride_p is False and p.d_eligible is False or p.d_budget_refused is True


def test_a_h5b_yield_only_for_a_decided_flip_not_a_foreseeable_one(sw):
    """rc18 NOT_D why=yield_or_gate: H5b/FETCH-COST yielded for a MERELY foreseeable flip (a big request waits its T_halb,
    R4). Now a kurz request rides P only when the flip is decided (go_p or D closing)."""
    f = _fresh()
    f.awake, f.state, f.admit_d = "D", "serving", True
    big = _pend("pdflip-0-9")
    big.est_uncached = X + 5000
    big.d_direct = False
    f.queue.append(big)
    assert pp.immediate_park_trigger([big], X) is big            # foreseeable: it needs P
    assert f._d_direct_yields("pdflip-0-5", uncached=25) is False
    f._set_st()["go_p"].add("pdflip-0-9")                         # committed, but D still open (no seat/KV yet): no epoch rule
    assert f._d_direct_yields("pdflip-0-5", uncached=25) is False
    f._set_st()["go_p"].clear()
    f.admit_d = False                                              # D closing for the flip: DECIDED and under way
    assert f._d_direct_yields("pdflip-0-5", uncached=25) is True


# ------------------------------------------------------------------------------------------------ (b) pdflip-2-115
def _arm_unreadable_pool(f, sleep_s=1.0):
    async def reading():
        await asyncio.sleep(sleep_s)
        return None                                              # D-POOL UNREADABLE (TimeoutError after sleep_s)
    f._d_pool_reading = reading


def _kurz_seat_wait(switch_on, sleep_s=1.0):
    async def body():
        f = _real_front(awake="D")
        await f._d_seat.acquire()
        running = Seat(f, "running", "short", tokens=2000)       # a live seat: the pool gate is armed
        f.groups["D"].outstanding["running"] = time.time()
        _arm_unreadable_pool(f, sleep_s)
        t0 = time.monotonic()
        with envs.FLLIPER_PDFLIP_DECIDER.override(switch_on):
            seat = await f._acquire_short_seat("pdflip-2-115", 500, [], uncached=500, x_set=RC2_X)
        el = time.monotonic() - t0
        if seat is not None:
            seat.release("t")
        running.release("t")
        return f, seat, el
    return asyncio.run(body())


def test_b_kurz_does_not_wait_for_an_unreadable_d_pool():
    f, seat, el = _kurz_seat_wait(True)
    assert seat is not None and el < 0.3, el                     # R2: no wait of its own
    assert f.counters["r2_kurz_pool_nowait"] == 1


def test_b_off_waits_the_timeout_as_before():
    f, seat, el = _kurz_seat_wait(False)
    assert seat is not None and el >= 0.9, el                    # the S2a path: the 1.0 s read before D_NOW (the live case)


def test_b_a_fresh_reading_still_blocks_a_kurz_request_budget_gate_stays(sw):
    """The pool gate stays a gate: a FRESH reading that says no is respected (d_budget -> WAIT_SEAT); only the round trip
    is gone."""
    async def body():
        f = _real_front(awake="D")
        await f._d_seat.acquire()
        running = Seat(f, "running", "short", tokens=2000)
        f.groups["D"].outstanding["running"] = time.time()
        f._d_pool = {"t": time.time(), "available": 10, "limit": 27466, "occupied": 0}
        why = []
        seat = await f._acquire_short_seat("k", 500, why, uncached=500, x_set=RC2_X)
        running.release("t")
        return seat, why
    seat, why = asyncio.run(body())
    assert seat is None and why == ["d_budget"]


# ------------------------------------------------------------------------------------------------ (c) pdflip-104-2646
X_C = 7622                                    # rc19 05:26 X
RID_K, RID_G = "pdflip-104-2646", "pdflip-104-2647"


def _beacon_reading(f, tmp_path, rid, t_start_s, rest_start, ext_len, *, done, ranks=1, done_ranks=None, sid=7):
    """The REAL beacon (v3 writer ``DExtendBeacon`` on ``ranks`` files, v3 reader ``read_group_full``) behind the front's reading:
    the rid is received, its extend batch starts at ``t_start_s`` (rest ``rest_start``, ``ext_len`` computed by the batch) and, with
    ``done``, the forward ENDS (``note_extend_done``) -- on ``done_ranks`` of the ranks (default all)."""
    clock = {"t": 0}
    done_ranks = ranks if done_ranks is None else done_ranks
    for r in range(ranks):
        bc = dxb.DExtendBeacon(str(tmp_path / ("D-pid%d.bin" % (r + 1))), clock_ns=lambda: clock["t"])
        bc.note_recv(rid=rid)
        clock["t"] = int(t_start_s * 1e9)
        bc.note_extend(rids=[rid], rests=[rest_start], extend_lens=[ext_len])
        if done and r < done_ranks:
            bc.note_extend_done(rids=[rid])
    f._d_ext_reading = lambda: dxb.read_group_full(str(tmp_path), "D", sid, session_of=lambda pid: sid)


def _front_c(now, kurz_u, tmp_path=None, *, done=True, ranks=1, done_ranks=None):
    f = _fake_front(running=["dec"], n=6, kv={"available": 400000, "evictable": 0})   # D decodes (R4 waits)
    f.tp_prefill_max_tokens = X_C
    f.t_awake = now - 60.0
    f._sync_batch_gate = lambda: None
    f._d_pf_book().grant(rid=RID_K, tokens=kurz_u, now=now - 0.9)
    f._d_pf_book().enter_leg2(rid=RID_K)
    f.groups["D"].outstanding[RID_K] = now - 0.9
    f._rvp_state()
    f.__dict__.setdefault("_d_prefill_legs", {})[RID_K] = now - 0.9
    f._leg2_lookahead.add(RID_K)
    F.Front._note_x_decision(f, RID_K, X_C, kurz_u)              # D-NOW at the decision X, priced 566
    # leg2 POSTs the hand-over: _front_x_headers POPS the arrival's decision record (front.py leg2) before D can begin the extend
    F.Front._front_x_headers(f, RID_K, None, False)
    assert RID_K not in (f.__dict__.get("_x_decision") or {})
    # rc19 05:26:34: the extend begins .243 (182 tokens left of the 566 after the prefix), the forward ENDS .918, the verdict falls .960
    _beacon_reading(f, tmp_path, RID_K, now - 0.717, 182, 182, done=done, ranks=ranks, done_ranks=done_ranks)
    big = F.Pending(RID_G, "/v1/messages", {}, "t", now - 2.4, asyncio.new_event_loop().create_future(),
                    est_prompt=200000, est_uncached=198000)
    big.x_set = X_C
    f.queue.append(big)
    return f


def test_c_a_prefill_whose_forward_has_ended_is_no_part_of_s_no_r6(monkeypatch, caplog, tmp_path):
    """(c) rc19 05:26:34.960 pdflip-104-2646: D's forward of the 182-token rest ended at .918, the verdict fell 42 ms later with
    ``S=198182``. SOLL R6 counts a prefill that is RUNNING; a computed one is not. Beacon v3 reports rest_after = 0 -> nothing
    counts, the big request waits its T_halb (R4). After the real leg2 header POP of the arrival record."""
    _on(monkeypatch)
    now = time.time()
    f = _front_c(now, 566, tmp_path)
    with envs.FLLIPER_PDFLIP_DECIDER.override(True), caplog.at_level(logging.INFO, logger="pdflip.front"):
        assert f._dc_carried_running() == 0                      # the rest after the forward is 0
        dc = f._decode_collect(now)
    sv = [m for m in caplog.messages if "PDFLIP SET-VERDICT" in m]
    assert dc == "hold", (dc, sv)                                  # R4: the big request waits T_halb (2.4 s < 4.5 s)
    assert not any("verdict=REROUTE_P" in m for m in sv), sv
    assert f.counters["set_reroute_p"] == 0 and f.counters["r6_redirected"] == 0 and RID_K not in f._rvp_requeue
    assert f.counters["r6_rest_computed"] >= 1


def test_c_a_prefill_whose_forward_is_still_running_counts_only_its_unlaunched_rest(monkeypatch, tmp_path):
    """The SOLL R6 / user decision 18.10. side, same fixture, same real pop. NF-RC30-R6STALE (R6 + R7): the single extend batch of
    the 182-token rest has STARTED and its forward has not ended -- the whole rest is in a launched forward, it ends on D and P
    never computes it: nothing counts (named ``r6_rest_in_flight``), the big request waits its T_halb (R4). A chunked prefill
    whose later chunk is not launched yet counts that part: rest 1000, chunk 400 in flight -> 600, S > X -> REROUTE_P."""
    _on(monkeypatch)
    now = time.time()
    f = _front_c(now, 566, tmp_path, done=False)
    with envs.FLLIPER_PDFLIP_DECIDER.override(True):
        assert f._dc_carried_running() == 0 and f.counters["r6_rest_in_flight"] >= 1
        assert f._decode_collect(now) == "hold" and f.counters["set_reroute_p"] == 0 and f.counters["r6_redirected"] == 0
    d2 = tmp_path / "chunked"
    d2.mkdir()
    f = _front_c(now, 566, d2, done=False)
    _beacon_reading(f, d2, RID_K, now - 0.717, 1000, 400, done=False)
    with envs.FLLIPER_PDFLIP_DECIDER.override(True):
        assert f._dc_carried_running() == 600
        assert f._decode_collect(now) == "P" and f.counters["set_reroute_p"] == 1 and f.counters["r6_redirected"] == 1


def test_c_rank_equality_the_max_over_ranks_keeps_the_rest_until_every_rank_is_done(monkeypatch, tmp_path):
    """R11: the front takes the MAX over the ranks -- one rank that has not finished its forward keeps the rest alive; all ranks
    done -> 0. Same answer whatever the file order. NF-RC30-R6STALE: the rest stays alive (``rest_of`` 182: D still computes it),
    but nothing of it is unlaunched, so R6 counts 0."""
    _on(monkeypatch)
    now = time.time()
    f = _front_c(now, 566, tmp_path, ranks=2, done_ranks=1)
    with envs.FLLIPER_PDFLIP_DECIDER.override(True):
        assert dxb.rest_of(f._d_ext_reading(), RID_K, now - 0.9) == 182
        assert f._dc_carried_running() == 0
    d2 = tmp_path / "all_done"
    d2.mkdir()
    f = _front_c(now, 566, d2, ranks=2, done_ranks=2)
    with envs.FLLIPER_PDFLIP_DECIDER.override(True):
        assert f._dc_carried_running() == 0


def test_c_chunked_prefill_between_two_chunks_still_has_a_rest(monkeypatch, tmp_path):
    """A chunked prefill (rest 1000, chunk 400) has 600 left after the first forward: it still runs, the 600 count."""
    _on(monkeypatch)
    now = time.time()
    f = _front_c(now, 566, tmp_path)
    d2 = tmp_path / "chunk"
    d2.mkdir()
    _beacon_reading(f, d2, RID_K, now - 0.717, 1000, 400, done=True)
    with envs.FLLIPER_PDFLIP_DECIDER.override(True):
        assert f._dc_carried_running() == 600


def test_c_off_is_the_s2a_code_s_includes_the_rest_of_a_running_prefill(monkeypatch, caplog, tmp_path):
    _on(monkeypatch)
    now = time.time()
    f = _front_c(now, 566, tmp_path, done=False)
    _beacon_reading(f, tmp_path, RID_K, now - 0.717, 1000, 818, done=False)
    with envs.FLLIPER_PDFLIP_DECIDER.override(False), caplog.at_level(logging.INFO, logger="pdflip.front"):
        assert f._dc_carried_running() == 182          # NF-RC30-R6STALE: rest 1000, chunk 818 in flight -> 182 unlaunched
        f._decode_collect(now)
    sv = [m for m in caplog.messages if "PDFLIP SET-VERDICT" in m]
    assert any("verdict=REROUTE_P" in m and "S=198182" in m for m in sv), sv


def test_c_a_halb_member_that_is_running_counts_its_rest(monkeypatch, tmp_path):
    """R6 for a prefill that began as halb (u >= halbX): while its forward runs, its rest counts -- NF-RC30-R6STALE: the part no
    forward was launched for (rest 1000, chunk 818 in flight -> 182)."""
    _on(monkeypatch)
    now = time.time()
    f = _front_c(now, 5000, tmp_path, done=False)
    _beacon_reading(f, tmp_path, RID_K, now - 0.717, 1000, 818, done=False)
    with envs.FLLIPER_PDFLIP_DECIDER.override(True):
        assert f._dc_carried_running() == 182


# ------------------------------------------------------------------------------------------------ (d) open points
def _early_h():
    loop = asyncio.new_event_loop()
    g = loop.create_future()
    g.set_result(True)
    return SimpleNamespace(gate=g, why="", rid="pdflip-1-1", t_verdict=time.time(), t_begin=time.time())


def test_d_early_flip_is_the_lead_of_a_decided_flip_no_economics_no_dwell_verdict(caplog):
    f = _fresh()
    f._flip_economics_ok = lambda _fair: False
    f._dwell_ok = lambda *a, **k: False
    h = _early_h()
    with envs.FLLIPER_PDFLIP_DECIDER.override(True), caplog.at_level(logging.INFO, logger="pdflip.front"):
        assert asyncio.run(f._early_flip_gate(h, time.time() - 0.2)) is True
    assert any("kind=arrival action=FLIP_P rule=R8" in m for m in _lines(caplog, "WEG2 DECISION "))
    h2 = _early_h()
    with envs.FLLIPER_PDFLIP_DECIDER.override(False):
        assert asyncio.run(f._early_flip_gate(h2, time.time() - 0.2)) is False and h2.why == "economics-hold"


def _refuse_nonum(f, rid, p=None):
    p = p or _pend(rid)
    body = b'{"error": "W50: re-routed through P"}'                   # D names NO extent

    async def go():
        t = asyncio.create_task(f._requeue_after_x_refusal(_Req({}, "/v1/messages"), rid, {}, "t", False, p, None, body))
        await asyncio.sleep(0.05)
        t.cancel()
        try:
            await t
        except BaseException:  # noqa: BLE001
            pass
    asyncio.run(go())
    p.fut = asyncio.new_event_loop().create_future()
    return p


def _d_awake(f):
    f.awake, f.state, f.admit_d = "D", "serving", True
    f._dc_decoding = lambda: True
    return f


def test_d_x_ref_nonum_is_measure_not_p_only_by_state(caplog):
    f = _d_awake(_fresh())
    with envs.FLLIPER_PDFLIP_DECIDER.override(True), caplog.at_level(logging.INFO, logger="pdflip.front"):
        p = _refuse_nonum(f, "pdflip-0-1")
        assert p.measure_reoffer is True and p.x_ref_nonum is False and p.d_eligible is True
        assert pp.immediate_park_trigger([p], X) is None                       # no P trigger from a state flag
        live = [q for q in f.queue if not q.fut.done()]
        assert p not in F.Front._set_cands(f, live, f._set_st())               # no number: no set member either
        assert p.est_uncached <= HALF_X - 1                                    # priced at the barrier, never an invented number
        handed = [h.rid for h in F.Front._asr_queued_short_to_d(f, live, time.time())]
    assert handed == ["pdflip-0-1"]                                            # D's again, under the MEASURE barrier
    assert f.counters["x_ref_nonum_measure"] == 1 and f.counters["measure_on_d"] >= 1
    assert any("action=MEASURE_ON_D" in m for m in _lines(caplog, "WEG2 DECISION "))


def test_d_measure_reoffer_is_never_a_p_trigger_whatever_its_stale_estimate():
    p = _pend("pdflip-0-3")
    p.est_uncached = X + 100                      # the stale whole-prompt estimate over X
    p.measure_reoffer = True
    with envs.FLLIPER_PDFLIP_DECIDER.override(True):
        assert pp.immediate_park_trigger([p], X) is None
        p.measure_reoffer = False
        assert pp.immediate_park_trigger([p], X) is p


def test_d_x_ref_nonum_off_is_p_only_by_state():
    f = _d_awake(_fresh())
    with envs.FLLIPER_PDFLIP_DECIDER.override(False):
        p = _refuse_nonum(f, "pdflip-0-1")
        assert p.x_ref_nonum is True and not p.measure_reoffer
        assert pp.immediate_park_trigger([p], X) is p


def test_d_x_ref_nonum_second_refusal_with_a_number_forms_the_class_w35_bounds_rounds():
    f = _d_awake(_fresh())
    with envs.FLLIPER_PDFLIP_DECIDER.override(True):
        p = _refuse_nonum(f, "pdflip-0-1")
        assert p.measure_reoffer
        from test_ez_s1_review_1019 import _refuse
        f._ready_for_d.clear()
        p = _refuse(f, "pdflip-0-1", 3500, p=p)                                # D now names its extent
        assert p.measure_reoffer is False and p.x_requeues == 2


# ------------------------------------------------ t9c on the real harness, switch ON (x_ref_nonum -> MEASURE, review H1)
def test_d_t9c_on_a_refusal_without_a_number_is_reoffered_to_d_not_prefilled_on_p_and_w35_still_bounds_it(pdflip_band_off_fx):
    from test_pdflip_scheduling_slice_a_0907 import Harness

    async def body():
        async with Harness(awake="D", p_concurrency=8, d_bs=4, tp_prefill_max_tokens=10 ** 9, flip_min_work_tokens=1,
                           idle_layout="D") as h:
            h.d.refuse_x_for["q1"] = 1                                   # refuse once (no extent), then serve
            status, _ = await asyncio.wait_for(h.post("q1"), timeout=30)
            assert status == 200
            assert h.front.counters["W50_PdFlipTpPrefillExceeded"] == 1
            assert h.front.counters["x_ref_nonum_measure"] == 1          # D measures: re-offered under the barrier
            assert h.p.gen_marks == []                                   # no P prefill by the state "refused"
            h.d.refuse_x_for["q2"] = 9                                   # refuse for ever
            status, text = await asyncio.wait_for(h.post("q2"), timeout=30)
            assert status == 503 and "W35 PdFlipXReQueueLoop" in text
            assert h.d.x_refusals.count("q2") == 3                       # n=3 ends it (P never ran: two re-offers), never a 4th
    asyncio.run(body())
