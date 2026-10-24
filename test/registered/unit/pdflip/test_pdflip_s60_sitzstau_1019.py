"""S60 (NF rc15 19.10., 09:04-09:09Z, TTFT p90 233 s): SITZ-STAU -- who holds the D seats, and in which order.

Front-Log boot_weg2_...dauer10190714_489909d138_1019_071429.front.log. Each test pins one of the metal cases:

F1 (SOLL R5, Ueberbuchung)  09:09:32,481-,496: pdflip-108-778, -108-781, -110-782 all went to D ON CREDIT with
     ``victims=['pdflip-110-783']`` -> D-ADMIT seat 7/6, 8/6, 9/6. The victim claim lapsed as soon as the grant became
     a Seat (``_arrival_seat_taken`` settles a seated grant), so every next head planned on the same younger decode.
     rc15 whole boot: 107 on-credit grants, 51 pairs naming one victim within 1 s, D-ADMIT up to seat=14/6.
F2 (PRIO=ALTER)  09:09:32,373: pdflip-110-783 (arrived 09:09:31,5) backfilled at free=87296 while the older
     pdflip-108-780 (09:07:27, need 70242 -- fits the same room) still carried its "kv" verdict from free=42944.
F3 (PRIO=ALTER, R2/R5)  the batch gate kept kurz arrivals outside every age order: rc15 23 seatings of a YOUNGER
     hand-off while an older kurz request stood at the gate (09:06:34 pdflip-107-775 before 104-769/106-770..773);
     5 kurz requests hit the 120 s gate bound and went to P (106-770..773, 108-776).
F4 (nichts schaetzen)  09:09:17,093 ``kv_age_ms=312645``: the KV test decided on a reading from epoch 104 (D slept and
     woke twice since).
F5 (PRIO=ALTER)  the arrival stamp was POPPED at a request's first token -- every streaming decode then counted as
     the OLDEST request on D (``_younger``: no stamp = old) and was never a younger victim of an older head.
Every switch off = the path before."""
from __future__ import annotations

import asyncio
import logging
import os
import time
import types

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from test_pdflip_arrival_seat_rule_0929 import _Fut, _front  # noqa: E402

from flliper.srt.pdflip import front as F  # noqa: E402
from flliper.srt.pdflip.front import Front  # noqa: E402


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_ENABLE_ARRIVAL_SEAT_RULE", "1")
    monkeypatch.setenv("FLLIPER_PDFLIP_PBOUND_FLIP_NOW", "0")
    monkeypatch.setenv("FLLIPER_PDFLIP_PRIO_AS_AGE", "0")
    for k in ("FLLIPER_PDFLIP_ENABLE_ARRIVAL_SEAT_AGE_PLAN", "FLLIPER_PDFLIP_AGE_SEAT_ON_CREDIT",
              "FLLIPER_PDFLIP_SHORT_GATE_BY_AGE", "FLLIPER_PDFLIP_SEAT_ORDER_AFTER_FLIP"):
        monkeypatch.delenv(k, raising=False)   # defaults: on


def _stamp(f, rid, t):
    f.__dict__.setdefault("_asr_arrive", {})[rid] = t


def _kv(free):
    return {"available": int(free), "evictable": 0, "decode_clip": 4096}


def _taken(f, rid=None):
    """The seat count a waiter reads (``for_rid`` since S60 F3; the base tree has no such argument)."""
    try:
        return f._arrival_seat_taken(for_rid=rid)
    except TypeError:
        return f._arrival_seat_taken()


T0 = 1792400000.0   # 09:06:40 of the boot day (epoch seconds; only the order matters)


# ------------------------------------------------------------------------------------------------------------ F1
def _f1_front():
    """09:09:32: six on D (five older decodes + the just-backfilled pdflip-110-783), every seat taken (6/6)."""
    run = ["pdflip-106-771", "pdflip-106-772", "pdflip-106-773", "pdflip-108-776", "pdflip-108-777", "pdflip-110-783"]
    f = _front(running=run, n=6, kv=_kv(393479))
    arr = {"pdflip-106-771": T0 - 77, "pdflip-106-772": T0 - 46, "pdflip-106-773": T0 - 30, "pdflip-108-776": T0 + 4,
           "pdflip-108-777": T0 + 39, "pdflip-110-783": T0 + 171.5}
    for r, t in arr.items():
        _stamp(f, r, t)
    heads = [("pdflip-108-778", T0 + 40.4, 102181, 293), ("pdflip-108-781", T0 + 69.2, 98061, 1101),
             ("pdflip-110-782", T0 + 156.5, 97131, 43)]
    st = f._asr_st()
    for rid, t, _p, _u in heads:
        _stamp(f, rid, t)
        st["waiters"][rid] = t
    return f, heads


def _tick_all(f, heads):
    """One 50 ms round of the three waiters, oldest first, as ``_arrival_seat_wait`` runs it: the seat count, the age
    tick, and on a grant the on-credit Seat ``_acquire_short_seat_core`` takes at once (S46 F5)."""
    st = f._asr_st()
    granted = []

    async def run():
        for rid, t, est, unc in heads:
            if rid not in st["waiters"]:
                continue
            taken, n = _taken(f, rid)
            free = taken < n
            ok = await f._arrival_seat_wait_age(rid, est, 64000, unc, t, free, taken, n)
            if ok:
                granted.append(rid)
                st["waiters"].pop(rid, None)
                F.Seat(f, rid, "short", tokens=est, on_credit=True)
                f._arrival_seat_taken()          # settles the seated grant (as the next waiter's tick does)

    asyncio.run(run())
    return granted


def test_f1_one_younger_victim_makes_room_for_one_head_not_three(caplog):
    f, heads = _f1_front()
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        granted = _tick_all(f, heads)
    assert granted == ["pdflip-108-778"], granted
    plans = [r.getMessage() for r in caplog.records if "SEAT-AGE-PLAN" in r.getMessage()]
    assert sum("plan=displace" in m for m in plans) == 1, plans
    assert f.counters["arrival_seat_age_to_d"] == 1
    # the front never promises more seats than D has plus ONE per distinct claimed victim
    taken, n = f._arrival_seat_taken()
    assert taken <= n + 1


def test_f1_the_claim_ends_when_d_ran_the_head_then_the_next_head_may_plan_on_the_victim():
    f, heads = _f1_front()
    assert _tick_all(f, heads) == ["pdflip-108-778"]
    # D ran 108-778 (its first content): D's SEAT-AGE has decided -- 110-783 still runs (D found room otherwise)
    Front._age_claim_head_ran(f, "pdflip-108-778")
    assert "pdflip-108-778" not in f._asr_st().get("age_victims", {})
    assert _tick_all(f, heads) == ["pdflip-108-781"]


def test_f1_the_claim_ends_when_the_head_leg_ends_and_releases_its_seat():
    f, heads = _f1_front()
    assert _tick_all(f, heads) == ["pdflip-108-778"]
    seat = [s for s in f._d_seats_live if s.rid == "pdflip-108-778"][0]
    seat.release("leg2_finished")
    assert _tick_all(f, heads) == ["pdflip-108-781"]


# ------------------------------------------------------------------------------------------------------------ F2
def _f2_front():
    run = ["pdflip-106-771", "pdflip-106-772", "pdflip-106-773", "pdflip-108-776"]
    f = _front(running=run, n=6, kv=_kv(87296))
    for i, r in enumerate(run):
        _stamp(f, r, T0 - 100 + i)
    st = f._asr_st()
    st["waiters"]["pdflip-108-780"] = T0 + 47.4
    st["waiters"]["pdflip-110-783"] = T0 + 171.5
    _stamp(f, "pdflip-108-780", T0 + 47.4)
    _stamp(f, "pdflip-110-783", T0 + 171.5)
    # 108-780's last tick (09:09:31,478): why=kv at free=42944, taken=5, its elders block its plan
    st["fits"]["pdflip-108-780"] = False
    st.setdefault("age_plan", {})["pdflip-108-780"] = "none"
    st.setdefault("fits_at", {})["pdflip-108-780"] = (42944, 5)
    return f


def test_f2_a_younger_waiter_does_not_backfill_past_an_older_one_whose_verdict_predates_the_new_room():
    f = _f2_front()

    async def run():
        taken, n = _taken(f, "pdflip-110-783")
        return await f._arrival_seat_wait_age("pdflip-110-783", 48152, 64000, 792, T0 + 171.5, taken < n, taken, n)

    assert asyncio.run(run()) is False
    assert f.counters["arrival_seat_backfill"] == 0


def test_f2_the_older_waiter_takes_the_seat_on_its_own_tick_on_the_new_room():
    f = _f2_front()

    async def run():
        taken, n = _taken(f, "pdflip-110-783")
        assert await f._arrival_seat_wait_age("pdflip-110-783", 48152, 64000, 792, T0 + 171.5, True, taken, n) is False
        taken, n = _taken(f, "pdflip-108-780")
        return await f._arrival_seat_wait_age("pdflip-108-780", 66146, 64000, 354, T0 + 47.4, True, taken, n)

    assert asyncio.run(run()) is True


def test_f2_an_older_waiter_blocked_on_the_current_room_still_lets_the_younger_backfill():
    f = _f2_front()
    f._asr_st()["fits_at"]["pdflip-108-780"] = (87296, 4)   # its verdict IS on this room: it does not fit (106277)

    async def run():
        taken, n = _taken(f, "pdflip-110-783")
        return await f._arrival_seat_wait_age("pdflip-110-783", 48152, 64000, 792, T0 + 171.5, True, taken, n)

    assert asyncio.run(run()) is True
    assert f.counters["arrival_seat_backfill"] == 1


# ------------------------------------------------------------------------------------------------------------ F3
def _hand_off(rid, t):
    return types.SimpleNamespace(rid=rid, t_arrive=t, fut=_Fut(), est_prompt=46000, leg1_prompt_tokens=0,
                                 est_uncached=0, payload={}, lane=0)


def test_f3_an_older_kurz_does_not_wait_at_the_gate_for_a_younger_hand_off():
    """09:06:34: pdflip-107-775 (hand-off, arrived 09:06:22) in the D line; pdflip-106-770 (kurz, 09:05:17) waits."""
    f = _front(running=["a"], n=6, kv=_kv(400000))
    f._ready_for_d.append(_hand_off("pdflip-107-775", T0 - 18))
    f._batch_gate.clear()        # what ``_sync_batch_gate`` does with a non-empty D line
    asyncio.run(asyncio.wait_for(Front._short_gate_wait_by_age(f, "pdflip-106-770"), 0.5))
    assert f.counters["short_gate_age_waits"] == 0


def test_f3_a_younger_kurz_waits_for_an_older_hand_off_and_times_out_at_the_old_bound():
    f = _front(running=["a"], n=6, kv=_kv(400000))
    f.drain_deadline_s = 0.2
    f._ready_for_d.append(_hand_off("pdflip-104-765", T0 - 120))

    async def run():
        await asyncio.wait_for(Front._short_gate_wait_by_age(f, "pdflip-108-779"), 2.0)   # a hang guard

    t0 = time.monotonic()
    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(run())
    assert time.monotonic() - t0 < 1.0, "the gate wait must end at its own bound (no hang)"
    assert f.counters["short_gate_age_waits"] == 1


def test_f3_an_older_kurz_takes_a_free_seat_past_a_younger_hand_off_in_the_d_line():
    """The whole kurz seat path: D line = one younger hand-off, five decodes run, one seat left. Before: the kurz stood
    at the closed batch gate until the drain deadline and fell through to BATCH (``None``); now it takes the seat."""
    f = _front(running=["r1", "r2", "r3", "r4", "r5"], n=6, kv=_kv(400000))
    f.drain_deadline_s = 0.3
    f._ready_for_d.append(_hand_off("pdflip-107-775", T0 - 18))
    f._batch_gate.clear()
    _stamp(f, "pdflip-106-770", T0 - 83)
    seat = asyncio.run(asyncio.wait_for(
        f._acquire_short_seat("pdflip-106-770", 47272, [], max_tokens=64000, uncached=424), 2.0))
    assert seat is not None and seat.rid == "pdflip-106-770"
    assert f.counters["short_gate_timeout_to_batch"] == 0


def test_f3_the_waiters_seat_count_holds_only_older_d_line_entries():
    f = _front(running=["r1", "r2", "r3", "r4", "r5"], n=6, kv=_kv(400000))
    f._ready_for_d.append(_hand_off("pdflip-107-775", T0 - 18))
    assert f._arrival_seat_taken()[0] == 6                           # the controller's count: unchanged
    assert f._arrival_seat_taken(for_rid="pdflip-106-770")[0] == 5   # the older kurz: the younger hand-off holds none
    assert f._arrival_seat_taken(for_rid="pdflip-108-779")[0] == 6   # a younger kurz: the older hand-off holds one


def test_f3_the_d_line_yields_to_an_older_waiter_that_is_not_blocked():
    f = _front(running=["r1"], n=6, kv=_kv(400000))
    st = f._asr_st()
    st["waiters"]["pdflip-106-770"] = T0 - 83
    p = _hand_off("pdflip-107-775", T0 - 18)
    assert Front._d_line_hold_for_older_waiter(f, p) == "pdflip-106-770"
    # blocked on the current room (does not fit, its elders block its plan): the hand-off backfills
    st["fits"]["pdflip-106-770"] = False
    st.setdefault("age_plan", {})["pdflip-106-770"] = "none"
    st.setdefault("fits_at", {})["pdflip-106-770"] = (None, 1)
    st["last_kv_free"] = None
    assert Front._d_line_hold_for_older_waiter(f, p) is None
    # a younger waiter never holds the line
    st["waiters"].clear()
    st["waiters"]["pdflip-108-779"] = T0 + 40
    assert Front._d_line_hold_for_older_waiter(f, p) is None


def test_f3_off_is_the_old_gate(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_SHORT_GATE_BY_AGE", "0")
    f = _front(running=["r1", "r2", "r3", "r4", "r5"], n=6, kv=_kv(400000))
    f._ready_for_d.append(_hand_off("pdflip-107-775", T0 - 18))
    assert f._arrival_seat_taken(for_rid="pdflip-106-770")[0] == 6
    f._asr_st()["waiters"]["pdflip-106-770"] = T0 - 83
    assert Front._d_line_hold_for_older_waiter(f, _hand_off("pdflip-107-775", T0 - 18)) is None
    f.drain_deadline_s = 0.1
    f._batch_gate.clear()
    assert asyncio.run(f._acquire_short_seat_core("pdflip-106-770", 3000, [], 256, 424, None, time.time())) is None
    assert f.counters["short_gate_timeout_to_batch"] == 1


# ------------------------------------------------------------------------------------------------------------ F4
def _f4_front(cached_t, awake_t):
    f = _front(running=["r1"], n=6)
    del f._arrival_seat_kv_reading       # the real reading path
    st = f._asr_st()
    st["kv"] = (cached_t, _kv(393479))   # the 09:04:05 reading (epoch 104)
    f.t_awake = awake_t
    calls = []

    async def fetch(gen):
        calls.append(gen)
        await asyncio.sleep(0.2)          # D is inside a pass: longer than the 0.05 s budget
        got = _kv(42944)
        st["kv"] = (time.time(), got)
        return got

    f._arrival_seat_kv_fetch = fetch
    return f, calls


def test_f4_a_reading_from_before_this_d_phase_never_decides():
    now = time.time()
    f, calls = _f4_front(cached_t=now - 312.6, awake_t=now - 16.0)
    got = asyncio.run(Front._arrival_seat_kv_reading(f))
    assert got["available"] == 42944 and calls
    assert f.counters["arrival_seat_kv_prev_phase"] == 1


def test_f4_inside_one_phase_the_budget_still_decides_on_the_last_reading():
    # EZ-S1 (plan A10): the form below is the rc17 one (switch FLLIPER_PDFLIP_EXACT_CLASS off); with it on a reading
    # older than FLLIPER_PDFLIP_KV_FRESH_MS is ABSENT -- see the next test
    from flliper.srt.environ import envs

    with envs.FLLIPER_PDFLIP_EXACT_CLASS.override(False):
        now = time.time()
        f, calls = _f4_front(cached_t=now - 3.0, awake_t=now - 16.0)
        got = asyncio.run(Front._arrival_seat_kv_reading(f))
        assert got["available"] == 393479          # a pass boundary late, same phase: the budget's trade, unchanged
        assert f.counters["arrival_seat_kv_prev_phase"] == 0


def test_f4_ez_s1_inside_one_phase_a_reading_older_than_the_bound_is_absent():
    now = time.time()
    f, calls = _f4_front(cached_t=now - 3.0, awake_t=now - 16.0)
    got = asyncio.run(Front._arrival_seat_kv_reading(f))
    assert got is None                         # older than FLLIPER_PDFLIP_KV_FRESH_MS: not a number (plan A10)
    assert f.counters["arrival_seat_kv_prev_phase"] == 0 and f.counters["arrival_seat_kv_absent"] == 1


# ------------------------------------------------------------------------------------------------------------ F5
def test_f5_the_first_token_keeps_the_arrival_stamp_and_notes_the_ttft_once():
    f = _front(running=["pdflip-108-776", "pdflip-110-783"], n=6)
    _stamp(f, "pdflip-110-783", T0 + 171.5)
    assert Front._asr_ttft_arrival(f, "pdflip-110-783") == T0 + 171.5
    assert Front._asr_ttft_arrival(f, "pdflip-110-783") is None        # a resumed leg's first content: no new TTFT
    assert f._asr_arrival_of("pdflip-110-783") == T0 + 171.5           # the age stays


def test_f5_the_leg2_first_content_reads_the_stamp_instead_of_popping_it():
    import inspect

    src = inspect.getsource(Front)
    assert "_asr_arrival_of(rid, pop=True)" not in src
    assert "_asr_ttft_arrival(self, rid)" in src and "_age_claim_head_ran(self, rid)" in src


def test_f5_a_streaming_younger_decode_is_a_victim_of_an_older_head():
    """pdflip-110-783 streams (first token given) -- before, its stamp was gone and it counted as OLD: the older head
    had no younger victim and waited ``plan=none``."""
    run = ["pdflip-106-771", "pdflip-106-772", "pdflip-106-773", "pdflip-108-776", "pdflip-108-777", "pdflip-110-783"]
    f = _front(running=run, n=6, kv=_kv(393479))
    for i, r in enumerate(run[:-1]):
        _stamp(f, r, T0 - 100 + i)
    _stamp(f, "pdflip-110-783", T0 + 171.5)
    Front._asr_ttft_arrival(f, "pdflip-110-783")                        # its first content
    plan = f._arrival_seat_age_verdict("pdflip-108-778", T0 + 40.4, False, 0)
    assert plan == ["pdflip-110-783"]


# ------------------------------------------------------------------------------------------------------------ F6
def test_f6_a_parked_request_the_front_holds_no_leg_for_is_aborted_on_d():
    """08:48:47 PARK-RUNNING ``unknown_to_front=['pdflip-80-579', 'pdflip-84-637', 'pdflip-84-643', 'pdflip-85-644']``:
    their clients left at 08:48:20, D kept them (80-579 decoded to 09:13:55)."""
    f = _front(running=["pdflip-106-771"], n=6)
    f._ready_for_d.append(_hand_off("pdflip-104-769", T0 - 148))

    async def run():
        n = Front._d_orphan_abort(f, ["pdflip-80-579", "pdflip-106-771", "pdflip-104-769",
                                      "HEALTH_CHECK_5bf35897a4c449be91d1059bb392f438"])
        await asyncio.gather(*list(f.__dict__.get("_d_orphan_tasks", ())))
        return n

    assert asyncio.run(run()) == 1
    assert f.rpc_calls == [("/abort_request", {"rid": "pdflip-80-579"})]
    assert f.counters["d_orphan_abort"] == 1


def test_f6_the_park_hands_its_unknown_rids_to_the_orphan_abort():
    import inspect

    import ast
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(Front._wait_bound_park)))
    hits = [n for n in ast.walk(tree) if isinstance(n, ast.If) and isinstance(n.test, ast.Name)
            and n.test.id == "unknown"
            and any(isinstance(c, ast.Call) and getattr(c.func, "attr", "") == "_d_orphan_abort"
                    for b in n.body for c in ast.walk(b))]
    assert hits, "the park hands its unknown_to_front rids to the orphan abort (guarded only by their presence)"


def test_f6_off_names_only(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_D_ORPHAN_ABORT", "0")
    f = _front(running=[], n=6)
    assert Front._d_orphan_abort(f, ["pdflip-80-579"]) == 0
    assert f.rpc_calls == []


def test_f3_a_grant_that_never_became_a_seat_holds_no_seat_and_no_d_line():
    """A granted kurz request that falls through (D budget) left its grant in ``granted`` for 10 s: a phantom seat in
    ``taken`` and -- with the D-line yield -- a held D line."""
    f = _front(running=["r1"], n=6, kv=_kv(400000))
    f._d_token_budget_blocks = lambda *a, **k: True
    _stamp(f, "pdflip-106-770", T0 - 83)
    refused = []
    seat = asyncio.run(f._acquire_short_seat("pdflip-106-770", 47272, refused, max_tokens=64000, uncached=424))
    assert seat is None and "d_budget" in refused
    assert "pdflip-106-770" not in f._asr_st()["granted"]
    assert Front._d_line_hold_for_older_waiter(f, _hand_off("pdflip-107-775", T0 - 18)) is None


# ------------------------------------------------------------------------------------------------- S60r (review)
def _orphan_run(f, rids):
    async def run():
        n = Front._d_orphan_abort(f, rids)
        await asyncio.gather(*list(f.__dict__.get("_d_orphan_tasks", ())))
        return n

    return asyncio.run(run())


def _abort_rule(monkeypatch, exact: bool):
    from flliper.srt.pdflip import abort_match

    monkeypatch.setenv(abort_match.ENV, "1" if exact else "0")
    abort_match._reset_for_tests()
    monkeypatch.setattr(abort_match, "_ARMED", None)  # restored -> re-read after the test


def test_s60r_a1_an_orphan_whose_rid_prefixes_a_live_rid_is_not_aborted(monkeypatch):
    """Under the PREFIX abort rule (``FLLIPER_PDFLIP_ABORT_EXACT_RID=0``: ``rid.startswith(abort.rid)``) D's abort of
    the orphan ``pdflip-2-6`` would drop the live ``pdflip-2-60`` (boot 1018_153151 holds both in one epoch)."""
    _abort_rule(monkeypatch, exact=False)
    f = _front(running=["pdflip-2-60"], n=6)
    assert _orphan_run(f, ["pdflip-2-6", "pdflip-2-60"]) == 0
    assert f.rpc_calls == []
    assert f.counters["d_orphan_abort_prefix_skip"] == 1


def test_s60r_a1_a_prefix_inside_the_park_list_is_skipped_the_other_orphan_still_goes(monkeypatch):
    _abort_rule(monkeypatch, exact=False)
    f = _front(running=[], n=6)
    assert _orphan_run(f, ["pdflip-2-6", "pdflip-2-61"]) == 1
    assert f.rpc_calls == [("/abort_request", {"rid": "pdflip-2-61"})]


def test_s60r_a1_exact_rule_aborts_the_orphan_and_spares_the_live_rid(monkeypatch):
    """#1580 ABORT-EXACT-RID (default on): D's abort of ``pdflip-2-6`` hits only ``pdflip-2-6`` (and its non-digit
    children), so the orphan IS aborted -- no shadow skip, the live ``pdflip-2-60`` is not touched."""
    _abort_rule(monkeypatch, exact=True)
    f = _front(running=["pdflip-2-60"], n=6)
    assert _orphan_run(f, ["pdflip-2-6", "pdflip-2-60"]) == 1
    assert f.rpc_calls == [("/abort_request", {"rid": "pdflip-2-6"})]
    assert f.counters.get("d_orphan_abort_prefix_skip", 0) == 0
    f2 = _front(running=[], n=6)
    assert _orphan_run(f2, ["pdflip-2-6", "pdflip-2-61"]) == 2


def test_s60r_a2_a_rid_the_front_still_holds_on_p_or_queued_is_no_orphan():
    """A W50 / park-handback requeue pops ``D.outstanding`` and goes on via P under the SAME rid; D's abort would drop
    that rid's tail staging and park parts under the hand-off coming back."""
    f = _front(running=[], n=6)
    f.groups["P"] = types.SimpleNamespace(name="P", outstanding={"pdflip-104-766": 1.0}, url="http://p")
    f.queue.append(_hand_off("pdflip-104-767", T0))
    f._rvp_inflight = {"pdflip-104-768"}
    f._asr_st()["waiters"]["pdflip-104-765"] = T0
    assert _orphan_run(f, ["pdflip-104-765", "pdflip-104-766", "pdflip-104-767", "pdflip-104-768"]) == 0
    assert f.rpc_calls == []


def test_s60r_a3_a_grant_that_became_a_seat_is_settled_at_once():
    f = _front(running=["r1"], n=6, kv=_kv(400000))
    _stamp(f, "pdflip-106-770", T0 - 83)
    seat = asyncio.run(f._acquire_short_seat("pdflip-106-770", 4000, [], max_tokens=256, uncached=424))
    assert seat is not None
    assert "pdflip-106-770" not in f._asr_st()["granted"]
    seat.release("leg2_finished")
    # the leg ended: nothing older holds the D line for a younger hand-off
    assert Front._d_line_hold_for_older_waiter(f, _hand_off("pdflip-107-775", T0 - 18)) is None


def test_s60r_a3_a_lapsed_or_running_grant_does_not_hold_the_d_line():
    f = _front(running=["pdflip-104-769"], n=6, kv=_kv(400000))
    st = f._asr_st()
    st["granted"]["pdflip-104-768"] = time.time() - 11.0     # lost grant, past the 10 s lapse
    st["granted"]["pdflip-104-769"] = time.time()            # runs on D already
    assert Front._d_line_hold_for_older_waiter(f, _hand_off("pdflip-107-775", T0 - 18)) is None
    st["granted"]["pdflip-104-767"] = time.time()            # a live, unseated, older grant still holds it
    assert Front._d_line_hold_for_older_waiter(f, _hand_off("pdflip-107-775", T0 - 18)) == "pdflip-104-767"


def test_s60r_a3_a_request_cancelled_between_grant_and_seat_leaves_no_grant():
    f = _front(running=["r1"], n=6, kv=_kv(400000))

    async def core(rid, *a, **k):
        f._asr_st()["granted"][rid] = time.time()
        raise asyncio.CancelledError()

    f._acquire_short_seat_core = core
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(f._acquire_short_seat("pdflip-106-771", 4000, [], max_tokens=256, uncached=424))
    assert "pdflip-106-771" not in f._asr_st()["granted"]
