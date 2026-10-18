"""S46 (NF 18.10.): short requests waited 60-73 s for their first token (rc12 1018_153151, 18:50Z; rc13 1018_200305).

pdflip-110-998 (u=1147, kurz): 18:50:17 no D seat (6/6) -> 18:50:36,202 ``SEAT-AGE-PLAN plan=fits seat_free=True`` and in
the same tick ``ARRIVAL-SEAT why=kv taken=5 ... kv need=134779 free=242816`` -> the one seat went to the older 996, its
D prefill plus the halb 1000 made S > X (R6) -> flip, 998 rode it (R1a) -> on P behind the YOUNGEST (1000, then 999)
-> back on D the seat went to 999 (younger) before 996/997/998 -> ttft 66 s.

F1 (R2/R5)  a seat freed during the KV await while the KV fits: the request fits NOW -- no lost tick, no false ``why=kv``.
F2 (SA #244) the D line re-picks by age when an older hand-off joined it during the seat wait.
F3 (SA #244) a BATCH fall-through is queued for P at its age, not behind younger entries.
F4 (SA #244) while P is awake a younger hand-off takes no seat before an older request still on P (D is asleep).
F5 (AGE PLAN / SITZPOLITIK VEREINT) a to_d grant with every front seat held goes to D on credit -- D's SEAT-AGE seat
             trigger displaces the youngest; before, the head waited at the front's semaphore (rc12: 124 grants, p50 5.3 s,
             max 72 s) and D never saw it.
Every switch off = the path before (one test each)."""
from __future__ import annotations

import asyncio
import logging
import os
import time

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from test_pdflip_arrival_seat_rule_0929 import X, _front  # noqa: E402

from flliper.srt.pdflip import front as F  # noqa: E402
from flliper.srt.pdflip.front import Front  # noqa: E402

KV_OK = {"available": 400000, "evictable": 0, "decode_clip": 4096}


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_ENABLE_ARRIVAL_SEAT_RULE", "1")
    monkeypatch.setenv("FLLIPER_PDFLIP_PBOUND_FLIP_NOW", "0")
    for k in ("FLLIPER_PDFLIP_ENABLE_ARRIVAL_SEAT_AGE_PLAN", "FLLIPER_PDFLIP_SEAT_ROTATE",
              "FLLIPER_PDFLIP_AGE_SEAT_ON_CREDIT", "FLLIPER_PDFLIP_SEAT_ORDER_AFTER_FLIP"):
        monkeypatch.delenv(k, raising=False)   # defaults: on


def _stamp(f, rid, t):
    f.__dict__.setdefault("_asr_arrive", {})[rid] = t


# ------------------------------------------------------------------------------------------------------------ F1
def _f1_front():
    f = _front(running=["pdflip-1-1", "pdflip-1-2", "pdflip-1-3", "pdflip-1-4", "pdflip-1-5", "pdflip-1-6"], n=6,
               kv=KV_OK)
    now = time.time()
    for i in range(1, 7):
        _stamp(f, "pdflip-1-%d" % i, now - 100.0 + i)
    D = f.groups["D"]

    async def kv_while_a_seat_frees():
        D.outstanding.pop("pdflip-1-6", None)    # pdflip-110-993 finished during the KV read (D-REFILL 18:50:36,2)
        await asyncio.sleep(0.01)
        return KV_OK

    f._arrival_seat_kv_reading = kv_while_a_seat_frees
    return f


def test_f1_a_seat_freed_during_the_kv_read_is_granted_in_the_same_tick(caplog):
    f = _f1_front()
    rid = "pdflip-110-998"
    _stamp(f, rid, time.time())
    f._asr_st()["waiters"][rid] = time.time()

    async def run():
        return await f._arrival_seat_wait_age(rid, 1147, 256, 1147, time.time(), False, 6, 6)

    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        assert asyncio.run(run()) is True
    lines = [r.getMessage() for r in caplog.records if "ARRIVAL-SEAT" in r.getMessage()]
    assert not any("why=kv" in m for m in lines), lines
    assert any("verdict=d_prefill" in m and "no seat" not in m for m in lines), lines
    assert f.counters["arrival_seat_seat_freed_in_tick"] == 1


def test_f1_a_kv_that_really_does_not_fit_still_waits_named_kv(caplog):
    f = _f1_front()
    D = f.groups["D"]

    async def kv_short():
        D.outstanding.pop("pdflip-1-6", None)
        return {"available": 1000, "evictable": 0, "decode_clip": 4096}

    f._arrival_seat_kv_reading = kv_short
    rid = "pdflip-110-998"
    _stamp(f, rid, time.time())
    f._asr_st()["waiters"][rid] = time.time()

    async def run():
        return await f._arrival_seat_wait_age(rid, 130683, 64000, 1147, time.time(), False, 6, 6)

    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        assert asyncio.run(run()) is False
    assert any("why=kv" in r.getMessage() for r in caplog.records)


# ------------------------------------------------------------------------------------------------------------ F5
def _f5_front():
    """Every front seat held: six running decodes with a Seat each, the youngest arrived AFTER the head."""
    now = time.time()
    run = ["pdflip-1-1", "pdflip-1-2", "pdflip-1-3", "pdflip-1-4", "pdflip-1-5", "pdflip-9-50"]
    f = _front(running=run, n=6, kv=KV_OK)
    seats = {}
    for i, r in enumerate(run):
        _stamp(f, r, now - 100.0 + i if r != "pdflip-9-50" else now + 0.5)
        seats[r] = None

    async def take():
        for r in run:
            await f._d_seat.acquire()
            seats[r] = F.Seat(f, r, "short", tokens=3000)

    asyncio.run(take())
    return f, seats


def test_f5_an_age_plan_head_goes_to_d_on_credit_when_every_front_seat_is_held():
    f, seats = _f5_front()
    head = "pdflip-9-40"

    async def run():
        _stamp(f, head, time.time())
        seat = await asyncio.wait_for(f._acquire_short_seat(head, 2000, [], max_tokens=256, uncached=400), 1.5)
        assert seat is not None and seat.rid == head and seat.on_credit is True
        assert f.counters["arrival_seat_age_to_d"] == 1
        assert f.counters["arrival_seat_age_seat_on_credit"] == 1
        assert not f.rpc_calls, "the front parks nothing: D's SEAT-AGE displaces"
        # the next normal seat that frees passes its permit to the debtor -- the semaphore stays at 0 permits
        seats["pdflip-1-1"].release("leg2_finished")
        assert seat.on_credit is False
        assert f._d_seat._value == 0
        assert f.counters["d_seat_credit_settled"] == 1
        # the settled seat releases like any other
        seat.release("leg2_finished")
        assert f._d_seat._value == 1

    asyncio.run(run())


def test_f5_switch_off_the_head_waits_at_the_semaphore(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_AGE_SEAT_ON_CREDIT", "0")
    f, _ = _f5_front()
    head = "pdflip-9-40"

    async def run():
        _stamp(f, head, time.time())
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(f._acquire_short_seat(head, 2000, [], max_tokens=256, uncached=400), 0.8)

    asyncio.run(run())


def test_f5_no_credit_without_victims_a_free_permit_is_taken_normally():
    f = _front(running=["pdflip-1-1"], n=6, kv=KV_OK)
    _stamp(f, "pdflip-1-1", time.time() - 50.0)

    async def run():
        _stamp(f, "pdflip-9-40", time.time())
        seat = await asyncio.wait_for(f._acquire_short_seat("pdflip-9-40", 2000, [], max_tokens=256, uncached=400),
                                      1.0)
        assert seat is not None and seat.on_credit is False
        assert f.counters["arrival_seat_age_seat_on_credit"] == 0

    asyncio.run(run())


# ------------------------------------------------------------------------------------------------- F2 / F3 / F4
def _real_front():
    return Front("http://p", "http://d", "D", "s46", "", 0, 0, {}, 45.0, tp_prefill_max_tokens=6119)


def _pending(rid, tokens=1147):
    fut = asyncio.get_running_loop().create_future()
    p = F.Pending(rid, "/generate", {}, "x", time.time(), fut, est_prompt=tokens, est_uncached=tokens,
                  d_eligible=True)
    return p


async def _drop(*tasks):
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


def _f2_run(monkeypatch, on: bool):
    if not on:
        monkeypatch.setenv("FLLIPER_PDFLIP_SEAT_ORDER_AFTER_FLIP", "0")

    async def body():
        f = _real_front()
        for _ in range(f.d_bs):
            await f._d_seat.acquire()          # every D seat held by a running decode
        young = _pending("pdflip-110-999")
        f._ready_for_d.append(young)           # P served 999 first (18:50:46,3)
        adm = asyncio.ensure_future(f.d_admitter())
        try:
            await asyncio.sleep(0.2)           # the admitter picked 999 and waits for a seat
            old = _pending("pdflip-110-996")
            f._ready_for_d.append(old)         # 996 finished P later (18:50:53,2) -- older
            await asyncio.sleep(0.1)
            f._d_seat.release()                # pdflip-110-995 finished (18:51:04,6)
            done, _ = await asyncio.wait({old.fut, young.fut}, timeout=2.0, return_when=asyncio.FIRST_COMPLETED)
            return [p.rid for p in (old, young) if p.fut in done]
        finally:
            await _drop(adm)

    return asyncio.run(body())


def test_f2_the_seat_goes_to_the_older_hand_off_that_joined_during_the_wait(monkeypatch):
    assert _f2_run(monkeypatch, True) == ["pdflip-110-996"]


def test_f2_switch_off_the_picked_younger_one_takes_it(monkeypatch):
    assert _f2_run(monkeypatch, False) == ["pdflip-110-999"]


def test_f3_a_fall_through_is_queued_at_its_age(monkeypatch):
    async def body():
        f = _real_front()
        f.queue.append(_pending("pdflip-110-1000", 4105))   # COLLECT-P-QUEUED 18:50:36,393
        for r in ("pdflip-110-999", "pdflip-110-997", "pdflip-110-998"):   # NOT_D fall-throughs, in that order
            Front._queue_at_age(f, _pending(r))
        new = _pending("pdflip-110-1001")
        Front._queue_at_age(f, new)                            # a new arrival: the end, as before
        return [q.rid for q in f.queue]

    assert asyncio.run(body()) == ["pdflip-110-997", "pdflip-110-998", "pdflip-110-999", "pdflip-110-1000",
                                   "pdflip-110-1001"]


def test_f3_switch_off_appends(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_SEAT_ORDER_AFTER_FLIP", "0")

    async def body():
        f = _real_front()
        f.queue.append(_pending("pdflip-110-1000", 4105))
        Front._queue_at_age(f, _pending("pdflip-110-997"))
        return [q.rid for q in f.queue]

    assert asyncio.run(body()) == ["pdflip-110-1000", "pdflip-110-997"]


def test_f4_while_p_is_awake_a_younger_hand_off_waits_for_the_older_one_on_p(monkeypatch):
    async def body():
        f = _real_front()
        f.awake, f.state, f.dormant_admit = "P", "serving", True
        young = _pending("pdflip-110-1000", 4105)
        older_on_p = _pending("pdflip-110-996", 2667)
        Front._p_legs_live(f)[id(older_on_p)] = (older_on_p, None)   # its leg 1 runs on P
        assert Front._dormant_hold_for_older(f, young) is True
        older_queued = _pending("pdflip-110-997")
        Front._p_legs_live(f).clear()
        f.queue.append(older_queued)
        assert Front._dormant_hold_for_older(f, young) is True
        f.queue.clear()
        assert Front._dormant_hold_for_older(f, young) is False   # nothing older on P: it takes the seat
        f.queue.append(older_queued)
        f.awake = "D"
        assert Front._dormant_hold_for_older(f, young) is False   # D awake: the D line's own age order decides
        # a finished leg not yet in the line counts 2 s, then lapses
        f.awake = "P"
        f.queue.clear()
        Front._p_legs_live(f)[id(older_on_p)] = (older_on_p, time.time() - 5.0)
        assert Front._dormant_hold_for_older(f, young) is False
        assert not Front._p_legs_live(f)

    asyncio.run(body())


def test_f4_the_admitter_holds_the_dormant_seat_for_the_older_request(monkeypatch):
    async def body():
        f = _real_front()
        f.awake, f.state, f.dormant_admit, f.admit_d = "P", "serving", True, True
        young = _pending("pdflip-110-1000", 4105)
        older = _pending("pdflip-110-996", 2667)
        Front._p_legs_live(f)[id(older)] = (older, None)
        f._ready_for_d.append(young)
        adm = asyncio.ensure_future(f.d_admitter())
        try:
            await asyncio.sleep(0.3)
            assert not young.fut.done(), "the younger hand-off took the seat while the older one was on P"
            Front._p_legs_live(f).pop(id(older))
            f._ready_for_d.append(older)                     # the older leg 1 is done
            await asyncio.wait_for(older.fut, 2.0)
        finally:
            await _drop(adm)

    asyncio.run(body())


def test_f4_switch_off_no_hold(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_SEAT_ORDER_AFTER_FLIP", "0")

    async def body():
        f = _real_front()
        f.awake, f.state, f.dormant_admit = "P", "serving", True
        young = _pending("pdflip-110-1000", 4105)
        older = _pending("pdflip-110-996", 2667)
        Front._p_legs_live(f)[id(older)] = (older, None)
        return Front._dormant_hold_for_older(f, young)

    assert asyncio.run(body()) is False
