"""HANDOVER-KVFITS 1024 (SOLL R5, R11; deskq/done/27b-lifetime-gate-1024.md, Offene Risiken 1).

27B int4ar _1023_223241 pdflip-30-274 (kurz u=415, prompt 58911): ARRIVAL-SEAT ``wait_seat why=kv need=63007 >
free=43101``, then RIDE_P (R1a), P leg 1 23:10:50 -- and the SAME second the front's D line gave it ``D-ADMIT seat=5/6
source=batch``. The d_admitter checked only D's HOST pool (``_d_budget_terms``, #915), never D's DEVICE KV
(``_asr.kv_fits``) as ARRIVAL-SEAT does. D's device pool held the running 98k/115k decodes (available=42809
evictable=0); D refused 30-274 by ``gate=lifetime`` 1293 times for 190.5 s, until 30-264 finished. The load-back of the
hand-off (d_prefix 0 on D, N-1 rows from L2/L3) is DEVICE need: the need is the whole prompt + D's decode clip.

Fix: the hand-off takes no D seat while D's device KV cannot hold it -- the same ``_arrival_seat_kv`` test -- it stays
parked in the front at its age (R5), and goes when a running request ends (the Seat release asks D's reading at once).

Danger directions, one test group each:
  * Uebergabe trotz vollem Device-KV  -- a, h (recheck after the seat wait), l (the load-back counts)
  * Request verhungert in der Front    -- b, c (released at the finish), d (nothing runs: never held), e (no number:
                                          never a first hold), g (backfill stops at the bound), i (kurz arrivals are not
                                          held behind it)
"""
from __future__ import annotations

import asyncio
import logging
import os
import time

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import front as F  # noqa: E402
from flliper.srt.pdflip.front import Front, Seat  # noqa: E402

HEAD = "pdflip-30-274"
OLD = "pdflip-30-264"
YOUNG = "pdflip-30-280"

# D TP0's pdflip_kv at 23:10:59 (available=42809 evictable=0) with 30-264 running; and after it finished
FULL = {"available": 42809, "evictable": 0, "decode_clip": 4096, "running_rids": [OLD], "displaceable_rids": [OLD]}
ROOM = {"available": 150000, "evictable": 0, "decode_clip": 4096, "running_rids": [], "displaceable_rids": []}


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_ENABLE_ARRIVAL_SEAT_RULE", "1")
    monkeypatch.setenv("FLLIPER_PDFLIP_PBOUND_FLIP_NOW", "0")
    for k in ("FLLIPER_PDFLIP_ENABLE_ARRIVAL_SEAT_AGE_PLAN", "FLLIPER_PDFLIP_SEAT_ORDER_AFTER_FLIP",
              "FLLIPER_PDFLIP_SHORT_GATE_BY_AGE", "FLLIPER_PDFLIP_KV_FRESH_MS", "FLLIPER_PDFLIP_EXACT_CLASS"):
        monkeypatch.delenv(k, raising=False)   # defaults: on / 1000 ms


def _front(box):
    f = Front("http://p", "http://d", "D", "hkv", "", 0, 0, {}, 45.0, tp_prefill_max_tokens=12288)
    f.awake, f.state, f.admit_d = "D", "serving", True

    async def _kv():
        return box.get("kv")

    async def _no_pool():
        return None

    f._arrival_seat_kv_reading = _kv
    f._d_reading_if_armed = _no_pool          # D's host pool is not the question here: it fits
    return f


def _stamp(f, rid, t):
    f.__dict__.setdefault("_asr_arrive", {})[rid] = t


def _pending(rid, prompt=58911, uncached=415, max_tokens=64000):
    fut = asyncio.get_running_loop().create_future()
    return F.Pending(rid, "/generate", {"max_tokens": max_tokens}, "x", time.time(), fut, est_prompt=prompt,
                     est_uncached=uncached, leg1_prompt_tokens=prompt, d_eligible=True)


async def _running(f, rid, tokens=98000):
    """A decode running on D: its front seat (one permit) and its D ledger entry."""
    await f._d_seat.acquire()
    f.groups["D"].outstanding[rid] = time.time()
    return Seat(f, rid, "batch", tokens=tokens)


async def _drop(*tasks):
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


# ------------------------------------------------------------------------- a/b: held while full, released at the finish
def test_a_b_handover_waits_in_the_front_while_device_kv_is_full_and_goes_when_the_running_one_ends(caplog):
    caplog.set_level(logging.INFO)

    async def body():
        box = {"kv": FULL}
        f = _front(box)
        now = time.time()
        _stamp(f, OLD, now - 200.0)
        _stamp(f, HEAD, now - 30.0)
        seat = await _running(f, OLD)
        head = _pending(HEAD)
        f._ready_for_d.append(head)
        adm = asyncio.ensure_future(f.d_admitter())
        try:
            await asyncio.sleep(0.4)
            # a: no seat, no D-ADMIT -- before the fix: D-ADMIT seat=2/6 source=batch, then 1293 lifetime refusals on D
            assert not head.fut.done(), "the hand-off took a D seat while D's device KV could not hold it"
            assert f.counters["d_admit_kv_held"] >= 1
            assert (f.__dict__.get("_d_admit_skip") or {}).get(HEAD) == "kv_device"
            assert f._d_seat._value == f.d_bs - 1, "a permit was taken for the held hand-off"
            assert list(f._ready_for_d) == [head], "the held head must stay parked in the front at its age"
            assert "D-ADMIT-KV-WAIT rid=%s" % HEAD in caplog.text
            assert "need=63007" in caplog.text        # l: prompt 58911 + clip 4096 -- the load-back is device need
            # b: 30-264 finished (23:14:08) -- D's KV holds it now: it is handed over
            box["kv"] = ROOM
            f.groups["D"].outstanding.pop(OLD, None)
            seat.release("leg2_finished")
            await asyncio.wait_for(asyncio.shield(head.fut), 2.0)
            assert f.counters["d_admit_kv_released"] == 1
            assert "D-ADMIT-KV-FITS rid=%s" % HEAD in caplog.text
            assert "D-ADMIT rid=%s" % HEAD in caplog.text
        finally:
            await _drop(adm)

    asyncio.run(body())


# ------------------------------------------------------------------------- c: the finish is the event (no 250 ms poll)
def test_c_a_running_request_ending_asks_d_for_its_kv_at_once(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_KV_FRESH_MS", "30000")   # without the release kick the FULL reading stood 30 s

    async def body():
        box = {"kv": FULL}
        f = Front("http://p", "http://d", "D", "hkv", "", 0, 0, {}, 45.0, tp_prefill_max_tokens=12288)
        f.awake, f.state, f.admit_d = "D", "serving", True

        async def _no_pool():
            return None

        f._d_reading_if_armed = _no_pool
        fetches = []

        async def _fetch(gen):                    # D's /server_info, as _arrival_seat_kv_fetch stores it
            st = f._asr_st()
            got = dict(box["kv"])
            fetches.append(got["available"])
            if int(st.get("kv_gen", 0)) == int(gen):
                st["kv"] = (time.time(), got)
            return got

        f._arrival_seat_kv_fetch = _fetch
        now = time.time()
        _stamp(f, OLD, now - 200.0)
        _stamp(f, HEAD, now - 30.0)
        seat = await _running(f, OLD)
        head = _pending(HEAD)
        f._ready_for_d.append(head)
        adm = asyncio.ensure_future(f.d_admitter())
        try:
            await asyncio.sleep(0.4)
            assert not head.fut.done()
            box["kv"] = ROOM
            f.groups["D"].outstanding.pop(OLD, None)
            seat.release("leg2_finished")
            await asyncio.wait_for(asyncio.shield(head.fut), 2.0)
            assert f.counters["kv_fetch_seat_release_kv_hold"] == 1
            assert fetches[-1] == ROOM["available"]
        finally:
            await _drop(adm)

    asyncio.run(body())


def test_c2_no_kick_without_a_held_head():
    async def body():
        f = _front({"kv": ROOM})
        seat = await _running(f, OLD)
        seat.release("leg2_finished")
        assert f.counters["kv_fetch_seat_release_kv_hold"] == 0

    asyncio.run(body())


# ------------------------------------------------------------------------- d: nothing runs on D -> never held
def test_d_nothing_running_on_d_never_holds_the_head():
    async def body():
        f = _front({"kv": FULL})
        head = _pending(HEAD)
        f._ready_for_d.append(head)
        assert await Front._d_line_kv_holds(f, head) is False   # no finish can free a row: waiting would starve it
        adm = asyncio.ensure_future(f.d_admitter())
        try:
            await asyncio.wait_for(asyncio.shield(head.fut), 2.0)
        finally:
            await _drop(adm)
        assert f.counters["d_admit_kv_held"] == 0

    asyncio.run(body())


# ------------------------------------------------------------------------- e: no number
def test_e_no_reading_gives_no_first_hold_but_a_held_head_keeps_its_hold():
    async def body():
        box = {"kv": None}
        f = _front(box)
        now = time.time()
        _stamp(f, OLD, now - 200.0)
        _stamp(f, HEAD, now - 30.0)
        await _running(f, OLD)
        head = _pending(HEAD)
        f._ready_for_d.append(head)
        assert await Front._d_line_kv_holds(f, head) is False    # never measured: as ARRIVAL-SEAT kv=unread
        box["kv"] = FULL
        assert await Front._d_line_kv_holds(f, head) is True
        box["kv"] = None                                          # D inside a long pass: no number this time
        assert await Front._d_line_kv_holds(f, head) is True, "an absent reading released a held head (fail-open)"
        assert f.counters["d_admit_kv_held_unread"] == 1
        box["kv"] = ROOM
        assert await Front._d_line_kv_holds(f, head) is False

    asyncio.run(body())


# ------------------------------------------------------------------------- f: AGE PLAN (PRIO=ALTER)
def test_f_an_older_head_goes_to_d_when_displacing_a_younger_decode_makes_it_fit():
    async def body():
        kv = dict(FULL, running_rids=[YOUNG], displaceable_rids=[YOUNG])
        f = _front({"kv": kv})
        now = time.time()
        _stamp(f, HEAD, now - 30.0)
        _stamp(f, YOUNG, now - 10.0)                              # the running decode is YOUNGER than the head
        await _running(f, YOUNG, tokens=60000)
        head = _pending(HEAD)
        f._ready_for_d.append(head)
        assert await Front._d_line_kv_holds(f, head) is False     # D's SEAT-AGE displaces for it (D decides)
        st = f._asr_st()
        assert st["age_victims"][HEAD] == [YOUNG]
        assert f.counters["d_admit_kv_age_to_d"] == 1
        assert Front._age_claim_live(f, st, HEAD) is True          # its victim is spoken for while it waits for a seat
        f._ready_for_d.clear()
        assert Front._age_claim_live(f, st, HEAD) is False

    asyncio.run(body())


def test_f2_older_running_decodes_block_the_plan_the_head_waits():
    async def body():
        f = _front({"kv": FULL})
        now = time.time()
        _stamp(f, OLD, now - 200.0)
        _stamp(f, HEAD, now - 30.0)
        await _running(f, OLD)
        head = _pending(HEAD)
        f._ready_for_d.append(head)
        assert await Front._d_line_kv_holds(f, head) is True
        assert HEAD not in (f._asr_st().get("age_victims") or {})

    asyncio.run(body())


# ------------------------------------------------------------------------- g: KV backfill of the D line, bounded
def _g_run(bound):
    async def body():
        f = _front({"kv": FULL})
        f.d_wait_bound_s = bound
        now = time.time()
        _stamp(f, OLD, now - 200.0)
        _stamp(f, HEAD, now - 30.0)
        _stamp(f, YOUNG, now - 20.0)
        await _running(f, OLD)
        head = _pending(HEAD)
        young = _pending(YOUNG, prompt=20000, uncached=20000)     # need 24096 <= 42809: it fits
        f._ready_for_d.extend([head, young])
        Front._d_kv_hold_st(f)[HEAD] = now - 5.0                  # the head waits 5 s for device KV already
        adm = asyncio.ensure_future(f.d_admitter())
        try:
            await asyncio.sleep(0.5)
            return head.fut.done(), young.fut.done(), f.counters["d_line_kv_backfill"]
        finally:
            await _drop(adm)

    return asyncio.run(body())


def test_g_a_younger_hand_off_that_fits_backfills_past_the_kv_held_head():
    assert _g_run(60.0) == (False, True, 1)


def test_g2_past_the_bound_no_younger_hand_off_overtakes_the_held_head():
    assert _g_run(1.0) == (False, False, 0)


# ------------------------------------------------------------------------- h: the seat wait changes D's KV
def test_h_kv_rechecked_after_the_seat_wait_the_permit_goes_back():
    async def body():
        box = {"kv": ROOM}
        f = _front(box)
        now = time.time()
        _stamp(f, OLD, now - 200.0)
        _stamp(f, HEAD, now - 30.0)
        await _running(f, OLD)
        for _ in range(f.d_bs - 1):
            await f._d_seat.acquire()                             # every seat held
        head = _pending(HEAD)
        f._ready_for_d.append(head)
        adm = asyncio.ensure_future(f.d_admitter())
        try:
            await asyncio.sleep(0.3)                              # it fit, it waits at the semaphore
            box["kv"] = FULL                                      # meanwhile D's device pool filled up
            f._d_seat.release()
            await asyncio.sleep(0.3)
            assert not head.fut.done(), "handed over on a KV verdict older than the seat wait"
            assert f._d_seat._value == 1, "the permit of the held head was not given back"
            assert f.counters["d_admit_kv_held_after_seat"] >= 1
        finally:
            await _drop(adm)

    asyncio.run(body())


# ------------------------------------------------------------------------- i: a kurz arrival is not held behind it
def test_i_kurz_arrival_is_not_held_at_the_short_gate_by_a_kv_held_older_head():
    async def body():
        f = _front({"kv": FULL})
        f.drain_deadline_s = 0.3
        head = _pending(HEAD)
        f._ready_for_d.append(head)
        Front._d_admit_skip_note(f, HEAD, "kv_device")
        await asyncio.wait_for(Front._short_gate_wait_by_age(f, "pdflip-30-290"), 0.2)   # returns at once
        Front._d_admit_skip_note(f, HEAD, None)                   # a head that takes the next seat still holds it
        with pytest.raises(asyncio.TimeoutError):
            await Front._short_gate_wait_by_age(f, "pdflip-30-290")

    asyncio.run(body())


# ------------------------------------------------------------------------- k: the rule off = the path before
def test_k_rule_off_the_hand_off_takes_its_seat_as_before(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_ENABLE_ARRIVAL_SEAT_RULE", "0")

    async def body():
        f = _front({"kv": FULL})
        now = time.time()
        _stamp(f, OLD, now - 200.0)
        _stamp(f, HEAD, now - 30.0)
        await _running(f, OLD)
        head = _pending(HEAD)
        f._ready_for_d.append(head)
        adm = asyncio.ensure_future(f.d_admitter())
        try:
            await asyncio.wait_for(asyncio.shield(head.fut), 2.0)
        finally:
            await _drop(adm)

    asyncio.run(body())
