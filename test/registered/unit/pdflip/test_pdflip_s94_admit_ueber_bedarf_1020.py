"""S94 (NF 20.10.): the front hands D legs D cannot begin -- the AGE PLAN "to D" + SEAT-AGE ON-CREDIT path.

THE METAL
* rc19 ``..._10e9b2b492_1019_230333``, 07:15:48 pdflip-262-3339: ``ARRIVAL-SEAT verdict=d_prefill taken=6 n=6 no seat;
  kv need=134763 free=312036 (ladder 524288-212252=312036, stage free 312000)``, victims ['pdflip-258-3324'], ON-CREDIT
  seat=7/6. D 07:15:57 ``D-WALL-HEAD ... available=13568 reported_evictable=305024 payable_evictable=0: the backup wall
  stands`` -- the front's "free" was D's REPORTED evictable, not what D could pay. 07:15:57 pdflip-262-3340 went to D
  with ``kv=unread`` (seat 9/6). D ran 0 decodes from 07:14:58 ("6 queued, 0 running"): no victim was displaceable.
* rc20 ``..._b40f47c91c_1020_081757``, 08:34:48 pdflip-12-82: ``kv need=91354 > free=33664 (ladder 524288-495152=29136,
  stage free 33664)``, AGE PLAN victims ['pdflip-8-58'] -- 8-58 was no running decode on D but a queued load-back
  (D 08:34:51 ``LOAD-BACK ROOM rid=pdflip-8-58 need=81475 available=33728``); ON-CREDIT seat=9/6. The unbegun legs held
  front seats until the 08:36:07-08:36:34 stall.

THE FIX (SOLL R5: a kurz request is prefilled on D only when D has a seat for it, else it waits / rides a decided
flip, R1a; R7: D not taking one is no flip reason; R11: the front alone decides, D only reports):
* F1 D publishes ``pdflip_kv.payable_evictable`` / ``backup_wall``; under the wall the ARRIVAL-SEAT KV room is
  ``available + payable_evictable`` and a displaced victim frees no KV row (D's ``kv_displace_would_fit``).
* F2 no AGE PLAN grant without a D reading (``kv=unread``): the head waits for its seat.
* F3 D publishes ``pdflip_kv.running_rids``; the AGE PLAN plans only on decodes D runs.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest  # noqa: E402

from test_pdflip_arrival_seat_rule_0929 import _front  # noqa: E402

from flliper.srt.pdflip import arrival_seat_rule as R  # noqa: E402
from flliper.srt.pdflip import front as F  # noqa: E402

OLD = ["pdflip-1-1", "pdflip-1-2", "pdflip-1-3", "pdflip-1-4", "pdflip-1-5"]

# rc19 07:15:57 D-WALL-HEAD / 07:15:48 front ladder
RC19_WALL = {"available": 13568, "evictable": 305024, "payable_evictable": 0, "backup_wall": True,
             "ladder_ceiling": 524288, "ladder_used": 212252, "decode_clip": 4096}


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_ENABLE_ARRIVAL_SEAT_RULE", "1")
    monkeypatch.setenv("FLLIPER_PDFLIP_PBOUND_FLIP_NOW", "0")
    for k in ("FLLIPER_PDFLIP_ENABLE_ARRIVAL_SEAT_AGE_PLAN", "FLLIPER_PDFLIP_SEAT_ROTATE",
              "FLLIPER_PDFLIP_AGE_SEAT_ON_CREDIT", "FLLIPER_PDFLIP_SEAT_ORDER_AFTER_FLIP"):
        monkeypatch.delenv(k, raising=False)   # defaults: on


def _stamp(f, rid, t):
    f.__dict__.setdefault("_asr_arrive", {})[rid] = t


def _full_front(kv, victim="pdflip-9-50", victim_tokens=3000):
    """Every front seat held: five older running decodes and one YOUNGER than the head (the AGE PLAN's victim)."""
    now = time.time()
    run = OLD + [victim]
    f = _front(running=run, n=6, kv=kv)
    for i, r in enumerate(OLD):
        _stamp(f, r, now - 100.0 + i)
    _stamp(f, victim, now + 0.5)

    async def take():
        for r in run:
            await f._d_seat.acquire()
            F.Seat(f, r, "short", tokens=victim_tokens if r == victim else 3000)

    asyncio.run(take())
    return f


def _try(f, head, est, timeout):
    async def run():
        _stamp(f, head, time.time())
        try:
            return await asyncio.wait_for(f._acquire_short_seat(head, est, [], max_tokens=256, uncached=400), timeout)
        except asyncio.TimeoutError:
            return None
    return asyncio.run(run())


# --------------------------------------------------------------------------------------------- the rule, D's numbers
def test_f1_rule_rc19_3339_the_wall_room_is_available_plus_payable():
    need = 130667 + 4096                                       # 07:15:48: prompt 130667 + D clip 4096 = 134763
    fits, why = R.kv_fits(need, RC19_WALL)
    assert fits is False, why
    assert "payable room 13568 = available 13568 + payable 0" in why
    assert R.kv_free(RC19_WALL) == 13568
    assert R.payable_room(RC19_WALL) == 13568 and R.backup_wall(RC19_WALL) is True


def test_f1_rule_unchanged_without_the_wall_or_on_an_old_d():
    old_d = {k: v for k, v in RC19_WALL.items() if k not in ("payable_evictable", "backup_wall")}
    # D's 07:15:57 split (13568 + 305024 = 318592 stage, ladder 312036): the larger decides, as before S94
    assert R.kv_fits(134763, old_d) == (True, "kv need=134763 free=318592 (ladder 524288-212252=312036, stage free "
                                               "318592)")
    no_wall = dict(old_d, payable_evictable=305024, backup_wall=False)
    assert R.kv_fits(134763, no_wall) == R.kv_fits(134763, old_d)
    assert R.kv_free(no_wall) == R.kv_free(old_d) == 318592
    assert R.payable_room(no_wall) is None


# --------------------------------------------------------------------------------------------- F1 through the front
def test_f1_rc19_3339_under_the_wall_no_age_plan_grant_no_credit(caplog):
    """rc19 07:15:48: the head does not fit D's payable room and victims free nothing under the wall -> it waits
    for room (R5); no to_d, no ON-CREDIT seat 7/6. RED before S94: free read 312036, the plan handed it to D."""
    kv = dict(RC19_WALL, running_rids=OLD + ["pdflip-9-50"])
    f = _full_front(kv)
    caplog.set_level(logging.INFO, logger="pdflip.front")
    seat = _try(f, "pdflip-262-3339", 130667, 0.8)
    assert seat is None, "the head went to D although D could not pay its KV"
    assert f.counters["arrival_seat_age_to_d"] == 0
    assert f.counters["arrival_seat_age_seat_on_credit"] == 0
    assert not any(s.on_credit for s in f._d_seats_live)
    assert any("S94 backup wall on D" in r.getMessage() for r in caplog.records)


def test_f1_under_the_wall_a_head_that_fits_the_payable_room_still_takes_a_seat_by_displacement():
    """The wall blocks only what D cannot pay: a head whose KV fits available + payable and only lacks a SEAT goes
    to D on credit (D's seat trigger displaces the youngest running decode -- k = max(1, k) on D too)."""
    kv = dict(RC19_WALL, running_rids=OLD + ["pdflip-9-50"])
    f = _full_front(kv)
    seat = _try(f, "pdflip-9-40", 2000, 1.5)
    assert seat is not None and seat.on_credit is True
    assert f.counters["arrival_seat_age_to_d"] == 1


# ------------------------------------------------------------------------------------------------------------ F2
def test_f2_rc19_3340_no_reading_no_age_plan_grant(caplog):
    """rc19 07:15:57 pdflip-262-3340 ``kv=unread`` -> to D on credit (seat 9/6). Without D's number the head waits
    for its seat (R5). RED before S94."""
    f = _full_front(None)
    caplog.set_level(logging.INFO, logger="pdflip.front")
    seat = _try(f, "pdflip-262-3340", 2000, 0.8)
    assert seat is None
    assert f.counters["arrival_seat_age_to_d"] == 0
    assert f.counters["arrival_seat_age_no_reading"] >= 1
    assert any("plan=held-no-reading" in r.getMessage() for r in caplog.records)


def test_f2_no_reading_with_a_free_seat_the_plain_seat_path_is_unchanged():
    f = _front(running=OLD, n=6, kv=None)
    seat = _try(f, "pdflip-9-40", 2000, 1.5)
    assert seat is not None and seat.on_credit is False


# ------------------------------------------------------------------------------------------------------------ F3
def test_f3_rc20_12_82_the_victim_d_does_not_run_is_no_victim():
    """rc20 08:34:48: need 91354 > free 33664, the AGE PLAN's victim pdflip-8-58 (front price 81411) was a queued
    load-back on D, not a running decode. D's running set does not name it -> no plan, no to_d, no seat 9/6.
    RED before S94."""
    kv = {"available": 33664, "evictable": 0, "payable_evictable": 0, "backup_wall": False,
          "ladder_ceiling": 524288, "ladder_used": 495152, "decode_clip": 4096, "running_rids": list(OLD)}
    f = _full_front(kv, victim="pdflip-8-58", victim_tokens=81411)
    seat = _try(f, "pdflip-12-82", 87258, 0.8)
    assert seat is None, "a head was handed to D on a victim D does not run"
    assert f.counters["arrival_seat_age_to_d"] == 0
    assert f.counters["arrival_seat_age_seat_on_credit"] == 0


def test_f3_the_same_head_goes_to_d_when_d_runs_the_victim():
    """The other direction (AGE PLAN unchanged, Nutzer 30.09.): D runs the younger decode -> it is displaceable,
    the head goes to D on credit."""
    kv = {"available": 33664, "evictable": 0, "payable_evictable": 0, "backup_wall": False,
          "ladder_ceiling": 524288, "ladder_used": 495152, "decode_clip": 4096,
          "running_rids": OLD + ["pdflip-8-58"]}
    f = _full_front(kv, victim="pdflip-8-58", victim_tokens=81411)
    seat = _try(f, "pdflip-12-82", 87258, 1.5)
    assert seat is not None and seat.on_credit is True


def test_f3_rc19_d_runs_nothing_no_victim_at_all():
    """rc19 07:15:49: D ran 0 decodes (07:14:58 "6 queued, 0 running") while the front planned on 258-3324 / the
    57 ms old credit head 262-3339. ``running_rids=[]`` -> nothing to displace -> the head waits."""
    kv = {"available": 400000, "evictable": 0, "decode_clip": 4096, "running_rids": []}
    f = _full_front(kv)
    assert _try(f, "pdflip-262-3338", 2000, 0.8) is None
    assert f.counters["arrival_seat_age_to_d"] == 0


def test_f3_old_d_without_running_rids_is_unchanged():
    f = _full_front({"available": 400000, "evictable": 0, "decode_clip": 4096})
    seat = _try(f, "pdflip-9-40", 2000, 1.5)
    assert seat is not None and seat.on_credit is True


# --------------------------------------------------------------------------------------------- D publishes the numbers
def test_d_publishes_payable_wall_and_running_rids(monkeypatch):
    """The D half: ``Scheduler.get_internal_state`` puts payable_evictable / backup_wall / running_rids into
    ``pdflip_kv`` (source check on the one site -- the full method needs a live scheduler)."""
    import inspect

    from flliper.srt.managers import scheduler as S

    src = inspect.getsource(S.Scheduler.get_internal_state)
    i = src.index('ret["pdflip_kv"] = {')
    j = src.index("except Exception", i)
    block = src[i:j]
    for key in ('["payable_evictable"]', '["backup_wall"]', '["running_rids"]', "payable_evictable_or("):
        assert key in block, key


def test_f1_under_the_wall_a_big_victim_frees_nothing():
    """Under the wall the victim's FRONT price (here 200000, enough for the deficit 121195 of 3339) is no room: D sizes
    it 0 (``kv_displace_would_fit``: its retained KV goes back to the tree as unpayable as the rest). Without the wall
    term the plan would hand the head to D on that price."""
    kv = dict(RC19_WALL, running_rids=OLD + ["pdflip-9-50"])
    f = _full_front(kv, victim_tokens=200000)
    assert _try(f, "pdflip-262-3339", 130667, 0.8) is None
    assert f.counters["arrival_seat_age_to_d"] == 0


# ------------------------------------------------------------------------------- review A2: the basis is the head's own
def test_a2_another_waiters_reading_between_the_heads_test_and_its_grant_does_not_lend_it_a_number(caplog):
    """S94 review A2: the head's own KV test had NO reading (``kv=unread``); during its later await (an older waiter
    keeps it from being ``first`` -> ``_asr_older_blocked``) another waiter's KV test read a number. The AGE PLAN and
    F2 must decide on the HEAD's basis -- before, the shared ``st["last_kv_*"]`` slot carried the other reading into
    F2 and the head went to D on credit without a number of its own."""
    f = _full_front(None)
    present = {"available": 400000, "evictable": 0, "decode_clip": 4096, "running_rids": OLD + ["pdflip-9-50"]}
    st = f._asr_st()
    st["waiters"]["pdflip-0-1"] = time.time() - 50.0            # an older waiter: the head is not ``first``
    st.setdefault("fits", {})["pdflip-0-1"] = False
    st.setdefault("age_plan", {})["pdflip-0-1"] = "none"
    own_reading = f._arrival_seat_kv_reading

    async def _older_blocked_with_a_foreign_reading(rid, t_mine, seat_is_free):
        async def _present():
            return present
        f._arrival_seat_kv_reading = _present
        try:
            await F.Front._arrival_seat_kv(f, 2000, 256, "pdflip-0-1", 400)   # the other waiter's test of the tick
        finally:
            f._arrival_seat_kv_reading = own_reading
        return True

    f._asr_older_blocked = _older_blocked_with_a_foreign_reading
    caplog.set_level(logging.INFO, logger="pdflip.front")
    assert _try(f, "pdflip-262-3340", 2000, 0.8) is None
    assert f.counters["arrival_seat_age_to_d"] == 0
    assert f.counters["arrival_seat_age_seat_on_credit"] == 0
    assert f.counters["arrival_seat_age_no_reading"] >= 1


def test_a2_the_heads_own_number_still_takes_it_to_d_when_a_foreign_test_read_none():
    """The other direction: the head HAD a number (D runs the younger decode, KV fits) and a foreign test in between
    read none -- the head's own basis decides, it still goes to D by displacement."""
    kv = {"available": 400000, "evictable": 0, "decode_clip": 4096, "running_rids": OLD + ["pdflip-9-50"]}
    f = _full_front(kv)
    st = f._asr_st()
    st["waiters"]["pdflip-0-1"] = time.time() - 50.0
    st.setdefault("fits", {})["pdflip-0-1"] = False
    st.setdefault("age_plan", {})["pdflip-0-1"] = "none"
    own_reading = f._arrival_seat_kv_reading

    async def _older_blocked_with_no_reading(rid, t_mine, seat_is_free):
        async def _none():
            return None
        f._arrival_seat_kv_reading = _none
        try:
            await F.Front._arrival_seat_kv(f, 2000, 256, "pdflip-0-1", 400)
        finally:
            f._arrival_seat_kv_reading = own_reading
        return True

    f._asr_older_blocked = _older_blocked_with_no_reading
    seat = _try(f, "pdflip-9-40", 2000, 1.5)
    assert seat is not None and seat.on_credit is True


def test_a1_the_wall_room_is_named_tp0():
    """S94 review A1 (transition): the room is D TP0's -- an upper bound of the group -- and the line says so."""
    fits, why = R.kv_fits(130667 + 4096, RC19_WALL)
    assert not fits and "room=TP0" in why
