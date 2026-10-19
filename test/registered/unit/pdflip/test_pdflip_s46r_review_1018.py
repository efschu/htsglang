"""S46r (review of S46, NF 18.10.): the danger directions of F4/F5 that the S46 tests left open.

(a) a seat taken ON CREDIT holds no permit: released while still on credit it returns NOTHING to the semaphore (else the
    front holds more than ``--d-bs`` permits); two debtors are settled oldest first; the permit count never drifts.
(a) ``_handoff_in_flight`` counts a credit seat like a normal one: a batch hand-off waiting for its POST is not masked
    by a credit request already in D.outstanding (the D->P park / flip guards, FIX 2 round 2, W1/W2 class).
(a) no credit past an OLDER request in the front's D line (D's SEAT-AGE cannot see it; the permit would pass to the
    younger debtor).
(b) F4 holds no hand-off while the P->D flip runs (P serves nothing more; the hold would only delay the dormant read).
"""
from __future__ import annotations

import asyncio
import os
import time

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from test_pdflip_s46_sitzwarten_1018 import _env, _f5_front, _pending, _real_front, _stamp  # noqa: E402,F401

from flliper.srt.pdflip import front as F  # noqa: E402
from flliper.srt.pdflip.front import Front  # noqa: E402


def _credit(f, head):
    async def run():
        _stamp(f, head, time.time())
        return await asyncio.wait_for(f._acquire_short_seat(head, 2000, [], max_tokens=256, uncached=400), 1.5)

    return asyncio.run(run())


def test_a_credit_seat_released_on_credit_returns_no_permit():
    f, seats = _f5_front()
    seat = _credit(f, "pdflip-9-40")
    assert seat is not None and seat.on_credit is True
    assert f._d_seat._value == 0
    seat.release("aborted")                       # the credit request died before any normal seat freed
    assert f._d_seat._value == 0, "a credit seat must not mint a permit: the front would hold d_bs + 1"
    assert f.counters["d_seat_credit_settled"] == 0
    # the six normal seats free -> exactly d_bs permits, never more
    for s in seats.values():
        s.release("leg2_finished")
    assert f._d_seat._value == f.d_bs


def test_a_two_debtors_are_settled_oldest_first_and_the_permits_add_up():
    f, seats = _f5_front()
    d1 = F.Seat(f, "pdflip-9-41", "short", tokens=10, on_credit=True)
    d1.t_taken -= 5.0
    d2 = F.Seat(f, "pdflip-9-42", "short", tokens=10, on_credit=True)
    seats["pdflip-1-1"].release("leg2_finished")
    assert d1.on_credit is False and d2.on_credit is True
    seats["pdflip-1-2"].release("leg2_finished")
    assert d2.on_credit is False
    assert f._d_seat._value == 0
    for s in list(seats.values()) + [d1, d2]:
        s.release("leg2_finished")
    assert f._d_seat._value == f.d_bs


def test_a_handoff_in_flight_counts_a_credit_seat():
    f, seats = _f5_front()
    D = f.groups["D"]
    # five decodes in D, the sixth normal seat is a batch hand-off whose POST has not come yet
    D.outstanding.pop("pdflip-9-50", None)
    assert Front._handoff_in_flight(f) == 1
    # a credit request lands in D.outstanding: the hand-off is still in flight
    c = F.Seat(f, "pdflip-9-40", "short", tokens=10, on_credit=True)
    D.outstanding["pdflip-9-40"] = object()
    assert Front._handoff_in_flight(f) == 1, "the credit request masked the batch hand-off in flight"
    assert Front.seats_free(f) == 0
    c.release("leg2_finished")


def test_a_no_credit_past_an_older_request_in_the_front_d_line():
    f, _ = _f5_front()

    async def run():
        loop = asyncio.get_running_loop()
        older = F.Pending("pdflip-9-30", "/generate", {}, "x", time.time(), loop.create_future(), est_prompt=500,
                          est_uncached=500, d_eligible=True)
        f._ready_for_d.append(older)
        _stamp(f, "pdflip-9-40", time.time())
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(f._acquire_short_seat("pdflip-9-40", 2000, [], max_tokens=256, uncached=400),
                                   0.8)
        # S60 F3: the kurz request now already waits at the age gate behind the OLDER D-line entry (it never reaches
        # the credit test); either way no seat is taken on credit past the older request
        assert (f.counters["arrival_seat_age_credit_older_in_line"] >= 1
                or f.counters["short_gate_age_waits"] >= 1)
        assert f.counters["arrival_seat_age_seat_on_credit"] == 0

    asyncio.run(run())


def test_a_a_younger_d_line_entry_does_not_block_the_credit():
    f, _ = _f5_front()

    async def run():
        loop = asyncio.get_running_loop()
        younger = F.Pending("pdflip-9-60", "/generate", {}, "x", time.time(), loop.create_future(), est_prompt=500,
                            est_uncached=500, d_eligible=True)
        f._ready_for_d.append(younger)
        _stamp(f, "pdflip-9-40", time.time())
        seat = await asyncio.wait_for(f._acquire_short_seat("pdflip-9-40", 2000, [], max_tokens=256, uncached=400),
                                      1.5)
        assert seat is not None and seat.on_credit is True

    asyncio.run(run())


def test_b_no_dormant_hold_while_the_p_to_d_flip_runs():
    async def body():
        f = _real_front()
        f.awake, f.state, f.dormant_admit = "P", "serving", True
        young = _pending("pdflip-110-1000", 4105)
        f.queue.append(_pending("pdflip-110-997"))     # left over in P's queue at the phase end
        assert Front._dormant_hold_for_older(f, young) is True
        f.state = "flipping"                            # flip("P","D") began: P serves nothing more
        assert Front._dormant_hold_for_older(f, young) is False

    asyncio.run(body())
