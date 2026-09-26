"""UNIFY H111d (NF 49458baedc) on the unified weight_updater, bb1 form, for BOTH profiles.

The waiter-side fix only helps where the sleeper's on-card staging is BOOKED on the
leg's credit counter (``staged_bytes``) until its leg-end drain refunds it. On the
unified tree both profiles stage through ONE path -- ``_weg2_stage_charge`` (H11
StageBooking) on ``_weg2_leg_credit`` -- and the sleeper publishes
``leg_complete()`` BEFORE ``_weg2_xchg_drain_outstanding()`` (the bb1 order). This
pins (a) the order, (b) that the charge books the staging on the counter whatever
the profile, and (c) the bb1 race end to end through that charge: the waker waits
for the refund instead of W35, and still refuses by name when the refund never
comes. Hermetic: no GPU.
"""
from __future__ import annotations

import inspect
import os
import threading
import time
from types import SimpleNamespace

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers.scheduler_components import weight_updater as WU  # noqa: E402
from sglang.srt.managers.weg2_memory_saver import (  # noqa: E402
    VramCredit,
    Weg2VramCreditRefused,
)
from sglang.srt.weg2 import form as F  # noqa: E402

MIB = 1 << 20
EPOCH = "1790432215.1"
STAGE = 617 * MIB
FLOOR = 701 * MIB
NEED = 1000 * MIB


def _form(profile):
    arch, experts, draft, kv = (("dense", "none", "dflash", "paged_dcp") if profile == "qwen27b"
                                else ("moe", "offload", "mtp", "qsa_forma"))
    return F.Weg2Form(arch=arch, experts=experts, draft=draft, p_draft="none", kv=kv,
                      flip="family", vision="off", profile=profile, model="m").env_value()


def test_the_sleeper_publishes_leg_complete_before_its_drain():
    """(a) the bb1 order on the unified tree: leg_complete, then the drain that
    refunds the staging -- the window the waiter-side fix covers."""
    src = inspect.getsource(WU)
    i = src.index("credit.leg_complete()")
    j = src.index("self._weg2_xchg_drain_outstanding()", i)
    assert i < j


def _sleeper(credit):
    """A weight_updater stand-in with exactly what ``_weg2_stage_charge`` reads."""
    ns = SimpleNamespace(_weg2_leg_credit=credit,
                         _weg2_free_bytes=lambda: 20000 * MIB,
                         _weg2_corridor_floor_bytes=lambda: FLOOR)
    return lambda: WU.SchedulerWeightUpdaterManager._weg2_stage_charge(ns)


@pytest.mark.parametrize("profile", ["qwen27b", "nextflash"])
def test_bb1_through_the_stage_charge(tmp_path, monkeypatch, profile):
    monkeypatch.setenv(F.FORM_ENV, _form(profile))
    c = VramCredit(f"card2-{profile}", credit_dir=str(tmp_path))
    c.begin_leg(EPOCH)
    c.publish("weights_family", 9266 * MIB)
    c.claim("weights_2..7", 11127 * MIB, epoch=EPOCH, overdraw=True)
    charge = _sleeper(c)()
    assert charge is not None, "the unified stage charge books against the leg credit"
    bookings = [charge(STAGE), charge(STAGE)]
    assert all(b is not None for b in bookings)
    for b in bookings:
        b.live()                    # H11: the cudaMalloc returned
    assert int(c.read().get("staged_bytes", 0)) == 2 * STAGE, "(b) the staging is on the counter"
    c.leg_complete()

    free = {"v": 1699 * MIB}   # 1699 - 701 = 998 < 1000: bb1's refusing reading

    def _drain():  # the sleeper's leg-end drain, ~50 ms later
        time.sleep(0.05)
        free["v"] = 2879 * MIB
        for b in bookings:
            b()                     # StageBooking: called, it refunds

    t = threading.Thread(target=_drain)
    t.start()
    try:
        c.wait_for(NEED, budget_s=5.0, tag="weights_8", free_bytes_now=1049 * MIB,
                   free_reader=lambda: free["v"], floor_bytes=FLOOR, epoch=EPOCH,
                   poll_s=0.01, alloc_poll_s=30.0)
    finally:
        t.join()


@pytest.mark.parametrize("profile", ["qwen27b", "nextflash"])
def test_a_refund_that_never_comes_still_refuses_by_name(tmp_path, monkeypatch, profile):
    monkeypatch.setenv(F.FORM_ENV, _form(profile))
    c = VramCredit(f"card2n-{profile}", credit_dir=str(tmp_path))
    c.begin_leg(EPOCH)
    c.publish("weights_family", 9266 * MIB)
    c.claim("weights_2..7", 11127 * MIB, epoch=EPOCH, overdraw=True)
    charge = _sleeper(c)()
    booking = charge(STAGE)
    assert booking is not None
    booking.live()
    c.leg_complete()
    free = {"v": 1100 * MIB}   # 1100 - 701 + 617 = 1016 >= 1000: owed, but never refunded
    with pytest.raises(Weg2VramCreditRefused):
        c.wait_for(NEED, budget_s=5.0, tag="weights_8", free_bytes_now=1049 * MIB,
                   free_reader=lambda: free["v"], floor_bytes=FLOOR, epoch=EPOCH,
                   poll_s=0.01, alloc_poll_s=0.2)
