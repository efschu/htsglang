"""REVIEW of WAKE-FLOOR-1024 (desk/27b-wake-floor-1024 @ bd169f2f08).

Three findings, each as a test that states the DESIRED property. They are
expected RED on bd169f2f08 -- that is the finding, not a broken test.

R1 (instrument, OOM acceptance): the x37 loop path re-grades the waker's
   overdraw under the counter lock, but a refusal there is SILENT. The metal
   case of the ticket (TP0 weights_5, 08:17:38 / 08:18:59, "0.0 s INTO THE
   WAIT") took exactly that loop path, so the acceptance count
   "PDFLIP-CREDIT-RECHECK" the author names would read 0 on the very path the
   fix is for.

R2 (OOM direction, residual hole): the early exit with an OPEN counter that
   has nothing published yet (credit_bytes == 0, epoch matches) returns
   without booking anything -- no consumed, no pending. The co-located
   sleeper's overdraw debit then still stages on the waker's claimed, not yet
   mapped bytes: the same double grant, one branch away.

R3 (instrument, duo acceptance) -- RETRACTED by its own run (remote
   508d5dee73_20261024T111755Z, green): with the real monotonic clock the
   wait's ``waited`` is a few microseconds even when nothing is owed, so the
   PDFLIP-WAKE-KV-STAGE-WAIT line is printed (waited_ms=0
   peer_staged_left=0). Kept green as the guard of that property: a repeat of
   duo's 106 with nothing owed must still leave the line.
"""
from __future__ import annotations

import logging
import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.managers import pdflip_memory_saver as PMS  # noqa: E402

MIB = 1 << 20
NEAR_OOM_MIB = 400


class _Card:
    def __init__(self, free_mib):
        self.free_b = int(free_mib) * MIB

    def free(self):
        return self.free_b

    def alloc(self, nbytes):
        self.free_b -= int(nbytes)


class _Grab(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


@pytest.fixture
def grab():
    h = _Grab()
    loggers = [PMS.logger]
    from flliper.srt.managers.scheduler_components import weight_updater as wu
    loggers.append(wu.logger)
    olds = []
    for lg in loggers:
        olds.append(lg.level)
        lg.addHandler(h)
        lg.setLevel(logging.DEBUG)
    try:
        yield h
    finally:
        for lg, old in zip(loggers, olds):
            lg.removeHandler(h)
            lg.setLevel(old)


def test_r1_loop_path_recheck_refusal_is_logged(tmp_path, grab):
    c = PMS.VramCredit("GPU-rv-r1", credit_dir=str(tmp_path))
    ep = "1792829161.11"
    c.begin_leg(ep)
    c.publish("published", 100 * MIB)
    c.claim("prior", 200 * MIB, epoch=ep, overdraw=True)   # consumed > credit: balance 0
    c.claim_landed("prior", epoch=ep)
    card = _Card(3596)
    need, floor, stage = 1678 * MIB, 767 * MIB, 1656 * MIB
    orig = c._staging_to_subtract
    fired = {"n": 0}

    def _racing(before):
        v = orig(before)
        if not fired["n"]:
            # the sleeper books AND mallocs its staging between the waker's
            # loop reading and the waker's claim (the metal interleaving)
            fired["n"] = 1
            assert c.debit("ipc-stage", stage, free_bytes=card.free(), floor_bytes=0)
            card.alloc(stage)
            c.mark_live(stage)
        return v

    c._staging_to_subtract = _racing
    with pytest.raises(PMS.PdFlipVramCreditRefused):
        c.wait_for(need, budget_s=0.3, tag="weights_5",
                   free_bytes_now=1976 * MIB,          # entry reading short -> loop
                   free_reader=card.free, floor_bytes=floor, epoch=ep)
    assert fired["n"] == 1
    # the fix worked: nothing was claimed on the peer's bytes ...
    assert card.free() >> 20 == 3596 - 1656
    # ... but the refusal must be visible in the log the acceptance counts
    assert any("PDFLIP-CREDIT-RECHECK" in ln for ln in grab.lines), (
        "loop-path recheck refusal left no PDFLIP-CREDIT-RECHECK line")


def test_r2_open_counter_with_nothing_published_still_books_the_claim(tmp_path):
    c = PMS.VramCredit("GPU-rv-r2", credit_dir=str(tmp_path))
    ep = "1792829161.12"
    c.begin_leg(ep)                          # the sleeper's leg is open, nothing published
    card = _Card(3600)
    need = 1678 * MIB
    rec = c.wait_for(need, budget_s=0.3, tag="weights_0",
                     free_bytes_now=card.free(), free_reader=card.free,
                     floor_bytes=767 * MIB, epoch=ep)
    assert rec["waited_s"] == 0.0            # early exit: the waker resumes now
    # before the waker's map lands, the sleeper stages on the same card
    staged = c.debit("ipc-stage", 1656 * MIB, free_bytes=card.free(),
                     floor_bytes=1055 * MIB, free_reader=card.free)
    if staged:
        card.alloc(1656 * MIB)
    card.alloc(need)                         # resume(weights_0) maps
    assert card.free() >> 20 >= NEAR_OOM_MIB, (
        f"double grant through the credit==0 early exit: card free "
        f"{card.free() >> 20} MiB (3600 - 1678 - 1656)")


def test_r3_kv_wait_names_a_floor_breach_it_does_not_wait_for(tmp_path, grab):
    from flliper.srt.managers.scheduler_components.weight_updater import (
        SchedulerWeightUpdaterManager as WU,
    )
    c = PMS.VramCredit("GPU-rv-r3", credit_dir=str(tmp_path))
    c.begin_leg("ep9")                       # this flip's counter, nothing staged
    stub = SimpleNamespace(_pdflip_credit_reader=lambda e: (c, str(e)),
                           _pdflip_free_bytes=lambda: 10432 * MIB)
    got = WU._pdflip_kv_wait_peer_staging(stub, "ep9", 10432 * MIB, 10326 * MIB, 1024 * MIB)
    assert got == 10432 * MIB                # order, not a reserve: no refusal
    hits = [ln for ln in grab.lines if "PDFLIP-WAKE-KV-STAGE-WAIT" in ln]
    assert not any("skipped" in ln for ln in hits), hits   # the wait itself must run
    assert hits, (
        "pool lands 106 MiB free, below the 1024 floor, and nothing says the "
        "wait was skipped because the counter owed nothing")
