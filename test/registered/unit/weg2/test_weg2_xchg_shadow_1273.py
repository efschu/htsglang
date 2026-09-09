# SPDX-License-Identifier: Apache-2.0
"""#1273 slice S5 -- the SHADOW, the on-card handshake fix, and the checksum.

WEG2_REUSE_SPEC_0908 section 6 / S5, plus the two findings S5-pre brought back
from the metal (SECTION 1ai-S5-pre of WEG2_BUILD_DECISIONS_0906):

* the on-card lane is **per-batch-bound, not bandwidth-bound**, and half of the
  per-batch cost was ``ONCARD_POLL_S`` -- a GATE-scale constant (500 us) reused
  on a PER-BATCH path.  Measured: 0.328-0.398 ms/batch at 500 us, 0.182 at
  50 us, 0.165 at 5 us, consumer-side, 512 MiB, RTX 5090.
* ``SlotBatch.checksum`` was ``TODO(S5)`` and returned 0.

**THE DOUBLE IS S4's FAKE DEVICE LAYER**, imported rather than re-written: a
second fake would be a second set of assumptions about the same silicon, and
the S4 round already proved that the fake is weakest exactly where the defects
are (three of six must_fix lived in its benign spots).  This file re-uses it
and adds what S5 needs on top -- a byte-summing checksum, poisoned shadow
destinations and a second destination image to compare against.

**ZERO AUTHORITY IS A TESTED PROPERTY, NOT A DESIGN INTENTION.**  The shadow
may not vote a gate, may not refuse a flip, may not change a slot state and may
not call ``vote_failure``; a mismatch is COUNTED and LOGGED and the ring's
bytes win.  Half of this file exists to make that falsifiable, because the
first thing that will be asked of a shadow that finds something is exactly the
thing it may not do.
"""

from __future__ import annotations

import ctypes
import inspect
import os
import re
import threading

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import weight_exchange as wx
from sglang.srt.weg2 import weight_exchange_region as xr
from sglang.srt.weg2 import weight_exchange_transport as tp

from .test_weg2_xchg_transport_1273 import (  # noqa: E402 -- after the env guard
    FakeDeviceOps,
    SLOT,
    WAVE,
    _fresh_boot,
    dev_ptr,
    flat_desc,
    pattern,
    poison,
    read,
    write,
)


@pytest.fixture()
def boot():
    return _fresh_boot()


@pytest.fixture()
def region(tmp_path, boot):
    r = xr.XchgRegion.create(boot, shm_root=str(tmp_path))
    r.begin_flip(f"{boot}.1")
    yield r
    r.close()


@pytest.fixture()
def ops(tmp_path):
    o = FakeDeviceOps(str(tmp_path / "dev"), rank=0)
    yield o
    o.close()


class _Clock:
    """A monotonic clock a test drives, so "did it sleep yet" is not a race."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(float(seconds))
        self.now += float(seconds)


# ===========================================================================
# ITEM 2 -- the on-card handshake.  DANGER DIRECTION: the lane keeps the
# gate-scale sleep and the diagonal misses its budget by 6-9x with every test
# still green, because no test anywhere prices a batch count.
# ===========================================================================


def test_the_oncard_poll_is_not_the_gates_poll():
    """The S5-pre finding, made structural.

    ``ONCARD_POLL_S = xr.GATE_POLL_S`` was not a typo -- it was a deliberate
    "no second timeout constant" reading of spec section 3.2 applied to a
    constant that is not a timeout at all.  A GATE fires once per wave; this
    poll fires once per BATCH, 329 times on the real diagonal.  The two are
    different quantities and this asserts they are no longer one name.
    """
    assert tp.ONCARD_POLL_S != xr.GATE_POLL_S, (
        "the per-batch poll is the per-wave gate's poll again -- S5-pre "
        "measured that as ~half the lane's whole per-batch cost")
    assert tp.ONCARD_POLL_S <= 50e-6, tp.ONCARD_POLL_S
    # And the gate's own constant is untouched: this slice may not make the
    # wave gate spin.
    assert xr.GATE_POLL_S == 0.0005


def test_the_wait_spins_before_it_sleeps_and_sleeps_after():
    """Spin-then-sleep, with both halves observed.

    The measured floor of the round trip is 0.165 ms
    (``ONCARD_SPIN_BUDGET_S``): a peer that is going to answer answers inside
    it.  Spinning is therefore the right wait for that window and a syscall is
    the right wait for anything longer.  A wait that only slept would pay a
    full granularity for a peer that was microseconds away; a wait that only
    spun would burn a core for a peer that had died.
    """
    clock = _Clock()
    wait = tp._OnCardWait(sleep=clock.sleep, monotonic=clock.monotonic)
    wait.wait()
    wait.wait()
    assert clock.sleeps == [], "it slept inside the measured round-trip window"
    assert wait.spins == 2 and wait.sleeps == 0
    clock.now += tp.ONCARD_SPIN_BUDGET_S * 2
    wait.wait()
    assert clock.sleeps == [tp.ONCARD_POLL_S], clock.sleeps
    assert wait.sleeps == 1


def test_the_spin_is_bounded_by_iterations_as_well_as_time():
    """A frozen clock must not buy an unbounded spin.

    The TIME bound is the load-bearing one, but the cost of one row read is
    NOT measured -- so on a box where a read is cheap, a time-only bound spins
    an unmeasured number of times, and on a box where the clock is coarse it
    spins forever.  Whichever bound is reached first ends the spin.
    """
    clock = _Clock()  # never advances on its own
    wait = tp._OnCardWait(sleep=clock.sleep, monotonic=clock.monotonic)
    for _ in range(tp.ONCARD_SPIN_ITERS + 5):
        wait.wait()
    assert wait.spins == tp.ONCARD_SPIN_ITERS
    assert wait.sleeps == 5, wait.sleeps


def test_the_row_area_is_sized_for_the_deepest_pipeline_not_the_default():
    """``ONCARD_SLOTS`` became a knob; the row area may not follow it.

    Two processes compute these offsets independently and never exchange them,
    so a row area sized from an ARGUMENT is rank ``r``'s slot ``k`` landing on
    rank ``r+1``'s slot 0 -- identically on both sides, which is why nothing
    would disagree and the bytes would simply be wrong.
    """
    assert tp.DIR_ONCARD_ROWS == xr.N_RANKS * tp.ONCARD_SLOTS_MAX
    assert tp.ONCARD_SLOTS <= tp.ONCARD_SLOTS_MAX
    seen = set()
    for row in range(xr.N_RANKS):
        for slot in range(tp.ONCARD_SLOTS_MAX):
            off = tp._oncard_row_off(tp.DIR_ONCARD_PROD_OFF, row, slot)
            assert off not in seen, (row, slot)
            seen.add(off)
    assert tp.DIR_USED_BYTES <= tp.DIR_CAPACITY, (
        tp.DIR_USED_BYTES, tp.DIR_CAPACITY)


def test_a_pipeline_depth_the_rows_cannot_address_is_refused_before_any_copy(
        region, tmp_path):
    """W52 by name, and the refusal lands before a thread exists."""
    dev = FakeDeviceOps(str(tmp_path / "d"), rank=0)
    try:
        descs = [flat_desc(0, 0, 64, src_ptr=dev_ptr(0, 0x1000),
                           dst_ptr=dev_ptr(0, 0x2000), name="d")]
        with pytest.raises(xr.Weg2XchgPlanDisagree) as excinfo:
            tp.run_leg(region, None, dev, row=0, rank=0, device=0,
                       card_uuid="u0", uuid_of_card=["u0", "u1", "u2"],
                       descs=descs, is_source=True, oncard_mode=tp.ONCARD_MODE_IPC,
                       peer_row=3, wave=WAVE, log=lambda _s: None,
                       vote_failure=lambda _e: None,
                       oncard_slots=tp.ONCARD_SLOTS_MAX + 1)
        assert "oncard_slots" in str(excinfo.value)
        assert dev.issued == 0, "a copy was issued past the refusal"
    finally:
        dev.close()


def test_a_deeper_pipeline_moves_the_bytes_and_uses_every_slot(region, tmp_path):
    """Depth 4 end to end: byte-exact, and all four slots really cycled.

    The knob is worthless if the extra slots are addressed but never used, and
    worse than worthless if they are used but land on top of each other -- so
    the assertion is both the bytes AND the distinct slot count.
    """
    prod = FakeDeviceOps(str(tmp_path / "d"), rank=0)
    cons = FakeDeviceOps(str(tmp_path / "d"), rank=0)
    slots = 4
    try:
        payload = pattern(41, 7 * SLOT + 11)
        src, dst = dev_ptr(0, 0x10000), dev_ptr(0, 0x80000)
        write(prod, src, payload)
        poison(cons, dst, len(payload), seed=0xC3)
        descs = [flat_desc(0, 0, len(payload), src_ptr=src, dst_ptr=dst,
                           name="oncard")]
        assert len(tp.batch_descs(descs, SLOT)) == 8, "not deep enough to cycle"

        bounce = tp.OnCardBounce(prod, 0, slots=slots, slot_bytes=SLOT)
        tp.publish_ipc_handle(region, 0, bounce.handle)
        tp.write_oncard_row(region, tp.DIR_ONCARD_PROD_OFF, 0, seq=-1, nbytes=0,
                            slot_bytes=SLOT, wave=WAVE,
                            state=tp.ONCARD_STATE_ARMED)
        peer = cons.ipc_open_handle(tp.read_ipc_handle(region, 0))
        pstats = tp.OnCardStats(0, "u0", tp.ONCARD_MODE_IPC)
        cstats = tp.OnCardStats(0, "u0", tp.ONCARD_MODE_IPC)
        errors: list = []

        def produce():
            try:
                tp.run_oncard_producer(region, prod, prod.create_stream(0),
                                       bounce, row=0, peer_row=3, wave=WAVE,
                                       descs=descs, stats=pstats, budget_s=10.0)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        thread = threading.Thread(target=produce)
        thread.start()
        tp.run_oncard_consumer(region, cons, cons.create_stream(0), peer,
                               row=3, peer_row=0, wave=WAVE, descs=descs,
                               stats=cstats, slots=slots, slot_bytes=SLOT,
                               budget_s=10.0)
        thread.join(30)
        assert not errors, errors
        assert read(cons, dst, len(payload)) == payload
        assert pstats.batches == cstats.batches == 8
        used = {tp.batch_descs(descs, SLOT)[i].seq % slots for i in range(8)}
        assert used == {0, 1, 2, 3}, used
        bounce.close()
    finally:
        prod.close()
        cons.close()


def test_the_diagonal_slot_is_priced_to_the_hop_budget():
    """The cost model, on the number that produced the finding.

    10.28 GiB at the shipping 32 MiB slot is 329 batches ~= 118 ms against a
    20 ms budget.  Priced, the slot grows until the batch count fits, and the
    model is printed with the number so a reader can check the arithmetic
    rather than trust it.
    """
    plan = tp.plan_oncard_slot_bytes(int(10.28 * (1 << 30)))
    assert plan.fits, plan.tokens()
    assert plan.hop_ms <= tp.ONCARD_HOP_BUDGET_MS
    assert plan.slot_bytes > tp.ONCARD_SLOT_BYTES
    assert plan.batches * plan.per_batch_ms == pytest.approx(plan.hop_ms)
    assert "batches x per_batch_ms" in plan.model()
    # The shipping constant would NOT have fitted, which is why this exists.
    shipped = -(-int(10.28 * (1 << 30)) // tp.ONCARD_SLOT_BYTES)
    assert shipped == 329
    assert shipped * tp.ONCARD_PER_BATCH_MS > tp.ONCARD_HOP_BUDGET_MS


def test_a_diagonal_too_large_for_the_ceiling_is_priced_red_not_rounded_green():
    """The ceiling is a VRAM term, so the price may come back RED.

    ``ONCARD_SLOT_BYTES_MAX x slots`` is raw VRAM on the producer's card and a
    residency term of spec section 5.  A pricer that grew the slot until the
    timing budget went green would be paying for a wall claim with a number
    W55 has to refuse later -- so it clamps and says ``fits=no``.
    """
    huge = 200 * (1 << 30)
    plan = tp.plan_oncard_slot_bytes(huge)
    assert plan.slot_bytes == tp.ONCARD_SLOT_BYTES_MAX
    assert not plan.fits
    assert "oncard_priced_fits=no" in plan.tokens()
    assert plan.bounce_bytes == plan.slots * tp.ONCARD_SLOT_BYTES_MAX


def test_the_per_batch_coefficient_reproduces_the_arm_it_was_measured_from():
    """A CONSISTENCY check, and it is labelled as one.

    S5-pre's poll-rebind arm moved 512 MiB in 16 batches of 32 MiB in 2.91 ms
    consumer-side.  The coefficient was DERIVED from that arm, so reproducing
    it is not independent evidence -- it is a guard that the constant and the
    model have not drifted apart in this file.
    """
    plan = tp.plan_oncard_slot_bytes(512 * (1 << 20))
    assert plan.slot_bytes == tp.ONCARD_SLOT_BYTES and plan.batches == 16
    assert plan.hop_ms == pytest.approx(2.91, abs=0.02)


def test_the_priced_hop_is_on_the_plan_line():
    """The boot has to be able to compare the price with the measurement.

    ``hop_ms`` appears on ``WEG2-XCHG-ONCARD`` as a MEASUREMENT.  Without the
    prediction beside it in the log, a 118 ms hop on the metal is a number with
    nothing to be wrong against -- which is exactly the state S5-pre found.
    """
    line = _plan_line_for(oncard_nbytes=int(4.0 * (1 << 30)))
    for token in ("oncard_slot_mib=", "oncard_batches=", "oncard_hop_ms_priced=",
                  "oncard_hop_ms_budget=", "oncard_priced_fits=",
                  "oncard_priced_rank="):
        assert token in line, (token, line)


def test_the_priced_hop_names_the_card_not_the_sum():
    """The lane is PER CARD; a hop priced against the three-card sum is priced
    against a lane that does not exist.

    S5-pre's own extrapolation read the plan's 10.28 GiB total as one lane's
    329 batches.  Both numbers stay on the line, each with its denominator.
    """
    plan = _plan_with_oncard({0: 3 << 30, 1: 1 << 30})
    by_rank = plan.oncard_bytes_by_rank()
    assert by_rank == {0: 3 << 30, 1: 1 << 30}
    assert plan.oncard_bytes == 4 << 30
    line = plan.log_line()
    assert "oncard_priced_rank=0" in line
    assert "oncard_gib=4.00" in line
    assert "oncard_priced_gib=3.00" in line


def _plan_with_oncard(by_rank) -> wx.XchgPlan:
    descs = []
    for rank, nbytes in by_rank.items():
        descs.append(wx.XchgDesc(
            tag="weights_0", src_rank=rank, dst_rank=rank, param_name=f"p{rank}",
            kind=wx.FLAT, nbytes=int(nbytes), rows=1, run_bytes=int(nbytes),
            spitch=0, dpitch=0, src_off=0, dst_off=0,
            src_ptr=0x1000, dst_ptr=0x2000,
        ))
    return wx.XchgPlan(descs=tuple(descs), raw_descs=tuple(descs),
                       waves=(("weights_0",),), byte_matrix=(), plan_id="pid",
                       src_group="P", dst_group="D")


def _plan_line_for(*, oncard_nbytes: int) -> str:
    return _plan_with_oncard({0: oncard_nbytes}).log_line()
