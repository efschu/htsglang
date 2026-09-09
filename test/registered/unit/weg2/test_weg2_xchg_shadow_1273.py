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
import threading
import time

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import weight_exchange as wx
from sglang.srt.weg2 import weight_exchange_region as xr
from sglang.srt.weg2 import weight_exchange_shadow as sh
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


# ===========================================================================
# ITEM 3 -- SlotBatch.checksum, wired.  DANGER DIRECTION: a checksum that is
# computed and never compared, compared against a number nobody computed,
# taken over slot padding, taken before the sync -- or one that ABORTS, which
# is the shadow taking authority it does not have.
# ===========================================================================


@pytest.fixture()
def sems(boot):
    xr.create_semaphores(boot)
    s = tp.SemSet(boot)
    yield s
    s.close()
    xr.unlink_semaphores(boot)


def byte_sum(ops: FakeDeviceOps):
    """A ``uint8_checksum``-shaped summer over the double's storage.

    Same VALUE contract as ``model_executor/weights_arena.uint8_checksum``: the
    exact integer sum of the unsigned bytes, so
    ``checksum_is_representable(v, n)`` grades it the same way it grades the
    metal's.  It is a different IMPLEMENTATION on purpose -- torch on a device
    tensor there, ctypes over a file mapping here -- because what is under test
    is the wiring, not torch.
    """

    def summer(addr: int, nbytes: int) -> int:
        return sum(ctypes.string_at(ops.real(int(addr)), int(nbytes)))

    return summer


def _cross_round_trip(region, sems, ops, descs, *, pair, slot_bytes=SLOT,
                      budget_s=10.0, checksum_bytes=None, on_checksum=None,
                      corrupt=None):
    pstats = tp.PairStats(*xr.CROSS_PAIRS[pair], "us", "ud")
    cstats = tp.PairStats(*xr.CROSS_PAIRS[pair], "us", "ud")
    errors: list = []

    def produce():
        try:
            tp.run_producer_pair(region, sems, ops, ops.create_stream(0),
                                 pair=pair, descs=descs, stats=pstats,
                                 budget_s=budget_s, slot_bytes=slot_bytes,
                                 checksum_bytes=checksum_bytes)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    thread = threading.Thread(target=produce)
    thread.start()
    try:
        if corrupt is not None:
            corrupt()
        tp.run_consumer_pair(region, sems, ops, ops.create_stream(0),
                             pair=pair, descs=descs, stats=cstats,
                             budget_s=budget_s, slot_bytes=slot_bytes,
                             checksum_bytes=checksum_bytes,
                             on_checksum=on_checksum)
    finally:
        thread.join(30)
    assert not errors, errors
    return pstats, cstats


def test_the_checksum_is_not_computed_when_no_summer_is_supplied():
    """The authoritative path must stay byte-identical, cost included.

    ``0`` has meant "not computed" since S4 and still does; what changed is
    that a summer makes it a number.  A batch that computed a sum nobody asked
    for would put a full read of every byte inside the flip budget -- the
    reason the field was ``TODO(S5)`` and not simply filled in.
    """
    batch = tp.SlotBatch(0, (tp.Piece(0, tp.FLAT, 16, 0, 0, 0),), 16)
    reads: list = []
    assert batch.checksum() == tp.CHECKSUM_NOT_COMPUTED
    assert batch.checksum(None) == 0
    assert not reads

    def spy(addr, nbytes):
        reads.append((addr, nbytes))
        return 7

    assert batch.checksum(spy, 0x1000) == 7
    assert reads == [(0x1000, 16)]


def test_the_checksum_covers_the_payload_and_not_the_slot_padding():
    """The range is the batch's payload, never the slot.

    A slot is reused; the bytes past ``total_bytes`` are the PREVIOUS batch's.
    Summing the whole slot would make a correct transport report a mismatch on
    every short batch, i.e. an instrument that fires on the healthy case --
    the worst kind, because the first response to it is to disable the check.
    """
    batch = tp.SlotBatch(3, (tp.Piece(0, tp.FLAT, 100, 0, 0, 0),), 100)
    seen: list = []

    def spy(addr, nbytes):
        seen.append(nbytes)
        return 0

    batch.checksum(spy, 0)
    assert seen == [100], "the summer was handed the slot, not the payload"


def test_the_producer_publishes_the_checksum_after_the_sync_not_before(
        region, sems, ops):
    """A checksum of issued-but-not-landed bytes is the previous batch's.

    The fake defers every copy to ``synchronize``, so a checksum taken one
    statement earlier sums the poison this test writes into the slot.  That is
    the same ordering ``bytes_filled`` already has and the same mutant (M4 of
    the first S4 round) one field over -- and S5-pre named it load-bearing here
    precisely because the shadow reads this field WITHOUT the semaphore.
    """
    payload = pattern(11, 300)
    src, dst = dev_ptr(0, 0x10000), dev_ptr(0, 0x20000)
    write(ops, src, payload)
    poison(ops, dst, len(payload), seed=0x5A)
    # Poison the STAGING slot, so "summed before the copy landed" is a
    # different number and not merely a zero.
    base = region.slot_address(0, 0)
    ctypes.memset(base, 0xE7, SLOT)
    descs = [flat_desc(0, 1, len(payload), src_ptr=src, dst_ptr=dst, name="p")]
    _cross_round_trip(region, sems, ops, descs, pair=0,
                      checksum_bytes=byte_sum(ops))
    assert sum(payload) != 0xE7 * len(payload)
    assert read(ops, dst, len(payload)) == payload


def test_a_corrupted_slot_is_reported_to_the_callback_and_never_raised(
        region, sems, ops):
    """THE ZERO-AUTHORITY PROPERTY, on the checksum path.

    The staged bytes are corrupted between the producer's publish and the
    consumer's read.  The consumer must NOTICE (a report with the two numbers)
    and must NOT act: no raise, no refusal, the copy still issued.  A shadow
    that stops a flip on its own finding is not a shadow, and this is the exact
    seam where it would be easiest to make it one.
    """
    payload = pattern(13, 200)
    src, dst = dev_ptr(0, 0x10000), dev_ptr(0, 0x20000)
    write(ops, src, payload)
    poison(ops, dst, len(payload), seed=0x33)
    descs = [flat_desc(0, 1, len(payload), src_ptr=src, dst_ptr=dst, name="p")]
    reports: list = []

    def corrupt():
        # Wait for the slot to carry this flip's PRODUCED record, then flip a
        # byte in the staging slot itself.
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            rec = region.read_slot(0, 0)
            if rec.state == xr.SLOT_PRODUCED and rec.checksum:
                base = region.slot_address(0, 0)
                ctypes.memset(base, 0xFF, 1)
                return
            time.sleep(0.001)
        raise AssertionError("the producer never published a checksum")

    _cross_round_trip(region, sems, ops, descs, pair=0,
                      checksum_bytes=byte_sum(ops), on_checksum=reports.append,
                      corrupt=corrupt)
    assert len(reports) == 1, reports
    report = reports[0]
    assert not report.match
    assert report.lane == "cross" and report.nbytes == len(payload)
    assert report.expected != report.got


def test_a_producer_that_computed_nothing_is_not_compared_against(
        region, sems, ops):
    """An unarmed instrument must not read as a passed one (spec 4.2).

    The consumer has a summer; the producer did not.  Comparing this rank's
    real sum against the other's ``0`` would report a mismatch on every batch
    of an entirely healthy flip.
    """
    payload = pattern(17, 180)
    src, dst = dev_ptr(0, 0x10000), dev_ptr(0, 0x20000)
    write(ops, src, payload)
    descs = [flat_desc(0, 1, len(payload), src_ptr=src, dst_ptr=dst, name="p")]
    reports: list = []
    _cross_round_trip(region, sems, ops, descs, pair=0,
                      checksum_bytes=None, on_checksum=reports.append)
    assert reports == []
    # And with BOTH halves armed the same payload does produce a report.
    region.release_slot(0, 0)
    reports2: list = []
    _cross_round_trip(region, sems, ops, descs, pair=0,
                      checksum_bytes=byte_sum(ops), on_checksum=reports2.append)
    assert len(reports2) == 1 and reports2[0].match


def test_the_oncard_row_carries_the_checksum_and_the_consumer_verifies_it(
        region, tmp_path):
    """The diagonal has no slot RECORD, so the row is the only channel.

    ``xr.CROSS_PAIRS`` has no diagonal by construction, so the on-card lane
    cannot borrow ``XchgSlot.checksum``.  The row grew the field; this proves
    it survives the seal, reaches the consumer, and is compared.
    """
    prod = FakeDeviceOps(str(tmp_path / "d"), rank=0)
    cons = FakeDeviceOps(str(tmp_path / "d"), rank=0)
    try:
        payload = pattern(23, 900)
        src, dst = dev_ptr(0, 0x10000), dev_ptr(0, 0x30000)
        write(prod, src, payload)
        poison(cons, dst, len(payload), seed=0xB1)
        descs = [flat_desc(0, 0, len(payload), src_ptr=src, dst_ptr=dst,
                           name="oncard")]
        bounce = tp.OnCardBounce(prod, 0, slots=tp.ONCARD_SLOTS, slot_bytes=SLOT)
        tp.publish_ipc_handle(region, 0, bounce.handle)
        tp.write_oncard_row(region, tp.DIR_ONCARD_PROD_OFF, 0, seq=-1, nbytes=0,
                            slot_bytes=SLOT, wave=WAVE,
                            state=tp.ONCARD_STATE_ARMED)
        peer = cons.ipc_open_handle(tp.read_ipc_handle(region, 0))
        pstats = tp.OnCardStats(0, "u0", tp.ONCARD_MODE_IPC)
        cstats = tp.OnCardStats(0, "u0", tp.ONCARD_MODE_IPC)
        reports: list = []
        errors: list = []

        def produce():
            try:
                tp.run_oncard_producer(region, prod, prod.create_stream(0),
                                       bounce, row=0, peer_row=3, wave=WAVE,
                                       descs=descs, stats=pstats, budget_s=10.0,
                                       checksum_bytes=byte_sum(prod))
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        thread = threading.Thread(target=produce)
        thread.start()
        tp.run_oncard_consumer(region, cons, cons.create_stream(0), peer,
                               row=3, peer_row=0, wave=WAVE, descs=descs,
                               stats=cstats, slots=tp.ONCARD_SLOTS,
                               slot_bytes=SLOT, budget_s=10.0,
                               checksum_bytes=byte_sum(cons),
                               on_checksum=reports.append)
        thread.join(30)
        assert not errors, errors
        assert read(cons, dst, len(payload)) == payload
        assert reports and all(r.lane == "oncard" for r in reports)
        assert all(r.match for r in reports), reports
        row = tp.read_oncard_row(region, tp.DIR_ONCARD_PROD_OFF, 0, 0)
        assert row["sealed"] and row["checksum"] != 0
        bounce.close()
    finally:
        prod.close()
        cons.close()


def test_the_checksum_seam_is_opt_in_at_every_entry_point():
    """Nothing pays for the shadow's instrument unless it asks for it.

    Four functions grew the pair of arguments; all four default to ``None``, so
    a caller that does not know about them -- which is every caller S6 will
    write for the authoritative path -- gets today's behaviour exactly.
    """
    for fn in (tp.run_producer_pair, tp.run_consumer_pair,
               tp.run_oncard_producer, tp.run_oncard_consumer, tp.run_leg):
        params = inspect.signature(fn).parameters
        if "checksum_bytes" in params:
            assert params["checksum_bytes"].default is None, fn.__name__
        if "on_checksum" in params:
            assert params["on_checksum"].default is None, fn.__name__
    assert "checksum_bytes" in inspect.signature(tp.run_leg).parameters
    assert "on_checksum" in inspect.signature(tp.run_leg).parameters


# ===========================================================================
# ITEM 1 -- THE SHADOW.  DANGER DIRECTION: an observer that acquires
# authority -- refusing a flip, holding a leg for the fence budget, OOMing the
# card it is watching, or reporting a framing error as a data corruption.
# ===========================================================================


def _vote_rows(region, rows, *, leg, vote=True, classes_hash=0, need_mib=0):
    for row in rows:
        sh.write_shadow_vote(region, row, leg=leg, vote=vote,
                             classes_hash=classes_hash, need_mib=need_mib)


def test_the_shadow_area_is_disjoint_from_s4s_and_inside_the_region():
    """Three owners now write the dir area; none may overlap another.

    S3 owns the region, S4 owns ``[DIR_OFF, DATA_OFF)`` and states its own
    sub-layout, and S5 appends its vote rows after S4's last byte.  An overlap
    here is not a crash -- it is two mechanisms silently editing each other's
    single-writer rows, which is the lost-update this region has already
    MEASURED once (``registered=5/6``).
    """
    assert sh.SHADOW_AREA_OFF == tp.DIR_USED_BYTES
    assert sh.SHADOW_AREA_OFF >= tp.DIR_ONCARD_REL_OFF + tp.DIR_ONCARD_REL_BYTES
    assert sh.SHADOW_AREA_END <= tp.DIR_CAPACITY, (
        sh.SHADOW_AREA_END, tp.DIR_CAPACITY)
    assert sh.SHADOW_SEAL_OFF + 8 <= sh.SHADOW_ROW_BYTES
    offs = {sh._row_off(r) for r in range(xr.N_RANKS)}
    assert len(offs) == xr.N_RANKS


def test_a_rank_that_cannot_afford_the_shadow_stops_every_rank_running_it(region):
    """THE #802 RULE, applied to an observer: rank-uniform or not at all.

    A subset of ranks running the exchange is not a smaller experiment -- it is
    a producer with no consumer, blocking on a slot until a budget expires
    inside somebody's authoritative leg.  So one refusal is six.
    """
    _vote_rows(region, [1, 2, 3, 4, 5], leg=7, vote=True)
    verdict = sh.shadow_gate(region, 0, leg=7, vote=False, classes_hash=0,
                             need_mib=900, log=lambda _s: None)
    assert not verdict.run
    assert verdict.refusers == (0,)
    assert "a rank refused" in verdict.reason
    assert verdict.joined == xr.N_RANKS
    assert f"leg=7" in verdict.line(leg=7, epoch="b.7")


def test_ranks_that_chose_different_subsets_do_not_shadow(region):
    """Six ranks shadowing different classes is six half-experiments.

    The subset is derived from the leg index and a sorted class list, so the
    six agree without a message; publishing the hash makes that a CHECK instead
    of an assumption -- and it costs one word in a row that is already there.
    """
    _vote_rows(region, [1, 2, 3, 4, 5], leg=2, vote=True, classes_hash=0xAAAA)
    verdict = sh.shadow_gate(region, 0, leg=2, vote=True, classes_hash=0xBBBB,
                             need_mib=10, log=lambda _s: None)
    assert not verdict.run
    assert "different class subsets" in verdict.reason


def test_the_gate_expiry_switches_the_shadow_off_and_never_raises(region):
    """DEVIATION 6, made falsifiable.

    Five of six ranks vote.  The authoritative wave gate would wait
    ``WEG2_GROUP_FENCE_BUDGET_S`` and then raise W53 -- correct there, fatal
    here: it would put 120 s of an observer's wait inside a flip leg.  The
    shadow waits its own small budget and switches itself off.
    """
    _vote_rows(region, [1, 2, 3, 4], leg=1, vote=True)
    started = time.monotonic()
    verdict = sh.shadow_gate(region, 0, leg=1, vote=True, classes_hash=0,
                             need_mib=0, log=lambda _s: None, budget_s=0.05)
    assert not verdict.run
    assert time.monotonic() - started < 5.0
    assert 5 in verdict.refusers
    assert "switches itself off" in verdict.reason
    assert verdict.joined == 5


def test_the_subset_rotates_and_every_rank_derives_the_same_one():
    """A bounded subset per leg, rotating, identical on all six ranks.

    Derived from the LEG INDEX and a sorted class list -- never from anything a
    rank measures -- so two ranks holding different directed slices of the plan
    still choose the same classes.
    """
    descs = _class_descs(["qkv_proj", "o_proj", "in_proj_qkvz"])
    seen = []
    for leg in range(6):
        subset = sh.select_subset(descs, leg=leg)
        assert len(subset.classes) == 1
        seen.append(subset.classes[0])
    assert set(seen) == {"qkv_proj", "o_proj", "in_proj_qkvz"}
    assert seen[:3] != seen[1:4] or len(set(seen[:3])) == 3
    # The same leg, from a DIFFERENT directed slice of the same plan, picks
    # the same class.
    half = [d for d in descs if d.dst_rank == 0]
    assert sh.select_subset(half, leg=4).classes == \
        sh.select_subset(descs, leg=4).classes


def test_the_classes_hash_is_stable_across_processes():
    """``hash()`` is salted per process; six ranks would publish six numbers.

    The same defect class as a plan id built from pointers, and it would make
    the gate refuse every leg for a reason that reads like a real disagreement.
    """
    assert sh.classes_hash(["b", "a"]) == sh.classes_hash(["a", "b"])
    assert sh.classes_hash(["a", "b"]) == xr.epoch_hash("a|b")


def test_the_5090_free_column_refuses_the_full_leg_by_name():
    """The rig's own reading, priced.

    sb5f measured the 5090 at **474 MiB** free under load -- already below the
    1024 MiB corridor floor before the shadow asks for anything.  The refusal
    names the card, both halves of the need, the free column and the floor, so
    a reader can see WHICH term made it impossible instead of being told it was.
    """
    price = sh.price_shadow("GPU-5090", 8 * (1 << 30), 474)
    assert not price.affordable
    msg = price.message()
    assert sh.UNAFFORDABLE_MARKER in msg
    for token in ("card=GPU-5090", "need_mib=", "free_mib=474", "floor_mib=1024",
                  "scratch="):
        assert token in msg, (token, msg)
    assert "REFUSED" in price.line()


def test_a_bounded_subset_fits_where_the_full_leg_does_not():
    """Which is the whole reason the subset is bounded and rotating."""
    free = 4096
    assert not sh.price_shadow("GPU-5090", 8 * (1 << 30), free).affordable
    small = sh.price_shadow("GPU-5090", 512 * (1 << 20), free)
    assert small.affordable
    # The scratch is charged, always: it is 64 MiB of real VRAM and the spec
    # budgets it by name.
    assert small.need_mib == 512 + sh.STRIPE_BYTES // sh.MIB


def test_shadow_mismatch_names_the_parameter():
    """Spec 6/S5's first red-first test, verbatim in intent.

    "checksum mismatch" is not a finding a reader can act on.  The line names
    the CLASS, the parameter, the stripe, the destination offset, the byte
    count and BOTH checksums.
    """
    stripe = sh.Stripe(index=3, tensor_class="in_proj_qkvz",
                       param_name="model.layers.7.linear_attn.in_proj_qkvz.weight",
                       nbytes=1024, shadow_sum=500, ring_sum=501,
                       first_run_dst=0xDEAD000)
    assert sh.classify(stripe) == sh.MISMATCH
    msg = sh.mismatch_message(stripe, verdict=sh.MISMATCH, leg=2, epoch="b.2")
    for token in ("W59 Weg2XchgShadowMismatch", "class=in_proj_qkvz",
                  "param=model.layers.7.linear_attn.in_proj_qkvz.weight",
                  "stripe=3", "dst_off=0xdead000", "nbytes=1024",
                  "shadow_checksum=500", "ring_checksum=501"):
        assert token in msg, (token, msg)
    assert "RING's bytes are authoritative" in msg


def test_shadow_asks_representability_before_reporting():
    """Spec 6/S5's second red-first test, and #656 register C22's lesson.

    A value outside ``[0, 255 * nbytes]`` was never a checksum of this payload:
    the two ends framed it differently.  Reporting that as a data corruption is
    what killed an instance for a corruption that had not happened.
    """
    unwritten = sh.Stripe(index=0, tensor_class="o_proj", param_name="p",
                          nbytes=16, shadow_sum=4626949667419791296, ring_sum=0)
    assert sh.classify(unwritten) == sh.NOT_REPRESENTABLE
    msg = sh.mismatch_message(unwritten, verdict=sh.NOT_REPRESENTABLE, leg=1,
                              epoch="b.1")
    assert "never a checksum of this payload" in msg
    assert "the DATA is not what is wrong" in msg
    negative = sh.Stripe(index=0, tensor_class="o_proj", param_name="p",
                         nbytes=16, shadow_sum=-4450328002521349435, ring_sum=0)
    assert sh.classify(negative) == sh.NOT_REPRESENTABLE


def test_the_comparison_walks_payload_runs_and_never_2d_padding(ops):
    """A 2-D destination's padding belongs to OTHER tensors.

    Between two runs of a row-parallel class sit bytes this rank's own
    exchange never wrote.  A span compare would report them as mismatches on a
    perfectly correct flip -- an instrument that fires on the healthy case,
    which is the worst kind because the first response to it is to switch it
    off.
    """
    rows, run, dpitch = 4, 64, 256
    ring = dev_ptr(0, 0x10000)
    shadow = dev_ptr(0, 0x40000)
    payload = pattern(7, rows * run)
    # Both sides get the SAME payload in their runs and DIFFERENT padding.
    for r in range(rows):
        write(ops, ring + r * dpitch, payload[r * run:(r + 1) * run])
        write(ops, shadow + r * dpitch, payload[r * run:(r + 1) * run])
        write(ops, ring + r * dpitch + run, b"\xAA" * (dpitch - run))
        write(ops, shadow + r * dpitch + run, b"\x55" * (dpitch - run))
    desc = _strided(rows=rows, run_bytes=run, dpitch=dpitch, dst_ptr=ring)
    layout = {id(desc): 0}
    stripes = sh.compare_stripes([desc], layout, shadow, byte_sum(ops))
    assert len(stripes) == 1
    assert stripes[0].nbytes == rows * run
    assert sh.classify(stripes[0]) == sh.MATCH, "the padding was compared"


def test_a_stripe_is_the_sum_of_its_runs_and_64_mib_wide(ops):
    """Stripes are built out of runs because the sum is EXACT and additive.

    ``weights_arena.uint8_checksum`` says so where it explains why chunking
    cannot change the value; that is what lets a 6 KiB run of a 2-D class
    contribute to a 64 MiB stripe instead of becoming a stripe of its own.
    """
    desc = _flat(nbytes=3000, dst_ptr=dev_ptr(0, 0x10000))
    write(ops, desc.dst_ptr, pattern(3, 3000))
    write(ops, dev_ptr(0, 0x40000), pattern(3, 3000))
    stripes = sh.compare_stripes([desc], {id(desc): 0}, dev_ptr(0, 0x40000),
                                 byte_sum(ops), stripe_bytes=1000)
    assert [s.nbytes for s in stripes] == [1000, 1000, 1000]
    assert all(s.match for s in stripes)
    assert sum(s.shadow_sum for s in stripes) == sum(pattern(3, 3000))


def test_the_shadow_moves_the_bytes_into_its_own_buffer_and_not_the_ring_s(
        region, tmp_path, boot):
    """END TO END on the fake: the diagonal, into a shadow buffer.

    The ring's destination is poisoned before and asserted UNCHANGED after --
    the shadow may read the source's still-mapped VRAM and may write only its
    own raw buffer.  A shadow that touched the destination would be the
    exchange with the ring's authority and none of its proof.
    """
    xr.create_semaphores(boot)
    sems = tp.SemSet(boot)
    src_ops = FakeDeviceOps(str(tmp_path / "d"), rank=0)
    dst_ops = FakeDeviceOps(str(tmp_path / "d"), rank=0)
    try:
        payload = pattern(31, 2000)
        src, ring_dst = dev_ptr(0, 0x10000), dev_ptr(0, 0x60000)
        write(src_ops, src, payload)
        poison(dst_ops, ring_dst, len(payload), seed=0x77)
        before = read(dst_ops, ring_dst, len(payload))
        descs = [flat_desc(0, 0, len(payload), src_ptr=src, dst_ptr=ring_dst,
                           name="model.layers.0.self_attn.qkv_proj.weight")]
        _vote_rows(region, [1, 2, 4, 5], leg=0, vote=True,
                   classes_hash=sh.classes_hash(["qkv_proj"]), need_mib=0)
        out: list = []

        def source():
            out.append(sh.shadow_transport(
                region=region, sems=sems, ops=src_ops, row=0, rank=0, device=0,
                card_uuid="u0", uuid_of_card=["u0", "u1", "u2"], descs=descs,
                is_source=True, oncard_mode=tp.ONCARD_MODE_IPC, peer_row=3,
                wave=WAVE, leg=0, direction="P->D", epoch=f"{boot}.1",
                free_mib=8192, log=lambda _s: None, budget_s=10.0))

        thread = threading.Thread(target=source)
        thread.start()
        run = sh.shadow_transport(
            region=region, sems=sems, ops=dst_ops, row=3, rank=0, device=0,
            card_uuid="u0", uuid_of_card=["u0", "u1", "u2"], descs=descs,
            is_source=False, oncard_mode=tp.ONCARD_MODE_IPC, peer_row=0,
            wave=WAVE, leg=0, direction="P->D", epoch=f"{boot}.1",
            free_mib=8192, log=lambda _s: None, budget_s=10.0,
            sum_bytes=byte_sum(dst_ops))
        thread.join(60)
        assert run.result.ran, run.result.counters.errors
        assert run.buffers is not None
        assert read(dst_ops, run.buffers.ptr, len(payload)) == payload
        assert read(dst_ops, ring_dst, len(payload)) == before, \
            "the shadow wrote into the RING's destination"
        # Now the ring 'restores' the same bytes and the compare must MATCH.
        write(dst_ops, ring_dst, payload)
        result = run.compare(byte_sum(dst_ops), lambda _s: None)
        assert result.counters.stripes == 1
        assert result.counters.match == 1 and result.counters.mismatch == 0
        line = result.line()
        for token in ("WEG2-XCHG-SHADOW ", "leg=0", "classes=1", "stripes=1",
                      "match=1", "mismatch=0", "oncard_ms=", "cross_ms=",
                      "ring_ms=", "verdict=MATCH", "subset=qkv_proj",
                      "dir=P->D", "pieces=", "xchg_ms=", "issue_ms=",
                      "slot_wait_ms=", "gate_skew_ms=", "lock_wait_ms="):
            assert token in line, (token, line)
    finally:
        src_ops.close()
        dst_ops.close()
        sems.close()
        xr.unlink_semaphores(boot)


def test_a_mismatch_is_counted_and_the_flip_is_never_told(region, tmp_path, boot):
    """The can-fail control for the acceptance line, and the authority rule.

    The 'ring' restores DIFFERENT bytes than the shadow pulled.  The line must
    go red (``verdict=MISMATCH``, ``mismatch=1``) and nothing may be raised:
    the ring's bytes were served, and this is a finding about the EXCHANGE.
    """
    xr.create_semaphores(boot)
    sems = tp.SemSet(boot)
    src_ops = FakeDeviceOps(str(tmp_path / "d"), rank=0)
    dst_ops = FakeDeviceOps(str(tmp_path / "d"), rank=0)
    try:
        payload = pattern(37, 1500)
        src, ring_dst = dev_ptr(0, 0x10000), dev_ptr(0, 0x60000)
        write(src_ops, src, payload)
        descs = [flat_desc(0, 0, len(payload), src_ptr=src, dst_ptr=ring_dst,
                           name="model.layers.0.mlp.down_proj.weight")]
        _vote_rows(region, [1, 2, 4, 5], leg=0, vote=True,
                   classes_hash=sh.classes_hash(["down_proj"]), need_mib=0)
        thread = threading.Thread(target=lambda: sh.shadow_transport(
            region=region, sems=sems, ops=src_ops, row=0, rank=0, device=0,
            card_uuid="u0", uuid_of_card=["u0", "u1", "u2"], descs=descs,
            is_source=True, oncard_mode=tp.ONCARD_MODE_IPC, peer_row=3,
            wave=WAVE, leg=0, direction="P->D", epoch=f"{boot}.1",
            free_mib=8192, log=lambda _s: None, budget_s=10.0))
        thread.start()
        run = sh.shadow_transport(
            region=region, sems=sems, ops=dst_ops, row=3, rank=0, device=0,
            card_uuid="u0", uuid_of_card=["u0", "u1", "u2"], descs=descs,
            is_source=False, oncard_mode=tp.ONCARD_MODE_IPC, peer_row=0,
            wave=WAVE, leg=0, direction="P->D", epoch=f"{boot}.1",
            free_mib=8192, log=lambda _s: None, budget_s=10.0,
            sum_bytes=byte_sum(dst_ops))
        thread.join(60)
        assert run.result.ran
        write(dst_ops, ring_dst, pattern(38, 1500))  # a DIFFERENT restore
        lines: list = []
        result = run.compare(byte_sum(dst_ops), lines.append)
        assert result.counters.mismatch == 1
        assert "verdict=MISMATCH" in result.line()
        assert any(sh.MISMATCH_MARKER in ln for ln in lines)
        assert any("class=down_proj" in ln for ln in lines)
    finally:
        src_ops.close()
        dst_ops.close()
        sems.close()
        xr.unlink_semaphores(boot)


def test_a_transport_failure_is_logged_and_never_raised(region, tmp_path, boot):
    """Zero authority under FAILURE, which is the case that matters.

    W52/W53/W54 stop a flip on the authoritative path.  Here the same events
    mean only "the shadow got no measurement": they are recorded, the line says
    ``ran=no`` with the reason, and the leg continues.
    """
    xr.create_semaphores(boot)
    sems = tp.SemSet(boot)
    dst_ops = FakeDeviceOps(str(tmp_path / "d"), rank=0, fail_ipc_open=True)
    try:
        descs = [flat_desc(0, 0, 512, src_ptr=dev_ptr(0, 0x10000),
                           dst_ptr=dev_ptr(0, 0x60000),
                           name="model.layers.0.self_attn.o_proj.weight")]
        _vote_rows(region, [1, 2, 4, 5], leg=0, vote=True,
                   classes_hash=sh.classes_hash(["o_proj"]), need_mib=0)
        lines: list = []
        run = sh.shadow_transport(
            region=region, sems=sems, ops=dst_ops, row=3, rank=0, device=0,
            card_uuid="u0", uuid_of_card=["u0", "u1", "u2"], descs=descs,
            is_source=False, oncard_mode=tp.ONCARD_MODE_IPC, peer_row=0,
            wave=WAVE, leg=0, direction="P->D", epoch=f"{boot}.1",
            free_mib=8192, log=lines.append, budget_s=0.4,
            sum_bytes=byte_sum(dst_ops))
        assert not run.result.ran
        assert run.result.counters.errors, "the failure was not even recorded"
        assert "ran=no" in run.result.line()
        assert "verdict=NOT-RUN" in run.result.line()
        # And the compare afterwards is a no-op that still refuses to say MATCH.
        result = run.compare(byte_sum(dst_ops), lines.append)
        assert result.counters.match == 0
    finally:
        dst_ops.close()
        sems.close()
        xr.unlink_semaphores(boot)


def test_the_shadow_hands_the_transport_a_recorder_not_the_group_vote():
    """``vote_failure`` is how one rank takes six down.  The shadow may not.

    An AST check, because the property is about what is PASSED and a runtime
    assertion would only prove it for the paths a test happens to drive.
    """
    import ast

    src = inspect.getsource(sh.shadow_transport)
    tree = ast.parse(src.lstrip())
    votes = [kw for node in ast.walk(tree)
             if isinstance(node, ast.Call)
             for kw in node.keywords if kw.arg == "vote_failure"]
    assert votes, "the transport call lost its vote_failure argument"
    for kw in votes:
        assert isinstance(kw.value, ast.Lambda), ast.dump(kw.value)
        body = ast.dump(kw.value.body)
        assert "counters" in body and "errors" in body, body


def test_the_shadow_mode_does_not_arm_the_exchange():
    """``shadow`` is RING-AUTHORITATIVE, and one predicate says so.

    ``exchange_armed()`` gates ``enable_cpu_backup`` and therefore whether
    ``pause`` is a pure unmap.  True under ``shadow`` would delete the ring
    restore the shadow compares against -- the ground truth removed by the
    instrument that needs it.
    """
    with wx.weight_source_for_test(sh.WEIGHT_SOURCE_SHADOW):
        assert wx.weight_source() == "shadow"
        assert wx.shadow_armed()
        assert not wx.exchange_armed()
    with wx.weight_source_for_test(wx.WEIGHT_SOURCE_EXCHANGE):
        assert wx.exchange_armed() and not wx.shadow_armed()
    with wx.weight_source_for_test("nonsense"):
        assert wx.weight_source() == wx.WEIGHT_SOURCE_RING, \
            "an unrecognised env value armed something"


def test_the_default_boot_publishes_no_region_env_and_pops_an_inherited_one():
    """The default arm stays byte-identical, INCLUDING what it inherits.

    Launcher OUTPUT is popped when this boot arms nothing -- the same rule the
    ring family and the duplex table already follow, and for the measured
    reason: a stale region path in the operator's shell is a rank mapping
    another boot's shared memory.
    """
    from sglang.srt.weg2 import launcher

    keys = ("SGLANG_WEG2_XCHG_REGION", "SGLANG_WEG2_XCHG_BOOT",
            "SGLANG_WEG2_WEIGHT_SOURCE")
    saved = {k: os.environ.get(k) for k in keys}
    try:
        for k in keys:
            os.environ[k] = "inherited"
        env = launcher.build_env(tree="/t", venv="/v", cvd="GPU-a",
                                 store_dir="/s", debug_hold=False, tag="t")
        for k in keys:
            assert k not in env, k
        armed = launcher.build_env(
            tree="/t", venv="/v", cvd="GPU-a", store_dir="/s",
            debug_hold=False, tag="t",
            xchg_env={"SGLANG_WEG2_XCHG_REGION": "/dev/shm/weg2-xchg-b/xchg.bin",
                      "SGLANG_WEG2_XCHG_BOOT": "b",
                      "SGLANG_WEG2_WEIGHT_SOURCE": "shadow"})
        assert armed["SGLANG_WEG2_WEIGHT_SOURCE"] == "shadow"
        assert armed["SGLANG_WEG2_XCHG_BOOT"] == "b"
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_the_shadow_arm_is_a_launcher_choice_and_arms_nothing_on_the_others():
    """``--weg2-weight-source shadow`` exists, and the other arms are untouched."""
    from sglang.srt.weg2 import launcher

    assert launcher.WEIGHT_SOURCE_CHOICES == ("ring", "exchange", "shadow")
    assert launcher.WEIGHT_SOURCE_DEFAULT == "ring"
    for arm in ("ring", "exchange"):
        assert launcher.prepare_shadow_env(lambda *_a: None, "b", arm) == {}


def _flat(*, nbytes, dst_ptr, name="model.layers.0.self_attn.qkv_proj.weight"):
    return wx.XchgDesc(tag="weights_0", src_rank=0, dst_rank=0,
                       param_name=name, kind=wx.FLAT, nbytes=nbytes, rows=1,
                       run_bytes=nbytes, spitch=0, dpitch=0, src_off=0,
                       dst_off=0, src_ptr=0x1000, dst_ptr=dst_ptr)


def _strided(*, rows, run_bytes, dpitch, dst_ptr,
             name="model.layers.0.self_attn.o_proj.weight"):
    return wx.XchgDesc(tag="weights_0", src_rank=0, dst_rank=0,
                       param_name=name, kind=wx.STRIDED2D,
                       nbytes=rows * run_bytes, rows=rows, run_bytes=run_bytes,
                       spitch=run_bytes, dpitch=dpitch, src_off=0, dst_off=0,
                       src_ptr=0x1000, dst_ptr=dst_ptr)


def _class_descs(classes):
    out = []
    for i, cls in enumerate(classes):
        for rank in (0, 1):
            out.append(wx.XchgDesc(
                tag="weights_0", src_rank=rank, dst_rank=rank,
                param_name=f"model.layers.{i}.attn.{cls}.weight",
                kind=wx.FLAT, nbytes=4096, rows=1, run_bytes=4096, spitch=0,
                dpitch=0, src_off=0, dst_off=0, src_ptr=0x1000,
                dst_ptr=0x2000))
    return out

