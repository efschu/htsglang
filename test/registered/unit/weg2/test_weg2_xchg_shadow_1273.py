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
are (three of six must_fix lived in its benign spots).  This file reuses it
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


_UNSET = object()


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
                      corrupt=None, producer_checksum=_UNSET):
    """``producer_checksum`` is separate on purpose.

    SURVIVING MUTANT (this slice's own mutant E): deleting the
    "a producer that published nothing is not compared against" guard left the
    suite green, because the one test that named that case handed NEITHER side
    a summer -- so ``_report_checksum`` returned at its FIRST guard and the one
    under test was never reached.  The asymmetric arm is the whole case.
    """
    pstats = tp.PairStats(*xr.CROSS_PAIRS[pair], "us", "ud")
    cstats = tp.PairStats(*xr.CROSS_PAIRS[pair], "us", "ud")
    errors: list = []

    prod_sum = (checksum_bytes if producer_checksum is _UNSET
                else producer_checksum)

    def produce():
        try:
            tp.run_producer_pair(region, sems, ops, ops.create_stream(0),
                                 pair=pair, descs=descs, stats=pstats,
                                 budget_s=budget_s, slot_bytes=slot_bytes,
                                 checksum_bytes=prod_sum)
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
    # THE ASYMMETRIC ARM: this consumer has a summer, the producer does not.
    # Handing neither side one would return at _report_checksum's FIRST guard
    # and never reach the one under test -- which is exactly how mutant E
    # survived the first round.
    _cross_round_trip(region, sems, ops, descs, pair=0,
                      checksum_bytes=byte_sum(ops), producer_checksum=None,
                      on_checksum=reports.append)
    assert reports == [], (
        "this consumer compared its own real sum against a 0 the producer "
        "never computed -- an unarmed instrument reading as a failed one")
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
    assert "leg=7" in verdict.line(leg=7, epoch="b.7")


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


def test_the_corridor_floor_is_applied_and_not_merely_printed():
    """SURVIVING MUTANT (this slice's own mutant K): dropping the floor from
    ``affordable`` left every test green.

    Both existing cases were decided by the SIGN -- 8 GiB against 474 MiB free
    is negative with or without a floor, 512 MiB against 4 GiB is comfortable
    either way -- so nothing exercised the floor itself.  The floor is the
    user's reserve (``Reserve-Semantik``: 1024 MiB per card is THEIR free
    space, not an internal allowance), and a shadow that spends it has taken
    something that was never the mechanism's to take.

    This is the case in between: a need that FITS the free column and leaves
    less than the reserve behind.
    """
    free = 1200
    need_mib = 500  # 1200 - (500 + 64) = 636 MiB left, under the 1024 floor
    price = sh.price_shadow("GPU-3080", (need_mib - 64) * sh.MIB, free)
    assert price.need_mib == need_mib
    assert price.free_mib - price.need_mib > 0, "the sign must not decide this"
    assert not price.affordable, (
        "the shadow fitted itself into the user's 1024 MiB reserve")
    assert "floor_mib=1024" in price.message()
    # And one MiB of headroom the other way is affordable, so the boundary is
    # the floor and not an accident.
    assert sh.price_shadow("GPU-3080", (need_mib - 64) * sh.MIB, free,
                           floor_mib=636).affordable


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
        # A HARNESS PROPERTY, not a product one: the two FakeDeviceOps share
        # one card's storage (that is what makes the on-card lane real here)
        # but each has its OWN bump allocator starting at 0, so the source's
        # bounce and the destination's shadow buffer would alias at offset 0.
        # The spacer moves the destination's allocations clear of it.  On the
        # metal two processes' cudaMalloc cannot alias.
        dst_ops.raw_malloc(0, 2 << 20)
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
                free_mib=8192, log=lambda _s: None, budget_s=10.0,
                slot_bytes=SLOT, oncard_slot_bytes=SLOT))

        thread = threading.Thread(target=source)
        thread.start()
        run = sh.shadow_transport(
            region=region, sems=sems, ops=dst_ops, row=3, rank=0, device=0,
            card_uuid="u0", uuid_of_card=["u0", "u1", "u2"], descs=descs,
            is_source=False, oncard_mode=tp.ONCARD_MODE_IPC, peer_row=0,
            wave=WAVE, leg=0, direction="P->D", epoch=f"{boot}.1",
            free_mib=8192, log=lambda _s: None, budget_s=10.0,
            slot_bytes=SLOT, oncard_slot_bytes=SLOT, stripe_bytes=1 << 20,
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
                      "slot_wait_ms=", "gate_skew_ms=", "lock_wait_ms=",
                      "tag=weights_0", "slot_checksum_not_representable="):
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
        dst_ops.raw_malloc(0, 2 << 20)  # see the spacer note above
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
            free_mib=8192, log=lambda _s: None, budget_s=10.0,
            slot_bytes=SLOT, oncard_slot_bytes=SLOT))
        thread.start()
        run = sh.shadow_transport(
            region=region, sems=sems, ops=dst_ops, row=3, rank=0, device=0,
            card_uuid="u0", uuid_of_card=["u0", "u1", "u2"], descs=descs,
            is_source=False, oncard_mode=tp.ONCARD_MODE_IPC, peer_row=0,
            wave=WAVE, leg=0, direction="P->D", epoch=f"{boot}.1",
            free_mib=8192, log=lambda _s: None, budget_s=10.0,
            slot_bytes=SLOT, oncard_slot_bytes=SLOT, stripe_bytes=1 << 20,
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
        # Row 0 is the co-located SOURCE, which does not run in this test:
        # without its vote the GATE is what stops the shadow, and the failure
        # under test (the transport's) would never be reached.
        _vote_rows(region, [0, 1, 2, 4, 5], leg=0, vote=True,
                   classes_hash=sh.classes_hash(["o_proj"]), need_mib=0)
        lines: list = []
        run = sh.shadow_transport(
            region=region, sems=sems, ops=dst_ops, row=3, rank=0, device=0,
            card_uuid="u0", uuid_of_card=["u0", "u1", "u2"], descs=descs,
            is_source=False, oncard_mode=tp.ONCARD_MODE_IPC, peer_row=0,
            wave=WAVE, leg=0, direction="P->D", epoch=f"{boot}.1",
            free_mib=8192, log=lines.append, budget_s=0.4,
            slot_bytes=SLOT, oncard_slot_bytes=SLOT, stripe_bytes=1 << 20,
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



# ===========================================================================
# ITEM 4 -- the two S6 must_fix, carried as far as they are S5-shaped.
# DANGER DIRECTION: a per-boot arm that only fails inside the transport, and a
# handshake that starts a leg carrying the previous leg's counts.
# ===========================================================================


def _publish_matrix(region, *, modes):
    zeros = [0] * xr.N_RANKS
    for row, mode in enumerate(modes):
        region.write_matrix_row(row, zeros, zeros, 0xABC, pid=1000 + row,
                                oncard_mode=mode)


def _check(region, row=0):
    return xr.gate0_check(region, row, census_unavailable_reason="s5-test",
                          budget_s=1.0)


def test_the_gate_0_row_carries_the_on_card_mode(region):
    """The field exists, survives the seal, and reads back by NAME.

    S4-fix closed the SILENT half of refusal A (the slot geometry, in the
    on-card row).  This is the LOUD half's channel: the mode is a per-BOOT arm,
    so Gate 0 -- which runs before a byte moves -- is where it belongs.
    """
    _publish_matrix(region, modes=[xr.ONCARD_MODE_WORD["ipc"]] * xr.N_RANKS)
    got = region.read_matrix_row(2)
    assert got.sealed and got.oncard_mode_name == "ipc"
    assert _check(region)["oncard_mode"] == "ipc"


def test_ranks_that_disagree_about_the_on_card_mode_are_refused_before_a_byte_moves(
        region):
    """The LOUD half of S4-fix refusal A.

    Without it the disagreement surfaces INSIDE the transport, after the flip
    has started: a zero IPC handle on one side, a ``HostBounce(create=False)``
    ENOENT on the other.  Both are loud, and both are late.
    """
    modes = [xr.ONCARD_MODE_WORD["ipc"]] * xr.N_RANKS
    modes[4] = xr.ONCARD_MODE_WORD["host"]
    _publish_matrix(region, modes=modes)
    with pytest.raises(xr.Weg2XchgPlanDisagree) as excinfo:
        _check(region)
    message = str(excinfo.value)
    assert "on-card MODE disagreement" in message
    assert "row=4:host" in message and "row=0:ipc" in message
    assert "no byte has moved" in message


def test_an_unstated_on_card_mode_is_not_a_disagreement(region):
    """Until S6 passes the arm down, every rank publishes 0.

    A gate that refused that would refuse every flip of every boot before S6
    exists -- a check that cannot pass is as useless as one that cannot fail,
    and it would be discovered at the worst possible moment.
    """
    _publish_matrix(region, modes=[xr.ONCARD_MODE_UNSTATED] * xr.N_RANKS)
    assert _check(region)["oncard_mode"] == "unstated"


def test_a_mode_stated_by_only_some_ranks_is_a_disagreement(region):
    """Half a boot's ranks knowing the arm is worse than none of them knowing.

    ``unstated`` is a coherent whole-boot state (pre-S6); ``unstated`` BESIDE a
    stated mode is a rank that did not get the arm, which is the launch defect
    this refusal exists to name.
    """
    modes = [xr.ONCARD_MODE_WORD["ipc"]] * xr.N_RANKS
    modes[5] = xr.ONCARD_MODE_UNSTATED
    _publish_matrix(region, modes=modes)
    with pytest.raises(xr.Weg2XchgPlanDisagree) as excinfo:
        _check(region)
    assert "on-card MODE disagreement" in str(excinfo.value)


def test_the_verdict_republish_keeps_the_mode(region):
    """``write_matrix_verdict`` is a read-modify-write of the rank's own row.

    It re-published four fields and would have zeroed the fifth, turning every
    rank that failed its local check into a mode disagreement as well -- one
    refusal manufacturing another, which is how a log stops naming the cause.
    """
    _publish_matrix(region, modes=[xr.ONCARD_MODE_WORD["host"]] * xr.N_RANKS)
    region.write_matrix_verdict(3, False)
    assert region.read_matrix_row(3).oncard_mode_name == "host"


def test_a_stale_full_count_refuses_the_leg_by_name(boot):
    """S4-fix refusal C, carried forward: the 24 counts at the leg's start.

    A flip abandoned after gate 1 rolls forward by design (W57) and leaves any
    posted-but-untaken ``full`` at 1 for the rest of the boot.  The next
    producer then blocks on ``empty`` for the whole fence budget and names a
    healthy consumer.
    """
    xr.create_semaphores(boot)
    sems = tp.SemSet(boot)
    try:
        assert tp.verify_sem_arm(sems, leg=1, epoch=f"{boot}.1")["stale"] == 0
        sems.post(2, 1, "full")  # the leftover a rolled-forward flip leaves
        with pytest.raises(tp.Weg2XchgSemaphoreNotRearmed) as excinfo:
            tp.verify_sem_arm(sems, leg=2, epoch=f"{boot}.2")
        message = str(excinfo.value)
        assert tp.SEM_NOT_REARMED_MARKER in message
        assert "count=1 armed=0" in message
        assert xr.sem_name(boot, 2, 1, "full") in message
        assert "does NOT drain them" in message
        # AND IT DID NOT DRAIN IT: a check that repairs is a rank deciding
        # another flip's leftovers were harmless.
        assert sems.getvalue(2, 1, "full") == 1
    finally:
        sems.close()
        xr.unlink_semaphores(boot)


def test_the_armed_census_is_logged_so_a_pass_is_visible(boot):
    """A check whose pass is invisible cannot be told from an absent one."""
    xr.create_semaphores(boot)
    sems = tp.SemSet(boot)
    lines: list = []
    try:
        out = tp.verify_sem_arm(sems, leg=1, epoch=f"{boot}.1", log=lines.append)
        assert out["checked"] == xr.N_PAIRS * xr.SLOTS_PER_PAIR * 2 == 24
        assert lines and lines[0].startswith("WEG2-XCHG-SEMS ")
        assert f"armed={out['checked']}/{out['checked']}" in lines[0]
    finally:
        sems.close()
        xr.unlink_semaphores(boot)


def test_the_armed_counts_are_read_from_one_place(boot):
    """The check may not carry its own copy of the creation's two numbers.

    ``create_semaphores`` arms ``empty`` at 1 and ``full`` at 0; a second
    statement of that would drift, and the drift would read as a stale count on
    a healthy boot.
    """
    xr.create_semaphores(boot)
    sems = tp.SemSet(boot)
    try:
        for pair in range(xr.N_PAIRS):
            for slot in range(xr.SLOTS_PER_PAIR):
                for kind, want in tp.SEM_ARMED_COUNTS.items():
                    assert sems.getvalue(pair, slot, kind) == want
    finally:
        sems.close()
        xr.unlink_semaphores(boot)


# ===========================================================================
# THE FIX ROUND -- one test per must_fix of the S5 review and the S5 refuter.
# DANGER DIRECTION of the whole block, and it is one sentence: an instrument
# that cannot go red, and VRAM this instrument spends without pricing it.
# ===========================================================================


def _report(*, expected, got, nbytes=1024):
    return tp.ChecksumReport(lane="cross", pair=0, slot=1, seq=4,
                             nbytes=nbytes, expected=expected, got=got)


def test_the_slot_checksum_asks_representability_before_it_says_mismatch():
    """MUST_FIX (S5 review 1): rule 3 was asked for STRIPES only.

    The slot path -- the transport's producer-vs-consumer comparison, which
    reaches the same W59 marker -- logged on a bare inequality.  A batch whose
    two ends framed the field differently was therefore reported as a data
    corruption, which is #656 register C22 exactly: an instance killed for a
    corruption that had not happened.
    """
    counters = sh.ShadowCounters()
    lines: list = []
    # 1024 bytes can sum to at most 261120; this value never was a checksum.
    verdict = sh.report_slot_checksum(
        _report(expected=4626949667419791296, got=17), counters,
        leg=1, epoch="b.1", log=lines.append)
    assert verdict == sh.NOT_REPRESENTABLE
    assert counters.checksum_not_representable == 1
    assert counters.checksum_mismatch == 0, (
        "a framing error was counted as a corruption of the staged bytes")
    assert "never a checksum of this batch" in lines[-1]
    assert sh.MISMATCH_MARKER in lines[-1] and "verdict=NOT-REPRESENTABLE" in lines[-1]
    # A REAL disagreement, both sides representable, still reports -- the fix
    # may not have turned the instrument off.
    assert sh.report_slot_checksum(_report(expected=100, got=101), counters,
                                   leg=1, epoch="b.1",
                                   log=lines.append) == sh.MISMATCH
    assert counters.checksum_mismatch == 1
    assert "STAGED bytes changed" in lines[-1]
    # And agreement logs nothing at all.
    before = len(lines)
    assert sh.report_slot_checksum(_report(expected=100, got=100), counters,
                                   leg=1, epoch="b.1",
                                   log=lines.append) == sh.MATCH
    assert len(lines) == before and counters.checksum_reports == 3


def test_one_representability_question_serves_both_checksums():
    """Rule 3 says the function is IMPORTED, never re-derived -- once."""
    assert sh.checksum_representable(255 * 16, 16)
    assert not sh.checksum_representable(255 * 16 + 1, 16)
    assert not sh.checksum_representable(-1, 16)


def test_a_compare_with_no_summer_may_not_read_as_a_pass(ops):
    """MUST_FIX (S5 review 2): the ``ran = False`` of the no-summer guard was
    unpinned -- the refuter's own mutant deleted it and 46 tests stayed green.

    Mutated, a destination whose compare got no summer emitted ``ran=yes``
    with ``stripes=0``, which is the "unarmed instrument reads as passed" the
    compare's own docstring claims that line prevents.  Same lesson as this
    slice's mutants E and K: a line no case exercises is not a property.
    """
    result = sh.ShadowResult(leg=0, epoch="b.1",
                             subset=sh.select_subset([], leg=0),
                             counters=sh.ShadowCounters(), ran=True)
    run = sh.ShadowRun(result, sh.ShadowBuffers(ops, 0, 4096), (), {})
    out = run.compare(None, lambda _s: None)
    assert out.ran is False, "an unarmed compare reported itself as a run"
    line = out.line()
    assert "ran=no" in line and "verdict=NOT-RUN" in line
    assert "stripes=0" in line
    assert ops.freed, "the buffers outlived the compare"


def test_a_compare_that_compared_nothing_is_not_a_match():
    """The third verdict, in the CODE and not only in the boot ticket's grep."""
    counters = sh.ShadowCounters()
    assert counters.verdict == sh.NO_STRIPES
    counters.stripes, counters.match = 2, 2
    assert counters.verdict == sh.MATCH
    counters.mismatch = 1
    assert counters.verdict == sh.MISMATCH
    ran_but_empty = sh.ShadowResult(leg=0, epoch="b.1",
                                    subset=sh.select_subset([], leg=0),
                                    counters=sh.ShadowCounters(), ran=True)
    assert "verdict=NO-STRIPES" in ran_but_empty.line()


def _diag(rank, nbytes, cls="qkv_proj"):
    return wx.XchgDesc(tag="weights_0", src_rank=rank, dst_rank=rank,
                       param_name=f"model.layers.0.attn.{cls}.weight",
                       kind=wx.FLAT, nbytes=nbytes, rows=1, run_bytes=nbytes,
                       spitch=0, dpitch=0, src_ptr=0x1000, dst_ptr=0x2000)


def test_the_sources_on_card_bounce_is_priced_and_can_refuse():
    """MUST_FIX (S5 review 3 / refuter 2): the source priced ZERO and then
    allocated the bounce anyway.

    ``run_leg``'s diagonal raw-mallocs ``slots x slot_bytes`` on the
    PRODUCER's card (``OnCardBounce``).  With ``need_bytes=0`` the source voted
    YES unconditionally and took up to 2 x 128 MiB on exactly the card the boot
    ticket expects to refuse -- the one failure this module names as forbidden,
    unpriced, on the leg nobody looked at.
    """
    descs = [_diag(0, 4 << 30)]
    price, slot = sh.price_leg("GPU-5090", descs, rank=0, is_source=True,
                               oncard_mode=tp.ONCARD_MODE_IPC, free_mib=8192)
    assert price.bounce_mib == tp.ONCARD_SLOTS * slot // sh.MIB > 0
    assert price.dst_mib == 0 and price.scratch_mib == 0
    assert price.need_mib == price.bounce_mib
    assert "oncard_bounce=" in price.message()
    assert f"bounce_mib={price.bounce_mib}" in price.line()
    # sb5f's own reading: the card the ticket expects to refuse now refuses on
    # the SOURCE leg too, which it could not before.
    tight, _ = sh.price_leg("GPU-5090", descs, rank=0, is_source=True,
                            oncard_mode=tp.ONCARD_MODE_IPC, free_mib=474)
    assert not tight.affordable
    # The IMPORTING side maps the exporter's allocation; the host degrade's
    # bounce is a shm file.  Neither is VRAM on this card.
    imported, _ = sh.price_leg("GPU-5090", descs, rank=0, is_source=False,
                               oncard_mode=tp.ONCARD_MODE_IPC, free_mib=8192)
    assert imported.bounce_mib == 0
    degraded, _ = sh.price_leg("GPU-5090", descs, rank=0, is_source=True,
                               oncard_mode=tp.ONCARD_MODE_HOST, free_mib=8192)
    assert degraded.bounce_mib == 0


def test_the_diagonal_slot_is_this_cards_lane_and_not_the_sum():
    """MUST_FIX (refuter 5): the slot was priced from the subset's on-card
    bytes summed across ALL THREE cards.

    ``run_leg`` selects the diagonal with ``src_rank == rank``: the lane runs
    once per CARD between two co-located processes.  The sum-as-one-lane
    reading is the exact error ``XchgPlan.oncard_bytes_by_rank`` was written to
    stop (S5-pre's own "10.28 GiB = 329 batches"), and it had been re-committed
    one seam over -- inflating the bounce above by the same factor.
    """
    descs = [_diag(0, 4 << 30), _diag(1, 4 << 30)]
    _, mine = sh.price_leg("GPU-5090", descs, rank=0, is_source=True,
                           oncard_mode=tp.ONCARD_MODE_IPC, free_mib=8192)
    assert mine == tp.plan_oncard_slot_bytes(4 << 30).slot_bytes
    summed = tp.plan_oncard_slot_bytes(8 << 30).slot_bytes
    assert mine < summed, "the two denominators must be distinguishable here"
    # Both co-located processes derive it from the same filter over the same
    # descriptors, which is what keeps the W52 slot_bytes check a cross-check.
    _, peer = sh.price_leg("GPU-5090", descs, rank=0, is_source=False,
                           oncard_mode=tp.ONCARD_MODE_IPC, free_mib=8192)
    assert peer == mine


def test_the_resume_reserve_is_subtracted_and_printed():
    """MUST_FIX (refuter 4), the half of it that is code.

    ``free_mib`` is read at HOOK time; the destination's ring ``resume`` maps
    its image AFTER that instant, while the shadow's buffers are still alive.
    A price that ignores the not-yet-resumed demand can pass a subset that then
    OOMs the authoritative ``cu_mem_create`` mid-flip.  The term is a caller's
    to supply and the line always prints it, so an unwired boot reads as
    unwired instead of as priced.
    """
    free, need_bytes = 2000, 512 * sh.MIB
    assert sh.price_shadow("GPU-5090", need_bytes, free).affordable
    with_reserve = sh.price_shadow("GPU-5090", need_bytes, free,
                                   reserve_bytes=600 * sh.MIB)
    assert not with_reserve.affordable
    assert "resume_reserve_mib=600" in with_reserve.line()
    assert "resume_reserve_mib=600" in with_reserve.message()
    params = inspect.signature(sh.shadow_transport).parameters
    assert params["resume_reserve_bytes"].default == 0


def _wide_descs(classes, nbytes):
    return [wx.XchgDesc(tag=f"weights_{i}", src_rank=1, dst_rank=0,
                        param_name=f"model.layers.{i}.attn.{cls}.weight",
                        kind=wx.FLAT, nbytes=nbytes, rows=1, run_bytes=nbytes,
                        spitch=0, dpitch=0, src_ptr=0x1000, dst_ptr=0x2000)
            for i, cls in enumerate(classes)]


def test_the_full_leg_is_priced_on_every_leg_and_grades_nothing(region):
    """MUST_FIX (S5 review 4): the automatic path priced only the SUBSET.

    The boot ticket predicts "the 5090 REFUSES at full size, and no refusal
    there is itself a finding" -- but no full-size budget line could ever
    appear on a default shadow boot, so that absence was guaranteed by the code
    and would have been read as a finding.  An absence that cannot occur is not
    evidence.  The full price is now printed on every leg, marked
    ``graded=no``, and the VOTE still comes from the subset.
    """
    descs = _wide_descs(["qkv_proj", "o_proj", "down_proj"], 900 * sh.MIB)
    # Every other rank votes NO so the gate returns at once: what is under test
    # is what was logged BEFORE it, and the ordering price -> gate is the
    # property that lets a rank vote its own card's arithmetic.
    _vote_rows(region, [1, 2, 3, 4, 5], leg=0, vote=False)
    lines: list = []
    run = sh.shadow_transport(
        region=region, sems=None, ops=None, row=0, rank=0, device=0,
        card_uuid="GPU-5090", uuid_of_card=["u0", "u1", "u2"], descs=descs,
        is_source=False, oncard_mode=tp.ONCARD_MODE_IPC, peer_row=3, wave=WAVE,
        leg=0, direction="P->D", epoch="b.1", free_mib=2100, log=lines.append)
    budgets = [ln for ln in lines if ln.startswith(sh.SHADOW_BUDGET_LINE_PREFIX)]
    assert len(budgets) == 2, budgets
    full = [ln for ln in budgets if "scope=full" in ln][0]
    subset = [ln for ln in budgets if "scope=subset" in ln][0]
    assert "graded=no" in full and "verdict=REFUSED" in full
    assert "graded=yes" in subset and "verdict=AFFORDABLE" in subset
    # The vote is the SUBSET's: the full price decided nothing.
    assert sh.read_shadow_vote(region, 0)["vote"] == sh.VOTE_YES
    assert not run.result.ran


class _FakeScratch:
    """A tensor-shaped scratch over the double's storage.

    ``data_ptr``/``numel``/``element_size``/slicing -- the four things the
    summer is allowed to know about it, which is what makes the coupling
    testable without torch.
    """

    def __init__(self, ops, nbytes: int):
        self.ops = ops
        self.nbytes = int(nbytes)
        self.ptr = ops.raw_malloc(0, self.nbytes) if self.nbytes else 0

    def data_ptr(self) -> int:
        return self.ptr

    def numel(self) -> int:
        return self.nbytes

    def element_size(self) -> int:
        return 1

    def __getitem__(self, sl):
        return read(self.ops, self.ptr, int(sl.stop))


def test_the_device_summer_cannot_be_a_dead_instrument(ops):
    """MUST_FIX (refuter 1): the scratch was bound AFTER construction, through
    an attribute with no caller anywhere.

    Unbound it was ``torch.empty(0)``: ``scratch[:n]`` empty, ``uint8_checksum``
    0 for BOTH sides of every stripe, ``verdict=MATCH`` on a leg that moved
    nothing -- with the boot's acceptance being ``mismatch=0``.  Nothing
    coupled the memcpy TARGET to the tensor's storage either.  Now the tensor
    is the argument, the target is its ``data_ptr()`` and the bound is its own
    ``numel() * element_size()``.
    """
    payload = pattern(11, 300)
    src = dev_ptr(0, 0x90000)
    write(ops, src, payload)
    scratch = _FakeScratch(ops, 512)
    stream = ops.create_stream(0)
    summer = sh.device_summer(ops, stream, scratch, checksum=lambda b: sum(b))
    assert summer(src, len(payload)) == sum(payload)
    assert summer(src, 0) == 0
    # THE BOUND COMES FROM THE SCRATCH, not from the module constant: a range
    # one byte over is refused instead of silently truncated to the slice.
    with pytest.raises(ValueError) as err:
        summer(src, 513)
    assert "513" in str(err.value) and "512" in str(err.value)
    # And a scratch that was never allocated is refused where it is handed
    # over, not paid for with a run of zeros.
    with pytest.raises(ValueError) as empty:
        sh.device_summer(ops, stream, _FakeScratch(ops, 0),
                         checksum=lambda b: sum(b))
    assert "EMPTY scratch" in str(empty.value)


class _Explosive(wx.XchgDesc):
    """A descriptor whose rewrite raises, to open the window under test."""

    def replace(self, **kw):
        raise RuntimeError("rewrite exploded")


def test_the_shadow_buffer_is_freed_when_the_rewrite_raises(region, ops):
    """MUST_FIX (refuter 3): the run was built AFTER the descriptor rewrite.

    An exception in that window reached the handler, which called ``close()``
    on the STALE run (``buffers=None``) -- so the whole destination buffer
    leaked for the life of the boot, on the tight card, which is the only card
    where any of this matters.  The priced refusal self-limits FUTURE legs; the
    leaked VRAM never comes back.
    """
    desc = _Explosive(tag="weights_0", src_rank=1, dst_rank=0,
                      param_name="model.layers.0.attn.qkv_proj.weight",
                      kind=wx.FLAT, nbytes=8192, rows=1, run_bytes=8192,
                      spitch=0, dpitch=0, src_ptr=0x1000, dst_ptr=0x2000)
    _vote_rows(region, [1, 2, 3, 4, 5], leg=0, vote=True,
               classes_hash=sh.classes_hash(["qkv_proj"]), need_mib=0)
    run = sh.shadow_transport(
        region=region, sems=None, ops=ops, row=0, rank=0, device=0,
        card_uuid="u0", uuid_of_card=["u0", "u1", "u2"], descs=[desc],
        is_source=False, oncard_mode=tp.ONCARD_MODE_IPC, peer_row=3, wave=WAVE,
        leg=0, direction="P->D", epoch="b.1", free_mib=8192,
        log=lambda _s: None, stripe_bytes=1 << 20)
    assert not run.result.ran
    assert "transport-failed:RuntimeError" in run.result.reason
    assert [n for _p, n in ops.freed if n == 8192], (
        f"the shadow buffer leaked: allocated 8192, freed {ops.freed}")


def test_the_line_carries_the_tag_the_shadowed_classes_fall_in():
    """MUST_FIX (S5 review 5): spec 6/S5's line names ``tag=`` and it was not
    there; the rotation unit of this slice is the CLASS, and that deviation is
    now stated on the line and in :attr:`ShadowSubset.tags` rather than implied.
    """
    descs = _wide_descs(["qkv_proj", "o_proj"], 4096)
    # ``o_proj`` sorts first, so leg 0 shadows the class carried by tag
    # ``weights_1`` -- which is the point: the tag on the line is DERIVED from
    # the chosen classes, never assumed to be the leg's own index.
    subset = sh.select_subset(descs, leg=0)
    assert subset.classes == ("o_proj",) and subset.tags == ("weights_1",)
    result = sh.ShadowResult(leg=0, epoch="b.1", subset=subset,
                             counters=sh.ShadowCounters())
    assert "tag=weights_1" in result.line()
    # A class subset spanning two tags says both, instead of naming one.
    both = sh.select_subset(descs, leg=0, classes=["qkv_proj", "o_proj"])
    assert both.tags == ("weights_0", "weights_1")
    assert "tag=weights_0,weights_1" in sh.ShadowResult(
        leg=0, epoch="b.1", subset=both, counters=sh.ShadowCounters()).line()
    assert "tag=none" in sh.ShadowResult(
        leg=0, epoch="b.1", subset=sh.select_subset([], leg=0),
        counters=sh.ShadowCounters()).line()


def test_the_gate_skew_is_the_six_ranks_spread_not_this_ranks_wait(region):
    """FINDING (refuter): ``gate_skew_ms`` carried ``waited_s``.

    Grading spec 6/S5's ``gate_skew_ms <= 100`` against a WAIT conflates two
    numbers: the rank that arrives LAST waits ~0 and would report a skew of
    zero on the leg with the largest one.  The rows carry ``ts_ns`` already.
    """
    for row in range(1, xr.N_RANKS):
        sh.write_shadow_vote(region, row, leg=0, vote=True, classes_hash=0,
                             need_mib=0)
    time.sleep(0.01)
    verdict = sh.shadow_gate(region, 0, leg=0, vote=True, classes_hash=0,
                             need_mib=0, log=lambda _s: None)
    assert verdict.run and verdict.waited_s < 0.005
    assert verdict.skew_ms >= 10.0, (
        "the spread of the six stamps was reported as this rank's wait")
    assert "skew_ms=" in verdict.line(leg=0, epoch="b.1")


def test_the_unmeasured_wall_fields_do_not_print_as_measurements():
    """FINDING (refuter): ``ring_ms=0.000`` and ``lock_wait_ms=0.000`` had no
    producer anywhere -- zeros that read as measurements of a restore that was
    never timed.  ``n/a`` until a hook fills them.
    """
    result = sh.ShadowResult(leg=0, epoch="b.1",
                             subset=sh.select_subset([], leg=0),
                             counters=sh.ShadowCounters())
    assert "ring_ms=n/a" in result.line()
    assert "lock_wait_ms=n/a" in result.line()
    result.ring_ms, result.lock_wait_ms = 12.5, 0.25
    assert "ring_ms=12.500" in result.line()
    assert "lock_wait_ms=0.250" in result.line()


def test_the_shadow_prices_its_own_hop_and_not_the_full_diagonals(region, ops):
    """The boot ticket needs a PREDICTION beside its measurement.

    The PLAN line's ``oncard_hop_ms_priced`` prices the FULL diagonal; a shadow
    leg moves a class SUBSET over the lane of ONE card, so grading its
    ``oncard_ms`` against the plan's number compares two different lanes.  The
    line therefore carries the subset lane's own slot, batch count and priced
    hop, all three from ``ONCARD_PER_BATCH_MS`` and its measured arm.
    """
    desc = _diag(0, 3 << 20, cls="qkv_proj")
    # The gate refuses, so no thread starts: the priced hop is a property of
    # the PLAN and is printed by a leg that never ran.
    _vote_rows(region, [1, 2, 3, 4, 5], leg=0, vote=False,
               classes_hash=sh.classes_hash(["qkv_proj"]), need_mib=0)
    run = sh.shadow_transport(
        region=region, sems=None, ops=ops, row=0, rank=0, device=0,
        card_uuid="u0", uuid_of_card=["u0", "u1", "u2"], descs=[desc],
        is_source=True, oncard_mode=tp.ONCARD_MODE_IPC, peer_row=3, wave=WAVE,
        leg=0, direction="P->D", epoch="b.1", free_mib=8192,
        log=lambda _s: None, oncard_slot_bytes=1 << 20)
    result = run.result
    assert not result.ran
    assert result.oncard_slot_mib == 1
    assert result.oncard_batches == 3
    assert result.oncard_hop_ms_priced == 3 * tp.ONCARD_PER_BATCH_MS
    line = result.line()
    assert "oncard_slot_mib=1" in line and "oncard_batches=3" in line
    assert f"oncard_hop_ms_priced={3 * tp.ONCARD_PER_BATCH_MS:.3f}" in line
    # A leg with no diagonal at all prices no hop, rather than a zero that
    # reads as a measured one.
    empty = sh.ShadowResult(leg=0, epoch="b.1", subset=sh.select_subset([], leg=0),
                            counters=sh.ShadowCounters())
    assert "oncard_hop_ms_priced=n/a" in empty.line()


# ===========================================================================
# S5b -- THE TWO LEG HOOKS.
#
# DANGER DIRECTION, four shapes, and every one of them is a hook that is
# WORSE THAN NO HOOK because it looks like an instrument while it damages the
# thing it observes:
#
#   1. a hook that writes into the LIVE ARENA -- the shadow's whole licence is
#      that the ring's bytes are untouched; a hook whose destination is the
#      ring's own pointer serves the exchange's bytes with the ring's
#      authority and none of its proof;
#   2. a hook that SWALLOWS A MISMATCH -- the acceptance is `mismatch=0`, so a
#      hook that drops the compare's counters on the floor turns the one
#      finding this slice exists for into a green line;
#   3. a hook whose FAILURE ABORTS THE LEG -- zero authority, enumerated in
#      the module docstring, is exactly the property a caller can undo;
#   4. a hook that runs ON ONE RANK ONLY -- the gate is rank-uniform, so a
#      rank that returns without voting is not "one less" but five ranks
#      waiting out a gate budget inside their own flip legs.
# ===========================================================================


def _inputs(hook, *, leg=0, rank=0, row=0, peer_row=3, epoch="e.0",
            direction="P->D", **kw):
    return sh.ShadowLegInputs(leg=leg, epoch=epoch, direction=direction,
                              hook=hook, rank=rank, row=row, peer_row=peer_row,
                              device=0, card_uuid="u0",
                              uuid_of_card=("u0", "u1", "u2"),
                              free_mib=kw.pop("free_mib", 8192), **kw)


@pytest.fixture()
def no_active_leg():
    """No shadow run may leak OUT of a test either."""
    sh.set_plan_provider(None)
    yield
    active = sh._active_leg()
    if active is not None:
        active.close(owner=active.token)
    sh.set_plan_provider(None)
    assert sh._active_leg() is None


# --- the arm: the ring path may not reach a single line of this ------------

def test_the_hooks_are_never_reached_on_the_ring_arm(monkeypatch, no_active_leg):
    """DEFAULT = RING = BYTE-IDENTICAL, and it is proven by a TRIPWIRE.

    Asserting "it returned None" would pass for a hook that did all its work
    and then discarded it, so every door out of the arm check is mined: a
    plan provider, a region opener and a semaphore set that all raise if the
    ring path so much as looks at them.
    """
    monkeypatch.delenv(wx.WEIGHT_SOURCE_ENV, raising=False)
    assert wx.weight_source() == wx.WEIGHT_SOURCE_RING
    assert sh.shadow_armed() is False

    def tripwire(*_a, **_k):
        raise AssertionError("the ring arm reached the shadow's machinery")

    monkeypatch.setattr(sh, "plan_for_leg", tripwire)
    monkeypatch.setattr(sh.xr.XchgRegion, "open", staticmethod(tripwire))
    monkeypatch.setattr(sh.tp, "SemSet", tripwire)
    monkeypatch.setattr(sh.tp, "verify_sem_arm", tripwire)
    lines: list = []
    assert sh.run_leg_hook(_inputs(sh.HOOK_DESTINATION), log=lines.append) is None
    assert lines == [], lines
    assert sh._active_leg() is None


def test_the_shadow_arm_reaches_the_hook(monkeypatch, no_active_leg):
    """The can-fail control for the test above: on ``shadow`` it DOES run.

    Without this the previous test passes for a hook that is dead on every
    arm, which is the unarmed-instrument-reading-as-a-passed-one shape this
    file already caught twice.
    """
    monkeypatch.setenv(wx.WEIGHT_SOURCE_ENV, sh.WEIGHT_SOURCE_SHADOW)
    assert sh.shadow_armed() is True
    lines: list = []
    result = sh.run_leg_hook(_inputs(sh.HOOK_DESTINATION), log=lines.append,
                             descs=[_diag(0, 4096)])
    assert result is not None
    assert result.reason == "no-region"
    assert any(ln.startswith("WEG2-XCHG-SHADOW ") for ln in lines)


# --- the plan seam ---------------------------------------------------------

def test_the_plan_seam_has_no_producer(no_active_leg):
    """``build_plan`` has ZERO product callers, and that is why W63 says so.

    This is a DENOMINATOR test, not a style one: the shadow's ``no-plan``
    reason claims an absence, and an absence nobody re-checks is the one that
    rots.  The day S6 wires a producer this goes red and the W63 docstring
    gets corrected instead of lying.
    """
    root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    hits = []
    for dirpath, _dirs, names in os.walk(os.path.join(root, "python", "sglang")):
        for n in names:
            if not n.endswith(".py") or n == "weight_exchange.py":
                continue
            p = os.path.join(dirpath, n)
            with open(p, encoding="utf-8") as fh:
                for i, line in enumerate(fh, 1):
                    if "build_plan(" in line:
                        hits.append(f"{os.path.relpath(p, root)}:{i}")
    assert hits == [], hits
    assert sh.plan_for_leg("P->D", 0, 0) == ()


def test_a_leg_without_descriptors_says_no_plan_and_never_says_match(no_active_leg):
    """MUTANT: the hook treats an empty plan as a run.  It must not.

    An empty plan produces no stripes, and ``verdict`` would then be
    ``NO-STRIPES`` -- already not ``MATCH`` -- but the LINE must also carry a
    reason a reader can act on, because ``NO-STRIPES`` alone does not say
    whether the subset was empty or the wiring is missing.
    """
    lines: list = []
    result = sh.run_leg_hook(_inputs(sh.HOOK_DESTINATION), log=lines.append,
                             descs=(), armed=True)
    assert result.reason == "no-plan"
    assert "verdict=NOT-RUN" in result.line()
    assert any(sh.RANK_LOCAL_SKIP_MARKER in ln for ln in lines)
    assert any("build_plan has no product caller" in ln for ln in lines)


def test_the_plan_provider_is_the_seam_and_it_is_consulted(no_active_leg):
    """The can-fail control for the seam: a provider IS read."""
    seen: list = []

    def provider(direction, leg, rank):
        seen.append((direction, leg, rank))
        return ()

    previous = sh.set_plan_provider(provider)
    try:
        sh.run_leg_hook(_inputs(sh.HOOK_SOURCE, leg=7, rank=2),
                        log=lambda _s: None, armed=True)
    finally:
        sh.set_plan_provider(previous)
    assert seen == [("P->D", 7, 2)]


# --- rank-uniformity (danger 4) -------------------------------------------

def test_a_rank_that_cannot_join_says_so_by_name_and_does_not_go_quiet(
        no_active_leg):
    """DANGER 4.  Every local skip is W63 with its reason, never a bare return.

    Enumerated rather than sampled: each door out of the hook before the gate
    must print the marker, because the gate is rank-uniform and a silent skip
    is indistinguishable at the other five rows from a card that refused on
    arithmetic.
    """
    for reason, kwargs in (
        ("no-region", {"descs": [_diag(0, 4096)]}),
        ("no-plan", {"descs": ()}),
    ):
        lines: list = []
        result = sh.run_leg_hook(_inputs(sh.HOOK_DESTINATION),
                                 log=lines.append, armed=True, **kwargs)
        assert result.reason == reason
        assert any(sh.RANK_LOCAL_SKIP_MARKER in ln for ln in lines), reason
        assert any(f"reason={reason}" in ln for ln in lines), reason
        assert any(f"NO for all {xr.N_RANKS} rows" in ln for ln in lines), reason


def test_an_explicit_caller_gets_the_skip_raised_and_the_leg_never_does(
        no_active_leg):
    """The W56/W61 two-arm shape, third instance: automatic degrades, explicit raises."""
    with pytest.raises(sh.Weg2XchgShadowRankLocalSkip):
        sh.run_leg_hook(_inputs(sh.HOOK_DESTINATION), log=lambda _s: None,
                        descs=(), armed=True, explicit=True)
    # ... and the automatic path with the same inputs does NOT raise.
    assert sh.run_leg_hook(_inputs(sh.HOOK_DESTINATION), log=lambda _s: None,
                           descs=(), armed=True).reason == "no-plan"


# --- the bound (item 5) ----------------------------------------------------

def test_the_hop_bound_is_the_spec_target_times_a_named_factor():
    assert sh.SHADOW_HOP_BOUND_MS_DEFAULT == (
        tp.ONCARD_HOP_BUDGET_MS * sh.SHADOW_HOP_BOUND_FACTOR)
    assert sh.hop_bound_ms() == sh.SHADOW_HOP_BOUND_MS_DEFAULT
    assert sh.hop_bound_ms(12.5) == 12.5


def test_a_malformed_bound_is_not_silently_replaced_by_the_default(monkeypatch,
                                                                   no_active_leg):
    """MUTANT: ``except ValueError: return DEFAULT``.  A bound nobody can read
    is a bound nothing is graded against, and it would read as a passed one."""
    monkeypatch.setenv(sh.ENV_HOP_BOUND_MS, "twenty")
    with pytest.raises(ValueError):
        sh.hop_bound_ms()
    result = sh.run_leg_hook(_inputs(sh.HOOK_DESTINATION), log=lambda _s: None,
                             descs=(), armed=True)
    assert result.reason.startswith("bad-bound:")


def test_a_priced_hop_over_the_bound_refuses_the_shadow_by_name_and_not_the_flip(
        region, tmp_path, boot, no_active_leg):
    """W61 ``scope=hop``: the leg is refused BEFORE a wall is spent.

    The descriptors are a diagonal big enough that the priced hop clears the
    bound; the assertion is that the leg prints the refusal, prints what it
    would have cost, and returns a result whose ``ran`` is False -- i.e. the
    ring carried the flip and the observer stood down.
    """
    xr.create_semaphores(boot)
    sems = tp.SemSet(boot)
    try:
        big = 400 * sh.MIB
        descs = [_diag(0, big, cls="qkv_proj")]
        lines: list = []
        result = sh.run_leg_hook(_inputs(sh.HOOK_SOURCE), log=lines.append,
                                 descs=descs, region=region, sems=sems,
                                 ops=object(), armed=True, bound_ms=1.0)
    finally:
        sems.close()
        xr.unlink_semaphores(boot)
    assert result.reason == "hop-over-bound"
    assert result.ran is False
    refusals = [ln for ln in lines if sh.UNAFFORDABLE_MARKER in ln]
    assert refusals, lines
    assert "scope=hop" in refusals[0]
    assert "bound_ms=1.000" in refusals[0]
    assert result.oncard_hop_ms_priced and result.oncard_hop_ms_priced > 1.0
    assert "hop_bound_ms=1.000" in result.line()


def test_a_priced_hop_under_the_bound_is_not_refused(region, boot, no_active_leg):
    """The can-fail control for the bound: it must be able to say YES.

    Without it the refusal test passes for a bound that refuses everything,
    which is mutant K of the previous round one field over.
    """
    xr.create_semaphores(boot)
    sems = tp.SemSet(boot)
    try:
        descs = [_diag(0, 1 * sh.MIB, cls="qkv_proj")]
        lines: list = []
        result = sh.run_leg_hook(_inputs(sh.HOOK_SOURCE), log=lines.append,
                                 descs=descs, region=region, sems=sems,
                                 ops=object(), armed=True, bound_ms=1e6)
    finally:
        sems.close()
        xr.unlink_semaphores(boot)
    assert result.reason != "hop-over-bound"
    assert not [ln for ln in lines if "scope=hop" in ln]


# --- W62 at leg start (item 3) --------------------------------------------

def test_a_stale_semaphore_stops_the_shadow_and_never_the_flip(region, boot,
                                                               no_active_leg):
    """W62 gets its caller, and the caller COUNTS it.

    ``verify_sem_arm`` refuses; on the authoritative path (S6's RPC preamble)
    that refusal stops a flip.  Here the same event may only stop the SHADOW,
    so the hook catches it, prints it, and reports ``reason=w62-stale`` with
    ``sems_armed=n/a`` -- an absent census, not a passed one.
    """
    xr.create_semaphores(boot)
    sems = tp.SemSet(boot)
    try:
        sems.post(0, 0, "full")  # the leftover a rolled-forward flip leaves
        lines: list = []
        result = sh.run_leg_hook(_inputs(sh.HOOK_DESTINATION), log=lines.append,
                                 descs=[_diag(0, 4096)], region=region,
                                 sems=sems, ops=object(), armed=True)
        assert result.reason == "w62-stale"
        assert result.sems_armed is None
        assert "sems_armed=n/a" in result.line()
        assert any(tp.SEM_NOT_REARMED_MARKER in ln for ln in lines)
    finally:
        sems.close()
        xr.unlink_semaphores(boot)


def test_an_armed_semaphore_set_is_reported_as_a_number_not_as_silence(
        region, boot, no_active_leg):
    """A check whose PASS is invisible cannot be told from an absent one."""
    xr.create_semaphores(boot)
    sems = tp.SemSet(boot)
    try:
        result = sh.run_leg_hook(_inputs(sh.HOOK_DESTINATION),
                                 log=lambda _s: None, descs=[_diag(0, 4096)],
                                 region=region, sems=sems, ops=object(),
                                 armed=True, bound_ms=0.0)
        assert result.reason == "hop-over-bound"
        assert result.sems_armed == xr.N_PAIRS * xr.SLOTS_PER_PAIR * 2
        assert f"sems_armed={result.sems_armed}" in result.line()
    finally:
        sems.close()
        xr.unlink_semaphores(boot)


# --- the Gate-0 mode word gets a producer ---------------------------------

def test_the_on_card_mode_word_has_a_producer_now(monkeypatch):
    """SECTION 1ai-S5 UNPROVEN 10 closed on the desk half."""
    assert sh.oncard_mode_word(tp.ONCARD_MODE_IPC) == xr.ONCARD_MODE_WORD["ipc"]
    assert sh.oncard_mode_word(tp.ONCARD_MODE_HOST) == xr.ONCARD_MODE_WORD["host"]
    # An unknown mode is UNSTATED and never a guess: gate0_check reads UNSTATED
    # as "not a disagreement" and a wrong word would BE one.
    assert sh.oncard_mode_word("banana") == xr.ONCARD_MODE_UNSTATED
    monkeypatch.setenv(tp.ENV_ONCARD_MODE, "host")
    assert sh.resolve_shadow_oncard_mode() == tp.ONCARD_MODE_HOST
    monkeypatch.delenv(tp.ENV_ONCARD_MODE)
    assert sh.resolve_shadow_oncard_mode() == tp.ONCARD_MODE_IPC


def test_the_shadow_never_probes_for_the_on_card_mode():
    """An observer that PROBES has allocated on the card it is observing."""
    src = inspect.getsource(sh.resolve_shadow_oncard_mode)
    body = src.split('"""')[2]  # everything after the docstring: the CODE
    # The DOCSTRING names both words -- it is what says the probe is refused --
    # so the assertion has to be over the body, which is the thing that runs.
    assert "resolve_oncard_mode" not in body, body
    assert "probe" not in body, body


# --- lifetime (item 4) -----------------------------------------------------

def test_a_stale_run_is_closed_by_name_and_never_carried_across_flips(
        no_active_leg):
    """MUTANT: the active slot is overwritten silently.

    One leaked raw ``cudaMalloc`` per flip on the card that is already 550 MiB
    below the corridor floor is the one failure a zero-authority instrument
    may not cause.
    """
    stale = sh.ShadowLeg(_inputs(sh.HOOK_SOURCE, leg=0), lambda _s: None).adopt()
    assert sh._active_leg() is stale
    lines: list = []
    sh.run_leg_hook(_inputs(sh.HOOK_DESTINATION, leg=1), log=lines.append,
                    descs=(), armed=True)
    assert stale.closed is True
    assert sh._active_leg() is None
    assert any("reason=stale-run" in ln for ln in lines), lines


def test_a_leg_never_closes_a_region_or_a_device_it_was_handed(region, boot,
                                                               no_active_leg):
    """MEASURED DEFECT of this round, and it presented as a SIGSEGV.

    ``close()`` used to close region, semaphores and device ops
    unconditionally, so a leg HANDED those by its caller unmapped them out from
    under it -- the end-to-end test's source leg finished first and pulled the
    shared mmap away while the destination leg was still reading it: the whole
    process died and ``junit.xml`` was never written, which is the run shape
    #1281 exists to make loud.  In the product it is the same event with two
    co-located ranks and one region.

    "A run cannot be closed by a caller that does not own it" now also reads:
    a run does not own what it was handed.
    """
    xr.create_semaphores(boot)
    sems = tp.SemSet(boot)
    try:
        leg = sh.ShadowLeg(_inputs(sh.HOOK_SOURCE), lambda _s: None)
        assert leg.attach(region=region, sems=sems, ops=object()) == ""
        assert leg._opened == set()
        assert leg.close(owner=leg.token) is True
        # Still usable: nothing of the caller's was closed.
        assert sems.getvalue(0, 0, "empty") == 1
        region.dir_view()
    finally:
        sems.close()
        xr.unlink_semaphores(boot)


def test_two_legs_in_one_interpreter_do_not_close_each_other(no_active_leg):
    """MEASURED DEFECT of this round: the active slot was PROCESS-wide.

    In the product the two hooks of one flip are two rank PROCESSES, so the two
    forms are the same thing there -- but wherever a second leg shares the
    interpreter (this file's own two-sided test) the second adopt() closed the
    first leg out from under itself, and it presented as a gate expiry
    ``only n/6 ranks published`` five seconds inside a flip.  The leg is owned
    by the THREAD that runs it; a stale run carried across FLIPS is the same
    thread and is still caught.
    """
    a = sh.ShadowLeg(_inputs(sh.HOOK_SOURCE, leg=0), lambda _s: None).adopt()
    seen: list = []

    def other():
        b = sh.ShadowLeg(_inputs(sh.HOOK_DESTINATION, leg=0),
                         lambda _s: None).adopt()
        seen.append(sh._active_leg() is b)
        b.close(owner=b.token)

    t = threading.Thread(target=other)
    t.start()
    t.join(30)
    assert seen == [True]
    assert a.closed is False, "the other thread's leg closed this one"
    assert sh._active_leg() is a
    a.close(owner=a.token)


def test_a_handler_cannot_close_a_run_it_does_not_own(no_active_leg):
    """The token is the whole guard: a wrong owner is a NO-OP, not a free."""
    leg = sh.ShadowLeg(_inputs(sh.HOOK_SOURCE), lambda _s: None)
    assert leg.close(owner="somebody-elses-token") is False
    assert leg.closed is False
    assert leg.close(owner=leg.token) is True
    assert leg.close(owner=leg.token) is True  # idempotent


def test_the_hook_closes_its_run_on_every_path_including_the_raising_one(
        no_active_leg):
    """The ``finally`` is the property, so it is tested through a RAISE."""
    boom = _inputs(sh.HOOK_DESTINATION)

    def exploding_provider(direction, leg, rank):
        raise RuntimeError("no plan for you")

    previous = sh.set_plan_provider(exploding_provider)
    try:
        result = sh.run_leg_hook(boom, log=lambda _s: None, armed=True)
    finally:
        sh.set_plan_provider(previous)
    assert result.reason.startswith("hook-failed:RuntimeError")
    assert sh._active_leg() is None


# --- danger 3: a hook whose failure aborts the leg ------------------------

def test_no_hook_failure_can_reach_the_leg(no_active_leg):
    """DANGER 3, enumerated over the failure classes the hook can meet."""
    for provider in (
        lambda _d, _lg, _r: (_ for _ in ()).throw(RuntimeError("plan")),
        lambda _d, _lg, _r: (_ for _ in ()).throw(MemoryError("oom")),
        lambda _d, _lg, _r: (_ for _ in ()).throw(KeyboardInterrupt()),
    ):
        previous = sh.set_plan_provider(provider)
        try:
            result = sh.run_leg_hook(_inputs(sh.HOOK_SOURCE),
                                     log=lambda _s: None, armed=True)
        finally:
            sh.set_plan_provider(previous)
        assert result is not None and result.ran is False
        assert result.reason.startswith("hook-failed:")


def test_the_weight_updater_adapters_catch_everything_and_return(no_active_leg):
    """The PRODUCT adapter, by source: a bare ``except Exception`` would let a
    ``MemoryError`` out of an observer and into a flip leg."""
    import ast as _ast

    root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    path = os.path.join(root, "python", "sglang", "srt", "managers",
                        "scheduler_components", "weight_updater.py")
    with open(path, encoding="utf-8") as fh:
        tree = _ast.parse(fh.read())
    fn = next(n for n in _ast.walk(tree)
              if isinstance(n, _ast.FunctionDef) and n.name == "_weg2_shadow_hook")
    handlers = [h for n in _ast.walk(fn) if isinstance(n, _ast.Try)
                for h in n.handlers]
    assert handlers, "the adapter has no handler at all"
    assert any(isinstance(h.type, _ast.Name) and h.type.id == "BaseException"
               for h in handlers)
    assert not any(isinstance(n, _ast.Raise) for n in _ast.walk(fn))


def test_the_two_call_sites_are_where_the_bytes_are(no_active_leg):
    """WIRING, pinned by SOURCE -- the precedent is
    ``test_weg2_xchg_cover_1273.WiringTest``.

    The SOURCE hook must sit BEFORE the pause loop: the exporter reads tensors
    allocated inside ``region(GPU_MEMORY_TYPE_WEIGHTS)``, and after
    ``pause(tag)`` those are unmapped pages -- the campaign (a) fault, pinned
    on the other side by ``test_weights_block_pause_is_the_last_statement``.
    The DESTINATION hook must sit after ``family_complete``'s reload: that is
    the only instant in the boot where the ring's restored bytes exist to be
    compared against.
    """
    root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    path = os.path.join(root, "python", "sglang", "srt", "managers",
                        "scheduler_components", "weight_updater.py")
    with open(path, encoding="utf-8") as fh:
        src = fh.read()
    assert "self._weg2_shadow_source_leg(recv_req, weights_tags, tag_bytes)" in src
    assert "self._weg2_shadow_destination_leg(" in src
    i_src = src.index("self._weg2_shadow_source_leg(recv_req")
    i_pause = src.index("self.memory_saver_adapter.pause(tag)")
    assert i_src < i_pause, "the source hook reads pages the pause has unmapped"
    i_dst = src.index("self._weg2_shadow_destination_leg(\n")
    i_reload = src.index("self._weg2_wake_reload_weights()")
    assert i_reload < i_dst, "the destination hook has no ring bytes to compare"


# --- danger 1 + 2, end to end on the double -------------------------------

def test_the_destination_hook_matches_the_ring_and_writes_nothing_into_it(
        region, tmp_path, boot, no_active_leg):
    """DANGER 1 + the acceptance line, end to end through the HOOKS.

    The ring's destination is poisoned before the pair runs and asserted
    byte-identical afterwards: the shadow may read the source's still-mapped
    VRAM and may write only its own raw buffer.  A hook that wrote into the
    live arena would be the exchange with the ring's authority and none of its
    proof -- and, because the compare's two sides would then both be the
    shadow's bytes, it would MATCH while doing it.
    """
    xr.create_semaphores(boot)
    sems_s, sems_d = tp.SemSet(boot), tp.SemSet(boot)
    src_ops = FakeDeviceOps(str(tmp_path / "d"), rank=0)
    dst_ops = FakeDeviceOps(str(tmp_path / "d"), rank=0)
    try:
        dst_ops.raw_malloc(0, 2 << 20)
        payload = pattern(41, 1800)
        src, ring_dst = dev_ptr(0, 0x10000), dev_ptr(0, 0x60000)
        write(src_ops, src, payload)
        # THE RING HAS ALREADY RESTORED, which is the product's order: this
        # hook runs after family_complete.  Identical bytes -> MATCH.
        write(dst_ops, ring_dst, payload)
        descs = [flat_desc(0, 0, len(payload), src_ptr=src, dst_ptr=ring_dst,
                           name="model.layers.0.self_attn.qkv_proj.weight")]
        _vote_rows(region, [1, 2, 4, 5], leg=0, vote=True,
                   classes_hash=sh.classes_hash(["qkv_proj"]), need_mib=0)
        lines: list = []
        thread = threading.Thread(target=lambda: sh.run_leg_hook(
            _inputs(sh.HOOK_SOURCE, row=0, peer_row=3), log=lines.append,
            descs=descs, region=region, sems=sems_s, ops=src_ops, armed=True,
            slot_bytes=SLOT, stripe_bytes=1 << 20, budget_s=10.0,
                oncard_slot_bytes=SLOT))
        thread.start()
        try:
            result = sh.run_leg_hook(
                _inputs(sh.HOOK_DESTINATION, row=3, peer_row=0),
                log=lines.append, descs=descs, region=region, sems=sems_d,
                ops=dst_ops, armed=True, sum_bytes=byte_sum(dst_ops),
                slot_bytes=SLOT, stripe_bytes=1 << 20, budget_s=10.0,
                oncard_slot_bytes=SLOT)
        finally:
            thread.join(60)
        assert result.ran, result.counters.errors
        assert read(dst_ops, ring_dst, len(payload)) == payload, \
            "the hook wrote into the RING's live destination"
        assert result.counters.stripes == 1
        assert result.counters.match == 1 and result.counters.mismatch == 0
        line = result.line()
        for token in ("WEG2-XCHG-SHADOW ", "hook=destination", "shadow_ms=",
                      "hop_bound_ms=", "sems_armed=24", "verdict=MATCH",
                      "resume_reserve_mib="):
            assert token in line, (token, line)
        assert any("hook=source" in ln for ln in lines)
        assert sh._active_leg() is None
    finally:
        src_ops.close()
        dst_ops.close()
        sems_s.close()
        sems_d.close()
        xr.unlink_semaphores(boot)


def test_the_destination_hook_cannot_swallow_a_mismatch(region, tmp_path, boot,
                                                        no_active_leg):
    """DANGER 2, and it is the can-fail control for the test above.

    The 'ring' restores DIFFERENT bytes.  The hook's own line must go red
    (``verdict=MISMATCH``), the W59 marker must be on the log, and nothing may
    be raised -- the ring's bytes were served, and this is a finding about the
    EXCHANGE.
    """
    xr.create_semaphores(boot)
    sems_s, sems_d = tp.SemSet(boot), tp.SemSet(boot)
    src_ops = FakeDeviceOps(str(tmp_path / "d"), rank=0)
    dst_ops = FakeDeviceOps(str(tmp_path / "d"), rank=0)
    try:
        dst_ops.raw_malloc(0, 2 << 20)
        payload = pattern(43, 1700)
        src, ring_dst = dev_ptr(0, 0x10000), dev_ptr(0, 0x60000)
        write(src_ops, src, payload)
        write(dst_ops, ring_dst, pattern(44, 1700))  # a DIFFERENT restore
        descs = [flat_desc(0, 0, len(payload), src_ptr=src, dst_ptr=ring_dst,
                           name="model.layers.0.mlp.down_proj.weight")]
        _vote_rows(region, [1, 2, 4, 5], leg=0, vote=True,
                   classes_hash=sh.classes_hash(["down_proj"]), need_mib=0)
        lines: list = []
        thread = threading.Thread(target=lambda: sh.run_leg_hook(
            _inputs(sh.HOOK_SOURCE, row=0, peer_row=3), log=lines.append,
            descs=descs, region=region, sems=sems_s, ops=src_ops, armed=True,
            slot_bytes=SLOT, stripe_bytes=1 << 20, budget_s=10.0,
                oncard_slot_bytes=SLOT))
        thread.start()
        try:
            result = sh.run_leg_hook(
                _inputs(sh.HOOK_DESTINATION, row=3, peer_row=0),
                log=lines.append, descs=descs, region=region, sems=sems_d,
                ops=dst_ops, armed=True, sum_bytes=byte_sum(dst_ops),
                slot_bytes=SLOT, stripe_bytes=1 << 20, budget_s=10.0,
                oncard_slot_bytes=SLOT)
        finally:
            thread.join(60)
        assert result.counters.mismatch == 1
        assert "verdict=MISMATCH" in result.line()
        assert any(sh.MISMATCH_MARKER in ln for ln in lines)
        assert any("class=down_proj" in ln for ln in lines)
    finally:
        src_ops.close()
        dst_ops.close()
        sems_s.close()
        sems_d.close()
        xr.unlink_semaphores(boot)


# --- the two producers (item 2) -------------------------------------------

def test_the_resume_reserve_and_the_ring_wall_have_producers_now(no_active_leg):
    """``resume_reserve_mib=`` and ``ring_ms=`` stop printing as absences.

    MUTANT: the hook passes 0 and None through.  Both fields exist precisely so
    an unwired boot is VISIBLY unwired, so a wired one has to be visibly wired
    -- and ``n/a`` on ``ring_ms`` must survive on the SOURCE hook, where the
    leg's wall does not exist yet.
    """
    dst = sh.run_leg_hook(
        _inputs(sh.HOOK_DESTINATION, resume_reserve_bytes=27 * sh.MIB,
                ring_ms=2311.5),
        log=lambda _s: None, descs=(), armed=True)
    assert "resume_reserve_mib=27" in dst.line()
    assert "ring_ms=2311.500" in dst.line()
    src = sh.run_leg_hook(_inputs(sh.HOOK_SOURCE), log=lambda _s: None,
                          descs=(), armed=True)
    assert "ring_ms=n/a" in src.line()
    assert "resume_reserve_mib=0" in src.line()


def test_the_adapter_reads_the_rings_own_numbers_and_estimates_neither():
    """SOURCE-pinned: both producers are READS of the leg's own instruments.

    ``resume_reserve_bytes`` is the sum of the per-tag byte counts the wake leg
    already read from the SAVER before its resume loop; ``ring_ms`` is the sum
    of the per-tag walls the leg already prints on WEG2-FLIP-TAG.  A hook that
    estimated either would be a second instrument beside the ring's own, which
    is the class UPSTREAM-MINIMAL refuses.
    """
    root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    path = os.path.join(root, "python", "sglang", "srt", "managers",
                        "scheduler_components", "weight_updater.py")
    with open(path, encoding="utf-8") as fh:
        src = fh.read()
    assert "ring_ms=sum(float(v[1]) for v in weg2_per_tag.values())" in src
    assert "int(sum(int(v) for v in (tag_bytes or {}).values()))" in src


def test_the_flip_index_is_read_back_from_the_fronts_own_epoch():
    """No second flip counter beside ``credit_epoch``'s."""
    from sglang.srt.managers.scheduler_components import weight_updater as wu

    assert wu._weg2_flip_index_of("weg2sb5g.7") == 7
    assert wu._weg2_flip_index_of(None) == -1
    assert wu._weg2_flip_index_of("no-dot") == -1
    assert wu._weg2_flip_index_of("boot.nope") == -1


# --- the launcher flag -----------------------------------------------------

def test_the_hop_bound_is_a_launcher_flag_and_is_published_to_the_ranks():
    """The bound is the LAUNCHER's, and the ring arm still publishes nothing.

    Read from the launcher SOURCE rather than by building a parser, for the
    same reason ``test_weg2_wcode_uniqueness_1263`` reads source: half of what
    is asserted here is inside an f-string help text and inside a ``pop`` loop,
    neither of which is an object at runtime.
    """
    from sglang.srt.weg2 import launcher as ln

    src = inspect.getsource(ln)
    assert "--weg2-shadow-hop-bound-ms" in src
    assert "ns.weg2_shadow_hop_bound_ms" in src
    assert "weight_exchange_shadow.ENV_HOP_BOUND_MS" in src
    assert ln.prepare_shadow_env(lambda _s: None, "b", "ring") == {}, \
        "the ring arm must publish nothing at all"


# --- W63 is a new, free code ----------------------------------------------

def test_w63_is_the_next_free_code_and_names_one_exception():
    assert sh.RANK_LOCAL_SKIP_MARKER == "W63 Weg2XchgShadowRankLocalSkip"
    assert sh.Weg2XchgShadowRankLocalSkip.__name__ in sh.RANK_LOCAL_SKIP_MARKER
    # The TIME term reuses W61 rather than taking a code of its own: one code
    # for one class of event ("the shadow cannot afford this leg"), so a census
    # of self-refusals cannot read low by exactly the time-refused ones.
    assert sh.UNAFFORDABLE_MARKER in sh.hop_refusal_message(
        card="u", priced_ms=1.0, bound_ms=0.5, batches=2, slot_mib=32.0, leg=0)
