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
from sglang.srt.weg2 import xchg_bounce as xb

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
    """W68 by name, and the refusal lands before a thread exists."""
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
    W71 has to refuse later -- so it clamps and says ``fits=no``.
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


def _vote_rows(region, rows, *, leg, vote=True, classes_hash=0, need_mib=0,
               plan_digest=0, piece_digest=0):
    for row in rows:
        sh.write_shadow_vote(region, row, leg=leg, vote=vote,
                             classes_hash=classes_hash, need_mib=need_mib,
                             plan_digest=plan_digest, piece_digest=piece_digest)


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
    ``WEG2_GROUP_FENCE_BUDGET_S`` and then raise W69 -- correct there, fatal
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
    for token in ("W75 Weg2XchgShadowMismatch", "class=in_proj_qkvz",
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

    W68/W69/W70 stop a flip on the authoritative path.  Here the same events
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
    # STEP 7: `exchange` now ARMS, and that is the blocker the HELD S6I order
    # named -- it used to publish nothing, so the exchange could not reach the
    # ranks at all.  Only the DEFAULT arm publishes nothing, which is what
    # keeps a stock boot byte-identical.
    assert launcher.WEIGHT_SOURCE_ARMED == ("exchange", "shadow")
    assert launcher.prepare_xchg_env(
        lambda *_a: None, "b", launcher.WEIGHT_SOURCE_DEFAULT) == {}


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

    A flip abandoned after gate 1 rolls forward by design (W73) and leaves any
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
    reaches the same W75 marker -- logged on a bare inequality.  A batch whose
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
    # descriptors, which is what keeps the W68 slot_bytes check a cross-check.
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
                              device=kw.pop("device", 0), card_uuid="u0",
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

def test_the_plan_seam_has_exactly_one_producer_and_it_is_the_derivation():
    """S5c: ``build_plan`` HAS a product caller now, and there is exactly one.

    This was ``test_the_plan_seam_has_no_producer`` and asserted the opposite,
    which is precisely why it was written as a DENOMINATOR test: the W79
    docstring claimed an absence and an absence nobody re-checks is the one
    that rots.  The absence is gone; the test does not.  What it now pins is
    the thing that would rot next -- a SECOND derivation growing somewhere
    else, which is the Zweitbuchhaltung UPSTREAM-MINIMAL refuses and the exact
    shape that made ``shadow_armed`` re-spell its own predicate one round ago.

    ONE caller, in ``weight_exchange_shadow.derive_leg_plan``.  A new call site
    anywhere else turns this red, and the reader then has to justify the second
    derivation rather than discover it later as two ranks disagreeing.
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
    # #1330 B4n: TWO CALL SITES NOW, AND THE GUARD IS STRONGER FOR IT.
    #
    # This test's own docstring sets the bar: a second call site "turns this
    # red, and the reader then has to justify the second derivation rather
    # than discover it later as two ranks disagreeing." Here is the
    # justification, and it is the opposite of Zweitbuchhaltung.
    #
    # They are not two derivations of one fact. They are two PRODUCERS
    # SELECTED BY THE ARM, and exactly one can run in a boot:
    #   * `exchange_armed()` -> the MANIFEST JOIN (xchg_manifest.plan_from_join)
    #   * every other arm     -> the rank-local derivation (derive_leg_plan)
    # `_weg2_shadow_plan` branches on that predicate and NEVER falls back from
    # the join to the derivation -- a missing peer manifest is a named refusal
    # carrying the expected file path, pinned by
    # `test_no_join_path_can_reach_derive_leg_plan`.
    #
    # And the second producer exists to REMOVE the very hazard this test
    # names. "Two ranks disagreeing" is what the rank-local derivation
    # PRODUCES: it builds the on-card diagonal from this rank's own model
    # (shard_axis=REPLICATED, both GroupLayouts tp_size=1) and asks a rank for
    # the peer's pointer, which is measured as src_resolved=0/N on all 24 legs
    # of weg2xsn20. The join reads what every rank WROTE, so the six ranks
    # cannot disagree by construction.
    #
    # What the guard pins now: exactly these TWO, by file and function, so a
    # THIRD still turns it red; and that the plan line says WHICH answered
    # (`facts.source`), so the two can never be confused in a log.
    assert len(hits) == 2, hits
    by_file = sorted(h.rsplit(":", 1)[0] for h in hits)
    assert by_file == [
        "python/sglang/srt/weg2/weight_exchange_shadow.py",
        "python/sglang/srt/weg2/xchg_manifest.py",
    ], hits
    assert any(h.endswith("weight_exchange_shadow.py:"
                          + str(_line_of("wx.build_plan(inventory, src, dst")))
               for h in hits), hits

    # THE TWO PROVENANCES MUST DIFFER, or a reader cannot tell which producer
    # answered and the "one of the two ran" claim is unverifiable in a log.
    from sglang.srt.weg2 import xchg_manifest as _xm

    assert _xm.JOIN_PLAN_SOURCE != sh.PLAN_SOURCE
    assert "manifest" in _xm.JOIN_PLAN_SOURCE
    assert "walk_live_tensors" in sh.PLAN_SOURCE
    # The module-level seam is still EMPTY in the product: the derivation is
    # passed as an object through ``run_leg_hook(plan=...)``, not installed as
    # a global by a rank.
    assert sh.plan_for_leg("P->D", 0, 0) == ()


def _line_of(needle: str) -> int:
    src = inspect.getsource(sh).splitlines()
    for i, line in enumerate(src, 1):
        if needle in line:
            return i
    raise AssertionError(needle)


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
    # THE REASON IS THE DERIVATION'S OWN WORD NOW, not one fixed sentence
    # about an absence that no longer exists (S5c).  With no reason handed
    # down the line says so rather than naming a cause it does not have.
    assert any("no derivation was handed to this leg" in ln for ln in lines)


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
    """DANGER 4.  Every local skip is W79 with its reason, never a bare return.

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
    """The W72/W77 two-arm shape, third instance: automatic degrades, explicit raises."""
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
    """W77 ``scope=hop``: the leg is refused BEFORE a wall is spent.

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


# --- W78 at leg start (item 3) --------------------------------------------

def test_a_stale_semaphore_stops_the_shadow_and_never_the_flip(region, boot,
                                                               no_active_leg):
    """W78 gets its caller, and the caller COUNTS it.

    ``verify_sem_arm`` refuses; on the authoritative path (S6's RPC preamble)
    that refusal stops a flip.  Here the same event may only stop the SHADOW,
    so the hook catches it, prints it, and reports ``reason=w78-stale`` with
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
        assert result.reason == "w78-stale"
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

class _NotAnException(BaseException):
    """A BaseException that is NOT an Exception, and not one pytest acts on.

    ``KeyboardInterrupt`` is the obvious member of this class and it is the
    WRONG probe: pytest treats it as a session abort, so a hook that stopped
    catching ``BaseException`` ended the run with ten tests never collected and
    a junit that said ``failures=0``.  Measured this round -- mutant C read
    ``VERDICT: OK`` at ``tests=80`` against a ``tests=90`` baseline, i.e. the
    kill was visible only in the DENOMINATOR.  A gate whose red is a smaller
    test count is not a gate.
    """


def test_no_hook_failure_can_reach_the_leg(no_active_leg):
    """DANGER 3, enumerated over the failure classes the hook can meet."""
    for provider in (
        lambda _d, _lg, _r: (_ for _ in ()).throw(RuntimeError("plan")),
        lambda _d, _lg, _r: (_ for _ in ()).throw(MemoryError("oom")),
        lambda _d, _lg, _r: (_ for _ in ()).throw(_NotAnException()),
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
    assert "self._weg2_shadow_source_leg(recv_req)" in src
    assert "self._weg2_shadow_destination_leg(" in src
    i_src = src.index("self._weg2_shadow_source_leg(recv_req")
    # S5b FIX, refuter must_fix 4: the leg's own clock starts BEFORE the hook,
    # or weg2_leg_ms -- the wall the sb5f flip band is read from and the wall
    # that gates a co-located rank's C14 credit -- excludes the whole observer.
    assert src.index("weg2_leg_t0 = time.perf_counter()") < i_src, \
        "the leg's own instrument cannot see what the source hook cost it"
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
    (``verdict=MISMATCH``), the W75 marker must be on the log, and nothing may
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
    # S5b FIX, refuter must_fix 2: the reserve is the STILL-UNMAPPED tags read
    # where the term is consumed, not the weights image read before a resume
    # that has already happened by the time this hook runs.
    assert "reserve_bytes=sum(self._weg2_tag_bytes(t)" in src
    assert "for t in pending_tags)" in src
    assert "int(sum(int(v) for v in (tag_bytes or {}).values()))" not in src, \
        "the destination still subtracts an image the resume already mapped"


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
    assert ln.prepare_xchg_env(lambda _s: None, "b", "ring") == {}, \
        "the ring arm must publish nothing at all"


# --- W79 is a new, free code ----------------------------------------------

def test_w79_is_the_next_free_code_and_names_one_exception():
    assert sh.RANK_LOCAL_SKIP_MARKER == "W79 Weg2XchgShadowRankLocalSkip"
    assert sh.Weg2XchgShadowRankLocalSkip.__name__ in sh.RANK_LOCAL_SKIP_MARKER
    # The TIME term reuses W77 rather than taking a code of its own: one code
    # for one class of event ("the shadow cannot afford this leg"), so a census
    # of self-refusals cannot read low by exactly the time-refused ones.
    assert sh.UNAFFORDABLE_MARKER in sh.hop_refusal_message(
        card="u", priced_ms=1.0, bound_ms=0.5, batches=2, slot_mib=32.0, leg=0)


# ===========================================================================
# S5b FIX -- THE SIX MUST_FIX OF THE S5b REFUTER.
#
# DANGER DIRECTION, and it is not the same one as the round above.  The hooks
# already refuse to raise, refuse to write into the live arena and refuse to
# outlive their leg.  What the refuter found is the class BELOW that: a hook
# that is harmless per statement and RUINOUS PER FLIP --
#
#   1. a rendezvous the two placements can never satisfy, whose expiry is paid
#      every flip, on the critical path of the credit a co-located rank waits
#      on (a deadlock broken only by a budget is still a deadlock);
#   2. a priced term consumed AFTER the demand it reserves for was paid, which
#      does not under-price -- it refuses everything, forever, silently;
#   3. an observer that allocates on another rank's card and leaves the
#      AUTHORITATIVE thread's CUDA device where it put it;
#   4. one enforced bound (60 ms) in front of 35 s of unbounded waiting;
#   5. refusals that publish nothing, so every peer pays a full budget
#      discovering a row that was decided milliseconds ago;
#   6. a compare with no summer -- NOT-RUN for two reasons while the record
#      names one.
# ===========================================================================


def _wu():
    from sglang.srt.managers.scheduler_components import weight_updater as wu

    return wu


def _spy_transport(monkeypatch) -> dict:
    """Capture ``shadow_transport``'s kwargs and stop the hook there."""
    seen: dict = {}

    def spy(**kw):
        seen.update(kw)
        raise RuntimeError("stop here")

    monkeypatch.setattr(sh, "shadow_transport", spy)
    return seen


def _wu_source(name: str) -> str:
    """The SOURCE of one ``weight_updater`` method.  Wiring is pinned by text
    here for the reason ``test_the_two_call_sites_are_where_the_bytes_are``
    states: this class cannot be constructed without a model runner, a torch
    process group and a device, so its wiring has no runtime object."""
    return inspect.getsource(
        getattr(_wu().SchedulerWeightUpdaterManager, name))


def _wu_ast(name: str):
    """The same method as an ``ast.FunctionDef``.

    ``textwrap.dedent`` and NOT ``inspect.cleandoc``: cleandoc leaves the FIRST
    line unindented and strips the rest, so a method whose signature wraps
    parses as an empty ``def`` followed by a docstring at column 4 -- an
    ``IndentationError`` that reads exactly like a broken product.  Found by
    this round's own run (2 failures, both this line).
    """
    import ast as _ast
    import textwrap

    return _ast.parse(textwrap.dedent(_wu_source(name))).body[0]


# --- must_fix 1: the rendezvous the placements can satisfy -----------------

def test_the_source_hook_does_not_wait_for_rows_the_flip_has_not_reached(region):
    """MUST_FIX 1.  The gate waits for the rows the hook MAY expect.

    The source hook runs on the SLEEPING group before its ``pause``; the
    destination runs on the WAKING group after its ``resume``.  On a co-located
    card the waking rank is fenced on the C14 credit the sleeper publishes
    inside that pause loop -- i.e. AFTER its own hook.  A source that waits for
    all six rows waits for three that cannot be written until it stops waiting.
    """
    _vote_rows(region, [1, 2], leg=1, vote=True, classes_hash=7)
    started = time.monotonic()
    verdict = sh.shadow_gate(region, 0, leg=1, vote=True, classes_hash=7,
                             need_mib=0, log=lambda _s: None,
                             expect_rows=(0, 1, 2), budget_s=5.0)
    assert verdict.run is True, verdict.reason
    assert verdict.joined == 3 and verdict.expected == 3
    assert time.monotonic() - started < 1.0, \
        "it waited for a row it was not entitled to expect"
    assert verdict.waited_s < 1.0
    assert "joined=3/3" in verdict.line(leg=1, epoch="e.1")


def test_the_same_rendezvous_across_all_six_rows_expires_every_time(region):
    """THE CAN-FAIL CONTROL for the test above, and the defect itself.

    This is what the shipping shape did on every sleeping rank of every flip:
    the three waking rows are not there, the budget is spent in full, and the
    shadow never ran.  If this ever passes, the fix above is measuring nothing.
    """
    _vote_rows(region, [1, 2], leg=1, vote=True, classes_hash=7)
    verdict = sh.shadow_gate(region, 0, leg=1, vote=True, classes_hash=7,
                             need_mib=0, log=lambda _s: None,
                             expect_rows=None, budget_s=0.05)
    assert verdict.run is False
    assert verdict.expected == xr.N_RANKS
    assert set(verdict.refusers) == {3, 4, 5}
    assert verdict.waited_s >= 0.05, "the expiry was not actually paid"
    assert "joined=3/6" in verdict.line(leg=1, epoch="e.1")


def test_the_two_hooks_expect_different_rows_and_the_adapter_says_which():
    """MUST_FIX 1, the producer.  The adapter is the only thing that knows the
    GROUP, so it is the only thing that can answer this."""
    rows = _wu().SchedulerWeightUpdaterManager._weg2_shadow_gate_rows
    assert rows(None, "source", "P", leg=1) == (0, 1, 2)
    assert rows(None, "source", "D", leg=1) == (3, 4, 5)
    # The destination runs LAST in the flip, so from leg 2 on the source rows
    # are already sealed with this epoch_hash: all six, for free.
    #
    # #1337: NOT ON LEG 1. There is no earlier instant on the first flip, so a
    # leg-1 destination that expected six waited the full 5.0 s gate budget for
    # rows nobody can write (XSN12: 4.953/4.956/4.960 s, then ran=no). On leg 1
    # it expects its OWN rows; test_weg2_shadow_gate_rows_leg1_1337 owns that.
    assert rows(None, "destination", "P", leg=1) == (0, 1, 2)
    assert rows(None, "destination", "P", leg=2) is None
    assert rows(None, "destination", "D", leg=1) == (3, 4, 5)
    assert rows(None, "destination", "D", leg=2) is None
    # MUTANT: an unknown group must not invent a group's rows.
    assert rows(None, "source", "?", leg=1) is None


def test_the_gate_rows_reach_the_transport_and_are_not_dropped(
        monkeypatch, region, sems, no_active_leg):
    """MUTANT: ``ShadowLegInputs.gate_rows`` exists and nothing forwards it."""
    seen = _spy_transport(monkeypatch)
    sh.run_leg_hook(_inputs(sh.HOOK_SOURCE, gate_rows=(0, 1, 2)),
                    log=lambda _s: None, descs=[_diag(0, 4096)],
                    region=region, sems=sems, ops=object(), armed=True)
    assert seen.get("gate_rows") == (0, 1, 2)


# --- must_fix 2: a term consumed after its demand was paid -----------------

def test_a_demand_the_resume_already_paid_is_not_subtracted_again():
    """MUST_FIX 2, as arithmetic.  This is the defect, not a style point.

    ``free_mib`` is read at hook time, i.e. AFTER the wake leg's resume mapped
    the weights image, so the image is already out of the free column.  The
    shipping shape subtracted it a SECOND time -- a per-rank image of order
    9-13 GiB against a free column the VRAM corridor law holds at 819-1229 MiB
    under load.  ``affordable`` was False by construction: every rank voted NO
    on every leg and no boot could ever have shown otherwise.
    """
    image_mib = 11 * 1024
    priced = sh.price_shadow("u0", 64 * sh.MIB, free_mib=4096,
                             reserve_bytes=image_mib * sh.MIB)
    assert priced.affordable is False, "the defect is not reproduced"
    # The demand that IS still unmapped when the hook runs (the rest of this
    # rpc's tags) is a far smaller number, and it is the one with a producer.
    honest = sh.price_shadow("u0", 64 * sh.MIB, free_mib=4096,
                             reserve_bytes=512 * sh.MIB)
    assert honest.affordable is True
    assert "resume_reserve_mib=512" in honest.line()


def test_the_reserve_is_read_from_the_tags_the_resume_has_not_reached():
    """MUST_FIX 2, the producer, pinned by source: the pending tags of THIS
    rpc, read from the saver at the instant the term is consumed."""
    src = inspect.getsource(_wu().SchedulerWeightUpdaterManager)
    assert "pending_tags = [" in src
    assert "and not is_weights_family_tag(t)" in src
    i_reserve = src.index("reserve_bytes=sum(self._weg2_tag_bytes(t)")
    i_pending = src.index("pending_tags = [")
    assert i_pending < i_reserve


# --- must_fix 3: the device -----------------------------------------------

def test_the_hook_runs_on_this_ranks_own_device_and_never_a_hardcoded_zero():
    """MUST_FIX 3.  The launcher hands every rank ALL THREE cards, so a rank's
    device is its ordinal; ``device=0`` allocated on another rank's card and
    priced it against this one's free column and uuid."""
    src = _wu_source("_weg2_shadow_hook")
    assert "device=int(device)" in src
    assert "device=0," not in src, "the hook is back on a hardcoded card"
    assert "self._weg2_device_index()" in src
    idx = _wu_source("_weg2_device_index")
    assert "torch.cuda.current_device()" in idx
    assert "return -1" in idx


def test_a_rank_with_no_readable_device_says_so_and_does_not_guess():
    """MUTANT: ``return 0`` on the unreadable path.  An observer that guesses a
    card allocates on somebody else's."""
    src = _wu_source("_weg2_shadow_hook")
    assert "if device < 0:" in src
    assert 'reason="no-device"' in src


def test_the_leg_threads_device_is_put_back_after_the_hook():
    """MUST_FIX 3, second half.  ``CudartDeviceOps.set_device`` is a bare
    ``cudaSetDevice`` with no save/restore, so without this the OBSERVER
    decides which device the authoritative leg continues on."""
    import ast as _ast

    fn = _wu_ast("_weg2_shadow_hook")
    finallies = [n for n in _ast.walk(fn) if isinstance(n, _ast.Try) and n.finalbody]
    assert finallies, "the hook has no finally at all"
    restored = any(
        isinstance(c, _ast.Call) and isinstance(c.func, _ast.Attribute)
        and c.func.attr == "_weg2_restore_device"
        for t in finallies for n in t.finalbody for c in _ast.walk(n))
    assert restored, "the device is restored only on the happy path, or not at all"
    assert "torch.cuda.set_device" in _wu_source("_weg2_restore_device")


# --- must_fix 4: one deadline for every wait ------------------------------

def test_every_wait_the_hook_makes_is_carved_out_of_one_deadline(
        monkeypatch, region, sems, no_active_leg):
    """MUST_FIX 4.  ``hop_bound_ms`` grades the priced HOP and nothing else.

    The gate (5 s) and the transport (30 s) sat outside it, on the leg's
    critical path, so the honest bound on the added wall was the sum of three
    constants in three places and the code enforced one.  Both waits now draw
    from ONE deadline, so their sum can never be paid.
    """
    seen = _spy_transport(monkeypatch)
    sh.run_leg_hook(_inputs(sh.HOOK_SOURCE), log=lambda _s: None,
                    descs=[_diag(0, 4096)], region=region, sems=sems,
                    ops=object(), armed=True, hook_budget_s=0.25,
                    gate_budget_s=5.0, budget_s=30.0)
    assert seen["gate_budget_s"] <= 0.25, seen["gate_budget_s"]
    assert seen["budget_s"] <= 0.25, seen["budget_s"]
    assert seen["gate_budget_s"] + seen["budget_s"] <= 0.5


def test_the_hook_budget_is_one_rendezvous_and_says_so():
    """The rule, not the number: the observer may cost the flip ONE gate
    budget, never a gate plus a transport plus a compare."""
    assert sh.SHADOW_HOOK_BUDGET_S == sh.SHADOW_GATE_BUDGET_S
    assert sh.SHADOW_HOOK_BUDGET_S < sh.SHADOW_TRANSPORT_BUDGET_S


def test_the_line_carries_the_deadline_beside_the_wall(no_active_leg):
    """A bound nobody can read on the line is a bound nothing is graded
    against -- the same argument ``hop_bound_ms`` already carries."""
    result = sh.run_leg_hook(_inputs(sh.HOOK_SOURCE), log=lambda _s: None,
                             descs=(), armed=True, hook_budget_s=2.0)
    assert "hook_budget_ms=2000.000" in result.line()
    assert "budget=ok" in result.line()
    over = sh.ShadowResult(leg=0, epoch="e.0", subset=sh.select_subset([], leg=0),
                           counters=sh.ShadowCounters())
    over.hook_budget_ms, over.shadow_ms = 100.0, 250.0
    assert "budget=OVER" in over.line()


# --- must_fix 5: a refusal that publishes ---------------------------------

def test_a_refusal_after_the_attach_publishes_its_no_so_no_peer_waits(
        region, boot, no_active_leg):
    """MUST_FIX 5.  ``hop_ms`` is priced from THIS card's diagonal, so one card
    can refuse while the other two do not -- exactly the asymmetric case where
    a silent return costs five peers a full gate budget each, inside their own
    flip legs.  W79's own docstring names the hazard; the code only logged it.
    """
    xr.create_semaphores(boot)
    sems = tp.SemSet(boot)
    try:
        lines: list = []
        result = sh.run_leg_hook(
            _inputs(sh.HOOK_SOURCE, row=0), log=lines.append,
            descs=[_diag(0, 400 * sh.MIB)], region=region, sems=sems,
            ops=object(), armed=True, bound_ms=1.0)
    finally:
        sems.close()
        xr.unlink_semaphores(boot)
    assert result.reason == "hop-over-bound"
    row = sh.read_shadow_vote(region, 0)
    assert row["sealed"] == 1, "the NO row is not a signal at all"
    assert row["vote"] == 0, "the refusing rank published nothing"
    assert row["leg"] == 0 and row["epoch_hash"] == region.epoch_hash
    assert any("vote=no" in ln and "reason=hop-over-bound" in ln
               for ln in lines), lines


def test_a_refusal_before_the_attach_has_nowhere_to_publish_and_admits_it(
        no_active_leg):
    """The honest half: ``no-region``/``no-sems``/``no-ops``/``no-plan`` happen
    with no row to write into, and W79 names them instead of the code
    pretending a vote was cast."""
    leg = sh.ShadowLeg(_inputs(sh.HOOK_SOURCE), lambda _s: None)
    assert leg.region is None
    assert sh.publish_no_vote(leg, reason="no-region") is False


# --- must_fix 6: the summer's product producer ----------------------------

def test_the_destination_builds_its_own_summer_when_the_caller_gives_none(
        region, tmp_path, boot, no_active_leg):
    """MUST_FIX 6.  The adapter passed no ``sum_bytes``, so ``compare`` took its
    ``no-summer`` exit and ``run_leg``'s slot checksums were disabled too --
    ``verdict=NOT-RUN`` for TWO independent reasons while the record named one.
    ``make_device_scratch`` and ``device_summer`` had no product caller at all.
    """
    xr.create_semaphores(boot)
    sems_s, sems_d = tp.SemSet(boot), tp.SemSet(boot)
    src_ops = FakeDeviceOps(str(tmp_path / "d"), rank=0)
    dst_ops = FakeDeviceOps(str(tmp_path / "d"), rank=0)
    built: list = []

    def fake_make_summer(ops, device, stripe_bytes, result):
        built.append((device, stripe_bytes))
        return byte_sum(dst_ops), None, None

    try:
        dst_ops.raw_malloc(0, 2 << 20)
        payload = pattern(47, 1600)
        src, ring_dst = dev_ptr(0, 0x10000), dev_ptr(0, 0x60000)
        write(src_ops, src, payload)
        write(dst_ops, ring_dst, payload)
        descs = [flat_desc(0, 0, len(payload), src_ptr=src, dst_ptr=ring_dst,
                           name="model.layers.0.self_attn.qkv_proj.weight")]
        _vote_rows(region, [1, 2, 4, 5], leg=0, vote=True,
                   classes_hash=sh.classes_hash(["qkv_proj"]), need_mib=0)
        previous = sh._make_summer
        sh._make_summer = fake_make_summer
        try:
            thread = threading.Thread(target=lambda: sh.run_leg_hook(
                _inputs(sh.HOOK_SOURCE, row=0, peer_row=3), log=lambda _s: None,
                descs=descs, region=region, sems=sems_s, ops=src_ops,
                armed=True, slot_bytes=SLOT, stripe_bytes=1 << 20,
                budget_s=10.0, hook_budget_s=60.0, oncard_slot_bytes=SLOT))
            thread.start()
            try:
                # NO sum_bytes -- the product's shape before the fix.
                result = sh.run_leg_hook(
                    _inputs(sh.HOOK_DESTINATION, row=3, peer_row=0),
                    log=lambda _s: None, descs=descs, region=region,
                    sems=sems_d, ops=dst_ops, armed=True, slot_bytes=SLOT,
                    stripe_bytes=1 << 20, budget_s=10.0, hook_budget_s=60.0,
                    oncard_slot_bytes=SLOT)
            finally:
                thread.join(60)
        finally:
            sh._make_summer = previous
    finally:
        src_ops.close()
        dst_ops.close()
        sems_s.close()
        sems_d.close()
        xr.unlink_semaphores(boot)
    assert built == [(0, 1 << 20)], built
    assert result.counters.stripes == 1 and result.counters.match == 1
    assert "verdict=MATCH" in result.line()


def test_the_summer_producer_is_the_modules_own_two_functions():
    """MUTANT: ``_make_summer`` returns a summer it invented.  The scratch IS
    the stripe and the sum IS ``uint8_checksum``; a second implementation here
    would be a second definition of the comparison the slice exists to make."""
    src = inspect.getsource(sh._make_summer)
    assert "make_device_scratch(" in src
    assert "device_summer(" in src
    assert "ops.create_stream(" in src
    hook = inspect.getsource(sh.run_leg_hook)
    assert "_make_summer(" in hook
    # AFTER the transport, never before: the scratch is 64 MiB of device
    # memory and the gate is what licenses an allocation.
    assert hook.index("run = shadow_transport(") < hook.index("_make_summer(")


def test_a_summer_that_cannot_be_built_is_not_a_match():
    """An unavailable summer must read as NOT-RUN, never as a pass."""
    class _NoStreams:
        def create_stream(self, _device):
            raise RuntimeError("no libcudart here")

    result = sh.ShadowResult(leg=0, epoch="e.0",
                             subset=sh.select_subset([], leg=0),
                             counters=sh.ShadowCounters())
    assert sh._make_summer(_NoStreams(), 0, sh.STRIPE_BYTES, result) == (
        None, None, None)
    assert any("no-summer-stream" in e for e in result.counters.errors)


def test_the_summers_stream_is_destroyed_before_the_leg_drops_its_ops():
    """``leg.close`` nulls ``ops``; a stream destroyed through a ``None`` is a
    leaked stream on the card the shadow may not disturb."""
    hook = inspect.getsource(sh.run_leg_hook)
    assert hook.index("destroy(summer_stream)") < hook.index(
        "leg.close(owner=leg.token)")


# --- the non-blocking findings the refuter also named ----------------------

def test_the_price_covers_the_buffer_the_transport_actually_allocates():
    """CARRIED FROM S5: ``price_leg`` filtered ZEROFILL out of ``mine_dst``
    while ``shadow_transport`` does not -- it must not, because leaving a
    ZEROFILL descriptor's ORIGINAL ``dst_ptr`` in the leg would have
    ``run_leg`` memset the RING's live destination.  So the buffer was bigger
    than the number that was priced: the unpriced-VRAM class again."""
    zero = wx.XchgDesc(tag="weights_0", src_rank=1, dst_rank=0,
                       param_name="model.layers.0.mlp.gate_proj.bias",
                       kind=wx.ZEROFILL, nbytes=8 * sh.MIB, rows=1,
                       run_bytes=8 * sh.MIB, spitch=0, dpitch=0,
                       src_ptr=0, dst_ptr=0x5000)
    price, _slot = sh.price_leg("u0", [zero], rank=0, is_source=False,
                                oncard_mode=tp.ONCARD_MODE_IPC, free_mib=8192)
    allocated = sh.shadow_layout([d for d in [zero] if d.dst_rank == 0])[1]
    assert allocated > 0
    assert price.dst_mib == -(-allocated // sh.MIB), (
        "the shadow allocates bytes nobody priced")


def test_the_armed_predicate_has_exactly_one_owner(monkeypatch):
    """Two spellings of one decision is the Zweitbuchhaltung shape
    UPSTREAM-MINIMAL refuses; this proves the delegation is real."""
    monkeypatch.setattr(wx, "shadow_armed", lambda: True)
    assert sh.shadow_armed() is True
    monkeypatch.setattr(wx, "shadow_armed", lambda: False)
    assert sh.shadow_armed() is False


def test_the_adapter_still_never_raises_and_that_is_why_signals_are_caught():
    """NAMED REFUSAL, not a fix (S5b refuter, non-blocking finding).

    The adapter catches ``BaseException``, which swallows a ``KeyboardInterrupt``
    or ``SystemExit`` arriving on the scheduler thread during a hook.  Narrowing
    it would put a ``raise`` inside an observer, and
    ``test_the_weight_updater_adapters_catch_everything_and_return`` pins the
    opposite -- the flip may not be aborted by its instrument.  The two rules
    conflict and the flip wins; this test states which, so the next reader
    finds a decision rather than an oversight.
    """
    import ast as _ast

    fn = _wu_ast("_weg2_shadow_hook")
    assert not any(isinstance(n, _ast.Raise) for n in _ast.walk(fn))
    assert any(isinstance(h.type, _ast.Name) and h.type.id == "BaseException"
               for n in _ast.walk(fn) if isinstance(n, _ast.Try)
               for h in n.handlers)


# ===========================================================================
# S5c -- THE DERIVATION.  build_plan gets its product caller.
#
# DANGER DIRECTION, and it is not the one the rounds before had.  Those asked
# "can the observer hurt the flip".  This one asks "can the observer LIE": a
# plan is the instrument's own definition of what it is measuring, so a plan
# that is wrong does not fail -- it reports about something else and calls the
# answer a match.  The four shapes, each with a test and a mutant:
#
#   1. a plan naming a class the ring does not carry on this card;
#   2. a plan whose card is not the one this rank runs on;
#   3. a plan built from a STALE flip-order map (a wave partition belonging to
#      another chunk_count, i.e. to another boot's family);
#   4. a rank building a different plan than its peers -- the one shape that
#      is invisible from inside a single rank and must therefore be caught in
#      the rendezvous, by name, and never by a vote.
# ===========================================================================


class _FakeParam:
    """A tensor-shaped double with the three things the derivation reads.

    Not a torch tensor: ``ParamGeom.of`` goes through ``StorageGeom.of``, which
    reads ``dim``/``stride``/``element_size``/``shape``, and the derivation
    additionally reads ``data_ptr``.  A double makes the GEOMETRY a parameter
    of the test instead of a property of whatever torch does on this box, which
    is what lets the two-sides-disagree case be built at all.
    """

    def __init__(self, rows, cols, *, itemsize=2, ptr=0x100000):
        self.shape = (rows, cols)
        self._stride = (cols, 1)
        self._itemsize = int(itemsize)
        self._ptr = int(ptr)
        # ``walk_live_tensors`` records a dtype and a shape on every
        # ``LiveTensor`` (S5c-fix must_fix 5: the population comes from that
        # producer now).  A double that cannot answer is refused BY NAME --
        # ``population-refused:AttributeError`` -- which is how this line got
        # written, and is itself the proof that the derivation never raises.
        self.dtype = "fake.bfloat16"

    def dim(self):
        return 2

    def stride(self, i=None):
        return self._stride if i is None else self._stride[i]

    def element_size(self):
        return self._itemsize

    def data_ptr(self):
        return self._ptr

    def is_contiguous(self):
        return True

    def numel(self):
        # ``walk_live_tensors`` -> ``_nbytes`` reads this when a double has no
        # ``untyped_storage()`` (S5c-fix must_fix 5: the derivation now asks
        # the ring's three-population producer, which sizes every member).
        return self.shape[0] * self.shape[1]


class _FakeModel:
    """A model with the THREE populations ``walk_live_tensors`` enumerates.

    It used to be "only its ``named_parameters()``, which is all that is read",
    and that sentence was the test-side half of S5c refuter must_fix 5: the
    derivation read one population and the acceptance line's ``undescribed=0``
    read as "nothing was left out".  Buffers under a chunk tag are real bytes
    the ring restores -- the rope ``cos_sin_cache`` is the measured example --
    so the double can carry them and the plan must COUNT them.
    """

    def __init__(self, params, buffers=()):
        self._params = list(params)
        self._buffers = list(buffers)

    def named_parameters(self, *_a, **_k):
        return iter(self._params)

    def named_buffers(self, *_a, **_k):
        return iter(self._buffers)

    def named_modules(self, *_a, **_k):
        return iter((("", self),))


def _sb4_model(*, layers=32, base_ptr=0x10000000, cols=32, buffers=()):
    """Two classes per layer plus the base tag's embedding, the sb4 shape."""
    out = []
    for layer in range(layers):
        out.append((f"model.layers.{layer}.self_attn.qkv_proj.weight",
                    _FakeParam(64, cols, ptr=base_ptr + layer * 0x10000)))
        out.append((f"model.layers.{layer}.mlp.down_proj.weight",
                    _FakeParam(32, cols, ptr=base_ptr + layer * 0x10000 + 0x8000)))
    out.append(("model.embed_tokens.weight",
                _FakeParam(128, cols, ptr=base_ptr + 0x900000)))
    return _FakeModel(out, buffers=buffers)


@pytest.fixture()
def chunked(monkeypatch):
    """The ring's own chunk geometry, armed the way the launcher arms it."""
    from sglang.srt.managers import weg2_memory_saver as ms

    monkeypatch.setenv(ms.WEIGHT_CHUNK_ENV_LAYERS, "8")
    monkeypatch.setenv(ms.WEIGHT_CHUNK_ENV_COUNT, "4")
    return ms


_DEFAULT_MODEL = object()


def _derive(hook="source", *, rank=0, group="P", model=_DEFAULT_MODEL, **kw):
    return sh.derive_leg_plan(
        hook=hook, group=group, peer=("D" if group == "P" else "P"), rank=rank,
        model=_sb4_model() if model is _DEFAULT_MODEL else model, **kw)


# --- the derivation reads producers, never a hand list ---------------------

def test_the_plan_is_derived_from_the_rings_own_producers(chunked):
    """Every fact on the line came from a NAMED producer, and each was asked.

    The can-fail control is the spy: a derivation that hardcoded the tag list
    or the wave partition would still produce a plausible line, and this test
    would pass on it -- unless the producers are observed being called and
    their answers are observed reaching the line.  So both are asserted.
    """
    asked = []

    def geometry():
        asked.append("weight_chunk_geometry")
        return 8, 4

    def family(count):
        asked.append(f"weights_family_tags({count})")
        return chunked.weights_family_tags(count)

    def waves(tags, tag_cards, cards):
        asked.append(f"derive_waves(tags={len(tags)}, map={tag_cards}, "
                     f"cards={tuple(cards)})")
        from sglang.srt.weg2 import weight_exchange as wx
        return wx.derive_waves(tags, tag_cards, cards)

    plan, reason = _derive(chunk_geometry=geometry, family_tags=family,
                           waves_of=waves)
    assert reason == "" and plan is not None, reason
    assert asked[0] == "weight_chunk_geometry"
    assert "weights_family_tags(4)" in asked
    # THE TAG->CARD MAP IS THE EMPTY ONE, and it is a stated deviation rather
    # than an oversight: a rank holds only its OWN stage's layer count, and
    # ``chunk_tag_cards``' own docstring reads an empty map as UNIFORM.
    assert any("map={}" in a for a in asked), asked
    line = plan.line()
    for token in ("WEG2-XCHG-PLAN card=0", "tags=", "classes=", "slots=",
                  "bytes=", "source="):
        assert token in line, (token, line)
    # The producers are NAMED on the line -- a reader who doubts a field greps
    # the name and lands on the function that answered.
    for producer in ("weight_chunk_geometry", "weights_family_tags",
                     "tag_of_parameter_name", "derive_waves", "build_plan",
                     "N_CARDS"):
        assert producer in line, (producer, line)
    assert plan.slots == len(plan.descs) and plan.nbytes > 0


def test_the_plan_carries_the_tags_and_classes_the_ring_gave_this_card(chunked):
    """MUTANT 1's target: the class list is the RING's, not a literal.

    The classes are the tensor classes of the CHUNK tags -- and the base tag's
    classes are deliberately absent, because ``chunk_tag_cards`` states that
    the base tag's bytes are not a layer band and therefore not card-uniform.
    """
    plan, reason = _derive()
    assert reason == ""
    assert plan.classes == ("down_proj", "qkv_proj"), plan.classes
    assert "embed_tokens" not in plan.classes, (
        "the base tag's classes are stage-dependent; a rotation over them "
        "makes the six ranks enumerate different lists and the classes_hash "
        "gate can then never open")
    # The tag set IS the ring's family, and the base tag is planned (its bytes
    # are still moved) even though its classes are not the rotation's unit.
    assert set(plan.tags) == {"weights", "weights_0", "weights_1", "weights_2",
                              "weights_3"}, plan.tags
    assert set(plan.facts.family_tags) == set(plan.tags)


def test_a_class_the_ring_does_not_carry_cannot_enter_the_plan(chunked):
    """DANGER 1.  A live tensor under a family tag this boot does not carry.

    ``weights_9`` is a weights-family tag by the predicate and is NOT in a
    4-chunk family.  Planning it would put a tag in a wave that the ring never
    pauses -- bytes the destination waits for that no leg produces.
    """
    model = _sb4_model()
    model._params.append(("model.layers.99.mlp.down_proj.weight",
                          _FakeParam(8, 8, ptr=0x999000)))

    def tag_of(name, region_tag=""):
        return "weights_9" if "layers.99" in name else \
            ("weights" if "layers." not in name else
             f"weights_{min(int(name.split('layers.')[1].split('.')[0]) // 8, 3)}")

    plan, reason = _derive(model=model, tag_of=tag_of)
    assert plan is None
    assert reason.startswith("tag-not-in-family:"), reason
    assert "weights_9" in reason


def test_a_plan_for_a_card_this_rank_does_not_run_on_is_refused(chunked):
    """DANGER 2.  ``rank`` IS the card (weight_exchange_region's own theorem).

    A rank outside ``range(N_CARDS)`` would silently become a ``stage`` no
    group layout has, ``_blocks_of`` would raise W68 deep inside ``build_plan``
    and the reason would name the plan rather than the card.
    """
    plan, reason = _derive(rank=xr.N_CARDS)
    assert plan is None and reason.startswith("wrong-card:"), reason
    assert f"rank={xr.N_CARDS}" in reason
    # The can-fail control: every legal card derives.
    for card in range(xr.N_CARDS):
        got, why = _derive(rank=card)
        assert got is not None and why == "", (card, why)
        assert got.card == card


def test_a_stale_flip_order_map_is_refused_by_name(chunked):
    """DANGER 3.  A wave partition that belongs to ANOTHER boot's family.

    The flip order map is derived per leg; a rank that kept one across a boot
    whose ``chunk_count`` changed would plan waves over tags this family does
    not have, and ``build_plan``'s own W74 would then refuse for a reason
    ("barren wave tag") that sends the reader to the inventory instead of to
    the map.  Named here, at the map.
    """
    def stale(tags, tag_cards, cards):
        return [["weights_0", "weights_1", "weights_2", "weights_3",
                 "weights_4", "weights_5", "weights_6", "weights_7",
                 "weights"]]

    plan, reason = _derive(waves_of=stale)
    assert plan is None and reason.startswith("stale-wave-map:"), reason
    # A DROPPED tag is the same class of fault and is caught by the same check:
    # a partition is a PERMUTATION of the family or it is not one.
    def short(tags, tag_cards, cards):
        return [[t for t in tags if t != "weights_3"]]

    plan, reason = _derive(waves_of=short)
    assert plan is None and reason.startswith("stale-wave-map:"), reason


def test_a_form_with_no_ring_layout_refuses_by_name_and_not_by_no_plan(
        monkeypatch):
    """The forms this slice does NOT serve, and they say which they are.

    A boot with no chunked weights family -- a P-only boot, a stock boot -- has
    no ring layout to derive from.  Before S5c every boot printed the same
    ``no-plan``; now the reason names the missing FACT, so the census of
    "legs the shadow could not plan" can be read by cause.
    """
    from sglang.srt.managers import weg2_memory_saver as ms

    monkeypatch.delenv(ms.WEIGHT_CHUNK_ENV_LAYERS, raising=False)
    monkeypatch.delenv(ms.WEIGHT_CHUNK_ENV_COUNT, raising=False)
    plan, reason = _derive()
    assert plan is None
    assert reason.startswith("no-ring-layout:"), reason
    assert "weight_chunk_geometry()=(0, 0)" in reason
    # A model that is not there is its own reason, not this one.
    assert _derive(model=None)[1] == "no-model"


def test_the_two_hooks_derive_the_same_bytes_from_opposite_directions(chunked):
    """The source fills ``src_ptr``, the destination fills ``dst_ptr``.

    ``XchgDesc`` states the contract in its own docstring -- pointers are
    ``None`` on the side a rank does not own -- and this is what makes the
    compare read the RING's restored bytes on the destination rather than a
    copy of the shadow's own buffer.
    """
    src, _ = _derive("source", group="P")
    dst, _ = _derive("destination", group="D")
    assert [d.key() for d in src.descs] == [d.key() for d in dst.descs], (
        "the two ends of one card's lane must plan the same GEOMETRY")
    assert all(d.src_ptr is not None and d.dst_ptr is None for d in src.descs)
    assert all(d.dst_ptr is not None and d.src_ptr is None for d in dst.descs)
    assert all(d.src_rank == d.dst_rank == 0 for d in src.descs)


# --- rank-uniformity by construction (danger 4) ---------------------------

def test_the_plan_digest_is_over_group_uniform_facts_only(chunked):
    """DANGER 4.  Two cards, two models, ONE digest -- by construction.

    The two ranks hold DIFFERENT bytes at DIFFERENT addresses (that is what a
    card is), so a digest over descriptors would differ between them and the
    gate could never open.  What must agree is the DERIVATION: the chunk
    geometry, the family, the wave partition, the card vector, the rotation and
    the producer that answered.
    """
    a, _ = _derive(rank=0, model=_sb4_model(base_ptr=0x10000000))
    b, _ = _derive(rank=1, model=_sb4_model(base_ptr=0x77000000))
    assert a.facts.digest == b.facts.digest
    assert a.card_digest == b.card_digest, (
        "same architecture on both cards -> same storage fingerprint")
    assert [d.src_ptr for d in a.descs] != [d.src_ptr for d in b.descs]


def test_a_rank_reading_a_different_boot_configuration_diverges(chunked):
    """The can-fail control for the digest: it MOVES when a fact moves.

    Enumerated over every field, because a digest that ignores one of them is
    a digest that cannot see the divergence that field produces -- and the
    field it would ignore is the one nobody thought to test.
    """
    base, _ = _derive()
    from sglang.srt.weg2 import weight_exchange as wx

    variants = {
        "layers per chunk": dict(chunk_geometry=lambda: (16, 4)),
        "the card vector": dict(n_cards=2),
        "wave partition": dict(waves_of=lambda t, m, c: [
            list(t)[:2], list(t)[2:]]),
    }
    for what, kw in variants.items():
        other, why = _derive(**kw)
        assert other is not None, (what, why)
        assert other.facts.digest != base.facts.digest, what
    assert wx.derive_waves is not None  # the producer this reads is the real one


def test_the_class_rotation_moves_the_card_digest_and_not_the_group_one(chunked):
    """S5c refuter, must_fix 4: the one live-tensor fact is out of the GROUP digest.

    ``classes`` is built from THIS rank's ``named_parameters()``, filtered by
    ``ParamGeom.of`` succeeding on THIS rank's live tensors.  Under P=PP a
    rank's inventory is its own stage's layer band, so on a hybrid layer stack
    a band that lacks one layer type yields a different class set -- and the
    old digest hashed that assertion instead of checking it, which would have
    surfaced as W80 ``scope=group`` under a cause sentence naming a stale chunk
    geometry.  Wrong place, wrong cause, and the reader sent to the wrong file.

    So the class set must move ``card_digest`` (a per-card reading by
    construction) and must NOT move ``plan_digest``.  Nothing is lost: the
    class agreement is already gated by ``classes_hash``, whose own refusal
    sentence says "the ranks chose different class subsets".
    """
    base, _ = _derive()
    fewer, why = _derive(model=_FakeModel([
        ("model.layers.0.mlp.down_proj.weight", _FakeParam(32, 32))]))
    assert fewer is not None, why
    assert fewer.classes != base.classes, "the control: the rotation moved"
    assert fewer.facts.digest == base.facts.digest, (
        "a per-rank fact must not sit in the group digest")
    assert fewer.card_digest != base.card_digest, (
        "and it must still be visible where it belongs")


def test_two_readers_of_the_chunk_geometry_that_disagree_are_refused(chunked):
    """The derivation has TWO readers of the same fact, and they can drift.

    ``weight_chunk_geometry()`` answers "how many chunks does this boot have",
    and ``tag_of_parameter_name`` answers "which chunk is THIS parameter in" --
    and the second reads the environment for itself.  A rank whose two readers
    disagree (a mid-boot environment change, a stale injection) would plan a
    tag the family does not contain, which is the same silent shape as the
    stale wave map.  It is refused by name, and this is where that is proven,
    because it is not obvious from either producer alone.
    """
    plan, reason = _derive(chunk_geometry=lambda: (8, 3))
    assert plan is None
    assert reason.startswith("tag-not-in-family:"), reason
    assert "weights_3" in reason


def test_a_divergent_plan_is_a_named_refusal_and_never_a_vote(region):
    """DANGER 4 at the rendezvous: W80, ``scope=group``, no vote taken.

    The gate is where a divergence between ranks is VISIBLE, and it is the only
    place: a rank cannot see another rank's derivation any other way.  It must
    not be folded into the class-subset disagreement -- that sentence sends the
    reader to the rotation, and the rotation is not where this is.
    """
    for row in (1, 2):
        sh.write_shadow_vote(region, row, leg=0, vote=True, classes_hash=7,
                             need_mib=0, plan_digest=0xAAAA)
    lines = []
    verdict = sh.shadow_gate(region, 0, leg=0, vote=True, classes_hash=7,
                             need_mib=0, log=lines.append, budget_s=0.5,
                             expect_rows=(0, 1, 2), plan_digest=0xBBBB)
    assert verdict.run is False
    assert verdict.reason == "plan-diverged-group"
    assert any(sh.PLAN_DIVERGED_MARKER in ln for ln in lines), lines
    assert any("scope=group" in ln and "field=plan_digest" in ln
               for ln in lines), lines
    # EVERY row is named, not just the odd one out: with two values and three
    # rows there is no odd one out, and picking one decides which side is wrong.
    assert all(f"row={r}" in "".join(lines) for r in (0, 1, 2))


def test_agreeing_plans_do_not_trip_the_new_check(region):
    """The can-fail control: the check is inert when the ranks agree.

    Including the shipping case where NOBODY derived a plan -- a table of
    zeros is uniform, so S3's and S4's callers are byte-unchanged.
    """
    for digest in (0, 0xAAAA):
        for row in (1, 2):
            sh.write_shadow_vote(region, row, leg=3, vote=True, classes_hash=7,
                                 need_mib=0, plan_digest=digest)
        verdict = sh.shadow_gate(region, 0, leg=3, vote=True, classes_hash=7,
                                 need_mib=0, log=lambda _s: None, budget_s=0.5,
                                 expect_rows=(0, 1, 2), plan_digest=digest)
        assert verdict.run is True, (digest, verdict.reason)


def test_the_co_located_pair_is_checked_on_its_card_geometry(region):
    """W80 ``scope=oncard-peer``: the two ends of ONE card's lane.

    A DIFFERENT question from the group one, and the honest answer on this
    rig's P=PP / D=TP form: the two groups do not hold the same bytes on a
    card, and that is a statement about two layouts, not a defect in the
    exchange.  Refused here, before a byte moves -- otherwise it surfaces as a
    W70 byte-count disagreement inside the on-card consumer, after the source
    has filled a bounce and while it waits out its drain.
    """
    for row in range(1, xr.N_RANKS):
        sh.write_shadow_vote(region, row, leg=0, vote=True, classes_hash=7,
                             need_mib=0, plan_digest=0xAAAA,
                             piece_digest=0x1111 if row != 3 else 0x2222)
    lines = []
    verdict = sh.shadow_gate(region, 0, leg=0, vote=True, classes_hash=7,
                             need_mib=0, log=lines.append, budget_s=0.5,
                             plan_digest=0xAAAA, piece_digest=0x1111,
                             peer_row=3)
    assert verdict.run is False
    assert verdict.reason == "plan-diverged-oncard-peer"
    assert any("scope=oncard-peer" in ln and "field=piece_digest" in ln
               for ln in lines), lines
    # The group digest AGREES here -- so this cannot be the same code path as
    # the test above, which is the whole reason for two scopes under one code.
    assert not any("scope=group" in ln for ln in lines)


def test_the_peer_card_check_is_only_asked_where_the_row_is_read(region):
    """It may never be asked on the SOURCE hook.

    The source expects its own group's three rows (S5b must_fix 1) and its
    co-located peer's row is one of the three it does NOT wait for.  Comparing
    a row it never reads would reintroduce exactly the circular wait that fix
    removed.
    """
    for row in (1, 2):
        sh.write_shadow_vote(region, row, leg=0, vote=True, classes_hash=7,
                             need_mib=0, plan_digest=0xAAAA, piece_digest=0x1111)
    verdict = sh.shadow_gate(region, 0, leg=0, vote=True, classes_hash=7,
                             need_mib=0, log=lambda _s: None, budget_s=0.5,
                             expect_rows=(0, 1, 2), plan_digest=0xAAAA,
                             piece_digest=0x9999, peer_row=3)
    assert verdict.run is True, verdict.reason


def test_the_widened_row_carries_both_digests_and_still_fits(region):
    """The row grew from 64 to 128 bytes; the two new words survive a round trip."""
    assert sh.SHADOW_ROW_BYTES == 128
    assert sh.SHADOW_SEAL_OFF + 8 <= sh.SHADOW_ROW_BYTES
    assert sh.SHADOW_AREA_END <= tp.DIR_CAPACITY
    sh.write_shadow_vote(region, 4, leg=9, vote=True, classes_hash=0x1234,
                         need_mib=11, plan_digest=0xDEADBEEF,
                         piece_digest=0xFEEDFACE)
    got = sh.read_shadow_vote(region, 4)
    assert got["sealed"] == 1
    assert got["plan_digest"] == 0xDEADBEEF
    assert got["piece_digest"] == 0xFEEDFACE
    assert got["classes_hash"] == 0x1234 and got["need_mib"] == 11


# --- the hooks reach the gate WITH a plan ---------------------------------

def test_both_hooks_reach_the_gate_with_a_plan(region, boot, tmp_path,
                                               no_active_leg, chunked):
    """ITEM 3: a plan present -> gate -> pricing -> transport, not ``no-plan``.

    The gate is deliberately made to EXPIRE here (the peers never publish), so
    this asserts the leg got PAST the plan step and INTO the rendezvous --
    which is what "the hooks proceed to the gate" means and is falsifiable
    without a second process.
    """
    plan, why = _derive("source", rank=0, group="P")
    assert plan is not None, why
    xr.create_semaphores(boot)
    sems = tp.SemSet(boot)
    ops = FakeDeviceOps(str(tmp_path / "dd"), rank=0)
    try:
        lines = []
        result = sh.run_leg_hook(
            _inputs(sh.HOOK_SOURCE, row=0, peer_row=3), log=lines.append,
            plan=plan, region=region, sems=sems, ops=ops, armed=True,
            gate_budget_s=0.05, hook_budget_s=0.5, slot_bytes=SLOT,
            oncard_slot_bytes=SLOT)
    finally:
        ops.close()
        sems.close()
        xr.unlink_semaphores(boot)
    assert result.reason != "no-plan", result.line()
    assert any(ln.startswith(sh.PLAN_LINE_PREFIX + " card=") for ln in lines)
    assert any(ln.startswith(sh.SHADOW_GATE_LINE_PREFIX) for ln in lines), lines
    assert any(ln.startswith(sh.SHADOW_BUDGET_LINE_PREFIX) for ln in lines)
    # ONE provenance line per leg, not one per descriptor and not one per class.
    assert sum(1 for ln in lines
               if ln.startswith(sh.PLAN_LINE_PREFIX + " card=")) == 1
    assert f"plan_digest={plan.facts.digest:#x}" in result.line()


def test_a_leg_with_no_plan_still_refuses_and_names_the_derivations_reason(
        no_active_leg):
    """The no-plan path SURVIVES, and it now carries a cause.

    The refusal is what serves every form without a ring layout, and it must
    keep returning BEFORE the region is opened -- otherwise a P-only boot pays
    an attach and reports ``no-region`` for a leg that had no plan.
    """
    lines = []
    result = sh.run_leg_hook(
        _inputs(sh.HOOK_DESTINATION,
                plan_reason="no-ring-layout:weight_chunk_geometry()=(0, 0)"),
        log=lines.append, descs=(), armed=True)
    assert result.reason == "no-plan"
    assert any("reason=no-plan" in ln and "no-ring-layout" in ln
               for ln in lines), lines
    assert not any(ln.startswith(sh.SHADOW_GATE_LINE_PREFIX) for ln in lines)


# --- item 4: the compare names class, stripes and bytes -------------------

def test_the_compare_names_the_class_the_stripes_and_the_bytes(
        region, tmp_path, boot, no_active_leg):
    """ITEM 4, for MATCH as well as for MISMATCH.

    W75 names a class only when something went WRONG, so a boot whose shadow
    agreed carried no per-class evidence: ``match=7`` with no way to say which
    seven, over how many bytes.  A compare that looked at nothing satisfies
    ``mismatch=0`` just as well as one that looked at everything, and the S5b
    ticket could grade only the latter.
    """
    xr.create_semaphores(boot)
    sems_s, sems_d = tp.SemSet(boot), tp.SemSet(boot)
    src_ops = FakeDeviceOps(str(tmp_path / "d"), rank=0)
    dst_ops = FakeDeviceOps(str(tmp_path / "d"), rank=0)
    try:
        dst_ops.raw_malloc(0, 2 << 20)
        payload = pattern(61, 1600)
        src, ring_dst = dev_ptr(0, 0x10000), dev_ptr(0, 0x60000)
        write(src_ops, src, payload)
        write(dst_ops, ring_dst, payload)
        descs = [flat_desc(0, 0, len(payload), src_ptr=src, dst_ptr=ring_dst,
                           name="model.layers.0.self_attn.qkv_proj.weight")]
        _vote_rows(region, [1, 2, 4, 5], leg=0, vote=True,
                   classes_hash=sh.classes_hash(["qkv_proj"]), need_mib=0)
        lines = []
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
        compare = [ln for ln in lines if ln.startswith(sh.COMPARE_LINE_PREFIX)]
        assert len(compare) == 1, lines
        for token in ("class=qkv_proj", "stripes=1", f"bytes={len(payload)}",
                      "match=1", "mismatch=0", "verdict=MATCH"):
            assert token in compare[0], (token, compare[0])
        # And the bytes on the line are the PAYLOAD, not the descriptor span or
        # the plan's claim -- the denominator of ``stripes=``.
        assert f"bytes={len(payload)}" in compare[0]
    finally:
        src_ops.close()
        dst_ops.close()
        sems_s.close()
        sems_d.close()
        xr.unlink_semaphores(boot)


# --- item 5: the block across the flip span is bounded AND priced ---------

def test_the_block_on_slot_drain_is_measured_when_the_wait_TIMES_OUT(tmp_path,
                                                                    region,
                                                                    boot):
    """S5c refuter, must_fix 1 AND must_fix 7 -- the defect and the harness bug.

    THE DEFECT, proven on the remote desk before this test existed: the
    accumulation ``stats.drain_wait_s += perf_counter() - _blocked`` sat AFTER
    the call that raises, so the one wait that ever costs the leg anything --
    the one that ran to the budget and threw -- was the one wait never priced.
    A leg that spent 402.0 of its 402.9 ms blocked in ``drain-final`` printed
    ``blocked_ms=0.004``: the microseconds of the ``seq<0`` early return, which
    never blocked at all.  The field this round exists to add read as its own
    opposite on the path it was added for.

    THE OLD TEST COULD NOT SEE IT, and that is the second finding: it asserted
    ``blocked_ms > 0.0`` and ``<= hook_budget_ms`` (3000 ms) against a payload
    whose only accumulating call was the no-op.  4 us and 400 ms both pass.
    An instrument-that-cannot-go-red, in the field the round exists to add --
    so this test grades the MAGNITUDE against the budget that produced it.

    AND IT NO LONGER STARTS A PEERLESS LEG ON THE DOUBLE, which S5b's own
    record forbids by name after a SIGSEGV cost a whole remote run its
    ``junit.xml`` (run ``7f3825bfb5_20260909T084353Z``, VERDICT JUNIT-MISSING):
    the fake's device storage is an ``mmap`` whose base is handed out raw and a
    leg's diagonal thread outlives the fixture.  This drives
    ``run_oncard_producer`` DIRECTLY -- one thread, synchronous, returns before
    teardown -- which is also the tighter test: it names the function whose
    ``finally`` is the fix.
    """
    ops = FakeDeviceOps(str(tmp_path / "blk"), rank=0)
    try:
        payload = pattern(29, 1200)
        src = dev_ptr(0, 0x10000)
        write(ops, src, payload)
        descs = [flat_desc(0, 0, len(payload), src_ptr=src, dst_ptr=0,
                           name="model.layers.0.mlp.down_proj.weight")]
        assert len(tp.batch_descs(descs, SLOT)) == 1, "one batch: only the terminal drain blocks"
        bounce = tp.OnCardBounce(ops, 0, slots=tp.ONCARD_SLOTS, slot_bytes=SLOT)
        stats = tp.OnCardStats(0, "u0", tp.ONCARD_MODE_IPC)
        budget = 0.4
        t0 = time.perf_counter()
        with pytest.raises(xr.Weg2XchgGateTimeout) as excinfo:
            # Row 3 never votes: the terminal drain runs to the budget and
            # raises, which is EXACTLY the path the old accumulation skipped.
            tp.run_oncard_producer(region, ops, ops.create_stream(0), bounce,
                                   row=0, peer_row=3, wave=WAVE, descs=descs,
                                   stats=stats, budget_s=budget)
        wall = time.perf_counter() - t0
        assert "what=drain-final" in str(excinfo.value), str(excinfo.value)
        bounce.close()
    finally:
        ops.close()
    # THE MAGNITUDE, graded against the budget that produced it -- not against
    # zero, and not against a 3 s ceiling that 4 us also satisfies.
    assert stats.drain_wait_s >= 0.8 * budget, (stats.drain_wait_s, budget)
    assert stats.drain_wait_s <= wall, (stats.drain_wait_s, wall)
    # The terminal drain is OUTSIDE ``elapsed_s`` by design, so the field that
    # ``issue_ms`` subtracts must carry it too.
    assert stats.drain_wait_outside_s >= 0.8 * budget, stats.drain_wait_outside_s


def test_the_blocked_wall_reaches_the_shadow_line_from_a_failed_leg(tmp_path,
                                                                   region):
    """The same wall, one seam further: the carry, end to end, no leg started.

    ``run_leg`` puts its partial ``LegResult`` on the exception
    (``weg2_leg_result``) and ``shadow_transport``'s handler reads it.  With
    must_fix 1 unfixed that object carried a zero, so the carry was live and
    the number it carried was wrong -- the reason a mutant that removed the
    carry still died while the product lied.
    """
    stats = tp.OnCardStats(0, "u0", tp.ONCARD_MODE_HOST)
    stats.drain_wait_s = 0.402
    result = tp.LegResult()
    result.oncard = stats
    exc = tp.Weg2XchgGateTimeout("W69 Weg2XchgGateTimeout oncard -- synthetic")
    setattr(exc, "weg2_leg_result", result)
    got = getattr(exc, "weg2_leg_result", None)
    assert got is not None and got.oncard is stats
    assert got.oncard.drain_wait_s * 1e3 == pytest.approx(402.0)


def test_the_issue_remainder_does_not_absorb_the_terminal_drain():
    """The DENOMINATOR of ``issue_ms``, named.

    ``issue_ms`` is a REMAINDER (``elapsed - cross - oncard``) and the
    producer's terminal drain sits OUTSIDE ``oncard.elapsed_s`` by design -- so
    without subtracting it, a wait that can span a whole flip is reported as
    issue overhead and graded against spec 6/S5's ``issue_ms <= 5 % of
    xchg_ms``.  Two fields, because only one of the two blocks is outside.
    """
    stats = tp.OnCardStats(0, "u0", tp.ONCARD_MODE_IPC)
    assert hasattr(stats, "drain_wait_s")
    assert hasattr(stats, "drain_wait_outside_s")
    src = inspect.getsource(sh.shadow_transport)
    assert "drain_wait_outside_s" in src
    i_outside = src.index("outside_ms")
    i_issue = src.index("result.issue_ms =")
    assert i_outside < i_issue, "the remainder must subtract it, not follow it"


# --- the adapter is the product caller, pinned by source ------------------

def test_the_adapter_derives_and_hands_the_plan_down(no_active_leg):
    """The WIRING, pinned by SOURCE the way every other adapter here is.

    An adapter that derived a plan and then dropped it would leave every boot
    on ``no-plan`` while this file's behaviour tests all passed -- the seam is
    exactly where a wiring defect is invisible from both sides.
    """
    src = _wu_source("_weg2_shadow_plan")
    assert "sh.derive_leg_plan(" in src
    assert "weights_region_tag_for" in src, (
        "the region tag must be the one this runner's weights were opened "
        "with, or a drafter plans bytes nobody exchanges")
    hook = _wu_source("_weg2_shadow_hook")
    assert "self._weg2_shadow_plan(" in hook
    assert "plan=plan" in hook, "the derivation must REACH the hook"
    assert "plan_reason=str(plan_reason)" in hook
    # And it still cannot raise into the leg.
    import ast as _ast

    tree = _wu_ast("_weg2_shadow_plan")
    assert not any(isinstance(n, _ast.Raise) for n in _ast.walk(tree))
    assert any(isinstance(n, _ast.ExceptHandler) for n in _ast.walk(tree))


def test_w80_is_the_next_free_code_and_names_one_exception():
    """W80, and it is not folded into the class-subset disagreement.

    W79 was the highest assigned code on this branch; W80 is the first free
    one above it, per ``test_weg2_wcode_uniqueness_1263``'s census rule.
    """
    assert sh.PLAN_DIVERGED_MARKER == "W80 Weg2XchgShadowPlanDiverged"
    assert sh.Weg2XchgShadowPlanDiverged.__name__ in sh.PLAN_DIVERGED_MARKER
    message = sh.plan_divergence_message(
        scope="group", leg=1, epoch="e.1", rows={0: 1, 1: 2},
        field="plan_digest")
    assert message.startswith(sh.PLAN_DIVERGED_MARKER)
    assert "the flip proceeds on the ring" in message


# ===========================================================================
# S5c-FIX -- the seven must_fix and two findings of the S5c refutation.
# Every test below names the defect it was written red against.
# ===========================================================================


def test_the_leg_budget_is_a_deadline_and_not_a_per_wait_ceiling():
    """must_fix 2: ``budget_s`` bounds ONE wait; a leg performs many.

    ``run_leg`` handed its one number to every ``_await_oncard``, and each
    restarts its own clock -- so a source leg's worst case was
    ``(batches + slots + 1) x budget``, while
    ``SHADOW_HOOK_BUDGET_S``'s docstring said in as many words that "the two
    together can never exceed this number no matter how the sub-budgets are
    tuned".  That constant is the one place the user law "never delays the
    leg's own completion beyond a named bound" is meant to be readable, so the
    gap was between the bound and its own statement of itself.

    The fix is a CALLABLE, not a smaller number: a deadline sampled once at
    entry is the identical defect one level up.
    """
    # The clamp itself: what is LEFT wins whenever it is smaller.
    assert tp._wait_budget(5.0, None) == 5.0, "no deadline -> pre-S5c semantics"
    assert tp._wait_budget(5.0, lambda: 0.25) == 0.25
    assert tp._wait_budget(0.1, lambda: 9.0) == 0.1
    assert tp._wait_budget(5.0, lambda: -3.0) == 0.0, "an expired deadline is 0, never negative"
    # And it is threaded, not merely present: every wait of the diagonal takes
    # its ceiling through the clamp rather than from the raw budget.
    for fn in (tp.run_oncard_producer, tp.run_oncard_consumer,
               tp.run_producer_pair, tp.run_consumer_pair, tp.run_leg):
        assert "budget_left" in inspect.signature(fn).parameters, fn.__name__
    src = inspect.getsource(tp.run_oncard_producer)
    assert src.count("_wait_budget(budget, budget_left)") == 2, src
    assert "_await_oncard(region, DIR_ONCARD_CONS_OFF, peer_row,\n" in src
    # THE CAN-FAIL CONTROL: a raw ``budget`` reaching a wait is the defect.
    for fn in (tp.run_oncard_producer, tp.run_oncard_consumer):
        body = inspect.getsource(fn)
        assert ", budget," not in body, (fn.__name__, "a raw per-wait budget")


def test_the_deadline_shrinks_across_successive_waits():
    """The same fix, as behaviour rather than as shape.

    A deadline that is re-evaluated gives the SECOND wait less than the first.
    Sampling it once -- the defect -- gives both the same number.
    """
    deadline = time.perf_counter() + 0.30
    left = lambda: max(0.0, deadline - time.perf_counter())  # noqa: E731
    first = tp._wait_budget(5.0, left)
    time.sleep(0.12)
    second = tp._wait_budget(5.0, left)
    assert second < first, (first, second)
    assert first + second < 2 * 5.0
    time.sleep(0.25)
    assert tp._wait_budget(5.0, left) == 0.0, "past the deadline nothing is granted"


def test_the_hook_hands_the_transport_the_deadline_itself(chunked):
    """must_fix 2 at the shadow seam: the callable reaches ``tp.run_leg``."""
    src = inspect.getsource(sh.run_leg_hook)
    assert "budget_left=left," in src, src
    assert "budget_s=min(float(budget_s), left())" in src
    forwarded = inspect.getsource(sh.shadow_transport)
    assert "budget_s=budget_s, budget_left=budget_left," in forwarded
    assert "budget_left" in inspect.signature(sh.shadow_transport).parameters


# --- must_fix 3: the lane whose peer cannot run -----------------------------


def test_the_product_adapter_declares_its_lane_undrainable():
    """must_fix 3: on ``weight_updater``'s placement the peer is DOWNSTREAM.

    The source hook is upstream of the pause loop; ``credit.publish`` is inside
    it; the co-located waking rank's ``resume`` is fenced (C14) on that credit
    and its destination hook runs after the resume.  So the consumer of this
    bounce cannot exist while the producer blocks on it, and with a real plan
    -- S5c's whole addition -- every armed source leg would fill a bounce,
    block in ``drain-final`` to the budget, vote its PROD row FAILED and take
    the destination down with it: seconds on the critical path of that credit,
    zero bytes ever compared.

    The default is ``True`` because it is the truth for every CONCURRENT caller
    (S6's handler, and every hermetic test that drives both ends at once); the
    adapter is what knows its own placement, so the adapter is what says False.
    """
    assert sh.ShadowLegInputs(
        leg=0, epoch="e", direction="d2h", hook=sh.HOOK_SOURCE, rank=0, row=0,
        peer_row=3, device=0, card_uuid="u").oncard_drainable is True
    hook = _wu_source("_weg2_shadow_hook")
    assert "oncard_drainable=False," in hook, hook
    # And it is stated where a reader looking for the cause will be.
    assert "C14" in hook


def test_an_undrainable_lane_compares_digests_and_moves_no_bytes(region, boot,
                                                                 tmp_path,
                                                                 monkeypatch,
                                                                 no_active_leg,
                                                                 chunked):
    """must_fix 3, and refuter finding 9 answered in the same run.

    The refusal is placed AFTER the gate on purpose: the rendezvous is the only
    part of the shadow that needs no lane, and it is where ``plan_digest`` and
    ``piece_digest`` are compared.  Refusing before it would leave the source
    exporting under a plan its consumer never saw -- finding 9 with the sign
    flipped.  So the gate must have run, the refusal must be on the log by
    name, and ``run_leg`` must never have been entered.

    S6 NARROWED WHERE THIS LINE IS THE RIGHT ANSWER, so this test now pins the
    ARM as well.  A store-and-forward deposit needs no concurrent peer and is
    what an undrainable ``host`` lane now gets instead; on the ``ipc`` arm no
    deposit is possible at all -- an exported bounce is freed with its leg --
    so THAT is where the blameless placement line survives.  The refusal's
    contract (gate ran, digests compared, no bytes moved, ``blocked_ms=0``) is
    unchanged, which is what the assertions below are for.
    """
    monkeypatch.setattr(sh, "resolve_shadow_oncard_mode",
                        lambda: tp.ONCARD_MODE_IPC)
    plan, why = _derive("source", rank=0, group="P")
    assert plan is not None, why
    xr.create_semaphores(boot)
    sems = tp.SemSet(boot)
    ops = FakeDeviceOps(str(tmp_path / "nd"), rank=0)
    # The three rows this source hook expects, all voting yes, with the digests
    # it derived: the gate opens, and only then is the lane refused.
    # S6 fix E: the plan's classes are the ROTATION, not the pin -- the hook
    # narrows to ONE class per leg, so a fixture that votes classes_hash from
    # a pinned all-class subset votes a hash the hook no longer computes and
    # the gate refuses (correctly). Same call the product now makes.
    subset = sh.select_subset(plan.descs, leg=0, rotation=plan.classes)
    _vote_rows(region, [1, 2], leg=0, vote=True, classes_hash=subset.hash,
               need_mib=0, plan_digest=plan.facts.digest)
    entered = []
    try:
        lines = []
        real = tp.run_leg
        tp.run_leg = lambda *a, **k: entered.append(1)  # noqa: E731
        try:
            result = sh.run_leg_hook(
                _inputs(sh.HOOK_SOURCE, row=0, peer_row=3, gate_rows=(0, 1, 2),
                        oncard_drainable=False),
                log=lines.append, plan=plan, region=region, sems=sems, ops=ops,
                armed=True, gate_budget_s=2.0, hook_budget_s=5.0,
                slot_bytes=SLOT, oncard_slot_bytes=SLOT)
        finally:
            tp.run_leg = real
    finally:
        ops.close()
        sems.close()
        xr.unlink_semaphores(boot)
    assert entered == [], "an undrainable lane must not reach the transport"
    assert result.reason == "oncard-not-drainable", result.line()
    assert result.ran is False
    refusal = [ln for ln in lines
               if ln.startswith(sh.ONCARD_NOT_DRAINABLE_PREFIX)]
    assert len(refusal) == 1, lines
    for token in ("hook=source", "row=0", "peer_row=3", "oncard_descs="):
        assert token in refusal[0], (token, refusal[0])
    assert "C14 credit" in refusal[0]
    # THE GATE RAN, which is the whole point of refusing here and not earlier.
    gate = [ln for ln in lines if ln.startswith(sh.SHADOW_GATE_LINE_PREFIX)]
    assert len(gate) == 1 and "run=yes" in gate[0], lines
    # ... and nothing blocked, which is the cost this refusal removes.
    assert result.blocked_ms == 0.0, result.line()


def test_a_drainable_lane_is_unchanged_by_the_refusal(region, boot, tmp_path,
                                                      no_active_leg, chunked):
    """The can-fail control: ``oncard_drainable=True`` still reaches ``run_leg``."""
    plan, why = _derive("source", rank=0, group="P")
    assert plan is not None, why
    xr.create_semaphores(boot)
    sems = tp.SemSet(boot)
    ops = FakeDeviceOps(str(tmp_path / "dr"), rank=0)
    # S6 fix E: the plan's classes are the ROTATION, not the pin -- the hook
    # narrows to ONE class per leg, so a fixture that votes classes_hash from
    # a pinned all-class subset votes a hash the hook no longer computes and
    # the gate refuses (correctly). Same call the product now makes.
    subset = sh.select_subset(plan.descs, leg=0, rotation=plan.classes)
    _vote_rows(region, [1, 2], leg=0, vote=True, classes_hash=subset.hash,
               need_mib=0, plan_digest=plan.facts.digest)
    entered = []
    try:
        real = tp.run_leg
        tp.run_leg = lambda *a, **k: entered.append(1)  # noqa: E731
        try:
            result = sh.run_leg_hook(
                _inputs(sh.HOOK_SOURCE, row=0, peer_row=3, gate_rows=(0, 1, 2)),
                log=[].append, plan=plan, region=region, sems=sems, ops=ops,
                armed=True, gate_budget_s=2.0, hook_budget_s=5.0,
                slot_bytes=SLOT, oncard_slot_bytes=SLOT)
        finally:
            tp.run_leg = real
    finally:
        ops.close()
        sems.close()
        xr.unlink_semaphores(boot)
    assert entered == [1], result.line()
    assert result.reason != "oncard-not-drainable"


# --- must_fix 5: the population has a producer, and the omission is counted --


def test_the_population_comes_from_the_rings_own_walk_and_is_counted(chunked):
    """must_fix 5: ``walk_live_tensors``, not a second enumeration.

    ``undescribed=`` counted ``ParamGeom.of`` refusals only, so a boot printed
    ``undescribed=0`` while an entire population -- buffers, plain module
    attributes -- had never been enumerated at all.  Those bytes carry CHUNK
    tags (the tags this plan claims to cover) and the ring restores them; the
    largest named one is the rope ``cos_sin_cache``, measured by
    ``walk_live_tensors``' own docstring at +300/+181/+210 MiB on D's
    ``weights_0``.
    """
    buf = ("model.layers.0.self_attn.rotary_emb.cos_sin_cache",
           _FakeParam(64, 64))
    plain, _ = _derive()
    withbuf, why = _derive(model=_sb4_model(buffers=[buf]))
    assert withbuf is not None, why
    assert withbuf.population == plain.population + 1
    assert withbuf.unplanned == 1 and plain.unplanned == 0
    assert withbuf.unplanned_bytes > 0
    assert withbuf.planned == plain.planned, "a buffer is not transported"
    line = withbuf.line()
    for token in ("population=", "planned=", "unplanned=1",
                  f"unplanned_bytes={withbuf.unplanned_bytes}"):
        assert token in line, (token, line)
    # The producer is NAMED on the line, so a reader can grep who answered.
    assert "walk_live_tensors" in line
    assert "walk_live_tensors" in sh.PLAN_SOURCE


def test_the_unplanned_population_moves_the_card_digest(chunked):
    """must_fix 5's second half: the digest was over the truncated set too.

    Two co-located ranks with different buffer populations agreed that they
    held the same bytes on this card, because the fingerprint only ever saw
    the parameters.
    """
    a, _ = _derive()
    b, why = _derive(model=_sb4_model(buffers=[
        ("model.layers.0.self_attn.rotary_emb.cos_sin_cache",
         _FakeParam(64, 64))]))
    assert b is not None, why
    assert b.card_digest != a.card_digest
    # ... and it stays a per-CARD reading: the group digest must not move.
    assert b.facts.digest == a.facts.digest


def test_a_model_the_walk_cannot_enumerate_is_refused_by_name(chunked):
    """The derivation never raises -- the walk gets the same treatment as ``build_plan``."""

    class _Hostile:
        def named_parameters(self, *_a, **_k):
            return iter(())

        def named_buffers(self, *_a, **_k):
            raise RuntimeError("no buffers on this shape")

        def named_modules(self, *_a, **_k):
            return iter((("", self),))

    plan, why = _derive(model=_Hostile())
    assert plan is None
    assert why.startswith("population-refused:RuntimeError"), why


# --- must_fix 6: the derivation is inside the deadline ----------------------


def test_the_derivation_prices_itself_and_the_hook_subtracts_it(chunked):
    """must_fix 6: the walk ran on the flip's critical path, unmeasured.

    ``_weg2_shadow_plan`` walks every named parameter, calls a regex tag
    function and ``ParamGeom.of`` per tensor, then ``build_plan`` sorts and
    emits per parameter -- all BEFORE ``run_leg_hook`` starts its clock, so it
    was outside ``shadow_ms``, outside ``hook_budget_ms`` and outside every
    ``budget=ok|OVER`` reading, while ``SHADOW_HOOK_BUDGET_S`` claimed to be
    "the ONE wall a flip leg pays for having an observer".
    """
    plan, why = _derive()
    assert plan is not None, why
    assert plan.derive_ms > 0.0, "the derivation must price itself"
    assert f"derive_ms={plan.derive_ms:.3f}" in plan.line()
    # The hook subtracts it from its own deadline rather than re-timing it.
    src = inspect.getsource(sh.run_leg_hook)
    assert 'derive_ms = float(getattr(plan, "derive_ms", 0.0) or 0.0)' in src
    assert "deadline = started + max(0.0, hook_budget_s - derive_ms / 1e3)" in src
    # And the verdict grades the SUM, not the half the hook happened to time.
    result = sh.ShadowResult(leg=0, epoch="e", subset=sh.select_subset((), leg=0),
                             counters=sh.ShadowCounters(), direction="d2h")
    result.shadow_ms, result.derive_ms, result.hook_budget_ms = 3.0, 4.0, 5.0
    assert result.observer_ms == pytest.approx(7.0)
    assert "budget=OVER" in result.line(), result.line()
    assert "derive_ms=4.000" in result.line()
    result.derive_ms = 0.5
    assert "budget=ok" in result.line(), result.line()


# --- finding 8: the wave map is an assumption and says so -------------------


def test_the_wave_map_is_printed_as_the_assumption_it_is(chunked):
    """finding 8: ``derive_waves`` was handed ``{}`` and the line read as a fact.

    For group P the ring's real flip-order map is NOT empty -- ``launcher``
    builds ``chunk_tag_cards`` and logs it as ``WEG2-FLIP-ORDER MAP group=P``,
    and the front picks the pause order from it per flip.  This derivation
    cannot produce that map (a rank holds only its own PP stage's layer count),
    which is a defensible deviation; printing ``waves=1`` as though it were a
    reading of the ring's own map is not.  A reader with both lines in one boot
    log must be able to see which is which from the lines.
    """
    plan, why = _derive()
    assert plan is not None, why
    assert "wave_map=uniform-assumed" in plan.line(), plan.line()
    assert f"waves={len(plan.facts.waves)}" in plan.line()
    src = inspect.getsource(sh.derive_leg_plan)
    assert "waves_of(family, {}, cards)" in src, "the literal is still the literal"


# --- S6: the store-and-forward deposit, and what it puts back on the log ----
#
# THE FINDING THIS SLICE ANSWERS (SECTION 1ai-S5c-fix): the source hook sits
# inside the pause loop, upstream of the peer's C14-fenced resume, so the
# on-card lane cannot drain across the flip -- `oncard_drainable=False`, six
# `WEG2-XCHG-SHADOW-ONCARD-REFUSED` per flip, and no compare at all.  A deposit
# sized `slots >= batches` needs no drain: the source fills every slot inside
# its own leg and returns, and the destination reads them in its own.


#: A DIAGONAL SLOT THE ROW AREA CAN ADDRESS AT ONE SLOT PER BATCH.  ``SLOT``
#: is 4 KiB -- deliberately tiny, so the cross lane's batching is exercised by
#: kilobytes -- and this fixture's diagonal cuts 48 batches at that size, which
#: is six times ``ONCARD_SLOTS_MAX``.  A deposit of 48 slots is a REAL refusal
#: (``batches-exceed-slots-max``) and it is tested as one below; a test that
#: wants to reach the funded path must therefore name a slot at which the
#: deposit fits, exactly as ``plan_oncard_slot_bytes`` raises the size itself
#: when it is allowed to choose.
DEPOSIT_SLOT = 64 * 1024


def _ledger_budget() -> int:

    return int(xb.staging_bytes_per_card(tp.ONCARD_SLOT_BYTES_MAX))


def _host_arm(monkeypatch) -> None:
    """The deposit is a ``host``-arm shape; ``resolve_shadow_oncard_mode``
    defaults to ``ipc``, where an exported bounce cannot outlive its leg."""
    monkeypatch.setattr(sh, "resolve_shadow_oncard_mode",
                        lambda: tp.ONCARD_MODE_HOST)


def test_an_unfunded_deposit_is_refused_by_name_and_moves_no_bytes(
        region, boot, tmp_path, no_active_leg, chunked, monkeypatch):
    """S6 W81: pinned host bytes nobody charged for are not a risk to accept.

    ``host_bounce_budget_bytes=0`` is "no ledger answer reached this rank", and
    it refuses exactly like a budget that is too small: an absent measurement
    never becomes a quiet zero, and the reap mark is a HARD bound
    (``host-schwelle-nie-uebertreten``), never a margin to spend.

    THE CONTRACT IS THE OTHER REFUSAL'S: the gate rendezvous has run, the two
    digests HAVE been compared, ``run_leg`` was never entered and nothing
    blocked.  What differs is the claim -- a W-code, because a deposit that was
    asked for and not funded is a configuration that is wrong, where the
    placement line is nobody's fault.
    """
    _host_arm(monkeypatch)
    plan, why = _derive("source", rank=0, group="P")
    assert plan is not None, why
    xr.create_semaphores(boot)
    sems = tp.SemSet(boot)
    ops = FakeDeviceOps(str(tmp_path / "nf"), rank=0)
    # S6 fix E: the plan's classes are the ROTATION, not the pin -- the hook
    # narrows to ONE class per leg, so a fixture that votes classes_hash from
    # a pinned all-class subset votes a hash the hook no longer computes and
    # the gate refuses (correctly). Same call the product now makes.
    subset = sh.select_subset(plan.descs, leg=0, rotation=plan.classes)
    _vote_rows(region, [1, 2], leg=0, vote=True, classes_hash=subset.hash,
               need_mib=0, plan_digest=plan.facts.digest)
    entered = []
    try:
        lines = []
        real = tp.run_leg
        tp.run_leg = lambda *a, **k: entered.append(1)  # noqa: E731
        try:
            result = sh.run_leg_hook(
                _inputs(sh.HOOK_SOURCE, row=0, peer_row=3, gate_rows=(0, 1, 2),
                        oncard_drainable=False, host_bounce_budget_bytes=0),
                log=lines.append, plan=plan, region=region, sems=sems, ops=ops,
                armed=True, gate_budget_s=2.0, hook_budget_s=5.0,
                slot_bytes=SLOT, oncard_slot_bytes=DEPOSIT_SLOT)
        finally:
            tp.run_leg = real
    finally:
        ops.close()
        sems.close()
        xr.unlink_semaphores(boot)
    assert entered == [], "an unfunded deposit must not reach the transport"
    assert result.reason == "deposit-unfundable", result.line()
    assert result.ran is False
    refusal = [ln for ln in lines if ln.startswith("W81 Weg2XchgDepositUnfundable")]
    assert len(refusal) == 1, lines
    for token in ("hook=source", "row=0", "peer_row=3",
                  f"reason={tp.DEPOSIT_REASON_UNFUNDED}",
                  "host_budget_mib=0", "oncard_batches=", "oncard_slots="):
        assert token in refusal[0], (token, refusal[0])
    # The placement line is NOT what fired: the deposit is possible here, it is
    # only unfunded, and the two refusals must never be read as one event.
    assert not [ln for ln in lines
                if ln.startswith(sh.ONCARD_NOT_DRAINABLE_PREFIX)], lines
    gate = [ln for ln in lines if ln.startswith(sh.SHADOW_GATE_LINE_PREFIX)]
    assert len(gate) == 1 and "run=yes" in gate[0], lines
    assert result.blocked_ms == 0.0, result.line()


def test_a_funded_deposit_runs_the_lane_with_no_concurrent_peer(
        region, boot, tmp_path, no_active_leg, chunked, monkeypatch):
    """S6: THE MIRROR of the refusal, and the whole product effect of S6.

    Same placement, same ``oncard_drainable=False``, same hook -- and now the
    transport IS entered, with ``oncard_store_forward=True`` and one slot per
    batch, which is the property that makes the source's every drain wait a
    negative-``seq`` early return.  The recorder stands in for the transport so
    this stays a wiring proof: the bytes themselves are proven in
    ``test_a_deposit_outlives_its_leg_and_is_read_after_the_source_is_gone``,
    against the real batcher.
    """
    _host_arm(monkeypatch)
    plan, why = _derive("source", rank=0, group="P")
    assert plan is not None, why
    xr.create_semaphores(boot)
    sems = tp.SemSet(boot)
    ops = FakeDeviceOps(str(tmp_path / "fd"), rank=0)
    # S6 fix E: the plan's classes are the ROTATION, not the pin -- the hook
    # narrows to ONE class per leg, so a fixture that votes classes_hash from
    # a pinned all-class subset votes a hash the hook no longer computes and
    # the gate refuses (correctly). Same call the product now makes.
    subset = sh.select_subset(plan.descs, leg=0, rotation=plan.classes)
    _vote_rows(region, [1, 2], leg=0, vote=True, classes_hash=subset.hash,
               need_mib=0, plan_digest=plan.facts.digest)
    seen = {}
    try:
        lines = []
        real = tp.run_leg

        def recorder(*a, **k):
            seen.update(k)
            return tp.LegResult()

        tp.run_leg = recorder
        try:
            result = sh.run_leg_hook(
                _inputs(sh.HOOK_SOURCE, row=0, peer_row=3, gate_rows=(0, 1, 2),
                        oncard_drainable=False,
                        host_bounce_budget_bytes=_ledger_budget()),
                log=lines.append, plan=plan, region=region, sems=sems, ops=ops,
                armed=True, gate_budget_s=2.0, hook_budget_s=5.0,
                slot_bytes=SLOT, oncard_slot_bytes=DEPOSIT_SLOT)
        finally:
            tp.run_leg = real
    finally:
        ops.close()
        sems.close()
        xr.unlink_semaphores(boot)
    assert seen, "a funded deposit MUST reach the transport"
    assert seen["oncard_store_forward"] is True, seen.get("oncard_store_forward")
    batches = sh.oncard_lane_batches(subset.descs, 0, seen["oncard_slot_bytes"])
    assert batches > 0, "this test proves nothing on an empty diagonal"
    assert seen["oncard_slots"] >= batches, (seen["oncard_slots"], batches)
    assert seen["oncard_slots"] != tp.ONCARD_SLOTS or batches == tp.ONCARD_SLOTS
    assert not [ln for ln in lines
                if ln.startswith(sh.ONCARD_NOT_DRAINABLE_PREFIX)
                or ln.startswith("W81 ")], lines
    assert result.blocked_ms == 0.0, result.line()
    # The deposit's shape is on the shadow's own line, with its provenance.
    line = result.line()
    for token in (f"oncard_slots={seen['oncard_slots']}",
                  "oncard_slots_source=store-forward-batches",
                  "oncard_deposit_mib="):
        assert token in line, (token, line)


def test_the_batch_count_the_deposit_is_sized_from_is_the_batchers_own(chunked):
    """S6: ``ceil(bytes / slot)`` is a LOWER bound on the batches, not the count.

    The hop model's denominator and the batcher's output are two different
    numbers whenever the descriptors do not pack flush -- ``batch_descs`` starts
    a new batch when the next piece does not fit.  Sizing the deposit from the
    ceil model would produce ``slots < batches`` from arithmetic alone, which is
    the silent overwrite the whole shape forbids.
    """
    plan, why = _derive("source", rank=0, group="P")
    assert plan is not None, why
    # S6 fix E: the plan's classes are the ROTATION, not the pin -- the hook
    # narrows to ONE class per leg, so a fixture that votes classes_hash from
    # a pinned all-class subset votes a hash the hook no longer computes and
    # the gate refuses (correctly). Same call the product now makes.
    subset = sh.select_subset(plan.descs, leg=0, rotation=plan.classes)
    lane = sh.oncard_lane_descs(subset.descs, 0)
    total = sh.oncard_lane_bytes(subset.descs, 0)
    assert lane and total > 0, "the fixture must have a diagonal"
    # ONE FILTER, three readers -- the bytes, the descriptors and the batches.
    assert sum(int(d.nbytes) for d in lane) == total
    for slot in (SLOT, 4 * SLOT):
        counted = sh.oncard_lane_batches(subset.descs, 0, slot)
        assert counted == len(tp.batch_descs(lane, slot))
        assert counted >= -(-total // slot), (counted, total, slot)


def test_the_adapter_reads_its_deposit_budget_from_the_ledger():
    """S6: the charge and the bound are ONE number, not two copies of it.

    The adapter is the producer for the same reason it produces ``gate_rows``:
    the budget is a property of the BOOT's arm and a rank hook cannot read the
    launcher's ladder.  What it must NOT do is spell the arithmetic again --
    a second copy of ``slots x slot_bytes`` here and in the ledger is how a
    charge and a bound drift apart by one edit.
    """
    hook = _wu_source("_weg2_shadow_hook")
    assert "host_bounce_budget_bytes=self._weg2_shadow_host_budget()" in hook, hook
    reader = _wu_source("_weg2_shadow_host_budget")
    assert "staging_bytes_per_card" in reader, reader
    # AN UNREADABLE LEDGER YIELDS 0, AND 0 REFUSES -- never a guessed budget.
    assert "return 0" in reader, reader
    assert sh.ShadowLegInputs(
        leg=0, epoch="e", direction="d2h", hook=sh.HOOK_SOURCE, rank=0, row=0,
        peer_row=3, device=0, card_uuid="u").host_bounce_budget_bytes == 0


def test_the_deposit_is_a_term_of_the_1269_host_ledger_and_the_arm_line_says_so():
    """S6: the exchange's pinned host carrier, priced where host bytes are priced.

    THE OMISSION WAS SELF-DECLARED: ``oncard_host_path``'s docstring has said
    since S4 that the degrade's bounce is host memory "that spec 0.2's ledger
    term does NOT carry ... the number belongs in the record".  It is a term
    now, in ``charge_terms`` -- the one authority the three consumers read --
    so ``size_store_gib``'s leftover shrinks by exactly the deposit and the
    predicted run peak carries it against the reap mark.

    THE RING ARM IS UNCHANGED, and that is asserted rather than assumed: the
    term is 0.00 there, so every existing boot number is the same number.
    """
    from sglang.srt.weg2 import host_ledger as hl

    per_card = xb.staging_bytes_per_card(tp.ONCARD_SLOT_BYTES_MAX)
    # S6 refuter, must_fix 3: the geometry's CEILING, read from its owner's own
    # name for it.  This asserted the slot FLOOR beside the slot count's
    # ceiling -- a quarter of what the planner may derive -- which made the
    # charge understate and `deposit_refusal_reason` falsely refuse.
    # AMENDMENT 5: SLOTS_PER_PAIR slots of the PUBLISHED slot, not the 8-slot
    # ceiling. The ceiling is retired from the ledger path; the shape maximum
    # is still the transport's and is deliberately 4x this.
    assert per_card == xb.SLOTS_PER_PAIR * tp.ONCARD_SLOT_BYTES_MAX
    assert per_card * 4 == tp.ONCARD_DEPOSIT_BYTES_MAX
    assert xr.N_CARDS * xb.staging_bytes_per_card(tp.ONCARD_SLOT_BYTES_MAX) == xr.N_CARDS * per_card
    ring = hl.price(120 << 30, 60 << 30, 1, 1200,
                    ring_bytes=30 << 30, ring_span1_bytes=10 << 30)
    armed = hl.price(120 << 30, 60 << 30, 1, 1200,
                     ring_bytes=30 << 30, ring_span1_bytes=10 << 30,
                     xchg_bounce_host_bytes=xr.N_CARDS * xb.staging_bytes_per_card(tp.ONCARD_SLOT_BYTES_MAX))
    assert ring.terms["xchg_bounce_gib"] == 0.0
    assert armed.terms["xchg_bounce_gib"] == xr.N_CARDS * xb.staging_bytes_per_card(tp.ONCARD_SLOT_BYTES_MAX) / hl.GIB
    # THE TERM IS SPENT AT BOTH MOMENTS, so the leftover -- and therefore the
    # store -- shrinks by exactly it and by nothing else.
    assert round(ring.run_leftover_gib - armed.run_leftover_gib, 6) == \
        round(armed.terms["xchg_bounce_gib"], 6)
    assert round(ring.launch_leftover_gib - armed.launch_leftover_gib, 6) == \
        round(armed.terms["xchg_bounce_gib"], 6)
    # ... and it is in the ONE authority, not added at the three call sites.
    assert "xchg_bounce_gib" in hl.charge_terms(
        1, 1200, 3, hl.resolve_image_terms(None))
    assert "xchg_bounce_gib" in inspect.getsource(hl._boot_charges_gib)


def test_the_arm_line_names_the_deposit_even_when_it_is_zero(tmp_path):
    """S6: an unarmed boot SAYS the term was priced at zero.

    A term that only appears when it is non-zero leaves a reader unable to tell
    "priced at zero" from "not priced", which is the reading this whole slice
    exists to remove from ``oncard_host_path``'s docstring.
    """
    from sglang.srt.weg2 import host_ledger as hl

    _arm, _store, lines = hl.choose(
        200 << 30, 150 << 30, ring_bytes=20 << 30,
        ring_span1_bytes=8 << 30)
    arms = [ln for ln in lines if ln.startswith("WEG2-HOST-LEDGER ARM ")]
    assert arms, lines
    assert all("xchg_bounce=0.00" in ln for ln in arms), arms

    _term = xr.N_CARDS * xb.staging_bytes_per_card(tp.ONCARD_SLOT_BYTES_MAX)
    _a2, _s2, armed_lines = hl.choose(
        200 << 30, 150 << 30, ring_bytes=20 << 30,
        ring_span1_bytes=8 << 30, xchg_bounce_host_bytes=_term)
    armed_arms = [ln for ln in armed_lines
                  if ln.startswith("WEG2-HOST-LEDGER ARM ")]
    expect = f"xchg_bounce={_term / hl.GIB:.2f}"
    assert all(expect in ln for ln in armed_arms), (expect, armed_arms)


def test_the_launcher_charges_the_deposit_only_on_an_armed_boot():
    """S6: the ARM STRING decides the charge, at the one ledger call site.

    Not inside the ledger: ``WEIGHT_SOURCE_CHOICES`` is the launcher's, and a
    ledger that knew about arm names would be a second reader of a decision
    that already has one.  ``ring`` -- the default and every boot that has run
    -- charges nothing, which is what keeps this slice off the default path.
    """
    from sglang.srt.weg2 import launcher as lc

    # S6-fix must_fix 4: the predicate is its own named producer now, so it
    # is provable without a host to read -- see
    # `test_the_deposit_is_charged_only_on_the_arm_that_can_allocate_it`.
    src = inspect.getsource(lc.choose_host_ledger)
    # #1332 B1b: the kwarg is now fed by `xchg_bounce_terms_for_arm`, which
    # keeps `xchg_bounce_arm_pins_host` as its predicate (asserted below and in
    # the test above).  The property this line defends is that the ledger's
    # `xchg_bounce_host_bytes` comes from THE ARM DECISION and never from a
    # constant -- so it names the producer that now feeds it.
    assert "xchg_bounce_host_bytes=_bounce_charge_bytes" in src, src
    assert "xchg_bounce_terms_for_arm(" in src, src
    predicate = inspect.getsource(lc.xchg_bounce_arm_pins_host)
    # AMENDMENT 5: the predicate is a BOOLEAN and carries no number at all --
    # that is the whole point of the reshape, so the guard asserts the absence
    # of a size rather than the presence of one.
    assert "ONCARD_MODE_HOST" in predicate, predicate
    assert "WEIGHT_SOURCE_DEFAULT" in predicate, predicate
    # THE PREDICATE CARRIES NO SIZE (AMENDMENT 5), asserted on the SIGNATURE
    # and not on the source text: the docstring names the retired
    # `host_ledger.xchg_bounce_bytes` in order to say it is retired, and a
    # text-absence guard reads that epitaph as a live member -- which is
    # e25a88c2a2's own finding, walked into one commit after quoting it.
    # `launcher.py` carries `from __future__ import annotations`, so the
    # annotation arrives as the STRING "bool" -- accepted as such rather than
    # resolved, because resolving it would import the module's namespace just
    # to learn what its own source already says.
    assert inspect.signature(
        lc.xchg_bounce_arm_pins_host).return_annotation in (bool, "bool")
    assert "host_ledger.xchg_bounce_bytes(" not in predicate, predicate
    main_src = inspect.getsource(lc.main)
    assert "weight_source=ns.weg2_weight_source" in main_src, \
        "the arm must reach the ledger call site"


def test_shadow_ms_still_covers_everything_the_hook_spends(region, boot,
                                                           tmp_path,
                                                           no_active_leg,
                                                           chunked, monkeypatch):
    """S6: the deposit's slot writes are INSIDE the graded wall, not beside it.

    ``shadow_ms`` is measured from the first statement of ``run_leg_hook`` to
    its ``finally``, so a leg that now COPIES where it used to WAIT reports the
    same field with different physics rather than a shorter wall and a hidden
    cost.  ``budget=ok|OVER`` keeps grading ``shadow_ms + derive_ms`` against
    the hook budget, which is the one place the user law is readable.
    """
    _host_arm(monkeypatch)
    plan, why = _derive("source", rank=0, group="P")
    assert plan is not None, why
    xr.create_semaphores(boot)
    sems = tp.SemSet(boot)
    ops = FakeDeviceOps(str(tmp_path / "sm"), rank=0)
    # S6 fix E: the plan's classes are the ROTATION, not the pin -- the hook
    # narrows to ONE class per leg, so a fixture that votes classes_hash from
    # a pinned all-class subset votes a hash the hook no longer computes and
    # the gate refuses (correctly). Same call the product now makes.
    subset = sh.select_subset(plan.descs, leg=0, rotation=plan.classes)
    _vote_rows(region, [1, 2], leg=0, vote=True, classes_hash=subset.hash,
               need_mib=0, plan_digest=plan.facts.digest)
    try:
        lines = []
        real = tp.run_leg

        def slow(*a, **k):
            time.sleep(0.05)
            return tp.LegResult()

        tp.run_leg = slow
        try:
            result = sh.run_leg_hook(
                _inputs(sh.HOOK_SOURCE, row=0, peer_row=3, gate_rows=(0, 1, 2),
                        oncard_drainable=False,
                        host_bounce_budget_bytes=_ledger_budget()),
                log=lines.append, plan=plan, region=region, sems=sems, ops=ops,
                armed=True, gate_budget_s=2.0, hook_budget_s=5.0,
                slot_bytes=SLOT, oncard_slot_bytes=DEPOSIT_SLOT)
        finally:
            tp.run_leg = real
    finally:
        ops.close()
        sems.close()
        xr.unlink_semaphores(boot)
    assert result.shadow_ms >= 50.0, result.line()
    assert "budget=ok" in result.line(), result.line()
    assert "hook_budget_ms=5000.000" in result.line(), result.line()


def test_the_two_budget_lines_keep_their_shape_under_the_deposit(region, boot,
                                                                 tmp_path,
                                                                 no_active_leg,
                                                                 chunked):
    """S6: the host term does NOT get a second bookkeeping beside the VRAM one.

    ``ShadowPrice`` is a VRAM affordability object with a VRAM verdict, and the
    host bound lives in ``host_ledger``.  Giving this line a host column and a
    second verdict would be the Zweitbuchhaltung UPSTREAM-MINIMAL refuses --
    two places that can disagree about one number.  So both budget lines keep
    exactly the fields they had, and the deposit is priced where host bytes are
    priced.
    """
    plan, why = _derive("source", rank=0, group="P")
    assert plan is not None, why
    xr.create_semaphores(boot)
    sems = tp.SemSet(boot)
    ops = FakeDeviceOps(str(tmp_path / "bl"), rank=0)
    # S6 fix E: the plan's classes are the ROTATION, not the pin -- the hook
    # narrows to ONE class per leg, so a fixture that votes classes_hash from
    # a pinned all-class subset votes a hash the hook no longer computes and
    # the gate refuses (correctly). Same call the product now makes.
    subset = sh.select_subset(plan.descs, leg=0, rotation=plan.classes)
    _vote_rows(region, [1, 2], leg=0, vote=True, classes_hash=subset.hash,
               need_mib=0, plan_digest=plan.facts.digest)
    try:
        lines = []
        real = tp.run_leg
        tp.run_leg = lambda *a, **k: tp.LegResult()  # noqa: E731
        try:
            sh.run_leg_hook(
                _inputs(sh.HOOK_SOURCE, row=0, peer_row=3, gate_rows=(0, 1, 2),
                        oncard_drainable=False,
                        host_bounce_budget_bytes=_ledger_budget()),
                log=lines.append, plan=plan, region=region, sems=sems, ops=ops,
                armed=True, gate_budget_s=2.0, hook_budget_s=5.0,
                slot_bytes=SLOT, oncard_slot_bytes=DEPOSIT_SLOT)
        finally:
            tp.run_leg = real
    finally:
        ops.close()
        sems.close()
        xr.unlink_semaphores(boot)
    budget = [ln for ln in lines if ln.startswith(sh.SHADOW_BUDGET_LINE_PREFIX)]
    assert len(budget) == 2, budget
    assert "scope=full" in budget[0] and "graded=no" in budget[0], budget
    assert "scope=subset" in budget[1] and "graded=yes" in budget[1], budget
    fields = lambda ln: sorted(t.split("=")[0] for t in ln.split() if "=" in t)  # noqa: E731
    assert fields(budget[0]) == fields(budget[1]), budget
    assert not any("host" in f for f in fields(budget[0])), fields(budget[0])


def test_a_deposit_with_more_batches_than_slots_is_refused_by_name(
        region, boot, tmp_path, monkeypatch, no_active_leg, chunked):
    """S6 W81: the second refusal arm, at a slot the row area cannot cover.

    Pinning ``oncard_slot_bytes`` to this file's 4 KiB cross slot makes this
    fixture's diagonal 48 batches -- six times ``ONCARD_SLOTS_MAX``, which
    sizes the handshake row area ONCE for both processes.  One slot per batch
    is impossible there, and a clamp to 8 would be the silent overwrite the
    deposit exists to forbid, so the leg refuses by name with the two numbers
    on the line.
    """
    _host_arm(monkeypatch)
    plan, why = _derive("source", rank=0, group="P")
    assert plan is not None, why
    xr.create_semaphores(boot)
    sems = tp.SemSet(boot)
    ops = FakeDeviceOps(str(tmp_path / "ov"), rank=0)
    # S6 fix E: the plan's classes are the ROTATION, not the pin -- the hook
    # narrows to ONE class per leg, so a fixture that votes classes_hash from
    # a pinned all-class subset votes a hash the hook no longer computes and
    # the gate refuses (correctly). Same call the product now makes.
    subset = sh.select_subset(plan.descs, leg=0, rotation=plan.classes)
    _vote_rows(region, [1, 2], leg=0, vote=True, classes_hash=subset.hash,
               need_mib=0, plan_digest=plan.facts.digest)
    entered = []
    try:
        lines = []
        real = tp.run_leg
        tp.run_leg = lambda *a, **k: entered.append(1)  # noqa: E731
        try:
            result = sh.run_leg_hook(
                _inputs(sh.HOOK_SOURCE, row=0, peer_row=3, gate_rows=(0, 1, 2),
                        oncard_drainable=False,
                        host_bounce_budget_bytes=_ledger_budget()),
                log=lines.append, plan=plan, region=region, sems=sems, ops=ops,
                armed=True, gate_budget_s=2.0, hook_budget_s=5.0,
                slot_bytes=SLOT, oncard_slot_bytes=SLOT)
        finally:
            tp.run_leg = real
    finally:
        ops.close()
        sems.close()
        xr.unlink_semaphores(boot)
    assert entered == [], "a deposit that cannot fit must not reach the transport"
    assert result.reason == "deposit-unfundable", result.line()
    refusal = [ln for ln in lines if ln.startswith("W81 Weg2XchgDepositUnfundable")]
    assert len(refusal) == 1, lines
    assert f"reason={tp.DEPOSIT_REASON_BATCHES}" in refusal[0], refusal[0]
    assert f"slots_max={tp.ONCARD_SLOTS_MAX}" in refusal[0], refusal[0]
    batches = sh.oncard_lane_batches(subset.descs, 0, SLOT)
    assert batches > tp.ONCARD_SLOTS_MAX, batches
    assert f"oncard_batches={batches}" in refusal[0], (batches, refusal[0])
    # ... and the same diagonal at a slot the row area CAN cover does fit,
    # which is what makes this a refusal about the shape and not about the lane.
    assert sh.oncard_lane_batches(subset.descs, 0, DEPOSIT_SLOT) \
        <= tp.ONCARD_SLOTS_MAX


# ===========================================================================
# S6 FIX -- must_fix 4 (the charge and the allocation share one predicate),
# finding 6 (the rank's budget asks the arm) and finding 8 (the deposit field).
# ===========================================================================


def test_the_deposit_is_charged_only_on_the_arm_that_can_allocate_it():
    """S6 refuter must_fix 4: 0.75 GiB charged on a boot that moves no byte.

    The charge fired on every non-``ring`` boot, but the bounce FILE exists
    only on the ``host`` on-card arm -- on ``ipc`` the verdict is
    ``DEPOSIT_REASON_IPC``, the lane logs the blameless not-drainable line and
    nothing is pinned.  ``SGLANG_WEG2_XCHG_ONCARD`` had NO producer in the
    launcher, so ``ipc`` was the only arm a launched boot could be on: every
    shadow boot shrank the store and the run-peak headroom for bytes that arm
    cannot allocate.  A charge for a thing that does not happen is the mirror
    of the omission the term was added to close.
    """
    from sglang.srt.weg2 import host_ledger as hl
    from sglang.srt.weg2 import launcher as lc

    assert "oncard_mode" in inspect.signature(lc.choose_host_ledger).parameters
    # #1332 B1b MOVED THE SEAM, and this pin follows it rather than being
    # deleted.  The ledger call site now asks `xchg_bounce_terms_for_arm`,
    # which asks THIS function for the predicate ("does this arm pin host
    # bytes") and adds only the SIZE (the measured widest layer).  So the arm
    # still decides the charge at the one call site, and there is still exactly
    # one reader of the arm strings -- the assertion is that both halves are
    # wired, not that the older name appears in the older place.
    assert "xchg_bounce_terms_for_arm(" in \
        inspect.getsource(lc.choose_host_ledger)
    assert "xchg_bounce_arm_pins_host(weight_source" in \
        inspect.getsource(lc.xchg_bounce_terms_for_arm)
    main_src = inspect.getsource(lc.main)
    assert "oncard_mode=ns.weg2_xchg_oncard" in main_src, \
        "the arm must reach BOTH the ledger call site and the shadow env"
    assert main_src.count("oncard_mode=ns.weg2_xchg_oncard") == 2, main_src

    charge = lc.xchg_bounce_arm_pins_host
    # THE ONLY ARM THAT PINS A BYTE IS THE ONLY ARM THAT IS CHARGED.
    assert charge("shadow", tp.ONCARD_MODE_HOST) is True
    assert charge("exchange", tp.ONCARD_MODE_HOST) is True
    assert charge("shadow", tp.ONCARD_MODE_IPC) == 0
    assert charge("ring", tp.ONCARD_MODE_HOST) == 0
    assert charge("ring", tp.ONCARD_MODE_IPC) == 0
    assert xr.N_CARDS * xb.staging_bytes_per_card(
        tp.ONCARD_SLOT_BYTES_MAX) > 0, "this proves nothing at a zero term"

    # ... and the term reaches the ARM LINE with exactly that value.
    def _arms(term):
        _a, _s, lines = hl.choose(200 << 30, 150 << 30,
                                  ring_bytes=20 << 30, ring_span1_bytes=8 << 30,
                                  xchg_bounce_host_bytes=term)
        return [ln for ln in lines if ln.startswith("WEG2-HOST-LEDGER ARM ")]

    # THE PREDICATE SELECTS, THE TERM SIZES (AMENDMENT 5). Before the reshape
    # this test passed `charge(...)` straight in as the byte count, which is
    # exactly the conflation the ruling removed: one function answered both
    # "does this arm pin?" and "how much?". Now the arm chooses between 0 and
    # the term, and the term has one owner.
    _term = xr.N_CARDS * xb.staging_bytes_per_card(tp.ONCARD_SLOT_BYTES_MAX)
    ipc = _arms(_term if charge("shadow", tp.ONCARD_MODE_IPC) else 0)
    host = _arms(_term if charge("shadow", tp.ONCARD_MODE_HOST) else 0)
    assert ipc and host
    assert all("xchg_bounce=0.00" in ln for ln in ipc), ipc
    expect = f"xchg_bounce={_term / hl.GIB:.2f}"
    assert expect != "xchg_bounce=0.00"
    assert all(expect in ln for ln in host), (expect, host)


def test_the_launcher_publishes_the_on_card_arm_it_charged_for(tmp_path):
    """S6 refuter must_fix 4, the producer half: the flag spec 3.7 named.

    ``--weg2-xchg-oncard {ipc|host}`` is the spec's own flag and had no
    producer at all -- the ranks read the environment and the launcher never
    wrote it.  It is published like the hop bound and POPPED like it, because
    an inherited ``host`` from an operator's shell would put ranks on an arm
    this boot's ledger charged nothing for.
    """
    from sglang.srt.weg2 import launcher as lc

    assert lc.ONCARD_MODE_DEFAULT == tp.ONCARD_MODE_IPC
    assert set(lc.ONCARD_MODE_CHOICES) == {tp.ONCARD_MODE_IPC,
                                           tp.ONCARD_MODE_HOST}
    ns = lc.build_parser().parse_args(["--tree", "/t", "--tag", "x"])
    assert ns.weg2_xchg_oncard == tp.ONCARD_MODE_IPC
    with pytest.raises(SystemExit):
        lc.build_parser().parse_args(["--tree", "/t", "--tag", "x",
                                      "--weg2-xchg-oncard", "staging"])
    env_src = inspect.getsource(lc.prepare_xchg_env)
    assert "weight_exchange_transport.ENV_ONCARD_MODE" in env_src, env_src
    build_src = inspect.getsource(lc.build_env)
    assert "weight_exchange_transport.ENV_ONCARD_MODE" in build_src, \
        "an inherited on-card arm must be popped like the hop bound"
    # The ring arm publishes NOTHING, which is what keeps it byte-identical.
    assert lc.prepare_xchg_env(lambda _s: None, "b1", "ring",
                                 oncard_mode="host") == {}


def test_the_rank_budget_is_zero_on_the_arm_the_ledger_charged_nothing_for(
        monkeypatch):
    """S6 refuter finding 6: the 'ledger answer' was two module constants.

    ``_weg2_shadow_host_budget`` returned the full per-card charge on EVERY
    arm, while the launcher charges it on ``host`` only.  A rank that
    authorised a deposit against a term nobody carried would pin host bytes
    above the reap mark by exactly the amount that was never paid --
    ``host-schwelle-nie-uebertreten``.  It now asks the arm the launcher
    published, which is the same string that decided the charge.
    """
    from sglang.srt.managers.scheduler_components import weight_updater as wu

    budget = wu.SchedulerWeightUpdaterManager._weg2_shadow_host_budget

    monkeypatch.setenv(tp.ENV_ONCARD_MODE, tp.ONCARD_MODE_IPC)
    assert budget(object()) == 0
    monkeypatch.delenv(tp.ENV_ONCARD_MODE, raising=False)
    assert budget(object()) == 0, "the default arm is ipc and funds nothing"
    monkeypatch.setenv(tp.ENV_ONCARD_MODE, tp.ONCARD_MODE_HOST)
    assert budget(object()) == xb.staging_bytes_per_card(tp.ONCARD_SLOT_BYTES_MAX)
    assert budget(object()) * 4 == tp.ONCARD_DEPOSIT_BYTES_MAX


def test_the_shadow_line_prices_the_copy_on_the_host_arm_and_not_on_ipc(
        region, ops):
    """S6 refuter must_fix 1, at the seam that PRINTS the number.

    ``shadow_transport`` multiplied the batch count by ``ONCARD_PER_BATCH_MS``
    BY HAND -- a second copy of a formula that already had an owner, and one
    with no bytes term at all.  Both readings now come out of
    ``plan_oncard_slot_bytes``, so the hop the hook grades and the hop the line
    prints cannot be two different models.
    """
    desc = _diag(0, 3 << 20, cls="qkv_proj")
    _vote_rows(region, [1, 2, 3, 4, 5], leg=0, vote=False,
               classes_hash=sh.classes_hash(["qkv_proj"]), need_mib=0)

    def _run(mode):
        run = sh.shadow_transport(
            region=region, sems=None, ops=ops, row=0, rank=0, device=0,
            card_uuid="u0", uuid_of_card=["u0", "u1", "u2"], descs=[desc],
            is_source=True, oncard_mode=mode, peer_row=3, wave=WAVE,
            leg=0, direction="P->D", epoch="b.1", free_mib=8192,
            log=lambda _s: None, oncard_slot_bytes=1 << 20)
        return run.result

    ipc = _run(tp.ONCARD_MODE_IPC)
    host = _run(tp.ONCARD_MODE_HOST)
    assert ipc.oncard_batches == host.oncard_batches == 3
    # The ipc arm is byte-identical to the pre-S6 reading ...
    assert ipc.oncard_hop_ms_priced == 3 * tp.ONCARD_PER_BATCH_MS
    # ... and the host arm carries the bytes the old model dropped.
    expect = 3 * tp.ONCARD_PER_BATCH_MS + \
        ((3 << 20) / (tp.ONCARD_HOST_COPY_GBPS * 1e9)) * 1e3
    assert host.oncard_hop_ms_priced == pytest.approx(expect)
    # The bytes term is the LARGER of the two on a 3 MiB diagonal already,
    # and it grows with the bytes while the batch term does not.
    assert host.oncard_hop_ms_priced - ipc.oncard_hop_ms_priced == \
        pytest.approx(((3 << 20) / (tp.ONCARD_HOST_COPY_GBPS * 1e9)) * 1e3)
    assert host.oncard_hop_ms_priced > 2 * ipc.oncard_hop_ms_priced
    # Finding 8: neither lane deposits, so neither prints a deposit.
    assert ipc.oncard_deposit_mib == 0.0 and host.oncard_deposit_mib == 0.0
    assert "oncard_deposit_mib=0" in host.line(), host.line()


# ---------------------------------------------------------------------------
# S6 fix C (#1273) -- THE RANK IDENTITY THE TWO HOOKS ARE GATED ON.
#
# BOOT weg2shadowC IS THE MEASUREMENT.  Two arms (C1 ipc, C2 host), two flips
# each, `--weg2-weight-source shadow` armed and its ARM line printed -- and
# across all four flips ZERO `WEG2-XCHG-PLAN`, zero SEMS, zero ONCARD-REFUSED,
# zero COMPARE, zero W77/W81.  The ledger half of both arms was exact, so the
# arm itself was real.
#
# The root is one gate and it is not the arm: `_weg2_rank` read
# `getattr(scheduler, "tp_rank")` / `"pp_rank"`, and the Scheduler HAS NEITHER
# -- it keeps its parallel identity on the `ParallelState` wrapper.  The tree
# says so in three places already (`scheduler.py:1766`, `:8157-8161`,
# `:14502-14504`), each of them written after the same mistake raised
# AttributeError somewhere else.  Here it could not raise, because the read is
# `getattr(..., None)` behind an `isinstance(..., int)` test, so it degraded to
# a silent -1 -- and `-1 < 0` is the adapter's second pre-flight gate, which
# returns without a word.  Every downstream line is behind that return, which
# is why one root produced five empty gates.
#
# THE CORROBORATION IS IN EVERY BOOT LOG WE HAVE: `rank=-1` on 210 of 210
# `WEG2-FLIP-TAG` lines across the four shadowC rank logs, and the same -1 far
# enough back that `ring_table` WIDENED ITS PARSER to `rank=(-?\d+)` and grew a
# synthetic per-card index for it (`ring_table.py:159-166`, W37's docstring at
# `:866`) instead of the emitter being fixed. The instrument was believed
# before it was checked -- INDIKATOR-GESETZ, in its plainest form.
# ---------------------------------------------------------------------------

def _scheduler_rank_surface(*, world_rank, ps_tp_rank, ps_pp_rank, ps_tp_size):
    """A double with the REAL Scheduler's rank surface, and no other.

    The load-bearing property is an ABSENCE: no `tp_rank` and no `pp_rank` on
    the object itself.  A double that carries them cannot fail on this defect
    -- which is exactly how it survived: `test_census_attribute_surface_583`
    exists because "a desk test that stubbed the attribute never noticed".
    """
    import types

    sched = types.SimpleNamespace()
    sched.ps = types.SimpleNamespace(
        tp_rank=ps_tp_rank, pp_rank=ps_pp_rank, tp_size=ps_tp_size)
    sched.world_group = types.SimpleNamespace(rank_in_group=world_rank)
    assert not hasattr(sched, "tp_rank") and not hasattr(sched, "pp_rank")
    return sched


def _rank_probe(sched):
    from sglang.srt.managers.scheduler_components.weight_updater import (
        SchedulerWeightUpdaterManager,
    )

    probe = SchedulerWeightUpdaterManager.__new__(SchedulerWeightUpdaterManager)
    probe.scheduler = sched
    return SchedulerWeightUpdaterManager, probe


@pytest.mark.parametrize(
    "group,world_rank,ps_tp_rank,ps_pp_rank,ps_tp_size",
    [
        # Group D: pp_size=1, tp_size=3 (launcher.py:6525) -- tp_rank IS the
        # identity, and the flat form reduces to it.
        ("D", 0, 0, 0, 3), ("D", 1, 1, 0, 3), ("D", 2, 2, 0, 3),
        # Group P: pp_size=3, tp_size=1 -- `ps.tp_rank` is 0 on ALL THREE
        # ranks.  This is why "read ps.tp_rank instead" is not the fix: it
        # would give three ranks row 0 of the six-row gate matrix, which is
        # silently WRONG where -1 was merely silently absent.
        ("P", 0, 0, 0, 1), ("P", 1, 0, 1, 1), ("P", 2, 0, 2, 1),
    ],
)
def test_the_rank_gate_reads_an_identity_the_scheduler_actually_has(
        group, world_rank, ps_tp_rank, ps_pp_rank, ps_tp_size):
    """RED ON `d891223f54`: -1 on all six, for both groups.

    `rank_row` accepts 0..2 and raises outside it, so -1 is not a degraded
    answer the shadow could still work from -- it is the gate.
    """
    M, probe = _rank_probe(_scheduler_rank_surface(
        world_rank=world_rank, ps_tp_rank=ps_tp_rank,
        ps_pp_rank=ps_pp_rank, ps_tp_size=ps_tp_size))

    got = M._weg2_rank(probe)
    assert got == world_rank, (
        f"group {group} rank {world_rank} resolved to {got}; the adapter's "
        f"`rank < 0` gate then returns before run_leg_hook and the whole "
        f"shadow is silent (boot weg2shadowC)"
    )
    # The identity is worth having only if it is the one the region indexes by.
    assert xr.rank_row(group, got) == (0 if group == "P" else 3) + world_rank


def test_the_rank_is_unreadable_rather_than_wrong_when_there_is_no_scheduler():
    """-1 stays the answer where there is genuinely no identity to read.

    The fix may not invent a 0: rank 0 is a REAL row that another rank owns,
    and a rank that guesses it would have two publishers on one row.
    """
    import types

    M, probe = _rank_probe(None)
    assert M._weg2_rank(probe) == -1
    _, probe2 = _rank_probe(types.SimpleNamespace())
    assert M._weg2_rank(probe2) == -1


def test_the_identity_gate_of_an_armed_shadow_is_never_silent():
    """THE DISCRIMINATOR shadowC did not have.

    Both pre-flight gates of `_weg2_shadow_hook` returned without a line, so
    "the arm never reached the rank" and "the rank has no identity" produced
    byte-identical evidence: nothing.  The ARM gate must stay silent (it fires
    on every ring boot, i.e. every boot that has ever run); the IDENTITY gate
    fires only under an armed shadow, where silence is the defect.
    """
    import ast as _ast

    root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    path = os.path.join(root, "python", "sglang", "srt", "managers",
                        "scheduler_components", "weight_updater.py")
    with open(path, encoding="utf-8") as fh:
        src = fh.read()
    fn = next(n for n in _ast.walk(_ast.parse(src))
              if isinstance(n, _ast.FunctionDef) and n.name == "_weg2_shadow_hook")

    # The gate: `if group not in ("P", "D") or rank < 0:` -- find it and prove
    # its body says something before it returns.
    gates = [n for n in _ast.walk(fn)
             if isinstance(n, _ast.If)
             and "rank < 0" in _ast.unparse(n.test).replace(" ", " ")]
    assert len(gates) == 1, f"expected exactly one identity gate, got {len(gates)}"
    body = _ast.unparse(gates[0])
    assert "rank_local_skip_message" in body, (
        "the identity gate returns silently -- a shadow boot that produces no "
        "PLAN line then cannot say whether the arm or the identity stopped it, "
        "which is exactly the unresolved half of boot weg2shadowC"
    )


def test_the_rank_reader_does_not_go_back_to_the_attributes_that_do_not_exist():
    """MUTATION PIN.  `scheduler.tp_rank` / `scheduler.pp_rank` do not exist.

    Three sites in `scheduler.py` already carry a comment saying so, each
    written after the same read raised somewhere else.  This one could not
    raise, so nothing taught it; the pin is the teaching.
    """
    import ast as _ast

    root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    path = os.path.join(root, "python", "sglang", "srt", "managers",
                        "scheduler_components", "weight_updater.py")
    with open(path, encoding="utf-8") as fh:
        src = fh.read()
    fn = next(n for n in _ast.walk(_ast.parse(src))
              if isinstance(n, _ast.FunctionDef) and n.name == "_weg2_rank")
    body = _ast.unparse(fn)
    for dead in ("'tp_rank', 'pp_rank'", '"tp_rank", "pp_rank"'):
        assert dead not in body, (
            f"_weg2_rank still walks {dead} on the Scheduler, which has "
            f"neither -- that is the shadowC root"
        )


# ---------------------------------------------------------------------------
# S6 fix D (#1273) -- THE IDENTITY CLASS, swept instead of instanced.
#
# Boot weg2shadowD proved the rank fix (GATE 0: rank=0/1/2 on 37/35/33
# FLIP-TAG lines, rank=-1 = 0; GATE 1: 15 PLAN lines, 0 no-identity) and then
# produced the SAME SHAPE one level up, verbatim:
#
#   W68 Weg2XchgPlanDisagree begin_flip epoch='1788964408.-1': flip index -1
#   does not advance past -1, which this view has already run
#
# The boot half of that stamp is right and the FLIP half is -1.  The three legs
# that produced it are group P's BOOT-TIME INITIAL SLEEP at its own READY
# (14:33:55Z), 73 s BEFORE the first flip began (14:35:08Z) -- and the W79 that
# carried it printed `epoch=` EMPTY, so the request had no epoch at all rather
# than a malformed one.  The front's plumbing is CORRECT: every real flip leg
# of that boot carried `epoch=1788964408.0` / `.1`.
#
# So the defect is not a missing counter, it is a SENTINEL THAT TRAVELS: an
# absent identity became -1, the -1 was composed into a region stamp
# (`weight_exchange_shadow.py` `f"{boot_nonce}.{int(i.leg)}"`), and the
# refusal that came back named the wrong thing -- "the plan disagrees" about a
# leg whose real condition is "this is not a flip".
# ---------------------------------------------------------------------------

SHADOWD_W52_VERBATIM = (
    "W68 Weg2XchgPlanDisagree begin_flip epoch='1788964408.-1': flip index -1 "
    "does not advance past -1, which this view has already run"
)


def _armed_probe(monkeypatch, *, leg_epoch, card="GPU-abc", free_bytes=8 << 30):
    """An adapter probe whose five identity reads are all satisfiable.

    Everything the hook needs is stubbed at CLASS level (the manager is a
    ``slots=True`` dataclass, so per-instance attributes are not an option),
    which is also what lets each identity be knocked out ONE AT A TIME.
    """
    from sglang.srt.managers.scheduler_components import weight_updater as wu

    M = wu.SchedulerWeightUpdaterManager
    monkeypatch.setattr(M, "_weg2_group_name", lambda self: "P")
    monkeypatch.setattr(M, "_weg2_rank", lambda self: 1)
    monkeypatch.setattr(M, "_weg2_device_index", lambda self: 0)
    monkeypatch.setattr(M, "_weg2_card_uuid", lambda self: card)
    monkeypatch.setattr(M, "_weg2_free_bytes", lambda self: free_bytes)
    monkeypatch.setattr(M, "_weg2_shadow_gate_rows",
                        lambda self, h, g, leg=1: ())
    monkeypatch.setattr(M, "_weg2_shadow_host_budget", lambda self: 0)
    monkeypatch.setattr(M, "_weg2_shadow_plan",
                        lambda self, h, g, r: (None, "stub"))
    import types
    probe = M.__new__(M)
    probe.scheduler = None
    return M, probe, types.SimpleNamespace(epoch=leg_epoch)


@pytest.mark.parametrize("epoch_value,expect_reason", [
    # The shadowD leg, exactly: the boot-time initial sleep carries no epoch.
    (None, "no-flip-epoch"),
    ("", "no-flip-epoch"),
    # A boot nonce with no flip half -- the other way _weg2_flip_index_of
    # reaches -1, and it must land on the same named refusal.
    ("1788964408", "no-flip-epoch"),
    # A malformed tail is not an index either.
    ("1788964408.x", "no-flip-epoch"),
])
def test_a_leg_with_no_flip_epoch_refuses_by_name_before_the_sentinel_travels(
        monkeypatch, caplog, epoch_value, expect_reason):
    """RED ON `57ef547466`: the -1 travels and comes back as W68.

    The assertion that matters is the NEGATIVE one: the shadowD string must not
    be reachable from a leg that simply has no flip.  `-1` is a fine internal
    answer; composing it into a region stamp is what made a non-flip look like
    a plan disagreement.
    """
    from sglang.srt.weg2 import weight_exchange as wxm

    with wxm.weight_source_for_test(wxm.WEIGHT_SOURCE_SHADOW):
        M, probe, req = _armed_probe(monkeypatch, leg_epoch=epoch_value)
        with caplog.at_level("INFO"):
            M._weg2_shadow_hook(probe, "source", recv_req=req)

    assert f"reason={expect_reason}" in caplog.text, caplog.text[-2000:]
    assert "Weg2XchgPlanDisagree" not in caplog.text, (
        "the flip-index sentinel still reaches begin_flip -- this is the "
        "shadowD W68, which names a plan disagreement on a leg that is simply "
        "not a flip"
    )
    assert "does not advance past -1" not in caplog.text


def test_the_real_flip_epoch_is_not_refused(monkeypatch, caplog):
    """MUTANT GUARD.  A gate that refuses everything would pass the test above.

    `1788964408.0` is shadowD's own first flip leg, verbatim -- flip index 0,
    which is a legitimate index and must survive the gate.  If this goes red
    the fix has turned the boot-time refusal into a refusal of every flip.
    """
    from sglang.srt.weg2 import weight_exchange as wxm

    with wxm.weight_source_for_test(wxm.WEIGHT_SOURCE_SHADOW):
        M, probe, req = _armed_probe(monkeypatch, leg_epoch="1788964408.0")
        with caplog.at_level("INFO"):
            M._weg2_shadow_hook(probe, "source", recv_req=req)

    assert "reason=no-flip-epoch" not in caplog.text, (
        "flip index 0 was refused as 'no flip epoch' -- 0 is a real index and "
        "the boot's FIRST flip carries it"
    )


@pytest.mark.parametrize("knock_out,expect_reason", [
    ("_weg2_rank", "no-identity"),
    ("_weg2_group_name", "no-identity"),
    ("_weg2_device_index", "no-device"),
    ("_weg2_card_uuid", "no-card"),
    ("_weg2_free_bytes", "no-free-column"),
])
def test_every_identity_the_hook_reads_has_a_named_refusal(
        monkeypatch, caplog, knock_out, expect_reason):
    """THE SWEEP, as a table.  One identity missing at a time, each NAMED.

    Two of these five used to travel as sentinels into `ShadowLegInputs`:
    `card_uuid="unknown"` (an unnamed card priced, charged and printed on the
    W77 line as though it were a real one) and `free_mib=0` (an UNREADABLE NVML
    free column priced as a FULL card, so `price_shadow` refuses UNAFFORDABLE
    while naming the wrong cause -- worse than not pricing at all).
    """
    from sglang.srt.weg2 import weight_exchange as wxm

    missing = {"_weg2_rank": -1, "_weg2_group_name": "?",
               "_weg2_device_index": -1, "_weg2_card_uuid": None,
               "_weg2_free_bytes": None}[knock_out]

    with wxm.weight_source_for_test(wxm.WEIGHT_SOURCE_SHADOW):
        M, probe, req = _armed_probe(monkeypatch, leg_epoch="1788964408.0")
        monkeypatch.setattr(M, knock_out, lambda self, *a, **k: missing)
        with caplog.at_level("INFO"):
            M._weg2_shadow_hook(probe, "source", recv_req=req)

    assert f"reason={expect_reason}" in caplog.text, (
        f"knocking out {knock_out} produced no named refusal; "
        f"log tail: {caplog.text[-1500:]}"
    )


def test_no_identity_read_on_the_hook_path_keeps_a_travelling_sentinel():
    """RATCHET.  The FORM, not the five instances.

    Scans `_weg2_shadow_hook` for the shape this whole fix round is about: a
    value read with a sentinel default (`or "..."`, `0 if x is None else ...`)
    that is then handed straight into `ShadowLegInputs`.  Two of those were the
    S6-fix-D findings; a third added later must fail here rather than on metal.
    """
    import ast as _ast

    root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    path = os.path.join(root, "python", "sglang", "srt", "managers",
                        "scheduler_components", "weight_updater.py")
    with open(path, encoding="utf-8") as fh:
        src = fh.read()
    fn = next(n for n in _ast.walk(_ast.parse(src))
              if isinstance(n, _ast.FunctionDef) and n.name == "_weg2_shadow_hook")
    call = next(n for n in _ast.walk(fn)
                if isinstance(n, _ast.Call)
                and _ast.unparse(n.func).endswith("ShadowLegInputs"))

    offenders = []
    for kw in call.keywords:
        text = _ast.unparse(kw.value)
        # `or <literal>` and `<lit> if <x> is None else ...` are the two shapes
        # that turned an absence into a value the region then had to recognise.
        if isinstance(kw.value, _ast.BoolOp) and isinstance(kw.value.op, _ast.Or):
            offenders.append(f"{kw.arg}={text}")
        if isinstance(kw.value, _ast.IfExp) and "is None" in _ast.unparse(kw.value.test):
            offenders.append(f"{kw.arg}={text}")
    assert not offenders, (
        "identity sentinels still travel into ShadowLegInputs: "
        + "; ".join(offenders)
        + " -- resolve the identity BEFORE the inputs and refuse it by name "
          "(rank_local_skip_message), so no downstream reader has to recognise "
          "a sentinel"
    )


def test_the_oncard_copy_arm_is_on_the_line_on_both_arms():
    """S6 fix D, finding 2.  `oncard_copy_gbps` / `oncard_copy_ms` were 0 hits
    in both of shadowD's rank logs while the other three oncard fields printed
    15 times each.

    The emitter was never missing -- `tp.OncardLane.tokens` prints both -- but
    `ShadowResult` hand-copies a SUBSET of the lane's fields and these two were
    not in it, so `tp.oncard_copy_gbps(mode)` was computed, handed to the
    planner and dropped.  `0.0` on the ipc arm is a VALUE (the copy is D2D and
    already inside ONCARD_PER_BATCH_MS), so it must PRINT, never be omitted:
    otherwise "ipc prices no separate copy" and "nobody priced it" read alike.
    """
    r = sh.ShadowResult(leg=0, epoch="b.0",
                        subset=sh.select_subset([], leg=0),
                        counters=sh.ShadowCounters())
    assert hasattr(r, "oncard_copy_gbps") and hasattr(r, "oncard_copy_ms")
    line = r.line()
    assert "oncard_copy_gbps=" in line and "oncard_copy_ms=" in line, line

    # The ipc arm's 0.0 is a design value, not an absence.
    assert tp.oncard_copy_gbps(tp.ONCARD_MODE_IPC) == 0.0
    assert tp.oncard_copy_gbps(tp.ONCARD_MODE_HOST) == tp.ONCARD_HOST_COPY_GBPS
    r.oncard_copy_gbps = tp.oncard_copy_gbps(tp.ONCARD_MODE_IPC)
    assert "oncard_copy_gbps=0 " in r.line(), r.line()


# ---------------------------------------------------------------------------
# S6 fix E (#1273) -- THE SUBSET IS ONE CLASS PER LEG, AND IT WAS THE WHOLE
# POPULATION.
#
# S6-fix ticket C2's design of record: "the subset is ONE tensor class per leg.
# Crossover ~218 MiB of per-card subset diagonal".  Boot weg2shadowD measured
# `classes=14` on EVERY shadow line, with `rotation=` advancing (19/20/21) but
# the count never narrowing, `planned=575 of population=650`, dst_buffers 11836
# MiB against free 8126 MiB -> UNAFFORDABLE on every card of every leg ->
# `ran=no` on all 15 legs and not one byte ever compared.
#
# ROOT, one line: `run_leg_hook` did `classes = classes or plan.classes` and
# handed that to `select_subset(classes=...)`, whose `classes` parameter PINS
# the subset ("an operator arm").  `LegPlan`'s own docstring states the contract
# this violated, verbatim: "`classes` is the rotation the subset is chosen FROM
# (never the subset itself)".  The population went into the pin, the
# `if classes:` branch took all of it, and the rotation was computed and thrown
# away.  Same falsy-default shape as fix D's identity sentinels, one field over.
#
# THE FIX IS NOT "DELETE THE PIN".  `rotation_of(descs)` reads THIS CARD's
# descriptors, and shadowD measured rotation lengths 19 / 20 / 21 on P's three
# ranks in one leg -- `leg % len(rotation)` over three different lengths picks
# three DIFFERENT classes, which is exactly what the gate's `classes_hash`
# exists to catch.  The group-uniform list is `LegPlan.classes`, whose
# `LegPlanFacts` digest measured IDENTICAL on all six rows (plan_digest
# 0x81c24054c5b571bf, 18/18 lines).  So it is handed in as the ROTATION.
# ---------------------------------------------------------------------------

class _Desc:
    """The two fields `select_subset` reads, plus what the by-rank sum needs."""

    def __init__(self, param_name, nbytes=1 << 20, dst_rank=0, tag="weights",
                 kind="copy"):
        self.param_name = param_name
        self.nbytes = nbytes
        self.dst_rank = dst_rank
        self.tag = tag
        self.kind = kind


def _population(classes, per_class=3, nbytes=1 << 20):
    return [_Desc(f"model.layers.{i}.{c}.weight", nbytes=nbytes)
            for c in classes for i in range(per_class)]


FOURTEEN = ("A_log", "down_proj", "dt_bias", "gate_up_proj", "in_proj_ba",
            "in_proj_qkvz", "input_layernorm", "k_norm", "norm", "o_proj",
            "out_proj", "post_attention_layernorm", "q_norm", "qkv_proj")


def test_one_class_per_leg_and_the_next_leg_takes_the_next_class():
    """The design of record, as arithmetic: |subset| == 1, and it rotates.

    RED on `b1d68dfad0` only through the hook (below); `select_subset` itself
    was always right -- which is the point: the builder was correct and the
    CALLER pinned it.
    """
    descs = _population(FOURTEEN)
    seen = []
    for leg in range(len(FOURTEEN) + 2):
        sub = sh.select_subset(descs, leg=leg)
        assert len(sub.classes) == 1, (leg, sub.classes)
        seen.append(sub.classes[0])
    rot = sorted(FOURTEEN)
    assert seen[:len(rot)] == rot, seen
    # and it wraps rather than running off the end
    assert seen[len(rot)] == rot[0] and seen[len(rot) + 1] == rot[1]


def test_the_group_uniform_rotation_beats_this_cards_own_class_list():
    """shadowD's 19/20/21, as a test.

    Three ranks whose OWN descriptor sets carry different class counts must
    still choose the SAME class for the same leg once the group-uniform
    rotation is handed in -- otherwise the six gate rows disagree by
    construction and `classes_hash` fires on a difference nobody introduced.
    """
    rank_a = _population(FOURTEEN)                      # 14 classes
    rank_b = _population(FOURTEEN + ("extra_1",))       # 15
    rank_c = _population(FOURTEEN + ("extra_1", "extra_2"))  # 16

    for leg in range(6):
        picked = {
            tuple(sh.select_subset(d, leg=leg, rotation=FOURTEEN).classes)
            for d in (rank_a, rank_b, rank_c)
        }
        assert len(picked) == 1, (leg, picked)
        assert len(next(iter(picked))) == 1

    # WITHOUT the group-uniform rotation this is exactly what went wrong:
    # somewhere in the leg range the three per-card rotations disagree.
    drift_seen = False
    for leg in range(20):
        drifted = {
            tuple(sh.select_subset(d, leg=leg).classes)
            for d in (rank_a, rank_b, rank_c)
        }
        if len(drifted) > 1:
            drift_seen = True
    assert drift_seen, (
        "the per-card rotation never disagreed in 20 legs -- this fixture no "
        "longer reproduces the hazard the rotation parameter exists for"
    )


def test_a_population_handed_in_as_the_pin_takes_every_class():
    """THE DEFECT, isolated, so the fix's necessity is visible.

    This is what `classes = classes or plan.classes` did.  `select_subset` is
    behaving correctly here -- pinning is a real operator arm -- which is why
    the fix belongs at the caller and not in this function.
    """
    descs = _population(FOURTEEN)
    pinned = sh.select_subset(descs, leg=0, classes=FOURTEEN)
    assert len(pinned.classes) == 14
    rotated = sh.select_subset(descs, leg=0, rotation=FOURTEEN)
    assert len(rotated.classes) == 1


def test_the_subset_bytes_land_in_the_crossover_band_not_the_whole_image():
    """~218 MiB per-card diagonal (ticket C2), not 11836 MiB.

    The assertion is a RATIO, not an absolute: one class of fourteen must cost
    about a fourteenth of the population, so a subset that silently widens
    again fails here even if the fixture's byte size changes.
    """
    descs = _population(FOURTEEN, per_class=5, nbytes=16 << 20)  # 14*5*16 MiB
    whole = sum(d.nbytes for d in descs)
    sub = sh.select_subset(descs, leg=3, rotation=FOURTEEN)
    got = sum(v for v in sub.bytes_by_rank.values())
    assert len(sub.classes) == 1
    assert got * 14 == whole, (got, whole)
    assert got < whole // 10


def test_run_leg_hook_hands_the_plans_classes_in_as_the_rotation_not_the_pin():
    """RED on `b1d68dfad0`: the line reads `classes = classes or plan.classes`.

    Source-level, because the alternative is constructing a real LegPlan with
    a live model runner.  The two assertions are the contract `LegPlan`'s
    docstring already states and the code did not keep.
    """
    import ast as _ast

    root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    path = os.path.join(root, "python", "sglang", "srt", "weg2",
                        "weight_exchange_shadow.py")
    with open(path, encoding="utf-8") as fh:
        src = fh.read()
    fn = next(n for n in _ast.walk(_ast.parse(src))
              if isinstance(n, _ast.FunctionDef) and n.name == "run_leg_hook")
    body = _ast.unparse(fn)

    assert "classes = classes or plan.classes" not in body, (
        "the plan's class POPULATION is still fed into select_subset's PIN -- "
        "this is the shadowD classes=14 defect"
    )
    calls = [n for n in _ast.walk(fn) if isinstance(n, _ast.Call)
             and _ast.unparse(n.func).endswith("select_subset")
             # the no-plan early result selects over NO descs; there is no
             # rotation to be uniform about and none is required.
             and _ast.unparse(n.args[0] if n.args else _ast.Constant(0)) != "()"]
    assert len(calls) >= 1
    for c in calls:
        kw = {k.arg for k in c.keywords}
        assert "rotation" in kw, (
            "select_subset is called without the group-uniform rotation; with "
            "only this card's descs the three ranks of a group pick different "
            "classes for the same leg (shadowD: rotation 19/20/21)"
        )


# ---------------------------------------------------------------------------
# S6 fix E, second half -- THE OPERATOR DECISION ON D2/bs6, as a flag.
#
# Boot weg2shadowD D2 refused: W20, every rung of the ladder under the 6 GiB
# store floor (store 0 / 2 / 4 GiB) where shadow C2 on the same `host` arm at
# --d-bs 4 still had store 7.  Measured delta: bs 4 -> 6 costs 2.12 GiB of
# leftover (12.40 -> 10.28) and 2.14 GiB of reap-bound (7.02 -> 4.88), ~1.07
# GiB per additional D seat, against a 3.00 GiB xchg_bounce term and only 1.00
# GiB of margin on the last fundable rung.  Without the bounce term D2's M=600
# store would be 4 + 3 = 7 >= floor 6.
#
# OPERATOR DECISION 2026-09-09: keep the worst-case charge honest -- no
# per-flip charge (it would fund the smallest flip and refuse none, which
# `xchg_bounce_bytes_per_card`'s own docstring says in those words), no
# non-uniform x4-only arm -- and make the CEILING settable instead.
# ---------------------------------------------------------------------------

def test_the_slot_ceiling_flag_defaults_to_a_byte_identical_boot():
    """128 MiB and a 3.00 GiB charge is every boot so far."""

    assert tp.ONCARD_SLOT_MIB_DEFAULT == 128
    assert tp.validate_oncard_slot_mib(128) == 128
    per_card = xb.staging_bytes_per_card(128 * xr.MIB)
    assert per_card == xb.SLOTS_PER_PAIR * 128 * xr.MIB
    assert (xr.N_CARDS * xb.staging_bytes_per_card(128 * xr.MIB)
            ) / (1 << 30) == 0.75


def test_slot_64_charges_1_50_gib_which_is_what_reopens_a_rung_at_bs6():
    """The number ticket E2 depends on, computed rather than asserted by hand."""

    assert tp.validate_oncard_slot_mib(64) == 64
    assert (xr.N_CARDS * xb.staging_bytes_per_card(64 * xr.MIB)
            ) / (1 << 30) == 0.375
    # D2's M=600 rung had store 4 against floor 6 with 3.00 charged; giving
    # 1.50 back clears the floor.  This is the arithmetic, not a prediction
    # about the boot -- the anchors term is unchanged by the slot size.
    assert 4 + (3.0 - 1.5) >= 6 - 0.5


@pytest.mark.parametrize("bad", [0, -64, 16, 48, 100, "x", None, 31])
def test_a_slot_ceiling_that_does_not_divide_the_geometry_refuses_by_name(bad):
    """W82, not a clamp.

    A clamp is what makes a ledger charge one geometry while the ranks
    allocate another -- the drift `ONCARD_DEPOSIT_BYTES_MAX`'s own comment was
    written after (it had charged the slot FLOOR beside the COUNT's ceiling and
    understated the maximum 4x).
    """
    with pytest.raises(tp.Weg2XchgOncardSlotRefused) as exc:
        tp.validate_oncard_slot_mib(bad)
    assert "W82" in str(exc.value)
    assert "does not divide" in str(exc.value) or "not an integer" in str(exc.value)


@pytest.mark.parametrize("good", [32, 64, 96, 128, 256])
def test_whole_multiples_of_the_32_mib_floor_are_accepted(good):
    assert tp.validate_oncard_slot_mib(good) == good
    assert good % (tp.ONCARD_SLOT_BYTES // xr.MIB) == 0


def test_the_launcher_publishes_the_ceiling_to_both_groups_and_pops_it():
    """Same discipline as the arm: launcher OUTPUT, published and popped.

    An inherited `SGLANG_WEG2_XCHG_ONCARD_SLOT_MIB` would size a deposit this
    boot's ledger charged a different worst case for -- the exact two-readings
    failure the arm's own publication comment names.
    """
    root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    path = os.path.join(root, "python", "sglang", "srt", "weg2", "launcher.py")
    with open(path, encoding="utf-8") as fh:
        src = fh.read()
    assert "--weg2-xchg-oncard-slot-mib" in src
    assert "ENV_ONCARD_SLOT_MIB] = str(slot_mib)" in src
    # Popped in build_env's launcher-OUTPUT loop and published by
    # prepare_xchg_env, whose dict build_env applies AFTER that loop -- the
    # ordering that matters is the runtime one (pop, then override), which the
    # arm's three names already rely on and share this loop with.
    # POSITION-INDEPENDENT, and that is a gate finding worth keeping (#1336
    # local pair gate): this asserted the slot followed by the pop-tuple's
    # CLOSING PAREN, so appending any name to that tuple broke the guard while
    # the property it exists for -- the slot IS popped -- still held. The
    # parked #1336 publisher hit exactly that. Assert membership, not place.
    assert "weight_exchange_transport.ENV_ONCARD_SLOT_MIB" in src
    i_loop = src.index("for key in (\"SGLANG_WEG2_XCHG_REGION\"")
    i_apply = src.index("for key, value in (xchg_env or {}).items():")
    assert i_loop < i_apply, "the pop loop must run before the xchg_env overrides"


def test_the_launcher_charge_follows_the_flag_not_its_own_import():
    """The trap this threading exists for.

    The launcher process starts WITHOUT the flag in its environment, so
    `tp.ONCARD_SLOT_BYTES_MAX` binds the 128 MiB default at import.  A charge
    read from that constant would price 128 MiB slots while the ranks, which DO
    have the published env, allocate 64 -- a ledger that funded one geometry
    and a boot that ran another.
    """
    import ast as _ast

    root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    path = os.path.join(root, "python", "sglang", "srt", "weg2", "launcher.py")
    with open(path, encoding="utf-8") as fh:
        src = fh.read()
    # AMENDMENT 5 MOVED THIS TRAP, it did not remove it. The arm predicate no
    # longer carries a size, so the flag cannot reach a charge THROUGH it; the
    # function that must see the flag is the one that SIZES the term. Graded
    # there, the trap is the same one: a term read from the launcher's
    # import-time `tp.ONCARD_SLOT_BYTES_MAX` would price the module default
    # while the ranks allocate the published value.
    fn = next(n for n in _ast.walk(_ast.parse(src))
              if isinstance(n, _ast.FunctionDef)
              and n.name == "xchg_bounce_terms_for_arm")
    assert "oncard_slot_mib" in {a.arg for a in fn.args.args}, (
        "the term cannot see the flag, so it prices the launcher's import-"
        "time default"
    )
    body = _ast.unparse(fn)
    assert "validate_oncard_slot_mib" in body, (
        "the charge accepts a slot size it never validated"
    )


# ---------------------------------------------------------------------------
# S6 fix F (#1273) -- `pieces=0` IS NOT A PIECE-BUILDER DEFECT.  THE LEG NEVER
# REACHED THE PIECE BUILDER.
#
# Boot weg2shadowE: the rotation fix landed (classes=1 on 12/12 lines, the
# subset rotates A_log/down_proj, W68 0, W77 0, 19 AFFORDABLE / 5 REFUSED,
# need_mib=32, deposit runs, hop priced 0.182) and every leg still read
# `ran=no ... pieces=0 stripes=0`.  The reason was already on the same line,
# 400 characters further along:
#
#     verdict=NOT-RUN reason=plan-diverged-oncard-peer
#
# and it is 6 of 6 shadow lines in BOTH groups -- 12 of 12 legs, one cause, no
# second.  `result.ran = True` and `result.pieces = ...` are set only AFTER
# `tp.run_leg` returns, so a leg refused at the gate reports the initialised
# zeros.  NONE of the four candidates put to this round applies: the piece
# builder does not read the pin, imposes no stripe minimum, filters no dtype,
# and its population handle is full -- it is simply never called.
#
# WHAT THE GATE FOUND IS REAL, AND THE GATE IS RIGHT.  W80 scope=oncard-peer
# compares the CO-LOCATED pair's `card_digest`:
#
#     W80 Weg2XchgShadowPlanDiverged scope=oncard-peer leg=0 field=card_digest
#       [row=0 card_digest=0xb580709a80b8e768] [row=3 card_digest=0x64a79d7d9e67e603]
#
# row 0 is P rank 0 and row 3 is D rank 0 -- the two ranks sharing card 0.
# `card_geometry_digest` hashes this rank's STORAGE (`name:tag:rows x cols x
# itemsize`, plus the unplanned population), and its own docstring says a
# disagreement "is the honest reading of 'these two groups do not hold the same
# bytes on this card'".  Measured across the run: SIX distinct card_digests,
# each appearing 3x in one group's log and 1x in the other's -- three are P's
# cards and three are D's, and no co-located pair agrees.
#
# THE PREMISE, NOT THE CODE, IS WHAT FAILS.  Group P runs pp_size=3 / tp_size=1
# and group D pp_size=1 / tp_size=3 (`launcher.py`), so P rank n holds a
# PIPELINE STAGE (whole layers, some layer indices) while D rank n holds a
# TENSOR SHARD (slices of every layer).  Co-located ranks genuinely do not hold
# the same parameter names or extents, so the on-card diagonal has no shared
# storage to exchange and the gate can never open on this placement.  That is a
# DESIGN question for the operator, not a defect to patch here -- and these
# tests exist so nobody "fixes" it by weakening the gate, which would compare
# two different experiments and call the result a MATCH.
# ---------------------------------------------------------------------------

def test_the_oncard_peer_gate_refuses_when_the_pair_hold_different_storage():
    """RED if the gate is ever weakened.  This refusal is CORRECT.

    Two co-located ranks whose storage differs must produce a NAMED refusal and
    no comparison.  Weakening this to "compare anyway" is the shape S5's own
    can-fail control already caught once (both sides of the compare became the
    shadow buffer, so every stripe matched itself).
    """
    from sglang.srt.weg2 import weight_exchange_shadow as m

    a = m.card_geometry_digest(
        [_Geom("model.layers.0.A_log", "weights_0", 4096, 1, 2)],
        ("A_log",), unplanned=())
    b = m.card_geometry_digest(
        [_Geom("model.layers.9.A_log", "weights_2", 4096, 1, 2)],
        ("A_log",), unplanned=())
    assert a != b, (
        "a pipeline stage and a tensor shard hashed to the same card geometry "
        "-- the gate that stops the shadow comparing two different experiments "
        "has stopped discriminating"
    )
    same = m.card_geometry_digest(
        [_Geom("model.layers.0.A_log", "weights_0", 4096, 1, 2)],
        ("A_log",), unplanned=())
    assert same == a, "the digest is not stable over identical storage"


class _Geom:
    def __init__(self, name, tag, rows_full, cols_full, itemsize):
        self.name = name
        self.tag = tag
        self.rows_full = rows_full
        self.cols_full = cols_full
        self.itemsize = itemsize


def test_a_one_class_subset_has_pieces_to_exchange():
    """THE FOUR CANDIDATES, refuted in one test.

    A single rotated class over a synthetic population yields a NON-EMPTY desc
    list and a NON-ZERO layout total.  So the piece builder does not read the
    (now empty) pin, imposes no minimum piece size that a 32 MiB class misses,
    filters nothing by tensor property, and its population handle is full.
    `pieces=0` on metal was the gate, not this.
    """
    descs = _population(FOURTEEN, per_class=4, nbytes=8 << 20)
    sub = sh.select_subset(descs, leg=0, rotation=FOURTEEN)
    assert len(sub.classes) == 1
    assert len(sub.descs) == 4, sub.classes
    assert sub.nbytes == 4 * (8 << 20)
    mine = [d for d in sub.descs if int(d.dst_rank) == 0]
    layout, total = sh.shadow_layout(mine)
    assert total > 0 and len(layout) == len(mine), (total, len(layout))


def test_a_refused_leg_says_why_beside_ran_not_four_hundred_chars_later():
    """The readability defect that cost boot weg2shadowE a cycle.

    `reason=` was already on the line and was missed by every reader, because
    the head of the line reads `ran=no ... pieces=0 stripes=0` and the cause sat
    at the far end.  The trailing `reason=` is kept so existing parsers do not
    move.
    """
    r = sh.ShadowResult(leg=0, epoch="b.0",
                        subset=sh.select_subset([], leg=0),
                        counters=sh.ShadowCounters())
    r.reason = "plan-diverged-oncard-peer"
    line = r.line()
    assert "why=plan-diverged-oncard-peer" in line
    assert line.index("ran=") < line.index("why=") < line.index("pieces="), line
    # and the trailing field is untouched
    assert "reason=plan-diverged-oncard-peer" in line

    ok = sh.ShadowResult(leg=0, epoch="b.0",
                         subset=sh.select_subset([], leg=0),
                         counters=sh.ShadowCounters())
    ok.ran = True
    assert "why=ok" in ok.line()


# ---------------------------------------------------------------------------
# S6 fix F2 (#1273) -- THE ON-CARD GATE IS REBUILT ON THE PLAN'S PIECES.
#
# Operator ruling 2026-09-09 on SECTION 1af / #1277: the exchange's unit under
# P=PP3 / D=TP3 was NEVER "the same tensors on the same card" -- it is the
# SUB-TENSOR OVERLAP.  The layer resident WHOLE on card n in P CONTAINS
# D-rank-n's TP shard of that layer, and #1277 measured exactly that as the
# diagonal: 10.285 GiB on-card (35.7 %) against 18.560 GiB cross-card, on
# byte-exact checksums.
#
# So `card_geometry_digest` (name:tag:rows x cols x itemsize over WHOLE
# storage) is the wrong predicate for asymmetric placement: it must always
# differ between a stage-holder and a shard-holder.  Boot weg2shadowE measured
# it doing exactly that -- six distinct card_digests, no co-located pair
# agreeing -- and refusing all 12 legs of a lane that has real bytes.
# ---------------------------------------------------------------------------

def _desc(name, *, src, dst, nbytes, src_off=0, dst_off=0, tag="weights_0"):
    return wx.XchgDesc(tag=tag, src_rank=src, dst_rank=dst, param_name=name,
                       kind="copy", nbytes=nbytes, rows=1, run_bytes=nbytes,
                       spitch=nbytes, dpitch=nbytes,
                       src_off=src_off, dst_off=dst_off)


def _overlap_plan_descs():
    """Card 0's diagonal: P holds layer 0 whole, D holds its TP shard of it.

    The pieces are SLICES of the resident tensor -- name plus offsets plus
    bytes -- which is what #1277 measured moving on-card.  Card 1 pieces are in
    the list too, so the per-card filter is exercised rather than assumed.
    """
    return [
        _desc("model.layers.0.qkv_proj.weight", src=0, dst=0,
              nbytes=4 << 20, src_off=0, dst_off=0),
        _desc("model.layers.0.o_proj.weight", src=0, dst=0,
              nbytes=2 << 20, src_off=4 << 20, dst_off=0),
        _desc("model.layers.9.qkv_proj.weight", src=1, dst=1, nbytes=8 << 20),
        _desc("model.layers.0.down_proj.weight", src=0, dst=2,
              nbytes=1 << 20),          # cross-card, must NOT enter card 0
    ]


def test_the_pair_agrees_on_pieces_while_their_whole_storage_digests_differ():
    """THE RULING, as one test, with the old predicate as the explicit can-fail.

    Both co-located ranks read the SAME group-uniform plan, so both derive the
    same on-card piece set for their card -- while their storage inventories
    (a pipeline stage vs a tensor shard) are genuinely different objects.
    """
    descs = _overlap_plan_descs()

    stage_side = sh.oncard_piece_digest(descs, 0)
    shard_side = sh.oncard_piece_digest(descs, 0)
    assert stage_side == shard_side
    assert sh.oncard_piece_count(descs, 0) == 2, "card 0's diagonal is 2 pieces"
    # the cross-card desc and card 1's diagonal stay out of card 0's set
    assert sh.oncard_piece_digest(descs, 1) != stage_side
    assert sh.oncard_piece_count(descs, 1) == 1

    # THE CAN-FAIL AGAINST REGRESSION: the OLD predicate refuses this pair.
    stage_storage = [_Geom("model.layers.0.qkv_proj.weight", "weights_0",
                           4096, 1024, 2)]
    shard_storage = [_Geom("model.layers.0.qkv_proj.weight", "weights_0",
                           1365, 1024, 2),
                     _Geom("model.layers.9.qkv_proj.weight", "weights_2",
                           1365, 1024, 2)]
    old_a = sh.card_geometry_digest(stage_storage, ("qkv_proj",), unplanned=())
    old_b = sh.card_geometry_digest(shard_storage, ("qkv_proj",), unplanned=())
    assert old_a != old_b, (
        "the whole-storage digests now AGREE for a stage/shard pair -- this "
        "fixture no longer reproduces the predicate that refused all 12 legs "
        "of boot weg2shadowE, so it no longer proves the gate was rebuilt"
    )


def test_a_mismatched_piece_set_still_diverges_and_names_a_piece():
    """The W70 hazard the gate exists for survives the rebuild.

    Two ranks that framed the SAME overlap differently have different piece
    keys -- a different slice offset is a different piece -- so the gate still
    catches it, now under field=piece_digest.
    """
    mine = _overlap_plan_descs()
    theirs = [_desc("model.layers.0.qkv_proj.weight", src=0, dst=0,
                    nbytes=4 << 20, src_off=0, dst_off=0),
              _desc("model.layers.0.o_proj.weight", src=0, dst=0,
                    nbytes=2 << 20, src_off=8 << 20, dst_off=0)]  # <- offset
    assert sh.oncard_piece_digest(mine, 0) != sh.oncard_piece_digest(theirs, 0)

    hint = sh.first_piece_difference(mine, 0, 0)
    assert "planned 2 on-card piece" in hint and "first=" in hint, hint


def test_a_pair_with_no_overlap_is_named_not_a_silent_pieces_zero():
    """'no on-card pieces on this pair', never `pieces=0 ran=no`.

    This is the shadowE lesson made structural: an initialised zero must never
    stand in for a reason.
    """
    cross_only = [_desc("model.layers.0.down_proj.weight", src=0, dst=2,
                        nbytes=1 << 20)]
    assert sh.oncard_piece_count(cross_only, 0) == 0
    assert "planned NO on-card pieces" in sh.first_piece_difference(
        cross_only, 0, 0)

    import ast as _ast
    root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    path = os.path.join(root, "python", "sglang", "srt", "weg2",
                        "weight_exchange_shadow.py")
    with open(path, encoding="utf-8") as fh:
        src = fh.read()
    fn = next(n for n in _ast.walk(_ast.parse(src))
              if isinstance(n, _ast.FunctionDef) and n.name == "run_leg_hook")
    body = _ast.unparse(fn)
    assert "'no-oncard-pieces'" in body or '"no-oncard-pieces"' in body
    assert "rank_local_skip_message" in body


def test_the_gate_votes_the_piece_digest_and_never_the_storage_geometry():
    """RANK-UNIFORMITY PIN, and the reason the rebuild is sound.

    The voted field must come from the PLAN (`plan.piece_digest`, derived from
    `plan.descs` and the card index) and never from a rank-local storage walk.
    A stage-holder and a shard-holder enumerate different tensors; only a
    plan-derived number can be equal on both sides.
    """
    import ast as _ast

    root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    path = os.path.join(root, "python", "sglang", "srt", "weg2",
                        "weight_exchange_shadow.py")
    with open(path, encoding="utf-8") as fh:
        src = fh.read()
    tree = _ast.parse(src)

    hook = next(n for n in _ast.walk(tree) if isinstance(n, _ast.FunctionDef)
                and n.name == "run_leg_hook")
    hb = _ast.unparse(hook)
    assert "piece_digest = plan.piece_digest" in hb, hb[:200]
    assert "card_digest" not in hb, "the hook still carries the storage digest"

    gate = next(n for n in _ast.walk(tree) if isinstance(n, _ast.FunctionDef)
                and n.name == "shadow_gate")
    gb = _ast.unparse(gate)
    assert "card_digest" not in gb, "the gate still compares whole storage"
    assert 'field="piece_digest"' in gb or "field='piece_digest'" in gb

    # the geometry digest survives as INFORMATION on the plan line
    plan_line = next(n for n in _ast.walk(tree) if isinstance(n, _ast.FunctionDef)
                     and n.name == "line")
    assert "card_digest=" in src and "piece_digest=" in src


def test_the_param_census_is_one_line_per_rank_and_never_repeats():
    """The cheap evidence SECTION 1ai-F-root asked for, and its bound.

    The pp/tp asymmetry must be readable off the boot log rather than off
    `launcher.py`.  It must also cost one walk per PROCESS, not one per leg --
    an observer that pays a leg's wall for evidence is the thing this whole
    slice may not be.
    """
    import ast as _ast

    root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    path = os.path.join(root, "python", "sglang", "srt", "managers",
                        "scheduler_components", "weight_updater.py")
    with open(path, encoding="utf-8") as fh:
        src = fh.read()
    fn = next(n for n in _ast.walk(_ast.parse(src))
              if isinstance(n, _ast.FunctionDef)
              and n.name == "_weg2_shadow_param_census")
    body = _ast.unparse(fn)
    assert "PARAM-CENSUS" in body
    assert "_weg2_param_census_done" in body, "the census repeats per leg"
    assert "named_parameters" in body
    handlers = [h for n in _ast.walk(fn) if isinstance(n, _ast.Try)
                for h in n.handlers]
    assert any(isinstance(h.type, _ast.Name) and h.type.id == "BaseException"
               for h in handlers), "the census can raise into a flip leg"
