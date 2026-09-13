# SPDX-License-Identifier: Apache-2.0
"""#1330 B4n -- THE PHASE SPLIT, and the root of weg2xsn20's 24/24 W74.

THE ROOT, verified at ``c479a97ff6`` and recorded in the plan of record:
``run_bounce_leg`` ran DEPOSIT and (COMPARE|COLLECT) over ONE descriptor list,
in ONE process, in ONE pass -- the loop at ``weight_exchange_bounce.py:893``
calls ``_deposit_band`` (``:657``, reads ``desc.src_ptr``, a DEVICE address)
and then, in the same iteration, ``_compare_band``/``_collect_band``
(``:678``, reads ``desc.dst_ptr``, also DEVICE).  So it needed both device
addresses at once, and at the destination hook ``src_ptr`` is by construction
the PEER's (``weight_exchange_shadow.py:3336-3344``, side chosen at
``:3329-3331``), which no process can read.  ``_missing_pointer`` then raised
W74 on every leg -- and the profile always read ``src_resolved=0/N
dst_resolved=N/N``, never the mirror, which is the signature of exactly this
cause and not of a missing plan.

THE DANGER DIRECTION OF THE FIX, pinned here rather than discovered on metal: a
collect that reads a slot no deposit has filled returns whatever the previous
band left -- the destination is served plausible bytes from the WRONG layer
with every counter green.  That is silent corruption, strictly worse than any
refusal.  Two mutants, both must be RED:

  M1  swap the order (collect before deposit)      -> W68 refusal
  M2  deposit that never posts `full`              -> W68 refusal

THE HARNESS IS THE PRODUCT'S OWN.  ``FakeDeviceOps`` (transport test, :115) is
cross-process by construction -- device memory is a file per rank, host
pointers are the REAL mapped shm -- and it moves real bytes, so "the deposit
landed" is a byte comparison and not a call count.
"""

from __future__ import annotations

import os
import sys

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sglang.srt.weg2 import weight_exchange as wx  # noqa: E402
from sglang.srt.weg2 import weight_exchange_bounce as wb  # noqa: E402
from test_weg2_xchg_transport_1273 import FakeDeviceOps  # noqa: E402

TAG = "weights_0"
SLOT = 1 << 16

#: PID-SCOPED, and #1344's argument is the whole reason: a bounce keyed on a
#: fixed literal writes /dev/shm/weg2-xchg-<literal>, which no per-test cleanup
#: can claim ownership of and which survives the run.  The name IS the proof of
#: ownership, so it carries this interpreter's pid and the autouse fixture
#: below can sweep it without a holder check.
NONCE = f"phz{os.getpid()}"


@pytest.fixture(autouse=True)
def _sweep_own_shm():
    """Remove THIS interpreter's bounce residue, and only ever its own."""
    yield
    import shutil

    # EVERY PREFIX THIS TEST CAN CREATE, and the list grew once already: slice
    # 4 added `weg2-xchg-bnc-<nonce>` (the bounce's own slot record) and the
    # sweep still named only the region prefix, so nine files accumulated
    # across runs. A cleanup that knows one of its own two artefacts is the
    # same half-measure as a check written against the last incident.
    for name in os.listdir("/dev/shm"):
        if (name == f"weg2-xchg-{NONCE}"
                or name.startswith(f"weg2-xchg-{NONCE}-")
                or name == f"{wb.BOUNCE_SLOT_PREFIX}{NONCE}"):
            target = os.path.join("/dev/shm", name)
            if os.path.isdir(target):
                shutil.rmtree(target, ignore_errors=True)
            else:
                try:
                    os.unlink(target)
                except OSError:
                    pass


class _Rendezvous:
    """The empty/full handshake, as a double that CAN be driven out of order.

    The product adapter wraps the region's CROSS semaphores -- which already
    exist and are already counted (weg2xsn20's teardown census: 24 cross + 12
    diagonal).  This double exists because the ONLY way to prove the refusal
    can fire is to violate the protocol on purpose, and a real semaphore set
    would simply block instead.
    """

    def __init__(self, *, drop_post: bool = False, lie_bytes: int = 0):
        self.posted = {}
        self.emptied = []
        self.drop_post = bool(drop_post)
        self.lie_bytes = int(lie_bytes)

    def wait_empty(self, *, slot, seq):
        """Slice 3 added this to the contract: a deposit claims its slot."""
        self.emptied_waits = getattr(self, "emptied_waits", 0) + 1
        return True

    def post_full(self, *, slot, seq, nbytes):
        if self.drop_post:
            return
        self.posted[(int(slot), int(seq))] = (
            self.lie_bytes if self.lie_bytes else int(nbytes))

    def wait_full(self, *, slot, seq):
        return self.posted.get((int(slot), int(seq)))

    def post_empty(self, *, slot, seq):
        self.emptied.append((int(slot), int(seq)))


# ---------------------------------------------------------------------------
# (1) THE PHASE-AWARE HOLE CHECK -- the function that raised 24/24
# ---------------------------------------------------------------------------


def _d(src, dst):
    return wx.XchgDesc(tag=TAG, src_rank=0, dst_rank=1, param_name="p",
                       src_ptr=src, dst_ptr=dst, kind=wx.FLAT, nbytes=16,
                       rows=1, run_bytes=16, spitch=0, dpitch=0, src_off=0,
                       dst_off=0)


def test_the_phases_need_only_the_pointer_they_actually_read():
    """THE FIX AT ITS ROOT.  A collect needs no source address at all.

    Before the split every phase needed both, which is why an unreachable PEER
    address became a refusal on every leg instead of a job for the other rank.
    """
    src_only = [_d(0x1000, None)]
    dst_only = [_d(None, 0x2000)]
    both = [_d(0x1000, 0x2000)]

    assert wb._missing_pointer(src_only, wb.PHASE_DEPOSIT) is None
    assert wb._missing_pointer(dst_only, wb.PHASE_COLLECT) is None
    assert wb._missing_pointer(both, wb.PHASE_BOTH) is None

    # ... and each phase still names the side IT needs.
    assert "no destination pointer" in wb._missing_pointer(
        src_only, wb.PHASE_COLLECT)
    assert "no source pointer" in wb._missing_pointer(
        dst_only, wb.PHASE_DEPOSIT)
    assert "no source pointer" in wb._missing_pointer(dst_only, wb.PHASE_BOTH)


def test_the_default_phase_is_both_so_the_unsplit_form_is_unchanged():
    """``both`` is the single-process form and must stay byte-identical."""
    import inspect

    sig = inspect.signature(wb.run_bounce_leg)
    assert sig.parameters["phase"].default == wb.PHASE_BOTH
    assert sig.parameters["rendezvous"].default is None
    assert wb.PHASE_CHOICES == (wb.PHASE_DEPOSIT, wb.PHASE_COLLECT,
                                wb.PHASE_BOTH)


def test_an_unknown_phase_refuses_rather_than_defaulting():
    with pytest.raises(ValueError) as exc:
        wb.run_bounce_leg([], ops=None, boot_nonce=NONCE, slot_bytes=SLOT,
                          depth=1, phase="deposit-ish")
    assert "unknown bounce phase" in str(exc.value)


def test_a_phased_leg_without_a_handshake_is_refused():
    """M2's sibling: running a phase optimistically IS the unordered read."""
    for phase in (wb.PHASE_DEPOSIT, wb.PHASE_COLLECT):
        with pytest.raises(wb.Weg2XchgBouncePhaseUnordered) as exc:
            wb.run_bounce_leg([_d(0x1000, 0x2000)], ops=None, boot_nonce=NONCE,
                              slot_bytes=SLOT, depth=1, phase=phase,
                              rendezvous=None)
        assert "W68" in str(exc.value)
    # `both` needs none and gets past the guard (it fails later, on ops=None).
    with pytest.raises(Exception) as exc:
        wb.run_bounce_leg([_d(0x1000, 0x2000)], ops=None, boot_nonce=NONCE,
                          slot_bytes=SLOT, depth=1, phase=wb.PHASE_BOTH)
    assert not isinstance(exc.value, wb.Weg2XchgBouncePhaseUnordered)


# ---------------------------------------------------------------------------
# (2) THE TWO HALVES ON REAL BYTES, IN TWO "PROCESSES"
# ---------------------------------------------------------------------------


@pytest.fixture()
def ops(tmp_path):
    return FakeDeviceOps(str(tmp_path), rank=0)


def _seed(ops, ptr, nbytes, value):
    import ctypes

    ctypes.memset(ops.real(ptr), value, nbytes)


def _read(ops, ptr, nbytes):
    import ctypes

    return ctypes.string_at(ops.real(ptr), nbytes)


def test_deposit_then_collect_moves_the_bytes_and_needs_one_pointer_each(
        tmp_path):
    """THE GOAL SHAPE, on real bytes: two ranks, one slot, no peer pointer.

    The depositing rank holds ONLY ``src_ptr``; the collecting rank holds ONLY
    ``dst_ptr``.  Neither can see the other's memory -- which is the true
    situation on metal and the one the unsplit form could not express.
    """
    nbytes = 4096
    src_ops = FakeDeviceOps(str(tmp_path), rank=0)
    dst_ops = FakeDeviceOps(str(tmp_path), rank=1)
    src_ptr = src_ops.raw_malloc(0, nbytes)
    dst_ptr = dst_ops.raw_malloc(0, nbytes)
    _seed(src_ops, src_ptr, nbytes, 0xAB)
    _seed(dst_ops, dst_ptr, nbytes, 0x00)

    deposit_descs = [wx.XchgDesc(
        tag=TAG, src_rank=0, dst_rank=1, param_name="model.layers.0.w",
        src_ptr=src_ptr, dst_ptr=None, kind=wx.FLAT, nbytes=nbytes, rows=1,
        run_bytes=nbytes, spitch=0, dpitch=0, src_off=0, dst_off=0)]
    collect_descs = [wx.XchgDesc(
        tag=TAG, src_rank=0, dst_rank=1, param_name="model.layers.0.w",
        src_ptr=None, dst_ptr=dst_ptr, kind=wx.FLAT, nbytes=nbytes, rows=1,
        run_bytes=nbytes, spitch=0, dpitch=0, src_off=0, dst_off=0)]

    rv = _Rendezvous()
    nonce = "phase1"
    wb.run_bounce_leg(deposit_descs, src_ops, nonce, slot_bytes=SLOT, depth=1,
                      mode=wx.INJECT_AUTHORITATIVE, phase=wb.PHASE_DEPOSIT,
                      rendezvous=rv, shm_root=str(tmp_path), lane="p0")
    assert rv.posted, "the deposit must post `full` after its sync"

    wb.run_bounce_leg(collect_descs, dst_ops, nonce, slot_bytes=SLOT, depth=1,
                      mode=wx.INJECT_AUTHORITATIVE, phase=wb.PHASE_COLLECT,
                      rendezvous=rv, shm_root=str(tmp_path), lane="p0")
    assert rv.emptied, "the collect must release the slot"
    assert _read(dst_ops, dst_ptr, nbytes) == b"\xab" * nbytes, (
        "the bytes did not cross the slot")


def test_M1_a_collect_before_any_deposit_refuses_and_issues_nothing(tmp_path):
    """MUTANT M1 -- THE DANGER DIRECTION, and it must be a REFUSAL.

    Reading an unposted slot returns the previous band's bytes: the wrong
    layer, served with every counter green. The destination arena must be
    UNTOUCHED after the refusal, which is what "issues NOTHING" means.
    """
    nbytes = 4096
    dst_ops = FakeDeviceOps(str(tmp_path), rank=1)
    dst_ptr = dst_ops.raw_malloc(0, nbytes)
    _seed(dst_ops, dst_ptr, nbytes, 0x11)
    descs = [wx.XchgDesc(
        tag=TAG, src_rank=0, dst_rank=1, param_name="model.layers.0.w",
        src_ptr=None, dst_ptr=dst_ptr, kind=wx.FLAT, nbytes=nbytes, rows=1,
        run_bytes=nbytes, spitch=0, dpitch=0, src_off=0, dst_off=0)]

    with pytest.raises(wb.Weg2XchgBouncePhaseUnordered) as exc:
        wb.run_bounce_leg(descs, dst_ops, "m1", slot_bytes=SLOT, depth=1,
                          mode=wx.INJECT_AUTHORITATIVE,
                          phase=wb.PHASE_COLLECT, rendezvous=_Rendezvous(),
                          shm_root=str(tmp_path), lane="p0")
    assert "W68" in str(exc.value)
    assert "not posted full" in str(exc.value)
    assert _read(dst_ops, dst_ptr, nbytes) == b"\x11" * nbytes, (
        "the refusal must issue NOTHING -- the destination was written")


def test_M2_a_deposit_that_never_posts_full_makes_the_collect_refuse(tmp_path):
    """MUTANT M2 -- the deposit half silently skipping its post.

    Same wall from the other side: without the post the collecting rank has no
    evidence the band landed, and must refuse rather than read.
    """
    nbytes = 4096
    src_ops = FakeDeviceOps(str(tmp_path), rank=0)
    dst_ops = FakeDeviceOps(str(tmp_path), rank=1)
    src_ptr = src_ops.raw_malloc(0, nbytes)
    dst_ptr = dst_ops.raw_malloc(0, nbytes)
    _seed(src_ops, src_ptr, nbytes, 0xCD)
    _seed(dst_ops, dst_ptr, nbytes, 0x22)
    rv = _Rendezvous(drop_post=True)

    wb.run_bounce_leg([wx.XchgDesc(
        tag=TAG, src_rank=0, dst_rank=1, param_name="model.layers.0.w",
        src_ptr=src_ptr, dst_ptr=None, kind=wx.FLAT, nbytes=nbytes, rows=1,
        run_bytes=nbytes, spitch=0, dpitch=0, src_off=0, dst_off=0)],
        src_ops, "m2", slot_bytes=SLOT, depth=1,
        mode=wx.INJECT_AUTHORITATIVE, phase=wb.PHASE_DEPOSIT, rendezvous=rv,
        shm_root=str(tmp_path), lane="p0")
    assert not rv.posted

    with pytest.raises(wb.Weg2XchgBouncePhaseUnordered):
        wb.run_bounce_leg([wx.XchgDesc(
            tag=TAG, src_rank=0, dst_rank=1, param_name="model.layers.0.w",
            src_ptr=None, dst_ptr=dst_ptr, kind=wx.FLAT, nbytes=nbytes, rows=1,
            run_bytes=nbytes, spitch=0, dpitch=0, src_off=0, dst_off=0)],
            dst_ops, "m2", slot_bytes=SLOT, depth=1,
            mode=wx.INJECT_AUTHORITATIVE, phase=wb.PHASE_COLLECT,
            rendezvous=rv, shm_root=str(tmp_path), lane="p0")
    assert _read(dst_ops, dst_ptr, nbytes) == b"\x22" * nbytes


def test_a_slot_whose_byte_count_disagrees_refuses_before_the_first_copy(
        tmp_path):
    """The short-piece rule of ``run_consumer_pair`` (:1754), for this lane.

    A producer claiming a different size than this rank's own derivation means
    the two ends disagree about what is in the slot; copying the overlap would
    be the partial-layer read in a different costume.
    """
    nbytes = 4096
    dst_ops = FakeDeviceOps(str(tmp_path), rank=1)
    dst_ptr = dst_ops.raw_malloc(0, nbytes)
    _seed(dst_ops, dst_ptr, nbytes, 0x33)
    rv = _Rendezvous(lie_bytes=nbytes // 2)
    rv.post_full(slot=0, seq=0, nbytes=nbytes)

    with pytest.raises(wb.Weg2XchgBouncePhaseUnordered) as exc:
        wb.run_bounce_leg([wx.XchgDesc(
            tag=TAG, src_rank=0, dst_rank=1, param_name="model.layers.0.w",
            src_ptr=None, dst_ptr=dst_ptr, kind=wx.FLAT, nbytes=nbytes, rows=1,
            run_bytes=nbytes, spitch=0, dpitch=0, src_off=0, dst_off=0)],
            dst_ops, "short", slot_bytes=SLOT, depth=1,
            mode=wx.INJECT_AUTHORITATIVE, phase=wb.PHASE_COLLECT,
            rendezvous=rv, shm_root=str(tmp_path), lane="p0")
    assert "disagree about what is in the slot" in str(exc.value)
    assert _read(dst_ops, dst_ptr, nbytes) == b"\x33" * nbytes


# ---------------------------------------------------------------------------
# (3) THE PRODUCT ADAPTER -- slice 3
# ---------------------------------------------------------------------------


class _FakeSems:
    """The CROSS semaphores as counters, drivable OUT OF ORDER.

    A real ``SemSet`` would BLOCK instead of refusing, so the only way to prove
    the refusals can fire is a double whose counts a test sets by hand.
    """

    def __init__(self, *, empty=1, full=0):
        self.counts = {}
        self.init_empty = int(empty)
        self.init_full = int(full)
        self.log = []

    def _key(self, pair, slot, kind):
        k = (int(pair), int(slot), kind)
        if k not in self.counts:
            self.counts[k] = (self.init_empty if kind == "empty"
                              else self.init_full)
        return k

    def timedwait(self, pair, slot, kind, budget_s):
        k = self._key(pair, slot, kind)
        self.log.append(("wait", k))
        if self.counts[k] <= 0:
            return False
        self.counts[k] -= 1
        return True

    def post(self, pair, slot, kind):
        k = self._key(pair, slot, kind)
        self.log.append(("post", k))
        self.counts[k] += 1


class _FakeSlots:
    """The BOUNCE's own (seq, bytes) record -- never the region's.

    weg2xsn24 read `carries 1080 bytes` out of the region's slot record
    because `run_producer_pair` publishes the RING's counts there
    (weight_exchange_transport.py:1786). One ledger, two payloads.
    """

    def __init__(self):
        self.rows = {}

    def publish(self, *, slot, seq, nbytes, pair=None, card=None):
        self.rows[(pair, card, int(slot))] = (int(seq), int(nbytes))

    def read(self, *, slot, pair=None, card=None):
        return self.rows.get((pair, card, int(slot)), (-1, 0))


def test_pair_of_names_the_cross_pair_and_the_diagonal_separately():
    """The diagonal has no cross pair BY CONSTRUCTION, and that is not an error.

    Handing a diagonal id to ``sem_name`` is what weg2xsn9 reported 36 times as
    a bare IndexError, which is why that function refuses it by name (W15).
    """
    from sglang.srt.weg2 import weight_exchange_region as xr

    assert wb.pair_of(0, 1) == xr.CROSS_PAIRS.index((0, 1))
    assert wb.pair_of(2, 0) == xr.CROSS_PAIRS.index((2, 0))
    for r in range(xr.N_CARDS):
        assert wb.pair_of(r, r) is None


def test_the_descs_are_split_by_pair_because_the_semaphores_are_named_that_way():
    """A band mixing two pairs would post one pair's `full` for another's bytes."""
    descs = [_d(1, 2), _d(1, 2), _d(1, 2)]
    object.__setattr__(descs[0], "src_rank", 0)
    object.__setattr__(descs[0], "dst_rank", 1)
    object.__setattr__(descs[1], "src_rank", 2)
    object.__setattr__(descs[1], "dst_rank", 1)
    object.__setattr__(descs[2], "src_rank", 1)
    object.__setattr__(descs[2], "dst_rank", 1)
    groups = wb.group_descs_by_pair(descs)
    from sglang.srt.weg2 import weight_exchange_region as xr

    assert set(groups) == {xr.CROSS_PAIRS.index((0, 1)),
                           xr.CROSS_PAIRS.index((2, 1)), None}
    assert len(groups[None]) == 1, "the on-card descriptor is its own group"


def test_the_rendezvous_obeys_the_order_run_producer_pair_states():
    """fill -> sync -> publish -> post(full); never post before publish."""
    sems, slots = _FakeSems(), _FakeSlots()
    rv = wb.CrossSlotRendezvous(sems, slots, pair=0, budget_s=0.01)

    assert rv.wait_empty(slot=0, seq=0) is True
    rv.post_full(slot=0, seq=0, nbytes=4096)
    # The publish must already be visible when `full` is posted.
    kinds = [k for what, k in sems.log if what == "post"]
    assert kinds and kinds[-1][2] == "full"
    assert slots.read(slot=0, pair=0) == (0, 4096)
    assert rv.wait_full(slot=0, seq=0) == 4096
    rv.post_empty(slot=0, seq=0)
    assert sems.counts[(0, 0, "empty")] == 1


def test_a_collect_whose_producer_never_posted_gets_None_not_a_stale_count():
    sems, slots = _FakeSems(full=0), _FakeSlots()
    slots.publish(slot=0, seq=0, nbytes=9999, pair=0)
    rv = wb.CrossSlotRendezvous(sems, _FakeSlots(), pair=0, budget_s=0.01)
    assert rv.wait_full(slot=0, seq=0) is None, (
        "an unposted slot must read None even when the record still carries a "
        "byte count -- the record is not the handshake")


def test_a_deposit_cannot_claim_a_slot_the_consumer_has_not_drained(tmp_path):
    """The producing side of the unordered-read hazard, refused by name."""
    nbytes = 4096
    src_ops = FakeDeviceOps(str(tmp_path), rank=0)
    src_ptr = src_ops.raw_malloc(0, nbytes)
    sems, slots = _FakeSems(empty=0), _FakeSlots()   # nothing drained
    rv = wb.CrossSlotRendezvous(sems, _FakeSlots(), pair=0, budget_s=0.01)
    with pytest.raises(wb.Weg2XchgBouncePhaseUnordered) as exc:
        wb.run_bounce_leg([wx.XchgDesc(
            tag=TAG, src_rank=0, dst_rank=1, param_name="model.layers.0.w",
            src_ptr=src_ptr, dst_ptr=None, kind=wx.FLAT, nbytes=nbytes, rows=1,
            run_bytes=nbytes, spitch=0, dpitch=0, src_off=0, dst_off=0)],
            src_ops, NONCE, slot_bytes=SLOT, depth=1,
            mode=wx.INJECT_AUTHORITATIVE, phase=wb.PHASE_DEPOSIT,
            rendezvous=rv, shm_root=str(tmp_path), lane="p0")
    assert "still full when this deposit tried to claim it" in str(exc.value)


def test_the_adapter_passes_deposit_on_source_and_collect_on_the_import_hooks():
    """MUTANT: ignore `hook` and pass `both` -- this test dies.

    It drives the REAL adapter body with a recording stand-in for
    `run_bounce_leg`, so what is asserted is the argument the product actually
    passes, not a restatement of the intent.
    """
    from sglang.srt.managers.scheduler_components import weight_updater as wu

    seen = []

    class _Stub:
        _weg2_xchg_bounce_leg = wu.SchedulerWeightUpdaterManager._weg2_xchg_bounce_leg

        # #1358: the adapter reads its own identity for the host-slot lines.
        def _weg2_group_name(self):
            return "P"

        def _weg2_rank(self):
            return 0

    rendezvous_seen = []

    def _fake_leg(descs, ops, nonce, **kw):
        rv = kw.get("rendezvous")
        rendezvous_seen.append(rv)
        seen.append((kw.get("phase"), rv is not None,
                     [(d.src_rank, d.dst_rank) for d in descs]))
        return None

    descs = [_d(1, 2), _d(1, 2)]
    object.__setattr__(descs[0], "src_rank", 0)
    object.__setattr__(descs[0], "dst_rank", 1)
    object.__setattr__(descs[1], "src_rank", 1)   # on-card
    object.__setattr__(descs[1], "dst_rank", 1)

    import sglang.srt.weg2.weight_exchange_bounce as real

    orig = real.run_bounce_leg
    real.run_bounce_leg = _fake_leg
    try:
        for hook, expect in (("source", wb.PHASE_DEPOSIT),
                             ("destination", wb.PHASE_COLLECT),
                             ("authoritative", wb.PHASE_COLLECT)):
            seen.clear()
            rendezvous_seen.clear()
            _Stub()._weg2_xchg_bounce_leg(
                descs=descs, ops=None, boot_nonce=NONCE, slot_bytes=SLOT,
                depth=1, mode=wx.INJECT_AUTHORITATIVE, hook=hook,
                region=object(), sems=_FakeSems())
            phases = {p for p, _rv, _r in seen}
            # EVERY LEG IS PHASED, DIAGONAL INCLUDED -- and this assertion was
            # the OPPOSITE one commit ago, which is the finding worth keeping.
            # The brief said `both` stays "for the diagonal"; the provider
            # smoke refuted it on the first run: on the SOURCE hook the
            # diagonal group raised `_missing_pointer: ... has no destination
            # pointer`, because on-card means ONE CARD and TWO PROCESSES -- the
            # co-located pair of ranks -- so the peer's pages are as
            # unreachable there as across a link. `both` has no cross-process
            # caller at all; it is the single-process form for tests and tools.
            assert phases == {expect}, (hook, phases)
            assert wb.PHASE_BOTH not in phases, (
                "a flip leg may never run `both`: it asks this rank for the "
                "peer's device address, which is weg2xsn20's wall")
            # Both kinds carry a handshake; they differ in WHICH one -- the
            # cross lane's 24 by pair, the diagonal's 12 by card (#1334).
            assert len(seen) == 2 and all(s[1] for s in seen), seen
            pairs = {tuple(s[2][0]) for s in seen}
            assert pairs == {(0, 1), (1, 1)}, seen
            kinds = {type(rv).__name__ for rv in rendezvous_seen}
            assert kinds == {"CrossSlotRendezvous"}, kinds
            # ONE class, two keyings: the cross pair by pair index, the
            # diagonal by CARD (#1334). Slice 3 had a second class for the
            # diagonal whose byte check could not go red; it is gone.
            assert {(rv.pair is None) for rv in rendezvous_seen} == {True, False}
    finally:
        real.run_bounce_leg = orig


def test_without_a_handshake_a_cross_leg_is_refused_never_downgraded(tmp_path):
    """THE RATCHET: no cross leg may fall back to `both`.

    `both` asks this rank for the PEER's device address -- weg2xsn20's wall.
    The adapter takes the unsplit form when region/sems are missing, and
    `run_bounce_leg` must then refuse the cross pair by name.
    """
    from sglang.srt.managers.scheduler_components import weight_updater as wu

    class _Stub:
        _weg2_xchg_bounce_leg = wu.SchedulerWeightUpdaterManager._weg2_xchg_bounce_leg

        # #1358: the adapter reads its own identity for the host-slot lines.
        def _weg2_group_name(self):
            return "P"

        def _weg2_rank(self):
            return 0

    descs = [wx.XchgDesc(
        tag=TAG, src_rank=0, dst_rank=1, param_name="model.layers.0.w",
        src_ptr=None, dst_ptr=0x2000, kind=wx.FLAT, nbytes=16, rows=1,
        run_bytes=16, spitch=0, dpitch=0, src_off=0, dst_off=0)]
    # No region, no sems -> the unsplit form -> W74 on the missing source,
    # which is a REFUSAL and not a silently-run `both` leg.
    with pytest.raises(Exception) as exc:
        _Stub()._weg2_xchg_bounce_leg(
            descs=descs, ops=None, boot_nonce=NONCE, slot_bytes=SLOT, depth=1,
            mode=wx.INJECT_AUTHORITATIVE, hook="authoritative",
            region=None, sems=None)
    assert "W74" in str(exc.value) or "W68" in str(exc.value), str(exc.value)


def test_the_sems_cache_is_a_declared_field_because_the_manager_has_slots():
    """weg2xsn7 lost 24 of 24 legs to exactly this, on both groups."""
    import dataclasses

    from sglang.srt.managers.scheduler_components import weight_updater as wu

    names = {f.name for f in
             dataclasses.fields(wu.SchedulerWeightUpdaterManager)}
    assert "_weg2_xchg_sems_cache" in names


# ---------------------------------------------------------------------------
# (4) #1358 -- THE HOST-SLOT EMITTER, IN THE PRODUCTION PATH
# ---------------------------------------------------------------------------


def test_the_host_slot_emitter_lives_in_product_code_not_in_a_script():
    """IT DID NOT, AND THAT IS WHY ACCEPTANCE E2 COULD NOT PASS.

    `WEG2-XCHG-HOST-SLOT` existed only as a `print` in
    scripts/weg2/xchg_leg_replay.py, so no boot could ever have carried it and
    #1358's host ratchet (+1.2..1.7 GiB anon per P->D leg; xsn25 +4.65 GiB
    after ONE flip, then oom_kill) had no instrument at all. Built-but-not-
    wired, in my own instrument this time.
    """
    import inspect

    src = inspect.getsource(wb)
    assert wb.HOST_SLOT_MARKER == "WEG2-XCHG-HOST-SLOT"
    assert wb.HOST_SLOT_LEG_MARKER == "WEG2-XCHG-HOST-SLOT-LEG"
    # Emitted through the module logger, never printed.
    assert "print(" not in inspect.getsource(wb._host_slot_event)
    assert "logger.info" in src


def test_every_slot_says_its_bytes_at_alloc_and_at_free(tmp_path):
    """One line per slot per event, and the running total is the state AFTER."""
    wb._LIVE_SLOTS.clear()
    lines = []
    ops = FakeDeviceOps(str(tmp_path), rank=0)
    b = wb.LayerBounce(ops, NONCE, slot_bytes=SLOT, depth=1, create=True,
                       shm_root=str(tmp_path), lane="p0", group="P", rank=1,
                       leg="e1/source", slot_index=0)
    assert wb.host_slot_live_bytes() == SLOT
    b.close()
    assert wb.host_slot_live_bytes() == 0, "the free must clear the registry"

    # The line's own shape, built directly so the assertion names the fields.
    line = wb.host_slot_line(event="alloc", path="/x", nbytes=123, group="D",
                             rank=2, leg="e1/destination", slot=1,
                             region="shm,pinned")
    for field in ("group=D", "rank=2", "leg=e1/destination", "slot=1",
                  "event=alloc", "bytes=123", "region=shm,pinned",
                  "total_live_bytes="):
        assert field in line, (field, line)


def test_the_leg_summary_reads_zero_on_the_happy_path_and_nonzero_on_a_leak():
    """`bytes_live_after != 0` IS the leak suspicion #1358 asked for."""
    wb._LIVE_SLOTS.clear()
    clean = wb.host_slot_leg_line(group="P", rank=0, leg="e1/source",
                                  slots_before=0)
    assert "slots_live_after=0" in clean and "bytes_live_after=0" in clean

    # A PLANTED FORGOTTEN FREE: the alloc happened, the free did not.
    wb._host_slot_event("alloc", "/leaked", 4096, group="P", rank=0,
                        leg="e1/source", slot=0)
    leaked = wb.host_slot_leg_line(group="P", rank=0, leg="e1/source",
                                   slots_before=0)
    assert "slots_live_after=1" in leaked
    assert "bytes_live_after=4096" in leaked, leaked
    wb._LIVE_SLOTS.clear()


def test_a_free_with_no_alloc_says_so_rather_than_going_quiet():
    """The accounting and the buffer parting company is itself a finding."""
    wb._LIVE_SLOTS.clear()
    seen = []
    wb._host_slot_event("free", "/never-allocated", 99, log=seen.append)
    assert "unmatched-free" in seen[0], seen
    wb._LIVE_SLOTS.clear()


def test_unsplit_form_refuses_a_cross_leg_1330():
    """A CROSS leg without a handshake is REFUSED, not run as ``both``.

    THE COMMENT CLAIMED THIS AND THE CODE DID NOT DO IT. `_require_rendezvous`
    returns immediately for `PHASE_BOTH` (weight_exchange_bounce.py:937-938)
    and nothing below it separates a cross descriptor from a diagonal one, so
    before this test a cross leg that reached the unsplit form ran as `both`,
    asked its own rank for the PEER's device address, and died as
    `W74 ... src_resolved=0/N` -- weg2xsn24's wall. It is reachable on the
    cutover path: the single production caller
    (weight_updater.py:1175) passes `sems=self._weg2_xchg_sems()`, which
    returns None when its region is None or when its `except BaseException`
    swallows anything (weight_updater.py:3166-3176).
    """
    from sglang.srt.managers.scheduler_components import weight_updater as wu

    class _Stub:
        _weg2_xchg_bounce_leg = wu.SchedulerWeightUpdaterManager._weg2_xchg_bounce_leg

        def _weg2_group_name(self):
            return "P"

        def _weg2_rank(self):
            return 0

    cross = _d(1, 2)
    object.__setattr__(cross, "src_rank", 0)
    object.__setattr__(cross, "dst_rank", 1)
    with pytest.raises(wx.Weg2XchgPlanDisagree) as ei:
        _Stub()._weg2_xchg_bounce_leg(
            descs=[cross], ops=None, boot_nonce=NONCE, slot_bytes=SLOT,
            depth=1, mode=wx.INJECT_AUTHORITATIVE, hook="authoritative",
            region=object(), sems=None)
    assert "W68" in str(ei.value) and "cross cards" in str(ei.value)

    # THE DIAGONAL STILL TAKES THE UNSPLIT FORM: it holds both pointers in one
    # process, which is the one case `both` is honest for.
    diag = _d(1, 2)
    object.__setattr__(diag, "src_rank", 1)
    object.__setattr__(diag, "dst_rank", 1)
    import sglang.srt.weg2.weight_exchange_bounce as real

    orig, seen = real.run_bounce_leg, []
    real.run_bounce_leg = lambda descs, ops, nonce, **kw: seen.append(
        kw.get("phase")) or None
    try:
        _Stub()._weg2_xchg_bounce_leg(
            descs=[diag], ops=None, boot_nonce=NONCE, slot_bytes=SLOT,
            depth=1, mode=wx.INJECT_AUTHORITATIVE, hook="authoritative",
            region=object(), sems=None)
    finally:
        real.run_bounce_leg = orig
    assert seen == [None], seen


def test_no_production_caller_passes_phase_both_1330():
    """AST RATCHET: no shipped call site asks for ``phase=both``.

    The phase is derived from the hook at ONE site
    (weight_updater.py:3247-3248). A second site that named `both` explicitly
    would re-open exactly the wall this slice closed, and a grep would not see
    it through an alias -- so this walks the tree instead.
    """
    import ast
    import pathlib

    root = pathlib.Path(wx.__file__).resolve().parents[3]
    offenders = []
    for path in sorted(root.rglob("*.py")):
        if "/test" in str(path) or "/scripts/" in str(path):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:            # not ours to parse
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for kw_ in node.keywords:
                if kw_.arg != "phase":
                    continue
                v = kw_.value
                if isinstance(v, ast.Constant) and v.value == "both":
                    offenders.append(f"{path}:{node.lineno} phase='both'")
                if isinstance(v, ast.Attribute) and v.attr == "PHASE_BOTH":
                    offenders.append(f"{path}:{node.lineno} PHASE_BOTH")
                if isinstance(v, ast.Name) and v.id == "PHASE_BOTH":
                    offenders.append(f"{path}:{node.lineno} PHASE_BOTH")
    assert not offenders, offenders


def test_serving_env_path_yields_a_real_semaphore_set_1330(monkeypatch):
    """THE SERVING ARGV PATH, driven hermetically -- no boot, no GPU.

    The refusal added beside it would merely RENAME the wall if the armed
    serving form never built semaphores in the first place. It does: the
    launcher's own publisher `prepare_xchg_env` creates the region and puts
    `SGLANG_WEG2_XCHG_REGION` / `_BOOT` into every rank's environment
    (launcher.py:3463 onward, via `prepare_xchg_region`), and those are the
    exact two variables `_weg2_shadow_region` reads
    (weight_updater.py:2162-2165). Measured here end to end.
    """
    from sglang.srt.weg2 import launcher as L
    from sglang.srt.weg2 import weight_exchange_region as xr
    from sglang.srt.managers.scheduler_components import weight_updater as wu

    class _Log:
        def __call__(self, *a, **k):
            pass

        info = warn = error = __call__

    nonce = "b4nsemsprobe"
    env = L.prepare_xchg_env(_Log(), nonce, "exchange")
    assert env.get(xr.ENV_REGION_PATH) and env.get(xr.ENV_REGION_BOOT) == nonce
    try:
        for k, v in env.items():
            monkeypatch.setenv(str(k), str(v))

        class _M:
            _weg2_shadow_region = wu.SchedulerWeightUpdaterManager._weg2_shadow_region
            _weg2_xchg_sems = wu.SchedulerWeightUpdaterManager._weg2_xchg_sems

        m = _M()
        assert m._weg2_shadow_region() is not None
        assert m._weg2_xchg_sems() is not None
    finally:
        L.teardown_xchg_region(_Log(), nonce)


def test_unavailable_semaphores_are_named_not_swallowed_1330(monkeypatch, caplog):
    """#505: the swallowed exception is logged BY NAME, and refused when armed.

    `except BaseException: sems = None` (weight_updater.py:3173, before this)
    is why a lost handshake reached the boot as `W74 ... src_resolved=0/N` --
    an ADDRESS complaint for a HANDSHAKE cause.
    """
    import logging

    from sglang.srt.weg2 import weight_exchange_transport as tp
    from sglang.srt.managers.scheduler_components import weight_updater as wu

    def _boom(*a, **k):
        raise OSError("sem_open refused by the probe")

    monkeypatch.setattr(tp, "SemSet", _boom)

    class _M:
        _weg2_xchg_sems = wu.SchedulerWeightUpdaterManager._weg2_xchg_sems

        def _weg2_shadow_region(self):
            return type("_R", (), {"boot_nonce": "probe"})()

    # ARMED: refused by name, never silently None.
    monkeypatch.setattr(wx, "exchange_armed", lambda: True)
    with caplog.at_level(logging.INFO):
        with pytest.raises(wx.Weg2XchgPlanDisagree) as ei:
            _M()._weg2_xchg_sems()
    assert "OSError" in str(ei.value)
    assert "sem_open refused by the probe" in str(ei.value)
    armed_text = " ".join(r.getMessage() for r in caplog.records)
    assert "WEG2-XCHG-SEMS-UNAVAILABLE" in armed_text, armed_text
    assert "OSError" in armed_text and "armed=True" in armed_text, armed_text

    # UNARMED: the observer behaviour the `None` was written for, but LOUD.
    monkeypatch.setattr(wx, "exchange_armed", lambda: False)
    caplog.clear()
    with caplog.at_level(logging.INFO):
        assert _M()._weg2_xchg_sems() is None
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "WEG2-XCHG-SEMS-UNAVAILABLE" in text and "OSError" in text


def test_the_inject_line_names_its_phase_1330():
    """The INJECT line says WHICH HALF ran -- absence is not evidence.

    weg2xsn25 read three P legs as "no phase field, therefore unsplit". They
    had run `collect`: the W68 texts they carried
    (`was not posted full` / `carries N bytes`) are reachable ONLY under
    `rendezvous is not None and phase == PHASE_COLLECT`
    (weight_exchange_bounce.py:1176). The field was simply never printed.
    """
    for phase, want in ((wb.PHASE_DEPOSIT, "phase=deposit"),
                        (wb.PHASE_COLLECT, "phase=collect"),
                        ("", "phase=unsplit")):
        v = wb.InjectVerdict(mode=wx.INJECT_SHADOW, pieces=0,
                             bytes_compared=0, mismatches=0, phase=phase)
        assert want in v.line(), (phase, v.line())


def test_both_halves_of_the_handshake_have_a_production_caller_1330():
    """RATCHET: a depositing call site EXISTS, not only a collecting one.

    weg2xsn25 read `SEAM-DIGEST MATCH 0/6` with no phase bug anywhere: there
    was exactly ONE production caller, on the waking group, hard-coding
    `hook="authoritative"` -- which `leg_direction` derives as `collect`. Every
    rank that ran a leg collected and nobody deposited, so the collect legs
    waited on a band no one would post (`carries 0 bytes` against a derivation
    of 756323776). A missing caller is invisible to every test that asserts
    what a caller does.
    """
    import ast
    import pathlib

    src = pathlib.Path(
        "python/sglang/srt/managers/scheduler_components/weight_updater.py")
    if not src.exists():                       # worktree-relative fallback
        import sglang.srt.managers.scheduler_components.weight_updater as _wu
        src = pathlib.Path(_wu.__file__)
    tree = ast.parse(src.read_text(encoding="utf-8"))
    hooks = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if not (isinstance(fn, ast.Attribute)
                and fn.attr == "_weg2_xchg_bounce_leg"):
            continue
        for kw_ in node.keywords:
            if kw_.arg != "hook":
                continue
            v = kw_.value
            if isinstance(v, ast.Constant):
                hooks.add(str(v.value))
            elif isinstance(v, ast.Attribute):
                hooks.add(v.attr)               # sh.HOOK_SOURCE
    assert "HOOK_SOURCE" in hooks or "source" in hooks, (
        "no production call site runs the SOURCE hook: the exchange has no "
        f"depositor and cannot move a byte. hooks seen: {sorted(hooks)}")
    assert "authoritative" in hooks or "destination" in hooks, (
        f"no importing call site: nothing collects. hooks seen: {sorted(hooks)}")


def test_shadow_and_exchange_semaphore_names_are_disjoint_1330():
    """RATCHET FOR (B): both sets built from ONE region, names disjoint.

    `sem_name` (weight_exchange_region.py:1766) is total in its nonce and
    `SemSet.__init__` (weight_exchange_transport.py:934) takes nothing else, so
    two SemSets built from the same `region.boot_nonce` name the SAME 24
    semaphores -- identically, not probably. That is what boot weg2xsn25 hit:
    the observer's sample traffic (`classes=1 subset=A_log`, 0.00 MiB) took
    `empty` and posted `full`, and the exchange's collect legs then waited on a
    handshake already consumed.

    Separation therefore has to happen in the NONCE, and every derivation that
    hangs off it must move together -- which is why `shadow_nonce` is one
    function and `all_region_sem_names` (the list both `create_semaphores` and
    `unlink_semaphores` read) carries the observer's 24 rather than a second
    creator existing anywhere.
    """
    from sglang.srt.weg2 import weight_exchange_region as xr

    boot = "b4ndisjoint"
    exchange = set(xr.all_sem_names(boot))
    observer = set(xr.all_sem_names(xr.shadow_nonce(boot)))
    assert len(exchange) == 24 and len(observer) == 24
    assert exchange.isdisjoint(observer), sorted(exchange & observer)
    # AND BOTH ARE CREATED AND TORN DOWN BY THE ONE LIST, so the observer's set
    # cannot be the half that nobody unlinks.
    every = set(xr.all_region_sem_names(boot))
    assert exchange <= every and observer <= every, sorted(
        (exchange | observer) - every)


def test_the_bounce_leg_line_never_omits_the_inject_field_1330():
    """#1336's class: an absent field is a silence the reader fills in.

    `inject=` used to vanish when no verdict existed, so a SHADOW leg that
    graded nothing looked exactly like an AUTHORITATIVE leg, which has no
    comparison by construction (`comparing = mode == INJECT_SHADOW`). The two
    now say which silence they are, and only one of them is a finding.
    """
    mk = lambda mode: wb.BounceResult(  # noqa: E731
        units=0, bands=0, widest_unit_key=("", "u"), widest_unit_bytes=0,
        widest_run_bytes=0, slot_bytes=SLOT, depth=1, host_bytes_peak=0,
        deposited_bytes=0, collected_bytes=0, planned_bytes=0,
        deposit_ms=0.0, collect_ms=0.0, overlap="none", mode=mode)
    assert "inject=by-design-authoritative" in mk(wx.INJECT_AUTHORITATIVE).line()
    assert "inject=NOTHING-COMPARED" in mk(wx.INJECT_SHADOW).line()


def test_price_equals_allocation_for_the_assemble_buffer_1330(monkeypatch):
    """ONE AUTHORITY: what the ledger charges IS what the leg allocates.

    They were two expressions for one buffer:
      priced     xchg_bounce.py:189            widest * depth
      allocated  weight_exchange_bounce.py:1119 depth + (1 if comparing)
                 weight_exchange_bounce.py:502  slot_bytes * depth
    and the difference was exactly one widest layer whenever `mode=shadow` --
    721.4 MiB the ledger never charged, on EVERY S6I boot, since S6I boots
    shadow first.

    The anchor is boot weg2xsn25's own line, not a number invented here:
    `WEG2-XCHG-HOST-SLOT ... event=alloc bytes=2268971328`.
    """
    from sglang.srt.weg2 import xchg_bounce as xb

    WIDEST = 756323776          # weg2xsn25's widest layer, from its own census
    XSN25_ALLOCATED = 2268971328  # ... and the bytes it allocated for it

    for mode, comparing, want_slots in ((wx.INJECT_SHADOW, True, 3),
                                        (wx.INJECT_AUTHORITATIVE, False, 2)):
        monkeypatch.setenv(wx.INJECT_ENV, mode)
        # THE ARM IS READ BY THE AUTHORITY ITSELF -- no caller may hold an
        # opinion about it, which is the shape this closes.
        assert xb.assemble_slots(2) == want_slots, mode
        priced = xb.bounce_terms(
            bytes_per_direction=WIDEST * 64, n_layers=64,
            widest_layer_bytes=WIDEST, pairs=3, depth=2,
            slot_bytes=134217728).buffer_bytes
        allocated = WIDEST * xb.assemble_slots(2, comparing=comparing)
        assert priced == allocated, (mode, priced, allocated)

    monkeypatch.setenv(wx.INJECT_ENV, wx.INJECT_SHADOW)
    assert xb.assemble_buffer_bytes(WIDEST, 2) == XSN25_ALLOCATED


def test_no_second_expression_for_the_shadow_slot_1330():
    """AST RATCHET: the "one extra slot" idiom exists in ONE place.

    The shape is `<anything> + (1 if <cond> else 0)`. A second one would
    re-open the priced-vs-allocated gap silently, and grepping for the NUMBER
    would not find it -- the expression is what has to be unique.
    """
    import ast
    import pathlib

    from sglang.srt.weg2 import xchg_bounce as xb

    root = pathlib.Path(xb.__file__).resolve().parent
    owners = []
    for path in sorted(root.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.BinOp)
                    and isinstance(node.op, ast.Add)
                    and isinstance(node.right, ast.IfExp)):
                continue
            body, orelse = node.right.body, node.right.orelse
            if (isinstance(body, ast.Constant) and body.value == 1
                    and isinstance(orelse, ast.Constant) and orelse.value == 0):
                owners.append(f"{path.name}:{node.lineno}")
    assert len(owners) == 1 and owners[0].startswith("xchg_bounce.py:"), owners


def test_both_product_call_sites_pass_the_arm_terms_1330(monkeypatch):
    """EXECUTION SMOKE OF BOTH CALL SITES, through the real weight_updater.

    Boot weg2xsn26 died because ONE of the two product call sites passed a
    literal `terms=None`. The deposit path (`hook=source`, the sleeping group)
    raised nine times on D, took the group fence down and ended the boot at
    W17 Weg2GroupDead with 0 legs and 0 SEAM-DIGEST lines -- while the collect
    path (`hook=authoritative`) passed the arm's decision and was fine. A test
    that drives `run_bounce_leg` in isolation cannot see that: the defect is in
    what the CALLER hands it, so the caller is what has to run.

    Neither the xsn25 replay nor any unit test reached the deposit call site,
    which is why "built, never exercised at the D call site" survived a boot.
    """
    from sglang.srt.managers.scheduler_components import weight_updater as wu
    from sglang.srt.weg2 import xchg_bounce as xb

    terms = xb.bounce_terms(
        bytes_per_direction=64 * 4096, n_layers=64, widest_layer_bytes=4096,
        pairs=3, depth=2, slot_bytes=4096)
    monkeypatch.setenv(xb.ENV_BOUNCE_TERMS, xb.publish_terms(terms))
    monkeypatch.setenv(wx.WEIGHT_SOURCE_ENV, wx.WEIGHT_SOURCE_EXCHANGE)

    seen = []

    class _Stub:
        _weg2_xchg_deposit_before_sleep = (
            wu.SchedulerWeightUpdaterManager._weg2_xchg_deposit_before_sleep)

        def _weg2_group_name(self):
            return "P"

        def _weg2_rank(self):
            return 0

        def _weg2_device_index(self):
            return 0

        def _weg2_shadow_region(self):
            return object()

        def _weg2_xchg_sems(self):
            return object()

        def _weg2_xchg_device_ops(self):
            return object()

        def _weg2_shadow_plan(self, hook, group, rank, **kw):
            return type("_P", (), {"descs": ()})(), ""

        def _weg2_xchg_bounce_leg(self, **kw):
            seen.append(kw)

    monkeypatch.setenv("SGLANG_WEG2_XCHG_BOOT", "b4ndepositsmoke")
    _Stub()._weg2_xchg_deposit_before_sleep()

    assert seen, "the deposit call site did not run at all"
    kw = seen[0]
    assert kw["hook"] == "source", kw["hook"]
    # THE ONE ASSERTION THE BOOT NEEDED: a price, not None.
    assert kw["terms"] is not None, (
        "the deposit path passed terms=None -- run_bounce_leg derives no size "
        "of its own and refuses by W68, which is how weg2xsn26 died")
    assert int(kw["terms"].widest_layer_bytes) == 4096, kw["terms"]
