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
