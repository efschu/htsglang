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
        wb.run_bounce_leg([], ops=None, boot_nonce="pin", slot_bytes=SLOT,
                          depth=1, phase="deposit-ish")
    assert "unknown bounce phase" in str(exc.value)


def test_a_phased_leg_without_a_handshake_is_refused():
    """M2's sibling: running a phase optimistically IS the unordered read."""
    for phase in (wb.PHASE_DEPOSIT, wb.PHASE_COLLECT):
        with pytest.raises(wb.Weg2XchgBouncePhaseUnordered) as exc:
            wb.run_bounce_leg([_d(0x1000, 0x2000)], ops=None, boot_nonce="pin",
                              slot_bytes=SLOT, depth=1, phase=phase,
                              rendezvous=None)
        assert "W68" in str(exc.value)
    # `both` needs none and gets past the guard (it fails later, on ops=None).
    with pytest.raises(Exception) as exc:
        wb.run_bounce_leg([_d(0x1000, 0x2000)], ops=None, boot_nonce="pin",
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
                      rendezvous=rv, shm_root=str(tmp_path))
    assert rv.posted, "the deposit must post `full` after its sync"

    wb.run_bounce_leg(collect_descs, dst_ops, nonce, slot_bytes=SLOT, depth=1,
                      mode=wx.INJECT_AUTHORITATIVE, phase=wb.PHASE_COLLECT,
                      rendezvous=rv, shm_root=str(tmp_path))
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
                          shm_root=str(tmp_path))
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
        shm_root=str(tmp_path))
    assert not rv.posted

    with pytest.raises(wb.Weg2XchgBouncePhaseUnordered):
        wb.run_bounce_leg([wx.XchgDesc(
            tag=TAG, src_rank=0, dst_rank=1, param_name="model.layers.0.w",
            src_ptr=None, dst_ptr=dst_ptr, kind=wx.FLAT, nbytes=nbytes, rows=1,
            run_bytes=nbytes, spitch=0, dpitch=0, src_off=0, dst_off=0)],
            dst_ops, "m2", slot_bytes=SLOT, depth=1,
            mode=wx.INJECT_AUTHORITATIVE, phase=wb.PHASE_COLLECT,
            rendezvous=rv, shm_root=str(tmp_path))
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
            rendezvous=rv, shm_root=str(tmp_path))
    assert "disagree about what is in the slot" in str(exc.value)
    assert _read(dst_ops, dst_ptr, nbytes) == b"\x33" * nbytes
