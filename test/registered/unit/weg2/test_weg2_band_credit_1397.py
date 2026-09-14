# SPDX-License-Identifier: Apache-2.0
"""#1397 -- Option 3 (DESIGN_option3_band_credit_0914.md): BAND-GRANULARITY
slot reuse in a CROSS bounce lane, the alternative the #1374 W68 refusal has
named in its own text since the day it was written ("reduce the pause
granularity below the tag") and never built until now.

USER ORDER 2026-09-14, verbatim, on the finding that the alternative was
never built: *"ja dann - bauen!"*.

WHAT THIS FILE PINS.

(1) SIZING (pure, `xchg_bounce.py`): `band_credit=False` (the default) is
    BYTE-IDENTICAL to every pre-#1397 caller -- the ticket's own measured
    number, boot weg2xsn31/3's `xchg_bounce=15.75 GiB`
    (`slot_bytes=128 MiB`, `max_tag_bytes=2907 MiB`, `depth=2`, 5 lanes,
    3 pairs), is pinned as the RED-FIRST baseline. `band_credit=True` +
    `n_cross_lanes` prices the CROSS share at `depth`-order instead of the
    whole tag, and does NOT get swallowed by the `max(buffer_bytes,
    lane_buffer_bytes)` guard `total_bytes` still uses for the diagonal
    share (the #1386 trap this ticket's own text names) -- pinned by an
    explicit assertion that `cross_lane_buffer_bytes < buffer_bytes` on
    this rig's own figures, so a max() over the two would have picked the
    WRONG (bigger, unallocated) one.

(2) THE MECHANISM (`weight_exchange_bounce.py`): `CrossSlotRendezvous`'s new
    `post_band_drained`/`wait_band_drained`/`prime_band_drain`, and
    `run_bounce_leg` wiring them in for a CROSS phased leg only. Bands land
    in order and a slot is reused only after its drain credit -- proven on
    REAL bytes through `FakeDeviceOps` (cross-process by construction, see
    that fixture's own docstring), never by a call count.

(3) THE TWO MANDATORY MUTANTS, both RED:
      M1  the credit check is bypassed (a rendezvous double that always says
          "drained")            -> corruption: a stale band's bytes collected
      M2  the credit is posted BEFORE the collect's own device copy has
          landed ("too early")  -> corruption: the depositor's overwrite
                                    lands INSIDE the collect's still-open
                                    copy window, SILENTLY (no counter, no
                                    refusal -- the danger direction named in
                                    the ticket)

(4) BYTE IDENTITY OF WHAT MOVES: the SAME descriptor set, moved once under
    today's whole-tag sizing and once under Option 3's band-credit sizing,
    lands BYTE-IDENTICAL bytes at the destination. The BUFFER size changes
    on purpose (section 10 of the design); the BYTES that cross it do not.

THE HARNESS IS THE PRODUCT'S OWN, same doctrine as #1330/#1374/#1358's own
suites: `FakeDeviceOps` moves REAL bytes through REAL host pages, and
`CrossSlotRendezvous` runs over REAL POSIX semaphores
(`weight_exchange_region.create_semaphores` / `weight_exchange_transport.
SemSet`) and a REAL `BounceSlots` record, never a call-count mock, for
anything the two mandatory mutants must be able to falsify.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import sys

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_weg2_xchg_transport_1273 import FakeDeviceOps  # noqa: E402

from sglang.srt.weg2 import weight_exchange as wx  # noqa: E402
from sglang.srt.weg2 import weight_exchange_bounce as wb  # noqa: E402
from sglang.srt.weg2 import weight_exchange_region as xr  # noqa: E402
from sglang.srt.weg2 import weight_exchange_transport as tp  # noqa: E402
from sglang.srt.weg2 import xchg_bounce as xb  # noqa: E402

MIB = 1024 * 1024
GIB = 1024 * MIB
TAG = "weights_3"
PAIR = 0

#: PID-SCOPED, #1344's argument: a bounce keyed on a fixed literal writes
#: /dev/shm paths no per-test cleanup can claim ownership of.
NONCE = f"bc{os.getpid()}"

# The ticket's own headline figures (boot weg2xsn31/3, `xchg_bounce=15.75
# GiB`): widest tag 2907 MiB, slot 128 MiB, depth 2, 5 lanes, 3 pairs.
XSN31 = dict(bytes_per_direction=29119878266, n_layers=64,
             widest_layer_bytes=756323776, pairs=3, depth=2,
             n_lanes=5, max_tag_bytes=2907 * MIB)


@pytest.fixture(autouse=True)
def _sweep_own_shm():
    """Remove THIS interpreter's residue, and only ever its own."""
    yield
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


# ===========================================================================
# (1) SIZING -- xchg_bounce.py, pure, no device, no semaphore.
# ===========================================================================


class TestSizingIsByteIdenticalByDefault:
    def test_band_credit_off_reproduces_the_ticket_measured_1575_gib(self):
        """RED-FIRST BASELINE: today's number, `band_credit` untouched."""
        t = xb.bounce_terms(**XSN31, slot_bytes=128 * MIB)
        assert t.band_credit is False
        assert t.n_cross_lanes_priced == 0
        assert t.lane_slots == 24                       # ceil(2907/128)+shadow
        assert t.total_bytes == pytest.approx(15.75 * GIB, rel=1e-9)

    def test_band_credit_true_without_n_cross_lanes_changes_nothing(self):
        """The SAFE default: turning the flag on alone must never shrink
        anything -- a caller must ALSO count its cross lanes."""
        off = xb.bounce_terms(**XSN31, slot_bytes=128 * MIB)
        on_but_unstated = xb.bounce_terms(**XSN31, slot_bytes=128 * MIB,
                                          band_credit=True)
        assert on_but_unstated.total_bytes == off.total_bytes
        assert on_but_unstated.n_cross_lanes_priced == 0

    def test_the_round_trip_carries_the_two_new_fields(self):
        """A rank that only recomputes the PUBLISHED inputs must reach the
        SAME total the launcher priced -- #1358's own rule, extended."""
        t = xb.bounce_terms(**XSN31, slot_bytes=128 * MIB, band_credit=True,
                            n_cross_lanes=3)
        back = xb.read_published_terms(xb.publish_terms(t))
        assert back.band_credit is True
        assert back.n_cross_lanes == 3
        assert back.total_bytes == t.total_bytes


class TestBandCreditShrinksOnlyTheCrossShare:
    def test_cross_lane_slots_is_depth_order_not_tag_order(self):
        t = xb.bounce_terms(**XSN31, slot_bytes=128 * MIB)
        # depth=2, shadow mode (this rig's own default, `wx.inject_mode()`
        # unset) -> assemble_slots(2, comparing=True) == 3, not
        # ceil(2907 MiB / 128 MiB) + shadow == 24.
        assert t.cross_lane_slots == xb.assemble_slots(2, comparing=True)
        assert t.cross_lane_slots < t.lane_slots

    def test_the_max_with_buffer_bytes_would_have_swallowed_the_saving(self):
        """THE #1386 TRAP, MEASURED ON THIS RIG'S OWN FIGURES: pinned so a
        future edit cannot silently re-introduce `max(buffer_bytes,
        cross_lane_buffer_bytes)` for the cross share."""
        t = xb.bounce_terms(**XSN31, slot_bytes=128 * MIB)
        assert t.cross_lane_buffer_bytes < t.buffer_bytes, (
            "if this ever flips, `total_bytes`'s max() would start picking "
            "buffer_bytes again and the whole saving would vanish silently")

    def test_all_cross_naive_ceiling_at_slot_128(self):
        """The ticket's own second question, upper bound: EVERY one of the 5
        lanes counted cross (never actually safe unless a boot's topology
        truly has zero diagonal traffic -- see the design doc)."""
        t = xb.bounce_terms(**XSN31, slot_bytes=128 * MIB, band_credit=True,
                            n_cross_lanes=5)
        assert t.n_diag_lanes_priced == 0
        assert t.total_bytes == pytest.approx(2.625 * GIB, rel=1e-6)

    def test_all_cross_naive_ceiling_at_slot_64(self):
        """The ticket's own first question: slot=64 MiB, depth=2, upper
        bound."""
        t = xb.bounce_terms(**XSN31, slot_bytes=64 * MIB, band_credit=True,
                            n_cross_lanes=5)
        assert t.total_bytes == pytest.approx(1.3125 * GIB, rel=1e-6)

    def test_diagonal_lanes_never_move_off_the_whole_tag_floor(self):
        """The bound this rig's topology actually imposes: at most
        `xr.N_CARDS` of the 5 lanes can ever be diagonal, so at least
        `5 - N_CARDS` must be cross -- and whichever ARE diagonal are priced
        (and, in `run_bounce_leg`, allocated) at the UNCHANGED whole-tag
        floor regardless of `band_credit`."""
        assert xr.N_CARDS == 3
        off = xb.bounce_terms(**XSN31, slot_bytes=128 * MIB)
        worst = xb.bounce_terms(**XSN31, slot_bytes=128 * MIB,
                                band_credit=True,
                                n_cross_lanes=5 - xr.N_CARDS)
        assert worst.n_diag_lanes_priced == xr.N_CARDS
        # Even the WORST realistic split (every diagonal lane this topology
        # can have, all 5 active) still saves something -- it is bounded,
        # never erased, by the diagonal share alone.
        assert worst.total_bytes < off.total_bytes


# ===========================================================================
# (2) THE MECHANISM -- CrossSlotRendezvous, real semaphores + real record.
# ===========================================================================


@pytest.fixture()
def rendezvous_pair(tmp_path):
    """Two `CrossSlotRendezvous` (deposit/collect) over ONE real semaphore
    set and ONE real slot record -- `test_two_sides_on_one_pair_drain_each_
    other_1368`'s own harness, reused rather than re-invented."""
    nonce = f"{NONCE}-rv"
    xr.create_semaphores(nonce)
    sems = tp.SemSet(nonce)
    slots = wb.BounceSlots(nonce, shm_root=str(tmp_path), create=True,
                           rows_per_pair=4)
    deposit = wb.CrossSlotRendezvous(sems, slots, pair=PAIR, budget_s=0.2)
    collect = wb.CrossSlotRendezvous(sems, slots, pair=PAIR, budget_s=0.2)
    try:
        yield deposit, collect
    finally:
        slots.close()
        xr.unlink_semaphores(nonce)


class TestBandDrainCredit:
    def test_a_wait_before_any_post_is_a_real_timeout(self, rendezvous_pair):
        deposit, _collect = rendezvous_pair
        assert deposit.wait_band_drained() is False

    def test_post_then_wait_round_trips(self, rendezvous_pair):
        deposit, collect = rendezvous_pair
        collect.post_band_drained()
        assert deposit.wait_band_drained() is True
        # Consumed: a second wait with nothing further posted times out.
        assert deposit.wait_band_drained() is False

    def test_prime_drains_a_prior_tags_uncollected_tail_non_blocking(
            self, rendezvous_pair):
        """Two bands' worth of tag t's tail were never reused within tag t
        (`min(bands, slots)` residue, the module's own doctrine) -- `prime_
        band_drain` must clear them WITHOUT blocking, before tag t+1's own
        first wrap."""
        deposit, collect = rendezvous_pair
        collect.post_band_drained()
        collect.post_band_drained()
        deposit.prime_band_drain()               # must not block
        # The residue is gone: a fresh wait sees nothing until posted again.
        assert deposit.wait_band_drained() is False
        collect.post_band_drained()
        assert deposit.wait_band_drained() is True

    def test_it_never_touches_the_per_tag_drain_row(self, rendezvous_pair):
        """Row 0 of `empty` (the PER-TAG drain, #1374) and row 1 of `full`
        (THIS credit, #1397) must never interact -- pinned the same way
        `test_deposit_then_collect_moves_the_bytes...` pins the per-tag
        drain's own absence of a per-band post."""
        deposit, collect = rendezvous_pair
        collect.post_band_drained()
        # `empty` row 0 (the per-tag drain) starts ARMED at 1
        # (`create_semaphores`) -- #1374's own `prime_drain` exists
        # precisely to consume that inherited credit before it is
        # mistaken for a real one. Doing so here, and ONLY here, must not
        # touch row 1 (THIS credit) at all: it stays posted for the
        # `wait_band_drained` below.
        assert deposit.wait_drained(tag=TAG) is True   # the inherited 1
        assert deposit.wait_drained(tag=TAG) is False  # genuinely nothing now
        assert deposit.wait_band_drained() is True     # row 1, untouched

    def test_the_old_per_band_claim_still_cannot_return(self):
        """#1374's own ratchet, re-asserted: #1397 adds NEW methods, it does
        not resurrect `wait_empty`."""
        rv = wb.CrossSlotRendezvous(None, None, pair=PAIR)
        assert not hasattr(rv, "wait_empty")
        assert hasattr(rv, "wait_band_drained")
        assert hasattr(rv, "post_band_drained")


# ===========================================================================
# (3) `run_bounce_leg` INTEGRATION -- real bytes, cross phase split.
# ===========================================================================


def _band_credit_terms(*, n_bands: int, slot_bytes: int = 4096) -> xb.BounceTerms:
    """A tiny term whose `cross_lane_slots` is much smaller than `n_bands`,
    so every leg below is forced through at least one real wrap."""
    return xb.bounce_terms(
        bytes_per_direction=slot_bytes * n_bands, n_layers=1,
        widest_layer_bytes=slot_bytes * n_bands, pairs=1, depth=1,
        slot_bytes=slot_bytes, n_lanes=1,
        max_tag_bytes=slot_bytes * n_bands,   # Option 1 precondition
        band_credit=True, n_cross_lanes=1,
    )


def _descs(n_bands: int, nbytes: int, src_ops, dst_ops, *, seed_base=0):
    out = []
    for i in range(n_bands):
        src_ptr = src_ops.raw_malloc(0, nbytes)
        dst_ptr = dst_ops.raw_malloc(0, nbytes)
        ctypes.memmove(src_ops.real(src_ptr),
                       bytes([(seed_base + i) & 0xFF] * nbytes), nbytes)
        ctypes.memset(dst_ops.real(dst_ptr), 0xEE, nbytes)  # poison
        out.append((src_ptr, dst_ptr))
    return out


def test_bands_land_in_order_and_a_slot_is_reused_only_after_its_credit(
        tmp_path):
    """RED WITHOUT #1397: before this change no `band_credit` field exists
    and `run_bounce_leg` refuses (W68) the moment `bands >= slots` on a
    PHASE_DEPOSIT leg -- this is the GREEN shape the fix must reach: 6 bands
    through a 2-slot cross buffer, three full wraps, every band's OWN bytes
    collected.
    """
    n_bands, nbytes = 6, 4096
    terms = _band_credit_terms(n_bands=n_bands, slot_bytes=nbytes)
    assert terms.cross_lane_slots < n_bands, "the test must force a wrap"

    nonce = f"{NONCE}-order"
    xr.create_semaphores(nonce)
    sems = tp.SemSet(nonce)
    slots = wb.BounceSlots(nonce, shm_root=str(tmp_path), create=True,
                           rows_per_pair=max(terms.cross_lane_slots, 1))
    src_ops = FakeDeviceOps(str(tmp_path), rank=0)
    dst_ops = FakeDeviceOps(str(tmp_path), rank=1)
    try:
        pairs = _descs(n_bands, nbytes, src_ops, dst_ops)
        deposit_descs = [wx.XchgDesc(
            tag=TAG, src_rank=0, dst_rank=1, param_name=f"p{i}",
            src_ptr=sp, dst_ptr=None, kind=wx.FLAT, nbytes=nbytes, rows=1,
            run_bytes=nbytes, spitch=0, dpitch=0, src_off=0, dst_off=0)
            for i, (sp, dp) in enumerate(pairs)]
        collect_descs = [wx.XchgDesc(
            tag=TAG, src_rank=0, dst_rank=1, param_name=f"p{i}",
            src_ptr=None, dst_ptr=dp, kind=wx.FLAT, nbytes=nbytes, rows=1,
            run_bytes=nbytes, spitch=0, dpitch=0, src_off=0, dst_off=0)
            for i, (sp, dp) in enumerate(pairs)]

        deposit_rv = wb.CrossSlotRendezvous(sems, slots, pair=PAIR,
                                            budget_s=2.0)
        collect_rv = wb.CrossSlotRendezvous(sems, slots, pair=PAIR,
                                            budget_s=2.0)

        # Interleave by hand, one band at a time, so the collector really
        # drains band k before the depositor's band k+slots wraps onto it --
        # exactly the ordering `run_bounce_leg` enforces internally when both
        # halves run in the SAME process's two calls back to back would NOT
        # prove (it would just happen to work because nothing is concurrent).
        # Driving them through separate `run_bounce_leg` calls per band
        # exercises the SAME wrap-and-wait code path a real two-process leg
        # takes, one band's `slot_bytes` window at a time.
        for i in range(n_bands):
            wb.run_bounce_leg([deposit_descs[i]], src_ops, nonce,
                              slot_bytes=nbytes, terms=terms,
                              mode=wx.INJECT_AUTHORITATIVE,
                              phase=wb.PHASE_DEPOSIT, rendezvous=deposit_rv,
                              shm_root=str(tmp_path), lane="cr0")
            wb.run_bounce_leg([collect_descs[i]], dst_ops, nonce,
                              slot_bytes=nbytes, terms=terms,
                              mode=wx.INJECT_AUTHORITATIVE,
                              phase=wb.PHASE_COLLECT, rendezvous=collect_rv,
                              shm_root=str(tmp_path), lane="cr0")

        for i, (_sp, dp) in enumerate(pairs):
            got = ctypes.string_at(dst_ops.real(dp), nbytes)
            assert got == bytes([i & 0xFF] * nbytes), (
                f"band {i} arrived corrupted -- a slot was reused before "
                f"its own drain credit")
    finally:
        src_ops.close()
        dst_ops.close()
        slots.close()
        xr.unlink_semaphores(nonce)


def test_moved_bytes_are_identical_whole_tag_vs_band_credit(tmp_path):
    """(4) THE BUFFER SIZE CHANGES ON PURPOSE; THE BYTES DO NOT. Same
    descriptors, once through today's whole-tag sizing (`band_credit=False`)
    and once through Option 3 (`band_credit=True`), single-process `both`
    form so the comparison isolates the BYTES from the phase-split
    machinery already proven above."""
    n_bands, nbytes = 4, 256
    src_ops = FakeDeviceOps(str(tmp_path), rank=0)
    dst_ops_old = FakeDeviceOps(str(tmp_path), rank=1)
    dst_ops_new = FakeDeviceOps(str(tmp_path), rank=2)
    try:
        srcs = []
        for i in range(n_bands):
            sp = src_ops.raw_malloc(0, nbytes)
            ctypes.memmove(src_ops.real(sp), bytes([(i * 37 + 5) & 0xFF]
                                                    * nbytes), nbytes)
            srcs.append(sp)
        dst_old = [dst_ops_old.raw_malloc(0, nbytes) for _ in range(n_bands)]
        dst_new = [dst_ops_new.raw_malloc(0, nbytes) for _ in range(n_bands)]

        def _both(dst_ops, dst_ptrs, *, band_credit):
            terms = xb.bounce_terms(
                bytes_per_direction=nbytes * n_bands, n_layers=1,
                widest_layer_bytes=nbytes * n_bands, pairs=0, depth=1,
                slot_bytes=nbytes, n_lanes=1,
                max_tag_bytes=nbytes * n_bands,
                band_credit=band_credit,
                n_cross_lanes=(1 if band_credit else 0))
            descs = [wx.XchgDesc(
                tag=TAG, src_rank=0, dst_rank=1, param_name=f"p{i}",
                src_ptr=srcs[i], dst_ptr=dst_ptrs[i], kind=wx.FLAT,
                nbytes=nbytes, rows=1, run_bytes=nbytes, spitch=0, dpitch=0,
                src_off=0, dst_off=0) for i in range(n_bands)]
            # `both` never consults `rendezvous.pair`, so it never takes the
            # band-credit branch either way -- what is compared here is that
            # `terms.band_credit` alone changes NOTHING about the bytes a
            # single-process leg moves, which is the ANY-CALLER half of
            # "byte-identical bytes moved" (the cross-process half is
            # `test_bands_land_in_order...` above).
            wb.run_bounce_leg(descs, src_ops, f"{NONCE}-bi-{band_credit}",
                              slot_bytes=nbytes, depth=1, terms=terms,
                              mode=wx.INJECT_AUTHORITATIVE,
                              phase=wb.PHASE_BOTH, shm_root=str(tmp_path))

        _both(dst_ops_old, dst_old, band_credit=False)
        _both(dst_ops_new, dst_new, band_credit=True)

        for i in range(n_bands):
            old_bytes = ctypes.string_at(dst_ops_old.real(dst_old[i]), nbytes)
            new_bytes = ctypes.string_at(dst_ops_new.real(dst_new[i]), nbytes)
            assert old_bytes == new_bytes
            assert old_bytes == bytes([(i * 37 + 5) & 0xFF] * nbytes)
    finally:
        src_ops.close()
        dst_ops_old.close()
        dst_ops_new.close()


# ===========================================================================
# (M1) MUTANT -- the credit check bypassed ("Quittung entfernt").
# ===========================================================================


class _AlwaysDrainedRendezvous:
    """M1's double: a `wait_band_drained` that ALWAYS says yes, modelling
    "the check was removed" without editing the product. Everything else
    delegates to a real `CrossSlotRendezvous` so `post_full`/`wait_full`
    (band identity) still work exactly as they must."""

    def __init__(self, real: "wb.CrossSlotRendezvous"):
        self._real = real
        self.pair = real.pair
        self.budget_s = real.budget_s

    def __getattr__(self, name):
        return getattr(self._real, name)

    def wait_band_drained(self) -> bool:
        return True                       # THE HAZARD: never actually waits

    def post_band_drained(self) -> None:
        self._real.post_band_drained()

    def prime_band_drain(self) -> None:
        self._real.prime_band_drain()


def test_M1_bypassing_the_credit_check_collects_a_stale_band(tmp_path,
                                                              monkeypatch):
    """MUTANT M1, RED: with the credit check patched to always say "drained"
    (modelling its removal), a 1-slot cross buffer lets band 1's deposit
    overwrite band 0's bytes BEFORE band 0 was ever collected -- band 0's
    later collect then reads band 1's content under band 0's name. Silent:
    no exception, no counter, a green-looking leg with wrong bytes.
    """
    # AUTHORITATIVE, not this rig's ambient shadow default: pins
    # `cross_lane_slots` to exactly `assemble_slots(1, comparing=False) ==
    # 1` so the test forces a wrap at the smallest possible band count
    # (band 1) rather than depending on which mode happens to be ambient.
    monkeypatch.setattr(wx, "inject_mode", lambda *a, **k: wx.INJECT_AUTHORITATIVE)
    nbytes = 64
    terms = xb.bounce_terms(
        bytes_per_direction=nbytes * 2, n_layers=1,
        widest_layer_bytes=nbytes * 2, pairs=1, depth=1, slot_bytes=nbytes,
        n_lanes=1, max_tag_bytes=nbytes,      # 1 slot: band 1 MUST wrap
        band_credit=True, n_cross_lanes=1)
    assert terms.cross_lane_slots == 1

    nonce = f"{NONCE}-m1"
    xr.create_semaphores(nonce)
    sems = tp.SemSet(nonce)
    slots = wb.BounceSlots(nonce, shm_root=str(tmp_path), create=True,
                           rows_per_pair=1)
    src_ops = FakeDeviceOps(str(tmp_path), rank=0)
    dst_ops = FakeDeviceOps(str(tmp_path), rank=1)
    try:
        s0, s1 = src_ops.raw_malloc(0, nbytes), src_ops.raw_malloc(0, nbytes)
        d0, d1 = dst_ops.raw_malloc(0, nbytes), dst_ops.raw_malloc(0, nbytes)
        ctypes.memmove(src_ops.real(s0), bytes([0xAA] * nbytes), nbytes)
        ctypes.memmove(src_ops.real(s1), bytes([0xBB] * nbytes), nbytes)

        real_dep = wb.CrossSlotRendezvous(sems, slots, pair=PAIR,
                                          budget_s=0.2)
        broken_dep = _AlwaysDrainedRendezvous(real_dep)

        def _dep(desc):
            wb.run_bounce_leg([desc], src_ops, nonce, slot_bytes=nbytes,
                              terms=terms, mode=wx.INJECT_AUTHORITATIVE,
                              phase=wb.PHASE_DEPOSIT, rendezvous=broken_dep,
                              shm_root=str(tmp_path), lane="m1")

        desc0 = wx.XchgDesc(tag=TAG, src_rank=0, dst_rank=1, param_name="p0",
                            src_ptr=s0, dst_ptr=None, kind=wx.FLAT,
                            nbytes=nbytes, rows=1, run_bytes=nbytes,
                            spitch=0, dpitch=0, src_off=0, dst_off=0)
        desc1 = wx.XchgDesc(tag=TAG, src_rank=0, dst_rank=1, param_name="p1",
                            src_ptr=s1, dst_ptr=None, kind=wx.FLAT,
                            nbytes=nbytes, rows=1, run_bytes=nbytes,
                            spitch=0, dpitch=0, src_off=0, dst_off=0)

        # Band 0 deposits (no wrap yet: bands(0) < slots(1)).
        _dep(desc0)
        # NOBODY COLLECTED BAND 0 YET. Band 1's deposit wraps onto the same
        # slot and, with the broken double, is let through anyway.
        _dep(desc1)

        # Band 0's collect, arriving late, reads whatever is in the slot NOW.
        collect_rv = wb.CrossSlotRendezvous(sems, slots, pair=PAIR,
                                            budget_s=0.2)
        dst_for_band0 = wx.XchgDesc(
            tag=TAG, src_rank=0, dst_rank=1, param_name="p0", src_ptr=None,
            dst_ptr=d0, kind=wx.FLAT, nbytes=nbytes, rows=1,
            run_bytes=nbytes, spitch=0, dpitch=0, src_off=0, dst_off=0)
        wb.run_bounce_leg([dst_for_band0], dst_ops, nonce, slot_bytes=nbytes,
                          terms=terms, mode=wx.INJECT_AUTHORITATIVE,
                          phase=wb.PHASE_COLLECT, rendezvous=collect_rv,
                          shm_root=str(tmp_path), lane="m1")

        got = ctypes.string_at(dst_ops.real(d0), nbytes)
        assert got != bytes([0xAA] * nbytes), (
            "the mutant did not reproduce: band 0 was NOT overwritten, so "
            "this test is not exercising the hazard it claims to")
        assert got == bytes([0xBB] * nbytes), (
            "band 0's collector silently received band 1's bytes under "
            "band 0's name -- this IS the corruption `wait_band_drained` "
            "exists to prevent")
    finally:
        src_ops.close()
        dst_ops.close()
        slots.close()
        xr.unlink_semaphores(nonce)


# ===========================================================================
# (M2) MUTANT -- the credit posted BEFORE the D2H/H2D copy landed.
# ===========================================================================


def test_M2_posting_the_credit_before_the_copy_lands_corrupts_silently(
        tmp_path):
    """MUTANT M2, RED -- THE WORSE DIRECTION (named in the ticket): a credit
    posted BEFORE `ops.synchronize(c_stream)` actually runs the collector's
    queued device copy. `FakeDeviceOps` defers every `memcpy_async` to its
    matching `synchronize` (see that class's own docstring), so "posted too
    early" is reproduced EXACTLY and deterministically by calling `post_
    band_drained` before `synchronize` instead of after -- the one ordering
    `run_bounce_leg` itself always keeps, and the one this test intentionally
    violates to prove why.

    Sequence: collect queues band 0's device copy (not yet executed) ->
    credit posted TOO EARLY -> depositor, seeing the credit, immediately
    overwrites the SAME host slot with band 1's bytes -> collector's
    deferred `synchronize` FINALLY runs and copies whatever is in the slot
    NOW (band 1's bytes) into band 0's destination. No exception anywhere;
    the collector's own counters all read exactly as they would on the
    correct path.
    """
    nbytes = 64
    nonce = f"{NONCE}-m2"
    xr.create_semaphores(nonce)
    sems = tp.SemSet(nonce)
    slots = wb.BounceSlots(nonce, shm_root=str(tmp_path), create=True,
                           rows_per_pair=1)
    src_ops = FakeDeviceOps(str(tmp_path), rank=0)
    dst_ops = FakeDeviceOps(str(tmp_path), rank=1)
    try:
        d0 = dst_ops.raw_malloc(0, nbytes)
        ctypes.memset(dst_ops.real(d0), 0xEE, nbytes)   # poison

        dep_rv = wb.CrossSlotRendezvous(sems, slots, pair=PAIR, budget_s=1.0)
        col_rv = wb.CrossSlotRendezvous(sems, slots, pair=PAIR, budget_s=1.0)

        # ---- band 0's collect, HAND-DRIVEN with the two lines SWAPPED ----
        # This mirrors run_bounce_leg's own collect half
        # (`_collect_band` -> `ops.synchronize(c_stream)` -> `post_band_
        # drained`) with the last two lines reversed -- the exact mutant.
        c_stream = dst_ops.create_stream(0)
        bounce = wb.LayerBounce(dst_ops, nonce, slot_bytes=nbytes, depth=1,
                                create=True, shm_root=str(tmp_path),
                                lane="m2")
        try:
            # A REAL band 0 must already be sitting in the slot for the
            # collector to read (deposited by the source in the ordinary,
            # correct way -- the mutant is ONLY in the collect ordering).
            src_bounce = wb.LayerBounce(src_ops, nonce, slot_bytes=nbytes,
                                        depth=1, create=False,
                                        shm_root=str(tmp_path), lane="m2")
            try:
                ctypes.memmove(ctypes.c_void_p(src_bounce.slot_address(0)),
                               bytes([0xAA] * nbytes), nbytes)
            finally:
                src_bounce.close()
            dep_rv.post_full(slot=0, seq=0, nbytes=nbytes)

            class _Batch:
                seq = 0
                pieces = ()

            got_bytes = col_rv.wait_full(slot=0, seq=0)
            assert got_bytes == nbytes
            desc0 = wx.XchgDesc(
                tag=TAG, src_rank=0, dst_rank=1, param_name="p0",
                src_ptr=None, dst_ptr=d0, kind=wx.FLAT, nbytes=nbytes,
                rows=1, run_bytes=nbytes, spitch=0, dpitch=0, src_off=0,
                dst_off=0)
            from sglang.srt.weg2.weight_exchange_transport import batch_descs

            batch = next(iter(batch_descs([desc0], nbytes)))
            wb._collect_band(dst_ops, c_stream, [desc0], batch,
                             bounce.slot_address(0))
            # THE MUTATION: post BEFORE synchronize, not after.
            col_rv.post_band_drained()

            # ---- the depositor, seeing the (too-early) credit, reuses ----
            desc1 = wx.XchgDesc(
                tag=TAG, src_rank=0, dst_rank=1, param_name="p1",
                src_ptr=None, dst_ptr=None, kind=wx.FLAT, nbytes=nbytes,
                rows=1, run_bytes=nbytes, spitch=0, dpitch=0, src_off=0,
                dst_off=0)
            assert dep_rv.wait_band_drained() is True, (
                "the credit must be visible immediately -- that is the bug")
            src_bounce = wb.LayerBounce(src_ops, nonce, slot_bytes=nbytes,
                                        depth=1, create=False,
                                        shm_root=str(tmp_path), lane="m2")
            try:
                # The depositor's own overwrite: real product code always
                # syncs its OWN write immediately (`ops.synchronize(d_stream)`
                # right after `_deposit_band`, untouched by #1397), so this
                # lands on the host page instantly -- exactly what makes the
                # race deterministic instead of merely possible.
                ctypes.memmove(ctypes.c_void_p(src_bounce.slot_address(0)),
                               bytes([0xBB] * nbytes), nbytes)
            finally:
                src_bounce.close()

            # NOW the collector's deferred copy finally executes.
            dst_ops.synchronize(c_stream)
        finally:
            bounce.close()
            dst_ops.destroy_stream(c_stream)

        got = ctypes.string_at(dst_ops.real(d0), nbytes)
        assert got != bytes([0xAA] * nbytes), (
            "the mutant did not reproduce: band 0 arrived correctly, so "
            "this test is not exercising the hazard it claims to")
        assert got == bytes([0xBB] * nbytes), (
            "band 0's destination silently received band 1's bytes because "
            "the credit was posted before the collect's own device copy "
            "had landed -- this is the SILENT, WORSE mutant the ticket "
            "names: no exception, no counter, a green-looking leg")
    finally:
        src_ops.close()
        dst_ops.close()
        slots.close()
        xr.unlink_semaphores(nonce)
