"""#848, the half that is NOT closed: the reservation law is not a fixed point.

WHAT IS CLOSED, so nobody re-derives it. Window 7 framed #848 as "PP1's backing
126976 exceeds its own reservation 125052". That framing is RETRACTED by its own
root commit (cd0d871824): ``uniform_backed_tokens`` is CHUNK-GRANULAR -- ``min
over buffers of (committed // row_bytes) * tokens_per_row`` -- so it legitimately
sits up to one commit chunk per buffer ABOVE the reservation's row count. The two
numbers were never one quantity. The real defect was the SIZER
(``reserved_num_tokens=self.size``), and #851 F2 (e62b1fae26) replaced it with
``lawful_reservation_rows(size, admission_reserve, 0)``. Metal, W24 onward:
``RESERVATION-CAPPED 0 PASS (F2 holds; #848 wall gone)``.

WHAT THIS FILE PINS. The shipped law reserves ``size + 1 + reserve``, which
dominates the floor ``_floor_rows`` can demand AT THE BOOT SIZE. It does not
dominate the floor at the DIAL'S OWN CEILING, and the dial's ceiling IS the
reservation (memory_pool clamps the unvalidated ``n == backed`` branch to
``reserved_backing_rows``, #848ii). So::

    R      = size + 1 + reserve                    the reservation
    floor(max_live = R) = R + 1 + reserve          what the rung may then demand
    gap    = 1 + reserve                           permanently above the ceiling

Measured by this file on the W22 specimen (size 126976, reserve 4096): R = 131073
and the floor at the dial ceiling is 135170 -- short by 4097, which is the same
``1 + reserve`` shape #812 measured as "4097 rows under the floor".

WHY PINNED AND NOT PATCHED. Closing it means one of two things, and neither is
this module's call:

  * lower the dial's usable ceiling to ``R - 1 - reserve``, which is a CAPACITY
    decision (it hands back exposed rows) and belongs to the planner; or
  * reserve VA against a growth bound rather than the boot size -- free in
    principle, since a VA reservation is address space and not committed pages,
    but no caller supplies such a bound today.

Unobserved on metal to date: the dial has never come near the ceiling, and this
exit has read 0 in every window since W24. This file exists so that if it ever
reads non-zero again, the reader is not sent back to the closed half.

THE SECOND ASSERTION IS THE STRUCTURAL GUARD. ``lawful_reservation_rows`` and
``_floor_rows`` are two spellings of one formula in one file, 1200 lines apart,
with nothing tying them together. If a term is ever added to the floor and not
mirrored in the reservation, the #848 wall returns silently. The lockstep test
below fails the moment they disagree.

Hermetic: pure arithmetic plus one method called on a stub. No pool, no driver,
no CUDA, no boot.
"""

import unittest

from sglang.srt.managers.kv_backing_relief import (
    KvBackingRelief,
    lawful_reservation_rows,
)

#: W22 / #848, verbatim.
W22_SIZE = 126976
W22_RESERVE = 4096


def _floor_rows(max_live, reserve, margin=0, page_size=1):
    """``KvBackingRelief._floor_rows`` on a stub -- the real method, no pool.

    Called unbound so the test tracks the shipped implementation rather than a
    copy of it. The method reads only ``_pool.page_size``, ``_margin_rows`` and
    ``_admission_reserve_rows``.
    """
    stub = object.__new__(KvBackingRelief)
    stub._pool = type("P", (), {"page_size": page_size})()
    stub._margin_rows = margin
    stub._admission_reserve_rows = reserve
    return int(KvBackingRelief._floor_rows(stub, max_live))


class TestTheClosedHalfStaysClosed(unittest.TestCase):
    """Guard the F2 fix, so this file cannot be read as reopening #848."""

    def test_the_reservation_dominates_the_floor_at_the_boot_size(self):
        R = lawful_reservation_rows(W22_SIZE, W22_RESERVE, 0)
        self.assertGreaterEqual(R, _floor_rows(W22_SIZE, W22_RESERVE))

    def test_the_pre_F2_sizer_does_not(self):
        """CAN-FAIL TWIN. Reverting to ``reserved_num_tokens=self.size``
        fails the assertion above, which is what makes it worth asserting."""
        self.assertLess(W22_SIZE, _floor_rows(W22_SIZE, W22_RESERVE))


class TestTheLawIsNotAFixedPoint(unittest.TestCase):
    """The residual, with its arithmetic, so the number is not re-derived."""

    def test_the_floor_at_the_dial_ceiling_is_above_the_ceiling(self):
        R = lawful_reservation_rows(W22_SIZE, W22_RESERVE, 0)
        self.assertEqual(R, 131073)
        at_ceiling = _floor_rows(R, W22_RESERVE)
        self.assertEqual(at_ceiling, 135170)
        self.assertGreater(at_ceiling, R)

    def test_the_gap_is_exactly_one_plus_the_reserve(self):
        """Not a coincidence of one specimen: the shape, across the range.

        If this ever stops holding, either the floor law or the reservation law
        moved and the residual has a different size than the prose says.
        """
        for size in (1024, W22_SIZE, 212992):
            for reserve in (0, 512, W22_RESERVE, 16384):
                R = lawful_reservation_rows(size, reserve, 0)
                self.assertEqual(
                    _floor_rows(R, reserve) - R,
                    1 + reserve,
                    f"size={size} reserve={reserve}",
                )

    def test_one_more_application_does_not_converge(self):
        """The point of the word FIXED POINT: iterating never closes the gap."""
        size = W22_SIZE
        for _ in range(4):
            R = lawful_reservation_rows(size, W22_RESERVE, 0)
            self.assertGreater(_floor_rows(R, W22_RESERVE), R)
            size = R


class TestTheTwoLawsStayInLockstep(unittest.TestCase):
    """THE STRUCTURAL GUARD. One formula, two spellings, nothing tying them.

    ``lawful_reservation_rows(size, reserve, margin)`` must equal
    ``_floor_rows(size)`` for the same terms. A term added to one and not the
    other rebuilds the #848 wall silently, which is exactly how it was built
    the first time.
    """

    def test_the_reservation_law_reproduces_the_floor_law(self):
        for size in (1, 1024, W22_SIZE, 212992):
            for reserve in (0, 512, W22_RESERVE):
                for margin in (0, 256):
                    self.assertEqual(
                        lawful_reservation_rows(size, reserve, margin),
                        _floor_rows(size, reserve, margin),
                        f"size={size} reserve={reserve} margin={margin}",
                    )

    def test_it_holds_under_page_rounding(self):
        """``_floor_rows`` rounds up to the page; the reservation must not sit
        BELOW that rounded value, or the wall returns on any paged pool."""
        for page in (1, 8, 64):
            for size in (1024, W22_SIZE):
                self.assertGreaterEqual(
                    _floor_rows(size, W22_RESERVE, 0, page_size=page)
                    - lawful_reservation_rows(size, W22_RESERVE, 0),
                    0,
                    f"page={page} size={size}",
                )

    def test_the_lockstep_test_can_fail(self):
        """CAN-FAIL PROOF for the guard above: a floor with one extra term is
        detected. Without this, a lockstep assertion that could never fail
        would read identically to one that holds."""
        drifted = _floor_rows(W22_SIZE, W22_RESERVE + 1)
        self.assertNotEqual(drifted, lawful_reservation_rows(W22_SIZE, W22_RESERVE, 0))


if __name__ == "__main__":
    unittest.main()
