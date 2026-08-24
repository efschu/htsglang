"""#848 -- the floor-need actuator asks the arena what it can hold, first.

THE SPECIMEN IS WINDOW 7 (/spinning/gpu-arb/WINDOW7-RESULT.md), and it is the
window where the v2 exit census WORKED and named the wrong authority::

    exit=GAP           floor=126976 gap=1 need=126977
    exit=COMMIT-RAISED floor=126976 target=126977
    [#839-METAL] GROUP FLOOR CANNOT FUND THE LIVE SET ... (ValueError(
        'final_num_tokens=126977 must satisfy page_size=1 <= final <= reserved=125052'))

The refusal fired, which is what v2 was built for. But it reported
COMMIT-RAISED -- the DRIVER'S complaint -- when the thing that actually binds is
the arena's IMMUTABLE VA RESERVATION of 125052. A reader following that exit goes
to the commit path; the defect is in the sizer.

WHY THE RESERVATION CAN BIND AT ALL. It is fixed at construction from the pool's
size AT THAT MOMENT (``reserved_num_tokens=self.size``, memory_pool.py:2690) and
never reassigned, while ``size`` stays mutable and the #330 dial writes it on
every step (``reserved_rows`` docstring, kv_vmm_backing.py:1408-1425). A rank
whose dial has moved past its own boot reservation can never grow again.

PRIOR ART, AND THIS IS THE PATTERN BEING FOLLOWED. #684 found exactly this shape
one call site over -- "Recovery aims above the reservation, so the backing cap
never lifts" (2acfd7c2bb), whose body states "The pool's reservation is never
consulted; ``reserved`` is not read against a pool anywhere in the module" -- and
fixed it by reading ``_reserved_rows()`` before proposing a target. It also
recorded the same failure on metal: ``recovery to 270646 rows failed: ... <=
reserved=190596``, 59 times on three ranks, 2026-08-16. This actuator was written
AFTER that fix and did not inherit the lesson.

WHERE THIS DELIBERATELY DIFFERS FROM ``recover()``. Recovery clamps to the
ceiling, because it wants as much backing as it can get and a clamped target
still does useful work. This actuator has ONE job -- raise the group floor past
the live set -- and a target clamped BELOW the need does not do it: it would
commit pages, report progress, and leave the flip exactly as unfittable. So the
honest answer here is a REFUSAL that names the binding authority, not a clamp.

WHAT THIS DOES NOT FIX. The reservation being too small in the first place is a
boot-time sizing defect and is a separate posting. This ticket makes the seam
report it accurately instead of blaming the commit.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_floor_need_exits_839_v2 import (  # noqa: E402
    RESERVATION,
    W6_BACKED,
    W6_FLOOR,
    W6_GAP,
    W6_LIVE_SPAN,
    _Rank,
    _ballot,
    _exits,
    _group,
)

#: Window 7, to the row: PP1's arena reservation, BELOW its own size.
W7_RESERVED = 125052
#: PP1's size on that boot -- 1924 rows above its own reservation ceiling.
W7_SIZE = 126976


class _CappedRank(_Rank):
    """A rank whose arena reservation sits BELOW its current size.

    That inversion is the whole specimen and it is not hypothetical: it is what
    window 7 measured on PP1 after 30 minutes of load.
    """

    def __init__(self, name, rows, reserved):
        super().__init__(name, rows)
        self.pool.reserved_backing_rows = int(reserved)


def _capped_group(reserved=W7_RESERVED):
    ranks = _group()
    ranks["PP1"] = _CappedRank("PP1", W6_BACKED["PP1"], reserved)
    return ranks


class TheReservationIsConsultedBeforeTheTarget(unittest.TestCase):
    """RED on integ/round7: the exit name does not exist there."""

    def test_the_exit_name_exists(self):
        from sglang.srt.managers import kv_backing_relief as K

        self.assertTrue(
            hasattr(K, "FLOOR_NEED_RESERVATION_CAPPED"),
            "the binding authority needs its own name in the census",
        )
        self.assertIn(K.FLOOR_NEED_RESERVATION_CAPPED, K.FLOOR_NEED_EXITS)

    def test_the_window_7_constellation_reports_RESERVATION_CAPPED(self):
        from sglang.srt.managers import kv_backing_relief as K

        ranks = _capped_group()
        _ballot(ranks, close=False)
        grown = int(ranks["PP1"].relief.close_floor_need_gap())
        self.assertEqual(grown, 0)
        self.assertIn(K.FLOOR_NEED_RESERVATION_CAPPED, _exits(ranks["PP1"]))

    def test_the_doomed_commit_is_NOT_attempted(self):
        """Window 7 called the setter and let the driver raise. Do not.

        Asking the arena costs one attribute read; the failed commit costs a
        driver round trip and produces an exit naming the wrong authority.
        """
        ranks = _capped_group()
        _ballot(ranks, close=False)
        ranks["PP1"].pool.attempts.clear()
        ranks["PP1"].relief.close_floor_need_gap()
        self.assertEqual(
            ranks["PP1"].pool.attempts,
            [],
            "no commit may be attempted once the reservation is known to refuse it",
        )

    def test_the_refusal_names_the_RESERVATION_not_the_driver(self):
        ranks = _capped_group()
        _ballot(ranks, close=False)
        ranks["PP1"].relief.close_floor_need_gap()
        said = ranks["PP1"].relief.floor_need_refusal()
        self.assertIsNotNone(said)
        self.assertEqual(int(said["binding_rows"]), W6_FLOOR)
        self.assertEqual(int(said["need"]), W6_LIVE_SPAN)
        self.assertEqual(int(said["short"]), W6_GAP)
        self.assertIn(str(W7_RESERVED), said["why"])
        self.assertIn("reservation", said["why"].lower())

    def test_it_is_DISTINGUISHABLE_from_the_commit_exits(self):
        """The point of the ticket: window 7 could not tell these apart."""
        from sglang.srt.managers import kv_backing_relief as K

        capped = _capped_group()
        _ballot(capped, close=False)
        capped["PP1"].relief.close_floor_need_gap()

        raiser = _group()
        raiser["PP1"] = _Rank("PP1", W6_BACKED["PP1"], raise_at=W6_FLOOR + 1024)
        _ballot(raiser, close=False)
        raiser["PP1"].relief.close_floor_need_gap()

        self.assertIn(K.FLOOR_NEED_RESERVATION_CAPPED, _exits(capped["PP1"]))
        self.assertNotIn(K.FLOOR_NEED_COMMIT_RAISED, _exits(capped["PP1"]))
        self.assertIn(K.FLOOR_NEED_COMMIT_RAISED, _exits(raiser["PP1"]))
        self.assertNotIn(K.FLOOR_NEED_RESERVATION_CAPPED, _exits(raiser["PP1"]))


class TheGuardDoesNotOverReach(unittest.TestCase):
    """Danger direction: refusing a grow that WOULD have worked is the bug here."""

    def test_a_healthy_reservation_still_grows(self):
        from sglang.srt.managers import kv_backing_relief as K

        ranks = _capped_group(reserved=RESERVATION)  # far above the target
        _ballot(ranks, close=False)
        grown = int(ranks["PP1"].relief.close_floor_need_gap())
        self.assertEqual(grown, W6_GAP, "a reservation above the target must not refuse")
        self.assertIn(K.FLOOR_NEED_GROWN, _exits(ranks["PP1"]))
        self.assertNotIn(K.FLOOR_NEED_RESERVATION_CAPPED, _exits(ranks["PP1"]))

    def test_a_reservation_exactly_at_the_target_is_ALLOWED(self):
        """`_check_final` accepts final == reserved, so this guard must too.

        An off-by-one here would refuse the last legal row and turn a working
        grow into a permanent named refusal -- worse than the defect.
        """
        from sglang.srt.managers import kv_backing_relief as K

        ranks = _capped_group(reserved=W6_LIVE_SPAN)  # target == reservation
        _ballot(ranks, close=False)
        grown = int(ranks["PP1"].relief.close_floor_need_gap())
        self.assertEqual(grown, W6_GAP)
        self.assertNotIn(K.FLOOR_NEED_RESERVATION_CAPPED, _exits(ranks["PP1"]))

    def test_an_unreadable_reservation_keeps_the_previous_behaviour(self):
        """`_reserved_rows()` returns None for a pool with no arena (#684).

        None must mean "no ceiling to check", never "ceiling of zero" -- a pool
        without an arena has to behave exactly as it did before this ticket.
        """
        from sglang.srt.managers import kv_backing_relief as K

        ranks = _group()
        del ranks["PP1"].pool.reserved_backing_rows
        _ballot(ranks, close=False)
        grown = int(ranks["PP1"].relief.close_floor_need_gap())
        self.assertEqual(grown, W6_GAP)
        self.assertNotIn(K.FLOOR_NEED_RESERVATION_CAPPED, _exits(ranks["PP1"]))

    def test_a_zero_reservation_is_treated_as_unreadable_not_as_a_ceiling(self):
        from sglang.srt.managers import kv_backing_relief as K

        ranks = _capped_group(reserved=0)
        _ballot(ranks, close=False)
        grown = int(ranks["PP1"].relief.close_floor_need_gap())
        self.assertEqual(grown, W6_GAP, "0 means 'no arena', not 'ceiling of zero'")
        self.assertNotIn(K.FLOOR_NEED_RESERVATION_CAPPED, _exits(ranks["PP1"]))

    def test_the_ranks_that_are_not_the_floor_are_untouched(self):
        from sglang.srt.managers import kv_backing_relief as K

        ranks = _capped_group()
        _ballot(ranks, close=False)
        for name in ("PP0", "PP2"):
            ranks[name].relief.close_floor_need_gap()
            self.assertNotIn(K.FLOOR_NEED_RESERVATION_CAPPED, _exits(ranks[name]))


class TheExposedIdSpaceIsStillNeverRaised(unittest.TestCase):
    """#839 A's rule is untouched by this ticket, and stays pinned."""

    def test_a_reservation_capped_rank_announces_nothing(self):
        ranks = _capped_group()
        _ballot(ranks, close=False)
        before = ranks["PP1"].exposed()
        ranks["PP1"].relief.close_floor_need_gap()
        self.assertEqual(ranks["PP1"].exposed(), before)

    def test_the_group_never_exposes_above_the_floor(self):
        ranks = _capped_group()
        for _ in range(4):
            floor = _ballot(ranks)
            for name, r in ranks.items():
                self.assertLessEqual(r.exposed(), floor, name)
                self.assertLessEqual(r.exposed(), r.backed(), name)


if __name__ == "__main__":
    unittest.main()
