"""#853(ii) -- the F4 naming clause reaches EVERY corridor-guard emission.

THE W24 SPECIMEN, and the count is the point::

    "from [nothing]" payout-side     0    PASS
    "from [nothing]" gate-side     112    PARTIAL: 45 carry the F4 clause,
                                          67 UNCLAUSED

All 67 are standalone CORRIDOR-GUARD emissions -- 66 REFUSED and 1
CANNOT-FULLY-HOLD -- and the predecessor's suspicion that they lived on the
arm-refused path was CORRECTED by the window: the arm-refused and
FLIP-ABANDONED lines embed FUNDING-POSTS text, which already carries the
clause. The gap is on the guard's own log site.

THE TWO HOLES, both structural rather than accidental:

1. ``must_reclaim`` REFUSALS. The clause is appended under ``if not ok``, which
   runs BEFORE the ``must_reclaim`` branch flips ``ok`` to False -- and that
   branch then REBUILDS ``detail`` from scratch, so anything appended earlier
   would have been discarded anyway. A refusal born in that branch could never
   carry the clause by construction.
2. CANNOT-FULLY-HOLD. That line is emitted under ``ok and law_breached``. The
   ask succeeded, so ``if not ok`` never fired and the clause was never
   considered -- while the line is precisely a report about memory pressure,
   which is when a reader most needs to know a funder was in the room.

WHY IT MATTERS, in the words the corpus already used for the payout side: the
ladder's ``[nothing]`` is honest about the LADDER, but read as a sentence it
says "this rig had no memory to give", which sends the next reader to capacity
planning. The true statement is "this rig had a funder this gate may not
spend". One window was partly spent on that difference; W24 spent 67 more lines
on it.

APPENDED, NEVER SUBSTITUTED, unchanged from F4: the ladder's own provider list
stays exactly as it was, because it is the truthful record of what the gate
actually spent. And a clean, non-breaching success still says nothing -- an ask
that needed no money does not have to explain the money it did not need.

Hermetic: no CUDA, no NVML, no pool. CUDA_VISIBLE_DEVICES="".
"""

import unittest

from sglang.srt.managers import corridor_guard as cg

MIB = 1024 * 1024


def _guard(free_mib, *, floor_mib=1255, law_floor_mib=1024, witness=True):
    """A guard with an EMPTY ladder and a declared-but-unspendable funder --
    W22/W24's shape, and the only one in which `[nothing]` can be printed while
    a funder holds credit."""
    g = cg.CorridorGuard(
        0,
        floor_mib=floor_mib,
        probe=lambda: int(free_mib) * MIB,
        law_floor_mib=law_floor_mib,
    )
    if witness:
        g.declare_offledger_funder(lambda want: (("kv-slack", 2560 * MIB, ""),))
    return g


class TheClauseReachesTheMustReclaimRefusal(unittest.TestCase):
    def test_a_must_reclaim_refusal_names_the_funder(self):
        """HOLE 1. The ask FITS (so the ordinary verdict is ok), but nothing was
        reclaimed, and under must_reclaim that is a refusal. W24's 66."""
        res = _guard(4096).ensure_headroom(
            6 * MIB, reason="seam staging tp_to_pp", must_reclaim=True
        )
        self.assertFalse(res.ok)
        self.assertIn("must_reclaim", res.detail)
        self.assertIn("kv-slack", res.detail)
        self.assertIn("2560", res.detail)

    def test_the_must_reclaim_refusal_keeps_its_own_terms(self):
        """THE DANGER DIRECTION for hole 1. That branch deliberately states
        ONLY asked-vs-reclaimed, because quoting the free column next to a 6 MiB
        want was reported as reading like an inversion. The clause must be
        APPENDED to that message, not restore the terms it dropped."""
        res = _guard(4096).ensure_headroom(
            6 * MIB, reason="seam staging tp_to_pp", must_reclaim=True
        )
        self.assertIn("INCREMENTAL", res.detail)
        self.assertNotIn("arming floor", res.detail)
        self.assertNotIn("corridor law", res.detail)


class TheClauseReachesTheLawBreachReport(unittest.TestCase):
    def test_a_cannot_fully_hold_line_names_the_funder(self):
        """HOLE 2, W24's 1. The allocation fits, so the gate proceeds -- but it
        reports a predicted trough below the law, and a reader deciding what to
        do about that pressure needs to know a funder was declared."""
        res = _guard(2000, law_floor_mib=1024).ensure_headroom(
            1500 * MIB, reason="seam staging pp_to_tp"
        )
        self.assertTrue(res.ok)
        self.assertIn("kv-slack", res.detail)


class TheClauseStaysOffACleanSuccess(unittest.TestCase):
    """THE CAN-FAIL DIRECTION FOR THE WHOLE FILE. A fix that appends the clause
    unconditionally would pass every assertion above while burying every
    successful gate pass in funder prose. F4's rule is explicit: only failures
    carry the suffix."""

    def test_a_comfortable_success_says_nothing_about_funders(self):
        res = _guard(8192, law_floor_mib=1024).ensure_headroom(
            100 * MIB, reason="seam staging pp_to_tp"
        )
        self.assertTrue(res.ok)
        self.assertNotIn("kv-slack", res.detail)


class TheLadderRecordSurvivesEverywhere(unittest.TestCase):
    """F4's standing rule, re-pinned on the two newly-clauseed paths: the
    clause EXPLAINS the ladder's record, it never replaces it."""

    def test_the_must_reclaim_refusal_still_prints_the_empty_ladder(self):
        res = _guard(4096).ensure_headroom(6 * MIB, must_reclaim=True)
        self.assertIn("[nothing]", res.detail)
        self.assertLess(res.detail.index("[nothing]"), res.detail.index("kv-slack"))

    def test_an_undeclared_witness_leaves_every_line_exactly_as_it_was(self):
        """The backward pin. No funder declared -> no clause anywhere, on both
        of the paths this ticket touches."""
        refusal = _guard(4096, witness=False).ensure_headroom(
            6 * MIB, must_reclaim=True
        )
        breach = _guard(2000, law_floor_mib=1024, witness=False).ensure_headroom(
            1500 * MIB
        )
        self.assertNotIn("kv-slack", refusal.detail)
        self.assertNotIn("may not draw", refusal.detail)
        self.assertNotIn("kv-slack", breach.detail)
        self.assertNotIn("may not draw", breach.detail)

    def test_the_plain_refusal_path_is_unregressed(self):
        """F4's original acceptance, still green: the ordinary refusal (want
        exceeds free) keeps naming the funder."""
        res = _guard(2420).ensure_headroom(3248 * MIB, reason="seam staging")
        self.assertFalse(res.ok)
        self.assertIn("kv-slack", res.detail)
        self.assertIn("[nothing]", res.detail)


if __name__ == "__main__":
    unittest.main()
