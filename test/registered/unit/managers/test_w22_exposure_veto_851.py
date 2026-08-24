"""#851: the W22 specimen, as falsifiers that survive any fix shape.

W22 (2026-08-24, pin e5a37866d7, boot_w22_0824_0656.log) put serving in a state
where NO tp_to_pp flip could ever fund: 33 cutovers by 07:04:35 and not one
after, stand-down 7 of 8, while the box answered 123/123 requests at HTTP 200.
It could serve; it could not change layout.

THE ROOT, and it is NOT where my own first W22 write-up put it. I reported "the
floor compares a high-water ID against a backed-row count, so compute the floor
in COUNT units". That remedy is already built, already reverted, and guarded:
``KvBackingRelief._floor_rows`` carries "#770/#812: DETECT AND NAME AN
UNDER-BACKED RANK. DO NOT 'FIX' IT HERE. A FIRST VERSION OF THIS CLAMPED THE
FLOOR DOWN TO THE CAP, and that was WRONG in the dangerous direction ... it
authorises a cap below rows that are still in use", with
``test_residency_cap_flip_levelling_792`` red on exactly that clamp. The ID is
deliberate: ids above the high-water mark are the only ones guaranteed free.

The law actually broken is #822's EXPOSURE, quoted from the W22 log::

    [exposure] 343779 rows: the allocator could hand out 470755 rows while
    committed backing of 126976 (highest 470755); these are live

    exposed id space   470755
    committed backing  126976
    gap                343779   == the POOL CENSUS `withheld` term, exactly

The floor then fails DOWNSTREAM and correctly: ``max_live`` 126976 plus a 4096
admission reserve gives floor 131073, which cannot be backed because every id
above 126976 is exposed-but-unbacked. The rung asks for 4096 allocatable ids
and there are no pages behind any of them.

WHAT THESE TESTS PIN, deliberately chosen to be independent of whether the fix
lands on the allocator (enforce EXPOSURE), on the reduction (drop a
self-declared under-backed rank from the group MAX), or on both:

  1. The specimen IS an exposure violation, asserted by Law IDENTITY, never by
     message text (the 822 suite's own rule).
  2. floor > cap on the specimen numbers -- the rank's own self-report.
  3. THE ONE THAT MATTERS: a rank that has already declared itself under-backed
     must not veto a shrink its peers can fund. That is the standing
     kein-bindender-rang law, and ``floor_exceeds_local_cap``'s own docstring
     states it: "a floor above a rank's own cap is a DEFECT REPORT about that
     rank's backing, never a capacity verdict for its peers."

(3) is marked ``expectedFailure``: it is RED today, by design. When the
consolidated #851 fix lands it turns into an UNEXPECTED SUCCESS, which is loud
-- so nobody can claim the family is fixed while the veto still forms, and
nobody has to keep a red suite to remember. Both directions are pinned: the
legal-state twin of every red assertion must stay green.

Hermetic: pure functions, no scheduler, no pool, no CUDA.
"""

import unittest

from sglang.srt.managers.kv_backing_relief import (
    _SHRINK_SCALE,
    _floor_ppm,
    _shrink_ppm,
    collective_kv_shrink_ppm,
    collective_kv_target,
    floor_exceeds_local_cap,
)
from sglang.srt.mem_cache.kv_row_ownership import Law, RowSpace, RowOwnershipAuthority

# ---------------------------------------------------------------------------
# The W22 specimen, verbatim. Every number is quoted from
# /spinning/evidence-665-f1/boot_w22_0824_0656.log.
# ---------------------------------------------------------------------------

#: "the allocator could hand out 470755 rows"
W22_EXPOSED = 470755
#: "while committed backing of 126976"
W22_COMMITTED = 126976
#: "[exposure] 343779 rows" -- and the POOL CENSUS `withheld` term, identically.
W22_GAP = 343779
#: "current=126976 rows, floor=131073, slack=0"
W22_FLOOR = 131073
#: "FLOOR UNREACHABLE: it exceeds the current cap by 4097 rows"
W22_SHORT = 4097
#: "the deepest proportion any rank asked for is 72.2% of its own cap"
W22_PEER_ASK_PPM = 722_000


class TestTheSpecimenIsAnExposureViolation(unittest.TestCase):
    """(1) By Law identity. A reworded log line must not turn this green."""

    def test_the_gap_is_exposure_over_backing(self):
        self.assertEqual(W22_EXPOSED - W22_COMMITTED, W22_GAP)

    def test_the_authority_names_EXPOSURE_on_the_specimen(self):
        auth = RowOwnershipAuthority(
            RowSpace(exposed=W22_EXPOSED, committed=W22_COMMITTED)
        )
        laws = {v.law for v in auth.audit()}
        self.assertIn(Law.EXPOSURE, laws)

    def test_a_backed_pool_is_silent(self):
        """CAN-FAIL TWIN. Back the id space and the violation must vanish."""
        auth = RowOwnershipAuthority(
            RowSpace(exposed=W22_COMMITTED, committed=W22_COMMITTED)
        )
        laws = {v.law for v in auth.audit()}
        self.assertNotIn(Law.EXPOSURE, laws)


class TestTheRankSelfReportsUnderBacking(unittest.TestCase):
    """(2) The rank already knows. Nothing here is inferred."""

    def test_floor_exceeds_its_own_cap_by_the_logged_amount(self):
        self.assertTrue(floor_exceeds_local_cap(W22_FLOOR, W22_COMMITTED))
        self.assertEqual(W22_FLOOR - W22_COMMITTED, W22_SHORT)

    def test_a_rank_at_its_floor_is_not_a_defect(self):
        """CAN-FAIL TWIN. floor == cap is a full pool, not an under-backed one.

        Collapsing these two is what froze the group before #770/#812.
        """
        self.assertFalse(floor_exceeds_local_cap(W22_COMMITTED, W22_COMMITTED))


def _proposal(desire_rows: int, floor_rows: int, current_rows: int):
    """One rank's four-field proposal, built with the SHIPPED converters.

    ``propose()`` returns ``(_shrink_ppm(desire, current),
    -_floor_ppm(floor, current), current, -current)``. Fields 0 and 1 are
    PROPORTIONS of the rank's OWN cap (#796: a row id is meaningless to a peer
    on an uneven fleet), not row counts.

    Built by calling the real converters rather than by restating the formula:
    a first version of this file passed raw ROW COUNTS in fields 0 and 1, and
    every assertion below was then measuring a reduction that never happens.
    A falsifier standing on a hand-copied model of the thing it tests is worth
    nothing -- the same "check the indicator measures what it claims" rule that
    this whole family keeps paying for.
    """
    return [
        _shrink_ppm(desire_rows, current_rows),
        -_floor_ppm(floor_rows, current_rows),
        int(current_rows),
        -int(current_rows),
    ]


def _min_reduce(*proposals):
    """The element-wise MIN all-reduce the group actually performs."""
    return [min(col) for col in zip(*proposals)]


class TestTheGroupVerdict(unittest.TestCase):
    """(3) THE ONE THAT MATTERS. A defective rank must not bind its peers."""

    #: A pressed peer with a genuinely fundable plan: it can give up rows down
    #: to 72.2% of its own cap, which is what W22 logged every round.
    PEER_CAP = 212992
    PEER_DESIRE = int(212992 * 0.722)
    PEER_FLOOR = 115681  # comfortably under its own cap: this rank FITS

    def test_the_peer_alone_can_shrink(self):
        """CAN-FAIL TWIN, and the premise of the whole file.

        Without the under-backed rank in the reduction the group shrinks. If
        this ever goes red the specimen below proves nothing, because the
        group would have refused anyway.
        """
        reduced = _min_reduce(
            _proposal(self.PEER_DESIRE, self.PEER_FLOOR, self.PEER_CAP)
        )
        target = collective_kv_target(reduced, current_rows=self.PEER_CAP)
        self.assertIsNotNone(target)
        self.assertLess(target, self.PEER_CAP)

    def test_the_under_backed_rank_currently_vetoes(self):
        """CHARACTERISATION of today's behaviour -- passes NOW, on purpose.

        This is the defect, asserted as fact so the next reader does not have
        to re-derive it from a boot log. It is the twin of the expectedFailure
        below: exactly one of the two must flip when the fix lands.
        """
        reduced = _min_reduce(
            _proposal(self.PEER_DESIRE, self.PEER_FLOOR, self.PEER_CAP),
            _proposal(W22_COMMITTED, W22_FLOOR, W22_COMMITTED),
        )
        # The under-backed rank's floor is 131073/126976 = 103.2% of its own
        # cap. `_floor_ppm` CLAMPS that to 100% -- the log says so in as many
        # words ("102.9% clamps to 100%") -- and 100% is the neutral element of
        # the MIN, i.e. "no change". So the group's MAX floor becomes "nobody
        # shrinks", contributed by the one rank that could not shrink anyway.
        max_floor_ppm = -reduced[1]
        self.assertEqual(max_floor_ppm, _SHRINK_SCALE)
        target = collective_kv_target(reduced, current_rows=self.PEER_CAP)
        # The group's target is dragged up to the defective rank's floor, so
        # the peer's fundable plan never runs.
        self.assertTrue(target is None or target >= self.PEER_CAP)

    def test_the_reduction_MUST_veto_rather_than_cap_below_a_live_set(self):
        """THE FORBIDDEN-REMEDY GUARD. Green as-is, permanently.

        THIS TEST USED TO ASSERT THE OPPOSITE, and was wrong. It was written as
        `test_a_self_declared_under_backed_rank_MUST_NOT_veto`, xfail, as the
        acceptance for F1+F2 -- on the reasoning that a rank which has already
        declared itself under-backed is filing a defect report, not casting a
        capacity verdict for its peers. That reasoning is sound about the
        BACKING and wrong about the REDUCTION.

        At THIS layer the veto is CORRECT. The reduction is handed a floor and
        a cap as numbers; the only way to stop the veto here is to drop the
        rank's floor from the group MAX, and that rank still applies the
        resulting proportion to its own cap -- landing below its own live set,
        which is `cudaErrorIllegalAddress` and kills every rank rather than
        raising (#796). So the original assertion demanded a defect, and could
        never flip without one.

        What F1+F2 actually deliver is REACHABILITY -- floor > cap stops being
        PERMANENT because the pool can now grow to its lawful floor. That is
        pinned in `test_lawful_reservation_851::TestTheFloorIsREACHABLE`, at
        the layer that can deliver it. The two tests cover the two halves.

        THE RULE THIS COST, worth stating once: an acceptance test asserts a
        property THE FIX LAYER CAN DELIVER. A test that injects state into a
        deeper layer tests THAT layer's contract, not the fix.

        The law, from ``floor_exceeds_local_cap``'s own docstring: "a floor
        above a rank's own cap is a DEFECT REPORT about that rank's backing,
        never a capacity verdict for its peers." The rank in this reduction has
        ALREADY declared itself under-backed -- `floor_exceeds_local_cap` is
        True for it -- so its floor is a bug report, not a constraint, and the
        peer's fundable shrink must still run.

        FIX-SHAPE-INDEPENDENT. It goes green if the allocator stops exposing
        470755 ids against 126976 rows of backing (the rank's floor then fits
        its cap and there is no veto), and it goes green if the reduction
        learns to drop a self-declared defective rank. It does NOT prescribe
        which, and it must NOT be satisfied by clamping the floor down to the
        cap -- that is the #770/#812 corpse, and
        test_residency_cap_flip_levelling_792 stays red on it.
        """
        reduced = _min_reduce(
            _proposal(self.PEER_DESIRE, self.PEER_FLOOR, self.PEER_CAP),
            _proposal(W22_COMMITTED, W22_FLOOR, W22_COMMITTED),
        )
        target = collective_kv_target(reduced, current_rows=self.PEER_CAP)
        # The veto IS the correct outcome here: no shrink clears every rank's
        # live set, so the group declines rather than authorising a cap below
        # one. Asserted positively so a future "optimisation" that drops the
        # defective rank's floor from the MAX fails loudly.
        self.assertTrue(
            target is None or target >= self.PEER_CAP,
            "the reduction capped below a rank's live set -- cudaErrorIllegalAddress",
        )

    def test_two_healthy_ranks_still_respect_the_highest_real_floor(self):
        """CAN-FAIL TWIN for the fix. The MAX-floor rule is CORRECT when the
        floor is real: a target below a peer's live set is
        cudaErrorIllegalAddress, which kills every rank. Whatever #851 does, it
        must not turn this green-into-permissive.
        """
        reduced = _min_reduce(
            _proposal(100_000, 150_000, 212_992),
            _proposal(120_000, 180_000, 212_992),
        )
        ppm = collective_kv_shrink_ppm(reduced)
        target = collective_kv_target(reduced, current_rows=212_992)
        if ppm is not None:
            self.assertGreaterEqual(
                target, 180_000, "the target fell below a peer's REAL floor"
            )


if __name__ == "__main__":
    unittest.main()
