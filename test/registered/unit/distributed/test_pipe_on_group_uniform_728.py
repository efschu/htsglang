"""#728: one rank's failed JIT build must not give the group two layouts.

THE BUG (proved in #621, docs/FINDING_621_max_bytes_not_reconciled.md).

``pipe_on`` could be turned off on ONE RANK ALONE by a ``try/except`` around a
JIT build, with no collective. It feeds ``max_payload()``, which produces:

* ``max_bytes`` -- the ceiling ``handles()`` compares against
  (``if largest_round > self.max_bytes: return False``), so two ranks answer
  differently for the same payload and one enters a collective the other does
  not: a HANG;
* ``geometry()`` -- which fixes the slot OFFSETS.

WHY THE FIX RECONCILES THE INPUT AND NOT THE OUTPUT. Reconciling ``max_bytes``
after the fact -- e.g. a group minimum in the existing window ``all_gather`` --
would leave ``_geo`` already built from the local value. The function refuses
exactly that by name a few lines later:

    "No silent shrinking of the payload: the slot offsets are fixed in both
     kernels, and a rank with a different layout would write to the wrong
     place."

So ``pipe_on`` is AND-reduced across the group immediately after its build,
before anything is derived from it. Every downstream quantity is then uniform
by construction, and the window check keeps working untouched.

FAILURE DIRECTION, measured rather than assumed -- and it is NOT what the
obvious reading suggests. AND-reducing means one rank's failed build disables
the pipelined PATH for the whole session. The payload ceiling does not fall
with it: pipe-on costs slots, so turning it off RAISES max_bytes here from
7,880,704 to 14,909,440 bytes. What the group loses is the pipelined
optimisation, not capacity. Either way the trade is right -- losing a fast path
is slow, a per-rank layout hangs -- but the cost must be named accurately.
"""

import inspect
import unittest

from sglang.srt.distributed.device_communicators import barlink_bar1
from sglang.srt.distributed.device_communicators.barlink_bar1 import (
    max_payload,
    pipe_on_group_verdict,
)

WORLD = 4
REGION = 64 * 1024 * 1024
RING = 4
PIPE_RANGE = 2


def _max_bytes_for(pipe_on: bool) -> int:
    """What a rank computes at barlink_bar1.py:2311 given its own pipe_on."""
    return max_payload(WORLD, REGION, True, pipe_on, RING, PIPE_RANGE)


class TestTwoRanksAgreeAfterAFailedBuild(unittest.TestCase):
    """The harness: rank 0 built the extension, rank 1 did not."""

    GATHERED = [True, False]

    def test_UNRECONCILED_the_two_ranks_disagree(self):
        """The bug, stated as arithmetic. This is what the fix must remove."""
        a, b = _max_bytes_for(True), _max_bytes_for(False)
        self.assertNotEqual(a, b, "no divergence to fix -- the premise moved")
        straddling = (min(a, b) + max(a, b)) // 2
        self.assertNotEqual(
            straddling <= a,
            straddling <= b,
            "there must be a payload the two ranks answer differently on",
        )

    def test_RECONCILED_both_ranks_compute_the_SAME_ceiling(self):
        """RED-FIRST against the pre-#728 tree, where nothing reconciled
        pipe_on and each rank kept its own answer."""
        verdict = pipe_on_group_verdict(self.GATHERED)
        rank0 = _max_bytes_for(verdict)
        rank1 = _max_bytes_for(verdict)
        self.assertEqual(rank0, rank1)

    def test_RECONCILED_no_payload_splits_the_verdict(self):
        """The property that actually matters, sampled at the boundary.

        Sampled, not swept: the two ceilings are megabytes apart, so iterating
        the span would run for millions of steps -- the first cut of this test
        did exactly that and hung. The interesting sizes are the two ceilings
        and their neighbours, plus one size strictly between them, which is the
        only region where an unreconciled pair can disagree at all.
        """
        lo, hi = sorted((_max_bytes_for(True), _max_bytes_for(False)))
        ceiling = _max_bytes_for(pipe_on_group_verdict(self.GATHERED))
        # The rank that could not BUILD decides, which is the pipe-OFF value.
        # Note that is the LARGER ceiling, not the smaller: pipe-on costs
        # slots. "Weakest rank decides" is about capability, not about size.
        self.assertEqual(ceiling, _max_bytes_for(False))
        self.assertEqual(ceiling, hi, "pipe-off is the roomier layout")

        straddling = (lo + hi) // 2
        for nbytes in (lo - 1, lo, lo + 1, straddling, hi - 1, hi, hi + 1):
            with self.subTest(nbytes=nbytes):
                # Both ranks now derive from the SAME reconciled pipe_on, so
                # both compare against the same ceiling.
                rank0 = nbytes <= _max_bytes_for(pipe_on_group_verdict(self.GATHERED))
                rank1 = nbytes <= _max_bytes_for(pipe_on_group_verdict(self.GATHERED))
                self.assertEqual(rank0, rank1)

        # And the same size WOULD have split them before the fix -- otherwise
        # the agreement above is agreement about nothing.
        self.assertNotEqual(
            straddling <= _max_bytes_for(True),
            straddling <= _max_bytes_for(False),
            "the sampled straddling size must be one the unreconciled ranks "
            "actually disagreed on",
        )

    def test_the_group_takes_the_WEAKEST_rank(self):
        self.assertFalse(pipe_on_group_verdict([True, False]))
        self.assertFalse(pipe_on_group_verdict([False, True, True, True]))
        self.assertTrue(pipe_on_group_verdict([True, True]))

    def test_a_single_failure_anywhere_disables_the_group(self):
        for i in range(WORLD):
            with self.subTest(failing_rank=i):
                gathered = [True] * WORLD
                gathered[i] = False
                self.assertFalse(pipe_on_group_verdict(gathered))

    def test_an_empty_or_unreported_exchange_answers_FALSE(self):
        """'Nobody reported that they can' is not 'everybody can'. Defaulting
        the other way would enable the path on a failed exchange."""
        self.assertFalse(pipe_on_group_verdict([]))
        self.assertFalse(pipe_on_group_verdict([None, None]))
        self.assertTrue(pipe_on_group_verdict([True, None]))

    def test_the_reduction_is_AND_not_majority(self):
        self.assertFalse(pipe_on_group_verdict([True, True, True, False]))


class TestTheReconciliationHappensBEFOREAnythingDerivesFromIt(unittest.TestCase):
    """Source pins on the ORDER, which is the whole correctness argument.

    Behavioural pins cannot see the ordering: the gather needs a live process
    group. What can be pinned is that the reconciliation is present and sits
    ahead of both derivations -- and that is exactly the property a later
    refactor would break.
    """

    def _src(self):
        return inspect.getsource(barlink_bar1.BarlinkBar1Transport)

    def test_the_group_verdict_is_applied_at_all(self):
        self.assertIn("pipe_on_group_verdict(", self._src())

    def test_the_verdict_is_APPLIED_not_merely_COMPUTED(self):
        """Caught by mutation: removing the assignment left every other pin
        green, because they only proved the verdict was calculated.

        A group answer that nothing writes back is the orphaned-stop defect in
        another costume -- the reconciliation would run, log, and change
        nothing.
        """
        src = self._src()
        self.assertIn("self.pipe_on = group_pipe_on", src)
        self.assertLess(
            src.index("self.pipe_on = group_pipe_on"),
            src.index("max_bytes = max_payload("),
            "the verdict must be written back BEFORE anything derives from it",
        )

    def test_it_runs_BEFORE_max_payload(self):
        src = self._src()
        self.assertLess(
            src.index("pipe_on_group_verdict("),
            src.index("max_bytes = max_payload("),
            "max_bytes would be derived from an unreconciled pipe_on",
        )

    def test_it_runs_BEFORE_geometry(self):
        src = self._src()
        self.assertLess(
            src.index("pipe_on_group_verdict("),
            src.index("self._geo = geometry("),
            "the slot offsets would be derived from an unreconciled pipe_on, "
            "which is layout divergence, not just a hang",
        )

    def test_the_window_check_still_refuses_rather_than_shrinking(self):
        """The neighbouring rule this fix deliberately did not disturb."""
        self.assertIn("No silent shrinking of the payload", self._src())

    def test_the_docstring_no_longer_claims_max_bytes_is_all_gathered(self):
        """The claim that was false is gone, and the true one is stated."""
        doc = barlink_bar1.BarlinkBar1Transport.handles.__doc__
        self.assertNotIn("``_window_minimum`` and ``max_bytes`` from an", doc)
        self.assertIn("by construction", doc.lower())


if __name__ == "__main__":
    unittest.main()
