"""#783: persistence at the cutover must not hand over rows the request gets back.

WHAT THIS WOULD HAVE PREVENTED. W37-H arm B, 2026-08-25 15:37:07, all three
ranks, 33 s after the server became ready and before any load attached:

    ValueError: pool memory leak detected!
      [full] total=468981, available=108565, evictable=1, protected=0,
             session_held=0, uncached=0, withheld=360437

    108565 + 1 + 360437 = 469003  against  total 468981  ->  22 slots claimed twice

The change under test was one line: the #856 cutover retraction persisting its
residents with `release_kv_cache(..., is_insert=True)` instead of False. The
reasoning was that retraction routes through `cache_finished_req`, the FINISHED
path, so ownership transfers cleanly.

THE REASONING WAS WRONG AND THE SEAM SAYS SO TWO STATEMENTS LATER.
`readmit_seam_residents` puts every retracted request straight back on the
queue -- the seam's own "# W31: RE-ADMIT THEM. THIS IS THE HALF #856 NEVER
SHIPPED". The request RESUMES and goes on using, and later freeing, the very
rows the tree just took. `cache_finished_req` is the finish CODE PATH; it does
not make the request finished.

So the rule this file pins is not about branches, it is about populations:

    A participant whose population RETURNS after the cutover may be persisted
    only by COPY. Persisting it by OWNERSHIP TRANSFER is structurally a double
    claim, regardless of which code path performs the transfer.

WHY A LEDGER MODEL AND NOT A MOCK OF THE REAL POOL. The defect is arithmetic:
one row counted in two places. A model that tracks WHO CLAIMS EACH ROW
reproduces it exactly, and the arithmetic is then checked by the PRODUCTION
function -- `SchedulerInvariantChecker._check_pool_invariant`, the same static method
that raised on the rig -- rather than by a second copy of the equation. If
that equation ever changes, this test follows it instead of drifting.

The counts are scaled down (a 40-row pool, 22 shared rows) but the shape and
the resulting imbalance are the rig's: 22 rows claimed twice.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

from sglang.srt.managers.scheduler_components.invariant_checker import (
    SchedulerInvariantChecker,
)
from sglang.test.test_utils import CustomTestCase

#: The rig's shape, scaled. 22 is the exact number of doubly-claimed rows the
#: arm B crash reported, kept so the fixture is traceable to the specimen.
POOL_TOTAL = 40
RESIDENT_ROWS = 22


class SeamLedger:
    """Who claims each row across one cutover. Deliberately tiny.

    Three disjoint claimants, mirroring the production ledger's terms:
      free      -> `available`
      tree      -> `evictable`   (the radix cache owns it, may evict it)
      request   -> `withheld`    (a live request holds it out of circulation)

    A row in two sets at once is the defect; nothing in the model forbids it,
    which is the point -- the model must be ABLE to express the bug or the
    test cannot fail.
    """

    def __init__(self, total: int, resident: int) -> None:
        self.total = total
        self.rows = set(range(resident))
        self.free = set(range(resident, total))
        self.tree: set = set()
        self.request: set = set(self.rows)

    def retract(self, *, insert: bool) -> None:
        """The #856 cutover retraction. The request stops holding its rows."""
        self.request = set()
        if insert:
            # `release_kv_cache(is_insert=True)` -> the TREE takes the rows.
            self.tree |= self.rows
        else:
            # `is_insert=False` -> the rows go back to the allocator.
            self.free |= self.rows

    def readmit(self) -> None:
        """`readmit_seam_residents`: the population RETURNS and resumes.

        It resumes on its own rows when the tree took them (the prefix is
        "already there"), and re-allocates from the free list otherwise.
        """
        self.request |= self.rows
        self.free -= self.rows

    def check(self):
        """Run the PRODUCTION invariant over this ledger."""
        return SchedulerInvariantChecker._check_pool_invariant(
            "full",
            available=len(self.free),
            evictable=len(self.tree),
            protected=0,
            session_held=0,
            total=self.total,
            uncached=0,
            withheld=len(self.request),
        )


class TestOwnershipTransferAtASeamThatReadmits(CustomTestCase):
    def test_transfer_then_readmit_double_claims(self):
        """THE ARM B CRASH, at the desk, in milliseconds."""
        led = SeamLedger(POOL_TOTAL, RESIDENT_ROWS)
        led.retract(insert=True)
        led.readmit()
        leak, msg = led.check()
        self.assertTrue(leak, f"the double claim must be visible: {msg}")
        # The imbalance is exactly the shared population, as on the rig.
        accounted = len(led.free) + len(led.tree) + len(led.request)
        self.assertEqual(accounted - led.total, RESIDENT_ROWS)

    def test_copy_then_readmit_balances(self):
        """THE LAWFUL SHAPE: persist by COPY, rows stay where they are.

        The fence does exactly this (device->host staging) and transfers no row
        ownership, which is why it is the participant the design put before the
        seam.
        """
        led = SeamLedger(POOL_TOTAL, RESIDENT_ROWS)
        copied_to_host = set(led.rows)  # the copy lives OUTSIDE the pool ledger
        led.retract(insert=False)
        led.readmit()
        leak, msg = led.check()
        self.assertFalse(leak, f"a copy must not disturb the ledger: {msg}")
        self.assertEqual(len(copied_to_host), RESIDENT_ROWS)

    def test_no_readmission_makes_transfer_lawful_again(self):
        """THE DISCRIMINATOR. Ownership transfer is not wrong in itself -- it is
        wrong for a population that comes BACK. A genuinely finished request
        never calls `readmit`, and the same transfer then balances.

        This is what separates the rule from 'never insert', and it is why the
        predicate is keyed on the returning population and not on the branch."""
        led = SeamLedger(POOL_TOTAL, RESIDENT_ROWS)
        led.retract(insert=True)
        # no readmit(): the request is over, the tree owns the rows outright
        leak, msg = led.check()
        self.assertFalse(leak, f"a true finish must balance: {msg}")


class TestTheLedgerItselfCanFail(CustomTestCase):
    """Guard-the-guard: a model that cannot express the bug proves nothing."""

    def test_a_balanced_ledger_is_reported_balanced(self):
        led = SeamLedger(POOL_TOTAL, RESIDENT_ROWS)
        leak, msg = led.check()
        self.assertFalse(leak, msg)

    def test_the_production_equation_is_the_one_being_used(self):
        # If the production invariant ever stops counting `withheld`, this
        # fixture must notice rather than silently agree with it.
        leak, _ = SchedulerInvariantChecker._check_pool_invariant(
            "full",
            available=108565,
            evictable=1,
            protected=0,
            session_held=0,
            total=468981,
            uncached=0,
            withheld=360437,
        )
        self.assertTrue(leak, "the rig's own arm B numbers must read as a leak")


class TestBothHalvesTogetherAreOwnershipNeutral(CustomTestCase):
    """#783 GATE OVER THE COMBINATION, not over each half separately.

    Half 1 (copy at cutover retraction) and half 2 (restore at re-admission)
    are only ownership-neutral TOGETHER. Each half alone can look balanced
    while the pair leaks, so the ledger has to be able to kill the
    combination -- which is what killed arm B at the desk.

    The host copy lives OUTSIDE the pool ledger by construction: that is the
    whole difference from ownership transfer. It is modelled here as a value
    held aside, and the assertions check that it never appears as a pool claim.
    """

    def _copy_out(self, led):
        """Half 1: copy the state, then release the rows. Ownership untouched
        -- the rows go back to the allocator exactly as an ordinary retraction
        would leave them."""
        host_copy = {r: f"state@{r}" for r in led.rows}
        led.retract(insert=False)
        return host_copy

    def _restore_in(self, led, host_copy):
        """Half 2: the request is re-admitted, allocates rows again, and the
        copy is loaded into them. It allocates from the FREE list like any
        other admission -- it does not reclaim rows somebody else now owns."""
        led.readmit()
        return {r: host_copy[r] for r in led.rows}

    def test_copy_then_restore_balances_at_every_step(self):
        led = SeamLedger(POOL_TOTAL, RESIDENT_ROWS)
        leak, msg = led.check()
        self.assertFalse(leak, f"precondition: {msg}")

        host_copy = self._copy_out(led)
        leak, msg = led.check()
        self.assertFalse(leak, f"after copy+release: {msg}")

        restored = self._restore_in(led, host_copy)
        leak, msg = led.check()
        self.assertFalse(leak, f"after re-admission+restore: {msg}")

        self.assertEqual(len(restored), RESIDENT_ROWS)

    def test_the_host_copy_is_never_a_pool_claim(self):
        """The property that separates a copy from a handover. If the host
        copy were ever counted as a pool claimant, the pair would leak exactly
        as arm B did."""
        led = SeamLedger(POOL_TOTAL, RESIDENT_ROWS)
        host_copy = self._copy_out(led)
        self._restore_in(led, host_copy)
        accounted = len(led.free) + len(led.tree) + len(led.request)
        self.assertEqual(
            accounted,
            led.total,
            "the host copy must sit outside the pool ledger; counting it is "
            "the ownership transfer this design exists to avoid",
        )

    def test_the_gate_kills_a_restore_that_transfers_ownership(self):
        """CAN-FAIL OVER THE COMBINATION. If half 2 were implemented by giving
        the rows to the tree (the arm B shape) instead of loading into freshly
        allocated ones, the pair must redden -- even though half 1 alone was
        perfectly balanced."""
        led = SeamLedger(POOL_TOTAL, RESIDENT_ROWS)
        self._copy_out(led)
        leak, _ = led.check()
        self.assertFalse(leak, "half 1 alone is balanced -- that is the trap")

        led.tree |= led.rows  # a restore that hands the rows over
        led.readmit()
        leak, msg = led.check()
        self.assertTrue(
            leak,
            f"the combination must be detectable as a double claim: {msg}",
        )


if __name__ == "__main__":
    unittest.main()
