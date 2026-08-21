"""#785: is the context-local shard plan honoured by the code that SHAPES layers?

WHY THIS TEST EXISTS BEFORE THE FEATURE DOES. Deriving the flip's arena tail on
a first boot needs a TP-shaped model built at KV-sizing time, when no TP stack
exists yet. Building it means installing the flip vector for the duration of the
build and handing the process back unchanged -- which is what
``scoped_tp_partition_ratios`` is for. But every production caller that shapes a
real stack uses the process-global ``set_tp_partition_ratios`` instead
(``phase_flip_boot.py:655``), so NOTHING in the tree demonstrates that a model
constructed under the scope is sharded to the scope's vector rather than to the
process's installed plan.

That gap is not academic. The docstring of the scope names the exact failure it
guards: the only discriminator that decides whether a plan applies is
``len(ratios) == tp_size``, so a build under a mismatched vector does not raise
-- it falls back to the EVEN split and loads the wrong units, silently. A meta
probe that inherited the PP plan instead of the flip vector would therefore
return a plausible, wrong arena total, and the pool sized from it would be wrong
in exactly the way the two-boot cliff already is.

WHAT IS PROVEN HERE, and what is not. The chain that shapes a layer is
``layer construction -> tp_partition_size(...) -> get_tp_partition_ratios(family)
-> overlay``. This suite pins the whole of that chain below the layer, on the
three functions layer construction actually calls. It does NOT construct a real
model -- that needs a checkpoint and a device, and belongs in the integration
test that checks the derived tail against the measured {0: 82, 1: 0, 2: 2215}.
If this suite is green, an inherited-plan bug in the meta probe cannot be in the
overlay; if it is red, the fallback is the process-global setter wrapped in
try/finally, named as such rather than papered over.
"""

import threading
import unittest

from sglang.srt.distributed import utils as u

# A three-rank flip vector against a two-rank installed plan. The lengths differ
# on purpose: length is the only thing that decides whether a plan applies, so a
# leaked plan shows up as an even split rather than as an error.
FLIP_VEC = [32, 16, 16]
OTHER_VEC = [1, 1]
TOTAL, TP = 64, 3


class _CleanPlan(unittest.TestCase):
    def setUp(self):
        self._saved = (u._TP_PARTITION_RATIOS, u._TP_PARTITION_FAMILIES)
        u.set_tp_partition_ratios(None)

    def tearDown(self):
        u._TP_PARTITION_RATIOS, u._TP_PARTITION_FAMILIES = self._saved


class TheOverlayShapesTheSplit(_CleanPlan):
    def test_the_scope_is_read_by_the_sizes_helper(self):
        with u.scoped_tp_partition_ratios(FLIP_VEC):
            self.assertEqual(u.tp_partition_sizes(TOTAL, TP), [32, 16, 16])

    def test_the_scope_is_read_by_the_per_rank_helper(self):
        with u.scoped_tp_partition_ratios(FLIP_VEC):
            got = [u.tp_partition_size(TOTAL, TP, r) for r in range(TP)]
        self.assertEqual(got, [32, 16, 16])

    def test_the_scope_is_read_by_the_activity_predicate(self):
        self.assertFalse(u.tp_plan_active(TP))
        with u.scoped_tp_partition_ratios(FLIP_VEC):
            self.assertTrue(u.tp_plan_active(TP))

    def test_the_scope_beats_an_installed_process_plan(self):
        # The probe runs inside a booted process that already installed its own
        # plan. The scope must win for the duration of the build.
        u.set_tp_partition_ratios([16, 32, 16])
        with u.scoped_tp_partition_ratios(FLIP_VEC):
            self.assertEqual(u.tp_partition_sizes(TOTAL, TP), [32, 16, 16])

    def test_family_vectors_ride_along(self):
        with u.scoped_tp_partition_ratios(FLIP_VEC, {"mlp": [40, 12, 12]}):
            self.assertEqual(
                u.tp_partition_sizes(TOTAL, TP, family="mlp"), [40, 12, 12]
            )
            self.assertEqual(u.tp_partition_sizes(TOTAL, TP), [32, 16, 16])


class WithoutTheScopeTheBuildIsSilentlyEven(_CleanPlan):
    """CAN-FAIL PROOF for the failure the scope's own docstring names."""

    # A total the even split can actually divide, so the fallback is reached
    # instead of the divisibility assertion. That is the dangerous shape: an
    # even split is a plausible answer, and nothing distinguishes it from the
    # intended one.
    DIVISIBLE = 48

    def test_a_mismatched_installed_plan_does_not_raise_it_splits_evenly(self):
        u.set_tp_partition_ratios(OTHER_VEC)  # two entries, three ranks
        sizes = u.tp_partition_sizes(self.DIVISIBLE, TP)
        self.assertEqual(sizes, [16, 16, 16])
        self.assertNotEqual(sizes, [24, 12, 12])

    def test_and_the_scope_is_what_repairs_it(self):
        # Same 32:16:16 proportion as the flip vector, scaled to divide 48.
        u.set_tp_partition_ratios(OTHER_VEC)
        with u.scoped_tp_partition_ratios([2, 1, 1]):
            self.assertEqual(u.tp_partition_sizes(self.DIVISIBLE, TP), [24, 12, 12])


class TheProcessIsHandedBackUnchanged(_CleanPlan):
    def test_the_installed_plan_survives_the_scope(self):
        u.set_tp_partition_ratios([16, 32, 16], {"mlp": [8, 8, 48]})
        with u.scoped_tp_partition_ratios(FLIP_VEC):
            pass
        self.assertEqual(u.tp_partition_sizes(TOTAL, TP), [16, 32, 16])
        self.assertEqual(u.tp_partition_sizes(TOTAL, TP, family="mlp"), [8, 8, 48])

    def test_it_survives_an_exception_inside_the_scope(self):
        # A meta probe that raises must not leave the boot shaped by the flip
        # vector -- that would mis-shard everything built afterwards.
        u.set_tp_partition_ratios([16, 32, 16])
        with self.assertRaises(RuntimeError):
            with u.scoped_tp_partition_ratios(FLIP_VEC):
                raise RuntimeError("probe failed")
        self.assertEqual(u.tp_partition_sizes(TOTAL, TP), [16, 32, 16])

    def test_no_overlay_means_no_plan_not_an_even_plan(self):
        # The sentinel matters: None is a meaningful plan value ("even split"),
        # so "no overlay" needs to be distinguishable from "overlay of None".
        self.assertIsNone(u.get_tp_partition_ratios())
        with u.scoped_tp_partition_ratios(None):
            self.assertIsNone(u.get_tp_partition_ratios())
        self.assertIsNone(u.get_tp_partition_ratios())

    def test_scopes_nest(self):
        with u.scoped_tp_partition_ratios(FLIP_VEC):
            with u.scoped_tp_partition_ratios([8, 8, 48]):
                self.assertEqual(u.tp_partition_sizes(TOTAL, TP), [8, 8, 48])
            self.assertEqual(u.tp_partition_sizes(TOTAL, TP), [32, 16, 16])


class TheScopeDoesNotLeakAcrossThreads(_CleanPlan):
    """The property that makes a probe safe to run inside a live boot.

    A context variable is per-thread by construction. The probe installs the
    flip vector while the scheduler's own thread keeps reading the installed
    plan; without this, a concurrent forward would read the probe's vector.
    """

    def test_another_thread_keeps_the_installed_plan(self):
        u.set_tp_partition_ratios([16, 32, 16])
        seen = {}

        def other():
            seen["sizes"] = u.tp_partition_sizes(TOTAL, TP)

        with u.scoped_tp_partition_ratios(FLIP_VEC):
            self.assertEqual(u.tp_partition_sizes(TOTAL, TP), [32, 16, 16])
            t = threading.Thread(target=other)
            t.start()
            t.join()

        self.assertEqual(seen["sizes"], [16, 32, 16])


if __name__ == "__main__":
    unittest.main()
