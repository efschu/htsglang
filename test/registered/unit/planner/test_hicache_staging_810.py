"""#810: the staging-tier size derivation, so `--hicache-size` is not a hand-pin.

WHAT THESE TESTS PIN, beyond the arithmetic:

* that BOTH consumers are covered. Sizing a staging tier from the write side
  alone yields a tier that is correct and quietly serialises prefetch,
  because a read takes its landing slot from the same pool before the
  storage read is issued. A regression here is invisible in any capacity
  metric -- it shows up as latency.
* that an unsustainable ingest rate is ANSWERED, not sized. If the producer
  outruns the drain, the in-flight set grows without bound at any size, so
  emitting a number would be emitting a wrong one.
* that the derivation rounds UP. Rounding down emits a tier smaller than the
  derivation that justifies it.

    python -m pytest test/registered/unit/planner/test_hicache_staging_810.py -v
"""

import unittest

from sglang.srt.planner.hicache_staging import (
    BYTES_PER_GB,
    DEFAULT_BURST_MARGIN,
    MIN_STAGING_GB,
    describe_staging_size,
    read_landing_bytes,
    staging_size_gb,
    sustainable,
    write_staging_bytes,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

#: A drain rate in the shape this rig actually measures (~0.5 GB/s on the
#: ZFS-backed file tier). Used as a TEST INPUT only -- the module takes the
#: rate as a parameter precisely so no rig fit is baked into the planner.
DRAIN = 0.5 * BYTES_PER_GB


class TestWriteStaging(CustomTestCase):
    def test_bandwidth_delay_product(self):
        # 0.5 GB/s x 2 s x margin 1.0 = 1.0 GB in flight.
        self.assertAlmostEqual(write_staging_bytes(DRAIN, 2.0, 1.0), 1.0 * BYTES_PER_GB)

    def test_margin_multiplies(self):
        base = write_staging_bytes(DRAIN, 2.0, 1.0)
        self.assertAlmostEqual(write_staging_bytes(DRAIN, 2.0, 2.0), 2 * base)

    def test_zero_drain_is_refused_not_treated_as_small(self):
        # A zero drain does not mean "a tiny staging area"; it means the
        # residency is unbounded. Returning 0 here would emit the smallest
        # possible tier for the worst possible case.
        with self.assertRaises(ValueError) as cm:
            write_staging_bytes(0.0, 2.0)
        self.assertIn("unbounded", str(cm.exception))

    def test_zero_latency_is_refused(self):
        with self.assertRaises(ValueError):
            write_staging_bytes(DRAIN, 0.0)

    def test_margin_below_one_is_refused(self):
        # A margin under 1.0 sizes the tier BELOW its own steady-state
        # in-flight bytes -- guaranteed exhaustion, not a tuning choice.
        with self.assertRaises(ValueError) as cm:
            write_staging_bytes(DRAIN, 2.0, 0.5)
        self.assertIn("burst_margin", str(cm.exception))


class TestReadLanding(CustomTestCase):
    def test_scales_with_concurrency(self):
        self.assertEqual(read_landing_bytes(4, 1000), 4000.0)

    def test_no_prefetch_needs_no_landing_space(self):
        self.assertEqual(read_landing_bytes(0, 1000), 0.0)

    def test_negative_inputs_refused(self):
        with self.assertRaises(ValueError):
            read_landing_bytes(-1, 1000)


class TestSustainability(CustomTestCase):
    def test_drain_faster_than_ingest_is_sustainable(self):
        self.assertTrue(sustainable(1.0, 2.0))

    def test_equal_rates_are_sustainable(self):
        self.assertTrue(sustainable(2.0, 2.0))

    def test_producer_outrunning_the_drain_is_not(self):
        # The case no staging size fixes: this must be answerable so the
        # caller refuses instead of emitting a number that cannot hold.
        self.assertFalse(sustainable(3.0, 2.0))


class TestStagingSizeGb(CustomTestCase):
    def test_takes_the_larger_consumer_write_bound(self):
        # write: 0.5 GB/s x 4 s x 2.0 = 4 GB; read: 1 x 1 B ~ 0.
        self.assertEqual(staging_size_gb(DRAIN, 4.0, 1, 1), 4)

    def test_takes_the_larger_consumer_read_bound(self):
        # write: 0.5 x 1 x 2.0 = 1 GB; read: 16 pages x 0.5 GB = 8 GB.
        # THE DANGER DIRECTION: a derivation that ignored the read side would
        # return 1 here and silently serialise prefetch.
        self.assertEqual(
            staging_size_gb(DRAIN, 1.0, 16, int(0.5 * BYTES_PER_GB)),
            8,
        )

    def test_rounds_up_never_down(self):
        # write = 0.5 GB/s x 1 s x 2.0 = 1.0 GB exactly; nudge above it and
        # the emitted size must go to 2, not stay at 1.
        self.assertEqual(staging_size_gb(DRAIN * 1.01, 1.0, 0, 0), 2)

    def test_floor_is_the_flags_granularity(self):
        # A derivation far below 1 GB still has to emit a usable flag value.
        self.assertEqual(staging_size_gb(1.0, 1.0, 0, 0), MIN_STAGING_GB)

    def test_default_margin_is_applied_when_unspecified(self):
        explicit = staging_size_gb(DRAIN, 2.0, 0, 0, DEFAULT_BURST_MARGIN)
        implicit = staging_size_gb(DRAIN, 2.0, 0, 0)
        self.assertEqual(explicit, implicit)


class TestDescribeStagingSize(CustomTestCase):
    def test_names_the_binding_consumer_write(self):
        text = describe_staging_size(DRAIN, 4.0, 1, 1)
        self.assertIn("bound by write staging", text)
        self.assertIn("--hicache-size 4 GB", text)

    def test_names_the_binding_consumer_read(self):
        # The two have DIFFERENT remedies -- drain faster vs admit fewer
        # concurrent prefetches -- so the derivation has to say which one
        # bound it, or the reader tunes the wrong knob.
        text = describe_staging_size(DRAIN, 1.0, 16, int(0.5 * BYTES_PER_GB))
        self.assertIn("bound by read landing slots", text)

    def test_shows_the_factors_not_just_the_result(self):
        text = describe_staging_size(DRAIN, 2.0, 0, 0)
        self.assertIn("GB/s", text)
        self.assertIn("margin", text)


if __name__ == "__main__":
    unittest.main()


class TheDerivationTouchesNoBudgetRegistryTest(CustomTestCase):
    """#810 follow-up: deriving a size must have no allocation side effect.

    An earlier `fits_pinned_host_budget()` in this module called
    `check_and_register_pinned_post`, i.e. it registered a #729 post in the
    PLANNER process. That registry credits earlier posts back against live
    availability on the stated precondition that their bytes are already
    resident (`pinned_host_budget.py`: "the already-allocated posts must be
    credited back"). A planner allocates nothing, so such a post is precisely
    what that same comment calls "the real hazard" -- registered and never
    allocated, hence credited back without ever having been resident, charging
    the NEXT admission too little and waving through the over-commitment the
    registry exists to refuse.

    The function had no caller and no test anywhere in the tree, so nothing
    would have failed if it had grown one. This pins the property behaviourally
    instead of by absence: the whole derivation path runs and the registry is
    untouched. A re-added registering helper fails here.
    """

    def test_deriving_a_size_registers_nothing(self):
        from sglang.srt.mem_cache.pinned_host_budget import (
            clear_registered_posts,
            registered_posts,
        )

        clear_registered_posts()
        try:
            staging_size_gb(
                drain_bytes_per_s=0.5e9,
                drain_latency_s=0.2,
                max_concurrent_prefetch=64,
                page_bytes=1 << 20,
            )
            describe_staging_size(
                drain_bytes_per_s=0.5e9,
                drain_latency_s=0.2,
                max_concurrent_prefetch=64,
                page_bytes=1 << 20,
            )
            sustainable(0.4e9, 0.5e9)
            self.assertEqual(
                registered_posts(),
                (),
                "the planner's size derivation registered a pinned-host post: "
                "a post that never allocates is credited back as if it had, "
                "and the next admission is charged too little",
            )
        finally:
            clear_registered_posts()

    def test_the_module_exposes_no_budget_helper(self):
        """The removed name, pinned. A future reader reaching for it is sent to
        `ServerArgs._post_hicache_staging_host_ledger`, which prices the posts
        jointly and by the RANK PRODUCT rather than per rank."""
        import sglang.srt.planner.hicache_staging as mod

        self.assertFalse(
            hasattr(mod, "fits_pinned_host_budget"),
            "fits_pinned_host_budget is back; if a planner-side pre-check is "
            "wanted it must be PURE (joint_pinned_host_error, no registration) "
            "and rank-aware",
        )
