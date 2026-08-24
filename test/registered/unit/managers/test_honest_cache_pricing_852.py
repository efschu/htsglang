"""#852 -- the allocator-cache post is priced at what ``empty_cache()`` CAN
return, not at ``reserved - allocated``.

THE SPECIMEN, ``boot_w24_0824_0852.log``, 09:00:33 PP0 (and 42 more of the
same shape -- 43 of the window's 45 binding refusals)::

    #770 FUNDING POSTS: want 3168 MiB, covered 2304 MiB, SHORT 864 MiB from
      [allocator-cache[local] 0 MiB (derated to zero: last draw delivered
      0/324514816 of its promise); ...], cause=phantom_capacity

Every pass of the W24 stuck phase re-priced the promise at
``memory_reserved() - memory_allocated()`` (~309-324 MiB), drew on it with
``empty_cache()``, measured 0 bytes delivered, derated the post to zero, and
then did the whole thing again 60 s later. #828's law 2 made the VERDICT
honest -- ``cause=phantom_capacity`` names the defect -- but the PROMISE
stayed dishonest, and each cycle paid a device sync for a draw that was
provably going to return nothing.

WHY THE DRAW RETURNS NOTHING. ``reserved - allocated`` counts every free
block, including blocks fragmented INSIDE segments that still carry live
allocations. ``empty_cache()`` releases only whole free segments. The torch
allocator publishes exactly the discriminating figure:
``inactive_split_bytes`` is the free bytes trapped inside in-use segments, so

    releasable = reserved - allocated - inactive_split

is what a draw can actually hand the driver. On W24's PP0 that figure was ~0
for 23 straight minutes while the raw difference read ~320 MiB.

THE FIX, in three connected pieces this file asserts one by one:

1. ``authority_from_seam_snapshot`` accepts the releasable measurement and
   prices the post with it; a cache that is raw-nonzero but releasable-zero
   is an HONEST ZERO with a named fragmentation reason, not a phantom
   promise that law 2 has to claw back after a wasted draw.
2. ``_funding_post_census`` wires the measurement through (a law connected
   to nothing is this corpus's signature failure mode).
3. ``_staging_affordable`` skips the ``empty_cache()`` draw when the
   measurement says 0 -- the derate loop becomes a single named refusal --
   and still draws (and still measures, for law 2) whenever the measurement
   says there is something to collect or abstains.

An unmeasurable backend (no CUDA context, cudaMallocAsync without the
counter) abstains with ``None`` and every verdict stays byte-identical to
#828's -- that is the backward pin, asserted here in both directions.

Hermetic: no CUDA, no NVML, no pool. CUDA_VISIBLE_DEVICES="".
"""

import unittest

from sglang.srt.managers.funding_authority import (
    CAUSE_FUNDED,
    CAUSE_PHANTOM,
    CAUSE_SCARCITY,
    authority_from_seam_snapshot,
)
from sglang.srt.managers.phase_flip_runtime import (
    PhaseFlipRuntime,
    releasable_cache_bytes_from_stats,
)

MIB = 1024 * 1024

# -- boot_w24_0824_0852.log, 09:00:33 PP0 ------------------------------------
W24_RAW_CACHE = 324514816  # the derate denominator the log printed, ~309 MiB
W24_WANT = 3168 * MIB  # the corridor gate's ask at the same refusal


class TheCacheIsPricedAtWhatEmptyCacheCanReturn(unittest.TestCase):
    def test_the_w24_specimen_prices_zero_and_names_fragmentation(self):
        """THE SPECIMEN. ~309 MiB raw, 0 releasable: an honest zero, named."""
        auth = authority_from_seam_snapshot(
            allocator_cache_bytes=W24_RAW_CACHE,
            allocator_cache_releasable_bytes=0,
            kv_slack_rows=0,
            row_bytes=16384,
        )
        v = auth.can_fund(W24_WANT)
        self.assertFalse(v.ok)
        drawn = {d.post: d.drawn_bytes for d in v.draws}
        self.assertEqual(drawn["allocator-cache"], 0)
        line = v.describe()
        self.assertIn("releasable", line)
        self.assertIn("309 MiB cached", line)
        # The honest zero must not masquerade as a clawed-back promise.
        self.assertNotIn("derated to zero", line)
        # Nothing phantom about a post that never promised: this is scarcity,
        # so the stand-down machinery sees the truth instead of a retry hint.
        self.assertEqual(v.cause, CAUSE_SCARCITY)

    def test_an_unfragmented_cache_keeps_its_full_promise(self):
        """THE DANGER DIRECTION. A fix that prices every cache at zero would
        abandon flips the cache can fund."""
        auth = authority_from_seam_snapshot(
            allocator_cache_bytes=W24_RAW_CACHE,
            allocator_cache_releasable_bytes=W24_RAW_CACHE,
        )
        v = auth.can_fund(300 * MIB)
        self.assertTrue(v.ok)
        self.assertEqual(v.cause, CAUSE_FUNDED)

    def test_a_partially_fragmented_cache_prices_the_releasable_part(self):
        auth = authority_from_seam_snapshot(
            allocator_cache_bytes=400 * MIB,
            allocator_cache_releasable_bytes=100 * MIB,
        )
        v = auth.can_fund(400 * MIB)
        drawn = {d.post: d.drawn_bytes for d in v.draws}
        self.assertEqual(drawn["allocator-cache"], 100 * MIB)

    def test_an_unmeasured_backend_keeps_the_828_pricing_exactly(self):
        """The backward pin. ``None`` (the default, and what every pre-#852
        caller passes) must reproduce #828's verdict byte for byte: promise
        at raw, law-2 derate on the measured delivery, phantom named."""
        auth = authority_from_seam_snapshot(
            allocator_cache_bytes=334 * MIB,
            allocator_cache_delivered_bytes=0,
        )
        v = auth.can_fund(1746 * MIB)
        self.assertFalse(v.ok)
        self.assertEqual(v.cause, CAUSE_PHANTOM)
        self.assertIn("derated to zero", v.describe())

    def test_delivery_still_derates_the_releasable_promise(self):
        """Law 2 stays the backstop UNDER the honest promise: a releasable
        estimate that still over-promises is corrected by the measured draw."""
        auth = authority_from_seam_snapshot(
            allocator_cache_bytes=400 * MIB,
            allocator_cache_releasable_bytes=200 * MIB,
            allocator_cache_delivered_bytes=50 * MIB,
        )
        v = auth.can_fund(400 * MIB)
        drawn = {d.post: d.drawn_bytes for d in v.draws}
        self.assertEqual(drawn["allocator-cache"], 50 * MIB)

    def test_a_releasable_reading_above_the_cache_never_inflates_the_post(self):
        """A measurement above ``reserved - allocated`` means the MEASUREMENT
        is wrong, not the post -- same clamp philosophy as law 2's ratio."""
        auth = authority_from_seam_snapshot(
            allocator_cache_bytes=100 * MIB,
            allocator_cache_releasable_bytes=900 * MIB,
        )
        v = auth.can_fund(500 * MIB)
        drawn = {d.post: d.drawn_bytes for d in v.draws}
        self.assertEqual(drawn["allocator-cache"], 100 * MIB)


class TheSeamCensusSpendsTheReleasableMeasurement(unittest.TestCase):
    """The wiring edge, same shape as #828's: drive the real method on a bare
    stub, because ``_funding_post_census`` swallows every exception and a cut
    wire is silent."""

    class _Runtime:
        _census_scheduler = None
        _rank = 0

    def _census(self, **attrs):
        stub = self._Runtime()
        for k, v in attrs.items():
            setattr(stub, k, v)
        return PhaseFlipRuntime._funding_post_census(stub, W24_WANT)

    def test_a_stored_zero_releasable_reaches_the_refusal_line(self):
        """THE W24 LOOP, closed. The stored figures of the specimen pass must
        produce the named fragmentation zero, not another phantom cycle."""
        line = self._census(
            _last_cache_promised_bytes=W24_RAW_CACHE,
            _last_cache_delivered_bytes=0,
            _last_cache_releasable_bytes=0,
        )
        self.assertIn("#770 FUNDING POSTS", line)
        self.assertIn("releasable", line)
        self.assertNotIn("derated to zero", line)
        self.assertIn("cause=scarcity", line)

    def test_a_pass_without_the_measurement_is_priced_as_828_did(self):
        """Backward pin on the wiring: no releasable attribute stored (an
        older pass, or the abstaining backend) -> the phantom verdict #828
        shipped, unchanged."""
        line = self._census(
            _last_cache_promised_bytes=334 * MIB,
            _last_cache_delivered_bytes=0,
        )
        self.assertIn("cause=phantom_capacity", line)

    def test_the_census_still_never_raises(self):
        line = self._census(
            _last_cache_promised_bytes=W24_RAW_CACHE,
            _last_cache_delivered_bytes=0,
            _last_cache_releasable_bytes="not-an-int",
        )
        self.assertIsInstance(line, str)


class _Probe:
    """The 631 stand-in: driver/cache readings plus what a reclaim returns."""

    def __init__(self, driver_free_mib, cached_free_mib, returnable_mib=None):
        self.driver_free = int(driver_free_mib) * MIB
        self.cached_free = int(cached_free_mib) * MIB
        self.returnable = (
            self.cached_free if returnable_mib is None else int(returnable_mib) * MIB
        )
        self.reclaims = 0

    def __call__(self):
        return self.driver_free, self.cached_free

    def reclaim(self):
        self.reclaims += 1
        self.driver_free += self.returnable
        self.cached_free -= self.returnable
        self.returnable = 0


def _runtime(probe, releasable=None, reserve_mib=1024):
    r = PhaseFlipRuntime.__new__(PhaseFlipRuntime)
    r._mem_probe = probe
    r._mem_reclaim = probe.reclaim
    r._staging_reserve_bytes = int(reserve_mib) * MIB
    if releasable is not None:
        r._mem_releasable = releasable
    return r


class TheSeamSkipsADrawThatCannotPay(unittest.TestCase):
    def test_a_provably_unreleasable_cache_skips_the_draw(self):
        """THE LOOP'S COST. W24 paid an ``empty_cache()`` sync every 60-75 s
        for 23 minutes, each one measuring the 0 the estimate already knew.
        A measured 0 must refuse without drawing."""
        probe = _Probe(1024, 800, returnable_mib=0)
        r = _runtime(probe, releasable=lambda: 0)
        ok, _ = r._staging_affordable(584 * MIB)
        self.assertFalse(ok)
        self.assertEqual(probe.reclaims, 0)
        # And the measurement is recorded for the census that explains it.
        self.assertEqual(getattr(r, "_last_cache_releasable_bytes", None), 0)

    def test_a_releasable_cache_still_draws_and_still_measures(self):
        """THE DANGER DIRECTION. A skip that fires on nonzero releasable
        would starve law 2 of its measurement and the seam of its bytes."""
        probe = _Probe(1024, 800, returnable_mib=300)
        r = _runtime(probe, releasable=lambda: 300 * MIB)
        ok, _ = r._staging_affordable(584 * MIB)
        self.assertFalse(ok)  # 300 spendable < 584
        self.assertEqual(probe.reclaims, 1)

    def test_an_abstaining_measurement_keeps_the_old_reclaim_path(self):
        """``None`` -> the pre-#852 path exactly: draw, measure, judge."""
        probe = _Probe(1024, 800)
        r = _runtime(probe, releasable=lambda: None)
        ok, _ = r._staging_affordable(584 * MIB)
        self.assertTrue(ok)  # full 800 handed back -> 800 spendable
        self.assertEqual(probe.reclaims, 1)

    def test_a_fit_from_driver_alone_still_never_draws(self):
        """631's hysteresis is untouched: no reclaim when none is needed."""
        probe = _Probe(4096, 800, returnable_mib=0)
        r = _runtime(probe, releasable=lambda: 0)
        ok, _ = r._staging_affordable(584 * MIB)
        self.assertTrue(ok)
        self.assertEqual(probe.reclaims, 0)


class TheEstimatorAbstainsWhereItsArithmeticDoesNotHold(unittest.TestCase):
    """``reserved - allocated - inactive_split`` is only meaningful while
    ``reserved`` describes PHYSICAL bytes. Under ``expandable_segments:True``
    it does not: this tree measured ``reserved`` at 36910 MiB on a 32607 MiB
    card (``phase_flip_spill.py`` :369-377, "it counts a VIRTUAL extent ... it
    cannot be compared to a physical budget at all"), and the same tree already
    refuses a feature outright on that env (``adaptive_graph_memory.py`` :354).

    An estimator that keeps subtracting under that config UNDER-reports, and an
    under-report here suppresses a draw that WOULD have paid -- the danger
    direction of #852, and the one that would make the flip stickier rather
    than looser. Abstention hands the seam back to #828 unchanged.
    """

    # A cache the arithmetic can read: 800 MiB free, 300 of it trapped in
    # segments still in use, so a draw can return 500 MiB.
    WHOLE = {
        "reserved_bytes.all.current": 1000 * MIB,
        "allocated_bytes.all.current": 200 * MIB,
        "inactive_split_bytes.all.current": 300 * MIB,
    }

    def test_the_arithmetic_holds_on_a_normal_allocator(self):
        """THE CAN-FAIL DIRECTION for every abstention below: with no special
        allocator config this MUST produce the figure, or the abstention tests
        would pass against a function that only ever says None."""
        self.assertEqual(releasable_cache_bytes_from_stats(self.WHOLE), 500 * MIB)

    def test_expandable_segments_abstains_instead_of_under_reporting(self):
        self.assertIsNone(
            releasable_cache_bytes_from_stats(
                self.WHOLE, alloc_conf="expandable_segments:True"
            )
        )

    def test_an_unrelated_alloc_conf_still_measures(self):
        """The abstention must key on the segment mode, not on the presence of
        any PYTORCH_CUDA_ALLOC_CONF at all."""
        self.assertEqual(
            releasable_cache_bytes_from_stats(
                self.WHOLE, alloc_conf="max_split_size_mb:256"
            ),
            500 * MIB,
        )

    def test_a_backend_without_the_counter_abstains(self):
        """No ``inactive_split_bytes`` (cudaMallocAsync) -> None, never a guess
        of ``reserved - allocated`` that would re-promise the phantom."""
        self.assertIsNone(
            releasable_cache_bytes_from_stats(
                {
                    "reserved_bytes.all.current": 1000 * MIB,
                    "allocated_bytes.all.current": 200 * MIB,
                }
            )
        )

    def test_an_empty_reservation_abstains(self):
        self.assertIsNone(
            releasable_cache_bytes_from_stats(
                {
                    "reserved_bytes.all.current": 0,
                    "allocated_bytes.all.current": 0,
                    "inactive_split_bytes.all.current": 0,
                }
            )
        )

    def test_an_over_subtraction_floors_at_zero_rather_than_going_negative(self):
        """Counters sampled without a lock can disagree by a block. A negative
        releasable figure would clamp to a promise of zero anyway, but it would
        read as a corrupt census on the way there."""
        self.assertEqual(
            releasable_cache_bytes_from_stats(
                {
                    "reserved_bytes.all.current": 1000 * MIB,
                    "allocated_bytes.all.current": 900 * MIB,
                    "inactive_split_bytes.all.current": 500 * MIB,
                }
            ),
            0,
        )


if __name__ == "__main__":
    unittest.main()
