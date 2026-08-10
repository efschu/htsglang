"""#631: a flip must not spend the VRAM reserve to stage its own bytes.

MEASURED FAILURE this pins (2026-08-09, metal, all three ranks died):

    one session, 276214 tokens, 0.995 of the PP pool
    -> policy flip -> kv_reshard._exchange -> torch.empty(584 MiB)
    -> torch.OutOfMemoryError ("600.38 MiB is free")
    -> Fatal Python error: Aborted

The flip is not free of memory. It packs this rank's outgoing rows and
pre-allocates one receive buffer per peer, out of the SAME device memory
the KV pool has been filling. Nothing checked whether those buffers fit,
so a full pool turned a routine layout change into an instance-wide crash
-- and the free VRAM at that moment (600 MiB) was already below the
1024 MiB corridor floor, so even a flip that had squeezed through would
have broken the corridor.

The fix folds an affordability term into the EXISTING pre-flight verdict,
the one that already refuses a flip whose live set does not fit the target
pool. Same properties, deliberately: computed from the plan before a byte
is allocated, reduced across ranks so the abandon is unanimous, and
answered by abandoning the FLIP rather than the SERVER.

Two quantities that must not be conflated, and both directions are wrong:

* the allocator's cached-but-unhanded-out bytes are reusable and invisible
  to NVML, so spending them cannot move the corridor number -- counting
  only driver-free would abandon flips that fit;
* driver bytes DO move it, so only the amount above the reserve is
  spendable -- counting the allocator's view alone would spend the
  corridor.
"""

import unittest

from sglang.srt.managers.phase_flip_runtime import (
    DEFAULT_STAGING_RESERVE_BYTES,
    PhaseFlipRuntime,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

MIB = 1024 * 1024


class _Probe:
    """A stand-in for the two memory readings, in MiB for readability.

    ``returnable_mib`` is how much of the cache a reclaim would actually
    hand back to the driver. It defaults to all of it -- the case the old
    formula silently assumed everywhere -- and the interesting tests are
    the ones that set it lower, because a 584 MiB buffer cannot be cut out
    of cache that is scattered across blocks too small to serve it.
    """

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


def _runtime(probe, reserve_mib=1024):
    r = PhaseFlipRuntime.__new__(PhaseFlipRuntime)
    r._mem_probe = probe
    r._mem_reclaim = probe.reclaim
    r._staging_reserve_bytes = int(reserve_mib) * MIB
    return r


class TestStagingReserveDefault(CustomTestCase):
    def test_the_default_is_the_corridor_floor(self):
        # 1024 MiB, the same promise --rank-user-reserve-mib makes. If this
        # ever drifts, the flip starts spending the user's reserve.
        self.assertEqual(DEFAULT_STAGING_RESERVE_BYTES, 1024 * MIB)


class TestStagingAffordable(CustomTestCase):
    def test_the_measured_crash_is_now_refused(self):
        # The metal numbers: 600 MiB driver-free, nothing cached, a 584 MiB
        # buffer. Previously allocated and fatal; now refused.
        ok, detail = _runtime(_Probe(600, 0))._staging_affordable(584 * MIB)
        self.assertFalse(ok)
        self.assertIn("584", detail)
        self.assertIn("reserve", detail)

    def test_room_above_the_reserve_is_spendable(self):
        # 2048 free, reserve 1024 -> 1024 spendable; 584 fits.
        ok, detail = _runtime(_Probe(2048, 0))._staging_affordable(584 * MIB)
        self.assertTrue(ok)
        self.assertEqual(detail, "")

    def test_the_allocator_cache_counts_once_it_has_been_handed_back(self):
        # Driver-free is AT the reserve (nothing spendable from the driver),
        # but the allocator holds 800 MiB it CAN hand back. The verdict is
        # unchanged from the old formula -- affordable -- but it is now
        # reached by materialising the cache rather than by promising it,
        # so the corridor reading is restored as a side effect.
        probe = _Probe(1024, 800)
        ok, _ = _runtime(probe)._staging_affordable(584 * MIB)
        self.assertTrue(ok)
        self.assertEqual(probe.reclaims, 1)
        self.assertEqual(probe.driver_free, (1024 + 800) * MIB)

    def test_cache_that_cannot_be_handed_back_is_not_spendable(self):
        # THE FALSIFIER for the superseded formula, and the reason this
        # changed. Same 800 MiB of cache, but fragmented: a reclaim returns
        # nothing usable. The old `usable = cached_free + max(0, driver_free
        # - reserve)` blessed a 584 MiB staging here; the allocator would
        # then have gone to the DRIVER for it and spent the user's corridor,
        # or raised the OutOfMemoryError out of _exchange that took all
        # three ranks down. Refused now.
        probe = _Probe(1024, 800, returnable_mib=0)
        ok, detail = _runtime(probe)._staging_affordable(584 * MIB)
        self.assertFalse(ok)
        self.assertEqual(probe.reclaims, 1)
        self.assertIn("584", detail)

    def test_a_partial_hand_back_is_credited_only_for_what_came_back(self):
        # 800 MiB cached, only 300 returnable -> 300 spendable, 584 refused.
        # The middle case: neither "trust the whole cache" nor "trust none".
        probe = _Probe(1024, 800, returnable_mib=300)
        ok, _ = _runtime(probe)._staging_affordable(584 * MIB)
        self.assertFalse(ok)
        ok2, _ = _runtime(_Probe(1024, 800, returnable_mib=600))._staging_affordable(
            584 * MIB
        )
        self.assertTrue(ok2)

    def test_no_reclaim_when_the_driver_alone_already_affords_it(self):
        # Hysteresis: empty_cache is not free, and a flip that fits from
        # driver-free with the corridor intact must not pay for one.
        probe = _Probe(4096, 800)
        ok, _ = _runtime(probe)._staging_affordable(584 * MIB)
        self.assertTrue(ok)
        self.assertEqual(probe.reclaims, 0)

    def test_the_corridor_keeper_reclaims_even_when_staging_fits(self):
        # driver_free 900 is already UNDER the 1024 reserve while the
        # allocator sits on 800 MiB. Staging is tiny and would fit either
        # way, but the user's corridor law is continuous, so the hoard goes
        # back regardless.
        probe = _Probe(900, 800)
        ok, _ = _runtime(probe)._staging_affordable(1 * MIB)
        self.assertTrue(ok)
        self.assertEqual(probe.reclaims, 1)

    def test_the_cache_does_not_excuse_dipping_below_the_reserve(self):
        # 200 MiB cached, driver free BELOW the reserve: spendable is the
        # cache only, so a 584 MiB staging is still refused. The falsifier
        # for "just add the two numbers".
        ok, _ = _runtime(_Probe(700, 200))._staging_affordable(584 * MIB)
        self.assertFalse(ok)

    def test_exactly_fitting_is_allowed(self):
        # Boundary: spendable == needed. Refusing here would abandon flips
        # that provably fit.
        ok, _ = _runtime(_Probe(1024 + 584, 0))._staging_affordable(584 * MIB)
        self.assertTrue(ok)

    def test_one_byte_over_is_refused(self):
        ok, _ = _runtime(_Probe(1024 + 584, 0))._staging_affordable(584 * MIB + 1)
        self.assertFalse(ok)

    def test_a_larger_reserve_is_honoured(self):
        # The reserve is configuration, not a constant: at 2048 MiB the same
        # memory picture refuses.
        ok, _ = _runtime(_Probe(2048, 0), reserve_mib=2048)._staging_affordable(
            584 * MIB
        )
        self.assertFalse(ok)


class TestStagingBytesFromThePlan(CustomTestCase):
    """The byte count must match what the move actually allocates.

    REWRITTEN 2026-08-10 (successor 22). The superseded contract -- which
    the previous version of this class pinned by name,
    ``test_outgoing_is_counted_twice_and_incoming_once`` -- was wrong in
    two independent ways, and the test froze both:

    * outgoing was doubled to model a packer that concatenated per-layer
      reads. That packer is gone: ``_pack_outgoing`` fills one exact-size
      buffer in place, measured at the plan's floor on the hermetic
      three-rank flip (39.4 -> 19.2 MiB peak);
    * THE LOCAL RETAINED LEG WAS NOT COUNTED AT ALL, and on the rig it is
      the biggest of the three whenever a rank keeps more than it sends.
      That is the accounting hole behind every affordability refusal in
      this feature, including the pool-500000 livelock.

    A test pinning a formula is only as good as the formula, and this one
    made the hole look deliberate for five successors. The contract now is
    ``incoming + max(outgoing, local) + one_layer_window`` -- a MAX and
    not a sum, because the send buffers are released the moment the
    exchange returns and the local leg is not read until after that.
    """

    class _Plan:
        # One send peer (2 layers x 4 rows), one recv peer (3 layers x 2
        # rows), and a retained leg of 2 layers x 20 rows -- shaped like
        # PhaseFlipTransition's fields. The local leg is deliberately the
        # LARGEST, which is the case the superseded formula could not see.
        def __init__(self):
            self.send_layers = {1: [0, 1]}
            self.recv_layers = {1: [0, 1, 2]}
            self.send_rows = {1: _N(4)}
            self.recv_rows = {1: _N(2)}
            self.local_layers = (0, 1)
            self.local_pp_rows = _N(20)
            self.local_tp_rows = _N(20)

    class _Side:
        def __init__(self, row_nbytes, num_layers=3):
            self._n = row_nbytes
            self.num_layers = num_layers

        def row_nbytes(self, _layer):
            return self._n

    def _prepared(self):
        r = _runtime(_Probe(0, 0))
        r._src_layer_idx = lambda _d, f: f
        r._dst_layer_idx = lambda _d, f: f
        return r

    def test_the_peak_is_incoming_plus_the_larger_of_outgoing_and_local(self):
        r = self._prepared()
        src = self._Side(1000)
        dst = self._Side(100)
        got = r._staging_bytes(self._Plan(), "pp_to_tp", src, dst)
        # out   : 2 layers *  4 rows * 1000 + 8    =   8008
        # in    : 3 layers *  2 rows *  100 + 8    =    608
        # local : 2 layers * 20 rows * 1000        =  40000  <- dominates
        # window: widest 20 rows * widest row 1000 =  20000
        self.assertEqual(got, 608 + 40000 + 20000)

    def test_the_superseded_formula_would_have_under_reserved_here(self):
        """The hole, stated as arithmetic so it cannot come back quietly."""
        r = self._prepared()
        src = self._Side(1000)
        dst = self._Side(100)
        got = r._staging_bytes(self._Plan(), "pp_to_tp", src, dst)
        superseded = 2 * 8008 + 608  # 2 x outgoing + incoming
        self.assertLess(
            superseded,
            got,
            "the retained leg dominates in this plan, so the old formula "
            "must be short of the new one -- if it is not, the fixture no "
            "longer exercises the defect it exists for",
        )

    def test_outgoing_dominating_does_not_add_the_local_leg_on_top(self):
        """A MAX, not a sum: over-reserving is not the safe direction.

        The gate's only action is to refuse, and a refusal does not drain
        the resident set it refused on, so every MiB of invented headroom
        moves the livelock to a smaller request.
        """
        r = self._prepared()
        plan = self._Plan()
        plan.send_rows = {1: _N(400)}  # outgoing now far exceeds local
        src = self._Side(1000)
        dst = self._Side(100)
        got = r._staging_bytes(plan, "pp_to_tp", src, dst)
        outgoing = 2 * 400 * 1000 + 8
        local = 2 * 20 * 1000
        self.assertEqual(got, 608 + outgoing + 400 * 1000)
        self.assertLess(got, 608 + outgoing + local + 400 * 1000)

    def test_an_empty_plan_stages_nothing(self):
        r = _runtime(_Probe(0, 0))
        plan = self._Plan()
        plan.send_layers = {}
        plan.recv_layers = {}
        plan.send_rows = {}
        plan.recv_rows = {}
        plan.local_layers = ()
        plan.local_pp_rows = _N(0)
        plan.local_tp_rows = _N(0)
        # Sides are None on purpose: with nothing to stage the formula must
        # not reach for a row width at all.
        self.assertEqual(r._staging_bytes(plan, "pp_to_tp", None, None), 0)


class _N:
    """Minimal stand-in for a row tensor: only numel() is read."""

    def __init__(self, n):
        self._n = n

    def numel(self):
        return self._n


if __name__ == "__main__":
    unittest.main()
