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

from sglang.srt.layers.dcp.phase_flip_plan import (
    ordered_layer_waves,
    seam_transient_peaks,
)
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


class TestStagingIsBoundedByTheLayerMap(CustomTestCase):
    """THE ONE-REQUEST LIVELOCK, and the wave split that removes it.

    Successor 23 wedged the live instance with a SINGLE 270032-token
    request at bs=1 (pool 380000, purity strict). The prompt prefilled
    fine in PP; the pp_to_tp flip was then refused --

        staging 3855 MiB needed but only 3102 MiB is spendable

    -- and under strict purity decode may ONLY run in TP, so the request
    could never decode, stayed resident, kept the live set large, and the
    identical refusal repeated at ~1/s forever. /health went 503 and only
    a reboot recovered it.

    WHY IT HAPPENED, from the formula rather than from the log: the move
    was streamed across LAYERS but the seam swapped the two layouts'
    physical backing exactly ONCE, so every byte crossing it had to be
    resident at that instant. ``outgoing``, ``incoming`` and the retained
    ``local`` leg were each ``sum(row_nbytes * n_rows)`` over the whole
    plan, i.e. proportional to the resident live set and unbounded in the
    request length. The affordability gate was therefore HONEST -- it
    priced exactly what the move allocated. The defect was in the move.

    THE FIX is to wave the seam: release the source layout's backing and
    restore the destination's ONE LAYER WAVE AT A TIME, so only one wave's
    share is ever staged. The wave count is a property of the LAYER MAP,
    so what remains is bounded by the pool geometry.

    The fixture is the rig: layer map [28, 20, 16], token vector 14/10/8
    of 32, this rank = 1 (the binding one, 20 of 64 layers). The row width
    is calibrated so the UNWAVED formula reproduces the 3855 MiB that was
    actually measured -- a 2-layer toy makes the demand small enough to be
    affordable and hides the very condition under test.
    """

    #: [28, 20, 16] over 64 full-attention layers, ascending blocks.
    LAYER_MAP = (
        tuple(range(0, 28)),
        tuple(range(28, 48)),
        tuple(range(48, 64)),
    )
    RANK = 1
    #: The wedging request, and the ownership split of its slots under the
    #: 14/10/8 token vector.
    LIVE_ROWS = 270000
    MY_ROWS = LIVE_ROWS * 10 // 32
    PEER0_ROWS = LIVE_ROWS * 14 // 32
    PEER2_ROWS = LIVE_ROWS * 8 // 32
    #: Calibrated so the unwaved peak is the 3855 MiB seen on metal.
    ROW_NBYTES = 543
    POOL_ROWS = 380000
    #: What the gate reported as spendable in the wedged state.
    SPENDABLE_MIB = 3102

    class _Side:
        def __init__(self, row_nbytes, num_layers, num_rows):
            self._n = row_nbytes
            self.num_layers = num_layers
            self.num_rows = num_rows

        def row_nbytes(self, _layer):
            return self._n

    def _plan(self, rows=None):
        rows = self.LIVE_ROWS if rows is None else rows
        scale = rows / self.LIVE_ROWS
        mine = int(self.MY_ROWS * scale)
        mp = self.LAYER_MAP
        me = self.RANK

        class _P:
            direction = "pp_to_tp"
            layer_map = mp
            local_layers = mp[me]
            local_pp_rows = _N(mine)
            local_tp_rows = _N(mine)
            send_layers = {0: mp[me], 2: mp[me]}
            recv_layers = {0: mp[0], 2: mp[2]}

        p = _P()
        p.send_rows = {
            0: _N(int(TestStagingIsBoundedByTheLayerMap.PEER0_ROWS * scale)),
            2: _N(int(TestStagingIsBoundedByTheLayerMap.PEER2_ROWS * scale)),
        }
        p.recv_rows = {0: _N(mine), 2: _N(mine)}
        return p

    def _runtime_on_the_rig(self, driver_free_mib=0, swappable=False):
        r = _runtime(_Probe(driver_free_mib, 0))
        r._map = self.LAYER_MAP
        r._rank = self.RANK
        r._n_layers = 64
        r._n_waves = None
        # The rig's KV token share vector. Needed since #631 2.1b, because
        # the wave ORDER is chosen from it: rank 1's 10/32 is what ``_sides``
        # already encodes in the destination row count below, so the two
        # must not be allowed to drift apart.
        r._vec = (14, 10, 8)
        # #631 2.1b: order, wave-count default and slack accounting are one
        # switch, so the fixture has to state which design it is pricing.
        r._seam_restore_first = True
        r._pre_write_fns = (_FakeWavedSwap(),) if swappable else ()
        # Pool-local index of a global ordinal, both sides: the PP pool
        # holds this stage's block, the TP pool holds every ordinal. Only
        # the widths matter here, so identity is enough.
        r._src_layer_idx = lambda _d, f: 0
        r._dst_layer_idx = lambda _d, f: 0
        return r

    def _sides(self):
        src = self._Side(self.ROW_NBYTES, 20, self.POOL_ROWS)
        dst = self._Side(self.ROW_NBYTES, 64, self.POOL_ROWS * 10 // 32)
        return src, dst

    def test_the_fixture_reproduces_the_measured_unwaved_demand(self):
        """If this drifts, the rest of the class stops being about the wedge."""
        r = self._runtime_on_the_rig()
        src, dst = self._sides()
        unwaved = r._staging_bytes(self._plan(), "pp_to_tp", src, dst)
        self.assertAlmostEqual(unwaved / MIB, 3855, delta=12)

    def test_the_wedging_request_is_now_affordable(self):
        # #856 REPLACES test_the_unwaved_seam_cannot_afford_the_wedging_request.
        # THE FUNDING WIN, in unit form. That test proved the UNWAVED seam
        # could not afford the request that livelocked the instance -- true
        # while staging had to cover a move. The flip no longer moves KV, so
        # the ask no longer carries `wave_peak`, and the request that wedged
        # the seam is affordable on the same spendable budget. This is the
        # W25 counter claim (33 arms refused, 25 on the staging rate limit,
        # 17 FLIP ABANDONED) reduced to one assertion.
        r = self._runtime_on_the_rig()
        src, dst = self._sides()
        need = r._seam_reserve_bytes(self._plan(), "pp_to_tp", src, dst, waves=(None,))
        gate = _runtime(_Probe(self.SPENDABLE_MIB + 1024, 0))
        ok, _detail = gate._staging_affordable(need)
        self.assertTrue(
            ok,
            "the seam still cannot afford the wedging request; wave_peak is "
            "not retired from the ask",
        )

    def test_the_waved_seam_affords_it(self):
        """The wedge, gone, at the spendable figure the rig actually had."""
        r = self._runtime_on_the_rig()
        src, dst = self._sides()
        need = r._staging_bytes(
            self._plan(), "pp_to_tp", src, dst, r._flip_waves("pp_to_tp")
        )
        gate = _runtime(_Probe(self.SPENDABLE_MIB + 1024, 0))
        ok, detail = gate._staging_affordable(need)
        self.assertTrue(ok, detail)

    def test_the_wave_count_is_one_layer_per_wave(self):
        """#631 2.1b replaced the smallest-stage cap with one layer per wave.

        The OLD contract was 16 waves ([28, 20, 16] -> smallest stage) with
        every wave carrying a layer of every stage, because release-first
        made a wave's own releases pay for its own commits. Restore-first
        removes that coupling, so the cap is the layer count and a wave
        containing exactly one stage's layer is now normal and correct.

        What replaces the "every wave touches every stage" invariant is the
        transient-peak contract asserted below -- the property that
        invariant was a proxy for.
        """
        r = self._runtime_on_the_rig()
        waves = r._flip_waves("pp_to_tp")
        self.assertEqual(len(waves), 64)
        seen = [f for w in waves for f in w]
        self.assertEqual(sorted(seen), list(range(64)))
        self.assertTrue(
            all(len(w) == 1 for w in waves),
            "one layer per wave is the point of the lifted cap",
        )

    def test_the_rollback_switch_restores_the_WHOLE_old_design(self):
        """Order, wave count and slack accounting are one switch.

        The falsifier for splitting them. Rolling the order back while
        leaving the count at ``n_layers`` gives a one-layer wave under
        release-first -- a wave with no release of its own to pay for its
        commit, which is the netting rule that set the old cap in the first
        place. Priced on the rig geometry that combination lands at
        354,868 tokens against ~435,000 for release-first W=4, so it is a
        capacity REGRESSION wearing the word 'rollback'.
        """
        r = self._runtime_on_the_rig(swappable=True)
        r._seam_restore_first = False
        waves = r._flip_waves("pp_to_tp")
        self.assertEqual(len(waves), 16, "rollback must restore the smallest-stage cap")
        for w in waves:
            for stage in self.LAYER_MAP:
                self.assertTrue(
                    set(w) & set(stage),
                    f"rollback must restore the proportional split; {w} skips a stage",
                )

    def test_the_slack_accounting_follows_the_order(self):
        """Restore-first must be charged MORE than release-first.

        It holds the wave's commit against the previous wave's releases
        rather than its own, so the same plan is strictly more expensive.
        Equality here would mean the accounting ignored the order, which is
        a false verdict in whichever direction it happens to err.
        """
        src, dst = self._sides()
        waves = ordered_layer_waves(self.LAYER_MAP, (14, 10, 8), 64, "pp_to_tp")
        charged = {}
        for restore_first in (True, False):
            r = self._runtime_on_the_rig(swappable=True)
            r._seam_restore_first = restore_first
            charged[restore_first] = r._backing_slack_bytes("pp_to_tp", src, dst, waves)
        self.assertGreater(
            charged[True],
            charged[False],
            "restore-first holds a wave's commit against the PREVIOUS "
            "wave's releases, so it cannot cost the same or less",
        )

    def test_the_wave_order_keeps_the_transient_off_the_binding_ranks(self):
        """The ORDER is load-bearing, so it is asserted on its own.

        A wave commits its destination before releasing its source, so
        somebody holds both for an instant. The order decides who. The peak
        is measured in full-pool layer spans; the rank with the largest
        share is the card sized to absorb it, and every other rank is a
        binding card whose peak is what the corridor law pays for.

        Asserted against the naive ascending order rather than as a bare
        number, so the test states a COMPARATIVE claim that cannot be
        satisfied by accident.
        """
        r = self._runtime_on_the_rig()
        for direction in ("pp_to_tp", "tp_to_pp"):
            waves = r._flip_waves(direction)
            order = [f for w in waves for f in w]
            peaks = seam_transient_peaks(order, self.LAYER_MAP, r._vec, direction)
            naive = seam_transient_peaks(
                list(range(64)), self.LAYER_MAP, r._vec, direction
            )
            big = max(range(len(r._vec)), key=lambda i: r._vec[i])
            bind = [p for i, p in enumerate(peaks) if i != big]
            bind_naive = [p for i, p in enumerate(naive) if i != big]
            self.assertLessEqual(
                max(bind),
                max(bind_naive),
                f"{direction}: the chosen order is worse for the binding "
                f"ranks than doing nothing: {peaks} vs {naive}",
            )
            self.assertLess(
                max(bind),
                1.0,
                f"{direction}: a binding rank holds a full extra layer span "
                f"at the seam ({peaks}); that is the term the order exists "
                f"to move onto rank {big}",
            )

    def test_waving_no_longer_changes_the_price_at_all(self):
        # #856 REPLACES test_waving_divides_the_peak_by_about_the_wave_count.
        # The wave split existed to divide the MOVE's transient. The flip
        # carries no KV, `wave_peak` is retired from the ask, and the ask is
        # therefore identical however the layers are waved. A difference here
        # would mean the move is still being priced.
        r = self._runtime_on_the_rig()
        src, dst = self._sides()
        plan = self._plan()
        one = r._seam_reserve_bytes(plan, "pp_to_tp", src, dst, waves=(None,))
        many = r._seam_reserve_bytes(
            plan, "pp_to_tp", src, dst, r._flip_waves("pp_to_tp")
        )
        self.assertEqual(one, many)

    def test_the_full_pool_live_set_is_still_affordable(self):
        """THE ACTUAL CLOSURE, and it is stronger than a ratio.

        The live set cannot exceed the pool, so pricing the seam at a FULL
        pool bounds it for every request that can exist in this
        configuration. If this holds, no request length can reach the
        refusal that livelocked -- which is the property the 270k
        reproducer exists to check on metal.
        """
        r = self._runtime_on_the_rig(swappable=True)
        src, dst = self._sides()
        full = self._plan(rows=self.POOL_ROWS)
        need = r._staging_bytes(full, "pp_to_tp", src, dst, r._flip_waves("pp_to_tp"))
        gate = _runtime(_Probe(self.SPENDABLE_MIB + 1024, 0))
        ok, detail = gate._staging_affordable(need)
        self.assertTrue(ok, f"{need / MIB:.0f} MiB at a full pool: {detail}")

    def test_the_swappable_seam_is_charged_its_wave_boundary_slack(self):
        """The one-layer drift is real memory and the gate must own it.

        Integer layer counts cannot always be split proportionally, so a
        wave boundary can carry up to one layer-span of extra residency.
        It is a constant of the pool geometry -- it must be charged, and
        it must NOT grow with the request.
        """
        r_plain = self._runtime_on_the_rig(swappable=False)
        r_swap = self._runtime_on_the_rig(swappable=True)
        src, dst = self._sides()
        waves = r_plain._flip_waves("pp_to_tp")
        small = r_swap._staging_bytes(
            self._plan(rows=1000), "pp_to_tp", src, dst, waves
        ) - r_plain._staging_bytes(self._plan(rows=1000), "pp_to_tp", src, dst, waves)
        big = r_swap._staging_bytes(
            self._plan(), "pp_to_tp", src, dst, waves
        ) - r_plain._staging_bytes(self._plan(), "pp_to_tp", src, dst, waves)
        self.assertGreater(small, 0)
        self.assertEqual(small, big, "the slack must not track the live set")


class _FakeWavedSwap:
    """Just enough of WavedBackingSwap for the pricing path to see a seam."""

    is_swappable = True

    def release_wave(self, direction, wave):
        pass

    def restore_wave(self, direction, wave):
        pass


if __name__ == "__main__":
    unittest.main()
