# SPDX-License-Identifier: Apache-2.0
"""#1317n -- D's L2 is DERIVED from the request cap, and the compensation is gone.

THE INVARIANT THIS RESTORES, in upstream's own words. `pool_host/base.py`
sizes the host pool as `ratio x device_pool` when `--hicache-size` is unset
(ratio 2.0, so L2 >= L1) and MIN-syncs a fixed size across ranks because "the
lockstep schedulers and the host radix index must agree on one slot count".
This fork shipped a fixed 1 GB to BOTH groups -- 30,518 rows on D, a #915
limit of 27,466 -- which broke it, and the #1317 window/chain/anchor layer plus
the 413 band was the compensation. The user ordered the compensation deleted
and the invariant restored (2026-09-10).

WHY THE ABSOLUTE FLAG STAYS RATHER THAN THE RATIO, priced not preferred: D's
profiled per-rank device capacity on boot weg2sn6p was [188788, 262358,
260566] tokens, so `ratio 2.0` against the largest is 17.2 GB on one rank and
51.5 GB across three, against 39 GiB MemAvailable and a reap mark at 95.90 GiB.
And the ratio path is not even admissible here: upstream's own docstring says
"ratio-based sizing already derives from the SYNCED device pool size", which is
true for even TP and false under this fork's uneven DCP -- an audit of the
MIN-synced consumers found `prefetch_budget.py` requiring `--hicache-size`
BECAUSE it is MIN-synced (the fork deleted its own MIN all_reduce in #1068
slice 2 on that basis) and `staging_write_ring.py` naming a rank-dependent
admission bound as the #645 defect. Uniform, derived, priced.
"""

import unittest

from sglang.srt.weg2 import host_ledger as hl

# Boot weg2sn6p, group D.
CAP = 262144
SHARE = 24 / 64          # installed ownership vector [17, 24, 23]
CELL = 32768
FRACTION = 0.9           # HICACHE_LOAD_POOL_USAGE_FRACTION


class TestTheDerivation(unittest.TestCase):
    def test_it_reproduces_the_ordered_numbers_exactly(self):
        t = hl.derive_d_hicache_size_gb(CAP, SHARE, CELL, FRACTION)
        self.assertEqual(int(t["s_gb"]), 4)
        self.assertEqual(int(t["rows_needed"]), 98304)
        self.assertEqual(int(t["rows"]), 122070)
        self.assertEqual(int(t["rows_lendable"]), 109863)
        self.assertAlmostEqual(t["gb_exact"], 3.58, places=2)

    def test_the_pool_holds_a_cap_sized_read_BEHIND_the_load_fraction(self):
        """The whole point: not "the pool is 4 GB" but "the controller will
        lend enough of it for one cap-sized read". The old 1 GB gave 30,518
        rows and a limit of 27,466 against the 98,304 a read needs."""
        t = hl.derive_d_hicache_size_gb(CAP, SHARE, CELL, FRACTION)
        self.assertGreaterEqual(t["rows_lendable"], t["rows_needed"])
        old_rows = int(1 * hl.GB // CELL)
        self.assertLess(int(old_rows * FRACTION), int(t["rows_needed"]))

    def test_a_misread_share_is_refused_not_sized(self):
        for bad in (0.0, -0.1, 1.5):
            with self.assertRaises(ValueError):
                hl.derive_d_hicache_size_gb(CAP, bad, CELL, FRACTION)

    def test_the_provenance_line_carries_every_term(self):
        t = hl.derive_d_hicache_size_gb(CAP, SHARE, CELL, FRACTION)
        line = hl.d_hicache_provenance(t, "flag-default 0.40 vs installed unknown")
        for frag in ("hicache_size=4 GB derived", "cap=262144", "fraction=0.90",
                     "cell=32768", "rows_needed=98304", "rows=122070",
                     "rows_lendable=109863", "flag-default 0.40"):
            self.assertIn(frag, line)


class TestTheTwoGroupsArePricedApart(unittest.TestCase):
    """NO BOOT WITH S_D=S_P IN THE LEDGER: D carries 4 GB where P carries 1,
    which is +8.38 GiB of rings against a reap mark nobody may touch."""

    def _images(self):
        import inspect
        n = len(inspect.signature(hl.ImageTerms).parameters)
        return hl.ImageTerms(*([0.0] * n))

    def test_the_default_is_byte_identical_to_the_old_single_budget(self):
        z = self._images()
        self.assertEqual(
            hl.charge_terms(1, 600, 3, z)["rings_gib"],
            hl.charge_terms(1, 600, 3, z, s_gb_d=1)["rings_gib"],
        )

    def test_the_d_budget_is_charged_and_the_delta_is_the_ordered_one(self):
        z = self._images()
        base = hl.charge_terms(1, 600, 3, z)["rings_gib"]
        got = hl.charge_terms(1, 600, 3, z, s_gb_d=4)["rings_gib"]
        self.assertAlmostEqual(got - base, 8.38, places=2)
        self.assertAlmostEqual(
            got, (hl.RING_P_MULT_GB_PER_S * 1 + hl.RING_D_MULT_GB_PER_S * 4)
            * hl.GB / hl.GIB, places=6)

    def test_the_arm_line_prints_which_d_budget_it_charged(self):
        z = self._images()
        priced = hl.charge_terms(1, 600, 3, z, s_gb_d=4)
        self.assertEqual(hl._arm_s_d(type("A", (), {"terms": priced})()), "4")
        # An arm with no D term says so rather than printing P's number.
        self.assertEqual(hl._arm_s_d(type("A", (), {"terms": {}})()), "=S")


class TestTheCompensationIsGone(unittest.TestCase):
    """Deletion is the deliverable, so absence is asserted rather than assumed
    -- an emitter that survives in one module is the second bookkeeping the
    user ordered removed."""

    def test_no_window_chain_or_anchor_census_survives(self):
        import inspect

        from sglang.srt.managers import scheduler
        from sglang.srt.mem_cache import unified_radix_cache
        from sglang.srt.mem_cache.unified_cache_components import mamba_component
        from sglang.srt.weg2 import front

        for mod in (scheduler, unified_radix_cache, mamba_component, front):
            src = inspect.getsource(mod)
            for gone in ("WINDOW-REISSUE", "release_staged_window",
                         "_weg2_issue_next_window", "#1317m",
                         "WINDOW-RELEASE", "exempt_carrier_exceeds",
                         "windowed_carrier", "_weg2_window_alloc_cap"):
                self.assertNotIn(gone, src, f"{mod.__name__} still carries {gone}")

    def test_design_a_and_the_1246_bound_are_KEPT(self):
        """The delete list is not "everything #1317 touched": the X gate is the
        phase law, not L2 compensation, and the carrier bound is what the
        acceptance reads against cap x share."""
        from sglang.srt.managers.scheduler import Scheduler

        self.assertTrue(hasattr(Scheduler, "_weg2_x_refuses"))
        self.assertTrue(hasattr(Scheduler, "_weg2_host_carry_tokens"))
        self.assertTrue(hasattr(Scheduler, "_weg2_local_store_matches"))


if __name__ == "__main__":
    unittest.main()
