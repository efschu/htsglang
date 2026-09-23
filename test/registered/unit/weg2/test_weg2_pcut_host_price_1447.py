# SPDX-License-Identifier: Apache-2.0
"""#1447: the P-cut solver prices each frontier candidate's pinned host bounce
beside its makespan and ships the fastest one the host can fund.

Boot weg2xsn204: the cap-floor rule shipped 40,12,12 and the ledger refused
the launch (W97, run peak 105 GiB) -- that cut's exchange bounce is priced at
the WORST-CASE lane count (no boot measured it), 27.75 vs 15.75 GiB.  Now
every frontier row at/above the floor is priced through the SAME chain the
ledger uses (host_price_for_cut) and the first fundable one ships; the
operator can grant room by number (SGLANG_WEG2_PCUT_BOUNCE_SLACK_GIB); the
cap-floor rule is the default again on the exchange arm.

Hermetic: the pure picker with fake rows and a fake price; AST pins for the
wiring.  No checkpoint headers are read.
"""
import inspect
import os
import unittest
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import launcher
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _row(layers, ms, pool):
    return SimpleNamespace(layers=layers, attn=(9, 4, 3), makespan_ms=ms, pool_tokens=pool)


ROWS = [_row((44, 10, 10), 536.0, 300000), _row((42, 11, 11), 600.0, 330000),
        _row((39, 13, 12), 684.0, 394648)]
PRICE = {(44, 10, 10): (22.12, 6, "WORST-CASE"), (42, 11, 11): (22.12, 6, "WORST-CASE"),
         (39, 13, 12): (18.5, 5, "measured")}


def _price(row):
    return PRICE[tuple(row.layers)]


class Picker(CustomTestCase):
    def test_no_slack_ships_the_incumbent_priced_cut(self):
        picked, lines = launcher.host_priced_pick(ROWS, _price, 18.5, 0.0)
        self.assertEqual(tuple(picked.layers), (39, 13, 12))
        self.assertEqual(len(lines), 3)           # every row priced and printed
        self.assertIn("over the host budget", lines[0])
        self.assertIn("FUNDABLE", lines[2])
        self.assertIn("WORST-CASE", lines[0])

    def test_slack_buys_the_faster_cut(self):
        picked, _ = launcher.host_priced_pick(ROWS, _price, 18.5, 4.0)
        self.assertEqual(tuple(picked.layers), (44, 10, 10))   # fastest first

    def test_nothing_fundable_is_none(self):
        picked, lines = launcher.host_priced_pick(ROWS, _price, 10.0, 0.0)
        self.assertIsNone(picked)
        self.assertEqual(len(lines), 3)

    def test_no_exchange_arm_prices_zero(self):
        ns = SimpleNamespace(weg2_weight_source=launcher.WEIGHT_SOURCE_DEFAULT,
                             weg2_xchg_oncard=launcher.ONCARD_MODE_DEFAULT)
        gib, lanes, src = launcher.host_price_for_cut(ns, "39,13,12")
        self.assertEqual((gib, lanes, src), (0.0, 1, "no exchange"))


class ExchangeArmPrice(CustomTestCase):
    def test_exchange_arm_converts_bytes_to_gib(self):
        """The exchange path, with the two producers faked: catches a NameError
        in the conversion (boot weg2xsn207's preflight died on 'GIB')."""
        ns = SimpleNamespace(weg2_weight_source=launcher.WEIGHT_SOURCE_EXCHANGE,
                             weg2_xchg_oncard="host", model="/nonexistent",
                             weg2_xchg_oncard_slot_mib=None, xchg_bounce_depth=1,
                             xchg_lanes_concurrent=None, xchg_band_credit=False,
                             weg2_xchg_legs="both")
        calls = {}
        orig_lanes, orig_terms = launcher.xchg_lane_count, launcher.xchg_bounce_terms_for_arm

        def fake_lanes(stage_ratio, legs, d_vector=launcher.XCHG_D_VECTOR_DEFAULT):
            calls["cut"] = (stage_ratio, legs)
            return 5, "WEG2-XCHG-LANES cut=x lanes=5 source=measured boot=t"

        def fake_terms(weight_source, oncard_mode, model_dir, oncard_slot_mib, depth,
                       n_lanes, lanes_concurrent, band_credit):
            calls["terms"] = (weight_source, oncard_mode, depth, n_lanes, lanes_concurrent, band_credit)
            return 15.75 * (1 << 30), ["line"]

        launcher.xchg_lane_count, launcher.xchg_bounce_terms_for_arm = fake_lanes, fake_terms
        try:
            gib, lanes, src = launcher.host_price_for_cut(ns, "39,13,12")
        finally:
            launcher.xchg_lane_count, launcher.xchg_bounce_terms_for_arm = orig_lanes, orig_terms
        # #1464: the exchange+host arm is the REGION form -- the cut pick is
        # priced at the region, the lane term is not consulted for the price
        self.assertAlmostEqual(gib, launcher.weight_exchange_region.REGION_BYTES / (1 << 30))
        self.assertEqual((lanes, src), (5, "region-form measured"))
        self.assertEqual(calls["cut"], ("39,13,12", "both"))
        self.assertNotIn("terms", calls)

    def test_lane_price_env_restores_the_file_bounce_price(self):
        """#1464: SGLANG_WEG2_PCUT_LANE_PRICE=1 prices the cut pick by the
        per-lane file bounce again (diagnosis only)."""
        import os
        ns = SimpleNamespace(weg2_weight_source=launcher.WEIGHT_SOURCE_EXCHANGE,
                             weg2_xchg_oncard="host", model="/nonexistent",
                             weg2_xchg_oncard_slot_mib=None, xchg_bounce_depth=1,
                             xchg_lanes_concurrent=None, xchg_band_credit=False,
                             weg2_xchg_legs="both")
        orig_lanes, orig_terms = launcher.xchg_lane_count, launcher.xchg_bounce_terms_for_arm
        launcher.xchg_lane_count = lambda sr, legs, d_vector=launcher.XCHG_D_VECTOR_DEFAULT: (
            9, "WEG2-XCHG-LANES cut=x lanes=9 source=WORST-CASE")
        launcher.xchg_bounce_terms_for_arm = lambda *a: (27.75 * (1 << 30), ["line"])
        os.environ["SGLANG_WEG2_PCUT_LANE_PRICE"] = "1"
        try:
            gib, lanes, src = launcher.host_price_for_cut(ns, "43,11,10")
        finally:
            os.environ.pop("SGLANG_WEG2_PCUT_LANE_PRICE", None)
            launcher.xchg_lane_count, launcher.xchg_bounce_terms_for_arm = orig_lanes, orig_terms
        self.assertAlmostEqual(gib, 27.75)
        self.assertEqual((lanes, src), (9, "WORST-CASE"))

    def test_region_form_prices_every_cut_the_same_so_the_fastest_ships(self):
        """#1464: an unmeasured cut (WORST-CASE 9 lanes) costs the region
        like the incumbent, so host_priced_pick takes the fastest row."""
        ns = SimpleNamespace(weg2_weight_source=launcher.WEIGHT_SOURCE_EXCHANGE,
                             weg2_xchg_oncard="host", model="/nonexistent",
                             weg2_xchg_oncard_slot_mib=None, xchg_bounce_depth=1,
                             xchg_lanes_concurrent=None, xchg_band_credit=False,
                             weg2_xchg_legs="both")
        orig_lanes = launcher.xchg_lane_count
        launcher.xchg_lane_count = lambda sr, legs, d_vector=launcher.XCHG_D_VECTOR_DEFAULT: (
            (5, "WEG2-XCHG-LANES cut=x lanes=5 source=measured boot=t") if sr == "39,13,12"
            else (9, "WEG2-XCHG-LANES cut=x lanes=9 source=WORST-CASE"))
        try:
            inc_gib, _l, _s = launcher.host_price_for_cut(ns, "39,13,12")
            rows = [_row((43, 11, 10), 553.6, 285776), _row((39, 13, 12), 667.3, 401020)]
            picked, lines = launcher.host_priced_pick(
                rows, lambda r: launcher.host_price_for_cut(ns, ",".join(map(str, r.layers))),
                inc_gib, 0.0)
        finally:
            launcher.xchg_lane_count = orig_lanes
        self.assertEqual(picked.layers, (43, 11, 10))
        self.assertIn("region-form WORST-CASE", lines[0])
        self.assertIn("FUNDABLE", lines[0])


class Wiring(CustomTestCase):
    def test_solver_applies_the_host_price(self):
        src = inspect.getsource(launcher.solve_p_cut)
        self.assertIn("host_priced_pick(", src)
        self.assertIn("host_price_for_cut(", src)
        self.assertIn("PCUT_BOUNCE_SLACK_ENV", src)
        # the ledger keeps the last word: the pick only re-reads the makespan choice
        self.assertLess(src.index("pick_shipped_cut("), src.index("host_priced_pick("))
        self.assertLess(src.index("host_priced_pick("), src.index("refuse_shipped_below_floors("))
        # cap floor default follows the arm
        self.assertIn('os.environ.get("SGLANG_WEG2_PCUT_CAP_FLOOR", _cap_floor_default)', src)
        self.assertEqual(launcher.PCUT_BOUNCE_SLACK_ENV, "SGLANG_WEG2_PCUT_BOUNCE_SLACK_GIB")
        # the host-price incumbent is the ORDERED default cut (the measured lane
        # set), never the scores' 32,18,14 (desk probe 2026-09-16 priced 27.75 GiB there)
        self.assertIn("_inc_cut = _csv(DEFAULT_PP_ORDERED_CUT)", src)
        self.assertEqual(tuple(launcher.DEFAULT_PP_ORDERED_CUT), (39, 13, 12))


if __name__ == "__main__":
    unittest.main()
