"""#1464b: the REGION form charges the lane term as a coverage constant.

xsn220 (32a8193805): the solver shipped 43,11,10 (#1464) and the ledger
charged its WORST-CASE 9 lanes = 27.75 GiB -> predicted run peak 103.00 GiB
-> W97 at launch, for per-lane buffers the region form never allocates.
`price_lane_cap` caps ONLY `lanes_priced`; `n_lanes` (the W102 guard count)
and `lanes_concurrent` (the LanePermit) are untouched.
"""
import os
import unittest
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import checkpoint_census, host_ledger, launcher
from sglang.srt.weg2 import xchg_bounce as xb
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-weg2-unit")


def _terms(**kw):
    base = dict(bytes_per_direction=64 * (1 << 30), n_layers=64,
                widest_layer_bytes=756323776, pairs=6, depth=1,
                max_tag_bytes=2907 * (1 << 20))
    base.update(kw)
    return xb.bounce_terms(**base)


class Test1464bPriceLaneCap(unittest.TestCase):
    def test_cap_prices_fewer_lanes_but_publishes_the_full_count(self):
        t9 = _terms(n_lanes=9)
        t9c = _terms(n_lanes=9, price_lane_cap=5)
        t5 = _terms(n_lanes=5)
        self.assertEqual((t9.lanes_priced, t9c.lanes_priced, t5.lanes_priced), (9, 5, 5))
        self.assertEqual(t9c.n_lanes, 9)                     # W102 guard count
        self.assertEqual(t9c.lanes_concurrent, 0)            # no LanePermit
        self.assertEqual(t9c.total_bytes, t5.total_bytes)    # the coverage constant
        self.assertGreater(t9.total_bytes, t9c.total_bytes)

    def test_cap_never_inflates_and_unstated_is_byte_identical(self):
        self.assertEqual(_terms(n_lanes=3, price_lane_cap=5).lanes_priced, 3)
        self.assertEqual(_terms(n_lanes=9, price_lane_cap=0).total_bytes, _terms(n_lanes=9).total_bytes)
        self.assertEqual(_terms(n_lanes=9, lanes_concurrent=2, price_lane_cap=5).lanes_priced, 2)

    def test_published_term_carries_the_cap_and_the_rank_recomputes_it(self):
        t = _terms(n_lanes=9, price_lane_cap=5)
        back = xb.read_published_terms(xb.publish_terms(t))
        self.assertEqual((back.n_lanes, back.price_lane_cap, back.lanes_priced), (9, 5, 5))
        self.assertEqual(back.total_bytes, t.total_bytes)

    def test_region_form_cap_is_the_incumbents_measured_count(self):
        self.assertEqual(launcher.region_form_lane_price_cap("tp3", "both"), 5)
        self.assertEqual(launcher.region_form_lane_price_cap("tp3", "d2p-only-nonexistent"), 0)

    def test_terms_for_arm_forwards_the_cap_to_the_census(self):
        seen = {}
        orig = checkpoint_census.widest_layer_terms

        def fake(model_dir, **kw):
            seen.update(kw)
            return _terms(n_lanes=int(kw["n_lanes"]), price_lane_cap=int(kw["price_lane_cap"])), "widest", "w"

        checkpoint_census.widest_layer_terms = fake
        try:
            nbytes, lines = launcher.xchg_bounce_terms_for_arm(
                launcher.WEIGHT_SOURCE_EXCHANGE, "host", "/nonexistent", None, 1, 9, 0, False,
                price_lane_cap=5)
        finally:
            checkpoint_census.widest_layer_terms = orig
        self.assertEqual((seen["n_lanes"], seen["price_lane_cap"]), (9, 5))
        self.assertEqual(nbytes, _terms(n_lanes=5).total_bytes)


if __name__ == "__main__":
    unittest.main()
