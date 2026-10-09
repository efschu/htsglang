"""History 08.10.: (1) decode tok/s in total and per batch size bs1..bs6 instead of "per stream",
(2) the input-token curve as LEVELS of the prefill in progress (user: "a jump to the tokens that come from the cache,
then the green curve of the newly prefilled tokens rises above the blue value -- not this odd up from zero and down
again").

The old curve stacked the RATES tok/s of the buckets (from cache, + recomputed P, + recomputed D): a request's cached
prefix is counted once at its first chunk and then smeared over the burst as tok/s, so the stack starts at the smeared
rate, never at the cached length, and falls back to 0 when the burst ends.  ``test_old_rate_stack_*`` reproduces that
with the same example the new levels are checked on.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rigdash import activity, history  # noqa: E402
from rigdash.tests.test_activity_0930 import ring_until  # noqa: E402

N = 60


def model():
    return activity.Model(ring_until(float(N)), [], [])


def chunk(s, e0, tok, cached):
    return {"s": s, "e0": e0, "e": e0, "tok": float(tok), "cached": float(cached), "n": 1, "g": "P"}


# one P request: 50 000 tokens from the cache, 4 x 16 384 newly prefilled, 10 s .. 26 s (a chunk per 4 s)
P_REQ = [chunk(10.0, 14.0, 16384, 50000), chunk(14.0, 18.0, 16384, 0), chunk(18.0, 22.0, 16384, 0), chunk(22.0, 26.0, 16384, 0)]


class TestDecodeByBatchSize(unittest.TestCase):
    def setUp(self):
        m = model()

        def iv(s, e, tok, bmin, bmax):
            return {"s": s, "e": e, "parts": [(s, e)], "dur": e - s, "tok": float(tok), "busy": e - s, "gpu_ms": 1000.0 * (e - s),
                    "bs_min": bmin, "bs_max": bmax, "bs_mean": (bmin + bmax) / 2.0, "seat_s": (e - s) * (bmin + bmax) / 2.0,
                    "steady": True, "run": bmax, "bs": bmax, "stream": None}
        # bs 2 for 10 s at 100 tok/s, bs 2 again for 10 s at 140 tok/s, bs 3 for 10 s at 150 tok/s,
        # a mixed 2..3 stretch (no class), bs 1 for 5 s at 40 tok/s
        m.dec = [iv(0, 10, 1000, 2, 2), iv(10, 20, 1400, 2, 2), iv(20, 30, 1500, 3, 3), iv(30, 40, 900, 2, 3), iv(40, 45, 200, 1, 1)]
        self.b = m.buckets(0.0, N, 1.0)

    def test_classes_take_only_pure_intervals(self):
        b = self.b
        self.assertAlmostEqual(sum(b["dec_bs2_tps"][i] or 0 for i in range(N)), 2400.0)      # 1000 + 1400 tokens
        self.assertAlmostEqual(sum(b["dec_bs2_busy"][i] or 0 for i in range(N)), 20.0)
        self.assertAlmostEqual(sum(b["dec_bs3_tps"][i] or 0 for i in range(N)), 1500.0)      # the 2..3 mix is in no class
        self.assertAlmostEqual(sum(b["dec_bs1_busy"][i] or 0 for i in range(N)), 5.0)
        self.assertTrue(all(v is None for v in b["dec_bs5_tps"] + b["dec_bs6_busy"]))        # empty class: None, never 0

    def test_average_per_class_over_the_shown_stretch(self):
        b = self.b
        series = {"m.dec_tps": b["dec_tps"], "m.dec_busy": b["dec_busy"], "m.dec_seat": b["dec_seat"],
                  "m.seats_min": b["dec_bs_min"], "m.seats_max": b["dec_bs_max"]}
        for k in activity.BS_CLASSES:
            series["m.dec_bs%d_tps" % k], series["m.dec_bs%d_busy" % k] = b["dec_bs%d_tps" % k], b["dec_bs%d_busy" % k]
        t = history.seat_tiles(series, 1.0)["dec_by_bs"]
        self.assertAlmostEqual(t["2"]["rate"], 120.0)       # (1000 + 1400) tok / 20 s
        self.assertAlmostEqual(t["3"]["rate"], 150.0)
        self.assertAlmostEqual(t["1"]["rate"], 40.0)
        self.assertIsNone(t["4"]["rate"])
        self.assertIsNone(t["6"]["rate"])


class TestInputTokenLevels(unittest.TestCase):
    def levels(self):
        m = model()
        m.pchunks = {"P": [dict(c) for c in P_REQ]}
        return m.buckets(0.0, N, 1.0)

    def test_jump_to_the_cached_prefix_then_new_tokens_rise_above_it(self):
        b = self.levels()
        run = range(10, 26)
        # blue: the cached prefix from the FIRST second of the request on, constant (a jump, not a ramp from 0)
        self.assertEqual([b["pf_cached"][i] for i in run], [50000.0] * 16)
        # green (new tokens so far): rises monotonically from the start and ends at the request's own new tokens
        new = [b["pf_new"][i] for i in run]
        self.assertEqual(new, sorted(new))
        self.assertAlmostEqual(new[0], 4096.0)
        self.assertAlmostEqual(new[-1], 4 * 16384.0)
        # the top of the stack = the input length of the request grows from 54 096 to 115 536 (cached + new)
        self.assertAlmostEqual(b["pf_cached"][25] + b["pf_new"][25], 50000 + 65536)

    def test_no_prefill_is_a_gap_not_zero(self):
        b = self.levels()
        for k in ("pf_cached", "pf_new"):
            self.assertTrue(all(b[k][i] is None for i in list(range(0, 10)) + list(range(26, N))), k)

    def test_second_request_jumps_to_its_own_prefix(self):
        m = model()
        m.pchunks = {"P": [dict(c) for c in P_REQ] + [chunk(40.0, 42.0, 3000, 20000)]}
        b = m.buckets(0.0, N, 1.0)
        self.assertEqual(b["pf_cached"][25], 50000.0)
        self.assertEqual(b["pf_cached"][40], 20000.0)             # jumps down to the next request's prefix
        self.assertAlmostEqual(b["pf_new"][41], 3000.0)
        self.assertIsNone(b["pf_cached"][30])

    def test_old_rate_stack_is_not_the_level_curve(self):
        """The old chart: cache + P + D rates per bucket.  It starts at the smeared rate, not at the cached length,
        and falls back to zero after the burst -- the 'from zero up and down' of the user's complaint."""
        b = self.levels()
        stack = [(b["tok_cache"][i] or 0) + (b["tok_comp_p"][i] or 0) for i in range(N)]
        self.assertLess(stack[10], 50000.0 / 2)        # nowhere near the 50 000 cached tokens at the start
        self.assertEqual(stack[30], 0.0)               # and down to 0 again after the burst


if __name__ == "__main__":
    unittest.main()
