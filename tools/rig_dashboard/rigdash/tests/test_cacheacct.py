"""Cache vs. hand-over P->D (user order 29.09. ~13:40Z, DASHBOARD-GRAFIKEN).

"aus Cache" = tokens whose KV existed BEFORE the request (prefix hit at admission).  D reading
from L2/L3 what P just prefilled for the SAME request is the hand-over P->D -- never cache.
Both cases are constructed here, once through the log path (WEG2-SERVED legs paired by rid) and
once through the IPC path (front.served_tokens with the D_after_P row agreed with NF), and both
must give the same classes.  Every prompt token lands in exactly one of cache / P-computed /
hand-over, or (D-direct) cache / D-computed.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rigdash import cacheacct, history  # noqa: E402

# rid A: P -> D.  P prefills 1000-token prompt, 200 of it a real prefix hit on P.  After the flip D
# reads 990 of the 1000 from L2/L3 (the hand-over) and re-extends 10.
# rid B: D-direct (small prefill under X).  Its 400-token prefix is on D from an EARLIER request
# (rid A's KV, handed over before B arrived) -> that IS cache for B.
LEGS = [
    (100.0, "P", 1, "A", 1000, 200, 0),
    (130.0, "D", 2, "A", 1000, 990, 50),
    (200.0, "D", 2, "B", 500, 400, 20),
]


class TestClassify(unittest.TestCase):
    def test_log_path_both_cases(self):
        tot = cacheacct.split_legs(LEGS)
        self.assertEqual(tot, {"cache": 200 + 400, "comp_p": 800, "comp_d": 10 + 100, "handoff": 990})

    def test_each_token_once_per_request(self):
        pr = set()
        a_p = cacheacct.classify_leg("P", "A", 1000, 200, pr)
        pr.add("A")
        a_d = cacheacct.classify_leg("D", "A", 1000, 990, pr)
        # request A: prompt = cache + P-computed; D's leg = hand-over + D's re-extend
        self.assertEqual(a_p["cache"] + a_p["comp_p"], 1000)
        self.assertEqual(a_d["handoff"] + a_d["comp_d"], 1000)
        self.assertEqual(a_d["cache"], 0, "the hand-over must never count as cache")
        b = cacheacct.classify_leg("D", "B", 500, 400, pr)
        self.assertEqual((b["cache"], b["comp_d"], b["handoff"]), (400, 100, 0))

    def test_naive_reading_is_the_trap(self):
        # what the old 'served_D cached / prompt' reading gave: 1390 / 1500 = 93 % "cache"
        naive = (990 + 400) / (1000 + 500)
        right = cacheacct.hit_share(cacheacct.split_legs(LEGS))
        self.assertGreater(naive, 0.9)
        self.assertAlmostEqual(right, 600 / 1510, places=6)

    def test_ipc_path_equals_log_path(self):
        st = {"P": {"n": 1, "prompt": 1000, "cached": 200, "completion": 0},
              "D": {"n": 2, "prompt": 1500, "cached": 1390, "completion": 70},
              "D_after_P": {"n": 1, "prompt": 1000, "cached": 990, "completion": 50}}
        self.assertEqual(cacheacct.from_served_tokens(st), cacheacct.split_legs(LEGS))

    def test_ipc_without_row_refuses(self):
        # today's image: no D_after_P -> no split from the IPC totals (the reader stays on the log)
        st = {"P": {"prompt": 1000, "cached": 200}, "D": {"prompt": 1500, "cached": 1390}}
        self.assertIsNone(cacheacct.from_served_tokens(st))

    def test_tiers_take_the_handoff_out(self):
        st = {"P": {"prompt": 1000, "cached": 200, "cached_tier": {"device": 200, "host": 0, "storage": 0}},
              "D": {"prompt": 1500, "cached": 1390, "cached_tier": {"device": 400, "host": 700, "storage": 290}},
              "D_after_P": {"prompt": 1000, "cached": 990, "cached_tier": {"device": 0, "host": 700, "storage": 290}}}
        self.assertEqual(cacheacct.tiers_from_served_tokens(st),
                         {"device": 600, "host": 0, "storage": 0, "unassigned": 0})

    def test_tiers_partial_detail(self):
        # only some answers carried CachedTokensDetails (today: /generate only): the rest is 'unassigned'
        st = {"P": {"prompt": 1000, "cached": 200},
              "D": {"prompt": 1500, "cached": 1390, "cached_tier": {"device": 100, "host": 0, "storage": 0}},
              "D_after_P": {"prompt": 1000, "cached": 990, "cached_tier": {"device": 0, "host": 0, "storage": 0}}}
        self.assertEqual(cacheacct.tiers_from_served_tokens(st),
                         {"device": 100, "host": 0, "storage": 0, "unassigned": 500})
        # D carries tiers but the hand-over subset does not: the split would count the hand-over
        st["D_after_P"].pop("cached_tier")
        self.assertIsNone(cacheacct.tiers_from_served_tokens(st))

    def test_delta_restart(self):
        prev = {"cache": 10, "comp_p": 10, "comp_d": 10, "handoff": 10}
        cur = {"cache": 15, "comp_p": 3, "comp_d": 10, "handoff": 12}
        self.assertEqual(cacheacct.delta(prev, cur), {"cache": 5, "comp_p": 3, "comp_d": 0, "handoff": 2})

    def test_buckets_keep_rid_across_windows(self):
        # P leg in an earlier (already processed) window: its rid must still mark A's D leg as hand-over
        pr = {"A"}
        b = history.cache_buckets(LEGS[1:], 125.0, 20, 5.0, pr)
        self.assertEqual(b[1]["handoff"], 990)       # t=130 -> bucket 1
        self.assertEqual(b[1]["cache"], 0)
        self.assertEqual(b[15]["cache"], 400)        # t=200 -> bucket 15


if __name__ == "__main__":
    unittest.main()
