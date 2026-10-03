# SPDX-License-Identifier: Apache-2.0
"""rc12f (NF rc12e f078f89a08, D-TP0, 27.09. 03:32-03:39Z): the trim threshold under-priced the extend.

rc12e trimmed the allocator cache before a D extend when the card held less
than floor + booked activation (767 + 1104 = 1871 MiB). Three 4096-row extends
started ABOVE that and still took the card to the edge, because one extend
window adds about twice its transient to torch reserved (new segments on a
cache it cannot reuse) -- WEG2-VRAM-PEAK, verbatim numbers:

  03:32:08 card_free 2349 -> 327  transient 989  reserved +1992
  03:33:37 card_free 2041 -> 1183 transient 917  reserved +2026  alloc_retries=1
  03:39:12 card_free 2371 -> 67   transient 920  reserved +2304

The measured maximum growth (2304) exceeds 2 x activation (2208), so the growth
itself is the post: ``D_EXTEND_GROWTH_MIB`` [2304, null, null], threshold =
floor + max(activation, growth) = 3071 on TP0. It is measured against the
window start, never against the threshold: no ratchet.
"""

import importlib.util
import os
import unittest
from pathlib import Path

from sglang.test.test_utils import CustomTestCase

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import extend_trim as ET  # noqa: E402
from sglang.srt.weg2 import launcher as L  # noqa: E402

MIB = 1 << 20
FLOOR, ACT = 767, 1104

#: (time, rows, transient, reserved at window start, peak reserved, card_free before)
RC12E = (
    ("03:32:08", 3995, 989, 28494, 30486, 2349),
    ("03:33:37", 4096, 917, 28692, 30718, 2041),
    ("03:39:12", 4096, 920, 28184, 30488, 2371),
)


class Card:
    """What maybe_trim reads: free bytes before, and an empty_cache that
    releases the cache and hands it back to the card."""

    def __init__(self, free_mib, cached_mib=900):
        self.free = free_mib
        self.cached = cached_mib
        self.empty_calls = 0

    def is_current_stream_capturing(self):
        return False

    def mem_get_info(self):
        return self.free * MIB, 32088 * MIB

    def memory_reserved(self):
        return (28000 + self.cached) * MIB

    def synchronize(self):
        pass

    def empty_cache(self):
        self.empty_calls += 1
        self.free += self.cached
        self.cached = 0


def _threshold(env):
    return float(env.split(",")[0])


def _rc12c_terms():
    p = Path(__file__).with_name("test_weg2_d_awake_rest_rc12c.py")
    spec = importlib.util.spec_from_file_location("_rc12c_terms_growth", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TheRecordIsTheMeasuredGrowth(CustomTestCase):
    def test_record_is_the_maximum_of_the_extend_windows(self):
        vals, boots = L.d_extend_growth_record("nextflash")
        measured = ET.extend_growth_mib([(r, t, s, p) for _ts, r, t, s, p, _cf in RC12E])
        self.assertEqual(measured, 2304)
        # 29.09.: the z30x2 flip boot (…_113056) raised TP0 to 2434 and measured the
        # 3080 ranks (2176 / 1728); rc12e's 2304 stays a window of the record
        self.assertEqual(vals, [2434.0, 2176.0, 1728.0])
        self.assertIn("dkrnfh91bar1dauer09270311", boots)
        self.assertIn("dkrnfh91dprsavisadoptstcutvsyncodx2bswre2cutz30x2bar1dauer09291130", boots)
        # the growth exceeds 2 x activation, so the growth itself is the post
        self.assertGreater(measured, 2 * ACT)

    def test_small_extends_do_not_price_it(self):
        self.assertIsNone(ET.extend_growth_mib([(511, 275, 28000, 28776)]))
        self.assertEqual(ET.extend_growth_mib([(603, 275, 28000, 28776)]), 776)

    def test_27b_has_no_record(self):
        self.assertEqual(L.d_extend_growth_record("qwen27b"), (None, ""))


class TheThresholdCoversTheExtend(CustomTestCase):
    def test_launcher_writes_floor_plus_growth(self):
        T = _rc12c_terms()
        terms = []
        _c, budgets = T._d_pass("nextflash", [], terms=terms)
        led = L.d_card_ledger(terms, budgets, "D")
        fits = T._fits(budgets, (90, 48, 48))
        g, _src = L.d_extend_growth_record("nextflash")
        # floor + max(activation, growth) on every rank: 767+2434, 700+2176, 701+1728
        self.assertEqual(L.d_extend_trim_env(led, fits, g), "3201,2876,2429")
        # the 3080 ranks no longer keep floor + activation (1724/1725 let TP1/TP2
        # fall to card_free 16/78 MiB with 4 retries in …_113056)
        self.assertNotEqual(L.d_extend_trim_env(led, fits, g).split(",")[1:],
                            L.d_extend_trim_env(led, fits).split(",")[1:])

    def test_the_three_rc12e_windows_trim_now_and_did_not_before(self):
        new = float(FLOOR + 2304)
        old = float(FLOOR + ACT)
        for ts, _r, _t, _s, _p, cf in RC12E:
            card = Card(cf)
            self.assertIsNone(ET.maybe_trim(card, 0, old), ts)
            self.assertEqual(card.empty_calls, 0, ts)
            card = Card(cf)
            line = ET.maybe_trim(card, 0, new)
            self.assertIsNotNone(line, ts)
            self.assertIn("threshold=3071", line)
            self.assertEqual(card.empty_calls, 1, ts)

    def test_no_trim_with_room(self):
        card = Card(3071)
        self.assertIsNone(ET.maybe_trim(card, 0, float(FLOOR + 2304)))
        self.assertEqual(card.empty_calls, 0)

    def test_growth_below_activation_changes_nothing(self):
        self.assertEqual(ET.launcher_thresholds([767.0], [1104.0], [900.0]), "1871")
        self.assertEqual(ET.launcher_thresholds([767.0], [1104.0], [None]), "1871")
        self.assertEqual(ET.launcher_thresholds([767.0], [1104.0]), "1871")


class TheMeasurementHasAFixpoint(CustomTestCase):
    def test_growth_does_not_depend_on_the_threshold(self):
        windows = [(r, t, s, p) for _ts, r, t, s, p, _cf in RC12E]
        first = ET.extend_growth_mib(windows)
        # a trim before the extend empties the cache: the window then starts
        # lower and re-creates at most the extend's own segment demand -- the
        # same extend measured after a trim cannot exceed its demand
        after_trim = [(r, t, s - 900, s - 900 + (p - s)) for r, t, s, p in windows]
        self.assertEqual(ET.extend_growth_mib(after_trim), first)
        thr = [ET.launcher_thresholds([767.0], [1104.0], [float(first)])]
        for _ in range(3):
            thr.append(ET.launcher_thresholds([767.0], [1104.0],
                                              [float(ET.extend_growth_mib(windows))]))
        self.assertEqual(len(set(thr)), 1)


if __name__ == "__main__":
    unittest.main()
