"""PRIORITY LANES 1008, part L1: the dashboard shows the lanes in ONE display row (agreed with the dashboard seat:
display only, no layout change). The row sits in the existing front block (tile "Flips / queue"), reads
``front.lane_floor`` / ``front.lanes`` of state.json (or of /weg2/state) and is absent when the front reports no
``lane_floor`` (SGLANG_WEG2_LANES off). The row's behaviour was also exercised in a browser (Playwright, static page,
``laneRow`` called with a state with / without ``lane_floor``)."""

import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rigdash import server  # noqa: E402

PAGE = os.path.join(server.STATIC, "index.html")


class TestLaneRow(unittest.TestCase):
    def setUp(self):
        with open(PAGE, encoding="utf-8") as fh:
            self.html = fh.read()

    def test_one_function_one_row_in_the_front_block(self):
        self.assertEqual(self.html.count("function laneRow(b)"), 1)
        # exactly one call, as one more entry of the existing kv rows of the "Flips" tile
        self.assertEqual(len(re.findall(r"^\s+laneRow\(b\),$", self.html, re.M)), 1)
        self.assertEqual(self.html.count("laneRow("), 2)    # the definition and that one call

    def test_the_row_is_absent_without_lane_floor(self):
        m = re.search(r"function laneRow\(b\) \{(.*?)\n\}\n", self.html, re.S)
        self.assertIsNotNone(m)
        body = m.group(1)
        self.assertIn("lane_floor != null", body)
        self.assertIn("if (!f) return null;", body)
        self.assertIn('["Lanes", `floor ${fmt(f.lane_floor)}, lanes ${esc(JSON.stringify(f.lanes || {}))}', body)

    def test_release_edition_keeps_the_row(self):
        """The row is part of the front block, not of the development part."""
        rel = server.edition_page(self.html, "release")
        self.assertIn("function laneRow(b)", rel)


if __name__ == "__main__":
    unittest.main()
