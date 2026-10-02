"""Nutzer 02.10. ~19:50Z: "laut dashboard haengt alles im flip die ganze zeit?" -- NF y8c
(nfint4h6feldauer-boot-20261002T192927Z-0cf3, 19:29-19:47Z, fixture fixtures/y8c_0cf3) replayed through the
server path.  The boot's first D>P was begun at 19:33:06 (flip_begin epoch_before 0) and begun anew at 19:33:20
with the same epoch_before; only the second got a flip_done.  The phase bar drew the first begin as an open flip
up to "now", so every later P phase showed as FLIP D->P (88 s, 230 s, 302 s ...).  Real flips here are 2-5 s.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))

from boot_replay import replay_boot  # noqa: E402
from rigdash import activity  # noqa: E402

FLIP_KINDS = ("flip_dp", "flip_pd")


class Y8cReplay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = replay_boot("y8c_0cf3")
        cls.done = [float((e.get("data") or {}).get("t") or e.get("ts"))
                    for e in cls.r["ipc"].get("ipc_events") or [] if e.get("type") == "flip_done"]

    def test_fixture_is_the_boot(self):
        self.assertEqual(self.r["ipc"].get("boot_id"), "nfint4h6feldauer-boot-20261002T192927Z-0cf3")
        self.assertGreaterEqual(len(self.done), 5)

    def test_no_flip_segment_longer_than_20s(self):
        long = [(x["k"], round(x["e"] - x["s"], 1)) for x in self.r["segs"]
                if x["k"] in FLIP_KINDS and x["e"] - x["s"] > 20]
        self.assertEqual(long, [])

    def test_every_dp_segment_ends_at_a_flip_done(self):
        for x in self.r["segs"]:
            if x["k"] == "flip_dp":
                gap = min(abs(x["e"] - t) for t in self.done)
                self.assertLessEqual(gap, 5.0, "flip_dp %.1f-%.1f ends %.1f s from any flip_done"
                                     % (x["s"], x["e"], gap))

    def test_prefill_phases_are_p(self):
        p = [x for x in self.r["segs"] if x["k"] == "P"]
        self.assertGreater(len(p), 0)
        self.assertGreater(sum(x["e"] - x["s"] for x in p), 30.0)

    def test_abandoned_begin_is_not_open(self):
        still_open = [x for x in self.r["model"].open_flips if x["open_end"] is None]
        self.assertEqual(still_open, [])


class OpenFlips(unittest.TestCase):
    def test_abandoned_begin_ends_at_next_begin_newest_stays_open(self):
        begins = [{"flip_begin_ts": 100.0}, {"flip_begin_ts": 114.0}, {"flip_begin_ts": 200.0}]
        out = activity.open_flips(begins, [(114.0, 117.0)])
        self.assertEqual([(x["flip_begin_ts"], x["open_end"]) for x in out], [(100.0, 114.0), (200.0, None)])

    def test_begin_with_done_is_not_open(self):
        self.assertEqual(activity.open_flips([{"flip_begin_ts": 50.0}], [(50.2, 53.0)]), [])


if __name__ == "__main__":
    unittest.main()
