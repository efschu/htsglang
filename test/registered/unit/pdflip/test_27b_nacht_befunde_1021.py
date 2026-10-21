# SPDX-License-Identifier: Apache-2.0
"""27B NACHT 21.10. -- open 27B metal findings of done/27b-int8-abnahme-1021.md, each as its metal numbers.

1. EXTEND-CAP-FLOOR under SOLL R7 (rc29 ..._041a871f4b_1021_001910, D 00:26:50Z): ``EXTEND-CAP-FLOOR rank=1 post=20``,
   ``#794 GROUP-NARROWED 4096 -> 63``, then ``refuses_over_width`` refused every fresh D extend over 63 tokens -- kurz
   requests 4-69 (u=486), 4-70 (234), 4-71 (639), 4-72 (797) at halfX=5000 -- W31 -> W50-REROUTE x_refusal_midstream to
   a sleeping P, and R7 takes no flip for them: deadlock. Under the central decider the floor now CRAWLS (D takes the
   request at the narrowed width); with the decider off the Q-710 refusal stays as it was.
2. ``rule=ALT:decode_collect`` (rc29 14 lines, rc29b 18): ``_decode_collect`` IS the set scheduler's one decision site;
   its ALT shadow made it read as an old path still running in flip mode. Dropped from the ALT table.
"""
import os
import types
import unittest
from unittest import mock

from flliper.srt.pdflip import extend_trim as ET

CONFIGURED = 4096
FLOOR_ENV = "FLLIPER_PDFLIP_EXTEND_CAP_FLOOR"


class CapFloorUnderSollR7(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from flliper.srt.managers.scheduler import Scheduler

        cls.fn = staticmethod(Scheduler._pdflip_cap_floor_verdict)

    def setUp(self):
        ET.reset_for_tests()
        self._p = mock.patch.dict(os.environ, {FLOOR_ENV: "1"})
        self._p.start()

    def tearDown(self):
        self._p.stop()
        ET.reset_for_tests()

    def _sched(self, width):
        return types.SimpleNamespace(uniform_corridor_width=lambda: width, page_size=1,
                                     chunked_prefill_size=CONFIGURED)

    def test_rc29_metal_kurz_requests_at_width_63_are_taken_not_refused(self):
        s = self._sched(63)
        for rid, u in (("pdflip-4-69", 486), ("pdflip-4-70", 234), ("pdflip-4-71", 639), ("pdflip-4-72", 797)):
            self.assertTrue(ET.refuses_over_width(63, u, CONFIGURED, 1, False))   # the old verdict would refuse
            with self.assertLogs("flliper.srt.managers.scheduler", level="WARNING") as cm:
                self.assertEqual(self.fn(s, types.SimpleNamespace(rid=rid), u, "admit"), "admit")
            self.assertTrue(any("EXTEND-CAP-FLOOR CRAWL rid=%s uncached=%d width=63" % (rid, u) in m
                                for m in cm.output), cm.output)
            self.assertIn(rid, s._pdflip_ecf_admitted)   # its chunk continuations are never priced again

    def test_the_crawl_line_is_written_once_per_rid(self):
        s = self._sched(63)
        req = types.SimpleNamespace(rid="pdflip-4-71")
        self.fn(s, req, 639, "admit")
        with mock.patch("flliper.srt.managers.scheduler.logger") as lg:
            self.assertEqual(self.fn(s, req, 600, "admit"), "admit")
        lg.warning.assert_not_called()

    def test_decider_off_keeps_the_q710_refusal(self):
        s = self._sched(63)
        with mock.patch.dict(os.environ, {"FLLIPER_PDFLIP_DECIDER": "0"}):
            self.assertEqual(self.fn(s, types.SimpleNamespace(rid="pdflip-4-69"), 486, "admit"), "W31")

    def test_crawl_mode_follows_the_decider_switch(self):
        with mock.patch.dict(os.environ, {"FLLIPER_PDFLIP_DECIDER": "1"}):
            self.assertTrue(ET.cap_floor_crawls_under_soll_r7())
        with mock.patch.dict(os.environ, {"FLLIPER_PDFLIP_DECIDER": "0"}):
            self.assertFalse(ET.cap_floor_crawls_under_soll_r7())


class AltShadowNoLongerShadowsTheSetSite(unittest.TestCase):
    def test_decode_collect_is_not_an_alt_site(self):
        from flliper.srt.pdflip import front as F

        self.assertNotIn("decode_collect", F.EZ_ALT_SITES)
        self.assertIsNone(getattr(F.Front._decode_collect, "_ez_alt_marker", None))
        self.assertEqual(F.EZ_ALT_MISSING, [])
        # the old sites stay shadowed
        self.assertEqual(getattr(F.Front._set_xsum_hold, "_ez_alt_marker", None), "set_xsum_hold")


if __name__ == "__main__":
    unittest.main()
