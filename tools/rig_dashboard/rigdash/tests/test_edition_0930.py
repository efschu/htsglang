"""Zwei Ausgaben, ein Code (Nutzer 30.09.: "im veröffentlichten fLLiper dashboard soll natürlich der
untere entwicklungsstanddashboard leer sein oder fehlen"): --edition release cuts the development
part out of the page and out of /api/live; rig serves everything."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rigdash import server  # noqa: E402

PAGE = os.path.join(server.STATIC, "index.html")
DEV_IDS = ("dev-head", "launch-card", "prod-card", "feat-card", "imgchg-card", "containers", "gpuq", "history",
           "launchbox", "gpuqlink", "weg2link", "plannerlink")
TOP_IDS = ("boots", "verlauf", "vl-c-pre", "vl-c-dec", "vl-c-cache", "vl-c-kv", "gpus-card", "hw-c-power")


class TestEdition(unittest.TestCase):
    def setUp(self):
        with open(PAGE, encoding="utf-8") as fh:
            self.html = fh.read()

    def test_release_page_has_no_development_part(self):
        rel = server.edition_page(self.html, "release")
        for i in DEV_IDS:
            self.assertNotIn('id="%s"' % i, rel, i)
        for w in ("Features Soll/Ist", "Bausteine", "Image-&Auml;nderungen", "Entwicklungsstand", "Startflags",
                  "GPU-Fensterplan", "DEV:BEGIN"):
            self.assertNotIn(w, rel, w)
        for i in TOP_IDS:
            self.assertIn('id="%s"' % i, rel, i)
        self.assertIn('data-edition="release"', rel)
        self.assertIn("<title>fLLiper Dashboard</title>", rel)

    def test_rig_page_unchanged(self):
        self.assertEqual(server.edition_page(self.html, "rig"), self.html)
        for i in DEV_IDS + TOP_IDS:
            self.assertIn('id="%s"' % i, self.html, i)

    def test_dev_blocks_balanced(self):
        self.assertEqual(self.html.count(server.DEV_BEGIN), self.html.count(server.DEV_END))
        with self.assertRaises(ValueError):
            server.edition_page("a" + server.DEV_BEGIN + "b", "release")

    def test_release_api_drops_dev_keys(self):
        snap = {"boots": [], "gpus": {}, "features": {"x": 1}, "image_changes": {}, "gpuq": {}}
        out = server.edition_snapshot(dict(snap), "release")
        for k in server.RELEASE_DROP_KEYS:
            self.assertNotIn(k, out)
        self.assertIn("boots", out)
        self.assertEqual(server.edition_snapshot(dict(snap), "rig"), snap)

    def test_one_logo(self):
        # the word mark and the square mark were BOTH shown: 'header .brand img {display:block}' (0,1,2)
        # beat 'header .logo-mark {display:none}' (0,1,1).  The hide rule must be at least as specific.
        self.assertIn("header .brand img.logo-mark { display: none; }", self.html)
        self.assertNotIn("header .logo-mark { display: none; }", self.html)


if __name__ == "__main__":
    unittest.main()
