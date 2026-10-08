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

    def test_release_boot_card_hides_build_identity(self):
        # NF-Operator 30.09.: Startform je Gruppe (argv + Schalter-Env), Image-SHAs, Zweig-, Profilnamen
        # und Env-Schalter gehören in den Entwicklungsteil: im Release weder auf der Seite noch in /api/live
        self.assertIn('${!DEV ? "" : ipc && Object.keys(ipc.launch || {}).length', self.html)
        self.assertIn("${!DEV ? relHead(b, ipc, topo) :", self.html)
        self.assertIn('${DEV ? esc(c.Names) : "Container"}', self.html)
        head = self.html[self.html.index("function relHead("):self.html.index("function bootCard(")]
        for w in ("rev", "profile", "image", "tag", "boot_id", "launch", "env", "Names"):
            self.assertNotIn("ipc." + w, head, w)
        mk = lambda: {"stem": "x", "meta": {"tag": "nfh91-profil", "sha": "abc", "launch": ["--x"], "model": "M"},
                "container": {"Names": "htsglang-acc-nf-h91", "State": "running", "Status": "Up"},
                "ipc": {"launch": {"P": {"argv": ["--tp-size", "1"], "env": {"FLLIPER_X": "1"}}}, "rev": "62357f2ba1",
                        "profile": "nf-h91", "image": "htsglang:cu130-weg2", "tag": "nfh91", "boot_id": "nfh91-boot",
                        "dir": "/spinning/docker-acceptance/nf/state/nfh91-boot", "lifecycle": "serving"}}
        out = server.edition_snapshot({"boots": [mk()], "docker": {"age_s": 1.0, "value": [
            {"Names": "htsglang-acc-nf-h91", "Image": "htsglang:cu130-weg2", "State": "running"}]}}, "release")
        self.assertEqual(out["docker"], {"age_s": 1.0})
        blob = repr(out)
        for w in ("62357f2ba1", "nf-h91", "nfh91", "cu130", "FLLIPER_X", "--tp-size", "htsglang-acc", "abc"):
            self.assertNotIn(w, blob, w)
        self.assertEqual(out["boots"][0]["ipc"]["lifecycle"], "serving")
        self.assertEqual(out["boots"][0]["container"], {"State": "running", "Status": "Up"})
        rig = server.edition_snapshot({"boots": [mk()]}, "rig")
        self.assertEqual(rig["boots"][0]["ipc"]["rev"], "62357f2ba1")

    def test_one_logo(self):
        # the word mark and the square mark were BOTH shown: 'header .brand img {display:block}' (0,1,2)
        # beat 'header .logo-mark {display:none}' (0,1,1).  The hide rule must be at least as specific.
        self.assertIn("header .brand img.logo-mark { display: none; }", self.html)
        self.assertNotIn("header .logo-mark { display: none; }", self.html)


if __name__ == "__main__":
    unittest.main()
