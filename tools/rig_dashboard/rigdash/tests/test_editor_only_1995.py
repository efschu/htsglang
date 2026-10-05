"""Auftrag 1995: rigdash als zweiter Dienst im Docker-Image -- Editor-only-Modus und Proxy-Schalter.

Gepinnt (1984 A.6 Punkt 1 und 4):

* ``--editor-only`` (nur mit ``--edition release``): die Seite traegt ``data-editor-only="1"`` und blendet alles ausser dem Reiter Profil aus;
  der Server antwortet NUR Seite, Module, /healthz und die drei Editor-Routengruppen (profil, modellprofil, hwprofil) -- /api/live, /api/health,
  /api/history, /api/launch, /api/kartenplan, /api/weg2 und alles Unbekannte sind 404. Kein Probennehmer: ``App.start`` startet keinen Thread.
* ``--trust-proxy`` (nur mit ``--edition release``): hinter einem eigenen Reverse-Proxy (X-Forwarded-*) antwortet der Editor statt 403; ohne
  den Schalter bleibt der Riegel (Voreinstellung unveraendert).
* ``main()`` verweigert beide Schalter in der Rig-Ausgabe.
* /healthz gibt es in beiden Modi (Docker-Gesundheit des Dienstes).
"""

import http.client
import json
import os
import shutil
import sys
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from rigdash import server as S  # noqa: E402
from rigdash.tests import test_hwprofil_950 as H  # noqa: E402
from rigdash.tests import test_profil_930 as P9  # noqa: E402
from rigdash.tests import test_profil_balken_1432 as B  # noqa: E402

class Est(B.Est):
    def models(self):
        return {"ok": True, "roots": [], "models": []}


XFF = {"X-Forwarded-For": "203.0.113.7"}


def page():
    with open(os.path.join(S.STATIC, "index.html"), encoding="utf-8") as fh:
        return fh.read()


class TestPage(unittest.TestCase):
    def test_editor_only_page_is_marked_and_the_release_page_is_not(self):
        eo = S.edition_page(page(), "release", editor_only=True)
        self.assertIn('data-edition="release" data-editor-only="1"', eo)
        self.assertNotIn('data-edition="release" data-editor-only', S.edition_page(page(), "release"))
        self.assertNotIn('data-edition=', S.edition_page(page(), "rig", editor_only=True))   # die Rig-Seite bleibt byte-gleich
        self.assertEqual(S.edition_page(page(), "rig", editor_only=True), page())

    def test_editor_only_page_still_carries_the_profil_tab_and_no_dev_part(self):
        eo = S.edition_page(page(), "release", editor_only=True)
        for needle in ('id="tab-profil"', 'id="pf-root"', "profil.js", "hwprofil.js"):
            self.assertIn(needle, eo, needle)
        for needle in ('id="tab-entwicklung"', "kartenplan.js", "GPU-Fensterplan"):
            self.assertNotIn(needle, eo, needle)

    def test_the_page_script_stops_polling_and_forces_the_profil_tab_when_editor_only(self):
        html = page()
        self.assertIn('document.documentElement.dataset.editorOnly === "1"', html)
        self.assertIn('if (EDITOR_ONLY) curTab = "profil";', html)
        self.assertIn("if (EDITOR_ONLY) return;", html)          # refresh() und hashchange
        self.assertIn('html[data-editor-only="1"] #tabbar', html)
        self.assertIn('html[data-editor-only="1"] .tab:not(#tab-profil)', html)


class Served(unittest.TestCase):
    editor_only = True
    trust_proxy = False
    edition = "release"

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="eo1995_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.ed, self.rel, self.usr = P9.editor(self.tmp)
        self.gq = H.FakeGpuq()
        hw_tree = os.path.join(self.tmp, "hwtree")
        d = os.path.join(hw_tree, "sglang", "srt", "rigmon")
        os.makedirs(d)
        with open(os.path.join(d, "hardware_profile.py"), "w") as fh:
            fh.write(H.STUB_MODULE)
        self.hw = H.hwprofil.HwProfil(http=self.gq, tree=hw_tree, synchronous=True, edition=self.edition)
        self.svc = B.fake_service(self.tmp)
        self.addCleanup(self.svc.close)
        app = SimpleNamespace(edition=self.edition, profil=self.ed, hwprofil=self.hw, modellprofil=Est(), couplings=self.svc,
                              version="t", kartenplaner=self.ed.kp, editor_only=self.editor_only, trust_proxy=self.trust_proxy)
        srv = ThreadingHTTPServer(("127.0.0.1", 0), S.make_handler(app))
        srv.daemon_threads = True
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.shutdown)
        self.addCleanup(srv.server_close)
        self.port = srv.server_address[1]

    def call(self, method, path, body=None, headers=None):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=60)
        h = dict(headers or {})
        data = None
        if body is not None:
            data = json.dumps(body)
            h["Content-Type"] = "application/json"
        c.request(method, path, body=data, headers=h)
        r = c.getresponse()
        return r.status, r.read().decode()


class TestEditorOnlyRoutes(Served):
    def test_what_is_answered(self):
        st, txt = self.call("GET", "/")
        self.assertEqual(st, 200)
        self.assertIn('data-editor-only="1"', txt)
        st, txt = self.call("GET", "/healthz")
        self.assertEqual(st, 200, txt)
        j = json.loads(txt)
        self.assertTrue(j["ok"] and j["editor_only"] and j["edition"] == "release")
        for path in ("/profil.js", "/hwprofil.js", "/modellprofil.js", "/profil_balken.js", "/grafik.js", "/logo.svg", "/api/profil/list",
                     "/api/profil/modelle", "/api/hwprofil"):
            self.assertEqual(self.call("GET", path)[0], 200, path)
        st, txt = self.call("POST", "/api/profil/load", {"kind": "release", "name": "demo"})
        self.assertEqual(st, 200, txt)

    def test_what_is_not_answered(self):
        for path in ("/api/live", "/api/live?lean=1&dev=0", "/api/health", "/api/history", "/api/launch", "/api/kartenplan/catalog",
                     "/api/weg2/options", "/weg2", "/kartenplan.js", "/nichts", "/api/profil", "/api/profile/list"):
            self.assertEqual(self.call("GET", path)[0], 404, path)
        for path in ("/api/live", "/api/launch", "/nichts", "/healthz", "/"):
            self.assertEqual(self.call("POST", path, {})[0], 404, path)

    def test_measuring_stays_shut_in_release(self):
        st, txt = self.call("POST", "/api/hwprofil/measure", {"cards": [0]})
        self.assertEqual(st, 403, txt)
        self.assertEqual(self.gq.log, [])

    def test_the_proxy_bar_stays_without_the_switch(self):
        for path in ("/api/profil/list", "/api/hwprofil", "/api/modellprofil/modelle"):
            self.assertEqual(self.call("GET", path, headers=XFF)[0], 403, path)
        self.assertEqual(self.call("POST", "/api/profil/load", {"kind": "release", "name": "demo"}, headers=XFF)[0], 403)


class TestTrustProxy(Served):
    trust_proxy = True

    def test_behind_an_own_reverse_proxy_the_editor_answers(self):
        for path in ("/api/profil/list", "/api/hwprofil", "/api/modellprofil/modelle"):
            self.assertEqual(self.call("GET", path, headers=XFF)[0], 200, path)
        st, txt = self.call("POST", "/api/profil/load", {"kind": "release", "name": "demo"}, headers={"X-Forwarded-Prefix": "/editor"})
        self.assertEqual(st, 200, txt)

    def test_trust_proxy_does_not_open_measuring(self):
        self.assertEqual(self.call("POST", "/api/hwprofil/measure", {"cards": [0]}, headers=XFF)[0], 403)
        self.assertEqual(self.gq.log, [])


class TestNotEditorOnlyKeepsTheFullRelease(Served):
    editor_only = False

    def test_healthz_exists_and_the_old_routes_are_unchanged(self):
        st, txt = self.call("GET", "/healthz")
        self.assertEqual(st, 200)
        self.assertFalse(json.loads(txt)["editor_only"])
        self.assertEqual(self.call("GET", "/api/profil/list")[0], 200)
        self.assertEqual(self.call("GET", "/api/launch")[0], 404)       # Release: weiter zu (nicht wegen editor-only)
        self.assertEqual(self.call("GET", "/api/profil/list", headers=XFF)[0], 403)   # Riegel wie vorher


class TestMain(unittest.TestCase):
    def test_both_switches_are_release_only(self):
        for flag in ("--editor-only", "--trust-proxy"):
            with self.assertRaises(SystemExit) as cm:
                S.main([flag])
            self.assertEqual(cm.exception.code, 2, flag)
            with self.assertRaises(SystemExit) as cm:
                S.main(["--edition", "rig", flag])
            self.assertEqual(cm.exception.code, 2, flag)

    def test_editor_only_app_starts_no_thread(self):
        before = threading.active_count()
        fake = SimpleNamespace(editor_only=True)         # start() darf nichts anfassen (kein boots, kein src ...)
        self.assertIsNone(S.App.start(fake))
        self.assertEqual(threading.active_count(), before)


if __name__ == "__main__":
    unittest.main()
