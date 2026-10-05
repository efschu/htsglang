"""Auftrag 1984 (A): der Reiter Profil in der Release-Edition.

Nutzer-Entscheid 05.10.: der Profil-Editor kommt INS Release.  Gepinnt:

* Seite: die Release-Seite traegt den Reiter Profil, sein Panel, die vier Module (profil, profil_balken, hwprofil, modellprofil) und das
  CSS; weiter FEHLEN Kartenplaner, Entwicklungsstand und alles andere hinter DEV-Markern (Startzeile, gpuq-Link, Features ...).
* Routen: Editor (list/modelle/load/edit/save/delete/export/dry), Balken (recompute), Hardwareprofil ANZEIGEN (GET), Modellprofil
  schaetzen sind in Release erreichbar -- im LAN, ueber den oeffentlichen Proxy weiter 403.
* Rig-Eingriffe bleiben in Release zu: Hardware MESSEN und Fenster zurueckgeben (bucht gpuq) -> 403 mit Klartext, keine Buchung;
  Kartenplaner, Startzeile, /api/launch, /api/weg2/* -> 404.
* hwprofil.js: in Release ist der Messknopf aus und sagt "braucht gpuq".
"""

import http.client
import json
import os
import shutil
import subprocess
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

STATIC = S.STATIC
NODE = B.NODE
PROFIL_MODULES = ("modellprofil.js", "hwprofil.js", "profil_balken.js", "profil.js")


def page():
    with open(os.path.join(STATIC, "index.html"), encoding="utf-8") as fh:
        return fh.read()


class TestPage(unittest.TestCase):
    def test_release_page_carries_the_profil_tab(self):
        rel = S.edition_page(page(), "release")
        self.assertIn('data-tab="profil"', rel)
        self.assertIn('id="tab-profil"', rel)
        self.assertIn('id="pf-root"', rel)
        for js in PROFIL_MODULES:
            self.assertIn('<script src="%s"></script>' % js, rel, js)
        # das CSS der Karte (Profil) und der gemeinsamen kp-Klassen (Urteil, Balken) ist da
        for css in ("#pf-root {", ".pf-top {", ".kp-verdict {", ".ks-over {"):
            self.assertIn(css, rel, css)
        self.assertNotIn("DEV:BEGIN", rel)

    def test_release_page_still_has_no_planer_and_no_development_part(self):
        rel = S.edition_page(page(), "release")
        for needle in ('data-tab="kartenplan"', 'data-tab="entwicklung"', 'id="kp-root"', "kartenplan.js", 'id="tab-entwicklung"',
                       "GPU-Fensterplan", "Startzeile", 'id="dev-head"', 'id="gpuqlink"'):
            self.assertNotIn(needle, rel, needle)

    def test_the_profil_tab_sits_before_nothing_dev_in_release_and_before_entwicklung_in_rig(self):
        html = page()
        rig = S.edition_page(html, "rig")
        self.assertLess(rig.index('data-tab="kartenplan"'), rig.index('data-tab="profil"'))
        self.assertLess(rig.index('data-tab="profil"'), rig.index('data-tab="entwicklung"'))
        self.assertIn("kartenplan.js", rig)

    def test_rig_page_is_byte_identical_to_the_source(self):
        self.assertEqual(S.edition_page(page(), "rig"), page())


class Served(unittest.TestCase):
    """Eine Release-Instanz mit allem, was der Editor braucht (Fake-Gpuq, Fake-Worker, Fixture-Baum)."""

    edition = "release"

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="pf1984_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.ed, self.rel, self.usr = P9.editor(self.tmp)
        self.gq = H.FakeGpuq()
        self.hw_tree = os.path.join(self.tmp, "hwtree")
        d = os.path.join(self.hw_tree, "sglang", "srt", "rigmon")
        os.makedirs(d)
        with open(os.path.join(d, "hardware_profile.py"), "w") as fh:
            fh.write(H.STUB_MODULE)
        self.hw = H.hwprofil.HwProfil(http=self.gq, tree=self.hw_tree, synchronous=True, edition=self.edition)
        self.est = B.Est()
        self.svc = B.fake_service(self.tmp)
        self.addCleanup(self.svc.close)
        app = SimpleNamespace(edition=self.edition, profil=self.ed, hwprofil=self.hw, modellprofil=self.est, couplings=self.svc,
                              version="t", kartenplaner=self.ed.kp)
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
        txt = r.read().decode()
        return r.status, txt


class TestReleaseRoutesOpen(Served):
    def test_the_editor_flow_works_in_release(self):
        st, txt = self.call("GET", "/api/profil/list")
        self.assertEqual(st, 200, txt)
        self.assertEqual([p["name"] for p in json.loads(txt)["release"]], ["demo"])
        self.assertEqual(self.call("GET", "/api/profil/modelle")[0], 200)
        st, txt = self.call("POST", "/api/profil/load", {"kind": "release", "name": "demo"})
        self.assertEqual(st, 200, txt)
        doc = json.loads(txt)["doc"]
        st, txt = self.call("POST", "/api/profil/edit", {"doc": doc, "edits": [{"key": "flag:--p-bs", "op": "set", "value": "3"}]})
        self.assertEqual(st, 200, txt)
        doc = json.loads(txt)["doc"]
        st, txt = self.call("POST", "/api/profil/export", {"doc": doc})
        self.assertEqual(st, 200)
        self.assertTrue(json.loads(txt)["verified"])
        st, txt = self.call("POST", "/api/profil/dry", {"doc": doc, "cards": P9.RIG[:2]})
        self.assertEqual(st, 200)
        self.assertIn("HW-COUNT", txt)
        st, txt = self.call("POST", "/api/profil/save", {"doc": doc, "name": "meins"})
        self.assertEqual(st, 200, txt)
        self.assertTrue(os.path.isfile(os.path.join(self.usr, "meins.json")))
        self.assertEqual(self.call("POST", "/api/profil/delete", {"name": "meins"})[0], 200)
        self.assertFalse(os.path.exists(os.path.join(self.usr, "meins.json")))

    def test_no_start_route_and_no_force_switch_in_release(self):
        self.assertEqual(self.call("POST", "/api/profil/start", {})[0], 404)
        self.assertEqual(self.call("POST", "/api/profil/force", {})[0], 404)

    def test_the_bars_recompute_in_release(self):
        doc = {"vars": [{"name": "PROFILE_MODEL", "value": "/m/nf"}], "args": [{"flag": "--pp-stage-ratio", "values": ["29,11,8"]}]}
        st, txt = self.call("POST", "/api/profil/recompute", {"doc": doc, "what": "bars"})
        self.assertEqual(st, 200, txt)
        self.assertTrue(json.loads(txt)["ok"])

    def test_the_hardware_profile_is_shown_without_asking_gpuq(self):
        st, txt = self.call("GET", "/api/hwprofil")
        self.assertEqual(st, 200, txt)
        j = json.loads(txt)
        self.assertEqual(j["profile"]["schema"], "flliper.hardware/1")
        self.assertFalse(j["gpuq"]["reachable"])
        self.assertIn("gpuq", j["gpuq"]["error"])
        self.assertEqual(self.gq.log, [], "die Release-Anzeige darf gpuq nicht ansprechen")

    def test_the_model_estimate_works_in_release(self):
        # ModelEstimator selbst ist ueber test_modellprofil_960 gepinnt; hier nur: die Routen sind in Release da (kein 404)
        app = SimpleNamespace(edition="release", version="t", modellprofil=self.est)
        srv = ThreadingHTTPServer(("127.0.0.1", 0), S.make_handler(app))
        srv.daemon_threads = True
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.shutdown)
        self.addCleanup(srv.server_close)
        c = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=20)
        c.request("POST", "/api/modellprofil/schaetzen", body=json.dumps({"path": "/nicht/hier"}), headers={"Content-Type": "application/json"})
        r = c.getresponse()
        r.read()
        self.assertEqual(r.status, 400)           # Est lehnt den Pfad ab -> 400 heisst: die Route lief
        for p in ("/modellprofil.js", "/hwprofil.js", "/profil.js", "/profil_balken.js"):
            c = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=20)
            c.request("GET", p)
            r = c.getresponse()
            r.read()
            self.assertEqual(r.status, 200, p)

    def test_the_public_proxy_stays_closed_in_release(self):
        px = {"X-Forwarded-For": "1.2.3.4"}
        self.assertEqual(self.call("GET", "/api/profil/list", headers=px)[0], 403)
        self.assertEqual(self.call("POST", "/api/profil/save", {"doc": {}, "name": "x"}, headers=px)[0], 403)
        self.assertEqual(self.call("POST", "/api/profil/recompute", {"doc": {}}, headers=px)[0], 403)
        self.assertEqual(self.call("GET", "/api/hwprofil", headers=px)[0], 403)
        self.assertEqual(self.call("GET", "/api/modellprofil/modelle", headers={"X-Forwarded-Prefix": "/r"})[0], 403)


class TestReleaseRigActionsStayClosed(Served):
    def test_measuring_and_cancelling_are_refused_and_never_book(self):
        for path, body in (("/api/hwprofil/measure", {"cards": [1]}), ("/api/hwprofil/cancel", {})):
            st, txt = self.call("POST", path, body)
            self.assertEqual(st, 403, path)
            j = json.loads(txt)
            self.assertFalse(j["ok"])
            self.assertIn("gpuq", j["error"])
            self.assertIn("Release", j["error"])
        self.assertEqual(self.gq.log, [])

    def test_the_class_itself_refuses_in_release(self):
        with self.assertRaises(ValueError) as cm:
            self.hw.measure({"cards": [1]})
        self.assertIn("gpuq", str(cm.exception))
        self.assertEqual(self.hw.cancel()["ok"], False)
        self.assertEqual(self.gq.log, [])

    def test_planer_launch_wizard_and_dev_state_stay_404(self):
        for method, path in (("GET", "/api/kartenplan/catalog"), ("GET", "/kartenplan.js"), ("GET", "/weg2"), ("GET", "/api/launch"),
                             ("GET", "/api/weg2/options"), ("GET", "/api/weg2/dry")):
            self.assertEqual(self.call(method, path)[0], 404, path)


class TestRigEditionUnchanged(Served):
    edition = "rig"

    def test_measuring_still_books_in_rig(self):
        st, txt = self.call("POST", "/api/hwprofil/measure", {"cards": [1]})
        self.assertEqual(st, 200, txt)
        self.assertEqual(json.loads(txt)["action"], "messung_gestartet")
        self.assertTrue(self.gq.calls("POST", "/api/v1/bookings"))

    def test_the_rig_still_asks_gpuq_for_the_card_list(self):
        st, txt = self.call("GET", "/api/hwprofil")
        self.assertEqual(st, 200)
        self.assertTrue(json.loads(txt)["gpuq"]["reachable"])

    def test_kartenplaner_is_still_served_in_rig(self):
        self.assertEqual(self.call("GET", "/kartenplan.js")[0], 200)


@unittest.skipUnless(NODE, "node fehlt")
class TestHwprofilJsInRelease(unittest.TestCase):
    SCRIPT = r"""
const H = require(process.argv[1]);
const els = {};
function fakeEl() { return { innerHTML: "", addEventListener() {}, getAttribute() { return null; } }; }
globalThis.fetch = async () => ({ status: 200, text: async () => process.argv[3] });
const el = fakeEl();
const m = H.mount(el, JSON.parse(process.argv[2]));
m.refresh().then(() => { process.stdout.write(el.innerHTML); });
"""

    def run_mount(self, opts):
        out = subprocess.run([NODE, "-e", self.SCRIPT, os.path.join(STATIC, "hwprofil.js"), json.dumps(opts),
                              json.dumps(dict(H.TestDisplayModule.DOC, window=None))],
                             capture_output=True, text=True, timeout=30)
        self.assertEqual(out.returncode, 0, out.stderr)
        return out.stdout

    def test_release_has_no_measure_button_but_says_why(self):
        h = self.run_mount({"edition": "release"})
        self.assertNotIn('data-act="measure"', h)
        self.assertIn("braucht gpuq", h)
        self.assertIn('data-act="refresh"', h)
        self.assertIn("Hardwareprofil", h)

    def test_rig_keeps_the_measure_button(self):
        h = self.run_mount({"edition": "rig"})
        self.assertIn('data-act="measure"', h)
        self.assertNotIn("braucht gpuq", h)

    def test_profil_js_hands_the_page_edition_to_the_module(self):
        with open(os.path.join(STATIC, "profil.js"), encoding="utf-8") as fh:
            js = fh.read()
        self.assertIn("data-edition", js)


if __name__ == "__main__":
    unittest.main()
