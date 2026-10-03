"""Auftrag 930 (Profil-Editor S1): laden, bearbeiten, Herkunft, Export, Trockenlauf, Speichern, Routen.

Gepinnt:
  * Ein Release-Profil (synthetische .env, echtes bash) wird zu ``flliper.server/1``; Zeilen tragen Erklaerung, Herkunft, Abhaengigkeiten.
  * Bearbeiten setzt die Herkunft ``nutzer``; Zuruecksetzen auf Profil/Planer stellt sie wieder her.
  * Export ist ein .env im heutigen Dialekt und ``geprueft`` (bash wertet es aus, gleich dem Profil); ein bearbeitetes Profil auch.
  * Trockenlauf: die ORIGINAL-Planer-Funktionen (Fixture-Baum) nennen HW-COUNT/HW-UNCALIBRATED/HW-TOPOLOGY/HW-ARCH mit Code; jede Ablehnung
    traegt Klasse, Begruendung und die Aussage, was Force beim Serverstart tut; HW-ARCH ist nicht forcebar. Das Dashboard hat KEINEN
    Force-Schalter und keine Start-Route.
  * Nutzerprofile liegen als JSON im Profilverzeichnis; ungueltige Namen werden abgewiesen (kein Pfadausbruch).
  * Routen: im LAN erreichbar, ueber den oeffentlichen Proxy 403, in der Release-Ausgabe 404.
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

from rigdash import kartenplan as K  # noqa: E402
from rigdash import profil as P  # noqa: E402
from rigdash import server as S  # noqa: E402

FIXTURE_TREE = os.path.join(HERE, "fixtures", "kartenplan", "planner_tree", "python")
REPO_PY = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "python"))
REPO_CATALOG = os.path.join(os.path.dirname(HERE), "profil_data", "catalog.json")

ENV = """\
# shellcheck shell=bash
# why the batch size: measured
PROFILE_NAME=demo
PROFILE_LINE=nf
PROFILE_FORMAT=int4-mixed
PROFILE_STATUS=experimentell
PROFILE_OWNER="the owner"
PROFILE_CARD_COUNT=3
PROFILE_INVENTORY=RTX5090,RTX3080,RTX3080
PROFILE_ARGS=(--model /m --p-bs 2 --pp-stage-ratio 29,11,8 --pp-attn-stage-ratio 8,4,4 --p-hostgap
              "--extra-p=--rank-moe-ratio 183,137,168" --env-p "SGLANG_MOE_SCRATCH_SLOTS=74,48,48")
profile_form_env() {
  _form SGLANG_WEG2_OWNED_BASE stated
}
"""


def editor(tmp):
    rel = os.path.join(tmp, "rel")
    usr = os.path.join(tmp, "usr")
    os.makedirs(rel)
    with open(os.path.join(rel, "demo.env"), "w") as fh:
        fh.write(ENV)
    kp = K.Kartenplaner(tree=FIXTURE_TREE)
    return P.ProfilEditor(kartenplaner=kp, release_dir=rel, user_dir=usr, tree=FIXTURE_TREE, catalog_file=REPO_CATALOG), rel, usr


RIG = [{"card": "rtx3080-20", "pcie": {"gen": 4, "lanes": 4}}, {"card": "rtx5090-32", "pcie": {"gen": 5, "lanes": 8}},
       {"card": "rtx3080-20", "pcie": {"gen": 4, "lanes": 8}}]


class Editor(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="pf930_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.ed, self.rel, self.usr = editor(self.tmp)

    def rows(self, r):
        return {x["key"]: x for x in r["view"]["rows"]}

    def test_list_and_load_a_release_profile(self):
        L = self.ed.list()
        self.assertEqual([p["name"] for p in L["release"]], ["demo"])
        self.assertTrue(any(r["code"] == "HW-ARCH" and not r["forcebar"] for r in L["register"]))
        r = self.ed.load("release", "demo")
        rows = self.rows(r)
        self.assertEqual(rows["flag:--p-bs"]["value"], "2")
        self.assertEqual(rows["extra:P:--rank-moe-ratio"]["value"], "183,137,168")
        self.assertEqual(rows["env:P:SGLANG_MOE_SCRATCH_SLOTS"]["value"], "74,48,48")
        for k in rows:
            self.assertEqual(rows[k]["origin"], "profil", k)
        ex = rows["flag:--pp-stage-ratio"]["explain"]
        self.assertEqual(ex["status"], "kuratiert")
        self.assertTrue(any(d["to"] == "--rank-gpu-memory-mib" and d["rel"] == "tauscht" for d in ex["depends"]))
        self.assertTrue(any(d["to"] == "--pp-attn-stage-ratio" and d["present"] for d in ex["depends"]))   # set in this profile
        self.assertEqual(rows["flag:--model"]["explain"]["status"] in ("geerntet", "kuratiert", "profil-kommentar", "unerklaert"), True)
        self.assertEqual(r["view"]["coverage"]["geaendert"], 0)

    def test_the_profile_comment_is_shown_as_its_own_source(self):
        r = self.ed.load("release", "demo")
        parts = self.rows(r)["var:PROFILE_NAME"]["explain"]["parts"]
        self.assertTrue(any(p["kind"] == "profil" and "why the batch size" in p["text"] for p in parts))      # the comment sits right above PROFILE_NAME
        self.assertTrue(any(p["kind"] == "kuratiert" for p in parts))

    def test_edit_origin_and_reset(self):
        r = self.ed.load("release", "demo")
        e = self.ed.edit(r["doc"], [{"key": "flag:--p-bs", "op": "set", "value": "4"}])
        row = self.rows(e)["flag:--p-bs"]
        self.assertEqual((row["value"], row["origin"], row["profile_value"], row["changed"]), ("4", "nutzer", "2", True))
        self.assertEqual(e["view"]["coverage"]["geaendert"], 1)
        back = self.ed.edit(e["doc"], [{"key": "flag:--p-bs", "op": "reset", "to": "profil"}])
        row = self.rows(back)["flag:--p-bs"]
        self.assertEqual((row["value"], row["origin"], row["changed"]), ("2", "profil", False))

    def test_export_is_a_checked_env(self):
        r = self.ed.load("release", "demo")
        x = self.ed.export_env(r["doc"])
        self.assertTrue(x["verified"], x["problems"])
        self.assertIn("PROFILE_ARGS=(", x["env"])
        self.assertIn("profile_form_env()", x["env"])
        self.assertEqual(x["filename"], "demo.env")
        self.assertIn("FLLIPER_FORCE=1", x["use"]["force_env"])
        e = self.ed.edit(r["doc"], [{"key": "env:P:SGLANG_MOE_SCRATCH_SLOTS", "op": "set", "value": "80,50,50"},
                                    {"key": "extra:D:--rank-gpu-memory-mib", "op": "set", "value": "26000,17000,17000"}])
        x2 = self.ed.export_env(e["doc"])
        self.assertTrue(x2["verified"], x2["problems"])
        self.assertIn("SGLANG_MOE_SCRATCH_SLOTS=80,50,50", x2["env"])
        self.assertIn("--rank-gpu-memory-mib 26000,17000,17000", x2["env"])

    def test_dry_run_reference_rig_needs_no_force_except_status(self):
        r = self.ed.load("release", "demo")
        d = self.ed.dry_run(r["doc"], RIG)
        self.assertEqual([q["code"] for q in d["rejections"]], ["PROFIL-STATUS"])           # the synthetic profile is experimentell
        self.assertTrue(d["rejections"][0]["forcebar"])
        self.assertIn("Nicht übergangen werden", d["force_note"])

    def test_dry_run_names_codes_classes_and_what_force_does(self):
        r = self.ed.load("release", "demo")
        two = self.ed.dry_run(r["doc"], RIG[:2])
        by = {}
        for q in two["rejections"]:
            by.setdefault(q["code"], []).append(q)
        self.assertTrue({"HW-COUNT", "HW-UNCALIBRATED"} <= set(by))
        for c in ("HW-COUNT", "HW-UNCALIBRATED"):
            for q in by[c]:
                self.assertEqual(q["klass"], "wert")
                self.assertEqual(q["force_state"], "force")
                self.assertIn("Force übergeht", q["force"])
                self.assertTrue(q["why_class"])
        self.assertTrue(any(q["text"].startswith("HW-COUNT:") for q in by["HW-COUNT"]))      # the planner's own text
        self.assertNotIn("HW-TOPOLOGY", by)                         # an unproven N inside the range is HW-COUNT, not "no topology"
        four = self.ed.dry_run(r["doc"], [{"card": "rtx4090-24", "pcie": {"gen": 4, "lanes": 16}}] + RIG)
        arch = {q["code"]: q for q in four["rejections"]}["HW-ARCH"]
        self.assertFalse(arch["forcebar"])
        self.assertEqual(arch["klass"], "nicht_forcebar")
        self.assertIn("nein", arch["force"])
        self.assertIn("bleiben auch mit Force bestehen", four["verdict"])
        one = self.ed.dry_run(r["doc"], RIG[:1])
        topo = {q["code"]: q for q in one["rejections"]}["HW-TOPOLOGY"]                      # no flip topology for one card
        self.assertEqual((topo["klass"], topo["forcebar"], topo["force_state"]), ("nicht_forcebar", False, "blockiert"))

    def test_no_force_switch_and_no_start_route_in_the_dashboard(self):
        src = open(S.__file__, encoding="utf-8").read()
        for needle in ("/api/profil/start", "/api/profil/force"):
            self.assertNotIn(needle, src)
        js = open(os.path.join(os.path.dirname(HERE), "static", "profil.js"), encoding="utf-8").read()
        self.assertNotIn("force: true", js)
        self.assertNotIn('type="checkbox" id="pf-force"', js)

    def test_user_profiles_in_the_state_volume(self):
        r = self.ed.load("release", "demo")
        e = self.ed.edit(r["doc"], [{"key": "flag:--p-bs", "op": "set", "value": "5"}])
        s = self.ed.save(e["doc"], "mein-nf")
        self.assertTrue(s["path"].startswith(self.usr))
        saved = json.load(open(s["path"]))
        self.assertEqual(saved["schema"], "flliper.server/1")
        self.assertEqual(saved["name"], "mein-nf")
        self.assertEqual(saved["meta"]["based_on"]["name"], "demo")
        self.assertEqual({v["name"]: v.get("value") for v in saved["vars"]}["PROFILE_NAME"], "mein-nf")
        self.assertEqual([p["name"] for p in self.ed.list()["user"]], ["mein-nf"])
        again = self.ed.load("user", "mein-nf")
        row = self.rows(again)["flag:--p-bs"]
        self.assertEqual((row["value"], row["origin"]), ("5", "nutzer"))
        self.assertTrue(self.ed.export_env(again["doc"])["verified"])
        self.ed.delete("mein-nf")
        self.assertEqual(self.ed.list()["user"], [])

    def test_names_cannot_escape(self):
        r = self.ed.load("release", "demo")
        for bad in ("../x", "a/b", "", "A", ".hidden", "x" * 70):
            with self.assertRaises(P.ProfilError):
                self.ed.save(r["doc"], bad)
        with self.assertRaises(P.ProfilError):
            self.ed.load("release", "../demo")
        with self.assertRaises(P.ProfilError):
            self.ed.load("user", "nope")

    def test_bad_input_is_a_named_error(self):
        with self.assertRaises(P.ProfilError):
            self.ed.edit({"schema": "x"}, [])
        with self.assertRaises(P.ProfilError):
            self.ed.dry_run(self.ed.load("release", "demo")["doc"], [])
        with self.assertRaises(P.ProfilError):
            self.ed.dry_run(self.ed.load("release", "demo")["doc"], [{"card": "no-such-card"}])


class Routes(unittest.TestCase):
    def serve(self, edition):
        tmp = tempfile.mkdtemp(prefix="pf930r_")
        self.addCleanup(shutil.rmtree, tmp, True)
        ed, _rel, _usr = editor(tmp)
        app = SimpleNamespace(edition=edition, profil=ed, version="t")
        srv = ThreadingHTTPServer(("127.0.0.1", 0), S.make_handler(app))
        srv.daemon_threads = True
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        self.addCleanup(srv.shutdown)
        self.addCleanup(srv.server_close)
        return srv.server_address[1]

    def call(self, port, method, path, body=None, headers=None):
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
        h = dict(headers or {})
        data = None
        if body is not None:
            data = json.dumps(body)
            h["Content-Type"] = "application/json"
        c.request(method, path, body=data, headers=h)
        r = c.getresponse()
        txt = r.read().decode()
        return r.status, txt

    def test_lan_flow(self):
        port = self.serve("rig")
        st, txt = self.call(port, "GET", "/api/profil/list")
        self.assertEqual(st, 200)
        self.assertEqual([p["name"] for p in json.loads(txt)["release"]], ["demo"])
        st, txt = self.call(port, "POST", "/api/profil/load", {"kind": "release", "name": "demo"})
        self.assertEqual(st, 200)
        doc = json.loads(txt)["doc"]
        st, txt = self.call(port, "POST", "/api/profil/edit", {"doc": doc, "edits": [{"key": "flag:--p-bs", "op": "set", "value": "3"}]})
        self.assertEqual(st, 200)
        doc = json.loads(txt)["doc"]
        st, txt = self.call(port, "POST", "/api/profil/export", {"doc": doc})
        self.assertTrue(json.loads(txt)["verified"])
        st, txt = self.call(port, "POST", "/api/profil/dry", {"doc": doc, "cards": RIG[:2]})
        self.assertIn("HW-COUNT", txt)
        st, txt = self.call(port, "POST", "/api/profil/load", {"kind": "release", "name": "../x"})
        self.assertEqual(st, 400)
        self.assertFalse(json.loads(txt)["ok"])
        self.assertEqual(self.call(port, "POST", "/api/profil/start", {})[0], 404)
        self.assertEqual(self.call(port, "POST", "/api/other", {})[0], 404)

    def test_public_proxy_is_refused_and_release_edition_has_no_editor(self):
        port = self.serve("rig")
        px = {"X-Forwarded-For": "1.2.3.4"}
        self.assertEqual(self.call(port, "GET", "/api/profil/list", headers=px)[0], 403)
        self.assertEqual(self.call(port, "POST", "/api/profil/load", {"kind": "release", "name": "demo"}, headers=px)[0], 403)
        self.assertEqual(self.call(port, "POST", "/api/profil/save", {"doc": {}, "name": "x"}, headers=px)[0], 403)
        rel = self.serve("release")
        self.assertEqual(self.call(rel, "GET", "/api/profil/list")[0], 404)
        self.assertEqual(self.call(rel, "POST", "/api/profil/load", {})[0], 404)
        self.assertEqual(self.call(rel, "GET", "/profil.js")[0], 404)

    def test_edition_page_carries_the_tab_only_in_rig(self):
        html = open(os.path.join(os.path.dirname(HERE), "static", "index.html"), encoding="utf-8").read()
        self.assertIn('data-tab="profil"', S.edition_page(html, "rig"))
        rel = S.edition_page(html, "release")
        for needle in ('data-tab="profil"', "pf-root", "profil.js"):
            self.assertNotIn(needle, rel)

    def test_answers_pass_the_secret_guard(self):
        port = self.serve("rig")
        st, txt = self.call(port, "POST", "/api/profil/load", {"kind": "release", "name": "demo"})
        self.assertNotIn("admin-api-key", txt.lower())


if __name__ == "__main__":
    unittest.main()
