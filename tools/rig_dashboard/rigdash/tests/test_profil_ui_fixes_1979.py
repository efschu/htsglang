"""Auftrag 1979 (Browsertest des Profil-Editors): die Fehler, die im Browser gesehen wurden, als Regressionstests.

* ``profil.js``: schlägt die ERSTE Balken-Rechnung fehl, darf ``draw()`` nicht werfen (vorher: ``Object.keys(b.phases)`` auf undefined,
  der ganze Reiter fror auf "computing …" ein); ein Fehler NACH erfolgreicher Rechnung kennzeichnet die alten Balken als veraltet;
  die Rechnung läuft beim Öffnen des Faltbereichs nur einmal (nicht je Neuzeichnen).  Läuft in node mit einer Attrappe des DOM.
* ``ProfilEditor.dry_run``: ``topology.plan_topology`` importiert für N != 3 ``flliper`` (``pdflip/weight_exchange_region``); im Dashboard-
  Prozess ohne flliper-Umgebung ist das KEINE Ablehnung und darf nicht als HTTP 500 enden, sondern wird als nicht geprüft vermerkt.
* CSS-Struktur: Kartenraster ohne Mindestbreite des Inhalts, Hardware-Tabelle im eigenen Scrollbereich (Seite scrollte bei 390 und 1100 px seitlich).
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from rigdash import profil as P  # noqa: E402

from .test_profil_930 import FIXTURE_TREE, RIG, editor  # noqa: E402

STATIC = os.path.join(os.path.dirname(HERE), "static")
NODE = shutil.which("node") or ("/opt/node-v22.14.0-linux-x64/bin/node" if os.path.exists("/opt/node-v22.14.0-linux-x64/bin/node") else None)

HARNESS = r"""
const fs = require("fs");
const STATIC = process.argv[2], MODE = process.argv[3];
let unhandled = [];
process.on("unhandledRejection", (e) => unhandled.push(String(e && e.message || e)));
process.on("uncaughtException", (e) => unhandled.push(String(e && e.message || e)));
const root = { innerHTML: "", _h: {}, addEventListener(t, f) { this._h[t] = f; }, querySelector() { return null; }, querySelectorAll() { return []; } };
let PICK = "release:p";
global.window = global;
global.document = {
  getElementById(id) { return id === "pf-root" ? root : id === "pf-pick" ? { value: PICK } : id === "pf-name" ? { value: "x" } : id === "tab-profil" ? { hidden: true } : null; },
  activeElement: null, body: { appendChild() {} }, createElement() { return { style: {}, setAttribute() {}, getBoundingClientRect() { return { width: 0, height: 0 }; } }; } };
global.localStorage = { getItem() { return null; }, setItem() {} };
global.CSS = { escape: (s) => s };
const calls = [];
const VIEW = { rows: [], planner_only: [], removed: [], coverage: { rows: 0, explained: 0, curated: 0, harvested: 0, profil_kommentar: 0, unexplained: 0, changed: 0 } };
const DOC = { name: "p", line: "nf", args: [], meta: {}, vars: [] };
const GOOD = { ok: true, result: { phases: { alle: { bars: [{ label: "K0", total_mib: 100, budget_mib: 90, overflow_mib: 0, free_mib: 10, segments: [{ key: "weights", label: "Gewichte", mib: 80, origin: "x" }] }], context_floor_tokens: 1 } }, hints: [] }, model_path: "/m" };
let recomputeAnswer = () => [400, { ok: false, error: "kaputt-1979" }];
global.fetch = async (url, opt) => {
  const p = String(url).replace(/^api\/profil\//, "");
  calls.push(p);
  let status = 200, body;
  if (p === "list") body = { ok: true, release: [{ name: "p" }], user: [], cards: [{ id: "a", label: "A", arch: "sm86" }], rig_preset: { cards: [] }, register: [] };
  else if (p === "load") body = { ok: true, doc: DOC, view: VIEW, name: "p", line: "nf", groups: [] };
  else if (p === "recompute") { await new Promise((r) => setTimeout(r, 120)); [status, body] = recomputeAnswer(); }
  else body = { ok: false, error: "unerwartet " + p };
  return { ok: status < 400, status, text: async () => JSON.stringify(body) };
};
require(STATIC + "/profil_balken.js");
require(STATIC + "/profil.js");
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const click = (dataset) => root._h.click({ target: { closest: () => ({ dataset }) } });
(async () => {
  await window.RigProfil.show(); await sleep(30);
  click({ act: "load" }); await sleep(60);
  const out = {};
  root._h.toggle({ target: { dataset: { fold: "bars" }, open: true } });
  await sleep(350);                                   // Entprellung 300 ms vorbei, die Rechnung (120 ms) läuft
  root._h.toggle({ target: { dataset: { fold: "bars" }, open: true } });   // Neuzeichnen setzt das offene <details> neu: toggle erneut
  await sleep(500);
  out.recomputeCalls = calls.filter((c) => c === "recompute").length;
  out.afterFirstError = root.innerHTML;
  const loadsBefore = calls.filter((c) => c === "load").length;
  click({ act: "load" }); await sleep(80);
  out.reloaded = calls.filter((c) => c === "load").length - loadsBefore;
  out.msgAfterReload = root.innerHTML.indexOf("Release profile p loaded") >= 0;
  if (MODE === "stale") {
    recomputeAnswer = () => [200, GOOD];
    click({ act: "load" }); await sleep(600);
    out.good = root.innerHTML.indexOf("K0") >= 0;
    recomputeAnswer = () => [400, { ok: false, error: "spaeter-kaputt" }];
    // zweite Rechnung anstossen: Profil erneut laden (setView -> scheduleRecompute)
    click({ act: "load" }); await sleep(700);
    out.afterLaterError = root.innerHTML;
  }
  out.unhandled = unhandled;
  console.log(JSON.stringify(out));
})();
"""


@unittest.skipUnless(NODE, "node fehlt")
class ProfilJsBars(unittest.TestCase):
    def run_js(self, mode):
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as fh:
            fh.write(HARNESS)
        self.addCleanup(os.unlink, fh.name)
        r = subprocess.run([NODE, fh.name, STATIC, mode], capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        return json.loads(r.stdout.strip().splitlines()[-1])

    def test_first_failed_recompute_shows_the_message_and_the_tab_stays_alive(self):
        o = self.run_js("first")
        self.assertEqual(o["unhandled"], [], "draw() darf bei fehlgeschlagener erster Rechnung nicht werfen")
        self.assertIn("kaputt-1979", o["afterFirstError"])
        self.assertNotIn("computing …", o["afterFirstError"])
        self.assertEqual(o["reloaded"], 1)
        self.assertTrue(o["msgAfterReload"], "der Reiter muss nach dem Fehler weiter neu zeichnen")

    def test_opening_the_fold_computes_once(self):
        self.assertEqual(self.run_js("first")["recomputeCalls"], 1)

    def test_error_after_a_good_computation_marks_the_old_bars_as_stale(self):
        o = self.run_js("stale")
        self.assertTrue(o["good"])
        self.assertIn("spaeter-kaputt", o["afterLaterError"])
        self.assertIn("STALE", o["afterLaterError"])
        self.assertEqual(o["unhandled"], [])


class DryRunWithoutFlliper(unittest.TestCase):
    def test_import_error_of_the_topology_probe_is_a_note_not_a_500(self):
        tmp = tempfile.mkdtemp(prefix="pf1979_")
        self.addCleanup(shutil.rmtree, tmp, True)
        ed, _, _ = editor(tmp)
        ci, tp = ed.kp._mods()

        class NoFlliperTopology:
            TopologyRefused = tp.TopologyRefused

            @staticmethod
            def plan_topology(n):
                raise ModuleNotFoundError("No module named 'flliper'")

        ed.kp._mods = lambda: (ci, NoFlliperTopology)
        d = ed.dry_run(ed.load("release", "demo")["doc"], RIG[:2])
        self.assertTrue(d["ok"])
        self.assertTrue(any("Topology for 2 card(s) not checked" in n for n in d["notes"]), d["notes"])
        self.assertNotIn("HW-TOPOLOGY", [q["code"] for q in d["rejections"]])


@unittest.skipUnless(NODE, "node fehlt")
class BarKeyboard(unittest.TestCase):
    def test_segments_are_focusable_and_the_tooltip_text_has_no_doubled_source(self):
        js = os.path.join(STATIC, "profil_balken.js")
        code = ("const M=require(%r);"
                "const bar={label:'K',total_mib:100,budget_mib:80,overflow_mib:0,free_mib:10,"
                "segments:[{key:'overflow',label:'Ueberlauf',mib:5,origin:'gerechnet',src:'gerechnet'},{key:'kv',label:'KV',mib:70,origin:'Modellprofil (Index)',src:'Index'}]};"
                "console.log(JSON.stringify({html:M.render([bar],{base:0}),t0:M.tip(bar,0),t1:M.tip(bar,1)}))" % js)
        r = subprocess.run([NODE, "-e", code], capture_output=True, text=True, timeout=30)
        self.assertEqual(r.returncode, 0, r.stderr)
        o = json.loads(r.stdout)
        self.assertEqual(o["html"].count('tabindex="0"'), 2)
        self.assertNotIn("<span class=\"muted\">(gerechnet)</span>", o["t0"])         # Quelle steht schon in der Herkunft
        self.assertNotIn("<span class=\"muted\">(Index)</span>", o["t1"])
        src = open(js, encoding="utf-8").read()
        self.assertIn('addEventListener("focusin"', src)


class CssStructure(unittest.TestCase):
    def test_card_grid_and_hardware_table_cannot_widen_the_page(self):
        html = open(os.path.join(STATIC, "index.html"), encoding="utf-8").read()
        self.assertIn("minmax(min(280px, 100%), 1fr)", html)
        self.assertIn(".pf-card select { flex: 1 1 auto; min-width: 0; }", html)
        hw = open(os.path.join(STATIC, "hwprofil.js"), encoding="utf-8").read()
        self.assertIn(".hwp{font:13px/1.45 system-ui,sans-serif;max-width:100%;overflow-x:auto}", hw)


if __name__ == "__main__":
    unittest.main()
