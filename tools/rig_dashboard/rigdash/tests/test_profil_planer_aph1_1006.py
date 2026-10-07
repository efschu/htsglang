"""AP-H1 (Plan Profil-Planer 06.10., Zeile AP-H Teil 1): die eine Seite -- Daten (``profil_planer.py``), Darstellungslogik (``static/profil_planer.js``,
Node) und die Verdrahtung in ``profil.js`` (Node mit Attrappen-DOM).

Gepinnt:
  * ``list`` traegt ``planer``: vier Betriebsformen mit je einem Satz, ``vorschlag`` nur fuer die Formen, die ``propose`` kann (flip, tp) und nur mit Orakel;
    jeder Name der Abschnitte A-C steht im Katalog, keiner doppelt; Reglergrenzen = ``ProfilEditor.ZIELE_INT``.
  * Die Standardwerte der Dual-ENV-Tabelle stehen im Quelltext (dual_green.py, dual_share.py, environ.py); der Katalog traegt die vier Envs kuratiert,
    mit den Kanten K109-K116 (Belege), und ``/profil_planer.js`` wird ausgeliefert.
  * Zustandschip je Wert (vorgeschlagen / unbelegt / vom Launcher geloest / von Ihnen uebersteuert / Profil), Verdikt-Chip (geht / nur mit --force / verweigert /
    Hinweis / nicht geprueft / ungeprueft seit Aenderung), nie als Sperre; Je-Karte-Felder mit einem Feld je Rang, Warnung bei falscher Laenge; Dual-Tabelle
    (Lesen, Schreiben, Standard bei fehlender Zeile); Escaping; Betriebsform aus dem Profil (--dual-share impliziert --dual-layout).
  * ``profil.js`` mit ``planer`` in der Antwort: sechs Schritte; Vorschlag POSTet {basis, form, inventar, ziele}; ohne ``planer`` bleibt die alte Seite.
"""

import ast
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from rigdash import profil as P  # noqa: E402
from rigdash import profil_planer as PL  # noqa: E402

STATIC = os.path.join(os.path.dirname(HERE), "static")
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
WEG2 = os.path.join(REPO_ROOT, "python", "sglang", "srt", "weg2")
CATALOG = os.path.join(os.path.dirname(HERE), "profil_data", "catalog.json")
NODE = shutil.which("node") or ("/opt/node-v22.14.0-linux-x64/bin/node" if os.path.exists("/opt/node-v22.14.0-linux-x64/bin/node") else None)


def _catalog():
    with open(CATALOG, encoding="utf-8") as fh:
        return json.load(fh)


def _src(name):
    with open(os.path.join(WEG2, name), encoding="utf-8") as fh:
        return fh.read()


class Daten(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cat = _catalog()

    def test_four_forms_each_with_a_sentence(self):
        ids = [f["id"] for f in PL.FORMEN]
        self.assertEqual(ids, ["einzel", "tp", "flip", "dual"])
        for f in PL.FORMEN:
            self.assertGreater(len(f["satz"]), 60, f["id"])
            self.assertTrue(f["quelle"], f["id"])
        self.assertEqual((PL.FORMEN[0]["n_min"], PL.FORMEN[0]["n_max"]), (1, 1))                # R2: Einzelkarte N=1
        self.assertTrue(all(f["n_min"] == 2 and f["n_max"] is None for f in PL.FORMEN[1:]))     # R2: sonst N>=2

    def test_proposal_only_for_forms_the_editor_can_and_with_an_oracle(self):
        ui = PL.ui_info(P.ProfilEditor.FORMS, self.cat["entries"], True)
        self.assertEqual({f["id"]: f["vorschlag"] for f in ui["formen"]}, {"einzel": True, "tp": True, "flip": True, "dual": True})      # AP-D Runde 2: alle vier Formen
        old = PL.ui_info(("flip", "tp"), self.cat["entries"], True)                        # ein Editor, der nur flip und tp kann: ehrlicher Hinweis statt "späteres Arbeitspaket"
        self.assertEqual({f["id"]: f["vorschlag"] for f in old["formen"]}, {"einzel": False, "tp": True, "flip": True, "dual": False})
        self.assertFalse(any("Arbeitspaket" in f.get("hinweis", "") for f in old["formen"]))
        for f in ui["formen"]:
            self.assertEqual("hinweis" in f, not f["vorschlag"], f["id"])
        no = PL.ui_info(P.ProfilEditor.FORMS, self.cat["entries"], False)
        self.assertFalse(any(f["vorschlag"] for f in no["formen"]))
        self.assertTrue(all("Orakel" in f["hinweis"] for f in no["formen"]))

    def test_every_section_name_is_in_the_catalog_exactly_once(self):
        names = PL.all_section_names()
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual([n for n in names if n not in self.cat["entries"]], [])
        self.assertEqual([a["id"] for a in PL.ABSCHNITTE], ["A", "B", "C"])

    def test_regulator_bounds_are_the_editors_bounds(self):
        self.assertEqual(PL.ZIELE["seats"], list(P.ProfilEditor.ZIELE_INT["seats"]))
        self.assertEqual(PL.ZIELE["kv_tokens"], list(P.ProfilEditor.ZIELE_INT["kv_tokens"]))

    def test_dual_defaults_are_in_the_source(self):
        g, s, e, dsh = _src("dual_green.py"), _src("dual_share.py"), None, _src("dual_share.py")
        self.assertIn("((2, 1, 0), (4, 2, 1), (10 ** 9, 3, 2))", g)
        self.assertEqual(PL.DUAL["table_default"], "2:1:0;4:2:1;1000000000:3:2")
        self.assertIn("rungs: Tuple[float, ...] = (1.0, 0.75, 0.5, 0.25)", s)
        self.assertEqual(PL.DUAL["rungs_default"], [1.0, 0.75, 0.5, 0.25])
        self.assertIn("starve_age_s: float = 60.0", s)
        self.assertIn("starve_max_rung: int = 1", s)
        self.assertEqual((PL.DUAL["starve_age_default"], PL.DUAL["starve_max_default"]), (60.0, 1))
        with open(os.path.join(REPO_ROOT, "python", "sglang", "srt", "environ.py"), encoding="utf-8") as fh:
            self.assertIn("SGLANG_WEG2_DUAL_GRANT_RETRY_MS = EnvInt(0)", fh.read())
        self.assertEqual(PL.DUAL["retry_default"], 0)
        self.assertIn('ENV_PREFIX = "SGLANG_WEG2_DUAL_SHARE_"', dsh)
        self.assertIn('g("TABLE")', g)
        self.assertIn('("STARVE_AGE_S", "starve_age_s", float)', s)
        self.assertIn('("STARVE_MAX_RUNG", "starve_max_rung", int)', s)

    def test_the_four_dual_envs_are_curated_with_edges(self):
        for role, name in PL.DUAL_ENV.items():
            if role == "rungs":
                continue
            e = self.cat["entries"][name]
            self.assertEqual(e["status"], "kuratiert", name)
            self.assertGreater(len(e["text"]), 80, name)
            self.assertTrue(e["depends"], name)
            self.assertTrue(all(d["belegt"] and d["kante"] for d in e["depends"]), name)
        deps = {d["to"] for d in self.cat["entries"][PL.DUAL_ENV["table"]]["depends"]}
        self.assertEqual(deps, {"--dual-priority", "--dual-share-actuators", "--dual-green-ladder", "--dual-d-min-rate-tps"})
        ui = PL.ui_info((), self.cat["entries"], False)
        self.assertEqual(len(ui["dual"]["werte"][PL.DUAL_ENV["table"]]["depends"]), 4)

    def test_list_carries_planer_and_the_route_serves_the_module(self):
        from rigdash import kartenplan as K
        tmp = tempfile.mkdtemp(prefix="aph1_")
        try:
            tree = os.path.join(HERE, "fixtures", "kartenplan", "planner_tree", "python")
            ed = P.ProfilEditor(kartenplaner=K.Kartenplaner(tree=tree), release_dir=os.path.join(tmp, "rel"), user_dir=os.path.join(tmp, "usr"), tree=tree, catalog_file=CATALOG, oracle=object())
            pl = ed.list()["planer"]
            self.assertEqual(pl["schema"], PL.SCHEMA)
            self.assertTrue(pl["oracle"])
            self.assertEqual([f["vorschlag"] for f in pl["formen"]], [True, True, True, True])      # einzel, tp, flip, dual: AP-D Runde 2
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        with open(os.path.join(os.path.dirname(HERE), "server.py"), encoding="utf-8") as fh:
            self.assertIn('"/profil_planer.js": ("profil_planer.js"', fh.read())
        with open(os.path.join(STATIC, "index.html"), encoding="utf-8") as fh:
            self.assertIn('<script src="profil_planer.js"></script>', fh.read())


JS_HEAD = r"""
const PX = require(process.argv[2] + "/profil_planer.js");
const out = {};
const dep = (d) => '<i class="dep">' + d.to + '</i>';
const base = (o) => Object.assign({ key: "flag:--pp-stage-ratio", name: "--pp-stage-ratio", scope: "launcher", value: "31,17,16", bare: false, multi: false, origin: "profil", origin_label: "Profil",
  profile_value: "31,17,16", planner_value: null, changed: false,
  explain: { status: "kuratiert", parts: [{ kind: "kuratiert", text: "Layer je Karte", source: "c.py" }], depends: [], gain: "", cost: "", group: "", level: "einfach", choices: null } }, o || {});
const ctx = (o) => Object.assign({ vecNames: new Set(__VEC__), posNames: new Set(__POS__), rankNames: new Set(__RANK__), n: 3, ranks: [], mode: "experte", prop: null, vsrc: "prop", dry: null, open: {}, cmsg: null, hasProfileValues: true,
  short: (r) => (r.explain.parts[0] || {}).text || "", explain: () => "<div>voll</div>", depChip: dep, isOpen: () => true }, o || {});
"""


JS_HEAD = JS_HEAD.replace("__VEC__", json.dumps(PL.vector_names())).replace("__POS__", json.dumps(PL.POSITIONAL_FLAGS + [t.rstrip("=") for t in PL.POSITIONAL_TOKENS])).replace("__RANK__", json.dumps(PL.RANK_VECTORS))


def run_node(body):
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as fh:
        fh.write(JS_HEAD + body + "\nconsole.log(JSON.stringify(out));\n")
    try:
        r = subprocess.run([NODE, fh.name, STATIC], capture_output=True, text=True, timeout=60)
    finally:
        os.unlink(fh.name)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1])


@unittest.skipUnless(NODE, "node fehlt")
class Darstellung(unittest.TestCase):
    def test_vector_detection_and_sum(self):
        o = run_node("""
out.v = [PX.vecSplit("31,17,16"), PX.vecSplit("0.5, 0.25"), PX.vecSplit("a:b,c"), PX.vecSplit("42"), PX.vecSplit("host,worker,worker"), PX.vecSplit("1,2 3,4")];
out.sum = [PX.vecSum(["31","17","16"]), PX.vecSum(["a","1"]), PX.vecSum(["0.1","0.2"])];
out.join = PX.vecJoin([" 1", "2 "]);
""")
        self.assertEqual(o["v"], [["31", "17", "16"], ["0.5", "0.25"], None, None, ["host", "worker", "worker"], None])
        self.assertEqual(o["sum"], [64, None, 0.3])
        self.assertEqual(o["join"], "1,2")

    def test_form_from_the_profile(self):
        o = run_node("""
const d = (flags) => ({ args: flags.map((f) => ({ flag: f, values: [] })) });
out.f = [PX.formOf(d(["--dual-share"]), 3), PX.formOf(d(["--dual-layout"]), 3), PX.formOf(d(["--d-only"]), 3), PX.formOf(d(["--p-bs"]), 3), PX.formOf(d([]), 1), PX.formOf(d([]), null)];
out.mm = [PX.formMismatch({ n_min: 1, n_max: 1 }, 3), PX.formMismatch({ n_min: 2, n_max: null }, 1), PX.formMismatch({ n_min: 2, n_max: null }, 3), PX.formMismatch({ n_min: 2, n_max: null }, null)];
""")
        self.assertEqual(o["f"], ["dual", "dual", "tp", "flip", "einzel", "flip"])      # --dual-share impliziert --dual-layout (launcher.py:14693)
        self.assertEqual(o["mm"], [True, True, False, False])

    def test_states(self):
        o = run_node("""
const W = (o) => Object.assign({ key: "flag:--pp-stage-ratio", wert: "1,2,3", alt: "3,2,1", zustand: "vorgeschlagen", herkunft: "H", grund: "G", geaendert: true, verdikte: [] }, o || {});
const z = (row, c) => PX.zustandOf(row, ctx(c));
out.z = [
  z(base({ origin: "nutzer" }), { prop: { werte: [W()] } }).id,
  z(base({ origin: "planer" }), { prop: { werte: [W({ zustand: "unbelegt" })] } }).id,
  z(base({ origin: "planer" }), { prop: { werte: [W()] } }).id,
  z(base({ planner_value: "31,17,16" })).id,
  z(base()).id,
  z(base({ absent: true })).id,
  z(base({ origin: "nutzer", planner_value: "31,17,16" }), { prop: { werte: [W({ zustand: "unbelegt" })] } }).id,
];
out.tip = z(base({ origin: "nutzer" }), { prop: { werte: [W()] } }).tip;
""")
        self.assertEqual(o["z"], ["uebersteuert", "unbelegt", "vorgeschlagen", "launcher", "profil", "standard", "uebersteuert"])
        self.assertIn("Vorschlag des Planers: 1,2,3", o["tip"])

    def test_verdicts_never_lock(self):
        o = run_node("""
const V = (code, o) => Object.assign({ code, ebene: "lauf", forcebar: true, force_state: "force", grund: code + " grund", konsequenz: "k" }, o || {});
const W = (vs) => ({ key: "flag:--pp-stage-ratio", wert: "1", zustand: "vorgeschlagen", geaendert: true, verdikte: vs });
const v = (vs, row, c) => PX.verdiktOf(row || base(), ctx(Object.assign({ prop: { werte: [W(vs)] } }, c || {})));
out.ids = [v([]).id, v([V("W40")]).id, v([V("W40"), V("X", { forcebar: false, force_state: "blockiert" })]).id, v([V("H", { forcebar: null, force_state: "hinweis" })]).id,
           v([V("U", { force_state: "ungeprueft" })]).id, PX.verdiktOf(base(), ctx()).id, v([V("W40")], base({ origin: "nutzer" })).id];
out.label = v([V("W40")]).label;
out.code = v([V("W40")]).code;
// Trockenlauf-Urteile: werte nennen Bezeichnungen, verglichen wird der Name
out.dry = PX.verdiktOf(base(), ctx({ dry: { verdikte: [V("W77", { werte: ["--extra-p --pp-stage-ratio"] }), V("W78", { werte: ["--anderer"] })] } })).code;
out.dryWins = PX.verdiktOf(base(), ctx({ prop: { werte: [W([])] }, vsrc: "dry", dry: { verdikte: [V("W77", { werte: ["--pp-stage-ratio"] })] } })).id;
// jedes Feld bleibt bedienbar, auch bei "verweigert"
const html = PX.renderRow(base(), ctx({ prop: { werte: [W([V("X", { forcebar: false, force_state: "blockiert" })])] } }));
out.html = html;
""")
        self.assertEqual(o["ids"], ["geht", "force", "verweigert", "hinweis", "ungeprueft", "keins", "alt"])
        self.assertEqual((o["label"], o["code"]), ("nur mit --force", "W40"))
        self.assertEqual(o["dry"], "W77")
        self.assertEqual(o["dryWins"], "force")
        self.assertIn("pfx-v-verweigert", o["html"])
        self.assertIn("auch mit Force nicht übergehbar", o["html"])
        self.assertNotIn("disabled", o["html"])
        self.assertNotIn("readonly", o["html"])

    def test_row_has_one_field_per_rank_and_warns_on_wrong_length(self):
        o = run_node("""
const c = ctx({ n: 3, ranks: [{ name: "NVIDIA GeForce RTX 5090", mib: 32607 }, { name: "RTX 3080", mib: 20480 }, { name: "RTX 3080", mib: 20480 }] });
out.ok = PX.renderRow(base(), c);
out.bad = PX.renderRow(base({ value: "1,2" }), c);
out.scalar = PX.renderRow(base({ key: "flag:--d-tp-objective", name: "--d-tp-objective", value: "decode-bs1" }), c);
out.choice = PX.renderRow(base({ key: "flag:--rank-tp-ratio", name: "--rank-tp-ratio", value: "a,b", explain: Object.assign({}, base().explain, { choices: ["a,b", "c"] }) }), c);
out.nonum = PX.renderRow(base({ key: "flag:--rank-tp-ratio", name: "--rank-tp-ratio", value: "a:1,b:2" }), c);
out.unnamed = PX.renderRow(base({ key: "flag:--dual-share-actuators", name: "--dual-share-actuators", value: "green,duty" }), c);
out.nolist = PX.renderRow(base(), Object.assign({}, c, { vecNames: undefined }));
""")
        self.assertEqual(len(re.findall(r'data-vi="', o["ok"])), 3)
        self.assertIn("Rang 0 · RTX 5090", o["ok"])
        self.assertIn("Σ 64", o["ok"])
        self.assertNotIn("Einträge, aber", o["ok"])
        self.assertEqual(len(re.findall(r'data-vi="', o["bad"])), 2)
        self.assertIn("2 Einträge, aber 3 Karten", o["bad"])
        self.assertNotIn("data-vi", o["scalar"])
        self.assertIn('data-k="flag:--d-tp-objective"', o["scalar"])
        self.assertNotIn("data-vi", o["choice"])                    # eine Auswahl bleibt eine Auswahl
        self.assertNotIn("data-vi", o["nonum"])
        self.assertNotIn("data-vi", o["unnamed"])                   # Review Runde 2: ein Kommatext allein ist noch kein Rang-Vektor
        self.assertNotIn("Einträge, aber", o["unnamed"])
        self.assertIn('data-k="flag:--dual-share-actuators" value="green,duty"', o["unnamed"])
        self.assertNotIn("data-vi", o["nolist"])                    # ohne Vektorliste vom Server: Textfeld

    def test_html_is_escaped(self):
        o = run_node("""
const evil = '"><img src=x onerror=alert(1)>';
out.h = PX.renderRow(base({ name: evil, key: "flag:" + evil, value: evil + ",2", explain: Object.assign({}, base().explain, { parts: [{ kind: "kuratiert", text: evil, source: "s" }] }) }), ctx({ short: () => evil }));
out.p = PX.renderProposal({ n: 2, form: evil, werte: [{ label: evil, alt: evil, wert: evil, zustand: "x", geaendert: true, verdikte: [{ code: evil }] }], verdikt: { ausgang: "geht", verdikte: [{ ebene: "lauf", code: evil, grund: evil, forcebar: true }] }, vorschlag: { cards: [{ name: evil, total_mib: 1 }] }, notes: [evil] });
""")
        for k in ("h", "p"):
            self.assertNotIn("<img", o[k])
            self.assertIn("&lt;img", o[k])                                                  # escaped, nicht roh

    def test_section_keeps_the_sections_order_and_reports_its_keys(self):
        o = run_node("""
const rows = [base({ key: "flag:--rank-gpu-memory-mib", name: "--rank-gpu-memory-mib" }), base({ key: "flag:--other", name: "--other" }), base()];
const sec = { id: "A", titel: "A  Aufteilung", satz: "s", namen: ["--pp-stage-ratio", "--rank-gpu-memory-mib", "--rank-kv-ratio"] };
const r = PX.renderSection(sec, rows, ctx(), "<i>EXTRA</i>", [{ key: "flag:--rank-kv-ratio", value: "1,1,1" }]);
out.keys = r.keys; out.html = r.html;
const e = PX.renderSection(sec, rows, ctx({ mode: "einfach" }), "", []);
out.einfach = e.html;
""")
        self.assertEqual(o["keys"], ["flag:--pp-stage-ratio", "flag:--rank-gpu-memory-mib"])
        self.assertLess(o["html"].index("--pp-stage-ratio"), o["html"].index("--rank-gpu-memory-mib"))
        self.assertNotIn("--other", o["html"])
        self.assertIn("EXTRA", o["html"])
        self.assertIn("vom Launcher gelöst: 1,1,1", o["html"])             # nicht im Profil, aber der Launcher rechnet ihn (planner_only)
        self.assertIn('data-take="flag:--rank-kv-ratio"', o["html"])
        self.assertIn('data-addflag="--rank-kv-ratio"', o["html"])
        self.assertNotIn("nicht im Profil gesetzt", o["einfach"])           # die Liste fehlender Werte nur in der Expertenansicht

    def test_green_table_round_trip(self):
        o = run_node("""
const g = PX.parseGreen("1:1:1;2:2:2;1000000000:3:3");
out.g = g; out.s = PX.serializeGreen(g.rows);
out.bad = [PX.parseGreen("").ok, PX.parseGreen("1:2").ok, PX.parseGreen("a:b:c").ok, PX.parseGreen("2:1:0;4:2:1").rows.length];
out.rp = [PX.rungPercents(null, [1, 0.75, 0.5, 0.25]).pct, PX.rungPercents("1,0.5", null).pct, PX.rungPercents("x", [1, 0.75]).pct];
""")
        self.assertEqual(o["s"], "1:1:1;2:2:2;1000000000:3:3")
        self.assertEqual(o["g"]["rows"][2], {"bs": 1000000000, "lo": 3, "hi": 3})
        self.assertEqual(o["bad"], [False, False, False, 2])
        self.assertEqual(o["rp"], [[100, 75, 50, 25], [100, 50], [100, 75]])

    def test_dual_block_reads_the_profile_and_falls_back_to_the_code_defaults(self):
        ui = PL.ui_info(("flip", "tp"), _catalog()["entries"], True)
        o = run_node("""
const info = %s;
const E = info.dual.env;
const mk = (name, value) => base({ key: "form:" + name, name, scope: "form", value, explain: Object.assign({}, base().explain, { level: "experte" }) });
out.full = PX.renderDual([mk(E.table, "1:1:1;2:2:2;1000000000:3:3"), mk(E.starve_age, "30"), mk(E.starve_max, "2"), mk(E.retry, "20")], info, ctx());
out.empty = PX.renderDual([], info, ctx());
out.none = PX.renderDual([], {}, ctx());
""" % json.dumps(ui))
        self.assertEqual(len(re.findall(r'data-gt="lo"', o["full"])), 3)
        self.assertIn('value="30"', o["full"])
        self.assertIn('<option value="2" selected>Stufe 2 · P mindestens 50 %</option>', o["full"])
        self.assertIn("alle größeren", o["full"])                           # Schwelle 10**9
        self.assertIn('data-gtab="form:SGLANG_WEG2_DUAL_SHARE_GREEN_TABLE"', o["full"])
        self.assertIn("Stufe 3 · P mindestens 25 % (Klemme AUS)", o["full"])
        # ohne Zeilen im Profil: Standard des Codes (2:1:0;4:2:1;10**9:3:2), jede Zeile mit "Standard"-Chip und den Kanten des Katalogs
        self.assertIn('value="60"', o["empty"])
        self.assertIn("Standard (nicht im Profil)", o["empty"])
        self.assertIn("--dual-priority", o["empty"])
        self.assertRegex(o["empty"], r'data-gt="lo" data-gi="0"[^>]*>(?:(?!</select>).)*<option value="1" selected>')     # Zeile 1: tau niedrig 1
        self.assertEqual(o["none"], "")

    def test_form_pick_regulators_and_proposal(self):
        ui = PL.ui_info(("flip", "tp"), _catalog()["entries"], True)
        o = run_node("""
const info = %s;
out.pick = PX.renderFormPick(info, "flip", 3);
out.pick1 = PX.renderFormPick(info, "einzel", 1);
const s = (o) => Object.assign({ form: "flip", seats: 6, seatsOn: false, ctx: 262144, ctxOn: false, busy: false, canPropose: true, whyNot: "", canCheck: true }, o || {});
out.c_ok = PX.renderControls(info, s());
out.c_dual = PX.renderControls(info, s({ form: "dual" }));
out.c_no = PX.renderControls(info, s({ canPropose: false, whyNot: "Erst ein Profil laden" }));
out.c_busy = PX.renderControls(info, s({ busy: true }));
out.prop = PX.renderProposal({ n: 3, form: "flip", werte: [{ label: "--d-bs", alt: "6", wert: "4", zustand: "vorgeschlagen", geaendert: true, verdikte: [] }, { label: "x", zustand: "unbelegt", geaendert: false }],
  verdikt: { ausgang: "geht_mit_force", verdikte: [{ ebene: "lauf", code: "W40", grund: "g", forcebar: true, force_state: "force" }] },
  vorschlag: { cards: [{ name: "NVIDIA GeForce RTX 5090", total_mib: 32607 }], fit: { level: "ja", margin_mib: 31544.2928, first: "" } }, notes: ["n1"] });
""" % json.dumps(ui))
        self.assertEqual(len(re.findall(r'data-form="', o["pick"])), 4)
        self.assertIn('data-form="flip" role="radio" aria-checked="true"', o["pick"])
        self.assertIn("Passt nicht zur Kartenzahl (3 Karten; diese Form braucht 1)", o["pick"])      # Einzelkarte bei 3 Karten
        self.assertNotIn("Passt nicht", o["pick1"].split('data-form="einzel"')[1].split("</button>")[0])
        self.assertRegex(o["c_ok"], r'data-act="propose"(?! disabled)')
        self.assertRegex(o["c_dual"], r'data-act="propose" disabled')
        self.assertIn("keinen Vorschlag", o["c_dual"])                           # der Grund steht da, nicht nur ein grauer Knopf (kein Verweis auf ein späteres Paket)
        self.assertRegex(o["c_no"], r'data-act="propose" disabled')
        self.assertIn("Erst ein Profil laden", o["c_no"])
        self.assertIn("rechnet …", o["c_busy"])
        self.assertIn("Rand 31544 MiB", o["prop"])
        self.assertIn("--d-bs", o["prop"])
        self.assertIn("nur mit --force", o["prop"])
        self.assertIn("keine Sperre", o["prop"])


HARNESS = r"""
const STATIC = process.argv[2], CASE = JSON.parse(process.argv[3]);
let unhandled = [];
process.on("unhandledRejection", (e) => unhandled.push(String(e && e.message || e)));
process.on("uncaughtException", (e) => unhandled.push(String(e && e.message || e)));
const root = { innerHTML: "", _h: {}, addEventListener(t, f) { this._h[t] = f; }, querySelector() { return null; }, querySelectorAll() { return []; } };
global.window = global;
global.document = {
  getElementById(id) { return id === "pf-root" ? root : id === "pf-pick" ? { value: "release:p" } : id === "pf-name" ? { value: "x" } : id === "tab-profil" ? { hidden: true } : null; },
  activeElement: null, documentElement: { getAttribute() { return "rig"; } }, body: { appendChild() {} }, createElement() { return { style: {}, setAttribute() {}, addEventListener() {}, getBoundingClientRect() { return { width: 0, height: 0 }; } }; } };
global.localStorage = { getItem() { return null; }, setItem() {} };
global.CSS = { escape: (s) => s };
const posted = [];
const PLANER = CASE.planer;
const row = (name, value) => ({ key: "flag:" + name, name, scope: "launcher", value, bare: false, origin: "profil", origin_label: "Profil", changed: false, profile_value: value, planner_value: null,
  explain: { status: "kuratiert", parts: [{ kind: "kuratiert", text: "Erklaerung " + name, source: "c.py" }], depends: [], gain: "", cost: "", group: "", level: "einfach", planner_derived: false, source: null, default: null, choices: null } });
const VIEW = { rows: [row("--pp-stage-ratio", "31,17,16"), row("--p-bs", "2")], planner_only: [], removed: [], kvheads: [],
  coverage: { rows: 2, erklaert: 2, kuratiert: 2, geerntet: 0, profil_kommentar: 0, unerklaert: 0, geaendert: 0 } };
if (CASE.extra) CASE.extra.forEach((e) => VIEW.rows.push(row(e[0], e[1])));
const DOC = { name: "p", line: "27b", args: [{ flag: "--pp-stage-ratio", values: ["31,17,16"] }], meta: {}, vars: [] };
global.fetch = async (url, opt) => {
  const u = String(url);
  let body;
  if (u === "api/hwprofil") body = { ok: true, profile: { cards: [{ name: "NVIDIA GeForce RTX 5090", vram_total_mib: { v: 32607 } }, { name: "RTX 3080", vram_total_mib: { v: 20480 } }, { name: "RTX 3080", vram_total_mib: { v: 20480 } }] } };
  else {
    const p = u.replace(/^api\/profil\//, "");
    if (p === "list") { body = { ok: true, release: [{ name: "p" }], user: [], cards: [{ id: "a", label: "A", arch: "sm86", preset: true }], rig_preset: { cards: [{ card: "a", pcie: { gen: 4, lanes: 8 } }] }, register: [] }; if (PLANER) body.planer = PLANER; }
    else if (p === "load") body = { ok: true, doc: DOC, view: VIEW, name: "p", line: "27b", groups: [] };
    else if (p === "propose") { posted.push(JSON.parse(opt.body)); body = { ok: true, n: 3, form: "flip", werte: [{ key: "flag:--pp-stage-ratio", label: "--pp-stage-ratio", wert: "30,18,16", alt: "31,17,16", zustand: "vorgeschlagen", herkunft: "H", grund: "G", geaendert: true, verdikte: [] }],
      verdikt: { ausgang: "geht", verdikte: [] }, vorschlag: { cards: [{ name: "RTX 5090", total_mib: 32607 }, { name: "RTX 3080", total_mib: 20480 }, { name: "RTX 3080", total_mib: 20480 }], fit: { level: "ja", margin_mib: 100 } }, notes: [],
      startprofil: { doc: Object.assign({}, DOC, { name: "p-vorschlag" }), view: Object.assign({}, VIEW, { rows: [Object.assign(row("--pp-stage-ratio", "30,18,16"), { origin: "planer", origin_label: "Planer", changed: true }), row("--p-bs", "2")] }), name: "p-vorschlag" } }; }
    else if (p === "recompute") body = { ok: false, error: "nicht im Test" };
    else body = { ok: false, error: "unerwartet " + p };
  }
  return { ok: true, status: 200, text: async () => JSON.stringify(body) };
};
require(STATIC + "/profil_balken.js");
if (CASE.module) require(STATIC + "/profil_planer.js");
require(STATIC + "/profil.js");
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const fire = (type, ev) => root._h[type](ev);
const click = (dataset) => fire("click", { target: { closest: () => ({ dataset }) } });
(async () => { try {
  await window.RigProfil.show(); await sleep(60);
  const out = { steps0: (root.innerHTML.match(/class="pfx-step"/g) || []).length };
  click({ act: "load" }); await sleep(60);
  out.afterLoad = root.innerHTML;
  click({ form: "tp" }); await sleep(10);
  out.tpSelected = /data-form="tp" role="radio" aria-checked="true"/.test(root.innerHTML);
  click({ form: "flip" }); await sleep(10);
  fire("change", { target: { dataset: { regOn: "seats" }, checked: true } }); await sleep(10);
  click({ act: "propose" }); await sleep(80);
  out.posted = posted;
  out.afterPropose = root.innerHTML;
  out.unhandled = unhandled;
  console.log(JSON.stringify(out));
} catch (e) { console.log(JSON.stringify({ crash: String(e && e.stack || e) })); } })();
"""


def run_harness(case):
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as fh:
        fh.write(HARNESS)
    try:
        r = subprocess.run([NODE, fh.name, STATIC, json.dumps(case)], capture_output=True, text=True, timeout=60)
    finally:
        os.unlink(fh.name)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert "crash" not in out, out["crash"]
    return out


@unittest.skipUnless(NODE, "node fehlt")
class ProfilJs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ui = PL.ui_info(("flip", "tp"), _catalog()["entries"], True)
        cls.new = run_harness({"planer": cls.ui, "module": True})
        cls.legacy_no_planer = run_harness({"planer": None, "module": True})
        cls.legacy_no_module = run_harness({"planer": cls.ui, "module": False})

    def test_nothing_unhandled(self):
        self.assertEqual(self.new["unhandled"], [])

    def test_six_steps_in_the_order_hardware_model_form_proposal_adapt_export(self):
        self.assertEqual(self.new["steps0"], 6)
        h = self.new["afterLoad"]
        idx = [h.index('id="pfx-h%d"' % i) for i in range(1, 7)]
        self.assertEqual(idx, sorted(idx))
        for t in ("Hardware", "Modell und Profil", "Betriebsform", "Vorschlag", "Anpassen", "Export"):
            self.assertIn(t, h)
        self.assertIn("3 Karten", h)                                  # das Hardwareprofil des Rigs (api/hwprofil)
        self.assertEqual(len(re.findall(r'data-vi="', h)), 3)         # --pp-stage-ratio als drei Felder
        self.assertTrue(self.new["tpSelected"])

    def test_propose_posts_basis_form_inventory_and_goals(self):
        self.assertEqual(self.new["posted"], [{"basis": {"kind": "release", "name": "p"}, "form": "flip", "inventar": "rig", "ziele": {"seats": 6}}])
        h = self.new["afterPropose"]
        self.assertIn("Vorschlag für 3 Karte", h)
        self.assertIn("pfx-z-vorgeschlagen", h)
        self.assertIn("Rang 0 · RTX 5090", h)                         # nach dem Vorschlag tragen die Felder die Karten in Rangfolge

    def test_without_planer_data_or_module_the_old_page_is_drawn(self):
        for o in (self.legacy_no_planer, self.legacy_no_module):
            self.assertEqual(o["unhandled"], [])
            self.assertEqual(o["steps0"], 0)
            self.assertNotIn("pfx-step", o["afterLoad"])
            self.assertIn("Trockenlauf", o["afterLoad"])


@unittest.skipUnless(NODE, "node fehlt")
class AllVectorsAreFields(unittest.TestCase):
    """Review-Befund 1: auch die positionalen Launcher-Vektoren ausserhalb der Namen A-C (Release-Profile nf*/27b*) bekommen ein Feld je Karte, kein Kommastring."""

    EXTRA = [["--d-foreign-context-mib", "1446,896,894"], ["--d-nontorch-mib", "1981,528,524"], ["SGLANG_WEG2_EXTEND_TRIM_MIB", "1200,0,0"],
             ["--d-reserve-mib", "100,200,300"], ["--pp-cut-reserve-mib", "10,20,30"]]

    @classmethod
    def setUpClass(cls):
        cls.ui = PL.ui_info(("flip", "tp"), _catalog()["entries"], True)
        cls.o = run_harness({"planer": cls.ui, "module": True, "extra": cls.EXTRA})
        cls.h = cls.o["afterLoad"]

    def test_the_launcher_vectors_are_named_in_the_sections(self):
        names = PL.all_section_names()
        for n in ("--d-foreign-context-mib", "--d-nontorch-mib", "--d-reserve-mib", "--pp-cut-reserve-mib", "SGLANG_WEG2_L15_MIB", "SGLANG_WEG2_EXTEND_TRIM_MIB"):
            self.assertIn(n, names)

    def test_every_vector_is_one_field_per_rank_and_no_comma_text_field(self):
        for name, val in self.EXTRA:
            self.assertEqual(len(re.findall(r'data-vk="flag:%s" data-vi="' % re.escape(name), self.h)), len(val.split(",")), name)
            self.assertNotIn('data-k="flag:%s" value="%s"' % (name, val), self.h, name)
            self.assertNotRegex(self.h, r'<input type="text" data-k="flag:%s" value="[^"]*,' % re.escape(name))

    def test_the_rest_list_does_not_repeat_them(self):
        i = self.h.index("E  Übrige Werte")
        self.assertNotIn("--d-foreign-context-mib", self.h[i:])      # steht in A, nicht noch einmal in E

    def test_nothing_unhandled(self):
        self.assertEqual(self.o["unhandled"], [])


@unittest.skipUnless(NODE, "node fehlt")
class OnlyNamedVectorsAreFields(unittest.TestCase):
    """Review-Befund 1 (Runde 2): Felder je Rang nur fuer eine ausdrueckliche Vektormenge, jede andere Kommaliste bleibt ein Textfeld."""

    EXTRA = [["--dual-share-actuators", "green,duty"], ["--cuda-graph-bs", "1,2,4,8"], ["--pp-layer-set", "0-2,4-6,7-9"],
             # Runde 3: vom Launcher aus der Je-Karte-Zaehlung genommen (_TOPOLOGY_VECTOR_FLAGS/-TOKENS): Textfeld, nie "N Einträge, aber M Karten"
             ["--p-barlink-bar1-window-mib", "24,PP_0=96"], ["--d-reshard-presets", "a:1,b:2"], ["SGLANG_WEG2_L15_MIB", "512,512"]]

    @classmethod
    def setUpClass(cls):
        cls.ui = PL.ui_info(("flip", "tp"), _catalog()["entries"], True)
        cls.o = run_harness({"planer": cls.ui, "module": True, "extra": cls.EXTRA})
        cls.h = cls.o["afterLoad"]

    def test_non_vector_comma_lists_stay_one_text_field(self):
        for name, val in self.EXTRA:
            self.assertNotIn('data-vk="flag:%s"' % name, self.h, name)
            self.assertIn('data-k="flag:%s" value="%s"' % (name, val), self.h, name)

    def test_no_made_up_rank_hint_for_them(self):
        self.assertNotIn("Einträge, aber", self.h)

    def test_the_launcher_excluded_ones_are_text_fields_not_vectors(self):
        """Review-Befund 1 (Runde 3): BAR1-Fenster, d_reshard-Presets, L1.5 zaehlt der Launcher nicht je Karte; kein PROFILE-VECTORS-Satz fuer sie."""
        for n in PL.LAUNCHER_NICHT_JE_KARTE:
            self.assertNotIn(n, PL.vector_names())
            self.assertNotIn(n, self.ui["vektoren"])
            self.assertNotIn(n, self.ui["positional"])
            self.assertNotRegex(self.h, r'data-vk="[a-z]+:%s"' % re.escape(n))
        for name, val in self.EXTRA[3:]:
            self.assertIn('data-k="flag:%s" value="%s"' % (name, val), self.h, name)
        self.assertNotIn("PROFILE-VECTORS", self.h)

    def test_the_vector_set_is_the_launchers_topology_set_plus_named_section_vectors(self):
        """vector_names()/POSITIONAL_* sind exakt das, was der Launcher je Karte zaehlt: _TOPOLOGY_VECTOR_FLAGS/-TOKENS, aus dem Quelltext nachgebildet (nicht abgeschrieben)."""
        tree = ast.parse(_src("launcher.py"))
        ns = {}
        want = ("POSITIONAL_VECTOR_FLAGS", "POSITIONAL_VECTOR_TOKENS", "_TOPOLOGY_VECTOR_FLAGS", "_TOPOLOGY_VECTOR_TOKENS")
        for node in tree.body:
            if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") in want:
                ns[node.targets[0].id] = eval(compile(ast.Expression(node.value), "launcher.py", "eval"), {}, dict(ns))
        self.assertEqual(sorted(ns), sorted(want))
        flags = ["--" + d.replace("_", "-") for d in ns["_TOPOLOGY_VECTOR_FLAGS"]]
        toks = [t.rstrip("=") for t in ns["_TOPOLOGY_VECTOR_TOKENS"]]
        self.assertEqual(PL.POSITIONAL_FLAGS, flags)
        self.assertEqual([t.rstrip("=") for t in PL.POSITIONAL_TOKENS], toks)
        # die volle Launcher-Liste minus genau die ausgenommenen
        full_f = ["--" + d.replace("_", "-") for d in ns["POSITIONAL_VECTOR_FLAGS"]]
        full_t = [t.rstrip("=") for t in ns["POSITIONAL_VECTOR_TOKENS"]]
        self.assertEqual(sorted(PL.POSITIONAL_FLAGS_ALL), sorted(full_f))
        self.assertEqual(sorted(t.rstrip("=") for t in PL.POSITIONAL_TOKENS_ALL), sorted(full_t))
        self.assertEqual(sorted(set(full_f + full_t) - set(flags + toks)), sorted(PL.LAUNCHER_NICHT_JE_KARTE))
        names = set(PL.vector_names())
        self.assertTrue(set(flags) | set(toks) <= names)
        self.assertEqual(len(PL.vector_names()), len(names))
        self.assertEqual(self.ui["vektoren"], PL.vector_names())
        entries = _catalog()["entries"]
        for n in names:
            self.assertIn(n, entries, n)
        for n in ("--dual-share-actuators", "--cuda-graph-bs", "--pp-layer-set", "--p-layer-split", "--kv-reshard-vectors"):
            self.assertNotIn(n, names)


@unittest.skipUnless(NODE, "node fehlt")
class RankIdIsPerRank(unittest.TestCase):
    """Review-Befund 2 (Runde 3): --rank-gpu-id hat je RANG ein Feld; Duplikate legen mehrere Raenge auf eine Karte."""

    def test_ui_info_names_it(self):
        self.assertEqual(PL.ui_info()["je_rang"], ["--rank-gpu-id"])
        self.assertIn("--rank-gpu-id", PL.vector_names())

    def test_fields_per_rank_without_card_names_sum_or_mismatch_warning(self):
        o = run_node("""
const c = ctx({ n: 3, ranks: [{ name: "NVIDIA GeForce RTX 5090", mib: 32607 }, { name: "RTX 3080", mib: 20480 }, { name: "RTX 3080", mib: 20480 }] });
const row = (v) => base({ key: "flag:--rank-gpu-id", name: "--rank-gpu-id", value: v });
out.four = PX.renderRow(row("1,1,0,2"), c);
out.three = PX.renderRow(row("1,0,2"), c);
out.other = PX.renderRow(base({ key: "flag:--rank-tp-ratio", name: "--rank-tp-ratio", value: "1,1,1" }), c);
""")
        for k, n in (("four", 4), ("three", 3)):
            self.assertEqual(len(re.findall(r'data-vi="', o[k])), n, k)
            self.assertIn("%d Ränge" % n, o[k])
            self.assertNotIn("Einträge, aber", o[k])
            self.assertNotIn("Σ", o[k])
            self.assertNotIn("RTX", o[k])                          # kein Kartenname am Rangfeld
            self.assertIn("Rang 3" if n == 4 else "Rang 2", o[k])
        self.assertIn("Σ 3", o["other"])                           # die anderen Vektoren bleiben je Karte mit Summe
        self.assertIn("Rang 0 · RTX 5090", o["other"])


class Css(unittest.TestCase):
    def test_phone_rules_exist(self):
        html = open(os.path.join(STATIC, "index.html"), encoding="utf-8").read()
        i = html.index("/* Telefon 390 px: alles stapelt */")
        self.assertIn("@media (max-width: 640px) {", html[i - 40:i])
        self.assertIn(".pfx-reg { grid-template-columns: 1fr 100px; }", html[i:i + 600])
        self.assertIn(".pfx-gt thead { display: none; }", html)
        self.assertIn(".pfx-sec .pf-dep { white-space: normal; max-width: 100%; }", html)


if __name__ == "__main__":
    unittest.main()
