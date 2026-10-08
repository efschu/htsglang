"""Auftrag 2002 (B + C): Force-Hinweis im Export (nur Text) und Kanten-Chips mit Beleg und Satz.

Nutzerentscheid 05.10.: Force-Hinweis Variante B (ohne Schalter, startet nichts) NACH der Register-Korrektur; Kantenkatalog Variante A.

Gepinnt:
  * Register-Korrektur: PROFIL-STATUS/SHM/STORE/MEMAVAIL gelten fuer den Docker-Start (Entrypoint) als forcebar (``force_via: entrypoint``,
    Text "forceable in the Docker start (entrypoint), not in a plain launcher call"), nicht mehr als "remains even with force".
  * Export: Beispiel-``docker run``; die Zeile ``-e FLLIPER_FORCE=1`` NUR wenn der letzte Trockenlauf forcebare Ablehnungen ergab; fuenf Faelle
    (+ "kein Trockenlauf"); nicht uebergehbare Codes als roter Klartext; der Server rechnet den Fall aus den CODES neu (Browser-force_state
    wird nicht geglaubt); "starts nothing"; kein Kommentar hinter einem ``\\`` (Bash).
  * Chips: Beleg + Satz + "ohne Beleg" + Bedingung (nur Text); Chips auf Ablehnungscodes zeigen auf das Register, nicht ins Leere; der
    veraltete Satz "comes with stage S4" ist weg.
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

from .test_profil_930 import RIG, Routes, editor  # noqa: E402

STATIC = os.path.join(os.path.dirname(HERE), "static")
NODE = shutil.which("node") or ("/opt/node-v22.14.0-linux-x64/bin/node" if os.path.exists("/opt/node-v22.14.0-linux-x64/bin/node") else None)


def reg_row(code, **kw):
    r = {"code": code, "klass": "value", "forcebar": True, "wired": False, "wired_entrypoint": False, "enforced_by": "launcher", "title": code + "-Titel"}
    r.update(kw)
    return r


class Verdict(unittest.TestCase):
    def test_entrypoint_only_code_is_forceable_in_docker_not_in_the_bare_launcher(self):
        text, state, via = P.force_verdict(reg_row("SHM", enforced_by="entrypoint", wired_entrypoint=True))
        self.assertEqual((state, via), ("force", "entrypoint"))
        self.assertIn("forceable in the Docker start (entrypoint), not in a plain launcher call", text)
        self.assertNotIn("bleibt", text)

    def test_launcher_wired_wins_and_old_registers_are_read_by_enforced_by(self):
        self.assertEqual(P.force_verdict(reg_row("HW-COUNT", wired=True, wired_entrypoint=True))[1:], ("force", "launcher"))
        old = {"code": "PROFIL-STATUS", "forcebar": True, "wired": False, "enforced_by": "entrypoint"}      # Register vor 2002 A: kein wired_entrypoint
        self.assertEqual(P.force_verdict(old)[1:], ("force", "entrypoint"))

    def test_hard_unwired_and_unchecked(self):
        self.assertEqual(P.force_verdict(reg_row("HW-ARCH", klass="nicht_forcebar", forcebar=False))[1:], ("is_blocked", None))
        t, s, v = P.force_verdict(reg_row("PP-CUT"))
        self.assertEqual((s, v), ("is_blocked", None))
        self.assertIn("wired neither in the launcher nor in the entrypoint", t)
        self.assertEqual(P.force_verdict(reg_row("X", enforced_by="planner-gate"))[1:], ("ungeprueft", None))
        self.assertEqual(P.force_verdict({})[1:], ("is_blocked", None))                                       # unbekannter Code


class WiredFromTheTree(unittest.TestCase):
    """07.10. (NF line): ``register()`` marks a forcebar code wired by what the launcher.py of the PLANNER TREE consults, not by the
    shipped catalog (which covers both lines and carries one launcher's list)."""

    def test_wired_comes_from_the_tree_launcher_not_from_the_catalog(self):
        from .test_profil_930 import FIXTURE_TREE
        tmp = tempfile.mkdtemp(prefix="pf2002w_")
        self.addCleanup(shutil.rmtree, tmp, True)
        tree = os.path.join(tmp, "python")
        wd = os.path.join(tree, "flliper", "srt", "pdflip")
        os.makedirs(wd)
        shutil.copy(os.path.join(FIXTURE_TREE, "flliper", "srt", "pdflip", "refusals.py"), wd)
        shutil.copy(os.path.join(FIXTURE_TREE, "flliper", "srt", "pdflip", "profile_json.py"), wd)
        with open(os.path.join(wd, "launcher.py"), "w") as fh:
            fh.write('refuse_value("HW-COUNT", "x")\nrefuse_value("HOST-MEM", "y")\n')
        ed2, _r, _u = editor(tmp)
        ed2.tree = tree
        ed2._mods = None
        wired = {r["code"] for r in ed2.register() if r.get("wired")}
        self.assertEqual(wired, {"HW-COUNT", "HOST-MEM"})
        # the catalog's own list (the other line's) is ignored when the launcher is readable
        self.assertNotEqual(wired, set(ed2.catalog().get("register_wired") or []))


class Dry(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="pf2002_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.ed, _r, _u = editor(self.tmp)
        self.doc = self.ed.load("release", "demo")["doc"]

    def test_profil_status_is_now_forceable_via_the_entrypoint(self):
        d = self.ed.dry_run(self.doc, RIG)
        q = {x["code"]: x for x in d["rejections"]}["PROFIL-STATUS"]
        self.assertEqual((q["force_state"], q["force_via"], q["wired_at"]), ("force", "entrypoint", "entrypoint"))
        self.assertIn("Docker start (entrypoint)", q["force"])
        self.assertIn("only in the Docker start", d["verdict"])
        self.assertNotIn("remain even with force", d["verdict"])
        self.assertIn("forceable in the Docker start (entrypoint), not in a plain launcher call", d["force_note"])
        self.assertIn("Not overridden", d["force_note"])

    def test_launcher_codes_keep_their_state(self):
        d = self.ed.dry_run(self.doc, RIG[:2])
        for q in d["rejections"]:
            if q["code"] in ("HW-COUNT", "HW-UNCALIBRATED"):
                self.assertEqual((q["force_state"], q["force_via"]), ("force", "launcher"))


class Hint(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="pf2002h_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.ed, _r, _u = editor(self.tmp)
        self.reg = self.ed.register()

    def hint(self, codes, line=""):
        return P.force_hint({"rejections": [{"code": c, "text": "Text " + c} for c in codes]}, self.reg, line)

    def run_lines(self, h, name="mein"):
        return P.docker_run_example(name, h)

    def test_case_0_no_dry_run_yet_no_force_line(self):
        for dry in (None, {}, {"rejections": "x"}, "boese"):
            h = P.force_hint(dry, self.reg)
            self.assertEqual(h["fall"], "kein_trockenlauf")
            self.assertFalse(h["show_line"])
            self.assertIsNone(h["force_env"])
            self.assertNotIn("FLLIPER_FORCE", "\n".join(self.run_lines(h)))
            self.assertIn("dry run", h["text"])

    def test_case_1_no_rejection_no_force_line_and_the_sentence(self):
        h = self.hint([])
        self.assertEqual((h["fall"], h["show_line"]), ("keine_ablehnung", False))
        self.assertEqual(h["text"], "The planner refuses nothing; force is not needed.")
        self.assertNotIn("FLLIPER_FORCE", "\n".join(self.run_lines(h)))

    def test_case_2_only_forceable_gets_the_line_with_the_code_list(self):
        h = self.hint(["PROFIL-STATUS", "SHM"])
        self.assertEqual((h["fall"], h["show_line"], h["force_env"]), ("nur_forcebar", True, "FLLIPER_FORCE=1"))
        self.assertEqual([c["code"] for c in h["force_codes"]], ["PROFIL-STATUS", "SHM"])
        self.assertEqual(h["blocked_codes"], [])
        lines = self.run_lines(h)
        self.assertIn("  -e FLLIPER_FORCE=1 \\", lines)
        self.assertEqual(lines[0], "# FLLIPER_FORCE=1 is needed only because the planner refuses: PROFIL-STATUS, SHM")
        self.assertIn("FORCED-PAST", h["text"])
        self.assertIn("no records", h["records_note"])

    def test_case_3_mixed_names_the_blocked_code_and_still_shows_the_line(self):
        h = self.hint(["PROFIL-STATUS", "HW-ARCH"])
        self.assertEqual((h["fall"], h["show_line"]), ("gemischt", True))
        self.assertEqual([c["code"] for c in h["blocked_codes"]], ["HW-ARCH"])
        self.assertIn("the server does not start this way", h["text"])

    def test_case_4_only_not_forceable_no_force_line_at_all(self):
        h = self.hint(["HW-TOPOLOGY"])
        self.assertEqual((h["fall"], h["show_line"], h["force_env"]), ("nur_nicht_forcebar", False, None))
        self.assertIn("Force does not help here", h["text"])
        self.assertNotIn("FLLIPER_FORCE", "\n".join(self.run_lines(h)))
        self.assertEqual([c["code"] for c in h["blocked_codes"]], ["HW-TOPOLOGY"])

    def test_case_5_unchecked_only_needs_no_force(self):
        reg = self.reg + [reg_row("GATE-X", enforced_by="planner-gate")]
        h = P.force_hint({"rejections": [{"code": "GATE-X", "text": "t"}]}, reg)
        self.assertEqual((h["fall"], h["show_line"]), ("ungeprueft", False))
        self.assertEqual([c["code"] for c in h["open_codes"]], ["GATE-X"])
        self.assertIn("does not check this yet", h["text"])

    def test_the_server_recomputes_from_codes_the_browser_state_is_not_believed(self):
        dry = {"rejections": [{"code": "HW-ARCH", "text": "x", "force_state": "force", "forcebar": True, "force_via": "launcher"},
                              {"code": "ERFUNDEN", "text": "y" * 5000, "force_state": "force"}, "kaputt", {"code": 7}, {"code": ""}]}
        h = P.force_hint(dry, self.reg)
        self.assertFalse(h["show_line"])                                      # HW-ARCH is hard, an invented code counts as blocked
        self.assertEqual(sorted(c["code"] for c in h["blocked_codes"]), ["ERFUNDEN", "HW-ARCH"])
        self.assertTrue(all(len(c["text"]) <= P.MAX_CODE_TEXT for c in h["blocked_codes"]))

    def test_duplicate_codes_listed_once_and_nf_gets_its_honest_note(self):
        h = self.hint(["PROFIL-STATUS", "PROFIL-STATUS"], line="nf")
        self.assertEqual(len(h["force_codes"]), 1)
        self.assertIn("only HW-COUNT, HW-UNCALIBRATED and HOST-MEM", h["line_note"])
        self.assertEqual(self.hint(["PROFIL-STATUS"], line="27b")["line_note"], "")

    def test_the_example_is_valid_bash_continuation_no_comment_behind_a_backslash(self):
        for codes in ([], ["PROFIL-STATUS"], ["PROFIL-STATUS", "HW-ARCH"]):
            lines = self.run_lines(self.hint(codes))
            for ln in lines:
                if ln.endswith("\\"):
                    self.assertNotIn("#", ln)
                if "#" in ln:
                    self.assertTrue(ln.startswith("#"))
                    self.assertTrue(lines.index(ln) < next(i for i, x in enumerate(lines) if x.startswith("docker run")))
            self.assertTrue(lines[-1].endswith("serve"))
            self.assertIn("  -e MODE=pdflip -e FLLIPER_PROFILE=mein \\", lines)

    def test_export_carries_the_hint_and_says_it_starts_nothing(self):
        r = self.ed.load("release", "demo")
        x0 = self.ed.export_env(r["doc"])
        self.assertIsNone(x0["use"]["force_env"])
        self.assertEqual(x0["use"]["force"]["fall"], "kein_trockenlauf")
        self.assertIn("starts nothing", x0["use"]["text"])
        self.assertNotIn("FLLIPER_FORCE=1", "\n".join(x0["use"]["docker_run"]))
        dry = self.ed.dry_run(r["doc"], RIG)
        x1 = self.ed.export_env(r["doc"], dry)
        self.assertEqual(x1["use"]["force_env"], "FLLIPER_FORCE=1")
        self.assertEqual(x1["use"]["force"]["fall"], "nur_forcebar")
        self.assertIn("  -e FLLIPER_FORCE=1 \\", x1["use"]["docker_run"])
        self.assertTrue(x1["verified"], x1["problems"])


class Route(Routes):
    def test_export_route_passes_the_last_dry_run_through(self):
        port = self.serve("rig")
        st, txt = self.call(port, "POST", "/api/profil/load", {"kind": "release", "name": "demo"})
        doc = json.loads(txt)["doc"]
        st, txt = self.call(port, "POST", "/api/profil/dry", {"doc": doc, "cards": RIG})
        dry = json.loads(txt)
        st, txt = self.call(port, "POST", "/api/profil/export", {"doc": doc, "dry": dry})
        self.assertEqual(st, 200)
        self.assertEqual(json.loads(txt)["use"]["force"]["fall"], "nur_forcebar")
        st, txt = self.call(port, "POST", "/api/profil/export", {"doc": doc})
        self.assertEqual(json.loads(txt)["use"]["force"]["fall"], "kein_trockenlauf")


class Catalog(unittest.TestCase):
    def test_shipped_catalog_carries_the_edge_status_and_fields(self):
        cat = json.load(open(os.path.join(os.path.dirname(HERE), "profil_data", "catalog.json"), encoding="utf-8"))
        k = cat["kanten"]
        self.assertTrue(k["geladen"])
        # Katalog-Neubau 07.10.: 131 Kanten (K117-K131 neu: Waechter-Envs, D-COMPACT, AUX-SPILL, --x-mode/--x-curves; neu 65 -> 80). AP-G 06.10.: 108 Kanten (K62-K108 neu: Form A / ungleiches DCP / Draft / D-only / Dual, alle belegt; neu 10 -> 57, ohne Beleg weiter 24). Davor 61 Kanten seit 05.10.: K60 (Graph-Kalibriertabelle <-> Layer-Schnitt) und K61 (Tabelle wirkt nur bei Politik auto) sind NEU dazugekommen
        # (neu 8 -> 10, alle 61 belegt); die 24 "ohne Beleg" sind kuratierte Kantenwünsche, unverändert
        # "ohne Beleg" = kuratierte Kanten ohne Katalogkante: 24 im Kern der 27B-Linie; der NF-eigene kuratierte Eintrag --pdflip-xchg-census-map (nur NF-Baum) traegt
        # eine weitere (braucht --pdflip-xchg-census), ist der Katalog aus dem NF-Kern gebaut (Neubau NF-Linie 07.10.), sind es 25
        nf_census_map = cat["entries"].get("--pdflip-xchg-census-map", {}).get("status") == "curated"
        self.assertEqual((k["edges_total"], k["verschmolzen"], k["neu"], k["edges_without_evidence"]), (131, 51, 80, 25 if nf_census_map else 24))
        deps = [d for e in cat["entries"].values() for d in e["depends"]]
        self.assertTrue(all("belegt" in d and "to_kind" in d for d in deps))
        self.assertTrue(any(d["to_kind"] == "refusal" for d in deps))


# ---------------------------------------------------------------------------------------------------------- node: Chips und Export-Anzeige

HARNESS = r"""
const STATIC = process.argv[2], CASE = JSON.parse(process.argv[3]);
let unhandled = [];
process.on("unhandledRejection", (e) => unhandled.push(String(e && e.message || e)));
process.on("uncaughtException", (e) => unhandled.push(String(e && e.message || e)));
const root = { innerHTML: "", _h: {}, addEventListener(t, f) { this._h[t] = f; }, querySelector() { return null; }, querySelectorAll() { return []; } };
global.window = global;
global.document = {
  getElementById(id) { return id === "pf-root" ? root : id === "pf-pick" ? { value: "release:p" } : id === "pf-name" ? { value: "x" } : id === "tab-profil" ? { hidden: true } : null; },
  activeElement: null, body: { appendChild() {} }, createElement() { return { style: {}, setAttribute() {}, getBoundingClientRect() { return { width: 0, height: 0 }; } }; } };
global.localStorage = { getItem() { return null; }, setItem() {} };
global.CSS = { escape: (s) => s };
const posted = {};
const row = (name, deps) => ({ key: "flag:" + name, name, scope: "launcher", value: "1", bare: false, origin: "profil", origin_label: "Profile", changed: false,
  profile_value: "1", planner_value: null, explain: { status: "curated", parts: [{ kind: "curated", text: "Erklaerung " + name, source: "c.py" }], depends: deps,
  gain: "", cost: "", group: "", level: "einfach", planner_derived: false, source: null, default: null, choices: null } });
const VIEW = { rows: [row("--a", CASE.deps)], planner_only: [], removed: [], coverage: { rows: 1, explained: 1, curated: 1, harvested: 0, profil_kommentar: 0, unexplained: 0, changed: 0 } };
const DOC = { name: "p", line: "27b", args: [], meta: {}, vars: [] };
global.fetch = async (url, opt) => {
  const p = String(url).replace(/^api\/profil\//, "");
  let body;
  if (p === "list") body = { ok: true, release: [{ name: "p" }], user: [], cards: [{ id: "a", label: "A", arch: "sm86" }], rig_preset: { cards: [{ card: "a", pcie: { gen: 4, lanes: 8 } }] }, register: CASE.register || [] };
  else if (p === "load") body = { ok: true, doc: DOC, view: VIEW, name: "p", line: "27b", groups: [] };
  else if (p === "dry") body = CASE.dry;
  else if (p === "export") { posted.export = JSON.parse(opt.body); body = CASE.export; }
  else if (p === "recompute") body = { ok: false, error: "nicht im Test" };
  else body = { ok: false, error: "unerwartet " + p };
  return { ok: true, status: 200, text: async () => JSON.stringify(body) };
};
require(STATIC + "/profil_balken.js");
require(STATIC + "/profil.js");
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const click = (dataset) => root._h.click({ target: { closest: () => ({ dataset }) } });
(async () => {
  await window.RigProfil.show(); await sleep(30);
  click({ act: "load" }); await sleep(60);
  const out = {};
  out.chips = root.innerHTML;
  click({ open: "flag:--a" }); await sleep(20);
  out.open = root.innerHTML;
  click({ goto: CASE.goto || "--zz" }); await sleep(20);
  out.gotoMsg = root.innerHTML;
  if (CASE.dry) { click({ act: "dry" }); await sleep(30); click({ act: "export" }); await sleep(60); }
  out.exported = root.innerHTML;
  out.posted = posted.export || null;
  out.unhandled = unhandled;
  console.log(JSON.stringify(out));
})();
"""


def run_js(case):
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as fh:
        fh.write(HARNESS)
    try:
        r = subprocess.run([NODE, fh.name, STATIC, json.dumps(case)], capture_output=True, text=True, timeout=60)
    finally:
        os.unlink(fh.name)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1])


DEPS = [
    {"to": "--b", "rel": "requires", "effect": "kuratierter Satz", "calc": "S4", "present": True, "belegt": True, "quelle": "catalog+curated", "edge": "K01",
     "evidence": {"file": "python/flliper/srt/pdflip/launcher.py", "zeile": 8617, "anchor": "must be given together"}, "satz": "Tradeoff-Satz K01", "value": None, "to_kind": "flag"},
    {"to": "--c", "rel": "trades", "effect": "nur curated", "calc": "text", "present": False, "belegt": False, "quelle": "curated", "evidence": None, "satz": "", "edge": "",
     "value": None, "to_kind": "flag"},
    {"to": "HW-COUNT", "rel": "requires", "effect": "Satz zur Ablehnung", "calc": "text", "present": None, "belegt": True, "edge": "K51", "satz": "Satz zur Ablehnung",
     "evidence": {"file": "entrypoint.sh", "zeile": 475, "anchor": "HW-COUNT"}, "value": None, "to_kind": "refusal"},
    {"to": "--d", "rel": "derived_from", "effect": "bedingt", "calc": "text", "present": True, "belegt": True, "edge": "K08", "satz": "bedingter Satz",
     "evidence": {"file": "launcher.py", "zeile": 1, "anchor": "a"}, "value": "auto", "to_kind": "flag", "rel_catalog": "scales_with"},
    {"to": "PROFILE_NIX", "rel": "requires", "effect": "x", "calc": "text", "present": False, "belegt": True, "edge": "K99", "satz": "x", "evidence": None, "value": None, "to_kind": "unbekannt"},
]
REGISTER = [{"code": "HW-COUNT", "title": "Kartenzahl ist nicht die bewiesene", "force_scope": "im Docker-Start (Entrypoint) und im Launcher forcebar"}]


@unittest.skipUnless(NODE, "node fehlt")
class ChipsJs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.o = run_js({"deps": DEPS, "register": REGISTER, "goto": "HW-COUNT"})

    def test_nothing_unhandled(self):
        self.assertEqual(self.o["unhandled"], [])

    def test_chip_shows_beleg_marker_and_sentence_in_the_title(self):
        h = self.o["chips"]
        self.assertIn("Tradeoff-Satz K01", h)
        self.assertIn("Evidence: python/flliper/srt/pdflip/launcher.py:8617", h)
        self.assertIn("must be given together", h)
        self.assertIn("pf-dep-b", h)

    def test_unbelegte_edge_is_marked_visibly(self):
        self.assertIn('<i class="pf-dep-nb">unverified</i>', self.o["chips"])
        self.assertIn("Unverified: curated only", self.o["chips"])

    def test_the_stale_s4_sentence_is_gone_and_replaced_by_the_truth(self):
        h = self.o["chips"] + self.o["open"]
        self.assertNotIn("comes with stage S4", h)
        self.assertIn("computes the consequence in MiB/tokens/ms in the card bars", h)

    def test_conditional_and_diverging_relation_are_text_only(self):
        h = self.o["chips"]
        self.assertIn("only for auto", h)
        self.assertIn("Applies only for value: auto", h)
        self.assertIn("not evaluated", h)
        self.assertIn("relation as “scales with”", h)

    def test_refusal_code_chip_points_at_the_register_not_into_the_void(self):
        h = self.o["chips"]
        self.assertIn("pf-dep-rej", h)
        self.assertIn("Refusal code of the planner, not a value in the profile: Kartenzahl ist nicht die bewiesene", h)
        self.assertNotIn("HW-COUNT</b></span>" + " Not set in this profile", h)
        self.assertIn("a refusal code of the planner", self.o["gotoMsg"])                   # Klick: Meldung, kein stilles Nichts
        self.assertIn("The dry run shows whether it applies here", self.o["gotoMsg"])

    def test_unknown_target_says_so(self):
        o = run_js({"deps": DEPS, "register": REGISTER, "goto": "PROFILE_NIX"})
        self.assertIn("neither a catalog entry nor a refusal code", o["gotoMsg"])
        o2 = run_js({"deps": DEPS, "register": REGISTER, "goto": "--c"})
        self.assertIn("is not set in this profile", o2["gotoMsg"])

    def test_open_row_lists_all_edges_with_sentence_and_evidence_without_hover(self):
        h = self.o["open"]
        self.assertIn("pf-depl", h)
        self.assertIn("Dependencies", h)
        self.assertIn("Tradeoff-Satz K01", h)
        self.assertIn("edge K01", h)

    def test_old_catalog_without_edge_fields_still_draws(self):
        old = [{"to": "--b", "rel": "requires", "effect": "alt", "calc": "text", "present": True}]
        o = run_js({"deps": old, "register": []})
        self.assertEqual(o["unhandled"], [])
        self.assertIn("alt", o["chips"])
        self.assertNotIn("unverified", o["chips"])                                        # no information is not "ohne Beleg"


def use_region(html):
    """Nur der Export-Hinweis (die Seite nennt FLLIPER_FORCE=1 auch im allgemeinen Kopfhinweis)."""
    return html[html.index('class="pf-use"'):html.index('id="pf-env"')]


def export_for(codes, line="27b"):
    """Antwort des Servers fuer einen Export nach einem Trockenlauf mit diesen Codes (der echte Rechenweg)."""
    tmp = tempfile.mkdtemp(prefix="pf2002j_")
    try:
        ed, _r, _u = editor(tmp)
        reg = ed.register()
        dry = None if codes is None else {"ok": True, "rejections": [{"code": c, "text": "Originaltext " + c} for c in codes]}
        h = P.force_hint(dry, reg, line)
        return dry, {"ok": True, "env": "PROFILE_NAME=p\n", "filename": "p.env", "verified": True, "problems": [], "check": "gleich.",
                     "use": {"profile_env": "FLLIPER_PROFILE=p", "force_env": h["force_env"], "force": h, "docker_run": P.docker_run_example("p", h), "text": "The dashboard starts nothing."}}
    finally:
        shutil.rmtree(tmp, True)


@unittest.skipUnless(NODE, "node fehlt")
class ExportJs(unittest.TestCase):
    def show(self, codes):
        dry, exp = export_for(codes)
        case = {"deps": [], "register": [], "dry": dry or {"ok": True, "rejections": [], "goes": True, "verdict": "", "notes": [], "force_note": "", "cards": []}, "export": exp}
        if dry is not None:
            case["dry"].update({"goes": not dry["rejections"], "verdict": "v", "notes": [], "force_note": "fn", "cards": []})
            for r in case["dry"]["rejections"]:
                r.update({"klass_label": "", "force": "", "force_state": "force"})
        o = run_js(case)
        self.assertEqual(o["unhandled"], [])
        return o

    def test_the_export_request_carries_the_last_dry_run(self):
        o = self.show(["PROFIL-STATUS"])
        self.assertEqual([q["code"] for q in o["posted"]["dry"]["rejections"]], ["PROFIL-STATUS"])

    def test_case_force_line_shown_in_the_block_and_nothing_starts(self):
        h = use_region(self.show(["PROFIL-STATUS", "SHM"])["exported"])
        self.assertIn("  -e FLLIPER_FORCE=1 \\", h)
        self.assertIn('data-fall="nur_forcebar"', h)
        self.assertIn("The dashboard starts nothing", h)
        self.assertIn("The server start overrides with", h)
        self.assertNotIn('data-act="start"', h)
        self.assertNotIn("type=\"checkbox\" data-force", h)                                   # no switch

    def test_case_no_rejection_has_no_force_line(self):
        h = use_region(self.show([])["exported"])
        self.assertNotIn("FLLIPER_FORCE=1", h)
        self.assertIn("force is not needed", h)

    def test_case_mixed_has_the_line_and_red_plain_text_for_the_blocked_code(self):
        h = use_region(self.show(["PROFIL-STATUS", "HW-ARCH"])["exported"])
        self.assertIn("  -e FLLIPER_FORCE=1 \\", h)
        self.assertIn('<li class="pf-red"><b class="mono">HW-ARCH</b> remains even with force: Originaltext HW-ARCH; the server does not start like this.', h)

    def test_case_only_hard_has_no_line_and_a_red_sentence(self):
        h = use_region(self.show(["HW-TOPOLOGY"])["exported"])
        self.assertNotIn("FLLIPER_FORCE=1", h)
        self.assertIn('<div class="pf-red"><b>Force does not help here', h)
        self.assertIn('class="pf-red"><b class="mono">HW-TOPOLOGY</b>', h)

    def test_no_dry_run_says_so_and_shows_no_force_line(self):
        dry, exp = export_for(None)
        o = run_js({"deps": [], "register": [], "dry": {"ok": True, "rejections": [], "goes": True, "verdict": "", "notes": [], "force_note": "", "cards": []}, "export": exp})
        self.assertNotIn("FLLIPER_FORCE=1", use_region(o["exported"]))


class SourcePins(unittest.TestCase):
    def test_no_switch_no_start_route_the_tooltip_text_is_gone(self):
        js = open(os.path.join(STATIC, "profil.js"), encoding="utf-8").read()
        self.assertNotIn("comes with stage S4", js)
        for needle in ('data-act="start"', "api(\"start\"", "api(\"force\""):
            self.assertNotIn(needle, js)


if __name__ == "__main__":
    unittest.main()
