"""AP-D Fix-Runde 2: ``POST /api/profil/propose`` bedient ALLE VIER Formen (Plan PLAN-PROFIL-PLANER-1006 Abschnitt 1.3 A3, Abschnitt 3 Zeilen AP-D/AP-E/AP-F).

Gepinnt:
  * Der Editor nimmt ``flip``, ``tp``, ``dual`` und ``single`` (``einzel`` ist ein Name dafuer); die Seite zeigt fuer keine Form mehr den Platzhalter "Vorschlag ist
    spaeteres Arbeitspaket".
  * ``single`` (Einzelkarte): genau EINE Karte, ein Modellpfad, kein Basisprofil noetig; Startprofil = neues ``flliper.server/1`` aus den Argumenten des normalen
    Servers (Herkunft ``planer``, nie die weg2-Zeilen eines Release-Profils); das Verdikt ist eine Planer-Rechnung, es gibt keinen Launcher-Lauf.
  * Mit dem ECHTEN Kindprozess: Referenz-Dual (``27b-nvfp4-dual``, Referenz-Rig, 3 Karten): Argv und Env des Vorschlags = der Golden des Dual (Diff 0), kein Wert
    geaendert, Verdikt traegt ``DUAL-PASSUNG`` als "planner calculation, not hw_fit", der Launcher-Trockenlauf des Vorschlags geht ohne Force durch.
    Einzelkarte 5090 + 27B NVFP4: Verdikt "passt" mit den Zahlen von AP-F (Bruchteil 0.874, KV fp8, Reserve 4050 MiB); 27B INT8: "passt nicht".
"""

import copy
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)

from rigdash import profil as P  # noqa: E402
from rigdash import profil_oracle as ORA  # noqa: E402
from rigdash import profil_planer as PL  # noqa: E402
from test_profil_orakel_apd_1006 import (  # noqa: E402
    CENSUS_27B, ENV, FakeOracle, PLANER_FIX, REPO_CATALOG, REPO_PY, REPLAY_REF, _doc, _hw_profile, _rows, editor)

REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
APF_FIX = os.path.join(REPO_ROOT, "test", "registered", "unit", "weg2", "fixtures", "planer_apf_1006")
GOLDEN_DUAL = os.path.join(PLANER_FIX, "golden", "launch_27b-nvfp4-dual.json")
MC = "/spinning/llm_stuff/club-3090/models-cache/"
HW1 = {"schema": "flliper.hardware/1", "cards": [{"ord": 0, "name": "NVIDIA GeForce RTX 5090", "vram_total_mib": {"v": 32607, "src": "NVML"}}]}


def _ed_single(tmp, oracle):
    ed = editor(tmp, oracle=oracle, hardware=lambda: {"ok": True, "profile": HW1})
    ed.check_path = lambda p, fld: p
    return ed


def _canned_single():
    werte = [{"key": "--model-path", "group": "-", "policy": "single", "alt": None, "wert": "/m", "eintraege": 1, "zustand": "vorgeschlagen",
              "herkunft": "Modellprofil (Pfad)", "grund": "Modellordner", "in_argv": True, "geaendert": True},
             {"key": "--mem-fraction-static", "group": "-", "policy": "single", "alt": None, "wert": "0.874", "eintraege": 1, "zustand": "unbelegt",
              "herkunft": "Planner calculation", "grund": "Bruchteil", "in_argv": True, "geaendert": True},
             {"key": "--max-running-requests", "group": "-", "policy": "single", "alt": None, "wert": "1", "eintraege": 1, "zustand": "vorgeschlagen",
              "herkunft": "Ziel", "grund": "Sitze", "in_argv": True, "geaendert": True},
             {"key": "--no-enable-multimodal", "group": "-", "policy": "single", "alt": None, "wert": "", "eintraege": 1, "zustand": "vorgeschlagen",
              "herkunft": "Ziel", "grund": "Sichtturm aus", "in_argv": True, "geaendert": True},
             {"key": "--speculative-algorithm", "group": "-", "policy": "single", "alt": None, "wert": None, "eintraege": 1, "zustand": "vorgeschlagen",
              "herkunft": "Planner calculation", "grund": "kein Draft", "in_argv": False, "geaendert": False}]
    return {"vorschlag": {"schema": "flliper.propose-a/1", "form": "einzel", "n": 1, "werte": werte, "ziele": {"seats": 1}, "cards": [{"name": "RTX 5090", "total_mib": 32607}],
                          "inventory": {}, "seeds": {}, "fit": {"level": "ja", "art": "Planner calculation"}, "unbelegt": [], "hinweise": [], "blocker": [],
                          "vektorlaengen": {}, "vektoren_ok": True, "vektoren_falsch": {}, "basis": "(kein Profil)"},
            "verdikt": _doc([], n=1, ausgang="passt", form="einzel", art="Planner calculation"), "je_wert": {},
            "launch": {"argv": ["--model-path", "/m"], "env": {}}}


class SingleRequests(unittest.TestCase):
    """``ProfilEditor.propose`` for the single card with a canned child (no process)."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="apd2_")
        self.orc = FakeOracle(propose=_canned_single())
        self.ed = _ed_single(self.tmp, self.orc)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _body(self, **kw):
        return dict({"form": "single", "basis": {"kind": "release", "name": "demo"}, "inventar": "rig", "model_path": "/m"}, **kw)

    def test_the_editor_serves_all_four_forms_and_the_page_has_no_placeholder(self):
        self.assertEqual(set(P.ProfilEditor.FORMS), {"flip", "tp", "dual", "single"})
        ui = PL.ui_info(P.ProfilEditor.FORMS, self.ed.catalog()["entries"], True)
        self.assertEqual([f["vorschlag"] for f in ui["formen"]], [True] * 4)
        self.assertFalse(any("hinweis" in f for f in ui["formen"]))
        for name in ("profil_planer.py", os.path.join("static", "profil_planer.js")):
            with open(os.path.join(os.path.dirname(HERE), name), encoding="utf-8") as fh:
                txt = fh.read()
            self.assertNotIn("later work package", txt, name)
            self.assertNotIn("own work package (AP-F)", txt, name)
            self.assertNotIn("eigenes Arbeitspaket (AP-E)", txt, name)
        self.assertEqual(P.ProfilEditor.BALKEN_FORM, {"flip": "flip", "tp": "d_only", "dual": "dual", "single": "single"})        # profile_couplings.FORMS

    def test_the_request_to_the_child_names_card_model_and_no_weg2_draft(self):
        r = self.ed.propose(self._body(form="einzel"))                               # "einzel" is a name of single
        self.assertTrue(r["ok"], r)
        kind, req, parts = self.orc.calls[0]
        self.assertEqual((kind, req["form"]), ("propose", "single"))
        self.assertEqual(req["model_path"], "/m")
        self.assertIsNone(req.get("draft_path"))                                       # the profile's PROFILE_DRAFT is a weg2 draft: not the single card's
        self.assertEqual(req["inventar"]["karte"], 0)
        self.assertIn(["karte", 0], parts["inventar"])                                 # another card is another question (cache)
        self.assertEqual(parts["form"], "single")

    def test_the_startprofil_is_a_new_profile_of_the_servers_arguments(self):
        r = self.ed.propose(self._body())
        self.assertTrue(r["ok"], r)
        self.assertEqual((r["form"], r["n"], r["schema"]), ("single", 1, "flliper.propose-d/1"))
        sp = r["startprofil"]
        self.assertTrue(sp["verifiziert"], sp["probleme"])
        rows = {x["key"]: x for x in sp["view"]["rows"]}
        self.assertEqual(rows["flag:--mem-fraction-static"]["value"], "0.874")
        self.assertEqual(rows["flag:--mem-fraction-static"]["origin"], "planer")
        self.assertIn("flag:--no-enable-multimodal", rows)                             # a switch
        self.assertNotIn("flag:--speculative-algorithm", rows)                          # not set: the server's default applies
        for k in rows:                                                                  # no row of a weg2 release profile (demo.env has --p-bs, --pp-stage-ratio ...)
            self.assertNotIn(k, ("flag:--p-bs", "flag:--pp-stage-ratio", "flag:--p-hostgap"))
        self.assertEqual(sp["doc"]["line"], "einzel")
        self.assertEqual(sp["doc"]["meta"]["vorschlag"]["form"], "single")
        self.assertEqual(r["balken"]["form"], "single")
        self.assertEqual(r["nicht_uebernommen"], [])
        by = {w["label"]: w for w in r["werte"]}
        self.assertEqual(by["--mem-fraction-static"]["key"], "flag:--mem-fraction-static")
        self.assertTrue(by["--max-running-requests"]["kanten"], "the catalog gives --max-running-requests its edges")
        self.assertTrue(any("single card" in n and "no weg2 launcher" in n for n in sp["doc"]["meta"]["notes"]))

    def test_without_a_profile_the_model_path_is_needed(self):
        with self.assertRaises(P.ProfilError) as cm:
            self.ed.propose({"form": "single", "inventar": "rig"})
        self.assertIn("give model_path", str(cm.exception))
        r = self.ed.propose({"form": "single", "inventar": "rig", "model_path": "/m"})              # no basis at all is fine for one card
        self.assertTrue(r["ok"], r)
        self.assertEqual(r["basis"]["kind"], "keines")
        self.assertEqual(r["startprofil"]["name"], "einzelkarte-vorschlag")

    def test_exactly_one_card_and_a_card_of_the_profile(self):
        two = [{"card": "rtx3080-20", "pcie": {"gen": 4, "lanes": 8}}] * 2
        with self.assertRaises(P.ProfilError) as cm:
            self.ed.propose(self._body(inventar=two))
        self.assertIn("select exactly one card", str(cm.exception))
        with self.assertRaises(P.ProfilError) as cm:
            self.ed.propose(self._body(karte=3))
        self.assertIn("the hardware profile has 1 cards", str(cm.exception))
        r = self.ed.propose(self._body(inventar=two[:1]))                                         # one catalog card that is not the rig's: a datasheet inventory
        self.assertTrue(r["ok"], r)
        self.assertIn("cards", self.orc.calls[-1][1]["inventar"])

    def test_the_goals_of_a_single_card(self):
        ok = self.ed._ziele({"host_ram_mib": 16384, "draft": "off", "pre_load_free_mib": 32000, "seats": 2}, "single")
        self.assertEqual(ok, {"host_ram_mib": 16384, "draft": "off", "pre_load_free_mib": 32000, "seats": 2})
        with self.assertRaises(P.ProfilError):
            self.ed._ziele({"host_ram_mib": 16384}, "flip")                                       # a goal only the single card knows
        with self.assertRaises(P.ProfilError):
            self.ed._ziele({"draft": "wild"}, "single")

    def test_dual_is_a_form_of_the_editor(self):
        orc = FakeOracle(propose=_canned_single())
        ed = editor(self.tmp + "_d", oracle=orc, hardware=lambda: {"ok": True, "profile": _hw_profile(_rows(3))}) if os.makedirs(self.tmp + "_d") is None else None
        try:
            ans = copy.deepcopy(_canned_single())
            ans["vorschlag"].update(form="dual", n=3)
            orc.propose = ans
            with mock.patch.object(P, "dual_line_probe", return_value=True):         # the fixture tree has no Dual module: this is the 27B line's Dual proposal
                r = ed.propose({"basis": {"kind": "release", "name": "demo"}, "form": "dual", "inventar": "rig"})
        finally:
            shutil.rmtree(self.tmp + "_d", ignore_errors=True)
        self.assertTrue(r["ok"], r)
        self.assertEqual(orc.calls[0][1]["form"], "dual")
        self.assertEqual(r["balken"]["form"], "dual")


try:
    from test_profil_planer_aph1_1006 import NODE, run_node
except Exception:       # noqa: BLE001 -- ohne die Seitentests keine Darstellungsprobe
    NODE, run_node = None, None


@unittest.skipUnless(NODE, "node fehlt")
class Darstellung(unittest.TestCase):
    """Die Seite zeigt das Urteil einer Planer-Rechnung als solches: nicht "Launcher-Trockenlauf", nicht "hw_fit"."""

    def test_the_single_card_proposal_is_labelled_planer_rechnung(self):
        o = run_node("""
out.single = PX.renderProposal({ n: 1, form: "single", werte: [{ label: "--mem-fraction-static", alt: null, wert: "0.874", zustand: "vorgeschlagen", geaendert: true, verdikte: [] }],
  verdikt: { ausgang: "passt", verdikte: [{ ebene: "fit", code: "EINZEL-PASSUNG", grund: "passt: statisch 28000 MiB", force_state: "geht", forcebar: null },
                                          { ebene: "fit", code: "FIT-STATIC", grund: "es fehlen 100 MiB", force_state: "blockiert", forcebar: null }] },
  vorschlag: { cards: [{ name: "NVIDIA GeForce RTX 5090", total_mib: 32607 }], fit: { level: "ja", margin_mib: 12.4, first: "x", art: "Planner calculation" } }, notes: [] });
out.dual = PX.renderProposal({ n: 3, form: "dual", werte: [], verdikt: { ausgang: "geht", verdikte: [{ ebene: "fit", code: "DUAL-PASSUNG", grund: "Dual fit: planner calculation, not hw_fit", force_state: "geht" },
                                                                                                { ebene: "fit", code: "FIT", grund: "hw_fit", force_state: "hinweis" }] },
  vorschlag: { cards: [], fit: { level: "ja", margin_mib: 1, first: "" } }, notes: [] });
out.flip = PX.renderProposal({ n: 3, form: "flip", werte: [], verdikt: { ausgang: "geht", verdikte: [] }, vorschlag: { cards: [], fit: { level: "ja", margin_mib: 1, first: "" } }, notes: [] });
""")
        s = o["single"]
        self.assertIn("Planner estimate: fits (no launcher run", s)
        self.assertNotIn("The launcher dry run passes without force", s)
        self.assertIn("Fit (Planner calculation)", s)
        self.assertIn("Fit as a planner estimate", s)
        self.assertIn("EINZEL-PASSUNG", s)
        self.assertIn("pfx-v-verweigert", s)                                                    # the failing check is shown as failing
        d = o["dual"]
        self.assertIn("Fit as a planner estimate", d)
        self.assertIn("DUAL-PASSUNG", d)
        self.assertNotIn("<b class=\"mono\">FIT</b>", d)                                         # hw_fit's FIT line stays the hw_fit line
        self.assertIn("Fit (hw_fit, necessary condition)", d)
        self.assertNotIn("Fit as a planner estimate", o["flip"])


@unittest.skipUnless(os.path.exists(CENSUS_27B) and os.path.isdir(REPO_PY) and os.path.exists(GOLDEN_DUAL)
                     and os.path.exists(MC + "Qwen3.8-27B-NVFP4-RadixArk/config.json") and os.path.exists(MC + "Qwen3.8-27B-DFlash2-NVFP4-RTNcal/config.json"),
                     "the real oracle needs the rig box (census, the 27B NVFP4 checkpoints) and the planner tree")
class RealForms(unittest.TestCase):
    """The REAL child process (this checkout's sglang) behind ``ProfilEditor.propose`` for the Dual and the single card."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="apd2_real_")
        cls.oracle = ORA.OracleService(REPO_PY, python=sys.executable)
        cls.rel = os.path.join(cls.tmp, "rel")
        os.makedirs(cls.rel)
        for n in ("27b-nvfp4-dual.env", "27b-base.env", "27b-nvfp4.pchunk.json"):          # the Dual profile sources 27b-base.env and names the pchunk file next to it
            shutil.copy(os.path.join(PLANER_FIX, "profiles", n), os.path.join(cls.rel, n))
        cls.hw3 = _hw_profile(_rows(3))
        with open(GOLDEN_DUAL, encoding="utf-8") as fh:
            cls.golden = json.load(fh)

    @classmethod
    def tearDownClass(cls):
        cls.oracle.close()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _ed(self, hw):
        from rigdash import kartenplan as K

        ed = P.ProfilEditor(kartenplaner=K.Kartenplaner(tree=REPO_PY), release_dir=self.rel, user_dir=os.path.join(self.tmp, "usr"), tree=REPO_PY,
                            catalog_file=REPO_CATALOG, oracle=self.oracle, hardware=lambda: {"ok": True, "profile": hw})
        ed.check_path = lambda p, fld: p
        return ed

    # ---------------------------------------------------------------- Dual
    @unittest.skipUnless(os.path.isfile(os.path.join(REPO_PY, "sglang", "srt", "weg2", "dual_layout_plan.py")),
                         "27B launcher line only: the Dual form (weg2/dual_layout_plan.py, dual_green.py) does not exist in this tree (measured 07.10. on the NF tree "
                         "2e68b3f94b: ImportError cannot import name 'dual_layout_plan' from 'sglang.srt.weg2')")
    def test_1_reference_dual_is_the_golden_with_zero_diff(self):
        ed = self._ed(self.hw3)
        r = ed.propose({"basis": {"kind": "release", "name": "27b-nvfp4-dual"}, "form": "dual", "inventar": "rig"})
        self.assertTrue(r["ok"], r)
        self.assertEqual((r["schema"], r["form"], r["n"]), ("flliper.propose-d/1", "dual", 3))
        # A1 (Plan 1.3): the proposal for the profile's own inventory IS the profile: argv and env equal the Dual golden of AP0, no value changed
        # (the golden was dumped on the image layout: the census sits under /opt/htsglang/profiles/27b there and under /spinning/gpu-arb/weg2/census on
        # this box; propose_oracle moves exactly that path -- the only difference allowed)
        # and the profile's own directory (the fixture copy of the pchunk file sits next to the profile in this test's release dir)
        def norm(t):
            t = str(t).replace("/opt/htsglang/profiles/27b/", "/spinning/gpu-arb/weg2/census/")
            return os.path.join(self.rel, "27b-nvfp4.pchunk.json") if os.path.basename(t) == "27b-nvfp4.pchunk.json" else t

        self.assertEqual(r["launch"]["argv"], [norm(t) for t in self.golden["argv"]])
        self.assertEqual(r["launch"]["env"], {k: norm(v) for k, v in self.golden["env"].items()})
        # (the only "changed" row is the P-cut seed: a planner-only value, written to meta.planner and never to a row of the profile)
        self.assertEqual([w["label"] for w in r["werte"] if w["geaendert"]], ["--pp-stage-ratio (Seed)"])
        self.assertIsNone(ed.doc_key("--pp-stage-ratio (Seed)"))
        base_rows = {x["key"]: x["value"] for x in ed.render_view(ed.load("release", "27b-nvfp4-dual")["doc"])["view"]["rows"]}
        new_rows = {x["key"]: x["value"] for x in r["startprofil"]["view"]["rows"]}
        self.assertEqual(new_rows, base_rows)                                                         # the Startprofil equals the profile, row for row
        self.assertEqual(r["vorschlag"]["dual"]["modus"], "profil", r["vorschlag"]["dual"].get("regeln"))
        self.assertTrue(r["vorschlag"]["dual"]["regeln"]["uebernommen"], r["vorschlag"]["dual"]["regeln"])
        self.assertTrue(r["startprofil"]["verifiziert"], r["startprofil"]["probleme"])
        self.assertEqual(r["balken"]["form"], "dual")
        # the verdicts: the launcher dry run of the proposal (as for the Flip) and the Dual coupling as a Planer-Rechnung, never as hw_fit
        vd = r["verdikt"]
        self.assertEqual(vd["ausgang"], "geht", [(x["code"], x["grund"][:120]) for x in vd["verdikte"]])
        self.assertEqual((vd["orakel"]["laeufe"], vd["forced"]), (1, []))
        codes = {x["code"]: x for x in vd["verdikte"]}
        self.assertIn("DUAL-PASSUNG", codes)
        self.assertIn("DUAL-PFLICHT", codes)
        self.assertEqual(codes["DUAL-PASSUNG"]["titel"], "Dual fit: planner calculation, not hw_fit")
        self.assertIn("planner calculation, not hw_fit", codes["DUAL-PASSUNG"]["text"])
        self.assertEqual(codes["DUAL-PASSUNG"]["ebene"], "fit")
        self.assertIn("dual not modelled", codes["FIT"]["text"])
        self.assertEqual(codes["FIT"]["force_state"], "hinweis")                         # hw_fit does not clear or block the Dual
        for w in r["werte"]:
            for f in ("key", "label", "wert", "zustand", "herkunft", "grund", "verdikte", "kanten"):
                self.assertIn(f, w)
        # the second equal question is a cache hit
        r2 = ed.propose({"basis": {"kind": "release", "name": "27b-nvfp4-dual"}, "form": "dual", "inventar": "rig"})
        self.assertTrue(r2["orakel"]["cached"])

    # ---------------------------------------------------------------- Einzelkarte
    def _single(self, model="q27_nvfp4", hw=None, **kw):
        hw = hw or self.hw3
        names = [c["name"] for c in hw["cards"]]
        karte = next(i for i, n in enumerate(names) if "5090" in n)                        # resolved by name, never a fixed index (project rule)
        ed = self._ed(hw)
        return ed, ed.propose(dict({"form": "single", "inventar": "rig", "karte": karte, "model_path": os.path.join(APF_FIX, model)}, **kw)), karte

    def test_2_single_5090_with_27b_nvfp4_fits_with_the_apf_numbers(self):
        ed, r, karte = self._single()
        self.assertTrue(r["ok"], r)
        self.assertEqual((r["form"], r["n"]), ("single", 1))
        vd = r["verdikt"]
        self.assertEqual((vd["ausgang"], vd["orakel"]["laeufe"]), ("passt", 0))
        self.assertTrue(vd["art"].startswith("Planner calculation"), vd["art"])                    # "+ ServerArgs-Parse" when the parse ran
        self.assertIsNone(vd["lauf"])
        by = {w["label"]: w for w in r["werte"]}
        # AP-F (test_nvfp4_passt_mit_zahlen, Handrechnung dort): fraction = floor3((32607-400-4050)/(32607-400)) = 0.874, KV fp8, MTP-Draft NEXTN
        self.assertEqual(by["--mem-fraction-static"]["wert"], "0.874")
        self.assertEqual(by["--kv-cache-dtype"]["wert"], "fp8_e4m3")
        self.assertEqual(by["--speculative-algorithm"]["wert"], "NEXTN")
        self.assertEqual(by["--context-length"]["wert"], "131072")
        fit = r["vorschlag"]["einzelkarte"]["fit"]
        self.assertEqual(fit["reserve_bedarf_mib"], 4050.0)
        self.assertAlmostEqual(fit["statisch_budget_mib"], 0.874 * 32207, delta=0.1)
        self.assertTrue(fit["passt"])
        codes = {x["code"]: x for x in vd["verdikte"]}
        self.assertEqual(codes["EINZEL-PASSUNG"]["force_state"], "geht")
        self.assertIn("EINZEL-PARSE", codes)
        self.assertIn(codes["EINZEL-PARSE"]["force_state"], ("geht", "hinweis"))              # parse ran, or says it could not: never a silent ok
        self.assertNotIn("blockiert", {x["force_state"] for x in vd["verdikte"]})
        # the Startprofil: a new profile of the server's arguments, origin planer, verified by bash
        sp = r["startprofil"]
        self.assertTrue(sp["verifiziert"], sp["probleme"])
        rows = {x["key"]: x for x in sp["view"]["rows"]}
        self.assertEqual(rows["flag:--mem-fraction-static"]["value"], "0.874")
        self.assertEqual(rows["flag:--mem-fraction-static"]["origin"], "planer")
        self.assertEqual(rows["flag:--model-path"]["value"], os.path.join(APF_FIX, "q27_nvfp4"))
        self.assertEqual(r["launch"]["argv"][:2], ["--model-path", os.path.join(APF_FIX, "q27_nvfp4")])
        self.assertEqual(r["launch"]["env"], {})
        # Herkunft, Verdikt und Kanten je Wert
        for w in r["werte"]:
            for f in ("key", "label", "wert", "zustand", "herkunft", "grund", "verdikte", "kanten"):
                self.assertIn(f, w)
        self.assertTrue(by["--mem-fraction-static"]["herkunft"] and by["--mem-fraction-static"]["grund"])
        self.assertTrue(by["--max-running-requests"]["kanten"], "the catalog gives --max-running-requests its edges")
        # cached: the second equal question
        r2 = ed.propose({"form": "single", "inventar": "rig", "karte": karte, "model_path": os.path.join(APF_FIX, "q27_nvfp4")})
        self.assertTrue(r2["orakel"]["cached"])
        self.assertEqual(r2["launch"]["argv"], r["launch"]["argv"])

    def test_3_single_27b_int8_does_not_fit_on_the_5090(self):
        _ed, r, _k = self._single("q27_int8")
        self.assertTrue(r["ok"], r)
        vd = r["verdikt"]
        self.assertEqual((vd["ausgang"], vd["geht"]), ("passt_nicht", False))
        codes = {x["code"]: x["force_state"] for x in vd["verdikte"]}
        self.assertEqual(codes["EINZEL-PASSUNG"], "blockiert")
        self.assertEqual(codes["FIT-STATIC"], "blockiert")
        by = {w["label"]: w for w in r["werte"]}
        self.assertIn("FIT-STATIC", [x["code"] for x in by["--mem-fraction-static"]["verdikte"]])

    def test_4_single_card_ordinal_is_another_question_and_another_card(self):
        names = [c["name"] for c in self.hw3["cards"]]
        k3080 = next(i for i, n in enumerate(names) if "3080" in n)
        ed = self._ed(self.hw3)
        r = ed.propose({"form": "single", "inventar": "rig", "karte": k3080, "model_path": os.path.join(APF_FIX, "q27_nvfp4")})
        self.assertTrue(r["ok"], r)
        self.assertEqual(r["vorschlag"]["cards"][0]["name"], names[k3080])
        self.assertEqual(r["vorschlag"]["cards"][0]["total_mib"], self.hw3["cards"][k3080]["vram_total_mib"]["v"])


if __name__ == "__main__":
    unittest.main()
