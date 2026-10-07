"""AP-D Fix-Runde 2: die Worker-Seite von ``POST /api/profil/propose`` fuer ALLE VIER Formen (Plan PLAN-PROFIL-PLANER-1006 Abschnitt 3 Zeilen AP-D/AP-E/AP-F).

* ``flip`` / ``tp`` / ``dual``: ``propose_verdict.run_propose`` fragt Planer (``propose``) + Orakel (Launcher-Trockenlauf).  Dual traegt die Dual-Passung
  (``DUAL-PASSUNG`` / ``DUAL-PFLICHT``) als "Planer-Rechnung, nicht hw_fit" im Verdikt-Dokument (Orakel-Dry-Run wie Flip; Test in ``test_planer_ape_dual_1006``).
* ``single`` (Einzelkarte): ``run_propose_single`` -- KEIN Launcher-Lauf (``topology.py`` MIN_CARDS=2), Verdikt = Planer-Rechnung aus ``propose_single``
  plus ServerArgs-Parse; Ausgabe im selben Format wie die uebrigen Formen (Werteliste, ``flliper.verdikt/1``, ``je_wert``, ``launch``).

Die Zahlen der Einzelkarte (5090, Qwen3.8-27B NVFP4: ``--mem-fraction-static`` 0.874, KV fp8, Reserve 4050 MiB) sind die des Tests ``test_planer_apf_einzelkarte_1006``
(Handrechnung dort); hier wird nur geprueft, dass sie UNVERAENDERT im Worker-Dokument ankommen.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

try:
    from sglang.srt.weg2 import propose_single as PS
    from sglang.srt.weg2 import propose_verdict as PV
except Exception as exc:  # pragma: no cover - no weg2 launcher in this build
    pytest.skip(f"weg2 launcher unavailable: {exc}", allow_module_level=True)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
FX = os.path.join(HERE, "fixtures", "planer_apf_1006")
CARD_5090_HW = {"schema": "flliper.hardware/1", "cards": [
    {"ord": 0, "name": "NVIDIA GeForce RTX 5090", "vram_total_mib": {"v": 32607, "src": "NVML"}},
    {"ord": 1, "name": "NVIDIA GeForce RTX 3080", "vram_total_mib": {"v": 20480, "src": "NVML"}}]}


def req(model: str = "q27_nvfp4", **kw):
    d = {"form": "single", "model_path": os.path.join(FX, model), "inventar": {"hardware": CARD_5090_HW, "karte": 0}, "ziele": {}, "parse": False}
    d.update(kw)
    return d


class TestSingleDocument(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.res = PV.run_propose(req(), tree="/nonexistent")          # single never touches the launcher tree

    def test_it_is_the_one_format_of_the_other_forms(self):
        r = self.res
        self.assertTrue(r["ok"], r)
        json.dumps(r)                                                  # the worker answers over a pipe
        self.assertEqual(r["schema"], PV.PROPOSE_SCHEMA)
        v = r["vorschlag"]
        self.assertEqual((v["schema"], v["form"], v["n"]), ("flliper.propose-a/1", "einzel", 1))
        for k in ("werte", "cards", "inventory", "seeds", "fit", "ziele", "unbelegt", "hinweise", "blocker", "vektorlaengen", "vektoren_ok", "vektoren_falsch", "basis"):
            self.assertIn(k, v)
        for w in v["werte"]:
            for k in ("key", "group", "policy", "alt", "wert", "eintraege", "zustand", "herkunft", "grund", "in_argv", "geaendert"):
                self.assertIn(k, w, w["key"])
        self.assertEqual(set(r["je_wert"]), {w["key"] for w in v["werte"]})
        self.assertEqual(r["launch"]["env"], {})
        self.assertIn("--model-path", r["launch"]["argv"])

    def test_the_fit_is_a_planer_calculation_with_the_apf_numbers(self):
        v = self.res["vorschlag"]
        w = {x["key"]: x for x in v["werte"]}
        self.assertEqual(w["--mem-fraction-static"]["wert"], "0.874")           # AP-F test_nvfp4_passt_mit_zahlen
        self.assertEqual(w["--kv-cache-dtype"]["wert"], "fp8_e4m3")
        self.assertEqual(w["--no-enable-multimodal"]["wert"], "")               # a switch is the empty value
        self.assertEqual(w["--context-length"]["wert"], "131072")
        self.assertEqual(w["--speculative-algorithm"]["wert"], "NEXTN")
        e = v["einzelkarte"]
        self.assertEqual(e["fit"]["reserve_bedarf_mib"], 4050.0)
        self.assertAlmostEqual(e["fit"]["statisch_budget_mib"], 0.874 * (32607 - PS.CONTEXT_OVERHEAD_DEFAULT_MIB), delta=0.1)
        self.assertEqual(v["fit"]["level"], "ja")
        self.assertEqual(v["fit"]["art"], "Planer-Rechnung")
        self.assertEqual(v["cards"], [{"name": "NVIDIA GeForce RTX 5090", "total_mib": 32607, "tflops_src": None}])

    def test_the_verdict_says_planer_rechnung_and_has_no_launcher_run(self):
        d = self.res["verdikt"]
        self.assertEqual((d["schema"], d["n"], d["form"], d["ausgang"], d["art"]), ("flliper.verdikt/1", 1, "einzel", "passt", "Planer-Rechnung"))
        self.assertTrue(d["geht"])
        self.assertIsNone(d["lauf"])
        self.assertEqual((d["orakel"]["laeufe"], d["forced"]), (0, []))
        self.assertEqual(len(d["profil"]["vorschlag_sha256"]), 64)
        codes = {x["code"]: x for x in d["verdikte"]}
        self.assertIn("EINZEL-PASSUNG", codes)
        self.assertEqual((codes["EINZEL-PASSUNG"]["force_state"], codes["EINZEL-PASSUNG"]["etikett"]), ("geht", "Planer-Rechnung"))
        self.assertIn("Planer-Rechnung", codes["EINZEL-PASSUNG"]["titel"])
        self.assertIn("EINZEL-PARSE", codes)                                    # parse=False in this request: stated as not checked, never as "ok"
        self.assertEqual(codes["EINZEL-PARSE"]["force_state"], "hinweis")
        self.assertIn("nicht geprueft", codes["EINZEL-PARSE"]["text"])
        # there is no Force for one card: nothing here is a register code
        self.assertTrue(all(x["forcebar"] is None for x in d["verdikte"]), [(x["code"], x["forcebar"]) for x in d["verdikte"]])
        self.assertEqual(d["zaehlung"], {"force": 0, "blockiert": 0, "ungeprueft": 0})

    def test_a_single_card_only_with_a_named_model(self):
        r = PV.run_propose(req(model_path=os.path.join(FX, "gibt-es-nicht")), tree="/nonexistent")
        self.assertFalse(r["ok"])
        self.assertIn("kein Modellprofil", r["error"])


class TestSingleVerdicts(unittest.TestCase):
    def test_an_int8_27b_does_not_fit_and_the_failing_check_is_a_blocked_verdict(self):
        r = PV.run_propose(req("q27_int8"), tree="/nonexistent")
        self.assertTrue(r["ok"], r)
        d = r["verdikt"]
        self.assertEqual((d["ausgang"], d["geht"], d["geht_mit_force"]), ("passt_nicht", False, False))
        codes = {x["code"]: x for x in d["verdikte"]}
        self.assertEqual(codes["EINZEL-PASSUNG"]["force_state"], "blockiert")
        self.assertEqual(codes["FIT-STATIC"]["force_state"], "blockiert")
        self.assertGreaterEqual(d["zaehlung"]["blockiert"], 2)
        self.assertEqual(r["vorschlag"]["fit"]["level"], "nein")
        # the failing check hangs on the values it concerns
        kinds = {k: [x["code"] for x in v] for k, v in r["je_wert"].items()}
        self.assertIn("FIT-STATIC", kinds["--mem-fraction-static"])

    def test_goals_of_the_page_are_mapped_and_the_rest_is_said(self):
        r = PV.run_propose(req(ziele={"seats": 2, "kv_tokens": 65536, "kv_dtype": "auto", "p_cut": "pin"}), tree="/nonexistent")
        self.assertTrue(r["ok"], r)
        w = {x["key"]: x for x in r["vorschlag"]["werte"]}
        self.assertEqual(w["--max-running-requests"]["wert"], "2")
        self.assertEqual(w["--context-length"]["wert"], "65536")
        self.assertEqual(w["--kv-cache-dtype"]["wert"], "auto")
        self.assertTrue(any("p_cut" in n for n in r["notizen"]), r["notizen"])

    def test_the_card_comes_from_the_ordinal_and_exactly_one_catalog_card_is_allowed(self):
        r = PV.run_propose(req(inventar={"hardware": CARD_5090_HW, "karte": 1}), tree="/nonexistent")
        self.assertTrue(r["ok"], r)
        self.assertEqual(r["verdikt"]["inventar"][0], {"index": 0, "name": "NVIDIA GeForce RTX 3080", "total_mib": 20480})
        two = [{"index": i, "name": "x", "total_bytes": 1 << 34, "cc_major": 8, "cc_minor": 6} for i in range(2)]
        r2 = PV.run_propose(req(inventar={"devices": two}), tree="/nonexistent")
        self.assertFalse(r2["ok"])
        self.assertIn("genau eine Karte", r2["error"])

    def test_an_unknown_goal_value_is_an_error_not_a_crash(self):
        r = PV.run_propose(req(ziele={"seats": 0}), tree="/nonexistent")
        self.assertFalse(r["ok"])
        self.assertIn("seats", r["error"])

    def test_the_serverargs_parse_is_part_of_the_verdict(self):
        r = PV.run_propose(req(parse=True), tree="/nonexistent")
        self.assertTrue(r["ok"], r)
        par = r["vorschlag"]["einzelkarte"]["parse"]
        pv = next(x for x in r["verdikt"]["verdikte"] if x["code"] == "EINZEL-PARSE")
        if par["available"]:
            self.assertTrue(par["ok"], par)
            self.assertEqual(pv["force_state"], "geht")
            self.assertIn("__post_init__ nicht gelaufen", pv["text"])
        else:                                                                  # no sglang import in the child: stated, never "ok"
            self.assertEqual(pv["force_state"], "hinweis")


if __name__ == "__main__":
    unittest.main()
