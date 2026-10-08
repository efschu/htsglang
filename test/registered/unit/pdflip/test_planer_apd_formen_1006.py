"""AP-D Fix-Runde 2: die Worker-Seite von ``POST /api/profil/propose`` fuer ALLE VIER Formen (Plan PLAN-PROFIL-PLANER-1006 Abschnitt 3 Zeilen AP-D/AP-E/AP-F).

* ``flip`` / ``tp`` / ``dual``: ``propose_verdict.run_propose`` fragt Planer (``propose``) + Oracle (Launcher-Trockenlauf).  Dual traegt die Dual-Passung
  (``DUAL-PASSUNG`` / ``DUAL-PFLICHT``) als "planner calculation, not hw_fit" im Verdikt-Dokument (Oracle-Dry-Run wie Flip; Test in ``test_planer_ape_dual_1006``).
* ``single`` (SingleCard): ``run_propose_single`` -- KEIN Launcher-Lauf (``topology.py`` MIN_CARDS=2), Verdikt = Planer-Rechnung aus ``propose_single``
  plus ServerArgs-Parse; Ausgabe im selben Format wie die uebrigen Formen (Werteliste, ``flliper.verdict/1``, ``per_value``, ``launch``).

Die Zahlen der SingleCard (5090, Qwen3.8-27B NVFP4: ``--mem-fraction-static`` 0.874, KV fp8, Reserve 4050 MiB) sind die des Tests ``test_planer_apf_einzelkarte_1006``
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
    from flliper.srt.pdflip import propose_single as PS
    from flliper.srt.pdflip import propose_verdict as PV
except Exception as exc:  # pragma: no cover - no pdflip launcher in this build
    pytest.skip(f"pdflip launcher unavailable: {exc}", allow_module_level=True)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
FX = os.path.join(HERE, "fixtures", "planer_apf_1006")
CARD_5090_HW = {"schema": "flliper.hardware/1", "cards": [
    {"ord": 0, "name": "NVIDIA GeForce RTX 5090", "vram_total_mib": {"v": 32607, "src": "NVML"}},
    {"ord": 1, "name": "NVIDIA GeForce RTX 3080", "vram_total_mib": {"v": 20480, "src": "NVML"}}]}


def req(model: str = "q27_nvfp4", **kw):
    d = {"form": "single", "model_path": os.path.join(FX, model), "inventory": {"hardware": CARD_5090_HW, "karte": 0}, "goals": {}, "parse": False}
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
        v = r["proposal"]
        self.assertEqual((v["schema"], v["form"], v["n"]), ("flliper.propose-a/1", "single", 1))
        for k in ("values", "cards", "inventory", "seeds", "fit", "goals", "unverified", "notes", "blocker", "vector_lengths", "vectors_ok", "vectors_wrong", "basis"):
            self.assertIn(k, v)
        for w in v["values"]:
            for k in ("key", "group", "policy", "alt", "value", "entries", "state", "source", "reason", "in_argv", "changed"):
                self.assertIn(k, w, w["key"])
        self.assertEqual(set(r["per_value"]), {w["key"] for w in v["values"]})
        self.assertEqual(r["launch"]["env"], {})
        self.assertIn("--model-path", r["launch"]["argv"])

    def test_the_fit_is_a_planer_calculation_with_the_apf_numbers(self):
        v = self.res["proposal"]
        w = {x["key"]: x for x in v["values"]}
        self.assertEqual(w["--mem-fraction-static"]["value"], "0.874")           # AP-F test_nvfp4_passt_mit_zahlen
        self.assertEqual(w["--kv-cache-dtype"]["value"], "fp8_e4m3")
        self.assertEqual(w["--no-enable-multimodal"]["value"], "")               # a switch is the empty value
        self.assertEqual(w["--context-length"]["value"], "131072")
        self.assertEqual(w["--speculative-algorithm"]["value"], "NEXTN")
        e = v["single_card"]
        self.assertEqual(e["fit"]["reserve_bedarf_mib"], 4050.0)
        self.assertAlmostEqual(e["fit"]["statisch_budget_mib"], 0.874 * (32607 - PS.CONTEXT_OVERHEAD_DEFAULT_MIB), delta=0.1)
        self.assertEqual(v["fit"]["level"], "ja")
        self.assertEqual(v["fit"]["art"], "Planner calculation")
        self.assertEqual(v["cards"], [{"name": "NVIDIA GeForce RTX 5090", "total_mib": 32607, "tflops_src": None}])

    def test_the_verdict_says_planer_rechnung_and_has_no_launcher_run(self):
        d = self.res["verdict"]
        self.assertEqual((d["schema"], d["n"], d["form"], d["outcome"], d["art"]), ("flliper.verdict/1", 1, "single", "passt", "Planner calculation"))
        self.assertTrue(d["geht"])
        self.assertIsNone(d["run"])
        self.assertEqual((d["oracle"]["runs"], d["forced"]), (0, []))
        self.assertEqual(len(d["profil"]["proposal_sha256"]), 64)
        codes = {x["code"]: x for x in d["verdikte"]}
        self.assertIn("EINZEL-PASSUNG", codes)
        self.assertEqual((codes["EINZEL-PASSUNG"]["force_state"], codes["EINZEL-PASSUNG"]["etikett"]), ("geht", "Planner calculation"))
        self.assertIn("planner calculation", codes["EINZEL-PASSUNG"]["title"])
        self.assertIn("EINZEL-PARSE", codes)                                    # parse=False in this request: stated as not checked, never as "ok"
        self.assertEqual(codes["EINZEL-PARSE"]["force_state"], "note")
        self.assertIn("not checked", codes["EINZEL-PARSE"]["text"])
        # there is no Force for one card: nothing here is a register code
        self.assertTrue(all(x["forcebar"] is None for x in d["verdikte"]), [(x["code"], x["forcebar"]) for x in d["verdikte"]])
        self.assertEqual(d["zaehlung"], {"force": 0, "is_blocked": 0, "ungeprueft": 0})

    def test_a_single_card_only_with_a_named_model(self):
        r = PV.run_propose(req(model_path=os.path.join(FX, "gibt-es-nicht")), tree="/nonexistent")
        self.assertFalse(r["ok"])
        self.assertIn("no model profile", r["error"])


class TestSingleVerdicts(unittest.TestCase):
    def test_an_int8_27b_does_not_fit_and_the_failing_check_is_a_blocked_verdict(self):
        r = PV.run_propose(req("q27_int8"), tree="/nonexistent")
        self.assertTrue(r["ok"], r)
        d = r["verdict"]
        self.assertEqual((d["outcome"], d["geht"], d["ok_with_force"]), ("does_not_fit", False, False))
        codes = {x["code"]: x for x in d["verdikte"]}
        self.assertEqual(codes["EINZEL-PASSUNG"]["force_state"], "is_blocked")
        self.assertEqual(codes["FIT-STATIC"]["force_state"], "is_blocked")
        self.assertGreaterEqual(d["zaehlung"]["is_blocked"], 2)
        self.assertEqual(r["proposal"]["fit"]["level"], "nein")
        # the failing check hangs on the values it concerns
        kinds = {k: [x["code"] for x in v] for k, v in r["per_value"].items()}
        self.assertIn("FIT-STATIC", kinds["--mem-fraction-static"])

    def test_goals_of_the_page_are_mapped_and_the_rest_is_said(self):
        r = PV.run_propose(req(goals={"seats": 2, "kv_tokens": 65536, "kv_dtype": "auto", "p_cut": "pin"}), tree="/nonexistent")
        self.assertTrue(r["ok"], r)
        w = {x["key"]: x for x in r["proposal"]["values"]}
        self.assertEqual(w["--max-running-requests"]["value"], "2")
        self.assertEqual(w["--context-length"]["value"], "65536")
        self.assertEqual(w["--kv-cache-dtype"]["value"], "auto")
        self.assertTrue(any("p_cut" in n for n in r["notizen"]), r["notizen"])

    def test_the_card_comes_from_the_ordinal_and_exactly_one_catalog_card_is_allowed(self):
        r = PV.run_propose(req(inventory={"hardware": CARD_5090_HW, "karte": 1}), tree="/nonexistent")
        self.assertTrue(r["ok"], r)
        self.assertEqual(r["verdict"]["inventory"][0], {"index": 0, "name": "NVIDIA GeForce RTX 3080", "total_mib": 20480})
        two = [{"index": i, "name": "x", "total_bytes": 1 << 34, "cc_major": 8, "cc_minor": 6} for i in range(2)]
        r2 = PV.run_propose(req(inventory={"devices": two}), tree="/nonexistent")
        self.assertFalse(r2["ok"])
        self.assertIn("needs exactly one card", r2["error"])

    def test_an_unknown_goal_value_is_an_error_not_a_crash(self):
        r = PV.run_propose(req(goals={"seats": 0}), tree="/nonexistent")
        self.assertFalse(r["ok"])
        self.assertIn("seats", r["error"])

    def test_the_serverargs_parse_is_part_of_the_verdict(self):
        r = PV.run_propose(req(parse=True), tree="/nonexistent")
        self.assertTrue(r["ok"], r)
        par = r["proposal"]["single_card"]["parse"]
        pv = next(x for x in r["verdict"]["verdikte"] if x["code"] == "EINZEL-PARSE")
        if par["available"]:
            self.assertTrue(par["ok"], par)
            self.assertEqual(pv["force_state"], "geht")
            self.assertIn("__post_init__ has not run", pv["text"])
        else:                                                                  # no flliper import in the child: stated, never "ok"
            self.assertEqual(pv["force_state"], "note")


if __name__ == "__main__":
    unittest.main()
