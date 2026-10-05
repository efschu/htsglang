"""KV-Köpfe je Rang (Stufe 1a, Nutzerentscheid 05.10.): reine Anzeige aus --rank-tp-ratio und der Kopfzahl des Modells.

Gepinnt:
  * Die zwei Regime der Laufzeit (distributed/utils.py): KV-Köpfe < Ränge = REPLIZIERT; KV-Köpfe = Ränge ist ausdrücklich NICHT repliziert
    (Z.1895-1914); KV-Köpfe > Ränge = Largest-Remainder. Zahlenfälle: 27B (4 KV-Köpfe) TP3 -> 2/1/1, TP4 -> 1/1/1/1, TP5 -> repliziert;
    NF (2 KV-Köpfe, Form A, 1,0,0) -> repliziert, Q [24,0,0] (Boot-Log D.log:503 vom 05.10.).
  * Die nachgebildeten Rechenregeln weichen nicht von der Laufzeit ab: die Funktionen werden aus utils.py per AST gezogen (ohne torch zu
    importieren) und gegen ``split_units`` verglichen.
  * Nichts wird geraten: Kopfzahl unbekannt, Gewichte "auto" oder eine Shell-Variable ergeben "nicht gerechnet", keine Zahl.
"""

import ast
import json
import os
import random
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from rigdash import kvheads as KV  # noqa: E402

UTILS = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "python", "sglang", "srt", "distributed", "utils.py"))


def _runtime_functions():
    """``_partition_units_raw`` und ``_partition_units_with_empty_ranks`` aus utils.py, ohne das Modul (torch) zu importieren."""
    with open(UTILS, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    want = {"_partition_units_raw", "_partition_units_with_empty_ranks"}
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in want]
    ns = {"Optional": __import__("typing").Optional, "Sequence": __import__("typing").Sequence}
    exec(compile(ast.Module(body=body, type_ignores=[]), UTILS, "exec"), ns)
    return ns["_partition_units_raw"], ns["_partition_units_with_empty_ranks"]


class Regime(unittest.TestCase):
    def test_27b_tp3_distributed_2_1_1(self):
        r = KV.compute(tp_size=3, ratios=[1, 1, 1], kv_heads=4)
        self.assertEqual((r["regime"], r["kv"]), ("verteilt", [2, 1, 1]))

    def test_27b_tp4_is_not_replicated(self):
        r = KV.compute(tp_size=4, ratios=[1, 1, 1, 1], kv_heads=4)
        self.assertEqual((r["regime"], r["kv"]), ("verteilt", [1, 1, 1, 1]))
        self.assertTrue(any("nicht repliziert" in n for n in r["notiz"]))

    def test_27b_tp5_replicated(self):
        r = KV.compute(tp_size=5, ratios=[1, 1, 1, 1, 1], kv_heads=4)
        self.assertEqual((r["regime"], r["kv"]), ("repliziert", [4] * 5))

    def test_nf_form_a_replicated_q_all_on_host(self):
        r = KV.compute(tp_size=3, ratios=[1, 0, 0], kv_heads=2, q_heads=24)
        self.assertEqual((r["regime"], r["kv"], r["q"]), ("repliziert", [2, 2, 2], [24, 0, 0]))

    def test_form_a_zero_weights_own_no_heads_when_kv_covers_ranks(self):
        r = KV.compute(tp_size=3, ratios=[1, 0, 0], kv_heads=4)
        self.assertEqual(r["kv"], [4, 0, 0])

class NichtGeraten(unittest.TestCase):
    def test_unknown_heads(self):
        r = KV.compute(tp_size=3, ratios=[1, 1, 1], kv_heads=None)
        self.assertEqual((r["status"], r["kv"]), ("unbekannt", None))

    def test_auto_and_unreadable_ratios_give_no_number(self):
        self.assertIsNone(KV.compute(tp_size=3, ratios="auto", kv_heads=4)["kv"])
        self.assertIsNone(KV.parse_ratios("$NF_RATIO"))
        out = KV.view([{"name": "--rank-tp-ratio", "scope": "D", "value": "$NF_RATIO"}], "")
        self.assertEqual(out[0]["status"], "unbekannt")
        self.assertIsNone(out[0]["kv"])

    def test_no_plan_even_split_only_when_divisible(self):
        self.assertEqual(KV.compute(tp_size=2, ratios=None, kv_heads=4)["kv"], [2, 2])
        self.assertIsNone(KV.compute(tp_size=3, ratios=None, kv_heads=4)["kv"])


class GegenLaufzeit(unittest.TestCase):
    """Drift-Wächter: gleiche Zahlen wie die Laufzeitfunktionen in utils.py (AST-Auszug, ohne torch)."""

    def test_split_units_equals_runtime(self):
        raw, empty = _runtime_functions()
        rng = random.Random(1005)
        for _ in range(400):
            n = rng.randint(1, 6)
            units = rng.randint(n, 40)
            w = [rng.randint(0, 9) for _ in range(n)]
            if not any(w):
                continue
            kept = sum(1 for x in w if x > 0)
            if units < kept:
                continue
            want = empty(units, w, None) if 0 in w else raw(units, w)
            self.assertEqual(KV.split_units(units, w), want, (units, w))


class RenameFest(unittest.TestCase):
    def test_json_keys_the_page_reads_are_not_keyword_arguments(self):
        """Das Rename-Kit (rename_rigdash.py) benennt BEZEICHNER um (``quelle`` -> ``source``), String-Schlüssel nicht. ``profil.js`` liest ``r.quelle``;
        stünde ``quelle=`` als Schlüsselwort-Argument in kvheads.py, hieße der JSON-Schlüssel im umbenannten Editor ``source`` und die Seite zeigte
        "(undefined)" (gefunden bei der Rename-Probe 05.10.)."""
        with open(os.path.join(os.path.dirname(HERE), "kvheads.py"), encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        page_keys = {"quelle", "regime", "kv", "q", "satz", "notiz", "belege", "heads", "ratios", "group"}
        bad = sorted({k.arg for n in ast.walk(tree) if isinstance(n, ast.Call) for k in n.keywords if k.arg in {"quelle"}})
        self.assertEqual(bad, [], "Schlüsselwort-Argumente, die das Rename-Kit umbenennt: %s (Seitenschlüssel: %s)" % (bad, sorted(page_keys)))


class Ansicht(unittest.TestCase):
    def test_view_reads_config_and_row(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as fh:
                json.dump({"text_config": {"num_attention_heads": 24, "num_key_value_heads": 2, "head_dim": 256, "num_hidden_layers": 48}}, fh)
            out = KV.view([{"name": "--rank-tp-ratio", "scope": "D", "value": "1,0,0"}, {"name": "--p-bs", "scope": "launcher", "value": "6"}], d)
        self.assertEqual(len(out), 1)
        self.assertEqual((out[0]["group"], out[0]["regime"], out[0]["kv"], out[0]["q"]), ("D", "repliziert", [2, 2, 2], [24, 0, 0]))
        self.assertEqual(out[0]["heads"], {"q": 24, "kv": 2, "head_dim": 256, "layers": 48})

    def test_planner_solved_ratio_is_shown_when_the_profile_sets_none(self):
        """27B-Profile setzen kein --rank-tp-ratio, der Planer rechnet es (planner_only, Schlüssel extra:D:...): ohne diesen Weg bliebe die Zeile leer."""
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as fh:
                json.dump({"text_config": {"num_attention_heads": 24, "num_key_value_heads": 4}}, fh)
            po = [{"key": "extra:D:--rank-tp-ratio", "value": "58,25,25"}, {"key": "extra:P:--pp-stage-ratio", "value": "45,10,9"}]
            out = KV.view([{"name": "--pp-stage-ratio", "scope": "P", "value": "45,10,9"}], d, po)
            self.assertEqual(len(out), 1)
            self.assertEqual((out[0]["group"], out[0]["quelle"], out[0]["regime"], out[0]["kv"]), ("D", "Planer", "verteilt", [2, 1, 1]))
            # ein im Profil gesetzter Wert hat Vorrang vor dem Planerwert derselben Gruppe
            both = KV.view([{"name": "--rank-tp-ratio", "scope": "D", "value": "1,1,1"}], d, po)
            self.assertEqual((len(both), both[0]["quelle"], both[0]["ratios"]), (1, "Profil", "1,1,1"))

    def test_unreadable_model_says_so(self):
        out = KV.view([{"name": "--rank-tp-ratio", "scope": "D", "value": "1,1,1"}], "/nonexistent")
        self.assertEqual((out[0]["status"], out[0]["kv"]), ("unbekannt", None))
        self.assertIn("nicht lesbar", out[0]["satz"])


if __name__ == "__main__":
    unittest.main()
