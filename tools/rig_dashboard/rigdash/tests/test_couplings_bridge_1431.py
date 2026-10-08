"""PROFIL-EDITOR S4a (Auftrag 1431): die Verdrahtung der Kopplungs-Engine im Dashboard-Brückenprozess.

Das Dashboard rechnet nichts selbst (kein flliper-Import, MemoryMax=1G): ``kartenplan_build.bridge.couplings`` startet den Kindprozess
``runner.py`` (op ``couplings``) mit dem Python der flliper-Umgebung; der ruft ``flliper.srt.planner.profile_couplings.run``.

Gepinnt:
* gegen einen Planer-Baum OHNE das Modul (der vendorte Fixture-Baum) antwortet die Brücke benannt ``fehlt im Planer-Baum``, kein Absturz,
  kein Traceback in der Oberfläche;
* gegen einen Baum MIT dem Modul (Umgebung ``COUPLINGS_TREE`` = ``<py-Integrationsbaum>/python``) rechnet die Brücke eine Anfrage durch und gibt
  die Auspackung ``{ok, result}`` ohne die Runner-Hülle zurück (läuft nur, wenn der Baum gesetzt ist);
* unbekannte Operationen und unrechenbare Eingaben kommen als ``ok: False`` mit Grund.
"""

import json
import os
import subprocess
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))

from kartenplan_build import bridge as B  # noqa: E402

VENDORED_TREE = os.path.join(HERE, "fixtures", "kartenplan", "planner_tree", "python")
REAL_TREE = os.environ.get("COUPLINGS_TREE")

SETUP_SCRIPT = r"""
import json, os, sys
sys.path.insert(0, sys.argv[1])
from flliper.srt.planner import profile_couplings as PC
from flliper.srt.pdflip import model_profile as MP
fx = os.path.join(sys.argv[1], "..", "test", "registered", "unit", "pdflip", "fixtures", "profil_s3_1003", sys.argv[2])
hw = PC.synthetic_hardware([("NVIDIA GeForce RTX 5090", 32607, 1400.0), ("NVIDIA GeForce RTX 3080", 20480, 700.0), ("NVIDIA GeForce RTX 3080", 20480, 700.0)])
print(json.dumps({"hw": hw, "model": MP.estimate(os.path.abspath(fx))}))
"""


def profiles_via_subprocess(tree, fixture):
    """Hardware- und Modellprofil im KINDPROZESS bauen: der Dashboard-Testprozess darf flliper nicht importieren
    (test_modellprofil_960::test_tree_is_found_by_file_path_not_by_import)."""
    p = subprocess.run([sys.executable, "-c", SETUP_SCRIPT, tree, fixture], capture_output=True, text=True, timeout=120,
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONWARNINGS="ignore"))
    if p.returncode != 0:
        raise RuntimeError(p.stderr[-600:])
    return json.loads(p.stdout)


def hardware():
    return {"schema": "flliper.hardware/1", "id": "t", "cards": [
        {"ord": 0, "name": "NVIDIA GeForce RTX 5090", "vram_total_mib": {"v": 32607.0, "src": "NVML"},
         "mem_gbs": {"gemv": {"v": 1400.0, "src": "gemessen"}}},
        {"ord": 1, "name": "NVIDIA GeForce RTX 3080", "vram_total_mib": {"v": 20480.0, "src": "NVML"},
         "mem_gbs": {"gemv": {"v": 700.0, "src": "gemessen"}}}]}


class TestCouplingsBridge(unittest.TestCase):
    def test_tree_without_the_module_is_named_not_crashed(self):
        out = B.couplings({"what": "compute", "hardware": hardware(), "model": {}, "settings": {}},
                          tree_python=VENDORED_TREE, python=sys.executable, timeout=120)
        self.assertFalse(out.get("ok"))
        self.assertIn("profile_couplings", out.get("error", ""))

    def test_runner_knows_the_couplings_op(self):
        out = B.call("nope", {}, tree_python=VENDORED_TREE, python=sys.executable, timeout=60)
        self.assertFalse(out.get("ok"))
        self.assertIn("unbekannte Operation", json.dumps(out) + str(out))

    @unittest.skipUnless(REAL_TREE and os.path.isdir(REAL_TREE), "COUPLINGS_TREE (<py-Baum>/python mit planner/profile_couplings.py) nicht gesetzt")
    def test_real_tree_computes_and_unwraps_the_runner_envelope(self):
        prof = profiles_via_subprocess(REAL_TREE, "qwen27b_int8_vocabembed")
        model = prof["model"]
        req = {"what": "compute", "hardware": prof["hw"], "model": model, "settings": {"stage_layers": [32, 16, 16], "kv_dtype": "fp8_e4m3"}}
        out = B.couplings(req, tree_python=REAL_TREE, python=sys.executable, timeout=180)
        self.assertTrue(out["ok"], out)
        self.assertEqual(len(out["result"]["c1"]["stages"]), 3)
        bad = B.couplings(dict(req, settings={"stage_layers": [64]}), tree_python=REAL_TREE, python=sys.executable, timeout=180)
        self.assertFalse(bad["ok"])
        self.assertIn("vector_length", bad["error"])


if __name__ == "__main__":
    unittest.main()
