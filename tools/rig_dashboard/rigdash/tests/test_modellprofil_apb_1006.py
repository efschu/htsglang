"""AP-B (Planer-Workflow 06.10.2026): ``ModelEstimator.status`` -- der Fall "Modell nicht gemountet" als Zustand statt als HTTP-400-Text.

Gepinnt:
  * ein FEHLENDER Pfad unter einer Modellwurzel ist ``ok`` mit ``state: not_mounted`` (der Planer liest ihn als ``unverified``), ein leeres
    Verzeichnis ``empty``; ein vollständiges Modell ``complete``;
  * Pfade außerhalb der Wurzeln, ``..`` und ungültige ``gguf_file`` bleiben ``ValueError`` (die Wurzel-Prüfung gilt auch für fehlende Pfade);
  * ein Planer-Baum ohne ``probe`` (zu alte Linie) wird benannt (``ModellprofilUnavailable``), nicht still übergangen;
  * ``estimate`` und ``check_path`` verhalten sich unverändert (fehlender Pfad = ValueError "does not exist").
"""

import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)

from rigdash import modellprofil as M  # noqa: E402
from test_modellprofil_960 import TREE as OLD_TREE, write_model  # noqa: E402

NEW_TREE = os.environ.get("MODELLPROFIL_TREE") or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE)))), "python")


@unittest.skipUnless(os.path.isfile(os.path.join(NEW_TREE, M.MODULE_REL)), "Planer-Baum ohne model_profile.py")
class TestStatus(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = os.path.realpath(self.tmp.name)
        self.model, _ = write_model(self.root, "tiny")
        self.est = M.ModelEstimator(tree=NEW_TREE, roots=[self.root])

    def tearDown(self):
        self.tmp.cleanup()

    def test_complete_model(self):
        r = self.est.status({"path": self.model})
        self.assertTrue(r["ok"])
        self.assertEqual((r["state"], r["estimable"], r["safetensors"], r["has_config"]), ("complete", True, 1, True))
        self.assertTrue(r["planner_module"].endswith(M.MODULE_REL))

    def test_missing_path_under_the_root_is_not_mounted_and_not_an_error(self):
        r = self.est.status({"path": os.path.join(self.root, "NF-nicht-da")})
        self.assertTrue(r["ok"])
        self.assertEqual((r["state"], r["estimable"]), ("not_mounted", False))
        self.assertIn("not mounted", r["reason"])

    def test_empty_directory_is_an_empty_mountpoint(self):
        os.makedirs(os.path.join(self.root, "mountpunkt"))
        r = self.est.status({"path": os.path.join(self.root, "mountpunkt")})
        self.assertEqual((r["state"], r["estimable"]), ("empty", False))

    def test_the_root_check_holds_for_missing_paths_too(self):
        for req, word in (({"path": "/gibt/es/nicht"}, "model root"), ({"path": os.path.join(self.root, "..", "x")}, "model root"),
                          ({"path": "relativ"}, "absolut"), ({}, "missing"), ({"path": self.model, "gguf_file": "../a.gguf"}, "gguf_file"),
                          ({"path": self.model, "gguf_file": "a.bin"}, "gguf_file")):
            with self.assertRaises(ValueError, msg=str(req)) as cm:
                self.est.status(req)
            self.assertIn(word, str(cm.exception), req)
        with self.assertRaises(ValueError):
            self.est.status("kein objekt")

    def test_estimate_still_refuses_a_missing_path_with_text(self):
        with self.assertRaises(ValueError) as cm:
            self.est.estimate({"path": os.path.join(self.root, "NF-nicht-da")})
        self.assertIn("does not exist", str(cm.exception))

    def test_an_old_planner_tree_without_probe_is_named(self):
        old = M.ModelEstimator(tree=OLD_TREE, roots=[self.root])
        with self.assertRaises(M.ModellprofilUnavailable) as cm:
            old.status({"path": self.model})
        self.assertIn("probe", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
