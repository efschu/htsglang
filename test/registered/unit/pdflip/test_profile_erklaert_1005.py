"""PROFIL-EDITOR 2016: ``ERKLAERT`` (maschinell geschriebene Ein-Satz-Erklaerungen) neben ``CURATED`` (handkuratierter Kern).

Gepinnt:
  * ``ERKLAERT`` und ``CURATED`` sind disjunkt; jeder Name aus ``ERKLAERT`` existiert im Launcher-Parser, in ServerArgs oder in ``environ.py``;
    der Text ist nicht leer; die Stufe ist "experte" (der Einfach-Modus zeigt nichts Ungeprueftes).
  * ``build_catalog(..., erklaert=...)`` setzt Status "erklaert" fuer diese Eintraege, "kuratiert" bleibt dem Kern; auf einen Namenskonflikt gewinnt CURATED.
  * ``profile_json.explain_row`` unterscheidet die Herkunft (Teil-Art und Status "erklaert").
"""

import importlib.util
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "python"))
PDFLIP = os.path.join(PY, "flliper", "srt", "pdflip")
SRT = os.path.join(PY, "flliper", "srt")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


#: line probe (module exists, never a sha or a branch name): the Dual form (dual_green.py ...) is a 27B-line feature
DUAL_LINE = os.path.isfile(os.path.join(PDFLIP, "dual_green.py"))
TREE = "27b" if DUAL_LINE else "nf"

PC = _load("t_pc_erklaert", os.path.join(PDFLIP, "profile_catalog.py"))
CU = _load("t_cu_erklaert", os.path.join(PDFLIP, "profile_catalog_curated.py"))


def _build(erklaert=None, curated=None):
    return PC.build_catalog(os.path.join(PDFLIP, "launcher.py"), os.path.join(SRT, "environ.py"),
                            CU.CURATED if curated is None else curated, "t", os.path.join(SRT, "server_args.py"),
                            erklaert=CU.ERKLAERT if erklaert is None else erklaert, srt_dir=SRT)


def _here(name, c):
    """True when the explained name is expected in THIS tree: a ``baeume_erwartet`` mark names the trees that carry it (the NF-only schalter
    ``["nf"]``, the seven 27B-only Dual names ``["27b"]``); an unmarked name is in both trees."""
    return TREE in c.get("baeume_erwartet", [TREE])


def _local_names():
    """Namen mit Erklärtext, die in DIESEM Baum stehen (ohne die Namen, die nur der andere Baum trägt)."""
    return [n for n, c in CU.ERKLAERT.items() if _here(n, c)]


class Erklaert(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cat = _build()

    def test_disjoint_from_curated(self):
        self.assertFalse(set(CU.ERKLAERT) & set(CU.CURATED))

    def test_every_name_exists_and_text_is_set(self):
        known = set(PC.launcher_flags(os.path.join(PDFLIP, "launcher.py"))) \
            | set(PC.server_args_flags(os.path.join(SRT, "server_args.py"))) \
            | set(PC.environ_fields(os.path.join(SRT, "environ.py"))) | set(PC.environ_constants(SRT))
        for name, e in CU.ERKLAERT.items():
            if _here(name, e):
                self.assertIn(name, known, name)
            else:
                # nur im anderen Baum vermerkt: steht der Name hier doch, stimmt der Vermerk nicht mehr (Baum nachgezogen) und muss weg
                self.assertNotIn(name, known, "%s steht jetzt in diesem Baum: baeume_erwartet prüfen" % name)
                if "baeume_erwartet" in e:
                    self.assertTrue(e.get("satz_quelle"), name)
            self.assertTrue(str(e.get("text", "")).strip(), name)
            self.assertEqual(e.get("level"), "experte", name)

    def test_status_is_erklaert_and_core_stays_kuratiert(self):
        for name in _local_names():
            self.assertEqual(self.cat["entries"][name]["status"], "erklaert", name)
        for name in set(CU.ERKLAERT) - set(_local_names()):
            self.assertNotIn(name, self.cat["entries"], "kein Geistereintrag für einen Namen aus dem anderen Baum")
        for name in CU.CURATED:
            self.assertEqual(self.cat["entries"][name]["status"], "kuratiert", name)
        st = self.cat["stats"]
        self.assertEqual(st["erklaert"], len(_local_names()))

    def test_curated_wins_on_name_clash(self):
        name = _local_names()[0]
        both = _build(curated={name: {"kind": "env", "text": "Kern", "gain": "", "cost": "", "depends": []}})
        self.assertEqual(both["entries"][name]["status"], "kuratiert")
        self.assertEqual(both["entries"][name]["text"], "Kern")

    def test_explain_row_keeps_the_origin(self):
        PJ = _load("t_pj_erklaert", os.path.join(PDFLIP, "profile_json.py"))
        name = _local_names()[0]
        ex = PJ.explain_row({"name": name}, self.cat["entries"], None)
        self.assertEqual(ex["status"], "erklaert")
        self.assertEqual(ex["parts"][0]["kind"], "erklaert")
        core = next(iter(CU.CURATED))
        self.assertEqual(PJ.explain_row({"name": core}, self.cat["entries"], None)["status"], "kuratiert")


if __name__ == "__main__":
    unittest.main()
