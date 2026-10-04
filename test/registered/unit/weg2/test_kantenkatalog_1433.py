"""PROFIL-EDITOR S4c (Auftrag 1433): der Kantenkatalog -- jede Kante zeigt auf etwas, das es gibt, und hat einen Beleg.

Gepinnt:
  * ``kantenkatalog_1004.json`` existiert, ist Schema ``flliper.kanten/1``, und jede Kante hat
    eindeutige id, Vokabular-wertiges ``rel``/``calc``, einen nicht-trivialen Satz und einen Beleg
    (Datei:Zeile:Anker) mit existierendem Namen auf beiden Seiten.
  * Namen existieren im Katalog: kuratiert, Launcher-Parser, ServerArgs, ``environ.py`` oder
    Ablehnungscode des Registers (wie im S1-Kuraten-Test).
  * ``wert`` (Wertebereich): liegt er an einem Flag mit ``choices`` vor, muss der Wert drin stehen.
  * Beleg-Plausibilitaet: Datei existiert, Zeile im gueltigen Bereich, Anker-Token innerhalb von
    ±2 Zeilen. Externe Belege (absolute Pfade, z. B. docker/entrypoint.sh) werden nur geprueft,
    wenn die Datei anwesend ist -- auf Maschinen ohne Docker-Baum bleibt der Test grün.
  * Abdeckung sinkt nicht: Mindestanzahl Kanten und Mindestabdeckung der Kuraten-Kernwerte.
  * ``offen.txt``: jede dort gelistete Kante zeigt ebenfalls auf existierende Namen (kein
    erfundener Wert im Offenen-Bestand).
"""

import importlib.util
import json
import os
import re
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "python"))
WEG2 = os.path.join(PY, "sglang", "srt", "weg2")
SRT = os.path.join(PY, "sglang", "srt")
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
KAT = os.path.join(WEG2, "kantenkatalog_1004.json")
OFFEN = os.path.join(WEG2, "offen.txt")

RELS = {"tauscht", "braucht", "schliesst_aus", "abgeleitet_von", "skaliert_mit"}
CALCS = {"text", "S4"}


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


PC = _load("t_kanten_catalog", os.path.join(WEG2, "profile_catalog.py"))
CU = _load("t_kanten_curated", os.path.join(WEG2, "profile_catalog_curated.py"))
RF = _load("t_kanten_refusals", os.path.join(WEG2, "refusals.py"))


class Basis:
    @classmethod
    def setUpClass(cls):
        cls.flags = PC.launcher_flags(os.path.join(WEG2, "launcher.py"))
        cls.server = PC.server_args_flags(os.path.join(SRT, "server_args.py"))
        cls.envs = PC.environ_fields(os.path.join(SRT, "environ.py"))
        cls.codes = {r.code for r in RF.REGISTER}
        cls.names = set(cls.flags) | set(cls.server) | set(cls.envs) | set(CU.CURATED) | set(cls.codes)
        with open(KAT, encoding="utf-8") as fh:
            cls.cat = json.load(fh)
        cls.kanten = cls.cat["kanten"]

    @classmethod
    def _resolve(cls, datei):
        return datei if os.path.isabs(datei) else os.path.join(REPO_ROOT, datei)


class KatalogForm(Basis, unittest.TestCase):
    def test_schema_and_vocabulary(self):
        self.assertEqual(self.cat["schema"], "flliper.kanten/1")
        self.assertEqual(set(self.cat["rel_vokabular"]), RELS)
        self.assertEqual(set(self.cat["calc_vokabular"]), CALCS)
        ids = [k["id"] for k in self.kanten]
        self.assertEqual(len(ids), len(set(ids)))
        for k in self.kanten:
            self.assertIn(k["rel"], RELS, k["id"])
            self.assertIn(k["calc"], CALCS, k["id"])
            self.assertTrue(len(k["satz"]) >= 20 and " " in k["satz"], k["id"])
            self.assertNotEqual(k["von"], k["nach"], k["id"])
            b = k["beleg"]
            self.assertIsInstance(b["zeile"], int)
            self.assertGreaterEqual(b["zeile"], 1)
            self.assertTrue(b["anker"], k["id"])
            self.assertTrue(b["datei"], k["id"])

    def test_every_name_exists_in_the_catalog(self):
        """Namen existieren: kuratiert, Parser, ServerArgs, environ.py oder Registercode."""
        bad = [(k["id"], n) for k in self.kanten for n in (k["von"], k["nach"]) if n not in self.names]
        self.assertEqual(bad, [])

    def test_wert_is_within_the_flag_choices(self):
        """Wertebereich: ein 'wert' muss zu den choices des Flags gehoeren (wenn bekannt)."""
        for k in self.kanten:
            w = k.get("wert")
            if w is None:
                continue
            entry = self.flags.get(k["von"]) or self.server.get(k["von"])
            if entry and entry.get("choices"):
                self.assertIn(w, entry["choices"], k["id"])


class Beleg(Basis, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.files = {}
        cls.skipped_external = []

    def _lines(self, path):
        if path not in self.files:
            if not os.path.isfile(path):
                self.files[path] = None
                return None
            with open(path, encoding="utf-8") as fh:
                self.files[path] = fh.readlines()
        return self.files[path]

    def test_every_edge_has_a_readable_anchor(self):
        missing, bad_anchor = [], []
        for k in self.kanten:
            b = k["beleg"]
            path = self._resolve(b["datei"])
            lines = self._lines(path)
            if lines is None:
                if os.path.isabs(b["datei"]):
                    self.skipped_external.append(k["id"])
                else:
                    missing.append(k["id"])
                continue
            if b["zeile"] > len(lines):
                missing.append(k["id"])
                continue
            window = "".join(lines[b["zeile"] - 3:b["zeile"] + 2])
            if b["anker"] not in window:
                bad_anchor.append(k["id"])
        self.assertEqual(missing, [])
        self.assertEqual(bad_anchor, [])

    def test_repo_belege_are_all_verified(self):
        """Repo-interne Belege duerfen niemals uebersprungen werden."""
        internal = [k["id"] for k in self.kanten
                    if not os.path.isabs(k["beleg"]["datei"]) and k["id"] in self.skipped_external]
        self.assertEqual(internal, [])


class Abdeckung(Basis, unittest.TestCase):
    def test_minimum_edge_count(self):
        self.assertGreaterEqual(len(self.kanten), 40)

    def test_coverage_of_the_curated_core(self):
        """Abdeckung der Kernwerte: Anteil kuratierter Namen, die in >=1 Kante vorkommen."""
        touched = {n for k in self.kanten for n in (k["von"], k["nach"]) if n in CU.CURATED}
        coverage = len(touched) / len(CU.CURATED)
        print(f"1433 KANTENKATALOG: {len(self.kanten)} Kanten, Abdeckung {len(touched)}/{len(CU.CURATED)} "
              f"Kuraten-Kernwerte ({coverage:.0%})")
        self.assertGreaterEqual(coverage, 0.5)


class Offen(Basis, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        with open(OFFEN, encoding="utf-8") as fh:
            cls.text = fh.read()

    def test_offen_lists_unproven_edges_with_existing_names(self):
        names = re.findall(r"^(\d+)\. (.*?) -> (.*?) \(", self.text, re.M)
        self.assertGreaterEqual(len(names), 8)
        for num, a, b in names:
            self.assertIn(a, self.names, f"offen {num}: {a}")
            self.assertIn(b, self.names, f"offen {num}: {b}")


if __name__ == "__main__":
    unittest.main()
