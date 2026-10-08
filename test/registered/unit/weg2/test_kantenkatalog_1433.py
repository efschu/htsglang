"""PROFIL-EDITOR S4c (Auftrag 1433): der Kantenkatalog -- jede Kante zeigt auf etwas, das es gibt, und hat einen Beleg.

Gepinnt:
  * ``kantenkatalog_1004.json`` existiert, ist Schema ``flliper.kanten/1``, und jede Kante hat
    eindeutige id, Vokabular-wertiges ``rel``/``calc``, einen nicht-trivialen Satz und einen Beleg
    (Datei:Zeile:Anker) mit existierendem Namen auf beiden Seiten.
  * Namen existieren im Katalog: kuratiert, Launcher-Parser, ServerArgs, ``environ.py`` oder
    Ablehnungscode des Registers (wie im S1-Kuraten-Test).
  * ``wert`` (Wertebereich): liegt er an einem Flag mit ``choices`` vor, muss der Wert drin stehen.
  * Beleg-Aufloesung (Auftrag 2013): der Anker-Text wird in der Datei gesucht (``PC.resolve_edge_belege``); die gespeicherte
    Zeile ist nur ein Hinweis. Bricht nur bei fehlendem/mehrdeutigem Anker, nicht bei verschobenen Zeilen. Externe Belege (absolute Pfade, z. B. docker/entrypoint.sh) werden nur geprueft,
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


#: line probe (module exists, never a sha or a branch name): the Dual form (dual_green.py, dual_d_kv_stage.py ...) is a 27B-line feature
DUAL_LINE = os.path.isfile(os.path.join(WEG2, "dual_green.py"))
#: edge endpoints the code of the NF line does not carry (measured 07.10. on 2e68b3f94b: K123 names the Dual ENV, the NF tree has no
#: ``dual_d_kv_stage``): the edge catalog is shared by both lines, so these edges are catalog text without a code behind them on the NF line
ABSENT_ON_NF_LINE = frozenset(("SGLANG_WEG2_DUAL_D_LIVE_YIELD_WAIT_S",))


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
        if DUAL_LINE:
            self.assertEqual(bad, [])
        else:       # NF line: exactly the named endpoints (a new miss is a defect, a stale entry would hide one)
            self.assertEqual(sorted({n for _i, n in bad}), sorted(ABSENT_ON_NF_LINE))

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
        """Der ANKER-TEXT ist der Beleg (Auftrag 2013): er muss in der Datei stehen und sich eindeutig aufloesen lassen.

        Eine bloss verschobene Zeilennummer bricht den Test NICHT mehr; es bricht nur ein fehlender (``veraltet``), ein
        nicht aufloesbar mehrdeutiger (``mehrdeutig``) Anker oder eine fehlende Repo-Datei (``datei_fehlt``)."""
        # Two lines (07.10.): an edge with ``baeume: ["27b"]`` documents code only the 27B line carries (the Dual form) and is ``andere_linie``
        # on the NF tree; every other edge must resolve in THIS tree (the same file holds for both lines).
        res = PC.resolve_edge_belege(self.kanten, REPO_ROOT, "27b" if DUAL_LINE else "nf")
        self.assertEqual(set(res), {k["id"] for k in self.kanten})
        problems = {i: (r["status"], r["datei"], r["treffer"]) for i, r in res.items() if r["status"] in PC.ANKER_PROBLEM}
        self.assertEqual(problems, {})
        for i, r in res.items():
            if r["status"] == "extern_fehlt":
                self.skipped_external.append(i)
            elif r["status"] in PC.ANKER_FREMD:
                self.assertFalse(DUAL_LINE, i)                            # the 27B tree checks every edge
                self.assertEqual(next(k for k in self.kanten if k["id"] == i).get("baeume"), ["27b"], i)
            else:
                self.assertIn(r["status"], PC.ANKER_OK, i)
                self.assertIsInstance(r["zeile"], int, i)

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


# AP-G (Planer-Workflow 06.10.): die Werte, die der Katalog neu kuratiert erklaert (Text aus argparse help= / environ.py-Kommentar),
# und die Kanten K62ff dazu. Der Pin zaehlt die Kanten: wer eine Kante hinzufuegt oder streicht, hebt ihn bewusst.
APG_NAMES = (
    "--rank-role", "--rank-kv-ratio", "--dcp-size", "--uneven-dcp", "--uneven-dcp-weighted", "SGLANG_UNEVEN_TOKEN_VECTOR",
    "--rank-vocab-ratio", "--speculative-draft-placement", "--draft-kv-on-p", "--d-only", "--profile-inventory",
    "--dual-layout", "--dual-share", "--dual-p-overhead-mib", "--dual-d-prefill-tokens", "--dual-p-duty", "--dual-p-sm-pct",
    "--dual-priority", "--dual-d-min-rate-tps", "--dual-p-min-share", "--dual-share-actuators", "--dual-green-ladder",
    "--dual-d-capture-prio", "--dual-p-mps-low-prio", "--dual-p-sleep", "--dual-unified-kv", "--dual-p-kv-max-tokens",
    "--dual-d-kv-max-tokens", "--dual-mps",
)
APG_PAIRS = (
    # C8 D-Seite: --rank-tp-ratio <-> --rank-kv-ratio / --dcp-size / --rank-role, Form A x DCP
    ("--rank-kv-ratio", "--rank-tp-ratio"), ("--rank-kv-ratio", "--dcp-size"),
    ("--dcp-size", "--rank-tp-ratio"), ("--rank-tp-ratio", "--dcp-size"), ("--rank-role", "--rank-tp-ratio"),
    ("--rank-tp-ratio", "--rank-role"), ("--dcp-size", "--rank-role"), ("--rank-role", "--dcp-size"),
    ("--uneven-dcp", "--rank-kv-ratio"), ("--uneven-dcp-weighted", "--rank-kv-ratio"),
    # Draft: Platzierung <-> Draft-KV auf P <-> Host-Budget
    ("--draft-kv-on-p", "--speculative-draft-placement"), ("--speculative-draft-placement", "--draft-kv-on-p"),
    ("--draft-kv-on-p", "--host-ledger-deviation"), ("--rank-role", "--speculative-draft-placement"),
    # Dual: --dual-share impliziert --dual-layout, erzwingt resident und idle tp, verweigert --weg2-d-adopt on
    ("--dual-share", "--dual-layout"), ("--flip-weights", "--dual-layout"), ("--idle-layout", "--dual-layout"),
    ("--dual-layout", "--weg2-d-adopt"),
)
# rel of the mandatory pairs where it is a danger direction (review fix 1/2, 06.10.): a forced value is a derivation, NOT an exclusion;
# only a real raise is schliesst_aus; --rank-kv-ratio never touches the weight split, so there is NO skaliert_mit edge tp-ratio -> kv-ratio.
APG_REL = {
    ("--dual-share", "--dual-layout"): "braucht", ("--flip-weights", "--dual-layout"): "abgeleitet_von",
    ("--idle-layout", "--dual-layout"): "abgeleitet_von", ("--dual-layout", "--weg2-d-adopt"): "schliesst_aus",
    ("--rank-kv-ratio", "--rank-tp-ratio"): "braucht",
    # review fix round 3: the code sets these values itself / the env wins, no raise -> abgeleitet_von (von = the forced/overruled flag, as K44)
    ("--tp-prefill-max-tokens", "--dual-d-prefill-tokens"): "abgeleitet_von", ("--x-ceiling-tokens", "--dual-d-prefill-tokens"): "abgeleitet_von",
    ("--d-short-drain-tokens", "--dual-d-prefill-tokens"): "abgeleitet_von", ("--rank-kv-ratio", "SGLANG_UNEVEN_TOKEN_VECTOR"): "abgeleitet_von",
}


class AuftragG(Basis, unittest.TestCase):
    def test_edge_count_is_pinned(self):
        self.assertEqual(len(self.kanten), 134)                         # 61 + 47 (K62-K108, AP-G; fix round 1 dropped 2 wrongly typed edges) + 8 (K109-K116, AP-H1: Dual-ENV-Tabelle) + 15 (K117-K131, Katalog-Neubau 07.10.: Waechter-Envs, D-COMPACT, AUX-SPILL, --x-mode/--x-curves) + 2 (K132-K133, H88-E 07.10.: --moe-act-int8 / SGLANG_MOE_ACT_INT8, baeume nf) + 1 (K134, H88-F 08.10.: L3-Identitaets-Kante --moe-act-int8 -> --hicache-storage-backend, Anker aus H88-D hicache_storage.py:396; offen.txt Nr. 17 ist damit erledigt))
        self.assertEqual([k["id"] for k in self.kanten], ["K%02d" % i for i in range(1, 135)])

    def test_apg_values_are_curated_and_explained_from_the_source(self):
        for n in APG_NAMES:
            self.assertIn(n, CU.CURATED, n)
            c = CU.CURATED[n]
            self.assertGreater(len(c["text"]), 40, n)
            self.assertTrue(c.get("satz_quelle"), n)                    # where the sentence comes from (help= / environ comment)
            # edges of the new values live ONLY in the edge catalog (with evidence): no unproven curated edge is added
            if n != "--draft-kv-on-p":
                self.assertEqual(c["depends"], [], n)

    def test_apg_edges_exist_and_each_one_is_new(self):
        have = {(k["von"], k["nach"]) for k in self.kanten}
        for pair in APG_PAIRS:
            self.assertIn(pair, have, pair)
        pairs = [(k["von"], k["nach"]) for k in self.kanten]
        self.assertEqual(len(pairs), len(set(pairs)))                   # the loader keys edges by (von, nach): a duplicate would be lost silently

    def test_apg_pair_rels_are_pinned(self):
        by = {(k["von"], k["nach"]): k for k in self.kanten}
        for pair, rel in APG_REL.items():
            self.assertEqual(by[pair]["rel"], rel, pair)
        # a forced-and-logged value is never an exclusion: the 'resident' edge names the forced value
        self.assertEqual(by[("--flip-weights", "--dual-layout")]["wert"], "resident")
        # the weight split is never affected by --rank-kv-ratio (server_args.py help): no coupling edge in either direction but 'braucht'
        self.assertNotIn(("--rank-tp-ratio", "--rank-kv-ratio"), by)
        # dual-layout may never be 'schliesst_aus' against a flag it merely forces
        self.assertNotIn(("--dual-layout", "--flip-weights"), by)
        self.assertNotIn(("--dual-layout", "--idle-layout"), by)

    def test_every_apg_edge_has_a_repo_anchor_in_a_source_file(self):
        for k in self.kanten[61:]:
            self.assertFalse(os.path.isabs(k["beleg"]["datei"]), k["id"])
            self.assertTrue(k["beleg"]["datei"].startswith("python/sglang/srt/"), k["id"])


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
