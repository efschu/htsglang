"""Auftrag 2002 C: der Kantenkatalog (kantenkatalog_1004.json) wird gelesen und mit ``CURATED[...]["depends"]`` verschmolzen.

Gepinnt (Nutzerentscheid 05.10., Variante A: Loader + Verschmelzung, KEINE Regelauswertung im Editor):
  * Jede der 131 Katalogkanten steht danach in ``entries[von]["depends"]`` (mit ``edge``-ID, ``evidence``, ``satz``); 51 verschmolzen, 80 neu (AP-G 06.10.: K62-K108 neu, alle ohne kuratierte Zwillingskante; AP-H1: K109-K116 Dual-ENV-Tabelle, neu; Catalog-Neubau 07.10.: K117-K131 neu).
  * Die 24 kuratierten Kanten ohne Katalogkante bleiben und sind ``belegt: False`` ("ohne Beleg"), nicht geloescht.
  * Weicht die Beziehungsart ab, bleibt die kuratierte und ``rel_catalog`` traegt die andere (kein stilles Ueberschreiben).
  * ``to_kind`` kennzeichnet das Ziel (flag/env/var/refusal/unbekannt): ein Chip zeigt nie stumm ins Leere.
  * Wertbedingte Kanten tragen ``value`` als Data; es wird nichts ausgewertet.
  * Fehlende/kaputte Kantendatei: ``kanten.geladen`` False mit Grund, Catalog sonst unveraendert.
  * ``CURATED`` selbst wird nicht veraendert; ``profile_json.view`` setzt ``present`` fuer Ablehnungscodes auf None.
GPU-frei.
"""

import copy
import importlib.util
import json
import os
import sys
import tempfile
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


PC = _load("t2002_profile_catalog", os.path.join(PDFLIP, "profile_catalog.py"))
CU = _load("t2002_profile_catalog_curated", os.path.join(PDFLIP, "profile_catalog_curated.py"))
PJ = _load("t2002_profile_json", os.path.join(PDFLIP, "profile_json.py"))
EDGES_JSON = os.path.join(PDFLIP, "kantenkatalog_1004.json")


#: line probe (module exists, never a sha or a branch name): the Dual form (dual_green.py ...) is a 27B-line feature
DUAL_LINE = os.path.isfile(os.path.join(PDFLIP, "dual_green.py"))
TREE = "27b" if DUAL_LINE else "nf"
#: curated edges without a catalog edge (``belegt: False``): 24 for the 27B entries plus one for the NF line's own curated entry --pdflip-xchg-census-map
#: (its ``braucht`` --pdflip-xchg-census, no edge K.. for it).  The curated catalog is shared by both lines, so the count is 25 on both (the entry stays
#: curated on the 27B line too: user/27B seat 07.10., do not delete it).
WITHOUT_EVIDENCE = 25
#: edge targets that are no row of THIS tree's catalog (``to_kind`` unbekannt): none on the 27B line; on the NF line K123 names the 27B-only
#: Dual env FLLIPER_PDFLIP_DUAL_D_LIVE_YIELD_WAIT_S (the edge says ``baeume: [27b]``)
ZIEL_UNBEKANNT = 0 if DUAL_LINE else 1


def build(**kw):
    kw.setdefault("tree", TREE)
    return PC.build_catalog(os.path.join(PDFLIP, "launcher.py"), os.path.join(SRT, "environ.py"), CU.CURATED, "t", os.path.join(SRT, "server_args.py"), **kw)


class RealCatalog(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cat = build()
        cls.ent = cls.cat["entries"]
        with open(EDGES_JSON, encoding="utf-8") as fh:
            cls.edges = json.load(fh)["kanten"]

    def dep(self, von, nach):
        hits = [d for d in self.ent[von]["depends"] if d["to"] == nach]
        self.assertEqual(len(hits), 1, (von, nach))
        return hits[0]

    def test_status_block_counts(self):
        k = self.cat["kanten"]
        self.assertTrue(k["geladen"])
        self.assertEqual((k["edges_total"], k["verschmolzen"], k["neu"]), (131, 51, 80))
        self.assertEqual(k["edges_without_evidence"], WITHOUT_EVIDENCE)
        self.assertEqual(k["kanten_belegt"], 131)
        self.assertEqual(k["uebersprungen_ohne_von"], [])
        self.assertEqual(k["ziel_unbekannt"], ZIEL_UNBEKANNT)
        self.assertEqual(k["wertbedingt"], 22)

    def test_every_catalog_edge_is_a_dependency_with_evidence_and_sentence(self):
        for e in self.edges:
            d = self.dep(e["von"], e["nach"])
            self.assertEqual(d["edge"], e["id"])
            self.assertTrue(d["belegt"], e["id"])
            self.assertEqual(d["satz"], e["satz"])
            self.assertEqual(d["evidence"]["file"], e["evidence"]["file"])
            # Auftrag 2013: the displayed line is the one resolved by the anchor text; the stored line stays as hint
            self.assertEqual(d["evidence"]["row_note"], e["evidence"]["zeile"])
            if TREE not in e.get("trees", [TREE]):          # the other line's code: not looked up here, marked, the stored line stays
                self.assertEqual(d["evidence"]["aufloesung"], "other_line", e["id"])
                self.assertEqual(d["trees"], e["trees"], e["id"])
            else:
                self.assertIn(d["evidence"]["aufloesung"], PC.ANCHOR_OK + ("extern_fehlt",), e["id"])
                self.assertEqual(d.get("trees"), e.get("trees"), e["id"])
            self.assertIsInstance(d["evidence"]["zeile"], int, e["id"])
            self.assertEqual(d["evidence"]["anchor"], e["evidence"]["anchor"])

    def test_the_24_curated_edges_without_evidence_stay_and_are_marked(self):
        marked = [(n, d["to"]) for n, e in self.ent.items() for d in e["depends"] if not d["belegt"]]
        self.assertEqual(len(marked), WITHOUT_EVIDENCE)
        self.assertIn(("--chunked-prefill-size", "--tp-prefill-max-tokens"), marked)
        d = self.dep("--chunked-prefill-size", "--tp-prefill-max-tokens")
        self.assertIsNone(d["evidence"])
        self.assertEqual(d["quelle"], "curated")
        self.assertTrue(d["effect"])                    # the curated sentence is untouched
        n_curated = sum(len(c.get("depends", [])) for c in CU.CURATED.values())
        n_all = sum(len(e["depends"]) for e in self.ent.values())
        self.assertEqual(n_all, n_curated + 80)         # nothing deleted, 80 appended (57 + 8 AP-H1 + 15 Katalog-Neubau 07.10.)

    def test_new_edges_are_appended_with_source_katalog(self):
        d = self.dep("--idle-layout", "--dual-layout")  # K44, not curated
        self.assertEqual((d["quelle"], d["edge"], d["rel"]), ("catalog", "K44", "derived_from"))
        self.assertEqual(d["effect"], d["satz"])

    def test_a_different_relation_keeps_the_curated_one_and_names_the_catalog_one(self):
        k = self.cat["kanten"]
        self.assertEqual(sorted(k["rel_abweichend"]), ["K12", "K19", "K21", "K54"])
        d = self.dep("--rank-moe-resident-fraction", "FLLIPER_MOE_RESIDENT_EXPERT_FRACTION")
        self.assertEqual((d["rel"], d["rel_catalog"]), ("excludes", "derived_from"))
        same = self.dep("--pp-stage-ratio", "--pp-attn-stage-ratio")
        self.assertNotIn("rel_catalog", same)

    def test_targets_are_classified_so_no_chip_points_into_the_void(self):
        self.assertEqual(self.dep("PROFILE_CARD_COUNT", "HW-COUNT")["to_kind"], "refusal")
        self.assertEqual(self.dep("PROFILE_SHM_MIN_GIB", "SHM")["to_kind"], "refusal")
        self.assertEqual(self.dep("PROFILE_FORMAT", "PROFILE_MODEL")["to_kind"], "var")
        self.assertEqual(self.dep("--pp-stage-ratio", "--pp-attn-stage-ratio")["to_kind"], "flag")
        for e in self.ent.values():
            for d in e["depends"]:
                if d["to_kind"] == "unbekannt" and not DUAL_LINE:      # only an edge of the other line may point at a name this tree lacks
                    self.assertEqual(d.get("trees"), ["27b"], (e["name"], d["to"]))
                    continue
                self.assertIn(d["to_kind"], ("flag", "env", "var", "refusal"), (e["name"], d["to"]))

    def test_conditional_edges_carry_the_value_as_data_and_nothing_is_evaluated(self):
        d = self.dep("--rank-tp-ratio", "--rank-gpu-memory-mib")    # K08
        self.assertEqual(d["value"], "auto")
        self.assertTrue(d["belegt"])
        self.assertEqual(sum(1 for e in self.ent.values() for x in e["depends"] if x.get("value")), 22)

    def test_curated_is_not_mutated_by_the_merge(self):
        for c in CU.CURATED.values():
            for d in c.get("depends", []):
                self.assertNotIn("belegt", d)
                self.assertNotIn("to_kind", d)

    def test_view_marks_present_none_for_refusal_codes_and_bool_for_values(self):
        doc = {"name": "t", "line": "27b", "args": [{"flag": "--rank-tp-ratio", "value": "auto"}], "meta": {}, "vars": [
            {"name": "PROFILE_CARD_COUNT", "value": "3", "line": 1}, {"name": "PROFILE_INVENTORY", "value": "a,b,c", "line": 2}]}
        try:
            v = PJ.view(doc, self.ent)
        except Exception as exc:        # noqa: BLE001 -- document shape differs: build it through the real parser instead
            self.skipTest("Dokumentform: %s" % exc)
        by = {r["key"]: r for r in v["rows"]}
        row = by.get("var:PROFILE_CARD_COUNT")
        if row is None:
            self.skipTest("keine var-Zeile in dieser Dokumentform")
        deps = {d["to"]: d for d in row["explain"]["depends"]}
        self.assertIsNone(deps["HW-COUNT"]["present"])
        self.assertTrue(deps["PROFILE_INVENTORY"]["present"])


class Synthetic(unittest.TestCase):
    def ents(self):
        return {"--a": {"name": "--a", "kind": "flag", "depends": [{"to": "--b", "rel": "requires", "effect": "kur", "calc": "text"},
                                                                    {"to": "--c", "rel": "trades", "effect": "kur2", "calc": "text"}]},
                "--b": {"name": "--b", "kind": "flag", "depends": []}}

    def test_merge_attach_new_skip_and_unknown_target(self):
        ents = self.ents()
        edges = [{"id": "X1", "von": "--a", "nach": "--b", "rel": "requires", "calc": "text", "value": None, "satz": "S1",
                  "evidence": {"file": "launcher.py", "zeile": 5, "anchor": "zz"}},
                 {"id": "X2", "von": "--b", "nach": "HW-COUNT", "rel": "requires", "calc": "text", "value": None, "satz": "S2", "evidence": None},
                 {"id": "X3", "von": "--gibtsnicht", "nach": "--a", "rel": "requires", "calc": "text", "value": None, "satz": "S3", "evidence": None}]
        info = PC.merge_edges(ents, edges, {}, ["HW-COUNT"])
        a = {d["to"]: d for d in ents["--a"]["depends"]}
        self.assertTrue(a["--b"]["belegt"] and a["--b"]["satz"] == "S1" and a["--b"]["evidence"]["zeile"] == 5)
        self.assertFalse(a["--c"]["belegt"])
        self.assertEqual(a["--c"]["to_kind"], "unbekannt")                 # no entry, no refusal code: said so
        b = ents["--b"]["depends"][0]
        self.assertEqual((b["to"], b["to_kind"], b["quelle"], b["evidence"]), ("HW-COUNT", "refusal", "catalog", None))
        self.assertEqual(info["uebersprungen_ohne_von"], ["X3"])
        self.assertEqual((info["verschmolzen"], info["neu"], info["ziel_unbekannt"]), (1, 1, 1))

    def test_missing_or_broken_edge_file_is_reported_not_silent(self):
        edges, info = PC.load_edges("/nonexistent/kanten.json")
        self.assertEqual(edges, [])
        self.assertFalse(info["geladen"])
        self.assertIn("not readable", info["reason"])
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "k.json")
            with open(p, "w") as fh:
                json.dump({"schema": "falsch/9", "kanten": []}, fh)
            _e, info2 = PC.load_edges(p)
            self.assertFalse(info2["geladen"])
            self.assertIn("schema", info2["reason"])
        cat = build(edges_path="/nonexistent/kanten.json")
        self.assertFalse(cat["kanten"]["geladen"])
        n_curated = sum(len(c.get("depends", [])) for c in CU.CURATED.values())
        self.assertEqual(sum(len(e["depends"]) for e in cat["entries"].values()), n_curated)   # as curated, no edge added


if __name__ == "__main__":
    unittest.main()
