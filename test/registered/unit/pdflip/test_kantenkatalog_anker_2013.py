"""Auftrag 2013: die Belege des Kantenkatalogs haengen am ANKER-TEXT, nicht an der Zeilennummer.

Gepinnt (rot -> gruen gegen ``ae25180e67``, wo es weder ``resolve_edge_evidence_items`` noch ``resolve_anchor`` gab):
  * ``resolve_anchor``: genau ein Treffer = eindeutig; mehrere = naechster zur erwarteten Zeile (``nah``); Gleichstand oder
    zu weit weg = ``mehrdeutig``; kein Treffer = ``veraltet`` (nie ein stilles Raten).
  * Verschiebung: der echte Quelltext (launcher.py, environ.py, profile_couplings.py) wird in einem Temp-Verzeichnis um N Zeilen
    verschoben (Prepend am Dateianfang UND Einschub mitten in launcher.py); JEDE der 131 Kanten loest weiter auf, auf genau
    die verschobene Zeile -- auch die mit Mehrfachtreffer-Ankern (``exchange`` 229x, ``DFLASH`` 73x, ``entries`` 30x).
  * Mutant: ein entfernter Anker wird ``veraltet`` und macht die Pruefung rot; ein verdoppelter eindeutiger Anker mit
    Gleichstand wird ``mehrdeutig``.
  * ``build_catalog`` zeigt die AUFGELOESTE Zeile (``zeile``), die gespeicherte bleibt ``row_note``.
  * Der Catalog ohne aufloesbares Repo (kein ``edges_root``, fremdes Layout) verhaelt sich wie vorher (keine Zusatzfelder).
  * Zwei Linien (NF-Catalog 07.10.): eine Kante mit ``trees: ["27b"]`` beschreibt Code, den nur die 27B-Linie traegt (Dual-Form); auf dem NF-Baum
    ist sie ``other_line`` (nicht gesucht, kein Problem), auf dem 27B-Baum wird sie wie jede andere geprueft. Jede andere Kante muss in BEIDEN
    Baeumen aufloesen. Die Linie wird an der Datei erkannt (dual_green.py da = 27B-Linie), nie an einem SHA oder Branchnamen.
"""

import importlib.util
import json
import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "python"))
PDFLIP = os.path.join(PY, "flliper", "srt", "pdflip")
SRT = os.path.join(PY, "flliper", "srt")
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
KAT = os.path.join(PDFLIP, "kantenkatalog_1004.json")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


PC = _load("t2013_profile_catalog", os.path.join(PDFLIP, "profile_catalog.py"))
CU = _load("t2013_profile_catalog_curated", os.path.join(PDFLIP, "profile_catalog_curated.py"))

with open(KAT, encoding="utf-8") as _fh:
    KANTEN_ALLE = json.load(_fh)["kanten"]
#: line probe (module exists, never a sha or a branch name): the Dual form (dual_green.py ...) is a 27B-line feature
DUAL_LINE = os.path.isfile(os.path.join(PDFLIP, "dual_green.py"))
TREE = "27b" if DUAL_LINE else "nf"
#: the edges this tree documents (an edge without ``baeume`` holds for both lines)
KANTEN = [k for k in KANTEN_ALLE if TREE in k.get("trees", [TREE])]
REPO_FILES = sorted({k["evidence"]["file"] for k in KANTEN if not os.path.isabs(k["evidence"]["file"])})


def problems(res):
    return {i: r["status"] for i, r in res.items() if r["status"] in PC.ANKER_PROBLEM}


def copy_tree(dst):
    """Copy the real source files the edges point at (repo-relative) into ``dst`` under the same relative paths."""
    for rel in REPO_FILES:
        t = os.path.join(dst, rel)
        os.makedirs(os.path.dirname(t), exist_ok=True)
        shutil.copyfile(os.path.join(REPO_ROOT, rel), t)


def edit(dst, rel, fn):
    p = os.path.join(dst, rel)
    with open(p, encoding="utf-8") as fh:
        text = fh.read()
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(fn(text))


class ResolveAnchor(unittest.TestCase):
    def test_no_hit_is_stale(self):
        self.assertEqual(PC.resolve_anchor([], 10), (None, "veraltet"))

    def test_single_hit_is_unique_wherever_it_is(self):
        self.assertEqual(PC.resolve_anchor([5000], 10), (5000, "eindeutig"))

    def test_several_hits_nearest_to_hint_plus_drift(self):
        self.assertEqual(PC.resolve_anchor([10, 500, 900], 12), (10, "nah"))
        self.assertEqual(PC.resolve_anchor([10, 500, 900], 12, drift=490), (500, "nah"))   # file moved by ~490 above

    def test_tie_is_ambiguous_not_guessed(self):
        self.assertEqual(PC.resolve_anchor([98, 102], 100), (None, "mehrdeutig"))

    def test_too_far_from_expectation_is_ambiguous(self):
        self.assertEqual(PC.resolve_anchor([10, 500], 200), (None, "mehrdeutig"))

    def test_boundary(self):
        self.assertEqual(PC.resolve_anchor([10, 500], 10 + PC.ANCHOR_TOL), (10, "nah"))
        self.assertEqual(PC.resolve_anchor([10, 500], 10 + PC.ANCHOR_TOL + 1), (None, "mehrdeutig"))

    def test_multiline_anchor_start_line_and_dedupe(self):
        text = "a\nfoo\nbar\nx\nfoo bar foo\n"
        self.assertEqual(PC.anchor_lines(text, "foo\nbar"), [2])
        self.assertEqual(PC.anchor_lines(text, "foo"), [2, 5])           # two on line 5 count once
        self.assertEqual(PC.anchor_lines(text, ""), [])


class ResolveEdges(unittest.TestCase):
    def test_real_tree_resolves_all_131(self):
        """All 131 edges, resolved against this tree as its own line: every one resolves, except the edges that name only the OTHER line."""
        res = PC.resolve_edge_evidence_items(KANTEN_ALLE, REPO_ROOT, TREE)
        self.assertEqual(len(res), 131)
        self.assertEqual(problems(res), {})
        for i, r in res.items():
            self.assertIn(r["status"], PC.ANCHOR_OK + PC.ANKER_FREMD + ("extern_fehlt",), i)
        fremd = sorted(i for i, r in res.items() if r["status"] in PC.ANKER_FREMD)
        self.assertEqual(fremd, sorted(k["id"] for k in KANTEN_ALLE if TREE not in k.get("trees", [TREE])))
        # the edges of THIS tree are checked for real: none of them was waved through as andere_linie
        self.assertEqual(sorted(set(res) - set(fremd)), sorted(k["id"] for k in KANTEN))

    def test_without_a_tree_label_every_edge_is_checked(self):
        """No label = no filter: on the NF tree the 44 ``trees: [27b]`` edges are then stale or missing (the label is what lets them pass)."""
        res = PC.resolve_edge_evidence_items(KANTEN_ALLE, REPO_ROOT)
        self.assertFalse(any(r["status"] in PC.ANKER_FREMD for r in res.values()))
        if not DUAL_LINE:
            self.assertEqual(sorted(i for i, r in res.items() if r["status"] in PC.ANKER_PROBLEM),
                             sorted(k["id"] for k in KANTEN_ALLE if k.get("trees") == ["27b"]))

    def test_mutant_unmarked_edge_with_a_missing_anchor_is_not_waved_through(self):
        """A stale anchor on an edge WITHOUT ``trees`` stays a problem under a tree label: ``trees`` is the only way out, never a silent skip."""
        victim = dict(next(k for k in KANTEN if k["id"] == "K01"))
        victim["evidence"] = dict(victim["evidence"], anchor="DIESER-TEXT-STEHT-NIRGENDS-1007")
        for label in ("27b", "nf"):
            self.assertEqual(PC.resolve_edge_evidence_items([victim], REPO_ROOT, label)["K01"]["status"], "veraltet", label)
        marked = dict(victim, trees=["27b"])
        self.assertEqual(PC.resolve_edge_evidence_items([marked], REPO_ROOT, "nf")["K01"]["status"], "other_line")
        self.assertEqual(PC.resolve_edge_evidence_items([marked], REPO_ROOT, "27b")["K01"]["status"], "veraltet")

    def test_dual_module_edges_are_marked_27b_only(self):
        """An edge whose evidence file is a Dual module (dual_*.py) documents 27B-line code only: it must say so (``trees: [27b]``)."""
        unmarked = [k["id"] for k in KANTEN_ALLE
                    if os.path.basename(k["evidence"]["file"]).startswith("dual_") and k.get("trees") != ["27b"]]
        self.assertEqual(unmarked, [])

    def test_prepend_shift_keeps_every_edge_and_moves_it_exactly(self):
        base = PC.resolve_edge_evidence_items(KANTEN, REPO_ROOT)
        for n in (1, 7, 44, 313):
            with tempfile.TemporaryDirectory() as d:
                copy_tree(d)
                for rel in REPO_FILES:
                    edit(d, rel, lambda t: "# pad\n" * n + t)
                res = PC.resolve_edge_evidence_items(KANTEN, d)
                self.assertEqual(problems(res), {}, n)
                for k in KANTEN:
                    i = k["id"]
                    if os.path.isabs(k["evidence"]["file"]):
                        continue
                    self.assertEqual(res[i]["zeile"], base[i]["zeile"] + n, (n, i))
                    self.assertEqual(res[i]["note"], k["evidence"]["zeile"])     # the stored line is kept as hint

    def test_mid_file_insertion_moves_only_what_is_below(self):
        base = PC.resolve_edge_evidence_items(KANTEN, REPO_ROOT)
        rel = "python/flliper/srt/pdflip/launcher.py"
        cut, n = 12000, 57
        with tempfile.TemporaryDirectory() as d:
            copy_tree(d)
            edit(d, rel, lambda t: "".join(t.splitlines(True)[:cut]) + "# pad\n" * n + "".join(t.splitlines(True)[cut:]))
            res = PC.resolve_edge_evidence_items(KANTEN, d)
            self.assertEqual(problems(res), {})
            for k in KANTEN:
                if k["evidence"]["file"] != rel:
                    continue
                i = k["id"]
                want = base[i]["zeile"] + (n if base[i]["zeile"] > cut else 0)
                self.assertEqual(res[i]["zeile"], want, i)

    def test_mutant_removed_anchor_goes_stale_and_the_check_goes_red(self):
        victim = next(k for k in KANTEN if k["id"] == "K14")                       # environ.py, unique anchor
        anchor = victim["evidence"]["anchor"]
        with tempfile.TemporaryDirectory() as d:
            copy_tree(d)
            edit(d, victim["evidence"]["file"], lambda t: t.replace(anchor, "ENTFERNT"))
            res = PC.resolve_edge_evidence_items(KANTEN, d)
            self.assertEqual(res["K14"]["status"], "veraltet")
            self.assertIn("K14", problems(res))
            # K15 points at the same anchor/line: stale too; nothing else is touched
            self.assertEqual(sorted(problems(res)), sorted(k["id"] for k in KANTEN
                                                           if k["evidence"]["file"] == victim["evidence"]["file"]
                                                           and k["evidence"]["anchor"] == anchor))

    def test_mutant_ambiguous_anchor_is_ambiguous_not_guessed(self):
        # K01 alone: remove the real anchor, plant it 2 lines before and 2 lines after the stored line -> equally near: a tie
        victim = next(k for k in KANTEN if k["id"] == "K01")
        rel, anchor, real = victim["evidence"]["file"], victim["evidence"]["anchor"], victim["evidence"]["zeile"]
        with tempfile.TemporaryDirectory() as d:
            copy_tree(d)

            def tie(t):
                t = t.replace(anchor, "ENTFERNT")
                ls = t.splitlines(True)
                for ln in (real - 2, real + 2):
                    ls[ln - 1] = ls[ln - 1].rstrip("\n") + " # " + anchor + "\n"
                return "".join(ls)
            edit(d, rel, tie)
            res = PC.resolve_edge_evidence_items([victim], d)["K01"]
            self.assertEqual((res["treffer"], res["status"]), (2, "mehrdeutig"))
            self.assertIn("K01", problems({"K01": res}))
            edit(d, rel, lambda t: t.replace(" # " + anchor, "", 1))             # one copy left: unique again
            res = PC.resolve_edge_evidence_items([victim], d)["K01"]
            self.assertEqual((res["treffer"], res["status"]), (1, "eindeutig"))

    def test_missing_repo_file_is_a_problem_missing_external_is_not(self):
        with tempfile.TemporaryDirectory() as d:        # empty tree: repo files absent
            res = PC.resolve_edge_evidence_items(KANTEN, d)
            for k in KANTEN:
                if os.path.isabs(k["evidence"]["file"]):
                    self.assertIn(res[k["id"]]["status"], ("extern_fehlt",) + PC.ANCHOR_OK)
                else:
                    self.assertEqual(res[k["id"]]["status"], "datei_fehlt")


class CatalogShowsResolvedLine(unittest.TestCase):
    def build(self, **kw):
        return PC.build_catalog(os.path.join(PDFLIP, "launcher.py"), os.path.join(SRT, "environ.py"), CU.CURATED, "t",
                                os.path.join(SRT, "server_args.py"), **kw)

    def test_resolved_line_displayed_stored_line_kept_as_hint(self):
        with tempfile.TemporaryDirectory() as d:
            # a shifted COPY of the tree the catalog is built from (every file an edge points at, same relative layout; AP-G 06.10.:
            # the edges now also point at server_args.py, rank_role.py and host_ledger.py, not only launcher/environ/couplings)
            copy_tree(d)
            for rel in ("python/flliper/srt/pdflip/launcher.py", "python/flliper/srt/environ.py", "python/flliper/srt/pdflip/refusals.py"):
                t = os.path.join(d, rel)
                os.makedirs(os.path.dirname(t), exist_ok=True)
                shutil.copyfile(os.path.join(REPO_ROOT, rel), t)
            n = 31
            for rel in REPO_FILES:
                edit(d, rel, lambda t: "# pad\n" * n + t)
            cat = PC.build_catalog(os.path.join(d, "python/flliper/srt/pdflip/launcher.py"), os.path.join(d, "python/flliper/srt/environ.py"),
                                   CU.CURATED, "t", os.path.join(SRT, "server_args.py"), tree=TREE)
            info = cat["kanten"]["beleg_aufloesung"]
            self.assertEqual(info["problem"], [])
            by_id = {e["id"]: e for e in KANTEN_ALLE}
            base = PC.resolve_edge_evidence_items(KANTEN, REPO_ROOT, TREE)
            seen = fremd = 0
            for ent in cat["entries"].values():
                for dep in ent["depends"]:
                    b = dep.get("evidence")
                    if not dep.get("belegt") or not b or os.path.isabs(b["file"]):
                        continue
                    if b["aufloesung"] in PC.ANKER_FREMD:        # the other line's code: not looked up, the stored line stays
                        fremd += 1
                        self.assertEqual(dep["trees"], by_id[dep["edge"]]["trees"], dep["edge"])
                        self.assertNotIn(TREE, dep["trees"], dep["edge"])
                        continue
                    seen += 1
                    stored = by_id[dep["edge"]]["evidence"]["zeile"]
                    self.assertEqual(b["row_note"], stored)
                    # shown line = resolved line = where THIS tree has the anchor, moved by the pad (the stored hint is the 27B launcher's line:
                    # on the NF tree the anchor sits elsewhere, the hint is only a hint)
                    self.assertEqual(b["zeile"] - base[dep["edge"]]["zeile"], n, dep["edge"])
                    self.assertIn(b["aufloesung"], PC.ANCHOR_OK)
            self.assertEqual(seen, sum(1 for k in KANTEN if not os.path.isabs(k["evidence"]["file"])))
            self.assertEqual(fremd, sum(1 for k in KANTEN_ALLE if TREE not in k.get("trees", [TREE]) and not os.path.isabs(k["evidence"]["file"])))

    def test_unresolvable_layout_leaves_the_catalog_as_before(self):
        # ``edges_root`` given but empty tree and a launcher outside the repo layout: no extra fields, no crash
        with tempfile.TemporaryDirectory() as d:
            lp = os.path.join(d, "launcher.py")
            shutil.copyfile(os.path.join(PDFLIP, "launcher.py"), lp)
            cat = PC.build_catalog(lp, os.path.join(SRT, "environ.py"), CU.CURATED, "t", os.path.join(SRT, "server_args.py"))
            self.assertNotIn("beleg_aufloesung", cat["kanten"])
            for ent in cat["entries"].values():
                for dep in ent["depends"]:
                    if dep.get("evidence"):
                        self.assertNotIn("aufloesung", dep["evidence"])
                        self.assertNotIn("row_note", dep["evidence"])


if __name__ == "__main__":
    unittest.main()
