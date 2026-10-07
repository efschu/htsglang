"""Auftrag 2013: die Belege des Kantenkatalogs haengen am ANKER-TEXT, nicht an der Zeilennummer.

Gepinnt (rot -> gruen gegen ``ae25180e67``, wo es weder ``resolve_edge_belege`` noch ``resolve_anchor`` gab):
  * ``resolve_anchor``: genau ein Treffer = eindeutig; mehrere = naechster zur erwarteten Zeile (``nah``); Gleichstand oder
    zu weit weg = ``mehrdeutig``; kein Treffer = ``veraltet`` (nie ein stilles Raten).
  * Verschiebung: der echte Quelltext (launcher.py, environ.py, profile_couplings.py) wird in einem Temp-Verzeichnis um N Zeilen
    verschoben (Prepend am Dateianfang UND Einschub mitten in launcher.py); JEDE der 116 Kanten loest weiter auf, auf genau
    die verschobene Zeile -- auch die mit Mehrfachtreffer-Ankern (``exchange`` 229x, ``DFLASH`` 73x, ``entries`` 30x).
  * Mutant: ein entfernter Anker wird ``veraltet`` und macht die Pruefung rot; ein verdoppelter eindeutiger Anker mit
    Gleichstand wird ``mehrdeutig``.
  * ``build_catalog`` zeigt die AUFGELOESTE Zeile (``zeile``), die gespeicherte bleibt ``zeile_hinweis``.
  * Der Katalog ohne aufloesbares Repo (kein ``edges_root``, fremdes Layout) verhaelt sich wie vorher (keine Zusatzfelder).
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
WEG2 = os.path.join(PY, "sglang", "srt", "weg2")
SRT = os.path.join(PY, "sglang", "srt")
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
KAT = os.path.join(WEG2, "kantenkatalog_1004.json")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


PC = _load("t2013_profile_catalog", os.path.join(WEG2, "profile_catalog.py"))
CU = _load("t2013_profile_catalog_curated", os.path.join(WEG2, "profile_catalog_curated.py"))

with open(KAT, encoding="utf-8") as _fh:
    KANTEN = json.load(_fh)["kanten"]
REPO_FILES = sorted({k["beleg"]["datei"] for k in KANTEN if not os.path.isabs(k["beleg"]["datei"])})


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
        self.assertEqual(PC.resolve_anchor([10, 500], 10 + PC.ANKER_TOL), (10, "nah"))
        self.assertEqual(PC.resolve_anchor([10, 500], 10 + PC.ANKER_TOL + 1), (None, "mehrdeutig"))

    def test_multiline_anchor_start_line_and_dedupe(self):
        text = "a\nfoo\nbar\nx\nfoo bar foo\n"
        self.assertEqual(PC.anchor_lines(text, "foo\nbar"), [2])
        self.assertEqual(PC.anchor_lines(text, "foo"), [2, 5])           # two on line 5 count once
        self.assertEqual(PC.anchor_lines(text, ""), [])


class ResolveEdges(unittest.TestCase):
    def test_real_tree_resolves_all_116(self):
        res = PC.resolve_edge_belege(KANTEN, REPO_ROOT)
        self.assertEqual(len(res), 116)
        self.assertEqual(problems(res), {})
        for i, r in res.items():
            self.assertIn(r["status"], PC.ANKER_OK + ("extern_fehlt",), i)

    def test_prepend_shift_keeps_every_edge_and_moves_it_exactly(self):
        base = PC.resolve_edge_belege(KANTEN, REPO_ROOT)
        for n in (1, 7, 44, 313):
            with tempfile.TemporaryDirectory() as d:
                copy_tree(d)
                for rel in REPO_FILES:
                    edit(d, rel, lambda t: "# pad\n" * n + t)
                res = PC.resolve_edge_belege(KANTEN, d)
                self.assertEqual(problems(res), {}, n)
                for k in KANTEN:
                    i = k["id"]
                    if os.path.isabs(k["beleg"]["datei"]):
                        continue
                    self.assertEqual(res[i]["zeile"], base[i]["zeile"] + n, (n, i))
                    self.assertEqual(res[i]["hinweis"], k["beleg"]["zeile"])     # the stored line is kept as hint

    def test_mid_file_insertion_moves_only_what_is_below(self):
        base = PC.resolve_edge_belege(KANTEN, REPO_ROOT)
        rel = "python/sglang/srt/weg2/launcher.py"
        cut, n = 12000, 57
        with tempfile.TemporaryDirectory() as d:
            copy_tree(d)
            edit(d, rel, lambda t: "".join(t.splitlines(True)[:cut]) + "# pad\n" * n + "".join(t.splitlines(True)[cut:]))
            res = PC.resolve_edge_belege(KANTEN, d)
            self.assertEqual(problems(res), {})
            for k in KANTEN:
                if k["beleg"]["datei"] != rel:
                    continue
                i = k["id"]
                want = base[i]["zeile"] + (n if base[i]["zeile"] > cut else 0)
                self.assertEqual(res[i]["zeile"], want, i)

    def test_mutant_removed_anchor_goes_stale_and_the_check_goes_red(self):
        victim = next(k for k in KANTEN if k["id"] == "K14")                       # environ.py, unique anchor
        anker = victim["beleg"]["anker"]
        with tempfile.TemporaryDirectory() as d:
            copy_tree(d)
            edit(d, victim["beleg"]["datei"], lambda t: t.replace(anker, "ENTFERNT"))
            res = PC.resolve_edge_belege(KANTEN, d)
            self.assertEqual(res["K14"]["status"], "veraltet")
            self.assertIn("K14", problems(res))
            # K15 points at the same anchor/line: stale too; nothing else is touched
            self.assertEqual(sorted(problems(res)), sorted(k["id"] for k in KANTEN
                                                           if k["beleg"]["datei"] == victim["beleg"]["datei"]
                                                           and k["beleg"]["anker"] == anker))

    def test_mutant_ambiguous_anchor_is_ambiguous_not_guessed(self):
        # K01 alone: remove the real anchor, plant it 2 lines before and 2 lines after the stored line -> equally near: a tie
        victim = next(k for k in KANTEN if k["id"] == "K01")
        rel, anker, real = victim["beleg"]["datei"], victim["beleg"]["anker"], victim["beleg"]["zeile"]
        with tempfile.TemporaryDirectory() as d:
            copy_tree(d)

            def tie(t):
                t = t.replace(anker, "ENTFERNT")
                ls = t.splitlines(True)
                for ln in (real - 2, real + 2):
                    ls[ln - 1] = ls[ln - 1].rstrip("\n") + " # " + anker + "\n"
                return "".join(ls)
            edit(d, rel, tie)
            res = PC.resolve_edge_belege([victim], d)["K01"]
            self.assertEqual((res["treffer"], res["status"]), (2, "mehrdeutig"))
            self.assertIn("K01", problems({"K01": res}))
            edit(d, rel, lambda t: t.replace(" # " + anker, "", 1))             # one copy left: unique again
            res = PC.resolve_edge_belege([victim], d)["K01"]
            self.assertEqual((res["treffer"], res["status"]), (1, "eindeutig"))

    def test_missing_repo_file_is_a_problem_missing_external_is_not(self):
        with tempfile.TemporaryDirectory() as d:        # empty tree: repo files absent
            res = PC.resolve_edge_belege(KANTEN, d)
            for k in KANTEN:
                if os.path.isabs(k["beleg"]["datei"]):
                    self.assertIn(res[k["id"]]["status"], ("extern_fehlt",) + PC.ANKER_OK)
                else:
                    self.assertEqual(res[k["id"]]["status"], "datei_fehlt")


class CatalogShowsResolvedLine(unittest.TestCase):
    def build(self, **kw):
        return PC.build_catalog(os.path.join(WEG2, "launcher.py"), os.path.join(SRT, "environ.py"), CU.CURATED, "t",
                                os.path.join(SRT, "server_args.py"), **kw)

    def test_resolved_line_displayed_stored_line_kept_as_hint(self):
        with tempfile.TemporaryDirectory() as d:
            # a shifted COPY of the tree the catalog is built from (every file an edge points at, same relative layout; AP-G 06.10.:
            # the edges now also point at server_args.py, rank_role.py and host_ledger.py, not only launcher/environ/couplings)
            copy_tree(d)
            for rel in ("python/sglang/srt/weg2/launcher.py", "python/sglang/srt/environ.py", "python/sglang/srt/weg2/refusals.py"):
                t = os.path.join(d, rel)
                os.makedirs(os.path.dirname(t), exist_ok=True)
                shutil.copyfile(os.path.join(REPO_ROOT, rel), t)
            n = 31
            for rel in REPO_FILES:
                edit(d, rel, lambda t: "# pad\n" * n + t)
            cat = PC.build_catalog(os.path.join(d, "python/sglang/srt/weg2/launcher.py"), os.path.join(d, "python/sglang/srt/environ.py"),
                                   CU.CURATED, "t", os.path.join(SRT, "server_args.py"))
            info = cat["kanten"]["beleg_aufloesung"]
            self.assertEqual(info["problem"], [])
            by_id = {e["id"]: e for e in KANTEN}
            seen = 0
            for ent in cat["entries"].values():
                for dep in ent["depends"]:
                    b = dep.get("beleg")
                    if not dep.get("belegt") or not b or os.path.isabs(b["datei"]):
                        continue
                    seen += 1
                    stored = by_id[dep["kante"]]["beleg"]["zeile"]
                    self.assertEqual(b["zeile_hinweis"], stored)
                    self.assertEqual(b["zeile"] - stored, n, dep["kante"])      # shown line = resolved line
                    self.assertIn(b["aufloesung"], PC.ANKER_OK)
            self.assertEqual(seen, 116 - sum(1 for k in KANTEN if os.path.isabs(k["beleg"]["datei"])))

    def test_unresolvable_layout_leaves_the_catalog_as_before(self):
        # ``edges_root`` given but empty tree and a launcher outside the repo layout: no extra fields, no crash
        with tempfile.TemporaryDirectory() as d:
            lp = os.path.join(d, "launcher.py")
            shutil.copyfile(os.path.join(WEG2, "launcher.py"), lp)
            cat = PC.build_catalog(lp, os.path.join(SRT, "environ.py"), CU.CURATED, "t", os.path.join(SRT, "server_args.py"))
            self.assertNotIn("beleg_aufloesung", cat["kanten"])
            for ent in cat["entries"].values():
                for dep in ent["depends"]:
                    if dep.get("beleg"):
                        self.assertNotIn("aufloesung", dep["beleg"])
                        self.assertNotIn("zeile_hinweis", dep["beleg"])


if __name__ == "__main__":
    unittest.main()
