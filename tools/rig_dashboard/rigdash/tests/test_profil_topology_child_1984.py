"""Auftrag 1984 (C): F2 sauber -- ``plan_topology`` im Kindprozess mit der Kopplungs-Python.

Befund (Browsertest 1979, F2): der Trockenlauf mit N != 3 Karten rechnete ``topology.plan_topology`` im Dashboard-Prozess; fuer N != 3
importiert es ``sglang.srt.weg2.weight_exchange_region`` -> ``No module named 'sglang'`` -> HTTP 500 bzw. (gepflastert) "nicht geprueft".
Jetzt: der Kopplungs-Worker (Kindprozess mit der sglang-Umgebung) beantwortet ``{"what": "topology", "n": N}``; der Editor fragt ihn zuerst.
Gepinnt:

* Worker: ``refused`` traegt den Text einer TopologyRefused, ``None`` heisst "durchgelassen"; ein Import-Fehler im Kind ist ``ok: false``
  mit Grund; der Worker bleibt oben, auch wenn ``profile_couplings`` im Baum fehlt (die Topologie braucht es nicht);
* Editor.dry_run: Kind-Urteil wird zu HW-COUNT bzw. HW-TOPOLOGY mit Quelle; ohne Kind (nicht ok) greift die alte Rechnung im Prozess, und
  scheitert auch die am Import, bleibt es die benannte Notiz "nicht geprueft" (nie 500);
* App verdrahtet den Editor mit ``couplings.topology``.
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import unittest
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from rigdash import profil_recompute as R  # noqa: E402
from rigdash import server as S  # noqa: E402
from rigdash.tests import test_profil_930 as P9  # noqa: E402

REAL_TREE = os.environ.get("COUPLINGS_TREE")

#: ein Baum, in dem sglang importierbar ist (Stubs): topology verweigert N=5 (Wert-Ablehnung), N=9 (keine Topologie), wirft ImportError fuer N=7
STUB_FILES = {
    "sglang/__init__.py": "",
    "sglang/srt/__init__.py": "",
    "sglang/srt/planner/__init__.py": "",
    "sglang/srt/planner/profile_couplings.py": "def run(req):\n    return {'ok': True, 'result': {'echo': req}}\n",
    "sglang/srt/weg2/__init__.py": "",
    "sglang/srt/weg2/topology.py": '''
class TopologyRefused(RuntimeError):
    pass


def plan_topology(n_cards, ctx=None):
    if n_cards == 7:
        from sglang.srt.weg2 import weight_exchange_region      # fehlt: ImportError im Kind
    if n_cards == 5:
        raise TopologyRefused("HW-COUNT: 5 cards would be P = (2,3), proven on metal only for N in [3]")
    if n_cards == 9:
        raise TopologyRefused("HW-TOPOLOGY: no topology for 9 cards")
    return object()
''',
}


def make_tree(root, files=STUB_FILES):
    for rel, text in files.items():
        p = os.path.join(root, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as fh:
            fh.write(text)
    return root


class TestWorkerTopology(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="pf1984t_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.tree = make_tree(os.path.join(self.tmp, "python"))

    def svc(self, tree=None):
        s = R.CouplingsService(tree or self.tree, python=sys.executable, start_timeout_s=60)
        self.addCleanup(s.close)
        return s

    def test_refusal_text_passes_through_and_acceptance_is_none(self):
        s = self.svc()
        ok = s.topology(3)
        self.assertEqual(ok, {"ok": True, "refused": None})
        r5 = s.topology(5)
        self.assertTrue(r5["ok"])
        self.assertIn("HW-COUNT", r5["refused"])
        self.assertIn("HW-TOPOLOGY", s.topology(9)["refused"])

    def test_an_import_error_in_the_child_is_ok_false_with_the_reason(self):
        r = self.svc().topology(7)
        self.assertFalse(r["ok"])
        self.assertRegex(r["error"], "ImportError|ModuleNotFoundError")
        self.assertIn("weight_exchange_region", r["error"])

    def test_one_worker_serves_topology_and_couplings(self):
        s = self.svc()
        s.topology(5)
        self.assertEqual(s.request({"what": "bars", "hardware": {}, "model": {}})["result"]["echo"]["what"], "bars")
        self.assertEqual(s.starts, 1)

    def test_topology_works_without_profile_couplings_in_the_tree(self):
        tree = make_tree(os.path.join(self.tmp, "python2"), {k: v for k, v in STUB_FILES.items() if "profile_couplings" not in k})
        s = self.svc(tree)
        self.assertIn("HW-COUNT", s.topology(5)["refused"])
        bars = s.request({"what": "bars"})
        self.assertFalse(bars["ok"])
        self.assertIn("profile_couplings", bars["error"])

    def test_no_python_or_tree_is_named(self):
        r = R.CouplingsService(None).topology(5)
        self.assertFalse(r["ok"])
        self.assertIn("no planner tree", r["error"])


class TestDryRunUsesTheChild(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="pf1984d_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.ed, _rel, _usr = P9.editor(self.tmp)
        self.doc = self.ed.load("release", "demo")["doc"]
        self.calls = []
        # der Planer-Baum des Dashboards hat KEIN sglang: plan_topology wirft fuer N != 3 den ImportError des Befunds F2
        ci, tp = self.ed.kp._mods()

        def in_process(n, *a, **kw):
            if n != 3:
                raise ModuleNotFoundError("No module named 'sglang'")
            return object()
        self.ed.kp._mods = lambda: (ci, SimpleNamespace(plan_topology=in_process, TopologyRefused=tp.TopologyRefused))

    def cards(self, n):
        return [P9.RIG[i % 3] for i in range(n)]

    def dry(self, n):
        return self.ed.dry_run(self.doc, self.cards(n))

    def codes(self, d):
        return [r["code"] for r in d["rejections"]]

    def test_the_childs_value_refusal_becomes_hw_count_with_its_source(self):
        def child(n):
            self.calls.append(n)
            return {"ok": True, "refused": "HW-COUNT: 4 cards would be P = (2,2), proven on metal only for N in [3]"}
        self.ed.topology = child
        d = self.dry(4)
        self.assertEqual(self.calls, [4])
        top = [r for r in d["rejections"] if r["source"] == "weg2/topology.plan_topology"]
        self.assertEqual([r["code"] for r in top], ["HW-COUNT"])
        self.assertFalse([n for n in d["notes"] if "not checked" in n])

    def test_the_childs_missing_topology_is_hw_topology(self):
        self.ed.topology = lambda n: {"ok": True, "refused": "no topology for 9 cards"}
        top = [r for r in self.dry(6)["rejections"] if r["source"] == "weg2/topology.plan_topology"]
        self.assertEqual([r["code"] for r in top], ["HW-TOPOLOGY"])

    def test_the_child_letting_the_count_through_adds_no_topology_rejection_and_no_note(self):
        self.ed.topology = lambda n: {"ok": True, "refused": None}
        d = self.dry(4)
        self.assertFalse([r for r in d["rejections"] if r["source"] == "weg2/topology.plan_topology"])
        self.assertFalse([n for n in d["notes"] if "not checked" in n])

    def test_without_a_child_the_old_path_and_its_note_remain(self):
        self.ed.topology = None
        d = self.dry(4)                                  # N=4: im Prozess ImportError (kein sglang) -> Notiz, kein 500
        self.assertTrue([n for n in d["notes"] if "Topology for 4 card(s) not checked" in n], d["notes"])

    def test_a_child_that_cannot_work_falls_back_and_names_both(self):
        self.ed.topology = lambda n: {"ok": False, "error": "no planner tree with planner/profile_couplings.py"}
        d = self.dry(4)
        note = [n for n in d["notes"] if "not checked" in n]
        self.assertTrue(note, d["notes"])
        self.assertIn("no planner tree", note[0])         # der Grund des Kindes steht in der Notiz
        self.assertTrue(d["ok"])

    def test_a_child_that_raises_is_a_note_not_a_500(self):
        def boom(n):
            raise RuntimeError("worker kaputt")
        self.ed.topology = boom
        d = self.dry(4)
        self.assertTrue([n for n in d["notes"] if "worker kaputt" in n], d["notes"])

    def test_n3_without_a_child_is_still_judged_in_process(self):
        self.ed.topology = lambda n: {"ok": False, "error": "kein Kind"}
        d = self.dry(3)
        self.assertFalse([n for n in d["notes"] if "not checked" in n], d["notes"])


class TestAppWiring(unittest.TestCase):
    def test_the_app_hands_the_editor_the_couplings_topology(self):
        with tempfile.TemporaryDirectory() as d:
            ns = argparse.Namespace(log_glob=[], docker_ssh="", docker_host_prefix="", front=[], gpuq="", state_dir="", release_profile=[],
                                    image_changes=os.path.join(d, "ic.json"), features=os.path.join(d, "f.json"), features_repo=d,
                                    edition="release")
            app = S.App(ns)
            self.assertTrue(callable(app.profil.topology))
            seen = []
            app.couplings = type("C", (), {"topology": lambda self, n: seen.append(n) or {"ok": True, "refused": None}})()
            self.assertEqual(app.profil.topology(5), {"ok": True, "refused": None})
            self.assertEqual(seen, [5])


@unittest.skipUnless(REAL_TREE and os.path.isdir(REAL_TREE), "COUPLINGS_TREE (<py-Baum>/python) nicht gesetzt")
class TestRealTree(unittest.TestCase):
    def test_n_not_3_is_judged_in_the_child_with_the_real_planner(self):
        py = os.environ.get("RIGDASH_COUPLINGS_PYTHON") or R.DEFAULT_PYTHON
        if not os.path.isfile(py):
            self.skipTest("Kopplungs-Python %s fehlt" % py)
        s = R.CouplingsService(REAL_TREE, python=py)
        self.addCleanup(s.close)
        out = {n: s.topology(n) for n in (2, 3, 4, 5, 6)}
        for n, r in out.items():
            self.assertTrue(r["ok"], (n, r))
        self.assertIsNone(out[3]["refused"])
        self.assertTrue(any(out[n]["refused"] for n in (2, 4, 5, 6)), out)
        for n in (2, 4, 5, 6):
            if out[n]["refused"]:
                self.assertRegex(out[n]["refused"], r"HW-|N=|cards|Karte")


if __name__ == "__main__":
    unittest.main()
