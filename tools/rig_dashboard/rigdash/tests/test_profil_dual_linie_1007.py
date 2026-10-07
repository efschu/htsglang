"""NF-Linie 07.10. (Nutzerentscheid "2 nein"): der Editor bietet die Betriebsform Dual nur an, wenn der Planer-Baum die Dual-Module traegt.

Gepinnt (die Sonde ``profil.dual_line_probe`` ist eine Dateisonde -- ``weg2/dual_layout_plan.py`` und ``weg2/dual_green.py`` --, nie ein Baum-SHA; sie wird in diesen
Faellen gemockt, so dass derselbe Test auf beiden Linien beide Zweige prueft):

  * ohne Dual-Module (NF-Linie): ``formen`` der Seite hat weder ``dual`` noch eine Dual-ENV-Tabelle (kein Schluessel ``dual``), ``dual_verfuegbar`` ist False,
    ``ProfilEditor.forms()`` nennt ``dual`` nicht; ``POST /api/profil/propose`` mit ``form=dual`` antwortet 400 "Dual auf dieser Linie
    nicht verfuegbar", OHNE das Orakel zu fragen (vorher endete der Aufruf im Kindprozess mit ImportError); die anderen Formen laufen unveraendert.
  * mit Dual-Modulen (27B-Linie): die Antwort ist die bisherige -- vier Formen, Dual-ENV-Tabelle, ``dual`` bedient ``propose``.
  * die Sonde selbst: beide Dateien noetig, eine allein genuegt nicht, kein Baum = nicht verfuegbar, Datei- und nicht Branch-Ebene.
  * Mutante: wuerde ``propose`` die Pruefung auslassen, kaeme die Anfrage beim Orakel an (der Test auf ``calls == []`` wird rot).
"""

import copy
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)

from rigdash import profil as P  # noqa: E402
from rigdash import profil_planer as PL  # noqa: E402
from test_profil_orakel_apd_1006 import (  # noqa: E402
    FakeOracle, ProposeRoute, ProposeStartprofil, REPO_CATALOG, _hw_profile, _rows, editor)

REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))


def _tree_with(tmp, modules):
    """A planner tree skeleton (``sglang/srt/weg2`` with the given module files)."""
    base = os.path.join(tmp, "sglang", "srt", "weg2")
    os.makedirs(base)
    for m in modules:
        with open(os.path.join(base, m), "w", encoding="utf-8") as fh:
            fh.write("# probe file\n")
    return tmp


class Probe(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="dl1007_")
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_both_dual_modules_are_needed_and_nothing_else_counts(self):
        self.assertEqual(P.DUAL_MODULES, ("dual_layout_plan.py", "dual_green.py"))
        for name, mods, want in (("none", (), False), ("plan_only", ("dual_layout_plan.py",), False), ("green_only", ("dual_green.py",), False),
                                 ("both", ("dual_layout_plan.py", "dual_green.py"), True),
                                 ("share_only", ("dual_share.py", "propose_dual.py"), False)):          # propose_dual.py exists on the NF line: it does NOT decide
            t = _tree_with(os.path.join(self.tmp, name), mods)
            self.assertIs(P.dual_line_probe(t), want, name)
        self.assertIs(P.dual_line_probe(None), False)
        self.assertIs(P.dual_line_probe(""), False)
        self.assertIs(P.dual_line_probe(os.path.join(self.tmp, "does-not-exist")), False)

    def test_the_probe_reads_the_tree_of_this_checkout_by_files(self):
        py = os.path.join(REPO_ROOT, "python")
        has = all(os.path.isfile(os.path.join(py, "sglang", "srt", "weg2", n)) for n in ("dual_layout_plan.py", "dual_green.py"))
        self.assertIs(P.dual_line_probe(py), has)


class PageData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(REPO_CATALOG, encoding="utf-8") as fh:
            cls.entries = json.load(fh)["entries"]

    def test_without_dual_the_page_has_no_dual_form_and_no_dual_table(self):
        ui = PL.ui_info(("flip", "tp", "single"), self.entries, True, dual=False)
        self.assertEqual([f["id"] for f in ui["formen"]], ["einzel", "tp", "flip"])
        self.assertNotIn("dual", ui)
        self.assertIs(ui["dual_verfuegbar"], False)
        self.assertNotIn("dual", {f["backend"] for f in ui["formen"]})
        self.assertNotIn("SGLANG_WEG2_DUAL_SHARE_GREEN_TABLE", json.dumps(ui))             # the Dual-ENV table is not in the page data at all

    def test_with_dual_the_answer_is_the_old_one(self):
        ui = PL.ui_info(("flip", "tp", "dual", "single"), self.entries, True)               # dual defaults to True: callers of the 27B line are unchanged
        self.assertEqual([f["id"] for f in ui["formen"]], ["einzel", "tp", "flip", "dual"])
        self.assertIs(ui["dual_verfuegbar"], True)
        self.assertEqual(sorted(ui["dual"]["werte"]), sorted(PL.DUAL_ENV.values()))
        self.assertEqual(ui, PL.ui_info(("flip", "tp", "dual", "single"), self.entries, True, dual=True))


class EditorBothLines(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="dl1007_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        base = ProposeStartprofil("test_the_request_to_the_child")
        base.tmp = self.tmp
        base.setUp()
        self.addCleanup(base.tearDown)
        self.base, self.ed, self.orc = base, base.ed, base.orc
        ans = copy.deepcopy(base.answer)
        ans["vorschlag"].update(form="dual", n=2)
        self.orc.propose = ans

    def _dual(self):
        return self.ed.propose(self.base._body(form="dual"))

    def test_nf_line_dual_is_not_offered_and_not_proposed(self):
        with mock.patch.object(P, "dual_line_probe", return_value=False):
            self.assertFalse(self.ed.dual_available())
            self.assertEqual(self.ed.forms(), ("flip", "tp", "single"))
            planer = self.ed.list()["planer"]
            self.assertEqual([f["id"] for f in planer["formen"]], ["einzel", "tp", "flip"])
            self.assertNotIn("dual", planer)
            self.assertIs(planer["dual_verfuegbar"], False)
            with self.assertRaises(P.ProfilError) as cm:
                self.ed.propose(self.base._body(form="dual"))
            self.assertEqual(str(cm.exception), "Dual auf dieser Linie nicht verfuegbar")
            self.assertEqual(self.orc.calls, [], "the child must not be asked (ImportError in the child before this fix)")
            r = self.ed.propose(self.base._body(form="flip"))                                  # the other forms are untouched
            self.assertTrue(r["ok"], r)
            self.assertEqual(len(self.orc.calls), 1)

    def test_nf_line_without_oracle_still_answers_dual_by_name(self):
        with mock.patch.object(P, "dual_line_probe", return_value=False):
            self.ed.oracle = None
            with self.assertRaises(P.ProfilError) as cm:
                self.ed.propose(self.base._body(form="dual"))
            self.assertEqual(str(cm.exception), P.DUAL_FEHLT)

    def test_27b_line_dual_is_offered_and_proposed_as_before(self):
        with mock.patch.object(P, "dual_line_probe", return_value=True):
            self.assertTrue(self.ed.dual_available())
            self.assertEqual(self.ed.forms(), P.ProfilEditor.FORMS)
            planer = self.ed.list()["planer"]
            self.assertEqual([f["id"] for f in planer["formen"]], ["einzel", "tp", "flip", "dual"])
            self.assertIs(planer["dual_verfuegbar"], True)
            self.assertEqual(sorted(planer["dual"]["werte"]), sorted(PL.DUAL_ENV.values()))
            r = self._dual()
            self.assertTrue(r["ok"], r)
            self.assertEqual(self.orc.calls[0][1]["form"], "dual")

    def test_the_editor_decides_by_its_own_tree(self):
        """Without the mock: the fixture tree carries no Dual module, a tree copy that carries both does."""
        self.assertFalse(self.ed.dual_available())
        t = _tree_with(os.path.join(self.tmp, "t27"), ("dual_layout_plan.py", "dual_green.py"))
        self.ed.tree = t
        self.assertTrue(self.ed.dual_available())


class ProposeRouteDualLine(unittest.TestCase):
    serve = ProposeRoute.serve
    call = ProposeRoute.call

    def test_http_400_with_a_clear_message_on_the_nf_line(self):
        port, base = self.serve()
        with mock.patch.object(P, "dual_line_probe", return_value=False):
            st, txt = self.call(port, "/api/profil/propose", base._body(form="dual"))
            self.assertEqual(st, 400, txt[:300])
            js = json.loads(txt)
            self.assertIs(js["ok"], False)
            self.assertEqual(js["error"], "Dual auf dieser Linie nicht verfuegbar")
            self.assertNotIn("ImportError", txt)
            self.assertEqual(base.orc.calls, [])
            st, txt = self.call(port, "/api/profil/propose", base._body(form="flip"))
            self.assertEqual(st, 200, txt[:300])

    def test_http_dual_goes_through_on_the_27b_line(self):
        port, base = self.serve()
        base.orc.propose = copy.deepcopy(base.answer)
        base.orc.propose["vorschlag"].update(form="dual", n=2)
        with mock.patch.object(P, "dual_line_probe", return_value=True):
            st, txt = self.call(port, "/api/profil/propose", base._body(form="dual"))
            self.assertEqual(st, 200, txt[:300])
            self.assertEqual(json.loads(txt)["form"], "dual")


if __name__ == "__main__":
    unittest.main()
