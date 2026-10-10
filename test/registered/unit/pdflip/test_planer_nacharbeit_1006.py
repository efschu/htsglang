"""Nacharbeit nach der Abnahme AP-J, Befund 6: W71 (UUID-gebundener Census) und W64 (kein gemessenes Dual-D-Log) sind im Verdikt-Register des Planers
(``propose_verdict.SUPPLEMENT_CODES``) mit Klasse, ``forcebar=False``, Grund und Quelle, nicht mehr ``LAUNCHER-UNKLASSIFIZIERT``.  ``refusals.REGISTER`` (der Launcher)
bleibt unveraendert (R1).  GPU-frei.

Launcher-Linien (07.10.): derselbe Testcode laeuft auf der 27B-Linie und auf der NF-Linie (``propose_oracle.launcher_line``, aus dem Argumentparser
des Launchers).  Was an die Zeilennummern und das Modul ``dual_w64`` des 27B-Baums gebunden ist (Dual-Form: auf der NF-Linie nicht implementiert),
laeuft nur dort; was ein Anker im Quelltext ist, laeuft auf jedem Baum.  Das Launcher-Register hat je Linie seine eigene Groesse (27B: 22 Codes,
NF: 23, gemessen 2026-10-07): geprueft wird, dass W71/W64 darin fehlen, nicht eine feste Zahl.
"""

from __future__ import annotations

import os
import pathlib
import re
import unittest

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

try:
    from flliper.srt.pdflip import launcher, refusals, xchg_residency
    from flliper.srt.pdflip import propose_oracle as O
    from flliper.srt.pdflip import propose_verdict as PV
except Exception as exc:  # pragma: no cover
    pytest.skip(f"pdflip launcher unavailable: {exc}", allow_module_level=True)

W71_MSG = ("W71 PdFlipXchgResidencyUnarmable: the exchange's predicted VRAM residency does not fit on 1 (card x direction) case(s).  This is spec section 5's arithmetic "
           "over a MEASURED census")
W64_MSG = ("W64 PdFlipTpOperatingPointInfeasible: position tp3 derives weights [20, 12, 8], which PerfCostModel.predict_capacity marks feasible=False against this "
           "boot's budgets [15000, 15000, 15000] -- the weight shards plus the mamba pool plus the reserves do not leave a positive KV pool on at least one rank.")


def _result(exc_type, exc_msg, forced=()):
    return O.DryRunResult(None, exc_type, exc_msg, "PLAN\n", "PLAN\n", [dict(f) for f in forced], ["--x"], "")


class TestSupplement(unittest.TestCase):
    def test_the_launcher_register_is_not_changed_and_has_neither_code(self):
        self.assertGreater(len(refusals.REGISTER), 0)
        codes = {r.code for r in refusals.REGISTER}
        for c in ("W71", "W64", "W71-CENSUS", "W64-OPPOINT", "W64-DUAL-D"):
            self.assertNotIn(c, codes)

    def test_w71_and_w64_are_classified_by_their_own_code(self):
        a = PV.classify_exception("PdFlipXchgResidencyUnarmable", W71_MSG)
        b = PV.classify_exception("PdFlipTpOperatingPointInfeasible", W64_MSG)
        self.assertEqual((a["kind"], a["code"], a["launcher_code"]), ("refusal", "W71-CENSUS", "W71"))
        self.assertEqual((b["kind"], b["code"], b["launcher_code"]), ("refusal", "W64-OPPOINT", "W64"))

    def test_other_launcher_codes_stay_unclassified(self):
        c = PV.classify_exception("PdFlipPCutRecutRefused", "W167 PdFlipPCutRecutRefused: x")
        self.assertEqual(c["code"], "LAUNCHER-UNKLASSIFIZIERT")

    def test_the_verdict_carries_class_not_forceable_reason_and_source(self):
        for exc, msg, code in (("PdFlipXchgResidencyUnarmable", W71_MSG, "W71-CENSUS"), ("PdFlipTpOperatingPointInfeasible", W64_MSG, "W64-OPPOINT")):
            d = PV.build_verdict(3, _result("PdFlipLaunchRefused", "HW-COUNT: x"), _result(exc, msg, forced=[{"code": "HW-COUNT", "text": "HW-COUNT: x"}]))
            last = [v for v in d["verdikte"] if v["level"] == "run"][-1]
            self.assertEqual(last["code"], code)
            self.assertIs(last["forcebar"], False)
            self.assertEqual(last["force_state"], PV.BLOCKED)
            self.assertEqual(last["class"], "nicht_forcebar")
            self.assertTrue(last["class_reason"] and last["consequence"] and last["title"])
            self.assertIn(msg[:40], last["reason"])                      # der Text der Launcher-Meldung, nicht ein Platzhalter
            self.assertRegex(last["quelle"], r"\.py:\d+")
            self.assertTrue(last["ergaenzung"])
            self.assertEqual(d["outcome"], "verweigert")
            self.assertEqual(d["zaehlung"][PV.BLOCKED], 1)

    def test_w64_is_form_neutral_without_the_dual_marker_and_dual_worded_with_it(self):
        """Der Launcher wirft W64 auch ohne dual_layout (Flip/nur TP); den Dual-Wortlaut gibt es nur, wenn die Meldung 'W64-DUAL:' traegt."""
        flip = PV.budget_verdict("W64-OPPOINT", level="run", text=W64_MSG, force_state=PV.BLOCKED)
        self.assertNotIn("Dual", flip["title"])
        self.assertNotIn("dual D log", flip["consequence"] + flip["class_reason"])
        self.assertNotIn("W64-DUAL", flip["reason"])
        self.assertEqual((flip["forcebar"], flip["class"]), (False, "nicht_forcebar"))
        dual_msg = W64_MSG + " W64-DUAL: no measured dual-share D log of m with weights [20, 12, 8] under /x; the model verdict stands"
        dual = PV.budget_verdict("W64-OPPOINT", level="run", text=dual_msg, force_state=PV.BLOCKED)
        self.assertIn("dual D", dual["title"])
        self.assertIn("dual D log", dual["consequence"])
        self.assertEqual((dual["code"], flip["code"]), ("W64-OPPOINT", "W64-OPPOINT"))
        # ohne Text (Fallback auf den Launcher-Wortlaut) bleibt es neutral
        self.assertNotIn("dual", PV.budget_verdict("W64-OPPOINT", level="run")["title"])
        # durch classify/build_verdikt: Flip-W64 und Dual-W64 tragen denselben Code
        for msg in (W64_MSG, dual_msg):
            d = PV.build_verdict(3, _result("PdFlipLaunchRefused", "HW-COUNT: x"), _result("PdFlipTpOperatingPointInfeasible", msg, forced=[{"code": "HW-COUNT", "text": "HW-COUNT: x"}]))
            last = [v for v in d["verdikte"] if v["level"] == "run"][-1]
            self.assertEqual(last["code"], "W64-OPPOINT")
            self.assertEqual("dual D" in last["title"], "W64-DUAL:" in msg)

    def test_w64_three_dual_cases_each_get_their_own_wording(self):
        """Nacharbeit 1006 Runde 6, Befund 3: kein Dual / Dual ohne Messung / Dual mit Messung INFEASIBLE / Meldung ohne Gewichtsvektor sind vier Woerter,
        nicht zwei (launcher.py:17235, :17242-17243, :17343-17347, dual_w64.py:117)."""
        neutral = PV.budget_verdict("W64-OPPOINT", level="run", text=W64_MSG, force_state=PV.BLOCKED)
        without = PV.budget_verdict("W64-OPPOINT", level="run", force_state=PV.BLOCKED,
                          text=W64_MSG + " | W64-DUAL: no measured dual-share D log of m with weights [20, 12, 8] under /x; the model verdict stands")
        measured_line = ("W64-DUAL MEASURED (/ev/boot_x.D.log, D weights [20, 12, 8], cell 18432 B): r0 budget=15000 - measured posts=14853 => 9000 tokens; "
                         "r1 budget=15000 - measured posts=15100 => -5000 tokens  <-- BELOW 1024 -> INFEASIBLE")
        measured = PV.budget_verdict("W64-OPPOINT", level="run", force_state=PV.BLOCKED, text=W64_MSG + " | " + measured_line)
        no_vector = PV.budget_verdict("W64-OPPOINT", level="run", force_state=PV.BLOCKED,
                                 text="W64 PdFlipTpOperatingPointInfeasible: x | W64-DUAL: the refusal names no weight vector; the model verdict stands")
        for v in (neutral, without, measured, no_vector):
            self.assertEqual((v["code"], v["forcebar"], v["class"]), ("W64-OPPOINT", False, "nicht_forcebar"))
        # kein Dual: formneutral, ohne Dual-Wort
        self.assertNotIn("Dual", neutral["title"])
        self.assertNotIn("dual D log", neutral["consequence"])
        # Dual ohne Messung: der Wortlaut "without a measured dual D log"
        self.assertIn("without a measured dual D log", without["title"])
        self.assertIn("dual D log", without["consequence"])
        # Dual mit Messung INFEASIBLE: sagt, dass gemessen wurde, nie "without a measured dual D log"
        self.assertIn("measured dual D log confirms", measured["title"])
        self.assertNotIn("without a measured", measured["title"] + measured["consequence"] + measured["class_reason"])
        self.assertIn("confirms", measured["consequence"])
        self.assertRegex(measured["quelle"], r"17343-17347.*dual_w64\.py:117")
        # Meldung ohne Gewichtsvektor: keine Suche, also nicht "without a measured dual D log"
        self.assertNotIn("without a measured", no_vector["title"] + no_vector["consequence"])
        self.assertIn("without weight vector", no_vector["title"])
        self.assertRegex(no_vector["quelle"], r"17235")
        self.assertEqual(len({neutral["title"], without["title"], measured["title"], no_vector["title"]}), 4)

    @unittest.skipUnless(O.pdflip_module_exists("dual_w64"),
                         "pdflip/dual_w64.py does not exist in this tree (NF launcher line: the Dual form is not implemented there); "
                         "the W64-DUAL marker texts are the 27B line's")
    def test_the_w64_dual_marker_texts_exist_in_the_launcher_sources(self):
        """Die drei Marker, nach denen _supp_view unterscheidet, stehen so im Quelltext (nicht aus dem Gedaechtnis)."""
        import inspect

        from flliper.srt.pdflip import dual_w64
        lsrc, dsrc = inspect.getsource(launcher), inspect.getsource(dual_w64)
        self.assertIn('"W64-DUAL: the refusal names no weight vector', lsrc)
        self.assertIn('"W64-DUAL: no measured dual-share D log of', lsrc)
        self.assertIn("W64-DUAL MEASURED (", dsrc)
        self.assertIn('"INFEASIBLE"', dsrc)

    def test_verdikt_without_text_falls_back_to_the_launcher_wording(self):
        v = PV.budget_verdict("W71-CENSUS", level="run")
        self.assertIn("W71 PdFlipXchgResidencyUnarmable", v["reason"])
        self.assertIs(v["forcebar"], False)

    def test_the_cited_anchors_exist_in_the_launcher_sources_of_this_tree(self):
        """Der zitierte Text steht im Quelltext DIESES Baums (Anker, ohne Zeilennummer: die Nummern sind die der 27B-Linie).  Die Dual-Anker
        (W64-DUAL, ``dual_layout``) nur dort, wo es die Dual-Form gibt."""
        xsrc = pathlib.Path(xchg_residency.__file__).read_text(encoding="utf-8")
        lsrc = pathlib.Path(launcher.__file__).read_text(encoding="utf-8")
        self.assertIn("W71 PdFlipXchgResidencyUnarmable: the exchange's predicted VRAM residency", xsrc)
        self.assertIn("--pdflip-weight-source ring", xsrc)
        self.assertIn("def load_census", xsrc)
        self.assertIn("W64 PdFlipTpOperatingPointInfeasible: position", lsrc)
        has_dual = O.pdflip_module_exists("dual_w64")
        # the two probes agree: a launcher with the Dual form knows --dual-priority and has dual_w64.py, one without has neither
        self.assertEqual(has_dual, O.launcher_knows("--dual-priority"))
        for anchor in ("W64-DUAL: no measured dual-share D log", "if mine and dual_layout:",
                       "if not dual_layout or not str(refusal).startswith"):
            self.assertEqual(anchor in lsrc, has_dual, anchor)

    @unittest.skipUnless(O.launcher_line() == O.LINE_27B,
                         "the cited line numbers (xchg_residency.py:711-723, launcher.py:16986-17352) are those of the 27B launcher line; "
                         "this tree's launcher line is %r (the anchors are proved by test_the_cited_anchors_exist_in_the_launcher_sources_of_this_tree)"
                         % O.launcher_line())
    def test_the_cited_lines_hold_the_launcher_text(self):
        """Die Quellenangaben sind keine Behauptung: der zitierte Text steht an der genannten Stelle des Quelltexts dieses Baums."""
        def lines(mod):
            return pathlib.Path(mod.__file__).read_text(encoding="utf-8").split("\n")

        xl, ll = lines(xchg_residency), lines(launcher)

        def span(src, a, b):
            return " ".join(src[a - 1:b])

        self.assertIn("W71 PdFlipXchgResidencyUnarmable: the exchange's predicted VRAM residency", span(xl, 711, 723))
        self.assertIn("--pdflip-weight-source ring", span(xl, 711, 723))
        self.assertIn("def load_census", span(xl, 313, 316))
        self.assertIn("W64 PdFlipTpOperatingPointInfeasible: position", span(ll, 16986, 16992))
        self.assertIn("W64-DUAL: no measured dual-share D log", span(ll, 17244, 17245))
        self.assertIn("if mine and dual_layout:", span(ll, 17342, 17342))        # Override nur im Dual ...
        self.assertIn("raise PdFlipLaunchRefused(mine[0]", span(ll, 17351, 17354))  # ... die Verweigerung gilt in jeder Form
        self.assertIn("if not dual_layout or not str(refusal).startswith", span(ll, 17233, 17233))
        for w in PV.SUPPLEMENT_CODES.values():
            for m in re.finditer(r"(\w+/)?[\w.]+\.py:(\d+)(?:-(\d+))?", w["quelle"]):
                self.assertGreater(int(m.group(2)), 0)

    def test_the_dashboard_shows_the_supplement_code_as_a_not_forceable_row(self):
        """Ohne Registerzeile baut das Dashboard die Zeile aus dem Verdikt (``ProfilEditor._register_row``): nicht forcebar, mit Titel und Folge."""
        import sys

        tools = str(pathlib.Path(launcher.__file__).resolve().parents[4] / "tools" / "rig_dashboard")
        if tools not in sys.path:
            sys.path.insert(0, tools)
        from rigdash import profil as P

        v = PV.budget_verdict("W71-CENSUS", level="run", text=W71_MSG, force_state=PV.BLOCKED, extra={"launcher_code": "W71"})
        row = P.ProfilEditor._register_row({"code": "W71-CENSUS", "verdict": v}, {})
        self.assertEqual((row["klass"], row["forcebar"]), ("nicht_forcebar", False))
        self.assertIn("W71", row["why_class"])
        self.assertTrue(row["consequence"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
