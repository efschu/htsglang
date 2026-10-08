"""Nacharbeit nach der Abnahme AP-J (done/planer-abnahme-1006.md, Abschnitt "Neu aus AP-J"): fuenf der sechs Befunde im Dashboard (der sechste, W71/W64 im
Verdikt-Register, steht in test/registered/unit/pdflip/test_planer_nacharbeit_1006.py).

  F1  Verdikt-Chip: nach "Neu pruefen" steht der Chip eines uebersteuerten Werts auf dem Urteil des Trockenlaufs; "ungeprueft seit Ihrer Aenderung" nur bis zum naechsten Lauf.
  F2  Seed-Zeile ``--pp-stage-ratio (Seed)``: nie "not set", wo das Profil den Wert setzt; als Aenderung zaehlt nur, was im argv steht.
  F3  Laufbericht: nimmt den Oracle-Lauf des Vorschlags auf, wenn es keinen Trockenlauf gibt; ein Trockenlauf geht vor.
  F4  Hardware-Issue-Text: die UUID steht nie drin (Tabelle und Erklaertext sagen dasselbe).
  F5  Kartenauswahl im synthetischen Inventar hat ein aria-label.
"""

import json
import os
import shutil
import re
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)

from rigdash import hwprofil  # noqa: E402
from rigdash import profil as P  # noqa: E402
from test_profil_issue_laufbericht_1006 import Base, RIG  # noqa: E402
from test_profil_planer_aph1_1006 import NODE, STATIC, run_node  # noqa: E402


@unittest.skipUnless(NODE, "node fehlt")
class F1ChipNachNeuPruefen(unittest.TestCase):
    BODY = """
const V = (code, o) => Object.assign({ code, level: "run", forcebar: true, force_state: "force", reason: code + " reason", consequence: "k", values: ["--pp-stage-ratio"] }, o || {});
const nutzer = base({ origin: "nutzer", value: "1,2,3" });
const W = { key: "flag:--pp-stage-ratio", value: "9,9,9", state: "vorgeschlagen", changed: true, verdikte: [] };
// vor dem Lauf: Vorschlag vorhanden, Wert uebersteuert, kein Trockenlauf seit der Aenderung
out.vor = PX.verdiktOf(nutzer, ctx({ prop: { values: [W] }, vsrc: "prop", dry: null })).id;
// nach "Neu pruefen" (profil.js doDry: st.dry gesetzt, vsrc "dry"): das Urteil des Trockenlaufs
out.nachForce = PX.verdiktOf(nutzer, ctx({ prop: { values: [W] }, vsrc: "dry", dry: { verdikte: [V("W40")] } })).id;
// der Lauf ging durch (outcome "geht"), ein Hinweis betrifft nur einen anderen Wert
out.nachGeht = PX.verdiktOf(nutzer, ctx({ prop: { values: [W] }, vsrc: "dry", dry: { oracle: { outcome: "geht" }, verdikte: [V("H1", { level: "note", forcebar: null, force_state: "note", values: ["--anderer"] })] } })).id;
out.nachVerweigert = PX.verdiktOf(nutzer, ctx({ prop: { values: [W] }, vsrc: "dry", dry: { verdikte: [V("X", { forcebar: false, force_state: "is_blocked" })] } })).id;
// ohne Vorschlag, nur Trockenlauf
out.nurDry = PX.verdiktOf(nutzer, ctx({ prop: null, vsrc: "dry", dry: { oracle: { outcome: "geht" }, verdikte: [] } })).id;
// jede neue Aenderung leert den Trockenlauf (doEdit: st.dry = null, vsrc "prop"): wieder "ungeprueft seit Ihrer Aenderung"
out.nachEdit = PX.verdiktOf(nutzer, ctx({ prop: { values: [W] }, vsrc: "prop", dry: null })).label;
out.html = PX.renderRow(nutzer, ctx({ prop: { values: [W] }, vsrc: "dry", dry: { verdikte: [V("W40")] } }));
"""

    @classmethod
    def setUpClass(cls):
        cls.o = run_node(cls.BODY)

    def test_before_the_run_the_chip_says_unchecked_since_your_change(self):
        self.assertEqual(self.o["vor"], "alt")
        self.assertEqual(self.o["nachEdit"], "unchecked since your change")

    def test_after_recheck_the_chip_is_the_verdict_of_the_dry_run(self):
        self.assertEqual((self.o["nachForce"], self.o["nachGeht"], self.o["nachVerweigert"], self.o["nurDry"]), ("force", "geht", "verweigert", "geht"))

    def test_the_row_after_recheck_shows_the_verdict_and_stays_overridden(self):
        h = self.o["html"]
        self.assertIn("pfx-v-force", h)
        self.assertNotIn("pfx-v-alt", h)
        self.assertIn("pfx-z-uebersteuert", h)                 # der Zustand "von Ihnen uebersteuert" bleibt: er sagt, wer den Wert gesetzt hat


@unittest.skipUnless(NODE, "node fehlt")
class F6NieGehtWasNichtBeurteiltIst(unittest.TestCase):
    """Runde 6, Befund 1 (Runde 7: ok_with_force und SingleCard stehen jetzt in der Tabelle von chipFor, Klasse F7): der Launcher bricht bei der ersten Verweigerung ab (launcher.py:17349-17352); W64 urteilt gegen die Budgets, aus denen
    --rank-gpu-memory-mib / --user-reserve-mib stammen.  Ein Wert, den kein Verdikt nennt, ist darum nach einem verweigerten Lauf NICHT 'geht'."""
    BODY = """
const V = (code, o) => Object.assign({ code, level: "run", forcebar: false, force_state: "is_blocked", reason: code + " reason", consequence: "k", values: [] }, o || {});
const w64 = V("W64-OPPOINT", { values: [], durchgelassen: false });
const reserve = base({ name: "--user-reserve-mib", key: "flag:--user-reserve-mib", origin: "nutzer", value: "9999" });
const mib = base({ name: "--rank-gpu-memory-mib", key: "flag:--rank-gpu-memory-mib", value: "15000" });
const W = (key) => ({ key, value: "1", state: "vorgeschlagen", changed: true, verdikte: [] });
const dry = (outcome, vs) => ({ oracle: { outcome }, verdikte: vs });
const id = (row, c) => PX.verdiktOf(row, ctx(c));
// Trockenlauf (frisch, "Neu pruefen"): W64 verweigert, kein Verdikt nennt den Wert -> nicht beurteilt, auch der uebersteuerte
out.dryReserve = id(reserve, { prop: { values: [W("flag:--user-reserve-mib")] }, vsrc: "dry", dry: dry("verweigert", [w64]) });
out.dryMib = id(mib, { prop: null, vsrc: "dry", dry: dry("verweigert", [w64]) });
// Absturz: ebenso; ok_with_force (Runde 7): der Lauf ging DURCH, der Wert ist beurteilt: geht
out.dryAbsturz = id(mib, { prop: null, vsrc: "dry", dry: dry("crash", [V("ABSTURZ", { level: "crash", durchgelassen: false })]) }).id;
out.dryForce = id(mib, { prop: null, vsrc: "dry", dry: dry("ok_with_force", [V("W40", { forcebar: true, force_state: "force", values: ["--anderer"], durchgelassen: true })]) }).id;
// der Lauf ging durch: geht
out.dryGeht = id(mib, { prop: null, vsrc: "dry", dry: dry("geht", []) }).id;
// der Wert wird ausdruecklich genannt: sein eigenes Urteil gilt, nicht 'nicht beurteilt'
out.genannt = id(mib, { prop: null, vsrc: "dry", dry: dry("verweigert", [V("W64-OPPOINT", { values: ["--rank-gpu-memory-mib"] })]) }).id;
// ausdruecklich als ok genannt (force_state 'geht')
out.okGenannt = id(mib, { prop: null, vsrc: "dry", dry: dry("verweigert", [w64, V("FIT", { level: "fit", forcebar: null, force_state: "geht", values: ["--rank-gpu-memory-mib"] })]) }).id;
// Quelle Vorschlag: die Verdikte des Vorschlags tragen den Ausgang des Oracle-Laufs
const prop = (outcome, vs) => ({ values: [W("flag:--rank-gpu-memory-mib")], verdict: { outcome, verdikte: vs } });
out.propRefused = id(mib, { prop: prop("verweigert", [w64]), vsrc: "prop" });
out.PropOk = id(mib, { prop: prop("geht", []), vsrc: "prop" }).id;
out.PropSingle = id(mib, { prop: prop("passt", []), vsrc: "prop" }).id;      // Runde 7: Planer-Rechnung, nie "geht"
out.propOhneAusgang = id(mib, { prop: { values: [W("flag:--rank-gpu-memory-mib")] }, vsrc: "prop" }).id;
out.html = PX.renderRow(reserve, ctx({ prop: null, vsrc: "dry", dry: dry("verweigert", [w64]) }));
"""

    @classmethod
    def setUpClass(cls):
        cls.o = run_node(cls.BODY)

    def test_a_refused_run_leaves_unnamed_values_not_judged_never_goes(self):
        for k in ("dryReserve", "dryMib", "propRefused"):
            self.assertEqual(self.o[k]["id"], "nichtbeurteilt", k)
            self.assertEqual(self.o[k]["label"], "not judged (run refused: W64-OPPOINT)", k)
        self.assertEqual(self.o["dryAbsturz"], "nichtbeurteilt")

    def test_goes_only_when_the_run_went_through_or_the_value_is_named_ok(self):
        self.assertEqual((self.o["dryGeht"], self.o["PropOk"]), ("geht", "geht"))
        self.assertEqual(self.o["dryForce"], "geht")             # Runde 7, Befund 1: ein Lauf, der mit --force durchlief, hat alle Werte gesehen
        self.assertEqual(self.o["PropSingle"], "planerpasst")    # Runde 7, Befund 2: kein Launcher-Lauf, also nicht "geht"
        self.assertEqual(self.o["okGenannt"], "geht")
        self.assertEqual(self.o["genannt"], "verweigert")        # der eigene Befund, nicht der Lauf
        self.assertEqual(self.o["propOhneAusgang"], "keinlauf")   # kein Ausgang = nicht belegt, nie 'geht'

    def test_the_row_chip_for_the_w64_case(self):
        h = self.o["html"]
        self.assertIn("pfx-v-nichtbeurteilt", h)
        self.assertNotIn("pfx-v-geht", h)
        self.assertIn("not judged (run refused: W64-OPPOINT)", h)
        self.assertIn("pfx-z-uebersteuert", h)

    def test_the_chip_has_a_style(self):
        html = open(os.path.join(STATIC, "index.html"), encoding="utf-8").read()
        self.assertIn(".pfx-v-nichtbeurteilt", html)


@unittest.skipUnless(NODE, "node fehlt")
class F2SeedRow(unittest.TestCase):
    BODY = """
const w = (o) => Object.assign({ key: null, label: "--pp-stage-ratio (Seed)", alt: null, value: "31,17,16", state: "unverified", source: "H", reason: "G", changed: true, in_argv: true, verdikte: [], seed: true, profile_value: null }, o || {});
const p = (values) => ({ n: 3, form: "dual", values, verdict: { outcome: "geht", verdikte: [] }, proposal: {} });
const real = { key: "flag:--p-bs", label: "--p-bs", alt: "1", value: "2", state: "vorgeschlagen", source: "H", reason: "G", changed: true, in_argv: true, verdikte: [] };
const removed = { key: "flag:--x", label: "--x", alt: "5", value: null, state: "unverified", source: "H", reason: "entfernt", changed: true, in_argv: false, verdikte: [] };
// Dual: das Profil setzt 31,17,16 schon (Referenz-Dual): keine Aenderung, kein "not set"
out.dual = PX.renderProposal(p([w({ profile_value: "31,17,16" })]));
// Flip: der Seed steht nicht im argv: keine Aenderung
out.flip = PX.renderProposal(p([w({ in_argv: false, changed: true })]));
// Seed im argv, Profil hat den Wert nicht: eine Aenderung, "not set" ist hier wahr
out.neu = PX.renderProposal(p([w()]));
// Seed im argv, Profil hat einen anderen Wert: der Profilwert steht als alt
out.anders = PX.renderProposal(p([w({ profile_value: "30,18,16" })]));
// ein echter Wert + ein entfernter Wert (in_argv false, value null) zaehlen weiter
out.mix = PX.renderProposal(p([real, removed, w({ in_argv: false })]));
"""

    @classmethod
    def setUpClass(cls):
        cls.o = run_node(cls.BODY)

    @staticmethod
    def _n(h):
        return int(re.search(r"(\d+) values changed", h).group(1))

    def test_dual_profile_value_equal_is_no_change_and_never_says_not_set(self):
        h = self.o["dual"]
        self.assertEqual(self._n(h), 0)
        self.assertNotIn("not set", h)
        self.assertNotIn("What the proposal changed", h)
        self.assertIn("The profile already sets the same value.", h)

    def test_flip_seed_outside_argv_is_not_counted(self):
        h = self.o["flip"]
        self.assertEqual(self._n(h), 0)
        self.assertIn("Not in the argv", h)
        self.assertNotIn("What the proposal changed", h)

    def test_seed_in_argv_and_new_is_a_change(self):
        h = self.o["neu"]
        self.assertEqual(self._n(h), 1)
        self.assertIn("not set", h)

    def test_seed_in_argv_shows_the_profile_value_as_the_old_one(self):
        h = self.o["anders"]
        self.assertEqual(self._n(h), 1)
        self.assertIn("30,18,16", h)
        self.assertNotIn("not set", h)

    def test_real_and_removed_values_still_count(self):
        self.assertEqual(self._n(self.o["mix"]), 2)


class F2Endpoint(unittest.TestCase):
    """Der Server nennt ``seed``, ``profile_value`` (Wert des BASISprofils) und ein ``in_argv``, das sagt, was der Launcher wirklich bekommt."""

    def test_argv_has(self):
        has = P.ProfilEditor.argv_has
        L = {"argv": ["--p-bs", "2", "--pp-stage-ratio", "31,17,16", "--extra-p=--pp-attn-stage-ratio 9,3,3"]}
        self.assertTrue(has(L, "--pp-stage-ratio", "31,17,16"))
        self.assertFalse(has(L, "--pp-stage-ratio", "42,11,11"))
        self.assertTrue(has(L, "--pp-attn-stage-ratio", "9,3,3"))                 # als --extra-p=...
        self.assertTrue(has({"argv": ["--pp-stage-ratio=1,2"]}, "--pp-stage-ratio", "1,2"))
        self.assertFalse(has(None, "--pp-stage-ratio", "1"))
        self.assertFalse(has({"argv": ["--pp-stage-ratio"]}, "--pp-stage-ratio", "1"))

    def test_argv_has_finds_the_flag_at_any_position_of_an_extra_token(self):
        has = P.ProfilEditor.argv_has
        L = {"argv": ["--extra-p=--a 1 --pp-attn-stage-ratio 9,3,3 --b 2", "--extra-d=--c --d-flag 7"]}
        self.assertTrue(has(L, "--pp-attn-stage-ratio", "9,3,3"))                 # in der Mitte
        self.assertTrue(has(L, "--a", "1"))                                       # am Anfang
        self.assertTrue(has(L, "--b", "2"))                                       # am Ende
        self.assertTrue(has(L, "--d-flag", "7"))
        self.assertFalse(has(L, "--pp-attn-stage-ratio", "9,3"))                  # kein Teilstring-Treffer
        self.assertFalse(has(L, "--a", "9,3,3"))
        self.assertTrue(has({"argv": ["--extra-p=--x '1 2' --y 3"]}, "--x", "1 2"))   # Anfuehrung bleibt ein Token

    def _propose(self, seed_value, argv):
        tmp = tempfile.mkdtemp(prefix="nach1006_")
        self.addCleanup(shutil.rmtree, tmp, True)
        from test_profil_orakel_apd_1006 import FakeOracle, _doc, _hw_profile, _rows, editor
        values = [{"key": "--pp-stage-ratio (Seed)", "group": "-", "policy": "cut_seed", "alt": None, "value": seed_value, "entries": 3, "state": "unverified",
                  "source": "H", "reason": "G", "in_argv": True, "changed": True}]
        answer = {"proposal": {"schema": "flliper.propose-a/1", "form": "dual", "n": 3, "values": values, "goals": {}, "cards": [], "inventory": {}, "seeds": {},
                                "fit": {"level": "ja"}, "unverified": [], "notes": [], "blocker": [], "vector_lengths": {}, "vectors_ok": True, "vectors_wrong": {}, "basis": "demo.env"},
                  "verdict": _doc([], n=3, outcome="geht"), "per_value": {}, "launch": {"argv": argv, "env": {}}}
        ed = editor(tmp, oracle=FakeOracle(propose=answer), hardware=lambda: {"ok": True, "profile": _hw_profile(_rows(3))})
        # the fixture planner tree carries no Dual module: this test is about the 27B line's Dual proposal, so the line probe is set (NF line 07.10.: no Dual
        # without the modules, see test_profil_dual_linie_1007)
        with mock.patch.object(P, "dual_line_probe", return_value=True):
            return ed.propose({"basis": {"kind": "release", "name": "demo"}, "form": "dual", "inventory": "rig"})

    def test_seed_that_the_launcher_does_not_get_is_not_in_argv_and_names_the_profile_value(self):
        # demo.env setzt --pp-stage-ratio 29,11,8; der argv des Vorschlags traegt weiter diesen Wert, nicht den Seed (Profil-Modus des Dual)
        r = self._propose("42,11,11", ["--pp-stage-ratio", "29,11,8"])
        w = [x for x in r["values"] if x.get("seed")][0]
        self.assertEqual((w["key"], w["value"], w["profile_value"]), (None, "42,11,11", "29,11,8"))
        self.assertIs(w["in_argv"], False)
        self.assertIs(w["in_argv_planer"], True)

    def test_seed_that_is_in_the_argv_stays_in_argv(self):
        r = self._propose("42,11,11", ["--pp-stage-ratio", "42,11,11"])
        w = [x for x in r["values"] if x.get("seed")][0]
        self.assertIs(w["in_argv"], True)
        self.assertEqual(w["profile_value"], "29,11,8")


class F3Laufbericht(Base):
    VD = {"schema": "flliper.verdict/1", "outcome": "ok_with_force", "oracle": {"runs": 2},
          "verdikte": [{"code": "HW-COUNT", "level": "run", "text": "HW-COUNT: 2 cards would be P = TP1 x PP2 ...", "reason": "g"},
                       {"code": "METAL-UNPROVEN", "level": "blocker", "parent": "HW-COUNT", "text": "no release boot", "reason": "g"},
                       {"code": "W64", "level": "run", "text": "W64 PdFlipTpOperatingPointInfeasible: x", "reason": "g"},
                       {"code": "FIT", "level": "fit", "text": "hw_fit: ja", "reason": "g"}]}

    def test_without_a_dry_run_the_report_takes_the_oracle_run_of_the_proposal(self):
        t = self.report(dry=None, proposal=self.VD)["text"]
        self.assertNotIn("No dry run was made", t)
        self.assertIn("Oracle run of the proposal (launcher dry run, 2 run(s)): passes only with force.", t)
        self.assertIn("`HW-COUNT`", t)
        self.assertIn("`W64`", t)
        self.assertNotIn("`METAL-UNPROVEN`", t)             # Blocker und Fit-Verdikte sind keine Ablehnung des Laufs
        self.assertNotIn("`FIT`", t)
        self.assertIn("changes made afterwards are not checked in it", t)

    def test_a_code_without_register_row_shows_its_class_and_consequence_but_stays_blocked(self):
        vd = {"schema": "flliper.verdict/1", "outcome": "verweigert", "oracle": {"runs": 2},
              "verdikte": [{"code": "W64-OPPOINT", "level": "run", "text": "W64 PdFlipTpOperatingPointInfeasible: x", "class": "nicht_forcebar",
                            "consequence": "Remains even with force.", "forcebar": True, "force_state": "force"}]}       # der Browser behauptet forcebar: nicht geglaubt
        t = self.report(dry=None, proposal=vd)["text"]
        row = [x for x in t.split("\n") if x.startswith("| `W64-OPPOINT`")][0]
        self.assertIn("| not forceable | blocked:", row)
        self.assertIn("Remains even with force.", row)
        self.assertIn("Remaining even with force: `W64-OPPOINT`", t)
        self.assertNotIn("FLLIPER_FORCE=1", t.split("### Versions")[0].split("### Verdicts and force")[1])

    def test_without_both_it_still_says_no_dry_run(self):
        self.assertIn("No dry run was made", self.report(dry=None)["text"])
        self.assertIn("No dry run was made", self.report(dry=None, proposal={"schema": "x"})["text"])

    def test_a_dry_run_is_younger_and_wins(self):
        dry = {"rejections": [], "verdict": "The planner refuses nothing.", "cards": [{"label": "RTX 5090"}]}
        t = self.report(dry=dry, proposal=self.VD)["text"]
        self.assertIn("Dry run: The planner refuses nothing.", t)
        self.assertNotIn("Oracle run of the proposal", t)

    def test_forceability_is_read_from_the_register_not_from_the_browser(self):
        vd = json.loads(json.dumps(self.VD))
        vd["verdikte"][0].update(forcebar=False, force_state="is_blocked")
        a = self.report(dry=None, proposal=vd)["text"]
        b = self.report(dry=None, proposal=self.VD)["text"]
        self.assertEqual(a, b)

    def test_dry_from_vorschlag_is_none_for_garbage(self):
        for bad in (None, {}, [], "x", {"schema": "flliper.verdict/1"}, {"schema": "flliper.verdict/1", "verdikte": "x"}):
            self.assertIsNone(P.dry_from_vorschlag(bad))

    def test_the_page_sends_the_proposal_verdict_with_the_issue_request(self):
        with open(os.path.join(STATIC, "profil.js"), encoding="utf-8") as fh:
            js = fh.read()
        self.assertIn("proposal: propVerdikt()", js)
        with open(os.path.join(os.path.dirname(HERE), "server.py"), encoding="utf-8") as fh:
            self.assertIn('proposal=body.get("proposal")', fh.read())


class F4HardwareIssueText(unittest.TestCase):
    DOC = {"schema": "flliper.hardware/1", "cards": [
        {"ord": 0, "nvml_index": 1, "name": "NVIDIA GeForce RTX 5090", "uuid": "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", "pci_bus_id": "00000000:01:00.0",
         "vram_total_mib": {"v": 32607, "src": "NVML"}},
        {"ord": 1, "nvml_index": 0, "name": "NVIDIA GeForce RTX 3080", "pci_bus_id": "00000000:02:00.0", "vram_total_mib": {"v": 20480, "src": "NVML"}}]}

    def test_the_uuid_is_never_in_the_text_even_when_it_does_not_look_like_a_secret(self):
        t = hwprofil.issue_text(self.DOC)
        self.assertNotIn("GPU-aaaaaaaa", t)
        row = [x for x in t.split("\n") if "RTX 5090" in x and "00000000:01:00.0" in x][0]
        self.assertTrue(row.rstrip().endswith("| 00000000:01:00.0 | <redacted> |"), row)
        self.assertIn("00000000:02:00.0 | unverified |", t)         # keine UUID gemeldet: "unbelegt", nicht "entfernt"

    def test_the_explanation_says_what_the_table_does(self):
        with open(os.path.join(STATIC, "hwprofil.js"), encoding="utf-8") as fh:
            js = fh.read()
        self.assertNotIn("UUID and PCI bus of the cards are included", js)
        self.assertIn("the UUID of the cards is always redacted as", js)
        self.assertIn("PCI bus are included", js)


class F5AriaLabel(unittest.TestCase):
    def test_card_select_of_the_synthetic_inventory_has_a_label(self):
        with open(os.path.join(STATIC, "profil.js"), encoding="utf-8") as fh:
            js = fh.read()
        sels = re.findall(r"<select data-cf=\"card\"[^>]*>", js)
        self.assertEqual(len(sels), 1)
        self.assertIn('aria-label="Choose card ${i + 1}"', sels[0])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()


class F1LaufberichtOhneLauf(Base):
    """Vorschlag ohne Launcher-Lauf (SingleCard does_not_fit / unverified, Oracle-Fehler): ein eigener Block, kein ``Trockenlauf:`` und kein Force-Satz
    ``Der Planer lehnt nichts ab``."""

    @staticmethod
    def VD(outcome, rank_verdicts, runs=0):
        return {"schema": "flliper.verdict/1", "outcome": outcome, "oracle": {"runs": runs}, "verdikte": rank_verdicts}

    def _v(self, vd):
        t = self.report(dry=None, proposal=vd)["text"]
        return t.split("### Verdicts and force")[1].split("### Versions")[0]

    def test_passt_nicht_is_a_planer_calculation_block_without_force_claims(self):
        vd = self.VD("does_not_fit", [{"code": "EINZEL-PASSUNG", "level": "fit", "text": "passt nicht: 3 MiB zu wenig"},
                                     {"code": "KV-MIN", "level": "fit", "text": "KV-Pool unter dem Minimum"}])
        v = self._v(vd)
        self.assertIn("Planner calculation: does not fit (no launcher run).", v)
        self.assertIn("`KV-MIN`: KV-Pool unter dem Minimum", v)
        for bad in ("Dry run:", "Oracle run of the proposal", "The planner refuses nothing", "force is not needed", "0 run", "FLLIPER_FORCE"):
            self.assertNotIn(bad, v)
        self.assertIn("No launcher run", v)
        self.assertIn("is shown only by a dry run", v)

    def test_unbelegt_and_zero_runs_are_no_dry_run_either(self):
        v = self._v(self.VD("unverified", []))
        self.assertIn("Planner calculation: cannot be calculated.", v)
        self.assertNotIn("Dry run:", v)
        v0 = self._v(self.VD("geht", [], runs=0))              # laeufe 0 allein genuegt
        self.assertNotIn("Oracle run of the proposal", v0)
        self.assertNotIn("The planner refuses nothing", v0)

    def test_orakel_fehler_is_named_and_force_does_not_help_is_not_said(self):
        vd = self.VD("oracle_error", [{"code": "ORAKEL-FEHLER", "level": "oracle", "text": "OSError: kein Kindprozess"}])
        v = self._v(vd)
        self.assertIn("Oracle error: OSError: kein Kindprozess.", v)
        self.assertIn("no verdict", v)
        for bad in ("Force does not help here", "Dry run:", "The planner refuses nothing", "0 run"):
            self.assertNotIn(bad, v)

    def test_a_real_run_keeps_its_header_and_the_run_count(self):
        vd = self.VD("ok_with_force", [{"code": "HW-COUNT", "level": "run", "text": "HW-COUNT: x"}], runs=2)
        t = self.report(dry=None, proposal=vd)["text"]
        self.assertIn("Oracle run of the proposal (launcher dry run, 2 run(s)): passes only with force.", t)

    def test_dry_from_vorschlag_shape_for_no_run(self):
        d = P.dry_from_vorschlag(self.VD("does_not_fit", []))
        self.assertIs(d["no_run"], True)
        self.assertEqual(d["rejections"], [])
        fh = P.force_hint(d, [])
        self.assertEqual((fh["fall"], fh["show_line"]), ("kein_launcher_lauf", False))


# Runde 7: die Chip-Entscheidung je Wert ist EINE Tabelle (chipFor in profil_planer.js); eine Zelle = ein Fall.
# Spalten: N nicht genannt | OK genannt ok | F verweigert, forcebar | B verweigert, nicht forcebar | H Hinweis | U Force ungeprueft
_COLS = {"OK": ("geht", "ok"), "F": ("force", "only with --force"), "B": ("verweigert", "refused"), "H": ("note", "note"), "U": ("ungeprueft", "force unchecked")}
_N = {
    "geht": ("geht", "ok"),
    "ok_with_force": ("geht", "ok"),
    "verweigert": ("nichtbeurteilt", "not judged (run refused: W64-OPPOINT)"),
    "crash": ("nichtbeurteilt", "not judged (run refused: ORAKEL-ABSTURZ)"),
    "oracle_error": ("orakelfehler", "oracle error"),
    "passt": ("planerpasst", "planner estimate: fits"),
    "does_not_fit": ("planerpasstnicht", "planner estimate: does not fit"),
    "unverified": ("planerunbelegt", "planner estimate: unverified"),
    "kein_dokument": ("keinlauf", "no run"),
}
TABLE = {(a, c): (_N[a] if c == "N" else _COLS[c]) for a in _N for c in ("N", "OK", "F", "B", "H", "U")}
LAUNCHER_RUNS = ("geht", "ok_with_force", "verweigert", "crash")


@unittest.skipUnless(NODE, "node fehlt")
class F7ChipForTabelle(unittest.TestCase):
    """Runde 7, Befunde 1 und 2: EINE reine Funktion chipFor(outcome, laufEbene, wertVerdikte, quelle); die Tabelle steht im Kommentar der Funktion."""
    BODY = """
const V = (code, o) => Object.assign({ code, level: "run", forcebar: true, force_state: "force", reason: code + " reason", consequence: "k", values: ["--x"] }, o || {});
const COLS = { N: [], OK: [V("FIT", { level: "fit", forcebar: null, force_state: "geht" })], F: [V("W40")], B: [V("X", { forcebar: false, force_state: "is_blocked" })],
  H: [V("H", { level: "note", forcebar: null, force_state: "note" })], U: [V("U", { force_state: "ungeprueft" })] };
const LAUF = {
  geht: [], ok_with_force: [V("W40", { durchgelassen: true })], verweigert: [V("W64-OPPOINT", { forcebar: false, force_state: "is_blocked", durchgelassen: false })],
  crash: [V("ORAKEL-ABSTURZ", { level: "crash", forcebar: false, force_state: "is_blocked", durchgelassen: false })] };
out.cells = {};
for (const a of PX.AUSGAENGE) for (const c of PX.SPALTEN) {
  const r = PX.chipFor(a, LAUF[a] || [], COLS[c], "dry run (oracle)");
  out.cells[a + "|" + c] = { id: r.id, label: r.label, tip: r.tip, code: r.code || null };
}
out.axes = [PX.AUSGAENGE.length, PX.SPALTEN.length];
const n = (a, run) => PX.chipFor(a, run, [], "proposal (oracle)");
// Widerspruch im Dokument: ok_with_force MIT einem beendenden Verdikt ist nicht durchgelaufen
out.forceWiderspruch = n("ok_with_force", [V("W64-OPPOINT", { durchgelassen: false })]);
// verweigert / crash ohne beendendes Verdikt: nie "geht", nie ein erfundener Code
out.verwOhne = n("verweigert", []);
out.absOhne = n("crash", []);
// unbekannter / fehlender Ausgang = kein Dokument
out.unbekannt = n("quark", []);
out.fehlt = n(undefined, []);
// durchgelassen=true allein beendet keinen Lauf
out.nurDurchgelassen = n("verweigert", [V("W40", { durchgelassen: true })]);
// der Hinweis der Startebene
out.start = ["geht", "ok_with_force", "verweigert", "passt", undefined].map((a) => PX.startChip(a, []));
out.startWiderspruch = PX.startChip("ok_with_force", [V("W64-OPPOINT", { durchgelassen: false })]);
// Zeilen: ok_with_force traegt den Start-Chip im Vorschlag, geht nicht
const p = (a, vs) => ({ n: 3, form: "dual", values: [], verdict: { outcome: a, verdikte: vs }, proposal: {} });
out.propForce = PX.renderProposal(p("ok_with_force", [V("W40", { durchgelassen: true })]));
out.PropOk = PX.renderProposal(p("geht", []));
// eine Zeile nach ok_with_force: nicht genannter Wert = geht, nicht "nicht beurteilt"
const row = base({ name: "--rank-gpu-memory-mib", key: "flag:--rank-gpu-memory-mib" });
const W = { key: row.key, value: "1", state: "vorgeschlagen", changed: true, verdikte: [] };
out.zeileForce = PX.renderRow(row, ctx({ prop: { values: [W], verdict: { outcome: "ok_with_force", verdikte: [V("W40", { durchgelassen: true })] } }, vsrc: "prop" }));
"""

    @classmethod
    def setUpClass(cls):
        cls.o = run_node(cls.BODY)

    def test_the_table_has_nine_rows_and_six_columns(self):
        self.assertEqual(self.o["axes"], [9, 6])
        self.assertEqual(len(self.o["cells"]), 54)

    def test_every_cell_of_the_table(self):
        for (a, c), (ident, label) in sorted(TABLE.items()):
            with self.subTest(outcome=a, spalte=c):
                cell = self.o["cells"][a + "|" + c]
                self.assertEqual((cell["id"], cell["label"]), (ident, label))
                self.assertTrue(cell["tip"], "jede Zelle hat einen Tooltip")

    def test_goes_only_for_a_run_that_went_through_or_a_value_named_ok(self):
        for key, cell in self.o["cells"].items():
            a, c = key.split("|")
            if cell["id"] == "geht" and c == "N":
                self.assertIn(a, ("geht", "ok_with_force"), key)

    def test_the_word_launcher_is_only_in_cells_where_a_launcher_run_happened(self):
        for key, cell in self.o["cells"].items():
            a, c = key.split("|")
            if c != "N":
                continue
            text = cell["label"] + " " + cell["tip"]
            if a in LAUNCHER_RUNS:
                self.assertIn("launcher", text, key)
            else:
                self.assertNotIn("launcher", text, key)

    def test_a_named_value_has_its_own_verdict_in_every_row(self):
        for key, cell in self.o["cells"].items():
            a, c = key.split("|")
            if c in ("F", "B", "H", "U"):
                self.assertTrue(cell["code"], key)

    def test_force_run_value_is_judged_not_unjudged_and_the_force_note_is_at_the_start(self):
        o = self.o
        self.assertEqual(o["cells"]["ok_with_force|N"]["label"], "ok")
        self.assertNotIn("run refused", o["zeileForce"])
        self.assertIn("pfx-v-geht", o["zeileForce"])
        self.assertNotIn("pfx-v-nichtbeurteilt", o["zeileForce"])
        self.assertIn("with --force", o["propForce"])
        self.assertIn("pfx-v-mitforce", o["propForce"])
        self.assertNotIn("pfx-v-mitforce", o["PropOk"])
        self.assertEqual([x["label"] if x else None for x in o["start"]], [None, "with --force", None, None, None])
        self.assertIsNone(o["startWiderspruch"])
        self.assertNotIn("with --force", o["zeileForce"].replace("only with --force", ""))      # nie je Wert

    def test_edge_cells_never_invent_a_judgement(self):
        o = self.o
        self.assertEqual(o["forceWiderspruch"]["id"], "nichtbeurteilt")
        self.assertIn("W64-OPPOINT", o["forceWiderspruch"]["label"])
        for k in ("verwOhne", "absOhne", "nurDurchgelassen"):
            self.assertEqual(o[k]["id"], "nichtbeurteilt", k)
            self.assertIn("no run verdict", o[k]["label"], k)
            self.assertNotIn("run refused", o[k]["label"], k)
        self.assertEqual((o["unbekannt"]["id"], o["fehlt"]["id"]), ("keinlauf", "keinlauf"))

    def test_every_chip_id_has_a_style(self):
        html = open(os.path.join(STATIC, "index.html"), encoding="utf-8").read()
        ids = {cell["id"] for cell in self.o["cells"].values()} | {"mitforce", "alt"}
        for i in sorted(ids):
            if i in ("geht", "force", "verweigert", "note", "ungeprueft"):
                continue
            self.assertIn(".pfx-v-" + i, html, i)
