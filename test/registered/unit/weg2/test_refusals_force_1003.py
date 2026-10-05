"""PROFIL-EDITOR S1 (Auftrag 930): das Ablehnungsregister und der EINE Force-Schalter (``--force``).

Nutzer-Entscheid 03.10. ~20:15Z: ein Force-Flag beim Serverstart hebt die Ablehnung der Werte auf und startet trotzdem.

Gepinnt:
  * Register: jede Zeile hat Klasse und Begruendung; Belegung der Karte, fehlendes Modell, Architektur sind NICHT forcebar.
  * Das Register luegt nicht ueber die Verdrahtung: ``wired_codes(launcher.py)`` ist Teilmenge der Wert-Codes; jeder Wert-Code, den der
    Launcher abfragt, existiert im Register.
  * Ohne ``--force`` ist jede Verweigerung unveraendert (gleicher Text, gleiche Ausnahme, ``from``-Kette).
  * Mit ``--force``: HW-COUNT und HW-UNCALIBRATED laufen durch, ``FORCED-PAST <CODE> <Grund>`` wird geloggt, der Boot schreibt keine Records
    (``host_ledger.append_measured_record`` kehrt zurueck), die Kinder erben die Markierung.
  * Nicht forcebare Codes werden auch mit ``--force`` geworfen (Belegung, HW-ARCH).
GPU-frei, NVML-frei.
"""

import json
import os
import tempfile
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import card_identity as CI
from sglang.srt.weg2 import host_ledger
from sglang.srt.weg2 import launcher as L
from sglang.srt.weg2 import refusals as R

HERE = os.path.dirname(os.path.abspath(__file__))
LAUNCHER_SRC = os.path.join(HERE, "..", "..", "..", "..", "python", "sglang", "srt", "weg2", "launcher.py")


def card(i, name, mib, cc, uuid=None):
    return L.Card(i, uuid or f"GPU-{i:04d}", name, mib, reserved_mib=0, cc=cc)


def rig():
    return [card(0, "NVIDIA GeForce RTX 3080", 20480, (8, 6)), card(1, "NVIDIA GeForce RTX 5090", 32607, (12, 0)),
            card(2, "NVIDIA GeForce RTX 3080", 20480, (8, 6))]


def ns_for(profile="nextflash", *extra):
    return L.build_parser().parse_args(["--tree", "/t", "--tag", "t", "--profile", profile, *extra])


class Base(unittest.TestCase):
    def setUp(self):
        self._env = os.environ.pop(R.ENV_FORCED_BOOT, None)
        R.arm(False)

    def tearDown(self):
        R.arm(False)
        os.environ.pop(R.ENV_FORCED_BOOT, None)
        if self._env is not None:
            os.environ[R.ENV_FORCED_BOOT] = self._env


class Register(Base):
    def test_every_row_has_a_class_and_a_reason(self):
        codes = [r.code for r in R.REGISTER]
        self.assertEqual(len(codes), len(set(codes)))
        for r in R.REGISTER:
            self.assertIn(r.klass, (R.CLASS_VALUE, R.CLASS_HARD))
            self.assertGreater(len(r.why_class), 40, r.code)
            self.assertTrue(r.source and r.enforced_by, r.code)

    def test_what_the_user_named_is_not_forceable(self):
        for c in ("HW-ARCH", "HW-TOPOLOGY", "KARTE-BELEGT", "SHM-BELEGT", "MODELL-FEHLT", "PORT-BELEGT"):
            self.assertFalse(R.by_code(c).forcebar, c)
        for c in ("HW-COUNT", "HW-UNCALIBRATED", "D-BUDGET", "P-CARD", "HOST-MEM", "WAKE-CREDIT"):
            self.assertTrue(R.by_code(c).forcebar, c)

    def test_the_register_does_not_claim_more_than_the_launcher_does(self):
        with open(LAUNCHER_SRC, encoding="utf-8") as fh:
            src = fh.read()
        wired = R.wired_codes(src)
        self.assertTrue(wired)
        for c in wired:
            self.assertIsNotNone(R.by_code(c), c)
            self.assertTrue(R.by_code(c).forcebar, c)               # a hard code is never consulted for Force
        for c in ("HW-COUNT", "HW-UNCALIBRATED", "HOST-MEM", "D-BUDGET", "WAKE-CREDIT", "P-CARD"):
            self.assertIn(c, wired)
        pub = {r["code"]: r for r in R.public_register(wired)}
        self.assertTrue(pub["HW-COUNT"]["wired"])
        self.assertFalse(pub["PP-CUT"]["wired"])                   # forcebar by class, not yet wired: said so
        self.assertIsNone(pub["HW-ARCH"]["wired"])

    def test_classify_by_prefix(self):
        self.assertEqual(R.classify("HW-COUNT: 2 card(s)"), "HW-COUNT")
        self.assertIsNone(R.classify("W40 something"))


class Switch(Base):
    def test_unarmed_refuse_value_raises_exactly_like_raise(self):
        cause = ValueError("c")
        with self.assertRaises(L.Weg2LaunchRefused) as cm:
            R.refuse_value("HW-COUNT", "HW-COUNT: x", L.Weg2LaunchRefused, cause=cause)
        self.assertEqual(str(cm.exception), "HW-COUNT: x")
        self.assertIs(cm.exception.__cause__, cause)
        with self.assertRaises(L.Weg2LaunchRefused):
            R.refuse_value("HW-COUNT", "plain", L.Weg2LaunchRefused)

    def test_forced_value_code_passes_logs_and_is_listed(self):
        lines = []
        R.arm(True)
        R.refuse_value("D-BUDGET", "KV does not fit\n  by 1840 MiB", L.Weg2LaunchRefused, lines.append)
        self.assertEqual(lines, ["FORCED-PAST D-BUDGET KV does not fit by 1840 MiB"])
        self.assertEqual(R.forced_list(), [{"code": "D-BUDGET", "text": "KV does not fit\n  by 1840 MiB"}])
        self.assertEqual(os.environ.get(R.ENV_FORCED_BOOT), "1")        # the front and the groups inherit it

    def test_hard_and_unknown_codes_raise_even_when_forced(self):
        R.arm(True)
        for code in ("HW-ARCH", "KARTE-BELEGT", "NOT-A-CODE"):
            with self.assertRaises(L.Weg2LaunchRefused):
                R.refuse_value(code, "no", L.Weg2LaunchRefused)
        self.assertEqual(R.forced_list(), [])

    def test_flush_logs_what_no_logger_saw_once(self):
        R.arm(True)
        R.refuse_value("HW-COUNT", "HW-COUNT: a", L.Weg2LaunchRefused)
        out = []
        self.assertEqual(R.flush(out.append), 1)
        self.assertEqual(R.flush(out.append), 0)
        self.assertEqual(out, ["FORCED-PAST HW-COUNT HW-COUNT: a"])

    def test_sink_gets_every_forced_refusal(self):
        got = []
        R.arm(True, sink=lambda c, t: got.append((c, t)))
        R.refuse_value("HOST-MEM", "low", L.Weg2LaunchRefused)
        self.assertEqual(got, [("HOST-MEM", "low")])


class LauncherWiring(Base):
    def test_parser_has_one_force_switch_default_off(self):
        self.assertFalse(ns_for().force)
        self.assertTrue(ns_for("nextflash", "--force").force)

    def test_hw_count_refused_without_force_passes_with_force(self):
        two = rig()[:2]
        with self.assertRaises(L.Weg2LaunchRefused) as cm:
            L.order_cards(two, 3)                                  # a caller that needs exactly N (HW-P1a: no count gate by default)
        self.assertIn("HW-COUNT", str(cm.exception))
        R.arm(True)
        ordered = L.order_cards(two, 3)
        self.assertEqual(len(ordered), 2)
        self.assertEqual(CI.class_label(ordered[0]), "RTX5090")
        self.assertEqual([x["code"] for x in R.forced_list()], ["HW-COUNT"])

    def test_topology_check_unproven_n_is_a_value_refusal_no_topology_is_not(self):
        ns = ns_for()
        two = L.order_cards(rig()[:2])
        with self.assertRaises(L.Weg2LaunchRefused) as cm:
            L.topology_check_line(ns, two)
        self.assertIn("HW-COUNT", str(cm.exception))                # N=2 is inside 2..8, not proven: a value refusal
        R.arm(True)
        line = L.topology_check_line(ns, two)
        self.assertTrue(line.startswith("HW-TOPOLOGY N=2: not proven, started with --force"), line)
        self.assertEqual([x["code"] for x in R.forced_list()], ["HW-COUNT"])
        one = L.order_cards(rig()[:1])
        with self.assertRaises(L.Weg2LaunchRefused) as cm2:           # no flip topology exists for 1 card: --force does not help
            L.topology_check_line(ns, one)
        self.assertIn("HW-TOPOLOGY", str(cm2.exception))

    def test_the_reference_rig_topology_line_is_unchanged_under_force(self):
        line = L.topology_check_line(ns_for(), L.order_cards(rig()))
        R.arm(True)
        self.assertEqual(L.topology_check_line(ns_for(), L.order_cards(rig())), line)
        self.assertEqual(R.forced_list(), [])

    def test_hw_uncalibrated_refused_without_force_passes_with_force_and_the_line_says_so(self):
        three_090 = [card(i, "NVIDIA GeForce RTX 3090", 24576, (8, 6)) for i in range(3)]
        ns = ns_for()
        with self.assertRaises(L.Weg2LaunchRefused) as cm:
            L.inventory_check_line(ns, L.order_cards(three_090))
        self.assertIn("HW-UNCALIBRATED", str(cm.exception))
        R.arm(True)
        line = L.inventory_check_line(ns, L.order_cards(three_090))
        self.assertTrue(line.endswith("MISMATCH (--force)"), line)
        self.assertEqual([x["code"] for x in R.forced_list()], ["HW-UNCALIBRATED"])

    def test_the_reference_rig_line_is_unchanged(self):
        line = L.inventory_check_line(ns_for(), L.order_cards(rig()))
        self.assertTrue(line.endswith(") MATCH"), line)
        R.arm(True)
        self.assertTrue(L.inventory_check_line(ns_for(), L.order_cards(rig())).endswith(") MATCH"))
        self.assertEqual(R.forced_list(), [])                              # nothing refused, nothing passed

    def test_arch_gate_is_not_lifted_by_force(self):
        R.arm(True)
        # sm_89 is admitted since SM89 1002 (HW-UNCALIBRATED, forcible); an arch with no cubins stays refused
        with self.assertRaises(CI.CardInventoryRefused):
            CI.arch_gate([card(0, "NVIDIA H100 80GB HBM3", 81559, (9, 0))])
        with self.assertRaises(CI.CardInventoryRefused):
            CI.arch_gate([card(0, "NVIDIA A100-SXM4-80GB", 81920, (8, 0))])
        CI.arch_gate([card(0, "NVIDIA GeForce RTX 4090", 24564, (8, 9))])      # passes the gate, no Force needed


class NoRecords(Base):
    def _write(self):
        d = tempfile.mkdtemp()
        p = os.path.join(d, "rec.json")
        host_ledger.append_measured_record(p, {"x": 1})
        return p

    def test_normal_boot_writes_the_record(self):
        p = self._write()
        with open(p) as fh:
            self.assertEqual(json.load(fh)["samples"], [{"x": 1}])

    def test_forced_boot_writes_no_record(self):
        R.arm(True)
        self.assertFalse(os.path.exists(self._write()))

    def test_a_child_that_inherited_the_environment_writes_none(self):
        os.environ[R.ENV_FORCED_BOOT] = "1"                              # what the front/groups see; no arm() in that process
        self.assertFalse(os.path.exists(self._write()))


if __name__ == "__main__":
    unittest.main()
