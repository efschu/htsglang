"""NF PORTS 1003 (order 980 item 4): the planner and launcher part of the profile
editor S1 (--profile / --force / FORCED-PAST, c32c640890 of order 930) on the NF
line, and the one thing the port adds: the card-count refusal of HW-P1a
(``topology_check_line``: HW-COUNT / HW-TOPOLOGY with the concrete blockers) goes
through the same Force switch as the other value refusals.

Order 930 left it open: "HW-TOPOLOGY: the launcher of this line does not call it".
Since HW-P1a it does, so a forced boot on two cards must be able to get past it
(and then runs into the next refusal, normally the first positional vector of
the wrong length -- the report said so). Without --force the refusal is the
raise it was: same text, same exception, same cause.
"""

import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import launcher as L
from sglang.srt.weg2 import refusals as R
from sglang.srt.weg2 import topology as T

HERE = os.path.dirname(os.path.abspath(__file__))
LAUNCHER_SRC = os.path.join(HERE, "..", "..", "..", "..", "python", "sglang", "srt", "weg2", "launcher.py")


def card(i, name, mib, cc):
    return L.Card(i, f"GPU-{i:04d}", name, mib, reserved_mib=0, cc=cc)


def rig():
    return [card(0, "NVIDIA GeForce RTX 3080", 20480, (8, 6)), card(1, "NVIDIA GeForce RTX 5090", 32607, (12, 0)),
            card(2, "NVIDIA GeForce RTX 3080", 20480, (8, 6))]


def ns_nf(*extra):
    return L.build_parser().parse_args(["--tree", "/t", "--tag", "t", "--profile", "nextflash",
                                        "--pp-stage-ratio", "29,11,8", *extra])


class Base(unittest.TestCase):
    def setUp(self):
        self._env = os.environ.pop(R.ENV_FORCED_BOOT, None)
        R.arm(False)

    def tearDown(self):
        R.arm(False)
        os.environ.pop(R.ENV_FORCED_BOOT, None)
        if self._env is not None:
            os.environ[R.ENV_FORCED_BOOT] = self._env


class TheNfLauncherCarriesTheSwitch(Base):

    def test_force_flag_and_register_are_on_the_nf_line(self):
        ns = ns_nf("--force")
        self.assertTrue(ns.force)
        self.assertFalse(ns_nf().force)
        self.assertEqual(ns.profile, "nextflash")

    def test_hw_count_and_hw_topology_are_wired(self):
        with open(LAUNCHER_SRC, encoding="utf-8") as fh:
            wired = R.wired_codes(fh.read())
        for c in ("HW-COUNT", "HW-TOPOLOGY", "HW-UNCALIBRATED"):
            self.assertIn(c, wired)
            self.assertTrue(R.by_code(c).forcebar, c)


class TopologyRefusalThroughTheSwitch(Base):

    def test_without_force_the_raise_is_unchanged(self):
        with self.assertRaises(L.Weg2LaunchRefused) as cm:
            L.topology_check_line(ns_nf(), L.order_cards(rig()[:2]), {})
        self.assertTrue(str(cm.exception).startswith("HW-COUNT: 2 cards would be P = TP1 x PP2"), str(cm.exception))
        self.assertIn(" || visible: nvml1 NVIDIA GeForce RTX 5090", str(cm.exception))
        self.assertIsInstance(cm.exception.__cause__, T.TopologyRefused)
        self.assertEqual(R.forced_list(), [])

    def test_one_card_is_hw_topology_without_force(self):
        with self.assertRaises(L.Weg2LaunchRefused) as cm:
            L.topology_check_line(ns_nf(), L.order_cards(rig()[1:2]), {})
        self.assertTrue(str(cm.exception).startswith("HW-TOPOLOGY: 1 card(s)"), str(cm.exception))

    def test_with_force_two_cards_are_listed_as_forced_past(self):
        lines = []
        R.arm(True)
        line = L.topology_check_line(ns_nf(), L.order_cards(rig()[:2]), {})
        self.assertTrue(line.startswith("HW-TOPOLOGY N=2: FORCED past "), line)
        self.assertIn("METAL-UNPROVEN", line)
        (entry,) = R.forced_list()
        self.assertEqual(entry["code"], "HW-COUNT")
        self.assertIn("2 cards would be P = TP1 x PP2", entry["text"])
        R.flush(lines.append)
        self.assertTrue(lines[0].startswith("FORCED-PAST HW-COUNT "), lines)
        self.assertFalse(R.records_allowed())

    def test_with_force_one_card_is_listed_as_hw_topology(self):
        R.arm(True)
        line = L.topology_check_line(ns_nf(), L.order_cards(rig()[1:2]), {})
        self.assertIn("SINGLE-MODE", line)
        self.assertEqual([e["code"] for e in R.forced_list()], ["HW-TOPOLOGY"])

    def test_with_force_the_proven_count_passes_as_before(self):
        R.arm(True)
        line = L.topology_check_line(ns_nf(), L.order_cards(rig()), {})
        self.assertEqual(line, "HW-TOPOLOGY N=3: P = TP1 x PP3, D = TP3 x PP1, rank-gpu-id 0,1,2, "
                               "host ordinal 0 -- proven on metal (N in [3])")
        self.assertEqual(R.forced_list(), [])

    def test_the_reference_rig_without_force_is_untouched(self):
        line = L.topology_check_line(ns_nf(), L.order_cards(rig()), {})
        self.assertTrue(line.startswith("HW-TOPOLOGY N=3:"))
        self.assertTrue(R.records_allowed())

    def test_the_hard_arch_refusal_stays_hard_under_force(self):
        R.arm(True)
        self.assertFalse(R.by_code("HW-ARCH").forcebar)
        with self.assertRaises(L.Weg2LaunchRefused):
            R.refuse_value("HW-ARCH", "HW-ARCH: sm100", L.Weg2LaunchRefused)


if __name__ == "__main__":
    unittest.main()
