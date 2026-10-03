"""NF PORTS 1003 (order 980 item 1): the NF line gets HW-P1ab (card count from
the inventory, ``--cards``, the hw_sim harness; cherry-picked from
desk/27b-hw-p1ab-1003) and the harness answers NF for 1..8 cards with honest
blockers: NF runs only on the reference rig's three cards plus system RAM
(the experts live in the host store), boots its RELEASE checkpoint (abl) and
says that the host RAM is not simulated.
"""

import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import form as F
from sglang.srt.weg2 import hw_sim as HS
from sglang.srt.weg2 import launcher as L
from sglang.srt.weg2 import topology as T

MC = "/spinning/llm_stuff/club-3090/models-cache/"
BASE = MC + "Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist"
ABL = BASE + "-abl-wxp"


def ns_for(*extra, model=ABL):
    return L.build_parser().parse_args(
        ["--tree", "/t", "--tag", "t", "--profile", "nextflash", "--model", model, *extra])


class HwSimGivesHonestNfRowsFor1To8(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.nf = HS.MODELS["NF"]
        cls.cells = list(HS.grid((1, 2, 3, 4, 5, 6, 7, 8), ["NF"]))

    def test_nf_boots_the_release_checkpoint_and_resolves_its_format(self):
        self.assertTrue(self.nf.derivative and self.nf.host_store)
        for c in self.cells:
            self.assertFalse([n for n in c.notes if n.startswith("PROFILE:")], c.row())

    def test_every_n_has_nf_rows(self):
        self.assertEqual(sorted({c.n_cards for c in self.cells}), [1, 2, 3, 4, 5, 6, 7, 8])

    def test_only_the_reference_rig_runs_nf(self):
        runs = [c.inventory for c in self.cells if c.result == HS.RUNS]
        self.assertEqual(runs, ["3x ref 5090+3080"])

    def test_refused_rows_name_the_nf_blockers(self):
        for c in self.cells:
            if c.result == HS.RUNS:
                continue
            if c.n_cards == 3:  # proven count: only the calibration inventory / arch gate may refuse
                self.assertNotIn("METAL-UNPROVEN", c.blockers, c.row())
                continue
            self.assertIn("METAL-UNPROVEN", c.blockers, c.row())
            self.assertIn("PROFILE-VECTORS", c.blockers, c.row())
            if c.n_cards == 1:
                self.assertIn("FORM-A-1", c.blockers, c.row())
            else:
                self.assertIn("PP-CUT-PIN", c.blockers, c.row())
                self.assertIn("XCHG-REGION", c.blockers, c.row())
            self.assertNotIn("FIT", c.blockers, c.row())  # the experts are in the host store

    def test_every_nf_cell_says_the_host_ram_is_not_simulated(self):
        for c in self.cells:
            self.assertTrue(any("host store" in n for n in c.notes), c.row())


if __name__ == "__main__":
    unittest.main()
