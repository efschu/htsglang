"""Review #227 FORCED-RECORDS (24.10.), Pins des Reviewers -- ROT auf 2d84e4a091 ist beabsichtigt (MUSS-3).

GEFAHRENRICHTUNG (OOM): der HW-COUNT-Klassifikator (refusals._hw_count_metal_only) sucht mit ``search`` und
schluckt hinter "blocker(s): ..." bzw. hinter "|| visible:" JEDEN weiteren ``||``-Anhang. Ein spaeter an den
HW-COUNT-Text angehaengtes Inventar-/Borrow-Urteil (die Form, die inventory_check_line fuer HW-UNCALIBRATED schon
heute baut) liesse einen Boot fremder Konfiguration Records schreiben -- still, denn der Produzenten-Pin ueber alle N
bleibt dabei gruen. Die Kreuzungsbreite ist dagegen schon Vollabgleich; HW-COUNT muss dieselbe Form haben:
genau der Topologie-Kopf, genau N Blocker, hoechstens EIN "|| visible:"-Schwanz ohne weiteres "||".
GPU-frei, NVML-frei.
"""

import unittest

from flliper.srt.pdflip import refusals as R

#: verbatim boot_weg2_dkr27bint4arrc4bar1int4ar10241254 (FORCED-PAST HW-COUNT), visible-Schwanz gekuerzt
HEAD = ("HW-COUNT: 2 cards would be P = TP1 x PP2, D = TP2 x PP1 (host ordinal 0); not yet runnable, 1 blocker(s): "
        "[METAL-UNPROVEN] no release boot on 2 cards yet (proven: [3]) (pdflip/topology.py PROVEN_CARD_COUNTS)")
VISIBLE = " || visible: nvml0 NVIDIA GeForce RTX 3080 20480 MiB sm86; nvml1 NVIDIA GeForce RTX 3080 20480 MiB sm86"
FOREIGN = " || HW-UNCALIBRATED: profile 'qwen27b' positional records were measured on another inventory"


class HwCountIsAFullMatch(unittest.TestCase):
    def test_the_verbatim_boot_line_stays_neutral(self):
        self.assertEqual(R.records_class("HW-COUNT", HEAD + VISIBLE), "HW-COUNT/METAL-UNPROVEN")
        self.assertEqual(R.records_class("HW-COUNT", HEAD), "HW-COUNT/METAL-UNPROVEN")

    def test_a_join_after_the_visible_tail_is_not_neutral(self):
        self.assertIsNone(R.records_class("HW-COUNT", HEAD + VISIBLE + FOREIGN))

    def test_a_join_before_the_visible_tail_is_not_neutral(self):
        self.assertIsNone(R.records_class("HW-COUNT", HEAD + FOREIGN + VISIBLE))

    def test_a_foreign_prefix_is_not_neutral(self):
        self.assertIsNone(R.records_class("HW-COUNT", "HW-BORROWED: census row borrowed || " + HEAD + VISIBLE))


if __name__ == "__main__":
    unittest.main()
