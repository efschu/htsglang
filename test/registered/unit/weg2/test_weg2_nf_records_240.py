# SPDX-License-Identifier: Apache-2.0
"""#240 (28.09.): the NF line booked group P's awake overshoot with the 27B row.

``P_OVERSHOOT_MIB`` [920, 0, 512] (boot weg2ls2b2, 07.09.) was BORROWED by the
nextflash profile: the P budget line of every NF boot said
``measured_awake_overshoot 920 (boot weg2ls2b2)`` on the 5090 and 512 on PP2,
and nothing on PP1 -- the one P rank whose free card fell to 698 MiB in the NF
boots of 28.09. (PP0 kept >= 1523, PP2 >= 1700).

The NF record is measured at the CARD over the ten NF boots that carry #242
(12:09-19:38Z): excess = corridor + booked overshoot - free card at the window
peak; the record is what the corridor's builtin 404 does not carry. The numbers
below are the worst window per ordinal, verbatim from the front/P logs:

  PP0 7d507357b9 14:47 chunk: corridor 1459, booked 920, free 1523 -> +856 -> 452
  PP1 dwell30   14:59 chunk: corridor 1499, booked 0,   free 698  -> +801 -> 397
  PP2 (all)                  corridor 1262, booked 512, free >= 1700 -> <= +74 -> 0
"""

import os

from sglang.test.test_utils import CustomTestCase

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import form as F  # noqa: E402
from sglang.srt.weg2 import launcher as L  # noqa: E402

BUILTIN_AWAKE = 404

#: (corridor, booked overshoot, free card at the window peak) of the worst window
WORST = {0: (1459, 920, 1523), 1: (1499, 0, 698), 2: (1262, 512, 1700)}


def _record_from_card(corridor, booked, free_at_peak, builtin=BUILTIN_AWAKE):
    excess = corridor + booked - free_at_peak
    return max(0, excess - builtin)


class TheNfRowOwnsItsPOvershoot(CustomTestCase):
    def test_the_record_is_the_card_measurement(self):
        want = tuple(_record_from_card(*WORST[o]) for o in range(3))
        self.assertEqual(want, (452, 397, 0))
        self.assertEqual(tuple(L._pconst("P_OVERSHOOT_MIB", "nextflash")), want)

    def test_measured_on_nf_not_borrowed(self):
        row = F.PROFILES["nextflash"]
        self.assertEqual(row.constants["P_OVERSHOOT_MIB"].measured_on, "nextflash")
        self.assertNotIn("P_OVERSHOOT_MIB", dict(F.borrowed_constants("nextflash")))

    def test_the_27b_row_keeps_its_own(self):
        self.assertEqual(tuple(L._pconst("P_OVERSHOOT_MIB", "qwen27b")), (920, 0, 512))
        self.assertEqual(L.P_OVERSHOOT_MIB, [920, 0, 512])

    def test_the_budget_line_names_the_nf_boots(self):
        prov = L._pconst_boots("P_OVERSHOOT_MIB", "nextflash")
        self.assertTrue(prov.startswith("boot dkrnfh91"), prov)
        self.assertNotIn("weg2ls2b2", prov)
        self.assertNotIn("measured_on=", prov)

    def test_after_the_record_the_worst_window_keeps_the_floor(self):
        # free' = free - booked + record: the budget moves by (booked - record),
        # the footprint follows it 1:1 (conservative), the free card lands on
        # corridor - builtin = the measured floor or above
        for o, (corridor, booked, free) in WORST.items():
            rec = L._pconst("P_OVERSHOOT_MIB", "nextflash")[o]
            self.assertGreaterEqual(free - booked + rec, corridor - BUILTIN_AWAKE, o)
