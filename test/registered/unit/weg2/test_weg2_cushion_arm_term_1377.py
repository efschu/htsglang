# SPDX-License-Identifier: Apache-2.0
"""#1377 (a) W11 -- the cushion floor is an ARM term, not a runtime-only gate.

BOOT weg2xsn31/3 (@b8d2c2c1f5) funded an arm and then stopped itself before
the first flip (front log :202, 17:01:51Z):

    W98 Weg2HostRateLatched cushion=0.20 < 1.50 now=92.40 shmem=67.51
    state=STOP  epoch=0

six seconds after the front's own argv, i.e. inside D's startup sleep leg. The
ARM line :111 had funded `predicted_run_peak 90.66` against a hard bound of
94.43. Both numbers were known at launch, and 90.66 + 1.50 = 92.16 < 94.43
looks fine -- until the bounce is resident: xchg_bounce was 15.75 GiB
(xsn31/2: 7.79), which is #1374's Option 1 sizing, five lanes x 24 slots x
128 MiB. xsn31/2's own cushion minimum was 6.99 GiB, and 6.99 - 7.96 < 0.

THE DEFECT IS TWO FLOORS IN TWO PLACES: the ladder bounded the PEAK, and
`RATE_LATCH_CUSHION_FLOOR_GIB` bounded the CUSHION at runtime, with no
arithmetic joining them. So a latch that was arithmetically PREDICTABLE at
launch was not predicted, and the boot was funded into its own watchdog. The
floor is now imported into the arm's verdict -- never restated -- so the
refusal happens BY NAME at the ARM, with both numbers on the line.
"""

import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import host_ledger as hl
from sglang.test.test_utils import CustomTestCase

GIB = hl.GIB
#: The #1360 specimen shape, which is the only one that reaches this ladder.
_KW = dict(
    ring_bytes=int(30.4 * GIB), ring_span1_bytes=int(15.0 * GIB),
    cg_current_bytes=int(9.0 * GIB), reclaimable_bytes=int(3.0 * GIB),
    cg_ceiling_bytes=int(123.78 * GIB), s_gb_d=4,
)


class TheFloorIsOneNumberInOnePlace(CustomTestCase):
    def test_the_arm_imports_the_runtime_floor_rather_than_restating_it(self):
        """Two constants would drift; W11 is what drift costs."""
        import inspect

        src = inspect.getsource(hl.choose)
        self.assertIn("RATE_LATCH_CUSHION_FLOOR_GIB", src,
                      "the arm must read the LATCH'S OWN floor")
        self.assertNotIn("1.5 ", src.split("cushion_ok")[1][:400],
                         "the floor is imported, never re-typed as a literal")

    def test_the_floor_is_the_one_the_latch_uses(self):
        self.assertEqual(hl.RATE_LATCH_CUSHION_FLOOR_GIB, 1.5)


class TheXsn31SlashThreeArmMustRefuse(CustomTestCase):
    """RED-FIRST: these are the boot's own numbers."""

    def test_an_arm_that_leaves_less_than_the_latch_demands_is_refused(self):
        with self.assertRaises(
                (hl.Weg2HostLedgerRefused, hl.Weg2HostRunPeakRefused)) as cm:
            hl.choose(
                int(123.78 * GIB), int(110.0 * GIB), arms=[(1, 2400)],
                flip_ratchet=hl.FlipRatchet(
                    per_flip_gib=8.697, flips_priced=1, source="test",
                    from_record=False),
                **_KW,
            )
        msg = str(cm.exception)
        self.assertTrue("CUSHION FLOOR" in msg or "RUN PEAK" in msg,
                        f"the refusal must name which bound it hit: {msg}")

    def test_the_refusal_names_both_numbers_when_the_cushion_is_what_binds(self):
        """A verdict that says only `predicted vs bound` cannot be acted on:
        the reader has to know it was the LATCH's floor that bound it."""
        import inspect

        src = inspect.getsource(hl.choose)
        i = src.index("CUSHION FLOOR (")
        window = src[i:i + 700]
        self.assertIn("floor =", window)
        self.assertIn("hard bound", window)
        self.assertIn("weg2xsn31/3", window,
                      "the boot that proves the case belongs on the line")


class TheXsn31SlashTwoArmStaysFundable(CustomTestCase):
    """The other half of red-first: the PREVIOUS boot's numbers must stay
    green, or this gate is just a lower ceiling wearing a new name."""

    def test_a_peak_with_room_for_the_floor_still_funds(self):
        """A FUNDABLE specimen, not the #1360 one: that arm is built to sit
        between the bound and the watermark and is refused by the peak check
        alone, so it could never show that THIS gate leaves fundable arms
        alone. Measured headroom here: 17.86 GiB."""
        arm, headroom, lines = hl.choose(
            int(123.78 * GIB), int(110.0 * GIB), arms=[(1, 150)],
            flip_ratchet=hl.resolve_flip_ratchet_gib(),
            ring_bytes=int(12.0 * GIB), ring_span1_bytes=int(4.0 * GIB),
            cg_current_bytes=int(9.0 * GIB), reclaimable_bytes=int(3.0 * GIB),
            cg_ceiling_bytes=int(123.78 * GIB), s_gb_d=4,
        )
        self.assertIsNotNone(arm)
        self.assertGreater(headroom, hl.RATE_LATCH_CUSHION_FLOOR_GIB,
                           "a funded arm must clear the latch's own floor")
        self.assertTrue(any("WEG2-HOST-LEDGER" in ln for ln in lines))

    def test_the_gate_is_exactly_the_floor_and_not_a_margin(self):
        """`predicted + floor <= bound`, nothing else. A safety factor on top
        would be a third floor, which is the defect one more time."""
        import inspect

        src = inspect.getsource(hl.choose)
        i = src.index("cushion_ok = (")
        expr = src[i:i + 240]
        self.assertIn("predicted + RATE_LATCH_CUSHION_FLOOR_GIB <= hard_bound_gib",
                      " ".join(expr.split()))


if __name__ == "__main__":
    unittest.main()
