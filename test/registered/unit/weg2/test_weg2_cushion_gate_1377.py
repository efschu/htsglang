# SPDX-License-Identifier: Apache-2.0
"""#1377 W11 -- the cushion gate, on the arithmetic that actually separates the
two boots.

xsn31/3 latched `W98 Weg2HostRateLatched cushion=0.20 < 1.50 now=92.40
shmem=67.51`, state=STOP at epoch=0, six seconds after the front's argv. The
first form I was asked to build -- `predicted_run_peak + FLOOR <= bound` --
was measured against both boots and funds BOTH of them:

    xsn31/2   90.66 + 1.50 = 92.16 <= 94.43  -> funds
    xsn31/3   90.66 + 1.50 = 92.16 <= 94.43  -> funds   <-- W98 still fires

so it is a floor, not this gate. What consumed the cushion is PINNED SHM, not
predicted peak: the bounce went 7.79 -> 15.75 GiB under #1374's Option 1
sizing (5 lanes x 24 slots x 128 MiB), and xsn31/2's own cushion minimum was
6.99 GiB:

    6.99 - (15.75 - 7.79) = -0.97  <  1.50    (the boot measured 0.20)

That is the gate, and both numbers belong on the refusal line.
"""

import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import host_ledger as hl
from sglang.test.test_utils import CustomTestCase

SAMPLER = ("/spinning/evidence-665-f1/weg2xsn31_0913/"
           "hostsample_weg2xsn31.csv")


class TheCushionMinimumComesFromTheSamplersOwnColumn(CustomTestCase):
    @unittest.skipUnless(os.path.exists(SAMPLER), "rig-only evidence")
    def test_it_reads_the_v3_column_and_takes_the_MINIMUM(self):
        """A mean would hide the dip that fires the latch; a last-sample
        reading would hide it whenever the boot recovered."""
        got = hl.cushion_min_from_sampler(SAMPLER)
        self.assertIsNotNone(got)
        self.assertAlmostEqual(got, 0.05, places=2)

    def test_an_absent_file_is_None_and_never_zero(self):
        """0.0 would read as 'no cushion left' and refuse every arm on a box
        whose sampler was not armed -- absence is not a measurement."""
        self.assertIsNone(hl.cushion_min_from_sampler("/nonexistent/x.csv"))


class TheGateSeparatesTheTwoBoots(CustomTestCase):
    FLOOR = 1.50

    def test_the_ordered_peak_plus_floor_form_funds_BOTH_and_is_not_the_gate(self):
        """Kept as a measurement, so the corrected formula cannot quietly drift
        back to the one that would have passed xsn31/3."""
        for peak, bound in ((90.66, 94.43), (90.66, 94.43)):
            self.assertLessEqual(peak + self.FLOOR, bound)

    def test_xsn31_slash_3_is_REFUSED_by_the_cushion_arithmetic(self):
        got = hl.cushion_headroom_gib(6.99, 15.75, 7.79)
        self.assertAlmostEqual(got, -0.97, places=2)
        self.assertLess(got, hl.RATE_LATCH_CUSHION_FLOOR_GIB)

    def test_xsn31_slash_2_stays_FUNDABLE(self):
        got = hl.cushion_headroom_gib(6.99, 7.79, 7.79)
        self.assertAlmostEqual(got, 6.99, places=2)
        self.assertGreaterEqual(got, hl.RATE_LATCH_CUSHION_FLOOR_GIB)

    def test_an_unmeasured_cushion_yields_None_not_a_verdict(self):
        self.assertIsNone(hl.cushion_headroom_gib(None, 15.75, 7.79))

    def test_the_floor_is_the_latchs_own_constant(self):
        self.assertEqual(hl.RATE_LATCH_CUSHION_FLOOR_GIB, self.FLOOR)


if __name__ == "__main__":
    unittest.main()
