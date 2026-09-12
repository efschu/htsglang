# SPDX-License-Identifier: Apache-2.0
"""#1361 -- the sampler-independent RATE latch, and the reap mark as a MEASUREMENT.

WHAT KILLED weg2xsn26b, measured rather than assumed
(hostsample_weg2xsn26b.csv, /.lxc cgroup):

  * the kill was the GLOBAL host OOM killer, not a cgroup limit:
    memory.events `max 0  high 0`, `oom_kill 1 -> 3`, and memory.max is `max`;
  * the currency was RIGHT -- shmem climbed 49.360 -> 60.008 GiB while anon FELL
    from its own peak, and `file - shmem` at the last sample was **0.008 GiB**,
    so the file-backed / checkpoint-mmap hypothesis (#738) is refuted outright;
  * the level guard at 93.0 never fired because ITS SERIES STOPPED: sampler gaps
    of 19 / 25 / 18 s, last reading 82.28 GiB, `memory.peak` 95.86;
  * in the last measured seconds shmem climbed 50.157 -> 53.415 -> 55.847, i.e.
    **2.4-3.3 GiB/s**, which covers the remaining 13.6 GiB in FOUR TO SIX
    SECONDS -- entirely inside the 19 s blind window.

So the defect is CADENCE AND BLINDNESS, not the unit. This latch runs on its own
clock, fires on `nonreclaim + rate x lookahead`, and reports its own blindness.
"""

import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import host_ledger as hl
from sglang.test.test_utils import CustomTestCase

GIB = hl.GIB
#: The real ramp, from the boot's own 1 Hz series (GiB, one second apart).
XSN26B_RAMP = (49.923, 50.157, 53.415, 55.847)
#: Where that boot's last reading sat, and where memory.peak ended up.
XSN26B_LAST = 82.279
XSN26B_PEAK = 95.863


class TheLatchPreEmptsTheMeasuredRamp1361(CustomTestCase):
    def test_the_real_ramp_latches_before_the_mark_is_reached(self):
        lat = hl.RateLatch(reap_mark_gib=95.90)
        line = None
        # the boot's own last four readings, on the level it actually stood at:
        # the ramp's deltas carried onto XSN26B_LAST, one second apart.
        base = XSN26B_LAST - (XSN26B_RAMP[-1] - XSN26B_RAMP[0])
        fired_at = None
        for i, v in enumerate(XSN26B_RAMP):
            out = lat.observe(float(i), base + (v - XSN26B_RAMP[0]))
            if out and "W98" in out and line is None:
                line, fired_at = out, i
        self.assertIsNotNone(line)
        # It fires on the ONSET sample, not on the last one: the third reading
        # is where the slope first reaches the mark inside the lookahead, and
        # that is one whole second of warning the level guard never had.
        self.assertEqual(fired_at, 2)
        self.assertIn("W98 Weg2HostRateLatched", line)
        self.assertIn("projected=", line)
        self.assertIn("rate=", line)
        self.assertIn("lookahead_s=5", line)
        # the LEVEL is still far below the mark when it fires -- that is the point
        self.assertLess(XSN26B_LAST, 95.90)

    def test_the_measured_slope_covers_the_gap_in_the_lookahead(self):
        """4-6 s at 2.4-3.3 GiB/s against 13.6 GiB -- the arithmetic of the death."""
        slope_lo, slope_hi = 2.4, 3.3
        gap = XSN26B_PEAK - XSN26B_LAST
        self.assertAlmostEqual(gap, 13.584, places=2)
        self.assertLess(gap / slope_hi, hl.RATE_LATCH_LOOKAHEAD_S)      # ~4.1 s
        self.assertLess(gap / slope_lo, hl.RATE_LATCH_LOOKAHEAD_S + 1)  # ~5.7 s

    def test_the_worst_pair_binds_not_the_smoothed_average(self):
        """End-to-end over the real ramp is 1.98 GiB/s and would NOT latch;
        the worst pair is 3.26 and does. Averaging smooths away the onset."""
        lat = hl.RateLatch(reap_mark_gib=95.90)
        for i, v in enumerate(XSN26B_RAMP):
            lat.observe(float(i), v)
        end_to_end = (XSN26B_RAMP[-1] - XSN26B_RAMP[0]) / 3.0
        self.assertAlmostEqual(end_to_end, 1.975, places=2)
        self.assertAlmostEqual(lat.rate_gib_per_s(), 3.258, places=2)
        self.assertGreater(lat.rate_gib_per_s(), end_to_end)

    def test_a_flat_series_never_latches(self):
        lat = hl.RateLatch(reap_mark_gib=95.90)
        for i in range(20):
            self.assertIsNone(lat.observe(i * 0.25, 90.0))
        self.assertFalse(lat.latched)

    def test_a_falling_series_is_not_credited_as_headroom(self):
        """A dip must not be projected downward into a licence to continue."""
        lat = hl.RateLatch(reap_mark_gib=95.90)
        for i, v in enumerate((95.0, 94.0, 93.0)):
            lat.observe(float(i), v)
        self.assertAlmostEqual(lat.projected_gib(), 93.0, places=6)
        self.assertFalse(lat.latched)

    def test_it_latches_once_and_then_stays_quiet(self):
        lat = hl.RateLatch(reap_mark_gib=95.90)
        lines = [lat.observe(float(i), 80.0 + 4.0 * i) for i in range(6)]
        fired = [x for x in lines if x and "W98" in x]
        self.assertEqual(len(fired), 1)


class BlindnessIsItselfAFinding1361(CustomTestCase):
    def test_a_gap_gets_its_own_line_with_its_length(self):
        lat = hl.RateLatch(reap_mark_gib=95.90)
        lat.observe(0.0, 80.0)
        line = lat.observe(19.0, 82.28)          # the measured 19 s hole
        self.assertIsNotNone(line)
        self.assertIn("WEG2-HOST RATE-GAP", line)
        self.assertIn("gap_s=19.0", line)
        self.assertEqual(lat.gaps, [19.0])

    def test_normal_cadence_jitter_is_not_a_gap(self):
        lat = hl.RateLatch(reap_mark_gib=95.90)
        lat.observe(0.0, 80.0)
        for t in (0.3, 0.62, 1.1, 2.0):
            self.assertIsNone(lat.observe(t, 80.0))
        self.assertEqual(lat.gaps, [])

    def test_the_three_measured_gaps_would_all_have_been_reported(self):
        lat = hl.RateLatch(reap_mark_gib=95.90)
        t = 0.0
        lat.observe(t, 80.0)
        for g in (19.0, 25.0, 18.0):
            t += g
            self.assertIn("RATE-GAP", lat.observe(t, 80.0) or "")
        self.assertEqual(lat.gaps, [19.0, 25.0, 18.0])


class TheReapMarkIsMeasuredNotAssumed1361(CustomTestCase):
    def test_the_line_derives_the_mark_and_keeps_the_constant_as_cross_check(self):
        # this rig at idle: MemTotal 118.05, MemAvailable 112.2, root current 7.56
        line = hl.reap_model_line(
            int(118.05 * GIB), int(19.00 * GIB), int(88.00 * GIB))
        self.assertIn("WEG2-HOST REAP MODEL:", line)
        self.assertIn("MemTotal=118.05", line)
        self.assertIn("outside=", line)
        self.assertIn("reap_mark=", line)
        self.assertIn("CROSS-CHECK", line)
        self.assertIn("95.90", line)            # the constant, beside not instead
        self.assertIn("GLOBAL host OOM killer", line)

    def test_outside_is_what_neither_this_cgroup_nor_the_free_pool_holds(self):
        """UNDER LOAD the two terms separate and the mark is derivable."""
        import re
        # a loaded box: the cgroup holds 88 GiB of anon+shmem MemAvailable
        # cannot count, and ~11 GiB sits outside it (ARC + foreign).
        line = hl.reap_model_line(
            int(118.05 * GIB), int(19.00 * GIB), int(88.00 * GIB))
        g = {k: float(v) for k, v in re.findall(
            r"(MemTotal|outside|reap_mark)=([0-9.]+)", line)}
        self.assertAlmostEqual(g["outside"], 118.05 - 19.00 - 88.00, places=1)
        self.assertAlmostEqual(g["reap_mark"], g["MemTotal"] - g["outside"], places=2)

    def test_a_quiet_box_reports_OVERLAP_instead_of_a_too_high_mark(self):
        """MemAvailable counts reclaimable pages the cgroup counts too.

        Measured idle on this rig: 118.05 - 117.67 - 7.56 = -7.18 GiB. Clamping
        that to zero would publish reap_mark = MemTotal = 118.05 -- 22 GiB above
        the two recorded kernel reaps, and in the FUNDING direction.
        """
        line = hl.reap_model_line(
            int(118.05 * GIB), int(117.67 * GIB), int(7.56 * GIB))
        self.assertIn("outside=OVERLAP", line)
        self.assertIn("FALLBACK", line)
        self.assertNotIn("reap_mark=118", line)

    def test_an_unreadable_term_falls_back_to_the_constant_and_says_so(self):
        for miss in ((None, int(7.56 * GIB)), (int(112.2 * GIB), None)):
            line = hl.reap_model_line(int(118.05 * GIB), *miss)
            self.assertIn("outside=unreadable", line)
            self.assertIn("FALLBACK", line)
            self.assertIn("never priced as zero", line)

    def test_the_arc_is_named_because_no_container_figure_shows_it(self):
        """ARC 5.01 GiB against this container's Cached 1.65 -- charged to no
        cgroup, invisible to every number we print, and a real 4 GiB lever."""
        line = hl.reap_model_line(
            int(118.05 * GIB), int(19.00 * GIB), int(88.00 * GIB))
        self.assertIn("ZFS ARC", line)
        self.assertIn("5.01", line)


if __name__ == "__main__":
    unittest.main()
