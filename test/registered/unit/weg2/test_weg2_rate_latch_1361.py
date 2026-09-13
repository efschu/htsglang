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


class TheLatchIsWIRED1361(CustomTestCase):
    """#1361 (1) RATCHET: a built-but-unwired guard is the #464 / #1017 class.

    The latch is worthless in a module. It has to ride the loop that already
    carries W22 -- the same clock, so `gaps_seen` counts THAT loop's blindness
    and not another sampler's, and the same named teardown, so a projection
    breach ends the way the user ordered a level breach to end on 2026-09-08:
    controlled, never a kernel kill.
    """

    def _loop_src(self):
        import inspect
        from sglang.srt.weg2 import front as fr
        src = inspect.getsource(fr)
        i = src.index("host_ledger.watermark_provenance(margin)")
        return src[i:src.index("def do_stop", i)]

    def test_the_guard_loop_constructs_the_rate_latch(self):
        self.assertIn("host_ledger.RateLatch(", self._loop_src())

    def test_it_is_fed_and_its_verdict_reaches_the_named_teardown(self):
        body = self._loop_src()
        self.assertIn("rate_latch.observe(", body)
        self.assertIn('do_stop("W98 Weg2HostRateLatched"', body)
        # ...and a GAP is reported but never torn down: the loop kept running,
        # it just could not look.
        self.assertIn("RATE-GAP", body)
        gap_branch = body[body.index("RATE-GAP"):body.index("elif _line is not None")]
        self.assertNotIn("do_stop", gap_branch)

    def test_the_latch_runs_faster_than_the_level_test(self):
        """A latch no faster than the guard that went blind buys nothing."""
        body = self._loop_src()
        self.assertIn("rate_period_s = min(0.5,", body)
        self.assertIn("_level_due", body)      # the level test keeps its period

    def test_both_halves_grade_the_same_reading(self):
        """EXACTLY ONE `read_cgroup_pressure` per tick, counted -- not greped.

        #1361 20c. The first version of this test carried this very docstring
        and asserted only that the string `_pr_fast = ...read_cgroup_pressure()`
        APPEARS. The loop took TWO readings and the test stayed green; it would
        have stayed green at ten. Docstring and assertion measured different
        things, which is the same class as "unit test and acceptance measured
        different things" one level tighter -- and it is why the property is
        now COUNTED.
        """
        body = self._loop_src()
        self.assertEqual(body.count("read_cgroup_pressure()"), 1)
        self.assertIn("_pr_fast = host_ledger.read_cgroup_pressure()", body)
        self.assertIn("pr = _pr_fast", body)


#: #1361b THE xsn27 STARTUP SLEEP LEG, from its own v2 sampler
#: (weg2xsn27_0913/hostsample_v2_weg2xsn27.csv), (t_s, nonreclaim_gib) with t
#: relative to 00:12:57.052. The kill fell inside the 33 s sampler hole that
#: opens after the last row.
XSN27_LEG = (
    (0.000, 71.08), (0.506, 71.58), (1.012, 72.00), (1.517, 72.17),
    (2.023, 72.47), (2.528, 72.63), (3.034, 72.74), (3.538, 72.74),
    (4.042, 72.82), (4.547, 72.97), (5.052, 73.00), (5.557, 73.00),
    (6.063, 73.89), (6.579, 78.44), (7.100, 80.80), (7.819, 82.47),
)
#: The plan that leg was writing: `WEG2-XCHG-PLAN d2h 16.4 + 5.4 + 5.4 GiB`.
XSN27_PLAN_GIB = 27.2
#: shmem at the leg's start, so "already written" is derivable per row.
XSN27_SHMEM0 = 49.69


def _xsn27_remaining(nr, shmem):
    return max(0.0, XSN27_PLAN_GIB - (shmem - XSN27_SHMEM0))


class TheReplayFiresBeforeTheKillAndNotBefore1361b(CustomTestCase):
    """#1361b ACCEPTANCE: the measured leg, replayed through the bounded latch.

    Two halves, and BOTH have to hold or the latch is useless:
      * it must fire on weg2xsn27's startup sleep leg BEFORE the sampler goes
        blind at 00:13:04.87 (the kernel reaped inside that 33 s hole);
      * it must NOT fire earlier on the same leg, where the climb is real but
        the plan still fits -- the naive `rate x 5 s` form fires at 00:12:57
        (71.1 + 5.4x5 = 98 > 95.9) on a leg that had every right to run, and a
        latch that fires there kills every healthy boot.
    """

    #: shmem per row, needed for the remaining-bytes bound.
    SHMEM = (49.69, 49.69, 49.69, 49.73, 49.78, 49.83, 49.88, 49.88, 49.90,
             50.02, 50.02, 50.02, 50.86, 55.40, 57.77, 60.39)

    def _run(self, bounded):
        lat = hl.RateLatch(reap_mark_gib=95.90)
        for (t, nr), sh in zip(XSN27_LEG, self.SHMEM):
            rem = _xsn27_remaining(nr, sh) if bounded else None
            out = lat.observe(t, nr, remaining_leg_gib=rem)
            if out and "W98" in out:
                return t, out
        return None, None

    def test_the_leg_is_refused_at_its_FIRST_sample_because_it_never_fitted(self):
        """THE MEASUREMENT CORRECTED THE DESIGN, and this is the number.

        The order asked for `min(rate x lookahead, remaining)`. Replayed against
        weg2xsn27's own series that is strictly WORSE: the rate is ~1 GiB/s for
        the first six seconds, so a rate-bounded projection stays silent through
        a leg that was already impossible -- 71.08 + 27.20 = **98.28** against
        the 95.90 mark AT THE FIRST ROW. The leg never fitted; nothing about its
        slope was the finding. So the test is `now + remaining`, it needs no
        rate, and it fires 7.8 s before the sampler goes blind instead of 1.2 s.
        """
        t, line = self._run(bounded=True)
        self.assertIsNotNone(line, "the latch must fire on this leg")
        self.assertEqual(t, 0.0)
        self.assertAlmostEqual(XSN27_LEG[0][1] + XSN27_PLAN_GIB, 98.28, places=2)
        self.assertGreater(XSN27_LEG[0][1] + XSN27_PLAN_GIB, 95.90)
        self.assertIn("remaining_leg=", line)
        self.assertIn("FEASIBILITY", line)
        self.assertIn("no rate needed", line)

    def test_the_rate_form_would_have_watched_six_seconds_of_a_doomed_leg(self):
        """CAN-FAIL for the design change: measured, not argued.

        `rate x lookahead` only reaches the mark once the ramp arrives at
        t=6.579 -- six seconds and 7.4 GiB later, with the sampler about to go
        blind. Same data, same mark, a whole leg of difference.
        """
        lat = hl.RateLatch(reap_mark_gib=95.90)
        fired = None
        for t, nr in XSN27_LEG:
            if lat.observe(t, nr) and fired is None:
                fired = t
        self.assertAlmostEqual(fired, 6.579, places=2)
        self.assertGreater(fired, 6.0)

    def test_the_rate_still_owns_the_case_with_no_plan_to_read(self):
        """`remaining=None` keeps the pre-#1361b behaviour, unchanged."""
        lat = hl.RateLatch(reap_mark_gib=95.90)
        for i, v in enumerate(XSN26B_RAMP):
            lat.observe(float(i), v + 26.4)
        self.assertTrue(lat.latched)

    def test_a_healthy_leg_with_room_never_latches(self):
        """rg6/sb5f shape: the same climb, a plan that fits. Must stay silent."""
        lat = hl.RateLatch(reap_mark_gib=95.90)
        nr = 60.0
        for i in range(12):
            nr += 1.5                      # a brisk, legitimate ring fill
            out = lat.observe(i * 0.5, nr, remaining_leg_gib=max(0.0, 18.0 - i * 1.5))
            self.assertIsNone(out, f"latched at i={i}, nr={nr}")
        self.assertFalse(lat.latched)


class TheLaunchMomentChargesBothAtOnce1361c(CustomTestCase):
    def test_the_two_terms_are_summed_not_maxed(self):
        peak, line = hl.launch_moment_peak_gib(
            anon_load_peak_gib=5.2, ring_fill_gib=11.0, origin_gib=67.2)
        self.assertAlmostEqual(peak, 83.4, places=2)
        self.assertIn("SIMULTANEOUS, not sequential", line)
        self.assertIn("weg2xsn27", line)

    def test_it_reproduces_the_boots_own_thirteen_seconds(self):
        """anon 17.8 -> 23.0 (+5.2) and shmem 49.4 -> 60.4 (+11.0) OVERLAP;
        nonreclaim went 67.2 -> 82.5, i.e. the SUM, not the larger."""
        peak, _l = hl.launch_moment_peak_gib(5.2, 11.0, 67.2)
        self.assertAlmostEqual(peak, 82.5, delta=1.0)
        self.assertGreater(peak, 67.2 + max(5.2, 11.0))   # a max() model misses it
