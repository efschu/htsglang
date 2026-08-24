"""#854 -- the economy detector must say it RAN, not only that it alarmed.

THE SPECIMEN IS A HUMAN EYE, and that is the defect.

W25 (`/spinning/evidence-665-f1/boot_w25_0824_1125.log`, 2026-08-24) ran a
sustained TP-sticky prefill phase: `Prefill batch phase=tp` live on all three
ranks at 11:38:51 with 16-24k prompts, for roughly six minutes, and no further
cutover after 11:33:57. The user found it BY EYE. Meanwhile::

    grep -c "LAYOUT-ECONOMY ANOMALY" boot_w25_0824_1125.log   ->   0

for the entire boot. That is the fourth time this failure form has been caught
by a person rather than by the code, against a standing order that it never be.

THE DETECTOR IS EXONERATED ON THE ECONOMICS, AND THAT MUST BE SAID PLAINLY.
The hold reasons over the sticky phase, from the same log::

    19x  pending prefill <= N=18614, running it in tp        (BELOW the bar)
     6x  > N=18614 but <= 27921 with 1 req decoding          (inside the band)
    25x  tp_to_pp arm refused by the staging rate limit
    18x  decode bundle running -- drain mode
     7x  min dwell

Pending was 16-20k against a LIVE break-even of 18614 -- #819 reprices N from
the measured seam, and the seam had risen to 8.50s, so the bar legitimately
moved 7004 -> 18614. The policy's own arithmetic said TP was cheaper, and
class 2's gate 5 requires `pending > the bar the policy applied`. An alarm here
would have been precisely the over-eager mutant `test_layout_conformance_838.py
::test_mutant_overeager_detector_dies` exists to kill. So this is NOT a missing
alarm, and this file does NOT add one. (The real defect behind the sticky phase
is the SEAM COST that raised the bar -- W25 root R1, tracked separately.)

WHAT IS BROKEN IS THAT THE ZERO CANNOT BE READ. `LAYOUT-ECONOMY ANOMALY = 0`
conflates at least four states:

    ran every round and correctly declined      (what actually happened)
    never ran                                   (the W24 shape, one ticket ago)
    not wired on this pin
    threw inside the try/except and was swallowed

A reader -- or a window acceptance grep -- cannot separate them without reading
the source, which is why a person's eye ended up being the instrument. This is
the #851 defect class, the silent multi-valued zero, occurring INSIDE the
detector, one commit after #853(i) had to remove the identical shape from the
exposure gate (four silent-zero exits, five exclusive markers). The same remedy
applies: the instrument announces what it decided.

So the property under test is LIVENESS, not sensitivity:

    after this change, a running box CANNOT be silent. Either the code is not
    executing, or a line appears.

The can-fail direction is the whole risk of the change and is pinned below: a
fix that downgraded real anomalies into heartbeats would satisfy every liveness
assertion here while destroying the detector.
"""

import logging
import unittest

from sglang.srt.managers import layout_conformance as lc
from sglang.srt.managers.phase_policy import PHASE_PP, PHASE_TP
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5)

# --------------------------------------------------------------------------
# The W25 sticky-phase specimen, verbatim from the boot log.
# --------------------------------------------------------------------------

#: The live break-even at the time, repriced by #819 from the measured seam.
W25_BAR = 18614
#: The secondary-band ceiling the policy applied at 1 req decoding.
W25_APPLIED_BAR = 27921
#: A pending value from the sticky phase -- inside the band, below the bar
#: the policy applied, which is why the detector correctly declined.
W25_PENDING = 21729
#: The measured seam that moved the bar. This is root R1, not this ticket.
W25_SEAM_S = 8.50
#: One decode window on that seam: max(10.0, 2 x 8.50).
W25_WINDOW_S = 17.0
#: The phase had been held far longer than one decode window.
W25_HELD_S = 240.0


def _verdict(**over):
    """The W25 sticky-phase round, as the detector saw it."""
    kw = dict(
        phase=PHASE_TP,
        held_s=W25_HELD_S,
        window_s=W25_WINDOW_S,
        pending_prefill_tokens=W25_PENDING,
        live_flip_tokens=W25_BAR,
        applied_bar_tokens=W25_APPLIED_BAR,
        live_flip_cost_s=W25_SEAM_S,
        price_measured=True,
        hold_reason=(
            f"pending prefill {W25_PENDING} tok > N={W25_BAR} but "
            f"<= {W25_APPLIED_BAR} with 1 req decoding: too short for the "
            f"round trip to beat prefilling it in tp"
        ),
        since_flip_s=W25_HELD_S,
        min_dwell_s=3.0,
        staging_active=False,
        running_bs=1,
        bundle_at_phase_entry=1,
        bundle_stall_s=1.0,
    )
    kw.update(over)
    return lc.economy_divergence_verdict(**kw)


class _Fixture(unittest.TestCase):
    def setUp(self):
        lc.reset_for_test()

    def tearDown(self):
        lc.reset_for_test()


class TheStickyPhaseIsNoLongerSilent(_Fixture):
    """RED FIRST: before this change the W25 phase produced no line at all."""

    def test_the_w25_round_declines_rather_than_alarms(self):
        alarm, detail = _verdict()
        self.assertFalse(
            alarm,
            "pending sat below the bar the policy applied; alarming here is "
            "the over-eager mutant",
        )
        self.assertIn("correct economics", detail)

    def test_and_that_decline_is_ANNOUNCED(self):
        """The whole ticket in one assertion."""
        _, detail = _verdict()
        with self.assertLogs(lc.logger, level=logging.INFO) as caught:
            said = lc.note_economy_declined(detail, now=1000.0)
        self.assertTrue(said)
        blob = "\n".join(caught.output)
        self.assertIn(lc.CHECKED_ECONOMY, blob)

    def test_the_announcement_carries_the_bar_that_declined_it(self):
        """A heartbeat that says only "fine" would not have helped the user.

        The line has to carry the number the decision turned on, because the
        NEXT question after "is it alive" is always "then why is it holding".
        """
        _, detail = _verdict()
        with self.assertLogs(lc.logger, level=logging.INFO) as caught:
            lc.note_economy_declined(detail, now=1000.0)
        blob = "\n".join(caught.output)
        self.assertIn(str(W25_APPLIED_BAR), blob)
        self.assertIn(str(W25_PENDING), blob)

    def test_it_is_logged_at_info_because_the_boot_ran_at_info(self):
        """#853(i) had to correct exactly this on the exposure clamp: a
        liveness proof emitted below the boot's log level is not one."""
        _, detail = _verdict()
        with self.assertLogs(lc.logger, level=logging.INFO) as caught:
            lc.note_economy_declined(detail, now=1000.0)
        self.assertTrue(
            any(r.levelno >= logging.INFO for r in caught.records),
            "the decline must be visible at INFO",
        )


class TheZeroBecomesReadable(_Fixture):
    """c2=0 alone was four states. c2 with c2ok separates them."""

    def test_healthy_reads_as_ran_and_declined(self):
        _, detail = _verdict()
        for i in range(5):
            lc.note_economy_declined(detail, now=1000.0 + i)
        c = lc.counters()
        self.assertEqual(0, c.economy_anomalies)
        self.assertEqual(5, c.economy_checks)

    def test_never_ran_is_now_distinguishable_from_healthy(self):
        """The state W24 could not tell apart, as a test."""
        c = lc.counters()
        self.assertEqual(0, c.economy_anomalies)
        self.assertEqual(
            0,
            c.economy_checks,
            "c2=0 AND c2ok=0 is the never-ran state and must stay legible",
        )

    def test_the_periodic_field_reports_the_check_count(self):
        _, detail = _verdict()
        lc.note_economy_declined(detail, now=1000.0)
        self.assertIn("c2ok=1", lc.counters().as_field())
        self.assertIn("c2=0", lc.counters().as_field())

    def test_the_count_is_not_throttled_even_though_the_log_is(self):
        """Same split #838 already got right for alarms: the throttle bounds
        the LOG, never the measurement."""
        _, detail = _verdict()
        said = [lc.note_economy_declined(detail, now=1000.0 + i) for i in range(20)]
        self.assertEqual(20, lc.counters().economy_checks)
        self.assertEqual(
            1, sum(said), "one heartbeat per DECLINE_REANNOUNCE_S, not twenty"
        )

    def test_the_heartbeat_returns_after_its_cadence(self):
        _, detail = _verdict()
        self.assertTrue(lc.note_economy_declined(detail, now=1000.0))
        self.assertFalse(
            lc.note_economy_declined(detail, now=1000.0 + lc.DECLINE_REANNOUNCE_S - 1)
        )
        self.assertTrue(
            lc.note_economy_declined(detail, now=1000.0 + lc.DECLINE_REANNOUNCE_S + 1)
        )


class TheDetectorIsNotDowngraded(_Fixture):
    """THE CAN-FAIL DIRECTION. This is the entire risk of the change."""

    def test_a_real_anomaly_still_alarms(self):
        alarm, detail = _verdict(pending_prefill_tokens=W25_APPLIED_BAR + 50_000)
        self.assertTrue(alarm)
        self.assertIn(lc.ALARM_ECONOMY, detail)

    def test_an_alarm_is_counted_as_an_alarm_and_never_as_a_check(self):
        _, detail = _verdict(pending_prefill_tokens=W25_APPLIED_BAR + 50_000)
        lc.note_economy_anomaly(detail, now=1000.0)
        self.assertEqual(1, lc.counters().economy_anomalies)
        self.assertEqual(
            0,
            lc.counters().economy_checks,
            "an anomaly recorded as a healthy check would hide the defect "
            "behind the very instrument added to expose it",
        )

    def test_the_two_markers_are_distinct_strings(self):
        """A grep for the alarm must never match a heartbeat.

        This assertion earned itself in #853(i), where a marker quoted another
        marker's text inside its own prose and restored the ambiguity while
        passing every other test.
        """
        self.assertNotIn(lc.CHECKED_ECONOMY, lc.ALARM_ECONOMY)
        self.assertNotIn(lc.ALARM_ECONOMY, lc.CHECKED_ECONOMY)

    def test_an_alarm_line_does_not_contain_the_checked_marker(self):
        _, detail = _verdict(pending_prefill_tokens=W25_APPLIED_BAR + 50_000)
        self.assertNotIn(lc.CHECKED_ECONOMY, detail)

    def test_a_declined_pp_round_is_still_a_decline_not_an_alarm(self):
        """The detector makes no claim in PP, and "no claim" is still a fact
        worth recording -- it is how a reader knows the box is in PP and the
        instrument is alive there too."""
        alarm, detail = _verdict(phase=PHASE_PP)
        self.assertFalse(alarm)
        with self.assertLogs(lc.logger, level=logging.INFO) as caught:
            lc.note_economy_declined(detail, now=1000.0)
        self.assertIn(lc.CHECKED_ECONOMY, "\n".join(caught.output))


if __name__ == "__main__":
    unittest.main()
