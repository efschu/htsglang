"""#699: detect the admission wedge that forward_ct and health-200 cannot see.

THE SPECIMEN, measured 2026-08-17 (#713). A TEN-token prompt waited 31.64 s for
its first token. Sampled every 2 s across the whole wait:

    num_running_reqs = 0        (nothing running)
    num_queue_reqs   = 1        (the request, queued)
    mamba_available  = 3        (never zero)
    kv_available     = hundreds of thousands

An 8-arm run put every TTFT between 11.87 s and 62.65 s, with NOT ONE arm under
3 s -- so this is the serving floor, not an outlier.

WHY THE EXISTING WATCHDOG IS BLIND, which is the reason this detector exists.
create_scheduler_watchdog fires when forward_ct stops advancing while a batch
exists. In this wedge BOTH read healthy: chunked prefill kept running so
forward_ct advanced, and cur_batch_for_debug stayed non-None. Health-200 is
blind for the same reason. The signal has to be QUEUE AGE VERSUS PROGRESS.
"""

import unittest

from sglang.srt.managers.scheduler_components.invariant_checker import (
    ADMISSION_WEDGE_SECONDS,
    admission_wedge_verdict,
)
from sglang.test.test_utils import CustomTestCase


class TestAdmissionWedgeDetector699(CustomTestCase):
    def test_the_31_second_specimen_alarms(self):
        """Replay of the measured shape: 1 queued, 0 running, 31.64 s."""
        alarm, detail = admission_wedge_verdict(1, 0, 31.64)
        self.assertTrue(alarm, detail)
        self.assertIn("ADMISSION-WEDGE", detail)
        self.assertIn("31.6", detail)

    def test_the_whole_sampled_sequence_alarms_once_past_threshold(self):
        """The specimen was sampled every 2 s; the detector must stay silent
        through the early samples and alarm only once the age qualifies."""
        seen = [admission_wedge_verdict(1, 0, t)[0] for t in range(0, 34, 2)]
        self.assertFalse(any(seen[: int(ADMISSION_WEDGE_SECONDS // 2)]))
        self.assertTrue(seen[-1], "must alarm by the end of a 31.64 s wait")

    def test_fastest_observed_wedge_is_still_caught(self):
        """The 8-arm run's fastest TTFT was 11.87 s -- but its wedge PERSISTS,
        so by the time the age reaches threshold it must fire. A threshold
        above the fastest arm would miss real wedges; this pins that the
        constant stays under it."""
        self.assertLess(ADMISSION_WEDGE_SECONDS, 62.65, "must catch the slowest arm")
        self.assertTrue(admission_wedge_verdict(1, 0, 25.0)[0])

    def test_busy_box_with_slow_prefill_is_SILENT(self):
        """CAN-FAIL, and the important one. A box that is SERVING must never
        alarm, however slow it is -- otherwise the detector cries wolf on every
        long prefill and gets filtered out exactly like health-200 was."""
        for running in (1, 2, 4):
            for age in (0.0, 30.0, 120.0):
                with self.subTest(running=running, age=age):
                    alarm, detail = admission_wedge_verdict(3, running, age)
                    self.assertFalse(alarm, detail)
                    self.assertIn("serving", detail)

    def test_empty_queue_is_silent_however_long_idle(self):
        """An idle box with nothing queued is not wedged, it is idle."""
        alarm, detail = admission_wedge_verdict(0, 0, 3600.0)
        self.assertFalse(alarm, detail)
        self.assertIn("no queue", detail)

    def test_below_threshold_is_silent(self):
        alarm, _ = admission_wedge_verdict(1, 0, ADMISSION_WEDGE_SECONDS - 0.1)
        self.assertFalse(alarm)
        alarm, _ = admission_wedge_verdict(1, 0, ADMISSION_WEDGE_SECONDS)
        self.assertTrue(alarm, "the threshold itself must qualify")

    def test_fires_WITHOUT_the_phase_policy_line(self):
        """The wedge class is broader than the phase-policy path. A detector
        that required corroboration would miss every wedge arising elsewhere."""
        alarm, detail = admission_wedge_verdict(1, 0, 31.64, idle_locked_seen=False)
        self.assertTrue(alarm)
        self.assertIn("broader than that path", detail)

    def test_corroboration_is_reported_when_present(self):
        alarm, detail = admission_wedge_verdict(1, 0, 31.64, idle_locked_seen=True)
        self.assertTrue(alarm)
        self.assertIn("IDLE-LOCKED TERMS corroborates", detail)

    def test_detail_always_quotes_its_terms(self):
        for args in ((1, 0, 31.64), (3, 2, 5.0), (0, 0, 99.0), (1, 0, 1.0)):
            with self.subTest(args=args):
                _, detail = admission_wedge_verdict(*args)
                self.assertTrue(detail and len(detail) > 20, detail)


if __name__ == "__main__":
    unittest.main()
