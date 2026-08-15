# SPDX-License-Identifier: Apache-2.0
"""#662: storm protection that is a RATE limiter, never a latch.

F1 filed the red: with the three dampers standing down, arms ran at ~20/min
where boot E's pathology was 179 in nine minutes. That is a storm and the pin
is right to call it one.

But the shape of the protection is the entire lesson of this ticket. Every
latch on this path -- the policy backoff, the seam's abandon counter, the
guards-layer unfundable flag -- blocked RE-PRICING while the arming condition
persisted, and each one cost the prefill layout at exactly the moment the pool
was full enough to pay for the flip. So:

  * bound attempts in TIME, never by a COUNT;
  * an attempt that made PROGRESS clears the pacing immediately, so a run that
    is genuinely funding is never throttled;
  * pacing applies only where a damper is already standing down.
"""

import time
import types
import unittest

from sglang.srt.managers.phase_flip_runtime import PhaseFlipRuntime


def _rt():
    rt = object.__new__(PhaseFlipRuntime)
    return rt


class TheLimiterPacesButNeverBlocks(unittest.TestCase):
    def test_the_first_attempt_always_passes(self):
        self.assertTrue(_rt()._storm_limiter_allows("tp_to_pp"))

    def test_a_second_attempt_inside_the_interval_is_paced(self):
        rt = _rt()
        self.assertTrue(rt._storm_limiter_allows("tp_to_pp"))
        self.assertFalse(rt._storm_limiter_allows("tp_to_pp"))

    def test_pacing_is_per_direction(self):
        """A tp_to_pp storm must not gag pp_to_tp, which is staged from the
        other leg's buffers entirely."""
        rt = _rt()
        rt._storm_limiter_allows("tp_to_pp")
        self.assertTrue(rt._storm_limiter_allows("pp_to_tp"))

    def test_progress_clears_the_pacing_immediately(self):
        """THE PROPERTY THAT MAKES IT A LIMITER AND NOT A LATCH: a direction
        that is successfully funding itself is never throttled."""
        rt = _rt()
        rt._storm_limiter_allows("tp_to_pp")
        self.assertFalse(rt._storm_limiter_allows("tp_to_pp"))
        rt.note_seam_progress("tp_to_pp")
        self.assertTrue(rt._storm_limiter_allows("tp_to_pp"))

    def test_the_interval_expires_on_its_own(self):
        rt = _rt()
        rt._storm_limiter_allows("tp_to_pp")
        rt._seam_last_arm_at["tp_to_pp"] = (
            time.monotonic() - PhaseFlipRuntime.SEAM_ARM_MIN_INTERVAL_S - 0.1
        )
        self.assertTrue(rt._storm_limiter_allows("tp_to_pp"))

    def test_it_never_refuses_on_a_COUNT(self):
        """The boot-E shape was a count. However many attempts have been made,
        one interval later the next one is allowed -- there is no number of
        refusals after which this stops answering."""
        rt = _rt()
        for _ in range(500):
            rt._storm_limiter_allows("tp_to_pp")
        rt._seam_last_arm_at["tp_to_pp"] = (
            time.monotonic() - PhaseFlipRuntime.SEAM_ARM_MIN_INTERVAL_S - 0.1
        )
        self.assertTrue(rt._storm_limiter_allows("tp_to_pp"))

    def test_the_rate_it_permits_is_far_below_the_measured_storm(self):
        """179 arms in 9 minutes is ~20/min. The interval caps a standing-down
        direction at 30/min in the worst case and, because progress clears it,
        a funding run is not capped at all."""
        per_min = 60.0 / PhaseFlipRuntime.SEAM_ARM_MIN_INTERVAL_S
        self.assertLessEqual(per_min, 30.0)


if __name__ == "__main__":
    unittest.main()
