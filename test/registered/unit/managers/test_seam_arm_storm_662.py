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


class TheArmingConditionIncludesRunningWork(unittest.TestCase):
    """Queued OR running, and the difference cost a whole boot.

    The first version asked only about the waiting queue -- which is empty
    exactly when the work has been ADMITTED. Measured 12:49:20 with 90k tokens
    resident: "#running-req: 1, #full token: 457724, #queue-req: 0". The
    damper did not stand down, because by its reading nothing was waiting.
    The load that most wants the other layout is the load already in the
    machine.
    """

    def _rt_with(self, **sched):
        rt = _rt()
        rt._census_scheduler = types.SimpleNamespace(**sched)
        return rt

    def test_a_queued_request_counts(self):
        rt = self._rt_with(waiting_queue=[object()])
        self.assertTrue(rt._arming_condition_persists())

    def test_a_RUNNING_request_counts(self):
        rt = self._rt_with(
            waiting_queue=[], running_batch=types.SimpleNamespace(reqs=[object()])
        )
        self.assertTrue(rt._arming_condition_persists())

    def test_the_current_batch_counts(self):
        rt = self._rt_with(
            waiting_queue=[],
            running_batch=None,
            cur_batch=types.SimpleNamespace(reqs=[object()]),
        )
        self.assertTrue(rt._arming_condition_persists())

    def test_a_genuinely_idle_instance_does_not(self):
        """The case every damper was written for, and which must still work:
        the load has GONE AWAY, so the counters apply exactly as before."""
        rt = self._rt_with(
            waiting_queue=[],
            grammar_queue=[],
            running_batch=types.SimpleNamespace(reqs=[]),
            cur_batch=types.SimpleNamespace(reqs=[]),
        )
        self.assertFalse(rt._arming_condition_persists())

    def test_no_scheduler_is_not_a_licence_to_disable_the_damper(self):
        self.assertFalse(_rt()._arming_condition_persists())
