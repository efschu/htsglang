"""#739: the detector must say WHICH wedge, or say UNCLEAR.

Both fixtures below are the real specimens' numbers, not convenient ones:

  A  wedge_802f_1712 (2026-08-22 17:10): all three ranks idle in blocking
     receives, batch lines stopped, relief posts UNCONSUMED five times.
  B  wedge_arm1_1845 (2026-08-22 18:45): 28-40 batch lines per minute
     THROUGHOUT the alarm, relief posts consumed and answered NOT APPLICABLE
     three times, resolved by ordinary completion.

The two CROSS combinations are the named falsifiers of the split itself and
must come back UNCLEAR. If either of them ever silently became A or B, the
classifier would have stopped being evidence -- which is the whole failure it
exists to prevent.

CPU-only: the classifier is a pure function and the call-edge case binds the
shipped detector to a stand-in.
"""

import types
import unittest

from sglang.srt.managers.wedge_class import (
    CLASS_PIPELINE_DEAD,
    CLASS_POOL_SATURATED,
    CLASS_UNCLEAR,
    classify_wedge,
)
from sglang.srt.managers.wedge_recovery import (
    STATE_ACTUATED,
    STATE_NOT_APPLICABLE,
    STATE_PENDING,
    STATE_UNCONSUMED,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15)

#: 1845 ran 28-40 batch lines per minute across a ~175 s alarm.
B_FORWARD_DELTA = 96
B_WINDOW_S = 175.0

#: 1712 produced none at all across its window.
A_FORWARD_DELTA = 0
A_WINDOW_S = 300.0


class WedgeClassification(unittest.TestCase):
    def test_the_1712_numbers_classify_as_a_pipeline_dead(self):
        got = classify_wedge(A_FORWARD_DELTA, STATE_UNCONSUMED, window_s=A_WINDOW_S)
        self.assertEqual(got.label, CLASS_PIPELINE_DEAD)
        self.assertIn("forward_delta=0", got.detail)
        self.assertIn(STATE_UNCONSUMED, got.detail)
        self.assertIn("#788", got.detail, "class A must name the family to look in")

    def test_the_1845_numbers_classify_as_b_pool_saturated(self):
        got = classify_wedge(B_FORWARD_DELTA, STATE_NOT_APPLICABLE, window_s=B_WINDOW_S)
        self.assertEqual(got.label, CLASS_POOL_SATURATED)
        self.assertIn(str(B_FORWARD_DELTA), got.detail)
        self.assertIn("Not a comms defect", got.detail)

    def test_can_fail_no_progress_plus_consumed_post_is_unclear_not_a(self):
        """THE NAMED FALSIFIER of this very split.

        Nothing computes, yet the scheduler thread evidently took the post.
        That contradicts A's own mechanism, so it must be visible rather than
        rounded into A -- which is exactly what a detector that "helpfully"
        picks the nearest class would do.
        """
        got = classify_wedge(0, STATE_NOT_APPLICABLE, window_s=A_WINDOW_S)
        self.assertEqual(
            got.label,
            CLASS_UNCLEAR,
            "the falsifier of the A/B split was absorbed into a class",
        )
        self.assertIn("falsifier", got.detail)

    def test_can_fail_progress_plus_unconsumed_post_is_unclear_not_b(self):
        """The mirror image, and equally a reason to doubt the split."""
        got = classify_wedge(B_FORWARD_DELTA, STATE_UNCONSUMED, window_s=B_WINDOW_S)
        self.assertEqual(got.label, CLASS_UNCLEAR)
        self.assertIn("mirror falsifier", got.detail)

    def test_missing_evidence_is_unclear_and_says_which_half_is_missing(self):
        no_sample = classify_wedge(None, STATE_UNCONSUMED)
        self.assertEqual(no_sample.label, CLASS_UNCLEAR)
        self.assertIn("no forward-progress sample", no_sample.detail)

        no_post = classify_wedge(0, None, window_s=A_WINDOW_S)
        self.assertEqual(no_post.label, CLASS_UNCLEAR)
        self.assertIn("has not answered", no_post.detail)

    def test_a_state_outside_the_split_is_unclear_by_name(self):
        for state in (STATE_ACTUATED, STATE_PENDING):
            got = classify_wedge(0, state, window_s=A_WINDOW_S)
            self.assertEqual(got.label, CLASS_UNCLEAR, f"{state} decided a class")
            self.assertIn(repr(state), got.detail)

    def test_can_fail_saturation_alone_never_decides_a_class(self):
        """Saturation is the steady state of a busy server.

        It was already true minutes BEFORE the 1845 alarm began, while the
        server decoded normally. A classifier that keyed on it would have
        called that specimen a wedge before it was one, so it may only ever
        decorate a verdict the two real signals already reached.
        """
        dead = classify_wedge(0, STATE_UNCONSUMED, usage_at_ceiling=True)
        self.assertEqual(dead.label, CLASS_PIPELINE_DEAD)
        unclear = classify_wedge(0, STATE_NOT_APPLICABLE, usage_at_ceiling=True)
        self.assertEqual(unclear.label, CLASS_UNCLEAR)
        busy = classify_wedge(5, STATE_NOT_APPLICABLE, usage_at_ceiling=True)
        self.assertEqual(busy.label, CLASS_POOL_SATURATED)
        self.assertIn("ceiling", busy.detail)


def _stub_scheduler(queued: int, running: int, age: float, forward_ct: int):
    return types.SimpleNamespace(
        is_initializing=False,
        waiting_queue=[object()] * queued,
        running_batch=types.SimpleNamespace(reqs=[object()] * running),
        last_first_token_progress_time=0.0,
        last_prefill_progress_time=None,
        forward_ct=forward_ct,
        _wedge_class_sample=None,
    )


class WedgeClassCallEdge(unittest.TestCase):
    """THE CALL EDGE, not just the helper.

    The class has to reach the SAME alarm line, or a future reader still has
    to correlate two logs -- which is the cost this change exists to remove.
    """

    def _fire(self, forward_ct_seq):
        from sglang.srt.managers.scheduler_components.invariant_checker import (
            check_admission_wedge_once,
        )

        sched = _stub_scheduler(queued=8, running=0, age=120.0, forward_ct=0)
        out = []
        for i, ct in enumerate(forward_ct_seq):
            sched.forward_ct = ct
            alarm, detail = check_admission_wedge_once(
                sched, now=120.0 + i * 60.0, log_on_alarm=False
            )
            out.append((alarm, detail))
        return sched, out

    def test_the_shipped_detector_appends_a_class_to_its_alarm(self):
        _sched, out = self._fire([0, 0])
        alarm, detail = out[-1]
        self.assertTrue(alarm, "the stub did not trip the detector at all")
        self.assertIn("CLASS=", detail, "the shipped alarm carries no class")

    def test_the_window_is_stamped_once_and_the_delta_grows(self):
        """The delta must be 'since this wedge began', not a per-call figure."""
        _sched, out = self._fire([0, 40])
        self.assertIn("forward_delta=40", out[-1][1])

    def test_the_stamp_clears_when_the_alarm_clears(self):
        from sglang.srt.managers.scheduler_components.invariant_checker import (
            check_admission_wedge_once,
        )

        sched = _stub_scheduler(queued=8, running=0, age=120.0, forward_ct=0)
        check_admission_wedge_once(sched, now=120.0, log_on_alarm=False)
        self.assertIsNotNone(sched._wedge_class_sample, "no window was stamped")
        sched.waiting_queue = []
        check_admission_wedge_once(sched, now=180.0, log_on_alarm=False)
        self.assertIsNone(
            sched._wedge_class_sample,
            "a cleared alarm kept its window, so the NEXT wedge would inherit "
            "this one's tail and read as progress it never made",
        )


if __name__ == "__main__":
    unittest.main()
