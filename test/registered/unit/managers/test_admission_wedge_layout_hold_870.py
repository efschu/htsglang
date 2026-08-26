"""#870: a legitimate layout-admission hold is not an admission wedge.

THE DEFECT THIS PINS. On the #857 strict-batch acceptance boot the #699
detector alarmed 41 times (23 distinct rounds across PP0/PP1/PP2) on a boot
whose acceptance PASSED and which answered a probe with HTTP 200 throughout.
Every line said "no phase-policy corroboration seen" -- while the phase policy
was logging, in the same second:

    PHASE-POLICY arming pp_to_tp: IDLE-LOCKED: no batch of either work class
    can be built in the pp layout (8 req resident, 0 tok prefill pending)

-- the same 8 requests the detector was calling a wedge.

TWO defects, and the second survives fixing the first:

1. ``idle_locked_seen`` was UNWIRED. The parameter existed
   (invariant_checker.py:554), was documented, and was covered by tests, but
   ``check_admission_wedge_once`` -- the only production caller -- never passed
   it. It could not be True on a live boot no matter what the policy did.
2. Even when True it only decorated the MESSAGE. The verdict never read it.

WHY THIS IS NOT "TURN THE ALARM DOWN". The suppression requires a NAMED hold
AND an ARMED flip AND a bounded age. Take any one away and the alarm returns:
a hold nothing is ending is a wedge, an arm nobody can attribute is not an
explanation, and a hold that outlives its window is a STUCK FLIP -- which still
alarms, but says so instead of blaming admission. That last direction is the
one that keeps this a detector rather than a mute button, and it is pinned
below in both directions.

The 17 true hits of #699/#739 carry none of these signals, so they are
untouched: no arm, no hold stamp -> the pre-#870 verdict, exactly.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

import unittest

from sglang.srt.managers.scheduler_components.invariant_checker import (
    ADMISSION_WEDGE_HOLD_GRACE_SECONDS,
    ADMISSION_WEDGE_SECONDS,
    admission_wedge_verdict,
)

_AGE = ADMISSION_WEDGE_SECONDS + 10.0
_REASON = (
    "IDLE-LOCKED: no batch of either work class can be built in the pp layout "
    "(8 req resident, 0 tok prefill pending)"
)


class ALegitimateLayoutHoldIsNotAWedge(unittest.TestCase):
    def test_the_live_857_signature_no_longer_alarms(self):
        """The exact numbers from the 23:46:26 line, plus the signals that were missing."""
        alarm, detail = admission_wedge_verdict(
            8,
            0,
            _AGE,
            seconds_since_prefill_progress=_AGE,
            flip_armed=True,
            seconds_since_layout_hold=2.0,
            layout_hold_reason=_REASON,
        )
        self.assertFalse(alarm, detail)
        self.assertIn("LAYOUT-ADMISSION HOLD", detail)
        self.assertIn("IDLE-LOCKED", detail)


class TheSuppressionMustBeAbleToFail(unittest.TestCase):
    """Can-fail in every direction, because a suppression that cannot fail is a mute."""

    def test_a_real_wedge_is_still_caught(self):
        # No arm, no hold: the #699/#739 specimen shape.
        alarm, detail = admission_wedge_verdict(
            1, 0, _AGE, seconds_since_prefill_progress=_AGE
        )
        self.assertTrue(alarm, detail)
        self.assertIn("ADMISSION-WEDGE", detail)

    def test_an_unreadable_hold_still_alarms(self):
        # The signals are absent (older scheduler, or a stand-in): pre-#870.
        alarm, _ = admission_wedge_verdict(
            1,
            0,
            _AGE,
            seconds_since_prefill_progress=_AGE,
            flip_armed=None,
            seconds_since_layout_hold=None,
            layout_hold_reason=None,
        )
        self.assertTrue(alarm)

    def test_a_hold_with_no_armed_flip_still_alarms(self):
        # A named hold that nothing is ending is exactly a wedge.
        alarm, _ = admission_wedge_verdict(
            1,
            0,
            _AGE,
            seconds_since_prefill_progress=_AGE,
            flip_armed=False,
            seconds_since_layout_hold=2.0,
            layout_hold_reason=_REASON,
        )
        self.assertTrue(alarm)

    def test_an_armed_flip_with_no_named_reason_still_alarms(self):
        # An arm this detector cannot attribute must not be trusted to explain
        # the silence.
        alarm, _ = admission_wedge_verdict(
            1,
            0,
            _AGE,
            seconds_since_prefill_progress=_AGE,
            flip_armed=True,
            seconds_since_layout_hold=2.0,
            layout_hold_reason=None,
        )
        self.assertTrue(alarm)

    def test_a_hold_that_outlives_its_window_alarms_as_a_STUCK_FLIP(self):
        # The boundary between two different defects, not a safety margin.
        alarm, detail = admission_wedge_verdict(
            1,
            0,
            _AGE,
            seconds_since_prefill_progress=_AGE,
            flip_armed=True,
            seconds_since_layout_hold=ADMISSION_WEDGE_HOLD_GRACE_SECONDS + 1.0,
            layout_hold_reason=_REASON,
        )
        self.assertTrue(alarm, detail)
        self.assertIn("STUCK FLIP", detail)
        self.assertIn("Look at the flip, not at admission", detail)

    def test_the_grace_boundary_is_exact(self):
        def at(hold_age):
            return admission_wedge_verdict(
                1,
                0,
                _AGE,
                seconds_since_prefill_progress=_AGE,
                flip_armed=True,
                seconds_since_layout_hold=hold_age,
                layout_hold_reason=_REASON,
            )[0]

        self.assertFalse(at(ADMISSION_WEDGE_HOLD_GRACE_SECONDS - 0.1))
        self.assertTrue(at(ADMISSION_WEDGE_HOLD_GRACE_SECONDS))


class TheOlderVerdictsAreUnchanged(unittest.TestCase):
    """#870 may not move any boundary #699 and #739 already pinned."""

    def test_running_still_beats_everything(self):
        alarm, _ = admission_wedge_verdict(
            8,
            1,
            _AGE,
            flip_armed=True,
            seconds_since_layout_hold=2.0,
            layout_hold_reason=_REASON,
        )
        self.assertFalse(alarm)

    def test_prefill_progress_still_suppresses_without_any_hold_signal(self):
        alarm, detail = admission_wedge_verdict(
            1, 0, _AGE, seconds_since_prefill_progress=1.0
        )
        self.assertFalse(alarm)
        self.assertIn("PREFILLING", detail)

    def test_below_threshold_is_still_ordinary_latency(self):
        alarm, _ = admission_wedge_verdict(1, 0, ADMISSION_WEDGE_SECONDS - 0.1)
        self.assertFalse(alarm)


class OneRootKillsBothTicketsIncluding866(unittest.TestCase):
    """#866 is DOWNSTREAM of #870, not a second defect.

    On the same boot the recovery actuator posted corridor-relief requests and
    the gate answered ``exit 'headroom-sufficient'`` -- it looked for a
    headroom shortage and correctly found none, because a layout hold is not a
    shortage. It ran at all only because the wedge alarm had been standing for
    longer than the recovery threshold.

    The link is structural rather than incidental: ``AdmissionWedgeRecovery.
    step`` returns ``None`` before allocating anything whenever ``alarm`` is
    false (invariant_checker.py:955). So a detector that no longer alarms on a
    legitimate hold cannot post a relief request for one either, and #866
    needs no separate fix. Pinned end to end below rather than argued.
    """

    def _sched(self, *, with_hold):
        from types import SimpleNamespace

        s = SimpleNamespace()
        s.is_initializing = False
        s.waiting_queue = [object()] * 8
        s.running_batch = SimpleNamespace(reqs=[])
        s.last_first_token_progress_time = -_AGE
        s.last_prefill_progress_time = -_AGE
        s.server_args = SimpleNamespace(enable_phase_flip=True)
        if with_hold:
            s.phase_flip_runtime = SimpleNamespace(is_armed=lambda: True)
            s.last_layout_hold_time = -2.0
            s.last_layout_hold_reason = _REASON
        return s

    def test_the_live_signature_end_to_end_alarms_only_without_the_hold(self):
        from sglang.srt.managers.scheduler_components.invariant_checker import (
            check_admission_wedge_once,
        )

        alarm_held, detail_held = check_admission_wedge_once(
            self._sched(with_hold=True), now=0.0
        )
        alarm_bare, _ = check_admission_wedge_once(
            self._sched(with_hold=False), now=0.0
        )
        self.assertFalse(alarm_held, detail_held)
        self.assertTrue(alarm_bare, "a wedge with no hold signals must still alarm")

    def test_no_alarm_means_no_866_relief_request(self):
        from sglang.srt.managers.scheduler_components.invariant_checker import (
            AdmissionWedgeRecovery,
        )

        sched = self._sched(with_hold=True)
        driver = AdmissionWedgeRecovery(sched)
        self.assertIsNone(driver.step(alarm=False))
        # The channel is created by the first post and by nothing else, so its
        # absence is the evidence that nothing was posted.
        self.assertIsNone(getattr(sched, "admission_wedge_recovery_channel", None))


class TheThirdInstanceLayoutConformance838(unittest.TestCase):
    """#838's permitted one-chunk TP self-prefill -- the same class again.

    The user permitted the TP phase to prefill up to ONE CHUNK itself on
    2026-08-25 and predicted this blind spot in the same breath. On the #857
    boot the detector flagged 165 batches -- EVERY TP prefill batch on a boot
    whose acceptance passed -- with new_tokens between 14 and 170 against a
    4096 budget.

    The discriminator is NOT "is the number small". It is "did the chunk
    budget BIND": a batch at the budget was truncated by it, so more prefill
    stands behind it. W37-D's 258 batches measured exactly 4096 with 0 cached,
    and a cap written `<=` would have called every one of them permitted.
    """

    BUDGET = 4096

    def _verdict(self, **kw):
        from sglang.srt.managers import layout_conformance

        args = dict(
            batch_class="prefill",
            phase="tp",
            strict=True,
            transport_verified=True,
            n_reqs=8,
            new_tokens=160,
            cached_tokens=0,
            now=0.0,
            chunk_budget=self.BUDGET,
        )
        args.update(kw)
        return layout_conformance.work_layout_verdict(**args)

    def test_the_live_857_batches_are_permitted(self):
        # Every distinct new_tokens value measured in the boot log.
        for n in (14, 18, 20, 25, 52, 67, 70, 160, 167, 170):
            self.assertIsNone(self._verdict(new_tokens=n), f"new_tokens={n}")

    def test_W37D_IS_STILL_A_VIOLATION(self):
        """The can-fail that matters: the defect #838 exists for."""
        detail = self._verdict(new_tokens=self.BUDGET, n_reqs=258)
        self.assertIsNotNone(detail, "a batch AT the budget is chunk-limited")
        self.assertIn("work_in_wrong_layout", detail)

    def test_beyond_the_budget_is_a_violation(self):
        self.assertIsNotNone(self._verdict(new_tokens=self.BUDGET + 1))

    def test_without_a_budget_the_pre_870_verdict_is_unchanged(self):
        self.assertIsNotNone(self._verdict(chunk_budget=None))

    def test_a_restore_of_any_size_never_trips_the_cap(self):
        # The cap is measured on COMPUTED tokens. Restored tokens are not in
        # that quantity, so restore is unlimited by construction.
        self.assertIsNone(self._verdict(new_tokens=8, cached_tokens=1_000_000))
        self.assertIsNone(self._verdict(new_tokens=1, cached_tokens=500_000))

    def test_a_huge_recompute_wearing_transports_clothes_still_alarms(self):
        # cached_tokens == 0 under a transport claim, above the budget: the
        # W37-G shape, untouched by the new allowance.
        self.assertIsNotNone(
            self._verdict(new_tokens=100_000, cached_tokens=0, transport_verified=True)
        )

    def test_non_strict_is_still_silent(self):
        self.assertIsNone(self._verdict(strict=False, new_tokens=self.BUDGET))

    def test_the_allowance_does_not_leak_to_decode_in_pp(self):
        # The permission the user gave is for TP self-PREFILL only.
        self.assertIsNotNone(
            self._verdict(batch_class="decode", phase="pp", new_tokens=160)
        )

    def test_the_budget_boundary_is_exact(self):
        self.assertIsNone(self._verdict(new_tokens=self.BUDGET - 1))
        self.assertIsNotNone(self._verdict(new_tokens=self.BUDGET))


if __name__ == "__main__":
    unittest.main()
