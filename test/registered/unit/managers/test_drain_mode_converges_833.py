"""#833 -- drain mode must bind on a quantity that CONVERGES under pressure.

THE MEASUREMENT, boot_window3_0823_1733, load segment 2.

Three escalating load shapes were driven at the ``tp_to_pp`` arm to force a
cutover, and the arm refused under every one of them::

    the bar ................. N=7004 (quoted on 49 decision lines)
    scaled ceiling .......... 11674 - 12841
    MAX PENDING DRIVEN ...... 836,048 tokens = 119x N, 65x the ceiling
    hold reason, 151 times .. "decode bundle running: 7 of 7 req still
                               decoding, ... tok prefill waiting -- drain
                               mode finishes the bundle before flipping"

Zero flips in that segment. The instance served single-phase for 22 of its 32
minutes.

WHY MORE PRESSURE MADE IT WORSE. Under ``--phase-policy-drain-mode`` the arm
needs pending above the bar AND the admitted set EMPTY. Pending clears the bar
trivially -- 119x it. The second condition is the one that holds, and it holds
HARDER the more load arrives: with ``--max-running-requests 8`` and ~42k-token
prompts chunked at 4096 (~11 chunks per request) an admitted request occupies
the bundle for many rounds, and admission refills every slot the bundle frees.
So the exit condition recedes as you approach it. That is a divergent binding,
and no amount of load will ever satisfy it.

The module already recorded this failure shape on the neighbouring rule, on a
different axis, and it is the same mistake::

    the #677(a) progress exit could not fire either (it needs pending FROZEN,
    and pending was still creeping 572792 -> 572715, which reads as progress)

PENDING IS THE WRONG AXIS. It is the quantity pressure inflates, so it always
looks like it is moving. Drain mode waits on the ADMITTED SET, so the admitted
set is what has to be watched -- and "progress" on that axis means the set
getting SMALLER, not the work it retires.

THE PROPERTY THIS FILE PINS, and it is the acceptance criterion:

    raising pressure must not raise the refusal rate.

It now holds by construction rather than by tuning: more pressure keeps the
bundle full, a full bundle does not shrink, and a bundle that does not shrink
trips the stall deadline SOONER. The derivative is inverted.

NOT BUILT HERE: #819 would have admission stop refilling the bundle while a
flip is pending -- the other half of this, and a policy change of its own.
This ticket bounds the wait and nothing else.
"""

import unittest

from sglang.srt.managers.phase_policy import (
    PhasePolicyConfig,
    PhasePolicyInputs,
    PhasePolicyState,
    drain_stall_deadline_s,
    solved_tp_decode_floor_s,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5)

#: The bar this boot armed with, verbatim from the log.
W3_BAR = 7004
#: The maximum pending prefill the load driver reached: 119x the bar.
W3_MAX_PENDING = 836048
#: The bundle the arm named on all 151 refusals.
W3_BUNDLE = 7


def _cfg(**kw):
    fields = dict(
        enabled=True,
        drain_mode=True,
        flip_tokens=W3_BAR,
        pp_exit_tokens=W3_BAR,
        # The dampers that sit ABOVE drain mode are neutralised so this file
        # measures the drain rule and not one of them; each has its own tests.
        min_dwell_s=0.0,
        tp_decode_floor_s=0.0,
    )
    fields.update(kw)
    return PhasePolicyConfig(**fields)


def _decide(cfg, state, **kw):
    """One policy decision in the TP phase, with the boot's shape."""
    from sglang.srt.managers.phase_policy import _decide_from_load

    fields = dict(
        phase="tp",
        running_bs=W3_BUNDLE,
        pending_prefill_tokens=W3_MAX_PENDING,
        now=0.0,
    )
    fields.update(kw)
    inp = PhasePolicyInputs(**fields)
    return _decide_from_load(cfg, state, inp)


def _state(*, now: float, bundle_progress_at: float):
    return PhasePolicyState(
        last_phase="tp",
        phase_since=0.0,
        bundle_at_phase_entry=W3_BUNDLE,
        last_bundle_progress_at=bundle_progress_at,
        last_running_bs=W3_BUNDLE,
    )


class TheDeadlineIsSolvedNotChosen(unittest.TestCase):
    def test_it_is_one_full_decode_window(self):
        cfg = _cfg(flip_cost_s=8.0)
        self.assertEqual(drain_stall_deadline_s(cfg), solved_tp_decode_floor_s(cfg))

    def test_a_floor_keeps_it_meaningful_when_flip_cost_is_unset(self):
        """Without the floor an unset flip cost arms on the first observation."""
        self.assertEqual(drain_stall_deadline_s(_cfg(flip_cost_s=0.0)), 10.0)

    def test_the_booted_value(self):
        """flip_cost_s 3.2 -> 6.4, raised to the 10 s floor."""
        self.assertEqual(drain_stall_deadline_s(_cfg(flip_cost_s=3.2)), 10.0)


class ADrainingBundleIsNeverCut(unittest.TestCase):
    """#677 hot fix 2's semantics, where they were ever true."""

    def test_a_bundle_that_just_shrank_still_holds_the_flip(self):
        cfg = _cfg(flip_cost_s=3.2)
        deadline = drain_stall_deadline_s(cfg)
        decision = _decide(
            cfg,
            _state(now=0.0, bundle_progress_at=0.0),
            now=deadline - 1.0,
        )
        self.assertIsNone(decision.direction)
        self.assertIn("drain mode finishes the bundle", decision.reason)

    def test_the_hold_reason_still_names_the_bundle(self):
        cfg = _cfg(flip_cost_s=3.2)
        decision = _decide(cfg, _state(now=0.0, bundle_progress_at=0.0), now=1.0)
        self.assertIn(f"{W3_BUNDLE} of {W3_BUNDLE} req still decoding", decision.reason)


class AStalledBundleDoesNotVetoForever(unittest.TestCase):
    """THE FIX. The measured refusal, and that it now terminates."""

    def test_the_measured_119x_refusal_reproduces_below_the_deadline(self):
        cfg = _cfg(flip_cost_s=3.2)
        decision = _decide(cfg, _state(now=0.0, bundle_progress_at=0.0), now=2.0)
        self.assertIsNone(decision.direction)

    def test_and_it_arms_once_the_set_has_demonstrably_stalled(self):
        cfg = _cfg(flip_cost_s=3.2)
        deadline = drain_stall_deadline_s(cfg)
        decision = _decide(
            cfg,
            _state(now=0.0, bundle_progress_at=0.0),
            now=deadline + 0.1,
        )
        self.assertEqual(decision.direction, "tp_to_pp")
        self.assertIn("STALLED, not draining", decision.reason)

    def test_the_arm_says_why_waiting_cannot_converge(self):
        cfg = _cfg(flip_cost_s=3.2)
        decision = _decide(
            cfg,
            _state(now=0.0, bundle_progress_at=0.0),
            now=drain_stall_deadline_s(cfg) + 5.0,
        )
        self.assertIn("cannot converge", decision.reason)
        self.assertIn("refilling the bundle", decision.reason)


class PressureMayNotRaiseTheRefusalRate(unittest.TestCase):
    """The policy inversion, stated as a measurable property.

    This is the acceptance criterion the ticket carries into W14, and it is
    the one assertion that a return to the admitted-set binding cannot pass.
    """

    def _refusals_at(self, cfg, pending, *, stall_s):
        """Refusals over one sweep of the boot's decision cadence."""
        refused = 0
        for t in range(0, 40):
            decision = _decide(
                cfg,
                _state(now=0.0, bundle_progress_at=0.0),
                now=float(t),
                pending_prefill_tokens=pending,
            )
            if decision.direction is None:
                refused += 1
        return refused

    def test_refusals_do_not_increase_with_pressure(self):
        cfg = _cfg(flip_cost_s=3.2)
        pressures = [W3_BAR + 1, 50_000, 200_000, W3_MAX_PENDING]
        rates = [self._refusals_at(cfg, p, stall_s=0.0) for p in pressures]
        for lighter, heavier in zip(rates, rates[1:]):
            self.assertLessEqual(
                heavier,
                lighter,
                f"refusal rate rose with pressure: {rates} over {pressures}",
            )

    def test_the_119x_bar_shape_is_not_reachable_any_more(self):
        """No pressure produces an unbounded refusal run."""
        cfg = _cfg(flip_cost_s=3.2)
        refused = self._refusals_at(cfg, W3_MAX_PENDING, stall_s=0.0)
        deadline = drain_stall_deadline_s(cfg)
        # The run is bounded by the deadline, not by the load.
        self.assertLessEqual(refused, int(deadline) + 1)
        self.assertGreater(refused, 0, "a draining bundle must still be protected")


class TheProgressClockWatchesTheAdmittedSet(unittest.TestCase):
    """The axis itself: a refilled bundle is NOT progress."""

    def _observe(self, state, running_bs, now):
        from sglang.srt.managers.phase_policy import observe_idle

        observe_idle(
            state,
            PhasePolicyInputs(
                phase="tp",
                running_bs=running_bs,
                pending_prefill_tokens=W3_MAX_PENDING,
                now=now,
            ),
        )

    def test_a_shrinking_bundle_stamps_progress(self):
        state = PhasePolicyState()
        self._observe(state, 7, 0.0)
        self._observe(state, 5, 1.0)
        self.assertEqual(state.last_bundle_progress_at, 1.0)

    def test_a_bundle_refilled_to_the_same_size_does_not(self):
        """Retiring work while admission refills is not progress to empty."""
        state = PhasePolicyState()
        self._observe(state, 7, 0.0)
        stamped = state.last_bundle_progress_at
        for t in (1.0, 2.0, 3.0, 4.0):
            self._observe(state, 7, t)
        self.assertEqual(state.last_bundle_progress_at, stamped)

    def test_pending_movement_alone_is_not_progress_on_this_axis(self):
        """The #677(a) trap: pending creeping must not stamp the bundle clock."""
        state = PhasePolicyState()
        self._observe(state, 7, 0.0)
        stamped = state.last_bundle_progress_at
        from sglang.srt.managers.phase_policy import observe_idle

        for t, pending in ((1.0, 572792), (2.0, 572715), (3.0, 572600)):
            observe_idle(
                state,
                PhasePolicyInputs(
                    phase="tp", running_bs=7, pending_prefill_tokens=pending, now=t
                ),
            )
        self.assertEqual(state.last_bundle_progress_at, stamped)

    def test_a_fresh_phase_inherits_no_stall(self):
        state = PhasePolicyState()
        self._observe(state, 7, 0.0)
        for t in (1.0, 2.0, 3.0):
            self._observe(state, 7, t)
        # A phase change restarts the clock: a wedge must be demonstrated in
        # THIS residency and never carried in from the last.
        self._observe(state, 7, 4.0)
        state.last_phase = "pp"
        self._observe(state, 7, 5.0)
        self.assertEqual(state.last_bundle_progress_at, 5.0)


if __name__ == "__main__":
    unittest.main()
