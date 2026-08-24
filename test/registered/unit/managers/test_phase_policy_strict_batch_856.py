"""#856 STRICT PHASE BATCHING: the economic band must be unreachable.

USER DIRECTIVE (2026-08-24, verbatim intent): all pending prefill is collected
and processed in the PP layout, then all decode in the TP layout, then prefill
again -- and NEVER any work in the wrong layout, "egal wie lang der flip
dauert". Flip duration is recorded, not traded against.

THE CONSEQUENCE FOR THE POLICY, and it is the whole of this slice: in this mode
the trigger is DRAIN-AND-FLIP, not economics. The break-even band exists to
answer "is this backlog worth the seam?", which is exactly the question strict
batching does not ask -- the answer is always yes, because the alternative is
running prefill in the decode layout, which the mode forbids outright.

WHERE THE BAND SITS, and why gating it is the minimal correct change: in
``_decide_from_load`` the band (phase_policy.py ~:2442) can `return _no(...)`
and hold TP BEFORE the drain exit (~:2527) is ever consulted. So the drain rule
being right is not sufficient; the band has to stop applying. That is one
condition, not a new engine -- `--phase-flip-policy` selects manual-vs-auto and
is deliberately NOT touched.

THE PP SIDE IS ALREADY STRICT (the DRAINED rule at ~:2741 flips at
`pending <= pp_exit_tokens`, default 0), so nothing changes there and this file
pins that it stays that way.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

import unittest

from sglang.srt.managers import phase_policy as pp
from sglang.test.test_utils import CustomTestCase

NOW = 1000.0


def _cfg(strict: bool):
    """A config whose economic band is genuinely OPEN.

    N_eff = flip_tokens x (flip_cost_s + weight x bs x pp_window_s)/flip_cost_s,
    so weight 1.0, pp_window 1.0 and bs 2 put the threshold at 3000 against an
    N of 1000. A test where the band cannot be entered would pass no matter
    what the code did, which is how this kind of assertion goes vacuous.
    """
    return pp.PhasePolicyConfig(
        enabled=True,
        drain_mode=True,
        drain_mode_strict=strict,
        prefill_runs_in_tp=True,
        flip_tokens=1000,
        flip_cost_s=1.0,
        decode_strand_weight=1.0,
        pp_window_s=1.0,
        min_dwell_s=0.0,
        tp_decode_floor_s=0.0,
        idle_dwell_s=0.0,
        pp_exit_tokens=0,
    )


def _state(running_bs: int, entered_with=None, phase_since=0.0):
    # `entered_with` is the bundle size at phase entry. It matters: with 0 the
    # "decode phase ran EMPTY" rule arms in EVERY mode, which would hide the
    # strict-vs-economic difference behind a rule neither of them owns.
    return pp.PhasePolicyState(
        last_flip_at=0.0,
        phase_since=phase_since,
        last_phase=pp.PHASE_TP,
        bundle_at_phase_entry=running_bs if entered_with is None else entered_with,
        last_bundle_progress_at=NOW,
        last_prefill_progress_at=NOW,
    )


def _inp(pending: int, running_bs: int, phase=pp.PHASE_TP):
    return pp.PhasePolicyInputs(
        phase=phase,
        pending_prefill_tokens=pending,
        running_bs=running_bs,
        now=NOW,
        nothing_can_run=False,
        target_can_admit=True,
        ready_carriers=0,
        queue_nonempty=True,
        kv_available_tokens=100000,
    )


class TestTheBandIsOpenAtAll(CustomTestCase):
    """Guards every other test in this file against going vacuous."""

    def test_the_threshold_really_is_above_N(self):
        cfg = _cfg(strict=False)
        self.assertEqual(pp.live_flip_tokens(cfg), 1000)
        self.assertEqual(pp.effective_flip_threshold(cfg, 2), 3000)


class TestStrictModeSkipsTheEconomicBand(CustomTestCase):
    IN_BAND = 2000  # 1000 < 2000 <= 3000

    def test_the_economic_mode_HOLDS_tp_on_the_band(self):
        # THE CAN-FAIL PARTNER, and it is the behaviour that must SURVIVE for
        # every other workload: without strict, a backlog inside the band is a
        # reason to stay in TP and prefill there.
        d = pp.decide(_cfg(strict=False), _state(2), _inp(self.IN_BAND, 2))
        self.assertIsNone(d.direction)
        self.assertIn("strand", (d.reason or "").lower())

    def test_strict_mode_does_NOT_hold_on_that_band(self):
        # THE POINT OF THE SLICE. Same load, same numbers, strict on: the band
        # must not be the reason for anything. Whatever holds TP now must be
        # the bundle, never the economics.
        d = pp.decide(_cfg(strict=True), _state(2), _inp(self.IN_BAND, 2))
        self.assertNotIn("strand", (d.reason or "").lower())

    def test_strict_mode_flips_once_the_bundle_is_drained(self):
        # Drain-and-flip: bundle empty and backlog above the exit chunk ->
        # go to the prefill layout, with no economic term consulted at all.
        d = pp.decide(_cfg(strict=True), _state(0), _inp(self.IN_BAND, 0))
        self.assertEqual(d.direction, pp.TP_TO_PP)

    def test_strict_mode_flips_even_for_a_backlog_far_BELOW_N(self):
        # The sharpest statement of the mode: 1 token of prefill is enough,
        # because prefill may not run in TP at all. Under the economic rule
        # this backlog is worthless (1 << N=1000) and would never flip.
        # A bundle WAS resident and has now drained, so the "decode phase ran
        # EMPTY" rescue rule -- which arms in every mode -- cannot be what
        # answers here. What is left is the mode's own trigger.
        drained = _state(0, entered_with=2)
        d = pp.decide(_cfg(strict=True), drained, _inp(1, 0))
        self.assertEqual(d.direction, pp.TP_TO_PP)
        # The comparison is against PURE economics -- drain OFF. With drain on
        # the existing rule already flips a drained bundle on any backlog, so
        # comparing against it would show no difference and prove nothing;
        # noting that explicitly because it is a trap this test fell into once.
        pure = pp.PhasePolicyConfig(
            enabled=True,
            drain_mode=False,
            prefill_runs_in_tp=True,
            flip_tokens=1000,
            flip_cost_s=1.0,
            decode_strand_weight=1.0,
            pp_window_s=1.0,
            min_dwell_s=0.0,
            tp_decode_floor_s=0.0,
            idle_dwell_s=0.0,
            pp_exit_tokens=0,
        )
        econ = pp.decide(pure, drained, _inp(1, 0))
        self.assertIsNone(econ.direction, "pure economics must NOT flip on 1 tok")

    def test_an_empty_backlog_still_does_not_flip(self):
        # Strict is not "flip always". With nothing waiting there is nothing to
        # go to the other layout FOR, and a mode that flipped here would
        # oscillate forever.
        d = pp.decide(_cfg(strict=True), _state(0), _inp(0, 0))
        self.assertIsNone(d.direction)


class TestStrictImpliesDrain(CustomTestCase):
    def test_strict_without_drain_is_refused_rather_than_half_applied(self):
        # Strict is an extension OF drain, not an alternative to it: the exit
        # it relies on lives in the drain block. Silently running with drain
        # off would apply the strict gate and then fall through to the very
        # economics it exists to remove.
        with self.assertRaises(pp.PhasePolicyError):
            pp.PhasePolicyConfig(enabled=True, drain_mode=False, drain_mode_strict=True)


class TestThePpSideIsUnchanged(CustomTestCase):
    """The PP half was already strict; this pins that the slice did not move it."""

    def test_pp_drains_then_flips_to_tp(self):
        d = pp.decide(
            _cfg(strict=True),
            _state(1, phase_since=NOW - 0.1),
            _inp(0, 1, phase=pp.PHASE_PP),
        )
        self.assertEqual(d.direction, pp.PP_TO_TP)

    def test_pp_holds_while_prefill_remains(self):
        # phase_since is RECENT here on purpose: the `pp_window_s` stopwatch is
        # a separate rule, and letting it expire would prove nothing about the
        # DRAINED rule this test is aimed at.
        d = pp.decide(
            _cfg(strict=True),
            _state(1, phase_since=NOW - 0.1),
            _inp(50000, 1, phase=pp.PHASE_PP),
        )
        self.assertIsNone(d.direction)


if __name__ == "__main__":
    unittest.main()
