"""The stranded-decode surcharge must be able to DELAY a flip, never to make
one unreachable.

THE DEFECT (#665-F1, live capture 2026-08-15)
---------------------------------------------
`effective_flip_threshold` charges every decoding request a full PP window and
charges the counterfactual -- staying in TP -- nothing at all:

    C_eff = flip_cost_s + weight x running_bs x pp_window_s
    N_eff = flip_tokens x C_eff / flip_cost_s

With the booted values (N 7004, C 3.2 s, W 15 s, weight 1) that is a ladder of
7004 / 39835 / 72666 / 105498 / 138329 tokens for 0..4 decoding requests. A
live 72,257-token prefill backlog at running_bs 2 was refused by 409 tokens,
and at `--max-running-requests 4` the gate wants 138,329 pending tokens -- more
than a 120k prompt can supply. Whenever the server is busy, long prefills are
pinned in the slow layout: the original NIAH complaint, back again.

WHY THE COUNTERFACTUAL IS NOT FREE
----------------------------------
Measured on this rig against the live instance, 2 decode streams + one 72k
prompt (`/spinning/evidence-665-f1/measure_decode_contention.py`):

    decode throughput, undisturbed      ~54 tok/s
    decode throughput, during a TP prefill    0 tok/s
    A-vs-A noise floor                        0.0%

Decode does not degrade during a co-resident TP prefill -- it STOPS. So the
decodes are stranded in BOTH branches, and the only question is for how long:

    stay in TP   decode stalls for  N / r_tp          (72k -> ~43 s)
    flip to PP   decode stalls for  2C + min(W, N/r_pp)   (72k -> ~16 s)

The old model compares 15 s of stranding against 0 s of stranding. The honest
comparison is 16 s against 43 s -- at which point flipping is better for the
decodes too, not just for the prefill.

THE MODEL
---------
Aggregate delay-seconds, with sigma = the measured fraction of decode
throughput lost while a prefill is co-resident in TP:

    stay = N/X + B x sigma x N/X
    flip = (C + N/P) + B x (2C + min(W, N/P))

Flip when `flip < stay`. Solving for N gives a threshold that is still
monotonically increasing in B -- a busier server still demands a bigger
backlog -- but BOUNDED. At the measured sigma = 1 it collapses to

    N_eff = flip_tokens x (1 + 2B) / (1 + B)

whose supremum is exactly 2 x flip_tokens: with decode stalled either way, an
infinitely busy server needs at most twice the break-even backlog, and the
factor 2 is simply the round trip (two seams) the decodes additionally pay.
The cap is derived, not decreed.

BACKWARD COMPATIBILITY
----------------------
`decode_contention` defaults to 0.0 = "not measured here", and the old
surcharge is used unchanged. Same idiom as `flip_cost_s`: a term justified
only by a measurement is gated on that measurement, not on a flag.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

from sglang.srt.managers.phase_policy import (
    PHASE_TP,
    TP_TO_PP,
    PhasePolicyConfig,
    PhasePolicyError,
    PhasePolicyInputs,
    PhasePolicyState,
    decide,
    effective_flip_threshold,
    observe_idle,
)
from sglang.test.test_utils import CustomTestCase

# The booted values on the dev instance, so the numbers below are live ones.
N = 7004
C = 3.2
W = 15.0


def _cfg(**kw):
    base = dict(
        enabled=True,
        flip_tokens=N,
        min_dwell_s=0.0,
        pp_window_s=W,
        tp_decode_floor_s=0.0,
        prefill_runs_in_tp=True,
        flip_cost_s=C,
    )
    base.update(kw)
    return PhasePolicyConfig(**base)


class TestTheOldModelIsUntouchedWithoutTheMeasurement(CustomTestCase):
    """A deployment that has not measured its contention behaves as before."""

    def test_the_unreachable_ladder_is_reproduced_exactly(self):
        cfg = _cfg(decode_contention=0.0)
        self.assertEqual(
            [effective_flip_threshold(cfg, b) for b in range(5)],
            [7004, 39835, 72666, 105498, 138329],
        )

    def test_the_live_refusal_is_reproduced(self):
        """72,257 pending at running_bs 2, refused by 409 tokens."""
        cfg = _cfg(decode_contention=0.0)
        self.assertIsNone(_drive(cfg, 72_257, 2).direction)


class TestTheMeasuredModelMakesTheGateReachable(CustomTestCase):
    def test_sigma_one_gives_the_closed_form_ladder(self):
        """N_eff = N x (1 + 2B) / (1 + B)."""
        cfg = _cfg(decode_contention=1.0)
        got = [effective_flip_threshold(cfg, b) for b in range(5)]
        want = [int(round(N * (1 + 2 * b) / (1 + b))) for b in range(5)]
        self.assertEqual(got, want)
        self.assertEqual(got, [7004, 10506, 11673, 12257, 12607])

    def test_the_live_refusal_becomes_a_flip(self):
        cfg = _cfg(decode_contention=1.0)
        self.assertEqual(_drive(cfg, 72_257, 2).direction, TP_TO_PP)

    def test_it_is_reachable_at_the_configured_max_running_requests(self):
        """--max-running-requests 4: a 120k prompt must be able to flip."""
        cfg = _cfg(decode_contention=1.0)
        self.assertEqual(_drive(cfg, 120_000, 4).direction, TP_TO_PP)

    def test_a_partial_measurement_still_bounds_the_ladder(self):
        """Even sigma = 0.25 -- decode losing only a quarter -- keeps every
        rung inside a 120k prompt's reach."""
        cfg = _cfg(decode_contention=0.25)
        for bs in range(5):
            with self.subTest(bs=bs):
                self.assertLess(effective_flip_threshold(cfg, bs), 120_000)


class TestTheBoundIsStructural(CustomTestCase):
    def test_it_still_rises_with_the_number_stranded(self):
        """The surcharge must keep DELAYING; only the divergence is removed."""
        cfg = _cfg(decode_contention=1.0)
        rungs = [effective_flip_threshold(cfg, b) for b in range(9)]
        self.assertEqual(rungs, sorted(rungs))
        self.assertGreater(rungs[4], rungs[1])

    def test_the_supremum_is_twice_the_break_even(self):
        cfg = _cfg(decode_contention=1.0)
        self.assertLessEqual(effective_flip_threshold(cfg, 10_000), 2 * N)
        self.assertGreater(effective_flip_threshold(cfg, 10_000), 2 * N - 10)

    def test_it_never_dips_below_the_break_even(self):
        """No amount of contention may justify a flip that cannot repay the
        seam -- that floor is what keeps short prompts out."""
        for sigma in (0.1, 0.5, 0.9, 1.0):
            cfg = _cfg(decode_contention=sigma)
            for bs in range(6):
                with self.subTest(sigma=sigma, bs=bs):
                    self.assertGreaterEqual(effective_flip_threshold(cfg, bs), N)

    def test_a_backlog_beyond_the_pp_window_stays_finite(self):
        """Past W x r_pp the prefill no longer fits one window, so the decode
        charge saturates at the window; the threshold must remain solvable."""
        cfg = _cfg(decode_contention=0.05)
        for bs in range(5):
            with self.subTest(bs=bs):
                t = effective_flip_threshold(cfg, bs)
                self.assertGreater(t, 0)
                self.assertLess(t, 10_000_000)

    def test_purity_strict_still_collapses_to_zero(self):
        cfg = _cfg(decode_contention=1.0, prefill_runs_in_tp=False)
        self.assertEqual(effective_flip_threshold(cfg, 4), 0)

    def test_at_sigma_one_the_prefill_ladder_cancels_out(self):
        """The calibration must not go stale when the instance is re-shipped.

        This rig re-solves its memory and KV vectors per boot, and the prefill
        ladder moves with them. At the measured sigma = 1 the (1 - r) factor
        divides out, so the threshold depends only on the break-even and the
        number of decodes -- nothing here needs re-measuring after a re-ship.
        """
        ladders = [(1681.0, 7245.5), (1194.0, 7245.5), (1322.0, 6842.6)]
        for bs in range(5):
            got = {
                effective_flip_threshold(
                    _cfg(
                        decode_contention=1.0,
                        tp_prefill_tok_s=tp,
                        pp_prefill_tok_s=pp,
                    ),
                    bs,
                )
                for tp, pp in ladders
            }
            with self.subTest(bs=bs):
                self.assertEqual(len(got), 1, f"ladder-dependent at bs={bs}")


class TestTheMeasurementIsValidated(CustomTestCase):
    def test_a_fraction_outside_zero_to_one_is_refused(self):
        for bad in (-0.1, 1.5):
            with self.subTest(bad=bad):
                with self.assertRaises(PhasePolicyError):
                    _cfg(decode_contention=bad)


class TestAntiThrashStillHolds(CustomTestCase):
    """The e0aa/5bd2 acceptance: small-request traffic causes ZERO flips.

    This is the property the bound must not buy its reachability with.
    """

    def test_eight_small_requests_cause_no_flip(self):
        cfg = _cfg(decode_contention=1.0)
        state = PhasePolicyState()
        flips = 0
        now = 1000.0
        pending = 0
        for i in range(8):
            # Each small request adds ~600 prompt tokens and then decodes.
            pending += 600
            now += 0.4
            inp = PhasePolicyInputs(
                phase=PHASE_TP,
                pending_prefill_tokens=pending,
                running_bs=min(i + 1, 4),
                now=now,
            )
            observe_idle(state, inp)
            if decide(cfg, state, inp).direction is not None:
                flips += 1
            pending = max(0, pending - 600)  # it prefills in TP and drains
        self.assertEqual(flips, 0)

    def test_a_prompt_just_under_the_break_even_never_flips(self):
        cfg = _cfg(decode_contention=1.0)
        for bs in range(5):
            with self.subTest(bs=bs):
                self.assertIsNone(_drive(cfg, N - 1, bs).direction)


class TestTheLiveCaptureReplay(CustomTestCase):
    """Replay of a capture taken from the running instance, 2026-08-15.

    One 72k prompt injected against 2 live decode streams. Chunked prefill
    (--chunked-prefill-size 512) walks the backlog up in steps, and the policy
    was sampled at each; these are the six distinct
    `PHASE-POLICY holding in tp: pending prefill ...` records it emitted, from
    /spinning/evidence-qwen38/boot_qwen38.log. Over the whole run the log
    shows 103 holds and ZERO tp_to_pp arms, and the prefill took 60.6 s in the
    TP layout with decode emitting nothing.
    """

    CAPTURE = [9205, 19957, 31221, 42997, 55797, 69109]
    RUNNING_BS = 2

    def test_the_old_model_refuses_every_single_sample(self):
        cfg = _cfg(decode_contention=0.0)
        armed = [
            n
            for n in self.CAPTURE
            if _drive(cfg, n, self.RUNNING_BS).direction is not None
        ]
        self.assertEqual(armed, [])

    def test_the_measured_model_arms_once_the_backlog_is_worth_it(self):
        """It arms at 19,957 -- and still refuses 9,205, which genuinely is
        too small to repay a round trip. Reachable, not trigger-happy."""
        cfg = _cfg(decode_contention=1.0)
        armed = [
            n
            for n in self.CAPTURE
            if _drive(cfg, n, self.RUNNING_BS).direction is not None
        ]
        self.assertEqual(armed, [19957, 31221, 42997, 55797, 69109])
        self.assertEqual(effective_flip_threshold(cfg, self.RUNNING_BS), 11673)


def _drive(cfg, pending, bs, phase=PHASE_TP):
    state = PhasePolicyState()
    inp = PhasePolicyInputs(
        phase=phase, pending_prefill_tokens=pending, running_bs=bs, now=1000.0
    )
    observe_idle(state, inp)
    return decide(cfg, state, inp)


if __name__ == "__main__":
    unittest.main()
