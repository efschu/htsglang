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
    PP_TO_TP,
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
        """N_eff = N0 x (1 + 2B) / (1 + B), with N0 the UNROUNDED break-even.

        The threshold is solved from C, X and P rather than from the rounded
        integer `flip_tokens`, so it does not inherit that rounding: N0 here
        is 7004.23, and scaling the rounded 7004 instead would drift a token
        or two per rung.
        """
        cfg = _cfg(decode_contention=1.0)
        n0 = C / (1.0 / 1681.0 - 1.0 / 7245.5)
        got = [effective_flip_threshold(cfg, b) for b in range(5)]
        want = [int(round(n0 * (1 + 2 * b) / (1 + b))) for b in range(5)]
        self.assertEqual(got, want)
        self.assertEqual(got, [7004, 10506, 11674, 12257, 12608])

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
        pp_ceiling = W * 7245.5  # past this the prefill exceeds one window
        prev = 0
        for bs in range(5):
            with self.subTest(bs=bs):
                t = effective_flip_threshold(cfg, bs)
                self.assertGreaterEqual(t, N)
                self.assertGreaterEqual(t, prev)  # still monotonic out here
                # A real bound, not just "not infinity": even at the weakest
                # contention this model accepts, four stranded decodes stay
                # inside a few window-fulls of prefill.
                self.assertLess(t, 4 * pp_ceiling)
                prev = t

    def test_a_negative_window_is_refused_rather_than_read_as_disabled(self):
        with self.assertRaises(PhasePolicyError):
            _cfg(decode_contention=1.0, pp_window_s=-1.0)

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
        from sglang.srt.managers.phase_policy import break_even_tokens

        ladders = [(1681.0, 7245.5), (1194.0, 7245.5), (1322.0, 6842.6)]
        for bs in range(5):
            shape = set()
            for tp, pp in ladders:
                # A real re-ship re-derives N from the new ladder; holding
                # flip_tokens fixed across ladders would test nothing but the
                # algebra of a substitution.
                n0 = break_even_tokens(C, tp, pp)
                cfg = _cfg(
                    decode_contention=1.0,
                    flip_tokens=n0,
                    tp_prefill_tok_s=tp,
                    pp_prefill_tok_s=pp,
                )
                got = effective_flip_threshold(cfg, bs)
                shape.add(round(got / (C / (1.0 / tp - 1.0 / pp)), 3))
            with self.subTest(bs=bs):
                # The threshold is always the same MULTIPLE of that ladder's
                # own break-even, so nothing but N needs re-deriving.
                self.assertEqual(len(shape), 1, f"ladder-dependent at bs={bs}")
                self.assertAlmostEqual(shape.pop(), (1 + 2 * bs) / (1 + bs), places=2)


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


class TestTheShippedDefaultWindow(CustomTestCase):
    """`pp_window_s` DEFAULTS TO 0, and every other test here overrides it.

    Found in review, not by the suite. With the window disabled the old code
    bypassed the regime-A validity check entirely, so as `den_a` approached
    its singularity from above the threshold spiked, and the moment `den_a`
    went negative it fell into regime B -- which reads `W = 0` as "stranding
    is free", the exact opposite of what an absent window means. The result
    was a spike to 548,049 tokens at running_bs 5 (worse than the 138,329 this
    branch exists to remove) followed by a ~497,000-token CLIFF down to 51,117
    at running_bs 6: a busier server becoming drastically more eager to flip.
    """

    LADDER_SIGMA_01 = [7004, 25373, 53365, 101225, 201739, 548049, 51117, 54963]

    def _cfg0(self, sigma):
        return PhasePolicyConfig(
            enabled=True,
            flip_tokens=N,
            flip_cost_s=C,
            pp_window_s=0.0,  # the SHIPPED default
            decode_contention=sigma,
            prefill_runs_in_tp=True,
        )

    def test_the_ladder_is_monotonic_with_the_window_disabled(self):
        for sigma in (0.05, 0.1, 0.25, 0.5, 1.0):
            cfg = self._cfg0(sigma)
            rungs = [effective_flip_threshold(cfg, b) for b in range(12)]
            with self.subTest(sigma=sigma):
                self.assertEqual(rungs, sorted(rungs), f"non-monotonic: {rungs}")

    def test_the_specific_cliff_is_gone(self):
        cfg = self._cfg0(0.1)
        rungs = [effective_flip_threshold(cfg, b) for b in range(8)]
        self.assertNotEqual(rungs, self.LADDER_SIGMA_01)
        # bs 5 -> 6 was the cliff: 548049 -> 51117.
        self.assertGreaterEqual(rungs[6], rungs[5])

    def test_never_repaying_is_said_plainly_rather_than_approximated(self):
        """Below the contention where a flip can ever repay an unbounded PP
        residency, the answer is 'never' -- not a large finite number that
        happens to look like a threshold, and not a small one."""
        from sglang.srt.managers.phase_policy import UNREACHABLE_FLIP_THRESHOLD

        cfg = self._cfg0(0.1)
        self.assertEqual(effective_flip_threshold(cfg, 6), UNREACHABLE_FLIP_THRESHOLD)

    def test_a_measured_full_stall_still_flips_with_no_window(self):
        """sigma = 1 is the measured case, and it must stay reachable even
        with the fairness window off."""
        cfg = self._cfg0(1.0)
        for bs in range(6):
            with self.subTest(bs=bs):
                self.assertLessEqual(effective_flip_threshold(cfg, bs), 2 * N)


class TestItDoesNotDependOnAStaleBreakEven(CustomTestCase):
    """The surcharge is solved from C, X and P, not by cancelling N0.

    `config_from_env` derives the default N from the module constant
    DEFAULT_FLIP_COST_S while `flip_cost_s` is separately overridable, and an
    operator may pin `flip_tokens` outright -- so `flip_tokens` need not be
    the break-even of the seam cost actually configured. Substituting it into
    the formula would then under-threshold silently, arming flips that do not
    repay the real seam.
    """

    def test_a_re_measured_seam_moves_the_threshold(self):
        cheap = _cfg(decode_contention=1.0, flip_cost_s=3.2)
        dear = _cfg(decode_contention=1.0, flip_cost_s=5.0)
        for bs in range(1, 5):
            with self.subTest(bs=bs):
                self.assertGreater(
                    effective_flip_threshold(dear, bs),
                    effective_flip_threshold(cheap, bs),
                )

    def test_a_stale_flip_tokens_cannot_lower_the_bar(self):
        """flip_tokens left at the 3.2 s break-even while the seam is really
        5.0 s must not produce the 3.2 s ladder."""
        stale = _cfg(decode_contention=1.0, flip_cost_s=5.0, flip_tokens=7004)
        got = [effective_flip_threshold(stale, b) for b in range(5)]
        self.assertNotEqual(got, [7004, 10506, 11673, 12257, 12607])
        # 5.0/3.2 = 1.5625x the seam, so every rung above B=0 scales with it.
        self.assertAlmostEqual(got[2] / 11673.0, 5.0 / 3.2, places=2)


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
        self.assertEqual(effective_flip_threshold(cfg, self.RUNNING_BS), 11674)


class TestItDegradesGracefullyOnAnUnfundableSeam(CustomTestCase):
    """Reachability must be safe on a vector that cannot fund the cutover.

    The dev instance is being re-shipped on a corridor-tight vector that
    reclaims the seam-staging headroom into the KV pool, on the reasoning that
    the flip gate barely fires anyway. That reasoning is downstream of the
    defect this branch fixes, so once the gate IS reachable the arms arrive on
    a vector whose corridor guard will refuse them.

    That must not become an arm/refuse loop. It does not: `note_flip_outcome`
    already backs off per direction, doubling from min_dwell_s to
    refusal_backoff_cap_s, and after refusal_degrade_after consecutive
    refusals declares the direction unfundable and only re-probes. This test
    pins that the combination is bounded, so the fix can land ahead of any
    decision about the vector -- it simply does not pay off until the seam is
    fundable, rather than doing harm.
    """

    def test_a_permanently_refused_seam_arms_a_bounded_number_of_times(self):
        from sglang.srt.managers.phase_policy import (
            note_flip_armed,
            note_flip_outcome,
        )

        cfg = _cfg(
            decode_contention=1.0,
            min_dwell_s=3.0,
            refusal_backoff_cap_s=60.0,
            refusal_degrade_after=8,
        )
        state = PhasePolicyState()
        now = 1000.0
        end = now + 600.0  # ten minutes of a sustained qualifying backlog
        arms = 0
        while now < end:
            inp = PhasePolicyInputs(
                phase=PHASE_TP,
                pending_prefill_tokens=60_000,
                running_bs=2,
                now=now,
            )
            observe_idle(state, inp)
            d = decide(cfg, state, inp)
            if d.direction is not None:
                arms += 1
                note_flip_armed(state, d, now)
                note_flip_outcome(
                    cfg,
                    state,
                    d.direction,
                    False,
                    "corridor guard refused the seam staging",
                    now,
                )
            now += 1.0

        # CONTRACT CHANGED 2026-08-15 by #662-F4's dwell pacing, and this
        # assertion was rewritten to match rather than to defend the old shape.
        #
        # It used to be "8 arms to reach degradation, then one re-probe per
        # 60 s cap" -- at most ~16 in ten minutes. F4 replaced the doubling
        # backoff with `paced_until = min(hold_until, last_refusal +
        # min_dwell_s)` on the reasoning that the load still wants this layout,
        # so the next attempt should re-ask the KV rung rather than wait out a
        # backoff. A persistently refused seam therefore re-probes every
        # min_dwell_s: ~200 arms in ten minutes, not 16.
        #
        # That is only safe because an ARM IS NO LONGER A STAGING ATTEMPT. The
        # group-abandon gate defers actual entry by arm COUNT ("next entry at
        # arm 24"), so the paced arms are cheap probes and only a few of them
        # reach the seam. Any acceptance that counts arms as if each one staged
        # -- mine did -- is measuring the wrong thing; count abandons.
        # CONTRACT, third revision, and this one is the durable shape.
        #
        # v1 (mine): doubling backoff, <=16 arms in ten minutes, then a latch.
        # v2 (F4's dwell pacing): ~200 arms -- the boot-E storm rate exactly,
        #     which is what the #656 guard exists to forbid, and it went red.
        # v3 (here): the pacing stays for CHEAP probes, but a STAGING attempt
        #     that abandons imposes a doubling, capped interval before the next
        #     one. Expensive failures are rate-limited; nothing latches.
        #
        # A persistently unfundable seam therefore settles at one staging
        # attempt per refusal_backoff_cap_s -- about a dozen in ten minutes,
        # forever, never zero. That is the distinction that matters: it keeps
        # asking, so the moment the rung can fund it, it funds.
        self.assertGreater(arms, 5, "a latch would stop re-probing entirely")
        self.assertLess(arms, 20, f"{arms} staging attempts is storm territory")
        self.assertTrue(state.arm_degraded.get(TP_TO_PP))

    def test_a_completion_clears_the_staging_rate_limit_outright(self):
        """The property that makes it a limiter and not a latch."""
        from sglang.srt.managers.phase_policy import (
            note_flip_armed,
            note_flip_completed,
            note_flip_outcome,
        )

        cfg = _cfg(decode_contention=1.0, min_dwell_s=0.0, flip_cost_s=5.918)
        state = PhasePolicyState()
        now = 1000.0
        inp = PhasePolicyInputs(
            phase=PHASE_TP, pending_prefill_tokens=60_000, running_bs=2, now=now
        )
        observe_idle(state, inp)
        d = decide(cfg, state, inp)
        note_flip_armed(state, d, now)
        note_flip_outcome(cfg, state, d.direction, False, "corridor", now)
        # Rate-limited one second later.
        nxt = PhasePolicyInputs(
            phase=PHASE_TP,
            pending_prefill_tokens=60_000,
            running_bs=2,
            now=now + 1.0,
        )
        self.assertIsNone(decide(cfg, state, nxt).direction)
        self.assertIn("rate limit", decide(cfg, state, nxt).reason)
        # A completion wipes the penalty; the next arm is free immediately.
        note_flip_completed(cfg, state, d.direction, now + 1.0)
        self.assertNotIn(TP_TO_PP, state.last_abandon_at)

    def test_the_refusal_hold_is_reported_rather_than_silent(self):
        from sglang.srt.managers.phase_policy import (
            note_flip_armed,
            note_flip_outcome,
        )

        cfg = _cfg(decode_contention=1.0, min_dwell_s=3.0)
        state = PhasePolicyState()
        now = 1000.0
        inp = PhasePolicyInputs(
            phase=PHASE_TP, pending_prefill_tokens=60_000, running_bs=2, now=now
        )
        observe_idle(state, inp)
        d = decide(cfg, state, inp)
        self.assertEqual(d.direction, TP_TO_PP)
        note_flip_armed(state, d, now)
        note_flip_outcome(cfg, state, d.direction, False, "corridor guard", now)

        nxt = decide(
            cfg,
            state,
            PhasePolicyInputs(
                phase=PHASE_TP,
                pending_prefill_tokens=60_000,
                running_bs=2,
                now=now + 1.0,
            ),
        )
        self.assertIsNone(nxt.direction)
        self.assertIn("refused", nxt.reason)


def _drive(cfg, pending, bs, phase=PHASE_TP):
    state = PhasePolicyState()
    inp = PhasePolicyInputs(
        phase=phase, pending_prefill_tokens=pending, running_bs=bs, now=1000.0
    )
    observe_idle(state, inp)
    return decide(cfg, state, inp)


if __name__ == "__main__":
    unittest.main()


class TestTheRuntimeToggle(CustomTestCase):
    """`with_decode_contention` -- the within-boot A/B lever.

    The measured and the one-sided threshold have to be comparable on the SAME
    memory vector, KV token vector and corridor, or the comparison carries
    boot-to-boot variance instead of the effect. So the fraction is settable at
    runtime, and this is the hermetic cover for the validation the scheduler
    delegates here.
    """

    def test_it_returns_a_new_config_and_leaves_the_original_alone(self):
        from sglang.srt.managers.phase_policy import with_decode_contention

        cfg = _cfg(decode_contention=0.0)
        got = with_decode_contention(cfg, 1.0)
        self.assertEqual(got.decode_contention, 1.0)
        self.assertEqual(cfg.decode_contention, 0.0)

    def test_it_actually_moves_the_ladder(self):
        from sglang.srt.managers.phase_policy import with_decode_contention

        one_sided = _cfg(decode_contention=0.0)
        measured = with_decode_contention(one_sided, 1.0)
        self.assertEqual(effective_flip_threshold(one_sided, 2), 72666)
        self.assertEqual(effective_flip_threshold(measured, 2), 11674)

    def test_a_string_fraction_is_accepted(self):
        """It arrives over JSON, so "1.0" must work as well as 1.0."""
        from sglang.srt.managers.phase_policy import with_decode_contention

        self.assertEqual(with_decode_contention(_cfg(), "0.5").decode_contention, 0.5)

    def test_nonsense_is_refused_rather_than_half_applied(self):
        from sglang.srt.managers.phase_policy import with_decode_contention

        for bad in ("banana", None, [1], 1.5, -0.1):
            with self.subTest(bad=bad):
                with self.assertRaises(PhasePolicyError):
                    with_decode_contention(_cfg(), bad)

    def test_the_round_trip_restores_the_old_behaviour_exactly(self):
        """The A arm of the A/B must be the shipped behaviour, not an
        approximation of it."""
        from sglang.srt.managers.phase_policy import with_decode_contention

        cfg = _cfg(decode_contention=0.0)
        there = with_decode_contention(cfg, 1.0)
        back = with_decode_contention(there, 0.0)
        self.assertEqual(
            [effective_flip_threshold(back, b) for b in range(5)],
            [effective_flip_threshold(cfg, b) for b in range(5)],
        )


class TestTheLadderIsSolvedNotConfigured(CustomTestCase):
    """#584 provenance: every rung is DERIVED from measurements, not tuned.

    All four inputs below come from the combined boot itself
    (`/spinning/evidence-662-F4/boot_combined.log`, 2026-08-15), not from the
    module constants, which were measured on a different checkpoint.

    C_ROUNDTRIP -- 21 `PHASE-FLIP DONE` records, slowest rank per direction,
    three complete tp_to_pp -> pp_to_tp round trips: 5901.0 / 5912.9 /
    5939.8 ms. Mean 5.918 s, spread 0.7 %.

    R_PP -- the PP phase may not decode under `prefill_in_tp` purity, so PP
    wall-clock minus its opening seam IS prefill time. 85,233 prefill tokens
    over three PP phases totalling 30 s, less 8.88 s of seams = 21.12 s, so
    4036 tok/s.

    R_TP -- taken from a prefill actually running in TP, which is the only
    honest source: a TP phase's wall clock is mostly decode. GATE B put
    72,000 tokens through TP with TTFT 71.9 s -> 1001 tok/s, and decode
    contributes nothing during it (sigma = 1). Pre-fix baselines on the same
    model gave 1194 and 1196 tok/s, so the band is 1001-1196.

    P* = C / (1/r_tp - 1/r_pp) is then 7878 (r_tp 1001) to 10059 (r_tp 1196),
    centre 8949.
    """

    C_ROUNDTRIP = 5.918
    R_PP = 4036.0
    R_TP_BAND = (1001.0, 1196.0)

    @classmethod
    def _p_star(cls, r_tp):
        return cls.C_ROUNDTRIP / (1.0 / r_tp - 1.0 / cls.R_PP)

    def test_p_star_on_this_checkpoint(self):
        lo = self._p_star(self.R_TP_BAND[1])
        hi = self._p_star(self.R_TP_BAND[0])
        self.assertAlmostEqual(hi, 7878, delta=40)
        self.assertAlmostEqual(lo, 10059, delta=40)

    def test_every_rung_is_within_the_derived_margin_of_p_star(self):
        """The margin is not chosen: it IS (1+2B)/(1+B), bounded by 2.

        Rungs above P* are correct by construction -- P* is the B=0
        break-even and higher rungs price stranded decodes. What must hold is
        that the multiple is the derived one and never exceeds 2.
        """
        cfg = _cfg(decode_contention=1.0)
        p_star = self._p_star(1001.0)  # most conservative r_tp -> smallest P*
        for bs in range(5):
            rung = effective_flip_threshold(cfg, bs)
            derived = (1 + 2 * bs) / (1 + bs)
            with self.subTest(bs=bs):
                self.assertLessEqual(rung / p_star, 2.0)
                self.assertLessEqual(rung / p_star, derived + 0.05)

    def test_the_old_ladder_was_many_multiples_past_break_even(self):
        """On record: what the shipped surcharge actually demanded."""
        p_star = self._p_star(1100.0)  # centre of the band
        old = [7004, 39835, 72666, 105498, 138329]
        mult = [round(r / p_star, 1) for r in old]
        self.assertEqual(mult, [0.8, 4.5, 8.1, 11.8, 15.5])

    def test_the_booted_flip_cost_is_stale_against_the_measured_seam(self):
        """The one real mis-calibration this run exposed.

        `flip_cost_s` is booted at 3.2 s; the seam measures 5.918 s here, so
        the B=0 rung sits BELOW break-even -- the gate fires slightly early
        with nothing decoding. Re-deriving N from the measured seam fixes the
        whole ladder at once, and every rung stays under 2 x P*.
        """
        self.assertLess(N, self._p_star(1100.0))
        # SHIPPED LADDER STANDS ON THE FAST END OF THE BAND, r_tp = 1196.
        #
        # Not the midpoint. 1001 comes from a GATE B whose wall clock also
        # contains 24 arms and 8 seam abandons, so it understates the true TP
        # prefill rate; 1194/1196 come from runs where the gate never armed at
        # all. Averaging a contaminated measurement with a clean one describes
        # neither, so the midpoint 1100 was withdrawn. The fast end is also the
        # conservative end -- a higher r_tp makes TP look better, so P* rises
        # and the bar is higher -- which is the right error while a wrongly
        # armed flip abandons and latches the direction.
        solved = [
            int(round(self._p_star(1196.0) * (1 + 2 * b) / (1 + b))) for b in range(5)
        ]
        self.assertEqual(solved, [10059, 15088, 16764, 17603, 18106])
        # The 2x bound is structural against the P* the ladder was SOLVED
        # from. Comparing a centre-solved ladder against the conservative end
        # of the r_tp band is a different quantity and is legitimately larger:
        # 1.8 x (8949 / 7878) = 2.045. Stated, not asserted away -- it is the
        # honest width of the r_tp measurement, and it is the reason r_tp
        # wants a dedicated ladder run rather than a single TTFT.
        shipped = self._p_star(1196.0)
        for rung in solved:
            self.assertLessEqual(rung / shipped, 2.0)
        # Against the far end of the band the multiple is larger; recorded
        # rather than asserted away, and the reason r_tp wants a dedicated
        # ladder run on this checkpoint rather than a TTFT.
        worst = max(solved) / self._p_star(1001.0)
        self.assertGreater(worst, 2.0)


class TestTheBootedLadderEqualsTheSolvedLadder(CustomTestCase):
    """A solved number the boot silently replaces is the #584 defect itself.

    `config_from_env` derived the break-even from the module CONSTANTS while
    the surcharge read the environment, so one ladder was solved against two
    different sets of numbers. Booting the measured 5.918 s seam produced
    [7004, 19430, 21589, 22669, 23316] -- rung 0 unmoved, because the seam
    knob never reached it. There was also no PP-rate knob at all, so the PP
    prefill rate was the one input that could not be re-measured per
    checkpoint.
    """

    ENV = {
        "SGLANG_PHASE_POLICY_FLIP_COST_S": "5.918",
        "SGLANG_PHASE_POLICY_TP_TOK_S": "1100",
        "SGLANG_PHASE_POLICY_PP_TOK_S": "4036",
        "SGLANG_PHASE_POLICY_DECODE_CONTENTION": "1.0",
        "SGLANG_PHASE_POLICY_PP_WINDOW_S": "15",
    }

    def _cfg_from_env(self):
        import dataclasses
        import os
        from unittest import mock

        from sglang.srt.managers.phase_policy import config_from_env

        with mock.patch.dict(os.environ, self.ENV, clear=False):
            cfg = config_from_env(True)
        return dataclasses.replace(cfg, prefill_runs_in_tp=True)

    def test_the_measured_seam_reaches_rung_zero(self):
        cfg = self._cfg_from_env()
        self.assertEqual(cfg.flip_tokens, 8949)
        self.assertNotEqual(cfg.flip_tokens, 7004)

    def test_all_three_measurements_reach_the_config(self):
        cfg = self._cfg_from_env()
        self.assertAlmostEqual(cfg.flip_cost_s, 5.918)
        self.assertAlmostEqual(cfg.tp_prefill_tok_s, 1100.0)
        self.assertAlmostEqual(cfg.pp_prefill_tok_s, 4036.0)

    def test_the_booted_ladder_is_the_solved_ladder(self):
        cfg = self._cfg_from_env()
        booted = [effective_flip_threshold(cfg, b) for b in range(5)]
        self.assertEqual(booted, [8949, 13423, 14915, 15660, 16108])
        # And not the half-solved ladder the mismatch produced.
        self.assertNotEqual(booted, [7004, 19430, 21589, 22669, 23316])

    def test_anti_thrash_survives_the_higher_rungs(self):
        """Re-verified, as required: rungs only rose, so 8 small requests
        still arm nothing."""
        cfg = self._cfg_from_env()
        state = PhasePolicyState()
        now, pending, flips = 1000.0, 0, 0
        for i in range(8):
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
            pending = max(0, pending - 600)
        self.assertEqual(flips, 0)


class TestThePpPhaseIsGovernedByDrainNotAStopwatch(CustomTestCase):
    """The 15 s window ejected PP with 23,313 tok still pending (live 12:25).

    Two seams (~12 s at the measured 5.918 s round trip) burned to defer work
    that one longer residency would have drained, then straight back into PP as
    pending regrew. A hand-set duty cycle deciding that is the same provenance
    defect the ladder fix removed from arming.
    """

    def _pp(self, cfg, pending, bs, in_pp):
        state = PhasePolicyState()
        state.phase_since = 1000.0
        inp = PhasePolicyInputs(
            phase="pp",
            pending_prefill_tokens=pending,
            running_bs=bs,
            now=1000.0 + in_pp,
        )
        return decide(cfg, state, inp)

    def _cfg_slo(self, slo):
        return _cfg(
            decode_contention=1.0,
            flip_cost_s=5.918,
            pp_window_s=0.0,
            decode_stall_slo_s=slo,
            pp_prefill_tok_s=4036.0,
            tp_prefill_tok_s=1196.0,
            min_dwell_s=0.0,
        )

    def test_the_solved_cap_is_the_slo_minus_both_seams(self):
        from sglang.srt.managers.phase_policy import pp_residency_cap_s

        self.assertAlmostEqual(pp_residency_cap_s(self._cfg_slo(45.0)), 33.164)
        # A carried decode pays the seam in BOTH directions on top of the
        # residency, so a 45 s budget buys 33.2 s in PP, not 45.
        self.assertAlmostEqual(pp_residency_cap_s(self._cfg_slo(0.0)), 0.0)

    def test_an_slo_tighter_than_the_round_trip_collapses_to_zero(self):
        from sglang.srt.managers.phase_policy import pp_residency_cap_s

        self.assertEqual(pp_residency_cap_s(self._cfg_slo(4.0)), 0.0)

    def test_a_prefill_mountain_is_drained_not_deferred(self):
        """The live defect, replayed: 23,313 pending at 3 decoding, 15 s in."""
        cfg = self._cfg_slo(45.0)
        self.assertIsNone(self._pp(cfg, 23_313, 3, 15.0).direction)
        self.assertIsNone(self._pp(cfg, 23_313, 3, 30.0).direction)

    def test_the_cap_still_protects_decode_from_starving(self):
        cfg = self._cfg_slo(45.0)
        d = self._pp(cfg, 23_313, 3, 34.0)
        self.assertEqual(d.direction, PP_TO_TP)
        self.assertIn("decode stall cap", d.reason)
        self.assertIn("45s budget", d.reason.replace("45.0s", "45s"))

    def test_drain_still_ends_the_phase_first_when_it_can(self):
        """Drained means below ONE CHUNK, not below the entry break-even."""
        import dataclasses

        cfg = dataclasses.replace(self._cfg_slo(45.0), pp_exit_tokens=512)
        d = self._pp(cfg, 100, 3, 5.0)
        self.assertEqual(d.direction, PP_TO_TP)
        self.assertIn("DRAINED", d.reason)
        self.assertIn("exit condition: drained", d.reason)
        # A residual above one chunk is finished in PP, where it is faster.
        self.assertIsNone(self._pp(cfg, 9_000, 3, 5.0).direction)

    def test_the_legacy_stopwatch_states_what_drain_would_have_done(self):
        """Requirement (3): the deferring line must be auditable in place."""
        cfg = _cfg(
            decode_contention=1.0,
            flip_cost_s=5.918,
            pp_window_s=15.0,
            decode_stall_slo_s=0.0,
            pp_prefill_tok_s=4036.0,
            min_dwell_s=0.0,
        )
        d = self._pp(cfg, 23_313, 3, 15.0)
        self.assertEqual(d.direction, PP_TO_TP)
        self.assertIn("HAND-SET STOPWATCH", d.reason)
        self.assertIn("would STAY", d.reason)
        self.assertIn("23313", d.reason)
        self.assertIn("SGLANG_PHASE_POLICY_DECODE_STALL_SLO_S", d.reason)

    def test_a_declared_slo_overrides_the_hand_set_window(self):
        cfg = _cfg(
            decode_contention=1.0,
            flip_cost_s=5.918,
            pp_window_s=15.0,
            decode_stall_slo_s=45.0,
            pp_prefill_tok_s=4036.0,
            min_dwell_s=0.0,
        )
        self.assertIsNone(self._pp(cfg, 23_313, 3, 15.0).direction)

    def test_the_tp_floor_is_solved_from_the_seam(self):
        from sglang.srt.managers.phase_policy import solved_tp_decode_floor_s

        self.assertAlmostEqual(solved_tp_decode_floor_s(self._cfg_slo(45.0)), 11.836)
