"""#893 -- the entry economy priced stranding against a knob that cannot fire.

THE DEFECT, stacked on #889 (base 2b13ba92d1)
---------------------------------------------
``effective_flip_threshold`` charges a flip for the decodes it strands: a
request carried into the PP layout cannot emit another token until the
instance is back in TP, so the surcharge is ``weight x running_bs x W`` with
``W`` = "how long is a carried decode stranded". Every reader of ``W`` in that
economy took it from ``cfg.pp_window_s``.

#889 established that ``cfg.pp_window_s`` is UNREACHABLE whenever a decode
stall SLO above the round trip is declared: ``decide`` reaches the hand-set
stopwatch only through ``cap <= 0``, and the solved cap
``slo - 2 x flip_cost_s`` governs instead. So on the live w38b/w39/w40 line
(window 15 s, SLO 180 s, seam 3.2 s) a carried decode really strands for
173.6 s while the economy charged the flip for 15 s.

MEASURED AT THE BASE, sigma = 0 (the shipped value):

    window 15 / SLO 180   ladder  [7004,  39835,  72666,  105498]
    with the TRUE W=173.6 ladder  [7004, 386971, 766938, 1146905]

an understatement of ~9.7x -- and every one of those flips was bought at a
price the instance did not actually pay.

WORSE, THE CONFIGURATION ``validate_purity_policy_pair`` RECOMMENDS. Clearing
``--phase-policy-pp-window-s`` and letting the SLO be the bound is the
sanctioned way to express "the SLO is the better bound". With window 0 the
surcharge term is ``weight x running_bs x 0`` and the ladder goes FLAT --
[7004, 7004, 7004, 7004] -- i.e. the stranded-decode surcharge is switched off
entirely, while decodes still strand for 173.6 s. The knob that silently
stopped bounding the phase silently stopped pricing it too.

WHAT THIS SUITE PINS
--------------------
1. The two falsifiers above: the 9.7x understatement and the flat ladder.
2. THE BASE RUNG DOES NOT MOVE. This fix makes flips ~10x dearer at every
   populated rung, deliberately -- but a reprice that lifted ``base`` too
   would price a flip out of reach at zero decodes, where there is nothing to
   strand. Only the SLOPE may grow.
3. THE ECONOMY IS INDIFFERENT TO HOW THE TERM ARISES (#889 M7, widened). For
   every configuration the ladder must equal the ladder of the config whose
   ``pp_window_s`` IS the effective term with no SLO declared. One invariant
   over all of the economy's ``W`` readers at once, which is what stops a
   second authority on "how long is a decode stranded" from growing back.
4. Configurations with no SLO are byte-identical to before -- the fix must
   move only the ones that were being misread.
5. The honest opposite direction: with the window cleared under an SLO the
   differential model used to fall into its "no bound at all" branch and
   return UNREACHABLE. There IS a bound, so the answer is finite now. The fix
   makes flips dearer where a bound was under-priced and CHEAPER where a real
   bound was denied; both follow from reading the term that governs.

THE BLAST RADIUS, MEASURED RATHER THAN ASSUMED
----------------------------------------------
``boot_w38rerun_0826_1304.log:83`` armed with ``decode contention 1``: the
SHIPPED sigma is 1, not the in-code default ``DEFAULT_DECODE_CONTENTION = 0``,
and its ladder [7004, 10506, 11674, 12257, 12608] stays in regime A at every
rung ``--max-running-requests 4`` can reach, where W does not enter the answer
at all. That config also runs ``--phase-flip-purity strict``, and strict purity
sets ``prefill_runs_in_tp`` False (scheduler.py), which collapses the threshold
to 0 outright. So on TODAY's ship config this reprice moves no arming decision;
what it moves there is the refusal line, which quoted the inert 15 s.

The reprice bites where the ladder is actually the gate: the sigma = 0
one-sided surcharge (the 9.7x above), small sigma at high ``running_bs`` where
regime B is entered, and any deployment not running strict purity.
``test_the_live_boot_shape_at_sigma_one_does_not_move`` pins that honestly, so
the next reader does not infer a live 10x from the ticket title.

THIS SUITE CHANGES NO POLICY. ``decide``'s exits are untouched, the trigger is
still drain, and no number from #856/#819 is recalibrated here -- those are
re-measured on a window after this lands, not converted at the desk.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import dataclasses
import unittest

from sglang.srt.managers.phase_policy import (
    PP_TO_TP,
    UNREACHABLE_FLIP_THRESHOLD,
    PhasePolicyConfig,
    PhasePolicyInputs,
    PhasePolicyState,
    decide,
    effective_flip_threshold,
    effective_pp_exit_term,
    stranded_decode_s,
)
from sglang.test.test_utils import CustomTestCase

# The booted values on the w38b/w39/w40 line, so every number below is live.
N = 7004
SEAM = 3.2
WINDOW = 15.0
SLO = 180.0
CAP = SLO - 2 * SEAM  # 173.6

#: What the economy computed while it read the inert knob, and what it must
#: compute now that it reads the bound that governs. Both measured on the base
#: at 2b13ba92d1, sigma = 0, strand weight 1.
LADDER_AT_THE_INERT_WINDOW = [7004, 39835, 72666, 105498, 138329]
LADDER_AT_THE_TRUE_TERM = [7004, 386971, 766938, 1146905, 1526872]


def _cfg(**kw):
    base = dict(
        enabled=True,
        flip_tokens=N,
        min_dwell_s=0.0,
        tp_decode_floor_s=0.0,
        prefill_runs_in_tp=True,
        flip_cost_s=SEAM,
        pp_window_s=WINDOW,
        decode_stall_slo_s=SLO,
        pp_prefill_tok_s=7245.5,
        tp_prefill_tok_s=1681.0,
    )
    base.update(kw)
    return PhasePolicyConfig(**base)


def _ladder(cfg, upto=5):
    return [effective_flip_threshold(cfg, b) for b in range(upto)]


class TestTheTwoFalsifiers(CustomTestCase):
    """Both were RED at 2b13ba92d1 and are the whole of #893."""

    def test_the_live_pair_prices_stranding_at_the_cap_not_the_inert_window(self):
        """FALSIFIER (a): the 9.7x understatement on the shipped line."""
        cfg = _cfg()
        # The premise: the window really is superseded here (#889).
        self.assertAlmostEqual(effective_pp_exit_term(cfg)[1], CAP)
        self.assertEqual(_ladder(cfg), LADDER_AT_THE_TRUE_TERM)
        # ... which is emphatically not what the inert knob would have said.
        self.assertNotEqual(_ladder(cfg), LADDER_AT_THE_INERT_WINDOW)

    def test_a_cleared_window_under_an_slo_is_not_free_stranding(self):
        """FALSIFIER (b): the flat ladder on the RECOMMENDED configuration.

        `validate_purity_policy_pair` accepts a declared SLO as the substitute
        bound, so clearing the window is the sanctioned way to say it. Doing
        so used to disable the stranded-decode surcharge outright.
        """
        cfg = _cfg(pp_window_s=0.0)
        self.assertAlmostEqual(effective_pp_exit_term(cfg)[1], CAP)
        ladder = _ladder(cfg)
        self.assertNotEqual(ladder, [N] * 5)  # the flat ladder, verbatim
        for lo, hi in zip(ladder, ladder[1:]):
            self.assertLess(lo, hi, ladder)
        self.assertEqual(ladder, LADDER_AT_THE_TRUE_TERM)


class TestTheBaseRungDoesNotMove(CustomTestCase):
    """THE GEFAHRRICHTUNG. This fix makes flips ~10x dearer on purpose; it must
    not make them impossible. Nothing is stranded at zero decodes, so the floor
    the ladder stands on is exactly the seam break-even, before and after."""

    def test_zero_decodes_is_the_unsurcharged_break_even(self):
        for cfg in _every_shape():
            with self.subTest(cfg=_shape_name(cfg)):
                self.assertEqual(effective_flip_threshold(cfg, 0), N)
                self.assertEqual(effective_flip_threshold(cfg, -1), N)

    def test_only_the_slope_grew_never_the_floor(self):
        """The comparison the reprice is allowed to make: same first rung,
        strictly larger ones above it."""
        before, after = _ladder(_cfg(decode_stall_slo_s=0.0)), _ladder(_cfg())
        self.assertEqual(before[0], after[0])
        self.assertEqual(before[0], N)
        for lo, hi in zip(before[1:], after[1:]):
            self.assertLess(lo, hi)

    def test_every_rung_still_clears_the_break_even_floor(self):
        for cfg in _every_shape():
            for b in range(5):
                with self.subTest(cfg=_shape_name(cfg), bs=b):
                    self.assertGreaterEqual(effective_flip_threshold(cfg, b), N)


class TestOneAuthorityOnHowLongADecodeStrands(CustomTestCase):
    """#889 M7 widened from `decide` vs the reporter to the ECONOMY.

    The invariant is stated without re-deriving the term: whatever the economy
    charges for stranding, it must be what it would charge if the effective
    term had been declared as a plain window. Any reader still holding
    `cfg.pp_window_s` breaks it.
    """

    def _as_declared_window(self, cfg):
        return dataclasses.replace(
            cfg, pp_window_s=stranded_decode_s(cfg), decode_stall_slo_s=0.0
        )

    def test_the_ladder_depends_only_on_the_effective_term(self):
        for cfg in _every_shape():
            with self.subTest(cfg=_shape_name(cfg)):
                self.assertEqual(_ladder(cfg), _ladder(self._as_declared_window(cfg)))

    def test_the_named_term_is_the_one_889_reports(self):
        """No second derivation: the economy's W is `effective_pp_exit_term`'s
        value, not a parallel solve of the same question."""
        for cfg in _every_shape():
            with self.subTest(cfg=_shape_name(cfg)):
                self.assertEqual(stranded_decode_s(cfg), effective_pp_exit_term(cfg)[1])

    def test_decide_leaves_pp_at_the_same_term_the_economy_charges(self):
        """The policy half is UNCHANGED here -- this only pins that the price
        and the exit are still talking about the same seconds."""
        for cfg in (_cfg(), _cfg(decode_stall_slo_s=0.0), _cfg(decode_stall_slo_s=4.0)):
            term = stranded_decode_s(cfg)
            with self.subTest(term=term):
                self.assertIsNone(_in_pp(cfg, term - 0.1).direction)
                self.assertEqual(_in_pp(cfg, term + 0.1).direction, PP_TO_TP)

    def test_the_refusal_line_names_the_seconds_it_refused_over(self):
        """The one artifact an operator reads when a flip is declined said
        `for a 15s window` while the decode would have stranded 173.6 s."""
        cfg = _cfg()
        pending = (N + effective_flip_threshold(cfg, 3)) // 2
        decision = decide(
            cfg,
            PhasePolicyState(),
            PhasePolicyInputs(
                phase="tp", pending_prefill_tokens=pending, running_bs=3, now=1000.0
            ),
        )
        self.assertIsNone(decision.direction, decision.reason)
        self.assertIn("strand", decision.reason)
        self.assertIn("173.6", decision.reason)


class TestTheDifferentialModelReadsItToo(CustomTestCase):
    """#665-F1's solve uses W twice more: as the regime boundary (`N/P <= W`)
    and as the saturation term in regime B. Both were on the inert knob."""

    def test_regime_b_saturates_at_the_term_that_governs(self):
        cfg = _cfg(decode_contention=0.05)
        # b=4 is past the regime boundary, so W enters the answer directly.
        self.assertEqual(effective_flip_threshold(cfg, 4), 1211239)
        self.assertNotEqual(effective_flip_threshold(cfg, 4), 154208)

    def test_the_regime_boundary_moves_with_the_term(self):
        """At b=3 the inert knob put the solve in regime B and the true term
        keeps it in regime A -- a different formula, not just a scaling."""
        cfg = _cfg(decode_contention=0.05)
        self.assertEqual(effective_flip_threshold(cfg, 3), 169633)
        self.assertNotEqual(effective_flip_threshold(cfg, 3), 123421)

    def test_a_declared_slo_is_a_bound_so_the_answer_is_finite(self):
        """THE HONEST OPPOSITE DIRECTION. With the window cleared the solve
        took its `no bound at all` branch -- "a carried decode waits out the
        whole prefill" -- and returned UNREACHABLE. An SLO IS a bound, so
        regime B applies and a large-but-reachable rung is the true answer."""
        cfg = _cfg(pp_window_s=0.0, decode_contention=0.03)
        self.assertLess(effective_flip_threshold(cfg, 4), UNREACHABLE_FLIP_THRESHOLD)
        self.assertEqual(effective_flip_threshold(cfg, 4), 1369040)

    def test_no_bound_of_any_kind_still_means_unreachable(self):
        """And the branch is not deleted: with neither knob declared there
        really is no bound on the stranding, and that answer stands."""
        cfg = _cfg(pp_window_s=0.0, decode_stall_slo_s=0.0, decode_contention=0.03)
        self.assertEqual(effective_flip_threshold(cfg, 4), UNREACHABLE_FLIP_THRESHOLD)


class TestConfigurationsThatWereNeverMisreadDoNotMove(CustomTestCase):
    """The fix must move exactly the configurations #889 showed were misread."""

    def test_a_window_only_deployment_is_byte_identical(self):
        self.assertEqual(
            _ladder(_cfg(decode_stall_slo_s=0.0)), LADDER_AT_THE_INERT_WINDOW
        )

    def test_an_slo_below_the_round_trip_hands_the_window_back(self):
        """cap collapses to 0, the stopwatch governs again, and so does its
        price -- the mirror direction of the supersession."""
        self.assertEqual(
            _ladder(_cfg(decode_stall_slo_s=4.0)), LADDER_AT_THE_INERT_WINDOW
        )

    def test_a_drain_only_deployment_charges_nothing_for_stranding(self):
        """No timed bound is declared at all, so under sigma = 0 there is no
        window to charge and the ladder is the break-even, as before."""
        cfg = _cfg(pp_window_s=0.0, decode_stall_slo_s=0.0)
        self.assertEqual(stranded_decode_s(cfg), 0.0)
        self.assertEqual(_ladder(cfg), [N] * 5)

    def test_the_surcharge_can_still_be_switched_off_by_weight(self):
        self.assertEqual(_ladder(_cfg(decode_strand_weight=0.0)), [N] * 5)

    def test_an_unmeasured_seam_still_disables_the_whole_reprice(self):
        self.assertEqual(_ladder(_cfg(flip_cost_s=0.0)), [N] * 5)

    def test_the_live_boot_shape_at_sigma_one_does_not_move(self):
        """THE HONEST BLAST RADIUS, pinned rather than claimed.

        `boot_w38rerun_0826_1304.log:83` armed with `decode contention 1` --
        the SHIPPED sigma is 1, not the in-code default of 0 -- and at
        sigma = 1 the solve stays in regime A at every rung this rig can reach
        (`--max-running-requests 4`), where W does not enter the answer at all.
        So this reprice moves nothing on today's ship config, and the 9.7x is
        measured on the sigma = 0 one-sided surcharge and on small sigma at
        high running_bs. Pinned here so the next reader does not infer a live
        10x from the ticket title.
        """
        live = _cfg(decode_contention=1.0)
        self.assertEqual(_ladder(live), [7004, 10506, 11674, 12257, 12608])
        self.assertEqual(
            _ladder(live), _ladder(_cfg(decode_contention=1.0, decode_stall_slo_s=0.0))
        )


def _every_shape():
    """The configurations the term can arise in, each hit by both models."""
    out = []
    for sigma in (0.0, 0.03, 0.05, 0.3):
        for window, slo in (
            (WINDOW, SLO),
            (WINDOW, 0.0),
            (0.0, SLO),
            (0.0, 0.0),
            (WINDOW, 4.0),
        ):
            out.append(
                _cfg(
                    decode_contention=sigma, pp_window_s=window, decode_stall_slo_s=slo
                )
            )
    return out


def _shape_name(cfg):
    return f"sigma={cfg.decode_contention} w={cfg.pp_window_s} slo={cfg.decode_stall_slo_s}"


def _in_pp(cfg, in_pp, pending=200_000, bs=3):
    state = PhasePolicyState()
    state.phase_since = 1000.0
    return decide(
        cfg,
        state,
        PhasePolicyInputs(
            phase="pp",
            pending_prefill_tokens=pending,
            running_bs=bs,
            now=1000.0 + in_pp,
        ),
    )


if __name__ == "__main__":
    unittest.main()
