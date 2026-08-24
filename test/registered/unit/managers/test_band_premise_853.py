"""#853 (iii) -- the secondary band's premise, and the bar the detector reads.

THE TICKET, verbatim from `/spinning/gpu-arb/WINDOW9-RESULT.md`::

    (iii) #835 secondary band premise: the ">N but <=30086 with 1 req
    decoding: too short for round trip" hold must break when the decode
    bundle is not draining (the 09:01:37 anomaly is the metal specimen).

THE METAL SPECIMEN, W24, `boot_w24_0824_0852.log` 09:01:37 (PP0)::

    LAYOUT-ECONOMY ANOMALY held=tp held_s=96.5 window_s=18.3
    pending_tok=22887 bar_tok=20057 seam_s=9.16 provenance=measured
    running_bs=1 bundle_entry=1 bundle_stall_s=10.0
    illegitimate=decode-bundle-not-draining

INVESTIGATED, AND THE TICKET'S OWN DIAGNOSIS DOES NOT SURVIVE THE ARITHMETIC.
Two separate defects are in that one line, and "break the band on a
non-draining bundle" is neither of them. Both halves are built here.

--------------------------------------------------------------------------
(A) THE DETECTOR READ A BAR THE POLICY DID NOT APPLY
--------------------------------------------------------------------------

The alarm prints ``bar_tok=20057`` and fires because ``pending 22887 >
20057``. But the hold it is contradicting is the SECONDARY BAND, whose own
text names a different bar: ``> N=20057 but <= 30086``. 30086 is
``effective_flip_threshold(cfg, running_bs=1)`` -- the #665-F1 differential
bar, which prices the decodes this cutover would strand ON TOP of the seam.
At the measured sigma = 1 that bar is ``N0 x (1 + 2B) / (1 + B)``, and at
B = 1 that is exactly ``20057 x 1.5 = 30086``.

So the policy compared 22887 against 30086 and held, correctly, by its own
arithmetic. The detector compared the same 22887 against 20057 and called
that hold a defect. **The alarm is a false positive, and the hold it names
was right.** The detector's own docstring shows how it got there: it reasons
about ``live_flip_tokens`` being repriced by a measured seam and is simply
unaware that a second, higher bar exists above it.

This is #819's rule -- "ONE READING ... the bar the policy APPLIED and the
bar the log REPORTS can never be two different numbers", pinned in
`test_flip_threshold_repricing_819.py::test_the_line_reports_the_bar_it_
actually_applied` -- holding INSIDE `phase_policy` and breaking at the module
boundary. It is also the #851 class root restated exactly
(`ANALYSE_851_funding_family.md` Sec. 1.2): "the DECIDERS still read their own
bookkeepers". Here the decider is a detector, and the bookkeeper it read was
its own.

The remedy is the one #819 already established: the applied bar is passed in
from the single authority that computes it, and it is REQUIRED, not
defaulted -- a default is what would let a caller silently re-create the
divergence.

--------------------------------------------------------------------------
(B) THE BAND HOLD HAS NO FALSIFIER FOR ITS OWN PREMISE
--------------------------------------------------------------------------

The real defect the specimen points at is structural, and it is not about the
bundle. THE BAND IS THE ONLY HOLD IN ``_decide_from_load`` THAT CANNOT BE
WRONG. Every neighbour carries a bound on its own premise:

  * min dwell        -- yielded by ``starved`` (#768), because a thrash bound
                        with nothing in flight protects no throughput;
  * drain mode       -- yielded by the #833 stall deadline, because a bundle
                        that holds station never converges;
  * the idle lock    -- yielded by the idle dwell (#748).

The band says "prefilling it in tp beats the round trip". That is a claim
about a RATE: the backlog is being prefilled here, at ``tp_prefill_tok_s``.
If the backlog is not being prefilled at all, the arithmetic is priced off a
rate that is not happening, and the hold becomes #833's shape one branch up:
a wait for something that has stopped.

WHY THE BUNDLE AXIS IS THE WRONG FALSIFIER, and this is why the ticket's
framing is not implemented. At the measured ``sigma = 1`` the scheduler gives
prefill absolute priority per iteration ("an iteration with any prefill chunk
pending runs THAT batch and never reaches the decode branch"), so while
prefill is pending in TP the decode bundle CANNOT shrink -- by construction,
not by defect. "Bundle not draining" is therefore IMPLIED by the very
condition the band operates under, and a band that broke on it would collapse
to the plain break-even N0 for every load with a decode resident, silently
deleting the #665-F1 differential model. That is a policy rewrite wearing a
bug fix, and it is refused here.

THE HONEST FALSIFIER IS ON THE AXIS THE CLAIM IS MADE ON: prefill progress.
``pending_prefill_tokens`` is "prompt tokens ADMITTED BUT NOT YET COMPUTED"
and is measured at the chunk fill boundary ``extend_range.end``
(`scheduler.py:10463`), so it drops by a chunk every round a chunk is
computed -- a long prompt under chunked prefill moves it continuously. Pending
FROZEN for longer than one decode window therefore means no chunk was
computed for a whole decode window, which is precisely "the prefill this band
promised is not happening". The clock already exists: ``#677(a)``'s
``state.last_prefill_progress_at``, stamped by ``observe_idle`` whenever
pending goes DOWN.

AND IT DOES NOT FIRE ON THE SPECIMEN. In W24's stuck phase pending oscillated
0 -> ~22.5k -> 0 on a ~5-min period, so prefill progress was live throughout
and this break would have stayed silent -- consistent with (A), which says
that hold was correct. The two halves agree on the specimen instead of
double-counting it, which is what makes them separable defects rather than
one defect described twice.

--------------------------------------------------------------------------
(C) NOT REBUILT: the staging rate limit already clears on a completion
--------------------------------------------------------------------------

The ticket pairs (iii) with "the staging rate limit (60 s cap) arms-blocks at
>13k pending -- interaction to be designed". PRIOR ART: the property is
already implemented (``note_flip_completed`` pops ``last_abandon_at``,
``arm_refusals``, ``arm_hold_until`` and ``arm_degraded`` for the direction)
and already pinned by
`test_phase_policy_flip_reachability.py::test_a_completion_clears_the_
staging_rate_limit_outright`. Nothing is rebuilt here. The interaction that
remains is that a band-break, like every other flip the load wants, is paced
by that limiter -- which is correct and is left alone: `_decide_rules`
applies the limiter AFTER `_decide_from_load`, so a break-out of the band is
storm-bounded by the same rule as any other arm, and `_demand_outweighs_a_
retry` still overrides it when the backlog is worth more than the wait.
"""

import inspect
import unittest

from sglang.srt.managers import layout_conformance as lc
from sglang.srt.managers import phase_policy as pp
from sglang.srt.managers.phase_policy import (
    PHASE_TP,
    TP_TO_PP,
    PhasePolicyConfig,
    PhasePolicyInputs,
    PhasePolicyState,
    decide,
    drain_stall_deadline_s,
    effective_flip_threshold,
    live_flip_tokens,
    observe_idle,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# --------------------------------------------------------------------------
# The W24 specimen, verbatim.
# --------------------------------------------------------------------------

#: The break-even the alarm printed as `bar_tok`.
W24_N0 = 20057
#: The bar the HOLD named and the policy actually compared against.
W24_APPLIED_BAR = 30086
#: Pending at the anomaly: inside the band, above N0, below the applied bar.
W24_PENDING = 22887
#: One request decoding, and it was already decoding at phase entry.
W24_BUNDLE = 1
#: The measured seam.
W24_SEAM_S = 9.16
#: `drain_stall_deadline_s` on that seam: max(10.0, 2 x 9.16).
W24_WINDOW_S = 18.32
#: How long the layout had been held when the alarm fired.
W24_HELD_S = 96.5
#: The bundle had shrunk 10s earlier -- inside the window, so the stall clock
#: was NOT what made the detector call it non-draining. Net progress was
#: (1 running vs 1 at entry), which is the sigma = 1 steady state.
W24_BUNDLE_STALL_S = 10.0


class _PricingFixture(CustomTestCase):
    """Save/restore the process-global seam estimator and boot pricing.

    The band is priced off module-global state, so a test that left it
    modified would make its neighbours' bars depend on execution order.
    """

    def setUp(self):
        self._saved_est = pp._FLIP_COST_ESTIMATOR
        self._saved_boot = pp._FLIP_TOKENS_AT_BOOT
        self._saved_pricing = pp._FLIP_TOKENS_PRICING
        self._saved_said = pp._FLIP_TOKENS_STALE_SAID
        pp._FLIP_COST_ESTIMATOR = None

    def tearDown(self):
        pp._FLIP_COST_ESTIMATOR = self._saved_est
        pp._FLIP_TOKENS_AT_BOOT = self._saved_boot
        pp._FLIP_TOKENS_PRICING = self._saved_pricing
        pp._FLIP_TOKENS_STALE_SAID = self._saved_said


def _cfg(**kw):
    """A config whose secondary band is non-empty at running_bs=1."""
    base = dict(
        enabled=True,
        flip_tokens=W24_N0,
        flip_cost_s=W24_SEAM_S,
        min_dwell_s=0.0,
        pp_window_s=15.0,
        tp_decode_floor_s=0.0,
        prefill_runs_in_tp=True,
        decode_contention=1.0,
    )
    base.update(kw)
    return PhasePolicyConfig(**base)


def _in_the_band(cfg, running_bs):
    """A pending value strictly inside this config's secondary band.

    Derived from the two bars rather than pinned, so the test cannot drift
    away from the band it means to exercise.
    """
    lo = live_flip_tokens(cfg)
    hi = effective_flip_threshold(cfg, running_bs)
    assert hi > lo + 1, f"no band to test: N={lo}, threshold={hi}"
    return (lo + hi) // 2


def _drive(cfg, state, pendings, *, running_bs=1, t0=1000.0, step=1.0):
    """Feed the policy a series of observations, one per second.

    ``observe_idle`` is what stamps the progress clocks, so driving the real
    accessor is what exercises them; injecting the clock directly would test
    the injection.
    """
    d = None
    for i, pending in enumerate(pendings):
        inp = PhasePolicyInputs(
            phase=PHASE_TP,
            pending_prefill_tokens=int(pending),
            running_bs=running_bs,
            now=t0 + i * step,
        )
        observe_idle(state, inp)
        d = decide(cfg, state, inp)
    return d


# ==========================================================================
# (B) THE POLICY -- the band's missing falsifier
# ==========================================================================


class TheBandHoldsWhileItsPremiseIsTrue(_PricingFixture):
    """The can-fail direction for the whole section.

    A break implemented as "always leave the band" would pass every assertion
    in the next class and would delete the #665-F1 differential model. These
    are the assertions that stop it.
    """

    def test_a_progressing_prefill_still_holds_the_band(self):
        cfg = _cfg()
        state = PhasePolicyState()
        pending = _in_the_band(cfg, 1)
        # A chunk computed every round: pending falls continuously, for well
        # over one decode window. The decrement is small enough that the
        # backlog stays INSIDE the band throughout -- walking out of the band
        # would change the verdict for a reason that has nothing to do with
        # progress, and would test the wrong thing.
        series = [pending - 50 * i for i in range(40)]
        d = _drive(cfg, state, series)
        self.assertIsNone(d.direction, f"expected a hold, got {d.reason}")
        self.assertIn("too short for the round trip", d.reason)

    def test_a_stall_shorter_than_one_decode_window_still_holds(self):
        cfg = _cfg()
        state = PhasePolicyState()
        pending = _in_the_band(cfg, 1)
        held = int(drain_stall_deadline_s(cfg)) - 2
        d = _drive(cfg, state, [pending] * held)
        self.assertIsNone(d.direction, f"expected a hold, got {d.reason}")
        self.assertIn("too short for the round trip", d.reason)

    def test_below_the_band_a_stall_does_not_manufacture_a_flip(self):
        """Under the break-even there is no band and no claim to falsify.

        A stall must not become a general-purpose flip trigger: below N0 the
        round trip does not repay at any drain rate, which is the #656
        anti-thrash property.
        """
        cfg = _cfg()
        state = PhasePolicyState()
        pending = live_flip_tokens(cfg) - 1
        d = _drive(cfg, state, [pending] * 60)
        self.assertIsNone(d.direction, f"expected a hold, got {d.reason}")
        self.assertNotIn("prefill progress", d.reason.lower())


class TheBandBreaksWhenItsPremiseIsFalse(_PricingFixture):
    """RED FIRST: today the band holds forever on a frozen backlog."""

    def test_a_frozen_backlog_breaks_the_band(self):
        cfg = _cfg()
        state = PhasePolicyState()
        pending = _in_the_band(cfg, 1)
        held = int(drain_stall_deadline_s(cfg)) + 3
        d = _drive(cfg, state, [pending] * held)
        self.assertEqual(
            TP_TO_PP,
            d.direction,
            f"the band held a backlog that is not being prefilled: {d.reason}",
        )

    def test_the_break_names_the_premise_it_falsified(self):
        cfg = _cfg()
        state = PhasePolicyState()
        pending = _in_the_band(cfg, 1)
        held = int(drain_stall_deadline_s(cfg)) + 3
        d = _drive(cfg, state, [pending] * held)
        # The reader must be able to tell this from an ordinary threshold
        # arm: it is a hold that was withdrawn, and the number that withdrew
        # it is the stall, not the backlog.
        self.assertIn("prefill progress", d.reason.lower())
        self.assertIn(str(pending), d.reason)
        for token in ("has not gone down", "deadline"):
            self.assertIn(token, d.reason)

    def test_the_break_uses_the_policys_own_decode_window(self):
        """DERIVED, not pinned: a config with a dearer seam waits longer.

        The window is `drain_stall_deadline_s`, the same quantity #833 and
        the #838 detector already use, so policy and detector cannot come to
        hold two different ideas of one decode window.
        """
        cheap = _cfg()
        dear = _cfg(flip_cost_s=40.0)
        self.assertLess(drain_stall_deadline_s(cheap), drain_stall_deadline_s(dear))
        held = int(drain_stall_deadline_s(cheap)) + 3
        state = PhasePolicyState()
        pending = _in_the_band(cheap, 1)
        self.assertEqual(TP_TO_PP, _drive(cheap, state, [pending] * held).direction)
        # The same elapsed stall, on the dearer seam, is still inside the
        # window and must not break.
        state2 = PhasePolicyState()
        pending2 = _in_the_band(dear, 1)
        self.assertIsNone(_drive(dear, state2, [pending2] * held).direction)

    def test_min_dwell_still_outranks_the_break(self):
        """The thrash bound is checked before any rule below it, and a new
        escape hatch must not be a way around it."""
        cfg = _cfg(min_dwell_s=600.0)
        state = PhasePolicyState()
        pending = _in_the_band(cfg, 1)
        held = int(drain_stall_deadline_s(cfg)) + 3
        state.last_flip_at = 1000.0
        d = _drive(cfg, state, [pending] * held)
        self.assertIsNone(d.direction, f"min dwell was bypassed: {d.reason}")
        self.assertIn("min dwell", d.reason)


# ==========================================================================
# (A) THE DETECTOR -- one bar, the one the policy applied
# ==========================================================================


def _economy(**over):
    """The W24 09:01:37 specimen, as the detector saw it."""
    kw = dict(
        phase=PHASE_TP,
        held_s=W24_HELD_S,
        window_s=W24_WINDOW_S,
        pending_prefill_tokens=W24_PENDING,
        live_flip_tokens=W24_N0,
        applied_bar_tokens=W24_APPLIED_BAR,
        live_flip_cost_s=W24_SEAM_S,
        price_measured=True,
        hold_reason=(
            f"pending prefill {W24_PENDING} tok > N={W24_N0} but "
            f"<= {W24_APPLIED_BAR} with 1 req decoding: too short for the "
            f"round trip to beat prefilling it in tp"
        ),
        since_flip_s=90.0,
        min_dwell_s=3.0,
        staging_active=False,
        running_bs=W24_BUNDLE,
        bundle_at_phase_entry=W24_BUNDLE,
        bundle_stall_s=W24_BUNDLE_STALL_S,
    )
    kw.update(over)
    return lc.economy_divergence_verdict(**kw)


class TheDetectorMeasuresAgainstTheBarThePolicyApplied(unittest.TestCase):
    """RED FIRST: today this fires, and the hold it indicts was correct."""

    def tearDown(self):
        lc.reset_for_test()

    def test_the_w24_specimen_is_correct_economics_not_an_anomaly(self):
        alarm, detail = _economy()
        self.assertFalse(
            alarm, f"a hold inside the policy's own band was alarmed on: {detail}"
        )
        self.assertIn("correct economics", detail)

    def test_the_decline_names_both_bars(self):
        _, detail = _economy()
        self.assertIn(str(W24_APPLIED_BAR), detail)
        self.assertIn(str(W24_PENDING), detail)

    def test_pending_above_the_applied_bar_still_fires(self):
        """THE CAN-FAIL DIRECTION for the class above.

        A detector that simply stopped alarming would pass every assertion
        up there. Above the bar the policy actually applied, the hold is
        contradicted by the policy's own arithmetic and must be announced.
        """
        alarm, detail = _economy(pending_prefill_tokens=W24_APPLIED_BAR + 1)
        self.assertTrue(alarm)
        self.assertIn(lc.ALARM_ECONOMY, detail)

    def test_exactly_at_the_applied_bar_is_silent(self):
        alarm, _ = _economy(pending_prefill_tokens=W24_APPLIED_BAR)
        self.assertFalse(alarm)

    def test_the_line_reports_the_applied_bar_when_it_differs(self):
        _, detail = _economy(pending_prefill_tokens=W24_APPLIED_BAR + 1)
        self.assertIn(f"bar_tok={W24_N0}", detail)
        self.assertIn(f"applied_bar_tok={W24_APPLIED_BAR}", detail)

    def test_the_applied_bar_is_required_and_not_defaulted(self):
        """A default is what would let the divergence come back silently.

        This is the same failure the #819 rule was written against: two
        numbers for one comparison, with the wrong one available for free.
        """
        sig = inspect.signature(lc.economy_divergence_verdict)
        param = sig.parameters["applied_bar_tokens"]
        self.assertIs(
            param.default,
            inspect.Parameter.empty,
            "applied_bar_tokens must be required; a default re-creates the "
            "two-bookkeeper divergence this fix removes",
        )

    def test_an_applied_bar_below_the_break_even_cannot_lower_the_gate(self):
        """The blast radius, stated as a test.

        The gate takes the HIGHER of the two bars, so this change can only
        ever remove false positives and can never invent an alarm. The case
        that reaches here is strict purity, where `effective_flip_threshold`
        returns 0 by construction -- a caller passing that must not turn the
        detector into one that alarms on every hold above zero tokens.
        """
        alarm, _ = _economy(pending_prefill_tokens=W24_N0 - 1, applied_bar_tokens=0)
        self.assertFalse(alarm)
        above, _ = _economy(pending_prefill_tokens=W24_N0 + 1, applied_bar_tokens=0)
        self.assertTrue(above, "the break-even must still stand as the floor")

    def test_a_window3_shaped_span_far_above_both_bars_still_fires(self):
        """The #838 specimen's own shape: 119x the bar, seven decoding.

        The differential bar is bounded above by 2 x N0 (the supremum proved
        in `test_phase_policy_flip_reachability.py`), so a backlog two orders
        of magnitude above N0 is above BOTH bars and stays an anomaly.
        """
        alarm, detail = _economy(
            pending_prefill_tokens=836048,
            live_flip_tokens=7004,
            applied_bar_tokens=13132,
            running_bs=7,
            bundle_at_phase_entry=7,
            bundle_stall_s=0.4,
            live_flip_cost_s=3.2,
            window_s=10.0,
            held_s=41.0,
            hold_reason="decode bundle running: 7 of 7 req still decoding",
        )
        self.assertTrue(alarm)
        self.assertIn(lc.ILLEGITIMATE_BUNDLE_NOT_DRAINING, detail)


if __name__ == "__main__":
    unittest.main()
