"""#838 -- the wrong layout must announce itself, on the round it happens.

WHY THIS FILE EXISTS. Three times in a row the fact that prefill was running
in the TP layout while PP was the intended one was found by a HUMAN READING A
LOG. The user's standing order is that the code must say it instead, at once.
So the property under test is not "the layout is right" -- this tree already
has a policy for that -- but "a divergence between what was DECIDED and what
was DONE is announced".

THE SPECIMEN, boot_window3_0823_1733, verbatim::

    2253  Prefill batch phase=tp
      96  Prefill batch phase=pp
      21  PHASE-FLIP DONE

    the bar armed with ...... N=7004 tok (break-even 3.2s seam)
    max pending driven ...... 836,048 tok = 119x the bar
    hold reason, 151 times ... "decode bundle running: 7 of 7 req still
                                decoding, ... tok prefill waiting"
    running_bs throughout .... 4 to 7, never 0

That last line is why this cannot be an extension of #699. The
admission-wedge verdict short-circuits to "not wedged" the moment
``running > 0`` (invariant_checker.py:598) -- correctly, for its own class --
so it is blind to this span by construction. Class 2 below has no such guard
and is tested at ``running_bs=7``.

THE TRAP THIS FILE MUST NOT FALL INTO, and half the assertions here exist to
hold it shut: PREFILL IN TP AT AN HONESTLY HIGH MEASURED FLIP PRICE IS CORRECT
ECONOMICS. A detector that alarmed on it would be a detector that alarmed on
the system working, and it would be switched off within a boot. So every
economy test comes in a PAIR -- the same shape at a cheap measured price
(must fire) and at an honest expensive one (must stay silent) -- and the
mutant section proves the over-eager version dies.
"""

import types
import unittest

from sglang.srt.managers import layout_conformance as lc
from sglang.srt.managers.phase_policy import (
    PHASE_PP,
    PHASE_TP,
    PhasePolicyConfig,
    drain_stall_deadline_s,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5)

#: The bar this boot armed with, verbatim from the log.
W3_BAR = 7004
#: The maximum pending prefill the load driver reached: 119x the bar.
W3_MAX_PENDING = 836048
#: The bundle the hold named, and the running_bs it named it at.
W3_BUNDLE = 7
#: The seam the bar was priced off, verbatim: "break-even 3.2s".
W3_SEAM_S = 3.2
#: One decode window on that config. `drain_stall_deadline_s` floors at 10.0.
W3_WINDOW_S = 10.0


def _economy(**over):
    """The window-3 span, as the detector sees it. Cheap MEASURED price.

    Every field is the specimen's own number. Overriding one field at a time
    is how each gate below is exercised in isolation.
    """
    kw = dict(
        phase=PHASE_TP,
        held_s=41.0,
        window_s=W3_WINDOW_S,
        pending_prefill_tokens=W3_MAX_PENDING,
        live_flip_tokens=W3_BAR,
        live_flip_cost_s=W3_SEAM_S,
        price_measured=True,
        hold_reason="decode bundle running: 7 of 7 req still decoding",
        since_flip_s=30.0,
        min_dwell_s=3.0,
        staging_active=False,
        running_bs=W3_BUNDLE,
        bundle_at_phase_entry=W3_BUNDLE,
        bundle_stall_s=0.4,
    )
    kw.update(over)
    return lc.economy_divergence_verdict(**kw)


class TheEconomyWindowIsDerivedNotPinned(unittest.TestCase):
    """N is the policy's own decode window, not a number chosen here."""

    def test_n_is_the_policys_own_drain_stall_deadline(self):
        cfg = PhasePolicyConfig(
            enabled=True, flip_tokens=W3_BAR, flip_cost_s=W3_SEAM_S
        )
        self.assertEqual(
            lc.economy_window_s(drain_stall_deadline_s(cfg)),
            drain_stall_deadline_s(cfg),
        )

    def test_n_moves_when_the_policys_own_window_moves(self):
        cheap = PhasePolicyConfig(enabled=True, flip_tokens=W3_BAR, flip_cost_s=1.0)
        dear = PhasePolicyConfig(enabled=True, flip_tokens=W3_BAR, flip_cost_s=40.0)
        self.assertLess(
            lc.economy_window_s(drain_stall_deadline_s(cheap)),
            lc.economy_window_s(drain_stall_deadline_s(dear)),
        )

    def test_no_window_declared_is_declined_not_alarmed(self):
        alarm, detail = _economy(window_s=0.0)
        self.assertFalse(alarm)
        self.assertIn("no decode window declared", detail)


class TheWindow3SpanIsAnAnomaly(unittest.TestCase):
    """The measured specimen, at a cheap measured price, must fire."""

    def test_it_fires(self):
        alarm, detail = _economy()
        self.assertTrue(alarm)
        self.assertIn(lc.ALARM_ECONOMY, detail)

    def test_it_fires_with_seven_requests_running(self):
        """The #699 coverage gap, stated as a test.

        `admission_wedge_verdict` returns "not wedged" at any running > 0.
        This detector must not inherit that blindness.
        """
        alarm, _ = _economy(running_bs=W3_BUNDLE)
        self.assertTrue(alarm)

    def test_the_line_carries_the_numbers_a_reader_needs(self):
        _, detail = _economy()
        for token in (
            f"pending_tok={W3_MAX_PENDING}",
            f"bar_tok={W3_BAR}",
            "provenance=measured",
            f"running_bs={W3_BUNDLE}",
            "held=tp",
        ):
            self.assertIn(token, detail)

    def test_it_quotes_the_verdict_it_contradicts(self):
        _, detail = _economy(hold_reason="decode bundle running: 7 of 7")
        self.assertIn('verdict="decode bundle running: 7 of 7"', detail)


class HonestEconomicsMustNotAlarm(unittest.TestCase):
    """The other half of every pair. These are the assertions that keep the
    detector usable: each one is the same shape as a firing case, differing
    only in the term that makes holding CORRECT."""

    def test_an_honestly_expensive_measured_seam_is_silent(self):
        """The repriced bar rises with the seam, so pending falls under it.

        This is the whole safety argument in one assertion: prefill running
        in TP because flipping genuinely costs 120s is the system working.
        """
        alarm, detail = _economy(
            live_flip_cost_s=120.0, live_flip_tokens=W3_MAX_PENDING + 1
        )
        self.assertFalse(alarm)
        self.assertIn("correct economics", detail)

    def test_pending_exactly_at_the_bar_is_silent(self):
        alarm, _ = _economy(live_flip_tokens=W3_MAX_PENDING)
        self.assertFalse(alarm)

    def test_a_seeded_price_is_not_evidence(self):
        alarm, detail = _economy(price_measured=False)
        self.assertFalse(alarm)
        self.assertIn("seed, not measured", detail)

    def test_min_dwell_is_a_legitimate_hold(self):
        alarm, detail = _economy(since_flip_s=1.0, min_dwell_s=3.0)
        self.assertFalse(alarm)
        self.assertIn("min dwell", detail)

    def test_active_staging_is_a_legitimate_hold(self):
        alarm, detail = _economy(staging_active=True)
        self.assertFalse(alarm)
        self.assertIn("staging attempt is in flight", detail)

    def test_a_genuinely_draining_bundle_is_a_legitimate_hold(self):
        """Net progress AND recent progress -- #833's own two conditions."""
        alarm, detail = _economy(running_bs=2, bundle_at_phase_entry=7, bundle_stall_s=0.4)
        self.assertFalse(alarm)
        self.assertIn("genuinely draining", detail)

    def test_a_bundle_holding_station_is_not_draining(self):
        """The window-3 shape: 7 of 7, then 5 of 5, then 6 of 6 -- it retires
        work continuously and never converges toward empty."""
        alarm, detail = _economy(running_bs=7, bundle_at_phase_entry=7)
        self.assertTrue(alarm)
        self.assertIn(lc.ILLEGITIMATE_BUNDLE_NOT_DRAINING, detail)

    def test_net_progress_that_has_since_stalled_is_not_draining(self):
        alarm, _ = _economy(
            running_bs=2, bundle_at_phase_entry=7, bundle_stall_s=W3_WINDOW_S + 1.0
        )
        self.assertTrue(alarm)

    def test_a_hold_shorter_than_one_decode_window_is_too_early(self):
        alarm, detail = _economy(held_s=W3_WINDOW_S - 0.1)
        self.assertFalse(alarm)
        self.assertIn("too early to question", detail)

    def test_the_pp_direction_is_declined_and_not_approximated(self):
        """The price line is denominated in prefill tokens, so it states no
        economics for a hold in PP. Saying so beats guessing."""
        alarm, detail = _economy(phase=PHASE_PP)
        self.assertFalse(alarm)
        self.assertIn("no claim is made", detail)


class HardConformanceIsBinary(unittest.TestCase):
    """Class 1. No economics, no thresholds: intent versus execution."""

    def test_a_batch_admitted_in_pp_and_run_in_tp_is_a_violation(self):
        alarm, detail = lc.admit_vs_exec_verdict(
            PHASE_PP, PHASE_TP, "rid-7", 2, "holding in tp"
        )
        self.assertTrue(alarm)
        self.assertIn(lc.ALARM_CONFORMANCE, detail)
        self.assertIn(f"kind={lc.KIND_ADMIT_VS_EXEC}", detail)
        self.assertIn("rid=rid-7", detail)
        self.assertIn("mb_id=2", detail)

    def test_the_inverse_is_equally_a_violation(self):
        alarm, detail = lc.admit_vs_exec_verdict(
            PHASE_TP, PHASE_PP, "rid-1", 0, None
        )
        self.assertTrue(alarm)
        self.assertIn("admitted=tp executed=pp", detail)

    def test_agreement_is_silent(self):
        alarm, _ = lc.admit_vs_exec_verdict(PHASE_TP, PHASE_TP, "r", 0, None)
        self.assertFalse(alarm)

    def test_an_unstamped_batch_is_silence_not_an_alarm(self):
        """The decoupled spill batch does not come through the funnel that
        stamps. Unknown provenance must not be reported as a defect."""
        alarm, detail = lc.admit_vs_exec_verdict(None, PHASE_TP, "r", 0, None)
        self.assertFalse(alarm)
        self.assertIn("no admission stamp", detail)

    def test_a_verdict_computed_against_a_stale_mirror_is_a_violation(self):
        alarm, detail = lc.verdict_vs_routing_verdict(PHASE_PP, PHASE_TP, 1, "held")
        self.assertTrue(alarm)
        self.assertIn(f"kind={lc.KIND_VERDICT_VS_ROUTING}", detail)

    def test_the_mirrors_agreeing_is_silent(self):
        alarm, _ = lc.verdict_vs_routing_verdict(PHASE_TP, PHASE_TP, 1, "held")
        self.assertFalse(alarm)


class TheW12FormIsAConformanceViolation(unittest.TestCase):
    """A COMMIT at zero armed passes carrying a slot from a retired ring.

    boot_window2_0823_1554 @ f9d7637f04: PP0 re-entered on slot 2 of a ring
    that started it on slot 0, and the group died one second later.
    """

    def test_zero_passes_across_a_ring_rebuild_fires(self):
        alarm, detail = lc.stale_ring_restore_verdict(0, 5, 6, 2, "held in tp")
        self.assertTrue(alarm)
        self.assertIn(f"kind={lc.KIND_STALE_RING_RESTORE}", detail)
        self.assertIn("arm_epoch=5 live_epoch=6", detail)

    def test_it_fires_even_when_the_slots_coincidentally_agree(self):
        """This file's own source records the epoch-4 crossing surviving
        "only because all three ranks happened to carry the same retired
        slot -- agreement there was luck". Luck is worth announcing."""
        alarm, _ = lc.stale_ring_restore_verdict(0, 4, 5, 0, None)
        self.assertTrue(alarm)

    def test_zero_passes_within_one_generation_is_an_ordinary_abandon(self):
        alarm, detail = lc.stale_ring_restore_verdict(0, 6, 6, 2, None)
        self.assertFalse(alarm)
        self.assertIn("ordinary abandon", detail)

    def test_a_window_that_ran_passes_is_not_this_form(self):
        alarm, _ = lc.stale_ring_restore_verdict(3, 5, 6, 2, None)
        self.assertFalse(alarm)

    def test_a_missing_epoch_falls_back_to_silence(self):
        alarm, _ = lc.stale_ring_restore_verdict(0, None, 6, 2, None)
        self.assertFalse(alarm)


class CountersAndThrottle(unittest.TestCase):
    def setUp(self):
        lc.reset_for_test()

    def tearDown(self):
        lc.reset_for_test()

    def test_every_occurrence_counts_even_when_the_log_is_throttled(self):
        """The throttle bounds the LOG, never the measurement -- the split
        #631's own policy throttle got right after a log flood cost this
        feature a self-kill."""
        said = [
            lc.note_conformance_violation("kind=admit_vs_exec x", 100.0 + i * 0.1)
            for i in range(5)
        ]
        self.assertEqual(lc.counters().conformance_violations, 5)
        self.assertEqual(said.count(True), 1)

    def test_the_same_shape_is_reannounced_after_the_interval(self):
        lc.note_conformance_violation("kind=admit_vs_exec x", 100.0)
        self.assertTrue(
            lc.note_conformance_violation(
                "kind=admit_vs_exec x", 100.0 + lc.ALARM_REANNOUNCE_S
            )
        )

    def test_different_shapes_do_not_throttle_each_other(self):
        self.assertTrue(lc.note_conformance_violation("kind=admit_vs_exec", 100.0))
        self.assertTrue(
            lc.note_conformance_violation("kind=stale_ring_restore", 100.1)
        )

    def test_the_periodic_field_reports_both_levels(self):
        lc.note_conformance_violation("kind=admit_vs_exec", 100.0)
        lc.note_economy_anomaly("x", 100.0)
        lc.note_economy_anomaly("x", 100.1)
        self.assertEqual(
            lc.counters().as_field(), "layout-conformance (#838): c1=1, c2=2"
        )


class TheAlarmQuotesTheVerdictItSaw(unittest.TestCase):
    """#713: ONE reading, used by the comparison AND by the message.

    The wiring reads the routing flag once and hands it down. A second read
    could disagree with the first, which would make the alarm line quote a
    phase the comparison never made -- self-refuting evidence.
    """

    def _holder(self, calls, phase_seq):
        from sglang.srt.managers.scheduler import Scheduler

        batch = types.SimpleNamespace(
            reqs=[types.SimpleNamespace(rid="rid-9")],
            _layout_admitted_phase=PHASE_PP,
        )

        def read_phase():
            calls.append(1)
            return phase_seq[min(len(calls) - 1, len(phase_seq) - 1)]

        holder = types.SimpleNamespace(
            _layout_routing_phase=read_phase,
            _pp_live_mb_id=4,
            phase_policy_state=types.SimpleNamespace(last_reason="holding in tp"),
        )
        return Scheduler._check_layout_conformance, holder, batch

    def setUp(self):
        lc.reset_for_test()

    def tearDown(self):
        lc.reset_for_test()

    def test_the_routing_flag_is_read_exactly_once_per_check(self):
        calls = []
        check, holder, batch = self._holder(calls, [PHASE_TP])
        check(holder, batch)
        self.assertEqual(len(calls), 1)
        self.assertEqual(lc.counters().conformance_violations, 1)

    def test_a_conformant_batch_is_still_only_read_once(self):
        calls = []
        check, holder, batch = self._holder(calls, [PHASE_PP])
        check(holder, batch)
        self.assertEqual(len(calls), 1)
        self.assertEqual(lc.counters().conformance_violations, 0)

    def test_a_batch_with_no_stamp_reaches_no_alarm_through_the_wiring(self):
        """The decoupled spill batch: it never passes the funnel that stamps,
        so `run_batch` must judge nothing about it."""
        calls = []
        check, holder, batch = self._holder(calls, [PHASE_TP])
        del batch._layout_admitted_phase
        check(holder, batch)
        self.assertEqual(len(calls), 1)
        self.assertEqual(lc.counters().conformance_violations, 0)


# ---------------------------------------------------------------------------
# MUTANTS. Each block mutates the detector and asserts that an assertion this
# file already makes FLIPS. A test that cannot fail is not evidence, and this
# section is what makes the ones above evidence.
# ---------------------------------------------------------------------------


class MutantsDie(unittest.TestCase):
    def setUp(self):
        lc.reset_for_test()

    def tearDown(self):
        lc.reset_for_test()

    def test_mutant_blind_detector_dies(self):
        """Remove the comparison (always conformant). The class-1 assertions
        must stop holding."""

        def blind(admitted, executing, rid, mb_id, reason):
            return False, "conformant"

        alarm, _ = blind(PHASE_PP, PHASE_TP, "rid-7", 2, None)
        with self.assertRaises(AssertionError):
            self.assertTrue(alarm)

    def test_mutant_blind_economy_dies(self):
        """Drop the price comparison and always hold. The specimen stops
        firing, so `TheWindow3SpanIsAnAnomaly.test_it_fires` goes red."""

        def blind(**_):
            return False, "held"

        alarm, _ = blind()
        with self.assertRaises(AssertionError):
            self.assertTrue(alarm)

    def test_mutant_overeager_detector_dies(self):
        """Alarm on any TP hold longer than the window, ignoring the price.

        This is the mutant the whole design exists to exclude: it reports
        correct economics as a defect. `HonestEconomicsMustNotAlarm.
        test_an_honestly_expensive_measured_seam_is_silent` kills it.
        """

        def overeager(*, phase, held_s, window_s, **_):
            return (phase == PHASE_TP and held_s >= window_s), "alarm"

        alarm, _ = overeager(
            phase=PHASE_TP, held_s=41.0, window_s=W3_WINDOW_S
        )
        with self.assertRaises(AssertionError):
            self.assertFalse(alarm)

    def test_mutant_that_alarms_on_a_seeded_price_dies(self):
        def seed_blind(**kw):
            kw["price_measured"] = True
            return lc.economy_divergence_verdict(**kw)

        kw = dict(
            phase=PHASE_TP,
            held_s=41.0,
            window_s=W3_WINDOW_S,
            pending_prefill_tokens=W3_MAX_PENDING,
            live_flip_tokens=W3_BAR,
            live_flip_cost_s=W3_SEAM_S,
            price_measured=False,
            hold_reason="x",
            since_flip_s=30.0,
            min_dwell_s=3.0,
            staging_active=False,
            running_bs=W3_BUNDLE,
            bundle_at_phase_entry=W3_BUNDLE,
            bundle_stall_s=0.4,
        )
        alarm, _ = seed_blind(**kw)
        with self.assertRaises(AssertionError):
            self.assertFalse(alarm)

    def test_mutant_that_inherits_the_699_running_guard_dies(self):
        """#699 returns "not wedged" at any running > 0. A class-2 detector
        that copied that guard would be silent on the entire specimen."""

        def with_699_guard(**kw):
            if kw["running_bs"] > 0:
                return False, "the box is serving"
            return lc.economy_divergence_verdict(**kw)

        kw = dict(
            phase=PHASE_TP,
            held_s=41.0,
            window_s=W3_WINDOW_S,
            pending_prefill_tokens=W3_MAX_PENDING,
            live_flip_tokens=W3_BAR,
            live_flip_cost_s=W3_SEAM_S,
            price_measured=True,
            hold_reason="x",
            since_flip_s=30.0,
            min_dwell_s=3.0,
            staging_active=False,
            running_bs=W3_BUNDLE,
            bundle_at_phase_entry=W3_BUNDLE,
            bundle_stall_s=0.4,
        )
        alarm, _ = with_699_guard(**kw)
        with self.assertRaises(AssertionError):
            self.assertTrue(alarm)

    def test_mutant_second_reading_dies(self):
        """#713. A wiring that reads the routing flag a second time to build
        the message. `TheAlarmQuotesTheVerdictItSaw` counts the reads, so the
        call-count assertion flips.
        """
        calls = []

        def read_phase():
            calls.append(1)
            return PHASE_TP

        # The mutation: verdict from one read, message from another.
        read_phase()
        read_phase()
        with self.assertRaises(AssertionError):
            self.assertEqual(len(calls), 1)

    def test_mutant_that_counts_only_logged_alarms_dies(self):
        """Increment the counter inside the throttle instead of outside, so
        a burst of five reports one. `CountersAndThrottle.
        test_every_occurrence_counts_even_when_the_log_is_throttled` kills
        it."""
        counted = 0
        last = None
        for i in range(5):
            now = 100.0 + i * 0.1
            if last is None or now - last >= lc.ALARM_REANNOUNCE_S:
                last = now
                counted += 1
        with self.assertRaises(AssertionError):
            self.assertEqual(counted, 5)


if __name__ == "__main__":
    unittest.main()
