"""#677: demand decides the layout, not the timer -- in BOTH directions.

MEASURED PROBLEM (#713 quantisation table, 2026-08-17 06:19). Demand-PULL
already worked: C1 arrived 06:19:06.56 and was served by the tp_to_pp at :09
-- 3.14 s, one seam plus overhead. What failed was the step AFTER: the box
flipped back at :12, so C2, which arrived 06:19:09.71 just as PP began, was not
served until :15.

    arm  arrive        first token   TTFT   served by
    C1   06:19:06.56   06:19:09.71   3.14   tp_to_pp @:09
    C2   06:19:09.71   06:19:15.52   5.81   MISSED :12, waited tp_to_pp @:15
    C3   06:19:15.52   06:19:15.63   0.11   landed ON the :15 flip

The layout left on a timer while the prefill that pulled it was still unserved,
so TTFT quantised to whole cycles: 0.1 / 3.1 / 5.9 s, nothing between. C2 is the
case this lever exists to fix.

BOUNDED BOTH WAYS ON PURPOSE. An unbounded hold is a starvation bug wearing a
fix's clothes -- it would simply trade the prefill side's starvation for the
decode side's.
"""

import unittest

from sglang.srt.managers.phase_policy import (
    LAYOUT_HOLD_MAX_ROUNDS,
    layout_hold_verdict,
)
from sglang.test.test_utils import CustomTestCase


class TestLayoutHold677(CustomTestCase):
    def test_C2_case_pp_holds_while_prefill_unserved(self):
        """THE SPECIMEN. In PP with prefill still unserved, pp_to_tp must NOT
        arm -- C2 arrived just as PP began and lost the layout at :12."""
        allow, why = layout_hold_verdict(
            "pp", prefill_pending_tokens=22, decode_waiting=0
        )
        self.assertFalse(allow, why)
        self.assertIn("HOLD", why)
        self.assertIn("22", why, "the measurement must be quoted")

    def test_pp_releases_when_nothing_is_pending(self):
        """CAN-FAIL: the hold must not become a permanent stay. With no prefill
        pending the timer legitimately gets the layout back."""
        allow, why = layout_hold_verdict("pp", 0, 0)
        self.assertTrue(allow, why)
        self.assertIn("timer may have the layout", why)

    def test_tp_pulls_for_waiting_prefill(self):
        """The other direction of the same rule: work waiting for PP pulls the
        cutover forward rather than waiting out the cycle."""
        allow, why = layout_hold_verdict("tp", 22, 0)
        self.assertTrue(allow, why)
        self.assertIn("PULL", why)

    def test_tp_stays_when_no_prefill_waits(self):
        allow, why = layout_hold_verdict("tp", 0, 3)
        self.assertFalse(allow, why)
        self.assertIn("nothing pulls", why)

    # ---- safety precedence -------------------------------------------------

    def test_never_decides_mid_flip(self):
        """A cutover in progress owns the layout, whatever demand says."""
        for phase in ("pp", "tp"):
            for pend in (0, 5000):
                with self.subTest(phase=phase, pending=pend):
                    allow, why = layout_hold_verdict(phase, pend, 4, mid_flip=True)
                    self.assertFalse(allow)
                    self.assertIn("cutover is in progress", why)

    def test_never_pulls_against_an_unfunded_seam(self):
        """A pull that cannot pay is an abandon, which is worse than waiting."""
        allow, why = layout_hold_verdict("tp", 99999, 0, seam_funded=False)
        self.assertFalse(allow, why)
        self.assertIn("unfunded", why)

    def test_unfunded_outranks_demand_in_both_phases(self):
        for phase in ("pp", "tp"):
            with self.subTest(phase=phase):
                allow, _ = layout_hold_verdict(phase, 5000, 5, seam_funded=False)
                self.assertFalse(allow)

    # ---- the both-sides tie, and BOTH starvation directions -----------------

    def test_both_sides_holds_but_says_it_is_a_tie(self):
        allow, why = layout_hold_verdict("pp", 22, 4)
        self.assertFalse(allow, why)
        self.assertIn("BOTH SIDES", why)
        self.assertIn("economic", why)

    def test_hold_cannot_starve_decode_forever(self):
        """STARVATION DIRECTION (a). The hold is bounded: past the bound the
        layout releases even with prefill still unserved, so a decode queue
        cannot be held out indefinitely."""
        allow, why = layout_hold_verdict(
            "pp", 22, 4, hold_rounds_so_far=LAYOUT_HOLD_MAX_ROUNDS
        )
        self.assertTrue(allow, why)
        self.assertIn("EXHAUSTED", why)

    def test_the_bound_is_reached_only_at_the_limit(self):
        """CAN-FAIL for the bound: it must not release early, or the C2 fix
        evaporates the moment any decode appears."""
        for n in range(LAYOUT_HOLD_MAX_ROUNDS):
            with self.subTest(round=n):
                allow, _ = layout_hold_verdict("pp", 22, 4, hold_rounds_so_far=n)
                self.assertFalse(allow, f"must still hold at round {n}")

    def test_pull_cannot_preempt_an_unserved_prefill(self):
        """STARVATION DIRECTION (b). While a hold is live in PP, the verdict is
        a hold -- the decode side cannot pull the layout out from under a
        prefill that has not been served."""
        allow, _ = layout_hold_verdict("pp", 22, 10_000)
        self.assertFalse(allow)

    def test_round_counter_is_reported_so_the_bound_is_auditable(self):
        _, why = layout_hold_verdict("pp", 22, 0, hold_rounds_so_far=3)
        self.assertIn(f"{LAYOUT_HOLD_MAX_ROUNDS}", why)
        self.assertIn("4", why, "round n+1 of N must be visible")

    def test_unknown_phase_makes_no_decision(self):
        allow, why = layout_hold_verdict("flip_in_progress", 22, 0)
        self.assertFalse(allow)
        self.assertIn("no decision", why)

    def test_negative_and_degenerate_inputs_do_not_crash(self):
        for args in (("pp", -5, -5), ("tp", -1, 0), ("pp", 0, -3)):
            with self.subTest(args=args):
                allow, why = layout_hold_verdict(*args)
                self.assertIsInstance(allow, bool)
                self.assertTrue(why)


if __name__ == "__main__":
    unittest.main()


class TestTpMirrorAndCounterLifecycle677(CustomTestCase):
    """Review findings on the rule/decide() seam, both pinned here.

    FINDING 1, THE MIRROR. An unconditional pull on pend>0 recreates the defect
    this lever fixes, phases swapped: after an EXHAUSTED release (pp->tp with
    prefill still pending) the next TP evaluation sees pend>0 and pulls straight
    back, so decode loses the layout before serving anything. Under sustained
    both-sides load that degenerates to max_hold PP rounds, ~0 TP rounds, and
    TWO seams per cycle -- strictly worse than the timer it replaced.

    FINDING 2, THE COUNTER. hold_rounds_so_far is caller-maintained; a stale
    counter makes the first hold of the next episode start half-exhausted.
    """

    def test_pull_does_not_preempt_a_running_decode_batch(self):
        allow, why = layout_hold_verdict(
            "tp", 22, 4, decode_serving=True, decode_rounds_so_far=0
        )
        self.assertFalse(allow, why)
        self.assertIn("preempt mid-batch", why)

    def test_pull_proceeds_once_the_decode_minimum_is_met(self):
        """CAN-FAIL for the mirror: it must be a BOUNDED yield, not a new
        starvation of the prefill side."""
        from sglang.srt.managers.phase_policy import MIN_DECODE_ROUNDS

        allow, why = layout_hold_verdict(
            "tp", 22, 4, decode_serving=True, decode_rounds_so_far=MIN_DECODE_ROUNDS
        )
        self.assertTrue(allow, why)
        self.assertIn("minimum", why)

    def test_no_running_decode_means_no_yield(self):
        """The mirror guards a RUNNING batch, not an idle TP layout."""
        allow, _ = layout_hold_verdict("tp", 22, 0, decode_serving=False)
        self.assertTrue(allow)

    def test_the_exhausted_release_cannot_ping_pong_straight_back(self):
        """THE DEGENERATE CYCLE, pinned. Release from PP under both-sides load,
        then evaluate in TP with a decode batch now running: the pull must
        YIELD, so the cycle cannot become 8 PP rounds + 0 TP rounds."""
        allow_release, _ = layout_hold_verdict(
            "pp", 22, 4, hold_rounds_so_far=LAYOUT_HOLD_MAX_ROUNDS
        )
        self.assertTrue(allow_release, "pp must release at the bound")
        allow_back, why = layout_hold_verdict(
            "tp", 22, 4, decode_serving=True, decode_rounds_so_far=0
        )
        self.assertFalse(allow_back, f"must not pull straight back: {why}")

    def test_counter_resets_on_phase_change(self):
        from sglang.srt.managers.phase_policy import next_hold_rounds

        self.assertEqual(next_hold_rounds(7, "pp", "tp", 22), 0)
        self.assertEqual(next_hold_rounds(7, "tp", "pp", 22), 0)

    def test_counter_resets_when_nothing_pends(self):
        from sglang.srt.managers.phase_policy import next_hold_rounds

        self.assertEqual(next_hold_rounds(7, "pp", "pp", 0), 0)

    def test_counter_advances_within_an_episode(self):
        """CAN-FAIL: a counter that always reset would make the hold unbounded,
        which is the starvation direction the bound exists to prevent."""
        from sglang.srt.managers.phase_policy import next_hold_rounds

        n = 0
        for _ in range(5):
            n = next_hold_rounds(n, "pp", "pp", 22)
        self.assertEqual(n, 5)

    def test_a_stale_counter_would_start_the_next_episode_exhausted(self):
        """The defect the reset prevents, stated as a test: carrying 8 across a
        phase change would release the very first hold of the next episode."""
        from sglang.srt.managers.phase_policy import next_hold_rounds

        stale = LAYOUT_HOLD_MAX_ROUNDS
        self.assertTrue(layout_hold_verdict("pp", 22, 0, hold_rounds_so_far=stale)[0])
        fresh = next_hold_rounds(stale, "tp", "pp", 22)
        self.assertFalse(layout_hold_verdict("pp", 22, 0, hold_rounds_so_far=fresh)[0])


class TestHoldCounterActuallyAdvances677(CustomTestCase):
    """RED-FIRST on 332cb3b345: the bound was documented but never advanced.

    layout_hold_verdict took hold_rounds_so_far from the caller and
    next_hold_rounds computed the successor, but NOTHING incremented the state
    -- so every round evaluated at 0 and the 8-round bound could never be
    reached. A documented invariant with no wiring is the #505/#421 class this
    tree has hunted twice this week; shipping it inert on purpose would be
    worse than not having the bound at all.

    These tests drive the real per-round observer, so they fail unless the
    counter is genuinely maintained.
    """

    def _round(self, state, phase, pending, running, now):
        from sglang.srt.managers.phase_policy import PhasePolicyInputs, observe_idle

        inp = PhasePolicyInputs(
            phase=phase,
            pending_prefill_tokens=pending,
            running_bs=running,
            now=now,
        )
        observe_idle(state, inp)
        return inp

    def test_hold_rounds_reaches_N_under_a_live_hold(self):
        from sglang.srt.managers.phase_policy import PhasePolicyState

        st = PhasePolicyState()
        for i in range(5):
            self._round(st, "pp", 22, 0, 1000.0 + i)
        self.assertEqual(
            getattr(st, "hold_rounds", 0),
            5,
            "the counter must advance once per round while the hold is live",
        )

    def test_exhausted_is_reachable_by_running_the_bound(self):
        """The EXHAUSTED path must be REACHABLE, not merely written."""
        from sglang.srt.managers.phase_policy import PhasePolicyState

        st = PhasePolicyState()
        for i in range(LAYOUT_HOLD_MAX_ROUNDS):
            self._round(st, "pp", 22, 0, 1000.0 + i)
        allow, why = layout_hold_verdict(
            "pp", 22, 4, hold_rounds_so_far=getattr(st, "hold_rounds", 0)
        )
        self.assertTrue(allow, f"the bound must release: {why}")
        self.assertIn("EXHAUSTED", why)

    def test_counter_resets_on_a_real_phase_change(self):
        from sglang.srt.managers.phase_policy import PhasePolicyState

        st = PhasePolicyState()
        for i in range(4):
            self._round(st, "pp", 22, 0, 1000.0 + i)
        self.assertGreater(getattr(st, "hold_rounds", 0), 0)
        self._round(st, "tp", 22, 0, 1010.0)
        self.assertEqual(getattr(st, "hold_rounds", 0), 0, "phase change resets")

    def test_counter_resets_when_the_work_drains(self):
        from sglang.srt.managers.phase_policy import PhasePolicyState

        st = PhasePolicyState()
        for i in range(4):
            self._round(st, "pp", 22, 0, 1000.0 + i)
        self._round(st, "pp", 0, 0, 1010.0)
        self.assertEqual(getattr(st, "hold_rounds", 0), 0, "pend->0 resets")
