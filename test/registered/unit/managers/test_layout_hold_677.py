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
