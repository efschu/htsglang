# SPDX-License-Identifier: Apache-2.0
"""#1378 xsn39: W17's group-dead gate learns the flip phase.

MEASURED (three boots in a row): the flip legs RUN in the groups' event
loops, /health goes legitimately silent, and W17 Weg2GroupDead killed the
boot ~2 min into EVERY ring-off flip (xsn36/37/39) -- with all lanes
healthy and no W-code from the legs. The gate now asks ONE named function
(``Front.group_dead_should_stop``): during ``flipping`` the streak is
logged, never a stop -- the flip's own stall detector (flip_stall, bound
120 s) is the authority for a stuck flip.

MUTANT (danger direction): a gate that ignores the flip state (or stops on
streak < 2) dies on the truth table below.
"""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers.scheduler_components import weight_updater as wu  # noqa: E402
from sglang.srt.weg2 import front as front_mod  # noqa: E402


class TheGroupDeadGateLearnsTheFlipPhase(unittest.TestCase):
    def setUp(self):
        self.f = object.__new__(front_mod.Front)

    def test_flipping_is_never_a_group_death(self):
        """THE WALL: /health silent + the process alive + streak 2 -- the
        exact signature of every ring-off flip's first minutes. During
        ``flipping`` this is the leg blocking the loop BY DESIGN."""
        self.assertFalse(self.f.group_dead_should_stop(
            state="flipping", ok=False, alive=True, streak=2))
        self.assertFalse(self.f.group_dead_should_stop(
            state="flipping", ok=False, alive=True, streak=9))

    def test_idle_group_death_still_stops(self):
        """The stop stays armed off the flip: a dead group at idle is a
        fact, not a phase."""
        self.assertTrue(self.f.group_dead_should_stop(
            state="serving", ok=False, alive=True, streak=2))

    def test_short_streak_never_stops(self):
        self.assertFalse(self.f.group_dead_should_stop(
            state="serving", ok=False, alive=True, streak=0))
        self.assertFalse(self.f.group_dead_should_stop(
            state="serving", ok=False, alive=True, streak=1))

    def test_a_200_with_a_live_process_is_never_a_death(self):
        """The serving-fact rule: a 200 plus a live process is not a death,
        in any state."""
        for state in ("flipping", "serving"):
            self.assertFalse(self.f.group_dead_should_stop(
                state=state, ok=True, alive=True, streak=5))

    def test_M_the_gate_ignoring_the_flip_state_dies(self):
        """MUTANT (danger direction): a gate that stops on the streak alone
        (the pre-fix shape that killed three healthy boots) must fail the
        truth table's flipping row."""
        def mutant_gate(*, state, ok, alive, streak):
            if streak < 2:
                return False
            if front_mod.health_is_serving_fact(ok, alive):
                return False
            return True  # the flip state is IGNORED -- the mutant

        with self.assertRaises(AssertionError):
            self.assertFalse(mutant_gate(
                state="flipping", ok=False, alive=True, streak=2),
                "the mutant stops a healthy flip -- the pin's truth table "
                "catches it")


class TheGateIsWiredInThePoller(unittest.TestCase):
    def test_the_poller_asks_the_named_gate(self):
        """ONE decision, one caller: the poller must consult
        ``group_dead_should_stop`` instead of an inline streak check -- a
        second inline copy would be the two-readings defect."""
        src = __import__("inspect").getsource(front_mod.Front.health_poller)
        self.assertIn("group_dead_should_stop(", src)
        self.assertNotIn("health_fail_streak >= 2 and not health_is_serving_fact",
                         src,
                         "the inline streak check must be gone from the "
                         "poller (one authority)")


if __name__ == "__main__":
    unittest.main()
