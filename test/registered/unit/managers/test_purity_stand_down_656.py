# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""#656 C22, the recovery half: A CONTROLLER THAT CANNOT FLIP MUST STILL SERVE.

Measured on metal, boot ``boot_m1`` 2026-08-13 14:46-14:52Z. The pre-move
frame ballot did its job -- it caught a real divergence, abandoned unanimously
and kept every rank alive -- and the instance then stopped emitting tokens
anyway::

    ballot refuses pp_to_tp, 8 times, same divergence every round
    process alive on all three ranks, still logging, every request's KV intact
    /health 503, "couldn't get a response from detokenizer"
    no tokens

The chain is short and entirely by design, which is why it needed a design
answer rather than a bug fix:

1. strict phase purity forbids decode in the PP layout
   (``PhasePurity.decode_allowed_in_pp`` returns False);
2. so decode work waits for a ``pp_to_tp`` flip;
3. the flip is refused every round;
4. nothing reaches the detokenizer, ``last_receive_tstamp`` stops advancing,
   and ``/health`` times out at 503 with every stack idle.

**The ballot converted an instance-fatal crash into an instance wedge, and a
wedge is not an acceptable terminal state.** Zero tolerance forbids it as much
as it forbids a crash. So when the flip is unavailable -- either stood down
for good by the seam-abandon cap, or abandoning the needed direction round
after round -- the purity prohibition on the work class that CANNOT DRAIN is
lifted, loudly, until the flip goes through again.

WHY THAT IS SAFE. Purity is a THROUGHPUT rule, not a correctness one. Its own
module says so: the modes ``threshold:<n>`` and ``off`` run decode in the PP
layout as a supported configuration, and the cost of doing so is named as
latency and throughput ("one decode step in the slow layout"), never wrong
answers. Serving degraded is therefore a strictly better terminal state than
serving nothing, and it is the one the operator can see in the log.

IT IS ALSO GROUP-UNIFORM, which the purity gate requires: both inputs are
already reduced quantities. ``_seam_abandons_in_a_row`` is booked from the
reduced verdict ("all three ranks increment together"), and the blocking
guard is installed on every rank from the same unanimous abandon.

Hermetic: plain objects, no CUDA, no distributed.
"""

from __future__ import annotations

import unittest

from sglang.srt.managers import phase_purity
from sglang.srt.managers.phase_policy import PHASE_PP, PHASE_TP, PP_TO_TP, TP_TO_PP


class _Args:
    def __init__(self, purity="strict"):
        self.enable_phase_flip = True
        self.phase_flip_purity = purity


class _Runtime:
    def __init__(self, guards=(), abandons=None):
        self.blocking_guards = tuple(guards)
        self._seam_abandons_in_a_row = dict(abandons or {})


class _PolicyState:
    def __init__(self, refusals=None):
        self.arm_refusals = dict(refusals or {})


class _Sched:
    def __init__(
        self, phase, *, purity="strict", guards=(), abandons=None, refusals=None
    ):
        self.server_args = _Args(purity)
        self.phase_flip_active_stack = phase
        self.phase_flip_runtime = _Runtime(guards, abandons)
        self.phase_policy_state = _PolicyState(refusals)


class ThePurityRuleStandsWhileTheFlipWorksTest(unittest.TestCase):
    """The default path must be bit-for-bit what it was."""

    def test_decode_is_blocked_in_pp_when_the_flip_is_healthy(self):
        self.assertTrue(phase_purity.decode_blocked_here(_Sched(PHASE_PP), 1))

    def test_prefill_is_blocked_in_tp_when_the_flip_is_healthy(self):
        self.assertTrue(phase_purity.prefill_blocked_here(_Sched(PHASE_TP)))

    def test_a_short_abandon_streak_is_not_a_stand_down(self):
        """One refused round is a retry, not a verdict."""
        sched = _Sched(PHASE_PP, abandons={PP_TO_TP: 1})
        self.assertTrue(phase_purity.decode_blocked_here(sched, 1))


class AStoodDownFlipMustNotStarveTheWorkClassTest(unittest.TestCase):
    """The wedge, closed: serving degraded beats serving nothing."""

    def test_the_seam_cap_verdict_lets_decode_run_in_pp(self):
        sched = _Sched(PHASE_PP, guards=("seam unfundable: pp_to_tp x8",))
        self.assertFalse(
            phase_purity.decode_blocked_here(sched, 1),
            "a flip that can never arm again must not hold decode for ever",
        )

    def test_the_seam_cap_verdict_lets_prefill_run_in_tp(self):
        sched = _Sched(PHASE_TP, guards=("seam unfundable: tp_to_pp x8",))
        self.assertFalse(phase_purity.prefill_blocked_here(sched))

    def test_a_persistent_divergence_in_decodes_leg_lets_decode_run(self):
        """The metal case: the ballot refuses pp_to_tp round after round."""
        sched = _Sched(PHASE_PP, abandons={PP_TO_TP: 8})
        self.assertFalse(phase_purity.decode_blocked_here(sched, 1))

    def test_the_backoff_cannot_hold_the_valve_shut(self):
        """Metal, 2026-08-13 15:40-15:44Z, and it is why one counter is not enough.

        Three group abandons of tp_to_pp armed the seam BACKOFF, which then
        declined the next arm requests without entering the seam at all. The
        abandon counter froze at 3 -- below any bound -- while the policy
        logged "tp_to_pp arm refused (7 in a row)" on all three ranks and a
        9-token prefill sat unrunnable in the TP layout for four minutes.
        """
        sched = _Sched(PHASE_TP, abandons={TP_TO_PP: 3}, refusals={TP_TO_PP: 7})
        self.assertFalse(
            phase_purity.prefill_blocked_here(sched),
            "the damping layer must not be able to hold the valve shut",
        )

    def test_the_policy_streak_is_keyed_on_the_direction_too(self):
        sched = _Sched(PHASE_PP, refusals={TP_TO_PP: 99})
        self.assertTrue(phase_purity.decode_blocked_here(sched, 1))

    def test_the_relaxation_is_keyed_on_the_direction_that_is_stuck(self):
        """A stuck tp_to_pp says nothing about decode's leg.

        Relaxing on the wrong direction would give up the purity rule for a
        failure that is not starving anything, which is how a safety valve
        becomes the normal path.
        """
        sched = _Sched(PHASE_PP, abandons={TP_TO_PP: 99})
        self.assertTrue(phase_purity.decode_blocked_here(sched, 1))

    def test_it_clears_when_the_flip_goes_through_again(self):
        sched = _Sched(PHASE_PP, abandons={PP_TO_TP: 8})
        self.assertFalse(phase_purity.decode_blocked_here(sched, 1))
        # A committed cutover resets the streak (phase_flip_runtime does this
        # on the reduced verdict, so every rank clears together).
        sched.phase_flip_runtime._seam_abandons_in_a_row[PP_TO_TP] = 0
        self.assertTrue(
            phase_purity.decode_blocked_here(sched, 1),
            "the relaxation is a valve, not a mode change",
        )

    def test_the_threshold_is_inside_the_health_check_timeout(self):
        """It must fire before /health does, or it has changed nothing.

        The health probe times out at 20 s (SGLANG_HEALTH_CHECK_TIMEOUT) and a
        refused round costs about 3 s on this rig, so the bound has to be
        small enough that the valve opens first.
        """
        self.assertLessEqual(phase_purity.stand_down_after(), 6)


class ARelaxationMustBeVisibleTest(unittest.TestCase):
    """A silent purity relaxation is the failure this module forbids."""

    def test_the_first_relaxation_is_logged_once_and_loudly(self):
        sched = _Sched(PHASE_PP, abandons={PP_TO_TP: 8})
        with self.assertLogs(phase_purity.logger, level="WARNING") as caught:
            phase_purity.decode_blocked_here(sched, 1)
            phase_purity.decode_blocked_here(sched, 1)
            phase_purity.decode_blocked_here(sched, 1)
        said = [r for r in caught.output if "STOOD DOWN" in r]
        self.assertEqual(len(said), 1, "once per stand-down, not per round")
        self.assertIn("decode", said[0])


class TheGateStillIgnoresEverythingWhenTheFlipIsOffTest(unittest.TestCase):
    def test_no_flip_no_gate(self):
        sched = _Sched(PHASE_PP, abandons={PP_TO_TP: 8})
        sched.server_args.enable_phase_flip = False
        self.assertFalse(phase_purity.decode_blocked_here(sched, 1))
        self.assertFalse(phase_purity.prefill_blocked_here(sched))


if __name__ == "__main__":
    unittest.main()
