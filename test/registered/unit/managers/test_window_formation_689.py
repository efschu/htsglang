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
"""#689: a decode window must open at width, not at one.

THE DEFECT, MEASURED. Decode ran bs=1 on 261 of 288 batches and bs=4 on only
12, while 13 requests sat queued, token usage was 0.44 and the mamba pool held
2 of its 12 slots. Nothing was throttling ADMISSION -- the pools were nearly
empty. What was wrong was WINDOW FORMATION: under pure drain a TP window can
only decode the carriers whose prefill completed in the PRECEDING PP window,
and the flip was fired by stall caps rather than by readiness, so a window
opened with one ready carrier and then served bs=1 for its whole length while
twelve requests waited for the next window.

THE BOUND IS A PLAIN TIME CAP AND IS LABELLED AS ONE. #677's window economics
is separate work; this is the bounded first cut that makes bs=4 real. The cap
is taken from this rig's own receipts -- one flip EXECUTES in 2459-3186 ms
(five FLIP DONE receipts, 2026-08-16) -- so waiting up to about one flip's
worth to quadruple the window that flip opens is plainly worth it. That is a
bound, not an optimum.

WHAT MUST NOT REGRESS, and it is the reason these two features share a file in
spirit: #688's idle-locked arm fires when NOTHING can run, and it is returned
before any load rule. Formation may only ever hold an arm while the PP layout
is still doing work. A formation gate that could hold the idle-locked arm
would reintroduce the very zero-GPU window #688 exists to remove.
"""

import unittest

from sglang.srt.managers.phase_policy import (
    PHASE_PP,
    PHASE_TP,
    PP_TO_TP,
    REST_PREFILL,
    TP_TO_PP,
    PhasePolicyConfig,
    PhasePolicyInputs,
    PhasePolicyState,
    decide,
    observe_idle,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5)

N = 30000
TARGET = 4


def cfg(**kw):
    base = dict(
        enabled=True,
        flip_tokens=N,
        min_dwell_s=0.0,
        idle_dwell_s=20.0,
        rest_state=REST_PREFILL,
        formation_target=TARGET,
        formation_cap_s=3.0,
    )
    base.update(kw)
    return PhasePolicyConfig(**base)


def inputs(**kw):
    """A PP window that HAS drained its prefill, so the rules want to flip."""
    base = dict(
        phase=PHASE_PP,
        pending_prefill_tokens=0,
        running_bs=1,
        now=100.0,
        ready_carriers=1,
        queue_nonempty=True,
    )
    base.update(kw)
    return PhasePolicyInputs(**base)


def armed_state(started=99.0):
    return PhasePolicyState(last_flip_at=0.0, formation_started_at=started)


class TheWindowWaitsForWidth(unittest.TestCase):
    def test_one_carrier_with_a_queue_behind_it_does_not_open_the_window(self):
        d = decide(cfg(), armed_state(), inputs())
        self.assertIsNone(d.direction)
        self.assertIn("forming decode window", d.reason)

    def test_a_full_window_opens(self):
        d = decide(cfg(), armed_state(), inputs(ready_carriers=TARGET))
        self.assertEqual(PP_TO_TP, d.direction)

    def test_over_full_opens_too(self):
        d = decide(cfg(), armed_state(), inputs(ready_carriers=TARGET + 3))
        self.assertEqual(PP_TO_TP, d.direction)


class TheWaitIsBounded(unittest.TestCase):
    def test_an_empty_queue_opens_immediately(self):
        """Nothing can arrive to widen it, so waiting is pure latency."""
        d = decide(cfg(), armed_state(), inputs(queue_nonempty=False))
        self.assertEqual(PP_TO_TP, d.direction)

    def test_the_time_cap_opens_a_short_window(self):
        d = decide(cfg(formation_cap_s=3.0), armed_state(99.0), inputs(now=103.0))
        self.assertEqual(PP_TO_TP, d.direction)
        self.assertIn("formation cap", d.reason)

    def test_below_the_cap_it_still_waits(self):
        d = decide(cfg(formation_cap_s=3.0), armed_state(99.0), inputs(now=101.0))
        self.assertIsNone(d.direction)


class ItNeverHoldsTheIdleLockedArm(unittest.TestCase):
    """#688 must outrank #689. A layout that can run NOTHING leaves at once."""

    def test_the_deadlock_escape_is_not_delayed_by_formation(self):
        d = decide(
            cfg(),
            armed_state(),
            inputs(
                ready_carriers=1,
                queue_nonempty=True,
                nothing_can_run=True,
                target_can_admit=True,
            ),
        )
        self.assertEqual(PP_TO_TP, d.direction)
        self.assertIn("IDLE-LOCKED", d.reason)


class ItOnlyTouchesTheTpWardArm(unittest.TestCase):
    def test_the_return_leg_is_untouched(self):
        d = decide(
            cfg(),
            PhasePolicyState(last_flip_at=0.0),
            inputs(phase=PHASE_TP, pending_prefill_tokens=10 * N, running_bs=0),
        )
        self.assertNotIn("forming decode window", d.reason or "")
        self.assertIn(d.direction, (TP_TO_PP, None))


class TheGateIsOffByDefault(unittest.TestCase):
    def test_target_zero_restores_previous_behaviour(self):
        d = decide(cfg(formation_target=0), armed_state(), inputs())
        self.assertEqual(PP_TO_TP, d.direction)

    def test_target_one_is_also_off(self):
        """A window of one IS the old behaviour; do not spend a round on it."""
        d = decide(cfg(formation_target=1), armed_state(), inputs())
        self.assertEqual(PP_TO_TP, d.direction)


class TheFormationClockIsDrivenByObservation(unittest.TestCase):
    """``decide`` is pure, so the clock lives in ``observe_idle``."""

    def test_it_starts_when_a_window_begins_filling(self):
        st = PhasePolicyState()
        observe_idle(st, inputs(now=50.0))
        self.assertEqual(50.0, st.formation_started_at)

    def test_it_does_not_restart_while_still_filling(self):
        st = PhasePolicyState()
        observe_idle(st, inputs(now=50.0))
        observe_idle(st, inputs(now=51.0, ready_carriers=2))
        self.assertEqual(50.0, st.formation_started_at)

    def test_it_clears_when_the_queue_drains(self):
        st = PhasePolicyState()
        observe_idle(st, inputs(now=50.0))
        observe_idle(st, inputs(now=51.0, queue_nonempty=False))
        self.assertEqual(0.0, st.formation_started_at)

    def test_it_clears_in_the_tp_phase(self):
        st = PhasePolicyState()
        observe_idle(st, inputs(now=50.0))
        observe_idle(st, inputs(now=51.0, phase=PHASE_TP))
        self.assertEqual(0.0, st.formation_started_at)


if __name__ == "__main__":
    unittest.main()
