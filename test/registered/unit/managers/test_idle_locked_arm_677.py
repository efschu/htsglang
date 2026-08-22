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
"""#677: leaving a layout that can run NOTHING, without a timer.

THE SPECIMEN. Metal, 2026-08-16 09:42:39-45, six seconds in which the log
produced not one line on any rank:

    #running-req 0   #queue-req 10   #pending-token 572715
    full token usage 0.77   mamba usage 0.67 (8 of 12 slots)
    PARKED-DECODE carriers 4 parked of 4 resident, cap counts 0
    KV pool free 107422 of 472306, cached 364884

Four carriers ready to decode, which PP forbids. 572715 tokens of prefill
queued, which could not be admitted because those same four carriers' KV held
~352k of the 472k-row pool. py-spy showed all three ranks ACTIVE in
``get_next_batch_to_run`` -- spinning, not blocked in a collective -- and the
only thing that ever released it was the 180 s decode-stall cap.

WHY THE EXISTING EXITS COULD NOT FIRE, which is what makes this a missing
edge rather than a tuning problem:

* ``DRAINED`` needs ``pending <= one chunk``. There were 572715 tokens.
* the #677(a) progress exit needs pending FROZEN across a stall window. Pending
  was still creeping (572792 -> 572715), so it read as PROGRESS -- the
  instance was making a rounding error's worth of headway and the detector
  believed it.

So the state was: a fact the round already knew ("I built nothing, and I can
build nothing"), waiting to be re-derived by a timer.

WHAT THE ARM MUST NOT BECOME. A rule that leaves a layout is one flip away
from a rule that ping-pongs between two. The protection here is NOT a dwell
floor -- the branch deliberately runs ahead of ``min_dwell_s`` -- it is that
the condition is ONE-SIDED, and that is what these tests pin: the arm requires
the current layout to build nothing AND the target to be able to build
something, so after the flip the target runs by premise and the condition is
false on the other side.
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
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5)

N = 30000


def cfg(**kw):
    base = dict(
        enabled=True,
        flip_tokens=N,
        min_dwell_s=10.0,
        idle_dwell_s=20.0,
        rest_state=REST_PREFILL,
    )
    base.update(kw)
    return PhasePolicyConfig(**base)


def inputs(**kw):
    """The 09:42:39 specimen by default; kwargs vary one term at a time."""
    base = dict(
        phase=PHASE_PP,
        pending_prefill_tokens=572715,
        running_bs=4,
        now=100.0,
        nothing_can_run=True,
        target_can_admit=True,
    )
    base.update(kw)
    return PhasePolicyInputs(**base)


class TheDeadlockIsLeftImmediately(unittest.TestCase):
    def test_the_specimen_arms(self):
        d = decide(cfg(), PhasePolicyState(), inputs())
        self.assertEqual(PP_TO_TP, d.direction)
        self.assertIn("IDLE-LOCKED", d.reason)

    def test_it_outranks_the_dwell_floor(self):
        """The whole point: a layout that can run nothing must not wait.

        ``min_dwell_s`` is the right damper for "is flipping WORTH it". It is
        the wrong damper for "can this layout do ANYTHING", because there the
        wait buys nothing and costs the entire window.
        """
        state = PhasePolicyState(last_flip_at=99.0)  # 1s ago, dwell is 10s
        d = decide(cfg(), state, inputs(now=100.0))
        self.assertEqual(PP_TO_TP, d.direction)

    def test_the_tp_side_arms_the_other_way(self):
        d = decide(cfg(), PhasePolicyState(), inputs(phase=PHASE_TP))
        self.assertEqual(TP_TO_PP, d.direction)


class TheConditionIsOneSided(unittest.TestCase):
    """No oscillation BY CONSTRUCTION, with no timer underneath it."""

    def test_a_target_that_cannot_admit_does_not_arm(self):
        d = decide(cfg(), PhasePolicyState(), inputs(target_can_admit=False))
        self.assertIsNone(d.direction)

    def test_a_layout_that_can_still_build_does_not_arm(self):
        d = decide(cfg(), PhasePolicyState(), inputs(nothing_can_run=False))
        self.assertNotIn("IDLE-LOCKED", d.reason or "")

    def test_both_terms_are_required(self):
        for nothing, target in ((False, False), (False, True), (True, False)):
            d = decide(
                cfg(),
                PhasePolicyState(),
                inputs(nothing_can_run=nothing, target_can_admit=target),
            )
            self.assertNotIn(
                "IDLE-LOCKED",
                d.reason or "",
                f"armed on nothing_can_run={nothing}, target_can_admit={target}; "
                "the arm must require BOTH or it is no longer one-sided and can "
                "ping-pong between two layouts with no dwell floor under it.",
            )

    def test_the_flip_cannot_immediately_arm_back(self):
        """The premise, made explicit.

        After PP->TP the target runs, so on the TP side the round builds a
        batch and ``nothing_can_run`` is False. Feeding the post-flip state
        back in must NOT produce the return flip, or the pair would ping-pong
        at zero dwell.
        """
        first = decide(cfg(), PhasePolicyState(), inputs())
        self.assertEqual(PP_TO_TP, first.direction)
        post = decide(
            cfg(),
            PhasePolicyState(last_flip_at=100.0),
            inputs(phase=PHASE_TP, now=100.1, nothing_can_run=False),
        )
        self.assertNotIn("IDLE-LOCKED", post.reason or "")


class TheRecordedPingPongStateMustNotArm(unittest.TestCase):
    """THE 2026-08-16 10:24 STATE, REPLAYED. This is the regression.

        10:24:18 arming tp_to_pp: IDLE-LOCKED ... (2 req resident, 910140 tok)
        10:24:21 arming pp_to_tp: IDLE-LOCKED ...
        10:24:25 arming tp_to_pp: IDLE-LOCKED ...

    Two requests resident, 910140 tokens queued, the KV pool full of those
    carriers' own KV. NEITHER layout could run: PP could not admit a prefill
    because the pool was full, and TP could not decode the carriers either.
    The first version of the rule certified each layout as the other's escape
    and flipped every three to four seconds -- worse than the gap it replaced.

    With admissibility SIMULATED rather than inferred, the target term is
    False on both sides, and the correct verdict is not a flip at all: the
    binding resource is KV, so it is an evict trigger.
    """

    def _both_blocked(self, phase):
        return decide(
            cfg(),
            PhasePolicyState(),
            inputs(
                phase=phase,
                running_bs=2,
                pending_prefill_tokens=910140,
                nothing_can_run=True,
                target_can_admit=False,
            ),
        )

    def test_it_does_not_arm_from_pp(self):
        d = self._both_blocked(PHASE_PP)
        self.assertIsNone(d.direction)
        self.assertIn("BOTH BLOCKED", d.reason)

    def test_it_does_not_arm_from_tp(self):
        d = self._both_blocked(PHASE_TP)
        self.assertIsNone(d.direction)
        self.assertIn("BOTH BLOCKED", d.reason)

    def test_neither_direction_can_be_reached_from_the_other(self):
        """The loop itself: replay both legs and require no flip on either."""
        self.assertIsNone(self._both_blocked(PHASE_PP).direction)
        self.assertIsNone(self._both_blocked(PHASE_TP).direction)

    def test_the_refusal_names_the_binding_resource(self):
        d = self._both_blocked(PHASE_PP)
        self.assertIn("KV", d.reason)
        self.assertIn("evict", d.reason.lower())


class TheRecordedPrematureArmMustNotArm(unittest.TestCase):
    """THE 2026-08-16 10:47:42 STATE, REPLAYED. The bs=1 defect's cause.

        Prefill batch, #new-seq: 1, #new-token: 25, #cached-token: 25600,
        full token usage: 0.05, mamba usage: 0.50, #running-req: 0,
        #queue-req: 1, #pending-token: 51250
        -> arming pp_to_tp: IDLE-LOCKED (1 req resident, 25625 tok pending)

    The pool was 5% used with ~446k rows free, 6 of 12 GDN slots were free and
    a request was still queued. PP could obviously have admitted more. The arm
    fired because ``nothing_can_run`` meant "this round happened to build
    nothing", which a single transient empty round satisfies -- and because
    #688 outranks #689 by design, the premature arm BYPASSED window formation
    and opened the decode window at ONE carrier. That is the bs=1 defect.

    With the current-layout term simulated as well, a layout that CAN admit
    does not report that nothing can run, whatever one round happened to do.
    """

    def test_a_layout_that_can_still_admit_does_not_arm(self):
        d = decide(
            cfg(),
            PhasePolicyState(),
            inputs(
                phase=PHASE_PP,
                running_bs=1,
                pending_prefill_tokens=25625,
                nothing_can_run=False,  # the simulation says PP can admit
                target_can_admit=True,
            ),
        )
        self.assertNotIn("IDLE-LOCKED", d.reason or "")


class AnIdleServerIsNotADeadlock(unittest.TestCase):
    """ "No batch" alone is not the trigger, deliberately.

    An empty instance also builds nothing, and flipping it is thrash with no
    work to show for it. The scheduler-side observation requires resident
    requests or pending tokens before it will even set the flag; this pins
    the policy half of that contract.
    """

    def test_no_work_anywhere_does_not_arm(self):
        d = decide(
            cfg(),
            PhasePolicyState(),
            inputs(
                pending_prefill_tokens=0,
                running_bs=0,
                nothing_can_run=False,
                target_can_admit=False,
            ),
        )
        self.assertNotIn("IDLE-LOCKED", d.reason or "")


class TheDefaultIsOff(unittest.TestCase):
    def test_inputs_without_the_fields_behave_as_before(self):
        """Callers that never learned about this edge must be unaffected --
        including the scheduler stand-ins the policy tests drive."""
        d = decide(
            cfg(),
            PhasePolicyState(),
            PhasePolicyInputs(
                phase=PHASE_PP,
                pending_prefill_tokens=572715,
                running_bs=4,
                now=100.0,
            ),
        )
        self.assertNotIn("IDLE-LOCKED", d.reason or "")


if __name__ == "__main__":
    unittest.main()
