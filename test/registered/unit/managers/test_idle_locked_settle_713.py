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
"""#713: the IDLE-LOCKED post-cutover settle, replayed against the recording.

THE SPECIMEN, verbatim from boot_bundle.log.20260817T070900Z:

    06:53:56 arming tp_to_pp: IDLE-LOCKED ... (0 req resident, 22 tok pending)
    06:53:59 cutover complete: active stack pp  (DONE tp_to_pp, 2847 ms)
    06:53:59 arming pp_to_tp: IDLE-LOCKED ... (1 req resident, 0 tok pending)
    06:54:01 cutover complete: active stack tp  (DONE pp_to_tp, 2284 ms)
    06:54:01 arming tp_to_pp: IDLE-LOCKED ... (0 req resident, 22 tok pending)
    06:54:04 cutover complete: active stack pp  (DONE tp_to_pp, 2848 ms)

Every arm lands in the SAME SECOND as the cutover that preceded it. That is the
whole defect: the layout is judged unable to run before it has run, because
Scheduler._idle_locked_inputs is gated on ``_round_built_nothing``, which a
layout that has only just been entered satisfies trivially.

The claim that used to stand at this branch -- "IT CANNOT OSCILLATE ... after
the flip the target runs by premise" -- was false on metal: alternating runs of
72 arms / 299 s, 12 / 31 s, 10 / 27 s twice, across the 16 rotations of
2026-08-17.

RED-FIRST. The replay below runs the RECORDED arms through the real policy
twice, once with the settle disabled and once with it on. Disabled, the policy
arms every one of them back and reproduces the ping-pong; enabled, it holds. If
someone deletes the guard, ``test_replay_unguarded_reproduces_the_pingpong``
keeps passing and the guarded replay goes red -- the recording is the fixture,
so the suite cannot drift away from what the instance actually did.

WHAT THIS GUARD MUST NOT BREAK. #688's escape: a layout that genuinely cannot
run anything must still leave IMMEDIATELY, ahead of ``min_dwell_s``. The settle
only refuses an emptiness observed within its window of ENTERING the layout,
which is the only place the transient exists. The cost is stated and pinned
below: a genuine idle lock forming inside that window is delayed, by at most
the settle, and the settle is capped at ``min_dwell_s`` so this branch can
never be slower than the ordinary dwell path it exists to bypass.
"""

import importlib.util
import pathlib
import re
import sys
import unittest

from sglang.srt.managers.phase_policy import (
    DEFAULT_IDLE_LOCKED_SETTLE_S,
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

REPO = pathlib.Path(__file__).resolve().parents[4]
FIXTURE = REPO / "scripts" / "fixtures" / "d2_injector_pingpong_excerpt.txt"
INJECTOR = REPO / "scripts" / "d2_phase_locked_injector.py"


def cfg(**kw):
    base = dict(
        enabled=True,
        flip_tokens=30000,
        min_dwell_s=10.0,
        idle_dwell_s=20.0,
        rest_state=REST_PREFILL,
    )
    base.update(kw)
    return PhasePolicyConfig(**base)


def inputs(**kw):
    """The 06:53:59 arm-back by default; kwargs vary one term at a time."""
    base = dict(
        phase=PHASE_PP,
        pending_prefill_tokens=0,
        running_bs=1,
        now=100.0,
        nothing_can_run=True,
        target_can_admit=True,
    )
    base.update(kw)
    return PhasePolicyInputs(**base)


def _load_injector():
    spec = importlib.util.spec_from_file_location("d2_injector_for_replay", INJECTOR)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _recorded_arms_with_phase_age(in_run: bool):
    """Recorded arms paired with how long their layout had been entered.

    Reuses the injector's parser rather than a second private one, so the
    harness and this suite cannot disagree about what the log says.

    ``in_run`` selects arms that belong to a detected alternating run -- the
    ping-pong itself -- or the arms outside it. THE SPLIT IS THE POINT. The
    first version of this replay assumed every arm in the excerpt was a
    ping-pong arm; the data refused, one arm being 35 s past its cutover. That
    arm is not noise, it is the control case: a layout that HAS settled and is
    genuinely idle-locked, which the guard must still let through.
    """
    inj = _load_injector()
    with open(FIXTURE, "r", errors="replace") as fh:
        markers = inj.parse_markers(fh)
    cutovers = [(inj._epoch(ts), stack) for ts, _rank, stack in markers["cutovers"]]
    run_arm_ids = {
        id(a) for run in inj.alternating_runs(markers["arms"]) for a in run.arms
    }
    out = []
    for arm in markers["arms"]:
        if (id(arm) in run_arm_ids) != in_run:
            continue
        t = inj._epoch(arm.ts)
        prior = [(ct, stack) for ct, stack in cutovers if ct <= t]
        if not prior:
            continue  # no observed entry into this layout inside the excerpt
        entered_at, stack = prior[-1]
        out.append((arm, stack, t - entered_at))
    return out


class TheRecordedPingPongReplays(unittest.TestCase):
    def _replay(self, arm, stack, age, **cfgkw):
        return decide(
            cfg(**cfgkw),
            PhasePolicyState(phase_since=100.0 - age),
            inputs(
                phase=PHASE_PP if stack == "pp" else PHASE_TP,
                running_bs=arm.running_bs,
                pending_prefill_tokens=arm.pending,
                now=100.0,
            ),
        )

    def test_the_excerpt_carries_the_recorded_run(self):
        """Guards the guard: an empty replay would pass everything below.

        THE SHAPE OF THE RUN IS THE WHOLE ARGUMENT, so it is pinned. The first
        arm is 35 s past its cutover -- a settled layout, genuinely idle-locked,
        a LEGITIMATE escape. Every one of the eleven that follow lands 0-1 s
        after a cutover, which is the transient. A replay that lost that split
        would be testing nothing.
        """
        arms = _recorded_arms_with_phase_age(in_run=True)
        self.assertEqual(12, len(arms), "the recorded run is 12 arms")
        settled = [a for a in arms if a[2] >= DEFAULT_IDLE_LOCKED_SETTLE_S]
        transient = [a for a in arms if a[2] < DEFAULT_IDLE_LOCKED_SETTLE_S]
        self.assertEqual(1, len(settled), "exactly one arm starts the run")
        self.assertEqual(11, len(transient), "the other eleven are arm-backs")

    def test_replay_unguarded_reproduces_the_pingpong(self):
        """Settle disabled: the policy arms all twelve and the loop is back."""
        armed = sum(
            1
            for arm, stack, age in _recorded_arms_with_phase_age(in_run=True)
            if self._replay(arm, stack, age, idle_locked_settle_s=0.0).direction
        )
        self.assertEqual(12, armed)

    def test_replay_guarded_leaves_only_the_initiating_arm(self):
        """Settle on: twelve arms become ONE. That is the fix, quantified.

        Not zero -- zero would mean the guard had eaten #688's escape along
        with the ping-pong. The arm that legitimately leaves a 35 s-settled
        idle-locked layout survives; the eleven arm-backs on 0-1 s-old layouts
        do not.
        """
        armed, held = [], []
        for arm, stack, age in _recorded_arms_with_phase_age(in_run=True):
            d = self._replay(arm, stack, age)
            (armed if d.direction else held).append((arm.ts, age, d.reason))
        self.assertEqual(1, len(armed), f"expected only the initiator, got {armed}")
        self.assertGreaterEqual(armed[0][1], DEFAULT_IDLE_LOCKED_SETTLE_S)
        self.assertEqual(11, len(held))
        for _ts, _age, reason in held:
            self.assertIn("settle", reason)
            self.assertIn("transient, not a verdict", reason)


class TheGuardDelaysRatherThanDisables(unittest.TestCase):
    def test_inside_the_settle_it_holds(self):
        d = decide(cfg(), PhasePolicyState(phase_since=99.0), inputs())  # 1s in
        self.assertIsNone(d.direction)
        self.assertIn("transient, not a verdict", d.reason)

    def test_past_the_settle_the_same_state_arms(self):
        """The escape is delayed, not removed -- this is the pair to the above."""
        d = decide(cfg(), PhasePolicyState(phase_since=97.0), inputs())  # 3s in
        self.assertEqual(PP_TO_TP, d.direction)
        self.assertIn("IDLE-LOCKED", d.reason)

    def test_688_escape_still_outranks_the_dwell(self):
        """A settled layout that can run nothing must not wait for min_dwell.

        This is the property #688 exists for and the reason this branch sits
        above the dwell at all. The settle must not have quietly reintroduced
        the dwell it was written to preserve.
        """
        state = PhasePolicyState(last_flip_at=99.0, phase_since=90.0)  # 10s in
        d = decide(cfg(min_dwell_s=10.0), state, inputs(now=100.0))
        self.assertEqual(PP_TO_TP, d.direction)
        self.assertNotIn("min dwell", d.reason)

    def test_the_tp_side_of_the_loop_is_guarded_too(self):
        """Both halves, or the loop simply runs in the other direction."""
        d = decide(
            cfg(),
            PhasePolicyState(phase_since=100.0),
            inputs(phase=PHASE_TP, running_bs=0, pending_prefill_tokens=22),
        )
        self.assertIsNone(d.direction)
        d = decide(
            cfg(),
            PhasePolicyState(phase_since=90.0),
            inputs(phase=PHASE_TP, running_bs=0, pending_prefill_tokens=22),
        )
        self.assertEqual(TP_TO_PP, d.direction)


class TheSettleIsBounded(unittest.TestCase):
    def test_it_is_capped_at_min_dwell(self):
        """A settle above the dwell would invert this branch's purpose.

        The branch exists to be FASTER than the dwell path. The cap is applied
        where the value is used, so an operator cannot configure the fast
        escape into being the slow one.
        """
        state = PhasePolicyState(phase_since=98.0)  # 2s into the layout
        d = decide(cfg(min_dwell_s=1.0, idle_locked_settle_s=30.0), state, inputs())
        self.assertEqual(PP_TO_TP, d.direction, "cap must bound the settle")

    def test_zero_disables_the_guard_exactly(self):
        d = decide(
            cfg(idle_locked_settle_s=0.0), PhasePolicyState(phase_since=100.0), inputs()
        )
        self.assertEqual(PP_TO_TP, d.direction)

    def test_unobserved_phase_degrades_to_pre_guard_behaviour(self):
        """None must not mean an infinite settle.

        A caller that never called observe_idle -- every pre-window unit test,
        and the five #677-era pins -- must keep the old behaviour rather than
        find the escape pinned shut.
        """
        d = decide(cfg(), PhasePolicyState(phase_since=None), inputs())
        self.assertEqual(PP_TO_TP, d.direction)

    def test_default_is_the_measured_value_and_under_the_rig_dwell(self):
        """2.0 s covers 84.7 % of the 162 measured cutover transients.

        Pinned because the number is derived, not chosen: it must stay under
        the 3 s min_dwell this rig boots with, or the cap above starts silently
        doing the deciding.
        """
        self.assertEqual(2.0, DEFAULT_IDLE_LOCKED_SETTLE_S)
        self.assertLess(DEFAULT_IDLE_LOCKED_SETTLE_S, 3.0)


class TheRetractedDiagnosisIsGone(unittest.TestCase):
    def test_both_blocked_no_longer_names_an_unmeasured_cause(self):
        """#712 was closed as unfounded; its text must not outlive it.

        The line used to redirect readers to "the state-slot bound (mamba/GDN
        slots)" -- a hypothesis written into a log string, then read back out
        and filed as a finding.
        """
        src = (
            REPO / "python" / "sglang" / "srt" / "managers" / "phase_policy.py"
        ).read_text()
        body = re.sub(r"#\s*#712 RETRACTED.*?binding = \(", "", src, flags=re.S)
        self.assertNotIn("state-slot bound", body)
        d = decide(
            cfg(),
            PhasePolicyState(phase_since=90.0),
            inputs(target_can_admit=False, kv_available_tokens=10**9),
        )
        self.assertIn("does not name the one that is", d.reason)


if __name__ == "__main__":
    unittest.main()
