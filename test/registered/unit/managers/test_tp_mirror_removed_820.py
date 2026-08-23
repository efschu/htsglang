"""#820: the #677 hold's "TP side mirror" was unreachable, so it is gone.

WHAT WAS THERE. ``layout_hold_verdict`` carried a ``phase == "tp"`` branch --
added by 332cb3b345, whose title is "Wire the hold, mirror it on the TP side".
It answered "does waiting prefill pull the layout out of tp, or does a RUNNING
decode batch get MIN_DECODE_ROUNDS first".

WHY IT COULD NEVER RUN. ``decide`` consults the verdict at exactly one place,
under ``if d.direction == PP_TO_TP and d.hold_eligible``, passing
``phase=inp.phase``. Every arm that can carry PP_TO_TP is built under
``inp.phase == PHASE_PP`` -- the three sites are the #759 escape (which derives
the direction FROM the phase) and the arms inside the ``if inp.phase ==
PHASE_PP`` block. So the phase handed to that call is always "pp", and the "tp"
branch had no reachable caller. That is a control-flow property, and
``TestEveryHoldableArmIsAPpPhaseArm`` below drives the real rules to pin it
rather than asserting it from a reading.

WHY WIRING IT WOULD HAVE BEEN WRONG, not merely unnecessary -- three reasons,
each independent:

1. THE AUTHOR ALREADY TRIED, in the same commit, and recorded the result:
   "AND THE GATE IS PP_TO_TP ONLY... My first wiring vetoed EVERY arm,
   including the idle return leg -- 11 tests red -- because the rules arm for
   reasons this lever cannot see."  The mechanism is visible in the branch
   itself: for ``pend <= 0`` it returned False. In PULL semantics False means
   "no pull"; read as a VETO -- which is all the wrapper can do -- the same
   False means "never leave tp", and unbounded, because that return never
   consulted ``max_hold_rounds``.

2. #817'S ADMISSION CONDITION IS UNSATISFIABLE ON THE TP SIDE. An arm may be
   hold_eligible only if a SECOND INDEPENDENT anti-starvation bound is armed in
   every state where it fires. The only TP_TO_PP arm the mirror could reach is
   the pending-prefill arm, and that arm is the only thing that takes prefill
   out of tp: there is a ``pp_residency_cap_s`` bounding a PP residency to
   protect decode, and no counterpart bounding a TP residency to protect
   prefill. Vetoing it is vetoing the last bound -- verbatim the verdict that
   made #817 rule the legacy stopwatch an EXIT.

3. THE CALL SITE NEVER HAD THE INPUT. It passed ``decode_rounds_so_far=
   state.hold_rounds`` -- the HOLD counter, whose lifecycle resets on a phase
   change or on pending reaching zero -- as a count of DECODE rounds. Two
   different quantities.

WHERE THE JOB IS ACTUALLY DONE. The protection the mirror expressed lives in
the rules, as the DECODE FLOOR on the pending-prefill arm: a pull yields while
``running_bs > 0`` and the phase is younger than ``tp_decode_floor_s``. Stated
against the real phase clock in seconds, not against a counter the caller does
not hold. ``TestTheMirrorsJobIsDoneByTheRules`` carries over the content of the
deleted mirror tests, driving the real rules; ``test_phase_purity_631.py::
test_decode_floor_stops_the_mirror_starvation`` has independently pinned the
same guard since before this removal.
"""

import unittest

from sglang.srt.managers import phase_policy
from sglang.srt.managers.phase_policy import (
    PHASE_PP,
    PHASE_TP,
    PP_TO_TP,
    TP_TO_PP,
    PhasePolicyConfig,
    PhasePolicyDecision,
    PhasePolicyInputs,
    PhasePolicyState,
    decide,
    observe_idle,
)
from sglang.test.test_utils import CustomTestCase


def _cfg(**kw):
    base = dict(enabled=True, flip_tokens=7004, min_dwell_s=3.0)
    base.update(kw)
    return PhasePolicyConfig(**base)


def _drive(cfg, state, phase, pending, bs, now, **kw):
    inp = PhasePolicyInputs(
        phase=phase, pending_prefill_tokens=pending, running_bs=bs, now=now, **kw
    )
    observe_idle(state, inp)
    return inp, decide(cfg, state, inp)


class TestEveryHoldableArmIsAPpPhaseArm(CustomTestCase):
    """THE LOAD-BEARING CLAIM of this removal, driven rather than read.

    If a PP_TO_TP arm could ever be produced while resident in tp, the deleted
    branch would have been reachable after all and deleting it would be a
    behaviour change. So sweep the rules across layouts, loads, clocks and
    configurations and check the implication on every arm they actually build.
    """

    CONFIGS = (
        _cfg(),
        _cfg(pp_window_s=15.0),
        _cfg(pp_window_s=15.0, tp_decode_floor_s=10.0),
        _cfg(tp_decode_floor_s=10.0, prefill_runs_in_tp=False),
        _cfg(decode_stall_slo_s=45.0, pp_window_s=15.0),
        _cfg(idle_dwell_s=2.0, rest_state="prefill"),
        _cfg(idle_dwell_s=2.0, rest_state="decode"),
        _cfg(drain_mode=True, pp_window_s=15.0, tp_decode_floor_s=10.0),
        _cfg(pp_window_s=15.0, prefill_runs_in_tp=True, flip_tokens=100),
    )

    def _sweep(self):
        """Every (config, layout, load, clock) decision the rules produce."""
        seen = []
        for cfg in self.CONFIGS:
            for phase in (PHASE_PP, PHASE_TP):
                for pending in (0, 22, 7003, 302757):
                    for bs in (0, 1, 4):
                        for carriers in (0, 3):
                            state = PhasePolicyState()
                            for now in (0.0, 3.0, 9.0, 15.0, 46.0, 200.0):
                                inp, d = _drive(
                                    cfg,
                                    state,
                                    phase,
                                    pending,
                                    bs,
                                    now,
                                    ready_carriers=carriers,
                                )
                                seen.append((inp, d))
        return seen

    def test_every_pp_to_tp_arm_the_rules_build_was_built_in_the_pp_layout(self):
        offenders = [
            (i.phase, d.reason)
            for i, d in self._sweep()
            if d.direction == PP_TO_TP and i.phase != PHASE_PP
        ]
        self.assertEqual(
            offenders,
            [],
            "a PP_TO_TP arm outside the pp layout would make the deleted "
            "tp branch reachable",
        )

    def test_the_sweep_is_not_vacuous(self):
        """CAN-FAIL GUARD. The implication above is trivially true over an
        empty set, so the sweep has to be shown to produce both directions --
        otherwise a broken harness reads as a proof."""
        dirs = {d.direction for _, d in self._sweep()}
        self.assertIn(PP_TO_TP, dirs, "the sweep must actually build pp_to_tp arms")
        self.assertIn(TP_TO_PP, dirs, "the sweep must actually build tp_to_pp arms")

    def test_every_tp_to_pp_arm_was_built_in_the_tp_layout(self):
        """The mirror image, and it is the reason the gate can stay one-sided:
        the two directions never cross layouts, so a lever scoped to one
        direction is also scoped to one layout."""
        offenders = [
            (i.phase, d.reason)
            for i, d in self._sweep()
            if d.direction == TP_TO_PP and i.phase != PHASE_TP
        ]
        self.assertEqual(offenders, [])


class TestTheGateStaysPpToTpOnly(CustomTestCase):
    """332cb3b345's "11 tests red", pinned so it cannot be relearned.

    The wrapper is fed a synthetic decision so the gate is tested directly,
    independent of which arms the rules happen to build today.
    """

    def _decide_with(self, d, *, phase, pending=22, bs=1, hold_rounds=0):
        state = PhasePolicyState()
        state.hold_rounds = hold_rounds
        inp = PhasePolicyInputs(
            phase=phase, pending_prefill_tokens=pending, running_bs=bs, now=100.0
        )
        original = phase_policy._decide_rules
        phase_policy._decide_rules = lambda cfg, st, i: d
        try:
            return decide(_cfg(), state, inp)
        finally:
            phase_policy._decide_rules = original

    def test_a_tp_to_pp_arm_is_not_vetoed_even_when_marked_hold_eligible(self):
        """THE IDLE RETURN LEG IS A TP_TO_PP ARM. A gate that vetoed it would
        park the server in a decode layout with nothing to decode."""
        out = self._decide_with(
            PhasePolicyDecision(
                TP_TO_PP, "idle: returning to rest", hold_eligible=True
            ),
            phase=PHASE_TP,
        )
        self.assertEqual(out.direction, TP_TO_PP)
        self.assertEqual(out.reason, "idle: returning to rest")

    def test_the_harness_can_see_a_veto(self):
        """CAN-FAIL for the test above: with the direction changed to PP_TO_TP
        and everything else held fixed, the same harness DOES observe the hold.
        Without this, "not vetoed" could mean "the lever never ran"."""
        out = self._decide_with(
            PhasePolicyDecision(PP_TO_TP, "a timer arm", hold_eligible=True),
            phase=PHASE_PP,
        )
        self.assertIsNone(out.direction, out.reason)
        self.assertIn("HOLD", out.reason)

    def test_an_unmarked_tp_to_pp_arm_is_untouched_too(self):
        out = self._decide_with(
            PhasePolicyDecision(TP_TO_PP, "pending prefill 302757 tok > 0"),
            phase=PHASE_TP,
        )
        self.assertEqual(out.direction, TP_TO_PP)


class TestTheMirrorsJobIsDoneByTheRules(CustomTestCase):
    """The deleted mirror tests, restated against the guard that really runs.

    Same three properties the mirror claimed -- yield to a running batch,
    proceed once the bound is met, do not hold an idle decode layout -- plus
    the degenerate-cycle pin, all through the real rules.
    """

    def test_a_running_decode_batch_is_not_preempted_by_the_prefill_pull(self):
        cfg = _cfg(tp_decode_floor_s=10.0, prefill_runs_in_tp=False)
        state = PhasePolicyState()
        _drive(cfg, state, PHASE_TP, 302757, 2, 0.0)
        _, held = _drive(cfg, state, PHASE_TP, 302757, 2, 9.0)
        self.assertIsNone(held.direction, held.reason)
        self.assertIn("decode floor", held.reason)

    def test_the_pull_proceeds_once_the_floor_is_met(self):
        """CAN-FAIL: the yield must be BOUNDED, or it becomes the prefill-side
        starvation the mirror itself warned about."""
        cfg = _cfg(tp_decode_floor_s=10.0, prefill_runs_in_tp=False)
        state = PhasePolicyState()
        _drive(cfg, state, PHASE_TP, 302757, 2, 0.0)
        _, out = _drive(cfg, state, PHASE_TP, 302757, 2, 10.0)
        self.assertEqual(out.direction, TP_TO_PP, out.reason)

    def test_an_idle_decode_layout_is_not_held(self):
        """The guard protects a RUNNING batch, not an empty tp layout: a long
        prompt arriving at a decode-idle server must reach pp inside its TTFT.
        """
        cfg = _cfg(min_dwell_s=0.0, tp_decode_floor_s=10.0, prefill_runs_in_tp=False)
        state = PhasePolicyState()
        _, out = _drive(cfg, state, PHASE_TP, 302757, 0, 0.5)
        self.assertEqual(out.direction, TP_TO_PP, out.reason)

    def test_the_layout_cannot_ping_pong_straight_back(self):
        """THE DEGENERATE CYCLE the mirror was written for. Leave pp under a
        sustained both-sides load, then evaluate in tp with a decode batch now
        running: the pull must not take the layout straight back, or the cycle
        becomes all pp rounds, no tp rounds and two seams apiece."""
        cfg = _cfg(pp_window_s=15.0, tp_decode_floor_s=10.0, prefill_runs_in_tp=False)
        state = PhasePolicyState()
        _drive(cfg, state, PHASE_PP, 302757, 2, 0.0)
        _, leaving = _drive(cfg, state, PHASE_PP, 302757, 2, 15.0)
        self.assertEqual(leaving.direction, PP_TO_TP, leaving.reason)
        tp_state = PhasePolicyState()
        _drive(cfg, tp_state, PHASE_TP, 302757, 2, 16.0)
        _, back = _drive(cfg, tp_state, PHASE_TP, 302757, 2, 17.0)
        self.assertIsNone(back.direction, back.reason)


class TestTheRemovedSurfaceIsGone(CustomTestCase):
    def test_the_mirror_constants_are_no_longer_defined(self):
        for name in ("MIN_DECODE_ROUNDS", "PULL_FOR_DEMAND"):
            with self.subTest(name=name):
                self.assertFalse(
                    hasattr(phase_policy, name),
                    f"{name} was read only by the deleted branch",
                )

    def test_the_verdict_no_longer_accepts_the_mirror_arguments(self):
        for kw in ("decode_serving", "decode_rounds_so_far", "min_decode_rounds"):
            with self.subTest(kw=kw):
                with self.assertRaises(TypeError):
                    phase_policy.layout_hold_verdict("pp", 22, 0, **{kw: 1})

    def test_the_pp_side_verdict_still_works(self):
        """CAN-FAIL for the removal: the half that IS reachable must be intact,
        so this is not a test that would pass against a deleted function."""
        allow, why = phase_policy.layout_hold_verdict("pp", 22, 0)
        self.assertFalse(allow)
        self.assertIn(phase_policy.HOLD_FOR_UNSERVED, why)

    def test_an_unreadable_phase_no_longer_falls_toward_holding(self):
        """The direction of failure, which the removal made load-bearing: "tp"
        went from a recognised phase to an unrecognised one, and the sole
        consumer treats False as a veto."""
        for phase in ("tp", "flip_in_progress", ""):
            with self.subTest(phase=phase):
                allow, why = phase_policy.layout_hold_verdict(phase, 302757, 4)
                self.assertTrue(allow, why)
                self.assertIn("no decision", why)


if __name__ == "__main__":
    unittest.main()
