"""#677 hot fix 2: finish the decode bundle, and never prefill in TP.

WHAT THE USER SAW, LIVE. The blocked-admission exit (#677a) broke the wedge but
handed back TP windows that did not do their job:

  * ``arming tp_to_pp: pending > N=7004`` fired while carriers were still
    mid-decode -- the log shows tp_to_pp taken with running bs 2-3 -- so a
    decode bundle was cut in half by a backlog that is ALWAYS above N under
    purity;
  * prefill then ran inside the TP layout (purity mode prefill_in_tp), so the
    carriers that survived the short window met a layout that was busy
    prefilling instead of finishing them.

Together those produced repeated flips at 400-500k pending with visible prefill
in TP, and five blocked-admission exits in one boot: the exit was firing over
and over because each TP window returned the same unfinished carriers.

THE USER'S TARGET SEMANTICS ARE EXPLICIT: prefill until empty, then decode the
bundle TO COMPLETION, then prefill again. Hot fix 2 makes the TP side match.

  1. TP EXIT = DECODE DRAINED. Under drain mode ``tp_to_pp`` arms only when
     ``running_bs == 0`` AND there is prefill worth going back for. A backlog
     above N no longer cuts a bundle: under purity that backlog is permanent,
     so treating it as a reason to leave means never finishing anything.
  2. NO PREFILL IN TP. Under drain mode the TP layout decodes and nothing else.
     Cards partially idle during TP is ACCEPTED in the user's model -- the
     trade is deliberate, and the alternative measured worse.

BOTH ARE OFF BY DEFAULT. ``drain_mode`` defaults False and every rule below is
byte-identical to today's until it is set, so no existing deployment changes.

THE MIN-DWELL AND THE BACKSTOPS ARE UNTOUCHED: the 180s decode-stall cap and
the #677a progress exit still sit underneath, so a bundle that cannot finish is
still bounded -- drain mode changes which condition ENDS a healthy window, not
what rescues a broken one.
"""

import unittest
import unittest.mock

from sglang.srt.managers import phase_policy as pp
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5)


def _cfg(**kw):
    base = dict(
        enabled=True,
        flip_tokens=7004,
        flip_cost_s=3.0,
        drain_mode=True,
        pp_exit_tokens=512,
        pp_prefill_tok_s=1000.0,
        min_dwell_s=0.0,
        tp_decode_floor_s=0.0,
        idle_dwell_s=1.0,
    )
    base.update(kw)
    return pp.PhasePolicyConfig(**base)


def _tp(now, *, pending, bs):
    return pp.PhasePolicyInputs(
        phase=pp.PHASE_TP, pending_prefill_tokens=pending, running_bs=bs, now=now
    )


def _drive(cfg, series):
    """observe -> decide over (now, pending, bs), as the loop does."""
    state = pp.PhasePolicyState()
    last = None
    for now, pending, bs in series:
        inp = _tp(now, pending=pending, bs=bs)
        pp.observe_idle(state, inp)
        last = pp.decide(cfg, state, inp)
    return last, state


class TheBundleIsFinished(unittest.TestCase):
    def test_a_backlog_above_N_no_longer_cuts_a_live_bundle(self):
        """PIN (a), first half. 400k pending with 3 still decoding must hold."""
        cfg = _cfg()
        decision, _ = _drive(cfg, [(0.0, 400_000, 4), (30.0, 400_000, 3)])
        self.assertIsNone(decision.direction, decision.reason)
        self.assertIn("decode", decision.reason.lower())

    def test_it_arms_the_moment_the_bundle_is_empty(self):
        """PIN (a), second half."""
        cfg = _cfg()
        decision, _ = _drive(
            cfg, [(0.0, 400_000, 4), (30.0, 400_000, 2), (60.0, 400_000, 0)]
        )
        self.assertEqual(pp.TP_TO_PP, decision.direction, decision.reason)

    def test_the_receipt_names_the_bundle_and_its_duration(self):
        cfg = _cfg()
        decision, _ = _drive(
            cfg, [(0.0, 400_000, 4), (60.0, 400_000, 0)]
        )
        self.assertIn("decode bundle complete", decision.reason)
        self.assertIn("4", decision.reason, "the bundle size must be named")
        self.assertIn("decode drained", decision.reason)

    def test_an_empty_bundle_with_nothing_pending_does_not_arm_on_this_rule(self):
        """Below one chunk there is nothing worth the round trip; the idle
        path owns that case, not this one."""
        cfg = _cfg()
        decision, _ = _drive(cfg, [(0.0, 10, 0), (5.0, 10, 0)])
        self.assertNotIn("decode bundle complete", decision.reason)

    def test_drain_mode_off_is_byte_identical_to_today(self):
        """The whole change is gated: with drain_mode False a backlog above N
        still arms with a live bundle, exactly as it does now."""
        cfg = _cfg(drain_mode=False)
        decision, _ = _drive(cfg, [(0.0, 400_000, 3), (30.0, 400_000, 3)])
        self.assertEqual(pp.TP_TO_PP, decision.direction, decision.reason)
        self.assertNotIn("decode bundle complete", decision.reason)


class NoPrefillRunsInTp(unittest.TestCase):
    def test_drain_mode_suppresses_prefill_in_the_tp_layout(self):
        """PIN (b), at the policy level."""
        self.assertTrue(pp.prefill_suppressed_in_tp(_cfg(), pp.PHASE_TP))

    def test_it_says_nothing_about_the_pp_layout(self):
        self.assertFalse(pp.prefill_suppressed_in_tp(_cfg(), pp.PHASE_PP))

    def test_without_drain_mode_it_suppresses_nothing(self):
        self.assertFalse(
            pp.prefill_suppressed_in_tp(_cfg(drain_mode=False), pp.PHASE_TP)
        )

    def test_the_purity_hook_consults_it(self):
        """THE CALL SITE. A predicate nothing asks is a predicate that does
        nothing -- the lesson the #684 serving-mark tick paid for."""
        import inspect

        from sglang.srt.managers import phase_purity

        src = inspect.getsource(phase_purity)
        self.assertIn("prefill_suppressed_in_tp", src)


class TheFullCycle(unittest.TestCase):
    """PIN (c): PP drains to carriers -> ONE flip -> TP decodes all of them ->
    ONE flip back. The wedge cycle produced five exits in a boot precisely
    because each TP window returned the same unfinished carriers."""

    def test_one_flip_each_way_per_bundle(self):
        cfg = _cfg()
        state = pp.PhasePolicyState()
        arms = []

        # PP: prefill until the pool is carrying its four and cannot admit.
        for now, pending, bs in ((0.0, 400_000, 4), (10.0, 400_000, 4)):
            inp = pp.PhasePolicyInputs(
                phase=pp.PHASE_PP, pending_prefill_tokens=pending,
                running_bs=bs, now=now,
            )
            pp.observe_idle(state, inp)
            d = pp.decide(cfg, state, inp)
            if d.wants_flip:
                arms.append((now, d.direction))
        self.assertEqual(1, len(arms), f"expected one PP exit, got {arms}")
        self.assertEqual(pp.PP_TO_TP, arms[0][1])

        # TP: the bundle decodes down. Exactly one arm, and only at zero.
        arms = []
        for now, bs in ((20.0, 4), (30.0, 3), (40.0, 1), (50.0, 0)):
            inp = _tp(now, pending=400_000, bs=bs)
            pp.observe_idle(state, inp)
            d = pp.decide(cfg, state, inp)
            if d.wants_flip:
                arms.append((now, d.direction, d.reason))
        self.assertEqual(1, len(arms), f"the bundle was cut short: {arms}")
        self.assertEqual(pp.TP_TO_PP, arms[0][1])
        self.assertEqual(50.0, arms[0][0], "it must arm only once bs reached 0")


class TheBackstopsAreUntouched(unittest.TestCase):
    """PIN (e). Drain mode changes which condition ENDS a healthy window; it
    must not disarm what rescues a broken one."""

    def test_the_progress_exit_still_fires_in_pp(self):
        cfg = _cfg()
        window = pp.pp_progress_stall_window_s(cfg)
        state = pp.PhasePolicyState()
        last = None
        for now in (0.0, window + 1.0):
            inp = pp.PhasePolicyInputs(
                phase=pp.PHASE_PP, pending_prefill_tokens=403_779,
                running_bs=4, now=now,
            )
            pp.observe_idle(state, inp)
            last = pp.decide(cfg, state, inp)
        self.assertEqual(pp.PP_TO_TP, last.direction)
        self.assertIn("blocked admission", last.reason)

    def test_the_decode_stall_cap_is_still_solved_when_declared(self):
        self.assertGreater(pp.pp_residency_cap_s(_cfg(decode_stall_slo_s=180.0)), 0.0)


if __name__ == "__main__":
    unittest.main()


class TheGateIsReachableFromABoot(unittest.TestCase):
    """A flag no boot can set is a flag that does nothing."""

    def test_the_env_knob_turns_drain_mode_on(self):
        import os

        old = os.environ.get(pp.ENV_DRAIN_MODE)
        try:
            os.environ[pp.ENV_DRAIN_MODE] = "1"
            self.assertTrue(pp.config_from_env(True).drain_mode)
            os.environ[pp.ENV_DRAIN_MODE] = "0"
            self.assertFalse(pp.config_from_env(True).drain_mode)
        finally:
            if old is None:
                os.environ.pop(pp.ENV_DRAIN_MODE, None)
            else:
                os.environ[pp.ENV_DRAIN_MODE] = old

    def test_unset_keeps_the_deployments_current_behaviour(self):
        import os

        old = os.environ.pop(pp.ENV_DRAIN_MODE, None)
        try:
            self.assertFalse(pp.config_from_env(True).drain_mode)
        finally:
            if old is not None:
                os.environ[pp.ENV_DRAIN_MODE] = old

    def test_the_purity_hook_reads_the_real_scheduler_attribute(self):
        """THE ATTRIBUTE NAME, pinned. `getattr(..., default)` turns a wrong
        name into a feature that silently never fires; this binds against the
        name the Scheduler actually sets (`phase_policy_cfg`)."""
        import types

        from sglang.srt.managers import phase_purity

        stub = types.SimpleNamespace(
            phase_policy_cfg=_cfg(),
            _phase_purity=None,
            server_args=None,
        )
        stub._active_phase = pp.PHASE_TP
        with unittest.mock.patch.object(
            phase_purity, "_active_phase", lambda s: pp.PHASE_TP
        ):
            self.assertTrue(phase_purity.prefill_blocked_here(stub))

    def test_without_drain_mode_the_hook_defers_to_purity_as_before(self):
        import types

        from sglang.srt.managers import phase_purity

        stub = types.SimpleNamespace(phase_policy_cfg=_cfg(drain_mode=False))
        with unittest.mock.patch.object(
            phase_purity, "_active_phase", lambda s: pp.PHASE_TP
        ), unittest.mock.patch.object(
            phase_purity, "purity_of", lambda s: types.SimpleNamespace(
                prefill_allowed_in_tp=lambda: True
            )
        ):
            self.assertFalse(phase_purity.prefill_blocked_here(stub))
