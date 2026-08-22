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
        decision, _ = _drive(cfg, [(0.0, 400_000, 4), (60.0, 400_000, 0)])
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
                phase=pp.PHASE_PP,
                pending_prefill_tokens=pending,
                running_bs=bs,
                now=now,
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
                phase=pp.PHASE_PP,
                pending_prefill_tokens=403_779,
                running_bs=4,
                now=now,
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
        with (
            unittest.mock.patch.object(
                phase_purity, "_active_phase", lambda s: pp.PHASE_TP
            ),
            unittest.mock.patch.object(
                phase_purity,
                "purity_of",
                lambda s: types.SimpleNamespace(prefill_allowed_in_tp=lambda: True),
            ),
        ):
            self.assertFalse(phase_purity.prefill_blocked_here(stub))


class SuppressionYieldsWhenTheFlipCannotBeFunded(unittest.TestCase):
    """LIVE INCIDENT 2026-08-16 06:47:48, and the defect was in hot fix 2.

    The policy armed tp_to_pp every ~3 s and CorridorGuard refused the seam
    staging on two ranks with static numbers:

        PP1  want 2163 MiB, free 2456, arming floor 1536  -> 293 short
        PP2  want 2858 MiB, free 3560, arming floor 1536  -> 702 short

    The want (2163-2858) EXCEEDS the arming floor (1536): the floor guarantees
    less than a 4-carrier bundle's seam actually needs, because staging scales
    with live KV cells. Every degradation already stood down -- abandon cap,
    backoff, entry margin -- and it still could not fund. 76 refusals in a row.

    THAT ALONE WOULD HAVE BEEN A SLOW BOOT. What made it a TOTAL wedge is hot
    fix 2: before it, a refused tp_to_pp still degraded to prefilling in the TP
    layout, so the backlog drained slowly instead of not at all. Suppressing
    prefill in TP removed the fallback, so an unfundable seam became an idle
    server with 727004 tokens waiting.

    The rule hot fix 1 was built on applies to hot fix 2 and I did not apply
    it: a failure must degrade to the fallback, never to a wedge. So the
    suppression YIELDS once the flip has proved it cannot be funded -- the same
    shape, and the same threshold, as the seam entry margin's own
    "YIELDED after 2 consecutive abandoned attempts".

    SUPERSEDED BY THE 07:02 WEDGE, and kept because the reasoning survived
    even though the mechanism did not. The refusal COUNTER was the wrong key
    -- see `TheValveOutranksDrainModeSuppression` -- but the rule it encoded,
    that suppression must not outlive the reachability of the layout it is
    waiting for, is the rule that fix keeps. These cases now assert the rule
    through its final key.
    """

    def test_suppression_holds_while_the_flip_is_reachable(self):
        self.assertTrue(
            pp.prefill_suppressed_in_tp(_cfg(), pp.PHASE_TP, flip_unavailable=False)
        )

    def test_it_yields_once_the_seam_cannot_be_entered(self):
        """THE INCIDENT, through the key that catches both of its shapes."""
        self.assertFalse(
            pp.prefill_suppressed_in_tp(_cfg(), pp.PHASE_TP, flip_unavailable=True)
        )

    def test_a_reachable_flip_restores_suppression(self):
        """Not a latch: the valve closes by itself on the first commit, so a
        rig that recovers returns to the user's semantics with nothing having
        to remember to restore it."""
        self.assertTrue(
            pp.prefill_suppressed_in_tp(_cfg(), pp.PHASE_TP, flip_unavailable=False)
        )


class TheValveOutranksDrainModeSuppression(unittest.TestCase):
    """LIVE WEDGE #3, 2026-08-16 07:02, on my own deploy ed7df3ec93.

    THIRD SHAPE, SAME TRAP, AND THIS TIME MY FIX WAS THE CAUSE. tp_to_pp
    could not enter: PP2 yielded its entry margin while PP1 WITHHELD the
    yield (a measured 710 MiB draw against 2522 MiB free predicts an 864 MiB
    trough, under the 1024 MiB law). The ranks disagreed and "consecutive
    delayed attempts" climbed 15/16/17 with no exit, because the withhold
    emits a DELAY, which is deliberately exempt from the stand-down cap.

    My hot-fix-2 yield never engaged: it counted `arm_refusals`, and these
    were DELAYS. A different counter, the same wedge -- the fourth time this
    chain has shipped a path that was not told its state.

    BUT THE VALVE WAS ALREADY RIGHT. `flip_unavailable_reason` reads BOTH
    books -- `_seam_abandons_in_a_row` (which the delays DO advance, to 17)
    and `arm_refusals` -- against a bound of 4. It would have opened, and the
    documented degradation would have run. It never got the chance, because
    hot fix 2 was checked BEFORE `prefill_allowed_in_tp` and returned True
    first. I made the valve a dead letter, which is why the WITHHELD line's
    promise that "the purity valve lets the starved work class run meanwhile"
    was false on metal: bs 0, GPU 0%, zero prefill batches.

    So suppression now defers to the valve instead of outranking it. Drain
    mode's premise is "prefill belongs in PP, so wait for PP" -- valid only
    while PP is REACHABLE. When the seam cannot enter, the premise is false.

    REQUIREMENT (b) IS SUPERSEDED, not dropped: the user's 2026-08-16
    decision removes the withhold entirely (warn-and-proceed), so the seam no
    longer needs to verify the valve's state -- there is no unbounded wait
    left to justify. The valve still has to be REAL, because point 1 leaves
    transient delays and bs 0 must never mean an idle box.

    ONE BOUND, NOT TWO. This deletes DRAIN_SUPPRESSION_YIELD_AFTER. Keying on
    the valve inherits its bound of 4 and its two books, so the refusal shape
    (wedge #2, 76 refusals) and the delay shape (wedge #3, 17 delays) exit
    through the SAME door. A second threshold of my own was one more counter
    that could be told the wrong thing.
    """

    def test_suppression_holds_while_the_flip_is_reachable(self):
        self.assertTrue(
            pp.prefill_suppressed_in_tp(_cfg(), pp.PHASE_TP, flip_unavailable=False)
        )

    def test_it_yields_the_moment_the_flip_is_unreachable(self):
        self.assertFalse(
            pp.prefill_suppressed_in_tp(_cfg(), pp.PHASE_TP, flip_unavailable=True)
        )

    def test_my_competing_threshold_is_gone(self):
        """One bound, and it is the valve's."""
        self.assertFalse(hasattr(pp, "DRAIN_SUPPRESSION_YIELD_AFTER"))

    def test_the_delay_shape_opens_the_valve_through_the_hook(self):
        """THE WEDGE. 17 consecutive DELAYS, zero refusals -- the shape my
        refusal counter was blind to -- must let the TP layout prefill."""
        import types

        from sglang.srt.managers import phase_purity

        rt = types.SimpleNamespace(
            _seam_abandons_in_a_row={"tp_to_pp": 17}, blocking_guards=()
        )
        sched = types.SimpleNamespace(
            phase_policy_cfg=_cfg(),
            phase_policy_state=pp.PhasePolicyState(arm_refusals={}),
            phase_flip_runtime=rt,
            server_args=None,
            _phase_purity=None,
        )
        with unittest.mock.patch.object(
            phase_purity, "_active_phase", lambda s: pp.PHASE_TP
        ):
            self.assertFalse(
                phase_purity.prefill_blocked_here(sched),
                "17 delays must open the valve; only refusals were counted",
            )

    def test_the_refusal_shape_still_opens_it(self):
        """Wedge #2 must not regress now that the counter changed."""
        import types

        from sglang.srt.managers import phase_purity

        rt = types.SimpleNamespace(_seam_abandons_in_a_row={}, blocking_guards=())
        sched = types.SimpleNamespace(
            phase_policy_cfg=_cfg(),
            phase_policy_state=pp.PhasePolicyState(arm_refusals={pp.TP_TO_PP: 76}),
            phase_flip_runtime=rt,
            server_args=None,
            _phase_purity=None,
        )
        with unittest.mock.patch.object(
            phase_purity, "_active_phase", lambda s: pp.PHASE_TP
        ):
            self.assertFalse(phase_purity.prefill_blocked_here(sched))

    def test_a_reachable_flip_still_suppresses_through_the_hook(self):
        """The user's semantics are the DEFAULT and must survive all this."""
        import types

        from sglang.srt.managers import phase_purity

        rt = types.SimpleNamespace(_seam_abandons_in_a_row={}, blocking_guards=())
        sched = types.SimpleNamespace(
            phase_policy_cfg=_cfg(),
            phase_policy_state=pp.PhasePolicyState(arm_refusals={}),
            phase_flip_runtime=rt,
            server_args=None,
            _phase_purity=None,
        )
        with unittest.mock.patch.object(
            phase_purity, "_active_phase", lambda s: pp.PHASE_TP
        ):
            self.assertTrue(phase_purity.prefill_blocked_here(sched))


class SuppressionNeedsABundleToProtect(unittest.TestCase):
    """LIVE WEDGE #4, 2026-08-16 08:03, and again my own fix caused it.

    An 18-token request hung for two minutes with GPU at 0% while the policy
    logged, every ten seconds:

        holding in tp: pending prefill 18 tok <= N=7004, running it in tp
        (pending prefill 18 tok, running bs 0)

    THE POLICY WAS RIGHT AND THE SUPPRESSION OVERRODE IT. Below the break-even
    N a flip costs more than it saves, so the policy deliberately keeps the
    work in TP and arms no flip. The purity valve cannot rescue that: the flip
    is not FAILING, it was never asked for, so `flip_unavailable` is false and
    suppression held a request that nothing was ever going to run.

    THE 4-CARRIER PROOFS HID IT. A 25625-token prompt sits far above N=7004,
    so it goes to PP and prefills there; every metal run so far used long
    prompts. Short requests -- the common case on this box -- could not be
    served at all.

    THE RULE. Drain mode exists to stop a TP window admitting the work it was
    entered to escape, and that window is defined by a decode bundle in
    flight. With `running_bs == 0` the bundle is finished, there is nothing to
    drain, and the only thing suppression can still do is idle the instance.
    """

    def test_a_running_bundle_is_still_protected(self):
        self.assertTrue(pp.prefill_suppressed_in_tp(_cfg(), pp.PHASE_TP, running_bs=4))
        self.assertTrue(pp.prefill_suppressed_in_tp(_cfg(), pp.PHASE_TP, running_bs=1))

    def test_an_empty_bundle_suppresses_nothing(self):
        """THE WEDGE."""
        self.assertFalse(pp.prefill_suppressed_in_tp(_cfg(), pp.PHASE_TP, running_bs=0))

    def test_an_unmeasured_bundle_is_not_read_as_empty(self):
        """-1 means the caller did not measure it. An unmeasured input must
        never become a licence -- that is how the drain contract would quietly
        stop applying wherever a call site forgot to pass the count."""
        self.assertTrue(pp.prefill_suppressed_in_tp(_cfg(), pp.PHASE_TP, running_bs=-1))

    def test_the_scheduler_passes_the_count(self):
        """THE CALL SITE, for the fourth time in this file. A condition the
        hook is never told about is a condition that never fires."""
        import inspect as _inspect

        from sglang.srt.managers import scheduler as _sched

        src = _inspect.getsource(_sched.Scheduler.get_next_batch_to_run)
        self.assertIn("phase_prefill_blocked_here(", src)
        self.assertIn("running_bs=", src)

    def test_the_hook_forwards_it(self):
        import inspect as _inspect

        from sglang.srt.managers import phase_purity

        src = _inspect.getsource(phase_purity.prefill_blocked_here)
        self.assertIn("running_bs=running_bs", src)
