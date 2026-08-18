"""#748 REFAIL: the IDLE-LOCK escape was fed a false premise, and its damper was blind.

THE SPECIMEN. ``boot_735_nohc.log`` (hicache off, PP=3/TP vector 32,16,16,
``--phase-flip-purity prefill_in_tp``), 2026-08-18 08:37:40 -> 08:54:42:
160 IDLE-LOCK armings in 1022 s -- 9.4 per minute -- in a strict alternation:

    08:39:35  arming tp_to_pp: IDLE-LOCKED (0 req resident, 163 tok pending)
    08:39:39  arming pp_to_tp: IDLE-LOCKED (1 req resident,   0 tok pending)

Zero of the 160 were economic arms. The distribution is only two shapes: 80x
``tp`` with 0 resident and 57-817 tok pending, 79x ``pp`` with 1 resident and
0 pending.

WHY #748 (1cc0d24ae7 / 256fe09fab) AND #759 (72c1ed9c18) DID NOT CLOSE IT.
Both fixed how the POLICY prices the escape. The escape was not mispriced.

(1) THE PREMISE WAS FALSE -- the root.

``Scheduler._layout_admits`` hardcoded "TP may only decode", a sentence true
only under ``strict`` purity, and never consulted the boot's actual rule. On
the 80 ``tp`` armings it early-returned False on ``running_bs <= 0``, so
``nothing_can_run`` was True and the #688 escape fired. The SAME log carries,
seconds apart:

    76x  PHASE-POLICY holding in tp: pending prefill 163 tok <= N=7004,
         running it in tp
    42x  Prefill batch phase=tp

so the config allowed TP prefill, the policy wanted TP prefill, and TP
executed TP prefill -- while the simulation that gates the escape reported TP
could run nothing at all. ``prefill_suppressed_in_tp`` lifts drain-mode
suppression outright at ``running_bs == 0``, which is exactly the specimen's
state, so the batch builder and the simulation gave opposite answers to one
question.

(2) THE DAMPER COULD NOT SEE THE LOCK -- why the second fix was inert.

#759 named PERSISTENCE as the qualifier and read it off ``state.idle_since``,
which ``observe_idle`` stamps only while ``running_bs == 0 and pending == 0``.
An idle lock is by definition the opposite: work exists and this layout cannot
run it. Of the 160 armings, ZERO had (0 resident, 0 pending), and all 160 log
lines carry the "no idle observation" branch. #759's own fixtures pass
``idle_since=`` by hand, so the desk was green while the clock was never
stamped on metal -- the delay was unreachable for the entire class of lock it
was written for.

THE FIX, in two parts matching the two findings:

* ``_layout_admits`` asks the purity rule which classes each layout may run,
  through the same oracle the batch builder uses. Absent a purity rule it
  keeps today's answer, so nothing that does not run phase flip moves.
* a LOCK clock, ``nothing_can_run_since``, stamped by ``observe_idle`` from
  the term the escape is keyed on, with ``idle_since`` retained as the
  fallback so every #759 fixture keeps its exact verdict.

#689's invariant is untouched in both parts: an unobserved lock still leaves
at once, and a backlog at or above the break-even still arms immediately.
"""

import unittest
from types import SimpleNamespace

from sglang.srt.managers.phase_policy import IDLE_LOCKED
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

# The specimen's own numbers, verbatim from boot_735_nohc.log.
TP_PENDING = 163  # 08:39:35 arming tp_to_pp, 0 req resident
TP_PENDING_MAX = 817  # the largest backlog that armed on the tp side
PP_RESIDENT = 1  # 08:39:39 arming pp_to_tp, 0 tok pending
ROWS_AVAIL = 434_146  # POOL CENSUS post-cutover, same second
MAMBA_SLOTS = 11  # 12 total, one held by the carrier
DRAFT_TOKENS = 4  # --speculative-num-draft-tokens 4
CHUNK = 512  # --chunked-prefill-size 512
FLIP_TOKENS = 7004  # the break-even the specimen lines print


def _purity(mode="prefill_in_tp"):
    from sglang.srt.managers.phase_purity import parse_purity

    return parse_purity(mode)


def _policy_cfg(**over):
    from sglang.srt.managers.phase_policy import PhasePolicyConfig

    base = dict(
        enabled=True,
        flip_tokens=FLIP_TOKENS,
        min_dwell_s=10.0,
        idle_dwell_s=3.0,
    )
    base.update(over)
    return PhasePolicyConfig(**base)


def _sched(
    purity="prefill_in_tp",
    avail=ROWS_AVAIL,
    slots=MAMBA_SLOTS,
    evictable=0,
    policy_cfg=None,
):
    """A Scheduler shell carrying only what the simulation reads.

    Deliberately the #713 fixture plus the two attributes the boot actually
    had -- a resolved purity rule and a policy config. #713's own fixture omits
    both, and that test file pins the omitted case as unchanged.
    """
    from sglang.srt.managers.scheduler import Scheduler

    s = Scheduler.__new__(Scheduler)
    s.server_args = SimpleNamespace(
        chunked_prefill_size=CHUNK,
        speculative_num_draft_tokens=DRAFT_TOKENS,
    )
    s.token_to_kv_pool_allocator = SimpleNamespace(available_size=lambda: avail)
    s.tree_cache = SimpleNamespace(full_evictable_size=lambda: evictable)
    s.req_to_token_pool = SimpleNamespace(
        mamba_allocator=SimpleNamespace(available_size=lambda: slots)
    )
    s._phase_purity = None if purity is None else _purity(purity)
    s.phase_policy_cfg = policy_cfg if policy_cfg is not None else _policy_cfg()
    return s


def _terms(s, phase, running_bs, pending):
    s._round_built_nothing = True
    s.phase_flip_active_stack = phase
    return s._idle_locked_inputs(running_bs, pending)


class TestTheTpPremiseWasFalse748(CustomTestCase):
    """RED-FIRST: the 80 tp-side armings must not be proposable at all."""

    def test_tp_admits_the_specimen_prefill(self):
        """08:39:35 verbatim: 0 resident, 163 pending, purity prefill_in_tp.

        The log proves TP could: 42 ``Prefill batch phase=tp`` lines in the
        same boot, and the policy's own ``running it in tp`` verdict 76 times.
        """
        self.assertTrue(
            _sched()._layout_admits("tp", 0, TP_PENDING),
            "the boot allowed prefill in tp and TP executed 42 of them; the "
            "simulation that gates the escape must not say otherwise",
        )

    def test_the_largest_tp_specimen_also_admits(self):
        self.assertTrue(_sched()._layout_admits("tp", 0, TP_PENDING_MAX))

    def test_the_tp_specimen_no_longer_reads_as_nothing_can_run(self):
        """The escape's input, on the specimen's exact numbers."""
        nothing_can_run, _ = _terms(_sched(), "tp", 0, TP_PENDING)
        self.assertFalse(
            nothing_can_run,
            "tp could prefill 163 tok, so 'nothing can run in tp' is false and "
            "the #688 escape has no premise",
        )

    def test_the_policy_therefore_does_not_arm_on_the_tp_specimen(self):
        from sglang.srt.managers.phase_policy import (
            PhasePolicyInputs,
            PhasePolicyState,
            decide,
        )

        s = _sched()
        nothing_can_run, target_can_admit = _terms(s, "tp", 0, TP_PENDING)
        d = decide(
            _policy_cfg(),
            PhasePolicyState(),
            PhasePolicyInputs(
                phase="tp",
                now=1000.0,
                running_bs=0,
                pending_prefill_tokens=TP_PENDING,
                nothing_can_run=nothing_can_run,
                target_can_admit=target_can_admit,
            ),
        )
        self.assertFalse(
            (d.reason or "").startswith(IDLE_LOCKED),
            f"the 9.4/min churn entered here: {d.reason!r}",
        )


class TestTheEscapeStillEscapes748(CustomTestCase):
    """CAN-FAIL COUNTERWEIGHTS. Every way the escape must survive."""

    def test_strict_purity_still_refuses_tp_prefill(self):
        """The sentence in the old docstring is true -- for ONE mode."""
        self.assertFalse(_sched(purity="strict")._layout_admits("tp", 0, TP_PENDING))

    def test_strict_purity_still_arms_the_escape(self):
        nothing_can_run, target_can_admit = _terms(
            _sched(purity="strict"), "tp", 0, TP_PENDING
        )
        self.assertTrue(nothing_can_run)
        self.assertTrue(target_can_admit, "pp can prefill it")

    def test_a_scheduler_without_a_purity_rule_is_unchanged(self):
        """#713's fixture shape: absent evidence is not a licence."""
        self.assertFalse(_sched(purity=None)._layout_admits("tp", 0, TP_PENDING))

    def test_a_starved_pool_still_refuses_tp_prefill(self):
        self.assertFalse(
            _sched(avail=0, evictable=0)._layout_admits("tp", 0, TP_PENDING)
        )

    def test_no_state_slot_still_refuses_tp_prefill(self):
        self.assertFalse(_sched(slots=0)._layout_admits("tp", 0, TP_PENDING))

    def test_drain_mode_with_a_live_bundle_still_refuses_tp_prefill(self):
        """The #677 hot-fix-2 contract: a TP window finishing a bundle must not
        admit the work it was entered to escape. Only ``running_bs == 0`` lifts
        it, and that is the specimen's state -- not this one's."""
        s = _sched(policy_cfg=_policy_cfg(drain_mode=True))
        self.assertFalse(s._layout_admits("tp", 2, TP_PENDING) is False)  # decode arm
        s2 = _sched(avail=1, policy_cfg=_policy_cfg(drain_mode=True))
        self.assertFalse(
            s2._layout_admits("tp", 2, TP_PENDING),
            "with a bundle in flight and no rows, tp admits nothing",
        )

    def test_drain_mode_lifts_suppression_at_zero_bundle(self):
        s = _sched(policy_cfg=_policy_cfg(drain_mode=True))
        self.assertTrue(s._layout_admits("tp", 0, TP_PENDING))

    def test_tp_still_decodes_when_nothing_is_pending(self):
        self.assertTrue(_sched()._layout_admits("tp", 2, 0))

    def test_the_pp_side_specimen_still_arms(self):
        """08:39:39 verbatim: 1 resident, 0 pending, in pp.

        This one is a REAL lock -- prefill_in_tp forbids decode in PP and there
        is no prefill to do -- so it must keep escaping. Fixing the tp side
        removes its cause, not its correctness.
        """
        nothing_can_run, target_can_admit = _terms(_sched(), "pp", PP_RESIDENT, 0)
        self.assertTrue(nothing_can_run, "pp cannot decode under prefill_in_tp")
        self.assertTrue(target_can_admit, "tp can decode the carrier")

    def test_purity_off_lets_pp_decode(self):
        """The mirror of the tp defect: under ``off`` both prohibitions are
        lifted, so PP holding a decodable carrier is not a lock either."""
        self.assertTrue(_sched(purity="off")._layout_admits("pp", PP_RESIDENT, 0))
        self.assertFalse(_sched(purity="strict")._layout_admits("pp", PP_RESIDENT, 0))


class TestTheDamperCouldNotSeeTheLock748(CustomTestCase):
    """RED-FIRST on finding (2): #759's clock is never stamped on this path."""

    def _observe(self, **kw):
        from sglang.srt.managers.phase_policy import (
            PhasePolicyInputs,
            PhasePolicyState,
            observe_idle,
        )

        state = PhasePolicyState()
        inp = PhasePolicyInputs(**kw)
        observe_idle(state, inp)
        return state

    def test_the_idle_clock_is_not_stamped_by_the_specimen(self):
        """The whole reason #759 was inert, as an assertion."""
        state = self._observe(
            phase="tp",
            now=1000.0,
            running_bs=0,
            pending_prefill_tokens=TP_PENDING,
            nothing_can_run=True,
            target_can_admit=True,
        )
        self.assertIsNone(
            state.idle_since,
            "idle_since needs an EMPTY box; a lock is a box with work it "
            "cannot run, so #759's qualifier could never fire here",
        )

    def test_the_lock_clock_is_stamped_by_the_specimen(self):
        state = self._observe(
            phase="tp",
            now=1000.0,
            running_bs=0,
            pending_prefill_tokens=TP_PENDING,
            nothing_can_run=True,
            target_can_admit=True,
        )
        self.assertEqual(state.nothing_can_run_since, 1000.0)

    def test_the_lock_clock_clears_when_the_lock_does(self):
        from sglang.srt.managers.phase_policy import PhasePolicyInputs, observe_idle

        state = self._observe(
            phase="pp",
            now=1000.0,
            running_bs=PP_RESIDENT,
            pending_prefill_tokens=0,
            nothing_can_run=True,
            target_can_admit=True,
        )
        self.assertEqual(state.nothing_can_run_since, 1000.0)
        observe_idle(
            state,
            PhasePolicyInputs(
                phase="pp",
                now=1001.0,
                running_bs=PP_RESIDENT,
                pending_prefill_tokens=200,
                nothing_can_run=False,
                target_can_admit=True,
            ),
        )
        self.assertIsNone(state.nothing_can_run_since)

    def test_a_freshly_seen_lock_below_break_even_is_not_armed(self):
        """OBSERVE THEN DECIDE, which is the order the scheduler uses and the
        order #759's fixtures never exercised."""
        from sglang.srt.managers.phase_policy import (
            PhasePolicyInputs,
            PhasePolicyState,
            decide,
            observe_idle,
        )

        cfg = _policy_cfg()
        state = PhasePolicyState()
        inp = PhasePolicyInputs(
            phase="pp",
            now=1000.0,
            running_bs=PP_RESIDENT,
            pending_prefill_tokens=0,
            nothing_can_run=True,
            target_can_admit=True,
        )
        observe_idle(state, inp)
        d = decide(cfg, state, inp)
        self.assertFalse(
            (d.reason or "").startswith(IDLE_LOCKED) and d.direction is not None,
            f"a lock seen for 0.0s is not a lock: {d.reason!r}",
        )

    def test_a_persistent_lock_below_break_even_still_escapes(self):
        """#689's guarantee. The delay is bounded, never a refusal."""
        from sglang.srt.managers.phase_policy import (
            PhasePolicyInputs,
            PhasePolicyState,
            decide,
            observe_idle,
        )

        cfg = _policy_cfg()
        state = PhasePolicyState()
        for t in (1000.0, 1004.0):
            inp = PhasePolicyInputs(
                phase="pp",
                now=t,
                running_bs=PP_RESIDENT,
                pending_prefill_tokens=0,
                nothing_can_run=True,
                target_can_admit=True,
            )
            observe_idle(state, inp)
            d = decide(cfg, state, inp)
        self.assertTrue(
            d.direction is not None and (d.reason or "").startswith(IDLE_LOCKED),
            f"a lock held 4.0s against a 3s dwell must escape: {d.reason!r}",
        )

    def test_an_unobserved_lock_still_arms_immediately(self):
        """#689's invariant, unmodified: no observation is not evidence of
        transience, so the escape is not delayed."""
        from sglang.srt.managers.phase_policy import (
            PhasePolicyInputs,
            PhasePolicyState,
            decide,
        )

        d = decide(
            _policy_cfg(),
            PhasePolicyState(),
            PhasePolicyInputs(
                phase="pp",
                now=1000.0,
                running_bs=PP_RESIDENT,
                pending_prefill_tokens=0,
                nothing_can_run=True,
                target_can_admit=True,
            ),
        )
        self.assertTrue(d.direction is not None)
        self.assertIn("no lock observation", d.reason or "")

    def test_gate_b_a_real_backlog_still_arms_immediately(self):
        """Above the break-even the persistence question is never asked."""
        from sglang.srt.managers.phase_policy import (
            PhasePolicyInputs,
            PhasePolicyState,
            decide,
            observe_idle,
        )

        cfg = _policy_cfg()
        state = PhasePolicyState()
        inp = PhasePolicyInputs(
            phase="tp",
            now=1000.0,
            running_bs=0,
            pending_prefill_tokens=72_000,
            nothing_can_run=True,
            target_can_admit=True,
        )
        observe_idle(state, inp)
        d = decide(cfg, state, inp)
        self.assertTrue(
            d.direction is not None and (d.reason or "").startswith(IDLE_LOCKED),
            f"Gate B died: a 72k backlog must escape at once: {d.reason!r}",
        )

    def test_a_completed_cutover_clears_the_lock_clock(self):
        """A lock measured in a layout that no longer exists is not evidence."""
        from sglang.srt.managers.phase_policy import (
            PhasePolicyState,
            note_flip_completed,
        )

        state = PhasePolicyState(nothing_can_run_since=1000.0)
        note_flip_completed(_policy_cfg(), state, "pp_to_tp", 1005.0)
        self.assertIsNone(state.nothing_can_run_since)


if __name__ == "__main__":
    unittest.main()
