"""#759: the IDLE-LOCK escape must not fund a 3.1s flip on a tiny backlog.

comp4 Gate A failed 3 of 3: 8 small requests, 2 armings (a PASS requires 0).
The specimen names both, from ``boot_735_comp4.log``:

    06:32:37  arming tp_to_pp: IDLE-LOCKED ... (0 req resident, 68 tok pending)
    06:32:50  arming pp_to_tp: IDLE-LOCKED ... (1 req resident,  0 tok pending)

A flip out and straight back, 13 s apart, funded by 68 tokens and then by
nothing at all.

WHICH CANDIDATE IT IS, checked rather than assumed:

* (a) inflated backlog -- NO. #731's dedup (``fdcf837206``) is an ancestor of
  this base, and the numbers in the specimen are small and plausible (68, 0),
  not doubled.
* (b) flip-cost term ignoring #704a's constants -- NO, and it cannot be:
  ``PHASE-POLICY arming ... IDLE-LOCKED`` never reaches the economic window.
  The branch at ``phase_policy.py:1653`` returns a decision before any cost
  term is consulted.
* (c) missing minimum-backlog floor -- YES, on the ESCAPE HATCH rather than on
  the economic window. ``if inp.nothing_can_run and inp.target_can_admit`` arms
  unconditionally, "arming immediately rather than waiting for a stall timer to
  notice" (its own comment).

Also checked and NOT the root here: that branch sits ABOVE the ``min_dwell``
gate, so it does bypass the thrash bound whose comment claims "no branch below
can bypass" it. Real, and filed -- but the two armings are 13 s apart against a
10 s dwell, so honouring the dwell would not have stopped either. The floor is
the root; the bypass is a separate latent hole.

THE FIX, using constants the module already derives rather than a new one:

* a backlog at or above ``flip_tokens`` -- the measured break-even -- still
  arms IMMEDIATELY. That is Gate B and it must not die.
* a smaller-but-nonzero backlog arms only once the lock has PERSISTED for
  ``idle_dwell_s``. A transient gap between two small requests is not a lock,
  and treating it as one is what produced the ping-pong.
* nothing pending and nothing resident never arms: there is no work a flip
  could serve.

The escape survives -- a genuinely stuck box still gets out, just not on the
first transient sample.
"""

import unittest

from sglang.srt.managers.phase_policy import IDLE_LOCKED
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

FLIP_TOKENS = 4096


def _cfg(**over):
    from sglang.srt.managers.phase_policy import PhasePolicyConfig

    base = dict(
        enabled=True,
        flip_tokens=FLIP_TOKENS,
        min_dwell_s=10.0,
        idle_dwell_s=3.0,
    )
    base.update(over)
    return PhasePolicyConfig(**base)


def _decide(*, phase, pending, running_bs, idle_since, now=1000.0, cfg=None):
    """Drive the idle-lock branch with the specimen's own shape."""
    from sglang.srt.managers.phase_policy import (
        PhasePolicyInputs,
        PhasePolicyState,
        decide,
    )

    inp = PhasePolicyInputs(
        phase=phase,
        now=now,
        running_bs=running_bs,
        pending_prefill_tokens=pending,
        nothing_can_run=True,
        target_can_admit=True,
    )
    state = PhasePolicyState(idle_since=idle_since)
    return decide(cfg or _cfg(), state, inp)


def _is_idle_lock_arm(d):
    return d.direction is not None and (d.reason or "").startswith(IDLE_LOCKED)


class TestGateAEightSmallRequests(CustomTestCase):
    """RED-FIRST: the 8-small-request scenario must arm 0x."""

    def test_the_68_token_specimen_does_not_arm_immediately(self):
        """06:32:37, verbatim: 0 resident, 68 pending, lock just observed."""
        d = _decide(phase="tp", pending=68, running_bs=0, idle_since=999.5)
        self.assertFalse(
            _is_idle_lock_arm(d),
            f"68 tokens must not buy a 3.1s flip: {d.reason!r}",
        )

    def test_the_zero_pending_specimen_does_not_arm_on_a_TRANSIENT_lock(self):
        """06:32:50, verbatim: 1 resident, 0 pending, lock just observed.

        My first version asserted "zero pending never arms". #689's own
        fixture refuted that: it expresses real queued work with
        ``queue_nonempty=True`` while ``pending_prefill_tokens`` is 0, so a
        work-COUNT rule would refuse a genuine deadlock escape. The qualifier
        is PERSISTENCE, and this is the transient case.
        """
        d = _decide(phase="pp", pending=0, running_bs=1, idle_since=999.5)
        self.assertFalse(
            _is_idle_lock_arm(d),
            f"a lock observed 0.5s ago is not a lock: {d.reason!r}",
        )

    def test_nothing_pending_on_a_transient_lock_does_not_arm(self):
        d = _decide(phase="pp", pending=0, running_bs=0, idle_since=999.5)
        self.assertFalse(_is_idle_lock_arm(d))

    def test_a_transient_lock_below_the_floor_is_not_a_lock(self):
        """The mechanism: a gap between two small requests is not a wedge."""
        d = _decide(phase="tp", pending=68, running_bs=0, idle_since=999.9)
        self.assertFalse(_is_idle_lock_arm(d))
        self.assertIn("idle", (d.reason or "").lower())


class TestGateBMustNotDie(CustomTestCase):
    """CAN-FAIL: the real 72k backlog must STILL arm, and immediately."""

    def test_the_72k_backlog_arms_immediately(self):
        d = _decide(phase="tp", pending=72_000, running_bs=0, idle_since=None)
        self.assertTrue(
            _is_idle_lock_arm(d),
            f"Gate B died: a 72k backlog must still escape an idle lock: {d.reason!r}",
        )

    def test_exactly_the_break_even_backlog_arms(self):
        d = _decide(phase="tp", pending=FLIP_TOKENS, running_bs=0, idle_since=None)
        self.assertTrue(_is_idle_lock_arm(d))

    def test_one_token_below_the_break_even_waits_for_the_dwell(self):
        """The boundary, both sides -- so the floor is a floor, not a fence."""
        d = _decide(
            phase="tp", pending=FLIP_TOKENS - 1, running_bs=0, idle_since=999.9
        )
        self.assertFalse(_is_idle_lock_arm(d))


class TestTheEscapeStillEscapes(CustomTestCase):
    """A genuinely stuck box must still get out -- the #748 failure direction.

    Refusing forever would trade Gate A's thrash for a wedge, which is the
    trade this fix must NOT make.
    """

    def test_a_persistent_small_backlog_arms_after_the_idle_dwell(self):
        d = _decide(phase="tp", pending=68, running_bs=0, idle_since=990.0)
        self.assertTrue(
            _is_idle_lock_arm(d),
            f"a PERSISTENT lock must still escape: {d.reason!r}",
        )

    def test_the_escape_reason_says_it_waited(self):
        d = _decide(phase="tp", pending=68, running_bs=0, idle_since=990.0)
        self.assertIn("idle", (d.reason or "").lower())

    def test_an_UNOBSERVED_lock_is_never_delayed(self):
        """#689's invariant, preserved rather than amended.

        With no idle stamp there is no evidence the lock is transient.
        Delaying on absence of evidence would turn an unproven suspicion into
        a wedge -- the #748 direction -- so "a layout that can run NOTHING
        leaves at once" still governs. This is the case
        ``test_window_formation_689.py::ItNeverHoldsTheIdleLockedArm`` pins,
        and that pin was NOT weakened to accommodate this fix.
        """
        d = _decide(phase="pp", pending=0, running_bs=1, idle_since=None)
        self.assertTrue(
            _is_idle_lock_arm(d),
            f"an unobserved lock must not be delayed: {d.reason!r}",
        )


if __name__ == "__main__":
    unittest.main()
