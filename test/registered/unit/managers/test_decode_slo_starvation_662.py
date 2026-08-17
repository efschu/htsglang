# SPDX-License-Identifier: Apache-2.0
"""The SLO invariant: decodes are NEVER held past the SLO by a funding failure.

THE HOLE. `flip_unavailable_reason` had two causes and both are COUNTS -- a
blocking guard, or an abandon streak reaching `stand_down_after()`. The
decode-stall SLO is a TIME. Nothing bridged them, so a funding failure that
never accumulates the count could hold decode indefinitely INSIDE the bound.

Measured 2026-08-15, the 15:14 window: the seam abandoned repeatedly with the
arm rate limiter pacing retries, and the abandon-cap guard was deliberately
stood down while work was waiting -- so neither count arrived. It was harmless
only because the running batch was empty. With decodes resident that is an
unbounded stall, and the mechanism was already proven.

WHY A PURITY ESCAPE AND NOT FORCE AUTHORITY FOR THE RUNG. Granting the rung
force to release down to the live-set floor cannot satisfy this invariant: the
whole failure catalogue of that day is the rung EXECUTING and returning zero
bytes (a self-locking marker, an average-depth target, nine zero-byte shrinks).
Force over an arena with nothing releasable still returns nothing, so it fails
in exactly the case it is written for. Running decode in the layout the
instance is already in needs no memory to succeed, and purity's own modes
(`threshold:<n>`, `off`) already run decode in PP as supported configurations
-- the documented cost is throughput, never a wrong answer.
"""

import time
import types
import unittest

from sglang.srt.managers import phase_purity as pp

SLO = 45.0


def _sched(*, running_streak=0, guards=(), slo=SLO, mode="strict", refusals=0):
    rt = types.SimpleNamespace(
        blocking_guards=list(guards),
        _seam_abandons_in_a_row={"pp_to_tp": running_streak, "tp_to_pp": 0},
    )
    s = types.SimpleNamespace(
        server_args=types.SimpleNamespace(
            phase_flip_purity=mode, enable_phase_flip=True
        ),
        phase_flip_runtime=rt,
        phase_policy_cfg=types.SimpleNamespace(decode_stall_slo_s=slo),
        phase_flip_active_stack="pp",
        _phase_purity=pp.parse_purity(mode),
        # The policy's own refusal book. Group-uniform for the reason
        # flip_unavailable_reason already reads it: every rank runs the same
        # policy over the same reduced verdicts.
        phase_policy_state=types.SimpleNamespace(arm_refusals={"pp_to_tp": refusals}),
    )
    return s


#: ONE abandon. Below every bound -- stand_down_after is 4, the abandon cap is
#: 8 -- so the counts can never free decode in these cases and only the clock
#: can. That is the shape of the hole, stated as a number.
ONE_REFUSAL = 1


class TheClockOnlyRunsWhenThereIsDecodeToHold(unittest.TestCase):
    def test_an_empty_batch_does_not_start_it(self):
        """Today's window looked harmless for exactly this reason. Starting
        the clock on an empty batch would spend the SLO before a request
        arrived."""
        s = _sched()
        pp.decode_blocked_here(s, running_bs=0)
        self.assertIsNone(getattr(s, "_decode_starved_since", None))

    def test_a_held_decode_starts_it(self):
        s = _sched(running_streak=ONE_REFUSAL)
        pp.decode_blocked_here(s, running_bs=2)
        self.assertIsNotNone(getattr(s, "_decode_starved_since", None))

    def test_decode_becoming_allowed_retires_it(self):
        s = _sched()
        pp.decode_blocked_here(s, running_bs=2)
        s._phase_purity = pp.parse_purity("off")  # now allowed in PP
        pp.decode_blocked_here(s, running_bs=2)
        self.assertIsNone(getattr(s, "_decode_starved_since", None))

    def test_leaving_the_PP_phase_retires_it(self):
        s = _sched()
        pp.decode_blocked_here(s, running_bs=2)
        s.phase_flip_active_stack = "tp"
        pp.decode_blocked_here(s, running_bs=2)
        self.assertIsNone(getattr(s, "_decode_starved_since", None))


class TheInvariant(unittest.TestCase):
    """RED-FIRST FALSIFIER: a funding refusal with running bs > 0 must not
    hold decode past the SLO. Before this change nothing in either cause was
    time-based, so this could not pass."""

    def test_decode_is_held_before_the_SLO(self):
        s = _sched()
        blocked = pp.decode_blocked_here(s, running_bs=3)
        self.assertTrue(blocked, "inside the SLO, purity still governs")

    def test_decode_PROCEEDS_once_the_SLO_is_exceeded(self):
        """The invariant itself."""
        s = _sched(running_streak=ONE_REFUSAL)
        pp.decode_blocked_here(s, running_bs=3)
        s._decode_starved_since = time.monotonic() - (SLO + 0.5)
        self.assertFalse(
            pp.decode_blocked_here(s, running_bs=3),
            "decode must proceed once held past the SLO by a funding failure",
        )

    def test_it_holds_with_NO_count_REACHING_ITS_BOUND(self):
        """The exact shape of the hole: no blocking guard, and a streak that
        never reaches stand_down_after. The counts can never free decode here,
        so time alone must.

        A NON-ZERO COUNT IS NOT A COUNT THAT REACHED ITS BOUND. One refusal is
        evidence that a funding failure is happening at all -- the precondition
        the invariant is stated over -- while the BOUNDS are what the SLO
        exists to outlive."""
        s = _sched(running_streak=ONE_REFUSAL, guards=())
        s._decode_starved_since = time.monotonic() - (SLO + 0.5)
        self.assertFalse(pp.decode_blocked_here(s, running_bs=3))
        self.assertLess(
            s.phase_flip_runtime._seam_abandons_in_a_row["pp_to_tp"],
            pp.stand_down_after(),
            "the count must stay below its bound, or this proves the old path",
        )
        self.assertEqual(s.phase_flip_runtime.blocking_guards, [])

    def test_the_bound_is_SLO_plus_one_iteration_not_a_multiple(self):
        s = _sched(running_streak=ONE_REFUSAL)
        pp.decode_blocked_here(s, running_bs=3)
        s._decode_starved_since = time.monotonic() - (SLO + 0.01)
        self.assertFalse(pp.decode_blocked_here(s, running_bs=3))

    def test_an_unset_SLO_changes_nothing(self):
        """slo=0 is 'no bound stated'; the pre-existing behaviour must stand."""
        s = _sched(slo=0.0)
        s._decode_starved_since = time.monotonic() - 10_000.0
        self.assertTrue(
            pp.decode_blocked_here(s, running_bs=3),
            "with no SLO the counts govern exactly as before",
        )

    def test_the_reason_names_the_bound_and_the_wait(self):
        s = _sched()
        s._decode_starved_since = time.monotonic() - (SLO + 2.0)
        reason = pp.flip_unavailable_reason(s, "decode")
        self.assertIn("decode-stall SLO", reason)
        self.assertIn("45", reason)

    def test_prefill_is_not_relaxed_by_a_decode_stall(self):
        """Keyed on the work class, like the causes beside it. A decode stall
        says nothing about prefill and relaxing the wrong one is how a safety
        valve becomes the normal path."""
        s = _sched()
        s._decode_starved_since = time.monotonic() - (SLO + 2.0)
        self.assertIsNone(pp.flip_unavailable_reason(s, "prefill"))


class TheExistingCausesStillWork(unittest.TestCase):
    def test_a_blocking_guard_still_relaxes_without_any_clock(self):
        s = _sched(guards=("seam unfundable: tp_to_pp abandoned 8 times",))
        self.assertIsNotNone(pp.flip_unavailable_reason(s, "decode"))

    def test_no_cause_at_all_is_still_None(self):
        s = _sched()
        self.assertIsNone(pp.flip_unavailable_reason(s, "decode"))


if __name__ == "__main__":
    unittest.main()


class TheVerdictMustBeTHESAMEONEVERYRANK(unittest.TestCase):
    """THE METAL DEADLOCK, AS A TEST. Added 2026-08-15 after boot_slo_proof_r2.

    The first version of this valve started its clock from ``running_bs > 0``
    -- THIS RANK'S OWN BATCH -- and argued group uniformity from "phase, purity
    and batch are identical on every rank at that point". That is true under TP
    and FALSE UNDER PP: in a pipeline the head holds the requests and the
    downstream ranks do not see them until it forwards them, and a head that is
    holding decode never forwards. So the downstream ranks sat at running_bs 0,
    their clocks were reset on every iteration, and only rank 0 crossed the SLO.

    WHAT THAT DID, measured: one "RELAXING PURITY FOR DECODE" line on PP0
    against two "FLIP ABANDONED" lines on every rank. Rank 0 admitted a decode
    batch into the PP layout while its peers still refused decode, then blocked
    in _pp_commit_comm_work on a send nobody would receive while PP1 and PP2
    blocked in recv_requests waiting for a forward rank 0 could no longer make.
    Three ranks alive, three cards at 0%, not one decode step. A valve that
    deadlocks the instance it is rescuing is worse than the stall it prevents.

    So the property is not "the clock starts when I have work". It is "every
    rank reaches the same verdict on the same iteration".
    """

    def test_a_rank_with_an_EMPTY_batch_still_starts_its_clock(self):
        """The downstream PP rank. This is the one that was broken."""
        s = _sched(running_streak=ONE_REFUSAL)
        pp.decode_blocked_here(s, running_bs=0)
        self.assertIsNotNone(
            getattr(s, "_decode_starved_since", None),
            "a rank whose head has not forwarded the batch yet is still in a "
            "group that cannot fund decode's layout",
        )

    def test_ranks_with_DIFFERENT_batches_agree(self):
        """The head and its downstream peers, same counters, same verdict."""
        head = _sched(running_streak=ONE_REFUSAL)
        tail = _sched(running_streak=ONE_REFUSAL)
        for s, bs in ((head, 3), (tail, 0)):
            pp.decode_blocked_here(s, running_bs=bs)
            s._decode_starved_since = time.monotonic() - (SLO + 0.5)
        self.assertEqual(
            pp.decode_blocked_here(head, running_bs=3),
            pp.decode_blocked_here(tail, running_bs=0),
            "the head and the tail must relax together or the chain deadlocks",
        )

    def test_no_funding_failure_anywhere_starts_no_clock(self):
        """The counters are the signal, so zero counters means no starvation --
        decode is merely waiting for a flip that is going to happen."""
        s = _sched(running_streak=0, refusals=0)
        pp.decode_blocked_here(s, running_bs=3)
        self.assertIsNone(getattr(s, "_decode_starved_since", None))

    def test_a_policy_refusal_with_no_abandon_still_counts(self):
        """The seam streak alone CANNOT advance once the backoff engages: the
        policy declines the arm without entering the seam. Measured
        2026-08-13 -- the abandon counter froze at 3 while the policy logged
        "arm refused (7 in a row)" and work sat unrunnable."""
        s = _sched(running_streak=0, refusals=ONE_REFUSAL)
        pp.decode_blocked_here(s, running_bs=0)
        self.assertIsNotNone(getattr(s, "_decode_starved_since", None))

    def test_a_committed_flip_retires_the_clock_on_every_rank(self):
        """The streak resets on a committed cutover, so the clock does too --
        without anything having to remember to clear it."""
        s = _sched(running_streak=ONE_REFUSAL)
        pp.decode_blocked_here(s, running_bs=0)
        self.assertIsNotNone(getattr(s, "_decode_starved_since", None))
        s.phase_flip_runtime._seam_abandons_in_a_row["pp_to_tp"] = 0
        s.phase_policy_state.arm_refusals["pp_to_tp"] = 0
        pp.decode_blocked_here(s, running_bs=0)
        self.assertIsNone(getattr(s, "_decode_starved_since", None))
