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
from unittest import mock

from sglang.srt.managers import phase_purity as pp

SLO = 45.0


def _sched(*, running_streak=0, guards=(), slo=SLO, mode="strict"):
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
    )
    return s


class TheClockOnlyRunsWhenThereIsDecodeToHold(unittest.TestCase):
    def test_an_empty_batch_does_not_start_it(self):
        """Today's window looked harmless for exactly this reason. Starting
        the clock on an empty batch would spend the SLO before a request
        arrived."""
        s = _sched()
        pp.decode_blocked_here(s, running_bs=0)
        self.assertIsNone(getattr(s, "_decode_starved_since", None))

    def test_a_held_decode_starts_it(self):
        s = _sched()
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
        s = _sched()
        pp.decode_blocked_here(s, running_bs=3)
        s._decode_starved_since = time.monotonic() - (SLO + 0.5)
        self.assertFalse(
            pp.decode_blocked_here(s, running_bs=3),
            "decode must proceed once held past the SLO by a funding failure",
        )

    def test_it_holds_with_NO_count_ever_arriving(self):
        """The exact shape of the hole: no blocking guard, no abandon streak.
        Both counts stay at zero for ever and time alone must free decode."""
        s = _sched(running_streak=0, guards=())
        s._decode_starved_since = time.monotonic() - (SLO + 0.5)
        self.assertFalse(pp.decode_blocked_here(s, running_bs=3))
        self.assertEqual(s.phase_flip_runtime._seam_abandons_in_a_row["pp_to_tp"], 0)
        self.assertEqual(s.phase_flip_runtime.blocking_guards, [])

    def test_the_bound_is_SLO_plus_one_iteration_not_a_multiple(self):
        s = _sched()
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
