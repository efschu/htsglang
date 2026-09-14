# SPDX-License-Identifier: Apache-2.0
"""#1378 xsn36/37 (coordinator requirements (a)+(c)): the collect's wait is
LIVENESS-coupled and sits OUTSIDE the card flock.

THE WALLS THIS PINS, both measured:
* weg2xsn35: held=124.45s vs a 120 s budget -> the whole-leg lock + the
  rendezvous wait inside it = the co-located pair's deadlock.
* weg2xsn36: the 600 s budget turned the same deadlock into a SILENT
  10-minute stall (cpu/gpu/pcie 0 %, operator: "aktuell kann ich wieder
  keine cpu/gpu/pcie last erkennen").

THE CONTRACT, three parts:
1. ORDER: the collect's wait_full runs BEFORE and OUTSIDE the per-copy
   pcie lock (the lock serialises copies, never waits).
2. LIVENESS: the chunked wait checks the co-located deposit rank between
   chunks (NVML co-card pids, fail-open); a DEAD peer dies immediately,
   named with the leg identity -- budget still running.
3. BUDGET: 120 s stays the detector; a correct handshake never reaches it,
   a reached one is a finding and must be loud.

MUTANT (danger direction): re-introducing the wait INSIDE the flock (the
pre-fix shape) fails the order pin -- the deadlock's return is caught at
test time, not on the metal.
"""

from __future__ import annotations

import inspect
import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import weight_exchange_bounce as bx  # noqa: E402


class TheCollectWaitsOutsideTheFlock(unittest.TestCase):
    def test_wait_full_sits_before_the_pcie_copy_lock(self):
        """The ORDER pin: in run_bounce_leg's source, the collect's
        wait_full call precedes the first per-copy pcie lock acquisition.
        The pre-fix shape (the whole leg under the flock, the wait inside
        it) is the deadlock this pin keeps dead."""
        src = inspect.getsource(bx.run_bounce_leg)
        self.assertIn("rendezvous.wait_full_liveness(", src,
                      "the liveness-coupled wait must be the collect's path")
        wait_at = src.index("rendezvous.wait_full_liveness(")
        lock_at = src.index("with pcie_copy_lock(")
        self.assertLess(wait_at, lock_at,
                        "the wait must sit OUTSIDE (before) the per-copy "
                        "pcie lock acquisition")

    def test_the_liveness_refusal_names_the_leg(self):
        """The PeerGone refusal carries the leg identity (name + rank) so
        the record names WHO died, not only which slot."""
        src = inspect.getsource(bx.CrossSlotRendezvous.wait_full_liveness)
        self.assertIn("W68 Weg2PeerGone", src)
        self.assertIn("{tag!r}", src)
        self.assertIn("{rank}", src)
        self.assertIn("liveness()", src,
                      "the refusal must come from the measured liveness, "
                      "never from a bare timeout")

    def test_budget_is_the_120s_detector_not_a_stall(self):
        """The coordinator's requirement (b): 120 s stays the detector. The
        600 s raise (this session's first attempt) is named in the docstring
        as the counter-example: a raised budget turned the deadlock into a
        silent stall."""
        self.assertEqual(bx.LANE_RENDEZVOUS_BUDGET_S, 120.0)
        doc = inspect.getdoc(bx.CrossSlotRendezvous.wait_full_liveness) or ""
        self.assertIn("120 s", doc, "the detector contract must be stated")


class TheMutantWholeLegLockReturnsDies(unittest.TestCase):
    """MUTANT (operator-required danger direction): re-introducing the
    WHOLE-LEG card lock (the pre-d491500c71 shape that deadlocked the
    co-located pair twice) must be caught by name. The flip legs use the
    retired no-op; the real lock on these sites is the deadlock's return."""

    def test_the_flip_legs_use_the_retired_no_op_not_the_real_lock(self):
        from sglang.srt.managers.scheduler_components import weight_updater
        src = inspect.getsource(weight_updater)
        self.assertIn('_weg2_pcie_lock_retired("sleep-D2H "', src,
                      "the sleep leg must use the retired no-op")
        self.assertIn('_weg2_pcie_lock_retired("wake-H2D "', src,
                      "the wake leg must use the retired no-op")
        # NARROW: the exact whole-leg forms on the FLIP legs. The disk
        # reloads' own single-copy locks ("wake-H2D weights[ _draft] reload")
        # are legitimate and stay.
        self.assertNotIn(
            'with self._weg2_pcie_lock("sleep-D2H " + ",".join(weights_tags))',
            src, "the whole-leg lock on the sleep leg is the deadlock's return")
        self.assertNotIn(
            'with self._weg2_pcie_lock("wake-H2D " + ",".join(weights_tags))',
            src, "the whole-leg lock on the wake leg is the deadlock's return")

    def test_the_retired_no_op_holds_no_flock(self):
        """The retired context manager must hold NO lock: the deadlock was
        the lock held across the rendezvous waits."""
        import contextlib
        from sglang.srt.managers.scheduler_components import weight_updater
        retired = weight_updater.SchedulerWeightUpdaterManager
        cm = retired._weg2_pcie_lock_retired(None, "probe")
        self.assertIsInstance(cm, contextlib.nullcontext,
                              "the retired wrap must hold no lock at all")


if __name__ == "__main__":
    unittest.main()
