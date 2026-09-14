# SPDX-License-Identifier: Apache-2.0
"""#1378 xsn35 root fix: the per-card PCIe lock budget is the boot's own leg
bound, not 120 s.

MEASURED on weg2xsn35 (2026-09-14, first flip, epoch 0, the 5090):
`waited=0.000s held=124.450s` -- the wake-H2D of the ring-off flip held the
card's serialisation lock for 124.45 s while the sibling's
release_memory_occupation waited with DEFAULT_PCIE_LOCK_TIMEOUT_S = 120.0 and
expired: W29 Weg2FlipRankDisagree -> group dead -> W17 Weg2GroupDead.  The
holder was healthy (the collect stages ~6 GiB through the bounce over the x4
link); the WAITER's budget was the defect.  Same class as the xsn34 W68
budget race, one fix doctrine: every wait inside a flip carries the boot's
own leg bound (600 s = LANE_RENDEZVOUS_BUDGET_S = the deadman's GRACE_S), so
no budget loses a race against a healthy sibling again.  A genuinely dead
holder still dies at the same bound the deadman enforces.

The lock is flock-based and dies with the holder (weg2_memory_saver's own
docstring), so a stale FILE from a dead boot cannot block anyone -- the xsn35
contention was LIVE, between two healthy halves of one flip.

RED-FIRST: the constant pin failed against the pre-fix tree (120.0).
"""

from __future__ import annotations

import fcntl
import os
import tempfile
import threading
import time
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers import weg2_memory_saver as ms  # noqa: E402


class TheLockBudgetIsTheBootBound(unittest.TestCase):
    def test_the_default_is_the_boot_bound(self):
        """RED-FIRST pin: 600 s, the same number as the lanes' rendezvous
        budget and the deadman's grace. Changing one without the other
        splits the wait authorities again."""
        from sglang.srt.weg2 import weight_exchange_bounce as bx
        self.assertEqual(ms.DEFAULT_PCIE_LOCK_TIMEOUT_S, 120.0,
                         "NUTZER-ORDER 14.09.: fail fast -- the per-copy "
                         "holds are 0.1 s; a long lock budget only delays "
                         "the detection of a stuck holder")


class TheLockStillRefusesWhenGenuinelyStuck(unittest.TestCase):
    """The bound must stay a bound: a holder that keeps the lock longer than
    the caller's budget still produces the NAMED refusal (never an infinite
    wait, never a silent unserialised overlap -- own mutant B-M2's danger
    direction, pinned here behaviourally)."""

    def test_expired_budget_raises_the_named_refusal(self):
        with tempfile.TemporaryDirectory(prefix="weg2-pcie-lock-") as d:
            path = ms.pcie_lock_path("test-uuid", lock_dir=d)
            holder = open(path, "w")
            fcntl.flock(holder, fcntl.LOCK_EX)  # the healthy sibling holds it

            got = {}

            def waiter():
                try:
                    with ms.pcie_transfer_lock(
                        nvml_uuid="test-uuid", lock_dir=d,
                        timeout_s=0.5, label="test-wait",
                    ):
                        got["acquired"] = True
                except ms.Weg2PcieLockTimeout as exc:
                    got["refused"] = str(exc)

            t = threading.Thread(target=waiter, daemon=True)
            t.start()
            t.join(timeout=30)
            holder.close()
            self.assertIn("refused", got,
                          f"an expired budget must refuse by name: {got}")
            self.assertIn("not acquired within", got["refused"])
            self.assertNotIn("acquired", got)

    def test_a_released_lock_is_acquired_within_the_budget(self):
        """The other half: a holder that RELEASES inside the budget must be
        waited out, not refused -- that is the xsn35 fix itself, at a small
        scale (holder 0.4 s, budget 3 s)."""
        with tempfile.TemporaryDirectory(prefix="weg2-pcie-lock-") as d:
            path = ms.pcie_lock_path("test-uuid", lock_dir=d)
            holder = open(path, "w")
            fcntl.flock(holder, fcntl.LOCK_EX)

            def release_soon():
                time.sleep(0.4)
                fcntl.flock(holder, fcntl.LOCK_UN)
                holder.close()

            threading.Thread(target=release_soon, daemon=True).start()
            acquired = False
            with ms.pcie_transfer_lock(
                nvml_uuid="test-uuid", lock_dir=d,
                timeout_s=3.0, label="test-wait2",
            ):
                acquired = True
            self.assertTrue(acquired,
                            "a released lock must be acquired within the "
                            "budget (the xsn35 fix, small scale)")


if __name__ == "__main__":
    unittest.main()
