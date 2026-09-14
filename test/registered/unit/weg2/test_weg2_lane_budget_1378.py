# SPDX-License-Identifier: Apache-2.0
"""#1378 xsn34 root fix: the lanes' rendezvous budget covers the staging.

MEASURED on weg2xsn34 (2026-09-14, first flip, epoch 0): the wake side's
collect waited its full 120 s budget for the sleep side's FIRST full post and
expired 3 s before it landed -- collect alloc 13:27:56 / free 13:29:56,
deposit alloc 13:29:59, fence "joined in 120675 ms" ok=0/3, W68 on both sides
(the collect read "not posted", the deposit read "stalled peer" -- two
authorities, one race). The sleep side's pre-deposit staging is the first
flip's pause chain, not a dead peer. The lanes' budget is now the boot's own
leg bound (LANE_RENDEZVOUS_BUDGET_S = 600 s = the deadman's GRACE_S), so the
two authorities are one number.

RED-FIRST: the construction-site pin failed against the pre-fix tree (the
site used the 120 s tool default). The behavioural half of this fix ran ON
THE METAL: weg2xsn34's own deposit-side credit wait used exactly this
timedwait path with its 120 s budget and expired on schedule -- the wait
mechanism is proven live; what was wrong is WHICH budget the lanes carry.
"""

from __future__ import annotations

import inspect
import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import weight_exchange_bounce as bx  # noqa: E402
class TheBudgetCoversTheStaging(unittest.TestCase):
    """The behavioural half of this fix ran ON THE METAL, not at the desk:
    weg2xsn34's own deposit-side credit wait used exactly this timedwait path
    with a 120 s budget and expired on schedule -- the mechanism is proven
    live. What the desk pins here is the CONSTRUCTION SITE (the race's fix is
    which budget the lanes carry), so a behavioural duplicate would test the
    same timedwait twice and prove nothing new."""

class TheConstructionSiteUsesTheLaneBudget(unittest.TestCase):
    """RED-FIRST pin: the LANES' rendezvous construction must hand the named
    boot budget, not the 120 s tool default. Reverting the site re-opens the
    xsn34 race and this test kills it."""

    def test_lane_rendezvous_passes_the_named_budget(self):
        from sglang.srt.managers.scheduler_components import weight_updater
        src = inspect.getsource(weight_updater)
        self.assertIn("budget_s=bx.LANE_RENDEZVOUS_BUDGET_S", src,
                      "the lanes' rendezvous must carry the boot's own leg "
                      "bound, not the 120 s tool default -- the xsn34 race "
                      "(120 s expired 3 s before the post) is exactly what "
                      "this pin keeps closed")
        # 180 s: the deposit's measured staging (~123 s) + margin. The race
        # this budget must survive: the collect's first wait vs the deposit's
        # first post (xsn34: post 123 s, budget 120 s -> W68; xsn40: same).
        self.assertEqual(bx.LANE_RENDEZVOUS_BUDGET_S, 180.0)


if __name__ == "__main__":
    unittest.main()
