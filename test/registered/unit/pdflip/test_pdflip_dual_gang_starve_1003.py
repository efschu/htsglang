# SPDX-License-Identifier: Apache-2.0
"""ITEM 200: the GangGate hold cap must not starve D of its duty share.

Verified against the code (numbers): hold = burst_wall*(1-duty)/duty capped at
GANG_MAX_HOLD_S=2.5 s. duty 0.25, burst 2.0 s -> wanted hold 6.0 s, capped 2.5 s,
D alone 2.5/(2.0+2.5) = 55.6 % instead of 75 %.

Fix guarded here (no reserve, cap unchanged, D never paused, decision on PP0):
when the hold would hit the cap, the NEXT burst runs fewer chunks (k_eff) so
burst_wall*(1-duty)/duty <= the cap; k_eff recovers when chunks get shorter.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from test_pdflip_dual_gang_1001 import _Rig

from flliper.srt.pdflip import dual_duty as DD
from flliper.test.ci.ci_register import register_cpu_ci
from flliper.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _run_bursts(r, g, n_bursts):
    """Drive launches until n_bursts holds happened; return [(wall, hold)]."""
    out = []
    target = g.bursts + n_bursts
    guard = 0
    while g.bursts < target:
        guard += 1
        assert guard < 10000
        before = g.bursts
        r.launch()
        if g.bursts > before:
            out.append((g.last_burst_wall, g.last_hold))
    return out


class GangStarve(CustomTestCase):
    def test_issue_numbers_on_the_old_formula(self):
        duty, burst = 0.25, 2.0
        want = burst * (1 - duty) / duty
        hold = min(DD.GANG_MAX_HOLD_S, want)
        self.assertEqual(want, 6.0)
        self.assertEqual(hold, 2.5)
        self.assertAlmostEqual(hold / (burst + hold), 0.5556, places=3)
        self.assertEqual(DD.GANG_MAX_HOLD_S, 2.5)  # the cap is NOT raised

    def test_next_burst_is_bounded_so_the_hold_needs_no_cap(self):
        r = _Rig(busy=True, lag=0.25)
        g = r.make(duty=0.25, k=8)  # 8 * ~0.26 s = ~2.1 s burst -> wanted hold ~6.3 s
        bursts = _run_bursts(r, g, 8)
        wall0, hold0 = bursts[0]
        self.assertGreater(wall0 * 0.75 / 0.25, DD.GANG_MAX_HOLD_S)  # first burst hits the cap
        self.assertAlmostEqual(hold0, DD.GANG_MAX_HOLD_S, places=1)
        self.assertGreaterEqual(g.capped_bursts, 1)
        self.assertLess(g.k_eff, 8)
        for wall, hold in bursts[2:]:  # converged
            self.assertLessEqual(wall * 0.75 / 0.25, DD.GANG_MAX_HOLD_S + 0.1)
            self.assertAlmostEqual(hold, wall * 3.0, delta=0.12)  # full, uncapped hold
            self.assertGreaterEqual(hold / (wall + hold), 0.75 - 0.01)  # D's share holds
        self.assertGreater(g.bounded_bursts, 0)

    def test_single_slow_chunk_floors_at_one(self):
        r = _Rig(busy=True, lag=1.5)  # one chunk alone is over the budget
        g = r.make(duty=0.25, k=8)
        _run_bursts(r, g, 10)  # first bursts time out the drain (lower-bound wall), then it settles
        self.assertEqual(g.k_eff, 1)  # never 0: P still makes progress

    def test_no_bound_when_the_hold_fits(self):
        r = _Rig(busy=True, lag=0.05)
        g = r.make(duty=0.5, k=4)  # 4 * 0.06 = 0.24 s burst, hold 0.24 s
        _run_bursts(r, g, 6)
        self.assertEqual(g.k_eff, 4)
        self.assertEqual((g.capped_bursts, g.bounded_bursts), (0, 0))

    def test_k_recovers_when_chunks_get_short(self):
        r = _Rig(busy=True, lag=0.25)
        g = r.make(duty=0.25, k=8)
        _run_bursts(r, g, 5)
        self.assertLess(g.k_eff, 8)
        r.lag = 0.02
        _run_bursts(r, g, 4)
        self.assertEqual(g.k_eff, 8)

    def test_timed_out_drain_does_not_collapse_k(self):
        r = _Rig(busy=True, lag=0.1, publish=False)  # no completions: drain times out
        g = r.make(duty=0.5, k=4)
        for _ in range(5):
            r.launch()
        self.assertGreaterEqual(g.rebased, 1)
        self.assertEqual(g.k_eff, 4)

    def test_budget_math(self):
        g = _Rig().make(duty=0.25, k=4)
        self.assertAlmostEqual(g.burst_wall_budget_s(), 2.5 / 3.0)
        g2 = _Rig().make(duty=0.5, k=4)
        self.assertAlmostEqual(g2.burst_wall_budget_s(), 2.5)
