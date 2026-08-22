# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""#685: a cold boot must price the seam, and price it against the right R'.

THE DEFECT, REPRODUCED LIVE. 2026-08-16 12:04, all three ranks:

    PHASE-FLIP-SEAM-RESERVE (rank N): seam reserve is COLD (no record at
    .../kv_budget-...-seam-rankN.json): this boot sizes with NO flip-seam
    term and may produce an i[nvalid pool]

and the pool came out at 550000 -- the raw --max-total-tokens pin, not a
solved figure. The warm boot minutes later solved 467708 against the same
budget. So a cold boot sizes ~17% larger than the same configuration knows to
be safe, which is the cold-overshoot OOM class: the first flip then meets a
pool that was never priced for it.

THE DESIGN DECISION THIS FILE PINS: which budget the anchor-free solve uses
for ``R'``.

    R' = budget_bytes NET OF THE FLOOR CHARGE.

The arming floor is memory that must STAY FREE for a flip to arm at all. It
is a reservation, not a spendable balance, so it can never become KV. Solving
against the pre-subtraction figure would size the pool as though the floor
were available -- which is exactly the overshoot above, arrived at by a
different route. The net figure is also what the WARM branch returns, so both
branches answer "how many bytes could become KV" with the same quantity and
the cold path cannot silently mean something else.

AND IT MUST NOT DOUBLE-CHARGE. The caller subtracts the floor charge once.
The cold branch therefore solves against the net figure and returns
``min(net, allowed * cell)`` -- never ``net - charge`` a second time, which
would undershoot by a whole arming floor and hold that VRAM free forever.
"""

import unittest

from sglang.srt.managers.phase_flip_seam_reserve import solve_pool_tokens
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5)

MIB = 1024 * 1024
CELL = 16287  # this rig's rank-0 bytes/token (7 attention layers)
SLOPE = 2326.7  # one received layer, derived
BUDGET = 24000 * MIB
FLOOR = 1728 * MIB  # the measured arming floor on rank 0


class ColdSizingMustPriceTheSeam(unittest.TestCase):
    def test_a_priced_seam_sizes_smaller_than_an_unpriced_one(self):
        """The whole defect in one comparison.

        An unpriced seam is ``a = 0``: staging costs nothing and every byte
        of R' becomes KV. That is what a cold boot did, and it is what
        produced 550000 where the warm boot solved 467708.
        """
        unpriced = solve_pool_tokens(BUDGET - FLOOR, CELL, 0, 0.0)
        priced = solve_pool_tokens(BUDGET - FLOOR, CELL, 0, SLOPE)
        self.assertLess(
            priced,
            unpriced,
            "a derived seam slope must reduce the cold pool; if it does not, "
            "the cold boot is still sizing as though the flip were free.",
        )

    def test_the_zero_receive_rank_is_not_charged(self):
        """The other half of #685: no received layers, no per-token seam."""
        charged = solve_pool_tokens(BUDGET - FLOOR, CELL, 0, SLOPE)
        exempt = solve_pool_tokens(BUDGET - FLOOR, CELL, 0, SLOPE, received_layers=0)
        self.assertGreater(exempt, charged)
        self.assertEqual(exempt, solve_pool_tokens(BUDGET - FLOOR, CELL, 0, 0.0))


class RPrimeIsNetOfTheFloorCharge(unittest.TestCase):
    """The design decision, pinned so it cannot drift back."""

    def test_solving_against_the_gross_budget_overshoots(self):
        gross = solve_pool_tokens(BUDGET, CELL, 0, SLOPE)
        net = solve_pool_tokens(BUDGET - FLOOR, CELL, 0, SLOPE)
        self.assertGreater(
            gross,
            net,
            "solving R' against the pre-subtraction budget prices the arming "
            "floor as if it could become KV. It cannot: it must stay free for "
            "a flip to arm at all.",
        )
        # And the overshoot is the whole floor's worth of tokens, not a rounding
        # difference -- which is why it reaches OOM rather than tightness.
        self.assertGreater((gross - net) * CELL, FLOOR // 2)

    def test_the_floor_is_never_charged_twice(self):
        """``net - charge`` again would hold an arming floor free forever."""
        net = solve_pool_tokens(BUDGET - FLOOR, CELL, 0, SLOPE)
        twice = solve_pool_tokens(BUDGET - 2 * FLOOR, CELL, 0, SLOPE)
        self.assertGreater(
            net,
            twice,
            "the caller subtracts the floor once; the cold branch must solve "
            "against that net figure and not subtract it again.",
        )


class TheSolveStaysMonotone(unittest.TestCase):
    """Sanity the shape must keep, or the cold path is not comparable to warm."""

    def test_more_budget_never_gives_fewer_tokens(self):
        prev = -1
        for mult in (0.5, 0.75, 1.0, 1.5, 2.0):
            got = solve_pool_tokens(int((BUDGET - FLOOR) * mult), CELL, 0, SLOPE)
            self.assertGreaterEqual(got, prev)
            prev = got

    def test_a_steeper_slope_never_gives_more_tokens(self):
        prev = None
        for slope in (0.0, 500.0, 1000.0, SLOPE, 5000.0):
            got = solve_pool_tokens(BUDGET - FLOOR, CELL, 0, slope)
            if prev is not None:
                self.assertLessEqual(got, prev)
            prev = got


if __name__ == "__main__":
    unittest.main()
