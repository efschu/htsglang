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
"""#696: the arming floor may not be excused by funding that evaporates.

THE DEFECT, MEASURED. 2026-08-16, sustained under lane load: ``pp_to_tp``
abandoned every ~3 s on PP1 -- 18 abandons against 9 cutovers in 8 minutes,
three ``ARM-UNFUNDED`` in nine seconds -- while decode sat at 187.5 s against a
180 s budget. The shortfall was the same every time, which is what gave it
away:

    staging 733 MiB needed but only 691 MiB is spendable      (x8)

A runtime fluctuation does not repeat to the megabyte. That is a SIZING-TIME
constant.

WHERE IT COMES FROM. ``SeamReserve.arming_draw_bytes`` excuses the weights-arena
tail from the arming floor when the KV rung is recorded as able to pay it::

    if arena > 0 and rung >= arena:
        return max(0, leg - arena, fixed)

On PP1 that is arena 814.9 MiB, ``rung_fund_bytes`` 953.8 MiB, so the draw
falls from 815 to 138.9 and **815 MiB stops being reserved**. The pool is then
sized that much larger, and the enlarged pool is what leaves 691 MiB spendable
against a 733 MiB need.

WHY THE EXCUSE IS UNSOUND, and it is not that the measurement was wrong.
``rung_fund_bytes`` is a SINGLE-POINT reading. The rung's actual deliverable is
``(current_rows - floor_rows) * bytes_per_row``, and ``floor_rows`` tracks the
LIVE SET. At the abandon:

    current=473088 rows, floor=471983, slack=1105  ->  ~12.3 MiB deliverable

954 MiB was promised; 12.3 MiB was available. The funding collapses precisely
when occupancy is high -- which is exactly when a seam is hard to fund. The
excuse is granted at low fill and spent at high fill.

THE TWO REPAIRS THE BRIEF OFFERS CONVERGE. "Price the excuse against the
worst-case deliverable at max fill" and "drop the excuse" are the same
instruction, because at max fill ``current`` approaches ``floor_rows`` and the
guaranteed deliverable is ZERO. So the predicate below is written in the
general form -- excuse only what the rung is GUARANTEED to deliver -- and a
record that has never measured a guarantee yields no excuse. The mechanism
survives for a future measurement instead of being deleted.
"""

import unittest

from sglang.srt.managers.phase_flip_seam_reserve import SeamReserve
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5)

MIB = 1024 * 1024

# PP1 exactly as recorded on the live rig.
PP1 = dict(
    fixed_bytes=145652736,  # 138.9 MiB, the drafter restore
    arena_fixed_bytes=854522624,  # 814.9 MiB, the weights-arena tail
    worst_leg_fixed_bytes=854522624,
    rung_fund_bytes=1000175360,  # 953.8 MiB, measured at ONE fill
    per_row_bytes=424.2,
    id_space=471470,
)


def _reserve(**over):
    fields = dict(PP1)
    fields.update(over)
    return SeamReserve(**fields)


class TheExcuseMustNotBeGrantedOnAPointMeasurement(unittest.TestCase):
    def test_pp1_draw_carries_the_arena_tail(self):
        """The regression: 954 MiB of one-off funding excused 815 MiB forever."""
        draw = _reserve().arming_draw_bytes()
        self.assertGreaterEqual(
            draw / MIB,
            814.0,
            "the arena tail was excused from the arming floor on the strength "
            "of rung_fund_bytes, a single-point reading. The rung delivered "
            "12.3 MiB at the fill where the seam actually ran, and PP1 "
            "abandoned pp_to_tp every ~3 s, 42 MiB short.",
        )

    def test_a_recorded_guarantee_may_still_excuse_it(self):
        """The mechanism is not deleted, only made honest.

        A record that states what the rung is GUARANTEED to deliver -- not what
        it happened to hold once -- may excuse up to that amount.
        """
        r = _reserve(rung_guaranteed_bytes=900 * MIB)
        self.assertLess(r.arming_draw_bytes() / MIB, 200.0)

    def test_a_guarantee_below_the_tail_does_not_excuse_it(self):
        r = _reserve(rung_guaranteed_bytes=100 * MIB)
        self.assertGreaterEqual(r.arming_draw_bytes() / MIB, 814.0)

    def test_the_drafter_restore_always_floors_the_result(self):
        """Relieving the arena must not relieve the other leg by accident."""
        r = _reserve(rung_guaranteed_bytes=10_000 * MIB)
        self.assertGreaterEqual(
            r.arming_draw_bytes(),
            PP1["fixed_bytes"],
            "the drafter's restore is not arena tail and is not paid by this "
            "provider; it floors the draw.",
        )


class RanksWithoutATailAreUnaffected(unittest.TestCase):
    """PP0 has arena 0 -- nothing to excuse, and nothing to change."""

    def test_pp0_draw_is_its_leg(self):
        r = _reserve(
            fixed_bytes=238763008,
            arena_fixed_bytes=0,
            worst_leg_fixed_bytes=238763008,
            rung_fund_bytes=238763008,
        )
        self.assertAlmostEqual(r.arming_draw_bytes() / MIB, 227.7, places=0)


class APreDeltaRecordIsPricedAsBefore(unittest.TestCase):
    def test_no_worst_leg_falls_back_to_total_fixed(self):
        r = _reserve(worst_leg_fixed_bytes=0)
        self.assertEqual(r.arming_draw_bytes(), r.total_fixed_bytes)


if __name__ == "__main__":
    unittest.main()
