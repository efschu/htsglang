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
"""#656 RUNG 3: the weights arena's tail leaves the card in the phase that
cannot reach it.

THE BYTES, measured on the 2026-08-10 boot from the ``TP stack built`` line:

    rank  arena/pp    tp         tail
    PP0   13482.18    13163.45    318.7 MiB
    PP1    8144.00     7923.95    220.1 MiB
    PP2    9114.95     7923.95   1191.0 MiB

``pp`` is the max on every rank, so the tail is idle in TP -- the phase that
binds on all three cards after rung 2. That confirms the 319/220/1191 record
and refutes the 1773/0/1191 one; the two disagreed and the disagreement is
closed by measurement.

WHAT THESE PIN

* **The commit is priced on the tp->pp leg, and only there.** Rung 2's restore
  is priced on pp->tp; this one is the opposite direction, because PP is the
  larger layout. Pricing both on one leg would leave the other unpriced --
  which is exactly the ``cuMemCreate`` death inside the no-return region that
  the affordability gate exists to prevent.

* **Ordering, both directions.** Release AFTER the PP->TP refill (its
  ``restore=`` arm rewrites the PP layout, which reaches into the tail) and
  commit BEFORE the TP->PP refill (which writes the PP layout). Getting either
  backwards is a fault on unbacked memory, not a clean error, so the order is
  asserted against the source.

* **The rung is opt-in.** The default depth must allocate the arena exactly as
  it always did. A residency change that switches itself on is not a dial.

Hermetic: a fake arena, no CUDA.
"""

from __future__ import annotations

import inspect
import unittest

from sglang.srt.managers import phase_flip_spill as sp
from sglang.srt.managers.phase_flip_boot import PhaseFlipStacks
from sglang.srt.managers.phase_flip_runtime import PhaseFlipRuntime

MIB = 1024 * 1024


class _FakeArena:
    """Records commit/decommit against a byte cursor; no driver involved."""

    granularity = 2 * MIB
    base = 0

    def __init__(self):
        self.committed = 0
        self.calls = []

    def allocate_carrier(self, nbytes):
        class _T:
            def data_ptr(self_inner):
                return 0

            def numel(self_inner):
                return nbytes

        return _T()

    def commit_range(self, offset, nbytes):
        self.calls.append(("commit", nbytes))
        self.committed = max(self.committed, int(nbytes))

    def decommit_range(self, offset, keep):
        self.calls.append(("decommit", keep))
        released = max(0, self.committed - int(keep))
        self.committed = int(keep)
        return released

    def committed_bytes(self, offset):
        return self.committed


def _carrier(total_mib=13482):
    return sp.VmmWeightsArenaCarrier(0, total_mib * MIB, arena=_FakeArena())


class TheTailIsReleasedAndRestoredTest(unittest.TestCase):
    def test_it_starts_fully_committed_because_the_boot_packs_pp(self):
        c = _carrier()
        self.assertEqual(c.committed_bytes, 13482 * MIB)

    def test_shrinking_to_the_tp_layout_releases_the_measured_tail(self):
        c = _carrier(13482)
        released = c.set_active_prefix(13163 * MIB)
        self.assertEqual(round(released), 319)
        self.assertEqual(c.committed_bytes, 13163 * MIB)

    def test_the_big_3080_tail_is_the_one_worth_having(self):
        # PP2: 9114.95 pp vs 7923.95 tp. Four times the drafter's 285 MiB, on
        # a card that binds.
        c = _carrier(9115)
        self.assertEqual(round(c.set_active_prefix(7924 * MIB)), 1191)

    def test_growing_back_commits_and_releases_nothing(self):
        c = _carrier(13482)
        c.set_active_prefix(13163 * MIB)
        self.assertEqual(c.set_active_prefix(13482 * MIB), 0.0)
        self.assertEqual(c.committed_bytes, 13482 * MIB)

    def test_setting_the_same_prefix_twice_is_a_no_op(self):
        c = _carrier(13482)
        c.set_active_prefix(13163 * MIB)
        before = len(c._arena.calls)
        self.assertEqual(c.set_active_prefix(13163 * MIB), 0.0)
        self.assertEqual(len(c._arena.calls), before)

    def test_a_prefix_beyond_the_arena_is_clamped_not_a_fault(self):
        c = _carrier(8144)
        c.set_active_prefix(99999 * MIB)
        self.assertEqual(c.committed_bytes, 8144 * MIB)


class ThePendingCommitIsPricedTest(unittest.TestCase):
    def test_nothing_pending_while_the_tail_is_backed(self):
        c = _carrier(13482)
        self.assertEqual(c.pending_tail_bytes(13482 * MIB), 0)

    def test_the_released_tail_is_exactly_what_the_gate_must_price(self):
        c = _carrier(13482)
        c.set_active_prefix(13163 * MIB)
        self.assertEqual(c.pending_tail_bytes(13482 * MIB), 319 * MIB)

    def test_the_runtime_prices_it_on_tp_to_pp_and_not_on_pp_to_tp(self):
        from sglang.srt.layers.dcp.phase_flip_plan import PP_TO_TP, TP_TO_PP

        c = _carrier(13482)
        c.set_active_prefix(13163 * MIB)

        class _Layout:
            total_bytes = 13482 * MIB

        class _Stacks:
            arena_carrier = c
            layout_pp = _Layout()

        class _Sched:
            phase_flip_stacks = _Stacks()

        r = PhaseFlipRuntime.__new__(PhaseFlipRuntime)
        r._census_scheduler = _Sched()
        self.assertEqual(r._arena_tail_bytes(TP_TO_PP), 319 * MIB)
        # The drafter is the pp->tp payload; this one must not be double
        # counted onto that leg.
        self.assertEqual(r._arena_tail_bytes(PP_TO_TP), 0)

    def test_no_carrier_prices_zero_rather_than_raising(self):
        from sglang.srt.layers.dcp.phase_flip_plan import TP_TO_PP

        r = PhaseFlipRuntime.__new__(PhaseFlipRuntime)
        r._census_scheduler = None
        self.assertEqual(r._arena_tail_bytes(TP_TO_PP), 0)

    def test_staging_bytes_folds_the_tail_in_with_max_not_sum(self):
        src = inspect.getsource(PhaseFlipRuntime._staging_bytes)
        self.assertIn("_arena_tail_bytes", src)
        self.assertIn("max(", src)
        self.assertNotIn("+ self._arena_tail_bytes", src)


class OrderingAroundTheRefillTest(unittest.TestCase):
    """Backwards in either direction is a fault on unbacked memory."""

    def test_release_happens_after_the_pp_to_tp_refill(self):
        src = inspect.getsource(PhaseFlipStacks.refill)
        head = src.index("PP_TO_TP:")
        tail = src.index("TP_TO_PP:")
        leg = src[head:tail]
        self.assertLess(leg.index("arena_refill"), leg.index("set_active_prefix"))

    def test_commit_happens_before_the_tp_to_pp_refill(self):
        src = inspect.getsource(PhaseFlipStacks.refill)
        leg = src[src.index("TP_TO_PP:") :]
        self.assertLess(leg.index("set_active_prefix"), leg.index("arena_refill"))

    def test_a_stack_without_a_carrier_refills_exactly_as_before(self):
        self.assertIsNone(PhaseFlipStacks.arena_carrier)


class TheRungIsOptInTest(unittest.TestCase):
    def test_arena_sits_below_draft_graphs_and_does_not_renumber_draft(self):
        # Existing evidence records integer depths; "draft" must keep meaning 2.
        self.assertEqual(sp.DEPTH_NAMES["draft"], 2)
        self.assertEqual(sp.DEPTH_NAMES["arena"], 3)
        self.assertEqual(sp.DEPTH_NAMES["draft+graphs"], 4)

    def test_the_rung_is_implemented_and_graphs_still_are_not(self):
        self.assertEqual(sp.IMPLEMENTED_DEPTH, sp.DEPTH_ARENA_TAIL)
        self.assertGreater(sp.DEPTH_DRAFT_GRAPHS, sp.IMPLEMENTED_DEPTH)

    def test_the_boot_only_builds_a_carrier_at_or_above_the_arena_depth(self):
        from sglang.srt.managers import phase_flip_boot as boot

        src = inspect.getsource(boot.build_phase_flip_tp_stack)
        self.assertIn("DEPTH_ARENA_TAIL", src)
        self.assertIn("allocate_arena(", src)  # the default path survives


if __name__ == "__main__":
    unittest.main()
