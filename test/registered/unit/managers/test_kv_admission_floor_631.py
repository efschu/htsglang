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
"""#656 register C20, residual 1: THE KV RUNG'S FLOOR PROTECTS ROWS, NOT WORK.

HANDOFF_681 §1a, measured on metal with ``SGLANG_SEAM_ENTRY_MARGIN_MIB=8192``:

    RuntimeError: Out of memory. Try to allocate 512 tokens.
    Available full tokens: 0 (full_available_size=0 + full_evictable_size=0)

The seam-entry margin is passed into ``collective_kv_backing_relief`` as part
of ``want_bytes``. The rung's deficit is ``floor + delta + want - free -
cheap``, so a margin no ladder can fund makes the deficit enormous on EVERY
seam and the rung shrinks to its floor. That floor is ``max_live + 1``: it
protects the rows that EXIST and reserves nothing to admit new work with, so
``available_size()`` reaches 0 and the next prefill raises inside the
scheduler loop -- two minutes and 42 cutovers after the first delay, which is
why the delay and yield branches were exonerated by their own timeline.

TWO PROPERTIES, and the first is the one that makes the second optional:

1. **The rung keeps an ADMISSION RESERVE.** It refuses to shrink below the
   live high-water mark PLUS a row reserve, so the pool that survives a shrink
   can still admit a chunked prefill. Group-uniform for free: the reduction
   already takes the MAX floor across ranks.
2. **The gate does not ASK for a margin the rung cannot fund.** The
   discretionary half of the ask (the C20 entry margin) is bounded by what
   this rank could return without crossing that floor. The mandatory half (the
   seam staging) is never bounded -- if the rung cannot fund it the guard
   refuses the seam, which is the pre-existing, survivable outcome.

Hermetic: the tensor-backed fakes from ``test_kv_backing_relief_631``, no CUDA.
"""

from __future__ import annotations

import unittest


from sglang.srt.managers import kv_backing_relief as kbr
from test_kv_backing_relief_631 import _Card, _FakeAllocator, _FakePool, _relief

MIB = 1024 * 1024
GIB = 1024 * MIB


def _rank(rows=100000, live_high=1000, free_mib=1200, **kw):
    """One rank's rung, with a live set whose top row is ``live_high``."""
    card = _Card(free_mib)
    pool = _FakePool(rows, card=card)
    alloc = _FakeAllocator(rows)
    # Take the live rows out of the free list so the allocator's own view
    # agrees with the live-set probe: the reserve is about what remains
    # ALLOCATABLE, which is a free-list property.
    alloc.alloc(live_high)
    relief = _relief(pool, alloc, live=(live_high,), card=card, **kw)
    return relief, pool, alloc, card


class TheRungKeepsAnAdmissionReserveTest(unittest.TestCase):
    def test_an_unfundable_want_does_not_shrink_below_the_reserve(self):
        """The 8 GiB margin, on the fixture, with the death it produced."""
        relief, pool, alloc, _card = _rank()
        proposal = relief.propose(
            want_bytes=8 * GIB,
            floor_bytes=1 * GIB,
            delta_bytes=768 * MIB,
            cheap_relief_bytes=0,
        )
        target = kbr.collective_kv_target(list(proposal))
        self.assertIsNotNone(target, "the rung declined a deficit it can fund")
        relief.apply_target(target)
        self.assertGreaterEqual(
            alloc.available_size(),
            kbr.DEFAULT_ADMISSION_RESERVE_ROWS,
            f"the rung shrank to {target} rows and left "
            f"{alloc.available_size()} allocatable: an admission floor that "
            f"reserves nothing to admit with is the HANDOFF_681 §1a death "
            f"('Try to allocate 512 tokens. Available full tokens: 0')",
        )

    def test_the_floor_itself_carries_the_reserve(self):
        """Stated on the proposal, so the group's MAX-floor reduction sees it."""
        relief, _pool, _alloc, _card = _rank(live_high=4242)
        proposal = relief.propose(
            want_bytes=8 * GIB, floor_bytes=1 * GIB, delta_bytes=0
        )
        floor_rows = -int(proposal[1])
        self.assertGreaterEqual(
            floor_rows, 4242 + 1 + kbr.DEFAULT_ADMISSION_RESERVE_ROWS
        )

    def test_the_reserve_is_uniform_across_ranks_by_the_existing_reduction(self):
        """No new collective: MIN of the negated floors already yields the MAX.

        The ranks differ in their live sets, which is the case that produced
        449039/451037/175225/145734 rows in one boot. Every rank must end
        above ITS OWN admission floor, and the reduction is what guarantees it.
        """
        ranks = [_rank(live_high=h) for h in (500, 9000, 3000)]
        proposals = [
            r.propose(want_bytes=8 * GIB, floor_bytes=1 * GIB, delta_bytes=0)
            for r, _p, _a, _c in ranks
        ]
        reduced = [min(col) for col in zip(*proposals)]
        target = kbr.collective_kv_target(reduced)
        self.assertIsNotNone(target)
        for (relief, _pool, alloc, _card), high in zip(ranks, (500, 9000, 3000)):
            relief.apply_target(target)
            self.assertGreaterEqual(
                target,
                high + 1 + kbr.DEFAULT_ADMISSION_RESERVE_ROWS,
                "the agreed target crosses a rank's admission floor",
            )
            self.assertGreaterEqual(
                alloc.available_size(), kbr.DEFAULT_ADMISSION_RESERVE_ROWS
            )

    def test_a_zero_reserve_reproduces_the_pre_fix_floor_exactly(self):
        """The off switch, and the mutation check for every test above.

        With the reserve at 0 this file's first assertion FAILS -- the rung
        shrinks to ``max_live + 1`` and leaves nothing allocatable. That is the
        shipped behaviour of HANDOFF_681 and the reason these tests exist, so
        it is pinned here rather than left as a claim.
        """
        relief, _pool, alloc, _card = _rank(admission_reserve_rows=0)
        proposal = relief.propose(
            want_bytes=8 * GIB, floor_bytes=1 * GIB, delta_bytes=0
        )
        target = kbr.collective_kv_target(list(proposal))
        relief.apply_target(target)
        self.assertLess(
            alloc.available_size(),
            kbr.DEFAULT_ADMISSION_RESERVE_ROWS,
            "with no reserve the rung is supposed to strand admission -- if "
            "this passes, the tests above are not testing the reserve",
        )

    def test_an_ordinary_ask_is_unaffected_by_the_reserve(self):
        """The reserve is a FLOOR, not a term in the ask.

        A deficit the rung can fund far above its floor must produce exactly
        the same target it produced before this change, or the fix has bought
        safety with capacity on every ordinary seam.
        """
        relief, _pool, _alloc, _card = _rank(free_mib=1200)
        with_reserve = relief.propose(
            want_bytes=64 * MIB, floor_bytes=1 * GIB, delta_bytes=0
        )
        bare, _p, _a, _c = _rank(free_mib=1200, admission_reserve_rows=0)
        without = bare.propose(want_bytes=64 * MIB, floor_bytes=1 * GIB, delta_bytes=0)
        self.assertEqual(with_reserve[0], without[0])


class TheGateDoesNotAskForWhatTheRungCannotFundTest(unittest.TestCase):
    """Property 2: the discretionary half of the ask is bounded."""

    def _relief_recording(self, fundable_rows):
        seen = {}
        relief, pool, alloc, card = _rank(rows=fundable_rows + 1000, live_high=999)
        original = relief.propose

        def spy(**kw):
            seen.update(kw)
            return original(**kw)

        relief.propose = spy
        return relief, seen

    def test_the_margin_is_capped_at_what_the_rung_can_fund(self):
        from sglang.srt.managers.phase_flip_spill import collective_kv_backing_relief

        relief, seen = self._relief_recording(4000)
        fundable = relief.fundable_bytes()
        self.assertGreater(fundable, 0, "fixture must have slack to fund")

        class _Guard:
            floor_bytes = 1 * GIB
            delta_bytes = 0
            device_index = 0

        class _Sched:
            pass

        sched = _Sched()
        setattr(sched, "phase_flip_kv_backing_relief", relief)
        collective_kv_backing_relief(
            sched,
            lambda vals: vals,
            want_bytes=64 * MIB + 8 * GIB,
            guard=_Guard(),
            direction="pp_to_tp",
            discretionary_bytes=8 * GIB,
        )
        self.assertLessEqual(
            seen["want_bytes"],
            64 * MIB + fundable,
            f"the gate asked the rung for {seen['want_bytes'] / MIB:.0f} MiB "
            f"when it can fund {fundable / MIB:.0f} MiB above its admission "
            f"floor: an unbounded discretionary ask is what drove the rung to "
            f"its floor on every seam",
        )
        self.assertGreaterEqual(
            seen["want_bytes"],
            64 * MIB,
            "the MANDATORY half (the seam staging) must never be bounded: if "
            "the rung cannot fund it the guard refuses the seam, which is the "
            "survivable outcome",
        )

    def test_without_a_discretionary_part_the_ask_is_passed_through(self):
        from sglang.srt.managers.phase_flip_spill import collective_kv_backing_relief

        relief, seen = self._relief_recording(4000)

        class _Guard:
            floor_bytes = 1 * GIB
            delta_bytes = 0
            device_index = 0

        class _Sched:
            pass

        sched = _Sched()
        setattr(sched, "phase_flip_kv_backing_relief", relief)
        collective_kv_backing_relief(
            sched,
            lambda vals: vals,
            want_bytes=777 * MIB,
            guard=_Guard(),
            direction="pp_to_tp",
        )
        self.assertEqual(seen["want_bytes"], 777 * MIB)


class TheGateWiresBothHalvesTest(unittest.TestCase):
    """The runtime must SAY which part of its ask is discretionary.

    Source-level, because the alternative is a live seam: the gate builds the
    ask from ``staging + margin`` and the rung must be told the margin's size
    or the cap above has nothing to bound.
    """

    def test_the_corridor_gate_declares_the_margin_as_discretionary(self):
        import inspect

        from sglang.srt.managers.phase_flip_runtime import PhaseFlipRuntime

        src = inspect.getsource(PhaseFlipRuntime._corridor_gate)
        gate = src[src.index("collective_kv_backing_relief(") :]
        self.assertIn("discretionary_bytes=", gate[:800])


if __name__ == "__main__":
    unittest.main()
