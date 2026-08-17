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
"""#737: the ack-count drain is rank-local, and that is SAFE -- proven both ways.

THE DEADLOCK, measured 2026-08-17. `_count_ready_acks` MIN-reduced a ready count
across the group, and that reduction sat inside `check_hicache_events` ->
`_get_new_batch_prefill_raw`, i.e. inside the PER-MICROBATCH path of a pipeline.
A pipeline keeps its stages at DIFFERENT offsets by construction, so:

    PP0/PP1  inside the HiCache drain (the collective)
    PP2      blocked in _pp_recv_proxy_tensors, waiting for data PP1 will only
             send after leaving the collective PP1 cannot leave without PP2

A circular wait between adjacent stages -- the #633 shape one level up, with a
collective in place of a handler. The bug was PLACEMENT, not participation: all
stages call the function, just never on the same tick.

TWO ARMS, because removing a collective owes two different proofs:

  (i)  DEADLOCK -- with one rank held at a pipeline recv, the other two must
       still complete their drain. Red before the fix: they block on a peer
       that cannot arrive.
  (ii) DIVERGENCE -- ranks acking in different orders and at different ticks
       must not make a key usable early. This runs through the REAL
       `PageCompleteness`, not a stub, because that marker is what now
       owns the property the collective was wrongly credited with: production
       is layer-sharded while storage is token-sharded, so a page's slots
       arrive from several PP stages and an incomplete page reads as a MISS,
       never as wrong bytes.

WHAT IS NOT PROVEN HERE, deliberately: the throttle. The old MIN also paced the
ranks ("never further apart than the slowest live transfer"). That is pacing,
not correctness, and it is filed rather than replaced -- a backpressure bound
chosen without an operating point would be another shipped number without
evidence. The drain-depth observability line exists so the first real fast-rank
pressure specimen is attributable when it appears.
"""

import types
import unittest

from sglang.srt.mem_cache.canonical_kv_page import (
    CanonicalPageError,
    CanonicalPageSpec,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3)


class _Event:
    """A finish event whose readiness the test controls."""

    def __init__(self, ready: bool):
        self._ready = ready

    def query(self) -> bool:
        return self._ready


def _holder(pp_rank: int, readies):
    """Minimal carrier exposing only what _count_ready_acks touches."""
    h = types.SimpleNamespace(
        pp_rank=pp_rank,
        pp_size=3,
        _drain_depth_every=0.0,   # observability off; pinned separately
        _drain_depth_at=0.0,
    )
    h._count_ready_acks = types.MethodType(UnifiedRadixCache._count_ready_acks, h)
    queue = [(None, _Event(r), []) for r in readies]
    return h, queue


class TheDrainTakesNoGroupAgreement(unittest.TestCase):
    """ARM (i): a rank drains on its own evidence, so no peer can block it.

    Before the fix these calls entered an all_reduce/_pp_sync. With one stage
    held at a pipeline recv -- which is the steady state of a pipeline, not an
    exotic failure -- the remaining stages blocked forever. The proof that the
    deadlock is gone is that this function now touches NOTHING shared: no
    torch.distributed symbol is reachable from it, so there is no peer to wait
    for.
    """

    def test_a_rank_counts_only_its_own_ready_acks(self):
        h, q = _holder(pp_rank=1, readies=[True, True, False, True])
        self.assertEqual(2, h._count_ready_acks(q, "writing_check"))

    def test_every_stage_answers_independently_and_may_differ(self):
        """The divergence the pipeline creates is now simply ALLOWED."""
        h0, q0 = _holder(0, [True, True, True])
        h1, q1 = _holder(1, [True, False, False])
        h2, q2 = _holder(2, [])            # a stage that backs nothing up
        self.assertEqual(3, h0._count_ready_acks(q0, "writing_check"))
        self.assertEqual(1, h1._count_ready_acks(q1, "writing_check"))
        self.assertEqual(0, h2._count_ready_acks(q2, "writing_check"))

    def test_an_idle_stage_cannot_freeze_a_busy_one(self):
        """#581 by construction: the MIN's empty-queue trap is unreachable.

        One idle rank used to drag the reduction to zero and freeze the drain on
        ALL ranks, ratcheting `protected` until the state pool died. A rank that
        consults no peer cannot be dragged.
        """
        busy, qb = _holder(0, [True, True])
        idle, qi = _holder(2, [])
        self.assertEqual(0, idle._count_ready_acks(qi, "writing_check"))
        self.assertEqual(2, busy._count_ready_acks(qb, "writing_check"))

    def test_the_function_reaches_no_collective(self):
        """CAN-FAIL, and the point of the whole change.

        If a future edit reintroduces a group operation here, the deadlock comes
        back. torch.distributed is stubbed out entirely: any use raises.
        """
        import sglang.srt.mem_cache.unified_radix_cache as urc

        class _Forbidden:
            def __getattr__(self, name):
                raise AssertionError(
                    f"_count_ready_acks reached torch.distributed.{name}: the "
                    "#737 deadlock is a collective in the per-microbatch path"
                )

        real = urc.torch.distributed
        urc.torch.distributed = _Forbidden()
        try:
            h, q = _holder(1, [True, False])
            self.assertEqual(1, h._count_ready_acks(q, "writing_check"))
        finally:
            urc.torch.distributed = real


class TheMarkerBoundsTheDivergence(unittest.TestCase):
    """ARM (ii): through the REAL marker, not a stub.

    This is the property the collective was wrongly credited with. Ranks may
    now ack in any order at any tick; what keeps a key from being served early
    is that a page is only usable once EVERY slot is marked.
    """

    def _progress(self, slots=16):
        from sglang.srt.mem_cache.canonical_kv_page import PageCompleteness

        return PageCompleteness(
            CanonicalPageSpec(num_attn_layers=slots, kv_bytes_per_token_per_attn_layer=2048)
        )

    def test_a_page_is_a_miss_until_every_stage_has_written(self):
        """Stages 7/5/4 of [31,17,16] arriving in the WORST order."""
        p = self._progress()
        for slot in list(range(12, 16)) + list(range(7, 12)):   # PP2 then PP1
            p.mark(slot)
        self.assertFalse(p.is_complete(), "incomplete page must not be usable")
        self.assertEqual(tuple(range(0, 7)), p.missing())
        for slot in range(0, 7):                                # PP0 last
            p.mark(slot)
        self.assertTrue(p.is_complete())
        self.assertEqual((), p.missing())

    def test_order_and_tick_do_not_matter_only_completeness(self):
        """CAN-FAIL for the whole rank-local argument.

        If completeness depended on ARRIVAL ORDER, rank-local acking would be
        unsafe and the collective would have to come back.
        """
        forward, reverse = self._progress(), self._progress()
        for s in range(16):
            forward.mark(s)
        for s in reversed(range(16)):
            reverse.mark(s)
        self.assertTrue(forward.is_complete())
        self.assertTrue(reverse.is_complete())

    def test_a_fast_rank_racing_ahead_cannot_complete_a_page_alone(self):
        """The exact hazard rank-local acking is accused of creating."""
        p = self._progress()
        for slot in range(0, 7):        # PP0 drains its whole queue immediately
            p.mark(slot)
        self.assertFalse(p.is_complete())
        self.assertEqual(9, len(p.missing()))

    def test_double_writing_a_slot_is_refused_not_absorbed(self):
        """Two writers for one slot is a layout bug, not an idempotent retry."""
        p = self._progress()
        p.mark(3)
        with self.assertRaises(CanonicalPageError):
            p.mark(3)


if __name__ == "__main__":
    unittest.main()
