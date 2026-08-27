"""#939 -- a re-issued prefetch must not clobber the record it replaces.

THE CHAIN. A phase cutover retracts every resident; the retracted request is
re-queued through ``Scheduler._add_request_to_queue``, which calls
``_prefetch_kvcache``, which re-runs ``prefetch_from_storage`` -- a fresh radix
walk under the CURRENT binding generation. So the re-issue this ticket asks for
already exists and already runs.

What did not exist is a guard on the registration. ``prefetch_from_storage``
ended with an unconditional ``self.ongoing_prefetch[req_id] = ...``, so when the
post-cutover registration arrived while the pre-cutover one was still ongoing,
the old record was simply overwritten: its ``host_indices`` were lost to every
owner (nothing else holds them) and its ``anchor_lock_params`` lock ref was
never decremented, leaving that node PROTECTED for good. The next cutover's
``drop_prefix_tree_returning_rows`` then orphans exactly those device rows --
one request's allocation per cutover, which is the #938 forensic shape, with no
in-flight write-through needed to explain it.

WHY RETIRE AND NOT FREE. Whether the displaced span is safe to release is
answered by ``can_terminate_prefetch``, which is COLLECTIVE, and
``HiCacheController.terminate_prefetch`` is only ``mark_terminate()`` -- a flag,
not a join. Freeing at the retire site would hand back memory the prefetch
transfer thread may still be writing into (the #913 IMA family), and running the
collective there would make participation depend on a rank-local clobber inside
the #580 participation region. So the record is retired, and the per-round drain
-- where collectives are legal -- reaps it.

ORDERING, BY CONSTRUCTION. The re-issue owns the ``req_id`` slot, so a late
completion resolves through ``check_prefetch_progress`` to the NEW record and
cannot reach the displaced one; the displaced one is reachable only from the
reap. Neither insert nor free happens on the retired record anywhere else.

INTERACTION WITH #938's INSTRUMENT, stated so a reader does not misdiagnose it:
a retired-but-unreaped record still holds its lock ref, so ``#938 PROTECTED
RESIDUE ORPHANED`` can show it TRANSIENTLY. Healthy looks like a counter that
RETURNS across reap rounds; monotone growth is still a leak.
"""

import queue
import types
import unittest

import torch

from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.hicache_phase_binding import binding_state
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    PrefetchOperation,
)
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache_components.tree_component import (
    ComponentType,
)
from sglang.srt.mem_cache.unified_radix_cache import (
    UnifiedRadixCache,
    _OngoingPrefetch,
)
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

# ~4s: one tiny CPU-only radix tree plus a tiny CPU host pool, no group.
register_cpu_ci(est_time=4, suite="base-a-test-cpu")

SPAN = 8
PAGE_SIZE = 1
REQ_ID = "req-939"


def _build_cache() -> UnifiedRadixCache:
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
    return UnifiedRadixCache(
        params=CacheInitParams(
            req_to_token_pool=ReqToTokenPool(
                size=8, max_context_len=64, device="cpu", enable_memory_saver=False
            ),
            token_to_kv_pool_allocator=TokenToKVPoolAllocator(
                size=64,
                dtype=torch.float16,
                device="cpu",
                kvcache=MHATokenToKVPool(
                    size=64,
                    page_size=PAGE_SIZE,
                    dtype=torch.float16,
                    head_num=2,
                    head_dim=4,
                    layer_num=2,
                    device="cpu",
                    enable_memory_saver=False,
                ),
                need_sort=False,
            ),
            page_size=PAGE_SIZE,
            disable=False,
            tree_components=(ComponentType.FULL,),
        )
    )


def _scenario():
    """A cache holding ONE registered prefetch record, ready to be displaced.

    Built through the same real pieces the #905/#937 harnesses use: a real host
    pool, the real generation state machine, a real self-stamping
    ``PrefetchOperation``, and the real bound controller methods the retire and
    reap paths call.
    """
    binding_state().reset()
    cache = _build_cache()
    device_pool = cache.token_to_kv_pool_allocator.get_kvcache()

    def _pool(ratio, label):
        return MHATokenToKVPoolHost(
            device_pool=device_pool,
            host_to_device_ratio=ratio,
            host_size=0,
            page_size=PAGE_SIZE,
            layout="layer_first",
            pin_memory=False,
            device="cpu",
            allocator_type="default",
            budget_label=label,
        )

    pool_gen1 = _pool(0.5, "gen1")
    pool_gen2 = _pool(0.125, "gen2")
    binding_state().advance("pp", host_pool=pool_gen1)

    fake_cc = types.SimpleNamespace(
        mem_pool_host=pool_gen1,
        host_mem_release_queue=queue.Queue(),
        prefetch_tokens_occupied=SPAN,
        write_policy="write_through_selective",
    )
    fake_cc.terminate_prefetch = HiCacheController.terminate_prefetch.__get__(fake_cc)
    fake_cc.append_host_mem_release = HiCacheController.append_host_mem_release.__get__(
        fake_cc
    )
    cache.cache_controller = fake_cc

    host_indices = pool_gen1.alloc(SPAN)
    operation = PrefetchOperation(
        request_id=REQ_ID,
        host_indices=host_indices,
        token_ids=list(range(1, SPAN + 1)),
    )
    operation.hash_value = [f"h{i}" for i in range(SPAN // PAGE_SIZE)]
    operation.increment(SPAN)

    cache.ongoing_prefetch[REQ_ID] = _OngoingPrefetch(
        cache.root_node,
        RadixKey(list(range(1, SPAN + 1))),
        host_indices,
        operation,
        None,
        {},
    )
    # THE CUTOVER. This is the situation the ticket is about: the record was
    # registered before the flip, the re-issue arrives after it. The retired
    # span therefore belongs to gen1 while gen2 is bound, which is exactly what
    # makes the generation-stamped release load-bearing rather than cosmetic.
    binding_state().advance("tp", host_pool=pool_gen2)
    fake_cc.mem_pool_host = pool_gen2
    return cache, pool_gen1, pool_gen2, operation


class TestTheDisplacedRecordSurvivesTheReIssue(CustomTestCase):
    """THE RED TEST. Today the second registration overwrites the first and the
    first is gone from every owner's books."""

    def setUp(self):
        self.addCleanup(binding_state().reset)

    def test_the_displaced_record_is_retired_not_dropped(self):
        cache, _, _, operation = _scenario()

        cache._retire_ongoing_prefetch(REQ_ID)

        self.assertEqual(len(cache._retired_prefetch), 1)
        self.assertIs(cache._retired_prefetch[0].operation, operation)
        self.assertNotIn(REQ_ID, cache.ongoing_prefetch)

    def test_the_displaced_operation_is_marked_terminated(self):
        cache, _, _, operation = _scenario()
        cache._retire_ongoing_prefetch(REQ_ID)
        self.assertTrue(operation.is_terminated())

    def test_retiring_frees_nothing(self):
        """The safety property: the transfer may still be reading this span."""
        cache, pool_gen1, pool_gen2, _ = _scenario()
        free_before = pool_gen1.free_slots.numel()

        cache._retire_ongoing_prefetch(REQ_ID)

        self.assertEqual(pool_gen1.free_slots.numel(), free_before)


class TestTheReapReturnsTheSpanAndTheLock(CustomTestCase):
    """The displaced record's rows come back, by the generation-stamped route,
    from the one site that is allowed to ask whether that is safe."""

    def setUp(self):
        self.addCleanup(binding_state().reset)

    def test_the_span_returns_to_its_minting_pool(self):
        cache, pool_gen1, pool_gen2, _ = _scenario()
        cache._retire_ongoing_prefetch(REQ_ID)
        self.assertEqual(pool_gen1.free_slots.numel(), pool_gen1.size - SPAN)

        reaped = cache.drain_retired_prefetch()

        self.assertEqual(reaped, 1)
        self.assertEqual(pool_gen1.free_slots.numel(), pool_gen1.size)

    def test_the_currently_bound_pool_is_never_touched(self):
        """The #905/#911 routing rule, load-bearing here: the span goes back to
        gen1, which minted it, never to gen2, which is merely bound now."""
        cache, _, pool_gen2, _ = _scenario()
        free_before = pool_gen2.free_slots.numel()

        cache._retire_ongoing_prefetch(REQ_ID)
        cache.drain_retired_prefetch()

        self.assertEqual(pool_gen2.free_slots.numel(), free_before)
        self.assertEqual(int(pool_gen2.slot_used.sum()), 0)

    def test_the_retired_list_empties(self):
        cache, _, _, _ = _scenario()
        cache._retire_ongoing_prefetch(REQ_ID)
        cache.drain_retired_prefetch()
        self.assertEqual(cache._retired_prefetch, [])

    def test_the_drain_is_independent_of_request_lifetime(self):
        """Build point (1): nothing about the request happens again -- no
        completion, no queue entry, no admission -- and the list still empties.
        Otherwise this fix would trade a clobber for unbounded growth."""
        cache, pool_gen1, pool_gen2, _ = _scenario()
        cache._retire_ongoing_prefetch(REQ_ID)
        cache.ongoing_prefetch.clear()  # the request is finished and gone

        cache.drain_retired_prefetch()

        self.assertEqual(cache._retired_prefetch, [])
        self.assertEqual(pool_gen1.free_slots.numel(), pool_gen1.size)

    def test_an_empty_list_drains_to_a_no_op(self):
        cache, _, _, _ = _scenario()
        cache.ongoing_prefetch.clear()
        self.assertEqual(cache.drain_retired_prefetch(), 0)


class TestTheLateCompletionCannotTouchTheRetiredRecord(CustomTestCase):
    """Build point (2): a completion landing after the re-issue must do NEITHER
    an insert NOR a free of its own. Only the reap frees."""

    def setUp(self):
        self.addCleanup(binding_state().reset)

    def test_the_slot_belongs_to_the_re_issue(self):
        cache, _, _, old_operation = _scenario()
        cache._retire_ongoing_prefetch(REQ_ID)

        # The re-issue takes the slot.
        new_operation = PrefetchOperation(
            request_id=REQ_ID,
            host_indices=cache.cache_controller.mem_pool_host.alloc(SPAN),
            token_ids=list(range(1, SPAN + 1)),
        )
        cache.ongoing_prefetch[REQ_ID] = _OngoingPrefetch(
            cache.root_node,
            RadixKey(list(range(1, SPAN + 1))),
            new_operation.host_indices,
            new_operation,
            None,
            {},
        )

        # A completion for this req_id resolves to the NEW record; the retired
        # one is reachable only from the reap.
        self.assertIs(cache.ongoing_prefetch[REQ_ID].operation, new_operation)
        self.assertIs(cache._retired_prefetch[0].operation, old_operation)

    def test_a_completion_does_not_free_the_retired_span(self):
        cache, pool_gen1, pool_gen2, _ = _scenario()
        cache._retire_ongoing_prefetch(REQ_ID)
        free_after_retire = pool_gen1.free_slots.numel()

        # No record under this req_id at all -> the completion path is a no-op.
        self.assertTrue(cache.check_prefetch_progress(REQ_ID))

        self.assertEqual(pool_gen1.free_slots.numel(), free_after_retire)
        self.assertEqual(len(cache._retired_prefetch), 1)


class TestTheReFetchBudgetIsReported(CustomTestCase):
    """Build point (3): a cutover cadence faster than a fetch completes gets a
    named line instead of an unexplained cached=0."""

    def setUp(self):
        self.addCleanup(binding_state().reset)

    def test_the_budget_line_names_the_recompute(self):
        cache, _, _, _ = _scenario()

        with self.assertLogs(
            "sglang.srt.mem_cache.unified_radix_cache", level="WARNING"
        ) as caught:
            for _ in range(3):
                cache._retire_ongoing_prefetch(REQ_ID)
                cache.ongoing_prefetch[REQ_ID] = _OngoingPrefetch(
                    cache.root_node,
                    RadixKey(list(range(1, SPAN + 1))),
                    torch.arange(SPAN, dtype=torch.int64),
                    PrefetchOperation(
                        request_id=REQ_ID,
                        host_indices=torch.arange(SPAN, dtype=torch.int64),
                        token_ids=list(range(1, SPAN + 1)),
                    ),
                    None,
                    {},
                )

        self.assertIn("#939 RE-FETCH BUDGET SPENT", "\n".join(caught.output))
        self.assertIn("recomputing", "\n".join(caught.output))


class TestTheRegistrationIsGuarded(CustomTestCase):
    """THE CLOBBER, pinned at its own site.

    The behavioural tests above drive the retire/reap mechanics directly. This
    one pins the thing that made the mechanics necessary: that
    ``prefetch_from_storage`` retires the incumbent BEFORE it assigns the slot.
    Driving the full registration end-to-end would need the whole storage stack
    stood up; this is the technique the sibling #905 file uses for the same
    reason, and it fails if a refactor ever drops the guard.
    """

    def test_the_incumbent_is_retired_before_the_slot_is_overwritten(self):
        import inspect

        src = inspect.getsource(UnifiedRadixCache.prefetch_from_storage)
        self.assertIn("_retire_ongoing_prefetch(req_id)", src)
        self.assertLess(
            src.index("_retire_ongoing_prefetch(req_id)"),
            src.index("self.ongoing_prefetch[req_id] = _OngoingPrefetch("),
        )

    def test_the_reap_runs_every_round_independent_of_the_queue(self):
        """Unconditional, and NOT inside the waiting-queue comprehension: a
        displaced record outlives its request, and a collective gated on a
        rank-local list is the #580 failure."""
        import inspect

        from sglang.srt.managers.scheduler import Scheduler

        src = inspect.getsource(Scheduler._drain_prefetch_progress)
        self.assertIn("drain_retired_prefetch()", src)
        self.assertLess(
            src.index("drain_retired_prefetch()"),
            src.index("for req in self.waiting_queue"),
        )


class TestTheAgreementIsEnforcedNotInferred(CustomTestCase):
    """The digest the membership vote is built on must be rank-independent, or
    the 'agreement' agrees with nothing."""

    def test_the_digest_is_deterministic_and_never_zero(self):
        a = UnifiedRadixCache._req_id_digest("req-abc")
        b = UnifiedRadixCache._req_id_digest("req-abc")
        c = UnifiedRadixCache._req_id_digest("req-abd")
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertGreater(a, 0)

    def test_the_membership_agreement_gates_the_reap(self):
        """STRUCTURAL, and deliberately so. On one rank the all_reduce is a
        no-op and the agreement is trivially satisfied, so no CPU test can make
        dropping it go red behaviourally -- it only bites on a group. Pinned at
        the source instead, because a refactor that removes it would reap a
        record one rank has not retired yet, on that rank alone."""
        import inspect

        src = inspect.getsource(UnifiedRadixCache.drain_retired_prefetch)
        self.assertIn("ReduceOp.MIN", src)
        self.assertIn("agreed_min != agreed_max", src)
        # ...and the reduce must be issued before the gate reads its result.
        self.assertLess(src.index("_all_reduce_attn_groups"), src.index("agreed_min"))

    def test_the_candidate_order_is_canonical_not_insertion_order(self):
        """Sorted by req_id, so every rank picks the same candidate from the
        same set regardless of the order retirements happened in."""
        import inspect

        src = inspect.getsource(UnifiedRadixCache.drain_retired_prefetch)
        self.assertIn("sorted(", src)
        self.assertIn("request_id", src)


if __name__ == "__main__":
    unittest.main()
