"""#966 -- a HiCache detach must release the retired prefetch records it strands.

THE CHAIN, and every link is a line in the shipped tree.

``UnifiedRadixCache._retired_prefetch`` (#939) holds three things per record: a
host span (nothing else holds it -- see the retire-site comment), an anchor host
lock ref that keeps that node PROTECTED, and a ``prefetch_tokens_occupied``
charge that gates future prefetches through ``prefetch_capacity_limit``
(cache_controller.py:2033).

Its ONLY release is ``drain_retired_prefetch``. That method's only caller is
``scheduler.py:8400``, which sits below ``if not self.enable_hicache_storage:
return {}`` (scheduler.py:8386). And ``_detach_hicache_storage_impl`` clears
exactly that flag (scheduler.py:10820). So after a detach the holder has no
reachable release path at all -- the class (e) shape.

``detach_storage_backend``'s own docstring states the contract this violates:
"drain the control queues BEFORE tearing the controller's threads down, or acks
and releases can no longer be matched to their nodes and host pages and locks
leak". The #939 holder arrived after that sentence and was never folded into it.

WHAT DOES *NOT* CLOSE IT (the coverage check, so nobody re-opens the question):
* ``_reset_full`` REBINDS ``self._retired_prefetch = []`` -- it drops the
  holders without releasing anything.
* ``cache_controller.reset()`` zeroes ``prefetch_tokens_occupied`` only under
  ``if self.enable_storage:`` (cache_controller.py:1161-1169), and the detach
  has just set that False (cache_controller.py:929). So the charge survives even
  a full reset, permanently throttling prefetch admission.
* ``cache_controller.detach_storage_backend()`` stops threads and closes the
  backend; it touches no prefetch record.

THE DANGER DIRECTION IS THE OTHER ONE. Releasing a record that is still LIVE is
worse than leaking one: its span may still be a transfer destination and may
still be adopted into the tree. ``TestTheLiveRecordIsNeverTouched`` is the
mutant guard for exactly that, and it fails on any fix that drains
``ongoing_prefetch`` along the way.
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
REQ_ID = "req-966"
LIVE_REQ_ID = "req-966-live"


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


def _scenario(cutover: bool = True):
    """A cache with ONE retired prefetch record, at the point of a detach.

    Same construction as the #939 harness -- a real host pool, the real
    generation state machine, a real self-stamping ``PrefetchOperation``, the
    real bound controller methods -- extended with the queues and the teardown
    hook ``detach_storage_backend`` actually touches.

    ``cutover`` selects the two routes ``append_host_mem_release`` can take, and
    both matter here:
    * ``True``  -- the span was minted under gen1 and gen2 is bound now, so the
      release is settled IMMEDIATELY against gen1 (the W35 stale route).
    * ``False`` -- the span is current, so the release is QUEUED and only
      becomes a free when the control-queue drain runs. This is the route that
      pins WHERE in the detach the release has to sit.
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

    order = []
    fake_cc = types.SimpleNamespace(
        mem_pool_host=pool_gen1,
        host_mem_release_queue=queue.Queue(),
        prefetch_revoke_queue=queue.Queue(),
        ack_backup_queue=queue.Queue(),
        prefetch_tokens_occupied=SPAN,
        write_policy="write_through_selective",
        teardown_order=order,
    )
    fake_cc.terminate_prefetch = HiCacheController.terminate_prefetch.__get__(fake_cc)
    fake_cc.append_host_mem_release = HiCacheController.append_host_mem_release.__get__(
        fake_cc
    )

    def _cc_detach():
        # Stands in for `_stop_storage_threads` + backend close. The ONLY thing
        # the tests need from it is that it happened, and when.
        order.append("threads-joined")
        fake_cc.enable_storage = False

    fake_cc.detach_storage_backend = _cc_detach
    cache.cache_controller = fake_cc
    cache.enable_storage = True

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
    if cutover:
        binding_state().advance("tp", host_pool=pool_gen2)
        fake_cc.mem_pool_host = pool_gen2

    # THE RETIREMENT. This is the state the ticket is about: a record displaced
    # by a re-issue, terminated, holding its span until a reap that a detach is
    # about to make unreachable.
    cache._retire_ongoing_prefetch(REQ_ID)
    return cache, pool_gen1, pool_gen2, operation, order


class TestTheDetachReleasesTheRetiredRecord(CustomTestCase):
    """THE RED TEST. Today the detach walks past the holder and the record can
    never be freed again: its reap sits under a gate the detach itself clears."""

    def setUp(self):
        self.addCleanup(binding_state().reset)

    def test_the_retired_list_is_empty_after_a_detach(self):
        cache, _, _, _, _ = _scenario()
        self.assertEqual(len(cache._retired_prefetch), 1)

        ok, _msg = cache.detach_storage_backend()

        self.assertTrue(ok)
        self.assertEqual(cache._retired_prefetch, [])

    def test_the_span_returns_to_the_pool_that_minted_it(self):
        """Delivery by EFFECT: the rows are free again, not merely reported."""
        cache, pool_gen1, _, _, _ = _scenario()
        self.assertEqual(pool_gen1.free_slots.numel(), pool_gen1.size - SPAN)

        cache.detach_storage_backend()

        self.assertEqual(pool_gen1.free_slots.numel(), pool_gen1.size)

    def test_the_currently_bound_pool_is_never_touched(self):
        """#905/#911 routing survives the new caller: the span goes back to
        gen1, which minted it, never to gen2, which is merely bound now."""
        cache, _, pool_gen2, _, _ = _scenario()
        free_before = pool_gen2.free_slots.numel()

        cache.detach_storage_backend()

        self.assertEqual(pool_gen2.free_slots.numel(), free_before)
        self.assertEqual(int(pool_gen2.slot_used.sum()), 0)

    def test_the_prefetch_charge_is_returned(self):
        """The charge outlives even a reset -- `cache_controller.reset()` zeroes
        it only `if self.enable_storage`, which the detach has just cleared. So
        leaving it here throttles prefetch admission for the rest of the boot."""
        cache, _, _, _, _ = _scenario()
        self.assertEqual(cache.cache_controller.prefetch_tokens_occupied, SPAN)

        cache.detach_storage_backend()

        self.assertEqual(cache.cache_controller.prefetch_tokens_occupied, 0)


class TestTheReleaseIsPlacedWhereItIsBothSafeAndEffective(CustomTestCase):
    """Two ordering constraints, one on each side. Between them there is exactly
    one legal slot in `detach_storage_backend`, and each half is pinned."""

    def setUp(self):
        self.addCleanup(binding_state().reset)

    def test_the_release_happens_after_the_transfer_threads_are_joined(self):
        """SAFETY. The retire site refuses to free because `mark_terminate()` is
        a flag, not a join -- the transfer thread may still be writing into the
        span. Releasing before the controller teardown would reinstate exactly
        that hazard, so the order is asserted rather than assumed."""
        cache, _, _, _, order = _scenario()

        # Observe the real method rather than having the source know about the
        # test: the wrapper records WHEN it ran and then runs it unchanged.
        real_release = cache._release_retired_prefetch_local

        def _observed_release():
            order.append("retired-released")
            return real_release()

        cache._release_retired_prefetch_local = _observed_release

        cache.detach_storage_backend()

        self.assertIn("threads-joined", order)
        self.assertIn("retired-released", order)
        self.assertLess(
            order.index("threads-joined"),
            order.index("retired-released"),
            "the retired span was released while the transfer threads could "
            "still have been writing into it",
        )

    def test_a_current_generation_span_is_actually_freed_before_detach_returns(self):
        """EFFECTIVENESS, and this is the half a plausible fix gets wrong.

        Without a cutover the release is QUEUED on `host_mem_release_queue`
        rather than settled immediately. A release appended AFTER the detach's
        final control-queue drain would sit on that queue forever -- the holder
        would look released, the counter would report it, and the rows would
        still be gone. Only a release placed BEFORE that drain actually frees.
        """
        cache, pool_gen1, _, _, _ = _scenario(cutover=False)
        self.assertEqual(pool_gen1.free_slots.numel(), pool_gen1.size - SPAN)

        cache.detach_storage_backend()

        self.assertTrue(
            cache.cache_controller.host_mem_release_queue.empty(),
            "the release was queued but nothing drained it",
        )
        self.assertEqual(pool_gen1.free_slots.numel(), pool_gen1.size)


class TestTheLiveRecordIsNeverTouched(CustomTestCase):
    """THE MUTANT GUARD, on the dangerous direction.

    Releasing a live prefetch is strictly worse than leaking a retired one: the
    retired record published NOTHING to the tree, so every row it holds is
    unclaimed, whereas a live record's span may still be a transfer destination
    and may still be adopted by `check_prefetch_progress`. A fix that drains
    `ongoing_prefetch` "while it is in there anyway" is a use-after-free and a
    double-free, and it fails here.
    """

    def setUp(self):
        self.addCleanup(binding_state().reset)

    def _with_live_record(self):
        cache, pool_gen1, pool_gen2, _, _ = _scenario()
        live_span = cache.cache_controller.mem_pool_host.alloc(SPAN)
        live_op = PrefetchOperation(
            request_id=LIVE_REQ_ID,
            host_indices=live_span,
            token_ids=list(range(1, SPAN + 1)),
        )
        cache.ongoing_prefetch[LIVE_REQ_ID] = _OngoingPrefetch(
            cache.root_node,
            RadixKey(list(range(1, SPAN + 1))),
            live_span,
            live_op,
            None,
            {},
        )
        return cache, pool_gen1, pool_gen2, live_op

    def test_a_live_record_keeps_its_slot(self):
        cache, _, _, live_op = self._with_live_record()

        cache.detach_storage_backend()

        self.assertIn(LIVE_REQ_ID, cache.ongoing_prefetch)
        self.assertIs(cache.ongoing_prefetch[LIVE_REQ_ID].operation, live_op)

    def test_a_live_span_is_not_freed(self):
        cache, _, pool_gen2, _ = self._with_live_record()
        free_after_live_alloc = pool_gen2.free_slots.numel()

        cache.detach_storage_backend()

        self.assertEqual(pool_gen2.free_slots.numel(), free_after_live_alloc)

    def test_a_live_operation_is_not_marked_terminated(self):
        cache, _, _, live_op = self._with_live_record()

        cache.detach_storage_backend()

        self.assertFalse(live_op.is_terminated())


class TestTheDetachReportsWhatItReleased(CustomTestCase):
    """The probe is UNCONDITIONAL, which is the #962a lesson applied here: a
    line emitted only on a find cannot distinguish "nothing to release" from
    "this code never ran". The window reads the counter off this line."""

    def setUp(self):
        self.addCleanup(binding_state().reset)

    def test_the_marker_line_names_the_count(self):
        cache, _, _, _, _ = _scenario()

        with self.assertLogs(
            "sglang.srt.mem_cache.unified_radix_cache", level="INFO"
        ) as caught:
            cache.detach_storage_backend()

        text = "\n".join(caught.output)
        self.assertIn("#966 RETIRED PREFETCH RELEASED AT DETACH", text)
        self.assertIn("released=1", text)

    def test_the_marker_line_is_emitted_even_with_nothing_to_release(self):
        cache, _, _, _, _ = _scenario()
        cache._retired_prefetch.clear()

        with self.assertLogs(
            "sglang.srt.mem_cache.unified_radix_cache", level="INFO"
        ) as caught:
            cache.detach_storage_backend()

        text = "\n".join(caught.output)
        self.assertIn("#966 RETIRED PREFETCH RELEASED AT DETACH", text)
        self.assertIn("released=0", text)

    def test_a_second_detach_releases_nothing_and_frees_nothing_twice(self):
        """Detach is idempotent by contract. A double free of the same span is
        the W35 corruption this release is routed to avoid."""
        cache, pool_gen1, _, _, _ = _scenario()

        cache.detach_storage_backend()
        free_after_first = pool_gen1.free_slots.numel()
        cache.detach_storage_backend()

        self.assertEqual(pool_gen1.free_slots.numel(), free_after_first)
        self.assertEqual(cache._retired_prefetch, [])


if __name__ == "__main__":
    unittest.main()
