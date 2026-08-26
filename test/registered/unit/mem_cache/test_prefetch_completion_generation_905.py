"""#905 -- a prefetch's host slots outlive the tier that minted them.

THE SPECIMEN (R6 diagnostic boot, 2026-08-26 18:15Z, all three ranks, six of
six completions), from ``unified_radix_cache.py``'s own #905 comment block::

    free-site: unclaimed_to=49 | pool id now=138604341884544 epoch 2
               size 30518 | at registration id=138604342978576 epoch 3
               size 703472 | MOVED=True
    -> Double-free: 49 of 49 in range but not allocated, span [0, 48],
       free_slots=30518 (i.e. the pool being freed against holds NOTHING)

A ``PrefetchOperation`` allocates its host slots from the host tier bound at
registration time (#719 generation N). A phase cutover then rebinds
``cache_controller.mem_pool_host`` to a DIFFERENT pool object (generation
N+1) before the prefetch's ``check_prefetch_progress`` completion runs. A raw
``.free()`` against ``mem_pool_host`` at that point frees against the pool
bound NOW, not the one that minted the indices -- and because 49 < 30518 the
indices are IN RANGE for the wrong pool, so the #718 index-axis guard
(628d9705b1, orphaned off this train) cannot see the corruption: same root,
the other side of the same axis. Only the pre-existing #719 authority
(``operation.binding_generation``, stamped at construction) names which pool
actually owns the slots.

THE FIX, b2df698e2b: route both completion-time frees through
``HiCacheController.append_host_mem_release(host_indices, generation=
operation.binding_generation)`` instead of calling ``mem_pool_host.free()``
directly. That routing function already existed (W35, #719) with exactly one
consumer before this fix (``_drain_revoke``); the fix adds the completion
free site as a second consumer of the same one authority.

WHAT THIS FILE DRIVES. Not a hand-modelled double: the REAL
``UnifiedRadixCache.check_prefetch_progress`` (the method the fix lives
inside), two REAL ``MHATokenToKVPoolHost`` instances standing in for
generation N and N+1, the REAL ``hicache_phase_binding`` generation state
machine, and a REAL ``PrefetchOperation`` (the hybrid-cache one, since that
is the concrete class ``check_prefetch_progress`` actually reads
``operation.pool_storage_result`` and ``operation.binding_generation`` off
of) that self-stamps its generation at construction exactly as production
does. Only the surrounding ``cache_controller`` is a lightweight stand-in
(a ``SimpleNamespace``) carrying the two REAL bound methods
(``append_host_mem_release``, ``terminate_prefetch``) the fix and its
routing depend on -- the pattern already used by
``test_release_outlives_binding_w35.py`` for the sibling W35 mechanism.

Because the test calls ``check_prefetch_progress`` itself rather than
``append_host_mem_release`` directly, reverting the two-line fix in
``unified_radix_cache.py`` back to a raw ``mem_pool_host.free()`` call flips
this test red: mutation-sensitivity was verified by hand (see the #905
closure notes) before this file was committed.

THE LANDING SETUP. To reach a nonzero, freed span at all, the fetched tail
must be REFUSED by the #841 contiguous-backup law (the walk lands on an
existing, unbacked device-only node) so that ``insert_result
.host_span_unclaimed`` is True and the whole completed span becomes
``unclaimed_to`` -- exactly the '49 of 49' shape in the specimen. This reuses
the device-prefix setup from ``test_prefetch_landing_condition_843.py``.
"""

import queue
import types
import unittest

import torch

from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import InsertParams
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

# ~4s: one tiny CPU-only radix tree plus two tiny CPU host pools. No
# accelerator, no group, no boot.
register_cpu_ci(est_time=4, suite="base-a-test-cpu")

DEVICE_PREFIX = 3  # tokens the device tree already holds (unbacked on host)
FETCH = 8  # tokens the prefetch completes with; all refused, all freed
PAGE_SIZE = 1
REQ_ID = "req-905"


def _build_cache() -> UnifiedRadixCache:
    """A real UnifiedRadixCache on the CPU, the #841/#843 harness."""
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


def _make_host_pool(device_pool, ratio: float, label: str) -> MHATokenToKVPoolHost:
    """A real, tiny CPU host pool. ``host_size`` is a GiB-scale absolute size
    (see ``HostKVCache.__init__``), so sizing is controlled via
    ``host_to_device_ratio`` against a small device pool instead -- exactly
    what ``test_mem_pool_host.py`` does."""
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


def _scenario():
    """Build the #905 shape: a prefetch opened under generation N, completing
    after a cutover has rebound ``mem_pool_host`` to generation N+1's pool.

    Returns ``(cache, pool_gen1, pool_gen2, host_indices)`` so tests can drive
    ``cache.check_prefetch_progress(REQ_ID)`` and then inspect both pools.
    """
    binding_state().reset()

    cache = _build_cache()
    device_pool = cache.token_to_kv_pool_allocator.get_kvcache()

    # device-only prefix already resident: the fetched tail below lands on an
    # unbacked node and is refused whole (#841), which is what makes
    # unclaimed_to the FULL completed span -- the specimen's "49 of 49".
    cache.insert(
        InsertParams(
            key=RadixKey(list(range(1, DEVICE_PREFIX + 1))),
            value=torch.arange(DEVICE_PREFIX, dtype=torch.int64),
        )
    )

    # device_pool.size == 64, page_size 1: ratio 0.5 -> 33 slots, ratio
    # 0.125 -> 9 slots (both after the +1 page-alignment HostKVCache applies).
    pool_gen1 = _make_host_pool(device_pool, 0.5, "gen1")
    pool_gen2 = _make_host_pool(device_pool, 0.125, "gen2")

    gen1 = binding_state().advance("pp", host_pool=pool_gen1)

    host_indices = pool_gen1.alloc(FETCH)

    operation = PrefetchOperation(
        request_id=REQ_ID,
        host_indices=host_indices,
        token_ids=list(range(1, FETCH + 1)),
    )
    operation.hash_value = [f"h{i}" for i in range(FETCH // PAGE_SIZE)]
    operation.increment(FETCH)
    assert operation.binding_generation == gen1

    # THE CUTOVER: a phase rebind repoints mem_pool_host at a DIFFERENT,
    # narrower pool. The operation above was opened under gen1 and does not
    # move with it -- exactly the #905 mechanism.
    binding_state().advance("tp", host_pool=pool_gen2)

    fake_cc = types.SimpleNamespace(
        mem_pool_host=pool_gen2,
        host_mem_release_queue=queue.Queue(),
        prefetch_tokens_occupied=FETCH,
        write_policy="write_through_selective",
    )
    # Real bound methods: the routing logic under test (append_host_mem_release)
    # and the trivial completion bookkeeping it needs (terminate_prefetch) are
    # both the actual HiCacheController implementations, not doubles.
    fake_cc.terminate_prefetch = HiCacheController.terminate_prefetch.__get__(fake_cc)
    fake_cc.append_host_mem_release = HiCacheController.append_host_mem_release.__get__(
        fake_cc
    )
    cache.cache_controller = fake_cc

    cache.ongoing_prefetch[REQ_ID] = _OngoingPrefetch(
        cache.root_node,
        RadixKey(list(range(1, FETCH + 1))),
        host_indices,
        operation,
        None,
        {},
    )
    return cache, pool_gen1, pool_gen2, host_indices


class TestPrefetchCompletionFreesAgainstTheMintingGeneration(CustomTestCase):
    """The #905 fix, driven through the real completion path it lives in."""

    def setUp(self):
        self.addCleanup(binding_state().reset)

    def test_the_free_lands_on_generation_ns_pool(self):
        cache, pool_gen1, pool_gen2, _ = _scenario()
        self.assertEqual(pool_gen1.free_slots.numel(), pool_gen1.size - FETCH)

        result = cache.check_prefetch_progress(REQ_ID)

        self.assertTrue(result)
        # The slots return to the pool that actually minted them (gen1),
        # fully -- this is the routed free, not a partial/queued one.
        self.assertEqual(pool_gen1.free_slots.numel(), pool_gen1.size)

    def test_generation_n_plus_1s_pool_is_never_touched(self):
        cache, _, pool_gen2, _ = _scenario()
        free_before = pool_gen2.free_slots.numel()
        self.assertEqual(free_before, pool_gen2.size)

        cache.check_prefetch_progress(REQ_ID)

        # THE SPECIMEN, avoided: a raw free() here would have asserted
        # (indices in range for gen2 but not allocated there). Routing must
        # not touch gen2's bookkeeping at all -- not even partially.
        self.assertEqual(pool_gen2.free_slots.numel(), free_before)
        self.assertEqual(int(pool_gen2.slot_used.sum()), 0)

    def test_the_request_is_cleared_from_ongoing_prefetch(self):
        cache, _, _, _ = _scenario()
        cache.check_prefetch_progress(REQ_ID)
        self.assertNotIn(REQ_ID, cache.ongoing_prefetch)


class TestTheNaiveFreeWouldHaveDoubleFreed(CustomTestCase):
    """What the pre-fix raw ``mem_pool_host.free()`` call would have done.

    Not a hand-modelled double: the same REAL ``MHATokenToKVPoolHost`` from
    the scenario above, freed the naive way. If this ever stops raising, the
    scenario has stopped modelling the real pool and the tests above are not
    proving what they claim to.
    """

    def setUp(self):
        self.addCleanup(binding_state().reset)

    def test_freeing_against_the_rebound_pool_directly_double_frees(self):
        _, pool_gen1, pool_gen2, host_indices = _scenario()
        with self.assertRaises(AssertionError) as caught:
            pool_gen2.free(host_indices)
        msg = str(caught.exception)
        self.assertIn("Double-free detected", msg)
        # In range for gen2 (size 9 > max index 7) but never allocated there
        # -- the specimen's "49 of 49 in range but not allocated" shape.
        self.assertLess(int(host_indices.max()), pool_gen2.size)


class TestTheRoutingIsWiredToTheRealAuthority(CustomTestCase):
    """Pins the wiring itself, so a refactor that silently drops the
    generation stamp is caught even if the scenario above stops exercising
    it for some incidental reason."""

    def test_the_fix_reads_operations_own_binding_generation(self):
        import inspect

        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        src = inspect.getsource(UnifiedRadixCache.check_prefetch_progress)
        self.assertIn('getattr(operation, "binding_generation", None)', src)
        self.assertIn("append_host_mem_release(", src)
        self.assertNotIn("mem_pool_host.free(host_indices[:unclaimed_to])", src)

    def test_both_completion_frees_pass_the_same_generation(self):
        import inspect

        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        src = inspect.getsource(UnifiedRadixCache.check_prefetch_progress)
        self.assertEqual(src.count("generation=_binding_generation"), 2)


if __name__ == "__main__":
    unittest.main()
