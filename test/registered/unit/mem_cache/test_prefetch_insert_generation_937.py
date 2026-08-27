"""#937 -- a prefetch ADOPTS a fetched span whose host slots were minted by a
tier that no longer exists.

THE SPECIMEN (2j soak boot ``boot_2h_4e855cc80a_0827_1056.log``, all three
ranks).  Every prompt at or above 256 tokens returns garbage, non-deterministic
at ``temperature=0``; 255 and below are correct.  The only length-monotone 256
in that boot's config is the HiCache storage-prefetch gate
(``prefetch_threshold`` = 256 *tokens*, ``unified_radix_cache.py``), and the
completions it produced enumerate the failing lengths exactly::

    HiCache prefetch success req=9872efae... completed_local=300
        completed_synced=300 matched=0 loaded=300 refused=0
    #905 PREFETCH-COMPLETE free-site: req=9872efae... completed=300
        | pool id now=125027714091728 epoch now=6
        | at registration id=125027713888576 epoch=7 | MOVED=True

``MOVED=True`` says the host tier was rebound between the prefetch's
registration and its completion -- every request on this boot crosses a
``pp_to_tp`` cutover between PP prefill and TP re-admission, so this is the
common case, not a rare race.  ``refused=0`` says the fetched span was
nevertheless ADOPTED into the radix tree, and ``loaded=300`` says 300 tokens of
it are now advertised to every later match walk.  Those 300 host rows belong to
generation N's pool; the tree now names them against generation N+1's.

THE ASYMMETRY THIS PINS.  ``check_prefetch_progress`` already knows the
completion may have outlived its binding -- #905/#719 stamped the two
completion FREES with ``operation.binding_generation`` and routes them to the
pool that minted them (``append_host_mem_release(..., generation=...)``), and
the diagnostic block between the two sites prints ``MOVED`` unconditionally.
The INSERT sitting between them consults none of it.  One side of the axis was
closed; this is the other side, and it is the side that reaches the model.

Same authority, no second stamp scheme: ``write_back_stamp_is_current`` is the
predicate ``append_host_mem_release`` already uses.

WHAT THIS FILE DRIVES.  The real ``UnifiedRadixCache.check_prefetch_progress``,
two real ``MHATokenToKVPoolHost`` instances standing in for generation N and
N+1, the real ``hicache_phase_binding`` generation state machine and a real
``PrefetchOperation`` that self-stamps at construction -- the harness of
``test_prefetch_completion_generation_905.py``, with one deliberate change: the
device-only prefix is NOT planted, so the #841 contiguous-backup law does not
refuse the tail and the completion takes the ADOPTING landing.  That is the
``refused=0`` branch, which #905's file never exercises because its own
mechanism needs the refusing one.
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

# ~4s: one tiny CPU-only radix tree plus two tiny CPU host pools. No
# accelerator, no group, no boot.
register_cpu_ci(est_time=4, suite="base-a-test-cpu")

FETCH = 8  # tokens the prefetch completes with
PAGE_SIZE = 1
REQ_ID = "req-937"


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


def _make_host_pool(device_pool, ratio: float, label: str) -> MHATokenToKVPoolHost:
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


def _scenario(cutover: bool):
    """A prefetch completing on the ADOPTING landing (no device-only prefix,
    so #841 does not refuse the tail -- the specimen's ``refused=0``).

    ``cutover=True`` rebinds the host tier to a second pool between the
    operation's construction and its completion, reproducing ``MOVED=True``.
    ``cutover=False`` is the same path with the binding left alone, and is the
    control that must keep inserting.
    """
    binding_state().reset()

    cache = _build_cache()
    device_pool = cache.token_to_kv_pool_allocator.get_kvcache()

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

    if cutover:
        # THE CUTOVER, i.e. MOVED=True: pp_to_tp rebinds mem_pool_host to a
        # different pool object. The operation was opened under gen1 and does
        # not travel with it.
        binding_state().advance("tp", host_pool=pool_gen2)
        bound_now = pool_gen2
    else:
        bound_now = pool_gen1

    fake_cc = types.SimpleNamespace(
        mem_pool_host=bound_now,
        host_mem_release_queue=queue.Queue(),
        prefetch_tokens_occupied=FETCH,
        write_policy="write_through_selective",
    )
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


def _loaded(cache) -> int:
    return int(cache.prefetch_loaded_tokens_by_reqid.get(REQ_ID, 0))


class TestASupersededSpanIsNotAdopted(CustomTestCase):
    """THE RED TEST. A completion whose binding generation is no longer current
    must not publish its host span into the tree."""

    def setUp(self):
        self.addCleanup(binding_state().reset)

    def test_the_stale_span_is_not_loaded_into_the_tree(self):
        cache, _, _, _ = _scenario(cutover=True)

        self.assertTrue(cache.check_prefetch_progress(REQ_ID))

        # The specimen reported loaded=300 here. Nothing minted under a
        # superseded binding may be advertised to a later match walk.
        self.assertEqual(_loaded(cache), 0)

    def test_a_later_match_walk_cannot_reach_the_stale_rows(self):
        cache, _, _, _ = _scenario(cutover=True)
        cache.check_prefetch_progress(REQ_ID)

        # Walk the same key the prefetch fetched. It must find nothing: the
        # rows it would otherwise name belong to gen1's pool while every
        # reader is now on gen2's.
        node = cache.root_node
        self.assertEqual(len(node.children), 0)


class TestTheHostSlotsStillGoHomeExactlyOnce(CustomTestCase):
    """Direction-is-a-reader-property: dropping the insert must not turn a
    published span into a leaked one, nor into a double free. The slots were
    minted by gen1 and must return to gen1, whole, once."""

    def setUp(self):
        self.addCleanup(binding_state().reset)

    def test_the_minting_pool_gets_every_slot_back(self):
        cache, pool_gen1, _, _ = _scenario(cutover=True)
        self.assertEqual(pool_gen1.free_slots.numel(), pool_gen1.size - FETCH)

        cache.check_prefetch_progress(REQ_ID)

        self.assertEqual(pool_gen1.free_slots.numel(), pool_gen1.size)

    def test_the_currently_bound_pool_is_never_touched(self):
        cache, _, pool_gen2, _ = _scenario(cutover=True)
        free_before = pool_gen2.free_slots.numel()

        cache.check_prefetch_progress(REQ_ID)

        self.assertEqual(pool_gen2.free_slots.numel(), free_before)
        self.assertEqual(int(pool_gen2.slot_used.sum()), 0)

    def test_the_request_is_cleared_from_ongoing_prefetch(self):
        cache, _, _, _ = _scenario(cutover=True)
        cache.check_prefetch_progress(REQ_ID)
        self.assertNotIn(REQ_ID, cache.ongoing_prefetch)


class TestTheCurrentGenerationStillAdopts(CustomTestCase):
    """COUNTER-DIRECTION. The guard must refuse superseded spans only. A
    completion still riding its own binding keeps loading, or the fix has
    simply disabled storage prefetch."""

    def setUp(self):
        self.addCleanup(binding_state().reset)

    def test_a_same_generation_completion_loads_its_span(self):
        cache, _, _, _ = _scenario(cutover=False)

        self.assertTrue(cache.check_prefetch_progress(REQ_ID))

        self.assertEqual(_loaded(cache), FETCH)

    def test_a_same_generation_completion_publishes_to_the_tree(self):
        cache, _, _, _ = _scenario(cutover=False)
        cache.check_prefetch_progress(REQ_ID)
        self.assertGreater(len(cache.root_node.children), 0)


class TestTheGuardIsWiredToTheRealAuthority(CustomTestCase):
    """Pins the wiring, so a refactor that drops the generation check is caught
    even if the scenario above stops exercising it incidentally."""

    def test_the_insert_is_gated_on_the_same_predicate_as_the_free(self):
        import inspect

        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        src = inspect.getsource(UnifiedRadixCache.check_prefetch_progress)
        # The #905 free-side authority, reused rather than re-invented.
        self.assertIn("write_back_stamp_is_current", src)
        # ...and consulted BEFORE the span is inserted, not after.
        self.assertLess(
            src.index("write_back_stamp_is_current"),
            src.index("_insert_helper_host("),
        )


if __name__ == "__main__":
    unittest.main()
