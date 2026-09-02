"""#1068 WEG 1 slice 2 fix 4 (WEG1_BUILD_SPEC_0901 A12.4 AMENDMENT (c), review
round 4 finding B2): ``drop_prefix_tree_returning_rows`` lets a failed
``tree.reset()`` PROPAGATE instead of swallowing it.

THE DEFECT (verified on e5b7eb3b79). The #856 wrapper around ``tree.reset()``
at the cutover (phase_flip_runtime.py, ``except Exception: logger.error(...)``
and return) catches EVERY exception the reset raises. Since slice 2 the
reset chain is ``UnifiedRadixCache._reset_full`` -> ``cache_controller.reset()``
-> ``_stop_storage_threads``, and that helper raises RuntimeError when a
storage thread is still alive after termination plus the join bound (the
torn-pipeline crash-stop of A12.4). ``_reset_full`` has already set
``enable_storage = False`` and cleared ``ongoing_prefetch`` by then, so on
the swallowing form the rank CONTINUED with enable_storage False, the stop
event set and no storage threads, and its next ``prefetch_from_storage``
returned before the #580 vote: a rank-divergent wedge, on one rank, instead
of a crash-stop on all of them. raenge-nie-uneins (ranks-never-disagree): a
torn rank crashes, it never compensates.

THE FIX: the wrapper still prints its #856 line (the instrument stays) and
then re-raises. The docstring sentence 'Never raises' is corrected: it never
raises on an eviction or count shortfall (those are logged), but a failed
reset propagates.

Hermetic: a REAL ``UnifiedRadixCache`` (CPU pools, one component), driven
through the real ``drop_prefix_tree_returning_rows`` and the real
``_reset_full``; only the cache controller is a stand-in whose ``reset()``
raises the exact RuntimeError ``_stop_storage_threads`` raises.

RED on e5b7eb3b79: the drop returns 0 and only logs. GREEN after fix 4.

    CUDA_VISIBLE_DEVICES='' PYTHONPATH=python python -m pytest \\
        test/registered/unit/managers/test_drop_tree_propagates_storage_stop_1068.py -q
"""

import types
import unittest

import torch

from sglang.srt.managers import phase_flip_runtime
from sglang.srt.managers.phase_flip_runtime import drop_prefix_tree_returning_rows
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.mem_cache.unified_cache_components.tree_component import (
    ComponentType,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

# ~3s: one tiny CPU-only radix tree, no accelerator, no group, no boot.
register_cpu_ci(est_time=3, suite="base-a-test-cpu")

POOL = 64
PAGE_SIZE = 1
# The exact message `HiCacheController._stop_storage_threads` raises.
STOP_FAILURE = "Failed to stop HiCache storage threads cleanly."


def _cache() -> UnifiedRadixCache:
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy", page_size=1))
    return UnifiedRadixCache(
        params=CacheInitParams(
            req_to_token_pool=ReqToTokenPool(
                size=8, max_context_len=64, device="cpu", enable_memory_saver=False
            ),
            token_to_kv_pool_allocator=TokenToKVPoolAllocator(
                size=POOL,
                dtype=torch.float16,
                device="cpu",
                kvcache=MHATokenToKVPool(
                    size=POOL,
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


class _Controller:
    """The three things ``_reset_full`` touches on the controller: ``reset()``,
    ``mem_pool_host.clear()`` and ``enable_storage``."""

    def __init__(self, raise_on_reset: bool):
        self.raise_on_reset = raise_on_reset
        self.reset_calls = 0
        self.mem_pool_host = types.SimpleNamespace(clear=lambda: None)
        self.enable_storage = True

    def reset(self):
        self.reset_calls += 1
        if self.raise_on_reset:
            raise RuntimeError(STOP_FAILURE)


class TestTheStorageStopFailurePropagatesOutOfTheDrop(CustomTestCase):
    def test_a_failed_storage_stop_propagates_through_the_drop(self):
        cache = _cache()
        cc = _Controller(raise_on_reset=True)
        cache.cache_controller = cc
        with self.assertLogs(
            phase_flip_runtime.logger, level="ERROR"
        ) as logs, self.assertRaises(RuntimeError) as raised:
            drop_prefix_tree_returning_rows(cache)
        self.assertEqual(str(raised.exception), STOP_FAILURE)
        self.assertEqual(cc.reset_calls, 1)
        # The #856 instrument line still fires before the raise.
        self.assertTrue(
            any("prefix tree reset failed" in m for m in logs.output), logs.output
        )
        # And THIS is the torn state that must not serve: _reset_full cleared
        # enable_storage before the controller raised, and the rebuild half
        # of the controller's reset() never ran.
        self.assertFalse(cache.enable_storage)

    def test_a_clean_reset_still_returns_the_row_count(self):
        """The companion: with a controller whose reset() returns, the drop
        returns its row count exactly as before (nothing else changed)."""
        cache = _cache()
        cc = _Controller(raise_on_reset=False)
        cache.cache_controller = cc
        self.assertEqual(drop_prefix_tree_returning_rows(cache), 0)
        self.assertEqual(cc.reset_calls, 1)
        self.assertTrue(cache.enable_storage)

    def test_the_docstring_no_longer_promises_never_raises(self):
        doc = drop_prefix_tree_returning_rows.__doc__ or ""
        self.assertNotIn("Never raises", doc)
        self.assertIn("propagat", doc.lower())


if __name__ == "__main__":
    unittest.main()
