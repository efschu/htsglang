"""#1417b (NF-Abnahme rc11b, 26.09. 19:30:47Z): the idle sanity walk kills a
rank while a #1417 prefetch pin legitimately holds a mamba anchor off the host
LRU.

MEASURED (D log boot_weg2_dkrnfh91bar1rc11b09261915, TP1; TP2 identical):

    19:30:47 #1423 INSERT-PLACED req=weg2-16- ... matched=2560 inserted=12480 deepest=249
    19:30:47 HiCache prefetch success req=weg2-16-31 ... loaded=12480
    19:30:47 #1423 INSERT-PLACED req=weg2-16- ... matched=15040 inserted=128 deepest=250
    19:30:47 HiCache prefetch success req=weg2-16-38 ... loaded=128
    19:30:47 Sanity check FAILED (1 violations across 7 nodes):
               mamba host LRU: +S3={249}, +lru=set()

debug-hold dump of the raising frame: ``s3_ids={249, 250, 245, 247}``,
``host_lru_ids={250, 245, 247}``.

THE MECHANISM, in the completion's own order (``check_prefetch_progress``:
``_insert_helper_host`` -> ``_pin_prefetched_span`` -> the components'
``commit_hicache_transfer(PREFETCH)``):

1. weg2-16-31 inserts its chain down to 249 and pins it BEFORE the mamba
   commit, so 249's mamba ``host_value`` is still None at pin time and the
   mamba lock is skipped. The commit then gives 249 its anchor and puts it on
   the mamba host LRU.
2. weg2-16-38 extends the same prefix by one node (250) and pins 250 -> 249
   -> ... . Now 249 HAS a mamba host copy, so ``acquire_component_lock(
   lock_host=True)`` takes 249 OFF the host LRU -- by design: a host-locked
   anchor must not be an eviction candidate.
3. Both requests wait for admission (D had just woken), the rank goes idle,
   ``sanity_check`` walks the tree and calls 249 an S3 node missing from the
   host LRU.

Nothing is inconsistent: 249 is exactly where the lock protocol puts it. The
check's S3 rule is upstream's, written for a tree in which no host lock
outlives a pass; #1417 pins span passes (until the admission pops them). The
rule now separates the two states the lock protocol defines: an UNLOCKED
host-only anchor must be on the host LRU, a host-LOCKED one must NOT be on it.
Both halves are checked -- a locked node left on the LRU is still named.

Not R12: the workers applied only no-op verdicts (``R12 HOST-VERDICT APPLIED
... kv_dropped=0 anchor_dropped=0``); the state arises on any rank whose idle
walk runs while two prefetch pins overlap on an anchored node.

Hermetic: real UnifiedRadixCache (FULL + MAMBA) on CPU, real insert, pin,
mamba commit, lock protocol and sanity_check. No CUDA.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(__file__)

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache_components.tree_component import (
    CacheTransferPhase,
    ComponentType,
)
from sglang.test.test_utils import CustomTestCase

from test_unified_radix_cache_unittest import CacheConfig, build_fixture

FULL, MAMBA = ComponentType.FULL, ComponentType.MAMBA
PAGE = 1
HEAD = 12  # tokens of the first prefetch (weg2-16-31)
TAIL = 4  # tokens the second prefetch adds below it (weg2-16-38)


class _Controller:
    """Only what the PREFETCH commit and the insert consult."""

    write_policy = "write_through"

    def append_host_mem_release(self, *args, **kwargs):
        return None


def _tree():
    cfg = CacheConfig(page_size=PAGE, components=(FULL, MAMBA))
    cache, _, _ = build_fixture(cfg)
    cache.cache_controller = _Controller()
    return cache


def _prefetch_complete(cache, rid, tokens, mamba_slot):
    """One storage prefetch completion, in check_prefetch_progress' order:
    insert the fetched span host-only, pin the inserted chain (#1417), then
    let the mamba component adopt its anchor (PREFETCH commit)."""
    key = RadixKey(list(tokens), None)
    host = torch.arange(1000, 1000 + len(tokens), dtype=torch.int64)
    hashes = [f"h{t}" for t in tokens]
    res = cache._insert_helper_host(cache.root_node, key, host, hashes)
    cache._pin_prefetched_span(rid, res.inserted_host_node, cache.root_node)
    cache.components[MAMBA].commit_hicache_transfer(
        cache.root_node,
        CacheTransferPhase.PREFETCH,
        [PoolTransfer(name=PoolName.MAMBA, host_indices=torch.tensor([mamba_slot]))],
        insert_result=res,
        pool_storage_result=SimpleNamespace(extra_pool_hit_pages={PoolName.MAMBA: 1}),
    )
    return res.inserted_host_node


def _metal_form(cache):
    """weg2-16-31 then weg2-16-38 on the same prefix, both still unadmitted."""
    n249 = _prefetch_complete(cache, "weg2-16-31", range(1, HEAD + 1), 7)
    n250 = _prefetch_complete(cache, "weg2-16-38", range(1, HEAD + TAIL + 1), 8)
    return n249, n250


class PrefetchPinHostLru1417b(CustomTestCase):
    def test_the_metal_state_is_the_lock_protocols_own(self):
        cache = _tree()
        n249, n250 = _metal_form(cache)
        cd = n249.component_data[MAMBA]
        host_lru = cache.host_lru_lists[MAMBA]
        # The rc11b state, exactly: anchor on host, off the device, host-locked
        # by the second pin, therefore not on the host LRU; its child is.
        self.assertIsNone(cd.value)
        self.assertIsNotNone(cd.host_value)
        self.assertEqual(cd.host_lock_ref, 1)
        self.assertFalse(host_lru.in_list(n249))
        self.assertTrue(host_lru.in_list(n250))

    def test_the_idle_walk_accepts_a_pinned_anchor(self):
        """RED before the fix: 'mamba host LRU: +S3={249}, +lru=set()'."""
        cache = _tree()
        _metal_form(cache)
        cache.sanity_check()

    def test_admission_returns_the_anchor_to_the_lru(self):
        cache = _tree()
        n249, _ = _metal_form(cache)
        cache.pop_prefetch_loaded_tokens("weg2-16-31")
        cache.pop_prefetch_loaded_tokens("weg2-16-38")
        self.assertEqual(n249.component_data[MAMBA].host_lock_ref, 0)
        self.assertTrue(cache.host_lru_lists[MAMBA].in_list(n249))
        cache.sanity_check()

    def test_a_locked_anchor_left_on_the_lru_is_still_named(self):
        """The check is not weakened: the other half of the lock protocol
        (a host-locked node must be OFF the LRU) is enforced now."""
        cache = _tree()
        n249, _ = _metal_form(cache)
        cache.host_lru_lists[MAMBA].insert_mru(n249)
        with self.assertRaises(AssertionError) as ctx:
            cache.sanity_check()
        self.assertIn("host-locked", str(ctx.exception))

    def test_an_unlocked_anchor_missing_from_the_lru_is_still_named(self):
        cache = _tree()
        n249, _ = _metal_form(cache)
        cache.pop_prefetch_loaded_tokens("weg2-16-31")
        cache.pop_prefetch_loaded_tokens("weg2-16-38")
        cache.host_lru_lists[MAMBA].remove_node(n249)
        with self.assertRaises(AssertionError) as ctx:
            cache.sanity_check()
        self.assertIn(f"+S3={{{n249.id}}}", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
