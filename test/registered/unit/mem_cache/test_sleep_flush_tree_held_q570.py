"""Q-570 (NF y8s, 03.10. 09:53:58Z): D TP1 died at the idle sanity walk with
"pool memory leak detected! [full] total=524288, available=8192,
evictable=478016" and "[mamba] total=38, available=38, evictable=22,
free_and_cached=22, #924 MAMBA SLOT ALIASING: mamba_num_used=-22".

THE METAL SEQUENCE (D log boot_weg2_dkrnfint4h6ablbar1dauer10030924_044316dd1a_1003_092457):

* 09:53:40 the front's quiesce answered "PDFLIP-FLUSH-NONBLOCK quiesced" (B1): tree, pools and
  in-flight writes kept, "the sleep leg drains, publishes, joins and resets BEFORE the kv_cache pause".
* 09:53:41 TP0/TP2: the sleep leg's flush_cache(zero_kv=False) reset (TP0 "Cache flushed successfully!").
* 09:53:43 TP1: the same flush ran its #1470 sweep (issued=1 in_flight_after=2) and then REFUSED
  rank-locally: "Cache not flushed ... not-idle because: hicache_backup(2) | single-rank verdict".
  The release leg ignored the return value and paused kv_cache anyway -- TP1 slept with a tree whose
  nodes still held device KV indices and 22 device mamba slots ("PDFLIP-DORMANT set" one line later).
* 09:53:58 the wake: "Reset HybridReqToTokenPool", "PDFLIP-WAKE-RESTORE pools cleared, radix tree KEPT"
  -- every device slot the kept tree references is now ALSO in the free lists. The first idle pass
  after the W50-REROUTE park raised the leak; TREE CENSUS on TP1 still showed MAMBA tracked_evictable=22.

Hermetic: the real UnifiedRadixCache (FULL + MAMBA) with the real CPU pools of
test_unified_radix_cache_unittest. Nothing here touches a GPU.
"""

from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(__file__)

import inspect
import unittest
from array import array
from types import SimpleNamespace

from flliper.srt.managers.schedule_batch import Req
from flliper.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager as WU,
)
from flliper.srt.mem_cache.base_prefix_cache import InsertParams
from flliper.srt.mem_cache.radix_cache import RadixKey
from flliper.srt.mem_cache.unified_cache_components.tree_component import (
    ComponentType,
)
from flliper.srt.sampling.sampling_params import SamplingParams
from flliper.test.test_utils import CustomTestCase

from test_unified_radix_cache_unittest import CacheConfig, build_fixture

FULL = ComponentType.FULL
MAMBA = ComponentType.MAMBA
SPAN = 48


def _metal_tree():
    """One finished D request's span on the DEVICE: full KV indices and its mamba anchor slot
    owned by the tree (the y8s TP1 tree at the refused sleep flush, in small)."""
    cfg = CacheConfig(page_size=1, components=(FULL, MAMBA))
    cache, allocator, r2t = build_fixture(cfg)
    req = Req(
        rid="pdflip-36-194",
        origin_input_text="",
        origin_input_ids=array("q"),
        sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
    )
    r2t.alloc([req])
    value = allocator.alloc(SPAN)
    cache.insert(
        InsertParams(
            key=RadixKey(array("q", range(1, SPAN + 1))),
            value=value,
            mamba_value=req.mamba_pool_idx.unsqueeze(0),
        )
    )
    return cache, allocator, r2t


def _aliased(cache, allocator, r2t):
    """The invariant checker's two sums: a slot counted free AND tree-held."""
    full = allocator.available_size() + cache.full_evictable_size() + cache.full_protected_size()
    mamba = (
        r2t.mamba_allocator.available_size()
        + cache.mamba_evictable_size()
        + cache.mamba_protected_size()
    )
    return full - allocator.size, mamba - r2t.mamba_pool.size


def _updater(cache, allocator, r2t, flush=None):
    calls = []
    sched = SimpleNamespace(
        req_to_token_pool=r2t,
        token_to_kv_pool_allocator=allocator,
        tree_cache=cache,
        draft_worker=None,
        enable_hierarchical_cache=False,
    )

    def _flush(**kw):
        calls.append(kw)
        return flush(**kw) if flush is not None else True

    return SimpleNamespace(scheduler=sched, flush_cache=_flush), calls


class WakeRestoreOnAKeptDeviceTree(CustomTestCase):
    def test_the_metal_form_before_the_wake_is_consistent(self):
        cache, allocator, r2t = _metal_tree()
        self.assertEqual(cache.full_evictable_size(), SPAN)
        self.assertEqual(cache.mamba_evictable_size(), 1)
        self.assertEqual(_aliased(cache, allocator, r2t), (0, 0))

    def test_wake_restore_never_clears_pools_under_device_values(self):
        """RED before the fix: the restore cleared both pools under the kept tree
        (full over-credited by SPAN, mamba by 1 -- the y8s 'free_and_cached=22')."""
        cache, allocator, r2t = _metal_tree()
        upd, _ = _updater(cache, allocator, r2t)
        try:
            WU._pdflip_wake_restore_pools(upd)
        except RuntimeError as exc:
            self.assertIn("W26b", str(exc))
        self.assertEqual(_aliased(cache, allocator, r2t), (0, 0))

    def test_wake_restore_on_a_reset_tree_clears_and_keeps_the_tree(self):
        cache, allocator, r2t = _metal_tree()
        cache.reset()
        upd, calls = _updater(cache, allocator, r2t)
        self.assertTrue(WU._pdflip_wake_restore_pools(upd))
        self.assertEqual(calls, [])  # no fallback flush on the #1455 path
        self.assertEqual(_aliased(cache, allocator, r2t), (0, 0))


class SleepFlushLeavesNoDeviceTree(CustomTestCase):
    def _guard(self):
        from flliper.srt.managers.pdflip_sleep_drain import sleep_flush_until_reset

        return sleep_flush_until_reset

    def test_a_refused_sleep_flush_is_drained_and_flushed_again(self):
        """RED before the fix (no guard): TP1's refused flush went straight to the pause."""
        guard = self._guard()
        cache, allocator, r2t = _metal_tree()
        verdicts = iter([False, True])  # hicache_backup(2), then idle after the group drain
        drains = []

        def flush():
            ok = next(verdicts)
            if ok:
                cache.reset()
                r2t.clear()
                allocator.clear()
            return ok

        retries = guard(flush=flush, tree=cache, drain=lambda: drains.append(1))
        self.assertEqual(retries, 1)
        self.assertEqual(drains, [1])
        self.assertEqual(cache.full_evictable_size() + cache.mamba_evictable_size(), 0)
        self.assertEqual(_aliased(cache, allocator, r2t), (0, 0))

    def test_a_reset_rank_follows_the_group_drain_without_a_second_flush(self):
        guard = self._guard()
        cache, allocator, r2t = _metal_tree()
        cache.reset()
        r2t.clear()
        allocator.clear()
        flushes, drains, votes = [], [], []
        group = iter([1, 0])  # TP1 still held in the first vote, drained in the second

        def group_max(values, *, label):
            votes.append((list(values), label))
            return [max(int(values[0]), next(group))]

        cache.hicache_group_max = group_max
        guard(flush=lambda: flushes.append(1) or True, tree=cache, drain=lambda: drains.append(1))
        self.assertEqual(flushes, [1])  # only the sleep leg's own flush
        self.assertEqual(drains, [1])  # the collective drain every rank posts
        self.assertEqual([v for v, _ in votes], [[0], [0]])
        self.assertTrue(all(lbl == "pdflip_sleep_flush/held" for _, lbl in votes))

    def test_a_tree_that_stays_held_refuses_by_name_before_the_pause(self):
        guard = self._guard()
        from flliper.srt.managers.pdflip_sleep_drain import (
            PdFlipSleepDrainRefused,
            PdFlipSleepFlushRefused,
        )

        cache, allocator, r2t = _metal_tree()
        drains = []
        with self.assertRaises(PdFlipSleepFlushRefused) as ctx:
            guard(flush=lambda: False, tree=cache, drain=lambda: drains.append(1), attempts=2)
        self.assertIsInstance(ctx.exception, PdFlipSleepDrainRefused)  # the W120 family
        self.assertIn("W120b", str(ctx.exception))
        self.assertIn("full=%d" % SPAN, str(ctx.exception))
        self.assertEqual(len(drains), 2)

    def test_the_release_leg_uses_the_guard_not_a_bare_flush(self):
        src = inspect.getsource(WU.release_memory_occupation)
        self.assertNotIn("self.flush_cache(zero_kv=False)", src)
        self.assertIn("self._pdflip_sleep_flush()", src)
        i_flush = src.index("self._pdflip_sleep_flush()")
        i_pause = src.index("self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_KV_CACHE)")
        self.assertLess(i_flush, i_pause)


if __name__ == "__main__":
    unittest.main()
