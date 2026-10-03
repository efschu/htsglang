"""Q-590: the 27B port of Q-570 SLEEP-FLUSH-HELD (NF fix 390d5bfddc) into y8t, L15-aware.

THE NF CLASS (y8s D TP1, 03.10. 09:53:43 -> 09:53:58Z): the release leg's
flush_cache(zero_kv=False) refused rank-locally ("not-idle because: hicache_backup(2) |
single-rank verdict") after its own #1470 publish, the refusal was dropped, TP1 paused
kv_cache with device values in its tree while TP0/TP2 reset, and the wake's #1455 restore
("pools cleared, radix tree KEPT") cleared the pools under the kept tree: free_and_cached=22,
#924 MAMBA SLOT ALIASING, RANK-DEATH.

THE 27B SITE (desk/27b-y8t-integ-1003 f111e8f7c7): the same bare release flush
(weight_updater, before the kv_cache pause), B1 NONBLOCK on D, the KEPT restore in the
no-hold branch. Not a 1:1 port, because the L1.5 hold keeps chains WITH device values
across the flip on purpose:

* the retaining flush reduces the tree to exactly the held chains; their books are
  recorded there (``_l15_tree_retained_books``) and are not "held" for the guard;
* the W26b tripwire sits only in the restore's no-hold branch (after the drop of a
  retained tree), never in the hold-aware branch;
* the flush's idle branch posts the L15 SLEEP-AGREE gathers, so the release flush's
  verdict is group-reduced (``sleep_group_verdict``) and every rank re-flushes in a retry.

Hermetic: the real UnifiedRadixCache (FULL + MAMBA) with the real CPU pools of
test_unified_radix_cache_unittest. Nothing here touches a GPU.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(__file__)

import inspect
import unittest
from array import array
from types import SimpleNamespace

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager as WU,
)
from sglang.srt.mem_cache.base_prefix_cache import InsertParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache_components.tree_component import (
    ComponentType,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.test_utils import CustomTestCase

from test_unified_radix_cache_unittest import CacheConfig, build_fixture

FULL = ComponentType.FULL
MAMBA = ComponentType.MAMBA
SPAN = 48


def _insert(cache, allocator, r2t, rid, first, span):
    req = Req(
        rid=rid,
        origin_input_text="",
        origin_input_ids=array("q"),
        sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
    )
    r2t.alloc([req])
    value = allocator.alloc(span)
    cache.insert(
        InsertParams(
            key=RadixKey(array("q", range(first, first + span))),
            value=value,
            mamba_value=req.mamba_pool_idx.unsqueeze(0),
        )
    )
    return value, int(req.mamba_pool_idx)


def _metal_tree():
    """One finished D request's span on the DEVICE: full KV indices and its mamba anchor slot
    owned by the tree (the y8s TP1 tree at the refused sleep flush, in small -- and, with the
    retained flag, an L15 held chain as reset_keep leaves it)."""
    cfg = CacheConfig(page_size=1, components=(FULL, MAMBA))
    cache, allocator, r2t = build_fixture(cfg)
    value, mslot = _insert(cache, allocator, r2t, "weg2-36-194", 1, SPAN)
    cache._q590_value, cache._q590_mslot = value, mslot
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


def _updater(cache, allocator, r2t, *, master_on=False, manifest=None, retained=False):
    calls = []
    sched = SimpleNamespace(
        req_to_token_pool=r2t,
        token_to_kv_pool_allocator=allocator,
        tree_cache=cache,
        draft_worker=None,
        enable_hierarchical_cache=False,
        _l15_tree_retained=retained,
    )

    def _flush(**kw):
        calls.append(kw)
        return True

    upd = SimpleNamespace(
        scheduler=sched,
        flush_cache=_flush,
        # (manifest, rank, keep_rows, master_on) -- the restore's own L15 signal
        _l15_wake_hold_signal=lambda: (manifest, 0, 0, master_on),
        _l15_clear_tms_keep_spans=lambda _s: 0,
        _l15_flush_zero_kv_bounded=lambda _s, _k: None,
    )
    return upd, calls


def _guard():
    from sglang.srt.managers.weg2_sleep_drain import sleep_flush_until_reset

    return sleep_flush_until_reset


class WakeRestoreOnAKeptDeviceTree(CustomTestCase):
    def test_the_metal_form_before_the_wake_is_consistent(self):
        cache, allocator, r2t = _metal_tree()
        self.assertEqual(cache.full_evictable_size(), SPAN)
        self.assertEqual(cache.mamba_evictable_size(), 1)
        self.assertEqual(_aliased(cache, allocator, r2t), (0, 0))

    def test_no_hold_restore_never_clears_pools_under_device_values(self):
        """RED before the fix: the no-hold restore cleared both pools under the kept tree
        (full over-credited by SPAN, mamba by 1 -- the y8s 'free_and_cached=22')."""
        cache, allocator, r2t = _metal_tree()
        upd, calls = _updater(cache, allocator, r2t, master_on=True)
        try:
            WU._weg2_wake_restore_pools(upd)
        except RuntimeError as exc:
            self.assertIn("W26b", str(exc))
        self.assertEqual(_aliased(cache, allocator, r2t), (0, 0))
        self.assertEqual(calls, [])  # W26b is not swallowed into the fallback flush

    def test_w26b_propagates_by_name(self):
        from sglang.srt.managers.weg2_sleep_drain import Weg2WakeTreeHeld

        cache, allocator, r2t = _metal_tree()
        upd, calls = _updater(cache, allocator, r2t, master_on=False)
        with self.assertRaises(Weg2WakeTreeHeld) as ctx:
            WU._weg2_wake_restore_pools(upd)
        self.assertIn("full=%d mamba=1" % SPAN, str(ctx.exception))
        self.assertEqual(calls, [])

    def test_restore_on_a_reset_tree_clears_and_keeps_the_tree(self):
        cache, allocator, r2t = _metal_tree()
        cache.reset()
        upd, calls = _updater(cache, allocator, r2t, master_on=True)
        self.assertTrue(WU._weg2_wake_restore_pools(upd))
        self.assertEqual(calls, [])  # no fallback flush on the #1455 path
        self.assertEqual(_aliased(cache, allocator, r2t), (0, 0))

    def test_a_retained_tree_without_a_hold_here_is_dropped_not_refused(self):
        """L15-FIX-NOHOLD-TREE: the sleep retained on this rank but the wake keeps no hold
        here -- the drop runs BEFORE the tripwire, so no W26b and no aliasing."""
        cache, allocator, r2t = _metal_tree()
        upd, calls = _updater(cache, allocator, r2t, master_on=True, retained=True)
        self.assertTrue(WU._weg2_wake_restore_pools(upd))
        self.assertFalse(upd.scheduler._l15_tree_retained)
        self.assertEqual(cache.full_evictable_size() + cache.mamba_evictable_size(), 0)
        self.assertEqual(_aliased(cache, allocator, r2t), (0, 0))
        self.assertEqual(calls, [])

    def test_the_hold_aware_restore_keeps_the_held_chain_without_w26b(self):
        """The L15 hold branch: the held chain stays in the tree with its device values,
        its KV slots are re-reserved -- no tripwire there, and nothing aliases on full."""
        cache, allocator, r2t = _metal_tree()
        value = [int(v) for v in cache._q590_value.tolist()]
        manifest = SimpleNamespace(
            spans=[SimpleNamespace(slots=value)], anchor_slots=cache._q590_mslot + 1
        )
        upd, calls = _updater(
            cache, allocator, r2t, master_on=True, manifest=manifest, retained=True
        )
        self.assertTrue(WU._weg2_wake_restore_pools(upd))
        self.assertEqual(calls, [])
        self.assertEqual(cache.full_evictable_size(), SPAN)  # the hold stays
        self.assertEqual(cache.mamba_evictable_size(), 1)
        self.assertEqual(_aliased(cache, allocator, r2t)[0], 0)


class SleepFlushLeavesNoDeviceTree(CustomTestCase):
    def test_a_refused_sleep_flush_is_drained_and_flushed_again(self):
        """RED before the fix (no guard): TP1's refused flush went straight to the pause."""
        guard = _guard()
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

    def test_every_rank_flushes_again_after_the_group_drain(self):
        """27B: a reset rank re-flushes too -- the flush's idle branch posts the L15
        SLEEP-AGREE gathers, so a retry runs on all ranks or none (the NF form, holder only,
        would post them alone)."""
        guard = _guard()
        cache, allocator, r2t = _metal_tree()
        cache.reset()
        r2t.clear()
        allocator.clear()
        flushes, drains, votes = [], [], []
        group = iter([1, 0])  # a peer still held in the first vote, drained in the second

        def group_max(values, *, label):
            votes.append((list(values), label))
            return [max(int(values[0]), next(group))]

        cache.hicache_group_max = group_max
        guard(flush=lambda: flushes.append(1) or True, tree=cache, drain=lambda: drains.append(1))
        self.assertEqual(flushes, [1, 1])  # the sleep leg's flush + the uniform re-flush
        self.assertEqual(drains, [1])  # the collective drain every rank posts
        self.assertEqual([v for v, _ in votes], [[0], [0]])
        self.assertTrue(all(lbl == "weg2_sleep_flush/held" for _, lbl in votes))

    def test_a_tree_that_stays_held_refuses_by_name_before_the_pause(self):
        guard = _guard()
        from sglang.srt.managers.weg2_sleep_drain import (
            Weg2SleepDrainRefused,
            Weg2SleepFlushRefused,
        )

        cache, allocator, r2t = _metal_tree()
        drains = []
        with self.assertRaises(Weg2SleepFlushRefused) as ctx:
            guard(flush=lambda: False, tree=cache, drain=lambda: drains.append(1), attempts=2)
        self.assertIsInstance(ctx.exception, Weg2SleepDrainRefused)  # the W120 family
        self.assertIn("W120b", str(ctx.exception))
        self.assertIn("full=%d" % SPAN, str(ctx.exception))
        self.assertEqual(len(drains), 2)

    def test_the_release_leg_uses_the_guard_not_a_bare_flush(self):
        src = inspect.getsource(WU.release_memory_occupation)
        self.assertNotIn("self.flush_cache(zero_kv=False)", src)
        self.assertIn("self._weg2_sleep_flush()", src)
        i_flush = src.index("self._weg2_sleep_flush()")
        i_pause = src.index("self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_KV_CACHE)")
        self.assertLess(i_flush, i_pause)


class L15HoldChainIsNotHeld(CustomTestCase):
    """The L15 hold keeps its chains with device values across the flip on purpose: they
    must neither trip W120b at the sleep nor be flushed away by the guard."""

    def _retained_sched(self, cache):
        from sglang.srt.managers.weg2_sleep_drain import tree_device_held

        # what Scheduler.flush_cache records where it sets _l15_tree_retained
        return SimpleNamespace(
            tree_cache=cache,
            _l15_tree_retained=True,
            _l15_tree_retained_books=tree_device_held(cache),
        )

    def test_an_l15_hold_chain_passes_without_a_drain_and_stays(self):
        from sglang.srt.managers.weg2_sleep_drain import l15_retained_books

        guard = _guard()
        cache, allocator, r2t = _metal_tree()
        sched = self._retained_sched(cache)
        self.assertEqual(l15_retained_books(sched), (SPAN, 1))
        flushes, drains = [], []
        retries = guard(
            flush=lambda: flushes.append(1) or True,
            tree=cache,
            drain=lambda: drains.append(1),
            retained=lambda: l15_retained_books(sched),
            attempts=1,
        )
        self.assertEqual(retries, 0)  # no W120b, no WEG2-SLEEP-FLUSH-HELD round
        self.assertEqual((flushes, drains), ([1], []))
        self.assertEqual(cache.full_evictable_size(), SPAN)  # the hold is kept
        self.assertEqual(cache.mamba_evictable_size(), 1)
        self.assertEqual(_aliased(cache, allocator, r2t), (0, 0))

    def test_a_device_value_beyond_the_hold_still_counts(self):
        from sglang.srt.managers.weg2_sleep_drain import (
            Weg2SleepFlushRefused,
            l15_retained_books,
        )

        guard = _guard()
        cache, allocator, r2t = _metal_tree()
        sched = self._retained_sched(cache)
        _insert(cache, allocator, r2t, "weg2-36-198", 1000, 8)  # not part of the hold
        drains = []
        with self.assertRaises(Weg2SleepFlushRefused) as ctx:
            guard(
                flush=lambda: False,
                tree=cache,
                drain=lambda: drains.append(1),
                retained=lambda: l15_retained_books(sched),
                attempts=1,
            )
        msg = str(ctx.exception)
        self.assertIn("full=8 mamba=1 beyond the L15 hold full=%d mamba=1" % SPAN, msg)
        self.assertEqual(drains, [1])

    def test_no_hold_books_without_the_flag_or_the_record(self):
        from sglang.srt.managers.weg2_sleep_drain import l15_retained_books

        self.assertEqual(l15_retained_books(None), (0, 0))
        self.assertEqual(
            l15_retained_books(
                SimpleNamespace(_l15_tree_retained=False, _l15_tree_retained_books=(9, 1))
            ),
            (0, 0),
        )
        # the flag without its record subtracts nothing (strict side)
        self.assertEqual(l15_retained_books(SimpleNamespace(_l15_tree_retained=True)), (0, 0))

    def test_the_retaining_flush_records_the_hold_books(self):
        from sglang.srt.managers.scheduler import Scheduler

        body = inspect.getsource(Scheduler.flush_cache)
        i_flag = body.index("self._l15_tree_retained = _l15_res is not None")
        i_books = body.index("self._l15_tree_retained_books = (")
        self.assertLess(i_flag, i_books)
        self.assertIn("_q590_held(self.tree_cache) if _l15_res is not None else (0, 0)", body)

    def test_the_sleep_flush_verdict_is_group_reduced_without_b1(self):
        from sglang.srt.managers.scheduler import Scheduler

        sig = inspect.signature(Scheduler.flush_cache)
        self.assertIn("sleep_group_verdict", sig.parameters)
        self.assertIs(sig.parameters["sleep_group_verdict"].default, False)
        body = inspect.getsource(Scheduler.flush_cache)
        self.assertIn("tp_group_verdict=tp_group_verdict or sleep_group_verdict", body)
        # B1's quiesce answer and non-blocking sweep stay on the front's quiesce only
        self.assertIn("self, group_idle, tp_group_verdict\n", body)
        self.assertIn("self, {}, tp_group_verdict)", body)
        src = inspect.getsource(WU._weg2_sleep_flush)
        self.assertIn("self.flush_cache(zero_kv=False, sleep_group_verdict=True)", src)
        self.assertIn("retained=lambda: l15_retained_books(sch)", src)


if __name__ == "__main__":
    unittest.main()
