"""#904 (load-then-invalidate): a pinned row is never freed under its pin.

USER HYPOTHESIS, second half: "something invalidates the kv after it was
loaded and before it is read. same for mamba."

A lock/pin is exactly the mechanism that covers that window --
``load_back`` takes ``inc_lock_ref`` on the anchor BEFORE the H2D copy
completes (``hiradix_cache.py:1612``, ``hi_mamba_radix_cache.py`` likewise)
and releases it only once the ack drains. So "free a row whose lock_ref > 0"
IS the hypothesis, expressed in code.

TWO SITES, TWO DIFFERENT STATES, AND THE DIFFERENCE MATTERS
-----------------------------------------------------------
1. ``UnifiedRadixCache._evict_component_and_detach_lru`` is the LIVE funnel
   (registry.py:107-110 routes hybrid SSM + hicache to UnifiedRadixCache).
   Its callers all selected an unlocked node, so the ordering held -- but it
   held as a property of six callers, not of the site that frees. That is an
   ASSUMED ordering. It is now ENFORCED.

2. ``HiMambaRadixCache._free_device_mamba`` read ``mamba_lock_ref`` and, on
   finding it positive, cleared it and freed anyway -- treating a live pin as
   bookkeeping to correct. That is the defect shape outright. The class has
   NO construction site (verified: only its own ``class`` statement matches
   ``HiMambaRadixCache(``), so this is PRESENT-BUT-UNREACHABLE, not a live
   fault. It is fixed rather than left, because the middle state is the one
   that gets read wrong in both directions.

The family convention this restores is an ACT-time check:
``mamba_radix_cache.py:1149-1178`` / ``:1286-1299``,
``swa_radix_cache.py:600`` / ``:638``, ``hi_mamba_radix_cache.py:1137``.
"""

import unittest
from unittest.mock import MagicMock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestLiveFunnelRefusesToFreeUnderAPin(CustomTestCase):
    """``_evict_component_and_detach_lru``, the site that actually frees."""

    def _cache_and_node(self, lock_ref):
        from sglang.srt.mem_cache.unified_cache_components import ComponentType
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        lru = MagicMock()
        lru.in_list.return_value = False
        cache = object.__new__(UnifiedRadixCache)
        cache.lru_lists = {ComponentType.MAMBA: lru}
        cache.host_lru_lists = {ComponentType.MAMBA: lru}

        cd = MagicMock()
        cd.value = [0, 1, 2]
        cd.lock_ref = lock_ref
        node = MagicMock()
        node.id = 7
        node.component_data = {ComponentType.MAMBA: cd}

        comp = MagicMock()
        comp.component_type = ComponentType.MAMBA
        comp.evict_component.return_value = (3, 0)
        return cache, node, comp

    def test_free_under_a_pin_is_refused(self):
        """RED-FIRST for #904 load-then-invalidate."""
        from sglang.srt.mem_cache.unified_cache_components import EvictLayer

        cache, node, comp = self._cache_and_node(lock_ref=1)
        with self.assertRaises(ValueError) as ctx:
            cache._evict_component_and_detach_lru(
                node, comp, target=EvictLayer.DEVICE, tracker=None
            )
        self.assertIn("904", str(ctx.exception))
        self.assertIn("lock_ref=1", str(ctx.exception))
        comp.evict_component.assert_not_called()

    def test_unlocked_row_is_freed_as_before(self):
        """The can-fail direction: a refusal that also refuses the normal
        case is a capacity bug wearing a correctness costume."""
        from sglang.srt.mem_cache.unified_cache_components import EvictLayer

        cache, node, comp = self._cache_and_node(lock_ref=0)
        freed, host_freed = cache._evict_component_and_detach_lru(
            node, comp, target=EvictLayer.DEVICE, tracker=None
        )
        self.assertEqual((freed, host_freed), (3, 0))
        comp.evict_component.assert_called_once()

    def test_a_row_with_no_device_value_is_not_blocked(self):
        """A node whose device half is already gone carries a stale lock_ref
        for the HOST half's sake; blocking on it would wedge host eviction."""
        from sglang.srt.mem_cache.unified_cache_components import EvictLayer

        cache, node, comp = self._cache_and_node(lock_ref=2)
        from sglang.srt.mem_cache.unified_cache_components import ComponentType

        node.component_data[ComponentType.MAMBA].value = None
        cache._evict_component_and_detach_lru(
            node, comp, target=EvictLayer.DEVICE, tracker=None
        )
        comp.evict_component.assert_called_once()

    def test_host_only_eviction_is_untouched(self):
        """The device pin says nothing about the host tier, which has its own
        ``host_lock_ref``. Widening the check to HOST would be a different
        (and unproven) claim."""
        from sglang.srt.mem_cache.unified_cache_components import EvictLayer

        cache, node, comp = self._cache_and_node(lock_ref=5)
        comp.evict_component.return_value = (0, 4)
        cache._evict_component_and_detach_lru(
            node, comp, target=EvictLayer.HOST, tracker=None
        )
        comp.evict_component.assert_called_once()


class TestDeadPathNoLongerClearsThePin(CustomTestCase):
    """``HiMambaRadixCache._free_device_mamba``: unreachable today, but the
    defect shape outright. Fixed so that wiring the class later cannot
    resurrect it."""

    def _cache_and_node(self, lock_ref):
        from sglang.srt.mem_cache.hi_mamba_radix_cache import HiMambaRadixCache

        cache = object.__new__(HiMambaRadixCache)
        cache.req_to_token_pool = MagicMock()
        cache.mamba_lru_list = MagicMock()
        cache.mamba_lru_list.in_list.return_value = False
        cache.mamba_protected_size_ = 100
        cache.mamba_evictable_size_ = 100

        node = MagicMock()
        node.id = 11
        node.mamba_value = [3]
        node.mamba_lock_ref = lock_ref
        return cache, node

    def test_locked_slot_is_refused_not_unlocked(self):
        cache, node = self._cache_and_node(lock_ref=1)
        with self.assertRaises(ValueError) as ctx:
            cache._free_device_mamba(node)
        self.assertIn("904", str(ctx.exception))
        cache.req_to_token_pool.mamba_allocator.free.assert_not_called()
        self.assertEqual(
            node.mamba_lock_ref, 1, "the pin must survive the refusal, not be cleared"
        )
        self.assertIsNotNone(node.mamba_value)

    def test_unlocked_slot_is_freed(self):
        cache, node = self._cache_and_node(lock_ref=0)
        freed = cache._free_device_mamba(node)
        self.assertEqual(freed, 1)
        cache.req_to_token_pool.mamba_allocator.free.assert_called_once()
        self.assertIsNone(node.mamba_value)
        self.assertEqual(cache.mamba_evictable_size_, 99)

    def test_no_value_is_a_no_op(self):
        cache, node = self._cache_and_node(lock_ref=0)
        node.mamba_value = None
        self.assertEqual(cache._free_device_mamba(node), 0)
        cache.req_to_token_pool.mamba_allocator.free.assert_not_called()

    def test_the_class_is_still_unreachable(self):
        """The determination this fix is filed under. If a construction site
        appears, the fix above stops being belt-and-braces and starts being
        load-bearing -- and this test says so at that moment."""
        import subprocess

        out = (
            subprocess.run(
                [
                    "grep",
                    "-rn",
                    "--include=*.py",
                    "HiMambaRadixCache(",
                    "python/sglang/",
                ],
                cwd=_repo_root(),
                capture_output=True,
                text=True,
            )
            .stdout.strip()
            .splitlines()
        )
        sites = [line for line in out if "class HiMambaRadixCache(" not in line]
        self.assertEqual(
            sites,
            [],
            "HiMambaRadixCache gained a construction site; the #904 pin "
            "refusal in _free_device_mamba is now on a LIVE path and needs a "
            "boot-level check, not only this unit test",
        )


def _repo_root() -> str:
    import os

    import sglang

    return os.path.abspath(
        os.path.join(os.path.dirname(sglang.__file__), os.pardir, os.pardir)
    )


if __name__ == "__main__":
    unittest.main()
