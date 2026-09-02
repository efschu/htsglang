"""#1068 WEG 1 slice 2 (WEG1_BUILD_SPEC_0901 section 4.2): the speculative
prefetch budget is the upstream ``buffer_only`` form -- a LIVE property of the
bound host pool, and a rate brake on LIVE occupancy.

THE DEFECT. ``prefetch_capacity_limit`` was a NUMBER stored once at storage
attach (``prefetch_capacity_limit_for(mem_pool_host.size)``), and the rate
brake compared the fork's own ``prefetch_tokens_occupied`` counter against
it. The phase flip rebinds ``mem_pool_host`` to a pool 23x smaller (measured
#905: 703472 rows PP vs 30518 TP), so the stored number described a pool the
controller no longer held, and the fork answered with a floor
(``PREFETCH_CAP_FLOOR_TOKENS``) and two symmetrize passes that re-derived the
number at each site -- second bookkeeping beside the upstream truth
(``upstream-minimal-statt-eigenbau``). Upstream (cache_controller.py:253 and
:575-581) makes the limit ``int(fraction * mem_pool_host.size)`` and the brake
``size - available_size() - write_staged >= limit`` (:1150-1163): no stored
number, nothing to re-derive at a rebind.

Hermetic: a controller shell built with ``__new__`` carries exactly the
attributes the property and the brake read; a tree shell carries the
symmetric predicate. Nothing allocates.

    CUDA_VISIBLE_DEVICES='' python -m pytest \\
        test/registered/unit/mem_cache/test_prefetch_limit_property_1068.py -q
"""

import inspect
import types
import unittest

from sglang.srt.managers.cache_controller import HiCacheController
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

#: the two host pools of the spec (section 5): PP0 retention-scale and the
#: --hicache-size 6 pool both phases are row-coupled to.
PP_ROWS = 923497
TP_ROWS = 366211


def _pool(size: int, available=None):
    avail = size if available is None else available
    return types.SimpleNamespace(size=size, available_size=lambda: avail)


def _controller(role: str = "staging", pool=None):
    cc = HiCacheController.__new__(HiCacheController)
    cc.host_role = role
    cc.mem_pool_host = pool
    cc.host_write_staged_tokens_fn = None
    return cc


class TestTheLimitFollowsTheBoundPool(CustomTestCase):
    """T6: the limit is derived from whatever pool is bound NOW."""

    def test_the_limit_follows_the_bound_pool(self):
        cc = _controller("staging", _pool(PP_ROWS))
        self.assertEqual(cc.prefetch_capacity_limit, int(0.9 * PP_ROWS))
        self.assertEqual(cc.prefetch_capacity_limit, 831147)
        # A rebind swaps the pool; no call re-derives anything.
        cc.mem_pool_host = _pool(TP_ROWS)
        self.assertEqual(cc.prefetch_capacity_limit, 329589)

    def test_retention_keeps_the_cache_mode_half(self):
        cc = _controller("retention", _pool(TP_ROWS))
        self.assertEqual(cc.prefetch_capacity_limit, int(0.5 * TP_ROWS))

    def test_the_limit_is_a_property_not_a_stored_number(self):
        """A stored number is exactly the defect: it survives the rebind
        that invalidates it. The property has no setter."""
        cc = _controller("staging", _pool(TP_ROWS))
        with self.assertRaises(AttributeError):
            cc.prefetch_capacity_limit = 5

    def test_no_host_pool_means_no_budget(self):
        cc = _controller("staging", None)
        self.assertEqual(cc.prefetch_capacity_limit, 0)


class TestRateLimitReadsLiveOccupancy(CustomTestCase):
    """T7: the brake reads the pool's live occupancy minus what the
    write-through ring has staged, never the fork's counter."""

    def test_rate_limit_reads_live_occupancy_minus_write_staged(self):
        cc = _controller("staging", _pool(TP_ROWS, available=10000))
        cc.host_write_staged_tokens_fn = lambda: 5000
        # The old gate's counter, deliberately at zero: if the brake still
        # read it, it could never trip below.
        cc.prefetch_tokens_occupied = 0
        # used = 366211 - 10000 - 5000 = 351211 >= 329589
        self.assertTrue(cc.prefetch_rate_limited())
        cc.mem_pool_host = _pool(TP_ROWS, available=200000)
        # used = 366211 - 200000 - 5000 = 161211 < 329589
        self.assertFalse(cc.prefetch_rate_limited())
        # The pair that DISCRIMINATES the subtraction: raw occupancy sits
        # 1622 rows above the limit, the ring holds 5000 of them.
        cc.mem_pool_host = _pool(TP_ROWS, available=35000)
        # used = 366211 - 35000 - 5000 = 326211 < 329589 -> not limited
        self.assertFalse(cc.prefetch_rate_limited())
        cc.host_write_staged_tokens_fn = None
        # used = 366211 - 35000 = 331211 >= 329589 -> limited
        self.assertTrue(cc.prefetch_rate_limited())

    def test_no_ring_means_nothing_is_subtracted(self):
        cc = _controller("staging", _pool(TP_ROWS, available=40000))
        cc.prefetch_tokens_occupied = 0
        # used = 326211 < 329589
        self.assertFalse(cc.prefetch_rate_limited())
        cc.mem_pool_host = _pool(TP_ROWS, available=36000)
        # used = 330211 >= 329589
        self.assertTrue(cc.prefetch_rate_limited())

    def test_the_counter_alone_cannot_trip_the_brake(self):
        """The fork counter stays an instrument; it no longer gates."""
        cc = _controller("staging", _pool(TP_ROWS, available=TP_ROWS))
        cc.prefetch_tokens_occupied = 10 * TP_ROWS
        self.assertFalse(cc.prefetch_rate_limited())


class TestRatioSizingUnderSymmetricStorageIsRefused(CustomTestCase):
    """T8 (G8): the symmetrize twins are gone; where they were needed --
    ratio-sized pools under uneven DCP with tp_world_size > 1 -- the boot is
    refused by name instead, because a property over rank-divergent pool
    sizes is a rank-divergent gate (the #580 desync)."""

    def _tree(self, cls):
        tree = cls.__new__(cls)
        tree._hicache_prefetch_symmetric = lambda: True
        return tree

    def test_ratio_sizing_under_symmetric_storage_is_refused(self):
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        tree = self._tree(UnifiedRadixCache)
        sa = types.SimpleNamespace(hicache_size=0)
        with self.assertRaises(ValueError) as cm:
            tree._refuse_ratio_sizing_under_symmetric_storage(sa)
        self.assertIn("#1068", str(cm.exception))
        self.assertIn("--hicache-size", str(cm.exception))
        sa.hicache_size = 6
        tree._refuse_ratio_sizing_under_symmetric_storage(sa)
        tree._hicache_prefetch_symmetric = lambda: False
        sa.hicache_size = 0
        tree._refuse_ratio_sizing_under_symmetric_storage(sa)

    def test_the_hiradix_twin_shares_the_refusal(self):
        from sglang.srt.mem_cache.hiradix_cache import HiRadixCache

        tree = self._tree(HiRadixCache)
        with self.assertRaises(ValueError):
            tree._refuse_ratio_sizing_under_symmetric_storage(
                types.SimpleNamespace(hicache_size=0)
            )
        tree._refuse_ratio_sizing_under_symmetric_storage(
            types.SimpleNamespace(hicache_size=6)
        )

    def test_init_refuses_before_the_ring_and_never_symmetrizes(self):
        from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        for src in (
            inspect.getsource(UnifiedRadixCache.init_hicache),
            inspect.getsource(HiRadixCache.__init__),
        ):
            self.assertIn("_refuse_ratio_sizing_under_symmetric_storage(", src)
            self.assertNotIn("_symmetrize_prefetch_capacity", src)
            self.assertLess(
                src.index("_refuse_ratio_sizing_under_symmetric_storage("),
                src.index("rebuild_staging_write_ring("),
            )

    def test_no_symmetrize_twin_survives(self):
        from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        self.assertFalse(hasattr(UnifiedRadixCache, "_symmetrize_prefetch_capacity"))
        self.assertFalse(hasattr(HiRadixCache, "_symmetrize_prefetch_capacity"))


if __name__ == "__main__":
    unittest.main()
