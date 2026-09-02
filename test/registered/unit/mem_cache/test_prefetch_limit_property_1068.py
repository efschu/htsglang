"""#1068 WEG 1 slice 2 (WEG1_BUILD_SPEC_0901 section 4.2, corrected by the
slice 2 review): the speculative prefetch budget is a LIVE property of the
bound host pool (upstream buffer_only fraction for a staging tier, the
cache-mode half for a retention tier), and the rate brake is the upstream
CACHE-MODE form ``prefetch_tokens_occupied >= prefetch_capacity_limit`` in
BOTH roles.

THE DEFECT (slice 2 as shipped). ``prefetch_capacity_limit`` was a NUMBER
stored once at storage attach (``prefetch_capacity_limit_for``), and the
phase flip rebinds ``mem_pool_host`` to a pool 23x smaller (measured #905:
703472 rows PP vs 30518 TP), so the stored number described a pool the
controller no longer held; the fork answered with a floor and two
symmetrize passes -- second bookkeeping beside the upstream truth. Slice 2
made the limit a property (kept here, T6) but transcribed the brake as the
upstream ``buffer_only`` LIVE form, ``size - available_size() - staged >=
limit``, unconditionally. Upstream dispatches that form on
``host_memory_mode == "buffer_only"`` only, where the host pool is a
transient buffer whose rows are freed once their storage write is acked.
This fork's host tier RETAINS in both roles: ``_drain_backup``
(unified_radix_cache.py) answers a storage ack with ``ring.release`` +
``dec_host_lock_ref`` and never frees the rows, ``evict_host`` has no role
branch, and ``available_size()`` is ``len(free_slots)``. On every warm tier
the live form read ``used == size``, tripped permanently at 0.5 (retention,
the default) or 0.9 (staging) of the pool, and refused every prefetch
BEFORE ``prefetch_from_storage`` reached its alloc -> evict_host -> retry
path -- store-read path B dead, every re-admission recomputing the whole
prefix: the double-prefill class WEG 1 exists to remove. Three mutants
survived the shipped suite (retention -> counter form; ``>=`` -> ``>``;
attach not copying ``host_role``). T7 and T7b below pin all three.

Hermetic: a controller shell built with ``__new__`` carries exactly the
attributes the property and the brake read; a tree shell carries the
symmetric predicate. Nothing allocates.

    CUDA_VISIBLE_DEVICES='' python -m pytest \\
        test/registered/unit/mem_cache/test_prefetch_limit_property_1068.py -q
"""

import ast
import inspect
import textwrap
import threading
import types
import unittest
from unittest import mock

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


class TestRateLimitIsTheCounterForm(CustomTestCase):
    """T7 (slice 2 fix, review finding B1): the brake is the upstream
    CACHE-MODE form in BOTH roles -- ``prefetch_tokens_occupied >= limit``
    (upstream cache_controller.py:1164-1166) -- because this fork's host
    tier retains rows in both roles, so the buffer_only live form would
    read ``used == size`` on every warm tier and refuse every prefetch
    before ``prefetch_from_storage`` reaches alloc -> evict_host -> retry."""

    def test_a_full_idle_pool_is_not_rate_limited(self):
        """Red-first (1): warm tier, nothing registered. available_size()=0,
        counter 0 -> NOT limited. The live form read
        used = 366211 >= 329589 here and parked store-read path B."""
        cc = _controller("staging", _pool(TP_ROWS, available=0))
        cc.prefetch_tokens_occupied = 0
        self.assertFalse(cc.prefetch_rate_limited())
        cc.host_role = "retention"
        self.assertFalse(cc.prefetch_rate_limited())

    def test_the_counter_gates_in_both_roles(self):
        """Red-first (2), kills mutant M-R1 (retention -> counter form,
        staging -> live form): the counter is the gate in retention AND
        staging, whatever the pool's live occupancy says."""
        for role, limit in (
            ("retention", int(0.5 * TP_ROWS)),
            ("staging", int(0.9 * TP_ROWS)),
        ):
            with self.subTest(role=role):
                cc = _controller(role, _pool(TP_ROWS, available=TP_ROWS))
                cc.prefetch_tokens_occupied = limit + 1
                self.assertTrue(cc.prefetch_rate_limited())
                cc.prefetch_tokens_occupied = limit - 1
                self.assertFalse(cc.prefetch_rate_limited())

    def test_the_boundary_is_inclusive(self):
        """Red-first (3), kills mutant M-R2: occupied == limit -> limited
        (``>=``, upstream :1165), never ``>``."""
        cc = _controller("staging", _pool(TP_ROWS, available=TP_ROWS))
        self.assertEqual(cc.prefetch_capacity_limit, 329589)
        cc.prefetch_tokens_occupied = 329589
        self.assertTrue(cc.prefetch_rate_limited())
        cc.prefetch_tokens_occupied = 329588
        self.assertFalse(cc.prefetch_rate_limited())

    def test_no_staged_tokens_reader_survives(self):
        """Zombie guard (slice 2 fix 2, Upstream-Minimal law): the ring
        occupancy reader of upstream :331 that slice 2 kept installed on the
        controller was read by nobody (the brake is the counter form), so it
        is gone from the controller's __init__ and from the ring module. It
        returns only together with the staging drain that frees host rows
        after ack_backup, i.e. with a consumer and its own tests."""
        from sglang.srt.mem_cache import staging_write_ring

        for src in (
            inspect.getsource(HiCacheController.__init__),
            inspect.getsource(staging_write_ring),
        ):
            self.assertNotIn("host_write_staged_tokens_fn", src)

    def test_the_brake_source_carries_no_live_form(self):
        """Zombie guard: the buffer_only live form returns only WITH a role
        dispatch AND a drain that frees rows after the ack -- never as an
        unconditional transcription again."""
        src = inspect.getsource(HiCacheController.prefetch_rate_limited)
        fn = ast.parse(textwrap.dedent(src)).body[0]
        # The docstring names the live form on purpose (it explains why it
        # is absent); only the CODE body is under test.
        body = [
            node
            for node in fn.body
            if not (
                isinstance(node, ast.Expr)
                and isinstance(getattr(node, "value", None), ast.Constant)
            )
        ]
        code = "\n".join(ast.unparse(node) for node in body)
        self.assertNotIn("available_size()", code)
        self.assertNotIn("host_write_staged_tokens_fn", code)
        self.assertIn(
            "self.prefetch_tokens_occupied >= self.prefetch_capacity_limit", code
        )


class TestAttachCopiesTheHostRole(CustomTestCase):
    """T7b, kills mutant M-R4: ``attach_storage_backend`` copies the storage
    config's ``host_role`` onto the controller -- the one reader on the
    controller; build_staging_write_ring and phase_flip_boot read the flag
    too -- and the budget fraction follows it. Without the
    copy a staging boot silently runs the 0.5 retention fraction (its ring
    sized to 0.5 x size instead of 0.1 x size), visible only in the L3
    ``role=`` term. Green on the shipped tree by design; red under M-R4."""

    def _shell(self):
        cc = HiCacheController.__new__(HiCacheController)
        cc.enable_storage = False
        cc.host_role = "retention"
        cc.page_size = 1
        cc.mem_pool_host = _pool(TP_ROWS)
        cc.storage_stop_event = threading.Event()
        for name in (
            "_stop_storage_threads",
            "_start_storage_threads",
            "_create_prefetch_sync_groups",
            "_destroy_prefetch_sync_groups",
            "_maybe_register_draft_with_storage",
        ):
            setattr(cc, name, lambda: None)
        cc._generate_storage_config = lambda *a, **k: types.SimpleNamespace(
            host_role="staging",
            dcp_owner_mode=False,
            canonical_kv_page=None,
            is_mla_model=False,
            tp_rank=0,
            extra_config={},
        )
        return cc

    def test_attach_copies_host_role_and_the_fraction_follows(self):
        cc = self._shell()
        backend = types.SimpleNamespace(
            register_mem_pool_host=lambda pool: None, close=lambda: None
        )
        with mock.patch(
            "sglang.srt.mem_cache.storage.StorageBackendFactory.create_backend",
            return_value=backend,
        ):
            cc.attach_storage_backend("file")
        self.assertTrue(cc.enable_storage)
        self.assertEqual(cc.host_role, "staging")
        self.assertEqual(cc.prefetch_capacity_fraction, 0.9)
        self.assertEqual(cc.prefetch_capacity_limit, 329589)
        self.assertEqual(cc.prefetch_tokens_occupied, 0)


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
