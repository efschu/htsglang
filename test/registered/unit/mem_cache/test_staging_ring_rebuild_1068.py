"""#1068 WEG 1 slice 2 (WEG1_BUILD_SPEC_0901 section 4.2, graft G2): the
write-through staging ring is rebuilt against the pool the readers are bound
to; its capacity is the complement of that pool's prefetch budget.

THE DEFECT. ``build_staging_write_ring`` had exactly one caller
(``init_hicache``), and ``_reset_full`` nulled the ring at every cutover.
After the first flip a staging boot ran with NO ring at all -- write-through
unbounded. The ring's capacity is the complement of
``prefetch_capacity_limit``, which is a property of the bound pool since
slice 2, so the ring must be rebuilt whenever the pool moves. (Slice 2 fix 2
removed the controller-side occupancy reader the ring used to install: the
brake is the cache-mode counter form and read it nowhere, so the pins that
described that wire are retired from this file.)

Hermetic: tree and controller shells via ``__new__``; the real
``build_staging_write_ring`` and the real ``StagingWriteRing`` run. Nothing
allocates.

    CUDA_VISIBLE_DEVICES='' python -m pytest \\
        test/registered/unit/mem_cache/test_staging_ring_rebuild_1068.py -q
"""

import inspect
import types
import unittest

from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache import hicache_phase_binding as hpb
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

PP_ROWS = 923497
TP_ROWS = 366211


def _pool(size: int):
    return types.SimpleNamespace(size=size, available_size=lambda: size)


def _tree(size: int, role: str = "staging"):
    tree = UnifiedRadixCache.__new__(UnifiedRadixCache)
    cc = HiCacheController.__new__(HiCacheController)
    cc.host_role = role
    cc.mem_pool_host = _pool(size)
    tree.cache_controller = cc
    tree.staging_write_ring = object()
    return tree, cc


class TestTheRingIsRebuiltAtRebind(CustomTestCase):
    def test_reset_drops_the_ring(self):
        tree, _cc = _tree(PP_ROWS)
        tree._drop_staging_write_ring()
        self.assertIsNone(tree.staging_write_ring)
        # ONE place nulls it: _reset_full goes through it.
        self.assertIn(
            "_drop_staging_write_ring()", inspect.getsource(UnifiedRadixCache._reset_full)
        )

    def test_rebuild_binds_capacity_to_the_current_pool(self):
        tree, cc = _tree(TP_ROWS)
        sa = types.SimpleNamespace(hicache_host_role="staging")
        tree.rebuild_staging_write_ring(sa)
        ring = tree.staging_write_ring
        self.assertIsNotNone(ring)
        self.assertEqual(ring.capacity_tokens, TP_ROWS - cc.prefetch_capacity_limit)
        self.assertEqual(ring.capacity_tokens, TP_ROWS - 329589)
        # A real ring, empty at build.
        self.assertEqual(ring.occupied_tokens, 0)
        self.assertTrue(ring.admit("page-a", 4096))
        self.assertEqual(ring.occupied_tokens, 4096)

    def test_rebuild_after_a_pool_swap_follows_the_new_pool(self):
        tree, cc = _tree(PP_ROWS)
        sa = types.SimpleNamespace(hicache_host_role="staging")
        tree.rebuild_staging_write_ring(sa)
        first = tree.staging_write_ring
        self.assertEqual(first.capacity_tokens, PP_ROWS - 831147)
        first.admit("page-a", 4096)
        cc.mem_pool_host = _pool(TP_ROWS)
        tree.rebuild_staging_write_ring(sa)
        second = tree.staging_write_ring
        self.assertIsNot(second, first)
        self.assertEqual(second.capacity_tokens, TP_ROWS - 329589)
        # The NEW ring starts empty; the old one's admission does not carry.
        self.assertEqual(second.occupied_tokens, 0)

    def test_retention_role_builds_nothing(self):
        tree, _cc = _tree(PP_ROWS, role="retention")
        tree.rebuild_staging_write_ring(
            types.SimpleNamespace(hicache_host_role="retention")
        )
        self.assertIsNone(tree.staging_write_ring)

    def test_the_cutover_rebind_rebuilds_the_ring_after_the_coherence_check(self):
        tree, cc = _tree(TP_ROWS)
        scheduler = types.SimpleNamespace(
            server_args=types.SimpleNamespace(hicache_host_role="staging")
        )
        readers = {"scheduler": scheduler, "tree_cache": tree, "cache_controller": cc}
        hpb.rebuild_staging_ring_after_rebind(readers, scheduler)
        self.assertIsNotNone(tree.staging_write_ring)
        self.assertEqual(tree.staging_write_ring.capacity_tokens, TP_ROWS - 329589)
        src = inspect.getsource(hpb._rebind_for_cutover_inner)
        self.assertIn("rebuild_staging_ring_after_rebind(", src)
        self.assertLess(
            src.index("coherence_check(readers)"),
            src.index("rebuild_staging_ring_after_rebind("),
        )

    def test_a_tree_without_the_hook_is_skipped_not_raised(self):
        """A plain RadixCache reader has no ring; the cutover must not die
        on it."""
        scheduler = types.SimpleNamespace(server_args=types.SimpleNamespace())
        readers = {"scheduler": scheduler, "tree_cache": object(), "cache_controller": None}
        hpb.rebuild_staging_ring_after_rebind(readers, scheduler)

    def test_l3_is_emitted_even_when_the_ring_rebuild_fails(self):
        """Slice 2 fix 3 (review round 3, non-blocking note): a failed ring
        rebuild after a committed rebind still emits the '#915 PREFETCH
        LIMIT' line, so the acceptance can count exactly one budget line per
        cutover. Before: the error path returned before L3."""
        tree, cc = _tree(TP_ROWS)

        def failing_rebuild(server_args):
            raise RuntimeError("ring refused")

        tree.rebuild_staging_write_ring = failing_rebuild
        scheduler = types.SimpleNamespace(
            server_args=types.SimpleNamespace(hicache_host_role="staging")
        )
        readers = {"scheduler": scheduler, "tree_cache": tree, "cache_controller": cc}
        from sglang.srt.mem_cache import prefetch_budget

        with self.assertLogs(prefetch_budget.logger, level="INFO") as logs:
            hpb.rebuild_staging_ring_after_rebind(readers, scheduler)
        self.assertEqual(
            len([m for m in logs.output if "#915 PREFETCH LIMIT" in m]), 1, logs.output
        )


    def test_a_failed_rebuild_after_the_committed_rebind_is_counted_not_raised(self):
        """Slice 4 fix (spec A12.5, the log-vs-raise decision): a rank whose
        ring could not be rebuilt after the COMMITTED rebind keeps running --
        no raise (the seam's `rebind_for_cutover` handler would only report
        the committed rebind as refused, the #861c form) -- and the failure
        is COUNTED under the #915 key `staging_ring_rebuild_failed` (a
        cutover-level key, not part of the intake partition) and spoken as
        ONE ERROR line naming the term, the phase and the generation. RED on
        af399f19c1: no key, and the line named no term."""
        from sglang.srt.mem_cache.match_refusal_census import (
            PREFETCH_CUTOVER_KEYS,
            PREFETCH_GATE_COUNTS,
            PREFETCH_INTAKE_PARTITION,
        )

        PREFETCH_GATE_COUNTS.clear()
        try:
            tree, cc = _tree(TP_ROWS)

            def failing_rebuild(server_args):
                raise RuntimeError("ring refused")

            tree.rebuild_staging_write_ring = failing_rebuild
            scheduler = types.SimpleNamespace(
                server_args=types.SimpleNamespace(hicache_host_role="staging")
            )
            readers = {"scheduler": scheduler, "tree_cache": tree, "cache_controller": cc}
            with self.assertLogs(hpb.logger, level="ERROR") as logs:
                hpb.rebuild_staging_ring_after_rebind(readers, scheduler)
            lines = [m for m in logs.output if "STAGING RING REBUILD FAILED" in m]
            self.assertEqual(len(lines), 1, logs.output)
            for term in (
                "term=staging_ring_rebuild_failed",
                "COMMITTED",
                "ring refused",
                "phase=",
                "generation=",
                "n=1",
            ):
                self.assertIn(term, lines[0])
            self.assertEqual(PREFETCH_GATE_COUNTS.get("staging_ring_rebuild_failed"), 1)
            self.assertIn("staging_ring_rebuild_failed", PREFETCH_CUTOVER_KEYS)
            self.assertNotIn("staging_ring_rebuild_failed", PREFETCH_INTAKE_PARTITION)
        finally:
            PREFETCH_GATE_COUNTS.clear()


if __name__ == "__main__":
    unittest.main()
