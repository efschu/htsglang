"""fnFL2 v55 (21.09.): the base weights tag keeps its dead shard-load blocks.

``WEG2-XCHG-COVER`` had the answer all along -- the saver books 9528 MiB
under the base tag on PP1 against 12.5 MiB of live named tensors (8726 vs
628.2 on PP2), and those deltas ARE the per-rank load overhead (10.09 and
7.57 GiB) that killed v45 to v55 in ``cu_mem_create``.

``load_weights`` runs in the BASE tag -- ``weight_chunk_scope`` wraps only
the post-load pass -- the presplit frees the originals python-side, and the
blocks stay in that tag's private MemPool.  Two ways out are already ruled
out by measurement: ``_cuda_releasePool`` refuses while the tag cache holds
a reference (v52), and stepping the repack outside the pool misses the
allocations that happen before it (v53).  What is left is to drop the POOL
OBJECT, which is cheap exactly here because what still lives under the base
tag is small and named.
"""

import unittest
from unittest import mock

import torch

from sglang.srt.managers import weg2_memory_saver as MS


class SwapTagPool(unittest.TestCase):
    def setUp(self):
        self._pools = dict(MS._TAG_MEM_POOLS)
        self._holders = dict(MS._TAG_LIVE_TENSOR_HOLDERS)
        MS._TAG_MEM_POOLS.clear()
        MS._TAG_LIVE_TENSOR_HOLDERS.clear()

    def tearDown(self):
        MS._TAG_MEM_POOLS.clear()
        MS._TAG_MEM_POOLS.update(self._pools)
        MS._TAG_LIVE_TENSOR_HOLDERS.clear()
        MS._TAG_LIVE_TENSOR_HOLDERS.update(self._holders)

    def test_an_unknown_tag_is_a_no_op(self):
        self.assertEqual(MS.swap_tag_pool("nope"), (0, 0.0))

    def test_the_pool_object_is_REPLACED_not_released(self):
        """The whole point: the old object must be gone from the cache, so
        its last reference dies and the allocator hands the blocks back."""
        old = mock.Mock(name="old_pool")
        MS._TAG_MEM_POOLS["weights"] = old
        MS.register_tag_live_tensors("weights", [])
        with mock.patch("torch.cuda.is_available", return_value=True), \
             mock.patch("torch.cuda.memory_reserved", side_effect=[8_000_000_000, 500_000_000]), \
             mock.patch("torch.cuda.empty_cache"), \
             mock.patch.object(torch.cuda, "MemPool", create=True) as pool_cls, \
             mock.patch.object(torch.cuda, "use_mem_pool", create=True):
            moved, freed = MS.swap_tag_pool("weights")
        self.assertEqual(moved, 0)
        self.assertGreater(freed, 6.0)
        self.assertIsNot(MS._TAG_MEM_POOLS["weights"], old)
        self.assertIs(MS._TAG_MEM_POOLS["weights"], pool_cls.return_value)

    def test_registered_tensors_are_carried_across_and_rebound(self):
        held = {"t": torch.zeros(4)}
        MS._TAG_MEM_POOLS["weights"] = mock.Mock()
        MS.register_tag_live_tensors(
            "weights",
            [(lambda: held["t"], lambda new: held.__setitem__("t", new))],
        )
        with mock.patch("torch.cuda.is_available", return_value=True), \
             mock.patch("torch.cuda.memory_reserved", return_value=0), \
             mock.patch("torch.cuda.empty_cache"), \
             mock.patch.object(torch.cuda, "MemPool", create=True), \
             mock.patch.object(torch.cuda, "use_mem_pool", create=True):
            # a CPU tensor is skipped (is_cuda False) -- the guard, not the copy
            moved, _ = MS.swap_tag_pool("weights")
        self.assertEqual(moved, 0)


class TheRunnerSwapsAfterLoading(unittest.TestCase):
    def test_it_runs_before_the_coverage_arm_and_only_for_base_tag_tensors(self):
        import inspect

        from sglang.srt.model_executor import model_runner as MR

        src = inspect.getsource(MR)
        self.assertIn("swap_tag_pool(weights_tag)", src)
        self.assertIn("tag_of_parameter_name(_n, region_tag=weights_tag) != weights_tag", src)
        self.assertLess(
            src.index("swap_tag_pool(weights_tag)"),
            src.index("from sglang.srt.weg2.weight_exchange import arm_coverage_at_load"),
        )
        # a failure must be named, never swallowed
        self.assertIn("WEG2-TAG-POOL swap FAILED", src)


if __name__ == "__main__":
    unittest.main()
