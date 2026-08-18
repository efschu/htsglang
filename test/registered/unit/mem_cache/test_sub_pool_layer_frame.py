"""Sub-pools are indexed in their OWN layer frame, not the global PP frame.

`KVCache.__init__` attaches an ownership map keyed by GLOBAL layer ids. That is
right for a pool the model addresses with global ids, and wrong for a SUB-pool
that a wrapper addresses with re-indexed ids.

`HybridLinearKVPool` (the GDN + full-attention pool the family plan uses) maps a
global id through `_transfer_full_attention_id` into a DENSE full-attention
index, then calls `full_kv_pool.get_key_buffer(mapped)`. Under
`SGLANG_PP_LAYER_SET` the sub-pool's map is keyed {35: 0, 39: 1, ...} while the
id arriving is 0..7 -- so every lookup misses.

It fails loudly rather than silently (the accessor refuses an unowned layer
instead of answering), but it fails. The fix is to say so at construction: a
sub-pool carries no global ownership map, so its accessor degenerates to the
plain subtraction in its own dense frame. This is a defect introduced with the
accessor itself, not a pre-existing one.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import inspect
import unittest

from sglang.srt.mem_cache.memory_pool import KVCache, mark_as_sub_pool
from sglang.test.test_utils import CustomTestCase

FA_STAGE = [35, 39, 43, 47, 51, 55, 59, 63]


class _Pool:
    local_slot = KVCache.local_slot

    def __init__(self, start_layer=0, owned=None):
        self.start_layer = start_layer
        self._local_slot_of = (
            None
            if owned is None
            else {layer: slot for slot, layer in enumerate(sorted(owned))}
        )


class TestTheGlobalMapIsTheProblem(CustomTestCase):
    def test_a_dense_sub_pool_id_misses_a_global_map(self):
        """Characterises the defect: the wrapper hands over 0..7, the map holds
        35..63, so the lookup refuses."""
        pool = _Pool(start_layer=35, owned=FA_STAGE)
        with self.assertRaises(KeyError):
            pool.local_slot(0)


class TestMarkingASubPoolFixesTheFrame(CustomTestCase):
    def test_it_drops_the_global_map(self):
        pool = _Pool(start_layer=35, owned=FA_STAGE)
        mark_as_sub_pool(pool)
        self.assertIsNone(pool._local_slot_of)

    def test_the_dense_ids_then_resolve(self):
        pool = _Pool(start_layer=0, owned=FA_STAGE)
        mark_as_sub_pool(pool)
        for i in range(8):
            with self.subTest(sub_id=i):
                self.assertEqual(pool.local_slot(i), i)

    def test_it_is_a_no_op_on_the_contiguous_path(self):
        """Nothing to drop when no set is configured -- the default path must
        not notice this at all."""
        pool = _Pool(start_layer=12)
        mark_as_sub_pool(pool)
        self.assertIsNone(pool._local_slot_of)
        self.assertEqual(pool.local_slot(15), 3)

    def test_it_tolerates_a_pool_that_never_ran_the_base_init(self):
        """SWAKVPool does not call super().__init__, so the attribute may be
        absent entirely."""

        class _Bare:
            start_layer = 0

        bare = _Bare()
        mark_as_sub_pool(bare)
        self.assertIsNone(bare._local_slot_of)


class TestTheWrappersMarkTheirSubPools(CustomTestCase):
    def test_hybrid_linear_pool_marks_its_full_pool(self):
        from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

        src = inspect.getsource(HybridLinearKVPool.__init__)
        self.assertIn("mark_as_sub_pool(", src)

    def test_swa_pool_marks_both_sub_pools(self):
        from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

        src = inspect.getsource(SWAKVPool.__init__)
        self.assertEqual(src.count("mark_as_sub_pool("), 2)


if __name__ == "__main__":
    unittest.main()
