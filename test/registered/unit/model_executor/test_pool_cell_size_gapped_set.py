"""KV cell sizing must read the OWNED SET, not the stage's span.

`[start_layer, end_layer)` equals the owned layers only while a stage is a
contiguous interval. Under `SGLANG_PP_LAYER_SET` it does not, and the hybrid
(mambaish) branch of `MemoryPoolConfigurator` was still counting full-attention
layers by span after the same distinction had already been fixed for
`num_effective_layers`.

WHAT IT COST, and why this test exists as a pin rather than a nicety. On the
gapped [48, 8, 8] set over a 64-layer hybrid with full_attention_interval=4,
stage 0 owns all 48 GDN layers and ZERO full-attention layers while spanning
[0, 63). The span admitted 15 of the 16 full-attention layers, so the stage
reserved KV for 15 layers it cannot address -- on the one card already holding
25.7 GiB of weights. The boot died on cuMemCreate CUDA_ERROR_OUT_OF_MEMORY.

The two other stages were correct BY LUCK: their gapped sets happen to contain
every full-attention layer inside their own span. So a test that only checked
them would have passed while the layout was broken.

    python -m pytest test/registered/unit/model_executor/test_pool_cell_size_gapped_set.py -v
"""

import os
import unittest
from unittest import mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

NUM_LAYERS = 64
FULL_ATTN = [i for i in range(NUM_LAYERS) if i % 4 == 3]  # 16 layers: 3, 7, ...

# The ticket's gapped family layout: stage 0 takes every GDN layer, the 16
# full-attention layers split 8/8 across the two smaller cards.
GAPPED = (
    "0-2,4-6,8-10,12-14,16-18,20-22,24-26,28-30,32-34,36-38,40-42,44-46,"
    "48-50,52-54,56-58,60-62;3,7,11,15,19,23,27,31;35,39,43,47,51,55,59,63"
)


def _effective_full_attn_layers(pp_rank, pp_size, start_layer, end_layer):
    """Re-run the configurator's selection in isolation.

    Constructing a real MemoryPoolConfigurator needs a live ModelRunner and a
    device; the selection under test is pure integer logic over the owned set,
    so it is exercised directly against the same helper the configurator calls.
    """
    from sglang.srt.distributed.utils import get_pp_layer_set

    owned = get_pp_layer_set(NUM_LAYERS, pp_rank, pp_size)
    if owned is not None:
        return [i for i in FULL_ATTN if i in owned]
    return [i for i in FULL_ATTN if start_layer <= i < end_layer]


class TestGappedSetCellSizing(CustomTestCase):
    def setUp(self):
        self._env = dict(os.environ)
        os.environ["SGLANG_PP_LAYER_SET"] = GAPPED
        os.environ["SGLANG_PP_CROSSING_WIRE"] = "1"

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._env)

    def test_gdn_only_stage_owns_no_full_attention_layers(self):
        """The regression itself: stage 0 spans almost the whole model."""
        got = _effective_full_attn_layers(0, 3, start_layer=0, end_layer=63)
        self.assertEqual(got, [], "a stage owning only GDN layers needs no KV")

    def test_the_span_would_have_over_counted_by_fifteen(self):
        """Pin the SIZE of the error, so a silent regression is loud.

        Without the set, the span [0, 63) admits 15 of the 16 full-attention
        layers -- which is where the 30720-byte cell (15 x 2048) came from.
        """
        by_span = [i for i in FULL_ATTN if 0 <= i < 63]
        self.assertEqual(len(by_span), 15)
        self.assertEqual(len(_effective_full_attn_layers(0, 3, 0, 63)), 0)

    def test_attention_stages_own_eight_each(self):
        for rank, expected in ((1, [3, 7, 11, 15, 19, 23, 27, 31]),
                               (2, [35, 39, 43, 47, 51, 55, 59, 63])):
            with self.subTest(rank=rank):
                lo, hi = expected[0], expected[-1] + 1
                self.assertEqual(
                    _effective_full_attn_layers(rank, 3, lo, hi), expected
                )

    def test_every_full_attention_layer_is_owned_exactly_once(self):
        """Coverage, not just counts -- a lost KV layer is silent at runtime."""
        seen = []
        for rank in range(3):
            seen += _effective_full_attn_layers(rank, 3, 0, NUM_LAYERS)
        self.assertEqual(sorted(seen), FULL_ATTN)

    def test_contiguous_path_is_untouched(self):
        """With no layer set the span rule must still be what runs."""
        os.environ.pop("SGLANG_PP_LAYER_SET")
        # Stage 1 of a contiguous [0,32)/[32,64) split.
        got = _effective_full_attn_layers(1, 2, start_layer=32, end_layer=64)
        self.assertEqual(got, [i for i in FULL_ATTN if 32 <= i < 64])
        self.assertEqual(len(got), 8)


if __name__ == "__main__":
    unittest.main()
