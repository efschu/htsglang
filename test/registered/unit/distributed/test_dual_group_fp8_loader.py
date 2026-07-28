# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""#274: the FP8 parameter-loader sighting test in lane scope.

Slice B validated the lane on the GGUF path only and said so. The named
handover was "does the FP8 loader path see the lane's vector, or does it hit
the tp_size=None trap". This file answers that on CPU, and it answers a
SECOND question the GGUF path never asked, because GGUF has no separate
scale axis:

    a block-quantized FP8 weight carries a SECOND sharded array whose length
    is ceil(out / block_n), not out.

Both answers came out BETTER than predicted, and the tests state what was
measured rather than what was feared:

* the tp_size=None trap is LOUD on this path, not silent. The v2 parameter
  carries a shard size computed elsewhere, so a contradiction between it and
  the installed vector is caught by name ("uneven-TP shard mismatch") --
  unlike the v1 linear.py path, where the layer is constructed even-split
  and therefore agrees with its own wrong offset.
* the weight/scale axis pair cannot mis-load silently either: both axes are
  partitioned with the same unit count, and a scale axis that is not a
  multiple of it is REFUSED rather than guessed.

So the FP8 constraint is a planning rule, not a runtime hazard: the lane
ratio must keep every rank's output partition a whole number of
quantization blocks. That is checkable before the boot (last test).

CPU only: none of this needs an fp8 kernel. The parameter classes below are
literally the ones fp8.py::create_weights registers, and the shard
arithmetic is dtype-blind.
"""

from __future__ import annotations

import unittest

import torch

from sglang.srt.distributed.dual_group import derive_nested_plan
from sglang.srt.distributed.utils import (
    partition_sizes,
    scoped_tp_partition_ratios,
)
from sglang.srt.layers.parameter import (
    BlockQuantScaleParameter,
    ChannelQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)

BIG = [2, 1, 1]
FAST = tuple(derive_nested_plan(BIG).fast_ratio)  # (2, 2)


def _column_param(cls, rows, cols, tp_units, **kw):
    p = cls(
        data=torch.zeros(rows, cols, dtype=torch.float32),
        input_dim=1,
        output_dim=0,
        weight_loader=None,
        **kw,
    )
    p.tp_units = tp_units
    return p


class TestFp8WeightLoaderSighting(unittest.TestCase):
    """The ModelWeightParameter fp8.py registers for the weight itself."""

    def test_weight_sees_the_lane_split_in_scope(self):
        out, in_size, units = 96, 32, 6
        full = torch.randn(out, in_size)
        sizes = partition_sizes(out, list(FAST), units)
        self.assertEqual(sizes, [48, 48])  # (2,2) over 6 units of 16
        with scoped_tp_partition_ratios(list(FAST)):
            p = _column_param(ModelWeightParameter, sizes[1], in_size, units)
            p.load_column_parallel_weight(full, tp_rank=1)
        torch.testing.assert_close(p.data, full[48:96])

    def test_asymmetric_lane_split_is_a_prefix_sum_not_an_even_split(self):
        # The case that distinguishes the two: FAST = (4, 2) over 6 units.
        big = [4, 1, 1]
        fast = list(derive_nested_plan(big).fast_ratio)
        out, in_size, units = 96, 32, 6
        full = torch.randn(out, in_size)
        sizes = partition_sizes(out, fast, units)
        self.assertEqual(sizes, [64, 32])
        with scoped_tp_partition_ratios(fast):
            p = _column_param(ModelWeightParameter, sizes[1], in_size, units)
            p.load_column_parallel_weight(full, tp_rank=1)
        # Prefix-sum offset 64, not the even-split offset 48.
        torch.testing.assert_close(p.data, full[64:96])

    def test_without_the_scope_the_fp8_path_raises_instead_of_mis_sharding(self):
        """Measured, and better than the GGUF path's behavior.

        On the v1 (linear.py) loader the same mistake is silent: the layer
        is CONSTRUCTED even-split, so parameter and offset agree with each
        other and disagree with the intent. The v2 parameter carries a shard
        size that was computed elsewhere, so when the installed vector says
        one thing and the parameter says another, tp_loaded_shard_start
        catches the contradiction and names it. For the FP8 arm that means
        the tp_size=None trap is a LOUD failure, not a silent one.
        """
        big = [4, 1, 1]
        out, in_size, units = 96, 32, 6
        full = torch.randn(out, in_size)
        with scoped_tp_partition_ratios(big):
            p = _column_param(ModelWeightParameter, 32, in_size, units)
            with self.assertRaises(ValueError) as ctx:
                p.load_column_parallel_weight(full, tp_rank=1)
        self.assertIn("uneven-TP shard mismatch", str(ctx.exception))

    def test_row_axis_takes_the_same_route(self):
        big = [4, 1, 1]
        fast = list(derive_nested_plan(big).fast_ratio)
        out, in_size, units = 32, 96, 6
        full = torch.randn(out, in_size)
        with scoped_tp_partition_ratios(fast):
            p = _column_param(ModelWeightParameter, out, 32, units)
            p.load_row_parallel_weight(full, tp_rank=1)
        torch.testing.assert_close(p.data, full[:, 64:96])


class TestFp8ScaleAxes(unittest.TestCase):
    """The axes GGUF does not have."""

    def test_per_tensor_scale_is_not_sharded_at_all(self):
        # One scale per logical matrix. If this ever went through the
        # column path it would be sliced to nothing on a 2-rank lane.
        p = PerTensorScaleParameter(
            data=torch.zeros(1, dtype=torch.float32), weight_loader=None
        )
        self.assertFalse(hasattr(p, "output_dim"))
        p.load_merged_column_weight(
            loaded_weight=torch.tensor(0.25), shard_id=0
        )
        self.assertEqual(float(p.data[0]), 0.25)

    def test_channel_scale_follows_the_weight_partition(self):
        big = [4, 1, 1]
        fast = list(derive_nested_plan(big).fast_ratio)
        out, units = 96, 6
        full = torch.randn(out, 1)
        with scoped_tp_partition_ratios(fast):
            # Column-only parameter: no input_dim.
            p = ChannelQuantScaleParameter(
                data=torch.zeros(32, 1, dtype=torch.float32),
                output_dim=0,
                weight_loader=None,
            )
            p.tp_units = units
            p.load_column_parallel_weight(full, tp_rank=1)
        torch.testing.assert_close(p.data, full[64:96])

    def test_block_scale_agrees_with_the_weight_when_blocks_divide(self):
        """The condition under which block-quantized FP8 nests."""
        big, block_n, units = [4, 1, 1], 16, 6
        fast = list(derive_nested_plan(big).fast_ratio)
        out = 96  # 6 units of 16; block_n == the unit size, so blocks divide
        w_sizes = partition_sizes(out, fast, units)
        s_sizes = partition_sizes(out // block_n, fast, units)
        self.assertEqual(w_sizes, [64, 32])
        self.assertEqual(s_sizes, [4, 2])
        # Every rank's weight rows are a whole number of blocks, and the
        # scale partition is exactly the weight partition divided by block_n.
        self.assertEqual([s * block_n for s in s_sizes], w_sizes)

        full_scale = torch.randn(out // block_n, 1)
        with scoped_tp_partition_ratios(fast):
            p = _column_param(BlockQuantScaleParameter, s_sizes[1], 1, units)
            p.load_column_parallel_weight(full_scale, tp_rank=1)
        torch.testing.assert_close(p.data, full_scale[4:6])

    def test_block_scale_axis_REFUSES_a_unit_count_it_cannot_carry(self):
        """The FP8 arm's real constraint, measured.

        The block-scale axis has ceil(out / block_n) entries while the weight
        axis has out. Both are partitioned with the SAME unit count, because
        the unit count travels on the parameter. When the scale axis is not
        a multiple of that unit count the split is undefined, and
        partition_sizes says so instead of guessing -- so a block-quantized
        FP8 lane cannot silently load mismatched scales; it fails at load.

        For the two-card FP8 lane this turns into a PLANNING rule rather
        than a runtime hazard: the lane ratio has to keep every rank's
        output partition a whole number of quantization blocks.
        """
        block_n = 128
        # 10 units of 64 rows = 640 rows, but only 5 blocks of 128.
        units, unit_rows = 10, 64
        out = units * unit_rows
        with self.assertRaises(ValueError) as ctx:
            partition_sizes(out // block_n, [7, 3], units)
        self.assertIn("not a multiple of its unit count", str(ctx.exception))
        # The weight axis itself is fine -- only the pair is not.
        self.assertEqual(partition_sizes(out, [7, 3], units), [448, 192])

    def test_the_condition_is_checkable_up_front(self):
        """State it as a predicate, so the FP8 arm can gate on it."""

        def blocks_divide(out, ratio, units, block_n):
            return all(
                s % block_n == 0 for s in partition_sizes(out, ratio, units)
            )

        self.assertTrue(blocks_divide(96, list(FAST), 6, 16))
        self.assertFalse(blocks_divide(640, [7, 3], 10, 128))


if __name__ == "__main__":
    unittest.main()
