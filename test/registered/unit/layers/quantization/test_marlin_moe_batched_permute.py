"""The MoE zero-point / scale repack helpers run over the whole expert stack
at once instead of one expert at a time (Flash-Next load profile
2026-09-16: 313 experts x 2 tensors x 48 layers of numpy per-expert calls
on the 5090 rank). Pinned row-identical against the per-expert loop."""

import torch

from sglang.srt.layers.quantization.marlin_utils import (
    awq_to_marlin_zero_points,
    marlin_moe_permute_scales,
    marlin_permute_scales,
    moe_awq_to_marlin_zero_points,
)


def test_batched_zero_points_equal_the_per_expert_loop():
    g = torch.Generator().manual_seed(0)
    for num_bits, size_n in ((4, 128), (8, 128)):
        E, size_k, pf = 5, 4, 32 // num_bits
        packed = torch.randint(-(2**31), 2**31 - 1, (E, size_k, size_n // pf), generator=g, dtype=torch.int32)
        loop = torch.stack([awq_to_marlin_zero_points(packed[e], size_k, size_n, num_bits) for e in range(E)])
        batched = moe_awq_to_marlin_zero_points(packed, size_k, size_n, num_bits)
        assert batched.shape == loop.shape and torch.equal(batched, loop), num_bits
    assert moe_awq_to_marlin_zero_points(packed[:0], size_k, size_n, 4).shape[0] == 0


def test_batched_scales_equal_the_per_expert_loop():
    g = torch.Generator().manual_seed(1)
    E, size_n = 6, 256
    for size_k, group_size, groups in ((512, 32, 16), (256, -1, 1), (128, 128, 1)):
        s = torch.randn(E, groups, size_n, generator=g).half()
        loop = torch.stack([marlin_permute_scales(s[e], size_k, size_n, group_size) for e in range(E)])
        batched = marlin_moe_permute_scales(s, size_k, size_n, group_size)
        assert batched.shape == loop.shape and torch.equal(batched, loop), (size_k, group_size)
