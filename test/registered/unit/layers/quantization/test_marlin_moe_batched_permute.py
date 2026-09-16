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


def test_torch_zero_point_path_is_bit_identical_to_numpy():
    from sglang.srt.layers.quantization.marlin_utils import awq_to_marlin_zero_points_torch

    g = torch.Generator().manual_seed(2)
    for num_bits, size_n, size_k in ((4, 256, 12), (8, 128, 7)):
        pf = 32 // num_bits
        packed = torch.randint(-(2**31), 2**31 - 1, (size_k, size_n // pf), generator=g, dtype=torch.int32)
        ref = awq_to_marlin_zero_points(packed, size_k, size_n, num_bits)
        got = awq_to_marlin_zero_points_torch(packed, size_k, size_n, num_bits)
        assert got.dtype == ref.dtype and got.device == ref.device and torch.equal(got, ref), num_bits


def test_batched_moe_zero_point_path_never_enters_numpy_unpack(monkeypatch):
    """fn2a (16.09.2026): the batched [E*k, n/pf] stack through the numpy
    unpack_cols/pack_cols loops took 93 s per tensor on the desk (strided
    scatter over a 128 MB array) against 0.23 s for the per-expert loop and
    0.32 s for the torch path -- 13..65 s per layer on the 5090 rank instead
    of 1.3 s. The MoE path must stay off the numpy helpers."""
    import sglang.srt.layers.quantization.marlin_utils as mu
    import sglang.srt.layers.quantization.utils as qu

    def boom(*a, **k):
        raise AssertionError("numpy unpack_cols/pack_cols entered on the batched MoE path")

    monkeypatch.setattr(qu, "unpack_cols", boom)
    monkeypatch.setattr(qu, "pack_cols", boom)
    monkeypatch.setattr(mu, "unpack_cols", boom, raising=False)
    monkeypatch.setattr(mu, "pack_cols", boom, raising=False)
    g = torch.Generator().manual_seed(3)
    packed = torch.randint(-(2**31), 2**31 - 1, (4, 8, 32), generator=g, dtype=torch.int32)
    out = mu.moe_awq_to_marlin_zero_points(packed, 8, 256, 4)
    assert out.shape == packed.shape and out.dtype == torch.int32
