"""NVFP4 W4A8-g16 on sm_86 INT8 tensor cores (Backlog #38).

Desk part (no GPU): the three bit-level claims the kernel stands on, checked exhaustively in numpy:
  * the E2M1 -> INT8 table the kernel builds with three PRMTs is 2 * e2m1 for all 2^16 inputs;
  * the raw-bit E4M3 -> FP32 conversion is e4m3 * 2^-120 for every finite code (subnormals included);
  * fma(1.5*2^23 + S, s, -1.5*2^23*s) is exact for every e4m3 s and every |S| <= 16*12*127;
  * the kernel's scale address formula reproduces the torch swizzle of modelopt_quant.py byte for byte,
    including the K-shard tile view of the layout contract (N4B, §4).

GPU part (TEST27B_GPU=1 in an own gpuq window, sm_86 only): quantiser parity and the GEMM against
(a) the exact fp64 reference with a DERIVED elementwise bound and (b) the fp32 reference the order names
("dequantise NVFP4 to fp32, quantise A per token to INT8, matmul fp32"), on the layout the REAL
ModelOptFp4LinearMethod.process_weights_after_loading produces on its native (sm_120 cutlass) branch.
"""

from __future__ import annotations

import os

# NVML order == CUDA order on this rig (CUDA's default is fastest-first: index 0 would be the 5090).
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import numpy as np
import pytest
import torch

E2M1 = np.array(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=np.float64,
)
MAGIC = 12582912.0  # 1.5 * 2^23


# ----------------------------------------------------------------------------------------------------------
# bit-level emulation of the device helpers (must mirror nvfp4_w4a8_sm86.cuh exactly)
# ----------------------------------------------------------------------------------------------------------
def _byte_perm(x: np.ndarray, y: np.ndarray, s: np.ndarray) -> np.ndarray:
    """CUDA __byte_perm: result byte n = input byte ((s >> 4n) & 7) of the 8-byte pool {x, y}."""
    x = x.astype(np.uint64)
    y = y.astype(np.uint64)
    s = s.astype(np.uint64)
    pool = x | (y << np.uint64(32))
    out = np.zeros_like(x)
    for n in range(4):
        sel = (s >> np.uint64(4 * n)) & np.uint64(7)
        byte = (pool >> (sel * np.uint64(8))) & np.uint64(0xFF)
        out |= byte << np.uint64(8 * n)
    return out.astype(np.uint32)


def _e2m1x4_to_s8x4(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.uint32)
    pos = _byte_perm(np.full_like(x, 0x03020100), np.full_like(x, 0x0C080604), x)
    neg = _byte_perm(np.full_like(x, 0xFDFEFF00), np.full_like(x, 0xF4F8FAFC), x)
    sel = ((x & np.uint32(0x8888)) >> np.uint32(1)) | np.uint32(0x3210)
    return _byte_perm(pos, neg, sel)


def _e4m3_scaled_bits(byte: np.ndarray) -> np.ndarray:
    y = byte.astype(np.uint32) << np.uint32(24)
    z = (y.view(np.int32) >> 4) & np.int32(np.uint32(0x87F00000).view(np.int32))
    return z.view(np.float32)


def _e4m3_values() -> np.ndarray:
    codes = torch.arange(256, dtype=torch.uint8)
    return codes.view(torch.float8_e4m3fn).to(torch.float64).numpy()


def test_e2m1_table_exhaustive():
    x = np.arange(1 << 16, dtype=np.uint32)
    got = _e2m1x4_to_s8x4(x).view(np.int8).reshape(-1, 4).astype(np.int64)
    nib = np.stack([(x >> (4 * i)) & 0xF for i in range(4)], axis=1)
    want = (2 * E2M1[nib]).astype(np.int64)
    np.testing.assert_array_equal(got, want)


def test_e4m3_raw_bits_exhaustive():
    vals = _e4m3_values()
    codes = np.arange(256, dtype=np.uint32)
    finite = np.isfinite(vals)
    got = _e4m3_scaled_bits(codes).astype(np.float64) * 2.0**120
    np.testing.assert_array_equal(got[finite], vals[finite])
    # the subnormal codes land in FP32 subnormals: the kernel must run without FTZ
    sub = (codes & 0x78) == 0
    assert np.all(
        np.abs(_e4m3_scaled_bits(codes[sub & (codes & 7 != 0)]))
        < np.finfo(np.float32).tiny
    )


def _e4m3_times_512(b: np.ndarray) -> np.ndarray:
    """Mirror of the device helper e4m3_times_512 (the exact integer epilogue's LUT)."""
    b = b.astype(np.int64)
    mag = b & 0x7F
    e, m = mag >> 3, mag & 7
    c = np.where(e > 0, (8 | m) << np.maximum(e - 1, 0), m)
    c = np.where(mag == 0x7F, 0, c)
    return np.where(b & 0x80, -c, c)


def test_e4m3_integer_lut_exhaustive():
    vals = _e4m3_values()
    codes = np.arange(256)
    finite = np.isfinite(vals)
    c = _e4m3_times_512(codes)
    np.testing.assert_array_equal(c[finite].astype(np.float64), vals[finite] * 512.0)
    # int64 headroom: |S * c| <= 24384 * 448 * 512 < 2^33, and 2^11 blocks (K = 32768) stay < 2^44
    assert np.abs(c).max() == 448 * 512 and 24384 * 448 * 512 < 2**33


def test_block_product_exact():
    """t = fma(M + S, s, -M s) == s*S exactly <=> s*S and -M*s are FP32-representable (FMA rounds once)."""
    s = _e4m3_scaled_bits(np.arange(256, dtype=np.uint32)).astype(np.float64)
    s = s[np.isfinite(s) & (s != 0)]
    S = np.arange(-24384, 24385, dtype=np.float64)
    prod = s[:, None] * S[None, :]  # exact in fp64 (<= 4 + 15 significant bits)
    np.testing.assert_array_equal(prod.astype(np.float32).astype(np.float64), prod)
    negc = -MAGIC * s
    np.testing.assert_array_equal(negc.astype(np.float32).astype(np.float64), negc)
    # |S| < 2^22 keeps 1.5*2^23 + S an exactly represented float whose bits are 0x4B400000 + S
    f = (
        (np.int64(0x4B400000) + S.astype(np.int64))
        .astype(np.int32)
        .view(np.float32)
        .astype(np.float64)
    )
    np.testing.assert_array_equal(f, MAGIC + S)


# ----------------------------------------------------------------------------------------------------------
# layout: the torch swizzle of modelopt_quant.py process_weights_after_loading (native branch) vs the kernel
# ----------------------------------------------------------------------------------------------------------
def _swizzle_like_modelopt(scales: torch.Tensor) -> torch.Tensor:
    """The exact op sequence of ModelOptFp4LinearMethod.process_weights_after_loading (native branch) and
    utils.swizzle_blockscale, minus the trailing .cuda()."""
    scales = scales.unsqueeze(0)
    B, M, K = scales.shape
    Mp = (M + 127) // 128 * 128
    Kp = (K + 3) // 4 * 4
    padded = torch.zeros((B, Mp, Kp), dtype=scales.dtype)
    padded[:B, :M, :K] = scales
    padded = padded.reshape(B, Mp // 128, 4, 32, Kp // 4, 4).permute((0, 1, 4, 3, 2, 5))
    return padded.contiguous().reshape(Mp, Kp)


def _kernel_scale_offset(
    n: np.ndarray, kb: np.ndarray, tile_row_stride: int
) -> np.ndarray:
    return (
        (n // 128) * tile_row_stride
        + (kb // 4) * 512
        + (n % 32) * 16
        + ((n % 128) // 32) * 4
        + kb % 4
    )


@pytest.mark.parametrize("n,k16", [(128, 4), (256, 320), (200, 7), (5120, 1088)])
def test_swizzle_address_formula(n, k16):
    g = torch.Generator().manual_seed(n * 1000 + k16)
    raw = torch.randint(0, 0x7F, (n, k16), generator=g, dtype=torch.uint8)
    swz = (
        _swizzle_like_modelopt(raw.view(torch.float8_e4m3fn))
        .view(torch.uint8)
        .numpy()
        .reshape(-1)
    )
    kp = (k16 + 3) // 4 * 4
    nn, kk = np.meshgrid(np.arange(n), np.arange(k16), indexing="ij")
    off = _kernel_scale_offset(nn, kk, kp * 128)
    np.testing.assert_array_equal(swz[off], raw.numpy())


def test_k_shard_tile_view_is_native_swizzle_of_the_shard():
    """Contract §4: a K-shard is a column slice of the tile view [N/128, Kp*128], byte-equal to the native
    swizzle of the shard's own [N, K_r/16] scales -- and the kernel's tile-row stride addresses it.
    """
    n, k16 = 256, 64
    raw = torch.randint(0, 0x7F, (n, k16), dtype=torch.uint8)
    full = _swizzle_like_modelopt(raw.view(torch.float8_e4m3fn)).view(torch.uint8)
    view = full.view(n // 128, k16 * 128)
    kb0, kr = 16, 24  # multiples of 4 scale columns
    shard_view = view[:, kb0 * 128 : (kb0 + kr) * 128]
    own = _swizzle_like_modelopt(
        raw[:, kb0 : kb0 + kr].contiguous().view(torch.float8_e4m3fn)
    ).view(torch.uint8)
    np.testing.assert_array_equal(
        shard_view.contiguous().numpy().reshape(-1), own.numpy().reshape(-1)
    )
    nn, kk = np.meshgrid(np.arange(n), np.arange(kr), indexing="ij")
    flat = full.numpy().reshape(-1)
    off = kb0 * 128 + _kernel_scale_offset(nn, kk, shard_view.stride(0))
    np.testing.assert_array_equal(flat[off], raw[:, kb0 : kb0 + kr].numpy())


# ----------------------------------------------------------------------------------------------------------
# GPU
# ----------------------------------------------------------------------------------------------------------
def _gpu_ok() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (8, 6)


gpu = pytest.mark.skipif(
    not _gpu_ok(), reason="needs an sm_86 GPU (TEST27B_GPU=1 in an own gpuq window)"
)


def _random_nvfp4(n: int, k: int, gen: torch.Generator, device="cuda"):
    """A checkpoint-shaped NVFP4 weight: uint8 [N, K/2], e4m3 [N, K/16] (incl. subnormal scales), ws2."""
    w = torch.randint(0, 256, (n, k // 2), generator=gen, dtype=torch.uint8)
    # block scales: mostly the normal range modelopt produces, some subnormals, some zeros
    codes = torch.randint(0x08, 0x7E, (n, k // 16), generator=gen, dtype=torch.uint8)
    sub = torch.rand((n, k // 16), generator=gen) < 0.03
    codes[sub] = torch.randint(
        0, 8, (int(sub.sum()),), generator=gen, dtype=torch.uint8
    )
    ws = codes.view(torch.float8_e4m3fn)
    ws2 = torch.tensor([1.7e-4], dtype=torch.float32)
    return w.to(device), ws.to(device), ws2.to(device)


def _dequant_fp64(w: torch.Tensor, ws: torch.Tensor) -> torch.Tensor:
    """[N, K] fp64 values of 2 * e2m1 * e4m3 (weight_scale_2 and the 1/2 applied by the caller)."""
    lut = torch.tensor(2 * E2M1, dtype=torch.float64, device=w.device)
    n = w.shape[0]
    codes = torch.empty((n, w.shape[1] * 2), dtype=torch.long, device=w.device)
    codes[:, 0::2] = (w & 0xF).long()
    codes[:, 1::2] = (w >> 4).long()
    vals = lut[codes].view(n, -1, 16) * ws.to(torch.float64).unsqueeze(-1)
    return vals.view(n, -1)


def _reference_and_bound(xq, xs, w, ws, ws2, splits: int, exact: bool = False):
    """Exact fp64 reference R and the derived elementwise bound for the bf16 kernel output.

    Kernel: t_b = s_b*S_b exact; acc = fp32 sum over nb blocks (plus <= splits-1 partial-sum adds);
    y = acc * factor with factor = fl(fl(xs*ws2) * 2^119) (2 roundings) and one rounding of the product;
    then bf16. |fl32 - R| <= (nb - 1 + splits - 1 + 3) * u32 * T (+O(u^2)), T = xs*ws2/2 * sum_b |t_b|;
    |bf16(y) - y| <= u_bf16 * |y|.  u32 = 2^-24, u_bf16 = 2^-8 (round to nearest, 8 significant bits).
    """
    m, kp = xq.shape
    n = w.shape[0]
    nb = kp // 16
    wd = _dequant_fp64(w, ws)  # [N, K]
    xqd = xq.to(torch.float64)
    blk = torch.einsum(
        "mbk,nbk->mnb", xqd.view(m, nb, 16), wd.view(n, nb, 16)
    )  # s_b*S_b, exact in fp64
    fac = (xs.to(torch.float64) * ws2.to(torch.float64) * 0.5).unsqueeze(1)
    R = blk.sum(-1) * fac
    T = blk.abs().sum(-1) * fac
    u32, ubf = 2.0**-24, 2.0**-8
    if exact:
        # INT64 epilogue: the block sums are accumulated exactly; only int64->fp32, fl(xs*ws2) and the
        # final product round (x 2^-10 is exact): |fl32 - R| <= 3 u32 |R| (+O(u^2)).
        e32 = 3 * u32 * R.abs() * 1.0001
        return R, ubf * (R.abs() + e32) + e32, wd
    e32 = (nb + splits + 1) * u32 * T * 1.0001
    bound = ubf * (R.abs() + e32) + e32
    return R, bound, wd


@gpu
def test_quant_act_matches_torch_emulation_and_per_token_quant_int8():
    from sglang.jit_kernel.nvfp4_w4a8 import nvfp4_w4a8_quantize_activation
    from sglang.srt.layers.quantization.int8_kernel import per_token_quant_int8

    g = torch.Generator(device="cuda").manual_seed(1)
    for m, k, kp in [
        (1, 5120, 5120),
        (7, 17408, 17408),
        (33, 96, 128),
        (300, 5120, 5120),
        (5, 40, 64),
    ]:
        x = (torch.randn((m, k), generator=g, device="cuda") * 3).to(torch.bfloat16)
        x[:, :: max(1, k // 7)] *= 40  # outlier channels
        xq, xs = nvfp4_w4a8_quantize_activation(x, k_padded=kp)
        xf = x.float()
        amax = xf.abs().amax(-1).clamp_min(1e-10)
        inv = (
            torch.full_like(amax, 127.0) / amax
        )  # true IEEE division (127.0 / t is reciprocal * 127)
        v = (xf * inv[:, None]).double()
        want = (torch.sign(v) * torch.floor(v.abs() + 0.5)).to(torch.int8)
        assert torch.equal(xq[:, :k], want)
        assert torch.all(xq[:, k:] == 0)
        # true IEEE division on both sides (torch's `t / 127.0` multiplies by the reciprocal)
        assert torch.equal(xs, amax / torch.full_like(amax, 127.0))
        if kp == k and k % 1 == 0:
            tq, ts = per_token_quant_int8(x)
            # Triton's division may differ by an ulp; count, do not hide it
            mism = (tq != xq).sum().item()
            assert mism <= xq.numel() * 1e-3, mism
            torch.testing.assert_close(ts.view(-1), xs, rtol=2e-7, atol=0)


@gpu
@pytest.mark.parametrize("exact", [False, True])
@pytest.mark.parametrize(
    "m,n,k,splits",
    [
        (1, 256, 128, 1),
        (5, 200, 96, 1),  # N not a multiple of 128/32, K tail (Kp % 64 == 32)
        (8, 384, 1024, 1),
        (16, 256, 2048, 4),
        (33, 512, 640, 1),
        (48, 256, 4096, 8),
        (64, 128, 512, 2),
        (100, 384, 1024, 1),
        (300, 256, 512, 3),
    ],
)
def test_gemm_vs_reference(m, n, k, splits, exact):
    from sglang.jit_kernel.nvfp4_w4a8 import (
        exact_config_for_m,
        nvfp4_w4a8_gemm,
        nvfp4_w4a8_quantize_activation,
    )
    from sglang.srt.layers.quantization.modelopt_quant import pad_nvfp4_weight
    from sglang.srt.layers.quantization.utils import swizzle_blockscale

    gen = torch.Generator().manual_seed(m * 7919 + n * 31 + k)
    w, ws, ws2 = _random_nvfp4(n, k, gen)
    wp, pad_cols = pad_nvfp4_weight(w)  # the sm_120 path's own padding (N, K to 32)
    wsz = swizzle_blockscale(ws)  # the sm_120 path's own swizzle
    kp = wp.shape[1] * 2
    x = (torch.randn((m, k), generator=gen) * 2).to(torch.bfloat16).cuda()
    xq, xs = nvfp4_w4a8_quantize_activation(x, k_padded=kp)
    cfg = exact_config_for_m(m) if exact else -1
    out = nvfp4_w4a8_gemm(xq, xs, wp, wsz, ws2, n, split_k=splits, config=cfg)
    torch.cuda.synchronize()

    wsp = torch.zeros((n, kp // 16), dtype=torch.float8_e4m3fn, device="cuda")
    wsp[:, : k // 16] = ws
    wpd = wp[:n].contiguous()
    R, bound, wd = _reference_and_bound(xq, xs, wpd, wsp, ws2, splits, exact=exact)
    err = (out.double() - R).abs()
    assert torch.all(err <= bound), f"max err/bound {(err / bound).max().item():.3f}"

    # the order's fp32 reference: dequant to fp32, A = xq * xs, fp32 matmul (no TF32)
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        w32 = (wd * (ws2.double() * 0.5)).float()
        a32 = xq.float() * xs[:, None]
        R32 = a32 @ w32.t()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev
    # fp32 GEMM bound: kp * u32 * sum|a*w|
    T32 = (xq.double().abs() * xs.double()[:, None]) @ (
        wd.abs() * ws2.double() * 0.5
    ).t()
    b32 = bound + kp * 2.0**-24 * T32
    assert torch.all((out.double() - R32.double()).abs() <= b32)


@gpu
@pytest.mark.parametrize("config", list(range(21)))
def test_every_tile_variant(config):
    from sglang.jit_kernel.nvfp4_w4a8 import (
        nvfp4_w4a8_gemm,
        nvfp4_w4a8_quantize_activation,
    )
    from sglang.srt.layers.quantization.utils import swizzle_blockscale

    gen = torch.Generator().manual_seed(100 + config)
    m, n, k = (
        150,
        384,
        1088,
    )  # 3 token tiles of 64 / 2 of 128, K tail at 64-granularity (17 stages)
    w, ws, ws2 = _random_nvfp4(n, k, gen)
    x = (torch.randn((m, k), generator=gen) * 2).to(torch.bfloat16).cuda()
    xq, xs = nvfp4_w4a8_quantize_activation(x)
    for splits in (1, 3):
        out = nvfp4_w4a8_gemm(
            xq, xs, w, swizzle_blockscale(ws), ws2, n, split_k=splits, config=config
        )
        R, bound, _ = _reference_and_bound(
            xq, xs, w, ws, ws2, splits, exact=8 <= config <= 14
        )
        err = (out.double() - R).abs()
        assert torch.all(
            err <= bound
        ), f"cfg {config} splits {splits}: max err/bound {(err / bound).max().item():.3f}"


@gpu
def test_real_process_weights_after_loading_native_branch_layout(monkeypatch):
    """Feed the kernel exactly what ModelOptFp4LinearMethod.process_weights_after_loading hands the sm_120
    cutlass kernel (native branch forced on the 3080: is_blackwell_supported patched, backend = cutlass),
    fused gate_up with two partitions, then a K-shard of a row-parallel layer through the tile view.
    """
    import sglang.srt.layers.quantization.fp4_utils as fp4_utils
    import sglang.srt.layers.quantization.modelopt_quant as mq
    from sglang.jit_kernel.nvfp4_w4a8 import nvfp4_w4a8_linear

    monkeypatch.setattr(mq, "is_blackwell_supported", lambda *a, **k: True)
    monkeypatch.setattr(
        fp4_utils, "FP4_GEMM_RUNNER_BACKEND", fp4_utils.Fp4GemmRunnerBackend.CUTLASS
    )

    cfg = mq.ModelOptFp4Config(is_checkpoint_nvfp4_serialized=True, group_size=16)
    method = mq.ModelOptFp4LinearMethod(cfg)
    layer = torch.nn.Module()
    k, parts = 512, [256, 256]
    method.create_weights(
        layer,
        k,
        parts,
        k,
        sum(parts),
        torch.bfloat16,
        weight_loader=lambda *a, **kw: None,
    )
    gen = torch.Generator().manual_seed(5)
    w, ws, ws2 = _random_nvfp4(sum(parts), k, gen)
    layer.weight.data = w.clone()
    layer.weight_scale.data = ws.clone()
    layer.input_scale.data = torch.tensor([0.01, 0.01], device="cuda")
    layer.weight_scale_2.data = torch.cat([ws2, ws2]).cuda()
    method.process_weights_after_loading(layer)

    assert layer.weight.dtype == torch.uint8 and tuple(layer.weight.shape) == (512, 256)
    assert (
        layer.weight_scale_interleaved is layer.weight_scale
    )  # no padding -> alias (contract §2.3)
    x = (
        torch.randn((24, k), generator=torch.Generator().manual_seed(6))
        .to(torch.bfloat16)
        .cuda()
    )
    y = nvfp4_w4a8_linear(
        x,
        layer.weight,
        layer.weight_scale_interleaved,
        layer.weight_scale_2.max(),
        layer.output_size_per_partition,
    )
    # reference from the RAW checkpoint tensors (w, ws), not from the processed ones
    from sglang.jit_kernel.nvfp4_w4a8 import nvfp4_w4a8_quantize_activation

    xq, xs = nvfp4_w4a8_quantize_activation(x.view(-1, k))
    R, bound, _ = _reference_and_bound(xq, xs, w, ws, ws2, 1)
    assert torch.all((y.double() - R).abs() <= bound)

    # K-shard: columns [128, 384) of the same weight, scale via the tile view (no copy)
    k0, k1 = 128, 384
    tile_view = layer.weight_scale_interleaved.view(torch.uint8).view(
        512 // 128, (k // 16) * 128
    )
    ws_shard = tile_view[:, (k0 // 16) * 128 : (k1 // 16) * 128]
    w_shard = layer.weight[:, k0 // 2 : k1 // 2]
    xs_ = x[:, k0:k1].contiguous()
    y2 = nvfp4_w4a8_linear(xs_, w_shard, ws_shard, layer.weight_scale_2.max(), 512)
    xq2, xs2 = nvfp4_w4a8_quantize_activation(xs_)
    R2, bound2, _ = _reference_and_bound(
        xq2,
        xs2,
        w[:, k0 // 2 : k1 // 2].contiguous(),
        ws[:, k0 // 16 : k1 // 16].contiguous(),
        ws2,
        1,
    )
    assert torch.all((y2.double() - R2).abs() <= bound2)


@gpu
def test_real_checkpoint_rows_quality_vs_w4a16():
    """Real Qwen3.8-27B NVFP4 rows (layer 0 down_proj, first 256 rows): the exact-reference bound, and the
    W4A8 vs W4A16 relative error as a quality measure (reported, loosely bounded)."""
    path = "/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-27B-NVFP4-RadixArk"
    if not os.path.isdir(path):
        pytest.skip("checkpoint not present")
    import json

    from safetensors import safe_open

    from sglang.jit_kernel.nvfp4_w4a8 import (
        nvfp4_w4a8_gemm,
        nvfp4_w4a8_quantize_activation,
    )
    from sglang.srt.layers.quantization.utils import swizzle_blockscale

    idx = json.load(open(os.path.join(path, "model.safetensors.index.json")))[
        "weight_map"
    ]
    base = "model.language_model.layers.0.mlp.down_proj."
    t = {}
    for s in ("weight", "weight_scale", "weight_scale_2"):
        with safe_open(os.path.join(path, idx[base + s]), "pt") as f:
            t[s] = (
                f.get_slice(base + s)[:256]
                if s != "weight_scale_2"
                else f.get_tensor(base + s)
            )
    w, ws, ws2 = (
        t["weight"].cuda(),
        t["weight_scale"].cuda(),
        t["weight_scale_2"].reshape(1).float().cuda(),
    )
    k = w.shape[1] * 2
    x = (
        torch.randn((16, k), generator=torch.Generator().manual_seed(7))
        .to(torch.bfloat16)
        .cuda()
    )
    xq, xs = nvfp4_w4a8_quantize_activation(x)
    y = nvfp4_w4a8_gemm(xq, xs, w, swizzle_blockscale(ws), ws2, 256, split_k=1)
    R, bound, wd = _reference_and_bound(xq, xs, w, ws, ws2, 1)
    assert torch.all((y.double() - R).abs() <= bound)
    y16 = x.double() @ (wd * ws2.double() * 0.5).t()
    rel = ((y.double() - y16).norm() / y16.norm()).item()
    assert rel < 2e-2, rel
