"""N4D W4A8 decode GEMV (sm_86, native NVFP4 layout): desk emulation of the data flow + GPU parity.

Desk (no GPU):
  * the row-tile mapping covers every row of a 128-row scale tile exactly once, and the lane's ONE 8-byte scale load
    is exactly (rA, 4G..4G+3) + (rB, 4G..4G+3) of the swizzle formula;
  * MODE 0 (diag): lane loads -> A fragments -> block-diagonal masked B -> m16n8k32 (emulated from the PTX fragment
    tables, not from the kernel's intent) gives C[row][n] = exact block-n sum;
  * MODE 1 (xpose): 4 PRMT + the shuffle quad transpose + m16n8k16 per block with the quantiser's permuted
    activation layout gives the exact block sums per (row, token);
  * the quantiser's permuted store formula is the permutation the kernel reads;
  * config rules return launchable tuples for every 27B shape and M 1..48; the seam routes small M to the decode
    GEMV and everything else (M > max, M == 0, K % 128 != 0, SGLANG_W4A8_DECODE_MAX_M=0) to N4A's GEMM.
GPU (TEST27B_GPU=1, own gpuq window, sm_86): every (mode, kw, rw, u) against the exact fp64 reference within the
derived bound of N4A's test, K-shard tile-view scales, N not a multiple of 128, and parity with N4A's GEMM.
"""

from __future__ import annotations

import itertools
import os
import sys

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_nvfp4_w4a8 as T4  # noqa: E402  (N4A's helpers: byte_perm, e2m1 table, swizzle, fp64 reference)

MAGIC_BITS = 0x4B400000
RNG = np.random.default_rng(38)


# ------------------------------------------------------------------------------------------------------------------
# PTX fragment tables (mma.m16n8k32 / m16n8k16, .s8, row.col) -- written from the ISA, independent of the kernel
# ------------------------------------------------------------------------------------------------------------------
def _a_k32(lane, reg, byte):
    g, q = lane >> 2, lane & 3
    return g + 8 * (reg & 1), 4 * q + byte + 16 * (reg >> 1)  # (row, k)


def _b_k32(lane, reg, byte):
    g, q = lane >> 2, lane & 3
    return 4 * q + byte + 16 * reg, g  # (k, col)


def _a_k16(lane, reg, byte):
    g, q = lane >> 2, lane & 3
    return g + 8 * reg, 4 * q + byte


def _b_k16(lane, byte):
    g, q = lane >> 2, lane & 3
    return 4 * q + byte, g


def _c(lane, reg):
    g, q = lane >> 2, lane & 3
    return g + 8 * (reg >> 1), 2 * q + (reg & 1)  # (row, col)


def _bytes_s8(word):
    return np.array([(int(word) >> (8 * i)) & 0xFF for i in range(4)], dtype=np.uint8).view(np.int8).astype(np.int64)


def mma_k32(A, B, C):
    """A [32][4], B [32][2] u32 words, C [32][4] int -> D [32][4] (int64, exact)."""
    Am = np.zeros((16, 32), np.int64)
    Bm = np.zeros((32, 8), np.int64)
    for lane in range(32):
        for r in range(4):
            for i, v in enumerate(_bytes_s8(A[lane][r])):
                Am[_a_k32(lane, r, i)] = v
        for r in range(2):
            for i, v in enumerate(_bytes_s8(B[lane][r])):
                Bm[_b_k32(lane, r, i)] = v
    Cm = Am @ Bm
    return [[C[lane][r] + Cm[_c(lane, r)] for r in range(4)] for lane in range(32)]


def mma_k16(A, B, C):
    Am = np.zeros((16, 16), np.int64)
    Bm = np.zeros((16, 8), np.int64)
    for lane in range(32):
        for r in range(2):
            for i, v in enumerate(_bytes_s8(A[lane][r])):
                Am[_a_k16(lane, r, i)] = v
        for i, v in enumerate(_bytes_s8(B[lane])):
            Bm[_b_k16(lane, i)] = v
    Cm = Am @ Bm
    return [[C[lane][r] + Cm[_c(lane, r)] for r in range(4)] for lane in range(32)]


def conv(x16):
    return int(T4._e2m1x4_to_s8x4(np.array([x16 & 0xFFFF], np.uint32))[0])


def byte_perm(x, y, s):
    return int(T4._byte_perm(np.array([x], np.uint32), np.array([y], np.uint32), np.array([s], np.uint32))[0])


def words(b16):
    return [int.from_bytes(bytes(b16[4 * i : 4 * i + 4]), "little") for i in range(4)]


def dec_rows(seg_bytes):
    """[16 rows, 64 B] packed E2M1 -> int64 [16, 128] of 2*e2m1 (element 2j low nibble)."""
    lo = (seg_bytes & 0xF).astype(np.int64)
    hi = (seg_bytes >> 4).astype(np.int64)
    codes = np.stack([lo, hi], axis=-1).reshape(16, 128)
    return (2 * T4.E2M1[codes]).astype(np.int64)


# ------------------------------------------------------------------------------------------------------------------
# desk
# ------------------------------------------------------------------------------------------------------------------
def row_a_of(T, g):
    t = T & 7
    return (T >> 3) * 128 + 64 * (t >> 2) + 8 * (t & 3) + g


def test_row_tiles_cover_the_scale_tile_and_one_load_holds_both_rows():
    rows = sorted(r for T in range(8) for g in range(8) for r in (row_a_of(T, g), row_a_of(T, g) + 32))
    assert rows == list(range(128))
    stride = 40 * 128  # [N_pad, 40] scales
    for T in range(16):
        t = T & 7
        for g in range(8):
            rA = row_a_of(T, g)
            rB = rA + 32
            for G in range(10):
                base = (T >> 3) * stride + (8 * (t & 3) + g) * 16 + 8 * (t >> 2) + G * 512
                for j in range(4):
                    assert base + j == T4._kernel_scale_offset(np.array(rA), np.array(4 * G + j), stride)
                    assert base + 4 + j == T4._kernel_scale_offset(np.array(rB), np.array(4 * G + j), stride)


def test_diag_mode_block_sums_exact():
    """Lane (g, q) loads bytes [16q, 16q+16) of mma rows g and g+8; 4 m16n8k32 with block-diagonal B."""
    seg = RNG.integers(0, 256, (16, 64), dtype=np.uint8)
    x = RNG.integers(-127, 128, 128).astype(np.int8)  # one token, one 128-element segment
    W = dec_rows(seg)
    C = [[MAGIC_BITS] * 4 for _ in range(32)]
    for j in range(4):
        A, B = [], []
        for lane in range(32):
            g, q = lane >> 2, lane & 3
            wa, wb = words(seg[g, 16 * q : 16 * q + 16]), words(seg[g + 8, 16 * q : 16 * q + 16])
            A.append([conv(wa[j]), conv(wb[j]), conv(wa[j] >> 16), conv(wb[j] >> 16)])
            act = q == (g >> 1)
            msk = 0xFFFFFFFF if (act and (j >> 1) == (g & 1)) else 0
            xw = words(x[16 * g : 16 * g + 16].view(np.uint8)) if act else [0] * 4
            B.append([xw[2 * (j & 1)] & msk, xw[2 * (j & 1) + 1] & msk])
        C = mma_k32(A, B, C)
    blk = (W.reshape(16, 8, 16) * x.astype(np.int64).reshape(1, 8, 16)).sum(-1)  # [row, block]
    for lane in range(32):
        g, q = lane >> 2, lane & 3
        assert C[lane][0] - MAGIC_BITS == blk[g, 2 * q]
        assert C[lane][1] - MAGIC_BITS == blk[g, 2 * q + 1]
        assert C[lane][2] - MAGIC_BITS == blk[g + 8, 2 * q]
        assert C[lane][3] - MAGIC_BITS == blk[g + 8, 2 * q + 1]


def quad_transpose(Y, lane_of_quad):
    """Emulation of the device quad_transpose over one quad: Y[q][t] -> Z[q][src] (shfl_xor within the quad)."""
    R = [[0] * 4 for _ in range(4)]
    for q in range(4):
        b0 = q & 1
        for j in range(2):
            snd_peer = Y[q ^ 1][2 * j] if (q ^ 1) & 1 else Y[q ^ 1][2 * j + 1]
            own = Y[q][2 * j + 1] if b0 else Y[q][2 * j]
            R[q][2 * j + 0] = snd_peer if b0 else own
            R[q][2 * j + 1] = own if b0 else snd_peer
    Z = [[0] * 4 for _ in range(4)]
    for q in range(4):
        b1 = (q >> 1) & 1
        for c in range(2):
            p = q ^ 2
            snd_peer = R[p][c] if (p >> 1) & 1 else R[p][2 + c]
            Z[q][c] = snd_peer if b1 else R[q][c]
            Z[q][2 + c] = R[q][2 + c] if b1 else snd_peer
    return Z


def permute_segment(xrow):
    """The quantiser's permuted store: element k = 16b + 4p + j of a segment -> byte 32p + 4b + j."""
    out = np.zeros(128, np.int8)
    for k in range(128):
        b, p, j = k // 16, (k % 16) // 4, k % 4
        out[32 * p + 4 * b + j] = xrow[k]
    return out


def test_quantiser_store_formula_is_the_permutation():
    # device: thread at k = 8i writes w[0] (elements k..k+3) at 32*pp + 4b and w[1] (k+4..k+7) at 32*(pp+1) + 4b
    xrow = RNG.integers(-127, 128, 256).astype(np.int8)
    dev = np.zeros(256, np.int8)
    for k in range(0, 256, 8):
        seg, kk = k >> 7, k & 127
        b, pp = kk >> 4, (kk & 15) >> 2
        dev[seg * 128 + 32 * pp + 4 * b : seg * 128 + 32 * pp + 4 * b + 4] = xrow[k : k + 4]
        dev[seg * 128 + 32 * (pp + 1) + 4 * b : seg * 128 + 32 * (pp + 1) + 4 * b + 4] = xrow[k + 4 : k + 8]
    want = np.concatenate([permute_segment(xrow[:128]), permute_segment(xrow[128:])])
    np.testing.assert_array_equal(dev, want)


def test_xpose_mode_block_sums_exact():
    seg = RNG.integers(0, 256, (16, 64), dtype=np.uint8)
    X = RNG.integers(-127, 128, (8, 128)).astype(np.int8)  # 8 tokens
    W = dec_rows(seg)
    Xp = np.stack([permute_segment(X[m]) for m in range(8)])
    # per lane: Y words for rows g and g+8, then the quad transpose
    ZA, ZB = [None] * 32, [None] * 32
    for g in range(8):
        YA, YB = [], []
        for q in range(4):
            wa, wb = words(seg[g, 16 * q : 16 * q + 16]), words(seg[g + 8, 16 * q : 16 * q + 16])
            sel = lambda t: 0x7632 if t & 1 else 0x5410  # noqa: E731
            YA.append([byte_perm(wa[t >> 1], wa[2 + (t >> 1)], sel(t)) for t in range(4)])
            YB.append([byte_perm(wb[t >> 1], wb[2 + (t >> 1)], sel(t)) for t in range(4)])
        za, zb = quad_transpose(YA, None), quad_transpose(YB, None)
        for q in range(4):
            ZA[4 * g + q], ZB[4 * g + q] = za[q], zb[q]
    for b in range(8):
        A, B = [], []
        for lane in range(32):
            g, q = lane >> 2, lane & 3
            hA = ZA[lane][b >> 1] >> 16 if b & 1 else ZA[lane][b >> 1]
            hB = ZB[lane][b >> 1] >> 16 if b & 1 else ZB[lane][b >> 1]
            A.append([conv(hA), conv(hB)])
            B.append(int.from_bytes(bytes(Xp[g, 32 * q + 4 * b : 32 * q + 4 * b + 4].view(np.uint8)), "little"))
        D = mma_k16(A, B, [[MAGIC_BITS] * 4 for _ in range(32)])
        blk = W[:, 16 * b : 16 * b + 16] @ X[:, 16 * b : 16 * b + 16].astype(np.int64).T  # [row, token]
        for lane in range(32):
            g, q = lane >> 2, lane & 3
            assert D[lane][0] - MAGIC_BITS == blk[g, 2 * q]
            assert D[lane][1] - MAGIC_BITS == blk[g, 2 * q + 1]
            assert D[lane][2] - MAGIC_BITS == blk[g + 8, 2 * q]
            assert D[lane][3] - MAGIC_BITS == blk[g + 8, 2 * q + 1]


SHAPES_27B = [(8192, 5120), (7936, 5120), (5120, 4096), (5120, 3968), (82816, 5120), (34816, 5120), (5120, 17408)]


def test_config_rules_are_launchable(monkeypatch):
    from sglang.jit_kernel import nvfp4_w4a8_decode as dec

    monkeypatch.delenv("SGLANG_W4A8_DECODE_CFG", raising=False)
    dec.config_for.cache_clear()
    for (n, k), m in itertools.product(SHAPES_27B, range(1, 49)):
        md, kw, rw, u = dec.config_for(m, n, k)
        assert md == (0 if m <= 4 else 1)
        assert kw >= 1 and rw >= 1 and kw * rw <= 8 and u in (1, 2, 4)
    monkeypatch.setenv("SGLANG_W4A8_DECODE_CFG", "1,2,4,1")
    dec.config_for.cache_clear()
    assert dec.config_for(3, 8192, 5120) == (1, 2, 4, 1)
    dec.config_for.cache_clear()


class _Spy:
    def __init__(self):
        self.calls = []

    def dec(self, x, w, s, gs, n, **kw):
        self.calls.append(("decode", x.shape[0]))
        return torch.zeros((x.shape[0], n), dtype=torch.bfloat16)

    def n4a(self, x, w, s, gs, n, **kw):
        self.calls.append(("n4a", x.reshape(-1, x.shape[-1]).shape[0]))
        return torch.zeros((*x.shape[:-1], n), dtype=torch.bfloat16)


def test_seam_routes_small_m_to_decode(monkeypatch):
    from sglang.jit_kernel import nvfp4_w4a8 as n4a
    from sglang.jit_kernel import nvfp4_w4a8_decode as dec
    from sglang.srt.layers.quantization import nvfp4_native_mixed as nm

    # importing the seam module registers its kernel in nm: keep nm's registry as it was for other tests
    monkeypatch.setattr(nm, "_W4A8_KERNEL", nm._W4A8_KERNEL)
    monkeypatch.setattr(nm, "_W4A8_KERNEL_NAME", nm._W4A8_KERNEL_NAME)
    from sglang.srt.layers.quantization import nvfp4_w4a8_int8 as seam

    spy = _Spy()
    monkeypatch.setattr(dec, "nvfp4_w4a8_decode_linear", spy.dec)
    monkeypatch.setattr(n4a, "nvfp4_w4a8_linear", spy.n4a)
    monkeypatch.delenv("SGLANG_W4A8_DECODE_MAX_M", raising=False)
    w = torch.zeros((256, 256), dtype=torch.uint8)  # K = 512
    s = torch.zeros((256, 32), dtype=torch.uint8)
    gs = torch.tensor(1.0)
    for m in (1, 4, 16, 48):
        y = seam.nvfp4_w4a8_int8_apply(torch.zeros((m, 512), dtype=torch.bfloat16), w, s, gs, 256)
        assert y.shape == (m, 256)
    y = seam.nvfp4_w4a8_int8_apply(torch.zeros((2, 3, 512), dtype=torch.bfloat16), w, s, gs, 256)
    assert y.shape == (2, 3, 256)
    seam.nvfp4_w4a8_int8_apply(torch.zeros((49, 512), dtype=torch.bfloat16), w, s, gs, 256)
    seam.nvfp4_w4a8_int8_apply(torch.zeros((0, 512), dtype=torch.bfloat16), w, s, gs, 256)
    w_odd = torch.zeros((256, 240), dtype=torch.uint8)  # K = 480, not a multiple of 128
    seam.nvfp4_w4a8_int8_apply(torch.zeros((2, 480), dtype=torch.bfloat16), w_odd, s, gs, 256)
    monkeypatch.setenv("SGLANG_W4A8_DECODE_MAX_M", "0")
    seam.nvfp4_w4a8_int8_apply(torch.zeros((2, 512), dtype=torch.bfloat16), w, s, gs, 256)
    assert spy.calls == [
        ("decode", 1), ("decode", 4), ("decode", 16), ("decode", 48), ("decode", 6),
        ("n4a", 49), ("n4a", 0), ("n4a", 2), ("n4a", 2),
    ]


# ------------------------------------------------------------------------------------------------------------------
# GPU
# ------------------------------------------------------------------------------------------------------------------
gpu = T4.gpu


def _all_cfgs(m):
    modes = (0, 1) if m <= 8 else (1,)
    pairs = [(kw, rw) for kw in (1, 2, 4, 8) for rw in (1, 2, 4, 8) if kw * rw <= 8]
    return [(md, kw, rw, u) for md in modes for (kw, rw) in pairs for u in (1, 2, 4)]


@gpu
@pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 8, 9, 16, 17, 33, 48])
@pytest.mark.parametrize("n,k", [(256, 1024), (224, 640), (1024, 384)])
def test_decode_every_config_within_bound(m, n, k):
    from sglang.jit_kernel import nvfp4_w4a8_decode as dec
    from sglang.jit_kernel.nvfp4_w4a8 import nvfp4_w4a8_quantize_activation

    gen = torch.Generator().manual_seed(m * 7919 + n + k)
    w, ws, ws2 = T4._random_nvfp4(n, k, gen)
    swz = T4._swizzle_like_modelopt(ws).cuda()  # [ceil128(n), k/16] -- n=224 exercises the partial tile
    x = (torch.randn((m, k), generator=gen) * 3).to(torch.bfloat16).cuda()
    xq, xs = nvfp4_w4a8_quantize_activation(x)
    for cfg in _all_cfgs(m):
        R, bound, _ = T4._reference_and_bound(xq, xs, w, ws, ws2, splits=cfg[1])
        y = dec.nvfp4_w4a8_decode_linear(x, w, swz, ws2, n, cfg=cfg).double()
        viol = (y - R).abs() - bound
        assert float(viol.max()) <= 0.0, (cfg, float(viol.max()))
        # the quantiser is N4A's arithmetic: identical int8 (natural layout) and scales
        xqd, xsd = dec.quantize_activation(x, permuted=False)
        assert torch.equal(xqd, xq) and torch.equal(xsd, xs)


@gpu
def test_decode_k_shard_tile_view_and_parity_with_n4a():
    from sglang.jit_kernel import nvfp4_w4a8_decode as dec
    from sglang.jit_kernel.nvfp4_w4a8 import nvfp4_w4a8_linear

    gen = torch.Generator().manual_seed(5)
    n, k = 256, 2048
    w, ws, ws2 = T4._random_nvfp4(n, k, gen)
    full = T4._swizzle_like_modelopt(ws).cuda().view(torch.uint8)
    view = full.view(n // 128, (k // 16) * 128)
    kb0, kr = 32, 64  # K columns [512, 1536)
    w_sh = w[:, kb0 * 8 : (kb0 + kr) * 8]
    sv = view[:, kb0 * 128 : (kb0 + kr) * 128]
    for m in (1, 4, 16, 48):
        x = torch.randn((m, kr * 16), generator=gen).to(torch.bfloat16).cuda()
        y = dec.nvfp4_w4a8_decode_linear(x, w_sh, sv, ws2, n)
        own = T4._swizzle_like_modelopt(ws[:, kb0 : kb0 + kr].contiguous()).cuda()
        y2 = nvfp4_w4a8_linear(x, w_sh.contiguous(), own, ws2, n)
        # same arithmetic, different FP32 accumulation order -> within a few bf16 ulp
        d = (y.float() - y2.float()).abs().max() / y2.float().abs().max()
        assert float(d) < 2e-2, (m, float(d))


def test_prebuild_names_the_modules_load_jit_builds():
    """The image pre-build (prebuild_nvfp4_w4a8) must name exactly the modules the serving path loads, or the
    first boot compiles after all."""
    import inspect

    from sglang.jit_kernel import nvfp4_w4a8, nvfp4_w4a8_decode, prebuild_nvfp4_w4a8

    names = [n for n, _ in prebuild_nvfp4_w4a8._modules()]
    assert names == ["nvfp4_w4a8_decode_sm86", "nvfp4_w4a8_sm86"]
    for name, mod in zip(names, (nvfp4_w4a8_decode, nvfp4_w4a8)):
        assert f'load_jit(\n        "{name}",' in inspect.getsource(mod)
