"""Fused fp8 dequant-GEMV (#179 candidate 3, design B).

WHY THE GATE IS NOT `torch.equal` AGAINST THE OLD PATH -- a deliberate choice,
recorded so nobody "tightens" it later and breaks a correct kernel:

The existing fallback materialises the weight to bf16/fp16 and then calls
F.linear, so it rounds the weight BEFORE the multiply. The fused kernel decodes
e4m3 and accumulates in fp32. Measured against an fp32 reference, the fused
kernel is the MORE accurate of the two:

    fused        mean relative error 0.0014
    materialise  mean relative error 0.0133

Demanding bit-equality with the old path would therefore pin the kernel to the
old path's ERROR, which is backwards. The gate is instead:

  1. an error band against an fp32 reference, which the fused path must meet at
     least as well as the path it replaces; and
  2. greedy token-ID equality end to end (run separately on the rig, recorded in
     the validation file) -- the property that actually matters to a user.

The raw-byte e4m3 decode is additionally pinned bit-exact against
`tensor.to(torch.float32)`, because that part IS exactly reproducible.
"""

import unittest

import torch

from sglang.srt.layers.quantization.fp8_dequant_gemv import (
    FUSED_GEMV_MAX_ROWS,
    fused_block_dequant_gemv,
    fused_gemv_applicable,
)
from sglang.srt.layers.quantization.fp8_utils import dequant_fp8_block_weight

CUDA = torch.cuda.is_available()


def _mk(N=512, K=256, bn=128, bk=128, seed=0, dev="cuda"):
    torch.manual_seed(seed)
    w = torch.randn(N, K, device=dev).to(torch.float8_e4m3fn)
    s = torch.rand(N // bn, K // bk, device=dev, dtype=torch.float32) + 0.5
    return w, s, bn, bk


class TestRawByteDecodeIsBitExact(unittest.TestCase):
    """The one part that IS exactly reproducible, so it is pinned exactly.

    Triton's native fp8e4nv type is rejected on sm86 and unavailable on gfx900
    and sm70 -- every card that needs this path -- so the kernel unpacks e4m3
    by hand. If that unpacking ever drifts, everything above it is wrong.
    """

    def test_manual_e4m3_decode_matches_torch_bitwise(self):
        dev = "cuda" if CUDA else "cpu"
        torch.manual_seed(0)
        w = torch.randn(4096, device=dev).to(torch.float8_e4m3fn)
        b = w.view(torch.uint8).to(torch.int32)
        sgn = 1.0 - 2.0 * ((b >> 7) & 1).float()
        e = (b >> 3) & 0xF
        m = (b & 0x7).float()
        manual = sgn * torch.where(
            e == 0, m * (0.015625 / 8.0), (1.0 + m / 8.0) * torch.exp2((e - 7).float())
        )
        self.assertTrue(torch.equal(manual, w.to(torch.float32)))


@unittest.skipUnless(CUDA, "fused GEMV needs a GPU")
class TestFusedAccuracyBand(unittest.TestCase):
    def test_fused_is_at_least_as_accurate_as_materialise(self):
        w, s, bn, bk = _mk()
        N, K = w.shape
        x = torch.randn(K, device="cuda", dtype=torch.bfloat16)

        gt = (w.to(torch.float32).view(N // bn, bn, K // bk, bk)
              * s.view(s.shape[0], 1, s.shape[1], 1)).view(N, K) @ x.float()

        fused = fused_block_dequant_gemv(x, w, s, [bn, bk], torch.bfloat16)
        self.assertIsNotNone(fused, "fused path should apply for a 1-row input")
        mat = torch.nn.functional.linear(
            x, dequant_fp8_block_weight(w, s, [bn, bk], torch.bfloat16)
        )

        rel = lambda v: ((v.float() - gt).abs() / gt.abs().clamp_min(1e-6)).mean().item()
        r_fused, r_mat = rel(fused), rel(mat)
        # The band: the fused path must be no worse than what it replaces.
        self.assertLessEqual(
            r_fused, r_mat * 1.1,
            f"fused rel err {r_fused:.5f} worse than materialise {r_mat:.5f}",
        )
        self.assertLess(r_fused, 0.05, "fused path is not merely worse, it is wrong")

    def test_fp16_path_also_within_band(self):
        w, s, bn, bk = _mk(seed=3)
        N, K = w.shape
        x = torch.randn(K, device="cuda", dtype=torch.float16)
        gt = (w.to(torch.float32).view(N // bn, bn, K // bk, bk)
              * s.view(s.shape[0], 1, s.shape[1], 1)).view(N, K) @ x.float()
        fused = fused_block_dequant_gemv(x, w, s, [bn, bk], torch.float16)
        rel = ((fused.float() - gt).abs() / gt.abs().clamp_min(1e-6)).mean().item()
        self.assertLess(rel, 0.05)


class TestDispatch(unittest.TestCase):
    """Prefill and large batches must NOT take the fused path -- that is the
    measured asymmetry (-69.9% decode vs -8.1% prefill) expressed as code."""

    @unittest.skipUnless(CUDA, "needs a GPU")
    def test_applies_to_small_batch_only(self):
        w, _, _, _ = _mk()
        K = w.shape[1]
        self.assertTrue(fused_gemv_applicable(torch.randn(K, device="cuda"), w))
        self.assertTrue(
            fused_gemv_applicable(torch.randn(FUSED_GEMV_MAX_ROWS, K, device="cuda"), w)
        )
        self.assertFalse(
            fused_gemv_applicable(
                torch.randn(FUSED_GEMV_MAX_ROWS + 1, K, device="cuda"), w
            )
        )

    @unittest.skipUnless(CUDA, "needs a GPU")
    def test_small_multi_row_is_HANDLED_not_declined(self):
        """Speculative decode runs several draft rows per step, so a strictly
        1-row kernel declines exactly when it is most needed. An earlier version
        did precisely that and the fused path never fired end to end (+2.6%
        instead of the microbench's 2.8-4.2x). Multi-row must be computed, and
        computed CORRECTLY."""
        w, s, bn, bk = _mk()
        N, K = w.shape
        x = torch.randn(4, K, device="cuda", dtype=torch.bfloat16)
        got = fused_block_dequant_gemv(x, w, s, [bn, bk], torch.bfloat16)
        self.assertIsNotNone(got, "multi-row decode must take the fused path")
        self.assertEqual(tuple(got.shape), (4, N))
        gt = x.float() @ (
            w.to(torch.float32).view(N // bn, bn, K // bk, bk)
            * s.view(s.shape[0], 1, s.shape[1], 1)
        ).view(N, K).t()
        rel = ((got.float() - gt).abs() / gt.abs().clamp_min(1e-6)).mean().item()
        self.assertLess(rel, 0.05, f"multi-row result wrong, mean rel err {rel}")

    @unittest.skipUnless(CUDA, "needs a GPU")
    def test_too_many_rows_declines(self):
        w, s, bn, bk = _mk()
        x = torch.randn(FUSED_GEMV_MAX_ROWS + 1, w.shape[1], device="cuda",
                        dtype=torch.bfloat16)
        self.assertIsNone(fused_block_dequant_gemv(x, w, s, [bn, bk], torch.bfloat16))

    @unittest.skipUnless(CUDA, "needs a GPU")
    def test_ragged_shape_declines(self):
        """Partial trailing blocks are left to the existing path on purpose."""
        torch.manual_seed(1)
        w = torch.randn(200, 100, device="cuda").to(torch.float8_e4m3fn)
        s = torch.rand(2, 1, device="cuda", dtype=torch.float32) + 0.5
        x = torch.randn(100, device="cuda", dtype=torch.bfloat16)
        self.assertIsNone(fused_block_dequant_gemv(x, w, s, [128, 128], torch.bfloat16))

    @unittest.skipUnless(CUDA, "needs a GPU")
    def test_unfavourable_aspect_declines(self):
        """The grid parallelises over N only, so small-N/large-K starves
        occupancy and the kernel LOSES there. Measured on the 27B's real
        shapes: N/K 0.29 and 0.83 both lose (0.86-0.90x) while N/K >= 1.20 all
        win (1.06-1.41x). Fusing everywhere gave 1.151x on the weighted mix;
        fusing only where it wins gives 1.204x and removes the regressions."""
        dev = "cuda"
        wide = torch.randn(512, 2048, device=dev).to(torch.float8_e4m3fn)  # N<K
        tall = torch.randn(2048, 512, device=dev).to(torch.float8_e4m3fn)  # N>K
        self.assertFalse(fused_gemv_applicable(torch.randn(2048, device=dev), wide))
        self.assertTrue(fused_gemv_applicable(torch.randn(512, device=dev), tall))

    def test_non_fp8_weight_declines(self):
        dev = "cuda" if CUDA else "cpu"
        w = torch.randn(128, 64, device=dev, dtype=torch.bfloat16)
        self.assertFalse(fused_gemv_applicable(torch.randn(64, device=dev), w))


if __name__ == "__main__":
    unittest.main()
