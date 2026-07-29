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

import os
import unittest

import torch

from sglang.srt.layers.quantization.fp8_dequant_gemv import (
    FUSED_GEMV_MAX_ROWS,
    fused_block_dequant_gemv,
    fused_channel_dequant_gemv,
    fused_channel_gemv_applicable,
    fused_gemv_applicable,
    fused_gemv_enabled,
)
from sglang.srt.layers.quantization.fp8_utils import (
    dequant_fp8_block_weight,
    dequant_fp8_weight,
)

CUDA = torch.cuda.is_available()


def _mk(N=512, K=256, bn=128, bk=128, seed=0, dev="cuda"):
    torch.manual_seed(seed)
    w = torch.randn(N, K, device=dev).to(torch.float8_e4m3fn)
    s = torch.rand(N // bn, K // bk, device=dev, dtype=torch.float32) + 0.5
    return w, s, bn, bk


def _mk_channel(N=512, K=256, seed=0, dev="cuda"):
    """Per-channel layout: one fp32 scale per OUTPUT row, shape (N, 1).

    That is exactly what compressed-tensors ``strategy: channel`` stores and
    what ``convert_to_channelwise`` produces from a per-tensor scale, so both
    wired call sites see this shape.
    """
    torch.manual_seed(seed)
    w = torch.randn(N, K, device=dev).to(torch.float8_e4m3fn)
    s = torch.rand(N, 1, device=dev, dtype=torch.float32) + 0.5
    return w, s


def _channel_gt(w, s, x):
    """fp32 ground truth, computed without either implementation."""
    N = w.shape[0]
    wd = w.to(torch.float32) * s.reshape(N, 1).to(torch.float32)
    return x.float() @ wd.t()


def _relerr(got, gt):
    return ((got.float() - gt).abs() / gt.abs().clamp_min(1e-6)).mean().item()


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

        gt = (
            w.to(torch.float32).view(N // bn, bn, K // bk, bk)
            * s.view(s.shape[0], 1, s.shape[1], 1)
        ).view(N, K) @ x.float()

        fused = fused_block_dequant_gemv(x, w, s, [bn, bk], torch.bfloat16)
        self.assertIsNotNone(fused, "fused path should apply for a 1-row input")
        mat = torch.nn.functional.linear(
            x, dequant_fp8_block_weight(w, s, [bn, bk], torch.bfloat16)
        )

        rel = (
            lambda v: ((v.float() - gt).abs() / gt.abs().clamp_min(1e-6)).mean().item()
        )
        r_fused, r_mat = rel(fused), rel(mat)
        # The band: the fused path must be no worse than what it replaces.
        self.assertLessEqual(
            r_fused,
            r_mat * 1.1,
            f"fused rel err {r_fused:.5f} worse than materialise {r_mat:.5f}",
        )
        self.assertLess(r_fused, 0.05, "fused path is not merely worse, it is wrong")

    def test_fp16_path_also_within_band(self):
        w, s, bn, bk = _mk(seed=3)
        N, K = w.shape
        x = torch.randn(K, device="cuda", dtype=torch.float16)
        gt = (
            w.to(torch.float32).view(N // bn, bn, K // bk, bk)
            * s.view(s.shape[0], 1, s.shape[1], 1)
        ).view(N, K) @ x.float()
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
        gt = (
            x.float()
            @ (
                w.to(torch.float32).view(N // bn, bn, K // bk, bk)
                * s.view(s.shape[0], 1, s.shape[1], 1)
            )
            .view(N, K)
            .t()
        )
        rel = ((got.float() - gt).abs() / gt.abs().clamp_min(1e-6)).mean().item()
        self.assertLess(rel, 0.05, f"multi-row result wrong, mean rel err {rel}")

    @unittest.skipUnless(CUDA, "needs a GPU")
    def test_too_many_rows_declines(self):
        w, s, bn, bk = _mk()
        x = torch.randn(
            FUSED_GEMV_MAX_ROWS + 1, w.shape[1], device="cuda", dtype=torch.bfloat16
        )
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


@unittest.skipUnless(CUDA, "fused GEMV needs a GPU")
class TestChannelAccuracyBand(unittest.TestCase):
    """Per-channel variant, same gate philosophy as the block variant.

    NOT equality against the old path: the fused kernel accumulates in fp32 and
    applies the scale once at the end, where materialisation rounds the whole
    weight to bf16/fp16 before the multiply. The fused path is therefore the
    more accurate of the two and must not be pinned to the other's error.
    """

    def test_fused_is_at_least_as_accurate_as_materialise(self):
        w, s = _mk_channel()
        x = torch.randn(w.shape[1], device="cuda", dtype=torch.bfloat16)
        gt = _channel_gt(w, s, x)

        fused = fused_channel_dequant_gemv(x, w, s, torch.bfloat16)
        self.assertIsNotNone(fused, "fused path should apply for a 1-row input")
        mat = torch.nn.functional.linear(x, dequant_fp8_weight(w, s, torch.bfloat16))

        r_fused, r_mat = _relerr(fused, gt), _relerr(mat, gt)
        self.assertLessEqual(
            r_fused,
            r_mat * 1.1,
            f"fused rel err {r_fused:.5f} worse than materialise {r_mat:.5f}",
        )
        self.assertLess(r_fused, 0.05, "fused path is not merely worse, it is wrong")

    def test_fp16_path_also_within_band(self):
        """fp16 is the compute dtype on the cards this exists for -- sm75 and
        gfx900 have no hardware bf16 -- so it is not an afterthought here."""
        w, s = _mk_channel(seed=3)
        x = torch.randn(w.shape[1], device="cuda", dtype=torch.float16)
        gt = _channel_gt(w, s, x)
        fused = fused_channel_dequant_gemv(x, w, s, torch.float16)
        self.assertIsNotNone(fused)
        self.assertLess(_relerr(fused, gt), 0.05)

    def test_multi_row_is_handled_not_declined(self):
        """Speculative decode runs several draft rows per step. A 1-row-only
        kernel declines exactly when it is most needed -- that mistake was made
        once on the block variant and cost the whole end-to-end gain."""
        w, s = _mk_channel(seed=5)
        N, K = w.shape
        x = torch.randn(4, K, device="cuda", dtype=torch.bfloat16)
        got = fused_channel_dequant_gemv(x, w, s, torch.bfloat16)
        self.assertIsNotNone(got, "multi-row decode must take the fused path")
        self.assertEqual(tuple(got.shape), (4, N))
        self.assertLess(_relerr(got, _channel_gt(w, s, x)), 0.05)


@unittest.skipUnless(CUDA, "fused GEMV needs a GPU")
class TestChannelLayoutsAndShapes(unittest.TestCase):
    """The two call sites hand in two different memory layouts and two scale
    shapes. Both must produce the same numbers, or the kernel is only correct
    for whichever one happened to be tested."""

    def test_transposed_weight_view_matches_row_major(self):
        """Fp8LinearMethod stores the weight as (in, out) for F.linear and
        passes ``.t()``; compressed-tensors W8A16 keeps (out, in). Same result,
        or one of the two call sites is silently wrong."""
        w, s = _mk_channel(seed=7)
        K = w.shape[1]
        x = torch.randn(2, K, device="cuda", dtype=torch.bfloat16)

        row_major = fused_channel_dequant_gemv(x, w, s, torch.bfloat16)
        # Same values, stored (K, N) row-major, handed in as a transposed view.
        w_t = w.t().contiguous().t()
        self.assertFalse(w_t.is_contiguous())
        transposed = fused_channel_dequant_gemv(x, w_t, s, torch.bfloat16)

        self.assertIsNotNone(transposed, "the .t() view layout must be supported")
        self.assertTrue(
            torch.equal(row_major, transposed),
            "the two wired layouts disagree -- one call site is wrong",
        )

    def test_scale_accepted_as_flat_and_column(self):
        w, s = _mk_channel(seed=9)
        x = torch.randn(w.shape[1], device="cuda", dtype=torch.bfloat16)
        a = fused_channel_dequant_gemv(x, w, s, torch.bfloat16)
        b = fused_channel_dequant_gemv(x, w, s.reshape(-1), torch.bfloat16)
        self.assertIsNotNone(b)
        self.assertTrue(torch.equal(a, b))

    def test_unexpected_scale_shape_declines(self):
        """A block or per-tensor scale must be handed back, not reinterpreted.
        Declining is free; guessing produces plausible wrong numbers."""
        w, _ = _mk_channel(seed=11)
        N, K = w.shape
        x = torch.randn(K, device="cuda", dtype=torch.bfloat16)
        blockish = torch.rand(N // 128, K // 128, device="cuda", dtype=torch.float32)
        self.assertIsNone(fused_channel_dequant_gemv(x, w, blockish, torch.bfloat16))
        scalar = torch.ones((), device="cuda", dtype=torch.float32)
        self.assertIsNone(fused_channel_dequant_gemv(x, w, scalar, torch.bfloat16))

    def test_non_divisible_shapes_are_computed_not_declined(self):
        """Unlike the block variant there is no block geometry to be ragged
        about, so a shape the block kernel hands back is computed here. The
        kernel masks both N and K; only the masking makes that true, so it is
        tested rather than asserted."""
        torch.manual_seed(13)
        w = torch.randn(200, 100, device="cuda").to(torch.float8_e4m3fn)
        s = torch.rand(200, 1, device="cuda", dtype=torch.float32) + 0.5
        x = torch.randn(100, device="cuda", dtype=torch.bfloat16)
        got = fused_channel_dequant_gemv(x, w, s, torch.bfloat16)
        self.assertIsNotNone(got, "per-channel has no divisibility requirement")
        self.assertEqual(tuple(got.shape), (200,))
        self.assertLess(_relerr(got, _channel_gt(w, s, x)), 0.05)

    def test_wide_weight_is_dispatched_unlike_the_block_variant(self):
        """The block variant declines N < K; the per-channel variant must NOT.

        Inheriting that guard was the obvious move and it is measured to be wrong:
        on the real 4B mix every shape wins, worst case 1.22x, and keeping the
        guard costs 4.52x -> 2.18x time-weighted on a 3080. See
        fused_channel_gemv_applicable for the table. This test pins the DIFFERENCE
        between the two predicates so a later tidy-up cannot quietly re-unify them.
        """
        dev = "cuda"
        wide = torch.randn(512, 2048, device=dev).to(torch.float8_e4m3fn)  # N < K
        x = torch.randn(2048, device=dev, dtype=torch.bfloat16)
        self.assertFalse(fused_gemv_applicable(x, wide), "block variant declines N<K")
        self.assertTrue(
            fused_channel_gemv_applicable(x, wide),
            "per-channel must take wide weights -- it wins on them",
        )
        s = torch.rand(512, 1, device=dev, dtype=torch.float32) + 0.5
        got = fused_channel_dequant_gemv(x, wide, s, torch.bfloat16)
        self.assertIsNotNone(got)
        self.assertLess(_relerr(got, _channel_gt(wide, s, x)), 0.05)

    def test_too_many_rows_declines(self):
        w, s = _mk_channel(seed=15)
        x = torch.randn(
            FUSED_GEMV_MAX_ROWS + 1, w.shape[1], device="cuda", dtype=torch.bfloat16
        )
        self.assertIsNone(fused_channel_dequant_gemv(x, w, s, torch.bfloat16))


@unittest.skipUnless(CUDA, "fused GEMV needs a GPU")
class TestChannelPurity(unittest.TestCase):
    """The kernel must be a pure function of its inputs and must not touch them.

    Same property the dequant cache rests on, checked the same way: falsified
    rather than assumed, because a kernel that mutates a weight or drifts
    between calls corrupts a model slowly instead of failing loudly.
    """

    def test_repeated_calls_are_bit_identical(self):
        w, s = _mk_channel(seed=17)
        x = torch.randn(3, w.shape[1], device="cuda", dtype=torch.bfloat16)
        a = fused_channel_dequant_gemv(x, w, s, torch.bfloat16)
        # Interleave unrelated GPU work and a call at another dtype.
        torch.randn(1024, 1024, device="cuda") @ torch.randn(1024, 1024, device="cuda")
        fused_channel_dequant_gemv(x.half(), w, s, torch.float16)
        b = fused_channel_dequant_gemv(x, w, s, torch.bfloat16)
        self.assertTrue(torch.equal(a, b))

    def test_inputs_are_not_mutated(self):
        w, s = _mk_channel(seed=19)
        x = torch.randn(2, w.shape[1], device="cuda", dtype=torch.bfloat16)
        w_before = w.view(torch.uint8).clone()
        s_before = s.clone()
        x_before = x.clone()
        fused_channel_dequant_gemv(x, w, s, torch.bfloat16)
        self.assertTrue(torch.equal(w.view(torch.uint8), w_before))
        self.assertTrue(torch.equal(s, s_before))
        self.assertTrue(torch.equal(x, x_before))


class TestFusedGemvEnvSwitch(unittest.TestCase):
    """``SGLANG_FP8_FUSED_GEMV=0`` must switch off BOTH variants.

    It exists for same-session A/B measurement; a flag that covered only one
    variant would compare two different things on two different checkpoints.
    """

    def setUp(self):
        self._saved = os.environ.get("SGLANG_FP8_FUSED_GEMV")
        fused_gemv_enabled.cache_clear()

    def tearDown(self):
        if self._saved is None:
            os.environ.pop("SGLANG_FP8_FUSED_GEMV", None)
        else:
            os.environ["SGLANG_FP8_FUSED_GEMV"] = self._saved
        fused_gemv_enabled.cache_clear()

    def test_default_is_on(self):
        os.environ.pop("SGLANG_FP8_FUSED_GEMV", None)
        fused_gemv_enabled.cache_clear()
        self.assertTrue(fused_gemv_enabled())

    def test_falsy_spellings_all_disable(self):
        for v in ("0", "false", "False", "off", "no"):
            os.environ["SGLANG_FP8_FUSED_GEMV"] = v
            fused_gemv_enabled.cache_clear()
            self.assertFalse(fused_gemv_enabled(), f"{v!r} should disable")

    @unittest.skipUnless(CUDA, "needs a GPU")
    def test_zero_disables_both_variants(self):
        os.environ["SGLANG_FP8_FUSED_GEMV"] = "0"
        fused_gemv_enabled.cache_clear()

        wb, sb, bn, bk = _mk()
        xb = torch.randn(wb.shape[1], device="cuda", dtype=torch.bfloat16)
        self.assertFalse(fused_gemv_applicable(xb, wb))
        self.assertIsNone(
            fused_block_dequant_gemv(xb, wb, sb, [bn, bk], torch.bfloat16)
        )

        wc, sc = _mk_channel(seed=21)
        xc = torch.randn(wc.shape[1], device="cuda", dtype=torch.bfloat16)
        self.assertFalse(fused_channel_gemv_applicable(xc, wc))
        self.assertIsNone(fused_channel_dequant_gemv(xc, wc, sc, torch.bfloat16))

    @unittest.skipUnless(CUDA, "needs a GPU")
    def test_default_leaves_both_variants_live(self):
        os.environ.pop("SGLANG_FP8_FUSED_GEMV", None)
        fused_gemv_enabled.cache_clear()
        wb, sb, bn, bk = _mk()
        xb = torch.randn(wb.shape[1], device="cuda", dtype=torch.bfloat16)
        self.assertIsNotNone(
            fused_block_dequant_gemv(xb, wb, sb, [bn, bk], torch.bfloat16)
        )
        wc, sc = _mk_channel(seed=23)
        xc = torch.randn(wc.shape[1], device="cuda", dtype=torch.bfloat16)
        self.assertIsNotNone(fused_channel_dequant_gemv(xc, wc, sc, torch.bfloat16))


if __name__ == "__main__":
    unittest.main()


class TestRowGateCoversADraftBlock(unittest.TestCase):
    """#274 round 7c: the row gate has to admit a whole DFLASH draft block.

    A DFLASH drafter does not propose one token per round, it proposes a BLOCK
    -- 16 rows for every released Qwen3.6 DFLASH export. With the gate at 8,
    every one of the drafter's decode rounds fell through to materialise+GEMM,
    so a measurement of "DFLASH on an fp8 card" would have measured the
    fallback rather than the kernel. The gate is still a DECODE gate: 16 rows
    is one request's draft block, not a serving batch.
    """

    def test_the_gate_admits_a_full_draft_block(self):
        from sglang.srt.models.dflash import DEFAULT_DFLASH_BLOCK_SIZE

        self.assertGreaterEqual(
            FUSED_GEMV_MAX_ROWS,
            DEFAULT_DFLASH_BLOCK_SIZE,
            "the fused fp8 GEMV declines a whole DFLASH draft block; every "
            "draft round would silently take the materialise+GEMM fallback",
        )

    def test_the_kernel_tile_covers_the_gate(self):
        """BLOCK_M is the tl.dot minimum, and it must not be exceeded.

        The gate and the kernel's row tile are two numbers that have to move
        together: a gate above BLOCK_M would hand the kernel more rows than one
        tile covers, which is a correctness question and not a performance one.
        """
        import inspect

        import sglang.srt.layers.quantization.fp8_dequant_gemv as mod

        src = inspect.getsource(mod)
        block_ms = {
            int(line.split("=")[1].split("#")[0].strip())
            for line in src.splitlines()
            if line.strip().startswith("BLOCK_M =")
        }
        self.assertTrue(block_ms, "no BLOCK_M found to check the gate against")
        for bm in block_ms:
            self.assertLessEqual(FUSED_GEMV_MAX_ROWS, bm)

    @unittest.skipUnless(CUDA, "needs a GPU")
    def test_a_draft_block_dispatches_to_the_fused_path(self):
        from sglang.srt.models.dflash import DEFAULT_DFLASH_BLOCK_SIZE

        w, _s, K = _mk()[0], None, 256
        w = torch.randn(512, K, device="cuda").to(torch.float8_e4m3fn)
        if not fused_gemv_enabled():
            self.skipTest("fused path disabled in this environment")
        self.assertTrue(
            fused_gemv_applicable(
                torch.randn(DEFAULT_DFLASH_BLOCK_SIZE, K, device="cuda"), w
            )
        )
        self.assertFalse(
            fused_gemv_applicable(
                torch.randn(FUSED_GEMV_MAX_ROWS + 1, K, device="cuda"), w
            )
        )
