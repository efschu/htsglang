# SPDX-License-Identifier: Apache-2.0
"""#398 numerical gates for the native GGUF MXFP4 (ggml type 39) kernels.

GPU-PENDING at the time of writing: written and importable off-GPU, but every
test skips without CUDA plus a wheel carrying the kernels. Run per arch --
sm86 (RTX 3080) and sm120 (RTX 5090) -- and record both in
``docs/dev/TICKET_398_mxfp4_validation.md``; the two are separate gates because
the MMVQ/MMQ launch shapes differ by device class and because only sm120 has
been exercised by the MXFP4-adjacent marlin path so far.

No downloads: the inputs are synthesized MXFP4 blocks plus the real tensor
fixture next to ``test_gguf_mxfp4_native.py``, so this file runs anywhere the
wheel does.

Reference and tolerance, per kernel:

* **dequantize** -- compared to the python host reference EXACTLY (zero
  tolerance) in fp32. There is no arithmetic in the kernel beyond a table
  lookup and one multiply by a power of two, and
  ``ggml_cuda_e8m0_to_fp32_half`` is bit-identical to the host's, so anything
  but equality is a bug, not rounding. fp16/bf16 outputs are compared to the
  host reference CAST to that dtype, again exactly.
* **MMVQ / MMQ** -- compared to dequantize-then-matmul. These kernels also
  quantize the ACTIVATIONS to q8_1, which the reference does not, so the
  difference is dominated by that. The tolerance is DERIVED from that one
  fact (:func:`activation_quant_sigma`), not inherited: see the long note on
  that function. A tighter check is layered on top: MMVQ vs MMQ against each
  other, both of which DO quantize activations, so they must agree far more
  closely than either agrees with the fp reference.

#511: the tolerances used to be ``atol=1.5, rtol=3e1`` (and ``rtol=1e4`` on
the MMQ arm), inherited from ``sgl-kernel/tests/test_gguf.py`` together with
the claim that these outputs are "centred near zero". They are not: with the
synthetic blocks below the reference has an RMS around 5.1e3 and a maximum
around 2.4e4, so ``atol=1.5`` was four orders of magnitude below the real
error and the gate was ``|a - b| <= 30 * |b|`` alone. Under that predicate an
all-zeros output passes for every input (``|0 - b| = |b| <= 30|b|``), and so
does a sign flip (``|-b - b| = 2|b| <= 30|b|``). The MMVQ-vs-MMQ arm had the
same hole from the other side: its denominator was
``a.abs().max().clamp_min(1e-6)``, so two kernels both returning zero divided
0 by 1e-6 and passed. Every gate in this file is now backed by
:class:`TestToleranceInstrument`, which runs OFF-GPU and shows each one
rejecting a zeroed and a sign-flipped output.
"""

from __future__ import annotations

import unittest

import numpy as np
import torch
from gguf.constants import GGMLQuantizationType as GGMLType
from test_gguf_mxfp4_native import (  # noqa: E402  (same directory)
    BLOCK_SIZE,
    TYPE_SIZE,
    mxfp4_host_dequant,
    synthetic_blocks,
)

from sglang.test.ci.ci_register import register_cpu_ci

# #511: TestToleranceInstrument below is hermetic (no CUDA, no wheel), so this
# file now carries CPU CI weight even though every kernel gate still skips.
register_cpu_ci(est_time=4, suite="base-a-test-cpu")

MXFP4 = int(GGMLType.MXFP4)


def _kernels_available() -> tuple[bool, str]:
    if not torch.cuda.is_available():
        return False, "no CUDA device"
    try:
        import sgl_kernel  # noqa: F401
    except ImportError as exc:
        return False, f"sgl_kernel not importable: {exc}"
    if not hasattr(torch.ops.sgl_kernel, "ggml_mxfp4_native"):
        return False, "installed sgl-kernel wheel predates #398 (no MXFP4 kernels)"
    return True, ""


_AVAILABLE, _WHY = _kernels_available()
requires_kernels = unittest.skipUnless(_AVAILABLE, f"MXFP4 kernels unavailable: {_WHY}")


def make_weight(rows: int, cols: int, seed: int = 398):
    """Return (quantized bytes as a cuda uint8 tensor, fp32 reference matrix).

    ``cols`` must be a multiple of 32. The reference is the HOST decoder, so a
    dequant mismatch is attributable to the kernel and not to a second
    reference implementation.
    """
    assert cols % BLOCK_SIZE == 0
    nblocks = rows * cols // BLOCK_SIZE
    blocks = synthetic_blocks(nblocks, seed=seed)
    ref = mxfp4_host_dequant(blocks).reshape(rows, cols)
    q = torch.from_numpy(blocks.reshape(rows, cols // BLOCK_SIZE * TYPE_SIZE)).cuda()
    return q, torch.from_numpy(ref).cuda()


@requires_kernels
class TestMXFP4Dequant(unittest.TestCase):
    def test_exact_against_the_host_reference_fp32(self):
        from sgl_kernel import ggml_dequantize

        for rows, cols in ((1, 32), (4, 256), (37, 2048), (256, 512)):
            q, ref = make_weight(rows, cols)
            out = ggml_dequantize(q, MXFP4, rows, cols, torch.float32)
            self.assertTrue(
                torch.equal(out, ref),
                f"{rows}x{cols}: max |delta| = " f"{(out - ref).abs().max().item()}",
            )

    def test_exact_against_the_host_reference_half_and_bfloat16(self):
        from sgl_kernel import ggml_dequantize

        for dtype in (torch.float16, torch.bfloat16):
            q, ref = make_weight(8, 1024)
            out = ggml_dequantize(q, MXFP4, 8, 1024, dtype)
            self.assertTrue(torch.equal(out, ref.to(dtype)), str(dtype))

    def test_real_tensor_rows(self):
        """The fixture out of blk.26.ffn_down_exps of the shipped UD-IQ3_XXS."""
        from pathlib import Path

        from sgl_kernel import ggml_dequantize

        rows = np.load(Path(__file__).with_name("mxfp4_real_rows.npy"))  # (2, 64*17)
        ref = torch.from_numpy(
            mxfp4_host_dequant(rows.reshape(-1, TYPE_SIZE)).reshape(2, 2048)
        ).cuda()
        out = ggml_dequantize(
            torch.from_numpy(rows).cuda(), MXFP4, 2, 2048, torch.float32
        )
        self.assertTrue(torch.equal(out, ref))

    def test_shard_that_is_a_multiple_of_32_but_not_of_256(self):
        """The #109-family guard. The launcher rounds up to whole 256-element
        super-blocks; without the per-block guard the tail would read past the
        weight buffer and write past the output. A canary page on both sides
        catches either."""
        from sgl_kernel import ggml_dequantize

        rows, cols = 3, 96  # 9 blocks total, 288 elements -> 2 super-blocks
        q, ref = make_weight(rows, cols)
        out = torch.full((rows, cols), float("nan"), device="cuda")
        got = ggml_dequantize(q, MXFP4, rows, cols, torch.float32, out=out)
        self.assertTrue(torch.equal(got, ref))
        self.assertFalse(torch.isnan(got).any())

    def test_odd_block_offsets_are_read_correctly(self):
        """block_mxfp4 is 17 bytes, so `qs` sits at an odd byte offset in every
        other block. A kernel using a 2-byte-aligned loader either faults or
        (worse) reads shifted data -- this shape puts many odd blocks in one
        row."""
        from sgl_kernel import ggml_dequantize

        q, ref = make_weight(1, 2048, seed=17)
        out = ggml_dequantize(q, MXFP4, 1, 2048, torch.float32)
        self.assertTrue(torch.equal(out, ref))


# ---------------------------------------------------------------------------
# Derived tolerance (#511)
# ---------------------------------------------------------------------------

#: q8_1 block length, as ggml defines it (QK8_1).
Q8_1_BLOCK = 32

#: How many predicted sigmas the gate allows. The error is a sum of ~cols
#: independent rounding terms, so it is Gaussian per output element and the
#: maximum over N elements sits near ``sqrt(2 ln N)`` sigmas: measured
#: max/sigma ratios over 8 seeds on the shapes this file uses are 2.97 (N=128),
#: 3.80 (512), 3.82 (1024/1152), 4.20 (1600), 4.32 (8192), against predicted
#: 3.11 / 3.53 / 3.72 / 3.84 / 4.25. Eight leaves ~1.85x headroom over the
#: worst observed case and a per-run false-failure probability below 1e-11,
#: while still being roughly three orders of magnitude tighter than the
#: rtol=3e1 predicate it replaces.
SIGMA_MULTIPLIER = 8.0


def q8_1_step(x: torch.Tensor, block: int = Q8_1_BLOCK) -> torch.Tensor:
    """Per-block q8_1 quantisation step ``d = max|x| / 127``.

    Shape ``(m, cols // block)``. This is ggml's own definition; the kernels
    call ``quantize_row_q8_1_cuda`` which computes exactly this.
    """
    m, k = x.shape
    d = x.reshape(m, k // block, block).abs().amax(dim=-1) / 127.0
    return torch.where(d == 0, torch.ones_like(d), d)


def activation_quant_sigma(
    x: torch.Tensor, w: torch.Tensor, block: int = Q8_1_BLOCK
) -> torch.Tensor:
    """Predicted std of ``kernel_out - (x @ w.T)``, per output element.

    Where the number comes from. The kernels compute ``sum_k xq_k * w_k`` with
    ``xq`` the q8_1 reconstruction of ``x``; the weights are NOT re-quantised
    (they are already the exact 4-bit lattice, and the dequant gate above pins
    that at zero tolerance), and the accumulation is fp32. So the whole
    difference against ``x @ w.T`` is the activation rounding

        xq_k = x_k + e_k,   e_k ~ U(-d_b/2, +d_b/2),  Var(e_k) = d_b^2 / 12

    with ``d_b`` the step of the q8_1 block that element k falls in. The
    rounding errors are independent of the weights, so for output ``(i, j)``

        Var(err_ij) = sum_k d_b(k)^2 / 12 * w_jk^2

    which is what this returns the square root of. Nothing here is fitted:
    the only inputs are ggml's q8_1 step definition and the weights.

    fp32 accumulation contributes about ``sqrt(cols) * 6e-8 * |out|``, which at
    the magnitudes this file works with is ~1e-2 against a sigma of ~6e1 --
    four orders below, hence not modelled.
    """
    m, k = x.shape
    n = w.shape[0]
    d = q8_1_step(x, block)  # (m, kb)
    w2 = w.reshape(n, k // block, block).pow(2).sum(-1)  # (n, kb)
    return ((d.pow(2) / 12.0) @ w2.t()).sqrt()  # (m, n)


def matmul_atol(
    x: torch.Tensor, w: torch.Tensor, multiplier: float = SIGMA_MULTIPLIER
) -> float:
    """Absolute tolerance for a q8_1-activation matmul against the fp reference.

    ABSOLUTE on purpose, with ``rtol=0``. The error is a property of the
    inputs, not of the individual output element: an output that happens to
    land near zero still carries the full ~sigma of activation noise, so a
    relative tolerance either has to be enormous (which is how ``rtol=3e1``
    came to swallow zeros and sign flips) or it rejects the near-zero outputs.
    """
    return float(activation_quant_sigma(x, w).max().item() * multiplier)


def assert_matmul_close(
    case: unittest.TestCase,
    out: torch.Tensor,
    ref: torch.Tensor,
    x: torch.Tensor,
    w: torch.Tensor,
    *,
    msg: str = "",
) -> None:
    """``out`` must match ``ref`` within the derived tolerance."""
    atol = matmul_atol(x, w)
    torch.testing.assert_close(
        out, ref, atol=atol, rtol=0.0, msg=lambda s: f"{msg}{': ' if msg else ''}{s}"
    )


def assert_kernels_agree(
    case: unittest.TestCase,
    a: torch.Tensor,
    b: torch.Tensor,
    ref: torch.Tensor,
    *,
    rel: float = 1e-3,
) -> None:
    """Two activation-quantising kernels must agree far more closely than
    either agrees with the fp reference.

    #511: the denominator is the REFERENCE scale, not ``a``'s own, and both
    arms have to carry that scale before their agreement counts. The previous
    form divided by ``a.abs().max().clamp_min(1e-6)``, under which two kernels
    that both return zero are in perfect agreement -- the instrument could not
    discriminate a working pair from a pair of dead ones. A spread
    precondition on the reference plus a magnitude precondition on each arm is
    what makes "they agree" evidence of anything.
    """
    scale = float(ref.abs().max().item())
    case.assertGreater(scale, 0.0, "reference has no spread; the gate is blind")
    for name, t in (("mmvq", a), ("mmq", b)):
        case.assertGreater(
            float(t.abs().max().item()),
            0.5 * scale,
            f"{name} output does not carry the reference's scale "
            f"(max |{name}| = {t.abs().max().item():.4g} vs reference "
            f"{scale:.4g}); a collapsed kernel is not an agreeing kernel",
        )
    case.assertLess(float((a - b).abs().max().item()) / scale, rel)


@requires_kernels
class TestMXFP4MMVQ(unittest.TestCase):
    """Decode path: dense MMVQ (ggml_mul_mat_vec_a8)."""

    def _run(self, m: int, rows: int, cols: int, dtype=torch.bfloat16):
        from sgl_kernel import ggml_mul_mat_vec_a8

        q, ref_w = make_weight(rows, cols)
        # Sample on CPU: on-GPU randn is not arch-identical (sm86 != sm120).
        x = torch.randn(m, cols, dtype=torch.float32).to(dtype).cuda()
        ref = x.float() @ ref_w.t()
        out = ggml_mul_mat_vec_a8(q, x, MXFP4, rows).float()
        return out, ref, x.float(), ref_w

    def test_m1_matches_the_dequant_reference(self):
        out, ref, x, w = self._run(1, 128, 1024)
        assert_matmul_close(self, out, ref, x, w)

    def test_batched_columns_up_to_eight(self):
        """The ncols_dst<=8 batched dispatch -- the spec-decode verify range."""
        for m in (1, 2, 4, 8):
            out, ref, x, w = self._run(m, 128, 1024)
            assert_matmul_close(self, out, ref, x, w, msg=f"m={m}")

    def test_above_the_batched_range_falls_to_the_legacy_kernel(self):
        out, ref, x, w = self._run(9, 128, 1024)
        assert_matmul_close(self, out, ref, x, w)


@requires_kernels
class TestMXFP4MMQ(unittest.TestCase):
    """Prefill / large-M path: dense MMQ (ggml_mul_mat_a8)."""

    def test_matches_the_dequant_reference(self):
        from sgl_kernel import ggml_mul_mat_a8

        rows, cols = 128, 1024
        q, ref_w = make_weight(rows, cols)
        x = torch.randn(64, cols, dtype=torch.float32).to(torch.bfloat16).cuda()
        ref = x.float() @ ref_w.t()
        out = ggml_mul_mat_a8(q, x, MXFP4, rows).float()
        assert_matmul_close(self, out, ref, x.float(), ref_w)

    def test_agrees_with_mmvq_far_more_closely_than_with_the_fp_reference(self):
        """Both kernels quantize the activations identically, so their
        remaining difference is only reduction order -- the ~1e-4 relmax class
        documented for the MMVQ<->MMQ dispatch, not the ~5e-3 either has
        against fp32."""
        from sgl_kernel import ggml_mul_mat_a8, ggml_mul_mat_vec_a8

        rows, cols = 128, 1024
        q, ref_w = make_weight(rows, cols)
        x = torch.randn(4, cols, dtype=torch.float32).to(torch.bfloat16).cuda()
        ref = x.float() @ ref_w.t()
        a = ggml_mul_mat_vec_a8(q, x, MXFP4, rows).float()
        b = ggml_mul_mat_a8(q, x, MXFP4, rows).float()
        # Each arm against the independent reference first: two kernels that
        # are BOTH wrong the same way (a shared sign flip, say) agree with each
        # other perfectly and would otherwise pass.
        assert_matmul_close(self, a, ref, x.float(), ref_w, msg="mmvq")
        assert_matmul_close(self, b, ref, x.float(), ref_w, msg="mmq")
        assert_kernels_agree(self, a, b, ref)

    def test_need_check_path_rows_not_a_multiple_of_the_tile(self):
        """MMQ_Y_MXFP4 is 32 on CUDA; a row count that is not a multiple of it
        takes the need_check=true instantiation."""
        from sgl_kernel import ggml_mul_mat_a8

        rows, cols = 100, 512
        q, ref_w = make_weight(rows, cols)
        x = torch.randn(16, cols, dtype=torch.float32).to(torch.bfloat16).cuda()
        ref = x.float() @ ref_w.t()
        out = ggml_mul_mat_a8(q, x, MXFP4, rows).float()
        assert_matmul_close(self, out, ref, x.float(), ref_w)


@requires_kernels
class TestMXFP4MoE(unittest.TestCase):
    """Expert kernels: MMVQ (decode) and MMQ (prefill) over stacked experts."""

    def _experts(self, e: int, rows: int, cols: int):
        q, ref = make_weight(e * rows, cols)
        return (
            q.reshape(e, rows, cols // BLOCK_SIZE * TYPE_SIZE),
            ref.reshape(e, rows, cols),
        )

    def test_moe_vec_matches_a_per_expert_reference(self):
        from sgl_kernel import ggml_moe_a8_vec

        e, rows, cols, tokens, top_k = 4, 64, 512, 3, 2
        w, ref_w = self._experts(e, rows, cols)
        x = torch.randn(tokens, cols, dtype=torch.float32).to(torch.bfloat16).cuda()
        topk_ids = torch.tensor(
            [[0, 1], [2, 3], [1, 3]], dtype=torch.int32, device="cuda"
        )
        out = ggml_moe_a8_vec(x, w, topk_ids, top_k, MXFP4, rows, tokens).float()
        ref = torch.stack(
            [
                x[t].float() @ ref_w[int(topk_ids[t, k])].t()
                for t in range(tokens)
                for k in range(top_k)
            ]
        )
        # Same derivation, over the flattened (token, slot) rows: each row is
        # one token's activations against one expert's weights, so the widest
        # per-expert sigma bounds them all.
        assert_matmul_close(
            self, out, ref, x.float(), ref_w.reshape(-1, ref_w.shape[-1])
        )

    def test_moe_block_size_is_registered(self):
        """#81 family: an unregistered type returns 0 and the MMQ MoE path
        would silently pick a zero tile."""
        from sgl_kernel import ggml_moe_get_block_size

        self.assertGreater(ggml_moe_get_block_size(MXFP4), 0)

    def test_moe_mmq_matches_a_per_expert_reference(self):
        from sgl_kernel import ggml_moe_a8, ggml_moe_get_block_size

        from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
            moe_align_block_size,
        )

        e, rows, cols, tokens, top_k = 4, 64, 512, 96, 2
        w, ref_w = self._experts(e, rows, cols)
        x = torch.randn(tokens, cols, dtype=torch.float32).to(torch.bfloat16).cuda()
        topk_ids = torch.randint(0, e, (tokens, top_k), dtype=torch.int32).cuda()
        block = ggml_moe_get_block_size(MXFP4)
        sorted_ids, expert_ids, npad = moe_align_block_size(topk_ids, block, e)
        expert_ids = expert_ids.masked_fill(expert_ids >= e, -1)
        out = ggml_moe_a8(
            x, w, sorted_ids, expert_ids, npad, MXFP4, rows, top_k, tokens
        ).float()
        ref = torch.stack(
            [
                x[t].float() @ ref_w[int(topk_ids[t, k])].t()
                for t in range(tokens)
                for k in range(top_k)
            ]
        )
        assert_matmul_close(
            self, out, ref, x.float(), ref_w.reshape(-1, ref_w.shape[-1])
        )


# ---------------------------------------------------------------------------
# The instrument itself (#511) -- runs OFF-GPU
# ---------------------------------------------------------------------------


def _cpu_weight(rows: int, cols: int, seed: int = 398) -> torch.Tensor:
    """``make_weight``'s reference half, without the ``.cuda()``."""
    blocks = synthetic_blocks(rows * cols // BLOCK_SIZE, seed=seed)
    return torch.from_numpy(mxfp4_host_dequant(blocks).reshape(rows, cols))


def _simulated_kernel(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """What a CORRECT q8_1-activation kernel produces, computed on the host.

    Not a second reference for the kernels -- the CUDA gates above still
    compare against the fp32 reference. This exists so the tolerance can be
    shown, here and now without a GPU, to accept a correct implementation.
    """
    m, k = x.shape
    d = q8_1_step(x).unsqueeze(-1)
    xb = x.reshape(m, k // Q8_1_BLOCK, Q8_1_BLOCK)
    xq = (torch.clamp(torch.round(xb / d), -127, 127) * d).reshape(m, k)
    return xq @ w.t()


class TestToleranceInstrument(unittest.TestCase):
    """Can-discriminate checks for every gate this file applies.

    CLAUDE.md: "an INSTRUMENT's verdict counts only after the instrument
    passes a can-discriminate check on known-different inputs". The CUDA
    tests above cannot run without a wheel and a card; these run anywhere and
    prove that what they assert is capable of failing.
    """

    ROWS, COLS = 128, 1024

    def _case(self, m: int = 4, seed: int = 0):
        w = _cpu_weight(self.ROWS, self.COLS)
        g = torch.Generator().manual_seed(seed)
        x = torch.randn(m, self.COLS, generator=g).to(torch.bfloat16).float()
        return x, w, x @ w.t()

    # -- the gate accepts a correct kernel ---------------------------------

    def test_a_correct_kernel_passes_at_every_shape_the_file_uses(self):
        """The tolerance must not redden the arms that already pass on GPU."""
        w = _cpu_weight(self.ROWS, self.COLS)
        for m in (1, 2, 4, 8, 9, 64):
            for seed in range(4):
                g = torch.Generator().manual_seed(seed)
                x = torch.randn(m, self.COLS, generator=g).to(torch.bfloat16).float()
                ref = x @ w.t()
                with self.subTest(m=m, seed=seed):
                    assert_matmul_close(self, _simulated_kernel(x, w), ref, x, w)

    def test_the_margin_over_a_correct_kernel_is_bounded_and_reported(self):
        """Headroom, as a number: the gate should be comfortably above the
        real error and far below the signal, or it is either flaky or blind."""
        x, w, ref = self._case(m=64)
        atol = matmul_atol(x, w)
        err = (_simulated_kernel(x, w) - ref).abs().max().item()
        self.assertGreater(atol, 1.5 * err, f"atol {atol:.4g} vs error {err:.4g}")
        self.assertLess(atol, 0.1 * ref.abs().max().item())

    # -- the gate rejects the two injected failures ------------------------

    def test_an_all_zeros_output_is_rejected(self):
        x, w, ref = self._case()
        with self.assertRaises(AssertionError):
            assert_matmul_close(self, torch.zeros_like(ref), ref, x, w)

    def test_a_sign_flipped_output_is_rejected(self):
        x, w, ref = self._case()
        with self.assertRaises(AssertionError):
            assert_matmul_close(self, -_simulated_kernel(x, w), ref, x, w)

    def test_the_old_predicate_accepted_both_and_that_is_why_it_changed(self):
        """The refuted baseline, executed rather than asserted about."""
        _x, _w, ref = self._case()
        for broken in (torch.zeros_like(ref), -ref):
            # atol=1.5, rtol=3e1 -- the pre-#511 gate.
            torch.testing.assert_close(broken, ref, atol=1.5, rtol=3e1)

    # -- the agreement gate ------------------------------------------------

    def test_two_agreeing_kernels_pass(self):
        x, w, ref = self._case()
        a = _simulated_kernel(x, w)
        b = a + torch.randn_like(a) * (a.abs().max() * 1e-5)
        assert_kernels_agree(self, a, b, ref)

    def test_two_kernels_that_both_return_zero_are_rejected(self):
        """The clamp_min(1e-6) hole, closed and shown closed."""
        _x, _w, ref = self._case()
        zero = torch.zeros_like(ref)
        with self.assertRaises(AssertionError):
            assert_kernels_agree(self, zero, zero, ref)

    def test_the_old_agreement_predicate_accepted_two_dead_kernels(self):
        _x, _w, ref = self._case()
        a = b = torch.zeros_like(ref)
        denom = a.abs().max().clamp_min(1e-6)
        self.assertLess(((a - b).abs().max() / denom).item(), 1e-3)

    def test_one_collapsed_kernel_is_rejected(self):
        x, w, ref = self._case()
        a = _simulated_kernel(x, w)
        with self.assertRaises(AssertionError):
            assert_kernels_agree(self, a, torch.zeros_like(a), ref)

    def test_a_shared_sign_flip_is_caught_by_the_reference_arms(self):
        """Two kernels wrong the SAME way agree perfectly; only the check
        against the independent reference sees it."""
        x, w, ref = self._case()
        a = b = -_simulated_kernel(x, w)
        assert_kernels_agree(self, a, b, ref)  # they do agree
        with self.assertRaises(AssertionError):  # and are still wrong
            assert_matmul_close(self, a, ref, x, w)

    # -- the derivation itself ---------------------------------------------

    def test_the_predicted_sigma_matches_the_observed_error_spread(self):
        """The model is not fitted: check it against the realised errors."""
        w = _cpu_weight(self.ROWS, self.COLS)
        worst = 0.0
        for seed in range(8):
            g = torch.Generator().manual_seed(seed)
            x = torch.randn(64, self.COLS, generator=g).to(torch.bfloat16).float()
            err = (_simulated_kernel(x, w) - x @ w.t()).abs()
            worst = max(worst, (err / activation_quant_sigma(x, w)).max().item())
        # Gaussian extreme value over N = 64*128 = 8192 elements is ~4.25
        # sigma; anything far above would mean the model misses a term.
        self.assertLess(worst, 6.0, f"worst |err|/sigma = {worst:.3f}")
        self.assertLess(worst, SIGMA_MULTIPLIER)

    def test_the_new_gate_is_orders_of_magnitude_tighter_than_the_old_one(self):
        x, w, ref = self._case()
        old = 1.5 + 3e1 * ref.abs().max().item()
        self.assertLess(matmul_atol(x, w) * 100, old)


if __name__ == "__main__":
    unittest.main()
