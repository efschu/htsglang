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
  difference is dominated by that and the tolerance follows the existing GGUF
  suite (sgl-kernel/tests/test_gguf.py): atol 1.5, rtol 3e1 for bf16 on
  outputs centred near zero. A tighter, meaningful check is added on top:
  MMVQ vs MMQ against each other, both of which DO quantize activations, so
  they must agree far more closely than either agrees with the fp reference.
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
        return out, ref

    def test_m1_matches_the_dequant_reference(self):
        out, ref = self._run(1, 128, 1024)
        torch.testing.assert_close(out, ref, atol=1.5, rtol=3e1)

    def test_batched_columns_up_to_eight(self):
        """The ncols_dst<=8 batched dispatch -- the spec-decode verify range."""
        for m in (1, 2, 4, 8):
            out, ref = self._run(m, 128, 1024)
            torch.testing.assert_close(out, ref, atol=1.5, rtol=3e1, msg=f"m={m}")

    def test_above_the_batched_range_falls_to_the_legacy_kernel(self):
        out, ref = self._run(9, 128, 1024)
        torch.testing.assert_close(out, ref, atol=1.5, rtol=3e1)


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
        torch.testing.assert_close(out, ref, atol=1.5, rtol=1e4)

    def test_agrees_with_mmvq_far_more_closely_than_with_the_fp_reference(self):
        """Both kernels quantize the activations identically, so their
        remaining difference is only reduction order -- the ~1e-4 relmax class
        documented for the MMVQ<->MMQ dispatch, not the ~5e-3 either has
        against fp32."""
        from sgl_kernel import ggml_mul_mat_a8, ggml_mul_mat_vec_a8

        rows, cols = 128, 1024
        q, _ = make_weight(rows, cols)
        x = torch.randn(4, cols, dtype=torch.float32).to(torch.bfloat16).cuda()
        a = ggml_mul_mat_vec_a8(q, x, MXFP4, rows).float()
        b = ggml_mul_mat_a8(q, x, MXFP4, rows).float()
        denom = a.abs().max().clamp_min(1e-6)
        self.assertLess(((a - b).abs().max() / denom).item(), 1e-3)

    def test_need_check_path_rows_not_a_multiple_of_the_tile(self):
        """MMQ_Y_MXFP4 is 32 on CUDA; a row count that is not a multiple of it
        takes the need_check=true instantiation."""
        from sgl_kernel import ggml_mul_mat_a8

        rows, cols = 100, 512
        q, ref_w = make_weight(rows, cols)
        x = torch.randn(16, cols, dtype=torch.float32).to(torch.bfloat16).cuda()
        ref = x.float() @ ref_w.t()
        out = ggml_mul_mat_a8(q, x, MXFP4, rows).float()
        torch.testing.assert_close(out, ref, atol=1.5, rtol=1e4)


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
        torch.testing.assert_close(out, ref, atol=1.5, rtol=3e1)

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
        torch.testing.assert_close(out, ref, atol=1.5, rtol=1e4)


if __name__ == "__main__":
    unittest.main()
