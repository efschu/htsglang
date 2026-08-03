# SPDX-License-Identifier: Apache-2.0
"""#512: the GGUF MoE MMQ expert offset must be computed in 64 bits.

``moe.cuh`` reaches an expert's weights with

    x = (const block_q_t*)((const char*)vx + exp_idx * exp_stride);

``exp_stride`` is ``W.stride(0)`` of the **uint8** expert tensor, i.e. a BYTE
stride, so ``exp_idx * exp_stride`` is the byte offset of the last expert --
about the size of the whole local tensor. It was declared ``const int``, so
the product wrapped to a negative offset once a rank's per-layer expert
weights passed 2 GiB: an out-of-bounds read far BELOW the tensor. The call
site in ``gguf_kernel.cu`` already passed ``W.stride(0)``, which is
``int64_t``, so the whole defect was an implicit narrowing at the parameter.

Same failure mode as #109/#112 (an out-of-range read reached through
``exp_idx``) from the other operand, which is why the #112 expert-id guard is
pinned here too -- this file has lost that guard once already.

Everything here is static or arithmetic: no GPU, no wheel, no large tensor.
The wrap is demonstrated on a synthetic stride just over the ceiling rather
than by allocating anything.

Usage:
    python3 -m pytest test/registered/unit/quantization/test_gguf_moe_stride_width_512.py -v
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

_ROOT = Path(__file__).resolve().parents[4]
_MOE_CUH = _ROOT / "sgl-kernel" / "csrc" / "quantization" / "gguf" / "moe.cuh"
_MOE_VEC_CUH = _ROOT / "sgl-kernel" / "csrc" / "quantization" / "gguf" / "moe_vec.cuh"
_LAUNCHER = _ROOT / "sgl-kernel" / "csrc" / "quantization" / "gguf" / "gguf_kernel.cu"

INT32_CEILING = 2**31


def _wrap(value: int, bits: int) -> int:
    """Two's-complement wrap of ``value`` into ``bits``, as C would."""
    span = 1 << bits
    value &= span - 1
    return value - span if value >= span // 2 else value


class TestExpertOffsetArithmetic(unittest.TestCase):
    """The defect and the fix, as arithmetic. No source, no device."""

    def test_a_byte_stride_product_wraps_negative_in_32_bits(self):
        # Synthetic, deliberately small: a stride just over the ceiling needs
        # only two experts to cross it. The real geometry is in the next test.
        exp_stride = INT32_CEILING // 2 + 4096  # ~1 GiB + a nudge
        exp_idx = 2
        self.assertGreater(exp_idx * exp_stride, INT32_CEILING)
        self.assertLess(_wrap(exp_idx * exp_stride, 32), 0)
        self.assertEqual(_wrap(exp_idx * exp_stride, 64), exp_idx * exp_stride)

    def test_the_documented_dsv4_geometry_crosses(self):
        """DeepSeek-V4-Flash w13, Q4_K, 256 experts -- the numbers in the
        moe.cuh comment, recomputed here so the comment is a testable claim."""
        elements_per_expert = 2 * 2048 * 4096  # 2 * moe_intermediate * hidden
        bytes_per_element = 144 / 256  # Q4_K: 144 bytes per 256 elements
        stride = int(elements_per_expert * bytes_per_element)
        self.assertEqual(stride, 9_437_184)  # 9.44 MB
        total = 256 * stride
        self.assertGreater(total, INT32_CEILING)
        # First local expert whose offset does not fit in int32.
        first_bad = INT32_CEILING // stride + 1
        self.assertEqual(first_bad, 228)
        self.assertLess(_wrap(first_bad * stride, 32), 0)
        # ...and the shard sizes that keep it under, which is why this rig
        # never saw it: TP=3 leaves ~86 local experts, TP=2 leaves 128.
        for local in (86, 128):
            self.assertLess(local * stride, INT32_CEILING)

    def test_a_negative_offset_reads_below_the_tensor(self):
        """What the wrap means at the pointer, in one line."""
        base = 0x7F00_0000_0000  # any plausible device pointer
        stride = INT32_CEILING // 2 + 4096
        wrapped = base + _wrap(3 * stride, 32)
        self.assertLess(wrapped, base, "the wrapped offset points below vx")


class TestDeclaredWidths(unittest.TestCase):
    """The fix, in the source. These are what regress if someone re-narrows."""

    def _moe_cuh(self) -> str:
        self.assertTrue(_MOE_CUH.exists(), f"missing {_MOE_CUH}")
        return _MOE_CUH.read_text()

    def _moe_cuh_code(self) -> str:
        """``moe.cuh`` with ``//`` comments stripped.

        Needed because the file DOCUMENTS the defects it fixed -- the #112
        note quotes the old ``exp_idx > 255`` guard and the #512 note quotes
        the 32-bit product -- and a check for those strings would otherwise
        match the explanation instead of the code.
        """
        return "\n".join(
            line.split("//", 1)[0] for line in self._moe_cuh().splitlines()
        )

    def test_every_exp_stride_parameter_is_int64(self):
        src = self._moe_cuh_code()
        narrow = re.findall(r"\bconst\s+int\s+exp_stride\b", src)
        self.assertEqual(
            narrow,
            [],
            f"{len(narrow)} exp_stride parameter(s) are still 32-bit in "
            f"{_MOE_CUH.name}; the product with exp_idx is a byte offset",
        )
        wide = re.findall(r"\bconst\s+int64_t\s+exp_stride\b", src)
        self.assertGreaterEqual(
            len(wide), 20, f"only {len(wide)} int64_t exp_stride declarations"
        )

    def test_the_offset_expression_still_goes_through_exp_stride(self):
        """Guards against a 'fix' that widens the parameter and then casts it
        back, or that changes the expression out from under this test."""
        src = self._moe_cuh_code()
        self.assertRegex(
            src, r"\(const char\*\)vx \+ exp_idx \* exp_stride", "offset expression"
        )
        self.assertNotRegex(src, r"\(int\)\s*exp_stride")
        self.assertNotRegex(src, r"static_cast<int>\s*\(\s*exp_stride")

    def test_the_call_site_passes_the_tensor_stride_unnarrowed(self):
        """W.stride(0) is int64_t in torch, so the narrowing was purely at the
        callee's parameter -- nothing on the caller side had to change."""
        src = _LAUNCHER.read_text()
        self.assertIn("W.stride(0),", src)
        self.assertNotIn("(int)W.stride(0)", src)
        self.assertNotIn("static_cast<int>(W.stride(0))", src)

    def test_the_112_expert_id_guard_is_still_in_place(self):
        """This file has lost an expert-id bound once (#109/#112). The guard
        and the byte-stride width are the two halves of the same read."""
        code = self._moe_cuh_code()
        self.assertIn("if (exp_idx >= num_experts || exp_idx < 0) return;", code)
        self.assertNotIn("exp_idx > 255", code)


class TestSiblingsInTheSameFamily(unittest.TestCase):
    """The other expert-offset multiply, checked rather than assumed."""

    def test_the_mmvq_moe_offset_is_in_block_units_and_has_headroom(self):
        """``moe_vec.cuh`` computes ``expert * nrows * blocks_per_row`` in int,
        but on a ``block_q_t*``, i.e. in units of 32 elements -- 16 to 32x more
        headroom than a byte stride. Left 32-bit deliberately; this test is the
        bound that says when that stops being true."""
        src = _MOE_VEC_CUH.read_text()
        self.assertIn("expert * nrows * blocks_per_row", src)
        # Worst geometry this fork serves: 256 experts, w13 rows = 2*2048,
        # cols = 7168 -> 224 blocks per row.
        worst = 256 * (2 * 2048) * (7168 // 32)
        self.assertLess(worst, INT32_CEILING)
        self.assertGreater(
            INT32_CEILING / worst, 9.0, "headroom fell below 9x; widen it too"
        )

    def test_the_mmq_row_and_column_indices_stay_in_range(self):
        """The other int products in moe_q are element indices into the output
        and into one expert's blocks, not byte offsets."""
        # dst index: ncols_dst * nrows_dst = tokens * top_k * rows.
        self.assertLess(8192 * 8 * 4096, INT32_CEILING)
        # x block index within ONE expert: nrows_x * blocks_per_row_x.
        self.assertLess(4096 * (7168 // 32), INT32_CEILING)


if __name__ == "__main__":
    unittest.main()
