# Copyright 2023-2026 SGLang Team
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

"""#418: the k-cache dequant reference must not flush the ue8m0 scale 0x00.

THE DEFECT. ``dequantize_k_cache_paged_ref`` decoded the ue8m0 scale and then
replaced any result below the fp32 smallest-normal with zero -- an explicit FTZ
emulation. Exactly one of the 256 encodings decodes below that threshold, the
byte ``0x00`` (``exp2(0 - 127) == 2**-127``), so the clause fired on that
encoding and nothing else. The Triton kernel has no such clause. Reference and
kernel therefore disagreed on every input carrying a 0x00 scale, which makes
the reference useless as an oracle in precisely that regime -- and the
regression suite had been narrowed to avoid the regime rather than fix it (see
test_dsv4_fp8_triton_compat_417.py, whose scale generator drew from [110, 141)
with the disagreement named in a comment).

WHICH SIDE IS WRONG: the reference. ue8m0 is the OCP MX v1.0 §5.4.1 E8M0 scale
format -- an 8-bit UNSIGNED exponent with no sign, no mantissa, no zero, no
infinity and no subnormals. Encodings 0..254 are the finite values
``2**(e-127)`` and 255 alone is NaN. ``e == 0`` is therefore an ordinary legal
code meaning ``2**-127``; it is not a subnormal encoding. The fp32 value it
decodes to happens to be subnormal, which is what the flush confused it with.

torch says the same thing in its own words, in
``torch/include/torch/headeronly/util/Float8_e8m0fnu.h`` (the header cites the
OCP MX spec at :11-13)::

    // if exponent is zero, need to special case to return 2^-127 instead of zero
    if (x == 0) {
      return c10::detail::fp32_from_bits(0x00400000);
    }

The special case exists to PREVENT the flush the reference performed. So the
kernel already had the conformant semantics and the reference is the side that
moved; the fix deletes the flush rather than adding one to the kernel.

The tests below pin the reference against torch's own ue8m0 decode -- an
oracle independent of both implementations -- and not merely against the
kernel, so "they agree" cannot be satisfied by making both wrong together.

HERMETIC: the Triton kernel runs under TRITON_INTERPRET=1 on CPU tensors, the
idiom this directory already uses. What that cannot settle is what a real card
returns for ``tl.exp2(-127.0)``: PTX ``ex2.approx.ftz.f32`` flushes subnormal
RESULTS to zero while plain ``ex2.approx.f32`` does not, and which form Triton
emits is a codegen detail the interpreter never exercises. If the shipping
build emits the ftz form, hardware would coincidentally match the old
reference. That question is settled by a PTX dump, not by a model run; see
docs/dev/HANDOFF_629_BUNDLE.md. It does not change which semantics are
correct, only whether the kernel currently achieves them on-device.
"""

import os
import unittest

# Must be set before triton is imported anywhere in this process.
os.environ.setdefault("TRITON_INTERPRET", "1")

import torch  # noqa: E402

from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402
from sglang.test.test_utils import CustomTestCase  # noqa: E402

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

# The one encoding the flush fired on, and the fp8 code that makes the
# disagreement maximally visible (0x7E is e4m3fn 448.0, the format maximum).
_SUBNORMAL_SCALE_BYTE = 0x00
_FP8_MAX_BYTE = 0x7E


def _spec_scale(byte: int) -> float:
    """The ue8m0 value of a scale byte, per torch's own decoder (the oracle)."""
    return float(
        torch.tensor([byte], dtype=torch.uint8).view(torch.float8_e8m0fnu).float()[0]
    )


def _subnormal_product_codes() -> set:
    """fp8 codes whose product with 2**-127 lands in the fp32 SUBNORMAL range.

    A separate axis from the ue8m0 decode: here the scale is decoded
    identically by both sides and it is the MULTIPLY whose result cannot be
    represented as a normal fp32.
    """
    fp32_min_normal = 2.0**-126
    codes = set()
    for byte in range(256):
        value = float(
            torch.tensor([byte], dtype=torch.uint8).view(torch.float8_e4m3fn).float()[0]
        )
        if value != value or value == 0.0:  # NaN and the two zeros
            continue
        if abs(value * (2.0**-127)) < fp32_min_normal:
            codes.add(byte)
    return codes


def _build_cache(page_size: int, scale_byte: int, nope_byte: int):
    """A single page whose token 0 carries one chosen fp8 code and scale."""
    from sglang.srt.layers.attention.dsv4.dequant_k_cache import (
        NOPE_ROPE_BYTES,
        PADDED_SCALE_PER_TOKEN,
    )

    raw = page_size * (NOPE_ROPE_BYTES + PADDED_SCALE_PER_TOKEN)
    bytes_per_page = ((raw + NOPE_ROPE_BYTES - 1) // NOPE_ROPE_BYTES) * NOPE_ROPE_BYTES
    cache = torch.zeros((1, bytes_per_page), dtype=torch.uint8)
    cache[0, 0] = nope_byte
    cache[0, page_size * NOPE_ROPE_BYTES] = scale_byte
    page_table = torch.zeros(1, dtype=torch.int32)
    return cache, page_table


def _run_kernel(cache, page_table, page_size):
    """The shipping Triton path, manual-decode branch forced (no fp8 pointer)."""
    import sglang.srt.layers.attention.dsv4.dequant_k_cache as dqc

    original = dqc.nope_cache_view
    try:
        dqc.nope_cache_view = lambda u8, _dtype: (u8, False)
        return dqc.dequantize_k_cache_paged(cache, page_table, page_size)
    finally:
        dqc.nope_cache_view = original


class TestDequantSubnormalScale(CustomTestCase):
    page_size = 4

    def test_the_spec_says_scale_byte_zero_is_two_to_the_minus_127(self):
        """The oracle, stated before anything is compared against it."""
        self.assertEqual(_spec_scale(_SUBNORMAL_SCALE_BYTE), 2.0**-127)
        self.assertNotEqual(_spec_scale(_SUBNORMAL_SCALE_BYTE), 0.0)

    def test_reference_does_not_flush_the_subnormal_scale(self):
        """THE falsifier. On the unfixed reference this value is 0.0."""
        import sglang.srt.layers.attention.dsv4.dequant_k_cache as dqc

        cache, page_table = _build_cache(
            self.page_size, _SUBNORMAL_SCALE_BYTE, _FP8_MAX_BYTE
        )
        ref = dqc.dequantize_k_cache_paged_ref(cache, page_table, self.page_size)

        expected = torch.tensor(448.0 * (2.0**-127), dtype=torch.bfloat16)
        self.assertEqual(
            ref[0, 0, 0].item(),
            expected.item(),
            "the reference flushed a legal ue8m0 scale to zero",
        )
        self.assertNotEqual(ref[0, 0, 0].item(), 0.0)

    def test_reference_and_kernel_agree_on_the_subnormal_scale(self):
        import sglang.srt.layers.attention.dsv4.dequant_k_cache as dqc

        cache, page_table = _build_cache(
            self.page_size, _SUBNORMAL_SCALE_BYTE, _FP8_MAX_BYTE
        )
        ref = dqc.dequantize_k_cache_paged_ref(cache, page_table, self.page_size)
        out = _run_kernel(cache, page_table, self.page_size)
        torch.testing.assert_close(out, ref, atol=0, rtol=0, equal_nan=True)

    def test_the_corpus_of_every_fp8_code_at_the_subnormal_scale(self):
        """Breadth: the flush hit almost every fp8 code, not a corner.

        Restricted to inputs whose fp32 PRODUCT stays normal. That is a
        different axis from the one under test -- see
        test_subnormal_products_are_the_only_remaining_divergence for why the
        rest is deliberately not asserted here.
        """
        import sglang.srt.layers.attention.dsv4.dequant_k_cache as dqc

        checked = 0
        for nope_byte in range(256):
            if nope_byte in _subnormal_product_codes():
                continue
            with self.subTest(nope_byte=nope_byte):
                cache, page_table = _build_cache(
                    self.page_size, _SUBNORMAL_SCALE_BYTE, nope_byte
                )
                ref = dqc.dequantize_k_cache_paged_ref(
                    cache, page_table, self.page_size
                )
                out = _run_kernel(cache, page_table, self.page_size)
                torch.testing.assert_close(out, ref, atol=0, rtol=0, equal_nan=True)
                checked += 1
        # Guard against the exclusion quietly swallowing the whole corpus.
        self.assertGreater(
            checked, 120, "the corpus excluded too much to prove anything"
        )

    def test_subnormal_products_are_the_only_remaining_divergence(self):
        """The boundary of what this hermetic suite can settle, pinned.

        With the scale decode fixed, reference and kernel agree EXACTLY on
        every (fp8 code, scale byte) pair whose fp32 product is normal. Where
        the product itself lands in the fp32 subnormal range, the two may still
        differ -- torch's CPU multiply and the Triton interpreter's round those
        differently. That is a question about fp32 subnormal arithmetic, not
        about the ue8m0 decode, and the interpreter is not evidence about the
        card either way (PTX ftz; see the module docstring).

        What IS asserted: the divergence never escapes that regime. If a future
        change makes the two disagree on a normal-product input, this fires.
        """
        import sglang.srt.layers.attention.dsv4.dequant_k_cache as dqc

        diverged = set()
        for nope_byte in range(256):
            cache, page_table = _build_cache(
                self.page_size, _SUBNORMAL_SCALE_BYTE, nope_byte
            )
            ref = dqc.dequantize_k_cache_paged_ref(cache, page_table, self.page_size)
            out = _run_kernel(cache, page_table, self.page_size)
            a, b = ref[0, 0, 0].item(), out[0, 0, 0].item()
            if not (a == b or (a != a and b != b)):
                diverged.add(nope_byte)

        escaped = diverged - _subnormal_product_codes()
        self.assertEqual(
            escaped,
            set(),
            f"reference and kernel disagree on fp8 code(s) {sorted(escaped)} "
            f"whose fp32 product is NORMAL -- that is a scale-decode defect, "
            f"not the known subnormal-arithmetic regime",
        )

    def test_the_corpus_of_every_scale_encoding(self):
        """All 256 scale bytes, reference vs kernel, exact."""
        import sglang.srt.layers.attention.dsv4.dequant_k_cache as dqc

        for scale_byte in range(256):
            with self.subTest(scale_byte=scale_byte):
                cache, page_table = _build_cache(
                    self.page_size, scale_byte, _FP8_MAX_BYTE
                )
                ref = dqc.dequantize_k_cache_paged_ref(
                    cache, page_table, self.page_size
                )
                out = _run_kernel(cache, page_table, self.page_size)
                torch.testing.assert_close(out, ref, atol=0, rtol=0, equal_nan=True)

    def test_the_reference_scale_matches_the_spec_across_all_encodings(self):
        """Not just "ref == kernel" -- both are checked against torch's ue8m0.

        Byte 255 is excluded and pinned separately below: it is a known
        NON-CONFORMANCE shared by both sides, and folding it in here would let
        this test assert that the two agree while both are wrong.
        """
        import sglang.srt.layers.attention.dsv4.dequant_k_cache as dqc

        for scale_byte in range(255):
            with self.subTest(scale_byte=scale_byte):
                cache, page_table = _build_cache(self.page_size, scale_byte, 0x38)
                ref = dqc.dequantize_k_cache_paged_ref(
                    cache, page_table, self.page_size
                )
                # 0x38 is e4m3fn 1.0, so the decoded element IS the scale.
                expected = torch.tensor(
                    _spec_scale(scale_byte), dtype=torch.bfloat16
                ).item()
                self.assertEqual(ref[0, 0, 0].item(), expected)

    def test_scale_byte_255_is_a_known_shared_non_conformance(self):
        """Documented, not silently tolerated.

        The spec makes 255 the sole NaN encoding; both sides compute
        ``exp2(128) == inf`` instead. Reference and kernel AGREE, so the oracle
        is not poisoned here and #418 does not cover it -- but an agreement
        between two wrong implementations is worth a name. If this ever starts
        failing because one side moved to NaN, the other must move with it.
        """
        import sglang.srt.layers.attention.dsv4.dequant_k_cache as dqc

        self.assertTrue(torch.isnan(torch.tensor(_spec_scale(255))))

        cache, page_table = _build_cache(self.page_size, 255, 0x38)
        ref = dqc.dequantize_k_cache_paged_ref(cache, page_table, self.page_size)
        out = _run_kernel(cache, page_table, self.page_size)
        self.assertTrue(torch.isinf(ref[0, 0, 0]))
        torch.testing.assert_close(out, ref, atol=0, rtol=0, equal_nan=True)


if __name__ == "__main__":
    unittest.main()
