"""#417 -- the DSV4 Triton kernels must read FP8 E4M3 KV bytes on sm86.

The V4 KV cache stores its nope half as FP8 E4M3 by architecture, and Triton's
NVIDIA backend refuses the ``fp8e4nv`` element type below compute capability
8.9. Boot 11 died on exactly that, on both 3080 ranks:

    triton.compiler.errors.CompilationError:
    ValueError("type fp8e4nv not supported in this architecture.
                The supported fp8 dtypes are ('fp8e4b15', 'fp8e5')")

The fix is a manual decode from the raw byte. This file proves the decode is
exact and proves the gate that selects it, without a GPU: the kernels run
under Triton's interpreter (``TRITON_INTERPRET=1``) on CPU tensors.

What is NOT proved here, and is a GPU-window item: that the kernels *compile*
on an actual sm86 card. The interpreter never invokes the NVIDIA backend, so
it cannot see that failure mode. What it does prove is the arithmetic, which
is the part that could be silently wrong rather than loudly broken.

Reference discipline follows #262: the expected values come from a pure-Python
IEEE-754 decoder that never touches a torch fp8 dtype. A torch-based reference
would share any bug in torch's own Ampere decode and pass vacuously. Each
assertion is paired with a cross-probe that reading the same bytes under a
different fp8 interpretation *does* deviate, so a test that silently compared
nothing would fail.
"""

import os
import unittest
from unittest import mock

# Must be set before triton is imported anywhere in this process.
os.environ.setdefault("TRITON_INTERPRET", "1")

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

from sglang.srt.layers.attention.dsv4.fp8_triton_compat import (  # noqa: E402
    TRITON_FP8E4NV_MIN_CAPABILITY,
    e4m3fn_bits_to_f32,
    nope_cache_view,
    triton_fp8e4nv_supported,
)
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402
from sglang.test.test_utils import CustomTestCase  # noqa: E402

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


# --------------------------------------------------------------------------
# Reference decoders. Pure Python, no torch fp8 dtype anywhere (#262).
# --------------------------------------------------------------------------
def _decode_e4m3fn(byte: int) -> float:
    """IEEE-754-style E4M3-FN: 1 sign, 4 exp (bias 7), 3 mantissa, no inf."""
    sign = -1.0 if byte & 0x80 else 1.0
    exponent = (byte >> 3) & 0x0F
    mantissa = byte & 0x07
    if exponent == 0x0F and mantissa == 0x07:
        return float("nan")
    if exponent == 0:
        return sign * mantissa * (2.0**-9)
    return sign * (1.0 + mantissa / 8.0) * (2.0 ** (exponent - 7))


def _decode_e5m2(byte: int) -> float:
    """The cross-probe interpretation: 1 sign, 5 exp (bias 15), 2 mantissa."""
    sign = -1.0 if byte & 0x80 else 1.0
    exponent = (byte >> 2) & 0x1F
    mantissa = byte & 0x03
    if exponent == 0x1F:
        return float("nan") if mantissa else sign * float("inf")
    if exponent == 0:
        return sign * mantissa * (2.0**-16)
    return sign * (1.0 + mantissa / 4.0) * (2.0 ** (exponent - 15))


@triton.jit
def _decode_all_bytes_kernel(out_ptr, in_ptr, N: tl.constexpr):
    """Thin wrapper so the *production* device function is what gets tested."""
    offs = tl.arange(0, N)
    tl.store(out_ptr + offs, e4m3fn_bits_to_f32(tl.load(in_ptr + offs)))


def _run_decode(raw: torch.Tensor) -> torch.Tensor:
    out = torch.empty(raw.numel(), dtype=torch.float32)
    _decode_all_bytes_kernel[(1,)](out, raw, N=raw.numel())
    return out


class TestE4M3ManualDecode(CustomTestCase):
    """The manual decode must be exact for every byte a KV cache can hold."""

    ALL_BYTES = torch.arange(256, dtype=torch.uint8)

    def test_all_256_bytes_match_the_pure_python_reference(self):
        decoded = _run_decode(self.ALL_BYTES)

        deviating = 0
        for byte in range(256):
            expected = _decode_e4m3fn(byte)
            actual = float(decoded[byte])
            with self.subTest(byte=hex(byte)):
                if expected != expected:  # NaN
                    self.assertTrue(
                        actual != actual,
                        f"0x{byte:02x} must decode to NaN, got {actual}",
                    )
                    continue
                self.assertEqual(
                    expected,
                    actual,
                    f"0x{byte:02x}: expected {expected}, got {actual}",
                )
            if expected == expected and expected != _decode_e5m2(byte):
                deviating += 1

        # Cross-probe: if the reference and the code under test had somehow
        # collapsed onto a trivially-agreeing pair, reading the same bytes as
        # e5m2 would agree just as often. It must not.
        self.assertGreater(
            deviating,
            200,
            "e4m3 and e5m2 readings of the same bytes barely differ -- the "
            "comparison above is not actually discriminating",
        )

    def test_signed_zero_survives(self):
        """0x80 is -0.0, not +0.0.

        A decode built as ``where(sign, -magnitude, magnitude)`` loses this in
        some backends; the production code multiplies by -1.0 instead. Torch's
        own conversion keeps the sign, so a byte gate against an sm120 rank
        would flag it.
        """
        decoded = _run_decode(self.ALL_BYTES)
        self.assertEqual(float(decoded[0x00]), 0.0)
        self.assertEqual(float(decoded[0x80]), 0.0)
        self.assertEqual(
            torch.tensor([decoded[0x80]]).view(torch.int32).item(),
            torch.tensor([-0.0], dtype=torch.float32).view(torch.int32).item(),
            "0x80 must decode to negative zero",
        )

    def test_boundary_codes(self):
        decoded = _run_decode(self.ALL_BYTES)
        # Largest finite magnitude, the value torch.finfo(float8_e4m3fn).max
        # and the clamp bound the quantizer uses.
        self.assertEqual(float(decoded[0x7E]), 448.0)
        self.assertEqual(float(decoded[0xFE]), -448.0)
        # Smallest subnormal and the largest subnormal, 2**-9 apart.
        self.assertEqual(float(decoded[0x01]), 2.0**-9)
        self.assertEqual(float(decoded[0x07]), 7 * (2.0**-9))
        # First normal: exponent 1, mantissa 0.
        self.assertEqual(float(decoded[0x08]), 2.0**-6)
        # Only two NaN codes exist; 0x7E/0xFE next to them must be finite.
        self.assertFalse(float(decoded[0x7E]) != float(decoded[0x7E]))
        self.assertTrue(float(decoded[0x7F]) != float(decoded[0x7F]))
        self.assertTrue(float(decoded[0xFF]) != float(decoded[0xFF]))

    def test_the_reference_can_fail(self):
        """Can-fail proof: perturb one mantissa bit and the check must break.

        Without this, a decode that returned the reference unchanged -- or a
        comparison that compared nothing -- would pass silently.
        """
        decoded = _run_decode(self.ALL_BYTES)
        perturbed = decoded.clone()
        # 0x09 is exponent 1, mantissa 1: flipping the low mantissa bit moves
        # it by exactly one e4m3 ulp at that exponent.
        perturbed[0x09] = _decode_e4m3fn(0x08)

        self.assertNotEqual(
            float(perturbed[0x09]),
            _decode_e4m3fn(0x09),
            "the perturbation did not change the value -- the falsifier is inert",
        )
        with self.assertRaises(AssertionError):
            for byte in range(256):
                expected = _decode_e4m3fn(byte)
                if expected != expected:
                    continue
                self.assertEqual(expected, float(perturbed[byte]))


class TestPagedDequantKernel(CustomTestCase):
    """The whole paged dequant kernel, manual branch, against the torch ref.

    Covers what the byte-level test cannot: the page/scale/rope address
    arithmetic around the decode, exercised through the production kernel.
    """

    def _make_cache(self, num_pages, page_size, num_tokens, seed=0):
        from sglang.srt.layers.attention.dsv4.dequant_k_cache import (
            NOPE_ROPE_BYTES,
            PADDED_SCALE_PER_TOKEN,
        )

        torch.manual_seed(seed)
        raw_bytes = page_size * (NOPE_ROPE_BYTES + PADDED_SCALE_PER_TOKEN)
        bytes_per_page = (
            (raw_bytes + NOPE_ROPE_BYTES - 1) // NOPE_ROPE_BYTES
        ) * NOPE_ROPE_BYTES
        cache = torch.randint(0, 256, (num_pages, bytes_per_page), dtype=torch.uint8)
        # The nope bytes stay fully random -- all 256 codes including both NaNs
        # are what this exercises.
        #
        # #418: the ue8m0 scale window used to be [110, 141), narrowed because
        # "the reference flushes subnormal scales to zero, the kernel does
        # not". That disagreement is fixed (the flush is gone; see
        # test_dequant_k_cache_subnormal_scale_418.py), so the window is no
        # longer bounded by a defect -- it is bounded by fp32 range, which is
        # a property of the arithmetic rather than of either implementation.
        #
        # With the nope byte unrestricted, the product must stay a normal fp32
        # for a bit-exact comparison to mean anything:
        #   low  -- smallest non-zero e4m3fn is 2**-9, so 2**(e-127) * 2**-9
        #           >= 2**-126 requires e >= 10;
        #   high -- largest e4m3fn is 448 < 2**9, so 2**(e-127) * 2**9 < 2**128
        #           requires e < 246.
        # [10, 246) is that whole safe span: 236 of the 256 encodings, up from
        # 31. The ends belong to the #418 suite, which controls the nope byte
        # and can therefore reach them without leaving fp32 range.
        scale_start = page_size * NOPE_ROPE_BYTES
        cache[:, scale_start:] = torch.randint(
            10,
            246,
            (num_pages, bytes_per_page - scale_start),
            dtype=torch.uint8,
        )
        page_table = torch.randint(
            0, num_pages * page_size, (num_tokens,), dtype=torch.int32
        )
        return cache, page_table

    def test_manual_branch_matches_the_torch_reference(self):
        import sglang.srt.layers.attention.dsv4.dequant_k_cache as dqc

        page_size = 4
        cache, page_table = self._make_cache(
            num_pages=3, page_size=page_size, num_tokens=7
        )

        original = dqc.nope_cache_view
        try:
            # Force the manual branch regardless of what this host is.
            dqc.nope_cache_view = lambda u8, _dtype: (u8, False)
            out = dqc.dequantize_k_cache_paged(cache, page_table, page_size)
        finally:
            dqc.nope_cache_view = original

        expected = dqc.dequantize_k_cache_paged_ref(cache, page_table, page_size)

        self.assertEqual(out.shape, expected.shape)
        # The rope tail is a straight bf16 copy on both paths; the nope half is
        # where the manual decode lives. Compare the whole row exactly.
        torch.testing.assert_close(out, expected, atol=0, rtol=0, equal_nan=True)

    def test_manual_branch_is_not_trivially_equal(self):
        """Can-fail proof for the kernel test: corrupt one byte, it must diverge."""
        import sglang.srt.layers.attention.dsv4.dequant_k_cache as dqc

        page_size = 4
        cache, page_table = self._make_cache(
            num_pages=3, page_size=page_size, num_tokens=7, seed=1
        )
        corrupted = cache.clone()
        corrupted[0, 0] = (int(corrupted[0, 0]) + 1) % 256

        original = dqc.nope_cache_view
        try:
            dqc.nope_cache_view = lambda u8, _dtype: (u8, False)
            out = dqc.dequantize_k_cache_paged(corrupted, page_table, page_size)
        finally:
            dqc.nope_cache_view = original

        expected = dqc.dequantize_k_cache_paged_ref(cache, page_table, page_size)
        self.assertFalse(
            torch.equal(out.view(torch.int16), expected.view(torch.int16)),
            "a corrupted KV byte produced an identical result -- the "
            "comparison is not reading the cache",
        )


class TestCapabilityGate(CustomTestCase):
    """The gate, and the #192 pairing invariant it has to hold."""

    def setUp(self):
        triton_fp8e4nv_supported.cache_clear()
        self.addCleanup(triton_fp8e4nv_supported.cache_clear)

    def _with_capability(self, major, minor, cuda=True):
        import sglang.srt.layers.attention.dsv4.fp8_triton_compat as mod

        return mock.patch.multiple(
            mod,
            is_cuda=lambda: cuda,
            get_device_capability_no_init=lambda device_id: (major, minor),
        )

    def test_ampere_takes_the_manual_path(self):
        for major, minor in ((8, 0), (8, 6)):
            triton_fp8e4nv_supported.cache_clear()
            with self.subTest(sm=f"{major}{minor}"), self._with_capability(
                major, minor
            ):
                self.assertFalse(triton_fp8e4nv_supported(0))

    def test_ada_and_newer_keep_the_native_path(self):
        for major, minor in ((8, 9), (9, 0), (10, 0), (12, 0)):
            triton_fp8e4nv_supported.cache_clear()
            with self.subTest(sm=f"{major}{minor}"), self._with_capability(
                major, minor
            ):
                self.assertTrue(triton_fp8e4nv_supported(0))

    def test_non_cuda_stays_native(self):
        """ROCm stores E4M3-FNUZ, a different encoding this module does not
        decode, and its Triton backend accepts it natively."""
        with self._with_capability(9, 4, cuda=False):
            self.assertTrue(triton_fp8e4nv_supported(0))

    def test_threshold_is_the_documented_one(self):
        self.assertEqual(TRITON_FP8E4NV_MIN_CAPABILITY, (8, 9))

    def test_view_and_flag_are_returned_together(self):
        """#192 pairing invariant: a caller cannot get the uint8 view while
        still believing the pointer is fp8-typed, or the reverse."""
        raw = torch.arange(64, dtype=torch.uint8)

        with self._with_capability(8, 6):
            tensor, native = nope_cache_view(raw, torch.float8_e4m3fn)
            self.assertFalse(native)
            self.assertEqual(tensor.dtype, torch.uint8)

        triton_fp8e4nv_supported.cache_clear()
        with self._with_capability(12, 0):
            tensor, native = nope_cache_view(raw, torch.float8_e4m3fn)
            self.assertTrue(native)
            self.assertEqual(tensor.dtype, torch.float8_e4m3fn)

    def test_gate_answers_per_device(self):
        """#343: one process, two architectures. The gate is cached per device
        id, not in a single slot, or the second card gets the first's answer.
        """
        import sglang.srt.layers.attention.dsv4.fp8_triton_compat as mod

        caps = {0: (12, 0), 1: (8, 6)}
        with mock.patch.multiple(
            mod,
            is_cuda=lambda: True,
            get_device_capability_no_init=lambda device_id: caps[device_id],
        ):
            self.assertTrue(triton_fp8e4nv_supported(0))
            self.assertFalse(triton_fp8e4nv_supported(1))
            # Re-ask in the other order; the cache must not have collapsed.
            self.assertFalse(triton_fp8e4nv_supported(1))
            self.assertTrue(triton_fp8e4nv_supported(0))


if __name__ == "__main__":
    unittest.main()
