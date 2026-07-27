"""#192: the SGLANG_DETERMINISTIC_FP8_GEMM gate, tested without a GPU.

The numerical claim -- that the flag turns a nondeterministic fp8 linear into a
bit-identical one -- needs an sm8x card and lives in
``scripts/determinism/fp8_det_flag_falsifier.py`` (Marlin 24/1200 mismatching
repeats at M=512 vs 0/1200 with the flag, RTX 3080). What is testable here, and
is where the bugs would actually hide, is the ROUTING:

  * the flag must fire on sm80..88 and nowhere else -- in particular sm89, sm90
    and sm120 must be untouched, because the defect was measured on Ampere and
    the flag is not a global "no fp8";
  * it must be off by default, so the stock path does not move;
  * it must beat an explicit SGLANG_FORCE_FP8_MARLIN, otherwise a determinism
    request would silently keep the nondeterministic kernel;
  * and it must PAIR: switching Marlin off has to arm the dequant fallback in
    the same step. On sm8x Marlin is the only fp8 GEMM there is, so a gate that
    forgot the pairing would leave a block-scaled checkpoint with no fp8 GEMM at
    all -- exactly the state #190 said a fix must not produce.

Capability and vendor are faked rather than sniffed, so the whole matrix runs on
any machine including CPU-only CI.
"""

import unittest
from unittest import mock

from sglang.srt.environ import envs
from sglang.srt.layers.quantization import fp8_utils as U


def _clear():
    U.deterministic_fp8_marlin_disabled.cache_clear()
    U.fp8_needs_dequant_fallback.cache_clear()


class _Env:
    """Set the flag and drop the lru_caches that read it, on both edges."""

    def __init__(self, value, sm=(8, 6), cuda=True):
        self.value, self.sm, self.cuda = value, sm, cuda

    def __enter__(self):
        self._old = envs.SGLANG_DETERMINISTIC_FP8_GEMM.get()
        envs.SGLANG_DETERMINISTIC_FP8_GEMM.set(self.value)
        _clear()
        self._patches = [
            mock.patch.object(U, "is_cuda", lambda: self.cuda),
            mock.patch.object(U, "get_device_capability", lambda *a, **k: self.sm),
            # The NVIDIA-namespace reader (#171): None off NVIDIA, so a fake
            # non-CUDA vendor cannot answer an NVIDIA capability question.
            mock.patch.object(
                U,
                "get_cuda_sm",
                lambda *a, **k: (
                    self.sm[0] * 10 + self.sm[1] if self.cuda else None
                ),
            ),
        ]
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in self._patches:
            p.stop()
        envs.SGLANG_DETERMINISTIC_FP8_GEMM.set(self._old)
        _clear()
        return False


class TestDeterministicFp8Gate(unittest.TestCase):
    def tearDown(self):
        _clear()

    def test_default_is_off_on_every_capability(self):
        """No flag, no change -- at any sm. This is the backward-compat gate."""
        for sm in [(7, 5), (8, 0), (8, 6), (8, 9), (9, 0), (12, 0)]:
            with _Env(False, sm=sm):
                self.assertFalse(
                    U.deterministic_fp8_marlin_disabled(),
                    f"flag fired while unset at sm{sm}",
                )

    def test_fires_only_on_sm80_to_sm88(self):
        for sm in [(8, 0), (8, 6), (8, 7), (8, 8)]:
            with _Env(True, sm=sm):
                self.assertTrue(
                    U.deterministic_fp8_marlin_disabled(), f"missed sm{sm}"
                )

    def test_sm89_and_above_untouched(self):
        """sm89+/sm90/sm120 keep their native / flashinfer fp8 path.

        #190 measured 0/2000 at the same shape on sm120, so there is nothing to
        fix there and disabling fp8 would be a pure regression.
        """
        for sm in [(8, 9), (9, 0), (10, 0), (12, 0)]:
            with _Env(True, sm=sm):
                self.assertFalse(
                    U.deterministic_fp8_marlin_disabled(),
                    f"flag wrongly fired at sm{sm}",
                )

    def test_below_sm80_untouched(self):
        """sm75 has no Marlin fp8 to begin with; it is already on the fallback."""
        with _Env(True, sm=(7, 5)):
            self.assertFalse(U.deterministic_fp8_marlin_disabled())

    def test_non_cuda_untouched(self):
        """gfx900 reports (9, 0) -- the vendor-namespace trap. Guarded by is_cuda."""
        with _Env(True, sm=(9, 0), cuda=False):
            self.assertFalse(U.deterministic_fp8_marlin_disabled())
        with _Env(True, sm=(8, 0), cuda=False):
            self.assertFalse(U.deterministic_fp8_marlin_disabled())

    def test_unknown_capability_leaves_it_alone(self):
        with _Env(True, sm=(8, 6)):
            with mock.patch.object(
                U, "get_device_capability", side_effect=RuntimeError("no device")
            ):
                _clear()
                self.assertFalse(U.deterministic_fp8_marlin_disabled())

    def test_pairs_with_the_dequant_fallback(self):
        """The load-bearing one: Marlin off MUST mean fallback on.

        Without this clause an sm8x rank would refuse Marlin and never arm
        use_block_dequant, leaving a block-scaled fp8 checkpoint with no GEMM.
        """
        with _Env(True, sm=(8, 6)):
            self.assertTrue(U.fp8_needs_dequant_fallback())

    def test_fallback_not_armed_when_flag_is_off(self):
        """An sm8x rank without the flag keeps Marlin, so no fallback."""
        with _Env(False, sm=(8, 6)):
            with mock.patch.object(U, "fp8_native_gemm_available", lambda: False), \
                 mock.patch.object(U, "cutlass_fp8_supported", lambda: False), \
                 mock.patch.object(U, "can_auto_enable_marlin_fp8", lambda: True):
                _clear()
                self.assertFalse(U.fp8_needs_dequant_fallback())

    def test_fallback_untouched_on_sm120(self):
        with _Env(True, sm=(12, 0)):
            with mock.patch.object(U, "fp8_native_gemm_available", lambda: True):
                _clear()
                self.assertFalse(U.fp8_needs_dequant_fallback())

    def test_can_auto_enable_marlin_is_hardware_only(self):
        """The flag is applied at the consumers, not in the capability probe.

        Gating the probe itself would also silence the fp8 MoE and FBGEMM paths,
        which have no fallback on sm8x -- see the docstring there.
        """
        with _Env(True, sm=(8, 6)):
            self.assertTrue(U.can_auto_enable_marlin_fp8())


class TestLinearMethodRouting(unittest.TestCase):
    """The gate as the dense linear method actually consumes it."""

    def tearDown(self):
        _clear()

    def _method(self, block):
        from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod

        cfg = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=[128, 128] if block else None,
        )
        return Fp8LinearMethod(cfg)

    def _patched(self, flag, sm, block):
        from sglang.srt.layers.quantization import fp8 as F

        with _Env(flag, sm=sm):
            with mock.patch.object(F, "_is_cuda", True), mock.patch.object(
                F, "can_auto_enable_marlin_fp8", lambda: 80 <= sm[0] * 10 + sm[1] < 89
            ):
                return self._method(block)

    @unittest.skipUnless(
        __import__("torch").cuda.is_available(), "constructing the method needs CUDA"
    )
    def test_sm86_block_checkpoint_switches_lane(self):
        off = self._patched(False, (8, 6), block=True)
        self.assertTrue(off.use_marlin)
        self.assertFalse(off.use_block_dequant)

        on = self._patched(True, (8, 6), block=True)
        self.assertFalse(on.use_marlin)
        self.assertTrue(on.use_block_dequant, "no fp8 GEMM left -- the #190 trap")

    @unittest.skipUnless(
        __import__("torch").cuda.is_available(), "constructing the method needs CUDA"
    )
    def test_sm86_per_channel_checkpoint_switches_lane(self):
        on = self._patched(True, (8, 6), block=False)
        self.assertFalse(on.use_marlin)
        self.assertTrue(on.use_dequant, "no fp8 GEMM left -- the #190 trap")

    @unittest.skipUnless(
        __import__("torch").cuda.is_available(), "constructing the method needs CUDA"
    )
    def test_sm120_unchanged_by_the_flag(self):
        off = self._patched(False, (12, 0), block=True)
        on = self._patched(True, (12, 0), block=True)
        self.assertEqual(off.use_marlin, on.use_marlin)
        self.assertEqual(off.use_block_dequant, on.use_block_dequant)


if __name__ == "__main__":
    unittest.main()
