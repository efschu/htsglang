"""flashinfer autotune gate: the sm_12x W4A16 switch turns warmup autotune on
for an NVFP4 model even when the W4A4 backend is the fork's CUTLASS -- desk."""

import types
import unittest
from unittest import mock

from sglang.srt.layers.quantization import fp4_utils
from sglang.srt.layers.quantization import nvfp4_sm12x_w4a16 as S
from sglang.srt.model_executor.runner import flashinfer_autotune as A
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _mr(quant="modelopt_mixed"):
    return types.SimpleNamespace(
        device="cuda",
        server_args=types.SimpleNamespace(
            disable_flashinfer_autotune=False, moe_runner_backend="auto", moe_a2a_backend="none"
        ),
        model_config=types.SimpleNamespace(quantization=quant),
        spec_algorithm=types.SimpleNamespace(is_speculative=lambda: False),
        is_draft_model_runner=False,
    )


class TestGate(unittest.TestCase):
    def _gate(self, env, quant="modelopt_mixed"):
        S._reset_for_tests()
        e = {S.MAX_M_ENV: env} if env is not None else {}
        with mock.patch.dict("os.environ", e, clear=False), mock.patch.object(
            fp4_utils, "FP4_GEMM_RUNNER_BACKEND", fp4_utils.Fp4GemmRunnerBackend("cutlass")
        ), mock.patch.object(A, "cuda_sm_at_least", return_value=True), mock.patch(
            "sglang.srt.layers.quantization.fp8_utils.get_fp8_gemm_runner_backend",
            return_value=types.SimpleNamespace(is_flashinfer_cutlass=lambda: False),
        ), mock.patch("sglang.srt.utils.is_sm100_supported", return_value=False):
            if env is None:
                import os

                os.environ.pop(S.MAX_M_ENV, None)
            try:
                return A.should_run_flashinfer_autotune(_mr(quant))
            finally:
                S._reset_for_tests()

    def test_switch_on_turns_autotune_on(self):
        self.assertTrue(self._gate("16"))

    def test_switch_off_keeps_the_old_answer(self):
        self.assertFalse(self._gate(None))

    def test_non_fp4_model_unaffected(self):
        self.assertFalse(self._gate("16", quant="awq"))


if __name__ == "__main__":
    unittest.main()
