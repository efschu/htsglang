"""Task #257: the GGUF dequant scratch must be a post in the KV budget.

THE DEFECT
----------
`_reserve_dequant_workspace` sizes a persistent per-rank dequant buffer at
WEIGHT-LOAD time, i.e. before the KV budget is profiled -- that is the whole
point of #63, and it is why the bytes it holds are charged correctly. But it
refuses two classes of target:

  * over the cap (`SGLANG_GGUF_DEQUANT_WS_CAP_MIB`, default 512 MiB), and
  * every target at all when the installed kernel has no `ggml_dequantize(
    ..., out=)` schema.

Both refusals mean the same thing at forward time: a FRESH allocation of the
target's full size. On the eager path -- which includes the pre-capture
warmup forwards, run after profiling -- nothing else covers those bytes, so
the KV pool was sized as if they were free. Qwen3.6-27B Q3_K_M at TP=1: the
lm_head dequant target is 2.37 GiB (1.19 GiB at TP=2, #70), and
`--mem-fraction-static 0.90` OOMed in ggml_dequantize during the decode-graph
warmup while 0.82 carried.

WHAT IS PINNED HERE
-------------------
 1. The peak target is recorded through BOTH refusals, not only the accepted
    ones -- a recorder that sits after the cap check would report 512 MiB for
    a 2.37 GiB demand.
 2. The residual is peak MINUS what the workspace already holds. The held
    part is resident before profiling and must not be charged twice.
 3. MAX across dtypes, not sum: over-cap dequants are transient and
    sequential on one stream.
 4. Zero for a model that never records a peak -> the non-GGUF budget
    arithmetic is untouched.
 5. The budget site actually subtracts it, with a stubbed workspace value.

CPU only: this is arithmetic over recorded shapes, not a kernel.
"""

import unittest
from unittest import mock

import torch

from sglang.srt.layers.quantization import gguf as G
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

_MIB = 1 << 20
_CPU = torch.device("cpu")


class _CleanRegistries(CustomTestCase):
    """Every test starts from an empty workspace + peak registry."""

    def setUp(self):
        super().setUp()
        self._ws = dict(G._DEQUANT_WS)
        self._peak = dict(G._DEQUANT_PEAK_TARGET)
        G._DEQUANT_WS.clear()
        G._DEQUANT_PEAK_TARGET.clear()

    def tearDown(self):
        G._DEQUANT_WS.clear()
        G._DEQUANT_WS.update(self._ws)
        G._DEQUANT_PEAK_TARGET.clear()
        G._DEQUANT_PEAK_TARGET.update(self._peak)
        super().tearDown()


class PeakIsRecordedThroughBothRefusalsTest(_CleanRegistries):
    def test_over_cap_target_is_recorded_but_not_held(self):
        """The 2.37 GiB lm_head case, scaled down to the same shape."""
        with (
            mock.patch.object(G, "_dequant_supports_out", return_value=True),
            mock.patch.object(G, "_dequant_ws_cap_bytes", return_value=64 * _MIB),
        ):
            numel = 128 * _MIB // 2  # 128 MiB in bf16, i.e. 2x the cap
            G._reserve_dequant_workspace(numel, torch.bfloat16, _CPU)

        key = (_CPU, torch.bfloat16)
        self.assertEqual(
            G._DEQUANT_PEAK_TARGET[key],
            128 * _MIB,
            "the over-cap target was not recorded; the budget would be "
            "short by exactly the allocation that OOMs the warmup forward",
        )
        self.assertNotIn(
            key,
            G._DEQUANT_WS,
            "an over-cap target must still NOT be pinned in the workspace",
        )
        self.assertEqual(G.gguf_dequant_scratch_residual_bytes(), 128 * _MIB)

    def test_old_wheel_without_out_schema_records_every_target(self):
        """No `out=` schema -> every dequant fresh-allocates, cap or not."""
        with (
            mock.patch.object(G, "_dequant_supports_out", return_value=False),
            mock.patch.object(G, "_dequant_ws_cap_bytes", return_value=512 * _MIB),
        ):
            G._reserve_dequant_workspace(8 * _MIB // 2, torch.bfloat16, _CPU)

        self.assertEqual(G._DEQUANT_WS, {})
        self.assertEqual(G.gguf_dequant_scratch_residual_bytes(), 8 * _MIB)

    def test_peak_is_the_max_over_layers(self):
        with (
            mock.patch.object(G, "_dequant_supports_out", return_value=True),
            mock.patch.object(G, "_dequant_ws_cap_bytes", return_value=1 * _MIB),
        ):
            for mib in (4, 32, 12):
                G._reserve_dequant_workspace(mib * _MIB // 2, torch.bfloat16, _CPU)
        self.assertEqual(G.gguf_dequant_scratch_residual_bytes(), 32 * _MIB)


class ResidualSubtractsWhatIsAlreadyHeldTest(_CleanRegistries):
    def test_held_workspace_is_not_charged_twice(self):
        """A target UNDER the cap is allocated at load time -> residual 0.

        Those bytes are resident when the profiler reads free memory, so
        charging them again would shrink the KV pool for nothing.
        """
        with (
            mock.patch.object(G, "_dequant_supports_out", return_value=True),
            mock.patch.object(G, "_dequant_ws_cap_bytes", return_value=64 * _MIB),
        ):
            G._reserve_dequant_workspace(16 * _MIB // 2, torch.bfloat16, _CPU)

        self.assertIn((_CPU, torch.bfloat16), G._DEQUANT_WS)
        self.assertEqual(
            G.gguf_dequant_scratch_residual_bytes(),
            0,
            "a workspace that is already allocated must not be charged "
            "against the KV budget a second time",
        )

    def test_residual_is_peak_minus_held(self):
        """Stubbed workspace value: 40 MiB held against a 100 MiB peak."""
        key = (_CPU, torch.bfloat16)
        G._DEQUANT_PEAK_TARGET[key] = 100 * _MIB
        G._DEQUANT_WS[key] = torch.empty(40 * _MIB // 2, dtype=torch.bfloat16)
        self.assertEqual(G.gguf_dequant_scratch_residual_bytes(), 60 * _MIB)

    def test_max_not_sum_across_dtypes(self):
        """Over-cap dequants are transient and sequential on one stream."""
        G._DEQUANT_PEAK_TARGET[(0, torch.bfloat16)] = 100 * _MIB
        G._DEQUANT_PEAK_TARGET[(0, torch.float16)] = 60 * _MIB
        self.assertEqual(G.gguf_dequant_scratch_residual_bytes(), 100 * _MIB)

    def test_device_filter_keeps_a_foreign_rank_out_of_this_budget(self):
        G._DEQUANT_PEAK_TARGET[(0, torch.bfloat16)] = 100 * _MIB
        G._DEQUANT_PEAK_TARGET[(1, torch.bfloat16)] = 900 * _MIB
        self.assertEqual(G.gguf_dequant_scratch_residual_bytes(0), 100 * _MIB)
        self.assertEqual(G.gguf_dequant_scratch_residual_bytes(1), 900 * _MIB)

    def test_no_gguf_layers_means_no_post(self):
        """The byte-identity claim for every other quantization."""
        self.assertEqual(G.gguf_dequant_scratch_residual_bytes(), 0)
        self.assertEqual(G.gguf_dequant_scratch_residual_bytes(0), 0)


class BudgetSiteTest(_CleanRegistries):
    """The mixin helper, driven with a stubbed workspace value."""

    def _runner(self, device="cpu"):
        class FakeRunner:
            pass

        r = FakeRunner()
        r.device = device
        return r

    def _scratch_gb(self, runner):
        from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
            ModelRunnerKVCacheMixin,
        )

        return ModelRunnerKVCacheMixin._gguf_dequant_scratch_gb(runner)

    def test_helper_reports_the_recorded_residual_in_gib(self):
        G._DEQUANT_PEAK_TARGET[(_CPU, torch.bfloat16)] = 2048 * _MIB
        self.assertAlmostEqual(self._scratch_gb(self._runner()), 2.0, places=6)

    def test_helper_is_zero_without_gguf_layers(self):
        self.assertEqual(self._scratch_gb(self._runner()), 0.0)

    def test_budget_site_subtracts_the_post(self):
        """Source assertion: the profiler must charge it, not just log it.

        The arithmetic itself needs a loaded model and a GPU; what a unit
        test can pin is that the post is wired into `rest_memory` and into
        the suggested minimum fraction, so a boot that cannot fit says so
        instead of OOMing minutes later inside ggml_dequantize.
        """
        import inspect

        from sglang.srt.model_executor import model_runner_kv_cache_mixin as M

        src = inspect.getsource(M.ModelRunnerKVCacheMixin._profile_available_bytes)
        self.assertIn("gguf_scratch_gb = self._gguf_dequant_scratch_gb()", src)
        self.assertIn("rest_memory -= gguf_scratch_gb", src)
        self.assertIn(
            "(available_gpu_memory - gguf_scratch_gb) / pre_model_load_memory",
            src,
            "the suggested --mem-fraction-static still ignores the GGUF "
            "post, so it would name a value that fails for the same reason",
        )


if __name__ == "__main__":
    unittest.main()
