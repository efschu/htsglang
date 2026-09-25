"""sm_12x small-M W4A16 native seam (nvfp4_sm12x_w4a16) -- desk, no GPU.

The kernel itself is proven on the 5090 by
benchmark/nvfp4_native/bench_fi_next_sm12x.py (numerics vs an fp32 reference
from the checkpoint-form bytes); here: the decision, the default-off contract,
the arch gate, the hard error on a missing kernel, and what reaches
mm_bf16_fp4.
"""

import sys
import types
import unittest
from unittest import mock

import torch

from sglang.srt.layers.quantization import nvfp4_sm12x_w4a16 as S
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _layer(n=256, k=128, *, global_scale=None, pad_cols=0, out_size=None):
    L = torch.nn.Module()
    L.weight = torch.nn.Parameter(torch.zeros(n, k // 2, dtype=torch.uint8), requires_grad=False)
    L.weight_scale_interleaved = torch.nn.Parameter(
        torch.zeros(((n + 127) // 128) * 128, ((k // 16 + 3) // 4) * 4, dtype=torch.float8_e4m3fn),
        requires_grad=False,
    )
    L.alpha = torch.nn.Parameter(torch.tensor(0.5, dtype=torch.float32), requires_grad=False)
    L.input_scale_inv = torch.nn.Parameter(torch.tensor(4.0, dtype=torch.float32), requires_grad=False)
    if global_scale is not None:
        L.weight_global_scale = torch.nn.Parameter(torch.tensor(global_scale, dtype=torch.float32), requires_grad=False)
    L.weights_padding_cols = pad_cols
    L.output_size_per_partition = out_size if out_size is not None else n
    return L


class _Env:
    def __init__(self, value, major=12, present=True):
        self.value, self.major, self.present = value, major, present

    def __enter__(self):
        S._reset_for_tests()
        env = {} if self.value is None else {S.MAX_M_ENV: str(self.value)}
        self._p = [
            mock.patch.dict("os.environ", env, clear=False),
            mock.patch("sglang.srt.utils.common.get_device_capability", return_value=(self.major, 0)),
            mock.patch.object(S, "native_w4a16_kernel_present", return_value=self.present),
        ]
        if self.value is None:
            import os

            os.environ.pop(S.MAX_M_ENV, None)
        for p in self._p:
            p.start()
        return self

    def __exit__(self, *a):
        for p in reversed(self._p):
            p.stop()
        S._reset_for_tests()


def _fake_flashinfer(record):
    def mm_bf16_fp4(a, b, b_descale, alpha=None, *, backend, out_dtype=None, out=None, block_size=16, enable_pdl=True):
        record.append(dict(a=a, b=b, sf=b_descale, alpha=alpha, backend=backend, out_dtype=out_dtype))
        return torch.ones(a.shape[0], b.shape[0], dtype=out_dtype or a.dtype)

    return types.SimpleNamespace(mm_bf16_fp4=mm_bf16_fp4)


class TestChoice(unittest.TestCase):
    def test_pure_choice(self):
        c = S.choose_sm12x_fp4_kernel
        self.assertEqual(c(1, 32, 12), S.W4A16_NATIVE)
        self.assertEqual(c(32, 32, 12), S.W4A16_NATIVE)
        self.assertEqual(c(33, 32, 12), S.W4A4)
        self.assertEqual(c(4096, 32, 12), S.W4A4)
        self.assertEqual(c(1, 0, 12), S.W4A4)  # default off
        self.assertEqual(c(1, 32, 8), S.W4A4)  # 3080 never
        self.assertEqual(c(1, 32, 10), S.W4A4)
        self.assertEqual(c(0, 32, 12), S.W4A4)


class TestSeam(unittest.TestCase):
    def _run(self, x, layer, backend="cutlass", **env):
        rec = []
        with _Env(**env), mock.patch.dict(sys.modules, {"flashinfer": _fake_flashinfer(rec)}):
            out = S.maybe_apply_sm12x_w4a16(layer, x, None, backend)
        return out, rec

    def test_default_off_is_none_and_never_touches_the_device(self):
        with _Env(None, major=12), mock.patch(
            "sglang.srt.utils.common.get_device_capability", side_effect=AssertionError("no device query when off")
        ):
            self.assertIsNone(S.maybe_apply_sm12x_w4a16(_layer(), torch.zeros(1, 128, dtype=torch.bfloat16), None, "cutlass"))

    def test_small_m_goes_to_cute_dsl_native_with_weight_scale_2(self):
        L = _layer(global_scale=0.001)
        out, rec = self._run(torch.zeros(4, 128, dtype=torch.bfloat16), L, value=16)
        self.assertEqual(tuple(out.shape), (4, 256))
        (call,) = rec
        self.assertEqual(call["backend"], "cute-dsl-native")
        self.assertIs(call["b"], L.weight)
        self.assertIs(call["sf"], L.weight_scale_interleaved)
        self.assertEqual(call["alpha"].dtype, torch.float32)
        self.assertEqual(tuple(call["alpha"].shape), (1,))
        self.assertAlmostEqual(float(call["alpha"][0]), 0.001, places=7)

    def test_alpha_without_global_scale_strips_the_activation_scale(self):
        # alpha = input_scale * weight_scale_2, input_scale_inv = 1/input_scale
        L = _layer()
        _, rec = self._run(torch.zeros(1, 128, dtype=torch.bfloat16), L, value=8)
        self.assertAlmostEqual(float(rec[0]["alpha"][0]), 0.5 * 4.0, places=6)

    def test_alpha_cache_follows_an_in_place_rewrite(self):
        L = _layer(global_scale=0.001)
        a1 = S.w4a16_alpha(L)
        self.assertIs(S.w4a16_alpha(L), a1)
        L.weight_global_scale.data.copy_(torch.tensor(0.002))  # flip / reload rewrites in place
        self.assertAlmostEqual(float(S.w4a16_alpha(L)[0]), 0.002, places=7)

    def test_large_m_stays_w4a4(self):
        out, rec = self._run(torch.zeros(64, 128, dtype=torch.bfloat16), _layer(), value=32)
        self.assertIsNone(out)
        self.assertEqual(rec, [])

    def test_sm86_rank_never_takes_it(self):
        out, rec = self._run(torch.zeros(1, 128, dtype=torch.bfloat16), _layer(), value=32, major=8)
        self.assertIsNone(out)
        self.assertEqual(rec, [])

    def test_missing_kernel_on_sm12x_is_a_hard_error(self):
        with self.assertRaisesRegex(RuntimeError, "native_bf16_fp4"):
            self._run(torch.zeros(1, 128, dtype=torch.bfloat16), _layer(), value=32, present=False)

    def test_non_native_layouts_and_inputs_stand_down(self):
        x = torch.zeros(1, 128, dtype=torch.bfloat16)
        for backend in ("flashinfer_trtllm", "marlin", "w4a8_int8"):
            self.assertIsNone(self._run(x, _layer(), backend=backend, value=32)[0], backend)
        self.assertIsNone(self._run(x, _layer(pad_cols=32), value=32)[0])
        freed = _layer()
        freed.weight.data = torch.empty(0, dtype=torch.uint8)
        self.assertIsNone(self._run(x, freed, value=32)[0])
        self.assertIsNone(self._run((x, x), _layer(), value=32)[0])  # prequantized tuple
        self.assertIsNone(self._run(x.to(torch.float16), _layer(), value=32)[0])

    def test_n_padding_is_sliced_and_bias_added(self):
        L = _layer(n=256, out_size=250)
        rec = []
        with _Env(16), mock.patch.dict(sys.modules, {"flashinfer": _fake_flashinfer(rec)}):
            out = S.maybe_apply_sm12x_w4a16(L, torch.zeros(2, 128, dtype=torch.bfloat16), torch.full((250,), 2.0, dtype=torch.bfloat16), "cutlass")
        self.assertEqual(tuple(out.shape), (2, 250))
        self.assertTrue(torch.all(out == 3))

    def test_bad_env_value_is_loud(self):
        with self.assertRaises(ValueError):
            with _Env("abc"):
                S.sm12x_w4a16_max_m()


class TestModelOptHook(unittest.TestCase):
    def test_apply_calls_the_seam_before_w4a4(self):
        import inspect

        from sglang.srt.layers.quantization import modelopt_quant as MQ

        src = inspect.getsource(MQ.ModelOptFp4LinearMethod.apply)
        self.assertIn("maybe_apply_sm12x_w4a16", src)
        self.assertLess(src.index("maybe_apply_sm12x_w4a16"), src.index("fp4_quantize(x"))
        self.assertLess(src.index("is_marlin()"), src.index("maybe_apply_sm12x_w4a16"))


if __name__ == "__main__":
    unittest.main()
