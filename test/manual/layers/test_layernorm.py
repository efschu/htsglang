import itertools
import unittest
from unittest import mock

import torch

from sglang.srt.layers import layernorm as layernorm_mod
from sglang.srt.layers.layernorm import (
    Gemma3RMSNorm,
    GemmaRMSNorm,
    LayerNorm,
    RMSNorm,
)
from sglang.test.test_utils import CustomTestCase


class TestRMSNorm(CustomTestCase):
    DTYPES = [torch.half, torch.bfloat16]
    NUM_TOKENS = [7, 83, 4096]
    HIDDEN_SIZES = [768, 769, 770, 771, 5120, 5124, 5125, 5126, 8192, 8199]
    ADD_RESIDUAL = [False, True]
    SEEDS = [0]

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        torch.set_default_device("cuda")

    def _run_rms_norm_test(self, num_tokens, hidden_size, add_residual, dtype, seed):
        torch.manual_seed(seed)

        layer = RMSNorm(hidden_size).to(dtype=dtype)
        layer.weight.data.normal_(mean=1.0, std=0.1)
        scale = 1 / (2 * hidden_size)
        x = torch.randn(num_tokens, hidden_size, dtype=dtype) * scale
        residual = torch.randn_like(x) * scale if add_residual else None

        with torch.inference_mode():
            ref_out = layer.forward_native(x, residual)
            out = layer(x, residual)

        if add_residual:
            self.assertTrue(torch.allclose(out[0], ref_out[0], atol=1e-2, rtol=1e-2))
            self.assertTrue(torch.allclose(out[1], ref_out[1], atol=1e-2, rtol=1e-2))
        else:
            self.assertTrue(torch.allclose(out, ref_out, atol=1e-2, rtol=1e-2))

    def test_rms_norm(self):
        for params in itertools.product(
            self.NUM_TOKENS,
            self.HIDDEN_SIZES,
            self.ADD_RESIDUAL,
            self.DTYPES,
            self.SEEDS,
        ):
            with self.subTest(
                num_tokens=params[0],
                hidden_size=params[1],
                add_residual=params[2],
                dtype=params[3],
                seed=params[4],
            ):
                self._run_rms_norm_test(*params)


class TestGemmaRMSNorm(CustomTestCase):
    DTYPES = [torch.half, torch.bfloat16]
    NUM_TOKENS = [7, 83, 4096]
    HIDDEN_SIZES = [768, 769, 770, 771, 5120, 5124, 5125, 5126, 8192, 8199]
    ADD_RESIDUAL = [False, True]
    SEEDS = [0]

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        torch.set_default_device("cuda")

    def _run_gemma_rms_norm_test(
        self, num_tokens, hidden_size, add_residual, dtype, seed
    ):
        torch.manual_seed(seed)

        layer = GemmaRMSNorm(hidden_size).to(dtype=dtype)
        layer.weight.data.normal_(mean=1.0, std=0.1)
        scale = 1 / (2 * hidden_size)
        x = torch.randn(num_tokens, hidden_size, dtype=dtype) * scale
        residual = torch.randn_like(x) * scale if add_residual else None

        with torch.inference_mode():
            ref_out = layer.forward_native(x, residual)
            out = layer(x, residual)

        if add_residual:
            self.assertTrue(torch.allclose(out[0], ref_out[0], atol=1e-3, rtol=1e-3))
            self.assertTrue(torch.allclose(out[1], ref_out[1], atol=1e-3, rtol=1e-3))
        else:
            self.assertTrue(torch.allclose(out, ref_out, atol=1e-3, rtol=1e-3))

    def test_gemma_rms_norm(self):
        for params in itertools.product(
            self.NUM_TOKENS,
            self.HIDDEN_SIZES,
            self.ADD_RESIDUAL,
            self.DTYPES,
            self.SEEDS,
        ):
            with self.subTest(
                num_tokens=params[0],
                hidden_size=params[1],
                add_residual=params[2],
                dtype=params[3],
                seed=params[4],
            ):
                self._run_gemma_rms_norm_test(*params)


class TestGemma3RMSNorm(CustomTestCase):
    """Covers Gemma3RMSNorm, whose CUDA path had no test.

    Includes the rank-3 shapes that q_norm/k_norm receive, non-contiguous
    inputs, and an fp32 weight against half-precision activations -- which is
    how the module is constructed when it is built outside the loader's
    `set_default_torch_dtype` context (`nn.Parameter(torch.zeros(dim))`).

    The upstream residual subtest is not carried over: this fork's
    Gemma3RMSNorm takes no residual argument and no caller passes one, so the
    fused-add path would be unreachable from serving.
    """

    DTYPES = [torch.half, torch.bfloat16]
    NUM_TOKENS = [1, 7, 83, 4096]
    HIDDEN_SIZES = [256, 768, 1152, 5120, 5126]
    SEEDS = [0]

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        torch.set_default_device("cuda")

    def _make(self, hidden_size, dtype, seed, weight_fp32):
        torch.manual_seed(seed)
        layer = Gemma3RMSNorm(hidden_size)
        layer.weight.data.normal_(mean=0.0, std=0.1)
        if not weight_fp32:
            layer.weight.data = layer.weight.data.to(dtype)
        return layer

    def _check(self, layer, x):
        with torch.inference_mode():
            ref_out = layer.forward_native(x)
            out = layer(x)
        self.assertEqual(out.shape, ref_out.shape)
        self.assertFalse(torch.isnan(out).any() or torch.isinf(out).any())
        self.assertTrue(torch.allclose(out, ref_out, atol=1e-2, rtol=1e-2))

    def test_gemma3_rms_norm(self):
        for num_tokens, hidden_size, dtype, seed, weight_fp32 in itertools.product(
            self.NUM_TOKENS,
            self.HIDDEN_SIZES,
            self.DTYPES,
            self.SEEDS,
            [True, False],
        ):
            with self.subTest(
                num_tokens=num_tokens,
                hidden_size=hidden_size,
                dtype=dtype,
                seed=seed,
                weight_fp32=weight_fp32,
            ):
                scale = 1 / (2 * hidden_size)
                x = torch.randn(num_tokens, hidden_size, dtype=dtype) * scale
                self._check(self._make(hidden_size, dtype, seed, weight_fp32), x)

    def test_gemma3_rms_norm_3d(self):
        """q_norm / k_norm are called with [tokens, heads, head_dim]."""
        for num_tokens, heads, head_dim, dtype, weight_fp32 in itertools.product(
            [1, 37], [1, 4, 8], [128, 256], self.DTYPES, [True, False]
        ):
            with self.subTest(
                num_tokens=num_tokens,
                heads=heads,
                head_dim=head_dim,
                dtype=dtype,
                weight_fp32=weight_fp32,
            ):
                scale = 1 / (2 * head_dim)
                x = torch.randn(num_tokens, heads, head_dim, dtype=dtype) * scale
                self._check(self._make(head_dim, dtype, 0, weight_fp32), x)

    def test_gemma3_rms_norm_non_contiguous(self):
        """A transposed or sliced view must not be assumed contiguous."""
        for dtype in self.DTYPES:
            head_dim = 256
            scale = 1 / (2 * head_dim)
            layer = self._make(head_dim, dtype, 0, weight_fp32=False)

            # [tokens, heads, head_dim] produced by transposing [heads, tokens, .]
            base = torch.randn(4, 37, head_dim, dtype=dtype) * scale
            transposed = base.transpose(0, 1)
            self.assertFalse(transposed.is_contiguous())
            with self.subTest(dtype=dtype, case="transposed"):
                self._check(layer, transposed)

            # a strided slice along the last dimension
            wide = torch.randn(37, 4, head_dim * 2, dtype=dtype) * scale
            sliced = wide[..., :head_dim]
            self.assertFalse(sliced.is_contiguous())
            with self.subTest(dtype=dtype, case="sliced"):
                self._check(layer, sliced)


class TestGemma3RMSNormDispatch(CustomTestCase):
    """CPU falsifier for Gemma3RMSNorm.forward_cuda's dispatch decisions.

    TestGemma3RMSNorm above needs a GPU, so on a CPU-only checkout nothing
    checks that the fused branch is entered under the right conditions, that
    the kernel is handed a 2-D contiguous tensor, or that the leading
    dimensions are restored. Those are properties of the dispatch, not of the
    kernel, so they are testable here: `gemma_rmsnorm` is a module-global,
    replaced with a torch stand-in that records what it was called with and
    computes the same function. The kernel's own numerics stay covered by the
    CUDA class.
    """

    def setUp(self):
        self.calls = []

        def fake_gemma_rmsnorm(x, weight, eps):
            self.calls.append(
                {
                    "shape": tuple(x.shape),
                    "contiguous": x.is_contiguous(),
                    "dtype": x.dtype,
                    "weight_dtype": weight.dtype,
                }
            )
            # Same semantics as the fused kernel: the weight multiply happens
            # in the activation dtype, not in fp32.
            var = x.float().pow(2).mean(-1, keepdim=True)
            normed = (x.float() * torch.rsqrt(var + eps)).to(x.dtype)
            return normed * (1.0 + weight)

        # `create=True`: the sgl_kernel import block is guarded by `_is_cuda`,
        # so on a CPU-only checkout the name does not exist at module scope at
        # all. Patching it in is exactly what makes this class runnable there.
        self._patches = [
            mock.patch.object(
                layernorm_mod, "gemma_rmsnorm", fake_gemma_rmsnorm, create=True
            ),
            mock.patch.object(layernorm_mod, "_has_sgl_rmsnorm", True),
        ]
        for p in self._patches:
            p.start()
            self.addCleanup(p.stop)

    @staticmethod
    def _layer(dim, dtype):
        torch.manual_seed(0)
        layer = Gemma3RMSNorm(dim)
        layer.weight.data.normal_(mean=0.0, std=0.1)
        layer.weight.data = layer.weight.data.to(dtype)
        return layer

    def test_high_rank_is_fused_and_shape_restored(self):
        """[1, s, h, head_dim] -- the shape gemma3_causal hands to q_norm."""
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                self.calls.clear()
                head_dim = 128
                layer = self._layer(head_dim, dtype)
                x = torch.randn(1, 37, 8, head_dim, dtype=dtype) / (2 * head_dim)

                out = layer.forward_cuda(x)

                self.assertEqual(len(self.calls), 1, "fused branch was not taken")
                self.assertEqual(self.calls[0]["shape"], (37 * 8, head_dim))
                self.assertEqual(out.shape, x.shape)
                torch.testing.assert_close(
                    out.float(), layer.forward_native(x).float(), atol=2e-2, rtol=2e-2
                )

    def test_two_dim_is_fused_without_reshape(self):
        layer = self._layer(256, torch.bfloat16)
        x = torch.randn(83, 256, dtype=torch.bfloat16) / 512

        out = layer.forward_cuda(x)

        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0]["shape"], (83, 256))
        self.assertEqual(out.shape, x.shape)

    def test_non_contiguous_input_reaches_kernel_contiguous(self):
        """A transposed or last-dim-sliced view must be made contiguous."""
        head_dim = 128
        layer = self._layer(head_dim, torch.bfloat16)

        base = torch.randn(8, 37, head_dim, dtype=torch.bfloat16) / (2 * head_dim)
        transposed = base.transpose(0, 1)
        self.assertFalse(transposed.is_contiguous())
        out = layer.forward_cuda(transposed)
        self.assertTrue(self.calls[-1]["contiguous"])
        self.assertEqual(out.shape, transposed.shape)
        torch.testing.assert_close(
            out.float(),
            layer.forward_native(transposed).float(),
            atol=2e-2,
            rtol=2e-2,
        )

        wide = torch.randn(37, 4, head_dim * 2, dtype=torch.bfloat16) / (2 * head_dim)
        sliced = wide[..., :head_dim]
        self.assertFalse(sliced.is_contiguous())
        out = layer.forward_cuda(sliced)
        self.assertTrue(self.calls[-1]["contiguous"])
        self.assertEqual(out.shape, sliced.shape)
        torch.testing.assert_close(
            out.float(), layer.forward_native(sliced).float(), atol=2e-2, rtol=2e-2
        )

    def test_guards_fall_back_to_native(self):
        """Every guard clause must keep the kernel out of the call path.

        The fused kernel returns NaN rather than raising on a dtype mismatch,
        so "falls back" is the only safe behaviour, and a missing guard is
        silent.
        """
        cases = {}

        layer = self._layer(256, torch.bfloat16)
        layer.weight.data = layer.weight.data.float()
        cases["fp32 weight, bf16 activations"] = (
            layer,
            torch.randn(8, 256, dtype=torch.bfloat16) / 512,
        )

        layer = self._layer(256, torch.bfloat16)
        cases["fp32 activations"] = (layer, torch.randn(8, 256) / 512)

        for name, (layer, x) in cases.items():
            with self.subTest(case=name):
                self.calls.clear()
                out = layer.forward_cuda(x)
                self.assertEqual(self.calls, [], "fused kernel must not be reached")
                torch.testing.assert_close(out, layer.forward_native(x))

    def test_width_mismatch_does_not_reach_the_kernel(self):
        """A last dim that is not the norm width must not enter the kernel.

        The kernel takes the width from the weight, so it would read past the
        row; the eager path raises instead. Loud, not silent.
        """
        layer = self._layer(256, torch.bfloat16)
        x = torch.randn(8, 128, dtype=torch.bfloat16) / 512

        with self.assertRaises(RuntimeError):
            layer.forward_cuda(x)
        self.assertEqual(self.calls, [], "fused kernel must not be reached")

    def test_missing_sgl_kernel_falls_back(self):
        """Below sm_80 sgl_kernel is absent and gemma_rmsnorm is None.

        RMSNorm and Gemma4RMSNorm already guard on `_has_sgl_rmsnorm`;
        Gemma3RMSNorm is the third sibling in the same file and was the one
        left out.
        """
        layer = self._layer(256, torch.bfloat16)
        x = torch.randn(8, 256, dtype=torch.bfloat16) / 512
        with (
            mock.patch.object(layernorm_mod, "_has_sgl_rmsnorm", False),
            mock.patch.object(layernorm_mod, "gemma_rmsnorm", None, create=True),
        ):
            out = layer.forward_cuda(x)
        torch.testing.assert_close(out, layer.forward_native(x))


class TestLayerNorm(CustomTestCase):
    DTYPES = [torch.half, torch.bfloat16]
    PARAM_DTYPES = [torch.bfloat16, torch.float32]
    NUM_TOKENS = [7, 83, 1024]
    HIDDEN_SIZES = [128, 512, 1536, 5120, 5124, 5125, 5126, 7168]
    USE_AFFINE = [False, True]
    USE_BIAS = [False, True]
    SEEDS = [0]

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        torch.set_default_device("cuda")

    def _run_layer_norm_test(
        self, num_tokens, hidden_size, use_affine, use_bias, dtype, seed, param_dtype
    ):
        torch.manual_seed(seed)

        layer = LayerNorm(
            hidden_size, elementwise_affine=use_affine, bias=use_bias, dtype=param_dtype
        )
        if use_affine:
            layer.weight.data.normal_(mean=1.0, std=0.1)
            if use_bias:
                layer.bias.data.normal_(mean=0.0, std=0.1)

        scale = 1 / (2 * hidden_size)
        x = torch.randn(num_tokens, hidden_size, dtype=dtype) * scale

        with torch.inference_mode():
            ref_out = layer.forward_native(x)
            out = layer(x)

        self.assertTrue(torch.allclose(out, ref_out, atol=1e-2, rtol=1e-3))

        if (
            use_affine
            and use_bias
            and not (dtype == torch.bfloat16 and param_dtype == torch.float32)
        ):
            layer.dtype = torch.float32
            layer.weight.data = layer.weight.data.to(torch.float32)
            layer.bias.data = layer.bias.data.to(torch.float32)
            with torch.inference_mode():
                cuda_out = layer(x.to(torch.bfloat16)).to(x.dtype)

            self.assertTrue(torch.allclose(cuda_out, ref_out, atol=2e-2, rtol=1e-3))

    def test_layer_norm(self):
        for params in itertools.product(
            self.NUM_TOKENS,
            self.HIDDEN_SIZES,
            self.USE_AFFINE,
            self.USE_BIAS,
            self.DTYPES,
            self.SEEDS,
            self.PARAM_DTYPES,
        ):
            with self.subTest(
                num_tokens=params[0],
                hidden_size=params[1],
                use_affine=params[2],
                use_bias=params[3],
                dtype=params[4],
                seed=params[5],
                param_dtype=params[6],
            ):
                self._run_layer_norm_test(*params)


if __name__ == "__main__":
    unittest.main(verbosity=2)
