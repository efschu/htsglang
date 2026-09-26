"""``--fp4-gemm-backend native-mixed`` (Backlog #38): one native NVFP4 byte
layout on every rank, the kernel chosen per rank by compute capability.

Contract: docs/NVFP4_NATIVE_LAYOUT_CONTRACT.md. Pure functions, mocked device
capabilities, CPU tensors; no GPU, no server, no checkpoint.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import contextlib
import unittest
from unittest import mock

import torch

from sglang.srt.layers.quantization import fp4_utils
from sglang.srt.layers.quantization import nvfp4_native_mixed as nm
from sglang.srt.layers.quantization.fp4_utils import Fp4GemmRunnerBackend
from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptFp4Config,
    ModelOptFp4LinearMethod,
)
from sglang.test.test_utils import CustomTestCase


@contextlib.contextmanager
def _fp4_state():
    saved = (
        fp4_utils.FP4_GEMM_RUNNER_BACKEND,
        fp4_utils.FP4_NATIVE_MIXED,
        fp4_utils.FP4_NATIVE_MIXED_SHARED_LAYOUT,
        nm._W4A8_KERNEL,
        nm._W4A8_KERNEL_NAME,
    )
    try:
        yield
    finally:
        (
            fp4_utils.FP4_GEMM_RUNNER_BACKEND,
            fp4_utils.FP4_NATIVE_MIXED,
            fp4_utils.FP4_NATIVE_MIXED_SHARED_LAYOUT,
            nm._W4A8_KERNEL,
            nm._W4A8_KERNEL_NAME,
        ) = saved


def _fake_w4a8(x, weight, scale, gscale, n):
    return torch.zeros(x.shape[0], n, dtype=x.dtype)


def _resolve_rig(backend, caps, *, kernel=True, sm8x="w4a8", allow_broken=None, sm12x=None):
    """One scheduler process per rank: TP0 = 5090, TP1/TP2 = 3080. ``sm8x=None``: the env default.
    ``allow_broken`` / ``sm12x`` None: the env default of SGLANG_FP4_ALLOW_BROKEN_SM8X_MARLIN /
    SGLANG_FP4_NATIVE_MIXED_SM12X."""
    args = mock.Mock(fp4_gemm_runner_backend=backend)
    out = []
    def env_patch():
        if sm8x is None:
            return contextlib.nullcontext()
        return mock.patch("sglang.srt.environ.envs.SGLANG_FP4_NATIVE_MIXED_SM8X.get", return_value=sm8x)

    def allow_patch():
        if allow_broken is None:
            return contextlib.nullcontext()
        return mock.patch(
            "sglang.srt.environ.envs.SGLANG_FP4_ALLOW_BROKEN_SM8X_MARLIN.get", return_value=allow_broken
        )

    def sm12x_patch():
        if sm12x is None:
            return contextlib.nullcontext()
        return mock.patch("sglang.srt.environ.envs.SGLANG_FP4_NATIVE_MIXED_SM12X.get", return_value=sm12x)

    for cap in caps:
        with (
            env_patch(),
            allow_patch(),
            sm12x_patch(),
            mock.patch("sglang.srt.utils.common.get_device_capability", return_value=cap),
            mock.patch.object(fp4_utils, "get_device_capability", return_value=cap),
            mock.patch.object(fp4_utils, "is_sm100_supported", return_value=cap[0] == 10),
            mock.patch.object(fp4_utils, "is_sm120_supported", return_value=cap[0] == 12),
            mock.patch.object(fp4_utils, "is_cuda", return_value=True),
            mock.patch.object(fp4_utils, "has_fork_nvfp4_cutlass_kernel", return_value=True),
            mock.patch.object(nm, "_try_autoload_w4a8_kernel", return_value=None),
            mock.patch.object(nm, "_flashinfer_fp4_gemm_available", return_value=True),
        ):
            if kernel:
                nm.register_w4a8_kernel(_fake_w4a8, "fake")
            else:
                nm.unregister_w4a8_kernel()
            fp4_utils.initialize_fp4_gemm_config(args)
            out.append(
                (
                    fp4_utils.get_fp4_gemm_runner_backend(),
                    fp4_utils.is_fp4_native_mixed(),
                    fp4_utils.is_fp4_native_mixed_shared_layout(),
                )
            )
    return out


RIG = [(12, 0), (8, 6), (8, 6)]


class TestRankResolution(CustomTestCase):
    def test_pure_table(self):
        c = nm.resolve_rank_backend((12, 0), w4a8_available=False)
        self.assertEqual((c.backend, c.shared_layout), ("cutlass", True))
        # F bench p6vrwg: FlashInfer CUTLASS when importable (same bytes), else fork
        c = nm.resolve_rank_backend(
            (12, 0), w4a8_available=False, sm12x_choice="flashinfer_cutlass",
            flashinfer_fp4_available=True,
        )
        self.assertEqual((c.backend, c.shared_layout), ("flashinfer_cutlass", True))
        c = nm.resolve_rank_backend(
            (12, 0), w4a8_available=False, sm12x_choice="flashinfer_cutlass",
            flashinfer_fp4_available=False,
        )
        self.assertEqual((c.backend, c.shared_layout), ("cutlass", True))
        self.assertIn("not importable", c.reason)
        with self.assertRaises(nm.NativeMixedUnsupported):
            nm.resolve_rank_backend((12, 0), w4a8_available=False, sm12x_choice="b12x")
        # sm_8x default (user order 25.09. ~17:33Z): W4A8 on the INT8 tensor cores, native bytes
        c = nm.resolve_rank_backend((8, 6), w4a8_available=True)
        self.assertEqual((c.backend, c.shared_layout), ("w4a8_int8", True))
        c = nm.resolve_rank_backend((8, 6), w4a8_available=True, sm8x_choice="w4a8")
        self.assertEqual((c.backend, c.shared_layout), ("w4a8_int8", True))
        # no silent fallback to Marlin when the kernel is missing
        with self.assertRaisesRegex(nm.NativeMixedUnsupported, "W4A8 INT8 kernel"):
            nm.resolve_rank_backend((8, 6), w4a8_available=False)
        # Marlin W4A16 in place only on explicit request -- and since 26.09. only with the
        # diagnosis override (NVFP4-SM8X-MARLIN-GUARD, TestSm8xMarlinGuard below)
        c = nm.resolve_rank_backend(
            (8, 6), w4a8_available=True, sm8x_choice="marlin", allow_broken_sm8x_marlin=True
        )
        self.assertEqual((c.backend, c.shared_layout), ("marlin_native_inplace", True))
        c = nm.resolve_rank_backend(
            (8, 6), w4a8_available=False, sm8x_choice="marlin", allow_broken_sm8x_marlin=True
        )
        self.assertEqual(c.backend, "marlin_native_inplace")
        c = nm.resolve_rank_backend((10, 0), w4a8_available=False)
        self.assertEqual(c.backend, "flashinfer_cutedsl")

    def test_refusals_are_named(self):
        with self.assertRaisesRegex(nm.NativeMixedUnsupported, "W4A8 INT8 kernel"):
            nm.resolve_rank_backend((8, 6), w4a8_available=False, sm8x_choice="w4a8")
        with self.assertRaisesRegex(nm.NativeMixedUnsupported, "neither 'marlin' nor 'w4a8'"):
            nm.resolve_rank_backend((8, 6), w4a8_available=True, sm8x_choice="int4")
        with self.assertRaisesRegex(nm.NativeMixedUnsupported, "7.5"):
            nm.resolve_rank_backend((7, 5), w4a8_available=True)
        with self.assertRaisesRegex(nm.NativeMixedUnsupported, "CUTLASS"):
            nm.resolve_rank_backend((12, 0), w4a8_available=True, native_sm120_available=False)

    def test_mixed_rig_native_mixed_w4a8(self):
        with _fp4_state():
            got = _resolve_rig("native-mixed", RIG, sm8x="w4a8")
        self.assertEqual(
            got,
            [
                (Fp4GemmRunnerBackend.FLASHINFER_CUTLASS, True, True),
                (Fp4GemmRunnerBackend.W4A8_INT8, True, True),
                (Fp4GemmRunnerBackend.W4A8_INT8, True, True),
            ],
        )

    def test_mixed_rig_default_is_w4a8(self):
        """Env unset: 5090 native W4A4, both 3080 W4A8 (user order 25.09.)."""
        with _fp4_state(), mock.patch.dict("os.environ", {}, clear=False):
            import os as _os

            _os.environ.pop("SGLANG_FP4_NATIVE_MIXED_SM8X", None)
            got = _resolve_rig("native-mixed", RIG, sm8x=None)
        self.assertEqual(
            got,
            [
                (Fp4GemmRunnerBackend.FLASHINFER_CUTLASS, True, True),
                (Fp4GemmRunnerBackend.W4A8_INT8, True, True),
                (Fp4GemmRunnerBackend.W4A8_INT8, True, True),
            ],
        )

    def test_mixed_rig_marlin_inplace_is_opt_in(self):
        with _fp4_state():
            got = _resolve_rig("native-mixed", RIG, kernel=False, sm8x="marlin", allow_broken=True)
        self.assertEqual(
            got,
            [
                (Fp4GemmRunnerBackend.FLASHINFER_CUTLASS, True, True),
                (Fp4GemmRunnerBackend.MARLIN_NATIVE_INPLACE, True, True),
                (Fp4GemmRunnerBackend.MARLIN_NATIVE_INPLACE, True, True),
            ],
        )

    def test_mixed_rig_w4a8_without_kernel_refuses(self):
        with _fp4_state(), self.assertRaises(nm.NativeMixedUnsupported):
            _resolve_rig("native-mixed", RIG, kernel=False, sm8x="w4a8")

    def test_default_paths_unchanged(self):
        """auto / marlin resolve exactly as before and leave the mode off."""
        with _fp4_state():
            got = _resolve_rig("auto", RIG)
            self.assertEqual(
                [g[0] for g in got],
                [Fp4GemmRunnerBackend.CUTLASS, Fp4GemmRunnerBackend.MARLIN, Fp4GemmRunnerBackend.MARLIN],
            )
            self.assertTrue(all(not g[1] for g in got))
            got = _resolve_rig("marlin", RIG)
            self.assertEqual({g[0] for g in got}, {Fp4GemmRunnerBackend.MARLIN})
            self.assertTrue(all(not g[1] for g in got))

    def test_cli_choice_registered(self):
        from sglang.srt.server_args import FP4_GEMM_RUNNER_BACKEND_CHOICES

        self.assertIn("native-mixed", FP4_GEMM_RUNNER_BACKEND_CHOICES)
        self.assertNotIn("w4a8_int8", FP4_GEMM_RUNNER_BACKEND_CHOICES)
        self.assertNotIn("marlin_native_inplace", FP4_GEMM_RUNNER_BACKEND_CHOICES)


def _swizzle_reference(s):
    """The exact expression of modelopt_quant.py process_weights_after_loading."""
    scales = s.unsqueeze(0)
    B, M, K = scales.shape
    Mp, Kp = (M + 127) // 128 * 128, (K + 3) // 4 * 4
    p = torch.zeros((B, Mp, Kp), dtype=scales.dtype)
    p[:B, :M, :K] = scales
    p = p.reshape(B, Mp // 128, 4, 32, Kp // 4, 4).permute((0, 1, 4, 3, 2, 5))
    return p.contiguous().reshape(Mp, Kp)


def _kernel_offset(m, kb, kp):
    """nvfp4_quant.cuh cvt_quant_to_fp4_get_sf_out_offset."""
    return (m // 128) * (kp // 4) * 512 + (kb // 4) * 512 + (m % 32) * 16 + ((m % 128) // 32) * 4 + kb % 4


class TestSm8xMarlinGuard(CustomTestCase):
    """NVFP4-SM8X-MARLIN-GUARD (26.09.): boot dkr27bnvfp4bar1marlin09261251 (d98b3ba08a, arm
    n4old = SGLANG_FP4_NATIVE_MIXED_SM8X=marlin, 5090 on flashinfer_cutlass) served wrong
    tokens. The in-place Marlin rank is refused unless SGLANG_FP4_ALLOW_BROKEN_SM8X_MARLIN=1."""

    def test_env_default_is_off(self):
        from sglang.srt.environ import envs

        with mock.patch.dict("os.environ", {}, clear=False):
            import os as _os

            _os.environ.pop(nm.SM8X_MARLIN_ALLOW_ENV, None)
            self.assertFalse(envs.SGLANG_FP4_ALLOW_BROKEN_SM8X_MARLIN.get())
        self.assertEqual(nm.SM8X_MARLIN_ALLOW_ENV, "SGLANG_FP4_ALLOW_BROKEN_SM8X_MARLIN")

    def test_guard_refuses_marlin_by_default(self):
        for kernel in (True, False):
            with self.assertRaisesRegex(nm.NativeMixedUnsupported, nm.SM8X_MARLIN_GUARD) as cm:
                nm.resolve_rank_backend((8, 6), w4a8_available=kernel, sm8x_choice="marlin")
            msg = str(cm.exception)
            self.assertIn("SGLANG_FP4_ALLOW_BROKEN_SM8X_MARLIN=1", msg)
            self.assertIn("w4a8", msg)
        # case/whitespace variants of the value hit the same guard
        with self.assertRaisesRegex(nm.NativeMixedUnsupported, nm.SM8X_MARLIN_GUARD):
            nm.resolve_rank_backend((8, 9), w4a8_available=True, sm8x_choice=" Marlin ")

    def test_guard_refuses_on_the_rig_via_env(self):
        with _fp4_state(), self.assertRaisesRegex(nm.NativeMixedUnsupported, nm.SM8X_MARLIN_GUARD):
            _resolve_rig("native-mixed", RIG, kernel=True, sm8x="marlin")
        with _fp4_state(), self.assertRaisesRegex(nm.NativeMixedUnsupported, nm.SM8X_MARLIN_GUARD):
            _resolve_rig("native-mixed", RIG, kernel=True, sm8x="marlin", allow_broken=False)

    def test_override_lets_it_run_and_warns(self):
        with self.assertLogs(nm.logger, level="WARNING") as logs:
            c = nm.resolve_rank_backend(
                (8, 6), w4a8_available=True, sm8x_choice="marlin", allow_broken_sm8x_marlin=True
            )
        self.assertEqual((c.backend, c.shared_layout), ("marlin_native_inplace", True))
        self.assertTrue(any(nm.SM8X_MARLIN_GUARD in r and "KNOWN-BROKEN" in r for r in logs.output))
        with _fp4_state():
            got = _resolve_rig("native-mixed", RIG, kernel=True, sm8x="marlin", allow_broken=True)
        self.assertEqual(
            [g[0] for g in got],
            [
                Fp4GemmRunnerBackend.FLASHINFER_CUTLASS,
                Fp4GemmRunnerBackend.MARLIN_NATIVE_INPLACE,
                Fp4GemmRunnerBackend.MARLIN_NATIVE_INPLACE,
            ],
        )

    def test_other_values_unchanged(self):
        # the override changes nothing for any other value or arch
        for allow in (False, True):
            c = nm.resolve_rank_backend(
                (8, 6), w4a8_available=True, sm8x_choice="w4a8", allow_broken_sm8x_marlin=allow
            )
            self.assertEqual(c.backend, "w4a8_int8")
            with self.assertRaisesRegex(nm.NativeMixedUnsupported, "neither 'marlin' nor 'w4a8'"):
                nm.resolve_rank_backend(
                    (8, 6), w4a8_available=True, sm8x_choice="int4", allow_broken_sm8x_marlin=allow
                )
            # the 5090 never sees the SM8X value
            c = nm.resolve_rank_backend(
                (12, 0), w4a8_available=True, sm8x_choice="marlin", sm12x_choice="flashinfer_cutlass",
                flashinfer_fp4_available=True, allow_broken_sm8x_marlin=allow,
            )
            self.assertEqual(c.backend, "flashinfer_cutlass")
            c = nm.resolve_rank_backend(
                (12, 0), w4a8_available=True, sm8x_choice="marlin", sm12x_choice="cutlass",
                allow_broken_sm8x_marlin=allow,
            )
            self.assertEqual(c.backend, "cutlass")
            c = nm.resolve_rank_backend(
                (10, 0), w4a8_available=False, sm8x_choice="marlin", allow_broken_sm8x_marlin=allow
            )
            self.assertEqual(c.backend, "flashinfer_cutedsl")
        with _fp4_state():
            got = _resolve_rig("native-mixed", RIG, kernel=True, sm8x=None, allow_broken=None)
        self.assertEqual(
            [g[0] for g in got],
            [
                Fp4GemmRunnerBackend.FLASHINFER_CUTLASS,
                Fp4GemmRunnerBackend.W4A8_INT8,
                Fp4GemmRunnerBackend.W4A8_INT8,
            ],
        )

    def test_sm12x_marlin_refusal_is_named_and_not_overridable(self):
        for allow in (False, True):
            with self.assertRaisesRegex(nm.NativeMixedUnsupported, "SM12X='marlin' does not exist") as cm:
                nm.resolve_rank_backend(
                    (12, 0), w4a8_available=True, sm12x_choice="marlin", allow_broken_sm8x_marlin=allow
                )
            self.assertIn(nm.SM8X_MARLIN_GUARD, str(cm.exception))
        with _fp4_state(), self.assertRaisesRegex(nm.NativeMixedUnsupported, "does not exist"):
            _resolve_rig("native-mixed", RIG[:1], kernel=True, sm12x="marlin", allow_broken=True)


class TestSwizzleAndSlicing(CustomTestCase):
    def setUp(self):
        self.g = torch.Generator().manual_seed(7)

    def _rand(self, n, k):
        return torch.randint(0, 256, (n, k), dtype=torch.uint8, generator=self.g)

    def test_helper_matches_the_loader_and_the_kernel(self):
        s = self._rand(300, 22)  # both paddings active
        sw = nm.swizzle_128x4(s)
        self.assertTrue(torch.equal(sw, _swizzle_reference(s)))
        flat = sw.reshape(-1)
        kp = sw.shape[1]
        for m, kb in ((0, 0), (5, 3), (127, 21), (128, 4), (299, 17), (37, 9)):
            self.assertEqual(int(flat[_kernel_offset(m, kb, kp)]), int(s[m, kb]))

    def test_k_shard_is_a_column_slice_of_the_tile_view(self):
        """down_proj under D 73/32/31 units of 128: K/16 = 584/256/248."""
        full = self._rand(5120, 1088)
        F = nm.sf_tile_view(nm.swizzle_128x4(full))
        kb0 = 0
        for w in (584, 256, 248):
            shard = nm.swizzle_128x4(full[:, kb0 : kb0 + w])
            self.assertTrue(
                torch.equal(nm.sf_tile_view(shard), F[:, kb0 * 128 : (kb0 + w) * 128])
            )
            kb0 += w

    def test_plain_column_slice_is_wrong(self):
        """The danger direction: a plain COLS cut of the swizzled scale."""
        full = self._rand(5120, 1088)
        F = nm.swizzle_128x4(full)
        self.assertFalse(torch.equal(nm.swizzle_128x4(full[:, :584]), F[:, :584]))

    def test_fused_row_shard_is_layout_neutral(self):
        """gate_up [gate | up], I = 17408, D rows 9344 / 4096 / 3968."""
        full = self._rand(2 * 17408, 320)
        F = nm.swizzle_128x4(full)
        I, a = 17408, 0
        for w in (9344, 4096, 3968):
            shard = nm.swizzle_128x4(torch.cat([full[a : a + w], full[I + a : I + a + w]]))
            self.assertTrue(torch.equal(shard, torch.cat([F[a : a + w], F[I + a : I + a + w]])))
            a += w

    def test_27b_shards_need_no_padding(self):
        for name, n, k, comps in (
            ("P.gate_up", 34816, 5120, (17408, 17408)),
            ("P.down", 5120, 17408, ()),
            ("D.gate_up", 18688, 5120, (9344, 9344)),
            ("D.gate_up.r2", 7936, 5120, (3968, 3968)),
            ("D.down", 5120, 9344, ()),
            ("D.down.r2", 5120, 3968, ()),
            ("lm_head.D", 82816, 5120, ()),
            ("lm_head.P", 248320, 5120, ()),
        ):
            nm.check_shard_alignment(name, n, k, comps)

    def test_misaligned_shards_are_refused(self):
        with self.assertRaisesRegex(nm.NativeMixedUnsupported, "128x4"):
            nm.check_shard_alignment("x", 9300, 5120)
        with self.assertRaisesRegex(nm.NativeMixedUnsupported, "128x4"):
            nm.check_shard_alignment("x", 5120, 9344 + 16)  # K/16 not % 4
        with self.assertRaisesRegex(nm.NativeMixedUnsupported, "straddle"):
            nm.check_shard_alignment("x", 18688, 5120, (9344 + 64, 9344 - 64))


def _make_layer(method, n_parts, n_each, k):
    layer = torch.nn.Module()
    method.create_weights(layer, k, [n_each] * n_parts, k, n_each * n_parts, torch.bfloat16, weight_loader=None)
    g = torch.Generator().manual_seed(1)
    layer.weight.data.copy_(torch.randint(0, 256, layer.weight.shape, dtype=torch.uint8, generator=g))
    raw = torch.randint(0, 120, layer.weight_scale.shape, dtype=torch.uint8, generator=g).view(torch.float8_e4m3fn)
    layer.weight_scale.data.copy_(raw)
    layer.input_scale.data.fill_(0.02)
    layer.weight_scale_2.data.fill_(0.003)
    return layer, raw.clone()


class TestLoaderUnderNativeMixed(CustomTestCase):
    """process_weights_after_loading on a (mocked) sm_86 W4A8 rank, on CPU."""

    def _run(self, n_parts, n_each, k):
        method = ModelOptFp4LinearMethod(ModelOptFp4Config(is_checkpoint_nvfp4_serialized=True, group_size=16))
        layer, raw = _make_layer(method, n_parts, n_each, k)
        with (
            _fp4_state(),
            mock.patch.object(torch.Tensor, "cuda", lambda self, *a, **kw: self),
            mock.patch(
                "sglang.srt.layers.quantization.modelopt_quant.is_blackwell_supported",
                return_value=False,
            ),
        ):
            fp4_utils.FP4_GEMM_RUNNER_BACKEND = Fp4GemmRunnerBackend.W4A8_INT8
            fp4_utils.FP4_NATIVE_MIXED = True
            fp4_utils.FP4_NATIVE_MIXED_SHARED_LAYOUT = True
            method.process_weights_after_loading(layer)
            nm.register_w4a8_kernel(_fake_w4a8, "fake")
            out = method.apply(layer, torch.randn(3, k, dtype=torch.bfloat16))
        return layer, raw, out

    def test_w4a8_rank_keeps_the_native_layout(self):
        layer, raw, out = self._run(2, 256, 512)
        # the swizzled bytes live under the name weight_scale (alias, no pad)
        self.assertIs(layer.weight_scale_interleaved, layer.weight_scale)
        self.assertTrue(
            torch.equal(layer.weight_scale.view(torch.uint8), nm.swizzle_128x4(raw.view(torch.uint8)))
        )
        self.assertTrue(nm.is_swizzled(layer.weight_scale))
        self.assertEqual(tuple(layer.weight.shape), (512, 256))
        self.assertAlmostEqual(float(layer.weight_global_scale), 0.003, places=6)
        self.assertAlmostEqual(float(layer.alpha), 0.02 * 0.003, places=7)
        names = [n for n, _ in layer.named_parameters()]
        self.assertEqual(
            names,
            ["weight", "input_scale", "weight_scale_2", "weight_scale", "alpha", "input_scale_inv",
             "weight_global_scale", "weight_global_scale_w4a16"],
        )
        self.assertEqual(tuple(out.shape), (3, 512))

    def test_misaligned_shard_fails_at_load(self):
        with self.assertRaises(nm.NativeMixedUnsupported):
            self._run(1, 200, 512)


class TestW4A8RankHasNoFlipReshape(CustomTestCase):
    """User order 25.09.: a W4A8 sm_8x rank computes on the native bytes, so the #38 L8 flip hooks
    (to_native before the deposit, to_marlin after the wake) must find nothing and touch nothing."""

    def _loaded(self, backend):
        method = ModelOptFp4LinearMethod(ModelOptFp4Config(is_checkpoint_nvfp4_serialized=True, group_size=16))
        layer, _ = _make_layer(method, 2, 256, 512)
        with (
            _fp4_state(),
            mock.patch.object(torch.Tensor, "cuda", lambda self, *a, **kw: self),
            mock.patch(
                "sglang.srt.layers.quantization.modelopt_quant.is_blackwell_supported",
                return_value=False,
            ),
        ):
            fp4_utils.FP4_GEMM_RUNNER_BACKEND = backend
            fp4_utils.FP4_NATIVE_MIXED = True
            fp4_utils.FP4_NATIVE_MIXED_SHARED_LAYOUT = True
            method.process_weights_after_loading(layer)
        model = torch.nn.Module()
        model.mlp = layer
        return model, layer

    def test_hooks_are_noops_on_a_w4a8_rank(self):
        from sglang.srt.layers.quantization import nvfp4_marlin_inplace as mi

        model, layer = self._loaded(Fp4GemmRunnerBackend.W4A8_INT8)
        self.assertFalse(getattr(layer, mi.LAYER_FLAG, False))
        self.assertEqual(mi.flagged_layers([model]), [])
        w0 = layer.weight.detach().clone()
        s0 = layer.weight_scale.detach().view(torch.uint8).clone()
        with mock.patch.object(mi, "layer_to_native") as to_nat, mock.patch.object(mi, "layer_to_marlin") as to_mar:
            self.assertEqual(mi.model_to_native([model]), 0)
            self.assertEqual(mi.model_to_marlin([model], delivered_native=True), 0)
            self.assertEqual(mi.model_to_marlin([model], delivered_native=False), 0)
        to_nat.assert_not_called()
        to_mar.assert_not_called()
        self.assertTrue(torch.equal(layer.weight, w0))
        self.assertTrue(torch.equal(layer.weight_scale.view(torch.uint8), s0))

    def test_weight_updater_hooks_do_nothing_on_a_w4a8_rank(self):
        """The two WeightUpdater methods the sleep/wake legs call, on a stub owner."""
        from sglang.srt.layers.quantization import nvfp4_marlin_inplace as mi
        from sglang.srt.managers.scheduler_components.weight_updater import (
            SchedulerWeightUpdaterManager as WeightUpdater,
        )

        model, layer = self._loaded(Fp4GemmRunnerBackend.W4A8_INT8)
        stub = mock.Mock()
        stub._weg2_wake_models.return_value = [model]
        stub._weg2_nvfp4_draft_disk_reloaded = True
        w0 = layer.weight.detach().clone()
        with mock.patch.object(mi, "layer_to_native") as to_nat, mock.patch.object(mi, "layer_to_marlin") as to_mar:
            WeightUpdater._weg2_nvfp4_marlin_to_native(stub)
            WeightUpdater._weg2_nvfp4_marlin_after_wake(stub)
        to_nat.assert_not_called()
        to_mar.assert_not_called()
        stub._weg2_wake_weight_carrier.assert_not_called()  # returned before asking for the carrier
        self.assertFalse(stub._weg2_nvfp4_draft_disk_reloaded)  # read-and-clear still happens
        self.assertTrue(torch.equal(layer.weight, w0))

    def test_marlin_opt_in_still_flags_the_layer(self):
        """Control: the same load on the opt-in Marlin rank IS flagged (the hooks have work there)."""
        from sglang.srt.layers.quantization import nvfp4_marlin_inplace as mi

        with mock.patch.object(mi, "prepare_layer", side_effect=lambda l, **kw: setattr(l, mi.LAYER_FLAG, True)):
            model, layer = self._loaded(Fp4GemmRunnerBackend.MARLIN_NATIVE_INPLACE)
        self.assertEqual(mi.flagged_layers([model]), [layer])


#: hf_quant_config.json of Qwen3.8-27B-DFlash2-NVFP4-RTNcal (HF maurienne-ai @ bd7a934213c4), verbatim.
DRAFT_HF_QUANT_CONFIG = {
    "producer": {"name": "modelopt", "version": "dflash2-nvfp4-rtn-calibrated-1.0"},
    "quantization": {
        "quant_algo": "NVFP4",
        "kv_cache_quant_algo": "FP8",
        "group_size": 16,
        "exclude_modules": [
            "candidate_selector.hidden_projection",
            "fc",
        ]
        + [f"layers.{i}.{m}.kernel_projection" for i in range(5) for m in ("attention_conv", "mlp_conv")],
    },
}


class TestNvfp4DraftTakesTheSameBackend(CustomTestCase):
    """User order 25.09.: main model AND NVFP4 draft on the same per-rank path (3080 W4A8, 5090 W4A4).
    The draft's config is modelopt NVFP4 (not MIXED_PRECISION): its linears get the SAME
    ModelOptFp4LinearMethod as the target's MLP, and that method dispatches on the ONE process-wide
    FP4 backend (fp4_utils, resolved once per scheduler process) -- no draft-specific branch exists."""

    def _draft_method(self, prefix):
        from sglang.srt.layers.linear import LinearBase

        cfg = ModelOptFp4Config.from_config(DRAFT_HF_QUANT_CONFIG)
        # set by the model loader from the model class (qwen3-style fused projections)
        cfg.packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"], "gate_up_proj": ["gate_proj", "up_proj"]}
        return cfg.get_quant_method(mock.Mock(spec=LinearBase), prefix)

    def test_draft_linears_are_modelopt_fp4(self):
        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

        for p in ("layers.0.mlp.gate_up_proj", "layers.0.mlp.down_proj", "layers.4.self_attn.qkv_proj",
                  "layers.2.self_attn.o_proj"):
            self.assertIsInstance(self._draft_method(p), ModelOptFp4LinearMethod, p)
        for p in ("fc", "layers.0.mlp_conv.kernel_projection"):
            self.assertIsInstance(self._draft_method(p), UnquantizedLinearMethod, p)

    def test_draft_layer_on_a_w4a8_rank_runs_the_w4a8_kernel_without_reshape(self):
        from sglang.srt.layers.quantization import nvfp4_marlin_inplace as mi

        method = self._draft_method("layers.0.mlp.gate_up_proj")
        layer, raw = _make_layer(method, 2, 256, 512)
        calls = []

        def spy(x, weight, scale, gscale, n):
            calls.append((tuple(x.shape), weight is layer.weight, scale is layer.weight_scale))
            return torch.zeros(x.shape[0], n, dtype=x.dtype)

        with (
            _fp4_state(),
            mock.patch.object(torch.Tensor, "cuda", lambda self, *a, **kw: self),
            mock.patch(
                "sglang.srt.layers.quantization.modelopt_quant.is_blackwell_supported",
                return_value=False,
            ),
        ):
            fp4_utils.FP4_GEMM_RUNNER_BACKEND = Fp4GemmRunnerBackend.W4A8_INT8
            fp4_utils.FP4_NATIVE_MIXED = True
            fp4_utils.FP4_NATIVE_MIXED_SHARED_LAYOUT = True
            method.process_weights_after_loading(layer)
            nm.register_w4a8_kernel(spy, "spy")
            out = method.apply(layer, torch.randn(8, 512, dtype=torch.bfloat16))
        self.assertEqual(calls, [((8, 512), True, True)])
        self.assertEqual(tuple(out.shape), (8, 512))
        self.assertTrue(torch.equal(layer.weight_scale.view(torch.uint8), nm.swizzle_128x4(raw.view(torch.uint8))))
        self.assertFalse(getattr(layer, mi.LAYER_FLAG, False))

    def test_draft_and_target_resolve_the_same_backend_per_rank(self):
        """One resolution per scheduler process; target and draft both read get_fp4_gemm_runner_backend()."""
        with _fp4_state():
            got = _resolve_rig("native-mixed", RIG, sm8x=None)
        self.assertEqual([g[0] for g in got], [Fp4GemmRunnerBackend.FLASHINFER_CUTLASS,
                                                Fp4GemmRunnerBackend.W4A8_INT8, Fp4GemmRunnerBackend.W4A8_INT8])


class TestExchangeTileView(CustomTestCase):
    """weight_exchange geometry for stamped (native-mixed) swizzled scales."""

    def _scale(self, n, kb, *, k_sharded):
        p = torch.nn.Parameter(torch.zeros(n, kb, dtype=torch.float8_e4m3fn), requires_grad=False)
        nm.mark_swizzled(p, k_sharded=k_sharded)
        return p

    def test_row_parallel_scale_uses_the_tile_view(self):
        from sglang.srt.weg2 import weight_exchange as wx

        g = wx.StorageGeom.of(self._scale(5120, 1088, k_sharded=True))
        self.assertEqual((g.rows, g.cols, g.pitch, g.itemsize), (40, 1088 * 128, 1088 * 128, 1))
        self.assertEqual(g.nbytes, 5120 * 1088)

    def test_unstamped_and_column_parallel_unchanged(self):
        from sglang.srt.weg2 import weight_exchange as wx

        for t in (
            torch.zeros(5120, 1088, dtype=torch.float8_e4m3fn),
            self._scale(34816, 320, k_sharded=False),
        ):
            g = wx.StorageGeom.of(t)
            self.assertEqual((g.rows, g.cols), tuple(t.shape))

    def test_the_join_reads_a_plain_cols_cut_in_the_tile_view(self):
        from sglang.srt.weg2 import weight_exchange as wx
        from sglang.srt.weg2 import xchg_manifest as xm

        def piece(n, kb):
            g = wx.StorageGeom.of(self._scale(n, kb, k_sharded=True))
            return xm.ManifestPiece(
                param_name="model.layers.0.mlp.down_proj.weight_scale",
                tensor_class="down_proj", rows_full=g.rows, cols_full=g.cols,
                itemsize=g.itemsize, tag="weights_0", nbytes=g.nbytes,
            )

        axis, rows_full, cols_full, widths, pad = xm._axis_of(
            "down_proj.weight_scale", piece(5120, 1088),
            [piece(5120, 584), piece(5120, 256), piece(5120, 248)],
        )
        self.assertEqual(axis, wx.COLS)
        self.assertEqual((rows_full, cols_full, pad), (40, 1088 * 128, 0))
        self.assertEqual(widths, (584 * 128, 256 * 128, 248 * 128))

    def test_component_rows_in_tiles(self):
        from sglang.srt.weg2 import weight_exchange_shadow as sh

        owner = torch.nn.Module()
        owner.weight_scale = self._scale(256, 8, k_sharded=True)
        self.assertEqual(sh._in_nvfp4_sf_tiles(owner, "weight_scale", (128, 128)), (1, 1))
        self.assertEqual(sh._in_nvfp4_sf_tiles(owner, "weight_scale", (192, 64)), ())
        owner.weight = torch.nn.Parameter(torch.zeros(256, 4, dtype=torch.uint8), requires_grad=False)
        self.assertEqual(sh._in_nvfp4_sf_tiles(owner, "weight", (128, 128)), (128, 128))


class TestLauncherFlag(CustomTestCase):
    def test_default_argv_unchanged(self):
        from sglang.srt.weg2 import launcher as L

        self.assertEqual(L.uniform_marlin_argv("modelopt", True), ["--fp4-gemm-backend", "marlin"])
        self.assertEqual(L.uniform_marlin_argv("modelopt", False), [])
        self.assertEqual(L.uniform_marlin_argv("compressed-tensors", True), [])
        self.assertIsNone(L.fp4_native_mixed_refusal("modelopt", True, False))
        self.assertIsNone(L.fp4_native_mixed_refusal("fp8", False, False))

    def test_native_mixed_argv(self):
        from sglang.srt.weg2 import launcher as L

        self.assertEqual(
            L.uniform_marlin_argv("modelopt", True, True), ["--fp4-gemm-backend", "native-mixed"]
        )
        self.assertIsNone(L.fp4_native_mixed_refusal("modelopt", True, True))

    def test_native_mixed_refusals(self):
        from sglang.srt.weg2 import launcher as L

        self.assertIn("--fp8-uniform-marlin", L.fp4_native_mixed_refusal("modelopt", False, True))
        self.assertIn("ModelOpt", L.fp4_native_mixed_refusal("compressed-tensors", True, True))

    def test_pinned_marlin_extra_is_refused_against_native_mixed(self):
        from sglang.srt.weg2 import launcher as L

        with self.assertRaises(L.Weg2LaunchRefused):
            L.with_uniform_marlin_argv(
                "--fp4-gemm-backend marlin", L.uniform_marlin_argv("modelopt", True, True)
            )


if __name__ == "__main__":
    unittest.main()
