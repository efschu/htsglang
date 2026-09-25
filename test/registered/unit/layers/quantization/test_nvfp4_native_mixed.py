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


def _resolve_rig(backend, caps, *, kernel=True, sm8x="w4a8"):
    """One scheduler process per rank: TP0 = 5090, TP1/TP2 = 3080."""
    args = mock.Mock(fp4_gemm_runner_backend=backend)
    out = []
    for cap in caps:
        with (
            mock.patch("sglang.srt.utils.common.get_device_capability", return_value=cap),
            mock.patch.object(fp4_utils, "get_device_capability", return_value=cap),
            mock.patch.object(fp4_utils, "is_sm100_supported", return_value=cap[0] == 10),
            mock.patch.object(fp4_utils, "is_sm120_supported", return_value=cap[0] == 12),
            mock.patch.object(fp4_utils, "is_cuda", return_value=True),
            mock.patch.object(fp4_utils, "has_fork_nvfp4_cutlass_kernel", return_value=True),
            mock.patch.object(nm, "_try_autoload_w4a8_kernel", return_value=None),
            mock.patch(
                "sglang.srt.environ.envs.SGLANG_FP4_NATIVE_MIXED_SM8X.get",
                return_value=sm8x,
            ),
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
        # sm_8x default (#38 N4C): Marlin W4A16 on the SHARED native layout
        c = nm.resolve_rank_backend((8, 6), w4a8_available=False)
        self.assertEqual((c.backend, c.shared_layout), ("marlin_native_inplace", True))
        c = nm.resolve_rank_backend((8, 6), w4a8_available=True)
        self.assertEqual(c.backend, "marlin_native_inplace")
        c = nm.resolve_rank_backend((8, 6), w4a8_available=True, sm8x_choice="w4a8")
        self.assertEqual((c.backend, c.shared_layout), ("w4a8_int8", True))
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
                (Fp4GemmRunnerBackend.CUTLASS, True, True),
                (Fp4GemmRunnerBackend.W4A8_INT8, True, True),
                (Fp4GemmRunnerBackend.W4A8_INT8, True, True),
            ],
        )

    def test_mixed_rig_default_is_marlin_inplace(self):
        with _fp4_state():
            got = _resolve_rig("native-mixed", RIG, kernel=False, sm8x="marlin")
        self.assertEqual(
            got,
            [
                (Fp4GemmRunnerBackend.CUTLASS, True, True),
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
