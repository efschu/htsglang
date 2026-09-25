"""#38 L8 (N4C): Marlin W4A16 on the SHARED native NVFP4 layout.

The 3080 rank of ``--fp4-gemm-backend native-mixed`` keeps the native parameter
set (names, shapes, dtypes, storages) and permutes only the CONTENT between the
native layout (what the exchange moves) and Marlin's (what the kernel reads),
per N-band so the transient stays bounded. Everything here is CPU: the Marlin
repack is the pure-torch reference, which the upstream GPU test
(test/registered/jit/test_gptq_marlin_repack.py) pins against gptq_marlin_repack.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

import contextlib
import os
import unittest
from unittest import mock

import numpy as np
import torch

from sglang.srt.layers.quantization import fp4_utils
from sglang.srt.layers.quantization import nvfp4_marlin_inplace as mi
from sglang.srt.layers.quantization import nvfp4_native_mixed as nm
from sglang.srt.layers.quantization.fp4_utils import Fp4GemmRunnerBackend
from sglang.srt.layers.quantization.marlin_utils import marlin_permute_scales
from sglang.srt.layers.quantization.marlin_utils_fp4 import nvfp4_marlin_process_scales
from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptFp4Config,
    ModelOptFp4LinearMethod,
)
from sglang.test.test_utils import CustomTestCase

G = torch.Generator().manual_seed(38)


def _rand_u8(*shape, hi=256):
    return torch.randint(0, hi, shape, dtype=torch.uint8, generator=G)


def _nonneg_e4m3(rows, cols):
    # non-negative, finite E4M3 bytes (0x00..0x7E; 0x7F is NaN)
    return _rand_u8(rows, cols, hi=0x7F).view(torch.float8_e4m3fn)


def _nibbles_kn(w_u8):
    return mi.native_rows_to_nibbles_kn(w_u8)


def _marlin_weights_numpy(q_kn, perm):
    """Verbatim algorithm of sglang.test.test_marlin_utils.marlin_weights
    (marlin_permute_weights + numpy packing), which the upstream repack test
    compares with gptq_marlin_repack on GPU."""
    size_k, size_n = q_kn.shape
    tile = 16
    q_w = q_kn.to(torch.int64).reshape((size_k // tile, tile, size_n // tile, tile))
    q_w = q_w.permute((0, 2, 1, 3)).reshape((size_k // tile, size_n * tile))
    q_w = q_w.reshape((-1, perm.numel()))[:, perm].reshape(q_w.shape)
    q_w = q_w.numpy().astype(np.uint32)
    q_packed = np.zeros((q_w.shape[0], q_w.shape[1] // 8), dtype=np.uint32)
    for i in range(8):
        q_packed |= q_w[:, i::8] << 4 * i
    return torch.from_numpy(q_packed.astype(np.int32))


def _upstream_perm():
    from sglang.srt.layers.quantization.moe_wna16 import get_weight_perm

    return get_weight_perm(4)


class TestWeightPermutation(CustomTestCase):
    def test_perm_is_the_upstream_perm(self):
        self.assertTrue(torch.equal(mi._weight_perm_cpu(), torch.as_tensor(_upstream_perm()).long()))

    def test_reference_pack_matches_the_upstream_reference(self):
        w = _rand_u8(128, 256 // 2)  # R=128 rows, K=256
        q_kn = _nibbles_kn(w)
        self.assertTrue(
            torch.equal(mi.marlin_pack_ref(q_kn), _marlin_weights_numpy(q_kn, mi._weight_perm_cpu()))
        )

    def test_gptq_packing_of_the_native_bytes(self):
        """qweight = weight.view(int32).T is GPTQ packing along K (nibble i of
        int32 [k8, n] = element 8*k8+i of row n), which is what the loader feeds
        gptq_marlin_repack (marlin_utils_fp4.prepare_nvfp4_layer_for_marlin)."""
        w = _rand_u8(64, 64)
        q = w.view(torch.int32).t().contiguous()
        self.assertTrue(
            torch.equal(mi._ref_repack(q, 128, 64), mi.marlin_pack_ref(_nibbles_kn(w)))
        )

    def test_unpack_inverts_pack(self):
        q_kn = _rand_u8(512, 384, hi=16)
        self.assertTrue(torch.equal(mi.marlin_unpack_ref(mi.marlin_pack_ref(q_kn)), q_kn))

    def test_native_nibbles_round_trip(self):
        w = _rand_u8(256, 320)
        self.assertTrue(torch.equal(mi.nibbles_kn_to_native_rows(_nibbles_kn(w)), w))


class TestBands(CustomTestCase):
    SHAPES_27B = (
        ("P.gate_up", 34816, 5120),
        ("P.down", 5120, 17408),
        ("P.lm_head", 248320, 5120),
        ("D.gate_up.3080", 8192, 5120),
        ("D.gate_up.5090", 18688, 5120),
        ("D.down.3080", 5120, 4096),
        ("D.lm_head", 82816, 5120),
    )

    def test_bands_cover_whole_tiles_within_the_limit(self):
        mb = mi.band_max_bytes()
        for name, n, k in self.SHAPES_27B:
            bands = mi.band_table(n, k)
            self.assertEqual(bands[0][0], 0, name)
            self.assertEqual(bands[-1][1], n, name)
            for (a, b), (c, _) in zip(bands, bands[1:]):
                self.assertEqual(b, c, name)
            for a, b in bands:
                self.assertEqual((b - a) % 128, 0, name)
                self.assertLessEqual((b - a) * k // 2, mb, name)

    def test_transient_bound_is_32_mib_or_less(self):
        """Operator law (keine-korridor-reserve-nie): the conversion is bounded
        by a small fixed working set, not priced as a reserve."""
        self.assertLessEqual(mi.transient_bound_bytes(), 32 * mi.MIB)

    def test_band_table_refuses_unaligned(self):
        with self.assertRaises(ValueError):
            mi.band_table(200, 5120)


@contextlib.contextmanager
def _small_bands(band_bytes, chunk_bytes):
    with mock.patch.dict(
        os.environ,
        {
            "SGLANG_FP4_NATIVE_MIXED_BAND_MIB": str(band_bytes),
            "SGLANG_FP4_NATIVE_MIXED_CHUNK_MIB": str(chunk_bytes),
        },
    ):
        yield


class TestBandConversion(CustomTestCase):
    def test_weight_band_round_trip_and_equals_reference(self):
        for rows, k in ((128, 512), (256, 1024), (384, 128)):
            w = _rand_u8(rows, k // 2)
            band = w.clone()
            with mock.patch.object(mi, "chunk_max_bytes", return_value=rows * 16):
                mi.weight_band_to_marlin_(band, k, mi._ref_repack)  # several K chunks
            ref = mi.marlin_pack_ref(_nibbles_kn(w))
            self.assertTrue(torch.equal(band.view(-1).view(torch.int32).view(ref.shape), ref))
            with mock.patch.object(mi, "chunk_max_bytes", return_value=rows * 48):
                mi.weight_band_to_native_(band, k)
            self.assertTrue(torch.equal(band, w))

    def test_scale_band_matches_the_loader_expressions(self):
        rows, k = 256, 1024
        kb = k // 16
        raw = _nonneg_e4m3(rows, kb)
        sw = nm.swizzle_128x4(raw.view(torch.uint8)).view(torch.float8_e4m3fn)
        band = sw.clone()
        mi.scale_band_to_marlin_(band, k, torch.bfloat16)
        # prepare_nvfp4_layer_for_marlin on the RAW [N, K/16] scale
        ref = raw.T.contiguous().to(torch.bfloat16)
        ref = marlin_permute_scales(s=ref, size_k=k, size_n=rows, group_size=16)
        ref = nvfp4_marlin_process_scales(ref)
        self.assertTrue(
            torch.equal(band.view(torch.uint8).view(-1), ref.view(torch.uint8).reshape(-1))
        )
        mi.scale_band_to_native_(band, k)
        self.assertTrue(torch.equal(band.view(torch.uint8), sw.view(torch.uint8)))

    def test_scale_round_trip_covers_subnormals_and_zero(self):
        rows, k = 128, 256
        raw = torch.arange(0, 128 * 16, dtype=torch.int64).remainder(0x7F).to(torch.uint8)
        raw = raw.reshape(rows, k // 16).view(torch.float8_e4m3fn)
        sw = nm.swizzle_128x4(raw.view(torch.uint8)).view(torch.float8_e4m3fn)
        band = sw.clone()
        mi.scale_band_to_marlin_(band, k, torch.bfloat16)
        mi.scale_band_to_native_(band, k)
        self.assertTrue(torch.equal(band.view(torch.uint8), sw.view(torch.uint8)))

    def test_unswizzle_inverts_swizzle(self):
        s = _rand_u8(384, 64)
        self.assertTrue(torch.equal(mi.unswizzle_128x4(nm.swizzle_128x4(s), 384, 64), s))
        self.assertTrue(torch.equal(mi.swizzle_128x4_exact(s), nm.swizzle_128x4(s)))


# ---------------------------------------------------------------------------
# Layer level through the real ModelOpt loader (CPU, mocked backend).
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _backend(b):
    saved = (
        fp4_utils.FP4_GEMM_RUNNER_BACKEND,
        fp4_utils.FP4_NATIVE_MIXED,
        fp4_utils.FP4_NATIVE_MIXED_SHARED_LAYOUT,
    )
    try:
        with (
            mock.patch.object(torch.Tensor, "cuda", lambda self, *a, **kw: self),
            mock.patch(
                "sglang.srt.layers.quantization.modelopt_quant.is_blackwell_supported",
                # the 5090 rank is sm_120 (native), the 3080 rank sm_86
                return_value=(b == Fp4GemmRunnerBackend.CUTLASS),
            ),
            mock.patch(
                "sglang.srt.layers.quantization.marlin_utils.marlin_make_workspace",
                side_effect=lambda dev, *a, **kw: torch.zeros(8, dtype=torch.int32),
            ),
        ):
            fp4_utils.FP4_GEMM_RUNNER_BACKEND = b
            fp4_utils.FP4_NATIVE_MIXED = True
            fp4_utils.FP4_NATIVE_MIXED_SHARED_LAYOUT = True
            yield
    finally:
        (
            fp4_utils.FP4_GEMM_RUNNER_BACKEND,
            fp4_utils.FP4_NATIVE_MIXED,
            fp4_utils.FP4_NATIVE_MIXED_SHARED_LAYOUT,
        ) = saved


def _loaded_layer(backend, n_parts=2, n_each=256, k=512, seed=5):
    method = ModelOptFp4LinearMethod(
        ModelOptFp4Config(is_checkpoint_nvfp4_serialized=True, group_size=16)
    )
    layer = torch.nn.Module()
    method.create_weights(
        layer, k, [n_each] * n_parts, k, n_each * n_parts, torch.bfloat16, weight_loader=None
    )
    g = torch.Generator().manual_seed(seed)
    layer.weight.data.copy_(torch.randint(0, 256, layer.weight.shape, dtype=torch.uint8, generator=g))
    layer.weight_scale.data.copy_(
        torch.randint(0, 0x7F, layer.weight_scale.shape, dtype=torch.uint8, generator=g).view(
            torch.float8_e4m3fn
        )
    )
    layer.input_scale.data.fill_(0.02)
    layer.weight_scale_2.data.fill_(0.003)
    with _backend(backend):
        method.process_weights_after_loading(layer)
    return method, layer


def _param_set(layer):
    return [(n, tuple(p.shape), p.dtype) for n, p in layer.named_parameters()]


def _dequant_native(layer_like_w, sw, n, k):
    """W[n, k] from native bytes (weight u8 [N,K/2], swizzled e4m3 [N,K/16])."""
    lut = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6])
    w = layer_like_w
    codes = torch.stack((w & 0xF, w >> 4), dim=-1).reshape(n, k).long()
    s = mi.unswizzle_128x4(sw.view(torch.float8_e4m3fn), n, k // 16).to(torch.float32)
    return lut[codes] * s.repeat_interleave(16, dim=1)


class TestLayer(CustomTestCase):
    def test_parameter_set_equals_the_native_rank(self):
        """The join sees the SAME parameter set on the 5090 (cutlass) and the
        3080 (marlin_native_inplace): names, shapes, dtypes."""
        _, l_nat = _loaded_layer(Fp4GemmRunnerBackend.CUTLASS)
        _, l_mar = _loaded_layer(Fp4GemmRunnerBackend.MARLIN_NATIVE_INPLACE)
        self.assertEqual(_param_set(l_nat), _param_set(l_mar))
        self.assertIn(mi.GSCALE_PARAM, dict(l_nat.named_parameters()))
        self.assertTrue(
            torch.equal(getattr(l_nat, mi.GSCALE_PARAM), getattr(l_mar, mi.GSCALE_PARAM))
        )
        self.assertFalse(getattr(l_nat, mi.LAYER_FLAG, False))
        self.assertTrue(getattr(l_mar, mi.LAYER_FLAG))

    def test_content_round_trip_to_the_native_bytes(self):
        _, l_nat = _loaded_layer(Fp4GemmRunnerBackend.CUTLASS)
        with _small_bands(1, 1):
            _, l_mar = _loaded_layer(Fp4GemmRunnerBackend.MARLIN_NATIVE_INPLACE)
        self.assertEqual(l_mar.weight.nvfp4_content, mi.MARLIN)
        self.assertFalse(torch.equal(l_mar.weight, l_nat.weight))
        ptr_w, ptr_s = l_mar.weight.data_ptr(), l_mar.weight_scale.data_ptr()
        mi.model_to_native([l_mar], log=lambda m: None)
        self.assertTrue(torch.equal(l_mar.weight, l_nat.weight))
        self.assertTrue(
            torch.equal(l_mar.weight_scale.view(torch.uint8), l_nat.weight_scale.view(torch.uint8))
        )
        # in place: same storages (CUDA graphs and exchange descriptors hold them)
        self.assertEqual((ptr_w, ptr_s), (l_mar.weight.data_ptr(), l_mar.weight_scale.data_ptr()))
        # idempotent
        self.assertEqual(mi.model_to_native([l_mar], log=lambda m: None), 0)

    def test_wake_after_exchange_forces_native_then_marlin(self):
        """The exchange wrote native bytes into a parameter whose stamp still
        says marlin (first wake after the load): delivered_native re-stamps."""
        _, l_nat = _loaded_layer(Fp4GemmRunnerBackend.CUTLASS)
        _, l_mar = _loaded_layer(Fp4GemmRunnerBackend.MARLIN_NATIVE_INPLACE)
        marlin_bytes = l_mar.weight.clone()
        l_mar.weight.data.copy_(l_nat.weight)  # "the exchange"
        l_mar.weight_scale.data.copy_(l_nat.weight_scale)
        self.assertEqual(l_mar.weight.nvfp4_content, mi.MARLIN)  # stale stamp
        mi.model_to_marlin([l_mar], delivered_native=True, log=lambda m: None)
        self.assertTrue(torch.equal(l_mar.weight, marlin_bytes))
        # a TMS restore (not delivered by the exchange) trusts the stamp
        self.assertEqual(mi.model_to_marlin([l_mar], delivered_native=False, log=lambda m: None), 0)

    def test_negative_scales_are_refused(self):
        method = ModelOptFp4LinearMethod(ModelOptFp4Config(is_checkpoint_nvfp4_serialized=True, group_size=16))
        layer = torch.nn.Module()
        method.create_weights(layer, 256, [128], 256, 128, torch.bfloat16, weight_loader=None)
        layer.weight_scale.data.copy_(torch.full(layer.weight_scale.shape, 0xB8, dtype=torch.uint8).view(torch.float8_e4m3fn))
        layer.input_scale.data.fill_(0.02)
        layer.weight_scale_2.data.fill_(0.003)
        with _backend(Fp4GemmRunnerBackend.MARLIN_NATIVE_INPLACE), self.assertRaisesRegex(RuntimeError, "negative"):
            method.process_weights_after_loading(layer)

    def test_apply_runs_one_gemm_per_band_and_concatenates(self):
        """A fake Marlin GEMM that reads the Marlin band views back through the
        reference inverse: the banded apply must equal the dense native GEMM."""
        n_each, k = 256, 512
        _, l_nat = _loaded_layer(Fp4GemmRunnerBackend.CUTLASS, n_each=n_each, k=k)
        with _small_bands(1, 1):
            method, l_mar = _loaded_layer(Fp4GemmRunnerBackend.MARLIN_NATIVE_INPLACE, n_each=n_each, k=k)
        n = 2 * n_each
        l_mar._nvfp4_marlin_bands = [(0, 128), (128, 384), (384, 512)]
        # re-cut: convert back and forth under the new table
        l_mar.weight.nvfp4_content = mi.NATIVE
        l_mar.weight_scale.nvfp4_content = mi.NATIVE
        l_mar.weight.data.copy_(l_nat.weight)
        l_mar.weight_scale.data.copy_(l_nat.weight_scale)
        mi.layer_to_marlin(l_mar, repack=mi._ref_repack)
        calls = []

        def fake_gemm(input, weight, weight_scale, weight_global_scale, workspace, size_n, size_k, bias):
            calls.append(size_n)
            wb = weight.contiguous().view(torch.uint8).view(size_n, size_k // 2).clone()
            mi.weight_band_to_native_(wb, size_k)
            sb = weight_scale.contiguous().view(torch.uint8).view(size_n, size_k // 16).clone()
            mi.scale_band_to_native_(sb, size_k)
            return input.float() @ _dequant_native(wb, sb, size_n, size_k).t()

        x = torch.randn(3, k)
        out = mi.apply(l_mar, x, gemm=fake_gemm)
        ref = x @ _dequant_native(l_nat.weight, l_nat.weight_scale, n, k).t()
        self.assertEqual(calls, [128, 256, 128])
        self.assertTrue(torch.allclose(out, ref, rtol=0, atol=0))

    def test_modelopt_apply_routes_to_the_inplace_path(self):
        method, l_mar = _loaded_layer(Fp4GemmRunnerBackend.MARLIN_NATIVE_INPLACE)
        with _backend(Fp4GemmRunnerBackend.MARLIN_NATIVE_INPLACE), mock.patch.object(
            mi, "apply", return_value=torch.zeros(1)
        ) as ap:
            method.apply(l_mar, torch.randn(2, 512, dtype=torch.bfloat16))
        ap.assert_called_once()


class TestSm12xSeam(CustomTestCase):
    """Strand F's sm_12x W4A16 hook: ONE call in apply, after the sm_8x
    branches; the 3080 (marlin_native_inplace) never reaches it."""

    def test_the_5090_rank_asks_the_hook_and_none_falls_through(self):
        method, l_nat = _loaded_layer(Fp4GemmRunnerBackend.CUTLASS)
        with _backend(Fp4GemmRunnerBackend.CUTLASS), mock.patch(
            "sglang.srt.layers.quantization.modelopt_quant.maybe_apply_sm12x_w4a16",
            side_effect=lambda layer, x, bias, be: torch.ones(x.shape[0], 512) if x.shape[0] <= 4 else None,
        ) as hook:
            out = method.apply(l_nat, torch.randn(2, 512, dtype=torch.bfloat16))
            self.assertTrue(torch.equal(out, torch.ones(2, 512)))
            with mock.patch(
                "sglang.srt.layers.quantization.modelopt_quant.fp4_quantize",
                side_effect=RuntimeError("fell through to W4A4"),
            ), self.assertRaisesRegex(RuntimeError, "fell through"):
                method.apply(l_nat, torch.randn(8, 512, dtype=torch.bfloat16))
        self.assertEqual(hook.call_count, 2)
        self.assertEqual(hook.call_args[0][3], "cutlass")

    def test_the_3080_rank_never_reaches_the_hook(self):
        method, l_mar = _loaded_layer(Fp4GemmRunnerBackend.MARLIN_NATIVE_INPLACE)
        with _backend(Fp4GemmRunnerBackend.MARLIN_NATIVE_INPLACE), mock.patch(
            "sglang.srt.layers.quantization.modelopt_quant.maybe_apply_sm12x_w4a16"
        ) as hook, mock.patch.object(mi, "apply", return_value=torch.zeros(1)):
            method.apply(l_mar, torch.randn(2, 512, dtype=torch.bfloat16))
        hook.assert_not_called()

    def test_weight_global_scale_is_bound_on_every_native_mixed_rank(self):
        """F's alpha = weight_global_scale (= max(weight_scale_2))."""
        for b in (Fp4GemmRunnerBackend.CUTLASS, Fp4GemmRunnerBackend.MARLIN_NATIVE_INPLACE):
            _, layer = _loaded_layer(b)
            self.assertAlmostEqual(float(layer.weight_global_scale), 0.003, places=6)


class TestFlipHooks(CustomTestCase):
    """weight_updater's two hooks, driven on a stub carrying their inputs."""

    def _stub(self, models, carrier):
        from sglang.srt.managers.scheduler_components.weight_updater import (
            SchedulerWeightUpdaterManager as WU,
        )

        stub = mock.Mock()
        stub._weg2_wake_models = lambda: models
        stub._weg2_wake_weight_carrier = lambda: carrier
        stub.CARRIER_EXCHANGE = WU.CARRIER_EXCHANGE
        return WU, stub

    def test_sleep_then_exchange_wake(self):
        _, l_nat = _loaded_layer(Fp4GemmRunnerBackend.CUTLASS)
        _, l_mar = _loaded_layer(Fp4GemmRunnerBackend.MARLIN_NATIVE_INPLACE)
        marlin_bytes = l_mar.weight.clone()
        WU, stub = self._stub([l_mar], "exchange")
        WU._weg2_nvfp4_marlin_to_native(stub)
        self.assertTrue(torch.equal(l_mar.weight, l_nat.weight))  # the deposit reads native
        WU._weg2_nvfp4_marlin_after_wake(stub)
        self.assertTrue(torch.equal(l_mar.weight, marlin_bytes))

    def test_a_draft_reloaded_from_disk_keeps_its_own_stamp(self):
        """The disk refill of the draft goes through the loader, which leaves
        Marlin content stamped 'marlin'; forcing it 'native' would permute
        Marlin bytes a second time. The target stays exchange-carried."""
        _, t_nat = _loaded_layer(Fp4GemmRunnerBackend.CUTLASS, seed=5)
        _, target = _loaded_layer(Fp4GemmRunnerBackend.MARLIN_NATIVE_INPLACE, seed=5)
        _, draft = _loaded_layer(Fp4GemmRunnerBackend.MARLIN_NATIVE_INPLACE, seed=6)
        t_marlin, d_marlin = target.weight.clone(), draft.weight.clone()
        WU, stub = self._stub([target, draft], "exchange")
        stub.tp_worker.model_runner.model = target
        stub._weg2_model_for_group = lambda g: draft if g == "D" else target
        target.weight.data.copy_(t_nat.weight)  # the exchange wrote native bytes
        target.weight_scale.data.copy_(t_nat.weight_scale)
        stub._weg2_nvfp4_draft_disk_reloaded = True
        WU._weg2_nvfp4_marlin_after_wake(stub)
        self.assertTrue(torch.equal(target.weight, t_marlin))
        self.assertTrue(torch.equal(draft.weight, d_marlin))  # untouched
        self.assertFalse(stub._weg2_nvfp4_draft_disk_reloaded)  # read-and-clear

    def test_no_flagged_layer_is_a_no_op(self):
        _, l_nat = _loaded_layer(Fp4GemmRunnerBackend.CUTLASS)
        before = l_nat.weight.clone()
        WU, stub = self._stub([l_nat], "exchange")
        WU._weg2_nvfp4_marlin_to_native(stub)
        WU._weg2_nvfp4_marlin_after_wake(stub)
        self.assertTrue(torch.equal(l_nat.weight, before))


class TestExchangeClassesL4(CustomTestCase):
    NAMES = (
        "model.layers.3.mlp.gate_up_proj.weight_scale_interleaved",
        "model.layers.3.mlp.gate_up_proj.alpha",
        "model.layers.3.mlp.down_proj.input_scale_inv",
        "model.layers.3.mlp.down_proj.weight_scale_2",
        "model.layers.3.mlp.down_proj.weight_global_scale",
        "model.layers.3.mlp.down_proj.weight_global_scale_w4a16",
    )

    def test_native_mixed_boot_classes_them_under_their_linear(self):
        from sglang.srt.weg2 import weight_exchange_shadow as sh

        with mock.patch.dict(os.environ, {sh.NVFP4_NATIVE_LEAFS_ENV: "1"}):
            got = {sh.tensor_class(n) for n in self.NAMES}
        self.assertEqual(got, {"gate_up_proj", "down_proj"})

    def test_todays_marlin_class_names_unchanged(self):
        from sglang.srt.weg2 import weight_exchange_shadow as sh

        env = {k: v for k, v in os.environ.items() if k != sh.NVFP4_NATIVE_LEAFS_ENV}
        with mock.patch.dict(os.environ, env, clear=True):
            got = [sh.tensor_class(n) for n in self.NAMES]
            self.assertEqual(
                got,
                ["weight_scale_interleaved", "alpha", "input_scale_inv", "weight_scale_2",
                 "weight_global_scale", "weight_global_scale_w4a16"],
            )
            self.assertEqual(sh.tensor_class("model.layers.3.mlp.down_proj.weight_scale"), "down_proj")

    def test_launcher_sets_the_switch_only_for_native_mixed(self):
        from sglang.srt.weg2 import launcher as L
        from sglang.srt.weg2 import weight_exchange_shadow as sh

        self.assertEqual(
            L.fp4_native_mixed_env(L.uniform_marlin_argv("modelopt", True, True)),
            {sh.NVFP4_NATIVE_LEAFS_ENV: "1"},
        )
        self.assertEqual(L.fp4_native_mixed_env(L.uniform_marlin_argv("modelopt", True, False)), {})
        self.assertEqual(L.fp4_native_mixed_env([]), {})


class TestPlannerLanesL7(CustomTestCase):
    ENTRIES = [
        {"name": "NVIDIA GeForce RTX 5090", "gemm_tflops": 232.0, "gemm_lanes": {"int8_native": 678.0}},
        {"name": "NVIDIA GeForce RTX 3080", "gemm_tflops": 59.0, "gemm_lanes": {"int8_native": 178.0}},
        {"name": "NVIDIA GeForce RTX 3080", "gemm_tflops": 59.0, "gemm_lanes": {"int8_native": 182.0}},
    ]

    def test_native_mixed_scores_from_the_measured_record(self):
        from sglang.srt import uneven_perf as up

        with mock.patch.dict(os.environ, {up.NVFP4_NATIVE_MIXED_ENV: "1"}):
            scores, labels, warns = up.rank_gemm_scores(self.ENTRIES, "nvfp4_a4")
        self.assertEqual(scores, [805.4, 55.4, 55.4])
        self.assertIn("native", labels[0])
        self.assertIn("Marlin", labels[1])
        self.assertTrue(all("zcx7pv" in l for l in labels))
        self.assertEqual(warns, [])

    def test_profile_measurement_wins_over_the_record(self):
        from sglang.srt import uneven_perf as up

        entries = [dict(e) for e in self.ENTRIES]
        entries[1] = dict(entries[1], gemm_lanes={"nvfp4_marlin": 57.0})
        with mock.patch.dict(os.environ, {up.NVFP4_NATIVE_MIXED_ENV: "1"}):
            scores, labels, _ = up.rank_gemm_scores(entries, "nvfp4_a4")
        self.assertEqual(scores[1], 57.0)
        self.assertNotIn("zcx7pv", labels[1])

    def test_default_boot_is_the_pre_38_view(self):
        from sglang.srt import uneven_perf as up

        env = {k: v for k, v in os.environ.items() if k != up.NVFP4_NATIVE_MIXED_ENV}
        with mock.patch.dict(os.environ, env, clear=True):
            scores, labels, warns = up.rank_gemm_scores(self.ENTRIES, "nvfp4_a4")
        self.assertEqual(scores, [232.0, 59.0, 59.0])  # dense bf16 fallback, as before
        self.assertEqual(len(warns), 3)


if __name__ == "__main__":
    unittest.main()
