"""H88-B: compressed-tensors MoE scheme int4 weights x int8 activations (W4A8).

Unit under test (new files, the A16 scheme is untouched):
  schemes/compressed_tensors_wNa16a8_moe.py   CompressedTensorsWNA16A8MoE + helpers + switch
  moe/fused_moe_triton/fused_marlin_moe_w4a8.py   the fused function (CPU-testable parts)
  dispatch: compressed_tensors.py get_moe_scheme (behind SGLANG_MOE_ACT_INT8)

DESK PART (no GPU, no CUDA call; runs under pytest_gedeckelt.sh):
  * dispatch old/new (switch = H88-E moe_act_int8_requested): switch off = exactly the classes of before (A16 scheme,
    NotImplementedError for a declared-W4A8 checkpoint, also with the switch on), switch on + CUDA =
    the A8 scheme; non-CUDA = A16 even with the switch on,
  * repack shapes / scales (int16 x4096, one factor per layer tensor) / zero points,
    checked against the independent CPU reference of H88-A (marlin_w4a8_utils),
  * the whole process_weights_after_loading flow with the CUDA-bound pieces replaced
    by the CPU reference, then an fp64 emulation of the two-GEMM MoE from the
    PROCESSED tensors against the dequantised-weights reference,
  * the refusals (store on, placeholder weights, row cut, EP, act-order, dims),
  * default path unchanged: sha256 pins of the untouched A16 files, a fixture hash of
    the A16 scale permutation, and "the diff to the base commit only adds lines" for
    the three files that were extended (dispatch, scheme __init__, loader name lists).

GPU PART (skipped unless torch.cuda.is_available() and H88_GPU_TESTS=1; for the metal
window of the NF seat, one card per run -- 3080 (sm86) and 5090 (sm120)):

    H88_GPU_TESTS=1 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<nvml idx> \\
      PYTHONPATH=python:sgl-kernel/python python3 -m pytest -q -x \\
      test/registered/unit/layers/quantization/test_ct_wna16a8_moe_h88b_1007.py -k Gpu

  A small MoE layer (E=8, hidden 512, intermediate 256, g128, top-2, sym) runs through
    old = CompressedTensorsWNA16MoE (A16 Marlin)   and   new = CompressedTensorsWNA16A8MoE
  and both are compared with an fp32 reference built from the dequantised weights.
  Tolerances are UNBELEGT until the first run (no GPU at the desk): the A16 path is
  expected well below 0.03, the A8 path below 0.08 (two chained GEMMs + SwiGLU on
  int8-quantised activations; the single-GEMM criterion of upstream is 0.04).
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="base-a-test-cpu")

import hashlib
import logging
import os
import subprocess
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from sglang.jit_kernel import marlin_w4a8_utils as U
from sglang.srt.layers.moe.fused_moe_triton import fused_marlin_moe_w4a8 as F8
from sglang.srt.layers.quantization.compressed_tensors import (
    compressed_tensors as ct_module,
)
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsWNA16A8MoE,
    CompressedTensorsWNA16MoE,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    compressed_tensors_wNa16a8_moe as S,
)
from sglang.test.test_utils import CustomTestCase

GPU_TESTS = bool(torch.cuda.is_available() and os.environ.get("H88_GPU_TESTS") == "1")

_REPO = Path(__file__).resolve().parents[5]
_PY = _REPO / "python" / "sglang"
BASE_COMMIT = "6cb6580482"  # desk/nf-vorlauf-hebel-1007

#: files of the A16 path that this AP must not touch (sha256 at BASE_COMMIT)
A16_UNTOUCHED = {
    "srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16_moe.py": "854387ab7fad9a9d93a0773321913dfddc376c568a70aaa32440513c29b7c7a4",
    "srt/layers/moe/fused_moe_triton/fused_marlin_moe.py": "b4601772d56f01184a7f42cebbe60596205214fc4e618d7ca22305e9bcd235ba",
    "srt/layers/moe/moe_runner/marlin.py": "2371f178e9683c10b9da44620cae34ec768d71359807e90025c88eaa67a5883f",
    "srt/layers/quantization/marlin_utils.py": "c7907a13d173b7b97b9292f10d1d0beeda03b2eaa0cf392461fd2105df7c60bc",
    "srt/hardware_backend/gpu/quantization/gptq_kernels.py": "58cbe6aed917a83ba921155b936d18aca774f8b1099891651d5e2d3f24827a2f",
}
#: files that were extended; against BASE_COMMIT they may only gain lines
A16_EXTENDED = (
    "srt/layers/quantization/compressed_tensors/compressed_tensors.py",
    "srt/layers/quantization/compressed_tensors/schemes/__init__.py",
    "srt/layers/moe/fused_moe_triton/layer.py",
)
#: sha256 of marlin_moe_permute_scales(arange fixture) -- A16 numerics fixture
A16_SCALE_PERM_FIXTURE_SHA = "4a601dbb5c0f4f54dad3b19619079e5b9f669c9676d690130c1a1e168cbf8253"

EXPERTS_LAYER = "model.layers.0.mlp.experts"
#: the switch (H88-E: flag --moe-act-int8 OR this env; read by moe_act_int8.moe_act_int8_requested)
ENV_SWITCH = "SGLANG_MOE_ACT_INT8"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _ct_config(num_bits=4, group_size=128, symmetric=True, w4a8_checkpoint=False):
    cfg = {
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "config_groups": {
            "group_0": {
                "targets": ["re:.*mlp.experts.*"],
                "weights": {
                    "num_bits": num_bits,
                    "type": "int",
                    "symmetric": symmetric,
                    "strategy": "group",
                    "group_size": group_size,
                },
                "input_activations": None,
            }
        },
        "ignore": ["lm_head", "re:.*self_attn.*", "re:.*mlp.gate$"],
    }
    if w4a8_checkpoint:
        # a weight-only format ("pack-quantized") makes the parser drop input_activations
        cfg["format"] = "int-quantized"
        cfg["config_groups"]["group_0"]["input_activations"] = {
            "num_bits": 8,
            "type": "int",
            "symmetric": True,
            "strategy": "token",
            "dynamic": True,
        }
    return cfg


def _weight_quant(group_size=128, symmetric=True, strategy="group", num_bits=4):
    from compressed_tensors.quantization import QuantizationArgs

    kw = dict(num_bits=num_bits, type="int", symmetric=symmetric, strategy=strategy)
    if strategy == "group":
        kw["group_size"] = group_size
    return QuantizationArgs(**kw)


def _quant_config(**kw):
    return CompressedTensorsConfig.from_config(_ct_config(**kw))


def gptq_pack(q: torch.Tensor) -> torch.Tensor:
    """[K, N] values 0..15 -> int32 [K/8, N], packed along K (CT/GPTQ checkpoint)."""
    K, N = q.shape
    qn = q.cpu().numpy().astype(np.uint32)
    res = np.zeros((K // 8, N), dtype=np.uint32)
    for i in range(8):
        res |= qn[i::8, :] << (4 * i)
    return torch.from_numpy(res.astype(np.int32))


def gptq_unpack(p: torch.Tensor, K: int, N: int) -> torch.Tensor:
    pn = p.cpu().numpy().astype(np.uint32)
    q = np.zeros((K, N), dtype=np.uint32)
    for i in range(8):
        q[i::8, :] = (pn >> (4 * i)) & 0xF
    return torch.from_numpy(q.astype(np.int32))


def awq_pack(zp: torch.Tensor) -> torch.Tensor:
    """[G, N] zero points -> AWQ-packed int32 [G, N/8] (column interleave, pack along N)."""
    G, N = zp.shape
    inter = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    z = zp.cpu().numpy().reshape((-1, 8))[:, inter].reshape((G, N))
    return U.pack_cols(torch.from_numpy(z), 4, G, N)


def cpu_repack_fn(packed, size_k, size_n, num_bits):
    """Stand-in for gptq_marlin_moe_repack_w4a8: the H88-A CPU layout reference."""
    assert num_bits == 4
    perm = U.get_weight_perm(4, True)
    out = []
    for e in range(packed.shape[0]):
        q = gptq_unpack(packed[e], size_k, size_n)
        out.append(U.marlin_weights(q, size_k, size_n, 4, perm, True))
    return torch.stack(out).to(packed.dtype)


def quantize_expert(K, N, gs, asym, dtype, gen):
    """Random expert matrix -> (w_ref [K,N] float32 from dtype-rounded scales,
    q_stored [K,N] 0..15, scales [G,N] float32 (dtype-rounded), zp [G,N]|None)."""
    w = (torch.randn(K, N, generator=gen) * 0.05).float()
    _, q, s, zp = U.quantize_weights_ref(w, gs, asym)
    s = s.to(dtype).float()
    gs_eff = K if gs == -1 else gs
    centre = zp.repeat_interleave(gs_eff, dim=0).float() if asym else 8.0
    w_ref = (q.float() - centre) * s.repeat_interleave(gs_eff, dim=0)
    return w_ref, q, s, zp


class Fixture:
    """E experts, hidden K, intermediate N: reference weights and the CT-format
    checkpoint tensors the scheme's create_weights tensors are filled with."""

    def __init__(self, E=4, K=128, N=64, gs=32, asym=False, dtype=torch.bfloat16, seed=0):
        self.E, self.K, self.N, self.gs, self.asym, self.dtype = E, K, N, gs, asym, dtype
        gen = torch.Generator().manual_seed(seed)
        self.w13_ref, self.w2_ref = [], []
        w13_p, w2_p, w13_s, w2_s, w13_z, w2_z = [], [], [], [], [], []
        for _ in range(E):
            gate, up = (quantize_expert(K, N, gs, asym, dtype, gen) for _ in range(2))
            wr = torch.cat([gate[0], up[0]], dim=1)  # [K, 2N]
            q13 = torch.cat([gate[1], up[1]], dim=1)
            s13 = torch.cat([gate[2], up[2]], dim=1)
            self.w13_ref.append(wr)
            w13_p.append(gptq_pack(q13))
            w13_s.append(s13)
            down = quantize_expert(N, K, gs, asym, dtype, gen)
            self.w2_ref.append(down[0])
            w2_p.append(gptq_pack(down[1]))
            w2_s.append(down[2])
            if asym:
                w13_z.append(awq_pack(torch.cat([gate[3], up[3]], dim=1)))
                w2_z.append(awq_pack(down[3]))
                self.zp13 = getattr(self, "zp13", []) + [torch.cat([gate[3], up[3]], dim=1)]
                self.zp2 = getattr(self, "zp2", []) + [down[3]]
        self.w13_packed = torch.stack(w13_p)
        self.w2_packed = torch.stack(w2_p)
        self.w13_scale = torch.stack(w13_s).to(dtype)
        self.w2_scale = torch.stack(w2_s).to(dtype)
        self.w13_zero = torch.stack(w13_z) if asym else None
        self.w2_zero = torch.stack(w2_z) if asym else None
        self.w13_ref = torch.stack(self.w13_ref)
        self.w2_ref = torch.stack(self.w2_ref)


def new_layer(scheme, fx: Fixture):
    """A stand-in FusedMoE module with the scheme's tensors filled from the fixture."""
    layer = torch.nn.Module()
    layer.moe_tp_size = 1
    layer.layer_id = 0
    layer.num_local_experts = fx.E
    scheme.create_weights(layer, fx.E, fx.K, fx.N, fx.dtype)
    layer.w13_weight_packed.data.copy_(fx.w13_packed)
    layer.w2_weight_packed.data.copy_(fx.w2_packed)
    layer.w13_weight_scale.data.copy_(fx.w13_scale)
    layer.w2_weight_scale.data.copy_(fx.w2_scale)
    if fx.asym:
        layer.w13_weight_zero_point.data.copy_(fx.w13_zero)
        layer.w2_weight_zero_point.data.copy_(fx.w2_zero)
    return layer


_ORIG_REPACK = S.w4a8_repack_moe_weights


def cpu_patches(stack: ExitStack):
    """Replace the CUDA-bound pieces of process_weights_after_loading by CPU stand-ins."""
    stack.enter_context(
        mock.patch.object(
            S,
            "w4a8_repack_moe_weights",
            lambda packed, pf, nb, repack_fn=None: _ORIG_REPACK(
                packed, pf, nb, repack_fn=cpu_repack_fn
            ),
        )
    )
    stack.enter_context(
        mock.patch(
            "sglang.srt.layers.quantization.marlin_utils.marlin_make_workspace",
            lambda device, n=1: torch.zeros(4, dtype=torch.int32),
        )
    )
    stack.enter_context(
        mock.patch(
            "sglang.srt.layers.moe.expert_offload.presplit_expert_offload_after_repack",
            lambda layer: None,
        )
    )
    stack.enter_context(mock.patch.object(S, "_rang_karte", lambda layer: torch.device("cpu")))


def converted_layer(fx: Fixture, group_size=None):
    scheme = CompressedTensorsWNA16A8MoE(
        _quant_config(group_size=fx.gs, symmetric=not fx.asym),
        weight_quant=_weight_quant(fx.gs, symmetric=not fx.asym),
    )
    layer = new_layer(scheme, fx)
    with ExitStack() as st:
        cpu_patches(st)
        scheme.process_weights_after_loading(layer)
    return scheme, layer


# ---- decode of the processed tensors (independent of the scheme code) -----------


def _inv_perm(p):
    inv = torch.empty_like(p)
    inv[p] = torch.arange(p.numel(), dtype=p.dtype)
    return inv


def decode_scales(s_proc_e, factor, G, N):
    """Processed scales of ONE expert [G, N] -> float32 in checkpoint column order."""
    _, single = U.get_scale_perms()
    inv = _inv_perm(torch.tensor(single))
    s = s_proc_e
    if factor is not None:
        s = U.marlin_act_int8_decode_scales(s, factor, torch.float32)
    else:
        s = s.float()
    return s.reshape(-1, len(single))[:, inv].reshape(G, N)


def decode_zero(zp_proc_e, G, N):
    perm, _ = U.get_scale_perms()
    inv = _inv_perm(torch.tensor(perm))
    z = U.unpack_cols(zp_proc_e, 4, G, N)
    return z.reshape(-1, len(perm))[:, inv].reshape(G, N)


def decode_expert(packed_e, s_proc_e, factor, zp_proc_e, K, N, gs):
    G = 1 if gs == -1 else K // gs
    perm = U.get_weight_perm(4, True)
    q = U.marlin_unpack_weights(packed_e, K, N, 4, perm, True).float()
    s = decode_scales(s_proc_e, factor, G, N)
    gs_eff = K if gs == -1 else gs
    if zp_proc_e is None:
        centre = 8.0
    else:
        centre = decode_zero(zp_proc_e, G, N).float().repeat_interleave(gs_eff, dim=0)
    return (q - centre) * s.repeat_interleave(gs_eff, dim=0)


def emulate_moe(layer, scheme, x, topk_w, topk_ids, K, N, gs, a8=True):
    """fp64 emulation of fused_marlin_moe_w4a8 from the PROCESSED layer tensors:
    per-token int8 activations x dequantised int4 weights, SwiGLU in between."""
    E = layer.w13_weight_packed.shape[0]
    w13 = torch.stack(
        [
            decode_expert(
                layer.w13_weight_packed[e],
                layer.w13_weight_scale[e],
                layer.w13_act_scale_factor,
                None if scheme.sym else layer.w13_weight_zero_point[e],
                K,
                2 * N,
                gs,
            )
            for e in range(E)
        ]
    ).double()
    w2 = torch.stack(
        [
            decode_expert(
                layer.w2_weight_packed[e],
                layer.w2_weight_scale[e],
                layer.w2_act_scale_factor,
                None if scheme.sym else layer.w2_weight_zero_point[e],
                N,
                K,
                gs,
            )
            for e in range(E)
        ]
    ).double()
    M, topk = topk_ids.shape
    out = torch.zeros(M, K, dtype=torch.float64)
    for t in range(M):
        for j in range(topk):
            e = int(topk_ids[t, j])
            if e < 0:
                continue
            xq, xs = U.per_token_quant_int8_ref(x[t : t + 1]) if a8 else (None, None)
            xin = (xq.double() * xs.double()) if a8 else x[t : t + 1].double()
            h = xin @ w13[e]
            act = torch.nn.functional.silu(h[:, :N]) * h[:, N:]
            if a8:
                aq, asc = U.per_token_quant_int8_ref(act.float())
                act = aq.double() * asc.double()
            out[t] += float(topk_w[t, j]) * (act @ w2[e]).squeeze(0)
    return out


def reference_moe(fx: Fixture, x, topk_w, topk_ids):
    """fp64 reference with the dequantised weights and unquantised activations."""
    M, topk = topk_ids.shape
    N, K = fx.N, fx.K
    out = torch.zeros(M, K, dtype=torch.float64)
    for t in range(M):
        for j in range(topk):
            e = int(topk_ids[t, j])
            if e < 0:
                continue
            h = x[t : t + 1].double() @ fx.w13_ref[e].double()
            act = torch.nn.functional.silu(h[:, :N]) * h[:, N:]
            out[t] += float(topk_w[t, j]) * (act @ fx.w2_ref[e].double()).squeeze(0)
    return out


def rel_diff(a, b):
    return float((a.double() - b.double()).abs().mean() / b.double().abs().mean())


# ---------------------------------------------------------------------------
# switch + dispatch
# ---------------------------------------------------------------------------


class TestDispatch(CustomTestCase):
    def setUp(self):
        self._saved = os.environ.pop(ENV_SWITCH, None)

    def tearDown(self):
        os.environ.pop(ENV_SWITCH, None)
        if self._saved is not None:
            os.environ[ENV_SWITCH] = self._saved

    def _scheme(self, **kw):
        return _quant_config(**kw).get_moe_scheme(torch.nn.Module(), layer_name=EXPERTS_LAYER)

    def test_switch_off_is_the_a16_scheme_exactly(self):
        with mock.patch.object(ct_module, "_is_cuda", True):
            sch = self._scheme()
        self.assertIs(type(sch), CompressedTensorsWNA16MoE)

    def test_switch_on_cuda_int4_is_the_a8_scheme(self):
        os.environ[ENV_SWITCH] = "1"
        with mock.patch.object(ct_module, "_is_cuda", True):
            sch = self._scheme()
        self.assertIs(type(sch), CompressedTensorsWNA16A8MoE)
        self.assertEqual(sch.w4a8_group_size, 128)
        self.assertIsInstance(sch, CompressedTensorsWNA16MoE)  # create_weights inherited

    def test_switch_on_without_cuda_stays_a16(self):
        os.environ[ENV_SWITCH] = "1"
        with mock.patch.object(ct_module, "_is_cuda", False):
            sch = self._scheme()
        self.assertIs(type(sch), CompressedTensorsWNA16MoE)

    def test_switch_on_int8_experts_stay_a16(self):
        os.environ[ENV_SWITCH] = "1"
        with mock.patch.object(ct_module, "_is_cuda", True):
            sch = self._scheme(num_bits=8)
        self.assertIs(type(sch), CompressedTensorsWNA16MoE)

    def test_declared_w4a8_checkpoint_switch_off_still_not_implemented(self):
        with mock.patch.object(ct_module, "_is_cuda", True):
            with self.assertRaises(NotImplementedError):
                self._scheme(w4a8_checkpoint=True)

    def test_declared_w4a8_checkpoint_is_not_routed_even_with_the_switch(self):
        # H88-B serves the A16 compressed-tensors checkpoint (input_activations null) with
        # int8 activations. A checkpoint that DECLARES int8 input activations ("int-quantized")
        # has an unknown packed layout here (no such checkpoint exists in the house, the
        # scheme's constructor demands pack-quantized): its branch is unchanged.
        os.environ[ENV_SWITCH] = "1"
        with mock.patch.object(ct_module, "_is_cuda", True):
            with self.assertRaises(NotImplementedError):
                self._scheme(w4a8_checkpoint=True)

    def test_a8_scheme_refuses_a_non_pack_quantized_format(self):
        qc = _quant_config(w4a8_checkpoint=True)  # format int-quantized
        with self.assertRaises(ValueError):
            CompressedTensorsWNA16A8MoE(qc, weight_quant=_weight_quant())

    def test_the_tree_declares_the_scheme_to_the_h88e_guard(self):
        from sglang.srt.layers.quantization import moe_act_int8 as M

        self.assertTrue(M.HAS_W4A8_MOE_SCHEME)
        os.environ[ENV_SWITCH] = "1"
        M.require_w4a8_moe_scheme()  # no RuntimeError any more

    def test_switch_on_with_explicit_triton_backend_is_a_conflict_not_an_override(self):
        os.environ[ENV_SWITCH] = "1"
        triton = mock.Mock(is_triton=lambda: True, is_flashinfer_trtllm=lambda: False)
        with mock.patch.object(ct_module, "_is_cuda", True), mock.patch.object(
            ct_module, "get_moe_runner_backend", lambda: triton
        ):
            with self.assertRaisesRegex(RuntimeError, "MOE-ACT-INT8 requested together with"):
                self._scheme()

    def test_flag_form_selects_the_a8_scheme_too(self):
        from sglang.srt.layers.quantization import moe_act_int8 as M

        real = M.moe_act_int8_requested
        # the dispatch reads the switch through moe_act_int8.moe_act_int8_requested at call time
        with mock.patch.object(M, "moe_act_int8_requested", lambda sa=None: real(mock.Mock(moe_act_int8="on"))):
            with mock.patch.object(ct_module, "_is_cuda", True):
                sch = self._scheme()
        self.assertIs(type(sch), CompressedTensorsWNA16A8MoE)

    def test_loader_name_lists_know_the_a8_scheme(self):
        from sglang.srt.layers.moe.fused_moe_triton.layer import ct_method_transposes

        sch = CompressedTensorsWNA16A8MoE(_quant_config(), _weight_quant())
        self.assertTrue(ct_method_transposes(sch))  # the CT loader transposes like for A16
        self.assertTrue(ct_method_transposes(CompressedTensorsWNA16MoE.__new__(CompressedTensorsWNA16MoE)))
        self.assertFalse(ct_method_transposes(object()))


class TestMarker(CustomTestCase):
    def test_marker_line_counts_layers_and_groups(self):
        S._reset_marker_for_tests()
        with self.assertLogs(S.logger, level="INFO") as cm:
            CompressedTensorsWNA16A8MoE(_quant_config(group_size=128), _weight_quant(128))
            CompressedTensorsWNA16A8MoE(_quant_config(group_size=32), _weight_quant(32))
        lines = [r for r in cm.output if "MOE-ACT-INT8 active layers=" in r]
        self.assertEqual(len(lines), 2)
        self.assertIn("layers=1 groups=128", lines[0])
        self.assertIn("layers=2 groups=32,128", lines[1])
        S._reset_marker_for_tests()


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------


class TestDims(CustomTestCase):
    def test_nf_shapes_pass(self):
        S.validate_w4a8_moe_dims(2560, 640, 128)  # main experts
        S.validate_w4a8_moe_dims(2560, 640, 32)  # draft experts

    def test_rejections(self):
        for args in ((2560, 320, 128), (2560, 640, 16), (2560, 96, 32), (100, 640, 128), (2560, 640, 48)):
            with self.assertRaises(ValueError, msg=str(args)):
                S.validate_w4a8_moe_dims(*args)

    def test_channelwise_has_no_group_divisibility(self):
        S.validate_w4a8_moe_dims(2560, 640, -1)


class TestScales(CustomTestCase):
    def test_group_scales_int16_x4096_and_one_factor(self):
        g = torch.Generator().manual_seed(3)
        s = (torch.rand(5, 4, 128, generator=g) * 0.02 + 1e-3).to(torch.bfloat16)
        out, factor = S.w4a8_process_moe_scales(s)
        self.assertEqual(out.shape, s.shape)
        self.assertEqual(out.dtype, s.dtype)
        self.assertEqual(factor.dtype, torch.float32)
        self.assertEqual(factor.dim(), 0)
        ints = out.view(torch.int16)
        self.assertEqual(int(ints.max()), 4096)  # the largest scale maps to 4096
        self.assertTrue(bool((ints >= 0).all()))
        self.assertAlmostEqual(float(factor), float(s.float().max()) / 4096, places=9)
        # decode (int16 * factor, un-permute) recovers the scales to the int16 step
        for e in range(5):
            dec = decode_scales(out[e], factor, 4, 128)
            self.assertLessEqual(float((dec - s[e].float()).abs().max()), 0.5 * float(factor) * 1.001)

    def test_channelwise_scales_are_permuted_only_and_have_no_factor(self):
        s = torch.rand(3, 1, 64).to(torch.float16) + 0.5
        out, factor = S.w4a8_process_moe_scales(s)
        self.assertIsNone(factor)
        self.assertEqual(out.dtype, torch.float16)
        for e in range(3):
            torch.testing.assert_close(decode_scales(out[e], None, 1, 64), s[e].float(), rtol=0, atol=0)

    def test_all_zero_scales_do_not_divide_by_zero(self):
        out, factor = S.w4a8_process_moe_scales(torch.zeros(2, 2, 64, dtype=torch.bfloat16))
        self.assertEqual(int(out.view(torch.int16).abs().max()), 0)
        self.assertEqual(float(factor), 1.0)

    def test_zero_pad_expert_stays_zero(self):
        s = (torch.rand(3, 2, 64) + 0.1).to(torch.bfloat16)
        s[0] = 0  # the pad expert of the generic expert shard
        out, _ = S.w4a8_process_moe_scales(s)
        self.assertEqual(int(out[0].view(torch.int16).abs().max()), 0)

    def test_no_experts(self):
        out, factor = S.w4a8_process_moe_scales(torch.empty(0, 2, 64, dtype=torch.bfloat16))
        self.assertEqual(out.shape, (0, 2, 64))
        self.assertIsNone(factor)


class TestZeroPoints(CustomTestCase):
    def test_equals_the_h88a_reference(self):
        g = torch.Generator().manual_seed(5)
        E, G, N = 3, 4, 128
        zp = torch.randint(0, 16, (E, G, N), generator=g, dtype=torch.int32)
        packed = torch.stack([awq_pack(zp[e]) for e in range(E)])
        out = S.w4a8_moe_zero_points(packed, N)
        self.assertEqual(out.shape, packed.shape)
        self.assertEqual(out.dtype, torch.int32)
        for e in range(E):
            ref = U.marlin_zero_points(zp[e], G, N, 4, True)
            torch.testing.assert_close(out[e], ref, rtol=0, atol=0)

    def test_no_experts(self):
        z = torch.empty(0, 2, 8, dtype=torch.int32)
        self.assertEqual(S.w4a8_moe_zero_points(z, 64).shape, z.shape)


class TestRepackShapes(CustomTestCase):
    def test_w13_w2_shapes_and_values(self):
        fx = Fixture(E=3, K=128, N=64, gs=32)
        w13 = S.w4a8_repack_moe_weights(fx.w13_packed, 8, 4, repack_fn=cpu_repack_fn)
        w2 = S.w4a8_repack_moe_weights(fx.w2_packed, 8, 4, repack_fn=cpu_repack_fn)
        self.assertEqual(tuple(w13.shape), (3, 128 // 16, 2 * 64 * 2))  # [E, K/16, 2N*16/8]
        self.assertEqual(tuple(w2.shape), (3, 64 // 16, 128 * 2))  # [E, N/16, K*16/8]
        self.assertEqual(w13.dtype, torch.int32)
        shapes = U.w4a8_expected_shapes(128, 2 * 64, 32, False, num_experts=3)
        self.assertEqual(tuple(w13.shape), shapes["b_q_weight"])
        perm = U.get_weight_perm(4, True)
        for e in range(3):
            q = U.marlin_unpack_weights(w13[e], 128, 128, 4, perm, True)
            torch.testing.assert_close(q.to(torch.int32), gptq_unpack(fx.w13_packed[e], 128, 128), rtol=0, atol=0)


# ---------------------------------------------------------------------------
# the scheme end to end on CPU
# ---------------------------------------------------------------------------


class TestSchemeFlowCpu(CustomTestCase):
    def _check(self, fx: Fixture, tol_w=None):
        scheme, layer = converted_layer(fx)
        K, N, E = fx.K, fx.N, fx.E
        self.assertTrue(layer.is_marlin_converted)
        # shapes of the converted tensors
        self.assertEqual(tuple(layer.w13_weight_packed.shape), (E, K // 16, 2 * N * 2))
        self.assertEqual(tuple(layer.w2_weight_packed.shape), (E, N // 16, K * 2))
        self.assertEqual(layer.w13_weight_scale.dtype, fx.dtype)
        groups13 = 1 if fx.gs == -1 else K // fx.gs
        groups2 = 1 if fx.gs == -1 else N // fx.gs
        self.assertEqual(tuple(layer.w13_weight_scale.shape), (E, groups13, 2 * N))
        self.assertEqual(tuple(layer.w2_weight_scale.shape), (E, groups2, K))
        for f, g in ((layer.w13_act_scale_factor, groups13), (layer.w2_act_scale_factor, groups2)):
            if g > 1:
                self.assertEqual(f.dtype, torch.float32)
                self.assertEqual(f.dim(), 0)
            else:
                self.assertIsNone(f)
        self.assertEqual(layer.w13_weight_g_idx.shape, (E, 0))
        # decoded weights = reference weights to the int16 scale step
        for e in range(E):
            w13 = decode_expert(
                layer.w13_weight_packed[e], layer.w13_weight_scale[e], layer.w13_act_scale_factor,
                None if scheme.sym else layer.w13_weight_zero_point[e], K, 2 * N, fx.gs,
            )  # fmt: skip
            f = layer.w13_act_scale_factor
            bound = 8.0 * (float(f) if f is not None else 0.0) + 1e-6
            self.assertLessEqual(float((w13 - fx.w13_ref[e]).abs().max()), max(bound, 1e-6), f"w13 expert {e}")
            w2 = decode_expert(
                layer.w2_weight_packed[e], layer.w2_weight_scale[e], layer.w2_act_scale_factor,
                None if scheme.sym else layer.w2_weight_zero_point[e], N, K, fx.gs,
            )  # fmt: skip
            f = layer.w2_act_scale_factor
            bound = 8.0 * (float(f) if f is not None else 0.0) + 1e-6
            self.assertLessEqual(float((w2 - fx.w2_ref[e]).abs().max()), max(bound, 1e-6), f"w2 expert {e}")
        return scheme, layer

    def test_sym_g32(self):
        self._check(Fixture(E=4, K=128, N=64, gs=32))

    def test_sym_g128_fp16(self):
        self._check(Fixture(E=3, K=256, N=128, gs=128, dtype=torch.float16))

    def test_sym_channelwise(self):
        scheme = None
        fx = Fixture(E=2, K=128, N=64, gs=-1)
        cfg = _ct_config(group_size=128)
        cfg["config_groups"]["group_0"]["weights"]["strategy"] = "channel"
        cfg["config_groups"]["group_0"]["weights"].pop("group_size")
        qc = CompressedTensorsConfig.from_config(cfg)
        scheme = CompressedTensorsWNA16A8MoE(qc, _weight_quant(strategy="channel"))
        self.assertEqual(scheme.w4a8_group_size, -1)
        layer = new_layer(scheme, fx)
        with ExitStack() as st:
            cpu_patches(st)
            scheme.process_weights_after_loading(layer)
        self.assertIsNone(layer.w13_act_scale_factor)
        self.assertIsNone(layer.w2_act_scale_factor)

    def test_asym_g32_zero_points_prepared(self):
        scheme, layer = self._check(Fixture(E=3, K=128, N=64, gs=32, asym=True))
        self.assertFalse(scheme.sym)
        self.assertEqual(layer.w13_weight_zero_point.dtype, torch.int32)

    def test_second_call_is_a_noop(self):
        fx = Fixture(E=2, K=128, N=64, gs=32)
        scheme, layer = converted_layer(fx)
        before = layer.w13_weight_packed.clone()
        with ExitStack() as st:
            cpu_patches(st)
            scheme.process_weights_after_loading(layer)  # is_marlin_converted: skipped
        torch.testing.assert_close(layer.w13_weight_packed, before, rtol=0, atol=0)

    def test_emulated_moe_matches_reference_within_a8_tolerance(self):
        fx = Fixture(E=4, K=128, N=64, gs=32, seed=7)
        scheme, layer = converted_layer(fx)
        g = torch.Generator().manual_seed(11)
        M, topk = 6, 2
        x = torch.randn(M, fx.K, generator=g).to(fx.dtype)
        ids = torch.stack([torch.randperm(fx.E, generator=g)[:topk] for _ in range(M)]).to(torch.int32)
        w = torch.softmax(torch.randn(M, topk, generator=g), dim=-1)
        got = emulate_moe(layer, scheme, x, w, ids, fx.K, fx.N, fx.gs)
        ref = reference_moe(fx, x, w, ids)
        self.assertLess(rel_diff(got, ref), 0.08)
        # the a16 emulation of the same tensors (no activation quantisation) is closer
        got16 = emulate_moe(layer, scheme, x, w, ids, fx.K, fx.N, fx.gs, a8=False)
        self.assertLess(rel_diff(got16, ref), 0.01)

    def test_sanitize_minus_one_entries_contribute_zero(self):
        # the emulation skips ids < 0; the kernel path computes expert 0 for them and
        # multiplies by the zeroed routing weight -> same output
        fx = Fixture(E=4, K=128, N=64, gs=32, seed=2)
        scheme, layer = converted_layer(fx)
        g = torch.Generator().manual_seed(4)
        M, topk = 4, 3
        x = torch.randn(M, fx.K, generator=g).to(fx.dtype)
        ids = torch.tensor([[0, 1, 2], [3, -1, 1], [-1, -1, 2], [1, 2, -1]], dtype=torch.int32)
        w = torch.rand(M, topk, generator=g)
        base = emulate_moe(layer, scheme, x, w, ids, fx.K, fx.N, fx.gs)
        eids = torch.tensor([0, 2, -1, 1], dtype=torch.int32)
        e2, w2 = F8.sanitize_routing_for_a8(eids, w, ids)
        self.assertEqual(e2.tolist(), [0, 2, 0, 1])
        self.assertTrue(bool((w2[ids < 0] == 0).all()))
        torch.testing.assert_close(w2[ids >= 0], w[ids >= 0], rtol=0, atol=0)
        ids0 = ids.clamp(min=0)
        via_kernel_semantics = emulate_moe(layer, scheme, x, w2, ids0, fx.K, fx.N, fx.gs)
        torch.testing.assert_close(via_kernel_semantics, base, rtol=1e-9, atol=1e-9)


class TestHelpersFused(CustomTestCase):
    def test_block_size_rule_matches_a16_without_the_8(self):
        def a16(M, topk, E):
            for bs in [8, 16, 32, 48, 64]:
                if M * topk / E / bs < 0.9:
                    break
            return bs

        for M in (1, 2, 8, 64, 256, 1024, 4096):
            for topk in (2, 8, 10):
                for E in (8, 64, 512):
                    want = a16(M, topk, E)
                    got = F8.select_block_size_m(M, topk, E)
                    self.assertEqual(got, max(want, 16), (M, topk, E))
                    self.assertIn(got, F8.W4A8_MOE_BLOCK_SIZES)

    def test_block_sizes_equal_the_kernel_contract(self):
        self.assertEqual(F8.W4A8_MOE_BLOCK_SIZES, U.W4A8_MOE_BLOCK_SIZES)

    def test_a_scales_get_the_factor(self):
        a = torch.tensor([[0.5], [2.0]], dtype=torch.float32)
        torch.testing.assert_close(F8.scale_a_for_gemm(a, None), torch.tensor([0.5, 2.0]))
        torch.testing.assert_close(
            F8.scale_a_for_gemm(a, torch.tensor(0.25)), torch.tensor([0.125, 0.5])
        )


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------


class TestRefusals(CustomTestCase):
    def _scheme_and_layer(self):
        fx = Fixture(E=2, K=128, N=64, gs=32)
        scheme = CompressedTensorsWNA16A8MoE(_quant_config(group_size=32), _weight_quant(32))
        return scheme, new_layer(scheme, fx)

    def _run(self, scheme, layer):
        with ExitStack() as st:
            cpu_patches(st)
            scheme.process_weights_after_loading(layer)

    def test_store_on_refuses(self):
        scheme, layer = self._scheme_and_layer()
        with mock.patch.dict(os.environ, {"SGLANG_MOE_EXPERT_STORE_DIR": "/tmp/x"}):
            with self.assertRaisesRegex(RuntimeError, "MOE-ACT-INT8 REFUSED.*store"):
                self._run(scheme, layer)
        self.assertFalse(getattr(layer, "is_marlin_converted", False))

    def test_placeholder_weights_refuse(self):
        scheme, layer = self._scheme_and_layer()
        with mock.patch("sglang.srt.weg2.adopt.weights_are_placeholder", lambda: True):
            with self.assertRaisesRegex(RuntimeError, "REFUSED.*placeholder"):
                self._run(scheme, layer)

    def test_row_cut_refuses(self):
        scheme, layer = self._scheme_and_layer()
        layer._h2d_cut_rows = [0]
        with self.assertRaisesRegex(RuntimeError, "REFUSED.*subset"):
            self._run(scheme, layer)

    def test_group_act_order_refused(self):
        wq = _weight_quant(32)
        wq_dict = wq.model_dump()
        wq_dict["actorder"] = "group"
        from compressed_tensors.quantization import QuantizationArgs

        with self.assertRaises(NotImplementedError):
            CompressedTensorsWNA16A8MoE(_quant_config(group_size=32), QuantizationArgs(**wq_dict))

    def test_int8_weights_refused(self):
        with self.assertRaises(ValueError):
            CompressedTensorsWNA16A8MoE(_quant_config(num_bits=8), _weight_quant(num_bits=8))

    def test_bad_group_size_refused_at_construction(self):
        with self.assertRaises(ValueError):
            CompressedTensorsWNA16A8MoE(_quant_config(group_size=16), _weight_quant(16))

    def test_dims_refused_at_create_weights(self):
        scheme = CompressedTensorsWNA16A8MoE(_quant_config(group_size=128), _weight_quant(128))
        layer = torch.nn.Module()
        layer.moe_tp_size = 1
        with self.assertRaisesRegex(ValueError, "not a multiple of 64|not divisible"):
            scheme.create_weights(layer, 2, 2560, 320, torch.bfloat16)

    def test_dtype_refused_at_create_weights(self):
        scheme = CompressedTensorsWNA16A8MoE(_quant_config(group_size=128), _weight_quant(128))
        layer = torch.nn.Module()
        layer.moe_tp_size = 1
        with self.assertRaises(ValueError):
            scheme.create_weights(layer, 2, 256, 128, torch.float32)

    def test_expert_parallel_refused_at_apply(self):
        scheme, layer = self._scheme_and_layer()

        class D:
            local_expert_mapping = torch.zeros(2, dtype=torch.int32)

        layer.dispatcher = D()
        scheme.moe_runner_config = mock.Mock(activation="silu")
        with self.assertRaisesRegex(NotImplementedError, "expert parallelism"):
            scheme.apply_weights(layer, mock.Mock())

    def test_get_marlin_quant_info_refused(self):
        scheme, layer = self._scheme_and_layer()
        with self.assertRaises(NotImplementedError):
            scheme.get_marlin_quant_info(layer)

    def test_restore_before_loading_clears_factors(self):
        fx = Fixture(E=2, K=128, N=64, gs=32)
        scheme, layer = converted_layer(fx)
        self.assertIsNotNone(layer.w13_act_scale_factor)
        scheme.restore_weights_before_loading(layer)
        self.assertIsNone(layer.w13_act_scale_factor)
        self.assertFalse(layer.is_marlin_converted)


# ---------------------------------------------------------------------------
# default path unchanged
# ---------------------------------------------------------------------------


def _git(*args):
    return subprocess.run(
        ["git", "-C", str(_REPO), *args], capture_output=True, text=True, timeout=60
    )


class TestDefaultPathUnchanged(CustomTestCase):
    def test_a16_files_byte_identical(self):
        for rel, sha in A16_UNTOUCHED.items():
            data = (_PY / rel).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), sha, rel)

    def test_a16_scale_permutation_fixture_hash(self):
        from sglang.srt.layers.quantization.marlin_utils import marlin_moe_permute_scales

        s = (torch.arange(4 * 3 * 64) % 251).reshape(4, 3, 64).to(torch.bfloat16)
        o = marlin_moe_permute_scales(s, 384, 64, 128)
        got = hashlib.sha256(o.contiguous().view(torch.int16).numpy().tobytes()).hexdigest()
        self.assertEqual(got, A16_SCALE_PERM_FIXTURE_SHA)

    def test_extended_files_only_gain_lines(self):
        probe = _git("cat-file", "-e", f"{BASE_COMMIT}^{{commit}}")
        if probe.returncode != 0:
            self.skipTest(f"base commit {BASE_COMMIT} not available in this checkout")
        for rel in A16_EXTENDED:
            r = _git("diff", "--numstat", BASE_COMMIT, "--", f"python/sglang/{rel}")
            self.assertEqual(r.returncode, 0, r.stderr)
            if not r.stdout.strip():
                continue  # unchanged since base (e.g. already committed upstream of base)
            added, deleted, _ = r.stdout.split()[:3]
            self.assertEqual(int(deleted), 0, f"{rel}: {deleted} lines removed vs {BASE_COMMIT}")
            self.assertGreater(int(added), 0, rel)

    def test_switch_off_dispatch_source_is_gated(self):
        src = (_PY / "srt/layers/quantization/compressed_tensors/compressed_tensors.py").read_text()
        # every new return of the A8 scheme sits behind the switch call
        self.assertEqual(src.count("return CompressedTensorsWNA16A8MoE("), 1)
        # one call + the helper definition at the end of the file
        self.assertEqual(src.count("_w4a8_moe_requested()"), 2)


# ---------------------------------------------------------------------------
# GPU part (metal window of the NF seat)
# ---------------------------------------------------------------------------


@unittest.skipUnless(GPU_TESTS, "needs a GPU window: H88_GPU_TESTS=1 CUDA_VISIBLE_DEVICES=<idx>")
class TestGpuSmallMoeOldVsNew(CustomTestCase):
    """A16 Marlin scheme (old) vs W4A8 scheme (new) on one small MoE layer."""

    E, K, N, GS, TOPK = 8, 512, 256, 128, 2

    def _build(self, scheme_cls, fx, dev):
        wq = _weight_quant(self.GS)
        scheme = scheme_cls(_quant_config(group_size=self.GS), weight_quant=wq)
        with torch.device(dev):
            layer = torch.nn.Module()
            layer.moe_tp_size = 1
            layer.layer_id = 0
            layer.num_local_experts = fx.E
            scheme.create_weights(layer, fx.E, fx.K, fx.N, fx.dtype)
            layer.w13_weight_packed.data.copy_(fx.w13_packed)
            layer.w2_weight_packed.data.copy_(fx.w2_packed)
            layer.w13_weight_scale.data.copy_(fx.w13_scale)
            layer.w2_weight_scale.data.copy_(fx.w2_scale)
            scheme.process_weights_after_loading(layer)
        return scheme, layer

    def _apply(self, scheme, layer, x, w, ids):
        from types import SimpleNamespace

        # both schemes read only these four fields of the runner config in apply_weights
        scheme.moe_runner_config = SimpleNamespace(
            activation="silu", is_gated=True, routed_scaling_factor=None, swiglu_limit=None,
        )  # fmt: skip
        d = mock.Mock()
        d.hidden_states = x
        d.topk_output = (w, ids, torch.zeros(x.shape[0], self.E, device=x.device))
        return scheme.apply_weights(layer, d).hidden_states

    def _run(self, M):
        dev = "cuda"
        fx = Fixture(E=self.E, K=self.K, N=self.N, gs=self.GS, seed=21)
        g = torch.Generator().manual_seed(M)
        x = torch.randn(M, self.K, generator=g).to(fx.dtype).to(dev)
        ids = torch.stack([torch.randperm(self.E, generator=g)[: self.TOPK] for _ in range(M)]).to(torch.int32).to(dev)
        w = torch.softmax(torch.randn(M, self.TOPK, generator=g), dim=-1).to(dev)
        ref = reference_moe(fx, x.cpu(), w.cpu(), ids.cpu())
        old_s, old_l = self._build(CompressedTensorsWNA16MoE, fx, dev)
        new_s, new_l = self._build(CompressedTensorsWNA16A8MoE, fx, dev)
        out_old = self._apply(old_s, old_l, x, w, ids).cpu()
        out_new = self._apply(new_s, new_l, x, w, ids).cpu()
        self.assertTrue(bool(torch.isfinite(out_new.float()).all()))
        d_old, d_new = rel_diff(out_old, ref), rel_diff(out_new, ref)
        print(f"[H88-B gpu] M={M}: |A16-ref|={d_old:.4f} |A8-ref|={d_new:.4f} |A8-A16|={rel_diff(out_new, out_old):.4f}")
        self.assertLess(d_old, 0.03)
        self.assertLess(d_new, 0.08)

    def test_decode_batch(self):
        self._run(1)

    def test_small_batch(self):
        self._run(7)

    def test_prefill_chunk(self):
        self._run(256)

    def test_minus_one_topk_entries(self):
        dev = "cuda"
        fx = Fixture(E=self.E, K=self.K, N=self.N, gs=self.GS, seed=22)
        M = 8
        g = torch.Generator().manual_seed(1)
        x = torch.randn(M, self.K, generator=g).to(fx.dtype).to(dev)
        ids = torch.stack([torch.randperm(self.E, generator=g)[: self.TOPK] for _ in range(M)]).to(torch.int32)
        w = torch.softmax(torch.randn(M, self.TOPK, generator=g), dim=-1)
        ids[3, 1] = -1  # routed to a non-local expert
        ids[5, :] = -1  # a padded token
        new_s, new_l = self._build(CompressedTensorsWNA16A8MoE, fx, dev)
        out = self._apply(new_s, new_l, x, w.to(dev), ids.to(dev)).cpu()
        ref = reference_moe(fx, x.cpu(), w, ids)
        keep = [t for t in range(M) if t != 5]
        self.assertLess(rel_diff(out[keep], ref[keep]), 0.08)


if __name__ == "__main__":
    unittest.main()
