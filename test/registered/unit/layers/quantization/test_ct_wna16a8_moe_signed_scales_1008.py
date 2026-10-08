"""H88 scalefix 1008: SIGNED group scales through the W4A8 (int4 x int8-activation) MoE path.

Defect (found by the KLD seat on the real NF checkpoint, layer 23): the AutoRound group scales of the NF
experts are signed (w13 min -0.0699 / median -0.0032 / max +0.0731). ``marlin_act_int8_process_scales`` divided
by the SIGNED max, and both a8 Marlin templates read the int16 x4096 scale as ``uint16_t``: a negative scale
became 65536 - |v| (KLD seat: A8i output norm 5e6 instead of 1.3; this file on the old tree, fp16 layer: NaN
output for every M and stand-in). The H88 tests used positive random scales only.

Fix (one place, both sides of the contract): the sign stays in the int16 (factor = max|s| / 4096, the value
``w4a8_scale_proposal`` already published) and both templates read ``int16_t``. The kernel-level pins (template
read type, roundtrip through it, dense/MoE GPU GEMMs with signed scales) live in
python/sglang/jit_kernel/tests/test_marlin_w4a8_1007.py; this file covers the scheme:

DESK PART (CPU): the scheme's "local" path (factor=None) and the store's "published" path give the SAME bytes
and factor for signed scales (before the fix they did not: signed max vs abs max), int16 range +-4096, decode,
and the scheme flow emulated in fp64 with signed scales.

GPU PART (H88_GPU_TESTS=1, one card):
  * a small synthetic MoE layer with signed scales through the A16 scheme (old) and the A8 scheme (new) against
    an fp64 reference, thresholds IDENTICAL to TestGpuSmallMoeOldVsNew (A16 < 0.03, A8 < 0.08);
  * the REAL layer 23 experts of Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist-abl-wxp (512 experts, g128,
    sym int4, fp16 scales; env H88_NF_CKPT, default the rig path; skipped when absent) with the real router,
    M in {1, 8, 64, 256} and three activation STAND-INS (gaussian rows; gaussian x the channel scale of the
    layer's hc_norm weight; gaussian with four 30x outlier channels -- none of them is a real hidden state), A16
    and A8i against an fp32 reference (TF32 off) and against an fp32 EMULATION of the A8 arithmetic (int8
    per-token activations before both GEMMs x int4 weights with the int16-grid scales). Error d = per-token
    relative L2 error, checked on its mean and its max over the tokens.

    Bound, derived from the measurement on the fixed tree (3080, 2026-10-08), not assumed:
      - ``d(A8_emu, ref)`` is the error of int8 activations themselves -- the scheme's choice, not the kernel's.
        Measured 2.1-2.4 % (gauss / hc_norm stand-in), 8.3-8.8 % with outlier channels (per-token absmax
        quantisation); A16 measured 0.06-0.08 % (fp16 layer: the fp16 roundings of A16 are 2**-11).
      - The kernel's own deviation from that exact arithmetic (fp16 GEMM-1 output before SwiGLU, the int8
        re-rounding of the SwiGLU output from it, fp32/int32 summation order) measured, relative to
        ``d(A8_emu, ref)``: mean over tokens <= 0.19x, max over tokens <= 0.49x (hc_norm stand-in M=256; gauss
        M=256 0.32x). Bound ``d(A8_gpu, A8_emu) <= k * d(A8_emu, ref)`` with k = 0.5 (mean) / 1.0 (max), about 2x
        headroom over the worst measured run.
      - Those deviations are rounding noise, largely independent of the int8 noise: measured
        ``d(A8_gpu, ref) / d(A8_emu, ref)`` 0.99-1.01 (mean and max). Bound ``d(A8_gpu, ref) <= f * d(A8_emu, ref)
        + 2 * d(A16_gpu, ref)`` with f = 1.25 (mean) / 1.5 (max) = sqrt(1 + k**2) plus headroom; the A16 term
        carries the output-dtype rounding.
      - output norm within 5 % of the reference norm, finite. The old tree gives NaN (red); a wrong sign of
        individual scales gives d >> 1.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

import importlib.util
import json
import os
import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.jit_kernel import marlin_w4a8_utils as U
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsWNA16A8MoE,
    CompressedTensorsWNA16MoE,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    compressed_tensors_wNa16a8_moe as S,
)
from sglang.test.test_utils import CustomTestCase

# the H88-B fixtures (Fixture, converted_layer, emulate/reference MoE, decode) -- reused, not copied
_H88B_PATH = Path(__file__).with_name("test_ct_wna16a8_moe_h88b_1007.py")
_spec = importlib.util.spec_from_file_location("_h88b_fixtures_for_scalefix", _H88B_PATH)
H = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = H
_spec.loader.exec_module(H)

GPU_TESTS = bool(torch.cuda.is_available() and os.environ.get("H88_GPU_TESTS") == "1")
NF_CKPT = Path(
    os.environ.get(
        "H88_NF_CKPT",
        "/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist-abl-wxp",
    )
)
NF_LAYER = int(os.environ.get("H88_NF_LAYER", "23"))

#: real-layer bounds (derivation: module docstring); per statistic of the per-token error: 0 = mean, 2 = max
BOUND_NORM = 0.05
BOUND_A16_FACTOR = 2.0
BOUND_KERNEL_VS_EMU = {0: 0.5, 2: 1.0}
BOUND_EMU_FACTOR = {0: 1.25, 2: 1.5}


def signed_fixture(E=4, K=128, N=64, gs=32, dtype=torch.bfloat16, seed=0, p_neg=0.6):
    """H88-B Fixture with signed group scales (~60 % negative, the largest magnitude negative: |min| > max).
    The reference weights are linear in the scale, so they follow the sign exactly."""
    fx = H.Fixture(E=E, K=K, N=N, gs=gs, dtype=dtype, seed=seed)
    g = torch.Generator().manual_seed(500 + seed)
    for scale_attr, ref_attr, rows in (("w13_scale", "w13_ref", K), ("w2_scale", "w2_ref", N)):
        s = getattr(fx, scale_attr)
        sign = torch.where(torch.rand(s.shape, generator=g) < p_neg, -1.0, 1.0)
        sign.view(-1)[int(s.float().abs().argmax())] = -1.0
        setattr(fx, scale_attr, (s.float() * sign).to(dtype))
        gs_eff = rows if gs == -1 else gs
        setattr(fx, ref_attr, getattr(fx, ref_attr) * sign.repeat_interleave(gs_eff, dim=1))
    return fx


# ---------------------------------------------------------------------------
# DESK
# ---------------------------------------------------------------------------


class TestSignedScalesSchemeCpu(CustomTestCase):
    def _signed(self, seed=3):
        g = torch.Generator().manual_seed(seed)
        s = torch.rand(5, 4, 128, generator=g) * 0.02 + 1e-3
        sign = torch.where(torch.rand(s.shape, generator=g) < 0.6, -1.0, 1.0)
        sign.view(-1)[int(s.argmax())] = -1.0
        s = s * sign
        # |min| = 2.5 * max: the signed max (old factor) is NOT the abs max (the NF w2 tensor has |min| > max too)
        return torch.where(s > 0, s * 0.4, s).to(torch.bfloat16)

    def test_local_path_equals_the_published_path(self):
        # factor=None (one process, H88-B) and factor=w4a8_scale_proposal (store, H88-C) must agree byte for
        # byte: P and D (or a store row and a local conversion) meet in the same kernel
        for s in (self._signed(), -self._signed(4), self._signed(5).abs()):
            out_l, f_l = S.w4a8_process_moe_scales(s)
            prop = S.w4a8_scale_proposal(s)
            out_p, f_p = S.w4a8_process_moe_scales(s, factor=prop, what="test")
            self.assertEqual(float(f_l), float(f_p))
            self.assertEqual(float(f_l), float(s.float().abs().max()) / U.W4A8_SCALE_INT_RANGE)
            self.assertTrue(torch.equal(out_l.view(torch.int16), out_p.view(torch.int16)))

    def test_signed_int16_range_sign_and_decode(self):
        s = self._signed()
        out, factor = S.w4a8_process_moe_scales(s)
        ints = out.view(torch.int16)
        self.assertEqual(int(ints.abs().max()), U.W4A8_SCALE_INT_RANGE)
        self.assertLess(int(ints.min()), 0)
        for e in range(s.shape[0]):
            dec = H.decode_scales(out[e], factor, 4, 128)
            self.assertLessEqual(float((dec - s[e].float()).abs().max()), 0.5 * float(factor) * 1.001)
            self.assertTrue(bool((torch.sign(dec) == torch.sign(s[e].float())).all()))

    def test_scheme_flow_with_signed_scales_emulated(self):
        fx = signed_fixture(E=4, K=128, N=64, gs=32, seed=7)
        self.assertLess(float(fx.w13_scale.float().min()), 0.0)
        scheme, layer = H.converted_layer(fx)
        g = torch.Generator().manual_seed(11)
        M, topk = 6, 2
        x = torch.randn(M, fx.K, generator=g).to(fx.dtype)
        ids = torch.stack([torch.randperm(fx.E, generator=g)[:topk] for _ in range(M)]).to(torch.int32)
        w = torch.softmax(torch.randn(M, topk, generator=g), dim=-1)
        ref = H.reference_moe(fx, x, w, ids)
        # same tolerances as the H88-B positive-scale emulation test
        self.assertLess(H.rel_diff(H.emulate_moe(layer, scheme, x, w, ids, fx.K, fx.N, fx.gs), ref), 0.08)
        self.assertLess(H.rel_diff(H.emulate_moe(layer, scheme, x, w, ids, fx.K, fx.N, fx.gs, a8=False), ref), 0.01)


# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------


def _build(scheme_cls, E, K, N, dtype, tensors, dev="cuda", gs=128):
    scheme = scheme_cls(H._quant_config(group_size=gs), weight_quant=H._weight_quant(gs))
    with torch.device(dev):
        layer = torch.nn.Module()
        layer.moe_tp_size = 1
        layer.layer_id = 0
        layer.num_local_experts = E
        scheme.create_weights(layer, E, K, N, dtype)
        for name in ("w13_weight_packed", "w2_weight_packed", "w13_weight_scale", "w2_weight_scale"):
            getattr(layer, name).data.copy_(tensors[name])
        scheme.process_weights_after_loading(layer)
    return scheme, layer


def _apply(scheme, layer, E, x, w, ids):
    scheme.moe_runner_config = SimpleNamespace(
        activation="silu", is_gated=True, routed_scaling_factor=None, swiglu_limit=None,
    )  # fmt: skip
    d = mock.Mock()
    d.hidden_states = x
    d.topk_output = (w, ids, torch.zeros(x.shape[0], E, device=x.device))
    return scheme.apply_weights(layer, d).hidden_states


def _unpack_k(p: torch.Tensor) -> torch.Tensor:
    """[K/8, N] int32 packed along K (CT pack-quantized, transposed) -> [K, N] values 0..15."""
    k8, n = p.shape
    q = torch.empty(k8 * 8, n, dtype=torch.int32, device=p.device)
    for i in range(8):
        q[i::8] = (p >> (4 * i)) & 0xF
    return q


def _dequant(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """symmetric uint4b8: (q - 8) * s, s [G, N] -> fp32 [K, N]."""
    q = _unpack_k(packed).float() - 8.0
    gs = q.shape[0] // scale.shape[0]
    return q * scale.float().repeat_interleave(gs, dim=0)


def _int8_rows(x: torch.Tensor) -> torch.Tensor:
    """per-token symmetric int8 round trip (absmax/127), fp32."""
    amax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
    sc = amax / 127.0
    return torch.round(x / sc).clamp(-128, 127) * sc


def _int16_grid(s: torch.Tensor) -> torch.Tensor:
    """the scales as the int16 x4096 kernel sees them (factor = max|s|/4096 over the whole tensor), fp32."""
    f = s.float().abs().max() / U.W4A8_SCALE_INT_RANGE
    return torch.round(s.float() / f) * f


def moe_fp32(t, x, w, ids, N, a8_emulation=False):
    """fp32 MoE (TF32 off) from the CT-layout tensors; a8_emulation: int8 per-token activations before both
    GEMMs and int16-grid scales (what the W4A8 kernel computes exactly, minus its bf16 roundings)."""
    s13 = _int16_grid(t["w13_weight_scale"]) if a8_emulation else t["w13_weight_scale"]
    s2 = _int16_grid(t["w2_weight_scale"]) if a8_emulation else t["w2_weight_scale"]
    xf = x.float()
    out = torch.zeros_like(xf)
    for e in torch.unique(ids[ids >= 0]).tolist():
        tok, slot = torch.nonzero(ids == e, as_tuple=True)
        w13 = _dequant(t["w13_weight_packed"][e], s13[e])
        w2 = _dequant(t["w2_weight_packed"][e], s2[e])
        xin = xf[tok]
        if a8_emulation:
            xin = _int8_rows(xin)
        h = xin @ w13
        act = torch.nn.functional.silu(h[:, :N]) * h[:, N:]
        if a8_emulation:
            act = _int8_rows(act)
        out.index_add_(0, tok, (act @ w2) * w[tok, slot].float().unsqueeze(1))
    return out


def tok_err(out: torch.Tensor, ref: torch.Tensor):
    """per-token relative L2 error ||o_t - r_t|| / ||r_t||: (mean, p99, max) and the mean-abs ratio of H88-B."""
    o, r = out.float(), ref.float()
    e = (o - r).norm(dim=1) / r.norm(dim=1).clamp(min=1e-30)
    p99 = float(torch.quantile(e, 0.99)) if e.numel() > 1 else float(e[0])
    return float(e.mean()), p99, float(e.max()), H.rel_diff(o.cpu(), r.cpu())


@unittest.skipUnless(GPU_TESTS, "needs a GPU window: H88_GPU_TESTS=1 CUDA_VISIBLE_DEVICES=<idx>")
class TestGpuSignedScalesOldVsNew(CustomTestCase):
    """TestGpuSmallMoeOldVsNew (H88-B) with signed scales; the thresholds are the same ones."""

    E, K, N, GS, TOPK = 8, 512, 256, 128, 2

    def _run(self, M):
        fx = signed_fixture(E=self.E, K=self.K, N=self.N, gs=self.GS, seed=21)
        g = torch.Generator().manual_seed(M)
        x = torch.randn(M, self.K, generator=g).to(fx.dtype).cuda()
        ids = torch.stack([torch.randperm(self.E, generator=g)[: self.TOPK] for _ in range(M)]).to(torch.int32).cuda()
        w = torch.softmax(torch.randn(M, self.TOPK, generator=g), dim=-1).cuda()
        ref = H.reference_moe(fx, x.cpu(), w.cpu(), ids.cpu())
        t = {
            "w13_weight_packed": fx.w13_packed, "w2_weight_packed": fx.w2_packed,
            "w13_weight_scale": fx.w13_scale, "w2_weight_scale": fx.w2_scale,
        }  # fmt: skip
        old = _build(CompressedTensorsWNA16MoE, self.E, self.K, self.N, fx.dtype, t)
        new = _build(CompressedTensorsWNA16A8MoE, self.E, self.K, self.N, fx.dtype, t)
        out_old = _apply(*old, self.E, x, w, ids).cpu()
        out_new = _apply(*new, self.E, x, w, ids).cpu()
        self.assertTrue(bool(torch.isfinite(out_new.float()).all()))
        d_old, d_new = H.rel_diff(out_old, ref), H.rel_diff(out_new, ref)
        print(f"[H88 scalefix gpu] signed M={M}: |A16-ref|={d_old:.4f} |A8-ref|={d_new:.4f}")
        self.assertLess(d_old, 0.03)
        self.assertLess(d_new, 0.08)

    def test_decode_batch(self):
        self._run(1)

    def test_small_batch(self):
        self._run(7)

    def test_prefill_chunk(self):
        self._run(256)


def load_nf_layer(ckpt: Path, layer: int):
    """The expert tensors of one NF layer in the CT/sglang MoE layout (w13 = [gate | up] along N, transposed
    packed [E, K/8, 2N]), plus router weight and the norm weight in front of the MLP (this checkpoint has no
    post_attention_layernorm; the MLP input passes mlp_hyper_connection.hc_norm)."""
    from safetensors import safe_open

    idx = json.loads((ckpt / "model.safetensors.index.json").read_text())["weight_map"]
    base = f"model.language_model.layers.{layer}."
    want = {k: f for k, f in idx.items() if k.startswith(base + "mlp.experts.") or k in (
        base + "mlp.gate.weight", base + "mlp_hyper_connection.hc_norm.weight")}  # fmt: skip
    raw = {}
    for f in sorted(set(want.values())):
        with safe_open(str(ckpt / f), framework="pt") as h:
            for k in want:
                if want[k] == f and not k.endswith("weight_shape"):
                    raw[k] = h.get_tensor(k)
    E = 1 + max(int(k[len(base + "mlp.experts.") :].split(".")[0]) for k in raw if ".experts." in k)

    def ex(e, proj, what):
        return raw[f"{base}mlp.experts.{e}.{proj}.{what}"]

    t = {
        "w13_weight_packed": torch.stack([torch.cat([ex(e, "gate_proj", "weight_packed").t(), ex(e, "up_proj", "weight_packed").t()], 1) for e in range(E)]),
        "w13_weight_scale": torch.stack([torch.cat([ex(e, "gate_proj", "weight_scale").t(), ex(e, "up_proj", "weight_scale").t()], 1) for e in range(E)]),
        "w2_weight_packed": torch.stack([ex(e, "down_proj", "weight_packed").t() for e in range(E)]),
        "w2_weight_scale": torch.stack([ex(e, "down_proj", "weight_scale").t() for e in range(E)]),
    }  # fmt: skip
    t = {k: v.contiguous() for k, v in t.items()}
    return t, raw[base + "mlp.gate.weight"], raw.get(base + "mlp_hyper_connection.hc_norm.weight")


@unittest.skipUnless(GPU_TESTS, "needs a GPU window: H88_GPU_TESTS=1 CUDA_VISIBLE_DEVICES=<idx>")
@unittest.skipUnless((NF_CKPT / "model.safetensors.index.json").is_file(), f"NF checkpoint not mounted at {NF_CKPT}")
class TestGpuNfRealLayerSignedScales(CustomTestCase):
    """The real layer: A16 and A8i against fp32, A8i against its own fp32 emulation (bound: module docstring)."""

    TOPK = 10
    MS = (1, 8, 64, 256)

    @classmethod
    def setUpClass(cls):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        t, gate, norm_w = load_nf_layer(NF_CKPT, NF_LAYER)
        cls.E, k8, n2 = t["w13_weight_packed"].shape
        cls.K, cls.N = k8 * 8, n2 // 2
        cls.dtype = t["w13_weight_scale"].dtype
        for name in ("w13_weight_scale", "w2_weight_scale"):
            s = t[name].float()
            print(
                f"[H88 scalefix nf L{NF_LAYER}] {name} {tuple(s.shape)} {t[name].dtype}: min {float(s.min()):.4f} "
                f"median {float(s.median()):.5f} max {float(s.max()):.4f} neg {float((s < 0).float().mean()):.3f} "
                f"zero {int((s == 0).sum())} |min|>max {bool(-s.min() > s.max())}"
            )
        cls.t = {k: v.cuda() for k, v in t.items()}
        cls.gate = gate.cuda().float()
        cls.norm_w = None if norm_w is None else norm_w.cuda().float()
        if cls.norm_w is not None:
            nw = cls.norm_w
            print(
                f"[H88 scalefix nf L{NF_LAYER}] hc_norm weight {tuple(nw.shape)}: mean {float(nw.mean()):.4f} "
                f"min {float(nw.min()):.4f} max {float(nw.max()):.4f}"
            )
        cls.a16 = _build(CompressedTensorsWNA16MoE, cls.E, cls.K, cls.N, cls.dtype, cls.t)
        cls.a8 = _build(CompressedTensorsWNA16A8MoE, cls.E, cls.K, cls.N, cls.dtype, cls.t)
        cls.rows = []

    @classmethod
    def tearDownClass(cls):
        for r in cls.rows:
            print(r)
        del cls.a16, cls.a8, cls.t
        torch.cuda.empty_cache()

    def _x(self, kind, M):
        g = torch.Generator().manual_seed(1000 * M + len(kind))
        x = torch.randn(M, self.K, generator=g)
        if kind == "normw":
            self.assertIsNotNone(self.norm_w, "hc_norm weight missing: the normw stand-in would silently equal gauss")
            # hc_norm normalises the n hyper-connection streams together ([n * K]); its per-channel mean over the
            # streams is the channel scaling stand-in (a stand-in for the real MLP input, not a hidden state)
            self.assertEqual(self.norm_w.numel() % self.K, 0)
            w = self.norm_w.cpu().reshape(-1, self.K).mean(dim=0)
            # zero-centred RMSNorm weights (applied as 1 + w) vs plain weights
            x = x * ((1.0 + w) if float(w.mean().abs()) < 0.3 else w)
        elif kind == "outlier":
            x[:, torch.tensor([7, 311, 1500, 2047]) % self.K] *= 30.0
        return x.to(self.dtype).cuda()

    def _route(self, x):
        logits = x.float() @ self.gate.t()
        w, ids = torch.topk(torch.softmax(logits, dim=-1), self.TOPK, dim=-1)
        return (w / w.sum(dim=-1, keepdim=True)).float(), ids.to(torch.int32)

    def test_real_layer_a8_vs_a16_vs_fp32(self):
        runs = []
        for kind in ("gauss", "normw", "outlier"):
            for M in self.MS:
                x = self._x(kind, M)
                w, ids = self._route(x)
                ref = moe_fp32(self.t, x, w, ids, self.N)
                emu = moe_fp32(self.t, x, w, ids, self.N, a8_emulation=True)
                o16 = _apply(*self.a16, self.E, x, w, ids)
                o8 = _apply(*self.a8, self.E, x, w, ids)
                torch.cuda.synchronize()
                e16, e8, eemu, e8emu = tok_err(o16, ref), tok_err(o8, ref), tok_err(emu, ref), tok_err(o8, emu)
                finite = bool(torch.isfinite(o8.float()).all())
                norm_ratio = float(o8.float().norm() / ref.norm())
                self.rows.append(
                    f"[H88 scalefix nf L{NF_LAYER}] {kind:7s} M={M:3d} |ref|={float(ref.norm()):.3g} |A8|/|ref|={norm_ratio:.4g} "
                    f"A16 mean/p99/max {e16[0]:.4f}/{e16[1]:.4f}/{e16[2]:.4f} (mabs {e16[3]:.4f})  "
                    f"A8i {e8[0]:.4g}/{e8[1]:.4g}/{e8[2]:.4g} (mabs {e8[3]:.4g})  "
                    f"A8emu {eemu[0]:.4f}/{eemu[1]:.4f}/{eemu[2]:.4f}  A8i-vs-emu {e8emu[0]:.4g}/{e8emu[1]:.4g}/{e8emu[2]:.4g}"
                )
                print(self.rows[-1])
                runs.append((kind, M, finite, norm_ratio, e16, e8, eemu, e8emu))
        # assertions after every row is printed (an old-tree run shows the whole table, not the first failure)
        for kind, M, finite, norm_ratio, e16, e8, eemu, e8emu in runs:
            tag = f"{kind} M={M}"
            self.assertTrue(finite, f"{tag}: non-finite A8 output")
            self.assertLess(abs(norm_ratio - 1.0), BOUND_NORM, f"{tag}: |A8|/|ref| = {norm_ratio}")
            for i, what in ((0, "mean"), (2, "max")):
                kf, ef = BOUND_KERNEL_VS_EMU[i], BOUND_EMU_FACTOR[i]
                bound = ef * eemu[i] + BOUND_A16_FACTOR * e16[i]
                self.assertLessEqual(e8[i], bound, f"{tag}: A8 {what} {e8[i]} > {ef}*emu {eemu[i]} + {BOUND_A16_FACTOR}*A16 {e16[i]}")
                self.assertLessEqual(e8emu[i], kf * eemu[i], f"{tag}: A8 vs its emulation {what} {e8emu[i]} > {kf}*{eemu[i]}")


if __name__ == "__main__":
    unittest.main()
