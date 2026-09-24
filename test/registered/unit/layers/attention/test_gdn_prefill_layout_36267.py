"""GDN prefill reads the Qwen3.5 in_proj output through strided views (#36267).

Ported from upstream sglang #36267 ("[Performance] Optimize Qwen3.5 GDN
prefill projection layouts"), the parts that run on this rig:

* ``qwen3_5_gdn_prefill_projection_views`` -- the prefill split is a set of
  views of the block-contiguous ``[q|k|v|z]`` / ``[b|a]`` projection output
  instead of the per-token split + ``torch.cat`` (+ b/a/z copies) the 27B
  (v/k head ratio 3) took on every GDN layer;
* ``rms_norm_gated`` accepts the strided [T, Hv, Dv] gate in place.

NOT ported (not applicable here): ``gdn_prefill_qkv_prepare_fwd`` and the
``gdn_flashinfer.py`` hunk -- the FlashInfer GDN prefill backend is only
auto-selected on SM90/SM100 (``is_sm100_supported`` = major 10); sm86/sm120
always run the Triton FLA path.

What is pinned here (CPU):
1. the views equal the unfused split the fork used for ratio 3, for the 27B
   geometry with UNEVEN rank-local heads (a 5/15 and a 6/18 rank), on a
   contiguous projection and on a strided one (a fused-GEMM row view);
2. ``Qwen3_5GatedDeltaNet.forward`` in EXTEND (views path) produces exactly
   what the DECODE-mode split path produces for the same projection, and the
   attention/norm see strided views (no copy) on the prefill path;
3. ``rms_norm_gated`` with a strided 3-D gate equals the contiguous 2-D gate,
   run through Triton's interpreter on the production kernel
   (``_layer_norm_fwd_1pass_kernel``), norm-before-gate both ways.

Upstream's own tests need a GPU (``test/registered/attention/
test_gdn_prefill_layout.py``); 1 and 3 are those tests re-aimed at CPU.
"""

import json
import os
import subprocess
import sys
import textwrap
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.jit_kernel.triton.gdn_fused_proj import (
    qwen3_5_gdn_prefill_projection_views,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models import qwen3_5
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

HEAD = 16  # small head dims keep the test fast; the layout logic is dim-free


def _unfused_reference(qkvz, ba, k_heads, v_heads, head_k, head_v):
    """The fork's ratio-3 path: fix_query_key_value_ordering + cat."""
    k_tp, v_tp = k_heads * head_k, v_heads * head_v
    q, k, v, z = qkvz.split([k_tp, k_tp, v_tp, v_tp], dim=-1)
    b, a = ba.split([v_heads, v_heads], dim=-1)
    z = z.reshape(z.size(0), -1, head_v)
    mixed_qkv = torch.cat((q, k, v), dim=-1)
    return mixed_qkv, z, b.contiguous(), a.contiguous()


def _projection(tokens, k_heads, v_heads, strided):
    width_qkvz = 2 * k_heads * HEAD + 2 * v_heads * HEAD
    width_ba = 2 * v_heads
    if strided:
        # like _forward_input_proj's fused GEMM: one buffer, two column views
        fused = torch.randn(tokens, width_qkvz + width_ba)
        return fused[:, :width_qkvz], fused[:, width_qkvz:]
    return torch.randn(tokens, width_qkvz), torch.randn(tokens, width_ba)


class TestProjectionViews(CustomTestCase):
    def test_views_equal_unfused_split_for_uneven_27b_ranks(self):
        # Qwen3.8-27B: 16 k heads, 48 v heads; uneven TP3 gives e.g. 5/15 and
        # 6/18 per rank -- the views must use the rank-LOCAL counts.
        for k_heads, v_heads in ((5, 15), (6, 18), (16, 48)):
            for strided in (False, True):
                with self.subTest(k=k_heads, v=v_heads, strided=strided):
                    qkvz, ba = _projection(9, k_heads, v_heads, strided)
                    got = qwen3_5_gdn_prefill_projection_views(
                        qkvz, ba, k_heads, v_heads, HEAD, HEAD
                    )
                    want = _unfused_reference(qkvz, ba, k_heads, v_heads, HEAD, HEAD)
                    for g, w in zip(got, want):
                        torch.testing.assert_close(g, w, atol=0, rtol=0)
                    mixed_qkv, z, b, a = got
                    # views, not copies: they share the projection's storage
                    self.assertEqual(mixed_qkv.data_ptr(), qkvz.data_ptr())
                    self.assertEqual(b.data_ptr(), ba.data_ptr())
                    self.assertEqual(z.stride(-1), 1)
                    self.assertEqual(z.shape, (9, v_heads, HEAD))


class _Attn:
    """Stands in for RadixLinearAttention: records what it is handed and
    returns a deterministic function of it, [1, T, Hv, Dv]."""

    def __init__(self, v_heads):
        self.v_heads = v_heads
        self.seen = None

    def __call__(self, forward_batch, mixed_qkv, a, b):
        self.seen = (mixed_qkv, a, b)
        # compute on dense copies: CPU sigmoid/tanh take a vectorized path on
        # dense and a scalar path on strided operands and can differ by an ulp;
        # what the view path must preserve is the VALUES, checked bit-exact
        mixed_qkv, a, b = mixed_qkv.contiguous(), a.contiguous(), b.contiguous()
        tokens = mixed_qkv.shape[0]
        v = mixed_qkv[:, -self.v_heads * HEAD :].reshape(tokens, self.v_heads, HEAD)
        gate = torch.sigmoid(a).unsqueeze(-1) + torch.tanh(b).unsqueeze(-1)
        # elementwise only: a reduction would sum a strided and a contiguous
        # operand in different orders and differ in the last ulp
        q0 = mixed_qkv[:, 0:1].unsqueeze(-1)
        return (v * gate + q0).unsqueeze(0)


class _Norm:
    """Torch reference of RMSNormGated (norm_before_gate, swish) that accepts
    both gate forms the forward may hand it."""

    def __init__(self):
        self.gate_seen = None
        self.weight = torch.randn(HEAD)

    def __call__(self, x, z):
        self.gate_seen = z
        z2 = z.reshape(-1, z.shape[-1])  # copies only here, in the reference
        rstd = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        return x * rstd * self.weight * torch.nn.functional.silu(z2)


def _gdn(k_heads, v_heads):
    gdn = object.__new__(qwen3_5.Qwen3_5GatedDeltaNet)
    torch.nn.Module.__init__(gdn)
    gdn.num_k_heads = 16
    gdn.num_v_heads = 48  # ratio 3 -> the decode path is the unfused split
    gdn.local_num_k_heads = k_heads
    gdn.local_num_v_heads = v_heads
    gdn.head_k_dim = HEAD
    gdn.head_v_dim = HEAD
    gdn.attn = _Attn(v_heads)
    gdn.norm = _Norm()
    gdn.out_proj = lambda x: (x, None)
    return gdn


class TestGatedDeltaNetForward(CustomTestCase):
    def run_mode(self, gdn, mode, qkvz, ba):
        gdn._forward_input_proj = lambda hidden_states: (qkvz, ba)
        fb = SimpleNamespace(forward_mode=mode)
        with patch.object(qwen3_5, "_is_cuda", True):
            return gdn.forward(torch.empty(qkvz.shape[0], 1), fb)

    def test_extend_views_path_matches_decode_split_path(self):
        for k_heads, v_heads in ((5, 15), (6, 18)):
            for strided in (False, True):
                with self.subTest(k=k_heads, v=v_heads, strided=strided):
                    qkvz, ba = _projection(7, k_heads, v_heads, strided)
                    gdn = _gdn(k_heads, v_heads)
                    out_extend = self.run_mode(gdn, ForwardMode.EXTEND, qkvz, ba)
                    mixed_qkv, a, b = gdn.attn.seen
                    gate = gdn.norm.gate_seen
                    # prefill hands strided views on (no split/cat/z copy)
                    self.assertEqual(mixed_qkv.data_ptr(), qkvz.data_ptr())
                    self.assertFalse(mixed_qkv.is_contiguous())
                    self.assertEqual(gate.dim(), 3)
                    self.assertFalse(gate.is_contiguous())

                    out_decode = self.run_mode(gdn, ForwardMode.DECODE, qkvz, ba)
                    self.assertEqual(gdn.norm.gate_seen.dim(), 2)
                    torch.testing.assert_close(out_extend, out_decode, atol=0, rtol=0)

    def test_target_verify_keeps_the_split_path(self):
        qkvz, ba = _projection(4, 6, 18, False)
        gdn = _gdn(6, 18)
        self.run_mode(gdn, ForwardMode.TARGET_VERIFY, qkvz, ba)
        mixed_qkv, _, _ = gdn.attn.seen
        self.assertTrue(mixed_qkv.is_contiguous())
        self.assertEqual(gdn.norm.gate_seen.dim(), 2)


_WORKER = textwrap.dedent("""
    import json, os
    os.environ["TRITON_INTERPRET"] = "1"
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "99")
    import torch
    from sglang.srt.layers.attention.fla import layernorm_gated as lg
    from sglang.jit_kernel.triton.gdn_fused_proj import (
        qwen3_5_gdn_prefill_projection_views,
    )

    # CPU interpreter: no SM count, no PDL, no CUDA device context.
    lg.calc_rows_per_block = lambda M, device: lg.MAX_ROWS_PER_BLOCK
    lg.is_arch_support_pdl = lambda: False
    lg.device_context = lambda device: __import__("contextlib").nullcontext()

    torch.manual_seed(0)
    T, HK, HV, D = 5, 2, 6, 16
    fused = torch.randn(T, 2 * HK * D + 2 * HV * D + 2 * HV + 3)
    qkvz = fused[:, : 2 * HK * D + 2 * HV * D]
    ba = fused[:, 2 * HK * D + 2 * HV * D : 2 * HK * D + 2 * HV * D + 2 * HV]
    _, z, _, _ = qwen3_5_gdn_prefill_projection_views(qkvz, ba, HK, HV, D, D)
    assert not z.is_contiguous() and z.dim() == 3
    x = torch.randn(T * HV, D)
    w = torch.randn(D)
    out = {}
    for nbg in (True, False):
        want = lg.rms_norm_gated(x=x, weight=w, bias=None,
                                 z=z.contiguous().view(T * HV, D),
                                 norm_before_gate=nbg, is_rms_norm=True)
        got = lg.rms_norm_gated(x=x, weight=w, bias=None, z=z,
                                norm_before_gate=nbg, is_rms_norm=True)
        # and against torch, so both kernel forms are checked, not just equal
        xf = x
        zf = z.contiguous().view(T * HV, D)
        if nbg:
            ref = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6) * w
            ref = ref * torch.nn.functional.silu(zf)
        else:
            g = xf * torch.nn.functional.silu(zf)
            ref = g * torch.rsqrt(g.pow(2).mean(-1, keepdim=True) + 1e-6) * w
        out[str(nbg)] = dict(
            exact=bool(torch.equal(got, want)),
            ref_err=float((got - ref).abs().max()),
        )
    print("__RESULT__" + json.dumps(out))
    """)


def _probe():
    env = dict(os.environ)
    env["TRITON_INTERPRET"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = "99"
    proc = subprocess.run(
        [sys.executable, "-c", _WORKER],
        capture_output=True,
        text=True,
        timeout=900,
        env=env,
    )
    marker = [
        line for line in proc.stdout.splitlines() if line.startswith("__RESULT__")
    ]
    if not marker:
        raise AssertionError(
            "interpreter probe produced no result\n"
            f"exit={proc.returncode}\nstdout tail:\n{proc.stdout[-2000:]}\n"
            f"stderr tail:\n{proc.stderr[-3000:]}"
        )
    return json.loads(marker[-1][len("__RESULT__") :])


class TestStridedGateNorm(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.res = _probe()

    def test_strided_gate_matches_contiguous_gate(self):
        for nbg in ("True", "False"):
            with self.subTest(norm_before_gate=nbg):
                self.assertTrue(self.res[nbg]["exact"])
                self.assertLess(self.res[nbg]["ref_err"], 1e-4)


if __name__ == "__main__":
    unittest.main()
