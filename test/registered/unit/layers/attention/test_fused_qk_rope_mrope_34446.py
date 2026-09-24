"""The fused Qwen3.5 QK-norm+RoPE kernel honours mrope height/width (#34446).

Ported from upstream sglang #34446 ("[rotary] Fix the fused Qwen3.5 RoPE kernel
discarding mrope height and width"). Upstream's test needs a GPU; this file
drives the SAME production kernel (``_fused_qk_rmsnorm_rope_gate_kernel`` via
``fused_qk_gemma_rmsnorm_rope_gate``) through Triton's interpreter on CPU
tensors. It is the real kernel, not a mirror of it.

The bug
-------
Qwen3_5ForConditionalGeneration (the 27B's class) hands
``forward_batch.mrope_positions`` -- shape [3, T], rows temporal/height/width
-- to every full-attention layer, and on CUDA with ``attn_output_gate`` the
layer takes ``forward_prepare_cuda_fused``. The kernel read
``positions_ptr + token``, i.e. row 0: every rotary lane rotated by the
TEMPORAL position. For text t == h == w, so text was right; for image tokens
the height/width lanes were rotated by the wrong position.

What is pinned here
-------------------
* 1-D positions (the text-only path) still match the reference.
* [3, T] positions with the 27B's section ([11, 11, 10] interleaved, head_dim
  256, rotary_dim 64, pass-through tail) match ``MRotaryEmbedding.forward_native``
  on DISTINCT t/h/w rows; also [24, 20, 20] interleaved (no tail) and the
  contiguous layout. Positions come from a wider buffer, sliced like the CUDA
  graph runner's ``mrope_positions[:, :T]`` (row stride != T).
* Sensitivity: the row-0-only rotation differs from the reference on those
  rows, so a kernel that dropped h/w again would fail the comparison.
* Mismatched positions/map are rejected by the wrapper's assertions.

fp16, not bf16: the interpreter here cannot run bf16; the kernel's
``.to(out_dtype)`` rounding is the same code path under ``FP16=True``.

Why a subprocess: ``@triton.jit`` reads ``TRITON_INTERPRET`` at decoration
time and the module object is cached process-wide (see
test_gdn_chunk_h_pad_sentinel_611.py for the full reasoning).
"""

import json
import os
import subprocess
import sys
import textwrap
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

_WORKER = textwrap.dedent("""
    import json, os
    os.environ["TRITON_INTERPRET"] = "1"
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "99")
    from unittest.mock import patch

    import torch

    from sglang.srt.layers import fused_qk_rmsnorm_rope_gate as mod
    from sglang.srt.layers.rotary_embedding.mrope import MRotaryEmbedding
    from sglang.srt.server_args import (
        ServerArgs,
        set_global_server_args_for_scheduler,
    )

    # PDL is a CUDA launch attribute; the interpreter has no card to ask.
    mod._enable_pdl = lambda device: False
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
    cpu_patch = patch("sglang.srt.layers.rotary_embedding.base._is_cpu", True)
    cpu_patch.start()

    torch.manual_seed(0)
    EPS = 1e-6
    T, HQ, HKV = 5, 2, 1
    DT = torch.float16

    def gemma_norm(x, w):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + EPS)
        # the kernel rounds the normed value to the output dtype before RoPE
        return (xf * (1.0 + w.float())).to(DT).float()

    def build(section, head_dim, interleaved):
        return MRotaryEmbedding(
            head_size=head_dim,
            rotary_dim=2 * sum(section),
            max_position_embeddings=64,
            base=10000,
            is_neox_style=True,
            dtype=torch.float32,
            mrope_section=section,
            mrope_interleaved=interleaved,
        )

    def graph_buffer_positions():
        buf = torch.zeros(3, 4 * T, dtype=torch.int64)
        ar = torch.arange(T)
        buf[:, :T] = torch.stack([ar % 7, ar % 5 + 3, ar % 3 + 11])
        return buf[:, :T]

    def inputs(head_dim):
        q_gate = torch.randn(T, HQ * 2 * head_dim).to(DT)
        k = torch.randn(T, HKV * head_dim).to(DT)
        qw = (0.1 * torch.randn(head_dim)).to(DT)
        kw = (0.1 * torch.randn(head_dim)).to(DT)
        return q_gate, k, qw, kw

    def reference(rope, positions, q_gate, k, qw, kw, head_dim):
        packed = q_gate.view(T, HQ, 2 * head_dim)
        qn = gemma_norm(packed[..., :head_dim], qw).reshape(T, -1)
        kn = gemma_norm(k.view(T, HKV, head_dim), kw).reshape(T, -1)
        want_q, want_k = rope.forward_native(positions, qn, kn)
        return want_q, want_k, packed[..., head_dim:]

    def run_kernel(rope, positions, q_gate, k, qw, kw, head_dim, axis_map):
        return mod.fused_qk_gemma_rmsnorm_rope_gate(
            q_gate, k, qw, kw, rope.cos_sin_cache.float(), positions, EPS,
            HQ, HKV, head_dim, rope.rotary_dim, has_gate=True,
            mrope_axis_map=axis_map,
        )

    out = {}
    cases = {
        "qwen38_27b_interleaved": ([11, 11, 10], 256, True),
        "interleaved_no_tail": ([24, 20, 20], 128, True),
        "contiguous": ([11, 11, 10], 256, False),
    }
    for name, (section, head_dim, interleaved) in cases.items():
        rope = build(section, head_dim, interleaved)
        q_gate, k, qw, kw = inputs(head_dim)
        positions = graph_buffer_positions()
        assert positions.stride(0) == 4 * T and positions.stride(1) == 1
        want_q, want_k, want_gate = reference(
            rope, positions, q_gate, k, qw, kw, head_dim)
        q_out, k_out, gate_out = run_kernel(
            rope, positions, q_gate, k, qw, kw, head_dim, rope.axis_map)
        # the pre-#34446 semantics: every lane rotated by the temporal row
        t_only = positions[0].contiguous()
        old_q, _, _ = reference(
            rope, t_only.unsqueeze(0).expand(3, -1).contiguous(),
            q_gate, k, qw, kw, head_dim)
        out[name] = dict(
            q_err=float((q_out.float() - want_q).abs().max()),
            k_err=float((k_out.float() - want_k).abs().max()),
            gate_exact=bool(torch.equal(
                gate_out.reshape(T, HQ, head_dim), want_gate)),
            t_only_gap=float((old_q - want_q).abs().max()),
        )

    # 1-D positions: the text path, no map
    rope = build([11, 11, 10], 256, True)
    q_gate, k, qw, kw = inputs(256)
    flat = torch.arange(T, dtype=torch.int64) + 2
    want_q, want_k, _ = reference(rope, flat, q_gate, k, qw, kw, 256)
    q_out, k_out, _ = run_kernel(rope, flat, q_gate, k, qw, kw, 256, None)
    out["flat"] = dict(
        q_err=float((q_out.float() - want_q).abs().max()),
        k_err=float((k_out.float() - want_k).abs().max()),
    )

    # positions and map must come together
    rejected = []
    for positions, amap in (
        (flat.unsqueeze(0).repeat(3, 1), None),
        (flat, rope.axis_map),
        (graph_buffer_positions().t().contiguous().t(), rope.axis_map),
    ):
        try:
            run_kernel(rope, positions, q_gate, k, qw, kw, 256, amap)
            rejected.append(False)
        except AssertionError:
            rejected.append(True)
    out["rejected"] = rejected

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


TOL = 2e-2


class TestFusedQKRopeMrope34446(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        # One child run for the whole class: the interpreter is slow.
        cls.res = _probe()

    def test_text_path_unchanged(self):
        self.assertLess(self.res["flat"]["q_err"], TOL)
        self.assertLess(self.res["flat"]["k_err"], TOL)

    def test_mrope_matches_reference(self):
        for name in ("qwen38_27b_interleaved", "interleaved_no_tail", "contiguous"):
            with self.subTest(case=name):
                r = self.res[name]
                self.assertLess(r["q_err"], TOL)
                self.assertLess(r["k_err"], TOL)
                self.assertTrue(r["gate_exact"])

    def test_rows_are_distinct_enough_to_catch_the_defect(self):
        # If a kernel rotated by the temporal row only, its error against the
        # reference would be this gap -- far outside TOL (measured 0.26 for the
        # contiguous layout, whose h/w lanes sit at the low frequencies).
        for name in ("qwen38_27b_interleaved", "interleaved_no_tail", "contiguous"):
            with self.subTest(case=name):
                self.assertGreater(self.res[name]["t_only_gap"], 5 * TOL)

    def test_positions_and_map_apart_are_rejected(self):
        self.assertEqual(self.res["rejected"], [True, True, True])


if __name__ == "__main__":
    unittest.main()
