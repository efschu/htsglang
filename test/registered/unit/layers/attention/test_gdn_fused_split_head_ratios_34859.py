"""The fused GDN split serves the 27B's v/k head ratio 3 (#34859) on strided rows.

Ported from upstream sglang #34859 ("Qwen3.8-27B Model Support"), the part that
applies to this rig: ``fused_qkvzba_split_reshape_cat_contiguous`` walks a
non-power-of-two head group one head at a time (``V_POW2`` constexpr), and
qwen3_5 enables ratio 3 on CUDA. Before the port the 27B (16 k / 48 v heads)
took the unfused split + 2x .contiguous() + torch.cat + z copy on every GDN
layer of every decode / DFLASH verify step.

Carried along from upstream #37500 (missed by the fork's port of it,
e46a4dba51): the kernel reads rows with runtime ``qkvz_row_stride`` /
``ba_row_stride``. ``finalize_fused_in_proj`` (called for the NF model, whose
GDN ratio is also 3) hands the split two column views of ONE GEMM output, row
pitch = qkvz width + ba width; the old kernel assumed dense rows and would have
read the wrong rows for every token after the first once ratio 3 routes here.

Not ported from #34859 (not applicable on sm86/sm120 INT8): sm120 FP8 GEMV,
Hopper bf16 GEMV (SM90 only), SM90 FlashInfer GDN prefill default, NVFP4
SiLU+quant fusion, DSpark / qwen2_moe / modelopt hunks.

Pinned here (CPU):
1. the production kernel under TRITON_INTERPRET is bit-exact against plain
   slicing for ratios 1/2/3/4, for 27B uneven rank-local heads (5/15, 6/18),
   on dense rows and on fused-GEMM-strided rows (upstream's test re-aimed);
2. qwen3_5 routes decode/verify through the fused split for ratio 3 on CUDA,
   and prefill keeps the strided views (#36267).
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

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=40, suite="base-a-test-cpu")

_WORKER = textwrap.dedent("""
    import json, os
    os.environ["TRITON_INTERPRET"] = "1"
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "99")
    import torch
    from sglang.jit_kernel.triton.gdn_fused_proj import (
        fused_qkvzba_split_reshape_cat_contiguous as fused,
    )

    HEAD = 16  # power of two as in production (128); small for the interpreter

    def reference(qkvz, ba, hk, hv):
        tq, tv = hk * HEAD, hv * HEAD
        q = qkvz[:, :tq]
        k = qkvz[:, tq:2 * tq]
        v = qkvz[:, 2 * tq:2 * tq + tv]
        z = qkvz[:, 2 * tq + tv:2 * tq + 2 * tv]
        return (torch.cat((q, k, v), -1).contiguous(),
                z.reshape(-1, hv, HEAD).contiguous(),
                ba[:, :hv].contiguous(), ba[:, hv:2 * hv].contiguous())

    res = {}
    torch.manual_seed(0)
    for hk, hv in ((4, 4), (4, 8), (4, 12), (4, 16), (5, 15), (6, 18), (16, 48)):
        for strided in (False, True):
            T = 7
            wq = 2 * hk * HEAD + 2 * hv * HEAD
            wb = 2 * hv
            if strided:
                buf = torch.randn(T, wq + wb)
                qkvz, ba = buf[:, :wq], buf[:, wq:]
            else:
                qkvz, ba = torch.randn(T, wq), torch.randn(T, wb)
            got = fused(qkvz, ba, hk, hv, HEAD, HEAD)
            want = reference(qkvz, ba, hk, hv)
            res[f"{hk}/{hv}/{'strided' if strided else 'dense'}"] = all(
                torch.equal(g.reshape(-1), w.reshape(-1)) for g, w in zip(got, want)
            )
    print("__RESULT__" + json.dumps(res))
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


class TestFusedSplitHeadRatios(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.res = _probe()

    def test_bit_exact_for_every_ratio_dense_and_strided(self):
        self.assertEqual(len(self.res), 14)
        for case, ok in self.res.items():
            with self.subTest(case=case):
                self.assertTrue(ok)


HEAD = 16


class TestQwen35Dispatch(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from sglang.srt.models import qwen3_5

        cls.qwen3_5 = qwen3_5

    def gdn(self, k_heads, v_heads):
        gdn = object.__new__(self.qwen3_5.Qwen3_5GatedDeltaNet)
        torch.nn.Module.__init__(gdn)
        gdn.num_k_heads, gdn.num_v_heads = 16, 48
        gdn.local_num_k_heads, gdn.local_num_v_heads = k_heads, v_heads
        gdn.head_k_dim = gdn.head_v_dim = HEAD
        gdn.attn = lambda fb, mixed_qkv, a, b: torch.zeros(
            1, mixed_qkv.shape[0], v_heads, HEAD
        )
        gdn.norm = lambda x, z: x
        gdn.out_proj = lambda x: (x, None)
        width = 2 * k_heads * HEAD + 2 * v_heads * HEAD
        qkvz = torch.randn(3, width)
        ba = torch.randn(3, 2 * v_heads)
        gdn._forward_input_proj = lambda hidden_states: (qkvz, ba)
        return gdn

    def route(self, mode, k_heads=6, v_heads=18):
        calls = []

        def fused(*args):
            calls.append("fused")
            qkvz, ba, hk, hv, dk, dv = args
            t = qkvz.shape[0]
            return (
                torch.zeros(t, 2 * hk * dk + hv * dv),
                torch.zeros(t, hv, dv),
                torch.zeros(t, hv),
                torch.zeros(t, hv),
            )

        real_views = self.qwen3_5.qwen3_5_gdn_prefill_projection_views

        def views(*args):
            calls.append("views")
            return real_views(*args)

        q = self.qwen3_5
        with patch.object(q, "_is_cuda", True), patch.object(
            q, "_GDN_FUSED_QKVZBA_RATIOS", (1, 2, 3, 4)
        ), patch.object(
            q, "fused_qkvzba_split_reshape_cat_contiguous", fused
        ), patch.object(
            q, "qwen3_5_gdn_prefill_projection_views", views
        ):
            gdn = self.gdn(k_heads, v_heads)
            gdn.forward(torch.empty(3, 1), SimpleNamespace(forward_mode=mode))
        return calls

    def test_cuda_ratio_table_serves_ratio_3(self):
        # the module table is built at import from _is_cuda
        self.assertEqual(
            self.qwen3_5._GDN_FUSED_QKVZBA_RATIOS,
            (1, 2, 3, 4) if self.qwen3_5._is_cuda else (1, 2, 4),
        )

    def test_decode_and_verify_take_the_fused_split(self):
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        for mode in (ForwardMode.DECODE, ForwardMode.TARGET_VERIFY):
            with self.subTest(mode=mode):
                self.assertEqual(self.route(mode), ["fused"])

    def test_prefill_keeps_the_views(self):
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        self.assertEqual(self.route(ForwardMode.EXTEND), ["views"])


if __name__ == "__main__":
    unittest.main()
