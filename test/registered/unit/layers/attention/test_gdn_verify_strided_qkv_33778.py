"""GDN target verify hands the verify kernel strided q/k/v views (#33778).

Ported from upstream sglang #33778 ("Avoid materializing GDN QKV tensors during
target verification", main only). On the 27B's D group every DFLASH verify
step ran ``fused_qkv_split_gdn_prefill`` on every GDN layer -- one copy kernel
per layer per step -- although the Triton verify kernel
(``fused_sigmoid_gating_delta_rule_update``) reads q/k/v through their token
strides. Now target verify takes ``torch.split`` views when the chosen verify
kernel opts in (``supports_strided_target_verify_qkv``); prefill keeps the
fused split (the FLA chunk kernels want dense tensors).

Adapted: the fork has no ReplaySSM spec-fold / circular verify routes, so the
routing is the dispatcher capability alone; the Triton opt-in is limited to the
CUDA/HIP Triton kernel (NPU/CPU/XPU substitute other implementations).

Pinned here (CPU):
1. dispatcher capability: Triton verify (linear chain and tree) opts in; a
   verify kernel without the attribute does not (upstream's policy tests);
2. ``GDNAttnBackend.forward_extend`` in TARGET_VERIFY hands the verify kernel
   views of the post-conv mixed_qkv (shared storage, token stride = qkv
   width) and never calls the fused split; EXTEND still calls the fused split;
3. the production verify kernel under TRITON_INTERPRET gives bit-identical
   output and intermediate states for strided views and dense copies (27B-like
   head ratio 3, DFLASH-like linear chain, 2 requests x 4 draft tokens).
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

register_cpu_ci(est_time=25, suite="base-a-test-cpu")


def _heavy():
    """Backend imports pull in the model runner; import them only in the
    classes that need them, so the interpreter class stays light (the rig's
    agent-test cgroup is capped at 3 GiB and shared)."""
    global gdn_backend, GDNAttnBackend, GDNKernelDispatcher, TritonGDNKernel
    global LinearAttnKernelBackend, MambaPool, ForwardMode
    from sglang.srt.layers.attention.linear import gdn_backend
    from sglang.srt.layers.attention.linear.gdn_backend import (
        GDNAttnBackend,
        GDNKernelDispatcher,
    )
    from sglang.srt.layers.attention.linear.kernels.gdn_triton import (
        TritonGDNKernel,
    )
    from sglang.srt.layers.attention.linear.utils import LinearAttnKernelBackend
    from sglang.srt.mem_cache.memory_pool import MambaPool
    from sglang.srt.model_executor.forward_batch_info import ForwardMode


class TestDispatcherCapability(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        _heavy()

    def test_triton_verify_opts_in_and_others_do_not(self):
        self.assertTrue(TritonGDNKernel.supports_strided_target_verify_qkv)
        dispatcher = GDNKernelDispatcher(
            LinearAttnKernelBackend.TRITON, LinearAttnKernelBackend.TRITON
        )
        self.assertTrue(dispatcher.target_verify_supports_strided_qkv(None))
        # tree drafts route to the Triton tree kernel
        self.assertTrue(dispatcher.target_verify_supports_strided_qkv(object()))
        dispatcher.verify_kernel = SimpleNamespace()  # no opt-in attribute
        self.assertFalse(dispatcher.target_verify_supports_strided_qkv(None))


H, HV, D = 2, 6, 16
Q_DIM, V_DIM = H * D, HV * D
QKV = 2 * Q_DIM + V_DIM


def _layer():
    return SimpleNamespace(
        layer_id=0,
        q_dim=Q_DIM,
        k_dim=Q_DIM,
        v_dim=V_DIM,
        num_q_heads=H,
        num_k_heads=H,
        num_v_heads=HV,
        head_q_dim=D,
        head_k_dim=D,
        head_v_dim=D,
        A_log=torch.zeros(HV),
        dt_bias=torch.zeros(HV),
        conv_weights=None,
        bias=None,
        activation="silu",
    )


def _backend(verify_capture):
    be = object.__new__(GDNAttnBackend)
    dispatcher = GDNKernelDispatcher(
        LinearAttnKernelBackend.TRITON, LinearAttnKernelBackend.TRITON
    )
    dispatcher.target_verify = lambda **kw: verify_capture.update(kw) or torch.zeros(1)
    dispatcher.extend = lambda **kw: (torch.zeros(1), None, None)
    be.kernel_dispatcher = dispatcher
    be.forward_metadata = SimpleNamespace(
        query_start_loc=torch.tensor([0, 4, 8], dtype=torch.int32),
        mamba_cache_indices=torch.tensor([0, 1], dtype=torch.int32),
        retrieve_next_token=None,
        retrieve_next_sibling=None,
        retrieve_parent_token=None,
        has_mamba_track_mask=False,
    )
    spec = object.__new__(MambaPool.SpeculativeState)  # frozen dataclass
    for name, value in (
        ("conv", [torch.zeros(2, QKV, 3)]),
        ("temporal", torch.zeros(2, HV, D, D)),
        ("intermediate_ssm", torch.zeros(2, 4, HV, D, D)),
        ("intermediate_conv_window", [torch.zeros(2, 4, QKV, 3)]),
    ):
        object.__setattr__(spec, name, value)
    be.req_to_token_pool = SimpleNamespace(mamba2_layer_cache=lambda layer_id: spec)
    be.verify_intermediate_state_indices = torch.tensor([0, 1], dtype=torch.int32)
    # the conv step is not what this test is about: pass the window through
    be._target_verify_conv = lambda layer, x, *args: x
    return be


class TestForwardExtendRouting(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        _heavy()

    def run_mode(self, mode):
        captured = {}
        be = _backend(captured)
        split_calls = []

        def fake_split(mixed_qkv, *args):
            split_calls.append(mixed_qkv)
            t = mixed_qkv.shape[0]
            q, k, v = torch.split(mixed_qkv.contiguous(), [Q_DIM, Q_DIM, V_DIM], -1)
            return (
                q.contiguous().view(1, t, H, D),
                k.contiguous().view(1, t, H, D),
                v.contiguous().view(1, t, HV, D),
            )

        mixed_qkv = torch.randn(8, QKV)
        fb = SimpleNamespace(
            forward_mode=mode,
            spec_info=SimpleNamespace(draft_token_num=4),
            extend_prefix_lens=torch.zeros(2, dtype=torch.int64),
            extend_seq_lens_cpu=[4, 4],
        )
        with patch.object(gdn_backend, "is_cuda", return_value=True), patch.object(
            gdn_backend, "fused_qkv_split_gdn_prefill", fake_split, create=True
        ), patch.object(
            gdn_backend, "causal_conv1d_fn", lambda x, *a, **k: x
        ), patch.object(
            gdn_backend,
            "fused_gdn_gating",
            lambda A_log, a, b, dt_bias: (a, b),
        ):
            be.forward_extend(
                _layer(),
                fb,
                mixed_qkv,
                a=torch.randn(8, HV),
                b=torch.randn(8, HV),
            )
        return mixed_qkv, captured, split_calls

    def test_target_verify_hands_views_to_the_verify_kernel(self):
        mixed_qkv, captured, split_calls = self.run_mode(ForwardMode.TARGET_VERIFY)
        self.assertEqual(split_calls, [])
        q, k, v = captured["q"], captured["k"], captured["v"]
        self.assertEqual(q.data_ptr(), mixed_qkv.data_ptr())
        self.assertEqual(q.shape, (1, 8, H, D))
        self.assertEqual(v.shape, (1, 8, HV, D))
        for t in (q, k, v):
            self.assertEqual(t.stride(1), QKV)  # token stride = mixed width
            self.assertEqual(t.stride(-1), 1)
        torch.testing.assert_close(
            torch.cat([q.reshape(8, -1), k.reshape(8, -1), v.reshape(8, -1)], -1),
            mixed_qkv,
            atol=0,
            rtol=0,
        )

    def test_extend_keeps_the_fused_split(self):
        _, _, split_calls = self.run_mode(ForwardMode.EXTEND)
        self.assertEqual(len(split_calls), 1)


_WORKER = textwrap.dedent("""
    import json, os
    os.environ["TRITON_INTERPRET"] = "1"
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "99")
    import torch
    from sglang.srt.layers.attention.fla.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update,
    )

    torch.manual_seed(0)
    NREQ, STEPS, H, HV, D = 2, 4, 2, 6, 16
    T = NREQ * STEPS
    QKV = 2 * H * D + HV * D
    mixed = torch.randn(T, QKV)
    q, k, v = torch.split(mixed, [H * D, H * D, HV * D], dim=-1)
    views = (q.view(1, T, H, D), k.view(1, T, H, D), v.view(1, T, HV, D))
    dense = tuple(t.contiguous() for t in views)
    assert views[0].stride(1) == QKV and dense[0].stride(1) == H * D

    A_log = torch.randn(HV) * 0.1
    dt_bias = torch.randn(HV) * 0.1
    a = torch.randn(T, HV)
    b = torch.randn(T, HV)
    cu = torch.tensor([0, STEPS, 2 * STEPS], dtype=torch.int32)
    idx = torch.tensor([0, 1], dtype=torch.int32)

    def run(qkv):
        state = torch.randn(2, HV, D, D, generator=torch.Generator().manual_seed(1))
        inter = torch.zeros(2, STEPS, HV, D, D)
        o = fused_sigmoid_gating_delta_rule_update(
            A_log=A_log, a=a, dt_bias=dt_bias, softplus_beta=1.0,
            softplus_threshold=20.0, q=qkv[0], k=qkv[1], v=qkv[2], b=b,
            initial_state_source=state, initial_state_indices=idx,
            use_qk_l2norm_in_kernel=True, cu_seqlens=cu, is_kda=False,
            disable_state_update=True, intermediate_states_buffer=inter,
            intermediate_state_indices=idx, cache_steps=STEPS,
            retrieve_parent_token=None,
        )
        return o, inter, state

    o_v, i_v, s_v = run(views)
    o_d, i_d, s_d = run(dense)
    print("__RESULT__" + json.dumps(dict(
        out_exact=bool(torch.equal(o_v, o_d)),
        inter_exact=bool(torch.equal(i_v, i_d)),
        state_untouched=bool(torch.equal(s_v, s_d)),
        out_nonzero=bool(o_v.abs().sum() > 0),
        inter_written=bool(i_v.abs().sum() > 0),
    )))
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


class TestVerifyKernelHonoursStrides(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.res = _probe()

    def test_strided_views_equal_dense_copies(self):
        self.assertTrue(self.res["out_nonzero"])
        self.assertTrue(self.res["inter_written"])
        self.assertTrue(self.res["out_exact"])
        self.assertTrue(self.res["inter_exact"])
        self.assertTrue(self.res["state_untouched"])


if __name__ == "__main__":
    unittest.main()
