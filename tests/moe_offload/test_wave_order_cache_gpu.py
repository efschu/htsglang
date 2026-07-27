# SPDX-License-Identifier: Apache-2.0
"""End-to-end GPU gate for MoEExpertOffloadCache.run_waves under both wave
orders (#254).

test_wave_order_gpu.py proves the kernel-level equivalence in isolation; this
one drives the real cache -- install(), the pinned spill pool, the per-wave
resolve/fetch/LUT/remap and the k-slot combine -- over a stub FusedMoE layer
with the fp8-blockwise expert shapes the offloaded model uses.

Gates, all against the SAME layer with no offload at all:
  * token wave order  -> bit-identical (regression guard for today's path)
  * expert wave order -> bit-identical
  * expert order streams each spill expert exactly once, i.e. strictly less
    H2D than token order (the reason the flag exists)

Run:
  LD_LIBRARY_PATH=<venv nvidia libs> python -m pytest \
      tests/moe_offload/test_wave_order_cache_gpu.py -q
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="offload cache needs CUDA"
)

if torch.cuda.is_available():
    from sglang.srt.runtime_context import get_context
    from sglang.srt.server_args import ServerArgs

    if get_context()._server_args is None:
        get_context().set_server_args(ServerArgs(model_path="dummy"))

    from sglang.srt.environ import envs
    from sglang.srt.layers.moe.expert_offload import MoEExpertOffloadCache
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
        fused_experts_impl,
    )
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardCombineInput,
        StandardDispatchOutput,
    )
    from sglang.srt.layers.moe.topk import StandardTopKOutput

DEV = "cuda"
FP8_OK = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9

E, R_FRAC, H, INTER, TOPK, T = 64, 0.25, 1024, 512, 8, 512


class _StubLayer:
    """The subset of FusedMoE the offload cache and the apply touch."""

    layer_id = 0

    def __init__(self, w13, w2, w13_s, w2_s):
        self.w13_weight = w13
        self.w2_weight = w2
        self.w13_weight_scale_inv = w13_s
        self.w2_weight_scale_inv = w2_s
        self.num_local_experts = E
        self.moe_runner_config = MoeRunnerConfig(
            num_experts=E, num_local_experts=E, top_k=TOPK
        )

    def apply(self, dispatch_output):
        hs = dispatch_output.hidden_states
        tw, tid, _ = dispatch_output.topk_output
        out = fused_experts_impl(
            hs,
            self.w13_weight,
            self.w2_weight,
            tw,
            tid,
            inplace=False,
            use_fp8_w8a8=True,
            w1_scale=self.w13_weight_scale_inv,
            w2_scale=self.w2_weight_scale_inv,
            block_shape=[128, 128],
            routed_scaling_factor=self.moe_runner_config.routed_scaling_factor,
            filter_expert=True,
        )
        return StandardCombineInput(hidden_states=out)


def _build(seed=0):
    g = torch.Generator().manual_seed(seed)
    w13 = (
        (torch.randn(E, 2 * INTER, H, generator=g) * 0.1)
        .to(DEV)
        .to(torch.float8_e4m3fn)
    )
    w2 = (torch.randn(E, H, INTER, generator=g) * 0.1).to(DEV).to(torch.float8_e4m3fn)
    w13_s = (
        torch.rand(E, (2 * INTER + 127) // 128, (H + 127) // 128, generator=g) * 0.02
        + 0.01
    ).to(DEV)
    w2_s = (
        torch.rand(E, (H + 127) // 128, (INTER + 127) // 128, generator=g) * 0.02 + 0.01
    ).to(DEV)
    hidden = (torch.randn(T, H, generator=g) * 0.5).to(torch.bfloat16).to(DEV)
    logits = torch.randn(T, E, generator=g)
    tw, tid = torch.topk(torch.softmax(logits, dim=-1), TOPK, dim=-1)
    disp = StandardDispatchOutput(
        hidden_states=hidden,
        hidden_states_scale=None,
        topk_output=StandardTopKOutput(
            topk_weights=tw.float().to(DEV),
            topk_ids=tid.int().to(DEV),
            router_logits=logits.to(DEV),
        ),
    )
    return _StubLayer(w13, w2, w13_s, w2_s), disp


def _run(order, disp):
    layer, _ = _build()
    with envs.SGLANG_MOE_OFFLOAD_WAVE_ORDER.override(order):
        cache = MoEExpertOffloadCache(layer, R_FRAC)
    cache.install()
    out = cache.run_waves(disp, layer.apply)
    return out.hidden_states, cache.planner.stats


@pytest.mark.skipif(not FP8_OK, reason="fp8e4nv unsupported on this GPU")
@pytest.mark.parametrize("order", ["token", "expert"])
def test_run_waves_is_bit_identical_to_no_offload(order):
    ref_layer, disp = _build()
    ref = ref_layer.apply(disp).hidden_states

    out, stats = _run(order, disp)
    assert stats.overflow_forwards == 1, "the case must actually overflow"
    assert torch.equal(
        ref.view(torch.int16), out.view(torch.int16)
    ), f"{order}-major run_waves diverged from the no-offload apply"


@pytest.mark.skipif(not FP8_OK, reason="fp8e4nv unsupported on this GPU")
def test_expert_order_streams_each_spill_expert_once():
    _, disp = _build()
    _, tok = _run("token", disp)
    _, exp = _run("expert", disp)

    routed = set(disp.topk_output.topk_ids.reshape(-1).tolist())
    # resident_count = ceil(0.25*E); the rest of the routed set is spill.
    resident = -(-int(R_FRAC * E) // 1)
    spill = len([e for e in routed if e >= resident])

    assert exp.fetches == spill, "expert order must fetch each spill expert once"
    assert tok.fetches > exp.fetches * 4, (
        f"token order should re-fetch heavily (got {tok.fetches} vs "
        f"{exp.fetches}); if this ever stops holding the flag has no purpose"
    )
    assert exp.h2d_bytes < tok.h2d_bytes
