"""#49 (20.09., fn8c9b): the three-level router probe must see EVERY path that
hands topk_weights to the MoE kernel (the layer-0 ROUTER-W event ran through
the single-wave path, where the probe was not wired), and the fp32-gate
discriminator must replace the bf16 gate GEMM only when asked."""

import inspect
import types

import torch

from sglang.srt.layers.moe import expert_offload as eo
from sglang.srt.models import qwen3_moe as qm


def test_the_router_probe_sits_on_all_three_kernel_paths():
    src = inspect.getsource(eo)
    for fn in ("_run_single_wave", "_run_waves_expert_major"):
        body = inspect.getsource(getattr(eo.__dict__.get("_ExpertOffloadRunner", object), fn, None) or _method(fn))
        assert "_router_probe(" in body, fn
    # the multi-wave (prefill overflow) path sits between the two in the module
    assert src.count("_router_probe(") >= 4  # def + 3 call sites


def _method(name):
    for obj in vars(eo).values():
        if isinstance(obj, type) and hasattr(obj, name):
            return getattr(obj, name)
    raise AssertionError(name)


def test_probe_accepts_missing_flat_weights(monkeypatch):
    seen = {}
    monkeypatch.setattr(eo, "marlin_stage_probe_on", lambda: True, raising=False)
    src = inspect.getsource(eo._router_probe)
    assert "if flat_weights is None:" in src
    assert "topk_weights.reshape(-1, 1)" in src


def test_gate_fp32_switch_parses_and_stays_off_by_default():
    assert qm.moe_gate_fp32_on({}) is False
    assert qm.moe_gate_fp32_on({"SGLANG_MOE_GATE_FP32": "1"}) is True
    assert qm.moe_gate_fp32_on({"SGLANG_MOE_GATE_FP32": "0"}) is False


def test_router_logits_take_the_fp32_gemm_only_when_asked(monkeypatch):
    w = torch.randn(8, 4, dtype=torch.bfloat16)
    x = torch.randn(3, 4, dtype=torch.bfloat16)
    calls = []

    def gate(h):
        calls.append(h.dtype)
        return torch.nn.functional.linear(h, w), None

    gate.weight = w
    monkeypatch.delenv("SGLANG_MOE_GATE_FP32", raising=False)
    out = qm.moe_router_logits(gate, x)
    assert calls == [torch.bfloat16] and out.dtype == torch.bfloat16
    monkeypatch.setenv("SGLANG_MOE_GATE_FP32", "1")
    out32 = qm.moe_router_logits(gate, x)
    assert len(calls) == 1  # the module gate was NOT called
    assert out32.dtype == torch.float32
    torch.testing.assert_close(out32, torch.nn.functional.linear(x.float(), w.float()))


def test_forward_normal_routes_through_the_helper():
    src = inspect.getsource(qm.Qwen3MoeSparseMoeBlock.forward_normal)
    assert "moe_router_logits(self.gate, hidden_states)" in src
