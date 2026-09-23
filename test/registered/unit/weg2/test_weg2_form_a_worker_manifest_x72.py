"""fnFL2x72/x73/x74 (23.09.): a Form A WORKER publishes no tensor it never
loaded, no rank publishes its expert-stack ``weight_shape``, and the coverage
gate knows the same rule.

Bug regression.  ``rank_role.worker_keeps_parameter`` vetoes every
checkpoint tensor but the routed experts and the router on a Form A worker
(F3), so the worker's ``q_norm``/``k_norm``/``shared_expert_gate`` parameters
exist in the module tree with UNINITIALISED bytes.  ``card_inventory`` walked
``named_parameters()`` and published them anyway, and the exchange plan
prefers the CO-LOCATED source (``_pick_source``): at P's wake PP1/PP2 took
their attention norms and shared-expert gates from D TP1/TP2, the workers on
the same cards.  Seam digest x72 (P after its first wake): PP1 ``k_norm:3
q_norm:3 shared_expert_gate:11``, PP2 ``2/2/8``, PP0 (co-located with the
attention host) clean; x73 (D after its wake): TP0 the same classes in
layers 29-47 only -- the layers PP1/PP2 handed back.  Every P prefill runs
after a wake, so every flip needle since x52 was answered on garbage norms.

``mlp.experts.*_weight_shape`` is the second finding of the same digest:
29+29 pieces per P rank, the after-digest identical in every layer -- the
OTHER layout's logical expert shape, copied verbatim.  Read only at load;
never a source.  The dense layers' ``weight_shape`` did not move (x72) and
keep travelling.

x74: excluding a tensor from the plan made the coverage gate (W84) refuse it
as UNCOVERED on every P rank at the boot-time sleep -- the gate must read the
same rule (``TagCoverage.not_source``).

Hermetic: CPU tensors, no process group, no boot.
"""
from __future__ import annotations

import inspect
import os

import torch
from torch import nn

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt import rank_role as rr  # noqa: E402
from sglang.srt.managers import weg2_memory_saver as ms  # noqa: E402
from sglang.srt.weg2 import weight_exchange as wx  # noqa: E402
from sglang.srt.weg2 import weight_exchange_shadow as sh  # noqa: E402


class _Cfg:
    num_experts = 4


class _FakeModel:
    config = _Cfg()

    def __init__(self, pairs):
        self._params = list(pairs)

    def named_parameters(self):
        return list(self._params)

    def modules(self):
        return ()


def _p(*shape):
    return nn.Parameter(torch.zeros(*shape, dtype=torch.bfloat16), requires_grad=False)


def _params():
    return [
        ("model.layers.0.self_attn.q_norm.weight", _p(256)),
        ("model.layers.0.self_attn.k_norm.weight", _p(256)),
        ("model.layers.0.mlp.shared_expert_gate.weight", _p(1, 64)),
        ("model.layers.0.mlp.gate.weight", _p(4, 64)),
        ("model.layers.0.mlp.experts.w13_weight_packed", _p(4, 32, 16)),
        ("model.layers.0.mlp.experts.w13_weight_shape", _p(4, 2)),
        ("model.layers.0.linear_attn.out_proj.weight_shape", _p(2)),
    ]


def _inventory(monkeypatch, worker: bool):
    monkeypatch.setenv(ms.WEIGHT_CHUNK_ENV_LAYERS, "8")
    monkeypatch.setenv(ms.WEIGHT_CHUNK_ENV_COUNT, "8")
    monkeypatch.setattr(rr, "this_rank_is_form_a_worker", lambda: worker)
    monkeypatch.setattr(sh, "expert_buffer_tensors", lambda model: ())
    return sh.card_inventory(rank=1, model=_FakeModel(_params()))


def test_a_worker_publishes_only_what_it_loaded(monkeypatch):
    inv, skipped, walked, reason = _inventory(monkeypatch, worker=True)
    assert reason == "" and walked == 7
    assert sorted(g.name for g, _t in inv) == [
        "model.layers.0.mlp.experts.w13_weight_packed",
        "model.layers.0.mlp.gate.weight",
    ]
    assert sorted(skipped) == [
        ("model.layers.0.linear_attn.out_proj.weight_shape", "form-a-worker-unloaded"),
        ("model.layers.0.mlp.experts.w13_weight_shape", "layout-metadata"),
        ("model.layers.0.mlp.shared_expert_gate.weight", "form-a-worker-unloaded"),
        ("model.layers.0.self_attn.k_norm.weight", "form-a-worker-unloaded"),
        ("model.layers.0.self_attn.q_norm.weight", "form-a-worker-unloaded"),
    ]


def test_a_classic_rank_publishes_everything_but_the_expert_shape(monkeypatch):
    inv, skipped, walked, reason = _inventory(monkeypatch, worker=False)
    assert reason == "" and walked == 7
    assert skipped == [("model.layers.0.mlp.experts.w13_weight_shape", "layout-metadata")]
    assert len(inv) == 6


def test_the_veto_is_the_load_predicate_itself():
    """One authority: the manifest's skip is ``rank_role.worker_keeps_parameter``,
    not a second list of names."""
    assert not rr.worker_keeps_parameter("model.layers.3.self_attn.k_norm.weight", 4)
    assert not rr.worker_keeps_parameter("model.layers.3.mlp.shared_expert_gate.weight", 4)
    assert rr.worker_keeps_parameter("model.layers.3.mlp.experts.w2_weight_packed", 4)
    assert rr.worker_keeps_parameter("model.layers.3.mlp.gate.weight", 4)


def test_the_plan_walk_and_the_coverage_apply_the_same_rule_as_the_manifest():
    """Three walks over ``named_parameters()`` (the W80/W84/W19 family); all
    must ask ``not_a_source_here``."""
    for fn in (sh.derive_leg_plan, sh.card_inventory, wx.build_coverage):
        assert "not_a_source_here(model)" in inspect.getsource(fn), fn.__name__


class _WorkerModel(nn.Module):
    """A worker's layer as the module tree holds it: the never-loaded norm
    beside the routed expert stack."""

    config = _Cfg()

    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module()])
        layer = self.model.layers[0]
        layer.self_attn = nn.Module()
        layer.self_attn.k_norm = nn.Module()
        layer.self_attn.k_norm.weight = _p(256)
        layer.mlp = nn.Module()
        layer.mlp.experts = nn.Module()
        layer.mlp.experts.w13_weight_packed = _p(4, 32, 16)
        layer.mlp.experts.w13_weight_shape = _p(4, 2)


def test_x74_the_coverage_gate_counts_the_excluded_tensors_and_refuses_nothing(monkeypatch):
    """RED on 61ef4ebf8e: W84 Weg2XchgCoverageRefused 300/113/83 findings at
    P's boot-time sleep, every one an excluded ``weight_shape`` (UNCOVERED)."""
    monkeypatch.setenv(ms.WEIGHT_CHUNK_ENV_LAYERS, "8")
    monkeypatch.setenv(ms.WEIGHT_CHUNK_ENV_COUNT, "8")
    monkeypatch.setattr(rr, "this_rank_is_form_a_worker", lambda: True)
    model = _WorkerModel()
    planned = {}
    for name, p in model.named_parameters():
        if sh.not_a_source_here(model)(name) is None:
            tag = wx.tag_of_parameter_name(name, region_tag=wx.GPU_MEMORY_TYPE_WEIGHTS)
            planned.setdefault(tag, {})[name] = int(p.untyped_storage().nbytes())
    rows = wx.build_coverage(
        model, rank=1, planned_bytes_by_tag=planned, tag_bytes=lambda _t: 0,
        mode=wx.WEIGHT_SOURCE_EXCHANGE,
    )
    row = rows["weights_0"]
    assert row.uncovered == () and row.short == () and row.missing == ()
    assert row.not_source == (
        "model.layers.0.mlp.experts.w13_weight_shape",
        "model.layers.0.self_attn.k_norm.weight",
    )
    assert "not_source=2" in row.cover_line()
