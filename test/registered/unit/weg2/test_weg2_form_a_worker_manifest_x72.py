"""fnFL2x72/x73 (23.09.): a Form A WORKER publishes no tensor it never loaded,
and no rank publishes its layout-metadata tensors.

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

``*weight_shape`` is the second finding of the same digest: 29+29 pieces per
P rank, the after-digest identical in every layer -- the OTHER layout's
logical shape, copied verbatim.  Read only at load; never a source.

Guarded here: on a worker rank the inventory keeps exactly the tensors the
load kept and names the rest ``form-a-worker-unloaded``; every rank names
``weight_shape`` ``layout-metadata``; the plan's walk applies the same rule.
Hermetic: CPU tensors, no process group, no boot.
"""
from __future__ import annotations

import inspect
import os

import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt import rank_role as rr  # noqa: E402
from sglang.srt.managers import weg2_memory_saver as ms  # noqa: E402
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
    return torch.nn.Parameter(torch.zeros(*shape, dtype=torch.bfloat16), requires_grad=False)


def _params():
    return [
        ("model.layers.0.self_attn.q_norm.weight", _p(256)),
        ("model.layers.0.self_attn.k_norm.weight", _p(256)),
        ("model.layers.0.mlp.shared_expert_gate.weight", _p(1, 64)),
        ("model.layers.0.mlp.gate.weight", _p(4, 64)),
        ("model.layers.0.mlp.experts.w13_weight_packed", _p(4, 32, 16)),
        ("model.layers.0.mlp.experts.w13_weight_shape", _p(2)),
    ]


def _inventory(monkeypatch, worker: bool):
    monkeypatch.setenv(ms.WEIGHT_CHUNK_ENV_LAYERS, "8")
    monkeypatch.setenv(ms.WEIGHT_CHUNK_ENV_COUNT, "8")
    monkeypatch.setattr(rr, "this_rank_is_form_a_worker", lambda: worker)
    monkeypatch.setattr(sh, "expert_buffer_tensors", lambda model: ())
    return sh.card_inventory(rank=1, model=_FakeModel(_params()))


def test_a_worker_publishes_only_what_it_loaded(monkeypatch):
    inv, skipped, walked, reason = _inventory(monkeypatch, worker=True)
    assert reason == "" and walked == 6
    assert sorted(g.name for g, _t in inv) == [
        "model.layers.0.mlp.experts.w13_weight_packed",
        "model.layers.0.mlp.gate.weight",
    ]
    assert sorted(skipped) == [
        ("model.layers.0.mlp.experts.w13_weight_shape", "layout-metadata"),
        ("model.layers.0.mlp.shared_expert_gate.weight", "form-a-worker-unloaded"),
        ("model.layers.0.self_attn.k_norm.weight", "form-a-worker-unloaded"),
        ("model.layers.0.self_attn.q_norm.weight", "form-a-worker-unloaded"),
    ]


def test_a_classic_rank_publishes_everything_but_its_layout_metadata(monkeypatch):
    inv, skipped, walked, reason = _inventory(monkeypatch, worker=False)
    assert reason == "" and walked == 6
    assert skipped == [("model.layers.0.mlp.experts.w13_weight_shape", "layout-metadata")]
    assert len(inv) == 5


def test_the_veto_is_the_load_predicate_itself():
    """One authority: the manifest's skip is ``rank_role.worker_keeps_parameter``,
    not a second list of names."""
    assert not rr.worker_keeps_parameter("model.layers.3.self_attn.k_norm.weight", 4)
    assert not rr.worker_keeps_parameter("model.layers.3.mlp.shared_expert_gate.weight", 4)
    assert rr.worker_keeps_parameter("model.layers.3.mlp.experts.w2_weight_packed", 4)
    assert rr.worker_keeps_parameter("model.layers.3.mlp.gate.weight", 4)


def test_the_plan_walk_applies_the_same_rule_as_the_manifest():
    """The manifest and the plan are two walks over ``named_parameters()``
    (the W80/W84/W19 family); both must ask ``not_a_source_here``."""
    src = inspect.getsource(sh.derive_leg_plan)
    assert "not_a_source_here(model)" in src
    src = inspect.getsource(sh.card_inventory)
    assert "not_a_source_here(model)" in src
