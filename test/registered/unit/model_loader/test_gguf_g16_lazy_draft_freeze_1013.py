# SPDX-License-Identifier: Apache-2.0
"""NF-GGUF G16 (2026-10-13): ``freeze_parameters`` must not touch a not-yet-materialized GGUF parameter.

Metal G15 stage A attempt 5 (``boot_weg2_dkrnfdggufdbar110130724_f20d3aecff_1013_072512.D.log`` :6139-:6187, 13.10. 07:33:57Z): the G15 fix
itself killed the boot on TP1 and TP2 right after ``Load weight end``::

    qwen4_exp_mtp.py:164 __init__ -> :188 freeze_parameters -> self.requires_grad_(False)
    ValueError: Attempted to use an uninitialized parameter in <method 'requires_grad_' of 'torch._C.TensorBase' objects> ...
    GGUFUninitializedParameter

Root: a GGUF-quantized module is BORN with lazy parameters (``GGUFLinearMethod.create_weights`` registers ``qweight`` as a
``GGUFUninitializedParameter(requires_grad=False)``; ``load_weights`` materializes it). ``Module.requires_grad_`` calls
``p.requires_grad_`` on every parameter and torch refuses that call on a lazy one. The G15 test built the draft with plain torch backbone
modules only (``_FakeBackbone``), so it never met a lazy parameter. Ranks: TP0 builds the GGUF draft the same lazy way through the GGUF loader
(``loader.py`` ``_initialize_model`` then ``load_weights``); TP1/TP2 are the draft-solo shadow ranks, whose draft is a ``meta`` build without
``load_weights`` (``model_runner.py`` ``_build_weightless_worker_meta_model``) and which never run a draft forward.

What is pinned here, with a backbone that really carries a GGUF lazy linear:

* construction (real ``Qwen4ExpForCausalLMMTP.__init__``) does not raise, on CPU and on ``meta`` (the shadow build); the fusion and every
  materialized parameter are frozen, the lazy ones stay lazy (and are ``requires_grad=False`` by birth);
* after materialization + ``load_weights`` every parameter is ``requires_grad=False`` and the object is a plain ``Parameter``; a lazy
  parameter that had (artificially) been born trainable is frozen by the post-load pass;
* the metal error is reproduced by ``Module.requires_grad_`` on such a model (witness, independent of the fix);
* the G15 behaviour (fusion frozen, error chain of the fp8-KV write) stays in ``test_gguf_g15_draft_grad_1012.py``.
"""

from __future__ import annotations

import os
import types
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import gguf
import pytest
import torch
from torch import nn
from torch.nn.parameter import UninitializedParameter, is_lazy

from sglang.srt.layers.layernorm import GemmaRMSNorm
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.quantization import gguf as G
from sglang.srt.models import qwen4_exp_mtp as M
from sglang.srt.models.qwen3_5_mtp import Qwen3_5ForCausalLMMTP, lm_head_from_target

Q = gguf.GGMLQuantizationType

H = 16
HC = 2


class _GGUFBackbone(nn.Module):
    """Stands for ``Qwen4ExpModel`` of a GGUF draft: a Gemma norm, a plain ``nn.Linear`` and a REAL GGUF linear (lazy ``qweight``)."""

    def __init__(self, config, quant_config, prefix="", is_nextn=False):
        super().__init__()
        # the lazy GGUF linear is registered FIRST: a real layer stack has trainable-by-default modules (norms, plain linears) behind
        # its lazy ``qweight``, so a freeze that gives up at the first lazy parameter leaves them trainable (mutant M1)
        self.qkv = ReplicatedLinear(
            config.hidden_size, 2 * config.hidden_size, bias=False, quant_config=G.GGUFConfig([]),
            prefix="qkv", params_dtype=torch.bfloat16,
        )
        self.norm = GemmaRMSNorm(config.hidden_size)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)


def _cfg(hc=HC):
    return types.SimpleNamespace(
        hidden_size=H, rms_norm_eps=1e-6, hc_count=hc, num_hidden_layers=49, layer_types=["full_attention"] * 49,
        full_attention_interval=4, ple_layer_ids=[1], tie_word_embeddings=False, vocab_size=32,
    )


def _build(hc=HC, backbone=_GGUFBackbone):
    """The REAL ``Qwen4ExpForCausalLMMTP.__init__``; only the backbone, parallel-state lookups and the logits processor are stubs."""
    with mock.patch.object(M, "get_parallel", lambda: types.SimpleNamespace(tp_size=1)), \
         mock.patch.object(M, "get_pp_group", lambda: None), \
         mock.patch.object(M, "Qwen4ExpModel", backbone), \
         mock.patch.object(M, "LogitsProcessor", lambda c: nn.Identity()), \
         lm_head_from_target():
        return M.Qwen4ExpForCausalLMMTP(_cfg(hc), None, "")


def _lazy(model):
    return sorted(n for n, p in model.named_parameters() if is_lazy(p))


def _trainable(model):
    return sorted(n for n, p in model.named_parameters() if p.requires_grad)


def _materialize_like_the_loader(model):
    """What ``Qwen3_5ForCausalLMMTP.load_weights`` ends up doing to the GGUF linear: the real ``weight_loader`` of ``qweight``."""
    params = dict(model.model.qkv.named_parameters())
    w = torch.randn(2 * H, H, generator=torch.Generator().manual_seed(16)).to(torch.bfloat16)
    params["qweight_type"].weight_loader(params["qweight_type"], torch.tensor(int(Q.BF16), dtype=torch.uint8))
    params["qweight"].weight_loader(params["qweight"], w)


def _parent_load(materialize=True, extra=None):
    def load(self, weights, is_mtp=False):
        if materialize:
            _materialize_like_the_loader(self)
        if extra is not None:
            extra(self)
        return {"qweight"}

    return load


# ---------------------------------------------------------------------------------------------------------------------------
# the metal error, independent of the fix
# ---------------------------------------------------------------------------------------------------------------------------


def test_witness_module_requires_grad_raises_on_a_lazy_gguf_parameter():
    """The call G15 made (``self.requires_grad_(False)``) on a module that carries a GGUF lazy parameter: the ValueError of D.log:6187."""
    layer = ReplicatedLinear(H, 2 * H, bias=False, quant_config=G.GGUFConfig([]), prefix="qkv", params_dtype=torch.bfloat16)
    assert isinstance(layer.qweight, G.GGUFUninitializedParameter) and is_lazy(layer.qweight)
    with pytest.raises(ValueError, match=r"Attempted to use an uninitialized parameter in <method 'requires_grad_'"):
        layer.requires_grad_(False)


def test_a_gguf_lazy_parameter_is_born_frozen_and_materialize_keeps_it():
    """Why skipping a lazy parameter leaves nothing trainable: ``requires_grad`` is readable on it and False; ``materialize`` keeps it."""
    layer = ReplicatedLinear(H, 2 * H, bias=False, quant_config=G.GGUFConfig([]), prefix="qkv", params_dtype=torch.bfloat16)
    assert layer.qweight.requires_grad is False
    layer.qweight.materialize((2 * H, H), dtype=torch.bfloat16)
    assert type(layer.qweight) is nn.Parameter and not is_lazy(layer.qweight)
    assert layer.qweight.requires_grad is False


# ---------------------------------------------------------------------------------------------------------------------------
# construction with lazy parameters present
# ---------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("hc", [HC, 1], ids=["hc2_two_projections", "hc1_single_fc"])
def test_construction_with_a_lazy_gguf_parameter_does_not_raise_and_freezes_the_rest(hc):
    """Red on f20d3aecff: ``__init__`` raised the metal ValueError. The materialized parameters (fusion, norms, plain linear) are frozen."""
    model = _build(hc)
    assert _lazy(model) == ["model.qkv.qweight"]
    assert _trainable(model) == []
    fusion = {"fc_embedding.weight", "fc_hidden.weight"} if hc > 1 else {"fc.weight"}
    names = {n for n, _ in model.named_parameters()}
    assert fusion | {"pre_fc_norm_embedding.weight", "pre_fc_norm_hidden.weight", "model.norm.weight", "model.proj.weight"} <= names


def test_the_lazy_parameter_stays_lazy_through_construction():
    model = _build()
    p = model.model.qkv.qweight
    assert isinstance(p, G.GGUFUninitializedParameter) and p.requires_grad is False


def test_a_draft_solo_shadow_meta_build_does_not_raise():
    """TP1/TP2 of the metal boot: the draft is built under ``torch.device('meta')`` and never loaded."""
    with torch.device("meta"):
        model = _build()
    assert _lazy(model) == ["model.qkv.qweight"]
    assert _trainable(model) == []
    assert model.fc_embedding.weight.device.type == "meta"


def test_freeze_parameters_is_idempotent_with_lazy_parameters():
    model = _build()
    model.freeze_parameters()
    model.freeze_parameters()
    assert _trainable(model) == [] and _lazy(model) == ["model.qkv.qweight"]


def test_freezing_with_a_lazy_parameter_changes_no_value():
    model = _build()
    ref = {n: p.detach().clone() for n, p in model.named_parameters() if not is_lazy(p)}
    model.freeze_parameters()
    assert all(torch.equal(ref[n], p) for n, p in model.named_parameters() if not is_lazy(p))


# ---------------------------------------------------------------------------------------------------------------------------
# materialization + load_weights
# ---------------------------------------------------------------------------------------------------------------------------


def test_after_materialization_and_load_weights_every_parameter_is_frozen_and_materialized():
    model = _build()
    with mock.patch.object(Qwen3_5ForCausalLMMTP, "load_weights", _parent_load()):
        loaded = model.load_weights(iter(()))
    assert loaded == {"qweight"}  # the parent's return value is passed through
    assert _lazy(model) == []
    assert type(model.model.qkv.qweight) is nn.Parameter
    assert _trainable(model) == []


def test_a_lazy_parameter_born_trainable_is_frozen_once_the_load_materialized_it():
    """The post-load pass is what covers a lazy parameter that was NOT born frozen (no sglang site does that -- the sweep in the G15 file
    pins it -- but the invariant 'a served draft has no trainable parameter' must not rest on that)."""
    model = _build()
    old = model.model.qkv.qweight
    born_trainable = G.GGUFUninitializedParameter(requires_grad=True)
    for k, v in vars(old).items():  # the loader attributes (weight_loader, ...) the real one carries
        setattr(born_trainable, k, v)
    model.model.qkv.register_parameter("qweight", born_trainable)
    assert is_lazy(born_trainable) and born_trainable.requires_grad is True
    model.freeze_parameters()  # must neither raise nor claim to have frozen it
    assert born_trainable.requires_grad is True and is_lazy(born_trainable)
    with mock.patch.object(Qwen3_5ForCausalLMMTP, "load_weights", _parent_load()):
        model.load_weights(iter(()))
    assert not is_lazy(model.model.qkv.qweight)
    assert _trainable(model) == []


def test_a_parameter_registered_by_the_load_next_to_lazy_ones_is_frozen():
    def extra(self):
        self.register_parameter("fresh_after_load", nn.Parameter(torch.zeros(3)))
        self.fc_embedding.weight.requires_grad_(True)

    model = _build()
    with mock.patch.object(Qwen3_5ForCausalLMMTP, "load_weights", _parent_load(extra=extra)):
        model.load_weights(iter(()))
    assert _trainable(model) == []


def test_load_weights_that_materializes_nothing_leaves_the_lazy_parameter_lazy_and_does_not_raise():
    """A rank whose load skips the module (not our case on metal, but the code must not turn that into a second crash)."""
    model = _build()
    with mock.patch.object(Qwen3_5ForCausalLMMTP, "load_weights", _parent_load(materialize=False)):
        model.load_weights(iter(()))
    assert _lazy(model) == ["model.qkv.qweight"] and _trainable(model) == []


def test_a_materialized_draft_runs_a_forward_under_grad_mode_without_a_graph():
    """The point of the freeze (G15 chain): grad mode ON, the materialized GGUF linear on the fusion output keeps no grad_fn."""
    model = _build()
    with mock.patch.object(Qwen3_5ForCausalLMMTP, "load_weights", _parent_load()):
        model.load_weights(iter(()))
    assert torch.is_grad_enabled()
    emb = torch.randn(24, H)
    hid = torch.randn(24, HC * H)
    fused = model._mtp_input_fusion(emb, hid).view(24, HC, H)[:, 0, :].to(torch.bfloat16)  # fusion params are fp32 here, the GGUF payload bf16
    with mock.patch.dict(os.environ, {"SGLANG_GGUF_MMVQ_SAFE": "2"}):
        out = model.model.qkv.quant_method.apply(model.model.qkv, fused)
    assert not fused.requires_grad and not out.requires_grad and out.grad_fn is None
