# SPDX-License-Identifier: Apache-2.0
"""NF-GGUF G15 (2026-10-12): the draft-extend CUDA-graph capture died on an autograd error, not on a GGUF numeric.

Metal G14 stage A (``done/gguf-stufeA-g14-1012/...D.log`` :10214-:10302): after the G14 fix the target-verify and the draft-decode
captures ran; the draft-extend warm-up (bs=6, M=24) raised

    RuntimeError: Output 0 of ViewBackward0 is a view and is being modified inplace. This view is the output of a function that
    returns multiple views ...                                         (memory_pool.py:4559  ``cache_v.div_(v_scale)``)

Root, read from code and re-run here on CPU with the REAL classes: the MTP input fusion of ``Qwen4ExpForCausalLMMTP`` is built from plain
``nn.Linear`` (``fc_embedding`` / ``fc_hidden`` / ``fc``) whose weight is ``requires_grad=True`` by default -- every sglang linear is born
``requires_grad=False`` -- and the capture warm-up runs the draft forward without ``inference_mode``. The fused hidden state carries a
grad_fn; the GGUF dense / dequant ``x @ w.T`` (M > ``_MMQ_MAX_TOKENS``, ``fused_mul_mat_gguf``) is torch-native and keeps it; ``qkv.split``
makes ``k`` / ``v`` multi-output views of a grad tensor and the fp8-KV write divides them in place.

What is pinned here:

* every parameter of the draft is ``requires_grad=False`` after construction AND after ``load_weights`` (a post-load pass may register a
  fresh default ``Parameter``);
* the metal error itself, byte for byte, through the real fusion -> a real GGUF linear (BF16 payload, M = 24) -> split -> the real
  ``MHATokenToKVPool.set_kv_buffer`` (fp8 pool), raised for an unfrozen draft and gone for the frozen one;
* the sweep: the GGUF loader sites that build parameters (``gguf.py`` and the ``gguf_*.py`` loaders) all say ``requires_grad=False``; the
  parameters of a loaded GGUF module are frozen.

NOT checked here (metal only): that the draft-extend CAPTURE then completes, and why the draft-DECODE capture (M = 6, MMVQ kernel)
escaped the same error (the sgl_kernel ops need a card).
"""

from __future__ import annotations

import ast
import os
import pathlib
import types
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import gguf
import pytest
import torch
from torch import nn

from sglang.srt.layers.layernorm import GemmaRMSNorm
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.quantization import gguf as G
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.models import qwen4_exp_mtp as M
from sglang.srt.models.qwen3_5_mtp import Qwen3_5ForCausalLMMTP, lm_head_from_target

Q = gguf.GGMLQuantizationType

H = 16
HC = 2
HEAD_DIM = 8
KV_HEADS = 2
KV = KV_HEADS * HEAD_DIM  # 16
Q_ROWS = 4 * HEAD_DIM  # 32
TOKENS = 24  # bs 6 x num_tokens_per_bs 4: the metal warm-up shape


@pytest.fixture(autouse=True)
def _cpu_dispatch(monkeypatch):
    # fused_mul_mat_gguf sizes the MMVQ threshold from the CUDA device before the unquantized branch (as in the G14 tests)
    monkeypatch.setenv("SGLANG_GGUF_MMVQ_SAFE", "2")


class _FakeBackbone(nn.Module):
    """Stands for ``Qwen4ExpModel``: the plain-torch modules a real one carries (a Gemma norm, a bias-free linear)."""

    def __init__(self, config, quant_config, prefix="", is_nextn=False):
        super().__init__()
        self.norm = GemmaRMSNorm(config.hidden_size)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)


def _draft(dtype=torch.bfloat16):
    """The REAL ``Qwen4ExpForCausalLMMTP.__init__`` on CPU; only its backbone, parallel-state lookups and the logits processor are stubs."""
    cfg = types.SimpleNamespace(
        hidden_size=H, rms_norm_eps=1e-6, hc_count=HC, num_hidden_layers=49, layer_types=["full_attention"] * 49,
        full_attention_interval=4, ple_layer_ids=[1], tie_word_embeddings=False, vocab_size=32,
    )
    with mock.patch.object(M, "get_parallel", lambda: types.SimpleNamespace(tp_size=1)), \
         mock.patch.object(M, "get_pp_group", lambda: None), \
         mock.patch.object(M, "Qwen4ExpModel", _FakeBackbone), \
         mock.patch.object(M, "LogitsProcessor", lambda c: nn.Identity()), \
         lm_head_from_target():
        model = M.Qwen4ExpForCausalLMMTP(cfg, None, "")
    return model.to(dtype)


def _unfreeze(model):
    model.requires_grad_(True)  # what the constructor left behind before G15 (plain nn.Linear / Gemma norm default)
    return model


def _gguf_bf16_linear(n_out, n_in):
    """A real GGUF module whose payload is a typed BF16 matrix (the G14 form), loaded the way ``load_weights`` does."""
    layer = ReplicatedLinear(n_in, n_out, bias=False, quant_config=G.GGUFConfig([]), prefix="qkv", params_dtype=torch.bfloat16)
    params = dict(layer.named_parameters())
    g = torch.Generator().manual_seed(15)
    w = torch.randn(n_out, n_in, generator=g).to(torch.bfloat16)
    params["qweight_type"].weight_loader(params["qweight_type"], torch.tensor(int(Q.BF16), dtype=torch.uint8))
    params["qweight"].weight_loader(params["qweight"], w)
    layer.quant_method.process_weights_after_loading(layer)
    return layer


class _PoolStub:
    """The attributes ``MHATokenToKVPool.set_kv_buffer`` reads on an fp8 pool in its HND branch (the write itself is a plain index store)."""

    dtype = torch.float8_e4m3fn
    store_dtype = torch.float8_e4m3fn
    size = 64
    page_size = 1
    use_hnd = True
    k_buffer = None

    def __init__(self):
        self.k_buffer = [torch.zeros(64, KV_HEADS, 1, HEAD_DIM, dtype=torch.float8_e4m3fn)]
        self.v_buffer = [torch.zeros(64, KV_HEADS, 1, HEAD_DIM, dtype=torch.float8_e4m3fn)]

    def local_slot(self, layer_id):
        return layer_id


def _draft_extend_attention_write(model):
    """The metal chain, real code at every step it exists on CPU: MTP fusion -> GGUF qkv linear (M = 24) -> ``split`` -> fp8-KV write."""
    emb = torch.randn(TOKENS, H).to(torch.bfloat16)
    hid = torch.randn(TOKENS, HC * H).to(torch.bfloat16)
    fused = model._mtp_input_fusion(emb, hid)  # [24, HC*H]
    layer_input = fused.view(TOKENS, HC, H)[:, 0, :]  # the draft layer reads one hidden stream; a view, as in the model
    layer = _gguf_bf16_linear(Q_ROWS + 2 * KV, H)
    qkv = layer.quant_method.apply(layer, layer_input)
    _, k, v = qkv.split([Q_ROWS, KV, KV], dim=-1)
    # RadixAttention.forward: ``k.view(-1, tp_k_head_num, qk_head_dim)`` / ``v.view(-1, tp_v_head_num, v_head_dim)``
    k = k.view(-1, KV_HEADS, HEAD_DIM)
    v = v.view(-1, KV_HEADS, HEAD_DIM)
    pool = _PoolStub()
    loc = torch.arange(TOKENS)
    MHATokenToKVPool.set_kv_buffer(pool, None, loc, k, v, 1.0, 1.0, layer_id_override=0)
    return qkv


# ---------------------------------------------------------------------------------------------------------------------------
# the root: which tensors carry the gradient
# ---------------------------------------------------------------------------------------------------------------------------


def test_the_plain_torch_modules_of_the_fusion_are_the_trainable_parameters():
    """Witness for the root (kept independent of the fix): a freshly built plain ``nn.Linear`` / Gemma norm is trainable."""
    assert nn.Linear(4, 4, bias=False).weight.requires_grad
    assert GemmaRMSNorm(4).weight.requires_grad


def test_every_parameter_of_the_draft_is_frozen_after_construction():
    model = _draft()
    names = {n for n, _ in model.named_parameters()}
    # the four fusion parameters of the hc_count > 1 form are in the set that must be frozen
    assert {"fc_embedding.weight", "fc_hidden.weight", "pre_fc_norm_embedding.weight", "pre_fc_norm_hidden.weight"} <= names
    live = [n for n, p in model.named_parameters() if p.requires_grad]
    assert live == [], f"trainable draft parameters: {live}"


def test_the_single_stream_fusion_is_frozen_too():
    """hc_count == 1 builds ``fc`` (a [2H, H] ``nn.Linear``) instead of the two projections."""
    cfg = types.SimpleNamespace(
        hidden_size=H, rms_norm_eps=1e-6, hc_count=1, num_hidden_layers=49, layer_types=["full_attention"] * 49,
        full_attention_interval=4, ple_layer_ids=[1], tie_word_embeddings=False, vocab_size=32,
    )
    with mock.patch.object(M, "get_parallel", lambda: types.SimpleNamespace(tp_size=1)), \
         mock.patch.object(M, "get_pp_group", lambda: None), \
         mock.patch.object(M, "Qwen4ExpModel", _FakeBackbone), \
         mock.patch.object(M, "LogitsProcessor", lambda c: nn.Identity()), \
         lm_head_from_target():
        model = M.Qwen4ExpForCausalLMMTP(cfg, None, "")
    assert "fc.weight" in dict(model.named_parameters())
    assert [n for n, p in model.named_parameters() if p.requires_grad] == []


def test_a_parameter_registered_by_the_load_is_frozen_again():
    """A post-load pass may register a fresh default ``Parameter``; ``load_weights`` ends frozen either way."""
    model = _draft()

    def fake_parent_load(self, weights, is_mtp=False):
        self.register_parameter("fresh_after_load", nn.Parameter(torch.zeros(3)))  # requires_grad=True by default
        self.fc_embedding.weight.requires_grad_(True)
        return {"fresh_after_load"}

    with mock.patch.object(Qwen3_5ForCausalLMMTP, "load_weights", fake_parent_load):
        loaded = model.load_weights(iter(()))
    assert loaded == {"fresh_after_load"}  # the return value of the parent is passed through
    assert [n for n, p in model.named_parameters() if p.requires_grad] == []


def test_freezing_changes_no_value():
    model = _draft()
    ref = {n: p.detach().clone() for n, p in model.named_parameters()}
    model.freeze_parameters()
    assert all(torch.equal(ref[n], p) for n, p in model.named_parameters())


# ---------------------------------------------------------------------------------------------------------------------------
# the metal error itself
# ---------------------------------------------------------------------------------------------------------------------------


def test_the_metal_error_is_raised_for_an_unfrozen_draft():
    """Red on c4a9be13e8's draft (the state ``_unfreeze`` restores): the exact RuntimeError of D.log:10302, from the real
    ``MHATokenToKVPool.set_kv_buffer`` line ``cache_v.div_(v_scale)``."""
    model = _unfreeze(_draft())
    with pytest.raises(RuntimeError, match=r"Output 0 of ViewBackward0 is a view and is being modified inplace.*returns multiple views"):
        _draft_extend_attention_write(model)


def test_the_frozen_draft_writes_its_kv_under_grad_mode():
    """Green with the fix, with grad mode ON (the capture warm-up has no ``inference_mode``): no grad_fn anywhere in the chain."""
    assert torch.is_grad_enabled()
    model = _draft()
    qkv = _draft_extend_attention_write(model)
    assert not qkv.requires_grad and qkv.grad_fn is None


@pytest.mark.parametrize("m", [1, 4, 6, 24])
def test_the_frozen_draft_fusion_output_carries_no_graph_for_any_capture_m(m):
    model = _draft()
    out = model._mtp_input_fusion(torch.randn(m, H).to(torch.bfloat16), torch.randn(m, HC * H).to(torch.bfloat16))
    assert not out.requires_grad and out.grad_fn is None
    assert _unfreeze(_draft())._mtp_input_fusion(
        torch.randn(m, H).to(torch.bfloat16), torch.randn(m, HC * H).to(torch.bfloat16)
    ).requires_grad  # the unfrozen draft is the witness: the graph is there without the fix


def test_gguf_dense_apply_keeps_the_graph_of_a_grad_input_which_is_why_the_input_must_be_clean():
    """``fused_mul_mat_gguf`` for an unquantized payload is ``x @ qweight.T`` (gguf.py:1338): torch-native, so a grad input yields a grad
    output even though the GGUF weight itself is frozen. The fix is therefore on the input side (the draft), not the GGUF weights."""
    layer = _gguf_bf16_linear(Q_ROWS + 2 * KV, H)
    assert not layer.qweight.requires_grad
    x = torch.randn(TOKENS, H).to(torch.bfloat16).requires_grad_(True)
    assert layer.quant_method.apply(layer, x).requires_grad
    assert not layer.quant_method.apply(layer, x.detach()).requires_grad


# ---------------------------------------------------------------------------------------------------------------------------
# sweep: every place the GGUF loaders turn a tensor into a parameter
# ---------------------------------------------------------------------------------------------------------------------------

SRC = pathlib.Path(__file__).resolve()
while SRC.name != "test" and SRC.parent != SRC:
    SRC = SRC.parent
SRT = SRC.parent / "python" / "sglang" / "srt"

PARAM_CTORS = {"Parameter", "UninitializedParameter", "GGUFUninitializedParameter"}
GGUF_LOADER_FILES = [
    SRT / "layers" / "quantization" / "gguf.py",
    *sorted((SRT / "model_loader").glob("gguf*.py")),
    SRT / "model_loader" / "hibernate.py",
]


def _param_ctor_calls(path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
            if name in PARAM_CTORS:
                kw = {k.arg: ast.unparse(k.value) for k in node.keywords}
                yield node.lineno, name, kw.get("requires_grad")


def test_the_sweep_sees_the_loader_files_and_their_parameter_sites():
    for p in GGUF_LOADER_FILES:
        assert p.exists(), p
    assert sum(1 for _ in _param_ctor_calls(GGUF_LOADER_FILES[0])) >= 10  # gguf.py builds its qweight / qweight_type / flat params here


@pytest.mark.parametrize("path", GGUF_LOADER_FILES, ids=lambda p: p.name)
def test_no_gguf_loader_site_builds_a_trainable_parameter(path):
    bad = [(path.name, ln, name) for ln, name, rg in _param_ctor_calls(path) if rg != "False"]
    assert bad == [], f"Parameter built without requires_grad=False: {bad}"


def test_the_parameters_of_a_loaded_gguf_module_are_frozen():
    layer = _gguf_bf16_linear(Q_ROWS + 2 * KV, H)
    assert {n for n, _ in layer.named_parameters()} >= {"qweight", "qweight_type"}
    assert [n for n, p in layer.named_parameters() if p.requires_grad] == []
