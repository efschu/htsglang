# SPDX-License-Identifier: Apache-2.0
"""NF-GGUF AP G8 (2026-10-10): the vocabulary of a qwen4exp GGUF target is built QUANTIZED-RESIDENT.

THE HOLE (plan deskq/PLAN-GGUF-NF-1009.md section 5, "Gegenprobe" finding A; metal order section 0 point 1).
``Qwen4ExpModel._build_embed_tokens`` handed the vocabulary a quant_config only when ``getattr(quant_config, "config")`` was a
dict that NAMED the embedding (the compressed-tensors rule, Minachist ``re:.*embed_tokens``). ``GGUFConfig`` has no ``.config``
-> ``embed_tokens`` came out as a DENSE ``weight`` parameter, while the GGUF adapter ships ``token_embd`` as
``model.embed_tokens.qweight`` (+ ``qweight_type``): ``load_weights`` found no parameter, skipped the tensor with a warning and
the model ran without a vocabulary. (``lm_head`` never had the hole: ``Qwen3VLForConditionalGeneration`` hands it the GGUF
quant_config.)

THE FIX under test: ``qwen4_exp.gguf_vocab_config`` -- a GGUF target (isinstance ``GGUFConfig``) builds ``embed_tokens`` with
``GGUFEmbeddingMethod`` (packed rows, dequantized on gather); ``SGLANG_GGUF_DENSE_VOCAB=1`` stays a test hook that keeps the
module dense exactly like the loader and the lm_head honour it; every other quant_config is untouched.

What is checked (CPU, no checkpoint, no GPU; the one kernel op a CPU has not, ``ggml_dequantize``, is replaced by gguf-py's
own dequantizer -- the REFERENCE the lookup is compared against is gguf-py as well, computed independently of the sglang path):

* the decision function (GGUFConfig / dense hook / foreign configs / duck-typed name);
* the real ``_build_embed_tokens`` on a GGUF config: a ``qweight`` + ``qweight_type`` module, no ``weight``; the packed rows
  load through the parameter's own weight_loader; the lookup == the dequantized reference rows (Q8_0 = the real
  ``token_embd`` type); the lm_head (Q6_K = the real ``output`` type) holds the packed rows too and its logits == the
  dequantized-reference matmul;
* every vocabulary tensor the adapter yields (tiny qwen4exp GGUF of G1) has a parameter -- zero names left for the
  "not found in params_dict" skip;
* the shared-MTP draft: the draft's table is the deferred placeholder and ``set_embed_and_head_modules`` makes it read the
  TARGET's very module objects (``is``), both packed -- the share predicate of ``eagle_worker_v2`` is true;
* the defect itself, reproduced: with the GGUF detection removed (the pre-G8 behaviour) the module is dense and has NO
  parameter for ``model.embed_tokens.qweight``.
"""

from __future__ import annotations

import os

# BEFORE ``import torch`` (the G1 test set it after the import, where it does nothing: finding C of the cross-check)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import importlib.util
import pathlib
from types import SimpleNamespace
from unittest import mock

import gguf
import numpy as np
import pytest
import torch
from gguf.quants import dequantize, quantize

import sglang.srt.layers.vocab_parallel_embedding as vpe
from sglang.srt.layers.quantization import gguf as G
from sglang.srt.models import qwen4_exp as Q4X

Q = gguf.GGMLQuantizationType
HIDDEN = 256  # Q6_K block = 256 columns; the real model has 2560 (= 10 blocks)
VOCAB = 64
TP1 = SimpleNamespace(tp_rank=0, tp_size=1, attn_tp_rank=0, attn_tp_size=1)


# -- helpers --------------------------------------------------------------------------------------------------------------


def _random_q6_k(rows, cols, rng):
    """gguf-py can DEQUANTIZE Q6_K but not quantize it: random blocks of the real layout (ql[128] qh[64] scales int8[16]
    d fp16 = 210 B per 256 weights) with a finite small ``d``, decoded by gguf-py itself for the reference."""
    blocks = rng.integers(0, 256, size=(rows, cols // 256, 210), dtype=np.uint8)
    blocks[:, :, 208:] = np.frombuffer(np.float16(0.01).tobytes(), dtype=np.uint8)
    return blocks.reshape(rows, -1)


def _packed(qtype, rows=VOCAB, cols=HIDDEN, seed=1):
    rng = np.random.default_rng(seed)
    if qtype == Q.Q6_K:
        raw = _random_q6_k(rows, cols, rng)
    else:
        raw = quantize(rng.standard_normal((rows, cols), dtype=np.float32), qtype)
    ref = torch.from_numpy(dequantize(raw, qtype).copy())  # gguf-py: independent of the sglang path
    return torch.from_numpy(raw.copy()), ref


def _ref_dequantize(quant, qtype, m, n, dtype):
    """ggml_dequantize on a CPU: the CUDA op is flat (m*n elements in row order); gguf-py does the arithmetic."""
    out = dequantize(quant.cpu().numpy(), Q(int(qtype)))
    return torch.from_numpy(out.copy()).reshape(m, n).to(dtype)


def _stub_model(*, defer=False):
    return SimpleNamespace(
        pp_group=SimpleNamespace(is_first_rank=True, is_last_rank=True), _defer_embed=defer, _embed_quant_config=None
    )


def _build_embed(quant_config, *, defer=False):
    conf = SimpleNamespace(vocab_size=VOCAB, hidden_size=HIDDEN)
    with mock.patch.object(vpe, "get_parallel", return_value=TP1):
        return Q4X.Qwen4ExpModel._build_embed_tokens(_stub_model(defer=defer), conf, quant_config, "model")


def _load(module, **tensors):
    params = dict(module.named_parameters())
    with mock.patch.object(vpe, "get_parallel", return_value=TP1):
        for leaf, value in tensors.items():
            p = params[leaf]
            p.weight_loader(p, value)


@pytest.fixture(autouse=True)
def _no_dense_hook(monkeypatch):
    monkeypatch.delenv("SGLANG_GGUF_DENSE_VOCAB", raising=False)


@pytest.fixture
def cpu_dequant():
    with mock.patch.object(G, "ggml_dequantize", _ref_dequantize, create=True):
        yield


class _ForeignConfig:
    def __init__(self, name, config=None):
        self._n = name
        self.config = config

    def get_name(self):
        return self._n


# -- the decision ---------------------------------------------------------------------------------------------------------


def test_a_gguf_config_gets_a_quantized_vocab():
    cfg = G.GGUFConfig()
    assert Q4X.gguf_vocab_config(cfg) is cfg


def test_the_dense_vocab_hook_keeps_it_dense(monkeypatch):
    monkeypatch.setenv("SGLANG_GGUF_DENSE_VOCAB", "1")
    assert Q4X.gguf_vocab_config(G.GGUFConfig()) is None


def test_no_other_quantization_is_touched():
    assert Q4X.gguf_vocab_config(None) is None
    assert Q4X.gguf_vocab_config(_ForeignConfig("compressed-tensors", {"config_groups": {}})) is None
    assert Q4X.gguf_vocab_config(_ForeignConfig("modelopt_fp4")) is None
    # a duck-typed "gguf" that is not the GGUFConfig class is not trusted either
    assert Q4X.gguf_vocab_config(_ForeignConfig("gguf")) is None


def test_the_compressed_tensors_rule_is_unchanged_by_the_gguf_branch():
    """Minachist: a config group NAMES the vocab -> still the quant_config (the line the GGUF branch sits behind)."""
    ct = _ForeignConfig(
        "compressed-tensors", {"config_groups": {"group_3": {"targets": ["re:.*embed_tokens"]}}}
    )
    with mock.patch.object(Q4X, "VocabParallelEmbedding") as vp, mock.patch.object(
        Q4X, "skip_on_worker", return_value=None
    ), mock.patch.object(Q4X, "is_dp_attention_enabled", return_value=False), mock.patch.object(
        Q4X, "form_a_dense_is_unsharded", return_value=False
    ):
        Q4X.Qwen4ExpModel._build_embed_tokens(
            _stub_model(), SimpleNamespace(vocab_size=VOCAB, hidden_size=HIDDEN), ct, "model"
        )
    assert vp.call_args.kwargs["quant_config"] is ct


# -- the module the model builds ---------------------------------------------------------------------------------------------


def test_embed_tokens_is_packed_and_has_the_parameters_the_adapter_ships():
    emb = _build_embed(G.GGUFConfig())
    assert isinstance(emb.quant_method, G.GGUFEmbeddingMethod)
    names = sorted(n for n, _ in emb.named_parameters())
    assert names == ["qweight", "qweight_type"]  # no dense ``weight`` for the packed rows to fall past


def test_the_embedding_lookup_is_the_dequantized_reference_q8_0(cpu_dequant):
    """token_embd is Q8_0 in the unsloth header."""
    raw, ref = _packed(Q.Q8_0)
    emb = _build_embed(G.GGUFConfig())
    _load(emb, qweight_type=torch.tensor(int(Q.Q8_0)), qweight=raw)
    assert emb.qweight_type.weight_type == int(Q.Q8_0)
    ids = torch.tensor([[3, 0, 63], [5, 5, 17]])
    got = emb.quant_method.embedding(emb, ids)
    assert got.shape == (2, 3, HIDDEN)
    assert torch.equal(got.float(), ref[ids].to(got.dtype).float())


def test_lm_head_is_packed_and_its_logits_are_the_reference_q6_k(monkeypatch):
    """output is Q6_K in the unsloth header; the head is built under the GGUF quant_config by the VL wrapper."""
    raw, ref = _packed(Q.Q6_K, seed=2)
    with mock.patch.object(vpe, "get_parallel", return_value=TP1):
        head = vpe.ParallelLMHead(VOCAB, HIDDEN, quant_config=G.GGUFConfig(), prefix="lm_head")
    assert not hasattr(head, "weight") and head.qweight is not None
    _load(head, qweight_type=torch.tensor(int(Q.Q6_K)), qweight=raw)
    x = torch.randn(3, HIDDEN)

    def mat_vec(qweight, act, qtype, rows):  # the CUDA MMVQ kernel's contract: act @ dequant(W).T
        return act @ _ref_dequantize(qweight, qtype, rows, HIDDEN, act.dtype).T

    monkeypatch.setattr(G, "_mmvq_safe_for_device", lambda: 16)
    monkeypatch.setattr(G, "_kquant_prefers_mmq", lambda *a, **k: False)
    monkeypatch.setattr(G, "_mmq_threshold_prefers_mmq", lambda *a, **k: False)
    monkeypatch.setattr(G, "ggml_mul_mat_vec_a8", mat_vec, raising=False)
    got = head.quant_method.apply(head, x)
    want = x @ ref.T
    assert got.shape == (3, VOCAB)
    torch.testing.assert_close(got, want, atol=1e-3, rtol=1e-4)


def test_the_defect_pre_g8_dense_module_and_no_parameter_for_the_packed_rows(monkeypatch):
    """The behaviour G8 removes: without the GGUF detection ``embed_tokens`` is a dense ``weight`` and the loader has no
    parameter for ``model.embed_tokens.qweight`` (it skipped it with a warning)."""
    monkeypatch.setattr(Q4X, "gguf_vocab_config", lambda cfg: None)  # the mutant: GGUFConfig detection removed
    emb = _build_embed(G.GGUFConfig())
    assert [n for n, _ in emb.named_parameters()] == ["weight"]
    assert "qweight" not in dict(emb.named_parameters())


def test_the_dense_hook_builds_the_dense_module(monkeypatch):
    monkeypatch.setenv("SGLANG_GGUF_DENSE_VOCAB", "1")
    emb = _build_embed(G.GGUFConfig())
    assert [n for n, _ in emb.named_parameters()] == ["weight"]


# -- the adapter's stream against the module's parameters ---------------------------------------------------------------------


def _g1():
    spec = importlib.util.spec_from_file_location(
        "g8_g1_helpers",
        pathlib.Path(__file__).resolve().parents[1] / "model_loader" / "test_gguf_qwen4exp_g1_1009.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_every_vocabulary_tensor_the_adapter_yields_has_a_parameter(tmp_path):
    """Warnings counted: the names the GGUF stream carries for the vocabulary, minus the parameters of the two modules,
    is empty -- load_weights' skip-with-warning has nothing to skip."""
    g1 = _g1()
    path = str(tmp_path / "m-00001-of-00001.gguf")
    g1.build_tiny(path)
    _, _, out = g1.run_pipeline(path)
    stream = {n for n in out if n.startswith(("model.embed_tokens.", "lm_head."))}
    assert stream == {
        "model.embed_tokens.qweight",
        "model.embed_tokens.qweight_type",
        "lm_head.qweight",
        "lm_head.qweight_type",
    }
    emb = _build_embed(G.GGUFConfig())
    with mock.patch.object(vpe, "get_parallel", return_value=TP1):
        head = vpe.ParallelLMHead(VOCAB, HIDDEN, quant_config=G.GGUFConfig(), prefix="lm_head")
    params = {f"model.embed_tokens.{n}" for n, _ in emb.named_parameters()} | {
        f"lm_head.{n}" for n, _ in head.named_parameters()
    }
    assert stream - params == set()


# -- the shared-MTP draft -----------------------------------------------------------------------------------------------------


def test_the_draft_reads_the_targets_modules_not_tables_of_its_own():
    from sglang.srt.models.mtp_vocab_share import MtpEmbedDeferred
    from sglang.srt.models.qwen3_5_mtp import Qwen3_5ForCausalLMMTP
    from sglang.srt.speculative.eagle_worker_v2 import target_shares_vocab_modules

    # the draft (is_nextn, vocab shared): the deferred placeholder, built under a GGUF quant_config as well
    with mock.patch.object(Q4X, "skip_on_worker", return_value=None):
        assert isinstance(_build_embed(G.GGUFConfig(), defer=True), MtpEmbedDeferred)

    t_embed = _build_embed(G.GGUFConfig())
    with mock.patch.object(vpe, "get_parallel", return_value=TP1):
        t_head = vpe.ParallelLMHead(VOCAB, HIDDEN, quant_config=G.GGUFConfig(), prefix="lm_head")
    # eagle_worker_v2.init_lm_head: both target modules are packed, so the MODULES are shared
    assert not hasattr(t_embed, "weight") and not hasattr(t_head, "weight")
    assert target_shares_vocab_modules(t_head, t_embed)

    draft = SimpleNamespace(
        model=SimpleNamespace(embed_tokens=MtpEmbedDeferred()),
        lm_head=None,
        config=SimpleNamespace(tie_word_embeddings=False),
    )
    with mock.patch.object(torch.cuda, "empty_cache"), mock.patch.object(torch.cuda, "synchronize"):
        Qwen3_5ForCausalLMMTP.set_embed_and_head_modules(draft, t_embed, t_head)
    assert draft.model.embed_tokens is t_embed
    assert draft.lm_head is t_head


def test_the_vocabulary_price_is_the_packed_price_not_the_dense_one():
    """The memory price of the decision, from the real header's shape: 248 320 x 2 560 (token_embd Q8_0, output Q6_K).
    Packed: 34/32 and 210/256 bytes per element; dense bf16: 2 bytes. The numbers are ``gguf.GGML_QUANT_SIZES``'."""
    rows, cols = 248320, 2560
    mib = float(1 << 20)

    def packed(qt):
        block, size = gguf.GGML_QUANT_SIZES[qt]
        return rows * cols // block * size / mib

    dense = rows * cols * 2 / mib
    assert round(packed(Q.Q8_0), 1) == 644.1 and round(packed(Q.Q6_K), 1) == 497.3
    assert round(dense, 1) == 1212.5  # = the "1212.5 MiB" the draft vocab table is priced at in qwen3_5_mtp.py
    assert packed(Q.Q8_0) + packed(Q.Q6_K) < dense  # one dense table alone is dearer than BOTH packed ones


_REAL_DIR = "/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-Flash-Next-GGUF-unsloth/UD-IQ4_XS"


@pytest.mark.skipif(not os.path.isdir(_REAL_DIR), reason="unsloth qwen4exp export not on this machine")
def test_the_real_header_types_are_the_ones_the_module_and_the_price_assume():
    """token_embd Q8_0, output Q6_K, both [2560, 248320] (ne), in the 3-part export; bytes == the packed price above.
    Header only (GGUFReader mmaps; no tensor byte is read)."""
    want = {"token_embd.weight": ("Q8_0", 675430400), "output.weight": ("Q6_K", 521472000)}
    seen = {}
    for fn in sorted(os.listdir(_REAL_DIR)):
        if not fn.endswith(".gguf"):
            continue
        for t in gguf.GGUFReader(os.path.join(_REAL_DIR, fn), "r").tensors:
            if t.name in want:
                seen[str(t.name)] = (t.tensor_type.name, int(t.n_bytes), [int(x) for x in t.shape])
    assert {k: v[:2] for k, v in seen.items()} == want
    assert all(v[2] == [2560, 248320] for v in seen.values())
    # both types are ones the vocabulary methods serve
    assert int(Q.Q8_0) in G.DEQUANT_TYPES and int(Q.Q6_K) in G.DEQUANT_TYPES
    assert Q.Q6_K in G.MMVQ_QUANT_TYPES
