# SPDX-License-Identifier: Apache-2.0
"""NF-GGUF AP G3 (2026-10-10): the NEXTN/MTP draft head from its own GGUF
(``mtp-Qwen3.8-Flash-Next-{Q8_0,shared-Q8_0}.gguf`` of the unsloth export).

* a tiny ``GGUFWriter(path, "qwen4exp")`` draft (``block_count`` 49,
  ``nextn_predict_layers`` 1, the six ``nextn.*`` tensors + the 26 roles of a
  full-attention layer at ``blk.48``; both the shared and the self-contained
  variant) run through the real pipeline (``Qwen4ExpGGUFAdapter`` as a draft ->
  ``gguf_quant_weights_iterator`` -> ``transform_stream``) against the
  converter's forward math: ``eh_proj = cat([fc_embedding, fc_hidden], dim=1)``;
* the refusals (named, header level);
* the real headers of both MTP files (34 / 32 tensors) against the sibling
  config, and the real VALUES of the shared head against the safetensors
  ``mtp.*`` tensors of the nvidia checkpoint (same original model): eh_proj
  split and order, every Gemma norm, the fused indexer, hyper-connection
  mixers.

Converter reference: llama.cpp master ``conversion/qwen4exp.py`` (``_MTP_EXTRA``,
``filter_tensors``, the ``eh`` concat in ``modify_tensors``) and
``conversion/qwen.py`` (``_QwenMtpMixin.filter_tensors``, ``Qwen3NextModel.
modify_tensors`` ``+1`` rule), fetched 2026-10-10; see the module doc of
``gguf_qwen4exp.py``.
"""

from __future__ import annotations

import json
import os
import types

import numpy as np
import pytest
import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import gguf  # noqa: E402
from gguf.quants import dequantize, quantize  # noqa: E402

from sglang.srt.configs import load_config as load_config_mod  # noqa: E402
from sglang.srt.model_loader import gguf_qwen4exp as Q4  # noqa: E402
from sglang.srt.model_loader import gguf_shards  # noqa: E402
from sglang.srt.model_loader.weight_utils import gguf_quant_weights_iterator  # noqa: E402
from sglang.srt.weg2 import host_ledger as HL  # noqa: E402

Q = gguf.GGMLQuantizationType

# ---- tiny geometry (block-32 aligned so Q8_0 holds every row) -----------------
H = 64
HC = 2
HCH = HC * H
LOWRANK = 32
VOCAB = 96
HEADS, KV_HEADS, HEAD_DIM = 4, 2, 32
Q_ROWS = HEADS * HEAD_DIM * 2
IDX_HEADS, IDX_DIM = 2, 32
N_EXP, TOPK, FF = 4, 2, 32
BLOCK = 48  # block_count 49 - nextn_predict_layers 1, as the real files

REAL_MTP_DIR = (
    "/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-Flash-Next-GGUF-unsloth/MTP"
)
REAL_SHARED = os.path.join(REAL_MTP_DIR, "mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf")
REAL_FULL = os.path.join(REAL_MTP_DIR, "mtp-Qwen3.8-Flash-Next-Q8_0.gguf")
REAL_NVIDIA = (
    "/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-Flash-Next-NVFP4-nvidia"
)


def q8(t: torch.Tensor) -> np.ndarray:
    return quantize(t.float().numpy(), Q.Q8_0)


def bf16_bytes(t: torch.Tensor) -> np.ndarray:
    return quantize(t.float().numpy(), Q.BF16)


def deq(raw: np.ndarray, qtype=Q.Q8_0) -> torch.Tensor:
    return torch.from_numpy(dequantize(raw, qtype).copy())


class Ref:
    def __init__(self):
        self.t = {}

    def put(self, key, shape):
        g = torch.Generator().manual_seed(7000 + len(self.t))
        self.t[key] = torch.randn(*shape, generator=g)
        return self.t[key]


def draft_dir_tensors(ref: Ref, *, with_vocab: bool, block: int = BLOCK):
    """gguf name -> ndarray or (ndarray, qtype): the draft the converter makes."""
    out = {}

    def add(name, arr, qtype=None):
        out[name] = (arr, qtype) if qtype is not None else arr

    def norm(key, n):  # HF zero-centred gamma, the converter bakes +1
        return (ref.put(key, (n,)) + 1.0).numpy()

    if with_vocab:
        add("token_embd.weight", q8(ref.put("embed", (VOCAB, H))), Q.Q8_0)
        add("output.weight", q8(ref.put("lm_head", (VOCAB, H))), Q.Q8_0)
    p = f"blk.{block}."
    # eh_proj: torch [H, 2H] = cat([fc_embedding, fc_hidden], dim=1)  (converter)
    fe, fh = ref.put("fc_embedding", (H, H)), ref.put("fc_hidden", (H, H))
    add(p + "nextn.eh_proj.weight", q8(torch.cat([fe, fh], dim=1)), Q.Q8_0)
    add(p + "nextn.enorm.weight", norm("pre_fc_norm_embedding", H))
    add(p + "nextn.hnorm.weight", norm("pre_fc_norm_hidden", HCH))
    add(p + "nextn.hc_head_norm.weight", norm("mixer.hc_norm", HCH))
    add(p + "nextn.hc_head_down.weight", q8(ref.put("mixer.down", (LOWRANK, HCH))), Q.Q8_0)
    add(p + "nextn.hc_head_up.weight", q8(ref.put("mixer.up", (HCH, LOWRANK))), Q.Q8_0)
    for hc in ("attn", "ffn"):
        k = f"layer.{hc}"
        add(p + f"hc_{hc}_norm.weight", norm(k + ".norm", HCH))
        add(p + f"hc_{hc}_down.weight", q8(ref.put(k + ".down", (LOWRANK, HCH))), Q.Q8_0)
        add(p + f"hc_{hc}_up.weight", q8(ref.put(k + ".up", (HCH, LOWRANK))), Q.Q8_0)
        add(p + f"hc_{hc}_inject.weight", q8(ref.put(k + ".inject", (HC, HCH))), Q.Q8_0)
    add(p + "attn_q.weight", q8(ref.put("q", (Q_ROWS, H))), Q.Q8_0)
    add(p + "attn_k.weight", q8(ref.put("k", (KV_HEADS * HEAD_DIM, H))), Q.Q8_0)
    add(p + "attn_v.weight", q8(ref.put("v", (KV_HEADS * HEAD_DIM, H))), Q.Q8_0)
    add(p + "attn_output.weight", q8(ref.put("o", (H, HEADS * HEAD_DIM))), Q.Q8_0)
    add(p + "attn_q_norm.weight", norm("qn", HEAD_DIM))
    add(p + "attn_k_norm.weight", norm("kn", HEAD_DIM))
    n_q = IDX_HEADS * IDX_DIM
    qk = ref.put("idx", (n_q + IDX_DIM, H))
    add(p + "indexer.q_proj.weight", bf16_bytes(qk[:n_q]), Q.BF16)
    add(p + "indexer.k_proj.weight", bf16_bytes(qk[n_q:]), Q.BF16)
    add(p + "indexer.q_norm.weight", norm("iqn", IDX_DIM))
    add(p + "indexer.k_norm.weight", norm("ikn", IDX_DIM))
    add(p + "ffn_gate_inp.weight", ref.put("router", (N_EXP, H)).numpy())
    add(p + "ffn_gate_inp_shexp.weight", ref.put("sgate", (1, H)).squeeze(0).numpy())
    for nm, shp in (("gate", (FF, H)), ("up", (FF, H)), ("down", (H, FF))):
        add(p + f"ffn_{nm}_shexp.weight", q8(ref.put(f"sh.{nm}", shp)), Q.Q8_0)
        add(p + f"ffn_{nm}_exps.weight", q8(ref.put(f"ex.{nm}", (N_EXP,) + shp)), Q.Q8_0)
    return out


def write_draft(path, *, shared=False, drop=(), extra=(), nextn=1, block_count=BLOCK + 1,
                tensors=None, ref=None):
    ref = ref or Ref()
    ts = tensors if tensors is not None else draft_dir_tensors(ref, with_vocab=not shared)
    for n in drop:
        ts.pop(n)
    for name, arr in extra:
        ts[name] = arr
    w = gguf.GGUFWriter(str(path), "qwen4exp")
    w.add_block_count(block_count)
    w.add_embedding_length(H)
    w.add_head_count(HEADS)
    w.add_head_count_kv(KV_HEADS)
    w.add_key_length(HEAD_DIM)
    w.add_value_length(HEAD_DIM)
    for key, val in {
        "expert_count": N_EXP,
        "expert_used_count": TOPK,
        "expert_feed_forward_length": FF,
        "expert_shared_feed_forward_length": FF,
        "nextn_predict_layers": nextn,
        "hyper_connection.count": HC,
        "hyper_connection.low_rank": LOWRANK,
        "attention.indexer.head_count": IDX_HEADS,
        "attention.indexer.key_length": IDX_DIM,
        "attention.indexer.top_k": 16,
        "full_attention_interval": 4,
    }.items():
        w.add_uint32(f"qwen4exp.{key}", val)
    if shared:
        w.add_key_value("qwen4exp.nextn_shared_target_tensors", True, gguf.GGUFValueType.BOOL)
    for tname, val in ts.items():
        if isinstance(val, tuple):
            w.add_tensor(tname, val[0], raw_dtype=val[1])
        else:
            w.add_tensor(tname, val)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return ref


def draft_config(num_hidden_layers=1, **over):
    text = types.SimpleNamespace(
        model_type="qwen4_exp_text",
        num_hidden_layers=num_hidden_layers,
        hidden_size=H,
        hc_count=HC,
        dtype="bfloat16",
        **over,
    )
    cfg = types.SimpleNamespace(
        model_type="qwen4_exp",
        architectures=["Qwen4ExpForCausalLMMTP"],
        text_config=text,
    )
    cfg.get_text_config = lambda: text
    return cfg


@pytest.fixture(autouse=True)
def _fresh_caches():
    def clear():
        getattr(HL, "_GGUF_HEADER_FACTS", {}).clear()
        gguf_shards._RESOLVED_CACHE.clear()

    clear()
    yield
    clear()


@pytest.fixture(params=[True, False], ids=["shared", "self-contained"])
def tiny(request, tmp_path):
    shared = request.param
    path = str(tmp_path / "mtp-tiny.gguf")
    ref = write_draft(path, shared=shared)
    return path, ref, shared


def run_pipeline(path, config=None):
    adapter = Q4.Qwen4ExpGGUFAdapter(config or draft_config(), path)
    name_map = adapter.build_name_map()
    out = {}
    for name, w in adapter.transform_stream(gguf_quant_weights_iterator(path, name_map)):
        assert name not in out, f"duplicate {name}"
        out[name] = w
    return adapter, name_map, out


# ------------------------------------------------------------------------------
# name map
# ------------------------------------------------------------------------------


def test_draft_name_map_is_complete_and_leaves_the_vocabulary_alone(tiny):
    path, _, shared = tiny
    adapter = Q4.Qwen4ExpGGUFAdapter(draft_config(), path)
    assert adapter.is_draft
    m = adapter.build_name_map()
    names = {t.name for t in gguf.GGUFReader(path).tensors}
    assert len(names) == (32 if shared else 34)  # the real files' counts
    assert set(m) == names - set(Q4.DRAFT_VOCAB_ROLES)  # vocabulary: never mapped
    assert adapter.draft_block_index() == BLOCK
    assert adapter.draft_shares_target_vocab() is shared
    p = f"blk.{BLOCK}."
    assert m[p + "nextn.eh_proj.weight"] == "mtp.eh_proj.weight"  # internal, split in the stream
    assert m[p + "nextn.enorm.weight"] == "mtp.pre_fc_norm_embedding.weight"
    assert m[p + "nextn.hnorm.weight"] == "mtp.pre_fc_norm_hidden.weight"
    assert m[p + "nextn.hc_head_norm.weight"] == "mtp.hyper_connection_mixer.hc_norm.weight"
    assert m[p + "nextn.hc_head_down.weight"] == (
        "mtp.hyper_connection_mixer.input_mix_weight_down.weight"
    )
    assert m[p + "nextn.hc_head_up.weight"] == (
        "mtp.hyper_connection_mixer.input_mix_weight_up.weight"
    )
    assert m[p + "attn_q.weight"] == "mtp.layers.0.self_attn.q_proj.weight"
    assert m[p + "hc_attn_inject.weight"] == (
        "mtp.layers.0.attn_hyper_connection.block_inject_weight.weight"
    )
    assert m[p + "indexer.q_proj.weight"] == "mtp.layers.0.self_attn.indexer.index_qk_proj.__q.weight"
    assert m[p + "ffn_gate_inp_shexp.weight"] == "mtp.layers.0.mlp.shared_expert_gate.weight"
    assert m[p + "ffn_up_exps.weight"] == "mtp.layers.0.mlp.experts.up_proj.weight"


def test_adapter_dispatch_main_and_draft_do_not_cross():
    cfg_main = draft_config()
    cfg_main.architectures = ["Qwen4ExpForConditionalGeneration"]
    assert Q4.Qwen4ExpGGUFAdapter(cfg_main, "x.gguf").is_draft is False
    assert Q4.Qwen4ExpGGUFAdapter(draft_config(), "x.gguf").is_draft is True


def test_the_draft_block_comes_from_the_file_not_from_num_hidden_layers(tiny):
    path, _, _ = tiny
    outs = []
    for n_layers in (1, 48, 4):  # the draft ModelConfig says 1; a raw backbone config 48
        _, _, out = run_pipeline(path, draft_config(num_hidden_layers=n_layers))
        outs.append(out)
    assert set(outs[0]) == set(outs[1]) == set(outs[2])
    for k in outs[0]:
        assert torch.equal(outs[0][k], outs[1][k]) and torch.equal(outs[0][k], outs[2][k])


# ------------------------------------------------------------------------------
# stream: values against the converter's forward math
# ------------------------------------------------------------------------------


def test_eh_proj_is_split_columns_embedding_first_then_hidden(tiny):
    path, ref, _ = tiny
    _, _, out = run_pipeline(path)
    fe, fh = ref.t["fc_embedding"], ref.t["fc_hidden"]
    got_e, got_h = out["mtp.fc_embedding.weight"], out["mtp.fc_hidden.weight"]
    assert got_e.shape == got_h.shape == (H, H) and got_e.dtype == torch.bfloat16
    # Q8_0 round trip of the ORIGINAL halves (blocks run along the 2H columns, the
    # split at H is block aligned, so dequant(split) == dequant(quant(half)))
    full = deq(q8(torch.cat([fe, fh], dim=1)))
    torch.testing.assert_close(got_e.float(), full[:, :H], atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(got_h.float(), full[:, H:], atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(got_e.float(), fe, atol=0.05, rtol=0.05)
    torch.testing.assert_close(got_h.float(), fh, atol=0.05, rtol=0.05)
    # guard against a vacuous test: swapped halves must NOT pass
    assert (got_e.float() - fh).abs().max() > 0.5
    assert (got_h.float() - fe).abs().max() > 0.5
    # the internal name never leaves the adapter
    assert not any("eh_proj" in n for n in out)


def test_gemma_norms_of_the_draft_get_their_one_back(tiny):
    path, ref, _ = tiny
    _, _, out = run_pipeline(path)
    cases = {
        "mtp.pre_fc_norm_embedding.weight": "pre_fc_norm_embedding",
        "mtp.pre_fc_norm_hidden.weight": "pre_fc_norm_hidden",
        "mtp.hyper_connection_mixer.hc_norm.weight": "mixer.hc_norm",
        "mtp.layers.0.attn_hyper_connection.hc_norm.weight": "layer.attn.norm",
        "mtp.layers.0.mlp_hyper_connection.hc_norm.weight": "layer.ffn.norm",
        "mtp.layers.0.self_attn.q_norm.weight": "qn",
        "mtp.layers.0.self_attn.k_norm.weight": "kn",
        "mtp.layers.0.self_attn.indexer.q_layernorm.weight": "iqn",
        "mtp.layers.0.self_attn.indexer.k_layernorm.weight": "ikn",
    }
    for name, key in cases.items():
        torch.testing.assert_close(out[name].float(), ref.t[key], atol=1e-6, rtol=1e-6)


def test_hyper_connection_and_router_tensors_load_as_dense_module_params(tiny):
    path, ref, _ = tiny
    _, _, out = run_pipeline(path)
    for hf, key, shape in (
        ("mtp.hyper_connection_mixer.input_mix_weight_down.weight", "mixer.down", (LOWRANK, HCH)),
        ("mtp.hyper_connection_mixer.input_mix_weight_up.weight", "mixer.up", (HCH, LOWRANK)),
        ("mtp.layers.0.attn_hyper_connection.block_inject_weight.weight", "layer.attn.inject", (HC, HCH)),
        ("mtp.layers.0.mlp_hyper_connection.input_mix_weight_up.weight", "layer.ffn.up", (HCH, LOWRANK)),
    ):
        assert out[hf].shape == shape and out[hf].dtype == torch.bfloat16
        torch.testing.assert_close(out[hf].float(), ref.t[key], atol=0.05, rtol=0.05)
        assert hf.replace(".weight", ".qweight") not in out  # dense, not packed
    torch.testing.assert_close(out["mtp.layers.0.mlp.gate.weight"].float(), ref.t["router"], atol=1e-2, rtol=1e-2)
    assert out["mtp.layers.0.mlp.shared_expert_gate.weight"].shape == (1, H)


def test_indexer_halves_fuse_q_rows_then_k_rows(tiny):
    path, ref, _ = tiny
    _, _, out = run_pipeline(path)
    base = "mtp.layers.0.self_attn.indexer.index_qk_proj"
    assert not any(".__q" in n or ".__k" in n for n in out)
    assert int(out[base + ".qweight_type"].item()) == int(Q.BF16)
    assert np.array_equal(out[base + ".qweight"].numpy(), bf16_bytes(ref.t["idx"]))


def test_experts_of_the_draft_block_go_to_mtp_layers_0(tiny):
    path, _, _ = tiny
    _, _, out = run_pipeline(path)
    assert not any(n.startswith("model.layers.") for n in out)
    for e in range(N_EXP):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            base = f"mtp.layers.0.mlp.experts.{e}.{proj}"
            assert int(out[base + ".qweight_type"].item()) == int(Q.Q8_0)
            assert out[base + ".qweight"].dtype == torch.uint8


def test_vocabulary_tensors_are_not_streamed(tiny):
    path, _, _ = tiny
    _, _, out = run_pipeline(path)
    assert not any("embed_tokens" in n or "lm_head" in n or "token_embd" in n for n in out)


def test_stream_emits_exactly_the_draft_param_set(tiny):
    path, _, _ = tiny
    _, _, out = run_pipeline(path)
    dense = {n for n in out if ".experts." not in n}
    want = {
        "mtp.fc_embedding.weight", "mtp.fc_hidden.weight",
        "mtp.pre_fc_norm_embedding.weight", "mtp.pre_fc_norm_hidden.weight",
        "mtp.hyper_connection_mixer.hc_norm.weight",
        "mtp.hyper_connection_mixer.input_mix_weight_down.weight",
        "mtp.hyper_connection_mixer.input_mix_weight_up.weight",
    }
    for hc, mod in (("attn", "attn_hyper_connection"), ("ffn", "mlp_hyper_connection")):
        for leaf in ("hc_norm", "input_mix_weight_down", "input_mix_weight_up", "block_inject_weight"):
            want.add(f"mtp.layers.0.{mod}.{leaf}.weight")
    want |= {
        "mtp.layers.0.self_attn.q_norm.weight", "mtp.layers.0.self_attn.k_norm.weight",
        "mtp.layers.0.self_attn.indexer.q_layernorm.weight",
        "mtp.layers.0.self_attn.indexer.k_layernorm.weight",
        "mtp.layers.0.mlp.gate.weight", "mtp.layers.0.mlp.shared_expert_gate.weight",
    }
    for mod in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                "self_attn.indexer.index_qk_proj", "mlp.shared_expert.gate_proj",
                "mlp.shared_expert.up_proj", "mlp.shared_expert.down_proj"):
        want |= {f"mtp.layers.0.{mod}.qweight", f"mtp.layers.0.{mod}.qweight_type"}
    assert dense == want


def test_unquantized_module_prefixes_work_for_the_draft(tiny):
    path, _, _ = tiny
    prefixes = Q4.Qwen4ExpGGUFAdapter(draft_config(), path).unquantized_module_prefixes()
    assert not any(p.startswith("model.") for p in prefixes)
    assert not any("indexer.index_qk_proj." in p or p.endswith((".__q", ".__k")) for p in prefixes)


# ------------------------------------------------------------------------------
# refusals
# ------------------------------------------------------------------------------


def build(tmp_path, name, **kw):
    path = str(tmp_path / name)
    write_draft(path, **kw)
    return path


def test_a_backbone_file_is_not_a_draft(tmp_path):
    ref = Ref()
    ts = draft_dir_tensors(ref, with_vocab=False, block=BLOCK)
    ts = {k: v for k, v in ts.items() if ".nextn." not in k}
    path = build(tmp_path, "nonextn.gguf", tensors=ts, nextn=0)
    with pytest.raises(RuntimeError, match=r"no blk\.<N>\.nextn\.eh_proj\.weight"):
        Q4.Qwen4ExpGGUFAdapter(draft_config(), path).build_name_map()


def test_missing_role_is_a_named_refusal(tmp_path):
    p = build(tmp_path, "gap.gguf", shared=True, drop=[f"blk.{BLOCK}.hc_ffn_up.weight"])
    with pytest.raises(RuntimeError, match=r"incomplete or inconsistent.*hc_ffn_up"):
        Q4.Qwen4ExpGGUFAdapter(draft_config(), p).build_name_map()
    p = build(tmp_path, "gap2.gguf", shared=True, drop=[f"blk.{BLOCK}.nextn.hnorm.weight"])
    with pytest.raises(RuntimeError, match=r"hnorm"):
        Q4.Qwen4ExpGGUFAdapter(draft_config(), p).build_name_map()


def test_unknown_tensor_is_a_named_refusal(tmp_path):
    p = build(tmp_path, "unk.gguf", shared=True,
              extra=[(f"blk.{BLOCK}.mystery.weight", np.zeros((4, 4), np.float32))])
    with pytest.raises(RuntimeError, match=r"1 tensors not mapped.*mystery"):
        Q4.Qwen4ExpGGUFAdapter(draft_config(), p).build_name_map()


def test_gdn_tensor_in_the_draft_block_is_refused(tmp_path):
    p = build(tmp_path, "gdn.gguf", shared=True,
              extra=[(f"blk.{BLOCK}.ssm_a", np.zeros((4,), np.float32))])
    with pytest.raises(RuntimeError, match=r"1 tensors not mapped.*ssm_a"):
        Q4.Qwen4ExpGGUFAdapter(draft_config(), p).build_name_map()


def test_two_nextn_layers_are_refused(tmp_path):
    p = build(tmp_path, "two.gguf", shared=True, nextn=2, block_count=BLOCK + 2)
    with pytest.raises(RuntimeError, match=r"exactly one MTP layer"):
        Q4.Qwen4ExpGGUFAdapter(draft_config(), p).build_name_map()


def test_block_count_that_contradicts_the_tensors_is_refused(tmp_path):
    p = build(tmp_path, "bc.gguf", shared=True, block_count=BLOCK + 5)
    with pytest.raises(RuntimeError, match=r"NEXTN tensors are at blk\.48"):
        Q4.Qwen4ExpGGUFAdapter(draft_config(), p).build_name_map()


def test_shared_flag_with_vocabulary_tensors_is_refused(tmp_path):
    ref = Ref()
    ts = draft_dir_tensors(ref, with_vocab=True)
    p = str(tmp_path / "lie.gguf")
    write_draft(p, shared=True, tensors=ts, ref=ref)
    with pytest.raises(RuntimeError, match=r"nextn_shared_target_tensors=True.*token_embd"):
        Q4.Qwen4ExpGGUFAdapter(draft_config(), p).build_name_map()


def test_wrong_eh_proj_shape_is_refused(tmp_path):
    ref = Ref()
    ts = draft_dir_tensors(ref, with_vocab=False)
    ts[f"blk.{BLOCK}.nextn.eh_proj.weight"] = (q8(ref.put("bad", (H, H))), Q.Q8_0)  # [H, H], not [H, 2H]
    p = str(tmp_path / "shape.gguf")
    write_draft(p, shared=True, tensors=ts, ref=ref)
    with pytest.raises(RuntimeError, match=r"eh_proj.*ggml shape"):
        Q4.Qwen4ExpGGUFAdapter(draft_config(), p).build_name_map()


def test_hc_mixer_int8_env_is_refused_for_the_draft_too(tiny, monkeypatch):
    path, _, _ = tiny
    monkeypatch.setenv("SGLANG_HC_MIXER_INT8", "1")
    with pytest.raises(RuntimeError, match="SGLANG_HC_MIXER_INT8"):
        Q4.Qwen4ExpGGUFAdapter(draft_config(), path).build_name_map()


# ------------------------------------------------------------------------------
# combined export (backbone blocks + trailing NEXTN block): pure helpers
# ------------------------------------------------------------------------------


def test_combined_export_maps_only_the_draft_block():
    tensors = (
        ["token_embd.weight", "output.weight", "output_hc_norm.weight", "per_layer_token_embd.weight"]
        + [f"blk.{i}.attn_qkv.weight" for i in range(3)]
        + [f"blk.{BLOCK}.nextn.eh_proj.weight", f"blk.{BLOCK}.attn_q.weight", f"blk.{BLOCK}.hc_attn_norm.weight"]
    )
    m, unknown = Q4.build_qwen4exp_draft_name_map(tensors, BLOCK)
    assert unknown == []
    assert set(m) == {
        f"blk.{BLOCK}.nextn.eh_proj.weight",
        f"blk.{BLOCK}.attn_q.weight",
        f"blk.{BLOCK}.hc_attn_norm.weight",
    }
    assert Q4.qwen4exp_draft_blocks(tensors) == [BLOCK]


def test_experts_of_other_blocks_are_dropped_by_the_draft_stream(tmp_path):
    path = str(tmp_path / "x.gguf")
    write_draft(path, shared=True)
    ad = Q4.Qwen4ExpGGUFAdapter(draft_config(), path)
    fake = [
        ("model.layers.0.mlp.experts.3.up_proj.qweight", torch.zeros(2, 2, dtype=torch.uint8)),
        (f"model.layers.{BLOCK}.mlp.experts.3.up_proj.qweight", torch.ones(2, 2, dtype=torch.uint8)),
        (f"model.layers.{BLOCK}.mlp.experts.3.up_proj.qweight_type", torch.tensor(8)),
    ]
    got = list(ad._draft_pre(iter(fake)))
    assert [n for n, _ in got] == [
        "mtp.layers.0.mlp.experts.3.up_proj.qweight",
        "mtp.layers.0.mlp.experts.3.up_proj.qweight_type",
    ]


# ------------------------------------------------------------------------------
# draft load format (configs/load_config.py)
# ------------------------------------------------------------------------------


def test_draft_load_format_gguf_draft_beside_a_non_gguf_target(tmp_path):
    resolve = load_config_mod.resolve_draft_load_format
    gguf_file = str(tmp_path / "mtp.gguf")
    write_draft(gguf_file, shared=True)
    safetensors_dir = tmp_path / "albucino"
    safetensors_dir.mkdir()
    ns = lambda fmt, flag=None: types.SimpleNamespace(  # noqa: E731
        load_format=fmt, speculative_draft_load_format=flag
    )
    assert resolve(ns("auto"), gguf_file) == "gguf"  # NEW: the GGUF head beside an INT4/NVFP4 target
    assert resolve(ns(load_config_mod.LoadFormat.AUTO), gguf_file) == "gguf"
    assert resolve(ns("gguf"), gguf_file) == "gguf"  # unchanged: GGUF target, GGUF head
    # unchanged: the albucino safetensors draft beside a GGUF target loads auto, beside
    # a non-GGUF target it is the inherited object itself
    assert resolve(ns("gguf"), str(safetensors_dir)) == "auto"
    auto = "auto"
    assert resolve(ns(auto), str(safetensors_dir)) is auto
    assert resolve(ns(auto), None) is auto
    assert resolve(ns("safetensors"), gguf_file) == "safetensors"  # an explicit target format is not rewritten
    assert resolve(ns("auto", "dummy"), gguf_file) == "dummy"  # the flag still wins


# ------------------------------------------------------------------------------
# the real files
# ------------------------------------------------------------------------------

real_files = pytest.mark.skipif(
    not (os.path.isfile(REAL_SHARED) and os.path.isfile(REAL_FULL)),
    reason="unsloth qwen4exp MTP export not on this machine",
)
real_sibling = pytest.mark.skipif(
    not os.path.isfile(os.path.join(REAL_NVIDIA, "config.json")),
    reason="nvidia Qwen3.8-Flash-Next config not on this machine",
)


def _real_text_config():
    with open(os.path.join(REAL_NVIDIA, "config.json")) as f:
        return dict(json.load(f)["text_config"])


def _real_adapter(path):
    text = _real_text_config()
    text["num_hidden_layers"] = 1  # what the draft ModelConfig rewrite leaves
    text.setdefault("model_type", "qwen4_exp_text")
    tc = types.SimpleNamespace(**text)
    cfg = types.SimpleNamespace(
        model_type="qwen4_exp", architectures=["Qwen4ExpForCausalLMMTP"], text_config=tc
    )
    cfg.get_text_config = lambda: tc
    return Q4.Qwen4ExpGGUFAdapter(cfg, path)


@real_files
@real_sibling
@pytest.mark.parametrize(
    "path,n_tensors,shared",
    [(REAL_FULL, 34, False), (REAL_SHARED, 32, True)],
    ids=["mtp-Q8_0", "mtp-shared-Q8_0"],
)
def test_real_mtp_header_maps_completely(path, n_tensors, shared):
    ad = _real_adapter(path)
    names = {str(t.name) for t in gguf.GGUFReader(path).tensors}
    assert len(names) == n_tensors
    m = ad.build_name_map()
    assert ad.draft_block_index() == 48
    assert ad.draft_shares_target_vocab() is shared
    assert set(m) == names - set(Q4.DRAFT_VOCAB_ROLES)
    assert all(n.startswith("blk.48.") for n in m)
    assert len(m) == 32  # six nextn + 26 layer roles, in both variants
    # the header the launcher (W173) and the loader (reconcile_sibling_config) read: consistent
    facts = HL.gguf_header_facts(path)
    assert (facts.arch, facts.block_count, facts.nextn_predict_layers) == ("qwen4exp", 49, 1)
    assert facts.backbone_depth == 48
    problems = Q4.qwen4exp_meta_mismatches(
        {k: v for k, v in _real_text_config().items()},
        Q4.read_qwen4exp_kv(path),
        n_blocks_backbone=facts.backbone_depth,
    )
    assert problems == []


@real_files
@real_sibling
def test_real_shared_head_values_against_the_hf_checkpoint():
    """The full pipeline on the real shared-Q8_0 head against the ``mtp.*`` tensors
    of the nvidia safetensors (same original model): eh_proj split + order, every
    Gemma norm (exact), the fused indexer (exact), hyper-connection mixers and
    projections (Q8_0 error only)."""
    from safetensors import safe_open

    index_path = os.path.join(REAL_NVIDIA, "model.safetensors.index.json")
    if not os.path.isfile(index_path):
        pytest.skip("nvidia safetensors index not on this machine")
    weight_map = json.load(open(index_path))["weight_map"]
    ad = _real_adapter(REAL_SHARED)
    m = ad.build_name_map()
    out = {}
    n_expert_names = 0
    for name, w in ad.transform_stream(gguf_quant_weights_iterator(REAL_SHARED, m)):
        if ".experts." in name:
            n_expert_names += 1
            continue
        out[name] = w
    assert n_expert_names == 512 * 3 * 2  # qweight + qweight_type per expert and projection
    handles = {}

    def hf(name):
        f = weight_map[name]
        if f not in handles:
            handles[f] = safe_open(os.path.join(REAL_NVIDIA, f), "pt")
        return handles[f].get_tensor(name).float()

    types_ = {n[: -len(".qweight_type")]: int(out[n]) for n in out if n.endswith(".qweight_type")}
    checked = 0
    for name in sorted(out):
        if name.endswith(".qweight_type"):
            continue
        base = name.rsplit(".", 1)[0]
        hf_name = base + ".weight"
        assert hf_name in weight_map, f"{name}: no such tensor in the HF checkpoint"
        want = hf(hf_name)
        if name.endswith(".qweight"):
            t = Q(types_[base])
            got = (
                out[name].contiguous().view(torch.bfloat16).float()
                if t == Q.BF16
                else deq(out[name].numpy(), t)
            )
        else:
            got = out[name].float()
        assert got.shape == want.shape, f"{name}: {tuple(got.shape)} vs HF {tuple(want.shape)}"
        err = (got - want).abs().max().item()
        scale = max(want.abs().max().item(), 1e-9)
        if "norm" in name or "layernorm" in name or name.endswith(("indexer.index_qk_proj.qweight", "mlp.gate.weight")):
            assert err <= 1e-6 * max(scale, 1.0), f"{name}: exact tensor differs, max err {err}"
        else:
            assert err / scale < 0.006, f"{name}: Q8_0-level error expected, got {err / scale}"
        checked += 1
    # fc_embedding + fc_hidden + the 27 other non-expert tensors the HF head has
    assert checked == 29, checked
    for name in ("mtp.fc_embedding.weight", "mtp.fc_hidden.weight"):
        assert name in out
    # order matters: the swapped halves are far off
    swapped = (out["mtp.fc_embedding.weight"].float() - hf("mtp.fc_hidden.weight")).abs().max().item()
    assert swapped > 10 * (out["mtp.fc_embedding.weight"].float() - hf("mtp.fc_embedding.weight")).abs().max().item()
