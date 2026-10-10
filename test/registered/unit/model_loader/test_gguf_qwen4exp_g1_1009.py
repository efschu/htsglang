# SPDX-License-Identifier: Apache-2.0
"""NF-GGUF AP G1 (2026-10-09): the ``qwen4exp`` adapter (Qwen3.8-Flash-Next).

* a tiny ``GGUFWriter(path, "qwen4exp")`` carrying ALL 47 tensor roles with the
  llama.cpp FORWARD converter transforms applied (reference below is the
  converter's math, written out here), run through the real pipeline
  (``Qwen4ExpGGUFAdapter.build_name_map`` -> ``gguf_quant_weights_iterator`` ->
  ``transform_stream``) and compared with the original "HF" tensors;
* the metadata consistency check (loader ``reconcile_sibling_config`` + launcher
  W172 / W173);
* the real unsloth header (3 parts, 1224 tensors) when present: every tensor
  mapped, 0 unknown, shapes against the safetensors header stubs when present.

Header-only on real files; no tensor data of the 87 GiB export is read.
Converter reference: llama.cpp master 79e2e74eb110 ``conversion/qwen4exp.py``
and ``conversion/qwen.py`` (see the module doc of ``gguf_qwen4exp.py``).
"""

from __future__ import annotations

import json
import os
import struct
import types

# G8 (cross-check finding C): BEFORE the first ``import torch`` -- set after it, the line was a no-op (the wrapper
# pytest_gedeckelt.sh sets it empty anyway, a bare pytest run did not)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import pytest
import torch

import gguf  # noqa: E402
from gguf.quants import dequantize, quantize  # noqa: E402

from sglang.srt.model_loader import gguf_qwen4exp as Q4  # noqa: E402
from sglang.srt.model_loader import gguf_registry, gguf_shards  # noqa: E402
from sglang.srt.model_loader.weight_utils import gguf_quant_weights_iterator  # noqa: E402
from sglang.srt.weg2 import host_ledger as HL  # noqa: E402
from sglang.srt.weg2 import launcher as L  # noqa: E402

Q = gguf.GGMLQuantizationType

# ---- tiny geometry (block-32 aligned so Q8_0 holds every row) -----------------
H = 64  # hidden
HC = 2  # hyper-connection streams
HCH = HC * H  # 128
LOWRANK = 32
VOCAB = 96  # > hidden: the vocabulary check reads max(ne0, ne1) of token_embd, as the real file (248320 > 2560)
N_LAYERS = 4  # 0,1,2 GDN ; 3 full attention (interval 4); PLE on layer 1
NUM_K, NUM_V, DK, DV = 2, 6, 32, 32  # 3 v-heads per k-head: re-tiling active
QKV_ROWS = 2 * NUM_K * DK + NUM_V * DV  # 320
Z_ROWS = NUM_V * DV  # 192
HEADS, KV_HEADS, HEAD_DIM = 4, 2, 32
Q_ROWS = HEADS * HEAD_DIM * 2  # [q|gate] per head
IDX_HEADS, IDX_DIM = 2, 32
N_EXP, TOPK, FF = 4, 2, 32
PLE_LAYER = 1
PLE_ROWS, PLE_COLS = 96, 32
MULTS = [2**62 + 12345, 20109073645365, 8052911324071]  # > 2^53: exact or nothing
OFFS = [0, 20, 43]
SIZES = [20, 23, 33]  # sum 76 <= PLE_ROWS

REAL_DIR = (
    "/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-Flash-Next-GGUF-unsloth/UD-IQ4_XS"
)
REAL_PART1 = os.path.join(REAL_DIR, "Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf")
REAL_SIBLING_CFG = (
    "/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-Flash-Next-NVFP4-nvidia/config.json"
)
STUB_DIR = "/tmp/s3/hdr/Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist"


# ---- the converter's math, written out (reference) ----------------------------
def reorder_v_heads(tensor, dim, num_k_heads, num_v_per_k, head_dim):
    """llama.cpp conversion/qwen.py L468-L492 (_reorder_v_heads), verbatim math."""
    shape = list(tensor.shape)
    if dim < 0:
        dim += len(shape)
    new_shape = shape[:dim] + [num_k_heads, num_v_per_k, head_dim] + shape[dim + 1 :]
    tensor = tensor.reshape(*new_shape)
    perm = list(range(len(new_shape)))
    perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
    return tensor.permute(*perm).contiguous().reshape(*shape)


def tile(t, dim, head_units):
    return reorder_v_heads(t, dim, NUM_K, NUM_V // NUM_K, head_units)


def q8(t: torch.Tensor) -> np.ndarray:
    return quantize(t.float().numpy(), Q.Q8_0)


def bf16_bytes(t: torch.Tensor) -> np.ndarray:
    return quantize(t.float().numpy(), Q.BF16)


def deq(raw: np.ndarray, qtype=Q.Q8_0) -> torch.Tensor:
    return torch.from_numpy(dequantize(raw, qtype).copy())


def _rand(*shape, seed=0):
    g = torch.Generator().manual_seed(1000 + hash(shape) % 997 + seed)
    return torch.randn(*shape, generator=g)


class Ref:
    """The original ('HF') tensors, by role, and the GGUF the converter makes."""

    def __init__(self):
        self.t = {}  # key -> torch tensor (the HF-side truth)

    def put(self, key, shape, seed=0):
        self.t[key] = _rand(*shape, seed=seed + len(self.t))
        return self.t[key]


def build_tiny(path: str, *, drop=(), extra=()):
    """Write a tiny qwen4exp GGUF with all 47 roles; return the reference."""
    ref = Ref()
    tensors = {}  # gguf name -> ndarray or (ndarray, qtype)

    def add(name, arr, qtype=None):
        tensors[name] = (arr, qtype) if qtype is not None else arr

    def norm_pair(key, n):  # HF zero-centred gamma, the converter bakes +1
        t = ref.put(key, (n,))
        return t + 1.0

    add("token_embd.weight", q8(ref.put("embed", (VOCAB, H))), Q.Q8_0)
    add("output.weight", q8(ref.put("lm_head", (VOCAB, H))), Q.Q8_0)
    add("output_hc_norm.weight", norm_pair("out_hc_norm", HCH).numpy())
    add("output_hc_down.weight", q8(ref.put("out_hc_down", (LOWRANK, HCH))), Q.Q8_0)
    add("output_hc_up.weight", q8(ref.put("out_hc_up", (HCH, LOWRANK))), Q.Q8_0)
    # per_layer_token_embd: one tensor (IQ4_NL, zero payload: only its identity matters)
    blk, ts = gguf.GGML_QUANT_SIZES[Q.IQ4_NL]
    add(
        "per_layer_token_embd.weight",
        np.zeros((PLE_ROWS, PLE_COLS // blk * ts), dtype=np.uint8),
        Q.IQ4_NL,
    )

    for i in range(N_LAYERS):
        p = f"blk.{i}."
        gdn = (i + 1) % 4 != 0
        for hc in ("attn", "ffn"):
            k = f"l{i}.{hc}"
            add(p + f"hc_{hc}_norm.weight", norm_pair(k + ".norm", HCH).numpy())
            add(p + f"hc_{hc}_down.weight", q8(ref.put(k + ".down", (LOWRANK, HCH))), Q.Q8_0)
            add(p + f"hc_{hc}_up.weight", q8(ref.put(k + ".up", (HCH, LOWRANK))), Q.Q8_0)
            add(p + f"hc_{hc}_inject.weight", ref.put(k + ".inject", (HC, HCH)).numpy())
        if gdn:
            qkv = ref.put(f"l{i}.qkv", (QKV_ROWS, H))
            qk = 2 * NUM_K * DK
            conv = ref.put(f"l{i}.conv", (QKV_ROWS, 1, 4))
            z = ref.put(f"l{i}.z", (Z_ROWS, H))
            a = ref.put(f"l{i}.a", (NUM_V, H))
            b = ref.put(f"l{i}.b", (NUM_V, H))
            a_log = ref.put(f"l{i}.A_log", (NUM_V,))
            dt = ref.put(f"l{i}.dt", (NUM_V,))
            out = ref.put(f"l{i}.out", (H, Z_ROWS))
            norm = ref.put(f"l{i}.ssm_norm", (DV,))
            add(
                p + "attn_qkv.weight",
                q8(torch.cat([qkv[:qk], tile(qkv[qk:], 0, DV)], 0)),
                Q.Q8_0,
            )
            add(p + "attn_gate.weight", q8(tile(z, 0, DV)), Q.Q8_0)
            add(p + "ssm_alpha.weight", tile(a, 0, 1).numpy())
            add(p + "ssm_beta.weight", tile(b, 0, 1).numpy())
            add(p + "ssm_a", (-torch.exp(tile(a_log.unsqueeze(-1), 0, 1).squeeze(-1))).numpy())
            add(p + "ssm_dt.bias", tile(dt.unsqueeze(-1), 0, 1).squeeze(-1).numpy())
            c = conv.squeeze()
            add(p + "ssm_conv1d.weight", torch.cat([c[:qk], tile(c[qk:], 0, DV)], 0).numpy())
            add(p + "ssm_norm.weight", norm.numpy())  # NOT +1 (linear_attn.norm)
            add(p + "ssm_out.weight", q8(tile(out, 1, DV)), Q.Q8_0)
        else:
            add(p + "attn_q.weight", q8(ref.put(f"l{i}.q", (Q_ROWS, H))), Q.Q8_0)
            add(p + "attn_k.weight", q8(ref.put(f"l{i}.k", (KV_HEADS * HEAD_DIM, H))), Q.Q8_0)
            add(p + "attn_v.weight", q8(ref.put(f"l{i}.v", (KV_HEADS * HEAD_DIM, H))), Q.Q8_0)
            add(p + "attn_output.weight", q8(ref.put(f"l{i}.o", (H, HEADS * HEAD_DIM))), Q.Q8_0)
            add(p + "attn_q_norm.weight", norm_pair(f"l{i}.qn", HEAD_DIM).numpy())
            add(p + "attn_k_norm.weight", norm_pair(f"l{i}.kn", HEAD_DIM).numpy())
            n_q = IDX_HEADS * IDX_DIM
            qk = ref.put(f"l{i}.idx", (n_q + IDX_DIM, H))
            add(p + "indexer.q_proj.weight", bf16_bytes(qk[:n_q]), Q.BF16)
            add(p + "indexer.k_proj.weight", bf16_bytes(qk[n_q:]), Q.BF16)
            add(p + "indexer.q_norm.weight", norm_pair(f"l{i}.iqn", IDX_DIM).numpy())
            add(p + "indexer.k_norm.weight", norm_pair(f"l{i}.ikn", IDX_DIM).numpy())
        if i == PLE_LAYER:
            add(p + "ple_key.weight", q8(ref.put("ple.key", (HCH, H))), Q.Q8_0)
            add(p + "ple_value.weight", q8(ref.put("ple.value", (H, H))), Q.Q8_0)
            add(p + "ple_norm_conv.weight", norm_pair("ple.nc", HCH).numpy())
            add(p + "ple_norm_key.weight", norm_pair("ple.nk", HCH).numpy())
            add(p + "ple_norm_query.weight", norm_pair("ple.nq", HCH).numpy())
            pc = ref.put("ple.conv", (HCH, 1, 4))
            add(p + "ple_conv1d.weight", pc.squeeze().numpy())
        add(p + "ffn_gate_inp.weight", ref.put(f"l{i}.router", (N_EXP, H)).numpy())
        add(p + "ffn_gate_inp_shexp.weight", ref.put(f"l{i}.sgate", (1, H)).squeeze(0).numpy())
        for nm, shp in (("gate", (FF, H)), ("up", (FF, H)), ("down", (H, FF))):
            add(p + f"ffn_{nm}_shexp.weight", q8(ref.put(f"l{i}.sh.{nm}", shp)), Q.Q8_0)
            e = ref.put(f"l{i}.ex.{nm}", (N_EXP,) + shp)
            add(p + f"ffn_{nm}_exps.weight", q8(e), Q.Q8_0)

    for name in drop:
        tensors.pop(name)
    for name, arr in extra:
        tensors[name] = arr

    w = gguf.GGUFWriter(path, "qwen4exp")
    w.add_block_count(N_LAYERS)
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
        "ssm.conv_kernel": 4,
        "ssm.state_size": DK,
        "ssm.group_count": NUM_K,
        "ssm.time_step_rank": NUM_V,
        "ssm.inner_size": NUM_V * DV,
        "full_attention_interval": 4,
        "hyper_connection.count": HC,
        "hyper_connection.low_rank": LOWRANK,
        "attention.indexer.head_count": IDX_HEADS,
        "attention.indexer.key_length": IDX_DIM,
        "attention.indexer.top_k": 16,
        "ple.ngram_size": 3,
        "ple.heads_per_ngram": 8,
        "ple.conv_kernel": 4,
        "embedding_length_per_layer_input": PLE_COLS,
    }.items():
        w.add_uint32(f"qwen4exp.{key}", val)
    U32, U64, ARR = gguf.GGUFValueType.UINT32, gguf.GGUFValueType.UINT64, gguf.GGUFValueType.ARRAY
    w.add_key_value("qwen4exp.attention.compress_ratios", [0, 0, 0, 4], ARR, sub_type=U32)
    w.add_key_value("qwen4exp.ple.layers", [PLE_LAYER], ARR, sub_type=U32)
    w.add_key_value("qwen4exp.ple.layer_multipliers", MULTS, ARR, sub_type=U64)
    w.add_key_value("qwen4exp.ple.head_offsets", OFFS, ARR, sub_type=U64)
    w.add_key_value("qwen4exp.ple.head_vocab_sizes", SIZES, ARR, sub_type=U64)
    for tname, val in tensors.items():
        if isinstance(val, tuple):
            w.add_tensor(tname, val[0], raw_dtype=val[1])
        else:
            w.add_tensor(tname, val)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return ref


def text_config_dict(**over):
    cfg = {
        "model_type": "qwen4_exp_text",
        "num_hidden_layers": N_LAYERS,
        "hidden_size": H,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "vocab_size": VOCAB,
        "num_experts": N_EXP,
        "num_experts_per_tok": TOPK,
        "moe_intermediate_size": FF,
        "shared_expert_intermediate_size": FF,
        "hc_count": HC,
        "hc_lowrank": LOWRANK,
        "indexer_n_heads": IDX_HEADS,
        "indexer_head_dim": IDX_DIM,
        "indexer_budget": 16,
        "indexer_compress_ratio": 4,
        "ple_layer_ids": [PLE_LAYER + 1],
        "ngram_size": 3,
        "heads_per_ngram": 8,
        "ple_conv_kernel_size": 4,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": DK,
        "linear_num_key_heads": NUM_K,
        "linear_num_value_heads": NUM_V,
        "linear_value_head_dim": DV,
        "full_attention_interval": 4,
        "layer_types": [
            "full_attention" if (i + 1) % 4 == 0 else "linear_attention"
            for i in range(N_LAYERS)
        ],
        "split_ngram_parts": 8,
        "dtype": "bfloat16",
    }
    cfg.update(over)
    return cfg


def make_config(**over):
    text = types.SimpleNamespace(**text_config_dict(**over))
    cfg = types.SimpleNamespace(
        model_type="qwen4_exp",
        architectures=["Qwen4ExpForConditionalGeneration"],
        text_config=text,
        vision_config=object(),
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


@pytest.fixture
def tiny(tmp_path):
    path = str(tmp_path / "m-00001-of-00001.gguf")
    ref = build_tiny(path)
    return path, ref


def run_pipeline(path):
    adapter = Q4.Qwen4ExpGGUFAdapter(make_config(), path)
    name_map = adapter.build_name_map()
    out = {}
    for name, w in adapter.transform_stream(gguf_quant_weights_iterator(path, name_map)):
        assert name not in out, f"duplicate {name}"
        out[name] = w
    return adapter, name_map, out


# ------------------------------------------------------------------------------
# name table
# ------------------------------------------------------------------------------


def test_47_roles_and_the_tiny_file_carries_every_one(tiny):
    path, _ = tiny
    assert len(Q4.QWEN4EXP_ROLES) == len(set(Q4.QWEN4EXP_ROLES)) == 47
    names = {t.name for t in gguf.GGUFReader(path).tensors}
    roles = {n.split(".", 2)[2] if n.startswith("blk.") else n for n in names}
    assert roles == set(Q4.QWEN4EXP_ROLES)


def test_name_map_is_complete_with_the_loaders_spellings(tiny):
    path, _ = tiny
    adapter = Q4.Qwen4ExpGGUFAdapter(make_config(), path)
    m = adapter.build_name_map()
    names = {t.name for t in gguf.GGUFReader(path).tensors}
    assert set(m) == names
    assert m["token_embd.weight"] == "model.embed_tokens.weight"
    assert m["output.weight"] == "lm_head.weight"
    assert m["output_hc_down.weight"] == "model.hyper_connection_mixer.input_mix_weight_down.weight"
    assert m["blk.0.ssm_a"] == "model.layers.0.linear_attn.A_log"
    assert m["blk.0.ssm_dt.bias"] == "model.layers.0.linear_attn.dt_bias"
    assert m["blk.0.hc_attn_inject.weight"] == (
        "model.layers.0.attn_hyper_connection.block_inject_weight.weight"
    )
    assert m["blk.3.indexer.q_proj.weight"].endswith("indexer.index_qk_proj.__q.weight")
    assert m["blk.3.indexer.k_proj.weight"].endswith("indexer.index_qk_proj.__k.weight")
    assert m["blk.3.indexer.q_norm.weight"] == "model.layers.3.self_attn.indexer.q_layernorm.weight"
    assert m["blk.1.ple_conv1d.weight"] == "model.layers.1.ple.conv1d.weight"
    assert m["per_layer_token_embd.weight"] == (
        "model.layers.1.ple.ple_embedding.ngram_embedding.weight"
    )
    assert m["blk.2.ffn_gate_inp_shexp.weight"] == "model.layers.2.mlp.shared_expert_gate.weight"
    assert m["blk.2.ffn_up_exps.weight"] == "model.layers.2.mlp.experts.up_proj.weight"


def test_unknown_tensor_and_missing_role_are_named_refusals(tmp_path):
    p1 = str(tmp_path / "a.gguf")
    build_tiny(p1, extra=[("blk.0.mystery.weight", np.zeros((4, 4), np.float32))])
    with pytest.raises(RuntimeError, match=r"1 tensors not mapped.*mystery"):
        Q4.Qwen4ExpGGUFAdapter(make_config(), p1).build_name_map()
    p2 = str(tmp_path / "b.gguf")
    build_tiny(p2, drop=["blk.2.hc_ffn_inject.weight"])
    with pytest.raises(RuntimeError, match=r"incomplete tensor set.*blk\.2.*hc_ffn_inject"):
        Q4.Qwen4ExpGGUFAdapter(make_config(), p2).build_name_map()


def test_hc_mixer_int8_env_is_refused_by_name(tiny, monkeypatch):
    path, _ = tiny
    monkeypatch.setenv("SGLANG_HC_MIXER_INT8", "1")
    with pytest.raises(RuntimeError, match="SGLANG_HC_MIXER_INT8"):
        Q4.Qwen4ExpGGUFAdapter(make_config(), path).build_name_map()


def test_registry_dispatches_both_model_types_and_lists_the_arch():
    for mt in ("qwen4_exp", "qwen4_exp_text"):
        assert gguf_registry.get_gguf_adapter_class(mt) is Q4.Qwen4ExpGGUFAdapter
    assert "qwen4exp" in gguf_registry.sibling_config_gguf_archs()
    # the existing families still resolve to their own adapters
    assert gguf_registry.get_gguf_adapter_class("qwen3_5").__name__ == "Qwen35GGUFAdapter"


# ------------------------------------------------------------------------------
# inverse transforms against the converter's forward math
# ------------------------------------------------------------------------------


def test_untiling_inputs_really_differ_from_the_truth(tiny):
    """Guard against a vacuous round trip: the tiled file tensors are NOT the HF ones."""
    _, ref = tiny
    z = ref.t["l0.z"]
    assert not torch.equal(tile(z, 0, DV), z)
    a = ref.t["l0.a"]
    assert not torch.equal(tile(a, 0, 1), a)


def test_gdn_inverse_transforms(tiny):
    path, ref = tiny
    _, _, out = run_pipeline(path)
    for i in (0, 1, 2):
        p = f"model.layers.{i}.linear_attn."
        # A_log = log(-ssm_a), un-tiled
        torch.testing.assert_close(out[p + "A_log"].float(), ref.t[f"l{i}.A_log"], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(out[p + "dt_bias"].float(), ref.t[f"l{i}.dt"])
        # conv1d: [C, 4] -> [C, 1, 4], V channels un-tiled, q|k channels untouched
        conv = out[p + "conv1d.weight"]
        assert conv.shape == (QKV_ROWS, 1, 4)
        torch.testing.assert_close(conv.float(), ref.t[f"l{i}.conv"])
        # F32 a / b projections: rows un-tiled
        torch.testing.assert_close(out[p + "in_proj_a.weight"].float(), ref.t[f"l{i}.a"])
        torch.testing.assert_close(out[p + "in_proj_b.weight"].float(), ref.t[f"l{i}.b"])
        # Q8_0: dequant(un-tiled bytes) == dequant(quant(original)) (blocks are rows)
        for key, hf, shape in (
            ("qkv", "in_proj_qkv", (QKV_ROWS, H)),
            ("z", "in_proj_z", (Z_ROWS, H)),
        ):
            raw = out[p + hf + ".qweight"].numpy()
            torch.testing.assert_close(deq(raw), deq(q8(ref.t[f"l{i}.{key}"])))
        # out_proj: head_v_dim == one Q8_0 block -> byte-granular column un-tile
        raw = out[p + "out_proj.qweight"].numpy()
        torch.testing.assert_close(deq(raw), deq(q8(ref.t[f"l{i}.out"])))
        # ssm_norm (linear_attn.norm) is stored raw: no -1
        torch.testing.assert_close(out[p + "norm.weight"].float(), ref.t[f"l{i}.ssm_norm"])


def test_gemma_norms_get_their_one_back_everywhere(tiny):
    path, ref = tiny
    _, _, out = run_pipeline(path)
    cases = {
        "model.hyper_connection_mixer.hc_norm.weight": "out_hc_norm",
        "model.layers.0.attn_hyper_connection.hc_norm.weight": "l0.attn.norm",
        "model.layers.2.mlp_hyper_connection.hc_norm.weight": "l2.ffn.norm",
        "model.layers.3.self_attn.q_norm.weight": "l3.qn",
        "model.layers.3.self_attn.k_norm.weight": "l3.kn",
        "model.layers.3.self_attn.indexer.q_layernorm.weight": "l3.iqn",
        "model.layers.3.self_attn.indexer.k_layernorm.weight": "l3.ikn",
        "model.layers.1.ple.norm_conv.weight": "ple.nc",
        "model.layers.1.ple.norm_key.weight": "ple.nk",
        "model.layers.1.ple.norm_query.weight": "ple.nq",
    }
    for name, key in cases.items():
        torch.testing.assert_close(out[name].float(), ref.t[key], atol=1e-6, rtol=1e-6)


def test_indexer_fusion_is_q_rows_then_k_rows_with_one_type_marker(tiny):
    path, ref = tiny
    _, _, out = run_pipeline(path)
    base = "model.layers.3.self_attn.indexer.index_qk_proj"
    assert not any(".__q" in n or ".__k" in n for n in out)
    assert int(out[base + ".qweight_type"].item()) == int(Q.BF16)
    fused = out[base + ".qweight"]
    n_q = IDX_HEADS * IDX_DIM
    assert fused.shape[0] == n_q + IDX_DIM
    want = bf16_bytes(ref.t["l3.idx"])  # q rows first, then k rows
    assert np.array_equal(fused.numpy(), want)
    # swapped halves must not pass
    swapped = bf16_bytes(torch.cat([ref.t["l3.idx"][n_q:], ref.t["l3.idx"][:n_q]]))
    assert not np.array_equal(fused.numpy(), swapped)


def test_hyper_connection_tensors_load_as_dense_module_params(tiny):
    path, ref = tiny
    _, _, out = run_pipeline(path)
    for name, key in (
        ("model.layers.0.attn_hyper_connection.input_mix_weight_down.weight", "l0.attn.down"),
        ("model.layers.2.mlp_hyper_connection.input_mix_weight_up.weight", "l2.ffn.up"),
        ("model.hyper_connection_mixer.input_mix_weight_up.weight", "out_hc_up"),
    ):
        w = out[name]
        assert w.dtype == torch.bfloat16
        torch.testing.assert_close(w.float(), deq(q8(ref.t[key])), atol=0.02, rtol=0.02)
    # no quantized leaves for modules the model builds as nn.Linear
    assert not any(
        ("hyper_connection" in n and (n.endswith(".qweight") or n.endswith(".qweight_type")))
        for n in out
    )
    inj = out["model.layers.1.attn_hyper_connection.block_inject_weight.weight"]
    assert inj.shape == (HC, HCH) and inj.dtype == torch.bfloat16  # [4,10240] in the real file
    torch.testing.assert_close(inj.float(), ref.t["l1.attn.inject"], atol=0.02, rtol=0.02)


def test_router_shexp_gate_and_ple_conv_shapes(tiny):
    path, ref = tiny
    _, _, out = run_pipeline(path)
    r = out["model.layers.0.mlp.gate.weight"]
    assert r.dtype == torch.bfloat16 and r.shape == (N_EXP, H)
    sg = out["model.layers.0.mlp.shared_expert_gate.weight"]
    assert sg.shape == (1, H)
    torch.testing.assert_close(sg.float(), ref.t["l0.sgate"])
    pc = out["model.layers.1.ple.conv1d.weight"]
    assert pc.shape == (HCH, 1, 4)
    # NOT a GDN conv: no V-head un-tile on the PLE conv
    torch.testing.assert_close(pc.float(), ref.t["ple.conv"])


def test_experts_pass_through_per_expert_and_quantized_leaves_stay_packed(tiny):
    path, ref = tiny
    _, _, out = run_pipeline(path)
    for e in range(N_EXP):
        raw = out[f"model.layers.2.mlp.experts.{e}.down_proj.qweight"].numpy()
        torch.testing.assert_close(deq(raw), deq(q8(ref.t["l2.ex.down"][e])))
    assert "model.layers.2.mlp.experts.0.down_proj.qweight_type" in out
    # embeddings and lm_head are quantized-resident by default
    assert "model.embed_tokens.qweight" in out and "lm_head.qweight" in out
    # shared expert + attention stay Q8_0 packed
    assert int(out["model.layers.3.self_attn.q_proj.qweight_type"].item()) == int(Q.Q8_0)


def test_ple_table_is_one_quantized_tensor_and_the_hash_constants_arrive_exact(tiny):
    path, _ = tiny
    _, _, out = run_pipeline(path)
    base = "model.layers.1.ple.ple_embedding.ngram_embedding"
    assert out[base + ".qweight"].shape[0] == PLE_ROWS
    assert int(out[base + ".qweight_type"].item()) == int(Q.IQ4_NL)
    pre = "model.layers.1.ple.ple_embedding."
    for buf, want in (
        ("layer_multipliers", MULTS),
        ("ngram_heads_offsets", OFFS),
        ("ngram_heads_vocab_sizes", SIZES),
    ):
        t = out[pre + buf]
        assert t.dtype == torch.int64
        assert t.tolist() == want  # 2**62+12345 survives: not a float32/64 detour


def test_f32_modules_are_carved_out_and_indexer_halves_are_one_module(tiny):
    path, _ = tiny
    adapter = Q4.Qwen4ExpGGUFAdapter(make_config(), path)
    prefixes = adapter.unquantized_module_prefixes()
    # in_proj_a/b (F32) fold into the fused in_proj_ba, as in qwen35
    assert "model.layers.0.linear_attn.in_proj_ba" in prefixes
    assert not any(".__q" in p or ".__k" in p for p in prefixes)
    assert not any("index_qk_proj" in p for p in prefixes)  # BF16: quantized-resident
    assert adapter._module_prefix_spelling("m.indexer.index_qk_proj.__k") == "m.indexer.index_qk_proj"


# ------------------------------------------------------------------------------
# metadata consistency (loader) and the launcher refusals W172 / W173
# ------------------------------------------------------------------------------


def _write_cfg(directory, **over):
    with open(os.path.join(directory, "config.json"), "w") as fh:
        json.dump(
            {
                "architectures": ["Qwen4ExpForConditionalGeneration"],
                "text_config": text_config_dict(**over),
            },
            fh,
        )


def test_meta_check_clean_and_each_mutation_is_named(tiny):
    path, _ = tiny
    kv = Q4.read_qwen4exp_kv(path)
    cfg = text_config_dict()
    kw = dict(n_blocks_backbone=N_LAYERS, token_embd_rows=VOCAB, ple_table_rows=PLE_ROWS)
    assert Q4.qwen4exp_meta_mismatches(cfg, kv, **kw) == []
    for key, bad, needle in (
        ("num_experts", 8, "num_experts"),
        ("ple_layer_ids", [3], "ple_layer_ids"),
        ("hc_count", 4, "hc_count"),
        ("indexer_budget", 32, "indexer_budget"),
        ("linear_num_value_heads", 3, "linear_num_value_heads"),
        ("vocab_size", 99, "vocab_size"),
        ("num_hidden_layers", 5, "num_hidden_layers"),
        ("layer_types", ["linear_attention"] * N_LAYERS, "layer_types"),
        ("split_ngram_parts", 7, "split_ngram_parts"),
    ):
        out = Q4.qwen4exp_meta_mismatches(text_config_dict(**{key: bad}), kv, **kw)
        assert any(needle in line for line in out), (key, out)


def test_loader_reconcile_refuses_a_foreign_sibling_config_by_name(tiny):
    path, _ = tiny
    good = make_config()
    gguf_registry.reconcile_sibling_config(good, path, "qwen4exp")  # no raise
    bad = make_config(num_experts=512, ple_layer_ids=[2, 6])
    with pytest.raises(ValueError, match=r"num_experts: config.json says 512"):
        gguf_registry.reconcile_sibling_config(bad, path, "qwen4exp")


def test_w172_names_the_missing_sibling_config_and_stays_an_oserror(tiny, tmp_path):
    path, _ = tiny  # no config.json in tmp_path
    with pytest.raises(L.Weg2GgufConfigMissing, match="W172 Weg2GgufConfigMissing") as ei:
        L.model_config_path(path)
    assert isinstance(ei.value, OSError) and isinstance(ei.value, L.Weg2LaunchRefused)
    assert "qwen4exp" in str(ei.value) and "config.json" in str(ei.value)
    # a GGUF of an arch outside the sibling set keeps the plain FileNotFoundError
    other = str(tmp_path / "other.gguf")
    w = gguf.GGUFWriter(other, "llama")
    w.add_block_count(1)
    w.add_tensor("x", np.zeros((2, 2), np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    with pytest.raises(FileNotFoundError) as e2:
        L.model_config_path(other)
    assert not isinstance(e2.value, L.Weg2GgufConfigMissing)


def test_w173_names_a_sibling_config_of_another_model(tiny, tmp_path):
    path, _ = tiny
    _write_cfg(str(tmp_path))
    assert L.model_num_layers(path) == N_LAYERS  # consistent: unchanged behaviour
    _write_cfg(str(tmp_path), num_experts=512, hc_count=4)
    with pytest.raises(L.Weg2LaunchRefused, match=r"W173 Weg2GgufMetaMismatch.*num_experts"):
        L.model_num_layers(path)


def test_server_config_peek_refuses_a_missing_sibling_config_by_name(tiny):
    from sglang.srt.utils.hf_transformers import config as C

    path, _ = tiny
    try:
        C.get_config.__wrapped__(path, False)
    except FileNotFoundError as exc:
        assert "qwen4exp" in str(exc) and "config.json" in str(exc)
    except Exception as exc:  # pragma: no cover - environment without the gguf reader deps
        pytest.skip(f"config peek unavailable here: {exc!r}")
    else:  # pragma: no cover
        pytest.fail("expected the named refusal")


# ------------------------------------------------------------------------------
# the real export (header only)
# ------------------------------------------------------------------------------

_real = pytest.mark.skipif(
    not os.path.isfile(REAL_PART1), reason="unsloth qwen4exp export not on this machine"
)


def _real_parts():
    """The part files of the real export by directory listing (stat only; ``resolve_gguf_shard_paths`` reads headers)."""
    return sorted(
        os.path.join(REAL_DIR, f) for f in os.listdir(REAL_DIR) if f.endswith(".gguf")
    )


_REAL_ROWS = {}


def _real_rows():
    """``{tensor name: ggml ne tuple}`` of the 3 real parts -- G8 (cross-check finding D): the tensor directory walk
    (``iter_gguf_tensors`` over a 10.9 MB + 49.8 GB + 43.8 GB split set, GGUFReader parses the tokenizer KV of part 1 each
    time) took ~9 s per call and ~18 s more for the cold shard resolution, and three tests did it; under the 4 GB / timeout cap
    of pytest_gedeckelt.sh that was a timeout risk. Walked ONCE per process and kept in a JSON file keyed by (path, size,
    mtime_ns) of every part, so a changed export invalidates it; the first test that needs the headers pays, the rest and every
    later run read the cache. ``test_real_export_all_1224_tensors_mapped_none_unknown`` still runs the real shard resolution (and
    the walk itself on a cold cache) and checks the resolved parts against the directory listing the cache is keyed by."""
    if _REAL_ROWS:
        return _REAL_ROWS
    import hashlib
    import tempfile

    parts = _real_parts()
    key = hashlib.sha1(
        "|".join(f"{q}:{os.stat(q).st_size}:{os.stat(q).st_mtime_ns}" for q in parts).encode()
    ).hexdigest()[:16]
    cache = os.path.join(tempfile.gettempdir(), f"g1_qwen4exp_real_rows_{key}.json")
    try:
        with open(cache, encoding="utf-8") as fh:
            _REAL_ROWS.update({k: tuple(v) for k, v in json.load(fh).items()})
        return _REAL_ROWS
    except (OSError, ValueError):
        pass
    walked = {
        str(t.name): tuple(int(d) for d in t.shape)
        for t in gguf_shards.iter_gguf_tensors(gguf_shards.resolve_gguf_shard_paths(REAL_PART1))
    }
    _REAL_ROWS.update(walked)
    try:
        tmp = cache + f".{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({k: list(v) for k, v in walked.items()}, fh)
        os.replace(tmp, cache)
    except OSError:
        pass  # no cache: the next run walks again
    return _REAL_ROWS


def _stub_headers():
    out = {}
    for fn in sorted(os.listdir(STUB_DIR)):
        if not fn.endswith(".safetensors"):
            continue
        with open(os.path.join(STUB_DIR, fn), "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            h = json.loads(fh.read(n))
        for k, v in h.items():
            if k != "__metadata__":
                out[k] = (v["dtype"], v["shape"])
    return out


@_real
def test_real_export_all_1224_tensors_mapped_none_unknown():
    paths = gguf_shards.resolve_gguf_shard_paths(REAL_PART1)
    assert len(paths) == 3
    assert [os.path.realpath(q) for q in paths] == [os.path.realpath(q) for q in _real_parts()]
    rows = _real_rows()  # the cached walk of iter_gguf_tensors(paths) -- see _real_rows
    names = set(rows)
    assert len(names) == 1224
    mp, unknown = Q4.build_qwen4exp_name_map(names, 48, 1)
    assert unknown == [] and set(mp) == names
    assert Q4.qwen4exp_missing_roles(names, 48) == []
    roles = {n.split(".", 2)[2] if n.startswith("blk.") else n for n in names}
    assert roles == set(Q4.QWEN4EXP_ROLES)
    kv = Q4.read_qwen4exp_kv(REAL_PART1)
    assert kv["ple.layers"] == [1] and kv["expert_count"] == 512 and kv["block_count"] == 48
    assert kv["ple.layer_multipliers"] == [23703573157769, 20109073645365, 8052911324071]


@_real
@pytest.mark.skipif(not os.path.isfile(REAL_SIBLING_CFG), reason="sibling config source absent")
def test_real_export_against_the_safetensors_original_config():
    cfg = Q4.sibling_config_text(REAL_SIBLING_CFG)
    kv = Q4.read_qwen4exp_kv(REAL_PART1)
    facts = HL.gguf_header_facts(REAL_PART1)
    rows = {t.name: t.shape for t in facts.tensors}
    assert facts.arch == "qwen4exp" and facts.backbone_depth == 48
    assert (
        Q4.qwen4exp_meta_mismatches(
            cfg,
            kv,
            n_blocks_backbone=facts.backbone_depth,
            token_embd_rows=max(rows["token_embd.weight"][:2]),
            ple_table_rows=rows["per_layer_token_embd.weight"][-1],
        )
        == []
    )


@_real
@pytest.mark.skipif(not os.path.isfile(REAL_SIBLING_CFG), reason="sibling config source absent")
def test_real_export_adapter_builds_and_carves_out_the_f32_projections():
    text = types.SimpleNamespace(**Q4.sibling_config_text(REAL_SIBLING_CFG))
    cfg = types.SimpleNamespace(
        model_type="qwen4_exp",
        architectures=["Qwen4ExpForConditionalGeneration"],
        text_config=text,
    )
    cfg.get_text_config = lambda: text
    adapter = Q4.Qwen4ExpGGUFAdapter(cfg, REAL_PART1)
    assert len(adapter.build_name_map()) == 1224
    prefixes = adapter.unquantized_module_prefixes()
    assert sorted(prefixes) == sorted(
        f"model.layers.{i}.linear_attn.in_proj_ba" for i in range(48) if (i + 1) % 4 != 0
    )


@_real
@pytest.mark.skipif(not os.path.isdir(STUB_DIR), reason="HF header stubs absent (/tmp/s3/hdr)")
def test_real_export_shapes_against_the_hf_stubs():
    """Logical shapes: GGUF ne reversed (+ the inverse transforms' shape effect)
    against the safetensors header. Packed INT4/INT8 stubs carry the out dim in
    dim 0 only; dense stubs the full shape."""
    stub = _stub_headers()
    rows = {name: tuple(reversed(ne)) for name, ne in _real_rows().items()}  # cached walk, see _real_rows
    mp, _ = Q4.build_qwen4exp_name_map(rows, 48, 1)
    checked = 0
    for gname, hf in mp.items():
        shape = rows[gname]
        hf_stub = hf.replace("model.", "model.language_model.", 1) if hf.startswith("model.") else hf
        if "ngram_embedding" in hf:
            # 128 HF shards of [2500012, 160] vs ONE [320001536, 160] table
            assert shape == (128 * 2500012, 160)
            assert stub["model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight"][1] == [2500012, 160]
            checked += 1
            continue
        if ".mlp.experts." in hf:
            e = hf.split(".")[-2]  # gate_proj | up_proj | down_proj
            layer = hf.split(".")[2]
            ref = stub[f"model.language_model.layers.{layer}.mlp.experts.0.{e}.weight_packed"][1]
            assert shape[0] == 512 and shape[1] == ref[0], (gname, shape, ref)
            checked += 1
            continue
        if ".__q." in hf or ".__k." in hf:
            continue  # checked jointly below
        cands = [hf_stub, hf_stub.replace(".weight", ".weight_packed")]
        if hf_stub.endswith("A_log") or hf_stub.endswith("dt_bias"):
            cands = [hf_stub]
        key = next((c for c in cands if c in stub), None)
        assert key is not None, (gname, hf_stub)
        dtype, sshape = stub[key]
        if key.endswith("weight_packed"):
            assert shape[0] == sshape[0], (gname, shape, sshape)
        elif hf.endswith("conv1d.weight"):
            assert sshape == [shape[0], 1, shape[1]], (gname, shape, sshape)
        elif hf.endswith("shared_expert_gate.weight"):
            assert sshape == [1, shape[0]], (gname, shape, sshape)
        else:
            assert list(shape) == sshape, (gname, shape, sshape)
        checked += 1
    # indexer: q rows + k rows == the fused HF tensor
    for layer in range(3, 48, 4):
        q = rows[f"blk.{layer}.indexer.q_proj.weight"]
        k = rows[f"blk.{layer}.indexer.k_proj.weight"]
        fused = stub[f"model.language_model.layers.{layer}.self_attn.indexer.index_qk_proj.weight_packed"][1]
        assert q[0] + k[0] == fused[0] == 640
        checked += 2
    assert checked == 1224
