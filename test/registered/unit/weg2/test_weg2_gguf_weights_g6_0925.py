"""27B line G6 (2026-09-25): the launch-side weight readers on a GGUF ``--model``.

Three readers priced a checkpoint only from safetensors headers and stopped a
GGUF launch at "no *.safetensors under ...gguf":

(a) ``pp_cut.checkpoint_weight_terms`` -- ``solve_p_cut``'s weights;
(b) ``ring_table.checkpoint_stage_weights`` -- the ring's per-stage weight MiB
    (through (a));
(c) ``checkpoint_census.layer_census_from_headers`` -- widest layer, largest tag.

G6: all three read the GGUF tensor directory (``host_ledger.gguf_header_facts``,
one header read per file per process) under the loader's own names (the qwen35
adapter's ``build_name_map``), with each tensor's EXACT header bytes -- a UD mix
of quant types inside one layer is summed tensor by tensor, never estimated --
and with the NEXTN block ``blk.<depth>`` under the loader's ``mtp.*`` draft names:
it is not a backbone layer (unsloth 27B: block_count 65 - 1 = 64 layers).

Desk only: tiny qwen35 GGUFs written with gguf-py (raw bytes of real ggml quant
types) in tmp dirs; the real unsloth IQ4_XS header is read when present.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import gguf  # noqa: E402

from sglang.srt.model_loader import gguf_shards  # noqa: E402
from sglang.srt.planner import pp_cut  # noqa: E402
from sglang.srt.weg2 import checkpoint_census as CC  # noqa: E402
from sglang.srt.weg2 import host_ledger as HL  # noqa: E402
from sglang.srt.weg2 import ring_table as RT  # noqa: E402

MIB = 1024 * 1024
H = 256  # hidden: a multiple of 256, so every K / IQ quant type can hold a row
N_LAYERS = 4  # 3 GDN (linear) layers + 1 full-attention layer (index 3)
Q = gguf.GGMLQuantizationType

REAL_IQ4 = (
    "/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-27B-GGUF-unsloth/"
    "Qwen3.8-27B-UD-IQ4_XS.gguf"
)


def _text_config(n_layers=N_LAYERS):
    return {
        "model_type": "qwen3_5_text",
        "num_hidden_layers": n_layers,
        "hidden_size": H,
        "layer_types": [
            "full_attention" if (i + 1) % 4 == 0 else "linear_attention"
            for i in range(n_layers)
        ],
        "full_attention_interval": 4,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "linear_key_head_dim": 32,
        "linear_value_head_dim": 32,
        "vocab_size": 64,
        "tie_word_embeddings": False,
    }


def _raw(rows, cols, qtype):
    """Zero bytes of a real ggml type for a [rows, cols] tensor (cols = ne0)."""
    block, type_size = gguf.GGML_QUANT_SIZES[qtype]
    assert cols % block == 0, (cols, qtype)
    return np.zeros((rows, cols // block * type_size), dtype=np.uint8)


def _f32(*shape):
    return np.zeros(shape, dtype=np.float32)


def _layer(i, attention, mix):
    """``{gguf_name: array_or_(raw, qtype)}`` of one block, qwen35 spellings."""
    p = f"blk.{i}."
    t = {
        p + "attn_norm.weight": _f32(H),
        p + "post_attention_norm.weight": _f32(H),
        p + "ffn_gate.weight": (_raw(2 * H, H, mix["gate"]), mix["gate"]),
        p + "ffn_up.weight": (_raw(2 * H, H, mix["up"]), mix["up"]),
        p + "ffn_down.weight": (_raw(H, 2 * H, mix["down"]), mix["down"]),
    }
    if attention:
        t.update({
            p + "attn_q.weight": (_raw(2 * H, H, mix["q"]), mix["q"]),
            p + "attn_k.weight": (_raw(H // 4, H, Q.Q4_K), Q.Q4_K),
            p + "attn_v.weight": (_raw(H // 4, H, Q.Q5_K), Q.Q5_K),
            p + "attn_output.weight": (_raw(H, H, Q.Q5_K), Q.Q5_K),
            p + "attn_q_norm.weight": _f32(32),
            p + "attn_k_norm.weight": _f32(32),
        })
    else:
        t.update({
            p + "attn_qkv.weight": (_raw(2 * H, H, mix["qkv"]), mix["qkv"]),
            p + "attn_gate.weight": (_raw(H, H, Q.Q5_K), Q.Q5_K),
            p + "ssm_alpha.weight": (_raw(4, H, Q.Q8_0), Q.Q8_0),
            p + "ssm_beta.weight": (_raw(4, H, Q.Q8_0), Q.Q8_0),
            p + "ssm_a": _f32(4),
            p + "ssm_dt.bias": _f32(4),
            p + "ssm_conv1d.weight": _f32(4, 2 * H),
            p + "ssm_norm.weight": _f32(32),
            p + "ssm_out.weight": (_raw(H, H, Q.Q6_K), Q.Q6_K),
        })
    return t


MIX_UD = {"gate": Q.IQ2_XS, "up": Q.IQ3_S, "down": Q.Q4_K, "q": Q.IQ4_NL, "qkv": Q.IQ4_XS}
MIX_Q8 = {k: Q.Q8_0 for k in MIX_UD}


def _nextn(i):
    p = f"blk.{i}."
    t = _layer(i, attention=True, mix={k: Q.Q6_K for k in MIX_UD})
    t.update({
        p + "nextn.eh_proj.weight": (_raw(H, 2 * H, Q.Q6_K), Q.Q6_K),
        p + "nextn.enorm.weight": _f32(H),
        p + "nextn.hnorm.weight": _f32(H),
        p + "nextn.shared_head_norm.weight": _f32(H),
    })
    return t


def _write(directory, name="m-UD-IQ4_XS.gguf", mix=MIX_UD, block_count=N_LAYERS + 1,
           nextn=1, cfg_layers=N_LAYERS, model_type="qwen3_5_text"):
    """A qwen35 GGUF + sibling config; returns (path, {gguf_name: header bytes})."""
    os.makedirs(directory, exist_ok=True)
    text = _text_config(cfg_layers)
    text["model_type"] = model_type
    with open(os.path.join(directory, "config.json"), "w") as fh:
        json.dump({"architectures": ["Qwen3_5ForConditionalGeneration"],
                   "text_config": text}, fh)
    tensors = {
        "token_embd.weight": (_raw(64, H, Q.Q8_0), Q.Q8_0),
        "output.weight": (_raw(64, H, Q.Q6_K), Q.Q6_K),
        "output_norm.weight": _f32(H),
    }
    for i in range(N_LAYERS):
        tensors.update(_layer(i, attention=(i + 1) % 4 == 0, mix=mix))
    tensors.update(_nextn(N_LAYERS))
    path = os.path.join(directory, name)
    w = gguf.GGUFWriter(path, "qwen35")
    w.add_block_count(block_count)
    if nextn is not None:
        w.add_uint32("qwen35.nextn_predict_layers", nextn)
    nbytes = {}
    for tname, val in tensors.items():
        if isinstance(val, tuple):
            arr, qtype = val
            w.add_tensor(tname, arr, raw_dtype=qtype)
        else:
            arr = val
            w.add_tensor(tname, arr)
        nbytes[tname] = int(arr.nbytes)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path, nbytes


def _blk(nbytes, i):
    return sum(b for n, b in nbytes.items() if n.startswith(f"blk.{i}."))


@pytest.fixture(autouse=True)
def _fresh_caches():
    def clear():
        getattr(HL, "_GGUF_HEADER_FACTS", {}).clear()
        gguf_shards._RESOLVED_CACHE.clear()
        try:
            from sglang.srt.weg2 import gguf_census

            gguf_census._CENSUS_CACHE.clear()
        except ImportError:
            pass

    clear()
    yield
    clear()


@pytest.fixture
def ud(tmp_path):
    return _write(str(tmp_path / "ud"))


# --- the header rows -----------------------------------------------------------


def test_the_header_facts_keep_every_tensor_row_with_its_exact_bytes(ud):
    path, nbytes = ud
    facts = HL.gguf_header_facts(path)
    rows = {r.name: r for r in facts.tensors}
    assert set(rows) == set(nbytes)
    assert {n: r.n_bytes for n, r in rows.items()} == nbytes
    assert rows["blk.0.attn_qkv.weight"].ggml_type == "IQ4_XS"
    assert rows["blk.0.ffn_gate.weight"].ggml_type == "IQ2_XS"
    # the digest is the G2 algorithm over the same rows, unchanged
    import hashlib

    lines = sorted("%s:%s:%s" % (r.name, r.ggml_type, "x".join(map(str, r.shape)))
                   for r in facts.tensors)
    assert facts.tensor_digest == hashlib.sha256("\n".join(lines).encode()).hexdigest()


# --- (a) pp_cut.checkpoint_weight_terms -----------------------------------------


def test_a_gguf_file_yields_its_weight_terms_from_the_header(ud):
    """RED on 644de86ef9: DraftResidencyUnavailable 'no *.safetensors under ...gguf'."""
    path, nbytes = ud
    t = pp_cut.checkpoint_weight_terms(path)
    assert t.source == path
    assert t.n_layers == N_LAYERS, "the NEXTN block blk.4 is not a backbone layer"
    assert t.attention_layer_indices == (3,)
    assert t.attn_layer_weight_bytes == _blk(nbytes, 3)
    assert t.linear_layer_weight_bytes == sum(_blk(nbytes, i) for i in (0, 1, 2)) / 3
    assert t.embedding_weight_bytes == nbytes["token_embd.weight"]
    assert t.lm_head_weight_bytes == nbytes["output.weight"]
    assert t.replicated_breakdown == {"mtp": _blk(nbytes, N_LAYERS)}
    assert t.replicated_weight_bytes == _blk(nbytes, N_LAYERS)


def test_the_quant_mix_is_priced_tensor_by_tensor(tmp_path):
    """Same names, same shapes, other ggml types -> other bytes, exactly."""
    ud_path, ud_bytes = _write(str(tmp_path / "ud"))
    q8_path, q8_bytes = _write(str(tmp_path / "q8"), name="m-Q8_0.gguf", mix=MIX_Q8)
    a = pp_cut.checkpoint_weight_terms(ud_path)
    b = pp_cut.checkpoint_weight_terms(q8_path)
    assert a.linear_layer_weight_bytes == sum(_blk(ud_bytes, i) for i in (0, 1, 2)) / 3
    assert b.linear_layer_weight_bytes == sum(_blk(q8_bytes, i) for i in (0, 1, 2)) / 3
    assert a.linear_layer_weight_bytes != b.linear_layer_weight_bytes
    assert a.attn_layer_weight_bytes != b.attn_layer_weight_bytes


def test_a_safetensors_directory_never_reaches_the_gguf_reader(tmp_path):
    d = tmp_path / "st"
    d.mkdir()
    assert pp_cut._gguf_weight_sizes(str(d)) is None
    with pytest.raises(pp_cut.DraftResidencyUnavailable, match="no \\*.safetensors under"):
        pp_cut.checkpoint_weight_terms(str(d))


def test_a_depth_the_config_does_not_state_is_refused_by_name(tmp_path):
    # block_count 6 - nextn 1 = 5 layers in the file, 4 in the config (W162's case)
    path, _ = _write(str(tmp_path / "deep"), block_count=N_LAYERS + 2)
    with pytest.raises(pp_cut.DraftResidencyUnavailable, match="depth 5"):
        pp_cut.checkpoint_weight_terms(path)


def test_a_model_type_without_a_gguf_family_is_refused_by_name(tmp_path):
    path, _ = _write(str(tmp_path / "llama"), model_type="llama")
    with pytest.raises(pp_cut.DraftResidencyUnavailable, match="no bespoke GGUF family"):
        pp_cut.checkpoint_weight_terms(path)


# --- (b) ring_table.checkpoint_stage_weights ------------------------------------


def test_the_ring_prices_each_stage_from_the_gguf(ud):
    path, nbytes = ud
    lin = sum(_blk(nbytes, i) for i in (0, 1, 2)) / 3 / MIB
    stages = RT.checkpoint_stage_weights(path, [3, 1], [0, 1], carries_drafter=False)
    assert [s.linear_layers for s in stages] == [3, 0]
    assert [s.attn_layers for s in stages] == [0, 1]
    assert stages[0].linear_mib == pytest.approx(3 * lin)
    assert stages[1].attn_mib == pytest.approx(_blk(nbytes, 3) / MIB)
    assert stages[0].embedding_mib == pytest.approx(nbytes["token_embd.weight"] / MIB)
    assert stages[1].lm_head_mib == pytest.approx(nbytes["output.weight"] / MIB)
    # the MTP block is drafter-only: never replicated onto every stage
    assert [s.replicated_mib for s in stages] == [0.0, 0.0]
    last = RT.checkpoint_stage_weights(path, [3, 1], [0, 1], carries_drafter=True)[-1]
    assert last.drafter_mib == pytest.approx(
        (_blk(nbytes, N_LAYERS) + nbytes["token_embd.weight"] + nbytes["output.weight"]) / MIB)


# --- (c) checkpoint_census ------------------------------------------------------


def test_the_layer_census_reads_the_gguf_and_folds_mtp_like_safetensors(ud):
    """RED on 644de86ef9: W14 'no safetensors shard under ...gguf'."""
    path, nbytes = ud
    mtp = _blk(nbytes, N_LAYERS)
    mtp_layer = sum(b for n, b in nbytes.items()
                    if n.startswith(f"blk.{N_LAYERS}.") and ".nextn." not in n)
    unlayered = (nbytes["token_embd.weight"] + nbytes["output.weight"]
                 + nbytes["output_norm.weight"])
    # like the safetensors checkpoint: the MTP tree's `mtp.layers.0.` names fold
    # into layer 0 unless excluded, and its own tensors stay unlayered
    c = CC.layer_census_from_headers(path)
    assert c.n_layers == N_LAYERS and c.files == 1
    assert dict(c.layer_bytes)[0] == _blk(nbytes, 0) + mtp_layer
    assert c.unlayered_bytes == unlayered + (mtp - mtp_layer)
    # the weight tags' view (#1374): the MTP tree excluded by name
    x = CC.layer_census_from_headers(path, exclude_prefixes=CC.MTP_TREE_PREFIXES)
    assert dict(x.layer_bytes) == {i: _blk(nbytes, i) for i in range(N_LAYERS)}
    assert x.unlayered_bytes == unlayered
    assert dict(x.layer_classes)[0] == tuple(sorted({
        "A_log", "conv1d", "down_proj", "dt_bias", "gate_proj", "in_proj_a",
        "in_proj_b", "in_proj_qkv", "in_proj_z", "input_layernorm", "norm",
        "out_proj", "post_attention_layernorm", "up_proj"}))


def test_the_largest_tag_is_the_widest_window_of_backbone_layers(ud):
    path, nbytes = ud
    per = [_blk(nbytes, i) for i in range(N_LAYERS)]
    assert CC.max_tag_bytes_from_census(path, 2) == max(
        per[i] + per[i + 1] for i in range(N_LAYERS - 1))


def test_an_unpriceable_gguf_is_w14_not_a_default(tmp_path):
    path, _ = _write(str(tmp_path / "llama"), model_type="llama")
    with pytest.raises(CC.Weg2XchgWidestLayerUnreadable, match="W14"):
        CC.layer_census_from_headers(path)


# --- the real unsloth header, read-only -----------------------------------------


@pytest.mark.skipif(not os.path.isfile(REAL_IQ4), reason="unsloth IQ4_XS not present")
def test_the_real_unsloth_iq4_xs_header():
    """One header read (~8.5 s), every figure from the file's own rows."""
    facts = HL.gguf_header_facts(REAL_IQ4)
    assert (facts.arch, facts.block_count, facts.nextn_predict_layers) == ("qwen35", 65, 1)
    by = {}
    for r in facts.tensors:
        key = int(r.name.split(".")[1]) if r.name.startswith("blk.") else r.name
        by[key] = by.get(key, 0) + r.n_bytes
    total = sum(r.n_bytes for r in facts.tensors)

    t = pp_cut.checkpoint_weight_terms(REAL_IQ4)
    assert t.n_layers == 64, "blk.64 is the NEXTN block, not layer 64"
    assert t.attention_layer_indices == tuple(range(3, 64, 4))
    assert t.embedding_weight_bytes == by["token_embd.weight"] == 546304000
    assert t.lm_head_weight_bytes == by["output.weight"] == 874086400
    assert t.replicated_breakdown == {"mtp": by[64]} and by[64] == 351008768
    attn = [by[i] for i in range(3, 64, 4)]
    lin = [by[i] for i in range(64) if (i + 1) % 4]
    assert t.attn_layer_weight_bytes == pytest.approx(sum(attn) / 16)
    assert t.linear_layer_weight_bytes == pytest.approx(sum(lin) / 48)

    c = CC.layer_census_from_headers(REAL_IQ4, exclude_prefixes=CC.MTP_TREE_PREFIXES)
    assert c.n_layers == 64
    assert dict(c.layer_bytes)[0] == by[0] == 155351936
    assert dict(c.layer_bytes)[3] == by[3] == 187557888
    assert c.unlayered_bytes == by["token_embd.weight"] + by["output.weight"] + by["output_norm.weight"]
    assert c.layer_total_bytes + c.unlayered_bytes + by[64] == total
