"""Task #47/#48: the PP cut solver's weight terms on a checkpoint whose experts
live in a device pool (Next Flash). Hermetic: fake safetensors HEADERS only."""
import json
import os
import struct

import pytest

from sglang.srt.planner import pp_cut


def _write_shard(path, tensors):
    header = {n: {"dtype": "BF16", "shape": [int(s)], "data_offsets": [0, int(s) * 2]} for n, s in tensors.items()}
    blob = json.dumps(header).encode()
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)


def _moe_checkpoint(tmp_path):
    t = {}
    for L in range(4):
        base = f"model.language_model.layers.{L}."
        fam = "self_attn" if L == 3 else "linear_attn"
        t[base + f"{fam}.q_proj.weight"] = 1000
        t[base + "mlp.gate.weight"] = 100
        for e in range(8):
            t[base + f"mlp.experts.{e}.down_proj.weight_packed"] = 500
        t[base + "ple.key_proj.weight_packed"] = 4000
    t["model.language_model.embed_tokens.weight"] = 300
    t["lm_head.weight"] = 300
    _write_shard(str(tmp_path / "model-00001-of-00001.safetensors"), t)
    return str(tmp_path)


def test_experts_and_ple_are_measured_apart_from_the_family_means(tmp_path):
    terms = pp_cut.checkpoint_weight_terms(_moe_checkpoint(tmp_path))
    # family means hold only dense tensors (q_proj 1000 + gate 100 elements, bf16)
    assert terms.attn_layer_weight_bytes == pytest.approx(1100 * 2)
    assert terms.linear_layer_weight_bytes == pytest.approx(1100 * 2)
    assert terms.expert_layer_weight_bytes == pytest.approx(8 * 500 * 2)
    assert terms.ple_layer_weight_bytes == pytest.approx(4000 * 2)
    assert terms.num_experts == 8


def test_dense_checkpoint_terms_are_unchanged(tmp_path):
    t = {"model.layers.0.self_attn.q_proj.weight": 10, "model.layers.1.mlp.up_proj.weight": 20,
         "model.embed_tokens.weight": 5, "lm_head.weight": 5}
    _write_shard(str(tmp_path / "m.safetensors"), t)
    terms = pp_cut.checkpoint_weight_terms(str(tmp_path))
    assert terms.expert_layer_weight_bytes == 0.0 and terms.ple_layer_weight_bytes == 0.0
    assert terms.num_experts == 0
    assert terms.attn_layer_weight_bytes == pytest.approx(20) and terms.linear_layer_weight_bytes == pytest.approx(40)


def test_phase_pool_model_prices_layers_per_stage():
    m = pp_cut.PhasePoolModel(
        free_mib=(30000.0, 18000.0, 18000.0), weight_mib_per_layer=900.0,
        kv_mib_per_token_per_attn_layer=0.01, arming_floor_mib=(0.0, 0.0, 0.0),
        weight_mib_per_layer_by_stage=(500.0, 800.0, 800.0),
    )
    assert m.layer_mib(0) == 500.0 and m.layer_mib(2) == 800.0
    scalar = pp_cut.PhasePoolModel(
        free_mib=(1.0,), weight_mib_per_layer=900.0, kv_mib_per_token_per_attn_layer=0.01, arming_floor_mib=(0.0,)
    )
    assert scalar.layer_mib(0) == 900.0 and scalar.layer_mib(7) == 900.0
