"""[vram-census]: bytes per tensor family, storages counted once."""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch

from sglang.srt.model_executor.vram_family_census import census, family_of


def test_family_names_follow_the_qwen4exp_layout():
    assert family_of("model.language_model.layers.3.mlp.experts.w13_weight_packed") == "experts"
    assert family_of("model.language_model.layers.3.mlp.shared_expert.up_proj.weight") == "shared_expert"
    assert family_of("model.language_model.layers.3.mlp.gate.weight") == "moe_gate"
    assert family_of("model.language_model.layers.3.linear_attn.in_proj_qkv.weight") == "linear_attn"
    assert family_of("model.language_model.layers.11.self_attn.q_proj.weight") == "self_attn"
    assert family_of("model.language_model.layers.3.ple.table") == "ple"
    assert family_of("mtp.layers.0.mlp.experts.w2") == "mtp"
    assert family_of("model.visual.blocks.0.attn.qkv.weight") == "visual"
    assert family_of("lm_head.weight_packed") == "lm_head"
    assert family_of("model.language_model.embed_tokens.weight") == "embed_tokens"
    assert family_of("model.language_model.layers.3.attn_hyper_connection.w") == "hyper_connection"
    assert family_of("something.else") == "other"


def test_census_sums_bytes_and_counts_a_shared_storage_once():
    w = torch.zeros(4, 8, dtype=torch.bfloat16)
    named = [
        ("model.language_model.layers.0.mlp.experts.w", w),
        ("model.language_model.layers.0.mlp.experts.w_alias", w),  # same storage
        ("lm_head.weight", torch.zeros(2, 2, dtype=torch.float32)),
    ]
    fam = census(named, cuda_only=False)
    assert fam == {"experts": 64, "lm_head": 16}
    assert census(named, cuda_only=True) == {}  # nothing is on a device here
