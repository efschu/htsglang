# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
"""Head geometry derived from the actual GGUF weight tensors.

Gemma4 GGUFs OMIT ``attention.head_count_kv`` and report an
``attention.key_length`` (512) that disagrees with the real ``attn_q``
projection width (4096 for head_count=16 -> head_dim 256). The header scalars
alone are missing + inconsistent; the weight tensors settle the geometry
unambiguously. Trusting the scalars would hard-error (missing key) or, if
defaulted naively to head_count, 4x the KV-cache size.
"""
import sglang.srt.uneven_perf as U


def _meta(head_count_kv=None, with_attn_tensors=True):
    scalars = {
        "general.architecture": "gemma4",
        "gemma4.embedding_length": 2816,
        "gemma4.feed_forward_length": 2112,
        "gemma4.expert_count": 128,
        "gemma4.expert_used_count": 8,
        "gemma4.expert_feed_forward_length": 704,
        "gemma4.attention.head_count": 16,
        "gemma4.attention.key_length": 512,  # deliberately misleading
        "gemma4.block_count": 30,
    }
    if head_count_kv is not None:
        scalars["gemma4.attention.head_count_kv"] = head_count_kv
    array_lens = {"tokenizer.ggml.tokens": 256000}
    tensors = []
    if with_attn_tensors:
        # GGML weight dims = [in, out]; F32 (ggml_type 0) keeps byte-sizing valid.
        tensors = [
            {"name": "blk.0.attn_q.weight", "dims": [2816, 4096], "ggml_type": 0},
            {"name": "blk.0.attn_k.weight", "dims": [2816, 2048], "ggml_type": 0},
            {"name": "blk.0.attn_v.weight", "dims": [2816, 2048], "ggml_type": 0},
            {"name": "blk.0.attn_output.weight", "dims": [4096, 2816], "ggml_type": 0},
        ]
    return scalars, array_lens, tensors


def test_gemma4_kv_heads_and_head_dim_from_tensors(monkeypatch):
    monkeypatch.setattr(U, "_read_gguf_metadata", lambda p: _meta())
    tc = U._gguf_config_and_families("x.gguf")["text_config"]
    assert tc["num_attention_heads"] == 16
    assert tc["num_key_value_heads"] == 8  # 2048/256, NOT 16 and NOT 4 (from 512)
    assert tc["head_dim"] == 256  # 4096/16, NOT the misleading key_length 512


def test_explicit_head_count_kv_wins(monkeypatch):
    # When the header DOES carry head_count_kv, trust it over the tensor guess.
    monkeypatch.setattr(U, "_read_gguf_metadata", lambda p: _meta(head_count_kv=4))
    tc = U._gguf_config_and_families("x.gguf")["text_config"]
    assert tc["num_key_value_heads"] == 4


def test_fallback_to_mha_without_attn_tensors(monkeypatch):
    # Fused-QKV / unusual naming: no attn_q|k tensor and no head_count_kv key ->
    # fall back to key_length for head_dim and MHA (kv == q) for the group count.
    monkeypatch.setattr(
        U, "_read_gguf_metadata", lambda p: _meta(with_attn_tensors=False)
    )
    tc = U._gguf_config_and_families("x.gguf")["text_config"]
    assert tc["head_dim"] == 512  # key_length fallback
    assert tc["num_key_value_heads"] == 16  # == head_count (MHA last resort)
