"""Qwen4-Exp's loader veto (weight_name_needed): PLE shards by name only
under the checkpoint backend, foreign experts and foreign PP layers never
read, mtp / visual (language_model_only) dropped."""

from types import SimpleNamespace

from sglang.srt.models.qwen4_exp import Qwen4ExpForConditionalGeneration as M


class _Stub:
    """The model's veto methods on a bare object (no kernels, no init)."""

    _EXPERT_ID_RE = M._EXPERT_ID_RE
    weight_name_needed = M.weight_name_needed
    _ple_ngram_embedding = M._ple_ngram_embedding
    _owned_expert_range = M._owned_expert_range


def _model(expert_range=(0, 313), ckpt=True, start=0, end=48, lm_only=True):
    experts = SimpleNamespace(_gguf_expert_shard=expert_range is not None, _gguf_expert_range=expert_range)
    emb = SimpleNamespace(_ckpt_backend=ckpt)
    layers = [SimpleNamespace(mlp=SimpleNamespace(experts=experts), ple=None) for _ in range(3)]
    layers[1].ple = SimpleNamespace(ple_embedding=SimpleNamespace(ngram_embedding=emb))
    st = _Stub()
    st.model = SimpleNamespace(layers=layers, start_layer=start, end_layer=end)
    st.config = SimpleNamespace(num_hidden_layers=48)
    st.language_model_only = lm_only
    return st


def _needed(st, name):
    return st.weight_name_needed(name)


def test_ple_shards_are_meta_under_the_checkpoint_backend_and_read_otherwise():
    assert _needed(_model(), "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_5.weight") == "meta"
    assert _needed(_model(ckpt=False), "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_5.weight") is True


def test_only_owned_experts_are_read():
    st = _model(expert_range=(313, 417))
    assert _needed(st, "model.language_model.layers.4.mlp.experts.313.gate_proj.weight_packed") is True
    assert _needed(st, "model.language_model.layers.4.mlp.experts.312.gate_proj.weight_packed") is False
    assert _needed(st, "model.language_model.layers.4.mlp.experts.417.gate_proj.weight_packed") is False
    assert _needed(_model(expert_range=None), "model.language_model.layers.4.mlp.experts.500.up_proj.weight_packed") is True
    assert st._owned_expert_range() == (313, 417) and st._owned_expert_range_cache == (313, 417)


def test_pp_mtp_visual_and_the_rest():
    st = _model(start=29, end=40)
    assert _needed(st, "model.language_model.layers.28.linear_attn.out_proj.weight_packed") is False
    assert _needed(st, "model.language_model.layers.30.linear_attn.out_proj.weight_packed") is True
    assert _needed(st, "mtp.layers.0.mlp.gate.weight") is False
    assert _needed(st, "model.visual.blocks.0.attn.qkv.weight") is False
    assert _needed(_model(lm_only=False), "model.visual.blocks.0.attn.qkv.weight") is True
    assert _needed(st, "lm_head.weight") is True and _needed(st, "model.embed_tokens.weight") is True
