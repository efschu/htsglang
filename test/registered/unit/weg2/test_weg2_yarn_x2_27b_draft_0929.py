"""YaRN x2 for the 27B (F14, 29.09.): the DFlash draft beside a YaRN target.

The 27B takes the NF way unchanged (72c5941ef2): ``--context-length 524288``
plus ``{"language_model_only":true,"text_config":{"rope_parameters":{yarn x2},
"max_position_embeddings":524288}}`` in ``--extra-p``/``--extra-d``. What the
NF form never met is its DRAFT: NF drafts with an MTP head whose config is the
same multimodal wrapper, while the 27B drafts with DFlash2 (``DFlash2DraftModel``,
a plain ``qwen3`` text config, 5 x ``sliding_attention``, window 2048,
``max_position_embeddings`` 262144 inherited from the target). One
``--json-model-override-args`` reaches both (``ModelConfig.from_server_args``
passes ``server_args.json_model_override_args`` for the draft too), and one
``--context-length`` reaches both (``draft_worker_common``: the target's
context_len; ``model_runner`` DFLASH layer capture: ``server_args.context_length``).

Measured on the real checkpoints (desk probe, CPU, 895559fed2): the target
builds (ctx 524288, rope yarn x2), the draft dies with ``AssertionError`` in
``get_hf_text_config`` -- ``apply_model_override_args`` set ``text_config`` on
the text-only draft config as a plain dict, which then shadows the draft's own
text config and has no ``num_attention_heads``. Past that, the draft's derived
context (262144) is below 524288 and ``_derive_context_length`` refuses a draft
without ``SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN`` -- a global switch that
would also silence the TARGET's guard (a lost YaRN override would then boot
524288 on the default rope).

Now: (1) a partial ``text_config`` override on a config that has no text
sub-config (the config IS the text config) is not applied -- it addresses a
multimodal wrapper; a full one (with ``num_attention_heads``) stays upstream's
setattr. (2) A draft whose every attention layer is window-bounded takes the
target's longer context without the env: RoPE scores depend on the distance
m - n only, and the window caps that distance at ``sliding_window``, so no
position beyond the checkpoint's own ``max_position_embeddings`` puts an unseen
distance into any attention (the same horizon ``cross_algo_utils.
derive_ctx_gate_threshold`` already reads off this drafter). A draft with any
full-attention layer keeps the refusal."""
from __future__ import annotations

import json
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
from transformers import PretrainedConfig  # noqa: E402

from sglang.srt.utils.hf_transformers.config import apply_model_override_args  # noqa: E402

YARN2_OVERRIDE = {
    "language_model_only": True,
    "text_config": {
        "rope_parameters": {
            "rope_type": "yarn", "factor": 2.0, "original_max_position_embeddings": 262144,
            "rope_theta": 10000000, "partial_rotary_factor": 0.25,
            "mrope_section": [11, 11, 10], "mrope_interleaved": True,
        },
        "max_position_embeddings": 524288,
    },
}

#: The DFlash2 draft's config.json (Qwen3.8-27B-DFlash2-W8-lued) without its
#: quantization block -- the fields ModelConfig reads.
DRAFT_CONFIG = {
    "architectures": ["DFlash2DraftModel"],
    "model_type": "qwen3",
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 248044,
    "eos_token_id": 248044,
    "dflash_config": {"block_size": 8, "conv_group_size": 16, "conv_kernel_size": 2,
                      "mask_token_id": 248070, "selector_rank": 256, "selector_top_k": 16,
                      "target_layer_ids": [5, 19, 33, 47, 61]},
    "dtype": "bfloat16",
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 5120,
    "initializer_range": 0.02,
    "intermediate_size": 17408,
    "is_causal": False,
    "layer_types": ["sliding_attention"] * 5,
    "max_position_embeddings": 262144,
    "max_window_layers": 5,
    "num_attention_heads": 32,
    "num_hidden_layers": 5,
    "num_key_value_heads": 8,
    "num_target_layers": 64,
    "rms_norm_eps": 1e-06,
    "rope_parameters": {"rope_theta": 10000000, "rope_type": "default"},
    "sliding_window": 2048,
    "tie_word_embeddings": False,
    "use_cache": True,
    "use_sliding_window": True,
    "vocab_size": 248320,
}


def _draft_dir(tmp_path, **changes):
    cfg = dict(DRAFT_CONFIG)
    cfg.update(changes)
    d = tmp_path / "draft"
    d.mkdir()
    (d / "config.json").write_text(json.dumps(cfg))
    return str(d)


def _model_config(path, *, context_length, override, is_draft_model):
    from sglang.srt.configs.model_config import ModelConfig

    return ModelConfig(model_path=path, trust_remote_code=True, context_length=context_length,
                       model_override_args=json.dumps(override), is_draft_model=is_draft_model)


# -- 1. the override does not plant a text_config on a text-only config ----------

def test_partial_text_config_override_skips_a_text_only_config():
    """The draft's shape: no text sub-config. Before: ``text_config`` became a
    plain dict attribute (and get_hf_text_config asserted on it)."""
    cfg = PretrainedConfig(num_attention_heads=32, max_position_embeddings=262144)
    cfg.rope_parameters = {"rope_theta": 10000000, "rope_type": "default"}
    apply_model_override_args(cfg, json.loads(json.dumps(YARN2_OVERRIDE)))
    assert getattr(cfg, "text_config", None) is None
    assert cfg.language_model_only is True           # top-level keys still land
    assert cfg.rope_parameters == {"rope_theta": 10000000, "rope_type": "default"}
    assert cfg.max_position_embeddings == 262144      # the draft keeps its own rope


def test_full_text_config_override_on_text_only_config_is_upstreams_setattr():
    """Byte-equality guard: an override that CAN stand as a text config
    (it names num_attention_heads) keeps upstream's behaviour."""
    cfg = PretrainedConfig(num_attention_heads=32)
    full = {"num_attention_heads": 8, "hidden_size": 64}
    apply_model_override_args(cfg, {"text_config": dict(full)})
    assert cfg.text_config == full


def test_nested_override_still_merges_into_a_wrapper():
    """Byte-equality guard (72c5941ef2): a wrapper's sub-config is merged."""
    outer = PretrainedConfig()
    outer.text_config = PretrainedConfig(max_position_embeddings=262144, num_attention_heads=24)
    outer.text_config.rope_parameters = {"rope_type": "default", "rope_theta": 10000000,
                                         "mrope_section": [11, 11, 10]}
    apply_model_override_args(outer, json.loads(json.dumps(YARN2_OVERRIDE)))
    rp = outer.text_config.rope_parameters
    assert rp["rope_type"] == "yarn" and rp["factor"] == 2.0 and rp["mrope_section"] == [11, 11, 10]
    assert outer.text_config.max_position_embeddings == 524288


# -- 2. the window-bounded draft takes the target's context ----------------------

def test_dflash_draft_builds_under_the_yarn2_override_and_takes_524288(tmp_path, monkeypatch):
    monkeypatch.delenv("SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN", raising=False)
    mc = _model_config(_draft_dir(tmp_path), context_length=524288, override=YARN2_OVERRIDE,
                       is_draft_model=True)
    t = mc.hf_text_config
    assert mc.context_len == 524288
    assert t.max_position_embeddings == 524288        # the rope cache covers every position
    assert t.rope_parameters["rope_type"] == "default"  # no YaRN, no mrope on the draft
    assert getattr(mc.hf_config, "text_config", None) is None


def test_draft_with_a_full_attention_layer_keeps_the_refusal(tmp_path, monkeypatch):
    """Danger direction: a draft that sees the whole prefix at some layer
    would meet distances past its training -- still refused by name."""
    monkeypatch.delenv("SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN", raising=False)
    path = _draft_dir(tmp_path, layer_types=["sliding_attention"] * 4 + ["full_attention"])
    with pytest.raises(ValueError, match="SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"):
        _model_config(path, context_length=524288, override={}, is_draft_model=True)


def test_draft_with_sliding_window_disabled_keeps_the_refusal(tmp_path, monkeypatch):
    monkeypatch.delenv("SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN", raising=False)
    path = _draft_dir(tmp_path, use_sliding_window=False)
    with pytest.raises(ValueError, match="SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"):
        _model_config(path, context_length=524288, override={}, is_draft_model=True)


def test_target_side_is_not_loosened(tmp_path, monkeypatch):
    """The window rule is a DRAFT rule: the same window-bounded config loaded
    as a TARGET keeps the refusal (a lost YaRN override must not boot 524288
    on the default rope)."""
    monkeypatch.delenv("SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN", raising=False)
    with pytest.raises(ValueError, match="SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"):
        _model_config(_draft_dir(tmp_path), context_length=524288, override={},
                      is_draft_model=False)


def test_draft_at_its_own_context_is_unchanged(tmp_path, monkeypatch):
    """Byte-equality guard: x1 (262144) never reaches the new branch."""
    monkeypatch.delenv("SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN", raising=False)
    mc = _model_config(_draft_dir(tmp_path), context_length=262144,
                       override={"language_model_only": True}, is_draft_model=True)
    assert mc.context_len == 262144 and mc.hf_text_config.max_position_embeddings == 262144
