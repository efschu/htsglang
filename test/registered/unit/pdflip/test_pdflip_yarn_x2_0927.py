"""YaRN x2 (NF, user order 27.09.: 524288 context, optionally x3/x4).

Offline planner dry run on 21d7e3b188 (Fork A, tmp/r989/yarn2_dryrun.md):
the D solve priced its KV duty at the fixed CONTEXT_LENGTH_TOKENS (262144)
although the form said ``--context-length 524288 --max-total-tokens 524288``
-- TP0 (the only D rank holding KV in Form A) "fitted" by 3.5 GiB it does not
have. The rope override could not reach the text model either:
``PretrainedConfig.update`` set ``text_config`` to a plain dict, and the
derived context stayed 262144 (``get_context_length`` takes factor 1 once
``original_max_position_embeddings`` is named), so a 524288 boot needed
FLLIPER_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN.

Now: the D solve takes the form's own KV duty (``group_kv_tokens``), a nested
override merges into the sub-config (neighbours kept), and the context derives
from the YaRN factor (original x factor)."""
from __future__ import annotations

import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402
from transformers import PretrainedConfig  # noqa: E402

from flliper.srt.utils.hf_transformers.common import get_context_length, get_rope_config  # noqa: E402
from flliper.srt.utils.hf_transformers.config import apply_model_override_args  # noqa: E402
from flliper.srt.pdflip import launcher as L  # noqa: E402

NF_ROPE = {"mrope_interleaved": True, "mrope_section": [11, 11, 10], "partial_rotary_factor": 0.25,
           "rope_theta": 10000000, "rope_type": "default"}
YARN2 = {"rope_type": "yarn", "factor": 2.0, "original_max_position_embeddings": 262144}
OVERRIDE_X2 = ('{"language_model_only":true,"text_config":{"rope_parameters":'
               '{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":262144}}}')


def _ns(extra_d="", extra_p=""):
    return types.SimpleNamespace(extra_d=extra_d, extra_p=extra_p)


def _nf_config():
    """The shape of the NF checkpoint's config.json (qwen4_exp wrapper,
    qwen4_exp_text sub-config, v5 rope_parameters)."""
    outer = PretrainedConfig()
    outer.text_config = PretrainedConfig(max_position_embeddings=262144)
    outer.text_config.rope_parameters = dict(NF_ROPE)
    return outer


# -- 1. the planner's KV duty is the form's --------------------------------------

def test_x1_form_keeps_the_rigs_context():
    """x1 (nf-h91-dpr.env): 262144 everywhere -> exactly the old constant."""
    ns = _ns(extra_d="--max-total-tokens 262144 --page-size 64 --context-length 262144")
    assert L.group_kv_tokens(ns, "d")[0] == L.CONTEXT_LENGTH_TOKENS == 262144
    assert L.group_kv_tokens(_ns(), "d") == (262144, "CONTEXT_LENGTH_TOKENS")


def test_x2_form_books_524288():
    """x2 (nf-h91-dpr-yarn2.env): the LAST --context-length of the D extra."""
    ns = _ns(extra_d="--max-total-tokens 524288 --context-length 524288 --max-kv-per-request 524288")
    kv, src = L.group_kv_tokens(ns, "d")
    assert kv == 524288 and "--context-length" in src


def test_a_larger_pool_is_the_duty_and_the_last_flag_wins():
    ns = _ns(extra_d="--context-length 131072 --max-total-tokens 262144 --context-length 262144")
    assert L.group_context_tokens(ns, "d")[0] == 262144
    ns = _ns(extra_d="--context-length 262144 --max-total-tokens 524288")
    assert L.group_kv_tokens(ns, "d")[0] == 524288


def test_context_derived_from_a_yarn_override_alone():
    """No --context-length in the extra: the YaRN override's original x
    factor (the number the runtime derives, part 3)."""
    ns = _ns(extra_d="--json-model-override-args '%s'" % OVERRIDE_X2)
    assert L.group_context_tokens(ns, "d")[0] == 524288
    assert L.yarn_context_tokens('{"language_model_only":true}') is None
    assert L.yarn_context_tokens('{"rope_scaling":{"type":"yarn","factor":4,'
                                 '"original_max_position_embeddings":262144}}') == 1048576


def test_d_solve_prices_the_forms_kv_not_the_constant():
    """The three D-solve call sites take the form's duty (wiring)."""
    import inspect

    src = inspect.getsource(L)
    body = src[src.index("_d_kv, _d_kv_src = group_kv_tokens(ns, \"d\")"):]
    body = body[:body.index("if plan.refusal is not None:")]
    assert body.count("kv_tokens=_d_kv,") == 3
    assert "kv_tokens=CONTEXT_LENGTH_TOKENS" not in body


def test_x2_kv_post_on_tp0_is_the_hand_calculation():
    """Fork A's hand calculation from the D solve's own cell (14143 B per
    token on TP0, Form A): 524288 tokens = 7072 MiB, +3536 MiB against x1."""
    from flliper.srt.planner import expert_residency as er

    cell = 14143
    x1 = 262144 * cell / er.MIB
    x2 = 524288 * cell / er.MIB
    assert round(x1) == 3536 and round(x2) == 7072 and round(x2 - x1) == 3536


# -- 2. the nested override reaches the text model ------------------------------

def test_nested_override_merges_the_rope_block_and_keeps_its_neighbours():
    cfg = _nf_config()
    import json
    apply_model_override_args(cfg, json.loads(OVERRIDE_X2))
    tc = cfg.text_config
    assert isinstance(tc, PretrainedConfig), "the sub-config must stay a config object"
    rp = tc.rope_parameters
    assert rp["rope_type"] == "yarn" and rp["factor"] == 2.0
    assert rp["original_max_position_embeddings"] == 262144
    assert rp["mrope_section"] == [11, 11, 10] and rp["mrope_interleaved"] is True
    assert rp["partial_rotary_factor"] == 0.25 and rp["rope_theta"] == 10000000
    assert cfg.language_model_only is True


def test_top_level_override_is_unchanged():
    """Plain keys and a dict for a non-config attribute: setattr as before."""
    cfg = PretrainedConfig()
    cfg.rope_scaling = {"type": "linear", "factor": 2.0, "low": 1}
    apply_model_override_args(cfg, {"rope_scaling": {"type": "yarn"}, "foo": 3})
    assert cfg.rope_scaling == {"type": "yarn"} and cfg.foo == 3


def test_the_attention_gets_a_yarn_mrope():
    """qwen3_5 attention: get_rope_config(text config) -> get_rope. The
    merged block builds a YaRN mrope (factory: yarn + mrope_section)."""
    from flliper.srt.layers.rotary_embedding import get_rope
    from flliper.srt.layers.rotary_embedding.mrope import YaRNScalingMRotaryEmbedding
    from flliper.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
    cfg = _nf_config()
    apply_model_override_args(cfg, {"text_config": {"rope_parameters": dict(YARN2)}})
    theta, rope = get_rope_config(cfg.text_config)
    emb = get_rope(head_size=256, rotary_dim=256, max_position=262144, rope_scaling=rope,
                   base=theta, partial_rotary_factor=rope["partial_rotary_factor"],
                   is_neox_style=True, dtype=torch.float32)
    assert isinstance(emb, YaRNScalingMRotaryEmbedding), type(emb)
    assert list(emb.mrope_section) == [11, 11, 10]
    assert emb.mrope_interleaved is True and emb.scaling_factor == 2.0


# -- 3. the context derives from the YaRN factor -------------------------------

def test_context_derives_from_the_yarn_factor():
    cfg = _nf_config()
    assert get_context_length(cfg.text_config) == 262144
    apply_model_override_args(cfg, {"text_config": {"rope_parameters": dict(YARN2)}})
    assert get_context_length(cfg.text_config) == 524288
    apply_model_override_args(cfg, {"text_config": {"rope_parameters": {"factor": 4.0}}})
    assert get_context_length(cfg.text_config) == 1048576


def test_context_without_yarn_is_unchanged():
    cfg = PretrainedConfig(max_position_embeddings=32768)
    cfg.rope_scaling = {"rope_type": "linear", "factor": 4.0}
    assert get_context_length(cfg) == 131072
    cfg.rope_scaling = {"rope_type": "llama3", "factor": 8.0, "original_max_position_embeddings": 8192}
    assert get_context_length(cfg) == 32768
