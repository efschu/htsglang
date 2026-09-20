"""fn4l 19.09.: the Qwen3.8-Flash-Next NEXTN draft served from a SEPARATE
directory carrying DominikBucko's INT4 group-32 MTP experts
(albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE runtime/mtp-int4-g32). The derived
dir's config.json targets ``mtp.layers.N.mlp.experts`` by regex -- the
downloaded config names the vLLM class ``RoutedExperts``, which our FusedMoE
never matches, and a no-match silently builds the draft UNQUANTIZED (#290/#318
class: packed names reach nothing, accept ~1.0)."""

import json
import os

import pytest
import torch

DRAFT_DIR = (
    "/spinning/llm_stuff/club-3090/models-cache/"
    "Qwen3.8-Flash-Next-MTP-INT4-g32-albucino"
)
pytestmark = pytest.mark.skipif(
    not os.path.isfile(os.path.join(DRAFT_DIR, "config.json")),
    reason="derived INT4-g32 MTP draft dir not on this box",
)


def _ct_config():
    from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )

    cfg = json.load(open(os.path.join(DRAFT_DIR, "config.json")))
    return CompressedTensorsConfig.from_config(cfg["quantization_config"])


class _Moe(torch.nn.Module):
    pass


def test_draft_experts_resolve_to_int4_group32_and_dense_layers_stay_unquantized():
    from sglang.srt.layers.quantization.compressed_tensors.schemes import (
        CompressedTensorsWNA16MoE,
    )

    q = _ct_config()
    scheme = q.get_moe_scheme(_Moe(), layer_name="mtp.layers.0.mlp.experts")
    assert isinstance(scheme, CompressedTensorsWNA16MoE)
    assert scheme.num_bits == 4 and scheme.group_size == 32
    # the draft's dense layers and the TARGET's expert names are not covered
    for name in (
        "mtp.layers.0.self_attn.q_proj",
        "mtp.layers.0.mlp.shared_expert.gate_proj",
        "mtp.fc_hidden",
    ):
        assert q.get_linear_scheme(torch.nn.Linear(4, 4), name) is None, name
    # the TARGET's expert names are neither targeted nor ignored here: this
    # config is for the draft dir only and must never be handed to the target
    with pytest.raises(ValueError):
        q.get_moe_scheme(_Moe(), layer_name="model.layers.0.mlp.experts")


def test_draft_namespace_is_packed_not_dense():
    from sglang.srt.configs.model_config import _draft_checkpoint_is_dense

    assert _draft_checkpoint_is_dense(DRAFT_DIR) is False


def test_index_names_only_the_mtp_namespace_plus_the_shared_vocab_copies():
    from sglang.srt.models.qwen4_exp_mtp import Qwen4ExpForCausalLMMTP

    idx = json.load(open(os.path.join(DRAFT_DIR, "model.safetensors.index.json")))
    names = list(idx["weight_map"])
    needed = [n for n in names if Qwen4ExpForCausalLMMTP.weight_name_needed(None, n)]
    dropped = sorted(set(names) - set(needed))
    # embed/lm_head copies are never read: the draft shares the target's modules
    assert dropped == ["lm_head.weight", "model.language_model.embed_tokens.weight"]
    packed = [n for n in needed if n.endswith(".weight_packed")]
    assert len(packed) == 512 * 3
    assert all(".mlp.experts." in n for n in packed)
