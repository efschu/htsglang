# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
"""#414: PerfCostModel plans DeepSeek V4 GGUF with attn_units=1.

THE DEFECT, as it stood
``_gguf_config_and_families`` (the GGUF-only config synthesizer
``PerfCostModel`` reads for a .gguf checkpoint) never populated ``o_groups``
in its ``text_config``. DeepSeek V4 pins ``num_key_value_heads`` to 1 (#402),
so ``PerfCostModel.attn_units`` -- ``int(text.get("o_groups") or 0) or
max(self.kv_heads, 1)`` -- silently collapsed to 1 instead of the model's
real attention unit, 8. ``--rank-tp-ratio auto`` then gridded the attention
vector on a unit of 1 instead of whole o_groups, mis-planning the shard.

This is a genuinely separate config object from the one the real server
loads: the boot-time path (``reconcile_sibling_config`` /
``utils.hf_transformers.config.get_config``) reads the sibling config.json
next to the .gguf file, which DOES carry ``o_groups`` (default 8 in
``DeepSeekV4Config``). The planner never touches that object -- it parses the
raw GGUF header from scratch in ``_gguf_config_and_families`` -- so fixing
the sibling path (already correct) does nothing for the planner.

THE FIX
Unlike ``num_key_value_heads`` / ``head_dim`` (which the gemma4 case above
this one in the same file already derives from tensor shapes when the header
scalar is absent or misleading), ``o_groups`` has NO GGUF header scalar
counterpart at all -- llama.cpp's deepseek4 writer never declares it, sibling
config.json is the only place upstream even names it. But it does not need
to be read off disk a second time: DeepSeek V4's ``wo_a`` projection is
built as ``ColumnParallelLinear(n_heads * head_dim // o_groups, o_groups *
o_lora_rank)`` (models/deepseek_v4.py, #402's o_group coupling), so the
GGML "in" dim of the on-disk ``attn_output_a`` tensor is an EXACT closed form
for ``o_groups`` given ``n_heads`` and ``head_dim`` (both already derived a
few lines above from the same tensor directory). This is derived once in
``_gguf_config_and_families`` from tensor geometry alone -- no sibling
config.json access, no new file I/O in the planner.

CPU only, no GPU, no checkpoint on disk: ``_read_gguf_metadata`` is
monkeypatched exactly as ``test_gemma4_geometry.py`` does for the same
function.
"""

import sglang.srt.uneven_perf as U
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

# DeepSeek-V4-Flash-0731 geometry (verbatim from the #402 falsifier).
N_HEADS = 64
O_GROUPS = 8
HEAD_DIM = 512
O_LORA_RANK = 1024
WO_A_IN = N_HEADS * HEAD_DIM // O_GROUPS  # 4096: the global per-group width
WO_A_OUT = O_GROUPS * O_LORA_RANK  # 8192


def _meta(with_wo_a_tensor: bool):
    scalars = {
        "general.architecture": "deepseek4",
        "deepseek4.embedding_length": 4096,
        "deepseek4.expert_count": 256,
        "deepseek4.expert_used_count": 6,
        "deepseek4.expert_feed_forward_length": 2048,
        "deepseek4.attention.head_count": N_HEADS,
        "deepseek4.attention.head_count_kv": 1,  # V4 pins this to 1 (#402)
        "deepseek4.attention.key_length": HEAD_DIM,
        "deepseek4.block_count": 4,
    }
    array_lens = {"tokenizer.ggml.tokens": 129280}
    tensors = []
    if with_wo_a_tensor:
        # GGML weight dims = [in, out].
        tensors = [
            {
                "name": "blk.0.attn_output_a.weight",
                "dims": [WO_A_IN, WO_A_OUT],
                "ggml_type": 0,
            },
        ]
    return scalars, array_lens, tensors


def test_v4_gguf_derives_o_groups_from_the_wo_a_tensor(monkeypatch):
    """THE FIX: with the wo_a tensor present, o_groups is the exact closed
    form and the planner grids attention on it, not on kv_heads."""
    monkeypatch.setattr(U, "_read_gguf_metadata", lambda p: _meta(True))
    tc = U._gguf_config_and_families("x.gguf")["text_config"]
    assert tc["o_groups"] == O_GROUPS
    assert tc["num_key_value_heads"] == 1  # still pinned; not what units grid on

    model = U.PerfCostModel.__new__(U.PerfCostModel)
    model.kv_heads = tc["num_key_value_heads"]
    model.attn_units = int(tc.get("o_groups") or 0) or max(model.kv_heads, 1)
    assert model.attn_units == O_GROUPS


def test_v4_gguf_without_the_tensor_reproduces_the_reported_collapse(monkeypatch):
    """CAN-FAIL / pre-fix reproduction: no o_groups source at all (the exact
    pre-#414 shape -- the field was never populated) collapses the grid to
    kv_heads=1, which is the bug report's ``attn_units=1``."""
    monkeypatch.setattr(U, "_read_gguf_metadata", lambda p: _meta(False))
    tc = U._gguf_config_and_families("x.gguf")["text_config"]
    assert tc["o_groups"] == 0
    assert tc["num_key_value_heads"] == 1

    model = U.PerfCostModel.__new__(U.PerfCostModel)
    model.kv_heads = tc["num_key_value_heads"]
    model.attn_units = int(tc.get("o_groups") or 0) or max(model.kv_heads, 1)
    assert model.attn_units == 1


def test_o_groups_only_derived_when_the_division_is_exact(monkeypatch):
    """A malformed/unrelated tensor whose "in" dim does not evenly divide
    n_heads*head_dim must not produce a bogus o_groups -- stay at 0 (the
    neutral "no o_groups" value) rather than silently rounding."""
    scalars, array_lens, _ = _meta(False)
    tensors = [
        {
            "name": "blk.0.attn_output_a.weight",
            "dims": [WO_A_IN + 1, WO_A_OUT],  # deliberately non-dividing
            "ggml_type": 0,
        },
    ]
    monkeypatch.setattr(
        U, "_read_gguf_metadata", lambda p: (scalars, array_lens, tensors)
    )
    tc = U._gguf_config_and_families("x.gguf")["text_config"]
    assert tc["o_groups"] == 0
