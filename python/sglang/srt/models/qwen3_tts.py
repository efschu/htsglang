# SPDX-License-Identifier: Apache-2.0
"""Qwen3-TTS talker as a native model lane (#488), slice 1: the sharded trunk.

WHAT THIS FILE IS
-----------------
The autoregressive half of Qwen3-TTS, built against this fork's own layers so
it inherits uneven TP, the paged KV pool, and (once the decode regime lands)
CUDA-graph capture. Two modules:

* ``Qwen3TTSTalkerTrunk`` -- 28 Qwen3-shaped decoder layers, M-RoPE, q/k RMSNorm
  over ``head_dim``. ONE decode step per audio frame, one KV position per frame.
* ``Qwen3TTSCodePredictor`` -- the 5-layer depth transformer that expands one
  frame into its remaining 15 residual codebook entries.

THE POSITION INVARIANT, stated once
-----------------------------------
The 16 codebook entries of a frame are ONE talker sequence position, not 16.
The residual codes must never enter the talker's KV sequence. That is why the
code predictor deliberately does NOT use ``RadixAttention``: it runs over a
private scratch cache of at most ``num_code_groups`` slots that is reset at
every frame, while only the trunk touches the paged pool. Routing the predictor
through the paged pool would inflate the talker's sequence by 16x and desync
every position the M-RoPE depends on.

WHAT IS *NOT* IN THE LANE, AND WHY
----------------------------------
The speaker encoder (ECAPA-TDNN, ~9 MiB) and the 12 Hz codec/vocoder under
``speech_tokenizer/`` are NOT sharded and NOT loaded here. Both run ONCE PER
TURN, not once per decode step, so they carry no share of the real-time factor
and sharding them would buy nothing while adding two collective-bearing
families. They stay in-process modules with their own #286 ledger asset classes,
which is exactly how ``srt/translator/inprocess_tts.py`` already registers them
(``talker_trunk`` / ``code_predictor`` / ``speaker_encoder`` / ``codec``).
``load_weights`` therefore ACCOUNTS for those checkpoint prefixes explicitly
rather than dropping them silently -- see ``_NON_LANE_PREFIXES``.

PER-FAMILY SPLIT DECISIONS (CLAUDE.md PER-FAMILY x PER-PHASE law)
----------------------------------------------------------------
* ``text_embedding`` (151936 x 2048 = 593 MiB) is vocab-parallel: it is read
  once per PREFILL token and it is 68 % of the checkpoint, so the memory axis
  wins and the one all-reduce per prefill is free.
* ``codec_embedding`` (trunk, 6 MiB) and the predictor's 15 codebook embeddings
  (4 MiB each) are REPLICATED. They are read once per DECODE step each -- a
  vocab-parallel lookup would cost 16 all-reduces per audio frame against a
  6 MiB table. Different phase, different optimum, same family.
* ``codec_head`` is a normal ``ParallelLMHead``; the predictor's 15 heads are
  column-parallel with a gathered output, because their 2048-wide logits are
  sampled locally per step.
"""

from __future__ import annotations

import logging
from typing import Iterable, List, Optional, Tuple

import torch
from torch import nn

from sglang.srt.configs.qwen3_tts import Qwen3TTSConfig
from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.distributed.utils import (
    attn_kv_replicated,
    attn_q_partition_groups,
    attn_q_partition_units,
    tp_partition_size,
    tp_plan_active,
)
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)

#: Checkpoint prefixes that belong to in-process modules, not to this lane.
#: Listed rather than ignored: a prefix that is neither loaded nor listed is a
#: silently dropped weight, which is the failure class CLAUDE.md's
#: "SUCCESS CLAIMS ARE NOT EVIDENCE" rule exists for.
_NON_LANE_PREFIXES = ("speaker_encoder.",)


class Qwen3TTSMLP(nn.Module):
    """Standard gated MLP. Under an uneven plan both projections coarsen the
    intermediate dimension identically (the #385 symmetric-block rule lives in
    ``layers/linear.py``), so no per-family unit override is needed here."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation {hidden_act!r}; expected silu")
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


def _head_split(
    total_num_heads: int, total_num_kv_heads: int, tp_size: int, tp_rank: int
) -> Tuple[int, int, int, Optional[int], bool]:
    """Per-rank (q_heads, kv_heads, q_units, q_groups, kv_replicated).

    Uneven-TP aware by construction: the head counts come from the INSTALLED
    shard plan, the same source ``QKVParallelLinear`` and ``RowParallelLinear``
    consult, so the head axis and the o_proj input axis cannot disagree. This
    is what ``models/qwen3.py:63`` refuses a non-uniform ratio for -- deriving
    ``total // tp_size`` there would be an even split against ratio-following
    projections.

    Note for this checkpoint specifically: 16 q / 8 kv over 3 ranks does not
    divide, so even a UNIFORM ratio vector produces an uneven head split
    ([6,6,4] q over [3,3,2] kv). The plan path is therefore the only correct
    path at tp=3 -- the classic even branch asserts and would refuse the boot.
    """
    if tp_plan_active(tp_size):
        kv_replicated = attn_kv_replicated(tp_size, total_num_kv_heads)
        q_units = attn_q_partition_units(total_num_heads, total_num_kv_heads, tp_size)
        q_groups = attn_q_partition_groups(total_num_kv_heads, tp_size)
        num_heads = tp_partition_size(
            total_num_heads, tp_size, tp_rank, q_units, groups=q_groups
        )
        if kv_replicated:
            num_kv_heads = total_num_kv_heads
        else:
            num_kv_heads = tp_partition_size(
                total_num_kv_heads, tp_size, tp_rank, total_num_kv_heads
            )
        return num_heads, num_kv_heads, q_units, q_groups, kv_replicated

    if total_num_heads % tp_size != 0:
        raise ValueError(
            f"Qwen3-TTS talker has {total_num_heads} attention heads, which "
            f"does not divide across {tp_size} ranks. This geometry needs the "
            f"uneven-TP shard plan: pass --rank-tp-ratio (a uniform vector is "
            f"accepted and is what produces the [6,6,4]/[3,3,2] split here)."
        )
    if total_num_kv_heads >= tp_size:
        if total_num_kv_heads % tp_size != 0:
            raise ValueError(
                f"{total_num_kv_heads} kv heads do not divide across {tp_size} "
                f"ranks; use --rank-tp-ratio."
            )
    elif tp_size % total_num_kv_heads != 0:
        raise ValueError(
            f"{tp_size} ranks is not a multiple of {total_num_kv_heads} kv heads"
        )
    return (
        total_num_heads // tp_size,
        max(1, total_num_kv_heads // tp_size),
        total_num_kv_heads,
        None,
        False,
    )


class Qwen3TTSTalkerAttention(nn.Module):
    """Trunk attention: M-RoPE, q/k RMSNorm over head_dim, paged KV."""

    def __init__(
        self,
        config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.attn_tp_rank = get_parallel().attn_tp_rank
        self.attn_tp_size = get_parallel().attn_tp_size
        self.hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim or (self.hidden_size // self.total_num_heads)

        (
            self.num_heads,
            self.num_kv_heads,
            q_units,
            q_groups,
            self._kv_replicated,
        ) = _head_split(
            self.total_num_heads,
            self.total_num_kv_heads,
            self.attn_tp_size,
            self.attn_tp_rank,
        )
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=config.attention_bias,
            quant_config=quant_config,
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
            prefix=add_prefix("qkv_proj", prefix),
            q_shard_unit_count=q_units,
            q_shard_groups=q_groups,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=config.attention_bias,
            quant_config=quant_config,
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
            prefix=add_prefix("o_proj", prefix),
            # SAME source as qkv q_shard_units/groups (#116): a mismatch here
            # mis-shards o_proj instead of raising.
            tp_units=q_units,
            tp_q_groups=q_groups,
        )
        # rope_scaling arrives already normalised and gated by the config class
        # -- `mrope_interleaved` is the key the factory reads, and the
        # checkpoint writes `interleaved`. See configs/qwen3_tts.py.
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            base=config.rope_theta,
            rope_scaling=config.rope_scaling,
            is_neox_style=True,
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            prefix=add_prefix("attn", prefix),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = self.q_norm(q.reshape(-1, self.head_dim)).reshape(q.shape)
        k = self.k_norm(k.reshape(-1, self.head_dim)).reshape(k.shape)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output


class Qwen3TTSTalkerDecoderLayer(nn.Module):
    def __init__(
        self,
        config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3TTSTalkerAttention(
            config, layer_id, quant_config, prefix=add_prefix("self_attn", prefix)
        )
        self.mlp = Qwen3TTSMLP(
            config.hidden_size,
            config.intermediate_size,
            config.hidden_act,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states, forward_batch)
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual
        )
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3TTSTextProjection(nn.Module):
    """``linear_fc1`` -> silu -> ``linear_fc2``, both WITH bias.

    Named after the checkpoint, not after the runtime's usual ``gate_up/down``
    convention -- it is a plain resize MLP, not a gated one, so the standard
    stacked mapping does not apply to it (feasibility trap 3). Replicated: it
    runs once per prefill over the text prompt, and 12 MiB per rank is cheaper
    than the collective a split would need.
    """

    def __init__(
        self,
        input_size: int,
        intermediate_size: int,
        output_size: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.linear_fc1 = ReplicatedLinear(
            input_size,
            intermediate_size,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("linear_fc1", prefix),
        )
        self.linear_fc2 = ReplicatedLinear(
            intermediate_size,
            output_size,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("linear_fc2", prefix),
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.linear_fc1(x)
        x = self.act(x)
        x, _ = self.linear_fc2(x)
        return x


class Qwen3TTSTalkerTrunk(nn.Module):
    """The 28-layer autoregressive trunk. One step == one audio frame."""

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        # Vocab-parallel: 593 MiB and prefill-only. See module docstring.
        self.text_embedding = VocabParallelEmbedding(
            config.text_vocab_size,
            config.text_hidden_size,
            prefix=add_prefix("text_embedding", prefix),
        )
        # Replicated: 6 MiB, read once per DECODE step. A vocab-parallel
        # lookup here would add an all-reduce to every frame.
        self.codec_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [
                Qwen3TTSTalkerDecoderLayer(
                    config,
                    layer_id,
                    quant_config,
                    prefix=add_prefix(f"layers.{layer_id}", prefix),
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """``input_embeds`` is mandatory and that is the point.

        The talker's next input is ``codec_embedding[c0] + sum_q
        codec_embedding_q[c_q] + text_step`` -- a VECTOR, never a token id. The
        decode regime that drives this lane has to keep the embeds channel
        alive; ``schedule_batch.py`` clears it on entry to decode today, which
        is the single scheduler unblock this lane needs (DESIGN_466 §11.2).
        """
        hidden_states = input_embeds
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions, hidden_states, residual, forward_batch
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3TTSCodePredictorAttention(nn.Module):
    """Depth-transformer attention over a PRIVATE scratch cache.

    Deliberately not ``RadixAttention``: these steps are sub-positions of ONE
    talker frame and must not enter the paged KV sequence (module docstring,
    "THE POSITION INVARIANT"). The cache is at most ``num_code_groups`` slots
    and is reset per frame, so a dense SDPA over it is both correct and
    cheaper than a paged lookup would be.
    """

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.attn_tp_rank = get_parallel().attn_tp_rank
        self.attn_tp_size = get_parallel().attn_tp_size
        self.hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim or (self.hidden_size // self.total_num_heads)
        (
            self.num_heads,
            self.num_kv_heads,
            q_units,
            q_groups,
            _,
        ) = _head_split(
            self.total_num_heads,
            self.total_num_kv_heads,
            self.attn_tp_size,
            self.attn_tp_rank,
        )
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=config.attention_bias,
            quant_config=quant_config,
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
            prefix=add_prefix("qkv_proj", prefix),
            q_shard_unit_count=q_units,
            q_shard_groups=q_groups,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=config.attention_bias,
            quant_config=quant_config,
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
            prefix=add_prefix("o_proj", prefix),
            tp_units=q_units,
            tp_q_groups=q_groups,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            base=config.rope_theta,
            rope_scaling=None,
            is_neox_style=True,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        cache: List[Optional[torch.Tensor]],
    ) -> torch.Tensor:
        """``cache`` is a two-slot list ``[k, v]`` owned by the caller and
        reset per frame; it is grown in place along the sequence axis."""
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = self.q_norm(q.reshape(-1, self.head_dim)).reshape(q.shape)
        k = self.k_norm(k.reshape(-1, self.head_dim)).reshape(k.shape)
        q, k = self.rotary_emb(positions, q, k)

        n = q.shape[0]
        q = q.view(n, self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(n, self.num_kv_heads, self.head_dim).transpose(0, 1)
        v = v.view(n, self.num_kv_heads, self.head_dim).transpose(0, 1)
        if cache[0] is not None:
            k = torch.cat([cache[0], k], dim=1)
            v = torch.cat([cache[1], v], dim=1)
        cache[0], cache[1] = k, v

        if self.num_heads != self.num_kv_heads:
            repeat = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat, dim=0)
            v = v.repeat_interleave(repeat, dim=0)
        out = torch.nn.functional.scaled_dot_product_attention(
            q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), is_causal=n > 1
        )
        out = out.squeeze(0).transpose(0, 1).reshape(n, self.q_size)
        out, _ = self.o_proj(out)
        return out


class Qwen3TTSCodePredictorLayer(nn.Module):
    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3TTSCodePredictorAttention(
            config, quant_config, prefix=add_prefix("self_attn", prefix)
        )
        self.mlp = Qwen3TTSMLP(
            config.hidden_size,
            config.intermediate_size,
            config.hidden_act,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        cache: List[Optional[torch.Tensor]],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(positions, hidden_states, cache)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class Qwen3TTSCodePredictor(nn.Module):
    """Expands one frame's group-0 code into the remaining 15 residual codes."""

    def __init__(
        self,
        config,
        talker_hidden_size: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.num_code_groups = config.num_code_groups
        self.layers = nn.ModuleList(
            [
                Qwen3TTSCodePredictorLayer(
                    config,
                    quant_config,
                    prefix=add_prefix(f"model.layers.{i}", prefix),
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Replicated, per the codec-embedding rule in the module docstring.
        self.codec_embedding = nn.ModuleList(
            [
                nn.Embedding(config.vocab_size, talker_hidden_size)
                for _ in range(config.num_code_groups - 1)
            ]
        )
        self.lm_head = nn.ModuleList(
            [
                ReplicatedLinear(
                    config.hidden_size,
                    config.vocab_size,
                    bias=False,
                    quant_config=quant_config,
                    prefix=add_prefix(f"lm_head.{i}", prefix),
                )
                for i in range(config.num_code_groups - 1)
            ]
        )
        # Identity whenever the two hidden sizes agree, which is the case for
        # every published Qwen3-TTS size -- the checkpoint carries no weight
        # for it. Built from the CONFIG, not from the checkpoint's tensor list
        # (#497: geometry from config fields, never from what happens to be
        # present in one file).
        if config.hidden_size != talker_hidden_size:
            self.small_to_mtp_projection = ReplicatedLinear(
                talker_hidden_size,
                config.hidden_size,
                bias=True,
                prefix=add_prefix("small_to_mtp_projection", prefix),
            )
        else:
            self.small_to_mtp_projection = None

    def new_cache(self) -> List[List[Optional[torch.Tensor]]]:
        return [[None, None] for _ in self.layers]

    def step(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        group: int,
        cache: List[List[Optional[torch.Tensor]]],
    ) -> torch.Tensor:
        """One residual group. ``group`` selects this step's own head."""
        if self.small_to_mtp_projection is not None:
            hidden_states, _ = self.small_to_mtp_projection(hidden_states)
        for layer, layer_cache in zip(self.layers, cache):
            hidden_states = layer(positions, hidden_states, layer_cache)
        hidden_states = self.norm(hidden_states)
        logits, _ = self.lm_head[group](hidden_states)
        return logits


class Qwen3TTSForConditionalGeneration(nn.Module):
    """Lane entry point. The class name equals the checkpoint's
    ``architectures`` string verbatim, which is what ``models/registry.py``
    keys on."""

    def __init__(
        self,
        config: Qwen3TTSConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        talker = config.talker_config
        self.talker_config = talker
        self.quant_config = quant_config
        self.tp_size = get_tensor_model_parallel_world_size()

        self.model = Qwen3TTSTalkerTrunk(
            talker, quant_config, prefix=add_prefix("talker.model", prefix)
        )
        self.text_projection = Qwen3TTSTextProjection(
            talker.text_hidden_size,
            talker.text_hidden_size,
            talker.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("talker.text_projection", prefix),
        )
        self.codec_head = ParallelLMHead(
            talker.vocab_size,
            talker.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("talker.codec_head", prefix),
        )
        self.code_predictor = Qwen3TTSCodePredictor(
            talker.code_predictor_config,
            talker.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("talker.code_predictor", prefix),
        )

    # -- weight loading -----------------------------------------------------

    def _stacked_mapping(self) -> List[Tuple[str, str, str]]:
        # (destination fragment, source fragment, shard id)
        return [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

    def _translate(self, name: str) -> Optional[str]:
        """Checkpoint name -> module path, or ``None`` when not lane-owned."""
        if name.startswith(_NON_LANE_PREFIXES):
            return None
        if name.startswith("talker.model."):
            return "model." + name[len("talker.model.") :]
        if name.startswith("talker.text_projection."):
            return "text_projection." + name[len("talker.text_projection.") :]
        if name.startswith("talker.codec_head."):
            return "codec_head." + name[len("talker.codec_head.") :]
        if name.startswith("talker.code_predictor.model.layers."):
            return "code_predictor.layers." + name[
                len("talker.code_predictor.model.layers.") :
            ]
        if name.startswith("talker.code_predictor.model.codec_embedding."):
            return "code_predictor.codec_embedding." + name[
                len("talker.code_predictor.model.codec_embedding.") :
            ]
        if name.startswith("talker.code_predictor.model.norm."):
            return "code_predictor.norm." + name[
                len("talker.code_predictor.model.norm.") :
            ]
        if name.startswith("talker.code_predictor.lm_head."):
            return "code_predictor.lm_head." + name[
                len("talker.code_predictor.lm_head.") :
            ]
        if name.startswith("talker.code_predictor.small_to_mtp_projection."):
            return "code_predictor.small_to_mtp_projection." + name[
                len("talker.code_predictor.small_to_mtp_projection.") :
            ]
        return name

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> set:
        """Load, and REFUSE on any checkpoint name that is neither loaded nor
        explicitly accounted for.

        The reference loader reported ``Loading weights: 478/478`` while
        loading none of them (CLAUDE.md, "SUCCESS CLAIMS ARE NOT EVIDENCE").
        The defence here is structural: every incoming name must resolve to a
        parameter, to a listed non-lane prefix, or the call raises with the
        name. Returns the set of loaded module paths, so a caller can compare
        it against the model's own ``named_parameters``.
        """
        params_dict = dict(self.named_parameters())
        loaded: set = set()
        skipped_non_lane: List[str] = []
        unmatched: List[str] = []

        for name, loaded_weight in weights:
            target = self._translate(name)
            if target is None:
                skipped_non_lane.append(name)
                continue

            for dst_frag, src_frag, shard_id in self._stacked_mapping():
                if src_frag not in target:
                    continue
                stacked = target.replace(src_frag, dst_frag)
                if stacked not in params_dict:
                    continue
                param = params_dict[stacked]
                param.weight_loader(param, loaded_weight, shard_id)
                loaded.add(stacked)
                break
            else:
                if target not in params_dict:
                    unmatched.append(f"{name} -> {target}")
                    continue
                param = params_dict[target]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded.add(target)

        if unmatched:
            raise ValueError(
                "Qwen3-TTS checkpoint carries weights this lane has no home "
                f"for: {unmatched[:8]}{' ...' if len(unmatched) > 8 else ''}. "
                "Refusing rather than dropping them -- a silently unloaded "
                "tensor in a TTS model produces fluent, wrong-sounding audio, "
                "not a crash."
            )
        if skipped_non_lane:
            logger.info(
                "Qwen3-TTS: %d weights belong to in-process modules (%s) and "
                "are not lane-owned; they load with the #286-ledgered speaker "
                "encoder / codec, not here.",
                len(skipped_non_lane),
                ", ".join(_NON_LANE_PREFIXES),
            )
        return loaded

    # -- forward ------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            raise ValueError(
                "the Qwen3-TTS talker is driven by embeddings, never by token "
                "ids: its next input is a SUM of 16 codebook embeddings plus "
                "the text step. A caller that reaches here with input_embeds "
                "cleared has hit the schedule_batch decode-entry unblock "
                "(DESIGN_466 §11.2) rather than a model bug."
            )
        return self.model(positions, forward_batch, input_embeds)


EntryClass = Qwen3TTSForConditionalGeneration
