from typing import Any, Dict, Optional

import torch
from transformers import PretrainedConfig

from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaModel,
)
from sglang.srt.utils import add_prefix, make_layers


def _get_llama_4_attn_scale(
    positions_ids: torch.Tensor, beta: float, max_position_embeddings: int
) -> torch.Tensor:
    scaling = 1 + beta * torch.log(
        1 + torch.floor(positions_ids / max_position_embeddings)
    )
    return scaling.unsqueeze(-1)


class Ministral3Attention(LlamaAttention):
    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        start_layer: int = 0,
        rope_theta: float = 1000000.0,
        rope_scaling: Optional[Dict[str, Any]] = {},
        rope_is_neox_style: bool = True,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        bias: bool = False,
    ) -> None:
        super().__init__(
            config=config,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            layer_id=layer_id,
            start_layer=start_layer,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            rope_is_neox_style=rope_is_neox_style,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            prefix=prefix,
            bias=bias,
        )
        # Ministral3 specific: llama 4 style scaling beta
        self.llama_4_scaling_beta = config.rope_parameters.get("llama_4_scaling_beta")

        # #378: the sliding window, actually applied.
        #
        # This used to read config.sliding_window and then `pass`, on the
        # stated assumption that "RadixAttention handles this mostly via logic
        # in forward/flashinfer". It does not, and THREE gates missed at once
        # (each verified by execution, not by reading):
        #   * LlamaAttention built RadixAttention without a window, so it took
        #     the -1 default, which means NO window;
        #   * Ministral3ForCausalLM had no get_attention_sliding_window_size,
        #     so ModelRunner's first branch missed;
        #   * "Ministral3ForCausalLM" is not in is_hybrid_swa_model's arch
        #     set, so ModelRunner's second branch missed too.
        # The result was FULL attention on a model whose config declares a
        # window: correct up to the window length and silently wrong past it,
        # which is exactly what a short smoke test cannot see.
        #
        # `- 1` matches Gemma4's get_attention_sliding_window_size: the
        # backends take the maximum ATTENDED DISTANCE, not the window width.
        self.sliding_window = getattr(config, "sliding_window", None)
        if self.sliding_window is not None:
            self.attn.sliding_window_size = int(self.sliding_window) - 1

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # Apply RoPE
        q, k = self.rotary_emb(positions, q, k)

        # Ministral3 / Llama 4 scaling
        if self.llama_4_scaling_beta is not None:
            scale = _get_llama_4_attn_scale(
                positions, self.llama_4_scaling_beta, self.max_position_embeddings
            ).to(q.dtype)
            # q shape is [batch_size * seq_len, num_heads * head_dim] or [batch_size * seq_len, num_heads, head_dim]
            # positions is [batch_size * seq_len]
            # scale is [batch_size * seq_len, 1]
            # We need to reshape q to apply scale correctly if it's flattened
            # Assuming q is (total_tokens, num_heads * head_dim)
            q = q.view(-1, self.num_heads, self.head_dim)
            q = q * scale.unsqueeze(1)  # Broadcast over heads
            q = q.view(-1, self.num_heads * self.head_dim)

        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output


class Ministral3DecoderLayer(LlamaDecoderLayer):
    def __init__(
        self,
        config,
        layer_id=0,
        start_layer=0,
        quant_config=None,
        prefix="",
    ):
        super().__init__(
            config=config,
            layer_id=layer_id,
            start_layer=start_layer,
            quant_config=quant_config,
            prefix=prefix,
        )
        self.self_attn = Ministral3Attention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            start_layer=start_layer,
            rope_theta=config.rope_parameters["rope_theta"],
            rope_scaling=config.rope_parameters,  # rope_scaling is rope_parameters in Ministral3Config
            max_position_embeddings=getattr(
                config, "original_max_position_embeddings", 16384
            ),
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
            bias=getattr(config, "attention_bias", False)
            or getattr(config, "bias", False),
        )


class Ministral3Model(LlamaModel):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        # Override layer creation to use Ministral3Attention
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)

        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: Ministral3DecoderLayer(
                config=config,
                quant_config=quant_config,
                layer_id=idx,
                start_layer=self.start_layer,
                prefix=prefix,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix="model.layers",
        )


class Ministral3ForCausalLM(LlamaForCausalLM):
    def _init_model(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        return Ministral3Model(config=config, quant_config=quant_config, prefix=prefix)

    def get_attention_sliding_window_size(self):
        """#378: the hook ``ModelRunner`` asks for FIRST when sizing the KV
        path (``model_runner.py:2138``), the same one Gemma4 provides.

        Without it the runner fell through to the ``is_hybrid_swa`` branch,
        which is False for this architecture, and ended with
        ``sliding_window_size = None`` -- so nothing downstream of attention
        knew about the window either.

        Returns ``None`` when the config declares no window, which leaves the
        runner exactly where it was for such a checkpoint.
        """
        window = getattr(self.config, "sliding_window", None)
        if window is None:
            return None
        return int(window) - 1


EntryClass = [Ministral3ForCausalLM]
