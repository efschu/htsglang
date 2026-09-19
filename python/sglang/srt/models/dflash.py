# Adapted from the DFlash reference implementation (HF) but implemented with
# SGLang primitives (RadixAttention + SGLang KV cache). This model intentionally
# does not include token embeddings or an LM head; DFlash uses the target model's
# embedding/lm_head.

from __future__ import annotations

import logging
import math
from typing import Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from sglang.kernels.ops.speculative.dflash import selector_walk_triton
from sglang.srt.configs.laguna import normalize_gating
from sglang.srt.distributed.utils import tp_partition_size, tp_plan_active
from sglang.srt.distributed.communication_op import tensor_model_parallel_all_gather
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import (
    LogitsProcessorOutput,
    should_apply_lm_head_quant_method,
)
from sglang.srt.layers.radix_attention import AttentionType, RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.utils import apply_qk_norm
from sglang.srt.runtime_context import get_parallel
from sglang.srt.speculative.dflash_utils import (
    can_dflash_slice_qkv_weight,
    get_dflash_attention_sliding_window_size,
    get_dflash_layer_types,
    is_dense_head_weight,
    parse_dflash_draft_config,
)
from sglang.srt.utils import add_prefix, is_npu
from sglang.srt.utils import is_npu
from sglang.srt.utils.common import get_compiler_backend
from sglang.srt.utils.hf_transformers_utils import get_rope_config

_is_npu = is_npu()
if _is_npu:
    from sgl_kernel_npu.norm.split_qkv_rmsnorm_rope import split_qkv_rmsnorm_rope
logger = logging.getLogger(__name__)

# Rows a DFLASH drafter proposes per round when the checkpoint does not say.
# Named because it is not only this model's business: it is the row count the
# fp8 fused-GEMV gate has to admit, and two files agreeing on 16 by accident is
# how the drafter ends up silently on the materialise+GEMM fallback (#274
# round 7c).
DEFAULT_DFLASH_BLOCK_SIZE = 16
try:
    from flashinfer import top_k as _flashinfer_top_k
except ImportError:
    _flashinfer_top_k = None


def gather_candidate_topk(
    lm_head, hidden: torch.Tensor, k: int, *, use_quant_head: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The TP half of :meth:`DFlashDraftModel.compute_candidates`: this rank's
    lm_head-shard top-k, the two all-gathers (K logits/ids per rank, not the
    vocab), the global top-k -> ``(global_ids [N, K] long, top_vals [N, K])``.
    EVERY TP rank must call it -- under ``--speculative-draft-placement solo``
    the shadows (no draft model, a meta selector) join with the host's
    broadcast hidden rows via :func:`shadow_join_candidate_gather`."""
    shard = lm_head.shard_indices
    vals, ids = _radix_topk(
        _project_candidate_logits(
            hidden,
            lm_head,
            num_org=int(shard.num_org_elements),
            use_quant_head=use_quant_head,
        ),
        k,
    )
    global_ids = ids.long() + int(shard.org_vocab_start_index)
    gathered_vals = tensor_model_parallel_all_gather(vals.float(), dim=-1)
    gathered_ids = tensor_model_parallel_all_gather(global_ids, dim=-1)
    top_vals, sel = torch.topk(gathered_vals, k, dim=-1)
    return torch.gather(gathered_ids, -1, sel).long(), top_vals


def shadow_join_candidate_gather(lm_head, hidden: torch.Tensor, k: int) -> torch.Tensor:
    """A solo SHADOW's side of the host's ``compute_candidates``: the same
    shard projection and all-gathers, the global candidate ids as result
    (identical on every rank; the lattice and the sample stay on the host)."""
    quant_method = getattr(lm_head, "quant_method", None)
    use_quant_head = should_apply_lm_head_quant_method(lm_head, quant_method)
    ids, _vals = gather_candidate_topk(lm_head, hidden, k, use_quant_head=use_quant_head)
    return ids


def _radix_topk(scores: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    # The selector's largest single cost: it reads the whole logits tensor.
    if _flashinfer_top_k is not None:
        return _flashinfer_top_k(scores, k, sorted=True, deterministic=True)
    return torch.topk(scores, k, dim=-1)


def _project_candidate_logits(
    hidden: torch.Tensor, lm_head: nn.Module, *, num_org: int, use_quant_head: bool
) -> torch.Tensor:
    """Project draft hiddens through the target head, restricted to the org vocab."""
    if not use_quant_head:
        weight = lm_head.weight
        return torch.matmul(hidden.to(weight.dtype), weight[:num_org].T)
    # A packed weight can't be row-sliced to the org vocab like the dense path,
    # and flashinfer's radix top-k rejects the crop view (non-contiguous), so
    # mask the padded tail out of the top-k instead.
    logits = lm_head.quant_method.apply(lm_head, hidden, None).contiguous()
    if logits.shape[-1] > num_org:
        logits[:, num_org:] = float("-inf")
    return logits


def _get_dflash_attention_type(config, *, default: AttentionType) -> AttentionType:
    """Honor explicit causality while preserving legacy layer defaults."""
    _get = getattr(config, "get_text_config", None)
    text_config = _get() if callable(_get) else config  # test stubs carry no wrapper
    is_causal = getattr(text_config, "is_causal", None)
    if is_causal is None:
        return default
    return AttentionType.DECODER if is_causal else AttentionType.ENCODER_ONLY


def _get_dflash_layer_attention_params(
    config, layer_id: int
) -> Tuple[int, AttentionType]:
    layer_types = get_dflash_layer_types(config)
    if layer_types is None:
        return -1, AttentionType.ENCODER_ONLY
    if layer_id >= len(layer_types):
        raise ValueError(
            "DFLASH config.layer_types must contain one entry per draft layer. "
            f"Got {len(layer_types)} entries, layer_id={layer_id}."
        )

    layer_type = layer_types[layer_id]
    # #1486 (upstream 0dab252ffc / #34524, on this line only the helper had
    # survived the pick, unused): the checkpoint's ``is_causal`` decides.  The
    # DFlash2 draft (lued W8, 5x sliding_attention) says is_causal=false --
    # a block-diffusion draft attends BIDIRECTIONALLY over its mask block;
    # run causal, the late slots of the 8-block are blind and the accept
    # length stalls at ~2.3 (df2l4-l10) against upstream's 4.29 on the 5090.
    if layer_type == "full_attention":
        return -1, _get_dflash_attention_type(
            config, default=AttentionType.ENCODER_ONLY
        )
    if layer_type == "sliding_attention":
        sliding_window_size = get_dflash_attention_sliding_window_size(config)
        assert sliding_window_size is not None
        return sliding_window_size, _get_dflash_attention_type(
            config, default=AttentionType.DECODER
        )
    raise ValueError(
        "Unsupported DFLASH draft layer type. "
        f"layer_types[{layer_id}]={layer_type!r}."
    )


class DFlashAttention(nn.Module):
    def __init__(self, config, layer_id: int, quant_config=None) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        tp_size = int(get_parallel().tp_size)
        tp_rank = int(get_parallel().tp_rank)
        total_num_heads = int(config.num_attention_heads)
        total_num_kv_heads = int(
            getattr(config, "num_key_value_heads", total_num_heads)
        )
        head_dim = int(getattr(config, "head_dim", hidden_size // total_num_heads))

        self.hidden_size = hidden_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        if tp_plan_active(tp_size):
            # Uneven TP (--rank-tp-ratio): heads follow the shard plan with
            # kv heads as the indivisible units (whole GQA groups per rank),
            # exactly the split QKVParallelLinear performs internally — the
            # even ``total // tp_size`` below would silently disagree with
            # the shapes qkv_proj actually allocates. Mirrors llama.py /
            # qwen3_next.py; the per-rank counts also reach the attention
            # backend through RadixAttention, and flashinfer derives the
            # same numbers itself (see _local_attn_head_counts, f7ff51435).
            if self.total_num_kv_heads < tp_size:
                # REPLICATED-KV geometry (kv < tp) is not wired up for the
                # DFLASH draft: fail fast instead of building shapes that
                # disagree with QKVParallelLinear's replicated-kv branch.
                raise ValueError(
                    "DFLASH draft under uneven TP (--rank-tp-ratio) requires "
                    f"total_num_kv_heads >= tp_size; got "
                    f"total_num_kv_heads={self.total_num_kv_heads}, "
                    f"tp_size={tp_size}."
                )
            self.num_heads = tp_partition_size(
                self.total_num_heads, tp_size, tp_rank, self.total_num_kv_heads
            )
            self.num_kv_heads = tp_partition_size(
                self.total_num_kv_heads, tp_size, tp_rank, self.total_num_kv_heads
            )
        else:
            assert self.total_num_heads % tp_size == 0, (
                f"DFlashAttention requires total_num_heads divisible by tp_size. "
                f"total_num_heads={self.total_num_heads}, tp_size={tp_size}."
            )
            self.num_heads = self.total_num_heads // tp_size
            if self.total_num_kv_heads >= tp_size:
                assert self.total_num_kv_heads % tp_size == 0, (
                    f"DFlashAttention requires total_num_kv_heads divisible by tp_size when >= tp_size. "
                    f"total_num_kv_heads={self.total_num_kv_heads}, tp_size={tp_size}."
                )
            else:
                assert tp_size % self.total_num_kv_heads == 0, (
                    f"DFlashAttention requires tp_size divisible by total_num_kv_heads when total_num_kv_heads < tp_size. "
                    f"total_num_kv_heads={self.total_num_kv_heads}, tp_size={tp_size}."
                )
            self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim
        self.q_size = self.num_heads * head_dim
        self.kv_size = self.num_kv_heads * head_dim

        attention_bias = bool(getattr(config, "attention_bias", False))
        rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-6))

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=attention_bias,
            quant_config=quant_config,
            prefix="qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * head_dim,
            hidden_size,
            bias=attention_bias,
            quant_config=quant_config,
            prefix="o_proj",
            # Uneven TP: the o_proj input dim is partitioned like the q
            # heads in qkv_proj (kv-head units). Ignored on the default
            # path (no shard plan installed).
            tp_units=self.total_num_kv_heads,
        )

        # Per-head Q/K RMSNorm, matching HF Qwen3.
        self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps)

        rope_theta, rope_scaling = get_rope_config(config)
        rope_is_neox_style = bool(
            getattr(
                config, "rope_is_neox_style", getattr(config, "is_neox_style", True)
            )
        )
        max_position_embeddings = int(getattr(config, "max_position_embeddings", 32768))
        self.rotary_emb = get_rope(
            head_dim,
            rotary_dim=head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=rope_is_neox_style,
        )

        self.scaling = head_dim**-0.5
        self.sliding_window_size, self.attn_type = _get_dflash_layer_attention_params(
            config, layer_id
        )
        self.attn = RadixAttention(
            num_heads=self.num_heads,
            head_dim=head_dim,
            scaling=self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            sliding_window_size=self.sliding_window_size,
            attn_type=self.attn_type,
        )

    def forward_prepare_npu(self, positions, hidden_states):
        qkv, _ = self.qkv_proj(hidden_states)

        if self.attn.layer_id == 0:
            self.rotary_emb.get_cos_sin_with_position(positions)
        q, k, v = split_qkv_rmsnorm_rope(
            qkv,
            self.rotary_emb.position_sin,
            self.rotary_emb.position_cos,
            self.q_size,
            self.kv_size,
            self.head_dim,
            eps=self.q_norm.variance_epsilon,
            q_weight=self.q_norm.weight,
            k_weight=self.k_norm.weight,
            q_bias=getattr(self.q_norm, "bias", None),
            k_bias=getattr(self.k_norm, "bias", None),
        )
        return q, k, v

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        if _is_npu:
            q, k, v = self.forward_prepare_npu(positions, hidden_states)
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            q, k = apply_qk_norm(q, k, self.q_norm, self.k_norm, self.head_dim)
            q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        attn_output = self.apply_attention_output(attn_output, hidden_states)
        output, _ = self.o_proj(attn_output)
        return output

    def apply_attention_output(
        self, attn_output: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        return attn_output

    def kv_proj_only(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project hidden_states to K/V only (skip Q).

        This is used by DFlash to materialize ctx tokens into the draft KV cache:
        we only need K/V for the cached tokens; Q is never consumed.
        """
        # Fast path for unquantized weights: slice the fused QKV weight and run one GEMM.
        can_slice_qkv_weight, _ = can_dflash_slice_qkv_weight(self.qkv_proj)
        if can_slice_qkv_weight:
            kv_slice = slice(self.q_size, self.q_size + 2 * self.kv_size)
            weight = self.qkv_proj.weight[kv_slice]
            bias = (
                self.qkv_proj.bias[kv_slice] if self.qkv_proj.bias is not None else None
            )
            kv = F.linear(hidden_states, weight, bias)
            k, v = kv.split([self.kv_size, self.kv_size], dim=-1)
            return k, v

        # Fallback: compute full QKV and discard Q (keeps compatibility with quantized weights).
        qkv, _ = self.qkv_proj(hidden_states)
        _, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        return k, v

    def apply_k_norm(self, k: torch.Tensor) -> torch.Tensor:
        k_by_head = k.reshape(-1, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        return k_by_head.view_as(k)

    def apply_k_rope(self, positions: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        # Match K shape so RoPE kernel head-count check passes on all backends.
        dummy_q = k.new_empty(k.shape)
        _, k = self.rotary_emb(positions, dummy_q, k)
        return k


class DFlashMLP(nn.Module):
    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        intermediate_size = int(getattr(config, "intermediate_size", 0))
        if intermediate_size <= 0:
            raise ValueError(
                f"Invalid intermediate_size={intermediate_size} for DFlash MLP."
            )

        # Uneven TP (--rank-tp-ratio): the intermediate size is the unit
        # family — gate/up outputs and the down input share one per-rank
        # partition. Units are 16-element groups, not single elements: the
        # jit activation kernel vector-loads up to 16 bf16 elements and
        # rejects per-rank intermediate sizes not divisible by its vector
        # size (same rationale as LlamaMLP, task #100). Without this the
        # DFLASH draft fails at init whenever sum(ratio weights) does not
        # divide intermediate_size. Inert without an installed shard plan.
        _mlp_units = intermediate_size // math.gcd(intermediate_size, 16)
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix="gate_up_proj" if not prefix else f"{prefix}.gate_up_proj",
            tp_units=_mlp_units,
            tp_family="mlp",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix="down_proj" if not prefix else f"{prefix}.down_proj",
            tp_units=_mlp_units,
            tp_family="mlp",
        )
        hidden_act = getattr(config, "hidden_act", "silu")
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported DFlash activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


@torch.compile(dynamic=True, backend=get_compiler_backend(), disable=_is_npu)
def _grouped_conv(hidden_states, delta, base, block_size, num_groups, group_size, taps):
    blocks = hidden_states.unflatten(-1, (num_groups, group_size))
    coefficients = base.view(1, taps, num_groups, group_size) + delta.unsqueeze(-1)
    out = coefficients[:, 0] * blocks
    position = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    if block_size & (block_size - 1) == 0:
        position = position & (block_size - 1)
    else:
        position = position % block_size
    for tap in range(1, taps):
        shifted = F.pad(blocks[:-tap], (0, 0, 0, 0, tap, 0))
        out = out + coefficients[:, tap] * shifted * (position >= tap).view(-1, 1, 1)
    return out.flatten(-2)


class DFlashGroupedConv(nn.Module):
    """Grouped dynamic depthwise K-tap convolution across one DFlash block.

    Each sublayer is wrapped: `prepare` convolves its input and returns the kernel
    for `finish` to convolve its output, both from one projection of the input.
    """

    def __init__(
        self, hidden_size: int, block_size: int, taps: int, group_size: int
    ) -> None:
        super().__init__()
        if hidden_size % group_size:
            raise ValueError(
                f"DFLASH conv_group_size={group_size} must divide "
                f"hidden_size={hidden_size}."
            )
        hidden_size = int(hidden_size)
        self.block_size = int(block_size)
        self.taps = int(taps)
        self.group_size = int(group_size)
        self.num_groups = hidden_size // self.group_size
        # [input/output, tap, channel], the layout training exports.
        base_kernel = torch.zeros(2, self.taps, hidden_size)
        base_kernel[:, 0] = 1.0
        self.base_kernel = nn.Parameter(base_kernel)
        self.kernel_projection = nn.Linear(
            hidden_size, 2 * self.taps * self.num_groups, bias=False
        )

    def _convolve(self, hidden_states, delta, side: int) -> torch.Tensor:
        # Marked here, not inside: by the time the compiled function traces, the dim
        # is symbolic and the group index costs an integer div and mod per element.
        torch._dynamo.mark_static(hidden_states, 1)
        torch._dynamo.mark_static(delta, 1)
        torch._dynamo.mark_static(delta, 2)
        return _grouped_conv(
            hidden_states,
            delta,
            self.base_kernel[side],
            self.block_size,
            self.num_groups,
            self.group_size,
            self.taps,
        )

    def prepare(self, hidden_states: torch.Tensor):
        coefficients = self.kernel_projection(hidden_states).reshape(
            *hidden_states.shape[:-1], 2, self.taps, self.num_groups
        )
        return (
            self._convolve(hidden_states, coefficients[..., 0, :, :], side=0),
            coefficients[..., 1, :, :],
        )

    def finish(self, hidden_states: torch.Tensor, coefficients) -> torch.Tensor:
        return self._convolve(hidden_states, coefficients, side=1)


class DFlashDecoderLayer(nn.Module):
    attention_cls = DFlashAttention

    def __init__(
        self,
        config,
        layer_id: int,
        attention_conv: Optional[DFlashGroupedConv] = None,
        mlp_conv: Optional[DFlashGroupedConv] = None,
        quant_config=None,
    ) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-6))

        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = self.attention_cls(
            config=config, layer_id=layer_id, quant_config=quant_config
        )
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = DFlashMLP(config=config, quant_config=quant_config)

        self.attention_conv = attention_conv
        self.mlp_conv = mlp_conv

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if hidden_states.numel() == 0:
            # Keep return types consistent for upstream callers.
            if residual is None:
                residual = hidden_states
            return hidden_states, residual

        # Pre-norm attention with fused residual+norm when possible (Qwen3-style).
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        attention_kernel = None
        if self.attention_conv is not None:
            hidden_states, attention_kernel = self.attention_conv.prepare(hidden_states)

        attn_out = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )
        if attention_kernel is not None:
            attn_out = self.attention_conv.finish(attn_out, attention_kernel)

        hidden_states, residual = self.post_attention_layernorm(attn_out, residual)

        mlp_kernel = None
        if self.mlp_conv is not None:
            hidden_states, mlp_kernel = self.mlp_conv.prepare(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if mlp_kernel is not None:
            hidden_states = self.mlp_conv.finish(hidden_states, mlp_kernel)
        return hidden_states, residual


class DFlashDraftModel(nn.Module):
    """SGLang DFlash draft model (no embedding / lm_head weights).

    The checkpoint provides:
      - transformer weights for `layers.*`
      - `fc.weight`, `hidden_norm.weight` for projecting target context features
      - `norm.weight` for final normalization
    """

    decoder_layer_cls = DFlashDecoderLayer
    supports_fused_context_kv = True

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        super().__init__()
        self.config = config

        hidden_size = int(config.hidden_size)
        num_layers = int(config.num_hidden_layers)
        rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-6))
        draft_config = self.draft_config = parse_dflash_draft_config(
            draft_hf_config=config
        )
        self.block_size = draft_config.resolve_block_size(default=16)
        self.candidate_selector: Optional[nn.Module] = None

        def grouped_conv():
            if not draft_config.conv_kernel_size:
                return None
            return DFlashGroupedConv(
                hidden_size,
                self.block_size,
                draft_config.conv_kernel_size,
                draft_config.conv_group_size,
            )

        self.layers = nn.ModuleList(
            [
                self.decoder_layer_cls(
                    config=config,
                    layer_id=i,
                    attention_conv=grouped_conv(),
                    mlp_conv=grouped_conv(),
                    quant_config=quant_config,
                )
                for i in range(num_layers)
            ]
        )
        self.norm = RMSNorm(hidden_size, eps=rms_norm_eps)

        # Project per-token target context features:
        # concat(K * hidden_size) -> hidden_size, where K is the number of target-layer
        # feature tensors concatenated per token (not necessarily equal to num_layers).
        draft_config = parse_dflash_draft_config(draft_hf_config=config)
        if draft_config.num_target_layers is not None:
            target_num_layers = int(draft_config.num_target_layers)
        elif draft_config.target_layer_ids is not None:
            target_num_layers = max(draft_config.target_layer_ids) + 1
        else:
            target_num_layers = num_layers
        target_layer_ids = draft_config.resolve_target_layer_ids(
            target_num_layers=target_num_layers, draft_num_layers=num_layers
        )
        num_context_features = len(target_layer_ids)

        self.num_context_features = int(num_context_features)
        # ReplicatedLinear rather than nn.Linear, so ``fc`` can take a PACKED
        # weight. It is by far the largest single tensor in the draft (for
        # Qwen3.6-27B: [25600, 5120], 131 M of the checkpoint's 1.73 G
        # parameters), and every quantised DFLASH artefact in the wild ships it
        # quantised along with the rest -- a bare nn.Linear can only take a
        # dense one, which made the whole checkpoint unloadable rather than
        # just this tensor.
        #
        # REPLICATED and not column/row-parallel on purpose: ``fc`` consumes the
        # concatenated target-layer features and produces the draft's hidden
        # state, which every rank needs in full. It is also the shape that makes
        # solo placement (draft weight-TP=1 on one host rank) the same code
        # path as the split one.
        self.fc = ReplicatedLinear(
            self.num_context_features * hidden_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("fc", prefix),
        )
        self.hidden_norm = RMSNorm(hidden_size, eps=rms_norm_eps)

        self.block_size = draft_config.resolve_block_size(
            default=DEFAULT_DFLASH_BLOCK_SIZE
        )

    def set_block_size(self, block_size: int) -> None:
        """Adopt the block size the worker resolved.

        The convolutions are built from the checkpoint's block_size, which
        --speculative-num-draft-tokens may override; the layout they index
        depends on it, so the resolved value has to reach them.
        """
        self.block_size = int(block_size)
        for layer in self.layers:
            for conv in (layer.attention_conv, layer.mlp_conv):
                if conv is not None:
                    conv.block_size = self.block_size

    def get_attention_sliding_window_size(self) -> Optional[int]:
        return get_dflash_attention_sliding_window_size(self.config)

    def prepare_context_hidden_for_kv(
        self, layer: DFlashDecoderLayer, ctx_hidden: torch.Tensor
    ) -> torch.Tensor:
        return ctx_hidden

    def project_target_hidden(self, target_hidden: torch.Tensor) -> torch.Tensor:
        """Project concatenated target-layer hidden states into draft hidden_size."""
        # ``input_size`` rather than ``in_features``: ReplicatedLinear carries
        # the logical shape under its own name, and it is the only one that
        # survives quantisation (a packed weight has no ``in_features``).
        expected = int(self.fc.input_size)
        if target_hidden.ndim != 2 or int(target_hidden.shape[-1]) != expected:
            raise ValueError(
                "DFLASH target_hidden feature dim mismatch. "
                f"Expected shape [N, {expected}] "
                f"(num_context_features={self.num_context_features}, hidden_size={int(self.config.hidden_size)}), "
                f"but got shape={tuple(target_hidden.shape)}. "
                "This usually means the target model is capturing a different number of layer features than "
                "the draft checkpoint/config expects."
            )
        # ReplicatedLinear returns (output, bias); bias is None here (bias=False).
        projected, _ = self.fc(target_hidden)
        return self.hidden_norm(projected)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        get_embedding: bool = False,
        pp_proxy_tensors=None,
    ) -> LogitsProcessorOutput:
        if input_embeds is None:
            raise ValueError(
                "DFlashDraftModel requires `input_embeds` (use the target embedding)."
            )
        hidden_states = input_embeds
        residual: Optional[torch.Tensor] = None

        for layer in self.layers:
            hidden_states, residual = layer(
                positions, hidden_states, forward_batch, residual
            )

        if hidden_states.numel() != 0:
            if residual is None:
                hidden_states = self.norm(hidden_states)
            else:
                hidden_states, _ = self.norm(hidden_states, residual)

        return LogitsProcessorOutput(
            next_token_logits=None,
            hidden_states=hidden_states,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, weight_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())

        def resolve_param_name(name: str) -> Optional[str]:
            if name in params_dict:
                return name
            if name.startswith("model."):
                stripped_name = name[len("model.") :]
                if stripped_name in params_dict:
                    return stripped_name
            else:
                prefixed_name = f"model.{name}"
                if prefixed_name in params_dict:
                    return prefixed_name
            return None

        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if f".{weight_name}." not in name:
                    continue
                mapped_name = name.replace(weight_name, param_name)
                resolved_name = resolve_param_name(mapped_name)
                if resolved_name is None:
                    continue
                param = params_dict[resolved_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(resolved_name)
                break
            else:
                resolved_name = resolve_param_name(name)
                if resolved_name is None:
                    # Ignore unexpected weights (e.g., HF rotary caches).
                    continue
                param = params_dict[resolved_name]
                # K-mismatch guard, DENSE PATH ONLY. A packed ``fc.qweight``
                # carries block bytes, not the logical [out, in] shape, so
                # there is nothing to compare here; that case is caught on the
                # first forward by project_target_hidden, which checks the same
                # quantity against the same config field.
                if resolved_name.endswith("fc.weight") and tuple(
                    loaded_weight.shape
                ) != tuple(param.shape):
                    raise ValueError(
                        "DFLASH fc.weight shape mismatch. This usually means the draft checkpoint's "
                        "number of context features (K) does not match this config. "
                        f"Expected fc.weight.shape={tuple(param.shape)} "
                        f"(num_context_features={self.num_context_features}, hidden_size={int(self.config.hidden_size)}), "
                        f"but got {tuple(loaded_weight.shape)} for weight '{name}'."
                    )
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(resolved_name)

        # A name that resolves to nothing is skipped above, by design (HF
        # rotary caches). A PARAMETER that nothing ever wrote is a different
        # thing entirely: the drafter then runs on `torch.empty`, proposes
        # near-random tokens, and the only symptom is an accept rate that
        # looks like a bad drafter instead of an unloaded one. #290 lost two
        # GPU windows to exactly that -- a GGUF stream of `*.qweight` names
        # against a dense skeleton dropped all 36 weight matrices and kept
        # only the 22 F32 norms, and the load reported success.
        missing = sorted(set(params_dict) - loaded_params)
        if missing:
            raise ValueError(
                "DFLASH draft checkpoint left "
                f"{len(missing)}/{len(params_dict)} parameters unloaded: "
                f"{missing[:8]}{' ...' if len(missing) > 8 else ''}. "
                "The checkpoint's tensor names do not reach this model's "
                "parameters. A packed (GGUF) checkpoint loaded into a model "
                "built WITHOUT a quantization config is the usual cause: the "
                "stream carries `*.qweight` names and the skeleton only has "
                "dense `*.weight` parameters."
            )


class DFlashLagunaAttention(DFlashAttention):
    """Laguna DFlash attention with the trained Laguna softplus gate."""

    def __init__(self, config, layer_id: int, quant_config=None) -> None:
        super().__init__(config=config, layer_id=layer_id, quant_config=quant_config)
        hidden_size = int(config.hidden_size)
        total_num_heads = self.total_num_heads
        gating = normalize_gating(getattr(config, "gating", True))
        self.gating = gating
        self.gate_per_head = gating == "per-head"
        if self.gating == "disabled":
            self.g_proj = None
        else:
            g_out = (
                total_num_heads
                if self.gate_per_head
                else total_num_heads * self.head_dim
            )
            self.g_proj = ColumnParallelLinear(
                hidden_size,
                g_out,
                bias=False,
                quant_config=quant_config,
                prefix="g_proj",
                # Uneven TP: the gate output is per q head (or per q-head
                # channel), so it must follow the SAME kv-head-unit split
                # as the q block in qkv_proj — otherwise the gate shard
                # would not line up with this rank's attention output.
                # Ignored on the default path (no shard plan installed).
                tp_units=self.total_num_kv_heads,
            )

    def apply_attention_output(
        self, attn_output: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        if self.g_proj is None:
            return attn_output

        gate, _ = self.g_proj(hidden_states)
        gate = F.softplus(gate.float()).to(attn_output.dtype)
        if self.gate_per_head:
            attn_shape = attn_output.shape
            return (
                attn_output.view(*attn_shape[:-1], self.num_heads, self.head_dim)
                * gate.unsqueeze(-1)
            ).view(attn_shape)
        else:
            return attn_output * gate


class DFlashLagunaDecoderLayer(DFlashDecoderLayer):
    attention_cls = DFlashLagunaAttention


class DFlashLagunaForCausalLM(DFlashDraftModel):
    """Laguna DFlash draft model matching the exported Speculators checkpoint."""

    decoder_layer_cls = DFlashLagunaDecoderLayer
    supports_fused_context_kv = False

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)
        rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-6))
        hidden_size = int(config.hidden_size)
        self.aux_hidden_norms = nn.ModuleList(
            [
                RMSNorm(hidden_size, eps=rms_norm_eps)
                for _ in range(self.num_context_features)
            ]
        )

    def prepare_context_hidden_for_kv(
        self, layer: DFlashLagunaDecoderLayer, ctx_hidden: torch.Tensor
    ) -> torch.Tensor:
        return layer.input_layernorm(ctx_hidden)

    def project_target_hidden(self, target_hidden: torch.Tensor) -> torch.Tensor:
        expected = int(self.fc.in_features)
        if target_hidden.ndim != 2 or int(target_hidden.shape[-1]) != expected:
            raise ValueError(
                "Laguna DFLASH target_hidden feature dim mismatch. "
                f"Expected shape [N, {expected}] "
                f"(num_context_features={self.num_context_features}, hidden_size={int(self.config.hidden_size)}), "
                f"but got shape={tuple(target_hidden.shape)}."
            )

        num_slices = int(self.num_context_features)
        slice_size = int(target_hidden.shape[-1]) // num_slices
        slices = target_hidden.view(target_hidden.shape[0], num_slices, slice_size)
        compute_dtype = self.fc.weight.dtype
        if slices.dtype != compute_dtype:
            slices = slices.to(compute_dtype)
        normed = torch.empty_like(slices)
        for i, norm in enumerate(self.aux_hidden_norms):
            normed[:, i, :] = norm(slices[:, i, :])
        fused = normed.reshape(target_hidden.shape[0], -1)
        return self.hidden_norm(self.fc(fused))


@torch.compile(dynamic=True, backend=get_compiler_backend(), disable=_is_npu)
def _score_edges(
    *,
    predecessor_table: torch.Tensor,
    successor_table: torch.Tensor,
    candidate_ids: torch.Tensor,
    unary_logits: torch.Tensor,
    hidden: torch.Tensor,
    anchor_token_ids: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    keys = successor_table[candidate_ids]
    # Concatenate the ids and look them up once. Concatenating the looked-up rows
    # instead moves a [b, slots, k, rank] float tensor where this moves one id per
    # candidate, and it costs a second gather for the anchor.
    predecessor_ids = torch.cat(
        [anchor_token_ids[:, None, None].expand(-1, 1, top_k), candidate_ids[:, :-1]],
        dim=1,
    )
    predecessors = predecessor_table[predecessor_ids]
    return unary_logits[:, :, None] + torch.einsum(
        "blpr,blcr->blpc", predecessors * hidden[:, :, None], keys
    )


@torch.compile(dynamic=True, backend=get_compiler_backend(), disable=_is_npu)
def _follow_maps(maps, initial_indices, edges: int):
    index = initial_indices
    path = [index]
    for edge in range(edges):
        index = maps[:, edge].gather(-1, index[:, None])[:, 0]
        path.append(index)
    return torch.stack(path, dim=1)


class CandidateSelector(nn.Module):
    """Scores the K x K transitions between adjacent proposal slots, then walks them.

    The [vocab, r] tables are replicated on every TP rank rather than sharded like
    the LM head: candidate ids are gathered globally, so any rank can need any row.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        vocab_size: int,
        state_rank: int,
        top_k: int,
    ) -> None:
        super().__init__()
        if _flashinfer_top_k is None:
            logger.warning(
                "flashinfer is unavailable; the DFlash2 selector falls back to "
                "torch.topk, which roughly halves end-to-end throughput on a large "
                "vocabulary."
            )
        state_rank = int(state_rank)
        self.top_k = int(top_k)
        self.predecessor_codebook = nn.Parameter(
            torch.zeros(int(vocab_size), state_rank), requires_grad=False
        )
        self.successor_codebook = nn.Parameter(
            torch.zeros(int(vocab_size), state_rank), requires_grad=False
        )
        self.hidden_projection = nn.Linear(hidden_size, state_rank, bias=False)

    def build_lattice(
        self,
        *,
        candidate_ids: torch.Tensor,
        unary_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        anchor_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """score[b,e,p,c] = unary[b,e,c] + <A[pred[b,e,p]] * project(h[b,e]), B[c]>

        pred is cand[b,e-1], and the verified anchor for slot 0.
        """
        # Everything but the batch is a model constant. Left symbolic, inductor
        # recovers indices with an integer division per element instead of folding.
        hidden = self.hidden_projection(hidden_states)
        for tensor in (candidate_ids, unary_logits, hidden):
            torch._dynamo.mark_static(tensor, 1)
            torch._dynamo.mark_static(tensor, 2)
        return _score_edges(
            predecessor_table=self.predecessor_codebook,
            successor_table=self.successor_codebook,
            candidate_ids=candidate_ids,
            unary_logits=unary_logits,
            hidden=hidden,
            anchor_token_ids=anchor_token_ids,
            top_k=self.top_k,
        )

    def sample_path(
        self,
        *,
        candidate_ids: torch.Tensor,
        scores: torch.Tensor,
        uniforms: torch.Tensor,
        temperatures: torch.Tensor,
        greedy_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Walk one path, with q over the K candidates for the verify. greedy_mask
        rows take the argmax, selected rather than branched, so one captured graph
        serves greedy and sampling batches alike."""
        if scores.is_cuda:
            return selector_walk_triton(
                candidate_ids=candidate_ids,
                scores=scores,
                uniforms=uniforms,
                temperatures=temperatures,
                greedy_mask=greedy_mask,
            )
        top_k = self.top_k
        temps = temperatures.view(-1, 1)
        initial_probs = torch.softmax(scores[:, 0, 0].float() / temps, dim=-1)
        initial_indices = (
            uniforms[:, :1]
            .ge(initial_probs.cumsum(dim=-1))
            .sum(dim=-1)
            .clamp_max(top_k - 1)
        )
        transition_probs = torch.softmax(
            scores[:, 1:].float() / temps[:, :, None, None], dim=-1
        )
        local_maps = (
            uniforms[:, 1:, None, None]
            .ge(transition_probs.cumsum(dim=-1))
            .sum(dim=-1)
            .clamp_max(top_k - 1)
        )
        initial_indices = torch.where(
            greedy_mask, scores[:, 0, 0].argmax(dim=-1), initial_indices
        )
        local_maps = torch.where(
            greedy_mask[:, None, None], scores[:, 1:].argmax(dim=-1), local_maps
        )
        torch._dynamo.mark_static(local_maps, 1)
        torch._dynamo.mark_static(local_maps, 2)
        path_indices = _follow_maps(
            local_maps, initial_indices, int(scores.shape[1]) - 1
        )
        tokens = candidate_ids.gather(-1, path_indices.unsqueeze(-1))[:, :, 0]
        realized_rows = transition_probs.gather(
            2, path_indices[:, :-1, None, None].expand(-1, -1, 1, top_k)
        )[:, :, 0]
        q_rows = torch.cat((initial_probs.unsqueeze(1), realized_rows), dim=1)
        # Greedy rows walk the argmax, so their q is the point mass there, not
        # the temperature-1 softmax above. The triton walk stores the same.
        q_rows = torch.where(
            greedy_mask[:, None, None], F.one_hot(path_indices, top_k).float(), q_rows
        )
        return tokens, q_rows


class DFlash2DraftModel(DFlashDraftModel):
    """DFlash backbone + candidate selector. Reuses the DFLASH speculative worker."""

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)
        draft_config = self.draft_config
        if not draft_config.selector_rank:
            raise ValueError(
                "DFlash selector draft requires dflash_config.selector_rank."
            )
        self.candidate_selector = CandidateSelector(
            hidden_size=int(config.hidden_size),
            vocab_size=int(config.vocab_size),
            state_rank=draft_config.selector_rank,
            top_k=draft_config.selector_top_k,
        )
        # The draft has no head of its own; the worker points this at the target's
        # before capture.
        self.lm_head: Optional[nn.Module] = None

    def _transform_unary_logits(self, logits: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        if self.draft_config.output_multiplier != 1.0:
            logits.mul_(self.draft_config.output_multiplier)
        softcap = self.draft_config.final_logit_softcapping
        if softcap is not None:
            logits.div_(softcap).tanh_().mul_(softcap)
        return logits

    def compute_candidates(
        self, hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Top-k base candidates via the target lm_head: hidden [N, H] -> global
        candidate_ids / unary_logits [N, K]. Under TP (vocab-sharded lm_head): local top-k
        per shard, all-gather K logits/ids (not the full vocab), then a global top-k --
        identical candidates at O(tp*K) instead of O(vocab) gather bandwidth."""
        assert self.lm_head is not None, "draft_model.lm_head unset before capture"
        k = self.candidate_selector.top_k
        # The worker screens the head before capture, but its eager fallback
        # (_propose_selector_block) attaches whatever the target has.
        weight = getattr(self.lm_head, "weight", None)
        quant_method = getattr(self.lm_head, "quant_method", None)
        use_quant_head = should_apply_lm_head_quant_method(self.lm_head, quant_method)
        if not use_quant_head and not is_dense_head_weight(weight):
            raise RuntimeError(
                "DFlash2 selector requires a dense FP16/BF16/FP32 target lm_head "
                "or a supported lm_head.quant_method."
            )
        if get_parallel().tp_size == 1:
            org = int(self.lm_head.org_vocab_size)
            vals, ids = _radix_topk(
                _project_candidate_logits(
                    hidden, self.lm_head, num_org=org, use_quant_head=use_quant_head
                ),
                k,
            )
            return ids.long(), self._transform_unary_logits(vals)
        ids, top_vals = gather_candidate_topk(
            self.lm_head, hidden, k, use_quant_head=use_quant_head
        )
        return ids, self._transform_unary_logits(top_vals)


class MuseGlimmerAssistantModel(DFlashDraftModel):
    """Alias for checkpoints declaring architectures=["MuseGlimmerAssistantModel"]."""


EntryClass = [
    DFlashDraftModel,
    DFlashLagunaForCausalLM,
    MuseGlimmerAssistantModel,
    DFlash2DraftModel,
]
