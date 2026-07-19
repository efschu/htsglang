from __future__ import annotations

from sglang.srt.runtime_context import get_parallel

"""
Support different attention backends.
Now there are two backends: FlashInfer and Triton.
FlashInfer is faster and Triton is easier to customize.
Each backend supports two operators: extend (i.e. prefill with cached prefix) and decode.
"""

import logging
import os
import weakref
from dataclasses import dataclass
from enum import Enum, auto
from functools import partial
from typing import TYPE_CHECKING, Callable, List, Optional, Union

import torch

from sglang.kernel_api_logging import debug_kernel_api
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.environ import envs
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.distributed.utils import tp_partition_size
from sglang.srt.layers.attention.utils import (
    assert_buffer_fits,
    create_flashinfer_kv_indices_triton,
)
from sglang.srt.layers.dcp import (
    cp_all_gather_heads_uneven,
    cp_lse_ag_out_ar_mha_uneven,
    create_triton_kv_indices_for_dcp_triton,
    get_dcp_lens,
)
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    Phase,
    check_cuda_graph_backend,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    is_in_tc_piecewise_cuda_graph,
)
from sglang.srt.runtime_context import get_buffer
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType
from sglang.srt.speculative.spec_utils import (
    draft_kv_indices_buffer_width,
    draft_kv_indices_used_len,
    generate_draft_decode_kv_indices,
)
from sglang.srt.utils import (
    get_int_env_var,
    is_flashinfer_available,
    is_sm100_supported,
    next_power_of_2,
    require_gathered_buffer,
)

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


def _cuda_graph_capture_max_bs(server_args, max_bs: int) -> int:
    """Pad max_bs to the alignment cuda-graph capture uses (see get_batch_sizes_to_capture)."""
    mul_base = 1
    if server_args.enable_two_batch_overlap:
        mul_base *= 2
    if require_gathered_buffer(server_args):
        mul_base *= get_parallel().attn_tp_size
    if mul_base % get_parallel().attn_cp_size != 0:
        mul_base *= get_parallel().attn_cp_size
    return (max_bs + mul_base - 1) // mul_base * mul_base


if envs.SGLANG_ENABLE_TORCH_COMPILE.get():
    torch._logging.set_logs(dynamo=logging.ERROR)
    torch._dynamo.config.suppress_errors = True


if is_flashinfer_available():
    from flashinfer import (
        BatchDecodeWithPagedKVCacheWrapper,
        BatchPrefillWithPagedKVCacheWrapper,
        BatchPrefillWithRaggedKVCacheWrapper,
        fast_decode_plan,
    )
    from flashinfer.cascade import merge_state

    from sglang.kernels.ops.attention.merge_state import merge_state_triton

    # FlashInfer's MergeState CUDA kernel uses blockDim = (head_dim/vec_size, num_heads).
    # When num_heads is large (e.g. with DP attention where attention_tp_size=1), the
    # total threads per block can exceed CUDA's limit of 1024 and the kernel launch fails
    # with `invalid configuration argument`. Fall back to the in-tree Triton implementation,
    # which uses (token, head) as the launch grid and is therefore unaffected.
    _MERGE_STATE_CUDA_MAX_THREADS_PER_BLOCK = 1024

    def _merge_state_max_safe_num_heads(head_dim: int, element_size: int) -> int:
        # Mirrors flashinfer's vec_size selection in include/flashinfer/attention/cascade.cuh.
        vec_size = max(16 // element_size, head_dim // 32)
        bdx = head_dim // vec_size
        if bdx <= 0:
            return _MERGE_STATE_CUDA_MAX_THREADS_PER_BLOCK
        return _MERGE_STATE_CUDA_MAX_THREADS_PER_BLOCK // bdx

    def _safe_merge_state(
        v_a: torch.Tensor,
        s_a: torch.Tensor,
        v_b: torch.Tensor,
        s_b: torch.Tensor,
    ):
        num_heads = v_a.shape[1]
        head_dim = v_a.shape[2]
        max_heads = _merge_state_max_safe_num_heads(head_dim, v_a.element_size())
        if num_heads <= max_heads:
            return merge_state(v_a, s_a, v_b, s_b)
        return merge_state_triton(v_a, s_a, v_b, s_b)


class WrapperDispatch(Enum):
    SLIDING_WINDOW = auto()
    CROSS_ATTENTION = auto()


@dataclass
class MultiItemScoringParams:
    """Parameters for multi-item scoring in attention computation.

    Used when processing sequences with multiple items separated by delimiters,
    where each item needs specific attention patterns that respect item boundaries.

    Attributes:
        prefix_len_ptr: A uint32 1D tensor indicating the prefix length of each prompt.
                       The tensor size is equal to the batch size.
        token_pos_in_items_ptr: A uint16 1D tensor indicating the token position of each item
                               starting from 0 (delimiter) for each item. For batch size > 1,
                               sequences are concatenated with zero padding to ensure same length.
        token_pos_in_items_len: Zero padding length for token_pos_in_items_ptr to handle
                               batch_size > 1 case. Defines the padded length for each sequence.
        max_item_len_ptr: A uint16 tensor containing the max token length of all items
                         for each prompt in the batch.

    """

    prefix_len_ptr: Optional[torch.Tensor] = None
    token_pos_in_items_ptr: Optional[torch.Tensor] = None
    token_pos_in_items_len: int = 0
    max_item_len_ptr: Optional[torch.Tensor] = None

    def is_enabled(self) -> bool:
        """Check if multi-item scoring is enabled."""
        return self.prefix_len_ptr is not None


@dataclass
class DecodeMetadata:
    decode_wrappers: List[BatchDecodeWithPagedKVCacheWrapper]
    # full->SWA translated out_cache_loc (SWA KV-store write target)
    swa_out_cache_loc: Optional[torch.Tensor] = None


@dataclass
class PrefillMetadata:
    prefill_wrappers: List[BatchPrefillWithPagedKVCacheWrapper]
    use_ragged: bool
    extend_no_prefix: bool
    multi_item_params: Optional[MultiItemScoringParams] = None
    swa_out_cache_loc: Optional[torch.Tensor] = None


# Reuse this workspace buffer across all flashinfer wrappers

# Safety margin on the computed split-kv worst case for the dedicated
# full-CG prefill workspace (absorbs allocator alignment and minor
# flashinfer sizing drift across versions). Sizing logic lives in
# FlashInferAttnBackend._full_cg_prefill_workspace_bytes.
FULL_CG_PREFILL_WORKSPACE_MARGIN = 1.25

# Every allocated flashinfer float workspace (the shared global one, private
# init_new_workspace ones, the dedicated full-CG prefill one). Weak refs: the
# registry must not extend buffer lifetimes across re-inits. Keyed by id()
# in a WeakValueDictionary instead of a WeakSet: WeakSet membership falls
# back to elementwise Tensor.__eq__ on hash collisions ("Boolean value of
# Tensor ... is ambiguous").
_WORKSPACE_BUFFERS: "weakref.WeakValueDictionary[int, torch.Tensor]" = (
    weakref.WeakValueDictionary()
)


def register_flashinfer_workspace_buffer(buf: torch.Tensor) -> torch.Tensor:
    """Track a flashinfer float-workspace allocation for the per-request
    zeroing contract (see zero_flashinfer_workspaces)."""
    _WORKSPACE_BUFFERS[id(buf)] = buf
    return buf


def zero_flashinfer_workspaces() -> int:
    """Zero every registered flashinfer float workspace; returns the count.

    Root fix for #50's cross-request nondeterminism: the fa2 split-KV
    kernels' partial/merge scratch lives in these persistent workspaces,
    allocated as torch.empty and shared across every wrapper (target +
    draft + per-step spec backends). The attention path reads workspace
    regions the current forward did not write (proven by the round-11 GPU
    bisection: zeroing exactly _float_workspace_buffer after each request
    flattens the request-ordinal output sequence at the natural run-1
    value; int workspace / kv_lens wipes do not). On a fresh boot those
    regions are first-touch-zero cudaMalloc pages — i.e. the kernels were
    only ever validated against a zeroed workspace. Restoring that boot
    state at request boundaries makes every request see the same workspace
    contract instead of the previous request's residue.

    Called at request-finish (scheduler); a 384 MiB memset is ~0.5 ms per
    finished request — negligible against request latency. Within-request
    residue remains, but it is a deterministic function of the request
    itself, so run-to-run bit-stability is unaffected.
    """
    n = 0
    for buf in list(_WORKSPACE_BUFFERS.values()):
        buf.zero_()
        n += 1
    return n


def _local_attn_head_counts(model_runner: "ModelRunner") -> tuple:
    """This rank's real (num_qo_heads, num_kv_heads) for the flashinfer
    wrappers.

    Under uneven TP (--rank-tp-ratio) the attention heads are split
    UNEVENLY across ranks -- whole GQA groups per rank, kv heads as the
    indivisible units -- exactly as QKVParallelLinear / Qwen3NextAttention
    partition them (see qwen3_next.py: num_heads = tp_partition_size(...,
    units=total_num_kv_heads)). The q/k/v tensors handed to wrapper.run()
    therefore carry THIS rank's uneven head count (layer.tp_q_head_num).

    flashinfer's plan() builds the split-KV work partition (and the merge
    schedule) from the num_qo_heads / num_kv_heads it is told; run() also
    forwards those same planned counts to the kernel. If they are computed
    as a uniform ``num_attention_heads // attn_tp_size`` they diverge from
    the real per-rank q tensor (e.g. 24 heads over ratio [2,1,1]: real
    [12,6,6] vs uniform 8) and, worse, the planned gqa_group_size no longer
    matches the tensors. That mis-sizes the split-KV partition/merge for
    long contexts (split-KV only engages once the KV is long enough),
    silently corrupting attention on the long-context path -- divergently
    per rank -- while short prompts (single tile, no split) stay correct.
    Under deterministic inference (fixed_split_size) the same inconsistency
    makes the fixed schedule unsatisfiable and the prefill kernel hangs.

    num_kv_heads already auto-resolves to this rank via get_num_kv_heads()
    inside a worker; num_qo_heads did not -- both are made explicit here.

    Default path (no ratio plan installed): tp_partition_size degrades to
    the even ``total // tp_size`` and get_num_kv_heads(rank=...) to the even
    share, i.e. byte-identical to the previous behavior.
    """
    from sglang.srt.distributed.utils import (
        attn_q_partition_groups,
        attn_q_partition_units,
    )

    mc = model_runner.model_config
    tp_size = get_parallel().attn_tp_size
    tp_rank = get_parallel().attn_tp_rank
    _total_kv = mc.get_total_num_kv_heads()
    num_qo_heads = tp_partition_size(
        mc.num_attention_heads,
        tp_size,
        tp_rank,
        # Indivisible q units: the kv heads (whole GQA groups) normally;
        # kv_total-sized q-head packets under the REPLICATED-KV geometry
        # (TP > num_kv_heads, task #62), keeping num_qo % num_kv == 0 with
        # num_kv_heads == kv_total on every rank.
        attn_q_partition_units(mc.num_attention_heads, _total_kv, tp_size),
        # kv-boundary alignment (task #116) so this rank's q-head count
        # matches the aligned qkv/o_proj split.
        groups=attn_q_partition_groups(_total_kv, tp_size),
    )
    num_kv_heads = mc.get_num_kv_heads(tp_size, rank=tp_rank)
    return num_qo_heads, num_kv_heads

# Use as a fast path to override the indptr in flashinfer's plan function
# This is used to remove some host-to-device copy overhead.
global_override_indptr_cpu = None


def fast_prefill_plan(
    self,
    qo_indptr: torch.Tensor,
    paged_kv_indptr: torch.Tensor,
    paged_kv_indices: torch.Tensor,
    paged_kv_last_page_len: torch.Tensor,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim_qk: int,
    page_size: int,
    head_dim_vo: Optional[int] = None,
    custom_mask: Optional[torch.Tensor] = None,
    causal: bool = False,
    window_left: int = -1,
    q_data_type: Union[str, torch.dtype] = "float16",
    kv_data_type: Optional[Union[str, torch.dtype]] = None,
    o_data_type: Optional[Union[str, torch.dtype]] = None,
    non_blocking: bool = True,
    fixed_split_size: Optional[int] = None,
    prefix_len_ptr: Optional[torch.Tensor] = None,
    token_pos_in_items_ptr: Optional[torch.Tensor] = None,
    token_pos_in_items_len: int = 0,
    max_item_len_ptr: Optional[torch.Tensor] = None,
    # Required host-known metadata: lets us skip the per-replay device-to-host
    # copies upstream plan() always issues. Keyword-only with no default so a
    # caller that forgets them fails at the call boundary, not with a cryptic
    # None crash deeper in.
    *,
    qo_indptr_host: torch.Tensor,
    kv_indptr_host: torch.Tensor,
    kv_lens_host: torch.Tensor,
    max_q_len: int,
    max_kv_len: int,
) -> None:
    """Sync-free ``BatchPrefillWithPagedKVCacheWrapper.plan`` for the EAGLE
    draft-extend CUDA graph (FlashInfer fa2, cuda-graph mode only).

    Upstream plan() always does qo/paged_kv/last_page_len ``.to("cpu")`` to build
    its host scheduling metadata, a blocking D2H that drains the GPU queue every
    replay. The caller passes host-known qo/kv layout in, so we call the underlying
    ``_cached_module.plan`` directly with no readback; the ``_plan_info`` produced
    is identical to plan()'s.
    """
    assert self.is_cuda_graph_enabled, "fast_prefill_plan is cuda-graph only"
    assert (
        getattr(self, "_backend", None) == "fa2"
    ), "fast_prefill_plan supports the fa2 backend only"
    assert (
        getattr(self, "_cached_module", None) is not None
    ), "fast_prefill_plan requires _cached_module from a prior real plan() (capture)"

    if head_dim_vo is None:
        head_dim_vo = head_dim_qk
    batch_size = len(paged_kv_last_page_len)

    total_num_rows = int(qo_indptr_host[-1])
    self._qo_indptr_last = total_num_rows
    self._max_q_len = max_q_len
    self._max_kv_len = max_kv_len

    if self._max_total_num_rows is None:
        self._max_total_num_rows = total_num_rows

    self._batch_size = batch_size
    self._num_qo_heads = num_qo_heads
    self._num_kv_heads = num_kv_heads
    self._prefix_len_ptr = prefix_len_ptr
    self._token_pos_in_items_ptr = token_pos_in_items_ptr
    self._token_pos_in_items_len = token_pos_in_items_len
    self._max_item_len_ptr = max_item_len_ptr

    # Refresh the cuda-graph input buffers (device-to-device, non-blocking).
    self._qo_indptr_buf.copy_(qo_indptr, non_blocking=non_blocking)
    self._paged_kv_indptr_buf.copy_(paged_kv_indptr, non_blocking=non_blocking)
    self._paged_kv_last_page_len_buf.copy_(
        paged_kv_last_page_len, non_blocking=non_blocking
    )
    self._paged_kv_indices_buf[: len(paged_kv_indices)].copy_(
        paged_kv_indices,
        non_blocking=(paged_kv_indices.device == self.device) and non_blocking,
    )

    self._cached_q_data_type = q_data_type
    self._cached_kv_data_type = (
        kv_data_type if kv_data_type is not None else q_data_type
    )
    self._cached_o_data_type = o_data_type
    self._block_tables = None

    args = [
        self._float_workspace_buffer,
        self._int_workspace_buffer,
        self._pin_memory_int_workspace_buffer,
        qo_indptr_host,
        kv_indptr_host,
        kv_lens_host,
        self._max_total_num_rows or total_num_rows,
        batch_size,
        num_qo_heads,
        num_kv_heads,
        page_size,
        self.is_cuda_graph_enabled,
        head_dim_qk,
        head_dim_vo,
        causal,
        window_left,
        fixed_split_size if fixed_split_size is not None else -1,
        False,  # disable_split_kv
        0,  # num_colocated_ctas
    ]
    self._plan_info = self._cached_module.plan(*args)


class FlashInferAttnBackend(AttentionBackend):
    """Flashinfer attention kernels."""

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        kv_indptr_buf: Optional[torch.Tensor] = None,
        kv_last_page_len_buf: Optional[torch.Tensor] = None,
        init_new_workspace: bool = False,
    ):
        super().__init__()
        self.prefill_backend = "fa2"
        self.decode_backend = "fa2"

        self.req_to_token_pool = model_runner.req_to_token_pool
        self.token_to_kv_pool = model_runner.token_to_kv_pool
        self._swa_kv_pool: Optional[BaseSWAKVPool] = self._resolve_swa_kv_pool(
            model_runner
        )
        self.use_sliding_window_kv_pool = self._swa_kv_pool is not None
        self.enable_mis = model_runner.server_args.enable_mis

        # FIXME: remove dllm workarounds from flashinfer
        self.dllm_config = DllmConfig.from_server_args(model_runner.server_args)
        self.is_dllm_model = self.dllm_config is not None

        # Parse constants. Use THIS rank's real head counts so uneven TP
        # (--rank-tp-ratio) picks the decode kernel path from the actual
        # per-rank q/kv shapes (see _local_attn_head_counts).
        _num_qo_heads, _num_kv_heads = _local_attn_head_counts(model_runner)
        self.decode_use_tensor_cores = should_use_tensor_core(
            kv_cache_dtype=model_runner.kv_cache_dtype,
            num_attention_heads=_num_qo_heads,
            num_kv_heads=_num_kv_heads,
        )

        # Uneven-DCP (--rank-tp-ratio + dcp_size==tp_size): the full-attention
        # KV cache is TOKEN-sharded with the FULL kv-heads replicated on every
        # rank. The paged wrappers must be planned with the FULL gathered head
        # counts (24 q / 4 kv here) and per-rank DCP kv_indices; the attention
        # forward gathers this rank's uneven q/kv shards, runs local paged
        # attention over its owned token slots, and LSE-combines across the DCP
        # group. See _local_attn_head_counts / cp_*_uneven. Off by default
        # (predicate False) -> stock flashinfer path is bit-identical.
        from sglang.srt.distributed.utils import (
            attn_kv_replicated,
            attn_q_partition_groups,
            attn_q_partition_units,
            cp_token_prefix,
            tp_partition_sizes,
            uneven_dcp_active,
            uneven_dcp_kv_replicated,
        )

        self.dcp_size = get_parallel().attn_dcp_size
        self.dcp_rank = get_parallel().attn_dcp_rank
        # M4 (MTP+DCP): the DRAFT worker (EAGLE/NEXTN) does NOT token-shard its
        # tiny 1-layer KV pool BY DEFAULT (--draft-kv-layout replicated). It runs
        # as a plain uneven-TP model -- head-sharded kv (whole GQA groups per
        # rank, [2,1,1]) with the FULL token context resident on every rank -- so
        # its draft/verify KV indices never need the weighted DCP owner rule.
        # Only the TARGET model uses DCP token-sharding.
        #
        # #108 (--draft-kv-layout dcp): opt in to token-sharding the DRAFT KV
        # pool with the SAME weighted owner rule + replicated-kv-heads + LSE-merge
        # as the target. When enabled the draft worker takes the identical
        # uneven_dcp path below (owner-rule masked write, per-layer kv-head
        # all-gather, cross-rank LSE-merge); the draft pool is sized per-rank and
        # its heads replicated in model_runner_kv_cache_mixin. This is a lossless
        # memory optimization: at temp=0 the verified output is byte-identical to
        # 'replicated' (the target verify path is untouched). Gated OFF by
        # default so the replicated path stays bit-identical.
        _is_draft = getattr(model_runner, "is_draft_worker", False)
        _draft_kv_layout = getattr(
            getattr(model_runner, "server_args", None), "draft_kv_layout", "replicated"
        )
        _draft_replicated = _is_draft and _draft_kv_layout != "dcp"
        self.uneven_dcp = (
            uneven_dcp_kv_replicated(self.dcp_size) and not _draft_replicated
        )
        # WEIGHTED owner rule (SGLANG_UNEVEN_DCP_WEIGHTED): a non-uniform token
        # vector is installed, so this rank owns the contiguous virtual-block
        # offset range [cp_lo, cp_hi) of every block of cp_S slots (instead of
        # the even modulo residue == dcp_rank). Physical (compact) slot of an
        # out_cache_loc L owned by this rank is
        #   (L // cp_S) * cp_ratio + (L % cp_S - cp_lo)
        # which is an injective, ratio-proportional packing (reduces EXACTLY to
        # the even L // dcp_size when ratios are all 1). False -> even modulo.
        self.uneven_dcp_weighted = self.uneven_dcp and uneven_dcp_active(
            self.dcp_size
        )
        if self.uneven_dcp_weighted:
            _cp_prefix = cp_token_prefix(self.dcp_size)
            self.cp_S = _cp_prefix[-1]
            self.cp_lo = _cp_prefix[self.dcp_rank]
            self.cp_hi = _cp_prefix[self.dcp_rank + 1]
            self.cp_ratio = self.cp_hi - self.cp_lo
        self.dcp_kv_replicated_heads = False
        # Generic fail-fast for TP > num_kv_heads (task #62): the
        # REPLICATED-KV geometry stays free of KV-cache duplication only
        # because the uneven-DCP owner rule token-shards the pool. A target
        # model that needs it without DCP spanning the TP group would
        # silently duplicate the whole cache per rank — refuse instead.
        # (The NEXTN/EAGLE draft worker is exempt by design: its 1-layer
        # full-context pool is intentionally replicated.)
        if (
            attn_kv_replicated(
                get_parallel().attn_tp_size,
                model_runner.model_config.get_total_num_kv_heads(),
            )
            and not getattr(model_runner, "is_draft_worker", False)
            and not self.uneven_dcp
        ):
            raise ValueError(
                "TP > num_kv_heads requires the uneven-DCP token-sharded KV "
                "pool: enable SGLANG_UNEVEN_DCP=1 (and "
                "SGLANG_UNEVEN_DCP_WEIGHTED=1 for the weighted owner rule) "
                "so dcp_size == tp_size."
            )
        if self.uneven_dcp:
            mc = model_runner.model_config
            attn_tp_size = get_parallel().attn_tp_size
            total_kv = mc.get_total_num_kv_heads()
            # REPLICATED-KV geometry (TP > num_kv_heads, task #62): every
            # rank projects ALL kv heads itself (replicated k/v weights),
            # so the per-layer kv-head all-gather in _dcp_masked_write is
            # a no-op and is skipped; q splits in kv_total-sized units
            # (e.g. A3B 16 q / 2 kv over TP=3 -> [6,6,4]).
            self.dcp_kv_replicated_heads = attn_kv_replicated(
                attn_tp_size, total_kv
            )
            # Per-rank uneven head partitions, e.g. 24 q over [2,1,1] kv
            # units -> q [12,6,6], kv [2,1,1] (normal mode: whole GQA
            # groups per rank).
            self.dcp_q_head_counts = tp_partition_sizes(
                mc.num_attention_heads,
                attn_tp_size,
                units=attn_q_partition_units(
                    mc.num_attention_heads, total_kv, attn_tp_size
                ),
                # kv-boundary alignment (task #116): the per-rank q-head
                # counts (and the #105 straddle guard) follow the aligned
                # split, matching qkv/o_proj.
                groups=attn_q_partition_groups(total_kv, attn_tp_size),
            )
            self.dcp_kv_head_counts = (
                [total_kv] * attn_tp_size
                if self.dcp_kv_replicated_heads
                else tp_partition_sizes(total_kv, attn_tp_size, units=total_kv)
            )
            # FULL gathered counts the paged wrappers are planned with.
            self.dcp_full_qo_heads = mc.num_attention_heads
            self.dcp_full_kv_heads = total_kv

        self.max_context_len = model_runner.model_config.context_len
        self.skip_prefill = skip_prefill
        self.is_multimodal = model_runner.model_config.is_multimodal
        assert not (
            model_runner.sliding_window_size is not None
            and model_runner.model_config.is_encoder_decoder
        ), "Sliding window and cross attention are not supported together"

        if model_runner.sliding_window_size is not None:
            self.num_wrappers = 2
            self.dispatch_reason = WrapperDispatch.SLIDING_WINDOW
        elif model_runner.model_config.is_encoder_decoder:
            self.num_wrappers = 2
            self.dispatch_reason = WrapperDispatch.CROSS_ATTENTION
        else:
            self.num_wrappers = 1
            self.dispatch_reason = None

        # Qwen2/Qwen3 models require higher flashinfer workspace size
        if (
            "Qwen2ForCausalLM" in model_runner.model_config.hf_config.architectures
            or "Qwen3ForCausalLM" in model_runner.model_config.hf_config.architectures
            or "MiMoForCausalLM" in model_runner.model_config.hf_config.architectures
            or "Qwen3VLForConditionalGeneration"
            in model_runner.model_config.hf_config.architectures
            or "Qwen3VLMoeForConditionalGeneration"
            in model_runner.model_config.hf_config.architectures
        ):
            envs.SGLANG_FLASHINFER_WORKSPACE_SIZE.set(512 * 1024 * 1024)

        # When deterministic inference is enabled, tensor cores should be used for decode
        # Also set split tile sizes for prefill and decode from environment variables, and disable kv split for cuda graph
        # More information can be found here: https://github.com/flashinfer-ai/flashinfer/pull/1675
        self.enable_deterministic = (
            model_runner.server_args.enable_deterministic_inference
        )
        self.prefill_split_tile_size = None
        self.decode_split_tile_size = None
        self.disable_cuda_graph_kv_split = False
        if self.enable_deterministic:
            self.decode_use_tensor_cores = True
            self.prefill_split_tile_size = get_int_env_var(
                "SGLANG_FLASHINFER_PREFILL_SPLIT_TILE_SIZE", 4096
            )
            self.decode_split_tile_size = get_int_env_var(
                "SGLANG_FLASHINFER_DECODE_SPLIT_TILE_SIZE", 2048
            )
            self.disable_cuda_graph_kv_split = True
            envs.SGLANG_FLASHINFER_WORKSPACE_SIZE.set(2048 * 1024 * 1024)

        self.use_paged = envs.SGLANG_FLASHINFER_USE_PAGED.get()

        # Allocate buffers
        # different from flashinfer zero_init_global_workspace_buffer.
        # NOTE(#50): torch.empty is fine at boot ONLY because fresh cudaMalloc
        # pages read as zeros (first touch); the split-KV kernels read
        # workspace regions the current forward did not write, so the zeroed
        # state is the contract they were validated against. It is restored
        # at request boundaries via zero_flashinfer_workspaces() — every
        # workspace allocation below must be registered.
        global_workspace_buffer = register_flashinfer_workspace_buffer(
            get_buffer(
                "flashinfer_workspace",
                lambda: torch.empty(
                    envs.SGLANG_FLASHINFER_WORKSPACE_SIZE.get(),
                    dtype=torch.uint8,
                    device=model_runner.device,
                ),
            )
        )
        if init_new_workspace:
            self.workspace_buffer = register_flashinfer_workspace_buffer(
                torch.empty(
                    envs.SGLANG_FLASHINFER_WORKSPACE_SIZE.get(),
                    dtype=torch.uint8,
                    device=model_runner.device,
                )
            )
        else:
            self.workspace_buffer = global_workspace_buffer
        max_bs = _cuda_graph_capture_max_bs(
            model_runner.server_args, model_runner.req_to_token_pool.size
        )
        if kv_indptr_buf is None:
            self.kv_indptr = [
                torch.zeros(
                    (max_bs + 1,), dtype=torch.int32, device=model_runner.device
                )
                for _ in range(self.num_wrappers)
            ]
        else:
            assert self.num_wrappers == 1
            self.kv_indptr = [kv_indptr_buf]

        if kv_last_page_len_buf is None:
            self.kv_last_page_len = torch.ones(
                (max_bs,), dtype=torch.int32, device=model_runner.device
            )
        else:
            assert self.num_wrappers == 1
            self.kv_last_page_len = kv_last_page_len_buf

        if not self.skip_prefill:
            self.qo_indptr = [
                torch.zeros(
                    (max_bs + 1,), dtype=torch.int32, device=model_runner.device
                )
                for _ in range(self.num_wrappers)
            ]

        fmha_backend = "auto"
        if is_sm100_supported():
            # Disable CUTLASS backend when piecewise cuda graph is enabled
            # due to TMA descriptor initialization issues on SM100 GPUs.
            if not check_cuda_graph_backend(Phase.PREFILL, Backend.TC_PIECEWISE):
                fmha_backend = "cutlass"
        self.prefill_wrapper_ragged = BatchPrefillWithRaggedKVCacheWrapper(
            self.workspace_buffer, "NHD", backend=fmha_backend
        )
        self._fmha_backend = fmha_backend

        # Two wrappers: one for sliding window attention and one for full attention.
        # Using two wrappers is unnecessary in the current PR, but are prepared for future PRs
        self.prefill_wrappers_paged = []
        self.prefill_wrappers_verify = []
        self.decode_wrappers = []
        for _ in range(self.num_wrappers):
            if not skip_prefill:
                self.prefill_wrappers_paged.append(
                    BatchPrefillWithPagedKVCacheWrapper(
                        self.workspace_buffer,
                        "NHD",
                        backend=self.prefill_backend,
                    )
                )
                self.prefill_wrappers_verify.append(
                    BatchPrefillWithPagedKVCacheWrapper(
                        self.workspace_buffer,
                        "NHD",
                        backend=self.prefill_backend,
                    )
                )
            self.decode_wrappers.append(
                BatchDecodeWithPagedKVCacheWrapper(
                    self.workspace_buffer,
                    "NHD",
                    backend=self.decode_backend,
                    use_tensor_cores=self.decode_use_tensor_cores,
                )
            )

        # Create indices updater
        if not skip_prefill:
            self.indices_updater_prefill = FlashInferIndicesUpdaterPrefill(
                model_runner, self
            )  # for verify
        self.indices_updater_decode = FlashInferIndicesUpdaterDecode(model_runner, self)

        # Other metadata
        self.forward_metadata: Union[PrefillMetadata, DecodeMetadata] = None

        self.decode_cuda_graph_metadata = {}
        self.prefill_cuda_graph_metadata = {}  # For verify
        self.draft_extend_cuda_graph_metadata = {}  # For draft extend
        # Uneven-DCP target-verify under CUDA graphs: per-bucket graph-mode
        # RAGGED wrappers (see _get_verify_ragged_cg_wrapper). Keyed by bs.
        # _ragged_wrapper_override is installed by init_forward_metadata_out_graph
        # for the DCP verify buckets and cleared everywhere else, so all
        # non-DCP / eager paths keep using the shared prefill_wrapper_ragged.
        self.verify_ragged_cg_wrappers: dict = {}
        self._ragged_wrapper_override = None
        # Plain EXTEND under full prefill CUDA graph: one wrapper set
        # shared across all captured num_tokens buckets (bs fixed at 1).
        # Created lazily on first capture in _prepare_cuda_graph_metadata.
        self.full_cg_prefill_wrappers: Optional[
            List[BatchPrefillWithPagedKVCacheWrapper]
        ] = None

    @staticmethod
    def _resolve_swa_kv_pool(model_runner: ModelRunner) -> Optional[BaseSWAKVPool]:
        """Return the SWA KV pool to translate against, or None for non-SWA models.

        EAGLE-like draft workers share the target allocator for token bookkeeping,
        but own a separate draft KV pool. Do not use the target allocator's SWA
        mapping for that draft pool. FROZEN_KV MTP is the exception: its draft
        path reads target KV directly, so it still needs the allocator pool when
        the active pool is not SWA.
        """
        active_pool = model_runner.token_to_kv_pool
        if isinstance(active_pool, BaseSWAKVPool):
            return active_pool

        if model_runner.is_draft_worker:
            if not model_runner.spec_algorithm.is_frozen_kv_mtp():
                return None

        kvcache = model_runner.token_to_kv_pool_allocator.get_kvcache()
        return kvcache if isinstance(kvcache, BaseSWAKVPool) else None

    def _process_multi_item_scoring(
        self, forward_batch: ForwardBatch
    ) -> MultiItemScoringParams:
        """Process multi-item scoring tensors for FlashInfer attention.

        This method handles sequences containing multiple "items" separated by delimiter tokens,
        where each item needs specific attention patterns that respect item boundaries.

        The method produces four key tensors for FlashInfer:
        - prefix_len_ptr: uint32 tensor with prefix length for each prompt in batch
        - token_pos_in_items_ptr: uint16 tensor with token positions starting from 0 at delimiters
        - token_pos_in_items_len: padding length for batch processing
        - max_item_len_ptr: uint16 tensor with max item length for each prompt

        Args:
            forward_batch: The forward batch containing input sequences and delimiter info

        Returns:
            MultiItemScoringParams: The processed multi-item scoring parameters

        Examples:
            Following FlashInfer definition: for 3 items of length 3, 2, 4 respectively:
            token_pos_in_items_ptr = [0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3, 4, 0]

            Case 1: Single sequence
            Text: "What is the capital of France? <delim> London <delim> Paris <delim> Berlin <delim>"
            Tokens: [What, is, the, capital, of, France, ?, <delim>, London, <delim>, Paris, <delim>, Berlin, <delim>]
            Indices: [ 0,   1,  2,   3,      4,  5,     6,   7,     8,      9,     10,    11,    12,     13]
            - prefix_len_ptr: [7] (query length before first delimiter)
            - token_pos_in_items_ptr: [0, 1, 0, 1, 0, 1, 0] (delim=0, London=1, delim=0, Paris=1, delim=0, Berlin=1, delim=0)
            - token_pos_in_items_len: 7 (actual length)
            - max_item_len_ptr: [1] (max item length is 1 token - all options are single tokens)

            Case 2: Batch processing (batch_size=2)
            Sequence 1: 2 items of length 2, 1 → [0, 1, 2, 0, 1, 0] (6 elements)
            Sequence 2: 3 items of length 1, 3, 2 → [0, 1, 0, 1, 2, 3, 0, 1, 2, 0] (10 elements)
            After padding both to length 10:
            - token_pos_in_items_ptr: [0, 1, 2, 0, 1, 0, 0, 0, 0, 0,    0, 1, 0, 1, 2, 3, 0, 1, 2, 0]
            - token_pos_in_items_len: 10 (padded length for batch processing)
            - max_item_len_ptr: [2, 3] (max lengths per sequence)
        """

        if not self.enable_mis or forward_batch.forward_mode == ForwardMode.DECODE:
            return MultiItemScoringParams()

        precomputed_indices = forward_batch.multi_item_delimiter_indices
        if precomputed_indices is None:
            return MultiItemScoringParams()

        prefix_cache_lens = getattr(forward_batch, "extend_prefix_lens_cpu", None)
        extend_seq_lens = getattr(forward_batch, "extend_seq_lens_cpu", None)
        prefix_len_ptr, token_pos_in_items_ptr = [], []
        token_pos_in_items_len = 0
        device = forward_batch.input_ids.device

        # If no extend_seq_lens, treat whole batch as one sequence
        if extend_seq_lens is None or len(extend_seq_lens) <= 1:
            extend_seq_lens = [forward_batch.input_ids.size(0)]

        seq_start = 0
        for i, seq_len in enumerate(extend_seq_lens):
            seq_end = seq_start + seq_len
            delimiter_indices_cpu = precomputed_indices[i]
            if len(delimiter_indices_cpu) == 0:
                seq_start = seq_end
                continue

            first_delim = delimiter_indices_cpu[0].item()  # CPU .item(), no GPU sync
            delimiter_indices = delimiter_indices_cpu.to(device, non_blocking=True)
            prefix_len = first_delim + (
                prefix_cache_lens[i] if prefix_cache_lens is not None else 0
            )
            prefix_len_ptr.append(prefix_len)

            # Compute relative positions within items using searchsorted (no GPU sync).
            #   suffix_range      = [0, 1, 2, 3, 4, ...]
            #   searchsorted      = bucket index for each position
            #   last_delim        = delimiter offset at start of current bucket
            #   pos_within_item   = suffix_range - last_delim
            suffix_len = seq_len - first_delim
            relative_positions = delimiter_indices - first_delim

            suffix_range = torch.arange(suffix_len, dtype=torch.int64, device=device)
            bucket_idx = torch.searchsorted(
                relative_positions, suffix_range, right=True
            )
            last_delim = relative_positions[torch.clamp(bucket_idx - 1, min=0)]
            pos_within_item = suffix_range - last_delim

            token_pos_in_items_ptr.append(pos_within_item.to(torch.uint16))

            forward_batch.positions[seq_start + first_delim : seq_end] = (
                prefix_len + pos_within_item - 1
            )

            seq_start = seq_end

        # Pad token_pos_in_items_ptr for batch processing
        if token_pos_in_items_ptr:
            token_pos_in_items_len = max(t.numel() for t in token_pos_in_items_ptr)
            token_pos_in_items_ptr = [
                torch.cat(
                    [
                        t,
                        torch.zeros(
                            token_pos_in_items_len - t.numel(),
                            dtype=torch.uint16,
                            device=device,
                        ),
                    ]
                )
                for t in token_pos_in_items_ptr
            ]

        if not prefix_len_ptr or not token_pos_in_items_ptr:
            return MultiItemScoringParams()

        return MultiItemScoringParams(
            prefix_len_ptr=torch.tensor(
                prefix_len_ptr, dtype=torch.uint32, device=device
            ),
            token_pos_in_items_ptr=torch.cat(token_pos_in_items_ptr, dim=0),
            token_pos_in_items_len=token_pos_in_items_len & 0xFFFFFFFF,
            max_item_len_ptr=torch.stack(
                [
                    t.to(torch.int32).max().to(torch.uint16)
                    for t in token_pos_in_items_ptr
                ],
                dim=0,
            ),
        )

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        bs = forward_batch.batch_size
        req_pool_indices = forward_batch.req_pool_indices
        seq_lens = forward_batch.seq_lens
        seq_lens_cpu = forward_batch.seq_lens_cpu
        seq_lens_sum = forward_batch.seq_lens_sum
        encoder_lens = forward_batch.encoder_lens
        forward_mode = forward_batch.forward_mode
        spec_info = forward_batch.spec_info

        if (
            spec_info is not None
            and spec_info.ragged_verify_layout is not None
            and forward_mode.is_target_verify()
        ):
            raise NotImplementedError(
                "FlashInfer does not support ragged verify in cuda graph; "
                "disable SGLANG_RAGGED_VERIFY_MODE for this configuration."
            )

        if in_capture:
            num_tokens = forward_batch.positions.numel()
            self._prepare_cuda_graph_metadata(bs, num_tokens, forward_mode, spec_info)

        # Uneven-DCP target-verify graphs run the ragged draft->draft attention
        # inside the capture; that requires the per-bucket graph-mode ragged
        # wrapper (fixed indptr buffers). All other modes use the shared one.
        if self.uneven_dcp and forward_mode.is_target_verify():
            self._ragged_wrapper_override = self._get_verify_ragged_cg_wrapper(bs)
        else:
            self._ragged_wrapper_override = None

        if forward_mode.is_decode_or_idle():
            self.indices_updater_decode.update(
                req_pool_indices[:bs],
                seq_lens[:bs],
                seq_lens_cpu[:bs] if seq_lens_cpu is not None else None,
                seq_lens_sum,
                decode_wrappers=self.decode_cuda_graph_metadata[bs],
                encoder_lens=encoder_lens[:bs] if encoder_lens is not None else None,
                spec_info=spec_info,
                fixed_split_size=None,
                disable_split_kv=self.disable_cuda_graph_kv_split,
            )
        elif forward_mode.is_target_verify():
            self.indices_updater_prefill.update(
                req_pool_indices[:bs],
                seq_lens[:bs],
                seq_lens_cpu[:bs] if seq_lens_cpu is not None else None,
                seq_lens_sum,
                prefix_lens=None,
                prefill_wrappers=self.prefill_cuda_graph_metadata[bs],
                use_ragged=False,
                encoder_lens=encoder_lens[:bs] if encoder_lens is not None else None,
                spec_info=spec_info,
            )
        elif forward_mode.is_dllm_extend():
            self.indices_updater_prefill.update(
                req_pool_indices[:bs],
                seq_lens[:bs],
                seq_lens_cpu[:bs] if seq_lens_cpu is not None else None,
                seq_lens_sum,
                prefix_lens=seq_lens - self.dllm_config.block_size,
                prefill_wrappers=self.prefill_cuda_graph_metadata[bs],
                use_ragged=not self.use_paged,
                encoder_lens=encoder_lens[:bs] if encoder_lens is not None else None,
                spec_info=None,
            )
        elif forward_mode.is_draft_extend_v2():
            self.indices_updater_prefill.update(
                req_pool_indices[:bs],
                seq_lens[:bs],
                seq_lens_cpu[:bs] if seq_lens_cpu is not None else None,
                seq_lens_sum,
                prefix_lens=None,
                prefill_wrappers=self.draft_extend_cuda_graph_metadata[bs],
                use_ragged=False,
                encoder_lens=encoder_lens[:bs] if encoder_lens is not None else None,
                spec_info=spec_info,
            )
        elif forward_mode.is_extend():
            # Plain EXTEND under full prefill CUDA graph. plan() runs
            # out-of-graph against capture-stable wrappers; captured kernels
            # read the refreshed state at replay. Must stay below the
            # target-verify / draft-extend / dllm branches (also is_extend()).
            # Split-kv must stay on — its block_valid_mask is the only
            # early-exit for the captured fixed grid's padded/stale tiles.
            self.indices_updater_prefill.update(
                req_pool_indices[:bs],
                seq_lens[:bs],
                seq_lens_cpu[:bs] if seq_lens_cpu is not None else None,
                seq_lens_sum,
                prefix_lens=forward_batch.extend_prefix_lens[:bs],
                prefill_wrappers=self.full_cg_prefill_wrappers,
                use_ragged=False,
                encoder_lens=encoder_lens[:bs] if encoder_lens is not None else None,
                spec_info=None,
            )
        else:
            raise ValueError("Invalid forward mode")

        if in_capture and forward_mode.is_decode_or_idle():
            # fast_decode_plan needs _cached_module from the initial begin_forward
            # above, so install it only after that first plan has run.
            for w in self.decode_cuda_graph_metadata[bs]:
                w.begin_forward = partial(fast_decode_plan, w)

        if (
            in_capture
            and forward_mode.is_draft_extend_v2()
            and self.prefill_backend == "fa2"
            # Host-rebuilt layout only matches full attention (single wrapper);
            # SWA/cross-attn keep the plain plan().
            and self.dispatch_reason is None
        ):
            # Like decode: swap in fast_prefill_plan for replay, after the real
            # plan() above set up _cached_module (host metadata supplied per-replay
            # in call_begin_forward).
            for w in self.draft_extend_cuda_graph_metadata[bs]:
                w.begin_forward = partial(fast_prefill_plan, w)

        # Refill the SWA write-target buffer from the live out_cache_loc before
        # replay (bound onto the metadata at capture below).
        if self.use_sliding_window_kv_pool and forward_batch.out_cache_loc is not None:
            assert self._swa_kv_pool is not None
            n = forward_batch.out_cache_loc.shape[0]
            self.cuda_graph_swa_out_cache_loc[n:].zero_()
            self.cuda_graph_swa_out_cache_loc[:n].copy_(
                self._swa_kv_pool.translate_loc_from_full_to_swa(
                    forward_batch.out_cache_loc
                )
            )
            if in_capture:
                self.forward_metadata.swa_out_cache_loc = (
                    self.cuda_graph_swa_out_cache_loc[:n]
                )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        # Eager path: never use a graph-bucket ragged wrapper.
        self._ragged_wrapper_override = None
        swa_out_cache_loc = None
        if self.use_sliding_window_kv_pool and forward_batch.out_cache_loc is not None:
            assert self._swa_kv_pool is not None
            swa_out_cache_loc = self._swa_kv_pool.translate_loc_from_full_to_swa(
                forward_batch.out_cache_loc
            )

        if forward_batch.forward_mode.is_decode_or_idle():
            self.indices_updater_decode.update(
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.seq_lens_cpu,
                forward_batch.seq_lens_sum,
                decode_wrappers=self.decode_wrappers,
                encoder_lens=forward_batch.encoder_lens,
                spec_info=forward_batch.spec_info,
                fixed_split_size=self.decode_split_tile_size,
                disable_split_kv=False,
            )
            self.forward_metadata = DecodeMetadata(
                self.decode_wrappers, swa_out_cache_loc=swa_out_cache_loc
            )
        elif forward_batch.forward_mode.is_target_verify():
            self.indices_updater_prefill.update(
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.seq_lens_cpu,
                forward_batch.seq_lens_sum,
                prefix_lens=None,
                prefill_wrappers=self.prefill_wrappers_verify,
                use_ragged=False,
                encoder_lens=forward_batch.encoder_lens,
                spec_info=forward_batch.spec_info,
            )
            self.forward_metadata = PrefillMetadata(
                self.prefill_wrappers_verify,
                False,
                False,
                swa_out_cache_loc=swa_out_cache_loc,
            )
        else:
            prefix_lens = forward_batch.extend_prefix_lens

            # Disable ragged wrapper and ensure prefix handling for multimodal and multi-item scoring
            if self.is_multimodal or self.enable_mis:
                # use_ragged = False: Multi-item scoring requires the paged wrapper because:
                # 1. Ragged wrapper doesn't support the specialized multi-item parameters
                #    (prefix_len_ptr, token_pos_in_items_ptr, etc.)
                # 2. Paged wrapper provides better control over attention masking needed
                #    for respecting item boundaries in multi-item sequences
                # 3. Custom masking logic conflicts with ragged wrapper's assumptions
                use_ragged = False
                extend_no_prefix = False
            else:
                use_ragged = (
                    not self.enable_deterministic
                    and not is_in_tc_piecewise_cuda_graph()
                    and not self.use_paged
                )
                extend_no_prefix = not any(forward_batch.extend_prefix_lens_cpu)

            # Process multi-item scoring in attention backend instead of ForwardBatch
            multi_item_params = MultiItemScoringParams()
            if self.enable_mis:
                # Use new backend-specific implementation
                multi_item_params = self._process_multi_item_scoring(forward_batch)

            self.indices_updater_prefill.update(
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.seq_lens_cpu,
                forward_batch.seq_lens_sum,
                prefix_lens,
                prefill_wrappers=self.prefill_wrappers_paged,
                use_ragged=use_ragged,
                encoder_lens=forward_batch.encoder_lens,
                spec_info=None,
                fixed_split_size=self.prefill_split_tile_size,
                multi_item_params=multi_item_params,
                cross_attention_custom_mask=forward_batch.cross_attention_custom_mask,
                extend_prefix_lens_cpu=forward_batch.extend_prefix_lens_cpu,
            )
            self.forward_metadata = PrefillMetadata(
                self.prefill_wrappers_paged,
                use_ragged,
                extend_no_prefix,
                multi_item_params,
                swa_out_cache_loc=swa_out_cache_loc,
            )

    def init_cuda_graph_state(
        self,
        max_bs: int,
        max_num_tokens: int,
        kv_indices_buf: Optional[torch.Tensor] = None,
    ):
        if kv_indices_buf is None:
            cuda_graph_kv_indices = torch.zeros(
                (max_num_tokens * self.max_context_len,),
                dtype=torch.int32,
                device="cuda",
            )
        else:
            cuda_graph_kv_indices = kv_indices_buf

        self.cuda_graph_kv_indices = [cuda_graph_kv_indices] + [
            cuda_graph_kv_indices.clone() for _ in range(self.num_wrappers - 1)
        ]

        # SWA write-target buffer; refilled and bound onto forward_metadata in
        # init_forward_metadata_out_graph before each replay.
        self.cuda_graph_swa_out_cache_loc = (
            torch.zeros(max_num_tokens, dtype=torch.int64, device="cuda")
            if self.use_sliding_window_kv_pool
            else None
        )

        # Ensure tensors are properly allocated
        for i in range(self.num_wrappers):
            # Force allocation by performing a small operation
            if len(self.cuda_graph_kv_indices[i]) > 0:
                self.cuda_graph_kv_indices[i][0] = 0

        if not self.skip_prefill:
            self.cuda_graph_custom_mask = torch.zeros(
                (max_num_tokens * self.max_context_len),
                dtype=torch.uint8,
                device="cuda",
            )
            self.cuda_graph_qk_indptr = [x.clone() for x in self.kv_indptr]
            self.cuda_graph_qo_indptr = [x.clone() for x in self.kv_indptr]

    def _create_decode_wrappers(self, bs: int, num_tokens: int) -> list:
        return [
            BatchDecodeWithPagedKVCacheWrapper(
                self.workspace_buffer,
                "NHD",
                backend=self.decode_backend,
                use_cuda_graph=True,
                use_tensor_cores=self.decode_use_tensor_cores,
                paged_kv_indptr_buffer=self.kv_indptr[i][: num_tokens + 1],
                paged_kv_indices_buffer=self.cuda_graph_kv_indices[i],
                paged_kv_last_page_len_buffer=self.kv_last_page_len[:num_tokens],
            )
            for i in range(self.num_wrappers)
        ]

    def _create_prefill_wrappers(self, bs: int, use_custom_mask: bool = False) -> list:
        # FlashInfer's prefill wrapper decides mask mode based on whether
        # `custom_mask_buf` is initialized (not whether a custom mask is provided).
        # For cases like DFLASH draft (ENCODER_ONLY / non-causal) we do NOT use a
        # custom mask, so we must avoid initializing `custom_mask_buf`, otherwise
        # FlashInfer will treat the (zero) buffer as a real mask and block attention.
        wrappers = []
        for i in range(self.num_wrappers):
            extra = (
                {
                    "custom_mask_buf": self.cuda_graph_custom_mask,
                    "mask_indptr_buf": self.cuda_graph_qk_indptr[i][: bs + 1],
                }
                if use_custom_mask
                else {}
            )
            wrappers.append(
                BatchPrefillWithPagedKVCacheWrapper(
                    self.workspace_buffer,
                    "NHD",
                    use_cuda_graph=True,
                    backend=self.prefill_backend,
                    qo_indptr_buf=self.cuda_graph_qo_indptr[i][: bs + 1],
                    paged_kv_indptr_buf=self.kv_indptr[i][: bs + 1],
                    paged_kv_indices_buf=self.cuda_graph_kv_indices[i],
                    paged_kv_last_page_len_buf=self.kv_last_page_len[:bs],
                    **extra,
                )
            )
        return wrappers

    @staticmethod
    def _full_cg_prefill_workspace_bytes(
        num_slots: int,
        max_num_tokens: int,
        *,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        device: torch.device,
    ) -> int:
        """Split-kv worst-case float-workspace demand for the plain-EXTEND
        cudagraph wrappers, mirroring flashinfer's PrefillPlan sizing
        (scheduler.cuh, enable_cuda_graph=True) for the largest captured
        bucket:

          cta_tile_q = FA2DetermineCtaTileQ(max packed qo len, head_dim)
          tiles      = ceil(max_rows * gqa / cta_tile_q) + batch_size - 1
          padded     = max(2 * num_SMs / num_kv_heads, tiles)
          tmp_v      = num_qo_heads * padded * cta_tile_q * head_dim * fp32
          tmp_s      = num_qo_heads * padded * cta_tile_q * fp32

        Split-kv must stay enabled for these wrappers — its
        block_valid_mask is what lets the padded/stale tiles of the fixed
        captured grid exit early at replay; without it every replay
        re-runs capture-sized attention (measured ~6.5 ms/layer). If a
        future flashinfer outgrows the margin, plan() fails loudly at
        startup ("Increase the workspace buffer size").
        """
        gqa_group_size = num_qo_heads // num_kv_heads
        max_qo_len = (max_num_tokens - num_slots + 1) * gqa_group_size
        if max_qo_len > 64 and head_dim < 256:
            cta_tile_q = 128
        elif max_qo_len > 16:
            cta_tile_q = 64
        else:
            cta_tile_q = 16
        tiles = -(-max_num_tokens * gqa_group_size // cta_tile_q) + num_slots - 1
        num_sm = torch.cuda.get_device_properties(device).multi_processor_count
        padded_batch_size = max((2 * num_sm) // num_kv_heads, tiles)
        per_row = num_qo_heads * padded_batch_size * cta_tile_q * 4
        tmp_v = per_row * head_dim
        tmp_s = per_row
        return int((tmp_v + tmp_s) * FULL_CG_PREFILL_WORKSPACE_MARGIN)

    def _create_full_cg_prefill_wrappers(
        self, num_slots: int, max_num_tokens: int
    ) -> list:
        """Wrappers for plain EXTEND captured under a full prefill CUDA
        graph. plan() must keep its internal state at capture-stable
        addresses (use_cuda_graph=True); the decode-side cuda-graph
        wrappers permanently pin the shared workspace via their own
        plans, so these get a dedicated workspace sized from the largest
        captured bucket. The request-slot count is fixed at capture (the
        runner pads real batches up to it with zero-length sentinel
        requests); kv indices cover up to num_slots sequences of
        max_context_len.
        """
        device = self.workspace_buffer.device
        self.full_cg_prefill_req_slots = num_slots
        upd = self.indices_updater_prefill
        workspace_bytes = self._full_cg_prefill_workspace_bytes(
            num_slots,
            max_num_tokens,
            num_qo_heads=upd.num_qo_heads,
            num_kv_heads=upd.num_kv_heads,
            head_dim=upd.head_dim,
            device=device,
        )
        logger.info(
            "Full-CG prefill workspace: %.0f MB (max bucket %d tokens, "
            "%d request slots)",
            workspace_bytes / (1024 * 1024),
            max_num_tokens,
            num_slots,
        )
        self.full_cg_prefill_workspace_buffer = register_flashinfer_workspace_buffer(
            torch.empty(workspace_bytes, dtype=torch.uint8, device=device)
        )
        self.full_cg_prefill_qo_indptr = [
            torch.zeros((num_slots + 1,), dtype=torch.int32, device=device)
            for _ in range(self.num_wrappers)
        ]
        self.full_cg_prefill_kv_indptr = [
            torch.zeros((num_slots + 1,), dtype=torch.int32, device=device)
            for _ in range(self.num_wrappers)
        ]
        # call_begin_forward materializes paged_kernel_lens_sum + 256
        # indices; size the fixed buffer for the worst case.
        self.full_cg_prefill_kv_indices = [
            torch.zeros(
                (num_slots * self.max_context_len + 256,),
                dtype=torch.int32,
                device=device,
            )
            for _ in range(self.num_wrappers)
        ]
        return [
            BatchPrefillWithPagedKVCacheWrapper(
                self.full_cg_prefill_workspace_buffer,
                "NHD",
                use_cuda_graph=True,
                backend=self.prefill_backend,
                qo_indptr_buf=self.full_cg_prefill_qo_indptr[i],
                paged_kv_indptr_buf=self.full_cg_prefill_kv_indptr[i],
                paged_kv_indices_buf=self.full_cg_prefill_kv_indices[i],
                paged_kv_last_page_len_buf=self.kv_last_page_len[:num_slots],
            )
            for i in range(self.num_wrappers)
        ]

    @property
    def active_ragged_wrapper(self) -> BatchPrefillWithRaggedKVCacheWrapper:
        """The RAGGED wrapper the current forward must use. Identical to the
        shared prefill_wrapper_ragged unless the uneven-DCP target-verify CUDA
        graph installed a per-bucket graph-mode override (see
        _get_verify_ragged_cg_wrapper)."""
        return self._ragged_wrapper_override or self.prefill_wrapper_ragged

    def _get_verify_ragged_cg_wrapper(
        self, bs: int
    ) -> BatchPrefillWithRaggedKVCacheWrapper:
        """Per-bucket CUDA-graph-mode RAGGED wrapper for the uneven-DCP
        target-verify graph.

        ROOT CAUSE this fixes (MTP+graphs+DCP bs>1 illegal memory access):
        the DCP verify path runs the draft->draft chain attention through a
        RAGGED wrapper INSIDE the captured graph (_forward_extend_dcp). The
        shared prefill_wrapper_ragged is NOT in cuda-graph mode, so its plan()
        stores ``self._qo_indptr_buf = qo_indptr.to(self.device)`` -- for an
        already-on-device tensor that is a bare REFERENCE to the caller's
        transient ``torch.arange`` (call_begin_forward). The captured kernel
        freezes that raw pointer; the tensor is freed after the next plan()
        (later bucket captures / replays), the allocator reuses the block, and
        replaying any bs>1 bucket reads garbage indptr -> out-of-bounds ragged
        attention -> cudaErrorIllegalAddress. bs=1 only survived by allocator
        luck: it is captured LAST, its [0, draft_num] indptr block is never
        clobbered, and its content is constant.

        The fix is flashinfer's intended cuda-graph usage: one ragged wrapper
        per captured bucket with FIXED qo/kv indptr buffers (plan() then
        copy_()s into them, so the captured pointers stay valid and refreshed).
        Buffers are tiny (bs+1 int32); each wrapper owns its 8 MiB int
        workspace, kept alive in verify_ragged_cg_wrappers for the process
        lifetime -- sized by what capture+replay actually needs, no heuristics.
        """
        wrapper = self.verify_ragged_cg_wrappers.get(bs)
        if wrapper is None:
            device = self.kv_last_page_len.device
            qo_indptr_buf = torch.zeros((bs + 1,), dtype=torch.int32, device=device)
            kv_indptr_buf = torch.zeros((bs + 1,), dtype=torch.int32, device=device)
            wrapper = BatchPrefillWithRaggedKVCacheWrapper(
                self.workspace_buffer,
                "NHD",
                use_cuda_graph=True,
                qo_indptr_buf=qo_indptr_buf,
                kv_indptr_buf=kv_indptr_buf,
                backend=self._fmha_backend,
            )
            self.verify_ragged_cg_wrappers[bs] = wrapper
        return wrapper

    def _prepare_cuda_graph_metadata(
        self,
        bs: int,
        num_tokens: int,
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
    ) -> None:
        if forward_mode.is_decode_or_idle():
            decode_wrappers = self._create_decode_wrappers(bs, num_tokens)
            self.decode_cuda_graph_metadata[bs] = decode_wrappers
            self.forward_metadata = DecodeMetadata(decode_wrappers)
        elif forward_mode.is_target_verify() or forward_mode.is_dllm_extend():
            use_custom_mask = (
                forward_mode.is_target_verify()
                and spec_info is not None
                and getattr(spec_info, "custom_mask", None) is not None
                # Uneven-DCP verify NEVER plans a packed custom mask (chain
                # topk=1: the ragged wrapper handles draft->draft causality,
                # the paged prefix read is non-causal; tree-spec is guarded
                # off under DCP). flashinfer decides mask mode by BUFFER
                # PRESENCE, not by what plan() received (prefill.py run():
                # `if self._custom_mask_buf is not None: mask_mode=CUSTOM`),
                # so initializing custom_mask_buf here would force every
                # captured verify run into CUSTOM mask mode against the
                # never-refreshed all-zero cuda_graph_custom_mask -> the
                # committed-prefix read is fully masked out and the verify
                # logits are prompt-blind (M16, MTP+graphs+DCP corruption).
                and not self.uneven_dcp
            )
            prefill_wrappers = self._create_prefill_wrappers(bs, use_custom_mask)
            self.prefill_cuda_graph_metadata[bs] = prefill_wrappers
            self.forward_metadata = PrefillMetadata(
                prefill_wrappers, forward_mode.is_dllm_extend(), False
            )
        elif forward_mode.is_draft_extend_v2():
            # Draft-extend: causal paged prefill over the full sequence (no mask).
            prefill_wrappers = self._create_prefill_wrappers(bs, use_custom_mask=False)
            self.draft_extend_cuda_graph_metadata[bs] = prefill_wrappers
            self.forward_metadata = PrefillMetadata(prefill_wrappers, False, False)
        elif forward_mode.is_extend():
            if self.full_cg_prefill_wrappers is None:
                self.full_cg_prefill_wrappers = self._create_full_cg_prefill_wrappers(
                    bs, num_tokens
                )
            self.forward_metadata = PrefillMetadata(
                self.full_cg_prefill_wrappers, False, False
            )
        else:
            raise ValueError(f"Invalid mode: {forward_mode=}")

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    @debug_kernel_api
    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        prefill_wrapper_paged = self.forward_metadata.prefill_wrappers[
            self._get_wrapper_idx(layer)
        ]
        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )

        logits_soft_cap = layer.logit_cap

        q = q.contiguous()

        if self.uneven_dcp:
            # target-VERIFY: the committed prefix is ALWAYS present (decode-phase
            # verify), and its length is seq_lens (not extend_prefix_lens, which
            # is unset for verify). Force the prefix read so the draft tokens
            # attend the token-sharded context.
            force_prefix = forward_batch.forward_mode.is_target_verify()
            return self._forward_extend_dcp(
                q, k, v, layer, forward_batch, prefill_wrapper_paged, cache_loc,
                logits_soft_cap, save_kv_cache, force_prefix=force_prefix,
            )

        if not self.forward_metadata.use_ragged:
            if k is not None:
                assert v is not None
                if save_kv_cache:
                    self.token_to_kv_pool.set_kv_buffer(
                        layer,
                        KVWriteLoc(cache_loc, self.forward_metadata.swa_out_cache_loc),
                        k,
                        v,
                        layer.k_scale,
                        layer.v_scale,
                    )

            causal = (
                not layer.is_cross_attention
                and layer.attn_type != AttentionType.ENCODER_ONLY
            )
            o = prefill_wrapper_paged.forward(
                q.view(-1, layer.tp_q_head_num, layer.head_dim),
                self.token_to_kv_pool.get_kv_buffer(layer.layer_id),
                causal=causal,
                sm_scale=layer.scaling,
                # Disable sliding window attention for multi-item scoring:
                # - Sliding window could cut across item boundaries, breaking semantic coherence
                # - Multi-item sequences need full attention to properly handle delimiter tokens
                # - Specialized multi-item parameters (prefix_len_ptr, token_pos_in_items_ptr)
                #   provide more precise attention control than simple sliding windows
                # - Item-aware masking takes precedence over window-based masking
                window_left=(
                    layer.sliding_window_size
                    if not (
                        self.forward_metadata.multi_item_params
                        and self.forward_metadata.multi_item_params.is_enabled()
                    )
                    else -1
                ),
                logits_soft_cap=logits_soft_cap,
                # Must use _float to avoid device-to-host copy that breaks cuda graph capture.
                k_scale=layer.k_scale_float,
                v_scale=layer.v_scale_float,
            )
        else:
            # If `k`/`v` are not explicitly provided, fall back to the KV cache stored in
            # `self.token_to_kv_pool` for this layer. This enables attention over
            # previously cached context without re-materializing KV tensors (e.g., the
            # IQuestLoopCoder path uses token_to_kv_pool as the KV source).
            if k is None and v is None:
                k = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)[0]
                v = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)[1]
            causal = True
            if (
                layer.is_cross_attention
                or layer.attn_type == AttentionType.ENCODER_ONLY
            ):
                causal = False
            if not self.is_dllm_model and layer.attn_type == AttentionType.ENCODER_ONLY:
                save_kv_cache = False

            if self.forward_metadata.extend_no_prefix:
                # NOTE: FlashInfer currently has limitations with head_dim = 32 or other dimensions
                # The FlashInfer head_dim limitation itself is tracked here:
                # https://github.com/flashinfer-ai/flashinfer/issues/1048
                o = self.prefill_wrapper_ragged.forward(
                    q.view(-1, layer.tp_q_head_num, layer.head_dim),
                    k.view(-1, layer.tp_k_head_num, layer.head_dim),
                    v.view(-1, layer.tp_v_head_num, layer.head_dim),
                    causal=causal,
                    sm_scale=layer.scaling,
                    logits_soft_cap=logits_soft_cap,
                )

            else:
                swa_window_left = (
                    layer.sliding_window_size
                    if not (
                        self.forward_metadata.multi_item_params
                        and self.forward_metadata.multi_item_params.is_enabled()
                    )
                    else -1
                )
                o1, s1 = self.prefill_wrapper_ragged.forward_return_lse(
                    q.view(-1, layer.tp_q_head_num, layer.head_dim),
                    k.view(-1, layer.tp_k_head_num, layer.head_dim),
                    v.view(-1, layer.tp_v_head_num, layer.head_dim),
                    causal=causal,
                    sm_scale=layer.scaling,
                    window_left=swa_window_left,
                    logits_soft_cap=logits_soft_cap,
                )
                o2, s2 = prefill_wrapper_paged.forward_return_lse(
                    q.view(-1, layer.tp_q_head_num, layer.head_dim),
                    self.token_to_kv_pool.get_kv_buffer(layer.layer_id),
                    causal=False,
                    sm_scale=layer.scaling,
                    window_left=swa_window_left,
                    logits_soft_cap=logits_soft_cap,
                )

                o, _ = _safe_merge_state(o1, s1, o2, s2)

            if save_kv_cache:
                self.token_to_kv_pool.set_kv_buffer(
                    layer,
                    KVWriteLoc(cache_loc, self.forward_metadata.swa_out_cache_loc),
                    k,
                    v,
                    layer.k_scale,
                    layer.v_scale,
                )

        return o.view(-1, layer.tp_q_head_num * layer.head_dim)

    @debug_kernel_api
    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        decode_wrapper = self.forward_metadata.decode_wrappers[
            self._get_wrapper_idx(layer)
        ]
        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )

        if self.uneven_dcp:
            return self._forward_decode_dcp(
                q, k, v, layer, forward_batch, decode_wrapper, cache_loc, save_kv_cache
            )

        if k is not None:
            assert v is not None
            if save_kv_cache:
                self.token_to_kv_pool.set_kv_buffer(
                    layer,
                    KVWriteLoc(cache_loc, self.forward_metadata.swa_out_cache_loc),
                    k,
                    v,
                    layer.k_scale,
                    layer.v_scale,
                )

        # Call the wrapped function
        o = decode_wrapper.forward(
            q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
            self.token_to_kv_pool.get_kv_buffer(layer.layer_id),
            sm_scale=layer.scaling,
            logits_soft_cap=layer.logit_cap,
            # Must use _float to avoid device-to-host copy that breaks cuda graph capture.
            k_scale=layer.k_scale_float,
            v_scale=layer.v_scale_float,
        )

        return o.view(-1, layer.tp_q_head_num * layer.head_dim)

    def _dcp_masked_write(self, layer, forward_batch, cache_loc, k, v):
        """Gather this rank's local kv-head shard [2,1,1] up to the FULL
        replicated kv-heads [4] and write only the token slots this rank OWNS
        (even-modulo owner rule) at the compacted physical slot loc//dcp_size.

        REPLICATED-KV geometry (TP > num_kv_heads): every rank already
        projects ALL kv heads itself (identical replicated k/v weights ->
        identical K/V, deterministic recompute), so the gather is skipped."""
        group = get_parallel().dcp_group
        k = k.view(-1, layer.tp_k_head_num, layer.head_dim)
        v = v.view(-1, layer.tp_v_head_num, layer.head_dim)
        if self.dcp_kv_replicated_heads:
            k_full, v_full = k, v
        else:
            k_full = cp_all_gather_heads_uneven(k, group, self.dcp_kv_head_counts)
            v_full = cp_all_gather_heads_uneven(v, group, self.dcp_kv_head_counts)
        if self.uneven_dcp_weighted:
            # WEIGHTED owner rule: ownership + compact physical slot are derived
            # from the out_cache_loc itself (not the sequence position), so the
            # slot is an injective function of L and stays collision-free across
            # concurrent requests exactly like the even L // dcp_size. This rank
            # owns L iff (L % cp_S) in [cp_lo, cp_hi); its compact slot is
            # (L // cp_S) * cp_ratio + (L % cp_S - cp_lo).
            off = cache_loc % self.cp_S
            block = cache_loc // self.cp_S
            dcp_kv_mask = (off >= self.cp_lo) & (off < self.cp_hi)
            loc = torch.where(
                dcp_kv_mask,
                block * self.cp_ratio + (off - self.cp_lo),
                torch.zeros_like(cache_loc),
            )
        else:
            loc = cache_loc // self.dcp_size
            positions = forward_batch.positions
            if positions is not None and positions.numel() == loc.numel():
                dcp_kv_mask = positions % self.dcp_size == self.dcp_rank
            else:
                dcp_kv_mask = forward_batch.dcp_kv_mask
        self.token_to_kv_pool.set_kv_buffer(
            layer,
            loc,
            k_full.clone(),
            v_full.clone(),
            layer.k_scale,
            layer.v_scale,
            dcp_kv_mask=dcp_kv_mask,
        )

    def _forward_decode_dcp(
        self, q, k, v, layer, forward_batch, decode_wrapper, cache_loc, save_kv_cache
    ):
        group = get_parallel().dcp_group
        if k is not None and save_kv_cache:
            self._dcp_masked_write(layer, forward_batch, cache_loc, k, v)

        q_local = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
        q_full = cp_all_gather_heads_uneven(q_local, group, self.dcp_q_head_counts)
        o, lse = decode_wrapper.forward_return_lse(
            q_full,
            self.token_to_kv_pool.get_kv_buffer(layer.layer_id),
            sm_scale=layer.scaling,
            logits_soft_cap=layer.logit_cap,
            k_scale=layer.k_scale_float,
            v_scale=layer.v_scale_float,
        )
        # o: [tokens, 24, D], lse: [tokens, 24]; combine across the DCP token
        # shards and slice back to this rank's [12/6/6] head shard.
        o = cp_lse_ag_out_ar_mha_uneven(o, lse, group, self.dcp_q_head_counts)
        return o.reshape(-1, layer.tp_q_head_num * layer.head_dim).to(q.dtype)

    def _replicated_kv_ragged_reindex(self, local_q, local_kv, device):
        """REPLICATED-KV geometry (#105): map each of this rank's LOCAL kv
        slots to the GLOBAL kv head that its q group attends, so a uniform-GQA
        ragged kernel (gqa_local = local_q // local_kv) reproduces the GLOBAL
        q->kv mapping (gqa_global = total_q // total_kv).

        Returns a LongTensor of length local_kv to gather k/v heads with, or
        None when the mapping is already the identity [0,1,...] (no reindex
        needed). Cached per (local_q, local_kv) -- constant across layers and
        forwards for a given rank. Fails fast if this rank's q heads straddle a
        global kv-head boundary within one local slot (the uniform-GQA kernel
        cannot represent it -- pick a --rank-tp-ratio whose q-unit boundaries
        align with kv-head boundaries)."""
        cache = getattr(self, "_repl_kv_reindex_cache", None)
        if cache is None:
            cache = self._repl_kv_reindex_cache = {}
        key = (local_q, local_kv)
        if key in cache:
            return cache[key]
        q_off = sum(self.dcp_q_head_counts[: self.dcp_rank])
        global_gqa = self.dcp_full_qo_heads // self.dcp_full_kv_heads
        local_gqa = local_q // local_kv
        idx = []
        for m in range(local_kv):
            lo = q_off + m * local_gqa
            hi = q_off + (m + 1) * local_gqa - 1
            g = lo // global_gqa
            if hi // global_gqa != g:
                raise ValueError(
                    f"REPLICATED-KV current-chunk attention (#105): this rank's "
                    f"q heads (offset {q_off}, {local_q} heads over {local_kv} "
                    f"local kv slots) straddle a global kv-head boundary "
                    f"(global GQA group size {global_gqa}); the uniform-GQA "
                    f"ragged kernel cannot represent this split. Choose a "
                    f"--rank-tp-ratio whose q-unit boundaries align with "
                    f"kv-head boundaries."
                )
            idx.append(g)
        if idx == list(range(local_kv)):
            cache[key] = None
            return None
        t = torch.tensor(idx, device=device, dtype=torch.long)
        cache[key] = t
        return t

    def _forward_extend_dcp(
        self,
        q,
        k,
        v,
        layer,
        forward_batch,
        prefill_wrapper_paged,
        cache_loc,
        logits_soft_cap,
        save_kv_cache,
        force_prefix=False,
    ):
        group = get_parallel().dcp_group
        # 1. Write the current chunk's KV (full replicated kv-heads) into this
        #    rank's owned token slots (token-sharded cache).
        if k is not None and save_kv_cache:
            self._dcp_masked_write(layer, forward_batch, cache_loc, k, v)

        causal = (
            not layer.is_cross_attention
            and layer.attn_type != AttentionType.ENCODER_ONLY
        )
        q_local = q.view(-1, layer.tp_q_head_num, layer.head_dim)

        # 2. Current chunk: ragged LOCAL head-sharded attention (causal). k/v are
        #    the freshly-projected bf16 current tokens, fully present on every
        #    rank, so this is exact local GQA attention (no DCP gather needed).
        #    active_ragged_wrapper: under a captured verify graph this is the
        #    per-bucket graph-mode wrapper (fixed indptr buffers) -- the shared
        #    non-graph wrapper's transient plan tensors are NOT replay-safe.
        k_cur = k.view(-1, layer.tp_k_head_num, layer.head_dim)
        v_cur = v.view(-1, layer.tp_v_head_num, layer.head_dim)
        # REPLICATED-KV (TP > num_kv_heads, #105): this rank holds only a q-head
        # SLICE but ALL kv heads (replicated). flashinfer derives the ragged
        # GQA grouping from the LOCAL counts (gqa_local = local_q // local_kv),
        # which maps q heads to kv heads differently from the GLOBAL grouping
        # (gqa_global = total_q // total_kv) whenever this rank does not hold
        # every q head of its kv head(s). Re-index the local kv slots so each
        # carries the GLOBAL kv head its q group actually attends. (8q/2kv over
        # [4,2,2]: rank0 held q0-3 -- all -> kv0 globally -- but gqa_local=2 sent
        # q2-3 to kv1, corrupting short-prompt / first-chunk generation while the
        # gathered-q paged prefix/decode path stayed correct.) No-op reindex on
        # the aligned single-kv-head-per-rank case reduces to identity.
        if self.dcp_kv_replicated_heads:
            _kv_idx = self._replicated_kv_ragged_reindex(
                layer.tp_q_head_num, layer.tp_k_head_num, k.device
            )
            if _kv_idx is not None:
                k_cur = k_cur[:, _kv_idx, :]
                v_cur = v_cur[:, _kv_idx, :]
        o_cur, lse_cur = self.active_ragged_wrapper.forward_return_lse(
            q_local,
            k_cur,
            v_cur,
            causal=causal,
            sm_scale=layer.scaling,
            logits_soft_cap=logits_soft_cap,
        )

        # has_prefix is a global (rank-uniform) property, so every rank takes the
        # same branch -> the DCP collectives below stay balanced (no deadlock).
        # target-VERIFY (force_prefix) always reads the committed prefix, whose
        # length is seq_lens (extend_prefix_lens is unset for verify).
        if force_prefix:
            has_prefix = True
        else:
            prefix_cpu = forward_batch.extend_prefix_lens_cpu
            has_prefix = prefix_cpu is not None and any(prefix_cpu)
        if not has_prefix:
            return o_cur.contiguous().view(
                -1, layer.tp_q_head_num * layer.head_dim
            )

        # 3. Prefix: paged DCP read over this rank's OWNED prefix slots with the
        #    FULL gathered q-heads (non-causal: all prefix keys precede every
        #    current query). Combine the per-rank partials across the DCP group
        #    and slice back to this rank's local heads.
        q_full = cp_all_gather_heads_uneven(q_local, group, self.dcp_q_head_counts)
        o_pre, lse_pre = prefill_wrapper_paged.forward_return_lse(
            q_full,
            self.token_to_kv_pool.get_kv_buffer(layer.layer_id),
            causal=False,
            sm_scale=layer.scaling,
            logits_soft_cap=logits_soft_cap,
            k_scale=layer.k_scale_float,
            v_scale=layer.v_scale_float,
        )
        o_pre, lse_pre = cp_lse_ag_out_ar_mha_uneven(
            o_pre, lse_pre, group, self.dcp_q_head_counts, return_lse=True
        )

        # 4. Merge current chunk with prefix (both in this rank's local-head
        #    space) by natural-log LSE. lse_pre is a torch.logsumexp (natural
        #    log) from the cross-rank combine, and flashinfer's ragged
        #    forward_return_lse also returns natural-log LSE, so logaddexp mixes
        #    them consistently. (flashinfer's merge_state uses a different
        #    internal convention and must NOT be used across these two sources.)
        lse_cur = lse_cur.float()
        lse_pre = lse_pre.float()
        final_lse = torch.logaddexp(lse_cur, lse_pre)
        sc_cur = torch.nan_to_num(
            torch.exp(lse_cur - final_lse), nan=0.0, posinf=0.0, neginf=0.0
        ).unsqueeze(-1)
        sc_pre = torch.nan_to_num(
            torch.exp(lse_pre - final_lse), nan=0.0, posinf=0.0, neginf=0.0
        ).unsqueeze(-1)
        o = o_cur.float() * sc_cur + o_pre.float() * sc_pre
        return o.to(q.dtype).contiguous().view(
            -1, layer.tp_q_head_num * layer.head_dim
        )

    def _get_wrapper_idx(self, layer: RadixAttention):
        if self.num_wrappers == 1:
            return 0

        if self.dispatch_reason == WrapperDispatch.SLIDING_WINDOW:
            return layer.sliding_window_size == -1
        if self.dispatch_reason == WrapperDispatch.CROSS_ATTENTION:
            return layer.is_cross_attention

        raise ValueError(f"Unknown dispatch reason: {self.dispatch_reason}")


def _build_dcp_weighted_kv_indices(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    paged_kernel_lens: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_start_idx: Optional[torch.Tensor],
    cp_S: int,
    cp_lo: int,
    cp_hi: int,
    cp_ratio: int,
    pad: int = 0,
):
    """Weighted-DCP paged kv_indices: this rank's OWNED cache slots, compacted.

    Builds the full per-request out_cache_loc list for [kv_start, kv_start+len),
    keeps the slots this rank owns under the WEIGHTED owner rule
    (loc % cp_S in [cp_lo, cp_hi)) and maps each to its compact physical slot
    (loc // cp_S) * cp_ratio + (loc % cp_S - cp_lo) -- the exact inverse of the
    _dcp_masked_write packing, so a token is read from the slot it was written
    to. Writes the per-request owned counts into ``kv_indptr`` and returns
    (kv_indptr[:bs+1], kv_indices).
    """
    bs = len(req_pool_indices)
    device = req_pool_indices.device
    lens64 = paged_kernel_lens.to(torch.int64)
    full_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=device)
    full_indptr[1:] = torch.cumsum(lens64, dim=0).to(torch.int32)
    total = int(full_indptr[bs].item())
    full_kv = torch.empty(max(total, 1), dtype=torch.int32, device=device)
    if total > 0:
        create_flashinfer_kv_indices_triton[(bs,)](
            req_to_token,
            req_pool_indices,
            paged_kernel_lens,
            full_indptr,
            kv_start_idx,
            full_kv,
            req_to_token.shape[1],
        )
    full_kv = full_kv[:total]
    full_kv64 = full_kv.to(torch.int64)
    off = full_kv64 % cp_S
    owned = (off >= cp_lo) & (off < cp_hi)
    compact = ((full_kv64 // cp_S) * cp_ratio + (off - cp_lo)).to(torch.int32)
    kv_indices = compact[owned].contiguous()
    if pad > 0:
        kv_indices = torch.cat([kv_indices, kv_indices.new_zeros(pad)])
    if total > 0:
        seg = torch.repeat_interleave(
            torch.arange(bs, device=device, dtype=torch.int64), lens64
        )
        owned_per_req = torch.zeros(bs, dtype=torch.int64, device=device)
        owned_per_req.scatter_add_(0, seg, owned.to(torch.int64))
    else:
        owned_per_req = torch.zeros(bs, dtype=torch.int64, device=device)
    kv_indptr[1 : bs + 1] = torch.cumsum(owned_per_req, dim=0)
    return kv_indptr[: bs + 1], kv_indices


class FlashInferIndicesUpdaterDecode:
    def __init__(self, model_runner: ModelRunner, attn_backend: FlashInferAttnBackend):
        # Parse Constants
        # THIS rank's real head counts: under uneven TP (--rank-tp-ratio)
        # the split is uneven, and flashinfer's split-KV plan/merge must
        # match the actual per-rank q/kv tensors (see
        # _local_attn_head_counts). Degrades to the even split by default.
        self.num_qo_heads, self.num_kv_heads = _local_attn_head_counts(model_runner)
        self.head_dim = model_runner.model_config.head_dim
        self.data_type = model_runner.kv_cache_dtype
        self.q_data_type = model_runner.dtype
        self.sliding_window_size = model_runner.sliding_window_size
        self.attn_backend = attn_backend

        # Uneven-DCP: the paged decode wrapper reads the token-sharded KV with
        # FULL replicated kv-heads and the GATHERED q-heads, so plan it with the
        # full counts (see FlashInferAttnBackend.uneven_dcp). The forward gathers
        # this rank's q shard up to these full counts and slices the combined
        # output back down.
        if attn_backend.uneven_dcp:
            self.num_qo_heads = attn_backend.dcp_full_qo_heads
            self.num_kv_heads = attn_backend.dcp_full_kv_heads

        # Buffers and wrappers
        self.kv_indptr = attn_backend.kv_indptr
        self.kv_last_page_len = attn_backend.kv_last_page_len
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self._swa_kv_pool = attn_backend._swa_kv_pool

        # Dispatch the update function
        if self.attn_backend.dispatch_reason == WrapperDispatch.SLIDING_WINDOW:
            self.update = self.update_sliding_window
        elif self.attn_backend.dispatch_reason == WrapperDispatch.CROSS_ATTENTION:
            self.update = self.update_cross_attention
        else:
            assert self.attn_backend.num_wrappers == 1
            self.update = self.update_single_wrapper

    def update(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        seq_lens_sum: int,
        decode_wrappers: List[BatchDecodeWithPagedKVCacheWrapper],
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInput],
        fixed_split_size: Optional[int] = None,
        disable_split_kv: Optional[bool] = None,
    ):
        # Keep the signature for type checking. It will be assigned during runtime.
        raise NotImplementedError()

    def update_single_wrapper(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        seq_lens_sum: int,
        decode_wrappers: List[BatchDecodeWithPagedKVCacheWrapper],
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInput],
        fixed_split_size: Optional[int] = None,
        disable_split_kv: Optional[bool] = None,
    ):
        decode_wrappers = decode_wrappers or self.decode_wrappers
        self.call_begin_forward(
            decode_wrappers[0],
            req_pool_indices,
            seq_lens,
            seq_lens_sum,
            self.kv_indptr[0],
            None,
            spec_info,
            seq_lens_cpu,
            fixed_split_size=fixed_split_size,
            disable_split_kv=disable_split_kv,
        )

    def update_sliding_window(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        seq_lens_sum: int,
        decode_wrappers: List[BatchDecodeWithPagedKVCacheWrapper],
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInput],
        fixed_split_size: Optional[int] = None,
        disable_split_kv: Optional[bool] = None,
    ):
        assert self.sliding_window_size is not None
        for wrapper_id in range(2):
            if wrapper_id == 0:
                # Sliding window attention
                paged_kernel_lens_tmp = torch.clamp(
                    seq_lens, max=self.sliding_window_size + 1
                )
                if seq_lens_cpu is not None:
                    seq_lens_cpu_tmp = torch.clamp(
                        seq_lens_cpu, max=self.sliding_window_size + 1
                    )
                    paged_kernel_lens_sum_tmp = seq_lens_cpu_tmp.sum().item()
                else:
                    paged_kernel_lens_sum_tmp = paged_kernel_lens_tmp.sum().item()
                kv_start_idx_tmp = seq_lens - paged_kernel_lens_tmp
            else:
                # Full attention
                paged_kernel_lens_tmp = seq_lens
                paged_kernel_lens_sum_tmp = seq_lens_sum
                seq_lens_cpu_tmp = seq_lens_cpu
                kv_start_idx_tmp = None

            use_sliding_window_kv_pool = (
                wrapper_id == 0 and self._swa_kv_pool is not None
            )

            self.call_begin_forward(
                decode_wrappers[wrapper_id],
                req_pool_indices,
                paged_kernel_lens_tmp,
                paged_kernel_lens_sum_tmp,
                self.kv_indptr[wrapper_id],
                kv_start_idx_tmp,
                spec_info,
                seq_lens_cpu=seq_lens_cpu_tmp,
                use_sliding_window_kv_pool=use_sliding_window_kv_pool,
                fixed_split_size=fixed_split_size,
                disable_split_kv=disable_split_kv,
            )

    def update_cross_attention(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        seq_lens_sum: int,
        decode_wrappers: List[BatchDecodeWithPagedKVCacheWrapper],
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInput],
        fixed_split_size: Optional[int] = None,
        disable_split_kv: Optional[bool] = None,
    ):
        # Cache encoder_lens on CPU to avoid GPU→CPU transfer per call
        encoder_lens_cpu = encoder_lens.cpu() if encoder_lens is not None else None
        for wrapper_id in range(2):
            if wrapper_id == 0:
                paged_kernel_lens = seq_lens
                kv_start_idx = encoder_lens
                kv_lens_cpu = seq_lens_cpu
            else:
                # Cross-attention: attend to encoder tokens only
                paged_kernel_lens = encoder_lens
                kv_start_idx = torch.zeros_like(encoder_lens)
                seq_lens_sum = encoder_lens.sum().item()
                kv_lens_cpu = encoder_lens_cpu

            self.call_begin_forward(
                decode_wrappers[wrapper_id],
                req_pool_indices,
                paged_kernel_lens,
                seq_lens_sum,
                self.kv_indptr[wrapper_id],
                kv_start_idx,
                spec_info,
                seq_lens_cpu=kv_lens_cpu,
                fixed_split_size=fixed_split_size,
                disable_split_kv=disable_split_kv,
            )

    def call_begin_forward(
        self,
        wrapper: BatchDecodeWithPagedKVCacheWrapper,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        kv_indptr: torch.Tensor,
        kv_start_idx: torch.Tensor,
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
        use_sliding_window_kv_pool: bool = False,
        fixed_split_size: Optional[int] = None,
        disable_split_kv: Optional[bool] = None,
    ):
        # Host-side kv indptr for the DCP cuda-graph replay plan (see below);
        # None on every non-DCP / non-graph path.
        dcp_graph_indptr_host: Optional[torch.Tensor] = None
        if (
            self.attn_backend.uneven_dcp
            and (spec_info is None or getattr(spec_info, "kv_indptr", None) is None)
        ):
            # Uneven-DCP decode: this rank's paged read sees only the token
            # slots it OWNS (even-modulo owner rule pos % dcp_size == dcp_rank,
            # physical slot = global_loc // dcp_size). Build per-rank owned
            # lengths + compacted kv_indices via the DCP index kernel.
            bs = len(req_pool_indices)
            dcp_size = self.attn_backend.dcp_size
            dcp_rank = self.attn_backend.dcp_rank
            if self.attn_backend.uneven_dcp_weighted:
                # WEIGHTED owner rule: owned slots + compact indices from the
                # out_cache_loc (loc % cp_S in [cp_lo, cp_hi)); see
                # _build_dcp_weighted_kv_indices / _dcp_masked_write.
                kv_indptr, kv_indices = _build_dcp_weighted_kv_indices(
                    self.req_to_token,
                    req_pool_indices,
                    paged_kernel_lens,
                    kv_indptr,
                    kv_start_idx,
                    self.attn_backend.cp_S,
                    self.attn_backend.cp_lo,
                    self.attn_backend.cp_hi,
                    self.attn_backend.cp_ratio,
                )
            else:
                dcp_lens = get_dcp_lens(
                    paged_kernel_lens, dcp_size, dcp_rank, kv_start_idx
                )
                kv_indptr[1 : bs + 1] = torch.cumsum(dcp_lens, dim=0)
                kv_indptr = kv_indptr[: bs + 1]
                kv_indices = torch.empty(
                    int(dcp_lens.sum().item()), dtype=torch.int32, device="cuda"
                )
                create_triton_kv_indices_for_dcp_triton[(bs,)](
                    self.req_to_token,
                    req_pool_indices,
                    dcp_lens,
                    kv_indptr,
                    kv_start_idx,
                    kv_indices,
                    self.req_to_token.shape[1],
                    dcp_size,
                    dcp_rank,
                )
            # CUDA-graph REPLAY contract (root cause of the silent decode
            # corruption under uneven-DCP + graphs, all quantizations, M16):
            # the captured decode kernels read the wrapper's FIXED buffers
            # (_paged_kv_indices_buf / _paged_kv_indptr_buf, raw pointers
            # frozen at capture). After capture, begin_forward is
            # fast_decode_plan, which deliberately skips every
            # device-to-device copy into those buffers and assumes the caller
            # wrote them in place -- exactly what the stock non-DCP branch
            # below does by building indices directly in
            # wrapper._paged_kv_indices_buf. This DCP branch builds a FRESH
            # kv_indices tensor instead, so every replay kept reading the
            # STALE capture-time indices -> attention over garbage prompt KV
            # (the "prompt is all '!'" signature). kv_indptr is already
            # written in place (self.kv_indptr[0] is the same storage the
            # wrapper was created on); the indices must be copied explicitly.
            # Additionally, fast_decode_plan's host-side work partition must
            # see the OWNED per-rank lens: the module-global
            # global_override_indptr_cpu built from the FULL seq_lens below
            # would mis-partition split-KV against the owned device indptr.
            if (
                hasattr(wrapper.begin_forward, "func")
                and wrapper.begin_forward.func == fast_decode_plan
            ):
                num_owned = kv_indices.numel()
                assert_buffer_fits(
                    num_owned,
                    wrapper._paged_kv_indices_buf.numel(),
                    "uneven-DCP decode graph kv_indices buffer",
                    bs=bs,
                )
                wrapper._paged_kv_indices_buf[:num_owned].copy_(kv_indices)
                kv_indices = wrapper._paged_kv_indices_buf[:num_owned]
                # Blocking D2H of (bs+1) int32 once per decode step, outside
                # the graph -- negligible against the replay itself.
                dcp_graph_indptr_host = kv_indptr.to("cpu")
        elif spec_info is None or getattr(spec_info, "kv_indptr", None) is None:
            bs = len(req_pool_indices)
            kv_indptr[1 : bs + 1] = torch.cumsum(paged_kernel_lens, dim=0)
            kv_indptr = kv_indptr[: bs + 1]

            if wrapper.is_cuda_graph_enabled:
                # Directly write to the cuda graph input buffer
                kv_indices = wrapper._paged_kv_indices_buf
            else:
                kv_indices = torch.empty(
                    paged_kernel_lens_sum, dtype=torch.int32, device="cuda"
                )

            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                paged_kernel_lens,
                kv_indptr,
                kv_start_idx,
                kv_indices,
                self.req_to_token.shape[1],
            )
        else:
            kv_indptr, kv_indices = spec_info.kv_indptr, spec_info.kv_indices
            bs = kv_indptr.shape[0] - 1

        if use_sliding_window_kv_pool:
            assert self._swa_kv_pool is not None
            kv_last_index = kv_indptr[-1]
            kv_indices[:kv_last_index] = (
                self._swa_kv_pool.translate_loc_from_full_to_swa(
                    kv_indices[:kv_last_index]
                )
            )

        global global_override_indptr_cpu
        locally_override = False
        if seq_lens_cpu is not None and global_override_indptr_cpu is None:
            locally_override = True
            global_override_indptr_cpu = torch.empty_like(kv_indptr, device="cpu")
            global_override_indptr_cpu[0] = 0
            global_override_indptr_cpu[1 : bs + 1] = torch.cumsum(seq_lens_cpu, dim=0)

        # Check if this specific wrapper's begin_forward has been replaced with fast_decode_plan
        # by checking if it's a partial function with fast_decode_plan as the func
        wrapper_uses_fast_decode_plan = (
            hasattr(wrapper.begin_forward, "func")
            and wrapper.begin_forward.func == fast_decode_plan
        )

        if wrapper_uses_fast_decode_plan:
            # When begin_forward is replaced with fast_decode_plan, pass global_override_indptr_cpu
            wrapper.begin_forward(
                kv_indptr,
                kv_indices,
                self.kv_last_page_len[:bs],
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim,
                1,
                data_type=self.data_type,
                q_data_type=self.q_data_type,
                non_blocking=True,
                fixed_split_size=fixed_split_size,
                disable_split_kv=(
                    disable_split_kv if disable_split_kv is not None else False
                ),
                # DCP graph replay: host plan must use the OWNED per-rank
                # indptr, not the full-seq_lens override (see the DCP branch).
                global_override_indptr_cpu=(
                    dcp_graph_indptr_host
                    if dcp_graph_indptr_host is not None
                    else global_override_indptr_cpu
                ),
            )
        else:
            # When using original begin_forward, don't pass global_override_indptr_cpu
            wrapper.begin_forward(
                kv_indptr,
                kv_indices,
                self.kv_last_page_len[:bs],
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim,
                1,
                data_type=self.data_type,
                q_data_type=self.q_data_type,
                non_blocking=True,
                fixed_split_size=fixed_split_size,
                disable_split_kv=(
                    disable_split_kv if disable_split_kv is not None else False
                ),
            )

        if locally_override:
            global_override_indptr_cpu = None


class FlashInferIndicesUpdaterPrefill:
    def __init__(self, model_runner: ModelRunner, attn_backend: FlashInferAttnBackend):
        # Parse Constants
        # THIS rank's real head counts: under uneven TP (--rank-tp-ratio)
        # the split is uneven, and flashinfer's split-KV plan/merge must
        # match the actual per-rank q/kv tensors (see
        # _local_attn_head_counts). Degrades to the even split by default.
        self.num_qo_heads, self.num_kv_heads = _local_attn_head_counts(model_runner)
        self.head_dim = model_runner.model_config.head_dim
        self.data_type = model_runner.kv_cache_dtype
        self.q_data_type = model_runner.dtype
        self.sliding_window_size = model_runner.sliding_window_size
        self.attn_backend = attn_backend

        # Uneven-DCP: the RAGGED wrapper attends the current chunk with this
        # rank's LOCAL (head-sharded) q/kv shards; the PAGED wrapper attends the
        # token-sharded prefix with the FULL gathered q + replicated kv-heads.
        # Keep both head-count pairs so each wrapper is planned correctly.
        self.dcp_local_qo_heads = self.num_qo_heads
        self.dcp_local_kv_heads = self.num_kv_heads
        if attn_backend.uneven_dcp:
            self.num_qo_heads = attn_backend.dcp_full_qo_heads
            self.num_kv_heads = attn_backend.dcp_full_kv_heads

        # Buffers and wrappers
        self.kv_indptr = attn_backend.kv_indptr
        self.kv_last_page_len = attn_backend.kv_last_page_len
        self.qo_indptr = attn_backend.qo_indptr
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self._swa_kv_pool = attn_backend._swa_kv_pool
        self.prefill_wrapper_ragged = attn_backend.prefill_wrapper_ragged

        # Dispatch the update function
        if self.attn_backend.dispatch_reason == WrapperDispatch.SLIDING_WINDOW:
            self.update = self.update_sliding_window
        elif self.attn_backend.dispatch_reason == WrapperDispatch.CROSS_ATTENTION:
            self.update = self.update_cross_attention
        else:
            assert self.attn_backend.num_wrappers == 1
            self.update = self.update_single_wrapper

    def update(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        seq_lens_sum: int,
        prefix_lens: Optional[torch.Tensor],
        prefill_wrappers: List[BatchPrefillWithPagedKVCacheWrapper],
        use_ragged: bool,
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInput],
        fixed_split_size: Optional[int] = None,
        multi_item_params: Optional[MultiItemScoringParams] = None,
        cross_attention_custom_mask: Optional[torch.Tensor] = None,
        extend_prefix_lens_cpu: Optional[List[int]] = None,
    ):
        # Keep the signature for type checking. It will be assigned during runtime.
        raise NotImplementedError()

    def update_single_wrapper(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        seq_lens_sum: int,
        prefix_lens: Optional[torch.Tensor],
        prefill_wrappers: List[BatchPrefillWithPagedKVCacheWrapper],
        use_ragged: bool,
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInput],
        fixed_split_size: Optional[int] = None,
        multi_item_params: Optional[MultiItemScoringParams] = None,
        cross_attention_custom_mask: Optional[torch.Tensor] = None,
        extend_prefix_lens_cpu: Optional[List[int]] = None,
    ):
        if use_ragged:
            assert prefix_lens is not None
            paged_kernel_lens = prefix_lens
            if extend_prefix_lens_cpu is not None:
                # Host-known prefix lens; avoids a per-step D2H sync.
                paged_kernel_lens_sum = sum(extend_prefix_lens_cpu)
            else:
                paged_kernel_lens_sum = paged_kernel_lens.sum().item()
        else:
            paged_kernel_lens = seq_lens
            paged_kernel_lens_sum = seq_lens_sum

        self.call_begin_forward(
            # active_ragged_wrapper: per-bucket graph-mode wrapper under a
            # captured uneven-DCP verify graph, the shared one otherwise.
            self.attn_backend.active_ragged_wrapper,
            prefill_wrappers[0],
            req_pool_indices,
            paged_kernel_lens,
            paged_kernel_lens_sum,
            seq_lens,
            prefix_lens,
            None,
            self.kv_indptr[0],
            self.qo_indptr[0],
            use_ragged,
            spec_info,
            fixed_split_size=fixed_split_size,
            multi_item_params=multi_item_params,
            seq_lens_cpu=seq_lens_cpu,
        )

    def update_sliding_window(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        seq_lens_sum: int,
        prefix_lens: Optional[torch.Tensor],
        prefill_wrappers: List[BatchPrefillWithPagedKVCacheWrapper],
        use_ragged: bool,
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInput],
        fixed_split_size: Optional[int] = None,
        multi_item_params: Optional[MultiItemScoringParams] = None,
        cross_attention_custom_mask: Optional[torch.Tensor] = None,
        extend_prefix_lens_cpu: Optional[List[int]] = None,
    ):
        if prefix_lens is None:
            num_accept_tokens = getattr(spec_info, "num_accept_tokens", None)
            prefix_lens = (
                seq_lens
                if num_accept_tokens is None
                else seq_lens
                - num_accept_tokens[: seq_lens.shape[0]].to(
                    device=seq_lens.device, dtype=seq_lens.dtype
                )
            )
        sliding_window_size = self.sliding_window_size
        assert sliding_window_size is not None
        for wrapper_id in range(2):
            swa_paged_custom_mask = None
            if wrapper_id == 0:
                if use_ragged:
                    # K for extend tokens is written after the paged wrapper runs, so
                    # the paged wrapper sees prefix-only. Trim to the last `window` tokens
                    # (required for SWATokenToKVPoolAllocator; also keeps mask O(window)).
                    effective_start = torch.clamp(
                        prefix_lens - sliding_window_size, min=0
                    )
                    paged_kernel_lens = prefix_lens - effective_start
                    paged_kernel_lens_sum = paged_kernel_lens.sum().item()
                    kv_start_idx = effective_start
                    swa_paged_custom_mask = self._build_swa_prefix_custom_mask(
                        prefix_lens, seq_lens, effective_start
                    )
                else:
                    # window attention use paged only
                    paged_kernel_lens = torch.minimum(
                        seq_lens,
                        sliding_window_size + seq_lens - prefix_lens,
                    )
                    paged_kernel_lens_sum = paged_kernel_lens.sum().item()
                    kv_start_idx = seq_lens - paged_kernel_lens
            else:
                # full attention
                paged_kernel_lens = seq_lens
                paged_kernel_lens_sum = seq_lens_sum
                kv_start_idx = seq_lens - paged_kernel_lens
            use_sliding_window_kv_pool = (
                wrapper_id == 0 and self._swa_kv_pool is not None
            )

            self.call_begin_forward(
                self.prefill_wrapper_ragged,
                prefill_wrappers[wrapper_id],
                req_pool_indices,
                paged_kernel_lens,
                paged_kernel_lens_sum,
                seq_lens,
                prefix_lens,
                kv_start_idx,
                self.kv_indptr[wrapper_id],
                self.qo_indptr[wrapper_id],
                use_ragged,
                spec_info,
                use_sliding_window_kv_pool=use_sliding_window_kv_pool,
                fixed_split_size=fixed_split_size,
                multi_item_params=multi_item_params,
                cross_attention_custom_mask=swa_paged_custom_mask,
            )

    def _build_swa_prefix_custom_mask(
        self,
        prefix_lens: torch.Tensor,
        seq_lens: torch.Tensor,
        kv_start_idx: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Custom SWA mask for the paged wrapper in the ragged merge_state EXTEND path.

        Paged KV covers absolute positions [kv_start_idx[i], prefix_lens[i]).
        Returns None when every key is in-window for every extend query.
        """
        window = self.sliding_window_size
        if window is None or window < 0:
            return None

        prefix_lens_cpu = prefix_lens.detach().cpu().tolist()
        extend_lens_cpu = (seq_lens - prefix_lens).detach().cpu().tolist()
        kv_start_cpu = kv_start_idx.detach().cpu().tolist()
        if all(p == 0 for p in prefix_lens_cpu):
            return None

        device = prefix_lens.device
        mask_parts: List[torch.Tensor] = []
        need_mask = False
        for prefix_len, extend_len, kv_start in zip(
            prefix_lens_cpu, extend_lens_cpu, kv_start_cpu
        ):
            paged_len = int(prefix_len - kv_start)  # = min(prefix_len, window)
            if paged_len == 0 or extend_len == 0:
                continue
            q_abs = torch.arange(extend_len, device=device).view(-1, 1) + prefix_len
            k_abs = torch.arange(paged_len, device=device).view(1, -1) + kv_start
            block = (k_abs >= (q_abs - window)).to(torch.uint8)
            if not bool(block.all()):
                need_mask = True
            mask_parts.append(block.view(-1))

        if not need_mask or not mask_parts:
            return None
        return torch.cat(mask_parts)

    def update_cross_attention(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        seq_lens_sum: int,
        prefix_lens: Optional[torch.Tensor],
        prefill_wrappers: List[BatchPrefillWithPagedKVCacheWrapper],
        use_ragged: bool,
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInput],
        fixed_split_size: Optional[int] = None,
        multi_item_params: Optional[MultiItemScoringParams] = None,
        cross_attention_custom_mask: Optional[torch.Tensor] = None,
        extend_prefix_lens_cpu: Optional[List[int]] = None,
    ):
        for wrapper_id in range(2):
            if wrapper_id == 0:
                # normal attention
                paged_kernel_lens = seq_lens
                kv_start_idx = encoder_lens
                paged_kernel_lens_sum = seq_lens_sum
            else:
                # cross attention
                paged_kernel_lens = encoder_lens
                kv_start_idx = torch.zeros_like(encoder_lens)
                paged_kernel_lens_sum = paged_kernel_lens.sum().item()

            self.call_begin_forward(
                self.prefill_wrapper_ragged,
                prefill_wrappers[wrapper_id],
                req_pool_indices,
                paged_kernel_lens,
                paged_kernel_lens_sum,
                seq_lens,
                prefix_lens,
                kv_start_idx,
                self.kv_indptr[wrapper_id],
                self.qo_indptr[wrapper_id],
                use_ragged,
                spec_info,
                fixed_split_size=fixed_split_size,
                multi_item_params=multi_item_params,
                cross_attention_custom_mask=(
                    cross_attention_custom_mask if wrapper_id == 1 else None
                ),
            )

    def call_begin_forward(
        self,
        wrapper_ragged: BatchPrefillWithRaggedKVCacheWrapper,
        wrapper_paged: BatchPrefillWithPagedKVCacheWrapper,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        seq_lens: torch.Tensor,
        prefix_lens: Optional[torch.Tensor],
        kv_start_idx: torch.Tensor,
        kv_indptr: torch.Tensor,
        qo_indptr: torch.Tensor,
        use_ragged: bool,
        spec_info: Optional[SpecInput],
        use_sliding_window_kv_pool: bool = False,
        fixed_split_size: Optional[int] = None,
        multi_item_params: Optional[MultiItemScoringParams] = None,
        cross_attention_custom_mask: Optional[torch.Tensor] = None,
        seq_lens_cpu: Optional[torch.Tensor] = None,
    ):
        bs = len(seq_lens)
        if spec_info is None and self.attn_backend.uneven_dcp:
            # Uneven-DCP extend: the paged (prefix) wrapper reads only this
            # rank's OWNED prefix token slots (even-modulo owner rule), with the
            # FULL replicated kv-heads. The current-chunk k/v stay local and are
            # attended by the ragged wrapper (local heads) in the forward.
            assert prefix_lens is not None
            assert len(seq_lens) == len(req_pool_indices)
            # DCP ALWAYS splits current-chunk (ragged, local heads) from prefix
            # (paged, token-sharded, full heads), regardless of the incoming
            # use_ragged (this model reports is_multimodal -> use_ragged=False,
            # but DCP still needs the split because the paged prefix read cannot
            # causally mask a sparse owned-slot subset). The paged part covers
            # the PREFIX only; force the ragged plan below.
            use_ragged = True
            dcp_size = self.attn_backend.dcp_size
            dcp_rank = self.attn_backend.dcp_rank
            if self.attn_backend.uneven_dcp_weighted:
                kv_indptr, kv_indices = _build_dcp_weighted_kv_indices(
                    self.req_to_token,
                    req_pool_indices,
                    prefix_lens,
                    kv_indptr,
                    None,
                    self.attn_backend.cp_S,
                    self.attn_backend.cp_lo,
                    self.attn_backend.cp_hi,
                    self.attn_backend.cp_ratio,
                    pad=256,
                )
            else:
                dcp_lens = get_dcp_lens(prefix_lens, dcp_size, dcp_rank, None)
                kv_indptr[1 : bs + 1] = torch.cumsum(dcp_lens, dim=0)
                kv_indptr = kv_indptr[: bs + 1]
                kv_indices = torch.empty(
                    int(dcp_lens.sum().item()) + 256,
                    dtype=torch.int32,
                    device=req_pool_indices.device,
                )
                create_triton_kv_indices_for_dcp_triton[(bs,)](
                    self.req_to_token,
                    req_pool_indices,
                    dcp_lens,
                    kv_indptr,
                    None,
                    kv_indices,
                    self.req_to_token.shape[1],
                    dcp_size,
                    dcp_rank,
                )
            qo_indptr[1 : bs + 1] = torch.cumsum(seq_lens - prefix_lens, dim=0)
            qo_indptr = qo_indptr[: bs + 1]
            custom_mask = cross_attention_custom_mask
        elif (
            self.attn_backend.uneven_dcp
            and spec_info is not None
            and getattr(spec_info, "spec_input_type", None)
            == SpecInputType.EAGLE_VERIFY
        ):
            # M4 (MTP+DCP): target-VERIFY under uneven-weighted DCP. Split the
            # verify into (a) the draft tokens attending the committed PREFIX
            # (paged, token-sharded owned slots, FULL replicated kv-heads,
            # non-causal, cross-rank LSE-combined) and (b) the draft tokens
            # attending EACH OTHER (ragged, LOCAL heads). With
            # --speculative-eagle-topk 1 the draft is a linear CHAIN, so the
            # draft->draft mask is plain causal and the ragged wrapper needs no
            # tree custom_mask (a branching topk>1 tree would require slicing the
            # draft->draft block out of spec_info.custom_mask -- NOT handled
            # here; see _forward_target_verify_dcp). The prefix read excludes the
            # draft tokens (paged_kernel_lens == committed seq_lens); the draft
            # KV is written to owned compact slots by _dcp_masked_write and
            # attended only through the ragged wrapper.
            use_ragged = True
            draft_num = spec_info.draft_token_num
            # Paged prefix (committed context) over this rank's OWNED slots.
            if self.attn_backend.uneven_dcp_weighted:
                kv_indptr, kv_indices = _build_dcp_weighted_kv_indices(
                    self.req_to_token,
                    req_pool_indices,
                    paged_kernel_lens,  # == committed seq_lens (verify prefix)
                    kv_indptr,
                    None,
                    self.attn_backend.cp_S,
                    self.attn_backend.cp_lo,
                    self.attn_backend.cp_hi,
                    self.attn_backend.cp_ratio,
                    pad=256,
                )
            else:
                dcp_size = self.attn_backend.dcp_size
                dcp_rank = self.attn_backend.dcp_rank
                dcp_lens = get_dcp_lens(paged_kernel_lens, dcp_size, dcp_rank, None)
                kv_indptr[1 : bs + 1] = torch.cumsum(dcp_lens, dim=0)
                kv_indptr = kv_indptr[: bs + 1]
                kv_indices = torch.empty(
                    int(dcp_lens.sum().item()) + 256,
                    dtype=torch.int32,
                    device=req_pool_indices.device,
                )
                create_triton_kv_indices_for_dcp_triton[(bs,)](
                    self.req_to_token,
                    req_pool_indices,
                    dcp_lens,
                    kv_indptr,
                    None,
                    kv_indices,
                    self.req_to_token.shape[1],
                    dcp_size,
                    dcp_rank,
                )
            # Each of the draft_num draft tokens is a query row (both for the
            # paged prefix read and the ragged draft->draft attention).
            qo_indptr = torch.arange(
                0,
                (bs + 1) * draft_num,
                step=draft_num,
                dtype=torch.int32,
                device=req_pool_indices.device,
            )
            # Paged prefix read is non-causal (every draft query sees all prefix
            # keys); the causal draft->draft masking is done by the ragged
            # wrapper in the forward.
            custom_mask = None
        elif spec_info is None:
            assert prefix_lens is not None
            assert len(seq_lens) == len(req_pool_indices)
            # Normal extend
            kv_indptr[1 : bs + 1] = torch.cumsum(paged_kernel_lens, dim=0)
            kv_indptr = kv_indptr[: bs + 1]
            kv_indices = torch.empty(
                paged_kernel_lens_sum + 256,
                dtype=torch.int32,
                device=req_pool_indices.device,
            )
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                paged_kernel_lens,
                kv_indptr,
                kv_start_idx,
                kv_indices,
                self.req_to_token.shape[1],
            )
            qo_indptr[1 : bs + 1] = torch.cumsum(seq_lens - prefix_lens, dim=0)
            qo_indptr = qo_indptr[: bs + 1]

            custom_mask = cross_attention_custom_mask
        else:
            assert isinstance(spec_info, SpecInput)
            if spec_info.spec_input_type == SpecInputType.DFLASH_VERIFY:
                kv_indices, kv_indptr, qo_indptr, custom_mask = (
                    spec_info.generate_attn_arg_prefill(
                        req_pool_indices,
                        paged_kernel_lens,
                        paged_kernel_lens_sum,
                        self.req_to_token,
                        kv_start_idx=kv_start_idx,
                    )
                )
            else:
                kv_indices, kv_indptr, qo_indptr, custom_mask = (
                    spec_info.generate_attn_arg_prefill(
                        req_pool_indices,
                        paged_kernel_lens,
                        paged_kernel_lens_sum,
                        self.req_to_token,
                    )
                )

        # extend part
        if use_ragged:
            # Uneven-DCP: the current chunk is attended with this rank's LOCAL
            # head-sharded q/kv (the full replicated kv-head gather happens only
            # for the paged prefix read + the KV write, not the ragged chunk).
            ragged_qo_heads = (
                self.dcp_local_qo_heads
                if self.attn_backend.uneven_dcp
                else self.num_qo_heads
            )
            ragged_kv_heads = (
                self.dcp_local_kv_heads
                if self.attn_backend.uneven_dcp
                else self.num_kv_heads
            )
            wrapper_ragged.begin_forward(
                qo_indptr,
                qo_indptr,
                ragged_qo_heads,
                ragged_kv_heads,
                self.head_dim,
                q_data_type=self.q_data_type,
            )

        if use_sliding_window_kv_pool:
            assert self._swa_kv_pool is not None
            kv_last_index = kv_indptr[-1]
            kv_indices[:kv_last_index] = (
                self._swa_kv_pool.translate_loc_from_full_to_swa(
                    kv_indices[:kv_last_index]
                )
            )

        # cached part
        # Conditionally set multi-item parameters
        if multi_item_params is not None and multi_item_params.is_enabled():
            # Multi-item scoring is active - use specialized parameters and disable generic custom_mask
            use_custom_mask = None
            prefix_len_ptr = multi_item_params.prefix_len_ptr
            token_pos_in_items_ptr = multi_item_params.token_pos_in_items_ptr
            token_pos_in_items_len = multi_item_params.token_pos_in_items_len
            max_item_len_ptr = multi_item_params.max_item_len_ptr
        else:
            # No multi-item scoring - use standard parameters
            use_custom_mask = custom_mask
            prefix_len_ptr = None
            token_pos_in_items_ptr = None
            token_pos_in_items_len = 0
            max_item_len_ptr = None

        # fast_prefill_plan (installed at capture) is sync-free: it needs the
        # host-known qo/kv layout from the caller. Assert rather than silently
        # fall back to plan()'s blocking D2H on the replay hot-path.
        paged_plan_kwargs = {}
        num_tokens_per_req = getattr(spec_info, "num_tokens_per_req", None)
        uses_fast_prefill = (
            hasattr(wrapper_paged.begin_forward, "func")
            and wrapper_paged.begin_forward.func is fast_prefill_plan
        )
        if uses_fast_prefill:
            assert (
                seq_lens_cpu is not None
            ), "fast_prefill_plan replay requires host-known seq_lens_cpu (got None)"
            assert (
                num_tokens_per_req is not None and num_tokens_per_req > 0
            ), f"fast_prefill_plan replay requires num_tokens_per_req > 0 (got {num_tokens_per_req})"
            seq_lens_cpu_i32 = seq_lens_cpu.to(torch.int32)
            qo_indptr_host = torch.arange(
                0,
                (bs + 1) * num_tokens_per_req,
                step=num_tokens_per_req,
                dtype=torch.int32,
                device="cpu",
            )
            kv_indptr_host = torch.zeros(bs + 1, dtype=torch.int32, device="cpu")
            kv_indptr_host[1:] = torch.cumsum(seq_lens_cpu_i32, dim=0)
            paged_plan_kwargs = dict(
                qo_indptr_host=qo_indptr_host,
                kv_indptr_host=kv_indptr_host,
                kv_lens_host=seq_lens_cpu_i32,
                max_q_len=num_tokens_per_req,
                max_kv_len=int(seq_lens_cpu_i32.max()),
            )

        wrapper_paged.begin_forward(
            qo_indptr,
            kv_indptr,
            kv_indices,
            self.kv_last_page_len[:bs],
            self.num_qo_heads,
            self.num_kv_heads,
            self.head_dim,
            1,
            q_data_type=self.q_data_type,
            kv_data_type=self.data_type,
            custom_mask=use_custom_mask,
            non_blocking=True,
            fixed_split_size=fixed_split_size,
            prefix_len_ptr=prefix_len_ptr,
            token_pos_in_items_ptr=token_pos_in_items_ptr,
            token_pos_in_items_len=token_pos_in_items_len,
            max_item_len_ptr=max_item_len_ptr,
            **paged_plan_kwargs,
        )


class FlashInferMultiStepDraftBackend:
    """
    Wrap multiple flashinfer attention backends as one for multiple consecutive
    draft decoding steps.
    """

    def __init__(
        self,
        model_runner: ModelRunner,
        topk: int,
        speculative_num_steps: int,
    ):
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.generate_draft_decode_kv_indices = generate_draft_decode_kv_indices
        self.page_size = model_runner.page_size

        max_bs = _cuda_graph_capture_max_bs(
            model_runner.server_args, model_runner.req_to_token_pool.size * self.topk
        )
        self.kv_indptr = torch.zeros(
            (
                self.speculative_num_steps,
                max_bs + 1,
            ),
            dtype=torch.int32,
            device=model_runner.device,
        )
        self.kv_last_page_len = torch.ones(
            (max_bs,), dtype=torch.int32, device=model_runner.device
        )
        self.attn_backends: List[FlashInferAttnBackend] = []
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends.append(
                FlashInferAttnBackend(
                    model_runner,
                    skip_prefill=True,
                    kv_indptr_buf=self.kv_indptr[i],
                    kv_last_page_len_buf=self.kv_last_page_len,
                )
            )

        self.max_context_len = self.attn_backends[0].max_context_len

        # Cached variables for generate_draft_decode_kv_indices
        self.pool_len = model_runner.req_to_token_pool.req_to_token.shape[1]
        self.req_to_token_pool = model_runner.req_to_token_pool

    def common_template(
        self,
        forward_batch: ForwardBatch,
        kv_indices_buffer: torch.Tensor,
        call_fn: Callable,
    ):
        num_seqs = forward_batch.batch_size
        bs = self.topk * num_seqs
        seq_lens_sum = forward_batch.seq_lens_sum

        required_kv_indices_len = draft_kv_indices_used_len(
            seq_lens_sum, self.topk, bs, self.speculative_num_steps
        )
        assert_buffer_fits(
            required_kv_indices_len,
            kv_indices_buffer.shape[1],
            "EAGLE draft kv_indices row (size max_bs * topk * max_context_len)",
            bs=bs,
            seq_lens_sum=seq_lens_sum,
        )

        self.generate_draft_decode_kv_indices[
            (self.speculative_num_steps, num_seqs, self.topk)
        ](
            forward_batch.req_pool_indices,
            self.req_to_token_pool.req_to_token,
            forward_batch.seq_lens,
            kv_indices_buffer,
            self.kv_indptr,
            forward_batch.positions,
            self.pool_len,
            kv_indices_buffer.shape[1],
            self.kv_indptr.shape[1],
            next_power_of_2(num_seqs),
            next_power_of_2(self.speculative_num_steps),
            next_power_of_2(bs),
            self.page_size,
        )

        assert forward_batch.spec_info is not None
        assert forward_batch.spec_info.is_draft_input()

        # Copy the kv_indptr once to avoid multiple device-to-host copies in flashinfer's plan.
        indptr_cpu_whole = self.kv_indptr[:, : bs + 1].cpu()
        global global_override_indptr_cpu

        for i in range(self.speculative_num_steps - 1):
            forward_batch.spec_info.kv_indptr = self.kv_indptr[i, : bs + 1]
            forward_batch.spec_info.kv_indices = kv_indices_buffer[i][
                : draft_kv_indices_used_len(seq_lens_sum, self.topk, bs, i + 1)
            ]
            global_override_indptr_cpu = indptr_cpu_whole[i]
            call_fn(i, forward_batch)

        global_override_indptr_cpu = None

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        kv_indices_width = draft_kv_indices_buffer_width(
            forward_batch.batch_size, self.topk, self.max_context_len
        )
        kv_indices = torch.empty(
            (self.speculative_num_steps, kv_indices_width),
            dtype=torch.int32,
            device="cuda",
        )

        def call_fn(i, forward_batch):
            forward_batch.spec_info.kv_indptr = (
                forward_batch.spec_info.kv_indptr.clone()
            )
            forward_batch.spec_info.kv_indices = (
                forward_batch.spec_info.kv_indices.clone()
            )
            self.attn_backends[i].init_forward_metadata(forward_batch)

        self.common_template(forward_batch, kv_indices, call_fn)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        # generate_draft_decode_kv_indices packs topk per-branch sequences per row,
        # so the row needs the topk factor -- same as the eager init_forward_metadata
        # (batch_size * topk * max_context_len). Dropping it overflows the buffer.
        kv_indices_width = draft_kv_indices_buffer_width(
            max_bs, self.topk, self.max_context_len
        )
        self.cuda_graph_kv_indices = torch.zeros(
            (self.speculative_num_steps, kv_indices_width),
            dtype=torch.int32,
            device="cuda",
        )

        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_cuda_graph_state(
                max_bs, max_num_tokens, kv_indices_buf=self.cuda_graph_kv_indices[i]
            )

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        from sglang.srt.model_executor.forward_batch_info import build_inner_fb_view

        bs = forward_batch.batch_size

        def call_fn(i, fb):
            inner_fb = build_inner_fb_view(fb, bs=bs, forward_mode=ForwardMode.DECODE)
            self.attn_backends[i].init_forward_metadata_out_graph(
                inner_fb, in_capture=in_capture
            )

        self.common_template(forward_batch, self.cuda_graph_kv_indices, call_fn)

    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch) -> None:
        for attn_backend in self.attn_backends:
            attn_backend.init_forward_metadata_in_graph(forward_batch)


def should_use_tensor_core(
    kv_cache_dtype: torch.dtype,
    num_attention_heads: int,
    num_kv_heads: int,
) -> bool:
    """
    Determine whether to use tensor cores for attention computation.

    Args:
        kv_cache_dtype: Data type of the KV cache
        num_attention_heads: Number of attention heads
        num_kv_heads: Number of key/value heads

    Returns:
        bool: Whether to use tensor cores
    """
    # Try to use environment variable first
    env_override = os.environ.get("SGLANG_FLASHINFER_USE_TENSOR_CORE")
    if env_override is not None:
        return env_override.lower() == "true"

    # Try to use _grouped_size_compiled_for_decode_kernels if available
    # This is for flashinfer <=0.1.6. Otherwise, there is an accuracy bug
    try:
        from flashinfer.decode import _grouped_size_compiled_for_decode_kernels

        if not _grouped_size_compiled_for_decode_kernels(
            num_attention_heads,
            num_kv_heads,
        ):
            return True
        else:
            return False
    except (ImportError, AttributeError):
        pass

    # Calculate GQA group size
    gqa_group_size = num_attention_heads // num_kv_heads

    # For Flashinfer, a GQA group size of at least 4 is needed to efficiently
    # use Tensor Cores, as it fuses the head group with the token dimension in MMA.
    if kv_cache_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return True
    elif kv_cache_dtype in (torch.float16, torch.half, torch.bfloat16):
        return gqa_group_size >= 4
    else:
        return False
