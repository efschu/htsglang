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
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Union

import torch

from sglang.kernel_api_logging import debug_kernel_api
from sglang.srt.distributed.utils import tp_partition_size
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.environ import envs
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.flashinfer_workspace import (
    HIGH_WORKSPACE_ARCHITECTURES,
    WORKSPACE_ARCH_MIB,
    WORKSPACE_DETERMINISTIC_MIB,
)
from sglang.srt.layers.attention.utils import (
    assert_buffer_fits,
    create_flashinfer_kv_indices_triton,
)
from sglang.srt.layers.dcp import (
    build_dcp_weighted_kv_indices,
    cp_all_gather_heads_uneven,
    cp_lse_ag_out_ar_mha_uneven,
    create_triton_kv_indices_for_dcp_triton,
    dcp_even_write_mask,
    dcp_fresh_host_lens,
    dcp_host_even_total,
    dcp_host_lens,
    dcp_host_total_tokens,
    dcp_weighted_write_slots,
    get_dcp_lens,
)
from sglang.srt.layers.dcp.lockstep import (
    dcp_forces_prefix,
    draft_extend_prefix_lens,
    weightless_has_prefix,
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
from sglang.srt.speculative import adaptive_graph_memory
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

# Target-VERIFY spec inputs that take the uneven-DCP verify split in
# call_begin_forward: the committed prefix is read paged over this rank's OWNED
# token slots (non-causal, cross-rank LSE-merged) and the draft tokens attend
# EACH OTHER through the ragged wrapper on this rank's LOCAL heads. Both member
# types present a uniform draft_token_num query block per request and a linear
# (non-tree) draft chain, which is exactly what the split assumes.
_DCP_VERIFY_SPEC_INPUT_TYPES = frozenset(
    {SpecInputType.EAGLE_VERIFY, SpecInputType.DFLASH_VERIFY}
)


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
# init_new_workspace ones, the dedicated full-CG prefill one), grouped BY LANE
# (#274). Within a lane, keyed by id() in a WeakValueDictionary instead of a
# WeakSet: WeakSet membership falls back to elementwise Tensor.__eq__ on hash
# collisions ("Boolean value of Tensor ... is ambiguous"). Weak refs: the
# registry must not extend buffer lifetimes across re-inits.
#
# The lane grouping is what makes the #50 zeroing contract safe next to a
# CONCURRENT dual-group lane. Zeroing is driven by the SERVING group's request
# finish; a process-wide registry means that memset also lands on the
# workspace of a lane that is forwarding at that moment, on another thread and
# another stream, with live split-KV partials in it. Nothing asserts — the
# lane's attention simply returns wrong numbers. Keyed by lane, each group
# restores its OWN boot contract at its OWN job boundary (the lane's is
# DualGroupLane._finish), so #50 is kept rather than traded away.
_WORKSPACE_BUFFERS: Dict[
    Optional[int], "weakref.WeakValueDictionary[int, torch.Tensor]"
] = {}


def _workspace_lane() -> Optional[int]:
    """Registry key of the group in scope: the lane, or ``None`` for the
    serving group and for a SERIAL lane (``scope_lane_id is None``, which
    shares the serving group's workspaces by design and never runs at the same
    time as it).

    ``SGLANG_LANE_SHARED_ATTN_WORKSPACE=1`` collapses every group onto the
    ``None`` bucket, which is verbatim the pre-#274 behaviour: one registry,
    zeroed wholesale from the scheduler thread. It is the falsifier for this
    fix, not an operating mode (see runtime_context._buffers_ignore_lane).
    """
    from sglang.srt.runtime_context import _buffers_ignore_lane, current_lane_id

    return None if _buffers_ignore_lane() else current_lane_id()


def register_flashinfer_workspace_buffer(buf: torch.Tensor) -> torch.Tensor:
    """Track a flashinfer float-workspace allocation for the per-request
    zeroing contract (see zero_flashinfer_workspaces), under the lane in
    scope at allocation time."""
    lane = _workspace_lane()
    bucket = _WORKSPACE_BUFFERS.get(lane)
    if bucket is None:
        bucket = weakref.WeakValueDictionary()
        _WORKSPACE_BUFFERS[lane] = bucket
    bucket[id(buf)] = buf
    if os.environ.get("SGLANG_DEBUG_RUNTIME_BUFFER_POOL", "0") == "1":
        logger.info(
            "flashinfer-workspace-registry: lane=%s ptr=0x%x bytes=%d "
            "lanes_registered=%s",
            lane,
            buf.data_ptr(),
            buf.numel() * buf.element_size(),
            sorted(_WORKSPACE_BUFFERS, key=lambda k: (k is not None, k)),
        )
    return buf


def zero_flashinfer_workspaces() -> int:
    """Zero this group's registered flashinfer float workspaces; returns the
    count.

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

    SCOPED TO THE GROUP IN SCOPE (#274): the caller zeroes what it owns and
    nothing else. On the scheduler thread that is the serving group's set; on
    a concurrent lane's worker thread (DualGroupLane._finish) it is that
    lane's. Zeroing another group's workspace while it forwards is silent
    corruption, not a race that announces itself — see _WORKSPACE_BUFFERS.
    """
    from sglang.srt.speculative.adaptive_graph_memory import is_paused_tensor

    bucket = _WORKSPACE_BUFFERS.get(_workspace_lane())
    if bucket is None:
        return 0

    n = 0
    for buf in list(bucket.values()):
        # A workspace belonging to a paused adaptive runtime state is
        # physically unmapped; touching it would fault. Its zero-contract is
        # restored by the adaptive graph-memory manager at resume time
        # instead (see adaptive_graph_memory.AdaptiveGraphMemoryManager).
        if is_paused_tensor(buf):
            continue
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


def _tag_adaptive_int_workspace(wrapper, share_key=None):
    """Stage-2 adaptive graph-memory offload: swap *wrapper*'s 8 MiB int
    workspace for a state-private, pauseable one (via the public
    ``reset_workspace_buffer`` API, before the wrapper's first plan).

    Safe to pause because the int workspace carries no state a forward does
    not rewrite: flashinfer re-plans on every forward (host plan state ->
    pinned staging buffer -> device int-workspace copy), and its boot
    contract is fresh-cudaMalloc zero pages (#50 note), which the
    zero-on-resume of noted tensors restores exactly. The pinned host
    staging buffer is untouched by pause/resume (host memory).

    *share_key*: graph-mode wrappers are instantiated once PER CAPTURED
    BATCH BUCKET, but within one (backend, role, wrapper-slot) only the
    active bucket's wrapper is planned and replayed in any forward -- the
    buckets are mutually exclusive, and each forward's plan rewrites the
    workspace immediately before its graph reads it. All buckets of one
    slot therefore share a single tagged workspace (keyed per state build),
    cutting the per-state int-ws footprint ~4x (k5: 800 -> ~200 MiB), which
    is what lets the high-accept set meet the serving-margin check at the
    standard reserve. Wrappers that can be live in the same forward (init
    wrappers of different roles, different draft steps = different backend
    instances) must NOT share -- callers pass share_key=None or distinct
    keys for those.

    No-op outside a Stage-2 adaptive build scope, so the static path and
    Stage-1 offload keep their historical resident int workspaces.
    """
    if not adaptive_graph_memory.in_capture_offload_build():
        return wrapper
    old = getattr(wrapper, "_int_workspace_buffer", None)
    if old is None or not torch.is_tensor(old):
        return wrapper
    nbytes = old.untyped_storage().nbytes()
    if nbytes < adaptive_graph_memory.MIN_TAGGED_BYTES:
        return wrapper
    mgr = adaptive_graph_memory._ACTIVE_MANAGER
    if share_key is not None:
        shared = mgr.get_shared_state_tensor(share_key)
        if shared is not None and shared.untyped_storage().nbytes() == nbytes:
            wrapper.reset_workspace_buffer(wrapper._float_workspace_buffer, shared)
            return wrapper
    with adaptive_graph_memory.tagged_state_alloc(nbytes=nbytes):
        new_int = torch.zeros_like(old)
    wrapper.reset_workspace_buffer(wrapper._float_workspace_buffer, new_int)
    adaptive_graph_memory.note_state_tensor(new_int, kind="int_ws")
    if share_key is not None:
        mgr.put_shared_state_tensor(share_key, new_int)
    return wrapper


def reject_silently_inert_dcp(
    dcp_size: int,
    *,
    uneven_dcp: bool,
    draft_pool_replicated: bool,
) -> None:
    """Refuse a --dcp-size that this backend would silently ignore.

    Every DCP branch in FlashInferAttnBackend is gated on ``self.uneven_dcp``
    (an installed --rank-tp-ratio plan with dcp_size == tp_size, i.e. the
    replicated-KV token-sharded pool, or the weightless-KV fast lane). Upstream
    flashinfer has NO DCP path at all -- the pre-fork file contains zero ``dcp``
    references -- so with the predicate false the backend does not degrade to a
    slower DCP, it runs stock full-KV attention: no token sharding, no owner
    rule, no LSE merge, and no context-capacity gain. The user asked for
    decode context parallelism and got plain TP.

    Measured (Qwen3.5-2B, TP=2/DCP=2, SGLANG_UNEVEN_TOKEN_VECTOR=2,1, no
    plan): boots green, output token-identical to TP=1, ZERO uneven-machinery
    log lines. A configured-looking server doing nothing that was asked -- the
    silent-no-op half of the same family as a silent wrong answer, and the
    reason the token-vector-without-a-plan case was already made a hard reject
    in resolve_cp_token_ratios.

    A REPLICATED-POOL DRAFT WORKER IS EXEMPT, BY DESIGN, NOT BY OVERSIGHT (M4):
    dcp_size is a property of the parallel context, so an EAGLE/NEXTN draft
    runner sees dcp_size > 1 too, and under the default
    ``--draft-kv-layout replicated`` it deliberately does not token-shard its
    tiny 1-layer full-context KV pool (it runs as a plain uneven-TP model). Its
    uneven_dcp is forced False for that reason a few lines below, and rejecting
    it here would refuse the validated MTP + uneven-DCP arm.

    The parameter is ``draft_pool_replicated``, not "is a draft worker": under
    ``--draft-kv-layout dcp`` (#108) the draft runner IS on the DCP machinery,
    so the silent-no-op argument applies to it exactly as to the target and it
    must NOT be exempt. Callers pass
    ``layers.dcp.owner.draft_pool_is_replicated(...)``.

    Pure function of the decision inputs so the rule is testable on CPU.
    """
    if dcp_size <= 1 or uneven_dcp or draft_pool_replicated:
        return
    raise ValueError(
        f"--dcp-size {dcp_size} with --attention-backend flashinfer would be "
        f"SILENTLY IGNORED: this backend's decode-context-parallel machinery "
        f"is gated on the uneven-DCP replicated-KV geometry (a "
        f"--rank-tp-ratio shard plan with dcp_size == tp_size, or "
        f"--weightless-kv-fastlane), and neither is active here. Upstream "
        f"flashinfer has no plain even-DCP path, so the server would boot, "
        f"answer correctly, and give you exactly the plain-TP behaviour you "
        f"would get without the flag -- no token sharding and no extra "
        f"context capacity. Pass --rank-tp-ratio (uneven DCP), use "
        f"--attention-backend triton, which implements the even-modulo DCP "
        f"path, or drop --dcp-size."
    )


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
            weightless_head_counts,
            weightless_kv_active,
        )

        self.dcp_size = get_parallel().attn_dcp_size
        self.dcp_rank = get_parallel().attn_dcp_rank
        # Weightless-KV fast lane (Variant C Stage 1): one head rank holds ALL
        # heads/weights, the other DCP ranks are weightless (0 heads, KV-token-
        # shard only). It forces the DCP decode path (the Q all-gather becomes a
        # broadcast from the head rank, the O merge slices back to the head rank
        # only) even though no --rank-tp-ratio weight vector is installed.
        self.weightless_kv = weightless_kv_active() and not getattr(
            model_runner, "is_draft_worker", False
        )
        # M4 (MTP+DCP): the DRAFT worker (EAGLE/NEXTN) does NOT token-shard its
        # tiny 1-layer KV pool BY DEFAULT (--draft-kv-layout replicated). It runs
        # as a plain uneven-TP model -- head-sharded kv (whole GQA groups per
        # rank, [2,1,1]) with the FULL token context resident on every rank -- so
        # its draft/verify KV indices never need the weighted DCP owner rule.
        # Only the TARGET model uses DCP token-sharding. This is the
        # coordinator's "keep the draft heads local, not DCP-split"
        # simplification (minimal variant: head-sharded across ranks, reusing the
        # existing uneven-TP weight loading, rather than whole-on-one-card).
        #
        # #108 (--draft-kv-layout dcp): opt the DRAFT pool into token-sharding
        # with the SAME weighted owner rule + replicated-kv-heads + LSE merge as
        # the target. The draft worker then takes the identical uneven_dcp path
        # below -- no draft-specific branch anywhere past this line, which is
        # the whole point: the machinery is reused, not re-derived. Gated OFF by
        # default so the replicated path stays bit-identical.
        #
        # draft_pool_is_replicated() is shared with
        # model_runner_kv_cache_mixin's pool sizing; the two must never disagree
        # (pool shaped for a split the backend does not perform is #345's
        # silent right-token/wrong-slot corruption, not a crash).
        from sglang.srt.layers.dcp.owner import draft_pool_is_replicated

        # The phase-flip TP stack (#631) rides is_draft_worker for the
        # secondary-runner gates but is a TARGET-shaped model on the DCP
        # machinery: forcing its uneven_dcp off via the draft-replicated
        # exemption routed local-head KV writes into the full-width
        # token-sharded pool (store_cache row mismatch, first real-metal
        # flip boot 2026-08-08). Same is_draft_worker-ride family as the
        # attention-registry MTP shortcut.
        _is_draft = bool(getattr(model_runner, "is_draft_worker", False)) and not bool(
            getattr(model_runner, "is_phase_flip_tp_stack", False)
        )
        _draft_replicated = draft_pool_is_replicated(
            _is_draft, getattr(model_runner, "server_args", None)
        )
        self.uneven_dcp = (
            uneven_dcp_kv_replicated(self.dcp_size) or self.weightless_kv
        ) and not _draft_replicated
        # A --dcp-size this backend would silently ignore is refused here
        # rather than served as plain TP under a DCP-looking config. See
        # reject_silently_inert_dcp. A DCP-sharded draft worker (#108) is NOT
        # exempt: it is on the machinery, so the silent-no-op argument applies
        # to it exactly as to the target.
        reject_silently_inert_dcp(
            self.dcp_size,
            uneven_dcp=self.uneven_dcp,
            draft_pool_replicated=_draft_replicated,
        )
        # WEIGHTED owner rule (SGLANG_UNEVEN_DCP_WEIGHTED): a non-uniform token
        # vector is installed, so this rank owns the contiguous virtual-block
        # offset range [cp_lo, cp_hi) of every block of cp_S slots (instead of
        # the even modulo residue == dcp_rank). Physical (compact) slot of an
        # out_cache_loc L owned by this rank is
        #   (L // cp_S) * cp_ratio + (L % cp_S - cp_lo)
        # which is an injective, ratio-proportional packing (reduces EXACTLY to
        # the even L // dcp_size when ratios are all 1). False -> even modulo.
        self.uneven_dcp_weighted = self.uneven_dcp and uneven_dcp_active(self.dcp_size)
        if self.uneven_dcp_weighted:
            _cp_prefix = cp_token_prefix(self.dcp_size)
            self.cp_S = _cp_prefix[-1]
            self.cp_lo = _cp_prefix[self.dcp_rank]
            self.cp_hi = _cp_prefix[self.dcp_rank + 1]
            self.cp_ratio = self.cp_hi - self.cp_lo
            # #297: these four fields are an init-time SNAPSHOT of the token
            # vector. A phase-boundary KV reshard installs a new vector at
            # runtime; registering makes this instance reachable for the
            # cutover's refresh (weak ref -- swapped-out per-rung backends
            # are not kept alive).
            from sglang.srt.layers.dcp.owner import register_owner_bounds_consumer

            register_owner_bounds_consumer(self)
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
            if self.weightless_kv:
                # Weightless-KV fast lane: ALL heads live on the head rank
                # (full weights, TP=1); every other rank is weightless (0
                # heads). kv-heads are NOT replicated here -- only the head
                # rank projects K,V, so _dcp_masked_write broadcasts the new
                # token's K,V from the head rank via the [total_kv,0,0]
                # all-gather (the weightless ranks pass an empty [T,0,D] shard).
                self.dcp_kv_replicated_heads = False
                self.dcp_q_head_counts = weightless_head_counts(
                    mc.num_attention_heads, attn_tp_size
                )
                self.dcp_kv_head_counts = weightless_head_counts(total_kv, attn_tp_size)
                # Compute dtype for the weightless worker's empty [T,0,D]
                # contributions -- must match the head rank's projected Q/K/V
                # dtype so the padded all-gather shapes/dtypes agree.
                self._wl_dtype = mc.dtype
                # Stage B0: block-decode staging size (0 = OFF = monolithic,
                # byte-identical). >0 restructures this rank's intra-shard decode
                # attention into a block loop online-merged with merge_state,
                # BEFORE the cross-rank cp_lse. All KV resident in B0. See
                # _blockwise_decode_return_lse. Strictly scoped to this lane.
                self._wl_chunk_block_size = int(
                    getattr(
                        model_runner.server_args,
                        "weightless_kv_chunked_block_size",
                        0,
                    )
                )
                # Lazily-created persistent block decode wrapper (its own
                # workspace so re-planning per block never clobbers the main
                # decode/prefill wrappers' scheduling metadata).
                self._wl_block_wrapper = None
                # --- #136a: CUDA-graph streaming block-decode state ----------
                # _wl_graph_ladder: sorted list of captured block-count rungs
                #   (set by the decode graph runner before capture).
                # _wl_graph_capture_blocks: rung currently being CAPTURED (the
                #   forward dispatch routes to the fixed-count graph block loop
                #   iff this is not None).
                # _wl_graph_replay_blocks: rung chosen for the CURRENT replay
                #   step (set by wl_graph_can_replay, consumed by the
                #   out-of-graph prep + the runner's graph key).
                # _wl_graph_state: bs bucket -> per-bucket persistent wrappers
                #   and fixed index/staging buffers (shared by all rungs).
                self._wl_graph_ladder = []
                self._wl_graph_capture_blocks = None
                self._wl_graph_replay_blocks = None
                self._wl_graph_state = {}
                self._wl_graph_fallback_logged = False
                self._wl_graph_loc_offset = 0
                # Stage B1: host-spill tier. Static slot->tier map: compacted
                # slot s < _wl_dev_slots is device-resident (identity);
                # s >= _wl_dev_slots lives on host at s - _wl_dev_slots and is
                # streamed H2D through the staging region
                # [_wl_stage_base, _wl_stage_base + _wl_stage_cap) of the
                # device pool by _wl_stage_block. Deterministic + rank-uniform
                # by construction (a pure function of the broadcast slot ids);
                # all moves are rank-local memcpys on the current stream -- the
                # cross-rank collective count is untouched.
                self._wl_spill_active = (
                    bool(getattr(model_runner, "_wl_spill_phys_tokens", 0))
                    and not model_runner.is_draft_worker
                )
                if self._wl_spill_active:
                    assert self._wl_chunk_block_size > 0, (
                        "weightless-KV host spill requires the B0 block loop "
                        "(--weightless-kv-chunked-block-size > 0)"
                    )
                    self._wl_dev_slots = int(model_runner._wl_spill_device_tokens)
                    self._wl_stage_base = self._wl_dev_slots
                    self._wl_stage_cap = int(model_runner._wl_spill_staging_tokens)
                    # #136b: H2D prefetch/double-buffer is enabled iff the
                    # model runner carved TWO block-sized staging regions plus
                    # the owner-write scratch row (SGLANG_WL_H2D_PREFETCH; the
                    # carve is the single source of truth -- a single-block
                    # carve means the serial-copy #136a behavior everywhere).
                    self._wl_prefetch = (
                        self._wl_stage_cap >= 2 * self._wl_chunk_block_size + 1
                    )
                    self._wl_host_slots = int(model_runner._wl_spill_host_tokens)
                    self._wl_host_pool = getattr(
                        model_runner, "wl_spill_host_pool", None
                    )
                    assert self._wl_host_pool is not None, (
                        "weightless-KV host spill active but no host tier was "
                        "attached to the full-attention KV pool"
                    )
                    self._wl_full_pool = model_runner.token_to_kv_pool.full_kv_pool
                    # Lazily-created persistent block PREFILL wrapper for the
                    # streamed committed-prefix attention in extend (mirrors
                    # _wl_block_wrapper for decode).
                    self._wl_block_prefill_wrapper = None
            else:
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

        # #128 DCP collective/compute overlap: issue the per-layer DCP
        # collectives (kv-head gathers, q-head gather, LSE-merge) on a
        # dedicated comm stream so the INDEPENDENT adjacent compute (the
        # ragged current-chunk attention, the masked scatter-write, the
        # elementwise merge prep) runs concurrently on the main stream.
        # SCHEDULING-ONLY: the collective issue order (A_k, A_v, B, C, D per
        # layer) and every reduction/merge is unchanged -> byte-identical to
        # the sequential baseline (verified machine-zero in BOTH eager and
        # captured-graph mode, plus self-determinism).
        #
        # DEFAULT-OFF (mode 0). The overlap is byte-identical but PERF-NEUTRAL
        # on a compute/PCIe-saturated rig at small batch: the DCP collectives
        # sit on the critical path and there is no spare independent compute to
        # hide them behind, so side-streaming them cannot shorten a saturated
        # critical path (measured: prefill + decode parity on 5090+2x3080, TP=3
        # bs=1). A perf-neutral change must not alter the default execution
        # path, so the default under uneven-DCP is the ORIGINAL sequential
        # scheduling (mode 0). Set SGLANG_DCP_COMM_OVERLAP=1 to opt IN to the
        # dual-stream overlap for regimes where the balance favors it (larger
        # batch, faster interconnect, or a less-saturated / offloaded config
        # where the hidden compute is comparable to the collective duration).
        # The stream is leased HERE (init time, never inside cuda-graph capture
        # -- stream creation is a driver call, see RuntimeContext.get_stream).
        self.dcp_comm_stream = None
        # Diagnostic mode "2": run the OVERLAP issue order but keep everything
        # on the main stream (no side stream, no cross-stream edges). Isolates
        # "reorder is not order-equivalent" from "stream mechanics" bugs.
        self.dcp_overlap_reorder_only = False
        # Diagnostic mode "3": baseline order with ONLY the scatter-write
        # deferred past the paged prefix read + merge (single stream).
        self.dcp_overlap_scatter_late = False
        # Default "0" = original fully-sequential scheduling (unchanged default
        # execution path). "1" = dual-stream overlap (opt-in). "2"/"3" diag.
        _dcp_overlap_mode = os.environ.get("SGLANG_DCP_COMM_OVERLAP", "0")
        if self.uneven_dcp and _dcp_overlap_mode in ("1", "2"):
            from sglang.srt.runtime_context import get_stream as _get_named_stream

            self.dcp_comm_stream = _get_named_stream("dcp_comm")
            self.dcp_overlap_reorder_only = _dcp_overlap_mode == "2"
        elif self.uneven_dcp and _dcp_overlap_mode == "3":
            self.dcp_overlap_scatter_late = True
        # State the resolved mode. An A/B over this variable is otherwise
        # unfalsifiable: the mode is read once at backend init, and nothing
        # downstream announces it, so an arm that failed to bind (variable
        # unexported to the scheduler processes, or set after init) is
        # indistinguishable from one that bound and did nothing. Logged for
        # mode 0 too -- the control arm needs the same proof as the treatment.
        if self.uneven_dcp:
            logger.info(
                "DCP collective/compute overlap: mode %s (%s)",
                _dcp_overlap_mode,
                {
                    "0": "sequential baseline, default",
                    "1": "dual-stream overlap",
                    "2": "diagnostic: overlap issue order, single stream",
                    "3": "diagnostic: late scatter-write, single stream",
                }.get(_dcp_overlap_mode, "unrecognized -- treated as baseline"),
            )

        # Tree/branching speculative decoding (--speculative-eagle-topk > 1)
        # under uneven-DCP. With topk == 1 the draft is a linear CHAIN and the
        # ragged draft->draft verify attention is plain CAUSAL (no mask). With
        # topk > 1 the draft forms a TREE, so the draft->draft block must carry
        # the tree-topology mask. That mask acts ONLY on the local, replicated
        # draft tokens (every rank holds all draft-token activations), so it is
        # a rank-uniform local computation -- it never touches the token-sharded
        # committed prefix (paged, non-causal, LSE-merged across ranks). See
        # call_begin_forward (EAGLE_VERIFY) + _get_verify_ragged_cg_wrapper.
        _sa = model_runner.server_args
        self.dcp_tree_mask = bool(
            self.uneven_dcp
            and getattr(_sa, "speculative_eagle_topk", None) is not None
            and _sa.speculative_eagle_topk > 1
        )
        # #76/#139 defensive check: every server_args configuration that can
        # make self.uneven_dcp True here (--rank-tp-ratio uneven DCP, even-
        # modulo or weighted, and --weightless-kv-fastlane) hard-errors on
        # topk > 1 at arg-validation time (ServerArgs._handle_dcp_validation +
        # _handle_weightless_kv_fastlane), because the tree-mask verify below
        # is unvalidated (non-deterministic, non-greedy vs the topk=1 oracle).
        # If a FUTURE DCP variant flips uneven_dcp on without adding a matching
        # server_args guard, refuse here instead of silently emitting wrong
        # tokens. Remove this together with those guards only after
        # _build_dcp_ragged_tree_mask's ancestor semantics are audited (#76).
        if self.dcp_tree_mask:
            raise RuntimeError(
                "#76 guard hole: --speculative-eagle-topk="
                f"{_sa.speculative_eagle_topk} > 1 reached the DCP tree-mask "
                "activation (uneven_dcp=True) without being rejected by the "
                "server_args guards. This uneven-DCP variant must add a "
                "matching topk>1 hard error in ServerArgs; the tree-masked "
                "draft->draft verify is unvalidated on any uneven-DCP path."
            )
        self.dcp_draft_token_num = int(
            getattr(_sa, "speculative_num_draft_tokens", 0) or 0
        )

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

        # Qwen2/Qwen3 models require higher flashinfer workspace size.
        # The rule lives in flashinfer_workspace so that things which must know
        # the size BEFORE this backend exists (the VRAM ledger sizing a card,
        # the planner predicting capacity) apply the same one instead of
        # reading the raw env var and getting the pre-rewrite value.
        if (
            set(model_runner.model_config.hf_config.architectures)
            & HIGH_WORKSPACE_ARCHITECTURES
        ):
            envs.SGLANG_FLASHINFER_WORKSPACE_SIZE.set(WORKSPACE_ARCH_MIB * 1024 * 1024)

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
            # Overrides the architecture bump above; see flashinfer_workspace,
            # which encodes that precedence once for every reader.
            envs.SGLANG_FLASHINFER_WORKSPACE_SIZE.set(
                WORKSPACE_DETERMINISTIC_MIB * 1024 * 1024
            )

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
            # When built for an adaptive runtime state in offload mode, the
            # private workspace is tagged as pauseable per-state scratch: its
            # physical pages are unmapped while the state is inactive and it
            # is zeroed on every resume (restoring the #50 boot contract).
            with adaptive_graph_memory.tagged_state_alloc(
                nbytes=envs.SGLANG_FLASHINFER_WORKSPACE_SIZE.get()
            ):
                new_workspace_buffer = torch.empty(
                    envs.SGLANG_FLASHINFER_WORKSPACE_SIZE.get(),
                    dtype=torch.uint8,
                    device=model_runner.device,
                )
            adaptive_graph_memory.note_state_tensor(new_workspace_buffer)
            self.workspace_buffer = register_flashinfer_workspace_buffer(
                new_workspace_buffer
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
        self.prefill_wrapper_ragged = _tag_adaptive_int_workspace(
            BatchPrefillWithRaggedKVCacheWrapper(
                self.workspace_buffer, "NHD", backend=fmha_backend
            )
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
                    _tag_adaptive_int_workspace(
                        BatchPrefillWithPagedKVCacheWrapper(
                            self.workspace_buffer,
                            "NHD",
                            backend=self.prefill_backend,
                        )
                    )
                )
                self.prefill_wrappers_verify.append(
                    _tag_adaptive_int_workspace(
                        BatchPrefillWithPagedKVCacheWrapper(
                            self.workspace_buffer,
                            "NHD",
                            backend=self.prefill_backend,
                        )
                    )
                )
            self.decode_wrappers.append(
                _tag_adaptive_int_workspace(
                    BatchDecodeWithPagedKVCacheWrapper(
                        self.workspace_buffer,
                        "NHD",
                        backend=self.decode_backend,
                        use_tensor_cores=self.decode_use_tensor_cores,
                    )
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
        # #108 slice 2: the SAME problem for the DCP draft-extend graph, and a
        # SEPARATE dict on purpose. Draft-extend and verify are captured at the
        # same bs values but are different graphs with different qo strides
        # (num_tokens_per_req vs draft_token_num); one wrapper per bs shared
        # between them would have two captured graphs freezing the pointers of
        # one set of fixed buffers, and flashinfer latches _max_total_num_rows
        # on a wrapper's first plan (the #274 round-7a failure, one level up).
        self.draft_extend_ragged_cg_wrappers: dict = {}
        self._ragged_wrapper_override = None
        # Plain EXTEND under full prefill CUDA graph: one wrapper set
        # shared across all captured num_tokens buckets (bs fixed at 1).
        # Created lazily on first capture in _prepare_cuda_graph_metadata.
        self.full_cg_prefill_wrappers: Optional[
            List[BatchPrefillWithPagedKVCacheWrapper]
        ] = None

        # ---- kv-session-offload (S1) ------------------------------------
        # Per-step state of the spill tick (None on every other forward --
        # the ONLY dispatch key; unset means every path below is untouched).
        self._sess_spill = None
        # PS2 (deep prefill-spill): per-forward state of a born-spilled EXTEND
        # (None on every other forward -- same dispatch discipline as
        # _sess_spill, so flag OFF / non-PS2 forwards run byte-identically).
        self._sess_prefill_spill = None
        self._sess_enabled = bool(
            getattr(model_runner.server_args, "enable_kv_session_offload", False)
        ) and not getattr(model_runner, "is_draft_worker", False)
        if self._sess_enabled:
            self._sess_wire(model_runner)

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

    def refresh_dcp_owner_bounds(self) -> None:
        """#297 cutover hook: re-derive the cached weighted owner bounds
        (cp_S/cp_lo/cp_hi/cp_ratio) from the freshly installed token vector.
        Called by ``refresh_all_owner_bounds`` after a phase-boundary KV
        reshard, while the scheduler is idle -- never mid-forward. Everything
        else this backend derives from the bounds is computed per batch, so
        the next forward pass sees the new ownership consistently."""
        if not self.uneven_dcp_weighted:
            return
        from sglang.srt.layers.dcp.owner import dcp_weighted_owner_bounds

        (
            self.cp_S,
            self.cp_lo,
            self.cp_hi,
            self.cp_ratio,
        ) = dcp_weighted_owner_bounds(self.dcp_size, self.dcp_rank)

    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch) -> None:
        """IN-graph hook, recorded at the START of every captured decode body
        (run_once on the head runner and the weightless worker alike; a no-op
        for the plain flashinfer decode). #136b: fork the H2D prefetch side
        stream ONCE per step -- the side stream's copy pipeline for ALL layers
        descends from this point, so a layer's host-block copies can run while
        earlier layers still compute on the main stream (staging rows are
        per-layer; the only cross edges are the per-layer ow_ev/run_ev waits
        in `_wl_blockwise_decode_return_lse_graph`)."""
        if getattr(self, "_wl_graph_capture_blocks", None) is None:
            return
        if not getattr(self, "_wl_spill_active", False):
            return
        st = self._wl_graph_state.get(self._wl_graph_active_bucket)
        rung = self._wl_graph_capture_blocks
        if (
            st is None
            or not getattr(st, "prefetch", False)
            or not any(st.stage_cnt[j] > 0 for j in range(rung))
        ):
            return
        st.ow_recorded = False
        st.fork_ev.record(torch.cuda.current_stream())
        st.copy_stream.wait_event(st.fork_ev)

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        # S5 spill-tick graph: out-of-graph prep for the capture / replay of a
        # spill tick (PORT of the _wl out-graph -> _wl_graph_prepare_blocks
        # path). During the spill capture pass `_sess_capture_active` tags the
        # synthetic batch as a spill tick at the rung worst case; on replay the
        # spill flag + admitted rung are already set. Build the per-step state
        # (st + graph_plan) and plan/refill the fixed graph buffers, then
        # return (the block wrappers are on the graph bucket, not the decode
        # metadata). GPU-JUSTIFICATION: whether ALL of prepare must run here vs
        # partly in the captured body, and the exact seq_lens the synthetic
        # tick carries, are validated on GPU.
        if getattr(self, "_sess_capture_active", False) or (
            getattr(forward_batch, "kv_session_spill_tick", False)
            and getattr(self, "_sess_graph_replay_blocks", None) is not None
        ):
            if getattr(self, "_sess_capture_active", False):
                # Point the synthetic batch at the reserved capture row + a
                # host seq len that yields the capture rung.
                rpi = self._sess_capture_rpi
                L = min(
                    int(self._sess_graph_capture_blocks) * self._sess_block_size,
                    self._sess_region_tokens,
                )
                forward_batch.kv_session_spill_tick = True
                forward_batch.req_pool_indices[:1] = rpi
                if forward_batch.seq_lens_cpu is not None:
                    forward_batch.seq_lens_cpu[:1] = L
            # Derive st (+ graph_plan) and plan/refill the graph buffers.
            self._sess_spill = self._sess_prepare_step(forward_batch)
            return
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

        # #274 round 7a: the verify wrapper key, computed on BOTH paths (it
        # used to be needed only at capture). See _verify_cg_key.
        verify_key = self._verify_cg_key(bs, forward_mode, spec_info)
        if in_capture:
            num_tokens = forward_batch.positions.numel()
            self._prepare_cuda_graph_metadata(bs, num_tokens, forward_mode, spec_info)

        # Uneven-DCP target-verify graphs run the ragged draft->draft attention
        # inside the capture; that requires the per-bucket graph-mode ragged
        # wrapper (fixed indptr buffers). All other modes use the shared one.
        # #108 slice 2: the DCP draft-extend graph runs its current-chunk ragged
        # attention inside the capture for the same reason, so it needs the same
        # treatment -- from its OWN per-bucket dict (different qo stride).
        if self.uneven_dcp and forward_mode.is_target_verify():
            self._ragged_wrapper_override = self._get_verify_ragged_cg_wrapper(bs)
        elif self.uneven_dcp and forward_mode.is_draft_extend_v2():
            self._ragged_wrapper_override = self._get_draft_extend_ragged_cg_wrapper(bs)
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
                prefill_wrappers=self.prefill_cuda_graph_metadata[verify_key],
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
                prefill_wrappers=self.prefill_cuda_graph_metadata[verify_key],
                use_ragged=not self.use_paged,
                encoder_lens=encoder_lens[:bs] if encoder_lens is not None else None,
                spec_info=None,
                # #3287: spec_info=None + uneven_dcp is the branch that indexes
                # over prefix_lens, so without a mirror it takes the unbounded
                # blocking D2H. The mirror must describe THIS callsite's
                # prefix_lens, which is seq_lens - block_size and NOT
                # forward_batch.extend_prefix_lens -- forwarding the latter
                # would size the index buffer from a different vector, a silent
                # mis-size rather than a stall. Derived by the same subtraction
                # on the host, immediately below the device expression, so the
                # two cannot drift; this is also why the dLLM site needs no new
                # field on the decode replay view. Taken over [:bs] because the
                # builder consumes bs rows.
                extend_prefix_lens_cpu=(
                    None
                    if seq_lens_cpu is None
                    else [
                        int(s) - self.dllm_config.block_size for s in seq_lens_cpu[:bs]
                    ]
                ),
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
                # #3287: same branch, same missing mirror. Here prefix_lens IS
                # forward_batch.extend_prefix_lens, so its own mirror is the
                # right one. It is NOT padded to the graph's slot count while
                # the device vector is, and it does not need to be: both
                # consumers take only a SUM, and replay_prepare has already
                # zeroed the device tail, so the two sums are equal. Padding it
                # would be harmless but would also imply a shape contract that
                # does not exist on this arm.
                extend_prefix_lens_cpu=forward_batch.extend_prefix_lens_cpu,
            )
        else:
            raise ValueError("Invalid forward mode")

        if in_capture and forward_mode.is_decode_or_idle():
            # fast_decode_plan needs _cached_module from the initial begin_forward
            # above, so install it only after that first plan has run.
            for w in self.decode_cuda_graph_metadata[bs]:
                w.begin_forward = partial(fast_decode_plan, w)

        # #136a: weightless streaming block-decode -- per-step OUT-of-graph
        # prep (block kv indices + staging map into fixed buffers, per-block
        # plan/fast-plan). No-op unless the lane's block-decode graph path is
        # active for this capture/replay.
        if (
            forward_mode.is_decode_or_idle()
            and getattr(self, "weightless_kv", False)
            and getattr(self, "_wl_chunk_block_size", 0)
            and (
                (in_capture and self._wl_graph_capture_blocks is not None)
                or (not in_capture and self._wl_graph_replay_blocks is not None)
            )
        ):
            self._wl_graph_prepare_blocks(bs, in_capture=in_capture)

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
        # Eager path: never use a graph-bucket ragged wrapper. Tree-spec verify
        # (topk > 1) still needs an override: it plans a draft->draft custom
        # mask, and flashinfer's non-graph plan() does NOT reset _custom_mask_buf
        # when a later plan passes no mask -> a subsequent maskless prefill on
        # the SHARED ragged wrapper would silently run stale CUSTOM mode (an
        # M16-class bug). Route eager tree verify through a dedicated wrapper so
        # the shared prefill_wrapper_ragged is only ever planned maskless.
        if (
            self.uneven_dcp
            and self.dcp_tree_mask
            and forward_batch.forward_mode.is_target_verify()
        ):
            self._ragged_wrapper_override = self._get_eager_tree_verify_ragged_wrapper()
        else:
            self._ragged_wrapper_override = None
        swa_out_cache_loc = None
        if self.use_sliding_window_kv_pool and forward_batch.out_cache_loc is not None:
            assert self._swa_kv_pool is not None
            swa_out_cache_loc = self._swa_kv_pool.translate_loc_from_full_to_swa(
                forward_batch.out_cache_loc
            )

        # kv-session-offload (S1): (re)derive the spill-tick state for THIS
        # forward; None on every other batch so all paths stay untouched.
        # Set unconditionally -- a stale state from a previous spill tick
        # must never leak into a regular batch.
        if getattr(self, "_sess_enabled", False):
            is_spill = bool(getattr(forward_batch, "kv_session_spill_tick", False))
            self._sess_spill = (
                self._sess_prepare_step(forward_batch) if is_spill else None
            )
            # PS2 (deep prefill-spill): same discipline -- set unconditionally
            # so a born-spilled prefill's state can never leak into the next
            # (regular) forward. Rank-uniform: the flag rides the replicated
            # batch, so every rank takes the same branch in _dcp_write_scatter
            # and therefore issues the same per-layer collective sequence.
            self._sess_prefill_spill = (
                self._sess_prefill_prepare(forward_batch)
                if bool(getattr(forward_batch, "kv_session_prefill_spill", False))
                else None
            )
            # DECOUPLE S3: route THIS forward's DCP collectives to comm B when
            # it is a spill forward (serial per-forward flag; no-op unless the
            # second comm was built). Set unconditionally so a device forward
            # always resets to comm A. Rank-uniform (is_spill is replicated).
            if getattr(self, "_sess_decouple", False):
                from sglang.srt.distributed.parallel_state import (
                    set_dcp_spill_active,
                )

                set_dcp_spill_active(is_spill)

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
        # When called while an adaptive runtime state is being built in
        # offload mode, the large graph-state buffers are tagged as pauseable
        # per-state scratch (unmapped while the state is inactive, zeroed on
        # resume; every replay rewrites the lanes its graph reads and zeros
        # point at pool page 0, so the zeroed state equals the fresh-boot
        # pre-first-replay state). Static-path init is untouched: the tag
        # scope only exists during adaptive builds.
        if kv_indices_buf is None:
            with adaptive_graph_memory.tagged_state_alloc(
                nbytes=max_num_tokens * self.max_context_len * 4
            ):
                cuda_graph_kv_indices = torch.zeros(
                    (max_num_tokens * self.max_context_len,),
                    dtype=torch.int32,
                    device="cuda",
                )
            adaptive_graph_memory.note_state_tensor(cuda_graph_kv_indices)
        else:
            cuda_graph_kv_indices = kv_indices_buf

        extra_kv_indices = []
        for _ in range(self.num_wrappers - 1):
            with adaptive_graph_memory.tagged_state_alloc(
                nbytes=cuda_graph_kv_indices.numel()
                * cuda_graph_kv_indices.element_size()
            ):
                clone = cuda_graph_kv_indices.clone()
            adaptive_graph_memory.note_state_tensor(clone)
            extra_kv_indices.append(clone)
        self.cuda_graph_kv_indices = [cuda_graph_kv_indices] + extra_kv_indices

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
            with adaptive_graph_memory.tagged_state_alloc(
                nbytes=max_num_tokens * self.max_context_len
            ):
                self.cuda_graph_custom_mask = torch.zeros(
                    (max_num_tokens * self.max_context_len),
                    dtype=torch.uint8,
                    device="cuda",
                )
            adaptive_graph_memory.note_state_tensor(self.cuda_graph_custom_mask)
            # indptr clones are KiB-scale: below the tagging threshold, they
            # stay resident (small allocations must not share tagged
            # segments across states; see adaptive_graph_memory).
            self.cuda_graph_qk_indptr = [x.clone() for x in self.kv_indptr]
            self.cuda_graph_qo_indptr = [x.clone() for x in self.kv_indptr]

    def _create_decode_wrappers(self, bs: int, num_tokens: int) -> list:
        return [
            _tag_adaptive_int_workspace(
                BatchDecodeWithPagedKVCacheWrapper(
                    self.workspace_buffer,
                    "NHD",
                    backend=self.decode_backend,
                    use_cuda_graph=True,
                    use_tensor_cores=self.decode_use_tensor_cores,
                    paged_kv_indptr_buffer=self.kv_indptr[i][: num_tokens + 1],
                    paged_kv_indices_buffer=self.cuda_graph_kv_indices[i],
                    paged_kv_last_page_len_buffer=self.kv_last_page_len[:num_tokens],
                ),
                share_key=("decode_cg", id(self), i),
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
                _tag_adaptive_int_workspace(
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
                    ),
                    share_key=("prefill_cg", id(self), i, use_custom_mask),
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
            _tag_adaptive_int_workspace(
                BatchPrefillWithPagedKVCacheWrapper(
                    self.full_cg_prefill_workspace_buffer,
                    "NHD",
                    use_cuda_graph=True,
                    backend=self.prefill_backend,
                    qo_indptr_buf=self.full_cg_prefill_qo_indptr[i],
                    paged_kv_indptr_buf=self.full_cg_prefill_kv_indptr[i],
                    paged_kv_indices_buf=self.full_cg_prefill_kv_indices[i],
                    paged_kv_last_page_len_buf=self.kv_last_page_len[:num_slots],
                ),
                share_key=("full_cg_prefill", id(self), i),
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

    def _get_eager_tree_verify_ragged_wrapper(
        self,
    ) -> BatchPrefillWithRaggedKVCacheWrapper:
        """Dedicated NON-graph ragged wrapper for eager uneven-DCP tree verify
        (--speculative-eagle-topk > 1). Isolated from the shared
        prefill_wrapper_ragged so the draft->draft custom mask this path plans
        never leaks (as a stale _custom_mask_buf) onto a later maskless prefill.
        Non-graph plan() sets/updates its mask buffers dynamically per call, and
        this wrapper is only ever planned for verify (always masked), so no
        stale-mask hazard exists on it."""
        w = getattr(self, "_eager_tree_verify_ragged_wrapper", None)
        if w is None:
            w = BatchPrefillWithRaggedKVCacheWrapper(
                self.workspace_buffer,
                "NHD",
                backend=self._fmha_backend,
            )
            self._eager_tree_verify_ragged_wrapper = w
        return w

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

        TREE-SPEC (--speculative-eagle-topk > 1, self.dcp_tree_mask): the
        wrapper ALSO gets FIXED custom_mask_buf + mask_indptr_buf. flashinfer
        run() selects mask mode by BUFFER PRESENCE (`if self._custom_mask_buf
        is not None: mask_mode = CUSTOM`), so a wrapper created WITH a mask
        buffer runs CUSTOM every replay -- which is exactly what we want for
        the tree topology, and is safe here ONLY because call_begin_forward
        re-plans the draft->draft mask into the buffer on EVERY replay (the
        M16 lesson: never leave a mask buffer unwritten). topk == 1 wrappers
        are created WITHOUT the mask buffers -> mask mode falls back to the
        `causal` flag forward_return_lse sets == CAUSAL, byte-identical to the
        pre-tree-spec chain path.
        """
        wrapper = self.verify_ragged_cg_wrappers.get(bs)
        if wrapper is None:
            device = self.kv_last_page_len.device
            qo_indptr_buf = torch.zeros((bs + 1,), dtype=torch.int32, device=device)
            kv_indptr_buf = torch.zeros((bs + 1,), dtype=torch.int32, device=device)
            mask_bufs = {}
            if self.dcp_tree_mask:
                d = self.dcp_draft_token_num
                # Unpacked-sized uint8 buffer (bs * d * d bytes) is a safe upper
                # bound on flashinfer's packed mask (ceil(bs*d*d / 8) bytes);
                # plan() packs the fresh draft->draft mask into it each replay.
                mask_bufs = {
                    "custom_mask_buf": torch.zeros(
                        (bs * d * d,), dtype=torch.uint8, device=device
                    ),
                    "mask_indptr_buf": torch.zeros(
                        (bs + 1,), dtype=torch.int32, device=device
                    ),
                }
            wrapper = _tag_adaptive_int_workspace(
                BatchPrefillWithRaggedKVCacheWrapper(
                    self.workspace_buffer,
                    "NHD",
                    use_cuda_graph=True,
                    qo_indptr_buf=qo_indptr_buf,
                    kv_indptr_buf=kv_indptr_buf,
                    backend=self._fmha_backend,
                    **mask_bufs,
                ),
                share_key=("verify_ragged_cg", id(self)),
            )
            self.verify_ragged_cg_wrappers[bs] = wrapper
        return wrapper

    def _get_draft_extend_ragged_cg_wrapper(
        self, bs: int
    ) -> BatchPrefillWithRaggedKVCacheWrapper:
        """Per-bucket CUDA-graph-mode RAGGED wrapper for the uneven-DCP
        DRAFT-EXTEND graph (#108 slice 2).

        Same root cause and same fix as ``_get_verify_ragged_cg_wrapper``: the
        DCP draft-extend path runs the current-chunk causal attention through a
        RAGGED wrapper INSIDE the captured graph, and the shared
        ``prefill_wrapper_ragged`` is not in cuda-graph mode -- its plan() keeps
        a bare reference to the caller's transient ``torch.arange`` indptr,
        which the captured kernel freezes and the allocator later reuses. Read
        that method's docstring for the full argument; it is not repeated here,
        because the only thing that differs is which graph the buffers belong
        to.

        NO MASK BUFFERS, and that is a guarantee rather than an omission: a
        draft-extend chain is plain causal, and topk > 1 cannot reach this path
        at all -- ``--draft-kv-layout dcp`` refuses a tree draft at boot
        (ServerArgs._reject_unsupported_draft_kv_dcp). A wrapper created without
        a custom_mask_buf runs flashinfer's CAUSAL mode, selected by the
        ``causal`` flag ``forward_return_lse`` passes, which is what the
        current-chunk stage wants.
        """
        wrapper = self.draft_extend_ragged_cg_wrappers.get(bs)
        if wrapper is None:
            device = self.kv_last_page_len.device
            wrapper = _tag_adaptive_int_workspace(
                BatchPrefillWithRaggedKVCacheWrapper(
                    self.workspace_buffer,
                    "NHD",
                    use_cuda_graph=True,
                    qo_indptr_buf=torch.zeros(
                        (bs + 1,), dtype=torch.int32, device=device
                    ),
                    kv_indptr_buf=torch.zeros(
                        (bs + 1,), dtype=torch.int32, device=device
                    ),
                    backend=self._fmha_backend,
                ),
                share_key=("draft_extend_ragged_cg", id(self)),
            )
            self.draft_extend_ragged_cg_wrappers[bs] = wrapper
        return wrapper

    def _verify_cg_key(self, bs: int, forward_mode, spec_info):
        """Key for ``prefill_cuda_graph_metadata`` (#274 round 7a).

        It used to be ``bs`` alone, and that is right exactly while a runner
        captures ONE verify shape per bs. A chain-length LADDER captures
        several at the same bs -- 2, 3 and 4 candidate rows on the dual-group
        lane -- and every rung then overwrote the previous rung's wrappers. The
        surviving set was whichever rung was captured last, so every rung
        re-planned through it; flashinfer latches ``_max_total_num_rows`` on a
        wrapper's first plan, so a 3-row rung died with "the total number of
        rows in qo_indptr 3 ... cannot exceed ... 2" (measured, #274 round 7a
        boot 4).

        The row count comes from ``spec_info.draft_token_num`` -- the same
        source the eager path uses for its query-start-loc stride, and present
        on both the capture stand-in and the live verify. Anything without one
        (DLLM extend, and any caller whose view carries no spec_info) keeps the
        plain ``bs`` key, so nothing outside a verify ladder changes.
        """
        rows = None
        if forward_mode.is_target_verify():
            rows = getattr(spec_info, "draft_token_num", None)
        return (bs, int(rows)) if rows else bs

    def _prepare_cuda_graph_metadata(
        self,
        bs: int,
        num_tokens: int,
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
    ) -> None:
        if forward_mode.is_decode_or_idle():
            if (
                getattr(self, "_wl_chunk_block_size", 0)
                and bs in self.decode_cuda_graph_metadata
            ):
                # #136a rung-ladder captures revisit the same bs bucket
                # multiple times; reuse the bucket's wrappers instead of
                # leaking a fresh set (8 MB int workspace each) per rung.
                decode_wrappers = self.decode_cuda_graph_metadata[bs]
            else:
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
            # #274 round 7a: keyed by (bs, num_tokens), not by bs alone.
            # A chain-length LADDER captures several verify shapes at the SAME
            # bs -- 2, 3 and 4 candidate rows here -- and a bs-only key made
            # every rung overwrite the previous one's wrappers. The surviving
            # set was the last rung captured, so every rung's replay re-planned
            # through it: flashinfer latches ``_max_total_num_rows`` on its
            # first plan, so a 3-row rung then died with "the total number of
            # rows in qo_indptr 3 ... cannot exceed ... 2" (measured, boot 4).
            # For every runner without a ladder num_tokens is a function of bs,
            # so the key merely gains a redundant component.
            self.prefill_cuda_graph_metadata[
                self._verify_cg_key(bs, forward_mode, spec_info)
            ] = prefill_wrappers
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
            # Which modes ALWAYS read a committed prefix -- ONE rule, shared
            # with the weightless worker and pinned on CPU
            # (layers/dcp/lockstep.dcp_forces_prefix):
            #   target-VERIFY: the prefix is the decode-phase committed context,
            #     length seq_lens (extend_prefix_lens is unset for verify).
            #   draft-EXTEND (#108 slice 2): seq_lens already counts the tokens
            #     this step appends; the earlier draft context is in the pool
            #     and likewise carries no prefix vector.
            # A length-based test would skip the prefix stage entirely for
            # either -- and that stage is where the Q all-gather and the LSE
            # merge live, so a per-rank answer is a hang (#94 family).
            force_prefix = dcp_forces_prefix(
                forward_batch.forward_mode.is_target_verify(),
                forward_batch.forward_mode.is_draft_extend_v2(),
            )
            return self._forward_extend_dcp(
                q,
                k,
                v,
                layer,
                forward_batch,
                prefill_wrapper_paged,
                cache_loc,
                logits_soft_cap,
                save_kv_cache,
                force_prefix=force_prefix,
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
        # kv-session-offload spill tick: the host-streamed decode
        # (_sess_forward_decode_plain / _forward_decode_dcp spill branch) never
        # uses the paged decode_wrapper. On the LIVE tick self.forward_metadata
        # is decode metadata (init_forward_metadata built decode_wrappers), so
        # the fetch is a harmless read; but during the S0 spill-graph CAPTURE the
        # out-graph early-return (init_forward_metadata_out_graph) never builds
        # decode_wrappers -- and under a speculative server the normal decode
        # capture left self.forward_metadata as PrefillMetadata -- so a naive
        # fetch AttributeErrors. Skip the fetch when this is a spill tick; the
        # wrapper is unused on that path anyway (byte-identical for every non-
        # spill batch, whose _sess_spill is None).
        decode_wrapper = (
            self.forward_metadata.decode_wrappers[self._get_wrapper_idx(layer)]
            if self._sess_spill is None
            else None
        )
        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )

        if self.uneven_dcp:
            return self._forward_decode_dcp(
                q, k, v, layer, forward_batch, decode_wrapper, cache_loc, save_kv_cache
            )

        if self._sess_spill is not None:
            # kv-session-offload spill tick without DCP: host-streamed block
            # decode over this rank's kv-head slice of the whole session.
            return self._sess_forward_decode_plain(q, k, v, layer, save_kv_cache)

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

    def _dcp_write_gather(self, layer, k, v):
        """Collective half of the masked KV write: gather this rank's local
        kv-head shard [2,1,1] up to the FULL replicated kv-heads [4]
        (collectives A_k, A_v of the per-layer DCP sequence).

        REPLICATED-KV geometry (TP > num_kv_heads): every rank already
        projects ALL kv heads itself (identical replicated k/v weights ->
        identical K/V, deterministic recompute), so the gather is skipped."""
        group = get_parallel().dcp_group
        k = k.view(-1, layer.tp_k_head_num, layer.head_dim)
        v = v.view(-1, layer.tp_v_head_num, layer.head_dim)
        if self.dcp_kv_replicated_heads:
            return k, v
        # #132 Fusion 1: gather the k and v head-shards in ONE all-gather
        # instead of two. k and v share dcp_kv_head_counts, so stacking them
        # along the row dim (dim 0) and gathering along the head dim (dim 1)
        # produces per-tensor results IDENTICAL to two separate gathers --
        # pure data movement, NO reduction -> BYTE-IDENTICAL -- while halving
        # the kv all-gather launches on the latency-bound decode critical
        # path (5 -> 4 collectives/attn-layer; Stage-0 measured collectives at
        # ~64% of attention GPU time / ~38% of graph decode wall). Replicated-
        # KV (e.g. 122B, 2<3) skips A entirely, so this path never runs there.
        n = k.shape[0]
        kv_full = cp_all_gather_heads_uneven(
            torch.cat((k, v), dim=0), group, self.dcp_kv_head_counts
        )
        return kv_full[:n], kv_full[n:]

    def _dcp_masked_write(self, layer, forward_batch, cache_loc, k, v):
        """Gather this rank's local kv-head shard up to the FULL replicated
        kv-heads and write only the token slots this rank OWNS (even-modulo
        owner rule) at the compacted physical slot loc//dcp_size. Composition
        of _dcp_write_gather + _dcp_write_scatter (kept as the single-call
        form for the sequential/baseline paths)."""
        k_full, v_full = self._dcp_write_gather(layer, k, v)
        self._dcp_write_scatter(layer, forward_batch, cache_loc, k_full, v_full)

    def _dcp_write_scatter(self, layer, forward_batch, cache_loc, k_full, v_full):
        """Local half of the masked KV write (no collectives): compute this
        rank's owner mask + compact slots and scatter k_full/v_full into the
        token-sharded pool. Under #128 overlap this is deferred to run
        concurrently with the LSE-merge collectives -- its target slots are
        the current chunk's out_cache_loc, DISJOINT from the paged prefix
        read, so deferral is order-equivalent (byte-identical cache state at
        layer exit).

        Also shared (as ``_dcp_owner_write``) by the weightless KV-worker path,
        which receives k/v broadcast from the head rank: the owner-rule +
        compact-slot mapping is identical for both, so it lives here once
        (single source of truth)."""
        if self._sess_spill is not None and not self._sess_verify_active():
            # kv-session-offload spill tick (plain DECODE): the new token's slot
            # is a HOST sentinel -- never scatter it into the device pool. Route
            # the (gathered, replicated-head) K/V through the scratch-row D2H
            # owner-write instead. Strictly the plain-decode spill-tick lane.
            #
            # C4 (spec-in-spill-tick VERIFY): the num_draft candidate tokens are
            # written to REAL device candidate slots (out_cache_loc from
            # eagle_prepare_for_verify) -- they must take the NORMAL device
            # scatter below, not the single-token host owner-write (whose st has
            # no cur_owned field). The committed prefix stays host-resident;
            # only accepted candidates are promoted to the committed prefix (on
            # device) by the verify commit. So the verify spill tick falls
            # through to the normal DCP owner scatter.
            self._sess_owner_write(layer, k_full, v_full)
            return
        if self._sess_prefill_spill is not None:
            # PS2 (deep prefill-spill): this EXTEND's out_cache_loc is a row of
            # HOST sentinels -- there is no device slot to scatter into. Route
            # the whole chunk through the staging carve + copy-stream D2H.
            # Strictly the born-spilled prefill lane (the flag is set per
            # forward and cleared on every other batch).
            self._sess_prefill_owner_write(layer, k_full, v_full)
            return
        if self.uneven_dcp_weighted:
            # WEIGHTED owner rule: ownership + compact physical slot are derived
            # from the out_cache_loc itself (not the sequence position), so the
            # slot is an injective function of L and stays collision-free across
            # concurrent requests exactly like the even L // dcp_size. This rank
            # owns L iff (L % cp_S) in [cp_lo, cp_hi); its compact slot is
            # (L // cp_S) * cp_ratio + (L % cp_S - cp_lo).
            # Shared with the Triton backend and with the READ side
            # (build_dcp_weighted_kv_indices) via layers/dcp/owner.py, so the
            # two can no longer drift apart. Byte-identical to the expression
            # that used to be inlined here.
            loc, dcp_kv_mask = dcp_weighted_write_slots(
                cache_loc, self.cp_S, self.cp_lo, self.cp_hi, self.cp_ratio
            )
        else:
            loc = cache_loc // self.dcp_size
            # EVEN modulo owner rule via the shared helper: it refuses a padded
            # ``positions`` against a narrowed ``cache_loc`` (#472) instead of
            # falling through to an unmasked write. Same call as the Triton
            # twin, so the two lanes cannot classify a row differently.
            dcp_kv_mask = dcp_even_write_mask(
                forward_batch.positions,
                loc.numel(),
                self.dcp_size,
                self.dcp_rank,
                forward_batch.dcp_kv_mask,
            )
        if getattr(self, "_wl_spill_active", False):
            # Stage B1: owned slots may land in the HOST region of the logical
            # slot space; split the write (device part unchanged, host part
            # staged then copied D2H). Strictly the spill lane.
            self._wl_spill_owner_write(layer, loc, dcp_kv_mask, k_full, v_full)
            return
        self.token_to_kv_pool.set_kv_buffer(
            layer,
            loc,
            k_full.clone(),
            v_full.clone(),
            layer.k_scale,
            layer.v_scale,
            dcp_kv_mask=dcp_kv_mask,
        )

    # Historical name used by the weightless KV-worker (head-rank broadcast)
    # call sites -- same function, single source of truth.
    _dcp_owner_write = _dcp_write_scatter

    def _wl_spill_owner_write(self, layer, loc, dcp_kv_mask, k_full, v_full):
        """B1 owner-write over the tiered slot space. Device-region slots take
        the EXACT existing masked set_kv_buffer path. Host-region slots are
        first written into the staging region with the SAME set_kv_buffer
        semantics (identical dtype/scale handling -> the bytes are identical
        to what a device write would have produced) and then copied D2H into
        the pinned host tier for this layer -- a lossless byte move on the
        current stream. No mapping tables, no alloc state: host slot id =
        compacted slot - device_slots (the static tier map)."""
        # #136a: inside CUDA-graph capture the dynamic-shape split below
        # (.any()/.nonzero()/host chunking) is unrecordable -- route to the
        # fixed-shape graph-safe variant.
        if getattr(self, "_wl_graph_capture_blocks", None) is not None:
            return self._wl_spill_owner_write_graph(
                layer, loc, dcp_kv_mask, k_full, v_full
            )
        pool = self.token_to_kv_pool
        dev_limit = self._wl_dev_slots
        host_rows = dcp_kv_mask & (loc >= dev_limit)
        if not bool(host_rows.any().item()):
            pool.set_kv_buffer(
                layer,
                loc,
                k_full.clone(),
                v_full.clone(),
                layer.k_scale,
                layer.v_scale,
                dcp_kv_mask=dcp_kv_mask,
            )
            return
        from sgl_kernel.kvcacheio import transfer_kv_per_layer

        dev_mask = dcp_kv_mask & ~host_rows
        # Device part: identical masked write; host-region loc values are
        # masked OFF but clamp them anyway so no out-of-range slot id ever
        # reaches the device-pool scatter.
        safe_loc = torch.where(loc < dev_limit, loc, torch.zeros_like(loc))
        pool.set_kv_buffer(
            layer,
            safe_loc,
            k_full.clone(),
            v_full.clone(),
            layer.k_scale,
            layer.v_scale,
            dcp_kv_mask=dev_mask,
        )
        # Host part: stage -> D2H, chunked to the staging capacity.
        if not getattr(self, "_wl_spill_write_logged", False):
            self._wl_spill_write_logged = True
            logger.info(
                "Weightless-KV host spill: first D2H owner-write into the "
                "host tier (dcp_rank %d, layer %d).",
                self.dcp_rank,
                layer.layer_id,
            )
        idx = host_rows.nonzero(as_tuple=False).flatten()
        host_ids_all = loc[idx].to(torch.int64) - dev_limit
        n_miss = int((host_ids_all >= self._wl_host_slots).sum().item())
        if n_miss:
            raise RuntimeError(
                "weightless-KV host spill: owner-write beyond the host tier "
                f"({n_miss} slot(s) past {self._wl_host_slots} tokens, "
                f"dcp_rank {self.dcp_rank}, layer {layer.layer_id}). The "
                "allocator and the host tier disagree; refusing a silent "
                "out-of-range KV write."
            )
        fl = self._wl_full_layer_idx(layer)
        full_pool = self._wl_full_pool
        host_pool = self._wl_host_pool
        k_sel = k_full[idx]
        v_sel = v_full[idx]
        total = idx.numel()
        cap = self._wl_stage_cap
        for off in range(0, total, cap):
            end = min(off + cap, total)
            n = end - off
            stage_slots = torch.arange(
                self._wl_stage_base,
                self._wl_stage_base + n,
                dtype=torch.int64,
                device=loc.device,
            )
            # Same write kernel/dtype path as a device-resident slot.
            pool.set_kv_buffer(
                layer,
                stage_slots.to(loc.dtype),
                k_sel[off:end],
                v_sel[off:end],
                layer.k_scale,
                layer.v_scale,
            )
            # Lossless D2H byte copy of the staged rows into the host tier
            # (device kernel writing pinned host memory, current stream).
            transfer_kv_per_layer(
                src_k=full_pool.k_buffer[fl],
                dst_k=host_pool.k_data_refs[fl],
                src_v=full_pool.v_buffer[fl],
                dst_v=host_pool.v_data_refs[fl],
                src_indices=stage_slots,
                dst_indices=host_ids_all[off:end],
                item_size=host_pool.token_stride_size,
            )

    def _wl_block_decode_wrapper(self):
        """Persistent block decode wrapper for the Stage B0 block loop. Lazily
        created; re-planned per block. Shares the backend workspace -- safe
        because under block-decode the MAIN decode wrapper is never .run()
        (this loop replaces it entirely for the weightless full-attn layers)."""
        if self._wl_block_wrapper is None:
            self._wl_block_wrapper = BatchDecodeWithPagedKVCacheWrapper(
                self.workspace_buffer,
                "NHD",
                backend=self.decode_backend,
                use_tensor_cores=self.decode_use_tensor_cores,
            )
        return self._wl_block_wrapper

    def _wl_blockwise_decode_return_lse(self, q_full, layer):
        """Stage B0: block-decode this rank's OWNED KV token-shard.

        Replaces the single monolithic ``decode_wrapper.forward_return_lse`` over
        the owned shard with a BLOCK LOOP: the owned kv_indices are sliced into
        blocks of ``self._wl_chunk_block_size`` slots that pass through a bounded
        device staging region; each block runs a flashinfer split-KV PARTIAL
        decode (returns partial output + LSE); the partials are ONLINE-MERGED via
        the flashinfer LSE operator (``_safe_merge_state``) into a running
        accumulator. Byte-identical to the monolithic call up to fp
        reassociation (the merge is mathematically exact; it only reassociates).

        B0 keeps ALL KV RESIDENT -- "streaming" is just iterating resident slots
        in blocks. The block SOURCE is isolated to ``_wl_stage_block`` so B1 can
        swap it to a host->device copy WITHOUT touching this merge logic.

        Rank-uniform: the block COUNT is derived from the SHARED global seq_len
        (identical on the head + every worker), NOT from local owned length. The
        loop is intra-rank and issues NO cross-rank collective -- the lynchpin
        4-collectives/layer is unperturbed. Returns (o, lse) in the SAME
        shape/space as ``forward_return_lse`` so the existing cross-rank
        ``cp_lse_ag_out_ar_mha_uneven`` consumes it unchanged."""
        kv_indptr = self._dcp_decode_owned_kv_indptr  # int32 [bs+1] owned cumsum
        kv_indices = self._dcp_decode_owned_kv_indices  # int32 owned slots
        seq_lens = self._dcp_decode_global_seq_lens  # [bs] GLOBAL per-req lens
        B = self._wl_chunk_block_size
        kv_buffer = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)

        bs = kv_indptr.numel() - 1
        num_qo_heads = q_full.shape[1]
        num_kv_heads = self.indices_updater_decode.num_kv_heads
        head_dim = q_full.shape[2]

        # Rank-uniform block count from the SHARED global seq_len: under the
        # even-modulo owner rule each rank owns ~1/dcp_size of the tokens, so a
        # global window of (B * dcp_size) positions holds ~B owned slots on every
        # rank. Deriving the count from the max global seq_len keeps head +
        # workers on the SAME iteration count regardless of the +-1 owned skew;
        # the nested-ceiling identity guarantees it covers every owned slot.
        global_max = int(seq_lens.max().item())
        per_block_global = B * self.dcp_size
        num_blocks = max(1, (global_max + per_block_global - 1) // per_block_global)

        indptr_host = kv_indptr.to("cpu")
        wrapper = self._wl_block_decode_wrapper()
        last_page = self.kv_last_page_len
        iu = self.indices_updater_decode

        o_acc = None
        lse_acc = None
        for j in range(num_blocks):
            blk_indices, blk_indptr = self._wl_stage_block(
                kv_indices, indptr_host, bs, j, B, layer
            )
            if blk_indices is None:
                # Entirely empty block on this rank (kept in the rank-uniform
                # loop count; no attention, no merge -> no divergence).
                continue
            wrapper.plan(
                blk_indptr,
                blk_indices,
                last_page[:bs],
                num_qo_heads,
                num_kv_heads,
                head_dim,
                1,  # page_size
                q_data_type=iu.q_data_type,
                kv_data_type=iu.data_type,
            )
            o_b, lse_b = wrapper.forward_return_lse(
                q_full,
                kv_buffer,
                sm_scale=layer.scaling,
                logits_soft_cap=layer.logit_cap,
                k_scale=layer.k_scale_float,
                v_scale=layer.v_scale_float,
            )
            if o_acc is None:
                o_acc, lse_acc = o_b, lse_b
            else:
                o_acc, lse_acc = _safe_merge_state(o_acc, lse_acc, o_b, lse_b)
        if o_acc is None:
            # Degenerate: this rank owns ZERO slots in every block (e.g. a very
            # short sequence where the even-modulo split leaves this rank with no
            # tokens). The monolithic forward_return_lse still returns a shaped
            # empty-attention (o=0, lse=-inf) tensor -- reproduce that here so the
            # cross-rank cp_lse merge stays balanced (same contract as baseline;
            # the block loop must never return None or a rank desyncs the merge).
            wrapper.plan(
                kv_indptr,
                kv_indices,
                last_page[:bs],
                num_qo_heads,
                num_kv_heads,
                head_dim,
                1,
                q_data_type=iu.q_data_type,
                kv_data_type=iu.data_type,
            )
            o_acc, lse_acc = wrapper.forward_return_lse(
                q_full,
                kv_buffer,
                sm_scale=layer.scaling,
                logits_soft_cap=layer.logit_cap,
                k_scale=layer.k_scale_float,
                v_scale=layer.v_scale_float,
            )
        return o_acc, lse_acc

    def _wl_stage_block(self, kv_indices, indptr_host, bs, j, B, layer):
        """Stage block ``j`` of the owned KV shard into the (bounded) device
        region the block wrapper reads. B0 (all-resident): a pure index slice
        of ``kv_indices`` (no data move). B1 (host spill active): slots in the
        HOST region (compacted slot >= _wl_dev_slots) are streamed H2D for
        ``layer`` into the staging slots and their indices rewritten IN PLACE
        (order preserved) to point at the staged copies -- the caller's merge
        loop is unchanged, so the byte class is unchanged. Returns
        (block_kv_indices, block_kv_indptr), or (None, None) when the block is
        empty on this rank."""
        seg_list = []
        blk_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=kv_indices.device)
        for i in range(bs):
            start = int(indptr_host[i])
            end = int(indptr_host[i + 1])
            bstart = start + j * B
            bend = min(start + (j + 1) * B, end)
            blk_len = bend - bstart if bend > bstart else 0
            if blk_len > 0:
                seg_list.append(kv_indices[bstart:bend])
            blk_indptr[i + 1] = blk_indptr[i] + blk_len
        if not seg_list:
            return None, None
        blk_indices = torch.cat(seg_list).contiguous()
        if getattr(self, "_wl_spill_active", False):
            blk_indices = self._wl_stage_host_slots(blk_indices, layer)
        return blk_indices, blk_indptr

    def _wl_full_layer_idx(self, layer):
        """Raw index of ``layer`` into the full-attention sub-pool's per-layer
        k/v buffer lists. Both the host tier's transfer methods and the raw D2H
        spill write index the buffers with this.

        Two hops (DESIGN_pp_layer_set.md §8.2). ``layer.layer_id`` is a GLOBAL
        model layer id; ``_transfer_full_attention_id`` re-indexes it into the
        sub-pool's DENSE full-attention frame; the sub-pool then resolves that
        to a buffer slot. The second hop goes through the pool's own accessor
        so the FRAME is chosen by the pool rather than assumed here -- for a
        sub-pool that is the plain offset (its ``start_layer`` is 0, since
        HybridLinearKVPool builds it without one), which is what this line has
        always computed.
        """
        mapped = self.token_to_kv_pool._transfer_full_attention_id(layer.layer_id)
        return self._wl_full_pool.local_slot(mapped)

    def _wl_stage_host_slots(self, blk_indices, layer):
        """B1 block SOURCE swap: for every HOST-resident slot in this block,
        stream its KV for ``layer`` H2D into the bounded staging region and
        return the block's indices with those entries rewritten to the staging
        slots (order preserved -> the partial attention consumes byte-identical
        values in the identical layout). Device-resident slots pass through
        untouched, so an all-resident block is byte-for-byte the B0 path.

        FAIL LOUD on any slot that cannot be staged (staging overflow / host
        range miss): a silently skipped block would feed garbage KV into the
        merge -- wrong output, not a hang (the collective count would still
        match)."""
        hmask = blk_indices >= self._wl_dev_slots
        n_host = int(hmask.sum().item())
        if n_host == 0:
            return blk_indices
        if n_host > self._wl_stage_cap:
            raise RuntimeError(
                "weightless-KV host spill: block demands "
                f"{n_host} staged host slots but the staging region holds only "
                f"{self._wl_stage_cap} (dcp_rank {self.dcp_rank}, layer "
                f"{layer.layer_id}). This happens when multiple concurrently "
                "running requests are simultaneously past the device-resident "
                "capacity; B1 supports one spilled request per block "
                "iteration. Lower concurrency or raise "
                "--weightless-kv-chunked-block-size."
            )
        host_ids = (blk_indices[hmask].to(torch.int64)) - self._wl_dev_slots
        n_miss = int((host_ids >= self._wl_host_slots).sum().item())
        if n_miss:
            raise RuntimeError(
                "weightless-KV host spill: HOST MISS -- "
                f"{n_miss} required slot(s) beyond the {self._wl_host_slots}-"
                f"token host tier (dcp_rank {self.dcp_rank}, layer "
                f"{layer.layer_id}). The logical slot space and the host tier "
                "disagree; refusing to attend garbage KV."
            )
        stage_slots = torch.arange(
            self._wl_stage_base,
            self._wl_stage_base + n_host,
            dtype=torch.int64,
            device=blk_indices.device,
        )
        if not getattr(self, "_wl_spill_read_logged", False):
            self._wl_spill_read_logged = True
            logger.info(
                "Weightless-KV host spill: first H2D block stream engaged "
                "(dcp_rank %d, layer %d, %d host slot(s) staged).",
                self.dcp_rank,
                layer.layer_id,
                n_host,
            )
        # Lossless pinned-host -> device byte copy for THIS layer, on the
        # current stream (stream order alone guarantees the partial attention
        # below reads the staged bytes -- no extra sync needed).
        self._wl_host_pool.load_to_device_per_layer(
            self._wl_full_pool,
            host_ids,
            stage_slots,
            self._wl_full_layer_idx(layer),
            io_backend="kernel",
        )
        out = blk_indices.clone()
        out[hmask] = stage_slots.to(blk_indices.dtype)
        return out

    # ------------------------------------------------------------------
    # kv-session-offload (S1): host-streamed decode of ONE spilled session.
    #
    # A spilled session's whole full-attention KV shard lives in a pinned
    # host pool, in per-rank OWNED-POSITION order (host row i = this rank's
    # i-th owned token). The spill tick is a separate eager bs=1 decode
    # batch; per full-attention layer it
    #   * owner-writes the new token's K/V through ONE device scratch row
    #     (stock set_kv_buffer quantization -> byte-identical to a device
    #     write) and D2H-appends it to the host pool, then
    #   * streams the shard blockwise H2D through a bounded staging buffer,
    #     runs flashinfer partial decodes and online-LSE-merges them --
    #     the exact _wl_blockwise mechanism, but with a per-session dynamic
    #     row space instead of B1's static slot->tier map.
    # The per-layer DCP collective sequence (kv gather / q gather /
    # LSE-merge) is UNCHANGED; the block count is rank-uniform (max over
    # every rank's owned count, all derived from the replicated
    # req_to_token sentinels), so no rank can desync the lockstep.
    # ------------------------------------------------------------------

    def _sess_wire(self, model_runner):
        """One-time wiring of the session-offload state (flag-gated)."""
        from sglang.srt.distributed.utils import (
            cp_token_prefix,
            cp_token_split_factor,
        )
        from sglang.srt.managers.kv_session_offload import sentinel_base

        dcp = int(getattr(self, "dcp_size", 1) or 1)
        if dcp > 1:
            assert self.uneven_dcp, (
                "kv-session-offload: DCP without the uneven-DCP owner rule "
                "is out of S1 scope (should have failed at pool init)"
            )
            self._sess_mode = "weighted" if self.uneven_dcp_weighted else "even"
            self._sess_S = cp_token_split_factor(dcp)
            self._sess_prefix = (
                cp_token_prefix(dcp)
                if self.uneven_dcp_weighted
                else list(range(dcp + 1))
            )
        else:
            self._sess_mode = "plain"
            self._sess_S = 1
            self._sess_prefix = [0, 1]
        pool = model_runner.token_to_kv_pool
        self._sess_pool = pool
        self._sess_full_pool = getattr(pool, "full_kv_pool", pool)
        self._sess_host_pool = model_runner.kv_sess_host_pool
        self._sess_req_pool = model_runner.req_to_token_pool
        self._sess_block_size = int(
            model_runner.server_args.kv_session_offload_block_size
        )
        self._sess_host_base = sentinel_base(
            model_runner.token_to_kv_pool_allocator.size, self._sess_S
        )
        scratch = getattr(model_runner, "_kv_sess_scratch_slot", None)
        assert (
            scratch is not None
        ), "kv-session-offload: scratch row missing from the KV pool"
        dev = self._sess_full_pool.k_buffer[0].device
        self._sess_scratch_loc = torch.tensor([scratch], dtype=torch.int64, device=dev)
        proto = self._sess_full_pool.k_buffer[0]
        # S2: TWO block-sized staging regions (double buffer). Region r
        # occupies staging rows [r*B, (r+1)*B); the H2D copy of streamed
        # block g+1 (region (g+1)%2) runs on a dedicated copy stream while
        # attention reads block g from the other region -- including ACROSS
        # layers (the streamed schedule is flattened layer-major, so the
        # next layer's first block prefetches during the current layer's
        # last block).
        self._sess_staging_k = proto.new_empty(
            (2 * self._sess_block_size,) + tuple(proto.shape[1:])
        )
        self._sess_staging_v = proto.new_empty(
            (2 * self._sess_block_size,) + tuple(proto.shape[1:])
        )
        # The wrapper must see the pool's LOGICAL dtype (e.g. an fp8 view of
        # the uint8 store), exactly like get_kv_buffer() -- a raw uint8
        # tensor would be misread as an NVFP4 cache.
        fp = self._sess_full_pool
        if getattr(fp, "store_dtype", None) is not None and fp.store_dtype != fp.dtype:
            self._sess_staging_kv = (
                self._sess_staging_k.view(fp.dtype),
                self._sess_staging_v.view(fp.dtype),
            )
        else:
            self._sess_staging_kv = (self._sess_staging_k, self._sess_staging_v)
        # S2 streaming pipeline state -------------------------------------
        # * Dedicated copy stream: staged H2D reads of the pinned host pool
        #   NEVER run on the compute stream (isolation invariant: the
        #   spilled session must not serialize against device batches).
        # * Per-region events implement the classic double buffer: a copy
        #   into region r waits the region's last consumer (compute_done),
        #   a consumer waits the region's copy (copy_done). All event
        #   record/wait calls are issued from the single forward thread, so
        #   every wait binds to the intended recording.
        # * Persistent aranges: block staging index tensors are slices
        #   (views) of these -- no per-block allocations on the hot path.
        # * Wrapper pool: flashinfer decode plans are cached per
        #   (full|tail|empty, region) and REUSED across all layers and
        #   ticks; a replan happens only when that slot's staged row count
        #   changed (<= 2 replans per tick, instead of the S1
        #   plan-per-layer-per-block that dominated the 90 ms tick).
        self._sess_copy_stream = torch.cuda.Stream(device=dev)
        self._sess_copy_done = [torch.cuda.Event(), torch.cuda.Event()]
        self._sess_compute_done = [torch.cuda.Event(), torch.cuda.Event()]
        # DECOUPLE S2: a SEPARATE flashinfer float workspace for the spill-lane
        # wrappers. Today they share self.workspace_buffer -- safe only because
        # the spill tick is serial (the device wrapper is never .run()
        # concurrently). Concurrency (S4) would race the shared scratch, so
        # reserve a second workspace of the same size when decoupling is on.
        # Flag OFF -> alias the shared buffer (byte-identical). This is a
        # STANDING reservation (SGLANG_FLASHINFER_WORKSPACE_SIZE per rank).
        from sglang.srt.managers.kv_session_offload import spill_decouple_enabled

        self._sess_decouple = bool(spill_decouple_enabled())
        if self._sess_decouple:
            self._sess_workspace_buffer = register_flashinfer_workspace_buffer(
                torch.empty(
                    envs.SGLANG_FLASHINFER_WORKSPACE_SIZE.get(),
                    dtype=torch.uint8,
                    device=dev,
                )
            )
            logger.info(
                "kv-session-offload DECOUPLE: separate spill-lane flashinfer "
                "workspace reserved (%.0f MiB).",
                envs.SGLANG_FLASHINFER_WORKSPACE_SIZE.get() / 2**20,
            )
        else:
            self._sess_workspace_buffer = self.workspace_buffer
        self._sess_host_arange = torch.arange(
            self._sess_host_pool.size, dtype=torch.int64, device=dev
        )
        self._sess_stage_arange32 = torch.arange(
            2 * self._sess_block_size, dtype=torch.int32, device=dev
        )
        self._sess_stage_arange64 = torch.arange(
            2 * self._sess_block_size, dtype=torch.int64, device=dev
        )
        self._sess_scratch_i = int(scratch)
        self._sess_wrappers = {}
        # S4: persistent per-SESSION state, keyed by req_pool_idx. Each entry:
        #   .region_base  -- first host row of this session's host-pool region
        #   .host_row_base -- S3 wave-back drain (rows [region_base,
        #                     region_base+host_row_base) already migrated back)
        #   .head         -- S1b (boundary, dev_head_idx int32, n_head_own),
        #                    reset when the head grows (spill / wave-back)
        #   .count_cache  -- S2 incremental tail owned-count cache (L, counts)
        # Only these are per-session; the ticks are serialized so the staging
        # buffers, copy stream and wave event stay singular.
        self._sess_slots = {}
        self._sess_region_tokens = int(
            getattr(model_runner, "kv_sess_region_tokens", self._sess_host_pool.size)
        )
        # A never-recorded CUDA event reports .query() == True (no pending
        # work), so wave-back throttling starts in the "idle/complete" state.
        self._sess_wave_done = torch.cuda.Event()
        # ---- PS2 (deep prefill-spill) stage A/B' -------------------------
        # A born-spilled EXTEND allocates NO device KV slots, so its freshly
        # computed K/V cannot be scattered into the pool: it is quantised
        # through a device STAGING CARVE (the stock set_kv_buffer byte path,
        # so the bytes equal what a device write would have produced) and then
        # D2H-copied into this session's host region rows.
        #   * the carve is rank-uniformly SIZED at pool construction
        #     (prefill_stage_tokens) -- the FILL level differs per rank;
        #   * the D2H runs on _sess_copy_stream (stage B'), forked per layer
        #     from the compute stream and joined ONCE at the handover;
        #   * _sess_prefill_slots holds the per-session plan built at alloc
        #     time (owner indices + host rows), so the forward itself does no
        #     rank-local decision making.
        self._sess_prefill_stage_base = int(
            getattr(model_runner, "_kv_sess_prefill_stage_base", 0) or 0
        )
        self._sess_prefill_stage_cap = int(
            getattr(model_runner, "_kv_sess_prefill_stage_tokens", 0) or 0
        )
        self._sess_prefill_slots = {}
        self._sess_prefill_fork_ev = torch.cuda.Event()
        self._sess_prefill_done = torch.cuda.Event()
        self._sess_prefill_stage_arange = (
            torch.arange(
                self._sess_prefill_stage_base,
                self._sess_prefill_stage_base + self._sess_prefill_stage_cap,
                dtype=torch.int64,
                device=dev,
            )
            if self._sess_prefill_stage_cap > 0
            else None
        )
        # S5 bs=1 spill-graph: bucketed rung ladder over the host block count,
        # built ONCE (max blocks = a full region). None when the flag is off
        # -> the spill tick stays on the byte-identical eager block loop. The
        # actual torch.cuda.graph capture/replay of the fixed-count body is
        # wired by the spill-tick graph runner (GPU pass); this only supplies
        # the rung ladder + per-step out-of-graph plan (the #136a hoist).
        from sglang.srt.managers.kv_session_offload import (
            spill_graph_blocks_needed,
            spill_graph_enabled,
            spill_graph_rung_ladder,
        )

        self._sess_graph_enabled = bool(spill_graph_enabled())
        # S5 graph state (mirrors the _wl_graph_* fields):
        #   _sess_graph_capture_blocks -- rung currently being CAPTURED (set by
        #     the decode graph runner around each rung's capture; else None).
        #   _sess_graph_replay_blocks  -- rung chosen for the CURRENT replay by
        #     _sess_graph_can_replay (else None -> eager).
        #   _sess_graph_bucket -- the single bs=1 bucket: persistent per-block
        #     cuda-graph wrappers (wrappers[0..R-1], shared by all rungs), the
        #     dev-head graph wrapper, fixed indptr/indices/staging buffers.
        self._sess_graph_capture_blocks = None
        self._sess_graph_replay_blocks = None
        self._sess_graph_bucket = None
        self._sess_graph_fallback_logged = False
        # Rungs whose spill-tick graph the capture pass has recorded (GPU pass
        # populates this; empty -> every spill tick stays eager).
        self._sess_graph_captured_rungs = set()
        # Set only inside the spill-tick capture pass (synthetic session).
        self._sess_capture_active = False
        self._sess_capture_rpi = None
        if self._sess_graph_enabled:
            max_blocks = spill_graph_blocks_needed(
                self._sess_region_tokens, self._sess_block_size
            )
            self._sess_graph_ladder = spill_graph_rung_ladder(max_blocks)
            logger.info(
                "kv-session-offload S5 spill-graph ENABLED: block=%d "
                "region=%d -> ladder %s (eager fallback for over-ladder / "
                "non-graphable ticks; capture driven by the decode graph "
                "runner -- GPU-justification: capture-region + synthetic "
                "spill-tick batch + dev-head capture-at-max)",
                self._sess_block_size,
                self._sess_region_tokens,
                self._sess_graph_ladder,
            )
        else:
            self._sess_graph_ladder = None
        logger.info(
            "kv-session-offload backend wired: mode=%s S=%d prefix=%s "
            "host_base=%d scratch_slot=%d block=%d staging=%.1f MB",
            self._sess_mode,
            self._sess_S,
            self._sess_prefix,
            self._sess_host_base,
            scratch,
            self._sess_block_size,
            2 * self._sess_staging_k.numel() * proto.element_size() / 1e6,
        )

    def _sess_open_slot(self, rpi: int, region_base: int):
        """Register a newly spilled session's per-session state (S4). Called
        by the manager at spill time; region_base is the first host row of
        this session's host-pool region."""
        from types import SimpleNamespace

        self._sess_slots[int(rpi)] = SimpleNamespace(
            region_base=int(region_base),
            host_row_base=0,
            head=None,
            count_cache=None,
        )

    def _sess_close_slot(self, rpi: int):
        """Drop a session's per-session state on restore/finish (S4)."""
        self._sess_slots.pop(int(rpi), None)

    # ---- PS2 (deep prefill-spill) stage A/B' -----------------------------

    def _sess_prefill_open(self, rpi: int, boundary: int, own_idx, region_base: int):
        """Register the born-spilled EXTEND write plan for one session.

        Called from the manager's ``spill_extend_alloc`` in the SAME
        synchronous scheduler section that claims the region and writes the
        sentinel row -- never from a stream callback (U8).

        ``own_idx`` indexes the extend chunk (ascending position order); host
        row ``region_base + j`` receives the j-th owned token, which is exactly
        where ``_sess_prepare_step`` will look for it afterwards
        (``base = region_base + host_row_base``, rows ``[base, base + n_own)``,
        ``host_row_base == 0`` for a fresh region). That identity IS the PS2
        half of the lockstep invariant."""
        from types import SimpleNamespace

        n_own = int(own_idx.numel())
        if n_own > self._sess_prefill_stage_cap:
            raise RuntimeError(
                "kv-session-offload prefill-spill (PS2): chunk needs "
                f"{n_own} staging rows > carve {self._sess_prefill_stage_cap} "
                f"(dcp_rank {self.dcp_rank}). The carve is sized from "
                "--chunked-prefill-size; refusing an out-of-range write."
            )
        dev = self._sess_staging_k.device
        self._sess_prefill_slots[int(rpi)] = SimpleNamespace(
            boundary=int(boundary),
            own_idx=own_idx.to(dev),
            n_own=n_own,
            host_rows=torch.arange(
                region_base, region_base + n_own, dtype=torch.int64, device=dev
            ),
            stage_rows=(
                self._sess_prefill_stage_arange[:n_own]
                if self._sess_prefill_stage_arange is not None
                else None
            ),
        )

    def _sess_prefill_close(self, rpi: int):
        """Drop the born-spilled EXTEND plan at handover (the region is frozen
        from then on and the tick owns every further row)."""
        self._sess_prefill_slots.pop(int(rpi), None)

    def _sess_prefill_join(self):
        """Join the born-spilled prefill's copy-stream D2H (stage B').

        THE ONLY join edge, and it sits at the HANDOVER (U9): during the
        prefill nothing reads the rows back (the chunk attends its own RAGGED
        keys, not the pool), so the copy has a whole scheduler iteration to
        retire behind the compute. Every rank issues this wait at the same
        iteration -- it is a stream wait, not a collective, so differing
        per-rank copy volumes change only the WAIT TIME, never the collective
        sequence."""
        torch.cuda.current_stream().wait_event(self._sess_prefill_done)

    def _sess_prefill_prepare(self, forward_batch):
        """Per-forward state for a born-spilled EXTEND. Derived purely from
        the (replicated) req_pool_idx -> the plan registered at alloc time."""
        rpi = int(forward_batch.req_pool_indices[0].item())
        plan = self._sess_prefill_slots.get(rpi)
        assert plan is not None, (
            "kv-session-offload prefill-spill (PS2): forward flagged as "
            f"born-spilled but no write plan for req_pool_idx {rpi}"
        )
        return plan

    def _sess_prefill_owner_write(self, layer, k_full, v_full):
        """PS2 stage B/B': owner-write of a born-spilled EXTEND chunk.

        Device half: NOTHING. The chunk owns no device KV slots -- that is the
        whole point of "never materialize".

        Host half: select this rank's owned rows, quantise them through the
        staging carve with the stock ``set_kv_buffer`` byte path (identical
        dtype/scale handling, so the stored bytes equal a device write's), then
        copy them D2H into this session's host region rows.

        STAGE B' -- the D2H runs on ``_sess_copy_stream``, not the compute
        stream: on the compute stream a ``chunk_len x per_token_bytes`` copy
        would queue in front of the next layer's compute and stall both this
        prefill and any co-running device session (DESIGN_prefill_spill
        6b.2/6b.4). The fork event is re-recorded per layer and waited by the
        copy stream immediately, so each wait binds to that layer's record
        (single issuing thread -- the same pairing discipline as
        ``_sess_issue_copy``). ``_sess_prefill_done`` is re-recorded on the
        copy stream every layer; the handover waits its LAST recording.

        The weightless-lane owner write (``_wl_spill_owner_write``) keeps its
        current-stream copy untouched -- moving it would change that validated
        lane's behaviour."""
        st = self._sess_prefill_spill
        if st.n_own == 0:
            return
        from sgl_kernel.kvcacheio import transfer_kv_per_layer

        idx = st.own_idx
        self._sess_pool.set_kv_buffer(
            layer,
            st.stage_rows,
            k_full[idx].clone(),
            v_full[idx].clone(),
            layer.k_scale,
            layer.v_scale,
        )
        fl = self._sess_full_layer_idx(layer)
        host = self._sess_host_pool
        cs = self._sess_copy_stream
        # Fork: the staged rows must be written before the copy engine reads
        # them; join only at the handover.
        self._sess_prefill_fork_ev.record(torch.cuda.current_stream())
        with torch.cuda.stream(cs):
            cs.wait_event(self._sess_prefill_fork_ev)
            transfer_kv_per_layer(
                src_k=self._sess_full_pool.k_buffer[fl],
                dst_k=host.k_data_refs[fl],
                src_v=self._sess_full_pool.v_buffer[fl],
                dst_v=host.v_data_refs[fl],
                src_indices=st.stage_rows,
                dst_indices=st.host_rows,
                item_size=host.token_stride_size,
            )
            self._sess_prefill_done.record(cs)

    def _sess_slot_reset_head(self, rpi: int):
        """Force re-derivation of the head/tail split + tail counts on the
        next tick (after a spill or a wave-back grew the device head)."""
        slot = self._sess_slots.get(int(rpi))
        if slot is not None:
            slot.head = None
            slot.count_cache = None

    def _sess_full_layer_idx(self, layer):
        """Index of ``layer`` into the full-attention pool's per-layer k/v
        buffer lists (and the host pool's parallel k_data_refs).

        The two branches index DIFFERENT pools, so they are not one site twice
        (DESIGN_pp_layer_set.md §8.2). With a wrapper, ``_sess_full_pool`` is
        the SUB-pool and the id is re-indexed into its dense frame first.
        Without one, ``_sess_full_pool`` falls back to the MAIN pool and the id
        is still GLOBAL -- an ordinary global -> local translation, which under
        a non-contiguous layer set is a rank lookup rather than a subtraction.
        Routing both through the pool's accessor lets the pool decide.
        """
        pool = self._sess_pool
        if hasattr(pool, "_transfer_full_attention_id"):
            mapped = pool._transfer_full_attention_id(layer.layer_id)
            return self._sess_full_pool.local_slot(mapped)
        return self._sess_full_pool.local_slot(layer.layer_id)

    def _sess_prepare_step(self, forward_batch):
        """Derive the per-step spill state from replicated inputs only:
        seq len + the sentinel req_to_token row. Sets owned counts per rank
        and the RANK-UNIFORM block count. Called from init_forward_metadata
        for a kv_session_spill_tick decode batch.

        S2: owned counts are maintained INCREMENTALLY across ticks (the
        sentinel row only ever appends one token per tick, whose residue is
        a pure function of its position), so the per-tick bincount +
        row-min device syncs of S1 happen only on the first tick after a
        spill. Also builds the flattened layer-major streamed-block
        schedule and primes the double-buffered copy pipeline (depth 2) on
        the dedicated copy stream."""
        from types import SimpleNamespace

        from sglang.srt.managers.kv_session_offload import (
            new_token_residue,
            num_blocks_rank_uniform,
            owned_counts_even,
            owned_counts_weighted,
        )

        # C4 (spec-in-spill-tick): the spill tick's TARGET-VERIFY forward also
        # routes here (bs=1 session, num_draft+1 query rows over the committed
        # prefix). It builds a MINIMAL verify st (owned committed-prefix tail
        # count + device head) and early-returns -- no decode schedule /
        # double-buffer pipeline / spill-graph (the verify twin
        # _sess_blockwise_prefix_return_lse builds its own block loop). The
        # plain decode spill tick keeps its full path below (byte-identical).
        is_verify = forward_batch.forward_mode.is_target_verify()
        assert (
            forward_batch.forward_mode.is_decode() or is_verify
        ), "kv-session-offload: spill tick must be a decode or target-verify batch"
        assert (
            forward_batch.batch_size == 1
        ), "kv-session-offload: exactly one spilled session per tick"
        from sglang.srt.managers.kv_session_offload import owned_device_indices

        # S3: order this tick after any in-flight wave-back H2D so the device
        # head reads the just-restored slots' KV, not stale device memory. A
        # no-op when no wave is pending (event already complete).
        torch.cuda.current_stream().wait_event(self._sess_wave_done)
        L = int(forward_batch.seq_lens_cpu[0].item())
        rank = self.dcp_rank if self._sess_mode != "plain" else 0
        dcp = len(self._sess_prefix) - 1
        lo = self._sess_prefix[rank]
        hi = self._sess_prefix[rank + 1]

        # S4: this tick's session (rpi) -> its persistent per-session slot.
        rpi = int(forward_batch.req_pool_indices[0].item())
        slot = self._sess_slots[rpi]

        # S1b: head/tail split. The device-resident head [0, boundary) keeps
        # real slot ids; the host tail [boundary, L) carries sentinels. The
        # boundary (leading non-sentinel run) is CONSTANT between wave-backs
        # -- new tokens append to the host tail -- so it, and this rank's
        # owned head indices, are derived from the row ONCE and cached on the
        # slot (reset on spill / wave-back). All downstream counts (blocks,
        # host rows, cur_host_row) are TAIL counts; the head is attended
        # separately in _sess_blockwise_decode_return_lse.
        row = None
        boundary, dev_head_idx, n_head_own = 0, None, 0
        head = slot.head
        # C4 verify recomputes its own (suffix-aware) dev/host split below and
        # must NOT read or poison the cached decode-time slot.head (whose
        # boundary would wrongly fold the accepted device suffix into the head).
        if head is None and not is_verify:
            row = self._sess_req_pool.req_to_token[
                forward_batch.req_pool_indices[0], :L
            ]
            boundary = int((row < self._sess_host_base).sum().item())
            if boundary > 0:
                _, dh = owned_device_indices(
                    row[:boundary],
                    mode=self._sess_mode,
                    S=self._sess_S,
                    lo=lo,
                    hi=hi,
                    dcp_size=dcp if self._sess_mode != "plain" else 1,
                    dcp_rank=rank,
                )
                dev_head_idx = dh.to(torch.int32).contiguous()
            else:
                dev_head_idx = None
            n_head_own = 0 if dev_head_idx is None else int(dev_head_idx.numel())
            slot.head = (boundary, dev_head_idx, n_head_own)
        elif head is not None and not is_verify:
            boundary, dev_head_idx, n_head_own = head

        if is_verify:
            # C4 (spec-in-spill-tick VERIFY, Option A -- accepted tokens stay
            # DEVICE-resident): the committed prefix row is
            #   [0, boundary) real HEAD ++ [boundary, spill_L) host SENTINELS ++
            #   [spill_L, L) real accepted SUFFIX
            # -- the FROZEN spilled host region [boundary, spill_L) never grows
            # (draft_dev covers exactly it), and every new (bootstrap / accepted)
            # token lands on device as the suffix, its target+draft KV persisting
            # at its committed candidate slot. So the device-resident set is ALL
            # real slots (head + suffix, NOT contiguous), the host stream is the
            # sentinel block, and the twin's dev_head_idx spans both. Recomputed
            # every verify tick (the suffix grows) -- NOT the cached slot.head.
            # No in-flight current token, no count_cache mutation, no schedule/
            # pipeline/graph. Rank-uniform: derived from the replicated row.
            if row is None:
                row = self._sess_req_pool.req_to_token[
                    forward_batch.req_pool_indices[0], :L
                ]
            sentinel_mask = row >= self._sess_host_base
            host_slots = row[sentinel_mask]  # FROZEN spilled host region
            dev_slots = row[~sentinel_mask]  # head + accepted suffix (device)
            if int(dev_slots.numel()) > 0:
                _, dh = owned_device_indices(
                    dev_slots,
                    mode=self._sess_mode,
                    S=self._sess_S,
                    lo=lo,
                    hi=hi,
                    dcp_size=dcp if self._sess_mode != "plain" else 1,
                    dcp_rank=rank,
                )
                dev_head_idx = dh.to(torch.int32).contiguous()
                n_head_own = int(dev_head_idx.numel())
            else:
                dev_head_idx = None
                n_head_own = 0
            n_host = int(host_slots.numel())
            if self._sess_mode == "weighted":
                residues = host_slots.to(torch.int64) % self._sess_S
                counts = owned_counts_weighted(residues, self._sess_prefix)
            elif self._sess_mode == "even":
                # Positional ownership: the sentinel encodes its absolute
                # position as (slot - host_base) // S; count this rank's owned
                # host positions directly (position-set, not a [0,L) prefix diff).
                pos = (
                    host_slots.to(torch.int64) - self._sess_host_base
                ) // self._sess_S
                counts = [int((pos % dcp == r).sum().item()) for r in range(dcp)]
            else:
                counts = [n_host]
            assert sum(counts) == n_host, (
                f"kv-session-offload: verify host owned counts {counts} do not "
                f"cover host region {n_host} (rid slot-space corrupted)"
            )
            n_own = counts[rank]
            drain = slot.host_row_base
            assert drain + n_own <= self._sess_region_tokens, (
                f"kv-session-offload: verify region high-water {drain + n_own} "
                f"exceeds region ({self._sess_region_tokens} tokens)"
            )
            base = slot.region_base + drain
            return SimpleNamespace(
                verify=True,
                L=L,
                boundary=boundary,
                n_own=n_own,
                counts=counts,
                host_row_base=base,
                dev_head_idx=dev_head_idx,
                n_head_own=n_head_own,
                region_base=slot.region_base,
                device=self._sess_staging_k.device,
            )

        if self._sess_mode == "weighted":
            cache = slot.count_cache
            if cache is not None and cache[0] == L:
                counts = list(cache[1])
            elif cache is not None and cache[0] == L - 1:
                # Incremental: the one appended TAIL token (position L-1) has
                # residue (L-1) % S by the sentinel construction.
                res_new = new_token_residue(L - 1, self._sess_S)
                counts = list(cache[1])
                for r in range(dcp):
                    if self._sess_prefix[r] <= res_new < self._sess_prefix[r + 1]:
                        counts[r] += 1
                        break
            else:
                if row is None:
                    row = self._sess_req_pool.req_to_token[
                        forward_batch.req_pool_indices[0], :L
                    ]
                tailrow = row[boundary:L]
                assert tailrow.numel() == 0 or (
                    int(tailrow.min().item()) >= self._sess_host_base
                ), (
                    "kv-session-offload: non-sentinel slot id in a spill-tick "
                    "tail -- slot space corrupted, refusing to attend garbage KV"
                )
                residues = tailrow.to(torch.int64) % self._sess_S
                counts = owned_counts_weighted(residues, self._sess_prefix)
                assert sum(counts) == L - boundary, (
                    f"kv-session-offload: tail owned counts {counts} do not "
                    f"cover L-boundary={L - boundary}"
                )
            slot.count_cache = (L, tuple(counts))
            res_cur = new_token_residue(L - 1, self._sess_S)
            cur_owned = self._sess_prefix[rank] <= res_cur < self._sess_prefix[rank + 1]
        elif self._sess_mode == "even":
            full = owned_counts_even(L, dcp)
            headc = owned_counts_even(boundary, dcp)
            counts = [full[r] - headc[r] for r in range(dcp)]
            cur_owned = (L - 1) % dcp == rank
        else:
            counts = [L - boundary]
            cur_owned = True
        n_own = counts[rank]
        num_blocks = num_blocks_rank_uniform(counts, self._sess_block_size)
        # S3/S4: within this session's region, the active tail owns local host
        # rows [drain, drain + n_own); wave-back drained [0, drain). The
        # absolute host row is region_base + local. host_next = drain + n_own
        # is the high-water against the REGION capacity.
        drain = slot.host_row_base
        assert drain + n_own <= self._sess_region_tokens, (
            f"kv-session-offload: region high-water {drain + n_own} exceeds "
            f"the per-session region ({self._sess_region_tokens} tokens)"
        )
        base = slot.region_base + drain

        # Flattened layer-major streamed-block schedule. Every full-attn
        # layer streams the same k_local non-empty local blocks; regions
        # alternate in GLOBAL order (g & 1) so the copy of block g+1 --
        # also across a layer boundary -- overlaps attention on block g.
        # The tail block's LAST row is this tick's current token when this
        # rank owns it: that row is produced layer-by-layer DURING the
        # forward (owner-write), so the bulk H2D prefetch excludes it and a
        # tiny D2D fixup from the scratch row fills it at consume time.
        B = self._sess_block_size
        k_local = (n_own + B - 1) // B if n_own > 0 else 0
        layer_num = getattr(self._sess_host_pool, "layer_num", 0)
        sched = []
        g = 0
        for fl in range(layer_num):
            for j in range(k_local):
                s = j * B
                e = min((j + 1) * B, n_own)
                sched.append((fl, s, e, g & 1, cur_owned and e == n_own))
                g += 1
        st = SimpleNamespace(
            L=L,
            boundary=boundary,
            n_own=n_own,
            counts=counts,
            num_blocks=num_blocks,
            cur_owned=cur_owned,
            # host_row_base offsets the LOCAL block offsets (s, e in [0, n_own))
            # to ACTUAL host rows [base + s, base + e); the current token's row
            # is base + n_own - 1.
            host_row_base=base,
            cur_host_row=base + n_own - 1 if cur_owned else -1,
            cur_host_row_t=(
                self._sess_host_arange[base + n_own - 1 : base + n_own]
                if cur_owned
                else None
            ),
            device=self._sess_staging_k.device,
            sched=sched,
            k_local=k_local,
            consume_idx=0,
            issue_idx=0,
            # S1b device head: this rank's owned slots in [0, boundary) (real
            # pool indices, int32) attended once per layer alongside the host
            # tail blocks, LSE-merged. None when boundary == 0 (whole-suffix
            # spill) or this rank owns no head token.
            dev_head_idx=dev_head_idx,
            n_head_own=n_head_own,
            # S5: fixed-count graph rung + out-of-graph plan (None -> eager).
            region_base=slot.region_base,
            graph_rung=None,
            graph_plan=None,
        )
        # S5 spill-graph: pick the rank-uniform rung + build the out-of-graph
        # staging plan (the #136a plan hoist -- once per STEP, never inside the
        # captured region). num_blocks is already the MAX-over-ranks block
        # count, so the rung is rank-uniform and every rank captures/replays
        # the same fixed shape; per-rank extra blocks are (0,-inf)-sanitized
        # no-ops. Over-ladder (or the flag off) -> graph_rung None -> the
        # eager block loop below runs unchanged (byte-identical). The captured
        # body (_sess_blockwise_decode_return_lse_graph) is invoked by the
        # spill-tick graph runner during replay; eager prepare still fills the
        # plan so a first (uncaptured) pass is correct.
        if self._sess_graph_enabled and self._sess_graph_ladder:
            from sglang.srt.managers.kv_session_offload import (
                spill_graph_out_plan,
                spill_graph_pick_rung,
            )

            rung = spill_graph_pick_rung(num_blocks, self._sess_graph_ladder)
            st.graph_rung = rung
            if rung is not None:
                st.graph_plan = spill_graph_out_plan(
                    base, n_own, B, rung, device=st.device
                )
            # S5 graph prep (the #136a .plan() hoist -- ONCE per step, out of
            # graph). During CAPTURE the runner set _sess_graph_capture_blocks;
            # plan every rung wrapper at worst-case. During a graph REPLAY step
            # (_sess_graph_replay_blocks set by _sess_graph_can_replay) refill
            # the fixed buffers from st.graph_plan. Plain eager ticks skip this.
            # GPU-JUSTIFICATION: the exact ordering of prepare vs the runner's
            # capture/replay hooks (whether this or init_forward_metadata_out_
            # graph drives replay prep) is validated on GPU -- mirror the _wl
            # sequence (out_graph -> _wl_graph_prepare_blocks) if adjustment is
            # needed.
            self._sess_spill = st  # prepare_blocks reads _sess_spill
            if self._sess_graph_capture_blocks is not None:
                self._sess_graph_prepare_blocks(in_capture=True)
            elif (
                rung is not None
                and self._sess_graph_replay_blocks == rung
                and self._sess_graph_captured(rung)
            ):
                # Refill the captured rung's fixed buffers for this replay. On
                # GPU the runner may instead drive replay prep from its
                # out-of-graph hook (as _wl does via
                # init_forward_metadata_out_graph -> _wl_graph_prepare_blocks);
                # if so, move this call there. Gated on _sess_graph_captured so
                # flag-on-without-capture never builds graph state.
                self._sess_graph_prepare_blocks(in_capture=False)
        # Prime the double buffer for the EAGER block loop. Skip it on the
        # graph path (capture or an admitted replay): the graph body issues its
        # own copies, and priming here would run copy-stream ops outside the
        # captured region.
        graph_active = self._sess_graph_capture_blocks is not None or (
            st.graph_rung is not None
            and self._sess_graph_replay_blocks == st.graph_rung
            and self._sess_graph_captured(st.graph_rung)
        )
        if not graph_active:
            while st.issue_idx < min(2, len(sched)):
                self._sess_issue_copy(st, st.issue_idx)
                st.issue_idx += 1
        return st

    def _sess_issue_copy(self, st, idx):
        """Issue the H2D staging copy for flattened block ``idx`` on the
        dedicated copy stream (never the compute stream). Waits the
        region's last consumer via event; all record/wait calls come from
        the single forward thread, so the pairing is race-free."""
        from sgl_kernel.kvcacheio import transfer_kv_per_layer

        fl, s, e, region, need_cur = st.sched[idx]
        pe = e - 1 if need_cur else e  # current token's row arrives via D2D fixup
        host = self._sess_host_pool
        B = self._sess_block_size
        hb = st.host_row_base  # S3: local block offsets -> actual host rows
        cs = self._sess_copy_stream
        with torch.cuda.stream(cs):
            cs.wait_event(self._sess_compute_done[region])
            if pe > s:
                base = region * B
                transfer_kv_per_layer(
                    src_k=host.k_data_refs[fl],
                    dst_k=self._sess_staging_k,
                    src_v=host.v_data_refs[fl],
                    dst_v=self._sess_staging_v,
                    src_indices=self._sess_host_arange[hb + s : hb + pe],
                    dst_indices=self._sess_stage_arange64[base : base + (pe - s)],
                    item_size=host.token_stride_size,
                )
            self._sess_copy_done[region].record(cs)

    def _sess_owner_write(self, layer, k_full, v_full):
        """Spill-tick owner-write for one full-attention layer: quantize the
        new token's K/V through the device scratch row (stock set_kv_buffer
        byte path) and D2H-copy it to this rank's next host row. Rank-local;
        no collectives."""
        st = self._sess_spill
        # S5: during graph capture route to the fixed-shape graph owner-write
        # (always writes -- real host row or dump slot -- no cur_owned branch).
        if self._sess_graph_capture_blocks is not None:
            self._sess_owner_write_graph(layer, k_full, v_full)
            return
        if not st.cur_owned:
            return
        from sgl_kernel.kvcacheio import transfer_kv_per_layer

        self._sess_pool.set_kv_buffer(
            layer,
            self._sess_scratch_loc,
            k_full.clone(),
            v_full.clone(),
            layer.k_scale,
            layer.v_scale,
        )
        fl = self._sess_full_layer_idx(layer)
        host = self._sess_host_pool
        dst_row = st.cur_host_row_t
        transfer_kv_per_layer(
            src_k=self._sess_full_pool.k_buffer[fl],
            dst_k=host.k_data_refs[fl],
            src_v=self._sess_full_pool.v_buffer[fl],
            dst_v=host.v_data_refs[fl],
            src_indices=self._sess_scratch_loc,
            dst_indices=dst_row,
            item_size=host.token_stride_size,
        )

    def _sess_get_wrapper(self, key):
        """Wrapper pool for the streamed partial decodes: one persistent
        flashinfer decode wrapper per (kind, region) slot, sharing the
        backend float workspace (safe: on a spill tick the main decode
        wrapper is never .run() for the full-attention layers, and all
        .run() calls are stream-ordered). Each wrapper caches its planned
        staged-row count in ``_sess_planned``: the 'full' slots plan ONCE
        EVER (cnt == B, fixed indices), the 'tail' slots replan only when
        the tail size changed -- i.e. at most twice per tick instead of the
        S1 plan-per-layer-per-block."""
        w = self._sess_wrappers.get(key)
        if w is None:
            w = BatchDecodeWithPagedKVCacheWrapper(
                self._sess_workspace_buffer,
                "NHD",
                backend=self.decode_backend,
                use_tensor_cores=self.decode_use_tensor_cores,
            )
            w._sess_planned = None
            self._sess_wrappers[key] = w
        return w

    def _sess_plan_block(self, cnt, region, num_qo_heads, num_kv_heads, head_dim):
        """Planned wrapper for a staged block of ``cnt`` rows in staging
        region ``region`` (plan-cache hit unless the slot's cnt changed)."""
        B = self._sess_block_size
        iu = self.indices_updater_decode
        w = self._sess_get_wrapper(("full" if cnt == B else "tail", region))
        if w._sess_planned != cnt:
            base = region * B
            blk_indptr = torch.tensor(
                [0, cnt], dtype=torch.int32, device=self._sess_staging_k.device
            )
            w.plan(
                blk_indptr,
                self._sess_stage_arange32[base : base + cnt],
                self.kv_last_page_len[:1],
                num_qo_heads,
                num_kv_heads,
                head_dim,
                1,  # page_size
                q_data_type=iu.q_data_type,
                kv_data_type=iu.data_type,
            )
            w._sess_planned = cnt
        return w

    def _sess_plan_dev_head(self, num_qo_heads, num_kv_heads, head_dim):
        """Persistent decode wrapper for this rank's device-resident head
        [0, boundary) of a PARTIAL (S1b) spill. Indices are REAL-pool slot
        ids, frozen for the spill's life, so the plan is built ONCE per tick
        (guarded on ``st``) and reused across every full-attention layer.
        Returns None when this rank owns no head token (boundary == 0 or the
        rank's head shard is empty) -- then only the host tail is attended."""
        st = self._sess_spill
        idx = st.dev_head_idx
        if idx is None or int(idx.numel()) == 0:
            return None
        w = self._sess_get_wrapper(("dev_head",))
        if not getattr(st, "_dev_head_planned", False):
            iu = self.indices_updater_decode
            blk_indptr = torch.tensor(
                [0, int(idx.numel())], dtype=torch.int32, device=idx.device
            )
            w.plan(
                blk_indptr,
                idx,
                self.kv_last_page_len[:1],
                num_qo_heads,
                num_kv_heads,
                head_dim,
                1,  # page_size
                q_data_type=iu.q_data_type,
                kv_data_type=iu.data_type,
            )
            st._dev_head_planned = True
        return w

    def _sess_blockwise_decode_return_lse(self, q_full, layer):
        """Spill-tick decode attention over the host-resident shard (DCP
        form): consume this layer's streamed blocks from the double-
        buffered staging pipeline (H2D prefetch on the copy stream, plans
        cached) + flashinfer partial decodes + online LSE merge. Returns
        (o, lse) in the same shape/space as ``forward_return_lse`` so the
        cross-rank cp_lse merge consumes it unchanged. The loop count is
        rank-uniform (num_blocks) with empty local blocks skipped WITHOUT
        any collective (none is issued inside the loop).

        S5 spill-graph dispatch hook: when ``st.graph_rung`` is set (flag on +
        graphable) the spill-tick graph runner replays the fixed-count
        captured body instead of this eager loop -- it iterates EXACTLY
        ``st.graph_rung`` blocks from ``st.graph_plan`` (built out-of-graph in
        _sess_prepare_step), running each {H2D gather + pre-planned wrapper +
        merge} with empty trailing blocks sanitized to (0,-inf) in-graph (the
        #136a mechanics, region-scoped to this session's active tail). The
        capture/replay wiring + validation is the GPU pass; until then this
        eager loop runs unchanged (byte-identical), so ``graph_rung`` set with
        no active capture context is simply ignored here."""
        # S5 dispatch: while a rung is being CAPTURED (the decode graph runner
        # set _sess_graph_capture_blocks), record the fixed-count graph body;
        # replay re-runs the recorded ops (no Python re-entry). Every other
        # spill tick runs the eager loop below (byte-identical).
        if self._sess_graph_capture_blocks is not None:
            return self._sess_blockwise_decode_return_lse_graph(q_full, layer)
        st = self._sess_spill
        B = self._sess_block_size
        fl = self._sess_full_layer_idx(layer)
        num_qo_heads = q_full.shape[1]
        num_kv_heads = self.indices_updater_decode.num_kv_heads
        head_dim = q_full.shape[2]
        kv = self._sess_staging_kv
        cur = torch.cuda.current_stream()

        o_acc = None
        lse_acc = None
        # S1b HYBRID attention: attend the device-resident head [0, boundary)
        # from the REAL pool first (fast, no host stream), then online-merge
        # the host tail blocks below. Both partials are the same flashinfer
        # paged-decode family -> the same _safe_merge_state operator, so the
        # (o, lse) contract handed to the cross-rank cp_lse merge is
        # unchanged. Intra-rank: no collective is issued here.
        dev_w = self._sess_plan_dev_head(num_qo_heads, num_kv_heads, head_dim)
        if dev_w is not None:
            real_kv = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)
            o_acc, lse_acc = dev_w.forward_return_lse(
                q_full,
                real_kv,
                sm_scale=layer.scaling,
                logits_soft_cap=layer.logit_cap,
                k_scale=layer.k_scale_float,
                v_scale=layer.v_scale_float,
            )
        for _ in range(st.k_local):
            ent = st.sched[st.consume_idx]
            e_fl, s_row, e_row, region, need_cur = ent
            assert e_fl == fl, (
                f"kv-session-offload: streamed-block schedule out of order "
                f"(expected layer {e_fl}, consuming for layer {fl})"
            )
            # Wait the region's prefetch (recorded on the copy stream).
            cur.wait_event(self._sess_copy_done[region])
            if need_cur:
                # The tick's own token: its host row is produced THIS layer
                # by the owner-write (scratch row holds the just-quantized
                # K/V) -- tiny D2D into its staging slot on the compute
                # stream, ordered after the owner-write by stream order.
                dst = region * B + (e_row - 1 - s_row)
                sc = self._sess_scratch_i
                self._sess_staging_k[dst].copy_(self._sess_full_pool.k_buffer[fl][sc])
                self._sess_staging_v[dst].copy_(self._sess_full_pool.v_buffer[fl][sc])
            wrapper = self._sess_plan_block(
                e_row - s_row, region, num_qo_heads, num_kv_heads, head_dim
            )
            o_b, lse_b = wrapper.forward_return_lse(
                q_full,
                kv,
                sm_scale=layer.scaling,
                logits_soft_cap=layer.logit_cap,
                k_scale=layer.k_scale_float,
                v_scale=layer.v_scale_float,
            )
            # Release the region + refill the pipeline (depth 2: the copy
            # just issued targets the region we released above).
            self._sess_compute_done[region].record(cur)
            st.consume_idx += 1
            if st.issue_idx < len(st.sched):
                self._sess_issue_copy(st, st.issue_idx)
                st.issue_idx += 1
            if o_acc is None:
                o_acc, lse_acc = o_b, lse_b
            else:
                o_acc, lse_acc = _safe_merge_state(o_acc, lse_acc, o_b, lse_b)
        if o_acc is None:
            # This rank owns ZERO tokens of the session: reproduce the
            # shaped empty-attention (o=0, lse=-inf) contract so the
            # cross-rank cp_lse merge stays balanced (same as the _wl loop).
            w = self._sess_get_wrapper(("empty",))
            if w._sess_planned != 0:
                empty_indptr = torch.zeros(2, dtype=torch.int32, device=st.device)
                w.plan(
                    empty_indptr,
                    self._sess_stage_arange32[:0],
                    self.kv_last_page_len[:1],
                    num_qo_heads,
                    num_kv_heads,
                    head_dim,
                    1,
                    q_data_type=self.indices_updater_decode.q_data_type,
                    kv_data_type=self.indices_updater_decode.data_type,
                )
                w._sess_planned = 0
            o_acc, lse_acc = w.forward_return_lse(
                q_full,
                kv,
                sm_scale=layer.scaling,
                logits_soft_cap=layer.logit_cap,
                k_scale=layer.k_scale_float,
                v_scale=layer.v_scale_float,
            )
        return o_acc, lse_acc

    def _sess_forward_decode_plain(self, q, k, v, layer, save_kv_cache):
        """Spill-tick decode for plain TP (no DCP): every rank streams the
        WHOLE session shard (its kv-head slice) from host; the block-merged
        output needs no cross-rank LSE step (heads are sharded, exactly like
        the monolithic non-DCP decode)."""
        if k is not None and save_kv_cache:
            assert v is not None
            self._sess_owner_write(
                layer,
                k.view(-1, layer.tp_k_head_num, layer.head_dim),
                v.view(-1, layer.tp_v_head_num, layer.head_dim),
            )
        q_local = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
        o, _ = self._sess_blockwise_decode_return_lse(q_local, layer)
        return o.view(-1, layer.tp_q_head_num * layer.head_dim)

    # ==================================================================
    def _sess_verify_active(self) -> bool:
        """True on a spec-in-spill-tick target-VERIFY forward whose committed
        prefix is host-resident: route the DCP prefix read to the C4 twin
        (_sess_blockwise_prefix_return_lse) instead of the monolithic paged
        read. False for the plain decode spill tick and every non-spill batch
        (getattr default), so those paths stay byte-identical."""
        st = self._sess_spill
        return st is not None and getattr(st, "verify", False)

    # C4 (spec-in-spill-tick): target-VERIFY host-prefix attention. The
    # spill-tick verify forward has bs = num_draft+1 QUERY rows (the
    # candidate chain); they all attend the SAME committed prefix (device
    # head [0, boundary) + host tail [boundary, L)) NON-CAUSALLY (every
    # candidate lies strictly after the whole prefix). This is the direct
    # ``_sess`` twin of the weightless-KV extend prefix read
    # ``_wl_blockwise_prefix_return_lse`` (:4507): same online _safe_merge_
    # state, same (o, lse) prefill-paged contract, only the host-block SOURCE
    # is this session's spill region instead of the _wl host pool. Phase 1 is
    # EAGER (no spill-graph). The candidate-tail tree-masked partial and the
    # cross-rank cp_lse merge are handled by the caller (_forward_extend_dcp
    # target_verify branch) exactly as for the monolithic paged read.
    # ------------------------------------------------------------------

    def _sess_prefix_prefill_wrapper(self):
        """Persistent paged PREFILL wrapper for the C4 host-tail block read
        (multi-row query). Shares the spill-lane workspace -- safe: on a spill
        tick the main paged prefix wrapper is never .run() for the full-attn
        layers (this loop replaces it), and every .run() here is stream-
        ordered on the compute stream."""
        w = getattr(self, "_sess_prefix_prefill_wrapper_obj", None)
        if w is None:
            w = BatchPrefillWithPagedKVCacheWrapper(
                self._sess_workspace_buffer,
                "NHD",
                backend=self.prefill_backend,
            )
            w._sess_planned = None
            self._sess_prefix_prefill_wrapper_obj = w
        return w

    def _sess_dev_head_prefill_wrapper(self):
        """Persistent paged PREFILL wrapper for the C4 device-resident head
        [0, boundary) read (multi-row query, REAL pool slot ids)."""
        w = getattr(self, "_sess_dev_head_prefill_wrapper_obj", None)
        if w is None:
            w = BatchPrefillWithPagedKVCacheWrapper(
                self._sess_workspace_buffer,
                "NHD",
                backend=self.prefill_backend,
            )
            w._sess_planned = None
            self._sess_dev_head_prefill_wrapper_obj = w
        return w

    def _sess_blockwise_prefix_return_lse(self, q_full, layer, logits_soft_cap=None):
        """C4 target-VERIFY twin of ``_wl_blockwise_prefix_return_lse``:
        multi-row (num_draft+1) NON-CAUSAL paged read of this rank's committed
        prefix for the spill tick -- device-resident head [0, boundary) from
        the real pool + the host-resident tail [0, n_own) streamed blockwise
        from the session spill region -- online-merged with the SAME
        _safe_merge_state operator (both partials are the same flashinfer
        paged-prefill family, identical byte class). Returns (o, lse) in the
        same shape/space as ``prefill_wrapper_paged.forward_return_lse`` so the
        caller's cross-rank cp_lse merge and the candidate-tail logaddexp
        combine are UNCHANGED.

        Intra-rank: NO collective is issued inside the loop. The block count is
        rank-uniform (derived from the per-rank owned tail counts in
        _sess_prepare_step's st, whose max is rank-uniform), so every DCP rank
        streams the same number of blocks -> the caller's per-layer collective
        count is preserved. Correctness-first EAGER staging: each block is
        staged H2D on the compute stream, planned, attended, merged, in order
        (no double-buffer prefetch -- that is a Phase-1.x perf refinement; the
        decode lane keeps its pipeline)."""
        from sgl_kernel.kvcacheio import transfer_kv_per_layer

        st = self._sess_spill
        B = self._sess_block_size
        fl = self._sess_full_layer_idx(layer)
        Q = q_full.shape[0]
        num_qo_heads = q_full.shape[1]
        iu = self.indices_updater_prefill
        num_kv_heads = iu.num_kv_heads
        head_dim = q_full.shape[2]
        if logits_soft_cap is None:
            logits_soft_cap = layer.logit_cap
        dev = self._sess_staging_k.device
        qo_indptr = torch.tensor([0, Q], dtype=torch.int32, device=dev)
        last_page = self.kv_last_page_len
        kv_stage = self._sess_staging_kv
        host = self._sess_host_pool
        n_own = st.n_own
        base = st.host_row_base

        o_acc = None
        lse_acc = None
        # (1) Device-resident head [0, boundary): REAL pool slots, planned once
        # per tick (indices frozen for the spill's life), attended non-causal.
        idx = st.dev_head_idx
        if idx is not None and int(idx.numel()) > 0:
            w = self._sess_dev_head_prefill_wrapper()
            head_indptr = torch.tensor(
                [0, int(idx.numel())], dtype=torch.int32, device=idx.device
            )
            w.plan(
                qo_indptr,
                head_indptr,
                idx,
                last_page[:1],
                num_qo_heads,
                num_kv_heads,
                head_dim,
                1,  # page_size
                causal=False,
                q_data_type=iu.q_data_type,
                kv_data_type=iu.data_type,
            )
            real_kv = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)
            o_acc, lse_acc = w.forward_return_lse(
                q_full,
                real_kv,
                causal=False,
                sm_scale=layer.scaling,
                logits_soft_cap=logits_soft_cap,
                k_scale=layer.k_scale_float,
                v_scale=layer.v_scale_float,
            )
        # (2) Host tail [0, n_own): stream in B-row blocks from the session
        # spill region into staging region 0, attend non-causal, online-merge.
        w = self._sess_prefix_prefill_wrapper()
        k_local = (n_own + B - 1) // B if n_own > 0 else 0
        for j in range(k_local):
            s = j * B
            e = min((j + 1) * B, n_own)
            cnt = e - s
            transfer_kv_per_layer(
                src_k=host.k_data_refs[fl],
                dst_k=self._sess_staging_k,
                src_v=host.v_data_refs[fl],
                dst_v=self._sess_staging_v,
                src_indices=self._sess_host_arange[base + s : base + e],
                dst_indices=self._sess_stage_arange64[:cnt],
                item_size=host.token_stride_size,
            )
            blk_indptr = torch.tensor([0, cnt], dtype=torch.int32, device=dev)
            w.plan(
                qo_indptr,
                blk_indptr,
                self._sess_stage_arange32[:cnt],
                last_page[:1],
                num_qo_heads,
                num_kv_heads,
                head_dim,
                1,  # page_size
                causal=False,
                q_data_type=iu.q_data_type,
                kv_data_type=iu.data_type,
            )
            o_b, lse_b = w.forward_return_lse(
                q_full,
                kv_stage,
                causal=False,
                sm_scale=layer.scaling,
                logits_soft_cap=logits_soft_cap,
                k_scale=layer.k_scale_float,
                v_scale=layer.v_scale_float,
            )
            if o_acc is None:
                o_acc, lse_acc = o_b, lse_b
            else:
                o_acc, lse_acc = _safe_merge_state(o_acc, lse_acc, o_b, lse_b)
        if o_acc is None:
            # This rank owns ZERO prefix slots -> shaped empty attention
            # (o=0, lse=-inf) so the caller's cross-rank cp_lse merge stays
            # balanced (same contract as the decode/_wl empty branch).
            w = self._sess_prefix_prefill_wrapper()
            empty_indptr = torch.zeros(2, dtype=torch.int32, device=dev)
            w.plan(
                qo_indptr,
                empty_indptr,
                self._sess_stage_arange32[:0],
                last_page[:1],
                num_qo_heads,
                num_kv_heads,
                head_dim,
                1,
                causal=False,
                q_data_type=iu.q_data_type,
                kv_data_type=iu.data_type,
            )
            o_acc, lse_acc = w.forward_return_lse(
                q_full,
                kv_stage,
                causal=False,
                sm_scale=layer.scaling,
                logits_soft_cap=logits_soft_cap,
                k_scale=layer.k_scale_float,
                v_scale=layer.v_scale_float,
            )
        return o_acc, lse_acc

    def _sess_attn_selftest(self, model_runner):
        """ENV-gated (KVSO_ATTN_SELFTEST) WITHIN-BOOT identical-input
        correctness proof of the C4 target-verify host-prefix attention
        (_sess_blockwise_prefix_return_lse). Default OFF -> byte-inert.

        Self-contained (no _sess_prepare_step / no big-B / no big pool): builds
        a SMALL synthetic committed prefix (a device-resident head + a
        multi-block host tail, tiny so it fits any device pool), materializes
        KNOWN K/V BYTE-EXACT into both the device slots and the host tail rows
        (device slots written through get_kv_buffer, then transfer_kv_per_layer
        D2H so the host bytes are identical to the device mirror -- the same
        kernel the real spill uses), samples num_draft+1 NON-CAUSAL query rows,
        and compares WITHIN ONE BOOT on IDENTICAL inputs:
          * PATH T (twin): the host-stream blockwise device-head+tail attention
            over the SPILL region (the C4 code under test);
          * PATH R (ref):  the SAME committed-prefix KV gathered CONTIGUOUS on
            device, attended by ONE plain paged prefill wrapper (non-causal).
        Because T reads byte-identical KV, the ONLY gap T-vs-R is bounded
        blockwise fp reassociation (decode-class); a WRONG mask/merge/host-read
        diverges grossly. This is the silent-wrong-attention guard -- the caller
        (verify) then only needs accept-len health, since spec is output-
        preserving by construction. Intra-rank (no collective) -> rank-uniform,
        no desync risk. Runs a small block-size ladder to exercise the
        multi-block _safe_merge_state chain."""
        import os
        from types import SimpleNamespace

        from sgl_kernel.kvcacheio import transfer_kv_per_layer

        if not os.environ.get("KVSO_ATTN_SELFTEST"):
            return
        rank = getattr(model_runner, "tp_rank", 0)
        layers = [
            m
            for m in model_runner.model.modules()
            if type(m).__name__ == "RadixAttention"
        ]
        layers.sort(key=lambda a: a.layer_id)
        layers = [ly for ly in layers if self._is_full_attention_layer(ly)] or layers
        if not layers:
            logger.warning("kvso attn selftest: no attention layer found")
            return
        layer = layers[0]
        dev = self._sess_staging_k.device
        iu = self.indices_updater_prefill
        num_qo = sum(self.dcp_q_head_counts) if self.uneven_dcp else iu.num_qo_heads
        num_kv = iu.num_kv_heads
        head_dim = iu.head_dim
        fl = self._sess_full_layer_idx(layer)
        # num_draft+1 verify query rows. Use the server's speculative width when
        # configured, else a representative 4 (the attention is width-generic).
        sa = model_runner.server_args
        ndraft = int(getattr(sa, "speculative_num_draft_tokens", 0) or 0)
        Q = ndraft if ndraft > 0 else 4
        torch.manual_seed(20260724)
        qdt = getattr(iu, "q_data_type", torch.bfloat16)
        if not isinstance(qdt, torch.dtype):
            qdt = torch.bfloat16
        q = torch.randn(Q, num_qo, head_dim, device=dev).to(qdt)

        # SMALL synthetic sizes (fit any device pool): override the block size
        # to a tiny value so a few-hundred-token tail spans multiple blocks.
        Bt = 16
        head_n = 24  # device-resident head slots [1, 1+head_n)
        rungs = [1, 2, 3]  # tail = R*Bt blocks -> exercises multi-block merge
        kdev, vdev = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)
        real_k = self._sess_full_pool.k_buffer[fl]  # raw store (parallels host)
        real_v = self._sess_full_pool.v_buffer[fl]
        host = self._sess_host_pool
        head_slots = torch.arange(1, 1 + head_n, device=dev, dtype=torch.int64)

        def _plain_prefix_ref(idx_dev):
            """Reference: ONE plain paged prefill wrapper over the contiguous
            device slots `idx_dev` (the whole committed prefix), non-causal."""
            w = BatchPrefillWithPagedKVCacheWrapper(
                self._sess_workspace_buffer, "NHD", backend=self.prefill_backend
            )
            n = int(idx_dev.numel())
            w.plan(
                torch.tensor([0, Q], dtype=torch.int32, device=dev),
                torch.tensor([0, n], dtype=torch.int32, device=dev),
                idx_dev.to(torch.int32),
                self.kv_last_page_len[:1],
                num_qo,
                num_kv,
                head_dim,
                1,
                causal=False,
                q_data_type=iu.q_data_type,
                kv_data_type=iu.data_type,
            )
            o, _lse = w.forward_return_lse(
                q,
                (kdev, vdev),
                causal=False,
                sm_scale=layer.scaling,
                logits_soft_cap=layer.logit_cap,
                k_scale=layer.k_scale_float,
                v_scale=layer.v_scale_float,
            )
            return o

        def _write_kv(dev_slots, host_rows):
            """Fill `dev_slots` (device) with fresh random KV through the
            logical get_kv_buffer view, then D2H-mirror the SAME bytes into
            `host_rows` so the host tail is byte-identical to the device
            mirror (transfer_kv_per_layer = the real spill's copy kernel)."""
            n = int(dev_slots.numel())
            kk = torch.randn(n, num_kv, head_dim, device=dev).to(kdev.dtype)
            vv = torch.randn(n, num_kv, head_dim, device=dev).to(vdev.dtype)
            kdev[dev_slots] = kk
            vdev[dev_slots] = vv
            if host_rows is not None:
                transfer_kv_per_layer(
                    src_k=real_k,
                    dst_k=host.k_data_refs[fl],
                    src_v=real_v,
                    dst_v=host.v_data_refs[fl],
                    src_indices=dev_slots,
                    dst_indices=host_rows,
                    item_size=host.token_stride_size,
                )

        rows = []
        saved_block = self._sess_block_size
        try:
            self._sess_block_size = Bt
            # device head KV (shared by twin's head-read and the reference)
            _write_kv(head_slots, None)
            for R in rungs:
                tail_n = R * Bt
                host_rows = self._sess_host_arange[0:tail_n]  # region base 0
                # device mirror slots for the tail (contiguous, after the head)
                dev_tail = torch.arange(
                    1 + head_n, 1 + head_n + tail_n, device=dev, dtype=torch.int64
                )
                _write_kv(dev_tail, host_rows)
                # synthetic spill state the twin reads (fields: n_own,
                # host_row_base, dev_head_idx). boundary/head_n>0 -> the
                # head+tail LSE merge is exercised.
                st = SimpleNamespace(
                    n_own=tail_n,
                    host_row_base=0,
                    dev_head_idx=head_slots.to(torch.int32).contiguous(),
                )
                self._sess_spill = st
                # PATH T: the C4 twin (device head from real pool + host stream)
                o_t, _ = self._sess_blockwise_prefix_return_lse(q, layer)
                o_t = o_t.clone()
                # PATH R: monolithic contiguous device prefill over the SAME
                # committed prefix ([head slots] ++ [device tail mirror]).
                ref_idx = torch.cat([head_slots, dev_tail])
                o_r = _plain_prefix_ref(ref_idx)
                md = float((o_t - o_r).abs().max().item())
                denom = max(1e-6, float(o_r.abs().max().item()))
                rel = md / denom
                nan = bool(torch.isnan(o_t).any() or torch.isnan(o_r).any())
                # bf16 blockwise reassociation is decode-class (~1e-2 rel);
                # PASS well under that, FAIL on gross (wrong mask/merge/read).
                ok = (not nan) and rel < 5e-2
                verdict = (
                    "NAN"
                    if nan
                    else (
                        "MACHINE_ZERO"
                        if md == 0.0
                        else f"{'PASS' if ok else 'FAIL'} maxd={md:.3e} rel={rel:.3e}"
                    )
                )
                rows.append((R, tail_n, head_n, verdict))
        finally:
            self._sess_block_size = saved_block
            self._sess_spill = None
        logger.info(
            "==== kvso C4 VERIFY-attn SELFTEST (rank %d) twin-vs-monolithic, "
            "Q=%d head=%d block=%d (residual = bounded blockwise fp "
            "reassociation; gross divergence = wrong mask/merge/host-read) ====",
            rank,
            Q,
            head_n,
            Bt,
        )
        for R, tail_n, head_n_, verdict in rows:
            logger.info(
                "  blocks=%-3d owned_tail=%-6d dev_head=%-6d: %s",
                R,
                tail_n,
                head_n_,
                verdict,
            )
        logger.info("==== kvso C4 VERIFY-attn SELFTEST done (rank %d) ====", rank)

    # ==================================================================
    # S5: CUDA-graph capture/replay of the bs=1 spill tick. PORT of the
    # proven weightless-KV #136a mechanism (_wl_graph_bucket_state /
    # _wl_graph_prepare_blocks / _wl_blockwise_decode_return_lse_graph /
    # wl_graph_can_replay), region-scoped to ONE spilled session's active
    # host tail. Differences from _wl (the S1b hybrid): (a) EVERY block is a
    # pure host block (no per-block device/host tier split -- the device HEAD
    # is attended separately by a dedicated graph wrapper), (b) the host tail
    # rows are ALWAYS densely packed [base, base+n_own) so no contiguity gate
    # is needed, (c) the dev-head wrapper is captured-at-max / replayed-below
    # (its size varies per session, independent of the tail rung).
    #
    # GPU-JUSTIFICATION (the messagent iterates these; the STRUCTURE mirrors
    # the working _wl port so the tuning is minimal):
    #   * the capture harness builds a SYNTHETIC spill-tick batch per rung
    #     (decode graph runner) -- capture-region boundaries;
    #   * the dev-head capture-at-max count (`head_max`);
    #   * #136b side-stream prefetch is DEFERRED (single-region serial copies
    #     on the main stream, the byte-identical #136a-no-prefetch behavior).
    # ------------------------------------------------------------------

    def _sess_install_capture_state(self, bs: int, rung: int):
        """Context manager: install a SYNTHETIC spilled-session state so the
        per-rung capture records the rung worst case (every block FULL, head at
        max), then tear it down. Returns a contextmanager used by the decode
        graph runner's spill capture pass.

        Sets `_sess_capture_active` -> init_forward_metadata_out_graph treats
        the capture batch as a spill tick and runs _sess_prepare_step +
        _sess_graph_prepare_blocks(in_capture=True).

        GPU-JUSTIFICATION (messagent wires/iterates on GPU):
          * the reserved capture req_pool row (must not clobber a live request;
            during warmup capture the pool is idle -- confirm on GPU);
          * the synthetic seq_len / sentinel row that makes _sess_prepare_step
            yield exactly `rung` full blocks + a max device head;
          * install/teardown boundaries vs the capture region."""
        from contextlib import contextmanager

        from sglang.srt.managers.kv_session_offload import make_sentinels

        @contextmanager
        def _ctx():
            B = self._sess_block_size
            S = self._sess_S
            # Reserved capture row: the last req_pool slot (idle during warmup).
            rpi = int(self._sess_req_pool.req_to_token.shape[0]) - 1
            # Worst case for the rung, CAPPED at the region capacity so the
            # per-rank owned tail never exceeds the region (else the
            # region-high-water assert in _sess_prepare_step trips, and the
            # host rows would run past the pool). num_blocks may then come out
            # below `rung` -- fine: the capture uses _sess_graph_capture_blocks
            # (= rung) regardless, and the surplus blocks capture empty.
            L = min(rung * B, self._sess_region_tokens)
            self._sess_open_slot(rpi, region_base=0)
            self._sess_capture_active = True
            self._sess_capture_rpi = rpi
            saved = self._sess_req_pool.req_to_token[rpi, :L].clone()
            # DECOUPLE S4: route the CAPTURED spill-graph collectives to comm B
            # so the graph tick runs on comm B at replay (a CUDA graph bakes its
            # NCCL comm at capture time). No-op unless decoupling is on
            # (_DCP_SPILL is None). Rank-uniform: every rank captures the same
            # rung with the flag set.
            from sglang.srt.distributed.parallel_state import set_dcp_spill_active

            set_dcp_spill_active(True)
            try:
                # Whole-suffix synthetic spill: positions [0, L) are host tail
                # (boundary 0) so counts -> rung full blocks. Residues p % S.
                res = torch.arange(L, device=saved.device, dtype=torch.int64) % S
                sent = make_sentinels(self._sess_host_base, S, res, start=0)
                self._sess_req_pool.req_to_token[rpi, :L] = sent.to(torch.int32)
                yield
            finally:
                self._sess_req_pool.req_to_token[rpi, :L] = saved
                self._sess_capture_active = False
                self._sess_capture_rpi = None
                self._sess_close_slot(rpi)
                set_dcp_spill_active(False)

        return _ctx()

    def _sess_graph_captured(self, rung) -> bool:
        """Whether a spill-tick graph for ``rung`` has been captured. The
        per-rung capture pass (GPU-wired by the messagent) records each rung it
        captures into ``_sess_graph_captured_rungs``; until then this is empty
        and can_run_graph keeps the spill tick eager (safe: never replay a
        graph that was not recorded)."""
        return rung is not None and rung in getattr(
            self, "_sess_graph_captured_rungs", set()
        )

    def _sess_graph_selftest(self, model_runner):
        """ENV-gated (KVSO_GRAPH_SELFTEST) per-rung graph==eager numeric check.
        Default OFF -> untouched. Under real load only rung 1 is reachable
        (partial spill pins host_tail to ~1 block), so this injects a synthetic
        whole-suffix host tail landing on EACH ladder rung and compares, WITHIN
        ONE boot on IDENTICAL input, the fixed-count graph body vs the eager
        block loop (the same q, same host rows, same merge order). Machine-zero
        expected (like weightless graph==eager). CUDA-graph REPLAY is bit-exact
        to the captured body by construction, so body==eager => replay==eager.
        Rank-uniform: every DCP rank runs the identical rung sequence; the block
        decode is intra-rank (no collective) so there is no desync risk. Also
        checks the over-ladder eager fallback via _sess_graph_can_replay."""
        import os
        from types import SimpleNamespace

        from sglang.srt.managers.kv_session_offload import make_sentinels

        if not os.environ.get("KVSO_GRAPH_SELFTEST") or not self._sess_graph_enabled:
            return
        rank = getattr(model_runner, "tp_rank", 0)
        layers = [
            m
            for m in model_runner.model.modules()
            if type(m).__name__ == "RadixAttention"
        ]
        layers.sort(key=lambda a: a.layer_id)
        layers = [ly for ly in layers if self._is_full_attention_layer(ly)] or layers
        if not layers:
            logger.warning("kvso graph selftest: no attention layer found")
            return
        layer = layers[0]
        B = self._sess_block_size
        S = self._sess_S
        dev = self._sess_staging_k.device
        rpi = int(self._sess_req_pool.req_to_token.shape[0]) - 1
        iu = self.indices_updater_decode
        num_qo = sum(self.dcp_q_head_counts) if self.uneven_dcp else iu.num_qo_heads
        head_dim = iu.head_dim
        ladder = self._sess_graph_ladder
        torch.manual_seed(20260723)
        # q is in the QUERY dtype (bf16/fp16); randn has no fp8 kernel, so
        # sample in fp32 and cast to the wrapper's q_data_type.
        qdt = getattr(iu, "q_data_type", torch.bfloat16)
        if not isinstance(qdt, torch.dtype):
            qdt = torch.bfloat16
        q = torch.randn(1, num_qo, head_dim, device=dev).to(qdt)

        def make_fb(L):
            return SimpleNamespace(
                # The selftest injects a PLAIN decode spill tick; _sess_prepare_
                # step reads is_decode()/is_target_verify() (the C4 twin gate).
                forward_mode=SimpleNamespace(
                    is_decode=lambda: True, is_target_verify=lambda: False
                ),
                batch_size=1,
                seq_lens_cpu=torch.tensor([L], dtype=torch.int64),
                req_pool_indices=torch.tensor([rpi], dtype=torch.int64, device=dev),
                kv_session_spill_tick=True,
            )

        rows = []
        saved_full = None
        # A small REAL device head so both paths attend a head in the SAME
        # merge order (the real-load shape; boundary>0). A whole-suffix
        # (boundary==0) synthetic would make the graph run an EMPTY head
        # wrapper the eager loop omits -> a merge-ORDER difference that is
        # fp-order-sensitive (decode-class), not a graph bug -- avoid it here.
        head = min(B, 256)
        head_slots = torch.arange(1, head + 1, device=dev, dtype=torch.int64)
        try:
            for R in [r for r in ladder if r >= 2]:
                tail = min(R * B, self._sess_region_tokens)
                L = head + tail
                if saved_full is None:
                    saved_full = self._sess_req_pool.req_to_token[rpi, :L].clone()
                self._sess_open_slot(rpi, region_base=0)
                # row = [real head slots] ++ [tail sentinels for [head, L)]
                tail_res = torch.arange(head, L, device=dev, dtype=torch.int64) % S
                tail_sent = make_sentinels(
                    self._sess_host_base, S, tail_res, start=head
                )
                self._sess_req_pool.req_to_token[rpi, :L] = torch.cat(
                    [head_slots, tail_sent]
                ).to(torch.int32)
                fb = make_fb(L)
                # eager
                self._sess_graph_capture_blocks = None
                self._sess_graph_replay_blocks = None
                self._sess_spill = self._sess_prepare_step(fb)
                # HARNESS sync (not a graph concern): the eager path reads the
                # CURRENT token from the scratch slot (its D2D fixup), the graph
                # reads it from the host row. Under real load the owner-write
                # writes the same value to both; here no owner-write runs, so
                # copy host[cur_host_row] -> scratch for this layer so both read
                # identical current-token bytes (else only the owning rank shows
                # a ~1/owned_tail diff -- a harness gap, not a graph bug).
                if self._sess_spill.cur_owned:
                    from sgl_kernel.kvcacheio import transfer_kv_per_layer

                    _fl = self._sess_full_layer_idx(layer)
                    _chr = torch.tensor(
                        [self._sess_spill.cur_host_row], dtype=torch.int64, device=dev
                    )
                    transfer_kv_per_layer(
                        src_k=self._sess_host_pool.k_data_refs[_fl],
                        dst_k=self._sess_full_pool.k_buffer[_fl],
                        src_v=self._sess_host_pool.v_data_refs[_fl],
                        dst_v=self._sess_full_pool.v_buffer[_fl],
                        src_indices=_chr,
                        dst_indices=self._sess_scratch_loc,
                        item_size=self._sess_host_pool.token_stride_size,
                    )
                o_e, lse_e = self._sess_blockwise_decode_return_lse(q, layer)
                o_e = o_e.clone()
                # graph body on the SAME st (refill fixed buffers, route dispatch)
                self._sess_graph_replay_blocks = self._sess_spill.graph_rung
                self._sess_graph_prepare_blocks(in_capture=False)
                self._sess_graph_capture_blocks = self._sess_spill.graph_rung
                o_g, lse_g = self._sess_blockwise_decode_return_lse(q, layer)
                self._sess_graph_capture_blocks = None
                self._sess_graph_replay_blocks = None
                md = float((o_e - o_g).abs().max().item())
                nan = bool(torch.isnan(o_g).any() or torch.isnan(o_e).any())
                verdict = (
                    "MACHINE_ZERO"
                    if md == 0.0
                    else ("NAN" if nan else f"maxd={md:.3e}")
                )
                rows.append(
                    (self._sess_spill.graph_rung, self._sess_spill.n_own, verdict)
                )
                self._sess_close_slot(rpi)
        finally:
            if saved_full is not None:
                n = saved_full.numel()
                self._sess_req_pool.req_to_token[rpi, :n] = saved_full
            self._sess_graph_capture_blocks = None
            self._sess_graph_replay_blocks = None
            self._sess_spill = None
        logger.info(
            "==== kvso spill-graph SELFTEST (rank %d) graph==eager, head=%d, "
            "per PICKED rung (over-ladder fallback is pure-unit-tested; "
            "region cap makes it live-unreachable) ====",
            rank,
            head,
        )
        # dedup by picked rung (multiple synthetic tails can map to one rung)
        seen = {}
        for rung, n_own, verdict in rows:
            seen.setdefault(rung, (n_own, verdict))
        for rung in sorted(seen):
            n_own, verdict = seen[rung]
            logger.info("  picked_rung=%-3d owned_tail=%-6d: %s", rung, n_own, verdict)
        logger.info("==== kvso spill-graph SELFTEST done (rank %d) ====", rank)

    def _is_full_attention_layer(self, layer) -> bool:
        """Best-effort: a layer whose KV lives in the full-attention pool."""
        try:
            self._sess_full_layer_idx(layer)
            return True
        except Exception:
            return False

    def _sess_graph_log_fallback(self, reason: str) -> bool:
        if not self._sess_graph_fallback_logged:
            self._sess_graph_fallback_logged = True
            logger.info(
                "kv-session-offload spill-graph: eager fallback (%s). Logged "
                "once; later fallbacks silent.",
                reason,
            )
        return False

    def _sess_graph_can_replay(self, forward_batch) -> bool:
        """Replay admission for the spill-tick graph (analog of
        wl_graph_can_replay). Sets `_sess_graph_replay_blocks` on success.

        Rank-uniform: the rung is picked from `num_blocks_rank_uniform` over
        replicated counts, so every rank takes the same graph/eager decision
        -- the per-layer DCP collective count is preserved either way (the
        eager spill tick issues the identical cp_lse sequence)."""
        from sglang.srt.managers.kv_session_offload import (
            num_blocks_rank_uniform,
            owned_counts_even,
            owned_counts_weighted,
            spill_graph_pick_rung,
        )

        self._sess_graph_replay_blocks = None
        if not (self._sess_graph_enabled and self._sess_graph_ladder):
            return False
        if not getattr(forward_batch, "kv_session_spill_tick", False):
            return False
        if forward_batch.batch_size != 1:
            return self._sess_graph_log_fallback(
                f"spill-graph is bs=1, got bs={forward_batch.batch_size}"
            )
        # Admission runs BEFORE _sess_prepare_step (init_forward_metadata) --
        # so derive the RANK-UNIFORM host block count directly from the
        # replicated row + seq len HERE, exactly as wl_graph_can_replay does
        # (reading a not-yet-built st.graph_rung is the bug that forced eager
        # fallback on every tick). The tail host rows are dense [base,
        # base+n_own), so no contiguity check is needed (unlike _wl).
        L = int(forward_batch.seq_lens_cpu[0].item())
        row = self._sess_req_pool.req_to_token[forward_batch.req_pool_indices[0], :L]
        boundary = int((row < self._sess_host_base).sum().item())
        dcp = len(self._sess_prefix) - 1
        if self._sess_mode == "weighted":
            residues = row[boundary:L].to(torch.int64) % self._sess_S
            counts = owned_counts_weighted(residues, self._sess_prefix)
        elif self._sess_mode == "even":
            full = owned_counts_even(L, dcp)
            headc = owned_counts_even(boundary, dcp)
            counts = [full[r] - headc[r] for r in range(dcp)]
        else:
            counts = [L - boundary]
        num_blocks = num_blocks_rank_uniform(counts, self._sess_block_size)
        rung = spill_graph_pick_rung(num_blocks, self._sess_graph_ladder)
        if rung is None:
            return self._sess_graph_log_fallback(
                f"tick needs {num_blocks} blocks > max rung "
                f"{self._sess_graph_ladder[-1]}"
            )
        self._sess_graph_replay_blocks = rung
        return True

    def _sess_graph_bucket_state(self):
        """Persistent bs=1 bucket: max_blocks cuda-graph block wrappers (shared
        by every rung, rung R uses wrappers[0..R-1]) + a dev-head graph wrapper
        + fixed indptr/indices/staging-map buffers. Mirrors
        _wl_graph_bucket_state; single staging region (no #136b prefetch)."""
        import types

        B = self._sess_block_size
        max_blocks = max(self._sess_graph_ladder)
        dev = self.kv_last_page_len.device
        iu = self.indices_updater_decode
        # Dev-head capture-at-max: the largest head this rank can attend. Cap
        # at the region (a full max-context session) -- GPU-tunable.
        head_max = int(self._sess_region_tokens)
        st = types.SimpleNamespace(
            wrappers=[],
            indptr_dev=[],
            indptr_host=[],
            stage_ids=[],  # host source rows per block (refilled at replay)
            stage_dst=[],  # staging destination slots (fixed)
            head_wrapper=None,
            head_indices=torch.zeros(head_max, dtype=torch.int32, device=dev),
            head_indptr_dev=torch.zeros(2, dtype=torch.int32, device=dev),
            head_indptr_host=torch.zeros(2, dtype=torch.int32),
            ow_dst=torch.zeros(1, dtype=torch.int64, device=dev),  # owner-write dst
        )
        for j in range(max_blocks):
            indptr_buf = torch.zeros(2, dtype=torch.int32, device=dev)
            indices_buf = torch.zeros(B, dtype=torch.int32, device=dev)
            w = BatchDecodeWithPagedKVCacheWrapper(
                self._sess_workspace_buffer,
                "NHD",
                backend=self.decode_backend,
                use_cuda_graph=True,
                use_tensor_cores=self.decode_use_tensor_cores,
                paged_kv_indptr_buffer=indptr_buf,
                paged_kv_indices_buffer=indices_buf,
                paged_kv_last_page_len_buffer=self.kv_last_page_len[:1],
            )
            st.wrappers.append(w)
            st.indptr_dev.append(indptr_buf)
            st.indptr_host.append(torch.zeros(2, dtype=torch.int32))
            # Fixed staging destination for block j: staging slots [0, B). The
            # single region is reused serially across blocks (copy j after run
            # j-1 on the main stream, captured order).
            st.stage_ids.append(torch.zeros(B, dtype=torch.int64, device=dev))
            st.stage_dst.append(self._sess_stage_arange64[:B])
        # Dev-head graph wrapper (real device pool, indices = dev_head_idx).
        st.head_wrapper = BatchDecodeWithPagedKVCacheWrapper(
            self._sess_workspace_buffer,
            "NHD",
            backend=self.decode_backend,
            use_cuda_graph=True,
            use_tensor_cores=self.decode_use_tensor_cores,
            paged_kv_indptr_buffer=st.head_indptr_dev,
            paged_kv_indices_buffer=st.head_indices,
            paged_kv_last_page_len_buffer=self.kv_last_page_len[:1],
        )
        logger.info(
            "kv-session-offload spill-graph bucket built: %d block wrappers "
            "(ladder %s), block=%d, head_max=%d",
            max_blocks,
            self._sess_graph_ladder,
            B,
            head_max,
        )
        return st

    def _sess_graph_prepare_blocks(self, in_capture: bool):
        """OUT-of-graph per-step prep (analog of _wl_graph_prepare_blocks): plan
        the rung's block wrappers + the dev-head wrapper and fill the fixed
        staging/index/indptr buffers. Runs ONCE per spill tick (the #136a plan
        hoist). Capture plans the worst case (every block FULL, head at max);
        replay refills real counts via fast_decode_plan (capture-at-max /
        replay-below)."""
        iu = self.indices_updater_decode
        num_qo_heads = (
            sum(self.dcp_q_head_counts)
            if self.uneven_dcp
            else (self.indices_updater_decode.num_qo_heads)
        )
        num_kv_heads = iu.num_kv_heads
        head_dim = iu.head_dim
        last_page = self.kv_last_page_len[:1]
        B = self._sess_block_size
        if self._sess_graph_bucket is None:
            self._sess_graph_bucket = self._sess_graph_bucket_state()
        st = self._sess_graph_bucket

        if in_capture:
            rung = self._sess_graph_capture_blocks
            assert rung is not None, "spill-graph capture without a rung set"
            synth_indptr = torch.tensor(
                [0, B], dtype=torch.int32, device=last_page.device
            )
            synth_indices = torch.zeros(B, dtype=torch.int32, device=last_page.device)
            for j in range(rung):
                w = st.wrappers[j]
                w.plan(
                    synth_indptr,
                    synth_indices,
                    last_page,
                    num_qo_heads,
                    num_kv_heads,
                    head_dim,
                    1,
                    q_data_type=iu.q_data_type,
                    kv_data_type=iu.data_type,
                )
                w.begin_forward = partial(fast_decode_plan, w)
            # Dev-head captured at max.
            hmax = int(st.head_indices.numel())
            head_synth = torch.tensor(
                [0, hmax], dtype=torch.int32, device=last_page.device
            )
            st.head_wrapper.plan(
                head_synth,
                st.head_indices,
                last_page,
                num_qo_heads,
                num_kv_heads,
                head_dim,
                1,
                q_data_type=iu.q_data_type,
                kv_data_type=iu.data_type,
            )
            st.head_wrapper.begin_forward = partial(fast_decode_plan, st.head_wrapper)
            return

        # ---- replay prep: refill from _sess_spill's out-of-graph plan ----
        spill = self._sess_spill
        rung = self._sess_graph_replay_blocks
        assert (
            rung is not None and spill is not None
        ), "spill-graph replay prep without admission (can_replay first)"
        plan = spill.graph_plan  # list of {cnt, host_rows, indptr} (per block)
        for j in range(rung):
            w = st.wrappers[j]
            blk = plan[j] if j < len(plan) else {"cnt": 0}
            cnt = int(blk["cnt"])
            ic = st.indptr_host[j]
            ic[1] = cnt
            if cnt > 0:
                st.stage_ids[j][:cnt].copy_(blk["host_rows"], non_blocking=True)
                # block indices = the staging slots [0, cnt)
                w._paged_kv_indices_buf[:cnt].copy_(
                    self._sess_stage_arange32[:cnt], non_blocking=True
                )
            st.indptr_dev[j].copy_(ic, non_blocking=True)
            w.begin_forward(
                st.indptr_dev[j],
                w._paged_kv_indices_buf[:cnt],
                last_page,
                num_qo_heads,
                num_kv_heads,
                head_dim,
                1,
                data_type=iu.data_type,
                q_data_type=iu.q_data_type,
                non_blocking=True,
                fixed_split_size=None,
                disable_split_kv=self.disable_cuda_graph_kv_split,
                global_override_indptr_cpu=ic,
            )
        # Dev-head refill (replay-below the captured max).
        n_head = int(spill.n_head_own)
        st.head_indptr_host[1] = n_head
        if n_head > 0:
            st.head_indices[:n_head].copy_(spill.dev_head_idx, non_blocking=True)
        st.head_indptr_dev.copy_(st.head_indptr_host, non_blocking=True)
        st.head_wrapper.begin_forward(
            st.head_indptr_dev,
            st.head_indices[:n_head],
            last_page,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            1,
            data_type=iu.data_type,
            q_data_type=iu.q_data_type,
            non_blocking=True,
            fixed_split_size=None,
            disable_split_kv=self.disable_cuda_graph_kv_split,
            global_override_indptr_cpu=st.head_indptr_host,
        )
        # Owner-write destination (data-driven, not a branch): the current
        # token's host row if this rank owns it, else a never-read DUMP slot in
        # this session's region (sized with >=2 slack rows above max context,
        # so the last region row is always spare). Resolved here out-of-graph.
        dump = int(spill.region_base) + self._sess_region_tokens - 1
        st.ow_dst[0] = spill.cur_host_row if spill.cur_owned else dump

    def _sess_blockwise_decode_return_lse_graph(self, q_full, layer):
        """Graph-recordable fixed-count spill-tick block decode (PORT of
        _wl_blockwise_decode_return_lse_graph). Iterates EXACTLY the capture
        rung's blocks; per block {H2D gather (single region, main stream) +
        pre-planned wrapper run + merge}. The device HEAD is attended first via
        the persistent head wrapper. Empty blocks (zero-length indptr at
        replay) are sanitized to the (o=0, lse=-inf) contract in-graph so the
        online merge folds them as identity -- matching the eager loop's skip.
        NO .plan()/.item()/host sync inside (all hoisted to prepare_blocks)."""
        from sgl_kernel.kvcacheio import transfer_kv_per_layer

        st = self._sess_graph_bucket
        rung = self._sess_graph_capture_blocks
        spill = self._sess_spill
        fl = self._sess_full_layer_idx(layer)
        kv = self._sess_staging_kv
        host = self._sess_host_pool
        B = self._sess_block_size
        real_kv = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)
        neg_inf = float("-inf")
        # The current token's KV was already owner-written to host (real row or
        # dump slot) by _sess_owner_write's graph branch, run on the compute
        # stream before this block decode -- same ordering as the eager path.

        o_acc = lse_acc = empty_acc = None
        # Device head partial (real pool) -- first merge source.
        oh, lh = st.head_wrapper.forward_return_lse(
            q_full,
            real_kv,
            sm_scale=layer.scaling,
            logits_soft_cap=layer.logit_cap,
            k_scale=layer.k_scale_float,
            v_scale=layer.v_scale_float,
        )
        hlen = st.head_indptr_dev[1:] - st.head_indptr_dev[:-1]
        hem = hlen == 0
        lh = torch.where(hem.unsqueeze(1), torch.full_like(lh, neg_inf), lh)
        oh = torch.where(hem.view(-1, 1, 1), torch.zeros_like(oh), oh)
        o_acc, lse_acc, empty_acc = oh, lh, hem

        # 3. Host tail blocks (fixed count == rung).
        for j in range(rung):
            # BUGFIX: stage host rows into the separate STAGING buffer (the
            # wrapper reads _sess_staging_kv), NOT the real device KV pool.
            # Mirror the eager _sess_issue_copy: transfer_kv_per_layer host ->
            # _sess_staging_k/v. A FIXED B-row copy (stage_ids[j] has B valid
            # host rows: the first cnt real, the rest a valid pad row 0), so
            # the captured shape is constant; the wrapper attends only [0, cnt)
            # via its replay-below indptr. Using load_to_device_per_layer into
            # _sess_full_pool wrote device indices [0, B) into the 3600-slot
            # pool -> illegal access (df31391708).
            transfer_kv_per_layer(
                src_k=host.k_data_refs[fl],
                dst_k=self._sess_staging_k,
                src_v=host.v_data_refs[fl],
                dst_v=self._sess_staging_v,
                src_indices=st.stage_ids[j],
                dst_indices=self._sess_stage_arange64[:B],
                item_size=host.token_stride_size,
            )
            o_b, lse_b = st.wrappers[j].forward_return_lse(
                q_full,
                kv,
                sm_scale=layer.scaling,
                logits_soft_cap=layer.logit_cap,
                k_scale=layer.k_scale_float,
                v_scale=layer.v_scale_float,
            )
            lens = st.indptr_dev[j][1:] - st.indptr_dev[j][:-1]
            em = lens == 0
            lse_b = torch.where(em.unsqueeze(1), torch.full_like(lse_b, neg_inf), lse_b)
            o_b = torch.where(em.view(-1, 1, 1), torch.zeros_like(o_b), o_b)
            o_acc, lse_acc = _safe_merge_state(o_acc, lse_acc, o_b, lse_b)
            empty_acc = empty_acc & em
            lse_acc = torch.where(
                empty_acc.unsqueeze(1), torch.full_like(lse_acc, neg_inf), lse_acc
            )
            o_acc = torch.where(
                empty_acc.view(-1, 1, 1), torch.zeros_like(o_acc), o_acc
            )
        return o_acc, lse_acc

    def _sess_owner_write_graph(self, layer, k_full, v_full):
        """Graph-recordable owner-write (fixed shapes; PORT of
        _wl_spill_owner_write_graph's fixed formulation). Quantize the current
        token's K/V through the device scratch row (stock set_kv_buffer byte
        path, 1 row), then D2H-copy the scratch row to st.ow_dst -- the real
        host row when this rank owns the token, else the region dump slot
        (both resolved out-of-graph in _sess_graph_prepare_blocks). No
        .nonzero()/.any()/branch: when cur_owned is false the copy still runs
        but targets the never-read dump slot, byte-identical to the eager skip
        for every real host row."""
        from sgl_kernel.kvcacheio import transfer_kv_per_layer

        st = self._sess_graph_bucket
        self._sess_pool.set_kv_buffer(
            layer,
            self._sess_scratch_loc,
            k_full.clone(),
            v_full.clone(),
            layer.k_scale,
            layer.v_scale,
        )
        fl = self._sess_full_layer_idx(layer)
        host = self._sess_host_pool
        transfer_kv_per_layer(
            src_k=self._sess_full_pool.k_buffer[fl],
            dst_k=host.k_data_refs[fl],
            src_v=self._sess_full_pool.v_buffer[fl],
            dst_v=host.v_data_refs[fl],
            src_indices=self._sess_scratch_loc,
            dst_indices=st.ow_dst,
            item_size=host.token_stride_size,
        )

    # ------------------------------------------------------------------
    # #136a: CUDA-graph integration of the streaming block-decode.
    #
    # The eager B0/B1 block loop (`_wl_blockwise_decode_return_lse`) plans a
    # flashinfer wrapper PER BLOCK PER LAYER on the host -- unrecordable. The
    # graph path restructures this into the de-risked mechanism
    # (b2_graph_derisk.py, byte-identical max|d|=0):
    #   * a POOL of persistent cuda-graph-mode block wrappers per bs bucket
    #     (fixed indptr/indices buffers), shared by every rung,
    #   * all `.plan()` + index/staging-map construction hoisted OUT of the
    #     graph into `init_forward_metadata_out_graph` (once per step, instead
    #     of per layer x block on the eager path),
    #   * a captured FIXED-count block loop per rung: {H2D staging copy from
    #     live pinned host bytes + wrapper.run + merge_state} x R, no
    #     `.plan()`, no host sync inside,
    #   * a bucketed capture LADDER over the block count R = ceil(seq /
    #     (B * dcp_size)) (graphs cannot branch on the block count); replay
    #     picks the smallest captured rung covering the current seq len and
    #     plans the trailing blocks EMPTY (sanitized to the o=0/lse=-inf
    #     empty-attention contract, which merge_state folds in as identity).
    # Head + workers capture/replay SYMMETRICALLY (#133): both derive the
    # rung from the shared global seq len, so the per-layer DCP collectives
    # (4/layer) pair up by construction. Above the largest rung (or on any
    # admission miss) BOTH ranks fall back to the eager block loop, which is
    # guard-free for decode under graphs-enabled per the #133 rule.
    # ------------------------------------------------------------------

    def wl_block_graph_supported(self) -> bool:
        return bool(
            getattr(self, "weightless_kv", False)
            and getattr(self, "_wl_chunk_block_size", 0)
        )

    def _wl_blocks_needed(self, max_seq_len: int) -> int:
        per_block_global = self._wl_chunk_block_size * self.dcp_size
        return max(1, -(-int(max_seq_len) // per_block_global))

    def wl_build_graph_ladder(self) -> list:
        """Ladder of block-count rungs covering the model's full context
        length: dense (step 1) up to 8 blocks, then ~x1.5 geometric to R_max.
        Replay picks the smallest covering rung; every block beyond the
        current seq len is a captured no-op that still pays its fixed H2D
        copy + kernel launches, so a dense low range keeps the common rungs
        EXACT (measured: one wasted trailing block region can halve the
        streaming-decode rate). Capture cost is ~1 s/rung. Called by the
        decode graph runner before capture; stored so replay admission
        (wl_graph_can_replay) and the capture loop agree on the rungs."""
        r_max = self._wl_blocks_needed(self.max_context_len)
        ladder = set()
        r = 1
        while r < r_max:
            ladder.add(r)
            r = r + 1 if r < 8 else max(r + 1, int(r * 1.5))
        ladder.add(r_max)
        self._wl_graph_ladder = sorted(ladder)
        return self._wl_graph_ladder

    def _wl_graph_log_fallback(self, reason: str) -> bool:
        if not self._wl_graph_fallback_logged:
            self._wl_graph_fallback_logged = True
            logger.info(
                "Weightless-KV block-decode graph: falling back to the eager "
                "block loop (%s). Logged once; later fallbacks are silent.",
                reason,
            )
        return False

    def wl_graph_can_replay(self, forward_batch) -> bool:
        """Replay admission for the streaming block-decode graphs. Sets
        `_wl_graph_replay_blocks` (the chosen rung) on success.

        Rank-uniform by construction: every input (seq lens, batch size,
        req_to_token content, ladder, spill config) is identical on the head
        and every weightless worker, so all ranks take the same graph/eager
        decision -- the lockstep collective count is preserved either way
        (eager decode under graphs-enabled is also guard-free, #133 rule)."""
        self._wl_graph_replay_blocks = None
        if not self._wl_graph_ladder:
            return False
        if not forward_batch.forward_mode.is_decode():
            return False
        seq_lens_cpu = forward_batch.seq_lens_cpu
        if seq_lens_cpu is None:
            return self._wl_graph_log_fallback("no host seq lens")
        max_seq = int(seq_lens_cpu.max().item())
        if getattr(self, "_wl_spill_active", False):
            # The captured H2D staging copies have a FIXED per-block element
            # count, templated on the static tier map with graph blocks
            # anchored to SLOT WINDOWS (block j covers owned slot values
            # [jB, (j+1)B)). The template is exact whenever the request's loc
            # layout is OFFSET-LINEAR (loc(pos) = O + pos, i.e. one contiguous
            # allocation -- the canonical single over-VRAM request, including
            # radix-cached prefix reuse). Verify it; otherwise run eager
            # (correct, just slower).
            if forward_batch.batch_size != 1:
                return self._wl_graph_log_fallback(
                    f"spill-graph supports bs=1, got bs={forward_batch.batch_size}"
                )
            req = forward_batch.req_pool_indices[0]
            row = self.indices_updater_decode.req_to_token[req, :max_seq]
            loc_offset = int(row[0].item())
            ident = torch.arange(
                loc_offset,
                loc_offset + max_seq,
                device=row.device,
                dtype=row.dtype,
            )
            if not bool(torch.equal(row, ident)):
                return self._wl_graph_log_fallback(
                    "non-contiguous KV slot layout (allocator reuse/" "fragmentation)"
                )
            self._wl_graph_loc_offset = loc_offset
            # Rank-uniform rung covering every rank's LAST owned slot
            # ((O + seq - 1) // dcp is an upper bound across the +-1 skew).
            needed = (
                ((loc_offset + max_seq - 1) // self.dcp_size)
                // self._wl_chunk_block_size
            ) + 1
        else:
            needed = self._wl_blocks_needed(max_seq)
        rung = next((r for r in self._wl_graph_ladder if r >= needed), None)
        if rung is None:
            return self._wl_graph_log_fallback(
                f"seq len {max_seq} needs {needed} blocks > captured max "
                f"{self._wl_graph_ladder[-1]}"
            )
        self._wl_graph_replay_blocks = rung
        return True

    def _wl_graph_bucket_state(self, bs: int):
        """Create the per-bs-bucket persistent block-wrapper pool + fixed
        buffers, shared by every ladder rung (rung R uses wrappers[0..R-1])."""
        import types

        B = self._wl_chunk_block_size
        max_blocks = max(self._wl_graph_ladder)
        dev = self.kv_last_page_len.device
        spill = getattr(self, "_wl_spill_active", False)
        if spill:
            if bs != 1:
                raise ValueError(
                    "weightless-KV block-decode graph with host spill supports "
                    f"only the bs=1 capture bucket, got bs={bs} "
                    "(SGLANG_WL_GRAPH_MAX_BS must stay 1 with spill)."
                )
            # The graph-safe owner-write redirects NON-host rows' D2H copy to a
            # dump slot one past the logical host tier; require the physical
            # pool (sized in whole GB) to actually have that spare slot.
            if self._wl_host_pool.size <= self._wl_host_slots:
                raise ValueError(
                    "weightless-KV block-decode graph: host tier has no spare "
                    f"slot for the in-graph write dump (pool {self._wl_host_pool.size} "
                    f"== logical {self._wl_host_slots}). Lower "
                    "--weightless-kv-host-spill-tokens by 1."
                )
            self._wl_host_dump_slot = self._wl_host_slots
        # #136b H2D prefetch/double-buffer: with a 2-block staging carve,
        # block j uses staging region j % 2 (region bases B apart), so the
        # captured side-stream copy of block j+1 can run while attention
        # reads block j from the other region. Without the double carve both
        # "regions" alias the single #136a staging region and the copies stay
        # on the main stream (serial, byte-identical #136a behavior).
        prefetch = spill and bool(getattr(self, "_wl_prefetch", False))
        base0 = getattr(self, "_wl_stage_base", 0)
        region_bases = [base0, base0 + (B if prefetch else 0)]
        st = types.SimpleNamespace(
            wrappers=[],
            indptr_dev=[],
            indptr_host=[],
            stage_ids=[],
            stage_dst=[],
            stage_cnt=[],
            prefetch=prefetch,
            stage_slot_arange=[
                torch.arange(rb, rb + B, dtype=torch.int32, device=dev)
                for rb in region_bases
            ],
        )
        if prefetch:
            # Side copy stream + events, persistent for the bucket and
            # re-recorded per layer inside each rung's capture (event
            # record/wait during capture become intra-graph edges; the
            # objects are safely shared across layers and rung graphs).
            # fork_ev: recorded ONCE per captured body execution
            #   (init_forward_metadata_in_graph) -- the side stream's copy
            #   pipeline then runs AHEAD across layers (staging rows are
            #   per-layer, so layer L+1's copies never collide with layer L's)
            #   subject only to the per-layer edges below.
            # copy_ev[j]: side -> main, block j's staged bytes ready.
            # run_ev[j]:  main -> side, block j's attention (the last reader
            #   of staging region j % 2 in this layer) done; copy j+2 waits it.
            # ow_ev: main -> side, this layer's owner-write D2H done; only the
            #   LAST host block's copy waits it (its slot window contains the
            #   current token's host slot, written this step).
            st.copy_stream = torch.cuda.Stream()
            st.fork_ev = torch.cuda.Event()
            st.ow_ev = torch.cuda.Event()
            st.ow_recorded = False
            st.copy_ev = [torch.cuda.Event() for _ in range(max_blocks)]
            st.run_ev = [torch.cuda.Event() for _ in range(max_blocks)]
        for j in range(max_blocks):
            indptr_buf = torch.zeros(bs + 1, dtype=torch.int32, device=dev)
            indices_buf = torch.zeros(bs * B, dtype=torch.int32, device=dev)
            w = BatchDecodeWithPagedKVCacheWrapper(
                self.workspace_buffer,
                "NHD",
                backend=self.decode_backend,
                use_cuda_graph=True,
                use_tensor_cores=self.decode_use_tensor_cores,
                paged_kv_indptr_buffer=indptr_buf,
                paged_kv_indices_buffer=indices_buf,
                paged_kv_last_page_len_buffer=self.kv_last_page_len[:bs],
            )
            st.wrappers.append(w)
            st.indptr_dev.append(indptr_buf)
            st.indptr_host.append(torch.zeros(bs + 1, dtype=torch.int32))
            if spill:
                # Static per-block host-copy template under the linear slot
                # model: block j covers owned slots [jB, (j+1)B); its host
                # part is the tail above _wl_dev_slots. The captured copy has
                # this FIXED count; replay fills the first n_host entries with
                # real host ids (shortfall keeps stale-but-valid ids whose dst
                # staging slots are never referenced by the block's indices).
                cnt = min(B, max(0, (j + 1) * B - self._wl_dev_slots))
            else:
                cnt = 0
            st.stage_cnt.append(cnt)
            if cnt > 0:
                st.stage_ids.append(torch.zeros(cnt, dtype=torch.int64, device=dev))
                st.stage_dst.append(
                    torch.arange(
                        region_bases[j % 2],
                        region_bases[j % 2] + cnt,
                        dtype=torch.int64,
                        device=dev,
                    )
                )
            else:
                st.stage_ids.append(None)
                st.stage_dst.append(None)
        logger.info(
            "Weightless-KV block-decode graph: built bs=%d bucket state -- "
            "%d block wrappers (ladder %s), block size %d, spill=%s, "
            "h2d_prefetch=%s.",
            bs,
            max_blocks,
            self._wl_graph_ladder,
            B,
            spill,
            prefetch,
        )
        return st

    def _wl_graph_prepare_blocks(self, bs: int, in_capture: bool):
        """OUT-of-graph per-step prep for the captured block loop: build every
        block's kv indices + staging map into the FIXED buffers and (fast-)plan
        the block wrappers. This is the hoisted `.plan()` the de-risk proved
        graph-compatible -- it runs ONCE per decode step (vs per layer x block
        on the eager path)."""
        iu = self.indices_updater_decode
        num_qo_heads = sum(self.dcp_q_head_counts)
        num_kv_heads = iu.num_kv_heads
        head_dim = iu.head_dim
        last_page = self.kv_last_page_len[:bs]
        B = self._wl_chunk_block_size

        st = self._wl_graph_state.get(bs)
        if in_capture:
            rung = self._wl_graph_capture_blocks
            assert rung is not None, (
                "weightless block-decode capture without a rung set -- the "
                "decode graph runner must set _wl_graph_capture_blocks"
            )
            if st is None:
                st = self._wl_graph_bucket_state(bs)
                self._wl_graph_state[bs] = st
            self._wl_graph_active_bucket = bs
            # Synthetic worst-case plan (every block FULL: B slots/request) so
            # the frozen launch config covers any replay content; replay
            # re-plans shorter/empty blocks via fast_decode_plan (the same
            # capture-at-max / replay-below contract as the monolithic decode
            # graph wrapper). Indices point at slot 0 -- values are irrelevant
            # at capture, only shapes/launch configs are recorded.
            synth_indptr = torch.arange(
                0, bs * B + 1, B, dtype=torch.int32, device=last_page.device
            )
            synth_indices = torch.zeros(
                bs * B, dtype=torch.int32, device=last_page.device
            )
            for j in range(rung):
                w = st.wrappers[j]
                w.plan(
                    synth_indptr,
                    synth_indices,
                    last_page,
                    num_qo_heads,
                    num_kv_heads,
                    head_dim,
                    1,  # page_size
                    q_data_type=iu.q_data_type,
                    kv_data_type=iu.data_type,
                )
                w.begin_forward = partial(fast_decode_plan, w)
            return

        rung = self._wl_graph_replay_blocks
        assert rung is not None and st is not None, (
            "weightless block-decode graph replay prep without admission "
            "(wl_graph_can_replay must run first)"
        )
        self._wl_graph_active_bucket = bs
        indptr_host = getattr(self, "_dcp_decode_owned_kv_indptr_host", None)
        assert indptr_host is not None, (
            "weightless block-decode graph replay: missing host owned-indptr "
            "(monolithic decode wrapper not in fast_decode_plan mode?)"
        )
        kv_indices = self._dcp_decode_owned_kv_indices
        spill = getattr(self, "_wl_spill_active", False)
        dev_limit = self._wl_dev_slots if spill else 0
        if spill:
            # bs == 1 + offset-linear layout (verified by wl_graph_can_replay):
            # this rank's owned slots are CONTIGUOUS [s0, s0 + owned).
            owned = int(indptr_host[1])
            s0 = (self._wl_graph_loc_offset + self.dcp_rank) // self.dcp_size
            if s0 + owned > dev_limit + self._wl_host_slots:
                raise RuntimeError(
                    "weightless-KV block-decode graph: owned shard "
                    f"([{s0}, {s0 + owned}) slots) exceeds device+host "
                    f"capacity ({dev_limit}+{self._wl_host_slots}) on "
                    f"dcp_rank {self.dcp_rank}; refusing out-of-range host "
                    "access."
                )
        for j in range(rung):
            w = st.wrappers[j]
            indptr_cpu = st.indptr_host[j]
            total = 0
            if spill:
                # SLOT-WINDOW anchored block: block j covers owned slot VALUES
                # [jB, (j+1)B) -- the host/device split per block is then the
                # exact static template the captured fixed-count H2D copies
                # were built from, for ANY allocation offset. Leading blocks
                # below s0 (radix-cached predecessors) plan empty.
                lo = max(j * B, s0)
                hi = min((j + 1) * B, s0 + owned)
                ln = hi - lo if hi > lo else 0
                if ln > 0:
                    ls = lo - s0
                    w._paged_kv_indices_buf[:ln].copy_(
                        kv_indices[ls : ls + ln], non_blocking=True
                    )
                    # Host part = the block tail above dev_limit (slots
                    # ascend). Always <= the captured template count by
                    # construction of the slot window.
                    n_host = min(ln, max(0, hi - max(lo, dev_limit)))
                    assert n_host <= st.stage_cnt[j], (
                        j,
                        n_host,
                        st.stage_cnt[j],
                    )
                    if n_host > 0:
                        hs = max(lo, dev_limit) - dev_limit
                        st.stage_ids[j][:n_host].copy_(
                            torch.arange(
                                hs,
                                hs + n_host,
                                dtype=torch.int64,
                                device=st.stage_ids[j].device,
                            ),
                            non_blocking=True,
                        )
                        # Rewrite the staged tail of the block's indices to
                        # the staging slots of block j's region (order
                        # preserved; region j % 2 under #136b prefetch, the
                        # single region otherwise).
                        w._paged_kv_indices_buf[ln - n_host : ln].copy_(
                            st.stage_slot_arange[j % 2][:n_host],
                            non_blocking=True,
                        )
                    total = ln
                indptr_cpu[1] = total
            else:
                # All-resident (B0): owned-LIST-position blocks, identical to
                # the eager loop's partition (arbitrary slot values, bs >= 1).
                for i in range(bs):
                    s = int(indptr_host[i])
                    e = int(indptr_host[i + 1])
                    bstart = s + j * B
                    bend = min(s + (j + 1) * B, e)
                    ln = bend - bstart if bend > bstart else 0
                    if ln > 0:
                        w._paged_kv_indices_buf[total : total + ln].copy_(
                            kv_indices[bstart:bend], non_blocking=True
                        )
                        total += ln
                    indptr_cpu[i + 1] = total
            st.indptr_dev[j].copy_(indptr_cpu, non_blocking=True)
            w.begin_forward(
                st.indptr_dev[j],
                w._paged_kv_indices_buf[:total],
                last_page,
                num_qo_heads,
                num_kv_heads,
                head_dim,
                1,
                data_type=iu.data_type,
                q_data_type=iu.q_data_type,
                non_blocking=True,
                fixed_split_size=None,
                disable_split_kv=self.disable_cuda_graph_kv_split,
                global_override_indptr_cpu=indptr_cpu,
            )

    def _wl_blockwise_decode_return_lse_graph(self, q_full, layer):
        """Graph-recordable fixed-count variant of
        `_wl_blockwise_decode_return_lse`: iterate EXACTLY the capture rung's
        block count; per block {H2D staging copy (live pinned-host read) +
        pre-planned wrapper run + merge}. NO `.plan()`, NO `.item()`/host sync
        inside. Empty blocks (planned with zero-length indptr at replay) are
        sanitized to the (o=0, lse=-inf) empty-attention contract IN-graph from
        the live device indptr, so merge_state folds them in as identity --
        matching the eager loop's `continue` semantics exactly."""
        bs = self._wl_graph_active_bucket
        rung = self._wl_graph_capture_blocks
        st = self._wl_graph_state[bs]
        kv_buffer = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)
        spill = getattr(self, "_wl_spill_active", False)
        fl = self._wl_full_layer_idx(layer) if spill else None
        neg_inf = float("-inf")
        # #136b H2D prefetch/double-buffer: every host-spill staging copy is
        # recorded on the bucket's SIDE stream, whose whole per-step pipeline
        # was forked from the capture stream ONCE in
        # init_forward_metadata_in_graph. Because staging rows are PER-LAYER
        # device-pool slots, a layer's copies can run while EARLIER layers
        # still compute on the main stream (cross-layer prefetch); ordering
        # is expressed purely by in-capture events (they become graph edges):
        #   copy->run:    main waits copy_ev[j] before running block j;
        #   region reuse: side waits run_ev[j-2] before copying block j
        #                 (block j-2 was this layer's previous reader of
        #                 region j % 2) -- only when j-2 actually staged;
        #   owner-write:  the LAST host block's copy waits ow_ev (its slot
        #                 window holds the current token's host slot, D2H-
        #                 written by this layer's owner-write).
        # The last copy_ev wait also rejoins the side stream before capture
        # end. Copies are rank-local -- the collective count/order per layer
        # (the lockstep lynchpin) is untouched. Without prefetch (single
        # staging carve) the copies stay on the main stream: the exact #136a
        # serial behavior.
        prefetch = (
            spill
            and getattr(st, "prefetch", False)
            and any(st.stage_cnt[j] > 0 for j in range(rung))
        )
        if prefetch:
            main_stream = torch.cuda.current_stream()
            side_stream = st.copy_stream
            last_host = max(j for j in range(rung) if st.stage_cnt[j] > 0)
        o_acc = lse_acc = empty_acc = None
        for j in range(rung):
            if spill and st.stage_cnt[j] > 0:
                if prefetch:
                    with torch.cuda.stream(side_stream):
                        if j >= 2 and st.stage_cnt[j - 2] > 0:
                            side_stream.wait_event(st.run_ev[j - 2])
                        if j == last_host and st.ow_recorded:
                            side_stream.wait_event(st.ow_ev)
                        self._wl_host_pool.load_to_device_per_layer(
                            self._wl_full_pool,
                            st.stage_ids[j],
                            st.stage_dst[j],
                            fl,
                            io_backend="kernel",
                        )
                        st.copy_ev[j].record(side_stream)
                    main_stream.wait_event(st.copy_ev[j])
                else:
                    self._wl_host_pool.load_to_device_per_layer(
                        self._wl_full_pool,
                        st.stage_ids[j],
                        st.stage_dst[j],
                        fl,
                        io_backend="kernel",
                    )
            o_b, lse_b = st.wrappers[j].forward_return_lse(
                q_full,
                kv_buffer,
                sm_scale=layer.scaling,
                logits_soft_cap=layer.logit_cap,
                k_scale=layer.k_scale_float,
                v_scale=layer.v_scale_float,
            )
            if prefetch and st.stage_cnt[j] > 0:
                # Block j's attention was this layer's last reader of region
                # j % 2; copy j+2 (same region) waits this.
                st.run_ev[j].record(main_stream)
            lens = st.indptr_dev[j][1:] - st.indptr_dev[j][:-1]
            em = lens == 0
            lse_b = torch.where(em.unsqueeze(1), torch.full_like(lse_b, neg_inf), lse_b)
            o_b = torch.where(em.view(-1, 1, 1), torch.zeros_like(o_b), o_b)
            if o_acc is None:
                o_acc, lse_acc, empty_acc = o_b, lse_b, em
            else:
                o_acc, lse_acc = _safe_merge_state(o_acc, lse_acc, o_b, lse_b)
                # Requests empty in EVERY block so far must stay exactly at
                # the (0, -inf) contract (merge of two -inf partials is
                # undefined); re-pin them after each merge.
                empty_acc = empty_acc & em
                lse_acc = torch.where(
                    empty_acc.unsqueeze(1),
                    torch.full_like(lse_acc, neg_inf),
                    lse_acc,
                )
                o_acc = torch.where(
                    empty_acc.view(-1, 1, 1), torch.zeros_like(o_acc), o_acc
                )
        return o_acc, lse_acc

    def _wl_spill_owner_write_graph(self, layer, loc, dcp_kv_mask, k_full, v_full):
        """Graph-recordable owner-write over the tiered slot space (the eager
        `_wl_spill_owner_write` uses .any()/.nonzero()/host chunking -- none
        recordable). Fixed-shape formulation: masked device write of the
        device-region rows, then an UNCONDITIONAL stage of all T rows into the
        staging region followed by a fixed-count D2H gather whose destination
        is the real host slot for host-region rows and a reserved DUMP slot
        (one past the logical host tier, never read) for everything else.
        Byte-identical to the eager split for every real slot."""
        from sgl_kernel.kvcacheio import transfer_kv_per_layer

        pool = self.token_to_kv_pool
        dev_limit = self._wl_dev_slots
        host_rows = dcp_kv_mask & (loc >= dev_limit)
        dev_mask = dcp_kv_mask & ~host_rows
        safe_loc = torch.where(loc < dev_limit, loc, torch.zeros_like(loc))
        pool.set_kv_buffer(
            layer,
            safe_loc,
            k_full.clone(),
            v_full.clone(),
            layer.k_scale,
            layer.v_scale,
            dcp_kv_mask=dev_mask,
        )
        T = loc.numel()
        # #136b: under prefetch the D2H staging scratch lives in the dedicated
        # row(s) ABOVE both H2D regions (carve 2B + 1), so the side stream's
        # early cross-layer block copies (which fill [base, base + 2B)) never
        # collide with it. Without prefetch: the original base rows (#136a).
        st = self._wl_graph_state.get(getattr(self, "_wl_graph_active_bucket", None))
        prefetch = st is not None and getattr(st, "prefetch", False)
        scratch_base = self._wl_stage_base
        if prefetch:
            scratch_base += 2 * self._wl_chunk_block_size
            if scratch_base + T > self._wl_stage_base + self._wl_stage_cap:
                raise RuntimeError(
                    "weightless-KV #136b: owner-write scratch overflows the "
                    f"staging carve ({T} rows past slot {scratch_base}, cap "
                    f"{self._wl_stage_cap}); the graph bucket must stay bs=1."
                )
        stage_slots = torch.arange(
            scratch_base,
            scratch_base + T,
            dtype=torch.int64,
            device=loc.device,
        )
        # Same write kernel/dtype path as a device-resident slot; the staging
        # region is scratch (the block loop below re-streams over it).
        pool.set_kv_buffer(
            layer,
            stage_slots.to(loc.dtype),
            k_full.clone(),
            v_full.clone(),
            layer.k_scale,
            layer.v_scale,
        )
        host_dst = torch.where(
            host_rows,
            loc.to(torch.int64) - dev_limit,
            torch.full_like(loc, self._wl_host_dump_slot, dtype=torch.int64),
        )
        fl = self._wl_full_layer_idx(layer)
        transfer_kv_per_layer(
            src_k=self._wl_full_pool.k_buffer[fl],
            dst_k=self._wl_host_pool.k_data_refs[fl],
            src_v=self._wl_full_pool.v_buffer[fl],
            dst_v=self._wl_host_pool.v_data_refs[fl],
            src_indices=stage_slots,
            dst_indices=host_dst,
            item_size=self._wl_host_pool.token_stride_size,
        )
        if prefetch:
            # #136b: this layer's LAST host-block copy (whose slot window
            # holds the current token's host slot) waits this event on the
            # side stream -- the only per-layer ordering the prefetch
            # pipeline needs against the owner-write.
            st.ow_ev.record(torch.cuda.current_stream())
            st.ow_recorded = True

    def _wl_block_prefix_wrapper(self):
        """Persistent block PREFILL wrapper for the B1 streamed prefix read.
        Lazily created; re-planned per block. Shares the backend workspace --
        safe because whenever this loop runs, the MAIN paged prefix wrapper is
        never .run() for that batch (this loop replaces it entirely), and the
        ragged current-chunk wrapper's run precedes the loop on the same
        stream."""
        if self._wl_block_prefill_wrapper is None:
            self._wl_block_prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
                self.workspace_buffer,
                "NHD",
                backend=self.prefill_backend,
            )
        return self._wl_block_prefill_wrapper

    def _wl_blockwise_prefix_return_lse(
        self, q_full, layer, fallback_wrapper, logits_soft_cap=None
    ):
        """Stage B1: blockwise NON-CAUSAL paged read of this rank's OWNED
        committed-prefix slots for one extend chunk, streaming HOST-resident
        blocks through the staging region -- the extend twin of
        _wl_blockwise_decode_return_lse. Partials are online-merged with the
        SAME _safe_merge_state operator (both partials come from the same
        flashinfer paged-wrapper family), so the byte class matches the B0
        block loop. Returns (o, lse) in the same shape/space as the monolithic
        ``prefill_wrapper_paged.forward_return_lse`` so the existing cross-rank
        cp_lse merge and the ragged/prefix logaddexp combine are UNCHANGED.

        The loop is intra-rank (no collective inside); the cross-rank
        collective count of the extend layer forward is identical to the
        monolithic path by construction."""
        kv_indptr = self._dcp_extend_owned_kv_indptr
        kv_indices = self._dcp_extend_owned_kv_indices
        qo_indptr = self._dcp_extend_qo_indptr
        prefix_lens = self._dcp_extend_global_prefix_lens
        B = self._wl_chunk_block_size
        bs = kv_indptr.numel() - 1
        kv_buffer = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)
        num_qo_heads = q_full.shape[1]
        iu = self.indices_updater_prefill
        num_kv_heads = iu.num_kv_heads
        head_dim = q_full.shape[2]

        if logits_soft_cap is None:
            logits_soft_cap = layer.logit_cap

        # Rank-uniform block count from the GLOBAL prefix length (same nested-
        # ceiling identity as the decode loop: a global window of B * dcp_size
        # positions holds ~B owned slots on every rank and covers all of them).
        global_max = int(prefix_lens.max().item())
        per_block_global = B * self.dcp_size
        num_blocks = max(1, (global_max + per_block_global - 1) // per_block_global)

        indptr_host = kv_indptr.to("cpu")
        wrapper = self._wl_block_prefix_wrapper()
        last_page = self.kv_last_page_len

        o_acc = None
        lse_acc = None
        for j in range(num_blocks):
            blk_indices, blk_indptr = self._wl_stage_block(
                kv_indices, indptr_host, bs, j, B, layer
            )
            if blk_indices is None:
                continue
            wrapper.plan(
                qo_indptr,
                blk_indptr,
                blk_indices,
                last_page[:bs],
                num_qo_heads,
                num_kv_heads,
                head_dim,
                1,  # page_size
                causal=False,
                q_data_type=iu.q_data_type,
                kv_data_type=iu.data_type,
            )
            o_b, lse_b = wrapper.forward_return_lse(
                q_full,
                kv_buffer,
                causal=False,
                sm_scale=layer.scaling,
                logits_soft_cap=logits_soft_cap,
                k_scale=layer.k_scale_float,
                v_scale=layer.v_scale_float,
            )
            if o_acc is None:
                o_acc, lse_acc = o_b, lse_b
            else:
                o_acc, lse_acc = _safe_merge_state(o_acc, lse_acc, o_b, lse_b)
        if o_acc is None:
            # This rank owns ZERO prefix slots (only possible for tiny global
            # prefixes -- necessarily all device-resident, so the monolithic
            # wrapper is safe). Reproduce the monolithic empty-attention
            # contract (o=0, lse=-inf) exactly as the baseline does.
            o_acc, lse_acc = fallback_wrapper.forward_return_lse(
                q_full,
                kv_buffer,
                causal=False,
                sm_scale=layer.scaling,
                logits_soft_cap=logits_soft_cap,
                k_scale=layer.k_scale_float,
                v_scale=layer.v_scale_float,
            )
        return o_acc, lse_acc

    def _forward_decode_dcp(
        self, q, k, v, layer, forward_batch, decode_wrapper, cache_loc, save_kv_cache
    ):
        group = get_parallel().dcp_group
        if k is not None and save_kv_cache:
            self._dcp_masked_write(layer, forward_batch, cache_loc, k, v)

        q_local = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
        q_full = cp_all_gather_heads_uneven(q_local, group, self.dcp_q_head_counts)
        if self._sess_spill is not None:
            # kv-session-offload spill tick: attend the host-resident shard
            # via the streamed block loop. Same (o, lse) contract and the
            # SAME per-layer collective count as the monolithic path -- the
            # cross-rank cp_lse merge below is untouched.
            o, lse = self._sess_blockwise_decode_return_lse(q_full, layer)
        elif self.weightless_kv and getattr(self, "_wl_chunk_block_size", 0):
            # Stage B0: block-decode the owned shard (byte-identical up to fp
            # reassociation). Strictly the weightless lane; rank-uniform with the
            # workers' forward_decode_weightless_worker block loop. Under CUDA-
            # graph CAPTURE (#136a) record the fixed-count graph variant instead
            # (plans hoisted out-of-graph); replay never re-enters this Python.
            if self._wl_graph_capture_blocks is not None:
                o, lse = self._wl_blockwise_decode_return_lse_graph(q_full, layer)
            else:
                o, lse = self._wl_blockwise_decode_return_lse(q_full, layer)
        else:
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

    def forward_decode_weightless_worker(self, layer, forward_batch):
        """Weightless-KV WORKER dispatch for one full-attention layer (Option-B
        B1). This rank holds NO layer weights and projects NOTHING; it owns only
        a DCP token-shard of the KV. It contributes an empty [T,0,D] head slice
        to every per-layer collective (so the head rank's Q/K/V broadcast and the
        LSE-merge complete), writes the head's broadcast K,V into its owned KV
        slots, and attends its local KV shard. The merged output is delivered to
        the head rank only; this rank's slice is empty and discarded.

        Issues the IDENTICAL dcp_group collective sequence as the head rank's
        _forward_decode_dcp (fused KV all-gather [#132], Q all-gather,
        LSE-merge) so the two stay in lockstep; the anti-hang guard wraps each
        one."""
        assert self.weightless_kv and not self.dcp_kv_replicated_heads
        group = get_parallel().dcp_group
        cache_loc = forward_batch.out_cache_loc
        T = cache_loc.shape[0]
        hd = layer.head_dim
        dev = cache_loc.device
        # 0-head local contributions (this worker projects nothing). Shapes are
        # [T, 0, D] so the padded all-gather broadcasts the head rank's real Q/K/V.
        zk = torch.zeros((T, 0, hd), dtype=self._wl_dtype, device=dev)
        zv = torch.zeros((T, 0, hd), dtype=self._wl_dtype, device=dev)
        zq = torch.zeros((T, 0, hd), dtype=self._wl_dtype, device=dev)
        # (1) KV broadcast from the head rank -> full replicated kv-heads;
        # then write only this rank's owned token slots (shared owner-rule).
        # #132 Fusion 1: the head fuses its k+v gathers into ONE stacked
        # collective (_dcp_write_gather: cat along dim 0, split at T) -- this
        # worker MUST mirror it 1:1 (lockstep lynchpin: identical collective
        # count + order per layer, else NCCL pairs mismatched calls -> hang).
        kv_full = cp_all_gather_heads_uneven(
            torch.cat((zk, zv), dim=0), group, self.dcp_kv_head_counts
        )
        k_full, v_full = kv_full[:T], kv_full[T:]
        self._dcp_owner_write(layer, forward_batch, cache_loc, k_full, v_full)
        # (3) Q broadcast from the head rank -> q_full (all heads) on this rank.
        q_full = cp_all_gather_heads_uneven(zq, group, self.dcp_q_head_counts)
        # attend this rank's LOCAL KV token-shard -> partial o + lse (all heads).
        if getattr(self, "_wl_chunk_block_size", 0):
            # Stage B0: block-decode the owned shard, online-merged (byte-
            # identical up to fp reassociation). Rank-uniform with the head's
            # _forward_decode_dcp block loop -- SAME block count (derived from the
            # shared global seq_len), NO cross-rank collective added. Under
            # CUDA-graph CAPTURE (#136a) record the fixed-count graph variant
            # (symmetric with the head's captured block loop).
            if self._wl_graph_capture_blocks is not None:
                o, lse = self._wl_blockwise_decode_return_lse_graph(q_full, layer)
            else:
                o, lse = self._wl_blockwise_decode_return_lse(q_full, layer)
        else:
            decode_wrapper = self.forward_metadata.decode_wrappers[
                self._get_wrapper_idx(layer)
            ]
            o, lse = decode_wrapper.forward_return_lse(
                q_full,
                self.token_to_kv_pool.get_kv_buffer(layer.layer_id),
                sm_scale=layer.scaling,
                logits_soft_cap=layer.logit_cap,
                k_scale=layer.k_scale_float,
                v_scale=layer.v_scale_float,
            )
        # (4) LSE-merge: contribute this rank's partial; receive the empty
        # [T,0,D] slice (the merged output goes to the head rank only).
        cp_lse_ag_out_ar_mha_uneven(o, lse, group, self.dcp_q_head_counts)

    def forward_extend_weightless_worker(self, layer, forward_batch):
        """Weightless-KV WORKER dispatch for one full-attention layer in
        EXTEND/PREFILL (Option-B B1). Mirrors the head's _forward_extend_dcp DCP
        collective sequence with this rank's empty [T,0,D] head slice + its KV
        token-shard. The head's per-layer sequence is DATA-DEPENDENT:
          * no prefix (first chunk): fused K+V all-gather (#132)             -> 1
          * has prefix:              fused K+V, Q all-gather, LSE-merge       -> 3
        `has_prefix` is a rank-uniform (global) property, so the head and this
        worker take the identical branch -> the collective counts match. The
        head's current-chunk RAGGED attention is LOCAL (no collective), so the
        worker skips it entirely and contributes nothing there."""
        assert self.weightless_kv and not self.dcp_kv_replicated_heads
        group = get_parallel().dcp_group
        cache_loc = forward_batch.out_cache_loc  # current-chunk slots (multi-tok)
        T = cache_loc.shape[0]
        hd = layer.head_dim
        dev = cache_loc.device
        zk = torch.zeros((T, 0, hd), dtype=self._wl_dtype, device=dev)
        zv = torch.zeros((T, 0, hd), dtype=self._wl_dtype, device=dev)
        zq = torch.zeros((T, 0, hd), dtype=self._wl_dtype, device=dev)
        # (1) KV broadcast from the head (ONE fused collective, #132: the head's
        # _dcp_write_gather stacks k+v along dim 0 into a single all-gather;
        # this worker mirrors it 1:1 -- lockstep lynchpin) + multi-token owner
        # write into this rank's owned prefix slots (shared owner rule, already
        # multi-token). IDENTICAL to the head's _dcp_masked_write path.
        kv_full = cp_all_gather_heads_uneven(
            torch.cat((zk, zv), dim=0), group, self.dcp_kv_head_counts
        )
        k_full, v_full = kv_full[:T], kv_full[T:]
        self._dcp_owner_write(layer, forward_batch, cache_loc, k_full, v_full)
        # (2) current-chunk ragged attention: head-LOCAL, NO collective -> skip.
        # (3) has_prefix -- computed IDENTICALLY to the head (rank-uniform), so
        # the Q/LSE collectives below stay balanced with the head. ONE
        # expression, shared with _forward_extend_dcp; see
        # layers/dcp/lockstep.weightless_has_prefix for why forward_mode must
        # answer first (#180 D5).
        has_prefix = weightless_has_prefix(
            forward_batch.forward_mode.is_target_verify(),
            forward_batch.extend_prefix_lens_cpu,
        )
        if not has_prefix:
            return
        # (4) Q broadcast (1 collective) -> non-causal PAGED prefix read over this
        # rank's OWNED prefix slots (LOCAL) -> LSE-merge (1 collective). The
        # merged output slice is empty [T,0,D] for the worker and discarded.
        q_full = cp_all_gather_heads_uneven(zq, group, self.dcp_q_head_counts)
        prefill_wrapper_paged = self.forward_metadata.prefill_wrappers[
            self._get_wrapper_idx(layer)
        ]
        if getattr(self, "_wl_spill_active", False) and getattr(
            self, "_dcp_extend_has_host", False
        ):
            # Stage B1: stream this rank's host-resident owned prefix blocks
            # (rank-LOCAL loop, collective count unchanged -- mirrors the
            # head's _forward_extend_dcp branch).
            o_pre, lse_pre = self._wl_blockwise_prefix_return_lse(
                q_full, layer, prefill_wrapper_paged
            )
        else:
            o_pre, lse_pre = prefill_wrapper_paged.forward_return_lse(
                q_full,
                self.token_to_kv_pool.get_kv_buffer(layer.layer_id),
                causal=False,
                sm_scale=layer.scaling,
                logits_soft_cap=layer.logit_cap,
                k_scale=layer.k_scale_float,
                v_scale=layer.v_scale_float,
            )
        cp_lse_ag_out_ar_mha_uneven(o_pre, lse_pre, group, self.dcp_q_head_counts)

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
        do_write = k is not None and save_kv_cache

        causal = (
            not layer.is_cross_attention
            and layer.attn_type != AttentionType.ENCODER_ONLY
        )
        q_local = q.view(-1, layer.tp_q_head_num, layer.head_dim)

        # has_prefix is a global (rank-uniform) property, so every rank takes the
        # same branch -> the DCP collectives below stay balanced (no deadlock).
        # target-VERIFY (force_prefix) always reads the committed prefix, whose
        # length is seq_lens (extend_prefix_lens is unset for verify).
        # (CPU-side info, so it is hoisted above the kernels: the overlapped
        # comm lane must know up front whether the q-gather B will be issued.)
        # ONE expression, shared with forward_extend_weightless_worker.
        has_prefix = weightless_has_prefix(
            force_prefix, forward_batch.extend_prefix_lens_cpu
        )

        comm_stream = self.dcp_comm_stream
        if comm_stream is None:
            # ================= SEQUENTIAL (legacy) SCHEDULING =================
            # 1. Write the current chunk's KV (full replicated kv-heads) into
            #    this rank's owned token slots (token-sharded cache).
            _scatter_late = self.dcp_overlap_scatter_late
            k_full_seq = v_full_seq = None
            if do_write:
                if _scatter_late:
                    k_full_seq, v_full_seq = self._dcp_write_gather(layer, k, v)
                else:
                    self._dcp_masked_write(layer, forward_batch, cache_loc, k, v)
            # 2. Current chunk: ragged LOCAL head-sharded attention (causal).
            o_cur, lse_cur = self._dcp_ragged_current(
                q_local, k, v, layer, causal, logits_soft_cap
            )
            if not has_prefix:
                if do_write and _scatter_late:
                    self._dcp_write_scatter(
                        layer, forward_batch, cache_loc, k_full_seq, v_full_seq
                    )
                return o_cur.contiguous().view(-1, layer.tp_q_head_num * layer.head_dim)
            # 3. Prefix: paged DCP read over this rank's OWNED prefix slots with
            #    the FULL gathered q-heads (non-causal: all prefix keys precede
            #    every current query). Combine the per-rank partials across the
            #    DCP group and slice back to this rank's local heads.
            q_full = cp_all_gather_heads_uneven(q_local, group, self.dcp_q_head_counts)
            if self._sess_verify_active():
                # C4 (spec-in-spill-tick): this rank's committed prefix lives on
                # host (kv-session-offload spill) -- stream it blockwise via the
                # session spill region with the multi-row NON-CAUSAL verify twin.
                # Rank-LOCAL (no collective inside); the cp_lse below is reached
                # identically. (_sess and _wl spills are mutually exclusive.)
                o_pre_raw, lse_pre_raw = self._sess_blockwise_prefix_return_lse(
                    q_full, layer, logits_soft_cap
                )
            elif getattr(self, "_wl_spill_active", False) and getattr(
                self, "_dcp_extend_has_host", False
            ):
                # Stage B1: part of this rank's owned prefix lives on host --
                # stream it blockwise through the staging region. Rank-LOCAL
                # branch (no collective inside the loop); the cp_lse below is
                # reached identically either way.
                o_pre_raw, lse_pre_raw = self._wl_blockwise_prefix_return_lse(
                    q_full, layer, prefill_wrapper_paged, logits_soft_cap
                )
            else:
                o_pre_raw, lse_pre_raw = prefill_wrapper_paged.forward_return_lse(
                    q_full,
                    self.token_to_kv_pool.get_kv_buffer(layer.layer_id),
                    causal=False,
                    sm_scale=layer.scaling,
                    logits_soft_cap=logits_soft_cap,
                    k_scale=layer.k_scale_float,
                    v_scale=layer.v_scale_float,
                )
            o_pre, lse_pre = cp_lse_ag_out_ar_mha_uneven(
                o_pre_raw,
                lse_pre_raw,
                group,
                self.dcp_q_head_counts,
                return_lse=True,
            )
            if do_write and _scatter_late:
                if os.environ.get("SGLANG_DCP_DEBUG") == "1" and layer.layer_id < 8:
                    self._dcp_debug_overlap_probe(
                        layer, forward_batch, cache_loc, force_prefix
                    )
                self._dcp_write_scatter(
                    layer, forward_batch, cache_loc, k_full_seq, v_full_seq
                )
            return self._dcp_extend_final_merge(
                q, layer, o_cur, lse_cur, o_pre, lse_pre
            )

        # ================= OVERLAPPED SCHEDULING (#128) =================
        # Two-lane schedule. The per-layer collective ISSUE ORDER on the DCP
        # communicator is exactly the sequential order (A_k, A_v, B, C, D) --
        # only independent COMPUTE is moved to run concurrently on the other
        # lane. Every kernel, reduction and merge is unchanged, so the result
        # is byte-identical to the sequential scheduling.
        #
        #   comm lane:  A_k -> A_v -> B ......... (wait paged) C -> merge -> D
        #   main lane:  ragged current-chunk attn  (wait B) paged  scatter-write
        #
        # Cross-lane edges (wait_stream) keep every same-communicator
        # collective pair strictly ordered (never concurrent), and the next
        # layer's fork doubles as the back-edge that makes cross-stream
        # allocator reuse safe (comm-lane blocks are only reusable after the
        # main lane's consumers are ordered behind them).
        import contextlib

        _reorder_only = self.dcp_overlap_reorder_only
        _comm_ctx = (
            contextlib.nullcontext()
            if _reorder_only
            else torch.cuda.stream(comm_stream)
        )
        cur_stream = torch.cuda.current_stream()
        if not _reorder_only:
            comm_stream.wait_stream(cur_stream)  # fork: q/k/v are ready
        k_full = v_full = q_full = None
        with _comm_ctx:
            if do_write:
                # A_k, A_v (no-op without collectives in replicated-KV mode)
                k_full, v_full = self._dcp_write_gather(layer, k, v)
            if has_prefix:
                # B: q-head all-gather
                q_full = cp_all_gather_heads_uneven(
                    q_local, group, self.dcp_q_head_counts
                )
        # main lane: the ragged current-chunk attention is independent of
        # A/B/C/D and of the paged prefix read -> overlaps A_k/A_v/B.
        o_cur, lse_cur = self._dcp_ragged_current(
            q_local, k, v, layer, causal, logits_soft_cap
        )
        if not has_prefix:
            # Join, then scatter the (gathered) kv into this rank's owned
            # slots before returning -- same cache state as the sequential
            # path at layer exit.
            if not _reorder_only:
                cur_stream.wait_stream(comm_stream)
            if do_write:
                self._dcp_write_scatter(layer, forward_batch, cache_loc, k_full, v_full)
            return o_cur.contiguous().view(-1, layer.tp_q_head_num * layer.head_dim)

        # main lane waits for B (and transitively A) before the paged read.
        if not _reorder_only:
            cur_stream.wait_stream(comm_stream)
        if self._sess_verify_active():
            # C4 (spec-in-spill-tick): host-resident committed prefix -> the
            # multi-row NON-CAUSAL verify twin over the session spill region.
            # Rank-LOCAL main-lane compute (no collective inside the loop), so
            # the two-lane schedule is unchanged: C still follows on the comm
            # lane exactly as with the plain paged read.
            o_pre_raw, lse_pre_raw = self._sess_blockwise_prefix_return_lse(
                q_full, layer, logits_soft_cap
            )
        elif getattr(self, "_wl_spill_active", False) and getattr(
            self, "_dcp_extend_has_host", False
        ):
            # Stage B1 (weightless-KV): part of this rank's owned prefix lives
            # on host -- stream it blockwise through the staging region.
            # Rank-LOCAL main-lane compute (no collective inside the loop), so
            # the two-lane schedule is unchanged: C still follows on the comm
            # lane exactly as with the plain paged read.
            o_pre_raw, lse_pre_raw = self._wl_blockwise_prefix_return_lse(
                q_full, layer, prefill_wrapper_paged, logits_soft_cap
            )
        else:
            o_pre_raw, lse_pre_raw = prefill_wrapper_paged.forward_return_lse(
                q_full,
                self.token_to_kv_pool.get_kv_buffer(layer.layer_id),
                causal=False,
                sm_scale=layer.scaling,
                logits_soft_cap=logits_soft_cap,
                k_scale=layer.k_scale_float,
                v_scale=layer.v_scale_float,
            )
        # comm lane: C (LSE all-gather) + merge math + D (out all-reduce) --
        # unchanged function, unchanged reduction order, just issued on the
        # comm stream so the main lane can scatter-write concurrently.
        if not _reorder_only:
            comm_stream.wait_stream(cur_stream)
        _merge_ctx = (
            contextlib.nullcontext()
            if _reorder_only
            else torch.cuda.stream(comm_stream)
        )
        with _merge_ctx:
            o_pre, lse_pre = cp_lse_ag_out_ar_mha_uneven(
                o_pre_raw,
                lse_pre_raw,
                group,
                self.dcp_q_head_counts,
                return_lse=True,
            )
        # main lane: the masked scatter-write targets the current chunk's
        # out_cache_loc slots, disjoint from the paged prefix read above and
        # untouched by the merge -> overlaps C/merge/D.
        if do_write:
            self._dcp_write_scatter(layer, forward_batch, cache_loc, k_full, v_full)
        # join: the merged prefix partials must be complete before the final
        # local merge consumes them (the attention output is consumed by
        # o_proj right after this returns).
        if not _reorder_only:
            cur_stream.wait_stream(comm_stream)
        # NOTE (allocator safety, eager mode): o_pre_raw/lse_pre_raw are
        # main-lane allocations read by the comm lane; they stay referenced
        # until after this join, and any post-join reuse of their blocks is
        # ordered behind the comm lane by the join itself. Comm-lane
        # allocations (k_full/v_full/q_full/o_pre/lse_pre) are consumed by
        # the main lane after joins; their blocks only become reusable for
        # the comm lane at the NEXT fork, which waits on the main lane.
        return self._dcp_extend_final_merge(q, layer, o_cur, lse_cur, o_pre, lse_pre)

    def _dcp_debug_overlap_probe(self, layer, forward_batch, cache_loc, force_prefix):
        """Diagnostic (#128): report any intersection between this layer's
        paged PREFIX read slots and the current chunk's scatter-write slots."""
        try:
            req_to_token = self.indices_updater_prefill.req_to_token
            reqs = forward_batch.req_pool_indices
            if force_prefix:
                prefix_lens = forward_batch.seq_lens
            else:
                prefix_lens = forward_batch.extend_prefix_lens
            locs = []
            for i in range(len(reqs)):
                plen = int(prefix_lens[i].item())
                if plen > 0:
                    locs.append(req_to_token[reqs[i], :plen])
            if not locs:
                return
            pre = torch.cat(locs).to(torch.int64)
            off = pre % self.cp_S
            owned = (off >= self.cp_lo) & (off < self.cp_hi)
            pre_compact = (pre // self.cp_S) * self.cp_ratio + (off - self.cp_lo)
            pre_set = pre_compact[owned]
            wl = cache_loc.to(torch.int64)
            woff = wl % self.cp_S
            wowned = (woff >= self.cp_lo) & (woff < self.cp_hi)
            w_compact = (wl // self.cp_S) * self.cp_ratio + (woff - self.cp_lo)
            w_set = w_compact[wowned]
            inter = torch.isin(w_set, pre_set)
            n_inter = int(inter.sum().item())
            mode = "VERIFY" if force_prefix else "EXTEND"
            logger.info(
                "[DCP-DEBUG] %s layer=%d prefix_owned=%d write_owned=%d "
                "INTERSECT=%d pre_virt_minmax=(%d,%d) w_virt_minmax=(%d,%d)",
                mode,
                layer.layer_id,
                int(pre_set.numel()),
                int(w_set.numel()),
                n_inter,
                int(pre.min().item()),
                int(pre.max().item()),
                int(wl.min().item()) if wl.numel() else -1,
                int(wl.max().item()) if wl.numel() else -1,
            )
        except Exception as e:  # diagnostic only -- never break the forward
            logger.warning("[DCP-DEBUG] probe failed: %s", e)

    def _dcp_ragged_current(self, q_local, k, v, layer, causal, logits_soft_cap):
        """Current chunk: ragged LOCAL head-sharded attention (causal). k/v are
        the freshly-projected bf16 current tokens, fully present on every
        rank, so this is exact local GQA attention (no DCP gather needed).
        active_ragged_wrapper: under a captured verify graph this is the
        per-bucket graph-mode wrapper (fixed indptr buffers) -- the shared
        non-graph wrapper's transient plan tensors are NOT replay-safe.

        REPLICATED-KV (TP > num_kv_heads, #105): this rank holds only a q-head
        SLICE but ALL kv heads (replicated). flashinfer derives the ragged
        GQA grouping from the LOCAL counts (gqa_local = local_q // local_kv),
        which maps q heads to kv heads differently from the GLOBAL grouping
        (gqa_global = total_q // total_kv) whenever this rank does not hold
        every q head of its kv head(s). Re-index the local kv slots so each
        carries the GLOBAL kv head its q group actually attends. (8q/2kv over
        [4,2,2]: rank0 held q0-3 -- all -> kv0 globally -- but gqa_local=2 sent
        q2-3 to kv1, corrupting short-prompt / first-chunk generation while the
        gathered-q paged prefix/decode path stayed correct.) No-op reindex on
        the aligned single-kv-head-per-rank case reduces to identity."""
        k_cur = k.view(-1, layer.tp_k_head_num, layer.head_dim)
        v_cur = v.view(-1, layer.tp_v_head_num, layer.head_dim)
        if self.dcp_kv_replicated_heads:
            _kv_idx = self._replicated_kv_ragged_reindex(
                layer.tp_q_head_num, layer.tp_k_head_num, k.device
            )
            if _kv_idx is not None:
                k_cur = k_cur[:, _kv_idx, :]
                v_cur = v_cur[:, _kv_idx, :]
        return self.active_ragged_wrapper.forward_return_lse(
            q_local,
            k_cur,
            v_cur,
            causal=causal,
            sm_scale=layer.scaling,
            logits_soft_cap=logits_soft_cap,
        )

    def _dcp_extend_final_merge(self, q, layer, o_cur, lse_cur, o_pre, lse_pre):
        """4. Merge current chunk with prefix (both in this rank's local-head
        space) by natural-log LSE. lse_pre is a torch.logsumexp (natural
        log) from the cross-rank combine, and flashinfer's ragged
        forward_return_lse also returns natural-log LSE, so logaddexp mixes
        them consistently. (flashinfer's merge_state uses a different
        internal convention and must NOT be used across these two sources.)"""
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
        return o.to(q.dtype).contiguous().view(-1, layer.tp_q_head_num * layer.head_dim)

    def _get_wrapper_idx(self, layer: RadixAttention):
        if self.num_wrappers == 1:
            return 0

        if self.dispatch_reason == WrapperDispatch.SLIDING_WINDOW:
            return layer.sliding_window_size == -1
        if self.dispatch_reason == WrapperDispatch.CROSS_ATTENTION:
            return layer.is_cross_attention

        raise ValueError(f"Unknown dispatch reason: {self.dispatch_reason}")


def _build_dcp_ragged_tree_mask(
    full_mask: torch.Tensor,
    seq_lens: torch.Tensor,
    paged_kernel_lens_sum: int,
    draft_num: int,
    bs: int,
) -> torch.Tensor:
    """Slice the draft->draft tree sub-mask out of the EAGLE FULL_MASK for the
    uneven-DCP ragged verify wrapper (--speculative-eagle-topk > 1).

    ``full_mask`` is the flattened EAGLE verify mask (TreeMaskMode.FULL_MASK):
    for request i it is ``draft_num`` query rows, each with
    ``seq_lens[i] + draft_num`` columns (the first ``seq_lens[i]`` are the
    committed prefix -- all visible -- and the last ``draft_num`` are the tree
    topology over the draft tokens). The DCP verify splits attention into a
    paged NON-causal prefix read (token-sharded, LSE-merged) and a ragged
    draft->draft read (local heads); the tree mask affects ONLY the latter, so
    we extract just the trailing draft_num x draft_num block per request and
    hand it to the ragged wrapper. The prefix is never masked (correctness:
    the mask is a draft-local, rank-uniform property; token-sharding of the
    prefix is orthogonal to it).

    Returns a flat bool mask of length ``bs * draft_num * draft_num``, laid out
    per request row-major -- exactly flashinfer's ragged custom_mask layout for
    qo_indptr == kv_indptr == [0, d, 2d, ...].
    """
    device = full_mask.device
    d = draft_num
    seq64 = seq_lens.to(torch.int64)
    # Under CUDA-graph replay the batch is padded; the worker built full_mask
    # for the real requests only. Pad the tail with True (padded requests are
    # discarded) so every gather index below is in-bounds -- the exact pattern
    # EagleVerifyInput.generate_attn_arg_prefill uses for the non-DCP path.
    expected = paged_kernel_lens_sum * d + d * d * bs
    if full_mask.numel() < expected:
        # dtype follows full_mask: the worker's tree mask is bool, the
        # capture-time dummy (buffers.custom_mask) is uint8; both are
        # nonzero==visible for flashinfer's packbits.
        full_mask = torch.cat(
            [
                full_mask,
                torch.ones(
                    expected - full_mask.numel(),
                    dtype=full_mask.dtype,
                    device=device,
                ),
            ]
        )
    klen = seq64 + d  # per-request kv_len (prefix + draft)
    base = torch.zeros(bs, dtype=torch.int64, device=device)
    if bs > 1:
        base[1:] = torch.cumsum(d * klen, dim=0)[:-1]
    r = torch.arange(d, device=device)
    c = torch.arange(d, device=device)
    # idx[i, r, c] = base_i + r * klen_i + seq_i + c   (trailing draft columns)
    idx = (
        base.view(bs, 1, 1)
        + r.view(1, d, 1) * klen.view(bs, 1, 1)
        + seq64.view(bs, 1, 1)
        + c.view(1, 1, d)
    ).reshape(-1)
    return full_mask[idx]


# Wall A (cross-vendor uneven DCP): the weighted owner rule moved VERBATIM to
# sglang.srt.layers.dcp.owner -- it is pure torch + a shared Triton kernel and
# never had a flashinfer dependency, so it can also serve the Triton backend
# (and thus ROCm/gfx900 and sm75). Imported at module scope above and aliased
# here so every call site in this file stays byte-identical.
_build_dcp_weighted_kv_indices = build_dcp_weighted_kv_indices


def _dcp_host_total_tokens(
    extend_prefix_lens_cpu: Optional[Union[List[int], torch.Tensor]],
    expected_sum: Optional[int] = None,
) -> Optional[int]:
    """Host-side ``sum(prefix_lens)`` for the weighted-DCP index build, or None.

    #616c: ``build_dcp_weighted_kv_indices`` otherwise derives that sum with
    ``int(full_indptr[bs].item())`` -- an UNBOUNDED blocking device-to-host read
    sitting inside the collective window. That is the site the 2026-08-07 02:00
    wedge died on, with all three ranks stopped on exactly that line while their
    device queues each held a BAR1 spin kernel; a host parked in a CUDA sync can
    neither poll nor time out nor enqueue, so no rank could release the others
    and the peers' spin kernels ran out their 300e9-cycle deadline. barlink's own
    staged-status wait (``barlink_bar1._wait_ctl_event``) is a BOUNDED poll that
    returns and lets the host keep running, which is why wedges that park there
    recover and this one did not.

    ``prefix_lens`` is ``forward_batch.extend_prefix_lens`` (flashinfer_backend
    line ~1681) and ``extend_prefix_lens_cpu`` is its host mirror, so this sum is
    the same number the device read would have produced -- not an estimate. When
    no mirror is supplied the caller passes None and the old read is kept.

    #623: the other four builder call sites in this file take their mirror from
    a different vector (``seq_lens_cpu``, or the draft-extend subtraction of
    it), so the shared host math lives in ``layers/dcp/layout.py`` and this
    stays the named entry point for the extend site. ``expected_sum`` is the
    optional host-side staleness check documented there -- a mirror that does
    not sum to the caller's independently known total is refused, and the
    caller keeps its device read.
    """
    return dcp_host_total_tokens(extend_prefix_lens_cpu, expected_sum)


def _host_clamp_max(
    lens_cpu: Optional[Union[List[int], torch.Tensor]],
    cap: int,
) -> Optional[torch.Tensor]:
    """``min(lens, cap)`` on the host, or None when no mirror is usable (#629).

    The sliding-window paged length under use_ragged is written on the device
    as ``prefix_lens - clamp(prefix_lens - W, min=0)``. That is ``min(prefix,
    W)`` by identity for non-negative lengths, so this is the same integer
    vector, not a re-derivation that could drift from it.
    """
    lens = dcp_host_lens(lens_cpu)
    return None if lens is None else torch.clamp(lens, max=cap)


def _host_swa_paged_lens(
    seq_lens_cpu: Optional[Union[List[int], torch.Tensor]],
    prefix_lens_cpu: Optional[Union[List[int], torch.Tensor]],
    window: int,
) -> Optional[torch.Tensor]:
    """``min(seq, W + seq - prefix)`` on the host, or None (#629).

    The non-ragged window branch needs BOTH host vectors; with either missing
    there is no mirror to claim and the caller keeps its device read.
    """
    seq = dcp_host_lens(seq_lens_cpu)
    prefix = dcp_host_lens(prefix_lens_cpu)
    if seq is None or prefix is None or seq.shape != prefix.shape:
        return None
    return torch.minimum(seq, window + seq - prefix)


def _host_sum_or_device(
    lens_cpu: Optional[torch.Tensor],
    lens: torch.Tensor,
) -> int:
    """The length sum from the host mirror, else the old blocking read (#629)."""
    if lens_cpu is None:
        return lens.sum().item()
    return int(lens_cpu.sum())


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
        if self.attn_backend.uneven_dcp and (
            spec_info is None or getattr(spec_info, "kv_indptr", None) is None
        ):
            # Uneven-DCP decode: this rank's paged read sees only the token
            # slots it OWNS (even-modulo owner rule pos % dcp_size == dcp_rank,
            # physical slot = global_loc // dcp_size). Build per-rank owned
            # lengths + compacted kv_indices via the DCP index kernel.
            bs = len(req_pool_indices)
            dcp_size = self.attn_backend.dcp_size
            dcp_rank = self.attn_backend.dcp_rank
            # #623: host mirror of the length vector this branch indexes over.
            # Every caller of this method sets seq_lens_cpu to the mirror of the
            # paged_kernel_lens it passes (update_single_wrapper: seq_lens /
            # seq_lens_cpu; update_sliding_window: both clamped by the same
            # window; update_cross_attention: seq_lens or encoder_lens with the
            # matching .cpu()), and paged_kernel_lens_sum is that vector's sum
            # in all three -- so it doubles as the staleness check.
            host_lens = dcp_host_lens(seq_lens_cpu, expected_sum=paged_kernel_lens_sum)
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
                    # #623: same unbounded-D2H removal as the extend site.
                    # None when no usable mirror -> the old device read.
                    total_tokens=dcp_host_total_tokens(host_lens),
                )
            else:
                dcp_lens = get_dcp_lens(
                    paged_kernel_lens, dcp_size, dcp_rank, kv_start_idx
                )
                kv_indptr[1 : bs + 1] = torch.cumsum(dcp_lens, dim=0)
                kv_indptr = kv_indptr[: bs + 1]
                # #618: the even branch's own unbounded sync. kv_start_idx is
                # part of the length formula and is device-only here, so this
                # falls back whenever it is set (cross-attention, sliding
                # window) and only skips the read on the plain decode path.
                n_dcp = dcp_host_even_total(
                    host_lens, dcp_size, dcp_rank, start=kv_start_idx
                )
                kv_indices = torch.empty(
                    (n_dcp if n_dcp is not None else int(dcp_lens.sum().item())),
                    dtype=torch.int32,
                    device="cuda",
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
            # Stage B0 (weightless block-decode): stash THIS rank's owned kv
            # layout so the per-layer block loop can slice the owned shard into
            # staging blocks. Reference assignment only; a no-op when the block
            # feature is off (default), so the monolithic path is untouched.
            if getattr(self.attn_backend, "_wl_chunk_block_size", 0):
                self.attn_backend._dcp_decode_owned_kv_indptr = kv_indptr
                self.attn_backend._dcp_decode_owned_kv_indices = kv_indices
                self.attn_backend._dcp_decode_global_seq_lens = paged_kernel_lens
                # #136a graph replay prep consumes the host-side owned indptr
                # (None on the eager path, where the prep never runs).
                self.attn_backend._dcp_decode_owned_kv_indptr_host = (
                    dcp_graph_indptr_host
                )
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
            # #623: the host mirror OF paged_kernel_lens, whichever vector that
            # turned out to be. Named for the tensor it mirrors rather than for
            # its source, because the two DCP spec branches downstream index
            # over paged_kernel_lens and must not be handed the extend mirror by
            # accident -- that confusion is exactly what made paged_kernel_lens_sum
            # (sum of seq_lens) the wrong number at the extend site (NOTE_616h).
            paged_kernel_lens_cpu = extend_prefix_lens_cpu
        else:
            paged_kernel_lens = seq_lens
            paged_kernel_lens_sum = seq_lens_sum
            paged_kernel_lens_cpu = seq_lens_cpu

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
            # #616c: forwarded so the weighted-DCP branch of call_begin_forward
            # can derive sum(prefix_lens) on the host. The branch above only
            # consumes this mirror when use_ragged is True, and a multimodal
            # model forces use_ragged=False (line ~1684) while STILL taking the
            # DCP split -- which is exactly how this rig reached the blocking
            # read. Forwarding it makes the mirror available on both paths.
            extend_prefix_lens_cpu=extend_prefix_lens_cpu,
            # #623: the mirror of paged_kernel_lens for the two DCP SPEC
            # branches (target-verify, draft-extend), which index over that
            # vector and not over prefix_lens.
            paged_kernel_lens_cpu=paged_kernel_lens_cpu,
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
        # #629: extend_prefix_lens_cpu mirrors the prefix_lens the CALLER
        # passed. When prefix_lens arrives None it is derived below from
        # seq_lens (and possibly a device-only num_accept_tokens), and the
        # incoming mirror no longer describes it -- so the mirror is dropped
        # rather than paired with a vector it does not match.
        prefix_lens_cpu = extend_prefix_lens_cpu if prefix_lens is not None else None
        # #629: the mirror used for SIZING is freshness-guarded -- seq_lens_cpu
        # is a non-None but STALE slice exactly when seq_lens_sum is None, and
        # sizing an index buffer from it is a silent mis-size. The mirror
        # FORWARDED to call_begin_forward stays raw, matching
        # update_single_wrapper: fast_prefill_plan asserts it is not None, and
        # dropping it there would convert a slow path into a hard failure.
        fresh_seq_lens_cpu = dcp_fresh_host_lens(seq_lens_cpu, seq_lens_sum)
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
                    # #629: prefix - max(prefix - W, 0) == min(prefix, W), the
                    # same integers by identity rather than by re-derivation.
                    paged_kernel_lens_cpu = _host_clamp_max(
                        prefix_lens_cpu, sliding_window_size
                    )
                    paged_kernel_lens_sum = _host_sum_or_device(
                        paged_kernel_lens_cpu, paged_kernel_lens
                    )
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
                    # #629: min(seq, W + seq - prefix) needs BOTH host vectors;
                    # without either, no mirror is claimed.
                    paged_kernel_lens_cpu = _host_swa_paged_lens(
                        fresh_seq_lens_cpu, prefix_lens_cpu, sliding_window_size
                    )
                    paged_kernel_lens_sum = _host_sum_or_device(
                        paged_kernel_lens_cpu, paged_kernel_lens
                    )
                    kv_start_idx = seq_lens - paged_kernel_lens
            else:
                # full attention
                paged_kernel_lens = seq_lens
                paged_kernel_lens_sum = seq_lens_sum
                # #629: exact mirror, checked by seq_lens_sum downstream.
                paged_kernel_lens_cpu = fresh_seq_lens_cpu
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
                # #629: the second of the two PrefillWrapper paths left off the
                # #616h/#623 channel. Same two consequences as the cross-attn
                # twin: the weighted-DCP branch falls back to the unbounded
                # int(full_indptr[bs].item()), and fast_prefill_plan's
                # seq_lens_cpu assert has nothing to satisfy it.
                seq_lens_cpu=seq_lens_cpu,
                extend_prefix_lens_cpu=extend_prefix_lens_cpu,
                paged_kernel_lens_cpu=paged_kernel_lens_cpu,
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
                # #629: paged_kernel_lens IS seq_lens here, so seq_lens_cpu is
                # its exact mirror and seq_lens_sum the check on it -- via the
                # freshness guard, since a gpu_only batch's seq_lens_cpu is
                # stale rather than absent and would mis-size the buffer.
                paged_kernel_lens_cpu = dcp_fresh_host_lens(seq_lens_cpu, seq_lens_sum)
            else:
                # cross attention
                paged_kernel_lens = encoder_lens
                kv_start_idx = torch.zeros_like(encoder_lens)
                paged_kernel_lens_sum = paged_kernel_lens.sum().item()
                # #629: encoder_lens has no host mirror on the ForwardBatch, so
                # none is claimed -- None keeps the existing device read rather
                # than substituting a vector that mirrors something else.
                paged_kernel_lens_cpu = None

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
                # #629: this updater was one of the two PrefillWrapper paths
                # left off the #616h/#623 host-mirror channel. Beyond the
                # unbounded D2H in the weighted-DCP branch, call_begin_forward
                # ASSERTS seq_lens_cpu is not None once fast_prefill_plan is
                # installed, so an unforwarded mirror is a hard failure under a
                # captured prefill graph, not merely a slow path.
                seq_lens_cpu=seq_lens_cpu,
                extend_prefix_lens_cpu=extend_prefix_lens_cpu,
                paged_kernel_lens_cpu=paged_kernel_lens_cpu,
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
        # #616c: host mirror of prefix_lens. Optional and defaulted, so the
        # other callers of this method (sliding-window, cross-attention) keep
        # their current behaviour and simply fall back to the device read.
        extend_prefix_lens_cpu: Optional[List[int]] = None,
        # #623: host mirror of paged_kernel_lens -- a DIFFERENT vector from the
        # one above whenever use_ragged is False, which is precisely the case
        # the DCP spec branches run in. Same opt-in contract: only
        # update_single_wrapper supplies it, everyone else keeps the old read.
        paged_kernel_lens_cpu: Optional[Union[List[int], torch.Tensor]] = None,
    ):
        bs = len(seq_lens)
        # #623: host mirror of paged_kernel_lens, validated against the sum the
        # caller already computed for the same vector. None (-> keep the device
        # read) when no mirror was supplied or when the two disagree, which is
        # how a stale gpu_only seq_lens_cpu is refused instead of mis-sizing an
        # index buffer.
        # Gated on uneven_dcp so a non-DCP prefill does not even sum the mirror:
        # the default path keeps doing exactly what it did.
        host_paged_lens = (
            dcp_host_lens(paged_kernel_lens_cpu, expected_sum=paged_kernel_lens_sum)
            if self.attn_backend.uneven_dcp
            else None
        )
        # Draft->draft tree mask for the uneven-DCP ragged verify wrapper
        # (--speculative-eagle-topk > 1); stays None on every other path so the
        # ragged plan keeps its default (causal chain / non-causal extend).
        ragged_custom_mask = None
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
                    # #616c: kills the blocking D2H inside the collective
                    # window. prefix_lens IS forward_batch.extend_prefix_lens,
                    # so this host mirror is the same sum, not an estimate.
                    total_tokens=_dcp_host_total_tokens(extend_prefix_lens_cpu),
                )
            else:
                dcp_lens = get_dcp_lens(prefix_lens, dcp_size, dcp_rank, None)
                kv_indptr[1 : bs + 1] = torch.cumsum(dcp_lens, dim=0)
                kv_indptr = kv_indptr[: bs + 1]
                # #618: the even sibling of the weighted read above carries the
                # same unbounded D2H. The mirror is the extend one here, because
                # this branch indexes over prefix_lens.
                n_dcp = dcp_host_even_total(extend_prefix_lens_cpu, dcp_size, dcp_rank)
                kv_indices = torch.empty(
                    (n_dcp if n_dcp is not None else int(dcp_lens.sum().item())) + 256,
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
            # Stage B1 (weightless host spill): stash THIS rank's owned prefix
            # layout + whether any owned prefix slot is HOST-resident, so the
            # per-layer forward can stream the committed prefix blockwise
            # (_wl_blockwise_prefix_return_lse) instead of the monolithic paged
            # read (which would deref host slot ids as device slots). Reference
            # assignment only; dead when spill is off (default).
            if getattr(self.attn_backend, "_wl_spill_active", False):
                ab = self.attn_backend
                n_owned = int(kv_indptr[-1].item())
                ab._dcp_extend_owned_kv_indptr = kv_indptr
                ab._dcp_extend_owned_kv_indices = kv_indices
                ab._dcp_extend_qo_indptr = qo_indptr
                ab._dcp_extend_global_prefix_lens = prefix_lens
                ab._dcp_extend_has_host = bool(
                    (kv_indices[:n_owned] >= ab._wl_dev_slots).any().item()
                )
        elif (
            self.attn_backend.uneven_dcp
            and spec_info is not None
            and getattr(spec_info, "spec_input_type", None)
            == SpecInputType.EAGLE_DRAFT_EXTEND
        ):
            # DRAFT-EXTEND under a token-sharded DRAFT KV pool (#108 slice 2).
            #
            # Structurally the target-verify split below, with two differences,
            # which is exactly why it is its own branch and NOT a new member of
            # _DCP_VERIFY_SPEC_INPUT_TYPES:
            #
            #   1. THE PREFIX IS SHORTER THAN paged_kernel_lens. For verify,
            #      paged_kernel_lens IS the committed prefix (the draft tokens
            #      are not in seq_lens). For draft-extend, seq_lens ALREADY
            #      counts the num_tokens_per_req tokens this step appends --
            #      they are written into the pool by the owner-rule masked
            #      write at the top of the same forward. Reading the full
            #      seq_len here would let each query attend its OWN key through
            #      the non-causal paged stage as well as through the causal
            #      ragged stage, i.e. count it twice in the LSE merge. That is
            #      a wrong answer, not a crash, so the subtraction is the whole
            #      correctness content of this branch.
            #   2. The per-request query count is num_tokens_per_req, not
            #      draft_token_num.
            #
            # Everything else is shared with verify by construction: the same
            # weighted owner rule builds the owned-slot indices, the paged read
            # is non-causal, the ragged wrapper does the causal current-chunk
            # attention on LOCAL heads, and the cross-rank LSE merge combines
            # them. No new kernel and no new collective.
            #
            # The draft worker only reaches this branch when its pool really is
            # token-sharded: attn_backend.uneven_dcp is False for a draft runner
            # under the default --draft-kv-layout replicated (slice 1's
            # draft_pool_is_replicated predicate is the single source of that
            # decision, and nothing here re-derives it).
            num_tokens_per_req = int(getattr(spec_info, "num_tokens_per_req", 1) or 1)
            # Rank-uniform: seq_lens is replicated and num_tokens_per_req is a
            # constant of the batch (the draft-extend qo layout is a fixed
            # stride so it can be cuda-graph captured).
            dcp_prefix_lens = draft_extend_prefix_lens(
                paged_kernel_lens, num_tokens_per_req
            )
            # #623: the same subtraction on the host mirror. Running the SAME
            # function rather than re-deriving `- k` here is the point: the
            # clamp at zero is part of the length definition, and a host sum
            # that skipped it would over-size the index buffer on a request
            # whose whole sequence is this step's tokens.
            host_dcp_prefix_lens = (
                None
                if host_paged_lens is None
                else draft_extend_prefix_lens(host_paged_lens, num_tokens_per_req)
            )
            if self.attn_backend.uneven_dcp_weighted:
                kv_indptr, kv_indices = _build_dcp_weighted_kv_indices(
                    self.req_to_token,
                    req_pool_indices,
                    dcp_prefix_lens,
                    kv_indptr,
                    None,
                    self.attn_backend.cp_S,
                    self.attn_backend.cp_lo,
                    self.attn_backend.cp_hi,
                    self.attn_backend.cp_ratio,
                    pad=256,
                    # #623: kills the unbounded D2H on the draft-extend path.
                    total_tokens=dcp_host_total_tokens(host_dcp_prefix_lens),
                )
            else:
                dcp_size = self.attn_backend.dcp_size
                dcp_rank = self.attn_backend.dcp_rank
                dcp_lens = get_dcp_lens(dcp_prefix_lens, dcp_size, dcp_rank, None)
                kv_indptr[1 : bs + 1] = torch.cumsum(dcp_lens, dim=0)
                kv_indptr = kv_indptr[: bs + 1]
                # #618: even sibling, same host vector.
                n_dcp = dcp_host_even_total(host_dcp_prefix_lens, dcp_size, dcp_rank)
                kv_indices = torch.empty(
                    (n_dcp if n_dcp is not None else int(dcp_lens.sum().item())) + 256,
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
            # One query row per appended token, constant stride per request --
            # the layout the draft-extend graph captures.
            qo_indptr = torch.arange(
                0,
                (bs + 1) * num_tokens_per_req,
                step=num_tokens_per_req,
                dtype=torch.int32,
                device=req_pool_indices.device,
            )
            # Paged prefix read is non-causal (every appended query sees the
            # whole committed prefix); the causal masking among the appended
            # tokens is the ragged wrapper's job in the forward.
            custom_mask = None
            use_ragged = True
        elif (
            self.attn_backend.uneven_dcp
            and spec_info is not None
            and getattr(spec_info, "spec_input_type", None)
            in _DCP_VERIFY_SPEC_INPUT_TYPES
        ):
            # M4 (MTP+DCP): target-VERIFY under uneven-weighted DCP. Split the
            # verify into (a) the draft tokens attending the committed PREFIX
            # (paged, token-sharded owned slots, FULL replicated kv-heads,
            # non-causal, cross-rank LSE-combined) and (b) the draft tokens
            # attending EACH OTHER (ragged, LOCAL heads). With
            # --speculative-eagle-topk 1 the draft is a linear CHAIN, so the
            # draft->draft mask is plain causal and the ragged wrapper needs no
            # tree custom_mask. With topk > 1 the draft is a TREE: the
            # draft->draft block carries the tree-topology mask, sliced out of
            # spec_info.custom_mask (the EAGLE FULL_MASK) below and planned into
            # the ragged wrapper. That mask is draft-vs-draft ONLY -- a local,
            # rank-uniform property (every rank holds all draft-token
            # activations), orthogonal to the token-sharded prefix -- so it is
            # correct under DCP without any cross-rank coordination. The prefix
            # read excludes the draft tokens (paged_kernel_lens == committed
            # seq_lens) and stays NON-causal (never masked); the draft KV is
            # written to owned compact slots by _dcp_masked_write and attended
            # only through the ragged wrapper.
            use_ragged = True
            # DFLASH_VERIFY takes the SAME split as EAGLE/MTP verify. A DFLASH
            # draft block is a linear CHAIN (topk is fixed to 1, no tree), so
            # the draft->draft attention is plain causal ragged -- structurally
            # identical to the EAGLE topk==1 case, and the committed prefix is
            # read exactly the same way. Before this branch covered DFLASH, a
            # DFLASH verify under uneven DCP fell through to the generic spec
            # branch below, which leaves use_ragged False: the ragged wrapper is
            # then never planned, while _forward_extend_dcp unconditionally runs
            # the current-chunk ragged attention -> AttributeError
            # '_cached_q_data_type' on EVERY rank (the solo host merely lost the
            # race and was sigquit'd before it got there, which made this look
            # shadow-rank-specific).
            if spec_info.spec_input_type == SpecInputType.DFLASH_VERIFY:
                assert getattr(spec_info, "ragged_verify_layout", None) is None, (
                    "uneven DCP + DFLASH target-verify does not support the "
                    "ragged-verify layout (variable per-request verify lens); "
                    "the DCP split assumes a uniform draft_token_num per "
                    "request. Disable SGLANG_RAGGED_VERIFY_MODE."
                )
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
                    # #623: THE production wedge stack. The 02:02 crash and the
                    # 03:23 wedge dump both pinned owner.py's
                    # full_indptr[bs].item() reached from this file; the extend
                    # site was wired in #616h, this is the verify twin. The
                    # committed prefix IS paged_kernel_lens, so its host mirror
                    # is the exact same sum -- and NOT paged_kernel_lens_sum's
                    # cousin at the extend site, which sums seq_lens.
                    total_tokens=dcp_host_total_tokens(host_paged_lens),
                )
            else:
                dcp_size = self.attn_backend.dcp_size
                dcp_rank = self.attn_backend.dcp_rank
                dcp_lens = get_dcp_lens(paged_kernel_lens, dcp_size, dcp_rank, None)
                kv_indptr[1 : bs + 1] = torch.cumsum(dcp_lens, dim=0)
                kv_indptr = kv_indptr[: bs + 1]
                # #618: even sibling of the verify read.
                n_dcp = dcp_host_even_total(host_paged_lens, dcp_size, dcp_rank)
                kv_indices = torch.empty(
                    (n_dcp if n_dcp is not None else int(dcp_lens.sum().item())) + 256,
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
            # Tree-spec (topk > 1): slice the draft_num x draft_num tree block
            # out of the EAGLE FULL_MASK and hand it to the ragged wrapper so
            # its draft->draft attention uses the tree topology instead of a
            # plain causal chain. topk == 1 keeps ragged_custom_mask None (the
            # forward's causal=True path, byte-identical to before).
            if (
                self.attn_backend.dcp_tree_mask
                and getattr(spec_info, "custom_mask", None) is not None
            ):
                ragged_custom_mask = _build_dcp_ragged_tree_mask(
                    spec_info.custom_mask,
                    paged_kernel_lens,  # committed prefix lens (verify)
                    paged_kernel_lens_sum,
                    draft_num,
                    bs,
                )
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
                # Tree-spec only (topk > 1): flashinfer packs this bool mask into
                # the per-bucket ragged wrapper's custom_mask_buf on every replay
                # -> CUSTOM mask mode over the draft->draft block. None elsewhere.
                custom_mask=ragged_custom_mask,
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
        # Tagged as pauseable per-state scratch during adaptive offload
        # builds; rewritten by generate_draft_decode_kv_indices before every
        # replay (see the single-backend init_cuda_graph_state for details).
        with adaptive_graph_memory.tagged_state_alloc(
            nbytes=self.speculative_num_steps * kv_indices_width * 4
        ):
            self.cuda_graph_kv_indices = torch.zeros(
                (self.speculative_num_steps, kv_indices_width),
                dtype=torch.int32,
                device="cuda",
            )
        adaptive_graph_memory.note_state_tensor(self.cuda_graph_kv_indices)

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
