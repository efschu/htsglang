from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch

from sglang.kernels.ops.memory.common import (
    _get_last_loc_safe_kernel as _get_last_loc_safe_kernel,
)
from sglang.kernels.ops.memory.common import get_last_loc_kernel as get_last_loc_kernel
from sglang.kernels.ops.memory.common import (
    get_last_loc_triton,
    get_last_loc_triton_safe,
    write_req_to_token_pool_triton,
)
from sglang.srt.hardware_backend.npu.dsv4.dsv4_common_hooks import (
    maybe_evict_dsv4_state_on_swa,
    maybe_write_dsv4_decode,
    maybe_write_dsv4_extend,
)
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache, EvictParams
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool, ReqToTokenPool
from sglang.srt.runtime_context import get_server_args
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import is_cuda, is_hip, is_npu, support_triton
from sglang.srt.utils.common import ceil_align, is_pin_memory_available

_is_npu = is_npu()

_is_hip = is_hip()

_is_cuda = is_cuda()

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from sglang.srt.model_executor.forward_batch_info import DSV4StateLens

# Needs 2 + 1 slots for mamba request with prefix cache. 2 for ping pong cache, 1 for running mamba state.
MAMBA_STATE_PER_REQ_PREFIX_CACHE = 3
# Lazy mode: 1 + 1 slots (1 ping-pong + 1 running), second ping-pong allocated on demand at boundary.
MAMBA_STATE_PER_REQ_PREFIX_CACHE_LAZY = 2
MAMBA_STATE_PER_REQ_NO_CACHE = 1

logger = logging.getLogger(__name__)


def kv_to_page_indices(kv_indices: np.ndarray, page_size: int):
    # The page is guaranteed to be full except the last page.
    if page_size == 1:
        return kv_indices

    return kv_indices[::page_size] // page_size


def kv_to_page_num(num_kv_indices: int, page_size: int):
    return (num_kv_indices + page_size - 1) // page_size


def page_align_floor(length: int, page_size: int) -> int:
    return (length // page_size) * page_size


def free_swa_out_of_window_slots(
    req: Req,
    pre_len: int,
    *,
    sliding_window_size: int,
    page_size: int,
    req_to_token_pool: ReqToTokenPool,
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
    is_chunk_cache: bool = False,
) -> None:
    # For swa radix cache, we need to evict the tokens that are not in the tree cache and also not in the sliding window
    assert (
        req.cache_protected_len % page_size == 0
    ), "cache_protected_len must be page aligned"
    evict_floor = max(req.cache_protected_len, getattr(req, "swa_evict_floor", 0))
    if page_size > 1 and evict_floor > req.cache_protected_len:
        evict_floor = -(-evict_floor // page_size) * page_size
    req.swa_evicted_seqlen = max(req.swa_evicted_seqlen, evict_floor)

    if is_chunk_cache:
        # Chunk cache builds no radix tree, so no tombstone-leaf concern; evict
        # up to the window boundary (the trailing floor keeps it page-aligned).
        evict_threshold = pre_len - sliding_window_size
    else:
        # Radix cache: keep max(window, page). The trailing floor page-aligns the
        # frontier, and subtracting at least one page keeps it below the insert
        # boundary (page_floor(seq_len)) so the last leaf is never all-tombstone.
        # No extra page margin is needed.
        evict_threshold = pre_len - max(sliding_window_size, page_size)
    new_swa_evicted_seqlen = max(
        req.swa_evicted_seqlen,
        evict_threshold,
    )

    if page_size > 1:
        new_swa_evicted_seqlen = (new_swa_evicted_seqlen // page_size) * page_size

    if new_swa_evicted_seqlen > req.swa_evicted_seqlen:
        free_slots = req_to_token_pool.req_to_token[
            req.req_pool_idx, req.swa_evicted_seqlen : new_swa_evicted_seqlen
        ]
        token_to_kv_pool_allocator.free_swa(free_slots)
        maybe_evict_dsv4_state_on_swa(
            token_to_kv_pool_allocator, req_to_token_pool, req, new_swa_evicted_seqlen
        )
        req.swa_evicted_seqlen = new_swa_evicted_seqlen


def maybe_cache_unfinished_req(req: Req, tree_cache: BasePrefixCache, **kwargs):
    if getattr(req, "skip_radix_cache_insert", False):
        # Fake-bootstrap (warmup) requests must never INSERT into the prefix
        # cache -- but for a chunked prefill this call is ALSO the chunk ->
        # prefix conversion: `req.prefix_indices` must advance over the chunk
        # just computed, or `PrefillAdder.add_chunked_req` re-plans the SAME
        # first chunk forever while every pass allocates fresh KV rows, until
        # the token pool is exhausted (upstream bug, introduced by the
        # decode-side-radix PR that added this flag; task #106). Perform the
        # minimal ChunkCache-equivalent advance without touching the tree:
        # the rows stay request-owned and are freed at completion via
        # `cache_finished_req(is_insert=False)` (see `release_kv_cache`).
        kv_indices = tree_cache.req_to_token_pool.req_to_token[
            req.req_pool_idx, : req.extend_range.end
        ]
        req.prefix_indices = kv_indices.to(dtype=torch.int64, copy=True)
        return

    if getattr(req, "kv_spill_state", None) == "host":
        # kv-session-offload: this request's rows past `kv_spill_boundary` are
        # HOST SENTINELS, not device KV. Inserting them into the device radix
        # tree donates indices that address no device row: the tree's
        # `evictable_size` grows by tokens the pool does not own, and when the
        # session finishes and the tree lock drops, the accounting invariant
        # blows up ("pool memory leak detected", evictable > total).
        #
        # Measured on the mixed-GPU rig, identically on all three ranks, for a
        # PS2 born-spilled-DEEP session -- which holds NO device head at all
        # (boundary=0, every row a sentinel):
        #     D5DIAG unfinished  boundary=0 protected_before=0 extend_end=1967
        #     D5DIAG after       boundary=0 protected_after=1920
        # i.e. 1920 sentinel rows entered the tree. At completion that surfaced
        # as: evictable=4351 against total=3600, with the released session
        # reporting "protected=1920" against a host boundary of 994.
        #
        # The FINISH path already refuses the insert for exactly this reason
        # (see release_kv_cache below: "No radix insert -- the tail is on host,
        # there is no full device KV to donate"). This is the same rule at the
        # UNFINISHED seam, which was missing it. It also repairs the head free:
        # release_finished_spilled_req frees [cache_protected_len, boundary),
        # so a protected length inflated past the boundary made it free
        # NOTHING and leak the real device head too.
        #
        # The rows stay request-owned and are released by
        # release_finished_spilled_req. Advance prefix_indices without touching
        # the tree, exactly as the skip_radix_cache_insert branch above does,
        # so the chunk -> prefix bookkeeping is unchanged.
        kv_indices = tree_cache.req_to_token_pool.req_to_token[
            req.req_pool_idx, : req.extend_range.end
        ]
        req.prefix_indices = kv_indices.to(dtype=torch.int64, copy=True)
        return

    tree_cache.cache_unfinished_req(req, **kwargs)


def write_cache_indices(
    out_cache_loc: torch.Tensor,
    req_pool_indices_tensor: torch.Tensor,
    req_pool_indices_cpu: torch.Tensor,
    prefix_lens_tensor: torch.Tensor,
    prefix_lens_cpu: torch.Tensor,
    seq_lens_tensor: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    extend_lens_tensor: torch.Tensor,
    extend_lens_cpu: torch.Tensor,
    prefix_tensors: list[torch.Tensor],
    req_to_token_pool: ReqToTokenPool,
):
    if support_triton(get_server_args().attention_backend):
        prefix_pointers = torch.tensor(
            [t.data_ptr() for t in prefix_tensors],
            dtype=torch.uint64,
            pin_memory=is_pin_memory_available(req_to_token_pool.device),
        ).to(req_to_token_pool.device, non_blocking=True)
        # TODO: some tensors can be reused for ForwardBatchInfo (e.g., extend_lens, cumsum_start)
        write_req_to_token_pool_triton[(req_pool_indices_tensor.shape[0],)](
            req_to_token_pool.req_to_token,
            req_pool_indices_tensor,
            prefix_pointers,
            prefix_lens_tensor,
            seq_lens_tensor,
            extend_lens_tensor,
            out_cache_loc,
            req_to_token_pool.req_to_token.shape[1],
        )
    else:
        pt = 0
        for i in range(req_pool_indices_cpu.shape[0]):
            req_idx = req_pool_indices_cpu[i].item()
            prefix_len = prefix_lens_cpu[i].item()
            seq_len = seq_lens_cpu[i].item()
            extend_len = extend_lens_cpu[i].item()

            req_to_token_pool.write(
                (req_idx, slice(0, prefix_len)),
                prefix_tensors[i],
            )
            req_to_token_pool.write(
                (req_idx, slice(prefix_len, seq_len)),
                out_cache_loc[pt : pt + extend_len],
            )
            pt += extend_len


def get_last_loc(
    req_to_token: torch.Tensor,
    req_pool_indices_tensor: torch.Tensor,
    prefix_lens_tensor: torch.Tensor,
) -> torch.Tensor:
    attn_backend = get_server_args().attention_backend
    uses_triton_dispatch = attn_backend not in ("ascend", "torch_native")

    if _is_hip and uses_triton_dispatch:
        # HIP-only: the legacy get_last_loc_triton kernel emits a
        # mixed-width int32->int64 store that Triton mis-compiles on HIP,
        # producing out-of-range last_loc values under EAGLE +
        # page_size>1 (e.g. with aiter unified attention or the triton
        # attention backend). The bug is in the Triton HIP codegen, not
        # in any particular attention backend, so route every HIP path
        # that would otherwise use get_last_loc_triton through the
        # int32-safe variant. Non-HIP hardware keeps the original
        # dispatcher below.
        return get_last_loc_triton_safe(
            req_to_token, req_pool_indices_tensor, prefix_lens_tensor
        )

    if uses_triton_dispatch:
        impl = get_last_loc_triton
    else:
        impl = get_last_loc_torch

    return impl(req_to_token, req_pool_indices_tensor, prefix_lens_tensor)


def get_last_loc_torch(
    req_to_token: torch.Tensor,
    req_pool_indices_tensor: torch.Tensor,
    prefix_lens_tensor: torch.Tensor,
) -> torch.Tensor:
    return torch.where(
        prefix_lens_tensor > 0,
        req_to_token[req_pool_indices_tensor, prefix_lens_tensor - 1],
        torch.full_like(prefix_lens_tensor, -1),
    )


def get_alloc_len_per_decode(server_args: Optional[ServerArgs] = None) -> int:
    if server_args is None:
        server_args = get_server_args()

    if server_args.speculative_algorithm is None:
        return 1

    # Spec decoding allocates max(topk * num_steps, num_draft_tokens) per decode step.
    spec_steps = server_args.speculative_num_steps or 1
    spec_topk = server_args.speculative_eagle_topk or 1
    spec_tokens = server_args.max_speculative_num_draft_tokens
    page_size = server_args.page_size

    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    spec_algo = SpeculativeAlgorithm.from_string(server_args.speculative_algorithm)
    if page_size == 1 or spec_topk == 1 or not spec_algo.has_draft_kv():
        return max(spec_steps * spec_topk, spec_tokens)
    else:
        # spec v2 tree (page>1, topk>1): worst-case page-aligned footprint per
        # topk branch is ceil((page_size-1 + num_steps) / page) pages, each branch
        # duplicated -- reserve for all topk branches.
        num_new_pages_per_topk = (
            (page_size - 1) + spec_steps + page_size - 1
        ) // page_size
        return max(num_new_pages_per_topk * page_size * spec_topk, spec_tokens)


def get_commit_lag_per_decode(server_args: Optional[ServerArgs] = None) -> int:
    """Tokens a single in-flight forward can still commit behind the host's view.

    This is the L term of ``get_alloc_reserve_per_decode``; see the derivation
    there. Under overlap the scheduler prepares iteration i's allocation before
    iteration i-1's result has been processed, so ``req.kv_committed_len`` is one
    verify stale. One verify commits at most ``max_speculative_num_draft_tokens``
    tokens (batch_result_processor: ``kv_committed_len += len(accept_tokens)``
    with ``accept_tokens = next_token_ids[i*stride : i*stride + accept_lens[i]]``,
    ``stride == speculative_num_draft_tokens``), or exactly 1 without spec.

    ``max_speculative_num_draft_tokens`` -- not the currently active width -- is
    the right ceiling: under adaptive spec or --speculative-cross-algorithm the
    in-flight step may have run a WIDER rung than the one now being prepared.
    """
    if server_args is None:
        server_args = get_server_args()

    if server_args.disable_overlap_schedule:
        # No overlap: the previous result is processed before the next
        # prepare_for_decode, so kv_committed_len is exact and there is no lag.
        return 0

    if server_args.speculative_algorithm is None:
        return 1

    return server_args.max_speculative_num_draft_tokens or 1


def get_alloc_reserve_per_decode(
    server_args: Optional[ServerArgs] = None,
    alloc_len: Optional[int] = None,
) -> int:
    """KV length reserved per request ahead of ``kv_committed_len`` each decode step.

    DERIVATION (task #486). The reserve must satisfy, for the forward about to be
    launched::

        kv_allocated_len >= device_seq_len + W

    where ``device_seq_len`` is where the kernels start writing and ``W`` is how
    far they write. The host only knows ``kv_committed_len``, so the reserve is
    the sum of two independent terms -- it is NOT a multiplier on either one:

      W (write footprint) = get_alloc_len_per_decode()
          The draft write is anchored at ``batch.seq_lens`` and spans
          ``topk * num_steps`` slots (base_spec_worker.py, ``out_cache_loc =
          torch.empty((bs * topk * num_steps,))`` / the paged tree branch's
          ``num_new_pages * page_size`` per branch); the verify write spans
          ``num_draft_tokens``. Their max is exactly get_alloc_len_per_decode.

      L (commit lag)      = get_commit_lag_per_decode()
          ``device_seq_len - kv_committed_len`` at prepare time, bounded by one
          verify's accept run. See that helper.

    So ``reserve = W + L``. Correctness first: this is an upper bound on the true
    need, never below it, and it must never be shaved -- see
    ``test/registered/spec/test_alloc_reserve_need.py::...under_reservation...``.

    Why this is not the previous blanket ``2 * get_alloc_len_per_decode()``:
    that form is W + W. It coincides with W + L on the configurations where
    ``W == L`` -- no-spec (1+1) and the chain-draft topk=1 case where
    ``num_draft_tokens >= num_steps`` (our NEXTN production path, W = L = D) --
    and over-reserves by ``W - L`` everywhere else: topk>1 (by
    ``topk*num_steps - D``), the page>1 topk>1 tree (by
    ``num_new_pages*page_size*topk - D``, the largest case), and every
    non-overlap run (by the whole of W, since L collapses to 0).

    Not upstream #32574's fix. That PR drops the second term entirely (1x) on
    the premise that ``batch.seq_lens_cpu`` is "perfectly synchronous" with the
    device. In this tree it is not: ``FutureMap.resolve_seq_lens_cpu`` runs
    inside ``Scheduler.run_batch``, i.e. AFTER ``prepare_for_decode`` in the same
    event-loop iteration (scheduler.py ``event_loop_overlap``), so at prepare
    time ``batch.seq_lens_cpu`` carries the same one-verify staleness as
    ``kv_committed_len`` -- and it is ``None`` outright whenever
    ``decide_needs_cpu_seq_lens`` opted the backend out of the D2H mirror.
    Adopting 1x here would under-reserve by L and let the verify write past
    ``kv_allocated_len``.

    ``alloc_len`` overrides the W term for lanes that verify a fixed block whose
    width is not ``get_alloc_len_per_decode`` (DFLASH solo: one block per step).
    """
    if server_args is None:
        server_args = get_server_args()

    write_footprint = (
        get_alloc_len_per_decode(server_args) if alloc_len is None else alloc_len
    )
    commit_lag = get_commit_lag_per_decode(server_args)
    return write_footprint + commit_lag


def get_req_to_token_extra_context_len(server_args: ServerArgs) -> int:
    """req_to_token row headroom beyond the model context length.

    Sized to hold the decode over-allocation; the spec v2 page>1 topk>1 holey
    draft footprint can outgrow the default num_draft_tokens headroom.
    """
    # FIXME(lsyin): temporary fix for the context length issue under spec decoding
    extra = 4 + (server_args.max_speculative_num_draft_tokens or 0)
    if (
        server_args.speculative_algorithm is not None
        and server_args.page_size > 1
        and (server_args.speculative_eagle_topk or 1) > 1
    ):
        extra = max(extra, get_alloc_reserve_per_decode(server_args))
    return extra


def alloc_token_slots(
    tree_cache: BasePrefixCache,
    num_tokens: int,
    backup_state: bool = False,
):
    allocator = tree_cache.token_to_kv_pool_allocator
    evict_from_tree_cache(tree_cache, num_tokens)

    state = None
    if backup_state:
        state = allocator.backup_state()

    out_cache_loc = allocator.alloc(num_tokens)

    if out_cache_loc is None:
        error_msg = (
            f"Out of memory. Try to lower your batch size.\n"
            f"Try to allocate {num_tokens} tokens.\n"
            f"{available_and_evictable_str(tree_cache)}"
        )
        logger.error(error_msg)
        if tree_cache is not None:
            tree_cache.pretty_print()
        raise RuntimeError(error_msg)

    return (out_cache_loc, state) if backup_state else out_cache_loc


def evict_from_tree_cache(tree_cache: BasePrefixCache | None, num_tokens: int):
    if tree_cache is None:
        return

    if tree_cache.is_chunk_cache():
        return

    allocator = tree_cache.token_to_kv_pool_allocator

    if isinstance(allocator, SWATokenToKVPoolAllocator):
        # Hybrid allocator
        full_available_size = allocator.full_available_size()
        swa_available_size = allocator.swa_available_size()

        if full_available_size < num_tokens or swa_available_size < num_tokens:
            full_num_tokens = max(0, num_tokens - full_available_size)
            swa_num_tokens = max(0, num_tokens - swa_available_size)
            tree_cache.evict(
                EvictParams(num_tokens=full_num_tokens, swa_num_tokens=swa_num_tokens)
            )
    else:
        # Standard allocator
        if allocator.available_size() < num_tokens:
            tree_cache.evict(EvictParams(num_tokens=num_tokens))


def _compute_dsv4_state_lens(batch, *, is_decode: bool):
    """Per-req c{4,128}_state pool alloc lens (``DSV4StateLens``) for this step.
    None on CUDA / non-V4 paths (allocator has no ``compute_dsv4_state_lens_*``).
    """
    allocator = batch.token_to_kv_pool_allocator
    if not hasattr(allocator, "compute_dsv4_state_lens_extend"):
        return None
    if is_decode:
        return allocator.compute_dsv4_state_lens_decode(batch.reqs)
    return allocator.compute_dsv4_state_lens_extend(
        batch.reqs, batch.seq_lens_cpu.tolist()
    )


def alloc_paged_token_slots_extend(
    tree_cache: BasePrefixCache,
    prefix_lens: torch.Tensor,
    prefix_lens_cpu: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    last_loc: torch.Tensor,
    extend_num_tokens: int,
    backup_state: bool = False,
    req_pool_indices: Optional[torch.Tensor] = None,
    dsv4_state_lens: Optional[DSV4StateLens] = None,
    batch=None,
):
    # Over estimate the number of tokens: assume each request needs a new page.
    allocator = tree_cache.token_to_kv_pool_allocator
    num_tokens = extend_num_tokens + len(seq_lens_cpu) * allocator.page_size
    evict_from_tree_cache(tree_cache, num_tokens)

    state = None
    if backup_state:
        state = allocator.backup_state()

    is_dsv4 = req_pool_indices is not None and hasattr(allocator, "c4_attn_allocator")
    extra_alloc_kwargs = {}
    if is_dsv4:
        extra_alloc_kwargs["req_pool_indices"] = req_pool_indices
        # Per-call per-req tables for the c-pool / state last_loc lookup.
        if batch is not None:
            extra_alloc_kwargs["req_to_token_pool"] = batch.req_to_token_pool
        if dsv4_state_lens is not None:
            extra_alloc_kwargs["dsv4_state_lens"] = dsv4_state_lens

    out = allocator.alloc_extend(
        prefix_lens,
        prefix_lens_cpu,
        seq_lens,
        seq_lens_cpu,
        last_loc,
        extend_num_tokens,
        **extra_alloc_kwargs,
    )

    if is_dsv4:
        bundle = out
        out_cache_loc = None if bundle is None else bundle.out_full_loc
        if batch is not None:
            batch.out_cache_loc_dsv4 = bundle
    else:
        out_cache_loc = out

    if out_cache_loc is None:
        error_msg = (
            f"Prefill out of memory. Try to lower your batch size.\n"
            f"Try to allocate {extend_num_tokens} tokens.\n"
            f"{available_and_evictable_str(tree_cache)}"
        )
        logger.error(error_msg)
        if tree_cache is not None:
            tree_cache.pretty_print()
        raise RuntimeError(error_msg)

    return (out_cache_loc, state) if backup_state else out_cache_loc


def alloc_req_slots(
    req_to_token_pool: ReqToTokenPool,
    reqs: list[Req],
    tree_cache: BasePrefixCache | None,
) -> list[int]:
    """Allocate request slots from the pool.

    Fail-loud: raises ``RuntimeError`` if the pool can't satisfy the batch. An
    alloc failure here means the admission budget (``PrefillAdder``) was wrong
    and should surface rather than be masked.
    """
    num_reqs = len(reqs)
    if isinstance(req_to_token_pool, HybridReqToTokenPool):
        # Byte-coordinated for the shared allocator (accounts for the peer full
        # sub-pool's bytes); plain slot free count for the non-shared one.
        mamba_available_size = (
            req_to_token_pool.mamba_allocator.schedulable_available_size()
        )
        # Eviction headroom factor: 3x (or lazy variant) for radix COW, 1x for chunk.
        if tree_cache.supports_mamba():
            factor = (
                MAMBA_STATE_PER_REQ_PREFIX_CACHE_LAZY
                if req_to_token_pool.enable_mamba_extra_buffer_lazy
                else MAMBA_STATE_PER_REQ_PREFIX_CACHE
            )
        else:
            factor = MAMBA_STATE_PER_REQ_NO_CACHE
        mamba_state_needed = num_reqs * factor
        if mamba_available_size < mamba_state_needed:
            if tree_cache is not None and tree_cache.supports_mamba():
                mamba_num = max(0, mamba_state_needed - mamba_available_size)
                tree_cache.evict(EvictParams(num_tokens=0, mamba_num=mamba_num))
    req_pool_indices = req_to_token_pool.alloc(reqs)
    if req_pool_indices is None:
        raise RuntimeError(
            "alloc_req_slots runs out of memory. "
            "Please set a smaller number for `--max-running-requests`. "
            f"{req_to_token_pool.available_size()=}, {num_reqs=}, "
        )
    return req_pool_indices


def _alloc_page_size(batch: ScheduleBatch) -> int:
    # DCP swaps in an allocator whose page_size is server_args.page_size *
    # dcp_size, so it can be > 1 even when tree_cache.page_size is 1; branch on
    # the real allocator's page_size there. Elsewhere the two are equal.
    if (_is_hip or _is_cuda) and get_server_args().dcp_size > 1:
        return batch.tree_cache.token_to_kv_pool_allocator.page_size
    return batch.tree_cache.page_size


def alloc_for_extend(
    batch: ScheduleBatch,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Allocate KV cache for extend batch and write to req_to_token_pool.

    Returns ``(out_cache_loc, req_pool_indices_device, req_pool_indices_cpu)``
    (the last is the host/CPU mirror). ``alloc_req_slots`` raises ``RuntimeError``
    if the pool can't satisfy the batch (fail-loud — see its docstring).
    """
    # free out-of-window swa tokens
    batch.maybe_evict_swa()

    prefix_tensors = [r.prefix_indices for r in batch.reqs]

    # Create tensors for allocation
    prefix_lens_cpu = torch.tensor(batch.prefix_lens, dtype=torch.int64)
    extend_lens_cpu = torch.tensor(batch.extend_lens, dtype=torch.int64)
    prefix_lens_device = prefix_lens_cpu.to(batch.device, non_blocking=True)
    extend_lens_device = extend_lens_cpu.to(batch.device, non_blocking=True)

    # Allocate req slots (raises RuntimeError if the pool is exhausted)
    req_pool_indices = alloc_req_slots(
        batch.req_to_token_pool, batch.reqs, batch.tree_cache
    )
    req_pool_indices_cpu = torch.tensor(req_pool_indices, dtype=torch.int64)
    req_pool_indices_device = req_pool_indices_cpu.to(batch.device, non_blocking=True)

    # Allocate KV cache (throws exception on failure)
    if _alloc_page_size(batch) == 1:
        out_cache_loc = alloc_token_slots(batch.tree_cache, batch.extend_num_tokens)
    else:
        # Paged allocation - build last_loc
        last_loc = [
            (t[-1:] if len(t) > 0 else torch.tensor([-1], device=batch.device))
            for t in prefix_tensors
        ]
        out_cache_loc = alloc_paged_token_slots_extend(
            tree_cache=batch.tree_cache,
            prefix_lens=prefix_lens_device,
            prefix_lens_cpu=prefix_lens_cpu,
            seq_lens=batch.seq_lens,
            seq_lens_cpu=batch.seq_lens_cpu,
            last_loc=torch.cat(last_loc),
            extend_num_tokens=batch.extend_num_tokens,
            req_pool_indices=req_pool_indices_device,
            dsv4_state_lens=_compute_dsv4_state_lens(batch, is_decode=False),
            batch=batch,
        )

    # Write to req_to_token_pool
    write_cache_indices(
        out_cache_loc,
        req_pool_indices_device,
        req_pool_indices_cpu,
        prefix_lens_device,
        prefix_lens_cpu,
        batch.seq_lens,
        batch.seq_lens_cpu,
        extend_lens_device,
        extend_lens_cpu,
        prefix_tensors,
        batch.req_to_token_pool,
    )

    # DSV4-NPU hook: no-op on non-DSV4 paths.
    if _is_npu:
        maybe_write_dsv4_extend(
            batch,
            req_pool_indices_cpu,
            prefix_lens_cpu,
            batch.seq_lens_cpu,
        )

    return out_cache_loc, req_pool_indices_device, req_pool_indices_cpu


def alloc_paged_token_slots_decode(
    tree_cache: BasePrefixCache,
    seq_lens: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    last_loc: torch.Tensor,
    token_per_req: int = 1,
    req_pool_indices: Optional[torch.Tensor] = None,
    dsv4_state_lens: Optional[DSV4StateLens] = None,
    batch=None,
) -> torch.Tensor:
    """Allocate paged KV cache for decode batch."""
    allocator = tree_cache.token_to_kv_pool_allocator
    # Over estimate the number of tokens: assume each request needs a new page.
    num_tokens = len(seq_lens) * allocator.page_size
    evict_from_tree_cache(tree_cache, num_tokens)

    # DSV4-NPU allocator also needs req_pool_indices + per-req state lens and
    # returns a DSV4OutCacheLoc bundle; hasattr-gated so others stay unchanged.
    is_dsv4 = req_pool_indices is not None and hasattr(allocator, "c4_attn_allocator")
    extra_alloc_kwargs = {}
    if is_dsv4:
        extra_alloc_kwargs["req_pool_indices"] = req_pool_indices
        # Per-call per-req tables for the last_loc lookup.
        if batch is not None:
            extra_alloc_kwargs["req_to_token_pool"] = batch.req_to_token_pool
        if dsv4_state_lens is not None:
            extra_alloc_kwargs["dsv4_state_lens"] = dsv4_state_lens

    out = allocator.alloc_decode(seq_lens, seq_lens_cpu, last_loc, **extra_alloc_kwargs)

    if is_dsv4:
        bundle = out
        out_cache_loc = None if bundle is None else bundle.out_full_loc
        if batch is not None:
            batch.out_cache_loc_dsv4 = bundle
    else:
        out_cache_loc = out

    if out_cache_loc is None:
        error_msg = (
            f"Decode out of memory. Try to lower your batch size.\n"
            f"Try to allocate {len(seq_lens) * token_per_req} tokens.\n"
            f"{available_and_evictable_str(tree_cache)}"
        )
        logger.error(error_msg)
        if tree_cache is not None:
            tree_cache.pretty_print()
        raise RuntimeError(error_msg)

    return out_cache_loc


def alloc_for_decode(batch: ScheduleBatch, token_per_req: int) -> torch.Tensor:
    """
    Allocate KV cache for decode batch and write to req_to_token_pool.

    Returns:
        out_cache_loc: allocated cache locations
    """

    batch.maybe_evict_swa()

    seq_lens_gpu = batch.seq_lens
    bs = seq_lens_gpu.shape[0]

    if _alloc_page_size(batch) == 1:
        # Non-paged allocation
        out_cache_loc = alloc_token_slots(batch.tree_cache, bs * token_per_req)
    else:
        # Paged allocation
        last_loc = batch.req_to_token_pool.req_to_token[
            batch.req_pool_indices, seq_lens_gpu - 1
        ]
        seq_lens_next = seq_lens_gpu + token_per_req
        out_cache_loc = alloc_paged_token_slots_decode(
            tree_cache=batch.tree_cache,
            seq_lens=seq_lens_next,
            seq_lens_cpu=batch.seq_lens_cpu + token_per_req,
            last_loc=last_loc,
            token_per_req=token_per_req,
            req_pool_indices=batch.req_pool_indices,
            dsv4_state_lens=_compute_dsv4_state_lens(batch, is_decode=True),
            batch=batch,
        )

    # Write to req_to_token_pool
    if batch.model_config.is_encoder_decoder:
        locs = batch.encoder_lens + seq_lens_gpu
    else:
        locs = seq_lens_gpu.clone()

    batch.req_to_token_pool.write(
        (batch.req_pool_indices, locs), out_cache_loc.to(torch.int32)
    )

    # DSV4-NPU hook: no-op on non-DSV4 paths.
    if _is_npu:
        maybe_write_dsv4_decode(
            batch,
            batch.seq_lens_cpu + token_per_req,
            token_per_req,
        )

    return out_cache_loc


def release_kv_cache(req: Req, tree_cache: BasePrefixCache, is_insert: bool = True):
    # MambaRadixCache may alloc mamba state before alloc KV cache
    if req.req_pool_idx is None:
        assert (
            tree_cache.supports_mamba()
        ), "Only MambaRadixCache allow freeing before alloc"
        # TODO (csy, hanming): clean up this early allocation logic
        if req.mamba_pool_idx is not None:
            tree_cache.req_to_token_pool.mamba_allocator.free(
                req.mamba_pool_idx.unsqueeze(-1)
            )
            req.mamba_pool_idx = None
        return

    # kv-session-offload: a SPILLED session's req_to_token row holds host
    # SENTINELS in its tail (values >= host_base, NOT allocator slots). The
    # stock cache_finished_req / allocator.free below would run
    # torch.unique(free_index // page_size) over those sentinel values -> CUDA
    # illegal memory access (the spill x stock-retraction crash). Route to the
    # spill manager's release, which frees the retained device HEAD + tree
    # lock + Mamba + req slot + host region and NEVER touches the sentinel
    # tail -- the exact cleanup the finish-on-host path uses. Reached only via
    # retract / abort here (the finish path already routes to
    # release_finished_spilled_req directly before this function). The
    # subsequent reset_for_retract (retract caller) is safe: it resets logical
    # state and does not touch req_pool_idx. Device (non-spilled) reqs have
    # kv_spill_state None -> stock path unchanged, byte-identical.
    if getattr(req, "kv_spill_state", None) == "host":
        from sglang.srt.managers.kv_session_offload import (
            get_kv_session_offload_manager,
        )

        get_kv_session_offload_manager().release_finished_spilled_req(req)
        return

    tree_cache.cache_finished_req(
        req,
        is_insert=is_insert and not getattr(req, "skip_radix_cache_insert", False),
    )

    # StreamingSession.cache_finished_req handles speculative tail trim
    # and bookkeeping flag sync internally, then sets req_pool_idx = None.
    if req.req_pool_idx is None:
        return

    start_p, end_p = req.pop_overallocated_kv_cache()

    global_server_args = get_server_args()
    page_size = global_server_args.page_size
    spec_algo = global_server_args.speculative_algorithm

    # strip_thinking_cache intentionally reports output tokens as overallocated
    # so they fall into the free path below (#22373).
    if spec_algo is None and not global_server_args.strip_thinking_cache:
        assert (
            start_p == end_p
        ), f"Unexpected overallocated KV cache, {req.kv_committed_len=}, {req.kv_allocated_len=}"

    if page_size > 1:
        start_p = ceil_align(start_p, page_size)

    if start_p < end_p:
        indices_to_free = tree_cache.req_to_token_pool.req_to_token[req.req_pool_idx][
            start_p:end_p
        ]
        tree_cache.token_to_kv_pool_allocator.free(indices_to_free)
    # If the prefix cache doesn't manage mamba states, we must free them here.
    if isinstance(tree_cache.req_to_token_pool, HybridReqToTokenPool) and (
        not tree_cache.supports_mamba()
    ):
        assert (
            req.mamba_pool_idx is not None
        ), "mamba state is freed while the tree cache does not manage mamba states"
        tree_cache.req_to_token_pool.free_mamba_cache(req)
    # DSV4-NPU's free() also releases c4/c128 state pages; no-op for others.
    tree_cache.req_to_token_pool.free(req)


def available_and_evictable_str(tree_cache: BasePrefixCache) -> str:
    return tree_cache.available_and_evictable_str()
