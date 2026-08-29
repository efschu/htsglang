from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Callable, List, Optional

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
    assert req.cache_protected_len % page_size == 0, (
        "cache_protected_len must be page aligned"
    )
    evict_floor = max(req.cache_protected_len, req.swa_evict_floor)
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

    if req.kv_spill_state == "host":
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


def payable_size(allocator) -> int:
    """Tokens the pool can hand out once its open free group is applied.

    #790. THE DELIVERY MEASURE, and it is deliberately not
    ``available_size()``. Two different mechanisms in this tree hold freed
    tokens outside the pool's published availability, and they need opposite
    verdicts:

    * an open free group (#681 third root) holds tokens the caller CAN still
      reach, through ``flush_free_group``. Counting them here keeps a batching
      window from reading as an under-delivery it is not.
    * a residency cap (``KvBackingRelief``'s ``KvRowCap``) holds tokens the
      caller CANNOT reach at all: it subscribes to the allocator's free
      listener and pulls every freed id above its cap straight back out of the
      free list. Those are in neither term, which is exactly right -- they were
      never delivered.

    Never raises: this feeds a diagnostic and a cold-path relief rung, and a
    measurement that dies takes the real allocation error with it.
    """
    try:
        payable = int(allocator.available_size())
    except Exception:  # noqa: BLE001 - a measurement must not raise here
        return 0
    for chunk in getattr(allocator, "free_group", None) or ():
        try:
            payable += int(chunk.numel())
        except Exception:  # noqa: BLE001 - an exotic staging entry counts as 0
            continue
    return payable


def _residency_withheld_note(allocator) -> str:
    """One clause naming the confiscator, or ''. #790.

    ``KvRowCap`` publishes what it holds out of circulation as
    ``residency_withheld_slots`` (kv_backing_relief.py, ``_publish``) so the
    scheduler's idle invariant does not read it as a leak. The same number is
    the answer to "where did the tokens the tree just freed go", so the error
    that reports the under-delivery names it.
    """
    withheld = getattr(allocator, "residency_withheld_slots", 0)
    try:
        withheld = int(withheld)
    except Exception:  # noqa: BLE001 - a diagnostic must not raise
        return ""
    if withheld <= 0:
        return ""
    return (
        f" A RESIDENCY CAP IS ENGAGED and is holding {withheld} slot ids out "
        f"of the allocator's free list: every freed id above the cap is taken "
        f"straight back by the cap's free listener, so peeling nodes whose "
        f"slots live above it frees the TREE and pays the POOL nothing."
    )


def _eviction_shortfall_note(tree_cache, asked: int, evicted: int) -> str:
    """One clause naming the promise-versus-delivery gap, or ''.

    #681. The allocation error used to report ``available + evictable`` and
    nothing else, so an operator read "66039 tokens available" beside "failed
    to allocate 512" and had no way to see that the eviction in between had
    under-delivered. Naming it here is the difference between a confusing
    message and a diagnosis.

    #790: ``evicted`` is what the POOL RECEIVED, not what the tree counted.
    The caller measures it with :func:`payable_size` across the eviction,
    because the two numbers came apart on the 2026-08-21 01:54:55 specimen: the
    tree freed its full 512 tokens, a residency cap confiscated every one of
    them at the allocator's free listener, and this note -- fed the tree's own
    receipt -- returned '' on the one failure it exists to explain.
    """
    if evicted >= asked:
        return ""
    #: `evictable_size()` RAISES NotImplementedError on MambaRadixCache and
    #: SWARadixCache -- both split the count in two and say so -- and that is
    #: the class this note was written for, so the original single call
    #: reported -1 on exactly the boot it was meant to explain. Ask for the
    #: full-attention count first and fall back only for the flat classes.
    evictable = -1
    for name in ("full_evictable_size", "evictable_size"):
        getter = getattr(tree_cache, name, None)
        if getter is None:
            continue
        try:
            evictable = int(getter())
            break
        except Exception:  # noqa: BLE001 - a diagnostic must not raise
            continue
    allocator = getattr(tree_cache, "token_to_kv_pool_allocator", None)
    return (
        f"\nEVICTION UNDER-DELIVERED: asked for {asked} tokens, the pool "
        f"received {evicted}. The tree still reports {evictable} evictable "
        f"tokens.{_residency_withheld_note(allocator)} Since #681 the frontier "
        f"can pay every unlocked leaf it selects (mamba tombstone leaves "
        f"included), so a shortfall here is NOT the tree failing to find "
        f"victims -- it is something taking the freed slots between the tree "
        f"and the pool. With no confiscator named above, treat it as a "
        f"REGRESSION SIGNAL: a new class of node is being counted that the "
        f"peel cannot consume."
    )


def _flush_deferred_frees(allocator) -> int:
    """Apply frees the allocator has staged in an open batching group.

    #681 third root. Returns tokens applied, 0 when there is nothing staged or
    the allocator has no group protocol.

    A CAPABILITY PROBE, NOT A DEFENSIVE DEFAULT. ``flush_free_group`` is
    defined on ``BaseTokenToKVPoolAllocator``, so every allocator in the tree
    has it; the check is here because this module is also driven by test
    stand-ins and by allocators that stub the group protocol out entirely
    (``allocator/hisparse.py`` makes begin/end no-ops). A missing method means
    "this allocator never stages", which is a real answer -- unlike a missing
    MEASUREMENT, which is the case #606 says must never be defaulted.
    """
    flush = getattr(allocator, "flush_free_group", None)
    if flush is None:
        return 0
    try:
        return int(flush() or 0)
    except Exception as exc:  # noqa: BLE001 - a relief step must not mask the OOM
        logger.warning(
            "#681: flushing the allocator's staged frees raised (%s); the "
            "allocation continues to the relief ladder and, failing that, to "
            "the original error.",
            exc,
        )
        return 0


#: #790: how many further peels the confiscation rung may take. Each round
#: that pays NOTHING doubles its ask, so the ladder 512 -> 1024 -> ... -> 65536
#: walks a fully confiscated region in a handful of rounds instead of one
#: chunk at a time; a round that pays goes back to asking for the exact
#: remainder, so the rung never evicts more cache than the shortfall needs.
_CONFISCATION_PEEL_ROUNDS = 8


def _evict_past_confiscation(tree_cache, allocator, shortfall: int) -> int:
    """Peel until the POOL has received ``shortfall`` tokens (#790).

    Returns tokens DELIVERED -- the growth of :func:`payable_size` -- which is
    the only number this rung may believe. The tree's own receipt is what
    failed: ``KvRowCap`` (managers/kv_backing_relief.py) registers a free
    listener and pulls every freed id above its cap back out of the free list,
    so ``FullComponent.evict_component`` frees a leaf, counts its 512 tokens,
    and the pool's availability does not move.

    WHY PEELING AGAIN IS THE RIGHT ANSWER, not a wider admission gate. The
    confiscated ids are a REGION of the id space, not the whole of it: on the
    2026-08-21 specimen the cap sat at row 137135 of 161792, so ~114k evictable
    tokens below the cap were payable the whole time and the peel stopped
    before reaching one of them -- because it was told it had already been
    paid. An admission bound computed from the same wrong receipt would have
    admitted the batch too.

    GROUP-UNIFORM BY REFUSAL, the precedent being ``uniform_host_floor_active``.
    ``shortfall`` is rank-local, so the number of rounds this takes is
    rank-local, and under an active ``uniform_avail_floor`` -- the #616g state
    where the radix trees are replicas that must peel identically -- a
    rank-local peel is exactly the divergence that floor exists to prevent. So
    with a floor in force this rung declines and the original error raises:
    fail-loud beats a wedge.
    """
    if getattr(tree_cache, "uniform_avail_floor", None) is not None:
        return 0
    delivered = 0
    ask = max(1, int(shortfall))
    for _ in range(_CONFISCATION_PEEL_ROUNDS):
        before = payable_size(allocator)
        try:
            counted = int(evict_from_tree_cache(tree_cache, ask) or 0)
        except Exception as exc:  # noqa: BLE001 - a relief rung must not mask the OOM
            logger.warning("#790: the confiscation peel raised (%s); stopping", exc)
            break
        delivered += max(0, payable_size(allocator) - before)
        if delivered >= shortfall or counted <= 0:
            # ``counted <= 0`` is the frontier saying it has nothing left to
            # give: peeling again would be the same no-op forever.
            break
        ask = (shortfall - delivered) if delivered else ask * 2
    return delivered


def alloc_token_slots(
    tree_cache: BasePrefixCache,
    num_tokens: int,
    backup_state: bool = False,
):
    allocator = tree_cache.token_to_kv_pool_allocator
    # #790: MEASURE THE DELIVERY, DO NOT ASK THE TREE. ``evict_from_tree_cache``
    # returns the tree's own count of what it freed, and that count is true
    # about the TREE and false about the POOL whenever something intercepts the
    # free -- which is what a residency cap does, on every free, by design.
    payable_before = payable_size(allocator)
    evict_from_tree_cache(tree_cache, num_tokens)
    delivered = max(0, payable_size(allocator) - payable_before)

    state = None
    if backup_state:
        state = allocator.backup_state()

    out_cache_loc = allocator.alloc(num_tokens)

    if out_cache_loc is None:
        # #679: ONE BOUNDED RELIEF, ONE RETRY, THEN THE ORIGINAL ERROR.
        #
        # Measured 2026-08-15 23:41:01: available 0, evictable 0, and this
        # raise killed all three ranks at once -- "terminate called without an
        # active exception" as the death rattle. The only relief attempted at
        # this site is ``evict_from_tree_cache`` above, which with nothing
        # evictable is a guaranteed no-op that returns no signal, so the raise
        # was reached with no other path having been tried at all.
        #
        # THIS IS A NET, NOT THE GUARANTEE. The fix that prevents the state is
        # on the admission side, where the decision can be group-uniform; by
        # the time execution is here the group has already committed to a
        # batch. Providers are therefore rank-local by contract -- see
        # register_extend_relief_provider -- and a failure still raises, so
        # fail-loud remains the last word rather than being softened into a
        # silent stall.
        # #681 THIRD ROOT, AND IT IS TRIED FIRST BECAUSE IT IS NOT RELIEF.
        #
        # The rungs below give something up -- host bandwidth, a victim's
        # decode progress. This gives up nothing: it applies frees the tree has
        # ALREADY performed and already counted, which are sitting in
        # ``allocator.free_group`` because a batching window was open when the
        # eviction ran (batch_result_processor.py:92 and :741 open one inside
        # the event loop). Measured 2026-08-16 13:58:37 on all three ranks:
        # eviction reported its full 512 tokens, ``available_size`` stayed at
        # 392, and no under-delivery note printed -- because by the tree's
        # books the eviction HAD delivered. The pages were payable the whole
        # time; nothing had asked for them.
        #
        # Cold path only: this runs after an allocation has already failed, so
        # a healthy alloc pays one list check less than nothing.
        staged = _flush_deferred_frees(allocator)
        if staged > 0:
            if backup_state:
                # The snapshot above predates the flush, so rolling back to it
                # would drop the pages the flush just applied. No caller passes
                # backup_state on this path today; re-taking it keeps that a
                # fact about the callers rather than a dependency of this fix.
                state = allocator.backup_state()
            out_cache_loc = allocator.alloc(num_tokens)
            logger.warning(
                "extend allocation of %d tokens failed with %d tokens already "
                "freed but still staged in the allocator's batching group; "
                "applying them %s. This is #681's third root: an eviction "
                "inside a free-group window counts tokens the pool cannot yet "
                "hand out.",
                num_tokens,
                staged,
                "SUCCEEDED" if out_cache_loc is not None else "did not help",
            )

    if out_cache_loc is None and delivered < num_tokens:
        # #790 SECOND ROOT, AND IT IS TRIED BEFORE THE RELIEF LADDER BECAUSE IT
        # IS STILL ONLY CACHE. The rungs below give up a victim's decode
        # progress; this gives up more recomputable prefix, which is what the
        # eviction three lines up was already spending.
        #
        # Measured 2026-08-21 01:54:55 on PP0, 4m55s after a tp_to_pp flip:
        # eviction reported its full 512 tokens, ``full_available_size`` stayed
        # at 189, no under-delivery note printed -- because by the tree's books
        # the eviction HAD delivered -- and the scheduler died on a pool
        # reporting 138089 available tokens. The flip had carried 160822 live
        # slots back across the whole 161792-id space while the #631 backing
        # rung's row cap was still engaged at row 137135, so the leaves the
        # peel selected were exactly the ones the cap confiscates.
        gained = _evict_past_confiscation(tree_cache, allocator, num_tokens - delivered)
        if gained > 0:
            delivered += gained
            # The peel may have landed inside an open batching window; the
            # flush above already ran, so apply anything it staged now.
            _flush_deferred_frees(allocator)
            if backup_state:
                # Same reason as the flush rung: the snapshot predates these
                # frees and rolling back to it would drop them.
                state = allocator.backup_state()
            out_cache_loc = allocator.alloc(num_tokens)
            logger.warning(
                "extend allocation of %d tokens failed after an eviction the "
                "pool did not receive; peeling further delivered %d tokens and "
                "the retry %s. This is #790: the tree's eviction receipt counts "
                "slots a residency cap takes straight back out of the free "
                "list, so the peel stopped believing it had been paid.",
                num_tokens,
                gained,
                "SUCCEEDED" if out_cache_loc is not None else "did not help",
            )

    if out_cache_loc is None:
        freed = _attempt_extend_relief(num_tokens)
        if freed > 0:
            out_cache_loc = allocator.alloc(num_tokens)
            logger.warning(
                "extend allocation of %d tokens failed; rank-local relief "
                "returned %d tokens and the retry %s. Admission should have "
                "prevented this -- treat a recurring line here as an "
                "admission defect, not as relief working.",
                num_tokens,
                freed,
                "SUCCEEDED" if out_cache_loc is not None else "still failed",
            )

    if out_cache_loc is None:
        error_msg = (
            f"Out of memory. Try to lower your batch size.\n"
            f"Try to allocate {num_tokens} tokens.\n"
            f"{available_and_evictable_str(tree_cache)}"
            f"{_eviction_shortfall_note(tree_cache, num_tokens, delivered)}"
        )
        logger.error(error_msg)
        if tree_cache is not None:
            tree_cache.pretty_print()
        raise RuntimeError(error_msg)

    # #694: charge this draw against the published floor. Reached only on
    # success, so a failed allocation never inflates the ledger. Without this
    # the floor stays at its publish-time value all iteration, the evict
    # trigger reads it, skips, and the NEXT allocation raises on a tree full of
    # evictable tokens -- the two 2026-08-16 specimens.
    note_uniform_admitted(tree_cache, num_tokens)

    return (out_cache_loc, state) if backup_state else out_cache_loc


def fundable_extend_tokens(tree_cache) -> int:
    """Tokens an extend allocation could actually get RIGHT NOW, group-uniform.

    #679. The admission budget (``PrefillAdder.rem_total_tokens``) reads this
    rank's own ``available_size() + evictable_size()``. Under uneven DCP those
    differ per rank, so a BRANCH taken on them can split the group -- the
    rank-local-test-before-a-collective family this tree keeps paying for. The
    availability term therefore comes from ``uniform_avail_for_evict``, the
    same group-published floor eviction already decides on, so every rank
    reaches the same verdict on the same iteration.

    Evictable is added because eviction is exactly what ``alloc_token_slots``
    attempts before allocating: tokens the radix tree can give back are
    genuinely fundable. What is NOT counted is anything a relief provider might
    produce later -- that is a hope, and this number is used to decide whether
    work may be admitted now.
    """
    if tree_cache is None:
        return 0
    allocator = getattr(tree_cache, "token_to_kv_pool_allocator", None)
    if allocator is None:
        return 0
    try:
        avail = int(uniform_avail_for_evict(tree_cache, allocator))
    except Exception:  # noqa: BLE001 - an admission gate must not raise
        return 0
    try:
        evictable = int(tree_cache.evictable_size())
    except Exception:  # noqa: BLE001 - a cache without the accessor evicts none
        evictable = 0
    return max(0, avail) + max(0, evictable)


def published_fundable_floor(tree_cache) -> Optional[int]:
    """`fundable_extend_tokens`, but ONLY when a group floor was published.

    #681. The chunked gate one function up may safely read a 0 from
    `fundable_extend_tokens`: 0 means "park this chunk and retry next round",
    a self-clearing state. The NEW-request gate uses the same number as a
    BUDGET CEILING, and there a spurious 0 is not a park -- it admits nothing,
    for every request, on every subsequent round. A read failure would wedge
    the instance harder than the crash this ticket is about.

    `fundable_extend_tokens` cannot distinguish "the pool really is empty"
    from "the pool could not be read": every failure path returns 0. So the
    ceiling is applied only when the scheduler actually PUBLISHED a group
    floor -- i.e. the ranks' pools are uneven, which is the one state the
    ceiling exists for. With no floor (single rank, or pools that agree) the
    local budget is already the group's budget, the ceiling adds nothing, and
    returning None keeps the reference boot byte-identical.
    """
    if tree_cache is None:
        return None
    if getattr(tree_cache, "uniform_avail_floor", None) is None:
        return None
    return fundable_extend_tokens(tree_cache)


#: #679 rung 1-3: may an admission spend RELIEF before it parks?
#:
#: OFF BY DEFAULT, and the default is the whole compatibility argument: with
#: this unset the ladder returns 0 immediately and admission behaves exactly as
#: c4b88e1923 did, which is the boot currently serving. The ladder changes how
#: much the pool can fund; it never admits or refuses anything itself.
ENV_ADMISSION_RELIEF_LADDER = "SGLANG_ADMISSION_RELIEF_LADDER"

#: #679 rung 3 SEPARATELY, because it is the only rung that destroys progress.
#: A victim loses its whole decode and re-prefills, so an operator may want
#: rungs 1-2 (spill and throttle, which cost bandwidth and latency) without
#: rung 3. Requires the ladder itself to be on.
ENV_ADMISSION_RETRACTION = "SGLANG_ADMISSION_RELIEF_RETRACT"


def admission_relief_ladder_enabled() -> bool:
    return os.environ.get(ENV_ADMISSION_RELIEF_LADDER, "0") not in (
        "0",
        "",
        "false",
        "False",
    )


def admission_retraction_enabled() -> bool:
    """Rung 3 needs BOTH flags. A retraction rung reachable while the ladder
    is off would be a second admission-side actuator nobody asked for."""
    if not admission_relief_ladder_enabled():
        return False
    return os.environ.get(ENV_ADMISSION_RETRACTION, "0") not in (
        "0",
        "",
        "false",
        "False",
    )


def chunk_tokens_the_pool_can_fund(
    fundable_tokens: int, page_size: int, rem_chunk_tokens: int
) -> int:
    """How many tokens a chunked prefill may schedule. ZERO means PARK.

    #679, extracted as a pure function because the decision it makes is the one
    that killed the instance and a decision worth a post-mortem is worth a
    falsifier that does not need a scheduler to run.

    Below one page nothing useful can be allocated, so the chunk is parked and
    retried when memory frees. Above it the chunk takes what the pool can
    actually give rather than the nominal size -- the old code took the nominal
    size even when the pool reported nothing at all.
    """
    fundable = max(0, int(fundable_tokens))
    page = max(1, int(page_size))
    if fundable < page:
        return 0
    return max(0, min(int(rem_chunk_tokens), fundable))


#: #679: rank-local relief consulted when an extend allocation is about to fail.
#:
#: THE SHAPE IS ``_mem_create_reclaiming``'s (mem_cache/kv_vmm_backing.py): one
#: bounded reclaim, one retry, then the original error propagates unchanged.
#: That is the only "catch OOM, reclaim, retry" precedent in this tree and it
#: is deliberately followed rather than reinvented.
#:
#: RANK-LOCAL PROVIDERS ONLY, and the restriction is the whole safety argument.
#: This runs deep inside ``prepare_for_extend``, after the group has committed
#: to a batch; a provider that took a collective here would hang the first time
#: one rank reached it and its peers did not. Anything needing agreement
#: belongs on the admission path, which decides from a group-published floor.
_extend_relief_providers: List[Callable[[int], int]] = []

#: Emitted once per process when the alloc site is reached with an empty
#: registry, so "no relief existed" is a fact in the log rather than an
#: inference from its absence.
_announced_empty_relief = False


def register_extend_relief_provider(fn: Callable[[int], int]) -> None:
    """Register a RANK-LOCAL provider: ``fn(num_tokens) -> tokens freed``."""
    if fn not in _extend_relief_providers:
        _extend_relief_providers.append(fn)


def clear_extend_relief_providers() -> None:
    """Test seam, and the reason it exists is that a global registry which
    cannot be emptied makes every test after the first one a liar."""
    global _announced_empty_relief
    _extend_relief_providers.clear()
    _announced_empty_relief = False


def _attempt_extend_relief(num_tokens: int) -> int:
    """Ask every registered provider for ``num_tokens``. Never raises.

    A provider that raises is a provider that failed, not an instance that
    should die: the caller's next step is a re-raise of the real allocation
    error, which is strictly more informative than a relief bug's traceback.
    """
    if not _extend_relief_providers:
        # SAY THAT THE NET IS EMPTY, ONCE. Nothing registers a provider today:
        # the rank-local reliefs that could pay here (eviction) are already
        # spent by the time this runs, and the ones that could genuinely free
        # tokens -- retraction, session spill -- are collective and belong on
        # the admission path, not inside a batch the group has committed to.
        #
        # So this registry is a SEAM, and a seam that quietly does nothing is
        # the "present but inert" failure this tree keeps finding. An operator
        # reading a crash log must be able to see that no relief was available
        # rather than assume some was tried.
        global _announced_empty_relief
        if not _announced_empty_relief:
            _announced_empty_relief = True
            logger.warning(
                "extend allocation failed and NO relief provider is "
                "registered: the alloc-site net is empty on this boot, so the "
                "guarantee against this crash is entirely the admission guard "
                "(chunk_tokens_the_pool_can_fund). If you are reading this "
                "line, admission let through work the pool could not fund."
            )
        return 0
    freed = 0
    for fn in tuple(_extend_relief_providers):
        try:
            freed += max(0, int(fn(int(num_tokens))))
        except Exception as e:  # noqa: BLE001 - relief must not mask the OOM
            logger.warning("extend relief provider %r failed: %s", fn, e)
    return freed


def _ledger_tokens(value) -> int:
    """Read a per-iteration ledger attribute safely. THE NAMED LIST (#624).

    Both floor correctors -- the DEVICE one in
    :func:`uniform_avail_for_evict` and the HOST one in
    :func:`uniform_host_avail_for_backup` -- read a ledger attribute off the
    tree cache. Real caches declare it on ``BasePrefixCache`` with a 0 default;
    a test double usually does not, and ``int(Mock())`` is **1**, not 0. So an
    unconfigured stand-in silently shaved one token off whichever floor it fed:
    measured as ``499 != 500`` against the device ledger
    (test_a_published_floor_is_returned) and reproducible against the host one.

    Third appearance of the stub-drift class, so the guard lives in ONE place
    rather than as two isinstance checks that would drift apart. ``bool`` is
    excluded deliberately: it is an ``int`` subclass, and a truthy flag landing
    here would charge 1 token for the same reason a Mock did.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return int(value)


def uniform_avail_for_evict(tree_cache, allocator) -> int:
    """The availability a cache-mutation trigger must decide from (#616g).

    The scheduler publishes ``uniform_avail_floor`` once per iteration when the
    ranks' pools are uneven; it is the group MIN of ``available_size()``, so
    every rank evicts on the same predicate and the radix replicas stay
    identical. When it is None -- single rank, or pools that agree -- this is
    the live local value and the caller behaves exactly as it did before.
    """
    #: #694: CHARGE ALLOCATIONS AGAINST THE FLOOR, exactly as the HOST sibling
    #: below already does for backups. The floor is published ONCE per
    #: iteration (scheduler.py:4142-4144) as the group MIN at that instant;
    #: allocations made later in the same iteration are not reflected in it, so
    #: late in an iteration it is stale-OPTIMISTIC. With ``floor >= num_tokens``
    #: the eviction is SKIPPED entirely, the alloc then fails against the live
    #: pool, and the raise reports a tree full of evictable tokens nothing ever
    #: asked for -- the two 2026-08-16 specimens, 512 refused at ~167k
    #: evictable, both scheduler deaths.
    #:
    #: SUFFICIENT, by the same argument #645 makes for the host: this rank's
    #: live availability is at least ``avail_at_publish - admitted``, and
    #: ``avail_at_publish >= floor``, so a request clearing ``floor - admitted``
    #: fits the real pool too. RANK-UNIFORM by construction: ``num_tokens``
    #: comes from the replicated batch, so every rank charges the same amount at
    #: the same allocation and the predicate stays identical across ranks --
    #: which is the #616g invariant this must not break.
    floor = getattr(tree_cache, "uniform_avail_floor", None)
    if floor is None:
        return int(allocator.available_size())
    admitted = _ledger_tokens(getattr(tree_cache, "uniform_admitted_since_floor", 0))
    return max(0, int(floor) - admitted)


def uniform_host_avail_for_backup(tree_cache, mem_pool_host) -> int:
    """The HOST availability a write-through backup must decide from (#639).

    The device-side sibling one function up pins the two triggers that mutate
    the DEVICE tree. This pins the one that mutates whether a node has a HOST
    copy at all -- which under ``write_through`` decides whether the node
    SURVIVES its device eviction:
    ``UnifiedRadixCache._evict_device_leaf`` demotes a backed-up node (it
    stays in the tree, matchable and loadable back) and DELETES one without a
    backup. So a rank-local backup verdict is a rank-local tree edit, one tier
    below the #616g pins and upstream of them.

    Deciding it from this rank's own host pool is what produced the four
    2026-08-06/07 wedges: the host pools are 359652 / 287722 / 273336 slots,
    the node length is replicated, so the roomy rank backed up a node its
    peers refused, kept a prefix they deleted, matched longer, and entered
    every per-layer TP all_reduce of the next extend with a smaller token
    axis (rank 0 at 912/914/828/1690 tokens against peers' 2048/2048/1818).

    The scheduler publishes ``uniform_host_avail_floor`` once per iteration,
    from the reduce it already performs, when the ranks' HOST pools are
    uneven. None -- single rank, pools that agree, or no host tier at all --
    is the live local value and the caller behaves exactly as before.
    """
    #: #645: the floor is published ONCE per iteration, at the top of
    #: ``_update_uniform_pool_budget``, but the backup that reads it runs at
    #: the END of the iteration (``process_batch_result_prefill`` ->
    #: ``cache_unfinished_req`` -> ``insert`` -> ``write_backup``). Every
    #: backup admitted in between has already spent host slots that the
    #: published number still counts as free, so a stale floor over-admits:
    #: the rank whose pool IS the floor runs out for real while its peers,
    #: reading the same optimistic number, do not.
    #:
    #: Charging admissions against the floor removes the staleness without a
    #: second collective. The ledger is rank-uniform BY CONSTRUCTION: it is
    #: incremented only on this gate's own admissions, the gate compares two
    #: replicated numbers, and the node length is replicated -- so every rank
    #: charges the same amount at the same insert. And it is SUFFICIENT: this
    #: rank's live availability is at least ``avail_at_publish - admitted``,
    #: and ``avail_at_publish >= floor``, so a node that clears
    #: ``floor - admitted`` fits in the real pool too.
    floor = getattr(tree_cache, "uniform_host_avail_floor", None)
    if floor is None:
        return int(mem_pool_host.available_size())
    #: #624 (third appearance): guarded through the shared reader for the same
    #: reason the device sibling is -- an unconfigured double yields a Mock and
    #: ``int(Mock())`` is 1, which moved this floor too (reproduced at 499 vs
    #: 500 before the guard).
    admitted = _ledger_tokens(
        getattr(tree_cache, "uniform_host_admitted_since_floor", 0)
    )
    return max(0, int(floor) - admitted)


def uniform_host_floor_active(tree_cache) -> bool:
    """Whether a rank-uniform HOST floor is in force on this rank (#645).

    The backup path may only take its rank-LOCAL host-eviction branch when
    this is False. Under an active floor that branch is the wedge: #639 made
    the trigger ``host_avail < kv_tokens`` replicated, so every rank enters
    the eviction together, but what happens inside stays rank-local in two
    ways that both edit the tree.

    WHICH nodes are deleted. ``evict_host``'s victims are H-leaves, and
    ``_is_host_leaf`` requires ``node.evicted`` -- device-evicted. The device
    pools are rank-sized (190400 / 143840 / 143906), so each rank offers a
    different candidate set and ``_evict_host_leaf`` removes different nodes
    from different trees (``_remove_leaf_from_parent``). The rank that keeps
    a chunk node its peers deleted matches one chunk further: the 13:26
    specimen, rank 0 [22014] against peers [19967], difference 2047.

    WHETHER the eviction covered ``needed``. A rank with few H-leaves cannot
    raise the tokens and returns 0, so its node gets no backup -- and under
    ``write_through`` an un-backed-up node is DELETED at its next device
    eviction while a backed-up one is demoted and stays matchable. That rank
    loses the NEW node instead: the 12:15 specimen, rank 0 [2047] against
    peers [10238]. Both specimens, opposite directions, one branch.

    A floor cannot repair a SELECTION the way it repairs an admission: both
    sides of an admission compare can be made replicated, but no arithmetic
    on a published scalar makes two ranks pick the same nodes out of two
    different candidate sets. Doing that needs a group-agreed victim list,
    i.e. a new collective on the hot path. So the backup path refuses
    uniformly instead, at the price of a saturated host tier that no longer
    recycles itself on an uneven rig -- a throughput regression traded
    knowingly for a correctness one. Both directions are reproduced against
    the real ``write_backup`` in
    ``test/registered/unit/distributed/test_uniform_host_evict_floor_645.py``.
    """
    return getattr(tree_cache, "uniform_host_avail_floor", None) is not None


def note_uniform_admitted(tree_cache, tokens: int) -> None:
    """Charge an admitted DEVICE allocation against the published floor (#694).

    The device twin of :func:`note_uniform_host_admitted`. Only meaningful while
    a floor is active -- with no floor the attribute is never read, so charging
    is a no-op and the no-floor path stays byte-identical. Reset once per
    iteration by the scheduler when it publishes the next floor, so the ledger
    never outlives the number it corrects.
    """
    if getattr(tree_cache, "uniform_avail_floor", None) is None:
        return
    current = getattr(tree_cache, "uniform_admitted_since_floor", 0)
    tree_cache.uniform_admitted_since_floor = current + int(tokens)


def note_uniform_host_admitted(tree_cache, tokens: int) -> None:
    """Charge an admitted host backup against the published floor (#645).

    Called only on the success path of ``write_backup``, and only meaningful
    while a floor is active -- with no floor the attribute is never read.
    Reset once per iteration by the scheduler when it publishes the next
    floor, so the ledger never outlives the number it corrects.
    """
    if getattr(tree_cache, "uniform_host_avail_floor", None) is None:
        return
    current = getattr(tree_cache, "uniform_host_admitted_since_floor", 0)
    tree_cache.uniform_host_admitted_since_floor = current + int(tokens)


def note_uniform_host_refusal(tree_cache) -> int:
    """Count a refusal that skipped a rank-local host eviction (#645).

    Returns this floor generation's refusal count, so the caller can log the
    FIRST one and stay quiet for the rest. The condition should be rare --
    the ledger makes it arithmetically impossible for the backups this gate
    itself admitted -- but "should be rare" is not a reason to put an
    unthrottled warning on a path that runs at every insert, least of all
    one that fires hardest exactly when a rig is already in trouble.
    """
    count = getattr(tree_cache, "uniform_host_refusals_since_floor", 0) + 1
    tree_cache.uniform_host_refusals_since_floor = count
    return count


def uniform_mamba_avail_for_evict(tree_cache, local_avail: int) -> int:
    """The MAMBA-slot availability an eviction trigger must decide from (#639b).

    The two siblings above pin the KV token axis, device and host. This pins
    the third pool, and it is the one still open: the mamba slot pool is
    evicted by ``self.cache.evict(EvictParams(num_tokens=0, mamba_num=1))``
    fired from ``alloc(1) is None`` -- a rank-LOCAL test with no floor at all.

    Why a uniform pool SIZE does not make it uniform. ``max_mamba_cache_size``
    is min-reduced across ranks at startup, and a comment in
    ``MambaRadixCache._alloc_mamba_slot`` concluded from that the path is
    "rank-uniform without a collective". Size is not occupancy. WHICH node
    gets tombstoned is chosen by ``MambaComponent.drive_eviction`` from a
    rank-local LRU (``lru.get_lru_no_lock()``, which skips locked nodes), so
    two ranks with identically sized pools at different occupancy tombstone
    different nodes -- or one tombstones and the other does not.

    And the tombstone is a tree edit that the matcher reads.
    ``MambaComponent.evict_component`` sets ``cd.value = None`` for the mamba
    component only; the node keeps its KV and stays in the tree, but
    ``create_match_validator`` now refuses it. ``_match_prefix_helper``'s
    ``_all_valid`` advances the match only while EVERY component is resident,
    so the evicting rank's match stops at that node while its peer walks past
    -- which is the 2026-08-07 07:45 and 10:04 signature exactly
    (rank 0 sum=19711, rank 1 sum=16957, one request each).

    It compounds: a shorter match extends more tokens, which takes more
    slots, which forces more eviction, which shortens the next match. That is
    how the gap reached 2754 tokens.

    The floor is the group MIN of the mamba allocator's ``available_size()``,
    taken from the reduce the scheduler already runs once per iteration --
    NO new collective. None -- single rank, occupancy that agrees, or no
    mamba pool -- is the live local value and the caller behaves exactly as
    before.

    WHY ``min`` AND NOT A BARE SWAP, unlike the two siblings. They each govern
    exactly one call site reading exactly one quantity, so returning the floor
    outright is already <= the local value. This one governs sites reading TWO
    different quantities: ``alloc_req_slots`` compares
    ``schedulable_available_size()`` while the ``alloc(1) is None`` sites
    compare ``available_size()``, and on the shared
    ``UnifiedMambaSlotAllocator`` the schedulable view is the LARGER of the
    two (it credits the peer's drainable holes). Handing a schedulable-quantity
    caller a bare ``available_size``-derived floor would be direction-safe, but
    handing an ``available_size`` caller a schedulable-derived one would not.
    ``min`` is safe for either reading and needs no per-call-site coupling to
    the allocator class.

    UNIFORMITY IS EXACT, not approximate: the floor is the group MIN of
    ``available_size()``, and every caller's local quantity is >= its own
    ``available_size()`` >= that MIN, so when the floor is published
    ``min(local, floor) == floor`` on EVERY rank. When it is not published the
    ranks' values already agree. Either way every rank decides from the same
    number.

    DIRECTION IS SAFE BY CONSTRUCTION: ``min(local, floor) <= local``, so a
    rank evicts at least as often, and at least as much, as it did before the
    fix. Under-eviction -- the only way this pin could itself become a fault
    -- is arithmetically impossible. The price is that the rank with the
    roomiest mamba pool drops checkpoints it did not personally need, which is
    what keeping the replicas identical costs.
    """
    floor = getattr(tree_cache, "uniform_mamba_avail_floor", None)
    if floor is None:
        return int(local_avail)
    return min(int(local_avail), int(floor))


def peer_needs_mamba_evict(tree_cache, need: int = 1) -> bool:
    """Whether a PEER is out of mamba slots though this rank is not (#639b).

    The companion to the floor above, for the "allocate, and evict only if
    that failed" sites (``_alloc_mamba_slot`` and the two COW paths). Those
    cannot simply read the floor before allocating: ``MambaSlotAllocator.alloc``
    serves ``alloc(1)`` from the ``alloc_group_begin`` pre-allocation
    (``_alloc_iter``) when one is open, and those slots are already OUT of
    ``free_slots``. A pre-check on ``available_size()`` would therefore evict
    on a boot where the old code allocated straight from the iterator without
    evicting -- a behaviour change on the DEFAULT path, which is exactly what
    these pins are not allowed to cost.

    So the local fast path is left exactly as it was, and this answers the
    only question it cannot: "my alloc succeeded, must I evict anyway because
    a peer's did not?" Returns False whenever the floor is unpublished --
    single rank, or occupancy that agrees -- so a boot that cannot diverge
    takes literally the pre-#639b path.

    The evict that follows a True keeps the tombstone COUNT equal across the
    group: the dry rank evicts because its own alloc failed, this rank evicts
    because the group's minimum says the dry rank did. Same count, same
    replicated LRU, same tombstone set.

    ONE NAMED GAP, deliberately not closed here. ``_alloc_int8_ckpt_slot``
    (both cache classes) also fires a ``mamba_num=1`` eviction, but it decides
    it from the int8 CHECKPOINT pool -- a different pool with a different size,
    which this floor does not describe. Pinning it needs its own ``(x, -x)``
    pair in the scheduler's reduce, keyed on
    ``req_to_token_pool.mamba_ckpt_pool``. It is left open because that pool
    only exists under ``--enable-int8-mamba-checkpoint``, which is opt-in and
    off on the crashing deployment, so closing it here would add an unpinnable
    reduce term to every boot to cover a path none of them take. Named rather
    than silent, on the same principle as the kv-session-offload host-floor gap
    in ``Scheduler._update_uniform_pool_budget``.
    """
    floor = getattr(tree_cache, "uniform_mamba_avail_floor", None)
    if floor is None:
        return False
    return int(floor) < int(need)


def evict_from_tree_cache(tree_cache: BasePrefixCache | None, num_tokens: int) -> int:
    """Evict toward ``num_tokens``. Returns tokens ACTUALLY evicted (#681).

    The return value is the whole point: ``evictable_size_`` is a COUNT of
    unlocked tokens in the tree, while ``evict`` can only reach the leaf
    frontier. When those disagree the caller must know, because the next thing
    it does is allocate on the strength of the count.
    """
    if tree_cache is None:
        return 0

    if tree_cache.is_chunk_cache():
        return 0

    allocator = tree_cache.token_to_kv_pool_allocator

    if isinstance(allocator, SWATokenToKVPoolAllocator):
        # Hybrid allocator
        full_available_size = allocator.full_available_size()
        swa_available_size = allocator.swa_available_size()

        if full_available_size < num_tokens or swa_available_size < num_tokens:
            full_num_tokens = max(0, num_tokens - full_available_size)
            swa_num_tokens = max(0, num_tokens - swa_available_size)
            result = tree_cache.evict(
                EvictParams(num_tokens=full_num_tokens, swa_num_tokens=swa_num_tokens)
            )
            return int(getattr(result, "num_tokens_evicted", 0) or 0)
    else:
        # Standard allocator
        #
        # #616g: RANK-UNIFORM trigger. `num_tokens` is replicated (it comes
        # from the batch, e.g. `batch.extend_num_tokens`); `available_size()`
        # is this rank's own pool shard and differs across ranks under uneven
        # TP/DCP. Deciding from the local side makes the roomy rank skip an
        # eviction the tight ranks take, the radix trees stop being replicas,
        # and the next `match_prefix` hands back a rank-dependent prefix ->
        # rank-dependent `extend_num_tokens` -> TP collectives entered with
        # mismatched shapes (the #616 BAR1 stall). The floor is the group MIN
        # of exactly this quantity, published once per iteration by the
        # scheduler from a reduce that already ran. None => pools agree (or
        # single rank) => the live local value, unchanged.
        avail = uniform_avail_for_evict(tree_cache, allocator)
        if avail < num_tokens:
            # #681: READ THE RECEIPT. ``evict`` returns how many tokens it
            # ACTUALLY freed, and this call site threw that away -- so an
            # eviction that under-delivered was indistinguishable from one that
            # worked, and the allocation three lines later raised with a
            # message reporting plenty of memory.
            #
            # Measured 2026-08-16 01:46:10, all three ranks identically:
            #   Try to allocate 512 tokens.
            #   Available full tokens: 66039 (available=273 + evictable=65766)
            # 512 needed, 65766 reported evictable, allocation failed anyway.
            # #681 FOLLOW-UP, CORRECTING THIS COMMENT'S FIRST VERSION: the gap
            # was NOT "tokens behind a locked chain". Locking walks a node to
            # the ROOT and `_split_node` copies the ref onto the new upper half,
            # so an unlocked node can never have a locked descendant and the
            # peel always reaches it -- pinned by
            # test_evictable_reachability_681.TestLockRefsAreAncestorClosed.
            # The real gap was at the other end of the frontier: an unlocked
            # MAMBA TOMBSTONE LEAF was counted, was selected first, and made
            # `_evict_leaf_node` ASSERT rather than under-deliver. That is now
            # paid in `MambaRadixCache.evict_full`, so the receipt below should
            # always equal the request.
            result = tree_cache.evict(EvictParams(num_tokens=num_tokens))
            return int(getattr(result, "num_tokens_evicted", 0) or 0)
    return 0


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

    def _attempt_alloc():
        """One allocation attempt, including the DSV4 bundle unwrap.

        A closure so the RETRY below is the same call as the first attempt --
        the previous shape duplicated neither, and that is exactly how the
        retry came to be missing.
        """
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
            if batch is not None:
                batch.out_cache_loc_dsv4 = bundle
            return None if bundle is None else bundle.out_full_loc
        return out

    out_cache_loc = _attempt_alloc()

    if out_cache_loc is None:
        # #715 / #681 THIRD ROOT, AND IT IS TRIED FIRST BECAUSE IT IS NOT
        # RELIEF -- the same ordering, for the same reason, as in
        # alloc_token_slots.
        #
        # RULE 3 was applied to this path for the relief NET but not for the
        # third root, so a paged prefill reached its raise without ever asking
        # whether the pages it needed were already freed and merely staged.
        # That is the 2026-08-17 02:18 crash: 512 tokens refused with 147,456
        # counted evictable, because the eviction ran inside a free-group
        # window (batch_result_processor.py:92 and :741) and honestly reported
        # full delivery while available_size could not yet see the pages.
        #
        # Costs nothing: it applies frees the tree has ALREADY performed and
        # already counted. Cold path only -- reached after an allocation has
        # failed, so a healthy alloc pays one list check less than nothing.
        staged = _flush_deferred_frees(allocator)
        if staged > 0:
            if backup_state:
                # The snapshot above predates the flush; re-take it so a
                # rollback cannot drop the pages the flush just applied.
                state = allocator.backup_state()
            out_cache_loc = _attempt_alloc()
            logger.warning(
                "paged extend allocation of %d tokens failed with %d tokens "
                "already freed but still staged in the allocator's batching "
                "group; applying them %s. This is #681's third root on the "
                "paged path: an eviction inside a free-group window counts "
                "tokens the pool cannot yet hand out.",
                extend_num_tokens,
                staged,
                "SUCCEEDED" if out_cache_loc is not None else "did not help",
            )

    if out_cache_loc is None:
        # #681 RULE 3: every alloc path reachable from prefill admission gets
        # the same net. This is the page_size > 1 twin of alloc_token_slots and
        # it had none -- the audit found three raise sites on this path
        # (alloc_req_slots, alloc_token_slots, this one) and only one covered.
        #
        # #681 REMAINDER: and having asked for relief, SPEND IT. This path used
        # to consult the provider, log that relief had SUCCEEDED, and then fall
        # through to the raise without retrying -- the net was cast, the catch
        # announced, and the batch died anyway. alloc_token_slots has retried
        # since #679 (see its `allocator.alloc(num_tokens)` after the same
        # call); this is that discipline applied verbatim, not a new policy.
        # The raise below is unchanged, so fail-loud still has the last word.
        freed = _attempt_extend_relief(extend_num_tokens)
        if freed > 0:
            out_cache_loc = _attempt_alloc()
            logger.warning(
                "paged extend allocation of %d tokens failed; rank-local "
                "relief returned %d tokens and the retry %s. Admission should "
                "have prevented this -- treat a recurring line here as an "
                "admission defect, not as relief working.",
                extend_num_tokens,
                freed,
                "SUCCEEDED" if out_cache_loc is not None else "still failed",
            )
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
        # #639b: RANK-UNIFORM trigger AND magnitude. `mamba_state_needed` is
        # replicated (`num_reqs` comes from the batch and `factor` from
        # replicated server args); `mamba_available_size` is this rank's own
        # occupancy. Deciding from the local side diverges the mamba tombstone
        # set TWICE over: the roomy rank skips an eviction the tight rank
        # takes, and when both do evict they evict DIFFERENT AMOUNTS, because
        # `mamba_num` is itself computed from the local availability. Either
        # way `MambaComponent.evict_component` tombstones a different set of
        # nodes per rank (`cd.value = None`, KV left in place), the mamba
        # validator then refuses different nodes, `_match_prefix_helper`'s
        # `_all_valid` stops the match at different depths, and
        # `prepare_for_extend` enters the extend collectives with a
        # rank-dependent token axis -- the 2026-08-07 07:45/10:04 crashes.
        # The floor is the group MIN of the mamba slot pool's availability,
        # published once per iteration by the scheduler from a reduce that
        # already ran. None => occupancy agrees (or single rank) => the live
        # local value, unchanged.
        mamba_available_size = uniform_mamba_avail_for_evict(
            tree_cache, mamba_available_size
        )
        if mamba_available_size < mamba_state_needed:
            if tree_cache is not None and tree_cache.supports_mamba():
                mamba_num = max(0, mamba_state_needed - mamba_available_size)
                tree_cache.evict(EvictParams(num_tokens=0, mamba_num=mamba_num))
    req_pool_indices = req_to_token_pool.alloc(reqs)
    if req_pool_indices is None:
        # #583: name the pool that ACTUALLY ran out.
        #
        # `available_size()` is the REQUEST-slot count. On a
        # HybridReqToTokenPool the alloc needs a mamba state (and, with the
        # extra buffer, a ping-pong slot) as well, and it returns None if
        # EITHER is unavailable. Reporting only the request-slot count
        # therefore produced the boot-14 line
        #   available_size()=4, num_reqs=1
        # -- a message saying there was room, on a failure to find room. The
        # mamba pool was 96/96. A fail-loud path that names the wrong
        # resource costs more time than it saves.
        detail = f"{req_to_token_pool.available_size()=} (request slots), {num_reqs=}"
        if isinstance(req_to_token_pool, HybridReqToTokenPool):
            alloc = req_to_token_pool.mamba_allocator
            try:
                detail += (
                    f", mamba_available={alloc.available_size()}"
                    f", mamba_schedulable={alloc.schedulable_available_size()}"
                    f", mamba_total={alloc.size}"
                )
            except Exception:  # noqa: BLE001 - diagnostics must not mask the error
                detail += ", mamba_available=<unavailable>"
            detail += (
                ". A hybrid pool needs BOTH a request slot and a mamba state, "
                "so an exhausted mamba pool fails here even with request slots "
                "free -- raise --max-mamba-cache-size, or look for mamba slots "
                "held by unevictable radix checkpoints"
            )
        # #681 RULE 3: the third raise site on the prefill admission path.
        # This one exhausts REQUEST SLOTS (and mamba states), not KV tokens, so
        # a token-shaped relief cannot pay it -- the net is asked anyway
        # because a provider may free a whole request, and the ask is what
        # tells an operator the site was covered rather than forgotten.
        _attempt_extend_relief(len(reqs))
        raise RuntimeError(
            "alloc_req_slots runs out of memory. "
            "Please set a smaller number for `--max-running-requests`. " + detail
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
        # #715 / #681 THIRD ROOT, decode twin. The free-group window that
        # strands pages is opened by the event loop and does not care which
        # allocation runs inside it, so this path can reach its raise with the
        # pages it needs already freed and merely staged -- exactly as the
        # extend path did. Applying them gives up nothing; the raise below is
        # unchanged, so fail-loud still has the last word.
        staged = _flush_deferred_frees(allocator)
        if staged > 0:
            out = allocator.alloc_decode(
                seq_lens, seq_lens_cpu, last_loc, **extra_alloc_kwargs
            )
            if is_dsv4:
                bundle = out
                out_cache_loc = None if bundle is None else bundle.out_full_loc
                if batch is not None:
                    batch.out_cache_loc_dsv4 = bundle
            else:
                out_cache_loc = out
            logger.warning(
                "paged decode allocation of %d tokens failed with %d tokens "
                "already freed but still staged in the allocator's batching "
                "group; applying them %s.",
                len(seq_lens) * token_per_req,
                staged,
                "SUCCEEDED" if out_cache_loc is not None else "did not help",
            )

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
    # #969K RELEASE-EXIT PROBE (temporary). §O/§Q left one question: the
    # retention path's prefixes never reach the mamba backup writer, the writer
    # never declines (EMPTY=0), and #991 -- the decline inside
    # prepare_for_caching_req -- is 0 on every boot. So the retracted request
    # is not being declined anywhere; it must be leaving through an EARLIER
    # exit than the insert. This function has exactly three, and nothing says
    # which one is taken. Grep: "#969K RELEASE-EXIT".
    try:
        _e = (
            "no_req_pool_idx"
            if req.req_pool_idx is None
            else ("spilled_host" if req.kv_spill_state == "host" else "insert_path")
        )
        _skip = bool(getattr(req, "skip_radix_cache_insert", False))
        _c = globals().setdefault("_969k_counts", {})
        _k = f"{_e}|is_insert={bool(is_insert)}|skip={_skip}"
        _c[_k] = _c.get(_k, 0) + 1
        _n = sum(_c.values())
        if _n <= 40 or _n % 256 == 0:
            logger.warning("#969K RELEASE-EXIT n=%d counts=%s", _n, _c)
    except Exception:  # noqa: BLE001
        logger.warning("#969K RELEASE-EXIT PROBE RAISED", exc_info=True)

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
            # #991: the stamp describes the slot, so it dies with it.
            req.mamba_slot_acquired_this_admission = False
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
    if req.kv_spill_state == "host":
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
