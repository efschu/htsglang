# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Weighted uneven-DCP owner rule (backend-agnostic).

Relocated VERBATIM from ``layers/attention/flashinfer_backend.py`` — the
mechanism never had a flashinfer dependency, it only lived there. It is pure
torch plus the shared Triton kernel ``create_flashinfer_kv_indices_triton``,
so it compiles and runs on any backend Triton supports (CUDA incl. sm75, and
ROCm/HIP incl. gfx900). ``flashinfer_backend.py`` imports it back, so the
CUDA path is byte-identical to before the move.

The owner rule, stated once:
  rank owns cache slot ``L``  <=>  ``(L % cp_S)`` in ``[cp_lo, cp_hi)``
  compact physical slot       ==  ``(L // cp_S) * cp_ratio + (L % cp_S - cp_lo)``
Ownership derives from ``out_cache_loc`` itself rather than the sequence
position, which keeps the compact slot an injective, collision-free function
of ``L`` across concurrent requests — exactly like the even ``L // dcp_size``.
"""

import weakref
from typing import Optional

import torch

from sglang.kernels.ops.kvcache.kv_indices import create_flashinfer_kv_indices_triton

__all__ = [
    "build_dcp_weighted_kv_indices",
    "dcp_accounting_total_slots",
    "dcp_compact_pool_rows",
    "dcp_global_context_slots",
    "dcp_token_sharded_layer",
    "dcp_verify_mask_mode",
    "dcp_verify_paged_lens",
    "dcp_verify_window_is_disjoint",
    "dcp_weighted_owned_lengths",
    "dcp_weighted_owner_bounds",
    "dcp_weighted_read_slots",
    "dcp_weighted_write_slots",
    "refresh_all_owner_bounds",
    "register_owner_bounds_consumer",
    "swa_hybrid_dcp_lane",
]

#: #297 cutover registry: every object that SNAPSHOTS the weighted owner
#: bounds at construction (the attention backends' cached
#: cp_S/cp_lo/cp_hi/cp_ratio) registers itself here, so a live vector flip
#: can refresh EVERY instance -- including per-rung backends the adaptive
#: draft ladder swaps in later -- without any caller having to enumerate
#: them. Weak references: a swapped-out backend must not be kept alive by
#: the registry.
_OWNER_BOUNDS_CONSUMERS: "weakref.WeakSet" = weakref.WeakSet()


def register_owner_bounds_consumer(consumer) -> None:
    """Register an object exposing ``refresh_dcp_owner_bounds()`` for #297.

    Called by every attention backend that caches the weighted owner bounds
    at ``__init__`` time. Idempotent (WeakSet).
    """
    if not hasattr(consumer, "refresh_dcp_owner_bounds"):
        raise TypeError(
            f"{type(consumer).__name__} registers as an owner-bounds "
            "consumer but has no refresh_dcp_owner_bounds() method"
        )
    _OWNER_BOUNDS_CONSUMERS.add(consumer)


def refresh_all_owner_bounds() -> int:
    """#297 cutover: re-derive every registered consumer's cached bounds
    from the freshly installed token vector. Returns the consumer count
    (logged by the reshard runtime as cutover evidence)."""
    consumers = list(_OWNER_BOUNDS_CONSUMERS)
    for consumer in consumers:
        consumer.refresh_dcp_owner_bounds()
    return len(consumers)


def draft_kv_layout_is_dcp(server_args) -> bool:
    """#108: has the user opted the DRAFT KV pool into DCP token-sharding?

    Reads ``--draft-kv-layout`` defensively (``getattr``) because several test
    doubles and the CUDA-graph runners construct model runners with partial
    server-args stand-ins; an absent field means the unchanged default.
    """
    return getattr(server_args, "draft_kv_layout", "replicated") == "dcp"


def draft_pool_is_replicated(is_draft_worker: bool, server_args) -> bool:
    """THE single predicate for "this worker keeps the legacy replicated draft
    KV pool" (#108).

    Two independent sites decide the draft pool's geometry -- the KV pool
    sizing (``model_runner_kv_cache_mixin``: replicated kv-heads +
    ``dcp_compact_pool_rows``) and the attention backend
    (``FlashInferAttnBackend.uneven_dcp``: owner-rule masked write, per-layer
    kv-head all-gather, cross-rank LSE merge). If those two ever disagree the
    pool is shaped for a token split the backend does not perform, which is
    the right-token/wrong-slot corruption class #345 already cost a debugging
    round on -- not a crash, a silently wrong address that drifts with the
    slot id.

    So both sites call THIS function and neither re-derives the condition.
    The same coupling argument as ``_swa_hybrid_dcp_lane``.

    True  -> the M4 default: the draft runner keeps its tiny full-context,
             head-sharded 1-layer pool and every DCP branch stays off for it.
    False -> either not a draft worker at all, or ``--draft-kv-layout dcp``:
             the draft pool takes the identical target-model treatment.
    """
    return bool(is_draft_worker) and not draft_kv_layout_is_dcp(server_args)


def reject_multi_layer_draft_kv_dcp(num_draft_kv_layers: int, arch: str) -> None:
    """BOOT GATE tier 2 (#108): --draft-kv-layout dcp is a ONE-layer contract.

    ``ServerArgs._reject_unsupported_draft_kv_dcp`` bounds the algorithm and
    the topk from the CLI alone; it cannot see how many attention layers the
    resolved draft checkpoint actually carries. NEXTN/MTP heads normalise to
    ``num_nextn_predict_layers = 1`` (configs/model_config.py), but an EAGLE
    checkpoint resolving through the same alias can carry several.

    Why a second layer is not "the same thing twice": the owner rule shards a
    single linear append chain, one KV row per accepted token per layer. With
    L > 1 draft layers the draft's own decode reads positions written by an
    earlier layer of the SAME forward, and the cross-rank LSE merge would have
    to run per layer inside the draft step -- kernels that do not exist here
    (#76-scale work). Refuse at pool construction rather than emit tokens from
    rows nobody wrote.
    """
    if int(num_draft_kv_layers) <= 1:
        return
    raise ValueError(
        "--draft-kv-layout dcp supports a single-layer draft chain "
        f"(MTP/NEXTN) only, but the draft model {arch!r} has "
        f"{int(num_draft_kv_layers)} KV layers. Multi-layer EAGLE drafts need "
        "per-layer owner-rule kernels that are not implemented; run this "
        "draft with --draft-kv-layout replicated."
    )


def dcp_compact_pool_rows(global_tokens: int, cp_S: int, cp_ratio: int) -> int:
    """Physical KV rows this rank must hold for a GLOBAL context of
    ``global_tokens`` slots under the weighted owner rule.

    The owner rule stores global slot ``L`` at
    ``(L // cp_S) * cp_ratio + (L % cp_S - cp_lo)``, so a context of C slots
    needs ``(C // cp_S + 1) * cp_ratio`` rows -- note the CEIL to a whole owner
    block, which is not cosmetic: allocator slot ids reach ``C`` itself, and any
    slot in a trailing PARTIAL block (``C % cp_S != 0``, e.g. an explicit
    unaligned ``--max-total-tokens``) compacts to ``(C // cp_S) * cp_ratio +
    off`` -- one block PAST the floored sizing. Flooring let those top slots
    scatter out of bounds once free-list churn handed them out (async illegal
    memory access, found by the kv-session-offload S1 test with
    --max-total-tokens 3000 on cp_S=64). It costs under one block of rows.

    Stated ONCE here because two pool families need it (HybridLinearKVPool for
    the mamba/linear hybrids, SWAKVPool's full sub-pool for the SWA hybrids)
    and a sizing rule whose off-by-one has already cost a debugging round must
    not exist twice.
    """
    if cp_S <= 0 or cp_ratio <= 0:
        raise ValueError(
            f"dcp_compact_pool_rows: cp_S ({cp_S}) and cp_ratio ({cp_ratio}) "
            "must be positive; they come from dcp_weighted_owner_bounds and "
            "are only meaningful on the weighted lane."
        )
    return (int(global_tokens) // int(cp_S) + 1) * int(cp_ratio)


def dcp_accounting_total_slots(
    max_total_num_tokens: int,
    allocator_size: Optional[int],
    *,
    token_sharded_dcp: bool,
) -> int:
    """The ``total`` a KV-slot leak check must compare ``available`` against.

    ``available_size()`` counts the ALLOCATOR's index space, so the total it
    is checked against has to be the same quantity. Off the token-sharded DCP
    lane the two are the same number and this returns
    ``max_total_num_tokens`` unchanged.

    On the lane they can differ, and in exactly one direction. Under the
    WEIGHTED owner rule ``max_total_num_tokens`` is already the GLOBAL context
    budget C, which is also the allocator's index space -- nothing moves.
    Under the EVEN-MODULO rule (``SGLANG_UNEVEN_DCP=1`` with
    ``SGLANG_UNEVEN_DCP_WEIGHTED=0``) ``max_total_num_tokens`` is this rank's
    PHYSICAL pool while the allocator hands out global slot ids over
    ``max_total * cp_token_split_factor(dcp_size)`` of them. Comparing the two
    made the first idle check report ``available`` at exactly ``dcp_size`` x
    ``total`` and killed the server at boot -- "pool memory leak detected!
    [full] total=28640, available=57280" (#345). The pool geometry was never
    wrong there; one of the two scalars was simply not the quantity the other
    one counts.
    """
    if not token_sharded_dcp or allocator_size is None:
        return int(max_total_num_tokens)
    return int(allocator_size)


def dcp_global_context_slots(
    per_rank_pool_tokens: int,
    dcp_size: Optional[int],
) -> int:
    """The GLOBAL token span this DCP group can hold, from a rank's pool size.

    This is the counterpart of :func:`dcp_accounting_total_slots` for the
    consumers that decide how much CONTEXT a request may have -- the ceiling
    behind ``max_req_len`` / ``max_req_input_len`` and the generation budget
    ``Scheduler.init_req_max_new_tokens`` subtracts the input length from.
    Both compare against GLOBAL token counts (a request's length is a global
    quantity on every rank), so they need the global span, not this rank's
    physical pool. The two rules of the token-sharded lane differ in whether
    those are the same number:

    * WEIGHTED (a token vector is installed): ``max_total_num_tokens`` is
      ALREADY the global context budget C -- the allocator index space is C
      and each rank stores its ``ratio_r / S`` share. Identity.
    * EVEN-MODULO (``SGLANG_UNEVEN_DCP=1`` with ``SGLANG_UNEVEN_DCP_WEIGHTED=0``):
      ``max_total_num_tokens`` is this rank's PHYSICAL pool P and the
      allocator hands out ``P * cp_token_split_factor(dcp_size)`` global slot
      ids -- global slot L lives on rank ``L % S`` at compact row ``L // S``,
      so the span is P x S and a consumer reading P as a global count refuses
      context the group can hold (#346). The reclaimed window ``(P, P*S]``
      compacts to rows ``(P // S, P]``, all inside the pool's
      ``size + page_size`` rows and therefore inside the #352/#355
      ``kv_store_bound`` -- this widens a REPORT, never a bound.

    Off the lane, and on stock even DCP (no ``--rank-tp-ratio`` head plan),
    this is the identity: stock DCP's ceiling is upstream's to define and its
    allocator geometry (inflated page size) is not this fork's. That is a
    deliberate difference from ``Scheduler._pool_stats_dcp_factor``, which
    scales stock even DCP as well -- a utilization PERCENTAGE that comes out
    dcp_size too high is a wrong log line, while a raised ceiling is a
    behaviour change on a path this fork does not exercise. The two agree
    wherever this fork's lane is active.

    Resolving the lane HERE rather than at each caller is deliberate. #345
    (leak check), #352 (store bound) and this task are three instances of the
    same defect -- one consumer reading a per-rank number as a global one --
    and a second copy of the rule is how the fourth one happens.
    """
    from sglang.srt.distributed.utils import (
        cp_token_split_factor,
        uneven_dcp_active,
        uneven_dcp_kv_replicated,
        weightless_kv_active,
    )

    tokens = int(per_rank_pool_tokens)
    size = int(dcp_size or 1)
    if size <= 1:
        return tokens
    if not (uneven_dcp_kv_replicated(size) or weightless_kv_active()):
        return tokens
    if uneven_dcp_active(size):
        return tokens
    return tokens * cp_token_split_factor(size)


def swa_hybrid_dcp_lane(
    *,
    is_hybrid_swa: bool,
    uneven_plan: bool,
    is_draft_worker: bool,
    num_full_layers: int,
    num_swa_layers: int,
    swa_pool_sizing_capped: bool,
) -> bool:
    """Is this process serving SWA-hybrid uneven DCP (task #96, Stage B)?

    THE lane predicate: the ~10 global full-attention layers are token-sharded
    by the weighted owner rule while the ~50 sliding-window layers keep their
    local, unsharded path. Consumed by the KV-pool sizing, the Triton attention
    backend, and the geometry guard, so those three cannot disagree about which
    mode the process is in.

    A pure function of process-global CONFIGURATION -- server args and the model
    config -- and of nothing derived from a batch, a request, or a rank's data.
    That is what makes the per-layer dispatch built on it group-uniform, which
    is the whole safety argument: the sharded layers enter a q all-gather and an
    LSE merge, so every rank must classify every layer identically or the group
    deadlocks on a layer one rank skipped ([[rank-lokaler-test-vor-kollektiv]]).

    Conditions, and why each is necessary:

    ``is_hybrid_swa`` + ``num_swa_layers > 0`` + ``num_full_layers > 0``
        There must be both kinds of layer. A pure-SWA model has nothing to
        shard (window-bounded KV only): DCP would add collectives for no
        capacity. A model with no SWA layers is not this lane at all -- it is
        plain #173.
    ``uneven_plan``
        ``uneven_dcp_kv_replicated(dcp_size)``: a --rank-tp-ratio plan spanning
        the TP group with dcp_size == tp_size, i.e. the token-sharded pool with
        replicated kv-head rows that the owner rule assumes.
    ``not is_draft_worker``
        Exactly as in #173: the draft/NEXTN worker keeps a plain uneven-TP pool
        (local heads, FULL token context), so DCP is off for that instance.
    ``swa_pool_sizing_capped``
        Stage A (--swa-pool-sizing cap, or the legacy --disable-radix-cache
        route to the same configurator). In ratio mode the SWA pool is sized
        ``ratio * full_tokens``, and under DCP ``full_tokens`` is the GLOBAL
        context budget C -- several times any single rank's reach -- while the
        SWA pool is NOT sharded. That is the pre-#90 OOM disease multiplied by
        the token split. Stage B strictly builds on Stage A.
    """
    return bool(
        is_hybrid_swa
        and uneven_plan
        and not is_draft_worker
        and int(num_full_layers) > 0
        and int(num_swa_layers) > 0
        and swa_pool_sizing_capped
    )


def dcp_token_sharded_layer(is_swa_layer: bool, *, swa_hybrid_lane: bool) -> bool:
    """Is THIS layer's KV token-sharded across the DCP group?

    The per-layer dispatch of Stage B, in one place so the write path, the
    extend path and the decode path cannot drift apart: under the SWA-hybrid
    lane the sliding-window layers are replicated (every rank holds every
    in-window position of its own kv-head shard) and only the global
    full-attention layers are sharded. Off the lane every layer is sharded,
    which is byte-for-byte the #173 behaviour.

    ``is_swa_layer`` must come from the MODEL (the layer's configured sliding
    window, or the pool's ``layers_mapping``), never from a per-batch quantity;
    see ``swa_hybrid_dcp_lane`` for why.
    """
    return not (swa_hybrid_lane and is_swa_layer)


def dcp_weighted_owner_bounds(dcp_size: int, dcp_rank: int):
    """``(cp_S, cp_lo, cp_hi, cp_ratio)`` for this rank's weighted owner range.

    Single derivation shared by every backend, so a rank cannot end up reading
    with one set of bounds and writing with another.
    """
    from sglang.srt.distributed.utils import cp_token_prefix

    prefix = cp_token_prefix(dcp_size)
    cp_S = prefix[-1]
    cp_lo = prefix[dcp_rank]
    cp_hi = prefix[dcp_rank + 1]
    return cp_S, cp_lo, cp_hi, cp_hi - cp_lo


def dcp_weighted_write_slots(
    cache_loc: torch.Tensor,
    cp_S: int,
    cp_lo: int,
    cp_hi: int,
    cp_ratio: int,
):
    """WRITE side of the weighted owner rule: ``(loc, mask)`` for ``cache_loc``.

    ``mask[i]`` is True iff this rank owns slot ``cache_loc[i]``; ``loc[i]`` is
    the compact physical slot to write it to (0 where not owned, matching the
    established convention that unowned entries are masked out anyway).

    This is the EXACT INVERSE of the read-side mapping in
    ``build_dcp_weighted_kv_indices``:
        read:  compact = (L // cp_S) * cp_ratio + (L % cp_S - cp_lo)
        write: loc     = (L // cp_S) * cp_ratio + (L % cp_S - cp_lo)
    They are deliberately the same expression, defined ONCE here. Read and
    write drifting apart is the failure mode that makes a token be fetched
    from a slot it was never stored in -- silently wrong output rather than a
    crash -- so the two sides must not be maintained as separate copies in the
    flashinfer and Triton backends.
    """
    off = cache_loc % cp_S
    block = cache_loc // cp_S
    mask = (off >= cp_lo) & (off < cp_hi)
    loc = torch.where(
        mask,
        block * cp_ratio + (off - cp_lo),
        torch.zeros_like(cache_loc),
    )
    return loc, mask


def dcp_weighted_read_slots(
    cache_loc: torch.Tensor,
    cp_S: int,
    cp_lo: int,
    cp_hi: int,
    cp_ratio: int,
):
    """READ side of the weighted owner rule: ``(compact, owned)``.

    The exact counterpart of ``dcp_weighted_write_slots`` -- SAME expression,
    stated once -- for a flat list of GLOBAL cache slots (typically the
    ``req_to_token`` rows of a paged read). ``owned[i]`` is True iff this rank
    stores slot ``cache_loc[i]``; ``compact[i]`` is the physical row it stores
    it in. ``compact`` is computed for EVERY entry, not only the owned ones,
    so the caller can mask without a second pass; unowned entries carry a
    meaningless (possibly negative) value and must not be dereferenced.

    Extracted out of ``build_dcp_weighted_kv_indices`` so that the index math
    -- the part that decides whether uneven DCP is CORRECT -- is a pure tensor
    function with no Triton kernel in it, and can therefore be pinned against
    an independent reference on CPU, with no device and no collectives.
    """
    loc64 = cache_loc.to(torch.int64)
    off = loc64 % cp_S
    owned = (off >= cp_lo) & (off < cp_hi)
    compact = ((loc64 // cp_S) * cp_ratio + (off - cp_lo)).to(torch.int32)
    return compact, owned


def dcp_weighted_owned_lengths(
    owned: torch.Tensor,
    lens: torch.Tensor,
) -> torch.Tensor:
    """Per-request count of owned slots, given the flat ``owned`` mask.

    ``owned`` is the concatenation of the per-request slot lists in request
    order and ``lens`` their (global) lengths, so this is a segmented sum.
    Returned as int64, one entry per request. Pure tensor math -- CPU-safe,
    tested against numpy.
    """
    lens64 = lens.to(torch.int64)
    bs = lens64.numel()
    out = torch.zeros(bs, dtype=torch.int64, device=lens64.device)
    if owned.numel() == 0:
        return out
    seg = torch.repeat_interleave(
        torch.arange(bs, device=lens64.device, dtype=torch.int64), lens64
    )
    out.scatter_add_(0, seg, owned.to(torch.int64))
    return out


def build_dcp_weighted_kv_indices(
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
    req_to_token_stride: Optional[int] = None,
):
    """Weighted-DCP paged kv_indices: this rank's OWNED cache slots, compacted.

    Builds the full per-request out_cache_loc list for [kv_start, kv_start+len),
    keeps the slots this rank owns under the WEIGHTED owner rule
    (loc % cp_S in [cp_lo, cp_hi)) and maps each to its compact physical slot
    (loc // cp_S) * cp_ratio + (loc % cp_S - cp_lo) -- the exact inverse of the
    _dcp_masked_write packing, so a token is read from the slot it was written
    to. Writes the per-request owned counts into ``kv_indptr`` and returns
    (kv_indptr[:bs+1], kv_indices).

    ``req_to_token_stride`` defaults to ``req_to_token.shape[1]``, which is the
    row stride of the (contiguous) req_to_token table. Callers that hold a
    stride from the pool pass it explicitly rather than re-deriving it.

    Only the ``create_flashinfer_kv_indices_triton`` launch is device-bound;
    the actual owner-rule arithmetic is ``dcp_weighted_read_slots`` /
    ``dcp_weighted_owned_lengths``, which are pure tensor functions pinned on
    CPU against an independent reference.
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
            (
                req_to_token.shape[1]
                if req_to_token_stride is None
                else req_to_token_stride
            ),
        )
    full_kv = full_kv[:total]
    compact, owned = dcp_weighted_read_slots(full_kv, cp_S, cp_lo, cp_hi, cp_ratio)
    kv_indices = compact[owned].contiguous()
    if pad > 0:
        kv_indices = torch.cat([kv_indices, kv_indices.new_zeros(pad)])
    owned_per_req = dcp_weighted_owned_lengths(owned, lens64)
    kv_indptr[1 : bs + 1] = torch.cumsum(owned_per_req, dim=0)
    return kv_indptr[: bs + 1], kv_indices


# ---------------------------------------------------------------------------
# The M4 verify split (#180). Three pure decisions, so the parts of a
# speculative verify that decide whether uneven DCP is CORRECT can be pinned
# with no device, no process group and no model.
# ---------------------------------------------------------------------------


def dcp_verify_paged_lens(
    seq_lens: torch.Tensor,
    num_draft_tokens: int,
) -> torch.Tensor:
    """The PAGED read length vector for a target-verify step: ``seq_lens``.

    A verify step appends ``num_draft_tokens`` query positions per request, and
    the attention over them splits in two:

      * the COMMITTED prefix ``[0, seq_len)`` -- read from the KV pool, and
        therefore owner-sharded under the weighted rule;
      * the draft block -- attended out of this rank's freshly projected k/v,
        which is locally complete on every rank, so it is NOT sharded.

    ``seq_lens`` is the committed length: ``eagle_prepare_for_verify`` allocates
    ``out_cache_loc`` for the draft tokens but does NOT advance ``seq_lens``.

    The point of this function is what it does NOT return. Reading
    ``seq_lens + num_draft_tokens`` from the pool would double-count the draft
    block against the ragged stage AND dereference slots this rank may not own
    -- and it is the natural mistake, because the non-DCP verify branch this
    replaces also reads ``seq_lens``, for the unrelated reason that the draft
    k/v are handed to the kernel as tensors. Two paths, same length vector,
    different reasons; stating the reason once is cheaper than rediscovering it.

    ``num_draft_tokens`` is taken only to be validated -- a verify step with no
    draft tokens is not a verify step.
    """
    if num_draft_tokens <= 0:
        raise ValueError(
            f"target-verify needs num_draft_tokens >= 1, got {num_draft_tokens}"
        )
    return seq_lens


def dcp_verify_window_is_disjoint(num_draft_tokens: int, qo_stride: int) -> bool:
    """Do the paged and ragged stages together cover the attended context once?

    The verify invariant, checkable on integers alone: each of the ``d`` draft
    queries of a request attends ``seq_len`` paged rows plus its own causal
    prefix of the ``d`` ragged rows, and the two sets are disjoint because the
    paged read stops exactly where the draft block begins. So the split is
    complete and non-overlapping iff the ragged stage's per-request query count
    (``qo_stride``, the uniform ``qo_indptr`` step) equals ``num_draft_tokens``.

    A mismatch is what a stale ``num_draft_tokens`` looks like: the kernel grid
    covers fewer query blocks than there are draft tokens and silently drops
    the tail, which reads as a collapsed accept rate rather than as an error.
    That is live in this tree, not hypothetical: the Triton backend builds
    ``qo_indptr`` from the CONSTRUCTOR-time ``self.num_draft_tokens``
    (``server_args.speculative_num_draft_tokens``) while the verify input
    carries the step's actual ``draft_token_num``, and the adaptive draft-length
    ladder swaps a whole per-rung backend to keep the two in step. The check is
    exactly the question "did that swap happen".

    INTEGERS ONLY, deliberately. An earlier form also took ``seq_lens`` and
    returned ``(seq_lens >= 0).all()``. That term is vacuous -- a length vector
    is non-negative by construction -- and it is what kept the function
    uncallable: it is a device->host sync, so it costs a stall per verify step
    in eager and is illegal inside a CUDA-graph capture, which is where the two
    call sites are. Dropping it is what turned the stated invariant into an
    enforced one.
    """
    return num_draft_tokens > 0 and qo_stride == num_draft_tokens


def dcp_verify_mask_mode(topk: Optional[int], dflash_tree_verify: bool = False) -> str:
    """``"causal"`` or ``"tree"`` -- which draft->draft mask a verify needs.

    At ``topk == 1`` the EAGLE draft is a CHAIN, so its d x d draft->draft mask
    block is exactly lower-triangular: the plain causal mask. The verify can
    therefore drop ``custom_mask`` entirely, which is not merely an
    optimisation under DCP but a requirement -- the mask's row stride is the
    GLOBAL prefix length, which stops describing the rows the kernel walks once
    the prefix is owner-sharded.

    The equivalence is already relied on twice in this tree
    (``kernels/ops/attention/verify_splitkv.py`` gates on ``topk == 1`` for the
    same reason; the flashinfer verify branch sets ``custom_mask = None``), so
    this states it a third time in ONE place instead of a third copy.

    ``dflash_tree_verify`` is the second door onto a tree-masked verify
    (upstream #31069/#29587/#29907). It is read here for the same reason
    ``ServerArgs.tree_verify_activation_reason`` reads it: a new flag name
    reaching the same draft->draft tree mask is exactly how a guard gets
    bypassed without anyone deciding to bypass it.
    """
    if dflash_tree_verify:
        return "tree"
    if topk is not None and topk > 1:
        return "tree"
    return "causal"
