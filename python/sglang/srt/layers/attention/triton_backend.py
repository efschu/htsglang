from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Sequence, Union

import torch
import triton

from sglang.kernels.ops.attention.metadata import get_num_kv_splits_triton
from sglang.kernels.ops.kvcache.kv_indices import (
    create_flashinfer_kv_indices_triton,
)
from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.environ import envs
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.dcp import (
    build_dcp_weighted_kv_indices,
    cp_all_gather_heads_uneven,
    cp_lse_ag_out_ar_mha_uneven,
    create_triton_kv_indices_for_dcp_triton,
    dcp_even_write_mask,
    dcp_fresh_host_lens,
    dcp_host_even_total,
    dcp_host_total_tokens,
    dcp_token_sharded_layer,
    dcp_verify_mask_mode,
    dcp_verify_paged_lens,
    dcp_verify_window_is_disjoint,
    dcp_weighted_owner_bounds,
    dcp_weighted_write_slots,
    get_dcp_lens,
    swa_hybrid_dcp_lane,
)
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.model_executor.cuda_graph_config import cuda_graph_fully_disabled
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.runtime_context import get_parallel
from sglang.srt.speculative.spec_info import SpecInputType
from sglang.srt.speculative.spec_utils import (
    draft_kv_indices_buffer_width,
    draft_kv_indices_used_len,
    generate_draft_decode_kv_indices,
)
from sglang.srt.utils import (
    get_bool_env_var,
    get_device_core_count,
    get_int_env_var,
    is_cuda,
    is_gfx95_supported,
    is_gfx942_supported,
    is_xpu,
    next_power_of_2,
)

_is_cuda = is_cuda()
_is_gfx942 = is_gfx942_supported()
_is_xpu = is_xpu()

if _is_cuda:
    # PDL (Programmatic Dependent Launch) is a Hopper+ feature, and the helper
    # that reports it lives in sgl_kernel -- which has no code for sm75 (the
    # wheel is cubin-only with a gencode floor of sm_80). Without a guard a
    # Turing card cannot even construct the Triton attention backend, although
    # the answer for it is simply "no PDL".
    try:
        from sgl_kernel.utils import is_arch_support_pdl

        _has_pdl_probe = True
    except ImportError:
        _has_pdl_probe = False

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpecInput


# The verify-input types the M4 split is valid for (#180), kept in lockstep with
# flashinfer's _DCP_VERIFY_SPEC_INPUT_TYPES. Both present a uniform
# draft_token_num query block per request and a LINEAR (non-tree) draft chain,
# which is exactly what the split assumes: the draft block is attended ragged on
# local heads with a plain causal mask, and the committed prefix is read
# owner-sharded. A verify type with a different query layout would be routed
# into that assumption silently, so anything else is refused rather than served
# -- and refused rather than fallen back to the FULL un-sharded build, which is
# the out-of-bounds read this whole change exists to remove.
_DCP_VERIFY_SPEC_INPUT_TYPES = frozenset(
    {SpecInputType.EAGLE_VERIFY, SpecInputType.DFLASH_VERIFY}
)


def _verify_host_mirror(
    seq_lens_cpu: Optional[Union[Sequence[int], torch.Tensor]],
    num_draft_tokens: int,
) -> Optional[Union[Sequence[int], torch.Tensor]]:
    """The target-verify PAGED length mirror, or None (#623/#629).

    The mirror goes through the SAME length function as the device vector
    rather than assuming the identity here: ``dcp_verify_paged_lens`` is where
    the reason the verify paged length is NOT ``seq_lens + num_draft_tokens``
    is written down, and a mirror derived by a second, independent expression
    is exactly the drift the #616h wiring exists to prevent.

    Shared by the eager verify site and its cuda-graph replay-prep twin so the
    two cannot diverge -- they were separate inline expressions before #629,
    and only one of them had been wired.
    """
    if seq_lens_cpu is None:
        return None
    return dcp_verify_paged_lens(seq_lens_cpu, num_draft_tokens)


def _reject_stale_verify_window(spec_info, qo_stride: int) -> None:
    """Enforce the M4 split's disjointness invariant at a verify metadata build.

    ``qo_indptr`` is built with a uniform step of ``self.num_draft_tokens``,
    which is read once in the constructor from
    ``server_args.speculative_num_draft_tokens``. The verify input carries the
    step's ACTUAL block width in ``draft_token_num``. The two are kept in step
    by construction -- the adaptive draft-length ladder builds one backend per
    rung and swaps the whole object -- so this is a check on that swap, not on
    the arithmetic. If it ever misses, the grid covers fewer query blocks than
    there are draft tokens and drops the tail, which reads as a collapsed
    accept rate rather than as an error. Two ints, no device work: safe in the
    eager build and inside a graph capture alike.
    """
    d = getattr(spec_info, "draft_token_num", None)
    if d is None:
        return
    if not dcp_verify_window_is_disjoint(int(d), qo_stride):
        raise ValueError(
            f"uneven-DCP target-verify window is not disjoint: the verify input "
            f"declares draft_token_num={d} while qo_indptr is built with a step "
            f"of {qo_stride} (the backend's num_draft_tokens). The paged stage "
            f"reads the committed prefix and the ragged stage the draft block; "
            f"a mismatched step makes the kernel grid drop the draft tail "
            f"silently. This backend instance is stale for the current draft "
            f"length -- see the adaptive draft-length per-rung backend swap."
        )


_MLA_DECODE_MIN_BLOCK_KV = 32


def _mla_decode_kv_splits_cap(
    base_max_kv_splits: int, sm_count: int, max_context_len: int
) -> int:
    if sm_count <= 0:
        return base_max_kv_splits
    sm_cap = next_power_of_2(sm_count)
    ctx_cap = next_power_of_2(triton.cdiv(max_context_len, _MLA_DECODE_MIN_BLOCK_KV))
    return max(base_max_kv_splits, min(sm_cap, ctx_cap))


def _plan_aware_num_q_heads(model_config) -> int:
    """This rank's q-attention-head count for workspace sizing.

    Even TP (no --rank-tp-ratio plan): exactly total_q // tp_size — the
    previous behavior, bit for bit. Uneven TP: q heads follow the shard
    plan in whole-GQA-group units, so the per-rank count can EXCEED the
    even split (e.g. 32 heads over 3 ranks -> [8,14,10], not 10/10/10);
    sizing the decode attn_logits/attn_lse workspaces with the even split
    makes the decode kernel write out of bounds on the bigger ranks and
    silently corrupt neighboring allocations (this is the triton twin of
    the flashinfer per-rank head-count fix). Hybrid SWA models partition
    q differently per layer type (kv-head base 4 vs 16 -> different GQA
    units), so take the max across the model's kv bases.
    """
    from sglang.srt.distributed.utils import (
        attn_q_partition_groups,
        attn_q_partition_units,
        tp_partition_size,
        tp_plan_active,
    )

    tp_size = get_parallel().attn_tp_size
    total_q = model_config.num_attention_heads
    if not tp_plan_active(tp_size):
        return total_q // tp_size
    rank = get_parallel().attn_tp_rank
    kv_bases = {model_config.get_total_num_kv_heads()}
    swa_kv = getattr(model_config.hf_text_config, "swa_num_key_value_heads", None)
    if swa_kv:
        kv_bases.add(swa_kv)
    return max(
        tp_partition_size(
            total_q,
            tp_size,
            rank,
            attn_q_partition_units(total_q, kv, tp_size),
            # kv-boundary alignment (task #116): match the aligned qkv split.
            groups=attn_q_partition_groups(kv, tp_size),
        )
        for kv in kv_bases
    )


def _plan_aware_dcp_gathered_q_heads(model_config, dcp_size: int) -> int:
    """Head count of the GATHERED q set the decode workspaces must hold.

    Under DCP each rank's partial attention is merged across the DCP group, so
    the attn_logits/attn_lse workspaces are sized for the gathered head set --
    NOT for this rank's own shard.

    The previous expression was ``_plan_aware_num_q_heads(cfg) * dcp_size``.
    That double-counts under an uneven plan, because _plan_aware_num_q_heads is
    ALREADY plan-aware and returns THIS rank's (unequal) count. Concretely, with
    total_q=32, kv=8, tp=dcp=3 and a --rank-tp-ratio of [12,6,6], the real split
    is [16,8,8], so the old formula gave rank 0 16*3=48 (harmless over-alloc)
    but ranks 1 and 2 8*3=24 against a true gathered 32 -- EIGHT HEADS SHORT,
    i.e. exactly the out-of-bounds decode write _plan_aware_num_q_heads' own
    docstring warns about. Latent until the Triton backend is wired for uneven
    DCP; this lands in the same change as that wiring.

    The correct value follows from the partition being EXHAUSTIVE: the per-rank
    q counts sum to total_q (verified against the partition helpers for both
    [2,1,1] and [12,6,6] -> [16,8,8], sum 32). So when DCP spans the whole TP
    group the gathered set is every q head exactly once.

    Cases, in order:
      dcp_size == 1        -> no gather at all; this rank's own count.
                              Byte-identical to the old expression, and it is
                              what keeps uneven-TP-WITHOUT-DCP (a working,
                              validated path) unchanged.
      no plan active       -> (total_q // tp_size) * dcp_size, byte-identical.
      dcp_size == tp_size  -> total_q. The fix. (Every uneven-DCP path already
                              requires dcp_size == tp_size.)
      otherwise            -> max over ranks * dcp_size: an UPPER bound, since
                              max*n >= sum. Over-allocating is harmless;
                              under-allocating writes out of bounds, so this is
                              the correct direction to be wrong in.
    """
    from sglang.srt.distributed.utils import (
        attn_q_partition_groups,
        attn_q_partition_units,
        tp_partition_size,
        tp_plan_active,
    )

    if dcp_size == 1:
        return _plan_aware_num_q_heads(model_config)

    tp_size = get_parallel().attn_tp_size
    total_q = model_config.num_attention_heads
    if not tp_plan_active(tp_size):
        return (total_q // tp_size) * dcp_size
    if dcp_size == tp_size:
        return total_q

    kv_bases = {model_config.get_total_num_kv_heads()}
    swa_kv = getattr(model_config.hf_text_config, "swa_num_key_value_heads", None)
    if swa_kv:
        kv_bases.add(swa_kv)
    per_rank_max = max(
        tp_partition_size(
            total_q,
            tp_size,
            r,
            attn_q_partition_units(total_q, kv, tp_size),
            groups=attn_q_partition_groups(kv, tp_size),
        )
        for kv in kv_bases
        for r in range(tp_size)
    )
    return per_rank_max * dcp_size


def _plan_aware_dcp_group_q_head_counts(
    model_config, dcp_size: int, local_heads: int
) -> list:
    """Per-rank q-head counts of THIS rank's DCP group, in rank order.

    The DCP head collectives need the WHOLE group's per-rank counts, not just
    this rank's: an all-gather along the head dim has to know how many heads
    each peer contributes, and the LSE merge has to slice this rank's heads
    back out of the gathered set by prefix sum. With an equal split both
    reduce to ``H // world_size`` -- which is exactly why the assumption could
    sit unstated in the code for so long.

    ``local_heads`` is taken FROM THE MODEL (the q tensor actually handed to
    the forward), not re-derived, and it is what the no-plan case replicates.
    That keeps the default path a pure identity: whatever head count the layer
    reports is the count every peer is assumed to have, byte-for-byte the
    situation before this helper existed. Only with a --rank-tp-ratio plan
    installed do the counts come from the partition helpers, and then
    cp_all_gather_heads_uneven asserts counts[rank] == local_heads, so a model
    whose reported per-rank head count disagrees with the plan fails loudly
    at the first forward instead of issuing a mismatched collective.

    ONE KV-HEAD BASE HERE, NOT max() OVER BOTH (#96). The workspace-sizing
    helpers (_plan_aware_num_q_heads, _plan_aware_dcp_gathered_q_heads) take the
    max over a hybrid model's two kv-head bases, because over-allocating a
    buffer is harmless and under-allocating writes out of bounds. These counts
    are NOT a buffer size: they are a collective's per-rank byte counts and the
    prefix sum the LSE merge slices this rank's heads back out with, so they
    must be EXACT and EXHAUSTIVE. A hybrid model's two bases give two different
    (each exhaustive) q partitions -- Gemma-4-31B TP=3 uneven measured q sliding
    [8,14,10] and q full [8,16,8] -- and their per-rank max [8,16,10] sums to 34
    against a total of 32: not a partition of anything. It would trip
    cp_all_gather_heads_uneven's counts[rank] == local_heads assert on the rank
    whose max is not its own count, or, where that passes, slice the wrong heads
    out of the merge.

    The base is the FULL-ATTENTION one, because those are the only layers that
    enter a DCP collective: under the SWA-hybrid lane (#96) the sliding-window
    layers are replicated and never gathered, and off that lane a model has one
    base anyway (so single-base models are byte-identical to before).
    """
    from sglang.srt.distributed.utils import (
        attn_q_partition_groups,
        attn_q_partition_units,
        tp_partition_size,
        tp_plan_active,
    )

    if dcp_size <= 1:
        return [local_heads]

    tp_size = get_parallel().attn_tp_size
    if not tp_plan_active(tp_size):
        return [local_heads] * dcp_size

    total_q = model_config.num_attention_heads
    kv = model_config.get_total_num_kv_heads()
    # DCP groups are consecutive tp slices, so this rank's group is the
    # dcp_size-wide block its tp rank falls into.
    group_start = (get_parallel().attn_tp_rank // dcp_size) * dcp_size
    counts = [
        tp_partition_size(
            total_q,
            tp_size,
            r,
            attn_q_partition_units(total_q, kv, tp_size),
            groups=attn_q_partition_groups(kv, tp_size),
        )
        for r in range(group_start, group_start + dcp_size)
    ]
    if dcp_size == tp_size and sum(counts) != total_q:
        # The group spans the whole TP group, so the counts must partition the q
        # heads exactly. Anything else means the gathered set is not the model's
        # head set, and every consumer downstream (gather byte counts, merge
        # slice offsets) is then wrong in a way that reads as garbage output
        # rather than as an error. Fail here, naming the numbers.
        raise ValueError(
            f"DCP q-head counts {counts} sum to {sum(counts)} but the model has "
            f"{total_q} q heads (kv base {kv}, tp {tp_size}). The per-rank q "
            "shards must be an exhaustive partition -- a non-partition means "
            "the head geometry helpers and the model disagree."
        )
    return counts


def total_swa_kv_heads(model_config) -> Optional[int]:
    """The model's SECOND (sliding-window) total kv-head count, if it has one.

    Mirrors ModelConfig.get_swa_num_kv_heads' source selection, but returns
    the TOTAL rather than the per-GPU share: the DCP replication condition is
    about how many ranks share one kv head, so it needs the undivided count.
    Returns None for a model with a single kv-head base, which is every
    non-hybrid class and most hybrid ones.
    """
    hf = getattr(model_config, "hf_text_config", None)
    if hf is None:
        return None
    swa = getattr(hf, "swa_num_key_value_heads", None)
    if swa:
        return swa
    other = getattr(hf, "attention_other_setting", None)  # step3p5
    if isinstance(other, dict):
        return other.get("num_attention_groups")
    return None


def reject_unsupported_dcp_geometry(
    dcp_size: int,
    attn_tp_size: int,
    total_kv_heads: int,
    *,
    uneven_plan: bool,
    weighted_tokens: bool,
    weightless_kv: bool,
    swa_kv_heads: Optional[int] = None,
    mla: bool = False,
    speculative: bool = False,
    speculative_tree: bool = False,
    sliding_window: bool = False,
    swa_hybrid_dcp: bool = False,
) -> None:
    """Boot-time gate for every DCP geometry the Triton backend cannot serve.

    Pure function of the geometry so the constructor's decision is testable
    without a device, a process group, or a ModelRunner.

    THREE branches, in order:

    (1) THE UNEVEN-DCP LANE (a ``--rank-tp-ratio`` plan is installed, so the
        KV pool is token-sharded with the FULL replicated kv-head set per
        slot). #169.1 refused this outright, because the Triton backend then
        implemented exactly one owner rule -- the EVEN modulo one -- on both
        sides, and never gathered the kv heads before the write. #173 ports
        the fork's weighted machinery over from flashinfer:
          * write: ``dcp_weighted_write_slots`` + the fused kv-head gather;
          * read:  ``build_dcp_weighted_kv_indices``;
          * both from ``layers/dcp/owner.py``, the SAME functions the
            flashinfer path calls, so the two backends cannot drift apart.
        So the lane is now SERVED, and this branch only rejects the pieces of
        it that still have no Triton twin (see the sub-reasons below). #180
        then ported flashinfer's M4 verify split, so CHAIN speculative decoding
        (topk == 1) left that list; only the TREE-masked verify is still
        refused. The
        replication arithmetic of branch (3) deliberately does NOT apply here:
        under this lane every rank's pool row already holds every kv head, so
        the condition it checks is satisfied by construction of the pool.
        #96 (Stage B) additionally serves SWA-HYBRID models on this lane, but
        only under the Stage-B preconditions carried in ``swa_hybrid_dcp`` --
        see the sliding-window sub-reason below.

    (2) A TOKEN VECTOR WITHOUT A PLAN. ``uneven_dcp_active`` sizes the pool by
        the weighted rule while ``uneven_dcp_kv_replicated`` decides whether
        its rows carry the full kv-head set. A token vector with no plan is a
        half-installed state: weighted pool sizing, head-sharded rows. Neither
        backend serves it; refuse rather than pick one of the two layouts.
        Since #182 a BOOT can no longer reach this branch: the scheduler now
        resolves the token vector whenever one is set, and
        ``resolve_cp_token_ratios`` rejects "vector without a plan" earlier and
        with a message about the vector rather than about this backend. The
        branch stays as the backstop for any other way a vector gets installed
        (the post-profiling phase-2 install, tests) -- it keys on the INSTALLED
        vector, not on the flag, which is why it remains correct.

    (3) EVEN DCP WITHOUT KV-HEAD REPLICATION, ON ANY OF THE MODEL'S KV BASES.
        The even-DCP decode gathers the whole DCP group's q heads and attends
        them against THIS RANK'S local kv-head shard; the kernel remaps
        q-head -> kv-head as ``cur_head // (gathered_q // local_kv)``. Correct
        only when every rank of the group holds the same FULL kv-head set:
        ``tp_size // total_kv_heads >= dcp_size`` (consecutive ranks share a kv
        head, DCP groups are consecutive tp slices) -- the geometry of the
        path's origin (Qwen3.5-397B, TP=8/DCP=2, kv=4) and its only CI case.
        Measured outside it: Qwen2.5-1.5B (q12/kv2) TP=2/DCP=2 -> mojibake on
        NCCL and barlink alike; the same model at TP=4/DCP=2 (replicas 2 >= 2) is
        coherent; a 1-token prompt still garbles, so it is the head geometry
        and not the token-ownership layout.

    dcp_size <= 1 is inert, which is what keeps the default path untouched.
    """
    if dcp_size <= 1:
        return

    if uneven_plan:
        # ---- (1) the ported lane: reject only what has no Triton twin ----
        reasons = []
        if weightless_kv:
            reasons.append(
                "the weightless-KV fast lane is on (per-rank head counts "
                "[all, 0, 0, ...]), whose block-decode / host-spill / "
                "broadcast-K,V dispatch exists only in the flashinfer backend"
            )
        if mla:
            reasons.append(
                "the model is MLA, and the Triton MLA decode is a different "
                "kernel family that the uneven-DCP wiring here has never been "
                "run against"
            )
        if speculative and speculative_tree:
            reasons.append(
                "a TREE-masked speculative verify is on (--speculative-eagle-"
                "topk > 1, or the DFLASH tree-verify door). Chain verify "
                "(topk == 1) IS served here since #180, but the tree mask is "
                "not: its row stride is the GLOBAL prefix length, which stops "
                "describing the rows the kernel walks once the prefix is "
                "owner-sharded -- and #76 measured the flashinfer tree twin "
                "non-deterministic under this very lane, so there is no "
                "correct version to port. Use --speculative-eagle-topk 1"
            )
        if sliding_window and not swa_hybrid_dcp:
            # STAGE B (#96) OPENS THIS ONE, AND ONLY UNDER ITS PRECONDITIONS.
            # ``swa_hybrid_dcp`` is swa_hybrid_dcp_lane(...): an SWA-HYBRID model
            # (both sliding-window and global layers), cap-sized SWA pool, a
            # --rank-tp-ratio plan, target worker. Then the token split is
            # applied ONLY to the global full-attention layers -- the
            # sliding-window layers keep their unsharded local path, so nothing
            # ever has to causally mask a sparse owned-slot subset, which is
            # what this refusal was about. Every other windowed configuration
            # (pure-SWA model, ratio-sized SWA pool, draft worker) keeps the
            # refusal verbatim.
            reasons.append(
                "a sliding window is configured, and the DCP extend path "
                "cannot causally mask a sparse owned-slot subset"
            )
        if reasons:
            raise ValueError(
                f"Triton attention cannot serve --dcp-size {dcp_size} under "
                f"the uneven-DCP (token-sharded, replicated-kv-head) lane: "
                + "; ".join(reasons)
                + ". The weighted owner rule itself IS supported here (#173); "
                "these are the parts of the lane that are not. Use "
                "--attention-backend flashinfer, or drop the listed feature."
            )
        return

    if weighted_tokens:
        # ---- (2) half-installed uneven state ----
        raise ValueError(
            f"Triton attention cannot serve --dcp-size {dcp_size}: a "
            f"non-uniform DCP token vector is installed (weighted owner rule, "
            f"virtual block = sum of the token ratios) but no --rank-tp-ratio "
            f"shard plan is, so the KV pool is sized by the weighted rule "
            f"while its rows carry only this rank's kv-head shard. That "
            f"combination has no correct read: the weighted rule assumes the "
            f"FULL replicated kv-head set per row. Install a --rank-tp-ratio "
            f"plan (the supported uneven-DCP lane), or drop the token vector."
        )

    if weightless_kv:
        raise ValueError(
            f"Triton attention cannot serve --dcp-size {dcp_size}: the "
            f"weightless-KV fast lane is on, so the per-rank q/kv head counts "
            f"are [all, 0, 0, ...] instead of an equal split, and its "
            f"block-decode / host-spill / broadcast-K,V dispatch exists only "
            f"in the flashinfer backend. Use --attention-backend flashinfer, "
            f"or drop the fast lane."
        )

    # HYBRID CLASSES HAVE MORE THAN ONE KV-HEAD BASE. A hybrid/SWA model can
    # declare a second count (swa_num_key_value_heads, or step3p5's
    # attention_other_setting.num_attention_groups) for its sliding-window
    # layers, and the replication condition has to hold for EVERY layer kind
    # the model runs -- a rank whose SWA kv shard is not the full set attends
    # the gathered q heads against the wrong kv head just as a full-attention
    # rank would. The binding base is the LARGEST one, because more kv heads
    # means fewer replicas per head; reading only get_total_num_kv_heads()
    # (the full-attention base) would leave exactly the hybrid class -- the
    # one whose formal correctness under even DCP was the open question --
    # able to slip past the guard whenever its SWA base is the larger of the
    # two. max() can only make the condition stricter, never more permissive,
    # so a model with one base or a smaller SWA base keeps its old verdict.
    bases = [b for b in (total_kv_heads, swa_kv_heads) if b]
    binding_kv = max(bases) if bases else 0
    replicas = attn_tp_size // binding_kv if binding_kv else 0
    if replicas < dcp_size:
        which = (
            " (the sliding-window base, larger than the full-attention "
            f"{total_kv_heads})"
            if swa_kv_heads and swa_kv_heads > (total_kv_heads or 0)
            else ""
        )
        raise ValueError(
            f"Triton attention with --dcp-size {dcp_size} "
            f"requires the kv heads to be replicated across each DCP "
            f"group: tp_size // total_kv_heads >= dcp_size, but "
            f"{attn_tp_size} // {binding_kv}{which} = {replicas} < "
            f"{dcp_size}. Every rank would attend the gathered "
            f"q heads against a DIFFERENT kv-head shard and the "
            f"output is silently wrong from the first decode token. "
            f"Use --attention-backend flashinfer, raise tp so that "
            f"tp // kv_heads >= dcp_size, or drop --dcp-size."
        )


def replicated_kv_reindex(
    q_head_offset: int, local_q: int, local_kv: int, global_gqa: int
) -> Optional[list]:
    """REPLICATED-KV (#105): local kv slot -> the GLOBAL kv head its q group
    attends, or None when that is already the identity.

    Pure integer function of the head geometry so it is testable without a
    device or a ModelRunner; ``_replicated_kv_ragged_reindex`` is the cached
    tensor wrapper around it.

    The uniform-GQA extend kernel derives its grouping from the tensors it is
    handed (``kv_group_num = q.shape[1] // k.shape[1]``). Under REPLICATED-KV a
    rank holds a q-head SLICE but ALL kv heads, so local grouping
    ``local_q // local_kv`` is not the global ``total_q // total_kv``. Gathering
    the kv heads with the returned index list makes the uniform kernel
    reproduce the global mapping.

    Raises when one local slot's q group spans two global kv heads: the
    uniform-GQA kernel has no way to express that, and quietly picking one of
    the two would be a silently wrong answer rather than a crash.
    """
    idx = []
    local_gqa = local_q // local_kv
    for m in range(local_kv):
        lo = q_head_offset + m * local_gqa
        hi = q_head_offset + (m + 1) * local_gqa - 1
        g = lo // global_gqa
        if hi // global_gqa != g:
            raise ValueError(
                f"REPLICATED-KV current-chunk attention (#105): this rank's q "
                f"heads (offset {q_head_offset}, {local_q} heads over "
                f"{local_kv} local kv slots) straddle a global kv-head boundary "
                f"(global GQA group size {global_gqa}); the uniform-GQA Triton "
                f"extend kernel cannot represent this split. Choose a "
                f"--rank-tp-ratio whose q-unit boundaries align with kv-head "
                f"boundaries."
            )
        idx.append(g)
    return None if idx == list(range(local_kv)) else idx


def logit_capping_mod(logit_capping_method, logit_cap):
    # positive logit_cap -> tanh cap
    if logit_capping_method == "tanh":
        return logit_cap
    else:
        raise ValueError()


@dataclass
class ForwardMetadata:
    attn_logits: torch.Tensor
    attn_lse: torch.Tensor
    max_extend_len: int
    num_kv_splits: torch.Tensor
    kv_indptr: torch.Tensor
    kv_indices: torch.Tensor
    qo_indptr: torch.Tensor
    custom_mask: torch.Tensor
    mask_indptr: torch.Tensor
    # Sliding window
    window_kv_indptr: torch.Tensor
    window_kv_indices: torch.Tensor
    window_num_kv_splits: torch.Tensor
    window_kv_offsets: torch.Tensor
    # Separate attn_logits for SWA layers when v_head_dim differs
    swa_attn_logits: Optional[torch.Tensor] = None
    # full->SWA translated out_cache_loc (SWA KV-store write target)
    swa_out_cache_loc: Optional[torch.Tensor] = None
    # PHYSICAL full-attn write target for the unified pool (eager: translated tensor;
    # cuda-graph: capture-stable buffer view). None for non-unified pools.
    out_cache_loc_full_physical: Optional[torch.Tensor] = None


class TritonAttnBackend(AttentionBackend):
    # CUDA-graph replay rebuilds metadata from preallocated kv_indptr/kv_indices
    # buffers; it never reads seq_lens_cpu / seq_lens_sum.
    needs_cpu_seq_lens: bool = False

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        kv_indptr_buf: Optional[torch.Tensor] = None,
    ):
        # Lazy import to avoid the initialization of cuda context
        from sglang.kernels.ops.attention.decode_attention import (
            decode_attention_fwd,
        )
        from sglang.kernels.ops.attention.extend_attention import (
            build_unified_kv_indices,
            extend_attention_fwd,
            extend_attention_fwd_unified,
        )
        from sglang.kernels.ops.attention.verify_splitkv import (
            verify_splitkv_fwd,
        )

        super().__init__()

        self.decode_attention_fwd = torch.compiler.disable(decode_attention_fwd)
        self.extend_attention_fwd = torch.compiler.disable(extend_attention_fwd)
        self.extend_attention_fwd_unified = torch.compiler.disable(
            extend_attention_fwd_unified
        )
        self.build_unified_kv_indices = torch.compiler.disable(build_unified_kv_indices)
        # Split-KV EAGLE-verify kernel; enabled below once topk is known (valid only at topk == 1).
        self.verify_splitkv_fwd = torch.compiler.disable(verify_splitkv_fwd)

        # Parse args
        self.skip_prefill = skip_prefill
        max_bs = model_runner.req_to_token_pool.size
        self.sliding_window_size = model_runner.sliding_window_size
        self.req_to_token_pool = model_runner.req_to_token_pool
        self.token_to_kv_pool = model_runner.token_to_kv_pool
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.token_to_kv_pool_allocator = model_runner.token_to_kv_pool_allocator
        self.use_sliding_window_kv_pool = isinstance(self.token_to_kv_pool, SWAKVPool)
        # Lets the Triton wrappers specialize on PAGE_SIZE; page_size=1 is
        # byte-identical to the slot-based envelope.
        self.page_size = getattr(model_runner, "page_size", 1) or 1
        # Unified pool v2p hook (None = no-op): req_to_token holds VIRTUAL ids but
        # kernels need PHYSICAL. Applied eagerly so the captured graph has no translate.
        self._translate_kv_loc = getattr(
            self.token_to_kv_pool_allocator, "translate_kv_loc", None
        )
        self.num_draft_tokens = model_runner.server_args.speculative_num_draft_tokens
        self.speculative_num_steps = model_runner.server_args.speculative_num_steps
        self.topk = model_runner.server_args.speculative_eagle_topk or 0
        # Split-KV verify is bit-equivalent only for a pure-causal chain (topk==1)
        # and is gfx95-only; else fall back to extend_attention_fwd.
        self.use_verify_splitkv = (
            is_gfx95_supported()
            and envs.SGLANG_ENABLE_SPLITKV_VERIFY.get()
            and self.topk == 1
        )
        self.use_mla = model_runner.model_config.attention_arch == AttentionArch.MLA
        self.dcp_size = get_parallel().attn_dcp_size
        self.dcp_rank = get_parallel().attn_dcp_rank
        from sglang.srt.distributed.utils import (
            attn_kv_replicated,
            tp_partition_sizes,
            uneven_dcp_active,
            uneven_dcp_kv_replicated,
            weightless_kv_active,
        )

        _attn_tp_size = get_parallel().attn_tp_size
        _total_kv = model_runner.model_config.get_total_num_kv_heads()
        self.is_draft_worker = bool(getattr(model_runner, "is_draft_worker", False))
        # THE UNEVEN-DCP LANE (#173). A --rank-tp-ratio plan spanning the whole
        # TP group means the KV pool is TOKEN-sharded with the FULL replicated
        # kv-head set per row. The draft/NEXTN worker is excluded by design and
        # exactly as in the flashinfer backend: its pool keeps the full token
        # context with LOCAL heads, so DCP is simply OFF for that backend
        # instance -- and since every DCP branch here keys on `dcp_size > 1`,
        # turning it off means setting dcp_size to 1 for this object. (Reading
        # the draft's full-context pool through the owner rule would compact
        # indices that were never compacted.)
        _uneven_plan = uneven_dcp_kv_replicated(self.dcp_size)
        if _uneven_plan and self.is_draft_worker:
            self.dcp_size = 1
            self.dcp_rank = 0
        self.uneven_dcp = _uneven_plan and not self.is_draft_worker
        # WEIGHTED owner rule: a non-uniform token vector is installed, so this
        # rank owns the contiguous virtual-block offset range [cp_lo, cp_hi) of
        # every block of cp_S slots instead of the residue == dcp_rank. Bounds
        # come from the shared helper, so the read side
        # (build_dcp_weighted_kv_indices) and the write side
        # (dcp_weighted_write_slots) cannot end up with different ones.
        self.uneven_dcp_weighted = self.uneven_dcp and uneven_dcp_active(self.dcp_size)
        self.cp_S = self.cp_lo = self.cp_hi = self.cp_ratio = 0
        if self.uneven_dcp_weighted:
            (
                self.cp_S,
                self.cp_lo,
                self.cp_hi,
                self.cp_ratio,
            ) = dcp_weighted_owner_bounds(self.dcp_size, self.dcp_rank)
            # #297: the four fields are an init-time snapshot; register so a
            # runtime KV reshard's cutover can refresh this instance.
            from sglang.srt.layers.dcp.owner import register_owner_bounds_consumer

            register_owner_bounds_consumer(self)
        # NOT _plan_aware_num_q_heads(...) * dcp_size: that helper is already
        # plan-aware, so multiplying double-counts and UNDER-sizes the decode
        # workspaces on the smaller ranks under an uneven plan. See
        # _plan_aware_dcp_gathered_q_heads.
        self.num_head = _plan_aware_dcp_gathered_q_heads(
            model_runner.model_config, self.dcp_size
        )
        # Kept for the DCP head collectives, which need the group's per-rank
        # q-head counts and not just this rank's (_dcp_group_q_head_counts).
        self.dcp_model_config = model_runner.model_config
        self.num_kv_head = model_runner.model_config.get_num_kv_heads(_attn_tp_size)
        # KV-HEAD BOOKKEEPING FOR THE UNEVEN LANE. Under it the KV pool rows
        # hold the FULL total_num_kv_heads on every rank (see
        # model_runner_kv_cache_mixin: `_hybrid_kv_head_num =
        # get_total_num_kv_heads()` whenever uneven_dcp_kv_replicated), so the
        # split-KV schedule must be sized for that count and not for this
        # rank's projection share -- they differ exactly when the kv heads are
        # head-sharded rather than replicated.
        self.dcp_kv_replicated_heads = False
        self.dcp_kv_head_counts = None
        self._repl_kv_reindex_cache = {}
        self.dcp_full_qo_heads = model_runner.model_config.num_attention_heads
        self.dcp_full_kv_heads = _total_kv
        if self.uneven_dcp:
            # kv < tp -> every rank projects ALL kv heads itself (replicated
            # k/v weights), so the per-layer kv-head gather is a no-op and is
            # skipped. This is the 27B/35B Nordstern case and it issues NO
            # extra collective at all.
            self.dcp_kv_replicated_heads = attn_kv_replicated(_attn_tp_size, _total_kv)
            self.dcp_kv_head_counts = (
                [_total_kv] * _attn_tp_size
                if self.dcp_kv_replicated_heads
                else tp_partition_sizes(_total_kv, _attn_tp_size, units=_total_kv)
            )
            self.num_kv_head = _total_kv
        # SWA-HYBRID DCP LANE (#96, Stage B): the ~10 GLOBAL full-attention
        # layers are token-sharded by the owner rule above; the ~50
        # sliding-window layers keep their unsharded local path (each rank holds
        # every in-window position of its own kv-head shard). The predicate is
        # the shared swa_hybrid_dcp_lane(), the SAME function the KV-pool sizing
        # uses, so the pool cannot be sized for one mode while the backend
        # dispatches for the other. Configuration-only inputs, hence identical
        # on every rank -- which is what makes the per-layer dispatch built on it
        # safe to gate COLLECTIVES with.
        _mc = model_runner.model_config
        self.swa_hybrid_dcp = swa_hybrid_dcp_lane(
            is_hybrid_swa=bool(getattr(model_runner, "is_hybrid_swa", False)),
            uneven_plan=self.uneven_dcp,
            is_draft_worker=self.is_draft_worker,
            num_full_layers=len(getattr(_mc, "full_attention_layer_ids", None) or []),
            num_swa_layers=len(getattr(_mc, "swa_attention_layer_ids", None) or []),
            swa_pool_sizing_capped=(
                model_runner.server_args.swa_pool_sizing == "cap"
                or bool(model_runner.server_args.disable_radix_cache)
            ),
        )
        # DCP GEOMETRY GUARD -- see reject_unsupported_dcp_geometry for the
        # full reasoning of the three branches (the uneven lane and what of it
        # is still unserved; a token vector without a plan; even DCP needing
        # kv-head replication across the group). Everything the decision
        # depends on is read here and passed by value, so the rule itself stays
        # testable without a device.
        reject_unsupported_dcp_geometry(
            self.dcp_size,
            _attn_tp_size,
            _total_kv,
            uneven_plan=uneven_dcp_kv_replicated(self.dcp_size),
            weighted_tokens=uneven_dcp_active(self.dcp_size),
            weightless_kv=weightless_kv_active(),
            swa_kv_heads=total_swa_kv_heads(model_runner.model_config),
            mla=self.use_mla,
            speculative=bool(self.num_draft_tokens or self.speculative_num_steps),
            # The tree predicate is NOT re-derived here. dcp_verify_mask_mode
            # is the one place the "which draft->draft mask does a verify need"
            # question is answered, and it knows about BOTH doors onto a tree
            # mask (topk > 1 and --speculative-dflash-tree-verify) -- a second
            # copy is how the second door got missed the first time.
            speculative_tree=(
                dcp_verify_mask_mode(
                    self.topk,
                    bool(
                        getattr(
                            model_runner.server_args,
                            "speculative_dflash_tree_verify",
                            False,
                        )
                    ),
                )
                == "tree"
            ),
            sliding_window=(
                self.sliding_window_size is not None and self.sliding_window_size > 0
            ),
            swa_hybrid_dcp=self.swa_hybrid_dcp,
        )
        # The decode kernel's "// Lv" stride trick requires attn_logits.shape[-1]
        # to exactly match the layer's v_head_dim, so hybrid SWA models with
        # differing SWA/full v_head_dim need a second buffer for SWA layers.
        full_v_head_dim = model_runner.model_config.v_head_dim
        swa_v_head_dim = model_runner.model_config.swa_v_head_dim
        if self.sliding_window_size is not None and swa_v_head_dim != full_v_head_dim:
            self.v_head_dim = full_v_head_dim
            self.swa_v_head_dim = swa_v_head_dim
        elif (
            model_runner.hybrid_gdn_config is not None
            or model_runner.kimi_linear_config is not None
            or model_runner.linear_attn_model_spec is not None
        ):
            # For hybrid linear models, layer_id = 0 may not be full attention
            self.v_head_dim = model_runner.token_to_kv_pool.get_v_head_dim()
            self.swa_v_head_dim = None
        else:
            self.v_head_dim = model_runner.token_to_kv_pool.get_value_buffer(0).shape[
                -1
            ]
            self.swa_v_head_dim = None
        self.max_context_len = model_runner.model_config.context_len
        self.device = model_runner.device
        self.device_core_count = get_device_core_count(model_runner.gpu_id)
        self.static_kv_splits = get_bool_env_var(
            "SGLANG_TRITON_DECODE_ATTN_STATIC_KV_SPLITS", "false"
        )
        self.max_kv_splits = model_runner.server_args.triton_attention_num_kv_splits
        if self.use_mla and not _is_xpu:
            self.max_kv_splits = _mla_decode_kv_splits_cap(
                self.max_kv_splits,
                self.device_core_count,
                self.max_context_len,
            )
            if _is_gfx942:
                # gfx942's 304 CUs round up to 512 splits, doubling the persistent
                # fp32 attn_logits buffer to ~4 GiB on Kimi-K2.6 and faulting in
                # ROCm graph replay; pin to 256 to match validated gfx950 behavior.
                self.max_kv_splits = min(self.max_kv_splits, 256)
        if _is_cuda and _has_pdl_probe:
            self.use_pdl = is_arch_support_pdl()
        else:
            # No probe available => no PDL. Correct for sm75, which predates
            # the feature entirely, and unchanged on sm80+ where the probe is
            # present and answers for itself.
            self.use_pdl = False

        self.allow_bidirectional_attention_in_extend = (
            cuda_graph_fully_disabled()
            and model_runner.server_args.chunked_prefill_size == -1
        )

        self.enable_deterministic = (
            model_runner.server_args.enable_deterministic_inference
        )

        if self.enable_deterministic:
            # Fixed split tile size for batch invariance
            self.split_tile_size = get_int_env_var(
                "SGLANG_TRITON_DECODE_SPLIT_TILE_SIZE", 256
            )
            self.static_kv_splits = False
        else:
            self.split_tile_size = (
                model_runner.server_args.triton_attention_split_tile_size
            )

        if self.split_tile_size is not None:
            self.max_kv_splits = (
                self.max_context_len + self.split_tile_size - 1
            ) // self.split_tile_size

        assert not (
            model_runner.sliding_window_size is not None
            and model_runner.model_config.is_encoder_decoder
        ), "Sliding window and cross attention are not supported together"

        # TODO(Jianan Ji): verify behavior when kv_indptr_buf is provided and sliding window is enabled
        if kv_indptr_buf is None:
            self.kv_indptr = torch.zeros(
                (max_bs + 1,), dtype=torch.int32, device=model_runner.device
            )
        else:
            self.kv_indptr = kv_indptr_buf

        # Sliding window may need a second buffer for interleaved attention types
        self.window_kv_indptr = None
        if self.sliding_window_size is not None and self.sliding_window_size > 0:
            if kv_indptr_buf is None:
                self.window_kv_indptr = torch.zeros(
                    (max_bs + 1,), dtype=torch.int32, device=model_runner.device
                )
            else:
                self.window_kv_indptr = torch.zeros_like(kv_indptr_buf)

        if not self.skip_prefill:
            self.qo_indptr = torch.zeros(
                (max_bs + 1,), dtype=torch.int64, device=model_runner.device
            )

            self.mask_indptr = torch.zeros(
                (max_bs + 1,), dtype=torch.int64, device=model_runner.device
            )

        self.forward_metadata: ForwardMetadata = None

        self.cuda_graph_custom_mask = None

    def refresh_dcp_owner_bounds(self) -> None:
        """#297 cutover hook: re-derive the cached weighted owner bounds from
        the freshly installed token vector (see the flashinfer twin for the
        full rationale). Idle-boundary only, never mid-forward."""
        if not self.uneven_dcp_weighted:
            return
        (
            self.cp_S,
            self.cp_lo,
            self.cp_hi,
            self.cp_ratio,
        ) = dcp_weighted_owner_bounds(self.dcp_size, self.dcp_rank)

    def get_num_kv_splits(
        self,
        num_kv_splits: torch.Tensor,
        seq_lens: torch.Tensor,
    ):
        num_token, num_seq = num_kv_splits.shape[0], seq_lens.shape[0]
        # NOTE(alcanderian): Considering speculative_decodeing,
        # num_kv_splits.shape[0] will be topk * real_num_token.
        # And the real_num_token is num_seq in decoding phase.
        num_group = num_token // num_seq

        assert (
            num_group * num_seq == num_token
        ), f"num_seq({num_seq}), num_token({num_token}), something goes wrong!"

        if (
            self.static_kv_splits or self.device_core_count <= 0
        ) and not self.enable_deterministic:
            num_kv_splits.fill_(self.max_kv_splits)
            return

        if self.split_tile_size is not None and self.enable_deterministic:
            if num_group > 1:
                expanded_seq_lens = seq_lens.repeat_interleave(num_group)
            else:
                expanded_seq_lens = seq_lens

            num_kv_splits[:] = (
                expanded_seq_lens + self.split_tile_size - 1
            ) // self.split_tile_size
            return

        if num_seq < 256:
            SCHEDULE_SEQ = 256
        else:
            SCHEDULE_SEQ = triton.next_power_of_2(num_seq)

        get_num_kv_splits_triton[(1,)](
            num_kv_splits,
            seq_lens,
            num_seq,
            num_group,
            self.num_head,
            self.num_kv_head,
            self.max_kv_splits,
            self.device_core_count,
            MAX_NUM_SEQ=SCHEDULE_SEQ,
        )

    def _dcp_lens(self, lens: torch.Tensor, start: Optional[torch.Tensor] = None):
        return get_dcp_lens(lens, self.dcp_size, self.dcp_rank, start)

    def _dcp_kv_indices(
        self,
        req_pool_indices: torch.Tensor,
        lens: torch.Tensor,
        kv_indptr: torch.Tensor,
        kv_indices: Optional[torch.Tensor] = None,
        kv_start_idx: Optional[torch.Tensor] = None,
        lens_cpu: Optional[Union[Sequence[int], torch.Tensor]] = None,
        lens_sum: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Build per-DCP-rank sharded KV indptr/indices. eager passes kv_indices=None
        # (fresh tensor); cuda-graph passes an address-stable buffer to fill in place.
        #
        # #618/#623: lens_cpu is the host mirror of `lens` and lens_sum its
        # independently known host sum (the staleness check). Both default to
        # None, which keeps the blocking device read.
        #
        # #629: the cuda-graph replay-prep callers now supply the mirror too.
        # #623 left them on the device read on the grounds that no mirror was in
        # scope, which was true of the helpers' own signatures and false of the
        # entry point: init_forward_metadata_out_graph holds the ForwardBatch.
        # None still means "keep the old read", which is what a gpu_only batch
        # (no host mirror at all) and a mirror that fails the staleness check
        # both fall back to.
        if self.uneven_dcp_weighted:
            return self._dcp_weighted_kv_indices(
                req_pool_indices,
                lens,
                kv_indptr,
                kv_indices,
                kv_start_idx,
                lens_cpu=lens_cpu,
                lens_sum=lens_sum,
            )
        dcp_lens = self._dcp_lens(lens, kv_start_idx)
        kv_indptr[1 : len(req_pool_indices) + 1] = torch.cumsum(dcp_lens, dim=0)
        kv_indptr = kv_indptr[: len(req_pool_indices) + 1]
        if kv_indices is None:
            # #618: unbounded D2H unless the host mirror covers it.
            n_dcp = dcp_host_even_total(
                lens_cpu,
                self.dcp_size,
                self.dcp_rank,
                start=kv_start_idx,
                expected_sum=lens_sum,
            )
            kv_indices = torch.empty(
                (n_dcp if n_dcp is not None else int(dcp_lens.sum().item())),
                dtype=torch.int64,
                device=self.device,
            )
        create_triton_kv_indices_for_dcp_triton[(len(req_pool_indices),)](
            self.req_to_token,
            req_pool_indices,
            dcp_lens,
            kv_indptr,
            kv_start_idx,
            kv_indices,
            self.req_to_token.stride(0),
            self.dcp_size,
            self.dcp_rank,
        )
        return kv_indptr, kv_indices, dcp_lens

    def _dcp_weighted_kv_indices(
        self,
        req_pool_indices: torch.Tensor,
        lens: torch.Tensor,
        kv_indptr: torch.Tensor,
        kv_indices: Optional[torch.Tensor],
        kv_start_idx: Optional[torch.Tensor],
        lens_cpu: Optional[Union[Sequence[int], torch.Tensor]] = None,
        lens_sum: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """WEIGHTED owner rule read side (#173), same contract as the even one.

        ``build_dcp_weighted_kv_indices`` is the function the flashinfer backend
        calls; sharing it is the point -- read and write, and now flashinfer and
        Triton, all derive the compact slot from the SAME expression, so a token
        cannot be fetched from a row it was never stored in.

        BUFFER CONTRACT (the reason this is not a two-liner). The helper returns
        a FRESH tensor, while a captured CUDA graph reads the address-stable
        ``cuda_graph_kv_indices`` whose pointer was frozen at capture. Returning
        the fresh tensor to a replay would leave the graph reading whatever the
        buffer held at capture time -- a silent wrong-context decode, not a
        crash. So when the caller hands in a buffer, the result is copied INTO
        it and that same buffer object is returned; when it hands in None
        (eager), a fresh exactly-sized tensor is returned. Which of the two came
        back is decided by the caller's argument, never by anything computed in
        here.
        """
        bs = len(req_pool_indices)
        kv_indptr, compact = build_dcp_weighted_kv_indices(
            self.req_to_token,
            req_pool_indices,
            lens,
            kv_indptr,
            kv_start_idx,
            self.cp_S,
            self.cp_lo,
            self.cp_hi,
            self.cp_ratio,
            req_to_token_stride=self.req_to_token.stride(0),
            # #623: the Triton twin of the flashinfer wiring. Without this the
            # builder falls back to int(full_indptr[bs].item()), the unbounded
            # blocking D2H inside the collective window. None -> old read.
            total_tokens=dcp_host_total_tokens(lens_cpu, lens_sum),
        )
        # Same dtype as the even branch's get_dcp_lens result, because both feed
        # the same get_num_kv_splits kernel and a Triton pointer's element type
        # is inferred from the tensor it is handed.
        owned_lens = (kv_indptr[1 : bs + 1] - kv_indptr[:bs]).to(lens.dtype)
        n = compact.numel()
        if kv_indices is None:
            # int64 to match what the even branch hands the Triton kernels.
            # A rank can legitimately own ZERO rows of the whole batch (a short
            # prefix and a small token ratio); the attention kernels are then
            # driven by an all-zero kv_indptr and never dereference the index
            # tensor, but a 0-element tensor has no storage to take a pointer
            # from, so keep one dummy row.
            out = torch.zeros(max(n, 1), dtype=torch.int64, device=self.device)
            if n:
                out[:n].copy_(compact)
            return kv_indptr, out, owned_lens
        if n > kv_indices.numel():
            raise ValueError(
                f"weighted-DCP kv_indices buffer overflow: this rank owns {n} "
                f"cache rows for the current batch but the capture-stable "
                f"buffer holds {kv_indices.numel()}. The buffer is sized for "
                f"max_num_tokens * max_context_len, so this means the owned "
                f"share exceeded the global context -- a broken token ratio "
                f"vector or a stale cp_S."
            )
        if n:
            kv_indices[:n].copy_(compact)
        return kv_indptr, kv_indices, owned_lens

    def _fill_kv_indptr_and_indices(
        self,
        bs: int,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        kv_indices: torch.Tensor,
    ) -> torch.Tensor:
        kv_indptr = self.kv_indptr[: bs + 1]
        kv_indptr[1:] = torch.cumsum(seq_lens, dim=0)
        create_flashinfer_kv_indices_triton[(bs,)](
            self.req_to_token,
            req_pool_indices,
            seq_lens,
            kv_indptr,
            None,
            kv_indices,
            self.req_to_token.stride(0),
        )
        return kv_indptr

    def _update_decode_kv_buffers(
        self,
        bs: int,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens_cpu: Optional[Union[Sequence[int], torch.Tensor]] = None,
        seq_lens_sum: Optional[int] = None,
    ):
        """Fill KV (and SWA) cuda-graph buffers for decode/idle mode.

        Returns ``(kv_indptr, window_kv_indptr, window_kv_lens, num_kv_splits_lens)``
        where ``window_kv_lens`` is ``None`` when sliding-window is disabled and
        ``num_kv_splits_lens`` is the per-request length used to size kv splits
        (per-DCP-rank length clamped to >=1 when DCP is enabled, full seq_lens
        otherwise).

        ``seq_lens_cpu`` / ``seq_lens_sum`` are the #629 host mirror of
        ``seq_lens``, sliced to ``bs`` alongside it.
        """
        seq_lens = seq_lens[:bs]
        req_pool_indices = req_pool_indices[:bs]
        if seq_lens_cpu is not None:
            seq_lens_cpu = seq_lens_cpu[:bs]
        if self.dcp_size > 1:
            # DCP: per-rank sharded; write into the same cuda-graph buffers
            # _build_cuda_graph_forward_metadata reads back.
            _, _, dcp_seq_lens = self._dcp_kv_indices(
                req_pool_indices,
                seq_lens,
                self.kv_indptr,
                self.cuda_graph_kv_indices,
                None,
                # #629: the eager decode twin's mirror, reaching the replay-prep
                # fill. Without it this site sizes kv_indices from an unbounded
                # blocking D2H inside the collective window.
                lens_cpu=seq_lens_cpu,
                lens_sum=seq_lens_sum,
            )
            kv_indptr = self.kv_indptr[: bs + 1]
            num_kv_splits_lens = dcp_seq_lens.clamp_min(1)
        else:
            kv_indptr = self._fill_kv_indptr_and_indices(
                bs, seq_lens, req_pool_indices, self.cuda_graph_kv_indices
            )
            # Unified pool: VIRTUAL ids written here are translated to PHYSICAL in
            # init_forward_metadata_out_graph (replay-prep) so the captured graph
            # carries zero translate nodes.
            num_kv_splits_lens = seq_lens
        window_kv_indptr = self.window_kv_indptr
        window_kv_lens = None
        if self.sliding_window_size is not None and self.sliding_window_size > 0:
            # Unified pool: leave the window VIRTUAL too (translated alongside the
            # full kv_indices later); baseline SWA keeps the eager window translate.
            window_kv_indptr, _, window_kv_lens, _ = update_sliding_window_buffer(
                self.window_kv_indptr,
                self.req_to_token,
                self.sliding_window_size,
                seq_lens,
                req_pool_indices,
                bs,
                token_to_kv_pool=self.token_to_kv_pool,
                window_kv_indices=self.cuda_graph_window_kv_indices,
                skip_full_to_swa_translation=(self._translate_kv_loc is not None),
            )
        return kv_indptr, window_kv_indptr, window_kv_lens, num_kv_splits_lens

    def _update_target_verify_buffers(
        self,
        bs: int,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        spec_info,
        seq_lens_cpu: Optional[Union[Sequence[int], torch.Tensor]] = None,
        seq_lens_sum: Optional[int] = None,
    ):
        """Fill all cuda-graph buffers for target_verify mode.

        ``seq_lens_cpu`` / ``seq_lens_sum`` are the #629 host mirror of
        ``seq_lens``.
        """
        qo_indptr = self.qo_indptr[: bs + 1]
        qo_indptr[: bs + 1] = torch.arange(
            0,
            (1 + bs) * self.num_draft_tokens,
            step=self.num_draft_tokens,
            dtype=torch.int32,
            device=self.device,
        )
        if self.dcp_size > 1:
            # M4 verify split under capture (#180). Same owner-rule builder as
            # eager, over the COMMITTED seq_lens, and handed the address-stable
            # buffer so the D3 contract applies: _dcp_weighted_kv_indices
            # copies INTO cuda_graph_kv_indices and returns that same object.
            # Returning a fresh tensor here would leave a replay reading
            # whatever the buffer held at capture time -- a silently wrong
            # verify context, not a crash.
            _reject_stale_verify_window(spec_info, self.num_draft_tokens)
            self._dcp_kv_indices(
                req_pool_indices[:bs],
                dcp_verify_paged_lens(seq_lens[:bs], self.num_draft_tokens),
                self.kv_indptr,
                self.cuda_graph_kv_indices,
                None,
                # #629: the same shared mirror function as the eager verify
                # twin. dcp_verify_paged_lens returns seq_lens itself, so
                # seq_lens_sum is that vector's sum and stays a valid
                # staleness check.
                lens_cpu=_verify_host_mirror(
                    None if seq_lens_cpu is None else seq_lens_cpu[:bs],
                    self.num_draft_tokens,
                ),
                lens_sum=seq_lens_sum,
            )
            kv_indptr = self.kv_indptr[: bs + 1]
        else:
            kv_indptr = self._fill_kv_indptr_and_indices(
                bs, seq_lens, req_pool_indices, self.cuda_graph_kv_indices
            )
        window_kv_indptr = self.window_kv_indptr
        window_kv_indices = None
        window_num_kv_splits = None
        window_kv_offsets = None
        if self.sliding_window_size is not None and self.sliding_window_size > 0:
            window_kv_indices = self.cuda_graph_window_kv_indices
            window_num_kv_splits = self.cuda_graph_window_num_kv_splits
            window_kv_offsets = self.cuda_graph_window_kv_offsets
            window_kv_indptr, window_kv_indices, _, window_kv_offsets[:bs] = (
                update_sliding_window_buffer(
                    self.window_kv_indptr,
                    self.req_to_token,
                    self.sliding_window_size,
                    seq_lens[:bs],
                    req_pool_indices,
                    bs,
                    token_to_kv_pool=self.token_to_kv_pool,
                    window_kv_indices=window_kv_indices,
                )
            )
        if self.dcp_size > 1:
            # Chain verify on the DCP lane carries no mask; see the eager twin.
            custom_mask = None
            mask_indptr = None
        else:
            custom_mask = self.cuda_graph_custom_mask
            if (
                spec_info is not None
                and getattr(spec_info, "custom_mask", None) is not None
            ):
                custom_mask[: spec_info.custom_mask.shape[0]] = spec_info.custom_mask
            else:
                custom_mask = None
            seq_mask_len = self.num_draft_tokens * (seq_lens + self.num_draft_tokens)
            mask_indptr = self.mask_indptr[: bs + 1]
            mask_indptr[1 : bs + 1] = torch.cumsum(seq_mask_len, dim=0)
        return (
            qo_indptr,
            kv_indptr,
            custom_mask,
            mask_indptr,
            window_kv_indptr,
            window_kv_indices,
            window_num_kv_splits,
            window_kv_offsets,
        )

    def _update_draft_extend_buffers(
        self,
        bs: int,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
    ):
        """Fill QO + KV cuda-graph buffers for draft_extend mode."""
        seq_lens = seq_lens[:bs]
        # V2 draft-extend fills num_draft_tokens per req; num_steps+1 only equals
        # that when topk == 1.
        num_tokens_per_bs = (
            self.num_draft_tokens
            if forward_mode.is_draft_extend_v2()
            else self.speculative_num_steps + 1
        )
        qo_indptr = self.qo_indptr[: bs + 1]
        qo_indptr[: bs + 1] = torch.arange(
            0,
            bs * num_tokens_per_bs + 1,
            step=num_tokens_per_bs,
            dtype=torch.int32,
            device=self.device,
        )
        # DRAFT_EXTEND_V2: kv_indptr/kv_indices cover only the prefix (extend K/V go
        # separately). Capture warmup lacks extend_seq_lens_tensor -> fall back to
        # zeros; clamp at 0 so padded rows (seq_lens==fill 1) don't go negative.
        if (
            spec_info is not None
            and getattr(spec_info, "extend_seq_lens_tensor", None) is not None
        ):
            extend_seq_lens = spec_info.extend_seq_lens_tensor[:bs].to(torch.int32)
        else:
            extend_seq_lens = torch.zeros(bs, dtype=torch.int32, device=seq_lens.device)
        kv_lens = torch.clamp(seq_lens - extend_seq_lens, min=0).to(torch.int32)
        kv_indptr = self._fill_kv_indptr_and_indices(
            bs, kv_lens, req_pool_indices, self.cuda_graph_kv_indices
        )
        return qo_indptr, kv_indptr, num_tokens_per_bs

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        bs = forward_batch.batch_size
        req_pool_indices = forward_batch.req_pool_indices
        seq_lens = forward_batch.seq_lens
        forward_mode = forward_batch.forward_mode
        spec_info = forward_batch.spec_info
        # #629: the host mirror of seq_lens, for the owner-rule index builders
        # the replay-prep fills drive. #623 left these fills on the unbounded
        # blocking D2H "because the cuda-graph callers have no mirror in scope"
        # -- true of _update_*_buffers, false HERE, where the ForwardBatch is.
        #
        # seq_lens_sum is the "mirror present" signal, the same test
        # _translate_cuda_graph_shared_pool_locs uses: it is None-preserving,
        # while seq_lens_cpu is a non-None but STALE slice on gpu_only batches.
        # It is forwarded as the expected_sum rather than merely consulted, so
        # dcp_host_lens re-checks the two against each other and refuses a
        # mirror that disagrees -- a stale vector would silently MIS-SIZE the
        # index buffer, which is worse than the sync it replaces. When the
        # signal is absent altogether, dcp_fresh_host_lens drops the slice, so
        # "no sum" degrades to the old device read and never to an UNCHECKED
        # mirror.
        seq_lens_sum = forward_batch.seq_lens_sum
        seq_lens_cpu = dcp_fresh_host_lens(forward_batch.seq_lens_cpu, seq_lens_sum)

        if in_capture:
            assert forward_batch.encoder_lens is None, "Not supported"
            # Multi-step spec decode: kv buffers come from spec_info, not the
            # cuda-graph pool, so replay is not involved.
            if forward_mode.is_decode_or_idle() and spec_info is not None:
                self.forward_metadata = ForwardMetadata(
                    attn_logits=self.cuda_graph_attn_logits,
                    attn_lse=self.cuda_graph_attn_lse,
                    max_extend_len=None,
                    num_kv_splits=self.cuda_graph_num_kv_splits,
                    kv_indptr=spec_info.kv_indptr,
                    kv_indices=spec_info.kv_indices,
                    qo_indptr=None,
                    custom_mask=None,
                    mask_indptr=None,
                    window_kv_indptr=self.window_kv_indptr,
                    window_kv_indices=None,
                    window_num_kv_splits=None,
                    window_kv_offsets=None,
                    swa_attn_logits=self.cuda_graph_swa_attn_logits,
                )
                return

            self._apply_cuda_graph_metadata(
                bs=bs,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                forward_mode=forward_mode,
                spec_info=spec_info,
                seq_lens_cpu=seq_lens_cpu,
                seq_lens_sum=seq_lens_sum,
            )
            out_cache_loc_full_physical = self._translate_cuda_graph_shared_pool_locs(
                forward_batch, bs
            )
            swa_out_cache_loc = self._fill_cuda_graph_swa_out_cache_loc(forward_batch)
            self.forward_metadata = self._build_cuda_graph_forward_metadata(
                bs,
                forward_mode,
                spec_info,
                swa_out_cache_loc,
                out_cache_loc_full_physical,
            )
        else:
            self._apply_cuda_graph_metadata(
                bs=bs,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                forward_mode=forward_mode,
                spec_info=spec_info,
                seq_lens_cpu=seq_lens_cpu,
                seq_lens_sum=seq_lens_sum,
            )
            # Metadata view is reused from capture; just refill the buffers.
            self._translate_cuda_graph_shared_pool_locs(forward_batch, bs)
            self._fill_cuda_graph_swa_out_cache_loc(forward_batch)

    def _fill_cuda_graph_swa_out_cache_loc(
        self, forward_batch: ForwardBatch
    ) -> Optional[torch.Tensor]:
        """Refill the SWA write-target buffer from live out_cache_loc, returning the
        [:n] view (None for non-SWA / multi-step draft) so the captured store reads
        fresh slots on replay."""
        if not self.use_sliding_window_kv_pool:
            return None
        out_cache_loc = forward_batch.out_cache_loc
        if (
            out_cache_loc is None
            or out_cache_loc.shape[0] > self.cuda_graph_swa_out_cache_loc.shape[0]
        ):
            return None
        n = out_cache_loc.shape[0]
        self.cuda_graph_swa_out_cache_loc[n:].zero_()
        self.cuda_graph_swa_out_cache_loc[:n].copy_(
            self.token_to_kv_pool.translate_loc_from_full_to_swa(out_cache_loc)
        )
        return self.cuda_graph_swa_out_cache_loc[:n]

    def _translate_cuda_graph_shared_pool_locs(
        self, forward_batch: ForwardBatch, bs: int
    ) -> Optional[torch.Tensor]:
        """Unified pool: eager v2p translate of the cuda-graph read+write LOC buffers,
        run BEFORE graph.replay() reading the live post-compaction v2p, so the
        captured graph carries zero translate nodes. No-op for non-unified pools.

        Read buffers (full kv_indices, SWA window) are translated IN PLACE; the
        full-attn WRITE loc is RETURNED as the [:n] view of the backend-owned
        out_cache_loc_full_physical buffer. Eager .item() bounds are fine here
        (out-of-graph), so no in-graph translate variant is needed.
        """
        if self._translate_kv_loc is None:
            return None
        # seq_lens_sum is the reliable "mirror present" signal: it is
        # None-preserving into the replay view, unlike seq_lens_cpu (always a
        # non-None but stale slice for gpu_only batches). None -> fall back to a
        # per-step D2H `.item()` on the indptr.
        have_cpu_mirror = forward_batch.seq_lens_sum is not None
        # Full-attention read path. kv_indptr[bs] == seq_lens_sum.
        n_kv = (
            forward_batch.seq_lens_sum
            if have_cpu_mirror
            else int(self.kv_indptr[bs].item())
        )
        if n_kv > 0:
            self.cuda_graph_kv_indices[:n_kv] = self._translate_kv_loc(
                self.cuda_graph_kv_indices[:n_kv]
            )
        # SWA window read path. window_kv_indptr[bs] == sum(min(seq_len, window)).
        if self.sliding_window_size is not None and self.sliding_window_size > 0:
            if have_cpu_mirror:
                n_win = int(
                    forward_batch.seq_lens_cpu[:bs]
                    .clamp(max=self.sliding_window_size)
                    .sum()
                )
            else:
                n_win = int(self.window_kv_indptr[bs].item())
            if n_win > 0:
                self.cuda_graph_window_kv_indices[:n_win] = (
                    self.token_to_kv_pool.translate_loc_from_full_to_swa(
                        self.cuda_graph_window_kv_indices[:n_win]
                    )
                )
        # Full-attention write path: translate out_cache_loc -> physical into the
        # capture-stable buffer and RETURN the [:n] view.
        out_cache_loc = forward_batch.out_cache_loc
        n = out_cache_loc.shape[0]
        # Zero the padded tail first: a smaller replay batch leaves [n:] holding
        # stale ids that the captured store would write; send them to slot 0 (sink).
        self.cuda_graph_out_cache_loc_full_physical[n:].zero_()
        self.cuda_graph_out_cache_loc_full_physical[:n].copy_(
            self._translate_kv_loc(out_cache_loc)
        )
        return self.cuda_graph_out_cache_loc_full_physical[:n]

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init auxiliary variables for triton attention backend."""

        bs = forward_batch.batch_size
        window_kv_indptr = self.window_kv_indptr
        window_kv_indices = None
        window_num_kv_splits = None
        window_kv_offsets = None
        swa_attn_logits = None
        spec_info = forward_batch.spec_info
        # This rank's per-request OWNED kv length, when the DCP index build
        # already produced it. Under the WEIGHTED owner rule it is not
        # get_dcp_lens(seq_lens) -- ownership follows the token ratios -- so the
        # split-KV schedule has to be sized from the same numbers the index
        # build used. Stays None off the DCP path.
        dcp_seq_lens = None

        if forward_batch.forward_mode.is_decode_or_idle():
            if spec_info is None or spec_info.kv_indptr is None:
                # kv_indptr is None for draft-extend's idle batch; build from seq_lens.
                if self.dcp_size > 1:
                    # DCP: per-rank sharded KV indices, else each rank reads the
                    # whole KV instead of its owner shard.
                    kv_indptr, kv_indices, dcp_seq_lens = self._dcp_kv_indices(
                        forward_batch.req_pool_indices,
                        forward_batch.seq_lens,
                        self.kv_indptr,
                        # #618/#623: host mirror + its independently known sum.
                        # seq_lens_sum is the "mirror is fresh" signal (it is
                        # None-preserving for gpu_only batches while
                        # seq_lens_cpu is a stale slice), and it doubles as the
                        # equality check, so a stale vector is refused.
                        #
                        # #629: the signal has to be APPLIED, not just relied
                        # on. dcp_host_lens accepts a mirror unchecked when no
                        # expected_sum comes with it, so passing the pair raw
                        # meant a gpu_only batch handed over the stale slice
                        # with nothing to catch it -- a silent mis-size.
                        lens_cpu=dcp_fresh_host_lens(
                            forward_batch.seq_lens_cpu, forward_batch.seq_lens_sum
                        ),
                        lens_sum=forward_batch.seq_lens_sum,
                    )
                else:
                    # gpu_only: seq_lens_sum may be None; over-allocate is safe (ragged write).
                    seq_lens_sum = forward_batch.seq_lens_sum
                    if seq_lens_sum is None:
                        seq_lens_sum = bs * self.max_context_len
                    kv_indices = torch.empty(
                        seq_lens_sum, dtype=torch.int64, device=self.device
                    )
                    kv_indptr = self._fill_kv_indptr_and_indices(
                        bs,
                        forward_batch.seq_lens,
                        forward_batch.req_pool_indices,
                        kv_indices,
                    )
                    if self._translate_kv_loc is not None:
                        kv_indices = self._translate_kv_loc(kv_indices)
                if (
                    self.sliding_window_size is not None
                    and self.sliding_window_size > 0
                ):
                    window_kv_indptr, window_kv_indices, window_kv_lens, _ = (
                        update_sliding_window_buffer(
                            self.window_kv_indptr,
                            self.req_to_token,
                            self.sliding_window_size,
                            forward_batch.seq_lens,
                            forward_batch.req_pool_indices,
                            bs,
                            self.device,
                            self.token_to_kv_pool,
                        )
                    )
                    window_num_kv_splits = torch.empty(
                        (bs,), dtype=torch.int32, device=self.device
                    )
                    self.get_num_kv_splits(window_num_kv_splits, window_kv_lens)
            else:
                kv_indptr, kv_indices = spec_info.kv_indptr, spec_info.kv_indices
                bs = kv_indptr.shape[0] - 1

            attn_logits = torch.empty(
                (bs, self.num_head, self.max_kv_splits, self.v_head_dim),
                dtype=torch.float32,
                device=self.device,
            )
            if self.swa_v_head_dim is not None:
                swa_attn_logits = torch.empty(
                    (bs, self.num_head, self.max_kv_splits, self.swa_v_head_dim),
                    dtype=torch.float32,
                    device=self.device,
                )
            else:
                swa_attn_logits = None
            attn_lse = torch.empty(
                (bs, self.num_head, self.max_kv_splits),
                dtype=torch.float32,
                device=self.device,
            )
            num_kv_splits = torch.empty((bs,), dtype=torch.int32, device=self.device)
            if self.dcp_size > 1:
                # dcp_seq_lens, when set, is BY CONSTRUCTION the same tensor the
                # index build produced: on the even rule that is exactly
                # self._dcp_lens(seq_lens) (same call, same args), so this is a
                # no-op there; on the weighted rule it is the only correct
                # source, since ownership follows the ratios and not the modulo.
                split_lens = (
                    dcp_seq_lens
                    if dcp_seq_lens is not None
                    else self._dcp_lens(forward_batch.seq_lens)
                ).clamp_min(1)
            else:
                split_lens = forward_batch.seq_lens
            self.get_num_kv_splits(num_kv_splits, split_lens)

            qo_indptr = None
            custom_mask = None
            mask_indptr = None
            max_extend_len = None
        elif forward_batch.forward_mode.is_target_verify():
            bs = len(forward_batch.req_pool_indices)
            qo_indptr = torch.arange(
                0,
                (1 + bs) * self.num_draft_tokens,
                step=self.num_draft_tokens,
                dtype=torch.int32,
                device=self.device,
            )
            if self.dcp_size > 1:
                # The split assumes a uniform draft_token_num query block per
                # request and a LINEAR draft chain. A verify input with a
                # different layout would be routed into that assumption
                # SILENTLY, so it is refused here -- and refused rather than
                # allowed to fall through to the full un-sharded build below,
                # which is precisely the out-of-bounds read #180 removes.
                _spec_type = getattr(spec_info, "spec_input_type", None)
                if _spec_type not in _DCP_VERIFY_SPEC_INPUT_TYPES:
                    raise NotImplementedError(
                        f"Triton uneven-DCP target-verify (#180) serves "
                        f"{sorted(t.name for t in _DCP_VERIFY_SPEC_INPUT_TYPES)}, "
                        f"got {getattr(_spec_type, 'name', _spec_type)}. The M4 "
                        f"split assumes a uniform draft-token query block and a "
                        f"linear draft chain; use --attention-backend flashinfer "
                        f"or --dcp-size 1 for this algorithm."
                    )
                # THE M4 VERIFY SPLIT (#180). The verify attention is two
                # stages: the COMMITTED prefix, read from the pool and
                # therefore owner-sharded, and the draft block, attended out of
                # this rank's freshly projected k/v and NOT sharded (every rank
                # holds all of it for its own head shard). So the paged read
                # runs over seq_lens -- see dcp_verify_paged_lens for why that
                # is emphatically not seq_lens + num_draft_tokens -- through
                # the SAME owner-rule builder decode and extend call, and the
                # draft block never appears in kv_indices at all.
                _reject_stale_verify_window(spec_info, self.num_draft_tokens)
                kv_indptr, kv_indices, _ = self._dcp_kv_indices(
                    forward_batch.req_pool_indices,
                    dcp_verify_paged_lens(
                        forward_batch.seq_lens, self.num_draft_tokens
                    ),
                    self.kv_indptr,
                    # #618/#623: the verify paged length IS seq_lens (see
                    # dcp_verify_paged_lens for why it is emphatically not
                    # seq_lens + num_draft_tokens), so the mirror goes through
                    # the SAME function rather than assuming the identity here.
                    # #629: same freshness guard as the decode twin -- a
                    # gpu_only batch's stale seq_lens_cpu must not be sized
                    # from just because seq_lens_sum is None.
                    lens_cpu=_verify_host_mirror(
                        dcp_fresh_host_lens(
                            forward_batch.seq_lens_cpu, forward_batch.seq_lens_sum
                        ),
                        self.num_draft_tokens,
                    ),
                    lens_sum=forward_batch.seq_lens_sum,
                )
            else:
                # gpu_only: seq_lens_sum may be None; over-allocate is safe (ragged write).
                seq_lens_sum = forward_batch.seq_lens_sum
                if seq_lens_sum is None:
                    seq_lens_sum = bs * self.max_context_len
                kv_indices = torch.empty(
                    seq_lens_sum, dtype=torch.int64, device=self.device
                )
                kv_indptr = self._fill_kv_indptr_and_indices(
                    bs,
                    forward_batch.seq_lens,
                    forward_batch.req_pool_indices,
                    kv_indices,
                )

            if self.sliding_window_size is not None and self.sliding_window_size > 0:
                # window_kv_offsets gives the start position in custom mask
                (
                    window_kv_indptr,
                    window_kv_indices,
                    window_kv_lens,
                    window_kv_offsets,
                ) = update_sliding_window_buffer(
                    self.window_kv_indptr,
                    self.req_to_token,
                    self.sliding_window_size,
                    forward_batch.seq_lens,
                    forward_batch.req_pool_indices,
                    bs,
                    self.device,
                    self.token_to_kv_pool,
                )

            if self.dcp_size > 1:
                # CHAIN verify drops the mask (#180). At topk == 1 the EAGLE
                # draft is a chain, so its d x d draft->draft block IS the
                # causal mask, which the current-chunk stage applies anyway.
                # Under DCP dropping it is not an optimisation but a
                # requirement: the mask's row stride is the GLOBAL prefix
                # length and stage 2 offsets by it, so an owner-sharded prefix
                # would index it wrong. topk > 1 never reaches here -- the boot
                # guard refuses a tree mask on this lane by name.
                assert (
                    dcp_verify_mask_mode(self.topk) == "causal"
                ), "tree verify reached the DCP metadata build; guard hole"
                custom_mask = None
                mask_indptr = None
            else:
                custom_mask = spec_info.custom_mask
                seq_mask_len = self.num_draft_tokens * (
                    forward_batch.seq_lens + self.num_draft_tokens
                )
                mask_indptr = self.mask_indptr
                mask_indptr[1 : bs + 1] = torch.cumsum(seq_mask_len[:bs], dim=0)
                mask_indptr = mask_indptr[: bs + 1]
            max_extend_len = self.num_draft_tokens
            num_kv_splits = None
            attn_logits = None
            attn_lse = None

        else:
            if self.dcp_size > 1:
                kv_indptr, kv_indices, _ = self._dcp_kv_indices(
                    forward_batch.req_pool_indices,
                    forward_batch.extend_prefix_lens,
                    self.kv_indptr,
                    # #618/#623: extend_prefix_lens_cpu is the exact host mirror
                    # of extend_prefix_lens -- the non-DCP branch right below
                    # already sizes its kv_indices from sum() of it. No second
                    # host sum exists to check it against, so none is claimed.
                    lens_cpu=forward_batch.extend_prefix_lens_cpu,
                )
            else:
                # gpu_only leaves _cpu unset; over-allocate is safe (ragged write).
                if forward_batch.extend_prefix_lens_cpu is not None:
                    kv_indices_len = sum(forward_batch.extend_prefix_lens_cpu)
                else:
                    kv_indices_len = bs * self.max_context_len
                kv_indices = torch.empty(
                    kv_indices_len,
                    dtype=torch.int64,
                    device=self.device,
                )
                kv_indptr = self._fill_kv_indptr_and_indices(
                    bs,
                    forward_batch.extend_prefix_lens,
                    forward_batch.req_pool_indices,
                    kv_indices,
                )
                if self._translate_kv_loc is not None:
                    kv_indices = self._translate_kv_loc(kv_indices)
            if self.sliding_window_size is not None and self.sliding_window_size > 0:
                (
                    window_kv_indptr,
                    window_kv_indices,
                    window_kv_lens,
                    window_kv_offsets,
                ) = update_sliding_window_buffer(
                    self.window_kv_indptr,
                    self.req_to_token,
                    self.sliding_window_size,
                    forward_batch.extend_prefix_lens,
                    forward_batch.req_pool_indices,
                    bs,
                    self.device,
                    self.token_to_kv_pool,
                )

            qo_indptr = self.qo_indptr
            qo_indptr[1 : bs + 1] = torch.cumsum(forward_batch.extend_seq_lens, dim=0)
            qo_indptr = qo_indptr[: bs + 1]
            custom_mask = None
            mask_indptr = None
            attn_logits = None
            attn_lse = None
            # Defensive GPU-max fallback when extend_seq_lens_cpu is absent.
            if forward_batch.extend_seq_lens_cpu is not None:
                max_extend_len = max(forward_batch.extend_seq_lens_cpu)
            else:
                max_extend_len = int(forward_batch.extend_seq_lens.max())
            num_kv_splits = None

        swa_out_cache_loc = None
        if self.use_sliding_window_kv_pool and forward_batch.out_cache_loc is not None:
            swa_out_cache_loc = self.token_to_kv_pool.translate_loc_from_full_to_swa(
                forward_batch.out_cache_loc
            )

        # Unified pool full-attention WRITE loc (virtual out_cache_loc -> physical),
        # carried in the metadata (-> KVWriteLoc.full_loc). None for non-unified pools.
        out_cache_loc_full_physical = None
        if (
            self._translate_kv_loc is not None
            and forward_batch.out_cache_loc is not None
        ):
            out_cache_loc_full_physical = self._translate_kv_loc(
                forward_batch.out_cache_loc
            )

        self.forward_metadata = ForwardMetadata(
            attn_logits,
            attn_lse,
            max_extend_len,
            num_kv_splits,
            kv_indptr,
            kv_indices,
            qo_indptr,
            custom_mask,
            mask_indptr,
            window_kv_indptr,
            window_kv_indices,
            window_num_kv_splits,
            window_kv_offsets,
            swa_attn_logits=swa_attn_logits,
            swa_out_cache_loc=swa_out_cache_loc,
            out_cache_loc_full_physical=out_cache_loc_full_physical,
        )

    def init_cuda_graph_state(
        self,
        max_bs: int,
        max_num_tokens: int,
        kv_indices_buf: Optional[torch.Tensor] = None,
        cuda_graph_num_kv_splits_buf: Optional[torch.Tensor] = None,
    ):
        self.cuda_graph_attn_logits = torch.zeros(
            (max_num_tokens, self.num_head, self.max_kv_splits, self.v_head_dim),
            dtype=torch.float32,
            device=self.device,
        )
        if self.swa_v_head_dim is not None:
            self.cuda_graph_swa_attn_logits = torch.zeros(
                (
                    max_num_tokens,
                    self.num_head,
                    self.max_kv_splits,
                    self.swa_v_head_dim,
                ),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            self.cuda_graph_swa_attn_logits = None
        self.cuda_graph_attn_lse = torch.zeros(
            (max_num_tokens, self.num_head, self.max_kv_splits),
            dtype=torch.float32,
            device=self.device,
        )

        if cuda_graph_num_kv_splits_buf is None:
            self.cuda_graph_num_kv_splits = torch.full(
                (max_num_tokens,),
                self.max_kv_splits,
                dtype=torch.int32,
                device=self.device,
            )
        else:
            self.cuda_graph_num_kv_splits = cuda_graph_num_kv_splits_buf

        if kv_indices_buf is None:
            self.cuda_graph_kv_indices = torch.zeros(
                (max_num_tokens * self.max_context_len),
                dtype=torch.int64,
                device=self.device,
            )
        else:
            self.cuda_graph_kv_indices = kv_indices_buf

        if not self.skip_prefill:
            self.cuda_graph_custom_mask = torch.zeros(
                (max_num_tokens * self.max_context_len),
                dtype=torch.uint8,
                device=self.device,
            )

        if self.sliding_window_size is not None and self.sliding_window_size > 0:
            if kv_indices_buf is None:
                self.cuda_graph_window_kv_indices = torch.zeros(
                    (max_num_tokens * self.sliding_window_size),
                    dtype=torch.int64,
                    device=self.device,
                )
            else:
                self.cuda_graph_window_kv_indices = torch.zeros_like(kv_indices_buf)

            self.cuda_graph_window_num_kv_splits = torch.full(
                (max_num_tokens,),
                self.max_kv_splits,
                dtype=torch.int32,
                device=self.device,
            )

            self.cuda_graph_window_kv_offsets = torch.zeros(
                (max_bs,),
                dtype=torch.int32,
                device=self.device,
            )

        if self.use_sliding_window_kv_pool:
            # SWA write-target buffer; refilled at replay from out_cache_loc.
            self.cuda_graph_swa_out_cache_loc = torch.zeros(
                (max_num_tokens,),
                dtype=torch.int64,
                device=self.device,
            )

        if self._translate_kv_loc is not None:
            # Unified pool full-attention write-target buffer, refilled at replay
            # (-> KVWriteLoc.full_loc). Capture-stable, mirrors cuda_graph_swa_out_cache_loc.
            self.cuda_graph_out_cache_loc_full_physical = torch.zeros(
                (max_num_tokens,),
                dtype=torch.int64,
                device=self.device,
            )

    def _build_cuda_graph_forward_metadata(
        self,
        bs: int,
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        swa_out_cache_loc: Optional[torch.Tensor] = None,
        out_cache_loc_full_physical: Optional[torch.Tensor] = None,
    ) -> ForwardMetadata:
        """Construct ForwardMetadata from the current cuda-graph buffer state.

        Called by capture after the buffer-update helpers have already run
        (either via replay or directly).  All fields reference the same
        self.cuda_graph_* tensors that the captured graph kernels will
        read — the Python object is rebuilt each capture, but the underlying
        GPU memory addresses are stable. ``swa_out_cache_loc`` is the
        pre-allocated SWA write-target buffer view (or None for non-SWA).
        """
        swa = self.sliding_window_size is not None and self.sliding_window_size > 0
        if forward_mode.is_decode_or_idle():
            return ForwardMetadata(
                attn_logits=self.cuda_graph_attn_logits,
                attn_lse=self.cuda_graph_attn_lse,
                max_extend_len=None,
                num_kv_splits=self.cuda_graph_num_kv_splits,
                kv_indptr=self.kv_indptr[: bs + 1],
                kv_indices=self.cuda_graph_kv_indices,
                qo_indptr=None,
                custom_mask=None,
                mask_indptr=None,
                window_kv_indptr=self.window_kv_indptr[: bs + 1] if swa else None,
                window_kv_indices=self.cuda_graph_window_kv_indices if swa else None,
                window_num_kv_splits=(
                    self.cuda_graph_window_num_kv_splits if swa else None
                ),
                window_kv_offsets=None,
                swa_attn_logits=self.cuda_graph_swa_attn_logits,
                swa_out_cache_loc=swa_out_cache_loc,
                out_cache_loc_full_physical=out_cache_loc_full_physical,
            )
        elif forward_mode.is_target_verify():
            # On the DCP lane the chain verify carries no mask (#180) -- the
            # captured graph must agree with the buffer update, else it would
            # plan a tree-mask read the eager path already dropped.
            custom_mask = (
                self.cuda_graph_custom_mask
                if self.dcp_size <= 1
                and spec_info is not None
                and getattr(spec_info, "custom_mask", None) is not None
                else None
            )
            return ForwardMetadata(
                attn_logits=None,
                attn_lse=None,
                max_extend_len=self.num_draft_tokens,
                num_kv_splits=None,
                kv_indptr=self.kv_indptr[: bs + 1],
                kv_indices=self.cuda_graph_kv_indices,
                qo_indptr=self.qo_indptr[: bs + 1],
                custom_mask=custom_mask,
                mask_indptr=(
                    None if self.dcp_size > 1 else self.mask_indptr[: bs + 1]
                ),
                window_kv_indptr=self.window_kv_indptr[: bs + 1] if swa else None,
                window_kv_indices=self.cuda_graph_window_kv_indices if swa else None,
                window_num_kv_splits=(
                    self.cuda_graph_window_num_kv_splits if swa else None
                ),
                window_kv_offsets=self.cuda_graph_window_kv_offsets if swa else None,
                swa_out_cache_loc=swa_out_cache_loc,
                out_cache_loc_full_physical=out_cache_loc_full_physical,
            )
        elif forward_mode.is_draft_extend_v2():
            return ForwardMetadata(
                attn_logits=None,
                attn_lse=None,
                # Must match the per-req query count (num_tokens_per_bs) used to
                # build qo_indptr above, else the extend kernel grid is too small
                # for topk > 1 (num_draft_tokens > num_steps+1) and drops query
                # blocks.
                max_extend_len=(
                    self.num_draft_tokens
                    if forward_mode.is_draft_extend_v2()
                    else self.speculative_num_steps + 1
                ),
                num_kv_splits=None,
                kv_indptr=self.kv_indptr[: bs + 1],
                kv_indices=self.cuda_graph_kv_indices,
                qo_indptr=self.qo_indptr[: bs + 1],
                custom_mask=None,
                mask_indptr=None,
                window_kv_indptr=self.window_kv_indptr,
                window_kv_indices=None,
                window_num_kv_splits=None,
                window_kv_offsets=None,
                swa_out_cache_loc=swa_out_cache_loc,
                out_cache_loc_full_physical=out_cache_loc_full_physical,
            )
        else:
            raise ValueError(f"Invalid forward mode: {forward_mode=} for CUDA Graph.")

    def _apply_cuda_graph_metadata(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[Union[Sequence[int], torch.Tensor]] = None,
        seq_lens_sum: Optional[int] = None,
    ):
        """Shared capture+replay body for the cuda-graph init path.

        Public entry: :py:meth:`init_forward_metadata_out_graph`.

        ``seq_lens_cpu`` / ``seq_lens_sum`` are the #629 host mirror of
        ``seq_lens``; they reach the owner-rule index builders that the
        replay-prep fills drive, and default to None (old device read).
        """
        # NOTE: encoder_lens expected to be zeros or None
        if forward_mode.is_decode_or_idle():
            assert spec_info is None, "Multi-step cuda graph init is not done here."
            _, _, window_kv_lens, num_kv_splits_lens = self._update_decode_kv_buffers(
                bs,
                seq_lens,
                req_pool_indices,
                seq_lens_cpu=seq_lens_cpu,
                seq_lens_sum=seq_lens_sum,
            )
            self.get_num_kv_splits(
                self.cuda_graph_num_kv_splits[:bs], num_kv_splits_lens[:bs]
            )
            if window_kv_lens is not None:
                self.get_num_kv_splits(
                    self.cuda_graph_window_num_kv_splits[:bs], window_kv_lens[:bs]
                )
        elif forward_mode.is_target_verify():
            bs = len(req_pool_indices)
            self._update_target_verify_buffers(
                bs,
                seq_lens,
                req_pool_indices,
                spec_info,
                seq_lens_cpu=seq_lens_cpu,
                seq_lens_sum=seq_lens_sum,
            )
        elif forward_mode.is_draft_extend_v2():
            self._update_draft_extend_buffers(
                bs, seq_lens, req_pool_indices, forward_mode, spec_info
            )
        else:
            raise ValueError(
                f"Invalid forward mode: {forward_mode=} for CUDA Graph replay."
            )

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def get_verify_buffers_to_fill_after_draft(self):
        """
        Return buffers for verify attention kernels that needs to be filled after draft.

        Typically, these are tree mask and position buffers.
        """
        return [self.cuda_graph_custom_mask, None]

    def update_verify_buffers_to_fill_after_draft(
        self, spec_info: SpecInput, cuda_graph_bs: Optional[int]
    ):
        pass

    def _dcp_layer_token_sharded(self, layer: RadixAttention) -> bool:
        """Is this layer's KV token-sharded across the DCP group? (#96)

        THE per-layer dispatch: the write path, the extend path and the decode
        path all ask this one question, so they cannot classify a layer
        differently -- a layer written through the owner rule but read without it
        (or the reverse) is silently wrong output, and a layer one rank shards
        while another does not is a hang in the q all-gather.

        GROUP-UNIFORM BY CONSTRUCTION, which is the reason it is spelled out
        here instead of inline: both inputs are process-global configuration.
        ``self.swa_hybrid_dcp`` comes from server args + model config, and the
        layer's sliding window from the checkpoint's layer_types -- never from
        anything this batch, request or rank produced. (The forbidden version of
        this predicate is the one that asks "does THIS rank own any window
        slots", which is exactly what sharding the window itself would have
        forced; see docs_new/swa_dcp_stage_b_triton.md section 0.)
        """
        return dcp_token_sharded_layer(
            layer.sliding_window_size is not None and layer.sliding_window_size > -1,
            swa_hybrid_lane=self.swa_hybrid_dcp,
        )

    def _dcp_group_q_head_counts(self, local_heads: int) -> list:
        """This DCP group's per-rank q-head counts (see the module helper)."""
        return _plan_aware_dcp_group_q_head_counts(
            self.dcp_model_config, self.dcp_size, local_heads
        )

    def _replicated_kv_ragged_reindex(self, local_q: int, local_kv: int, device):
        """REPLICATED-KV geometry (#105): map each of this rank's LOCAL kv slots
        to the GLOBAL kv head its q group attends.

        The current-chunk stage of ``extend_attention_fwd`` is a uniform-GQA
        kernel: it derives the grouping from the tensors it is handed,
        ``kv_group_num = q_extend.shape[1] // k_extend.shape[1]``. Under
        REPLICATED-KV this rank holds a q-head SLICE but ALL kv heads, so the
        LOCAL grouping (local_q // local_kv) is not the GLOBAL one
        (total_q // total_kv) whenever the rank does not hold every q head of
        its kv head(s). Re-indexing the kv heads makes the uniform kernel
        reproduce the global mapping.

        (8q / 2kv over a [4,2,2] q split: rank 0 holds q0-3, all of which attend
        kv0 globally, but local GQA 2 would send q2-3 to kv1 -- short prompts
        and first chunks come out corrupted while the gathered-q paged prefix
        and decode paths stay correct, which is what makes it hard to see.)

        Returns a LongTensor of length local_kv, or None when the mapping is
        already the identity. Cached per (local_q, local_kv): both are constant
        across layers and forwards for a given rank. The rule itself lives in
        the module-level ``replicated_kv_reindex`` so it can be pinned without a
        device; this is the caching tensor wrapper. Byte-for-byte the flashinfer
        twin's rule.
        """
        cache = self._repl_kv_reindex_cache
        key = (local_q, local_kv)
        if key in cache:
            return cache[key]
        counts = self._dcp_group_q_head_counts(local_q)
        idx = replicated_kv_reindex(
            sum(counts[: self.dcp_rank]),
            local_q,
            local_kv,
            self.dcp_full_qo_heads // self.dcp_full_kv_heads,
        )
        out = (
            None
            if idx is None
            else torch.tensor(idx, device=device, dtype=torch.long)
        )
        cache[key] = out
        return out

    def _dcp_gather_q_heads(self, q_local: torch.Tensor, group) -> torch.Tensor:
        """All-gather the DCP group's q heads along dim=1.

        Was a bare ``group.all_gather(x, dim=1)``, i.e. an EQUAL-SHAPE
        collective whose precondition -- every rank of the group contributes
        the same number of q heads -- lived nowhere but in the reader's head.
        Under a --rank-tp-ratio plan the shards are unequal ([16,8,8] for
        total_q=32/kv=8/tp=3), the ranks disagree on the collective's byte
        count, and torch neither refuses nor repairs that: it hangs or returns
        garbage in global head order. Routing through
        cp_all_gather_heads_uneven makes the counts an explicit argument
        (pad-to-max, gather, slice each rank's true count, concatenate in rank
        order) and asserts this rank's count against the tensor it was handed.

        Equal counts take that helper's documented fast path -- the same plain
        collective as before -- so the reachable configurations are unchanged.
        """
        counts = self._dcp_group_q_head_counts(q_local.shape[1])
        return cp_all_gather_heads_uneven(q_local, group, counts)

    def _dcp_merge_q_heads(
        self,
        out: torch.Tensor,
        lse: torch.Tensor,
        group,
        local_heads: int,
        return_lse: bool = False,
    ):
        """LSE-merge the group's partial attention and slice this rank's heads.

        Counterpart of _dcp_gather_q_heads: ``cp_lse_ag_out_rs_mha`` slices the
        merged output with ``H // world_size * rank``, which silently picks the
        WRONG heads the moment the shards are unequal (with [16,8,8] rank 1
        would read heads 10:20 instead of 16:24, and heads 30 and 31 would be
        dropped by every rank). The uneven variant slices by prefix sum over
        the same counts; for an equal split the two expressions are the same
        number.
        """
        counts = self._dcp_group_q_head_counts(local_heads)
        assert sum(counts) == out.shape[1], (
            f"DCP head merge: per-rank counts {counts} sum to {sum(counts)}, "
            f"but the gathered attention output carries {out.shape[1]} heads"
        )
        return cp_lse_ag_out_ar_mha_uneven(
            out, lse, group, counts, return_lse=return_lse
        )

    def _dcp_write_gather(self, layer: RadixAttention, k: torch.Tensor, v: torch.Tensor):
        """Raise this rank's kv-head shard to the FULL replicated head set.

        Under the uneven-DCP lane every pool row holds all total_num_kv_heads,
        because the pool is sharded along TOKENS instead of heads. When the kv
        heads are also head-sharded across ranks (kv >= tp), the freshly
        projected k/v carry only this rank's slice and have to be gathered
        first. When kv < tp the k/v weights are REPLICATED -- every rank already
        projects all kv heads from identical weights -- so the gather is a no-op
        and, importantly, no collective is issued at all: that is the 27B/35B
        case this port targets.

        The gather is one FUSED all-gather of cat(k, v) along the head dim, not
        two: k and v share dcp_kv_head_counts, so stacking them on the row dim
        gives per-tensor results identical to two separate gathers (pure data
        movement, no reduction) at half the launches. Same construction as the
        flashinfer twin.

        RANK-UNIFORM: the branch is on dcp_kv_replicated_heads, a boot-time
        property of the geometry, identical on every rank of the group -- never
        on anything derived from this batch's data.
        """
        # Same view the layer itself uses, so a model with qk_head_dim !=
        # v_head_dim is reshaped correctly rather than by a shared head_dim.
        k = k.view(-1, layer.tp_k_head_num, layer.qk_head_dim)
        v = v.view(-1, layer.tp_v_head_num, layer.v_head_dim)
        if self.dcp_kv_replicated_heads:
            return k, v
        if layer.qk_head_dim != layer.v_head_dim:
            raise ValueError(
                "uneven-DCP kv-head gather with head-sharded kv heads requires "
                f"qk_head_dim == v_head_dim (got {layer.qk_head_dim} != "
                f"{layer.v_head_dim}); the fused cat(k, v) all-gather cannot "
                "stack two different head dims. Use --attention-backend "
                "flashinfer for this model, or a geometry with kv < tp so the "
                "kv heads are replicated and no gather is needed."
            )
        group = get_parallel().dcp_group
        n = k.shape[0]
        kv_full = cp_all_gather_heads_uneven(
            torch.cat((k, v), dim=0), group, self.dcp_kv_head_counts
        )
        return kv_full[:n], kv_full[n:]

    def _set_kv_buffer(
        self,
        forward_batch: ForwardBatch,
        layer: RadixAttention,
        loc_info,
        k: torch.Tensor,
        v: torch.Tensor,
        k_scale=None,
        v_scale=None,
    ) -> None:
        # DCP writes to the local physical shard (loc = out_cache_loc //
        # dcp_size) through the masked path so each rank only stores the tokens
        # it owns. Non-DCP keeps the original write loc and plain set_kv_buffer.
        #
        # #96: under the SWA-hybrid lane a sliding-window layer is NOT
        # token-sharded, so it takes the plain branch -- KVWriteLoc carries the
        # pre-translated swa_loc, the kv heads stay this rank's SWA shard, and no
        # dcp_kv_mask is produced at all. That write is then byte-for-byte the
        # validated non-DCP uneven-TP SWA write.
        if self.dcp_size > 1 and self._dcp_layer_token_sharded(layer):
            if self.uneven_dcp:
                k, v = self._dcp_write_gather(layer, k, v)
            if self.uneven_dcp_weighted:
                # WEIGHTED owner rule (#173): ownership AND the compact row are
                # derived from out_cache_loc itself, not from the sequence
                # position, which is what keeps the row an injective function of
                # the slot across concurrent requests -- exactly as the even
                # `loc // dcp_size` is. This rank owns L iff (L % cp_S) is in
                # [cp_lo, cp_hi) and stores it at
                # (L // cp_S) * cp_ratio + (L % cp_S - cp_lo).
                # dcp_weighted_write_slots is the SAME function the flashinfer
                # write path calls, and its expression is shared with the read
                # side (build_dcp_weighted_kv_indices), so the two cannot drift.
                loc, dcp_kv_mask = dcp_weighted_write_slots(
                    forward_batch.out_cache_loc,
                    self.cp_S,
                    self.cp_lo,
                    self.cp_hi,
                    self.cp_ratio,
                )
            else:
                loc = forward_batch.out_cache_loc // self.dcp_size
                # EVEN modulo owner rule via the shared helper: it refuses a
                # padded ``positions`` against a narrowed ``out_cache_loc``
                # (#472) instead of falling through to an unmasked write.
                dcp_kv_mask = dcp_even_write_mask(
                    forward_batch.positions,
                    loc.numel(),
                    self.dcp_size,
                    self.dcp_rank,
                    forward_batch.dcp_kv_mask,
                )
            kwargs = {"dcp_kv_mask": dcp_kv_mask}
        else:
            loc = loc_info
            kwargs = {}
        if k_scale is None and v_scale is None:
            self.token_to_kv_pool.set_kv_buffer(layer, loc, k, v, **kwargs)
        else:
            self.token_to_kv_pool.set_kv_buffer(
                layer, loc, k, v, k_scale, v_scale, **kwargs
            )

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        sinks=None,
    ):
        # TODO: reuse the buffer across layers
        attn_out = getattr(forward_batch, "_attn_output", None)
        if attn_out is not None:
            o = attn_out
        elif layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        if k is None and v is None:
            pool = self.token_to_kv_pool
            cache_loc = forward_batch.out_cache_loc
            if isinstance(pool, SWAKVPool) and pool.layers_mapping[layer.layer_id][1]:
                cache_loc = pool.translate_loc_from_full_to_swa(cache_loc)
            k_buffer, v_buffer = pool.get_kv_buffer(layer.layer_id)
            k = k_buffer[cache_loc]
            v = v_buffer[cache_loc]
        elif k is None or v is None:
            raise ValueError("Both k and v should be None or not None")
        else:
            # Save KV cache first (must do this before unified kernel)
            if save_kv_cache:
                loc_info = KVWriteLoc(
                    forward_batch.out_cache_loc,
                    self.forward_metadata.swa_out_cache_loc,
                    full_loc=self.forward_metadata.out_cache_loc_full_physical,
                )
                if layer.k_scale is None:
                    self._set_kv_buffer(forward_batch, layer, loc_info, k, v)
                elif self.use_mla:
                    # For MLA, scale K manually before storing since MLATokenToKVPool
                    # doesn't accept scale parameters. Clone to protect k from mutation
                    # since it's used later in the attention kernel.
                    k_scaled = k.clone().div_(layer.k_scale)
                    self.token_to_kv_pool.set_kv_buffer(
                        layer,
                        loc_info,
                        k_scaled,
                        v,
                    )
                else:
                    self._set_kv_buffer(
                        forward_batch,
                        layer,
                        loc_info,
                        k.clone(),  # cloned to protect k,v from in-place mutation in set_kv_buffer
                        v.clone(),
                        layer.k_scale,
                        layer.v_scale,
                    )

        logits_soft_cap = logit_capping_mod(layer.logit_capping_method, layer.logit_cap)

        causal = True
        if (
            layer.is_cross_attention
            or layer.attn_type == AttentionType.ENCODER_ONLY
            or (
                layer.attn_type == AttentionType.DECODER_BIDIRECTIONAL
                and self.allow_bidirectional_attention_in_extend
            )
        ):
            causal = False

        # #96: a sliding-window layer under the SWA-hybrid lane falls THROUGH to
        # the ordinary 2-stage window path below -- its KV is not token-sharded,
        # so there is no owned-slot subset to mask and no cross-rank merge to do.
        if self.dcp_size > 1 and self._dcp_layer_token_sharded(layer):
            return self._forward_extend_dcp(
                q, k, v, layer, forward_batch, causal, logits_soft_cap, sinks
            )

        # Deterministic mode: use unified 1-stage kernel
        if self.enable_deterministic:
            return self._forward_extend_unified(
                q, o, layer, forward_batch, causal, logits_soft_cap, sinks
            )

        # Normal mode: use original 2-stage kernel
        if layer.sliding_window_size is not None and layer.sliding_window_size > -1:
            sliding_window_size = (
                layer.sliding_window_size
            )  # Needed for sliding window mask
            kv_indptr = self.forward_metadata.window_kv_indptr
            kv_indices = self.forward_metadata.window_kv_indices
            window_kv_offsets = self.forward_metadata.window_kv_offsets
        else:
            sliding_window_size = -1
            kv_indptr = self.forward_metadata.kv_indptr
            kv_indices = self.forward_metadata.kv_indices
            window_kv_offsets = None

        if layer.k_scale is not None and layer.v_scale is not None:
            k_descale = layer.k_scale_float
            v_descale = layer.v_scale_float
        else:
            k_descale = 1.0
            v_descale = 1.0

        # Split-KV EAGLE-verify fast path (ROCm/Triton). On target-verify
        # (topk=1 causal chain), run the bandwidth-efficient split-KV kernel
        # instead of the serial-prefix extend kernel. verify_splitkv_fwd()
        # returns True if it ran (o written), or False for any case it cannot
        # serve bit-equivalently (its can_handle() gates on non-causal / sinks /
        # sliding-window / ragged / topk>1), so we fall through to
        # extend_attention_fwd below. Correctness is never at risk.
        if (
            self.use_verify_splitkv
            and forward_batch.forward_mode.is_target_verify()
            and self.verify_splitkv_fwd(
                q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                k.contiguous(),
                v.contiguous(),
                o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
                self.token_to_kv_pool.get_key_buffer(layer.layer_id),
                self.token_to_kv_pool.get_value_buffer(layer.layer_id),
                self.forward_metadata.qo_indptr,
                kv_indptr,
                kv_indices,
                self.forward_metadata.custom_mask,
                causal,
                self.forward_metadata.mask_indptr,
                self.forward_metadata.max_extend_len,
                k_descale,
                v_descale,
                layer.scaling,
                logit_cap=logits_soft_cap,
                sliding_window_size=sliding_window_size,
                sinks=sinks,
                window_kv_offsets=window_kv_offsets,
                xai_temperature_len=layer.xai_temperature_len,
                max_bs=self.req_to_token_pool.size,
            )
        ):
            return o

        self.extend_attention_fwd(
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            k.contiguous(),
            v.contiguous(),
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            self.token_to_kv_pool.get_key_buffer(layer.layer_id),
            self.token_to_kv_pool.get_value_buffer(layer.layer_id),
            self.forward_metadata.qo_indptr,
            kv_indptr,
            kv_indices,
            self.forward_metadata.custom_mask,
            causal,
            self.forward_metadata.mask_indptr,
            self.forward_metadata.max_extend_len,
            k_descale,
            v_descale,
            layer.scaling,
            logit_cap=logits_soft_cap,
            sliding_window_size=sliding_window_size,
            sinks=sinks,
            window_kv_offsets=window_kv_offsets,
            xai_temperature_len=layer.xai_temperature_len,
            page_size=self.page_size,
        )
        return o

    @staticmethod
    def _dcp_batch_has_prefix(
        forward_batch: ForwardBatch, kv_indices: torch.Tensor
    ) -> bool:
        """Does this BATCH have any paged prefix to read? A GROUP-UNIFORM answer.

        The DCP extend path skips the q-head all-gather and the LSE merge when
        there is no prefix. Those are COLLECTIVES, so the decision has to be
        identical on every rank of the group -- and the obvious local test,
        "did I get any kv indices", is not: a rank owns
        len//N + (rank < len%N) prefix slots under the even rule and a
        ratio-proportional share under the weighted one, so a short prefix over
        an uneven vector is routinely owned entirely by the high-ratio rank.
        The low-ratio ranks would return early, the high-ratio one would sit in
        an all-gather nobody joins, and the server hangs on a request whose only
        peculiarity is a 1-token prefix.

        extend_prefix_lens is a GLOBAL, replicated tensor, so basing the branch
        on it is uniform by construction. A rank that then owns zero rows still
        runs both collectives and contributes out=0 / lse=-inf, which is exactly
        the empty-attention contract the LSE merge already handles.

        TARGET-VERIFY IS ANSWERED FIRST AND UNCONDITIONALLY (#180). A verify
        batch carries NO prefix lengths -- forward_batch_info fills them from
        batch.prefix_lens, which a verify batch built out of a decode batch
        never sets -- so before #180 it fell through to the rank-local
        fallback below. That is the D5 defect arriving through a second door:
        the committed prefix of a verify step IS owner-sharded, so a short one
        over an uneven token vector is owned entirely by the high-ratio rank
        and the others would skip two collectives the owner then waits in.
        forward_mode is replicated, so keying on it is uniform by
        construction -- this is flashinfer's `force_prefix =
        forward_mode.is_target_verify()` (M4), stated the same way. A verify
        step always HAS a committed prefix; and even if it did not, forcing the
        branch costs two collectives over an empty read, which the LSE merge
        already handles as out=0 / lse=-inf.

        The old local test is kept only as the fallback for a batch that carries
        no prefix lengths at all AND is not a verify (it is then equivalent: no
        prefix lengths means no paged read was planned for anyone).
        """
        if forward_batch.forward_mode.is_target_verify():
            return True
        lens_cpu = getattr(forward_batch, "extend_prefix_lens_cpu", None)
        if lens_cpu is not None:
            return any(lens_cpu)
        lens = getattr(forward_batch, "extend_prefix_lens", None)
        if lens is not None:
            # gpu_only batches leave the _cpu mirror unset; one sync on a
            # bs-sized tensor, and only on that configuration.
            return bool(lens.numel()) and int(lens.max()) > 0
        return kv_indices.numel() > 0

    def _forward_extend_dcp(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        causal: bool,
        logits_soft_cap: float,
        sinks: Optional[torch.Tensor],
    ):
        if sinks is not None:
            raise NotImplementedError("DCP Triton extend does not support sinks")
        if self.forward_metadata.custom_mask is not None:
            # BACKSTOP, not the gate. Chain verify (topk == 1) is served here
            # since #180 and reaches this function with custom_mask already
            # dropped in the metadata build, because the mask's row stride is
            # the GLOBAL prefix length and an owner-sharded prefix would index
            # it wrong. A tree mask is refused at boot. So a non-None mask
            # arriving here means one of those two decisions has a hole.
            raise NotImplementedError(
                "DCP Triton extend does not support custom masks; a chain "
                "verify must arrive with custom_mask=None (#180) and a tree "
                "verify must be refused at boot"
            )
        if layer.sliding_window_size is not None and layer.sliding_window_size > -1:
            # UNREACHABLE BY CONSTRUCTION since #96: forward_extend routes
            # sliding-window layers to the local window path whenever the
            # SWA-hybrid lane is on, and reject_unsupported_dcp_geometry refuses
            # a windowed model on this lane when it is off. Kept as the assertion
            # it now is: if a future dispatch change lets a window layer in here,
            # the paged read would silently attend a sparse owned-slot subset
            # with no window mask (wrong output, no crash).
            raise NotImplementedError(
                "DCP Triton extend does not support sliding window"
            )

        group = get_parallel().dcp_group
        q_local = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim).contiguous()
        total_tokens, local_heads, _ = q_local.shape

        kv_indptr = self.forward_metadata.kv_indptr
        kv_indices = self.forward_metadata.kv_indices
        max_extend_len = self.forward_metadata.max_extend_len

        if layer.k_scale is not None and layer.v_scale is not None:
            k_descale = layer.k_scale_float
            v_descale = layer.v_scale_float
        else:
            k_descale = 1.0
            v_descale = 1.0

        k_buffer = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
        v_buffer = self.token_to_kv_pool.get_value_buffer(layer.layer_id)

        current_out = torch.zeros(
            (total_tokens, local_heads, layer.v_head_dim),
            device=q.device,
            dtype=torch.float32,
        )
        current_lse = torch.full(
            (total_tokens, local_heads),
            -float("inf"),
            device=q.device,
            dtype=torch.float32,
        )

        # Current chunk K/V is still local before masked cache write, so it can
        # use the original extend kernel's current-token stage directly.
        if k.numel() > 0:
            k_cur = k.contiguous()
            v_cur = v.contiguous()
            if self.uneven_dcp and self.dcp_kv_replicated_heads:
                # #105: this rank holds a q-head SLICE but ALL kv heads, so the
                # uniform-GQA kernel's local grouping is not the global q->kv
                # mapping. Re-index the kv heads so it becomes one. Identity (and
                # a no-op) whenever the local and global groupings coincide.
                _kv_idx = self._replicated_kv_ragged_reindex(
                    layer.tp_q_head_num, layer.tp_k_head_num, k.device
                )
                if _kv_idx is not None:
                    k_cur = k_cur.view(-1, layer.tp_k_head_num, layer.qk_head_dim)[
                        :, _kv_idx, :
                    ].contiguous()
                    v_cur = v_cur.view(-1, layer.tp_v_head_num, layer.v_head_dim)[
                        :, _kv_idx, :
                    ].contiguous()
            empty_kv_indptr = torch.zeros_like(kv_indptr)
            self.extend_attention_fwd(
                q_local,
                k_cur,
                v_cur,
                current_out,
                k_buffer,
                v_buffer,
                self.forward_metadata.qo_indptr,
                empty_kv_indptr,
                kv_indices[:0],
                None,
                causal,
                None,
                max_extend_len,
                1.0,
                1.0,
                sm_scale=layer.scaling,
                logit_cap=logits_soft_cap,
                xai_temperature_len=layer.xai_temperature_len,
                lse_extend=current_lse,
                skip_prefix=True,
            )

        if not self._dcp_batch_has_prefix(forward_batch, kv_indices):
            return current_out.reshape(-1, layer.tp_q_head_num * layer.v_head_dim).to(
                q.dtype
            )

        # Prefix KV is sharded across DCP ranks, so compute each rank's
        # partial attention with all gathered query heads and merge by LSE.
        q_all = self._dcp_gather_q_heads(q_local, group).contiguous()
        total_heads = q_all.shape[1]
        prefix_out = torch.zeros(
            (total_tokens, total_heads, layer.v_head_dim),
            device=q.device,
            dtype=torch.float32,
        )
        prefix_lse = torch.full(
            (total_tokens, total_heads),
            -float("inf"),
            device=q.device,
            dtype=torch.float32,
        )
        # The prefix stage reads k_buffer/v_buffer, but extend_attention_fwd
        # takes its GQA grouping from the tensors it is handed
        # (kv_group_num = q_extend.shape[1] // k_extend.shape[1]) -- and the
        # k_extend it is handed here is EMPTY. So this placeholder's head count
        # is what decides how the pool is indexed, and it must be the POOL's kv
        # head count. On the even path that is this rank's projection count and
        # `k[:0]` is right; under the uneven lane the pool row holds all
        # total_num_kv_heads while a head-sharded rank projects fewer, so the
        # placeholder is built from self.num_kv_head (which the constructor set
        # to the pool's count for that lane). Same dtype/device as k either way,
        # so the compiled kernel variant is unchanged on the even path.
        if self.uneven_dcp and not self.dcp_kv_replicated_heads:
            empty_k = k.new_empty((0, self.num_kv_head, k.shape[-1]))
            empty_v = v.new_empty((0, self.num_kv_head, v.shape[-1]))
        else:
            empty_k = k[:0].contiguous()
            empty_v = v[:0].contiguous()
        self.extend_attention_fwd(
            q_all,
            empty_k,
            empty_v,
            prefix_out,
            k_buffer,
            v_buffer,
            self.forward_metadata.qo_indptr,
            kv_indptr,
            kv_indices,
            None,
            False,
            None,
            max_extend_len,
            k_descale,
            v_descale,
            sm_scale=layer.scaling,
            logit_cap=logits_soft_cap,
            xai_temperature_len=layer.xai_temperature_len,
            lse_extend=prefix_lse,
            skip_extend=True,
        )

        prefix_out, prefix_lse = self._dcp_merge_q_heads(
            prefix_out, prefix_lse, group, layer.tp_q_head_num, return_lse=True
        )
        final_lse = torch.logaddexp(prefix_lse, current_lse)
        prefix_scale = torch.exp(prefix_lse - final_lse).unsqueeze(-1)
        current_scale = torch.exp(current_lse - final_lse).unsqueeze(-1)
        prefix_scale = torch.nan_to_num(prefix_scale, nan=0.0, posinf=0.0, neginf=0.0)
        current_scale = torch.nan_to_num(current_scale, nan=0.0, posinf=0.0, neginf=0.0)
        out = prefix_out * prefix_scale + current_out * current_scale
        return out.reshape(-1, layer.tp_q_head_num * layer.v_head_dim).to(q.dtype)

    def _forward_extend_unified(
        self,
        q: torch.Tensor,
        o: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        causal: bool,
        logits_soft_cap: float,
        sinks: Optional[torch.Tensor],
    ):
        """
        Unified 1-stage extend attention for deterministic inference.
        Both prefix and extend KV are accessed through unified kv_indices.
        """
        bs = forward_batch.batch_size

        # Determine sliding window settings
        if layer.sliding_window_size is not None and layer.sliding_window_size > -1:
            sliding_window_size = layer.sliding_window_size
            # Note: for unified kernel, we use full kv_indptr (not window)
            prefix_kv_indptr = self.forward_metadata.window_kv_indptr
            prefix_kv_indices = self.forward_metadata.window_kv_indices
            # Compute window start positions (absolute position of first key in window)
            # window_start_pos = seq_len - window_len
            window_kv_lens = prefix_kv_indptr[1 : bs + 1] - prefix_kv_indptr[:bs]
            # Handle TARGET_VERIFY mode where extend_prefix_lens might not be set
            if forward_batch.extend_prefix_lens is not None:
                window_start_pos = (
                    forward_batch.extend_prefix_lens[:bs] - window_kv_lens
                )
            else:
                # Infer from spec_info: prefix_len = seq_len - draft_token_num
                if forward_batch.spec_info is not None and hasattr(
                    forward_batch.spec_info, "draft_token_num"
                ):
                    extend_prefix_lens = (
                        forward_batch.seq_lens[:bs]
                        - forward_batch.spec_info.draft_token_num
                    )
                    window_start_pos = extend_prefix_lens - window_kv_lens
                else:
                    window_start_pos = None
        else:
            sliding_window_size = -1
            prefix_kv_indptr = self.forward_metadata.kv_indptr
            prefix_kv_indices = self.forward_metadata.kv_indices
            window_start_pos = None

        extend_kv_indices = forward_batch.out_cache_loc
        pool = self.token_to_kv_pool
        if (
            layer.sliding_window_size is not None
            and layer.sliding_window_size > -1
            and isinstance(pool, SWAKVPool)
            and pool.layers_mapping[layer.layer_id][1]
        ):
            extend_kv_indices = pool.translate_loc_from_full_to_swa(extend_kv_indices)

        # Handle cases where extend_seq_lens or extend_start_loc might not be set
        # In speculative decoding, we can infer these from spec_info or compute them
        if forward_batch.extend_seq_lens is None:
            # TARGET_VERIFY mode: infer extend_seq_lens from spec_info
            if forward_batch.spec_info is not None and hasattr(
                forward_batch.spec_info, "draft_token_num"
            ):
                draft_token_num = forward_batch.spec_info.draft_token_num
                extend_seq_lens = torch.full(
                    (bs,), draft_token_num, dtype=torch.int32, device=self.device
                )
            else:
                raise RuntimeError(
                    "extend_seq_lens is None but cannot infer from spec_info. "
                    "This should not happen in TARGET_VERIFY mode."
                )
        else:
            extend_seq_lens = forward_batch.extend_seq_lens

        # Check extend_start_loc separately - it might be None even when extend_seq_lens is set
        if forward_batch.extend_start_loc is None:
            # Compute extend_start_loc from extend_seq_lens
            # extend_start_loc[i] = sum(extend_seq_lens[0:i])
            extend_start_loc = torch.cat(
                [
                    torch.zeros(1, dtype=torch.int32, device=self.device),
                    torch.cumsum(extend_seq_lens[:-1], dim=0),
                ]
            )
        else:
            extend_start_loc = forward_batch.extend_start_loc

        unified_kv_indptr, unified_kv_indices, prefix_lens = (
            self.build_unified_kv_indices(
                prefix_kv_indptr,
                prefix_kv_indices,
                extend_start_loc,
                extend_seq_lens,
                extend_kv_indices,
                bs,
            )
        )

        # Convert prefix_lens to int32 for the kernel
        prefix_lens = prefix_lens.to(torch.int32)

        if layer.k_scale is not None and layer.v_scale is not None:
            k_descale = layer.k_scale_float
            v_descale = layer.v_scale_float
        else:
            k_descale = 1.0
            v_descale = 1.0

        # Call unified kernel
        self.extend_attention_fwd_unified(
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            self.token_to_kv_pool.get_key_buffer(layer.layer_id),
            self.token_to_kv_pool.get_value_buffer(layer.layer_id),
            k_descale,
            v_descale,
            self.forward_metadata.qo_indptr,
            unified_kv_indptr,
            unified_kv_indices,
            prefix_lens,
            self.forward_metadata.max_extend_len,
            custom_mask=self.forward_metadata.custom_mask,
            mask_indptr=self.forward_metadata.mask_indptr,
            sm_scale=layer.scaling,
            logit_cap=logits_soft_cap,
            is_causal=causal,
            sliding_window_size=sliding_window_size,
            sinks=sinks,
            window_start_pos=window_start_pos,
            xai_temperature_len=layer.xai_temperature_len,
            page_size=self.page_size,
        )

        return o

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        sinks=None,
    ):
        # During torch.compile, there is a bug in rotary_emb that causes the
        # output value to have a 3D tensor shape. This reshapes the output correctly.
        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)

        # TODO: reuse the buffer across layers
        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        logits_soft_cap = logit_capping_mod(layer.logit_capping_method, layer.logit_cap)

        if save_kv_cache:
            if self.use_mla:
                if layer.k_scale is not None:
                    # MLATokenToKVPool doesn't accept scale parameters; k is unused
                    # after this point in decode, so scale in place.
                    k.div_(layer.k_scale)
                self.token_to_kv_pool.set_kv_buffer(
                    layer,
                    forward_batch.out_cache_loc,
                    k,
                    v,
                )
            else:
                self._set_kv_buffer(
                    forward_batch,
                    layer,
                    KVWriteLoc(
                        forward_batch.out_cache_loc,
                        self.forward_metadata.swa_out_cache_loc,
                        full_loc=self.forward_metadata.out_cache_loc_full_physical,
                    ),
                    k,
                    v,
                    layer.k_scale,
                    layer.v_scale,
                )

        if layer.sliding_window_size is not None and layer.sliding_window_size > -1:
            kv_indptr = self.forward_metadata.window_kv_indptr
            kv_indices = self.forward_metadata.window_kv_indices
        else:
            kv_indptr = self.forward_metadata.kv_indptr
            kv_indices = self.forward_metadata.kv_indices

        if layer.k_scale is not None and layer.v_scale is not None:
            k_descale = layer.k_scale_float
            v_descale = layer.v_scale_float
        else:
            k_descale = 1.0
            v_descale = 1.0

        # Select the correctly-sized attn_logits buffer for this layer.
        # The triton kernel's // Lv stride trick requires attn_logits.shape[-1]
        # to exactly match the layer's v_head_dim.
        attn_logits = self.forward_metadata.attn_logits
        if (
            self.forward_metadata.swa_attn_logits is not None
            and layer.v_head_dim == self.swa_v_head_dim
        ):
            attn_logits = self.forward_metadata.swa_attn_logits

        # #96: a sliding-window layer under the SWA-hybrid lane skips the DCP
        # gather/merge entirely and takes the stock decode call below, reading
        # window_kv_indptr/window_kv_indices (unsharded, full->swa translated)
        # against its local kv-head shard. Every rank does the same work for that
        # layer, which is the point: its KV is replicated, not sharded.
        if self.dcp_size > 1 and self._dcp_layer_token_sharded(layer):
            group = get_parallel().dcp_group
            with use_symmetric_memory(group):
                q_for_decode = q.view(
                    -1, layer.tp_q_head_num, layer.qk_head_dim
                ).contiguous()
            q_for_decode = self._dcp_gather_q_heads(
                q_for_decode, group
            ).contiguous()
            o_for_decode = torch.empty(
                (q_for_decode.shape[0], q_for_decode.shape[1], layer.v_head_dim),
                dtype=torch.float32,
                device=q.device,
            )
            self.forward_metadata.attn_lse.fill_(-float("inf"))
            self.decode_attention_fwd(
                q_for_decode,
                self.token_to_kv_pool.get_key_buffer(layer.layer_id),
                self.token_to_kv_pool.get_value_buffer(layer.layer_id),
                o_for_decode,
                kv_indptr,
                kv_indices,
                attn_logits,
                self.forward_metadata.attn_lse,
                self.forward_metadata.num_kv_splits,
                self.max_kv_splits,
                layer.scaling,
                k_descale,
                v_descale,
                logit_cap=logits_soft_cap,
                sinks=sinks,
                xai_temperature_len=layer.xai_temperature_len,
            )
            local_lse = torch.logsumexp(
                self.forward_metadata.attn_lse[
                    : q_for_decode.shape[0], : q_for_decode.shape[1], :
                ],
                dim=-1,
            )
            o = self._dcp_merge_q_heads(
                o_for_decode, local_lse, group, layer.tp_q_head_num
            )
            return o.reshape(-1, layer.tp_q_head_num * layer.v_head_dim).to(q.dtype)

        self.decode_attention_fwd(
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            self.token_to_kv_pool.get_key_buffer(layer.layer_id),
            self.token_to_kv_pool.get_value_buffer(layer.layer_id),
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            kv_indptr,
            kv_indices,
            attn_logits,
            self.forward_metadata.attn_lse,
            self.forward_metadata.num_kv_splits,
            self.max_kv_splits,
            layer.scaling,
            k_descale,
            v_descale,
            logit_cap=logits_soft_cap,
            sinks=sinks,
            xai_temperature_len=layer.xai_temperature_len,
            has_mla=self.use_mla,
            use_pdl=self.use_pdl,
            page_size=self.page_size,
        )
        return o


class TritonMultiStepDraftBackend:
    """
    Wrap multiple triton attention backends as one for multiple consecutive
    draft decoding steps.
    """

    needs_cpu_seq_lens: bool = False

    def __init__(
        self,
        model_runner: ModelRunner,
        topk: int,
        speculative_num_steps: int,
    ):
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        max_bs = model_runner.req_to_token_pool.size * self.topk
        self.kv_indptr = torch.zeros(
            (
                self.speculative_num_steps,
                max_bs + 1,
            ),
            dtype=torch.int32,
            device=model_runner.device,
        )
        self.attn_backends: List[TritonAttnBackend] = []
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends.append(
                TritonAttnBackend(
                    model_runner,
                    skip_prefill=True,
                    kv_indptr_buf=self.kv_indptr[i],
                )
            )
        self.max_context_len = self.attn_backends[0].max_context_len
        self.num_head = _plan_aware_num_q_heads(model_runner.model_config)
        self.device = model_runner.device
        # Cached variables for generate_draft_decode_kv_indices
        self.req_to_token_pool = model_runner.req_to_token_pool
        self.pool_len = model_runner.req_to_token_pool.req_to_token.shape[1]
        self.page_size = model_runner.server_args.page_size

    def common_template(
        self,
        forward_batch: ForwardBatch,
        kv_indices_buffer: Optional[torch.Tensor],
        call_fn: int,
    ):
        if kv_indices_buffer is None:
            kv_indices_buffer = self.cuda_graph_kv_indices

        num_seqs = forward_batch.batch_size
        bs = self.topk * num_seqs
        seq_lens_sum = forward_batch.seq_lens_sum
        if seq_lens_sum is None:
            # seq_lens_sum here only slice-clamps a preallocated kv_indices buffer;
            # over-estimate is safe. Use a static UB to skip the per-iter .sum().item() D2H.
            seq_lens_sum = num_seqs * self.max_context_len

        generate_draft_decode_kv_indices[
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

        if call_fn is None:
            return

        for i in range(self.speculative_num_steps - 1):
            forward_batch.spec_info.kv_indptr = self.kv_indptr[i, : bs + 1]
            forward_batch.spec_info.kv_indices = kv_indices_buffer[i][
                : draft_kv_indices_used_len(seq_lens_sum, self.topk, bs, i + 1)
            ]
            call_fn(i, forward_batch)

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        kv_indices_width = draft_kv_indices_buffer_width(
            forward_batch.batch_size, self.topk, self.max_context_len
        )
        kv_indices = torch.empty(
            (self.speculative_num_steps, kv_indices_width),
            dtype=torch.int64,
            device=self.device,
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
        kv_indices_width = draft_kv_indices_buffer_width(
            max_bs, self.topk, self.max_context_len
        )
        self.cuda_graph_kv_indices = torch.zeros(
            (self.speculative_num_steps, kv_indices_width),
            dtype=torch.int64,
            device=self.device,
        )
        self.cuda_graph_num_kv_splits = torch.full(
            (max_num_tokens,),
            self.attn_backends[0].max_kv_splits,
            dtype=torch.int32,
            device=self.device,
        )

        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_cuda_graph_state(
                max_bs,
                max_num_tokens,
                kv_indices_buf=self.cuda_graph_kv_indices[i],
                cuda_graph_num_kv_splits_buf=self.cuda_graph_num_kv_splits,
            )

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        from sglang.srt.model_executor.forward_batch_info import build_inner_fb_view

        if in_capture:
            inner_fb = build_inner_fb_view(
                forward_batch,
                bs=forward_batch.batch_size,
                forward_mode=ForwardMode.DECODE,
            )

            def call_fn(i, _forward_batch):
                self.attn_backends[i].init_forward_metadata_out_graph(
                    inner_fb, in_capture=True
                )

            self.common_template(forward_batch, None, call_fn)
        else:
            bs = forward_batch.batch_size
            self.common_template(forward_batch, None, None)

            # NOTE: Multi-step's attention backends use the slice of
            # - kv_indptr buffer (cuda graph and non-cuda graph)
            # - kv_indices buffer (cuda graph only)
            # So we don't need to assign the KV indices inside the attention backend.

            # Compute num_kv_splits only once
            num_token = bs * self.topk
            self.attn_backends[-1].get_num_kv_splits(
                self.attn_backends[-1].cuda_graph_num_kv_splits[:num_token],
                forward_batch.seq_lens[:bs],
            )

    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch) -> None:
        for attn_backend in self.attn_backends:
            attn_backend.init_forward_metadata_in_graph(forward_batch)


def update_sliding_window_buffer(
    window_kv_indptr,
    req_to_token,
    sliding_window_size,
    seq_lens,
    req_pool_indices,
    bs,
    device=None,
    token_to_kv_pool=None,
    window_kv_indices=None,
    skip_full_to_swa_translation=False,
):
    """Fill window KV buffers for sliding-window attention.

    Pass ``window_kv_indices`` to write into a pre-allocated buffer (CUDA-graph
    path); omit it (or pass ``None``) to allocate a fresh tensor (eager path,
    requires ``device``).

    ``skip_full_to_swa_translation=True`` leaves ``window_kv_indices`` as VIRTUAL
    full-token ids (no eager full->swa translate). The unified-memory-pool cuda-graph
    builder passes this so the window translate is deferred to
    ``TritonAttnBackend._translate_cuda_graph_shared_pool_locs`` (run in
    ``init_forward_metadata_out_graph``, BEFORE ``graph.replay()``), which reads
    the live v2p and rewrites the static window buffer to swa-physical in place;
    baseline SWA leaves it False (eager).
    """
    window_kv_lens = torch.minimum(
        seq_lens,
        torch.tensor(sliding_window_size),
    )
    window_kv_indptr[1 : bs + 1] = torch.cumsum(window_kv_lens, dim=0)
    window_kv_indptr = window_kv_indptr[: bs + 1]
    if window_kv_indices is None:
        window_kv_indices = torch.empty(
            window_kv_indptr[-1], dtype=torch.int64, device=device
        )
    window_kv_start_idx = seq_lens - window_kv_lens
    create_flashinfer_kv_indices_triton[(bs,)](
        req_to_token,
        req_pool_indices,
        window_kv_lens,
        window_kv_indptr,
        window_kv_start_idx,
        window_kv_indices,
        req_to_token.stride(0),
    )
    if not skip_full_to_swa_translation and hasattr(
        token_to_kv_pool, "translate_loc_from_full_to_swa"
    ):
        kv_last_index = window_kv_indptr[-1]
        window_kv_indices[:kv_last_index] = (
            token_to_kv_pool.translate_loc_from_full_to_swa(
                window_kv_indices[:kv_last_index]
            )
        )
    return window_kv_indptr, window_kv_indices, window_kv_lens, window_kv_start_idx
