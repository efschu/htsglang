"""Memory pool configurators for profiling and sizing KV cache pools.

Each model architecture has its own configurator that computes pool sizes
from available GPU memory using a unified coeff+bias model:

    available_bytes = max_tokens * coeff + bias
    max_tokens = (available_bytes - bias) / coeff

Two entry points, same core computation:
- calculate_pool_sizes(available_bytes, page_size): profiling path
- calculate_pool_sizes_from_max_tokens(max_tokens, page_size): constraint path
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.configs.model_config import (
    get_dsa_index_head_dim,
    get_minimax_sparse_attention_config,
    get_minimax_sparse_disable_value_layer_ids,
    get_minimax_sparse_layer_ids,
    is_deepseek_dsa,
    is_deepseek_v4,
    is_minimax_sparse,
)
from sglang.srt.environ import envs
from sglang.srt.mem_cache.common import get_alloc_len_per_decode
from sglang.srt.mem_cache.deepseek_v4_memory_pool import get_compress_state_ring_size
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils.common import (
    ceil_align,
    ceil_div,
    is_float4_e2m1fn_x2,
    spec_decode_alloc_len_per_request,
)


@dataclass
class MemoryPoolConfig:
    """Resolved memory pool config, shared between target and draft workers."""

    max_total_num_tokens: int
    max_running_requests: Optional[int] = None
    full_max_total_num_tokens: Optional[int] = None
    swa_max_total_num_tokens: Optional[int] = None

    # DSV4 compressed-attention pool sizes (target only; draft workers leave at 0).
    c4_max_total_num_tokens: int = 0
    c128_max_total_num_tokens: int = 0
    c4_state_pool_size: int = 0
    c128_state_pool_size: int = 0

    mem_fraction_static: Optional[float] = None

    def __post_init__(self):
        if self.max_total_num_tokens <= 0:
            msg = "Not enough memory. Please try to increase --mem-fraction-static."
            if self.mem_fraction_static is not None:
                msg += f" Current value: mem_fraction_static={self.mem_fraction_static}"
            raise RuntimeError(msg)


if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


def _get_dsv4_compress_state_dtype_sizes() -> tuple[int, int]:
    dtype_name = envs.SGLANG_DSV4_COMPRESS_STATE_DTYPE.get().strip().lower()
    if dtype_name in ("float32", "fp32"):
        return 4, 4
    if dtype_name in ("bfloat16", "bf16"):
        if envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get():
            raise ValueError(
                "SGLANG_DSV4_COMPRESS_STATE_DTYPE=bf16 is not supported when "
                "SGLANG_OPT_USE_ONLINE_COMPRESS=1; online c128 state must stay float32."
            )
        return 2, 2
    raise ValueError(
        "Unsupported SGLANG_DSV4_COMPRESS_STATE_DTYPE="
        f"{dtype_name!r}. Expected one of: float32, fp32, bfloat16, bf16."
    )


def solo_draft_kv_cell_factor(mr: ModelRunner) -> float:
    """Multiplier for the DRAFT-KV part of a TARGET rank's per-token cell under
    ``--speculative-draft-placement solo``.  1.0 = unchanged.

    Why this is needed at all.  The configurator charges the draft KV pool by
    inflating the per-token cell: ``P_r = A_r / (t_target + t_draft)``, i.e. the
    draft is charged once per LOCAL physical token of this rank.  That is right
    under split placement, where every rank owns a draft-KV slice in the same
    proportion as its target-KV slice.  Under solo placement it is wrong on both
    kinds of rank:

    * SHADOW ranks allocate NO draft KV pool at all (the draft is a meta-device
      stub there), yet they are charged for a full one -> factor 0.0.
    * The solo HOST allocates ONE draft KV pool sized to the GLOBAL
      ``max_total_num_tokens`` C, because the draft runner inherits the target's
      MemoryPoolConfig unsharded.  Its target pool, however, holds only its own
      token share ``n_host = C * ratio_host / S`` under uneven-DCP token
      sharding (or ``C / dcp_size`` under the even owner rule).  Charging the
      draft once per LOCAL token therefore under-counts it by exactly
      ``S / ratio_host`` -> that is the factor.  Additionally the solo draft is
      built weight-TP=1, so it keeps ALL kv heads; when the target cell was
      computed with this rank's head SHARD (no DCP kv replication) the draft
      term needs the head ratio on top.

    The correct invariant is ``n_r * t_target + [host] * C * t_draft <= A_r``,
    which is exactly what ``P_r = A_r / (t_target + factor * t_draft)`` encodes
    once ``P_r`` is interpreted as ``C * ratio_r / S`` -- the definition
    ``_apply_token_constraints`` already uses.

    KNOWN, DELIBERATELY UNFIXED (split placement): the same "draft KV is sized
    to the GLOBAL token count, not this rank's share" mismatch exists for SPLIT
    placement under token-sharded DCP too, where it under-charges the draft on
    every rank.  It is left alone here on purpose -- fixing it would shrink the
    KV pool of every validated non-solo uneven-DCP arm, and the reference
    topology has enough slack to absorb it.  See the handoff note.

    #108 RESOLVES THAT MISMATCH FROM THE OTHER SIDE, for split placement, when
    ``--draft-kv-layout dcp`` is set: instead of correcting the CHARGE upward
    to match a globally-sized draft pool, it shrinks the POOL to
    ``C * ratio_r / S`` -- which is exactly what the ``1 + L_draft/L_target``
    per-LOCAL-token term above already charges.  Nothing here changes, and
    that is the point: the accounting was always written for the sharded
    shape, and the draft pool now has it.  Under the default
    ``--draft-kv-layout replicated`` the under-charge stands as described.
    """
    server_args = mr.server_args
    if not getattr(server_args, "speculative_draft_solo_active", None):
        return 1.0
    # T156 stage 2: under --speculative-cross-algorithm the draft-KV part of
    # the target cell is exclusively the DFLASH rung's pool, and that rung is
    # ALWAYS solo on rank 0 -- even when the forced (global) placement is
    # 'split' because NEXTN is the active rung. The NEXTN/MTP pool stays
    # uncharged either way, matching the plain NEXTN server.
    _cross_algo = getattr(server_args, "speculative_cross_algorithm", False)
    if not server_args.speculative_draft_solo_active() and not _cross_algo:
        return 1.0
    if mr.is_draft_worker:
        # The draft runner does not size anything (it is handed the target's
        # MemoryPoolConfig); leave its own cell computation untouched.
        return 1.0
    if mr.tp_rank != server_args.speculative_draft_solo_rank():
        return 0.0  # shadow rank: no draft KV pool exists here

    from sglang.srt.distributed.utils import (
        cp_token_split_factor,
        get_cp_token_ratios,
        uneven_dcp_active,
        uneven_dcp_kv_replicated,
    )

    dcp_size = int(getattr(mr, "dcp_size", 1) or 1)
    factor = 1.0
    # (a) token-axis share: the host's draft pool spans ALL C tokens.
    if uneven_dcp_active(dcp_size):
        ratios = get_cp_token_ratios()
        if ratios and len(ratios) == dcp_size:
            dcp_rank = int(get_parallel().attn_dcp_rank)
            ratio_r = int(ratios[dcp_rank])
            if ratio_r > 0:
                factor *= cp_token_split_factor(dcp_size) / ratio_r
    elif dcp_size > 1:
        factor *= float(dcp_size)
    # (b) head-axis share: the solo draft keeps all kv heads. When the target
    # cell already used the FULL kv-head count (uneven-DCP kv replication) the
    # draft term is on the same footing and no head correction applies.
    if not uneven_dcp_kv_replicated(get_parallel().attn_dcp_size):
        model_config = mr.model_config
        tp_size = get_parallel().attn_tp_size
        local_kv_heads = int(model_config.get_num_kv_heads(tp_size))
        total_kv_heads = int(model_config.get_total_num_kv_heads())
        if local_kv_heads > 0 and total_kv_heads > local_kv_heads:
            factor *= total_kv_heads / local_kv_heads
    return factor


def apply_solo_draft_kv_cell_factor(
    mr: ModelRunner, target_cell_size: int, cell_size_with_draft: int
) -> int:
    """Re-scale the draft-KV component of ``cell_size_with_draft`` (the part
    above ``target_cell_size``) by :func:`solo_draft_kv_cell_factor`.

    Returns ``cell_size_with_draft`` unchanged whenever the factor is 1.0, so
    every split-placement / non-speculative path stays byte-identical."""
    draft_part = int(cell_size_with_draft) - int(target_cell_size)
    if draft_part <= 0:
        return int(cell_size_with_draft)
    factor = solo_draft_kv_cell_factor(mr)
    if factor == 1.0:
        return int(cell_size_with_draft)
    scaled = int(target_cell_size) + int(round(draft_part * factor))
    logger.info(
        "Draft-solo KV planning: rank %d draft-KV cell term %d -> %d B/token "
        "(factor %.3f); per-token cell %d -> %d B.",
        mr.tp_rank,
        draft_part,
        scaled - int(target_cell_size),
        factor,
        int(cell_size_with_draft),
        scaled,
    )
    return max(int(target_cell_size), scaled)


class MemoryPoolConfigurator:
    """Base class for memory pool configurators.

    Subclasses compute pool sizes for their architecture via coeff+bias model.
    Both entry points return MemoryPoolConfig (with max_running_requests=None,
    to be filled by the consumer).
    """

    # A pipeline stage can legitimately hold NO full-attention layers, and then
    # its KV cell is 0 bytes per token. That is not a degenerate config: on a
    # gapped layer set over a hybrid model, the stage carrying only GDN/linear
    # layers keeps its state in the mamba pool and needs no KV cache at all.
    #
    # Such a stage must impose NO KV bound on the pipeline. Dividing by its cell
    # is a ZeroDivisionError; reporting 0 would be worse -- the pipeline's token
    # universe is the MINIMUM across stages, so a 0 here would collapse the pool
    # to nothing on the strength of a stage that does not use it (and
    # MemoryPoolConfig refuses <= 0 outright).
    #
    # THE MAGNITUDE IS CHOSEN, and the choice was corrected by a boot rather
    # than reasoned into place. Two requirements pull against each other:
    #
    #   NON-BINDING   it must exceed the largest universe a real stage can
    #                 bound, or it becomes the minimum and caps the pipeline.
    #                 Measured on this rig: 845279 tokens.
    #   ALLOCATABLE   max_total_num_tokens is not only COMPARED. Structures are
    #                 sized from it on this rank too, and they do not all scale
    #                 with the (zero) KV cell.
    #
    # 2**24 was tried first, on the assumption that a zero cell meant zero
    # allocation. It does not: the kvless stage went to `torch reserved
    # 49.70 GiB` on a 32.6 GiB card and died on cuMemCreate -- roughly
    # 16.7M tokens x ~1 KiB of per-token structure that is not the KV cell.
    # 2**20 clears the measured 845279 by 1.24x and costs about a sixteenth of
    # that allocation, which fits the headroom a GDN-only stage actually has.
    #
    # math.inf is unusable regardless: the value flows into integer arithmetic
    # (`// page_size`, MemoryPoolConfig fields) with no float path.
    #
    # IF A STAGE UNIVERSE EVER EXCEEDS THIS, the symptom is a silently capped
    # pool, so it is asserted rather than left to be noticed later.
    _KVLESS_STAGE_TOKENS = 1 << 20

    def calculate_pool_sizes(
        self, available_bytes: int, page_size: int
    ) -> MemoryPoolConfig:
        """Profiling path: compute pool sizes from available bytes."""
        raise NotImplementedError

    def calculate_pool_sizes_from_max_tokens(
        self, max_total_num_tokens: int, page_size: int
    ) -> MemoryPoolConfig:
        """Constraint path: recalculate pool sizes from a constrained max_tokens."""
        raise NotImplementedError

    def finalize_with_max_running_requests(
        self, config: MemoryPoolConfig
    ) -> MemoryPoolConfig:
        return config


class DefaultPoolConfigurator(MemoryPoolConfigurator):
    """Configurator for standard models: MHA, MLA, DSA, FP4.

    coeff = cell_size (bytes per token across all layers)
    bias = 0
    """

    def __init__(self, mr: ModelRunner):
        # Determine effective number of layers for KV cache
        if mambaish := mr.mambaish_config:
            # #pp-layer-set: [start_layer, end_layer) is the stage's SPAN, and
            # the span equals the owned set only while a stage is a contiguous
            # interval. Under SGLANG_PP_LAYER_SET it does not. The same
            # distinction was already fixed for ``num_effective_layers``
            # (model_runner.py, "end - start is the SPAN, which equals the
            # COUNT only while a stage is a contiguous interval") -- and the
            # non-mambaish branch below inherits that fix for free by reading
            # num_effective_layers. This branch does not, so it kept sizing
            # from the span.
            #
            # MEASURED COST OF THE OMISSION, 2026-08-18, gapped set [48, 8, 8]
            # on a 64-layer hybrid with full_attention_interval=4: stage 0 owns
            # all 48 GDN layers and ZERO full-attention layers, but spans
            # [0, 63), so the span admitted 15 of the 16 full-attention layers.
            # Its cell came out 30720 bytes instead of 0, it reserved 3.3 GiB
            # of KV it can never address on the one card that is already
            # holding 25.7 GiB of weights, and the boot died with cuMemCreate
            # CUDA_ERROR_OUT_OF_MEMORY. Sizing unpinned instead of dying, the
            # same stage then bound the whole pipeline's token universe to
            # 57925 tokens against 845283 and 754019 on the two stages that
            # actually hold the attention layers.
            #
            # Stages 1 and 2 were correct by luck, not by construction: their
            # gapped sets happen to contain every full-attention layer inside
            # their own span. That is exactly the kind of accident that keeps a
            # span-based rule looking right until a layout moves.
            owned = getattr(mr, "owned_layer_ids", None)
            if owned is None:
                from sglang.srt.distributed.utils import get_pp_layer_set

                owned = get_pp_layer_set(
                    mr.model_config.num_hidden_layers, mr.pp_rank, mr.pp_size
                )
            if owned is not None:
                effective_layer_ids = [
                    i for i in mambaish.full_attention_layer_ids if i in owned
                ]
            else:
                # Contiguous stage (and the no-PP case): byte-identical to the
                # previous behaviour, deliberately -- get_pp_layer_set returns
                # None whenever the set form is unused.
                effective_layer_ids = [
                    i
                    for i in mambaish.full_attention_layer_ids
                    if mr.start_layer <= i < mr.end_layer
                ]
            num_layers = len(effective_layer_ids)
        else:
            num_layers = mr.num_effective_layers

        target_cell_size = self._compute_cell_size(mr, num_layers)
        self._cell_size = target_cell_size

        # EAGLE/STANDALONE: scale cell_size to account for draft model KV cache.
        # Assumes draft and target share the same per-layer KV size (head_dim,
        # num_kv_heads, dtype), which holds for EAGLE/MTP draft models that
        # reuse the target architecture's attention config.
        if (
            mr.spec_algorithm.is_eagle() or mr.spec_algorithm.is_standalone()
        ) and not mr.is_draft_worker:
            eagle_draft_num_layers = getattr(mr, "eagle_draft_num_layers", None)
            if (
                eagle_draft_num_layers is not None
                and int(eagle_draft_num_layers) > 0
                and int(num_layers) > 0
            ):
                self._cell_size = int(
                    self._cell_size
                    * (1 + int(eagle_draft_num_layers) / int(num_layers))
                )

        # DFLASH/DSPARK: scale cell_size to account for draft model KV cache.
        # Under --speculative-cross-algorithm (T156 stage 2) the DFLASH draft
        # pool is resident even when the forced rung is NEXTN
        # (spec_algorithm == EAGLE), so the inflation applies there too; the
        # planning fields were filled by ModelRunner from the stashed dflash
        # shape.
        _cross_algo = (not mr.is_draft_worker) and getattr(
            mr.server_args, "speculative_cross_algorithm", False
        )
        if (
            mr.spec_algorithm.is_dflash_family() or _cross_algo
        ) and not mr.is_draft_worker:
            from sglang.srt.speculative.dflash_utils import (
                scale_kv_cell_size_per_token_for_dflash,
            )

            draft_num_layers = mr.dflash_family_draft_num_layers
            if (
                draft_num_layers is not None
                and int(draft_num_layers) > 0
                and int(num_layers) > 0
            ):
                self._cell_size = scale_kv_cell_size_per_token_for_dflash(
                    target_cell_size_per_token=self._cell_size,
                    target_num_layers=int(num_layers),
                    draft_num_layers=int(draft_num_layers),
                )

        # Draft-solo placement: re-scale ONLY the draft-KV part of the cell.
        # No-op (factor 1.0, byte-identical) on every split-placement path.
        self._cell_size = apply_solo_draft_kv_cell_factor(
            mr, target_cell_size, self._cell_size
        )

    def _compute_cell_size(self, mr: ModelRunner, num_layers: int) -> int:
        """Compute per-token KV cache cost in bytes. Subclasses can override."""
        # args to config cell size
        model_config = mr.model_config
        kv_cache_dtype = mr.kv_cache_dtype
        from sglang.srt.layers.cp.utils import (
            get_glm_dsa_layer_split_effective_num_layers,
        )

        effective_num_layers = get_glm_dsa_layer_split_effective_num_layers(
            mr, num_layers
        )

        kv_size = torch._utils._element_size(kv_cache_dtype)
        tp_size = get_parallel().attn_tp_size

        if mr.use_mla_backend:
            cell_size = (
                (model_config.kv_lora_rank + model_config.qk_rope_head_dim)
                * effective_num_layers
                * kv_size
            )
            if is_float4_e2m1fn_x2(kv_cache_dtype):
                # kv_scale_buffer
                scale_block_size = 16
                cell_size = (cell_size // 2) + (
                    (
                        (model_config.kv_lora_rank + model_config.qk_rope_head_dim)
                        // scale_block_size
                    )
                    * effective_num_layers
                    * kv_size
                )

            # Add indexer KV cache overhead for DSA models (DeepSeek V3.2)
            if is_deepseek_dsa(model_config.hf_config):
                index_head_dim = get_dsa_index_head_dim(model_config.hf_config)
                indexer_size_per_token = (
                    index_head_dim
                    + index_head_dim // DSATokenToKVPool.quant_block_size * 4
                )
                element_size = torch._utils._element_size(
                    DSATokenToKVPool.index_k_with_scale_buffer_dtype
                )
                cell_size += (
                    indexer_size_per_token * effective_num_layers * element_size
                )
        elif is_minimax_sparse(model_config.hf_config):
            # Mirrors MiniMaxSparseKVPool: main pool (K+V all layers) + indexer pool
            # (sparse-only, single-head; kv layers store K+V, k-only layers store K).
            sparse_cfg = get_minimax_sparse_attention_config(model_config.hf_config)
            dense_layer_ids, sparse_layer_ids = get_minimax_sparse_layer_ids(sparse_cfg)
            indexer_k_only_layer_ids = set(
                get_minimax_sparse_disable_value_layer_ids(sparse_cfg)
            )

            local_dense_layer_ids = [
                l for l in dense_layer_ids if mr.start_layer <= l < mr.end_layer
            ]
            local_sparse_layer_ids = [
                l for l in sparse_layer_ids if mr.start_layer <= l < mr.end_layer
            ]
            num_dense = len(local_dense_layer_ids)
            num_sparse = len(local_sparse_layer_ids)
            num_indexer_k_only = sum(
                1 for l in local_sparse_layer_ids if l in indexer_k_only_layer_ids
            )
            num_indexer_kv = num_sparse - num_indexer_k_only

            kv_heads = model_config.get_num_kv_heads(get_parallel().attn_tp_size)
            head_dim = model_config.head_dim
            indexer_head_dim = sparse_cfg["sparse_index_dim"]
            indexer_dtype_size = torch._utils._element_size(mr.dtype)

            main_pool_bytes = (
                (num_dense + num_sparse) * 2 * kv_heads * head_dim * kv_size
            )
            indexer_bytes = (
                (num_indexer_kv * 2 + num_indexer_k_only)
                * indexer_head_dim
                * indexer_dtype_size
            )
            # FP4 scale buffer adjustment doesn't apply to MiniMax sparse:
            # cell_size is already a sum over heterogeneous sub-pools.
            return main_pool_bytes + indexer_bytes
        else:
            # Uneven-DCP KV replication: every rank stores the FULL kv-heads
            # (replicated across the DCP group, not head-sharded) but only its
            # owned token slots, so per-token bytes reflect the FULL kv-head
            # count, not this rank's uneven projection share. Stock paths keep
            # the per-rank get_num_kv_heads(tp_size).
            # Weightless-KV fast lane: SAME geometry, different trigger. The
            # head rank projects the FULL kv-heads (weight-TP=1 override) and
            # broadcasts them; every rank -- head and weightless worker alike
            # -- writes all total_num_kv_heads into its owned token slots, so
            # the pool is built at get_total_num_kv_heads() (see
            # model_runner_kv_cache_mixin's `_hybrid_kv_head_num` and
            # `_pool_kv_head_num`). The lane runs with rank_tp_ratio=None, so
            # uneven_dcp_kv_replicated() is False and this cell would
            # otherwise be charged the ÷attn_tp_size per-rank share -- i.e.
            # UNDER-charged by total_kv/get_num_kv_heads(tp), inflating
            # max_total_num_tokens by the same factor and sizing the pool
            # past the rank's budget. Add the lane explicitly so the profile
            # math and the allocation agree.
            from sglang.srt.distributed.utils import (
                uneven_dcp_kv_replicated,
                weightless_kv_active,
            )

            if (
                uneven_dcp_kv_replicated(get_parallel().attn_dcp_size)
                or weightless_kv_active()
            ):
                num_kv_heads_cell = model_config.get_total_num_kv_heads()
            else:
                num_kv_heads_cell = model_config.get_num_kv_heads(tp_size)
            cell_size = (
                num_kv_heads_cell
                * (model_config.head_dim + model_config.v_head_dim)
                * effective_num_layers
                * kv_size
            )

            if is_float4_e2m1fn_x2(kv_cache_dtype):
                # kv_scale_buffer
                scale_block_size = 16
                n = model_config.get_num_kv_heads(tp_size)
                k = model_config.head_dim
                cell_size = (cell_size // 2) + (
                    (n * k * effective_num_layers * 2 * kv_size) // scale_block_size
                )

        return cell_size


    def calculate_pool_sizes(
        self, available_bytes: int, page_size: int
    ) -> MemoryPoolConfig:
        max_total_num_tokens = (
            self._KVLESS_STAGE_TOKENS
            if self._cell_size == 0
            else available_bytes // self._cell_size
        )
        max_total_num_tokens = max_total_num_tokens // page_size * page_size
        # #704: emit the LAST link of the sizing chain.
        #
        # The chain is: budget - sum(budget_posts) = rest (already emitted at
        # the profiler's success path), then rest MINUS a per-rank reserve
        # becomes `available_bytes`, then // cell_size becomes tokens. The
        # reserve is the only term never emitted anywhere, and on this rig it
        # is large and wildly non-uniform -- backed out of the live boot it is
        # 8,848 / 3,818 / 5,164 MiB across the three stages, a 2.3x spread. It
        # is not derivable from config: it tracks per-rank CUDA-graph capture,
        # which depends on the shard. planner/plan.py's auto_reserve_mib
        # docstring says the value is one "which the boot itself derives and
        # logs" -- the deriving is real, the logging was not; a 172 MB boot log
        # contains only the server_args echo.
        #
        # With this line the reserve is recoverable as (rest - available_bytes)
        # without a fourth external re-derivation, all three of which missed
        # (+20 %, -3.8 %, -12 %).
        logger.info(
            "KV pool sizing: available_bytes=%d (%.3f GiB), cell_size=%d, "
            "page_size=%d -> max_total_num_tokens=%d",
            int(available_bytes),
            float(available_bytes) / (1 << 30),
            int(self._cell_size),
            int(page_size),
            int(max_total_num_tokens),
        )
        return MemoryPoolConfig(max_total_num_tokens=max_total_num_tokens)

    def calculate_pool_sizes_from_max_tokens(
        self, max_total_num_tokens: int, page_size: int
    ) -> MemoryPoolConfig:
        max_total_num_tokens = max_total_num_tokens // page_size * page_size
        return MemoryPoolConfig(max_total_num_tokens=max_total_num_tokens)


class HybridSWAPoolConfigurator(MemoryPoolConfigurator):
    """Configurator for hybrid sliding window attention models (Gemma2, Command-R, MiMo).

    Splits available memory between full attention and SWA pools.
    Does NOT inherit DefaultPoolConfigurator — different coeff model.
    """

    def __init__(self, mr: ModelRunner):
        model_config = mr.model_config
        kv_cache_dtype = mr.kv_cache_dtype
        kv_size = torch._utils._element_size(kv_cache_dtype)
        tp_size = get_parallel().attn_tp_size

        self._full_layers_num = len(model_config.full_attention_layer_ids)
        self._swa_layers_num = len(model_config.swa_attention_layer_ids)
        assert (
            self._swa_layers_num > 0
        ), "Hybrid SWA model must have at least one SWA layer"

        self._swa_full_tokens_ratio = mr.server_args.swa_full_tokens_ratio

        # Full layer per-token memory (bytes).
        #
        # SWA-HYBRID UNEVEN DCP (#96, Stage B): the full-attention sub-pool's
        # rows carry ALL kv heads on every rank (they are sharded along TOKENS
        # instead, by the owner rule), so a full-pool token costs the REPLICATED
        # cell here -- the same branch DefaultPoolConfigurator already has for
        # the non-hybrid weighted lane. The per-rank token share is applied
        # later and elsewhere: calculate_pool_sizes returns this rank's PHYSICAL
        # token count P_r and _apply_token_constraints turns it into the global
        # context C = min_r(P_r // ratio_r) * S.
        # ``is True``, not ``bool(...)``: the predicate returns a strict bool
        # (ModelRunnerKVCacheMixin._swa_hybrid_dcp_lane), and a MagicMock-based
        # unit ModelRunner must stay on the DEFAULT path rather than be pulled
        # onto the DCP lane by a truthy mock attribute.
        self._swa_dcp_lane = mr._swa_hybrid_dcp_lane() is True
        _full_kv_heads = (
            model_config.get_total_num_kv_heads()
            if self._swa_dcp_lane
            else model_config.get_num_kv_heads(tp_size)
        )
        self._full_per_token = (
            _full_kv_heads * (model_config.head_dim + model_config.v_head_dim) * kv_size
        )

        # SWA layer per-token memory (bytes). NOT touched by the DCP lane: the
        # sliding-window layers keep this rank's kv-head shard AND every token
        # position (window-bounded), so their cost per token is unchanged.
        self._swa_per_token = (
            model_config.get_swa_num_kv_heads(tp_size)
            * (model_config.swa_head_dim + model_config.swa_v_head_dim)
            * kv_size
        )

        # EAGLE/STANDALONE draft KV pool inherits max_total tokens with its
        # full-attn layers; budget into the full term.
        self._draft_full_layers_num = 0
        if (
            mr.spec_algorithm.is_eagle() or mr.spec_algorithm.is_standalone()
        ) and not mr.is_draft_worker:
            draft_layers = getattr(mr, "eagle_draft_num_layers", None)
            if draft_layers is not None and int(draft_layers) > 0:
                self._draft_full_layers_num = int(draft_layers)

        # Bytes per token of max_total_num_tokens.
        #
        # Hybrid (full_layers > 0): max_total = full_tokens, so cell_size accounts
        # for both pools: F*nf + r*S*ns (where swa_tokens = full_tokens * r).
        #
        # All-SWA (full_layers == 0): max_total = swa_tokens directly. The ratio
        # is meaningless here -- there is no full pool to relate to, and every
        # token beyond the sliding window can be evicted. So cell_size = S*ns,
        # with no ratio factor applied.
        if self._full_layers_num == 0:
            self._cell_size = (
                self._swa_per_token * self._swa_layers_num
                + self._full_per_token * self._draft_full_layers_num
            )
        else:
            self._cell_size = (
                self._full_per_token
                * (self._full_layers_num + self._draft_full_layers_num)
                + self._swa_full_tokens_ratio
                * self._swa_per_token
                * self._swa_layers_num
            )

    def _solve_pool_sizes(
        self, max_total_num_tokens: int, page_size: int
    ) -> MemoryPoolConfig:
        """Core computation: split max_total_num_tokens into full/swa pool sizes."""

        def align_page_size(x: int) -> int:
            return (x // page_size) * page_size

        if self._full_layers_num == 0:
            # All-SWA: no full pool, max_total = actual SWA pool size.
            # Ratio is not applied -- see __init__ comment.
            swa_tokens = align_page_size(max_total_num_tokens)
            logger.info(
                f"Use sliding window memory pool (all SWA). "
                f"swa_layer_tokens={swa_tokens}"
            )
            return MemoryPoolConfig(
                max_total_num_tokens=swa_tokens,
                full_max_total_num_tokens=0,
                swa_max_total_num_tokens=swa_tokens,
            )

        # Hybrid: full_tokens = max_total_num_tokens, swa_tokens = full_tokens * ratio
        full_tokens = align_page_size(max_total_num_tokens)
        swa_tokens = align_page_size(int(full_tokens * self._swa_full_tokens_ratio))

        logger.info(
            f"Use sliding window memory pool. "
            f"full_layer_tokens={full_tokens}, swa_layer_tokens={swa_tokens}"
        )

        return MemoryPoolConfig(
            max_total_num_tokens=full_tokens,
            full_max_total_num_tokens=full_tokens,
            swa_max_total_num_tokens=swa_tokens,
        )

    def calculate_pool_sizes(
        self, available_bytes: int, page_size: int
    ) -> MemoryPoolConfig:
        max_total_num_tokens = int(
            self._KVLESS_STAGE_TOKENS
            if self._cell_size == 0
            else available_bytes // self._cell_size
        )
        return self._solve_pool_sizes(max_total_num_tokens, page_size)

    def calculate_pool_sizes_from_max_tokens(
        self, max_total_num_tokens: int, page_size: int
    ) -> MemoryPoolConfig:
        return self._solve_pool_sizes(max_total_num_tokens, page_size)


def swa_pool_token_cap(mr: ModelRunner, num_reqs: int) -> int:
    """Worst-case number of live SWA-pool tokens for ``num_reqs`` concurrent
    requests: sliding window + eviction lag + decode over-allocation per
    request, plus in-flight prefill chunks (or decode-disagg extra slots).

    This is the single source of truth for the SWA pool's per-request
    accounting, shared by SWAChunkCapPoolConfigurator (tight SWA pool sizing)
    and the hybrid-SWA physical token ceiling (#90) in
    ModelRunnerKVCacheMixin._swa_hybrid_kv_token_cap.
    """
    sa = mr.server_args
    page_size = mr.page_size
    window = mr.sliding_window_size
    draft_tokens = sa.speculative_num_draft_tokens or 1
    eviction_interval = max(1, envs.SGLANG_SWA_EVICTION_INTERVAL.get())

    """
    __________[padding][eviction_interval][window]
    Padding to make sure eviction point is page-aligned.
    """
    trailing_tokens = window + eviction_interval * draft_tokens + page_size
    if sa.speculative_algorithm is None:
        decode_alloc = page_size
    elif sa.disable_overlap_schedule:
        # spec-v1: new_tokens_required_next_decode per request.
        decode_alloc = spec_decode_alloc_len_per_request(sa)
    else:
        # spec-v2: the overlap allocator keeps 2 * alloc_len outstanding
        # (eagle_utils.eagle_prepare_for_decode: kv_committed_len + 2 * alloc_len).
        decode_alloc = 2 * get_alloc_len_per_decode(sa)
    per_request = trailing_tokens + decode_alloc

    if sa.disaggregation_mode == "decode":
        return (
            per_request * num_reqs
            + (window + page_size) * sa.disaggregation_decode_extra_slots
        )
    chunks_in_flight = 1 if sa.disable_overlap_schedule else 2
    return (
        per_request * num_reqs
        + chunks_in_flight * (sa.chunked_prefill_size or 0)
        + page_size
    )


class SWAChunkCapPoolConfigurator(HybridSWAPoolConfigurator):
    """Hybrid SWA configurator with the SWA pool sized from a fixed token cap.

    When max_running_requests is explicit, the SWA pool's worst-case
    footprint is bounded per request. The SWA pool is sized tightly from that
    cap and the freed memory is redirected to the full pool, instead of sizing
    both pools by swa_full_tokens_ratio.
    """

    def __init__(self, mr: ModelRunner):
        super().__init__(mr)
        assert self._full_layers_num > 0

        sa = mr.server_args
        num_reqs = sa.max_running_requests // mr.dp_size
        self._swa_cap = swa_pool_token_cap(mr, num_reqs)

    @staticmethod
    def is_applicable(mr: ModelRunner) -> bool:
        """True when SWAChunkCache can be sized from explicit max requests.

        Two routes select this configurator:
        - legacy auto route (conditions byte-identical): radix cache disabled,
          so the SWA pool holds only active requests and the cap is exact;
        - explicit ``--swa-pool-sizing cap`` (Stage A of task #91): the SWA
          pool is pinned at the same window-bounded worst case with the radix
          cache ALLOWED. Correctness holds because scheduler admission counts
          swa_available + swa_evictable and eviction is demand-driven
          (mem_cache/common.py evict_from_tree_cache), so cached in-window
          prefixes are reclaimed under pressure; the pin only bounds SWA-side
          cache RETENTION, trading swa prefix-cache hit rate for moving the
          freed budget to the full-attention pool (the part that actually
          grows with context length). Flag preconditions
          (max_running_requests, chunked prefill) are enforced in
          server_args._handle_cache_compatibility.
        """
        sa = mr.server_args
        if sa.max_running_requests is None:
            return False
        if not sa.disable_radix_cache and sa.swa_pool_sizing != "cap":
            return False
        if sa.chunked_prefill_size is None:
            return False
        if mr.sliding_window_size is None:
            return False
        return len(mr.model_config.full_attention_layer_ids) > 0

    def calculate_pool_sizes(
        self, available_bytes: int, page_size: int
    ) -> MemoryPoolConfig:
        # SWA pool sized tightly from the cap; the rest of the budget goes to full.
        swa_tokens = ceil_align(self._swa_cap, page_size)
        fixed_swa_bytes = swa_tokens * self._swa_per_token * self._swa_layers_num
        full_cell_size = self._full_per_token * (
            self._full_layers_num + self._draft_full_layers_num
        )
        full_tokens = (
            int((available_bytes - fixed_swa_bytes) // full_cell_size) // page_size
        ) * page_size
        if full_tokens <= 0:
            raise RuntimeError(
                f"SWA pool cap ({swa_tokens} tokens, "
                f"{fixed_swa_bytes / (1 << 30):.2f} GiB) leaves no room for the full "
                f"KV pool within the available {available_bytes / (1 << 30):.2f} GiB. "
                f"Reduce --max-running-requests, lower SGLANG_SWA_EVICTION_INTERVAL, "
                f"or increase --mem-fraction-static."
            )
        return MemoryPoolConfig(
            max_total_num_tokens=full_tokens,
            full_max_total_num_tokens=full_tokens,
            swa_max_total_num_tokens=swa_tokens,
        )

    def calculate_pool_sizes_from_max_tokens(
        self, max_total_num_tokens: int, page_size: int
    ) -> MemoryPoolConfig:
        # Constrained max_total goes to the full pool; SWA stays at its cap.
        swa_tokens = ceil_align(self._swa_cap, page_size)
        full_tokens = (max_total_num_tokens // page_size) * page_size
        return MemoryPoolConfig(
            max_total_num_tokens=full_tokens,
            full_max_total_num_tokens=full_tokens,
            swa_max_total_num_tokens=min(swa_tokens, max_total_num_tokens),
        )


@dataclass
class _DSV4PoolSizes:
    full_max_total_num_tokens: int
    swa_max_total_num_tokens: int
    c4_max_total_num_tokens: int
    c128_max_total_num_tokens: int
    c4_state_pool_size: int
    c128_state_pool_size: int


class DSV4PoolConfigurator(MemoryPoolConfigurator):
    """Configurator for DSV4 compressed-attention models.

    Splits available memory across full / swa / c4 / c128 + c4_state / c128_state
    pools. coeff is bytes_per_full_token (inflated by (T+D)/T when speculative
    decode reserves a draft worker, mirroring dflash's cell_size scaling); bias = 0.
    """

    def __init__(self, mr: ModelRunner):
        cfg = mr.model_config
        self.qk_nope_head_dim = cfg.qk_nope_head_dim
        self.qk_rope_head_dim = cfg.qk_rope_head_dim
        self.indexer_head_dim = cfg.index_head_dim
        self.context_len = mr.model_config.context_len
        # PP-local slice; matches DeepSeekV4TokenToKVPool's stage_ratios.
        self.compression_ratios = cfg.compress_ratios[mr.start_layer : mr.end_layer]
        if mr.pp_size > 1:
            logger.info(
                f"DSV4 pool PP slice: rank={mr.pp_group.rank_in_group} "
                f"layers=[{mr.start_layer},{mr.end_layer}) "
                f"local={len(self.compression_ratios)}/{len(cfg.compress_ratios)}"
            )
        self.swa_page_size = cfg.window_size
        self.swa_ratio = mr.server_args.swa_full_tokens_ratio
        self.is_speculative = mr.server_args.speculative_algorithm is not None
        self.online_c128_mtp_max_draft_tokens = (
            mr.server_args.max_speculative_num_draft_tokens or 0
        )
        self.requested_max_running_requests_per_worker = (
            mr.server_args.max_running_requests // mr.dp_size
            if mr.server_args.max_running_requests is not None
            else None
        )
        self.disaggregation_mode = mr.server_args.disaggregation_mode
        self.disaggregation_decode_extra_slots = (
            mr.server_args.disaggregation_decode_extra_slots or 0
        )
        if mr.enable_hisparse:
            from sglang.srt.mem_cache.sparsity import parse_hisparse_config

            self.c4_shrink_factor = parse_hisparse_config(
                mr.server_args
            ).host_to_device_ratio
        else:
            self.c4_shrink_factor = 1
        assert self.c4_shrink_factor >= 1
        if self.c4_shrink_factor > 1:
            logger.info(f"HiSparse c4 host-to-device ratio = {self.c4_shrink_factor}")

        self.c4_ring_size = get_compress_state_ring_size(4, self.is_speculative)
        self.c128_ring_size = get_compress_state_ring_size(128, self.is_speculative)

        self.num_layers_total = len(self.compression_ratios)
        self.num_layers_ca4 = sum(1 for r in self.compression_ratios if r == 4)
        self.num_layers_ca128 = sum(1 for r in self.compression_ratios if r == 128)

        self.bytes_per_full_token = self._get_bytes_per_full_token()
        if self.is_speculative:
            # Reserve memory for the speculative draft worker by inflating
            # per-token bytes by (target+draft)/target. Equivalent to dflash's
            # scale_kv_cell_size_per_token_for_dflash but applied to
            # bytes_per_full_token: tokens = avail / (bpft * (T+D)/T).
            draft_layers = 1
            target_layers = self.num_layers_total
            self.bytes_per_full_token *= (target_layers + draft_layers) / target_layers

        # Online c128 keeps a single in-progress (max, sum, kv) state per index
        # and assumes a strict forward-only schedule. Speculative decode (MTP)
        # would need rollback / replay across draft and verify, which the
        # online path doesn't support yet.
        if envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get():
            allow_experimental_online_c128_mtp = (
                envs.SGLANG_EXPERIMENTAL_ONLINE_C128_MTP.get()
                and mr.spec_algorithm.is_eagle()
            )
            assert mr.spec_algorithm.is_none() or allow_experimental_online_c128_mtp, (
                "SGLANG_OPT_USE_ONLINE_COMPRESS does not support speculative decode "
                "(MTP) yet, except the experimental EAGLE topk=1 path gated by "
                "SGLANG_EXPERIMENTAL_ONLINE_C128_MTP=1"
            )
            if allow_experimental_online_c128_mtp:
                assert self.online_c128_mtp_max_draft_tokens > 0, (
                    "SGLANG_EXPERIMENTAL_ONLINE_C128_MTP requires "
                    "speculative_num_draft_tokens to be set."
                )
                logger.warning(
                    "DSV4 compressed attention: experimental online c128 + MTP enabled "
                    f"(EAGLE topk=1 only, "
                    f"draft_banks={self.online_c128_mtp_max_draft_tokens}). "
                    "Validate correctness carefully."
                )
            else:
                logger.info(
                    "DSV4 compressed attention: online c128 enabled (ring_size=1)"
                )

    def _get_bytes_per_full_token(self) -> float:
        kv_bytes = self.qk_nope_head_dim + self.qk_rope_head_dim * 2 + 8

        quant_block_size = 128
        indexer_bytes = (
            self.indexer_head_dim + self.indexer_head_dim // quant_block_size * 4
        )

        attn_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        c4_state_dtype_size, c128_state_dtype_size = (
            _get_dsv4_compress_state_dtype_sizes()
        )
        c4_state_bytes = 2 * 2 * attn_head_dim * c4_state_dtype_size
        # Online c128 stores (max, sum, kv) per slot (3*head_dim) instead of
        # raw (kv, score) (2*head_dim). Combined with ring_size=1 this still
        # nets a large reduction (~3/256x) but the per-slot bytes go up.
        c128_online = envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get()
        c128_state_bytes = (
            (3 if c128_online else 2 * 1) * attn_head_dim * c128_state_dtype_size
        )
        c4_indexer_state_bytes = 2 * 2 * self.indexer_head_dim * c4_state_dtype_size

        c4_state_ratio = self.c4_ring_size / self.swa_page_size
        # C128 state is request-scoped and is finalized after
        # max_running_requests is known, so it should not scale with
        # full-token capacity here.
        c128_state_ratio = 0

        c4_frac = 1 / (4 * self.c4_shrink_factor)
        return (
            self.swa_ratio * kv_bytes * self.num_layers_total
            + c4_frac * kv_bytes * self.num_layers_ca4
            + 1 / 128 * kv_bytes * self.num_layers_ca128
            + 1 / 4 * indexer_bytes * self.num_layers_ca4
            + self.swa_ratio * c4_state_ratio * c4_state_bytes * self.num_layers_ca4
            + c128_state_ratio * c128_state_bytes * self.num_layers_ca128
            + self.swa_ratio
            * c4_state_ratio
            * c4_indexer_state_bytes
            * self.num_layers_ca4
        )

    def _compute_dsv4_sizes(self, full_token: int, page_size: int) -> _DSV4PoolSizes:
        full_token = full_token // page_size * page_size
        swa_tokens = int(full_token * self.swa_ratio) // page_size * page_size
        return _DSV4PoolSizes(
            full_max_total_num_tokens=full_token,
            swa_max_total_num_tokens=swa_tokens,
            c4_max_total_num_tokens=full_token // (4 * self.c4_shrink_factor),
            c128_max_total_num_tokens=full_token // 128,
            c4_state_pool_size=swa_tokens // self.swa_page_size * self.c4_ring_size,
            c128_state_pool_size=0,
        )

    def _get_num_req_slots(self, max_running_requests: int) -> int:
        if self.disaggregation_mode == "decode":
            return max_running_requests + self.disaggregation_decode_extra_slots + 1
        return max_running_requests + 1

    def _get_c128_state_fixed_bytes(self, max_running_requests: int) -> int:
        if self.num_layers_ca128 == 0:
            return 0

        _, c128_state_dtype_size = _get_dsv4_compress_state_dtype_sizes()
        attn_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        num_req_slots = self._get_num_req_slots(max_running_requests)

        if envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get():
            state_rows = num_req_slots + self.c128_ring_size + 1
            state_rows *= 1 + self.online_c128_mtp_max_draft_tokens
            state_last_dim = 3 * attn_head_dim
        else:
            state_pool_size = num_req_slots * self.c128_ring_size
            state_rows = state_pool_size + self.c128_ring_size + 1
            state_rows = ceil_div(state_rows, 128) * 128
            state_last_dim = 2 * attn_head_dim

        return (
            state_rows * state_last_dim * c128_state_dtype_size * self.num_layers_ca128
        )

    def _get_c128_state_fixed_bytes_for_token_capacity(
        self, token_capacity: int
    ) -> int:
        if self.requested_max_running_requests_per_worker is not None:
            return self._get_c128_state_fixed_bytes(
                self.requested_max_running_requests_per_worker
            )

        estimated = int(token_capacity / self.context_len * 512)
        estimated = max(min(estimated, 4096), 2048)
        max_running_requests = min(estimated, token_capacity // 2)
        return self._get_c128_state_fixed_bytes(max_running_requests)

    def _to_config(self, sizes: _DSV4PoolSizes) -> MemoryPoolConfig:
        full = sizes.full_max_total_num_tokens
        swa = sizes.swa_max_total_num_tokens
        logger.info(
            f"DSV4 pool sizes: full={full}, swa={swa}, "
            f"c4={sizes.c4_max_total_num_tokens}, "
            f"c128={sizes.c128_max_total_num_tokens}, "
            f"c4_state={sizes.c4_state_pool_size}, "
            f"c128_state={sizes.c128_state_pool_size}"
        )
        return MemoryPoolConfig(
            max_total_num_tokens=full,
            full_max_total_num_tokens=full,
            swa_max_total_num_tokens=swa,
            c4_max_total_num_tokens=sizes.c4_max_total_num_tokens,
            c128_max_total_num_tokens=sizes.c128_max_total_num_tokens,
            c4_state_pool_size=sizes.c4_state_pool_size,
            c128_state_pool_size=sizes.c128_state_pool_size,
        )

    def finalize_with_max_running_requests(
        self, config: MemoryPoolConfig
    ) -> MemoryPoolConfig:
        assert config.max_running_requests is not None
        num_req_slots = self._get_num_req_slots(config.max_running_requests)
        if envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get():
            config.c128_state_pool_size = num_req_slots
        else:
            config.c128_state_pool_size = num_req_slots * self.c128_ring_size
        return config

    def calculate_pool_sizes(
        self, available_bytes: int, page_size: int
    ) -> MemoryPoolConfig:
        assert (
            page_size % 128 == 0
        ), "page_size must be multiple of 128 for compressed attention"

        if self.requested_max_running_requests_per_worker is not None:
            c128_state_fixed_bytes = self._get_c128_state_fixed_bytes(
                self.requested_max_running_requests_per_worker
            )
        else:
            full_token = int(available_bytes / self.bytes_per_full_token)
            c128_state_fixed_bytes = (
                self._get_c128_state_fixed_bytes_for_token_capacity(full_token)
            )

        available_bytes_for_tokens = max(available_bytes - c128_state_fixed_bytes, 0)
        full_token = int(available_bytes_for_tokens / self.bytes_per_full_token)

        sizes = self._compute_dsv4_sizes(full_token, page_size)
        logger.info(
            f"DSV4 memory calculation: "
            f"bytes_per_full_token={self.bytes_per_full_token:.2f}, "
            f"available_bytes={available_bytes / (1 << 30):.2f} GB, "
            f"c128_state_fixed={c128_state_fixed_bytes / (1 << 30):.2f} GB, "
            f"full_token={sizes.full_max_total_num_tokens}"
        )
        return self._to_config(sizes)

    def calculate_pool_sizes_from_max_tokens(
        self, max_total_num_tokens: int, page_size: int
    ) -> MemoryPoolConfig:
        assert (
            page_size % 128 == 0
        ), "page_size must be multiple of 128 for compressed attention"
        sizes = self._compute_dsv4_sizes(max_total_num_tokens, page_size)
        return self._to_config(sizes)


def create_memory_pool_configurator(
    mr: ModelRunner,
) -> MemoryPoolConfigurator:
    """Factory: select the right configurator for the model architecture."""
    if is_deepseek_v4(mr.model_config.hf_config) and mr.is_hybrid_swa:
        configurator = DSV4PoolConfigurator(mr)
    elif mr.is_hybrid_swa:
        if SWAChunkCapPoolConfigurator.is_applicable(mr):
            configurator = SWAChunkCapPoolConfigurator(mr)
        else:
            configurator = HybridSWAPoolConfigurator(mr)
    else:
        # Future: MambaPoolConfigurator
        configurator = DefaultPoolConfigurator(mr)
    if mr.server_args.swa_pool_sizing == "cap" and not isinstance(
        configurator, SWAChunkCapPoolConfigurator
    ):
        logger.warning(
            "--swa-pool-sizing cap has no effect for this model/config "
            "(selected configurator: %s). It applies only to hybrid "
            "sliding-window models with global-attention layers (non-DSV4) "
            "when the hybrid SWA memory pool is enabled.",
            type(configurator).__name__,
        )
    return configurator
