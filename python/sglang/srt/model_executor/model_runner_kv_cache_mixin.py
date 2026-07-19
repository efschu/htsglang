from __future__ import annotations

import logging
import math
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
from sglang.srt.distributed.parallel_state import get_world_group
from sglang.srt.distributed.utils import (
    suggest_unit_rebalance_multi,
    tp_partition_size,
    tp_plan_active,
    uneven_dcp_active,
    weightless_kv_active,
)
from sglang.srt.environ import envs
from sglang.srt.mem_cache.allocator import (
    PagedTokenToKVPoolAllocator,
    TokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.allocator.hisparse import (
    DeepSeekV4HiSparseTokenToKVPoolAllocator,
    HiSparseTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.allocator.swa import (
    PureSWATokenToKVPoolAllocator,
    SWATokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.common import get_req_to_token_extra_context_len
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.mem_cache.hisparse_memory_pool import HiSparseDSATokenToKVPool
from sglang.srt.mem_cache.memory_pool import (
    DSATokenToKVPool,
    HybridLinearKVPool,
    HybridReqToTokenPool,
    MHATokenToKVPool,
    MHATokenToKVPoolFP4,
    MiniMaxSparseKVPool,
    MLATokenToKVPool,
    MLATokenToKVPoolFP4,
    NoOpMHATokenToKVPool,
    PageMajorMHATokenToKVPool,
    ReqToTokenPool,
)
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.platforms import current_platform
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils.common import (
    get_available_gpu_memory,
    get_device_memory_capacity,
    is_float4_e2m1fn_x2,
    is_hip,
    is_npu,
)

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.model_executor.pool_configurator import MemoryPoolConfig


def _should_enable_lazy_compaction() -> bool:
    """Lazy compaction default — ON unless
    `SGLANG_DISABLE_LAZY_COMPACTION=1` (escape hatch for A/B / rollback).
    Centralized here so both unified-memory-pool factory call sites stay in sync.
    """
    return not envs.SGLANG_DISABLE_LAZY_COMPACTION.get()


# the ratio of mamba cache pool size to max_running_requests
MAMBA_CACHE_SIZE_MAX_RUNNING_REQUESTS_RATIO = 3
MAMBA_CACHE_V2_ADDITIONAL_RATIO_OVERLAP = 2
MAMBA_CACHE_V2_ADDITIONAL_RATIO_OVERLAP_LAZY = 1
MAMBA_CACHE_V2_ADDITIONAL_RATIO_NO_OVERLAP = 1

# --- Demand-driven ("auto") mamba pool sizing (uneven-DCP path) ---------------
# The stock code reserves a fixed FRACTION of post-weights VRAM for the mamba
# state cache (--mamba-full-memory-ratio, default 0.9 -> ~47% of the remaining
# memory). Because the mamba pool is the concurrency limiter
# (max_num_reqs = max_mamba_cache_size // mamba_ratio), a fixed fraction
# over-provisions it several-fold for real workloads and steals that VRAM from
# the KV/token pool, capping max_total_num_tokens well below the achievable
# optimum. The auto path below sizes the pool to the actual serving concurrency
# and hands ALL remaining VRAM to KV -- mirroring vLLM's align-mode coupling of
# mamba blocks to running sequences -- so the KV ceiling reaches its optimum
# with NO manual --mamba-full-memory-ratio tuning.
MAMBA_FULL_MEMORY_RATIO_DEFAULT = 0.9
# Default steady-state concurrent-request target when --max-running-requests is
# unset. The pool holds `mamba_ratio` state slots per running request (base
# state + overlap ping-pong extra-buffer), so this couples the pool to a modest,
# realistic concurrency instead of a VRAM fraction. Comfortably above the
# 8-way concurrency correctness bar.
MAMBA_AUTO_TARGET_CONCURRENCY = 16
# Small headroom on the concurrency target (NOT a VRAM fraction). Keeps the pool
# from being sized to the exact target so a burst above it still admits.
MAMBA_AUTO_SAFETY_MARGIN = 1.25
# Prefill-activation scratch (MiB) folded back OUT of the KV budget on the auto
# path. Giving 100% of the freed VRAM to KV would let the token pool grow to the
# physical ceiling and starve the transient DCP-extend prefix-gather activation
# scratch, OOM'ing a large (~18k-token) prefill. Reserving this back lets the
# default --rank-auto-reserve-mib stand (no need to raise it to buy headroom).
MAMBA_AUTO_ACTIVATION_RESERVE_MIB = 1024

logger = logging.getLogger(__name__)


def _get_dsv4_compress_state_dtypes() -> tuple[torch.dtype, torch.dtype]:
    dtype_name = envs.SGLANG_DSV4_COMPRESS_STATE_DTYPE.get().strip().lower()
    if dtype_name in ("float32", "fp32"):
        return torch.float32, torch.float32
    if dtype_name in ("bfloat16", "bf16"):
        if envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get():
            raise ValueError(
                "SGLANG_DSV4_COMPRESS_STATE_DTYPE=bf16 is not supported when "
                "SGLANG_OPT_USE_ONLINE_COMPRESS=1; online c128 state must stay float32."
            )
        return torch.bfloat16, torch.bfloat16
    raise ValueError(
        "Unsupported SGLANG_DSV4_COMPRESS_STATE_DTYPE="
        f"{dtype_name!r}. Expected one of: float32, fp32, bfloat16, bf16."
    )


_is_npu = is_npu()
_is_hip = is_hip()


class ModelRunnerKVCacheMixin:
    def _profile_available_bytes(self: ModelRunner, pre_model_load_memory: int) -> int:
        # KV pool budget = currently-free GPU memory minus the non-static runtime
        # slack (pre_model_load_memory * (1 - mem_fraction_static)). Whatever is
        # already resident (model weights, etc.) is thus charged against it.
        #
        # --rank-gpu-memory-mib: mem_fraction_static was set per rank
        # (budget_mib / nvml_total of the rank's GPU, applied unmodified),
        # so the measurement must stay LOCAL — the classic min-reduce of
        # free bytes would collapse every rank onto the weakest GPU and
        # throw the uneven budgets away. Rank consistency is restored in
        # _apply_token_constraints by min-reducing the derived TOKEN
        # capacities (per-token bytes scale with each rank's kv-head
        # share, so proportional budgets yield near-equal token counts).
        uneven_memory = self.server_args.uneven_memory_budgets_active()
        if uneven_memory and get_world_group().world_size > 1:
            # Serialize the profiling of co-located ranks per physical
            # GPU: a barrier guarantees every rank finished loading its
            # weights before any rank reads the (shared) free memory, so
            # the measurement is deterministic.
            torch.distributed.barrier(group=get_world_group().cpu_group)
        available_gpu_memory = get_available_gpu_memory(
            self.device,
            self.gpu_id,
            distributed=get_world_group().world_size > 1 and not uneven_memory,
            cpu_group=get_world_group().cpu_group,
        )
        # Co-located ranks: mem_get_info() above charged this rank for its
        # sibling(s)' weights too. Add their own footprint back so only
        # this rank's weights are charged against its budget (no-op on the
        # default / even-TP / one-rank-per-GPU paths).
        available_gpu_memory += self._colocated_sibling_reserved_gb()

        if uneven_memory:
            # Absolute-budget accounting (PD disagg #99): --rank-gpu-memory-mib
            # is an ABSOLUTE per-rank allowance, so the KV budget is simply
            #   budget - (memory this rank consumed since the pre-load reading).
            # The legacy formula below charges pre_model_load * (1 - fraction)
            # as slack, which silently assumes pre_model_load ~= device total
            # (a fresh GPU). With a co-resident FOREIGN process (single-node
            # PD disaggregation: the prefill server shares the big GPU with a
            # decode rank), pre-load free memory is already reduced by that
            # process, and the slack double-charges it -- rest went several GB
            # negative on a rank whose budget was fully available. The
            # before/after DELTA is immune: a static co-resident process
            # appears in both readings and cancels out.
            mib = self.server_args.rank_gpu_memory_mib
            budget_gb = (
                mib if isinstance(mib, (int, float)) else mib[self.tp_rank]
            ) / 1024.0
            used_by_me_gb = pre_model_load_memory - available_gpu_memory
            rest_memory = budget_gb - used_by_me_gb
            if self.mambaish_config is not None and self.post_capture_kv_active:
                rest_memory -= (
                    self.server_args.mamba_pre_capture_reserve_mb(
                        get_device_memory_capacity(self.device)
                    )
                    / 1024
                )
        else:
            slack_gb = pre_model_load_memory * (1 - self.mem_fraction_static)
            if self.mambaish_config is not None and self.post_capture_kv_active:
                # Mamba state is a fixed pre-capture allocation, so it can't ride the ~0 post-capture slack.
                slack_gb = max(
                    slack_gb,
                    self.server_args.mamba_pre_capture_reserve_mb(
                        get_device_memory_capacity(self.device)
                    )
                    / 1024,
                )
            rest_memory = available_gpu_memory - slack_gb
        if self.mambaish_config is not None:
            rest_memory = self.handle_max_mamba_cache(rest_memory)

        # Loaded weights (target + draft) can exceed the static budget
        if rest_memory <= 0:
            minimum_mem_fraction_static = (
                1 - available_gpu_memory / pre_model_load_memory
            )
            suggested_mem_fraction_static = (
                math.ceil(minimum_mem_fraction_static * 1000) / 1000
            )
            if uneven_memory:
                # In this mode --mem-fraction-static is rejected up front;
                # phrase the fix in the budget's own unit (MiB).
                used_gb = pre_model_load_memory - available_gpu_memory
                raise ValueError(
                    f"Loaded weights leave no GPU memory for the KV cache "
                    f"under --rank-gpu-memory-mib on rank {self.tp_rank}: "
                    f"the rank already uses {used_gb:.2f} GiB "
                    f"(~{math.ceil(used_gb * 1024)} MiB) for weights and "
                    f"runtime state, which exhausts the per-rank budget. "
                    f"Raise --rank-gpu-memory-mib above that, or place "
                    f"fewer ranks on this GPU. If using speculative "
                    f"decoding, draft weights are now counted."
                )
            raise ValueError(
                f"Loaded weights leave no GPU memory for the KV cache under "
                f"--mem-fraction-static={self.mem_fraction_static}. "
                f"Raise --mem-fraction-static above "
                f"{suggested_mem_fraction_static:.3f} "
                f"(minimum viable = 1 - available/pre = "
                f"{minimum_mem_fraction_static:.4f}). If using speculative "
                f"decoding, draft weights are now counted."
            )

        return int(rest_memory * (1 << 30))  # return in bytes

    def _colocated_sibling_reserved_gb(self: ModelRunner) -> float:
        """GiB of GPU memory reserved by TP siblings that SHARE this rank's
        physical GPU (duplicate --rank-gpu-id entries), to be ADDED BACK to
        the profiled free memory.

        get_available_gpu_memory() reads torch.cuda.mem_get_info(), i.e.
        DRIVER-level free memory of the physical device. When two ranks are
        co-located on one GPU, that reading has BOTH ranks' weights
        subtracted, but the slack/headroom term only accounts for THIS
        rank's single budget — so rest_memory collapses to
        ``budget - own_weights - sibling_weights`` and goes negative once
        the summed per-rank budgets approach the card's capacity, crashing
        the whole TP group via the downstream min-reduce.

        Adding the siblings' own resident footprint back makes each rank
        see only its OWN weights charged. torch.cuda.memory_reserved() is
        per-PROCESS (unlike mem_get_info), so it is the rank-isolated
        primitive; the barrier in the callers guarantees every sibling's
        weights are resident before we gather.

        This restores single-rank semantics MINUS each sibling's CUDA
        context (~0.3-0.6 GB, which memory_reserved() does not count). That
        residual is deliberate and conservative: the resulting budget is a
        touch SMALLER than a true single-rank profile, never larger, so
        co-located ranks can never over-allocate. Do NOT "fix" the apparent
        context gap by adding it back — the front-end
        ``Σ rank_gpu_memory_mib ≤ card total`` check plus this pessimism is
        what keeps the summed co-located pools under the card's capacity.

        Returns 0.0 on the default path, in even TP, and for ranks that do
        not share their GPU — leaving those budgets byte-for-byte unchanged.

        Collective: the guard below is world-wide identical, so every rank
        either participates in the all_gather or none do — never a subset
        (which would hang).
        """
        rgid = self.server_args.rank_gpu_id
        if (
            not self.server_args.uneven_memory_budgets_active()
            or rgid is None
            or len(set(rgid)) == len(rgid)  # no co-location anywhere
            or get_world_group().world_size <= 1
        ):
            return 0.0
        my_gpu = rgid[self.tp_rank]
        own_reserved = torch.cuda.memory_reserved(self.gpu_id)
        gathered = [None] * get_world_group().world_size
        torch.distributed.all_gather_object(
            gathered,
            (self.tp_rank, my_gpu, own_reserved),
            group=get_world_group().cpu_group,
        )
        sibling_bytes = sum(
            reserved
            for (rk, gpu, reserved) in gathered
            if gpu == my_gpu and rk != self.tp_rank
        )
        return sibling_bytes / (1 << 30)

    def _sync_uneven_mamba_cache_size(self: ModelRunner) -> None:
        """Agree on one max_mamba_cache_size across uneven-TP ranks.

        Under --rank-gpu-memory-mib the memory-derived Mamba pool size
        differs per rank: the byte budget is rank-local and (with a shard
        plan) the per-request state size scales with the rank's k-head
        share. The pool size is a request COUNT that the scheduler and the
        (Mamba)RadixCache assume to be identical on every rank, so — like
        the KV token capacity — the ranks agree on the minimum. With
        budgets proportional to the head ratio the min is nearly lossless.
        No-op on the default path (byte-level MIN already unified it)."""
        if not self.server_args.uneven_memory_budgets_active():
            return
        if get_world_group().world_size <= 1:
            return
        local_size = self.server_args.max_mamba_cache_size
        tensor = torch.tensor(local_size, dtype=torch.int64)
        torch.distributed.all_reduce(
            tensor,
            op=torch.distributed.ReduceOp.MIN,
            group=get_world_group().cpu_group,
        )
        agreed = int(tensor.item())
        if agreed != local_size:
            self.server_args.override(
                "mamba_pool.uneven_tp_min", max_mamba_cache_size=agreed
            )

    def _auto_mamba_demand_active(self: ModelRunner) -> bool:
        """Whether to size the mamba state pool by DEMAND (concurrency) rather
        than by the fixed --mamba-full-memory-ratio fraction.

        Gated narrowly so stock behavior is byte-identical unless it is clearly
        safe to diverge:
          * uneven-DCP must be in force (a non-uniform token vector installed,
            i.e. SGLANG_UNEVEN_DCP + _WEIGHTED). The default path, the
            even-modulo DCP path, and single-GPU all keep the fixed fraction.
          * the user must NOT have pinned --max-mamba-cache-size (handled by the
            explicit branch above) or --mamba-full-memory-ratio (an explicit
            fraction is honored as an override -> fixed-fraction path).
          * radix cache must be enabled (the disable-radix branch owns its own
            request-count sizing).
        """
        sa = self.server_args
        return (
            uneven_dcp_active(sa.dcp_size)
            and sa.max_mamba_cache_size is None
            and not sa.disable_radix_cache
            and abs(sa.mamba_full_memory_ratio - MAMBA_FULL_MEMORY_RATIO_DEFAULT)
            < 1e-9
        )

    def _auto_mamba_target_concurrency(self: ModelRunner) -> int:
        """Effective per-worker concurrency target used to size the demand-driven
        mamba pool.

        Only a USER-supplied --max-running-requests is treated as an explicit
        concurrency target. When it was auto-defaulted by another handler
        (e.g. the speculative-decoding hook resets an unset value to 48), that
        value does NOT reflect real demand -- sizing the mamba pool to it
        over-provisions several GB (48 * ratio * safety slots) and OOMs at pool
        init once the draft weights are also counted. In that case fall back to
        the modest MAMBA_AUTO_TARGET_CONCURRENCY default, and never let an
        auto-defaulted value push the target above it."""
        sa = self.server_args
        per_worker = sa.dp_size if sa.enable_dp_attention else 1
        user_set = getattr(sa, "max_running_requests_user_set", False)
        if sa.max_running_requests is not None and user_set:
            return max(sa.max_running_requests // per_worker, 1)
        if sa.max_running_requests is not None:
            return min(
                max(sa.max_running_requests // per_worker, 1),
                MAMBA_AUTO_TARGET_CONCURRENCY,
            )
        return MAMBA_AUTO_TARGET_CONCURRENCY

    def _auto_mamba_demand_size(self: ModelRunner, ratio: int) -> int:
        """Demand-driven mamba pool size (a request-STATE-slot count).

        slots = ceil(target_concurrency * mamba_ratio * safety), where
        target_concurrency is _auto_mamba_target_concurrency(). mamba_ratio is
        the per-request slot multiplier (base state + overlap ping-pong
        extra-buffer), so the pool admits ~target_concurrency*safety requests
        (max_num_reqs = slots // ratio). Floored at `ratio` so at least one
        request is always admissible -- the scheduler is never starved
        ("state cache too small" / max_num_reqs=0 can't return)."""
        target = self._auto_mamba_target_concurrency()
        slots = math.ceil(target * ratio * MAMBA_AUTO_SAFETY_MARGIN)
        return int(max(slots, ratio))

    def handle_max_mamba_cache(self: ModelRunner, total_rest_memory):
        config = self.mambaish_config
        server_args = self.server_args
        assert config is not None

        has_spec_dec = not self.spec_algorithm.is_none()
        if has_spec_dec:
            assert server_args.speculative_num_draft_tokens is not None
            assert server_args.max_running_requests is not None

        if server_args.max_mamba_cache_size is not None:
            # Use explicitly set max_mamba_cache_size
            server_args.override(
                "mamba_pool.per_dp_shard",
                max_mamba_cache_size=server_args.max_mamba_cache_size
                // (server_args.dp_size if server_args.enable_dp_attention else 1),
            )
            # Reserve intermediate memory based on capped max_num_reqs
            if has_spec_dec:
                ratio = self._calculate_mamba_ratio()
                capped_reqs = min(
                    server_args.max_running_requests
                    // (self.dp_size if server_args.enable_dp_attention else 1),
                    server_args.max_mamba_cache_size // ratio,
                )
                intermediate_size = (
                    config.mamba2_cache_params.mamba_cache_per_req
                    * capped_reqs
                    * server_args.speculative_num_draft_tokens
                )
                total_rest_memory = total_rest_memory - (intermediate_size / (1 << 30))
        elif self._auto_mamba_demand_active():
            # === Demand-driven mamba pool (uneven-DCP auto-sizing) ===========
            # Size the pool to the real serving concurrency, NOT to a fixed
            # fraction of post-weights VRAM. All remaining VRAM then flows to
            # the KV/token pool, so its ceiling reaches the optimum with no
            # manual --mamba-full-memory-ratio flag (the "self-determined +
            # optimal" requirement). See the module-level constants for the
            # rationale.
            per_req = config.mamba2_cache_params.mamba_cache_per_req
            assert per_req > 0
            ratio = self._calculate_mamba_ratio()
            D = server_args.speculative_num_draft_tokens if has_spec_dec else 0
            demand_size = self._auto_mamba_demand_size(ratio)
            # Never exceed what the post-weights budget can physically hold
            # (main state + spec-decode intermediate state per admitted req).
            fit_cap = int(
                total_rest_memory * (1 << 30) // (per_req * (1 + D / ratio))
            )
            size = min(demand_size, max(fit_cap, 0))
            reserve_gb = MAMBA_AUTO_ACTIVATION_RESERVE_MIB / 1024.0
            effective_target = self._auto_mamba_target_concurrency()
            # Show the EFFECTIVE (possibly capped) target actually used for
            # sizing, plus the raw --max-running-requests so a capped
            # auto-default (e.g. spec-hook 48 -> capped 16) is visible.
            target_desc = (
                str(effective_target)
                if getattr(server_args, "max_running_requests_user_set", False)
                and server_args.max_running_requests is not None
                else f"{effective_target}(auto; max_running_requests="
                f"{server_args.max_running_requests})"
            )
            logger.info(
                "[auto-mamba] demand-driven mamba pool: target_concurrency=%s "
                "ratio=%d safety=%.2f -> max_mamba_cache_size=%d slots "
                "(%.2f GB @ per_req=%.2f MiB; fit_cap=%d) -> admits ~%d reqs; "
                "activation_reserve=%.2f GB; remaining VRAM -> KV pool.",
                target_desc,
                ratio,
                MAMBA_AUTO_SAFETY_MARGIN,
                size,
                size * per_req / (1 << 30),
                per_req / (1 << 20),
                fit_cap,
                size // ratio,
                reserve_gb,
            )
            server_args.override(
                "mamba_pool.demand_driven",
                max_mamba_cache_size=size,
            )
            # Uneven TP: agree on the min request count across ranks BEFORE it
            # feeds the intermediate-memory reservation below.
            self._sync_uneven_mamba_cache_size()
            if has_spec_dec:
                capped_reqs = min(
                    server_args.max_running_requests
                    // (self.dp_size if server_args.enable_dp_attention else 1),
                    server_args.max_mamba_cache_size // ratio,
                )
                intermediate_size = per_req * capped_reqs * D
                total_rest_memory = total_rest_memory - (intermediate_size / (1 << 30))
            # Fold prefill-activation headroom back OUT of the KV budget so the
            # token pool does not grow to the physical ceiling and starve the
            # transient DCP-extend prefix-gather scratch (which OOMs a large
            # prefill). This lets the default --rank-auto-reserve-mib stand.
            total_rest_memory = total_rest_memory - reserve_gb
        elif (
            server_args.disable_radix_cache
            and server_args.max_running_requests is not None
        ):
            # Radix cache disabled: max_running_requests is the natural pool
            # size (one slot per running request), but it must be jointly
            # fitted against the memory budget like the radix-enabled branch
            # below — main state + spec intermediate states together. The old
            # unconditional `size = max_running_requests` reserved
            # per_req * size * (1 + D) bytes regardless of budget, which on
            # small per-rank budgets ate the whole KV headroom and surfaced
            # as a misleading "no memory for KV cache" error downstream.
            per_req = config.mamba2_cache_params.mamba_cache_per_req
            assert per_req > 0
            requested_size = server_args.max_running_requests // (
                server_args.dp_size if server_args.enable_dp_attention else 1
            )
            mamba_budget_bytes = (
                total_rest_memory
                * server_args.mamba_full_memory_ratio
                / (1 + server_args.mamba_full_memory_ratio)
                * (1 << 30)
            )
            # ratio is 1 with the radix cache disabled; D/ratio mirrors the
            # joint solve of the radix-enabled branch.
            ratio = self._calculate_mamba_ratio()
            D = server_args.speculative_num_draft_tokens if has_spec_dec else 0
            budget_size = int(mamba_budget_bytes // (per_req * (1 + D / ratio)))
            size = min(requested_size, budget_size)
            if size < requested_size:
                logger.warning(
                    "Mamba pool with --disable-radix-cache: the memory budget "
                    "fits %d request states (per_req=%.2f MiB, spec draft "
                    "tokens=%d), capping the effective concurrency below "
                    "--max-running-requests=%d.",
                    size,
                    per_req / (1 << 20),
                    D,
                    requested_size,
                )
            server_args.override(
                "mamba_pool.from_max_running_requests",
                max_mamba_cache_size=size,
            )
            # Uneven TP: agree on the min across ranks before the
            # intermediate reservation consumes the size (see below).
            self._sync_uneven_mamba_cache_size()
            # Reserve intermediate memory based on the fitted size
            if has_spec_dec:
                intermediate_size = (
                    per_req
                    * server_args.max_mamba_cache_size
                    * server_args.speculative_num_draft_tokens
                )
                total_rest_memory = total_rest_memory - (intermediate_size / (1 << 30))
        else:
            # Use ratio-based calculation to auto-fit available memory
            assert config.mamba2_cache_params.mamba_cache_per_req > 0
            per_req = config.mamba2_cache_params.mamba_cache_per_req

            # Solve jointly for max_mamba_cache_size accounting for intermediate memory.
            # The mamba budget (from the ratio split) must cover both:
            #   1. main mamba state: max_mamba_cache_size * per_req
            #   2. intermediate states: (max_mamba_cache_size / ratio) * D * per_req
            # So: max_mamba_cache_size * per_req * (1 + D/ratio) = mamba_budget_bytes
            mamba_budget = (
                total_rest_memory
                * server_args.mamba_full_memory_ratio
                / (1 + server_args.mamba_full_memory_ratio)
            )
            mamba_budget_bytes = mamba_budget * (1 << 30)

            if has_spec_dec:
                ratio = self._calculate_mamba_ratio()
                D = server_args.speculative_num_draft_tokens
                # Joint solve: main_state + intermediate = mamba_budget
                server_args.override(
                    "mamba_pool.memory_budget_spec",
                    max_mamba_cache_size=int(
                        mamba_budget_bytes // (per_req * (1 + D / ratio))
                    ),
                )
                # Uneven TP: the size was derived from rank-local bytes /
                # per-req state; agree on the min BEFORE it feeds
                # capped_reqs and the intermediate-memory reservation.
                self._sync_uneven_mamba_cache_size()
                # Intermediate memory is included in mamba_budget, subtract it
                # so the return value only has main_state subtracted from total
                capped_reqs = min(
                    server_args.max_running_requests
                    // (self.dp_size if server_args.enable_dp_attention else 1),
                    server_args.max_mamba_cache_size // ratio,
                )
                intermediate_size = per_req * capped_reqs * D
                total_rest_memory = total_rest_memory - (intermediate_size / (1 << 30))
            else:
                server_args.override(
                    "mamba_pool.memory_budget",
                    max_mamba_cache_size=int(mamba_budget_bytes // per_req),
                )
                # Uneven TP: agree on the min across ranks (see above).
                self._sync_uneven_mamba_cache_size()

        # Uneven TP (--rank-gpu-memory-mib): the ratio-based auto-sizing
        # above ran on rank-LOCAL memory, so the derived request COUNT can
        # differ slightly per rank (the per-request state bytes scale with
        # each rank's head share, so proportional budgets give near-equal
        # counts). The schedulers run in lockstep and must agree on one
        # count — min-reduce it before anything consumes it.
        if (
            self.server_args.uneven_memory_budgets_active()
            and get_world_group().world_size > 1
        ):
            tensor = torch.tensor(server_args.max_mamba_cache_size, dtype=torch.int64)
            torch.distributed.all_reduce(
                tensor,
                op=torch.distributed.ReduceOp.MIN,
                group=get_world_group().cpu_group,
            )
            synced_size = int(tensor.item())
            if synced_size != server_args.max_mamba_cache_size:
                server_args.override(
                    "mamba_pool.uneven_tp_min_sync",
                    max_mamba_cache_size=synced_size,
                )

        # Validate: max_mamba_cache_size must be positive after memory allocation.
        # A non-positive value means GPU memory is insufficient for the requested
        # configuration. Fail fast with actionable advice instead of silently
        # producing garbled output at runtime.
        if server_args.max_mamba_cache_size <= 0:
            raise RuntimeError(
                f"Not enough GPU memory for hybrid (mamba/linear-attention) state cache. "
                f"Computed max_mamba_cache_size={server_args.max_mamba_cache_size} "
                f"(total_rest_memory={total_rest_memory:.2f} GB, "
                f"mamba_cache_per_req={config.mamba2_cache_params.mamba_cache_per_req / (1 << 20):.2f} MB). "
                f"Try: (1) reduce --max-running-requests, "
                f"(2) increase --mem-fraction-static, "
                f"(3) reduce --speculative-num-draft-tokens, or "
                f"(4) use GPUs with more memory."
            )

        mamba_state_memory = (
            server_args.max_mamba_cache_size
            * config.mamba2_cache_params.mamba_cache_per_req
            / (1 << 30)
        )
        return total_rest_memory - mamba_state_memory

    def calculate_mla_kv_cache_dim(self: ModelRunner) -> int:
        is_dsa_model = is_deepseek_dsa(self.model_config.hf_config)
        kv_cache_dtype = self.kv_cache_dtype
        kv_lora_rank = self.model_config.kv_lora_rank
        qk_rope_head_dim = self.model_config.qk_rope_head_dim
        kv_cache_dim = kv_lora_rank + qk_rope_head_dim  # default mla kv cache dim

        # For non-DSA models, MLA kv cache dim is simply kv_lora_rank + qk_rope_head_dim
        if not is_dsa_model:
            return kv_cache_dim

        # TRTLLM backend does not override kv_cache_dim for MLA kv cache
        # Assuming dsa prefill and decode backends are the same when using trtllm MLA backend,
        # since it is not compatible for trtllm and other mla attn backend due to the different
        # kv cache layout.
        if (
            self.server_args.dsa_prefill_backend == "trtllm"
            or self.server_args.dsa_decode_backend == "trtllm"
        ):
            return kv_cache_dim

        # On HIP, TileLang and AITER DSA kernels consume the raw MLA KV layout:
        # nope(512 fp8) + rope(64 fp8), without extra per-block scales.
        if _is_hip and (
            self.server_args.dsa_prefill_backend in ("tilelang", "aiter")
            or self.server_args.dsa_decode_backend in ("tilelang", "aiter")
        ):
            return kv_cache_dim

        quant_block_size = DSATokenToKVPool.quant_block_size
        rope_storage_dtype = DSATokenToKVPool.rope_storage_dtype
        # Calculate override_kv_cache_dim for FP8 storage in backends that use scaled KV layout
        # (excluding TRTLLM and HIP raw-layout kernels).
        # kv_lora_rank + scale storage (kv_lora_rank // quant_block_size * 4 bytes) + rope dimension storage
        # Note: rope dimension is stored in original dtype (bf16), not quantized to fp8
        if kv_cache_dtype == torch.float8_e4m3fn:
            assert (
                kv_lora_rank % quant_block_size == 0
            ), f"kv_lora_rank {kv_lora_rank} must be multiple of quant_block_size {quant_block_size}"

            return (
                kv_lora_rank
                + kv_lora_rank // quant_block_size * 4
                + qk_rope_head_dim * rope_storage_dtype.itemsize
            )

        return kv_cache_dim

    def _calculate_mamba_ratio(self: ModelRunner) -> int:
        if self.server_args.disable_radix_cache:
            return 1

        additional_ratio = 0
        if self.server_args.enable_mamba_extra_buffer():
            # ping-pong buffer size is 2 when overlap schedule is on, 1 otherwise.
            # Lazy mode saves 1 slot (2 → 1) for overlap; non-overlap already uses 1.
            if not self.server_args.disable_overlap_schedule:
                if self.server_args.enable_mamba_extra_buffer_lazy():
                    additional_ratio = MAMBA_CACHE_V2_ADDITIONAL_RATIO_OVERLAP_LAZY
                else:
                    additional_ratio = MAMBA_CACHE_V2_ADDITIONAL_RATIO_OVERLAP
            else:
                assert (
                    not self.server_args.enable_mamba_extra_buffer_lazy()
                ), "Lazy extra buffer requires overlap schedule (--disable-overlap-schedule is incompatible)"
                additional_ratio = MAMBA_CACHE_V2_ADDITIONAL_RATIO_NO_OVERLAP

        return MAMBA_CACHE_SIZE_MAX_RUNNING_REQUESTS_RATIO + additional_ratio

    def _validate_prefill_only_disable_kv_cache_pool_family(
        self: ModelRunner,
        is_dsa_model: bool,
        is_dsv4_model: bool,
        current_platform,
    ):
        if not self.server_args.prefill_only_disable_kv_cache or self.is_draft_worker:
            return

        unsupported_pool_family = None
        if is_dsv4_model:
            unsupported_pool_family = "DeepSeekV4TokenToKVPool"
        elif current_platform.is_out_of_tree() and not self.mambaish_config:
            unsupported_pool_family = "out-of-tree platform KV pool"
        elif (
            self.server_args.attention_backend == "ascend" and not self.mambaish_config
        ):
            unsupported_pool_family = "NPU/Ascend KV pool"
        elif self.use_mla_backend and is_dsa_model:
            unsupported_pool_family = "DSA/MLA KV pool"
        elif self.use_mla_backend and not self.mambaish_config:
            unsupported_pool_family = "MLA KV pool"
        elif self.is_hybrid_swa:
            unsupported_pool_family = "SWA KV pool"
        elif self.mambaish_config:
            unsupported_pool_family = "hybrid linear/Mamba KV pool"
        elif is_float4_e2m1fn_x2(self.kv_cache_dtype):
            unsupported_pool_family = "FP4 MHA KV pool"

        if unsupported_pool_family is not None:
            raise RuntimeError(
                "--prefill-only-disable-kv-cache is not supported for "
                f"{unsupported_pool_family}. Supported configurations today: plain MHA "
                "models on CUDA with the FA (fa3/fa4) prefill backend, --is-embedding, "
                "--chunked-prefill-size=-1, --disable-radix-cache, no context-parallel "
                "attention, no HiSparse, and --kv-cache-dtype != fp4_e2m1."
            )

    @property
    def post_capture_kv_active(self: ModelRunner) -> bool:
        return (
            self.server_args.post_capture_kv_sizing_planned()
            and current_platform.is_cuda()
            and not self.is_draft_worker
        )

    def post_capture_resize_kv_pool(self: ModelRunner) -> None:
        """Resize the KV pool after capture."""
        pool = self.token_to_kv_pool
        torch.cuda.synchronize()
        # --rank-gpu-memory-mib: keep the measurement rank-local (see
        # _profile_available_bytes); the resulting token count is clamped
        # by _apply_token_constraints via _config_from_budget below.
        uneven_memory = self.server_args.uneven_memory_budgets_active()
        if uneven_memory and get_world_group().world_size > 1:
            torch.distributed.barrier(group=get_world_group().cpu_group)
        free_gb = get_available_gpu_memory(
            self.device,
            self.gpu_id,
            distributed=get_world_group().world_size > 1 and not uneven_memory,
            cpu_group=get_world_group().cpu_group,
        )
        # Co-located ranks: undo the sibling's weight charge in the
        # physical-free reading (see _colocated_sibling_reserved_gb /
        # _profile_available_bytes). No-op on the default path.
        free_gb += self._colocated_sibling_reserved_gb()
        headroom_gb = self.pre_model_load_memory * (1 - self.mem_fraction_static)
        decode_cuda_graph_config = self.server_args.cuda_graph_config.decode
        decode_max_bs = int(decode_cuda_graph_config.max_bs or 0)
        running_requests = int(self.max_running_requests or decode_max_bs or 1)
        eager_decode_gap = (
            self.server_args.disaggregation_mode != "prefill"
            and decode_cuda_graph_config.backend != Backend.DISABLED
            and decode_max_bs < running_requests
        )
        if eager_decode_gap:
            logger.warning(
                "Post-capture KV sizing: decode CUDA graph max_bs=%d < "
                "max_running_requests=%d; reserving activation headroom",
                decode_max_bs,
                running_requests,
            )
        if eager_decode_gap or self.mambaish_config is not None:
            headroom_gb = max(
                headroom_gb,
                self.server_args.mamba_pre_capture_reserve_mb(
                    get_device_memory_capacity(self.device)
                )
                / 1024,
            )
        budget_bytes = (
            int(max(0.0, free_gb - headroom_gb) * (1 << 30))
            + pool.post_capture_backed_bytes
        )
        # Uneven-TP self-calibration: this post-capture budget is the
        # most accurate per-rank measurement (weights + graphs resident),
        # matching the restart the hint asks for.
        self._maybe_suggest_mlp_rebalance(budget_bytes)
        self._maybe_suggest_dcp_token_vector(budget_bytes)
        config = self._config_from_budget(
            budget_bytes, cap_tokens=self.max_total_num_tokens
        )
        pool.finalize_backing(config)
        self.token_to_kv_pool_allocator.resize(config)

        # Set the new pool size
        self.max_total_num_tokens = config.max_total_num_tokens
        if self.is_hybrid_swa:
            self.full_max_total_num_tokens = config.full_max_total_num_tokens
            self.swa_max_total_num_tokens = config.swa_max_total_num_tokens
        if self.memory_pool_config is not None:
            self.memory_pool_config.max_total_num_tokens = config.max_total_num_tokens
            self.memory_pool_config.full_max_total_num_tokens = (
                config.full_max_total_num_tokens
            )
            self.memory_pool_config.swa_max_total_num_tokens = (
                config.swa_max_total_num_tokens
            )
        if self.max_running_requests is not None:
            # Re-calculate max_running_requests for the now smaller pool
            capped_reqs = min(
                self.max_running_requests,
                self._resolve_max_num_reqs(config.max_total_num_tokens),
            )
            if capped_reqs < self.max_running_requests:
                logger.warning(
                    "Post-capture KV sizing: max_running_requests %d -> %d",
                    self.max_running_requests,
                    capped_reqs,
                )
                self.max_running_requests = capped_reqs
                if self.memory_pool_config is not None:
                    self.memory_pool_config.max_running_requests = capped_reqs
        logger.info(
            "Post-capture KV sizing: max_total_num_tokens=%d, free memory=%.2f GB",
            config.max_total_num_tokens,
            get_available_gpu_memory(self.device, self.gpu_id),
        )

    def _init_unified_mamba_pools(self: ModelRunner, max_num_reqs: int):
        """Build the shared-KV-pool stack for a hybrid-Mamba model:
        one byte buffer split between the full-attn MHA KV pool and the
        per-request Mamba state pool, with virtual slot ids above the
        allocator."""
        from sglang.srt.mem_cache.unified_memory_pool import init_unified_mamba_pools

        config = self.mambaish_config
        assert config is not None
        assert (
            not self.use_mla_backend
        ), "unified memory pool does not support MLA-hybrid-Mamba yet"
        # The full sub-pool is page-aware (via `MultiEndedAllocator(page_size=...)`);
        # the mamba sub-pool stays page=1.
        assert self.page_size >= 1, f"page_size must be >= 1, got {self.page_size}"
        # Mirror the non-shared path's extra_max_context_len computation.
        extra_max_context_len = 4
        if self.server_args.speculative_num_draft_tokens is not None:
            extra_max_context_len += self.server_args.speculative_num_draft_tokens

        mamba_layer_ids = [
            i
            for i in config.mamba2_cache_params.layers
            if self.start_layer <= i < self.end_layer
        ]
        full_attention_layer_ids = [
            i
            for i in config.full_attention_layer_ids
            if self.start_layer <= i < self.end_layer
        ]

        bundle = init_unified_mamba_pools(
            device=self.device,
            kv_cache_dtype=self.kv_cache_dtype,
            head_num=self.model_config.get_num_kv_heads(get_parallel().attn_tp_size),
            head_dim=self.model_config.head_dim,
            page_size=self.page_size,
            start_layer=self.start_layer,
            end_layer=self.end_layer,
            is_draft_worker=self.is_draft_worker,
            use_mla_backend=self.use_mla_backend,
            mamba_layer_ids=mamba_layer_ids,
            full_attention_layer_ids=full_attention_layer_ids,
            mamba2_cache_params=config.mamba2_cache_params,
            model_context_len=self.model_config.context_len,
            extra_max_context_len=extra_max_context_len,
            max_total_num_tokens=self.max_total_num_tokens,
            max_mamba_cache_size=self.server_args.max_mamba_cache_size,
            max_num_reqs=max_num_reqs,
            enable_memory_saver=self.server_args.enable_memory_saver,
            enable_mamba_extra_buffer=self.server_args.enable_mamba_extra_buffer(),
            speculative_num_draft_tokens=self.server_args.speculative_num_draft_tokens,
            disable_overlap_schedule=self.server_args.disable_overlap_schedule,
            need_sort=self.server_args.disaggregation_mode in ("decode", "prefill"),
            mamba_full_memory_ratio=self.server_args.mamba_full_memory_ratio,
            # Overlap mode: the allocator's `free` drops a wait_stream(forward_stream)
            # barrier so eager compaction serializes after the in-flight forward's
            # v2p/KV reads. Near-no-op in normal mode.
            forward_stream=self.forward_stream,
            # Lazy compaction: default ON, env-var escape hatch for rollback / A/B.
            lazy_compaction=_should_enable_lazy_compaction(),
        )
        self.req_to_token_pool = bundle.req_to_token_pool
        self.token_to_kv_pool = bundle.token_to_kv_pool
        self.token_to_kv_pool_allocator = bundle.token_to_kv_pool_allocator
        # Keep a reference so the shared byte buffer is not GC'd.
        self._unified_memory_pool = bundle.unified_memory_pool

    def _init_unified_swa_pools(self: ModelRunner, max_num_reqs: int):
        """Build the unified-pool stack for a hybrid-SWA model (Triton): one byte
        buffer split between the full-attention and SWA KV pools."""
        from sglang.srt.mem_cache.unified_memory_pool import init_unified_swa_pools

        assert self.is_hybrid_swa, "_init_unified_swa_pools called on a non-SWA model"
        # Both sub-pools are page-aware; the SWA composite runs alloc_extend_kernel
        # once in virtual space and binds the new pages on both sub-allocators.
        assert self.page_size >= 1, f"page_size must be >= 1, got {self.page_size}"
        assert (
            not self.use_mla_backend
        ), "unified memory pool does not support MLA-SWA hybrid yet"
        # Mirror the non-shared path's extra_max_context_len computation.
        extra_max_context_len = 4
        if self.server_args.speculative_num_draft_tokens is not None:
            extra_max_context_len += self.server_args.speculative_num_draft_tokens
        self.req_to_token_pool = ReqToTokenPool(
            size=max_num_reqs,
            max_context_len=self.model_config.context_len + extra_max_context_len,
            device=self.device,
            enable_memory_saver=self.server_args.enable_memory_saver,
        )

        head_num = self.model_config.get_num_kv_heads(get_parallel().attn_tp_size)
        head_dim = self.model_config.head_dim
        if self.is_hybrid_swa_compress:
            # Asymmetric head dims between full and SWA (NPU compress path):
            # pull SWA-specific dims from the hf text config.
            v_head_dim = self.model_config.hf_text_config.v_head_dim
            # Plan-aware per-rank SWA kv-head count (uneven TP via
            # --rank-tp-ratio); identical to the classic
            # max(1, swa_kv_heads // tp) without a plan.
            swa_head_num = self.model_config.get_swa_num_kv_heads(
                get_parallel().attn_tp_size
            )
            swa_head_dim = self.model_config.hf_text_config.swa_head_dim
            swa_v_head_dim = self.model_config.hf_text_config.swa_v_head_dim
        else:
            v_head_dim = head_dim
            swa_head_num = head_num
            swa_head_dim = head_dim
            swa_v_head_dim = head_dim

        # Filter layer ids to this worker's [start_layer, end_layer) range.
        swa_attention_layer_ids = [
            i
            for i in self.model_config.swa_attention_layer_ids
            if self.start_layer <= i < self.end_layer
        ]
        full_attention_layer_ids = [
            i
            for i in self.model_config.full_attention_layer_ids
            if self.start_layer <= i < self.end_layer
        ]

        bundle = init_unified_swa_pools(
            device=self.device,
            kv_cache_dtype=self.kv_cache_dtype,
            head_num=head_num,
            head_dim=head_dim,
            v_head_dim=v_head_dim,
            swa_head_num=swa_head_num,
            swa_head_dim=swa_head_dim,
            swa_v_head_dim=swa_v_head_dim,
            page_size=self.page_size,
            start_layer=self.start_layer,
            end_layer=self.end_layer,
            swa_attention_layer_ids=swa_attention_layer_ids,
            full_attention_layer_ids=full_attention_layer_ids,
            full_max_total_num_tokens=self.full_max_total_num_tokens,
            swa_max_total_num_tokens=self.swa_max_total_num_tokens,
            enable_memory_saver=self.server_args.enable_memory_saver,
            need_sort=self.server_args.disaggregation_mode in ("decode", "prefill"),
            # Overlap mode: same wait_stream(forward_stream) rationale as
            # `_init_unified_mamba_pools`.
            forward_stream=self.forward_stream,
            # Lazy compaction: default ON, with env var escape hatch for rollback / A/B.
            lazy_compaction=_should_enable_lazy_compaction(),
        )
        self.token_to_kv_pool = bundle.token_to_kv_pool
        self.token_to_kv_pool_allocator = bundle.token_to_kv_pool_allocator
        # Keep a reference so the shared byte buffer is not GC'd.
        self._unified_memory_pool = bundle.unified_memory_pool

    def _init_pools(self: ModelRunner):
        """Initialize the memory pools."""
        max_num_reqs = self.max_running_requests

        # Unified-pool fast path: build req_to_token + token_to_kv pool + allocator
        # from one byte buffer, then return. Gated to the target worker
        # (req_to_token_pool is None); supports hybrid Mamba and hybrid SWA (not DSV4).
        if (
            self.server_args.enable_unified_memory
            and self.server_args.disaggregation_mode == "null"
            and self.req_to_token_pool is None
        ):
            if self.mambaish_config is not None:
                self._init_unified_mamba_pools(max_num_reqs)
                return
            if self.is_hybrid_swa and not is_deepseek_v4(self.model_config.hf_config):
                self._init_unified_swa_pools(max_num_reqs)
                return
            # Fail loud, not silently fall through to the normal pools (which would
            # leave the flag a no-op). The feature replaces the HYBRID pools only.
            raise ValueError(
                "--enable-unified-memory only supports hybrid Mamba and "
                "hybrid sliding-window-attention models (DeepSeek-V4 excluded); "
                f"the current model ({self.model_config.hf_config.architectures}) "
                "is neither, so the unified memory pool cannot be built. Drop "
                "--enable-unified-memory for this model."
            )

        # Initialize req_to_token_pool
        if self.req_to_token_pool is None:
            max_spec_draft_tokens = self.server_args.max_speculative_num_draft_tokens
            extra_max_context_len = get_req_to_token_extra_context_len(self.server_args)

            if self.server_args.disaggregation_mode == "decode":
                from sglang.srt.disaggregation.decode import (
                    DecodeReqToTokenPool,
                    HybridMambaDecodeReqToTokenPool,
                )

                # Extra slots for pre-allocated requests
                pre_alloc_size = self.server_args.disaggregation_decode_extra_slots
                if config := self.mambaish_config:
                    self.req_to_token_pool = HybridMambaDecodeReqToTokenPool(
                        size=max_num_reqs,
                        max_context_len=self.model_config.context_len
                        + extra_max_context_len,
                        device=self.device,
                        enable_memory_saver=self.server_args.enable_memory_saver,
                        cache_params=config.mamba2_cache_params,
                        mamba_layer_ids=(
                            [
                                i
                                for i in config.mamba2_cache_params.layers
                                if self.start_layer <= i < self.end_layer
                            ]
                        ),
                        speculative_num_draft_tokens=max_spec_draft_tokens,
                        speculative_eagle_topk=self.server_args.speculative_eagle_topk,
                        enable_mamba_extra_buffer=self.server_args.enable_mamba_extra_buffer(),
                        pre_alloc_size=pre_alloc_size,
                        enable_overlap_schedule=not self.server_args.disable_overlap_schedule,
                        mamba_size=self.server_args.max_mamba_cache_size,
                        start_layer=self.start_layer,
                    )
                else:
                    self.req_to_token_pool = DecodeReqToTokenPool(
                        size=max_num_reqs,
                        max_context_len=self.model_config.context_len
                        + extra_max_context_len,
                        device=self.device,
                        enable_memory_saver=self.server_args.enable_memory_saver,
                        pre_alloc_size=pre_alloc_size,
                    )
            elif config := self.mambaish_config:
                self.req_to_token_pool = HybridReqToTokenPool(
                    size=max_num_reqs,
                    mamba_size=self.server_args.max_mamba_cache_size,
                    mamba_spec_state_size=max_num_reqs,
                    max_context_len=self.model_config.context_len
                    + extra_max_context_len,
                    device=self.device,
                    enable_memory_saver=self.server_args.enable_memory_saver,
                    cache_params=config.mamba2_cache_params,
                    mamba_layer_ids=(
                        [
                            i
                            for i in config.mamba2_cache_params.layers
                            if self.start_layer <= i < self.end_layer
                        ]
                    ),
                    enable_mamba_extra_buffer=self.server_args.enable_mamba_extra_buffer(),
                    enable_mamba_extra_buffer_lazy=self.server_args.enable_mamba_extra_buffer_lazy(),
                    speculative_num_draft_tokens=max_spec_draft_tokens,
                    speculative_eagle_topk=self.server_args.speculative_eagle_topk,
                    enable_overlap_schedule=not self.server_args.disable_overlap_schedule,
                    start_layer=self.start_layer,
                    enable_linear_replayssm=self.server_args.enable_linear_replayssm,
                    linear_replayssm_cache_len=self.server_args.linear_replayssm_cache_len,
                    mamba_envelope_layout=self.server_args.enable_page_major_kv_layout,
                )
            else:
                # DSV4 on NPU needs an extended ReqToTokenPool holding per-req
                # swa/c4/c128/c{4,128}_state tables; others stay on the stock one.
                req_to_token_pool_cls = ReqToTokenPool
                if _is_npu and is_deepseek_v4(self.model_config.hf_config):
                    from sglang.srt.hardware_backend.npu.dsv4.dsv4_req_to_token_pool import (
                        DSV4NPUReqToTokenPool,
                    )

                    req_to_token_pool_cls = DSV4NPUReqToTokenPool

                self.req_to_token_pool = req_to_token_pool_cls(
                    size=max_num_reqs,
                    max_context_len=self.model_config.context_len
                    + extra_max_context_len,
                    device=self.device,
                    enable_memory_saver=self.server_args.enable_memory_saver,
                )
        else:
            # Draft worker shares req_to_token_pool with the target worker.
            assert self.is_draft_worker

        # Initialize token_to_kv_pool
        is_dsa_model = is_deepseek_dsa(self.model_config.hf_config)
        is_dsv4_model = is_deepseek_v4(self.model_config.hf_config)

        self._validate_prefill_only_disable_kv_cache_pool_family(
            is_dsa_model, is_dsv4_model, current_platform
        )

        # Page-granularity envelope layout for the MHA-shaped (full / SWA) pools,
        # selected by swapping in the PageMajorMHATokenToKVPool subclass. The
        # default keeps upstream's per-layer layout. The Mamba state pool is routed
        # separately via `mamba_envelope_layout` on the req-to-token pool above.
        enable_page_major = self.server_args.enable_page_major_kv_layout
        mha_pool_class = (
            PageMajorMHATokenToKVPool if enable_page_major else MHATokenToKVPool
        )

        if is_dsv4_model:
            swa_page_size = self.page_size
            if not _is_npu:
                assert swa_page_size == 256, "In paged swa mode, page_size must be 256."

            if self.is_draft_worker:
                from sglang.srt.models.deepseek_v4_nextn import (
                    COMPRESS_RATIO_NEXTN_LAYER,
                )

                compression_ratios = [
                    COMPRESS_RATIO_NEXTN_LAYER
                ] * self.num_effective_layers
            else:
                compression_ratios = self.model_config.compress_ratios

            # NPU + DSV4 → paged-state subclass: the fused compressor kernel
            # needs cache_mode=1 (paged); Atlas A3 rejects cache_mode=2 (ring),
            # so the CUDA ring-buffer state path can't be shared. CUDA keeps
            # DeepSeekV4TokenToKVPool unchanged; NPU recomputes state sizes below.
            if _is_npu:
                from sglang.srt.hardware_backend.npu.dsv4.dsv4_memory_pool import (
                    DSV4NPUTokenToKVPool,
                    npu_state_pool_size,
                )

                pool_cls = DSV4NPUTokenToKVPool
                # Recompute state pool sizes for the NPU paged formula (CUDA's
                # ring sizes are dropped here). Tail-only allocation keeps the
                # per-req-budget formula sufficient at any prefill length: long
                # prompts allocate only ``tail+128`` (c4) / ``tail`` (c128)
                # slots (tail = seq_len % 128), and decode is drained by
                # sliding eviction in ``ScheduleBatch._evict_swa``.
                c4_state_pool_size = npu_state_pool_size(
                    ratio=4,
                    page_size=self.page_size,
                    max_num_reqs=self.max_running_requests,
                )
                c128_state_pool_size = npu_state_pool_size(
                    ratio=128,
                    page_size=self.page_size,
                    max_num_reqs=self.max_running_requests,
                )
            else:
                pool_cls = DeepSeekV4TokenToKVPool
                c4_state_pool_size = self.c4_state_pool_size
                c128_state_pool_size = self.c128_state_pool_size

            self.token_to_kv_pool = pool_cls(
                max_num_reqs=self.max_running_requests,
                # SWA ring is indexed by req_pool_idx; PD decode inflates req_to_token
                # past max_running_requests (pre-alloc), so size to the real capacity.
                num_req_slots=self.req_to_token_pool.req_to_token.shape[0],
                swa_size=self.swa_max_total_num_tokens,
                c4_size=self.c4_max_total_num_tokens,
                c128_size=self.c128_max_total_num_tokens,
                c4_state_pool_size=c4_state_pool_size,
                c128_state_pool_size=c128_state_pool_size,
                page_size=self.page_size,
                swa_page_size=swa_page_size,
                sliding_window=self.model_config.window_size,
                dtype=self.kv_cache_dtype,
                c4_state_dtype=self.c4_state_dtype,
                c128_state_dtype=self.c128_state_dtype,
                qk_nope_head_dim=self.model_config.qk_nope_head_dim,
                qk_rope_head_dim=self.model_config.qk_rope_head_dim,
                indexer_head_dim=self.model_config.index_head_dim,
                layer_num=self.num_effective_layers,
                device=self.device,
                enable_memory_saver=self.server_args.enable_memory_saver,
                compression_ratios=compression_ratios,
                start_layer=self.start_layer,
                end_layer=self.end_layer,
                enable_hisparse=self.enable_hisparse,
                online_mtp_max_draft_tokens=(
                    self.server_args.max_speculative_num_draft_tokens or 0
                ),
            )
        elif current_platform.is_out_of_tree() and not self.mambaish_config:
            if self.use_mla_backend and is_dsa_model:
                PoolCls = current_platform.get_dsa_kv_pool_cls()
                self.token_to_kv_pool = PoolCls(
                    self.max_total_num_tokens,
                    page_size=self.page_size,
                    dtype=self.kv_cache_dtype,
                    kv_lora_rank=self.model_config.kv_lora_rank,
                    qk_rope_head_dim=self.model_config.qk_rope_head_dim,
                    layer_num=self.num_effective_layers,
                    device=self.device,
                    kv_cache_dim=self.calculate_mla_kv_cache_dim(),
                    enable_memory_saver=self.server_args.enable_memory_saver,
                    start_layer=self.start_layer,
                    end_layer=self.end_layer,
                    index_head_dim=get_dsa_index_head_dim(self.model_config.hf_config),
                )
            elif self.use_mla_backend:
                PoolCls = current_platform.get_mla_kv_pool_cls()
                self.token_to_kv_pool = PoolCls(
                    self.max_total_num_tokens,
                    page_size=self.page_size,
                    dtype=self.kv_cache_dtype,
                    kv_lora_rank=self.model_config.kv_lora_rank,
                    qk_rope_head_dim=self.model_config.qk_rope_head_dim,
                    index_head_dim=(
                        self.model_config.index_head_dim if is_dsa_model else None
                    ),
                    layer_num=self.num_effective_layers,
                    device=self.device,
                    enable_memory_saver=self.server_args.enable_memory_saver,
                    start_layer=self.start_layer,
                    end_layer=self.end_layer,
                )
            else:
                PoolCls = current_platform.get_mha_kv_pool_cls()
                self.token_to_kv_pool = PoolCls(
                    self.max_total_num_tokens,
                    page_size=self.page_size,
                    dtype=self.kv_cache_dtype,
                    head_num=self.model_config.get_num_kv_heads(
                        get_parallel().attn_tp_size
                    ),
                    head_dim=self.model_config.head_dim,
                    layer_num=self.num_effective_layers,
                    device=self.device,
                    enable_memory_saver=self.server_args.enable_memory_saver,
                    start_layer=self.start_layer,
                    end_layer=self.end_layer,
                )
        elif (
            self.server_args.attention_backend == "ascend" and not self.mambaish_config
        ):
            if self.is_hybrid_swa:
                from sglang.srt.hardware_backend.npu.memory_pool_npu import (
                    NPUMHATokenToKVPool,
                )

                kwargs = {}
                if self.is_hybrid_swa_compress:
                    kwargs = {
                        # Plan-aware per-rank SWA kv-head count (uneven TP);
                        # falls back to max(1, swa_kv_heads // tp) without a
                        # plan.
                        "swa_head_num": self.model_config.get_swa_num_kv_heads(
                            get_parallel().attn_tp_size
                        ),
                        "swa_head_dim": self.model_config.swa_head_dim,
                        "swa_v_head_dim": self.model_config.swa_v_head_dim,
                        "v_head_dim": self.model_config.v_head_dim,
                    }
                self.token_to_kv_pool = SWAKVPool(
                    size=self.full_max_total_num_tokens,
                    size_swa=self.swa_max_total_num_tokens,
                    page_size=self.page_size,
                    dtype=self.kv_cache_dtype,
                    post_capture_active=self.post_capture_kv_active,
                    head_num=self.model_config.get_num_kv_heads(
                        get_parallel().attn_tp_size
                    ),
                    head_dim=self.model_config.head_dim,
                    swa_attention_layer_ids=self.model_config.swa_attention_layer_ids,
                    full_attention_layer_ids=self.model_config.full_attention_layer_ids,
                    device=self.device,
                    token_to_kv_pool_class=NPUMHATokenToKVPool,
                    **kwargs,
                )
            elif self.use_mla_backend:
                from sglang.srt.hardware_backend.npu.memory_pool_npu import (
                    NPUMLATokenToKVPool,
                )

                self.token_to_kv_pool = NPUMLATokenToKVPool(
                    self.max_total_num_tokens,
                    page_size=self.page_size,
                    dtype=self.kv_cache_dtype,
                    kv_lora_rank=self.model_config.kv_lora_rank,
                    qk_rope_head_dim=self.model_config.qk_rope_head_dim,
                    index_head_dim=(
                        self.model_config.index_head_dim if is_dsa_model else None
                    ),
                    layer_num=self.num_effective_layers,
                    device=self.device,
                    enable_memory_saver=self.server_args.enable_memory_saver,
                    start_layer=self.start_layer,
                    end_layer=self.end_layer,
                )
            else:
                from sglang.srt.hardware_backend.npu.memory_pool_npu import (
                    NPUMHATokenToKVPool,
                )

                self.token_to_kv_pool = NPUMHATokenToKVPool(
                    self.max_total_num_tokens,
                    page_size=self.page_size,
                    dtype=self.kv_cache_dtype,
                    head_num=self.model_config.get_num_kv_heads(
                        get_parallel().attn_tp_size
                    ),
                    head_dim=self.model_config.head_dim,
                    layer_num=self.num_effective_layers,
                    device=self.device,
                    enable_memory_saver=self.server_args.enable_memory_saver,
                    start_layer=self.start_layer,
                    end_layer=self.end_layer,
                )
        elif self.use_mla_backend and is_dsa_model:
            from sglang.srt.layers.cp.utils import get_glm_dsa_cp_layer_shard_info

            (
                dsa_cp_layer_shard_rank,
                dsa_cp_layer_shard_size,
            ) = get_glm_dsa_cp_layer_shard_info(self)
            pool_kwargs = {}
            if self.enable_hisparse:
                PoolCls = HiSparseDSATokenToKVPool
                from sglang.srt.mem_cache.sparsity import parse_hisparse_config

                pool_kwargs["host_to_device_ratio"] = parse_hisparse_config(
                    self.server_args
                ).host_to_device_ratio
            elif dsa_cp_layer_shard_rank is not None:
                # DSA cache layer split: shard KV/indexer layers across CP ranks.
                from sglang.srt.mem_cache.dsa_cache_layer_split import (
                    LayerSplitDSATokenToKVPool,
                )

                PoolCls = LayerSplitDSATokenToKVPool
                pool_kwargs["layer_shard_rank"] = dsa_cp_layer_shard_rank
                pool_kwargs["layer_shard_size"] = dsa_cp_layer_shard_size
            else:
                PoolCls = DSATokenToKVPool
            self.token_to_kv_pool = PoolCls(
                self.max_total_num_tokens,
                page_size=self.page_size,
                dtype=self.kv_cache_dtype,
                kv_lora_rank=self.model_config.kv_lora_rank,
                qk_rope_head_dim=self.model_config.qk_rope_head_dim,
                layer_num=self.num_effective_layers,
                device=self.device,
                kv_cache_dim=self.calculate_mla_kv_cache_dim(),
                enable_memory_saver=self.server_args.enable_memory_saver,
                start_layer=self.start_layer,
                end_layer=self.end_layer,
                index_head_dim=get_dsa_index_head_dim(self.model_config.hf_config),
                **pool_kwargs,
            )
        elif self.use_mla_backend and not self.mambaish_config:
            assert not is_dsa_model
            if is_float4_e2m1fn_x2(self.kv_cache_dtype):
                self.token_to_kv_pool = MLATokenToKVPoolFP4(
                    self.max_total_num_tokens,
                    page_size=self.page_size,
                    dtype=self.kv_cache_dtype,
                    kv_lora_rank=self.model_config.kv_lora_rank,
                    qk_rope_head_dim=self.model_config.qk_rope_head_dim,
                    layer_num=self.num_effective_layers,
                    device=self.device,
                    enable_memory_saver=self.server_args.enable_memory_saver,
                    start_layer=self.start_layer,
                    end_layer=self.end_layer,
                )
            else:
                self.token_to_kv_pool = MLATokenToKVPool(
                    self.max_total_num_tokens,
                    page_size=self.page_size,
                    dtype=self.kv_cache_dtype,
                    kv_lora_rank=self.model_config.kv_lora_rank,
                    qk_rope_head_dim=self.model_config.qk_rope_head_dim,
                    layer_num=self.num_effective_layers,
                    device=self.device,
                    enable_memory_saver=self.server_args.enable_memory_saver,
                    start_layer=self.start_layer,
                    end_layer=self.end_layer,
                )
        else:
            if self.is_hybrid_swa:
                kwargs = {}
                if self.is_hybrid_swa_compress:
                    kwargs = {
                        # Plan-aware per-rank SWA kv-head count (uneven TP);
                        # falls back to max(1, swa_kv_heads // tp) without a
                        # plan.
                        "swa_head_num": self.model_config.get_swa_num_kv_heads(
                            get_parallel().attn_tp_size
                        ),
                        "swa_head_dim": self.model_config.swa_head_dim,
                        "swa_v_head_dim": self.model_config.swa_v_head_dim,
                        "v_head_dim": self.model_config.v_head_dim,
                    }
                self.token_to_kv_pool = SWAKVPool(
                    size=self.full_max_total_num_tokens,
                    size_swa=self.swa_max_total_num_tokens,
                    page_size=self.page_size,
                    dtype=self.kv_cache_dtype,
                    head_num=self.model_config.get_num_kv_heads(
                        get_parallel().attn_tp_size
                    ),
                    head_dim=self.model_config.head_dim,
                    swa_attention_layer_ids=self.model_config.swa_attention_layer_ids,
                    full_attention_layer_ids=self.model_config.full_attention_layer_ids,
                    device=self.device,
                    enable_kv_cache_copy=(
                        self.server_args.speculative_algorithm is not None
                    ),
                    token_to_kv_pool_class=mha_pool_class,
                    **kwargs,
                )
            elif is_minimax_sparse(self.model_config.hf_config):
                _hf_config = self.model_config.hf_config
                sparse_cfg = get_minimax_sparse_attention_config(_hf_config)
                dense_layer_ids, sparse_layer_ids = get_minimax_sparse_layer_ids(
                    sparse_cfg
                )
                disable_value_sparse_layer_ids = (
                    get_minimax_sparse_disable_value_layer_ids(sparse_cfg)
                )
                self.token_to_kv_pool = MiniMaxSparseKVPool(
                    size=self.max_total_num_tokens,
                    page_size=self.page_size,
                    dtype=self.kv_cache_dtype,
                    index_dtype=self.dtype,
                    head_num=self.model_config.get_num_kv_heads(
                        get_parallel().attn_tp_size
                    ),
                    head_dim=self.model_config.head_dim,
                    idx_head_dim=sparse_cfg["sparse_index_dim"],
                    dense_layer_ids=dense_layer_ids,
                    sparse_layer_ids=sparse_layer_ids,
                    disable_value_sparse_layer_ids=disable_value_sparse_layer_ids,
                    device=self.device,
                    enable_memory_saver=self.server_args.enable_memory_saver,
                    start_layer=self.start_layer,
                    end_layer=self.end_layer,
                )
            elif config := self.mambaish_config:
                extra_args = {}
                if self.use_mla_backend:
                    extra_args = {
                        "kv_lora_rank": self.model_config.kv_lora_rank,
                        "qk_rope_head_dim": self.model_config.qk_rope_head_dim,
                    }
                # Uneven-DCP KV replication: the full-attention sub-pool stores
                # the FULL (replicated) kv-heads (gathered in the attention
                # forward from this rank's uneven [2,1,1] projection) but only
                # this rank's owned token slots. head_num must therefore be the
                # FULL total_num_kv_heads, not this rank's uneven share. Stock
                # paths keep the per-rank get_num_kv_heads(attn_tp_size).
                from sglang.srt.distributed.utils import (
                    cp_token_split_factor,
                    get_cp_token_ratios,
                    uneven_dcp_active,
                    uneven_dcp_kv_replicated,
                )

                # M4 (MTP+DCP): the DRAFT worker keeps a plain uneven-TP pool
                # (LOCAL head-sharded kv, FULL token context) -- it is NOT
                # DCP-token-sharded (see FlashInferAttnBackend.__init__ draft
                # gate). Only the TARGET model replicates heads + token-shards.
                _draft_non_dcp = self.is_draft_worker
                if uneven_dcp_kv_replicated(self.dcp_size) and not _draft_non_dcp:
                    _hybrid_kv_head_num = self.model_config.get_total_num_kv_heads()
                else:
                    _hybrid_kv_head_num = self.model_config.get_num_kv_heads(
                        get_parallel().attn_tp_size
                    )
                # WEIGHTED uneven-DCP: max_total_num_tokens is the shared CONTEXT
                # budget C; this rank physically stores only its owned share
                # C * ratio_r / S (ratio-proportional -- the 5090 holds more than
                # the 3080s). Even/default keep the uniform per-rank size. The
                # non-DCP draft pool keeps the full C tokens (raw out_cache_loc
                # index space; the draft is tiny -- 1 layer, local heads).
                if uneven_dcp_active(self.dcp_size) and not _draft_non_dcp:
                    _ratios = get_cp_token_ratios()
                    _S = cp_token_split_factor(self.dcp_size)
                    _ratio_r = _ratios[get_parallel().attn_dcp_rank]
                    _hybrid_pool_size = (self.max_total_num_tokens // _S) * _ratio_r
                else:
                    _hybrid_pool_size = self.max_total_num_tokens
                self.token_to_kv_pool = HybridLinearKVPool(
                    page_size=self.page_size,
                    size=_hybrid_pool_size,
                    dtype=self.kv_cache_dtype,
                    head_num=_hybrid_kv_head_num,
                    head_dim=self.model_config.head_dim,
                    # if draft worker, we only need 1 attention layer's kv pool
                    full_attention_layer_ids=(
                        [0]
                        if self.is_draft_worker
                        else [
                            i
                            for i in config.full_attention_layer_ids
                            if self.start_layer <= i < self.end_layer
                        ]
                    ),
                    device=self.device,
                    mamba_pool=self.req_to_token_pool.mamba_pool,
                    enable_memory_saver=self.server_args.enable_memory_saver,
                    enable_kv_cache_copy=(
                        self.server_args.speculative_algorithm is not None
                    ),
                    use_mla=self.use_mla_backend,
                    start_layer=self.start_layer,
                    full_kv_pool_class=mha_pool_class,
                    post_capture_active=self.post_capture_kv_active,
                    **extra_args,
                )
            else:
                if is_float4_e2m1fn_x2(self.kv_cache_dtype):
                    assert (
                        not enable_page_major
                    ), "page-major KV layout is not supported with fp4 KV cache"
                    self.token_to_kv_pool = MHATokenToKVPoolFP4(
                        self.max_total_num_tokens,
                        page_size=self.page_size,
                        dtype=self.kv_cache_dtype,
                        head_num=self.model_config.get_num_kv_heads(
                            get_parallel().attn_tp_size
                        ),
                        head_dim=self.model_config.head_dim,
                        v_head_dim=self.model_config.v_head_dim,
                        layer_num=self.num_effective_layers,
                        device=self.device,
                        enable_memory_saver=self.server_args.enable_memory_saver,
                        start_layer=self.start_layer,
                        end_layer=self.end_layer,
                        enable_alt_stream=not self.server_args.enable_pdmux,
                        enable_kv_cache_copy=(
                            self.server_args.speculative_algorithm is not None
                        ),
                    )
                else:
                    pool_cls = (
                        NoOpMHATokenToKVPool
                        if self.server_args.prefill_only_disable_kv_cache
                        else mha_pool_class
                    )
                    self.token_to_kv_pool = pool_cls(
                        self.max_total_num_tokens,
                        page_size=self.page_size,
                        dtype=self.kv_cache_dtype,
                        head_num=self.model_config.get_num_kv_heads(
                            get_parallel().attn_tp_size
                        ),
                        head_dim=self.model_config.head_dim,
                        v_head_dim=self.model_config.v_head_dim,
                        layer_num=self.num_effective_layers,
                        device=self.device,
                        enable_memory_saver=self.server_args.enable_memory_saver,
                        start_layer=self.start_layer,
                        end_layer=self.end_layer,
                        enable_alt_stream=not self.server_args.enable_pdmux,
                        enable_kv_cache_copy=(
                            self.server_args.speculative_algorithm is not None
                        ),
                        post_capture_active=self.post_capture_kv_active,
                    )

        # Initialize token_to_kv_pool_allocator
        need_sort = self.server_args.disaggregation_mode in ("decode", "prefill")
        if self.token_to_kv_pool_allocator is None:
            if current_platform.is_out_of_tree():
                AllocatorCls = current_platform.get_paged_allocator_cls()
                self.token_to_kv_pool_allocator = AllocatorCls(
                    self.max_total_num_tokens,
                    page_size=self.page_size,
                    dtype=self.kv_cache_dtype,
                    device=self.device,
                    kvcache=self.token_to_kv_pool,
                    need_sort=need_sort,
                )
            elif _is_npu and (
                self.server_args.attention_backend == "ascend"
                or is_dsv4_model
                or self.hybrid_gdn_config is not None
            ):
                if self.is_hybrid_swa:
                    # DSV4 on NPU: SWA allocator subclass that also drives the
                    # c4/c128 allocators, producing a DSV4OutCacheLoc per alloc.
                    if is_dsv4_model:
                        from sglang.srt.hardware_backend.npu.dsv4.dsv4_allocator import (
                            DSV4NPUTokenToKVPoolAllocator,
                        )

                        swa_allocator_cls = DSV4NPUTokenToKVPoolAllocator
                    else:
                        swa_allocator_cls = SWATokenToKVPoolAllocator
                    self.token_to_kv_pool_allocator = swa_allocator_cls(
                        self.full_max_total_num_tokens,
                        self.swa_max_total_num_tokens,
                        page_size=self.page_size,
                        dtype=self.kv_cache_dtype,
                        device=self.device,
                        kvcache=self.token_to_kv_pool,
                        need_sort=need_sort,
                    )
                else:
                    from sglang.srt.hardware_backend.npu.allocator_npu import (
                        NPUPagedTokenToKVPoolAllocator,
                    )

                    self.token_to_kv_pool_allocator = NPUPagedTokenToKVPoolAllocator(
                        self.max_total_num_tokens,
                        page_size=self.page_size,
                        dtype=self.kv_cache_dtype,
                        device=self.device,
                        kvcache=self.token_to_kv_pool,
                        need_sort=need_sort,
                    )
            else:
                if self.is_hybrid_swa and self.full_max_total_num_tokens == 0:
                    self.token_to_kv_pool_allocator = PureSWATokenToKVPoolAllocator(
                        self.swa_max_total_num_tokens,
                        page_size=self.page_size,
                        dtype=self.kv_cache_dtype,
                        device=self.device,
                        kvcache=self.token_to_kv_pool,
                        need_sort=need_sort,
                    )
                elif self.is_hybrid_swa:
                    self.token_to_kv_pool_allocator = SWATokenToKVPoolAllocator(
                        self.full_max_total_num_tokens,
                        self.swa_max_total_num_tokens,
                        page_size=self.page_size,
                        dtype=self.kv_cache_dtype,
                        device=self.device,
                        kvcache=self.token_to_kv_pool,
                        need_sort=need_sort,
                    )
                else:
                    if self.enable_hisparse:
                        from sglang.srt.mem_cache.sparsity import (
                            parse_hisparse_config,
                        )

                        hisparse_cfg = parse_hisparse_config(self.server_args)
                        self.token_to_kv_pool_allocator = (
                            HiSparseTokenToKVPoolAllocator(
                                self.max_total_num_tokens,
                                page_size=self.page_size,
                                dtype=self.kv_cache_dtype,
                                device=self.device,
                                kvcache=self.token_to_kv_pool,
                                need_sort=need_sort,
                                host_to_device_ratio=hisparse_cfg.host_to_device_ratio,
                            )
                        )
                    elif self.page_size == 1 and self.dcp_size == 1:
                        self.token_to_kv_pool_allocator = TokenToKVPoolAllocator(
                            self.max_total_num_tokens,
                            dtype=self.kv_cache_dtype,
                            device=self.device,
                            kvcache=self.token_to_kv_pool,
                            need_sort=need_sort,
                        )
                    elif (
                        self.server_args.rank_tp_ratio is not None
                        or weightless_kv_active()
                    ):
                        # Weightless-KV fast lane (Option-B) reuses this
                        # NATURAL-page_size DCP allocator branch. Without it,
                        # weightless (rank_tp_ratio=None) would fall to the stock
                        # even-DCP `else` branch which inflates page_size to
                        # page_size*dcp_size (=3 here), and the paged allocator's
                        # alloc_extend triton kernel does tl.arange(0, page_size)
                        # which REQUIRES a power-of-2 length -> CompilationError
                        # at dcp_size=3. This branch keeps page_size NATURAL (1)
                        # and puts the token interleave in the index space +
                        # owner rule instead, exactly as the uneven-DCP feature
                        # does. For weightless even-modulo DCP (no token vector,
                        # uneven_dcp_active=False) the index space is
                        # max_total * cp_token_split_factor(dcp_size) and the
                        # owner rule is loc // dcp_size -- the SAME split
                        # _dcp_owner_write / _dcp_masked_write use, so the
                        # allocator index space and the KV write compaction
                        # agree (no #80-class right-token/wrong-slot corruption).
                        #
                        # Uneven-DCP token-sharding (this feature): the
                        # allocator's index space is the GLOBAL token positions
                        # (max_total * split_factor); each rank physically stores
                        # 1/split_factor of them via the owner rule
                        # (loc // split_factor in the kernels).
                        #
                        # split_factor is dcp_size for even (modulo) DCP and
                        # sum(token ratios) for weighted DCP; it MUST equal the
                        # divisor the DCP kernels use so out_cache_loc //
                        # split_factor stays within the physical pool.
                        #
                        # CRITICAL (hybrid-mamba invariant): the paged page_size
                        # stays NATURAL (base page_size), never inflated by the
                        # factor. Inflating it (the stock `page_size * dcp_size`)
                        # forces the radix cache's page granularity to the DCP
                        # factor, which collides with mamba_cache_chunk_size and
                        # fails page-alignment asserts on non-factor-aligned
                        # sequence lengths under concurrent load. The token-
                        # interleave belongs in the indexing/owner layer, not in
                        # the allocator's page granularity.
                        from sglang.srt.distributed.utils import (
                            cp_token_split_factor,
                            uneven_dcp_active,
                        )

                        # Even (modulo) DCP: max_total_num_tokens is this rank's
                        # per-rank physical pool, and the GLOBAL virtual index
                        # space is max_total * split_factor. WEIGHTED DCP:
                        # max_total_num_tokens is ALREADY the global context C
                        # (= virtual index space); each rank stores only its
                        # ratio_r/S share via the weighted compact owner rule, so
                        # the allocator index space is C itself (not C * factor).
                        if uneven_dcp_active(self.dcp_size):
                            dcp_alloc_size = self.max_total_num_tokens
                        else:
                            dcp_alloc_size = self.max_total_num_tokens * (
                                cp_token_split_factor(self.dcp_size)
                            )
                        self.token_to_kv_pool_allocator = PagedTokenToKVPoolAllocator(
                            dcp_alloc_size,
                            page_size=self.page_size,
                            dtype=self.kv_cache_dtype,
                            device=self.device,
                            kvcache=self.token_to_kv_pool,
                            need_sort=need_sort,
                        )
                    else:
                        # Stock even-DCP (unchanged): interleave via inflated
                        # page granularity.
                        self.token_to_kv_pool_allocator = PagedTokenToKVPoolAllocator(
                            self.max_total_num_tokens * self.dcp_size,
                            page_size=self.page_size * self.dcp_size,
                            dtype=self.kv_cache_dtype,
                            device=self.device,
                            kvcache=self.token_to_kv_pool,
                            need_sort=need_sort,
                        )

            if self.enable_hisparse and is_dsv4_model:
                assert self.is_hybrid_swa, "DeepSeek V4 HiSparse requires SWA mode."
                self.token_to_kv_pool_allocator = (
                    DeepSeekV4HiSparseTokenToKVPoolAllocator(
                        self.token_to_kv_pool_allocator
                    )
                )

            # DSV4-NPU: wire allocator back-ref into req_to_token_pool so its
            # free(req) can release c4/c128 pool pages alongside the slot.
            if hasattr(self.req_to_token_pool, "register_dsv4_allocator"):
                self.req_to_token_pool.register_dsv4_allocator(
                    self.token_to_kv_pool_allocator
                )

        else:
            assert self.is_draft_worker
            if self.is_hybrid_swa:
                swa_allocator = getattr(
                    self.token_to_kv_pool_allocator,
                    "logical_attn_allocator",
                    self.token_to_kv_pool_allocator,
                )
                assert isinstance(swa_allocator, SWATokenToKVPoolAllocator)
                self.token_to_kv_pool.register_mapping(
                    swa_allocator.full_to_swa_index_mapping
                )

        # Defensive check: the explicit validation above should reject known
        # unsupported pool families before allocation. Keep this guard here so
        # future pool-selection refactors fail at boot instead of on first use.
        if (
            self.server_args.prefill_only_disable_kv_cache
            and not self.is_draft_worker
            and not isinstance(self.token_to_kv_pool, NoOpMHATokenToKVPool)
        ):
            raise RuntimeError(
                "--prefill-only-disable-kv-cache expected NoOpMHATokenToKVPool but the "
                f"runtime pool is {type(self.token_to_kv_pool).__name__}. This pool "
                "family is not yet supported by --prefill-only-disable-kv-cache. "
                "Supported configurations today: plain MHA models on CUDA with the FA "
                "(fa3/fa4) prefill backend, --is-embedding, --chunked-prefill-size=-1, "
                "--disable-radix-cache, no context-parallel attention, no HiSparse, "
                "and --kv-cache-dtype != fp4_e2m1."
            )

    def _hybrid_kv_token_cap(self: ModelRunner) -> Optional[int]:
        """Physically reachable ceiling on max_total_num_tokens for hybrid
        mamba/GDN + attention models (#79).

        In a hybrid model only the (few) full-attention layers carry a KV
        cache, so the per-token cell size is tiny and the profiling path's
        ``available_bytes // cell_size`` inflates the token pool to a value
        that can never be filled: serving concurrency is bounded by the mamba
        state pool (one live sequence per state slot) and each sequence spans
        at most ``context_len`` tokens. The most KV tokens ever in flight is
        therefore ``max_running_requests * (context_len + per-request decode
        headroom)``. Beyond that ceiling the KV pool holds slots that no mamba
        state can ever index (wasted VRAM) and the reported number swings with
        fragmentation across reboots (A3B-GGUF reported ~3.3M against a
        16*32768=524288 true bound).

        Returns ``None`` for non-hybrid models -- their pure-attention KV pools
        legitimately exceed running*ctx to back the prefix/radix cache -- and
        whenever concurrency or context length cannot be determined (no cap).
        """
        if self.mambaish_config is None:
            return None
        sa = self.server_args
        # Number of distinct sequences the system can hold state for. Take the
        # larger of the scheduler's admission limit and the mamba state pool's
        # own capacity (state slots // mamba_ratio, which INCLUDES the radix
        # cache's extra state buffer). Using the mamba capacity keeps the cap
        # from hurting prefix-cache-heavy workloads where a long shared prefix
        # occupies one mamba state but many KV tokens.
        concurrency = 0
        if sa.max_running_requests and sa.max_running_requests > 0:
            concurrency = sa.max_running_requests
        if sa.max_mamba_cache_size:
            ratio = max(self._calculate_mamba_ratio(), 1)
            concurrency = max(concurrency, sa.max_mamba_cache_size // ratio)
        if concurrency <= 0:
            return None
        ctx = self.model_config.context_len
        if not ctx or ctx <= 0:
            return None
        # Per-request decode/spec headroom beyond context_len, matching the
        # req_to_token allocation width.
        extra = get_req_to_token_extra_context_len(sa)
        return int(concurrency) * (int(ctx) + int(extra))

    def _swa_hybrid_kv_token_cap(self: ModelRunner) -> Optional[int]:
        """Physically reachable ceiling on max_total_num_tokens for hybrid
        SWA + global-attention models (Gemma-style; #90, same disease class
        as the mamba-hybrid cap above).

        In a hybrid-SWA model the sliding-window layers carry a CONSTANT
        per-request KV footprint (bounded by the window plus eviction lag),
        while only the global-attention layers grow with the context length.
        The profiling path's ``available_bytes // cell_size`` knows nothing of
        either bound, so it inflates the pools to fragmentation-dependent,
        physically unreachable sizes (Gemma4-31B TP=3 uneven: per-rank
        capacities 103024/116951/85679 for a 4-req / 8192-ctx server whose
        full-pool ceiling is 4*(8192+4); with weighted DCP the un-sharded SWA
        pool was even sized at the global 249472 budget and OOM'd outright).

        The ceiling follows the code's own pool accounting
        (HybridSWAPoolConfigurator: full = max_total, swa = ratio * max_total):
        - full pool need:  concurrency * (context_len + decode headroom)
          -- the same growing-part formula as the mamba-hybrid cap;
        - swa pool need:   swa_pool_token_cap() -- the SWAChunkCap
          per-request worst case (window + eviction interval + decode
          over-allocation, plus in-flight prefill chunks);
        and since the split pins swa = ratio * max_total, the binding cap is
          max(full_need, ceil(swa_need / ratio)).

        Returns ``None`` (no cap) for non-SWA-hybrid models, DeepSeek-V4
        (separate c4/c128 pool accounting), all-SWA models (max_total IS the
        swa pool and the radix cache legitimately holds many in-window
        prefixes), mamba hybrids (the #79 cap governs), and whenever
        concurrency, context length, or window size cannot be determined --
        so every other path is byte-for-byte unchanged.
        """
        if not self.is_hybrid_swa or is_deepseek_v4(self.model_config.hf_config):
            return None
        if self.mambaish_config is not None:
            return None
        if len(self.model_config.full_attention_layer_ids) == 0:
            return None
        sa = self.server_args
        if not sa.max_running_requests or sa.max_running_requests <= 0:
            return None
        ctx = self.model_config.context_len
        if not ctx or ctx <= 0:
            return None
        window = getattr(self, "sliding_window_size", None)
        if not window or window <= 0:
            return None
        concurrency = int(sa.max_running_requests)
        # Per-request decode/spec headroom beyond context_len, matching the
        # req_to_token allocation width.
        extra = get_req_to_token_extra_context_len(sa)
        full_need = concurrency * (int(ctx) + int(extra))
        # --swa-pool-sizing cap (task #91 Stage A): the SWA pool is PINNED at
        # its window-bounded worst case (SWAChunkCapPoolConfigurator) and no
        # longer scales as ratio * max_total, so the ceil(swa_need / ratio)
        # term below (which back-derives the max_total needed to make the
        # ratio-sized swa pool big enough) does not apply: the reachability
        # ceiling on max_total is the full-pool need alone. Ratio mode (the
        # default) keeps the #90 formula byte-identical.
        if sa.swa_pool_sizing == "cap":
            return full_need
        # Local import avoids a pool_configurator import cycle.
        from sglang.srt.model_executor.pool_configurator import swa_pool_token_cap

        swa_need = int(swa_pool_token_cap(self, concurrency))
        ratio = sa.swa_full_tokens_ratio
        if ratio and ratio > 0:
            return max(full_need, int(math.ceil(swa_need / ratio)))
        return full_need

    def _apply_hybrid_kv_token_cap(
        self: ModelRunner,
        token_capacity: int,
        cap: Optional[int],
        kind: str = "mamba",
    ) -> int:
        """Clamp a resolved token capacity to the hybrid physical ceiling and
        log once when the cap actually binds."""
        if cap is None or token_capacity <= cap:
            return token_capacity
        if kind == "swa":
            logger.info(
                "Hybrid SWA/global-attention KV cap: max_total_num_tokens "
                "%d -> %d (max_running_requests=%s x (context_len=%d + "
                "headroom) for the global layers; sliding-window layers are "
                "window-bounded (window=%s), so the profiled capacity is "
                "physically unreachable).",
                token_capacity,
                cap,
                self.server_args.max_running_requests,
                self.model_config.context_len,
                getattr(self, "sliding_window_size", None),
            )
            return cap
        logger.info(
            "Hybrid mamba/attention KV cap: max_total_num_tokens %d -> %d "
            "(max_running_requests=%s x (context_len=%d + headroom); the "
            "full-attention-only KV cell size otherwise overstates a "
            "physically unreachable capacity).",
            token_capacity,
            cap,
            self.server_args.max_running_requests,
            self.model_config.context_len,
        )
        return cap

    def _apply_token_constraints(self: ModelRunner, token_capacity: int) -> int:
        """Apply external constraints to token capacity: user cap, PP sync,
        and the hybrid mamba/attention physical ceiling (#79).

        Page alignment is handled by the configurator, not here.
        If constraints change the value, the configurator re-runs and re-aligns.
        """
        user_limit = self.server_args.max_total_tokens
        hybrid_cap = self._hybrid_kv_token_cap()
        hybrid_cap_kind = "mamba"
        if hybrid_cap is None:
            swa_cap = self._swa_hybrid_kv_token_cap()
            if swa_cap is not None:
                hybrid_cap, hybrid_cap_kind = swa_cap, "swa"

        # Apply user-specified upper bound
        if user_limit is not None:
            if user_limit > token_capacity:
                logging.warning(
                    f"max_total_tokens={user_limit} is larger than the profiled value "
                    f"{token_capacity}. Use the profiled value instead."
                )
            token_capacity = min(token_capacity, user_limit)

        # WEIGHTED uneven-DCP: decouple the reported CONTEXT budget from each
        # rank's non-uniform physical pool. token_capacity arrives as P_r =
        # this rank's physical token capacity (available_bytes // full-kv-head
        # cell). Under the weighted token split rank r physically stores
        # C * ratio_r / S tokens, so the largest context that fits EVERY rank
        # is C = min_r(P_r // ratio_r) * S. This C becomes max_total_num_tokens
        # (what the scheduler admits and reports); the per-rank pool + allocator
        # are sized from it separately (HybridLinearKVPool / allocator below).
        from sglang.srt.distributed.utils import (
            cp_token_split_factor,
            get_cp_token_ratios,
            uneven_dcp_active,
        )

        if (
            uneven_dcp_active(self.dcp_size)
            and get_world_group().world_size > 1
        ):
            ratios = get_cp_token_ratios()
            split_factor = cp_token_split_factor(self.dcp_size)
            ratio_r = ratios[get_parallel().attn_dcp_rank]
            local_blocks = torch.tensor(
                int(token_capacity) // int(ratio_r), dtype=torch.int64
            )
            torch.distributed.all_reduce(
                local_blocks,
                op=torch.distributed.ReduceOp.MIN,
                group=get_world_group().cpu_group,
            )
            token_capacity = int(local_blocks.item()) * split_factor
            if user_limit is not None:
                token_capacity = min(token_capacity, user_limit)
            token_capacity = self._apply_hybrid_kv_token_cap(
                token_capacity, hybrid_cap, hybrid_cap_kind
            )
            return token_capacity

        # Sync across PP ranks (each may have different layer counts) and
        # across uneven-TP ranks (--rank-gpu-memory-mib: byte profiling is
        # rank-local, and per-token bytes differ with each rank's kv-head
        # share — the TOKEN capacity is the unit every rank must agree
        # on, so the scheduler's single max_total_num_tokens stays
        # consistent; proportional budgets make the min nearly lossless).
        needs_capacity_sync = self.pp_size > 1 or (
            self.server_args.uneven_memory_budgets_active()
            and get_world_group().world_size > 1
        )
        if needs_capacity_sync:
            tensor = torch.tensor(token_capacity, dtype=torch.int64)
            torch.distributed.all_reduce(
                tensor,
                op=torch.distributed.ReduceOp.MIN,
                group=get_world_group().cpu_group,
            )
            token_capacity = tensor.item()

        token_capacity = self._apply_hybrid_kv_token_cap(
            token_capacity, hybrid_cap, hybrid_cap_kind
        )
        return token_capacity

    #: Weight families the self-calibration can shift, mapped to the
    #: environment variable of the restart hint. "mlp" = dense-MLP /
    #: shared experts (tp_family on the linear layers), "moe" = fused
    #: expert weights (moe_tp_family on FusedMoE).
    _CALIBRATION_FAMILY_ENV = {
        "mlp": "SGLANG_UNEVEN_MLP_VECTOR",
        "moe": "SGLANG_UNEVEN_MOE_VECTOR",
    }

    def _family_local_stats(self: ModelRunner, family: str) -> Optional[tuple]:
        """(per-rank units, family parameter bytes) of this rank for one
        shiftable weight family.

        Walks the model for layers partitioning under `family`: linear
        layers carry tp_family/tp_units (dense MLP / shared experts),
        FusedMoE carries moe_tp_family/moe_tp_units (expert weights).
        The unit currency is the finest unit count among those layers
        (quant-block-coarsened layers partition proportionally, which is
        exact enough for the calibration estimate). Returns None when
        the model has no layers of this family."""
        units_total = 0
        family_bytes = 0
        for module in self.model.modules():
            module_family = getattr(module, "tp_family", None) or getattr(
                module, "moe_tp_family", None
            )
            if module_family != family:
                continue
            module_units = getattr(module, "tp_units", None) or getattr(
                module, "moe_tp_units", None
            )
            if module_units:
                units_total = max(units_total, module_units)
            family_bytes += sum(
                p.numel() * p.element_size()
                for p in module.parameters(recurse=False)
                # GGUF lazy params (UninitializedParameter) have no shape
                # until materialize; skip them in this size estimate.
                if not isinstance(p, torch.nn.parameter.UninitializedParameter)
            )
        if units_total <= 0 or family_bytes <= 0:
            return None
        local_units = tp_partition_size(
            units_total, self.tp_size, self.tp_rank, units_total, family
        )
        return local_units, family_bytes

    def _maybe_suggest_mlp_rebalance(self: ModelRunner, budget_bytes: int) -> None:
        """Uneven-TP self-calibration: after the rank-local KV profiling,
        check whether shifting weight-family units (dense-MLP "mlp" and
        expert "moe") between ranks would raise the MIN-synced KV token
        pool, and log a one-line restart hint.

        Every rank contributes (profiled KV byte budget, local token
        capacity, and per family its current partition in units + the
        family's parameter bytes) via all_gather; when the token
        capacities diverge by more than 10%, the maximin solver computes
        the unit vectors that equalize them (each shed unit turns its
        weight bytes into KV budget on the pinned rank; on MoE models
        the "moe" family supplies most of the shiftable mass). Purely
        advisory — nothing is resized in-process; the hint asks for a
        restart with SGLANG_UNEVEN_MLP_VECTOR / SGLANG_UNEVEN_MOE_VECTOR.
        Silent when the active vectors already balance the ranks. All
        gating conditions are rank-uniform, so every rank reaches the
        collective or none does.

        Ceiling note: shifting weights conserves the summed free bytes,
        so the reachable pool is bounded by sum(budget) /
        sum(bytes_per_token) — growing beyond that needs bigger budgets
        or smaller weights (quantization), not a different vector."""
        if not self.server_args.uneven_memory_budgets_active():
            return
        if get_world_group().world_size <= 1:
            return
        if not tp_plan_active(self.tp_size):
            # The family vectors require an active base plan.
            return

        local_families = {}
        for family in self._CALIBRATION_FAMILY_ENV:
            stats = self._family_local_stats(family)
            if stats is not None:
                local_families[family] = stats
        if not local_families:
            # Model has no shiftable family layers (not converted for
            # uneven TP); identical on every rank, so returning is safe.
            return

        # Local token capacity of this rank's budget (pre-MIN-sync).
        from sglang.srt.model_executor.pool_configurator import (
            create_memory_pool_configurator,
        )

        configurator = create_memory_pool_configurator(self)
        local_tokens = configurator.calculate_pool_sizes(
            budget_bytes, self.page_size
        ).max_total_num_tokens
        if local_tokens <= 0:
            local_tokens = 0

        world = get_world_group().world_size
        payload = (
            float(budget_bytes),
            int(local_tokens),
            {
                family: (int(units), float(fam_bytes))
                for family, (units, fam_bytes) in local_families.items()
            },
        )
        gathered: list = [None] * world
        torch.distributed.all_gather_object(
            gathered, payload, group=get_world_group().cpu_group
        )
        free_bytes = [g[0] for g in gathered]
        tokens = [g[1] for g in gathered]
        if any(t <= 0 for t in tokens):
            return
        cur_min = min(tokens)

        # A user token cap below every rank's capacity pins the pool
        # anyway; rebalancing cannot grow it.
        user_limit = self.server_args.max_total_tokens
        if user_limit is not None and user_limit <= cur_min:
            return

        # Families present on every rank (the module sets are identical
        # across ranks, so this is normally all of them).
        family_names = set(gathered[0][2])
        for g in gathered[1:]:
            family_names &= set(g[2])
        families = {
            family: (
                [g[2][family][0] for g in gathered],
                [g[2][family][1] for g in gathered],
            )
            for family in sorted(family_names)
        }
        if not families:
            return

        bytes_per_token = [free_bytes[r] / tokens[r] for r in range(world)]
        suggestion = suggest_unit_rebalance_multi(
            free_bytes, bytes_per_token, families
        )
        if suggestion is None:
            # Balanced (max/min <= 1.10) or no strictly better partition
            # — in particular: active vectors that already equalize the
            # ranks stay silent.
            return
        changed, cur_min_tokens, projected = suggestion
        if self.tp_rank == 0:
            assignments = " ".join(
                f"{self._CALIBRATION_FAMILY_ENV[family]}="
                + ",".join(str(u) for u in changed[family])
                for family in sorted(changed)
            )
            logger.warning(
                "uneven TP: restart with %s to raise the KV pool from "
                "%d to ~%d tokens",
                assignments,
                cur_min_tokens,
                projected,
            )

    def _maybe_suggest_dcp_token_vector(
        self: ModelRunner, budget_bytes: int
    ) -> None:
        """Uneven-DCP token-vector self-calibration (analogue of vLLM's
        VLLM_UNEVEN_TOKEN_VECTOR): after the rank-local KV profiling, derive
        the OPTIMAL token-axis split vector from each rank's ACTUAL profiled
        token capacity P_r (not the rough pre-boot budget estimate) and, if it
        differs from the active vector, log a restart hint.

        Under the weighted owner rule the reported context budget is
        C = min_r(P_r // ratio_r) * sum(ratios); it is maximized when the
        ratios are proportional to the measured P_r. P_r = physical token
        capacity of this rank's budget with the full-kv-head cell — the same
        quantity _apply_token_constraints consumes, and dtype-independent
        (works for FP8 / AWQ / GGUF alike). The optimal vector is
        partition_units(64, [P_r...]) (largest-remainder), gcd-reduced.

        Purely advisory — nothing is resized in-process; the hint asks for a
        restart with SGLANG_UNEVEN_TOKEN_VECTOR=a,b,c, which resolve_cp_token_
        ratios honors on the next boot so the pool converges to the optimum.
        Because P_r is independent of the active vector, this converges in one
        feedback step. All gating conditions BEFORE the all_gather are
        rank-uniform (server args / installed vector / world size), so every
        rank reaches the collective or none does; the rank-LOCAL capacity
        P_r is gathered unconditionally (clamped to >= 0) and degenerate
        ranks are rejected by the uniform post-gather any(p <= 0) check —
        an early return on the local value would let one rank skip the
        collective the others entered (distributed hang)."""
        from sglang.srt.distributed.utils import (
            cp_token_split_factor,
            get_cp_token_ratios,
            partition_units,
            uneven_dcp_active,
        )

        if not uneven_dcp_active(self.dcp_size):
            return
        if get_world_group().world_size <= 1:
            return
        active = get_cp_token_ratios()
        if not active or len(active) != self.dcp_size:
            return

        # Local physical token capacity P_r of this rank's budget (full-kv-head
        # cell, pre owner-rule split) — matches _apply_token_constraints' input.
        from sglang.srt.model_executor.pool_configurator import (
            create_memory_pool_configurator,
        )

        configurator = create_memory_pool_configurator(self)
        # NOT an early-return guard: local_p is rank-LOCAL, so bailing here
        # would skip the collective below on this rank only while the other
        # ranks enter it (distributed hang). Clamp and gather unconditionally;
        # the uniform any(p <= 0) check after the gather rejects degenerate
        # ranks on EVERY rank alike (same pattern as _maybe_suggest_mlp_
        # rebalance's local_tokens handling).
        local_p = max(
            int(
                configurator.calculate_pool_sizes(
                    budget_bytes, self.page_size
                ).max_total_num_tokens
            ),
            0,
        )

        world = get_world_group().world_size
        payload = (int(get_parallel().attn_dcp_rank), int(local_p))
        gathered: list = [None] * world
        torch.distributed.all_gather_object(
            gathered, payload, group=get_world_group().cpu_group
        )
        # Order the capacities by DCP rank (the token vector is indexed by
        # attn_dcp_rank, which need not equal the global rank).
        p_by_rank = [0] * self.dcp_size
        for dcp_rank, p_val in gathered:
            if 0 <= dcp_rank < self.dcp_size:
                p_by_rank[dcp_rank] = p_val
        if any(p <= 0 for p in p_by_rank):
            return

        optimal = partition_units(64, p_by_rank)
        g = math.gcd(*optimal)
        optimal = [v // g for v in optimal]

        def _context_budget(vector: list) -> int:
            return min(p_by_rank[r] // vector[r] for r in range(self.dcp_size)) * sum(
                vector
            )

        c_active = _context_budget(active)
        c_optimal = _context_budget(optimal)

        if self.tp_rank == 0:
            if optimal == active or c_optimal <= c_active:
                logger.info(
                    "Uneven DCP token vector converged (balanced): %s, "
                    "max_total_num_tokens=%d (per-rank profiled capacity %s).",
                    active,
                    c_active,
                    p_by_rank,
                )
            else:
                logger.warning(
                    "Uneven DCP: restart with SGLANG_UNEVEN_TOKEN_VECTOR=%s to "
                    "raise max_total_num_tokens from %d to ~%d (per-rank "
                    "profiled capacity %s; active vector %s leaves ranks idle).",
                    ",".join(str(v) for v in optimal),
                    c_active,
                    c_optimal,
                    p_by_rank,
                    active,
                )

    def _resolve_max_num_reqs(self: ModelRunner, token_capacity: int) -> int:
        """Compute max concurrent requests (per dp worker) from the finalized
        token capacity."""
        # Estimate pool size (used as upper bound when user specifies max_running_requests)
        estimated = int(token_capacity / self.model_config.context_len * 512)
        estimated = max(min(estimated, 4096), 2048)

        max_num_reqs = self.server_args.max_running_requests
        if max_num_reqs is not None:
            requested_per_worker = max_num_reqs // self.dp_size
            max_num_reqs = min(requested_per_worker, token_capacity // 2)
        else:
            requested_per_worker = None
            max_num_reqs = min(estimated, token_capacity // 2)

        if self.mambaish_config is not None:
            ratio = self._calculate_mamba_ratio()
            max_num_reqs = min(
                max_num_reqs, self.server_args.max_mamba_cache_size // ratio
            )

            if max_num_reqs <= 0:
                raise RuntimeError(
                    f"Hybrid (mamba/linear-attention) state cache is too small to serve "
                    f"any requests. max_mamba_cache_size={self.server_args.max_mamba_cache_size}, "
                    f"mamba_ratio={ratio}, resulting max_num_reqs={max_num_reqs}. "
                    f"Try: (1) reduce --max-running-requests, "
                    f"(2) increase --mem-fraction-static, or "
                    f"(3) use GPUs with more memory."
                )
        if requested_per_worker is not None and max_num_reqs < requested_per_worker:
            logger.warning(
                "max_running_requests was reduced from the requested %d to %d "
                "(per dp worker) due to the available KV cache capacity.",
                requested_per_worker,
                max_num_reqs,
            )
        return max_num_reqs

    def _apply_memory_pool_config(self: ModelRunner, config: MemoryPoolConfig):
        """Apply a resolved MemoryPoolConfig and initialize pools."""
        self.max_total_num_tokens = config.max_total_num_tokens
        self.max_running_requests = config.max_running_requests
        if self.is_hybrid_swa:
            self.full_max_total_num_tokens = config.full_max_total_num_tokens
            self.swa_max_total_num_tokens = config.swa_max_total_num_tokens

        # DSV4 compressed-attention pool sizes. Draft worker reuses target's
        # full/swa sizes but does NOT own c4/c128/state pools (those live on
        # the target rank only); zero them out regardless of what config holds.
        if self.is_draft_worker:
            self.c4_max_total_num_tokens = 0
            self.c128_max_total_num_tokens = 0
            self.c4_state_pool_size = 0
            self.c128_state_pool_size = 0
        else:
            self.c4_max_total_num_tokens = config.c4_max_total_num_tokens
            self.c128_max_total_num_tokens = config.c128_max_total_num_tokens
            self.c4_state_pool_size = config.c4_state_pool_size
            self.c128_state_pool_size = config.c128_state_pool_size

        # Draft worker does not own the compression-state pools, but keep the
        # dtype attributes initialized so _init_pools can share one code path.
        if is_deepseek_v4(self.model_config.hf_config):
            self.c4_state_dtype, self.c128_state_dtype = (
                _get_dsv4_compress_state_dtypes()
            )

        self._init_pools()

    def _config_from_budget(
        self: ModelRunner, budget_bytes: int, *, cap_tokens: Optional[int] = None
    ) -> MemoryPoolConfig:
        """Turn a KV byte budget into a pool config via the configurator, re-applying
        the external token constraints (user cap, page alignment, PP sync) and the
        optional ``cap_tokens`` clamp."""
        # Local import avoids a pool_configurator import cycle.
        from sglang.srt.model_executor.pool_configurator import (
            create_memory_pool_configurator,
        )

        configurator = create_memory_pool_configurator(self)
        config = configurator.calculate_pool_sizes(budget_bytes, self.page_size)
        max_tokens = self._apply_token_constraints(config.max_total_num_tokens)
        if cap_tokens is not None:
            max_tokens = min(max_tokens, cap_tokens)
        if max_tokens != config.max_total_num_tokens:
            config = configurator.calculate_pool_sizes_from_max_tokens(
                max_tokens, self.page_size
            )
        return config

    def _resolve_memory_pool_config(
        self: ModelRunner, pre_model_load_memory: int
    ) -> MemoryPoolConfig:
        """Profile GPU memory and resolve all pool parameters into a config."""
        from sglang.srt.model_executor.pool_configurator import (
            create_memory_pool_configurator,
        )

        available_bytes = self._profile_available_bytes(pre_model_load_memory)
        if not self.post_capture_kv_active:
            # Uneven-TP self-calibration on the final profiled budget;
            # with post-capture sizing the (more accurate) post-capture
            # measurement runs it instead.
            self._maybe_suggest_mlp_rebalance(available_bytes)
            self._maybe_suggest_dcp_token_vector(available_bytes)
        config = self._config_from_budget(available_bytes)
        config.max_running_requests = self._resolve_max_num_reqs(
            config.max_total_num_tokens
        )
        configurator = create_memory_pool_configurator(self)
        config = configurator.finalize_with_max_running_requests(config)
        config.mem_fraction_static = self.server_args.mem_fraction_static
        return config

    def init_memory_pool(self: ModelRunner, pre_model_load_memory: int):
        if not self.spec_algorithm.is_none() and self.is_draft_worker:
            assert (
                self.memory_pool_config is not None
            ), "Draft worker requires memory_pool_config"
        else:
            self.memory_pool_config = self._resolve_memory_pool_config(
                pre_model_load_memory
            )

        self._apply_memory_pool_config(self.memory_pool_config)

        logger.info(
            f"Memory pool end. "
            f"avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )
