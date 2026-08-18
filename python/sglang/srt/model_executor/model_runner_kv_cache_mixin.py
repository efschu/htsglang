from __future__ import annotations

import dataclasses
import logging
import math
import os
from typing import TYPE_CHECKING, Optional, Tuple

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
    uneven_dcp_kv_replicated,
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
    empty_device_cache,
    get_available_gpu_memory,
    get_device_memory_capacity,
    is_float4_e2m1fn_x2,
    is_hip,
    is_npu,
)

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.model_executor.pool_configurator import MemoryPoolConfig


def _moe_offload_active() -> bool:
    """Is MoE expert offload on anywhere in the group?

    Imported lazily to keep this module's import graph unchanged. Group-wide
    on purpose: this gates where weights are placed, and a rank that answered
    differently from its peers would build a structurally different model.
    """
    from sglang.srt.layers.moe.resident_fraction import offload_active

    return offload_active()


def _current_card_uuid() -> str:
    """NVML UUID of the card this rank runs on, or ``""`` when unresolvable.

    Stamped into the measured KV-budget registry so a later boot can tell
    whether a stored per-rank balance belongs to the card that rank now sits
    on. Never raises: an unresolvable identity degrades the registry to its
    pre-#331 behaviour (accepted with a warning), it does not fail a boot.
    """
    try:
        from sglang.srt.registry import nvml

        return nvml.current_device_uuid()
    except Exception as exc:  # noqa: BLE001 - identity is advisory here
        logger.debug("Measured KV-budget: card identity unresolved (%s)", exc)
        return ""


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
# Trigger for the #307 ceiling fit, NOT a safety margin: when the pool the
# concurrency target asks for would leave the token pool less than this, the
# boot is lost either way (a KV pool below one prefill chunk cannot serve a
# request), so the pool is fitted to the budget instead of the budget being
# overspent. Above this the pool keeps the size it has today, byte for byte.
MAMBA_CEILING_FIT_MIN_KV_MIB = 256

#: Ledger label of the mamba post in the per-rank budget message. Named once
#: so the message's ceiling hint can find the post it has to reason about.
MAMBA_BUDGET_POST = (
    "mamba state pool + speculative intermediate state + prefill activation reserve"
)

#: The names :func:`decompose_mamba_budget_post` emits, in order. Anything that
#: sums "the mamba post" must accept BOTH shapes -- the lump and these parts --
#: or it silently reads zero on exactly the boots that carry the instrument.
MAMBA_POST_PART_NAMES = (
    "mamba state pool",
    "speculative intermediate state",
    "prefill activation reserve",
)


def mamba_post_total_gb(posts) -> float:
    """The mamba post total, whether it was emitted lumped or decomposed.

    Exists because the decomposition broke the ceiling hint in
    ``budget_exhausted_message``: that summed posts named ``MAMBA_BUDGET_POST``,
    which matches nothing once the post is emitted as three parts, so the
    ``--max-running-requests-ceiling`` advice disappeared from the refusal
    message precisely on the boots that have the new instrument. Callers should
    use this rather than matching a name.
    """
    return sum(
        gb
        for name, gb in posts
        if name == MAMBA_BUDGET_POST or name in MAMBA_POST_PART_NAMES
    )


def _note_mamba_component(runner, name: str, gb: float) -> None:
    """Record one NAMED sub-term of the lumped mamba budget post (#704).

    Module-level and duck-typed ON PURPOSE. As a bound method it required
    every test double to grow an attribute it had no reason to have -- the
    #624 stub-drift class -- and it broke eleven ceiling-fit tests the moment
    it landed. Production must not demand more of a stand-in than the
    behaviour under test actually needs.
    """
    acc = getattr(runner, "_mamba_budget_components", None)
    if acc is None:
        acc = {}
        setattr(runner, "_mamba_budget_components", acc)
    acc[name] = acc.get(name, 0.0) + float(gb)


def budget_holdback_mib(profiled_bytes: int, adjusted_bytes: int) -> float:
    """The TRUE per-rank holdback, in MiB (#704, fourth instrument).

    The quantity between the profiler's ``rest`` and what the configurator
    actually receives. On metal it lands at 6,688 / 3,561 / 5,166 MiB while
    ``derived_rank_auto_reserve_mib`` returns 4,160 UNIFORMLY for the same
    boot's arguments -- so the holdback is not that function's output, and with
    three exactly-collinear data points its form cannot be fitted (single-factor
    layer models are already falsified by non-monotonicity; see
    DESIGN_704_reserve_vs_layout.md). Emitting it settles the form in one boot
    instead of a regression that cannot separate collinear regressors.

    Reports rather than sanitises. A NEGATIVE holdback means the seam handed
    budget back, which is a real and loud state; clamping it to zero would hide
    exactly the accounting-error class this ticket has spent four instruments
    chasing.
    """
    return (float(profiled_bytes) - float(adjusted_bytes)) / float(1 << 20)


def budget_holdback_fraction(profiled_bytes: int, adjusted_bytes: int):
    """Holdback as a fraction of the profiled budget, or ``None`` if there was
    no budget to hold back from -- a division nobody should have to guard."""
    if not profiled_bytes:
        return None
    return (float(profiled_bytes) - float(adjusted_bytes)) / float(profiled_bytes)


def decompose_mamba_budget_post(total_gb: float, components: dict):
    """Split the lumped mamba budget post into its three NAMED components.

    The label has always named three terms; the post emitted one number. That
    lump is why an accounting gap could not be attributed: the post nominally
    covers MORE than the "Mamba Cache is allocated" line (it adds the prefill
    activation reserve) yet measures 0.155 / 0.111 / 0.089 GiB LESS on the three
    stages of the live boot. A term covering more cannot legitimately measure
    less, and with one number there is no way to say which part carries it.

    ``components`` holds the sub-terms the sizer actually measured. Whatever
    they do not explain IS the state pool, so it is emitted under that name
    rather than left as an anonymous remainder -- the parts always sum to the
    lump exactly, and a NEGATIVE residual is reported rather than clamped,
    because a negative state pool is precisely the error this instrument exists
    to surface.

    With no components (the other budget branches never populate them) the lump
    is returned unchanged, so nothing else has to know about this.
    """
    if not components:
        return [(MAMBA_BUDGET_POST, float(total_gb))]
    named = sum(float(v) for v in components.values())
    out = [(MAMBA_POST_PART_NAMES[0], float(total_gb) - named)]
    out.extend((str(k), float(v)) for k, v in components.items())
    return out


#: State-slot count a pipeline stage WITHOUT any linear-attention layers in
#: its layer window contributes to the world MIN-agreement on
#: max_mamba_cache_size (#201 slice 3). Such a stage allocates zero state
#: bytes, so its memory budget puts no bound on the slot count -- it must
#: not be the rank that binds the world minimum. Large but far below
#: int64 overflow territory for the downstream `size * per_req` products
#: (its per_req is 0 by construction).
PP_STAGE_NO_MAMBA_STATE_SLOTS = 1 << 30

logger = logging.getLogger(__name__)


def corridor_mode_active(server_args) -> bool:
    """--rank-kv-ratio corridor (#602), asked tolerantly.

    A module-level function rather than a mixin method because the token-vector
    path is exercised with stub runners that bind only the methods under test;
    a predicate reached through ``self`` would make every such caller carry
    scaffolding for a mode it does not use.

    A server_args view that predates the mode simply is not in it, which is the
    DEFAULT path and byte-identical -- so an absent predicate answers False
    rather than raising. Everything DOWNSTREAM of a True answer refuses loudly
    instead: once the mode is on, a missing corridor input is an error, never a
    quiet fallback to "no floor".
    """
    predicate = getattr(server_args, "uneven_kv_corridor_mode", None)
    return bool(predicate()) if callable(predicate) else False


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
    # === #119: expert-offload VRAM -> KV pool ==============================
    # The expert offload (#77/#123) parks cold experts in a pinned host pool and
    # gives their VRAM back. Nothing in the sizing code has to ADD that memory
    # anywhere: the budget below is derived from a live free-memory reading, so
    # the reclaim lands in it by construction -- as long as two ordering
    # properties hold. They are cheap to state and were previously unenforced:
    #
    #   1. RELEASE BEFORE MEASURE. The offload must be installed before the
    #      profiling runs. It is (model_runner.load_model -> the eager install),
    #      but only by call-site accident; the #77 lazy install ran AFTER sizing
    #      and cost the entire win silently. _assert_expert_offload_installed
    #      turns that back into a loud failure instead of a quiet regression.
    #   2. RELEASE BEFORE *ANY* RANK MEASURES. get_available_gpu_memory reads
    #      torch.cuda.mem_get_info(), which is DRIVER-level and therefore sees
    #      co-located siblings too, while the caching allocator only returns
    #      freed blocks to the driver at empty_cache(). Each rank empties its
    #      own cache inside its own reading, unsynchronized -- so a rank that
    #      reads early sees a sibling's already-freed expert weights as still
    #      occupied. Below the offload that skew was small; with the offload it
    #      is the whole reclaim (~18 GiB on the co-located card in the #77 122B
    #      run), and it lands in the min-reduce/co-location terms as pure noise.
    #      _expert_offload_release_sync makes the release a group-ordered step.
    #
    # No new budget term is introduced, so the graph-capture reserve (#68
    # derived_rank_auto_reserve_mib / reserve_for_graph_mb, both folded in
    # before this point) is untouched: the reclaim arrives as free bytes and is
    # spent net of the same reserves as any other free byte.

    def _expert_offload_lane_active(self: ModelRunner) -> bool:
        """True when the #119 reclaim handling applies at all.

        Both terms are env vars and therefore WORLD-UNIFORM: every rank agrees
        without communicating, so the collectives further down can never be
        entered by only a subset of ranks. With the offload off this returns
        False before any memory or collective operation, leaving the default
        sizing path byte-identical.
        """
        return envs.SGLANG_MOE_OFFLOAD_KV_REGAIN.get() and _moe_offload_active()

    def _assert_expert_offload_installed(self: ModelRunner) -> None:
        """Ordering invariant: no FusedMoE layer may still be waiting to install.

        A layer that installs lazily on its first forward releases its expert
        VRAM after this profiling step, so the KV pool is sized against the
        pre-offload footprint and the reclaim is lost (and, worse, the resident
        buffers then have to fit alongside a pool that already claimed their
        space). Rank-local raise, matching the rank-local ValueErrors that the
        budget check below already uses.
        """
        model = getattr(self, "model", None)
        if model is None or getattr(self, "is_weightless_worker", False):
            # Weightless workers hold a meta model: no expert weights, nothing
            # to install, and the walk would only produce false positives.
            return
        try:
            from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
        except Exception:  # pragma: no cover - import-time env differences
            return
        pending = [
            getattr(m, "layer_id", "?")
            for m in model.modules()
            if isinstance(m, FusedMoE)
            and getattr(m, "_moe_offload_enabled", False)
            and getattr(m, "_expert_offload", None) is None
            and not getattr(m, "_expert_offload_install_failed", False)
        ]
        if pending:
            raise ValueError(
                f"MoE expert-offload is active but {len(pending)} FusedMoE "
                f"layer(s) {pending[:8]} have not installed their offload cache "
                f"before the KV pool is sized (rank {self.tp_rank}). Those "
                f"layers would release their expert VRAM only on the first "
                f"forward, i.e. AFTER this profiling step, so the KV pool would "
                f"be sized against the pre-offload footprint and then collide "
                f"with the resident buffers. The eager install in "
                f"ModelRunner.load_model() must run before pool sizing; set "
                f"SGLANG_MOE_OFFLOAD_KV_REGAIN=0 to profile without it."
            )

    def _expert_offload_reclaim_active(self: ModelRunner) -> bool:
        """Group-minimum verdict: did EVERY rank actually release expert VRAM?

        The reclaim changes the pool the whole TP group shares, so the decision
        to treat it as present has to be rank-uniform. Test rank-locally first,
        then reduce with MIN -- a single rank that released nothing (an MoE-free
        shard, a failed install falling back to fully-resident) makes the group
        fall back to the plain path rather than half the ranks synchronizing
        while the other half reads unsynchronized memory.
        """
        from sglang.srt.layers.moe.expert_offload import (
            expert_offload_release_totals,
        )

        local = 1 if expert_offload_release_totals().layers > 0 else 0
        if get_world_group().world_size > 1:
            verdict = torch.tensor(local, dtype=torch.int64)
            torch.distributed.all_reduce(
                verdict,
                op=torch.distributed.ReduceOp.MIN,
                group=get_world_group().cpu_group,
            )
            local = int(verdict.item())
        return bool(local)

    def _expert_offload_release_sync(self: ModelRunner) -> None:
        """Return every rank's freed expert blocks to the driver, then barrier.

        Order is the whole point: collect -> empty_cache -> barrier. Only after
        the barrier may any rank read mem_get_info(), because only then is every
        co-located sibling's release visible at driver level. Without this the
        profiled free memory depends on which rank happened to run first.
        """
        import gc

        gc.collect()
        empty_device_cache(torch.cuda)
        if get_world_group().world_size > 1:
            torch.distributed.barrier(group=get_world_group().cpu_group)

    def _gguf_dequant_scratch_gb(self: ModelRunner) -> float:
        """GGUF dequant scratch not yet allocated at profiling time, in GiB.

        A budget POST, not a margin: the GGUF loader already records the
        largest dequant target per rank while sizing its persistent
        workspace, and this is that number minus the buffer the rank already
        holds. Non-GGUF models never record one, so this returns 0.0 and the
        budget arithmetic is byte-identical to before (#257).
        """
        try:
            from sglang.srt.layers.quantization.gguf import (
                gguf_dequant_scratch_residual_bytes,
            )
        except Exception:
            return 0.0
        device_index = None
        if self.device == "cuda" and torch.cuda.is_available():
            device_index = torch.cuda.current_device()
        return gguf_dequant_scratch_residual_bytes(device_index) / (1 << 30)

    # ------------------------------------------------------------------
    # #260: budget vs. physical availability under co-residence.
    #
    # --rank-gpu-memory-mib (and the --rank-auto-reserve-mib derivation that
    # feeds it) is an ABSOLUTE per-rank allowance: NVML total minus the
    # reserve. A process that shares the card -- a second sglang instance, a
    # PD prefill server, anything -- does not reduce that number. It reduces
    # what can be handed out. Keeping the two apart is the whole fix: the
    # budget arithmetic below never reads free memory (only the before/after
    # DELTA, which a static neighbour cancels out of), and the free reading
    # is used exclusively to say, in its own words, when the plan cannot be
    # realized.
    # ------------------------------------------------------------------
    def _rank_vector_index(self: ModelRunner) -> int:
        """This process's index into the per-rank vectors (--rank-gpu-id,
        --rank-gpu-memory-mib).

        Those vectors are laid out in WORLD-rank order, pp_rank * tp_size +
        tp_rank (#201). Without a pipeline that is the tp_rank, which is what
        this used to read directly -- under one, reading tp_rank makes every
        stage pick stage 0's entry, so stage 1 checks its own card against
        stage 0's budget.
        """
        world_rank = getattr(self.server_args, "world_rank", None)
        if world_rank is None:  # pragma: no cover - stubbed server args
            return self.tp_rank
        return world_rank(self.pp_rank, self.tp_rank)

    def _device_occupancy_gb(self: ModelRunner, device_free_gb: float) -> Tuple:
        """(device total, bytes held outside this process) in GiB.

        ``torch.cuda.memory_reserved`` is per-PROCESS while ``mem_get_info``
        is device-wide, so the difference is everything this process does not
        hold: co-resident processes, sibling ranks, and every CUDA context
        including this rank's own (contexts are not allocator memory). Named
        that way in the messages -- it is a diagnosis aid, not an accounting
        post.
        """
        try:
            _free_b, total_b = torch.cuda.mem_get_info(self.gpu_id)
            own_b = torch.cuda.memory_reserved(self.gpu_id)
        except Exception:  # pragma: no cover - diagnostics only
            return (0.0, 0.0)
        total_gb = total_b / (1 << 30)
        return (total_gb, max(0.0, total_gb - device_free_gb - own_b / (1 << 30)))

    def _nvml_process_reach_gb(
        self: ModelRunner,
    ) -> Optional[Tuple[float, float, float]]:
        """``(held by THIS process, card free, card total)`` in GiB, or None.

        Read through the registry identity map, never by assuming the CUDA
        ordinal equals the NVML index -- they differ on this rig (#397).

        Why this exists (#631). The delta arithmetic the budget ledger uses
        measures memory consumed SINCE this stack's pre-load reading, and
        for the physical-availability check that is the wrong quantity: with
        one rank on the card it collapses algebraically to

            used_by_me + device_free
              = (pre_load_free - now_free) + now_free
              = pre_load_free

        so the check degenerates into "does the budget fit in whatever was
        free when THIS STACK started loading". A phase-flip instance builds
        three stacks in one process (PP weights, TP weights, MTP draft),
        each re-entering init_torch_distributed and taking its own pre-load
        reading with the previous stack's pool still resident. Measured on
        the 5090, boot 2026-08-09 12:27: stack 1 began at 30.46 GiB free,
        stack 2 at 23.40 GiB, stack 3 at 8.78 GiB. The check therefore got
        STRICTER as the budget grew -- a larger budget makes stack 1
        allocate more, which lowers stack 2's baseline, which refuses the
        budget that caused it. That feedback loop was read as a hard wall
        ("the 5090 cannot be filled").

        NVML's per-process reading has neither property: it is per PROCESS,
        not per stack, so an earlier stack's pool shows up as memory this
        rank HOLDS instead of silently vanishing from what it could reach,
        and it counts what torch cannot see (CUDA context, VMM arena
        handles, raw cudaMalloc workspaces).

        Returns None when NVML has nothing to say about this pid -- MPS
        attributes device memory to the server process, and a pid namespace
        can hide it -- so the caller keeps the old arithmetic as fallback
        rather than trusting a zero.
        """
        try:
            from sglang.srt.registry import nvml

            if not nvml.is_available():
                return None
            uuid = nvml.current_device_uuid()
            mem = nvml.memory_info_for_uuid(uuid)
            held_b = nvml.process_bytes_on_uuid(uuid).get(os.getpid())
        except Exception:  # pragma: no cover - diagnostics path
            return None
        if not held_b:
            return None
        gib = float(1 << 30)
        return (held_b / gib, mem.free_bytes / gib, mem.total_bytes / gib)

    @staticmethod
    def budget_physical_shortfall_gb(
        budget_gb: float,
        used_by_me_gb: float,
        device_free_gb: float,
        ranks_on_gpu: int,
    ) -> float:
        """GiB by which an absolute per-rank budget exceeds what the device
        can still hand this rank, 0.0 when the budget is reachable.

        Reachable = what the rank already holds + its share of the memory
        that is physically free. Co-located ranks split the free bytes
        evenly, the same convention ``note_post_capture_leftover`` uses.
        """
        reachable_gb = used_by_me_gb + device_free_gb / max(1, ranks_on_gpu)
        return max(0.0, budget_gb - reachable_gb)

    def _assert_budget_physically_available(
        self: ModelRunner,
        budget_mib: int,
        budget_gb: float,
        used_by_me_gb: float,
        device_free_gb: float,
    ) -> None:
        ranks_on_gpu = self._ranks_on_my_gpu()
        # Prefer the NVML per-PROCESS reading: stack-invariant, and it sees
        # the bytes torch does not (context, VMM arena, raw workspaces).
        # The delta arithmetic stays as the fallback for the paths where
        # NVML cannot attribute this pid (MPS, pid namespaces).
        nvml_reach = self._nvml_process_reach_gb()
        basis = "delta"
        if nvml_reach is not None:
            held_gb, nvml_free_gb, nvml_total_gb = nvml_reach
            used_by_me_gb, device_free_gb = held_gb, nvml_free_gb
            basis = "nvml"
        shortfall_gb = self.budget_physical_shortfall_gb(
            budget_gb, used_by_me_gb, device_free_gb, ranks_on_gpu
        )
        if basis == "nvml":
            # One line per rank per stack, on purpose: three of them with
            # three different "holds" is the readable proof that the stacks
            # are distinct and that the check is no longer per-stack.
            logger.info(
                "BUDGET-REACH[nvml] rank %d: budget %d MiB, this process holds "
                "%.2f GiB, card free %.2f GiB of %.2f GiB total, %d rank(s) "
                "co-located -> reachable %.2f GiB, shortfall %.2f GiB",
                self.tp_rank,
                budget_mib,
                used_by_me_gb,
                device_free_gb,
                nvml_total_gb,
                ranks_on_gpu,
                used_by_me_gb + device_free_gb / max(1, ranks_on_gpu),
                shortfall_gb,
            )
        if shortfall_gb <= 0:
            return
        rank_gpu_id = getattr(self.server_args, "rank_gpu_id", None) or []
        try:
            gpu = rank_gpu_id[self._rank_vector_index()]
        except (IndexError, TypeError):  # pragma: no cover - defensive
            gpu = self.gpu_id
        free_share_gb = device_free_gb / max(1, ranks_on_gpu)
        total_gb, outside_gb = self._device_occupancy_gb(device_free_gb)
        raise ValueError(
            f"The per-rank budget of {budget_mib} MiB ({budget_gb:.2f} GiB) "
            f"for rank {self.tp_rank} on GPU {gpu} is not physically "
            f"available: the rank holds {used_by_me_gb:.2f} GiB and "
            f"{free_share_gb:.2f} GiB of the device is free to it "
            f"({device_free_gb:.2f} GiB free across {ranks_on_gpu} co-located "
            f"rank(s), {total_gb:.2f} GiB total, {outside_gb:.2f} GiB held "
            f"outside this process -- co-resident processes, sibling ranks "
            f"and CUDA contexts), which is {shortfall_gb:.2f} GiB short of "
            f"the budget. [basis={basis}: 'nvml' reads what this PROCESS "
            f"holds on the card and NVML's free column, so a previously "
            f"built stack counts as held rather than missing; 'delta' is "
            f"the per-stack fallback and understates the hold of a "
            f"multi-stack process.] The budget is an ABSOLUTE allowance and is "
            f"deliberately NOT reduced by that occupancy; size it for what "
            f"the co-resident process leaves free (raise "
            f"--rank-auto-reserve-mib on this GPU, or lower "
            f"--rank-gpu-memory-mib), or stop that process."
        )

    @staticmethod
    def budget_exhausted_message(
        tp_rank: int,
        budget_mib: int,
        budget_gb: float,
        posts,
        rest_memory_gb: float,
        device_free_gb: float,
        occupancy: Tuple,
        ceiling: Optional[int] = None,
        reserve_note: Optional[str] = None,
    ) -> str:
        """The per-rank "no room for KV" message, itemized.

        Every post that consumed budget is named with its size, so the
        shortfall can be read off instead of reconstructed from a boot log.
        The driver-free line is part of it on purpose: a shortfall on a
        shared card invites the theory that the neighbour was charged twice,
        and the numbers that refute (or confirm) it belong in the message.

        ``reserve_note`` (#458) is appended when the budget was DERIVED by
        ``--rank-auto-reserve-mib auto`` rather than chosen. Without it the
        message's own remedy -- "lower --rank-auto-reserve-mib for this GPU by
        the same amount" -- is not followable: under ``auto`` there is no value
        to lower, and the next boot derives the identical reserve. See
        ``ServerArgs.derived_reserve_infeasible_note``.

        ``ceiling`` is ``--max-running-requests-ceiling`` (#287). When it is
        set, the mamba post scales linearly with it, so the message can name
        the ceiling this budget WOULD carry instead of leaving the operator
        to bisect it (#307). The demand-driven pool fits itself and never
        reaches this message; the pinned-size and fixed-fraction paths can,
        and there the number is the whole answer.
        """
        ledger = "; ".join(f"{name} {gb:.2f} GiB" for name, gb in posts if gb > 0.005)
        spent_gb = sum(gb for _name, gb in posts)
        short_mib = math.ceil(-rest_memory_gb * 1024)
        total_gb, outside_gb = occupancy
        ceiling_note = ""
        mamba_post_gb = mamba_post_total_gb(posts)
        if ceiling and mamba_post_gb > 0.005:
            # The post is (slots + admitted*D) * per_req + a constant reserve,
            # and both slot terms are proportional to the ceiling -- so the
            # affordable ceiling scales with the affordable share of the post.
            affordable_gb = max(mamba_post_gb + rest_memory_gb, 0.0)
            fits = int(ceiling * affordable_gb / mamba_post_gb)
            ceiling_note = (
                f" --max-running-requests-ceiling={ceiling} is the value the "
                f"state pools were dimensioned for and it is what the "
                f"{mamba_post_gb:.2f} GiB post above is spent on; this budget "
                f"carries a ceiling of about {max(fits, 1)}"
                + (
                    ". No ceiling fits this budget -- the shortfall survives "
                    "even at one request."
                    if fits < 1
                    else "."
                )
            )
        return (
            f"The per-rank budget leaves no GPU memory for the KV cache "
            f"under --rank-gpu-memory-mib on rank {tp_rank}: the "
            f"{budget_mib} MiB ({budget_gb:.2f} GiB) budget is spent on "
            f"{ledger} -- {spent_gb:.2f} GiB together, {short_mib} MiB more "
            f"than the budget, before a single KV token. Raise the budget by "
            f"at least that much plus the KV pool you want (lower "
            f"--rank-auto-reserve-mib for this GPU by the same amount), or "
            f"place fewer ranks on this GPU. If using speculative decoding, "
            f"draft weights are counted in the first post. The budget itself "
            f"was handed out in full: {device_free_gb:.2f} GiB of the device "
            f"is free at this point ({total_gb:.2f} GiB total, "
            f"{outside_gb:.2f} GiB held outside this process), and a "
            f"co-resident process does not reduce the budget."
            + ceiling_note
            + (reserve_note or "")
        )

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
        # #119: on the offload lane, settle the release before anyone measures.
        # Both steps are no-ops off the lane, so the default path is unchanged.
        offload_reclaim = False
        if self._expert_offload_lane_active():
            self._assert_expert_offload_installed()
            offload_reclaim = self._expert_offload_reclaim_active()
            if offload_reclaim:
                self._expert_offload_release_sync()
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
        # Raw driver-level free memory of the DEVICE, kept before the
        # accounting correction below: the physical-availability check (#260)
        # has to reason about the bytes the card actually has left, not about
        # this rank's accounting view of them.
        device_free_gb = available_gpu_memory
        # #602 corridor: the free-VRAM reading the floor is evaluated against
        # has to be taken exactly HERE -- after the barrier above (so every
        # co-located rank has finished loading its weights and the card's
        # occupancy is deterministic) and before any pool is allocated. It is
        # read from NVML's free column rather than derived from this rank's
        # accounting view or from total-minus-used: NVML holds a per-card
        # driver carve-out back from BOTH total and used, so total-minus-used
        # over-states free by exactly that amount, and the corridor law is
        # written on the free column.
        self._corridor_card_free_bytes = self._read_corridor_card_free_bytes()
        # Co-located ranks: mem_get_info() above charged this rank for its
        # sibling(s)' weights too. Add their own footprint back so only
        # this rank's weights are charged against its budget (no-op on the
        # default / even-TP / one-rank-per-GPU paths).
        available_gpu_memory += self._colocated_sibling_reserved_gb()

        # Ledger of what the per-rank budget was spent on, in the order the
        # posts are charged. Only read when a budget turns out to be too
        # small -- but built unconditionally, because the whole point is that
        # the failure message must not have to guess (#260).
        budget_posts: list[Tuple[str, float]] = []
        budget_gb = 0.0
        budget_mib = 0
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
            budget_mib = int(
                mib if isinstance(mib, (int, float)) else mib[self._rank_vector_index()]
            )
            budget_gb = budget_mib / 1024.0
            used_by_me_gb = pre_model_load_memory - available_gpu_memory
            rest_memory = budget_gb - used_by_me_gb
            budget_posts.append(("weights + runtime state", used_by_me_gb))
            rest_memory, _reserve_post = self._gapped_corridor_holdback(rest_memory)
            if _reserve_post is not None:
                budget_posts.append(_reserve_post)
            # #260: the budget is ABSOLUTE, so a co-resident process must
            # never shrink it -- but it does bound what this rank can
            # physically allocate. That bound gets its own check and its own
            # words, so a foreign-occupancy shortfall is never reported as
            # "your own weights exhausted the budget" (which is what sent a
            # co-existence bring-up hunting through its weight accounting
            # while the real posts were the mamba/activation reservations
            # below).
            self._assert_budget_physically_available(
                budget_mib, budget_gb, used_by_me_gb, device_free_gb
            )
            if self.mambaish_config is not None and self.post_capture_kv_active:
                mamba_precapture_gb = (
                    self.server_args.activation_reserve_mb(
                        get_device_memory_capacity(self.device)
                    )
                    / 1024
                )
                rest_memory -= mamba_precapture_gb
                budget_posts.append(("mamba pre-capture reserve", mamba_precapture_gb))
        else:
            slack_gb = pre_model_load_memory * (1 - self.mem_fraction_static)
            if self.mambaish_config is not None and self.post_capture_kv_active:
                # Mamba state is a fixed pre-capture allocation, so it can't ride the ~0 post-capture slack.
                slack_gb = max(
                    slack_gb,
                    self.server_args.activation_reserve_mb(
                        get_device_memory_capacity(self.device)
                    )
                    / 1024,
                )
            rest_memory = available_gpu_memory - slack_gb
            # #753: the corridor is owed on this branch too. A gapped boot that
            # lets the ledger size it from free VRAM -- rather than being
            # handed a per-rank MiB budget -- must still keep the user's free
            # column, or it simply spends the whole card instead of the whole
            # budget and reaches the same OOM by the other road.
            rest_memory, _auto_reserve_post = self._gapped_corridor_holdback(
                rest_memory
            )
            if _auto_reserve_post is not None:
                budget_posts.append(_auto_reserve_post)
        if self.mambaish_config is not None:
            before_mamba_gb = rest_memory
            rest_memory = self.handle_max_mamba_cache(rest_memory)
            budget_posts.extend(
                decompose_mamba_budget_post(
                    before_mamba_gb - rest_memory,
                    getattr(self, "_mamba_budget_components", {}) or {},
                )
            )

        # #257: GGUF dequant scratch. The GGUF path dequantizes a weight into
        # a scratch buffer before the large-M cuBLAS GEMM, and targets above
        # the workspace cap fresh-allocate their FULL size at forward time --
        # including in the EAGER pre-capture warmup forwards, which run after
        # this profiling. Those bytes are neither resident nor covered by the
        # slack above, so a KV pool sized without them is oversized by
        # exactly the largest dequant this rank performs (2.37 GiB for the
        # Qwen3.6-27B Q3_K_M lm_head at TP=1 -> OOM in the decode-graph
        # warmup at --mem-fraction-static 0.90). Charge the measured demand.
        # Zero for every non-GGUF model, so other quants are unchanged.
        gguf_scratch_gb = self._gguf_dequant_scratch_gb()
        if gguf_scratch_gb:
            logger.info(
                "GGUF dequant scratch (rank %d): reserving %.2f GiB of the "
                "KV budget for the largest over-cap dequant target this rank "
                "still has to allocate at forward time (%.2f -> %.2f GiB).",
                self.tp_rank,
                gguf_scratch_gb,
                rest_memory,
                rest_memory - gguf_scratch_gb,
            )
            rest_memory -= gguf_scratch_gb
        budget_posts.append(("GGUF dequant scratch", gguf_scratch_gb))

        # Measured KV-budget correction (two-boot convergence): replace the
        # blind part of the slack heuristic with the PREVIOUS boot's measured
        # leftover (see note_post_capture_leftover). Applied on both budget
        # paths; 0 unless SGLANG_MEASURED_KV_BUDGET is set and a matching
        # fingerprinted measurement exists.
        correction_b = self._measured_kv_budget_correction_bytes()
        correction_gb = correction_b / (1 << 30)
        # #188: announce the cross-boot dependency on EVERY boot with the
        # mode on -- cold records are exactly the case that silently
        # disagrees with its own repeat. Warning level on purpose: this is
        # the line that tells a benchmark harness its capacity axis is not
        # reproducible.
        if envs.SGLANG_MEASURED_KV_BUDGET.get() and self.tp_rank == 0:
            logger.warning(
                "%s",
                self.measured_budget_provenance_note(
                    self._measured_kv_budget_provenance,
                    self._measured_kv_budget_cache_path(),
                    self._measured_kv_budget_ts,
                    correction_b,
                ),
            )
        if correction_gb:
            logger.info(
                "Measured KV-budget correction (rank %d): %+.2f GiB on top "
                "of the heuristic budget %.2f GiB (previous boot's measured "
                "leftover minus safety; SGLANG_MEASURED_KV_BUDGET).",
                self.tp_rank,
                correction_gb,
                rest_memory,
            )
            rest_memory += correction_gb

        # Loaded weights (target + draft) can exceed the static budget
        if rest_memory <= 0:
            # #257: say so when the GGUF scratch post is what tipped it. The
            # alternative is the boot that motivated this change: it looked
            # affordable here and OOMed minutes later inside ggml_dequantize.
            gguf_note = (
                f" This rank also reserves {gguf_scratch_gb:.2f} GiB of GGUF "
                f"dequant scratch (the largest over-cap dequant target it "
                f"must allocate at forward time); that reservation is part "
                f"of the shortfall."
                if gguf_scratch_gb
                else ""
            )
            # The suggestion has to clear that post too, otherwise it names a
            # fraction that fails again for the same reason (gguf_scratch_gb
            # is 0.0 off the GGUF path -> unchanged formula).
            minimum_mem_fraction_static = (
                1 - (available_gpu_memory - gguf_scratch_gb) / pre_model_load_memory
            )
            suggested_mem_fraction_static = (
                math.ceil(minimum_mem_fraction_static * 1000) / 1000
            )
            if uneven_memory:
                # In this mode --mem-fraction-static is rejected up front;
                # phrase the fix in the budget's own unit (MiB), and itemize
                # (#260). "The rank already uses X for weights, which
                # exhausts the budget" was true of only the first post: on
                # the boot that motivated this, weights were 4.32 of a 6.65
                # GiB budget and the other 2.50 GiB went to the mamba state
                # pool, the speculative intermediate state and the prefill
                # activation reserve -- none of them named. The message also
                # states the driver-free situation, because the shortfall
                # was read as co-residence double-counting when the budget
                # had in fact been handed out in full.
                raise ValueError(
                    self.budget_exhausted_message(
                        tp_rank=self.tp_rank,
                        budget_mib=budget_mib,
                        budget_gb=budget_gb,
                        posts=budget_posts,
                        rest_memory_gb=rest_memory,
                        device_free_gb=device_free_gb,
                        occupancy=self._device_occupancy_gb(device_free_gb),
                        ceiling=self.server_args.max_running_requests_ceiling,
                        # World-rank order, like every other read of the
                        # per-rank vectors (#201) -- tp_rank alone makes every
                        # PP stage name stage 0's card.
                        reserve_note=self.server_args.derived_reserve_infeasible_note(
                            self._rank_vector_index(), math.ceil(-rest_memory * 1024)
                        ),
                    )
                )
            raise ValueError(
                f"Loaded weights leave no GPU memory for the KV cache under "
                f"--mem-fraction-static={self.mem_fraction_static}. "
                f"Raise --mem-fraction-static above "
                f"{suggested_mem_fraction_static:.3f} "
                f"(minimum viable = 1 - available/pre = "
                f"{minimum_mem_fraction_static:.4f}). If using speculative "
                f"decoding, draft weights are now counted.{gguf_note}"
            )

        if offload_reclaim:
            # #119: state the reclaim in the log. The bytes are already inside
            # rest_memory (the free-memory reading above saw them); this line
            # exists so the win is verifiable from a boot log instead of being
            # inferred, and so a regression that loses it is visible as a drop
            # here rather than as an unexplained smaller KV pool.
            from sglang.srt.layers.moe.expert_offload import (
                expert_offload_release_totals,
            )

            released = expert_offload_release_totals()
            logger.info(
                "[offload-kv-regain] rank %d: expert offload released %.2f GiB "
                "of weight VRAM across %d MoE layer(s) (%.2f GiB moved to the "
                "pinned host pool); that VRAM is part of the %.2f GiB KV budget "
                "profiled here.",
                self.tp_rank,
                released.device_bytes / (1 << 30),
                released.layers,
                released.host_bytes / (1 << 30),
                rest_memory,
            )

        # Component-balance checkpoint "post-weights" (weights + CUDA context
        # are resident; pools/graphs are not): exact allocator numbers for
        # the balance log in note_post_capture_leftover.
        self._mem_ckpt_post_weights = (
            torch.cuda.memory_allocated(),
            torch.cuda.memory_reserved(),
        )

        # #704: EMIT the budget decomposition on the SUCCESS path too.
        #
        # Until now ``budget_posts`` was built on every boot and handed to
        # ``budget_exhausted_message`` only when the budget RAN OUT -- so the
        # sizer named every term of its own arithmetic exactly when it failed,
        # and discarded the naming when it worked. The planner therefore had no
        # instrument for the reserve it must not double-count, and re-deriving
        # it from config missed the measured boot by +20 %, -3.8 % and -12 % on
        # three independent attempts (NOTE_704_retro_prediction_terms.md).
        #
        # This line is the instrument that closes that gap: the pool solve
        # CONSUMES these posts instead of recomputing them, which is the same
        # discipline the #676 arming floor already follows. Two numbers that
        # must agree and are computed twice is a shape this corpus has paid for
        # repeatedly; here the second computation lives in a different process
        # and could not even be compared.
        # World rank, not tp_rank: under the flip's primary topology (tp=1,
        # pp=N) every PP stage has tp_rank 0, so labelling with tp_rank names
        # stage 0's card three times -- the #201 defect the budget-exhausted
        # path already guards against a few lines below. Confirmed on metal:
        # the first boot carrying this line emitted "[rank 0]" for PP0, PP1 and
        # PP2 alike. _rank_vector_index() is the existing accessor and falls
        # back to tp_rank when server_args is stubbed.
        logger.info(
            "[world_rank %d] KV budget posts (GiB): %s | rest=%.3f",
            self._rank_vector_index(),
            ", ".join(f"{name}={gb:.3f}" for name, gb in budget_posts),
            rest_memory,
        )

        return int(rest_memory * (1 << 30))  # return in bytes

    # ------------------------------------------------------------------
    # Measured KV-budget correction (two-boot convergence).
    #
    # The heuristic KV budget (mem-fraction slack) is decided BEFORE the
    # post-pool consumers exist (CUDA graphs, attention workspaces, adaptive
    # rung tags, draft-solo graph sets, ...), so it can only GUESS their
    # cost and systematically over- or under-reserves (measured 2026-07-22:
    # 5.5-6.5 GiB idle per shadow card on the cross-auto tp3 boot). Instead
    # of guessing better constants: after load + pools + capture, every rank
    # MEASURES its actual leftover (driver free memory of its device, split
    # among co-located ranks) and persists ``leftover - safety`` in a
    # config-fingerprinted cache next to the hardware profile; the next boot
    # of the SAME configuration adds the stored correction to its heuristic
    # budget. Fixed point: the leftover converges to the configured safety
    # margin (SGLANG_MEASURED_KV_BUDGET_SAFETY_MIB, default 400 MiB) within
    # one or two boots; overshoot self-corrects with a negative delta. The
    # measurement is offload-aware by construction: paused adaptive rung
    # tags hold no physical pages at the measurement point. Every value is
    # read from the driver or the cache — the only ASSUMED number left is
    # the safety margin itself. Opt-in via SGLANG_MEASURED_KV_BUDGET (the
    # default path stays byte-identical).
    # ------------------------------------------------------------------
    def _measured_kv_budget_cache_path(self: ModelRunner) -> str:
        # Fingerprint shared with the pre-boot weight planner (which READS
        # this registry to place weights): single source of truth in
        # uneven_perf so writer and reader can never drift apart.
        from sglang.srt.uneven_perf import measured_kv_budget_cache_path

        path = measured_kv_budget_cache_path(self.server_args)
        if self.pp_size > 1:
            # #201 slice 3 / #188 family: without this suffix BOTH stages'
            # tp_rank-0 processes write the SAME record (the fingerprint is
            # parse-time and knows no pp_rank), so the next boot sizes one
            # stage from the other stage's measured leftover -- the exact
            # "previous boot's leftover" trap, in cross-stage form. One
            # record per stage; the launcher-side weight planner reads the
            # UNSUFFIXED path and correctly stays cold under a pipeline.
            root, ext = os.path.splitext(path)
            path = f"{root}-stage{self.pp_rank}{ext}"
        return path

    def _measured_safety_mib(self: ModelRunner) -> int:
        """This rank's configured safety margin (MiB). Scalar env value, or a
        comma list with one value per TP rank (roles differ: the draft-solo
        host carries the dual-prefill / draft-append serving transients,
        which scale with prompt length — measured 2026-07-22: 10k prefill
        needs ~1 GiB, 50k ~2-3.5 GiB on the host, while shadow ranks served
        everything with ~1.6 GiB)."""
        raw_safety = str(envs.SGLANG_MEASURED_KV_BUDGET_SAFETY_MIB.get())
        parts = [p for p in raw_safety.split(",") if p.strip()]
        if len(parts) > 1:
            try:
                return int(parts[self.tp_rank])
            except (IndexError, ValueError):
                return int(parts[0])
        return int(parts[0]) if parts else 400

    def _resolved_mlp_vector(self: ModelRunner) -> list:
        """The weight (MLP family) vector this boot actually shards with —
        the registry key that decides whether a stored budget correction is
        transferable. Falls back to the base plan when no override is set."""
        sa = self.server_args
        vec = getattr(sa, "rank_mlp_ratio", None)
        if not isinstance(vec, list):
            base = getattr(sa, "rank_tp_ratio", None)
            vec = list(base) if isinstance(base, list) else [1] * sa.tp_size
        return [int(v) for v in vec]

    # ------------------------------------------------------------------
    # #188: this boot's KV capacity is a function of an ON-DISK record from
    # a PREVIOUS boot, so two identical commands legitimately size
    # differently (measured: max_total_num_tokens 380289 vs 447173). That is
    # the design, not a defect -- but it was SILENT, which turned it into a
    # harness trap: any capacity or byte comparison across trees compared
    # the harness to itself. Every read now records where the number came
    # from, and the profiler announces it on every boot (cold included).
    # ------------------------------------------------------------------
    #: Provenance of the last correction read. One of: "cold" (no record
    #: yet -- this boot sizes from the heuristic and the NEXT one will not),
    #: "malformed", "vector-reset", "safety-reset", "stored".
    _measured_kv_budget_provenance: str = "cold"
    _measured_kv_budget_ts: Optional[str] = None

    @staticmethod
    def measured_budget_provenance_note(
        provenance: str,
        path: str,
        ts: Optional[str],
        correction_b: int,
    ) -> str:
        """The boot-log line for the cross-boot dependency (see above).

        Emitted on EVERY boot with the mode on, not only when a correction
        exists: the cold case is exactly the one that silently differs from
        its own repeat.
        """
        gib = 1 << 30
        deterministic = (
            "For a reproducible capacity (A/B runs, byte gates, capacity "
            "comparisons across trees) pin --max-total-tokens, or unset "
            "SGLANG_MEASURED_KV_BUDGET."
        )
        if provenance == "stored":
            return (
                f"Measured KV-budget: this boot's KV capacity is sized with "
                f"a {correction_b / gib:+.2f} GiB correction MEASURED BY A "
                f"PREVIOUS BOOT (record {path}, written {ts}). Identical "
                f"commands against a different record state will size "
                f"differently. {deterministic}"
            )
        if provenance == "cold":
            return (
                f"Measured KV-budget: no stored measurement for this "
                f"configuration ({path}), so this boot sizes from the "
                f"heuristic budget alone and the next identical boot will "
                f"size differently once this boot's measurement lands. "
                f"{deterministic}"
            )
        return (
            f"Measured KV-budget: the stored measurement in {path} was "
            f"discarded ({provenance}), so this boot sizes from the "
            f"heuristic budget alone and re-measures; the next identical "
            f"boot will size differently. {deterministic}"
        )

    def _measured_kv_budget_correction_bytes(self: ModelRunner) -> int:
        self._measured_kv_budget_provenance = "cold"
        self._measured_kv_budget_ts = None
        if not envs.SGLANG_MEASURED_KV_BUDGET.get():
            return 0
        import json

        path = self._measured_kv_budget_cache_path()
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, ValueError):
            return 0
        vec = data.get("correction_bytes")
        if not isinstance(vec, list) or len(vec) != self.server_args.tp_size:
            logger.warning("Measured KV-budget cache %s is malformed; ignoring.", path)
            self._measured_kv_budget_provenance = "malformed"
            return 0
        # Corrections are VECTOR-SPECIFIC: they encode the previous boot's
        # leftover under that boot's weight distribution. The registry
        # fingerprint deliberately survives weight-vector changes (the
        # planner needs the component balance across vectors), so guard
        # here instead: a changed vector invalidates the correction (a
        # 6,1,1-measured +3.4 GiB applied to a 2,1,1 boot over-booked the
        # shadow pools ~2.5 GiB past the card and OOMed the DFLASH capture,
        # measured 2026-07-22). Components stay valid; the correction
        # re-converges from zero under the new vector (one extra boot).
        stored_vec = data.get("mlp_vector")
        current_vec = self._resolved_mlp_vector()
        if stored_vec != current_vec:
            if self.tp_rank == 0:
                logger.info(
                    "Measured KV-budget: stored correction was measured "
                    "under weight vector %s, this boot shards %s — "
                    "correction reset to 0 (re-measuring under the new "
                    "vector).",
                    stored_vec,
                    current_vec,
                )
            self._measured_kv_budget_provenance = "vector-reset"
            return 0
        # Same epoch rule for the SAFETY margin: the correction encodes
        # "leftover minus the safety of that epoch", and the consumption
        # gate deliberately freezes unconsumed growth — so after a safety
        # change the correction can never re-target the new required-free
        # level (measured 2026-07-22: shadows stuck at the old 1350 MiB
        # target + granularity slack, flapping over the corridor bound).
        # A changed per-rank safety therefore restarts the correction from
        # zero, exactly like a changed weight vector.
        stored_safety = None
        comps = data.get("components")
        if isinstance(comps, list) and len(comps) > self.tp_rank:
            try:
                stored_safety = int(comps[self.tp_rank].get("safety_mib"))
            except (TypeError, ValueError):
                stored_safety = None
        current_safety = self._measured_safety_mib()
        if stored_safety is not None and stored_safety != current_safety:
            logger.info(
                "Measured KV-budget (rank %d): stored correction was "
                "measured under safety %d MiB, this boot uses %d MiB — "
                "correction reset to 0 (re-measuring under the new safety).",
                self.tp_rank,
                stored_safety,
                current_safety,
            )
            self._measured_kv_budget_provenance = "safety-reset"
            return 0
        self._measured_kv_budget_provenance = "stored"
        ts = data.get("ts")
        self._measured_kv_budget_ts = str(ts) if ts is not None else None
        return int(vec[self.tp_rank])

    def _ranks_on_my_gpu(self: ModelRunner) -> int:
        """How many TP ranks share this rank's physical device (>= 1)."""
        rank_gpu_id = getattr(self.server_args, "rank_gpu_id", None)
        if not rank_gpu_id:
            return 1
        try:
            return max(
                1,
                list(rank_gpu_id).count(rank_gpu_id[self._rank_vector_index()]),
            )
        except (IndexError, TypeError):
            return 1

    @staticmethod
    def correction_growth_frozen(
        delta_b: int,
        prev_b: int,
        prev_leftover_b,
        free_b: int,
        component_shift: bool,
    ) -> bool:
        """Pure consumption-gate verdict (unit-tested; see the call site).

        Growth (positive delta on top of a positive stored correction) is
        frozen when the previous boot's leftover did NOT shrink -- a rank
        that is not the global binder must not accumulate fantasy budget.
        EXCEPTION (T156-D): ``component_shift`` -- a measured component of
        THIS rank changed materially between boots (e.g. the DFLASH solo
        pool releasing 4.6 GiB), so the leftover baseline moved for a
        structural reason the "did it shrink" test cannot see; the gate
        opens for this boot and the next boot's consumption re-arms it
        (measured 2026-07-22: rank 0 stuck at +291 MiB with 8.3 GiB
        leftover after the pool shrink, while a fresh registry consumed
        the same bytes immediately). Negative deltas (overshoot rollback)
        always apply."""
        return (
            delta_b > 0
            and prev_b > 0
            and not component_shift
            and prev_leftover_b is not None
            and free_b >= prev_leftover_b - (128 << 20)
        )

    @staticmethod
    def measure_free_after_own_cleanup(
        mem_get_info, empty_cache, gpu_id
    ) -> Tuple[int, int, int]:
        """Driver free/total bytes measured AFTER this process handed its
        own allocator cache back. Returns (free, total, released).

        #188: the leftover used to be a single bare ``mem_get_info``, which
        excludes whatever the caching allocator happened to be holding
        reserved-but-free at the post-capture point. That number is
        run-to-run variable (capture order, JIT builds that land inside
        capture, warmup transients), so the PERSISTED correction inherited
        that variance and two identical commands sized differently. After
        ``empty_cache`` the reading is a stable quantity: bytes physically
        resident on the device.

        The residual allocator posts (fragmentation, boot-time transient
        peak) are read by the caller BEFORE this runs, so the component
        balance still reports what the allocator actually held -- the
        cleanup normalises the MEASUREMENT, it does not hide the post.
        """
        free_before, _total_before = mem_get_info(gpu_id)
        empty_cache()
        free_b, total_b = mem_get_info(gpu_id)
        return int(free_b), int(total_b), max(0, int(free_b) - int(free_before))

    @staticmethod
    def unaccounted_used_bytes(
        total_b: int,
        free_b: int,
        ranks_on_gpu: int,
        reserved_b: int,
        ctx_allowance_b: int,
    ) -> int:
        """Bytes the driver reports used on THIS rank's share of the device
        that this rank cannot account for.

        ``total_b`` / ``free_b`` are DEVICE-wide driver numbers; both are
        split by ``ranks_on_gpu`` here, exactly like the leftover itself, so
        a co-located sibling's reservation is not mistaken for a foreign
        one. ``reserved_b`` is this process's own allocator reservation.

        After ``measure_free_after_own_cleanup`` every byte we hold is in
        ``reserved_b``, and the CUDA context costs at most
        ``ctx_allowance_b``. Anything beyond that is a FOREIGN consumer --
        on a shared box, typically a server from the previous boot that has
        not exited. It silently shrinks the measured leftover and therefore
        the persisted correction, which is precisely the "the number came
        from the previous boot" failure mode (#188).
        """
        ranks = max(1, int(ranks_on_gpu))
        used_share_b = max(0, int(total_b) // ranks - int(free_b) // ranks)
        return max(0, used_share_b - int(reserved_b) - int(ctx_allowance_b))

    @staticmethod
    def foreign_residue_warning(
        tp_rank: int, unaccounted_b: int, free_b: int
    ) -> Optional[str]:
        """The LOUD line for a detected foreign consumer, or None."""
        if unaccounted_b <= 0:
            return None
        gib = 1 << 30
        return (
            f"Measured KV-budget (rank {tp_rank}): "
            f"{unaccounted_b / gib:.2f} GiB of this rank's device share is "
            f"used by something this process did not allocate (own allocator "
            f"cache already released). A leftover server from a previous "
            f"boot is the usual cause on a shared box. The measured leftover "
            f"({free_b / gib:.2f} GiB) is therefore too SMALL, and the "
            f"correction persisted for the next boot will under-book the KV "
            f"pool. Kill the foreign process and re-measure, or raise "
            f"SGLANG_MEASURED_KV_BUDGET_CTX_ALLOWANCE_MIB if this device "
            f"legitimately hosts a co-resident consumer."
        )

    def note_post_capture_leftover(
        self: ModelRunner, draft_solo_pool_bytes: int = 0
    ) -> None:
        """Measure this boot's actual per-rank leftover and persist the
        accumulated budget correction for the next boot (see the section
        comment above). Must be called on every rank (collective gather);
        rank 0 writes the cache.

        ``draft_solo_pool_bytes``: measured size of a solo-resident draft KV
        pool on THIS rank (the cross gate's DFLASH pool on rank 0; 0
        elsewhere), passed in by the scheduler which owns the draft worker.
        Recorded as its own registry post — it scales with the global token
        count, so the weight planner must model it as the per-token solo
        cell, NOT as fixed graph residency."""
        if not envs.SGLANG_MEASURED_KV_BUDGET.get():
            return
        import json
        import time as _time

        torch.cuda.synchronize()

        safety_mib = self._measured_safety_mib()
        safety_b = safety_mib << 20

        # Component balance (rank-local, exact allocator/driver numbers):
        # every future misallocation should name the component that moved,
        # not just "card is not full". Weights double-checked against the
        # parameter/buffer tensor bytes.
        #
        # Read the ALLOCATOR posts first: they describe what this process was
        # holding at the post-capture point, and the #188 cleanup below
        # deliberately hands part of it back.
        alloc_now = torch.cuda.memory_allocated()
        reserved_now = torch.cuda.memory_reserved()
        ckpt_w = getattr(self, "_mem_ckpt_post_weights", (0, 0))
        ckpt_p = getattr(self, "_mem_ckpt_post_pools", (0, 0))
        try:
            param_bytes = sum(
                t.nbytes
                for t in list(self.model.parameters()) + list(self.model.buffers())
                if t.device.type == "cuda"
            )
        except Exception:  # balance log must never break boot
            param_bytes = 0
        gib = 1 << 30
        # Honest residual posts (never silently folded into the safety
        # margin): allocator fragmentation (reserved - allocated) and the
        # transient peak beyond the steady state (allocator peak stats over
        # this boot's own forwards: capture warmups + profiling). Serving
        # transients of REAL requests are NOT covered by this boot-time
        # peak — measured 2026-07-22: a rank left with 705 MiB free OOMed on
        # the first request while 961 MiB survived, so the configured safety
        # must exceed the serving transient (report, not hidden).
        stats = torch.cuda.memory_stats()
        peak_alloc = int(stats.get("allocated_bytes.all.peak", alloc_now))
        frag_b = max(0, reserved_now - alloc_now)
        transient_b = max(0, peak_alloc - alloc_now)

        # #188: measure the leftover only AFTER releasing this process's own
        # allocator cache, so the persisted correction encodes physically
        # resident bytes and not run-to-run allocator slack. The posts above
        # are already captured, so the balance still reports the pre-cleanup
        # truth. This changes the measured number (and therefore the next
        # boot's pool) on the SGLANG_MEASURED_KV_BUDGET path only -- the
        # default path never enters this function.
        free_b, total_b, released_b = self.measure_free_after_own_cleanup(
            mem_get_info=torch.cuda.mem_get_info,
            empty_cache=torch.cuda.empty_cache,
            gpu_id=self.gpu_id,
        )
        if released_b:
            logger.info(
                "Measured KV-budget (rank %d): released %.2f GiB of own "
                "allocator cache before measuring; leftover is measured "
                "against physically resident bytes (#188).",
                self.tp_rank,
                released_b / gib,
            )
        logger.info(
            "Measured KV-budget balance (rank %d): post-weights alloc "
            "%.2f/res %.2f GiB (param+buffer tensors %.2f GiB), pools "
            "+%.2f GiB, graphs/workspaces +%.2f GiB, now alloc %.2f/res "
            "%.2f GiB; residual posts: fragmentation %.2f GiB, boot-time "
            "transient peak %.2f GiB; driver free %.2f GiB.",
            self.tp_rank,
            ckpt_w[0] / gib,
            ckpt_w[1] / gib,
            param_bytes / gib,
            (ckpt_p[0] - ckpt_w[0]) / gib,
            (alloc_now - ckpt_p[0]) / gib,
            alloc_now / gib,
            reserved_now / gib,
            frag_b / gib,
            transient_b / gib,
            free_b / gib,
        )
        # Co-located ranks share the device's free memory: each may claim
        # only its share, or the next boot double-books the same bytes.
        ranks_on_gpu = self._ranks_on_my_gpu()

        # #188: harden the reading against FOREIGN residue before it is
        # persisted. Own allocator cache is already released above, so every
        # used byte beyond `reserved_now` + a CUDA-context allowance belongs
        # to somebody else -- on a shared box, typically a server from the
        # previous boot that never exited. Report LOUDLY instead of silently
        # persisting a too-small correction that then under-books the next
        # boot's KV pool.
        # `reserved_now` is the PRE-cleanup reservation while `free_b` is
        # post-cleanup: exact for one rank per card, conservative (fewer
        # false positives) when ranks are co-located. Deliberate -- a loud
        # line must not cry wolf.
        ctx_allowance_b = (
            int(envs.SGLANG_MEASURED_KV_BUDGET_CTX_ALLOWANCE_MIB.get()) << 20
        )
        foreign_b = self.unaccounted_used_bytes(
            total_b=total_b,
            free_b=free_b,
            ranks_on_gpu=ranks_on_gpu,
            reserved_b=reserved_now,
            ctx_allowance_b=ctx_allowance_b,
        )
        foreign_msg = self.foreign_residue_warning(
            self.tp_rank, foreign_b, int(free_b) // ranks_on_gpu
        )
        if foreign_msg is not None:
            logger.warning("%s", foreign_msg)

        free_b = int(free_b) // ranks_on_gpu

        # Explicit measured post: the largest PAUSED adaptive/rung graph tag.
        # ensure_active must be able to map it back at any switch, so that
        # many bytes must stay physically free on top of the safety margin
        # (measured 2026-07-22: cu_mem_create OOM inside torch_memory_saver
        # when a switch mapped the DFLASH tag during a long-prompt request).
        # Alongside the max, record EVERY rung tag as its own registry post,
        # virtual vs physical separated: ``noted_tensor_bytes`` is the
        # virtual footprint of the tag's noted tensors (address reservation,
        # survives a pause), ``paused_physical_bytes`` the measured physical
        # pages the tag frees when paused (noted tensors + its private
        # capture pool). The weight planner and any future balance audit
        # read these instead of re-deriving them from logs.
        max_tag_b = 0
        rung_tags: dict = {}
        try:
            from sglang.srt.speculative.adaptive_graph_memory import (
                get_active_manager,
            )

            mgr = get_active_manager()
            if mgr is not None:
                for key, rec in getattr(mgr, "_states", {}).items():
                    paused = int(getattr(rec, "paused_bytes", 0) or 0)
                    try:
                        noted = int(rec.nbytes)
                    except Exception:
                        noted = 0
                    rung_tags[str(getattr(rec, "tag", key))] = {
                        "noted_tensor_bytes": noted,
                        "paused_physical_bytes": paused,
                    }
                    max_tag_b = max(max_tag_b, paused)
        except Exception:  # never break boot for the balance
            max_tag_b = 0

        # Component balance as machine-readable registry posts (per rank).
        # KV-vs-mamba split inside the pool post: the KV pool reports its own
        # tensor bytes; the remainder of the pool checkpoint delta is the
        # mamba/SSM state pool plus small aux pools (req_to_token etc.),
        # labeled as such rather than silently folded together. The solo
        # draft pool (passed in) is carved OUT of the graphs/workspaces
        # delta: it scales with the global token count (a per-token cell),
        # everything else in that delta is token-independent residency.
        kv_pool_b = 0
        try:
            v = self.token_to_kv_pool.get_kv_size_bytes()
            kv_pool_b = int(sum(v)) if isinstance(v, (tuple, list)) else int(v)
        except Exception:
            kv_pool_b = 0
        pools_b = max(0, ckpt_p[0] - ckpt_w[0])
        graphs_ws_b = max(0, alloc_now - ckpt_p[0])
        draft_pool_b = max(0, int(draft_solo_pool_bytes or 0))
        # CUDA context + any co-resident consumer on this device: what the
        # driver reports used beyond this process's allocator reservation.
        # Split evenly among co-located ranks (assumption, exact for the
        # 1-rank-per-card case).
        # (total/ranks - free/ranks) is this rank's share of the device's
        # used bytes; reserved_now is per-process (this rank's allocator
        # reservation alone), so the difference is context + non-allocator
        # residue attributable to this rank. Display-only: paused
        # memory-saver tags inflate ``reserved`` (allocated-but-unmapped),
        # so this can clamp to 0 — the planner consumes the driver-derived
        # residual post below instead.
        used_share_b = max(0, int(total_b) // ranks_on_gpu - free_b)
        ctx_overhead_b = max(0, used_share_b - int(reserved_now))
        # THE planner-facing catch-all: every physically resident byte on
        # this rank that is neither weights, nor a sized pool, nor the solo
        # draft pool — CUDA context, NCCL buffers, CUDA graphs, attention
        # workspaces, allocator fragmentation, memory-saver pools. Derived
        # purely from the DRIVER's used bytes, so paused rung tags (virtual
        # reservation, no physical pages) are correctly excluded; their
        # remap requirement is carried by required_free_bytes instead.
        residual_residency_b = max(
            0,
            used_share_b
            - int(ckpt_w[0])
            - max(0, ckpt_p[0] - ckpt_w[0])
            - max(0, int(draft_solo_pool_bytes or 0)),
        )
        my_component = {
            # Which physical card this rank's balance was measured on (AUDIT
            # #331). ``device_total_bytes`` below is a property of THAT card;
            # on a mixed rig, replaying a stored balance against a different
            # card sizes the KV pool against the wrong total, which is the
            # #336 defect in cached form. The rank position alone cannot say
            # which card it was -- CUDA enumeration is not stable across
            # boots -- so the uuid is written down and checked on read.
            "card_uuid": _current_card_uuid(),
            "device_total_bytes": int(total_b),
            "ranks_on_gpu": int(ranks_on_gpu),
            "residual_residency_bytes": int(residual_residency_b),
            "ctx_overhead_bytes": int(ctx_overhead_b),
            "weights_alloc_bytes": int(ckpt_w[0]),
            "weights_param_bytes": int(param_bytes),
            "pools_bytes": int(pools_b),
            "kv_pool_bytes": int(kv_pool_b),
            "kv_pool_tokens": int(getattr(self.token_to_kv_pool, "size", 0) or 0),
            "mamba_aux_pool_bytes": int(max(0, pools_b - kv_pool_b)),
            "graphs_ws_bytes": int(graphs_ws_b),
            "draft_solo_pool_bytes": int(draft_pool_b),
            "graphs_ws_excl_draft_pool_bytes": int(max(0, graphs_ws_b - draft_pool_b)),
            "rung_tags": rung_tags,
            "frag_bytes": int(frag_b),
            "boot_transient_bytes": int(transient_b),
            "safety_mib": int(safety_mib),
            "max_paused_tag_bytes": int(max_tag_b),
            "required_free_bytes": int(safety_b + max_tag_b),
            "free_bytes_at_measure": int(free_b),
            "max_total_num_tokens": int(getattr(self, "max_total_num_tokens", 0) or 0),
        }
        logger.info(
            "Measured KV-budget components (rank %d): kv-pool %.2f GiB, "
            "mamba/aux pools %.2f GiB, draft-solo pool %.2f GiB, graphs/ws "
            "excl draft pool %.2f GiB, ctx overhead %.2f GiB, required free "
            "%.2f GiB (safety %d MiB + max paused tag %.0f MiB); rung tags: "
            "%s.",
            self.tp_rank,
            kv_pool_b / gib,
            max(0, pools_b - kv_pool_b) / gib,
            draft_pool_b / gib,
            max(0, graphs_ws_b - draft_pool_b) / gib,
            ctx_overhead_b / gib,
            (safety_b + max_tag_b) / gib,
            safety_mib,
            max_tag_b / (1 << 20),
            {
                k: f"{v['paused_physical_bytes'] / (1 << 20):.0f}MiB-phys/"
                f"{v['noted_tensor_bytes'] / (1 << 20):.0f}MiB-noted"
                for k, v in rung_tags.items()
            }
            or "none",
        )

        delta_b = free_b - safety_b - max_tag_b
        prev_b = self._measured_kv_budget_correction_bytes()
        # Consumption-gated growth: only grow the correction if the PREVIOUS
        # growth was actually consumed (leftover shrank). A rank that is not
        # the global binder (its pool is capped by another rank or by the
        # physical hybrid ceiling) keeps a stable correction instead of
        # accumulating fantasy budget boot over boot; negative deltas
        # (overshoot rollback) always apply.
        prev_leftover_b = None
        prev_draft_pool_b = None
        try:
            import json as _json

            with open(self._measured_kv_budget_cache_path()) as f:
                _prev = _json.load(f)
            vec = _prev.get("leftover_mib_at_measure")
            if isinstance(vec, list) and len(vec) > self.tp_rank:
                prev_leftover_b = int(vec[self.tp_rank]) << 20
            comps_prev = _prev.get("components")
            if isinstance(comps_prev, list) and len(comps_prev) > self.tp_rank:
                raw = comps_prev[self.tp_rank].get("draft_solo_pool_bytes")
                if raw is not None:
                    prev_draft_pool_b = int(raw)
        except (OSError, ValueError, TypeError):
            prev_leftover_b = None
        # Structural component shift of THIS rank's balance (T156-D: the
        # small DFLASH solo pool changes draft_solo_pool_bytes by GiBs when
        # toggled/resized) -- see correction_growth_frozen.
        component_shift = prev_draft_pool_b is not None and abs(
            int(draft_pool_b) - prev_draft_pool_b
        ) > (256 << 20)
        if self.correction_growth_frozen(
            delta_b, prev_b, prev_leftover_b, free_b, component_shift
        ):
            delta_b = 0
        elif component_shift and delta_b > 0:
            logger.info(
                "Measured KV-budget (rank %d): draft-solo pool component "
                "shifted %.2f -> %.2f GiB since the stored balance; "
                "consumption gate bypassed for this boot (+%.2f GiB "
                "correction growth).",
                self.tp_rank,
                prev_draft_pool_b / (1 << 30),
                draft_pool_b / (1 << 30),
                delta_b / (1 << 30),
            )
        my_correction = prev_b + delta_b

        world = get_world_group().world_size
        # #201 slice 3: the gather spans the WORLD group; under a pipeline
        # that is every stage, and two stages share the tp_rank index space.
        # The payload therefore carries pp_rank and each stage's writer only
        # folds in its OWN stage's entries -- without the filter, stage 1's
        # balance silently overwrote stage 0's in the shared index (and the
        # record itself is per-stage now, see _measured_kv_budget_cache_path).
        payload = (
            int(getattr(self, "pp_rank", 0) or 0),
            self.tp_rank,
            int(my_correction),
            int(free_b),
            my_component,
        )
        gathered: list = [None] * world
        torch.distributed.all_gather_object(
            gathered, payload, group=get_world_group().cpu_group
        )
        my_pp_rank = int(getattr(self, "pp_rank", 0) or 0)
        corrections = [0] * self.server_args.tp_size
        leftovers = [0] * self.server_args.tp_size
        components: list = [{}] * self.server_args.tp_size
        for pp, rank, corr, left, comp in gathered:
            if pp == my_pp_rank and 0 <= rank < self.server_args.tp_size:
                corrections[rank] = corr
                leftovers[rank] = left
                components[rank] = comp
        path = self._measured_kv_budget_cache_path()
        if self.tp_rank == 0:
            # The weight vector this balance was measured under — the weight
            # planner anchors its family model's absolute bytes against
            # ``weights_alloc_bytes`` AT this vector before predicting other
            # vectors, and the correction reader invalidates corrections
            # measured under a DIFFERENT vector.
            mlp_vec = self._resolved_mlp_vector()
            with open(path, "w") as f:
                json.dump(
                    {
                        "correction_bytes": corrections,
                        "leftover_mib_at_measure": [v >> 20 for v in leftovers],
                        "safety_mib": safety_b >> 20,
                        "max_paused_tag_mib_rank0": max_tag_b >> 20,
                        "mlp_vector": [int(v) for v in mlp_vec],
                        "components": components,
                        "ts": _time.strftime("%Y-%m-%d %H:%M:%S"),
                    },
                    f,
                    indent=1,
                )
            logger.info(
                "Measured KV-budget: per-rank leftover %s MiB at the "
                "post-capture point; required free = safety %d MiB + max "
                "paused rung tag %d MiB (this rank); persisted corrections "
                "%s MiB for the next boot (%s).",
                [v >> 20 for v in leftovers],
                safety_b >> 20,
                max_tag_b >> 20,
                [v >> 20 for v in corrections],
                path,
            )

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

        Under a pipeline (#201 slice 3) the same agreement is needed even
        with uniform budgets: every stage sizes from its OWN free memory
        and (stage-locally) its OWN linear-layer count, so the derived
        request counts legitimately differ per stage -- while an admitted
        request occupies one state slot on EVERY stage that holds linear
        layers. A stage without linear layers contributes the
        PP_STAGE_NO_MAMBA_STATE_SLOTS sentinel and never binds.

        No-op on the default path (byte-level MIN already unified it)."""
        if not (
            self.server_args.uneven_memory_budgets_active()
            or getattr(self, "pp_size", 1) > 1
        ):
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

    def _stage_mamba_layer_counts(self: ModelRunner, config) -> Tuple[int, int]:
        """(stage-local, global) linear-attention (GDN/Mamba) layer counts.

        Under PP every pool CONSTRUCTION site already filters the layer list
        to this stage's [start_layer, end_layer) window, so the state bytes
        actually allocated are stage-local -- but the budget arithmetic in
        handle_max_mamba_cache still counted every stage's layers (#201
        slice 3 item 3; Teil 2 par. 6.2 defect 1). This helper is the single
        source for the stage-local count. pp_size == 1 returns
        (global, global), keeping every consumer byte-identical.
        """
        layers = config.mamba2_cache_params.layers
        n_global = len(layers)
        if self.pp_size <= 1:
            return n_global, n_global
        n_local = sum(1 for lid in layers if self.start_layer <= lid < self.end_layer)
        return n_local, n_global

    def _stage_local_mamba_cache_per_req(self: ModelRunner, config) -> int:
        """Per-request linear-state bytes THIS STAGE actually allocates.

        ``mamba_cache_per_req`` is (per-layer state bytes) x (GLOBAL layer
        count) by construction (configs/mamba_utils.py), so the division
        below is exact. Under PP the sizing must charge only the layers in
        this stage's window; without this every stage budgeted the state of
        ALL stages and under-dimensioned max_mamba_cache_size by roughly a
        factor of pp_size. Returns 0 for a stage whose window holds no
        linear layers -- its sizing branches then contribute the
        PP_STAGE_NO_MAMBA_STATE_SLOTS sentinel to the world MIN instead of
        dividing by zero. pp_size == 1: byte-identical global value.
        """
        per_req = config.mamba2_cache_params.mamba_cache_per_req
        n_local, n_global = self._stage_mamba_layer_counts(config)
        if n_local == n_global or n_global == 0:
            return per_req
        return (per_req // n_global) * n_local

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
            and abs(sa.mamba_full_memory_ratio - MAMBA_FULL_MEMORY_RATIO_DEFAULT) < 1e-9
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
        from sglang.srt.mem_cache.mamba_pool_floor import mamba_hard_floor

        target = self._auto_mamba_target_concurrency()
        slots = math.ceil(target * ratio * MAMBA_AUTO_SAFETY_MARGIN)
        # #581: floor the auto size at the hard per-request demand, so the
        # derived value can never sit below what the running set structurally
        # requires. For the extra-buffer + overlap shape the two agree
        # (ratio == floor_per_req) and the safety margin still applies on top;
        # the floor binds for shapes where `ratio` understates the demand.
        floor = mamba_hard_floor(self.server_args, target)
        return int(max(slots, ratio, floor))

    def _mamba_pool_budget_cost_gb(
        self: ModelRunner, size: int, per_req: int, ratio: int, D: int
    ) -> float:
        """What a mamba pool of ``size`` state slots costs this rank's budget
        on the demand path: the main state plus the speculative intermediate
        state of the requests that pool can admit -- exactly the two posts
        ``handle_max_mamba_cache`` subtracts from ``total_rest_memory``."""
        sa = self.server_args
        per_worker = sa.dp_size if sa.enable_dp_attention else 1
        admitted = size // max(ratio, 1)
        if sa.max_running_requests is not None:
            admitted = min(sa.max_running_requests // per_worker, admitted)
        return (size * per_req + admitted * D * per_req) / (1 << 30)

    def _fit_mamba_pool_to_budget(
        self: ModelRunner,
        wanted: int,
        total_rest_memory: float,
        reserve_gb: float,
        per_req: int,
        ratio: int,
        D: int,
    ) -> int:
        """Fit the mamba pool to the rank's budget when the concurrency
        target cannot be afforded (#307).

        ``--max-running-requests-ceiling`` (#287) is the DIMENSIONING value:
        the pool is built for the ceiling and the admission limit floats
        below it. The state a hybrid model needs per admitted request is not
        elastic -- every running request owns a conv/temporal state slot on
        every rank for as long as it runs -- so a ceiling costs
        ``ceiling * ratio * safety`` slots plus the speculative intermediate
        state, linearly. On a small card a high ceiling therefore does not
        merely shrink the KV pool, it consumes the entire post-weights budget
        and the boot dies in the ledger check before the first KV token
        (measured: ceiling 64 -> 559 MiB short on a 20 GB card, ceiling 32 ->
        407 MiB short, while 16 booted).

        A ceiling that does not fit is a request for concurrency the card
        cannot hold, so the honest answer is to serve the largest ceiling
        that DOES fit rather than to refuse the boot: the pool is fitted to
        the mamba side of the ``--mamba-full-memory-ratio`` split of the
        budget, ``_resolve_max_num_reqs`` then caps ``max_running_requests``
        at the pool's capacity, and the scheduler's ``AdmissionLimiter``
        floats below THAT (it reads the resolved value, not the requested
        ceiling). The result is rank-uniform by the same min-reduce that
        already unifies the pool size across uneven-TP ranks, so the
        throttle/retract controller keeps deciding on replicated inputs.

        The fit engages ONLY when the size computed by the caller would
        leave less than ``MAMBA_CEILING_FIT_MIN_KV_MIB`` for the KV pool, so
        every configuration that boots today keeps its pool byte-identical.
        That trigger is not a safety margin: below it the token pool cannot
        hold a single prefill chunk and the boot is already lost."""
        usable_gb = total_rest_memory - reserve_gb
        cost_gb = self._mamba_pool_budget_cost_gb(wanted, per_req, ratio, D)
        if usable_gb - cost_gb >= MAMBA_CEILING_FIT_MIN_KV_MIB / 1024.0:
            return wanted
        # Give the mamba side the share the fixed-fraction path would have
        # given it; the rest of the budget stays with the KV pool. Per slot
        # the pool costs the main state plus the spec intermediate of the
        # request that slot group admits (D/ratio per slot).
        r = self.server_args.mamba_full_memory_ratio
        mamba_share_gb = max(usable_gb, 0.0) * r / (1.0 + r)
        per_slot_bytes = per_req * (1.0 + D / max(ratio, 1))
        fitted = int(mamba_share_gb * (1 << 30) // per_slot_bytes)
        fitted = max(0, min(fitted, wanted))
        if fitted >= wanted:
            return wanted
        logger.warning(
            "[auto-mamba] the concurrency target does not fit this rank's "
            "budget: %d state slots would cost %.2f GiB of the %.2f GiB left "
            "after weights and the %.2f GiB activation reserve, leaving no KV "
            "pool. Fitting the pool to the budget instead: %d slots "
            "(%.2f GiB, admits ~%d requests per rank). "
            "--max-running-requests-ceiling=%s is the requested ceiling; the "
            "effective one is the min over ranks of the fitted capacity and "
            "is reported by the scheduler. --mamba-full-memory-ratio=%.2f "
            "decides the mamba/KV split of the fitted budget.",
            wanted,
            cost_gb,
            total_rest_memory,
            reserve_gb,
            fitted,
            self._mamba_pool_budget_cost_gb(fitted, per_req, ratio, D),
            fitted // max(ratio, 1),
            self.server_args.max_running_requests_ceiling,
            r,
        )
        return fitted

    def handle_max_mamba_cache(self: ModelRunner, total_rest_memory):
        # #704: start a fresh component ledger for THIS sizing pass.
        self._mamba_budget_components = {}
        config = self.mambaish_config
        server_args = self.server_args
        assert config is not None

        has_spec_dec = not self.spec_algorithm.is_none()
        if has_spec_dec:
            assert server_args.speculative_num_draft_tokens is not None
            assert server_args.max_running_requests is not None

        # #201 slice 3: the per-request state bytes THIS rank's budget pays
        # for. Identical to the global value at pp_size == 1 (byte-identical
        # default path); under PP it counts only this stage's layer window,
        # and is 0 for a stage without linear layers (the sizing branches
        # then produce the PP_STAGE_NO_MAMBA_STATE_SLOTS sentinel and the
        # world MIN-sync below replaces it with the real binding count).
        # getattr: the non-PP arm must stay reachable from minimal runner
        # stubs that predate pp_size (several unit suites drive this
        # function directly).
        if getattr(self, "pp_size", 1) > 1:
            per_req = self._stage_local_mamba_cache_per_req(config)
        else:
            per_req = config.mamba2_cache_params.mamba_cache_per_req
            assert per_req > 0

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
                    per_req
                    * capped_reqs
                    # Max width: adaptive k-ladder rungs / the cross-algorithm
                    # secondary rung can exceed the boot shape's draft tokens.
                    * server_args.max_speculative_num_draft_tokens
                )
                _spec_gb = intermediate_size / (1 << 30)
                _note_mamba_component(self, "speculative intermediate state", _spec_gb)
                total_rest_memory = total_rest_memory - _spec_gb
        elif self._auto_mamba_demand_active():
            # === Demand-driven mamba pool (uneven-DCP auto-sizing) ===========
            # Size the pool to the real serving concurrency, NOT to a fixed
            # fraction of post-weights VRAM. All remaining VRAM then flows to
            # the KV/token pool, so its ceiling reaches the optimum with no
            # manual --mamba-full-memory-ratio flag (the "self-determined +
            # optimal" requirement). See the module-level constants for the
            # rationale.
            ratio = self._calculate_mamba_ratio()
            D = server_args.max_speculative_num_draft_tokens if has_spec_dec else 0
            demand_size = self._auto_mamba_demand_size(ratio)
            # Never exceed what the post-weights budget can physically hold
            # (main state + spec-decode intermediate state per admitted req).
            # A stage without linear layers (per_req == 0, PP only) holds no
            # state bytes, so nothing needs fitting.
            if per_req > 0:
                fit_cap = int(
                    total_rest_memory * (1 << 30) // (per_req * (1 + D / ratio))
                )
            else:
                fit_cap = demand_size
            size = min(demand_size, max(fit_cap, 0))
            reserve_gb = MAMBA_AUTO_ACTIVATION_RESERVE_MIB / 1024.0
            # #307: fit_cap above ignores the activation reserve and leaves
            # the KV pool nothing, so a concurrency target the card cannot
            # hold (a high --max-running-requests-ceiling on a small card)
            # overspends the budget and kills the boot. Fit the pool instead;
            # no-op unless the budget would actually be exhausted.
            size = self._fit_mamba_pool_to_budget(
                size, total_rest_memory, reserve_gb, per_req, ratio, D
            )
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
                _spec_gb = intermediate_size / (1 << 30)
                _note_mamba_component(self, "speculative intermediate state", _spec_gb)
                total_rest_memory = total_rest_memory - _spec_gb
            # Fold prefill-activation headroom back OUT of the KV budget so the
            # token pool does not grow to the physical ceiling and starve the
            # transient DCP-extend prefix-gather scratch (which OOMs a large
            # prefill). This lets the default --rank-auto-reserve-mib stand.
            _note_mamba_component(self, "prefill activation reserve", reserve_gb)
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
            D = server_args.max_speculative_num_draft_tokens if has_spec_dec else 0
            if per_req > 0:
                budget_size = int(mamba_budget_bytes // (per_req * (1 + D / ratio)))
            else:
                # PP stage without linear layers: no state bytes, no budget
                # bound -- the world MIN-sync below carries the real bound.
                budget_size = requested_size
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
                    * server_args.max_speculative_num_draft_tokens
                )
                _spec_gb = intermediate_size / (1 << 30)
                _note_mamba_component(self, "speculative intermediate state", _spec_gb)
                total_rest_memory = total_rest_memory - _spec_gb
        else:
            # Use ratio-based calculation to auto-fit available memory
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
                D = server_args.max_speculative_num_draft_tokens
                # Joint solve: main_state + intermediate = mamba_budget
                server_args.override(
                    "mamba_pool.memory_budget_spec",
                    max_mamba_cache_size=(
                        int(mamba_budget_bytes // (per_req * (1 + D / ratio)))
                        if per_req > 0
                        else PP_STAGE_NO_MAMBA_STATE_SLOTS
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
                _spec_gb = intermediate_size / (1 << 30)
                _note_mamba_component(self, "speculative intermediate state", _spec_gb)
                total_rest_memory = total_rest_memory - _spec_gb
            else:
                server_args.override(
                    "mamba_pool.memory_budget",
                    max_mamba_cache_size=(
                        int(mamba_budget_bytes // per_req)
                        if per_req > 0
                        else PP_STAGE_NO_MAMBA_STATE_SLOTS
                    ),
                )
                # Uneven TP / PP stages: agree on the min across ranks
                # (see _sync_uneven_mamba_cache_size).
                self._sync_uneven_mamba_cache_size()

        # Uneven TP (--rank-gpu-memory-mib): the ratio-based auto-sizing
        # above ran on rank-LOCAL memory, so the derived request COUNT can
        # differ slightly per rank (the per-request state bytes scale with
        # each rank's head share, so proportional budgets give near-equal
        # counts). The schedulers run in lockstep and must agree on one
        # count — min-reduce it before anything consumes it.
        # #201 slice 3: a pipeline needs the same agreement even with
        # uniform budgets -- each stage sized from its own free memory and
        # its own stage-local layer window, and every admitted request
        # occupies a slot on every stage that holds linear layers.
        if (
            self.server_args.uneven_memory_budgets_active()
            or getattr(self, "pp_size", 1) > 1
        ) and get_world_group().world_size > 1:
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

        # #364 resident-slot cap. Applied AFTER every profiling branch and
        # after the uneven-TP min-sync, so it is the last word on the pool
        # geometry and cannot be undone by a branch that recomputes a size.
        # Rank-uniform without a collective: it is a server arg, identical on
        # every rank by construction, and it only ever lowers -- so the
        # min-reduced agreement above still holds after it.
        #
        # This IS the accounting route: mamba_state_memory below is computed
        # from max_mamba_cache_size and subtracted from the budget the KV
        # sizing then spends, so the slots the cap keeps out become KV tokens
        # with no transfer and no second ledger. What the cap does NOT do is
        # grow the KV pool at RUNTIME when a session vacates -- the pool is
        # fixed at boot; a vacated slot is reused by another session, not
        # returned to the KV pool (see the #364 remainder).
        _resident_cap = getattr(server_args, "gdn_resident_state_slots", None)
        if _resident_cap is not None:
            from sglang.srt.mem_cache.gdn_slot_ladder import (
                cap_is_binding,
                effective_state_slots,
                remember_profiled_state_slots,
            )

            # Write-once on the ARGS, not on this runner. A phase-flip
            # instance sizes a SECOND stack from a deepcopy of these args
            # taken AFTER the override below, so a plain read of
            # max_mamba_cache_size would hand that stack the CAPPED count
            # and it would size its admission ceiling from the shrunken
            # pool -- the two stacks then disagree on max_num_reqs and the
            # flip's boot guard refuses the instance. See
            # remember_profiled_state_slots for the measured failure.
            _profiled = remember_profiled_state_slots(server_args)
            # #364 slice 3: preserve the PRE-CAP profiled slot count so the
            # concurrency ceiling (max_running_requests) is sized from the
            # SESSION budget, not from the shrunken resident pool. Without
            # this, _resolve_max_num_reqs reads the capped max_mamba_cache_size
            # and craters concurrency (capped_slots // ratio -> 4 // 5 = 0).
            # The physical pool stays capped; only the admission ceiling
            # decouples, and the overflow sessions run vacated.
            self._gdn_profiled_state_slots = _profiled
            if cap_is_binding(_resident_cap, _profiled):
                _capped = effective_state_slots(_profiled, _resident_cap)
                logger.info(
                    "GDN resident-slot cap: state slots %d -> %d "
                    "(--gdn-resident-state-slots %d); %.2f GB stays out of "
                    "the state pool and is spent on KV instead. Sessions "
                    "beyond the cap run with a vacated (host-blob) state "
                    "where an idle slot holder can exist -- under pp_size>1 "
                    "the vacate's population (kv-session-offload) is refused, "
                    "so there they WAIT for a slot instead (throughput cost, "
                    "not an OOM).",
                    _profiled,
                    _capped,
                    int(_resident_cap),
                    (_profiled - _capped) * per_req / (1 << 30),
                )
                server_args.override(
                    "mamba_pool.gdn_resident_state_slots",
                    max_mamba_cache_size=_capped,
                )
            else:
                logger.info(
                    "GDN resident-slot cap: --gdn-resident-state-slots %d is "
                    "at or above the profiled %d state slots; the cap is a "
                    "ceiling, not a demand, and nothing changes.",
                    int(_resident_cap),
                    _profiled,
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
                f"mamba_cache_per_req={per_req / (1 << 20):.2f} MB"
                + (
                    f" stage-local, {self._stage_mamba_layer_counts(config)[0]} of "
                    f"{self._stage_mamba_layer_counts(config)[1]} linear layers on "
                    f"pp_rank {self.pp_rank}"
                    if getattr(self, "pp_size", 1) > 1
                    else ""
                )
                + "). "
                "Try: (1) reduce --max-running-requests, "
                "(2) increase --mem-fraction-static, "
                "(3) reduce --speculative-num-draft-tokens, or "
                "(4) use GPUs with more memory."
            )

        # Stage-local bytes: what THIS rank's pools will actually allocate
        # (the construction sites filter to the stage window). Charging the
        # global per-request bytes here would subtract other stages' state
        # from this rank's KV budget (#201 slice 3 item 3).
        mamba_state_memory = server_args.max_mamba_cache_size * per_req / (1 << 30)
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
            assert kv_lora_rank % quant_block_size == 0, (
                f"kv_lora_rank {kv_lora_rank} must be multiple of quant_block_size {quant_block_size}"
            )

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
                assert not self.server_args.enable_mamba_extra_buffer_lazy(), (
                    "Lazy extra buffer requires overlap schedule (--disable-overlap-schedule is incompatible)"
                )
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

    def _gapped_corridor_holdback(self: ModelRunner, rest_memory: float):
        """Hold back the user reserve on a gapped cut. Returns (rest, post).

        ``rank_user_reserve_mib`` is the corridor headroom -- the free column a
        card is meant to keep for the user -- and it had exactly ONE consumer,
        ``phase_flip_runtime``. A plain PP boot driven by
        ``--rank-gpu-memory-mib`` therefore spent the budget to the last MiB:
        boot v7pp8 priced PP2 at weights 6.248 + mamba 0.384 + pool 11.923 GiB
        against an 18.55 GiB budget and left 0.15 GB free, then died on a
        32 MiB decode allocation. The pool was not mis-priced; there was simply
        nothing left for the transients the price does not cover.

        GATED TO THE GAPPED PATH ON PURPOSE. Every shipped configuration was
        solved against today's arithmetic, and silently moving the reserve into
        the posts would shrink every one of their KV pools by a gigabyte per
        card -- a change to solved, measured configurations that no boot here
        has evidence for. The gapped cut is the new path and the one with the
        demonstrated OOM, so it is the only one whose sizing moves.

        The corridor is a TARGET and not a hard floor (softened 2026-08-16:
        undershoot is allowed with a warning, the hard rule is OOM avoidance),
        so this holds back at most what is actually available and never drives
        the pool negative.
        """
        try:
            from sglang.srt.distributed.utils import pp_gapped_ownership_active

            pp_size = int(getattr(self.server_args, "pp_size", 1) or 1)
            if not pp_gapped_ownership_active(pp_size):
                return rest_memory, None
            # None means "unset, take the default"; 0 means "the operator asked
            # for no holdback". The ``or`` idiom conflates the two and would
            # make the reserve impossible to switch off.
            configured = getattr(self.server_args, "rank_user_reserve_mib", None)
            reserve_mib = 1024 if configured is None else int(configured)
            if reserve_mib <= 0:
                return rest_memory, None
            reserve_gb = reserve_mib / 1024.0
            if reserve_gb >= rest_memory:
                # Never negative: a budget this tight is a configuration
                # problem, and it is reported by the exhausted-budget path
                # with its own words rather than disguised as a tiny pool.
                logger.warning(
                    "gapped corridor holdback of %.3f GiB exceeds the %.3f GiB "
                    "left for KV on this rank; holding back nothing and "
                    "letting the budget check speak for itself.",
                    reserve_gb,
                    rest_memory,
                )
                return rest_memory, None
            return rest_memory - reserve_gb, ("gapped corridor holdback", reserve_gb)
        except Exception as e:  # noqa: BLE001 - sizing may never die on this
            logger.warning("gapped corridor holdback skipped: %s", e)
            return rest_memory, None

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
                self.server_args.activation_reserve_mb(
                    get_device_memory_capacity(self.device)
                )
                / 1024,
            )
        budget_bytes = (
            int(max(0.0, free_gb - headroom_gb) * (1 << 30))
            + pool.post_capture_backed_bytes
        )
        # #656: the flip seam is charged inside _config_from_budget, which
        # every sizing path funnels through -- including this one.
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
        assert not self.use_mla_backend, (
            "unified memory pool does not support MLA-hybrid-Mamba yet"
        )
        # The full sub-pool is page-aware (via `MultiEndedAllocator(page_size=...)`);
        # the mamba sub-pool stays page=1.
        assert self.page_size >= 1, f"page_size must be >= 1, got {self.page_size}"
        # Mirror the non-shared path's extra_max_context_len computation.
        # Max width (not the boot shape's): adaptive k-ladder rungs and the
        # cross-algorithm secondary rung reserve up to max_... slots per
        # decode; mem_cache/common.get_req_to_token_extra_context_len already
        # uses the max on the classic path.
        extra_max_context_len = 4
        if self.server_args.max_speculative_num_draft_tokens is not None:
            extra_max_context_len += self.server_args.max_speculative_num_draft_tokens

        # SET-AWARE ON BOTH LISTS, for the reason spelled out at
        # ``stage_owned_layer_ids``: under SGLANG_PP_LAYER_SET the
        # interval is this stage's SPAN, and the gapped layout is exactly the
        # case where span and set differ. The unified path reaches the same
        # HybridLinearKVPool as the branch above, so leaving it on the
        # interval would have kept half the defect alive behind a flag.
        from sglang.srt.distributed.utils import stage_owned_layer_ids

        mamba_layer_ids = stage_owned_layer_ids(
            config.mamba2_cache_params.layers, self.start_layer, self.end_layer
        )
        full_attention_layer_ids = stage_owned_layer_ids(
            config.full_attention_layer_ids, self.start_layer, self.end_layer
        )

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
            # Max width: the intermediate SSM/conv verify caches are indexed
            # by draft-token step; adaptive rungs (k4/k5 -> 5/6 tokens) and
            # the cross-algorithm DFLASH rung (16 tokens) exceed the boot
            # shape's value -- sizing with it would OOB the step axis.
            speculative_num_draft_tokens=self.server_args.max_speculative_num_draft_tokens,
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
        assert not self.use_mla_backend, (
            "unified memory pool does not support MLA-SWA hybrid yet"
        )
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

    def _wl_attach_spill_host_pool(self: ModelRunner):
        """Weightless-KV Stage B1: attach the pinned-host overflow tier to the
        full-attention KV pool (`HybridLinearKVPool.full_kv_pool`).

        The tier is a stock ``MHATokenToKVPoolHost`` (the de-risked lossless
        byte substrate). Its constructor sizes in whole GB, so we request the
        smallest GB budget covering ``_wl_spill_host_tokens`` and simply use
        host slots [0, H); the static slot->tier map (compacted slot s ->
        host slot s - device_tokens) replaces the HostKVCache alloc/free state
        machine entirely -- no radix/HiCache controller is involved. The
        GDN/linear-attention state (mamba pool) keeps its small resident state
        and is NOT tiered."""
        from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost

        full_pool = self.token_to_kv_pool.full_kv_pool
        per_token_bytes = (
            full_pool.head_num
            * full_pool.head_dim
            * full_pool.layer_num
            * full_pool.store_dtype.itemsize
            * 2  # K + V
        )
        need_bytes = self._wl_spill_host_tokens * per_token_bytes
        host_size_gb = max(1, -(-need_bytes // 10**9))  # ceil to whole GB
        host_pool = MHATokenToKVPoolHost(
            device_pool=full_pool,
            host_to_device_ratio=0.0,
            host_size=host_size_gb,
            page_size=self.page_size,
            layout="layer_first",
            pin_memory=True,
        )
        if host_pool.size < self._wl_spill_host_tokens:
            raise ValueError(
                "weightless-KV host spill: allocated host tier holds "
                f"{host_pool.size} tokens < requested "
                f"{self._wl_spill_host_tokens}."
            )
        # #127 GRENZ-ASSERTION. Every host<->device move on this tier is a raw
        # byte copy driven by host_pool.token_stride_size; nothing downstream
        # re-derives the element type. A host tier built from a different
        # dtype than the device pool would therefore NOT fail -- it would
        # reinterpret KV bytes at the wrong stride and return plausible
        # garbage. That seam was theoretical while one dtype was group-wide;
        # with per-role precision it is not. Check it once, here, where both
        # objects exist.
        from sglang.srt.layers.dcp.role_kv_dtype import host_tier_stride_mismatch

        _mismatch = host_tier_stride_mismatch(
            device_store_itemsize=full_pool.store_dtype.itemsize,
            device_head_num=full_pool.head_num,
            device_head_dim=full_pool.head_dim,
            host_itemsize=host_pool.dtype.itemsize,
            host_token_stride_size=host_pool.token_stride_size,
        )
        if _mismatch is not None:
            raise ValueError(f"weightless-KV host spill: {_mismatch}")
        self.wl_spill_host_pool = host_pool
        logger.info(
            "Weightless-KV host spill (B1): attached %d-token pinned host "
            "tier (%.2f GB) to the full-attention KV pool.",
            self._wl_spill_host_tokens,
            need_bytes / 1e9,
        )

    def _kv_sess_attach_host_pool(self: ModelRunner):
        """kv-session-offload (S1): pinned host pool for ONE spilled
        session's full-attention KV shard on this rank.

        Sized for the worst case: a max-context session whose owned share is
        the LARGEST rank's share (rank-uniform GB request -> the host pool's
        cross-rank min-sync is a no-op). Host rows are used positionally
        [0, n) per spilled session -- the HostKVCache alloc/free state
        machine is not used. GDN/Mamba state is never tiered."""
        from sglang.srt.distributed.utils import (
            cp_token_prefix,
            cp_token_split_factor,
        )
        from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost

        full_pool = self.token_to_kv_pool.full_kv_pool
        dcp = int(getattr(self, "dcp_size", 1) or 1)
        if dcp > 1:
            prefix = cp_token_prefix(dcp)
            S = cp_token_split_factor(dcp)
            max_ratio = max(prefix[r + 1] - prefix[r] for r in range(len(prefix) - 1))
        else:
            S, max_ratio = 1, 1
        ctx = int(self.model_config.context_len)
        # S4: N equal per-session regions -> size for N max-context sessions.
        max_spills = max(1, int(self.server_args.kv_session_offload_max_spills))
        region_tokens = (ctx // S + 2) * max_ratio
        need_tokens = region_tokens * max_spills
        per_token_bytes = (
            full_pool.head_num
            * full_pool.head_dim
            * full_pool.layer_num
            * full_pool.store_dtype.itemsize
            * 2  # K + V
        )
        budget_gib = float(
            getattr(self.server_args, "kv_session_offload_host_ram_gib", 0.0) or 0.0
        )
        if budget_gib > 0:
            # ---- P2 (deep-offload S1): RAM-BUDGETED host pool --------------
            # Separate branch on purpose: with the flag unset (else:) not a
            # single statement below runs, so the default path stays exactly
            # what it was.
            #
            # The budget is a PHYSICAL CEILING on the allocation, nothing
            # else -- it is NEVER read by the tick regulator / wave-back /
            # admission logic (P3 guard, unit-tested). The per-session DEPTH
            # (region_tokens) is untouched: a session can never hold more than
            # context_len, so the budget instead bounds HOW MANY regions are
            # dimensioned == the effective max_spills.
            #
            # RANK-UNIFORM without a new collective. Careful: the REQUESTED GB
            # is min(context_need, budget/rank), and the context_need term is
            # rank-LOCAL under uneven TP (different kv-head shares -> different
            # bytes/token), exactly as in the flag-OFF path below. What makes
            # the result uniform is HostKVCache's OWN min-all-reduce over the
            # token capacity (sync_fixed_hicache_size, pool_host/base.py) --
            # active precisely when the per-token bytes differ (uneven TP); with
            # even TP every rank computes the identical number anyway. The
            # effective region count is therefore derived from the POST-sync
            # host_pool.size, so it is one integer on every rank -- including
            # the eff<1 fail-fast below, which then fires on ALL ranks.
            from sglang.srt.managers.kv_session_offload import (
                host_pool_effective_max_spills,
                host_pool_request_gb,
            )

            # PP/DP are rejected for this feature (server_args), and draft
            # workers never attach a pool -> the TP ranks are exactly the
            # processes that share the node-wide budget.
            n_pool_ranks = max(1, int(getattr(self, "tp_size", 1) or 1))
            host_size_gb = host_pool_request_gb(
                need_tokens, per_token_bytes, budget_gib, n_pool_ranks
            )
            host_pool = MHATokenToKVPoolHost(
                device_pool=full_pool,
                host_to_device_ratio=0.0,
                host_size=host_size_gb,
                page_size=self.page_size,
                layout="layer_first",
                pin_memory=True,
                # Name this pool's post in the joint pinned-host-RAM guard
                # (#550). Without it the spill pool would be booked under the
                # HiCache class name and a refusal would point the operator at
                # the wrong flag.
                budget_label="kv-session-offload spill pool",
                budget_flag="--kv-session-offload-host-ram-gib",
            )
            eff_max_spills = host_pool_effective_max_spills(
                host_pool.size, region_tokens, max_spills
            )
            if eff_max_spills < 1:
                raise ValueError(
                    "kv-session-offload: --kv-session-offload-host-ram-gib="
                    f"{budget_gib:g} GiB (node-wide, {n_pool_ranks} ranks -> "
                    f"{host_size_gb:.2f} GB/rank) cannot hold even ONE "
                    f"full-context session on this rank: the allocated pool "
                    f"holds {host_pool.size} tokens < {region_tokens} tokens "
                    f"per region (context_len {ctx}, S {S}, max_ratio "
                    f"{max_ratio}, {per_token_bytes} B/token -> "
                    f"{region_tokens * per_token_bytes / 1e9:.2f} GB per "
                    "region per rank). Raise the budget or lower "
                    "--context-length."
                )
            self.kv_sess_host_pool = host_pool
            self.kv_sess_region_tokens = region_tokens
            # The manager must partition into the EFFECTIVE count, not the
            # configured one, or its region assert / _free_regions would run
            # past the allocated pool.
            self.kv_sess_max_spills = eff_max_spills
            logger.info(
                "kv-session-offload (P2 budget): attached %d-token pinned host "
                "pool (%.2f GB/rank of a %g GiB node budget over %d ranks, "
                "%d full-attention layers) for up to %d concurrent spill "
                "sessions (configured %d, %d tokens/region).",
                host_pool.size,
                host_pool.size * per_token_bytes / 1e9,
                budget_gib,
                n_pool_ranks,
                full_pool.layer_num,
                eff_max_spills,
                max_spills,
                region_tokens,
            )
            if eff_max_spills < max_spills:
                logger.warning(
                    "kv-session-offload (P2 budget): effective max_spills "
                    "reduced %d -> %d by --kv-session-offload-host-ram-gib="
                    "%g (per-session depth %d tokens is unchanged).",
                    max_spills,
                    eff_max_spills,
                    budget_gib,
                    region_tokens,
                )
            return
        host_size_gb = max(1, -(-(need_tokens * per_token_bytes) // 10**9))
        host_pool = MHATokenToKVPoolHost(
            device_pool=full_pool,
            host_to_device_ratio=0.0,
            host_size=host_size_gb,
            page_size=self.page_size,
            layout="layer_first",
            pin_memory=True,
            budget_label="kv-session-offload spill pool",
            budget_flag="--kv-session-offload-host-ram-gib",
        )
        if host_pool.size < need_tokens:
            raise ValueError(
                "kv-session-offload: allocated host pool holds "
                f"{host_pool.size} tokens < required {need_tokens} "
                f"(context_len {ctx}, S {S}, max_ratio {max_ratio}, "
                f"max_spills {max_spills})."
            )
        self.kv_sess_host_pool = host_pool
        # S4: the manager partitions [0, size) into max_spills regions of
        # >= region_tokens each; expose the per-region capacity it must use.
        self.kv_sess_region_tokens = region_tokens
        self.kv_sess_max_spills = max_spills
        logger.info(
            "kv-session-offload: attached %d-token pinned host pool "
            "(%.2f GB, %d full-attention layers) for up to %d concurrent "
            "spill sessions (%d tokens/region).",
            host_pool.size,
            need_tokens * per_token_bytes / 1e9,
            full_pool.layer_num,
            max_spills,
            region_tokens,
        )

    def _pool_kv_head_num(self: ModelRunner) -> int:
        """The kv-head count this rank's KV pool must be shaped for.

        Normally this rank's own shard, ``get_num_kv_heads(attn_tp_size)``.

        EXCEPTION -- draft-solo placement (``--speculative-draft-placement
        solo``): the SOLO HOST builds its draft model under a weight-TP=1
        override (same mechanism as the weightless-KV head rank), so the draft
        attention projects the FULL ``total_num_kv_heads``. The draft KV pool
        must be shaped the same way, otherwise ``set_kv_buffer``'s
        ``view(-1, pool_heads, head_dim)`` reinterprets the surplus heads as
        surplus TOKENS and the store kernel rejects the loc/kv batch mismatch
        ("expected 64 but got 32") during the draft cuda-graph capture.

        Shadow ranks build no draft pool at all, and every TARGET pool plus the
        whole split-placement path keeps the per-rank shard -> byte-identical.
        """
        if self.is_draft_worker and getattr(self, "is_draft_solo_host", False):
            return self.model_config.get_total_num_kv_heads()
        # EXCEPTION -- weightless-KV fast lane: identical reasoning. The head
        # rank builds its model under the same weight-TP=1 override and
        # broadcasts the FULL total_num_kv_heads each step; every rank writes
        # all of them into its owned token slots. The HYBRID pool site already
        # spells this out inline (`_hybrid_kv_head_num`); stating it here too
        # keeps the plain-MHA pool (non-hybrid models on the lane) from being
        # shaped for a per-rank shard the broadcast would not match. Since #143
        # the lane admits chain speculation, but every draft runner on it goes
        # through the solo-host branch above -- `is_draft_worker` is never paired
        # with a weightless role -- so no draft pool reaches this line either.
        if not self.is_draft_worker and weightless_kv_active():
            return self.model_config.get_total_num_kv_heads()
        # EXCEPTION -- uneven-DCP KV replication (#345). Under
        # ``uneven_dcp_kv_replicated`` the full-attention KV cache is TOKEN-
        # sharded and every rank stores the FULL, replicated kv-heads: the
        # attention write gathers this rank's uneven projection shard up to
        # ``get_total_num_kv_heads()`` (``_dcp_write_gather``) and the paged
        # wrappers are planned with ``dcp_full_kv_heads``. The HYBRID
        # (mamba/GDN) and SWA-hybrid pool sites state this inline and were
        # therefore correct; the plain-MHA pool -- the family a DENSE model
        # lands in -- read this function and got the per-rank SHARD instead.
        #
        # That is not an over/under-allocation, it is silent corruption:
        # ``masked_set_kv_buffer_kernel`` writes at ``loc * H * D`` with H
        # taken from the CACHE tensor (the full count), so a pool row stride
        # of the smaller shard makes every owned slot land at the wrong
        # address -- an offset that grows with the slot id, i.e. depends on
        # what the allocator handed out, i.e. on request order. Measured on
        # Llama-3.1-8B (8 kv heads) at --rank-tp-ratio 3,1: pools of 6 and 2
        # heads against an 8-head write/read, first decode step 57% wrong at
        # L00.o_proj while prefill was at the noise floor.
        #
        # #108: with --draft-kv-layout dcp the DRAFT pool joins this exception
        # -- it is token-sharded by the same owner rule, so it too stores the
        # FULL replicated kv-heads. draft_pool_is_replicated() is the single
        # predicate the attention backend reads as well.
        from sglang.srt.layers.dcp.owner import draft_pool_is_replicated

        if not draft_pool_is_replicated(
            self.is_draft_pool_worker, self.server_args
        ) and uneven_dcp_kv_replicated(self.dcp_size):
            return self.model_config.get_total_num_kv_heads()
        return self.model_config.get_num_kv_heads(get_parallel().attn_tp_size)

    def _dcp_token_sharded_pool_rows(self: ModelRunner, global_rows: int) -> int:
        """Physical rows a full-attention MHA pool needs for ``global_rows``
        GLOBAL token slots on THIS rank (#345).

        Under the WEIGHTED uneven-DCP owner rule ``max_total_num_tokens`` is
        the shared GLOBAL context budget C, and this rank stores only its
        ``ratio_r / S`` share of it -- exactly as the HybridLinearKVPool and
        the SWA-hybrid full sub-pool already do, through the same
        ``dcp_compact_pool_rows`` rule. Sizing the plain-MHA pool at C instead
        allocated the whole global context per rank, which with the head-count
        fix above (per-rank shard -> full replicated heads) is an OOM at boot
        rather than merely waste.

        Returns ``global_rows`` unchanged off the weighted lane -- including
        the even-modulo owner rule, where ``max_total_num_tokens`` already IS
        the per-rank pool and the allocator's index space is the inflated one
        (``max_total * cp_token_split_factor``).
        """
        from sglang.srt.distributed.utils import (
            cp_token_split_factor,
            get_cp_token_ratios,
        )
        from sglang.srt.layers.dcp.owner import draft_pool_is_replicated

        # #108: --draft-kv-layout dcp lets the draft pool through to the same
        # ratio-proportional row count. Default 'replicated' keeps the early
        # return, i.e. the full global context per rank, byte-identical.
        if draft_pool_is_replicated(
            self.is_draft_pool_worker, self.server_args
        ) or not uneven_dcp_active(self.dcp_size):
            return int(global_rows)
        if not uneven_dcp_kv_replicated(self.dcp_size):
            return int(global_rows)
        from sglang.srt.layers.dcp.owner import dcp_compact_pool_rows

        ratios = get_cp_token_ratios()
        rows = dcp_compact_pool_rows(
            int(global_rows),
            cp_token_split_factor(self.dcp_size),
            int(ratios[get_parallel().attn_dcp_rank]),
        )
        logger.info(
            "Uneven-DCP token-sharded MHA pool: %d global context slots -> %d "
            "physical rows on dcp_rank %d (vector %s), %d replicated kv heads.",
            int(global_rows),
            rows,
            get_parallel().attn_dcp_rank,
            ratios,
            self.model_config.get_total_num_kv_heads(),
        )
        return rows

    def _decoupled_kv_pool_override(
        self: ModelRunner, all_attn_layer_ids, default_size: int
    ):
        """#704b B1: swap layer-ownership sizing for token-share sizing.

        Returns ``(layer_ids_or_None, size)``. ``None`` means "unchanged" and
        is returned for every rank on every path unless B1 is explicitly
        enabled, so the default build is byte-identical -- the caller keeps its
        own expression rather than receiving a reconstructed copy of it, which
        is what makes "unchanged" mean unchanged instead of "recomputed the
        same way".

        Enabled by ``SGLANG_DECOUPLED_KV=1``. B1 has no CLI surface yet by
        decision (the flag is module-level, dcp_size was deliberately NOT
        overloaded), so the build gate is an env var rather than a server arg
        that would have to be threaded through a shared parse path.
        """
        from sglang.srt.utils import get_bool_env_var

        if not get_bool_env_var("SGLANG_DECOUPLED_KV"):
            return None, default_size

        from sglang.srt.distributed.utils import (
            cp_token_split_factor,
            get_cp_token_ratios,
        )
        from sglang.srt.mem_cache.decoupled_kv_arming import record_pool_plan
        from sglang.srt.mem_cache.decoupled_kv_pool_plan import (
            plan_for_rank,
            pool_build_args,
        )

        _S = cp_token_split_factor(self.dcp_size)
        _ratios = get_cp_token_ratios()
        _ratio_r = int(_ratios[get_parallel().attn_dcp_rank])
        plan = plan_for_rank(
            list(all_attn_layer_ids),
            self.start_layer,
            self.end_layer,
            self.max_total_num_tokens,
            # Per-layer cell bytes are not needed for the two build arguments;
            # the plan carries them only for the world-conservation check, so
            # a placeholder here would silently corrupt that check. Pass the
            # real per-token-per-layer cell instead.
            self._decoupled_kv_cell_bytes(),
            armed=True,
            share=_ratio_r / _S,
            period=_S,
        )
        # Record what was BUILT, so arming can be checked against reality.
        record_pool_plan(plan)
        layer_ids, size = pool_build_args(plan)
        logger.info(
            "#704b B1 decoupled KV pool: %d attention layers (all), %d rows "
            "for token share %d/%d on dcp_rank %d (stage-local would have been "
            "%d layers x %d rows)",
            len(layer_ids),
            size,
            _ratio_r,
            _S,
            get_parallel().attn_dcp_rank,
            sum(1 for i in all_attn_layer_ids if self.start_layer <= i < self.end_layer),
            default_size,
        )
        return list(layer_ids), size

    def _decoupled_kv_cell_bytes(self: ModelRunner) -> int:
        """KV bytes per token per full-attention layer, FROM CONFIG.

        K and V, this rank's kv heads, head_dim, element size -- the same
        quantities and the same idiom as ``pool_configurator._compute_cell_size``
        (``:315-331``), deliberately not a fitted constant: fitting this against
        an observed pool is how #704 previously arrived at a 2x-wrong cell that
        a compensating fudge hid.
        """
        import torch

        return (
            2
            * self.model_config.get_num_kv_heads(get_parallel().attn_tp_size)
            * self.model_config.head_dim
            * torch._utils._element_size(self.kv_cache_dtype)
        )

    def _swa_hybrid_dcp_lane(self: ModelRunner) -> bool:
        """Is this rank serving SWA-hybrid uneven DCP? (#96, Stage B)

        Thin adapter over the shared predicate ``swa_hybrid_dcp_lane`` -- the
        SAME function ``TritonAttnBackend.__init__`` calls -- so the KV pool and
        the attention backend can never be in different modes (pool sized for a
        token split the backend does not perform, or the reverse: the classic
        right-token/wrong-slot corruption).

        The two configurations that would be silently wrong rather than merely
        unsupported are rejected HERE, at pool construction, before a byte is
        allocated:

        * ratio-sized SWA pool. In ratio mode the SWA pool is
          ``swa_full_tokens_ratio * full_tokens``, and under DCP ``full_tokens``
          is the GLOBAL context C -- while the SWA pool is not sharded at all.
          That is the pre-#90 OOM disease multiplied by the token split (the
          measured "un-sharded SWA pool sized at the global 249472 budget and
          OOM'd outright" in _swa_hybrid_kv_token_cap's docstring). Stage B
          requires Stage A: --swa-pool-sizing cap (or --disable-radix-cache,
          which routes to the same configurator).
        * HiCache. ``cache_controller._dcp_kv_transfer_pairs`` translates device
          indices through the owner rule and drops the unowned ones; it is
          pool-agnostic, so it would apply that compaction to the SWA index
          stream too, where slots are LOCAL and unsharded.
        """
        from sglang.srt.distributed.utils import uneven_dcp_kv_replicated
        from sglang.srt.layers.dcp.owner import swa_hybrid_dcp_lane

        if not self.is_hybrid_swa or self.is_draft_worker:
            return False
        if not uneven_dcp_kv_replicated(self.dcp_size):
            return False
        n_full = len(self.model_config.full_attention_layer_ids)
        n_swa = len(self.model_config.swa_attention_layer_ids)
        sa = self.server_args
        capped = sa.swa_pool_sizing == "cap" or bool(sa.disable_radix_cache)
        lane = swa_hybrid_dcp_lane(
            is_hybrid_swa=True,
            uneven_plan=True,
            is_draft_worker=False,
            num_full_layers=n_full,
            num_swa_layers=n_swa,
            swa_pool_sizing_capped=capped,
        )
        if not lane:
            if n_full > 0 and n_swa > 0 and not capped:
                raise ValueError(
                    "SWA-hybrid uneven DCP (--dcp-size with a --rank-tp-ratio "
                    "plan on a sliding-window model) requires the cap-sized SWA "
                    "pool: pass --swa-pool-sizing cap (task #91 Stage A) or "
                    "--disable-radix-cache. In ratio mode the SWA pool would be "
                    f"sized {sa.swa_full_tokens_ratio} x the GLOBAL context "
                    "budget while it is not token-sharded at all -- every rank "
                    "would try to hold a multiple of its own reach and OOM. "
                    "Alternatively drop --dcp-size."
                )
            return False
        if getattr(self.server_args, "enable_hierarchical_cache", False):
            raise ValueError(
                "SWA-hybrid uneven DCP (#96) does not support the hierarchical "
                "cache: the device<->host index translation applies the DCP "
                "owner rule to every stream, but the sliding-window sub-pool is "
                "local and unsharded, so its slots would be compacted and "
                "dropped. Drop --enable-hierarchical-cache, or drop --dcp-size."
            )
        return True

    def _init_pools(self: ModelRunner):
        """Initialize the memory pools."""
        max_num_reqs = self.max_running_requests

        # ---- kv-session-offload (S1): scope fail-fast ----------------------
        self._kv_sess_scratch_slot = None
        # PS2 (deep prefill-spill) staging carve; stays 0/None unless
        # --kv-session-offload-prefill is set (flag OFF reserves nothing).
        self._kv_sess_prefill_stage_base = 0
        self._kv_sess_prefill_stage_tokens = 0
        if self.server_args.enable_kv_session_offload and not self.is_draft_worker:
            if self.mambaish_config is None:
                raise ValueError(
                    "--enable-kv-session-offload (S1) supports hybrid "
                    "Mamba/GDN models only (full-attention KV spilled, GDN "
                    "state resident); this model has no mamba config."
                )
            if self.post_capture_kv_active:
                raise ValueError(
                    "--enable-kv-session-offload cannot be combined with "
                    "post-capture KV sizing "
                    "(SGLANG_ENABLE_POST_CAPTURE_KV_SIZING=1): the backing "
                    "finalize would leave the appended scratch row unbacked."
                )
            _dcp = int(getattr(self, "dcp_size", 1) or 1)
            if _dcp > 1 and self.server_args.rank_tp_ratio is None:
                raise ValueError(
                    "--enable-kv-session-offload with DCP requires the "
                    "natural-page uneven-DCP allocator (--rank-tp-ratio); "
                    "the stock even-DCP inflated-page layout re-interprets "
                    "slot identity and is out of S1 scope."
                )
            if _dcp <= 1:
                logger.warning(
                    "kv-session-offload: no token-sharded DCP active -- the "
                    "spilled session streams its WHOLE context over a single "
                    "PCIe link every decode step and will be very slow at "
                    "long contexts. Uneven DCP (SGLANG_UNEVEN_DCP=1 + "
                    "--rank-tp-ratio) splits the stream across ranks."
                )

        # ---- Weightless-KV Stage B1: host-spill pool split -----------------
        # Split the PROFILED per-rank pool D (what actually fits in VRAM) into
        #   [0, D - B)   allocatable DEVICE slots,
        #   [D - B, D)   the bounded STAGING region (B = the B0 block size;
        #                never handed out by the allocator -- the block loop
        #                streams host-resident blocks H2D into it), and append
        #   H = weightless_kv_host_spill_tokens HOST slots to the LOGICAL
        # compacted slot space: max_total_num_tokens becomes (D - B) + H, so
        # the DCP allocator index space ((D-B+H) * dcp_size) and scheduler
        # admission grow past VRAM while the device tensors stay exactly the
        # profiled size D. The slot->tier map is STATIC and rank-uniform by
        # construction: compacted slot s >= D - B lives on host at s - (D - B).
        self._wl_spill_phys_tokens = 0
        _wl_spill = int(
            getattr(self.server_args, "weightless_kv_host_spill_tokens", 0) or 0
        )
        if _wl_spill > 0 and not self.is_draft_worker:
            # Alias the import: a bare local import would SHADOW the module-
            # level `weightless_kv_active` for this whole function scope
            # (UnboundLocalError on the default path).
            from sglang.srt.distributed.utils import (
                weightless_kv_active as _wl_kv_active,
            )

            if _wl_kv_active():
                _wl_stage = int(self.server_args.weightless_kv_chunked_block_size)
                # #136b H2D prefetch/double-buffer: carve TWO block-sized
                # staging regions (block j uses region j % 2) so the captured
                # side-stream copy pipeline can fill one region while
                # attention reads the other, PLUS one scratch row for the
                # graph-safe owner-write staging (moved out of the regions so
                # early cross-layer prefetch copies never collide with it).
                # The backend derives its prefetch enable purely from the
                # carve size (staging >= 2 blocks + 1), so this is the single
                # switch point.
                if envs.SGLANG_WL_H2D_PREFETCH.get():
                    _wl_stage = 2 * _wl_stage + 1
                _wl_phys = int(self.max_total_num_tokens)
                if _wl_phys <= 2 * _wl_stage:
                    raise ValueError(
                        "weightless-KV host spill: the profiled device pool "
                        f"({_wl_phys} tokens/rank) is too small to carve out a "
                        f"{_wl_stage}-slot staging region (need > "
                        f"{2 * _wl_stage}). Lower "
                        "--weightless-kv-chunked-block-size or free VRAM."
                    )
                self._wl_spill_phys_tokens = _wl_phys
                self._wl_spill_staging_tokens = _wl_stage
                self._wl_spill_device_tokens = _wl_phys - _wl_stage
                _wl_cap = int(
                    getattr(self.server_args, "weightless_kv_spill_device_cap", 0) or 0
                )
                if 0 < _wl_cap < self._wl_spill_device_tokens:
                    # Cap the device-resident KV even though VRAM could hold
                    # more (identical on every rank -- this is a server arg,
                    # so the static slot->tier map stays rank-uniform). The
                    # PHYSICAL pool shrinks to cap + staging as well, so the
                    # VRAM actually saved is real (forces host streaming for
                    # any context past the cap; also the byte-parity test
                    # knob).
                    logger.warning(
                        "Weightless-KV host spill: DEVICE-CAP active -- "
                        "allocatable device slots clamped %d -> %d, physical "
                        "pool %d -> %d tokens (forces host streaming).",
                        self._wl_spill_device_tokens,
                        _wl_cap,
                        self._wl_spill_phys_tokens,
                        _wl_cap + _wl_stage,
                    )
                    self._wl_spill_device_tokens = _wl_cap
                    self._wl_spill_phys_tokens = _wl_cap + _wl_stage
                self._wl_spill_host_tokens = _wl_spill
                self.max_total_num_tokens = self._wl_spill_device_tokens + _wl_spill
                # ---- Stage B2: per-rank pool shrink -------------------------
                # Under the even-modulo DCP owner rule a single global sequence
                # of length L occupies only L // dcp_size PER-RANK slots (each
                # rank owns 1/dcp_size of the tokens). The GLOBAL addressable
                # KV is therefore max_total_num_tokens * dcp_size -- the
                # allocator index space is already built at that size
                # (dcp_alloc_size = max_total * cp_token_split_factor). But the
                # single-request admission cap (max_req_len = min(context_len,
                # max_token_pool_size)) reads the PER-RANK pool, so stock
                # weightless caps one sequence at max_total (1/dcp_size of what
                # physically fits) -- a dcp_size x under-utilization. Expose the
                # GLOBAL capacity for that cap so a single over-VRAM sequence
                # can span the whole tiered pool: to serve context C the per-
                # rank host tier only needs ~C/dcp_size slots, not C (this is
                # the "cheap 3x win" -- for 262144 on dcp=3, H drops from ~222k
                # to ~47k tokens/rank). See max_token_pool_size.
                self._wl_spill_global_capacity = (
                    self.max_total_num_tokens * self.dcp_size
                )
                logger.info(
                    "Weightless-KV host spill (B1/B2): profiled device pool %d "
                    "tokens/rank -> %d allocatable device slots + %d staging "
                    "slots + %d HOST slots; per-rank KV pool %d tokens, GLOBAL "
                    "single-sequence capacity %d tokens (x dcp_size=%d).",
                    _wl_phys,
                    self._wl_spill_device_tokens,
                    _wl_stage,
                    _wl_spill,
                    self.max_total_num_tokens,
                    self._wl_spill_global_capacity,
                    self.dcp_size,
                )

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
                        enable_mamba_extra_buffer_lazy=self.server_args.enable_mamba_extra_buffer_lazy(),
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
                    self._dcp_token_sharded_pool_rows(self.max_total_num_tokens),
                    page_size=self.page_size,
                    dtype=self.kv_cache_dtype,
                    head_num=self._pool_kv_head_num(),
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
                    head_num=self._pool_kv_head_num(),
                    head_dim=self.model_config.head_dim,
                    swa_attention_layer_ids=self.model_config.swa_attention_layer_ids,
                    full_attention_layer_ids=self.model_config.full_attention_layer_ids,
                    device=self.device,
                    enable_memory_saver=self.server_args.enable_memory_saver,
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
                    self._dcp_token_sharded_pool_rows(self.max_total_num_tokens),
                    page_size=self.page_size,
                    dtype=self.kv_cache_dtype,
                    head_num=self._pool_kv_head_num(),
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
                if self.is_hybrid_swa_compress or self._swa_hybrid_dcp_lane():
                    kwargs = {
                        # Plan-aware per-rank SWA kv-head count (uneven TP);
                        # falls back to max(1, swa_kv_heads // tp) without a
                        # plan. Under the SWA-hybrid DCP lane (#96) this is
                        # passed for ANY hybrid model, not only the compress
                        # archs: the two sub-pools then genuinely differ (full =
                        # replicated total kv heads, swa = this rank's shard),
                        # so the SWA sub-pool must state its own count.
                        "swa_head_num": self.model_config.get_swa_num_kv_heads(
                            get_parallel().attn_tp_size
                        ),
                        "swa_head_dim": self.model_config.swa_head_dim,
                        "swa_v_head_dim": self.model_config.swa_v_head_dim,
                        "v_head_dim": self.model_config.v_head_dim,
                    }
                # SWA-HYBRID UNEVEN DCP (#96, Stage B). Only the GLOBAL
                # full-attention layers are token-sharded:
                #   full sub-pool: rows = this rank's owned share of the global
                #     context C, each row carrying ALL total_num_kv_heads
                #     (the attention write gathers them) -- exactly the
                #     HybridLinearKVPool treatment a few branches above;
                #   swa sub-pool: UNCHANGED. Every rank holds every in-window
                #     position of its own kv-head shard, so its size stays the
                #     rank-local window-bounded cap and its head count stays the
                #     per-rank SWA shard. Sharding the window instead was
                #     measured against and rejected (see #91 section 4).
                _swa_full_size = self.full_max_total_num_tokens
                _swa_full_head_num = self._pool_kv_head_num()
                if self._swa_hybrid_dcp_lane():
                    from sglang.srt.distributed.utils import (
                        cp_token_split_factor,
                        get_cp_token_ratios,
                    )
                    from sglang.srt.layers.dcp.owner import dcp_compact_pool_rows

                    _swa_S = cp_token_split_factor(self.dcp_size)
                    _swa_ratio_r = get_cp_token_ratios()[get_parallel().attn_dcp_rank]
                    _swa_full_size = dcp_compact_pool_rows(
                        self.full_max_total_num_tokens, _swa_S, _swa_ratio_r
                    )
                    _swa_full_head_num = self.model_config.get_total_num_kv_heads()
                    logger.info(
                        "SWA-hybrid uneven DCP (#96): full sub-pool %d rows "
                        "(global context %d, S=%d, ratio %d) x %d replicated kv "
                        "heads for %d global layers; swa sub-pool %d tokens x %d "
                        "kv heads for %d window layers (unsharded).",
                        _swa_full_size,
                        self.full_max_total_num_tokens,
                        _swa_S,
                        _swa_ratio_r,
                        _swa_full_head_num,
                        len(self.model_config.full_attention_layer_ids),
                        self.swa_max_total_num_tokens,
                        kwargs["swa_head_num"],
                        len(self.model_config.swa_attention_layer_ids),
                    )
                self.token_to_kv_pool = SWAKVPool(
                    size=_swa_full_size,
                    size_swa=self.swa_max_total_num_tokens,
                    page_size=self.page_size,
                    dtype=self.kv_cache_dtype,
                    head_num=_swa_full_head_num,
                    head_dim=self.model_config.head_dim,
                    swa_attention_layer_ids=self.model_config.swa_attention_layer_ids,
                    full_attention_layer_ids=self.model_config.full_attention_layer_ids,
                    device=self.device,
                    enable_memory_saver=self.server_args.enable_memory_saver,
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
                    size=self._dcp_token_sharded_pool_rows(self.max_total_num_tokens),
                    page_size=self.page_size,
                    dtype=self.kv_cache_dtype,
                    index_dtype=self.dtype,
                    head_num=self._pool_kv_head_num(),
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
                #
                # #108 --draft-kv-layout dcp opts the draft pool INTO the same
                # treatment. The predicate lives in layers/dcp/owner.py because
                # the attention backend reads the identical one -- pool and
                # backend must never disagree about whether these rows are
                # token-sharded (#345's right-token/wrong-slot class).
                from sglang.srt.layers.dcp.owner import draft_pool_is_replicated

                _draft_non_dcp = draft_pool_is_replicated(
                    self.is_draft_pool_worker, self.server_args
                )
                # Weightless-KV fast lane (Option-B): the head rank projects the
                # FULL kv-heads (built under the weight-TP=1 override) and
                # broadcasts them; every rank writes all total_num_kv_heads to
                # its owned token slots. So the full-attention KV pool must store
                # the FULL total_num_kv_heads, exactly like the uneven-DCP
                # replicated-KV geometry -- not the ÷attn_tp_size per-rank share
                # (which would mismatch the head's broadcast at the attention
                # dispatch). weightless has rank_tp_ratio=None so
                # uneven_dcp_kv_replicated() is False; add it explicitly.
                if (
                    uneven_dcp_kv_replicated(self.dcp_size) or weightless_kv_active()
                ) and not _draft_non_dcp:
                    _hybrid_kv_head_num = self.model_config.get_total_num_kv_heads()
                else:
                    _hybrid_kv_head_num = self._pool_kv_head_num()
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
                    # This rank's OWNED rows for the global context budget.
                    # dcp_compact_pool_rows carries the ceil-to-a-whole-owner-
                    # block rule and the reason for it; it is shared with the
                    # SWA-hybrid full sub-pool (#96) so the two pool families
                    # cannot drift on a sizing rule whose off-by-one already
                    # cost one out-of-bounds-scatter debugging round.
                    from sglang.srt.layers.dcp.owner import dcp_compact_pool_rows

                    _hybrid_pool_size = dcp_compact_pool_rows(
                        self.max_total_num_tokens, _S, _ratio_r
                    )
                    # #297 fitted ceiling: with --kv-reshard-vectors set, the
                    # pool must hold this rank's rows under EVERY declared
                    # vector, so a later phase-boundary reshard never grows
                    # the pool (stable addresses -> decode CUDA graphs stay
                    # valid without recapture). The context budget C was
                    # already min-ruled over the same set in
                    # _apply_token_constraints, so this maximum is funded.
                    if getattr(self.server_args, "kv_reshard_vectors", None):
                        from sglang.srt.layers.dcp.reshard_plan import (
                            reshard_ceiling_rows,
                            reshard_vector_set,
                        )

                        _reshard_vecs = reshard_vector_set(
                            self.server_args.kv_reshard_vectors,
                            self.dcp_size,
                            tuple(_ratios),
                        )
                        _ceiling = reshard_ceiling_rows(
                            self.max_total_num_tokens,
                            _reshard_vecs,
                            get_parallel().attn_dcp_rank,
                        )
                        if _ceiling != _hybrid_pool_size:
                            logger.info(
                                "#297 fitted ceiling: full-attention pool "
                                "rows %d -> %d (vector set %s)",
                                _hybrid_pool_size,
                                _ceiling,
                                _reshard_vecs,
                            )
                            _hybrid_pool_size = _ceiling
                elif getattr(self, "_wl_spill_phys_tokens", 0):
                    # Weightless-KV B1 host spill: the DEVICE tensors are sized
                    # to the PROFILED capacity D (device slots + staging), NOT
                    # to the enlarged logical slot space (device + HOST slots).
                    # Host-region slots never reach the device scatter -- the
                    # owner-write / block-stage paths redirect them.
                    _hybrid_pool_size = self._wl_spill_phys_tokens
                else:
                    _hybrid_pool_size = self.max_total_num_tokens
                # kv-session-offload (S1): append ONE scratch row to the
                # physical full-attention pool (never handed out by the
                # allocator). The spill tick's owner-write quantizes the new
                # token's K/V into this row via the stock set_kv_buffer path
                # (byte-identical to a device write) and then D2H-copies it
                # into the session host pool. A few KB of VRAM; flag-gated.
                if (
                    self.server_args.enable_kv_session_offload
                    and not self.is_draft_worker
                ):
                    # Pool rows = size + page_size, and slot ids can reach
                    # `size` (page 0 is the dummy): with size inflated by 1
                    # the very last row (old_size + 1) is reachable by
                    # NEITHER the allocator (<= old_size) nor the DCP
                    # compaction (< physical share) -- that row is the
                    # scratch.
                    self._kv_sess_scratch_slot = _hybrid_pool_size + 1
                    _hybrid_pool_size += 1
                    # PS2 (deep prefill-spill): a born-spilled EXTEND allocates
                    # NO device KV slots, so its whole chunk is quantised
                    # through a STAGING CARVE appended right after the scratch
                    # row (also never handed out by the allocator) and then
                    # copied D2H into the session's host region. Sized RANK-
                    # UNIFORMLY from replicated config (chunk size, S,
                    # max_ratio) so every rank reserves the same rows even
                    # though their owned fill levels differ (S3b.4 item 3).
                    # Flag-gated on --kv-session-offload-prefill: without it
                    # not a single row is reserved.
                    if getattr(self.server_args, "kv_session_offload_prefill", False):
                        from sglang.srt.managers.kv_session_offload import (
                            prefill_stage_tokens,
                        )

                        _chunk = int(
                            self.server_args.chunked_prefill_size
                            or self.max_total_num_tokens
                        )
                        if _chunk <= 0:
                            _chunk = int(self.max_total_num_tokens)
                        if uneven_dcp_active(self.dcp_size) and not _draft_non_dcp:
                            _stage_S = _S
                            _stage_ratio = max(get_cp_token_ratios())
                        else:
                            _stage_S = max(1, self.dcp_size)
                            _stage_ratio = 1
                        _stage = prefill_stage_tokens(_chunk, _stage_S, _stage_ratio)
                        self._kv_sess_prefill_stage_base = _hybrid_pool_size + 1
                        self._kv_sess_prefill_stage_tokens = _stage
                        _hybrid_pool_size += _stage
                        logger.info(
                            "kv-session-offload PS2 (prefill-spill): device "
                            "staging carve of %d rows reserved at slot %d "
                            "(chunk=%d, S=%d, max_ratio=%d).",
                            _stage,
                            self._kv_sess_prefill_stage_base,
                            _chunk,
                            _stage_S,
                            _stage_ratio,
                        )
                # #330 --enable-vram-dial: reserve the VA upper bound for the
                # best declared vector's ceiling; physically back only the
                # boot fitted ceiling right after construction. Addresses are
                # reserved once, so later runtime grow/shrink never moves a
                # tensor and captured CUDA graphs keep replaying.
                _dial_initial_rows = None
                _dial_reserve = _hybrid_pool_size
                _dial_chunk = None
                if getattr(self.server_args, "enable_vram_dial", False):
                    from sglang.srt.managers.vram_dial import (
                        get_boot_capacity_plan,
                    )

                    if envs.SGLANG_USE_HND_KVCACHE.get():
                        raise RuntimeError(
                            "--enable-vram-dial does not support the HND KV "
                            "layout (row-slice zeroing and span math assume "
                            "slot-major rows)."
                        )
                    _plan = get_boot_capacity_plan()
                    if _plan is None:
                        raise RuntimeError(
                            "--enable-vram-dial: no boot capacity plan was "
                            "recorded; the uneven-DCP token sizing path did "
                            "not run for this configuration. The dial "
                            "requires weighted uneven DCP (see DESIGN_330 "
                            "section 7)."
                        )
                    if _draft_non_dcp:
                        _dial_reserve = max(_plan.max_cap, _hybrid_pool_size)
                    else:
                        _dial_reserve = max(
                            _plan.reserve_rows_for_rank(get_parallel().attn_dcp_rank),
                            _hybrid_pool_size,
                        )
                    _dial_initial_rows = _hybrid_pool_size
                    _dial_chunk = envs.SGLANG_VRAM_DIAL_CHUNK_MIB.get() << 20
                    logger.info(
                        "#330 vram-dial pool lane (%s): VA reserve %d rows, "
                        "boot backing %d rows, commit chunk %d MiB",
                        "draft" if _draft_non_dcp else "target",
                        _dial_reserve,
                        _dial_initial_rows,
                        _dial_chunk >> 20,
                    )
                # #704b B1: token-share sizing instead of layer-ownership
                # sizing. Returns (None, _dial_reserve) unless explicitly
                # enabled, so the default build below is untouched.
                _b1_ids, _b1_size = self._decoupled_kv_pool_override(
                    config.full_attention_layer_ids, _dial_reserve
                )
                from sglang.srt.distributed.utils import (
                    stage_owned_layer_ids,
                )

                self.token_to_kv_pool = HybridLinearKVPool(
                    page_size=self.page_size,
                    size=_b1_size,
                    dtype=self.kv_cache_dtype,
                    head_num=_hybrid_kv_head_num,
                    head_dim=self.model_config.head_dim,
                    # if draft worker, we only need 1 attention layer's kv pool.
                    # A dual-group lane runner is constructed as a draft worker
                    # (secondary-runner gates) but runs the FULL model: it
                    # needs every full-attention layer's kv pool.
                    full_attention_layer_ids=(
                        _b1_ids
                        if _b1_ids is not None
                        else [0]
                        # A dual-group lane TARGET and the #631 phase-flip
                        # TP stack are draft-gated secondary runners that
                        # run the FULL model: every full-attention layer.
                        if self.is_draft_pool_worker
                        and not getattr(self, "is_dual_group_lane_target", False)
                        else stage_owned_layer_ids(
                            config.full_attention_layer_ids,
                            self.start_layer,
                            self.end_layer,
                        )
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
                    post_capture_active=(
                        self.post_capture_kv_active or _dial_initial_rows is not None
                    ),
                    vmm_commit_chunk_bytes=_dial_chunk,
                    # #631: under the phase flip the two layouts' KV pools
                    # must be able to hold physical pages EXCLUSIVELY --
                    # otherwise both are resident for process life and each
                    # can only be sized against half the per-rank budget.
                    # This is the VA-backed allocation only; sizing is
                    # unchanged (post-capture sizing stays gated off here).
                    # Both stacks must qualify, and they answer to DIFFERENT
                    # signals: derive_tp_stack_server_args deliberately
                    # clears enable_phase_flip on the TP copy (it DESCRIBES
                    # a TP stack, it does not enable a nested flip), so the
                    # flag alone catches only the PP side and the TP pool
                    # would come up unswappable -- which is exactly how this
                    # first failed on metal.
                    swappable_backing=bool(
                        self.server_args.enable_phase_flip
                        or getattr(self, "is_phase_flip_tp_stack", False)
                    ),
                    **extra_args,
                )
                if _dial_initial_rows is not None:
                    from sglang.srt.managers.vram_dial import (
                        register_dial_participant,
                    )

                    self.token_to_kv_pool.initial_backing_rows(_dial_initial_rows)
                    register_dial_participant(
                        self,
                        self.token_to_kv_pool,
                        is_target=not _draft_non_dcp,
                        reserved_tokens=_dial_reserve,
                    )
                if getattr(self, "_wl_spill_phys_tokens", 0):
                    self._wl_attach_spill_host_pool()
                if getattr(self, "_kv_sess_scratch_slot", None) is not None:
                    self._kv_sess_attach_host_pool()
            else:
                if is_float4_e2m1fn_x2(self.kv_cache_dtype):
                    assert not enable_page_major, (
                        "page-major KV layout is not supported with fp4 KV cache"
                    )
                    self.token_to_kv_pool = MHATokenToKVPoolFP4(
                        self._dcp_token_sharded_pool_rows(self.max_total_num_tokens),
                        page_size=self.page_size,
                        dtype=self.kv_cache_dtype,
                        head_num=self._pool_kv_head_num(),
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
                        self._dcp_token_sharded_pool_rows(self.max_total_num_tokens),
                        page_size=self.page_size,
                        dtype=self.kv_cache_dtype,
                        head_num=self._pool_kv_head_num(),
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
                        #
                        # #487 REACH AUDIT (no behaviour change here). This
                        # branch inflates BOTH the index space and the page
                        # granularity by dcp_size, i.e. it assumes the pool
                        # behind it is token-sharded across the DCP group. A
                        # draft worker at --draft-kv-layout replicated (the
                        # default) has the opposite geometry -- full token
                        # context, head-sharded -- which the pool sizing above
                        # knows (it guards on draft_pool_is_replicated) and
                        # this selection does not mention at all. #108 never
                        # audited the combination.
                        # On CUDA the combination is UNREACHABLE, by two real
                        # predicates rather than by luck -- one per producer of
                        # is_draft_worker=True. (1) A SPECULATIVE draft worker:
                        # ServerArgs._handle_dcp_validation refuses dcp_size>1
                        # + speculation on CUDA unless the boot is either
                        # uneven-weighted DCP (requires rank_tp_ratio) or the
                        # weightless-KV fast lane -- which are exactly the two
                        # disjuncts of the gate above, so such a boot always
                        # takes the branch above this one. (2) A #274
                        # DUAL-GROUP LANE runner, which also sets
                        # is_draft_worker=True and is NOT speculative, so (1)
                        # does not cover it: _lane_server_args_view forces
                        # view.dcp_size = 1, so a lane never enters this chain,
                        # and at dcp_size==1 both multipliers here are
                        # identities anyway.
                        # On HIP/ROCm that refusal does not run (the validator
                        # returns at `if is_hip(): return` before the CUDA
                        # branch), so there the combination IS admitted and
                        # this branch would size an allocator for a token
                        # split the replicated draft pool does not perform.
                        # Left as a named residual: this fork does not serve
                        # ROCm and the audit had no ROCm hardware, and a
                        # desk-guessed change to an address computation is the
                        # #345 right-token/wrong-slot class waiting to happen.
                        # Pinned by
                        # test/registered/unit/distributed/
                        # test_stock_dcp_allocator_reach_487.py
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

        # Multi-group runtime (#274): a dual-group lane runner sizes RANK-LOCAL
        # by contract. Its scoped server_args view makes both sync predicates
        # below false already; this guard makes the contract independent of
        # that view (a ReduceOp.MIN here would hang the serving group, the
        # slice-A finding this gate exists for).
        if getattr(self, "is_dual_group_lane", False):
            return self._apply_hybrid_kv_token_cap(
                token_capacity, hybrid_cap, hybrid_cap_kind
            )

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

        if uneven_dcp_active(self.dcp_size) and get_world_group().world_size > 1:
            ratios = get_cp_token_ratios()
            split_factor = cp_token_split_factor(self.dcp_size)
            ratio_r = ratios[get_parallel().attn_dcp_rank]
            local_unit = int(token_capacity) // int(ratio_r)
            if getattr(self.server_args, "kv_reshard_vectors", None):
                # #297 fitted ceiling: C must fit EVERY declared reshard
                # vector on EVERY rank, not only the boot vector. One
                # min-reduction over the per-vector local units (the flag is
                # group-uniform, so the payload shape is too), then the
                # binding vector decides C. Flag unset keeps the original
                # single-scalar reduce below, byte-identical.
                from sglang.srt.layers.dcp.reshard_plan import reshard_vector_set

                _vecs = reshard_vector_set(
                    self.server_args.kv_reshard_vectors,
                    self.dcp_size,
                    tuple(ratios),
                )
                _rank = get_parallel().attn_dcp_rank
                _units = torch.tensor(
                    [int(token_capacity) // int(v[_rank]) for v in _vecs],
                    dtype=torch.int64,
                )
                torch.distributed.all_reduce(
                    _units,
                    op=torch.distributed.ReduceOp.MIN,
                    group=get_world_group().cpu_group,
                )
                _caps = [int(u) * sum(v) for u, v in zip(_units.tolist(), _vecs)]
                token_capacity = min(_caps)
                _binding = _vecs[_caps.index(token_capacity)]
                logger.info(
                    "#297 fitted-ceiling token sizing: candidate capacities "
                    "%s for vectors %s -> C = %d (binding vector %s)",
                    _caps,
                    _vecs,
                    token_capacity,
                    _binding,
                )
                if getattr(self.server_args, "enable_vram_dial", False):
                    # #330: record the PER-VECTOR achievable ceilings (each
                    # clamped by the same external caps as C itself) -- the
                    # VA reservation and the runtime C re-raise both derive
                    # from them.
                    from sglang.srt.managers.vram_dial import (
                        set_boot_capacity_plan,
                    )

                    _dial_caps = {}
                    for _u, _v in zip(_units.tolist(), _vecs):
                        _cv = int(_u) * sum(_v)
                        if user_limit is not None:
                            _cv = min(_cv, user_limit)
                        if hybrid_cap is not None:
                            _cv = min(_cv, hybrid_cap)
                        _dial_caps[tuple(_v)] = _cv
                    set_boot_capacity_plan(_vecs, _dial_caps)
                if user_limit is not None:
                    token_capacity = min(token_capacity, user_limit)
                return self._apply_hybrid_kv_token_cap(
                    token_capacity, hybrid_cap, hybrid_cap_kind
                )
            local_blocks = torch.tensor(local_unit, dtype=torch.int64)
            torch.distributed.all_reduce(
                local_blocks,
                op=torch.distributed.ReduceOp.MIN,
                group=get_world_group().cpu_group,
            )
            token_capacity = int(local_blocks.item()) * split_factor
            # Sizing-chain evidence log (T156 VRAM underfill diagnosis): which
            # rank binds the unit, and whether the hybrid cap clamps after it.
            logger.info(
                "Uneven-DCP token sizing: rank %d local capacity %d tokens / "
                "ratio %d = unit %d; min-reduced unit %d -> global "
                "max_total_num_tokens %d (vector %s, hybrid %s cap %s).",
                self.tp_rank,
                int(token_capacity) if False else local_unit * int(ratio_r),
                int(ratio_r),
                local_unit,
                int(local_blocks.item()),
                token_capacity,
                ratios,
                hybrid_cap_kind,
                hybrid_cap,
            )
            if user_limit is not None:
                token_capacity = min(token_capacity, user_limit)
            token_capacity = self._apply_hybrid_kv_token_cap(
                token_capacity, hybrid_cap, hybrid_cap_kind
            )
            if getattr(self.server_args, "enable_vram_dial", False):
                # #330 without --kv-reshard-vectors: the plan holds only the
                # boot vector (the dial works; the cross-vector re-raise has
                # nothing to raise to).
                from sglang.srt.managers.vram_dial import set_boot_capacity_plan

                set_boot_capacity_plan(
                    [tuple(ratios)], {tuple(ratios): int(token_capacity)}
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
            if self.pp_size > 1:
                # #201 slice 3 (world-MIN by construction): apply every
                # PER-RANK cap BEFORE the world reduce. The hybrid #79/#90
                # ceilings are computed rank-locally; applied only after the
                # reduce (the old order) a stage whose cap undercuts the
                # agreed value would silently end below it -- stage 0 then
                # admits tokens another stage's pool cannot hold. With the
                # caps folded in first, the MIN below is the final word and
                # the post-reduce re-application is a no-op. pp_size == 1
                # keeps the stock order byte-identically.
                token_capacity = self._apply_hybrid_kv_token_cap(
                    token_capacity, hybrid_cap, hybrid_cap_kind
                )
            local_capacity = int(token_capacity)
            tensor = torch.tensor(token_capacity, dtype=torch.int64)
            torch.distributed.all_reduce(
                tensor,
                op=torch.distributed.ReduceOp.MIN,
                group=get_world_group().cpu_group,
            )
            token_capacity = tensor.item()
            # #127 sizing evidence. Under the even-modulo owner rule the slot
            # space is rank-uniform, so this MIN is what turns per-rank
            # budgets into ONE capacity: the group gets dcp_size x the
            # SMALLEST rank's token capacity, and every other rank's surplus
            # slots are stranded. Anyone changing a rank's per-token bytes
            # (e.g. --weightless-kv-worker-cache-dtype fp8_e5m2) needs to know
            # whether THIS rank is the one that binds -- otherwise the change
            # buys nothing. Each rank logs its own line; the binding rank is
            # the one whose local == agreed.
            agreed = int(token_capacity)
            stranded = local_capacity - agreed
            logger.info(
                "KV token sizing: rank %d local capacity %d tokens, "
                "min-reduced across ranks to %d (%s; %d stranded on this "
                "rank). Global addressable KV = %d x dcp_size(%d).",
                self.tp_rank,
                local_capacity,
                agreed,
                "THIS RANK BINDS" if stranded == 0 else "another rank binds",
                stranded,
                agreed,
                self.dcp_size,
            )

        token_capacity = self._apply_hybrid_kv_token_cap(
            token_capacity, hybrid_cap, hybrid_cap_kind
        )
        if self.pp_size > 1:
            self._assert_pp_world_kv_capacity_agreement(int(token_capacity))
        return token_capacity

    def _assert_pp_world_kv_capacity_agreement(
        self: ModelRunner, token_capacity: int
    ) -> None:
        """Prove -- do not assume -- that every rank of the pipeline world
        resolved the SAME token capacity (#201 slice 3).

        max_total_num_tokens is the admission currency: a request's tokens
        occupy KV on EVERY stage (each in its own layers), so the ceiling is
        only meaningful as a world minimum. The reduce above establishes it;
        this check makes any future rank-local adjustment AFTER the reduce
        (a re-ordered cap, a new clamp) fail the boot loudly instead of
        letting stage 0 admit tokens another stage cannot hold.

        Collective discipline: gated on pp_size (world-uniform by
        construction) and world_size only -- every rank participates in the
        gather or none does.
        """
        if get_world_group().world_size <= 1:
            return
        gathered: list = [None] * get_world_group().world_size
        torch.distributed.all_gather_object(
            gathered,
            (int(self.pp_rank), int(self.tp_rank), int(token_capacity)),
            group=get_world_group().cpu_group,
        )
        distinct = {cap for (_, _, cap) in gathered}
        if len(distinct) > 1:
            per_rank = ", ".join(
                f"pp{pp}/tp{tp}={cap}" for (pp, tp, cap) in sorted(gathered)
            )
            raise RuntimeError(
                "Pipeline KV world agreement violated: the stages resolved "
                f"different max_total_num_tokens ({per_rank}). Admission "
                "would follow stage 0 while another stage's pool is smaller "
                "-- a guaranteed overflow. This means a rank-local cap or "
                "clamp ran AFTER the world MIN-reduce; fold it in before "
                "the reduce (see _apply_token_constraints)."
            )

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
        suggestion = suggest_unit_rebalance_multi(free_bytes, bytes_per_token, families)
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
                "uneven TP: restart with %s to raise the KV pool from %d to ~%d tokens",
                assignments,
                cur_min_tokens,
                projected,
            )

    def _is_solo_draft_kv_host(self: ModelRunner) -> bool:
        """True on the TARGET runner of the draft-solo host rank — the only
        rank that allocates a draft KV pool sized to the GLOBAL context.
        Mirrors pool_configurator.solo_draft_kv_cell_factor's conditions.

        T156 VRAM-underfill fix (2026-07-22): under
        --speculative-cross-algorithm the DFLASH rung's draft pool is ALWAYS
        solo on the solo rank even when the GLOBAL placement is 'split'
        (NEXTN is the primary shape) — solo_draft_kv_cell_factor already
        knew this, but this detector did not, so the capacity-mode install
        skipped the solo fixed-point correction, treated the host's profiled
        capacity as vector-independent, overshot the host's ownership share
        (its cell inflates with sum(ratios)/ratio_host), and the min-rule
        clawed max_total_num_tokens back from the predicted ~628k to 215k
        while the shadow ranks idled at 13.4/20.5 GiB (measured on the
        tp3-uneven cross-auto boot; evidence: 'Uneven-DCP token sizing'
        rank-0 unit 3367 vs rank-1/2 units 10143/10493)."""
        server_args = self.server_args
        if self.is_draft_worker:
            return False
        if not getattr(server_args, "speculative_draft_solo_active", None):
            return False
        cross_algo = getattr(server_args, "speculative_cross_algorithm", False)
        if not server_args.speculative_draft_solo_active() and not cross_algo:
            return False
        return self.tp_rank == server_args.speculative_draft_solo_rank()

    def _solo_host_capacity_curve(
        self: ModelRunner, budget_bytes: int, active: list
    ) -> Optional[Tuple[float, float]]:
        """Draft-solo host only: the (alpha, beta) of its capacity curve

            1 / P(g) = alpha + beta * g,   g = sum(ratios) / ratio_host

        Returns None on every other rank / placement (non-solo stays untouched).

        WHY this exists: for a normal rank the profiled token capacity P_r is
        INDEPENDENT of the token-ownership vector, which is what lets
        _maybe_suggest_dcp_token_vector feed the measured P_r straight into
        partition_units. On the draft-solo host it is not: its draft KV pool
        spans the GLOBAL context C, so the per-token cell carries a draft term
        scaled by g = C / n_host = S / ratio_host (see
        pool_configurator.solo_draft_kv_cell_factor). Shrinking the host's
        ownership share therefore SHRINKS its own capacity too — installing
        partition_units(64, [P_measured...]) raw would hand the host a share it
        can no longer fund, and the min-rule would claw the global pool back.

        The cell is affine in g, so 1/P is affine in g and two evaluations of
        the real configurator pin it down exactly — no duplicated cell-size
        math, and every architecture-specific configurator (Default, HybridSWA,
        DSV4, ...) is covered by construction."""
        if not self._is_solo_draft_kv_host():
            return None
        from sglang.srt.distributed.utils import (
            get_cp_token_ratios,
            set_cp_token_ratios,
        )
        from sglang.srt.model_executor.pool_configurator import (
            create_memory_pool_configurator,
        )

        dcp_rank = int(get_parallel().attn_dcp_rank)
        if not (0 <= dcp_rank < len(active)):
            return None

        def probe(vector: list) -> Tuple[float, float]:
            saved = get_cp_token_ratios()
            try:
                set_cp_token_ratios(list(vector))
                conf = create_memory_pool_configurator(self)
                p = int(
                    conf.calculate_pool_sizes(
                        budget_bytes, self.page_size
                    ).max_total_num_tokens
                )
            finally:
                set_cp_token_ratios(saved)
            return float(p), float(sum(vector)) / float(vector[dcp_rank])

        # Two points on the curve: the active vector, and a synthetic one that
        # squeezes the host's share to the minimum (large g). Purely arithmetic
        # probes — nothing is allocated and the installed vector is restored.
        probe_vec = [1 if r == dcp_rank else 8 for r in range(len(active))]
        p1, g1 = probe(list(active))
        p2, g2 = probe(probe_vec)
        if p1 <= 0 or p2 <= 0 or g1 == g2:
            return None
        beta = (1.0 / p2 - 1.0 / p1) / (g2 - g1)
        alpha = 1.0 / p1 - beta * g1
        # Sizing-chain evidence log (T156 VRAM underfill diagnosis).
        logger.info(
            "Draft-solo capacity curve: P(g=%.3f)=%d, P(g=%.3f)=%d -> "
            "alpha=%.3e beta=%.3e (budget %.2f GB).",
            g1,
            int(p1),
            g2,
            int(p2),
            alpha,
            beta,
            budget_bytes / (1 << 30),
        )
        if not (beta > 0.0) or not (alpha + beta > 0.0):
            # No draft-KV term (beta == 0) or a degenerate fit: the capacity is
            # vector-independent after all, so the raw measurement is correct.
            return None
        return (alpha, beta)

    @staticmethod
    def _solo_fixed_point_capacity(
        curve: Tuple[float, float], q_tokens: int
    ) -> Optional[int]:
        """Solve the draft-solo host's self-consistent capacity.

        With capacity-proportional ownership the host's share satisfies
        ``ratio_host / S = p_h / (p_h + Q)`` (Q = the shadows' total capacity,
        which IS vector-independent), i.e. ``g = (p_h + Q) / p_h``. Substituted
        into the curve ``1/p_h = alpha + beta * g``:

            p_h = (1 - beta * Q) / (alpha + beta)

        which is the measured-quantity twin of the planner's predicted
        ``_solo_rank_token_capacity`` closed form. Clamped to >= 1 (the vector
        cannot give a rank zero units); the pool sizing's min-rule remains the
        binding safety net either way."""
        alpha, beta = curve
        denom = alpha + beta
        if denom <= 0.0:
            return None
        p_h = (1.0 - beta * float(q_tokens)) / denom
        if not math.isfinite(p_h):
            return None
        return max(int(p_h), 1)

    def _corridor_card_key(self: ModelRunner):
        """This rank's entry in --rank-gpu-id, i.e. the key every parse-time
        per-card number (#602 post-sizing demand, user reserve) is filed
        under. None when there is no explicit placement."""
        ids = getattr(self.server_args, "rank_gpu_id", None)
        if not ids:
            return None
        idx = self._rank_vector_index()
        if not 0 <= idx < len(ids):
            return None
        return ids[idx]

    def _read_corridor_card_free_bytes(self: ModelRunner):
        """NVML free bytes of the PHYSICAL card this rank runs on, or None.

        Only read in --rank-kv-ratio corridor: outside it nothing consumes
        the number, and an NVML round-trip on every boot is not free. The card
        is resolved through the registry's identity map (nvml.current_device_
        uuid), never by assuming the CUDA ordinal equals the NVML index --
        they differ on this rig.
        """
        if not corridor_mode_active(self.server_args):
            return None
        from sglang.srt.registry import nvml

        uuid = nvml.current_device_uuid()
        return int(nvml.memory_info_for_uuid(uuid).free_bytes)

    def _corridor_local_capacity(self: ModelRunner, configurator) -> int:
        """Q_r: the KV tokens this rank may hold and still leave its card at
        or above the operator's free-VRAM floor (#602).

        Deliberately expressed in the CONFIGURATOR's own currency rather than
        by dividing bytes by a cell size guessed here: the same function that
        turns a budget into a token count for P_r turns the corridor budget
        into Q_r, so the two are directly comparable and no second, drifting
        copy of the cell arithmetic exists.
        """
        from sglang.srt.distributed.corridor_vector import corridor_pool_bytes

        card = self._corridor_card_key()
        free_bytes = getattr(self, "_corridor_card_free_bytes", None)
        if card is None or free_bytes is None:
            raise ValueError(
                "--rank-kv-ratio corridor could not resolve this rank's "
                f"physical card (rank_gpu_id entry {card!r}, free reading "
                f"{free_bytes!r}). The floor is per card, so there is nothing "
                "to enforce it against; this refuses rather than sizing the "
                "pool as if no floor existed."
            )
        post_sizing = (self.server_args.corridor_post_sizing_mib or {}).get(card)
        if post_sizing is None:
            raise ValueError(
                f"--rank-kv-ratio corridor has no post-sizing demand for GPU "
                f"{card}. It is priced once at parse time "
                "(_handle_corridor_kv_ratio); its absence here means the "
                "worker is running a server_args this mode never validated."
            )
        ids = list(self.server_args.rank_gpu_id)
        colocated = ids.count(card)
        reserve_mib = self.server_args.user_reserve_mib_per_gpu(ids)[card]
        allow_bytes = corridor_pool_bytes(
            free_bytes,
            reserve_mib=reserve_mib,
            post_sizing_mib=post_sizing,
            colocated_ranks=colocated,
        )
        if allow_bytes <= 0:
            raise ValueError(
                f"--rank-kv-ratio corridor: GPU {card} cannot fund its "
                f"free-VRAM floor. NVML free at the post-weight-load "
                f"measuring point is {free_bytes >> 20} MiB; the operator "
                f"reserve is {reserve_mib} MiB and the demand that still has "
                f"to materialize on this card (graph capture, activation "
                f"peak, attention workspaces) is {post_sizing} MiB, across "
                f"{colocated} co-located rank(s). That leaves nothing for a "
                "KV pool. Lower --rank-user-reserve-mib, move a rank off "
                "this card, or shrink a demand term -- this refuses rather "
                "than allocating into the reserve."
            )
        tokens = int(
            configurator.calculate_pool_sizes(
                allow_bytes, self.page_size
            ).max_total_num_tokens
        )
        logger.info(
            "#602 corridor capacity (rank %d, GPU %d): NVML free %d MiB - "
            "reserve %d MiB - post-sizing demand %d MiB = %d MiB over %d "
            "co-located rank(s) -> %d MiB -> %d KV tokens for this rank.",
            self.tp_rank,
            card,
            free_bytes >> 20,
            reserve_mib,
            post_sizing,
            (free_bytes >> 20) - reserve_mib - post_sizing,
            colocated,
            allow_bytes >> 20,
            tokens,
        )
        return max(tokens, 0)

    def _maybe_suggest_dcp_token_vector(
        self: ModelRunner, budget_bytes: int, allow_install: bool = False
    ) -> None:
        """Uneven-DCP token-vector self-calibration (analogue of vLLM's
        VLLM_UNEVEN_TOKEN_VECTOR): after the rank-local KV profiling, derive
        the OPTIMAL token-axis split vector from each rank's ACTUAL profiled
        token capacity P_r (not the rough pre-boot budget estimate) and, if it
        differs from the active vector, log a restart hint.

        --rank-kv-ratio capacity (task #88, ``allow_install=True`` from the
        pre-pool sizing path only): INSTALL the measured optimal vector
        instead of hinting — one-boot convergence of the decoupled KV-token
        ownership. Safe exactly here because nothing has snapshotted the
        vector yet (pools, allocator, attention backends and CUDA graphs are
        all built afterwards); the post-capture resize pass must stay
        hint-only (the vector is frozen by then). Suppressed by an explicit
        pin (SGLANG_UNEVEN_TOKEN_VECTOR env or a --rank-kv-ratio vector) and
        for the draft worker. The install decision is rank-uniform (server
        args + the all-gathered P_r), so every rank installs the identical
        vector — the same determinism invariant as the hint path.

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

        # Draft-solo host: its measured capacity is a FUNCTION of the active
        # vector (its draft KV pool spans the GLOBAL context), so it cannot be
        # fed into the optimizer raw — see _solo_host_capacity_curve. The curve
        # is gathered with the capacities so every rank re-derives the identical
        # corrected value (rank-uniform decision). None on every non-solo path.
        curve = self._solo_host_capacity_curve(budget_bytes, active)

        # #602: Q_r rides the SAME collective as P_r. It is rank-local (this
        # rank's card, its free memory) and the vector must stay a pure
        # function of all-gathered values, so it cannot be recomputed
        # per-rank after the gather. The mode predicate is rank-uniform, so
        # either every rank contributes a Q or none does.
        corridor_mode = corridor_mode_active(self.server_args)
        local_q = self._corridor_local_capacity(configurator) if corridor_mode else None

        world = get_world_group().world_size
        payload = (
            int(get_parallel().attn_dcp_rank),
            int(local_p),
            curve,
            local_q,
        )
        gathered: list = [None] * world
        torch.distributed.all_gather_object(
            gathered, payload, group=get_world_group().cpu_group
        )
        # Order the capacities by DCP rank (the token vector is indexed by
        # attn_dcp_rank, which need not equal the global rank).
        p_by_rank = [0] * self.dcp_size
        q_by_rank: list = [None] * self.dcp_size
        solo_host_rank, solo_curve = None, None
        for entry in gathered:
            # Positional, tolerant unpack: the Q slot was appended to a
            # 3-tuple payload, and a gather that predates it is still a valid
            # no-corridor payload.
            dcp_rank, p_val, rank_curve = entry[0], entry[1], entry[2]
            q_val = entry[3] if len(entry) > 3 else None
            if 0 <= dcp_rank < self.dcp_size:
                p_by_rank[dcp_rank] = p_val
                q_by_rank[dcp_rank] = q_val
                if rank_curve is not None:
                    solo_host_rank, solo_curve = dcp_rank, rank_curve
        if any(p <= 0 for p in p_by_rank):
            return

        # ``p_measured`` is the truth UNDER THE ACTIVE VECTOR (what the pools
        # would be sized to right now). On the draft-solo host the capacity is
        # a FUNCTION of the vector (affine cell in g = S / ratio_host), so the
        # optimal vector cannot come from proportional partitioning of static
        # capacities — that assumption degenerates exactly when the host is
        # budget-poor (fixed point p_h <= 0 -> "capacity 1" -> c_optimal 64 ->
        # install skipped -> the shadows idle; measured 2026-07-22 on the
        # cross-auto tp3 boot). Instead, search the host share directly: for
        # every v_host in [1, S), the host holds P_h(g = S/v_host) tokens per
        # the measured curve, the other ranks split the remaining units
        # proportionally to their (vector-independent) capacities, and
        # C(v) = min_r(P_r // v_r) * S. S=64 matches partition_units'
        # granularity; the argmax is exact within it. All inputs are the
        # all-gathered capacities + curve, so every rank computes the same
        # argmax (rank-uniform install decision, unchanged invariant).
        p_measured = list(p_by_rank)

        def _context_budget(vector: list, capacities: list) -> int:
            return min(capacities[r] // vector[r] for r in range(self.dcp_size)) * sum(
                vector
            )

        c_active = _context_budget(active, p_measured)

        if corridor_mode:
            # #602. Two changes against 'capacity', and only two:
            #
            #   1. The capacity a rank is solved against is min(P_r, Q_r) --
            #      the budget model AND the measured floor, whichever binds.
            #      Because C(v) = min_r(cap_r // v_r) * sum(v) means rank r
            #      holds exactly unit * v_r <= cap_r tokens, clamping the
            #      capacity IS enforcing the floor. It stays a hard
            #      constraint and never becomes a term the token objective
            #      can trade away.
            #   2. The vector is solved exactly instead of by proportional
            #      rounding at a fixed grain (see solve_token_vector).
            #
            # c_active is re-scored against the same clamped capacities:
            # comparing a corridor-respecting candidate to an active vector
            # scored on the unclamped P_r would compare two different
            # feasible sets and could report a "regression" for the act of
            # coming back above the floor.
            from sglang.srt.distributed.corridor_vector import (
                CorridorInfeasible,
                RankCapacity,
                solve_corridor_vector,
            )

            if solo_curve is not None:
                raise ValueError(
                    "--rank-kv-ratio corridor does not cover draft-solo KV "
                    "placement: the solo host's token capacity is a FUNCTION "
                    "of the vector being solved for, so a fixed per-rank "
                    "capacity clamp is not the right constraint for it. Use "
                    "--rank-kv-ratio capacity for that topology."
                )
            if any(q is None for q in q_by_rank):
                raise ValueError(
                    "--rank-kv-ratio corridor: rank(s) "
                    f"{[r for r, q in enumerate(q_by_rank) if q is None]} "
                    "contributed no corridor capacity to the collective. The "
                    "mode predicate is rank-uniform, so this means the ranks "
                    "disagree about server_args -- refusing rather than "
                    "solving on a partial floor."
                )
            try:
                solution = solve_corridor_vector(
                    [
                        RankCapacity(r, p_by_rank[r], q_by_rank[r])
                        for r in range(self.dcp_size)
                    ]
                )
            except CorridorInfeasible as exc:
                raise ValueError(
                    f"--rank-kv-ratio corridor cannot satisfy the free-VRAM "
                    f"floor: {exc}. Profiled capacities {p_by_rank}, corridor "
                    f"capacities {q_by_rank} tokens per rank."
                ) from exc
            optimal = solution.vector
            c_optimal = solution.context_tokens
            if len(set(optimal)) == 1 and len(set(active)) != 1:
                # An all-equal vector is the EVEN-MODULO owner rule, not a
                # weighted one (uneven_dcp_active is False for it). Switching
                # owner rules is a different change from re-weighting one, and
                # nothing here has validated the even path's geometry for this
                # boot -- so the corridor keeps the weighted vector and says
                # the capacities have equalized.
                if self.tp_rank == 0:
                    logger.info(
                        "#602 corridor: the solved vector %s is uniform, "
                        "which would switch the owner rule from weighted to "
                        "even modulo. Keeping the active weighted vector %s "
                        "(capacities %s have equalized; pass an explicit "
                        "--rank-kv-ratio to change the owner rule).",
                        optimal,
                        active,
                        solution.capacities,
                    )
                return
            capped = [r for r in range(self.dcp_size) if q_by_rank[r] < p_by_rank[r]]
            c_active = _context_budget(active, solution.capacities)
            active_unit = min(
                solution.capacities[r] // active[r] for r in range(self.dcp_size)
            )
            waste_before = sum(
                solution.capacities[r] - active_unit * active[r]
                for r in range(self.dcp_size)
            )
            if self.tp_rank == 0:
                logger.info(
                    "#602 corridor solve: profiled capacity %s, corridor "
                    "capacity %s -> effective %s tokens per rank (floor binds "
                    "on rank(s) %s). Vector %s -> %s, max_total_num_tokens "
                    "%d -> %d, per-rank tokens %s, unallocated %s (was %s).",
                    p_by_rank,
                    q_by_rank,
                    solution.capacities,
                    capped or "none",
                    active,
                    optimal,
                    c_active,
                    c_optimal,
                    solution.per_rank_tokens,
                    solution.total_waste_tokens,
                    waste_before,
                )
        elif solo_curve is not None:
            alpha, beta = solo_curve
            grain = 64
            other_ranks = [r for r in range(self.dcp_size) if r != solo_host_rank]
            best_vec, best_c, best_ph = None, -1, 0
            # Every non-host rank needs >= 1 unit of the remaining grain.
            for v_host in range(1, grain - len(other_ranks) + 1):
                denom = alpha + beta * (float(grain) / v_host)
                if denom <= 0.0:
                    continue
                p_host = int(1.0 / denom)
                if p_host < v_host:
                    continue
                rest = partition_units(
                    grain - v_host, [p_by_rank[r] for r in other_ranks]
                )
                vec = [0] * self.dcp_size
                vec[solo_host_rank] = v_host
                for r, units in zip(other_ranks, rest):
                    vec[r] = units
                if any(u <= 0 for u in vec):
                    continue
                caps = list(p_by_rank)
                caps[solo_host_rank] = p_host
                c = _context_budget(vec, caps)
                if c > best_c:
                    best_vec, best_c, best_ph = vec, c, p_host
            if best_vec is not None:
                if self.tp_rank == 0:
                    logger.info(
                        "Draft-solo KV planning: solo host (DCP rank %d) "
                        "capacity is vector-dependent (measured curve "
                        "alpha=%.3e beta=%.3e); direct share search -> "
                        "vector %s (host holds %d tokens, predicted "
                        "max_total_num_tokens ~%d vs %d under the active "
                        "vector %s).",
                        solo_host_rank,
                        alpha,
                        beta,
                        best_vec,
                        best_ph,
                        best_c,
                        c_active,
                        active,
                    )
                g = math.gcd(*best_vec)
                optimal = [v // g for v in best_vec]
                c_optimal = best_c
            else:
                optimal = list(active)
                c_optimal = c_active
        else:
            optimal = partition_units(64, p_by_rank)
            g = math.gcd(*optimal)
            optimal = [v // g for v in optimal]
            c_optimal = _context_budget(optimal, p_by_rank)

        # --rank-kv-ratio capacity / speed: install the measured vector
        # (rank-uniform decision: server args + gathered P_r only). An explicit
        # pin (env vector or flag list) and the draft worker keep hint-only
        # semantics.
        install = (
            allow_install
            and self.server_args.uneven_kv_derived_mode()
            and not envs.SGLANG_UNEVEN_TOKEN_VECTOR.get()
            and not self.is_draft_worker
        )

        # --rank-kv-ratio speed: the objective is the DECODE step, not the
        # context budget, so the vector is shifted from the capacity
        # proportion toward the per-rank memory-bandwidth proportion. Under
        # DCP each rank runs attention over the tokens it owns and at bs=1 the
        # group waits on the slowest rank, so the deep-context part of the step
        # follows token ownership (#210: -24.5 % on that term, 27B FP8 TP=3 at
        # 120 k resident tokens, against a 1.07 % boot-to-boot noise floor).
        # How far the shift may go is bounded by --rank-perf-loose-ctx-percent,
        # measured against the EFFECTIVE budget min(kv_budget, hybrid cap) --
        # while the hybrid mamba/SWA cap binds, the shift is free and is taken
        # in full even at the default 0 %.
        if install and self.server_args.uneven_kv_speed_mode():
            from sglang.srt.distributed.utils import cp_token_speed_vector

            bw = self.server_args.rank_kv_speed_weights
            if not bw or len(bw) != self.dcp_size or any(w <= 0 for w in bw):
                if self.tp_rank == 0:
                    logger.warning(
                        "Uneven DCP speed mode (--rank-kv-ratio speed): no "
                        "per-rank memory-bandwidth scores available "
                        "(hardware profile missing? --rank-tp-ratio is not "
                        "auto-performance?) -- falling back to the "
                        "capacity-optimal vector."
                    )
            else:
                hard_cap = self._hybrid_kv_token_cap()
                speed_vec, c_speed, t = cp_token_speed_vector(
                    p_by_rank,
                    bw,
                    self.server_args.rank_perf_loose_ctx_percent,
                    hard_cap=hard_cap,
                )
                if self.tp_rank == 0:
                    logger.info(
                        "Uneven DCP speed mode (--rank-kv-ratio speed): "
                        "bandwidth weights %s, capacity-optimal vector %s "
                        "(budget %d), loose-ctx %.1f %%, hybrid cap %s -> "
                        "speed vector %s (budget %d, %.0f %% of the way from "
                        "the capacity proportion to the bandwidth "
                        "proportion).",
                        bw,
                        optimal,
                        c_optimal,
                        self.server_args.rank_perf_loose_ctx_percent,
                        hard_cap,
                        speed_vec,
                        c_speed,
                        100.0 * t,
                    )
                optimal, c_optimal = speed_vec, c_speed

        if install:
            from sglang.srt.distributed.utils import set_cp_token_ratios

            # 'speed' deliberately accepts a SMALLER context budget than the
            # active vector -- that is the trade it exists to make -- so it
            # must not be gated on c_optimal > c_active the way 'capacity' is.
            # The budget floor was already enforced inside
            # cp_token_speed_vector.
            #
            # 'corridor' is likewise not gated on an improvement, for the
            # opposite reason: when the floor binds, the whole point of the
            # install is to hold a SMALLER pool than the active vector would
            # take. Gating on c_optimal > c_active would refuse exactly the
            # case the mode exists for and leave the card below its reserve.
            # c_optimal and c_active are both scored on the clamped
            # capacities above, so the logged pair stays comparable.
            if self.server_args.uneven_kv_speed_mode() or corridor_mode:
                improves = c_optimal >= 0
            else:
                improves = c_optimal > c_active
            if optimal != active and improves:
                mode = self.server_args.rank_kv_ratio
                set_cp_token_ratios(optimal)
                if self.tp_rank == 0:
                    logger.info(
                        "Uneven DCP %s mode (--rank-kv-ratio %s): installed "
                        "measured KV-token ownership vector %s (pre-boot "
                        "estimate was %s), max_total_num_tokens %d -> ~%d "
                        "(per-rank profiled capacity %s).",
                        mode,
                        mode,
                        optimal,
                        active,
                        c_active,
                        c_optimal,
                        p_by_rank,
                    )
            elif self.tp_rank == 0:
                mode = self.server_args.rank_kv_ratio
                logger.info(
                    "Uneven DCP %s mode (--rank-kv-ratio %s): pre-boot "
                    "estimate %s is already the %s optimum "
                    "(max_total_num_tokens=%d, per-rank profiled capacity "
                    "%s).",
                    mode,
                    mode,
                    active,
                    mode,
                    c_active,
                    p_by_rank,
                )
            return

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
            # #364 slice 3: the concurrency ceiling is sized from the SESSION
            # admission budget, not from the (possibly capped) resident pool.
            # Off the cap this is exactly max_mamba_cache_size -- byte-
            # identical. Under a binding --gdn-resident-state-slots the pool
            # is physically smaller but sessions beyond the cap run with their
            # state vacated by the between-tick runtime (armed by the same
            # flag, hence vacate_available), so admission is sized from the
            # pre-cap profiled slot count. The budget never exceeds that
            # profiled count (the over-admission interlock: it was already
            # profiled to fit), and the KV backing stays guarded by the
            # token_capacity // 2 clamp above.
            from sglang.srt.mem_cache.gdn_slot_ladder import (
                recall_profiled_state_slots,
                session_admission_slots,
            )

            _resident_cap = getattr(self.server_args, "gdn_resident_state_slots", None)
            _profiled = getattr(self, "_gdn_profiled_state_slots", None)
            if _profiled is None:
                # The stack that did not apply the cap ITSELF -- the flip's
                # TP stack, built from a post-cap deepcopy -- has no runner
                # attribute, but the args still carry the pre-cap count.
                # Without this it would size the ceiling from the capped
                # pool and diverge from the PP stack. Read-only, so the
                # cap-unset path records nothing and stays byte-identical.
                _profiled = recall_profiled_state_slots(self.server_args)
            if _profiled is None:
                _profiled = self.server_args.max_mamba_cache_size
            _budget_slots = session_admission_slots(
                _profiled,
                _resident_cap,
                vacate_available=_resident_cap is not None,
            )
            max_num_reqs = min(max_num_reqs, _budget_slots // ratio)

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

    # ------------------------------------------------------------------
    # #656: the flip seam as a sizing post. See
    # sglang/srt/managers/phase_flip_seam_reserve.py for the law, the fixed
    # point and why the terms are measured rather than derived.
    # ------------------------------------------------------------------
    def _seam_world_rank(self: ModelRunner) -> int:
        """The rank the seam record is keyed by.

        The flip's primary topology is (tp=1, pp=N), so the flat world rank
        IS pp_rank there -- and the arena tail, the biggest term in the
        record, differs by ~1 GiB between ranks on this rig. Keying by
        tp_rank under a pipeline would give every stage rank 0's record.
        """
        if int(getattr(self, "pp_size", 1) or 1) > 1:
            return int(getattr(self, "pp_rank", 0) or 0)
        return int(getattr(self, "tp_rank", 0) or 0)

    def _seam_reserve(self: ModelRunner):
        """This rank's seam reserve, read once and announced once."""
        cached = getattr(self, "_seam_reserve_cached", None)
        if cached is not None:
            return cached
        from sglang.srt.managers import phase_flip_seam_reserve as seam

        if not getattr(self.server_args, "enable_phase_flip", False):
            cached = seam.SeamReserve(provenance=seam.PROVENANCE_DISABLED)
        else:
            rank = self._seam_world_rank()
            cached = seam.read_seam_reserve(self.server_args, rank)
            # Announced on EVERY flip boot, cold included: a capacity that
            # depends on an on-disk record from a previous boot is a harness
            # trap when it is silent (#188).
            logger.info(
                "%s (rank %d): %s",
                seam.LOG_PREFIX,
                rank,
                seam.describe(cached, seam.record_path(self.server_args, rank)),
            )
        self._seam_reserve_cached = cached
        return cached

    def _maybe_price_cold_seam(self: ModelRunner, reserve, configurator):
        """On a COLD record only, derive the per-token seam slope (#685).

        THE DEFECT THIS CLOSES. A cold ``SeamReserve`` has every measured
        field at zero, ``per_row_bytes`` included, so ``solve_pool_tokens``
        evaluates ``staging(T) = A + max(F, a*T)`` with ``a = 0`` and the
        per-token term vanishes. A first boot is then sized floor-only and
        grants a pool whose cutover it cannot fund. The arming FLOOR has been
        charged on a cold record since #662-F4/A0; the SLOPE never was.

        WHY IT CAN BE DERIVED HERE. #685: a rank's per-row seam cost is the
        full-attention layers it must RECEIVE at the cutover,
        ``max(0, share*n_total - held)``, and every term is rank-LOCAL and
        known at boot -- the share from the flip vector, ``held`` from the
        attention modules this runner actually built, the total from the model
        config, and the per-layer cell from the configurator's own cell
        divided by this rank's attention count. No collective, no measurement,
        nothing transferred from another rig.

        A FALLBACK, NEVER AN OVERRIDE. Any non-cold provenance is returned
        untouched: a stored record is a measurement of THIS rig and outranks a
        model of it.

        ABSTAIN RATHER THAN APPROXIMATE. Every input is checked and a missing
        one leaves the reserve at zero -- the same refusal the cell read below
        already makes for a configurator with no single cell. A rank that
        receives NOTHING (it sheds, or breaks even) is also left at zero:
        charging it a per-token seam would reserve against a transfer that
        does not happen, which is the over-reservation #685 named.

        WHAT THIS DOES NOT YET DO, AND WHY IT STOPS HERE. The slope is derived
        and carried, but the reserve stays INACTIVE: ``SeamReserve.active``
        requires ``id_space > 0``, and the downstream solve anchors on that
        measurement point together with ``have_bytes``
        (``t_floor = t_m + (have_m - A - F) // cell``). A derived slope has no
        measurement point, so setting those to make it active would fabricate
        the anchor. The budget path therefore still returns floor-only on a
        cold boot; what has changed is that the number now EXISTS and is
        announced, instead of being silently zero.

        Consuming it needs an anchor-free branch. ``solve_pool_tokens`` is
        exactly that form and takes no anchor -- but it currently has no live
        caller, so which budget it is solved against (``budget_bytes`` net of
        the floor charge, or something the sizer has not yet subtracted) is a
        design decision on the boot path rather than a wiring detail. Left to
        be decided rather than picked here.
        """
        from sglang.srt.managers import phase_flip_seam_reserve as seam

        if reserve.provenance != seam.PROVENANCE_COLD or reserve.per_row_bytes:
            return reserve

        def _abstain(why: str):
            logger.debug(
                "%s: cold seam slope NOT derived (%s); the per-token term "
                "stays 0 and the pool is sized floor-only.",
                seam.LOG_PREFIX,
                why,
            )
            return reserve

        raw = getattr(self.server_args, "phase_flip_tp_vector", None)
        if not raw:
            return _abstain("no --phase-flip-tp-vector")
        try:
            vector = [int(x) for x in str(raw).split(",")]
        except (TypeError, ValueError):
            return _abstain(f"unparsable flip vector {raw!r}")
        rank = int(self._seam_world_rank())
        if not 0 <= rank < len(vector):
            return _abstain(f"rank {rank} outside the {len(vector)}-stage vector")
        held = int(self._lane_kv_bearing_layer_count() or 0)
        if held <= 0:
            return _abstain("this runner holds no full-attention layer")
        cell = int(getattr(configurator, "_cell_size", 0) or 0)
        if cell <= 0:
            return _abstain("the configurator has no single per-token cell")
        n_layers = int(getattr(self.model_config, "num_hidden_layers", 0) or 0)
        interval = getattr(self.model_config, "full_attention_interval", None)
        total = n_layers // int(interval) if interval else n_layers
        if total <= 0:
            return _abstain("the model config reports no full-attention layer")

        from sglang.srt.managers.seam_slope import derive_seam_slope_for_rank

        slope = derive_seam_slope_for_rank(
            flip_tp_vector=vector,
            rank=rank,
            attention_held=held,
            kv_bytes_per_token_per_attn_layer=float(cell) / float(held),
            n_attention_total=total,
        )
        if slope <= 0.0:
            return _abstain(
                f"rank {rank} receives no layer at the cutover (holds {held} "
                f"of {total}, flip share {vector[rank]}/{sum(vector)})"
            )
        logger.info(
            "%s (rank %d): COLD record -- per-token seam DERIVED, not "
            "measured: %.1f B/row. This rank holds %d of %d full-attention "
            "layers and the flip hands it %d/%d of the token axis, so it must "
            "RECEIVE %.2f layer(s) of KV per row at a pp->tp cutover. Without "
            "this the cold pool is sized floor-only and grants a pool whose "
            "cutover it cannot fund (#685). A measured record from any later "
            "boot supersedes this.",
            seam.LOG_PREFIX,
            rank,
            slope,
            held,
            total,
            vector[rank],
            sum(vector),
            slope / (float(cell) / float(held)),
        )
        return dataclasses.replace(reserve, per_row_bytes=float(slope))

    def _seam_adjusted_budget(
        self: ModelRunner, budget_bytes: int, configurator
    ) -> int:
        """Charge the flip seam against the KV budget. See
        managers/phase_flip_seam_reserve.py for the law and the solve."""
        reserve = self._maybe_price_cold_seam(self._seam_reserve(), configurator)
        from sglang.srt.managers import phase_flip_seam_reserve as seam

        # #662-F4 / A0: THE ARMING FLOOR IS CHARGED EVEN ON A COLD RECORD.
        #
        # ``reserve.active`` is False when every MEASURED field is zero, which
        # is exactly a first boot -- and returning here is what sized
        # boot_maxfill.log's rank 1 to ~875 MiB free against a 1536 MiB arming
        # floor. Every prefill then stayed in the TP layout, in both
        # directions, with no runtime recovery possible because the pool is
        # fixed at boot; the operator had to hand-pin --max-total-tokens.
        #
        # A cold record means THE STAGING COST IS UNKNOWN. It does not mean the
        # arming LEVEL is unknown: that one is derived from the corridor law
        # the operator already stated, and is knowable on every rig at boot.
        # So the two are charged separately from here on.
        flips_on = bool(getattr(self.server_args, "enable_phase_flip", False))
        # The floor is resolved from the SAME two inputs the guard resolves it
        # from -- the operator's override and this rank's measured seam draw --
        # so the pool reserves for the number the gate will actually enforce.
        # On this rig the override is 1536 MiB against a derived 1331: sizing
        # for the derived one would leave every rank 205 MiB short of its own
        # arming floor, which looks like a healthy boot that never flips.
        arming_floor = (
            seam.arming_floor_target_bytes(
                configured_mib=seam.configured_arming_floor_mib(self.server_args),
                measured_draw_mib=(
                    int(reserve.arming_draw_bytes()) >> 20 if reserve.active else 0
                ),
            )
            if flips_on
            else 0
        )
        # ONE CHARGE, COMPUTED ONCE, LOGGED AND APPLIED. #678 caught this line
        # reporting the law-baseline number while
        # ``seam_adjusted_budget_bytes`` applied the residual one -- a log that
        # disagrees with the arithmetic it describes, which is worse than no
        # log because it is the number an operator reads back.
        already_reserved = seam.seam_solve_reserved_free_bytes(reserve)
        floor_charge = seam.arming_floor_subtrahend_bytes(
            arming_floor, already_reserved
        )
        if floor_charge or arming_floor:
            logger.info(
                "%s (rank %d): ARMING FLOOR %d MiB (corridor law + this rank's "
                "measured one-leg seam draw + load margin) must stay free for a "
                "flip to arm at all. Already held free by %s: %d MiB, so the "
                "pool gives up the %d MiB difference and no more -- charging the "
                "whole floor over a solve that already reserved it is what "
                "returned 284181 tokens where 550000 arms and flips.",
                seam.LOG_PREFIX,
                self._seam_world_rank(),
                arming_floor >> 20,
                (
                    "the seam solve"
                    if already_reserved > seam._corridor_law_bytes()
                    else "the corridor law"
                ),
                max(already_reserved, seam._corridor_law_bytes()) >> 20,
                floor_charge >> 20,
            )
        # #685: HOW MANY ATTENTION LAYERS DOES THIS RANK ACTUALLY RECEIVE at
        # the flip? A rank that receives none pays no per-token seam -- its
        # measured per_row_bytes is BASELINE (checksums, the one-layer
        # streaming window, allocator grain), not transfer, and reserving it
        # per token holds back memory for bytes that never move. Measured
        # 2026-08-16: the binding rank was 192 MiB short on a 1059 MiB need
        # whose per-token term was ~190 MiB.
        #
        # DERIVED, NEVER FROZEN, and LOUD ABOUT ITS SOURCE. The frozen triple
        # in phase_flip_seam_reserve is a drift watchdog, not an input; the
        # live value is per-boot measured. The map comes from
        # derive_pp_full_attn_layer_map, whose own docstring warns that the
        # layer-id list must be the UNMUTATED global one (the PP stack's
        # model_config is rewritten by adjust_hybrid_swa_layers_for_pp), so
        # the source is logged and any doubt falls back to the previous
        # arithmetic rather than guessing.
        # HOISTED ABOVE THE COLD BRANCH THAT READS IT. `cell` used to be
        # computed after the reserve.active check; the cold seam pricing needs
        # it, and taking it later cost a second startup crash
        # ("UnboundLocalError: cannot access local variable 'cell'", 12:54).
        # It depends only on the configurator, never on the reserve, so there
        # is nothing to compute later.
        cell = int(getattr(configurator, "_cell_size", 0) or 0)

        # ALL THREE NAMES ARE DEFINED BEFORE ANY BRANCH CAN READ THEM. The
        # first version assigned attn_counts and rank only inside the success
        # path of the try below, and defined received_layers only after the
        # cold branch that consumes it -- which sigquit every rank at startup
        # with "UnboundLocalError: cannot access local variable
        # 'received_layers'" (12:37, 30030 down). Same shape as the silent
        # guard before it: a branch that skips the derivation must also define
        # what the consumer reads, not merely skip the assignment.
        received_layers = None
        attn_counts = None
        rank = 0
        try:
            from sglang.srt.managers.phase_flip_runtime import (
                derive_pp_full_attn_layer_map,
            )
            from sglang.srt.managers.seam_slope import received_attention_layers

            vec = getattr(self.server_args, "phase_flip_tp_vector", None)
            if isinstance(vec, str):
                vec = [float(x) for x in vec.split(",") if x.strip()]
            # THE LIST LIVES ON hf_text_config, NOT ON model_config. The flip
            # runtime reads exactly this path
            # (tp_model_config.hf_text_config.full_attention_layer_ids) and its
            # own comment explains why: the attention registry reads it via
            # runner.mambaish_config. Reading model_config directly returned an
            # EMPTY list here and silently disarmed the whole derivation --
            # measured, tp_vector and num_hidden=64 and pp_size=3 all present,
            # full_attention_layer_ids=0 entries. model_config is kept as a
            # fallback because the PP stack's copy is the mutated one and an
            # empty list from either source must not look like "no attention".
            ids = list(
                getattr(
                    getattr(self.model_config, "hf_text_config", None),
                    "full_attention_layer_ids",
                    None,
                )
                or getattr(self.model_config, "full_attention_layer_ids", None)
                or []
            )
            n_hidden = int(getattr(self.model_config, "num_hidden_layers", 0) or 0)
            pp_size = int(getattr(self.server_args, "pp_size", 1) or 1)
            rank = int(getattr(self, "pp_rank", getattr(self, "tp_rank", 0)) or 0)
            if vec and ids and n_hidden > 0 and pp_size > 1:
                layer_map = derive_pp_full_attn_layer_map(ids, n_hidden, pp_size)
                attn_counts = [len(stage) for stage in layer_map]
                n_attn_total = sum(attn_counts)
                received = received_attention_layers(vec, attn_counts, n_attn_total)
                received_layers = int(received[rank])
                logger.info(
                    "%s (rank %d): received-layer derivation -- attention map "
                    "%s over %d total, tp vector %s -> this rank RECEIVES %d "
                    "layer(s) at the flip. %s",
                    seam.LOG_PREFIX,
                    rank,
                    attn_counts,
                    n_attn_total,
                    vec,
                    received_layers,
                    (
                        "Zero received: its measured per_row_bytes is baseline, "
                        "not seam, so no per-token seam is reserved."
                        if received_layers <= 0
                        else "Nonzero: the measured per-token slope stands."
                    ),
                )
            else:
                # THE GUARD MUST SPEAK WHEN IT REFUSES. The first version
                # logged only on the success branch, so a falsy input left
                # received_layers=None with NO line anywhere and the fix sat
                # inert through two boots looking like it had simply not been
                # reached. A silent precondition is indistinguishable from
                # dead code.
                logger.info(
                    "%s: received-layer derivation SKIPPED -- tp_vector=%s "
                    "full_attention_layer_ids=%d entr(y/ies) num_hidden=%s "
                    "pp_size=%s. The falsy one is the missing input; the "
                    "per-token seam charge stays at its measured value.",
                    seam.LOG_PREFIX,
                    vec,
                    len(ids),
                    n_hidden,
                    pp_size,
                )
        except Exception as exc:  # noqa: BLE001 - sizing must not fail on a probe
            logger.info(
                "%s: received-layer derivation unavailable (%r); the per-token "
                "seam charge stays at its measured value.",
                seam.LOG_PREFIX,
                exc,
            )

        if not reserve.active:
            # #685 COLD BOOT: PRICE THE SEAM FROM THE LAYOUT, do not skip it.
            #
            # Reproduced live 2026-08-16 12:04, all three ranks "seam reserve
            # is COLD ... sizes with NO flip-seam term": the pool came out at
            # the raw 550000 pin where the warm boot minutes later solved
            # 467708 against the same budget. A cold boot sized ~17% above
            # what the same configuration knows to be safe, and the first flip
            # then meets a pool that was never priced for it.
            #
            # There is no measured record to read, but the SLOPE does not need
            # one: it is a property of the layout (which layers this rank
            # receives) and the KV geometry (bytes per token per attention
            # layer), both known at boot. The measured record adds the fixed
            # and arena terms and the id-space anchor; without it we charge
            # the per-token term alone, which UNDERCHARGES relative to warm and
            # is deliberately the safe direction to be wrong in on a first
            # boot -- strictly better than charging nothing.
            #
            # R' IS NET OF THE FLOOR CHARGE. The arming floor must stay free
            # for a flip to arm at all; it is a reservation, never a spendable
            # balance, so it cannot become KV. Solving against the
            # pre-subtraction budget would price it as if it could, which is
            # the same overshoot by another route. The warm branch returns
            # this same net quantity, so both branches answer "how many bytes
            # could become KV" with one number.
            #
            # AND THE FLOOR IS CHARGED ONCE. This returns min(net, allowed *
            # cell) -- never net minus the charge a second time, which would
            # hold a whole arming floor free for the life of the instance.
            net = max(0, int(budget_bytes) - floor_charge)
            cold_slope = 0.0
            try:
                attn_local = (
                    attn_counts[rank]
                    if received_layers is not None and attn_counts
                    else 0
                )
                if received_layers and attn_local > 0 and int(cell) > 0:
                    kv_per_attn_layer = float(cell) / float(attn_local)
                    cold_slope = float(received_layers) * kv_per_attn_layer
            except Exception:  # noqa: BLE001 - sizing must not fail on a probe
                cold_slope = 0.0
            if cold_slope <= 0.0 or int(cell) <= 0:
                logger.info(
                    "%s (rank %s): COLD and the seam slope could not be "
                    "derived (received=%s cell=%s); sizing with NO per-token "
                    "seam term, the pre-#685 behaviour.",
                    seam.LOG_PREFIX,
                    self._seam_world_rank(),
                    received_layers,
                    cell,
                )
                return net
            allowed_cold = seam.solve_pool_tokens(
                net, int(cell), 0, cold_slope, 0, received_layers=received_layers
            )
            priced = max(0, min(net, allowed_cold * int(cell)))
            logger.info(
                "%s (rank %s): COLD seam priced from the layout -- receives %d "
                "layer(s), %.1f B/token/attn-layer, slope %.1f B/token; R' = "
                "%d MiB net of the %d MiB floor charge -> %d tokens, budget "
                "%d -> %d MiB. No measured record, so the fixed and arena "
                "terms are NOT charged; this undercharges versus warm, which "
                "is the safe direction on a first boot.",
                seam.LOG_PREFIX,
                self._seam_world_rank(),
                int(received_layers or 0),
                (float(cell) / float(attn_counts[rank])) if attn_counts else 0.0,
                cold_slope,
                net >> 20,
                floor_charge >> 20,
                allowed_cold,
                net >> 20,
                priced >> 20,
            )
            return priced

        # Per-RANK per-token bytes, which is what the record's slope means.
        # A configurator with no single cell (hybrid SWA, MiniMax sparse)
        # gets no invented one: abstaining is a smaller error than charging
        # a slope against a cell that does not exist.
        # WHAT A REFUSED SEAM COSTS decides whether the pool pays for the
        # guarantee. Under strict purity a refused tp_to_pp means prefill
        # never runs -- boot E held the corridor and served nothing -- so the
        # pool must shrink until the seam is affordable. Where prefill may run
        # in the TP layout the refusal costs one flip, and shrinking the pool
        # below what the corridor law already pays for would hold VRAM free
        # ABOVE the law for the life of the instance instead. Read from the
        # purity mode, never configured on its own.
        # #685 DIAGNOSTIC: the received-layer derivation below did not appear
        # in the 12:04 boot's log although the pool WAS seam-adjusted, so the
        # function is returning before it on the live path. One unconditional
        # line at the point of no return, naming the terms every early exit
        # keys on, decides which branch without another bisect.
        logger.info(
            "%s: seam-adjusted budget reached the survivability step -- "
            "reserve.active=%s per_row=%.1f fixed=%d arena=%d id_space=%s",
            seam.LOG_PREFIX,
            getattr(reserve, "active", "?"),
            float(getattr(reserve, "per_row_bytes", 0.0) or 0.0),
            int(getattr(reserve, "fixed_bytes", 0) or 0),
            int(getattr(reserve, "arena_fixed_bytes", 0) or 0),
            getattr(reserve, "id_space", "?"),
        )
        from sglang.srt.managers.phase_purity import purity_from_server_args

        try:
            survivable = bool(
                purity_from_server_args(self.server_args).prefill_allowed_in_tp()
            )
        except Exception:
            survivable = False
        new_bytes, allowed = seam.seam_adjusted_budget_bytes(
            budget_bytes,
            cell,
            reserve,
            abandon_is_survivable=survivable,
            arming_floor_bytes=arming_floor,
            received_layers=received_layers,
        )
        logger.info(
            "%s (rank %d): seam floor %d MiB + %.1f B/token, measured with "
            "%d MiB spendable at an id space of %d -> this rank can fund up "
            "to %d tokens, so the %d MiB budget becomes %d MiB. Sizing "
            "without this is what produced a boot that held the corridor and "
            "served nothing (#656 boot E/G).",
            seam.LOG_PREFIX,
            self._seam_world_rank(),
            reserve.fixed_bytes >> 20,
            reserve.per_row_bytes,
            reserve.have_bytes >> 20,
            reserve.id_space,
            allowed,
            int(budget_bytes) >> 20,
            new_bytes >> 20,
        )
        return new_bytes

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
        # #656: charge the flip seam HERE, the one funnel every sizing path
        # reaches. Keyed to the post-capture path instead, it never fired at
        # all on the ship config, whose pool is decided pre-capture (boot H).
        _profiled_bytes = budget_bytes
        budget_bytes = self._seam_adjusted_budget(budget_bytes, configurator)
        # #704: EMIT the true per-rank holdback at the one funnel every sizing
        # path reaches. This is the term that lands at 6,688 / 3,561 / 5,166 MiB
        # on metal while derived_rank_auto_reserve_mib reports 4,160 uniformly,
        # and the term no external re-derivation reproduced (+20 %, -3.8 %,
        # -12 % on three attempts). With it emitted, one boot identifies the
        # form that three collinear points could not.
        _frac = budget_holdback_fraction(_profiled_bytes, budget_bytes)
        logger.info(
            "KV budget holdback: profiled=%.1f MiB, adjusted=%.1f MiB, "
            "holdback=%.1f MiB (%s of profiled)",
            _profiled_bytes / (1 << 20),
            budget_bytes / (1 << 20),
            budget_holdback_mib(_profiled_bytes, budget_bytes),
            "n/a" if _frac is None else f"{_frac:.3%}",
        )
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
        # #330 initial budgets: coarse profile-time clamp so the boot never
        # vastly overshoots a declared budget; the capacity runtime reconciles
        # exactly (against the measured floor) at the first consensus
        # boundary. Uses device-level used bytes -- the closest honest proxy
        # for this process's floor at profiling time.
        if getattr(self.server_args, "enable_vram_dial", False) and getattr(
            self.server_args, "vram_budget_mib", None
        ):
            budgets = self.server_args.parsed_vram_budget_mib()
            if len(budgets) != self.tp_size:
                raise ValueError(
                    f"--vram-budget-mib has {len(budgets)} entries for "
                    f"tp_size={self.tp_size}"
                )
            budget_bytes = int(budgets[self.tp_rank]) << 20
            free_b, total_b = torch.cuda.mem_get_info(self.gpu_id)
            used_b = int(total_b) - int(free_b)
            kv_allow = budget_bytes - used_b
            if kv_allow <= 0:
                raise ValueError(
                    f"--vram-budget-mib {budgets[self.tp_rank]} MiB for rank "
                    f"{self.tp_rank} is below what is already resident at "
                    f"profiling time ({used_b >> 20} MiB used on device "
                    f"{self.gpu_id}); nothing is left for KV."
                )
            if kv_allow < available_bytes:
                logger.info(
                    "#330 initial budget clamp: profiled KV budget %d MiB -> "
                    "%d MiB (budget %d MiB, %d MiB already used)",
                    available_bytes >> 20,
                    kv_allow >> 20,
                    budgets[self.tp_rank],
                    used_b >> 20,
                )
                available_bytes = kv_allow
        if not self.post_capture_kv_active:
            # Uneven-TP self-calibration on the final profiled budget;
            # with post-capture sizing the (more accurate) post-capture
            # measurement runs it instead.
            self._maybe_suggest_mlp_rebalance(available_bytes)
            self._maybe_suggest_dcp_token_vector(available_bytes, allow_install=True)
        elif self.server_args.uneven_kv_derived_mode():
            # --rank-kv-ratio capacity/speed with post-capture sizing planned: the
            # token vector must be FINAL before pools/backends/graphs
            # snapshot it, so the measured install runs on the pre-capture
            # profiling here; the post-capture pass stays hint-only (any
            # residual delta shows up as the usual restart hint).
            self._maybe_suggest_dcp_token_vector(available_bytes, allow_install=True)
        config = self._config_from_budget(available_bytes)
        config.max_running_requests = self._resolve_max_num_reqs(
            config.max_total_num_tokens
        )
        configurator = create_memory_pool_configurator(self)
        config = configurator.finalize_with_max_running_requests(config)
        config.mem_fraction_static = self.server_args.mem_fraction_static
        return config

    def _resolve_dual_group_lane_pool_config(self: ModelRunner) -> MemoryPoolConfig:
        """Multi-group runtime (#274): rank-local pool sizing from the lane's
        explicit MiB budget.

        No profiling (the budget is given), no cross-rank sync (the lane is
        rank-local; `_apply_token_constraints` short-circuits on the lane
        flag), no barrier (`_profile_available_bytes` is never called). The
        configurator itself is reused unchanged, so the hybrid GDN pools,
        page alignment and request clamps stay the stock derivations -- just
        against the lane's scoped server_args view (its own
        max_running_requests / max_mamba_cache_size).
        """
        from sglang.srt.model_executor.pool_configurator import (
            create_memory_pool_configurator,
        )

        budget_mib = self.server_args.dual_group_lane_budget_mib
        if not budget_mib or budget_mib <= 0:
            raise ValueError(
                "dual-group lane runner needs --dual-group-lane-budget-mib > 0."
            )
        budget_bytes = int(budget_mib) << 20
        if getattr(self, "is_dual_group_lane_draft", False):
            return self._resolve_dual_group_lane_draft_pool_config(budget_bytes)
        config = self._config_from_budget(budget_bytes)
        config.max_running_requests = self._resolve_max_num_reqs(
            config.max_total_num_tokens
        )
        configurator = create_memory_pool_configurator(self)
        config = configurator.finalize_with_max_running_requests(config)
        config.mem_fraction_static = self.server_args.mem_fraction_static
        logger.info(
            "dual-group lane %d pool sizing (rank-local): budget %d MiB -> "
            "max_total_num_tokens=%d, max_running_requests=%d.",
            self.dual_group_lane_id,
            budget_mib,
            config.max_total_num_tokens,
            config.max_running_requests,
        )
        return config

    def _lane_kv_bearing_layer_count(self: ModelRunner) -> int:
        """How many layers of THIS runner's assembled model hold KV.

        Counted off the built module tree rather than derived from the model
        CONFIG, because for the lane's NEXTN head the two disagree: the head
        is one decoder layer, but its config is the target's (the draft
        config is built from the same checkpoint, so it still reports the
        target's 64 layers and the target's hybrid full-attention layer ids).
        The configurator derives the layer count from that config, intersects
        it with this runner's layer range, and lands on 0 -> a per-token cell
        size of 0 -> a division by zero. Counting the attention modules that
        actually exist is family-neutral and cannot disagree with the model
        that will run.
        """
        from sglang.srt.layers.radix_attention import RadixAttention

        model = getattr(self, "model", None)
        if model is None:
            return 0
        return sum(1 for m in model.modules() if isinstance(m, RadixAttention))

    def _resolve_dual_group_lane_draft_pool_config(
        self: ModelRunner, budget_bytes: int
    ) -> MemoryPoolConfig:
        """Rank-local KV sizing for the lane's NEXTN head, from its OWN slice
        of the lane budget.

        Deliberately not the stock arrangement. A serving NEXTN draft is
        handed the TARGET's ``memory_pool_config`` and never sizes anything;
        the lane's head cannot take that route, because the lane target's
        config describes the lane target's 64-layer pool and the head needs
        its own, tiny one. So the head gets a budget post of its own
        (``split_lane_budget``) and turns it into tokens the same way the
        lane target does -- bytes per token, page-aligned, no profiling, no
        collective.

        The budget post is sized so that this pool comes out at the TARGET's
        token count, because the head follows the target's sequences. Two
        guards keep that promise honest against a mis-derived post: the cap
        below trims a pool that came out LARGER (those bytes would buy slots
        the head can never reach), and the shortfall warning names a pool
        that came out SMALLER, which is the failure mode -- the head runs out
        of KV mid-sequence while the target still has room.
        """
        from sglang.srt.model_executor.pool_configurator import (
            MemoryPoolConfig,
            create_memory_pool_configurator,
        )

        n_layers = self._lane_kv_bearing_layer_count()
        if n_layers <= 0:
            raise ValueError(
                "dual-group lane head: the assembled head has no "
                "KV-bearing attention layer -- refusing to size a pool for a "
                "model that cannot use one."
            )
        kv_heads = self.model_config.get_total_num_kv_heads()
        head_dim = self.model_config.head_dim
        v_head_dim = getattr(self.model_config, "v_head_dim", head_dim)
        kv_elem = torch._utils._element_size(self.kv_cache_dtype)
        cell = kv_heads * (head_dim + v_head_dim) * n_layers * kv_elem
        tokens = (budget_bytes // cell) // self.page_size * self.page_size
        cap = int(getattr(self, "dual_group_lane_token_cap", 0) or 0)
        if cap > 0 and tokens > cap:
            tokens = cap // self.page_size * self.page_size
        elif cap > 0 and tokens < cap:
            # Not fatal -- short jobs never reach the shortfall -- but it is
            # never intended, so it is named with both numbers and the MiB it
            # would take to close, rather than surfacing later as a pool
            # exhaustion on the head alone.
            logger.warning(
                "dual-group lane %d HEAD pool is SHORTER than the lane "
                "target's: %d tokens against %d. The head follows the "
                "target's sequences, so a job longer than %d tokens will "
                "exhaust the head's KV while the target still has room. "
                "Closing it costs %d MiB of the lane budget.",
                self.dual_group_lane_id,
                tokens,
                cap,
                tokens,
                (((cap - tokens) * cell) + (1 << 20) - 1) >> 20,
            )
        if tokens <= 0:
            raise ValueError(
                f"dual-group lane head: budget {budget_bytes >> 20} MiB is "
                f"smaller than one page of its KV ({cell * self.page_size} "
                "bytes). Raise --dual-group-lane-budget-mib or lower "
                "--dual-group-lane-speed-dial."
            )
        config = MemoryPoolConfig(max_total_num_tokens=int(tokens))
        config.max_running_requests = self._resolve_max_num_reqs(
            config.max_total_num_tokens
        )
        configurator = create_memory_pool_configurator(self)
        config = configurator.finalize_with_max_running_requests(config)
        config.mem_fraction_static = self.server_args.mem_fraction_static
        logger.info(
            "dual-group lane %d HEAD pool sizing (rank-local): budget %d MiB, "
            "%d KV-bearing layer(s), %d B/token -> max_total_num_tokens=%d, "
            "max_running_requests=%d.",
            self.dual_group_lane_id,
            budget_bytes >> 20,
            n_layers,
            cell,
            config.max_total_num_tokens,
            config.max_running_requests,
        )
        return config

    def init_memory_pool(self: ModelRunner, pre_model_load_memory: int):
        if getattr(self, "is_dual_group_lane", False):
            self.memory_pool_config = self._resolve_dual_group_lane_pool_config()
        elif not self.spec_algorithm.is_none() and self.is_draft_pool_worker:
            # is_draft_POOL_worker, not is_draft_worker: this is a
            # pool-shape decision, and the flip's TP stack rides the
            # is_draft_worker construction gates while its POOLS take the
            # target-model treatment (see ModelRunner.is_draft_pool_worker,
            # which exists to make exactly this distinction).
            #
            # Before #631 armed speculation on the flip stack, spec was
            # refused alongside the flip, so this branch could not be
            # reached by a flip runner and the wrong flag was harmless. It
            # is reachable now: the TP stack would have demanded a
            # caller-supplied pool config it is supposed to RESOLVE, and
            # every rank died here with "Draft worker requires
            # memory_pool_config" (measured, boot 15, 2026-08-08).
            assert self.memory_pool_config is not None, (
                "Draft worker requires memory_pool_config"
            )
        else:
            self.memory_pool_config = self._resolve_memory_pool_config(
                pre_model_load_memory
            )

        self._apply_memory_pool_config(self.memory_pool_config)
        # Component-balance checkpoint "post-pools" (KV/mamba/aux pools now
        # allocated on top of weights; graphs/workspaces still missing).
        self._mem_ckpt_post_pools = (
            torch.cuda.memory_allocated(),
            torch.cuda.memory_reserved(),
        )

        logger.info(
            f"Memory pool end. "
            f"avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )
