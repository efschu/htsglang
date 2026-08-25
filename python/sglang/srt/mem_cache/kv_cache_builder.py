from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

from dataclasses import dataclass
from typing import Optional


def _mamba_slot_reorder_active(server_args) -> bool:
    """#755, imported lazily to keep this module's import graph unchanged."""
    from sglang.srt.mem_cache.mamba_pool_floor import mamba_slot_reorder_active

    return mamba_slot_reorder_active(server_args)


@dataclass(frozen=True, slots=True, kw_only=True)
class KVCacheBuildResult:
    is_hybrid_swa: bool
    is_hybrid_ssm: bool
    sliding_window_size: Optional[int]
    full_tokens_per_layer: Optional[int]
    swa_tokens_per_layer: Optional[int]
    req_to_token_pool: object
    token_to_kv_pool_allocator: object
    disable_radix_cache: bool
    tree_cache: object


from typing import TYPE_CHECKING

from sglang.srt.configs.model_config import ModelImpl
from sglang.srt.environ import envs
from sglang.srt.managers.mm_utils import init_mm_embedding_cache
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.registry import TreeCacheBuildContext, create_tree_cache
from sglang.srt.model_loader.utils import get_resolved_model_impl
from sglang.srt.runtime_context import get_parallel

if TYPE_CHECKING:

    from torch.distributed import ProcessGroup

    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.distributed.parallel_state import GroupCoordinator
    from sglang.srt.distributed.parallel_state_wrapper import ParallelState
    from sglang.srt.managers.tp_worker import BaseTpWorker
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


def get_draft_kv_pool(
    *,
    draft_worker: BaseTpWorker,
    spec_algorithm: SpeculativeAlgorithm,
    server_args: ServerArgs,
):
    """Return the draft token-to-KV pool for the current draft worker,
    or None when no draft KV pool is available."""
    if draft_worker is None or spec_algorithm.is_ngram():
        return None

    # V2 workers nest the draft runner under `.draft_worker`.
    if server_args.enable_multi_layer_eagle:
        draft_runner = draft_worker.draft_worker.draft_runner_list[0]
    else:
        draft_runner = draft_worker.draft_worker.draft_runner
    # Solo-shadow guard (--speculative-draft-placement solo): shadow ranks
    # never allocate a draft KV pool (the draft runs only on the solo host),
    # so their runner exposes token_to_kv_pool as None — and this builder is
    # reached even with disaggregation_mode "null" (scheduler
    # init_disaggregation calls it before checking the mode). "No draft KV
    # pool" is the correct answer for a shadow, not an error.
    return getattr(draft_runner, "token_to_kv_pool", None)


#: #861: the phase that OWNS the drafter, and therefore the only phase in which
#: draft-half device-tier I/O may run. There is exactly one drafter in a
#: phase-flip process and it lives on the flip's TP stack
#: (``phase_flip_boot.build_flip_draft_worker``); the PP prefill phase has none
#: by design, so a draft backup taken there would persist rows no drafter ever
#: wrote, under a content-addressed key, for the TP phase to load as valid.
DRAFT_OWNER_PHASE_FLIP = "tp"


@dataclass(frozen=True, slots=True, kw_only=True)
class DraftRegistration:
    """What the draft half must be registered AS, for one binding generation.

    Separated from the allocation on purpose: the *resolution* -- which drafter
    owns the draft KV, which phase may use it, which generation's host pool its
    indices are 1-to-1 with -- is the part that was wrong (#861), and it is the
    part a hermetic test can pin. The allocation below it needs a real pool.
    """

    pool: object
    owner_phase: Optional[str]
    generation: Optional[int]
    identity: str


def drafter_identity_hash(server_args) -> str:
    """A short hash of everything that decides what a draft KV byte MEANS.

    #861 guard, and the reason it exists before the pages it protects. Fix (0)
    newly makes ``{hash}.draft`` pages readable at L3, and
    ``compute_model_identity_hash`` -- which every other HiCache key suffix is
    built from -- covers ``model_path | revision | dtype | quantization |
    kv_cache_dtype`` and the uneven-TP vectors, and NOTHING about the drafter.
    Two boots that agree on the target and differ in drafter (a different NEXTN
    checkpoint, MTP<->EAGLE, the #156 cross-algorithm switch) would therefore
    read each other's draft KV AS VALID, with blob length the only accidental
    guard -- and equal geometry is the common case, so there is often no guard
    at all.

    This is NOT the full fix (task #861 item (a): fold drafter identity into
    ``compute_model_identity_hash`` so every backend and both key routes carry
    it). It is the cheapest CORRECT form of it for the route fix (0) opens: the
    generic file backend's draft key. A page written by another drafter simply
    does not exist under this suffix -- a clean MISS, exactly the argument
    ``HiCacheFile`` already makes for the identity hash it does carry ("Old-layout
    keys ... simply no longer match: clean miss instead of a silent wrong-format
    hit").

    ``draft_kv_layout`` is included deliberately, and it is the one field
    ``compute_model_identity_hash`` must NOT grow: DESIGN_631b records that it is
    a parallelism decision rather than a weights one, so it does not belong in
    the TARGET key -- but it decides the draft pool's row space and per-row
    byte length, so it absolutely belongs in the DRAFT key.
    """
    import hashlib

    parts = [
        str(getattr(server_args, "speculative_algorithm", "") or ""),
        str(getattr(server_args, "speculative_draft_model_path", "") or ""),
        str(getattr(server_args, "speculative_draft_model_revision", "") or ""),
        str(getattr(server_args, "speculative_num_steps", "") or ""),
        str(getattr(server_args, "speculative_eagle_topk", "") or ""),
        str(getattr(server_args, "speculative_num_draft_tokens", "") or ""),
        str(getattr(server_args, "draft_kv_layout", "replicated") or "replicated"),
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def resolve_draft_registration(scheduler, phase: Optional[str]) -> Optional[
    "DraftRegistration"
]:
    """The draft half's registration for the phase being ENTERED, or None.

    THE #861 DEFECT IS ONE LINE ABOVE THIS FUNCTION'S REASON TO EXIST.
    ``Scheduler.__init__`` calls ``maybe_register_hicache_draft`` with
    ``self.draft_worker`` and ``self.spec_algorithm`` -- and on a phase-flip
    boot #631 has DELIBERATELY nulled both (``scheduler.py``: the configured
    algorithm is parked in ``flip_spec_algorithm`` and the drafter is built on
    ``phase_flip_stacks``, because the boot phase is PP and PP has no drafter).
    So ``get_draft_kv_pool`` returns None, ``set_draft_kv_pool`` is never
    called, ``has_draft`` stays False for process life, and every HiCache
    read-through in the TP phase restores a TARGET-ONLY prefix whose draft rows
    hold the previous occupants' bytes. Nothing raises; acceptance collapses.

    Resolved AT THE CUTOVER rather than at boot, and that is the #719/#847
    lesson rather than a preference. The draft host pool's indices are 1-to-1
    with the TARGET host pool's (see the ratio below), and the target host pool
    MOVES at a rebind -- ``hicache_phase_binding._stamp`` re-points
    ``mem_pool_host`` to the incoming phase's. A registration captured at boot
    names generation 0's pool; consumed after a rebind it would index a
    different pool's slot space. So the registration carries the generation it
    was minted at, from ``current_generation()`` -- the SAME authority the
    releases and the write-backs already ride, not a second stamp scheme.

    Returns None (never raises) when there is simply nothing to register: no
    hierarchical cache, no speculation, no drafter, or the phase being entered
    is not the drafter's. A refusal that must be LOUD belongs to the caller,
    which can log it once per cutover.
    """
    if not getattr(scheduler, "enable_hierarchical_cache", False):
        return None
    tree_cache = getattr(scheduler, "tree_cache", None)
    if getattr(tree_cache, "cache_controller", None) is None:
        return None

    server_args = getattr(scheduler, "server_args", None)
    if server_args is None:
        return None

    # OWNERSHIP IS A PROPERTY OF THE INSTANCE, NOT OF WHERE THE HANDLE HAPPENS
    # TO BE. This is the trap the first draft of this function fell into.
    # ``rebind_for_cutover`` runs AFTER the active stack swap
    # (``phase_flip_runtime``: `scheduler.draft_worker = want_draft`, then the
    # rebind), so on the pp->tp leg the flip's drafter IS reachable through
    # ``scheduler.draft_worker`` -- and deriving "is this a flip instance" from
    # "did I have to fall back to the stacks" would then answer NO on exactly
    # the leg that needs the phase term most, arming the draft half for both
    # phases. So the phase is read from the stacks' EXISTENCE, and the handle
    # is looked up wherever it currently lives.
    stacks = getattr(scheduler, "phase_flip_stacks", None)
    owner_phase = DRAFT_OWNER_PHASE_FLIP if stacks is not None else None

    draft_worker = getattr(scheduler, "draft_worker", None)
    spec_algorithm = getattr(scheduler, "spec_algorithm", None)
    if draft_worker is None:
        # Before the swap, or on the pp leg: the flip's drafter lives on the
        # stacks and the algorithm is parked (#631).
        draft_worker = getattr(stacks, "draft_worker", None)
        spec_algorithm = getattr(scheduler, "flip_spec_algorithm", None)
    if spec_algorithm is not None and spec_algorithm.is_none():
        # The scheduler's own pair can be the nulled boot-phase values while the
        # parked pair is real. Consult the parked one before giving up.
        parked = getattr(scheduler, "flip_spec_algorithm", None)
        if parked is not None and not parked.is_none():
            spec_algorithm = parked
    if draft_worker is None or spec_algorithm is None or spec_algorithm.is_none():
        return None

    # The drafter belongs to ONE phase. Registering it while the process is
    # entering the other one would arm a backup that persists rows no drafter
    # wrote -- worse than the missing registration this fixes.
    if owner_phase is not None and phase is not None and phase != owner_phase:
        return None

    pool = get_draft_kv_pool(
        draft_worker=draft_worker,
        spec_algorithm=spec_algorithm,
        server_args=server_args,
    )
    if pool is None:
        return None

    # UNWRAPPED HERE, the way every other consumer unwraps. The draft runner's
    # pool is a HybridLinearKVPool on this model family and the host pool is
    # built from -- and the transfers name -- the INNER full-attention pool.
    # Registering the wrapper is the W34 arm-2 shape: the attribute the shape
    # check reads lives one dereference away.
    from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

    if isinstance(pool, HybridLinearKVPool):
        pool = pool.full_kv_pool

    from sglang.srt.mem_cache.hicache_phase_binding import current_generation

    return DraftRegistration(
        pool=pool,
        owner_phase=owner_phase,
        generation=current_generation(),
        identity=drafter_identity_hash(server_args),
    )


def rebind_hicache_draft_for_phase(scheduler, phase: str) -> bool:
    """#861: (re)arm or disarm the draft half for the phase being entered.

    Called from ``hicache_phase_binding.rebind_for_cutover`` AFTER the target
    readers are coherent, so the draft half can never be armed against a binding
    that itself failed to move. One authority, one call site, no parallel
    scheme -- the same rule W32 was written to enforce.

    THE HOST POOL IS BUILT ONCE AND RE-STAMPED THEREAFTER. Allocating a pinned
    host pool per cutover would charge the host budget on every flip, and on
    this box that budget is the binding constraint (DESIGN_706 C1). So the pool
    is cached on the scheduler and each later cutover only re-checks the 1-to-1
    invariant against the now-bound target host pool and re-stamps the
    generation. A size change means the invariant is GONE, and that is refused
    loudly rather than papered over -- indexing a draft pool with another
    pool's slot ids is the #345 right-token/wrong-slot class.

    Returns True when the draft half is armed after this call.
    """
    tree_cache = getattr(scheduler, "tree_cache", None)
    cc = getattr(tree_cache, "cache_controller", None)
    # A controller with no draft surface at all has no draft half to arm or
    # disarm, and the TARGET rebind must not fail because of that: a refused
    # rebind leaves #718's disarm standing over the whole device tier, which is
    # strictly worse than an unarmed draft half. Same reason `readers_of`
    # tolerates a missing controller rather than asserting one.
    if cc is None or not hasattr(cc, "disarm_draft_kv_pool"):
        return False

    reg = resolve_draft_registration(scheduler, phase)
    if reg is None:
        cc.disarm_draft_kv_pool(
            f"the '{phase}' phase does not own a draft KV pool"
        )
        return False

    cached = getattr(scheduler, "_hicache_draft_host_pool", None)
    primary = cc.mem_pool_host
    if cached is None:
        cached = _build_draft_host_pool(
            pool=reg.pool,
            primary=primary,
            server_args=scheduler.server_args,
            page_size=scheduler.page_size,
        )
        if cached is None:
            cc.disarm_draft_kv_pool(
                f"no host pool could be built for draft pool "
                f"{type(reg.pool).__name__}"
            )
            return False
        scheduler._hicache_draft_host_pool = cached
    elif int(getattr(cached, "size", 0)) < int(getattr(primary, "size", -1)):
        # ADDRESSABILITY, NOT EQUALITY -- and #861c is the correction of my own
        # guard rather than of the pools it guards.
        #
        # W37-C, all three ranks, every cutover after the first:
        #   "the draft host pool holds 30519 slots but the target host pool now
        #    bound holds 30518 ... Refusing"
        # The draft half was therefore never re-stamped, binding_generation
        # stayed 1 for 18 flips, and C1 could not pass.
        #
        # The one-slot excess is not drift. Every host pool goes through
        # `pool_host/base.py:146-147`:
        #     self.page_num = self.size // self.page_size + 1
        #     self.size = self.page_num * self.page_size
        # which adds a whole page UNCONDITIONALLY, even when the size is
        # already page-aligned. At `page_size == 1` that is exactly +1, always.
        # So a draft pool derived from a target host pool is SYSTEMATICALLY one
        # slot larger, and an equality check could never hold once a second
        # comparison happened at all.
        #
        # What the shared index space actually requires is that every host
        # index the TARGET can hand out is addressable in the draft pool. A
        # LARGER draft pool satisfies that completely -- the extra tail row is
        # simply never named. Only a SMALLER one is unsafe, and that is what is
        # refused here: it would leave the top of the target's index range
        # writing outside the draft pool, which is the #345 class this guard
        # was written for.
        raise ValueError(
            f"#861: the draft host pool holds {getattr(cached, 'size', None)} "
            f"slots, FEWER than the {getattr(primary, 'size', None)} the target "
            f"host pool now bound can hand out. Draft host indices are the "
            f"target's, so the top of that range would address rows outside the "
            f"draft pool (#345). Refusing -- the draft tier stays disarmed, "
            f"which is the state that held before #861. (A LARGER draft pool is "
            f"fine and expected: pool_host/base.py page-aligns every pool up by "
            f"one whole page, so at page_size 1 a derived pool is always +1.)"
        )

    cc.set_draft_kv_pool(
        reg.pool,
        cached,
        owner_phase=reg.owner_phase,
        binding_generation=reg.generation,
        drafter_identity=reg.identity,
    )
    return True


def _build_draft_host_pool(*, pool, primary, server_args, page_size):
    """Allocate the draft host pool, or None when the pool type is unsupported."""
    from sglang.srt.mem_cache.memory_pool import (
        MHATokenToKVPool,
        MLATokenToKVPool,
    )
    from sglang.srt.mem_cache.pool_host.mha import get_mha_host_pool_cls
    from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost

    # Same slot count as the target host pool, so host indices stay 1-to-1
    # between the target and draft KV caches.
    kw = dict(
        host_to_device_ratio=primary.size / pool.size,
        host_size=0,
        page_size=page_size,
        layout=server_args.hicache_mem_layout,
        allocator_type=server_args.hicache_storage_backend,
    )
    if isinstance(pool, MHATokenToKVPool):
        return get_mha_host_pool_cls(pool)(pool, **kw)
    if isinstance(pool, MLATokenToKVPool):
        return MLATokenToKVPoolHost(pool, **kw)
    logger.warning(
        "Draft pool type %s not supported for HiCache, skipping.",
        type(pool).__name__,
    )
    return None


def maybe_register_hicache_draft(
    *,
    tree_cache: BasePrefixCache,
    draft_worker: BaseTpWorker,
    spec_algorithm: SpeculativeAlgorithm,
    server_args: ServerArgs,
    enable_hierarchical_cache: bool,
    page_size: int,
) -> None:
    """Register draft KV pool with HiCacheController for piggyback L2/L3 ops.

    The BOOT-time call, unchanged in behaviour for every non-flip deployment.
    On a phase-flip boot it still returns early -- ``draft_worker`` is None
    there by #631's design -- and ``rebind_hicache_draft_for_phase`` takes over
    at the first pp->tp cutover, where the drafter and the binding generation
    both exist. See ``resolve_draft_registration`` for why that is the right
    boundary rather than a wider boot-time reach.
    """
    if not enable_hierarchical_cache:
        return

    draft_kv_pool = get_draft_kv_pool(
        draft_worker=draft_worker,
        spec_algorithm=spec_algorithm,
        server_args=server_args,
    )
    if draft_kv_pool is None:
        return

    from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

    pool = draft_kv_pool
    if isinstance(pool, HybridLinearKVPool):
        pool = pool.full_kv_pool

    draft_host_pool = _build_draft_host_pool(
        pool=pool,
        primary=tree_cache.cache_controller.mem_pool_host,
        server_args=server_args,
        page_size=page_size,
    )
    if draft_host_pool is None:
        return

    # owner_phase stays None on this route: an instance whose drafter is the
    # scheduler's own has one phase, so there is no phase in which the draft
    # half must be disarmed. The identity travels either way -- a persisted
    # draft page must never be readable across a drafter change, flip or not.
    tree_cache.cache_controller.set_draft_kv_pool(
        pool,
        draft_host_pool,
        owner_phase=None,
        binding_generation=None,
        drafter_identity=drafter_identity_hash(server_args),
    )


def build_kv_cache(
    *,
    server_args: ServerArgs,
    model_config: ModelConfig,
    tp_worker: BaseTpWorker,
    page_size: int,
    spec_algorithm: SpeculativeAlgorithm,
    attn_tp_cpu_group: ProcessGroup,
    tp_cpu_group: ProcessGroup,
    attn_cp_cpu_group: ProcessGroup,
    enable_metrics: bool,
    enable_kv_cache_events: bool,
    ps: ParallelState,
    tp_group: GroupCoordinator,
    pp_group: GroupCoordinator,
    enable_hierarchical_cache: bool,
) -> KVCacheBuildResult:
    sliding_window_size: Optional[int] = None
    full_tokens_per_layer: Optional[int] = None
    swa_tokens_per_layer: Optional[int] = None
    uses_transformers_backend = (
        get_resolved_model_impl(model_config) == ModelImpl.TRANSFORMERS
    )

    # Hybrid memory pool
    is_hybrid_swa = tp_worker.is_hybrid_swa
    _spec = tp_worker.model_runner.linear_attn_model_spec
    _registry_needs_mamba = _spec.uses_mamba_radix_cache if _spec is not None else False
    is_hybrid_ssm = (
        tp_worker.model_runner.hybrid_gdn_config is not None
        or tp_worker.model_runner.mamba2_config is not None
        or _registry_needs_mamba
        or tp_worker.model_runner.kimi_linear_config is not None
        or tp_worker.model_runner.hybrid_lightning_config is not None
    )

    sliding_window_size = None
    if is_hybrid_swa:
        sliding_window_size = tp_worker.sliding_window_size
        full_tokens_per_layer, swa_tokens_per_layer = (
            tp_worker.get_tokens_per_layer_info()
        )

    req_to_token_pool, token_to_kv_pool_allocator = tp_worker.get_memory_pool()

    disable_radix_cache = server_args.disable_radix_cache or (
        model_config.is_multimodal and uses_transformers_backend
    )
    if disable_radix_cache and not server_args.disable_radix_cache:
        logger.warning(
            "Radix cache is disabled for multimodal models with the "
            "Transformers backend to avoid multimodal prefix-cache mismatches."
        )

    # Decode radix cache is unsupported with hybrid SWA/SSM models —
    # these use specialized memory pools incompatible with the
    # prefix-match-and-lock allocation path.
    if (
        server_args.disaggregation_decode_enable_radix_cache
        and server_args.disaggregation_mode == "decode"
    ):
        if is_hybrid_swa:
            raise ValueError(
                "--disaggregation-decode-enable-radix-cache is incompatible "
                "with sliding window attention (SWA) models"
            )
        if is_hybrid_ssm:
            raise ValueError(
                "--disaggregation-decode-enable-radix-cache is incompatible "
                "with Mamba/SSM models"
            )

    effective_chunked_prefill_size = server_args.chunked_prefill_size
    if model_config.is_multimodal and uses_transformers_backend:
        effective_chunked_prefill_size = None

    params = CacheInitParams(
        disable=disable_radix_cache,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        # When dcp enabled, kv_pool_allocator.page_size is page_size * dcp_size.
        # TreeCache.page_size should keep the same as allocator.page_size to
        # avoid kv page eviction conflicts.
        page_size=(
            page_size
            if not get_parallel().dcp_enabled
            else token_to_kv_pool_allocator.page_size
        ),
        is_eagle=spec_algorithm.is_eagle(),
        tp_cache_group=(
            attn_tp_cpu_group if server_args.enable_dp_attention else tp_cpu_group
        ),
        attn_cp_cache_group=attn_cp_cpu_group,
        attn_tp_cache_group=attn_tp_cpu_group,
        pp_cache_group=pp_group.cpu_group,
        eviction_policy=server_args.radix_eviction_policy,
        enable_metrics=enable_metrics,
        enable_kv_cache_events=enable_kv_cache_events,
        enable_session_radix_cache=server_args.enable_session_radix_cache,
        enable_mamba_extra_buffer=server_args.enable_mamba_extra_buffer(),
        enable_mamba_extra_buffer_lazy=server_args.enable_mamba_extra_buffer_lazy(),
        mamba_slot_reorder=_mamba_slot_reorder_active(server_args),
        pp_rank=ps.pp_rank,
        pp_size=ps.pp_size,
        chunked_prefill_size=effective_chunked_prefill_size,
        sliding_window_size=sliding_window_size,
    )

    tree_cache = create_tree_cache(
        TreeCacheBuildContext(
            server_args=server_args,
            params=params,
            is_hybrid_swa=is_hybrid_swa,
            full_tokens_per_layer=full_tokens_per_layer,
            is_hybrid_ssm=is_hybrid_ssm,
            enable_hierarchical_cache=enable_hierarchical_cache,
            disable_radix_cache=disable_radix_cache,
            effective_chunked_prefill_size=effective_chunked_prefill_size,
            tp_worker=tp_worker,
            model_config=model_config,
            tp_size=ps.tp_size,
            tp_rank=ps.tp_rank,
            tp_group=tp_group,
        )
    )

    embedding_cache_size = envs.SGLANG_VLM_CACHE_SIZE_MB.get()
    init_mm_embedding_cache(embedding_cache_size * 1024 * 1024)

    return KVCacheBuildResult(
        is_hybrid_swa=is_hybrid_swa,
        is_hybrid_ssm=is_hybrid_ssm,
        sliding_window_size=sliding_window_size,
        full_tokens_per_layer=full_tokens_per_layer,
        swa_tokens_per_layer=swa_tokens_per_layer,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        disable_radix_cache=disable_radix_cache,
        tree_cache=tree_cache,
    )
