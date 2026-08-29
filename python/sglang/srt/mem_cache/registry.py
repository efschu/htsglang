"""Registry for pluggable RadixCache factories.

If `--radix-cache-backend` is unset (by default), the built-in selection
chain is used to pick a cache implementation.

To plug in a custom backend, register it under a string name via
`register_radix_cache_backend(name, factory)`, then select it with
`--radix-cache-backend <name>` (the flag accepts only registered names).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional

from sglang.srt.environ import envs
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.utils.tensor_bridge import use_mlx

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


@dataclass
class TreeCacheBuildContext:
    """Radix Cache construction arguments."""

    server_args: ServerArgs
    params: CacheInitParams
    is_hybrid_swa: bool
    is_hybrid_ssm: bool
    enable_hierarchical_cache: bool
    disable_radix_cache: bool
    effective_chunked_prefill_size: Optional[int]
    tp_worker: Any
    model_config: ModelConfig
    tp_size: int
    tp_rank: int
    tp_group: Any
    full_tokens_per_layer: Optional[int] = None


RadixCacheFactory = Callable[[TreeCacheBuildContext], BasePrefixCache]

_RADIX_CACHE_REGISTRY: dict[str, RadixCacheFactory] = {}


def register_radix_cache_backend(name: str, factory: RadixCacheFactory) -> None:
    """Register a radix-cache factory under `name`.

    Raises ValueError if `name` is empty/whitespace-only or already
    registered.
    """
    if not name.strip():
        raise ValueError(
            f"register_radix_cache_backend: name must be non-empty, got {name!r}"
        )
    if name in _RADIX_CACHE_REGISTRY:
        raise ValueError(
            f"register_radix_cache_backend: {name!r} is already registered"
        )
    _RADIX_CACHE_REGISTRY[name] = factory


def get_radix_cache_factory(name: str) -> Optional[RadixCacheFactory]:
    return _RADIX_CACHE_REGISTRY.get(name)


def registered_radix_cache_backends() -> list[str]:
    return list(_RADIX_CACHE_REGISTRY.keys())


def default_radix_cache_factory(ctx: TreeCacheBuildContext) -> BasePrefixCache:
    """Built-in Radix Cache selection chain."""
    server_args = ctx.server_args
    params = ctx.params

    if ctx.effective_chunked_prefill_size is not None and ctx.disable_radix_cache:
        if not ctx.is_hybrid_swa:
            from sglang.srt.mem_cache.chunk_cache import ChunkCache

            return ChunkCache(params)
        if ctx.full_tokens_per_layer == 0:
            from sglang.srt.mem_cache.chunk_cache import PureSWAChunkCache

            return PureSWAChunkCache(params)
        from sglang.srt.mem_cache.chunk_cache import SWAChunkCache

        return SWAChunkCache(params)

    if envs.SGLANG_EXPERIMENTAL_CPP_RADIX_TREE.get():
        # lazy import to avoid JIT overhead
        from sglang.srt.mem_cache.radix_cache_cpp import RadixCacheCpp

        logger.info("Using experimental C++ radix tree implementation.")
        return RadixCacheCpp(params=params, server_args=server_args)

    if envs.SGLANG_ENABLE_UNIFIED_RADIX_TREE.get() or use_mlx():
        return _create_unified_radix_cache(ctx, server_args, params)

    # Hybrid SSM/SWA under hierarchical cache ALWAYS takes UnifiedRadixCache.
    # HiMambaRadixCache has no construction site anywhere (see its module docstring).
    if ctx.enable_hierarchical_cache:
        if ctx.is_hybrid_ssm or ctx.is_hybrid_swa:
            # HybridModel launches HiCache via UnifiedRadixCache by default.
            return _create_unified_radix_cache(ctx, server_args, params)
        else:
            from sglang.srt.mem_cache.hiradix_cache import HiRadixCache

            cache = HiRadixCache(params=params, server_args=server_args)
        ctx.tp_worker.register_hicache_layer_transfer_counter(
            cache.cache_controller.layer_done_counter
        )
        return cache

    if ctx.is_hybrid_swa:
        if ctx.full_tokens_per_layer == 0:
            from sglang.srt.mem_cache.pure_swa_radix_cache import PureSWARadixCache

            return PureSWARadixCache(params=params)
        from sglang.srt.mem_cache.swa_radix_cache import SWARadixCache

        return SWARadixCache(params=params)

    if ctx.is_hybrid_ssm:
        from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache

        return MambaRadixCache(params)

    if server_args.enable_lmcache:
        from sglang.srt.mem_cache.storage.lmcache.lmc_radix_cache import (
            LMCRadixCache,
        )

        return LMCRadixCache(
            params=params,
            model_config=ctx.model_config,
            tp_size=ctx.tp_size,
            rank=ctx.tp_rank,
            tp_group=ctx.tp_group,
        )

    if server_args.enable_flexkv:
        # Importing the package side-effect registers the explicit
        # ``--radix-cache-backend=flexkv`` factory; we then call the
        # factory directly so --enable-flexkv stands on its own.
        import os

        from sglang.srt.mem_cache.storage.flexkv import _flexkv_factory

        # Honor a CLI --flexkv-config-file by forwarding it via the env
        # var that FlexKV's config loader actually reads.
        if server_args.flexkv_config_file and not os.environ.get("FLEXKV_CONFIG_PATH"):
            os.environ["FLEXKV_CONFIG_PATH"] = server_args.flexkv_config_file
        return _flexkv_factory(ctx)

    from sglang.srt.mem_cache.radix_cache import RadixCache

    return RadixCache(params)


def _create_unified_radix_cache(
    ctx: TreeCacheBuildContext,
    server_args: ServerArgs,
    params: CacheInitParams,
) -> BasePrefixCache:
    """Initialize a UnifiedRadixCache with proper components and optional HiCache."""
    from sglang.srt.mem_cache.unified_cache_components import ComponentType
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    tree_components = [ComponentType.FULL]
    if ctx.is_hybrid_swa:
        tree_components.append(ComponentType.SWA)
    if ctx.is_hybrid_ssm:
        tree_components.append(ComponentType.MAMBA)

    params.tree_components = tuple(tree_components)
    if use_mlx() and ctx.is_hybrid_ssm:
        from sglang.srt.hardware_backend.mlx.kv_cache.auxiliary_state import (
            MlxAuxiliaryStateComponent,
        )

        params.component_registry_override = {
            ComponentType.MAMBA: MlxAuxiliaryStateComponent,
        }
    cache = UnifiedRadixCache(params)
    if ctx.enable_hierarchical_cache:
        cache.init_hicache(server_args, params)
        ctx.tp_worker.register_hicache_layer_transfer_counter(
            cache.cache_controller.layer_done_counter
        )
    return cache


def create_tree_cache(ctx: TreeCacheBuildContext) -> BasePrefixCache:
    """Route to the matching factory to construct Radix Cache."""
    name = ctx.server_args.radix_cache_backend
    if name:
        factory = get_radix_cache_factory(name)
        if factory is None:
            raise ValueError(
                f"--radix-cache-backend={name!r} is not registered. "
                f"Registered backends: {registered_radix_cache_backends()}. "
                "External backends must call register_radix_cache_backend(...) at import time."
            )
        cache = factory(ctx)
        source = f"registered({name!r})"
    else:
        cache = default_radix_cache_factory(ctx)
        source = "default"

    streaming_wrapped = False
    if (
        ctx.server_args.enable_streaming_session
        and not cache.supports_streaming_session()
    ):
        from sglang.srt.session.streaming_session import StreamingSession

        cache = StreamingSession(cache)
        streaming_wrapped = True

    logger.info(
        "Tree cache initialized: source=%s impl=%s hybrid_swa=%s hybrid_ssm=%s "
        "hierarchical=%s streaming_wrapped=%s",
        source,
        type(cache).__name__,
        ctx.is_hybrid_swa,
        ctx.is_hybrid_ssm,
        ctx.enable_hierarchical_cache,
        streaming_wrapped,
    )
    _log_hicache_impl_banner(cache, ctx)
    return cache


def _hicache_impl_fields(cache) -> "dict[str, str]":
    """Name the HiCache implementation ACTUALLY bound, half by half.

    One-path discipline (user order 2026-08-29): this fork carries two
    lineages of the hierarchical cache -- the ``MambaRadixCache``/
    ``HiRadixCache`` line and the ``UnifiedRadixCache`` + component line --
    and the selection chain in ``default_radix_cache_factory`` is long enough
    that reading the launch flags does NOT tell you which one a boot got. The
    concrete miss this closes: ``mamba_radix_cache.py`` line numbers were
    carried through a whole handover as the defect site while the boot that
    produced the defect ran ``UnifiedRadixCache``, whose ``MambaComponent``
    is a different file. Without a line naming the bound classes there is no
    cache verdict, only a guess about which file the evidence belongs to.

    Reports CLASSES, never flags: a flag says what was asked for, a class
    says what answered. ``none`` means that half is genuinely absent (no
    controller, no storage backend, no mamba lineage), which is itself a
    verdict and not a gap in the instrument.
    """
    unwrapped = getattr(cache, "tree_cache", cache)  # StreamingSession wrapper
    controller = getattr(unwrapped, "cache_controller", None)
    fields = {"tree": type(unwrapped).__name__}

    if controller is None:
        fields["write"] = "none"
        fields["read"] = "none"
        fields["store"] = "none"
    else:
        cname = type(controller).__name__
        fields["write"] = f"{cname}:{getattr(controller, 'write_policy', '?')}"
        fields["read"] = f"{cname}:{getattr(controller, 'io_backend', '?')}"
        backend = getattr(controller, "storage_backend", None)
        if backend is None or not getattr(controller, "enable_storage", False):
            fields["store"] = "none"
        else:
            fields["store"] = type(backend).__name__

    # Mamba half: the component in the unified line, the class itself in the
    # MambaRadixCache line, `none` when the tree carries no mamba lineage.
    mamba = "none"
    components = getattr(unwrapped, "components", None)
    if isinstance(components, dict):
        for comp in components.values():
            ct = getattr(comp, "component_type", None)
            if ct is not None and getattr(ct, "is_mamba", False):
                mamba = type(comp).__name__
                break
    if mamba == "none" and getattr(unwrapped, "supports_mamba", lambda: False)():
        mamba = type(unwrapped).__name__
    fields["mamba"] = mamba

    host = None
    if isinstance(components, dict):
        for comp in components.values():
            ct = getattr(comp, "component_type", None)
            if ct is not None and getattr(ct, "is_mamba", False):
                host = getattr(comp, "_mamba_pool_host", None)
                break
    if host is None:
        host = getattr(unwrapped, "mamba_pool_host", None)
    fields["mambahost"] = "none" if host is None else type(host).__name__

    kvhost = getattr(controller, "mem_pool_host", None) if controller else None
    fields["kvhost"] = "none" if kvhost is None else type(kvhost).__name__

    # #1016 BRACKET THE BOOT-TIME POOL SHORTFALL. Boots 7 and 10 both died on
    # the first idle with EXACTLY leaked_full_pages={1..10} and
    # leaked_mamba_pages={2} -- boot 7 with a request in flight, boot 10 with
    # zero /generate ever received, so the request was a bystander and the
    # deficit is the same ten rows both times. Both free lists are built as
    # `arange(1, size + 1)` with slot 0 reserved, so available == total at
    # construction and those ten rows were HANDED OUT during boot by an owner
    # the census cannot name. Reading both pools here, at tree-cache
    # construction, brackets it: a shortfall already visible on this line was
    # taken before the tree existed; a full pool here moves the owner later in
    # the boot sequence. Two numbers, no behaviour.
    try:
        alloc = getattr(unwrapped, "token_to_kv_pool_allocator", None)
        if alloc is not None:
            fields["full_pool"] = f"{alloc.available_size()}/{alloc.size}"
        rtp = getattr(unwrapped, "req_to_token_pool", None)
        mamba_alloc = getattr(rtp, "mamba_allocator", None) if rtp else None
        if mamba_alloc is not None:
            fields["mamba_pool"] = (
                f"{mamba_alloc.available_size()}/{mamba_alloc.size}"
            )
    except Exception as exc:  # noqa: BLE001 - a banner must never break a boot
        fields["full_pool"] = f"unreadable({exc!r})"
    return fields


def _log_hicache_impl_banner(cache, ctx: TreeCacheBuildContext) -> None:
    """Emit the single grepable HICACHE-IMPL line for this boot."""
    try:
        fields = _hicache_impl_fields(cache)
    except Exception as exc:  # noqa: BLE001 - a banner must never break a boot
        logger.warning("HICACHE-IMPL banner unavailable: %r", exc)
        return
    logger.info(
        "HICACHE-IMPL tree=%s write=%s read=%s mamba=%s store=%s "
        "mambahost=%s kvhost=%s hierarchical=%s full_pool=%s mamba_pool=%s",
        fields["tree"],
        fields["write"],
        fields["read"],
        fields["mamba"],
        fields["store"],
        fields["mambahost"],
        fields["kvhost"],
        ctx.enable_hierarchical_cache,
        fields.get("full_pool", "n/a"),
        fields.get("mamba_pool", "n/a"),
    )
