"""#1068 (WEG 1 slice 2): the two pieces both tree caches share around the
speculative-prefetch budget, so that neither cache carries a twin.

The budget itself is ``HiCacheController.prefetch_capacity_limit`` -- a
property of the host pool the controller is bound to right now (upstream
buffer_only form). What lives here:

* the G8 refusal: under uneven DCP with ``tp_world_size > 1`` the per-rank
  host pools are ratio-sized from per-rank DEVICE pools and therefore
  differ per rank, so a per-rank budget property would make
  ``prefetch_rate_limited()`` answer differently per rank -- the #580 desync
  (a rank that skips the prefetch registration skips the collectives its
  peers enter). The fork used to repair that with a MIN all_reduce over the
  pool sizes at two init sites (the symmetrize twins, deleted in #1068
  slice 2); the upstream-minimal answer is to REQUIRE ``--hicache-size``
  there, because a fixed
  size is MIN-synced across ranks by ``sync_fixed_hicache_size``
  (pool_host/base.py) and the property is then uniform by construction.
* the L3 log line, emitted once at boot and once after every cutover
  rebind, so the acceptance can read the budget that is actually in force.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def refuse_ratio_sized_pools_under_symmetric_prefetch(
    *, symmetric: bool, server_args: Any
) -> None:
    """Raise when storage prefetch is group-decided but the host pools are
    ratio-sized (G8). No-op otherwise."""
    if not symmetric:
        return
    if int(getattr(server_args, "hicache_size", 0) or 0) > 0:
        return
    raise ValueError(
        "#1068 HiCache storage under uneven DCP with tp_world_size>1 requires "
        "--hicache-size (absolute): ratio-sized host pools differ per rank "
        "(unified_radix_cache.py _hicache_prefetch_symmetric) and a "
        "rank-divergent prefetch gate is the #580 desync"
    )


def host_pool_identity(cache_controller: Any) -> int:
    """``id()`` of the KV host pool a controller is bound to, unwrapping a
    pool group to its anchor entry the way the prefetch registration stamp
    does (unified_radix_cache.py, ``_host_pool_id_at_reg``)."""
    pool = getattr(cache_controller, "mem_pool_host", None)
    anchor = getattr(getattr(pool, "anchor_entry", None), "host_pool", None)
    return id(anchor or pool)


def log_prefetch_limit(cache_controller: Any, *, site: str) -> None:
    """L3: ``#915 PREFETCH LIMIT now=...`` from the live property.

    Every term is named: the budget, the fraction and the pool size it was
    derived from, the role that chose the fraction, the pool identity, and
    the binding phase and generation the readers currently carry.
    """
    if cache_controller is None:
        return
    try:
        from sglang.srt.mem_cache.hicache_phase_binding import (
            bound_phase,
            current_generation,
        )

        pool = getattr(cache_controller, "mem_pool_host", None)
        logger.info(
            "#915 PREFETCH LIMIT now=%d (fraction=%.1f x host size %d) role=%s "
            "pool_id=%d phase=%s generation=%d site=%s",
            int(cache_controller.prefetch_capacity_limit),
            float(cache_controller.prefetch_capacity_fraction),
            int(getattr(pool, "size", 0) or 0),
            getattr(cache_controller, "host_role", "?"),
            host_pool_identity(cache_controller),
            bound_phase(),
            int(current_generation()),
            site,
        )
    except Exception:  # noqa: BLE001 - an instrument may never break a boot or a rebind
        logger.warning("#915 PREFETCH LIMIT line could not be formed at %s", site)
