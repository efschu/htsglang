"""#1068 (WEG 1 slice 2): the two pieces both tree caches share around the
speculative-prefetch budget, so that neither cache carries a twin.

The budget itself is ``HiCacheController.prefetch_capacity_limit`` -- a
property of the host pool the controller is bound to right now (upstream
:575-584: the buffer_only fraction for a staging tier, the cache-mode half
for a retention tier; the brake that consumes it is the upstream cache-mode
counter form in both roles, see ``prefetch_rate_limited``). What lives here:

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


def host_pool_anchor(cache_controller: Any) -> Any:
    """The KV host pool a controller is bound to, a pool GROUP unwrapped to
    its anchor entry's pool (``None`` when no pool is bound).

    ONE unwrap for every reader that stamps or prints the pool identity: the
    prefetch registration stamp (``_host_pool_id_at_reg``), the
    PREFETCH-COMPLETE free-site diagnostic, the #915 L1/L2 terms and
    ``host_pool_identity`` (slice 4 fix: those were four copies of the same
    two lines, the A12.5 cleanup candidate)."""
    pool = getattr(cache_controller, "mem_pool_host", None)
    anchor = getattr(getattr(pool, "anchor_entry", None), "host_pool", None)
    return anchor or pool


def host_pool_identity(cache_controller: Any) -> int:
    """``id()`` of the KV host pool a controller is bound to (see
    ``host_pool_anchor``)."""
    return id(host_pool_anchor(cache_controller))


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


def prefetch_residency(tree_cache: Any) -> Any:
    """The host-pool terms the #915 prefetch gate ITSELF applies, for a reader
    outside this process (FIX 4, round 4).

    WHY THIS EXISTS.  The Weg-2 front hands group D ``--d-bs`` seats at once
    and had no channel carrying D's real host-tier state, so its aggregate
    admission gate priced the pool by "budget minus what I admitted" -- a
    number that is back to zero at every D epoch and therefore blind to the
    standing residency that actually refuses the store read.  Measured, boot
    weg2sc1 2026-09-07 20:14:17Z, the same second on both logs: the front's
    proxy said 27,466 rows were free while D printed ``#915 PREFETCH REFUSED
    reason=vote_negative need=8629 available=5418 occupied=25100 limit=27466``.
    The indicator was 25,100 rows optimistic.  This function publishes the
    terms of THAT line, so the front prices the pool by the pool.

    WHICH TERMS BIND, in the order the gate applies them
    (``unified_radix_cache._prefetch_from_storage``):

    * ``available`` -- ``mem_pool_host.available_size()``.  The ALLOC term:
      a prefetch of ``need`` rows calls ``alloc(need)`` and, under the #580
      participation vote, a failure on ANY rank lowers the group vote to
      ``vote_negative`` with no truncation.  This is the term that refused
      boot weg2sc1 (5418 < 8629), and it is the only one of the three that
      sees rows retained by earlier requests as well as rows locked by
      registered prefetches.
    * ``occupied``/``limit`` -- the RATE brake
      (``HiCacheController.prefetch_rate_limited``): registered-but-not-yet-
      completed prefetch tokens against ``prefetch_capacity_limit``.  It
      bounds CONCURRENT spans, not residency (that method's own docstring).

    RANK-UNIFORMITY, honestly.  ``limit`` is rank-uniform BY CONSTRUCTION
    with ``--hicache-size`` (``sync_fixed_hicache_size`` MIN-syncs the pool;
    see ``prefetch_capacity_limit``'s docstring).  ``available`` and
    ``occupied`` are not guaranteed uniform -- they are per-rank readings,
    and only the control rank answers ``/server_info``.  So this is a
    READING, never a verdict: the front's use of it is an ESTIMATE in exactly
    the sense law 4's X already is (the front estimates, D enforces).  When
    the estimate is optimistic the request meets D's own gate and is refused
    there, which is the behaviour that existed before this channel -- the
    reading can make the front more conservative, never less correct.

    THE TWIN, NAMED (Ein-Job-ein-Mover).  ``UnifiedRadixCache._prefetch_line_terms``
    reads the same three terms for the ``#915`` log lines and is NOT folded
    into this function: it degrades PER TERM (a missing one prints -1 and the
    other two still speak), while a reading that is only partly true is worse
    than no reading for a caller that decides on it, so this one is
    all-or-nothing.  The two are pinned against each other by
    ``test_a7_the_reading_is_group_d_s_own_915_terms`` -- if they ever drift,
    the front would decide on one arithmetic while D refuses on another.

    Diagnostic form (#1035): a collaborator that lacks a term is reported as
    ``None`` rather than raising.  ``prefetch_capacity_limit`` RAISES with no
    pool bound (A12.4), which is exactly a moment when there is nothing to
    publish, so the whole reading is ``None`` then.
    """
    cc = getattr(tree_cache, "cache_controller", None)
    if cc is None:
        return None
    try:
        pool = host_pool_anchor(cc)
        if pool is None:
            return None
        from sglang.srt.mem_cache.hicache_phase_binding import (
            bound_phase,
            current_generation,
        )

        return {
            "available": int(pool.available_size()),
            "occupied": int(cc.prefetch_tokens_occupied),
            "limit": int(cc.prefetch_capacity_limit),
            "size": int(getattr(pool, "size", 0) or 0),
            "threshold": int(getattr(tree_cache, "prefetch_threshold", 0) or 0),
            "pool_id": host_pool_identity(cc),
            "phase": bound_phase(),
            "generation": int(current_generation()),
        }
    except Exception:  # noqa: BLE001 - an instrument may never break a poll
        return None
