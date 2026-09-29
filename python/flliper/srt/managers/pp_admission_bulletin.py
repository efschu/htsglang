# Copyright 2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""#631: the match-writer half of PP0's geometry authority (row-fed).

WHY THIS EXISTS (DESIGN_631_PP0_GEOMETRY_AUTHORITY.md, boot-proven chain).
Six boots of 2026-08-31 died at the #631 width guard because two PP ranks
BUILT different batches for the same slot: the batch geometry consumed
rank-local, wall-time-dependent facts (a storage prefetch that had completed
on one rank and not yet on another -- boot 1065sets: PP0 prefix 32768 /
extend 902 vs PP1 prefix 33148 / extend 522, delta 380 = exactly the
prefetch gain).

SECOND CUT (this file's /dev/shm half is RETIRED). The first cut published
PP0's geometry over a /dev/shm side file, because the admission row on the
proxy frame arrived one pass after the receiver had planned (#1059 SITE 4)
and so could not cover the intake pass. Boot 631cut then measured the side
channel's own leak on metal: the file went stale across cutover epochs 5x
in 9 min, and on the stale laps the DEVICE prefix stayed rank-local -- which
is exactly the divergence that killed that boot at lap 15
(PP1 prefix 36575 / PP2 prefix 3807, output stash [2] vs [1]). The row now
arrives BEFORE the plan (receive-before-plan in _event_loop_pp_body), so the
frame itself is the synchronizer, and one mover carries one payload
(Ein-Job-ein-Mover): `plant_from_row` plants the reconciled told-map for
exactly one plan call, and `cap_req_geometry` enforces it at the one site
that writes match geometry onto a request
(schedule_batch.init_next_round_input, the old #1059 SITE 5 position):

  * device prefix above PP0's told  -> truncate down (#930-paired helper;
    always materializable, surplus stays in the tree),
  * async host adoption (host_hit_length, the #988 raise input) capped to
    max(0, told - device_prefix); a rid the row does not name gets cap 0 --
    a rank may never raise its geometry on a fact PP0 has not published.

NEVER BLOCKING, NEVER A COLLECTIVE, NEVER A REFUSAL LOOP: enforcement only
lowers, never waits. Divergence that still occurs stays DETECTED by the
#631 width guard -- this module narrows construction, it does not smooth
detection (RAENGE-NIE-UNEINS).
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

#: Tree-cache attribute the apply site reads. Planted only on downstream
#: ranks, only for the duration of one plan call -- PP0 and non-PP forms
#: never see it and the apply site no-ops on absence.
GEOMETRY_ATTR = "_pp_bulletin_geometry"

#: Cumulative counters, printed on every emitted line (#1047: an acceptance
#: metric may not be sampled without its totals).
_STATS = {
    "planted": 0,
    "capped_device": 0,
    "capped_host": 0,
    "no_bulletin_host_zeroed": 0,
    "under_coverage": 0,
}


def plant_from_row(scheduler, told_by_rid: Optional[Dict[str, int]]) -> None:
    """Downstream: plant the reconciled row's told-map for ONE plan call.

    ``told_by_rid`` is the `effective` map `reconcile_pp_admission_decision`
    returned for THIS pass's received row -- exactly the rids this rank may
    admit, each at PP0's prefix. Planted on the tree cache because the apply
    site (`Req.init_next_round_input`) has the request and the tree but not
    the scheduler. An EMPTY map is planted as {} on purpose: "PP0 named
    nothing adoptable" caps every async raise at 0 rather than leaving the
    rank free to derive its own geometry.
    """
    tree_cache = getattr(scheduler, "tree_cache", None)
    if tree_cache is None:
        return
    planted = {str(rid): int(told) for rid, told in (told_by_rid or {}).items()}
    setattr(tree_cache, GEOMETRY_ATTR, planted)
    _STATS["planted"] += 1


def clear_after_plan(scheduler) -> None:
    """Remove the planted map so no other path can read a stale bulletin."""
    tree_cache = getattr(scheduler, "tree_cache", None)
    if tree_cache is not None and hasattr(tree_cache, GEOMETRY_ATTR):
        setattr(tree_cache, GEOMETRY_ATTR, None)


def cap_req_geometry(req, tree_cache) -> None:
    """Enforce PP0's published ceiling on one request, at the match writer.

    Runs at the old #1059 SITE 5 position in ``init_next_round_input``:
    after the match wrote ``prefix_indices``/``host_hit_length`` and after
    ``cache_protected_len`` was set (so ``truncate_prefix_to``'s #930
    pairing operates on fresh state), and strictly before ``add_one_req``
    derives ``extend_range`` from the current prefix -- the same ordering
    contract the #791 truncate site documents.

    The map is ONLY planted on downstream PP ranks during their own plan
    call; everywhere else this is a getattr miss and a return.
    """
    geometry = getattr(tree_cache, GEOMETRY_ATTR, None) if tree_cache else None
    if geometry is None:
        return  # PP0, non-PP, bulletin disabled, or outside a plan call.
    rid = str(getattr(req, "rid", ""))
    told = geometry.get(rid) if geometry else None
    local = 0 if req.prefix_indices is None else len(req.prefix_indices)
    if told is None:
        # No published fact for this rid (or no bulletin at all): a rank may
        # not raise its geometry on an async fact PP0 has not published.
        hh = int(getattr(req, "host_hit_length", 0) or 0)
        if hh > 0:
            req.host_hit_length = 0
            _STATS["no_bulletin_host_zeroed"] += 1
            n = _STATS["no_bulletin_host_zeroed"]
            if n <= 5 or n % 256 == 0:
                logger.info(
                    "#631 BULLETIN CAP rid=%s: host adoption %d -> 0, no "
                    "published fact (n=%d, capped_host=%d, under=%d). Loss is "
                    "bounded by the prefetch gain; the content stays in the "
                    "host tier for a later, published lap.",
                    rid[:8],
                    hh,
                    n,
                    _STATS["capped_host"],
                    _STATS["under_coverage"],
                )
        return
    told = int(told)
    if local > told:
        # Device prefix above PP0's ceiling (eviction skew, or an earlier
        # uncapped raise): truncate down. Always materializable; the #930
        # pairing keeps cache_protected_len consistent, and add_one_req
        # re-derives extend_range from the truncated prefix afterwards.
        req.truncate_prefix_to(told)
        local = told
        _STATS["capped_device"] += 1
        n = _STATS["capped_device"]
        if n <= 20 or n % 256 == 0:
            logger.warning(
                "#631 BULLETIN CAP rid=%s: device prefix truncated to told=%d "
                "(n=%d, capped_host=%d, under=%d). This is the divergence "
                "direction that killed boots 26-35; the truncate makes this "
                "rank run PP0's geometry instead of its own.",
                rid[:8],
                told,
                n,
                _STATS["capped_host"],
                _STATS["under_coverage"],
            )
    allowed_host = max(0, told - local)
    hh = int(getattr(req, "host_hit_length", 0) or 0)
    if hh > allowed_host:
        req.host_hit_length = allowed_host
        _STATS["capped_host"] += 1
    elif local + hh < told:
        # Under-coverage: PP0 published more than this rank can currently
        # materialize. Counted loudly; the #631 width guard remains the
        # detector if the pass diverges (construction narrowed, detection
        # untouched).
        _STATS["under_coverage"] += 1
        n = _STATS["under_coverage"]
        if n <= 20 or n % 64 == 0:
            logger.warning(
                "#631 BULLETIN UNDER-COVERAGE rid=%s: told=%d, local=%d, "
                "host_hit=%d (n=%d). PP0 raised on a fact this rank has not "
                "materialized yet; if this pass runs divergent the #631 guard "
                "will name it. A rising counter here orders the next slice "
                "(PP0 defers its raise to the group floor).",
                rid[:8],
                told,
                local,
                hh,
                n,
            )


def stats() -> Dict[str, int]:
    return dict(_STATS)
