# Copyright 2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""RU (nf_rank_divergence): TP-uniform match verdicts on an ASYMMETRIC host tier.

WHAT DIED, TWICE (NF D group, TP3, ``--rank-tp-ratio 1,0,0``, dcp_size=1)
------------------------------------------------------------------------
The D group's host tier is not symmetric across TP ranks: TP0 holds every
attention/GDN head (q heads split [24, 0, 0]) and runs the ARENA host pool
(``ArenaMHAHostPool`` 0.05 GB, mamba host 0.35 GB, direct arena writes), while
TP1/TP2 run zero-width plain pools (``MHATokenToKVPoolHost`` 0.00 GB, mamba
host 0.00 GB, staging + store writes). The radix trees are replicas by
insertion, but their host LIFECYCLE is not: a node can be host-backed on TP0
and evicted without backup on TP1/TP2 (boot dkrnfbar1final09260301,
``#1469 EVICT node=237 backuped=True`` on TP0 vs ``backuped=False`` on
TP1/TP2). Two control-flow verdicts were then formed from that rank-local
state, each in front of group collectives:

1. the PREFIX MATCH (anchor verdict): the mamba validator / the #928 refusal
   zeroes the match on the ranks without the state and keeps it on the rank
   with it -> TP0 takes #988 LOADBACK + the tail skip-extend, TP1/TP2 extend
   from 0 -> different forwards -> TP0 hangs in chain-recv (rc9i, rid
   weg2-32-26).
2. the PREFETCH REGISTRATION: with a 192-token span TP0 declines as
   ``too_short`` and returns, TP1/TP2 register a 29,888-token prefetch ->
   TP1/TP2 enter ``can_terminate_prefetch`` (3 x int32, 4-byte gloo chunks)
   while TP0 is already in the packed MIN of ``_update_uniform_pool_budget``
   (int64, 248-byte chunks) -> gloo ``248 vs 4`` (rc9k, rid weg2-21-19).

The #580 participation vote exists for (2) and would have made registration
uniform, but it is gated on ``uneven_dcp_active()`` -- False on this form,
because the asymmetry here comes from the rank-TP plan, not from a DCP token
vector. (1) had no agreement at all: the #823 head vote MIN-reduces the match
length but only the X gate and the queue order read it; admission used the
rank-local match.

THE TWO CLOSES, both on reduces that already run
-------------------------------------------------
* :func:`host_tier_asymmetric` widens the #580 predicate to an uneven
  ``--rank-tp-ratio`` plan, so the prefetch participation vote (MIN == AND) is
  in force on this form. Even TP (all ratios equal, no DCP vector) stays
  byte-identical.
* The USABLE-MATCH arm rides the same packed MIN as the head-order arm, on the
  same canonical head. Each rank votes its match length, or 0 when the match
  ends on a node whose recurrent anchor it cannot use (the #928 (a)/(b) test,
  evaluated WITHOUT the COW side effects -- the vote walk runs with
  ``cow_mamba=False`` and so never reaches #928 itself). At admission, inside
  ``MambaComponent.finalize_match_result`` and BEFORE the COW, a rank whose
  local match is > 0 while the group's usable match is 0 zeroes its match
  exactly the way #928 does. Zero is materializable on every rank (no state
  needed), so this is the delay-never-force direction: slower (a re-prefill
  the rank could have skipped), never wrong.

WHAT THIS DOES NOT CLOSE, named rather than implied: a group usable match
strictly between 0 and the local match is only COUNTED
(``RU FLOOR ABOVE-GROUP``). The benign population there is host-hit skew
that the synced prefetch extent + #988 LOADBACK already converge (rc9i
weg2-33-27: host_hit 18304 vs 15744, identical extend); truncating to a
non-zero group depth would need a recurrent state at exactly that depth on
this rank, which only a second agreement could establish.

PURE ON PURPOSE: every decision is a function of its arguments, so the tests
drive the real verdict with mock collectives instead of grepping for it.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)

#: Tree-cache attribute the admission site reads. Planted by the scheduler for
#: exactly one plan call (``get_new_batch_prefill``) and cleared in its
#: ``finally``; every other match (the vote walk itself, the intake re-match,
#: the deferral retry) sees ``None`` and is untouched.
TREE_ATTR = "_tp_match_floor_group"

#: "This rank does not hold this rid" / unused slot. MIN-reduces to itself, so
#: one rank missing a rid removes the group's opinion for it (abstain).
ABSENT = -1

_STATS = {"zeroed": 0, "above_group": 0, "unusable_votes": 0}


# --------------------------------------------------------------------------
# Close 1: the #580 predicate
# --------------------------------------------------------------------------


def rank_tp_plan_uneven(rank_tp_ratio: Any) -> bool:
    """True for an installed ``--rank-tp-ratio`` list whose entries differ.

    ``None``, ``"auto"`` (unresolved) and all-equal lists are even: they give
    every rank the same heads, pools and write path, which is the stock
    HiCache premise the non-voting path relies on.
    """
    if not isinstance(rank_tp_ratio, (list, tuple)) or len(rank_tp_ratio) < 2:
        return False
    try:
        values = [int(v) for v in rank_tp_ratio]
    except (TypeError, ValueError):
        return False
    return len(set(values)) > 1


def host_tier_asymmetric(*, dcp_uneven: bool, server_args: Any) -> bool:
    """Can the per-rank host tiers of one TP group diverge BY CONSTRUCTION?

    Uneven DCP (the original #580 condition) or an uneven rank-TP plan. Under
    ``--rank-tp-ratio 1,0,0`` the ranks do not even share a host-pool class.
    """
    if dcp_uneven:
        return True
    return rank_tp_plan_uneven(getattr(server_args, "rank_tp_ratio", None))


# --------------------------------------------------------------------------
# Close 2: the usable-match arm
# --------------------------------------------------------------------------


def anchor_unusable(tree_cache: Any, node: Any) -> bool:
    """The #928 test on ``node``, side-effect free.

    (a) no recurrent state on device or host, or (b) a device state whose
    bytes the computing phase cannot read. Mirrors
    ``MambaComponent.finalize_match_result`` exactly; never raises (a vote may
    never break the reduce -- an unreadable node is voted as usable, i.e. the
    pre-RU behaviour for that rid).
    """
    try:
        if node is None or node is getattr(tree_cache, "root_node", None):
            return False
        supports = getattr(tree_cache, "supports_mamba", None)
        if supports is None or not supports():
            return False
        from sglang.srt.mem_cache.unified_cache_components.tree_component import (
            ComponentType,
        )

        data = node.component_data
        if len(data) <= int(ComponentType.MAMBA):
            return False
        cd = data[ComponentType.MAMBA]
        value = getattr(cd, "value", None)
        host_value = getattr(cd, "host_value", None)
        if value is None and host_value is None:
            return True
        if value is not None:
            from sglang.srt.mem_cache.mamba_state_pool import anchor_bytes_reachable

            if not anchor_bytes_reachable(tree_cache, value):
                return True
        return False
    except Exception:  # noqa: BLE001 - a vote may never break the reduce
        return False


def local_usable_matches(
    tree_cache: Any, by_rid: Mapping[str, Any], matches: Mapping[str, int]
) -> Dict[str, int]:
    """This rank's usable-match vote, from the head vote's own walk.

    ``matches`` is what ``_local_head_prefix_matches`` just measured (rids it
    could not price are absent there and stay absent here). The walk left
    ``best_match_node`` on each request; a match that ends on an unusable
    anchor is voted 0 -- what admission's #928 would reduce it to.
    """
    out: Dict[str, int] = {}
    for rid, n in matches.items():
        n = int(n)
        req = by_rid.get(rid)
        if n > 0 and req is not None and anchor_unusable(
            tree_cache, getattr(req, "best_match_node", None)
        ):
            _STATS["unusable_votes"] += 1
            n = 0
        out[rid] = n
    return out


def build_usable_match_payload(
    canonical: Sequence[str], local_usable: Mapping[str, int], slots: int
) -> List[int]:
    """One slot per canonical rid; absent rids and unused slots ride ABSENT."""
    payload = [ABSENT] * slots
    for i, rid in enumerate(list(canonical)[:slots]):
        payload[i] = int(local_usable.get(rid, ABSENT))
    return payload


def decode_group_usable(
    canonical: Sequence[str], reduced: Sequence[int]
) -> Dict[str, int]:
    """rid -> group usable match, for rids every rank held (MIN > ABSENT)."""
    out: Dict[str, int] = {}
    for rid, value in zip(list(canonical), list(reduced)):
        value = int(value)
        if value > ABSENT:
            out[str(rid)] = value
    return out


def plant(tree_cache: Any, group_usable: Optional[Dict[str, int]]) -> None:
    if tree_cache is not None:
        setattr(tree_cache, TREE_ATTR, group_usable)


def clear(tree_cache: Any) -> None:
    if tree_cache is not None and getattr(tree_cache, TREE_ATTR, None) is not None:
        setattr(tree_cache, TREE_ATTR, None)


def floor_verdict(local_match: int, group_usable: Optional[int]) -> str:
    """``no_opinion`` | ``agree`` | ``zero`` | ``above_group``.

    ``zero`` is the only verdict that acts, and it acts only in the direction
    every rank can materialize.
    """
    if group_usable is None:
        return "no_opinion"
    local_match = int(local_match)
    group_usable = int(group_usable)
    if local_match <= 0 or local_match <= group_usable:
        return "agree"
    if group_usable == 0:
        return "zero"
    return "above_group"


def group_floor_zeroes(tree_cache: Any, req: Any, result: Any) -> bool:
    """Admission-site verdict: must THIS rank zero its match for ``req``?

    Called from ``MambaComponent.finalize_match_result`` with the match result
    as the full component produced it (device rows + host hit), before any COW.
    """
    group = getattr(tree_cache, TREE_ATTR, None) if tree_cache is not None else None
    if not group or req is None:
        return False
    rid = str(getattr(req, "rid", "") or "")
    local = 0
    try:
        di = getattr(result, "device_indices", None)
        local = (0 if di is None else len(di)) + int(
            getattr(result, "host_hit_length", 0) or 0
        )
    except Exception:  # noqa: BLE001
        return False
    verdict = floor_verdict(local, group.get(rid))
    if verdict == "zero":
        _STATS["zeroed"] += 1
        n = _STATS["zeroed"]
        if n <= 20 or n % 256 == 0:
            logger.warning(
                "RU FLOOR ZERO rid=%s local_match=%d group_usable=0 (n=%d): some "
                "TP rank cannot use this prefix (no recurrent anchor, or the "
                "store/host walk refused it), so every rank re-prefills from 0 "
                "instead of this one alone resuming -- the #928 shape, taken "
                "group-uniformly (raenge-nie-uneins).",
                rid[:16],
                local,
                n,
            )
        return True
    if verdict == "above_group":
        _STATS["above_group"] += 1
        n = _STATS["above_group"]
        if n <= 20 or n % 256 == 0:
            logger.info(
                "RU FLOOR ABOVE-GROUP rid=%s local_match=%d group_usable=%d "
                "(n=%d): counted, not acted on -- the extent/#988 LOADBACK "
                "convergence owns this band; a split forward after this line "
                "names the residual.",
                rid[:16],
                local,
                int(group[rid]),
                n,
            )
    return False


def stats() -> Dict[str, int]:
    return dict(_STATS)
