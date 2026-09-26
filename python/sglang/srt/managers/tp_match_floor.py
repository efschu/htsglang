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

H96 (rc9l, boot dkrnfbar1rc9l09260540, rid weg2-21-21): the band strictly
between 0 and the local match was only COUNTED here, on the claim that the
synced prefetch extent + #988 LOADBACK converge it. They do not: TP0 matched
19712 on a host-backed anchor (MAMBA-HOST-RESUME depth=19712), TP1/TP2 16384;
the group usable match was 16384; TP0 logged ``RU FLOOR ABOVE-GROUP``, set
``#1042 EXTENT 19712`` and ``#988 LOADBACK prefix moved to 19712`` while
TP1/TP2 took 16384 -- different extend shapes, TP0 JIT-built alone, the group
stood until the watchdog. The band is therefore ACTED ON now:
:func:`group_floor_cap` names the group depth and the admission site
re-matches this rank's key cut to exactly that depth
(:func:`rematch_at_group_depth`). The group value is the MIN of every rank's
usable vote, so on the ranks that voted it the depth carries a usable anchor;
on the asymmetric host tier TP0's host coverage is a superset of TP1/TP2's,
so the same node carries one here too. If the re-match cannot reach exactly
that depth on this rank, the rank stops LOUDLY (:class:`RankFloorCapMiss`)
instead of resuming alone -- a named death, never a silent split
(raenge-nie-uneins).

PURE ON PURPOSE: every decision is a function of its arguments, so the tests
drive the real verdict with mock collectives instead of grepping for it.
"""

from __future__ import annotations

import logging
from array import array
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


# --------------------------------------------------------------------------
# H97: the MAX arm and the realizability round (rc9m, dkrnfbar1rc9m09260642)
# --------------------------------------------------------------------------
#
# rc9m, rid weg2-18-15: TP0 voted 18112 (host anchor there), TP1/TP2 15552;
# group MIN 15552. H96 cut TP0's key to 15552 -- and TP0 holds NO recurrent
# anchor at 15552 (its host-row release policy differs from TP1/TP2's, so the
# ranks' anchors sit at DIFFERENT depths, not in a superset): capped_match=0,
# H96 CAP-MISS, D dead. The MIN of each rank's own deepest usable depth is
# not a depth every rank can USE. So the group now also learns the MAX (a
# negated arm on the same reduce) and, only for rids where MIN < MAX -- the
# skew every rank sees identically, so every rank takes the same path -- runs
# ONE more small MIN over "I can realize exactly the group depth" (a
# side-effect-free match on the key cut to that depth: length == depth and a
# usable anchor there). A 0 anywhere plants group usable 0 for that rid: every
# rank re-prefills from 0 (slower, never wrong) instead of one rank dying.

#: MIN-neutral value of the MAX arm for a rid this rank does not hold (the
#: MIN arm already abstains for it; every present vote is <= 0).
MAX_ARM_NEUTRAL = 1


def build_usable_max_payload(
    canonical: Sequence[str], local_usable: Mapping[str, int], slots: int
) -> List[int]:
    """The negated usable vote, so the same MIN reduce yields the group MAX."""
    payload = [MAX_ARM_NEUTRAL] * slots
    for i, rid in enumerate(list(canonical)[:slots]):
        if rid in local_usable:
            payload[i] = -int(local_usable[rid])
    return payload


def decode_group_max(canonical: Sequence[str], reduced: Sequence[int]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for rid, value in zip(list(canonical), list(reduced)):
        value = int(value)
        if value <= 0:
            out[str(rid)] = -value
    return out


def skewed_rids(
    group_usable: Mapping[str, int], group_max: Mapping[str, int]
) -> Dict[str, int]:
    """rid -> group depth for every rid with 0 < MIN < MAX. Identical on every
    rank (both inputs come from the same reduce)."""
    out: Dict[str, int] = {}
    for rid, g in group_usable.items():
        g = int(g)
        if g > 0 and int(group_max.get(rid, g)) > g:
            out[rid] = g
    return out


def can_realize(tree_cache: Any, req: Any, depth: int) -> bool:
    """Can THIS rank admit exactly ``depth`` tokens of ``req``'s prefix?

    Side-effect free on the request (no ``match_prefix_for_req``: it writes
    the req's match fields): a plain ``match_prefix`` on the key cut to
    ``depth`` (bigram-aware -- ``depth`` bigrams span ``depth + 1`` tokens),
    ``cow_mamba=False`` so no COW and no #928 refusal side effects. Never
    raises: an unpriceable rank votes 0 (the safe direction)."""
    try:
        from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
        from sglang.srt.mem_cache.radix_cache import RadixKey

        token_ids = list(req.origin_input_ids) + list(req.output_ids)
        span = int(depth) + (1 if getattr(tree_cache, "is_eagle", False) else 0)
        if span > len(token_ids):
            return False
        result = tree_cache.match_prefix(
            MatchPrefixParams(
                key=RadixKey(
                    token_ids=array("q", token_ids[:span]),
                    extra_key=getattr(req, "extra_key", None),
                ),
                cow_mamba=False,
                req=None,
            )
        )
        if _local_match_len(result) != int(depth):
            return False
        return not anchor_unusable(tree_cache, getattr(result, "best_match_node", None))
    except Exception:  # noqa: BLE001 - a vote may never break the reduce
        return False


def build_realize_payload(
    canonical: Sequence[str],
    skewed: Mapping[str, int],
    local_usable: Mapping[str, int],
    tree_cache: Any,
    by_rid: Mapping[str, Any],
    slots: int,
) -> List[int]:
    """1 = this rank can admit the group depth (or has no stake), 0 = it cannot."""
    payload = [1] * slots
    for i, rid in enumerate(list(canonical)[:slots]):
        if rid not in skewed:
            continue
        g = int(skewed[rid])
        if int(local_usable.get(rid, -1)) == g:
            continue  # this rank's own usable match IS the group depth
        req = by_rid.get(rid)
        payload[i] = 1 if (req is not None and can_realize(tree_cache, req, g)) else 0
    return payload


def apply_realize_verdict(
    group_usable: Dict[str, int],
    canonical: Sequence[str],
    skewed: Mapping[str, int],
    reduced: Sequence[int],
) -> Dict[str, int]:
    """Plant 0 for every skewed rid some rank cannot realize."""
    out = dict(group_usable)
    for rid, flag in zip(list(canonical), list(reduced)):
        if rid in skewed and int(flag) <= 0:
            out[rid] = 0
            _STATS["skew_zeroed"] = _STATS.get("skew_zeroed", 0) + 1
            n = _STATS["skew_zeroed"]
            if n <= 20 or n % 256 == 0:
                logger.warning(
                    "RU FLOOR SKEW-ZERO rid=%s group_min=%d (n=%d): some TP rank "
                    "cannot admit the group depth (no usable anchor there), so the "
                    "group re-prefills this rid from 0 on every rank (H97, rc9m "
                    "weg2-18-15 died on H96 CAP-MISS instead).",
                    str(rid)[:16],
                    int(skewed[rid]),
                    n,
                )
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
    return False


class RankFloorCapMiss(RuntimeError):
    """H96: this rank cannot materialize the group's usable match depth."""


def _local_match_len(result: Any) -> int:
    di = getattr(result, "device_indices", None)
    return (0 if di is None else len(di)) + int(
        getattr(result, "host_hit_length", 0) or 0
    )


def group_floor_cap(tree_cache: Any, req: Any, result: Any) -> Optional[int]:
    """Admission-site verdict: the depth THIS rank must cap its match to, or
    None. Non-None exactly when 0 < group usable match < local match (the
    ``above_group`` band): every rank then admits the group depth."""
    group = getattr(tree_cache, TREE_ATTR, None) if tree_cache is not None else None
    if not group or req is None:
        return None
    rid = str(getattr(req, "rid", "") or "")
    try:
        local = _local_match_len(result)
    except Exception:  # noqa: BLE001
        return None
    if floor_verdict(local, group.get(rid)) != "above_group":
        return None
    return int(group[rid])


def rematch_at_group_depth(tree_cache: Any, params: Any, cap: int, local: int) -> Any:
    """Re-run this rank's match on its key cut to ``cap`` (the group depth).

    Called from ``MambaComponent.finalize_match_result`` BEFORE the COW, so the
    outer match leaves no copy source and no slot behind; the nested match runs
    the full component chain (and its own COW) at the group depth, where the
    planted group value now reads ``agree``. ``params.key`` is already in the
    match's own units (``maybe_to_bigram_view`` flips it in place), so
    ``key[:cap]`` cuts at exactly ``cap`` match positions."""
    import dataclasses

    cut = dataclasses.replace(params, key=params.key[: int(cap)])
    capped = tree_cache.match_prefix(cut)
    got = _local_match_len(capped)
    rid = str(getattr(getattr(params, "req", None), "rid", "") or "")
    _STATS["above_group"] += 1
    n = _STATS["above_group"]
    if got != int(cap):
        raise RankFloorCapMiss(
            f"H96 RU FLOOR CAP-MISS rid={rid[:16]} local_match={local} "
            f"group_usable={cap} capped_match={got}: this rank has no usable "
            "recurrent anchor at the group depth, so it cannot admit what the "
            "other TP ranks admit -- stopping instead of resuming alone "
            "(raenge-nie-uneins; a split extend would hang the group)."
        )
    if n <= 20 or n % 256 == 0:
        logger.warning(
            "RU FLOOR CAP rid=%s local_match=%d group_usable=%d capped=%d "
            "(n=%d): this rank matched deeper than the group can use; it "
            "admits the group depth so every TP rank runs the same extend "
            "(H96, rc9l weg2-21-21 hung on the uncapped split).",
            rid[:16],
            local,
            cap,
            got,
            n,
        )
    return capped


def stats() -> Dict[str, int]:
    return dict(_STATS)
