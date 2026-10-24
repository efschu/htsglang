"""KVD-1024 (front KV class D): the rows a running decode is GUARANTEED to give back when it leaves D.

SCHEDULER-SOLL v2 R7 (a request under X that D does not take is a defect, never a flip reason) and R11 (the front
decides alone, D delivers numbers): an AGE PLAN that hands an older request to D on a displacement must count only
what the displacement really frees.

THE GAP (27B int4ar ``_1024_003732``, report front-kvoffset-1024 class D). Front and D priced a victim by its seat
length (``origin_input_ids + output_ids``): the front's ``_arrival_seat_age_verdict`` (``Seat.tokens``) and D's own
``d_park_runtime.kv_displace_would_fit`` (``_req_kv_tokens``). 01:02:43 the front planned
``victims=['pdflip-18-189'] deficit=38313`` (seat 54133); D's verdict ``need=104478 free=64346 -> 1 youngest must
leave`` (seat 54190 + reserve 401); D displaced 18-189 and its lifetime budget rose from 64355 to 100799 (+36444 =
36043 rows + 401 reserve) -- 18147 rows short of the seat, and the price 104477 stayed above the budget: 181
refusals in 18.7 s. The rows that stayed are the 18137-token system prompt every running agent request shares
(``cached_device=18137`` of the fresh 19-206/19-208 a few minutes later): its tree nodes are locked by the three other
running decodes, so the victim's release (``release_kv_cache(..., is_insert=True)`` -> ``dec_lock_ref``) leaves them
protected. ``pages_out=34816`` on the SEAT-AGE DISPLACE line is NOT a freed count: it is the ``#248 keep_role``
demotion window (the older one's shortfall against ``available``, clipped to the chain).

THE BOUND (``leaving_rows``). A retained release inserts the request's span into the tree and drops the request's lock
along its path (FULL component: ``lock_ref -= 1`` up to the root; a node turns evictable at 1 -> 0). So the rows that
are guaranteed to reach ``available`` or the DELIVERABLE evictable count are

    seat - rows of path nodes another holder keeps locked (FULL ``lock_ref >= 2``)
         - rows of path nodes that are already evictable or not on the device (``lock_ref == 0`` / value None:
           they are in the reading already, or not D's rows at all)
         - rows of the request's exclusive path nodes an aux (mamba) lock of someone else keeps unpeelable
           (``evict_frontier_census.blocked_tokens``: an aux-locked node and its FULL-unlocked ancestors)

The rows past the request's tree path (its decode tail, not inserted yet) are its own: inserted unlocked or freed to
the allocator on the release. This is upstream vLLM's rule (``BlockPool.free_blocks``: ``ref_cnt -= 1``, only a block
at 0 goes back to the free queue; ``get_num_common_prefix_blocks`` reads the shared blocks by ``ref_cnt``) on this
fork's radix tree.

JOINT RELEASE (review MUSS 2, :func:`joint_leaving_rows`): rows several victims hold ONLY between them (the system
prompt once no older request holds it) come free when all of them leave, never by one alone -- the sum of the single
bounds never counts them, and a head near the pool size would never get a plan. For the youngest-first prefix of k
victims a path node comes free when its FULL ``lock_ref`` equals the number of set members whose path contains it
(vLLM ``get_num_common_prefix_blocks``: ``ref_cnt == number of requests``); a foreign pin still counts as foreign. It
stays a lower bound and is >= the sum of the single bounds.

UNREADABLE TREE (a class without an int lock on the path, a desk double): the seat, as before -- the old behaviour,
named and counted. No tree path at all (chunk cache, nothing inserted): the seat is exact. A broken parent chain (or a
cycle): 0 for that request. Every fallback is counted (:data:`FALLBACKS`) and logged rate-limited
(``KVD-LEAVING-FALLBACK kind=error|unreadable|aux|broken``, review MUSS 3).

NOT COVERED (named): the engaged residency cap (``cap_above_published``: rows freed above the cap are taken straight
back by the pool) -- per-row slot ids are device tensors, reading them would sync; the backup wall is handled by the
callers (a victim is sized 0 under it). Never raises: any failure is the seat for an unreadable class and 0 for a
broken one, so a reading never breaks the info route or a scheduler pass."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

#: ``unified_cache_components.tree_component.ComponentType`` FULL / MAMBA (int enum, indexes ``component_data``)
_FULL = 0
_MAMBA = 2

#: review MUSS 3: every fallback by kind, cumulative per process (published in ``pdflip_kv.leaving_fallbacks``)
FALLBACKS: Dict[str, int] = {"error": 0, "unreadable": 0, "aux": 0, "broken": 0}
_FALLBACK_LOG_EVERY_S = 10.0
_fallback_last_log: Dict[str, float] = {}


def note_fallback(kind: str, detail: str = "") -> None:
    """Count one fallback of ``kind`` and log it at most once per 10 s per kind (with the running count)."""
    FALLBACKS[kind] = FALLBACKS.get(kind, 0) + 1
    now = time.monotonic()
    if now - _fallback_last_log.get(kind, -1e9) >= _FALLBACK_LOG_EVERY_S:
        _fallback_last_log[kind] = now
        logger.warning("KVD-LEAVING-FALLBACK kind=%s n=%d detail=%s (%s)", kind, FALLBACKS[kind], detail[:160],
                       {"error": "an exception: the request is credited 0 rows (no displacement on it)",
                        "unreadable": "a tree class without an int lock: the seat, as before KVD",
                        "aux": "the aux-lock walk failed: nothing counted as blocked (optimistic)",
                        "broken": "a broken or cyclic parent chain: 0 rows for that request"}.get(kind, "?"))


class _Unreadable(Exception):
    """The tree's lock is not an int on this node: a class this bound does not know."""


class _Broken(Exception):
    """The parent chain ends before the root or revisits a node."""


def _n(x) -> int:
    """len() of a list OR a tensor; never ``x or ()`` (a tensor has no truth value)."""
    if x is None:
        return 0
    numel = getattr(x, "numel", None)
    if callable(numel):
        try:
            return int(numel())
        except Exception:  # noqa: BLE001
            return 0
    try:
        return int(len(x))
    except TypeError:
        return 0


def seat_tokens(req) -> int:
    """The seat length both sides priced before (``origin_input_ids + output_ids``)."""
    return _n(getattr(req, "origin_input_ids", None)) + _n(getattr(req, "output_ids", None))


def _int(v) -> Optional[int]:
    if isinstance(v, bool) or not isinstance(v, int):
        return None
    return int(v)


def _full_cell(node):
    """``(rows, lock)`` of the node's FULL KV part: ``rows`` None when its value is not on the device; raises
    :class:`_Unreadable` when the lock is not an int."""
    cd = getattr(node, "component_data", None)
    if isinstance(cd, (list, tuple)) and len(cd) > _FULL:
        cell = cd[_FULL]
        value, lock = getattr(cell, "value", None), _int(getattr(cell, "lock_ref", None))
    else:
        value = getattr(node, "value", None)
        lock = _int(getattr(node, "full_lock_ref", None))
        if lock is None:
            lock = _int(getattr(node, "lock_ref", None))
    if lock is None:
        raise _Unreadable()
    return (None if value is None else _n(value)), lock


def _key_rows(node) -> int:
    return _n(getattr(node, "key", None))


def _aux_lock(node) -> int:
    cd = getattr(node, "component_data", None)
    if isinstance(cd, (list, tuple)) and len(cd) > _MAMBA:
        return _int(getattr(cd[_MAMBA], "lock_ref", 0)) or 0
    return _int(getattr(node, "mamba_lock_ref", 0)) or 0


def aux_blocked_ids(tree, own_last_nodes: Set[int]) -> Set[int]:
    """Ids of the nodes an aux (mamba) lock keeps unpeelable once their FULL lock is gone: every aux-locked node and
    all its ancestors (``evict_frontier_census.blocked_tokens``' walk, carried to the root: under the JOINT release a
    node several victims hold may come free too, so a FULL lock >= 2 no longer ends the walk). An aux lock that is a
    running request's OWN (the node is its ``last_node`` and the aux count is 1) leaves with it and blocks nothing.
    A failing walk: the empty set, counted (``kind=aux``)."""
    try:
        from flliper.srt.mem_cache.evict_frontier_census import AUX_LOCKED_ATTR

        aux = getattr(tree, AUX_LOCKED_ATTR, None)
        if not isinstance(aux, dict) or not aux:
            return set()
        root = getattr(tree, "root_node", None)
        seen: Set[int] = set()
        for node in list(aux.values()):
            if id(node) in own_last_nodes and _aux_lock(node) <= 1:
                continue
            cur = node
            while cur is not None and cur is not root and id(cur) not in seen:
                seen.add(id(cur))
                cur = getattr(cur, "parent", None)
        return seen
    except Exception as e:  # noqa: BLE001 -- named, counted
        note_fallback("aux", repr(e))
        return set()


def _path(tree, req, root):
    """``(cells, tail)``: the request's tree path from its ``last_node`` up to (not incl.) the root as
    ``[(id, rows_or_None, lock, key_rows)]``, and its rows past the path (``seat - path depth``). None = no tree path
    (every row its own). Raises :class:`_Unreadable` / :class:`_Broken`."""
    node = getattr(req, "last_node", None)
    if node is None or root is None or node is root:
        return None
    cells = []
    seen: Set[int] = set()
    depth = 0
    cur = node
    while cur is not root:
        if cur is None or id(cur) in seen:   # review SOLL 4: a visited set, no hop cap
            raise _Broken()
        seen.add(id(cur))
        rows, lock = _full_cell(cur)
        k = _key_rows(cur) if rows is None else rows
        cells.append((id(cur), rows, lock, k))
        depth += k
        cur = getattr(cur, "parent", None)
    return cells, max(0, seat_tokens(req) - depth)


def joint_leaving_rows(tree, reqs: Iterable[Any], blocked: Optional[Set[int]] = None) -> List[int]:
    """The rows the first k of ``reqs`` (in the given order, the displacement order) are GUARANTEED to give back when
    they ALL leave, for k = 1..n (cumulative, non-decreasing). A path node comes free when it is on the device, not
    behind a foreign aux lock and its FULL lock equals the number of set members whose path holds it. Fallbacks (counted):
    a tree class it cannot read -- the running sum of the seats (as before KVD); a broken chain -- that request adds 0;
    any other exception -- 0 for every request (no displacement on a number nobody can vouch for)."""
    reqs = list(reqs or ())
    if not reqs:
        return []
    try:
        root = getattr(tree, "root_node", None) if tree is not None else None
        paths: List[Optional[Tuple[list, int]]] = []
        broken: List[bool] = []
        for r in reqs:
            try:
                paths.append(_path(tree, r, root))
                broken.append(False)
            except _Broken:
                note_fallback("broken", "rid=%s" % getattr(r, "rid", "?"))
                paths.append(None)
                broken.append(True)
        if blocked is None:
            own = {id(getattr(r, "last_node", None)) for r in reqs if getattr(r, "last_node", None) is not None}
            blocked = aux_blocked_ids(tree, own) if any(p for p in paths) else set()
        count: Dict[int, int] = {}
        cell_of: Dict[int, Tuple[Optional[int], int]] = {}
        tails = 0
        out: List[int] = []
        for r, p, bad in zip(reqs, paths, broken):
            if bad:
                pass
            elif p is None:
                tails += seat_tokens(r)          # nothing of it in a tree: every row its own
            else:
                cells, tail = p
                tails += tail
                for nid, rows, lock, _k in cells:
                    count[nid] = count.get(nid, 0) + 1
                    cell_of[nid] = (rows, lock)
            freed = 0
            for nid, c in count.items():
                rows, lock = cell_of[nid]
                if rows is not None and lock == c and nid not in blocked:
                    freed += rows
            out.append(tails + freed)
        return out
    except _Unreadable:
        note_fallback("unreadable", type(tree).__name__)
        acc, out = 0, []
        for r in reqs:
            acc += seat_tokens(r)
            out.append(acc)
        return out
    except Exception as e:  # noqa: BLE001 -- named, counted: nothing guaranteed
        note_fallback("error", repr(e))
        return [0] * len(reqs)


def leaving_rows(tree, req, blocked: Optional[Set[int]] = None) -> int:
    """The rows ``req`` ALONE is guaranteed to give back when it leaves D with its span retained (see the module doc):
    :func:`joint_leaving_rows` of the one request."""
    return joint_leaving_rows(tree, [req], blocked)[0]


def leaving_rows_by_rid(tree, reqs: Iterable[Any]) -> Dict[str, int]:
    """:func:`leaving_rows` for each request alone, the aux walk done once."""
    reqs = list(reqs or ())
    own = {id(getattr(r, "last_node", None)) for r in reqs if getattr(r, "last_node", None) is not None}
    blocked = aux_blocked_ids(tree, own) if tree is not None and reqs else set()
    return {str(getattr(r, "rid", "")): leaving_rows(tree, r, blocked) for r in reqs}


def marginals(cumulative: List[int]) -> List[int]:
    """Cumulative joint release -> per-victim increments (their prefix sums are the joint release again)."""
    out, prev = [], 0
    for v in cumulative:
        out.append(max(0, int(v) - prev))
        prev = max(prev, int(v))
    return out
