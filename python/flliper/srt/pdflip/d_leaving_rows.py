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

UNREADABLE TREE (a class without an int lock on the path, a desk double): the seat, as before -- the old behaviour,
named. No tree path at all (chunk cache, nothing inserted): the seat is exact. A broken parent chain: 0.

NOT COVERED (named): the engaged residency cap (``cap_above_published``: rows freed above the cap are taken straight
back by the pool) -- per-row slot ids are device tensors, reading them would sync; the backup wall is handled by the
callers (a victim is sized 0 under it). Never raises: any failure is the seat for an unreadable class and 0 for a
broken one, so a reading never breaks the info route or a scheduler pass."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Set

#: ``unified_cache_components.tree_component.ComponentType`` FULL / MAMBA (int enum, indexes ``component_data``)
_FULL = 0
_MAMBA = 2
#: a parent walk longer than this is a broken chain (a 1M-token path in 16-token nodes is 65536 hops)
MAX_HOPS = 1 << 20


class _Unreadable(Exception):
    """The tree's lock is not an int on this node: a class this bound does not know."""


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
    """``(rows, lock)`` of the node's FULL KV part: ``rows`` None when its value is not on the device (then the
    caller counts the key length); raises :class:`_Unreadable` when the lock is not an int."""
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
    its ancestors up to the first node another holder keeps FULL-locked (``>= 2``; above it nothing is anybody's
    exclusive row). An aux lock that is a running request's OWN (the node is its ``last_node`` and the aux count is
    1) leaves with it and blocks nothing. Same walk as ``evict_frontier_census.blocked_tokens``, one level deeper
    (a FULL lock of 1 is the leaving request's)."""
    try:
        from flliper.srt.mem_cache.evict_frontier_census import AUX_LOCKED_ATTR
    except Exception:  # noqa: BLE001 -- no census module: no aux set to read
        return set()
    aux = getattr(tree, AUX_LOCKED_ATTR, None)
    if not isinstance(aux, dict) or not aux:
        return set()
    root = getattr(tree, "root_node", None)
    seen: Set[int] = set()
    for node in list(aux.values()):
        if id(node) in own_last_nodes and _aux_lock(node) <= 1:
            continue
        cur, hops = node, 0
        while cur is not None and cur is not root and id(cur) not in seen and hops < MAX_HOPS:
            try:
                _rows, lock = _full_cell(cur)
            except _Unreadable:
                break
            if lock >= 2:
                break
            seen.add(id(cur))
            cur, hops = getattr(cur, "parent", None), hops + 1
    return seen


def leaving_rows(tree, req, blocked: Optional[Set[int]] = None) -> int:
    """The rows ``req`` is guaranteed to give back when it leaves D with its span retained (see the module doc).
    ``blocked``: :func:`aux_blocked_ids` when the caller sizes several requests (computed here otherwise)."""
    seat = seat_tokens(req)
    try:
        node = getattr(req, "last_node", None)
        root = getattr(tree, "root_node", None) if tree is not None else None
        if node is None or root is None or node is root:
            return seat                       # nothing of it in a tree: every row is its own
        held = 0
        excl: Dict[int, int] = {}
        cur, hops = node, 0
        while cur is not root:
            if cur is None or hops >= MAX_HOPS:
                return 0                      # broken parent chain: nothing is guaranteed
            rows, lock = _full_cell(cur)
            if rows is None:
                held += _key_rows(cur)        # not on the device: not D's rows to give
            elif lock == 1:
                excl[id(cur)] = rows          # only this request holds it: it turns evictable
            else:
                held += rows                  # another holder keeps it (>= 2) / evictable already (0)
            cur, hops = getattr(cur, "parent", None), hops + 1
        if excl:
            if blocked is None:
                blocked = aux_blocked_ids(tree, {id(node)})
            held += sum(v for k, v in excl.items() if k in blocked)
        return max(0, seat - held)
    except _Unreadable:
        return seat                           # a tree class this bound does not read: as before (old D)
    except Exception:  # noqa: BLE001 -- an instrument never breaks a pass: nothing guaranteed
        return 0


def leaving_rows_by_rid(tree, reqs: Iterable[Any]) -> Dict[str, int]:
    """:func:`leaving_rows` for each request, the aux walk done once."""
    reqs = list(reqs or ())
    own = {id(getattr(r, "last_node", None)) for r in reqs if getattr(r, "last_node", None) is not None}
    try:
        blocked = aux_blocked_ids(tree, own) if tree is not None else set()
    except Exception:  # noqa: BLE001
        blocked = set()
    return {str(getattr(r, "rid", "")): leaving_rows(tree, r, blocked) for r in reqs}
