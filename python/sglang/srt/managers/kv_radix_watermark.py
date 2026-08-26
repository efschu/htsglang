"""#662 -- lower the KV high-water mark by ID-TARGETED radix eviction.

WHY THIS EXISTS
---------------
``kv_backing_relief`` releases backing that NO row occupies: the slack
between the live high-water mark and the pool's reservation. Its own
module docstring names what is missing:

    Lowering the watermark FURTHER requires evicting cached prefix entries
    (data discarded, recomputable) [...]. Both lower ``max_live`` and then
    reuse exactly this code path; they are separate providers at higher
    cost, not changes to this one.

This module is that provider. It is the difference between a seam funded
by EMPTY VRAM and a seam funded by EVICTABLE CONTENT.

The distinction is the whole point. A pool sized so that the seam's
transient always fits in already-free memory must hold that memory free at
rest -- on this rig 3361/5070/4303 MiB, ~12.7 GiB, against a corridor law
that says ~1024 MiB per card. Every one of those bytes is VRAM the KV pool
could have been using to hold context. Funding the same seam by evicting
recomputable prefix cache at ARMING TIME costs nothing at rest, and costs
a cache miss only on the flips that the amortisation gate has already
decided are worth 4.8 s of seam.

WHY ID-TARGETED, AND NOT SIMPLY ``tree_cache.evict(n)``
------------------------------------------------------
The quantity that pins committed backing is ``max(live row id)``, not the
number of live rows. An LRU eviction frees whichever nodes are coldest,
and their row ids are uncorrelated with the high-water mark: freeing 90 %
of the cache can leave the mark exactly where it was, because one cold row
at a high id still pins every page beneath it. To lower the mark by K rows
you must free the rows ABOVE it, whichever nodes hold them.

So this evictor orders candidates by the HIGHEST ROW ID THEY HOLD and
evicts from the top down. That is the minimum cache loss for a given
watermark reduction -- the opposite of what an LRU pass would give up.

WHAT IT WILL NOT TOUCH
----------------------
Rows held by a RESIDENT REQUEST. Those are not recomputable-on-demand in
the sense that matters here: the request is in flight, its KV is its
state, and evicting it would abort work. ``resident_ceiling`` is therefore
a hard floor on how far this provider can lower the mark, and a target
below it is REFUSED rather than partially applied -- a half-applied
watermark is a pool whose backing does not cover its own live set, which
is a fault, not a degraded mode.

WHAT IT RETURNS IS MEASURED
---------------------------
``evict_rows_above`` returns rows actually freed, counted from the tree's
own eviction primitives. It deliberately does NOT report bytes: bytes are
what the arena returns to the driver, this returns pool rows, and the two
are only equal once ``runtime_set_backing_rows`` has run. The caller that
owns the byte accounting is ``kv_backing_relief``, which probes NVML
before and after. Keeping the units apart here is what stops this rung
claiming credit for bytes the driver never gave back -- the exact failure
three payloads in this chain have already shipped.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

LOG_PREFIX = "KV-WATERMARK"

#: Bound on the number of top-down passes. Evicting a leaf can expose its
#: parent as a new leaf, so lowering the mark past a deep chain needs more
#: than one pass -- but an unbounded loop inside a seam is a wedge, and a
#: mark that has not converged after this many passes is a finding.
MAX_PASSES = 64


def _node_rows(node: Any) -> Optional[Any]:
    """The row-id tensor a node holds, or None when it holds none."""
    value = getattr(node, "value", None)
    if value is None:
        return None
    try:
        if int(value.numel()) == 0:
            return None
    except Exception:
        # A list-backed stub (the hermetic tests) has no numel.
        if len(value) == 0:
            return None
    return value


def _max_row(node: Any) -> int:
    """The highest row id this node pins, or -1 when it pins none."""
    value = _node_rows(node)
    if value is None:
        return -1
    try:
        return int(value.max())
    except Exception:
        return int(max(value))


def _is_unlocked(node: Any) -> bool:
    """Evictable by the cache's own policy: nothing holds a reference.

    Both counters are consulted because the mamba tree locks the two
    payloads independently, and a node whose mamba state is locked cannot
    be passed to ``_evict_leaf_node`` -- it asserts on exactly that.
    """
    for attr in ("full_lock_ref", "mamba_lock_ref", "lock_ref"):
        ref = getattr(node, attr, 0)
        try:
            if int(ref) > 0:
                return False
        except Exception:
            return False
    return True


def _children(node: Any) -> List[Any]:
    kids = getattr(node, "children", None)
    if not kids:
        return []
    try:
        return list(kids.values())
    except AttributeError:
        return list(kids)


def _iter_nodes(tree_cache: Any) -> List[Any]:
    """Every node below the root, breadth-first. Root is never evictable."""
    root = getattr(tree_cache, "root_node", None)
    if root is None:
        return []
    out: List[Any] = []
    frontier = _children(root)
    seen = 0
    while frontier:
        seen += 1
        if seen > 1_000_000:  # pragma: no cover - a cycle guard, never a gate
            logger.warning("%s node walk exceeded its bound; stopping", LOG_PREFIX)
            break
        node = frontier.pop()
        out.append(node)
        frontier.extend(_children(node))
    return out


#: Cache-class/node-class pairs already reported by
#: ``_warn_if_node_rows_unreadable``. Once per pair per process: this condition
#: is permanent for a given tree type, and a line per pricing call would bury
#: the log it exists to inform.
_UNREADABLE_WARNED: set = set()


def _warn_if_node_rows_unreadable(tree_cache: Any, nodes: List[Any]) -> None:
    """#872 SIBLING: the tree has nodes, and not one of them yields rows.

    ``_node_rows`` reads ``node.value`` duck-typed, and its miss returns
    ``None`` -- the same answer a genuinely empty node gives. So a tree whose
    node type simply does not carry ``.value`` prices out at "0 rows
    evictable", which is INDISTINGUISHABLE from a tree that has nothing to
    give up, and the consumer in ``kv_backing_relief`` narrates that zero as
    "healthy if those rows are genuinely live". That is the #872 shape exactly:
    a duck-typed probe whose miss wears the healthy answer's clothes.

    It is live, not hypothetical. ``UnifiedTreeNode``
    (``unified_radix_cache.py``) has no ``.value`` at all -- its payload sits
    at ``component_data[BASE_COMPONENT_TYPE].value`` -- so on the
    ``UnifiedRadixCache`` this build actually binds, this whole #662 rung has
    always freed exactly zero rows, silently, on every call.

    THIS FUNCTION ONLY REPORTS. Teaching ``_node_rows`` to read
    ``component_data`` is NOT enough and is deliberately not done here: the
    eviction primitives this module then reaches for (``_evict_leaf_node``,
    ``_delete_leaf``) are equally absent from ``UnifiedRadixCache``, whose
    eviction is the multi-component ``evict(EvictParams)`` /
    ``_evict_component_and_detach_lru`` / ``_remove_leaf_from_parent`` shape.
    Half-fixing the read would turn a silent no-op into a walk that frees rows
    and never unlinks the node -- a dangling tree node over freed KV rows,
    which is the #718 silent-corruption family and strictly worse than doing
    nothing. The rung stays a no-op until someone builds the
    ``UnifiedRadixCache``-shaped actuator; it just stops being a SILENT one.
    """
    if not nodes:
        return
    # The already-reported check comes FIRST because it is O(1) and the scan
    # below is O(nodes). This runs on the gate's unconditional pricing path,
    # and re-walking every node on every call to re-derive a permanent,
    # already-reported property is a cost the diagnosis does not need to keep
    # paying.
    key = f"{type(tree_cache).__name__}/{type(nodes[0]).__name__}"
    if key in _UNREADABLE_WARNED:
        return
    if any(_node_rows(node) is not None for node in nodes):
        return
    _UNREADABLE_WARNED.add(key)
    logger.error(
        "%s #872 UNREADABLE NODES: %s holds %d node(s) of type %s and NONE of "
        "them yields a row tensor via `.value`, so this rung prices every "
        "tree as 'nothing evictable' and frees nothing, always. That zero is "
        "a PROBE MISS, not a measurement, and the caller cannot tell the "
        "difference -- do not read it as 'the tree is genuinely live'. "
        "%s's rows are reached differently (see UnifiedTreeNode.component_data) "
        "and its eviction primitive is not the `_delete_leaf` pair this module "
        "assumes, so the rung needs a cache-shaped actuator, not a wider "
        "getattr.",
        LOG_PREFIX,
        type(tree_cache).__name__,
        len(nodes),
        type(nodes[0]).__name__,
        type(nodes[0]).__name__,
    )


def _leaves_above(tree_cache: Any, target_row: int) -> List[Any]:
    """Unlocked LEAF nodes pinning at least one row above ``target_row``.

    Leaves only: the tree's eviction primitive asserts leafness, and a
    node with children still has its rows reachable through them.
    """
    nodes = _iter_nodes(tree_cache)
    _warn_if_node_rows_unreadable(tree_cache, nodes)
    out = []
    for node in nodes:
        if _children(node):
            continue
        if not _is_unlocked(node):
            continue
        if _max_row(node) > target_row:
            out.append(node)
    return out


def tree_ceiling(tree_cache: Any) -> int:
    """The highest row id the radix tree pins, or -1 for an empty tree."""
    nodes = _iter_nodes(tree_cache)
    _warn_if_node_rows_unreadable(tree_cache, nodes)
    ceiling = -1
    for node in nodes:
        ceiling = max(ceiling, _max_row(node))
    return ceiling


def evictable_rows_above(tree_cache: Any, target_row: int) -> Tuple[int, int]:
    """MEASURE the payload: ``(rows, nodes)`` this rung could give up.

    Pure. Called on the gate's unconditional path to price the rung before
    anything is evicted, because this chain has repeatedly built relief for
    payloads that turned out to be empty (see the ``_describe_live_split``
    diagnostic that motivated this module).

    Counts every row of every qualifying node, not only the rows above the
    mark: evicting a node frees all of its rows, and over-reporting the
    cost of the move is the safe direction.
    """
    rows = 0
    nodes = 0
    for node in _leaves_above(tree_cache, target_row):
        value = _node_rows(node)
        if value is None:
            continue
        try:
            rows += int(value.numel())
        except Exception:
            rows += len(value)
        nodes += 1
    return rows, nodes


def _evict_one(tree_cache: Any, node: Any) -> int:
    """Evict ONE leaf through the tree's own primitive. Returns rows freed.

    The tree's primitive is used rather than a private free/delete pair so
    that every book the cache keeps -- lru lists, evictable/protected
    sizes, tombstones, host-tier writeback, the allocator's free list --
    stays consistent. A watermark actuator that maintained its own
    bookkeeping would be a second source of truth about which rows are
    live, and that is the silent-wrong-context class.
    """
    primitive = getattr(tree_cache, "_evict_leaf_node", None)
    if callable(primitive):
        freed, _mamba, _x, _x_next = primitive(node, False)
        return int(freed)

    # The plain radix tree has no combined primitive: free then unlink.
    value = _node_rows(node)
    freed = 0
    if value is not None:
        try:
            freed = int(value.numel())
        except Exception:
            freed = len(value)
        tree_cache.token_to_kv_pool_allocator.free(value)
    delete = getattr(tree_cache, "_delete_leaf", None)
    if callable(delete):
        delete(node)
    return freed


def evict_rows_above(
    tree_cache: Any,
    target_row: int,
    *,
    resident_ceiling: int = -1,
) -> int:
    """ACT: free every evictable row above ``target_row``. Returns rows freed.

    Highest row id first, so the mark comes down for the least cache given
    up. Repeated in passes because evicting a leaf can expose its parent.

    REFUSES OUTRIGHT when a resident request pins a row above the target
    (``resident_ceiling > target_row``). Partially applying such a target
    would leave rows live above the point the caller is about to unmap.
    """
    if tree_cache is None:
        return 0
    if resident_ceiling > int(target_row):
        logger.warning(
            "%s REFUSED target row %d: a resident request pins row %d, which "
            "this rung may not evict. Nothing was evicted.",
            LOG_PREFIX,
            int(target_row),
            int(resident_ceiling),
        )
        return 0

    freed_total = 0
    for _ in range(MAX_PASSES):
        candidates = _leaves_above(tree_cache, int(target_row))
        if not candidates:
            break
        # The ordering IS the mechanism: highest row id first.
        candidates.sort(key=_max_row, reverse=True)
        progressed = False
        for node in candidates:
            # Re-check: an earlier eviction in this same pass may have
            # unlinked this node or turned it into a non-leaf.
            if _children(node) or not _is_unlocked(node):
                continue
            if _max_row(node) <= int(target_row):
                continue
            try:
                freed = _evict_one(tree_cache, node)
            except Exception as e:  # pragma: no cover - defensive
                logger.warning(
                    "%s could not evict a node above row %d: %s",
                    LOG_PREFIX,
                    int(target_row),
                    e,
                )
                continue
            if freed > 0:
                freed_total += freed
                progressed = True
        if not progressed:
            break
    else:  # pragma: no cover - the bound is a finding, not a gate
        logger.warning(
            "%s watermark did not converge in %d passes; ceiling is still %d "
            "against target %d",
            LOG_PREFIX,
            MAX_PASSES,
            tree_ceiling(tree_cache),
            int(target_row),
        )
    return freed_total
