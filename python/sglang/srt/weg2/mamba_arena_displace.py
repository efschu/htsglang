"""fnFL2 H19 (24.09.2026): the mamba host arena keeps the DEEPEST anchors of a
request, not the first ones -- and the END ANCHOR always gets a slot.

Boot fnFL2x130 (chunk 512, 97841 tokens = 191 chunk nodes, 32 arena slots):
P publishes one mamba anchor per chunk node (`_weg2_publish_at_chunk`). Every
landed write takes a reader reference per P rank (`complete_write`) that is
held for as long as the node keeps its host value, and nothing releases it
during the prefill -- the host LRU never runs, because arena rows are outside
the host pool's size. `arena_claim` then answers 4 (no free slot) from the
31st anchor on, `arena_evict_candidates` skips every slot (refcount != 0),
`_weg2_direct_claim` refuses the node whole (KV claim aborted too) and the
sweep stops (`stopped: mamba_full`, xsn342). First-come: the arena held pages
0..239 (15360 tokens), the end anchor at 97792 was refused ten times at the
retain, D capped the claim at the deepest anchor it found (`#1028B FETCH CAP
... caps={mamba: 240}`), W50 requeue, second prefill.

The hand-over (FETCH CAP = the deepest anchor in range) needs ONE anchor per
request: the deepest. Here, per request (the rid that created the node):

* SHARE -- a request holds at most `cap` arena anchors on this rank. Before
  its next (deeper) anchor is claimed, its SHALLOWEST settled anchor is
  released (this rank's reference) and dropped once no rank references it
  (`arena_drop_unreferenced`, no disk write). The trigger is a count along
  the request's own chain -- the same on every PP rank -- so all ranks
  release the same anchor at the same logical step; a purely "arena full"
  trigger would not converge, because a slot frees only after the LAST of the
  three P ranks released it, and the ranks reach a full arena at different
  times.
* FULL -- when the arena still refuses (other requests' anchors, a rank that
  lags), the node releases ONE more anchor of its own request, once per node
  and rank. A request with no anchor of its own to give (its first anchor, or
  its end anchor behind a full arena) may take the shallowest INTERMEDIATE
  anchor of another request -- never an end anchor, never a request's deepest
  anchor, never an untagged node.

Anchors outside the request's own chain are never touched by SHARE; the
end-anchor node (`_weg2_end_anchor`, #1481) is never a victim.
SGLANG_WEG2_MAMBA_ARENA_RID_ANCHORS: -1 (default) = max(2, arena_slots // 4)
(32 slots -> 8: chunk 16384 makes 6-7 anchors per 98k request and is
untouched), N >= 2 = N, 0 = off (first-come as before).
"""
from __future__ import annotations

from typing import Callable, Iterable, Optional

import msgspec

AUTO = -1
_STATS_KEEP = 64


def rid_anchor_cap(*, configured: int, arena_slots: int) -> int:
    """Arena anchors one request may hold on this rank; 0 = displacement off."""
    if configured == 0:
        return 0
    if configured < 0:
        return max(2, int(arena_slots) // 4)
    return max(2, int(configured))


class OwnedAnchor(msgspec.Struct, frozen=True):
    """One arena anchor on a request's chain. `slots` is None while the anchor
    may not be released (end anchor, write in flight, host lock)."""

    node: object
    depth: int
    rid: str
    slots: Optional[tuple] = None


class RidAnchorStats(msgspec.Struct):
    written: int = 0
    displaced_share: int = 0
    displaced_full: int = 0
    dropped: int = 0
    refused: int = 0
    held: int = 0
    deepest: int = 0
    end_anchor: str = "none"   # none | slot | refused (claim) | missing (at the retain)

    def line(self, *, rid: str, cap: int, slots: int) -> str:
        return (
            f"WEG2 MAMBA-ARENA rid={rid[:12]} written={self.written} "
            f"displaced={self.displaced_share + self.displaced_full}"
            f"(share={self.displaced_share},full={self.displaced_full}) dropped={self.dropped} "
            f"refused={self.refused} held={self.held} deepest={self.deepest} "
            f"end_anchor={self.end_anchor} cap={cap} slots={slots}"
        )


class RidAnchorLedger:
    """Per-request counters of this rank, bounded to the last 64 requests."""

    def __init__(self):
        self._stats: dict = {}

    def of(self, rid: str) -> RidAnchorStats:
        st = self._stats.get(rid)
        if st is None:
            st = self._stats[rid] = RidAnchorStats()
            while len(self._stats) > _STATS_KEEP:
                self._stats.pop(next(iter(self._stats)))
        return st

    def peek(self, rid: str) -> Optional[RidAnchorStats]:
        return self._stats.get(rid)


def ancestor_path(*, target, root) -> tuple[list, int]:
    """The target's ancestors, ROOT FIRST, root excluded, with token depths:
    [(node, depth_at_node_end), ...] and the target's own end depth."""
    path = []
    node = target.parent
    while node is not None and node is not root:
        path.append(node)
        node = node.parent
    path.reverse()
    out, depth = [], 0
    for n in path:
        depth += len(n.key)
        out.append((n, depth))
    return out, depth + len(target.key)


def owned_anchors(
    *, path, rid: str, anchor_of: Callable[[object], tuple]
) -> list:
    """The request's own arena anchors on `path`, SHALLOWEST first.
    `anchor_of(node) -> (held, slots)`: held = an arena anchor is there,
    slots = its slots when releasable, else None."""
    out = []
    for n, depth in path:
        if n.weg2_anchor_rid != rid:
            continue
        held, slots = anchor_of(n)
        if held:
            out.append(OwnedAnchor(node=n, depth=depth, rid=rid,
                                   slots=None if slots is None else tuple(slots)))
    return out


def pick_own_victim(owned: Iterable[OwnedAnchor]) -> Optional[OwnedAnchor]:
    """The shallowest releasable anchor of the request itself."""
    for a in owned:
        if a.slots is not None:
            return a
    return None


def pick_foreign_victim(candidates: Iterable[OwnedAnchor], *, rid: str) -> Optional[OwnedAnchor]:
    """The shallowest releasable INTERMEDIATE anchor of another request: each
    request keeps its deepest anchor (its hand-over), end anchors carry
    slots=None already."""
    by_rid: dict = {}
    for a in candidates:
        if a.rid != rid:
            by_rid.setdefault(a.rid, []).append(a)
    pool = []
    for anchors in by_rid.values():
        deepest = max(anchors, key=lambda a: a.depth)
        pool.extend(a for a in anchors if a is not deepest and a.slots is not None)
    if not pool:
        return None
    return min(pool, key=lambda a: (a.depth, a.rid))


def tree_anchors(*, root, anchor_of: Callable[[object], tuple]) -> list:
    """Every tagged arena anchor of the tree (BFS, token depths) -- only on
    the FULL path of a request without an anchor of its own to give."""
    out = []
    queue = [(c, len(c.key)) for c in root.children.values()]
    while queue:
        n, depth = queue.pop()
        queue.extend((c, depth + len(c.key)) for c in n.children.values())
        if n.weg2_anchor_rid is None:
            continue
        held, slots = anchor_of(n)
        if held:
            out.append(OwnedAnchor(node=n, depth=depth, rid=n.weg2_anchor_rid,
                                   slots=None if slots is None else tuple(slots)))
    return out
