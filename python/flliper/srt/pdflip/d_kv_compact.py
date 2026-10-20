"""D-MEM-SCHED KV-COMPACT: the tree's KV pages above a stage boundary move DOWN,
so a decided shrink is not held by a cached prefix (S77-A, 19.10.).

THE METAL (NF rc17 7d93470b5e, boot 1019_171636, D.log from line 32274,
17:28:28Z): with D decoding for 15 min without a flip, every end event read
``end event: live pages up to 405056 hold S9 (wanted S8)`` (up to 418176; rc16:
263 times) while ``used`` (278k-383k) fit S8 = 393216. The pages above the
boundary were not the running requests' -- ``used`` counts those -- but the
finished turns' prefixes the tree RETAINS (``#1469 RETAIN``, ``#59 RESUMABLE``,
86-98k tokens each) and the next turn of the same session locks again. LRU
eviction never reaches them (they are the hottest), so S9 held, and on every
3080 rank 6-10 expert seat rows stayed off (rows_on bs2 rc16 ~21 / rc17 ~9-12,
misses per forward x2.2-2.4 on TP1/TP2, S9 vs S8: bs4 +14 %, bs3 +10 % gpu-ms).
User law (memory freier-kv-platz-gehoert-experten-0929): free VRAM belongs to
the experts; D-MEM-SCHED's need is ``used + incoming + air`` -- a cached prefix
is not need, and ``runtime_tick``'s own docstring shrinks "only below pages no
request and no unbacked tree node holds". ``max_live_page`` counted EVERY
non-free page, a backed idle tree page included: that is the gap.

THE PRODUCT RULE (NF-Operator 30.09., ``d_seat_compact``, the same move for the
GDN states): "KOMPAKTIEREN, nicht verdrängen. Die Zustände des Baums, die über
dem Limit liegen, ziehen per Device-Kopie in freie Slots darunter um; die
Referenzen im Baum werden dabei umgehängt. Verdrängen nur als Rückfall: wenn
unten kein Platz frei ist und der Zustand schon im L2 gesichert ist (backuped).
Nie einen ungesicherten Zustand verlieren."

THE MOVE. A tree node whose FULL device value has pages above the boundary is
moved when nothing else reads or writes those pages: no lock (a running request,
a prefetch/load-back pin), no write-through in flight (the D2H copy reads the
source), no load-back in flight. Its pages above the boundary are copied page by
page into free pages below it -- in every KV tensor the stage form trims on this
rank (``d_seat_vram.KvPageCopier``: FA K/V or their owner share, QSA compressed
keys, draft K/V) -- then the node's value is re-pointed (same tokens, same
offsets in the page) and the sources go back to the allocator. Un-backed nodes
take the free pages first, then the backed ones, newest pages first (page ids:
replicated, never the rank-local access clock); a backed node left over is EVICTED from the device (its host copy stays, a later
hit reads it back -- never a recompute), leaf first, never one with a device
child that stays. A page above the boundary that no tree node holds (a running
request's own page, a session slot) or a pinned node stops the plan by name:
no partial move, the next end event looks again.

Rank-local execution, group verdict and group check (``d_seat_vram.
_kv_compact_for_shrink``): the plan is surveyed on every rank, the group MIN of
"my plan clears boundary j" picks the lowest stage every rank reaches; the plan
DIGEST must be equal on every rank, the MOVES go first and their outcome must be
equal on every rank, only then the fallback evictions -- any difference after a
mutation is a named crash-stop (review N3). The pages move in EVERY token-indexed
buffer of every KV pool of the rank (``d_seat_vram.kv_page_buffers``), stage-born
or not (review N1: TP0 on NF holds the QSA index and the draft for every token
with ``born=0``).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

MARKER = "PDFLIP D-MEM-SCHED KV-COMPACT"


class CompactRefused(RuntimeError):
    """The move could not be carried out on this rank (named)."""


@dataclass
class NodePages:
    """One tree node that holds device KV pages."""

    node: Any
    pages: List[int]
    #: something reads or writes its pages (lock, write-through, load-back)
    pinned: bool
    #: its host (L2) copy has landed for every component it holds on the device
    backed: bool
    #: the rank-local access clock -- logged only, never a decision (review N4)
    last: float = 0.0


@dataclass
class Survey:
    page_size: int
    num_pages: int
    #: the lowest boundary page surveyed: only pages above it are listed
    floor_page: int
    #: live pages (not free, not withheld) above ``floor_page``, ascending
    live_above: List[int]
    #: free pages (in a free list, not withheld), ascending
    free: List[int]
    #: page above ``floor_page`` -> the tree node holding it
    owner: Dict[int, NodePages]
    nodes: List[NodePages]
    why: Optional[str] = None
    #: S84: pages above ``floor_page`` a cap filter OTHER than the stage cap withholds
    #: (the backing-relief rung's KvRowCap), ascending. Not live, not free -- but
    #: ``max_live_page`` (the floor) counts them, and must: that filter's release()
    #: hands them back without the stage cap's hook. A plan that leaves one above its
    #: boundary would move pages for a floor that cannot drop.
    held_above: List[int] = field(default_factory=list)


@dataclass
class Plan:
    boundary_page: int
    moves: List[Tuple[NodePages, List[int], List[int]]] = field(default_factory=list)
    evicts: List[NodePages] = field(default_factory=list)

    @property
    def moved_pages(self) -> int:
        return sum(len(s) for _n, s, _d in self.moves)


@dataclass
class Result:
    moved_nodes: int
    moved_pages: int
    evicted_nodes: int
    evicted_tokens: int
    bytes: int
    ms: float


def _full_ct():
    from flliper.srt.mem_cache.unified_cache_components.tree_component import ComponentType

    return ComponentType


def tree_why(cache: Any) -> Optional[str]:
    """None when ``cache`` is a tree this module can move pages in (the unified
    hierarchical tree, FULL component, no SWA index translation); else why not."""
    if cache is None:
        return "no_tree"
    comps = getattr(cache, "components", None)
    if not isinstance(comps, dict) or not hasattr(cache, "_collect_all_nodes"):
        return "tree_kind_unknown (not the unified tree)"
    CT = _full_ct()
    if CT.FULL not in comps:
        return "no_full_component"
    if CT.SWA in comps:
        return "swa (SWA pages are translated from FULL ids)"
    return None


def _id_tensor(t):
    import torch

    if t is None or not getattr(t, "numel", lambda: 0)():
        return torch.empty(0, dtype=torch.int64)
    return t.detach().reshape(-1).to("cpu", torch.int64)


def cap_filters(allocator: Any) -> List[Any]:
    """S84: every id-set CAP FILTER on this allocator -- the KV stage cap
    (``_pdflip_kv_stage_cap``) and every free listener that declares itself one
    (``allocator.base.mark_listener``, role ``cap-filter``, owner = the KvRowCap).
    Their withheld ids are neither live nor free, and no page move may take a
    destination above their cap: the cap is what keeps an id off unmapped memory
    (a freed source above it is withheld again by the filter's own free hook)."""
    from flliper.srt.mem_cache.allocator.base import (
        LISTENER_OWNER_ATTR,
        LISTENER_ROLE_CAP_FILTER,
        listener_role,
    )

    out: List[Any] = []
    stage = getattr(allocator, "_pdflip_kv_stage_cap", None)
    if stage is not None:
        out.append(stage)
    for on_free, _on_clear in list(getattr(allocator, "_free_listeners", None) or ()):
        if listener_role(on_free) != LISTENER_ROLE_CAP_FILTER:
            continue
        owner = getattr(on_free, LISTENER_OWNER_ATTR, None)
        if owner is not None and all(owner is not o for o in out):
            out.append(owner)
    return out


def survey(cache: Any, allocator: Any, page_size: int, floor_page: int = 0) -> Survey:
    """The allocator's live pages above ``floor_page`` and the tree node holding
    each (ONE D2H copy of all node values, tensor ops; Python lists only for the
    pages above ``floor_page``). Pages in no free list and not withheld by the stage
    cap are live; a page two nodes name is pinned in both. S84: every cap filter
    (``cap_filters``) counts, and no free page above the lowest engaged cap is offered."""
    import torch

    P = max(1, int(page_size))
    n = int(getattr(allocator, "num_pages", 0) or 0)
    B = max(0, int(floor_page))
    free_m = torch.zeros(n + 2, dtype=torch.bool)
    for name in ("free_pages", "release_pages"):
        ids = _id_tensor(getattr(allocator, name, None))
        ids = ids[(ids >= 0) & (ids <= n)]
        free_m[ids] = True
    staged = torch.zeros(n + 2, dtype=torch.bool)
    for t in (getattr(allocator, "free_group", None) or []):  # staged frees: neither live nor claimable
        ids = _id_tensor(t) // P
        staged[ids[(ids >= 0) & (ids <= n)]] = True
    held_m = torch.zeros(n + 2, dtype=torch.bool)
    other_m = torch.zeros(n + 2, dtype=torch.bool)
    top = n                     # S84: no destination above the lowest engaged cap
    stage_cap = getattr(allocator, "_pdflip_kv_stage_cap", None)
    for cap in cap_filters(allocator):
        held = _id_tensor(getattr(cap, "_withheld", None))
        held = held[(held >= 0) & (held <= n)]
        held_m[held] = True
        if cap is not stage_cap:
            other_m[held] = True
        c = getattr(cap, "cap", None)
        if c is not None:
            top = min(top, int(c))
    idx = torch.arange(n + 2)
    live_m = (~free_m) & (~held_m) & (~staged) & (idx >= 1) & (idx <= n)
    live_above = [int(x) for x in torch.nonzero(live_m & (idx > B)).flatten().tolist()]
    free = [int(x) for x in torch.nonzero(free_m & (~held_m) & (idx >= 1) & (idx <= top)).flatten().tolist()]
    held_above = [int(x) for x in torch.nonzero(other_m & (idx > B)).flatten().tolist()]
    why = tree_why(cache)
    if why is not None:
        return Survey(P, n, B, live_above, free, {}, [], why=why, held_above=held_above)
    CT = _full_ct()
    full, mamba = int(CT.FULL), int(CT.MAMBA)
    root = getattr(cache, "root_node", None)
    rows = []
    for node in cache._collect_all_nodes():
        if node is root:
            continue
        v = node.component_data[full].value
        if v is not None and v.numel():
            rows.append((node, v.reshape(-1)))
    owner: Dict[int, NodePages] = {}
    nodes: List[NodePages] = []
    if not rows:
        return Survey(P, n, B, live_above, free, owner, nodes, held_above=held_above)
    try:
        ids = torch.cat([v for _n, v in rows]).to("cpu", torch.int64)
    except RuntimeError:  # mixed devices: one copy each
        ids = torch.cat([v.to("cpu", torch.int64) for _n, v in rows])
    counts = torch.tensor([int(v.numel()) for _n, v in rows], dtype=torch.int64)
    nidx = torch.repeat_interleave(torch.arange(len(rows), dtype=torch.int64), counts)
    pages = ids // P
    sel = pages > B
    pairs = torch.unique(pages[sel] * len(rows) + nidx[sel])
    pp, nn = (pairs // len(rows)).tolist(), (pairs % len(rows)).tolist()
    wt = getattr(cache, "ongoing_write_through", None) or {}
    lb = getattr(cache, "ongoing_load_back", None) or {}
    by_row: Dict[int, NodePages] = {}
    for p, r in zip(pp, nn):
        np_ = by_row.get(r)
        if np_ is None:
            node = rows[r][0]
            nid = getattr(node, "id", None)
            in_wt = nid in wt or getattr(node, "write_through_pending_id", None) is not None
            cds = node.component_data
            pinned = (in_wt or nid in lb
                      or any(int(getattr(cd, "lock_ref", 0) or 0) > 0 for cd in cds))
            mamba_ok = (len(cds) <= mamba or cds[mamba].value is None
                        or cds[mamba].host_value is not None
                        or bool(getattr(node, "pdflip_anchor_secured", False)))
            np_ = NodePages(node=node, pages=[], pinned=bool(pinned),
                            backed=bool(cds[full].host_value is not None and not in_wt and mamba_ok),
                            last=float(getattr(node, "last_access_time", 0) or 0))
            by_row[r] = np_
            nodes.append(np_)
        np_.pages.append(int(p))
        prev = owner.get(int(p))
        if prev is not None and prev is not np_:
            prev.pinned = np_.pinned = True  # a page two nodes name moves for neither
        owner[int(p)] = np_
    return Survey(P, n, B, live_above, free, owner, nodes, held_above=held_above)


def _device_children(node: Any) -> List[Any]:
    full = int(_full_ct().FULL)
    return [c for c in getattr(node, "children", {}).values()
            if c.component_data[full].value is not None]


#: S98 page codes of the group vote (``page_codes`` / ``agreed_survey``). Above the
#: surveyed floor the MIN picks the most blocking answer of any rank: a cap filter
#: other than the stage cap withholds it < live and blocking (outside the tree, pinned,
#: two owners) < a live tree page NOT backed < a live tree page backed < not live and
#: not claimable (withheld by the stage cap, a staged free) < free (a destination).
CODE_HELD = -1
CODE_BLOCK = 0
CODE_UNBACKED = 1
CODE_BACKED = 2
CODE_IDLE = 3
CODE_FREE = 4


def page_codes(s: Survey, top_page: int) -> Tuple[List[int], List[int], Optional[str]]:
    """S98: this rank's vote on its survey -- (dest, above, why).

    ``dest[p - 1]`` for pages 1..floor_page: 1 = free here (a destination), else 0.
    ``above[i]`` for pages floor_page+1..top_page: one ``CODE_*`` each (a free page
    there is a destination for a higher candidate boundary). ``why`` names
    a rank-local reason the survey cannot be voted on (then the caller votes "no").
    The vector length is a function of (floor_page, top_page) alone -- both
    replicated -- so every rank enters the collective with the same length."""
    B, T = int(s.floor_page), int(top_page)
    free = set(int(p) for p in s.free)
    dest = [1 if p in free else 0 for p in range(1, B + 1)]
    above = [CODE_FREE if p in free else CODE_IDLE for p in range(B + 1, T + 1)]
    if s.why is not None:
        return dest, above, s.why
    for p in s.held_above:
        if B < p <= T:
            above[p - B - 1] = CODE_HELD
    for p in s.live_above:
        if p > T:
            return dest, above, "page %d live above the stage top %d" % (p, T)
        if above[p - B - 1] == CODE_HELD:
            continue
        np_ = s.owner.get(int(p))
        if np_ is None or np_.pinned:
            above[p - B - 1] = CODE_BLOCK
        else:
            above[p - B - 1] = CODE_BACKED if np_.backed else CODE_UNBACKED
    return dest, above, None


def agreed_survey(s: Survey, top_page: int, dest_min: Sequence[int], above_min: Sequence[int],
                  above_max: Sequence[int]) -> Survey:
    """S98: the survey every rank plans on, built from the GROUP vote (element-wise
    MIN, and MAX for the pages above the floor) -- the same on every rank whose tree
    groups the pages into the same nodes (the plan digest still checks that).

    rc22/rc23 (NF Form A): ``backed`` reads host copies and mamba anchors, and the
    attention host (real GDN states) and the expert workers (R12 shadow pool,
    KV-only anchors under eviction) hold those differently; a rank-local ``backed``
    picked different evictions and destinations per rank -> a different digest, a
    crash-stop. Here a node is backed only when EVERY rank says so for every one of
    its pages (group AND: never evict what one rank has not secured), a page blocks
    when it blocks on ANY rank, a destination is a page free on EVERY rank, and a
    page live on some ranks and free on others refuses by name (the replicas differ;
    nothing moves anywhere)."""
    B, T = int(s.floor_page), int(top_page)
    free = [p for p in range(1, B + 1) if int(dest_min[p - 1]) >= 1]
    free += [B + 1 + i for i in range(max(0, T - B)) if int(above_min[i]) >= CODE_FREE]
    live: List[int] = []
    held: List[int] = []
    owner: Dict[int, NodePages] = {}
    copies: Dict[int, NodePages] = {}
    blocked_nodes = set()
    unbacked_nodes = set()
    why = s.why
    for i in range(max(0, T - B)):
        p, lo, hi = B + 1 + i, int(above_min[i]), int(above_max[i])
        if lo >= CODE_IDLE:
            continue           # not live on any rank (a destination when free on all)
        if lo == CODE_HELD:
            held.append(p)
            continue
        live.append(p)
        if lo >= CODE_UNBACKED and hi >= CODE_IDLE and why is None:
            why = ("page %d (tokens from %d) is live on some ranks and free on others -- the "
                   "allocators are no replicas, no move" % (p, p * s.page_size))
        np_ = s.owner.get(p)
        if np_ is None:
            continue           # blocks by "held outside the tree" (CODE_BLOCK on this rank)
        cp = copies.get(id(np_))
        if cp is None:
            cp = copies[id(np_)] = NodePages(node=np_.node, pages=list(np_.pages), pinned=False,
                                             backed=True, last=np_.last)
        if lo == CODE_BLOCK:
            blocked_nodes.add(id(np_))
        elif lo == CODE_UNBACKED:
            unbacked_nodes.add(id(np_))
        owner[p] = cp
    for k, cp in copies.items():
        cp.pinned = k in blocked_nodes
        cp.backed = k not in unbacked_nodes
    return Survey(s.page_size, s.num_pages, B, live, free, owner, list(copies.values()),
                  why=why, held_above=held)


def plan_for(s: Survey, boundary_page: int) -> Tuple[Optional[Plan], str]:
    """Moves (and, only as the fallback, evictions of BACKED nodes) that leave no
    live page above ``boundary_page`` -- or None and why. All or nothing: a plan
    that cannot clear the boundary moves no page."""
    if s.why is not None:
        return None, s.why
    B = int(boundary_page)
    if B < int(s.floor_page):
        return None, "boundary %d below the surveyed floor page %d" % (B, s.floor_page)
    held = [p for p in s.held_above if p > B]
    if held:
        return None, ("page %d (tokens from %d) withheld above the boundary by a cap filter other "
                      "than the stage cap (backing relief) -- the floor counts it, no move" % (
                          held[0], held[0] * s.page_size))
    above = [p for p in s.live_above if p > B]
    if not above:
        return Plan(boundary_page=B), ""
    movers: Dict[int, NodePages] = {}
    for p in above:
        np_ = s.owner.get(p)
        if np_ is None:
            return None, "page %d (tokens from %d) held outside the tree (a running request / a session)" % (
                p, p * s.page_size)
        if np_.pinned:
            return None, "page %d pinned (node %s: lock / write-through / load-back)" % (
                p, getattr(np_.node, "id", "?"))
        movers[id(np_)] = np_
    free = [p for p in s.free if p <= B]
    # N4 (review S77-A): the order is REPLICATED state only -- page ids, never the
    # rank-local access clock. Un-backed first (they must not be lost); then the
    # backed ones whose pages were allocated LAST (highest page above the boundary:
    # the allocator hands out the lowest free page, so the newest turns sit highest)
    # keep the room, the older ones are the fallback evictions. The plan digest (N3)
    # still compares the whole plan across the ranks.
    def _key(m):
        return (-max(m.pages), min(m.pages))

    # S98: ``backed`` is NOT replicated on Form A -- it reads the host copy and the
    # mamba anchor, which the attention host (TP0, the real GDN states) and an expert
    # worker (byteless R12 shadow pool, KV-only anchors under eviction) hold
    # differently. With room for every mover it decides nothing (every node moves),
    # so it must not reorder the destinations either: a different order is a
    # different digest, a crash-stop before anything moved. Page ids alone then.
    need = sum(sum(1 for p in m.pages if p > B) for m in movers.values())
    if len(free) >= need:
        order = sorted(movers.values(), key=_key)
    else:
        order = (sorted((m for m in movers.values() if not m.backed), key=_key)
                 + sorted((m for m in movers.values() if m.backed), key=_key))
    plan = Plan(boundary_page=B)
    rest: List[NodePages] = []
    for m in order:
        src = [p for p in m.pages if p > B]
        if len(free) >= len(src):
            dst, free = free[:len(src)], free[len(src):]
            plan.moves.append((m, src, dst))
        elif m.backed:
            rest.append(m)
        else:
            return None, "no free page below %d for un-backed node %s (%d pages)" % (
                B, getattr(m.node, "id", "?"), len(src))
    if rest:
        out = {id(m.node) for m in rest}
        for m in rest:
            stays = [c for c in _device_children(m.node) if id(c) not in out]
            if stays:
                return None, "backed node %s has no room below %d and a device child that stays (%s)" % (
                    getattr(m.node, "id", "?"), B, getattr(stays[0], "id", "?"))
        # leaf first: a node is evicted only after every evicted child below it
        depth = {}
        for m in rest:
            d, x = 0, m.node
            while getattr(x, "parent", None) is not None:
                d, x = d + 1, x.parent
            depth[id(m)] = d
        plan.evicts = sorted(rest, key=lambda m: -depth[id(m)])
    return plan, ""


def _claim_pages(allocator: Any, pages: Sequence[int]) -> bool:
    """Take ``pages`` out of the allocator's free lists (all of them, or none)."""
    import torch

    want = torch.tensor(sorted({int(p) for p in pages}), dtype=torch.int64)
    if not want.numel():
        return True
    found = torch.zeros(0, dtype=torch.int64)
    lists = {}
    for name in ("free_pages", "release_pages"):
        ids = getattr(allocator, name, None)
        if ids is None or not ids.numel():
            continue
        cpu = ids.detach().to("cpu", torch.int64)
        hit = torch.isin(cpu, want)
        lists[name] = (ids, hit)
        found = torch.cat([found, cpu[hit]])
    if torch.unique(found).numel() != want.numel():
        return False
    for name, (ids, hit) in lists.items():
        setattr(allocator, name, ids[(~hit).to(ids.device)])
    return True


def _remap_value(value, P: int, remap: Dict[int, int]):
    """The node's token ids with every moved page replaced (offset in page kept)."""
    import torch

    cpu = value.reshape(-1).to("cpu", torch.int64)
    pages = cpu // P
    hi = int(max(int(pages.max()) if cpu.numel() else 0, max(remap) if remap else 0))
    lut = torch.arange(hi + 1, dtype=torch.int64)
    if remap:
        lut[torch.tensor(list(remap.keys()), dtype=torch.int64)] = torch.tensor(
            list(remap.values()), dtype=torch.int64)
    out = lut[pages] * P + cpu % P
    return out.to(dtype=value.dtype, device=value.device).reshape(value.shape)


def plan_digest(stage: int, plan: Plan) -> int:
    """N3: a 62-bit digest of what this rank will do -- the target stage, every move
    (sources, destinations) and every eviction (its pages above the boundary), all
    in plan order. Equal on every rank exactly when the plans are the same."""
    import hashlib

    h = hashlib.blake2b(digest_size=8)
    h.update(repr((int(stage), int(plan.boundary_page),
                   [(tuple(s), tuple(d)) for _m, s, d in plan.moves],
                   [tuple(m.pages) for m in plan.evicts])).encode())
    return int.from_bytes(h.digest(), "big") >> 2


def _sync_device(sync=None) -> None:
    import torch

    if sync is not None:
        sync()
    elif torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.synchronize()


def execute_moves(cache: Any, allocator: Any, plan: Plan, copier: Any, *, sync=None) -> Result:
    """The MOVES of the plan on this rank: claim the destinations, sync, copy, sync,
    re-point the nodes (transactional), free the sources. Raises CompactRefused
    (named) with everything rolled back -- no node re-pointed, destinations free.

    OVERLAP (H1 of the review): the tick runs in ``get_next_batch_to_run`` while the
    previous batch may still run. That batch writes only its own ``out_cache_loc``
    pages (never a free destination, never an idle tree page: a tree page is a full
    page of committed tokens) and reads at most tree pages (read against read). The
    first sync still makes the copy start after it; the second lands the copy before
    a node names the destination and before a source goes back to the allocator."""
    import torch

    t0 = time.perf_counter()
    P = int(copier.page_size)
    full = _full_ct().FULL
    nbytes = 0
    dev = getattr(getattr(allocator, "free_pages", None), "device", "cpu")
    if plan.moves:
        src_all = [p for _m, s, _d in plan.moves for p in s]
        dst_all = [p for _m, _s, d in plan.moves for p in d]
        if not _claim_pages(allocator, dst_all):
            raise CompactRefused("destination pages %s.. not all free" % dst_all[:8])
        try:
            _sync_device(sync)
            nbytes = copier.copy(src_all, dst_all)
            _sync_device(sync)
        except Exception as exc:  # noqa: BLE001 -- nothing re-pointed yet: destinations go back
            allocator.free(torch.tensor([p * P for p in dst_all], dtype=torch.int64, device=dev))
            raise CompactRefused("copy failed before any node moved: %s: %s"
                                 % (type(exc).__name__, exc)) from exc
        done = []
        try:
            for m, src, dst in plan.moves:
                cd = m.node.component_data[full]
                orig = cd.value
                cd.value = _remap_value(orig, P, dict(zip(src, dst)))
                done.append((cd, orig))
        except Exception as exc:  # noqa: BLE001 -- roll back, then refuse by name
            for cd, orig in reversed(done):
                cd.value = orig
            allocator.free(torch.tensor([p * P for p in dst_all], dtype=torch.int64, device=dev))
            raise CompactRefused("repoint failed after %d of %d nodes (rolled back, destinations "
                                 "freed): %s: %s" % (len(done), len(plan.moves), type(exc).__name__,
                                                     exc)) from exc
        allocator.free(torch.tensor([p * P for p in src_all], dtype=torch.int64, device=dev))
    return Result(moved_nodes=len(plan.moves), moved_pages=plan.moved_pages, evicted_nodes=0,
                  evicted_tokens=0, bytes=int(nbytes), ms=(time.perf_counter() - t0) * 1000.0)


def execute_evicts(cache: Any, plan: Plan) -> Tuple[int, int]:
    """The fallback EVICTIONS (after the group saw every rank's moves land): backed
    device leaves, leaf first, through the tree's own ``_evict_to_host`` (the host
    copy stays). Returns (nodes, tokens)."""
    full = _full_ct().FULL
    nodes = tokens = 0
    for m in plan.evicts:
        node = m.node
        if not cache._is_device_leaf(node):
            raise CompactRefused("node %s is no device leaf at its eviction" % getattr(node, "id", "?"))
        tracker = {ct: 0 for ct in cache.tree_components}
        cache._evict_to_host(node, tracker)
        nodes += 1
        tokens += int(tracker.get(full, 0))
    return nodes, tokens


def execute(cache: Any, allocator: Any, plan: Plan, copier: Any, *, sync=None) -> Result:
    """Single-rank form (tests, no group): the moves, then the evictions -- the order
    the group path keeps with a collective between the two (d_seat_compact's order:
    a refused move leaves nothing evicted)."""
    res = execute_moves(cache, allocator, plan, copier, sync=sync)
    res.evicted_nodes, res.evicted_tokens = execute_evicts(cache, plan)
    return res
