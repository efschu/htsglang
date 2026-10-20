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

S112 (NF rc27 893d09fa3f, boot 1020_145331: 0 moved, three refusal kinds). (b) the
lock of a RUNNING decode request (WITNESS ``tree(node=11 locks=[1, 0, 0] ...),
request(rid=pdflip-0-14 running)`` on all three ranks, 136 s, then the next turn) is
counted, not just seen: a lock the running requests and the in-flight write-throughs
explain moves -- the requests' ``req_to_token`` rows and ``prefix_indices`` with it
(own prior art: weg2 ``dual_d_compact`` 733ee1ca5b) -- and is never evicted. (a) room
is MADE: agreed backed idle device leaves are evicted in cut-independent UNITS (a chain
of single-device-child links), below the boundary too, leaf first, chosen to close the
group's gap. (c) the allocators of the Form A ranks are no replicas (n=64: page 3996
free on TP0, node 189 on TP1/TP2): the MOVES are rank-local, the group decides the
stage and the evictions.

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
    """One tree node that holds device KV pages -- or (``node`` None) the pages running
    requests hold outside the tree (S112)."""

    node: Any
    pages: List[int]
    #: something reads or writes its pages that the move cannot follow (an unexplained
    #: lock, a load-back, a write op still queued on the host, two owners)
    pinned: bool
    #: its host (L2) copy has landed for every component it holds on the device
    backed: bool
    #: the rank-local access clock -- logged only, never a decision (review N4)
    last: float = 0.0
    #: S112: movable, never evicted -- a lock that the running decode requests and the
    #: in-flight write-throughs on its path explain, a page a running request's row
    #: names, or a write-through in flight whose device read is already enqueued
    move_only: bool = False


@dataclass
class Survey:
    page_size: int
    num_pages: int
    #: the lowest boundary page surveyed
    floor_page: int
    #: live pages (not free, not withheld) above ``floor_page``, ascending
    live_above: List[int]
    #: free pages (in a free list, not withheld), ascending
    free: List[int]
    #: live page -> the tree node (or the running requests) holding it -- every page
    #: since S112 (the room-making evictions need the nodes below the boundary too)
    owner: Dict[int, NodePages]
    nodes: List[NodePages]
    why: Optional[str] = None
    #: S84: pages above ``floor_page`` a cap filter OTHER than the stage cap withholds
    #: (the backing-relief rung's KvRowCap), ascending. Not live, not free -- but
    #: ``max_live_page`` (the floor) counts them, and must: that filter's release()
    #: hands them back without the stage cap's hook. A plan that leaves one above its
    #: boundary would move pages for a floor that cannot drop.
    held_above: List[int] = field(default_factory=list)
    #: S98c: pages that block on SOME rank (the group vote's MIN is CODE_BLOCK) -- per
    #: page, never spread over a rank-local node (the node cuts differ between ranks)
    blocked: frozenset = frozenset()
    #: S112: live pages at or below ``floor_page``, ascending
    live_below: List[int] = field(default_factory=list)
    #: S112: running request rows read for the survey / pages they name
    run_reqs: int = 0
    run_pages: int = 0
    #: S112: the lowest engaged cap (page): no room above it, freed or free
    cap_top: int = 1 << 40


@dataclass
class Plan:
    boundary_page: int
    #: per LOCAL owner (the re-point groups); never part of the digest (S98c)
    moves: List[Tuple[NodePages, List[int], List[int]]] = field(default_factory=list)
    #: S112: leaf first (a unit's chain bottom-up, units in the order they were chosen)
    evicts: List[NodePages] = field(default_factory=list)
    #: the page pairs (src, dst) in source-page order -- RANK-LOCAL since S112
    pairs: List[Tuple[int, int]] = field(default_factory=list)
    #: every device page the evictions free (all pages of the evicted nodes) -- what
    #: the group digest compares (S112: the evictions change the tree, the moves do not)
    evict_pages: List[int] = field(default_factory=list)

    @property
    def moved_pages(self) -> int:
        return sum(len(s) for _n, s, _d in self.moves)

    @property
    def evict_first(self) -> bool:
        """S112: an eviction that frees a page at or below the boundary makes ROOM the
        moves take: then the evictions go first. Otherwise the moves first (S77 order: a
        refused move leaves nothing evicted). From the agreed eviction pages and the
        boundary only -- the same on every rank, so is the collective order."""
        return any(int(p) <= int(self.boundary_page) for p in self.evict_pages)


@dataclass
class Result:
    moved_nodes: int
    moved_pages: int
    evicted_nodes: int
    evicted_tokens: int
    bytes: int
    ms: float
    #: S112: request rows / prefix tensors re-pointed with the pages
    remapped_rows: int = 0


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


def writes_not_enqueued(cache: Any) -> int:
    """S112: write-through ops whose device READ is not yet on a stream -- queued on the
    host (``cache_controller.write_queue``: ``consume_gate`` may hold it) or waiting in
    the S108 turn-end copier thread (``_s108_copier.inflight``). Such an op captured the
    source indices and reads them LATER: its node must not move. 0 = every in-flight
    write's read is enqueued on the device, so the compaction's own device sync before
    the copy lands it (read against read, no new wait). No controller with an in-flight
    write = unknown = -1 (fail-closed)."""
    cc = getattr(cache, "cache_controller", None)
    if cc is None:
        return -1        # an in-flight write and no controller to ask: unknown
    n = len(getattr(cc, "write_queue", None) or ())
    cop = getattr(cc, "__dict__", {}).get("_s108_copier")
    if cop is not None:
        n += max(0, int(getattr(cop, "inflight", 0) or 0))
    return n


def _path_counts(cache: Any, starts) -> Tuple[Dict[int, int], Dict[int, int]]:
    """(path, self) lock counts the FULL path lock of ``inc_lock_ref(node)`` leaves:
    every device node from ``node`` (its evicted bottom segment skipped) up to the root
    +1 (``full_component.acquire_component_lock``); the node itself +1 (the single-node
    components, Mamba)."""
    full = int(_full_ct().FULL)
    root = getattr(cache, "root_node", None)
    path: Dict[int, int] = {}
    own: Dict[int, int] = {}
    for node in starts:
        if node is None:
            continue
        own[id(node)] = own.get(id(node), 0) + 1
        cur = node
        while cur is not None and cur is not root and cur.component_data[full].value is None:
            cur = getattr(cur, "parent", None)
        while cur is not None and cur is not root and getattr(cur, "parent", None) is not None:
            path[id(cur)] = path.get(id(cur), 0) + 1
            cur = cur.parent
    return path, own


def _req_len(req: Any) -> int:
    """The row length a running request names: its allocated KV (spec-v2 reserves ahead
    of ``kv_committed_len``) or its sequence, the longer."""
    n = int(getattr(req, "kv_allocated_len", 0) or 0)
    seq = int(len(getattr(req, "origin_input_ids", ()) or ()) + len(getattr(req, "output_ids", ()) or ()))
    return max(n, seq)


def survey(cache: Any, allocator: Any, page_size: int, floor_page: int = 0, *,
           running: Sequence[Any] = (), req_to_token: Any = None) -> Survey:
    """The allocator's live pages and the tree node holding each (ONE D2H copy of all
    node values, tensor ops). Pages in no free list and not withheld by the stage cap
    are live; a page two nodes name is pinned in both. S84: every cap filter
    (``cap_filters``) counts, and no free page above the lowest engaged cap is offered.

    S112 (NF rc27 893d09fa3f, boot 1020_145331, every refusal of the boot): the locks
    are COUNTED, not just seen. A running decode request locks its matched chain
    (``_req_inc_lock_ref(req.last_node)``: FULL path lock), an in-flight write-through
    locks the written node's chain (``write_backup``: ``inc_lock_ref``). When a node's
    locks are exactly those (``running`` + ``ongoing_write_through``), its pages can MOVE
    -- the requests' rows and prefix tensors are re-pointed with them -- but never be
    evicted (``move_only``). Any lock more (a chunked request, an unmerged extend, a
    session, an admission pin) stays a pin. ``running`` rows name the requests' own
    pages outside the tree (the decode tail since their last insert): those move too."""
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
    live_below = [int(x) for x in torch.nonzero(live_m & (idx <= B)).flatten().tolist()]
    free = [int(x) for x in torch.nonzero(free_m & (~held_m) & (idx >= 1) & (idx <= top)).flatten().tolist()]
    held_above = [int(x) for x in torch.nonzero(other_m & (idx > B)).flatten().tolist()]
    why = tree_why(cache)

    def _mk(owner, nodes, **kw):
        return Survey(P, n, B, live_above, free, owner, nodes, held_above=held_above,
                      live_below=live_below, cap_top=int(top), **kw)

    if why is not None:
        return _mk({}, [], why=why)
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
    wt = getattr(cache, "ongoing_write_through", None) or {}
    lb = getattr(cache, "ongoing_load_back", None) or {}
    queued = writes_not_enqueued(cache) if (wt or any(
        getattr(nd, "write_through_pending_id", None) is not None for nd, _v in rows)) else 0
    wt_starts = []
    for e in list(wt.values()):
        lock_node = e[0] if isinstance(e, tuple) else getattr(e, "node", None)
        params = e[1] if isinstance(e, tuple) and len(e) > 1 else getattr(e, "lock_params", None)
        if params is not None:
            wt_starts.append(lock_node)
    running = [r for r in (running or ()) if getattr(r, "req_pool_idx", None) is not None]
    wt_path, wt_own = _path_counts(cache, wt_starts)
    run_path, run_own = _path_counts(cache, [getattr(r, "last_node", None) for r in running])
    if rows:
        try:
            ids = torch.cat([v for _n, v in rows]).to("cpu", torch.int64)
        except RuntimeError:  # mixed devices: one copy each
            ids = torch.cat([v.to("cpu", torch.int64) for _n, v in rows])
        counts = torch.tensor([int(v.numel()) for _n, v in rows], dtype=torch.int64)
        nidx = torch.repeat_interleave(torch.arange(len(rows), dtype=torch.int64), counts)
        pages = ids // P
        sel = pages >= 1
        pairs = torch.unique(pages[sel] * len(rows) + nidx[sel])
        pp, nn = (pairs // len(rows)).tolist(), (pairs % len(rows)).tolist()
    else:
        pp, nn = [], []
    by_row: Dict[int, NodePages] = {}
    for p, r in zip(pp, nn):
        np_ = by_row.get(r)
        if np_ is None:
            node = rows[r][0]
            nid = getattr(node, "id", None)
            in_wt = nid in wt or getattr(node, "write_through_pending_id", None) is not None
            cds = node.component_data
            locks = [int(getattr(cd, "lock_ref", 0) or 0) for cd in cds]
            k = id(node)
            explained = (locks[full] == wt_path.get(k, 0) + run_path.get(k, 0) and all(
                lk <= wt_own.get(k, 0) + run_own.get(k, 0) for i, lk in enumerate(locks) if i != full))
            locked = any(lk > 0 for lk in locks)
            pinned = (nid in lb or (locked and not explained) or (in_wt and queued != 0))
            mamba_ok = (len(cds) <= mamba or cds[mamba].value is None
                        or cds[mamba].host_value is not None
                        or bool(getattr(node, "pdflip_anchor_secured", False)))
            np_ = NodePages(node=node, pages=[], pinned=bool(pinned),
                            backed=bool(cds[full].host_value is not None and not in_wt and mamba_ok),
                            last=float(getattr(node, "last_access_time", 0) or 0),
                            move_only=bool((locked or in_wt) and not pinned))
            by_row[r] = np_
            nodes.append(np_)
        np_.pages.append(int(p))
        prev = owner.get(int(p))
        if prev is not None and prev is not np_:
            prev.pinned = np_.pinned = True  # a page two nodes name moves for neither
        owner[int(p)] = np_
    # S112: the running requests' rows -- their own pages outside the tree move with
    # them; a tree page a running row names is never evicted, whatever its lock says
    run_pages = 0
    if running and req_to_token is not None:
        req_np = NodePages(node=None, pages=[], pinned=False, backed=False, move_only=True)
        seen = set()
        for r in running:
            m = min(_req_len(r), int(req_to_token.shape[1]))
            if m <= 0:
                continue
            rp = torch.unique(req_to_token[int(r.req_pool_idx), :m].to("cpu", torch.int64) // P)
            for p in rp.tolist():
                p = int(p)
                if p < 1 or p > n or not bool(live_m[p]) or p in seen:
                    continue
                seen.add(p)
                o = owner.get(p)
                if o is None:
                    req_np.pages.append(p)
                    owner[p] = req_np
                elif not o.pinned:
                    o.move_only = True
        run_pages = len(seen)
        if req_np.pages:
            nodes.append(req_np)
    return _mk(owner, nodes, run_reqs=len(running), run_pages=run_pages)


def _device_children(node: Any, gone=frozenset()) -> List[Any]:
    full = int(_full_ct().FULL)
    return [c for c in getattr(node, "children", {}).values()
            if c.component_data[full].value is not None and id(c) not in gone]


#: S98 page codes of the group vote (``page_codes``). The MIN picks the most blocking
#: answer of any rank: a cap filter other than the stage cap withholds it < live and
#: blocking (outside the tree and no running row, an unexplained lock, a load-back, a
#: queued write, two owners) < live and movable only (not backed, a running request's,
#: an explained lock, a write in flight -- S112) < a live tree page backed and idle <
#: not live and not claimable (withheld by the stage cap, a staged free) < free.
CODE_HELD = -1
CODE_BLOCK = 0
CODE_UNBACKED = 1
CODE_BACKED = 2
CODE_IDLE = 3
CODE_FREE = 4


def page_codes(s: Survey, top_page: int) -> Tuple[List[int], Optional[str]]:
    """S112: this rank's code for every page 1..``top_page`` -- (codes, why). The vector
    length is a function of ``top_page`` alone (replicated), so every rank enters the
    collective with the same length. ``why`` names a rank-local reason the survey
    cannot be voted on (then the caller votes "no")."""
    T = int(top_page)
    free = set(int(p) for p in s.free)
    codes = [CODE_FREE if p in free else CODE_IDLE for p in range(1, T + 1)]
    if s.why is not None:
        return codes, s.why
    for p in s.held_above:
        if 1 <= p <= T:
            codes[p - 1] = CODE_HELD
    for p in list(s.live_below) + list(s.live_above):
        if p > T:
            return codes, "page %d live above the stage top %d" % (p, T)
        if codes[p - 1] == CODE_HELD:
            continue
        np_ = s.owner.get(int(p))
        if np_ is None or np_.pinned:
            codes[p - 1] = CODE_BLOCK
        elif np_.move_only or not np_.backed or np_.node is None:
            codes[p - 1] = CODE_UNBACKED
        else:
            codes[p - 1] = CODE_BACKED
    return codes, None


@dataclass
class Agreement:
    """S112: what the GROUP agreed in the vote -- per page the MIN and MAX code over
    the ranks (1..T), and per candidate boundary page the MAX over the ranks of the
    local gap (live pages above it minus free pages at or below it)."""

    lo: Sequence[int]
    hi: Sequence[int]
    gaps: Dict[int, int] = field(default_factory=dict)

    def code(self, p: int) -> Tuple[int, int]:
        if 1 <= p <= len(self.lo):
            return int(self.lo[p - 1]), int(self.hi[p - 1])
        return CODE_BLOCK, CODE_BLOCK


def own_agreement(s: Survey, top_page: int, boundaries: Sequence[int] = ()) -> Agreement:
    """The single-rank agreement: this rank's own codes and gaps (tests, no group)."""
    codes, _w = page_codes(s, top_page)
    return Agreement(codes, codes, {int(b): local_gap(s, int(b)) for b in boundaries})


def local_gap(s: Survey, boundary_page: int) -> int:
    B = int(boundary_page)
    return (sum(1 for p in s.live_above if p > B)
            - sum(1 for p in s.free if p <= B))


def _evictable(np_: NodePages, ag: Agreement) -> bool:
    """A node may be EVICTED only when every rank holds every one of its pages live,
    backed and idle (group MIN and MAX both CODE_BACKED): never what one rank has not
    secured, never a page whose replicas differ (S112: the allocators are no replicas
    under Form A -- rc27 n=64 page 3996 free on TP0, node 189 on TP1/TP2)."""
    if np_.node is None or np_.pinned or np_.move_only or not np_.backed or not np_.pages:
        return False
    full = int(_full_ct().FULL)
    v = np_.node.component_data[full].value
    if v is None or not v.numel():
        return False
    for p in np_.pages:
        lo, hi = ag.code(int(p))
        if lo != CODE_BACKED or hi != CODE_BACKED:
            return False
    return True


def evict_units(s: Survey, ag: Agreement, gone=frozenset()) -> List[List[NodePages]]:
    """S112: the evictable UNITS -- from every device leaf upward while the parent is
    evictable and has exactly one device child (the one below), leaf first.

    Cut-independent: a node split that only one rank made (rc24 _113244: the same 177
    slots in 32 nodes on TP0, 35 on TP1/TP2) is a single-device-child link, and a chain
    of such links is ONE unit -- the same pages on every rank whose tree holds the same
    content; branch points are content, the same everywhere. A unit never leaves a
    device node under an evicted one (the FULL invariant)."""
    by_node = {id(np_.node): np_ for np_ in s.nodes if np_.node is not None}
    out: List[List[NodePages]] = []
    for np_ in s.nodes:
        if np_.node is None or id(np_.node) in gone or _device_children(np_.node, gone):
            continue
        chain: List[NodePages] = []
        x = np_
        while x is not None and _evictable(x, ag):
            chain.append(x)
            par = getattr(x.node, "parent", None)
            if par is None or getattr(par, "parent", None) is None:
                break                      # the root
            if len(_device_children(par, gone)) != 1:
                break                      # a branch point: the other branch stays
            x = by_node.get(id(par))
        if chain:
            out.append(chain)
    return out


def _unit_pages(unit: List[NodePages]) -> List[int]:
    return sorted({int(p) for m in unit for p in m.pages})


def choose_evictions(s: Survey, ag: Agreement, gap: int) -> Tuple[Optional[List[NodePages]], str]:
    """S112: evictions that close ``gap`` (every evicted page closes it by one: a page
    above the boundary needs no room any more, a page at or below it IS room). From the
    replicated inputs only (agreed codes, the agreed gap, the units' pages): best fit --
    the smallest unit that closes the rest, else the largest one and go on; ties by the
    unit's pages (older = lower first, never the rank-local clock, N4). Returns the
    nodes leaf first, or None and why."""
    gone: set = set()
    out: List[NodePages] = []
    rest = int(gap)
    while rest > 0:
        units = evict_units(s, ag, frozenset(gone))
        if not units:
            return None, "no evictable unit left (agreed backed, idle device leaves) for %d pages" % rest
        sized = [(len(_unit_pages(u)), max(_unit_pages(u)), min(_unit_pages(u)), i)
                 for i, u in enumerate(units)]
        fits = [t for t in sized if t[0] >= rest]
        pick = min(fits) if fits else max(sized, key=lambda t: (t[0], -t[1], -t[2], -t[3]))
        unit = units[pick[3]]
        out.extend(unit)
        gone.update(id(m.node) for m in unit)
        rest -= pick[0]
    return out, ""


def plan_for(s: Survey, boundary_page: int, ag: Optional[Agreement] = None) -> Tuple[Optional[Plan], str]:
    """Moves (rank-local, S112) and room-making evictions (group-agreed units of backed
    idle device leaves, S112) that leave no live page above ``boundary_page`` -- or None
    and why. All or nothing: a plan that cannot clear the boundary moves no page."""
    if s.why is not None:
        return None, s.why
    B = int(boundary_page)
    if B < int(s.floor_page):
        return None, "boundary %d below the surveyed floor page %d" % (B, s.floor_page)
    local = ag is None
    if local:
        ag = own_agreement(s, max([B] + list(s.live_above) + list(s.free) + [s.floor_page]), [B])
    held = [p for p in s.held_above if p > B]
    if held:
        return None, ("page %d (tokens from %d) withheld above the boundary by a cap filter other "
                      "than the stage cap (backing relief) -- the floor counts it, no move" % (
                          held[0], held[0] * s.page_size))
    above = [p for p in s.live_above if p > B]
    if not above:
        return Plan(boundary_page=B), ""
    def _local():
        for p in above:
            np_ = s.owner.get(p)
            if np_ is None:
                return "page %d (tokens from %d) held outside the tree (a running request / a session)" % (
                    p, p * s.page_size)
            if np_.pinned:
                return "page %d pinned (node %s: lock / write-through / load-back)" % (
                    p, getattr(np_.node, "id", "?"))
        return None

    def _group():
        # S98c: a page that blocks on ANY rank refuses the candidate on every rank, by name
        for p in sorted(set(int(q) for q in s.blocked) | {q for q in range(B + 1, len(ag.lo) + 1)
                                                           if ag.code(q)[0] == CODE_BLOCK}):
            if p > B:
                return ("page %d (tokens from %d) blocks on a rank (outside the tree / lock / "
                        "write-through / load-back there)" % (p, p * s.page_size))
        return None

    # a group verdict names the group reason on every rank alike; alone, the own one
    for check in ((_local, _group) if local else (_group, _local)):
        w = check()
        if w is not None:
            return None, w
    free = [p for p in s.free if p <= B]
    gap = max(len(above) - len(free), int(ag.gaps.get(B, len(above) - len(free))))
    plan = Plan(boundary_page=B)
    ev_set: set = set()
    if gap > 0:
        evicts, why = choose_evictions(s, ag, gap)
        if evicts is None:
            stay = [s.owner[p] for p in above if not s.owner[p].backed or s.owner[p].move_only]
            if stay:
                m = stay[-1]
                return None, ("no free page below %d for un-backed node %s (%d pages): %s" % (
                    B, getattr(m.node, "id", "req"), sum(1 for p in above if s.owner[p] is m), why))
            return None, "no free page below %d for %d pages above it: %s" % (B, len(above), why)
        plan.evicts = evicts
        full = int(_full_ct().FULL)
        for m in evicts:
            v = m.node.component_data[full].value
            ev_set.update(int(x) for x in (v.reshape(-1).to("cpu") // s.page_size).unique().tolist())
        plan.evict_pages = sorted(ev_set)
    stay = [p for p in above if p not in ev_set]
    room = sorted(set(free) | {p for p in ev_set if p <= min(B, int(s.cap_top))})
    if len(room) < len(stay):
        return None, "no free page below %d for %d pages above it after %d evictions here" % (
            B, len(stay), len(plan.evicts))
    # S98c: PAGE-level -- the pages above the boundary ascending onto the room below it
    # ascending; the nodes only group the re-pointing, locally. S112: both lists are this
    # rank's own (the allocators are no replicas under Form A); the group agrees on the
    # stage and on the evictions, never on the page ids of a move.
    plan.pairs = list(zip(stay, room[:len(stay)]))
    groups: Dict[int, Tuple[NodePages, List[int], List[int]]] = {}
    for src, dst in plan.pairs:
        m = s.owner[src]
        g = groups.get(id(m))
        if g is None:
            g = groups[id(m)] = (m, [], [])
            plan.moves.append(g)
        g[1].append(src)
        g[2].append(dst)
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
    """N3: a 62-bit digest of what the GROUP must do alike -- the target stage, its
    boundary and every page an eviction frees. S112: the moves are rank-local (the
    allocators are no replicas under Form A: rc27 1020_145331 n=64, page 3996 free on
    TP0 and node 189 on TP1/TP2 after the same finish inserted at anchor depth 90880
    there and 90624 here); they change no tree, only page ids, so they are no part of
    the digest. The evictions change the tree's device presence: equal on every rank
    exactly when the ranks evict the same pages."""
    import hashlib

    h = hashlib.blake2b(digest_size=8)
    h.update(repr((int(stage), int(plan.boundary_page),
                   [int(p) for p in sorted(plan.evict_pages)])).encode())
    return int.from_bytes(h.digest(), "big") >> 2


def _sync_device(sync=None) -> None:
    import torch

    if sync is not None:
        sync()
    elif torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.synchronize()


def execute_moves(cache: Any, allocator: Any, plan: Plan, copier: Any, *, sync=None,
                  remap_refs=None) -> Result:
    """The MOVES of the plan on this rank: claim the destinations, sync, copy, sync,
    re-point the nodes (transactional), free the sources. Raises CompactRefused
    (named) with everything rolled back -- no node re-pointed, destinations free.

    OVERLAP (H1 of the review): the tick runs in ``get_next_batch_to_run`` while the
    previous batch may still run. That batch writes only its own ``out_cache_loc``
    pages (never a free destination) and reads tree pages (read against read). The
    first sync makes the copy start after it -- S112: also after its writes into a
    running request's own pages, which now move too; the second lands the copy before
    a node or a request row (``remap_refs``) names the destination and before a source
    goes back to the allocator. The next batch is prepared after the tick, from the
    re-pointed rows; the previous batch's result frees through ``req_to_token``
    (``release_kv_cache``), i.e. through the re-pointed rows as well. An in-flight
    write-through whose device read is enqueued (``writes_not_enqueued`` == 0) is
    landed by the same first sync (read against read)."""
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
        remapped = 0
        try:
            for m, src, dst in plan.moves:
                if m.node is None:
                    continue       # S112: a running request's own pages -- its row follows below
                cd = m.node.component_data[full]
                orig = cd.value
                cd.value = _remap_value(orig, P, dict(zip(src, dst)))
                done.append((cd, orig))
            if remap_refs is not None:
                # S112: every request row and prefix tensor that names a moved page is
                # re-pointed (computed first, assigned at once -- nothing half-written)
                remapped = int(remap_refs(dict(zip(src_all, dst_all))) or 0)
        except Exception as exc:  # noqa: BLE001 -- roll back, then refuse by name
            for cd, orig in reversed(done):
                cd.value = orig
            allocator.free(torch.tensor([p * P for p in dst_all], dtype=torch.int64, device=dev))
            raise CompactRefused("repoint failed after %d of %d nodes (rolled back, destinations "
                                 "freed): %s: %s" % (len(done), len(plan.moves), type(exc).__name__,
                                                     exc)) from exc
        allocator.free(torch.tensor([p * P for p in src_all], dtype=torch.int64, device=dev))
        return Result(moved_nodes=len(plan.moves), moved_pages=plan.moved_pages, evicted_nodes=0,
                      evicted_tokens=0, bytes=int(nbytes), ms=(time.perf_counter() - t0) * 1000.0,
                      remapped_rows=remapped)
    return Result(moved_nodes=0, moved_pages=0, evicted_nodes=0, evicted_tokens=0, bytes=0,
                  ms=(time.perf_counter() - t0) * 1000.0)


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


def execute(cache: Any, allocator: Any, plan: Plan, copier: Any, *, sync=None,
            remap_refs=None) -> Result:
    """Single-rank form (tests, no group): S112 the EVICTIONS first when they make the
    room the moves take (``Plan.evict_first``: a backed idle unit below the boundary is
    a destination once evicted), else the moves first (S77); the group path keeps a
    collective between the two.
    An eviction is never a loss: only agreed backed nodes go, their host copy stays
    (a later hit reads it back). A refused move after it leaves the tree with less on
    the device on every rank alike, nothing lost."""
    ev = execute_evicts(cache, plan) if plan.evict_first else (0, 0)
    res = execute_moves(cache, allocator, plan, copier, sync=sync, remap_refs=remap_refs)
    if plan.evicts and not plan.evict_first:
        ev = execute_evicts(cache, plan)
    res.evicted_nodes, res.evicted_tokens = ev
    return res
