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

S112b (review nf-review-s112-1020.md, JA MIT AUFLAGE). A1: after rank-local moves the
same page id names other content on another rank -- the evictions are agreed by CONTENT
(``Segment``: a streaming hash of the path's tokens, cut-independent; the vote carries a
content map, the digest the content keys), never by page id. A2: no host wait -- the
survey reads a non-blocking host snapshot (``Snapshot``; a pass whose snapshot has not
landed on every rank defers by name), the move is ordered by events (E0/E1, the F3
ROW-FENCE pattern), the claim and the re-points are device ops, the sources go back
host-informed. A3: every request holder of the scheduler is re-pointed.

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
    #: KVZIEL-1021: STORE-backed -- ``l3_present`` (the group's store verdict), no Full host
    #: copy (the L2 arena spilled it), no write in flight. Secured like an L2-backed node, but
    #: it leaves the device only through the S39 drop (TP0 decides at its broadcast, every
    #: rank follows): never by ``_evict_to_host`` (no host copy to demote to)
    store: bool = False


@dataclass
class Segment:
    """S112b (review A1): a CONTENT unit of the device tree -- a maximal chain of
    single-device-child links whose nodes share one eviction status. Its ``key`` is the
    content (the token stream of the path up to its bottom, its start depth, its length)
    -- never a page id, never a node id: after rank-local moves the same page id names
    other content on another rank, and node cuts differ between ranks (rc24: 32 vs 35)."""

    nodes: List[NodePages]          # top .. bottom
    key: int
    tokens: int
    pages: List[int]
    evictable: bool
    children: List["Segment"] = field(default_factory=list)
    #: KVZIEL-1021: 1 = L2-backed (``_evict_to_host`` now), 2 = store-backed (the S39 drop at
    #: TP0's next broadcast), 0 = not evictable
    kind: int = 0

    @property
    def size(self) -> int:
        return len(self.pages)


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
    owner: Dict[int, NodePages]
    nodes: List[NodePages]
    why: Optional[str] = None
    #: S84: pages above ``floor_page`` a cap filter OTHER than the stage cap withholds
    #: (the backing-relief rung's KvRowCap), ascending. Not live, not free -- but
    #: ``max_live_page`` (the floor) counts them, and must: that filter's release()
    #: hands them back without the stage cap's hook. A plan that leaves one above its
    #: boundary would move pages for a floor that cannot drop.
    held_above: List[int] = field(default_factory=list)
    blocked: frozenset = frozenset()
    #: S112: live pages at or below ``floor_page``, ascending
    live_below: List[int] = field(default_factory=list)
    #: S112: running request rows read for the survey / pages they name
    run_reqs: int = 0
    run_pages: int = 0
    #: S112: the lowest engaged cap (page): no room above it, freed or free
    cap_top: int = 1 << 40
    #: S112b: the content segments of the device tree
    segments: List[Segment] = field(default_factory=list)
    #: S112b: per running request (id) the pages its row names (from the snapshot)
    run_rows: Dict[int, frozenset] = field(default_factory=dict)
    #: S112b: pages a staged free (``free_group``) holds -- neither live nor claimable
    staged: frozenset = frozenset()


@dataclass
class Plan:
    boundary_page: int
    #: per LOCAL owner (the re-point groups); never part of the digest
    moves: List[Tuple[NodePages, List[int], List[int]]] = field(default_factory=list)
    #: leaf first (segments in the order chosen, each bottom-up)
    evicts: List[NodePages] = field(default_factory=list)
    #: the page pairs (src, dst) in source-page order -- RANK-LOCAL
    pairs: List[Tuple[int, int]] = field(default_factory=list)
    #: every device page the evictions free HERE (local ids, information only)
    evict_pages: List[int] = field(default_factory=list)
    #: S112b: the CONTENT keys of the evicted segments -- what the group digest compares
    evict_keys: List[int] = field(default_factory=list)
    #: KVZIEL-1021: store-backed nodes handed to the S39 drop (leaf first, in the chosen
    #: order); a plan with any is STAGED: this pass evicts what can go now and queues these,
    #: the moves follow in a later pass once TP0's broadcast dropped them on every rank
    handoff: List[NodePages] = field(default_factory=list)
    #: KVZIEL-1021: per chosen segment (content key, state: 1 now, 2 handoff, 3 waits) -- the
    #: digest compares it, so every rank stages alike or the plan check refuses uniformly
    split: List[Tuple[int, int]] = field(default_factory=list)

    @property
    def staged(self) -> bool:
        return bool(self.handoff) or any(st == 3 for _k, st in self.split)

    @property
    def moved_pages(self) -> int:
        return sum(len(s) for _n, s, _d in self.moves)

    @property
    def evict_first(self) -> bool:
        """S112b: the agreed evictions always go first -- with rank-local page ids, whether
        an eviction is ROOM is a rank-local question; "always first" is replicated (the
        evictions themselves are agreed by content), so is the collective order."""
        return bool(self.evicts)


@dataclass
class Result:
    moved_nodes: int
    moved_pages: int
    evicted_nodes: int
    evicted_tokens: int
    bytes: int
    ms: float
    #: request rows / prefix tensors re-pointed with the pages
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
    write's read is enqueued on the device; the move's fence (E0: the current stream
    waits on the write stream's tail, device side) orders the copy after it. No
    controller with an in-flight write = unknown = -1 (fail-closed)."""
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


# ---- S112b/c: the survey reads a HOST SNAPSHOT, the move is ordered by events ---------------------
#
# S112c (re-review nf-rereview-s112b-1020, item 0): the stream handles, the E0/E1 fence and the deferred
# D2H read are F3's (``pdflip.d_row_fence``: ``ops()``, ``RowFence``, ``DeferredRead``) -- one fence
# implementation on the line. KV-COMPACT keeps only its own stream SET (the KV streams, not the expert
# row streams ``RowFence.begin`` would take).

_KV_STREAMS: List[Any] = []


def register_kv_stream(stream) -> None:
    """S112c (B6): a stream that reads or writes KV pages (idempotent; None ignored). The move's fence
    orders against every registered one besides the caller's explicit set."""
    if stream is None:
        return
    for s in _KV_STREAMS:
        if s is stream:
            return
    _KV_STREAMS.append(stream)


def _rf():
    from flliper.srt.pdflip import d_row_fence as RF

    return RF


def kv_fence_begin(streams: Sequence[Any]):
    """E0 with an EXPLICIT stream set: the current stream waits (device side) on an event at the tail of
    every KV stream; returns F3's ``RowFence`` (``close()`` = E1, those streams wait on the current one)."""
    RF = _rf()
    o = RF.ops()
    cur = o.current_stream()
    out: List[Any] = []
    for s in list(streams) + list(_KV_STREAMS):
        if s is None or RF._same(s, cur) or any(RF._same(s, x) for x in out):
            continue
        out.append(s)
    for s in out:
        ev = o.event()
        o.record(ev, s)
        o.wait(cur, ev)
    return RF.RowFence(o, cur, out)


def _to_dev(vals: Sequence[int], device):
    """Host ints onto ``device`` without a host wait (pinned, non-blocking H2D)."""
    import torch

    t = torch.tensor([int(v) for v in vals], dtype=torch.int64)
    if getattr(device, "type", str(device)) == "cuda" or str(device).startswith("cuda"):
        return t.pin_memory().to(device, non_blocking=True)
    return t.to(device)


class SyncCount:
    """S112c (B1): MEASURED host waits -- ``torch.cuda.set_sync_debug_mode('warn')`` around a block counts
    every synchronizing call PyTorch reports there (``nonzero`` of a mask index, ``.item()``, a pageable
    copy, ...). 0 without CUDA (nothing can wait there)."""

    def __init__(self):
        self.n = 0
        self._prev = None
        self._rec = None
        self._cm = None

    def __enter__(self):
        try:
            import torch

            if torch.cuda.is_available() and torch.cuda.is_initialized():
                import warnings

                self._prev = torch.cuda.get_sync_debug_mode()
                self._cm = warnings.catch_warnings(record=True)
                self._rec = self._cm.__enter__()
                warnings.simplefilter("always")
                torch.cuda.set_sync_debug_mode("warn")
        except Exception:  # noqa: BLE001 -- an instrument never fails the move
            self._prev = None
        return self

    def __exit__(self, *exc):
        if self._prev is not None:
            import torch

            torch.cuda.set_sync_debug_mode(self._prev)
            self.n = sum(1 for w in (self._rec or []) if "synchroniz" in str(w.message).lower())
            self._cm.__exit__(*exc)
        return False


class Snapshot:
    """S112b/c: the survey's inputs read device -> host by F3's ``DeferredRead`` (pinned, non-blocking, on
    a side stream behind an event of the current stream; on a CPU desk rank a real CLONE, B3). Issued in
    one pass, read in a later one once landed (``ready``: a query, never a wait), and only while every
    source is still the SAME OBJECT (``valid``, B2): the snapshot HOLDS the free-list tensors, the staged
    frees, each cap's withheld tensor and cap, every device node with its value tensor and every running
    request with its row's page count -- an identity check, so a block the caching allocator handed to a
    new tensor (same pointer, length, offset) is never mistaken for the old one."""

    def __init__(self, cache: Any, allocator: Any, page_size: int, running: Sequence[Any] = (),
                 req_to_token: Any = None):
        RF = _rf()
        self.P = max(1, int(page_size))
        self.t0 = time.perf_counter()
        self.running = [r for r in (running or ()) if getattr(r, "req_pool_idx", None) is not None]
        self.src_free = getattr(allocator, "free_pages", None)
        self.src_release = getattr(allocator, "release_pages", None)
        self.src_group = list(getattr(allocator, "free_group", None) or [])
        self.src_caps = [(c, getattr(c, "cap", None), getattr(c, "_withheld", None)) for c in cap_filters(allocator)]
        self.src_nodes: List[Tuple[Any, Any]] = []
        if cache is not None and tree_why(cache) is None:
            full = int(_full_ct().FULL)
            root = getattr(cache, "root_node", None)
            for n in cache._collect_all_nodes():
                v = None if n is root else n.component_data[full].value
                if v is not None:
                    self.src_nodes.append((n, v))
        self.src_rows = [(r, int(r.req_pool_idx), -(-_req_len(r) // self.P)) for r in self.running]
        tensors, self._lay = [], []

        def put(tag, t):
            if t is not None and getattr(t, "numel", lambda: 0)():
                self._lay.append(tag)
                tensors.append(t.detach().reshape(-1))

        put(("free",), self.src_free)
        put(("release",), self.src_release)
        for i, t in enumerate(self.src_group):
            put(("group", i), t)
        for n, v in self.src_nodes:
            put(("node", id(n)), v)
        if req_to_token is not None:
            for r, idx, _pc in self.src_rows:
                m = min(_req_len(r), int(req_to_token.shape[1]))
                if m > 0:
                    put(("row", id(r)), req_to_token[idx, :m])
        self.read = RF.DeferredRead(tensors) if tensors else None
        self.issue_ms = (time.perf_counter() - self.t0) * 1000.0
        self.passes = 0
        self._host = None

    def ready(self) -> bool:
        return self.read is None or bool(self.read.ready())

    def _materialize(self) -> None:
        import torch

        if self._host is not None:
            return
        got = self.read.get() if self.read is not None else []
        if got is None:
            raise RuntimeError("KV-COMPACT snapshot read before it landed")
        self._host = {}
        for tag, h in zip(self._lay, got):
            self._host[tag] = h.to(torch.int64)

    def _get(self, tag):
        import torch

        self._materialize()
        return self._host.get(tag, torch.empty(0, dtype=torch.int64))

    @property
    def free(self):
        return self._get(("free",))

    @property
    def release(self):
        return self._get(("release",))

    @property
    def group(self):
        return [self._get(("group", i)) for i in range(len(self.src_group))]

    def value(self, node):
        return self._get(("node", id(node)))

    def row(self, req):
        tag = ("row", id(req))
        self._materialize()
        return self._host.get(tag)

    def changed(self, cache: Any, allocator: Any, running: Sequence[Any]) -> Optional[str]:
        """None while every source is the same object; else WHICH part changed (B5: named in the
        deferral line): free / release / group / caps / nodes / running."""
        if getattr(allocator, "free_pages", None) is not self.src_free:
            return "free"
        if getattr(allocator, "release_pages", None) is not self.src_release:
            return "release"
        g = list(getattr(allocator, "free_group", None) or [])
        if len(g) != len(self.src_group) or any(a is not b for a, b in zip(g, self.src_group)):
            return "group"
        caps = cap_filters(allocator)
        if len(caps) != len(self.src_caps) or any(
                c is not c0 or getattr(c, "cap", None) != cap0 or getattr(c, "_withheld", None) is not w0
                for c, (c0, cap0, w0) in zip(caps, self.src_caps)):
            return "caps"
        if cache is not None and tree_why(cache) is None:
            full = int(_full_ct().FULL)
            root = getattr(cache, "root_node", None)
            now = [(n, n.component_data[full].value) for n in cache._collect_all_nodes()
                   if n is not root and n.component_data[full].value is not None]
            if len(now) != len(self.src_nodes) or any(
                    a is not c or v is not w for (a, v), (c, w) in zip(now, self.src_nodes)):
                return "nodes"
        run = [r for r in (running or ()) if getattr(r, "req_pool_idx", None) is not None]
        if len(run) != len(self.src_rows) or any(
                r is not r0 or int(r.req_pool_idx) != i0 or -(-_req_len(r) // self.P) != pc0
                for r, (r0, i0, pc0) in zip(run, self.src_rows)):
            return "running"
        return None

    def valid(self, cache: Any, allocator: Any, running: Sequence[Any]) -> bool:
        return self.changed(cache, allocator, running) is None


def _content_state(node: Any, parent_state: Any, root: Any):
    """(hash state, KV-token depth) of the path up to and including ``node`` -- a
    streaming hash over the raw tokens, so a node split on one rank only gives the same
    state below the split (cut-independent). Bigram keys share their boundary token with
    the parent: it is skipped. Cached on the node while its key and parent state hold.
    None when the node carries no readable key (fail-closed: never evicted)."""
    import hashlib
    from array import array

    key = getattr(node, "key", None)
    toks = getattr(key, "token_ids", None)
    if toks is None or parent_state is None:
        return None
    pst, pdepth = parent_state
    n = len(toks)
    lim = getattr(key, "limit", None)
    if lim is not None and lim < n:
        n = int(lim)
    start = 1 if (getattr(key, "is_bigram", False) and getattr(node, "parent", None) is not root) else 0
    full = int(_full_ct().FULL)
    v = node.component_data[full].value
    vlen = int(v.numel()) if v is not None else 0
    tag = (pst.digest(), n, start, vlen)
    c = getattr(node, "_s112_ck", None)
    if c is not None and c[0] == tag:
        return c[1]
    try:
        raw = memoryview(toks)[start:n].tobytes()
    except TypeError:
        raw = array("q", [int(x) for x in list(toks)[start:n]]).tobytes()
    st = pst.copy()
    st.update(raw)
    out = (st, pdepth + vlen)
    try:
        node._s112_ck = (tag, out)
    except Exception:  # noqa: BLE001 -- a slotted node: no cache, recomputed
        pass
    return out


def _root_state(cache: Any, extra_key: Any = None):
    import hashlib

    st = hashlib.blake2b(digest_size=16)
    st.update(repr(extra_key).encode())
    return (st, 0)


def content_key_of_tokens(tokens: Sequence[int], page_size: int, extra_key: Any = None) -> int:
    """The head key (``_head_key``) of a segment whose head is a child of the root holding ``tokens``
    (tests, probes)."""
    from array import array
    from types import SimpleNamespace

    node = SimpleNamespace(key=SimpleNamespace(token_ids=array("q", [int(t) for t in tokens]),
                                               extra_key=extra_key, is_bigram=False, limit=None),
                           parent=None)
    return _head_key(_root_state(None, extra_key), node, None, page_size)


def _seg_key(bottom_state, top_depth: int, tokens: int) -> int:
    import hashlib

    h = hashlib.blake2b(digest_size=8)
    h.update(bottom_state.digest())
    h.update(repr((int(top_depth), int(tokens))).encode())
    return int.from_bytes(h.digest(), "big") >> 2


def _head_key(parent_state, node: Any, root: Any, page_size: int) -> int:
    """S112c (B4): a segment is keyed at its HEAD -- the path state up to its first node plus that
    node's first page of tokens and the start depth. Under Form A the same session ends at different
    depths per rank (rc27 n=64: anchor depth 90880 on TP0, 90624 on the workers); a key at the bottom
    made the whole branch unagreeable ("no evictable unit left"), the head is the same on every rank.
    -1 without a readable key (fail-closed: never evicted)."""
    import hashlib

    key = getattr(node, "key", None)
    toks = getattr(key, "token_ids", None)
    if toks is None or parent_state is None:
        return -1
    start = 1 if (getattr(key, "is_bigram", False) and getattr(node, "parent", None) is not root) else 0
    n = len(toks)
    lim = getattr(key, "limit", None)
    if lim is not None and lim < n:
        n = int(lim)
    end = min(n, start + max(1, int(page_size)))
    try:
        raw = memoryview(toks)[start:end].tobytes()
    except TypeError:
        from array import array

        raw = array("q", [int(x) for x in list(toks)[start:end]]).tobytes()
    h = hashlib.blake2b(digest_size=8)
    h.update(parent_state[0].digest())
    h.update(raw)
    h.update(repr(int(parent_state[1])).encode())
    return int.from_bytes(h.digest(), "big") >> 2


def segments(cache: Any, nodes: List[NodePages], page_size: int) -> List[Segment]:
    """S112b: the content segments of this rank's device tree (see ``Segment``). A node
    is evictable here when it is backed, idle (no lock, no write in flight), not pinned,
    names whole pages that are its own, and carries a key."""
    full = int(_full_ct().FULL)
    root = getattr(cache, "root_node", None)
    by_node = {id(np_.node): np_ for np_ in nodes if np_.node is not None}
    P = max(1, int(page_size))

    def dkids(n):
        return [c for c in getattr(n, "children", {}).values()
                if c.component_data[full].value is not None and id(c) in by_node]

    def ev(np_):
        # KVZIEL-1021: the eviction KIND (0 none, 1 L2-backed, 2 store-backed) -- a segment
        # is a chain of one kind, so it leaves the device by one path
        v = np_.node.component_data[full].value
        if (np_.pinned or np_.move_only or v is None
                or len(np_.pages) != -(-int(v.numel()) // P)):
            return 0
        if np_.backed:
            return 1
        return 2 if np_.store else 0

    states: Dict[int, Any] = {}
    out: List[Segment] = []
    seg_of: Dict[int, Segment] = {}
    order = []
    stack = list(dkids(root)) if root is not None else []
    while stack:                       # top-down: the path state of every device node
        nd = stack.pop()
        par = nd.parent
        ps = (_root_state(cache, getattr(getattr(nd, "key", None), "extra_key", None))
              if par is root else states.get(id(par)))
        states[id(nd)] = _content_state(nd, ps, root)
        order.append(nd)
        stack.extend(dkids(nd))
    pstates: Dict[int, Any] = {}
    for nd in order:
        par = nd.parent
        pstates[id(nd)] = (_root_state(cache, getattr(getattr(nd, "key", None), "extra_key", None))
                           if par is root else states.get(id(par)))
    for nd in order:
        np_ = by_node[id(nd)]
        par = nd.parent
        if (par is not root and id(par) in by_node and len(dkids(par)) == 1
                and ev(by_node[id(par)]) == ev(np_)):
            continue                   # nd continues its parent's segment
        chain, x = [np_], nd
        while True:
            kids = dkids(x)
            if len(kids) != 1 or ev(by_node[id(kids[0])]) != ev(np_):
                break
            x = kids[0]
            chain.append(by_node[id(x)])
        bot = states.get(id(chain[-1].node))
        pages = sorted({int(p) for m in chain for p in m.pages})
        toks = sum(int(m.node.component_data[full].value.numel()) for m in chain)
        evict = bool(ev(np_) and bot is not None and all(states.get(id(m.node)) is not None for m in chain))
        key = _head_key(pstates.get(id(nd)), nd, root, P) if bot is not None else -1
        seg = Segment(nodes=chain, key=key, tokens=toks, pages=pages, evictable=evict,
                      kind=int(ev(np_)) if evict else 0)
        out.append(seg)
        seg_of[id(chain[0].node)] = seg
    for seg in out:
        seg.children = [seg_of[id(c)] for c in dkids(seg.nodes[-1].node) if id(c) in seg_of]
    return out


def survey(cache: Any, allocator: Any, page_size: int, floor_page: int = 0, *,
           running: Sequence[Any] = (), req_to_token: Any = None,
           snap: Optional[Snapshot] = None, store_ok: bool = False) -> Survey:
    """The allocator's live pages, the tree node holding each and the content segments.
    S112b (review A2): with ``snap`` every device value is read from the host snapshot
    (no device read here); without it (single-rank form, the CPU desk) from the live
    tensors. Pages in no free list and not withheld by the stage cap are live; a page two
    nodes name is pinned in both. S84: every cap filter counts, and no free page above
    the lowest engaged cap is offered.

    S112 (NF rc27 893d09fa3f, boot 1020_145331): the locks are COUNTED. A running decode
    request locks its matched chain (``_req_inc_lock_ref(req.last_node)``: FULL path
    lock), an in-flight write-through the written node's chain (``write_backup``:
    ``inc_lock_ref``). When a node's locks are exactly those, its pages can MOVE -- the
    requests' rows and prefix tensors are re-pointed with them -- but never be evicted
    (``move_only``). Any lock more stays a pin. ``running`` rows name the requests' own
    pages outside the tree: those move too.

    KVZIEL-1021: with ``store_ok`` (the S39 drop is on) a node whose KV is in the store
    only (``l3_present``, no Full host copy) is ``store``-backed: secured, evictable through
    the S39 hand-off (``Plan.handoff``)."""
    import torch

    P = max(1, int(page_size))
    n = int(getattr(allocator, "num_pages", 0) or 0)
    B = max(0, int(floor_page))
    running = [r for r in (running or ()) if getattr(r, "req_pool_idx", None) is not None]
    free_m = torch.zeros(n + 2, dtype=torch.bool)
    lists = ((snap.free, snap.release) if snap is not None else
             tuple(_id_tensor(getattr(allocator, nm, None)) for nm in ("free_pages", "release_pages")))
    for ids in lists:
        ids = ids.to(torch.int64)
        ids = ids[(ids >= 0) & (ids <= n)]
        free_m[ids] = True
    staged = torch.zeros(n + 2, dtype=torch.bool)
    groups = snap.group if snap is not None else [_id_tensor(t) for t in (getattr(allocator, "free_group", None) or [])]
    for t in groups:  # staged frees: neither live nor claimable
        ids = t.to(torch.int64) // P
        staged[ids[(ids >= 0) & (ids <= n)]] = True
    held_m = torch.zeros(n + 2, dtype=torch.bool)
    other_m = torch.zeros(n + 2, dtype=torch.bool)
    top = n                     # S84: no destination above the lowest engaged cap
    stage_cap = getattr(allocator, "_pdflip_kv_stage_cap", None)
    for cap in cap_filters(allocator):
        held = _id_tensor(getattr(cap, "_withheld", None))   # host tensor (KvRowCap keeps it there)
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
                      live_below=live_below, cap_top=int(top),
                      staged=frozenset(int(x) for x in torch.nonzero(staged).flatten().tolist()), **kw)

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
            hv = (snap.value(node) if ("node", id(node)) in snap._lay else None) if snap is not None \
                else _id_tensor(v)
            if hv is None:
                return _mk({}, [], why="node %s not in the snapshot" % getattr(node, "id", "?"))
            rows.append((node, hv.to(torch.int64)))
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
    wt_path, wt_own = _path_counts(cache, wt_starts)
    run_path, run_own = _path_counts(cache, [getattr(r, "last_node", None) for r in running])
    if rows:
        ids = torch.cat([v for _n, v in rows])
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
                            move_only=bool((locked or in_wt) and not pinned),
                            store=bool(store_ok and getattr(node, "l3_present", False)
                                       and cds[full].host_value is None and not in_wt))
            by_row[r] = np_
            nodes.append(np_)
        np_.pages.append(int(p))
        prev = owner.get(int(p))
        if prev is not None and prev is not np_:
            prev.pinned = np_.pinned = True  # a page two nodes name moves for neither
        owner[int(p)] = np_
    # the running requests' rows -- their own pages outside the tree move with them; a
    # tree page a running row names is never evicted, whatever its lock says
    run_pages = 0
    run_rows: Dict[int, frozenset] = {}
    if running and (req_to_token is not None or snap is not None):
        req_np = NodePages(node=None, pages=[], pinned=False, backed=False, move_only=True)
        seen = set()
        for r in running:
            if snap is not None:
                hrow = snap.row(r)
            else:
                m = min(_req_len(r), int(req_to_token.shape[1]))
                hrow = _id_tensor(req_to_token[int(r.req_pool_idx), :m]) if m > 0 else None
            if hrow is None or not hrow.numel():
                continue
            rp = [int(p) for p in torch.unique(hrow.to(torch.int64) // P).tolist()]
            run_rows[id(r)] = frozenset(rp)
            for p in rp:
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
    segs = segments(cache, nodes, P)
    return _mk(owner, nodes, run_reqs=len(running), run_pages=run_pages, segments=segs,
               run_rows=run_rows)


def _device_children(node: Any, gone=frozenset()) -> List[Any]:
    full = int(_full_ct().FULL)
    return [c for c in getattr(node, "children", {}).values()
            if c.component_data[full].value is not None and id(c) not in gone]


#: S112b (review A1): the vote's content map -- one slot per (segment key mod BITMAP);
#: the group MIN says "evictable on EVERY rank". A collision can only claim a segment
#: for a rank that does not hold it: that rank then picks otherwise and the content
#: digest refuses uniformly -- never a wrong eviction.
BITMAP = 1024


def local_bitmap(s: Survey) -> List[int]:
    """The content map of the segments L2-backed HERE (KVZIEL-1021: kind 1 -- what every rank
    can demote to its host copy in this pass)."""
    out = [0] * BITMAP
    for seg in s.segments:
        if seg.evictable and seg.kind == 1 and seg.key >= 0:
            out[seg.key % BITMAP] = 1
    return out


def local_secured_bitmap(s: Survey) -> List[int]:
    """KVZIEL-1021: the content map of the segments SECURED here -- L2-backed or store-backed
    (kind 1 or 2). The group MIN: a copy exists on every rank's side, so the content may leave
    every device (now, or through the S39 drop)."""
    out = [0] * BITMAP
    for seg in s.segments:
        if seg.evictable and seg.key >= 0:
            out[seg.key % BITMAP] = 1
    return out


def local_sizes(s: Survey) -> List[int]:
    """S112c (B4): per content slot this rank's segment size (pages; the smallest when two share a
    slot). The group MIN is the size the choice counts on every rank -- each rank frees at least that
    much (its own segment is never smaller), so the agreed choice closes every rank's gap."""
    out = [0] * BITMAP
    for seg in s.segments:
        if seg.evictable and seg.key >= 0:
            i = seg.key % BITMAP
            out[i] = seg.size if out[i] == 0 else min(out[i], seg.size)
    return out


@dataclass
class Agreement:
    """S112b: what the GROUP agreed in the vote -- the content map of segments evictable
    on every rank (MIN) and per candidate boundary page the MAX of the local gaps."""

    bitmap: Optional[Sequence[int]] = None      # None: this rank's own (single-rank form)
    gaps: Dict[int, int] = field(default_factory=dict)
    #: S112c (B4): per slot the group MIN of the segment sizes (None: the local size)
    sizes: Optional[Sequence[int]] = None
    #: KVZIEL-1021: per slot the group MIN of "secured here" (L2 or store); None = only the
    #: L2 map counts (a vote without it)
    secured: Optional[Sequence[int]] = None

    def agreed(self, seg: Segment) -> bool:
        """May the content leave every device (chosen for room)?"""
        if not seg.evictable or seg.key < 0:
            return False
        if self.secured is not None:
            return int(self.secured[seg.key % BITMAP]) >= 1
        if seg.kind != 1:
            return self.bitmap is None
        return True if self.bitmap is None else int(self.bitmap[seg.key % BITMAP]) >= 1

    def now(self, seg: Segment) -> bool:
        """KVZIEL-1021: does it leave by ``_evict_to_host`` in this pass -- L2-backed on EVERY
        rank (the L2 map's MIN)? Else the S39 drop takes it."""
        if not seg.evictable or seg.kind != 1 or seg.key < 0:
            return False
        return True if self.bitmap is None else int(self.bitmap[seg.key % BITMAP]) >= 1

    def size(self, seg: Segment) -> int:
        if self.sizes is None:
            return seg.size
        return max(0, int(self.sizes[seg.key % BITMAP]))


def local_gap(s: Survey, boundary_page: int) -> int:
    B = int(boundary_page)
    return (sum(1 for p in s.live_above if p > B)
            - sum(1 for p in s.free if p <= B))


def choose_evictions(s: Survey, ag: Agreement, gap: int) -> Tuple[Optional[List[Segment]], str]:
    """S112b: content segments that close ``gap`` (every evicted page closes it by one:
    a page above the boundary needs no room any more, a page at or below it IS room).
    Replicated inputs only (the agreed content map, the agreed gap, the segments'
    content keys and sizes): a segment is available when every child segment is chosen
    already (leaf first); best fit -- the smallest available one that closes the rest,
    else the largest; ties by the content key. Never a page id, never a clock."""
    chosen: List[Segment] = []
    done: set = set()
    rest = int(gap)
    while rest > 0:
        avail = [g for g in s.segments if id(g) not in done and ag.agreed(g)
                 and all(id(c) in done for c in g.children)]
        sz = {id(g): ag.size(g) for g in avail}
        avail = [g for g in avail if sz[id(g)] > 0]
        if not avail:
            if chosen:
                # S112c (B4): the agreed sizes are the group MIN -- a rank whose segment is longer
                # (Form A: TP0's leaf deeper) frees more; whether its room now suffices is its own
                # check in plan_for, and the plan check's MIN makes that verdict the group's
                return chosen, ""
            return None, "no evictable unit left (agreed backed idle content) for %d pages" % rest
        fits = [g for g in avail if sz[id(g)] >= rest]
        pick = (min(fits, key=lambda g: (sz[id(g)], g.key)) if fits
                else max(avail, key=lambda g: (sz[id(g)], -g.key)))
        chosen.append(pick)
        done.add(id(pick))
        rest -= sz[id(pick)]
    return chosen, ""


def split_evictions(segs: List[Segment], ag: Agreement) -> List[Tuple[Segment, int]]:
    """KVZIEL-1021: how each chosen segment leaves, from REPLICATED inputs (the chosen order,
    the agreed maps, the segment structure the digest covers): 1 = now (``_evict_to_host``,
    L2-backed on every rank, every chosen descendant goes now too), 2 = the S39 hand-off
    (store-backed somewhere; every chosen child goes now or is handed off before it -- the
    S39 queue drops in insertion order, so the child leaves first and the node is a device
    leaf at the flush), 3 = waits (an L2 segment above a hand-off: it is no device leaf
    before the drop -- the next pass takes it). Leaf first: ``choose_evictions`` chose every
    child before its parent."""
    state: Dict[int, int] = {}
    out: List[Tuple[Segment, int]] = []
    for g in segs:
        kids = [state.get(id(c), 3) for c in g.children]
        if ag.now(g):
            st = 1 if all(k == 1 for k in kids) else 3
        else:
            st = 2 if all(k in (1, 2) for k in kids) else 3
        state[id(g)] = st
        out.append((g, st))
    return out


def plan_for(s: Survey, boundary_page: int, ag: Optional[Agreement] = None) -> Tuple[Optional[Plan], str]:
    """Moves (rank-local) and room-making evictions (group-agreed CONTENT segments of
    backed idle device nodes) that leave no live page above ``boundary_page`` -- or None
    and why. All or nothing: a plan that cannot clear the boundary moves no page."""
    if s.why is not None:
        return None, s.why
    B = int(boundary_page)
    if B < int(s.floor_page):
        return None, "boundary %d below the surveyed floor page %d" % (B, s.floor_page)
    if ag is None:
        ag = Agreement(None, {B: local_gap(s, B)})
    held = [p for p in s.held_above if p > B]
    if held:
        return None, ("page %d (tokens from %d) withheld above the boundary by a cap filter other "
                      "than the stage cap (backing relief) -- the floor counts it, no move" % (
                          held[0], held[0] * s.page_size))
    above = [p for p in s.live_above if p > B]
    if not above:
        return Plan(boundary_page=B), ""
    for p in above:
        np_ = s.owner.get(p)
        if np_ is None:
            return None, "page %d (tokens from %d) held outside the tree (a running request / a session)" % (
                p, p * s.page_size)
        if np_.pinned:
            return None, "page %d pinned (node %s: lock / write-through / load-back)" % (
                p, getattr(np_.node, "id", "?"))
    free = [p for p in s.free if p <= B]
    gap = max(len(above) - len(free), int(ag.gaps.get(B, len(above) - len(free))))
    plan = Plan(boundary_page=B)
    ev_set: set = set()
    n_ev_nodes = 0
    if gap > 0:
        segs, why = choose_evictions(s, ag, gap)
        if segs is None:
            stay = [s.owner[p] for p in above if not (s.owner[p].backed or s.owner[p].store)
                    or s.owner[p].move_only]
            if stay:
                m = stay[-1]
                return None, ("no free page below %d for un-backed node %s (%d pages): %s" % (
                    B, getattr(m.node, "id", "req"), sum(1 for p in above if s.owner[p] is m), why))
            return None, "no free page below %d for %d pages above it: %s" % (B, len(above), why)
        now_pages: set = set()
        for g, st in split_evictions(segs, ag):
            plan.split.append((int(g.key), st))
            if st == 1:
                plan.evicts.extend(reversed(g.nodes))       # bottom-up: leaf first
                now_pages.update(g.pages)
            elif st == 2:
                plan.handoff.extend(reversed(g.nodes))
            ev_set.update(g.pages)
        plan.evict_keys = [g.key for g in segs]
        plan.evict_pages = sorted(now_pages)
        n_ev_nodes = sum(len(g.nodes) for g in segs)
    stay = [p for p in above if p not in ev_set]
    room = sorted(set(free) | {p for p in ev_set if p <= min(B, int(s.cap_top))})
    if len(room) < len(stay):
        return None, "no free page below %d for %d pages above it after %d evictions here" % (
            B, len(stay), n_ev_nodes)
    if plan.staged:
        # KVZIEL-1021: the room needs store-backed content the S39 drop takes at TP0's next
        # broadcast -- this pass evicts what can go now and queues the rest; NO move (the
        # destinations are planned once the drops landed on every rank)
        return plan, ""
    # PAGE-level, rank-local: the pages above the boundary ascending onto this rank's
    # room ascending; the nodes only group the re-pointing
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


def holder_census(s: Survey, boundary_page: int, ag: Optional[Agreement] = None) -> Dict[str, int]:
    """KVZIEL-1021 instrument (rank-local, log only): WHAT holds the live pages above
    ``boundary_page`` -- the refusal ``no free page below N for N pages above it after N
    evictions here`` (rc29 1021_022209 02:27:54, all of S0..S8) named no page and so no
    witness. Per page one class: the backing of its node (``l2`` / ``store`` / ``unbacked``,
    or ``move_only`` / ``pinned`` / ``running_rows`` / ``outside_tree``) and for a secured
    node whether its segment can go (``_ok``), waits behind a device child that cannot
    (``_child_blocked``), or is not secured on every rank (``_not_agreed``). Plus the free
    pages at or below the boundary."""
    B = int(boundary_page)
    ag = ag if ag is not None else Agreement(None)
    seg_of: Dict[int, Segment] = {}
    for g in s.segments:
        for p in g.pages:
            seg_of[int(p)] = g
    avail: Dict[int, bool] = {}

    def can_go(g: Segment) -> bool:
        k = id(g)
        if k not in avail:
            avail[k] = False            # a cycle cannot happen; fail closed meanwhile
            avail[k] = bool(ag.agreed(g) and all(can_go(c) for c in g.children))
        return avail[k]

    out: Dict[str, int] = {}

    def add(cls: str) -> None:
        out[cls] = out.get(cls, 0) + 1

    for p in s.live_above:
        if p <= B:
            continue
        m = s.owner.get(p)
        if m is None:
            add("outside_tree")
            continue
        if m.node is None:
            add("running_rows")
            continue
        if m.pinned:
            add("pinned")
            continue
        if m.move_only:
            add("move_only")
            continue
        base = "l2" if m.backed else ("store" if m.store else None)
        if base is None:
            add("unbacked")
            continue
        g = seg_of.get(int(p))
        if g is None or not g.evictable:
            add(base + "_partial")
        elif not ag.agreed(g):
            add(base + "_not_agreed")
        elif can_go(g):
            add(base + "_ok")
        else:
            add(base + "_child_blocked")
    out["free_below"] = sum(1 for p in s.free if p <= B)
    return out


def plan_digest(stage: int, plan: Plan) -> int:
    """N3: a 62-bit digest of what the GROUP must do alike -- the target stage, its
    boundary and the CONTENT of every eviction (S112b, review A1: after rank-local moves
    the same page id names other content on another rank, so evicted page ids prove
    nothing; the content keys do). The moves are rank-local and no part of it."""
    import hashlib

    h = hashlib.blake2b(digest_size=8)
    if plan.split and any(st != 1 for _k, st in plan.split):
        # KVZIEL-1021: a staged plan -- how each content leaves is part of what the group must do alike
        h.update(repr((int(stage), int(plan.boundary_page), sorted(int(k) for k in plan.evict_keys),
                       sorted((int(k), int(st)) for k, st in plan.split))).encode())
    else:
        h.update(repr((int(stage), int(plan.boundary_page), sorted(int(k) for k in plan.evict_keys))).encode())
    return int.from_bytes(h.digest(), "big") >> 2


# ---- S112b (review A2): the move, ordered by events, never a host wait -----------------------

def _take_counts(snap: Optional["Snapshot"], allocator: Any, pages: Sequence[int],
                 evicted: Sequence[int]) -> Dict[str, int]:
    """S112c (B1): how many of ``pages`` each free list holds -- known on the HOST: from the snapshot's
    host copy (exact, its sources unchanged, B2) plus the pages this pass's evictions handed back (they
    land in ``release_pages`` under ``need_sort``, else ``free_pages``). Without a snapshot (the CPU
    desk) from the live host lists."""
    want = {int(p) for p in pages}
    if snap is not None:
        lists = {"free_pages": set(int(x) for x in snap.free.tolist()),
                 "release_pages": set(int(x) for x in snap.release.tolist())}
    else:
        lists = {nm: set(int(x) for x in _id_tensor(getattr(allocator, nm, None)).tolist())
                 for nm in ("free_pages", "release_pages")}
    ev_list = "release_pages" if getattr(allocator, "need_sort", False) else "free_pages"
    lists[ev_list] |= {int(p) for p in evicted}
    return {nm: len(want & ids) for nm, ids in lists.items()}


def _take_pages(allocator: Any, pages: Sequence[int], counts: Optional[Dict[str, int]] = None) -> bool:
    """Take ``pages`` out of the free lists. S112c (B1): no boolean-mask index on the device (that is a
    ``nonzero``: the host waits for the current stream, which waits on the forward in flight). Each list
    is reordered by a STABLE argsort of its hit mask (the kept ids first, in their order) and cut at the
    length the host knows (``numel - counts[list]``, ``_take_counts``) -- an int-index gather, no host
    read. On host lists (the CPU desk) the hits are counted and checked: all of them, or none."""
    import torch

    out, found = {}, 0
    for name in ("free_pages", "release_pages"):
        t = getattr(allocator, name, None)
        if t is None or not t.numel():
            continue
        hit = torch.isin(t, _to_dev(pages, t.device))
        if t.device.type == "cuda":
            k = int((counts or {}).get(name, 0))
        else:
            k = int(hit.sum())
            if counts is not None and k != int(counts.get(name, 0)):
                return False
        found += k
        order = torch.argsort(hit.to(torch.int8), stable=True)
        out[name] = t.index_select(0, order[:int(t.numel()) - k])
    if found != len({int(p) for p in pages}):
        return False
    for name, t in out.items():
        setattr(allocator, name, t)
    return True


def _give_back(allocator: Any, pages: Sequence[int]) -> None:
    """Hand ``pages`` back, host-informed: the ids are known on the host, so the cap
    filters' share (ids above their cap: withheld, KvRowCap's ``_apply``) is booked on
    the host and the rest goes to the free list by a device cat -- no ``.any()`` /
    ``.cpu()`` read of the free lists (``kv_listener_refusal`` admits cap filters only)."""
    import torch

    keep = sorted({int(p) for p in pages})
    if not keep:
        return
    for cap in cap_filters(allocator):
        c = getattr(cap, "cap", None)
        if c is None:
            continue
        above = [p for p in keep if p > int(c)]
        if above:
            w = getattr(cap, "_withheld", None)
            add = torch.tensor(above, dtype=torch.int64)
            cap._withheld = add if w is None else torch.cat((w.to("cpu", torch.int64), add))
            page = max(1, int(getattr(allocator, "page_size", 1) or 1))
            try:
                allocator.residency_withheld_slots = int(cap._withheld.numel()) * page
            except Exception:  # noqa: BLE001
                pass
            keep = [p for p in keep if p <= int(c)]
    if not keep:
        return
    name = "release_pages" if getattr(allocator, "need_sort", False) else "free_pages"
    cur = getattr(allocator, name)
    setattr(allocator, name, torch.cat((_to_dev(keep, cur.device), cur)))
    touch = getattr(allocator, "_owner_placement_touch", None)
    if getattr(allocator, "_owner_placement", None) is not None and callable(touch):
        touch()


def _remap_tensor(t, P: int, lut):
    """Token ids with every moved page replaced (offset in page kept) -- on ``t``'s
    device, no host read."""
    import torch

    v = t.to(torch.int64)
    hi = int(lut.numel()) - 1
    pg = torch.div(v, P, rounding_mode="floor")
    ok = (v >= 0) & (pg <= hi)
    out = torch.where(ok, lut.to(v.device)[pg.clamp(0, hi)] * P + (v - pg * P), v)
    return out.to(t.dtype)


def lut_for(pairs: Dict[int, int], device):
    import torch

    hi = max(max(pairs), max(pairs.values()))
    lut = torch.arange(hi + 1, dtype=torch.int64)
    lut[torch.tensor(list(pairs.keys()), dtype=torch.int64)] = torch.tensor(list(pairs.values()), dtype=torch.int64)
    if getattr(device, "type", "cpu") == "cuda":
        return lut.pin_memory().to(device, non_blocking=True)
    return lut.to(device)


def execute_moves(cache: Any, allocator: Any, plan: Plan, copier: Any, *, remap_refs=None,
                  streams: Sequence[Any] = (), sync=None, check=None,
                  counts: Optional[Dict[str, int]] = None) -> Result:
    """The MOVES of the plan on this rank: claim the destinations, E0, copy, re-point the
    nodes and the request references, E1, free the sources. Raises CompactRefused (named)
    with everything rolled back.

    No host wait (S112b/c): the destinations are known free (the snapshot, valid by identity
    in this pass), the claim is a stable-argsort gather to a host-known length (B1); E0/E1
    are F3's ``RowFence`` over the KV streams (``kv_fence_begin``); the next forward waits on
    the schedule stream anyway (``run_batch``). The sources go back host-informed
    (``_give_back``). ``sync`` is accepted for old callers and ignored."""
    t0 = time.perf_counter()
    P = int(copier.page_size)
    full = _full_ct().FULL
    if not plan.moves:
        return Result(moved_nodes=0, moved_pages=0, evicted_nodes=0, evicted_tokens=0, bytes=0,
                      ms=(time.perf_counter() - t0) * 1000.0)
    if not getattr(allocator, "is_not_in_free_group", True):
        raise CompactRefused("inside a free group: freed pages are staged, not free -- no move (B8)")
    if check is not None and not check():
        raise CompactRefused("the survey snapshot no longer holds (an allocation or a tree "
                             "change since): nothing moved")
    src_all = [p for _m, s, _d in plan.moves for p in s]
    dst_all = [p for _m, _s, d in plan.moves for p in d]
    pairs = dict(zip(src_all, dst_all))
    saved = (getattr(allocator, "free_pages", None), getattr(allocator, "release_pages", None))
    if not _take_pages(allocator, dst_all, counts):
        raise CompactRefused("destination pages %s.. not all free" % dst_all[:8])
    fence = kv_fence_begin(streams)
    try:
        nbytes = copier.copy(src_all, dst_all)
    except Exception as exc:  # noqa: BLE001 -- nothing re-pointed yet: the lists go back
        allocator.free_pages, allocator.release_pages = saved
        raise CompactRefused("copy failed before any node moved: %s: %s"
                             % (type(exc).__name__, exc)) from exc
    done = []
    remapped = 0
    try:
        luts: Dict[Any, Any] = {}
        for m, src, dst in plan.moves:
            if m.node is None:
                continue       # a running request's own pages -- its row follows below
            cd = m.node.component_data[full]
            orig = cd.value
            d = orig.device
            if d not in luts:
                luts[d] = lut_for(pairs, d)
            cd.value = _remap_tensor(orig, P, luts[d]).reshape(orig.shape)
            done.append((cd, orig))
        if remap_refs is not None:
            remapped = int(remap_refs(pairs) or 0)
    except Exception as exc:  # noqa: BLE001 -- roll back, then refuse by name
        for cd, orig in reversed(done):
            cd.value = orig
        allocator.free_pages, allocator.release_pages = saved
        raise CompactRefused("repoint failed after %d of %d nodes (rolled back, destinations "
                             "freed): %s: %s" % (len(done), len(plan.moves), type(exc).__name__,
                                                 exc)) from exc
    fence.close()
    _give_back(allocator, src_all)
    return Result(moved_nodes=len(plan.moves), moved_pages=plan.moved_pages, evicted_nodes=0,
                  evicted_tokens=0, bytes=int(nbytes), ms=(time.perf_counter() - t0) * 1000.0,
                  remapped_rows=remapped)


def execute_evicts(cache: Any, plan: Plan) -> Tuple[int, int]:
    """The agreed EVICTIONS: backed device nodes, leaf first, through the tree's own
    ``_evict_to_host`` (the host copy stays). Returns (nodes, tokens)."""
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
            remap_refs=None, streams: Sequence[Any] = ()) -> Result:
    """Single-rank form (tests, no group): the agreed evictions first (``evict_first``),
    then the moves; the group path keeps a collective between the two. An eviction is
    never a loss: only backed nodes go, their host copy stays."""
    ev = execute_evicts(cache, plan) if plan.evict_first else (0, 0)
    res = execute_moves(cache, allocator, plan, copier, remap_refs=remap_refs, streams=streams)
    res.evicted_nodes, res.evicted_tokens = ev
    return res
