"""D-SEAT-REWAKE COMPACT: the tree's GDN states above a smaller seat count's
slot limit move DOWN, so an empty D can shrink to the fewest seats.

NF-Operator 30.09. (Produktentscheid): "KOMPAKTIEREN, nicht verdrängen. Die
GDN-Zustände des Baums, die über dem Limit liegen, ziehen per Device-Kopie in
freie Slots darunter um; die Slot-Referenzen im Baum werden dabei umgehängt.
Verdrängen nur als Rückfall: wenn unten kein Platz frei ist und der Zustand
schon im L2 gesichert ist (backuped). Nie einen ungesicherten Zustand
verlieren."

Befund y4x (16:33:26-36Z): D empty, n=6, the tree held GDN states in slots
above the limit of n=1 (7) -- the idle SHRINK (50bed49e3a) could only go as
far as the highest held slot allowed (``seats_covering``), or not at all.

The move is ``MambaPool.copy_from`` (the COW copy the tree already uses for
its checkpoints: conv + temporal + the slot siblings, the ReplaySSM cursor of
the destination reset -- a tree state is a flushed checkpoint), then the
node's value is re-pointed, then the source slot goes back to the allocator.
A state is moved only when nothing else reads its slot: no lock (a running
request, a session, a load-back pin), no write-through in flight (the D2H
copy reads the source), no load-back in flight (the H2D copy writes it).
Eviction takes only a state whose host copy has LANDED.

Replicated planning (the allocator's ledger and the tree are replicas on the
D ranks, the same assumption the slot limit rests on); the verdict goes
through the group MIN in ``d_seat_rewake.tick`` -- the ranks agree on the
target n before anything moves, and on the success of the move before the
limit comes down.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


class CompactRefused(RuntimeError):
    """The move could not be carried out on this rank (named)."""


@dataclass
class Anchor:
    """One GDN state the tree holds in the active mamba pool."""

    node: Any
    slot: int
    #: something reads or writes the slot (lock, write-through, load-back, odd shape)
    pinned: bool
    #: the host (L2) copy exists and has landed
    backed: bool


@dataclass
class Plan:
    limit: int
    moves: List[Tuple[Anchor, int]] = field(default_factory=list)
    evicts: List[Anchor] = field(default_factory=list)


@dataclass
class Result:
    moved: int
    evicted: int
    bytes: int
    ms: float


class TreeView:
    """The two tree lines that carry mamba states: ``UnifiedRadixCache`` (the
    MAMBA component's device value; the hierarchical line) and
    ``MambaRadixCache`` (``node.mamba_value``). ``why`` names a tree whose
    states do not live in the active allocator (none, int8 checkpoints)."""

    def __init__(self, cache: Any):
        self.cache = cache
        self.kind: Optional[str] = None
        self.why: Optional[str] = None
        if cache is None:
            self.why = "no_tree"
            return
        pool = getattr(cache, "req_to_token_pool", None)
        if getattr(pool, "mamba_ckpt_pool", None) is not None:
            self.why = "int8_ckpt (the tree's states live in the checkpoint pool)"
            return
        comps = getattr(cache, "components", None)
        if isinstance(comps, dict):
            from flliper.srt.mem_cache.unified_cache_components.tree_component import ComponentType

            if ComponentType.MAMBA in comps:
                self.kind, self.ct = "unified", ComponentType.MAMBA
                return
            self.why = "no_mamba_component"
            return
        if hasattr(cache, "root_node") and hasattr(cache, "_collect_all_nodes"):
            self.kind = "mamba"
            return
        self.why = "tree_kind_unknown"

    def _in_flight(self, node: Any) -> Tuple[bool, bool]:
        wt = getattr(self.cache, "ongoing_write_through", None) or {}
        lb = getattr(self.cache, "ongoing_load_back", None) or {}
        nid = getattr(node, "id", None)
        return nid in wt, nid in lb

    def _value(self, node: Any):
        if self.kind == "unified":
            data = getattr(node, "component_data", None)
            if data is None or len(data) <= int(self.ct):
                return None
            return data[self.ct].value
        return getattr(node, "mamba_value", None)

    def anchors(self) -> List[Anchor]:
        """Every device state of the tree, with its slot id (ONE D2H copy)."""
        if self.kind is None:
            return []
        root = getattr(self.cache, "root_node", None)
        rows = []
        for node in self.cache._collect_all_nodes():
            if node is root:
                continue
            v = self._value(node)
            if v is None:
                continue
            rows.append((node, v))
        if not rows:
            return []
        import torch

        flat = [v.reshape(-1) for _n, v in rows]
        counts = [int(f.numel()) for f in flat]
        try:  # ONE D2H copy for the whole tree (the values share the pool's device)
            ids = torch.cat(flat).tolist()
        except RuntimeError:  # mixed devices: one copy each
            ids = torch.cat([f.to("cpu") for f in flat]).tolist()
        out, at = [], 0
        for (node, _v), c in zip(rows, counts):
            slots = [int(x) for x in ids[at:at + c]]
            at += c
            wt, lb = self._in_flight(node)
            if self.kind == "unified":
                cd = node.component_data[self.ct]
                lock, host = int(cd.lock_ref), cd.host_value
            else:
                lock, host = int(getattr(node, "mamba_lock_ref", 0)), getattr(node, "mamba_host_value", None)
            pinned = lock > 0 or wt or lb or c != 1
            for s in slots:
                out.append(Anchor(node=node, slot=s, pinned=pinned, backed=host is not None and not wt))
        return out

    def repoint(self, node: Any, slot: int) -> None:
        """The node's value names ``slot`` now (same shape, dtype, device)."""
        v = self._value(node)
        nv = v.clone()
        nv.fill_(int(slot))
        if self.kind == "unified":
            node.component_data[self.ct].value = nv
        else:
            node.mamba_value = nv

    def set_value(self, node: Any, value: Any) -> None:
        """Put ``value`` (the node's original tensor) back -- the rollback of repoint."""
        if self.kind == "unified":
            node.component_data[self.ct].value = value
        else:
            node.mamba_value = value

    def evict_device(self, node: Any) -> int:
        """Tombstone the node's DEVICE state only (its KV and host copy stay);
        the slot goes back to the allocator. Returns the slots freed."""
        if self.kind != "unified":
            raise CompactRefused("evict on a %s tree (no host tier)" % self.kind)
        from flliper.srt.mem_cache.unified_cache_components.tree_component import EvictLayer

        comp = self.cache.components[self.ct]
        tracker = {c: 0 for c in self.cache.tree_components}
        self.cache._evict_component_and_detach_lru(node, comp, target=EvictLayer.DEVICE, tracker=tracker)
        self.cache._update_evictable_leaf_sets(node)
        return int(tracker.get(self.ct, 0))


def used_ids(slot_used) -> List[int]:
    idx = slot_used.nonzero()
    return [int(x) for x in idx.reshape(-1).tolist()]


def plan_for(used: Sequence[int], anchors: Sequence[Anchor], limit: int) -> Tuple[Optional[Plan], str]:
    """Moves (and, only as the fallback, evictions of states whose host copy
    landed) that leave no slot above ``limit`` in use -- or None and why.

    The un-backed states take the free slots first (they must never be lost),
    then the backed ones; a backed state left over is evicted. The highest
    slots move first, into the lowest free ones."""
    limit = int(limit)
    used_set = set(int(u) for u in used)
    by_slot: Dict[int, Anchor] = {}
    for a in anchors:
        by_slot.setdefault(int(a.slot), a)
    above = sorted((s for s in used_set if s > limit), reverse=True)
    movers = []
    for s in above:
        a = by_slot.get(s)
        if a is None:
            return None, "slot %d held outside the tree" % s
        if a.pinned:
            return None, "slot %d pinned (lock / write-through / load-back)" % s
        movers.append(a)
    free = [i for i in range(1, limit + 1) if i not in used_set]
    order = [a for a in movers if not a.backed] + [a for a in movers if a.backed]
    moves = list(zip(order, free))
    rest = order[len(moves):]
    lost = [a.slot for a in rest if not a.backed]
    if lost:
        return None, "no free slot below %d for un-backed state(s) %s" % (limit, lost)
    return Plan(limit=limit, moves=[(a, int(d)) for a, d in moves], evicts=list(rest)), ""


def slot_bytes(pool: Any) -> int:
    """Bytes one slot's move copies (conv + temporal, every layer)."""
    mc = getattr(pool, "mamba_cache", None)
    if mc is None:
        return 0
    total = 0
    for t in list(getattr(mc, "conv", None) or ()) + [getattr(mc, "temporal", None)]:
        if t is None or t.dim() < 2 or t.shape[1] == 0:
            continue
        total += int(t[:, 0].numel()) * int(t.element_size())
    return total


def _bytes_pool(cache: Any, value: Any):
    from flliper.srt.mem_cache.mamba_state_pool import active_mamba_state_pool, anchor_bytes_pool

    return anchor_bytes_pool(cache, value) or active_mamba_state_pool(cache)


def execute(cache: Any, allocator: Any, view: TreeView, plan: Plan) -> Result:
    """Carry the plan out on this rank: claim the destinations, copy (per
    bytes pool, #928), sync, re-point the nodes, free the sources; then the
    fallback evictions. Raises CompactRefused (named) before anything moved
    when a destination is not free."""
    import torch

    t0 = time.perf_counter()
    nbytes = 0
    if plan.moves:
        claim = getattr(allocator, "claim_free_slots", None)
        if claim is None:
            raise CompactRefused("allocator %s cannot claim given slots" % type(allocator).__name__)
        dev = allocator.free_slots.device
        dst_all = torch.tensor([d for _a, d in plan.moves], dtype=torch.int64, device=dev)
        if not claim(dst_all):
            raise CompactRefused("destinations %s not all free" % [d for _a, d in plan.moves])
        try:
            groups: Dict[int, Tuple[Any, List[int], List[int]]] = {}
            for a, d in plan.moves:
                pool = _bytes_pool(cache, view._value(a.node))
                g = groups.setdefault(id(pool), (pool, [], []))
                g[1].append(int(a.slot))
                g[2].append(int(d))
            for pool, src, dst in groups.values():
                pdev = pool.mamba_cache.temporal.device
                pool.copy_from(torch.tensor(src, dtype=torch.int64, device=pdev),
                               torch.tensor(dst, dtype=torch.int64, device=pdev))
                nbytes += slot_bytes(pool) * len(src)
                sync = getattr(pool, "_sync_device", None)
                if callable(sync):
                    sync()
        except Exception as exc:  # noqa: BLE001 -- nothing re-pointed yet: the destinations go back
            allocator.free(dst_all)
            raise CompactRefused("copy failed before any node moved: %s: %s" % (type(exc).__name__, exc)) from exc
        # TRANSACTIONAL (qwen review of 48345d52ca, NF-Operator 30.09.): a
        # repoint that raises mid-loop must not leave some nodes on their new
        # slot and every destination claimed. The nodes already re-pointed go
        # back to their ORIGINAL value (the same tensor object), the #928
        # ledger entries go back, all destinations are freed -- then refused.
        # The sources are freed only after every node moved.
        ledger = getattr(cache, "_mamba_anchor_pool", None)
        done = []  # (node, original value, ledger move (src, dst) or None)
        try:
            for a, d in plan.moves:
                orig = view._value(a.node)
                view.repoint(a.node, d)
                moved_ledger = None
                if isinstance(ledger, dict) and int(a.slot) in ledger:
                    ledger[int(d)] = ledger.pop(int(a.slot))
                    moved_ledger = (int(a.slot), int(d))
                done.append((a.node, orig, moved_ledger))
        except Exception as exc:  # noqa: BLE001 -- roll back, then refuse by name
            for node, orig, mv in reversed(done):
                view.set_value(node, orig)
                if mv is not None and isinstance(ledger, dict) and mv[1] in ledger:
                    ledger[mv[0]] = ledger.pop(mv[1])
            allocator.free(dst_all)
            raise CompactRefused("repoint failed after %d of %d nodes (rolled back, destinations freed): "
                                 "%s: %s" % (len(done), len(plan.moves), type(exc).__name__, exc)) from exc
        allocator.free(torch.tensor([int(a.slot) for a, _d in plan.moves], dtype=torch.int64, device=dev))
    evicted = 0
    for a in plan.evicts:
        evicted += view.evict_device(a.node)
    return Result(moved=len(plan.moves), evicted=evicted, bytes=nbytes,
                  ms=(time.perf_counter() - t0) * 1000.0)


@dataclass
class Survey:
    """One rank's answer for every seat count ``lo..n-1``: reachable (with or
    without a move) and whether it needs one."""

    lo: int
    n: int
    fit: int
    plans: Dict[int, Optional[Plan]]
    why: Dict[int, str]

    def reach(self, m: int) -> bool:
        return m >= self.fit or self.plans.get(m) is not None

    def no_move(self, m: int) -> bool:
        return m >= self.fit

    def flags(self) -> List[bool]:
        ms = range(self.lo, self.n)
        return [self.reach(m) for m in ms] + [self.no_move(m) for m in ms]


def survey(cache: Any, allocator: Any, size: int, cap: int, lo: int, n: int, fit: int,
           limit_of) -> Tuple[Survey, Optional[TreeView]]:
    """Plans for every m in lo..fit-1 (m >= fit needs no move)."""
    plans: Dict[int, Optional[Plan]] = {}
    why: Dict[int, str] = {}
    view = TreeView(cache)
    if int(fit) > int(lo):
        if view.kind is None:
            for m in range(lo, min(fit, n)):
                plans[m], why[m] = None, view.why or "no_tree"
        else:
            used = used_ids(allocator.slot_used)
            anchors = view.anchors()
            for m in range(lo, min(fit, n)):
                plans[m], why[m] = plan_for(used, anchors, limit_of(size, m, cap))
    return Survey(lo=int(lo), n=int(n), fit=int(fit), plans=plans, why=why), view


def agreed_target(s: Survey, mins: Sequence[int]) -> Tuple[Optional[int], bool]:
    """From the group MIN of ``Survey.flags()``: the fewest seats every rank
    reaches, and whether some rank must move states for it."""
    k = s.n - s.lo
    reach, nomove = list(mins[:k]), list(mins[k:2 * k])
    for i, m in enumerate(range(s.lo, s.n)):
        if reach[i]:
            return m, not bool(nomove[i])
    return None, False
