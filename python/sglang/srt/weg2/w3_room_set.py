"""F1 (NF h16s 1f4b3e73d4, boot 1016_103428, pgap root nf-pgap-wurzel-1016): ONE slot decision for
the W3 arena spill of a PP group, applied alike on every stage.

THE GAP. On P (PP3, local trees, NOT replicas) the KV arena was 6485/6485 pinned by the three
stages' own host copies; a slot is ONE page for all three stages and frees only when every stage
let go. ``_w3_arena_spill`` / ``host_only_spill`` chose their candidates PER RANK (group-equal order
by reference count, but node boundaries differ per stage): of 214-227 pages released per spill only
13-15 slots became free (``ref_free``), the fills held their landed pages and ended SHORT, the
requests were recomputed (36 % P idle in the pgap census).

THE RULE.
* PP0 decides: from its own candidates (device-resident nodes with a host copy, host-only leaves;
  group-equal order) it takes nodes until ``want`` pages (at most the per-call page cap) and
  publishes their ARENA SLOTS in ``<arena>.w3set`` (magic, PP0's nonce, generation, slots). No
  carry-over of an earlier set (review F1 A1: ``refs > 0`` cannot tell a laggard from a page that was
  freed and adopted again -- the set grew to the whole arena and evicted fresh pages again).
* Every stage (PP0 too) applies a set ONCE: each of its nodes whose host rows name a slot of the set
  gets its L3 copy (``secure_rows_to_l3``: COMPLETE slots only) and gives its host copy back -- a
  device-resident node keeps its device value and is marked ``l3_present``, a host-only leaf leaves
  the tree (``_evict_host_leaf``), its parent next when it names a slot of the set. A node of the set
  the stage cannot release now (a running request, a host lock, the page cap, no L3 copy yet) goes on
  the stage's own DEBT LIST -- by node identity, with the slots it named -- and is retried at every
  round until released or gone (a page adopted again later is a new node: never a debt).
* Followers apply a new set at every READ-ROOM round (``tick``: one stat per 20 ms), at every arena
  claim and at their own spill, and acknowledge it in ``<arena>.w3set.ack<pp_rank>``. PP0 decides no
  new set while a follower has not acknowledged its last one (review A4): its spill is then
  UNDECIDED (no empty backoff; ``undecided(cache)`` for the room verdict).
* A follower may also release its own candidates whose slots no other process references (refs <= 1:
  its own reference, the K15c rule; review A3) -- a slot only a follower holds would otherwise never
  free. A slot another rank references is never a follower's choice.
* Never a candidate: a node locked by a running request (device ``lock_ref``), a host-locked node
  (prefetch span pins #1417 hold a read's span until admission -- the told window), a node with a
  pending write, a node with a slot not COMPLETE (L3-COMPLETE A1), a host-only aux state, an aux
  DEVICE value on a host-only leaf (K15b A4), a node the W3-KEEP-PAYABLE filter keeps, and the
  claimer's own chain (review A2: the old device round's protections).
* A set file from an earlier boot is never applied (review A5): it is valid only when written after
  this process started; the temp name carries pid and thread id.
* No collective, no timer, no reserve: the decision is a file next to the shared arena, the apply is
  rank-local and idempotent. Form A (R12) keeps its own TP0-decides protocol (SPILL events).

Switch ``SGLANG_WEG2_W3_ROOM_SET`` (default on; 0 = the per-rank choice as before). Page cap per call
``SGLANG_WEG2_W3_ROOM_SET_MAX_SECURE_PAGES`` (default 128; the L3 write is synchronous on the
scheduler thread)."""
from __future__ import annotations

import logging
import os
import struct
import threading
import time
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

SUFFIX = ".w3set"
ACK = ".ack"
ENV = "SGLANG_WEG2_W3_ROOM_SET"
CAP_ENV = "SGLANG_WEG2_W3_ROOM_SET_MAX_SECURE_PAGES"
_MAGIC = 0x57335253_00000002            # "W3RS" v2
_TICK_S = 0.02
_N = {"publish": 0, "apply": 0, "own": 0, "wait": 0}


def _proc_start_ns() -> int:
    try:
        import psutil

        return int(psutil.Process().create_time() * 1e9)
    except Exception:  # noqa: BLE001 - no psutil: the import time of this module
        return time.time_ns()


_PROC_T0_NS = _proc_start_ns()


def on(env=None) -> bool:
    return (os.environ if env is None else env).get(ENV, "1").strip().lower() not in (
        "0", "false", "no", "off", "")


def page_cap(env=None) -> Optional[int]:
    """Pages per call; None = no cap (env 0). Review F1b B5: 'used up' (0 left) and 'off' are apart."""
    try:
        v = int((os.environ if env is None else env).get(CAP_ENV, "128"))
    except ValueError:
        v = 128
    return None if v <= 0 else v


#: review F1b B3: PP0 waits for a follower's acknowledgement at most this long, then decides anew (named)
WAIT_TIMEOUT_S = 2.0


def armed(cache: Any) -> bool:
    """A PP group (pp_size > 1) outside Form A."""
    if not on():
        return False
    try:
        from sglang.srt.mem_cache import form_a_host_shadow as _r12

        if _r12.role() is not None:
            return False
    except Exception:  # noqa: BLE001
        pass
    return int(getattr(cache, "pp_size", 1) or 1) > 1


def is_decider(cache: Any) -> bool:
    return int(getattr(cache, "pp_rank", 0) or 0) == 0


def _path(arena: Any) -> Optional[str]:
    p = getattr(arena, "path", None)
    return (str(p) + SUFFIX) if p else None


def _write_atomic(path: str, raw: bytes) -> None:
    tmp = "%s.tmp%d.%d" % (path, os.getpid(), threading.get_ident())
    with open(tmp, "wb") as f:
        f.write(raw)
    os.replace(tmp, path)


def read_full(arena: Any) -> Tuple[int, int, List[int]]:
    """``(nonce, gen, slots)`` of the current set; ``(0, 0, [])`` when there is none, it is of an older
    format, or it was written before this process started (an earlier boot's file, review A5)."""
    path = _path(arena)
    if path is None:
        return 0, 0, []
    try:
        st = os.stat(path)
        if st.st_mtime_ns < _PROC_T0_NS:
            return 0, 0, []
        with open(path, "rb") as f:
            raw = f.read()
    except OSError:
        return 0, 0, []
    if len(raw) < 32:
        return 0, 0, []
    magic, nonce, gen, n = struct.unpack("<QQqq", raw[:32])
    if magic != _MAGIC or n < 0 or len(raw) < 32 + 8 * n:
        return 0, 0, []
    return int(nonce), int(gen), list(struct.unpack("<%dq" % n, raw[32:32 + 8 * n]))


def read(arena: Any) -> Tuple[int, List[int]]:
    _nonce, gen, slots = read_full(arena)
    return gen, slots


def _nonce(cache: Any) -> int:
    n = cache.__dict__.get("_w3_room_nonce")
    if not n:
        n = int.from_bytes(os.urandom(8), "little") or 1
        cache._w3_room_nonce = n
    return n


def publish(arena: Any, slots: Iterable[int], cache: Any = None) -> int:
    """PP0: the new slot set -- exactly ``slots``, no carry-over (review A1). Returns the gen."""
    path = _path(arena)
    if path is None:
        return 0
    _n0, gen, _prev = read_full(arena)
    gen = max(gen, int(getattr(cache, "_w3_room_pub_gen", 0) or 0)) + 1
    out = sorted({int(s) for s in slots})
    raw = struct.pack("<QQqq", _MAGIC, _nonce(cache) if cache is not None else 1, gen, len(out))
    if out:
        raw += struct.pack("<%dq" % len(out), *out)
    _write_atomic(path, raw)
    if cache is not None:
        cache._w3_room_pub_gen = gen
    return gen


def _ack_path(arena: Any, rank: int) -> Optional[str]:
    p = _path(arena)
    return None if p is None else "%s%s%d" % (p, ACK, int(rank))


def ack(cache: Any, arena: Any, nonce: int, gen: int) -> None:
    """This rank applied set (nonce, gen); the file also names the slots it still OWES (its debts), so a
    peer can tell a re-adopted page from a laggard's (review F1b B6)."""
    path = _ack_path(arena, int(getattr(cache, "pp_rank", 0) or 0))
    if path is None:
        return
    owed = sorted({int(x) for e in _debts(cache).values() for x in e[1]})
    try:
        raw = struct.pack("<QQqq", _MAGIC, int(nonce), int(gen), len(owed))
        if owed:
            raw += struct.pack("<%dq" % len(owed), *owed)
        _write_atomic(path, raw)
        cache._w3_room_acked = (int(nonce), int(gen))
    except OSError:
        logger.warning("W3-ROOM-SET ack write failed", exc_info=True)


def _read_ack(arena: Any, rank: int) -> Optional[Tuple[int, int, Set[int]]]:
    path = _ack_path(arena, rank)
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except (OSError, TypeError):
        return None
    if len(raw) < 32:
        return None
    magic, n, g, k = struct.unpack("<QQqq", raw[:32])
    if magic != _MAGIC or k < 0 or len(raw) < 32 + 8 * k:
        return None
    return int(n), int(g), set(struct.unpack("<%dq" % k, raw[32:32 + 8 * k]))


def missing_acks(cache: Any, arena: Any) -> List[int]:
    """PP0: the follower ranks that have not acknowledged PP0's last published set (nonce, generation)."""
    gen = int(getattr(cache, "_w3_room_pub_gen", 0) or 0)
    if gen <= 0:
        return []
    nonce = _nonce(cache)
    out = []
    for r in range(1, int(getattr(cache, "pp_size", 1) or 1)):
        a = _read_ack(arena, r)
        if a is None or a[0] != nonce or a[1] < gen:
            out.append(r)
    return out


def acked_all(cache: Any, arena: Any) -> bool:
    return not missing_acks(cache, arena)


def node_slots(pool: Any, node: Any) -> List[int]:
    from sglang.srt.mem_cache.pool_host.arena_pool import _arena_mask, _psz
    from sglang.srt.mem_cache.unified_radix_cache import BASE_COMPONENT_TYPE

    hv = node.component_data[BASE_COMPONENT_TYPE].host_value
    if hv is None or hv.numel() == 0:
        return []
    import torch

    rows = hv.reshape(-1).cpu().to(torch.int64)
    keep = _arena_mask(pool, rows)
    return sorted({int(x) for x in ((rows[keep] - int(pool.staging_rows)) // _psz(pool)).tolist()})


_UNSET = object()


class _Ctx:
    """One call's view: the claimer's chain, the keep-payable walk, the told heads, the page cap, nodes
    already tried. The keep-payable walk (a whole-tree pass) and the told heads are computed on first
    use only (review F1b B2: an idle round builds nothing)."""

    def __init__(self, cache: Any, pool: Any, claimer: Any = None, cap: Any = _UNSET):
        self.cache, self.pool = cache, pool
        self.skip: Set[int] = set()
        n = claimer
        while n is not None and n is not cache.root_node:
            self.skip.add(id(n))      # the claimer's own chain: its parent rule reads backed/l3_present
            n = n.parent
        self._kp: Any = _UNSET
        self._told: Any = _UNSET
        self.cap_left: Optional[int] = page_cap() if cap is _UNSET else cap   # None = no cap (B5)
        self.tried: Set[int] = set()
        self.capped = 0
        self.why: Dict[int, str] = {}     # node id -> why it stayed (B4: transient or standing)

    @property
    def keep_payable(self):
        if self._kp is _UNSET:
            from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

            try:
                self._kp = UnifiedRadixCache._w3_keep_payable_state(self.cache)
            except Exception:  # noqa: BLE001 - unreadable walk: no filter (the old spill's behaviour)
                self._kp = None
        return self._kp

    @property
    def told(self) -> Set[int]:
        if self._told is _UNSET:
            from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

            try:
                self._told = UnifiedRadixCache._told_head_ids(self.cache)   # A7 TOLD-HEAD
            except Exception:  # noqa: BLE001 - a hand-built tree without the pins
                self._told = set()
        return self._told

    def payable(self, node: Any) -> bool:
        kp = self.keep_payable
        if kp is None:
            return True
        return all(kp.get(id(c), (0, False, False))[2] for c in node.children.values())


#: why a node of a set stays: TRANSIENT blockers clear by themselves within a round or two (the room
#: decision is still open -> UNDECIDED); STANDING ones (a running request, a read's told host lock,
#: keep-payable, the claimer's chain, a told head, an aux host life) do not (review F1b B4)
_TRANSIENT = frozenset({"write_pending", "capped", "not_secured", "error", "tried"})


def _stay_reason(cache: Any, node: Any, pool: Any, ctx: _Ctx) -> str:
    from sglang.srt.mem_cache.unified_radix_cache import BASE_COMPONENT_TYPE

    if node.id in cache.ongoing_write_through:
        return "write_pending"
    cd = node.component_data[BASE_COMPONENT_TYPE]
    if int(getattr(cd, "lock_ref", 0) or 0) > 0:
        return "running"
    if any(int(getattr(c, "host_lock_ref", 0) or 0) > 0 for c in node.component_data):
        return "host_locked"
    return "standing"


def _device_candidate(cache: Any, node: Any, pool: Any, ctx: Optional[_Ctx] = None) -> bool:
    from sglang.srt.mem_cache.unified_radix_cache import BASE_COMPONENT_TYPE, _aux_components

    if node is cache.root_node:
        return False
    if ctx is not None and (id(node) in ctx.skip or id(node) in ctx.told):
        return False                  # the claimer's chain / the head above an open told's span (A7)
    cd = node.component_data[BASE_COMPONENT_TYPE]
    hv = cd.host_value
    if hv is None or cd.value is None or hv.numel() == 0:
        return False
    if int(getattr(cd, "lock_ref", 0) or 0) > 0:
        return False                  # a running request holds it
    if node.id in cache.ongoing_write_through:
        return False
    if any(int(getattr(c, "host_lock_ref", 0) or 0) > 0 for c in node.component_data):
        return False                  # a load / a read's told window (#1417 span pin)
    if any(node.component_data[c.component_type].host_value is not None
           and node.component_data[c.component_type].value is None for c in _aux_components(cache)):
        return False
    if int(hv.min()) < int(getattr(pool, "staging_rows", 0)):
        return False
    if ctx is not None and node.children and not ctx.payable(node):
        return False                  # W3-KEEP-PAYABLE (ARENAFULL-1015): its drop could not clear a child
    return True


def _host_leaf_candidate(cache: Any, node: Any, pool: Any, ctx: Optional[_Ctx] = None) -> bool:
    from sglang.srt.weg2 import host_only_spill as _hos

    return _hos._eligible(cache, node, pool, ctx.skip if ctx is not None else set())


def _candidates(cache: Any, pool: Any, ctx: _Ctx) -> List[Any]:
    nodes = [n for n in cache._collect_all_nodes()
             if _device_candidate(cache, n, pool, ctx) or _host_leaf_candidate(cache, n, pool, ctx)]
    keys = cache._w3_spill_keys(pool, nodes)
    inc = getattr(keys, "incomplete", None)
    if callable(inc):
        nodes = [n for n in nodes if not inc(n)]   # L3-COMPLETE A1: no stage could secure it
    nodes.sort(key=lambda x: keys[id(x)])
    return nodes


def decide(cache: Any, pool: Any, want: int, claimer: Any = None, ctx: Optional[_Ctx] = None) -> List[int]:
    """PP0: the slots of its candidates in group-equal order until ``want`` pages (at most the cap)."""
    ctx = ctx or _Ctx(cache, pool, claimer)
    want = int(want) if ctx.cap_left is None else min(int(want), ctx.cap_left)
    chosen: Set[int] = set()
    for n in _candidates(cache, pool, ctx):
        if len(chosen) >= want:
            break
        chosen.update(node_slots(pool, n))
    return sorted(chosen)


def _debts(cache: Any) -> Dict[int, Tuple[Any, frozenset]]:
    d = cache.__dict__.get("_w3_room_debt")
    if d is None:
        d = cache._w3_room_debt = {}
    return d


def _attached(cache: Any, node: Any) -> bool:
    """Reachable from the CURRENT root (review F1b B1: a reset tree's subtree still has its parent links)."""
    n, hops = node, 0
    while n is not None and hops < 100000:
        p = getattr(n, "parent", None)
        if p is None:
            return n is cache.root_node
        if not any(c is n for c in p.children.values()):
            return False
        n, hops = p, hops + 1
    return False


def forget(cache: Any) -> None:
    """The tree is torn down (reset / flush): its debts name dead nodes whose arena references the reset
    already gave back -- paying them would free a reference twice (review F1b B1)."""
    cache.__dict__.pop("_w3_room_debt", None)
    cache.__dict__.pop("_w3_room_undecided", None)


def _readopted(cache: Any, pool: Any, slots, peers: Optional[Dict[int, Optional[Tuple[int, int, Set[int]]]]]) -> bool:
    """Review F1b B6: a debt slot holds MORE references than this rank plus the peers that may still hold
    it (a peer that has not acknowledged the current set, or names the slot among its debts) -- the page
    was adopted again since: paying the debt gains no room."""
    arena = getattr(pool, "arena", None)
    reader = getattr(arena, "slot_refs", None)
    if not callable(reader) or peers is None:
        return False
    nonce, gen, _s = read_full(arena)
    try:
        refs = [int(r) for r in reader(sorted(slots))]
    except Exception:  # noqa: BLE001
        return False
    for slot, r in zip(sorted(slots), refs):
        explain = 1
        for a in peers.values():
            if a is None or (a[0], a[1]) != (nonce, gen) or slot in a[2]:
                explain += 1
        if r > explain:
            return True
    return False


def _peer_acks(cache: Any, pool: Any) -> Optional[Dict[int, Optional[Tuple[int, int, Set[int]]]]]:
    arena = getattr(pool, "arena", None)
    if arena is None or _path(arena) is None:
        return None
    me = int(getattr(cache, "pp_rank", 0) or 0)
    return {r: _read_ack(arena, r) for r in range(int(getattr(cache, "pp_size", 1) or 1)) if r != me}


def apply(cache: Any, pool: Any, slot_set: Set[int], released_out: Optional[list] = None,
          claimer: Any = None, ctx: Optional[_Ctx] = None) -> Dict[str, int]:
    """Every stage: give back the host copy of each own node naming a slot of ``slot_set`` (one pass
    over the tree, parents of released leaves next) plus the stage's debts (by node identity)."""
    out = {"released": 0, "nodes": 0, "kept": 0, "capped": 0, "debts": 0, "transient": 0}
    ctx = ctx or _Ctx(cache, pool, claimer)
    debts = _debts(cache)
    work: List[Tuple[Any, frozenset]] = []
    peers = _peer_acks(cache, pool) if debts else None
    for key, entry in list(debts.items()):
        node, slots = entry[0], entry[1]
        live = set(node_slots(pool, node)) & slots if _attached(cache, node) else set()
        if not live:
            debts.pop(key, None)      # released elsewhere, evicted, rewritten or a dead tree's: no debt
            continue
        if _readopted(cache, pool, live, peers):
            debts.pop(key, None)      # B6: the page was adopted again since -- paying gains no room
            continue
        work.append((node, frozenset(live)))
    if slot_set:
        for node in cache._collect_all_nodes():
            if node is cache.root_node or id(node) in debts:
                continue
            rel = slot_set.intersection(node_slots(pool, node))
            if rel:
                work.append((node, frozenset(rel)))
    i = 0
    while i < len(work):
        node, rel = work[i]
        i += 1
        try:
            r = _release_one(cache, pool, node, ctx, out)
        except Exception:  # noqa: BLE001 - review F1b B3: one node's failure is its debt, not the set's
            ctx.why[id(node)] = "error"
            r = None
            _N["error"] = _N.get("error", 0) + 1
            if _N["error"] <= 8 or _N["error"] % 256 == 0:
                logger.warning("W3-ROOM-SET apply: a node's release raised (n=%d) -- kept as a debt",
                               _N["error"], exc_info=True)
        if r is None:
            why = ctx.why.get(id(node)) or _stay_reason(cache, node, pool, ctx)
            debts[id(node)] = (node, rel, None, why)
            out["kept"] += 1
            if why in _TRANSIENT:
                out["transient"] += 1
            continue
        debts.pop(id(node), None)
        out["released"] += r[0]
        out["nodes"] += 1
        if released_out is not None:
            released_out.extend(sorted(rel))
        parent = r[1]
        if parent is not None and parent is not cache.root_node and slot_set:
            prel = slot_set.intersection(node_slots(pool, parent))
            if prel:
                work.append((parent, frozenset(prel)))
    out["debts"] = len(debts)
    owed = frozenset(int(x) for e in debts.values() for x in e[1])
    if owed != cache.__dict__.get("_w3_room_owed", frozenset()):
        cache._w3_room_owed = owed
        acked = cache.__dict__.get("_w3_room_acked")
        if acked is not None:          # keep the peers' view of this rank's debts current (B6)
            ack(cache, getattr(pool, "arena", None), acked[0], acked[1])
    return out


def _release_one(cache: Any, pool: Any, node: Any, ctx: _Ctx, out: Dict[str, int]):
    """``(pages released, parent of a released host leaf or None)``; None = kept (a debt)."""
    from sglang.srt.mem_cache.unified_radix_cache import BASE_COMPONENT_TYPE, EvictLayer

    if id(node) in ctx.tried:
        ctx.why[id(node)] = "tried"
        return None                   # review A6: one secure per node and call
    dev = _device_candidate(cache, node, pool, ctx)
    leaf = (not dev) and _host_leaf_candidate(cache, node, pool, ctx)
    if not (dev or leaf):
        return None
    slots = node_slots(pool, node)
    if ctx.cap_left is not None and len(slots) > ctx.cap_left:
        ctx.capped += 1
        out["capped"] += 1
        ctx.why[id(node)] = "capped"
        return None
    ctx.tried.add(id(node))
    if ctx.cap_left is not None:
        ctx.cap_left -= len(slots)
    hv = node.component_data[BASE_COMPONENT_TYPE].host_value
    sec = pool.secure_rows_to_l3(hv)
    if int(sec.get("lost", 0)) or int(sec.get("unpaired", 0)) or int(sec.get("pages", 0)) < len(slots):
        ctx.why[id(node)] = "not_secured"
        return None                   # no complete L3 copy for every page (A1): the node stays
    P = max(1, int(getattr(cache, "page_size", 1) or 1))
    if dev:
        base = cache.components.get(BASE_COMPONENT_TYPE)
        _d, host_freed = cache._evict_component_and_detach_lru(node, base, target=EvictLayer.HOST, tracker=None)
        cache.evictable_host_leaves.discard(node)
        node.l3_present = True
        return int(host_freed or 0) // P, None
    parent = node.parent
    tracker = {ct: 0 for ct in cache.tree_components}
    cache._evict_host_leaf(node, tracker)
    return int(tracker.get(BASE_COMPONENT_TYPE, 0)) // P, parent


def _own_only(cache: Any, pool: Any, need: int, ctx: _Ctx, released_out: Optional[list]) -> Dict[str, int]:
    """Review A3: a follower releases its own candidates whose slots no other process references."""
    out = {"released": 0, "nodes": 0, "kept": 0, "capped": 0}
    reader = getattr(getattr(pool, "arena", None), "slot_refs", None)
    if not callable(reader) or need <= 0:
        return out
    for node in _candidates(cache, pool, ctx):
        if out["released"] >= need:
            break
        slots = node_slots(pool, node)
        try:
            if not slots or any(int(r) != 1 for r in reader(slots)):
                continue              # B6: exactly its own reference; another rank's slot is never its choice
        except Exception:  # noqa: BLE001 - unreadable = not provably own
            continue
        r = _release_one(cache, pool, node, ctx, out)
        if r is None:
            continue
        out["released"] += r[0]
        out["nodes"] += 1
        if released_out is not None:
            released_out.extend(slots)
    return out


def _resolve(pool: Any) -> Any:
    if pool is not None and not hasattr(pool, "secure_rows_to_l3"):
        from sglang.srt.mem_cache.unified_radix_cache import _w3_anchor_spill_pool

        pool = _w3_anchor_spill_pool(pool)
    return pool


def _apply_new(cache: Any, pool: Any, arena: Any, ctx: _Ctx, released_out: Optional[list]) -> Optional[Dict[str, int]]:
    """A follower: apply a set it has not applied yet (and acknowledge it), plus its debts."""
    nonce, gen, slots = read_full(arena)
    seen = cache.__dict__.get("_w3_room_seen", (0, 0))
    fresh = gen > 0 and (nonce, gen) != seen and (nonce != seen[0] or gen > seen[1])
    if not fresh and not _debts(cache):
        return None
    r = {"released": 0, "nodes": 0, "kept": 0, "transient": 0}
    try:
        r = apply(cache, pool, set(slots) if fresh else set(), released_out, ctx=ctx)
    finally:
        if fresh:
            # review F1b B3: acknowledged even when the apply raised -- what stayed is the debt list's
            cache._w3_room_seen = (nonce, gen)
            cache._w3_room_gen = gen
            ack(cache, arena, nonce, gen)
    if fresh:
        _log("apply", cache, gen, len(slots), r)
    return r


def apply_if_new(cache: Any, pool: Any, claimer: Any = None, debts: bool = True) -> Optional[Dict[str, int]]:
    """At a READ-ROOM round (``debts``) / an arena claim (new sets only): a follower applies a new set;
    every stage retries its debts. None = nothing to do."""
    if not armed(cache):
        return None
    pool = _resolve(pool)
    arena = getattr(pool, "arena", None)
    if arena is None:
        return None
    if not debts:
        nonce, gen, _s = read_full(arena)
        if is_decider(cache) or gen <= 0 or (nonce, gen) == cache.__dict__.get("_w3_room_seen", (0, 0)):
            return None
    ctx = _Ctx(cache, pool, claimer)          # lazy: builds no walk unless a node is examined (B2)
    if is_decider(cache):
        return apply(cache, pool, set(), ctx=ctx) if _debts(cache) else None
    return _apply_new(cache, pool, arena, ctx, None)


def tick(cache: Any) -> None:
    """Once per scheduler round (READ-ROOM ``tick``), at most every 20 ms: ``apply_if_new``. Never raises."""
    try:
        if not armed(cache):
            return
        t = time.monotonic()
        if t - float(cache.__dict__.get("_w3_room_tick_t", 0.0)) < _TICK_S:
            return
        cache._w3_room_tick_t = t
        apply_if_new(cache, cache._weg2_direct_pool())
    except Exception:  # noqa: BLE001 - a room round never breaks the scheduler round
        _N["tick_err"] = _N.get("tick_err", 0) + 1
        if _N["tick_err"] <= 8 or _N["tick_err"] % 1024 == 0:     # B3: throttled
            logger.warning("W3-ROOM-SET tick raised (n=%d)", _N["tick_err"], exc_info=True)


def backoff_ok(cache: Any) -> bool:
    """May an empty spill set the empty backoff? Only PP0's own decided 'nothing' (review F1b B6: a
    follower serving before PP0's first / next set gives back nothing and must stay ready to apply)."""
    return is_decider(cache) and not undecided(cache)


def undecided(cache: Any) -> bool:
    """The last spill of this rank released nothing BECAUSE the group decision is still open (PP0 waits
    for a follower's acknowledgement, a follower still holds debts of PP0's set) -- not a verdict
    'nothing to release' (review A4: the room verdict must not end a wait on it)."""
    return bool(cache.__dict__.get("_w3_room_undecided", False))


def spill(cache: Any, pool: Any, need_pages: int, want: int, released_out: Optional[list] = None,
          claimer: Any = None) -> int:
    """The W3 spill of a PP group: PP0 decides and applies, a follower applies the current set (then its
    own-only slots). Sets ``undecided``."""
    arena = getattr(pool, "arena", None)
    cache._w3_room_undecided = False
    if arena is None:
        return 0
    ctx = _Ctx(cache, pool, claimer)          # lazy (B2)
    if is_decider(cache):
        released = 0
        if _debts(cache):
            released += int(apply(cache, pool, set(), released_out, ctx=ctx)["released"])
        missing = missing_acks(cache, arena)
        if missing:
            waited = time.monotonic() - float(cache.__dict__.get("_w3_room_pub_t", 0.0))
            if waited < WAIT_TIMEOUT_S:
                # review A4: no new set while a follower has not applied the last one
                cache._w3_room_undecided = released == 0
                _N["wait"] += 1
                if _N["wait"] <= 16 or _N["wait"] % 256 == 0:
                    logger.info("W3-ROOM-SET WAIT pp_rank=0 gen=%d missing=%s (n=%d): a follower has not "
                                "acknowledged the last set -- no new decision, undecided",
                                int(getattr(cache, "_w3_room_pub_gen", 0)), missing, _N["wait"])
                return released
            # review F1b B3: bounded -- a follower that never acknowledges does not freeze PP0
            _N["timeout"] = _N.get("timeout", 0) + 1
            if _N["timeout"] <= 16 or _N["timeout"] % 256 == 0:
                logger.warning("W3-ROOM-SET WAIT-TIMEOUT pp_rank=0 gen=%d missing=%s waited_s=%.1f (n=%d): "
                               "no acknowledgement within %.1f s -- PP0 decides anew",
                               int(getattr(cache, "_w3_room_pub_gen", 0)), missing, waited, _N["timeout"],
                               WAIT_TIMEOUT_S)
        slots = decide(cache, pool, want, ctx=ctx)
        if not slots:
            return released
        gen = publish(arena, slots, cache)
        cache._w3_room_pub_t = time.monotonic()
        cache._w3_room_gen = gen
        r = apply(cache, pool, set(slots), released_out, ctx=ctx)
        ack(cache, arena, _nonce(cache), gen)     # PP0's own debts, for the followers' B6 check
        _log("publish", cache, gen, len(slots), r)
        return released + int(r["released"])
    r = _apply_new(cache, pool, arena, ctx, released_out) or {"released": 0, "nodes": 0, "kept": 0}
    released = int(r["released"])
    if released < int(need_pages):
        o = _own_only(cache, pool, int(need_pages) - released, ctx, released_out)
        if o["nodes"]:
            _N["own"] += 1
            if _N["own"] <= 16 or _N["own"] % 128 == 0:
                logger.info("W3-ROOM-SET OWN pp_rank=%s released_pages=%d nodes=%d (n=%d): slots only this "
                            "follower references (refs <= 1) -- no other rank's page", getattr(cache, "pp_rank", "?"),
                            o["released"], o["nodes"], _N["own"])
        released += int(o["released"])
    # review F1b B4: undecided only while a debt of this rank waits on a TRANSIENT blocker (a write in
    # flight, the page cap, an L3 copy not complete yet, an error) -- a standing debt (a running request,
    # the read's own told host lock, keep-payable) is a decided 'not now'
    cache._w3_room_undecided = released == 0 and any(
        len(e) > 3 and e[3] in _TRANSIENT for e in _debts(cache).values())
    return released


def _log(kind: str, cache: Any, gen: int, nslots: int, r: Dict[str, int]) -> None:
    _N[kind] = _N.get(kind, 0) + 1
    k = _N[kind]
    if k <= 16 or k % 128 == 0 or r.get("kept"):
        logger.info("W3-ROOM-SET %s pp_rank=%s gen=%d slots=%d released_pages=%d nodes=%d kept=%d capped=%d "
                    "debts=%d (n=%d): one slot set for the PP group (PP0 decides from the arena order, every "
                    "stage gives back its host copies of exactly these slots, L3 copy first -- COMPLETE slots "
                    "only; kept = nodes it cannot release now, on its debt list by node identity)",
                    kind.upper(), getattr(cache, "pp_rank", "?"), gen, nslots, r.get("released", 0),
                    r.get("nodes", 0), r.get("kept", 0), r.get("capped", 0), r.get("debts", 0), k)
