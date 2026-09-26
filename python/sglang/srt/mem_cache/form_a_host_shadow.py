"""R12 (fLLiper release table row 12): on a Form A D group every rank keeps
the SAME host and anchor entries -- the root of the rank-divergence class
rc9i/rc9k/rc9l/rc9m/rc9o.

THE ROOT. NF D runs Form A: TP0 holds attention, GDN/mamba state, KV and the
arena host pools (``ArenaMHAHostPool``, ``ArenaMambaPoolHost``); TP1/TP2 are
expert workers with 0-byte plain host pools (``MHATokenToKVPoolHost 0.00 GB``,
``MambaPoolHost 0.00 GB``) and a null storage tier. Every rank decided the
host life of a node LOCALLY:

* store ack (``UnifiedRadixCache._drain_storage_control_queues_impl`` ->
  ``_drain_backup``): TP0 rebinds the rows to the arena and keeps them
  (``_weg2_rebind_host_to_arena``), a worker frees them as transit
  (``_weg2_release_chain_piece_host``, HOST layer of EVERY component, the
  mamba anchor included) -- rc9m: ``WEG2 PUBLISH-CHAIN host released`` on
  TP1/TP2 only;
* load-back end (``loading_check``): the worker frees again (no arena_read);
* on the next device eviction TP0's node survives as a host node with its
  anchor (``#1469 EVICT ... backuped=True host=True``), the worker's dies
  (``backuped=False host=False``) -> anchors at different depths on
  different ranks -> every verdict built on the tree (usable match, #928,
  prefetch span, too_short) can split. H93, H96, H97, H98, H99 each close
  one of those VERDICTS; this module removes the difference in the TREES.

THE POLICY (variant c + a, no new collective):

(c) A Form A worker keeps its host entries as byteless bookkeeping -- the
    rows carry 0 bytes, keeping them costs nothing but row capacity. It
    never frees a row as transit and never evicts host on its own; TP0 is
    the one rank whose host life has meaning.
(a) TP0's OWN host decisions reach the workers over a wire that already
    exists: the TP request broadcast (``request_receiver.recv_requests`` ->
    ``broadcast_pyobj`` over ``tp_cpu_group``, the ``tp<-reqs`` site) -- the
    same stream the #969 section W3 flip decision rides. TP0 appends one
    :class:`FormAHostVerdict` at the origin; right after the broadcast
    every rank takes it out of the list and applies it, BEFORE any request
    of that pass is processed, so no admission, intake or vote of the pass
    sees a difference. Events are keyed by content (the node's last page
    hash + its end depth), never by the rank-local node id.

    * TRANSIT: a store ack whose arena rebind FAILED on TP0 -- TP0 used to
      free the rows there and then; now it defers, and every rank (TP0
      included) runs the same ``_weg2_release_chain_piece_host`` at the
      next broadcast.
    * STATE: a host drop TP0 had to take at once (H19 mamba-arena
      displacement, its own ``evict_host``, a refused backup): TP0 sends the
      node's state (exists, KV host, anchor host); a worker reconciles to
      it (drops what TP0 no longer holds). These happen at insert/backup
      time, after the pass's admission and before the next broadcast.

(b) Capacity from replicated inputs: TP0 addresses ``staging + arena``
    anchors, a worker's mamba pool had the synced 6 slots only. A byteless
    worker pool gets ``2 x SGLANG_HICACHE_ARENA_MAMBA_SLOTS`` extra rows (the
    arena's own slot count, the same environment on every rank, twice for
    drops in flight). The KV pool of a worker already covers TP0's id space
    (fnFL2x62 ``_carrier_capacity_bid``).

Switch ``SGLANG_WEG2_ENABLE_FORM_A_HOST_SHADOW``; it acts only with an
installed Form A role plan whose host is TP rank 0, on a pure TP group
(no DP attention, pp 1). Off (or a classic boot) = the pre-R12 code paths,
unchanged.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

#: event kinds
TRANSIT = "transit"
STATE = "state"

#: Tree attribute: while set to a list, ``_evict_component_and_detach_lru``
#: records every node whose HOST layer it frees (TP0's own evict_host).
REC_ATTR = "_r12_rec"


class FormAHostVerdict:
    """TP0's host decisions of one pass; rides the ``tp<-reqs`` broadcast.

    ``events``: tuples ``(kind, last_hash, depth, exists, kv_host, anchor_host)``.
    """

    __slots__ = ("seq", "events")

    def __init__(self, seq: int, events: List[Tuple[str, str, int, int, int, int]]):
        self.seq = int(seq)
        self.events = list(events)

    def __getstate__(self):
        return (self.seq, self.events)

    def __setstate__(self, state):
        self.seq, self.events = state

    def __repr__(self) -> str:
        return f"FormAHostVerdict(seq={self.seq}, events={len(self.events)})"


class _State:
    """Per-process state: one D tree per scheduler process."""

    def __init__(self) -> None:
        self.ledger: List[Tuple[str, str, int, int, int, int]] = []
        self.pending: Dict[Tuple[str, int], Tuple[str, str, int, int, int, int]] = {}
        self.absent: set = set()
        self.tree: Any = None
        self.seq = 0
        self.counts: Dict[str, int] = {}
        self.announced = False


_S = _State()


def _reset_state_for_tests() -> None:
    global _S
    _S = _State()


# ---------------------------------------------------------------- predicates
def shadow_active() -> bool:
    """Group-uniform: switch on, a Form A plan installed, its host is TP rank 0
    (the request origin). The same answer on every rank of the group."""
    from sglang.srt.environ import envs

    if not envs.SGLANG_WEG2_ENABLE_FORM_A_HOST_SHADOW.get():
        return False
    from sglang.srt.rank_role import installed_role_plan

    plan = installed_role_plan()
    if plan is None:
        return False
    try:
        return int(getattr(plan, "host_rank", 0)) == 0
    except (TypeError, ValueError):
        return False


def role() -> Optional[str]:
    """``"host"`` (TP0, the source), ``"worker"`` (follows), or None (off)."""
    if not shadow_active():
        return None
    from sglang.srt.rank_role import this_rank_is_form_a_worker

    return "worker" if this_rank_is_form_a_worker() else "host"


def wire_active(server_args: Any, ps: Any) -> bool:
    """The broadcast carries verdicts: shadow on, pure TP group (the
    ``tp<-reqs`` branch of ``_broadcast_reqs_across_ranks``), pp 1."""
    try:
        if getattr(server_args, "enable_dp_attention", False):
            return False
        if int(getattr(ps, "pp_size", 1)) != 1 or int(getattr(ps, "tp_size", 1)) <= 1:
            return False
    except (TypeError, ValueError):
        return False
    return shadow_active()


def _note(key: str, every: int = 64) -> int:
    n = _S.counts.get(key, 0) + 1
    _S.counts[key] = n
    return n if (n <= 8 or n % every == 0) else 0


def _announce(r: str) -> None:
    if not _S.announced:
        _S.announced = True
        logger.info(
            "R12 HOST-SHADOW ACTIVE role=%s (worker host rows are byteless "
            "bookkeeping kept while TP0 holds them; TP0 host decisions ride the "
            "tp<-reqs broadcast)", r)


def register_tree(tree: Any) -> None:
    _S.tree = tree


# ---------------------------------------------------------------- node keys
def node_key(node: Any) -> Optional[Tuple[str, int]]:
    """(last page hash, end depth in tokens) -- rank-invariant content key."""
    try:
        hv = getattr(node, "hash_value", None)
        if not hv:
            return None
        depth = 0
        n = node
        while n is not None and getattr(n, "parent", None) is not None:
            depth += len(n.key)
            n = n.parent
        return str(hv[-1]), int(depth)
    except Exception:  # noqa: BLE001 - an unkeyable node is counted, not fatal
        return None


def _attached(node: Any) -> bool:
    n = node
    while getattr(n, "parent", None) is not None:
        p = n.parent
        if not any(c is n for c in p.children.values()):
            return False
        n = p
    return True


def _anchor_ct():
    from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType

    return ComponentType.MAMBA


def _base_ct():
    from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType

    return ComponentType.FULL


def _host_flags(node: Any) -> Tuple[int, int]:
    cd = node.component_data
    kv = int(cd[int(_base_ct())].host_value is not None)
    act = int(_anchor_ct())
    m = int(len(cd) > act and cd[act].host_value is not None)
    return kv, m


# ---------------------------------------------------------------- TP0 side
def record_transit(tree: Any, node: Any) -> bool:
    """TP0: the store ack wants to free this node's host rows as transit.
    Deferred to the next broadcast, where every rank runs the same release.
    False = not keyable -> the caller releases at once (pre-R12)."""
    k = node_key(node)
    if k is None:
        if _note("unkeyed_transit"):
            logger.warning("R12 HOST-VERDICT unkeyed transit node=%s (released at once)",
                           getattr(node, "id", "?"))
        return False
    register_tree(tree)
    _announce("host")
    _S.ledger.append((TRANSIT, k[0], k[1], 1, 0, 0))
    return True


def record_state(tree: Any, node: Any, why: str = "") -> None:
    """TP0: its host life of ``node`` changed by a decision of its own. Send the
    node's state now; a worker reconciles to it."""
    if node is None or node is getattr(tree, "root_node", None):
        return
    k = node_key(node)
    if k is None:
        if _note("unkeyed_state"):
            logger.warning("R12 HOST-VERDICT unkeyed state node=%s why=%s",
                           getattr(node, "id", "?"), why)
        return
    register_tree(tree)
    _announce("host")
    exists = int(_attached(node))
    kv, m = _host_flags(node) if exists else (0, 0)
    if not kv:
        _S.absent.add(k)
    else:
        _S.absent.discard(k)
    _S.ledger.append((STATE, k[0], k[1], exists, kv, m))


def note_backup_ok(tree: Any, node: Any) -> None:
    """TP0 backed up a node it earlier reported without host: say so, so a
    worker's still-pending drop for it is superseded."""
    if not _S.absent:
        return
    k = node_key(node)
    if k is not None and k in _S.absent:
        record_state(tree, node, why="backup_ok")


def attach(recv_reqs: List) -> List:
    """Origin (TP0), right before the broadcast: put the pass's verdict FIRST."""
    if not _S.ledger:
        return recv_reqs
    _S.seq += 1
    v = FormAHostVerdict(_S.seq, _S.ledger)
    _S.ledger = []
    n = _note("sent")
    if n:
        kinds = [e[0] for e in v.events]
        logger.info("R12 HOST-VERDICT SENT seq=%d events=%d transit=%d state=%d (n=%d)",
                    v.seq, len(v.events), kinds.count(TRANSIT), kinds.count(STATE), n)
    return [v] + list(recv_reqs)


def consume(recv_reqs: Optional[List], tree: Any = None) -> Optional[List]:
    """Every rank, right after the broadcast: take the verdicts out of the
    list and apply them before any request of this pass is processed."""
    if not recv_reqs:
        if _S.pending and (tree or _S.tree) is not None:
            apply(tree or _S.tree, [])
        return recv_reqs
    verdicts = [r for r in recv_reqs if isinstance(r, FormAHostVerdict)]
    if not verdicts:
        if _S.pending and (tree or _S.tree) is not None:
            apply(tree or _S.tree, [])
        return recv_reqs
    rest = [r for r in recv_reqs if not isinstance(r, FormAHostVerdict)]
    events: List = []
    for v in verdicts:
        events.extend(v.events)
    t = tree or _S.tree
    if t is None:
        # no tree has touched HiCache yet on this rank: keep for the next pass
        for e in events:
            _S.pending[(e[1], int(e[2]))] = e
        return rest
    apply(t, events, seq=verdicts[-1].seq)
    return rest


def on_tree_reset(tree: Any) -> None:
    """The tree is dropped (flush/flip, on every rank at the same point): no
    event may outlive the nodes it names -- a re-inserted prefix has the same
    hash."""
    if _S.ledger or _S.pending or _S.absent:
        logger.info("R12 HOST-VERDICT reset drops ledger=%d pending=%d",
                    len(_S.ledger), len(_S.pending))
    _S.ledger = []
    _S.pending = {}
    _S.absent = set()


# ---------------------------------------------------------------- apply
def _index(tree: Any) -> Dict[str, List[Any]]:
    idx: Dict[str, List[Any]] = {}
    stack = [tree.root_node]
    while stack:
        n = stack.pop()
        for c in n.children.values():
            stack.append(c)
            hv = getattr(c, "hash_value", None)
            if hv:
                idx.setdefault(str(hv[-1]), []).append(c)
    return idx


def _find(idx, key: str, depth: int):
    for n in idx.get(key, ()):
        k = node_key(n)
        if k is not None and k[1] == int(depth):
            return n
    return None


def _busy(tree: Any, node: Any) -> bool:
    if any(int(getattr(cd, "host_lock_ref", 0) or 0) > 0 for cd in node.component_data):
        return True
    owt = getattr(tree, "ongoing_write_through", None) or {}
    return node.id in owt or getattr(node, "write_through_pending_id", None) is not None


def apply(tree: Any, events: List, seq: int = 0) -> Dict[str, int]:
    """Apply TP0's events on THIS rank. The same code on every rank; TRANSIT
    runs everywhere (TP0 deferred its own release to here), STATE only on a
    worker (TP0 is already in that state)."""
    r = role()
    c = dict(transit=0, kv=0, anchor=0, deleted=0, short=0, missing=0,
             pending=0, unreconcilable=0, noop=0)
    for e in events:
        _S.pending[(e[1], int(e[2]))] = e  # a newer event for the key supersedes
    if not _S.pending:
        return c
    work = list(_S.pending.values())
    _S.pending = {}
    idx = _index(tree)
    for e in work:
        kind, h, depth, exists, kv, m = e
        node = _find(idx, h, depth)
        if node is None:
            c["missing"] += 1
            continue
        if kind == TRANSIT:
            before = _host_flags(node)
            tree._weg2_release_chain_piece_host(node)
            c["transit" if _host_flags(node) != before else "noop"] += 1
            continue
        if r != "worker":
            continue
        out = _reconcile(tree, node, exists, kv, m)
        if out == "pending":
            _S.pending[(h, int(depth))] = e
        c[out] += 1
    n = _note("applied")
    if n and (events or any(c[k] for k in ("transit", "kv", "anchor", "deleted"))):
        logger.info(
            "R12 HOST-VERDICT APPLIED seq=%d role=%s events=%d transit=%d kv_dropped=%d "
            "anchor_dropped=%d deleted=%d short=%d missing=%d pending=%d unreconcilable=%d (n=%d)",
            seq, r, len(events), c["transit"], c["kv"], c["anchor"], c["deleted"], c["short"],
            c["missing"], c["pending"], c["unreconcilable"], n)
    if c["short"] and _note("short_apply", 16):
        logger.warning("R12 SHADOW-SHORT at reconcile: %d node(s) TP0 holds on host, "
                       "this worker does not (its own backup was refused)", c["short"])
    if c["unreconcilable"] and _note("unreconcilable", 16):
        logger.error("R12 SHADOW-UNRECONCILABLE %d node(s): TP0 dropped host of a node "
                     "that is device-evicted and interior here", c["unreconcilable"])
    return c


def _reconcile(tree: Any, node: Any, exists: int, kv: int, m: int) -> str:
    from sglang.srt.mem_cache.unified_cache_components.tree_component import EvictLayer

    has_kv, has_m = _host_flags(node)
    want_kv = bool(exists and kv)
    want_m = bool(exists and m)
    if (want_kv and not has_kv) or (want_m and not has_m):
        return "short"
    if (has_kv and not want_kv) or (has_m and not want_m):
        if _busy(tree, node):
            return "pending"
    if has_kv and not want_kv:
        if not node.evicted:
            for comp in tree._components_tuple:
                tree._evict_component_and_detach_lru(node, comp, target=EvictLayer.HOST, tracker=None)
            tree.evictable_host_leaves.discard(node)
            tree._update_evictable_leaf_sets(node)
            return "kv"
        if tree._is_host_leaf(node):
            tracker = {ct: 0 for ct in tree.tree_components}
            tree._evict_host_leaf(node, tracker)
            return "deleted"
        return "unreconcilable"
    if has_m and not want_m:
        comp = tree.components[_anchor_ct()]
        tree._evict_component_and_detach_lru(node, comp, target=EvictLayer.HOST, tracker=None)
        tree._update_evictable_leaf_sets(node)
        return "anchor"
    return "noop"


# ---------------------------------------------------------------- worker side
def worker_keeps(tree: Any, site: str, node: Any) -> None:
    """A worker kept rows it used to free as transit (store ack / load-back)."""
    register_tree(tree)
    _announce("worker")
    n = _note("keep_" + site, 256)
    if n:
        logger.info("R12 SHADOW-KEEP site=%s node=%s tokens=%d (byteless; TP0 holds "
                    "the bytes) n=%d", site, getattr(node, "id", "?"),
                    len(getattr(node, "key", None) or ()), n)


def worker_refuses_own_evict(tree: Any, component_type: Any, num_tokens: int) -> None:
    register_tree(tree)
    n = _note("own_evict", 16)
    if n:
        logger.warning(
            "R12 SHADOW-OWN-EVICT-REFUSED comp=%s need=%d n=%d (a worker never "
            "evicts host on its own -- its pool is short of TP0's; the caller "
            "refuses instead)", getattr(component_type, "name", component_type),
            int(num_tokens), n)


def worker_short(tree: Any, why: str, node: Any) -> None:
    register_tree(tree)
    n = _note("short_" + why.split(":", 1)[0], 64)
    if n:
        logger.warning("R12 SHADOW-SHORT why=%s node=%s (this worker refused a backup "
                       "TP0 may take) n=%d", why, getattr(node, "id", "?"), n)


def mamba_shadow_extra_rows(size_per_slot: int) -> int:
    """Extra byteless anchor rows for a Form A worker (0 B/slot only):
    twice the mamba arena's slot count, the value the arena itself reads
    (``HiCacheFile._arena_for``) -- a replicated input, no collective."""
    try:
        if int(size_per_slot) != 0 or role() != "worker":
            return 0
        from sglang.srt.mem_cache.pool_host.arena_pool import arena_host_enabled

        if not arena_host_enabled():
            return 0
        # Read exactly as the arena reads it (HiCacheFile._arena_for, raw
        # environment, default 48): the point is the SAME number TP0's arena
        # gets, so it must not go through a second parse with its own default.
        slots = int(os.environ.get("SGLANG_HICACHE_ARENA_MAMBA_SLOTS", "48"))
    except Exception:  # noqa: BLE001 - sizing falls back to the synced count
        return 0
    extra = 2 * max(0, slots)
    logger.info("R12 SHADOW-CAPACITY mamba anchor rows +%d (2 x arena slots %d, 0 B/row)",
                extra, slots)
    return extra
