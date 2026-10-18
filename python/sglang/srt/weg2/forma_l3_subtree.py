"""S39 (18.10.) FORMA-L3-SUBTREE: a store-backed D device leaf leaves the device WITHOUT an L2 slot even when it carries
host-only children or a device anchor -- decided by TP0, applied by every rank of the Form A group at the same broadcast.

METAL (fLLiper rc11 49f9c2ea4d, boot ..._dauer10181332_49f9c2ea4d_1018_133242.D.log, D TP0 13:46:27-13:48:03Z, 96 s
without a D round, ``ADMISSION-WEDGE: 7 queued, 0 running``; the same class at rc9 10:09 and the rc8 death 12:23):
``EVICT-FRONTIER-CENSUS request=36544 delivered_before=0 delivered_after_repair=0 reported_evictable=467328
on_frontier=448960``. Every frontier leaf took the same three steps, 709 times in the boot:

  1. ``_weg2_l3_drop`` -> ``_weg2_l3_drop_forma`` REFUSED: the leaf had host-only children (``UD-REFUSED ... children=1``
     / ``children=2`` on 15 of the 17 refused leaves: node 241 178816 tokens, 304 178944, 318 92032 ...) or a device
     Mamba anchor (``#924D station=donate ... node=241``). ``L3-DROP FORMA`` fired 0 times in the whole boot.
  2. ``write_backup`` asked a NEW arena claim for pages the store already holds (``#1421 BACKUP-REFUSED why=arena_claim
     node=241 tokens=178816``; the leaf is the load-back of pdflip-12-40, whose 2794 pages ``#248 PARK-DEMOTE`` had
     put on L3 -- ``l3_pages=2794 absent=0 complete=1`` -- and the wake read had marked ``l3_present``). The KV arena
     was full of the parked requests' host-only spans (``ARENA-REF-HOLDERS tree=5819``, ``ARENA-DROP freed=0``, the
     W3 spill ``candidates=0``, the host-leaf spill ``refused=aux_host_only:10,device_resident:5,has_children:5``).
  3. the UD drop is the local-PP floor's only (``UD-REFUSED why=not_local_pp_floor``) -> the peel delivered 0.

The host-only children are the next turns' spans of the same conversations, inserted by the wake read (pdflip-12-41
behind pdflip-12-40): host-only, ``l3_present`` (``_mark_l3_present_span``: their pages came out of the store), their
arena pages COMPLETE. Write-behind is NOT the wall: D TP0's passes found ``new=653 on_disk=651`` (two pages to write).

THE RULE. A store-backed device leaf (``l3_present``, no host copy, unlocked, no write in flight) whose children are a
host-only subtree -- every node off the device, host-backed, ``l3_present``, no device or host lock, no write in
flight, no live aux device state -- leaves the device WITHOUT an arena claim: the subtree leaves the tree (children
first, ``_evict_host_leaf``: the arena references go back; a COMPLETE slot is written to L3 before the clock frees it,
#257, and a host anchor below is put on L3 here first), then the leaf is dropped (``_ud_drop_unbacked_leaf``). Its KV
comes back from the store (arena adopt or KV-DURCHLAUF). A device anchor on the leaf goes only when the store holds the
state of its last page (``ArenaMambaPoolHost.anchor_held``); otherwise TP0 issues the anchor-only copy (S25) and the
leaf waits for it. Nothing is ever dropped that has no copy: a leaf or child outside the store, a lock, a write in
flight, an anchor nowhere else -- the leaf stays, NAMED (``S39 L3-DROP FORMA KEPT why=...``).

GROUP-UNIFORM (SCHEDULER-SOLL R11, raenge-nie-uneins): the peel runs on every rank and refuses such a leaf on every
rank alike; TP0 queues it. At TP0's broadcast (``form_a_host_shadow.attach``) TP0 checks it on its own tree, drops
it and sends ``L3DROP`` (the leaf's content key); every worker applies the SAME drop right after the broadcast, before
any request of the pass -- the trees and the device room are equal again before the next admission. Inputs a worker
cannot read (the arena, the store, the anchor) are TP0's alone. A worker that cannot follow at once (a lock of its own)
keeps the event pending and is named (``S39 L3-DROP FORMA DIVERGED``) after ``MAX_PASSES`` broadcasts.

``SGLANG_WEG2_FORMA_L3_SUBTREE``: new feature, default ON (user order 14.10.); ``0`` only with a user word -- the
pre-S39 refusal byte for byte.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

ENV = "SGLANG_WEG2_FORMA_L3_SUBTREE"
MARK = "S39 L3-DROP FORMA"
#: broadcasts a queued leaf may wait (anchor-only ack, a transient lock) before it is named and dequeued; a worker's
#: pending drop is named DIVERGED after as many
MAX_PASSES = 16
#: child host anchors put on L3 synchronously per broadcast (each ~59 MB on NF; already on disk = a stat only)
ANCHOR_SECURE_PER_FLUSH = 4

#: refusals that mean "the leaf is in use or already gone" -- dequeued silently, the peel queues it again if needed
_SILENT = ("detached", "not_on_device", "backuped", "locked", "not_device_leaf")


class _State:
    def __init__(self) -> None:
        # id(node) -> [tree, node, passes, last_why]
        self.queue: Dict[int, list] = {}
        # worker: (hash, depth) -> passes pending
        self.wpend: Dict[Tuple[str, int], int] = {}
        self.n = {"queued": 0, "dropped": 0, "kept": 0, "anchor_save": 0, "worker_dropped": 0,
                  "worker_pending": 0, "diverged": 0}


_ST = _State()


def _reset_for_tests() -> None:
    global _ST
    _ST = _State()


def reset() -> None:
    """The tree is dropped (flush/flip, every rank at the same point): no queued node may outlive it."""
    _ST.queue.clear()
    _ST.wpend.clear()


def on(env=None) -> bool:
    e = os.environ if env is None else env
    return str(e.get(ENV, "1")).strip().lower() not in ("0", "false", "no", "off", "")


def _r12():
    from sglang.srt.mem_cache import form_a_host_shadow as r12

    return r12


def _cts():
    from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType
    from sglang.srt.mem_cache.unified_radix_cache import BASE_COMPONENT_TYPE

    return BASE_COMPONENT_TYPE, getattr(ComponentType, "MAMBA", None)


def _in_flight(tree: Any, node: Any) -> bool:
    nid = getattr(node, "id", None)
    return (nid in (getattr(tree, "ongoing_write_through", None) or {})
            or getattr(node, "write_through_pending_id", None) is not None
            or nid in (getattr(tree, "_weg2_direct_mamba_rows", None) or {}))


def _device_locked(node: Any) -> bool:
    return any(int(getattr(cd, "lock_ref", 0) or 0) > 0 for cd in node.component_data)


def _host_locked(node: Any) -> bool:
    return any(int(getattr(cd, "host_lock_ref", 0) or 0) > 0 for cd in node.component_data)


def _aux_device(tree: Any, node: Any) -> bool:
    base, _ = _cts()
    return any(node.component_data[c.component_type].value is not None
               for c in getattr(tree, "_components_tuple", ()) if c.component_type != base)


def _anchor_on_device(node: Any) -> bool:
    _, mamba = _cts()
    return (mamba is not None and len(node.component_data) > int(mamba)
            and node.component_data[mamba].value is not None)


def _child_anchor_host(node: Any):
    _, mamba = _cts()
    if mamba is None or len(node.component_data) <= int(mamba):
        return None
    return node.component_data[mamba].host_value


def subtree(tree: Any, node: Any, *, tp0: bool = True) -> Tuple[Optional[List[Any]], Optional[Tuple[str, Any]]]:
    """The host-only subtree below ``node``, children before their parents, or ``(None, (why, node id))``.

    Every node: off the device, host-backed, no device or host lock, no write in flight, no live aux device state.
    On TP0 additionally ``l3_present`` (the group's store verdict). A worker (``tp0=False``) applies TP0's verdict and
    checks only what would make the removal itself unsafe on its tree (locks, writes, device rows)."""
    base, _ = _cts()
    order: List[Any] = []
    stack = list(node.children.values())
    while stack:
        d = stack.pop()
        did = getattr(d, "id", "?")
        if d.component_data[base].value is not None:
            return None, ("child_on_device", did)
        if d.component_data[base].host_value is None:
            return None, ("child_unbacked", did)
        if _device_locked(d):
            return None, ("child_device_locked", did)
        if _host_locked(d):
            return None, ("child_host_locked", did)
        if _in_flight(tree, d):
            return None, ("child_write_in_flight", did)
        if _aux_device(tree, d):
            return None, ("child_aux_device", did)
        if tp0 and not getattr(d, "l3_present", False):
            return None, ("child_not_in_store", did)
        order.append(d)
        stack.extend(d.children.values())
    order.reverse()   # a DFS pre-order reversed: every node after all of its descendants
    return order, None


def _leaf_refusal(tree: Any, node: Any) -> Optional[str]:
    """TP0 at its broadcast: may ``node`` still leave as a store-backed device leaf?"""
    r12 = _r12()
    base, _ = _cts()
    if not r12._attached(node) or node is getattr(tree, "root_node", None):
        return "detached"
    if node.component_data[base].value is None:
        return "not_on_device"
    if node.component_data[base].host_value is not None:
        return "backuped"            # the ordinary demotion takes it, no claim needed
    if _device_locked(node):
        return "locked"
    if not tree._is_device_leaf(node):
        return "not_device_leaf"
    if not getattr(node, "l3_present", False):
        return "not_in_store"        # D-NORECOMPUTE: the device copy is the only one
    if _in_flight(tree, node):
        return "write_in_flight"
    if r12.node_key(node) is None:
        return "unkeyed"             # no verdict could reach the workers
    return None


def _anchor_ready(tree: Any, node: Any) -> Tuple[bool, str]:
    """A device anchor leaves only when the store holds the state of the leaf's last page. TP0 only."""
    if not _anchor_on_device(node):
        return True, "none"
    mp = tree._weg2_mamba_pool() if hasattr(tree, "_weg2_mamba_pool") else None
    held = None
    probe = getattr(mp, "anchor_held", None)
    if callable(probe) and getattr(node, "hash_value", None):
        try:
            held = probe(node.hash_value[-1])
        except Exception as exc:  # noqa: BLE001 - unreadable = not held (fail closed)
            logger.warning("%s anchor probe raised: %r", MARK, exc)
            held = None
    if held:
        return True, "held"
    issue = getattr(tree, "write_backup_anchor_only", None)
    if callable(issue) and not _in_flight(tree, node):
        try:
            if issue(node):
                _ST.n["anchor_save"] += 1
                return False, "anchor_save_issued"
        except Exception as exc:  # noqa: BLE001 - the leaf keeps its anchor
            logger.warning("%s anchor-only save raised: %r", MARK, exc)
    return False, "anchor_not_in_store"


def _secure_child_anchors(tree: Any, order: List[Any], budget: List[int]) -> Optional[Tuple[str, Any]]:
    """Every host anchor of the subtree gets its L3 copy before its node leaves (#257 holds for an unreferenced slot;
    this makes it independent of the clock). None = all secured; else the refusal. TP0 only."""
    needs = [d for d in order if _child_anchor_host(d) is not None]
    if not needs:
        return None
    mp = tree._weg2_mamba_pool() if hasattr(tree, "_weg2_mamba_pool") else None
    arena = getattr(mp, "arena", None)
    backend = getattr(mp, "_backend", None)
    settled = getattr(mp, "settled_anchor_slots", None)
    secure = getattr(backend, "arena_secure_to_disk", None)
    if arena is None or not callable(settled) or not callable(secure):
        return "child_anchor_no_pool", getattr(needs[0], "id", "?")
    stat = getattr(backend, "_stat_stems", None)
    total = int(getattr(arena, "slot_bytes", 0) or 0)
    for d in needs:
        slots = settled(_child_anchor_host(d))
        if not slots:
            return "child_anchor_in_flight", getattr(d, "id", "?")
        try:
            stems = [arena.slot_stem(int(s)) for s in slots]
            if callable(stat) and all(stems) and set(stems) <= set(stat(stems)):
                continue
        except Exception:  # noqa: BLE001 - no reading: write it below
            pass
        if budget[0] <= 0:
            return "child_anchor_cap", getattr(d, "id", "?")
        budget[0] -= len(slots)
        try:
            res = secure(arena, [(int(s), None, None, total) for s in slots])
        except Exception as exc:  # noqa: BLE001 - fail closed
            logger.warning("%s child anchor secure raised: %r", MARK, exc)
            return "child_anchor_lost", getattr(d, "id", "?")
        if int(res.get("lost", 0)) or int(res.get("on_disk", 0)) + int(res.get("written", 0)) < len(slots):
            return "child_anchor_lost", getattr(d, "id", "?")
    return None


def _execute(tree: Any, node: Any, order: List[Any]) -> Tuple[int, int]:
    """Every rank: the host-only subtree leaves the tree (children first), then the leaf leaves the device.
    Returns (device tokens freed, host tokens freed)."""
    base, _ = _cts()
    host_tracker = {ct: 0 for ct in tree.tree_components}
    for d in order:
        tree._evict_host_leaf(d, host_tracker)
    tracker = {ct: 0 for ct in tree.tree_components}
    tree._ud_drop_unbacked_leaf(node, tracker)
    return int(tracker.get(base, 0)), int(host_tracker.get(base, 0))


def room_pending(tree: Any = None) -> int:
    """TP0: device tokens of the store-backed leaves of ``tree`` (any tree when None) queued for the next broadcast's
    drop (0 on a worker, off, or an empty queue). PW-R reads it: a head this queue makes room for at the next
    broadcast is not unservable (S41, review S39 -- PW-R runs after the pass's admission, i.e. between the peel that
    queued and the drop)."""
    if not _ST.queue or not on():
        return 0
    return sum(len(getattr(ent[1], "key", ()) or ()) for ent in _ST.queue.values() if tree is None or ent[0] is tree)


def note_refused(tree: Any, node: Any, why: str) -> bool:
    """The peel (every rank) refused a store-backed leaf for its children or its device anchor. TP0 queues it for
    its next broadcast, at most the tokens the peel still asked for (``tree._s39_peel`` = [need, queued])."""
    if not on() or _r12().role() != "host":
        return False
    key = id(node)
    if key in _ST.queue:
        return True
    peel = getattr(tree, "_s39_peel", None)
    if isinstance(peel, list) and len(peel) == 2 and peel[0] > 0:
        if peel[1] >= peel[0]:
            return False
        peel[1] += len(getattr(node, "key", ()) or ())
    _ST.queue[key] = [tree, node, 0, why]
    _ST.n["queued"] += 1
    return True


def _kept(tree: Any, node: Any, ent: list, why: str, extra: str = "") -> None:
    ent[2] += 1
    ent[3] = why
    if ent[2] > MAX_PASSES:
        _ST.queue.pop(id(node), None)
        _ST.n["kept"] += 1
        k = _ST.n["kept"]
        if k <= 32 or (k & (k - 1)) == 0:
            logger.warning("%s KEPT n=%d node=%s tokens=%d why=%s%s passes=%d (a store-backed device leaf could not "
                           "leave without an arena claim: nothing is dropped without a copy -- the peel queues it "
                           "again when it needs the room)", MARK, k, getattr(node, "id", "?"),
                           len(getattr(node, "key", ()) or ()), why, extra, ent[2])


def flush_at_broadcast(tree: Any = None) -> int:
    """TP0, in ``form_a_host_shadow.attach`` right before the broadcast: every queued leaf that still qualifies is
    dropped here (subtree first) and its ``L3DROP`` verdict rides THIS broadcast. Returns device tokens freed."""
    if not _ST.queue or not on():
        return 0
    r12 = _r12()
    if r12.role() != "host":
        _ST.queue.clear()
        return 0
    freed_total = 0
    budget = [ANCHOR_SECURE_PER_FLUSH]
    for key, ent in list(_ST.queue.items()):
        t, node, _passes, _why = ent
        try:
            why = _leaf_refusal(t, node)
            if why in _SILENT:
                _ST.queue.pop(key, None)
                continue
            if why is not None:
                _kept(t, node, ent, why)
                continue
            order, cwhy = subtree(t, node, tp0=True)
            if order is None:
                _kept(t, node, ent, cwhy[0], " holder=%s" % (cwhy[1],))
                continue
            if order:
                # S22: a host-only leaf of the subtree that a worker holds on its device (or with children of its own)
                # cannot be deleted there -- TP0 leaves the drop alone (the K23 rule). S41 (review S39): the same for
                # an INTERIOR node of the subtree a worker holds on its device (``dv``): the worker's apply would
                # answer ``child_on_device`` with DIVERGED and keep a node TP0 deleted for good
                nd = r12.worker_nodelete_keys(t)
                dv = r12.worker_device_keys(t)
                stuck = next((d for d in order
                              if (not d.children and r12.node_key(d) in nd) or r12.node_key(d) in dv),
                             None) if (nd or dv) else None
                if stuck is not None:
                    r12.kept_note("s39_l3drop")
                    _kept(t, node, ent, "worker_cannot_delete", " holder=%s" % (getattr(stuck, "id", "?"),))
                    continue
            ok, awhy = _anchor_ready(t, node)
            if not ok:
                _kept(t, node, ent, awhy)
                continue
            swhy = _secure_child_anchors(t, order, budget)
            if swhy is not None:
                _kept(t, node, ent, swhy[0], " holder=%s" % (swhy[1],))
                continue
            k = r12.node_key(node)
            tokens = len(getattr(node, "key", ()) or ())
            dev, host = _execute(t, node, order)
            _ST.queue.pop(key, None)
            r12.record_l3_drop(t, k)
            freed_total += dev
            _ST.n["dropped"] += 1
            n = _ST.n["dropped"]
            if n <= 32 or (n & (n - 1)) == 0:
                logger.warning("%s n=%d node=%s tokens=%d freed=%d children=%d child_host_tokens=%d anchor=%s "
                               "(a store-backed D device leaf left the device without an arena claim; its host-only "
                               "subtree left the tree first -- every page is in the store, the KV comes back from "
                               "there; the workers apply the same drop at this broadcast)", MARK, n,
                               getattr(node, "id", "?"), tokens, dev, len(order), host, awhy)
        except Exception:  # noqa: BLE001 - a failed drop keeps the leaf (named), never breaks the broadcast
            _ST.queue.pop(key, None)
            logger.warning("%s flush raised on node=%s (the leaf stays)", MARK, getattr(node, "id", "?"),
                           exc_info=True)
    return freed_total


def apply_drop(tree: Any, node: Any, key: Tuple[str, int]) -> str:
    """A worker, right after the broadcast: the same drop TP0 took. ``dropped`` / ``pending`` / ``diverged`` /
    ``missing``."""
    r12 = _r12()
    base, _ = _cts()
    if node is None or not r12._attached(node):
        _ST.wpend.pop(key, None)
        return "missing"
    order, why = subtree(tree, node, tp0=False)
    busy = None
    if order is None and why[0] in ("child_device_locked", "child_host_locked", "child_write_in_flight"):
        busy = why[0]
    elif order is None:
        _ST.wpend.pop(key, None)
        return _diverged(node, key, why[0])
    if busy is None and _device_locked(node):
        busy = "locked"
    if busy is None and _in_flight(tree, node):
        busy = "write_in_flight"
    if busy is not None:
        p = _ST.wpend.get(key, 0) + 1
        _ST.wpend[key] = p
        _ST.n["worker_pending"] += 1
        if p > MAX_PASSES:
            _ST.wpend.pop(key, None)
            return _diverged(node, key, busy)
        return "pending"
    _ST.wpend.pop(key, None)
    if node.component_data[base].value is None:
        # this worker had the leaf off the device already (its own device life): the subtree and the host-only leaf go
        host_tracker = {ct: 0 for ct in tree.tree_components}
        for d in order:
            tree._evict_host_leaf(d, host_tracker)
        if tree._is_host_leaf(node):
            tree._evict_host_leaf(node, host_tracker)
        _ST.n["worker_dropped"] += 1
        return "dropped"
    _execute(tree, node, order)
    _ST.n["worker_dropped"] += 1
    return "dropped"


def _diverged(node: Any, key: Tuple[str, int], why: str) -> str:
    _ST.n["diverged"] += 1
    k = _ST.n["diverged"]
    if k <= 32 or (k & (k - 1)) == 0:
        logger.error("%s DIVERGED n=%d node=%s depth=%s hash=%s why=%s (TP0 dropped this store-backed leaf at its "
                     "broadcast and this worker could not follow -- the trees and the device room differ until the "
                     "node leaves here too)", MARK, k, getattr(node, "id", "?"), key[1], str(key[0])[:16], why)
    return "diverged"
