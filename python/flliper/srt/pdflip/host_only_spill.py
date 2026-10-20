"""W3-HOST-LEAF: the W3 arena spill also releases HOST-ONLY H-leaves on NF P -- a
PORT of the 27B line's Q-697c (b) ``dual_arena_spill.spill_host_only`` (27B integ
86ff356d0d, python/flliper/srt/pdflip/dual_arena_spill.py; its file head has the 27B
metal), there behind the dual-P gate, here behind an NF gate.

NF METAL (int22, boot ..._6b3bd1a6df_1008_171755, P 18:33:37-18:34:20Z):
``W3-ARENA SPILL n=1 need=64 released_pages=0 nodes=0 candidates=0`` on all three
stages (P.log 216254-216572). The candidate filter of
``UnifiedRadixCache._w3_arena_spill`` takes only nodes that still carry a DEVICE
value; the KV arena (6485 slots, ``arena_complete=6396`` at D's 18:32:57 reset,
D's tree 0 there) freed nothing for a claim (``#1427 ARENA-DROP ... freed=0``), every
backup was refused (``#1421 arena_claim`` / ``parent_unbacked``) and PP1's peel stopped
at two un-backed frontier leaves whose host-only children nobody could clear
(``EVICT-FRONTIER-CENSUS on_frontier=8192 behind_device_child=187776``, ``Prefill out
of memory``). PP1's tree dump after the death holds ~321k tokens (~5000 pages) in
HOST-ONLY nodes -- a reader reference each (a reading of the dump: no
ARENA-REF-HOLDERS census was logged between P's 18:31:25 reset and the death).

THE RULE (1:1 the 27B filter and loop, ``_reason`` / ``spill_host_only``): after the
device-resident round, a claim that is still short spills host-only H-LEAVES in
node-id order; each leaf first gets its L3 copy for EVERY page
(``secure_rows_to_l3``, #257: ``lost`` != 0 or a page not COMPLETE keeps it), then
leaves the tree whole (``_evict_host_leaf``: its references go back), its parent
next when it became an eligible leaf. A leaf with a host-only aux state (mamba
anchor), one in flight (the V1 guards of ``_subtree_blocked``) or without page
hashes stays.

NOT PORTED: the #1500i census (``dual_pkvwait_instr``) and the ``budget_s`` brake (callers on the 27B line only).

S109 (NF 20.10.) THE AUX STAGE -- the Q-1190b ``allow_aux`` stage of the 27B line (``dual_arena_spill``), without its
timer. METAL: the D backup wall (``PDFLIP D-WALL-HEAD ... payable_evictable=0``, 55 heads on D TP0 in the boots of 20.10.
07:26-09:47) stood on a KV arena held by D's OWN host-only leaves whose END mamba anchor lives on the host only:
``S83 VICTIM-CENSUS ... pinned=6485 tree_host_only=6130 spill_why=host_only:4/6130`` next to ``W3-HOST-LEAF SPILL
candidates=0 refused=...,aux_host_only:3`` (rc20 _081757 08:23:03Z; the same shape 08:57:05Z ``host_only:3/7145`` and
rc22 09:54:29Z ``aux_host_only:4/4908``). Every one of those leaves was refused by the single rule "an aux state lives
on the host only" -- so no claim found a victim, no device leaf could be backed up (``#1421 BACKUP-REFUSED
why=arena_claim``), the peel delivered 0 and D refused requests under X (SOLL R7: a defect, no flip reason).
THE RULE: after every plain leaf, a host-only leaf whose ONLY refusal is a host-only MAMBA anchor is a candidate too
(heap key ``(1, ...)`` behind ``(0, ...)``). It leaves like a plain one -- every KV page gets its L3 copy first
(``secure_rows_to_l3``) -- and its anchor gets its L3 copy too, by the S39 helper
(``forma_l3_subtree._secure_child_anchors``). S109b: neither the KV pages nor the anchor are written or stat'ed
here -- the write-behind thread's shared record (``mem_cache/l3_secured``) says which have their copy; the rest is
queued for its next pass and the leaf stays (``incomplete`` / ``aux_kept=child_anchor_pending``), asked again at the
next spill. An anchor in flight, not settled or not secured keeps the leaf, named (``aux_kept``). Any other host-only aux component keeps the old refusal. No switch, no timer, no reserve; on a Form A
group TP0 decides and the drop goes to the workers as a node STATE, exactly like a plain leaf (K23).

K23 (17.10.): the gate also opens on TP0 of a Form A group (``form_a_host``); its
drops are sent to the workers as node STATEs by the caller (``_w3_arena_spill``).

GATE: ``FLLIPER_PDFLIP_ENABLE_W3_SPILL_HOST_LEAVES`` (default on since the user
decision of 09.10.; on hybrid boots it is reached only through the pool that
``FLLIPER_PDFLIP_ENABLE_W3_SPILL_ANCHOR_POOL`` carries, default off) AND the local-PP
floor (``pp_slot_fidelity.FLOOR_LOCAL_PP_ATTR``: tp group of one, the tree is this
rank's own -- UD-H's scope; a TP group's replicas never edit rank-locally). ``0``:
the spill byte for byte as before.
"""
from __future__ import annotations

import heapq
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

MARK = "W3-HOST-LEAF SPILL"

_N = {"calls": 0}


def armed(cache: Any) -> bool:
    from flliper.srt.environ import envs
    from flliper.srt.pdflip import pp_slot_fidelity as _sf

    if not envs.FLLIPER_PDFLIP_ENABLE_W3_SPILL_HOST_LEAVES.get():
        return False
    if bool(_sf.enabled() and getattr(cache, _sf.FLOOR_LOCAL_PP_ATTR, False)):
        return True
    return form_a_host(cache)


def form_a_host(cache: Any) -> bool:
    """K23 (NF h16u2 17.10. 20:01:10Z, D TP0/TP1/TP2): TP0 of a Form A TP group
    (R12 role ``host``) is the one rank that decides a host drop; every worker
    follows its STATE events (``form_a_host_shadow.record_state``). The spill is
    therefore rank-uniform there: TP0 drops, the workers reconcile at the next
    broadcast. A worker never gets here (``_w3_arena_spill`` returns early and
    asks TP0)."""
    try:
        from flliper.srt.mem_cache import form_a_host_shadow as _r12

        return _r12.role() == "host"
    except Exception:  # noqa: BLE001 - no shadow module: not a Form A group
        return False


def _strict() -> bool:
    """H16P-NACHBESSERUNG A2 (``FLLIPER_PDFLIP_L3_PRESENT_STRICT``): an unpaired page (KV without its QSA index) is
    not in L3 -- the leaf stays, like on ``lost``."""
    try:
        from flliper.srt.environ import envs

        return bool(envs.FLLIPER_PDFLIP_L3_PRESENT_STRICT.get())
    except Exception:  # noqa: BLE001
        return True


def _leaf_reason(node: Any) -> str:
    """Why ``cache._is_host_leaf`` said no (the same checks in the same order, for the census only)."""
    if not getattr(node, "evicted", True):
        return "device_resident"
    if not getattr(node, "backuped", False):
        return "unbacked"
    if any(getattr(cd, "host_lock_ref", 0) > 0 for cd in node.component_data):
        return "host_locked"
    if len(getattr(node, "children", ()) or ()) > 0:
        return "has_children"
    return "not_host_leaf"


def _subtree_blocked(tree: Any, n: Any, ongoing) -> bool:
    """The V1 guards of the 27B ``pp_slot_fidelity._subtree_blocked`` (default path,
    ``end_anchor_yield=False``), which the NF line does not carry: in flight by id,
    by ``write_through_pending_id`` (a split node keeps the OLD id), a #1427 direct
    write's mamba rows in flight, and a host-backed END anchor D may not have read."""
    nid = getattr(n, "id", None)
    if nid in ongoing or getattr(n, "write_through_pending_id", None) is not None:
        return True
    if nid in (getattr(tree, "_pdflip_direct_mamba_rows", None) or {}):
        return True
    if getattr(n, "_pdflip_end_anchor", False):
        try:
            from flliper.srt.mem_cache.unified_cache_components.tree_component import ComponentType

            if n.component_data[ComponentType.MAMBA].host_value is not None:
                return True
        except Exception:  # noqa: BLE001 -- no mamba component on this tree: no END anchor to hold
            pass
    return False


#: S109: host anchors put on L3 per spill call (each ~59 MB on NF; already on disk = a stat only) -- the S39 bound
ANCHOR_SECURE_PER_CALL = 4


def _aux_host_only_kinds(cache: Any, node: Any) -> list:
    """The aux component types that keep a HOST-ONLY state on ``node``."""
    from flliper.srt.mem_cache.unified_radix_cache import _aux_components

    return [c.component_type for c in _aux_components(cache)
            if node.component_data[c.component_type].host_value is not None
            and node.component_data[c.component_type].value is None]


def _mamba_ct():
    from flliper.srt.mem_cache.unified_cache_components.tree_component import ComponentType

    return getattr(ComponentType, "MAMBA", None)


def _reason(cache: Any, node: Any, pool: Any, skip: set, allow_aux: bool = False) -> Optional[str]:
    """None = a host-only H-leaf whose host copy may leave L2 once it has an L3 copy;
    else the NAME of the first check that refused it (27B ``_reason``). ``allow_aux`` (S109, the AUX stage): a leaf whose
    only host-only aux state is its MAMBA anchor is a candidate too; any other host-only aux component still refuses."""
    from flliper.srt.mem_cache.unified_radix_cache import BASE_COMPONENT_TYPE, _aux_components

    if node is cache.root_node:
        return "root"
    if id(node) in skip:
        return "skip"
    if not cache._is_host_leaf(node):    # evicted, backuped, no children, no host lock
        return _leaf_reason(node)
    cd = node.component_data[BASE_COMPONENT_TYPE]
    hv = cd.host_value
    if hv is None or hv.numel() == 0:
        return "no_host_value"
    if cd.value is not None:
        return "device_value"
    if node.id in cache.ongoing_write_through:
        return "write_through_ongoing"    # pending write: the slots are not COMPLETE yet
    if _subtree_blocked(cache, node, cache.ongoing_write_through):
        return "blocked"
    if not getattr(node, "hash_value", None):
        return "no_hash"                  # the page count cannot be checked
    _kinds = _aux_host_only_kinds(cache, node)
    if _kinds and not (allow_aux and all(k == _mamba_ct() for k in _kinds)):
        return "aux_host_only"            # an aux state lives on the host only (anchor, park_l3)
    if any(node.component_data[c.component_type].value is not None for c in _aux_components(cache)):
        return "aux_device_value"         # K15b A4: _evict_host_leaf(ALL) would free a live device state
    if int(hv.min()) < int(getattr(pool, "staging_rows", 0)):
        return "staging"                  # staging rows are not arena slots
    return None


def _eligible(cache: Any, node: Any, pool: Any, skip: set, allow_aux: bool = False) -> bool:
    return _reason(cache, node, pool, skip, allow_aux) is None


def _secure_anchor(cache: Any, node: Any, budget: list) -> Optional[str]:
    """S109: the host-only MAMBA anchor of ``node`` gets its L3 copy (the S39 helper); None = secured, else the name
    of the refusal (the leaf stays)."""
    if not _aux_host_only_kinds(cache, node):
        return None
    try:
        from flliper.srt.pdflip import forma_l3_subtree as _fls

        r = _fls._secure_child_anchors(cache, [node], budget)
    except Exception as exc:  # noqa: BLE001 -- fail closed: the leaf keeps its anchor
        logger.warning("%s aux anchor secure raised: %r", MARK, exc)
        return "anchor_secure_raised"
    return None if r is None else str(r[0])


def spill_host_only(cache: Any, pool: Any, want: int, page_size: int, skip: set,
                    released_out: Optional[list] = None, max_secure_pages: int = 0) -> Dict[str, int]:
    """Spill host-only H-leaves until ``want`` pages were released here (group-equal
    order, see ``UnifiedRadixCache._w3_arena_spill``; node-id order where the
    arena names no slot). Returns the counts; the log line names them (27B loop, no census).

    K23 review A1 (17.10.): ``max_secure_pages`` > 0 is the REST of the caller's synchronous-L3-write budget
    (H16Q review A3, ``FLLIPER_PDFLIP_W3_FORMA_SPILL_MAX_PAGES``): a leaf whose pages do not fit what is left is
    passed over (``over_cap``), never partly written; ``secure_pages`` in the result = the pages handed to
    ``secure_rows_to_l3`` here. 0 = no bound (the local-PP floor, as before)."""
    from flliper.srt.mem_cache.unified_radix_cache import BASE_COMPONENT_TYPE

    P = max(1, int(page_size or 1))
    heap = []
    refused: Dict[str, int] = {}
    eligible = []
    from flliper.srt.mem_cache import form_a_host_shadow as _r12

    # S22 (R12 SPILL-DIVERGED class): on a Form A group TP0's drop of a host-only leaf is a STATE(exists=0) verdict; a
    # worker that has the node device-resident or with children cannot delete it and would keep a node TP0 dropped
    nodel = _r12.worker_nodelete_keys(cache) if _r12.role() == "host" else ()
    for n in cache._collect_all_nodes():
        r = _reason(cache, n, pool, skip, True)
        if r is None and nodel and _r12.node_key(n) in nodel:
            r = "worker_cannot_delete"
        if r is None:
            eligible.append(n)
        elif r != "root":
            refused[r] = refused.get(r, 0) + 1
    # NF arena fix (14.10.): the order is the W3 spill's group-equal one (fewest
    # reader references, then slot id, then node id) -- see _w3_arena_spill; a
    # leaf the arena names no slot for keeps the node-id order
    keys = cache._w3_spill_keys(pool, eligible)
    # L3-COMPLETE review A1 (16.10.): a leaf with a slot not COMPLETE in the shared header is no
    # candidate (like a pending write) -- counted apart as ``incomplete``, never as a loss
    incomplete = sum(1 for n in eligible if _incomplete(keys, n))
    # S109: plain leaves first, the aux-anchor leaves behind them (``(aux, key, id)``)
    heap = [(int(bool(_aux_host_only_kinds(cache, n))), keys[id(n)], n.id, n)
            for n in eligible if not _incomplete(keys, n)]
    candidates = len(heap)
    aux_candidates = sum(1 for h in heap if h[0])
    heapq.heapify(heap)
    released = leaves = unsecured = on_disk = written = stale = 0
    cap_used = over_cap = 0
    aux_leaves = 0
    aux_kept: Dict[str, int] = {}
    anchor_budget = [ANCHOR_SECURE_PER_CALL]
    while heap and released < want:
        _aux, _key, _id, node = heapq.heappop(heap)
        if not _eligible(cache, node, pool, skip, True) or (nodel and _r12.node_key(node) in nodel):
            # REVIEW S27 A1: a cascaded parent is checked against the workers' nodelete set like a first-level leaf
            stale += 1
            continue
        cd = node.component_data[BASE_COMPONENT_TYPE]
        if max_secure_pages > 0:
            _np = -(-int(cd.host_value.numel()) // P)
            if cap_used + _np > int(max_secure_pages):
                over_cap += 1
                continue
            cap_used += _np
        sec = pool.secure_rows_to_l3(cd.host_value)
        if int(sec.get("incomplete", 0)):
            incomplete += 1               # a slot turned out not COMPLETE: kept, no loss (review A5)
            continue
        if (int(sec.get("lost", 0)) or int(sec.get("pages", 0)) != len(node.hash_value)
                or (int(sec.get("unpaired", 0)) and _strict())):
            unsecured += 1                # no L3 copy for every page: it stays
            continue
        from flliper.srt.pdflip.read_room import l3_complete as _l3_complete

        if not _l3_complete(node, sec):
            unsecured += 1                # review readroomd B3: a page neither on disk nor written now: it stays
            continue
        is_aux = bool(_aux_host_only_kinds(cache, node))
        if is_aux:
            _aw = _secure_anchor(cache, node, anchor_budget)
            if _aw is not None:
                aux_kept[_aw] = aux_kept.get(_aw, 0) + 1   # S109: the anchor has no L3 copy (yet): the leaf stays
                continue
        on_disk += int(sec.get("on_disk", 0))
        written += int(sec.get("written", 0))
        if released_out is not None:
            # READ-ROOM review A3: the slots this leaf's rows name, read before the leaf goes
            released_out.extend(cache._w3_spill_keys(pool, [node]).slots_of(node))
        parent = node.parent
        tracker = {ct: 0 for ct in cache.tree_components}
        cache._evict_host_leaf(node, tracker)
        released += int(tracker.get(BASE_COMPONENT_TYPE, 0)) // P
        leaves += 1
        aux_leaves += int(is_aux)
        if (parent is not None and parent is not cache.root_node and _eligible(cache, parent, pool, skip, True)
                and not (nodel and _r12.node_key(parent) in nodel)):
            _pk = cache._w3_spill_keys(pool, [parent])
            if _incomplete(_pk, parent):
                incomplete += 1
            else:
                heapq.heappush(heap, (int(bool(_aux_host_only_kinds(cache, parent))), _pk[id(parent)], parent.id,
                                      parent))
    _N["calls"] += 1
    k = _N["calls"]
    if k <= 8 or k % 64 == 0 or unsecured or aux_leaves or aux_kept:
        top = sorted(refused.items(), key=lambda kv: -kv[1])[:4]
        logger.info(
            "%s n=%d want=%d released_pages=%d leaves=%d candidates=%d unsecured_kept=%d incomplete=%d stale=%d "
            "secure_pages=%d cap=%d over_cap=%d "
            "l3=on_disk:%d,written:%d refused=%s aux_candidates=%d aux_leaves=%d aux_kept=%s "
            "(a host-only leaf leaves L2 only with its L3 copy "
            "for every page; incomplete = leaves kept because a slot is not COMPLETE yet, no loss; "
            "port of the 27B Q-697c spill, local-PP floor only; S109 aux = a leaf whose host-only mamba anchor got its "
            "L3 copy too, after every plain leaf)",
            MARK, k, int(want), released, leaves, candidates, unsecured, incomplete, stale,
            cap_used, int(max_secure_pages), over_cap, on_disk, written,
            ",".join("%s:%d" % kv for kv in top) or "-", aux_candidates, aux_leaves,
            ",".join("%s:%d" % kv for kv in sorted(aux_kept.items())) or "-")
    return {"released": released, "leaves": leaves, "candidates": candidates,
            "unsecured": unsecured, "incomplete": incomplete,
            "secure_pages": cap_used, "over_cap": over_cap, "aux_leaves": aux_leaves}


def _incomplete(keys: Any, node: Any) -> bool:
    fn = getattr(keys, "incomplete", None)
    return bool(fn(node)) if callable(fn) else False
