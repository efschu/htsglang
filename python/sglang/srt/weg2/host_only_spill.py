"""W3-HOST-LEAF: the W3 arena spill also releases HOST-ONLY H-leaves on NF P -- a
PORT of the 27B line's Q-697c (b) ``dual_arena_spill.spill_host_only`` (27B integ
86ff356d0d, python/sglang/srt/weg2/dual_arena_spill.py; its file head has the 27B
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

NOT PORTED: the #1500i census (``dual_pkvwait_instr``), the Q-1190b ``allow_aux``
stage and the ``budget_s`` brake (callers on the 27B line only).

GATE: ``SGLANG_WEG2_ENABLE_W3_SPILL_HOST_LEAVES`` (default off, like
``SGLANG_WEG2_ENABLE_W3_SPILL_ANCHOR_POOL`` that carries the pool) AND the local-PP
floor (``pp_slot_fidelity.FLOOR_LOCAL_PP_ATTR``: tp group of one, the tree is this
rank's own -- UD-H's scope; a TP group's replicas never edit rank-locally). Off:
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
    from sglang.srt.environ import envs
    from sglang.srt.weg2 import pp_slot_fidelity as _sf

    if not envs.SGLANG_WEG2_ENABLE_W3_SPILL_HOST_LEAVES.get():
        return False
    return bool(_sf.enabled() and getattr(cache, _sf.FLOOR_LOCAL_PP_ATTR, False))


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
    if nid in (getattr(tree, "_weg2_direct_mamba_rows", None) or {}):
        return True
    if getattr(n, "_weg2_end_anchor", False):
        try:
            from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType

            if n.component_data[ComponentType.MAMBA].host_value is not None:
                return True
        except Exception:  # noqa: BLE001 -- no mamba component on this tree: no END anchor to hold
            pass
    return False


def _reason(cache: Any, node: Any, pool: Any, skip: set) -> Optional[str]:
    """None = a host-only H-leaf whose host copy may leave L2 once it has an L3 copy;
    else the NAME of the first check that refused it (27B ``_reason``, allow_aux=False)."""
    from sglang.srt.mem_cache.unified_radix_cache import BASE_COMPONENT_TYPE, _aux_components

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
    if any(
        node.component_data[c.component_type].host_value is not None
        and node.component_data[c.component_type].value is None
        for c in _aux_components(cache)
    ):
        return "aux_host_only"            # an aux state lives on the host only (anchor, park_l3)
    if int(hv.min()) < int(getattr(pool, "staging_rows", 0)):
        return "staging"                  # staging rows are not arena slots
    return None


def _eligible(cache: Any, node: Any, pool: Any, skip: set) -> bool:
    return _reason(cache, node, pool, skip) is None


def spill_host_only(cache: Any, pool: Any, want: int, page_size: int, skip: set) -> Dict[str, int]:
    """Spill host-only H-leaves until ``want`` pages were released here (node-id
    order). Returns the counts; the log line names them (27B loop, no census)."""
    from sglang.srt.mem_cache.unified_radix_cache import BASE_COMPONENT_TYPE

    P = max(1, int(page_size or 1))
    heap = []
    refused: Dict[str, int] = {}
    for n in cache._collect_all_nodes():
        r = _reason(cache, n, pool, skip)
        if r is None:
            heap.append((n.id, n))
        elif r != "root":
            refused[r] = refused.get(r, 0) + 1
    candidates = len(heap)
    heapq.heapify(heap)
    released = leaves = unsecured = on_disk = written = stale = 0
    while heap and released < want:
        _id, node = heapq.heappop(heap)
        if not _eligible(cache, node, pool, skip):
            stale += 1
            continue
        cd = node.component_data[BASE_COMPONENT_TYPE]
        sec = pool.secure_rows_to_l3(cd.host_value)
        if int(sec.get("lost", 0)) or int(sec.get("pages", 0)) != len(node.hash_value):
            unsecured += 1                # no L3 copy for every page, or a page not COMPLETE: it stays
            continue
        on_disk += int(sec.get("on_disk", 0))
        written += int(sec.get("written", 0))
        parent = node.parent
        tracker = {ct: 0 for ct in cache.tree_components}
        cache._evict_host_leaf(node, tracker)
        released += int(tracker.get(BASE_COMPONENT_TYPE, 0)) // P
        leaves += 1
        if parent is not None and parent is not cache.root_node and _eligible(cache, parent, pool, skip):
            heapq.heappush(heap, (parent.id, parent))
    _N["calls"] += 1
    k = _N["calls"]
    if k <= 8 or k % 64 == 0 or unsecured:
        top = sorted(refused.items(), key=lambda kv: -kv[1])[:4]
        logger.info(
            "%s n=%d want=%d released_pages=%d leaves=%d candidates=%d unsecured_kept=%d stale=%d "
            "l3=on_disk:%d,written:%d refused=%s (a host-only leaf leaves L2 only with its L3 copy "
            "for every page; port of the 27B Q-697c spill, local-PP floor only)",
            MARK, k, int(want), released, leaves, candidates, unsecured, stale, on_disk, written,
            ",".join("%s:%d" % kv for kv in top) or "-")
    return {"released": released, "leaves": leaves, "candidates": candidates,
            "unsecured": unsecured}
