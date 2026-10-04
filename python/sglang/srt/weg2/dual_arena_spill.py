"""Q-697c DUAL HOST-ONLY SPILL (27B NVFP4 dual C, boot
dkr27bnvfp4dual1mpsleepsharebar1fs10032243, image 54a30199b1, P PP0 22:51:19Z; the
analysis is deskq/done/1010-dual-p-kv-wait.out section 3.2).

METAL. The shared KV L2 arena of group P (720896 slots) ran full, every slot
referenced by P's OWN host tree (ARENA-REF-HOLDERS pool=FULL tree=651489
own_held=651489, 22:51:18 complete=720896 pinned=720811). From 22:51:19 every
backup was refused: '#1421 BACKUP-REFUSED why=arena_claim' (98x), then
'parent_unbacked' (220x), '#1427 ARENA-DROP need=4011 freed=0', 'PUBLISH-SWEEP
issued=0 refused=8'. The page tail of weg2-0-19 (128265 tokens) never reached the
store, D's prefetch looped 5.5 min ('PREFETCH DEFERRED store_prefix_short'),
23:02:36 CLIENT-GONE. The W3-ARENA spill made for exactly this wrote NOT ONE line
('W3-ARENA' count 0 in the P log, and in no log of the evidence trees).

TWO CAUSES, both in ``UnifiedRadixCache._w3_arena_spill``:

1. The W3 spill never ran on the 27B. Its first gate is
   ``hasattr(pool, "secure_rows_to_l3")``; the 27B's host pool is the hybrid
   ``HostPoolGroup`` (anchor = the KV arena pool) whose ``__getattr__`` forwards
   only alloc_read / alloc_write / complete_write / abort_write / pin_slots.
   ``secure_rows_to_l3`` raised AttributeError, ``hasattr`` said False, the spill
   returned 0 silently -- before its first log line, which is why the log has none.
2. Even with the pool resolved, the spill takes only nodes that still carry a
   DEVICE value ('host-only nodes stay: their host copy is their only copy on P').
   On the dual layout P gives its whole device KV away at every idle
   (``dual_p_kv_stage.on_idle``: evict + release_all), so P's tree is almost
   entirely HOST-ONLY and invisible to the spill. Their host copy is no longer
   the only copy once it has an L3 copy: the store holds the page, D reads it
   from there (arena_fill_from_disk / prefetch).

THE FIX (dual P only, ``dual_p_kv_stage.armed``):

  a. ``anchor_spill_pool``: where the pool lacks ``secure_rows_to_l3``, the spill
     uses the group's anchor host pool (the arena pool that ``alloc_write``
     already forwards to).
  b. ``spill_host_only``: after the device-resident round, a claim that is still
     short spills HOST-ONLY H-LEAVES: in node-id order (the PP ranks' trees are
     replicas, a slot is free only when every rank's reference is gone), each
     leaf first gets its L3 copy (``secure_rows_to_l3``, #257: never released
     without one -- ``lost`` != 0 or a page not COMPLETE keeps the leaf), then the
     leaf is evicted whole (``_evict_host_leaf``: its references go back, the leaf
     is deleted, a parent that became a leaf is next). A leaf that carries an aux
     host state (mamba anchor, y5h/park_l3) stays: it needs its own L3 copy and is
     another fix. The stop is NAMED: ``Q-697c DUAL HOST-ONLY SPILL`` says how many
     candidates, how many were released and why the rest stayed.

Off the dual P layout (flip, 27B INT8, NF, dual D, anything without the gate)
nothing here runs: the spill behaves byte for byte as before.
"""
from __future__ import annotations

import heapq
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

MARK = "Q-697c DUAL HOST-ONLY SPILL"

_N = {"calls": 0, "no_pool": 0}


def armed(env=None) -> bool:
    """The dual P layout only (the gate every dual fix sits behind)."""
    from sglang.srt.weg2 import dual_p_kv_stage as _dpk

    return bool(_dpk.armed(env))


def anchor_spill_pool(pool: Any, env=None) -> Optional[Any]:
    """The pool that carries ``secure_rows_to_l3`` when ``pool`` itself does not
    (the hybrid HostPoolGroup forwards only the claim calls): the group's anchor
    host pool -- the arena pool its ``alloc_write`` already forwards to. None off
    the dual P layout (the spill then stops as it always did) or without one."""
    if not armed(env):
        return None
    hp = getattr(getattr(pool, "anchor_entry", None), "host_pool", None)
    if hp is not None and hasattr(hp, "secure_rows_to_l3"):
        return hp
    _N["no_pool"] += 1
    if _N["no_pool"] <= 8 or _N["no_pool"] % 64 == 0:
        logger.warning(
            "%s n=%d STOP no_spill_pool: the host pool %s has no secure_rows_to_l3 and no anchor "
            "host pool that has -- nothing to spill with", MARK, _N["no_pool"], type(pool).__name__)
    return None


def _eligible(cache: Any, node: Any, pool: Any, skip: set) -> bool:
    """A host-only H-leaf whose host copy may leave L2 once it has an L3 copy."""
    from sglang.srt.mem_cache.unified_radix_cache import BASE_COMPONENT_TYPE, _aux_components

    if node is cache.root_node or id(node) in skip:
        return False
    if not cache._is_host_leaf(node):    # evicted, backuped, no children, no host lock
        return False
    cd = node.component_data[BASE_COMPONENT_TYPE]
    hv = cd.host_value
    if hv is None or cd.value is not None or hv.numel() == 0:
        return False
    if node.id in cache.ongoing_write_through:
        return False                      # pending write: the slots are not COMPLETE yet
    if not getattr(node, "hash_value", None):
        return False                      # the page count cannot be checked
    if any(
        node.component_data[c.component_type].host_value is not None
        and node.component_data[c.component_type].value is None
        for c in _aux_components(cache)
    ):
        return False                      # an aux state lives on the host only (anchor, park_l3)
    if int(hv.min()) < int(getattr(pool, "staging_rows", 0)):
        return False                      # staging rows are not arena slots
    return True


def spill_host_only(cache: Any, pool: Any, want: int, page_size: int, skip: set) -> Dict[str, int]:
    """Spill host-only H-leaves of a dual P tree until ``want`` pages were
    released here (node-id order). Returns the counts; the log line names them."""
    from sglang.srt.mem_cache.unified_radix_cache import BASE_COMPONENT_TYPE

    P = max(1, int(page_size or 1))
    heap = [(n.id, n) for n in cache._collect_all_nodes() if _eligible(cache, n, pool, skip)]
    candidates = len(heap)
    heapq.heapify(heap)
    released = leaves = unsecured = on_disk = written = 0
    while heap and released < want:
        _id, node = heapq.heappop(heap)
        if not _eligible(cache, node, pool, skip):
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
        logger.info(
            "%s n=%d want=%d released_pages=%d leaves=%d candidates=%d unsecured_kept=%d l3=on_disk:%d,"
            "written:%d (a host-only leaf leaves L2 only with its L3 copy for every page; the dual P tree "
            "is host-only after each idle release)",
            MARK, k, int(want), released, leaves, candidates, unsecured, on_disk, written)
    return {"released": released, "leaves": leaves, "candidates": candidates, "unsecured": unsecured}
