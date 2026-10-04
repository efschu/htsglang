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
nothing of Q-697c runs: the spill behaves byte for byte as before. The Q-1190 part
at the end (D-ARENA-YIELD) runs on the dual layout only (either group).
"""
from __future__ import annotations

import heapq
import logging
import os
import struct
import time
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


# ---------------------------------------------------------------- Q-1190 D-ARENA-YIELD
# Q-1190 (27B NVFP4 dual y9d1, boot dkr27bnvfp4dual1mpsleepsharebar1fs10040258 @507edf614c,
# 03:07-03:16Z; analysis deskq/done/1190-dual-wedge.out). The KV L2 arena is ONE file
# shared by group P and group D (arena-32768.bin, 720896 slots). Q-697c made P give its
# own references back (P PP0 own_held 574601 -> 90262), but D's tree held the arena:
# 'ARENA-REF-HOLDERS pool=FULL tree=675527 tree_in_use=0 arena_pinned=694560' on every D
# rank -- the pages D adopted from P's hand-offs, kept as host-only leaves after each
# cache yield, used by no request. A slot frees only when no rank of either group holds
# it, so P's spill freed nothing ('#1427 ARENA-DROP freed=0' 2893x), every P claim
# scanned the full arena and was refused (PASS-STALL 871x, a 31k prompt took 67-188 s),
# P's hand-off pages and Mamba anchors did not reach the store, D refused them
# (W50 Weg2TpPrefillExceeded) and P prefilled them a second time (W53). P had no way to
# tell D, and the Q-697c spill is dual P only.
#
# THE FIX (dual layout only): a claim the arena refused after the W3/Q-697c round posts
# its page need next to the shared arena (``<arena>.dualneed``, max semantics, like
# R12's ``.w3need``); D's TP0 takes it in ``dual_d_kv_stage.tick`` and the need rides the
# tick's existing group collective, so every D rank spills the SAME number of pages of
# HOST-ONLY H-leaves (``spill_host_only``: L3 copy first, node-id order). Named lines:
# ``Q-1190 DUAL D-ARENA-YIELD`` (D) and ``Q-1190 DUAL ARENA-NEED POSTED`` (the refusing
# side). Off the dual layout nothing here runs and no file is written.

YIELD_MARK = "Q-1190 DUAL D-ARENA-YIELD"
POST_MARK = "Q-1190 DUAL ARENA-NEED POSTED"
NEED_SUFFIX = ".dualneed"
#: pages one D yield moves at least (a P claim is one node's pages; a yield that frees only
#: that makes every next claim of the same prompt wait for the next D tick)
D_YIELD_MIN_PAGES = 4096
#: after a yield that released nothing, D's TP0 does not take a request for this long
D_YIELD_EMPTY_BACKOFF_S = 0.5

_Y = {"posts": 0, "yields": 0, "no_pool": 0, "quiet_until": 0.0}


def armed_any(env=None) -> bool:
    """The dual layout, either group (P with its KV cap, D with its KV cap)."""
    from sglang.srt.weg2 import dual_d_kv_stage as _ddk

    return armed(env) or bool(_ddk.armed(env))


def _spill_pool(pool: Any) -> Optional[Any]:
    """The pool that carries ``secure_rows_to_l3`` and the arena: ``pool`` itself, else
    the hybrid group's anchor host pool. Called behind the gate only."""
    if pool is None:
        return None
    if hasattr(pool, "secure_rows_to_l3") and getattr(pool, "arena", None) is not None:
        return pool
    hp = getattr(getattr(pool, "anchor_entry", None), "host_pool", None)
    if hp is not None and hasattr(hp, "secure_rows_to_l3") and getattr(hp, "arena", None) is not None:
        return hp
    return None


def _need_path(pool: Any) -> Optional[str]:
    sp = _spill_pool(pool)
    path = getattr(getattr(sp, "arena", None), "path", None)
    return (str(path) + NEED_SUFFIX) if path else None


def post_need(pool: Any, pages: int, env=None) -> bool:
    """A refused arena claim of ``pages`` pages (after this rank's own spill round):
    leave the need next to the shared arena for group D. False off the gate (nothing
    read, nothing written) or when there is no arena path."""
    if not armed_any(env) or int(pages) <= 0:
        return False
    path = _need_path(pool)
    if path is None:
        return False
    try:
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            raw = os.pread(fd, 8, 0)
            cur = struct.unpack("<q", raw)[0] if len(raw) == 8 else 0
            if int(pages) > cur:
                os.pwrite(fd, struct.pack("<q", int(pages)), 0)
        finally:
            os.close(fd)
    except OSError as exc:
        logger.warning("%s STOP post_failed path=%s: %r", POST_MARK, path, exc)
        return False
    _Y["posts"] += 1
    k = _Y["posts"]
    if k <= 8 or k % 256 == 0:
        logger.info("%s n=%d pages=%d path=%s (the shared KV arena refused this claim after the "
                    "rank's own spill; group D's tick takes it)", POST_MARK, k, int(pages), path)
    return True


def d_take_need(sched, env=None) -> int:
    """D TP0, once per tick: the largest pending need (pages), cleared. 0 off the D gate,
    without a tree pool, or while the last yield found nothing (backoff)."""
    from sglang.srt.weg2 import dual_d_kv_stage as _ddk

    if not _ddk.armed(env):
        return 0
    if time.monotonic() < _Y["quiet_until"]:
        return 0
    tree = getattr(sched, "tree_cache", None)
    get = getattr(tree, "_weg2_direct_pool", None)
    path = _need_path(get()) if callable(get) else None
    if path is None:
        return 0
    try:
        fd = os.open(path, os.O_RDWR)
    except FileNotFoundError:
        return 0                          # nobody posted yet
    try:
        raw = os.pread(fd, 8, 0)
        cur = struct.unpack("<q", raw)[0] if len(raw) == 8 else 0
        if cur:
            os.pwrite(fd, struct.pack("<q", 0), 0)
        return max(0, int(cur))
    finally:
        os.close(fd)


def d_yield_arena(sched, need: int) -> Dict[str, int]:
    """Every D rank, with the GROUP need (the same number on every rank): spill host-only
    H-leaves of D's tree until ``max(need, D_YIELD_MIN_PAGES)`` pages were released here.
    The stop is named when the tree has no spill pool."""
    tree = getattr(sched, "tree_cache", None)
    get = getattr(tree, "_weg2_direct_pool", None)
    pool = _spill_pool(get()) if callable(get) else None
    if pool is None:
        _Y["no_pool"] += 1
        if _Y["no_pool"] <= 8 or _Y["no_pool"] % 64 == 0:
            logger.warning("%s n=%d STOP no_spill_pool need=%d: D's tree has no arena pool with "
                           "secure_rows_to_l3 -- nothing yielded", YIELD_MARK, _Y["no_pool"], int(need))
        return {"released": 0, "leaves": 0, "candidates": 0, "unsecured": 0}
    want = max(int(need), D_YIELD_MIN_PAGES)
    got = spill_host_only(tree, pool, want, int(getattr(tree, "page_size", 1) or 1), set())
    if got["released"] == 0:
        _Y["quiet_until"] = time.monotonic() + D_YIELD_EMPTY_BACKOFF_S
    _Y["yields"] += 1
    k = _Y["yields"]
    if k <= 8 or k % 64 == 0 or got["released"] == 0:
        logger.info("%s n=%d need=%d want=%d released_pages=%d leaves=%d candidates=%d unsecured_kept=%d "
                    "(a claim on the shared KV arena was refused; D gives back host-only leaves it holds "
                    "but no request uses, L3 copy first)", YIELD_MARK, k, int(need), want, got["released"],
                    got["leaves"], got["candidates"], got["unsecured"])
    return got
