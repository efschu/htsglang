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
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

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
    from sglang.srt.weg2 import pp_slot_fidelity as _sf

    if _sf._subtree_blocked(cache, node, cache.ongoing_write_through):
        # the V1 guards (one source): write_through_pending_id (a split node keeps the OLD id), the
        # #1427 direct write's mamba rows in flight, a host-backed END anchor D may not have read yet
        return False
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


def spill_host_only(cache: Any, pool: Any, want: int, page_size: int, skip: set,
                    budget_s: float = 0.0) -> Dict[str, int]:
    """Spill host-only H-leaves of a dual P tree until ``want`` pages were
    released here (node-id order). Returns the counts; the log line names them.

    ``budget_s`` > 0 (V2 ARENA-TRIM only; the claim-driven callers pass none): a wall-clock
    brake -- the L3 copy of a leaf is file I/O on the scheduler thread, so the loop stops
    before the NEXT leaf once the budget is spent (``braked`` = 1 in the result)."""
    from sglang.srt.mem_cache.unified_radix_cache import BASE_COMPONENT_TYPE

    P = max(1, int(page_size or 1))
    heap = [(n.id, n) for n in cache._collect_all_nodes() if _eligible(cache, n, pool, skip)]
    candidates = len(heap)
    heapq.heapify(heap)
    released = leaves = unsecured = on_disk = written = braked = 0
    t_end = (time.monotonic() + float(budget_s)) if budget_s and budget_s > 0 else None
    while heap and released < want:
        if t_end is not None and time.monotonic() >= t_end:
            braked = 1
            break
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
    return {"released": released, "leaves": leaves, "candidates": candidates, "unsecured": unsecured,
            "braked": braked}


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


# ---------------------------------------------------------------- V2 ARENA-TRIM
# Q-1500 UD-V2 (27B NVFP4 dual y9d3, boot ...10040710 @c2f6e819cb, P PP0 07:18:49Z; analysis
# deskq/done/1290-dual-y9d3-oom.out Gl. 7-10, option V2). The shared KV arena filled to 98-99 %
# pinned within five minutes of burst load and the holder was P itself: every P leg leaves its
# whole ~31k pages as host-only leaves in P's tree, P is never idle under load (so the idle
# release never runs), and every Q-697c spill is bound to the spilling rank's OWN claim refusal.
# Q-1190 made D give back ~91k pages and arena_pinned rose anyway: a slot frees only when EVERY
# holder (3 P ranks + 3 D ranks) gave its reference back, and the ranks gave back different
# subsets at different times. At the wall every backup was refused (BACKUP-REFUSED why=arena_claim)
# and the P pool died on its last device leaf (V1 turns that death into a cache loss; V2 keeps the
# arena below the wall in the first place).
#
# THE FIX (dual P only: ``armed`` + SGLANG_WEG2_DUAL_ARENA_TRIM, default 1; 0 = off):
#   * PP0 reads the shared arena's fill -- ``arena_pinned / slots`` from the shared header
#     (``ShmArena.ref_census``: the one number every rank of both groups would read the same;
#     ``stats()`` is the O(1) pre-filter: pinned <= complete) -- at most once per
#     ..._MIN_S. Above ``_HI`` (0.90) a trim episode starts and runs until the fill is <= ``_LO``
#     (0.80): each decision orders ``pinned - LO*slots`` pages, at most ``_MAX_PAGES`` per
#     decision (the L3 write of a leaf is file I/O on the scheduler thread -- "HiCache bremst nie").
#   * RANK CONGRUENCE (RAENGE-NIE-UNEINS). The decision is PP0's ALONE and rides the request wire
#     in band, like PP0's pass clock (``anchor_tails.Weg2BurstClock``): PP0 puts ONE
#     ``Weg2DualArenaTrim(seq, want)`` on the list it SENDS in pass m, every follower relays it and
#     takes it off before dispatch, and every stage (PP0 after its send, a follower in its pass m
#     after the relay) executes the SAME order -- the same ``want``, the same node-id order over
#     replicated trees -- at the same logical pass. No follower reads the header, a clock or an
#     env-threshold of its own, so the trigger cannot differ between P ranks. What may still
#     differ is only what a rank CAN give (a node that rank has locked / not yet COMPLETE / not
#     yet in its tree): it keeps that leaf, the slot stays pinned by that one rank and frees at
#     the next order -- never an inconsistent tree: a host-only leaf with its L3 copy for every
#     page leaving a tree is the Q-697c operation, which the claim path already ran per rank,
#     without peers, on the metal.
#   * Each leaf is released only after its L3 copy (``spill_host_only``, #257); a leaf without one
#     (``unsecured``) stays. A wall-clock budget per order (``_BUDGET_S``) brakes the loop before
#     the next leaf; the rest follows with the next order.
#   * Backoff: an order PP0 itself could not serve (released 0) holds the next decision for
#     ``_EMPTY_BACKOFF_S`` (the D_YIELD_EMPTY_BACKOFF_S rule).
#   * Group D (Q-1190): every order also posts its page need for D (``post_need``, the ``.dualneed``
#     file; max semantics, D's TP0 takes it in its next tick and every D rank gives host-only leaves,
#     L3 copy first): a slot frees only when D's references are gone too, and the oldest pages P
#     gives are the ones D adopted first. ..._POST_D=0 leaves D alone.
# Off the dual P layout (flip, 27B INT8, NF, dual D, no cap) NOTHING here runs: no header read, no
# object on the wire, the follower scan finds none.

TRIM_MARK = "Q-1500 UD-V2 DUAL ARENA-TRIM"
TRIM_ENV = "SGLANG_WEG2_DUAL_ARENA_TRIM"
TRIM_HI_ENV = "SGLANG_WEG2_DUAL_ARENA_TRIM_HI"
TRIM_LO_ENV = "SGLANG_WEG2_DUAL_ARENA_TRIM_LO"
TRIM_MIN_S_ENV = "SGLANG_WEG2_DUAL_ARENA_TRIM_MIN_S"
TRIM_MAX_PAGES_ENV = "SGLANG_WEG2_DUAL_ARENA_TRIM_MAX_PAGES"
TRIM_BUDGET_S_ENV = "SGLANG_WEG2_DUAL_ARENA_TRIM_BUDGET_S"
TRIM_EMPTY_BACKOFF_S_ENV = "SGLANG_WEG2_DUAL_ARENA_TRIM_EMPTY_BACKOFF_S"
TRIM_POST_D_ENV = "SGLANG_WEG2_DUAL_ARENA_TRIM_POST_D"       # 1 (default): every order also posts the need for group D
TRIM_STALE_N_ENV = "SGLANG_WEG2_DUAL_ARENA_TRIM_STALE_N"
TRIM_PAUSE_S_ENV = "SGLANG_WEG2_DUAL_ARENA_TRIM_PAUSE_S"
TRIM_HI_DEFAULT = 0.90
TRIM_LO_DEFAULT = 0.80
#: PP0 reads the arena header / decides at most this often (also the distance between two orders)
TRIM_MIN_S_DEFAULT = 1.0
#: pages one order asks for at most (an episode from 0.97 to 0.80 is several orders)
TRIM_MAX_PAGES_DEFAULT = 8192
#: wall-clock brake of one order's loop (the L3 write is file I/O on the scheduler thread)
TRIM_BUDGET_S_DEFAULT = 0.25
#: after an order PP0 could not serve at all, no decision for this long
TRIM_EMPTY_BACKOFF_S_DEFAULT = 2.0
#: this many orders in a row that did not lower arena_pinned pause the episode (the holder is not P's tree)
TRIM_STALE_N_DEFAULT = 3
#: ... for this long
TRIM_PAUSE_S_DEFAULT = 30.0


class Weg2DualArenaTrim(NamedTuple):
    """PP0's order for one pass, riding the request wire: give back ``want`` pages of host-only
    leaves (node-id order, L3 copy first). ``fill_ppm`` is PP0's reading when it decided (for the
    log line only -- no rank branches on it)."""

    seq: int
    want: int
    fill_ppm: int


_T: Dict[str, Any] = {
    "active": False, "next_t": 0.0, "seq": 0, "episode_cmds": 0, "orders": 0, "execs": 0,
    "reads": 0, "census_reads": 0, "no_arena": 0, "need_posts": 0, "last_fill": 0.0, "decided_t": 0.0, "stops": 0, "errors": 0,
    "stale": 0, "prev_pinned": None, "prev_want": 0, "pauses": 0,
}


def _reset_trim_for_tests() -> None:
    _T.update(active=False, next_t=0.0, seq=0, episode_cmds=0, orders=0, execs=0, reads=0,
              census_reads=0, no_arena=0, need_posts=0, last_fill=0.0, decided_t=0.0, stops=0, errors=0,
              stale=0, prev_pinned=None, prev_want=0, pauses=0)


def trim_enabled(env=None) -> bool:
    """The dual P layout (``dual_p_kv_stage.armed``) AND the switch; the one gate of the whole V2 part."""
    e = os.environ if env is None else env
    if str(e.get(TRIM_ENV, "1")).strip() == "0":
        return False
    return armed(env)


def _f(env, key: str, default: float, lo: float, hi: float) -> float:
    try:
        v = float(env.get(key, "") or default)
    except ValueError:
        v = default
    return min(max(v, lo), hi)


def trim_cfg(env=None) -> Dict[str, float]:
    """The knobs, clamped: 0.05 < LO < HI <= 1.0 (a LO at or above HI would never end an episode)."""
    e = os.environ if env is None else env
    hi = _f(e, TRIM_HI_ENV, TRIM_HI_DEFAULT, 0.10, 1.0)
    lo = _f(e, TRIM_LO_ENV, TRIM_LO_DEFAULT, 0.05, 1.0)
    lo = min(lo, hi - 0.02)
    return {
        "hi": hi, "lo": max(0.01, lo),
        "min_s": _f(e, TRIM_MIN_S_ENV, TRIM_MIN_S_DEFAULT, 0.0, 600.0),
        "max_pages": int(_f(e, TRIM_MAX_PAGES_ENV, TRIM_MAX_PAGES_DEFAULT, 1, 1 << 30)),
        "budget_s": _f(e, TRIM_BUDGET_S_ENV, TRIM_BUDGET_S_DEFAULT, 0.0, 60.0),
        "backoff_s": _f(e, TRIM_EMPTY_BACKOFF_S_ENV, TRIM_EMPTY_BACKOFF_S_DEFAULT, 0.0, 600.0),
        "stale_n": int(_f(e, TRIM_STALE_N_ENV, TRIM_STALE_N_DEFAULT, 1, 1000)),
        "pause_s": _f(e, TRIM_PAUSE_S_ENV, TRIM_PAUSE_S_DEFAULT, 0.0, 3600.0),
    }


def _tree_pool(sched) -> Optional[Any]:
    tree = getattr(sched, "tree_cache", None)
    get = getattr(tree, "_weg2_direct_pool", None)
    return _spill_pool(get()) if callable(get) else None


def read_fill(arena: Any, trimming: bool, hi: float) -> Tuple[Optional[int], int]:
    """``(pinned, slots)`` of the shared arena. ``pinned`` = COMPLETE slots with a reader reference
    (the ``arena_pinned`` of ARENA-REF-HOLDERS, the wall's number). The strided header census is
    taken only when it can matter: ``stats()`` (O(1) counters) bounds it from above (pinned <=
    complete); outside an episode and below HI by that bound, ``pinned`` is None."""
    st = arena.stats()
    slots = max(1, int(st["slots"]))
    _T["reads"] += 1
    if not trimming and int(st["complete"]) <= hi * slots:
        return None, slots
    _T["census_reads"] += 1
    return int(arena.ref_census()[0]), slots


def own_held_pages(pool: Any, tree: Any) -> int:
    """The arena pages THIS process's references name (the ARENA-REF-HOLDERS ``own_held``: the reference
    ledger's sum; without a ledger the host pages of the tree). The most P can give back by itself."""
    led = getattr(getattr(pool, "arena", None), "_ledger", None)
    if led is not None:
        try:
            return int(led.held.sum())
        except Exception:  # noqa: BLE001 - fall back to the tree
            pass
    from sglang.srt.mem_cache.unified_radix_cache import BASE_COMPONENT_TYPE

    total = 0
    for n in tree._collect_all_nodes():
        hv = n.component_data[BASE_COMPONENT_TYPE].host_value if n is not tree.root_node else None
        if hv is not None:
            total += int(hv.numel())
    return total


def pp0_decide(sched, now: Optional[float] = None, env=None) -> Optional[Weg2DualArenaTrim]:
    """PP0, once per pass (``pp0_stamp``): the order of this pass, or None. Reads the header at
    most once per MIN_S; None off the gate, on a follower, without an arena."""
    if not trim_enabled(env):
        return None
    ps = getattr(sched, "ps", None)
    if int(getattr(ps, "pp_rank", 0) or 0) != 0 or int(getattr(ps, "pp_size", 1) or 1) < 2:
        return None
    t = time.monotonic() if now is None else float(now)
    if t < _T["next_t"]:
        return None
    pool = _tree_pool(sched)
    arena = getattr(pool, "arena", None)
    if arena is None:
        _T["no_arena"] += 1
        if _T["no_arena"] <= 3:
            logger.warning("%s n=%d STOP no_arena: the P tree has no arena pool -- no trim", TRIM_MARK,
                           _T["no_arena"])
        _T["next_t"] = t + 30.0
        return None
    cfg = trim_cfg(env)
    _T["next_t"] = t + cfg["min_s"]
    _T["decided_t"] = t
    try:
        pinned, slots = read_fill(arena, bool(_T["active"]), cfg["hi"])
    except Exception as exc:  # noqa: BLE001 - an instrument never breaks the pass
        logger.warning("%s STOP header_read_failed: %r", TRIM_MARK, exc)
        _T["next_t"] = t + 30.0
        return None
    if pinned is None:
        return None
    fill = pinned / float(slots)
    _T["last_fill"] = fill
    if _T["active"] and fill <= cfg["lo"]:
        _T["active"] = False
        _T["stale"], _T["prev_pinned"], _T["prev_want"] = 0, None, 0
        logger.info("%s DONE orders=%d fill=%.3f pinned=%d slots=%d (<= LO %.2f: the arena is back "
                    "below the wall)", TRIM_MARK, _T["episode_cmds"], fill, pinned, slots, cfg["lo"])
        return None
    if not _T["active"]:
        if fill <= cfg["hi"]:
            return None
        _T["active"], _T["episode_cmds"] = True, 0
        _T["stale"], _T["prev_pinned"], _T["prev_want"] = 0, None, 0
        logger.info("%s START fill=%.3f pinned=%d slots=%d (> HI %.2f): group P gives back host-only "
                    "leaves down to LO %.2f, L3 copy first", TRIM_MARK, fill, pinned, slots, cfg["hi"],
                    cfg["lo"])
    else:
        # OVER-TRIM BRAKE (review 09:04Z): the order is sized by the arena's pinned count, but the pin
        # can be D's (or a hand-off's). Orders that do not lower it are P's tree emptied for nothing.
        prev = _T["prev_pinned"]
        if prev is not None:
            if pinned > prev - max(1, int(_T["prev_want"]) // 4):
                _T["stale"] += 1
            else:
                _T["stale"] = 0
        if _T["stale"] >= cfg["stale_n"]:
            _T["active"] = False
            _T["next_t"] = t + cfg["pause_s"]
            _T["pauses"] += 1
            logger.warning("%s PAUSED n=%d fill=%.3f pinned=%d slots=%d: %d orders in a row did not lower "
                           "arena_pinned (last order %d pages, pinned %s -> %d) -- the holder is not P's "
                           "tree (group D / hand-off pins); no order for %.0f s, a new episode starts "
                           "when the fill is above HI again", TRIM_MARK, _T["pauses"], fill, pinned, slots,
                           _T["stale"], int(_T["prev_want"]), prev, pinned, cfg["pause_s"])
            _T["stale"], _T["prev_pinned"], _T["prev_want"] = 0, None, 0
            return None
    # ... and P never orders more than its own references name (a pin of D's is not P's to clear)
    own = own_held_pages(pool, getattr(sched, "tree_cache", None))
    want = min(int(cfg["max_pages"]), max(1, pinned - int(cfg["lo"] * slots)), max(1, own))
    _T["prev_pinned"], _T["prev_want"] = pinned, want
    if str(os.environ.get(TRIM_POST_D_ENV, "1") if env is None else env.get(TRIM_POST_D_ENV, "1")).strip() != "0":
        # the same beat for group D (Q-1190): a slot frees only when D's references are gone too, and
        # the pages P gives back first (oldest) are the ones D adopted first (y9d3: D's yield alone
        # freed nothing because P still held them -- and the other way round)
        if post_need(pool, want, env):
            _T["need_posts"] += 1
    _T["seq"] += 1
    _T["episode_cmds"] += 1
    _T["orders"] += 1
    return Weg2DualArenaTrim(int(_T["seq"]), int(want), int(round(fill * 1_000_000)))


def pp0_stamp(sched, wire_reqs: Sequence[Any], now: Optional[float] = None, env=None) -> Tuple[List[Any], Optional[Weg2DualArenaTrim]]:
    """PP0, before the chain send: ``(list to SEND, order or None)``. The dispatched list stays the
    caller's (the ``weg2_store_told.pp0_publish`` convention). Off the gate: ``wire_reqs`` itself."""
    try:
        cmd = pp0_decide(sched, now, env)
    except Exception:  # noqa: BLE001 - an optional trim never takes the pass (and the rank) down
        _T["errors"] = int(_T.get("errors", 0)) + 1
        if _T["errors"] <= 8 or _T["errors"] % 64 == 0:
            logger.exception("%s n=%d STOP decide_failed: no order this pass", TRIM_MARK, _T["errors"])
        _T["next_t"] = time.monotonic() + 30.0
        return wire_reqs, None
    if cmd is None:
        return wire_reqs, None
    return list(wire_reqs or ()) + [cmd], cmd


def without_trim_order(recv_reqs: Sequence[Any]) -> List[Any]:
    """The list as the request trace should see it (an order is not a request)."""
    return [r for r in (recv_reqs or ()) if not isinstance(r, Weg2DualArenaTrim)]


def follower_absorb(sched, recv_reqs: List[Any], env=None) -> List[Any]:
    """A follower, after relaying the list onward and before dispatch: take PP0's order(s) off it and
    execute them (in seq order). The list is returned as is when it carries none."""
    if not recv_reqs:
        return recv_reqs
    cmds = [r for r in recv_reqs if isinstance(r, Weg2DualArenaTrim)]
    if not cmds:
        return recv_reqs
    rest = [r for r in recv_reqs if not isinstance(r, Weg2DualArenaTrim)]
    for c in sorted(cmds, key=lambda c: c.seq):
        execute(sched, c, env)
    return rest


def execute(sched, cmd: Weg2DualArenaTrim, env=None) -> Dict[str, int]:
    """Every P stage with the SAME order: spill host-only H-leaves of this stage's tree until
    ``cmd.want`` pages were released here (node-id order, L3 copy first, budget-braked)."""
    none = {"released": 0, "leaves": 0, "candidates": 0, "unsecured": 0, "braked": 0}
    if not trim_enabled(env):
        return none
    from sglang.srt.mem_cache import form_a_host_shadow as _r12

    tree = getattr(sched, "tree_cache", None)
    pool = _tree_pool(sched)
    rank = int(getattr(getattr(sched, "ps", None), "pp_rank", 0) or 0)
    if pool is None or _r12.role() is not None:
        _T["stops"] = int(_T.get("stops", 0)) + 1
        if _T["stops"] <= 8 or _T["stops"] % 64 == 0:
            logger.warning("%s n=%d STOP rank=%d seq=%d no_spill_pool_or_form_a: nothing given back here (the "
                           "order is not executed on this stage)", TRIM_MARK, _T["stops"], rank, cmd.seq)
        return none
    cfg = trim_cfg(env)
    t0 = time.monotonic()
    try:
        got = spill_host_only(tree, pool, int(cmd.want), int(getattr(tree, "page_size", 1) or 1), set(),
                              budget_s=cfg["budget_s"])
    except Exception:  # noqa: BLE001 - L3 I/O or a leaf eviction failed: this order ends here, the
        # next one starts from whatever tree state each leaf's own eviction left (every leaf is evicted
        # whole or not at all); a raise would end the rank, which is what V2 exists to prevent
        _T["errors"] = int(_T.get("errors", 0)) + 1
        if _T["errors"] <= 8 or _T["errors"] % 64 == 0:
            logger.exception("%s n=%d STOP execute_failed rank=%d seq=%d", TRIM_MARK, _T["errors"], rank, cmd.seq)
        return none
    dt = time.monotonic() - t0
    _T["execs"] += 1
    k = _T["execs"]
    if rank == 0 and got["released"] == 0:
        # relative to PP0's decision clock (the same clock ``next_t`` runs on)
        _T["next_t"] = max(_T["next_t"], float(_T.get("decided_t", 0.0)) + cfg["backoff_s"])
    if k <= 8 or k % 64 == 0 or got["released"] == 0 or got.get("braked"):
        logger.info("%s n=%d rank=%d seq=%d want=%d released_pages=%d leaves=%d candidates=%d "
                    "unsecured_kept=%d braked=%d took_ms=%.1f fill=%.3f (PP0's order, the same on every P "
                    "stage: host-only leaves, node-id order, L3 copy first; the slot frees when every "
                    "holder of both groups gave its reference back)", TRIM_MARK, k, rank, cmd.seq,
                    int(cmd.want), got["released"], got["leaves"], got["candidates"], got["unsecured"],
                    int(got.get("braked", 0)), dt * 1000.0, cmd.fill_ppm / 1_000_000.0)
    return got
