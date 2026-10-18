"""READ-ROOM 1015 (NF h16p, zero avoidable tokens): an L3 -> L2 read that finds the shared arena FULL
asks the trees that pin it to give their L3-backed copies back -- it never ends the prefix there.

THE GAP (boots ..._226aad7105_1015_{115748,122948,124245,133855}, P): 46 ``PDFLIP-READ-DEPTH cause=arena_full``
cuts on PP0 in the four boots; pdflip-4-25 (124245, 12:52:35) is the textbook one: the store held 1568 pages of
the 100423-token prompt, the fill needed 977 slots (``l3fill_pages=977``), ``L3-FILL ROOM-WAIT stems=970
claimed=0 still_full=970 rounds=102 waited_ms=5025``, 54 pages landed, ``#257 PREFETCH BELOW-ANCHOR read=3456
of 100352 anchored=3008``, 97408 tokens ran through the prefill again although every page was on the L3 disk.

WHY. The read's rows are arena slots (``alloc_read``: placeholders, resolved at the read). The fill claims a
slot per stem and fills it from the file; the only room-maker on that thread is the arena clock, which frees
COMPLETE slots NO reference holds. The census of the same minute (``ARENA-REF-HOLDERS`` 12:52:39, PP1):
``tree=6478 tree_in_use=0 prefetch=54`` -- the P tree's own host copies (a page the tree still holds on the
device, or a finished host-only chain) pin every slot, ``pinned=6485 of 6485``. The tree's spill
(``UnifiedRadixCache._w3_arena_spill``: L3 copy first, then the host copy goes, ``l3_present``) exists since
8f2fe84949, but it is called from the CLAIM paths of the write-backs only (``_pdflip_direct_claim``,
``_pdflip_full_arena_refuses``). A read never reaches it: on an arena boot ``alloc_read`` hands out placeholders
that take no slot, so upstream's ``evict_host(prefetch_length)`` before the alloc (hiradix
``prefetch_from_storage``) never fires -- 25 W3-ARENA SPILL lines in the whole boot, none for a read.

THE FIX (this file + three seams):

1. ``post`` -- the fill (prefetch io thread, ``arena_fill_from_disk`` / ``_await_fill_room``) writes its REAL
   need (stems that found no slot after the clock) next to the arena, in a small shared file
   (``<arena>.room``): a slot is free only when EVERY rank's reference is gone and the PP ranks' trees are not
   replicas, so a request must reach the peers' schedulers too, not only the poster's. No collective: the file
   is what the ranks already share (the arena), like the group-equal order of the spill.
2. ``tick`` -- the scheduler thread, once per round (``check_hicache_events``): a request posted by anybody
   (itself included) and not yet seen runs this rank's ``_w3_arena_spill(pool, deficit)``. ``deficit`` is
   recomputed from the arena header (free + clock-takable + this need), so a rank over-spills nothing the
   census already covers.
3. ``_await_fill_room`` waits for the answer (slots keep coming = the wait goes on, the JOIN-LIVENESS rule),
   and a wait that ends short names itself (``READ-ROOM SHORT class=``): not_posted / unserved /
   served_nothing_to_spill / served_spilled_short, from THIS wait's spills. Nothing ends silent.

Scope: the local-PP floor (group P: the tree is this rank's own, UD-H's scope -- a TP group's replicas never
edit rank-locally); ``armed`` says so. Switch ``FLLIPER_PDFLIP_READ_ROOM`` (default on); 0 = the old fill byte
for byte (no post, the 5 s room wait, no extra lines).

READ-ROOM-D (follow-up, same day). The arena is ONE file shared by group P and group D, and a slot is free only
when no rank of EITHER group holds it. The P stages served the request; group D never did -- ``armed`` was the
local-PP floor only, and ``_w3_arena_spill`` returns 0 under R12 (``form_a_host_shadow``: on a Form A D group a
worker's host rows mirror TP0's verdicts). Boot ..._226aad7105_1015_124245, pdflip-70-293 (13:35:34-39): the read
found ``pinned=6485 of 6485 refs=21495``; P held ``own_held=2258`` per stage (3 x 2258 = 6774), the rest was D's
tree: ``ARENA-REF-HOLDERS`` on D TP0/TP1/TP2 (13:35:42) ``tree=3215 tree_in_use=1692 own_held=4907`` -- the
host copies of D's live tree and of the spans of the requests D had just parked (``PDFLIP-D-PARK park_running
... span retained, forced host write-through``, ``#248 PARK-DEMOTE ... l3_pages=996``: every page already on
the L3 disk). D gives them back only at its sleep's reset (``#1427 RESET-RELEASE``); until then nobody on D
answered a room request.

Now D answers, as a GROUP: TP0 (R12 role ``host``) decides and serves (:func:`form_a_spill`): a device-resident
node's host copy leaves L2 only after ``secure_rows_to_l3`` gave every page an L3 copy (``lost`` != 0 keeps it),
aux host rows first and the KV rows last (the #248 LOADED-HOST-RELEASE rule, the tree invariant "aux requires
Full"), and the node's R12 STATE verdict rides the next ``tp<-reqs`` broadcast -- every worker drops the same
rows (``form_a_host_shadow._reconcile``), never by its own decision. A worker posts and waits like every rank
(the same bound on every rank of the group) but never serves. A node holding a Mamba anchor on the host is not
touched: a parked request reads its anchors exactly as before, and the KV pages it gave back are read again by
stem (L2 while the slot still holds them, else the L3 copy written before the release) -- the park's resume
reads the whole span to its last anchor, from the store, never short and never recomputed. Host-only nodes
stay (their arena copy is the one the sleep reset gives back). Same switch."""

from __future__ import annotations

import logging
import mmap
import os
import struct
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

MARK = "READ-ROOM"

#: shared request file: 64 slots of (read_id, need, t_ms, pid) u64, one per READ (review A3/B5: a re-post of the same
#: fill carries the same read id; two fills of one process take two slots -- needs add up, never max). A poster takes
#: the slot that carries its (pid, read_id), else a free or stale one, else the oldest.
_SLOTS = 64
_REC = struct.Struct("<QQQQ")
_SIZE = _SLOTS * _REC.size
_STALE_MS = 30_000
_POLL_S = 0.02

_LOCK = threading.Lock()


class _State:
    armed = False
    rounds = 0
    posted = 0
    served = 0
    spilled_pages = 0
    short = 0
    served_empty = 0
    last_post_ms: Dict[str, float] = {}
    seen: Dict[Tuple[str, int], int] = {}
    maps: Dict[str, Optional[mmap.mmap]] = {}
    seq = 0
    last_serve_t = 0.0
    arena: Any = None
    res_t = 0.0
    open_t = 0.0
    poll_t = 0.0
    role: Optional[str] = None
    role_t = -1e9
    d_spills = 0
    #: review readroomd B2: (tree, node, host rows) TP0 decided to spill, released at the broadcast
    d_pending: List[Tuple[Any, Any, Any]] = []
    #: review A3: per logical read id the pages THIS rank released for it, as (slot, stem) -- a post of the same read
    #: (a re-post, or a peer rank's fill of it) counts those a peer still holds as room on its way
    released: Dict[Any, Tuple[float, List[Tuple[int, str]]]] = {}
    #: review A2: (serve number, pages released) of the last serves -- a wait's class is its own serves' spills
    serve_log: Deque[Tuple[int, int]] = deque(maxlen=4096)
    #: review A1: the armed room waits of this process, [start, end) on time.monotonic (the told clock)
    waits: Deque[List[float]] = deque(maxlen=512)
    read_n = 0


_ST = _State


def enabled() -> bool:
    try:
        from flliper.srt.environ import envs

        return bool(envs.FLLIPER_PDFLIP_READ_ROOM.get())
    except Exception:  # noqa: BLE001 - a hermetic tree without the field: default on
        return True


def served(arena: Any) -> bool:
    """Somebody answers room requests for THIS arena: the scheduler of this process resolved it as its tree's KV
    arena (``tick``). A fill on another arena of the process (the 32-slot mamba anchor arena) posts nothing and
    keeps the old wait -- a request nobody reads must not stretch a wait."""
    if not _ST.armed or _ST.arena is None or arena is None:
        return False
    a, b = str(getattr(_ST.arena, "path", "") or ""), str(getattr(arena, "path", "") or "")
    return bool(a) and a == b


def wait_idle_ms(default_ms: int, arena: Any = None) -> int:
    """The idle bound of the fill's room wait (no slot gained for this long ends it): the old 5 s bound is
    shorter than a P stage's scheduler round (a 2-3 s chunk forward) plus the peers' rounds; with the room
    request armed on this arena it is ``FLLIPER_PDFLIP_READ_ROOM_WAIT_MS`` (default 10 s, below the told Frist cap
    of 12 s). Unarmed / not this process's served arena = the old bound."""
    if not served(arena):
        return int(default_ms)
    try:
        from flliper.srt.environ import envs

        return max(int(default_ms), int(envs.FLLIPER_PDFLIP_READ_ROOM_WAIT_MS.get()))
    except Exception:  # noqa: BLE001
        return max(int(default_ms), 10000)


def _repost_ms() -> int:
    try:
        from flliper.srt.environ import envs

        return max(100, int(envs.FLLIPER_PDFLIP_READ_ROOM_REPOST_MS.get()))
    except Exception:  # noqa: BLE001
        return 1500


def _local_pp(cache: Any) -> bool:
    try:
        from flliper.srt.pdflip import pp_slot_fidelity as _sf

        return bool(_sf.enabled() and getattr(cache, _sf.FLOOR_LOCAL_PP_ATTR, False))
    except Exception:  # noqa: BLE001
        return False


def form_a_role() -> Optional[str]:
    """READ-ROOM-D: the R12 role of this rank on a Form A TP group (``"host"`` = TP0, the group's one decider;
    ``"worker"``), None elsewhere. Group-uniform: the installed role plan and the switch are the same on every
    rank. S20: read EVERY time, no one-second cache -- ``role()`` is a switch read plus two attribute reads (~1 us; ``tick``
    runs ~1500 rounds/s idle), and the cache carried a role across anything that changed it: a test that ran
    ``form_a_host_shadow.attach`` under a patched role left ``host`` behind for a second, and the h16t F1 room-set tests
    (plain PP trees) that ran next took the Form A dispatch of ``serve`` (SERVE-DBG role='host' on pp=0/3)."""
    try:
        from flliper.srt.mem_cache import form_a_host_shadow as _r12

        _ST.role = _r12.role()
    except Exception:  # noqa: BLE001 - no plan readable = not a Form A group
        _ST.role = None
    return _ST.role


def armed(cache: Any) -> bool:
    """This rank takes part in room requests (posts, waits the longer bound): the switch is on and the tree is a
    local-PP tree (group P) or a rank of a Form A TP group (group D: TP0 and its workers alike)."""
    if not enabled():
        return False
    return _local_pp(cache) or form_a_role() in ("host", "worker")


def serves(cache: Any) -> bool:
    """This rank ANSWERS room requests: a local-PP tree (its own tree), or TP0 of a Form A group (the decision of
    the whole group; the workers follow its R12 verdicts). A Form A worker never releases on its own."""
    if not enabled():
        return False
    return _local_pp(cache) or form_a_role() == "host"


# --- the arena's own numbers -------------------------------------------------------------------------

def census(arena: Any) -> Dict[str, int]:
    """slots / complete / claimed / pinned (COMPLETE with a reader reference) / refs, from the arena header."""
    st = arena.stats()
    out = {"slots": int(st["slots"]), "complete": int(st["complete"]), "claimed": int(st["claimed"]),
           "pinned": -1, "refs": -1}
    rc = getattr(arena, "ref_census", None)
    if callable(rc):
        try:
            pinned, refs, _complete = rc()
            out["pinned"], out["refs"] = int(pinned), int(refs)
        except Exception:  # noqa: BLE001 - a census is an instrument
            pass
    return out


def deficit(arena: Any, need: int) -> Tuple[int, Dict[str, int]]:
    """Pages a read of ``need`` stems is still short of: free slots plus what the arena clock may take (COMPLETE
    without any reference) count as room; the rest has to come from the trees."""
    c = census(arena)
    free = c["slots"] - c["complete"] - c["claimed"]
    clock = (c["complete"] - c["pinned"]) if c["pinned"] >= 0 else 0
    c["free"], c["clock"] = free, clock
    return max(0, int(need) - max(0, free) - max(0, clock)), c


# --- the shared request file -------------------------------------------------------------------------

def _path(arena: Any) -> str:
    return str(getattr(arena, "path", "") or "") + ".room"


def _map(arena: Any, create: bool) -> Optional[mmap.mmap]:
    p = _path(arena)
    if not p or p == ".room":
        return None
    with _LOCK:
        m = _ST.maps.get(p)
        if m is not None:
            return m
        if p in _ST.maps and not create:
            # a failed open is retried at the caller's pace (tick, every 64th round)
            pass
        try:
            fd = os.open(p, (os.O_RDWR | os.O_CREAT) if create else os.O_RDWR, 0o666)
        except OSError:
            _ST.maps[p] = None
            return None
        try:
            if os.fstat(fd).st_size < _SIZE:
                os.ftruncate(fd, _SIZE)
            m = mmap.mmap(fd, _SIZE)
        except (OSError, ValueError):
            m = None
        finally:
            os.close(fd)
        _ST.maps[p] = m
        return m


def _now_ms() -> int:
    return int(time.time() * 1000.0)


def new_read_id(stems=None) -> int:
    """Review A3/B3: the id of a LOGICAL read, carried by the fill's first post and every re-post. From the
    content when the fill's stems are given (the last stem: PP0, PP1 and PP2 each fill the same stems of one rid,
    so their posts carry the SAME id and a rank that already let go for it never spills again for a peer's post);
    a fresh process-local number otherwise. Never 0 (0 marks an empty slot)."""
    if stems:
        import hashlib

        last = stems[-1]
        last = last[1] if isinstance(last, tuple) else last
        h = int.from_bytes(hashlib.blake2b(str(last).encode(), digest_size=8).digest(), "little")
        return (h & ((1 << 63) - 1)) or 1
    with _LOCK:
        _ST.read_n += 1
        return (1 << 63) | _ST.read_n


def _pick_slot(m: mmap.mmap, me: int, read_id: int) -> int:
    """The read's slot: the one carrying this (pid, read_id), else a free / stale one, else the oldest. Two
    fills of one process take two slots (review B5: their needs add up)."""
    now = _now_ms()
    free = oldest = None
    for s in range(_SLOTS):
        rid, _n, t, pid = _REC.unpack(m[s * _REC.size:(s + 1) * _REC.size])
        if pid == me and rid == read_id:
            return s
        if free is None and (not rid or now - t > _STALE_MS):
            free = s
        if oldest is None or t < oldest[1]:
            oldest = (s, t)
    return free if free is not None else oldest[0]


def post(arena: Any, need: int, *, site: str = "fill", read_id: Optional[int] = None) -> bool:
    """Prefetch io thread: this fill is short of ``need`` slots after its own room round. Re-posts of the same
    fill (same ``read_id``) are paced (``FLLIPER_PDFLIP_READ_ROOM_REPOST_MS``) and carry the fill's CURRENT need.
    ``read_id`` None = a fresh id (one-shot caller). False = not armed / nothing posted."""
    if need <= 0 or not served(arena):
        return False
    if read_id is None:
        read_id = new_read_id()
    key = _path(arena)
    now = time.perf_counter() * 1000.0
    last = _ST.last_post_ms.get((key, int(read_id)))
    if last is not None and now - last < _repost_ms():
        return False
    m = _map(arena, create=True)
    if m is None:
        return False
    me = os.getpid()
    with _LOCK:
        _ST.last_post_ms[(key, int(read_id))] = now
        if len(_ST.last_post_ms) > 4096:
            _ST.last_post_ms.pop(next(iter(_ST.last_post_ms)))
        _ST.posted += 1
        n = _ST.posted
        try:
            slot = _pick_slot(m, me, int(read_id))
            m[slot * _REC.size:(slot + 1) * _REC.size] = _REC.pack(int(read_id), int(need), _now_ms(), me)
        except (ValueError, OSError, struct.error):
            return False
    if n <= 16 or n % 64 == 0:
        logger.info("%s POST n=%d site=%s need=%d slot=%d read=%d (the fill found the arena FULL after the "
                    "clock: the trees that pin it are asked, on every rank of P and D, to hand back L3-backed "
                    "copies)", MARK, n, site, int(need), slot, int(read_id))
    return True


def _unseen(arena: Any, seen: Optional[Dict[Tuple[str, int], Any]] = None) -> int:
    """The largest need among requests nobody on this rank served yet (0 = none); marks them seen."""
    return max((n for _k, n in _unseen_reads(arena, seen)), default=0)


def _unseen_reads(arena: Any, seen: Optional[Dict[Tuple[str, int], Any]] = None) -> List[Tuple[int, int]]:
    """``[(read_id, need)]`` of the posts nobody on this rank served yet, oldest first; marks them seen
    (``seen`` = this process's ledger; a test names another to play a peer rank)."""
    if seen is None:
        seen = _ST.seen
    key = _path(arena)
    m = _ST.maps.get(key)
    if m is None:
        # the file appears with the first post of any rank: look for it at most every 0.2 s (a failed open is
        # one ENOENT; a scheduler round under load is seconds, a round COUNT would be minutes)
        t_now = time.monotonic()
        if t_now - _ST.open_t < 0.2:
            return []
        _ST.open_t = t_now
        with _LOCK:
            _ST.maps.pop(key, None)
        m = _map(arena, create=False)
        if m is None:
            return []
    out = []
    now = _now_ms()
    for s in range(_SLOTS):
        try:
            rid, n, t, pid = _REC.unpack(m[s * _REC.size:(s + 1) * _REC.size])
        except (ValueError, struct.error):
            continue
        if not rid or not n:
            continue
        if seen.get((key, s)) == (rid, t) or now - t > _STALE_MS:
            continue
        seen[(key, s)] = (rid, t)
        out.append((t, int(rid), int(n)))
    out.sort()
    return [(k, n) for _t, k, n in out]


def _forget_other_maps(arena: Any) -> None:
    """Review A6: the arena of a phase changed (rebind): this process drops the maps of the other ``.room`` files.
    The file itself lies in the boot's arena directory (``<FLLIPER_HICACHE_ARENA_DIR>/...room``, /dev/shm/pdflip-arena-
    <tag>) and goes with it (launcher ``SHM_OWN_PREFIXES`` "pdflip-arena-" sweep); one process never unlinks it while
    its peers may still post into it."""
    keep = _path(arena) if arena is not None else None
    with _LOCK:
        for p in [p for p in _ST.maps if p != keep]:
            m = _ST.maps.pop(p, None)
            try:
                if m is not None:
                    m.close()
            except Exception:  # noqa: BLE001
                pass


# --- the room-wait lifesign (review A1, coupled to told2 DEFER-LIFESIGN) -----------------------------------

def room_wait_begin(arena: Any) -> Optional[List[float]]:
    """The fill enters an ARMED room wait (posted on the served arena): a named, bounded wait of this process's own
    store read -- a lifesign, not idleness. None = not armed (no lifesign; the old wait, byte for byte)."""
    if not served(arena):
        return None
    iv = [time.monotonic(), float("inf")]
    with _LOCK:
        _ST.waits.append(iv)
    return iv


def room_wait_end(iv: Optional[List[float]]) -> None:
    if iv is not None:
        iv[1] = time.monotonic()


def lifesign_s(since: float, until: float) -> float:
    """Seconds of [since, until] (time.monotonic, the told clock) this process spent inside armed room waits --
    the UNION of the intervals (overlapping fills of several reads count once). PP0's told Frist moves its
    ``intake + TOTAL_S`` clamp by this much (``pdflip_told_fallback.deadline``, capped there by told2's absolute
    ceiling ``FLLIPER_PDFLIP_TOLD_WAIT_ABS_S``)."""
    lo, hi = float(since), float(until)
    if hi <= lo:
        return 0.0
    with _LOCK:
        ivs = sorted((max(lo, a), min(hi, b if b != float("inf") else hi)) for a, b in _ST.waits)
    total, cur_a, cur_b = 0.0, None, None
    for a, b in ivs:
        if b <= a:
            continue
        if cur_b is None or a > cur_b:
            if cur_b is not None:
                total += cur_b - cur_a
            cur_a, cur_b = a, b
        else:
            cur_b = max(cur_b, b)
    if cur_b is not None:
        total += cur_b - cur_a
    return total


# --- the scheduler side ------------------------------------------------------------------------------

def tick(cache: Any) -> int:
    """Once per scheduler round (``check_hicache_events``): arm state + serve what was posted. Never raises."""
    try:
        _ST.rounds += 1
        # F1 review A4: the PP group's slot set is applied (and the debts retried) in every round, at most
        # every 20 ms -- not only at a backup claim (an idle P stage claims nothing)
        from flliper.srt.pdflip import w3_room_set as _w3rs

        _w3rs.tick(cache)
        _ST.armed = armed(cache)    # per round: a handful of getattr/env reads; the floor mark moves per iteration
        if not _ST.armed:
            return 0
        from flliper.srt.pdflip import room_wait_arbiter as _arbiter

        _arbiter.note_role(cache)
        # the arena binds per phase (flip/rebind): re-resolve at most once a second, not per round
        t_now = time.monotonic()
        arena = _ST.arena
        if arena is None or t_now - _ST.res_t >= 1.0:
            _ST.res_t = t_now
            arena = _ST.arena = getattr(cache._pdflip_direct_pool(), "arena", None)
            _forget_other_maps(arena)
        if arena is None:
            return 0
        if not serves(cache) or form_a_role() == "host":
            # READ-ROOM-D: a Form A worker posts and waits (``served`` above); TP0 answers for the group at its
            # broadcast (``serve_at_broadcast``, review readroomd B2), never inside the round
            return 0
        # poll the request file at most every 20 ms: an idle P stage runs ~1500 rounds/s, a loaded one a round per
        # 2-3 s; 64 struct reads per round would be pure overhead in the first case and late by nothing in the second
        t_poll = time.monotonic()
        if t_poll - _ST.poll_t < _POLL_S:
            return 0
        _ST.poll_t = t_poll
        # ROOM-ARBITER 1016 (F3): PP0 announces which waiting request's fill may wait (the oldest); no-op elsewhere
        _arbiter.decide(arena)
        # review A2: a yielded fill's give-back is room on its way until its read is reaped on this rank
        from flliper.srt.pdflip import room_verdicts as _rv

        _rv.reap_givebacks(arena, cache)
        return serve_unseen(cache, None, arena)
    except Exception:  # noqa: BLE001 - a room request never breaks the round
        logger.warning("%s tick raised", MARK, exc_info=True)
        return 0


def serve_unseen(cache: Any, pool: Any, arena: Any, seen: Optional[Dict[Tuple[str, int], Any]] = None) -> int:
    """Every post nobody on this rank served yet, one read at a time (oldest first). ``pool`` None = the tree's
    direct pool (the scheduler's case); a test passes its own and a ``seen`` ledger to play a peer rank."""
    reads = _unseen_reads(arena, seen)
    if not reads:
        return 0
    if pool is None:
        pool = cache._pdflip_direct_pool()
        if getattr(pool, "arena", None) is None:
            return 0
        arena = pool.arena
    got = 0
    for key, need in reads:
        got += serve(cache, pool, arena, need, read_key=key)
    return got


def _still_pinned_mine(arena: Any, read_key: Any) -> int:
    """Review A3: pages this rank already released for ``read_key`` that a peer still references -- room on its
    way for that read (the last holder frees it), so a post of the same read never spills again for them. By PAGE
    identity (the stem still COMPLETE in a slot with references), never by slot number: a freed slot is reused."""
    rec = _ST.released.get(read_key)
    if not rec or not rec[1]:
        return 0
    pages = rec[1]
    reader = getattr(arena, "slot_refs", None)
    finder = getattr(arena, "find_slots_np", None)
    try:
        if callable(finder) and all(st for _s, st in pages):
            slots, states = finder([st for _s, st in pages])
            live = [int(x) for x, y in zip(slots, states) if int(x) >= 0 and int(y) == 2]
        else:
            live = [int(x) for x, _st in pages]
        if not live:
            return 0
        if not callable(reader):
            return len(live)
        return sum(1 for r in reader(live) if int(r) > 0)
    except Exception:  # noqa: BLE001 - no header read: count them all (never over-spill)
        return len(pages)


def _note_released(read_key: Any, arena: Any, slots: List[int]) -> None:
    if not slots:
        return
    stem_of = getattr(arena, "slot_stem", None)
    pages = []
    for x in slots:
        st = ""
        if callable(stem_of):
            try:
                st = str(stem_of(int(x)) or "")
            except Exception:  # noqa: BLE001
                st = ""
        pages.append((int(x), st))
    now = time.monotonic()
    old = _ST.released.get(read_key)
    _ST.released[read_key] = (now, (old[1] if old else []) + pages)
    for k in [k for k, v in _ST.released.items() if now - v[0] > _STALE_MS / 1000.0]:
        _ST.released.pop(k, None)


_SPILL_KW: Dict[type, bool] = {}


def _spill_takes_out(cache: Any) -> bool:
    """Does this tree's ``_w3_arena_spill`` take ``released_out`` (the real one does; a test stand-in may not)."""
    t = type(cache)
    if t not in _SPILL_KW:
        import inspect

        try:
            _SPILL_KW[t] = "released_out" in inspect.signature(cache._w3_arena_spill).parameters
        except (TypeError, ValueError):
            _SPILL_KW[t] = False
    return _SPILL_KW[t]


def serve(cache: Any, pool: Any, arena: Any, need: int, read_key: Any = None) -> int:
    """This rank hands back what the room request is short of: ``_w3_arena_spill`` (L3 copy first, then the host
    copy goes; device nodes first, host-only leaves after) for the DEFICIT the header still shows, less what this
    rank already released for the same read and a peer still holds (review A3)."""
    t0 = time.perf_counter()
    d, c = deficit(arena, need)
    # one tree per process in production; keyed by the tree too, so ranks played in one test process keep apart
    read_key = (id(cache), read_key)
    mine = _still_pinned_mine(arena, read_key)
    d = max(0, d - mine)
    c["mine_pinned"] = mine
    n = _ST.served + 1
    _ST.served = n
    # ROOM-VERDICTS 1016 (F2): this rank's answer goes into the shared file -- the fills end a wait no rank can
    # answer on that fact (pdflip/room_verdicts.py), not on their idle bound
    from flliper.srt.pdflip import room_verdicts as _rv

    if d <= 0:
        # review A1: room is there (free + clock) or on its way (this rank let go of it for the read and a peer still
        # holds it) -- never an answer that ends a wait
        _rv.note_verdict(arena, 0, _rv.transient_blocked(cache), False, room_seen=True)
        _ST.served_empty += 1
        if n <= 16 or n % 64 == 0:
            logger.info("%s SERVE n=%d need=%d deficit=0 slots=%d complete=%d claimed=%d pinned=%d free=%d "
                        "clock=%d mine_pinned=%d (the clock already has the room, or this rank already let go of "
                        "it for this read and a peer still holds it)", MARK, n, int(need), c["slots"],
                        c["complete"], c["claimed"], c["pinned"], c["free"], c["clock"], mine)
        return 0
    slots: List[int] = []
    _quiet = _rv.spill_quiet(cache)     # read BEFORE the spill: in its empty backoff it answers 0 without looking
    if form_a_role() == "host":
        # READ-ROOM-D: TP0 of a Form A group decides for the group (R12 verdicts carry it to the workers)
        released = int(form_a_spill(cache, pool, d, released_out=slots) or 0)
    elif _spill_takes_out(cache):
        released = int(cache._w3_arena_spill(pool, d, released_out=slots) or 0)
    else:
        released = int(cache._w3_arena_spill(pool, d) or 0)
    # F1b x F2 (h16t): an UNDECIDED group spill (PP0 waits for a follower's acknowledgement, a follower still holds
    # debts of PP0's set) answers like a spill that did not look -- room may still come, never a futility answer
    from flliper.srt.pdflip import w3_room_set as _w3rs

    _rv.note_verdict(arena, released, _rv.transient_blocked(cache), _quiet or _w3rs.undecided(cache))
    _note_released(read_key, arena, slots)
    _ST.spilled_pages += released
    _ST.serve_log.append((n, released))
    after = census(arena)
    if n <= 32 or n % 16 == 0 or released == 0:
        logger.info("%s SERVE n=%d need=%d deficit=%d released_pages=%d ms=%.0f | before: slots=%d complete=%d "
                    "claimed=%d pinned=%d free=%d clock=%d | after: complete=%d pinned=%d (a slot is free "
                    "only when every PP rank has let go; the peers serve the same request on their rounds)",
                    MARK, n, int(need), d, released, (time.perf_counter() - t0) * 1000.0, c["slots"],
                    c["complete"], c["claimed"], c["pinned"], c["free"], c["clock"], after["complete"],
                    after["pinned"])
    return released


def _l3_strict() -> bool:
    try:
        from flliper.srt.environ import envs

        return bool(envs.FLLIPER_PDFLIP_L3_PRESENT_STRICT.get())
    except Exception:  # noqa: BLE001 - a hermetic tree without the field: default on
        return True


def l3_complete(node: Any, sec: Dict[str, int]) -> bool:
    """Review readroomd B3: ``secure_rows_to_l3`` proved that EVERY page of ``node`` has an L3 copy -- no page lost,
    none unpaired (the QS pair gate wrote nothing), the COMPLETE pages are all the node's pages (``len(hash_value)``;
    a page some owner has not finished drops out of ``pages``) and each is on disk or was written now."""
    if int(sec.get("lost", 0)):
        return False
    # review readroomd2 A3: only the UNPAIRED rule follows FLLIPER_PDFLIP_L3_PRESENT_STRICT (H16P A2's switch -- off: a
    # KV page held back for its QSA index counts as copied, as before A2); the page count never does
    unpaired = int(sec.get("unpaired", 0))
    if unpaired and _l3_strict():
        return False
    pages = int(sec.get("pages", 0))
    hv = getattr(node, "hash_value", None)
    need = len(hv) if hv else pages
    copied = int(sec.get("on_disk", 0)) + int(sec.get("written", 0)) + (0 if _l3_strict() else unpaired)
    return pages > 0 and pages == need and copied == need


def flush_forma_pending() -> int:
    """Review readroomd B1/B2: TP0, in ``form_a_host_shadow.attach`` right before the broadcast: the host copies
    ``form_a_spill`` decided (L3 copy proven) leave L2 now, ``l3_present`` is set, and the R12 verdicts (STATE: the
    KV host rows are gone; L3: the node is store-present) ride THIS broadcast -- every worker applies both before
    the pass's admission. A node that changed since the decision (gone, other host rows, busy) keeps its copy.
    Returns the pages released."""
    todo, _ST.d_pending = list(_ST.d_pending), []
    if not todo:
        return 0
    from flliper.srt.mem_cache import form_a_host_shadow as _r12
    from flliper.srt.mem_cache.unified_radix_cache import BASE_COMPONENT_TYPE, EvictLayer

    def _release(cache, node, base, aux) -> int:
        """The host copy of ``node`` leaves L2 (real funnel), ``l3_present`` set; returns host rows freed."""
        for c in aux:   # aux first: never an aux host row without the KV row
            if node.component_data[c.component_type].host_value is not None:
                cache._evict_component_and_detach_lru(node, c, target=EvictLayer.HOST, tracker=None)
        _dev, host_freed = cache._evict_component_and_detach_lru(node, base, target=EvictLayer.HOST, tracker=None)
        cache.evictable_host_leaves.discard(node)
        node.l3_present = True
        return int(host_freed or 0)

    out = 0
    # S22: the workers' state is read again right before the broadcast (a peel since the decision keeps the copy);
    # once per flush, not per node
    _wev_now = _r12.worker_evicted_keys(todo[0][0])
    for cache, node, hv in todo:
        if _r12.node_key(node) in _wev_now:
            if _r12._note("rr_flush_worker_evicted", 16):
                logger.info("%s D-SPILL KEPT worker_evicted node=%s (a worker took it off the device after the "
                            "decision; TP0 keeps its host copy)", MARK, getattr(node, "id", "?"))
            continue
        comps = tuple(cache._components_tuple)
        base = next((c for c in comps if c.component_type == BASE_COMPONENT_TYPE), None)
        aux = [c for c in comps if c.component_type != BASE_COMPONENT_TYPE]
        cd = node.component_data[BASE_COMPONENT_TYPE]
        owt = getattr(cache, "ongoing_write_through", None) or {}
        if (base is None or cd.host_value is not hv or cd.value is None or node.children
                or _r12.node_key(node) is None      # review readroomd2 A2: no verdict could reach the workers
                or not _r12._attached(node) or node.id in owt
                or getattr(node, "write_through_pending_id", None) is not None
                or any(int(getattr(c, "host_lock_ref", 0) or 0) > 0 for c in node.component_data)):
            continue
        P = max(1, int(getattr(cache, "page_size", 1) or 1))
        l3_before = bool(getattr(node, "l3_present", False))
        try:
            out += _release(cache, node, base, aux) // P
        finally:
            # review readroomd2 A7: the verdict goes out whatever the release did -- STATE reads the node's ACTUAL
            # host flags, so a partial release is mirrored as it is, never left on TP0 alone
            _r12.record_state(cache, node, why="read-room")
            if node.l3_present and not l3_before:
                # review readroomd B1: l3_present is TP0's group verdict -- every worker sets it from this event
                _r12.record_l3(cache, node)
    return out


def serve_at_broadcast(tree: Any) -> int:
    """Review readroomd B2/B5: Form A TP0 serves room requests HERE, from ``form_a_host_shadow.attach`` right before
    the ``tp<-reqs`` broadcast (the K10 wire) -- its host drops and the R12 verdict that mirrors them reach every
    worker before any admission of the pass; never from ``check_hicache_events``. No broadcast (sleep drain) = no
    service. Never raises."""
    try:
        if form_a_role() != "host" or not enabled():
            return 0
        pool = tree._pdflip_direct_pool()
        arena = getattr(pool, "arena", None)
        if arena is None:
            return 0
        _ST.arena = arena
        _ST.armed = True
        return serve_unseen(tree, pool, arena)
    except Exception:  # noqa: BLE001 - a room request never breaks the broadcast
        logger.warning("%s serve_at_broadcast raised", MARK, exc_info=True)
        return 0


def form_a_spill(cache: Any, pool: Any, need_pages: int, released_out: Optional[list] = None) -> int:
    """READ-ROOM-D, TP0 of a Form A D group only: hand back up to ``max(need, W3_SPILL_MIN_PAGES)`` pages of the
    tree's L3-backed host copies for the GROUP. Candidates are device-resident nodes with a KV host copy (the
    device keeps the KV, ``l3_present`` names the store copy): no host lock (a load in flight), no pending write,
    no aux state on the host only, NO Mamba anchor on the host (the anchors of a parked request stay as they
    are), not staging rows. Order: the group-equal key of ``_w3_spill_keys`` (pages a P stage already let go of
    first -- the last holder frees the slot). Per node: ``secure_rows_to_l3`` first (``lost`` != 0 or no page
    COMPLETE keeps the node), then aux host rows, then the KV host rows, then the R12 STATE verdict -- every
    worker drops the same rows at the next broadcast. Returns the pages released on TP0."""
    from flliper.srt.mem_cache import form_a_host_shadow as _r12

    if _r12.role() != "host":
        return 0   # never rank-local: a worker follows TP0's verdicts, a non-Form-A tree has its own spill
    from flliper.srt.mem_cache.unified_cache_components.tree_component import ComponentType
    from flliper.srt.mem_cache.unified_radix_cache import BASE_COMPONENT_TYPE, EvictLayer

    if not hasattr(pool, "secure_rows_to_l3"):
        # the hybrid HostPoolGroup forwards the claim calls only: the arena pool is its anchor host pool -- the
        # resolution of W3-ANCHOR-POOL, here under READ-ROOM's own switch: the NF profiles set
        # FLLIPER_PDFLIP_ENABLE_W3_SPILL_ANCHOR_POOL for group P only (NF_ENV_P), D would stop here for good
        host = getattr(getattr(pool, "anchor_entry", None), "host_pool", None)
        pool = host if host is not None and hasattr(host, "secure_rows_to_l3") else None
        if pool is None:
            _ST.d_spills += 1
            if _ST.d_spills <= 8 or _ST.d_spills % 64 == 0:
                logger.warning("%s D-SPILL STOP no_spill_pool need=%d (no host pool with secure_rows_to_l3)",
                               MARK, int(need_pages))
            return 0
    t0 = time.perf_counter()
    root = cache.root_node
    want = max(int(need_pages), int(getattr(cache, "W3_SPILL_MIN_PAGES", 64)))
    P = max(1, int(getattr(cache, "page_size", 1) or 1))
    mamba = getattr(ComponentType, "MAMBA", None)
    comps = tuple(cache._components_tuple)
    base = next((c for c in comps if c.component_type == BASE_COMPONENT_TYPE), None)
    if base is None:
        return 0
    aux = [c for c in comps if c.component_type != BASE_COMPONENT_TYPE]
    staging = int(getattr(pool, "staging_rows", 0) or 0)
    owt = getattr(cache, "ongoing_write_through", None) or {}
    skip = {"anchor": 0, "busy": 0, "aux_host_only": 0, "host_only": 0}
    tree_nodes = 0
    cands = []
    wev = _r12.worker_evicted_keys(cache)   # S22: a node a worker has device-evicted is no spill candidate
    for node in cache._collect_all_nodes():
        if node is root:
            continue
        tree_nodes += 1
        cd = node.component_data[BASE_COMPONENT_TYPE]
        hv = cd.host_value
        if hv is None or hv.numel() == 0:
            # S23 (h16v4b 1018_013102: 59x D-SPILL candidates=0 with EVERY skip 0): a node without a host copy was the
            # one silent continue -- an empty tree and a tree of device-only nodes read alike. Counted now.
            skip["no_host"] = skip.get("no_host", 0) + 1
            continue
        if cd.value is None or getattr(node, "evicted", False):
            skip["host_only"] += 1
            continue
        if mamba is not None and node.component_data[mamba].host_value is not None:
            skip["anchor"] += 1
            continue
        if mamba is not None and node.component_data[mamba].value is not None:
            # review readroomd B4: a node carrying a Mamba state leaves the device only whole (L3-DROP below would
            # drop the anchor with it) -- the anchors of D stay where they are
            skip["anchor_dev"] = skip.get("anchor_dev", 0) + 1
            continue
        if node.children:
            # review readroomd B4: a spilled node must be able to leave the device without a new arena claim; the
            # Form A L3-DROP (UnifiedRadixCache._pdflip_l3_drop) drops a CHILDLESS store-backed leaf only
            skip["children"] = skip.get("children", 0) + 1
            continue
        if node.id in owt or getattr(node, "write_through_pending_id", None) is not None \
                or any(int(getattr(c, "host_lock_ref", 0) or 0) > 0 for c in node.component_data):
            skip["busy"] += 1
            continue
        if any(node.component_data[c.component_type].host_value is not None
               and node.component_data[c.component_type].value is None for c in aux):
            skip["aux_host_only"] += 1
            continue
        if int(hv.min()) < staging:
            continue
        if _r12.node_key(node) is None:
            # review readroomd2 A2: an unkeyed node gets no R12 verdict (record_state/record_l3 drop it) -- TP0
            # alone would change; it is no candidate
            skip["unkeyed"] = skip.get("unkeyed", 0) + 1
            continue
        if _r12.node_key(node) in wev:
            # S22 (R12 SPILL-DIVERGED class): a worker has this node off the device, the STATE verdict has no state
            # to move it to there (interior: unreconcilable, leaf: deleted)
            skip["worker_evicted"] = skip.get("worker_evicted", 0) + 1
            continue
        cands.append(node)
    keys = cache._w3_spill_keys(pool, cands)
    cands.sort(key=lambda x: keys[id(x)])
    released = nodes = unsecured = on_disk = written = 0
    for node in cands:
        if released >= want:
            break
        hv = node.component_data[BASE_COMPONENT_TYPE].host_value
        sec = pool.secure_rows_to_l3(hv)
        if int(sec.get("lost", 0)) or not int(sec.get("pages", 0)):
            unsecured += 1
            continue
        if not l3_complete(node, sec):
            # review readroomd B3: every page of the node needs its L3 copy (on disk or written now); an unpaired
            # page (QS gate) or a page no owner finished is not in the store -- the node keeps its host copy
            unsecured += 1
            continue
        on_disk += int(sec.get("on_disk", 0))
        written += int(sec.get("written", 0))
        if released_out is not None:
            released_out.extend(keys.slots_of(node))
        # review readroomd B2: decided and secured now, RELEASED at the broadcast (flush_forma_pending in attach) --
        # like an R12 TRANSIT, no rank's tree changes inside the pass
        _ST.d_pending.append((cache, node, hv))
        released += int(hv.numel()) // P
        nodes += 1
    _ST.d_spills += 1
    k = _ST.d_spills
    if k <= 32 or k % 16 == 0 or released == 0:
        logger.info("%s D-SPILL n=%d need=%d want=%d released_pages=%d nodes=%d candidates=%d unsecured_kept=%d "
                    "l3=on_disk:%d,written:%d tree_nodes=%d kept=no_host:%d,anchor:%d,anchor_dev:%d,children:%d,"
                    "unkeyed:%d,busy:%d,aux_host_only:%d,host_only:%d,kept_worker_evicted=%d ms=%.0f (Form A "
                    "TP0 decides for group D: an L3-backed host copy of D's tree leaves L2, the device keeps the "
                    "KV, the R12 STATE verdict drops the workers' rows at the next broadcast; anchors untouched)",
                    MARK, k, int(need_pages), want, released, nodes, len(cands), unsecured, on_disk, written,
                    tree_nodes, skip.get("no_host", 0), skip["anchor"], skip.get("anchor_dev", 0), skip.get("children", 0),
                    skip.get("unkeyed", 0), skip["busy"], skip["aux_host_only"], skip["host_only"],
                    skip.get("worker_evicted", 0),
                    (time.perf_counter() - t0) * 1000.0)
    return released


def note_short(arena: Any, stems: int, rounds: int, waited_ms: float, served_before: int) -> None:
    """The room wait ended and ``stems`` still have no slot: the cut is named, with the holders' picture, so
    a read that ends here is a counted, classified event and never a silent re-prefill."""
    _ST.short += 1
    k = _ST.short
    c = census(arena)
    posted = _ST.posted
    # review A2/B2: the class of THIS wait, from the serves after ``served_before`` -- never the process lifetime
    spilled = sum(r for k, r in list(_ST.serve_log) if k > int(served_before))
    cls = "not_posted" if not posted else (
        "unserved" if _ST.served == served_before else
        ("served_nothing_to_spill" if spilled <= 0 else "served_spilled_short"))
    if k <= 32 or k % 16 == 0:
        logger.warning("%s SHORT n=%d stems=%d rounds=%d waited_ms=%.0f class=%s posted=%d served=%d "
                       "spilled_pages=%d | slots=%d complete=%d claimed=%d pinned=%d refs=%d "
                       "(this read's pages stay on the disk and are prefilled again -- named, counted: "
                       "spilled_pages = this wait's; unserved = no scheduler round answered in the wait; served_nothing_to_spill = no tree "
                       "copy was spillable (live reads/loads pin the arena); served_spilled_short = this rank "
                       "let go of pages during the wait and the read is still short -- what is left is held by "
                       "references no spill may take, here or on a peer rank of P or D; review B5)",
                       MARK, k, int(stems), int(rounds), float(waited_ms), cls, posted, _ST.served,
                       spilled, c["slots"], c["complete"], c["claimed"], c["pinned"], c["refs"])


def reset_for_tests() -> None:
    _ST.armed = False
    _ST.rounds = 0
    _ST.posted = _ST.served = _ST.spilled_pages = _ST.short = _ST.served_empty = 0
    _ST.last_post_ms = {}
    _ST.seen = {}
    for m in list(_ST.maps.values()):
        try:
            if m is not None:
                m.close()
        except Exception:  # noqa: BLE001
            pass
    _ST.maps = {}
    _ST.seq = 0
    _ST.arena = None
    _ST.res_t = _ST.open_t = _ST.poll_t = 0.0
    _ST.role = None
    _ST.role_t = -1e9
    _ST.d_spills = 0
    _ST.d_pending = []
    _ST.released = {}
    _ST.serve_log = deque(maxlen=4096)
    _ST.waits = deque(maxlen=512)
    _ST.read_n = 0
