"""READ-ROOM 1015 (NF h16p, zero avoidable tokens): an L3 -> L2 read that finds the shared arena FULL
asks the trees that pin it to give their L3-backed copies back -- it never ends the prefix there.

THE GAP (boots ..._226aad7105_1015_{115748,122948,124245,133855}, P): 46 ``WEG2-READ-DEPTH cause=arena_full``
cuts on PP0 in the four boots; weg2-4-25 (124245, 12:52:35) is the textbook one: the store held 1568 pages of
the 100423-token prompt, the fill needed 977 slots (``l3fill_pages=977``), ``L3-FILL ROOM-WAIT stems=970
claimed=0 still_full=970 rounds=102 waited_ms=5025``, 54 pages landed, ``#257 PREFETCH BELOW-ANCHOR read=3456
of 100352 anchored=3008``, 97408 tokens ran through the prefill again although every page was on the L3 disk.

WHY. The read's rows are arena slots (``alloc_read``: placeholders, resolved at the read). The fill claims a
slot per stem and fills it from the file; the only room-maker on that thread is the arena clock, which frees
COMPLETE slots NO reference holds. The census of the same minute (``ARENA-REF-HOLDERS`` 12:52:39, PP1):
``tree=6478 tree_in_use=0 prefetch=54`` -- the P tree's own host copies (a page the tree still holds on the
device, or a finished host-only chain) pin every slot, ``pinned=6485 of 6485``. The tree's spill
(``UnifiedRadixCache._w3_arena_spill``: L3 copy first, then the host copy goes, ``l3_present``) exists since
8f2fe84949, but it is called from the CLAIM paths of the write-backs only (``_weg2_direct_claim``,
``_weg2_full_arena_refuses``). A read never reaches it: on an arena boot ``alloc_read`` hands out placeholders
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
edit rank-locally); ``armed`` says so. Switch ``SGLANG_WEG2_READ_ROOM`` (default on); 0 = the old fill byte
for byte (no post, the 5 s room wait, no extra lines).

Group D (Form A) is NOT served here: its tree's host rows mirror TP0 over R12, and the room it holds is
handed back only as a group decision (READ-ROOM-D, separate branch)."""

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
        from sglang.srt.environ import envs

        return bool(envs.SGLANG_WEG2_READ_ROOM.get())
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
    request armed on this arena it is ``SGLANG_WEG2_READ_ROOM_WAIT_MS`` (default 10 s, below the told Frist cap
    of 12 s). Unarmed / not this process's served arena = the old bound."""
    if not served(arena):
        return int(default_ms)
    try:
        from sglang.srt.environ import envs

        return max(int(default_ms), int(envs.SGLANG_WEG2_READ_ROOM_WAIT_MS.get()))
    except Exception:  # noqa: BLE001
        return max(int(default_ms), 10000)


def _repost_ms() -> int:
    try:
        from sglang.srt.environ import envs

        return max(100, int(envs.SGLANG_WEG2_READ_ROOM_REPOST_MS.get()))
    except Exception:  # noqa: BLE001
        return 1500


def _local_pp(cache: Any) -> bool:
    try:
        from sglang.srt.weg2 import pp_slot_fidelity as _sf

        return bool(_sf.enabled() and getattr(cache, _sf.FLOOR_LOCAL_PP_ATTR, False))
    except Exception:  # noqa: BLE001
        return False


def armed(cache: Any) -> bool:
    """This rank takes part in room requests (posts, waits the longer bound, serves): the switch is on and the tree
    is a local-PP tree (group P)."""
    if not enabled():
        return False
    return _local_pp(cache)


def serves(cache: Any) -> bool:
    """This rank ANSWERS room requests: a local-PP tree (its own tree)."""
    return armed(cache)


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
    fill (same ``read_id``) are paced (``SGLANG_WEG2_READ_ROOM_REPOST_MS``) and carry the fill's CURRENT need.
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
                    "clock: the trees that pin it are asked, on every PP rank of group P, to hand back L3-backed "
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
    The file itself lies in the boot's arena directory (``<SGLANG_HICACHE_ARENA_DIR>/...room``, /dev/shm/weg2-arena-
    <tag>) and goes with it (launcher ``SHM_OWN_PREFIXES`` "weg2-arena-" sweep); one process never unlinks it while
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
    ``intake + TOTAL_S`` clamp by this much (``weg2_told_fallback.deadline``, capped there by told2's absolute
    ceiling ``SGLANG_WEG2_TOLD_WAIT_ABS_S``)."""
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
        _ST.armed = armed(cache)    # per round: a handful of getattr/env reads; the floor mark moves per iteration
        if not _ST.armed:
            return 0
        from sglang.srt.weg2 import room_wait_arbiter as _arbiter

        _arbiter.note_role(cache)
        # the arena binds per phase (flip/rebind): re-resolve at most once a second, not per round
        t_now = time.monotonic()
        arena = _ST.arena
        if arena is None or t_now - _ST.res_t >= 1.0:
            _ST.res_t = t_now
            arena = _ST.arena = getattr(cache._weg2_direct_pool(), "arena", None)
            _forget_other_maps(arena)
        if arena is None:
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
        from sglang.srt.weg2 import room_verdicts as _rv

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
        pool = cache._weg2_direct_pool()
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
    # answer on that fact (weg2/room_verdicts.py), not on their idle bound
    from sglang.srt.weg2 import room_verdicts as _rv

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
    if _spill_takes_out(cache):
        released = int(cache._w3_arena_spill(pool, d, released_out=slots) or 0)
    else:
        released = int(cache._w3_arena_spill(pool, d) or 0)
    _rv.note_verdict(arena, released, _rv.transient_blocked(cache), _quiet)
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
    _ST.released = {}
    _ST.serve_log = deque(maxlen=4096)
    _ST.waits = deque(maxlen=512)
    _ST.read_n = 0
