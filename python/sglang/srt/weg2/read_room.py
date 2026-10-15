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
   and a wait that ends short names itself (``READ-ROOM SHORT``): which holders, was it served, released how
   many. Nothing ends silent.

Scope: the local-PP floor (group P: the tree is this rank's own, UD-H's scope -- a TP group's replicas never
edit rank-locally); ``armed`` says so. Switch ``SGLANG_WEG2_READ_ROOM`` (default on); 0 = the old fill byte
for byte (no post, the 5 s room wait, no extra lines)."""

from __future__ import annotations

import logging
import mmap
import os
import struct
import threading
import time
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

MARK = "READ-ROOM"

#: shared request file: 64 slots of (seq, need, t_ms, pid) u64, one per poster process (a poster takes the slot
#: that carries its pid, else a free or stale one, else the oldest -- two ranks never share a slot by pid % N)
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


def armed(cache: Any) -> bool:
    """This tree serves room requests: the switch is on and it is a local-PP tree (group P)."""
    if not enabled():
        return False
    try:
        from sglang.srt.weg2 import pp_slot_fidelity as _sf

        return bool(_sf.enabled() and getattr(cache, _sf.FLOOR_LOCAL_PP_ATTR, False))
    except Exception:  # noqa: BLE001
        return False


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


def _pick_slot(m: mmap.mmap, me: int, need: int) -> Tuple[int, int]:
    """The poster's slot (its pid's, else a free / stale one, else the oldest) and its need: a second post of
    the same process within 3 s keeps the larger need (two prefetch threads of one rank, one slot)."""
    now = _now_ms()
    free = oldest = None
    for s in range(_SLOTS):
        seq, n, t, pid = _REC.unpack(m[s * _REC.size:(s + 1) * _REC.size])
        if pid == me:
            return s, (max(need, int(n)) if now - t < 3000 else need)
        if free is None and (not seq or now - t > _STALE_MS):
            free = s
        if oldest is None or t < oldest[1]:
            oldest = (s, t)
    return (free if free is not None else oldest[0]), need


def post(arena: Any, need: int, *, site: str = "fill") -> bool:
    """Prefetch io thread: this fill is short of ``need`` slots after its own room round. Re-posts of the same
    fill are paced (``SGLANG_WEG2_READ_ROOM_REPOST_MS``). False = not armed / nothing posted."""
    if need <= 0 or not served(arena):
        return False
    key = _path(arena)
    now = time.perf_counter() * 1000.0
    last = _ST.last_post_ms.get(key)
    if last is not None and now - last < _repost_ms():
        return False
    m = _map(arena, create=True)
    if m is None:
        return False
    me = os.getpid()
    with _LOCK:
        _ST.seq += 1
        seq = _ST.seq
        _ST.last_post_ms[key] = now
        _ST.posted += 1
        n = _ST.posted
        try:
            slot, need = _pick_slot(m, me, int(need))
            m[slot * _REC.size:(slot + 1) * _REC.size] = _REC.pack(seq, int(need), _now_ms(), me)
        except (ValueError, OSError, struct.error):
            return False
    if n <= 16 or n % 64 == 0:
        logger.info("%s POST n=%d site=%s need=%d slot=%d seq=%d (the fill found the arena FULL after the "
                    "clock: the trees that pin it are asked, on every PP rank, to hand back L3-backed copies)",
                    MARK, n, site, int(need), slot, seq)
    return True


def _unseen(arena: Any, seen: Optional[Dict[Tuple[str, int], Any]] = None) -> int:
    """The largest need among requests nobody on this rank served yet (0 = none); marks them seen
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
            return 0
        _ST.open_t = t_now
        with _LOCK:
            _ST.maps.pop(key, None)
        m = _map(arena, create=False)
        if m is None:
            return 0
    need = 0
    now = _now_ms()
    for s in range(_SLOTS):
        try:
            seq, n, t, _pid = _REC.unpack(m[s * _REC.size:(s + 1) * _REC.size])
        except (ValueError, struct.error):
            continue
        if not seq or not n:
            continue
        if seen.get((key, s)) == (seq, t) or now - t > _STALE_MS:
            continue
        seen[(key, s)] = (seq, t)
        need = max(need, int(n))
    return need


# --- the scheduler side ------------------------------------------------------------------------------

def tick(cache: Any) -> int:
    """Once per scheduler round (``check_hicache_events``): arm state + serve what was posted. Never raises."""
    try:
        _ST.rounds += 1
        _ST.armed = armed(cache)    # per round: a handful of getattr/env reads; the floor mark moves per iteration
        if not _ST.armed:
            return 0
        # the arena binds per phase (flip/rebind): re-resolve at most once a second, not per round
        t_now = time.monotonic()
        arena = _ST.arena
        if arena is None or t_now - _ST.res_t >= 1.0:
            _ST.res_t = t_now
            arena = _ST.arena = getattr(cache._weg2_direct_pool(), "arena", None)
        if arena is None:
            return 0
        # poll the request file at most every 20 ms: an idle P stage runs ~1500 rounds/s, a loaded one a round per
        # 2-3 s; 64 struct reads per round would be pure overhead in the first case and late by nothing in the second
        t_poll = time.monotonic()
        if t_poll - _ST.poll_t < _POLL_S:
            return 0
        _ST.poll_t = t_poll
        need = _unseen(arena)
        if need <= 0:
            return 0
        pool = cache._weg2_direct_pool()
        arena = getattr(pool, "arena", None)
        if arena is None:
            return 0
        return serve(cache, pool, arena, need)
    except Exception:  # noqa: BLE001 - a room request never breaks the round
        logger.warning("%s tick raised", MARK, exc_info=True)
        return 0


def serve(cache: Any, pool: Any, arena: Any, need: int) -> int:
    """This rank hands back what the room request is short of: ``_w3_arena_spill`` (L3 copy first, then the host
    copy goes; device nodes first, host-only leaves after) for the DEFICIT the header still shows."""
    t0 = time.perf_counter()
    d, c = deficit(arena, need)
    n = _ST.served + 1
    _ST.served = n
    if d <= 0:
        _ST.served_empty += 1
        if n <= 16 or n % 64 == 0:
            logger.info("%s SERVE n=%d need=%d deficit=0 slots=%d complete=%d claimed=%d pinned=%d free=%d "
                        "clock=%d (the clock already has the room)", MARK, n, int(need), c["slots"],
                        c["complete"], c["claimed"], c["pinned"], c["free"], c["clock"])
        return 0
    released = int(cache._w3_arena_spill(pool, d) or 0)
    _ST.spilled_pages += released
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
    cls = "not_posted" if not posted else (
        "unserved" if _ST.served == served_before else
        ("served_nothing_to_spill" if _ST.spilled_pages == 0 else "served_peers_hold"))
    if k <= 32 or k % 16 == 0:
        logger.warning("%s SHORT n=%d stems=%d rounds=%d waited_ms=%.0f class=%s posted=%d served=%d "
                       "spilled_pages=%d | slots=%d complete=%d claimed=%d pinned=%d refs=%d "
                       "(this read's pages stay on the disk and are prefilled again -- named, counted: "
                       "unserved = no scheduler round answered in the wait; served_nothing_to_spill = no tree "
                       "copy was spillable (live reads/loads pin the arena); served_peers_hold = this rank "
                       "let go, another PP rank did not)",
                       MARK, k, int(stems), int(rounds), float(waited_ms), cls, posted, _ST.served,
                       _ST.spilled_pages, c["slots"], c["complete"], c["claimed"], c["pinned"], c["refs"])


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
