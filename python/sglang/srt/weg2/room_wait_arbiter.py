"""ROOM-ARBITER 1016 (NF h16s P-gap root): who may wait for arena room, decided by PP0.

THE GAP (boot ..._1f4b3e73d4_1016_103428, P, 11:27:18-11:27:43, 25 s without a prefill batch, 5 queued): the
shared KV arena was pinned 6485/6485, and on every P rank SEVERAL store reads waited for room at once (PP1:
stems 225, 452, 509; PP0: 225, 452), each holding the pages it had already landed (REF-AT-LAND: ``ARENA-REF-HOLDERS
gap=1122/1109/407`` = references no tree/prefetch class names -- the landings of fills still inside their batch).
Every one of them waited for room that only the others' landings could have given; all of them ended together at
the idle bound (``READ-ROOM SHORT ... waited_ms=10022..10456`` at 11:27:25, again at 11:27:35). Hold-and-wait.

THE RULE (F3, user law PRIO = ALTER): only the fill of the OLDEST waiting request waits for room. Every other fill
ends its room wait at once -- its short read is cut at its anchor by the existing #257 path at the reap, which gives
the landed references beyond the anchor back -- and the oldest one gets what they held.

RANKS NEVER DISAGREE: PP0 decides. Every waiting fill registers its request in a small shared file next to the
arena (``<arena>.room2``, like READ-ROOM's ``.room``: the ranks share the arena, so they share the file -- no
collective). PP0's scheduler round (``read_room.tick``) reads every rank's registrations and announces the owner:
the request with the lowest arrival number. The age is group-uniform without a clock: the front's rid
``weg2-<epoch>-<n>`` carries the global arrival counter ``n``; a rid of another shape is ordered by the moment PP0
first saw it registered (PP0's own order -- still one decision). Followers -- and PP0's own fills -- act on the
announcement only: no owner announced yet = wait as before; an owner that is not this fill's request = end now.

Scope: a fill that knows its request (the prefetch io thread's ``_arena_page_get`` names the operation through
:func:`fill_scope`) on an arena READ-ROOM serves (``read_room.served``, group P). Every other fill waits exactly
as before. Switch ``SGLANG_WEG2_ROOM_ARBITER`` (default on; 0 = the room wait byte for byte as before).
"""

from __future__ import annotations

import contextlib
import logging
import mmap
import os
import re
import struct
import threading
import time
import zlib
from typing import Any, Dict, Iterator, List, Optional, Tuple

logger = logging.getLogger(__name__)

MARK = "ROOM-ARBITER"
ENV = "SGLANG_WEG2_ROOM_ARBITER"

#: header: magic, owner rid hash, owner age key, announcing pid, announce seq, announce t_ms, pp_size, reserved
_HDR = struct.Struct("<QQQQQQQQ")
_MAGIC = 0x52304F4D32524F31
#: wait registrations: rid hash, age key (0 = none parsed), pid, t_ms (refreshed by the waiting fill), active
_W = struct.Struct("<QQQQQ")
_WAIT_SLOTS = 64
_W_OFF = _HDR.size
_SIZE_F3 = _W_OFF + _WAIT_SLOTS * _W.size
_STALE_MS = 30_000

_LOCK = threading.Lock()
_TL = threading.local()
_RID_N = re.compile(r"-(\d+)$")


class _State:
    pp_rank: Optional[int] = None
    pp_size: int = 0
    maps: Dict[str, Optional[mmap.mmap]] = {}
    first_seen: Dict[int, int] = {}
    seen_n = 0
    announced: Tuple[int, int] = (0, 0)
    seq = 0
    ended_young = 0
    announces = 0


_A = _State


def enabled() -> bool:
    raw = (os.environ.get(ENV, "") or "").strip().lower()
    return raw not in ("0", "false", "no", "off")


def rid_hash(rid: Any) -> int:
    """Never 0 (0 marks an empty slot)."""
    return (zlib.crc32(str(rid).encode()) & 0xFFFFFFFF) | (1 << 40)


def age_key(rid: Any) -> int:
    """The front's global arrival counter of ``weg2-<epoch>-<n>`` (+1, so 0 means 'no counter'); 0 otherwise."""
    m = _RID_N.search(str(rid or ""))
    return int(m.group(1)) + 1 if m else 0


# --- the fill's request (prefetch io thread) ---------------------------------------------------------

@contextlib.contextmanager
def fill_scope(rid: Any) -> Iterator[None]:
    """``_arena_page_get`` names the request of the fill it runs on this thread."""
    prev = getattr(_TL, "rid", None)
    _TL.rid = None if rid is None else str(rid)
    try:
        yield
    finally:
        _TL.rid = prev


def current_rid() -> Optional[str]:
    return getattr(_TL, "rid", None)


# --- the shared file ---------------------------------------------------------------------------------

def _path(arena: Any) -> str:
    p = str(getattr(arena, "path", "") or "")
    return p + ".room2" if p else ""


def _size() -> int:
    """F3's header + wait slots, then F2's verdict region (``room_verdicts``) when that module is present."""
    try:
        from sglang.srt.weg2 import room_verdicts as _rv
    except ImportError:
        return _SIZE_F3
    return _SIZE_F3 + _rv.REGION_SIZE


def mapped(arena: Any, create: bool = True) -> Optional[mmap.mmap]:
    p = _path(arena)
    if not p:
        return None
    with _LOCK:
        m = _A.maps.get(p)
        if m is not None:
            return m
        try:
            fd = os.open(p, (os.O_RDWR | os.O_CREAT) if create else os.O_RDWR, 0o666)
        except OSError:
            return None
        try:
            size = _size()
            if os.fstat(fd).st_size < size:
                os.ftruncate(fd, size)
            m = mmap.mmap(fd, size)
        except (OSError, ValueError):
            m = None
        finally:
            os.close(fd)
        _A.maps[p] = m
        return m


def _now_ms() -> int:
    return int(time.time() * 1000.0)


def _alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except OSError:
        return True   # exists, not ours to signal


def _read_hdr(m: mmap.mmap) -> Tuple[int, ...]:
    return _HDR.unpack(m[0:_HDR.size])


def owner(arena: Any) -> Optional[Tuple[int, int, int, int]]:
    """``(owner rid hash, age key, announcing pid, announce t_ms)`` PP0 announced, or None (none / stale / its PP0
    gone)."""
    m = mapped(arena, create=False)
    if m is None:
        return None
    try:
        magic, h, key, pid, _seq, t, _pps, _r = _read_hdr(m)
    except (ValueError, struct.error):
        return None
    if magic != _MAGIC or not h:
        return None
    if _now_ms() - t > _STALE_MS or not _alive(int(pid)):
        return None
    return int(h), int(key), int(pid), int(t)


def _wait_slot(m: mmap.mmap, me: int, h: int) -> int:
    free = oldest = None
    now = _now_ms()
    for s in range(_WAIT_SLOTS):
        off = _W_OFF + s * _W.size
        wh, _k, pid, t, act = _W.unpack(m[off:off + _W.size])
        if pid == me and wh == h:
            return s
        if free is None and (not act or now - t > _STALE_MS):
            free = s
        if oldest is None or t < oldest[1]:
            oldest = (s, t)
    return free if free is not None else oldest[0]


def register(arena: Any, rid: str) -> Optional[int]:
    """The fill of ``rid`` waits (or keeps waiting) for room: (re)write its registration, refreshed per round.
    The caller keeps :func:`now_ms` of its FIRST registration (``may_wait(since_ms=)``)."""
    m = mapped(arena, create=True)
    if m is None:
        return None
    me, h = os.getpid(), rid_hash(rid)
    with _LOCK:
        s = _wait_slot(m, me, h)
        off = _W_OFF + s * _W.size
        m[off:off + _W.size] = _W.pack(h, age_key(rid), me, _now_ms(), 1)
    return s


def unregister(arena: Any, slot: Optional[int], rid: str) -> None:
    if slot is None:
        return
    m = mapped(arena, create=False)
    if m is None:
        return
    off = _W_OFF + int(slot) * _W.size
    with _LOCK:
        try:
            wh, _k, pid, _t, _a = _W.unpack(m[off:off + _W.size])
            if pid == os.getpid() and wh == rid_hash(rid):
                m[off:off + _W.size] = _W.pack(0, 0, 0, 0, 0)
        except (ValueError, struct.error):
            pass


def registrations(arena: Any) -> List[Tuple[int, int, int]]:
    """``[(rid hash, age key, pid)]`` of the live wait registrations of every rank."""
    m = mapped(arena, create=False)
    if m is None:
        return []
    now = _now_ms()
    out = []
    for s in range(_WAIT_SLOTS):
        off = _W_OFF + s * _W.size
        try:
            h, k, pid, t, act = _W.unpack(m[off:off + _W.size])
        except (ValueError, struct.error):
            continue
        if not act or not h or now - t > _STALE_MS or not _alive(int(pid)):
            continue
        out.append((int(h), int(k), int(pid)))
    return out


# --- PP0's decision (scheduler round) ----------------------------------------------------------------

def note_role(cache: Any) -> None:
    """``read_room.tick``: this process's PP rank and the group size (the tree carries both)."""
    try:
        _A.pp_rank = int(getattr(cache, "pp_rank"))
        _A.pp_size = int(getattr(cache, "pp_size", 0) or 0)
    except (AttributeError, TypeError, ValueError):
        _A.pp_rank = None


def is_pp0() -> bool:
    return _A.pp_rank == 0


def _order_key(h: int, k: int) -> Tuple[int, int]:
    """The front's arrival number when the rid carries one; an unnumbered rid sorts after every numbered one, in the
    order PP0 first saw it registered (PP0's own order -- one decision either way)."""
    if h not in _A.first_seen:
        _A.seen_n += 1
        _A.first_seen[h] = _A.seen_n
        if len(_A.first_seen) > 4096:
            _A.first_seen.pop(next(iter(_A.first_seen)))
    return (k if k else (1 << 62) + _A.first_seen[h], _A.first_seen[h])


def decide(arena: Any) -> Optional[Tuple[int, int]]:
    """PP0 only: announce the oldest registered request as the one fill that may wait; clear the announcement when
    nobody waits. Returns the announced ``(rid hash, age key)`` (None = nobody waits / not PP0 / not mapped)."""
    if not enabled() or not is_pp0():
        return None
    regs = registrations(arena)
    m = mapped(arena, create=True)
    if m is None:
        return None
    if not regs:
        if _A.announced != (0, 0):
            _A.announced = (0, 0)
            with _LOCK:
                m[0:_HDR.size] = _HDR.pack(_MAGIC, 0, 0, os.getpid(), _A.seq, _now_ms(), _A.pp_size, 0)
        return None
    h, k = min(((h, k) for h, k, _p in regs), key=lambda hk: _order_key(*hk))
    _A.seq += 1
    with _LOCK:
        m[0:_HDR.size] = _HDR.pack(_MAGIC, h, k, os.getpid(), _A.seq, _now_ms(), _A.pp_size, 0)
    if (h, k) != _A.announced:
        _A.announced = (h, k)
        _A.announces += 1
        n = _A.announces
        if n <= 32 or n % 64 == 0:
            logger.info("%s OWNER n=%d owner=%x age=%d waiting=%d ranks=%s (PP0 decides: only the oldest request's "
                        "fill waits for arena room, every other fill ends its wait at once)", MARK, n, h, k,
                        len(regs), sorted({p for _h, _k, p in regs}))
    return h, k


# --- the fill's question -----------------------------------------------------------------------------

def now_ms() -> int:
    return _now_ms()


def may_wait(arena: Any, rid: Optional[str], since_ms: Optional[int] = None) -> bool:
    """False = PP0 announced another request as the one that waits, AFTER this fill registered at ``since_ms``
    (PP0 saw this fill and chose an older request): this fill ends its room wait now. True = this fill's request
    is the owner, nobody is announced yet, PP0 has not decided since this fill registered (a newcomer is never
    sent away by an announcement made without it), or the fill is not arbitrated (switch off, no request named,
    an arena READ-ROOM does not serve)."""
    if rid is None or not enabled():
        return True
    try:
        from sglang.srt.weg2 import read_room as _rr

        if not _rr.served(arena):
            return True
    except Exception:  # noqa: BLE001 - no READ-ROOM: the old wait
        return True
    o = owner(arena)
    if o is None or o[0] == rid_hash(rid):
        return True
    if since_ms is not None and o[3] <= int(since_ms):
        return True
    return False


def note_ended_young(arena: Any, rid: str, stems: int) -> None:
    _A.ended_young += 1
    n = _A.ended_young
    o = owner(arena)
    if n <= 32 or n % 64 == 0:
        logger.info("%s YIELD n=%d rid=%s stems=%d owner=%s (a younger request's fill ends its room wait at once; "
                    "the #257 cut at its anchor gives its landed references beyond the anchor back at the reap -- "
                    "the oldest request's fill gets the room)", MARK, n, rid, int(stems),
                    "-" if o is None else "%x/age=%d" % (o[0], o[1]))


def reset_for_tests() -> None:
    with _LOCK:
        for m in list(_A.maps.values()):
            try:
                if m is not None:
                    m.close()
            except Exception:  # noqa: BLE001
                pass
        _A.maps = {}
    _A.pp_rank = None
    _A.pp_size = 0
    _A.first_seen = {}
    _A.seen_n = 0
    _A.announced = (0, 0)
    _A.seq = 0
    _A.ended_young = 0
    _A.announces = 0
    _TL.rid = None
