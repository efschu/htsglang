"""S109b (NF 20.10.): WHICH arena pages have their L3 copy -- read without file I/O, written by the write-behind.

User order 20.10. (verbatim): "UND WEHE DAS SCHREIBEN IN DEN CACHE BLOCKIERT DURCH IRGENDEINE SYNCHRON ODER
WARTESCHEISSE". The spills that may give an L2 page up (K23 host-leaf, S109 aux anchors, S39 child anchors, the W3
device round, READ-ROOM-D, W3-ROOM-SET, S86 sleep release) all called ``secure_rows_to_l3`` / ``arena_secure_to_disk``
in the SCHEDULER thread: a stat per stem and, for a page without a copy, a ``pageio.write_pages`` with fsync -- the
very synchronous write the order forbids.

THE RULE. The one writer of L2 -> L3 copies is the existing write-behind thread (``HiCacheFile.l3_write_behind_pass``,
thread ``pdflip-l3-write-behind``, D TP0 / P PP0: the store's LRU owner). It already keeps, per arena slot, the
(generation, key) it secured (``arena._l3wb_sec``, its per-process visit baseline -- unchanged). Every page it
secures (written, found on disk, or completed by the sibling group's writer) is now also PUBLISHED into a SHARED
mapping next to the arena (``<arena>.l3sec`` in the boot's /dev/shm arena directory), so every rank of every group
reads it as memory:

* ``secured(arena, slots)`` -- bool per slot: COMPLETE now and the write-behind secured THIS generation of it.
  No stat, no write, no wait. A mapping not there yet (no write-behind in this boot so far) = nothing secured.
* ``request(arena, slots)`` -- the scheduler only queues: the owner's next pass visits these slots first. On a rank
  without the write-behind it is a no-op (the owner's own cycle covers every COMPLETE page anyway).

A spill gives a page up only when ``secured`` says so; otherwise the node stays (``pending``) and the next spill
asks again -- like an anchor in flight. Rank agreement: the decision stays where it was (Form A: TP0 alone, the
workers follow the K23/R12 STATE); the shared record is the same bytes for every reader.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

SUFFIX = ".l3sec"
#: the arena attribute caching this process's attachment of the shared record (the write-behind keeps its own
#: per-process visit baseline ``_l3wb_sec`` as before and PUBLISHES every page it secured into this record)
ATTR = "_l3sec_pub"
_LOCK = threading.Lock()
#: requested slots kept per arena at most (a rank without the write-behind never drains its set)
URGENT_MAX = 65536


def _path(arena: Any) -> Optional[str]:
    p = getattr(arena, "path", None)
    return (str(p) + SUFFIX) if p else None


def _attach(path: str, n: int):
    gens = np.memmap(path, dtype=np.int64, mode="r+", shape=(n,), offset=0)
    lo = np.memmap(path, dtype=np.uint64, mode="r+", shape=(n,), offset=8 * n)
    return gens, lo


def arrays(arena: Any, create: bool = False):
    """``(sec_gen, sec_lo)`` of ``arena``: the shared mapping (attached once per process and cached), created by the
    write-behind (``create=True``). A reader never creates: None until the owner made it. An arena without a path
    (unit scaffolds) keeps process-local arrays when ``create`` is set, as before."""
    sec = getattr(arena, ATTR, None)
    if sec is not None:
        return sec
    n = int(getattr(arena, "slots", 0) or 0)
    if n <= 0:
        return None
    path = _path(arena)
    with _LOCK:
        sec = getattr(arena, ATTR, None)
        if sec is not None:
            return sec
        try:
            if path and not os.path.exists(path):
                if not create:
                    return None
                tmp = "%s.%d.tmp" % (path, os.getpid())
                with open(tmp, "wb") as f:
                    f.truncate(16 * n)
                g = np.memmap(tmp, dtype=np.int64, mode="r+", shape=(n,), offset=0)
                g[:] = -1
                g.flush()
                del g
                try:
                    os.link(tmp, path)          # atomic: the first creator wins, the others attach
                except FileExistsError:
                    pass
                finally:
                    try:
                        os.unlink(tmp)
                    except OSError:
                        pass
            if path:
                sec = _attach(path, n)
            elif create:
                sec = (np.full(n, -1, dtype=np.int64), np.zeros(n, dtype=np.uint64))
            else:
                return None
        except Exception as exc:  # noqa: BLE001 -- no shared record: nothing reads as secured (fail closed)
            logger.warning("S109b L3-SECURED mapping for %s unavailable: %r", path, exc)
            if not create:
                return None
            sec = (np.full(n, -1, dtype=np.int64), np.zeros(n, dtype=np.uint64))
        try:
            setattr(arena, ATTR, sec)
        except Exception:  # noqa: BLE001 -- an arena object that takes no attribute: not cached
            pass
        return sec


def secured(arena: Any, slots) -> np.ndarray:
    """Bool per slot: COMPLETE now and its current generation secured on L3 by the write-behind. Memory reads only
    (the arena header and the shared record); unreadable = False (fail closed)."""
    s = np.ascontiguousarray(np.asarray(slots, dtype=np.int64))
    out = np.zeros(s.shape, dtype=bool)
    if not s.shape[0]:
        return out
    sec = arrays(arena, create=False)
    if sec is None:
        return out
    try:
        gens = np.asarray(arena.slot_gens_np(s), dtype=np.int64)
        states = np.asarray(arena.slot_states([int(x) for x in s.tolist()]), dtype=np.int64)
        sec_gen = sec[0]
        ok = (s >= 0) & (s < sec_gen.shape[0])
        idx = np.where(ok, s, 0)
        out = ok & (states == 2) & (gens >= 0) & (np.asarray(sec_gen[idx]) == gens)
    except Exception as exc:  # noqa: BLE001 -- fail closed
        logger.warning("S109b L3-SECURED read failed: %r", exc)
        return np.zeros(s.shape, dtype=bool)
    return out


def request(arena: Any, slots) -> int:
    """Queue ``slots`` for the owner's next write-behind pass (visited first). Never I/O, never a wait."""
    vals = [int(x) for x in np.asarray(slots, dtype=np.int64).tolist()]
    if not vals:
        return 0
    with _LOCK:
        u = getattr(arena, "_l3wb_urgent", None)
        if u is None:
            u = set()
            try:
                arena._l3wb_urgent = u
            except Exception:  # noqa: BLE001
                return 0
        if len(u) + len(vals) > URGENT_MAX:
            u.clear()
        u.update(vals)
    return len(vals)


def take_urgent(arena: Any) -> np.ndarray:
    """The write-behind thread: the requested slots, drained."""
    with _LOCK:
        u = getattr(arena, "_l3wb_urgent", None)
        if not u:
            return np.zeros(0, dtype=np.int64)
        out = np.fromiter(u, dtype=np.int64, count=len(u))
        u.clear()
    return out


def mark(arena: Any, slots) -> int:
    """The writer: record that ``slots`` (their CURRENT generation) have their L3 copy. Called by the writer's own
    form (``ArenaMHAHostPool.secure_rows_to_l3_now`` / ``forma_l3_subtree._secure_child_anchors_now``) after a proven
    copy; the write-behind pass marks through its own ``sec`` arrays (the same mapping)."""
    s = np.asarray(slots, dtype=np.int64)
    if not s.shape[0]:
        return 0
    sec = arrays(arena, create=True)
    if sec is None:
        return 0
    try:
        gens = np.asarray(arena.slot_gens_np(s), dtype=np.int64)
        ok = (s >= 0) & (s < sec[0].shape[0]) & (gens >= 0)
        sec[0][s[ok]] = gens[ok]
    except Exception as exc:  # noqa: BLE001 -- an unmarked page stays "pending": safe
        logger.warning("S109b L3-SECURED mark failed: %r", exc)
        return 0
    return int(ok.sum())


def check(arena: Any, slots, backend: Any = None, writer: str = "write_behind") -> np.ndarray:
    """The scheduler's form: ``secured`` plus ``request`` for what is not -- bool per slot. ``backend`` is unused here
    (the writer's form ``check_now`` takes it; the unit tests of the spill LOGIC swap it in as "the write-behind has
    run", conftest seam)."""
    ok = secured(arena, slots)
    if not bool(ok.all()):
        request(arena, [int(s) for s, k in zip(np.asarray(slots, dtype=np.int64).tolist(), ok.tolist()) if not k])
    return ok


def check_now(arena: Any, slots, backend: Any = None, writer: str = "write_behind") -> np.ndarray:
    """The WRITER's form (never the scheduler thread): what ``check`` would queue is copied to L3 now by the store's
    own writer (``arena_secure_to_disk``, a page already on disk is not written again) and marked."""
    s = [int(x) for x in np.asarray(slots, dtype=np.int64).tolist()]
    ok = secured(arena, s)
    todo = [x for x, k in zip(s, ok.tolist()) if not k]
    secure = getattr(backend, "arena_secure_to_disk", None)
    if todo and callable(secure):
        total = int(getattr(arena, "slot_bytes", 0) or 0)
        try:
            res = secure(arena, [(x, 0, 0, total) for x in todo], writer=writer)
        except Exception as exc:  # noqa: BLE001 -- not secured
            logger.warning("S109b check_now secure raised: %r", exc)
            return ok
        if not int(res.get("lost", 0) or 0) and int(res.get("on_disk", 0) or 0) + int(res.get("written", 0) or 0) \
                >= len(todo):
            mark(arena, todo)
            return np.ones(len(s), dtype=bool)
    return ok


def publish(arena: Any, slots, gens) -> None:
    """The write-behind: publish secured (slot, generation) pairs into the shared record."""
    sec = arrays(arena, create=True)
    if sec is None:
        return
    try:
        s = np.asarray(slots, dtype=np.int64)
        g = np.asarray(gens, dtype=np.int64)
        if s.shape[0]:
            sec[0][s] = g
    except Exception as exc:  # noqa: BLE001 -- an unpublished page stays "pending": safe
        logger.warning("S109b L3-SECURED publish failed: %r", exc)
