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
#: a reader without the record re-tries attaching it at most this often (s) -- no stat per call
ATTACH_RETRY_S = 1.0
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
    if not create:
        # a reader asks the file system at most once per ATTACH_RETRY_S per arena until the owner made the record
        # (afterwards the attachment is cached: memory reads only)
        import time as _t

        _now = _t.monotonic()
        if _now < float(getattr(arena, "_l3sec_retry_at", 0.0) or 0.0):
            return None
        try:
            arena._l3sec_retry_at = _now + ATTACH_RETRY_S
        except Exception:  # noqa: BLE001
            pass
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


def register_backend(arena: Any, backend: Any) -> None:
    """S109-A1: the store backend that owns ``arena`` (its #1459 index answers ``l3_listed``); set where the backend
    opens the arena, so every reader judges with the index too."""
    try:
        arena._l3sec_backend = backend
    except Exception:  # noqa: BLE001 -- an arena that takes no attribute: callers pass the backend
        pass


def secured(arena: Any, slots, backend: Any = None, evicting_ok: bool = False) -> np.ndarray:
    """Bool per slot: COMPLETE now AND its page on L3. S109-A1 (review 20.10., R4/R5): BOTH sources must say so --
    the write-behind's published record (this generation was copied) AND the shared #1459 L3 stem index
    (``HiCacheFile.l3_listed``: the evictor's exact bookkeeping; its LRU unlink removes the entry, so a file the L3 cap
    took is NOT secured any more). The record alone is never enough while a backend is known (passed, or registered
    with the arena); an unopened index lists nothing (fail closed). Memory reads only; unreadable = False.
    ``evicting_ok``: the clock's own candidates (state EVICTING, 3) are judged too."""
    s = np.ascontiguousarray(np.asarray(slots, dtype=np.int64))
    out = np.zeros(s.shape, dtype=bool)
    if not s.shape[0]:
        return out
    if backend is None:
        backend = getattr(arena, "_l3sec_backend", None)
    sec = arrays(arena, create=False)
    if sec is None:
        return out
    try:
        gens = np.asarray(arena.slot_gens_np(s), dtype=np.int64)
        states = np.asarray(arena.slot_states([int(x) for x in s.tolist()]), dtype=np.int64)
        sec_gen = sec[0]
        ok = (s >= 0) & (s < sec_gen.shape[0])
        idx = np.where(ok, s, 0)
        _live = (states == 2) | ((states == 3) if evicting_ok else False)
        out = ok & _live & (gens >= 0) & (np.asarray(sec_gen[idx]) == gens)
        listed = getattr(backend, "l3_listed", None) if backend is not None else None
        cand = np.nonzero(out)[0]
        if callable(listed) and cand.shape[0]:
            stems = [arena.slot_stem(int(x)) or "" for x in s[cand].tolist()]
            has = listed(stems)
            out[cand] = np.asarray([bool(st) and bool(h) for st, h in zip(stems, has)], dtype=bool)
    except Exception as exc:  # noqa: BLE001 -- fail closed
        logger.warning("S109b L3-SECURED read failed: %r", exc)
        return np.zeros(s.shape, dtype=bool)
    return out


def invalidate_stems(arena: Any, stems) -> int:
    """S109-A1: the L3 LRU evicted these files -- the record entries of their slots in ``arena`` are void (memory)."""
    sec = arrays(arena, create=False)
    if sec is None or not stems:
        return 0
    try:
        found = [int(sl) for sl, _st in arena.find_slots(list(stems)) if int(sl) >= 0]
        if found:
            sec[0][np.asarray(found, dtype=np.int64)] = -1
        return len(found)
    except Exception as exc:  # noqa: BLE001 -- the index side (AND) still answers "not on L3"
        logger.warning("S109-A1 L3-SECURED invalidate failed: %r", exc)
        return 0


def request(arena: Any, slots) -> int:
    """Queue ``slots`` for the owner's next write-behind pass (visited first) and wake it (S109-A4). Never I/O, never
    a wait."""
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
    WAKE.set()   # S109-A4: the write-behind runs now, not at its next tick
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
    ok = secured(arena, slots, backend)
    if not bool(ok.all()):
        request(arena, [int(s) for s, k in zip(np.asarray(slots, dtype=np.int64).tolist(), ok.tolist()) if not k])
    return ok


def check_now(arena: Any, slots, backend: Any = None, writer: str = "write_behind") -> np.ndarray:
    """The WRITER's form (never the scheduler thread): what ``check`` would queue is copied to L3 now by the store's
    own writer (``arena_secure_to_disk``, a page already on disk is not written again) and marked."""
    s = [int(x) for x in np.asarray(slots, dtype=np.int64).tolist()]
    ok = secured(arena, s, backend)
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


def unsecured_keys_lo(arena: Any, backend: Any = None):
    """S109d: the key low words of every COMPLETE slot WITHOUT an L3 copy (record or index) -- the arena clock's keep
    list, so no room-making frees a page that has no copy -- and those slots queued for the write-behind. One census
    call (``complete_census``, C) plus memory reads; never I/O. Returns a sorted uint64 array."""
    try:
        slots, gens, klo, _khi = arena.complete_census()
    except Exception as exc:  # noqa: BLE001 -- no census: nothing is known secured -- keep nothing extra, but say so
        logger.warning("S109d unsecured census failed: %r", exc)
        return np.zeros(0, dtype=np.uint64)
    slots = np.asarray(slots, dtype=np.int64)
    if not slots.shape[0]:
        return np.zeros(0, dtype=np.uint64)
    if backend is None:
        backend = getattr(arena, "_l3sec_backend", None)
    sec = arrays(arena, create=False)
    if sec is not None:
        mine = np.asarray(sec[0][slots]) == np.asarray(gens, dtype=np.int64)
    else:
        mine = np.zeros(slots.shape, dtype=bool)
    # S109-A1: record AND index (a file the L3 LRU took is no copy)
    cand = np.nonzero(mine)[0]
    listed = getattr(backend, "l3_listed", None) if backend is not None else None
    if callable(listed) and cand.shape[0]:
        stems = [arena.slot_stem(int(x)) or "" for x in slots[cand].tolist()]
        has = listed(stems)
        mine[cand] = np.asarray([bool(st) and bool(h) for st, h in zip(stems, has)], dtype=bool)
    un = ~mine
    if un.any():
        request(arena, slots[un])
    return np.unique(np.asarray(klo, dtype=np.uint64)[un])


def unsecured_keys_lo_none(arena: Any, backend: Any = None):
    """The WRITER's form of ``unsecured_keys_lo`` (conftest seam only): no keep list -- the old clock, whose writer
    (``verify_candidates_now``) copies what it takes."""
    return np.zeros(0, dtype=np.uint64)


def verify_candidates(arena: Any, cands, backend: Any = None, writer: str = "claim_room") -> dict:
    """S109d, the scheduler's form: the clock's candidates ``(slot, key_lo, key_hi, total)`` -- EVICTING now -- judged
    by the published record AND the #1459 index, never written: ``{on_disk, written=0, lost=0, uncopied, mask}``. A
    candidate with ``mask`` False re-completed between the census and the clock: the caller does NOT free it but hands
    it over (``hand_over``: copied on the hand-over writer's own thread, then freed) -- S109-A3."""
    ok = secured(arena, [int(c[0]) for c in cands], backend, evicting_ok=True)
    return {"on_disk": int(ok.sum()), "written": 0, "lost": 0, "uncopied": int((~ok).sum()),
            "mask": [bool(x) for x in ok.tolist()]}


def verify_candidates_now(arena: Any, cands, backend: Any = None, writer: str = "claim_room") -> dict:
    """The WRITER's form (never the scheduler thread; conftest seam): the pre-S109d #257 (d) write of the candidates
    by the store's own writer, byte for byte, and the copies marked."""
    secure = getattr(backend, "arena_secure_to_disk", None)
    if not callable(secure):
        return {"on_disk": 0, "written": 0, "lost": len(cands)}
    try:
        sec = secure(arena, cands, writer=writer)
    except TypeError:
        sec = secure(arena, cands)
    if not int(sec.get("lost", 0) or 0):
        mark(arena, [int(c[0]) for c in cands])
    return sec


# ---------------------------------------------------------------------------------------------------------------------
# S109-A3/A4: the hand-over writer and the write-behind's wake-up.
#
# A clock / claim candidate is EVICTING (out of the index) once chosen; one that turns out to have no L3 copy (it
# re-completed between the census and the clock) must neither be freed (no copy) nor written on the scheduler thread.
# It goes to this process's hand-over writer: one daemon thread that copies it with the store's own writer
# (``arena_secure_to_disk``, the evictor's reserve/commit) and then frees it (generation-bound ``free_evicted_named``);
# a failed copy keeps the slot EVICTING and names it (the aged reaper is the old last resort). Every process gets it,
# because the write-behind runs only on the store owner (D TP0 / P PP0) while claims happen on every rank.
#
# ``WAKE`` is set by ``request`` and ``hand_over``: the write-behind and the hand-over writer wait on it, not on their
# tick alone (review S109-4: a fresh copy no longer costs a whole 2 s tick).
WAKE = threading.Event()
_HAND: list = []
_HAND_LOCK = threading.Lock()
_HAND_THREAD: list = []
_HAND_WAKE = threading.Event()
HAND_STATS = {"queued": 0, "freed": 0, "copy_failed": 0}


def wait_or_stop(stop: "threading.Event", timeout_s: float) -> bool:
    """The write-behind loop: wait up to ``timeout_s`` or until a spill asked for a copy; True = stop."""
    if stop.is_set():
        return True
    WAKE.wait(max(0.0, float(timeout_s)))
    WAKE.clear()
    return stop.is_set()


def hand_over(arena: Any, cands, backend: Any, why: str) -> int:
    """S109-A3: queue EVICTING candidates without a copy for the hand-over writer (never I/O here)."""
    if not cands:
        return 0
    items = [(arena, backend, (int(c[0]), int(c[1]), int(c[2]), int(c[3])), str(why)) for c in cands]
    with _HAND_LOCK:
        _HAND.extend(items)
        HAND_STATS["queued"] += len(items)
        if not _HAND_THREAD:
            t = threading.Thread(target=_hand_loop, name="pdflip-l3-evict-handover", daemon=True)
            _HAND_THREAD.append(t)
            t.start()
    _HAND_WAKE.set()
    logger.warning("S109-A3 EVICT-HANDOVER why=%s n=%d (queued total %d): candidates re-completed between the census "
                   "and the clock have no L3 copy -- NOT freed; the hand-over writer copies them, then frees",
                   why, len(items), HAND_STATS["queued"])
    return len(items)


def _hand_drain() -> int:
    from flliper.srt.mem_cache.storage.file.hicache_arena import free_evicted_named

    with _HAND_LOCK:
        work, _HAND[:] = list(_HAND), []
    done = 0
    for arena, backend, cand, why in work:
        secure = getattr(backend, "arena_secure_to_disk", None)
        ok = False
        if callable(secure):
            try:
                res = secure(arena, [cand], writer="evict_handover")
                ok = not int(res.get("lost", 0) or 0) and int(res.get("on_disk", 0) or 0) + \
                    int(res.get("written", 0) or 0) >= 1
            except Exception as exc:  # noqa: BLE001 -- the slot stays EVICTING, named
                logger.warning("S109-A3 EVICT-HANDOVER copy raised for slot %d: %r", cand[0], exc)
        if ok:
            free_evicted_named(arena, [cand[0]], "evict_handover")
            HAND_STATS["freed"] += 1
            done += 1
        else:
            HAND_STATS["copy_failed"] += 1
            logger.warning("S109-A3 EVICT-HANDOVER slot=%d why=%s: no L3 copy could be made -- kept EVICTING (the aged "
                           "reaper is the last resort), named", cand[0], why)
    return done


def _hand_loop() -> None:
    while True:
        try:
            _HAND_WAKE.wait(1.0)
            _HAND_WAKE.clear()
            if _HAND:
                _hand_drain()
        except Exception as exc:  # noqa: BLE001 -- the writer never dies
            logger.warning("S109-A3 EVICT-HANDOVER loop error: %r", exc)
