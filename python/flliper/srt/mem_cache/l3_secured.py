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
#: FLIPTIME-1023: the stem lookups of ``secured`` / ``unsecured_keys_lo`` as arrays (``_listed_mask``); "0" = the
#: per-slot ``slot_stem`` + ``l3_listed`` form as before (A/B switch, default ON)
STEM_PTRS_ENV = "FLLIPER_L3SEC_STEM_PTRS"
#: FLIPTIME-1023: lookups that fell back to the per-slot form although the switch is on (an arena or backend without
#: the array form, or the array form raised) -- counted, the first ones named
FAST_STATS = {"fast": 0, "fallback": 0}
#: PPHASE-GAPS-1024 (27B rc39a P, boot dkr27bbar1drei10240803_b9cddabda1_1024_080355): every chunk or retain publish
#: whose claim had to make room in the 720896-slot KV arena (``#1427 ARENA-DROP slot_bytes=32768``) cost PP0's
#: scheduler thread 430-1148 ms (int8) / 1326-1863 ms (duo 1024_070103), mean of its publishes; a publish whose claim
#: dropped only in the 112-slot mamba arena cost 20-66 ms, one without a drop 15-23 ms. The arena-size term is this
#: module's census (``unsecured_keys_lo``, called by ``ArenaMHAHostPool._evict_for_claim`` for EVERY claim that needs
#: room): it asks the #1459 index again for every COMPLETE slot the record calls copied -- ~0.5 M C lookups per claim
#: on 27B (page_size 1: one 32 KiB page per token), the FLIPTIME-1023 characterisation's 128-143 ms per 262144 slots,
#: x2.75 for the KV arena, more under the write-behind's GIL. In the P phases that was 29.8 s (int8) / 41.3 s (duo)
#: of card idle on PP0 between forwards (deskq/done/pphase-gaps-1024.md).
#:
#: THE MEMO: per arena, the generation at which a slot's stem was last FOUND in the index. A slot whose record names
#: that same generation is not asked again; only slots the record newly calls copied (or whose generation moved) are.
#: The memo is PROCESS-LOCAL (an attribute of this process's ShmArena object); the record (``<arena>.l3sec``) and the
#: #1459 index are SHARED by every rank of both groups. Safety rests on exactly two things, both cross-process:
#:   (a) the RECORD GATE -- the record is read for EVERY slot on EVERY census and the memo is consulted ONLY for slots
#:       the record calls copied at their current generation; a file any process's L3 LRU took voids the shared record
#:       (``invalidate_stems`` from its on_evict), so the slot reads uncopied here whatever this process's memo holds;
#:   (b) the claim's free decision ``verify_candidates`` -> ``secured`` asks the index for every candidate WITHOUT the
#:       memo, so no page leaves L2 without its copy (an unlisted candidate is handed over, S109-A3).
#: ``census(memo) == census(no memo)`` therefore holds only where the record and the index agree. Named rests where
#: they disagree and the memo keeps a page OFF the keep list (the clock may pick it; (b) refuses it): S109-A1's own
#: window (index entry removed, on_evict not yet run); a sister process voiding AND re-publishing the same generation
#: under index lag between two censuses of THIS process (narrowed: a census that sees the record negative forgets the
#: slot's memo entry). ``invalidate_stems`` clears this process's memo; ``void_all`` (HiCacheFile.clear) voids the
#: record of the whole arena (review SOLL 1). Residue: the ``_stat_stems`` strike removes index entries without a
#: record void (a void there stalled the write-behind's QSA pair gate, test_l3_reuse_after_restart_0928 test_b).
#: Only positives are remembered: a slot not listed is asked again next time. ``LISTED_MEMO = False`` = ask every slot
#: (the FLIPTIME-1023 form); no env switch (catalog unchanged).
LISTED_MEMO = True
_MEMO_ATTR = "_l3sec_listed_gen"
#: lookups asked / answered from the memo by ``unsecured_keys_lo`` (instrument + tests)
MEMO_STATS = {"asked": 0, "memo": 0}


def _listed_memo(arena: Any, backend: Any) -> Optional[np.ndarray]:
    """The memo array of ``arena`` for ``backend`` (int64 per slot, -1 = not known listed), created on first use; a new
    backend (another index) starts a fresh memo. None when off or the arena has no slot count."""
    if not LISTED_MEMO:
        return None
    n = int(getattr(arena, "slots", 0) or 0)
    if n <= 0:
        return None
    cur = getattr(arena, _MEMO_ATTR, None)
    if cur is not None and cur[0] is backend and cur[1].shape[0] == n:
        return cur[1]
    memo = np.full(n, -1, dtype=np.int64)
    try:
        setattr(arena, _MEMO_ATTR, (backend, memo))
    except Exception:  # noqa: BLE001 -- an arena that takes no attribute: no memo (every slot asked, as before)
        return None
    return memo


def void_all(arena: Any) -> int:
    """PPHASE-GAPS-1024 (review SOLL 1): the whole store was cleared (``HiCacheFile.clear``: the #1459 index emptied,
    every file unlinked, no on_evict) -- every slot's record, this process's write-behind baseline and memo are void,
    so no census or spill reads a page as copied whose file is gone, and the write-behind sees every page as new
    again. Memory only. Returns the slots voided (0 = no record attached here)."""
    n = 0
    sec = arrays(arena, create=False)
    if sec is not None:
        sec[0][:] = -1
        n = int(sec[0].shape[0])
    base = getattr(arena, "_l3wb_sec", None)
    if base is not None:
        try:
            base[0][:] = -1
        except Exception:  # noqa: BLE001 -- a baseline of another shape: the shared record already says "not copied"
            pass
    cur = getattr(arena, _MEMO_ATTR, None)
    if cur is not None:
        cur[1][:] = -1
    return n


def _forget_listed(arena: Any, slots) -> None:
    """``invalidate_stems``: these slots' files left L3 -- the memo must not answer for them any more."""
    cur = getattr(arena, _MEMO_ATTR, None)
    if cur is None:
        return
    s = np.asarray(slots, dtype=np.int64)
    s = s[(s >= 0) & (s < cur[1].shape[0])]
    if s.shape[0]:
        cur[1][s] = -1


def stem_ptrs_on() -> bool:
    return (os.environ.get(STEM_PTRS_ENV, "1") or "1").strip().lower() not in ("0", "false", "no", "off")


def _listed_mask(arena: Any, backend: Any, slots) -> Optional[np.ndarray]:
    """FLIPTIME-1023 (boot ..._a45112428e_1023_140032): ``slot_stem(s) non-empty AND l3_listed(stem)`` per slot,
    without a Python call per slot. The S109-A7 quiesce census ran the per-slot form over every COMPLETE slot of the
    KV arena on the scheduler thread (D TP0 / P PP0), twice per flip: PDFLIP-SLEEP-SUB sweep 6 -> 2161 ms as the
    arena filled 1k -> 260k COMPLETE slots, the D>P flip 1.6 -> 6.4 s (9.9 s at 14:21Z). Here the slot headers' stem
    addresses go to the same #1459 C lookup (``L3Index.has_ptrs``) as one ``char **`` -- the same bytes, the same
    index, the same fail-closed answers (no index = nothing listed, an empty stem = not listed). None = this
    arena/backend has no array form (or the switch is off): the caller keeps the per-slot form."""
    if not stem_ptrs_on():
        return None
    ptrs_fn = getattr(arena, "slot_stem_ptrs", None)
    has_fn = getattr(backend, "l3_listed_ptrs", None) if backend is not None else None
    # the array form answers for ``l3_listed`` only where it IS that method's twin (the same ``_l3idx``): an ``l3_listed``
    # replaced on the instance or in a subclass keeps its own per-slot form
    _ll = getattr(backend, "l3_listed", None) if backend is not None else None
    if not getattr(getattr(_ll, "__func__", _ll), "_l3sec_ptrs_twin", False):
        has_fn = None
    if not callable(ptrs_fn) or not callable(has_fn):
        FAST_STATS["fallback"] += 1
        return None
    try:
        s = np.ascontiguousarray(np.asarray(slots, dtype=np.int64)).reshape(-1)
        if not s.shape[0]:
            return np.zeros(0, dtype=bool)
        ptrs, nonempty = ptrs_fn(s)
        ptrs = np.ascontiguousarray(ptrs, dtype=np.uint64)   # kept alive by this frame across the C call
        import ctypes

        has = has_fn(ptrs.ctypes.data_as(ctypes.POINTER(ctypes.c_char_p)), int(s.shape[0]))
        if has is None:                     # the index has no pointer form: the per-slot form answers
            FAST_STATS["fallback"] += 1
            return None
        out = np.asarray(nonempty, dtype=bool) & (np.asarray(has).reshape(-1) != 0)
        FAST_STATS["fast"] += 1
        return out
    except Exception as exc:  # noqa: BLE001 -- the per-slot form answers instead (named once in a while)
        FAST_STATS["fallback"] += 1
        if FAST_STATS["fallback"] <= 8 or FAST_STATS["fallback"] % 256 == 0:
            logger.warning("FLIPTIME-1023 L3SEC-STEM-PTRS fallback n=%d: %r", FAST_STATS["fallback"], exc)
        return None


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
    with _LOCK:
        if not any(a is arena and b is backend for a, b in _REG):
            _REG.append((arena, backend))


#: S109-A7: every (arena, backend) this process opened (register_backend) -- the quiesce wake-up walks it
_REG: list = []


def queue_unsecured_all(why: str = "quiesce") -> int:
    """S109-A7 (re-review 20.10.): at a quiesce's entrance (flip or sleep) every COMPLETE slot without an L3 copy is
    queued for the write-behind and the write-behind is woken -- the whole quiesce (53-410 ms, median 98 ms) is its
    time before the S86 veto. Only where the write-behind runs (the store's LRU owner: D TP0 / P PP0); one C census
    plus memory reads per arena (``unsecured_keys_lo``: ~5 ms CPU on a small arena, O(slots) -- see PPHASE-GAPS-1024), never I/O, never a wait. Returns the slots queued."""
    with _LOCK:
        reg = list(_REG)
    n = 0
    for arena, backend in reg:
        if getattr(backend, "_l3wb_thread", None) is None:
            continue
        try:
            n += int(unsecured_keys_lo(arena, backend, keys=False))
        except Exception as exc:  # noqa: BLE001 -- a wake-up, never a failure of the quiesce
            logger.warning("S109-A7 QUIESCE-WAKE %s failed: %r", why, exc)
    if n:
        k = QUIESCE_STATS["n"] = QUIESCE_STATS["n"] + 1
        QUIESCE_STATS["queued"] += n
        if k <= 32 or k % 64 == 0:
            logger.info("S109-A7 QUIESCE-WAKE n=%d why=%s queued=%d (every COMPLETE page without its L3 copy is asked "
                        "of the write-behind at the quiesce entrance; S86 vetoes not_on_l3 only what it could not "
                        "copy in the quiesce)", k, why, n)
    return n


QUIESCE_STATS = {"n": 0, "queued": 0}
#: S109-A5-M: how often the L3 LRU took the file of a page still COMPLETE in L2 (each one a rewrite trigger)
LRU_L2_STATS = {"n": 0, "bytes": 0}


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
        _st_np = getattr(arena, "slot_states_np", None) if stem_ptrs_on() else None
        states = (np.asarray(_st_np(s), dtype=np.int64) if callable(_st_np)
                  else np.asarray(arena.slot_states([int(x) for x in s.tolist()]), dtype=np.int64))
        sec_gen = sec[0]
        ok = (s >= 0) & (s < sec_gen.shape[0])
        idx = np.where(ok, s, 0)
        _live = (states == 2) | ((states == 3) if evicting_ok else False)
        out = ok & _live & (gens >= 0) & (np.asarray(sec_gen[idx]) == gens)
        listed = getattr(backend, "l3_listed", None) if backend is not None else None
        cand = np.nonzero(out)[0]
        if callable(listed) and cand.shape[0]:
            fast = _listed_mask(arena, backend, s[cand])
            if fast is not None:
                out[cand] = fast
            else:
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
            _f = np.asarray(found, dtype=np.int64)
            sec[0][_f] = -1
            _forget_listed(arena, _f)   # PPHASE-GAPS-1024: the memo never answers for a file the LRU took
            # S109-A5 (re-review R11): this process's write-behind visit baseline forgets them too (the pass also
            # compares against the shared record, which covers the sister group's writer)
            base = getattr(arena, "_l3wb_sec", None)
            if base is not None:
                base[0][_f] = -1
        return len(found)
    except Exception as exc:  # noqa: BLE001 -- the index side (AND) still answers "not on L3"
        logger.warning("S109-A1 L3-SECURED invalidate failed: %r", exc)
        return 0


def request(arena: Any, slots) -> int:
    """Queue ``slots`` for the owner's next write-behind pass (visited first) and wake it (S109-A4). Never I/O, never
    a wait."""
    vals = np.asarray(slots, dtype=np.int64).tolist()   # PPHASE-GAPS-1024: tolist() already yields Python ints
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
    if ok.any():
        promote(arena, [int(s) for s, k in zip(np.asarray(slots, dtype=np.int64).tolist(), ok.tolist()) if k],
                backend)
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


def unsecured_keys_lo(arena: Any, backend: Any = None, keys: bool = True):
    """S109d: the key low words of every COMPLETE slot WITHOUT an L3 copy (record or index) -- the arena clock's keep
    list, so no room-making frees a page that has no copy -- and those slots queued for the write-behind. One census
    call (``complete_census``, C) plus memory reads; never I/O. Returns a sorted uint64 array; ``keys=False``
    (the S109-A7 quiesce wake-up) the COUNT of slots queued, an int on EVERY path.

    S109-A7-TYPE (W65 22.10., boot 1022_003738: D TP0 00:40:15 x3 and P PP0 00:39:02 x2 "S109-A7 QUIESCE-WAKE quiesce
    failed: TypeError('only length-1 arrays can be converted to Python scalars')"): the first quiesce of a boot finds an
    empty arena, and the two early exits below returned the empty KEY ARRAY to ``int(...)`` in ``queue_unsecured_all``
    -- the count contract held only on the last line."""
    try:
        slots, gens, klo, _khi = arena.complete_census()
    except Exception as exc:  # noqa: BLE001 -- no census: nothing is known secured -- keep nothing extra, but say so
        logger.warning("S109d unsecured census failed: %r", exc)
        return np.zeros(0, dtype=np.uint64) if keys else 0
    slots = np.asarray(slots, dtype=np.int64)
    if not slots.shape[0]:
        return np.zeros(0, dtype=np.uint64) if keys else 0
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
    memo = _listed_memo(arena, backend) if callable(listed) else None
    if memo is not None and cand.shape[0] < slots.shape[0]:
        # PPHASE-GAPS-1024 (review SOLL 2): the record says "not copied" -- whatever this process remembered of the
        # slot is void (a later re-publish of the same generation must be asked again)
        memo[slots[~mine]] = -1
    if callable(listed) and cand.shape[0]:
        # PPHASE-GAPS-1024: a slot found listed at THIS generation before is not asked again (module note above).
        # The memo is consulted ONLY for record-positive slots (the record gate, (a) above).
        gens_np = np.asarray(gens, dtype=np.int64)
        if memo is not None:
            ask = cand[memo[slots[cand]] != gens_np[cand]]
            MEMO_STATS["memo"] += int(cand.shape[0] - ask.shape[0])
        else:
            ask = cand
        MEMO_STATS["asked"] += int(ask.shape[0])
        if ask.shape[0]:
            fast = _listed_mask(arena, backend, slots[ask])
            if fast is not None:
                mine[ask] = fast
            else:
                stems = [arena.slot_stem(int(x)) or "" for x in slots[ask].tolist()]
                has = listed(stems)
                mine[ask] = np.asarray([bool(st) and bool(h) for st, h in zip(stems, has)], dtype=bool)
            if memo is not None:
                hit = ask[mine[ask]]
                memo[slots[hit]] = gens_np[hit]
    un = ~mine
    if un.any():
        request(arena, slots[un])
    if not keys:
        return int(un.sum())
    return np.unique(np.asarray(klo, dtype=np.uint64)[un])


def unsecured_keys_lo_none(arena: Any, backend: Any = None, keys: bool = True):
    """The WRITER's form of ``unsecured_keys_lo`` (conftest seam only): no keep list -- the old clock, whose writer
    (``verify_candidates_now``) copies what it takes."""
    return np.zeros(0, dtype=np.uint64) if keys else 0


def verify_candidates(arena: Any, cands, backend: Any = None, writer: str = "claim_room") -> dict:
    """S109d, the scheduler's form: the clock's candidates ``(slot, key_lo, key_hi, total)`` -- EVICTING now -- judged
    by the published record AND the #1459 index, never written: ``{on_disk, written=0, lost=0, uncopied, mask}``. A
    candidate with ``mask`` False re-completed between the census and the clock: the caller does NOT free it but hands
    it over (``hand_over``: copied on the hand-over writer's own thread, then freed) -- S109-A3."""
    ok = secured(arena, [int(c[0]) for c in cands], backend, evicting_ok=True)
    if ok.any():
        promote(arena, [int(c[0]) for c, k in zip(cands, ok.tolist()) if k], backend)
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
HAND_STATS = {"queued": 0, "freed": 0, "free_refused": 0, "copy_failed": 0, "reaped": 0, "voided": 0, "dropped_without_l3": 0}


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
    # S109-A8: the generation the clock saw (``evict_candidates`` keeps it, memory) -- the writer copies only while
    # the slot is still EVICTING under it
    try:
        from flliper.srt.mem_cache.storage.file.hicache_arena import _evict_gens_for

        eg = _evict_gens_for(arena.path) if getattr(arena, "path", None) else {}
    except Exception:  # noqa: BLE001
        eg = {}
    items = [(arena, backend, (int(c[0]), int(c[1]), int(c[2]), int(c[3])), str(why), eg.get(int(c[0])))
             for c in cands]
    with _HAND_LOCK:
        _HAND.extend(items)
        HAND_STATS["queued"] += len(items)
    _hand_thread_start()
    logger.warning("S109-A3 EVICT-HANDOVER why=%s n=%d (queued total %d): candidates re-completed between the census "
                   "and the clock have no L3 copy -- NOT freed; the hand-over writer copies them, then frees",
                   why, len(items), HAND_STATS["queued"])
    return len(items)


def _hand_thread_start() -> None:
    with _HAND_LOCK:
        if not _HAND_THREAD:
            t = threading.Thread(target=_hand_loop, name="pdflip-l3-evict-handover", daemon=True)
            _HAND_THREAD.append(t)
            t.start()
    _HAND_WAKE.set()


def _dropped_without_l3(n: int) -> None:
    """S109-A8: a hand-over page that leaves L2 with no L3 copy is the #257 loss -- counted where the claim's
    ``#1427 ARENA-DROP ... dropped_without_l3`` total counts it."""
    try:
        from flliper.srt.mem_cache.pool_host.arena_pool import ArenaMHAHostPool as _P

        _P._257_dropped_without_l3 = getattr(_P, "_257_dropped_without_l3", 0) + int(n)
    except Exception:  # noqa: BLE001
        pass
    HAND_STATS["dropped_without_l3"] = HAND_STATS.get("dropped_without_l3", 0) + int(n)


def _hand_still_ours(arena: Any, slot: int, gen) -> tuple:
    """S109-A8/A8-N1: is ``slot`` still the candidate -- EVICTING under ``gen``? Stamps it against the aged reaper
    first (``ShmArena.touch_evicting``: the reaper re-reads touched_ms after its CAS and backs off), so a True stays
    true for the stale age. ``(ok, state, generation)``."""
    try:
        touch = getattr(arena, "touch_evicting", None)
        touched = bool(touch(int(slot), gen)) if callable(touch) else True
        st = int(arena.slot_states([slot])[0])
        g = int(np.asarray(arena.slot_gens_np([slot]))[0]) if gen is not None else None
    except Exception:  # noqa: BLE001 -- cannot tell: never copy into an unknown slot
        return False, -1, None
    return (touched and st == 3 and (gen is None or g == int(gen))), st, g


def _void_copy(backend: Any, arena: Any, stem: Optional[str]) -> None:
    """S109-A8-N1 backstop: the slot stopped being the candidate WHILE its copy ran (the copy outlived the stale
    age) -- the file written under the old stem may hold new bytes: unlinked, struck from the LRU and the #1459
    index, the record voided (the hand-over thread, never the scheduler)."""
    if not stem:
        return
    try:
        path = backend._sharded_path(stem)
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        ev = getattr(backend, "_evictor", None)
        if ev is not None and callable(getattr(ev, "forget", None)):
            ev.forget([stem])
        idx = getattr(backend, "_l3idx", None)
        if idx is not None and callable(getattr(idx, "remove", None)):
            idx.remove([stem])
        invalidate_stems(arena, [stem])
    except Exception as exc:  # noqa: BLE001 -- named; the counters say it happened
        logger.warning("S109-A8-N1 EVICT-HANDOVER void of %s failed: %r", stem, exc)


def _hand_drain() -> int:
    from flliper.srt.mem_cache.storage.file.hicache_arena import free_evicted_named

    with _HAND_LOCK:
        work, _HAND[:] = list(_HAND), []
    done = 0
    for item in work:
        arena, backend, cand, why = item[:4]
        gen = item[4] if len(item) > 4 else None
        if _PROMOTE:
            _promote_drain()                       # re-review (low): a promotion never waits behind a long drain
        # S109-A8 (re-review): still OUR candidate? EVICTING under the generation the clock saw. A slot the aged
        # reaper (arena_reap_stale, 60 s) freed -- and a new claim may have filled -- is never copied (new bytes under
        # the old stem: the READ-DEPTH 1014 class); its page left L2 without a copy: dropped_without_l3.
        # S109-A8-N1: the check STAMPS the slot first (touch_evicting), so the reaper cannot take it between this
        # check and the copy below.
        ok_slot, st, g = _hand_still_ours(arena, cand[0], gen)
        if not ok_slot:
            HAND_STATS["reaped"] = HAND_STATS.get("reaped", 0) + 1
            _dropped_without_l3(1)
            logger.warning("S109-A8 EVICT-HANDOVER slot=%d why=%s state=%d gen=%s/%s: reaped or re-claimed before its "
                           "copy -- not copied (no new bytes under the old stem); dropped_without_l3=1 (total %d)",
                           cand[0], why, st, g, gen, HAND_STATS["dropped_without_l3"])
            continue
        try:
            stem = (getattr(arena, "_stems", None) or {}).get((int(cand[1]), int(cand[2]))) or arena.slot_stem(cand[0])
        except Exception:  # noqa: BLE001
            stem = None
        secure = getattr(backend, "arena_secure_to_disk", None)
        ok = False
        res = {}
        if callable(secure):
            try:
                res = secure(arena, [cand], writer="evict_handover") or {}
                ok = not int(res.get("lost", 0) or 0) and int(res.get("on_disk", 0) or 0) + \
                    int(res.get("written", 0) or 0) >= 1
            except Exception as exc:  # noqa: BLE001 -- the slot stays EVICTING, named
                logger.warning("S109-A3 EVICT-HANDOVER copy raised for slot %d: %r", cand[0], exc)
        # S109-A8-N1 backstop: the copy outlived the stamp (> the stale age) and the slot moved meanwhile
        still, st2, g2 = _hand_still_ours(arena, cand[0], gen)
        if not still:
            if int(res.get("written", 0) or 0):
                _void_copy(backend, arena, stem)
                HAND_STATS["voided"] = HAND_STATS.get("voided", 0) + 1
            HAND_STATS["reaped"] = HAND_STATS.get("reaped", 0) + 1
            _dropped_without_l3(1)
            logger.warning("S109-A8-N1 EVICT-HANDOVER slot=%d why=%s state=%d gen=%s/%s: the slot moved WHILE its copy "
                           "ran -- the copy is voided (written=%s), dropped_without_l3=1 (total %d)", cand[0], why, st2,
                           g2, gen, res.get("written", 0), HAND_STATS["dropped_without_l3"])
            continue
        if ok:
            freed = free_evicted_named(arena, [cand[0]], "evict_handover")
            if freed is None or int(freed) >= 1:   # S109-A8-N2: counted only when the free really happened
                HAND_STATS["freed"] += 1
                done += 1
            else:
                HAND_STATS["free_refused"] = HAND_STATS.get("free_refused", 0) + 1
                logger.warning("S109-A8-N2 EVICT-HANDOVER slot=%d why=%s: copied, but the generation-bound free was "
                               "refused (the slot moved after the copy) -- not counted freed", cand[0], why)
        else:
            HAND_STATS["copy_failed"] += 1
            _dropped_without_l3(1)
            logger.warning("S109-A3 EVICT-HANDOVER slot=%d why=%s: no L3 copy could be made -- kept EVICTING, the aged "
                           "reaper frees it without a copy: dropped_without_l3=1 (total %d)", cand[0], why,
                           HAND_STATS["dropped_without_l3"])
    return done


# ---------------------------------------------------------------------------------------------------------------------
# S109-A6 (re-review R8, ZERO-ANCHOR 1015): a page that leaves L2 because L3 holds it makes that L3 file the ONLY copy
# -- it must be the youngest in the store's LRU (``HiCacheFile._l3_refresh_kept``, in-memory form), or the LRU takes
# exactly it next (pdflip-40-179: lost from BOTH tiers). The old writer form promoted inside ``arena_secure_to_disk``;
# the scheduler forms (``check``, ``verify_candidates``, ``secure_rows_to_l3``, the S39 child anchors) only QUEUE the
# stems here; the hand-over thread promotes them (the evictor's lock is never taken on the scheduler thread). Only the
# eviction OWNER keeps an LRU (review anchor0 B1): elsewhere this is a no-op, as before.
_PROMOTE: dict = {}
PROMOTE_MAX = 262144
PROMOTE_STATS = {"queued": 0, "refreshed": 0, "missed": 0, "overflow": 0}


def promote(arena: Any, slots, backend: Any = None) -> int:
    """Queue the L3 LRU promotion of ``slots``' stems (memory: ``slot_stem``); never I/O, never the evictor lock."""
    if backend is None:
        backend = getattr(arena, "_l3sec_backend", None)
    ev = getattr(backend, "_evictor", None) if backend is not None else None
    if ev is None or not getattr(ev, "enabled", False) or not callable(getattr(backend, "_l3_refresh_kept", None)):
        return 0
    stems = []
    for x in slots:
        try:
            st = arena.slot_stem(int(x))
        except Exception:  # noqa: BLE001
            st = None
        if st:
            stems.append(st)
    if not stems:
        return 0
    with _HAND_LOCK:
        ent = _PROMOTE.get(id(backend))
        if ent is None:
            ent = _PROMOTE[id(backend)] = (backend, {})
        if len(ent[1]) + len(stems) > PROMOTE_MAX:
            PROMOTE_STATS["overflow"] += len(stems)
            logger.warning("S109-A6 L3-PROMOTE queue full (%d): %d stems not promoted", len(ent[1]), len(stems))
            return 0
        for st in stems:
            ent[1][st] = None                      # insertion order = release order (the LRU's order of touches)
        PROMOTE_STATS["queued"] += len(stems)
    _hand_thread_start()
    return len(stems)


def _promote_drain() -> int:
    with _HAND_LOCK:
        work = [(b, list(d)) for b, d in _PROMOTE.values() if d]
        for _b, d in _PROMOTE.values():
            d.clear()
    n = 0
    for backend, stems in work:
        try:
            # S109-A6-M: a stem no longer in the LRU when the drain comes was taken in the window between the
            # release and its promotion (missed)
            lru = getattr(getattr(backend, "_evictor", None), "_lru", None)
            if lru is not None:
                PROMOTE_STATS["missed"] += sum(1 for st in stems if st not in lru)
            n += int(backend._l3_refresh_kept(stems, mtime=False) or 0)
        except Exception as exc:  # noqa: BLE001 -- recency is best effort, named
            logger.warning("S109-A6 L3-PROMOTE failed for %d stems: %r", len(stems), exc)
    PROMOTE_STATS["refreshed"] += n
    return n


STATS_LOG_S = 60.0
_STATS_LAST = {"t": 0.0, "snap": None}


def stats_line(force: bool = False) -> Optional[str]:
    """S109-A6-M / A5-M: one throttled line with every S109 counter (promotion, hand-over, quiesce wake-up, L3 LRU
    takes of L2 pages) -- at most every STATS_LOG_S, and only when something moved."""
    import time as _t

    now = _t.monotonic()
    snap = (tuple(PROMOTE_STATS.values()), tuple(HAND_STATS.values()), tuple(QUIESCE_STATS.values()),
            tuple(LRU_L2_STATS.values()))
    if not force and (now - _STATS_LAST["t"] < STATS_LOG_S or snap == _STATS_LAST["snap"]):
        return None
    _STATS_LAST["t"], _STATS_LAST["snap"] = now, snap
    line = ("S109 L3-STATS promote queued=%d refreshed=%d missed=%d overflow=%d | handover queued=%d freed=%d "
            "free_refused=%d copy_failed=%d reaped=%d voided=%d dropped_without_l3=%d | quiesce n=%d queued=%d | lru_took_l2 n=%d "
            "bytes=%d" % (PROMOTE_STATS["queued"], PROMOTE_STATS["refreshed"], PROMOTE_STATS["missed"],
                          PROMOTE_STATS["overflow"], HAND_STATS["queued"], HAND_STATS["freed"],
                          HAND_STATS["free_refused"], HAND_STATS["copy_failed"], HAND_STATS["reaped"], HAND_STATS["voided"],
                          HAND_STATS["dropped_without_l3"], QUIESCE_STATS["n"], QUIESCE_STATS["queued"],
                          LRU_L2_STATS["n"], LRU_L2_STATS["bytes"]))
    logger.info(line)
    return line


def _hand_loop() -> None:
    while True:
        try:
            _HAND_WAKE.wait(1.0)
            _HAND_WAKE.clear()
            if _PROMOTE:
                _promote_drain()                   # S109-A6: before a hand-over copy can make the LRU evict
            if _HAND:
                _hand_drain()
            stats_line()                           # S109-A6-M: throttled, only when a counter moved
        except Exception as exc:  # noqa: BLE001 -- the writer never dies
            logger.warning("S109-A3 EVICT-HANDOVER loop error: %r", exc)
