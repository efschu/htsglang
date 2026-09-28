"""#257 (a)/(e): a store probe HOLDS the arena pages it reports until the read.

Vision boot 0928 (P PP0 06:20:28, weg2-4-27): the prefetch probe counted 813
of 830 pages (the first 768 in six full batches of 128) and took no reference;
the read, batch by batch in the aux thread, found page 219 neither COMPLETE in
the arena nor on disk and ended there (completed=14016 of 52032 host tokens,
0.04 s after the operation started, budget 52.8 s). Between the two, any claim
on the shared arena (PP0/PP1/PP2 and D TP0, 5461 slots, 92-99 % complete) may
free an unreferenced COMPLETE slot (#1427 ``_evict_for_claim``) -- exactly the
pages the probe had just promised.

Here the probe takes one reader reference per reported page that is COMPLETE
in the arena (``pin``), the group's MIN trims the pins above the agreed count
(``release`` from that page on), the read ADOPTS a held page instead of
referencing it again (``adopt``), and the read's end, a revoke or a failure
gives the rest back (``release``). A page on disk only is not pinned: the read
fills it from L3.

(e) A hold never blocks a claim: a pinned slot is merely not an eviction
candidate, and the pins of one process are bounded to half the arena
(``_HOLD_SHARE``) so a claim always finds room elsewhere; a hold older than
``PROBE_HOLD_MAX_S`` is given back by name before the read.
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

logger = logging.getLogger(__name__)

#: a probe's hold is released, named, if the read has not started after this
PROBE_HOLD_MAX_S = 30.0
#: the pins of ONE process never exceed this share of the arena's slots
_HOLD_SHARE = 0.5

_lock = threading.Lock()
_held_total = 0
_log_n = 0


def held_total() -> int:
    return _held_total


def _arena_of(pool):
    arena = getattr(pool, "arena", None)
    if arena is None or not getattr(pool, "arena_read", False):
        return None
    return arena


def _note(msg, *args):
    global _log_n
    _log_n += 1
    if _log_n <= 16 or _log_n % 256 == 0:
        logger.info(msg, *args)


def pin(operation, pool, stems) -> int:
    """Reference every page of ``stems`` that is COMPLETE in the arena now.
    Stores ``operation.probe_pins`` (int64 per page, -1 = not held) and
    ``operation.probe_pins_t0``. Returns how many pages are held."""
    global _held_total
    arena = _arena_of(pool)
    n = len(stems)
    if arena is None or n == 0:
        return 0
    fs, st = arena.find_slots_np(stems)
    fs = np.asarray(fs, dtype=np.int64)
    st = np.asarray(st)
    pins = np.full(n, -1, dtype=np.int64)
    cand = np.flatnonzero((fs >= 0) & (st == 2))
    with _lock:
        room = max(0, int(int(getattr(arena, "slots", 0) or 0) * _HOLD_SHARE) - _held_total)
    capped = 0
    if cand.size > room:
        capped = int(cand.size - room)
        cand = cand[:room]
    held = 0
    batch = getattr(arena, "ref_slots_mask_np", None)
    if callable(batch) and cand.size:
        # PB (28.09., 27B rc12z24 P weg2-0-13: 45055 pages, 7.1 s of
        # LOAD-DEVICE queue_ms=7809 before its 50 ms read): ONE call with the
        # per-slot verdict instead of one ctypes call per page -- a slot that
        # left COMPLETE between find and ref still refuses alone, the rest stay
        # exact (and ledgered by name)
        ok = np.asarray(batch(fs[cand]), dtype=np.int8)
        took = cand[ok == 1]
        pins[took] = fs[took]
        held = int(took.size)
    else:
        for i in cand.tolist():
            # one slot at a time: a slot that left COMPLETE between find and ref
            # refuses its reference alone, the rest stay exact (and ledgered)
            if arena.ref_slots([int(fs[i])], +1) == 1:
                pins[i] = int(fs[i])
                held += 1
    with _lock:
        _held_total += held
    operation.probe_pins = pins
    operation.probe_pins_t0 = time.monotonic()
    _note("#257 PROBE-HOLD rid=%s pages=%d held=%d capped=%d held_total=%d",
          getattr(operation, "request_id", "?"), n, held, capped, _held_total)
    return held


def release(operation, pool, start: int = 0, reason: str = "") -> int:
    """Give back every held page from ``start`` on. Returns how many."""
    global _held_total
    pins = getattr(operation, "probe_pins", None)
    if pins is None:
        return 0
    sel = pins[int(start):]
    live = sel[sel >= 0]
    n = int(live.size)
    if n:
        arena = getattr(pool, "arena", None)
        if arena is not None:
            arena.ref_slots(live.tolist(), -1)
        with _lock:
            _held_total = max(0, _held_total - n)
    pins[int(start):] = -1
    if n and reason:
        _note("#257 PROBE-HOLD released rid=%s pages=%d from=%d reason=%s held_total=%d",
              getattr(operation, "request_id", "?"), n, int(start), reason, _held_total)
    return n


def adopt(operation, start: int, count: int):
    """The held slots of pages [start, start+count) (int64, -1 = not held),
    or None when this operation holds nothing."""
    pins = getattr(operation, "probe_pins", None)
    if pins is None:
        return None
    out = pins[int(start):int(start) + int(count)].copy()
    if out.size < count:
        out = np.concatenate([out, np.full(int(count) - out.size, -1, dtype=np.int64)])
    return out


def consumed(operation, start: int, count: int) -> None:
    """Pages [start, start+count) were read: their held reference is now the
    read's own (released with the node as every read reference is)."""
    global _held_total
    pins = getattr(operation, "probe_pins", None)
    if pins is None or count <= 0:
        return
    sel = pins[int(start):int(start) + int(count)]
    n = int((sel >= 0).sum())
    sel[:] = -1
    with _lock:
        _held_total = max(0, _held_total - n)


def expire_if_stale(operation, pool, now: float = None) -> bool:
    """(e): a hold older than PROBE_HOLD_MAX_S is given back, named."""
    t0 = getattr(operation, "probe_pins_t0", None)
    if t0 is None:
        return False
    now = time.monotonic() if now is None else now
    if now - t0 <= PROBE_HOLD_MAX_S:
        return False
    n = release(operation, pool, 0)
    logger.warning("#257 PROBE-HOLD EXPIRED rid=%s held=%.1fs > %.0fs: %d page(s) given back "
                   "before the read (a hold never outlives its bound)",
                   getattr(operation, "request_id", "?"), now - t0, PROBE_HOLD_MAX_S, n)
    return True
