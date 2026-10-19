"""S59 (19.10.): D reads what the store holds BEFORE it refuses by X.

The X gate prices a request from the group's USABLE floor when that lies below
the store arm (#1424d: the floor is what every rank admits). When the floor
fell because the request's prefix left D's tree after its read -- an earlier
admission's eviction in the same pass (rc14 D pdflip-16-211: ``H98d RU FORM-A
STALE-VOTE DEFER local=0 group=37888``, next pass ``X-GATE-TERMS head=0
store=27200 floor=0`` -> W31 on 38649, P read 37888 and computed 761) or a
SEAT-AGE displacement that requeued it with no read (rc15 D pdflip-72-525:
displaced 08:30:46, ``X-GATE-TERMS total=98592 head=0 store=68608 floor=0``
at 08:31:34 -> W31, while P's probe 11 s later read ``store_now=98560``) --
nothing re-read the store: the read that would have restored the floor is
issued only at intake, at a wake, or on a deferral mark. The refusal then
flipped the group for a context the store held.

THE RULE: a W31 whose group store arm exceeds the priced floor is not taken
on that pass. The request is marked ``store_prefix_short`` (the existing
A12.2 arm: ``_retry_deferred_prefetches`` re-issues the read from the
current match, ``_pdflip_x_defers`` holds the gate pending until it lands,
bounded by the store-read bound) and priced again after the read. Bounded
here too: one re-read per (floor, store arm) pair and at most
:data:`MAX_REREADS` per request -- a read that brings nothing new is refused
by name on the next pass, never re-read for ever.

Every input is a replicated term of the same packed reduce (total, head, store
arm, usable floor) or request state written identically on every rank, so the
verdict is one verdict on every rank (R11 / #1004 / raenge-nie-uneins).

Switch ``FLLIPER_PDFLIP_S59_STORE_REREAD`` (default ON, "0" = off).
"""

from __future__ import annotations

import os
from typing import Optional, Sequence, Tuple

ENV = "FLLIPER_PDFLIP_S59_STORE_REREAD"
MAX_REREADS = 2


def enabled(env=None) -> bool:
    e = os.environ if env is None else env
    return (e.get(ENV, "1") or "1").strip().lower() not in ("0", "false", "no", "off")


def priced_match(terms) -> Optional[int]:
    """The match the X gate priced from, re-derived from its own terms
    ``(total, head, store, floor)`` exactly as ``pdflip_uncached_extent``
    combines them (store arm credited over the head, the floor where it
    differs). None without a group opinion."""
    if not terms or terms[1] is None:
        return None
    total, head, store, floor = terms
    priced = int(head) if store is None else max(int(head), int(store))
    if floor is not None and int(floor) < priced:
        priced = int(floor)
    elif floor is not None and int(floor) > priced:
        priced = int(floor)
    return min(int(priced), int(total))


def reread_due(terms, done: Sequence[Tuple[int, int]] = (), on: bool = True):
    """``(due, key)``: is a store re-read owed instead of the W31 refusal?

    Due when the group store arm holds more prefix than the gate priced (the
    floor fell below what the store carries), this (priced, store) pair was
    not re-read yet, and the request is under :data:`MAX_REREADS`."""
    if not on or not terms or terms[1] is None:
        return False, None
    store = terms[2]
    if store is None:
        return False, None
    priced = priced_match(terms)
    if priced is None or int(store) <= int(priced):
        return False, None
    key = (int(priced), int(store))
    done = tuple(done or ())
    if key in done or len(done) >= MAX_REREADS:
        return False, key
    return True, key
