"""H91a (25.09.2026): group P admits its queued backlog in order; the intake
stall is left to the request this phase cannot admit at all.

THE SPECIMEN. cu130 acceptance dkrnfbar1agent09252021 (1a4e6ff19a), P log
20:40:49-20:42:10: rid 53572be2 (42,442 tokens) sat in P's queue for 61 s
with NOTHING running, ``pool_avail=220480`` and the finished neighbour's
41,664 rows ``tracked_evictable`` (0 locked) -- the pool never bound. The
admission-wedge detector then answered it 503 WEG2-INTAKE-STALL
(``gate=admission-wedge``; ``rem_total_tokens=-1`` only means "no adder on
that path") and the front paid a flip for it.

WHAT BLOCKED. #1400's told verdict (``weg2_store_told.admission``) is taken
ONCE: it pops ``told_map[rid]`` (and the loaded credit) on the first
admission visit. That visit ran at 20:40:58 (PHASE-PURITY STORE WITNESS n=30,
the line emitted right after the told gate returns) in the pass that carried
d762c570's chunked continuation; the adder had no room for a second request
in that pass, so 53572be2 stayed queued. Every later visit found no told
(``pp0_publish`` had already dropped the rid from its held set, so nothing
re-publishes) and skipped it as ``weg2_store_told_pending`` -- the witness
never printed again on any rank (n=30 is the last call).

1. :func:`told_admission` KEEPS a rank's told verdict until the request
   leaves the waiting queue. A later visit of the SAME request object admits
   on the kept verdict (same prefix cap, same credit); a fresh told for the
   rid (a re-intake) replaces it; :func:`settle_told` drops what left the
   queue (admitted or aborted). Every rank keeps its own verdict the same
   way, so the ranks stay equal.

2. :func:`intake_phase_verdict` decides whether an observed refusal is the
   intake stall at all. A finished prefill of this phase is EVICTABLE (its
   rows are released into the tree at the finish; under ``write_back`` an
   eviction writes an unbacked node to the host arena first and demotes it,
   the arena spills to the L3 file store, and D reads the arena or, after an
   arena eviction, the file via the #1433 L3->L2 fill). So a request that fits
   free + evictable rows is admitted in order as soon as the adder reaches it
   -- never answered 503. Only a request larger than the pool, or one the
   pool cannot give even after every evictable row with nothing in flight
   that could free more, is the stall.

Bookkeeping and reads only; nothing here allocates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional

from sglang.srt.weg2.intake_stall import (
    INTAKE_FITS,
    INTAKE_IMPOSSIBLE,
    INTAKE_WAITS,
    intake_verdict,
)

logger = logging.getLogger(__name__)

#: scheduler attribute holding the kept verdicts ({rid: _Kept})
KEPT_ATTR = "_weg2_told_kept"
_LOG_FIRST = 8
_LOG_EVERY = 256


@dataclass
class _Kept:
    req: Any
    told: int
    credit: int


def _kept(scheduler) -> Dict[str, _Kept]:
    kept = getattr(scheduler, KEPT_ATTR, None)
    if kept is None:
        kept = {}
        setattr(scheduler, KEPT_ATTR, kept)
    return kept


def told_admission(
    scheduler,
    req,
    note_skip: Callable[[str, Any], None],
    admission: Callable[..., Optional[int]],
) -> Optional[int]:
    """The #1400 admission gate with the verdict kept across visits.

    ``admission`` is ``weg2_store_told.admission``. Returns what it returns:
    ``None`` = skip this pass (no told yet), else the loaded credit."""
    kept = _kept(scheduler)
    rid = str(getattr(req, "rid", ""))
    told_map = getattr(scheduler, "_weg2_store_told", None) or {}
    entry = kept.get(rid)
    if entry is not None:
        if entry.req is req and rid not in told_map:
            try:
                # TK: the same RAW-token cap admission planted (told counts
                # keys; bigram trees need +1 -- prefix_cap_tokens)
                from sglang.srt.managers.weg2_store_told import prefix_cap_tokens

                req._weg2_prefix_cap = prefix_cap_tokens(
                    getattr(scheduler, "tree_cache", None), int(entry.told)
                )
            except Exception:  # noqa: BLE001 - a double without the field
                pass
            n = getattr(scheduler, "_h91_told_kept_n", 0) + 1
            scheduler._h91_told_kept_n = n
            if n <= _LOG_FIRST or n % _LOG_EVERY == 0:
                logger.info(
                    "H91 STORE-TOLD KEPT rid=%s told=%d credit=%d (n=%d): the "
                    "verdict of an earlier visit that did not admit stands "
                    "(it used to be consumed there and the rid skipped as "
                    "weg2_store_told_pending for ever: cu130 53572be2)",
                    rid[:8], entry.told, entry.credit, n,
                )
            return entry.credit
        # another request object under this rid, or a fresh told arrived for
        # it (re-intake): the old verdict is not this visit's
        kept.pop(rid, None)
    told = told_map.get(rid)
    credit = admission(scheduler, req, note_skip)
    if credit is not None and told is not None:
        kept[rid] = _Kept(req=req, told=int(told), credit=int(credit))
    return credit


def settle_told(scheduler, waiting_queue: Iterable[Any]) -> int:
    """After a committed pass: drop the kept verdicts of requests that are no
    longer queued (admitted into the batch, or aborted). Returns the count."""
    kept = getattr(scheduler, KEPT_ATTR, None)
    if not kept:
        return 0
    queued = {id(r) for r in waiting_queue}
    gone = [rid for rid, e in kept.items() if id(e.req) not in queued]
    for rid in gone:
        kept.pop(rid, None)
    return len(gone)


def pool_terms(tree, alloc):
    """(free, evictable, inflight) as the adder funds an extend, or None when
    the terms cannot be read (a desk double): the caller then keeps the
    pre-H91 answer. ``free`` is the group-published floor where one exists
    (``uniform_avail_for_evict``), the local pool otherwise."""
    if tree is None or alloc is None:
        return None
    try:
        # mem_cache.common.uniform_avail_for_evict, read inline: that module
        # pulls the quantization stack, and this module must stay light.
        floor = getattr(tree, "uniform_avail_floor", None)
        if floor is None:
            free = int(alloc.available_size())
        else:
            admitted = getattr(tree, "uniform_admitted_since_floor", 0)
            if isinstance(admitted, bool) or not isinstance(admitted, int):
                admitted = 0
            free = max(0, int(floor) - int(admitted))
        ev_fn = getattr(tree, "full_evictable_size", None)
        if not callable(ev_fn):
            ev_fn = getattr(tree, "evictable_size")
        evictable = int(ev_fn())
    except Exception:  # noqa: BLE001 - unreadable terms decide nothing
        return None
    inflight = bool(getattr(tree, "ongoing_write_through", None)) or bool(
        getattr(tree, "ongoing_load_back", None)
    )
    return free, evictable, inflight


def intake_phase_verdict(scheduler, rid: str, need_tokens: int,
                         pool_tokens: int, gate: str) -> Optional[str]:
    """INTAKE_FITS / INTAKE_WAITS (not a stall: the request stays queued and
    is admitted in order), INTAKE_IMPOSSIBLE (the stall), None = unknown.
    A non-stall verdict is named once per rid together with the last pass's
    skip census, so a request that fits and still does not start names its
    gate instead of being answered 503. The seat gate (``gate=seats``: no
    request slot with nothing running) is not a token question and keeps
    its pre-H91 answer (None)."""
    if str(gate).startswith("gate=seats") or int(need_tokens) < 0:
        return None
    terms = pool_terms(
        getattr(scheduler, "tree_cache", None),
        getattr(scheduler, "token_to_kv_pool_allocator", None),
    )
    if terms is None:
        return None
    free, evictable, inflight = terms
    verdict = intake_verdict(
        need_tokens=need_tokens, pool_tokens=pool_tokens, free_tokens=free,
        evictable_tokens=evictable, inflight=inflight,
    )
    if verdict in (INTAKE_FITS, INTAKE_WAITS):
        named = getattr(scheduler, "_h91_intake_named", None)
        if named is None:
            named = scheduler._h91_intake_named = set()
        if str(rid) not in named:
            named.add(str(rid))
            logger.warning(
                "H91 WEG2-INTAKE-ADMISSIBLE rid=%s verdict=%s need_tokens=%d "
                "free=%d evictable=%d inflight=%d pool_tokens=%d %s decline=%s -- "
                "not an intake stall: the request stays queued and is admitted "
                "in order (no 503, no flip)",
                str(rid)[:16], verdict, int(need_tokens), free, evictable,
                int(inflight), int(pool_tokens), gate,
                getattr(scheduler, "_admission_decline_note", None),
            )
    return verdict


def forget(scheduler, rid: Optional[str]) -> None:
    """An abort (prefix semantics, None = all): the rid may be named again
    when it comes back in a later phase."""
    named = getattr(scheduler, "_h91_intake_named", None)
    if not named:
        return
    if rid is None:
        named.clear()
        return
    for r in [r for r in named if str(r).startswith(str(rid))]:
        named.discard(r)


__all__ = [
    "INTAKE_FITS",
    "INTAKE_IMPOSSIBLE",
    "INTAKE_WAITS",
    "KEPT_ATTR",
    "forget",
    "intake_phase_verdict",
    "pool_terms",
    "settle_told",
    "told_admission",
]
