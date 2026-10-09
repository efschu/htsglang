"""H91a (25.09.2026): group P admits its queued backlog in order; the intake
stall is left to the request this phase cannot admit at all.

THE SPECIMEN. cu130 acceptance dkrnfbar1agent09252021 (1a4e6ff19a), P log
20:40:49-20:42:10: rid 53572be2 (42,442 tokens) sat in P's queue for 61 s
with NOTHING running, ``pool_avail=220480`` and the finished neighbour's
41,664 rows ``tracked_evictable`` (0 locked) -- the pool never bound. The
admission-wedge detector then answered it 503 PDFLIP-INTAKE-STALL
(``gate=admission-wedge``; ``rem_total_tokens=-1`` only means "no adder on
that path") and the front paid a flip for it.

WHAT BLOCKED. #1400's told verdict (``pdflip_store_told.admission``) is taken
ONCE: it pops ``told_map[rid]`` (and the loaded credit) on the first
admission visit. That visit ran at 20:40:58 (PHASE-PURITY STORE WITNESS n=30,
the line emitted right after the told gate returns) in the pass that carried
d762c570's chunked continuation; the adder had no room for a second request
in that pass, so 53572be2 stayed queued. Every later visit found no told
(``pp0_publish`` had already dropped the rid from its held set, so nothing
re-publishes) and skipped it as ``pdflip_store_told_pending`` -- the witness
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

from flliper.srt.pdflip.intake_stall import (
    INTAKE_FITS,
    INTAKE_IMPOSSIBLE,
    INTAKE_WAITS,
    intake_verdict,
)

logger = logging.getLogger(__name__)

#: scheduler attribute holding the kept verdicts ({rid: _Kept})
KEPT_ATTR = "_pdflip_told_kept"
SKIP_TOLD_PENDING = "pdflip_store_told_pending"  # = pdflip_store_told.SKIP_TOLD_PENDING
_LOG_FIRST = 8
_LOG_EVERY = 256


@dataclass
class _Kept:
    req: Any
    told: int
    credit: int
    #: W27-UNIFORM: the told was an ABSOLUTE twin told on this follower, whose
    #: live tree must reach it at the later admission as well
    reach: bool = False


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

    ``admission`` is ``pdflip_store_told.admission``. Returns what it returns:
    ``None`` = skip this pass (no told yet), else the loaded credit."""
    kept = _kept(scheduler)
    rid = str(getattr(req, "rid", ""))
    told_map = getattr(scheduler, "_pdflip_store_told", None) or {}
    entry = kept.get(rid)
    if entry is not None:
        if entry.req is req and rid not in told_map:
            try:
                # TK: the same RAW-token cap admission planted (told counts
                # keys; bigram trees need +1 -- prefix_cap_tokens)
                from flliper.srt.managers.pdflip_store_told import prefix_cap_tokens

                req._pdflip_prefix_cap = prefix_cap_tokens(
                    getattr(scheduler, "tree_cache", None), int(entry.told)
                )
            except Exception:  # noqa: BLE001 - a double without the field
                pass
            if entry.reach and _reach_kept(scheduler, req, entry):
                # item 180: the tree is still short of told -- not seated on
                # this rank's own say; PP0 is re-asked, the request is held
                note_skip(SKIP_TOLD_PENDING, rid)
                return None
            n = getattr(scheduler, "_h91_told_kept_n", 0) + 1
            scheduler._h91_told_kept_n = n
            if n <= _LOG_FIRST or n % _LOG_EVERY == 0:
                logger.info(
                    "H91 STORE-TOLD KEPT rid=%s told=%d credit=%d (n=%d): the "
                    "verdict of an earlier visit that did not admit stands "
                    "(it used to be consumed there and the rid skipped as "
                    "pdflip_store_told_pending for ever: cu130 53572be2)",
                    rid, entry.told, entry.credit, n,
                )
            return entry.credit
        # another request object under this rid, or a fresh told arrived for
        # it (re-intake): the old verdict is not this visit's. The wait budget
        # and the held mark of that verdict go with it (item 180): settle_told
        # never sees the rid again when no fresh told follows.
        _drop_reach_state(scheduler, rid)
        kept.pop(rid, None)
    told = told_map.get(rid)
    # W27-UNIFORM: admission consumes the twin-follower mark; ask before it
    reach = _is_follower_twin(scheduler, rid)
    credit = admission(scheduler, req, note_skip)
    if credit is not None and told is not None:
        kept[rid] = _Kept(req=req, told=int(told), credit=int(credit), reach=reach)
    return credit


def _is_follower_twin(scheduler, rid: str) -> bool:
    try:
        from flliper.srt.managers import pdflip_store_told as _st
        from flliper.srt.pdflip import p_twin_defer as _twin

        if _st.is_pp0(scheduler):
            return False
        st = getattr(scheduler, _twin._ATTR, None)
        return bool(st) and str(rid) in (getattr(st, "twin_follower", None) or {})
    except Exception:  # noqa: BLE001 - a double without the form: no check
        return False


def _drop_reach_state(scheduler, rid: str) -> None:
    """The follower's per-rid W27 state (re-read wait budget, held mark)."""
    for attr in ("_w27u_wait_spent_s", "_w27u_unreached"):
        m = getattr(scheduler, attr, None)
        if m:
            m.pop(rid, None)
    named = getattr(scheduler, "_w27u_spent_named", None)
    if named:
        named.discard(rid)


def _reach_kept(scheduler, req, entry: "_Kept") -> bool:
    """W27-UNIFORM (open point (a) of item 025): the verdict kept across visits
    was checked against the follower's live tree when it was made; between that
    visit and the one that finally seats the request the tree can move (the
    sibling's read released, eviction), and only the TOLD-PIN held the anchor.
    The same ``follower_reach_told`` check runs at the later admission: a tree
    that fell short is re-read to told (wait bounded per rid, WAIT_CAP_S), PP0's
    prefix stands, nothing is refused here.

    Returns True when the tree is STILL short after the bounded re-read
    (item 180): the caller holds the request instead of returning the kept
    credit (PP0 re-asked through ``pdflip_store_told.follower_hold_unreached``)."""
    try:
        from flliper.srt.managers import pdflip_store_told as _st

        before = getattr(scheduler, "_w27u_reread_n", 0)
        _st.follower_reach_told(scheduler, req, int(entry.told), int(entry.told), site="kept")
        if getattr(scheduler, "_w27u_reread_n", 0) != before:
            # the re-read registered a fresh credit (and kept pin) of its own
            tree = getattr(scheduler, "tree_cache", None)
            fresh = int(_st._pop_credit_keep_pin(tree, str(getattr(req, "rid", ""))) or 0)
            if fresh > 0:
                entry.credit = fresh
        return bool(_st.follower_hold_unreached(scheduler, req, "kept"))
    except Exception:  # noqa: BLE001 - a probe never breaks the admission
        logger.warning("W27-UNIFORM kept-verdict reach check skipped for rid=%s",
                       str(getattr(req, "rid", "")), exc_info=True)
        return False


def told_pending(scheduler, req) -> bool:
    """#791T: would :func:`told_admission` SKIP ``req`` on this rank right now
    because PP0's told has not arrived here yet? -- the same condition, read
    without side effects (no skip census, no verdict consumed).

    True only on a told-ARMED FOLLOWER (PP0 decides its own verdict) for a
    queued request with no told in ``_pdflip_store_told`` and no kept verdict of
    an earlier visit (H91 STORE-TOLD KEPT) for this very request object. The
    #631 row probe reads it: a forwarded frame that admits such a rid has
    overtaken its ``PdFlipStoreTold`` on the request chain (rc12z20 13:03:27,
    PP2 pdflip-0-8: r24 request located, r25 told still in flight), so the
    frame waits for the hop instead of being planned into a #791 skip.
    Any failure answers False (= today's behaviour: no defer)."""
    try:
        from flliper.srt.managers import pdflip_store_told as _st

        if not _st.armed(scheduler) or _st.is_pp0(scheduler):
            return False
        rid = str(getattr(req, "rid", ""))
        if not rid:
            return False
        if rid in (getattr(scheduler, "_pdflip_store_told", None) or {}):
            return False
        entry = (getattr(scheduler, KEPT_ATTR, None) or {}).get(rid)
        if entry is not None and entry.req is req:
            return False
        return True
    except Exception:  # noqa: BLE001 - advisory: never turn a probe into a crash
        return False


def settle_told(scheduler, waiting_queue: Iterable[Any]) -> int:
    """After a committed pass: drop the kept verdicts of requests that are no
    longer queued (admitted into the batch, or aborted). Returns the count."""
    kept = getattr(scheduler, KEPT_ATTR, None)
    if not kept:
        return 0
    queued = {id(r) for r in waiting_queue}
    gone = [rid for rid, e in kept.items() if id(e.req) not in queued]
    tree = getattr(scheduler, "tree_cache", None)
    unpin = getattr(tree, "_unpin_prefetched_span", None)
    for rid in gone:
        kept.pop(rid, None)
        _drop_reach_state(scheduler, rid)
        if callable(unpin):
            # TOLD-PIN: the admission kept the #1417 pin (anchor included)
            # while the verdict stood; the request left the queue, the pin goes
            try:
                unpin(rid)
            except Exception:  # noqa: BLE001 - bookkeeping, never a gate
                pass
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
    # PR (pdflip/pp_room_vote.py; mem_cache.common.PP_ROOM_CAP_ATTR, read by its
    # literal name -- this module stays light): the room every PP stage can
    # pay. A P that runs empty with a head no stage can hold is then the stall.
    room_cap = getattr(tree, "pdflip_pp_room_cap", None)
    if isinstance(room_cap, int) and not isinstance(room_cap, bool):
        total = min(free + evictable, max(0, room_cap))
        free = min(free, total)
        evictable = total - free
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
                "H91 PDFLIP-INTAKE-ADMISSIBLE rid=%s verdict=%s need_tokens=%d "
                "free=%d evictable=%d inflight=%d pool_tokens=%d %s decline=%s -- "
                "not an intake stall: the request stays queued and is admitted "
                "in order (no 503, no flip)",
                str(rid), verdict, int(need_tokens), free, evictable,
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
