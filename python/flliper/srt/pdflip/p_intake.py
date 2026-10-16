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


#: NF-FOLLOWER-ROUND 1014 C2 switch (default ON); ``0`` = the told gate holds a
#: row-named rid on the follower as before (e7cfda7fbe).
TOLD_ROW_YIELD_ENV = "FLLIPER_PDFLIP_TOLD_ROW_YIELD"
ROW_NAMED = "named"
ROW_NOT_NAMED = "not_named"


def told_row_yield_enabled() -> bool:
    import os

    return (os.environ.get(TOLD_ROW_YIELD_ENV, "1") or "1").strip().lower() not in (
        "0", "false", "no", "off")


def told_row_verdict(scheduler, req) -> Optional[str]:
    """C2 (NF-FOLLOWER-ROUND 1014, inventory B4): does the told gate of THIS
    follower stand under PP0's executed row?

    ``None``: no row in hand (PP0, a non-PP rank, a follower without the row
    authority, the switch off) -- the gate decides as before. Otherwise the row
    IS the membership verdict, exactly as for the LB-VERDICT gate since
    31aa655fff (``pp_loadback_verdict.gated`` yields to ``ROW_ATTR``):

    * ``ROW_NAMED`` -- the row names the rid: the gate yields, the rid goes to
      the row's admission arm with PP0's told; a read this rank still lacks is
      materialised by ``execute_scheduled_prefix`` on its bounded clock, which
      stops the group BY NAME when it cannot (#968 station) -- never a skip
      that leaves the follower's batch one rid short of PP0's (PPScheduleRefused
      / #1004, the rank-local veto over a row that raenge-nie-uneins forbids).
    * ``ROW_NOT_NAMED`` -- the row does not name it: the row's own arm skips it
      (``pp_not_named``); a held rid is NO barrier here (W27-HELD-STOP guards a
      follower that plans its own membership, which a row-executing one does
      not) -- breaking on it would cut the row's rids behind it."""
    if not told_row_yield_enabled():
        return None
    ps = getattr(scheduler, "ps", None)
    if ps is None or int(getattr(ps, "pp_size", 1) or 1) <= 1:
        return None
    if int(getattr(ps, "pp_rank", 0) or 0) == 0:
        return None
    incoming = getattr(scheduler, "_pp_admission_incoming_effective", None)
    if incoming is None:
        return None
    rid = getattr(req, "rid", None)
    return ROW_NAMED if rid in incoming else ROW_NOT_NAMED


def note_told_row_yield(scheduler, req, verdict: str) -> None:
    """The C2 instrument: one bounded line per yield (``TOLD-ROW-YIELD``)."""
    n = getattr(scheduler, "_pdflip_told_row_yield_n", 0) + 1
    scheduler._pdflip_told_row_yield_n = n
    if n <= _LOG_FIRST or n % _LOG_EVERY == 0:
        rid = str(getattr(req, "rid", ""))
        told_map = getattr(scheduler, "_pdflip_store_told", None) or {}
        logger.info(
            "TOLD-ROW-YIELD rank pp=%s rid=%s row=%s told_here=%s kept=%s (n=%d): the "
            "told gate held this rid on this follower; PP0's executed row is the "
            "verdict (C2, like LBV_ROW_YIELD) -- the row decides, the station "
            "materialises or stops by name",
            getattr(getattr(scheduler, "ps", None), "pp_rank", "?"), rid, verdict,
            told_map.get(rid), rid in (getattr(scheduler, KEPT_ATTR, None) or {}), n,
        )


GATE_ADMIT = "admit"
GATE_SKIP = "skip"
GATE_BREAK = "break"


def told_pending_gate(
    scheduler,
    req,
    note_skip: Callable[[str, Any], None],
    held_barrier: Callable[[Any, Any], bool],
) -> str:
    """The #1400 told gate's answer when :func:`told_admission` returned None
    (no verdict this visit): ``GATE_ADMIT``, ``GATE_SKIP`` or ``GATE_BREAK``.

    ``held_barrier`` is ``pdflip_store_told.held_barrier`` (passed in: this module
    stays light). Without a row in hand (:func:`told_row_verdict` None) this is
    the gate of e7cfda7fbe exactly: the W27 barrier ends the loop, else the rid
    is skipped. With PP0's executed row in hand (C2) a rid the row NAMES is
    admitted -- the row's told and the materialisation station govern it, the
    credit stays None as for any uncredited rid -- and one it does not name is
    skipped and is no barrier."""
    row = told_row_verdict(scheduler, req)
    if row == ROW_NAMED:
        note_told_row_yield(scheduler, req, row)
        return GATE_ADMIT
    if row == ROW_NOT_NAMED:
        return GATE_SKIP
    if held_barrier(scheduler, req):
        # W27-HELD-STOP (NF h16g 17:49:53Z): a told rid held for a tree short
        # of PP0's told is a BARRIER -- overtaking it planned [pdflip-0-9,
        # pdflip-1-11] against PP0's three.
        note_skip("w27_held_barrier", getattr(req, "rid", None))
        return GATE_BREAK
    return GATE_SKIP


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


def _st_stop_types() -> tuple:
    """The named stops ``_reach_kept`` must let through (its catch-all keeps a
    failing PROBE from breaking the admission, never a stop)."""
    try:
        from flliper.srt.managers.pdflip_store_told import PdFlipStoreToldMismatch

        return (PdFlipStoreToldMismatch,)
    except Exception:  # noqa: BLE001 - a double without the module
        return ()


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
    except _st_stop_types():
        raise  # W27-HELD-STOP: the named group stop is no probe failure
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


#: K9b: an open store backup of the tree is in flight for the intake verdict (default ON, ``0`` = off).
ENV_INTAKE_BACKUP_INFLIGHT = "FLLIPER_PDFLIP_INTAKE_BACKUP_INFLIGHT"


def intake_backup_inflight_enabled(env=None) -> bool:
    import os as _os

    e = _os.environ if env is None else env
    return (e.get(ENV_INTAKE_BACKUP_INFLIGHT, "") or "").strip().lower() not in ("0", "false", "no", "off")


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
    if not inflight and intake_backup_inflight_enabled():
        # K9b INTAKE-BACKUP-INFLIGHT (NF h16r 15.10. 11:47:19Z, 7 double flips in 45 min): the stall's premise is
        # "nothing runs, nothing frees". At the refusal P's own store backups of the hand-off it had JUST finished
        # were still open (``hicache_backup(3)`` on PP0, PDFLIP-INTAKE-STALL-CENSUS tracked_evictable=157632 rows of
        # pdflip-74-212 against pool_avail=104512): their acks are what makes those rows payable again. While a
        # backup of this tree is open the refusal is a WAIT, not the stall -- bounded by the backups themselves;
        # once they are done the ordinary verdict (and its 1 s hold) decides. ``0`` = the old term.
        inflight = bool(getattr(tree, "ongoing_backup", None))
    # PR (pdflip/pp_room_vote.py; mem_cache.common.PP_ROOM_CAP_ATTR, read by its
    # literal name -- this module stays light): the room every PP stage can
    # pay. A P that runs empty with a head no stage can hold is then the stall.
    room_cap = getattr(tree, "pdflip_pp_room_cap", None)
    if isinstance(room_cap, int) and not isinstance(room_cap, bool):
        total = min(free + evictable, max(0, room_cap))
        free = min(free, total)
        evictable = total - free
    return free, evictable, inflight


def stall_cause(scheduler, need_tokens: int, pool_tokens: int) -> str:
    """NF design defect 1 (15.10.2026): the NAMED reason of a stall that survived the in-flight premise (nothing runs on this
    stage, the hold is over). The design says a request that fits the pool always fits -- finished rows are evictable -- so
    the stall at idle is a deviation, and which term stands for it decides the fix:

    * ``estimator``: free + reported evictable cover the request and nothing is locked; the payability estimators (ED
      deliverable, PR room cap, the measured unpayable wall) are what refuse it. THE deviation: rows that exist are not counted.
    * ``locked``: only the locked (protected) rows would cover it, with nothing in flight: a lock holder that is not running.
    * ``pool``: free + evictable + protected cannot cover it at all.
    * ``unread``: the terms could not be read (a stand-in); never raises.

    Format ``cause=<c> free=<n> evict=<n> prot=<n> deliv=<n|-> cap=<n|-> bk=<open backups> flight=0``: compact, because the
    front keeps the first 300 characters of the refusal in its own log."""
    tree = getattr(scheduler, "tree_cache", None)
    alloc = getattr(scheduler, "token_to_kv_pool_allocator", None)
    try:
        free = int(alloc.available_size())
        evict = int(tree.evictable_size())
    except Exception:  # noqa: BLE001 - diagnostics never raise
        return "cause=unread"

    def _int_or_none(fn):
        try:
            v = fn()
            return None if isinstance(v, bool) or not isinstance(v, int) else int(v)
        except Exception:  # noqa: BLE001
            return None

    prot = _int_or_none(getattr(tree, "protected_size", lambda: None)) or 0
    # looked up on the type, as ``mem_cache.common.deliverable_evictable_or`` does: a duck-typed stand-in answers every getattr
    dfn = getattr(type(tree), "deliverable_evictable_size", None)
    deliv = _int_or_none(lambda: dfn(tree)) if callable(dfn) else None
    cap = getattr(tree, "pdflip_pp_room_cap", None)
    cap = int(cap) if isinstance(cap, int) and not isinstance(cap, bool) else None
    try:
        backups = len(getattr(tree, "ongoing_backup", None) or ())
    except Exception:  # noqa: BLE001
        backups = 0
    need = int(need_tokens)
    if free + evict >= need:
        cause = "estimator"
    elif free + evict + prot >= need:
        cause = "locked"
    else:
        cause = "pool"
    return (f"cause={cause} free={free} evict={evict} prot={prot} "
            f"deliv={'-' if deliv is None else deliv} cap={'-' if cap is None else cap} bk={backups} flight=0")


def note_stall_deferred(scheduler, req, busy: str, note: str = "") -> None:
    """The in-flight premise held a refusal back (``PDFLIP INTAKE-STALL-DEFERRED``): the metal count of the defect. Bounded
    (the first eight, then every power of two); never raises."""
    try:
        n = getattr(scheduler, "_pdflip_intake_deferred_n", 0) + 1
        scheduler._pdflip_intake_deferred_n = n
        # review A2: remember WHICH request was held back, so its admission can be named (``note_intake_resumed``)
        rids = getattr(scheduler, "_pdflip_intake_deferred_rids", None)
        if rids is None:
            rids = scheduler._pdflip_intake_deferred_rids = {}
        rid = str(getattr(req, "rid", ""))
        rids[rid] = rids.get(rid, 0) + 1
        if n <= _LOG_FIRST or (n & (n - 1)) == 0:
            logger.info(
                "PDFLIP INTAKE-STALL-DEFERRED rid=%s in_flight=%s %s (n=%d): the refusal is a WAIT -- this stage still runs "
                "(a launched PP pass whose result is not consumed, a chunk, anchor tails); the hold starts at the first "
                "idle pass, and the finishing request's rows are counted by then",
                str(getattr(req, "rid", "")), busy, note or "gate=adder_no_token", n,
            )
    except Exception:  # noqa: BLE001 - an instrument never blocks the admission
        pass


def note_intake_resumed(scheduler, req, adder=None) -> None:
    """Review A2: a request that was held back (``PDFLIP INTAKE-STALL-DEFERRED``) is admitted now: one line naming what PP0 sized
    the admission on, so the metal tells a fresh follower fact from an aged one (F2), and a follower OOM / W27 after an
    admission out of a wait is attributable.

    ``PDFLIP INTAKE-RESUMED rid= deferred_n= cap_src=fact(rank,executed,processed)|aged|own_only|none cap= rem_total=``.
    Bounded like the deferral line (the first eight, then every power of two); the mark of the rid is consumed; reads only,
    never raises."""
    try:
        rids = getattr(scheduler, "_pdflip_intake_deferred_rids", None)
        if not rids:
            return
        rid = str(getattr(req, "rid", ""))
        deferred_n = rids.pop(rid, 0)
        if not deferred_n:
            return
        tree = getattr(scheduler, "tree_cache", None)
        cap = getattr(tree, "pdflip_pp_room_cap", None)
        cap = int(cap) if isinstance(cap, int) and not isinstance(cap, bool) else None
        book = getattr(scheduler, "_pdflip_pr_book", None)
        src = "none"
        if cap is not None:
            src = book.cap_source() if book is not None else "own_only"
        elif book is not None and book.facts:
            src = "aged"
        try:
            rem_total = int(adder.rem_total_tokens) if adder is not None else -1
        except Exception:  # noqa: BLE001
            rem_total = -1
        n = getattr(scheduler, "_pdflip_intake_resumed_n", 0) + 1
        scheduler._pdflip_intake_resumed_n = n
        if n <= _LOG_FIRST or (n & (n - 1)) == 0:
            logger.info(
                "PDFLIP INTAKE-RESUMED rid=%s deferred_n=%d cap_src=%s cap=%s rem_total=%d (n=%d)",
                rid, int(deferred_n), src, "-" if cap is None else cap, rem_total, n,
            )
    except Exception:  # noqa: BLE001 - an instrument never blocks the admission
        pass


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
    deferred = getattr(scheduler, "_pdflip_intake_deferred_rids", None)
    if deferred:
        if rid is None:
            deferred.clear()
        else:
            for r in [r for r in deferred if str(r).startswith(str(rid))]:
                deferred.pop(r, None)
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
    "note_intake_resumed",
    "note_stall_deferred",
    "pool_terms",
    "settle_told",
    "stall_cause",
    "told_admission",
]
