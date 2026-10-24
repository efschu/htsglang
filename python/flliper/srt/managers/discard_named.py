"""#221 Fix A -- "benannt verwerfen": a prefetch mark (or the read an owed re-issue
rides on) that is discarded is discarded BY NAME, never silently.

THE DEFECT CLASS (analysis qwen-221-analyse-1024, boot 1ecda02a58, rid
pdflip-12-170): three places let a #1068 A12.2 deferral mark, or the read that
guarded it, disappear without a line and without a census exit, so the
#915 identity ``deferred == landed + exits`` carried 5 marks without an exit in
the 24.10. boots (0803: 2, 1106: 2, duo0126: 1):

1. ``reset_join``         ``HiCacheController.reset()`` -- the ``#1068 RESET JOIN``
   line counts what the stop OBSERVED (terminated_ops / drained_ops), while
   ``UnifiedRadixCache._reset_full`` drops every ``ongoing_prefetch`` RECORD
   without a trace: a read that had finished its probe but was not yet consumed
   by ``check_prefetch_progress`` (the owed re-issue of 12-168) vanished with
   ``drained_ops=0``.
2. ``hold_refetch_clear`` ``Scheduler._pdflip_refetch_one`` / ``_pdflip_refetch_budget``
   -- the HOLD-REFETCH (#1456/#1471) cleared the mark ("the shortfall mark is ours
   to re-issue") with no census exit.
3. ``sleep_reset``        the D sleep flush (``flush_cache`` -> ``tree_cache.reset()``)
   killed the operation that guarded the marks of the held population and ran no
   clearer, unlike the cutover's ``_clear_prefetch_deferral_for_reissue``.
4. ``wake_rearm``         (27B) ``park_l3.rearm_early_reads``: the wake's fallback
   reset killed the early hold read and the re-arm set ``prefetch_deferred = None``
   with no census exit (review #221 MUSS-3; the re-armed read is issued by the
   release, so the mark is safe to close -- only its exit was missing).

THE POPULATION OF SITE 3 (review #221 MUSS-1, rule R7 "kein stiller Verlust" and
its danger direction W31/W50 "never a request without host KV into the pricing"):
a mark is closed there ONLY for a request whose re-read is independent of the
mark -- the non-settle members of ``pdflip_d_parked`` and the ``pdflip_dormant_hold``
(both re-enter through the intake: the dormant ``#248`` DEFER_ATTR wake read, or
``_prefetch_kvcache`` directly). The ``#1471`` post-wake settle and the parked
members folded from it (``FROM_SETTLE_ATTR``) are KEPT: ``_pdflip_refetch_one``
answers "complete" for an absent record when ``_1471_short`` is unset, and the
settle also releases on its bound lapse and on an exception -- their only re-read
on every exit is the standing mark (X gate pending + ``_retry_deferred_prefetches``).
They are named on one ``#221 DISCARD-KEPT`` line (no census exit: a live mark is
no loss and exits through its own lifecycle).

WHAT THIS MODULE DOES (Variant A, nothing else): every discard is named with ONE
line carrying the stable tag ``#221 DISCARD-NAMED`` and ``site= rid= marks= tokens=``,
and the census gets exactly one exit per discarded mark. NO re-issue is added
(Variant B is a separate ticket).

CENSUS KEYS (the #915 ``prefetch-gate`` dict, same line as ``deferred`` / ``landed``):

* ``defer_cleared_refetch`` (site hold_refetch_clear) and ``defer_lost_at_reset``
  (sites sleep_reset and wake_rearm) are MARK exits and extend the identity::

      deferred == landed + defer_expired + defer_released + defer_dropped
                  + defer_cleared_reissue + defer_cleared_refetch + defer_lost_at_reset

* ``discard_reset_join`` (+ ``discard_reset_join_tokens``) counts discarded READ
  UNITS (operation or tree record), not marks: the controller does not see the
  ``Req`` and must never close a mark, so it is deliberately NOT a term of the
  identity. The mark of the same rid, if one still stands, is closed by site 3.

THE DANGER DIRECTION (tested): a census exit never closes a LIVE mark. An exit is
counted only when a mark stood before the clear AND is gone after it
(``discard_mark``); a clear that did not take (a partial scheduler without the
clearer), a request that only carried a landed-hold, or a request outside the
cleared population (the queue outside a park, the settle, FROM_SETTLE) count
nothing and keep their mark.

Switch ``FLLIPER_DISCARD_NAMED`` (default ON; ``0``/``false``/``no``/``off`` =
the pre-#221 behaviour byte for byte: no line, no census, site 3 clears nothing).
"""

from __future__ import annotations

import os
from typing import Callable, Dict, Iterable, List, Optional, Tuple

TAG = "#221 DISCARD-NAMED"
TOTAL_TAG = "#221 DISCARD-NAMED-TOTAL"
#: marks a site deliberately does NOT close (their re-read is the mark itself)
KEPT_TAG = "#221 DISCARD-KEPT"
#: #221 Fix A "benannt verwerfen" (rule R7, kein stiller Verlust), DEFAULT ON: every discarded #1068
#: deferral mark and every read unit a reset drops is named on one "#221 DISCARD-NAMED site=... rid=..." line
#: with one census exit (defer_cleared_refetch / defer_lost_at_reset; read units discard_reset_join).
#: 0/false/no/off = the pre-#221 behaviour: no line, no census, site 3 clears nothing, sites 2/4 clear silently.
ENV = "FLLIPER_DISCARD_NAMED"

SITE_RESET_JOIN = "reset_join"
SITE_HOLD_REFETCH_CLEAR = "hold_refetch_clear"
SITE_SLEEP_RESET = "sleep_reset"
SITE_WAKE_REARM = "wake_rearm"
SITES = (SITE_RESET_JOIN, SITE_HOLD_REFETCH_CLEAR, SITE_SLEEP_RESET, SITE_WAKE_REARM)

#: mark exits (terms of the #915 identity), per site
EXIT_KEY = {
    SITE_HOLD_REFETCH_CLEAR: "defer_cleared_refetch",
    SITE_SLEEP_RESET: "defer_lost_at_reset",
    SITE_WAKE_REARM: "defer_lost_at_reset",
}
#: read-unit census key of site 1 (NOT an identity term, see the module doc)
UNIT_KEY = "discard_reset_join"


def enabled(env=None) -> bool:
    env = os.environ if env is None else env
    return str(env.get(ENV, "1") or "1").strip().lower() not in ("0", "false", "no", "off")


def _census(key: str, tokens: int = 0) -> None:
    from flliper.srt.mem_cache.match_refusal_census import note_prefetch_gate

    note_prefetch_gate(key, tokens)


def format_line(site: str, rid, marks: int, tokens: int, **extra) -> str:
    """The ONE per-rid line. Stable key order; ``extra`` keys follow in the
    order given, so a grep on ``site=<name> rid=`` never moves."""
    tail = "".join(" %s=%s" % (k, v) for k, v in extra.items())
    return "%s site=%s rid=%s marks=%d tokens=%d%s" % (
        TAG, site, rid, int(marks), int(tokens), tail)


def format_total(site: str, rids: List[str], marks: int, tokens: int) -> str:
    """The aggregate line of a batch site, UNCONDITIONAL (n=0 included): a zero
    from an absent line is not a zero (denominator law). A different tag from
    the per-rid line, so ``grep -c 'DISCARD-NAMED site='`` counts discards only."""
    return "%s site=%s n=%d marks=%d tokens=%d rids=%s" % (
        TOTAL_TAG, site, len(rids), int(marks), int(tokens), ",".join(rids) or "-")


def format_kept(site: str, rids: List[str], why: str) -> str:
    """The marks a site leaves standing ON PURPOSE, one line, no census move
    (a different tag: neither ``DISCARD-NAMED site=`` nor the TOTAL grep sees it)."""
    return "%s site=%s n=%d rids=%s why=%s" % (KEPT_TAG, site, len(rids), ",".join(rids) or "-", why)


# ---------------------------------------------------------------- marks (sites 2, 3, 4)


def mark_standing(req) -> bool:
    """True iff the A12.2 MARK itself stands. The landed-hold flag alone is not
    a mark: its exit (LANDED) was counted when it was set."""
    return getattr(req, "prefetch_deferred", None) is not None


def _mark_tokens(req) -> int:
    try:
        return int(getattr(req, "_prefetch_span_tokens", 0) or 0)
    except (TypeError, ValueError):
        return 0


def discard_mark(
    req,
    site: str,
    log,
    clear_fields: Optional[Callable[[object], object]],
    clear_unmarked: bool = True,
    **extra,
) -> bool:
    """Clear the A12.2 fields of ``req`` through ``clear_fields`` (the
    scheduler's ``_clear_prefetch_deferral_fields``; None = no clearer on this
    object, nothing is cleared, exactly as before) and, iff a mark stood before
    and is gone after, name it and count its exit. ``clear_unmarked=False``
    touches only a request whose MARK stands (a landed-hold alone is left as it
    is). Returns whether an exit was counted. Never raises into the caller."""
    on = enabled()
    stood = on and mark_standing(req)
    if stood:
        facts = dict(
            reason=getattr(req, "prefetch_deferred", None),
            attempts=int(getattr(req, "prefetch_defer_attempts", 0) or 0),
            passes=int(getattr(req, "prefetch_defer_passes", 0) or 0),
        )
        tokens = _mark_tokens(req)
    if clear_fields is not None and (clear_unmarked or mark_standing(req)):
        clear_fields(req)
    if not stood or mark_standing(req):
        return False  # no mark, or the clear did not take: a live mark is never closed
    try:
        _census(EXIT_KEY[site])
        log.info(format_line(site, getattr(req, "rid", "?"), 1, tokens, **facts, **extra))
    except Exception:  # noqa: BLE001 -- an instrument never breaks the caller
        pass
    return True


def discard_marks(
    population: Iterable[object],
    site: str,
    log,
    clear_fields: Optional[Callable[[object], object]],
) -> Tuple[List[str], int, int]:
    """:func:`discard_mark` over a population (each object once; only a request
    whose mark stands is touched), then ONE aggregate line. Switch off = nothing
    is touched. Returns (rids, marks, tokens)."""
    if not enabled():
        return [], 0, 0
    rids: List[str] = []
    tokens = 0
    seen = set()
    for req in population:
        if id(req) in seen:
            continue
        seen.add(id(req))
        t = _mark_tokens(req)
        if discard_mark(req, site, log, clear_fields, clear_unmarked=False):
            rids.append(str(getattr(req, "rid", "?")))
            tokens += t
    try:
        log.info(format_total(site, rids, len(rids), tokens))
    except Exception:  # noqa: BLE001
        pass
    return rids, len(rids), tokens


# ---------------------------------------------------------------- read units (site 1)


def _unit_tokens(obj) -> int:
    for attr in ("token_ids", "host_indices"):
        v = getattr(obj, attr, None)
        if v is not None:
            try:
                return int(len(v))
            except TypeError:
                continue
    return 0


def snapshot_records(ongoing_prefetch) -> Dict[object, Tuple[str, int]]:
    """The tree's ``ongoing_prefetch`` records as {unit key: (rid, tokens)},
    taken BEFORE ``_reset_full`` clears the dict. The unit key is ``id`` of the
    record's operation, so the controller's own view of the same operation is
    the same unit."""
    out: Dict[object, Tuple[str, int]] = {}
    if not enabled() or not ongoing_prefetch:
        return out
    try:
        for rid, rec in list(ongoing_prefetch.items()):
            op = getattr(rec, "operation", None)
            key = id(op) if op is not None else ("record", str(rid))
            out[key] = (str(rid), _unit_tokens(op) if op is not None else _unit_tokens(rec))
    except Exception:  # noqa: BLE001
        return {}
    return out


def snapshot_ops(controller) -> Dict[object, Tuple[str, int]]:
    """The operations the controller's join will terminate, {id(op): (rid,
    tokens)}, taken BEFORE the stop (afterwards the loops have consumed them)."""
    out: Dict[object, Tuple[str, int]] = {}
    fn = getattr(controller, "_inflight_prefetch_ops", None)
    if not enabled() or fn is None:
        return out
    try:
        for op in fn():
            out[id(op)] = (str(getattr(op, "request_id", "?")), _unit_tokens(op))
    except Exception:  # noqa: BLE001
        return {}
    return out


def name_reset_join(log, observed: Dict[object, Tuple[str, int]],
                    records: Optional[Dict[object, Tuple[str, int]]]) -> Tuple[int, int]:
    """Name every read unit the reset discards: the ones the join OBSERVED
    (``observed``) and the tree records it did not (``records`` minus
    ``observed``). One line per rid (``marks`` = distinct units, ``observed`` =
    how many of them the join saw), one census unit count each, ONE aggregate
    line. Returns (units, tokens)."""
    if not enabled():
        return 0, 0
    try:
        units: Dict[object, Tuple[str, int, bool]] = {}
        for k, (rid, tok) in (records or {}).items():
            units[k] = (rid, tok, False)
        for k, (rid, tok) in observed.items():
            units[k] = (rid, tok, True)
        by_rid: Dict[str, List[int]] = {}
        for rid, tok, seen in units.values():
            row = by_rid.setdefault(rid, [0, 0, 0])
            row[0] += 1
            row[1] += tok
            row[2] += 1 if seen else 0
            _census(UNIT_KEY, tok)
        total_tok = 0
        for rid in sorted(by_rid):
            n, tok, seen = by_rid[rid]
            total_tok += tok
            log.info(format_line(SITE_RESET_JOIN, rid, n, tok, observed=seen))
        log.info(format_total(SITE_RESET_JOIN, sorted(by_rid), len(units), total_tok))
        return len(units), total_tok
    except Exception:  # noqa: BLE001 -- an instrument never breaks a reset
        return 0, 0
