"""D-NORECOMPUTE (user law 02.10.: after a flip D computes nothing -- E2,
HANDBACK d_compute=0).

A request the #1471 settle releases after the wake with a SHORT store read
used to go to D's extend, which computed ``[delivered, N)`` on D:

* y6z pdflip-12-64 (08:24:58): the wake read delivered 45568 of 51520 (the sleep
  flush had lost 13 anchors, see unified_radix_cache ``_pdflip_flush_spill_room``),
  the settle lapsed, D computed 5952 tokens in 3 chunks (08:25:50-08:26:02),
  every chunk 1.2-1.8 s in pool.host_fetch;
* y6z pdflip-16-79 (08:27:29): delivered 54144 of 54336, '#x38 SETTLE-TAIL'
  released the 192-token tail to the extend: 201 tokens took 1.56 s and held
  five woken seats from decoding.

Now the release asks :func:`d_would_compute`: 0 when the read is whole or an
agreed F4 PARK-END / E2 tail covers it (``delivered >= page_prefix``), else the
tokens D would compute. Such a request goes through RESUME-VIA-P instead
(``resume_via_p.keep_on_d`` reason ``store_short_after_flip``): P prefills its
context from the shared store, D resumes it after the flip back under E2. A
request RESUME-VIA-P cannot take (non-stream, multimodal, attempts spent) is
released as before and NAMED (``D-NORECOMPUTE-FALLBACK``), never silently.

X-GESETZ F-D (user 16.10., law 4): the law covers a HAND-OVER after a P prefill
only. A D-direct leg 2 whose prefill never ran (the front priced it SHORT and sent
it to D with its decision X, ``req.weg2_front_x``; no output yet) and that a flip
parked (``PARK-HANDBACK kept=committed`` -- "D prefills it after the flip") is
D's to extend when its remainder is within that X: :func:`d_direct_extends`.
Diverting it made a D->P flip of its own for a SHORT (NF h16s pdflip-6-55, 375
tokens: W50-REROUTE store_short_after_flip -> PARK-IMMEDIATE over-x p_only ->
six decodes parked). D's X gate still bounds the realised extent (W31 by name).

Switch ``FLLIPER_PDFLIP_D_NORECOMPUTE`` (default on; 0 = the release as before).
Rank-uniform: the release list is the group's MIN verdict, ``delivered`` the
synced prefix, the agreed tail the group's answer.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

ENV = "FLLIPER_PDFLIP_D_NORECOMPUTE"
MARK = "PDFLIP D-NORECOMPUTE"
REASON = "store_short_after_flip"


def enabled(env=None) -> bool:
    e = os.environ if env is None else env
    raw = (e.get(ENV, "") or "").strip().lower()
    return raw not in ("0", "false", "no", "off")


def span_base(req: Any, record: Any) -> int:
    """SHORT-FLIP-REPRICE-1024 (SOLL R2/R7/R11; 27B rc41 boot ..._1024_110644, D 11:40:21-22, pdflip-21-215): the
    absolute prompt depth at which ``record``'s store read STARTED. A ``PrefetchOutcome`` counts span-relative --
    ``deliverable`` is ``len(prefetch_key)`` and ``materialized = matched + loaded`` is measured from the span start
    (``unified_radix_cache``: ``loaded = min_completed - insert_result.prefix_len``, the insert rooted at the span's
    host node). Every reader of the store-short witness (``_pdflip_store_delivered``: ``d_would_compute``, the settle
    remainder, the tail-adopt window, the capacity park, the P-span label) reads an ABSOLUTE depth. 215 kept 53750
    tokens on D's device across the flip, read [53750, 70994), got 17091 of 17244, and the witness 17091 was taken as
    the depth -- ``would_compute=53908`` instead of 158, over the front X: D-NORECOMPUTE wrote a needs-p, the front
    classed a kurz request gross and P prefilled 158 tokens.

    Review MUSS-1: the start is the RECORD's own (``record.span_start``, copied from the read's operation, stamped once
    at registration), never the request's ``_prefetch_registered_prefix_len`` -- every later ``_prefetch_kvcache`` call
    (deferral retry, #1456 HOLD-REFETCH, a declined re-issue, regroup) rewrites that stamp while the old record stands
    (215 itself: D 11:41:02 ``#1456 HOLD-REFETCH ... declined:vote_negative``), and a rank-local re-match makes it
    differ per rank (R11). ``req`` is kept for the defensive bound only.

    0 for a Form A END-vote read (``synced_end`` is absolute already), a bare ``int`` record, a record without a start,
    or a start that cannot be this read's (start + deliverable beyond the request's ids): the old reading."""
    if record is None or getattr(record, "synced_end", None) is not None or not hasattr(record, "materialized"):
        return 0
    try:
        base = int(getattr(record, "span_start", None) or 0)
    except (TypeError, ValueError):
        return 0
    if base <= 0:
        return 0
    ids = getattr(req, "full_untruncated_fill_ids", None)
    try:
        total = len(ids) if ids is not None else None
        deliverable = int(getattr(record, "deliverable", 0) or 0)
    except (TypeError, ValueError):
        return 0
    if total is not None and base + deliverable > total:
        return 0   # not this read's start: never invent a depth the request does not have
    return base


def store_read_depth(req: Any, record: Any) -> Optional[int]:
    """SHORT-FLIP-REPRICE-1024: the ABSOLUTE prompt depth ``record``'s store read left resident -- the END-vote's
    reduced ``synced_end`` when present, else :func:`span_base` + ``materialized``. None = no annotated record."""
    if record is None:
        return None
    se = getattr(record, "synced_end", None)
    if se is not None:
        return int(se)
    if not hasattr(record, "materialized"):
        return None
    return span_base(req, record) + int(getattr(record, "materialized", 0) or 0)


def refresh_delivered(req: Any, record: Any) -> bool:
    """TAIL-CONSUME-ON-FULL companion (h48 pdflip-182-1661, D 00:49:44-45): the
    witness ``_pdflip_store_delivered`` is written by the STORE-SHORT arm only
    for an INCOMPLETE record -- the first, short wake read left 51840 there,
    the #1456 HOLD-REFETCH then landed every page (``completed_synced=90688``)
    and nothing updated it: D-NORECOMPUTE priced 90701 - 51840 = 38861 tokens
    (``would_compute=38861``) instead of the 13-token tail an agreed E2 window
    adopts, and W50-REROUTE sent the request to P a second time. Before the
    release asks :func:`d_would_compute`, the witness follows the record's
    same quantity the STORE-SHORT arm writes (``scheduler`` #1324 arm: the
    reduced END ``synced_end`` on a Form A END-vote read -- one value on every
    rank --, else ``materialized``), and only for a terminated store read whose
    record is no longer incomplete; never downwards. True = raised."""
    old = getattr(req, "_pdflip_store_delivered", None)
    if old is None or record is None:
        return False
    if int(getattr(record, "deliverable", 0) or 0) <= 0 or getattr(record, "is_incomplete", True):
        return False
    # SHORT-FLIP-REPRICE-1024: the record's ABSOLUTE depth (END vote, else span start + materialized)
    now = int(store_read_depth(req, record) or 0)
    if now <= int(old):
        return False
    req._pdflip_store_delivered = now
    logger.info("%s WITNESS rid=%s delivered %d -> %d (the record's group-agreed completion after a re-read; "
                "the short read's witness was stale)", MARK, str(getattr(req, "rid", "?")), int(old), now)
    return True


def d_would_compute(req: Any) -> int:
    """Tokens D's extend would compute for ``req`` after its short read
    (0 = whole read, no short-read stamp, or an agreed tail that adopts)."""
    delivered = getattr(req, "_pdflip_store_delivered", None)
    ids = getattr(req, "full_untruncated_fill_ids", None)
    if delivered is None or ids is None:
        return 0
    rem = max(0, len(ids) - int(delivered))
    if rem <= 0:
        return 0
    try:
        from flliper.srt.pdflip import tail_adopt as _ta

        entry = _ta._AGREED.get(str(getattr(req, "rid", "")))
        if entry is not None and entry.agreed and int(delivered) >= int(entry.staged.spec.page_prefix):
            return 0  # the F4 / E2 tail adopts [page_prefix, N): no target forward
    except Exception:  # noqa: BLE001 -- no tail module: the remainder stands
        pass
    return rem


def d_direct_extends(sched: Any, req: Any, rem: int) -> bool:
    """X-GESETZ F-D (16.10.): True = ``req`` is a D-direct leg whose prefill never
    ran and ``rem`` is within the front's decision X (+ one page, the same limit
    D's X gate applies, ``front_x.resolve_limit``) -- D extends it itself (law 4),
    it is NOT diverted to P. False for a hand-over after a P prefill (no front X
    on the leg), for a resumed request (output already produced: the static X and
    RESUME-VIA-P own it, front_x rule 3) and for a remainder over that X.
    Group-uniform: ``rem`` comes from the synced witness, the front X is a field
    of the broadcast request, ``page_size`` and the static X are launch args."""
    try:
        from flliper.srt.pdflip import front_x as _fx

        fx, _priced = _fx.request_front_x(req)
        if fx is None and int(rem) > 0:
            return _resumed_tail_extends(sched, req, int(rem), _fx)
        if fx is None or int(rem) <= 0:
            return False
        static_x = int(getattr(getattr(sched, "server_args", None), "tp_prefill_max_tokens", 0) or 0)
        limit = int(_fx.resolve_limit(req, static_x, getattr(sched, "page_size", None))[0])
    except Exception:  # noqa: BLE001 -- unknown = the law as before (divert)
        return False
    if limit <= 0 or int(rem) > limit:
        return False
    logger.info("%s D-DIRECT-EXTEND rid=%s would_compute=%d front_x=%d limit=%d delivered=%s -- a D-direct "
                "leg whose prefill never ran: D extends it under X (law 4, X-GESETZ 16.10.), not diverted "
                "to P", MARK, str(getattr(req, "rid", "?")), int(rem), int(fx), limit,
                getattr(req, "_pdflip_store_delivered", None))
    return True


def _resumed_tail_extends(sched: Any, req: Any, rem: int, _fx: Any) -> bool:
    """H16V F3 / review A3 (SCHEDULER-SOLL R2 + R7 + R11; NF h16u2 17.10.): a request that already decoded and D
    parked for a flip comes back with a SHORT store read: its tail ``rem`` is D's own work when it is kurz -- under the
    halbX THE FRONT carried with the D leg (``weg2_front_halfx``, F_halb x the X of the front's decision at the depth D
    continues from, computed with the front's flag), and only then. NO front X or halbX carried (the request never came
    through a front leg that stamped them, the carry switch is off, ...): the old way, RESUME-VIA-P -- there is NO fixed
    substitute bound (SOLL section 2: "X nie ein fester Wert"; the former 0.5 x static X = 6144 lay above the curve X)
    and no rank-local flag is read (R11). Five of the six h16u2 flips under X were exactly this: W50-REROUTE
    ``store_short_after_flip`` with d_extent 77 / 693 / 1563 / 1926 / 2054 -> p_only -> a whole D->P flip and six
    parked decodes for a few hundred tokens ("Kann D einen Request unter X nicht nehmen, ist das ein Defekt, kein
    Flipgrund", R7). A tail of halbX or more stays on RESUME-VIA-P as before. Group-uniform: ``rem`` is the synced
    witness, ``weg2_front_x`` / ``weg2_front_halfx`` fields of the broadcast request. The static X still bounds it."""
    try:
        static_x = int(getattr(getattr(sched, "server_args", None), "tp_prefill_max_tokens", 0) or 0)
        if static_x > 0 and rem > static_x:
            return False
        x = getattr(req, _fx.ATTR_X, None)
        halfx = getattr(req, _fx.ATTR_HALFX, None)
        if x is None or int(x) <= 0 or halfx is None or int(halfx) <= 0:
            return False   # nothing carried by the front: RESUME-VIA-P, no invented bound
        if not rem < int(halfx):
            return False
    except Exception:  # noqa: BLE001 -- unknown = the law as before (divert)
        return False
    logger.info("%s D-RESUME-EXTEND rid=%s would_compute=%d front_x=%d front_halfx=%d delivered=%s -- a resumed "
                "request whose tail is under the front's halbX: D extends it itself (SOLL R2/R7), no flip for it",
                MARK, str(getattr(req, "rid", "?")), int(rem), int(x), int(halfx),
                getattr(req, "_pdflip_store_delivered", None))
    return True


def divert(sched: Any, req: Any, rem: int) -> bool:
    """RESUME-VIA-P for a released short read. True = kept parked for P (the
    caller does not queue it); False = not eligible (released, named)."""
    from flliper.srt.pdflip import resume_via_p as _rvp

    x = int(getattr(getattr(sched, "server_args", None), "tp_prefill_max_tokens", 0) or 0)
    if not _rvp.eligible(req, sched=sched):
        logger.warning("%s-FALLBACK rid=%s would_compute=%d (RESUME-VIA-P not eligible: non-stream, "
                       "multimodal or attempts spent) -- released to D's extend, named",
                       MARK, str(getattr(req, "rid", "?")), int(rem))
        return False
    _rvp.keep_on_d(sched, req, int(rem), x, reason=REASON)
    try:
        from flliper.srt.pdflip import d_park_runtime as _dpr

        # an awake re-queue gives it back to the settle, never to the X gate
        setattr(req, _dpr.FROM_SETTLE_ATTR, True)
    except Exception:  # noqa: BLE001
        pass
    logger.info("%s rid=%s would_compute=%d delivered=%s -> RESUME-VIA-P (P prefills from the store, "
                "D resumes under E2 after the flip back; no D compute after a flip)",
                MARK, str(getattr(req, "rid", "?")), int(rem),
                getattr(req, "_pdflip_store_delivered", None))
    return True
