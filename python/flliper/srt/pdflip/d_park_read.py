"""PARK-RETAIN READ (27.09., metal dkr27bparkdraftbar1w209270645 + NF dauer 0859):
a flip-parked D request reads back EXACTLY what its park retained -- no more,
and not less than one page.

THE ROOT, measured on both models:

* WHAT THE PARK RETAINS. ``park_running`` retracts with ``retain=True``;
  ``UnifiedRadixCache.cache_finished_req`` asks every component for its
  retention length and inserts ``min(...)`` of them. On a hybrid GDN model
  with ``--mamba-scheduler-strategy extra_buffer`` the mamba component answers
  ``req.mamba_last_track_seqlen`` -- the last DECODE TRACK point, a multiple of
  ``--mamba-track-interval`` (256) -- because a recurrent state exists only
  there (the live state is not donated; off-grid positions are declined,
  #747). The KV above that point is freed and never written:
  27B pdflip-2-5 ``#1469 RETAIN token_ids_len=417 cache_len=256`` (arena census
  of the 416 keys: 255 COMPLETE, 161 never claimed); NF pdflip-1-13
  47262 -> 47104, pdflip-6-29 45697 -> 45568. That is correct: without a state
  at 417 the KV 256..416 cannot be resumed from, D recomputes it anyway.
* WHAT THE READ ASKED. The dormant-hold read (#1443/#1456) and the post-wake
  settle (#1471) register the request's WHOLE context
  (``full_untruncated_fill_ids``, 417 / 47262 tokens). The store can never
  complete that span -- the tail was never written:
  - NF: ``prefetch INCOMPLETE delivered=47104 deliverable=47232 shortfall=128``,
    every re-read of the 128 declined below the prefetch threshold
    (``vote_negative``), ``#1471 SETTLE-RELEASE state=wait lapsed=True
    held_after_wake_s=20.0`` -- 20 s for bytes that do not exist;
  - 27B: the store held 255 bigram pages -- ONE BELOW the prefetch threshold
    (256 tokens): the controller revoked every read (``storage_hit_count <
    prefetch_threshold``), the record materialized 0 (``#1456 HOLD-REFETCH
    reason=zero-answer`` x4), ``held_after_wake_s=20.1``, then a fresh read
    after the wake (the hand-off keys are dropped at the wake) answered
    ``#1035 R13 EMPTY KV PREFIX keys=417`` and D prefilled all 418 tokens.

THE REPAIR (one switch, ``FLLIPER_PDFLIP_PARK_READ_CAP``, default on; ``0`` = the
pre-fix read byte for byte):

* the park stamps, per retracted request, the raw token span of the key it
  inserted (``RETAINED_ATTR``, set by ``cache_finished_req``) together with
  the request's length at the park;
* while the request has not grown since (it is parked: held, settling, or
  re-admitted before its first new token), its store read ends at that span
  (:func:`park_match_end`) -- the read lands COMPLETE, nothing is short, the
  settle has nothing to wait for, and the resume's extend computes the tail
  (< one track interval, far below X);
* the read may be smaller than the prefetch threshold
  (:func:`park_read_min_tokens` = 1): the park wrote it, it is there, and it
  is the only copy of the anchor (27B: 255 pages).

Every term is a pure function of the request's token ids and the retained key,
which every rank of the group computes identically -- no collective, no file.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

ENV = "FLLIPER_PDFLIP_PARK_READ_CAP"
#: set by ``UnifiedRadixCache.cache_finished_req`` on every insert: the raw
#: token count the inserted key spans (bigram: units + 1).
RETAINED_ATTR = "_pdflip_retained_raw_tokens"
#: set by the flip park: ``(retained_raw_tokens, request_tokens_at_park)``.
CAP_ATTR = "_pdflip_park_read_cap"


def enabled(env=None) -> bool:
    e = os.environ if env is None else env
    raw = (e.get(ENV, "") or "").strip().lower()
    return raw not in ("0", "false", "no", "off")


def retained_raw_tokens(radix_key) -> int:
    """Raw token span of an inserted key (bigram keys hold N+1 raw tokens for N
    units; an empty key spans nothing)."""
    n = len(radix_key)
    if n <= 0:
        return 0
    return n + 1 if getattr(radix_key, "is_bigram", False) else n


def _request_tokens(req) -> Optional[int]:
    try:
        return len(req.origin_input_ids or ()) + len(req.output_ids or ())
    except (AttributeError, TypeError):
        return None


def stamp_parked(req) -> Optional[int]:
    """Called by ``park_running`` right after the retaining retraction. Returns
    the cap (raw tokens) or None (switch off / nothing retained / no stamp)."""
    setattr(req, CAP_ATTR, None)
    if not enabled():
        return None
    retained = getattr(req, RETAINED_ATTR, None)
    ntok = _request_tokens(req)
    if retained is None or ntok is None:
        return None
    retained = int(retained)
    setattr(req, CAP_ATTR, (retained, int(ntok)))
    return retained


def clamp_to_resumable(req, depth, *, is_bigram: bool) -> Optional[int]:
    """PARK-READ = RESUMABLE (29.09., 27B z30j pdflip-154-176 00:51:55): lower
    the park's cap to the depth the resume can take (``#59b`` park depth, the
    group-uniform admission probe). Returns the new cap, None = unchanged.

    The retaining insert keeps the full-attention KV at full length and puts a
    mamba TOMBSTONE above the last track point (#783/#1012: an off-grid end
    declines the anchor, not the length) -- ``retained=72271 of 72272``. But no
    admission resumes above the anchor, and the #248 park mark keeps only the
    resumable chain (``PARK-MARK pages=72191``), so the store never holds the
    79 units in between. A cap at the retained span therefore asked for bytes
    that do not exist: every such resume ran ``prefetch INCOMPLETE shortfall=79``
    -> ``#1068 DEFERRED store_prefix_short`` -> ``#1456 HOLD-REFETCH`` ->
    ``vote_negative`` -> ``PREFETCH-DEFER-FALLBACK`` before it took 72191 and
    extended the tail anyway. At the resumable depth the read lands COMPLETE on
    the first pass; the tail extend is the same (without a recurrent state
    above the anchor the tail is computed either way -- the #747 grid)."""
    cap = getattr(req, CAP_ATTR, None)
    if not cap or not enabled() or depth is None:
        return None
    retained, ntok = cap
    try:
        depth = int(depth)
    except (TypeError, ValueError):
        return None
    if depth <= 0:
        return None  # 0 is also the probe's "cannot price" -- keep the retained cap
    # the depth counts key units; a bigram key of N units spans N + 1 raw tokens
    raw = depth + 1 if is_bigram else depth
    if raw >= int(retained):
        return None
    setattr(req, CAP_ATTR, (raw, int(ntok)))
    return raw


def read_cap(req) -> Optional[int]:
    """The park's cap while it applies: the request has not grown since the
    park (a resumed request that decoded is past it; its next park restamps)."""
    cap = getattr(req, CAP_ATTR, None)
    if not cap or not enabled():
        return None
    retained, ntok = cap
    if _request_tokens(req) != ntok:
        return None
    return int(retained)


def park_match_end(req, match_end: int) -> int:
    """End of ``req``'s store-read span: ``match_end`` unless the park's cap
    lies below it."""
    cap = read_cap(req)
    if cap is None or cap >= int(match_end):
        return match_end
    return cap


def park_read_min_tokens(req) -> Optional[int]:
    """1 while the park's cap applies (the read may be below the prefetch
    threshold), else None = the tree's threshold."""
    return 1 if read_cap(req) is not None else None


#: W88 CYCLE (29.09., NF dauer09290232 D 03:21:06, pdflip-122-144 / pdflip-126-148):
#: everything the store-read chain remembers ABOUT ONE READ. The #1324 stamp
#: (``_pdflip_store_delivered``) was already cleared at a new cycle; the progress
#: witness and the store-short bound were not. The next cycle's first drain
#: then compared its fresh read (29696 of 97024) against the previous cycle's
#: best (96000, "no growth"), counted on from the previous cycle's passes (9)
#: and wall clock (49 s old >= the 30 s bound) -> W88 terminal 0.6 s after the
#: read started, on ONE pass (pdflip-126-148: no_progress_passes=1).
READ_CYCLE_ATTRS = (
    "_pdflip_store_delivered",
    "_pdflip_best_delivered",
    "_pdflip_progress_terms",
    "_pdflip_no_progress_passes",
    "_pdflip_no_progress_t0",
    "_pdflip_store_short_cycle_best",
    "_pdflip_store_short_cycles",
    "_pdflip_store_short_fallback",
)


def clear_read_cycle(req) -> int:
    """A NEW store-read cycle of ``req`` begins (flip park, RESUME-VIA-P hold):
    forget what the chain witnessed about the previous cycle's read. Returns
    how many terms were standing. Replicated: every rank parks / holds the same
    requests in the same pass."""
    n = 0
    for attr in READ_CYCLE_ATTRS:
        if getattr(req, attr, None) not in (None, False, 0):
            n += 1
        if hasattr(req, attr):
            setattr(req, attr, None)
    return n


def describe(req) -> str:
    cap = getattr(req, CAP_ATTR, None)
    if not cap:
        return "cap=none"
    retained, ntok = cap
    return "retained=%d of %d (tail %d = the resume's extend, not read)" % (
        retained, ntok, max(0, ntok - retained))
