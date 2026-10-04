# SPDX-License-Identifier: Apache-2.0
"""P4b (28.09.): who can still fill a SHORT post-wake store read, and has it?

NF rc12z17-s0 (103bfdf29a, boot ...09281111): three requests routed ``short``
straight to D (weg2-2-17, weg2-4-36, weg2-5-37) read the store short at the
wake (shortfall 448 / 1728 / 128 tokens), re-read every 2 s -- ``#1456
HOLD-REFETCH zero-answer``, ``#1442 HANDOFF-KEYS NONE registry=[]``,
``#1472 READ-TRACE why=no-file`` -- and sat the whole 20 s ``#1471`` settle
bound out (``SETTLE-RELEASE lapsed=True held_after_wake_s=20.0``) before the
X gate admitted exactly the remainder it would have admitted at once
(uncached 509 / 1757 / 191). Nobody was writing: the missing pages were D's
own decode tail of the previous turn, never secured when D slept, and P never
saw these requests.

The only process that can still fill a parked read after D's wake is P (D
awake = P asleep; D's own write-through writes pages D holds on its device,
and a page on D's device is a device match, never a store read). What D can
SEE of P, per rid, in the shared hand-off directory:

* the hand-off chain (``weg2.handoff`` record, cached on the request as
  ``handoff_keys.CHAIN_ATTR``) -- P prefilled this rid;
* the E-tail parts (``weg2.tail_handoff``): a ``*.tmp`` part or a manifest
  short of its PP ranks = P's publish threads are still WRITING; a complete
  manifest = P's publish ACK.

States, and what the settle hold does with them:

``none``        no chain, no tail part, nothing under write: no writer exists.
                Decided now -- the remainder goes to admission (within X D
                prefills it; over X the X gate refuses by name and the front
                re-routes via P). Never another re-read.
``p-writing``   P's tail parts are being written: wait for the ack, no
                re-read (the re-read cannot see pages that are not there).
``p-published`` P's parts are complete: the ack -- re-read now, once.
``p-handoff``   P prefilled it, no tail parts to watch: P's page write-through
                has no ack D can see; the 2 s re-read stays (unchanged).
``unknown``     no shared hand-off directory (``SGLANG_HICACHE_ARENA_DIR``
                unset): nothing of P is visible, so no writer is provable --
                the 2 s re-read stays (unchanged).

Pure functions over observed facts, plus :func:`observe`, which gathers them.
"""

from __future__ import annotations

import glob
import os
from typing import Optional, Tuple

NONE = "none"
P_WRITING = "p-writing"
P_PUBLISHED = "p-published"
P_HANDOFF = "p-handoff"
#: no shared hand-off directory: D cannot see P, so "no writer" is not provable
UNKNOWN = "unknown"

#: states in which nothing may be re-read yet (a writer is at work)
PENDING = frozenset({P_WRITING})


def classify(*, chain: bool, handoff_file: bool, tail_state: str, tail_tmp: bool) -> str:
    """The writer state of one parked rid from what D can see.

    ``tail_state`` is ``tail_handoff.manifest_state(...)[0]`` ('none',
    'partial', 'complete', 'excess', 'differs', 'legacy')."""
    if tail_tmp or tail_state == "partial":
        return P_WRITING
    if tail_state in ("complete", "legacy", "excess", "differs"):
        return P_PUBLISHED
    if chain or handoff_file:
        return P_HANDOFF
    return NONE


def step(prev: Optional[str], now: str, ack_seen: Optional[str]) -> Tuple[str, Optional[str]]:
    """What the settle hold does this tick: (action, new ack marker).

    action: ``wait`` (a writer is at work: no re-read), ``reread`` (the ack
    arrived since the last tick: re-read now, past the 2 s timer),
    ``decide`` (no writer: release to admission now), ``poll`` (a writer
    without a visible ack: the unchanged 2 s re-read). A published manifest
    is an ack ONCE; after its re-read a still-short read has no writer left."""
    if now in PENDING:
        return "wait", ack_seen
    if now == P_PUBLISHED:
        if ack_seen != P_PUBLISHED:
            return "reread", P_PUBLISHED
        return "decide", ack_seen
    if now == NONE:
        if prev in PENDING:
            # the parts vanished while we waited (consumed / pruned): read once more
            return "reread", ack_seen
        return "decide", ack_seen
    return "poll", ack_seen


# ---------------------------------------------------------------------------
# NW (30.09.): a read the HOST BUDGET refused is not a read that came back short.
#
# NF y3u (5bedac26f1, boot ...0930_002717, D TP0): five requests parked over the
# flip, three of them ~130k tokens (weg2-0-5 130048, weg2-30-49 131136,
# weg2-30-50 131136 = 392320 registered). The wake issued their reads first, in
# hold order, and the prefetch budget (limit 373536) was spent: the reads of
# weg2-30-52 (77695 tokens) and weg2-31-53 (77854) were REFUSED before they ran
# (``#915 PREFETCH REFUSED reason=vote_negative need=77632 occupied=392320
# limit=373536`` at the wakes 00:41:52, 00:42:08, 00:42:21). P had published
# their tails (``WEG2-TAIL-PUBLISH ... of=3``; D: ``SETTLE-WRITER
# writer=p-published action=reread``), the re-read came back
# ``declined:rate_limited`` -- and the settle took that as the ack spent and,
# one tick later, ``SETTLE-NO-WRITER`` ("nothing can fill the rest"). Admission
# then priced the whole prompt on a read that never ran (``X-GATE-TERMS
# total=77695 head=0 store=0``), the front re-routed via P twice, and the third
# refusal reached the client as W50 (``LEG2-TERMINAL-NAMED reason=W50``).
#
# The law: a refused registration is "not read yet", never "read short". Such a
# request stays parked (bounded by the #1471 settle bound), is re-read as soon
# as the budget has room (not on the 2 s timer), keeps the writer's ack, and
# is never decided "no writer" and never tail-settled on a stale stamp.
# ---------------------------------------------------------------------------

#: the request's last store read was refused by the host budget / the group vote
BUDGET_ATTR = "_1471b_budget"
#: monotonic time of that refused attempt (the retry cadence)
BUDGET_T_ATTR = "_1471b_t"
#: the refusal terms that mean "the budget had no room" (match_refusal_census
#: PREFETCH_DECLINE_ORDER minus anchor / too_short, which are answers, not room)
BUDGET_TERMS = frozenset({
    "rate_limited",
    "host_pool_exhausted",
    "host_alloc_failed",
    "anchor_pool_exhausted",
    "vote_negative",
    "alloc_failed_post_vote",
})
#: the shortest interval between two refused attempts of one request -- a retry
#: is a group collective (the #580 vote), so not on every scheduling pass
BUDGET_RETRY_S = 0.25


#: #287 NEED0 (30.09., NF y4k weg2-0-4): ``vote_negative`` is the GROUP's exit
#: -- a common span below the prefetch threshold, 0 included. It means "no
#: room" only when this rank asked for something the host could not give
#: (y3u: need=77632 over a full budget). ``need=0 available=415040`` (a
#: 25-token prompt, retained=0, nothing registered to read) is an ANSWER:
#: 533 group re-reads over 12 D phases, 525 s wall for 2 tokens, one D seat
#: held through the whole bench (bs6 ran as bs5).
VOTE_TERMS_ATTR = "_weg2_refusal_terms"


def refusal_terms(tree, rid) -> Optional[Tuple[str, int, int]]:
    """``(reason, need, room)`` of the last #915 refusal of ``rid`` on this
    rank's tree (``UnifiedRadixCache._log_prefetch_refused``), or None. ``room``
    = min(host pool available, prefetch limit - occupied): y3u refused
    need=77824 with available=415040 but occupied=392320 > limit=373536."""
    terms = getattr(tree, VOTE_TERMS_ATTR, None) if tree is not None else None
    if not terms:
        return None
    return terms.get(str(rid))


def budget_refused(verdict, terms: Optional[Tuple[str, int, int]] = None) -> bool:
    """True when ``_prefetch_kvcache``'s verdict is a budget refusal.

    ``vote_negative`` counts only with the rank's own terms showing a real
    shortage (``need > 0`` and ``need > room``, the room the smaller of the
    host pool's free rows and the prefetch budget left); without them, or with
    ``need == 0``, it is the group's answer "nothing worth reading"."""
    v = str(verdict or "")
    if not v.startswith("declined:"):
        return False
    term = v.split(":", 1)[1]
    if term not in BUDGET_TERMS:
        return False
    if term == "vote_negative":
        if terms is None:
            return False
        _reason, need, room = terms
        return int(need) > 0 and int(need) > int(room)
    return True


def note_read_verdict(req, verdict, now: float, tree=None) -> bool:
    """Record what the last read attempt of ``req`` did; True = budget-refused.
    Any other verdict (issued, in flight, too short, ...) clears the mark.
    ``tree``: this rank's radix tree, whose last #915 terms of ``req`` decide
    a ``vote_negative`` (see :func:`budget_refused`)."""
    refused = budget_refused(verdict, refusal_terms(tree, getattr(req, "rid", None)))
    setattr(req, BUDGET_ATTR, refused)
    if refused:
        setattr(req, BUDGET_T_ATTR, float(now))
    return refused


def budget_pending(req) -> bool:
    return bool(getattr(req, BUDGET_ATTR, False))


def budget_retry_due(req, now: float, rate_limited: bool) -> bool:
    """A budget-refused read is re-issued once this rank's budget has room and
    the retry interval passed (rank-local; the caller takes the group MIN)."""
    if rate_limited:
        return False
    return now - float(getattr(req, BUDGET_T_ATTR, 0.0) or 0.0) >= BUDGET_RETRY_S


def gate_action(act: str, req) -> str:
    """The settle action under a pending budget refusal: ``decide`` becomes
    ``poll`` -- the read has not run, so "no writer" is not proven."""
    if act == "decide" and budget_pending(req):
        return "poll"
    return act


def keep_ack_if_unread(req, act: str) -> None:
    """A ``reread`` whose re-read the budget refused has not spent the ack:
    the next tick re-reads again once the budget has room."""
    if act == "reread" and budget_pending(req):
        req._1471w_ack = None


def settle_since_for_wake(req, now: float) -> float:
    """#287 NEED0 (b): the #1471 settle clock of a held request at a new wake.

    A request whose last store read was budget-refused keeps the clock of its
    FIRST refused wake: the bound (``WEG2_POST_WAKE_SETTLE_S``) runs over
    wakes, so a request no D phase of < 20 s could release (y4k weg2-0-4,
    12 phases of 10-18 s) is released at the next wake instead of being
    re-stamped for ever. Every other request starts the bound at this wake,
    as before."""
    since = getattr(req, "_1471_since", None)
    carry = getattr(req, CARRY_ATTR, None)
    setattr(req, CARRY_ATTR, None)
    setattr(req, CARRIED_ATTR, False)
    if budget_pending(req) and since is not None:
        return float(since)
    if (carry is not None and carry_enabled()
            and getattr(req, "_weg2_settled_wake", None) == carry[1]):
        # SC (#1210): folded out of the settle into the flip park and still
        # short at this wake -- the clock of the FIRST short wake runs on; this
        # wake's own read is awaited once (:func:`carry_holds_lapse`).
        setattr(req, CARRIED_ATTR, True)
        return float(carry[0])
    return float(now)


# ---------------------------------------------------------------------------
# SC (#1210, 04.10., NF y9nf4 e23f7dff30 boot 1004_031945, D TP0): the sleep
# 03:33:18 found the Mamba anchor arena full (``#1427 ARENA-CLAIM REFUSED
# statuses=[4]`` x8, ``#1421 BACKUP-REFUSED why=mamba_claim/parent_unbacked``)
# and dropped the park anchors of weg2-8-79 (49408) and weg2-8-82 (76032)
# (``WEG2-ANCHOR-LOST at=flush``). Every later store read stopped at the
# deepest host anchor (47104 / 67840) -- for good, nobody could write it --
# yet the settle polled (``writer=p-handoff``), each park_running folded the
# two into the flip park (``settle-folded``) and each wake re-stamped the 20-s
# clock; no D phase (15-19 s) let it lapse: 10 wakes, 5.5 min, 427 s wall,
# and the park barrier held a 69-token newcomer (weg2-12-99) for 262 s.
# The bound runs over wakes for a folded request now (as #287 NEED0 for a
# budget-refused one); the wake's own read is awaited once, so a store P did
# complete in its phase is still read before the lapse.
# ---------------------------------------------------------------------------

#: the settle clock carried out of the settle by a flip-park fold:
#: ``(_1471_since, _weg2_settled_wake at the fold)`` -- stale once the request
#: was released (any release stamps ``_weg2_settled_wake``)
CARRY_ATTR = "_1471_carry"
#: this wake's settle clock is a carried one (cleared once its read answered)
CARRIED_ATTR = "_1471_carried"
CARRY_ENV = "SGLANG_WEG2_SETTLE_CLOCK_CARRY"


def carry_enabled(env=None) -> bool:
    """SC, default ON; 0 = every wake starts the bound anew (as before)."""
    e = os.environ if env is None else env
    return str(e.get(CARRY_ENV, "1")).strip().lower() not in ("0", "false", "no", "off")


def note_fold(req) -> None:
    """A settle request folded into the flip park: remember its clock."""
    since = getattr(req, "_1471_since", None)
    if since is None:
        return
    setattr(req, CARRY_ATTR, (float(since), getattr(req, "_weg2_settled_wake", None)))


def carry_holds_lapse(req, state: str) -> bool:
    """True while a carried clock waits for this wake's OWN read (still in
    flight). Once that read has answered the hold is spent for this wake --
    a later 2-s re-read never holds the lapse again (rank-local; the release
    is the group MIN of the lapse flags)."""
    if not getattr(req, CARRIED_ATTR, False):
        return False
    if state == "reading":
        return True
    setattr(req, CARRIED_ATTR, False)
    return False


def reset_for_wake(req) -> None:
    """A new wake follows a P phase in which P may have written this rid again
    (RESUME-VIA-P publishes the same tail files anew): the writer view of the
    previous wake -- state, spent ack -- does not carry over."""
    for attr in ("_1471w_state", "_1471w_ack", "_1471w_t"):
        setattr(req, attr, None)


def _tail_facts(rid: str) -> Tuple[str, bool]:
    from sglang.srt.weg2 import tail_handoff as _th

    try:
        headers = _th.headers_for(rid)
        state = _th.manifest_state(headers)[0]
    except Exception:  # noqa: BLE001 - unreadable parts read as a writer at work
        return "partial", False
    d = _th._dir()
    tmp = bool(d) and bool(glob.glob(os.path.join(d, f"{glob.escape(rid)}.tail.*.tmp")))
    return state, tmp


def observe(req) -> str:
    """Gather the facts for ``req`` and classify them (rank-local; the caller
    takes the group MIN of every verdict built on it)."""
    from sglang.srt.weg2 import handoff as _ho
    from sglang.srt.weg2.handoff_keys import CHAIN_ATTR, SEEN_ATTR

    if not _ho._dir():
        return UNKNOWN
    rid = str(getattr(req, "rid", "") or "")
    if not rid:
        return UNKNOWN
    # SEEN_ATTR (P4b-fix): the record this rank once read may be gone after the wake
    chain = bool(getattr(req, CHAIN_ATTR, None)) or bool(getattr(req, SEEN_ATTR, False))
    handoff_file = _ho.read(rid) is not None
    tail_state, tail_tmp = _tail_facts(rid)
    return classify(chain=chain, handoff_file=handoff_file, tail_state=tail_state, tail_tmp=tail_tmp)


def remainder(req, records) -> Optional[int]:
    """Tokens D prefills if ``req`` is admitted on what its read delivered:
    the registered span minus the group-synced delivered prefix (the #1324
    stamp when present). None = no span known."""
    span = int(getattr(req, "_prefetch_span_tokens", 0) or 0)
    delivered = getattr(req, "_weg2_store_delivered", None)
    if delivered is None:
        have = (records or {}).get(str(getattr(req, "rid", "")))
        delivered = int(getattr(have, "materialized", have) or 0) if have is not None else None
    ids = getattr(req, "full_untruncated_fill_ids", None)
    total = len(ids) if ids is not None else span
    if not total or delivered is None:
        return None
    return max(0, int(total) - int(delivered))


def route(rem: Optional[int], x: int) -> str:
    """Where the decided remainder goes: D prefills within X (or with no
    riegel); over X the X gate refuses by name and the front re-routes via P."""
    if rem is None or x <= 0 or rem <= x:
        return "D"
    return "P"


