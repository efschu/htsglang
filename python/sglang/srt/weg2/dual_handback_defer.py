"""D-HANDBACK-DEFER (dual layout, group D): a hand-back whose tail is not yet
readable in the store is DEFERRED, not refused.

Metal gmps13 (boot ...dual1mpsleepbar1fs10020145_4a632a3a94, D TP0): P's leg 1
for weg2-0-19 ended at 01:50:25.003 (19869 tokens, 17425 cached, END-ANCHOR ok
on PP0/1/2), the front admitted it to D 66 ms later, and D's presence probe with
P's hand-off keys answered ``covered=2443 pages=0 present=False``. The X gate
priced the 2444-token tail against X=1 (W31), RESUME-VIA-P had P prefill the
SAME tail a second time (the no-double-prefill law), and D picked the stream up
again only at its park requeue: 29.8 s of a frozen client stream. 17 rids, 33 such
P re-legs in 10 minutes; the cached negative presence verdict
(``_pp_store_presence_cache``) made later attempts refuse without asking again.

The repair, in the machinery that already exists:

* ``begin`` -- the X gate's W31 on a dual-D request (every D request of the dual
  layout is a hand-back; D never prefills) marks the request ONCE and the
  admission loop skips it instead of refusing. The W31 verdict is the group's
  (#823 MIN-reduced match), so the mark is set on the same pass on every rank.
* ``pending`` -- this rank's VOTE into the existing X-completion arm
  (``Scheduler._weg2_store_read_is_pending`` -> the packed MIN reduce ->
  ``_weg2_x_defers``): pending while the mark stands and no read has been issued
  for it. The verdict is the group's MIN, bounded by the existing length-priced
  store-read bound (``_weg2_x_store_read_bound_s``); past it the request is
  priced, W31 fires again and ``begin`` answers spent -> the refusal as before.
  The vote also stops on its own at that bound, so a windowed path (whose gate
  bound is unbounded) can never wedge on it.
* ``retry`` -- once per pass, beside ``_retry_deferred_prefetches``, the marked
  requests re-issue their store read in kv_arrival_seq order on a PASS-COUNTED
  back-off (1, 2, 4, 8, then every 8th pass): the set, its order and the count
  are identical on every rank, so every rank enters the #580 vote together. The
  negative presence verdict is dropped before every re-issue.
* ``note_admit`` / the spent branch -- one ``D-HANDBACK-DEFER`` line per state
  change: ``n=`` episode, ``passes=`` passes seen, ``ms=`` wall since the mark
  (rank-local, instrument only), ``tail=`` the extent the first W31 priced.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)

LINE = "D-HANDBACK-DEFER"
MARK_ATTR = "_weg2_hb_defer"
#: the longest pass interval between two re-issues of one marked hand-back
RETRY_MAX_INTERVAL = 8
_N = [0]


def armed(env=None) -> bool:
    """Group D of the dual layout (the same predicate as the retract tripwire)."""
    e = os.environ if env is None else env
    return ((e.get("SGLANG_WEG2_DUAL_LAYOUT", "") or "").strip() == "1"
            and (e.get("SGLANG_WEG2_GROUP", "") or "").strip().upper() == "D")


def forget_presence(req) -> None:
    """Drop the cached presence verdicts, so the next probe ASKS again."""
    for attr in ("_pp_store_presence_cache", "_weg2_store_match_cache"):
        try:
            setattr(req, attr, None)
        except Exception:  # noqa: BLE001 -- a request without the slot has nothing cached
            pass


def _line(st: dict, req, state: str, now, extra: str = "") -> None:
    logger.warning("%s n=%d passes=%d ms=%d tail=%d rid=%s state=%s%s", LINE, int(st["n"]), int(st["passes"]),
                   int((float(now()) - float(st["t0"])) * 1000), int(st["tail"]),
                   str(getattr(req, "rid", "?"))[:16], state, extra)


def _park_site(req):
    from sglang.srt.weg2 import d_seats  # lazy: d_seats reads this module back

    return d_seats.park_site(req)


def d_own_tail(req, env=None) -> int:
    """Q-692 D-OWN-TAIL: the extent at the end of ``req`` that D owns, or 0.

    Metal 27B NVFP4 dual fs10031727 (bc2bd121c0), weg2-0-373: D decoded 25
    tokens, SEAT-AGE DISPLACE parked it (pressure park, span retained), and
    the retain kept the KV of the decoded tokens but no GDN/Mamba state at the
    span's end -- the resume depth is P's anchor 94918, ``uncached=26`` against
    the dual layout's X=1. That tail is the ``output_ids`` D generated plus the
    decode input: D's own extend. P never writes it (P's leg 1 ends at the
    prompt), so neither the X gate's W31 nor a hand-back defer can ever be
    answered by P.

    ``len(output_ids) + 1`` when ``req`` carries output D decoded (dual D
    only), else 0. Replicated: every rank decoded the same tokens."""
    if not armed(env):
        return 0
    out = len(getattr(req, "output_ids", None) or ())
    return out + 1 if out > 0 else 0


def d_owned(req, env=None) -> bool:
    """Q-692: ``req`` carries output D decoded or a PRESSURE park site (D
    retracted/displaced it while it ran) -- its end is D's, not a P hand-back
    (dual D only). A FLIP site without output is #248h's capacity park of a
    fresh hand-back, whose tail IS P's: that one keeps the defer. Replicated
    terms (the tokens; the park is a group verdict)."""
    if not armed(env):
        return False
    if len(getattr(req, "output_ids", None) or ()) > 0:
        return True
    from sglang.srt.weg2 import d_seats

    return _park_site(req) == d_seats.SITE_PRESSURE


X_DEFER_ATTR = "_weg2_x_deferring"


def note_x_defer(req, deferring: bool, env=None) -> None:
    """Q-692: the X-completion arm's group verdict for ``req`` on this pass
    (True = WEG2 X-DEFER), read by the NEXT pass's D-park barrier. Dual D only."""
    if not armed(env):
        return
    try:
        setattr(req, X_DEFER_ATTR, bool(deferring))
    except Exception:  # noqa: BLE001 -- a request without the slot has no barrier vote
        pass


def defer_exempt(req, env=None) -> bool:
    """Q-692 PARK BARRIER: a parked request that is held in D-HANDBACK-DEFER
    (an open mark) or in WEG2 X-DEFER waits for a store read, not for a seat --
    it must hold no younger newcomer back (metal: ADMISSION-WEDGE 4-6 queued,
    0 running, 83 s behind weg2-0-373). Both terms are group verdicts (W31 /
    the MIN-reduced pending arm), so the barrier stays the group's."""
    if not armed(env):
        return False
    st = getattr(req, MARK_ATTR, None)
    if st is not None and not st.get("spent") and not st.get("done"):
        return True
    return bool(getattr(req, X_DEFER_ATTR, False))


REARM_LINE = "#1420r DEFER-REARM"
REARM_ENV = "SGLANG_WEG2_DUAL_HANDBACK_DEFER_REARM"
#: at most one REARM line per this many seconds (instrument only, rank-local clock)
REARM_LOG_EVERY_S = 1.0
_REARM_LAST_LOG = [None]


def rearm_max(env=None) -> int:
    """The re-arm limit (``SGLANG_WEG2_DUAL_HANDBACK_DEFER_REARM``); 0 = off (default)."""
    e = os.environ if env is None else env
    try:
        return max(0, int((e.get(REARM_ENV, "") or "0").strip() or "0"))
    except ValueError:
        return 0


def _rearm(req, st: dict, now, env=None) -> bool:
    """#1420r: a second W31 on a mark whose store read was already issued (the read
    landed empty: P's write-through of the tail anchor is asynchronous) re-arms the
    mark instead of spending it, up to ``rearm_max`` times. Dual D only (``begin``
    checked ``armed``); off (0) = False = the old single-shot behaviour.

    RANK AGREEMENT (a rank divergence is forbidden): the decision reads only
    (a) the W31 verdict that called ``begin`` -- the group's MIN-reduced match (#823),
    so every rank is here on the same pass -- (b) ``st["issued"]``, set by ``retry``
    from the pass-counted back-off over the replicated waiting queue, and (c) the
    counter ``st["rearm"]``, advanced only here. No wall clock enters; the wall bound
    stays the existing length-priced MIN vote in ``pending`` (a re-armed mark is
    pending again, ``issued`` False). ``retry_seen`` keeps counting, so the back-off
    keeps widening (1, 2, 4, 8, then every 8th pass)."""
    limit = rearm_max(env)
    done = int(st.get("rearm", 0))
    if limit <= 0 or done >= limit:
        return False
    st["rearm"] = done + 1
    st["issued"] = False
    forget_presence(req)
    t = float(now())
    last = _REARM_LAST_LOG[0]
    if last is None or t - last >= REARM_LOG_EVERY_S or done + 1 >= limit:
        _REARM_LAST_LOG[0] = t
        logger.warning("%s n=%d rearm=%d max=%d passes=%d retry_seen=%d tail=%d rid=%s ms=%d -- the re-read "
                       "landed empty (P's tail write is still in flight); mark re-armed, not spent",
                       REARM_LINE, int(st["n"]), done + 1, limit, int(st["passes"]),
                       int(st.get("retry_seen", 0)), int(st["tail"]), str(getattr(req, "rid", "?"))[:16],
                       int((t - float(st["t0"])) * 1000))
    return True


def begin(req, tail: int, *, now=time.monotonic, env=None) -> bool:
    """The X gate priced ``req`` W31. True = defer it this pass (first time: the
    mark is set); False = refuse as before (not dual D, or the mark is spent --
    the bound expired or the group priced it short after the read).

    Q-692: never for a request whose end is D's own (``d_owned``: output D
    decoded, or a D park site) -- P never writes that tail, so the defer
    could only run into its bound (94.7 s for weg2-0-373)."""
    if not armed(env):
        return False
    if d_owned(req, env):
        if not getattr(req, "_weg2_hb_d_own_said", False):
            try:
                req._weg2_hb_d_own_said = True
            except Exception:  # noqa: BLE001
                pass
            logger.warning("%s tail=%d rid=%s state=d_own out=%d site=%s -- D's own end (decoded "
                           "before a park), not a P hand-back: no defer", LINE, int(tail),
                           str(getattr(req, "rid", "?"))[:16],
                           len(getattr(req, "output_ids", None) or ()), _park_site(req))
        return False
    st = getattr(req, MARK_ATTR, None)
    if st is not None:
        if not st.get("spent") and not st.get("done") and st.get("issued") and _rearm(req, st, now, env):
            return True
        if not st.get("spent"):
            st["spent"] = True
            _line(st, req, "refused", now, " -- the tail stayed unreadable past the length-priced bound; "
                  "the refusal (RESUME-VIA-P) follows")
        return False
    _N[0] += 1
    st = {"n": _N[0], "t0": float(now()), "passes": 0, "tail": int(tail), "spent": False, "issued": False,
          "retry_seen": 0}
    setattr(req, MARK_ATTR, st)
    forget_presence(req)
    _line(st, req, "begin", now, " -- P's tail is not readable in the store yet; deferred instead of "
          "refused (no second P prefill)")
    return True


def pending(req, bound_s: float, *, now=time.monotonic) -> bool:
    """This rank's vote: the mark stands, no read has been issued for it yet, and
    this rank's own wait is inside the length-priced bound."""
    st = getattr(req, MARK_ATTR, None)
    if st is None or st.get("spent") or st.get("issued") or st.get("done"):
        # Q-692: an admitted mark's episode is over -- a later park of the
        # same request (D decoded, displaced) is no hand-back wait
        return False
    if bound_s <= 0 or float(now()) - float(st["t0"]) > float(bound_s):
        return False
    return True


def _due(st: dict) -> bool:
    """Pass-counted back-off: re-issue on the 1st, 2nd, 4th, 8th, then every 8th
    pass of this mark. Rank-identical: the count is per mark, not per clock."""
    n = int(st.get("retry_seen", 0)) + 1
    st["retry_seen"] = n
    interval = min(RETRY_MAX_INTERVAL, 1 << (n.bit_length() - 1))
    return n % interval == 0


def retry(sched, *, now=time.monotonic) -> int:
    """Once per pass on every D rank: re-issue the store read of every marked,
    still-unread hand-back that is due. Returns how many were re-issued."""
    if not armed():
        return 0
    marked = [r for r in (getattr(sched, "waiting_queue", None) or ())
              if (lambda st: st is not None and not st.get("spent") and not st.get("issued")
                  and not st.get("done"))(
                  getattr(r, MARK_ATTR, None))]
    if not marked:
        return 0
    marked.sort(key=lambda r: int(getattr(r, "kv_arrival_seq", 0) or 0))
    issued = 0
    for req in marked:
        st = getattr(req, MARK_ATTR)
        st["passes"] += 1
        if not _due(st):
            continue
        forget_presence(req)
        verdict = str(sched._prefetch_kvcache(req) or "")
        if verdict.startswith("issued") or verdict == "declined:already_in_flight":
            st["issued"] = True
            issued += 1
            _line(st, req, "read", now, " verdict=%s -- the tail is readable now; the read registered" % verdict)
    return issued


def note_admit(req, *, now=time.monotonic) -> None:
    """The X gate admitted a marked request: the episode's end line."""
    st = getattr(req, MARK_ATTR, None)
    if st is None or st.get("spent") or st.get("done"):
        return
    st["done"] = True
    _line(st, req, "admitted", now, " -- no refusal, no second P prefill")
