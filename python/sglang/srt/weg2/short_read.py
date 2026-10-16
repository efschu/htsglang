"""A short wake read whose pages the store HAS is re-read at once.

y3m 09292136 (D-Nachlauf wächst über den Boot, ep24/ep46): once the agent
contexts outgrew the 4 GiB arena (wake read sets 395-411k tokens from 21:49
on, L3 fill 0 before), the wake reads came back short although the group's
store probe had found every page (``#1157 PREFETCH REAPED requested_pages=3841
hit_pages=3841 completed=0``): the batch that could not reference or fill a
page (arena full of the sibling reads' references) ended the read. The #1471
settle then re-read on its 2 s timer -- ep46 weg2-36-49: five re-reads
(3008 -> 5440 -> 7936 -> ... -> 64256 tokens), held 10.2 s; ep24 22-38 held
3.6 s. Nobody was writing those pages; the timer only waited.

The rule: a read reaped short while the group's probe held its whole span
is a READ failure, not a missing writer -- the next settle tick re-reads it
(past the 2 s timer) as long as each read moved the record forward. A read
that did not move falls back to the timer (an arena that frees nothing is
not hammered). ``held`` comes from the reap's group-MIN hit count, so every
rank records the same verdict; the re-read itself still goes through the
settle's group MIN on "due".
"""

from __future__ import annotations

import logging
from typing import Dict

logger = logging.getLogger(__name__)

MARK = "WEG2-SHORT-READ HELD"
#: rids remembered (a reap without a later settle leaves one entry behind)
KEEP = 64
_HELD: Dict[str, bool] = {}
_N = [0]
PROGRESS_ATTR = "_weg2_short_read_have"


#: P-HELD-REREAD-1014: rids whose read ended short of what the store holds
#: (hit > completed), the span need not be whole -- PP0 re-reads before its told
_MORE: Dict[str, bool] = {}


def store_holds_more(rid: str) -> bool:
    return bool(_MORE.get(str(rid), False))


#: ZERO-ANCHOR 1015 (Typ A, boot 133855 epoch 127, weg2-126-651): the front's
#: ``start_hint`` (PLE-HINT) is the depth D was MEASURED to hold for this prompt
#: (``d_leg2_cached`` / ``d_seq_anchor``). D runs write_back: its un-backed tail
#: (weg2-126-628, cache_len 45824, store 38656) reaches the store only in the
#: flip's flush (``PUBLISH-SWEEP unbacked=50 issued=37`` at 15:17:09.1) --
#: the same second P's dormant read of 651 completed at the store's 38656. A
#: read that ended below the hint is re-read once the flush is done (PP0 awake).
_HINT: Dict[str, int] = {}


def note_hint(rid, tokens) -> None:
    """The front's start_hint for ``rid`` (tokens; < 0 = none)."""
    try:
        t = int(tokens)
    except (TypeError, ValueError):
        return
    if rid is None or t <= 0:
        return
    _HINT[str(rid)] = t
    while len(_HINT) > 4 * KEEP:
        _HINT.pop(next(iter(_HINT)))


def hint_tokens(rid) -> int:
    return int(_HINT.get(str(rid), 0) or 0)


def hint_beyond(rid, have: int, page_size: int) -> bool:
    """The hint names at least one full page past ``have``."""
    return hint_tokens(rid) - int(have) >= max(1, int(page_size))


def note_reap(rid: str, requested_pages: int, hit_pages: int, completed_tokens: int, page_size: int) -> None:
    """At the reap: remember whether a short read's span was held by the store."""
    rid = str(rid)
    if int(hit_pages) * max(1, int(page_size)) > int(completed_tokens):
        _MORE[rid] = True
        while len(_MORE) > KEEP:
            _MORE.pop(next(iter(_MORE)))
    else:
        _MORE.pop(rid, None)
    short = int(completed_tokens) < int(requested_pages) * max(1, int(page_size))
    held = short and int(requested_pages) > 0 and int(hit_pages) >= int(requested_pages)
    if held:
        _HELD[rid] = True
        while len(_HELD) > KEEP:
            _HELD.pop(next(iter(_HELD)))
        _N[0] += 1
        if _N[0] <= 20 or _N[0] % 200 == 0:
            logger.info("%s rid=%s completed=%d of %d pages hit=%d (n=%d) -- the store holds the span: "
                        "the settle re-reads at its next tick, not on the 2 s timer",
                        MARK, rid, int(completed_tokens) // max(1, int(page_size)),
                        int(requested_pages), int(hit_pages), _N[0])
    else:
        _HELD.pop(rid, None)


def reread_now(req, have: int) -> bool:
    """True when the settle may re-read ``req`` now instead of waiting for
    the 2 s timer: its last read was reaped short with the span held, and
    the record moved since the previous re-read."""
    if not _HELD.get(str(req.rid), False):
        return False
    return int(have) > int(getattr(req, PROGRESS_ATTR, -1))


def note_reissue(req, have: int) -> None:
    """The settle re-issued ``req``'s read with ``have`` tokens on record:
    the next early re-read needs a record past this."""
    setattr(req, PROGRESS_ATTR, int(have))


def forget(rid: str) -> None:
    _HELD.pop(str(rid), None)
    _MORE.pop(str(rid), None)
    _HINT.pop(str(rid), None)
