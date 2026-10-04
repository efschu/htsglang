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


def note_reap(rid: str, requested_pages: int, hit_pages: int, completed_tokens: int, page_size: int) -> None:
    """At the reap: remember whether a short read's span was held by the store."""
    rid = str(rid)
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
