"""PARK-RETAIN READ (27.09.): the controller honours a read's own floor.

``UnifiedRadixCache.prefetch_from_storage(min_tokens=...)`` lets a caller price
a read below the tree's ``prefetch_threshold`` (xsn437: the tail that completes
a short store read; PARK-RETAIN READ: the span a flip park retained). The
tree-level gate honoured it, but the prefetch thread applied the SAME threshold
a second time to the store hit -- ``storage_hit_count < self.prefetch_threshold``
-> revoke -- so such a read could never land: 27B pdflip-2-5 (metal
dkr27bparkdraftbar1w209270645) had 255 pages in the store, one under 256, and
four re-reads answered zero before the 20 s settle bound lapsed.

The floor travels by request id, registered before the operation is queued and
consumed once by the prefetch thread. It never RAISES the threshold, and a read
registered without a floor (every read but the two above) sees the threshold
unchanged. Group-uniform: the caller derives the floor from replicated request
state, and the store hit it is compared against is the group MIN.

Module functions (not methods) so a stand-in controller cannot miss them.
"""
from __future__ import annotations

from typing import Optional

ATTR = "pdflip_min_hit_tokens"
_CAP = 4096


def note_min_hit_tokens(controller, request_id, min_tokens: Optional[int]) -> None:
    """Register (``min_tokens`` set) or clear (None) ``request_id``'s floor."""
    if controller is None:
        return
    table = getattr(controller, ATTR, None)
    if table is None:
        if min_tokens is None:
            return
        table = {}
        try:
            setattr(controller, ATTR, table)
        except AttributeError:
            return
    if min_tokens is None:
        table.pop(request_id, None)
        return
    table[request_id] = max(1, int(min_tokens))
    while len(table) > _CAP:
        table.pop(next(iter(table)))


def revoke_threshold(controller, operation) -> int:
    """The store-hit floor below which the prefetch thread revokes
    ``operation``: the controller's threshold, lowered (never raised) by the
    floor its caller registered. Consumes the registration."""
    thr = int(getattr(controller, "prefetch_threshold", 0) or 0)
    table = getattr(controller, ATTR, None)
    if not table:
        return thr
    floor = table.pop(getattr(operation, "request_id", None), None)
    if floor is None:
        return thr
    return min(thr, max(1, int(floor)))
