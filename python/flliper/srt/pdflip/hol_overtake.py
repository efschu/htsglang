"""Group D admission: overtake by capacity, with a starvation guard for the head.

User rule (30.09., verbatim): "die regel ist ja die decode sitze bestmöglich
zu befüllen"; standing rule "freier D-Sitz + KV passt -> sofort prefillen".

Metal dual20 (...09301417, D TP0): the user's short OpenWebUI request
pdflip-0-10 (358 tokens) got D-ADMIT seat=5/6 at 14:27:26 and was served only
at 14:29:32 (126 s). D stood at running 2, #queue-req 3, #pending-token
253861: the head of D's waiting queue was a 112k leg 2 waiting for KV growth,
and every AddReqResult != CONTINUE -- NO_TOKEN included -- ended the whole
admission round with ``break`` (scheduler ``_get_new_batch_prefill_raw``).
One head that does not fit blocked everything behind it.

Now: on group D (flip form and dual layout alike; not P's PP path, which
follows PP0's row authority) a NO_TOKEN of the pass's first unfundable
request -- the HEAD -- lets the round go on to later requests; each one
that fits the FREE KV is admitted as usual (add_one_req decides, nothing is
forced). Bounded by ``MAX_SCAN`` refusals per pass.

The oldest keeps its precedence (user, verbatim: "der ältere ist immer
bevorzugt"): queue order is untouched, so it is tried FIRST in every pass and
any KV that frees or grows goes to it before an overtaker sees it; and
``d_park_runtime.displace_for_age`` (SEAT-AGE, KV trigger) parks the youngest
running seats -- only as many as the oldest needs, and only when that is then
enough (``victims_needed``) -- so a backfill never holds it off. No clock, no
FIFO fallback: backfill stays while the oldest waits for older large seats.

Switch ``FLLIPER_PDFLIP_D_HOL_OVERTAKE`` (default on for group D; ``0`` = the
old break). Off group D (no ``FLLIPER_PDFLIP_GROUP=D``) nothing changes."""

from __future__ import annotations

import logging
import os
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

ENV = "FLLIPER_PDFLIP_D_HOL_OVERTAKE"
#: further NO_TOKEN refusals a pass tolerates while scanning past the head
MAX_SCAN = 8


def enabled(sched: Any) -> bool:
    if str(os.environ.get(ENV, "1")).strip() == "0":
        return False
    if str(os.environ.get("FLLIPER_PDFLIP_GROUP", "")).strip().upper() != "D":
        return False
    try:
        if int(getattr(getattr(sched, "ps", None), "pp_size", 1) or 1) != 1:
            return False
    except (TypeError, ValueError):
        return False
    return True


class HolPass:
    """One admission round. ``may_overtake(req)`` is asked on a NO_TOKEN; True
    = keep scanning (the caller does its not-added cleanup, then continues)."""

    def __init__(self, sched: Any):
        self.s = sched
        self.on = enabled(sched)
        self.head: Optional[str] = None
        self.refusals = 0

    def _state(self) -> dict:
        return self.s.__dict__.setdefault("_pdflip_hol", {"rid": None, "passes": 0, "logged": 0})

    def may_overtake(self, req: Any) -> bool:
        if not self.on:
            return False
        st = self._state()
        rid = str(getattr(req, "rid", ""))
        if self.head is None:
            self.head = rid
            if st["rid"] != rid:
                st.update(rid=rid, passes=0)
            st["passes"] += 1
        self.refusals += 1
        return self.refusals <= MAX_SCAN

    def finish(self, admitted: List[Any]) -> None:
        """After the round: name the overtakers (throttled)."""
        if not self.on or self.head is None:
            return
        rids = [str(getattr(r, "rid", "")) for r in admitted or ()]
        if not rids:
            return
        st = self._state()
        st["logged"] += 1
        n = st["logged"]
        if n <= 16 or n % 64 == 0:
            logger.info("PDFLIP D-HOL-OVERTAKE head=%s (blocked %d pass(es), waits for KV) admitted past it: %s "
                        "(n=%d)", self.head[:16], st["passes"], [r[:16] for r in rids[:6]], n)


def note_head_admitted(sched: Any, rid: str) -> None:
    """The head ran: its guard state goes."""
    st = getattr(sched, "__dict__", {}).get("_pdflip_hol")
    if st and st.get("rid") == str(rid):
        st.update(rid=None, passes=0)
