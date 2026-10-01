"""HANDBACK (01.10., user via 27B: "den anker fix koennen alle brauchen"): what
group D holds and computes after a P hand-off -- the NF side of 27B fe5c55041b.

NF's contract is not 27B's N-1. P computes all N prompt tokens, publishes the
END section (rows [page_prefix, N) + the state after N + its sampled token,
H24 E2 / H63 fold) and D takes it without a target forward: d_prefix = N,
d_compute = 0, the END key checked against the prompt (``tail_key``) and the
rows by digest. Measured on the last 12 NF boots (D TP0, exact rid pairing):
492 of 565 hand-offs (87 %) ran exactly that (TAIL-READY adopt=done, SKIP-EXTEND
prefix=N). The rest fell back to an extend from the store's page anchor
(floor_page(N-2), the CLAIM ANCHOR P files its recurrent state at): 36 with no
tail at all (park/resume shapes), 25 ``no_parts``, 10 ``end_only:batch_not_empty``
(the END state dropped in a batch that already runs a forward), 4 park windows.

So 27B's other half does NOT come to NF:
  * the D claim stays the upstream N-1 RAW tokens (= N-2 bigram units) --
    P's CLAIM ANCHOR (floor_page(N-2)) and D's claim are one geometry, and the
    exact keying + E2 already deliver the state after N. Widening the claim on
    an exact-bigram D (27B ``handback_bigram_claim``) would move D's match
    without moving P's anchor;
  * the L3 store identity stays byte for byte (NF has run the exact keying
    since fe031f2c80; its directory holds exact anchors already).

What does come: a hand-off read is read whatever its length (below).
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

HANDBACK_MARK = "WEG2-HANDBACK"
_N = [0]


def handback_min_tokens(has_handoff: bool = False, env=None):
    """1 for a P hand-off read on a group-D rank -- a request carrying P's
    #1442 hand-off chain, or any D request of the dual layout -- whatever its
    length; else None (the tree's #915 threshold, unchanged for D's own reads).

    Metal NF (12 boots, D TP0): 33x '#915 PREFETCH REFUSED reason=vote_negative
    need=64..255 keys=handoff' -- pages P had written and handed over, refused as
    "too short" by the 256-token threshold, and D prefilled them again. The
    threshold prices whether opening a FRESH read pays; a hand-off is not fresh,
    P's pages are the request's own prefix (27B fe5c55041b, NF y4a TS for the
    told read: same reasoning, ``weg2_store_told.told_read_min_tokens``)."""
    e = os.environ if env is None else env
    if (e.get("SGLANG_WEG2_GROUP", "") or "").strip().upper() != "D":
        return None
    if has_handoff or (e.get("SGLANG_WEG2_DUAL_LAYOUT", "") or "").strip() == "1":
        return 1
    return None


def handback_line(rid, n_tokens: int, d_prefix: int, d_compute: int, path: str) -> str:
    """One line per hand-off at D's admission: what D holds (``d_prefix``) and
    what its target forward still computes (``d_compute``), by path:
    ``skip`` (E2: P's END state + token, contract d_prefix = N, d_compute = 0),
    ``e1`` (rows to c, extend [c, N)), ``extend:<why>`` (the tail refused:
    today's extend from the page anchor)."""
    _N[0] += 1
    line = (
        f"{HANDBACK_MARK} rid={str(rid)[:24]} N={int(n_tokens)} d_prefix={int(d_prefix)} "
        f"d_compute={int(d_compute)} path={path} (contract NF: E2 d_prefix = N, d_compute = 0; "
        f"else d_compute = N - d_prefix) n={_N[0]}"
    )
    logger.info(line)
    return line
