"""E2 in a mixed wake cohort: the hand-offs that take P's END state first.

Bestform x178 (25.09.): every P->D wake held only hand-offs, all of them
took P's END state (H24 E2, H24c: a wake's skips share one pass), the pass
ran no target forward (POST-WAKE run_ms ~30) and the first decode token
came ~2.1 s after P's end. Under agent load (y3m 09292136) a wake also
brings requests that need a real extend -- flip-parked resumes, D-direct
arrivals with 200-600 fresh tokens -- and two things broke E2:

1. ORDER. ``order_waiting`` puts the parked requests first; the adder took
   their extend, and every hand-off behind them met a non-empty batch:
   ``adopt=skipped:end_only:batch_not_empty``, its END state dropped, and the
   whole cohort ran ONE extend (ep8 21:43:24: 903 new tokens over 147904
   cached, 3.56 s) before any first token.
2. THE PASS AFTER THE SKIP. When the skips did run first (ep6 21:42:25,
   weg2-4-13, run_ms 29), the very next pass admitted the real extend
   (611 tokens, 2.5 s); the skip batch's result -- P's token -- was processed
   only after that extend, so its first token still waited 3 s.

Both are repaired without new mechanism: the END-state requests go to the
head of the pass (their seats are the wake's, ``note_wake_seats`` counts
hand-offs and parked alike), and the pass right after a skip batch runs the
decode round before any new extend. Every input is rank-uniform: the skip
verdict is the group's agreed vote (``tail_adopt.skip_joinable``), the last
batch is the same on every rank.
"""

from __future__ import annotations

import logging
from typing import Callable, FrozenSet, List, Sequence, Tuple

logger = logging.getLogger(__name__)

ORDER_MARK = "WEG2-SKIP-FIRST ORDER"
HOLD_MARK = "WEG2-SKIP-FIRST DECODE-HOLD"
_N = {"order": 0, "hold": 0}


def order(waiting: Sequence, joinable: Callable) -> Tuple[List, FrozenSet[str]]:
    """The queue with every request ``joinable`` says takes the END state
    moved to the head (stable on both sides), and their rids. The queue is
    returned unchanged when none does or when they already lead."""
    skips = [r for r in waiting if joinable(r)]
    if not skips:
        return list(waiting), frozenset()
    rest = [r for r in waiting if not any(r is s for s in skips)]
    rids = frozenset(str(r.rid) for r in skips)
    moved = [r for r in waiting[: len(skips)] if not any(r is s for s in skips)]
    if moved:
        _N["order"] += 1
        n = _N["order"]
        if n <= 20 or n % 200 == 0:
            logger.info("%s n=%d skips=%s ahead_of=%s -- the END-state hand-offs take their "
                        "pass before any request that needs a forward",
                        ORDER_MARK, n, sorted(rids)[:8], [str(r.rid) for r in moved][:8])
    return skips + rest, rids


def hold_prefill_after_skip(last_batch, running_batch) -> bool:
    """True once for the pass right after a skip-extend batch that left
    requests running: that pass runs their decode round (it delivers P's
    token and the first decode token) before any new extend. The mark is
    consumed, so an aliased running batch never holds twice."""
    if last_batch is None or not getattr(last_batch, "weg2_skip_extend", False):
        return False
    last_batch.weg2_skip_extend = False
    mode = getattr(last_batch, "forward_mode", None)
    if mode is None or not mode.is_extend():
        return False  # the batch already ran as a decode (aliased running batch)
    if running_batch is None or running_batch.is_empty():
        return False
    _N["hold"] += 1
    n = _N["hold"]
    if n <= 20 or n % 200 == 0:
        logger.info("%s n=%d running=%d -- the skip batch's first tokens go out with this "
                    "decode round, the next extend waits one pass", HOLD_MARK, n,
                    len(running_batch.reqs))
    return True
