"""DQH: a refused D queue head is re-evaluated on a state change, not every round.

NF int18, 08.10. (D log boot_weg2_dkrnfint4h6ablxcbar1dauer10081149_e69f28a7f6_
1008_115011.D.log, TP0): pdflip-28-97 stood at the head of D's queue 12:12:54-
12:17:30 and was refused by the KV gate (``H105d FORM-A-CUT LOAD-BACK ROOM``,
NO_TOKEN) in EVERY scheduler round -- 1245 ``PDFLIP X-GATE ... verdict=admit`` with
its matches, ~4.3 ``#1427 ARENA-DROP freed=0`` and one ``EVICT-FRONTIER-CENSUS``
per decode round. Measured per decode round (Decode rank batch ``t:`` deltas
minus ``gpu-ms``): host gap 5-9 ms at 12:11 (no head), 124-129 ms at 12:14-12:16;
rounds with the evaluation 103.6 ms, rounds without it -3.2 ms (12:12:54-12:13:14).

THE RULE (group D, Form A token cut -- NF only): after a pass that built nothing
while the HOL head (``pdflip/hol_overtake.py``, the pass's first NO_TOKEN) waits,
the next passes are not built as long as nothing the verdict reads has changed --
the head, the waiting set, the running set, the landed prefetches, no chunked
continuation -- and at the latest every ``REEVAL_EVERY_PASSES`` passes (the
rank-local room: host arena, write-through acks, evictions). Every input is
replicated and the pass count is the same on every rank, so the group holds and
evaluates on the same iterations (the elif chain of ``get_next_batch_to_run``
keeps the budget site's collectives untouched). The held head stays named: a
throttled ``PDFLIP D-HEAD-HOLD`` line. Nothing is given up -- the head is evaluated
again at the next change and at least every ``REEVAL_EVERY_PASSES`` passes."""

from __future__ import annotations

import logging
from typing import Any, Callable, Iterable, Optional

import msgspec

logger = logging.getLogger(__name__)

#: a held head is evaluated again at the latest after this many held passes
REEVAL_EVERY_PASSES = 16


class HeadHoldState(msgspec.Struct):
    """The armed fingerprint (None = not armed), the held passes since the last
    evaluation, and the totals for the status line."""

    fingerprint: Optional[tuple] = None
    held: int = 0
    held_total: int = 0
    log_n: int = 0


def fingerprint(
    *,
    hol_state: Optional[dict],
    waiting: Iterable[Any],
    running: Iterable[Any],
    prefetch_verdicts: Optional[dict],
    continuation: bool,
) -> Optional[tuple]:
    """What the head's verdict reads, as replicated values; None when no head
    can be held (no HOL head, it left the queue, a chunked continuation runs)."""
    if continuation or not hol_state:
        return None
    head = hol_state.get("rid")
    waiting_rids = tuple(sorted(str(q.rid) for q in waiting))
    if head is None or str(head) not in waiting_rids:
        return None
    landed = tuple(sorted(str(k) for k, v in (prefetch_verdicts or {}).items() if v))
    running_rids = tuple(sorted(str(r.rid) for r in running))
    return (str(head), waiting_rids, running_rids, landed)


class DHeadHold:
    """One per scheduler. ``group_d`` is fixed at construction; ``token_cut`` is
    read per call (the Form A plan is installed after the scheduler is built) --
    both group-uniform."""

    def __init__(self, *, group_d: bool, token_cut: Callable[[], bool]):
        self.group_d = bool(group_d)
        self.token_cut = token_cut
        self.state = HeadHoldState()

    @property
    def enabled(self) -> bool:
        return self.group_d and bool(self.token_cut())

    def should_hold(self, fp: Optional[tuple]) -> bool:
        """True = do not build this pass (nothing the verdict reads changed)."""
        st = self.state
        if fp is None or st.fingerprint is None or not self.enabled:
            return False
        if fp != st.fingerprint or st.held >= REEVAL_EVERY_PASSES:
            st.fingerprint = None
            return False
        st.held += 1
        st.held_total += 1
        self._status(fp)
        return True

    def note_pass(self, *, fp: Optional[tuple], built_nothing: bool) -> None:
        """After an evaluated pass: arm on a refused head, disarm otherwise."""
        st = self.state
        st.fingerprint = fp if (self.enabled and built_nothing and fp is not None) else None
        st.held = 0

    def _status(self, fp: tuple) -> None:
        st = self.state
        st.log_n += 1
        n = st.log_n
        if n <= 4 or (n & (n - 1)) == 0:
            logger.info(
                "PDFLIP D-HEAD-HOLD head=%s held_total=%d waiting=%d running=%d (n=%d): the "
                "refused head is evaluated again when the head, the waiting or running set or "
                "a landed prefetch changes, and at the latest every %d passes -- not in every "
                "decode round", fp[0], st.held_total, len(fp[1]), len(fp[2]), n,
                REEVAL_EVERY_PASSES,
            )


def build(*, group: str) -> DHeadHold:
    from flliper.srt.rank_role import form_a_token_cut_active

    return DHeadHold(group_d=str(group).strip().upper() == "D", token_cut=form_a_token_cut_active)
