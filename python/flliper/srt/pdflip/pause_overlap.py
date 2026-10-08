"""PAUSE-OVERLAP (30.09., NF y4h/y4i): the sleeper's pause of tag t runs on a
worker while the loop deposits tag t+1.

THE MEASUREMENT (D->P, the D sleep leg is the flip's critical path: D-TP1 in
37/39 flips on y4h, D-TP2 in 23/31 on y4i). Per tag the loop runs
``deposit(t) -> sync -> pause(t) -> credit(t)`` in series. On the two 3080
D ranks ``pause_ms`` is ~26-28 ms per ~1 GiB tag (median; 5090: 6 ms), i.e.
465-517 ms of every ~1.7 s leg, with ``sync_ms = 0`` on EVERY tag -- the
device is idle when the pause starts, so the cost is not a device backlog
but the unmap/release work itself (the P ranks pause a third of that per
byte on the SAME cards; D's expert banks are H95c span maps, one handle per
lattice cell). The pause cannot be made cheaper from Python; it can leave
the chain.

THE FORM (switch ``FLLIPER_PDFLIP_ENABLE_SLEEP_PAUSE_OVERLAP``, default off =
the per-tag chain byte for byte; groups ``FLLIPER_PDFLIP_SLEEP_PAUSE_OVERLAP_GROUPS``,
default ``D``):

* the law per tag stays: DEPOSIT(t), then PAUSE(t), then CREDIT(t) -- the
  worker runs pause and credit of ONE tag in that order, and only after the
  loop's deposit of that tag returned;
* at most ONE pause is in flight: the next submit joins the previous one, so
  pauses and credits keep the tag order;
* BEFORE A DEPOSIT ON THE ON-CARD (diagonal) LANE the loop joins the pending
  pause: that lane's staging lives on this card and the co-located waker's
  claim reads this card's free memory, so both see exactly what the chain
  gave them (the reason H111b keeps the diagonal lane in place -- moving it
  ahead of a pause would be a reserve);
* the leg joins its last pause before it returns; a pause that raised is
  re-raised on the loop thread at the next join, by its own type.

NO NEW WAIT ON A CYCLE PATH: the pause only waits for this process's own
device work (cuMemUnmap's implicit sync) and the saver's mutex; the lanes
wait host-side (bar1 flags, sockets), so no device work of this process ever
waits for a peer, and the only rank that reads this card's credit (the
co-located waker) is served by the join before its lane. No VRAM is held
beyond what the chain held -- the release comes at most one deposit later.
"""

from __future__ import annotations

import concurrent.futures as _cf
import time
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

LINE = "PDFLIP-PAUSE-OVERLAP"


def overlap_on(group: Optional[str]) -> bool:
    """Armed for this sleeping ``group`` (None = the switch alone)."""
    from flliper.srt.environ import envs

    if not bool(envs.FLLIPER_PDFLIP_ENABLE_SLEEP_PAUSE_OVERLAP.get()):
        return False
    if group is None:
        return True
    groups = {g.strip() for g in str(envs.FLLIPER_PDFLIP_SLEEP_PAUSE_OVERLAP_GROUPS.get() or "").split(",")}
    return str(group) in groups


class PauseOverlap:
    """One worker, at most one pause in flight, joins named by the loop."""

    def __init__(self, diag_tags: Iterable[str], thread_init: Optional[Callable[[], None]] = None,
                 clock: Callable[[], float] = time.perf_counter):
        self.diag_tags: Set[str] = set(diag_tags)
        self._clock = clock
        self._ex = _cf.ThreadPoolExecutor(max_workers=1, thread_name_prefix="pdflip-pause",
                                          initializer=thread_init)
        self._pending: Optional[_cf.Future] = None
        self._pending_tag: Optional[str] = None
        self.submitted = 0
        self.overlapped = 0     # pauses that ran beside a following deposit
        self.join_wait_ms = 0.0  # loop time spent waiting in joins
        self.joins: List[Tuple[str, str, float]] = []  # (reason, tag, waited ms)

    @property
    def pending_tag(self) -> Optional[str]:
        return self._pending_tag

    def join(self, reason: str = "join") -> float:
        """Wait for the pending pause; re-raise its exception. Returns ms waited."""
        fut, tag = self._pending, self._pending_tag
        if fut is None:
            return 0.0
        t0 = self._clock()
        try:
            fut.result()
        finally:
            self._pending, self._pending_tag = None, None
            waited = (self._clock() - t0) * 1000.0
            self.join_wait_ms += waited
            self.joins.append((reason, str(tag), waited))
        return waited

    def before_deposit(self, tag: str) -> float:
        """Called by the loop before depositing ``tag``: joins the pending
        pause when ``tag`` has an on-card lane on this rank, else counts the
        pending pause as overlapped with this deposit."""
        if self._pending is None:
            return 0.0
        if tag in self.diag_tags:
            return self.join("diag")
        self.overlapped += 1
        return 0.0

    def submit(self, tag: str, step: Callable[[], None]) -> None:
        """Run ``step`` (pause + credit + its line) for ``tag`` on the worker,
        after the previous tag's step has finished."""
        self.join("order")
        self._pending = self._ex.submit(step)
        self._pending_tag = tag
        self.submitted += 1

    def close(self) -> None:
        """Join the last pause (re-raising) and stop the worker."""
        try:
            self.join("leg-end")
        finally:
            self._ex.shutdown(wait=True)

    def summary(self) -> str:
        by: Dict[str, float] = {}
        for reason, _tag, ms in self.joins:
            by[reason] = by.get(reason, 0.0) + ms
        return ("%s tags=%d overlapped=%d join_wait_ms=%.0f (%s) -- pause(t)+credit(t) on a "
                "worker beside deposit(t+1); joined before every on-card-lane deposit and "
                "at the leg end" % (LINE, self.submitted, self.overlapped, self.join_wait_ms,
                                    " ".join("%s=%.0f" % kv for kv in sorted(by.items())) or "-"))


def chain_end_ms(steps: Sequence[Tuple[str, float, float]], diag_tags: Iterable[str],
                 start_ms: float = 0.0, overlap: bool = True) -> Tuple[float, Dict[str, float]]:
    """The sleeper's leg end under the rule above, from measured per-tag costs.

    ``steps`` = ``(tag, rest_ms, pause_ms)`` in the leg's order, where
    ``rest_ms`` is everything of the tag that stays on the loop (deposit, sync,
    gap -- the measured ``total_ms - pause_ms``) and ``pause_ms`` the pause
    plus its credit. Returns ``(leg_end_ms, {tag: deposit_end_ms})``; the
    deposit end is when a waker's collect of that tag can complete. Assumes a
    pause's duration does not change beside a deposit (the upper bound of the
    gain; the PDFLIP-PAUSE-SUB and PDFLIP-PAUSE-OVERLAP lines measure it)."""
    diag = set(diag_tags)
    t = float(start_ms)
    pend_end: Optional[float] = None
    dep_end: Dict[str, float] = {}
    for tag, rest, pause in steps:
        if overlap and pend_end is not None and tag in diag:
            t = max(t, pend_end)
            pend_end = None
        t += float(rest)
        dep_end[tag] = t
        if not overlap:
            t += float(pause)
            continue
        if pend_end is not None:     # submit joins the previous pause
            t = max(t, pend_end)
        pend_end = t + float(pause)
    if pend_end is not None:
        t = max(t, pend_end)
    return t, dep_end
