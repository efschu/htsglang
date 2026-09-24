"""Per-lane turns for the wake side's host/IPC lanes (H11, fnFL2x83-x105).

A host or on-card IPC lane (``c<card>``, a ``p<k>`` without a BAR1 window)
carries its tags as one byte stream: the collector reads the lane's slot
counter, runs the tag's units and advances the counter, so two tags on the
SAME lane must not overlap (xsn369/370/371). The gate that enforced this
waited for EVERY earlier tag of the wake order to be collected, whatever lane
it rode. Under the round-robin pause order (587a0de31e) the earlier tags of
another source card sit between two tags of the same lane, so a lane waited
for bytes that never touch it -- measured D TP0 ``WEG2-TAG-GATE`` 790-889 ms
per P->D flip on x83/x87/x104/x105, and through the depth-2 drain rule PP0's
next deposit on ``c0`` waited with it.

A turn is what the BAR1 lanes already have (``Bar1Lanes.register_turns``):
the wake loop's main thread registers every tag, in tag order, on every lane
before submitting its collect; the collect releases the lanes it does not use
the moment it knows them and each used lane when that lane's run is over; a
lane runs a tag only when no earlier registered tag is still pending on it.
Order on a lane is the only thing the stream needs, so this is the gate's
exact requirement and nothing more.
"""
from __future__ import annotations

import threading
import time
from typing import Iterable, Optional, Tuple


class LaneTurns:
    """The pending tag indices per lane, oldest runs first."""

    def __init__(self, lanes: Iterable[str]):
        self._pending = {str(lk): set() for lk in lanes}
        self._cv = threading.Condition()

    def register(self, index: int) -> None:
        """Main thread, in tag order: the tag is pending on every lane."""
        with self._cv:
            for pend in self._pending.values():
                pend.add(int(index))
            self._cv.notify_all()

    def release(self, index: int, used: Optional[Iterable[str]] = None) -> None:
        """Drop the tag from the lanes it does not use (``used`` given) or
        from every lane (its collect is over, whatever happened)."""
        keep = None if used is None else {str(u) for u in used}
        with self._cv:
            for lk, pend in self._pending.items():
                if keep is None or lk not in keep:
                    pend.discard(int(index))
            self._cv.notify_all()

    def leave(self, lane: str, index: int) -> None:
        """The tag's run on this lane is over."""
        with self._cv:
            pend = self._pending.get(str(lane))
            if pend is not None:
                pend.discard(int(index))
            self._cv.notify_all()

    def take(self, lane: str, index: int, timeout_s: float) -> Tuple[bool, Tuple[int, ...]]:
        """Wait until no EARLIER registered tag is pending on ``lane``.

        Returns ``(True, waited_for)`` -- the earlier indices that were
        pending when the wait began, empty when there was nothing to wait
        for -- or ``(False, still_pending)`` when ``timeout_s`` ran out. A
        lane or tag that was never registered passes at once."""
        idx = int(index)
        deadline = time.monotonic() + float(timeout_s)
        with self._cv:
            pend = self._pending.get(str(lane))
            if pend is None:
                return True, ()
            first = tuple(sorted(j for j in pend if j < idx))
            while True:
                earlier = [j for j in pend if j < idx]
                if not earlier:
                    return True, first
                left = deadline - time.monotonic()
                if left <= 0:
                    return False, tuple(sorted(earlier))
                self._cv.wait(min(left, 0.5))
