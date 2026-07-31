"""Deferred-read device clock for one lane (#274 slice D4, task #284).

The lane already reports two times per forward: wall time around the call and
device time from a CUDA event pair on its own stream.  Their difference is the
documented discriminator for a degradation (``_timed_forward``): device time is
what the lane's kernels took *including* the SM share they lost to the serving
group, wall time additionally contains everything the lane waited for before
those kernels could run.  What was missing is a way to carry that pair as a
MONOTONE per-window quantity, cheaply enough to leave switched on.

Two rules shape the whole module.

1. READ THE EVENTS DEFERRED.  ``elapsed_time`` on a pair that has not retired
   blocks, and a blocking read inside the lane's round would serialise the lane
   against its own kernels and then measure the result.  Events are recorded in
   the hot path and read on a later call, once ``query()`` says the pair has
   retired.  ``harvest()`` is the only place a number is produced, it is O(the
   pairs that have completed), and it never blocks -- except at the ring cap,
   where it prefers a named, counted, bounded block over silently dropping
   device time out of the accounting.

2. NO ``import torch`` AT MODULE LEVEL.  The clock is fed a stream and an event
   factory; the CUDA types are duck-typed.  That is what makes the accounting
   testable on a machine with no GPU, which is where every rule in it can be
   checked and where its arithmetic went wrong before.

The counters it maintains are monotone by construction, because that is what
:mod:`sglang.srt.model_executor.lane_share` differences per window:

``device_ms_total``
    Sum over retired spans of the lane's own device time.
``busy_wall_ms_total``
    Wall time the lane spent holding work.  ``device/busy_wall`` is the
    submission efficiency, ``busy_wall/wall`` is the duty cycle, and
    ``device/wall`` is the lane's actual share of the card.
``spans_total`` / ``pending`` / ``forced_reads``
    The instrument's own health, so a window can say whether its device time
    is complete rather than assuming it.
"""

from __future__ import annotations

import contextlib
import dataclasses
import threading
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from collections import deque

__all__ = ["LaneDeviceClock", "DeviceClockSnapshot"]

# Above this many un-harvested pairs the clock stops trusting the deferred read
# to catch up on its own.  Reached only if nothing calls ``harvest()`` for
# hundreds of forwards, which is itself worth knowing; the response is a bounded
# synchronize on the OLDEST pair, counted in ``forced_reads``, never a drop.
DEFAULT_MAX_PENDING: int = 256


@dataclasses.dataclass(frozen=True)
class DeviceClockSnapshot:
    """The monotone counters, as the meter consumes them."""

    device_ms: float
    busy_wall_ms: float
    spans: int
    pending: int
    forced_reads: int

    def to_counters(self) -> Dict[str, float]:
        return {
            "device_ms": self.device_ms,
            "busy_wall_ms": self.busy_wall_ms,
        }

    def to_json(self) -> Dict[str, object]:
        return {
            "device_ms": round(self.device_ms, 3),
            "busy_wall_ms": round(self.busy_wall_ms, 3),
            "spans": self.spans,
            "pending": self.pending,
            "forced_reads": self.forced_reads,
        }


class LaneDeviceClock:
    """Accumulates a lane's device time from event pairs read after the fact.

    ``event_factory`` returns a timing-enabled CUDA event; ``stream`` is the
    stream the lane submits on, or ``None`` for the serial path, where the lane
    owns the current stream for the duration of its tick and recording on it is
    the same measurement.
    """

    def __init__(
        self,
        event_factory: Callable[[], Any],
        stream: Any = None,
        *,
        clock: Optional[Callable[[], float]] = None,
        max_pending: int = DEFAULT_MAX_PENDING,
    ) -> None:
        self._event_factory = event_factory
        self._stream = stream
        if clock is None:
            import time as _time

            clock = _time.perf_counter
        self._clock = clock
        self._max_pending = int(max_pending)

        self._lock = threading.Lock()
        self._pending: Deque[Tuple[Any, Any]] = deque()
        self._free: List[Any] = []
        self.device_ms_total: float = 0.0
        self.busy_wall_ms_total: float = 0.0
        self.spans_total: int = 0
        self.forced_reads: int = 0
        self._busy_since: Optional[float] = None

    def bind_stream(self, stream: Any) -> None:
        """Point the clock at the stream the lane submits on.

        Called once, when the concurrent worker creates that stream.  Pairs
        already in flight were recorded on the previous stream and stay valid:
        an event's timestamp does not depend on which stream reads it.
        """
        self._stream = stream

    # -- span recording ---------------------------------------------------

    def _event(self) -> Any:
        if self._free:
            return self._free.pop()
        return self._event_factory()

    def _record(self, ev: Any) -> None:
        # torch's Event.record() takes the stream, or the current one when
        # called with no argument.  Both spellings appear on the lane's paths.
        if self._stream is None:
            ev.record()
        else:
            ev.record(self._stream)

    @contextlib.contextmanager
    def span(self):
        """Bracket one lane forward with a recorded event pair.

        Costs two event records and one deque append.  It does NOT read the
        pair: that happens in :meth:`harvest`, by rule 1.
        """
        start = self._event()
        self._record(start)
        try:
            yield
        finally:
            end = self._event()
            self._record(end)
            with self._lock:
                self._pending.append((start, end))
                self.spans_total += 1
            self.harvest()

    def add_device_ms(self, ms: Optional[float]) -> None:
        """Fold in device time that was already measured elsewhere.

        The lane's verify and decode forwards carry their OWN event pair and
        synchronize the lane stream for the sampled token anyway, so their
        device time is already on the table; recording a second pair around
        them would pay for the same number twice.  The head's forwards have no
        such pair, which is what :meth:`span` is for.
        """
        if ms is None:
            return
        with self._lock:
            self.device_ms_total += float(ms)
            self.spans_total += 1

    # -- busy wall --------------------------------------------------------

    def mark_busy(self) -> None:
        """The lane now holds work.  Idempotent: a second call does not restart
        the interval, so a re-entrant caller cannot lose the time in front of
        it."""
        with self._lock:
            if self._busy_since is None:
                self._busy_since = self._clock()

    def mark_idle(self) -> None:
        """The lane has nothing left.  Closes the open busy interval, if any."""
        with self._lock:
            if self._busy_since is None:
                return
            self.busy_wall_ms_total += (self._clock() - self._busy_since) * 1000.0
            self._busy_since = None

    # -- deferred read ----------------------------------------------------

    def harvest(self, *, budget: int = 64) -> int:
        """Fold every RETIRED pair at the head of the queue into the counters.

        Returns how many pairs were folded.  Never blocks below the ring cap:
        one ``query()`` per candidate, stopping at the first pair still in
        flight, because pairs on one stream retire in the order they were
        recorded and a later completion behind an earlier in-flight pair is not
        possible.
        """
        folded = 0
        while folded < budget:
            with self._lock:
                if not self._pending:
                    return folded
                start, end = self._pending[0]
                over_cap = len(self._pending) > self._max_pending
            if not over_cap and not _retired(end):
                return folded
            if over_cap and not _retired(end):
                # Bounded, named and counted: the alternative is to drop the
                # span, which would make device_ms silently under-report
                # exactly when the lane is busiest.
                end.synchronize()
                with self._lock:
                    self.forced_reads += 1
            ms = float(start.elapsed_time(end))
            with self._lock:
                self._pending.popleft()
                self.device_ms_total += ms
                self._free.append(start)
                self._free.append(end)
            folded += 1
        return folded

    def drain(self) -> int:
        """Harvest everything, blocking on whatever is still in flight.

        For window boundaries in an offline driver, never for the hot path.
        """
        folded = 0
        while True:
            with self._lock:
                if not self._pending:
                    return folded
                start, end = self._pending[0]
            end.synchronize()
            ms = float(start.elapsed_time(end))
            with self._lock:
                self._pending.popleft()
                self.device_ms_total += ms
                self._free.append(start)
                self._free.append(end)
            folded += 1

    # -- readout ----------------------------------------------------------

    def snapshot(self) -> DeviceClockSnapshot:
        """The counters as they stand.

        The OPEN busy interval is included, so a lane that has been busy for
        the whole window does not report zero busy time until it goes idle.
        ``device_ms`` deliberately is NOT extrapolated the same way -- an
        unretired span has no measured duration, and inventing one is how an
        instrument starts reporting the thing it is supposed to detect.
        """
        with self._lock:
            busy = self.busy_wall_ms_total
            if self._busy_since is not None:
                busy += (self._clock() - self._busy_since) * 1000.0
            return DeviceClockSnapshot(
                device_ms=self.device_ms_total,
                busy_wall_ms=busy,
                spans=self.spans_total,
                pending=len(self._pending),
                forced_reads=self.forced_reads,
            )


def _retired(event: Any) -> bool:
    try:
        return bool(event.query())
    except Exception:  # pragma: no cover - a driver error is not this module's
        return True
