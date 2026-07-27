# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Split a per-rank forward span into compute time and collective time.

The per-rank prefill line reports one ``gpu-ms`` number: the CUDA-event span
of the prefill forward. Under tensor parallelism that span is the same on
every rank, because the ranks run lock-step: whoever finishes a layer shard
first blocks in that layer's collective until the slowest rank arrives. The
single number therefore cannot distinguish "this rank computes a lot" from
"this rank waits a lot", which is exactly the question an uneven-TP shard
split raises.

This module accounts the device time spent *inside* collectives during an
armed region. ``wait`` is that sum; ``compute`` is the remainder of the
forward span. ``wait`` includes the transfer itself, not only the rendezvous
— on a homogeneous link the transfer part is equal across ranks, so the
*spread* of ``wait`` across ranks is the shard-imbalance signal.

Mechanics, matching the DeviceTimer discipline:

- Two CUDA events per collective, recorded on the current stream. No host
  sync anywhere on the hot path; durations are read deferred, once the
  enclosing forward's end event has completed.
- Events are pooled and re-recorded, so an armed forward with ~100
  collectives does not allocate ~200 event objects per chunk.
- Off by default: the clock is armed only around the forwards that the
  per-rank prefill log already times (plain prefill). Decode and every
  other collective in the process pay one attribute read.
- Collectives issued while a CUDA graph is capturing are skipped: a timing
  event cannot be recorded into a capture. Collectives inside a *replayed*
  graph are invisible for the same reason the Python body does not run, so
  a graph-covered forward reports no split at all rather than a wrong zero
  (see ``Slot.graph_capture_skipped``).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import List, Optional, Tuple

import torch

__all__ = ["CollectiveClock", "Slot", "collective_clock"]


class Slot:
    """Collective event pairs recorded during one armed region."""

    __slots__ = ("pairs", "graph_capture_skipped")

    def __init__(self) -> None:
        self.pairs: List[Tuple[torch.cuda.Event, torch.cuda.Event]] = []
        # True if at least one collective ran while a graph was capturing and
        # could therefore not be timed. Makes the slot's total a lower bound.
        self.graph_capture_skipped = False


class CollectiveClock:
    def __init__(self) -> None:
        self._slot: Optional[Slot] = None
        self._pool: List[torch.cuda.Event] = []

    # -- arming ---------------------------------------------------------

    @property
    def armed(self) -> bool:
        return self._slot is not None

    def arm(self) -> None:
        self._slot = Slot()

    def disarm(self) -> Optional[Slot]:
        slot, self._slot = self._slot, None
        return slot

    # -- recording ------------------------------------------------------

    @contextmanager
    def span(self):
        """Time one collective. Caller must have checked ``armed`` first."""
        slot = self._slot
        if slot is None:
            yield
            return
        if torch.cuda.is_current_stream_capturing():
            slot.graph_capture_skipped = True
            yield
            return
        # Disarm for the duration of the body so that a collective built out
        # of other collectives is counted once, not once per level.
        self._slot = None
        start = self._acquire()
        start.record()
        try:
            yield
        finally:
            end = self._acquire()
            end.record()
            slot.pairs.append((start, end))
            self._slot = slot

    def _acquire(self) -> torch.cuda.Event:
        if self._pool:
            return self._pool.pop()
        return torch.cuda.Event(enable_timing=True)

    # -- harvesting -----------------------------------------------------

    def harvest(self, slot: Optional[Slot]) -> Optional[float]:
        """Seconds spent in collectives, or None if the slot is not ready.

        Query-only: never synchronizes. The caller is expected to harvest
        only after the enclosing forward's end event has completed, in which
        case every pair here is already complete and this returns a number.
        """
        if slot is None:
            return None
        if not slot.pairs:
            return 0.0
        for _, end in slot.pairs:
            if not end.query():
                return None
        total_ms = 0.0
        for start, end in slot.pairs:
            total_ms += start.elapsed_time(end)
            self._pool.append(start)
            self._pool.append(end)
        slot.pairs.clear()
        return total_ms / 1000.0


_CLOCK = CollectiveClock()


def collective_clock() -> CollectiveClock:
    return _CLOCK
