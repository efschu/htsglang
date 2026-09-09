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

import dataclasses
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import torch

__all__ = [
    "ClockBackend",
    "CollectiveClock",
    "FamilyStat",
    "HarvestResult",
    "RoundSpan",
    "RoundResult",
    "Slot",
    "TorchCudaBackend",
    "collective_clock",
    "OTHER",
]

#: Family of a span whose caller said nothing. Never an error: an unlabeled
#: collective is still counted in the grand total, it just lands here.
OTHER = "other"


#: #1241. Separator between a forward's PHASE prefix and the collective family
#: the dispatch site named. ``spec_verify:tp.all_reduce`` is the tp all-reduce
#: of a speculative VERIFY forward; ``tp.all_reduce`` without a prefix is the
#: plain decode/prefill one. Kept as a prefix rather than as a second axis so
#: the existing one-dimensional ``wait by family`` reader (and its regex,
#: debug_utils/rank_phase_summary.py:34, ``[A-Za-z0-9_.]+``) parses both --
#: hence a character that regex already accepts is NOT usable, and ':' is
#: deliberately outside it so an old reader drops the prefixed families rather
#: than silently merging them into the unprefixed ones.
FAMILY_PHASE_SEP = ":"


class ClockBackend:
    """The whole of this module's contact with CUDA, in one place.

    Two calls, both on the hot path, both replaceable in a test: create a
    timing event, and ask whether the current stream is capturing a graph.
    Everything else here is arithmetic over what those two return.

    The point is not abstraction for its own sake -- it is that the decode
    round clock (#1241) has to be exercised WITHOUT a GPU, and the only
    honest way to do that is to move the device behind one seam instead of
    letting each test monkeypatch ``torch.cuda`` differently.
    """

    def event(self):  # pragma: no cover - trivial, overridden in tests
        raise NotImplementedError

    def is_capturing(self) -> bool:  # pragma: no cover
        raise NotImplementedError


class TorchCudaBackend(ClockBackend):
    """The real device. The default, and the only one serving ever uses."""

    def event(self):
        return torch.cuda.Event(enable_timing=True)

    def is_capturing(self) -> bool:
        return torch.cuda.is_current_stream_capturing()


@dataclasses.dataclass(frozen=True)
class FamilyStat:
    """One collective family's contribution to an armed region."""

    total_ms: float
    count: int
    #: Longest single span in the family. A family can hold the same total
    #: as many small transfers or as one long stall, and only the second is
    #: a latency problem -- the sum alone cannot tell those apart.
    max_ms: float


@dataclasses.dataclass(frozen=True)
class HarvestResult:
    total_s: float
    families: Dict[str, FamilyStat]


class Slot:
    """Collective event pairs recorded during one armed region."""

    __slots__ = ("pairs", "graph_capture_skipped")

    def __init__(self) -> None:
        # (start, end, family). The family rides along with the pair rather
        # than in a parallel list so a pair can never drift away from its
        # label when pairs are appended from different call sites.
        self.pairs: List[Tuple[torch.cuda.Event, torch.cuda.Event, str]] = []
        # True if at least one collective ran while a graph was capturing and
        # could therefore not be timed. Makes the slot's total a lower bound.
        self.graph_capture_skipped = False


@dataclasses.dataclass
class RoundSpan:
    """One decode round's bracket: the events around it and its collectives.

    The bracket is recorded on the SAME stream as the collectives inside it,
    so ``end.query()`` being true implies every pair in ``slot`` is complete
    -- the same argument ``SplitDeviceTimer`` makes for its interval, and the
    reason a round can be harvested without ever synchronizing.
    """

    start: object
    end: object = None
    slot: Optional[Slot] = None
    #: True when at least one forward of this round ran from a REPLAYED CUDA
    #: graph. The Python body of a collective inside a replay does not run,
    #: so its span is never recorded and ``slot`` would report a wait of 0.0
    #: -- a wrong zero, not a measurement. Set by the caller, honoured by
    #: :meth:`harvest_round`, which then withholds the split.
    graph_replayed: bool = False
    #: True when this round was opened while the slot was ALREADY armed by
    #: the prefill half (#252). The round then owns no slot: it brackets the
    #: device time and withholds the split, and -- decisively -- it does not
    #: touch ``_slot``, so the prefill arming it found still reaches its own
    #: ``disarm``. Overwriting instead would have stolen the shipped line's
    #: compute/wait column, which is a regression of the other half of this
    #: same instrument and is the reason the guard exists rather than an
    #: assertion in a comment.
    contended: bool = False


@dataclasses.dataclass(frozen=True)
class RoundResult:
    """One round's reading. ``compute_ms`` is ``round_ms - wait_ms``."""

    round_ms: float
    wait_ms: Optional[float]
    compute_ms: Optional[float]
    families: Optional[Dict[str, FamilyStat]]
    #: Why the split is absent, or ``None`` when it is present. Never an
    #: empty string: a reader must be able to print the reason verbatim.
    split_refused: Optional[str] = None


class CollectiveClock:
    def __init__(self, backend: Optional[ClockBackend] = None) -> None:
        self._slot: Optional[Slot] = None
        self._pool: List[torch.cuda.Event] = []
        self._label_hint: Optional[str] = None
        #: #1241. Phase prefix applied to every family recorded while it is
        #: set. Distinct from ``_label_hint``: the hint REPLACES what the
        #: dispatch site said, the prefix KEEPS it and adds who was running.
        self._phase_prefix: Optional[str] = None
        self._backend: ClockBackend = backend or TorchCudaBackend()
        #: #1241 CONTENTION, COUNTED NOT ASSUMED. Both halves of this
        #: instrument arm the one slot: ``arm``/``disarm`` for a prefill
        #: forward (#252) and ``open_round``/``close_round`` for a decode
        #: round. They are supposed to be mutually exclusive -- a batch is
        #: prefill XOR decode -- but "supposed to" is a premise about the
        #: scheduler, not a property of this object, and the failure mode of
        #: a broken premise here is SILENT: the second arming overwrites the
        #: first, the first reads an empty slot and reports ``wait 0.0``,
        #: which has the exact shape of a measurement. So neither arming
        #: overwrites the other, both count how often they had to refuse,
        #: and the refusing side withholds its split by name.
        self._arm_refusals: int = 0
        self._round_contentions: int = 0
        #: LIFO depth of armings that found the slot taken. ``disarm`` pops
        #: it instead of stealing a slot it never created.
        self._refused_arm_depth: int = 0

    # -- arming ---------------------------------------------------------

    @property
    def armed(self) -> bool:
        return self._slot is not None

    @property
    def contention_counts(self) -> Tuple[int, int]:
        """``(prefill armings refused, decode rounds opened contended)``.

        Both are ZERO on a correct boot. A non-zero pair is not a tuning
        number, it is the statement that the two halves overlapped and that
        one of the two lines is therefore carrying a withheld split -- read
        it before reading any compute/wait mean from that boot.
        """
        return (self._arm_refusals, self._round_contentions)

    def arm(self) -> None:
        if self._slot is not None:
            # A decode round already owns the slot. Refuse rather than
            # overwrite: overwriting would silently empty the round's slot.
            self._arm_refusals += 1
            self._refused_arm_depth += 1
            return
        self._slot = Slot()

    def disarm(self) -> Optional[Slot]:
        if self._refused_arm_depth > 0:
            # This disarm belongs to an arming that never took the slot.
            # Returning None makes the caller report an UNTIMED line, which
            # the prefill reporter already renders; returning the slot would
            # hand it somebody else's collectives.
            self._refused_arm_depth -= 1
            return None
        slot, self._slot = self._slot, None
        return slot

    # -- recording ------------------------------------------------------

    @contextmanager
    def label_scope(self, label: str):
        """Name the collective family for everything issued inside.

        The generic dispatch sites know the group and the operation
        (``dcp.all_gather``); only the caller knows what the transfer is FOR
        (``cp.lse_ag`` vs ``cp.all_gather_heads``). A scope therefore WINS
        over the label the dispatch site derives -- the more specific
        statement comes from whoever has the more specific knowledge.

        Keep scopes tight. An unrelated collective issued inside an open
        scope is attributed to it, which is the one way this can mislead.
        """
        prev = self._label_hint
        self._label_hint = label
        try:
            yield
        finally:
            self._label_hint = prev

    @contextmanager
    def phase_scope(self, phase: Optional[str]):
        """Name WHO is running, without overwriting WHAT was issued (#1241).

        ``label_scope`` answers "what is this transfer for" and therefore
        REPLACES the dispatch site's family. A decode round needs the other
        question answered at the same time -- a speculative verify forward
        issues the very same ``tp.all_reduce`` the plain decode forward
        issues, and the round line has to be able to say which of the two
        the wait sat in. So this PREFIXES instead of replacing:
        ``spec_verify:tp.all_reduce``.

        ``None`` is the no-op form, so a call site can pass the phase it has
        (or has not) without branching.
        """
        if phase is None:
            yield
            return
        prev = self._phase_prefix
        self._phase_prefix = str(phase)
        try:
            yield
        finally:
            self._phase_prefix = prev

    @contextmanager
    def span(self, label: Optional[str] = None):
        """Time one collective. Caller must have checked ``armed`` first.

        ``label`` is the DEFAULT family, as derived by the dispatch site. An
        active :meth:`label_scope` overrides it; if neither says anything the
        span is counted under ``other`` rather than dropped.
        """
        slot = self._slot
        if slot is None:
            yield
            return
        if self._backend.is_capturing():
            slot.graph_capture_skipped = True
            yield
            return
        family = self._label_hint or label or OTHER
        if self._phase_prefix:
            family = self._phase_prefix + FAMILY_PHASE_SEP + family
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
            slot.pairs.append((start, end, family))
            self._slot = slot

    def _acquire(self) -> torch.cuda.Event:
        if self._pool:
            return self._pool.pop()
        return self._backend.event()

    # -- harvesting -----------------------------------------------------

    def harvest_detail(self, slot: Optional[Slot]) -> Optional[HarvestResult]:
        """Grand total AND the per-family decomposition, or None if not ready.

        Query-only: never synchronizes. The caller is expected to harvest
        only after the enclosing forward's end event has completed, in which
        case every pair here is already complete and this returns a number.

        Consumes the slot: the pairs are returned to the event pool, so this
        and :meth:`harvest` are two views of ONE reading, not two readings.
        """
        if slot is None:
            return None
        if not slot.pairs:
            return HarvestResult(total_s=0.0, families={})
        for _, end, _ in slot.pairs:
            if not end.query():
                return None
        total_ms = 0.0
        acc: Dict[str, List[float]] = {}
        for start, end, family in slot.pairs:
            ms = start.elapsed_time(end)
            total_ms += ms
            slot_acc = acc.get(family)
            if slot_acc is None:
                acc[family] = [ms, 1.0, ms]
            else:
                slot_acc[0] += ms
                slot_acc[1] += 1.0
                if ms > slot_acc[2]:
                    slot_acc[2] = ms
            self._pool.append(start)
            self._pool.append(end)
        slot.pairs.clear()
        return HarvestResult(
            total_s=total_ms / 1000.0,
            families={
                name: FamilyStat(total_ms=v[0], count=int(v[1]), max_ms=v[2])
                for name, v in acc.items()
            },
        )

    # -- decode rounds (#1241) -------------------------------------------

    def open_round(self) -> RoundSpan:
        """Bracket a decode round and arm the clock for its collectives.

        The prefill half of this instrument (#252) brackets ONE forward
        through ``SplitDeviceTimer``. A decode round is not one forward --
        under speculation it is a draft extend plus a target verify, and the
        question the ladder asks ("ms per round per rank") is about the sum.
        So the bracket is opened here, at the round, and the forwards inside
        it record into one slot.

        No synchronization: two event records on the current stream.

        CONTENTION IS REFUSED, NEVER RESOLVED BY OVERWRITING. If the slot is
        already armed the prefill half (#252) is mid-forward; taking the slot
        from it would make ``SplitDeviceTimer.disarm`` return an empty slot
        and drop the SHIPPED prefill line's compute/wait column. The round is
        opened anyway -- its device time is honest either way -- with no slot
        and ``contended`` set, and :meth:`harvest_round` then withholds the
        decode split by name.
        """
        if self._slot is not None:
            self._round_contentions += 1
            start = self._acquire()
            start.record()
            return RoundSpan(start=start, slot=None, contended=True)
        slot = Slot()
        self._slot = slot
        start = self._acquire()
        start.record()
        return RoundSpan(start=start, slot=slot)

    def close_round(self, span: Optional[RoundSpan]) -> Optional[RoundSpan]:
        """Close the bracket. Still no synchronization, still no reading.

        Disarms only the slot this span actually armed. A contended span
        armed nothing, so clearing ``_slot`` here would disarm the PREFILL
        forward that owns it -- the same theft the open guards against, one
        scope later.
        """
        if span is None:
            return None
        end = self._acquire()
        end.record()
        span.end = end
        if not span.contended and self._slot is span.slot:
            self._slot = None
        return span

    def harvest_round(self, span: Optional[RoundSpan]) -> Optional[RoundResult]:
        """Read a CLOSED round, or ``None`` while its events are still in
        flight. Query-only: the caller reads round N while round N+1 runs.

        Three outcomes, and the third is the one this instrument exists to
        get right:

        * ready and observable -> ``round_ms`` with a compute/wait split;
        * ready and GRAPH-REPLAYED -> ``round_ms`` only, ``split_refused``
          naming the replay. The collectives ran; their Python bodies did
          not, so nothing recorded them. Reporting ``wait 0.0`` here would
          be a fabrication with the same shape as a measurement;
        * not ready -> ``None``, and the caller keeps the span.
        """
        if span is None or span.end is None:
            return None
        if not span.end.query():
            return None
        round_ms = span.start.elapsed_time(span.end)
        self._pool.append(span.start)
        self._pool.append(span.end)
        span.start = None
        span.end = None
        if span.contended:
            # No slot was ever armed for this round. Not a zero wait: an
            # unknown one, named so the reader can count how many rounds the
            # overlap cost and never average over them silently.
            span.slot = None
            return RoundResult(
                round_ms=round_ms,
                wait_ms=None,
                compute_ms=None,
                families=None,
                split_refused="slot-contended-with-prefill",
            )
        if span.graph_replayed:
            slot = span.slot
            span.slot = None
            # Drain the slot's events back to the pool without pricing them:
            # a partial wait over a round whose graph part is invisible is
            # not a smaller wait, it is an unknown one.
            if slot is not None:
                for st, en, _ in slot.pairs:
                    self._pool.append(st)
                    self._pool.append(en)
                slot.pairs.clear()
            return RoundResult(
                round_ms=round_ms,
                wait_ms=None,
                compute_ms=None,
                families=None,
                split_refused="graph-replay",
            )
        detail = self.harvest_detail(span.slot)
        span.slot = None
        if detail is None:
            return RoundResult(
                round_ms=round_ms,
                wait_ms=None,
                compute_ms=None,
                families=None,
                split_refused="collective-events-unread",
            )
        wait_ms = detail.total_s * 1000.0
        return RoundResult(
            round_ms=round_ms,
            wait_ms=wait_ms,
            # Clamped for the same reason the prefill line clamps: the
            # collective spans sit INSIDE the bracket, so their sum cannot
            # legitimately exceed it, but event granularity can push the
            # difference a hair below zero.
            compute_ms=max(round_ms - wait_ms, 0.0),
            families=dict(detail.families),
        )

    def harvest(self, slot: Optional[Slot]) -> Optional[float]:
        """Seconds spent in collectives, or None if the slot is not ready.

        Kept as the float-returning view of :meth:`harvest_detail` rather
        than widened in place: this signature has callers and pinned tests,
        and a decomposition is an ADDITION to what the clock reports, not a
        change to what ``wait`` means.
        """
        result = self.harvest_detail(slot)
        return None if result is None else result.total_s


_CLOCK = CollectiveClock()


def collective_clock() -> CollectiveClock:
    return _CLOCK
