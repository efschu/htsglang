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
- Collectives issued while a CUDA graph is capturing WITHOUT an armed
  capture scope are skipped, and a collective inside a *replayed* graph
  never runs its Python body, so such a forward reports no split at all
  rather than a wrong zero (see ``Slot.graph_capture_skipped``).

#1241b -- READING THE SPLIT OUT OF A REPLAYED GRAPH
--------------------------------------------------

The bullet above is why every graph-replayed decode round of boot
``weg2dec1_0909`` printed ``split unavailable: graph-replay``: 5695 rounds
per rank withheld, against 11 eager rounds whose mean was 16.6x longer.
Averaging those 11 as if they described the form is the denominator trap,
and ``--disable-cuda-graph`` is not the answer either -- it changes the
measured form, and full-perf validation is done WITH graphs and spec.

So the events move INTO the graph. Three CUDA facts carry this, and each is
a documented guarantee rather than an observation of one driver:

1. ``cudaEventRecord`` issued on a stream that is CAPTURING does not record
   a timestamp; it becomes an **event-record node** of the graph under
   construction. The node is part of the graph body and is therefore
   re-executed on **every replay**, like any other node.
2. ``cudaEventElapsedTime(pre, post)`` over two such nodes is valid once the
   replay that executed them has **completed**. Completion is established
   here the way it already is for a round: ``Event.query()``, never a
   ``synchronize()`` inside the round. A round is read one round late.
3. Events used this way must be **timing-enabled** (never
   ``cudaEventDisableTiming``) -- ``elapsed_time`` on a disable-timing event
   is an error, not a zero -- and there must be **one pair per wrapped
   region per graph**. A pair shared by two regions of the SAME graph is
   overwritten by the second region's record node on every replay, and the
   resulting span silently covers the compute between them.

Hence :meth:`CollectiveClock.capture_scope`: armed around the capture of one
graph, it makes ``span`` lay a fresh, pre-created pre/post pair into the
graph for EVERY collective occurrence, keyed by that graph. Nothing is
allocated on the replay path -- allocating an event inside a round is the
mutant this design exists to refuse, because event creation on a hot stream
is a host-side device call in the middle of the span being measured.

THE ONE THING THIS CANNOT DO, NAMED RATHER THAN DISCOVERED. The nodes belong
to the graph, not to the round: replay N+1 OVERWRITES the timestamps replay N
left. So a reading is valid only while no newer replay of that key has been
launched. :meth:`note_graph_replay` bumps a per-key generation at the launch
site and :meth:`harvest_round` refuses a round whose generation has moved on
(``graph-replay-nodes-overwritten``) instead of pricing somebody else's
replay. Whether that window is open in practice is a property of the
SCHEDULER, not of this module: with the overlap scheduler the host runs a
batch ahead of the device, so a round may not have completed by the time the
next replay is launched. The refusal counts are printed, so a boot answers
the question instead of a comment claiming it.
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
    "GraphNodes",
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

    def materialize(self, event) -> None:
        """Force the DEVICE-side event object into existence, now (#1241b).

        ``torch.cuda.Event`` is lazy: the ``cudaEvent_t`` is created on the
        first ``record()``. For an event destined to become a graph NODE
        that laziness would move the creation inside the capture, which is
        the one place this design promises not to allocate. So the pairs a
        capture will use are materialized BEFORE the capture, by recording
        them once on the (not yet capturing) current stream -- a record whose
        timestamp nobody reads and whose only purpose is the allocation.
        """
        event.record()


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
class GraphNodes:
    """The event-record NODES one captured CUDA graph carries (#1241b).

    ONE PAIR PER WRAPPED REGION, never per family: a family issued sixty
    times inside one graph owns sixty pairs, in occurrence order. Folding
    them onto one pair would not sum sixty transfers, it would span from the
    first record to the last and swallow all the compute in between -- the
    same shape of fabrication as a wait of 0.0, which is what the whole
    instrument exists to refuse.
    """

    key: object
    #: Phase prefix the capture ran under, e.g. ``spec_verify``. Carried so
    #: the families a REPLAY reports are spelled exactly like the ones an
    #: eager round reports -- the capture has no ``phase_scope`` of its own,
    #: because the scheduler is not in a decode round when it captures.
    phase: Optional[str] = None
    #: (pre, post, family), occurrence order.
    pairs: List[Tuple[object, object, str]] = dataclasses.field(
        default_factory=list
    )
    #: Bumped at every DECLARED replay of this key. A round holding an older
    #: generation is reading timestamps a later replay has overwritten.
    generation: int = 0
    #: Pairs whose events were not pre-created before the capture because the
    #: pre-allocation hint was too small. Not an error -- the pair is still
    #: correct -- but it is an allocation inside a capture, so it is counted
    #: and printed rather than left to be discovered.
    late_created: int = 0


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
    #: #1241b. The captured graph this round replayed, as declared at the
    #: replay site, or ``None`` when the round replayed a graph that carries
    #: no event nodes (or declared nothing at all).
    graph_nodes: Optional["GraphNodes"] = None
    #: The generation ``graph_nodes`` held when THIS round launched. Read
    #: back at harvest: a mismatch means a later replay has overwritten the
    #: timestamps and the split is refused by name.
    graph_generation: int = 0


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
        # -- #1241b: event NODES inside captured graphs -------------------
        #: graph key -> the nodes that graph carries. Written only by
        #: ``capture_scope`` (which REPLACES the entry, so a recapture of the
        #: same key cannot leave pairs of a destroyed graph readable).
        self._graphs: Dict[object, GraphNodes] = {}
        #: The capture currently being recorded into, or None.
        self._capture: Optional[GraphNodes] = None
        #: Events created BEFORE the capture, drawn from during it. Sized by
        #: the widest capture seen so far, so the first capture of a process
        #: grows it and every later one allocates nothing inside the capture.
        self._capture_prealloc: List[object] = []
        self._capture_pair_hint: int = 0
        #: Captures refused because one was already open. Never 0-by-luck:
        #: nested captures would interleave two graphs' nodes into one list.
        self._capture_refusals: int = 0
        #: Rounds whose nodes a later replay had already overwritten, and
        #: rounds whose nodes were not complete. Both are REFUSALS, both are
        #: printed: a split that is absent for one reason is not the same
        #: finding as a split absent for the other.
        self._graph_stale_reads: int = 0
        self._graph_unready_reads: int = 0
        #: The round bracket currently open, so the replay site can declare
        #: WHICH graph it is about to replay without threading the span
        #: through four dispatch layers.
        self._open_span: Optional[RoundSpan] = None

    # -- arming ---------------------------------------------------------

    @property
    def armed(self) -> bool:
        """True when a collective issued now would be recorded.

        #1241b widened this from "a round or a prefill forward is bracketed"
        to "... or a graph is being captured with a scope armed". The
        dispatch sites (parallel_state.py:1438 and its three siblings) gate
        their ``span`` on this one read, so a capture that does not flip it
        would lay no nodes at all -- the guard, not the ``span``, is what
        decides whether the capture is instrumented.
        """
        return self._slot is not None or self._capture is not None

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
        capture = self._capture
        if capture is not None and self._backend.is_capturing():
            # #1241b. Lay a dedicated pre/post pair into the graph. The pair
            # is this occurrence's own -- see GraphNodes for why sharing one
            # per family would measure the compute between the occurrences.
            family = self._label_hint or label or OTHER
            if capture.phase:
                family = capture.phase + FAMILY_PHASE_SEP + family
            pre, post = self._acquire_capture_pair(capture)
            capture.pairs.append((pre, post, family))
            # Same re-entry discipline as the slot path: a collective built
            # out of other collectives gets ONE region, not one per level.
            self._capture = None
            pre.record()
            try:
                yield
            finally:
                post.record()
                self._capture = capture
            return
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

    # -- graph capture / replay (#1241b) ---------------------------------

    def _acquire_capture_pair(self, capture: "GraphNodes"):
        """Two timing events for one wrapped region of the graph.

        Drawn from the pool created BEFORE the capture. A draw that finds it
        empty still returns a correct pair -- refusing to time a region would
        make the graph's wait a silent lower bound -- but it allocates inside
        the capture and says so through ``late_created``.
        """
        out = []
        for _ in range(2):
            if self._capture_prealloc:
                out.append(self._capture_prealloc.pop())
            else:
                capture.late_created += 1
                out.append(self._backend.event())
        return out[0], out[1]

    @contextmanager
    def capture_scope(self, key, phase: Optional[str] = None):
        """Record event NODES into the graph being captured under ``key``.

        Wrap the capture, not the warmups: outside a capture the scope costs
        one attribute read per collective and lays nothing (``span`` requires
        ``is_capturing()`` as well as this scope).

        Registration happens on the way OUT, and only on success -- a capture
        that raises leaves no half-populated node list behind, and a
        RE-capture of the same key REPLACES the entry, because the pairs of a
        destroyed graph would otherwise still be readable and would still
        carry a plausible number.

        Nested captures are refused rather than merged: two graphs recording
        into one list would attribute one graph's regions to the other.
        """
        if self._capture is not None:
            self._capture_refusals += 1
            yield
            return
        # Pre-create the pairs the widest capture so far needed. The first
        # capture of a process has no hint and grows the pool through
        # ``late_created``; every later capture allocates nothing inside the
        # capture window.
        want = 2 * self._capture_pair_hint - len(self._capture_prealloc)
        for _ in range(max(0, want)):
            ev = self._backend.event()
            self._backend.materialize(ev)
            self._capture_prealloc.append(ev)
        capture = GraphNodes(key=key, phase=phase)
        self._capture = capture
        try:
            yield
        finally:
            self._capture = None
        self._graphs[key] = capture
        if len(capture.pairs) > self._capture_pair_hint:
            self._capture_pair_hint = len(capture.pairs)

    def captured_graph(self, key) -> Optional["GraphNodes"]:
        """The nodes recorded for ``key``, or None if that graph carries none.

        Read-only accessor, so a test can assert the ONE-PAIR-PER-REGION rule
        (the mutant that folds a family onto one shared pair produces a graph
        that still replays and still prints a number) without reaching into a
        private dict and pinning its name.
        """
        return self._graphs.get(key)

    def note_graph_replay(self, key) -> None:
        """Declare, AT THE REPLAY SITE, which captured graph runs now.

        Two things happen, and the second one matters even when no round is
        open: the per-key generation is bumped, because this replay
        overwrites whatever timestamps the graph's nodes held -- including
        those a pending round is still waiting to read. A replay issued
        outside a decode round (warmup, a prefill piecewise graph) therefore
        invalidates a pending decode round's reading, and the round says so
        by name instead of pricing this replay as its own.
        """
        nodes = self._graphs.get(key)
        if nodes is None:
            return
        nodes.generation += 1
        span = self._open_span
        if span is not None:
            span.graph_nodes = nodes
            span.graph_generation = nodes.generation

    @property
    def graph_node_counts(self) -> Tuple[int, int, int, int, int]:
        """``(graphs, event nodes, late-created, stale reads, unready reads)``.

        The denominator of every graph-replayed split line. ``graphs`` and
        ``event nodes`` say how much of the boot CAN be split at all; the
        two read counts say how much of what could be, was not -- and why.
        """
        pairs = sum(len(g.pairs) for g in self._graphs.values())
        late = sum(g.late_created for g in self._graphs.values())
        return (
            len(self._graphs),
            2 * pairs,
            late,
            self._graph_stale_reads,
            self._graph_unready_reads,
        )

    def _read_graph_nodes(
        self, nodes: "GraphNodes"
    ) -> Optional[Dict[str, FamilyStat]]:
        """Per-family stats of the LAST completed replay, or None.

        Query-only, like everything else here. ``None`` means at least one
        node had not completed; the caller refuses the split rather than
        blocking on it or pricing the pairs that did complete.
        """
        acc: Dict[str, List[float]] = {}
        for pre, post, family in nodes.pairs:
            if not post.query():
                self._graph_unready_reads += 1
                return None
            ms = pre.elapsed_time(post)
            cell = acc.get(family)
            if cell is None:
                acc[family] = [ms, 1.0, ms]
            else:
                cell[0] += ms
                cell[1] += 1.0
                if ms > cell[2]:
                    cell[2] = ms
        return {
            name: FamilyStat(total_ms=v[0], count=int(v[1]), max_ms=v[2])
            for name, v in acc.items()
        }

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
            span = RoundSpan(start=start, slot=None, contended=True)
            self._open_span = span
            return span
        slot = Slot()
        self._slot = slot
        start = self._acquire()
        start.record()
        span = RoundSpan(start=start, slot=slot)
        # #1241b: reachable by ``note_graph_replay`` until this span closes.
        self._open_span = span
        return span

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
        if self._open_span is span:
            self._open_span = None
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
            nodes = span.graph_nodes
            span.graph_nodes = None
            refused = None
            graph_families: Optional[Dict[str, FamilyStat]] = None
            if nodes is None:
                # The graph ran, and it carries no event nodes: captured
                # before this instrument existed, captured by a runner that
                # arms no capture scope, or replayed without a declaration.
                # NAME THE MISSING THING -- the old reason said "graph-replay",
                # which read as "graphs cannot be split" and is now false.
                refused = "graph-replay-no-event-nodes"
            elif nodes.generation != span.graph_generation:
                # A later replay of the same key has already re-executed the
                # nodes and overwritten their timestamps. The numbers sitting
                # in them are somebody else's round.
                self._graph_stale_reads += 1
                refused = "graph-replay-nodes-overwritten"
            else:
                graph_families = self._read_graph_nodes(nodes)
                if graph_families is None:
                    refused = "graph-replay-nodes-unread"
            if refused is not None:
                # Drain the slot's events back to the pool without pricing
                # them: a partial wait over a round whose graph part is
                # invisible is not a smaller wait, it is an unknown one.
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
                    split_refused=refused,
                )
            # The EAGER collectives of this same forward -- everything the
            # replay did not cover (load_batch, a pre-replay rendezvous) --
            # are in the slot and belong in the same wait.
            eager = self.harvest_detail(slot)
            if eager is None:
                return RoundResult(
                    round_ms=round_ms,
                    wait_ms=None,
                    compute_ms=None,
                    families=None,
                    split_refused="collective-events-unread",
                )
            families = dict(eager.families)
            for name, stat in graph_families.items():
                have = families.get(name)
                families[name] = (
                    stat
                    if have is None
                    else FamilyStat(
                        total_ms=have.total_ms + stat.total_ms,
                        count=have.count + stat.count,
                        max_ms=max(have.max_ms, stat.max_ms),
                    )
                )
            wait_ms = sum(st.total_ms for st in families.values())
            return RoundResult(
                round_ms=round_ms,
                wait_ms=wait_ms,
                compute_ms=max(round_ms - wait_ms, 0.0),
                families=families,
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
