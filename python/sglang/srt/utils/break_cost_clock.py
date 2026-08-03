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
"""#494: what one CUDA-graph BREAK costs, decomposed, per rank.

WHY THIS EXISTS
---------------
The #462 breakable route replays a decode step as ``segment, break, segment,
break, ...``: the captured segments hold the compute, and between them an eager
break slot fetches the routed experts and republishes the slot vector
(``layers/moe/breakable_offload.py``). ``TICKET_462_f2_and_replay.md`` §3 (F2)
gates default-on on ONE number -- what a break costs, per layer per step -- and
until this module existed there was no instrument that produced it. F2 was an
implementation task; with this it is a measurement.

WHAT IT MEASURES, PER CROSSING
------------------------------
A crossing is one ``segment_end -> slot -> segment_start`` triple. Three device
terms, taken from CUDA events recorded on the replay stream:

    gap_in_ms   segment i's last device work -> the slot's first device work.
                The device sits idle here while the host runs the break's
                rendezvous and planning. This is WAIT.
    slot_ms     the slot's own device span (the pinned publish + the expert
                H2D the fetch issues). This is COMPUTE (device-busy).
    gap_out_ms  the slot's last device work -> segment i+1's first device work.
                Device idle again: relaunch. This is WAIT.

plus one host term, ``host_ms`` (wall clock around the break function), itself
decomposed by :func:`break_cost_phase` into F2's named terms -- ``rendezvous``
(``topk_ids.tolist()``), ``planning`` (observe/split/resolve), ``publish``
(remap + the blocking pinned copy) and ``fetch`` (the expert DMA issue).

Per round the module also reports ``compute_ms`` (segment spans + slot spans),
``wait_ms`` (all gaps) and ``span_ms`` (first event to last), so the
compute-vs-wait split of the ms/round canon is available per rank without
subtracting logs by hand. ``span_ms - (compute_ms + wait_ms)`` is a coherence
residual and is emitted rather than hidden.

READ LATE, NEVER SYNCHRONISE
----------------------------
The one way an instrument like this lies is by measuring itself: reading an
event in the round that recorded it forces the host to wait for the very stall
under measurement. So events are read DEFERRED -- a round's numbers are computed
only once ``SGLANG_BREAK_COST_DEFER_ROUNDS`` (default 2) further rounds have
started, and only through ``Event.query()``. There is no ``synchronize()`` and
no ``elapsed_time()`` on an unready event anywhere in this module; a round whose
events are not ready simply stays in the queue.

Events are pooled and re-recorded, so a steady-state armed run allocates no
events and no per-round lists beyond the queue it drains.

OFF BY DEFAULT, AND BYTE-NEUTRAL WHEN OFF
-----------------------------------------
:func:`break_cost_clock` returns ``None`` unless ``SGLANG_BREAK_COST_PROBE=1``,
and every call site branches on that ``None`` into the code it ran before. With
the probe off: no event is created, no event is recorded, nothing is allocated
per round, and :func:`break_cost_phase` returns one process-wide no-op object
(the same object every time -- ``test_break_cost_probe_494`` pins the identity,
which is the allocation-free proof). See
``tests/moe_offload/test_break_cost_probe_494.py``.

OUTPUT
------
One JSON object per round per rank, one per line, appended to
``SGLANG_BREAK_COST_PATH`` (default ``/tmp/break_cost.<rank_tag>.jsonl``).
Machine-readable on purpose: the F2 write-up reads these lines, it does not
grep prose.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from collections import deque

logger = logging.getLogger(__name__)

ENV_ENABLE = "SGLANG_BREAK_COST_PROBE"
ENV_PATH = "SGLANG_BREAK_COST_PATH"
ENV_DEFER = "SGLANG_BREAK_COST_DEFER_ROUNDS"
ENV_WARMUP = "SGLANG_BREAK_COST_WARMUP_ROUNDS"
ENV_DETAIL = "SGLANG_BREAK_COST_DETAIL"
ENV_RANK_TAG = "SGLANG_BREAK_COST_RANK_TAG"

#: Schema version of an emitted record. Bump when a field changes meaning.
RECORD_VERSION = 1

#: Rounds that must have STARTED after a round before its events are read.
#: 1 would already satisfy "never read in the recording round"; the default
#: keeps one further round of slack so a slow DMA does not make every harvest
#: attempt miss and grow the queue.
DEFAULT_DEFER_ROUNDS = 2

#: The F2 phase names, in the order ``prepare_breakable`` runs them. Fixed here
#: so a consumer can rely on the key set rather than discovering it per record.
PHASES: Tuple[str, ...] = ("rendezvous", "planning", "publish", "fetch")


__all__ = [
    "BreakCostClock",
    "DEFAULT_DEFER_ROUNDS",
    "ENV_ENABLE",
    "ENV_PATH",
    "PHASES",
    "RECORD_VERSION",
    "Round",
    "break_cost_clock",
    "break_cost_phase",
    "reset_break_cost_clock_for_test",
]


class _NoPhase:
    """The disabled-path phase context: one shared instance, no allocation."""

    __slots__ = ()

    def __enter__(self) -> "_NoPhase":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


#: The single instance returned while the probe is off. Its identity IS the
#: no-allocation guarantee, and the pin test asserts on that identity.
NO_PHASE = _NoPhase()


class _Phase:
    """A host wall-clock span attributed to one named F2 term.

    One instance per name per clock, reused across rounds, so an armed run does
    not allocate a context manager per break per layer per step.
    """

    __slots__ = ("_clock", "_name", "_t0")

    def __init__(self, clock: "BreakCostClock", name: str) -> None:
        self._clock = clock
        self._name = name
        self._t0 = 0.0

    def __enter__(self) -> "_Phase":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._clock._add_phase(self._name, (time.perf_counter() - self._t0) * 1e3)
        return False


class Round:
    """Events and host timings of ONE replay of one breakable graph.

    Held in the clock's queue until the deferred read is allowed AND every
    event reports ready. Nothing here reads an event.
    """

    __slots__ = (
        "graph_key",
        "index",
        "seg_events",
        "slot_events",
        "slot_names",
        "slot_host_ms",
        "slot_phases",
        "born_at",
        "wall_ms",
        "_open_slot_t0",
    )

    def __init__(self, graph_key: str, index: int, born_at: int) -> None:
        self.graph_key = graph_key
        self.index = index
        self.born_at = born_at
        self.seg_events: List[List[Any]] = []
        self.slot_events: List[List[Any]] = []
        self.slot_names: List[str] = []
        self.slot_host_ms: List[float] = []
        self.slot_phases: List[Dict[str, float]] = []
        self.wall_ms = 0.0
        self._open_slot_t0 = 0.0


class BreakCostClock:
    """Records break crossings, aggregates them deferred, emits one line/round.

    The clock is created only when the probe is enabled. ``event_factory`` and
    ``sink`` exist so the hermetic test drives the REAL aggregation and the REAL
    record shape with a scripted fake event -- see the test's docstring for the
    three places its fake is deliberately STRICTER than ``torch.cuda.Event``.
    """

    def __init__(
        self,
        event_factory: Optional[Callable[[], Any]] = None,
        sink: Optional[Callable[[Dict[str, Any]], None]] = None,
        defer_rounds: int = DEFAULT_DEFER_ROUNDS,
        warmup_rounds: int = 0,
        detail: bool = True,
        rank_tag: str = "rank0",
    ) -> None:
        self._event_factory = event_factory or _torch_event
        self._sink = sink or _null_sink
        self._defer_rounds = max(1, int(defer_rounds))
        self._warmup_rounds = max(0, int(warmup_rounds))
        self._detail = bool(detail)
        self.rank_tag = rank_tag
        self._pool: List[Any] = []
        self._queue: Deque[Round] = deque()
        self._round_counters: Dict[str, int] = {}
        self._rounds_started = 0
        self._phases: Dict[str, _Phase] = {n: _Phase(self, n) for n in PHASES}
        self._phase_acc: Dict[str, float] = {}
        self._current: Optional[Round] = None
        #: Host cost of the instrument's own harvest+emit during a round. Kept
        #: so an end-to-end ms/token taken under the probe can be corrected.
        self._sink_ms = 0.0
        self.events_created = 0
        self.records_emitted = 0
        self.rounds_dropped = 0

    # -- recording ------------------------------------------------------

    def begin_round(self, graph_key: str) -> Round:
        index = self._round_counters.get(graph_key, 0)
        self._round_counters[graph_key] = index + 1
        self._rounds_started += 1
        rnd = Round(graph_key, index, self._rounds_started)
        self._current = rnd
        rnd.wall_ms = time.perf_counter()
        return rnd

    def segment_begin(self, rnd: Round) -> None:
        rnd.seg_events.append([self._mark(), None])

    def segment_end(self, rnd: Round) -> None:
        rnd.seg_events[-1][1] = self._mark()

    def slot_begin(self, rnd: Round, name: str) -> None:
        self._phase_acc = {}
        rnd.slot_names.append(name)
        rnd.slot_events.append([self._mark(), None])
        rnd._open_slot_t0 = time.perf_counter()

    def slot_end(self, rnd: Round) -> None:
        rnd.slot_host_ms.append((time.perf_counter() - rnd._open_slot_t0) * 1e3)
        rnd.slot_events[-1][1] = self._mark()
        rnd.slot_phases.append(self._phase_acc)
        self._phase_acc = {}

    def end_round(self, rnd: Round) -> None:
        rnd.wall_ms = (time.perf_counter() - rnd.wall_ms) * 1e3
        self._current = None
        if not _round_complete(rnd):
            # An exception unwound the replay mid-round. Half a round has no
            # crossing to price, and keeping it would block the in-order drain
            # forever, so its events go back to the pool and it is dropped.
            self._recycle(rnd)
            self.rounds_dropped += 1
            return
        self._queue.append(rnd)
        t0 = time.perf_counter()
        self._drain()
        self._sink_ms = (time.perf_counter() - t0) * 1e3

    def phase(self, name: str):
        """Host timer for one F2 term. Unknown names are accepted and reported."""
        p = self._phases.get(name)
        if p is None:
            p = self._phases[name] = _Phase(self, name)
        return p

    def _add_phase(self, name: str, ms: float) -> None:
        self._phase_acc[name] = self._phase_acc.get(name, 0.0) + ms

    def _mark(self) -> Any:
        ev = self._pool.pop() if self._pool else self._new_event()
        ev.record()
        return ev

    def _new_event(self) -> Any:
        self.events_created += 1
        return self._event_factory()

    # -- deferred harvest ------------------------------------------------

    def _drain(self) -> None:
        """Emit every queued round that is both old enough and complete.

        Query-only. A round that is not ready stops the drain: rounds are
        emitted in the order they were recorded, and a later round cannot be
        ready before an earlier one on the same stream.
        """
        while self._queue:
            rnd = self._queue[0]
            if self._rounds_started - rnd.born_at < self._defer_rounds:
                return
            if not _round_ready(rnd):
                return
            self._queue.popleft()
            record = self._aggregate(rnd)
            self._recycle(rnd)
            if rnd.index >= self._warmup_rounds:
                self._sink(record)
                self.records_emitted += 1

    def _recycle(self, rnd: Round) -> None:
        for pair in rnd.seg_events:
            self._pool.extend(e for e in pair if e is not None)
        for pair in rnd.slot_events:
            self._pool.extend(e for e in pair if e is not None)
        rnd.seg_events.clear()
        rnd.slot_events.clear()

    # -- aggregation ------------------------------------------------------

    def _aggregate(self, rnd: Round) -> Dict[str, Any]:
        seg_ms = [s.elapsed_time(e) for s, e in rnd.seg_events]
        crossings: List[Dict[str, Any]] = []
        wait_ms = 0.0
        slot_total = 0.0
        for i in range(len(rnd.slot_events)):
            prev_end, next_start = _crossing_bounds(rnd, i)
            start, end = rnd.slot_events[i]
            gap_in = prev_end.elapsed_time(start) if prev_end is not None else 0.0
            slot = start.elapsed_time(end)
            gap_out = end.elapsed_time(next_start) if next_start is not None else 0.0
            wait_ms += gap_in + gap_out
            slot_total += slot
            phases = rnd.slot_phases[i] if i < len(rnd.slot_phases) else {}
            crossings.append(
                {
                    "i": i,
                    "name": rnd.slot_names[i],
                    "gap_in_ms": gap_in,
                    "slot_ms": slot,
                    "gap_out_ms": gap_out,
                    "host_ms": rnd.slot_host_ms[i],
                    "phases": phases,
                }
            )

        compute_ms = sum(seg_ms) + slot_total
        span_ms = 0.0
        if rnd.seg_events:
            first = rnd.seg_events[0][0]
            last = rnd.seg_events[-1][1]
            span_ms = first.elapsed_time(last)

        record: Dict[str, Any] = {
            "v": RECORD_VERSION,
            "rank_tag": self.rank_tag,
            "graph": rnd.graph_key,
            "round": rnd.index,
            "segments": len(seg_ms),
            "crossings": len(crossings),
            "span_ms": span_ms,
            "compute_ms": compute_ms,
            "wait_ms": wait_ms,
            "segment_ms": sum(seg_ms),
            "slot_ms": slot_total,
            "host_ms": sum(rnd.slot_host_ms),
            "wall_ms": rnd.wall_ms,
            # span - (compute + wait) must be ~0 for a coherent round; emitted
            # rather than asserted so a real run can show its own residual.
            "residual_ms": span_ms - (compute_ms + wait_ms),
            "probe_sink_ms": self._sink_ms,
            "by_name": _by_name(crossings),
        }
        if self._detail:
            record["crossing_detail"] = crossings
        return record


def _crossing_bounds(rnd: Round, i: int) -> Tuple[Any, Any]:
    """The two segment events crossing ``i`` is measured against.

    Crossing ``i`` sits between segment ``i`` and segment ``i+1``: its incoming
    gap ends at segment ``i``'s END event and its outgoing gap ends at segment
    ``i+1``'s START event. This one mapping is what makes a crossing number
    mean anything, so it is a named function -- the can-fail arm in
    ``test_break_cost_probe_494`` replaces exactly this function with an
    off-by-one variant and requires the timeline gate to go red.
    """
    prev_end = rnd.seg_events[i][1] if i < len(rnd.seg_events) else None
    next_start = rnd.seg_events[i + 1][0] if i + 1 < len(rnd.seg_events) else None
    return prev_end, next_start


def _round_complete(rnd: Round) -> bool:
    """Every opened event pair got its closing event."""
    if not rnd.seg_events:
        return False
    for pair in rnd.seg_events:
        if pair[1] is None:
            return False
    for pair in rnd.slot_events:
        if pair[1] is None:
            return False
    return len(rnd.slot_host_ms) == len(rnd.slot_events)


def _round_ready(rnd: Round) -> bool:
    for pair in rnd.seg_events:
        for ev in pair:
            if ev is None or not ev.query():
                return False
    for pair in rnd.slot_events:
        for ev in pair:
            if ev is None or not ev.query():
                return False
    return True


def _by_name(crossings: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Per break-point totals -- the row F2 quotes (43 x one MoE layer)."""
    out: Dict[str, Dict[str, Any]] = {}
    for c in crossings:
        agg = out.get(c["name"])
        if agg is None:
            agg = out[c["name"]] = {
                "count": 0,
                "gap_in_ms": 0.0,
                "slot_ms": 0.0,
                "gap_out_ms": 0.0,
                "host_ms": 0.0,
                "phases": {},
            }
        agg["count"] += 1
        agg["gap_in_ms"] += c["gap_in_ms"]
        agg["slot_ms"] += c["slot_ms"]
        agg["gap_out_ms"] += c["gap_out_ms"]
        agg["host_ms"] += c["host_ms"]
        for name, ms in c["phases"].items():
            agg["phases"][name] = agg["phases"].get(name, 0.0) + ms
    return out


def _torch_event() -> Any:
    import torch

    return torch.cuda.Event(enable_timing=True)


def _null_sink(record: Dict[str, Any]) -> None:
    return None


class _JsonlSink:
    """Append one JSON object per line. Line-buffered: a killed server keeps
    every round it already finished."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._fh = open(path, "a", buffering=1)

    def __call__(self, record: Dict[str, Any]) -> None:
        self._fh.write(json.dumps(record, separators=(",", ":")) + "\n")


def _resolve_rank_tag() -> str:
    tag = os.environ.get(ENV_RANK_TAG)
    if tag:
        return tag
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return f"rank{dist.get_rank()}"
    except Exception:  # pragma: no cover - probe must never break a boot
        pass
    for env in ("RANK", "LOCAL_RANK", "SGLANG_DP_RANK"):
        val = os.environ.get(env)
        if val is not None and val.isdigit():
            return f"rank{val}"
    return "rank0"


_CLOCK: Optional[BreakCostClock] = None
_RESOLVED = False


def break_cost_clock() -> Optional[BreakCostClock]:
    """The clock, or ``None`` when the probe is off.

    Hot-path contract: two module-global reads and a return. No allocation, no
    event, no import of torch when disabled.
    """
    global _RESOLVED, _CLOCK
    if _RESOLVED:
        return _CLOCK
    _RESOLVED = True
    if os.environ.get(ENV_ENABLE, "0") not in ("1", "true", "TRUE", "yes"):
        return None
    rank_tag = _resolve_rank_tag()
    path = os.environ.get(ENV_PATH) or f"/tmp/break_cost.{rank_tag}.jsonl"
    _CLOCK = BreakCostClock(
        sink=_JsonlSink(path),
        defer_rounds=int(os.environ.get(ENV_DEFER, DEFAULT_DEFER_ROUNDS)),
        warmup_rounds=int(os.environ.get(ENV_WARMUP, 0)),
        detail=os.environ.get(ENV_DETAIL, "1") == "1",
        rank_tag=rank_tag,
    )
    logger.info(
        "break-cost probe ARMED (%s): deferred read %d rounds, records -> %s",
        rank_tag,
        _CLOCK._defer_rounds,
        path,
    )
    return _CLOCK


def break_cost_phase(name: str):
    """Host timer for one F2 term, or the shared no-op when the probe is off."""
    clock = break_cost_clock()
    if clock is None:
        return NO_PHASE
    return clock.phase(name)


def reset_break_cost_clock_for_test(clock: Optional[BreakCostClock] = None) -> None:
    """Install ``clock`` (or clear it) and re-arm env resolution. Tests only."""
    global _RESOLVED, _CLOCK
    _CLOCK = clock
    _RESOLVED = clock is not None
