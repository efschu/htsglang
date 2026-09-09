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
"""#1241. The decode half of the per-rank compute/wait clock.

THE MISSING HALF, NOT A SECOND INSTRUMENT. #252 gave the PREFILL line a
per-rank ``gpu-ms: T (compute Tc, wait Tw)`` split, measured by
``utils/collective_clock.CollectiveClock`` at 0.13 % overhead. #1031 recorded
what decode has: nothing. There is no compute/wait column on any decode line
in any boot of this campaign, so the one question an uneven-TP decode split
raises -- WHICH RANK BINDS THE ROUND -- has never been answerable from a log.
This module answers it with the same clock, the same discipline and a line
the same reader parses.

THE LINE, deliberately named so ONE grep finds both families::

    Decode rank batch, rank: 1, #round: 4120, t: 1757397412.318, bs: 6,
    #rows: 24, #fwd: 2, gpu-ms: 31.4 (compute 19.8, wait 11.6)
    (wait by family: tp.all_reduce 9.2/129x, spec_verify:tp.all_reduce 2.4/3x)

``grep -E ' rank batch, '`` returns the prefill and the decode family
together, and ``debug_utils/rank_phase_summary.py`` -- whose docstring has
promised a ``Decode rank batch`` pattern since #252 and whose regex is
``\\w+\\s+rank batch`` -- parses it with no change to its arithmetic.

WHAT A ROUND IS, AND WHERE IT ENDS. One scheduler decode step: the funnel
``Scheduler._run_batch_forward`` already counts as ``_decode_steps_this_phase``.
Under speculation that step is SEVERAL forwards (a draft extend, then a target
verify), so a round is a FOLD of brackets exactly as the prefill line's
``#chunks: K`` folds several prefill forwards into one line. ``#fwd`` is that
K, and ``gpu-ms`` is the sum of the folded brackets' device time -- device
time, so the host-side gap BETWEEN two forwards of one round is not in it.
That gap is the PP bubble's separate term (``pp_bubble.PPBubbleMeter``) and is
never derived from this one, the same rule the prefill line already states.

THE ROUND IS CLOSED BY THE SAME FUNNEL THAT OPENS IT, and this is the whole
of its lifetime -- it is not a state that persists between batches. EVERY
batch entering ``_run_batch_forward`` retires whatever round was open; only a
DECODE batch then opens a new one. The first version of this module opened at
the funnel and retired only at the next ``begin_round``, which is a different
and wrong lifetime: on any boot that interleaves prefill with decode (chunked
prefill upstream; group D's own ``--tp-prefill-max-tokens`` phase-prefill
here) the round stayed open across the batch boundary, so the next PREFILL
forward's device time and collectives folded into the previous decode round,
were labelled ``spec_draft:`` by ``PHASE_OF_CATEGORY['extend']``, and -- worst
-- the round's bracket took the slot that ``SplitDeviceTimer`` had just armed
for that prefill forward, dropping the SHIPPED #252 line's compute/wait
column. ``CollectiveClock.open_round`` now refuses a contended slot instead of
taking it (belt), and the funnel closes the round on every non-decode batch
(braces). Both, because the failure was silent in both directions.

FLUSHED WHERE THE PREFILL HALF IS FLUSHED. ``metrics_reporter`` already
drains ``rank_prefill_log`` at the decode report, the prefill report and the
idle tick; the decode log is drained at the same three, and the idle tick
additionally RETIRES the open round -- that is the "idle flush" this
docstring used to name without anyone having written it, and it is why the
last round of a decode burst is emitted at all.

GRAPH-REPLAY HONEST -- WHICH SINCE #1241b MEANS SPLIT, NOT REFUSED. Decode is
the phase that actually runs from captured CUDA graphs, and a collective
inside a REPLAYED graph never executes the Python body that would record its
events. Its span is therefore not small, it is ABSENT -- and a slot with no
pairs reports ``wait 0.0``, which has the exact shape of a measurement and is
a fabrication. Slice 1 therefore withheld the split from every graphed round,
and boot ``weg2dec1_0909`` showed what that costs on the real full-perf form:
5695 withheld rounds per rank against 11 eager ones whose mean was 16.6x
longer -- a sample that cannot describe the form, and a denominator trap for
anyone who averages it anyway.

#1241b closes it at the source instead of changing the form. The collectives
are wrapped at CAPTURE time with event-record NODES that the graph re-executes
on every replay (``utils/collective_clock.capture_scope``), and the round
reads them one round late through the same query-only path. A graphed round
now prints the ordinary split line, with its families spelled exactly as an
eager round spells them::

    ... #fwd: 1, gpu-ms: 12.9 (compute 9.4, wait 3.5)
    (wait by family: tp.all_reduce 3.1/56x, spec_verify:tp.all_reduce 0.4/2x)

``split unavailable`` survives for the cases that genuinely are unknown, and
each now NAMES THE MISSING THING rather than naming the mechanism:

``graph-replay-no-event-nodes``
    the graph was captured with no scope armed (or the replay declared no
    key), so there is nothing to read;
``graph-replay-nodes-overwritten-by-N``
    a later replay of the same key re-executed the nodes before this round
    was read, or WHILE it was being read, and no reading of them survives in
    the ring. ``N`` is the LAG, in replays: a bare ``overwritten`` reads as a
    property of the mechanism, which since #1302 it is not. The witness is
    the launch FENCE and not the generation -- a replay that has been issued
    but not executed has destroyed nothing, and refusing on it cost 13,893 of
    13,950 rounds on boot weg2dec2c. The read is not atomic against the
    forward thread, so the fence is checked before AND after the read;
``graph-replay-nodes-unread``
    a node had not completed, or a node a concurrent replay re-recorded
    mid-read returned ``cudaErrorNotReady``. Never blocked on, never
    partially priced;
``graph-replay-key-replayed-twice``
    one bracketed forward declared the SAME graph twice. Two DIFFERENT
    graphs in one bracket sum, and are reported normally; the same graph
    twice cannot, because the second replay overwrote the nodes the first
    would have been read from.

``gpu-ms`` survives in every case -- the bracket is recorded AROUND the
replay, on the same stream. The eager arm (``--d-disable-cuda-graph`` on the
Weg-2 launcher) stays available as a CONTROL, and only that: it changes the
measured form, and full-perf validation is done with graphs and spec. WHICH
ROUNDS WERE GRAPHED IS STILL PRINTED, not assumed: the ``graphed-fwd K/N``
field is the denominator law applied to this line.

READ ONE ROUND LATE, NEVER A SYNC IN THE ROUND. Every reading is
``Event.query()``, never ``synchronize()``. Round N's events are read when
round N+1 begins; a round that is not ready yet is kept and read at the next
round instead of forcing the device. A round is therefore emitted one round
after it ran, and the last round of a boot is emitted by the idle flush or
not at all -- never by a sync.

JOINABLE TO THE LADDER -- ON WALL TIME, WHICH IS THE ONLY AXIS BOTH SIDES
ACTUALLY HAVE. ``devtools/probe_decode_ladder.py`` drives the front and
reports per-arm aggregates. Three candidate join keys, two of which are
category errors and are named here so nobody re-derives them:

* ``#round`` is the scheduler's ``forward_ct``, incremented on EVERY forward
  including prefill and idle, thousands deep into a boot. The probe's own
  ``rec["round"]`` is the REPEAT INDEX of an arm (0, 1, 2). Same word, two
  quantities; joining them is meaningless. ``#round`` is printed because it
  orders and de-duplicates rounds WITHIN one rank -- not to join across
  processes. Whether the three D ranks hold the same ``forward_ct`` for the
  same round is UNPROVEN: nothing checks it, and a single divergent forward
  on one rank would offset every later round. Do not assume it.
* ``#rows`` is rows SUBMITTED (``bs x rows_per_seq``; under MTP a verify
  submits ``num_draft_tokens`` rows per sequence). The probe counts
  ``completion_tokens``, i.e. tokens ACCEPTED. They differ by the acceptance
  rate, so they are not two readings of one quantity and must never be
  equated. The field was called ``#tokens`` in the first version of this
  module, which invited exactly that. Renamed.
* ``t`` is the join key: UNIX epoch seconds with milliseconds, taken at
  emission. Epoch and not ``monotonic`` deliberately -- the probe is a
  different process (and may be a different tool entirely), and epoch is the
  one clock both can read. It is on the line as a FIELD because the emission
  is one round LATE by construction, so the log prefix's timestamp is the
  time the line was WRITTEN, not the time the round RAN; ``t`` is stamped
  when the round is OPENED, which is the quantity a ladder window needs.
  The two differ by about one round, below the ladder's arm granularity,
  so a window join is sound and a per-round cross-process join is not
  offered.

``rank`` is on the line as a FIELD rather than left to the log prefix,
because the two groups of a Weg-2 boot write their prefixes differently and a
cross-rank join must not depend on the formatter.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = ["DecodeRoundLog", "RoundAcc"]


class RoundAcc:
    """One round's folded brackets, before it is readable."""

    __slots__ = ("round_id", "bs", "rows", "spans", "categories", "wall")

    def __init__(self, round_id: int, bs: int, rows: int) -> None:
        self.round_id = int(round_id)
        self.bs = int(bs)
        #: Rows SUBMITTED this round, never rows accepted. See the module
        #: docstring's join-key section for why the distinction is load
        #: bearing against the ladder's completion count.
        self.rows = int(rows)
        #: UNIX epoch at the moment the round was opened. The join axis.
        self.wall = time.time()
        # (span, category, graphed)
        self.spans: List[Tuple[object, str, bool]] = []
        self.categories: List[str] = []


class DecodeRoundLog:
    """Per-rank, per-round decode clock. One line per round per rank."""

    #: How many rounds the host-side cost of this instrument is measured over
    #: before it is stated once. #252 published 0.13 % for the prefill half;
    #: an instrument that does not price itself is not admissible as evidence
    #: about a timing regression it might itself have caused.
    OVERHEAD_ROUNDS: int = 200

    #: A round whose events never became readable is not kept forever. The
    #: cap is a MEMORY bound, not a timing one: an unreadable round is
    #: dropped with a named count, never emitted with invented numbers.
    MAX_PENDING_ROUNDS: int = 64

    def __init__(self, clock=None, rank: int = 0) -> None:
        self.clock = clock
        self.rank = int(rank)
        #: Set by the scheduler at the round boundary; read by every bracket
        #: opened inside that round. ``None`` = not in a decode round, and
        #: every bracket is then a no-op, which is how prefill forwards and
        #: warmup forwards stay outside this instrument.
        self.round_id: Optional[int] = None
        self._open: Optional[RoundAcc] = None
        self._pending: OrderedDict[int, RoundAcc] = OrderedDict()
        self._overhead_ns: int = 0
        self._overhead_rounds: int = 0
        self._overhead_gpu_ms: float = 0.0
        self._overhead_reported: bool = False
        self._dropped_rounds: int = 0
        #: #363-style structured tap, same contract as RankPrefillLog's:
        #: the LAST emitted round's numbers plus a monotone sequence, so a
        #: consumer can tell a new sample from the same one seen again.
        self.last_round_ms: Optional[float] = None
        self.last_wait_ms: Optional[float] = None
        self.last_compute_ms: Optional[float] = None
        self.last_split_known: bool = False
        self.last_seq: int = 0

    # -- round boundary --------------------------------------------------

    def begin_round(self, round_id: int, bs: int, rows: int) -> None:
        """Open round ``round_id`` and read whatever earlier rounds are ready.

        The flush happens BEFORE the new round is opened, so the reading of
        round N-1 is charged to the boundary and never sits between two
        forwards of round N.
        """
        t0 = time.perf_counter_ns()
        self._retire_open()
        self.flush()
        self._open = RoundAcc(round_id, bs, rows)
        self.round_id = int(round_id)
        self._overhead_ns += time.perf_counter_ns() - t0

    def end_round(self) -> None:
        """Close the open round without opening another, and drain.

        THE OTHER HALF OF ``begin_round``, and the reason a round's lifetime
        is one batch rather than "until the next decode batch". Called from
        the funnel for every NON-decode batch and from the idle tick, so that

        * a prefill/extend/idle forward is never folded into a decode round
          it did not run in, and
        * the last round of a decode burst is emitted rather than sitting
          open until the next burst -- or, at the end of a phase, forever.

        Idempotent: outside a round it is two branches and a flush.
        """
        t0 = time.perf_counter_ns()
        self._retire_open()
        self.flush()
        self._overhead_ns += time.perf_counter_ns() - t0

    @property
    def has_pending(self) -> bool:
        """True while a round is open or a retired round is unread.

        Same name and same role as ``RankPrefillLog.has_pending``, so the
        three flush sites in ``metrics_reporter`` guard both halves of the
        instrument with the same shape of condition.
        """
        return self._open is not None or bool(self._pending)

    def _retire_open(self) -> None:
        if self._open is None:
            return
        acc, self._open = self._open, None
        self.round_id = None
        if not acc.spans:
            # A decode step that issued no timed forward (a rank that idled
            # this step, a mode this instrument does not bracket). Counted
            # nowhere and emitted nowhere: an empty round is not a zero round.
            return
        self._pending[acc.round_id] = acc
        while len(self._pending) > self.MAX_PENDING_ROUNDS:
            self._pending.popitem(last=False)
            self._dropped_rounds += 1

    # -- brackets --------------------------------------------------------

    #: Forward category -> the PHASE prefix its collectives carry (#1241).
    #: One writer for this mapping, here, so the four dispatch sites pass
    #: only the category they already compute for ``device_timer``.
    PHASE_OF_CATEGORY: Dict[str, str] = {
        "target_verify": "spec_verify",
        "draft_extend": "spec_draft",
        "extend": "spec_draft",
    }

    @contextmanager
    def segment(self, category: str, graphed: bool):
        """Bracket one forward of the currently open round.

        ``graphed`` is the caller's statement that this forward ran from a
        REPLAYED cuda graph. It is a parameter and not a guess, because the
        only site that knows is the dispatch site that chose the replay.

        An EXTEND-shaped forward inside a decode round is a speculation
        forward -- the scheduler only opens a round for a decode batch -- so
        its collectives are prefixed and stay tellable apart from the target
        model's own ``tp.all_reduce``.
        """
        acc = self._open
        if acc is None or self.clock is None:
            yield
            return
        t0 = time.perf_counter_ns()
        span = self.clock.open_round()
        span.graph_replayed = bool(graphed)
        phase = self.PHASE_OF_CATEGORY.get(str(category))
        self._overhead_ns += time.perf_counter_ns() - t0
        try:
            with self.clock.phase_scope(phase):
                yield
        finally:
            t1 = time.perf_counter_ns()
            self.clock.close_round(span)
            acc.spans.append((span, str(category), bool(graphed)))
            acc.categories.append(str(category))
            self._overhead_ns += time.perf_counter_ns() - t1

    # -- reading ---------------------------------------------------------

    def flush(self) -> None:
        """Emit every pending round whose events have all become readable.

        Query-only. Rounds are emitted in round order; a round that is not
        ready STOPS the drain, so the log never reports round 7 before
        round 6 and a reader can treat the sequence as ordered.
        """
        if self.clock is None:
            return
        while self._pending:
            round_id = next(iter(self._pending))
            acc = self._pending[round_id]
            results = []
            for span, category, graphed in acc.spans:
                r = self.clock.harvest_round(span)
                if r is None:
                    self._snapshot_pending()
                    return
                results.append((r, category, graphed))
            self._pending.pop(round_id)
            self._emit(acc, results)

    def _snapshot_pending(self) -> None:
        """#1302. Take the reading of every round that cannot be READ yet.

        This is the whole mechanism, and it lives here rather than in the
        clock because only the log knows which rounds are still waiting. A
        round whose bracket has not completed is not emittable -- but the
        graph it replayed IS readable at this instant, and will not be once
        the next replay of that key executes, which under the overlap
        scheduler is one round away. Taking the reading now and keeping it
        in the clock's ring is what lets the round be emitted with a split
        one or more replays later, instead of with
        ``graph-replay-nodes-overwritten`` on 99.6 % of rounds.

        Every pending round, not just the blocking one: any of them can be
        the next to age out. Query-only, and a snapshot that finds nothing
        readable refuses nothing and counts nothing.
        """
        snapshot = getattr(self.clock, "snapshot_round", None)
        if snapshot is None:
            return
        for acc in self._pending.values():
            for span, _category, _graphed in acc.spans:
                snapshot(span)

    def _emit(self, acc: RoundAcc, results) -> None:
        round_ms = 0.0
        wait_ms = 0.0
        split_known = True
        graphed_fwd = 0
        refusals: List[str] = []
        family_acc: Dict[str, List[float]] = {}
        for r, _category, graphed in results:
            round_ms += r.round_ms
            if graphed:
                graphed_fwd += 1
            if r.wait_ms is None:
                split_known = False
                if r.split_refused and r.split_refused not in refusals:
                    refusals.append(r.split_refused)
                continue
            wait_ms += r.wait_ms
            for name, stat in (r.families or {}).items():
                slot = family_acc.get(name)
                if slot is None:
                    family_acc[name] = [stat.total_ms, float(stat.count)]
                else:
                    slot[0] += stat.total_ms
                    slot[1] += stat.count

        line = (
            "Decode rank batch, rank: %d, #round: %d, t: %.3f, bs: %d, "
            "#rows: %d, #fwd: %d, gpu-ms: %.1f"
        )
        args: list = [
            self.rank,
            acc.round_id,
            acc.wall,
            acc.bs,
            acc.rows,
            len(results),
            round_ms,
        ]
        if split_known:
            line += " (compute %.1f, wait %.1f)"
            args += [max(round_ms - wait_ms, 0.0), wait_ms]
            if family_acc:
                parts = ", ".join(
                    "%s %.1f/%dx" % (name, total_ms, int(count))
                    for name, (total_ms, count) in sorted(
                        family_acc.items(), key=lambda kv: -kv[1][0]
                    )
                )
                line += " (wait by family: %s)"
                args.append(parts)
        else:
            # NEVER a zero here. The reason is printed with its denominator,
            # so "no split" can be told from "no wait".
            line += " (split unavailable: %s, graphed-fwd %d/%d)"
            args += [
                "+".join(refusals) or "unknown",
                graphed_fwd,
                len(results),
            ]
        logger.info(line, *args)

        self.last_round_ms = round_ms
        self.last_wait_ms = wait_ms if split_known else None
        self.last_compute_ms = max(round_ms - wait_ms, 0.0) if split_known else None
        self.last_split_known = split_known
        if split_known:
            self.last_seq += 1

        self._overhead_rounds += 1
        self._overhead_gpu_ms += round_ms
        self._maybe_report_overhead()

    def _maybe_report_overhead(self) -> None:
        """State this instrument's own host cost ONCE, with its denominator."""
        if self._overhead_reported or self._overhead_rounds < self.OVERHEAD_ROUNDS:
            return
        self._overhead_reported = True
        us_per_round = self._overhead_ns / 1000.0 / max(1, self._overhead_rounds)
        mean_gpu_ms = self._overhead_gpu_ms / max(1, self._overhead_rounds)
        share = (
            100.0 * (us_per_round / 1000.0) / mean_gpu_ms if mean_gpu_ms > 0 else 0.0
        )
        # THE EMISSION RATE, STATED. One line per round per rank is a lot of
        # log at 30 ms rounds, and the operator reading a ladder window out of
        # this log is entitled to know how much of that log is the instrument
        # before deciding to run the window with the emitter on. Derived from
        # the same denominator as the overhead, so it cannot drift from it.
        lines_per_s = 1000.0 / mean_gpu_ms if mean_gpu_ms > 0 else float("nan")
        arm_refusals, round_contentions = (0, 0)
        counts = getattr(self.clock, "contention_counts", None)
        if counts is not None:
            arm_refusals, round_contentions = counts
        # #1241b DENOMINATOR OF THE GRAPHED SPLIT. `graphs`/`nodes` say how
        # much of this boot CAN be split at all -- zero nodes and every
        # graphed round is refused for a structural reason, not a timing one.
        # `stale` and `unready` say how much of what could be, was not: stale
        # is the scheduler running a replay ahead of the read (try the
        # `--d-disable-cuda-graph` control arm, or `--d-disable-overlap-
        # schedule`, before reading anything into the round times), unready is
        # the device not having finished. Neither is ever waited on.
        graphs = nodes = late = stale = unready = reused = 0
        ring_depth = ring_hits = 0
        gcounts = getattr(self.clock, "graph_node_counts", None)
        if gcounts is not None:
            (
                graphs,
                nodes,
                late,
                stale,
                unready,
                reused,
                ring_depth,
                ring_hits,
            ) = gcounts
        # #1302 THREE DENOMINATORS, LABELLED APART ON THE LINE ITSELF. The
        # dec2c boot record read "196 overwritten" off this line as NODES and
        # set it against the 1552 node total -- two orders of magnitude out,
        # because the counter is FORWARDS refused, and because this line is
        # emitted ONCE after the first OVERHEAD_ROUNDS rounds and is
        # therefore a running total as of that round, never a boot total.
        # The per-round withheld-reason tally at the end of a boot is the
        # only boot-wide population. Both facts now ride on the line.
        logger.info(
            "Decode rank clock overhead, rank: %d, %.1f us/round host-side over "
            "%d rounds = %.3f %% of the mean round gpu-ms %.2f, emitting about "
            "%.0f lines/s on this rank. Host-side only: the device cost is two "
            "event records per forward and is inside the bracket it measures. "
            "Dropped rounds (events never readable): %d. Slot contention with "
            "the prefill half (#252), both MUST be 0: prefill armings refused "
            "%d, decode rounds opened contended %d. Graph event nodes (#1241b): "
            "%d graphs carry %d nodes (%d created inside a capture); reads "
            "skipped without blocking, counted in FORWARDS and not in nodes, "
            "and as of this line only (it is emitted once, after %d rounds): "
            "%d overwritten by a later replay, %d "
            "not yet complete, %d rounds that replayed one graph twice. "
            "Reading ring (#1302): depth %d replays, %d rounds served from a "
            "reading taken before the overwrite.",
            self.rank,
            us_per_round,
            self._overhead_rounds,
            share,
            mean_gpu_ms,
            lines_per_s,
            self._dropped_rounds,
            arm_refusals,
            round_contentions,
            graphs,
            nodes,
            late,
            self._overhead_rounds,
            stale,
            unready,
            reused,
            ring_depth,
            ring_hits,
        )
