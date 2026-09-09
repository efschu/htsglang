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

    Decode rank batch, rank: 1, #round: 4120, bs: 6, #tokens: 24, #fwd: 2,
    gpu-ms: 31.4 (compute 19.8, wait 11.6)
    (wait by family: tp.all_reduce 9.2/129x, spec_verify:tp.all_reduce 2.4/3x)

``grep -E ' rank batch, '`` returns the prefill and the decode family
together, and ``debug_utils/rank_phase_summary.py`` -- whose docstring has
promised a ``Decode rank batch`` pattern since #252 and whose regex is
``\\w+\\s+rank batch`` -- parses it with no change to its arithmetic.

WHAT A ROUND IS. One scheduler decode step: the funnel
``Scheduler._run_batch_forward`` already counts as ``_decode_steps_this_phase``.
Under speculation that step is SEVERAL forwards (a draft extend, then a target
verify), so a round is a FOLD of brackets exactly as the prefill line's
``#chunks: K`` folds several prefill forwards into one line. ``#fwd`` is that
K, and ``gpu-ms`` is the sum of the folded brackets' device time -- device
time, so the host-side gap BETWEEN two forwards of one round is not in it.
That gap is the PP bubble's separate term (``pp_bubble.PPBubbleMeter``) and is
never derived from this one, the same rule the prefill line already states.

GRAPH-REPLAY HONEST, WHICH HERE MEANS MOSTLY REFUSING. Decode is the phase
that actually runs from captured CUDA graphs, and a collective inside a
REPLAYED graph never executes the Python body that would record its events.
Its span is therefore not small, it is ABSENT -- and a slot with no pairs
reports ``wait 0.0``, which has the exact shape of a measurement and is a
fabrication. So a round with any graph-replayed forward prints::

    ... #fwd: 1, gpu-ms: 12.9 (split unavailable: graph-replay, graphed-fwd 1/1)

``gpu-ms`` survives -- the bracket is recorded AROUND the replay, on the same
stream, so the round's device time is honest -- and the split is withheld.
A boot that wants the split for the whole ladder runs the decode arm eager
(``--disable-cuda-graph``) or reads the split from the rounds that fell out
of the capture (bs above the captured maximum, or a verify shape the decode
graph does not cover). WHICH ROUNDS THOSE WERE IS PRINTED, not assumed: the
``graphed-fwd K/N`` field is the denominator law applied to this line.

READ ONE ROUND LATE, NEVER A SYNC IN THE ROUND. Every reading is
``Event.query()``, never ``synchronize()``. Round N's events are read when
round N+1 begins; a round that is not ready yet is kept and read at the next
round instead of forcing the device. A round is therefore emitted one round
after it ran, and the last round of a boot is emitted by the idle flush or
not at all -- never by a sync.

JOINABLE TO THE LADDER. ``devtools/probe_decode_ladder.py`` reports per-round
aggregates over the front. The join key is printed on this line and nowhere
derived: ``#round`` (the scheduler's own monotone decode-step counter),
``bs`` and ``#tokens``. ``rank`` is on the line as a FIELD rather than left to
the log prefix, because the two groups of a Weg-2 boot write their prefixes
differently and a cross-rank join must not depend on the formatter.
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

    __slots__ = ("round_id", "bs", "tokens", "spans", "categories")

    def __init__(self, round_id: int, bs: int, tokens: int) -> None:
        self.round_id = int(round_id)
        self.bs = int(bs)
        self.tokens = int(tokens)
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

    def begin_round(self, round_id: int, bs: int, tokens: int) -> None:
        """Open round ``round_id`` and read whatever earlier rounds are ready.

        The flush happens BEFORE the new round is opened, so the reading of
        round N-1 is charged to the boundary and never sits between two
        forwards of round N.
        """
        t0 = time.perf_counter_ns()
        self._retire_open()
        self.flush()
        self._open = RoundAcc(round_id, bs, tokens)
        self.round_id = int(round_id)
        self._overhead_ns += time.perf_counter_ns() - t0

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
                    return
                results.append((r, category, graphed))
            self._pending.pop(round_id)
            self._emit(acc, results)

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
            "Decode rank batch, rank: %d, #round: %d, bs: %d, #tokens: %d, "
            "#fwd: %d, gpu-ms: %.1f"
        )
        args: list = [
            self.rank,
            acc.round_id,
            acc.bs,
            acc.tokens,
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
        logger.info(
            "Decode rank clock overhead, rank: %d, %.1f us/round host-side over "
            "%d rounds = %.3f %% of the mean round gpu-ms %.2f. Host-side only: "
            "the device cost is two event records per forward and is inside the "
            "bracket it measures. Dropped rounds (events never readable): %d.",
            self.rank,
            us_per_round,
            self._overhead_rounds,
            share,
            mean_gpu_ms,
            self._dropped_rounds,
        )
