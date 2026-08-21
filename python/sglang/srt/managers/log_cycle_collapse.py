# Copyright 2026 SGLang Team
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
"""#788: collapse a per-pass instrument that has started repeating itself.

THE SPECIMEN. Boot instr14 wrote 51212468 bytes in 9m59s, peaking at
13.03 MB/min under burst load, and two per-pass instruments were 89% of it::

    141513 lines  #788 PP-ADMISSION verdict=... (20367381 B)
    140184 lines  PARKED-DECODE carriers ...    (25373304 B)
      9600 lines  everything else               ( 5471783 B)

WHY PLAIN RUN-LENGTH COLLAPSE IS NOT ENOUGH, which is the whole reason this
module is a cycle detector and not an ``if key == last_key``. Measured on
instr14, CONSECUTIVE LINES ARE ALMOST NEVER IDENTICAL. The scheduler cycles
through a small number of states and the instrument faithfully reports each
one, so the stream is periodic rather than constant::

    PARKED-DECODE carriers 4 parked (+4 -2) of 4 resident; ...
    PARKED-DECODE carriers 2 parked (+2 -4) of 2 resident; ...
    PARKED-DECODE carriers 2 parked (+2 -2) of 2 resident; ...
    PARKED-DECODE carriers 4 parked (+4 -2) of 4 resident; ...   <- period 3

    #788 PP-ADMISSION verdict=DECLINE ... queue=1 running=2 chunked=0
    #788 PP-ADMISSION verdict=DECLINE ... queue=1 running=4 chunked=0
    #788 PP-ADMISSION verdict=DECLINE ... queue=1 running=2 chunked=0   <- period 2

Replaying instr14 through a ``max_period=1`` collapser -- exactly "collapse
consecutive identical payloads" -- removes NOTHING: 13581 admission lines in
the 5 MB tail become 9073 lines plus 4508 roll-ups, and the 13545 parked
receipts become 13542 plus 3. The run lengths are 1. Admitting period up to
3 is what turns the same tail into 71+24 and 39+21 lines respectively.

THE PREDICATE. A pass is redundant iff the last ``2p`` observed keys are two
identical halves, for some period ``p`` in ``1..max_period``; the smallest
such ``p`` wins. So a cycle is printed IN FULL, TWICE, before a single line
of it is suppressed -- the reader has seen every state and the exact order
they occur in before the instrument goes quiet about them. Anything the
cycle does not already contain breaks the match and prints immediately.

THE PROPERTY THIS MUST KEEP, and the reason ``observe`` takes an opaque key
and nothing else. The #788 admission trace exists to prove RANK CONGRUENCE:
the acceptance gate (evidence-665-f1/verdict_790.sh, step 3) diffs the
emitted payloads across PP0/PP1/PP2 and requires one group per event. That
diff is only meaningful if every rank decides to speak or stay silent from
data every rank has identically. The verdict is therefore a PURE FUNCTION OF
THE OBSERVED KEY SEQUENCE -- never wall-clock time, never a per-rank
counter, never log volume, never a random sample -- and the caller is
responsible for building keys out of CONGRUENT payload fields only. A
consecutive-repeat collapse is congruent precisely because the payload is
congruent, so the run length is too. If rank 0 suppressed a line rank 1
emitted, the gate would report a divergence that never happened, and an
instrument that manufactures false positives is worse than no instrument.

SILENCE MUST NOT BE AMBIGUOUS. The collapser never simply goes quiet: it
reports a suppressed COUNT to the caller both every ``rollup_every``
suppressed passes and as a flush in front of the next emitted line, so a
reader can always separate "nothing changed" from "the instrument died". The
cadence is a COUNT, not a duration, for the same congruence reason: an
iteration counter advances identically on every rank, a clock does not.

NOTHING HERE IMPORTS TORCH. Keys are ordinary hashables and the whole module
is testable on CPU in one process with no CUDA present.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Hashable, NamedTuple, Optional

__all__ = [
    "CYCLE_COLLAPSE_MAX_PERIOD",
    "CYCLE_COLLAPSE_ROLLUP_EVERY",
    "CollapseVerdict",
    "CycleCollapse",
]

#: Longest repeating cycle the collapser will recognise. Measured against
#: instr14, whose two floods have period 3 and period 2; 8 leaves headroom
#: for a recipe with more scheduler states without making the window
#: expensive -- detection is O(max_period^2) over at most 2*max_period
#: cached keys, i.e. a few dozen comparisons per pass.
#:
#: Raising this trades latency-to-first-report for compression: a cycle of
#: period p costs 2p printed lines before it is recognised.
CYCLE_COLLAPSE_MAX_PERIOD = 8

#: How many consecutive redundant passes are swallowed before a roll-up is
#: reported. A COUNT, never a duration: a wall-clock cadence would make the
#: emission points rank-local and destroy the congruence property above.
CYCLE_COLLAPSE_ROLLUP_EVERY = 1024


class CollapseVerdict(NamedTuple):
    """What the caller should do about one observed pass.

    ``rollup`` is reported FIRST when both it and ``emit`` are set: the
    swallowed passes happened before the line that flushed them.
    """

    #: True when this pass is not a repetition and must be logged.
    emit: bool
    #: Passes swallowed since the last report, or 0 when there is nothing to
    #: report. Non-zero means the caller must log a roll-up naming it.
    rollup: int
    #: Period of the cycle the ``rollup`` count belongs to, so the roll-up
    #: line can say what it was repeating. 0 when ``rollup`` is 0.
    period: int
    #: Key of the LAST SUPPRESSED pass, so the roll-up can name what it was
    #: repeating rather than describing the line that flushed it. None when
    #: ``rollup`` is 0.
    last: Optional[Any] = None


class CycleCollapse:
    """Suppresses passes that repeat a cycle already printed in full twice.

    Stateful and single-threaded, one instance per instrument per process.
    """

    def __init__(
        self,
        max_period: int = CYCLE_COLLAPSE_MAX_PERIOD,
        rollup_every: int = CYCLE_COLLAPSE_ROLLUP_EVERY,
    ) -> None:
        if max_period < 1:
            raise ValueError(f"max_period must be >= 1, got {max_period}")
        if rollup_every < 1:
            raise ValueError(f"rollup_every must be >= 1, got {rollup_every}")
        self.max_period = int(max_period)
        self.rollup_every = int(rollup_every)
        # Exactly the window the predicate reads: two cycles of the longest
        # period recognised. Bounded on purpose -- an instrument's memory
        # must not grow with the run it is measuring.
        self._history: Deque[Hashable] = deque(maxlen=2 * self.max_period)
        self._run = 0
        self._run_period = 0
        self._run_key: Optional[Any] = None

    def _repeat_period(self) -> int:
        """Smallest p in 1..max_period whose doubled cycle ends the window.

        Smallest wins so a period-1 stretch inside a period-4 recipe is
        still reported as the constant it is.
        """
        history = self._history
        n = len(history)
        for period in range(1, self.max_period + 1):
            if n < 2 * period:
                break
            base = n - 2 * period
            if all(
                history[base + i] == history[base + period + i] for i in range(period)
            ):
                return period
        return 0

    def observe(self, key: Hashable) -> CollapseVerdict:
        """Record one pass and say whether it should be logged."""
        self._history.append(key)
        period = self._repeat_period()
        if period:
            self._run += 1
            self._run_period = period
            self._run_key = key
            if self._run >= self.rollup_every:
                run, self._run = self._run, 0
                return CollapseVerdict(emit=False, rollup=run, period=period, last=key)
            return CollapseVerdict(emit=False, rollup=0, period=0, last=None)
        run, self._run = self._run, 0
        run_period, self._run_period = self._run_period, 0
        run_key, self._run_key = self._run_key, None
        if not run:
            return CollapseVerdict(emit=True, rollup=0, period=0, last=None)
        return CollapseVerdict(emit=True, rollup=run, period=run_period, last=run_key)
