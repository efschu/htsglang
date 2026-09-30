# SPDX-License-Identifier: Apache-2.0
"""The PP INTER-FORWARD BUBBLE: the gap between one forward and the next.

WHY THIS EXISTS, AND WHY THE EXISTING WAIT TERM COULD NOT ANSWER IT.

``RankPrefillLog`` already reports ``gpu-ms: T (compute Tc, wait Tw)``. That
``wait`` is measured by ``utils/collective_clock.py`` and sums the collective
spans **inside** one forward's event window. Group P of the Weg-2 layout runs
``tp_size=1, dcp_size=1``: it has no intra-forward collective at all, so the
term is structurally zero and stays zero however badly the pipeline stalls.
MEASURED on boot ``bsscale`` (/spinning/gpu-arb/weg2/BSSCALE_0907.md, tip
37c884b0b0): **861 of 861** ``Prefill rank batch`` lines ended at exactly
``wait 0.0``, while the same window showed PP0 at 41.7 % compute duty against
PP1 at 94.8 % -- i.e. the 5090 idle most of the window with no instrument
naming where the time went. The pipeline bubble lives BETWEEN forwards, and
``wait`` cannot see between forwards by construction.

So this is a SECOND, INDEPENDENT term, never a re-derivation of the first:

    wait_ms    device time inside a forward, spent in collectives
    bubble_ms  HOST time between the end of one forward and the start of
               the next, on this rank, with no forward running at all

They are measured by different clocks at different boundaries and are not
convertible into one another. A reader who computes one from the other has
made a category error; the test suite pins that they are different terms.

THE CLOCK. ``time.perf_counter`` at the ``_pp_launch_batch`` boundaries, not a
CUDA event. The quantity asked for is "how long did this rank hold no forward",
which is a scheduling fact on the host: the launch call returns as soon as the
work is enqueued, so a device-event pair would measure device idle, and device
idle under an async launch is not the same question. The cost is one
``perf_counter`` call per forward boundary.

DENOMINATOR, stated in the line itself (denominator law). ``share`` is
``gap_total / (gap_total + forward_total)`` over the forwards of ONE window,
and the emitted line names exactly that -- ``share=<pct>% of gap+forward``.
The only wall time this denominator excludes is the time OUTSIDE the
window's forwards: a rank asleep, at a debug stop, or waiting on the very
first request of a boot has no PREVIOUS forward to be between, so the first
forward after a reset contributes a forward span and NO gap
(``n_gaps == n_forwards - 1`` within a window), which is why both counts are
printed.

WHAT IS *NOT* EXCLUDED, and why the line has a second numerator (FIX 1r/2).
A rank that is idle BETWEEN two forwards contributes that whole idle as a
gap: ``begin`` measures from ``_last_end`` unconditionally, and it must, or
the term would not be a bubble measure at all. An earlier draft of this
docstring claimed such idle was exempt; it never was. MEASURED: two real
100 ms forwards with 30 s of no-work idle between them read
``share=99.3 %, mean=30000.0 ms, n=2``. So a raw share cannot answer the
question PP0's 41.7 %% duty poses -- a PIPELINE stall (work admitted, this
rank held anyway) and QUEUE STARVATION (nothing admitted to run) print the
same number.

The scheduler is the only thing that knows which, and it knows it at the one
site that decides: the ``if cur_batch:`` guard around ``_pp_launch_batch``.
Every event-loop visit that launches no forward calls ``note_no_batch()``, so
a gap containing at least one such visit is charged to ``starved_ms`` as
well as to ``bubble_ms``. ``bubble_ms - starved_ms`` is then the part of the
gap during which this rank had work and was not running it -- the pipeline
bubble proper. ``max=`` is printed beside ``mean=`` for the same reason: one
30 s starvation gap must not be able to hide inside a mean over 300 short
ones.
"""

from __future__ import annotations

import time
from typing import Callable, Optional, Tuple

#: Emit the per-window summary at most this often (seconds of wall clock).
#: A 15 s measurement window -- the shape every Weg-2 sweep uses -- therefore
#: carries at least two summary lines, so the share is computable from the log
#: for such a window without folding two windows into one number.
DEFAULT_WINDOW_S: float = 5.0


class PPBubbleMeter:
    """Accumulate inter-forward gaps on one rank.

    Usage is exactly two calls per forward, at the one site that runs every
    PP forward (``scheduler_pp_mixin._pp_launch_batch``)::

        meter.begin(mb_id)
        ...  # the forward
        line = meter.end()
        if line: logger.info("%s", line)

    ``begin`` returns the gap in milliseconds since the previous ``end`` (or
    ``None`` when there is no previous forward to be between) and parks it as
    the PENDING sample, which ``RankPrefillLog.record`` picks up so the same
    number reaches the per-batch line.
    """

    def __init__(
        self,
        rank: int = 0,
        clock: Callable[[], float] = time.perf_counter,
        window_s: float = DEFAULT_WINDOW_S,
    ) -> None:
        self.rank = int(rank)
        self._clock = clock
        self._window_s = float(window_s)
        # Boundary state.
        self._last_end: Optional[float] = None
        self._begun: Optional[float] = None
        # Window accumulators.
        self._gap_s = 0.0
        self._forward_s = 0.0
        self._n_forwards = 0
        self._n_gaps = 0
        self._window_start: Optional[float] = None
        # The sample RankPrefillLog picks up: (gap_ms, mb_id).
        self._pending: Optional[Tuple[float, int]] = None
        # The starvation half of the gap, and the largest single gap. Both
        # exist because a share alone cannot tell a pipeline stall from
        # queue starvation, and a mean alone cannot tell one long gap from
        # many short ones (FIX 1r/2).
        self._starved_s = 0.0
        self._max_gap_s = 0.0
        #: Visits to the event loop since the last forward ended that had
        #: nothing to launch. Set by ``note_no_batch``, consumed by the next
        #: ``begin``: it classifies the gap that is closing, never a later one.
        self._no_batch_visits = 0

    # -- boundaries ---------------------------------------------------------

    def note_no_batch(self) -> None:
        """One event-loop visit on this rank launched no forward.

        Called from the ``else`` of the ``if cur_batch:`` guard around
        ``_pp_launch_batch`` -- the one place the scheduler decides not to
        run. It is a CLASSIFIER, not a timer: it carries no duration and only
        marks the gap currently open as containing queue starvation, so the
        next ``begin`` can charge that gap to ``starved_ms`` as well as to
        ``bubble_ms``. A boot that never calls it prints ``starved_ms=0.0``
        and the bubble total is unchanged.
        """
        self._no_batch_visits += 1

    def begin(self, mb_id: int) -> Optional[float]:
        now = self._clock()
        if self._window_start is None:
            self._window_start = now
        gap_ms: Optional[float] = None
        if self._last_end is not None:
            gap_s = now - self._last_end
            # A non-monotonic reading (a clock stand-in in a test, a rank
            # whose previous end was never recorded because the forward
            # raised) is dropped rather than accumulated as a negative gap.
            if gap_s >= 0.0:
                self._gap_s += gap_s
                self._n_gaps += 1
                if gap_s > self._max_gap_s:
                    self._max_gap_s = gap_s
                if self._no_batch_visits:
                    # This rank visited the loop with nothing to launch while
                    # the gap was open: queue starvation, not a pipeline
                    # stall. Charged to BOTH terms on purpose -- starved_ms
                    # is a part of bubble_ms, never a rival total.
                    self._starved_s += gap_s
                gap_ms = gap_s * 1000.0
        self._no_batch_visits = 0
        self._begun = now
        self._pending = None if gap_ms is None else (gap_ms, int(mb_id))
        return gap_ms

    def end(self) -> Optional[str]:
        """Close the forward. Returns the window summary line when one is due."""
        now = self._clock()
        if self._begun is not None:
            span = now - self._begun
            if span >= 0.0:
                self._forward_s += span
                self._n_forwards += 1
        self._begun = None
        self._last_end = now
        return self._window_summary(now)

    # -- the pending per-batch sample ---------------------------------------

    def take_pending(self) -> Optional[Tuple[float, int]]:
        """The gap that preceded the forward currently in flight, once.

        Taken rather than read: a record that folds no forward must not
        inherit the previous forward's gap, which is the carry-forward defect
        the ``last_gpu_ms`` fields of ``RankPrefillLog`` document (#363/8b).
        """
        out = self._pending
        self._pending = None
        return out

    # -- the window ---------------------------------------------------------

    @property
    def share(self) -> Optional[float]:
        """``gap / (gap + forward)`` for the open window, or None if empty."""
        busy = self._gap_s + self._forward_s
        if busy <= 0.0 or self._n_forwards == 0:
            return None
        return self._gap_s / busy

    @property
    def mean_gap_ms(self) -> Optional[float]:
        if self._n_gaps == 0:
            return None
        return self._gap_s * 1000.0 / self._n_gaps

    @property
    def max_gap_ms(self) -> float:
        """The largest single gap in the open window."""
        return self._max_gap_s * 1000.0

    @property
    def starved_ms(self) -> float:
        """The part of the gap total that contained a nothing-to-launch visit.

        A SUBSET of the bubble total, not a rival to it: subtract it to get
        the pipeline stall, the term a PP cut can move.
        """
        return self._starved_s * 1000.0

    def summary_line(self) -> Optional[str]:
        share = self.share
        mean = self.mean_gap_ms
        if share is None:
            return None
        return (
            "PP-BUBBLE rank=%d share=%.1f%% of gap+forward, mean=%.1f ms, "
            "max=%.1f ms, n=%d (n_gaps=%d, forward_ms=%.1f, bubble_ms=%.1f, "
            "starved_ms=%.1f; bubble = host time BETWEEN forwards, a "
            "different term from the 'wait' inside one; starved = the part "
            "of it in which this rank visited the loop with nothing to "
            "launch, so bubble_ms - starved_ms is the pipeline stall)"
            % (
                self.rank,
                share * 100.0,
                0.0 if mean is None else mean,
                self.max_gap_ms,
                self._n_forwards,
                self._n_gaps,
                self._forward_s * 1000.0,
                self._gap_s * 1000.0,
                self.starved_ms,
            )
        )

    def _window_summary(self, now: float) -> Optional[str]:
        if self._window_start is None:
            return None
        if now - self._window_start < self._window_s:
            return None
        line = self.summary_line()
        self._reset_window(now)
        return line

    def _reset_window(self, now: float) -> None:
        self._gap_s = 0.0
        self._forward_s = 0.0
        self._n_forwards = 0
        self._n_gaps = 0
        self._starved_s = 0.0
        self._max_gap_s = 0.0
        self._window_start = now
