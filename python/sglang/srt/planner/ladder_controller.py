"""#704 part A: the rung controller.

Turns a solved ladder plus a live-fill reading into rung changes. The failure
mode this class exists to prevent is not a wrong rung, it is a CHATTERING rung:
a fill level parked near a boundary that drives a layout change every step,
each one moving 451-901 MiB of weights over PCIe.

Three rules, all of them consequences of the solved ladder rather than choices
made here:

1. **The bands come from the solver.** A step fires against
   ``Transition.ascend_above_tokens`` / ``descend_below_tokens``, which were
   derived from the measured link reach and fill rate. This class contains no
   threshold of its own.
2. **Only a solved step may fire.** ``solve_layout_ladder`` prunes pairs whose
   bands cross or fall below zero fill. A pruned pair is not a step, and the
   controller cannot route around it -- so a ladder with no fundable
   transitions pins the layout, which is the honest outcome.
3. **One rung per observation, at a chunk boundary.** Multi-rung leaps would
   move several layers at once and blow the budget the hysteresis was derived
   against. A decision taken while a chunked prefill is mid-sequence is held
   pending, not dropped and not applied mid-chunk.

STATUS: this is the decision function. It is deliberately pure -- it returns a
target rung and mutates only its own index. It does NOT move weights, and
nothing is wired to it yet; the actuator is Slice 1. Stated explicitly because
a decision function that is mistaken for an actuator is exactly the
counter-versus-actuator confusion that cost this project five bugs.
"""

from __future__ import annotations

from sglang.srt.planner.layout_ladder import Ladder, Rung


class LadderController:
    """Selects a rung from live fill. Holds no thresholds of its own."""

    def __init__(self, ladder: Ladder, start_index: int = 0) -> None:
        if not ladder.rungs:
            raise ValueError("a ladder with no rungs cannot be controlled.")
        self._ladder = ladder
        self._index = 0
        self.force_rung(start_index)
        self._pending: int | None = None
        # Steps are keyed by the shallower rung's index. A pair pruned by the
        # solver is simply absent, and its absence is what pins the controller.
        # Transitions cannot be zipped positionally with rungs precisely
        # because pruning makes the two lists different lengths.
        index_of = {r.counts: i for i, r in enumerate(ladder.rungs)}
        self._steps = {index_of[t.shallower]: t for t in ladder.transitions}

    @property
    def rung(self) -> Rung:
        return self._ladder.rungs[self._index]

    @property
    def index(self) -> int:
        return self._index

    def active_descend_threshold(self) -> float | None:
        """The descend boundary in force at the current rung, if any.

        Exposed because a test that jitters around an INACTIVE threshold passes
        vacuously -- a controller with its hysteresis collapsed survives it
        unchanged. The falsifier has to probe the live boundary.
        """
        step = self._steps.get(self._index)
        return None if step is None else step.descend_below_tokens

    def active_ascend_threshold(self) -> float | None:
        """The ascend boundary in force at the current rung, if any."""
        step = self._steps.get(self._index - 1)
        return None if step is None else step.ascend_above_tokens

    def force_rung(self, index: int) -> None:
        """Set the current rung directly, for boot and for tests."""
        if not 0 <= index < len(self._ladder.rungs):
            raise IndexError(
                f"rung {index} is off a ladder of {len(self._ladder.rungs)} rungs."
            )
        self._index = int(index)

    def observe(self, live_tokens: float, quiescent: bool = True) -> Rung | None:
        """Take a fill reading; return the rung to move to, or ``None``.

        ``quiescent`` is the scheduler's statement that no chunked prefill is
        mid-sequence. A decision reached while busy is remembered and committed
        at the next boundary, so pressure that builds during a long chunk is
        not silently discarded.
        """
        target = (
            self._pending if self._pending is not None else self._decide(live_tokens)
        )
        if target is None or target == self._index:
            self._pending = None
            return None
        if not quiescent:
            self._pending = target
            return None
        self._pending = None
        self._index = target
        return self._ladder.rungs[target]

    def _decide(self, live_tokens: float) -> int | None:
        """One step, or none. Ascend takes priority: it is the safe direction."""
        # Ascend toward more pool (lower index) when the current rung is
        # running out. The step between index-1 and index is keyed at index-1.
        up = self._steps.get(self._index - 1)
        if up is not None and live_tokens > up.ascend_above_tokens:
            return self._index - 1
        # Descend toward speed (higher index) only when the deeper rung can
        # hold the live set with its move window to spare.
        down = self._steps.get(self._index)
        if down is not None and live_tokens < down.descend_below_tokens:
            return self._index + 1
        return None
