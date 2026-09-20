"""Recapture the prefill CUDA graphs when the executed layer range moves.

A captured CUDA graph is a recorded list of kernel launches. The prefill
runner captures the model's own forward (``prefill_cuda_graph_runner.py:512``),
and that forward iterates ``owned_layer_ids(self.layers, self.start_layer,
self.end_layer)`` (``models/qwen3_5.py:1720``). **The executed layer set is
therefore fixed at capture time**, and a replay ignores every later change to
the boundary.

That makes an unguarded P-layout switch fail in the worst available shape: the
model reports the new range, the eager path honours it, and only the replayed
graphs -- i.e. only the fast path, i.e. almost all of production -- keep
running the old one. An eager smoke test would show the switch working.

So after a flip every captured shape is in exactly one of two states, never a
third:

* **re-recorded** under the new range, at the price of one capture; or
* **discarded**, so ``can_run`` answers False and the shape falls back to the
  eager path -- slower, and correct.

The one state this module exists to make unreachable is "still captured under
the old range and still replayable".
"""

from __future__ import annotations

import dataclasses
from typing import Iterable, Optional, Sequence

from sglang.srt.model_executor.runner.shape_key import ShapeKey


class GraphRecaptureError(RuntimeError):
    """A boundary change that the captured graphs cannot be made to follow."""


@dataclasses.dataclass(frozen=True)
class RecaptureReport:
    """What a recapture did, per shape. Numbers, not a belief."""

    frm_range: tuple[int, int]
    to_range: tuple[int, int]
    recaptured: tuple[int, ...]
    discarded: tuple[int, ...]
    seconds: Optional[float]

    def as_line(self) -> str:
        cost = "unpriced" if self.seconds is None else f"{self.seconds:.2f}s"
        return (
            f"[p-layout-recapture] {self.frm_range} -> {self.to_range}: "
            f"recaptured {len(self.recaptured)} shape(s) {list(self.recaptured)}, "
            f"discarded {len(self.discarded)} {list(self.discarded)}, cost {cost}"
        )


class PrefillGraphRecaptureRegistry:
    """The ``registry`` that :func:`layout_boundary.cuda_graph_observer` wants.

    It exposes exactly the two names that observer looks for --
    ``captured_range`` and ``recapture(range)`` -- so registering it is what
    turns that observer from a refusal into a repair.

    **Which shapes are recaptured, and why not all of them.** Every prefill
    graph bakes the range, so every captured shape is affected. What differs is
    the WORTH of paying for it: a boot captures the whole
    ``capture_num_tokens`` ladder, while a given backlog replays a handful of
    widths. ``hot_shapes`` names the ones worth re-recording; the rest are
    discarded rather than recaptured, which costs them their graph but not
    their correctness. Default is every captured shape -- the conservative
    choice, because it keeps the post-switch performance profile the one that
    was measured.

    **The cost.** One recapture is one full forward at that width, under
    capture, plus the graph's own memory. ``per_shape_seconds`` must be
    MEASURED and passed in; there is no default, and :meth:`cost_seconds`
    refuses rather than inventing one, because this number is a term in
    ``p_layout_switch.decide`` and an invented term there would decide
    switches.
    """

    def __init__(
        self,
        runner,
        captured_range: tuple[int, int],
        captured_shapes: Iterable[int],
        *,
        per_shape_seconds: Optional[float] = None,
        hot_shapes: Optional[Iterable[int]] = None,
    ):
        self.runner = runner
        self._range = (int(captured_range[0]), int(captured_range[1]))
        self._shapes = tuple(sorted({int(s) for s in captured_shapes}))
        self._per_shape_seconds = (
            None if per_shape_seconds is None else float(per_shape_seconds)
        )
        if hot_shapes is None:
            self._hot = set(self._shapes)
        else:
            hot = {int(s) for s in hot_shapes}
            unknown = sorted(hot - set(self._shapes))
            if unknown:
                raise GraphRecaptureError(
                    f"hot_shapes names {unknown}, which were never captured; "
                    f"captured shapes are {list(self._shapes)}. A shape that "
                    "was never captured has no graph to recapture and naming "
                    "it here would hide a stale ladder rather than shrink one."
                )
            self._hot = hot
        self.last_report: Optional[RecaptureReport] = None

    # -- the two names cuda_graph_observer reads -------------------------

    @property
    def captured_range(self) -> tuple[int, int]:
        return self._range

    def recapture(self, new_range) -> RecaptureReport:
        lo, hi = int(new_range[0]), int(new_range[1])
        if (lo, hi) == self._range:
            report = RecaptureReport(self._range, (lo, hi), (), (), 0.0)
            self.last_report = report
            return report
        if hi <= lo:
            raise GraphRecaptureError(
                f"cannot capture for the empty range [{lo},{hi}): a graph over "
                "no layers is a graph that computes nothing, and it would "
                "replay silently."
            )
        was = self._range
        hot = tuple(s for s in self._shapes if s in self._hot)
        cold = tuple(s for s in self._shapes if s not in self._hot)

        # Discard FIRST. If the recapture below raises, every shape is either
        # gone (eager, correct) or about to be re-recorded -- never left
        # replayable under the old range, which is the one outcome that is
        # silently wrong.
        discarded = self._discard(cold)
        self._discard(hot)
        try:
            done = self._run_recapture(hot)
        except Exception as exc:
            self._range = (lo, hi)
            raise GraphRecaptureError(
                f"recapture for range [{lo},{hi}) failed after the old graphs "
                f"were discarded: {exc!r}. Every shape is now eager, which is "
                "correct and slow; the boundary itself is the caller's to roll "
                "back."
            ) from exc
        self._range = (lo, hi)
        report = RecaptureReport(
            frm_range=was,
            to_range=(lo, hi),
            recaptured=done,
            discarded=discarded,
            seconds=self.cost_seconds_or_none(len(done)),
        )
        self.last_report = report
        return report

    # -- internals -------------------------------------------------------

    def _discard(self, sizes: Sequence[int]) -> tuple[int, ...]:
        backend = getattr(self.runner, "backend", None)
        discard = getattr(backend, "discard_shape", None)
        if not callable(discard):
            raise GraphRecaptureError(
                f"graph backend {type(backend).__name__} offers no "
                "discard_shape(), so a graph captured for the old layer range "
                "cannot be taken out of service. Without it a boundary change "
                "would leave replayable graphs running the old range while the "
                "model reports the new one. Add discard_shape() to the backend "
                "or do not move the boundary on it."
            )
        gone = []
        for size in sizes:
            if discard(ShapeKey(size=int(size))):
                gone.append(int(size))
        return tuple(gone)

    def _run_recapture(self, sizes: Sequence[int]) -> tuple[int, ...]:
        if not sizes:
            return ()
        recapture_shapes = getattr(self.runner, "recapture_shapes", None)
        if not callable(recapture_shapes):
            raise GraphRecaptureError(
                f"runner {type(self.runner).__name__} offers no "
                "recapture_shapes(); a capture must re-enter the same context "
                "stack as the boot capture (model_capture_mode, freeze_gc, "
                "graph_capture, the backend's capture_session) and that stack "
                "is the runner's to own."
            )
        return tuple(recapture_shapes(sizes))

    # -- pricing ---------------------------------------------------------

    def cost_seconds(self, n_shapes: Optional[int] = None) -> float:
        """Seconds a recapture of ``n_shapes`` costs. Refuses when unmeasured."""
        if self._per_shape_seconds is None:
            raise GraphRecaptureError(
                "per_shape_seconds was not measured, so the recapture cost "
                "cannot be priced. It is a term in the switch decision "
                "(p_layout_switch.decide), and a guessed term there does not "
                "produce a rough answer -- it produces switches taken for a "
                "reason that was never measured. Time one capture_one_shape() "
                "at boot and pass the number."
            )
        n = len(self._hot) if n_shapes is None else int(n_shapes)
        return self._per_shape_seconds * n

    def cost_seconds_or_none(self, n_shapes: Optional[int] = None):
        try:
            return self.cost_seconds(n_shapes)
        except GraphRecaptureError:
            return None

    @property
    def hot_shapes(self) -> tuple[int, ...]:
        return tuple(sorted(self._hot))

    @property
    def captured_shapes(self) -> tuple[int, ...]:
        return self._shapes


def recapture_cost_seconds(registry, new_range) -> float:
    """The recapture term for ``decide``: zero when the range does not move."""
    if tuple(registry.captured_range) == (int(new_range[0]), int(new_range[1])):
        return 0.0
    return float(registry.cost_seconds())


def no_recapture_registry(captured_range: tuple[int, int]) -> object:
    """A registry that REFUSES: for a runner whose graphs cannot be re-recorded.

    Used where the graph backend has no ``discard_shape`` (tc_piecewise, whose
    graphs are per-compiled-piece and whose recapture is a different question).
    It carries ``captured_range`` and deliberately no ``recapture``, which is
    exactly what makes ``cuda_graph_observer`` refuse the flip by name.
    """

    class _Refusing:
        def __init__(self, rng):
            self.captured_range = (int(rng[0]), int(rng[1]))

    return _Refusing(captured_range)


__all__ = [
    "GraphRecaptureError",
    "RecaptureReport",
    "PrefillGraphRecaptureRegistry",
    "recapture_cost_seconds",
    "no_recapture_registry",
]
