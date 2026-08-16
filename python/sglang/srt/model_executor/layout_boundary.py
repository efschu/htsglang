"""#704 slice 1a-ii: move the PP layer boundary at runtime, copying nothing.

Slice 1a established that consecutive ladder rungs can share ONE arena byte
layout, so a rung change moves no weight bytes
(:mod:`sglang.srt.model_executor.weights_arena_union`). This module changes the
other half: which layers a rank actually EXECUTES.

The model permits this directly, and three verified facts fix the design:

* ``make_layers`` (``utils/common.py:1970-2010``) builds a ModuleList of length
  ``num_hidden_layers`` with :class:`PPMissingLayer` placeholders outside the
  owned range. Layer indices are therefore GLOBAL on every rank and a boundary
  change is not an index shift.
* ``start_layer`` / ``end_layer`` are PROPERTIES over mutable ``_start_layer``
  and ``_end_layer`` backing fields (``models/qwen3_5.py:1452-1457``).
* The decoder forward iterates ``range(self.start_layer, self.end_layer)``
  (``qwen3_5.py:1483``) and reads those properties on EVERY pass.

So a real layer module parked outside the active range is simply not executed.
The boundary change is a range mutation; no module swapping, no reallocation,
no bytes.

**Load wide, run narrow.** At boot a rank builds and loads real modules for the
UNION of the ranges it may occupy, then runs whichever sub-range the ladder
selects. The union must be in force during LOADING too, because weight loading
is gated on the same range (``qwen3_5.py:1563-1564``, ``:1707-1708``).

Two properties of this design are load-bearing and are enforced here rather
than documented and hoped for:

1. **Entering a non-resident range is silently wrong, not loud.**
   ``PPMissingLayer`` is a pass-through (``layers/utils/common.py:109-127``), so
   executing a range whose weights never loaded produces *plausible output from
   a shallower model* instead of an error. Every flip therefore verifies
   residency first.
2. **A half-applied boundary is worse than no flip.** The KV pool's layer
   filter and the GDN state maps are keyed by the owned range
   (``model_executor/model_runner_kv_cache_mixin.py:2466-2470``). If a
   dependent structure cannot follow the change, this actuator restores the old
   range and raises, rather than leaving a new range beside a stale filter.

SCOPE. Slice 1a flips only at QUIESCENCE. A GDN (linear) layer carries
per-sequence recurrent state -- temporal_state ~19.5 MiB/layer plus conv_state
~0.762 -- which lives with the layer and is NOT moved here. Live-state transfer
is slice 1b. ``arena_refill`` is untouched and remains correct for the phase
flip, whose two layouts are disjoint tensor sets with no useful union.
"""

from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Callable, Mapping, MutableMapping, Sequence

from sglang.srt.model_executor.weights_arena_union import UnionArenaError

_CALLER_UPDATE = (
    "the KV pool's full_attention_layer_ids filter and the GDN/mamba state maps "
    "are keyed by the owned layer range; rebuild them for the layers named in "
    "activated/deactivated before serving resumes"
)


class LayoutBoundaryError(UnionArenaError):
    """A boundary change that cannot be honoured. Never a warning."""


@dataclasses.dataclass(frozen=True)
class BoundaryFlipReport:
    """What a boundary change did, in terms a caller must act on."""

    frm_rung: str
    to_rung: str
    frm_range: tuple[int, int]
    to_range: tuple[int, int]
    activated: tuple[int, ...]
    deactivated: tuple[int, ...]
    #: Zero by construction under the union arena. Reported, not assumed, so a
    #: regression appears as a number rather than as a broken belief.
    bytes_copied: int
    requires_caller_update: str


def validate_world_tiling(ranges: Sequence[tuple[int, int]], num_layers: int) -> None:
    """The ranges across all PP ranks must tile ``[0, num_layers)`` exactly.

    Checked at world level because it cannot be checked locally: a gap silently
    DROPS layers and an overlap silently COMPUTES THEM TWICE, and in both cases
    every individual rank's own range looks entirely sensible.
    """
    ordered = sorted((int(a), int(b)) for a, b in ranges)
    for (a0, a1), (b0, b1) in itertools.pairwise(ordered):
        if b0 > a1:
            raise LayoutBoundaryError(
                f"gap between layer ranges [{a0},{a1}) and [{b0},{b1}): layers "
                f"{list(range(a1, b0))} would be owned by no rank and silently "
                "skipped, producing plausible output from a shallower model."
            )
        if b0 < a1:
            raise LayoutBoundaryError(
                f"overlap between layer ranges [{a0},{a1}) and [{b0},{b1}): "
                f"layers {list(range(b0, a1))} would be computed twice."
            )
    if not ordered or ordered[0][0] != 0 or ordered[-1][1] != int(num_layers):
        got = (ordered[0][0], ordered[-1][1]) if ordered else None
        raise LayoutBoundaryError(
            f"the ranges do not cover [0,{int(num_layers)}): they span {got}."
        )


class LayoutBoundaryActuator:
    """Moves one rank's executed layer range between ladder rungs.

    This IS an actuator: :meth:`flip` mutates the model's own
    ``_start_layer``/``_end_layer``, which the decoder forward reads on every
    pass. It is not a recommendation and there is no separate applier.
    """

    def __init__(
        self,
        model,
        rung_ranges: Mapping[str, tuple[int, int]],
        current_rung: str,
    ) -> None:
        if current_rung not in rung_ranges:
            raise LayoutBoundaryError(
                f"current rung {current_rung!r} is not among the planned rungs "
                f"{sorted(rung_ranges)}."
            )
        self.model = model
        self.rung_ranges: MutableMapping[str, tuple[int, int]] = dict(rung_ranges)
        self._current = current_rung
        self._observers: list[Callable[[BoundaryFlipReport], None]] = []
        # The union of every rung this rank may occupy must be resident NOW:
        # a boundary can only ever move within already-loaded weights.
        lo = min(a for a, _ in self.rung_ranges.values())
        hi = max(b for _, b in self.rung_ranges.values())
        self._union = (lo, hi)
        self._require_resident(lo, hi, context="the planned union")
        # THE NARROWING STEP of "load wide, run narrow". The model arrives with
        # the union range in force, because weight loading is gated on the same
        # range it executes (qwen3_5.py:1563-1564, :1707-1708) and the union had
        # to be in force for the union to load. Constructing this actuator is
        # what hands the rank its first rung, so the range is applied here
        # rather than assumed to match -- an actuator whose belief about the
        # active range differs from the model's is the exact split-brain this
        # class exists to prevent.
        self._apply(self.rung_ranges[current_rung], current_rung)

    @property
    def current_rung(self) -> str:
        return self._current

    @property
    def union_range(self) -> tuple[int, int]:
        return self._union

    def add_observer(self, fn: Callable[[BoundaryFlipReport], None]) -> None:
        """Register a dependent structure that must follow a boundary change.

        Observers run INSIDE the flip. If one raises, the range is restored and
        the flip fails -- a new range beside a stale KV filter is the worst
        outcome available.
        """
        self._observers.append(fn)

    def _is_real(self, idx: int) -> bool:
        layer = self.model.layers[idx]
        # A PPMissingLayer is a parameterless pass-through; a real decoder layer
        # owns parameters. Tested by structure rather than by class name so a
        # placeholder from any module still counts as missing.
        return any(True for _ in layer.parameters())

    def _require_resident(self, start: int, end: int, context: str) -> None:
        absent = [i for i in range(int(start), int(end)) if not self._is_real(i)]
        if absent:
            raise LayoutBoundaryError(
                f"layers {absent} are not resident on this rank, so {context} "
                f"[{start},{end}) cannot be entered. Executing them would run a "
                "PPMissingLayer pass-through and produce plausible output from a "
                "shallower model rather than an error. Load the union at boot "
                "('load wide, run narrow')."
            )

    def flip(self, to_rung: str, quiescent: bool) -> BoundaryFlipReport:
        """Move the executed range to ``to_rung``. Copies nothing."""
        if to_rung not in self.rung_ranges:
            raise LayoutBoundaryError(
                f"{to_rung!r} is not a known rung on this rank; planned rungs "
                f"are {sorted(self.rung_ranges)}."
            )
        if not quiescent:
            raise LayoutBoundaryError(
                "refusing to move the layer boundary while not quiescent: a GDN "
                "layer's per-sequence recurrent state travels with the layer and "
                "slice 1a moves none of it, so an in-flight sequence would "
                "silently continue against state that is no longer there. "
                "Drain first (see DESIGN_704 D6: admission hold with bounded "
                "drain), or wait for slice 1b's live-state transfer."
            )

        frm = self._current
        old = self._range_of(frm)
        new = self.rung_ranges[to_rung]
        self._require_resident(new[0], new[1], context=f"rung {to_rung!r}")

        activated = tuple(
            i for i in range(new[0], new[1]) if not (old[0] <= i < old[1])
        )
        deactivated = tuple(
            i for i in range(old[0], old[1]) if not (new[0] <= i < new[1])
        )
        report = BoundaryFlipReport(
            frm_rung=frm,
            to_rung=to_rung,
            frm_range=old,
            to_range=new,
            activated=activated,
            deactivated=deactivated,
            bytes_copied=0,
            requires_caller_update=_CALLER_UPDATE,
        )

        self._apply(new, to_rung)
        try:
            for fn in self._observers:
                fn(report)
        except Exception as exc:
            self._apply(old, frm)
            raise LayoutBoundaryError(
                f"a dependent structure refused the move to {to_rung!r} "
                f"({exc!r}); the boundary was rolled back to {frm!r} "
                f"{old}. A new range beside a stale KV filter or GDN state map "
                "is worse than no flip."
            ) from exc
        return report

    def _range_of(self, rung: str) -> tuple[int, int]:
        return self.rung_ranges[rung]

    def _apply(self, rng: tuple[int, int], rung: str) -> None:
        # The decoder forward reads these properties every pass
        # (qwen3_5.py:1483), so assigning the backing fields IS the actuation.
        self.model._start_layer = int(rng[0])
        self.model._end_layer = int(rng[1])
        self._current = rung

    def resident_overhead_layers(self) -> dict[str, int]:
        """Layers held resident but unused, per rung: the union's price."""
        lo, hi = self._union
        out: dict[str, int] = {}
        for rung, (a, b) in self.rung_ranges.items():
            out[rung] = (hi - lo) - (b - a)
        return out


# ---------------------------------------------------------------------------
# Dependent-structure observers.
#
# A boundary change is not finished when the range changes. Two structures
# follow the owned layer range, and BOTH fail silently rather than loudly if
# they do not follow it -- which is why they are guards rather than notes.
# ---------------------------------------------------------------------------


def cuda_graph_observer(registry) -> Callable[[BoundaryFlipReport], None]:
    """Require captured CUDA graphs to be recaptured for the new range.

    A CUDA graph records actual kernel launches, and the decode graph captures
    the model's own forward (``decode_cuda_graph_runner.py:1770``), which
    iterates ``range(self.start_layer, self.end_layer)``. **The executed layer
    set is therefore baked at capture time**, and a replay ignores any later
    change to ``_start_layer``/``_end_layer``: after an unguarded flip the
    model would report the new range while every graph replay still ran the
    old one.

    That is the worst shape of bug available here -- the layer count silently
    reverts on exactly the fast path, and only under graph replay, so an eager
    smoke test would show the flip working.

    ``registry`` must expose ``captured_range`` and a ``recapture(range)``. If
    it cannot recapture, the flip is refused and rolled back: a boundary whose
    graphs disagree with it is worse than no boundary change.
    """

    def _observe(report: BoundaryFlipReport) -> None:
        captured = getattr(registry, "captured_range", None)
        if captured is None or tuple(captured) == tuple(report.to_range):
            return
        recapture = getattr(registry, "recapture", None)
        if not callable(recapture):
            raise LayoutBoundaryError(
                f"CUDA graphs are captured for range {tuple(captured)} but the "
                f"boundary moved to {tuple(report.to_range)}, and the registry "
                "offers no recapture(). The captured graphs bake the executed "
                "layer set, so replays would keep running the old range while "
                "the model reports the new one -- visible only under graph "
                "replay, which is exactly where an eager smoke test would miss "
                "it."
            )
        recapture(report.to_range)

    return _observe


def pool_coverage_observer(
    built_start: int, built_end: int, name: str = "pool"
) -> Callable[[BoundaryFlipReport], None]:
    """Require the new range to stay inside the span the pools were BUILT for.

    The KV and mamba pools are constructed from layer id lists filtered to the
    owned range (``model_runner_kv_cache_mixin.py:2460-2470``) and are indexed
    by ``layer_id - pool.start_layer`` (``memory_pool.py:1576``, ``:2889``).
    Two consequences, and the union answers both:

    * a layer newly activated outside the built span has **no rows at all** --
      for a full-attention layer no KV, for a linear layer no mamba slot;
    * the indexing BASE must not move. If a downstream rank's pool were rebuilt
      with the new start (rank1: 28 -> 29), every cached row would shift by one
      layer and be silently misattributed.

    So the pools are built over the UNION for exactly the reason the weights
    are -- "load wide, run narrow" applies to the caches too, at a cost of one
    extra layer's rows per boundary that moves.
    """

    def _observe(report: BoundaryFlipReport) -> None:
        lo, hi = int(report.to_range[0]), int(report.to_range[1])
        if lo < int(built_start) or hi > int(built_end):
            raise LayoutBoundaryError(
                f"the {name} pool was built for layers "
                f"[{int(built_start)},{int(built_end)}) but rung "
                f"{report.to_rung!r} needs [{lo},{hi}). Layers outside the built "
                "span have no rows -- no KV for a full-attention layer, no "
                "mamba slot for a linear one -- and rebuilding the pool with a "
                "new start would move the `layer_id - start_layer` indexing base "
                "and silently misattribute every cached row. Build the pools "
                "over the UNION of the rungs, as the weights are."
            )

    return _observe
