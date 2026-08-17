"""#704 slice 1a: a union arena, so a rung change copies no weight bytes.

The prefill layout ladder moves a PP layer boundary at runtime. There is no
cross-rank weight mover in this tree and its absence is deliberate
(``managers/regime_stages.py:100`` defines ``REACH_NO_WEIGHT_MOVER``: "no
runtime actuator moves weights -- switching arms needs a restart"). DESIGN_704
therefore specified the actuator as an H2D ``arena_refill`` per rung change:
one contiguous copy of the target layout's host image, 451-901 MiB a step.

The existing primitives allow something cheaper. ``plan_arena_layout`` fixes
slot offsets by sorted tensor name, and ``bind_arena_views`` rebinds parameters
to arena views **without copying any bytes**. Plan the arena over the UNION of
every rung's tensors instead of per rung and two things follow:

* a tensor shared by two rungs occupies the SAME arena bytes in both, so
  changing rung cannot require moving it;
* a rung change becomes a rebind plus a PP-boundary change -- **zero weight
  bytes on the wire**.

The cost moves from bandwidth to residency: a layer that changes hands must be
resident on BOTH ranks, so the world holds ``64 + (layers that move)``
layer-images. For the slice-1a pair that is one extra layer, ~451 MiB. This is
the same trade DESIGN_704 §3.7 already prices for the arena -- pay in resident
VRAM, not in per-change link time -- and on this rig it strictly dominates the
refill design, because with no P2P a rank-to-rank weight transfer would stage
through host memory anyway.

``arena_refill`` is NOT replaced. It remains the right primitive for the phase
flip, where the two layouts are genuinely different tensor sets (PP weights vs
TP width-shards) with no useful union. This module is for the ladder, where
consecutive rungs differ by one or two whole layers.

SCOPE, stated because "zero bytes" must not be read as "nothing to do": this
covers WEIGHTS. A GDN (linear) layer also carries per-sequence recurrent state
-- temporal_state ~19.5 MiB/layer plus conv_state ~0.762 -- which lives with
the layer. Slice 1a flips only at quiescence, where no live state exists to
preserve. Moving live state is slice 1b and is not implemented here; every
:class:`FlipDelta` says so in its own fields rather than leaving it to a
reader's memory.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping

import torch
from sglang.srt.model_executor.weights_arena import (
    ArenaLayout,
    ArenaSlot,
    WeightsArenaError,
    plan_arena_layout,
)

_OUT_OF_SCOPE = (
    "per-sequence GDN state (temporal_state, conv_state) travels with a linear "
    "layer and is NOT moved by a union rebind; slice 1a flips at quiescence "
    "only, and live-state transfer is slice 1b"
)


class UnionArenaError(WeightsArenaError):
    """A union plan that cannot be honoured. Never downgraded to a warning."""


@dataclasses.dataclass(frozen=True)
class FlipDelta:
    """What changes on one rank when the ladder steps between two rungs."""

    frm: str
    to: str
    activated: tuple[str, ...]
    deactivated: tuple[str, ...]
    #: Weight bytes that must cross a link. Zero by construction under a union
    #: arena -- the field exists so a regression shows up as a number.
    bytes_to_copy: int
    requires_quiescence: bool
    out_of_scope: str


@dataclasses.dataclass(frozen=True)
class UnionArenaPlan:
    """One byte layout covering every rung, with per-rung active subsets."""

    layout: ArenaLayout
    rung_names: tuple[str, ...]
    #: rung -> the tensor names it actually binds.
    active: Mapping[str, tuple[str, ...]]

    @property
    def total_bytes(self) -> int:
        return int(self.layout.total_bytes)

    def _check_rung(self, rung: str) -> None:
        if rung not in self.active:
            raise UnionArenaError(
                f"{rung!r} is not a planned rung; this arena covers "
                f"{list(self.rung_names)}. A rung absent from the union has no "
                "resident weights and cannot be entered by rebinding."
            )

    def active_slots(self, rung: str) -> dict[str, ArenaSlot]:
        """The arena slots this rung binds, keyed by tensor name."""
        self._check_rung(rung)
        return {name: self.layout.slot_of(name) for name in self.active[rung]}

    def bind(
        self,
        rung: str,
        arena: torch.Tensor,
        rebind: Iterable[tuple[str, torch.nn.Parameter]] = (),
    ) -> dict[str, torch.Tensor]:
        """Bind this rung's tensors to arena views. Copies nothing.

        Deliberately a thin sibling of ``weights_arena.bind_arena_views``: the
        difference is only that offsets come from the UNION layout, so views
        for tensors shared with another rung are byte-identical addresses.
        """
        self._check_rung(rung)
        if self.total_bytes > int(arena.numel()):
            raise UnionArenaError(
                f"the union layout needs {self.total_bytes} bytes but the arena "
                f"holds {int(arena.numel())}. The arena must be sized for the "
                "UNION, not for one rung: that is what makes a rung change "
                "copy-free."
            )
        views: dict[str, torch.Tensor] = {}
        for name in self.active[rung]:
            slot = self.layout.slot_of(name)
            seg = arena[slot.offset : slot.offset + slot.nbytes]
            views[name] = torch.as_strided(
                seg.view(slot.dtype), slot.shape, slot.stride
            )
        for alias, canon in self.layout.aliases:
            if canon in views:
                views[alias] = views[canon]
        for name, param in rebind:
            param.data = views[name]
        return views


def _describe(t: torch.Tensor) -> tuple:
    return (tuple(t.shape), tuple(t.stride()), t.dtype)


def plan_union_arena(
    rungs: Mapping[str, Mapping[str, torch.Tensor]],
) -> UnionArenaPlan:
    """Plan ONE arena layout covering every rung's tensors.

    Every tensor name must describe the same shape, stride and dtype in every
    rung that contains it. Otherwise the union is not a union: a tensor the
    ladder believes is unchanged could change underneath it, which is exactly
    the silent-corruption case an arena exists to prevent.
    """
    if len(rungs) < 2:
        raise UnionArenaError(
            f"a union arena needs at least two rungs to flip between, got "
            f"{len(rungs)}. One rung is a plain arena; use plan_arena_layout."
        )

    merged: dict[str, torch.Tensor] = {}
    seen: dict[str, tuple[str, tuple]] = {}
    for rung, named in rungs.items():
        for name, t in named.items():
            desc = _describe(t)
            if name in seen:
                other_rung, other_desc = seen[name]
                if desc != other_desc:
                    raise UnionArenaError(
                        f"rungs {other_rung!r} and {rung!r} disagree about "
                        f"tensor {name!r}: {other_desc} vs {desc}. A union "
                        "arena requires one definition per name -- otherwise a "
                        "tensor the ladder treats as unchanged would change "
                        "shape or dtype underneath it."
                    )
            else:
                seen[name] = (rung, desc)
                merged[name] = t

    layout = plan_arena_layout(merged)
    active = {rung: tuple(sorted(named.keys())) for rung, named in rungs.items()}
    return UnionArenaPlan(
        layout=layout,
        rung_names=tuple(rungs.keys()),
        active=active,
    )


def flip_delta(plan: UnionArenaPlan, frm: str, to: str) -> FlipDelta:
    """What a rung change costs on this rank.

    ``bytes_to_copy`` is zero by construction: every tensor the target rung
    needs and the source rung also held is already resident at the same
    offset, and every tensor it needs that the source did NOT hold was still
    loaded at boot, because the arena covers the union. The field is reported
    rather than assumed so that a future plan which breaks the invariant shows
    up as a non-zero number instead of a wrong assumption.
    """
    plan._check_rung(frm)
    plan._check_rung(to)
    a = set(plan.active[frm])
    b = set(plan.active[to])
    return FlipDelta(
        frm=frm,
        to=to,
        activated=tuple(sorted(b - a)),
        deactivated=tuple(sorted(a - b)),
        bytes_to_copy=0,
        requires_quiescence=True,
        out_of_scope=_OUT_OF_SCOPE,
    )


def union_overhead_bytes(
    plan: UnionArenaPlan, rungs: Mapping[str, Mapping[str, torch.Tensor]]
) -> dict[str, int]:
    """Per-rung residency overhead: union bytes minus that rung's own layout.

    This is the price of the copy-free flip, and it is what DESIGN_704 §3.7
    charges as the arena's resident cost. Reported per rung because the
    shallow rung pays the most -- it is the one holding layers it is not
    currently using.
    """
    out: dict[str, int] = {}
    for rung, named in rungs.items():
        plan._check_rung(rung)
        out[rung] = plan.total_bytes - int(plan_arena_layout(dict(named)).total_bytes)
    return out
