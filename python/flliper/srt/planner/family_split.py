"""#705 as planner RULES: family placement solved, not hand-chosen.

Binding directive (``PLAN_PERF_PIPELINE_2026-08-16``, "PLANNER-SOLVED,
UNIVERSAL"): the family-split verdict must land as an objective/constraint set
so ``solve()`` picks family ratios on any hardware. Collective costs come from
the census instruments, bandwidths from the pair matrix, capacity from the
ledger. Nothing in this module names a GPU or a model.

The rules:

* A family's BLOCKING collectives vanish iff its placement is concentrated on
  exactly ONE rank. Partial concentration buys nothing -- a two-rank shard still
  all-reduces. DEFERRED collectives never vanish (they are already latency-hidden
  by the issue/join window, #597), so a family whose collectives are all deferred
  can never repay concentration.
* Sharded time depends on the SHARD POLICY. Under an equal width shard the
  SLOWEST rank binds. Under a bandwidth-proportional shard (this fork's uneven
  TP, ``--rank-tp-ratio``) every rank finishes together and the family costs
  ``bytes / sum(bandwidth)``. The proportional baseline is faster, so it raises
  the bar that concentration must clear -- the honest baseline is the best one
  the hardware can actually run.
* Concentration therefore pays only when it escapes a slow BINDING rank. On
  uniform hardware there is nothing to escape and the rules reject it.
* Capacity is a hard per-rank constraint. Under the TP sum rule concentration
  REDISTRIBUTES capacity (the world total is conserved); it does not consume it.
* A decision with unmeasured collective cost is REFUSED, not guessed.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, Optional, Sequence, Tuple


@dataclasses.dataclass(frozen=True)
class RankHardware:
    """One rank, from the pair matrix and the capacity ledger."""

    name: str
    bandwidth_mib_per_s: float
    capacity_mib: float


@dataclasses.dataclass(frozen=True)
class FamilySpec:
    """One layer family (e.g. linear-attention, full-attention, MoE).

    ``blocking_collectives_per_layer`` are the ones concentration can remove;
    ``deferred_collectives_per_layer`` survive any placement.
    """

    name: str
    n_layers: int
    weight_mib_per_layer: float
    state_mib_per_layer_per_slot: float
    blocking_collectives_per_layer: int
    deferred_collectives_per_layer: int

    def round_mib(self, active_slots: int) -> float:
        """Bytes streamed per decode round: all weights plus the live states."""
        return float(self.n_layers) * (
            float(self.weight_mib_per_layer)
            + float(self.state_mib_per_layer_per_slot) * float(active_slots)
        )

    def resident_mib(self, mamba_slots: int) -> float:
        """Bytes that must RESIDE: all weights plus every state slot."""
        return float(self.n_layers) * (
            float(self.weight_mib_per_layer)
            + float(self.state_mib_per_layer_per_slot) * float(mamba_slots)
        )

    @property
    def blocking_total(self) -> int:
        return int(self.n_layers) * int(self.blocking_collectives_per_layer)


@dataclasses.dataclass(frozen=True)
class CollectiveCost:
    """Per-collective cost, WITH provenance.

    ``source`` must state where the numbers came from. A cost whose components
    are ``None`` is unmeasured and the solver refuses to decide on it rather
    than substituting a plausible value.
    """

    blocking_us: Optional[float]
    deferred_us: Optional[float]
    source: str

    @property
    def is_measured(self) -> bool:
        return self.blocking_us is not None and self.deferred_us is not None


@dataclasses.dataclass(frozen=True)
class FamilyPlacement:
    name: str
    ratios: Tuple[float, ...]
    is_solo: bool
    host_rank: Optional[int]
    bandwidth_ms_sharded: float
    bandwidth_ms_solo: float
    bandwidth_ms_chosen: float
    bandwidth_delta_ms: float
    blocking_removed: int
    deferred_removed: int
    collective_saving_ms: float
    net_ms: float
    break_even_us: float


@dataclasses.dataclass(frozen=True)
class FamilySplitSolution:
    by_family: Dict[str, FamilyPlacement]
    net_ms: float
    capacity_delta_mib: Tuple[float, ...]


def bandwidth_ratios(ranks: Sequence[RankHardware]) -> Tuple[float, ...]:
    total = sum(float(r.bandwidth_mib_per_s) for r in ranks)
    if total <= 0:
        raise ValueError("total bandwidth across ranks is zero.")
    return tuple(float(r.bandwidth_mib_per_s) / total for r in ranks)


def _sharded_ms(
    mib: float, ranks: Sequence[RankHardware], ratios: Sequence[float]
) -> float:
    """The SLOWEST participating rank binds."""
    return (
        max(
            mib * float(w) / float(r.bandwidth_mib_per_s)
            for r, w in zip(ranks, ratios)
            if w > 0.0
        )
        * 1000.0
    )


def solve_family_placement(
    families: Sequence[FamilySpec],
    ranks: Sequence[RankHardware],
    collective_cost: CollectiveCost,
    mamba_slots: int,
    active_slots: int = 1,
    allow_uneven_shards: bool = True,
) -> FamilySplitSolution:
    """Choose, per family, between a shard and concentration on one rank.

    ``allow_uneven_shards`` selects the shard policy used as the BASELINE: the
    bandwidth-proportional shard this fork can actually run (default), or the
    equal width shard. The choice moves the break-even materially, so it is
    explicit rather than implied.
    """
    if not collective_cost.is_measured:
        raise ValueError(
            "refusing to solve family placement on unmeasured collective cost "
            f"(source={collective_cost.source!r}). The verdict turns entirely on "
            "the per-collective figure, so it must come from the census "
            "instruments, not from an estimate."
        )
    n = len(ranks)
    prop = bandwidth_ratios(ranks)
    base_ratios = prop if allow_uneven_shards else tuple(1.0 / n for _ in ranks)

    # Resident bytes each rank carries under the all-sharded baseline.
    base_resident = [0.0] * n
    for f in families:
        for i, w in enumerate(base_ratios):
            base_resident[i] += f.resident_mib(mamba_slots) * w

    placements: Dict[str, FamilyPlacement] = {}
    delta = [0.0] * n
    net_total = 0.0

    for f in families:
        round_mib = f.round_mib(active_slots)
        resident = f.resident_mib(mamba_slots)
        sharded_ms = _sharded_ms(round_mib, ranks, base_ratios)
        saving_ms = f.blocking_total * float(collective_cost.blocking_us) / 1000.0

        best: Optional[Tuple[float, int, float]] = None  # (net, host, solo_ms)
        for h in range(n):
            solo_ms = round_mib / float(ranks[h].bandwidth_mib_per_s) * 1000.0
            # Capacity: the host must hold the whole family on top of everything
            # else it already carries under the baseline.
            others = base_resident[h] - resident * base_ratios[h]
            if others + resident > float(ranks[h].capacity_mib):
                continue
            cand_net = saving_ms - (solo_ms - sharded_ms)
            if cand_net > 0.0 and (best is None or cand_net > best[0]):
                best = (cand_net, h, solo_ms)

        fastest = max(range(n), key=lambda i: ranks[i].bandwidth_mib_per_s)
        solo_ms_ref = round_mib / float(ranks[fastest].bandwidth_mib_per_s) * 1000.0
        bw_delta_ref = solo_ms_ref - sharded_ms
        break_even = (
            float("inf")
            if f.blocking_total == 0
            else bw_delta_ref * 1000.0 / f.blocking_total
        )

        if best is None:
            placements[f.name] = FamilyPlacement(
                name=f.name,
                ratios=tuple(base_ratios),
                is_solo=False,
                host_rank=None,
                bandwidth_ms_sharded=sharded_ms,
                bandwidth_ms_solo=solo_ms_ref,
                bandwidth_ms_chosen=sharded_ms,
                bandwidth_delta_ms=bw_delta_ref,
                blocking_removed=0,
                deferred_removed=0,
                collective_saving_ms=0.0,
                net_ms=0.0,
                break_even_us=break_even,
            )
            continue

        cand_net, host, solo_ms = best
        one_hot = tuple(1.0 if i == host else 0.0 for i in range(n))
        for i in range(n):
            delta[i] += resident * (one_hot[i] - base_ratios[i])
        net_total += cand_net
        placements[f.name] = FamilyPlacement(
            name=f.name,
            ratios=one_hot,
            is_solo=True,
            host_rank=host,
            bandwidth_ms_sharded=sharded_ms,
            bandwidth_ms_solo=solo_ms,
            bandwidth_ms_chosen=solo_ms,
            bandwidth_delta_ms=bw_delta_ref,
            blocking_removed=f.blocking_total,
            deferred_removed=0,
            collective_saving_ms=saving_ms,
            net_ms=cand_net,
            break_even_us=break_even,
        )

    return FamilySplitSolution(
        by_family=placements,
        net_ms=net_total,
        capacity_delta_mib=tuple(delta),
    )
