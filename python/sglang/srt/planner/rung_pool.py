"""#704: the single pool entry point for ladder rungs.

Slot-3's ``LadderInputs`` already consumes ``PhasePoolModel`` and requires an
``arming_floor_for`` provider, but it builds the model itself in
``pool_model_for``. That is two solvers which must agree and are computed
twice -- the shape this corpus has paid for repeatedly. ``solve_rung_pool`` is
the one surface the rung table and the boundary actuator call.

It inherits the refusal semantics rather than restating them:

* **no default reserve.** The per-rank reserve tracks CUDA-graph capture and
  does not transfer between layouts: 3,818 / 5,164 / 8,848 MiB across three
  stages of a single boot, a 2.3x spread. A rung with no reserve source raises.
* **provenance is a field**, so an extrapolated rung cannot be read as a
  measured one further down the ladder.
* **calibration coverage is always returned**, so a rung whose calibration
  cannot identify a rank says so at the point of use.

Providers may be callables or mappings, so this can match existing call sites
without a second surface appearing to serve them.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

from sglang.srt.planner.calibration_coverage import (
    RankCoverage,
    calibration_coverage,
)

_MIB = 1024.0 * 1024.0
#: K+V bytes per token per ATTENTION layer (fp8_e4m3, 2 x 4 kv-heads x 256).
_KV_BYTES_PER_TOKEN_PER_ATTN_LAYER = 2048

Provider = Union[Callable[[Sequence[int]], Any], dict, None]


@dataclasses.dataclass(frozen=True)
class RungPoolSolution:
    counts: Tuple[int, ...]
    pool_tokens: int
    binding_stage: int
    per_stage_tokens: Tuple[int, ...]
    provenance: str  # "measured" | "extrapolated"
    reserve_source: str
    coverage: Tuple[RankCoverage, ...]
    caveats: Tuple[str, ...]


def _resolve(provider: Provider, counts: Sequence[int], name: str):
    if provider is None:
        return None
    key = tuple(int(n) for n in counts)
    if isinstance(provider, dict):
        return provider.get(key)
    try:
        return provider(key)
    except Exception as exc:  # pragma: no cover - provider's own failure
        raise ValueError(f"{name} raised for cut {key}: {exc}") from exc


def solve_rung_pool(
    counts: Sequence[int],
    attn_counts: Sequence[int],
    *,
    arming_floor_for: Provider,
    reserve_for: Provider = None,
    rest_for: Provider = None,
    measured: bool = False,
    reference_cut: Optional[Sequence[int]] = None,
) -> RungPoolSolution:
    """Pool for one ladder rung, from instruments, with provenance.

    ``arming_floor_for`` is required and keeps Slot-3's existing contract. It is
    NOT subtracted here: the floor is held back after the profiler, so it
    already sits inside the recovered reserve, and charging it again would
    understate every rung by ~2 GiB. It is resolved anyway, because a rung whose
    floor cannot be produced is a rung nobody has sized.
    """
    counts = tuple(int(n) for n in counts)
    attn_counts = tuple(int(n) for n in attn_counts)
    if len(counts) != len(attn_counts):
        raise ValueError(
            f"stage count mismatch: cut {counts} has {len(counts)} stages but "
            f"attention counts {attn_counts} has {len(attn_counts)}."
        )
    if arming_floor_for is None:
        raise ValueError(
            "arming_floor_for is required: the arming floor is per-LAYOUT and "
            "moves with the cut (measured 1728/1825/2467 MiB on [28,20,16] "
            "against 2255/1728/2467 on [32,16,16]). Pass the #676 solver "
            "(phase_flip_seam_reserve.arming_floor_target_bytes) or a measured "
            "mapping; it must not be defaulted."
        )
    floors = _resolve(arming_floor_for, counts, "arming_floor_for")
    if floors is None:
        raise ValueError(
            f"arming_floor_for produced no floor for cut {counts}; that layout "
            "has not been sized."
        )

    rest = _resolve(rest_for, counts, "rest_for")
    if rest is None:
        raise ValueError(
            f"no rest (budget minus budget posts) available for cut {counts}. "
            "It comes from the boot's 'KV budget posts ... | rest=' line."
        )

    reserve = _resolve(reserve_for, counts, "reserve_for")
    if reserve is None:
        raise ValueError(
            f"no reserve available for cut {counts}. The per-rank reserve "
            "tracks CUDA-graph capture and does NOT transfer between layouts "
            "(measured spread 3,818-8,848 MiB within one boot), so it cannot be "
            "defaulted or carried over from another rung. Supply the reserve "
            "recovered from a boot of THIS layout (rest - available_bytes), or "
            "do not claim a pool for it."
        )

    per_stage: List[int] = []
    for stage, (a, r, v) in enumerate(zip(attn_counts, rest, reserve)):
        if int(a) <= 0:
            raise ValueError(
                f"stage {stage} of cut {counts} holds no full-attention layer, "
                "so it carries no token-scaling KV and its capacity is "
                "unbounded. That is a modelling artifact, not a configuration."
            )
        available = (float(r) - float(v)) * _MIB
        cell = int(a) * _KV_BYTES_PER_TOKEN_PER_ATTN_LAYER
        per_stage.append(0 if available <= 0 else int(available) // cell)

    pool = min(per_stage)
    binder = per_stage.index(pool)

    caveats: List[str] = []
    provenance = "measured" if measured else "extrapolated"
    if not measured:
        caveats.append(
            "extrapolated: no boot of this exact layout supplied the reserve, "
            "so the pool inherits another layout's capture behaviour."
        )

    coverage: Tuple[RankCoverage, ...] = ()
    if reference_cut is not None:
        coverage = calibration_coverage(counts, reference_cut)
        for cov in coverage:
            if not cov.identified:
                caveats.append(
                    f"rank{cov.rank} is UNIDENTIFIED by the pair "
                    f"{counts} <-> {tuple(int(n) for n in reference_cut)}: its "
                    "layer count does not move, so no sample size recovers its "
                    "term."
                )

    return RungPoolSolution(
        counts=counts,
        pool_tokens=int(pool),
        binding_stage=int(binder),
        per_stage_tokens=tuple(per_stage),
        provenance=provenance,
        reserve_source=("measured boot" if measured else "carried"),
        coverage=coverage,
        caveats=tuple(caveats),
    )
