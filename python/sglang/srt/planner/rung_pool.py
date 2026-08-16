"""#704: the single pool entry point for ladder rungs.

Slot-3's ``LadderInputs`` consumed ``PhasePoolModel`` but built the model itself
in ``pool_model_for`` -- two solvers that must agree, computed twice. This is
the one surface his rung table and boundary actuator call.

Shaped to his five requirements, each tied to a bug already paid for:

1. ``attn_counts`` is DERIVED from ``layer_families``, never passed alongside
   the cut. A separate argument admits an inconsistent pair, and the attention
   count is the #702 divisor.
2. ``arming_floor_for`` is required with no default (E3).
3. The binding rank and per-rank caps are RETURNED, not just the min, so the
   controller does not recompute them in a second code path.
4. Cheap enough for an enumeration loop: build ``PoolContext`` once, then each
   solve is pure arithmetic -- no NVML, no census, no config parsing per call.
5. The KV cell comes from ``pp_cut.kv_mib_per_token_per_attn_layer_from_config``
   via :meth:`PoolContext.from_config`, never a caller-invented scalar.

Plus the refusal semantics this module inherits:

* **no default reserve.** The per-rank reserve tracks CUDA-graph capture and
  does not transfer between layouts -- measured on metal at 6.53 / 3.48 / 5.05
  GiB across three stages of one boot. A rung with no reserve source raises.
* **provenance is a field**, so an extrapolated rung cannot be read as measured.
* **calibration coverage** is reported, so a rung whose pair cannot identify a
  rank says so at the point of use.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from sglang.srt.planner.calibration_coverage import (
    RankCoverage,
    calibration_coverage,
)
from sglang.srt.planner.pp_cut import (
    attention_counts,
    kv_mib_per_token_per_attn_layer_from_config,
)

_MIB = 1024.0 * 1024.0

Provider = Union[Callable[[Sequence[int]], Any], dict, None]


@dataclasses.dataclass(frozen=True)
class PoolContext:
    """Built ONCE by the caller, then reused across an enumeration.

    Holds only cut-independent facts, so a solve stays pure arithmetic.
    Construct with :meth:`from_config`: that is the path which derives the KV
    cell from the checkpoint and the kv dtype, and a hand-supplied cell is how
    a wrong 2x lands silently in every pool number.
    """

    layer_families: Tuple[str, ...]
    kv_mib_per_token_per_attn_layer: float

    @classmethod
    def from_config(
        cls,
        config: Dict,
        kv_cache_dtype: str,
        layer_families: Sequence[str],
    ) -> "PoolContext":
        return cls(
            layer_families=tuple(layer_families),
            kv_mib_per_token_per_attn_layer=(
                kv_mib_per_token_per_attn_layer_from_config(config, kv_cache_dtype)
            ),
        )

    def attn_counts(self, counts: Sequence[int]) -> Tuple[int, ...]:
        if sum(int(n) for n in counts) != len(self.layer_families):
            raise ValueError(
                f"cut {tuple(counts)} sums to {sum(int(n) for n in counts)} but "
                f"the model has {len(self.layer_families)} layers; the attention "
                "counts cannot be derived from a cut that does not cover it."
            )
        return attention_counts(self.layer_families, counts)


@dataclasses.dataclass(frozen=True)
class StageTerms:
    """Per-stage breakdown, for diagnosable misses (``explain=True``)."""

    stage: int
    attn_layers: int
    gdn_layers: int
    rest_mib: float
    reserve_mib: float
    arming_floor_mib: float
    free_for_kv_mib: float
    cell_bytes: int
    tokens: int


@dataclasses.dataclass(frozen=True)
class RungPoolSolution:
    counts: Tuple[int, ...]
    attn_counts: Tuple[int, ...]
    pool_tokens: int
    binding_stage: int
    per_stage_tokens: Tuple[int, ...]
    provenance: str  # "measured" | "extrapolated"
    reserve_source: str
    coverage: Tuple[RankCoverage, ...]
    caveats: Tuple[str, ...]
    terms: Tuple[StageTerms, ...] = ()


def _resolve(provider: Provider, counts: Sequence[int], name: str):
    if provider is None:
        return None
    key = tuple(int(n) for n in counts)
    if isinstance(provider, dict):
        return provider.get(key)
    return provider(key)


def solve_rung_pool(
    counts: Sequence[int],
    ctx: PoolContext,
    *,
    arming_floor_for: Provider,
    reserve_for: Provider = None,
    rest_for: Provider = None,
    measured: bool = False,
    reference_cut: Optional[Sequence[int]] = None,
    explain: bool = False,
) -> RungPoolSolution:
    """Pool for one ladder rung, from instruments, with provenance.

    The arming floor is required but NOT subtracted: it is held back after the
    profiler, so it already sits inside the recovered reserve, and charging it
    again would understate every rung by ~2 GiB. It is resolved regardless,
    because a rung whose floor cannot be produced is a rung nobody has sized.
    """
    counts = tuple(int(n) for n in counts)
    attn = ctx.attn_counts(counts)

    if arming_floor_for is None:
        raise ValueError(
            "arming_floor_for is required: the arming floor is per-LAYOUT and "
            "moves with the cut (1728/1825/2467 MiB on [28,20,16] against "
            "2255/1728/2467 on [32,16,16]). Pass the #676 solver or a measured "
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
            f"no rest (budget minus budget posts) for cut {counts}. It comes "
            "from the boot's 'KV budget posts ... | rest=' line."
        )

    reserve = _resolve(reserve_for, counts, "reserve_for")
    if reserve is None:
        raise ValueError(
            f"no reserve for cut {counts}. The per-rank reserve tracks "
            "CUDA-graph capture and does NOT transfer between layouts (measured "
            "6.53/3.48/5.05 GiB within one boot), so it cannot be defaulted or "
            "carried from another rung. Supply the reserve recovered from a boot "
            "of THIS layout (rest - available_bytes), or do not claim a pool."
        )

    per_stage: List[int] = []
    terms: List[StageTerms] = []
    for stage, (n, a, r, v, f) in enumerate(zip(counts, attn, rest, reserve, floors)):
        if int(a) <= 0:
            raise ValueError(
                f"stage {stage} of cut {counts} holds no full-attention layer, "
                "so it carries no token-scaling KV and its capacity is "
                "unbounded. That is a modelling artifact, not a configuration."
            )
        cell = int(round(int(a) * ctx.kv_mib_per_token_per_attn_layer * _MIB))
        free_mib = float(r) - float(v)
        tokens = 0 if free_mib <= 0 else int(free_mib * _MIB) // cell
        per_stage.append(tokens)
        if explain:
            terms.append(
                StageTerms(
                    stage=stage,
                    attn_layers=int(a),
                    gdn_layers=int(n) - int(a),
                    rest_mib=float(r),
                    reserve_mib=float(v),
                    arming_floor_mib=float(f),
                    free_for_kv_mib=free_mib,
                    cell_bytes=cell,
                    tokens=tokens,
                )
            )

    pool = min(per_stage)
    binder = per_stage.index(pool)

    caveats: List[str] = []
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
                    f"rank{cov.rank} is UNIDENTIFIED by the pair {counts} <-> "
                    f"{tuple(int(x) for x in reference_cut)}: its layer count "
                    "does not move, so no sample size recovers its term."
                )

    return RungPoolSolution(
        counts=counts,
        attn_counts=attn,
        pool_tokens=int(pool),
        binding_stage=int(binder),
        per_stage_tokens=tuple(per_stage),
        provenance="measured" if measured else "extrapolated",
        reserve_source="measured boot" if measured else "carried",
        coverage=coverage,
        caveats=tuple(caveats),
        terms=tuple(terms),
    )
