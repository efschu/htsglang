"""Build ``PPCutInputs`` from a previous boot's residency census (#485).

Why a previous boot, and not the config
---------------------------------------
Every attempt so far to price this cut from the checkpoint's config arithmetic
has been wrong, and each error was invisible until the cut moved:

* the attention layer is 364.88 MiB resident against the 325.0 MiB the
  parameter formula gives, because ``attn_output_gate`` adds a second q-sized
  projection (482 MiB across 16 layers);
* the input embedding, ``lm_head`` and the vision tower are bf16 and outside
  any per-layer census (~5.7 GiB);
* the recurrent-state pool is 51.2 MiB per linear layer and lived, unnamed,
  inside a fitted residual.

So the calibration comes from bytes a running process reported
(``residency_census.py``), and the terms this module cannot source are
REFUSED rather than defaulted -- the same discipline
``planner/cost_model.AbsentRate`` applies to probe rates, for the same reason:
a zero here does not read as "unknown", it reads as "free", and the gate then
certifies a configuration that cannot run.

THE ONE THING THIS MODULE CANNOT DO FOR YOU (C39). The per-rank residual it
derives -- graphs, workspaces, CUDA context, communicator buffers -- is NOT
cut-invariant. Measured on rank0 of the reference rig: 3174 MiB at 28 layers,
4419 at 40, 4440 at 42. Flat between neighbouring cuts, 1250 MiB apart across
the ship/planner boundary. A census taken on one cut family therefore
predicts within that family to tens of MiB and across families to ~1250 MiB,
and :func:`load_census_calibration` records which cut it was taken on so the
caller can say so out loud.
"""

from __future__ import annotations

import dataclasses
import glob
import json
import os
from typing import Dict, List, Optional, Tuple

from sglang.srt.planner import pp_cut

__all__ = [
    "CensusCalibration",
    "PPCutCalibrationError",
    "load_census_calibration",
]


class PPCutCalibrationError(ValueError):
    """A calibration input is missing or inconsistent. Never defaulted."""


@dataclasses.dataclass(frozen=True)
class CensusCalibration:
    """Measured per-rank terms, in cut order (stage 0 first)."""

    attn_layer_mib: float
    linear_layer_mib: float
    embedding_mib: float
    lm_head_mib: float
    replicated_mib: float
    state_per_linear_mib: float
    #: graphs + workspaces + context + communicators, per rank
    residual_mib: Tuple[float, ...]
    #: what the driver reports the card holding, per rank
    total_visible_mib: Tuple[float, ...]
    #: the cut this census was taken on -- the neighbourhood it is valid in
    calibrated_on_counts: Tuple[int, ...]
    #: each stage's card, AS THAT RANK REPORTED IT. Never re-derived from a
    #: --rank-gpu-id index: on the reference rig rank 0 is NVML index 1
    #: (CUDA_VISIBLE_DEVICES is set by UUID), and torch's device order is not
    #: NVML's. The IdentityMap travels in the artifact.
    gpu_names: Tuple[Optional[str], ...] = ()
    calibrated_on_pool: Optional[int] = None

    def describe(self) -> str:
        return (
            f"census calibration from cut {list(self.calibrated_on_counts)}: "
            f"attn layer {self.attn_layer_mib:.1f} MiB, linear layer "
            f"{self.linear_layer_mib:.1f} MiB, embedding "
            f"{self.embedding_mib:.0f} MiB, lm_head {self.lm_head_mib:.0f} "
            f"MiB, replicated {self.replicated_mib:.0f} MiB, recurrent state "
            f"{self.state_per_linear_mib:.1f} MiB/linear layer, per-rank "
            f"residual {[round(r) for r in self.residual_mib]} MiB"
        )


def _require(value, what: str, where: str):
    if value is None:
        raise PPCutCalibrationError(
            f"residency census at {where} does not report {what}. The cut "
            f"gate refuses to default it: an unpriced term reads as free "
            f"memory and certifies a configuration that cannot run. Re-take "
            f"the census with SGLANG_RESIDENCY_CENSUS=1 on a boot of this "
            f"model."
        )
    return value


def load_census_calibration(census_dir: str) -> CensusCalibration:
    """Read every ``census_pp*.json`` in ``census_dir`` into one calibration.

    Refuses loudly on a partial census: a per-rank residual derived from some
    ranks and guessed for others is exactly the kind of half-measured number
    this whole module exists to stop.
    """
    if not os.path.isdir(census_dir):
        raise PPCutCalibrationError(
            f"--pp-solve-cut needs a residency-census directory; "
            f"{census_dir!r} is not one. Produce it by booting once with "
            f"SGLANG_RESIDENCY_CENSUS=1 and "
            f"SGLANG_RESIDENCY_CENSUS_DIR={census_dir}."
        )
    paths = sorted(glob.glob(os.path.join(census_dir, "census_pp*.json")))
    if not paths:
        raise PPCutCalibrationError(
            f"no census_pp*.json in {census_dir!r}. The census writes one "
            f"file per rank; an empty directory means the probe never ran."
        )

    ranks: Dict[int, dict] = {}
    for p in paths:
        with open(p) as fh:
            blob = json.load(fh)
        pp_rank = int(blob.get("pp_rank", -1))
        if pp_rank < 0:
            raise PPCutCalibrationError(f"{p}: census has no pp_rank.")
        ranks[pp_rank] = blob

    expected = list(range(len(ranks)))
    if sorted(ranks) != expected:
        raise PPCutCalibrationError(
            f"{census_dir!r} holds census files for ranks {sorted(ranks)}, "
            f"which is not a complete 0..N-1 set. A calibration missing a "
            f"rank cannot price that rank."
        )

    counts: List[int] = []
    residual: List[float] = []
    totals: List[float] = []
    names: List[Optional[str]] = []
    attn_per: List[float] = []
    linear_per: List[float] = []
    embedding = lm_head = replicated = None

    for pp_rank in expected:
        blob = ranks[pp_rank]
        params = blob.get("params_mib") or {}
        n_attn = int(_require(blob.get("n_attn_layers"), "n_attn_layers", census_dir))
        n_lin = int(
            _require(blob.get("n_linear_layers"), "n_linear_layers", census_dir)
        )
        counts.append(n_attn + n_lin)

        if n_attn:
            attn_per.append(float(params.get("layers_attention", 0.0)) / n_attn)
        if n_lin:
            linear_per.append(float(params.get("layers_linear", 0.0)) / n_lin)

        if "embed_tokens" in params:
            embedding = float(params["embed_tokens"])
        if "lm_head" in params:
            lm_head = float(params["lm_head"])
        vis = float(params.get("visual", 0.0))
        replicated = vis if replicated is None else max(replicated, vis)

        used = float(_require(blob.get("nvml_used_mib"), "nvml_used_mib", census_dir))
        pools = float(_require(blob.get("pools_mib"), "pools_mib", census_dir))
        params_total = sum(float(v) for v in params.values())
        residual.append(used - params_total - pools)
        totals.append(
            float(_require(blob.get("nvml_total_mib"), "nvml_total_mib", census_dir))
        )
        names.append(blob.get("gpu_name"))

    if not attn_per or not linear_per:
        raise PPCutCalibrationError(
            f"{census_dir!r}: the census saw no {'attention' if not attn_per else 'linear'} "
            f"layers on any rank, so the per-layer weight cannot be derived. "
            f"A cut cannot be solved for a family the calibration never saw."
        )

    # The recurrent-state pool is what the pool post holds beyond the KV
    # arena. It cannot be separated here without the arena geometry, so it is
    # left to the caller to supply or to zero DELIBERATELY -- see the
    # state_bytes_per_linear_layer note below.
    state_per_linear = 0.0

    return CensusCalibration(
        attn_layer_mib=sum(attn_per) / len(attn_per),
        linear_layer_mib=sum(linear_per) / len(linear_per),
        embedding_mib=float(_require(embedding, "embed_tokens", census_dir)),
        lm_head_mib=float(_require(lm_head, "lm_head", census_dir)),
        replicated_mib=float(replicated or 0.0),
        state_per_linear_mib=state_per_linear,
        residual_mib=tuple(residual),
        total_visible_mib=tuple(totals),
        calibrated_on_counts=tuple(counts),
        gpu_names=tuple(names),
    )


def with_arena_split_state(
    calibration: CensusCalibration,
    *,
    census_dir: str,
    kv_bytes_per_token_per_attn_layer: float,
    pool_tokens: int,
    tp_token_shares: Optional[Tuple[float, ...]] = None,
) -> CensusCalibration:
    """Split the census's pool post into KV arena and recurrent state.

    The census reports one ``pools_mib`` number per rank. The arena part is
    computable from the geometry the same boot ran at, so what remains is the
    recurrent-state pool, and dividing it by that rank's linear-layer count
    gives the per-layer term. Derived per rank and averaged, so a single
    rank's rounding does not become the model.
    """
    paths = sorted(glob.glob(os.path.join(census_dir, "census_pp*.json")))
    per_layer: List[float] = []
    for p in paths:
        with open(p) as fh:
            blob = json.load(fh)
        n_attn = int(blob.get("n_attn_layers", 0))
        n_lin = int(blob.get("n_linear_layers", 0))
        if not n_lin:
            continue
        stage = int(blob.get("pp_rank", 0))
        arena_layers = float(n_attn)
        if tp_token_shares is not None and stage < len(tp_token_shares):
            n_full_total = sum(
                int(json.load(open(q)).get("n_attn_layers", 0)) for q in paths
            )
            arena_layers = max(
                arena_layers, tp_token_shares[stage] * float(n_full_total)
            )
        arena_mib = (
            arena_layers * kv_bytes_per_token_per_attn_layer * float(pool_tokens)
        ) / pp_cut.MIB
        remainder = float(blob.get("pools_mib", 0.0)) - arena_mib
        if remainder > 0:
            per_layer.append(remainder / n_lin)
    if not per_layer:
        return calibration
    return dataclasses.replace(
        calibration,
        state_per_linear_mib=sum(per_layer) / len(per_layer),
        calibrated_on_pool=pool_tokens,
    )
