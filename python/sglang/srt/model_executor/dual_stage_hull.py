"""DUAL-TP3PP3 stage 1b (F26): the P stage computes on D's TP3 shards.

User idea 2026-09-29: D (TP3, uneven) and P (PP3) are resident at the same
time, and P adds only the DIFF of its stage's layer bytes to what D already
holds on the card. For that P's linears must be able to consume D's shard
AS IT IS -- a contiguous tensor of D's own shape, never a strided view into a
full matrix. The #274 lane already solved exactly this algebra in one
process (``dual_group_lane.py``): a full-width HULL whose parallel linears are
SHELLS over N sharded PART trees; column-parallel = concat of the parts'
outputs, row-parallel = sum of the parts' partial products, both local ops,
no communicator. NVFP4 block scales stay intact because every part is an
ordinary TP shard with its own quant method (partial GEMMs, not a strided
GEMM).

This module reuses that machinery for a P STAGE:

* the parts are D's TP3 geometry (``--rank-tp-ratio`` and the family vectors
  of D, handed in by env, never re-derived here -- a vector that differs by
  one unit makes the "shared" part a different shard), one segment per D rank
  (``((0,), (1,), (2,))``: FAST ratio == BIG ratio, so nesting holds by
  construction);
* the PP geometry of the P rank stays in force while the parts load, so each
  part holds only THIS stage's layers;
* the hull is the P stage at TP1; ``assemble_lane_shells`` swaps its linears.

Which part is byte-identical to D's resident shard on this card is the part
of the D rank that lives on the same card (``d_rank_on_this_card``); the
union bind (``weg2/union_arena_bind.py``) then makes it ONE set of bytes
across the two processes. This module only builds the stage; sharing is the
bind's job and is checked there by content checksum.

Default off (``SGLANG_WEG2_DUAL_SHARE`` unset): nothing here is imported on
the load path.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import time
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

DUAL_SHARE_ENV = "SGLANG_WEG2_DUAL_SHARE"
#: D's base vector, e.g. "58,25,25" (the INSTALLED --rank-tp-ratio of D).
D_TP_RATIO_ENV = "SGLANG_WEG2_DUAL_D_TP_RATIO"
#: D's family vectors, e.g. "mlp=98,19,19;vocab=1,1,1" (only those D installs).
D_FAMILIES_ENV = "SGLANG_WEG2_DUAL_D_FAMILIES"
#: D rank living on the card of each P stage, e.g. "0,1,2" (P stage s -> D rank).
D_RANK_OF_STAGE_ENV = "SGLANG_WEG2_DUAL_D_RANK_OF_STAGE"


class DualShareError(ValueError):
    pass


def _ints(text: str, what: str) -> Tuple[int, ...]:
    try:
        out = tuple(int(x) for x in str(text).replace(" ", "").split(",") if x != "")
    except ValueError:
        raise DualShareError(f"DUAL-TP3PP3: {what} {text!r} is not a comma list of integers")
    if not out:
        raise DualShareError(f"DUAL-TP3PP3: {what} is empty")
    return out


@dataclasses.dataclass(frozen=True)
class DualShareSpec:
    tp_ratio: Tuple[int, ...]
    families: Tuple[Tuple[str, Tuple[int, ...]], ...]
    d_rank_of_stage: Tuple[int, ...]

    @property
    def d_size(self) -> int:
        return len(self.tp_ratio)

    def d_rank_on_stage(self, pp_rank: int) -> int:
        if not 0 <= pp_rank < len(self.d_rank_of_stage):
            raise DualShareError(
                f"DUAL-TP3PP3: P stage {pp_rank} has no entry in {D_RANK_OF_STAGE_ENV}="
                f"{list(self.d_rank_of_stage)}")
        return self.d_rank_of_stage[pp_rank]

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None, pp_size: int = 3) -> "DualShareSpec":
        env = os.environ if env is None else env
        raw = env.get(D_TP_RATIO_ENV, "").strip()
        if not raw:
            raise DualShareError(
                f"DUAL-TP3PP3: {DUAL_SHARE_ENV}=1 needs {D_TP_RATIO_ENV} (D's installed "
                "--rank-tp-ratio). It is not derived here: a vector one unit off makes the "
                "'shared' part a different shard.")
        tp = _ints(raw, D_TP_RATIO_ENV)
        fams: List[Tuple[str, Tuple[int, ...]]] = []
        for item in filter(None, (s.strip() for s in env.get(D_FAMILIES_ENV, "").split(";"))):
            name, _, vec = item.partition("=")
            name = name.strip()
            if name not in ("mlp", "moe", "vocab"):
                raise DualShareError(f"DUAL-TP3PP3: unknown family {name!r} in {D_FAMILIES_ENV}")
            v = _ints(vec, f"{D_FAMILIES_ENV}[{name}]")
            if len(v) != len(tp):
                raise DualShareError(
                    f"DUAL-TP3PP3: family {name} has {len(v)} entries, D's base vector {len(tp)}")
            fams.append((name, v))
        stage = env.get(D_RANK_OF_STAGE_ENV, "").strip()
        ranks = _ints(stage, D_RANK_OF_STAGE_ENV) if stage else tuple(range(pp_size))
        if len(ranks) != pp_size or sorted(ranks) != list(range(len(tp))):
            raise DualShareError(
                f"DUAL-TP3PP3: {D_RANK_OF_STAGE_ENV}={list(ranks)} must map the {pp_size} P "
                f"stages one-to-one onto the {len(tp)} D ranks (one D rank and one P stage per card)")
        return cls(tp, tuple(fams), ranks)

    def nested_plan(self):
        from sglang.srt.distributed.dual_group import NestedGroupPlan

        return NestedGroupPlan(
            big_ratio=self.tp_ratio,
            segments=tuple((r,) for r in range(self.d_size)),
            family_ratios=self.families,
        )


def dual_share_armed(env: Optional[Mapping[str, str]] = None) -> bool:
    env = os.environ if env is None else env
    if str(env.get(DUAL_SHARE_ENV, "")).strip().lower() not in ("1", "on", "true"):
        return False
    return str(env.get("SGLANG_WEG2_GROUP", "")).strip().upper() == "P"


def build_dual_stage_model(runner, spec: Optional[DualShareSpec] = None):
    """The P rank's load_model body under ``SGLANG_WEG2_DUAL_SHARE=1``.

    Returns the stage HULL (TP1, this rank's PP layers) whose linears are
    shells over D's three TP shards of those layers. The parts are loaded by
    the stock loader under D's geometry; the PP geometry stays, so a part is
    this stage only.
    """
    from sglang.srt.model_executor.dual_group_lane import (
        _build_hull,
        _fill_hull_buffers,
        _finalize_hull_params,
        _load_lane_part,
        _refresh_captured_linear_attention_tensors,
        assemble_lane_shells,
        hull_needs_real_storage,
    )

    pp_rank = int(getattr(runner, "pp_rank", 0))
    pp_size = int(getattr(runner, "pp_size", 1))
    if int(getattr(runner, "tp_size", 1)) != 1:
        raise DualShareError(
            f"DUAL-TP3PP3: the P stage hull is built at TP1, this P rank runs tp_size="
            f"{runner.tp_size}")
    spec = spec or DualShareSpec.from_env(pp_size=pp_size)
    plan = spec.nested_plan()
    local = spec.d_rank_on_stage(pp_rank)
    t0 = time.perf_counter()
    parts = [_load_lane_part(runner, plan, r, gpu_id=runner.gpu_id) for r in range(plan.fast_size)]
    on_meta = not hull_needs_real_storage(runner.model_config)
    hull = _build_hull(runner, device="meta" if on_meta else None)
    counts = assemble_lane_shells(hull, parts)
    fill = _finalize_hull_params(hull, parts[local], parts)
    fill["buffers"] = _fill_hull_buffers(hull, parts[local]) if on_meta else 0
    fill["captured"] = _refresh_captured_linear_attention_tensors(hull) if on_meta else 0
    runner.dual_share_part_models = parts
    runner.dual_share_local_part = local
    logger.info(
        "DUAL-TP3PP3 P stage %d/%d assembled in %.1f s: shells over D's %d TP shards "
        "(D ratio %s, families %s), part %d = D rank on this card (the bytes the union "
        "bind shares); hull on %s; shells column=%d row=%d embed=%d lm_head=%d composed=%d; "
        "aliased=%d composed_vec=%d buffers=%d captured=%d",
        pp_rank, pp_size, time.perf_counter() - t0, plan.fast_size, list(spec.tp_ratio),
        dict(spec.families), local, "meta" if on_meta else "device", counts["column"],
        counts["row"], counts["embedding"], counts["lm_head"], counts["composed"],
        fill["aliased"], fill["composed_vec"], fill["buffers"], fill["captured"])
    return hull


def shared_part_named_parameters(runner) -> Dict[str, "object"]:
    """The parameters of the part that is D's resident shard on this card --
    the tensor set the union image of this P rank publishes (by D's names)."""
    parts = getattr(runner, "dual_share_part_models", None)
    local = getattr(runner, "dual_share_local_part", None)
    if parts is None or local is None:
        return {}
    return dict(parts[local].named_parameters())
