# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""``GET /registry/plan``'s second opinion: the existing offline planner.

The ledger answers one question -- do the declared bytes fit beside what other
tenants already hold. That question is class-agnostic and cheap, and it is the
one the arbiter enforces. It is also not the same question as "is this budget
enough to actually run the model", which for Class 1 the planner package
(``python/sglang/srt/planner/``) already answers with a real cost model.

Asking both is worth the twenty lines: a spec can pass the ledger and still be
a bad plan (a 3 GiB budget for a 27B model fits any card and boots nothing),
and a spec can fail the ledger while being a perfectly good plan that simply
needs a neighbour to move. Reporting them separately keeps the two verdicts
from being confused for each other.

Class 2 and Class 3 have no offline cost model yet, so the bridge returns
``None`` for them rather than inventing agreement.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

from sglang.srt.registry.spec import EngineClass, EngineSpec

logger = logging.getLogger(__name__)


def planner_opinion(
    spec: EngineSpec,
    cards: tuple[str, ...],
    *,
    card_indices: Mapping[str, int] | None = None,
) -> dict[str, Any] | None:
    """Run the offline planner for a Class-1 spec. Never raises.

    Returns ``None`` when the planner cannot be consulted -- wrong class, no
    NVML, an unreadable checkpoint. A missing second opinion must not turn a
    dry-run into an error: the ledger verdict stands on its own.
    """
    if int(spec.klass) != int(EngineClass.AUTOREGRESSIVE):
        return None
    launch = dict(spec.launch)
    model_path = launch.get("model_path")
    budget_mib = launch.get("rank_gpu_memory_mib")
    if not model_path or budget_mib is None:
        return None
    try:
        from sglang.srt.planner.feasibility import PlanRejected, plan  # noqa: PLC0415
        from sglang.srt.planner.hardware import hardware_from_nvml  # noqa: PLC0415

        hardware = hardware_from_nvml()
        indices = dict(card_indices or {})
        if not indices:
            from sglang.srt.registry.nvml import list_devices  # noqa: PLC0415

            indices = {d.uuid: d.index for d in list_devices()}
        rank_cards = launch.get("rank_cards") or list(cards)
        rank_gpu_id = [indices[str(c)] for c in rank_cards]
        tp_size = int(launch.get("tp_size", len(rank_gpu_id)))
        result = plan(
            str(model_path),
            hardware,
            tp_size=tp_size,
            rank_gpu_id=rank_gpu_id,
            rank_gpu_memory_mib=[int(budget_mib)] * tp_size,
        )
    except PlanRejected as exc:  # type: ignore[misc]
        return {"available": True, "fits": False, "reasons": list(exc.args[0])}
    except Exception as exc:  # noqa: BLE001 - a second opinion is optional
        logger.debug(
            "registry: planner opinion unavailable for %s: %s", spec.engine_id, exc
        )
        return None
    return {
        "available": True,
        "fits": bool(getattr(result, "fits", False)),
        "reasons": list(getattr(result, "reasons", []) or []),
        "summary": _summarise(result),
    }


def _summarise(result: Any) -> dict[str, Any]:
    """Pick the few planner fields an operator reads first, if present."""
    keys = (
        "max_total_num_tokens",
        "kv_tokens",
        "context_tokens",
        "weights_mib",
        "kv_mib",
    )
    return {k: getattr(result, k) for k in keys if hasattr(result, k)}
