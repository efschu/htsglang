# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""#553 cut 3 (as planned): apply a chosen RELIEF rung by delegating to its feature.

THE WIRE THAT WAS MISSING. #287's ladder ranks rungs and enforces the ordering
invariant, :data:`kv_pressure_ladder.RELIEF_FEATURES` names the five features a
relief rung may reference, ``planner/kv_ladder_table.py`` validates that a
rung's ``relief_feature`` is one of them, and each feature has a real actuator
behind a flag. Grepping ``RELIEF_FEATURES`` across the tree found exactly two
consumers: the ladder that defines it and the table that checks membership.

**Nothing mapped a chosen rung to the actuator of the feature it names.** The
ladder could rank a ``dcp_ratio`` rung above an ``admission_cap`` one, a caller
could ascend to it, and nothing would change. A counter with no actuator.

DELEGATION ONLY. This module reimplements no feature -- the ladder's own
docstring says relief features are "referenced BY NAME ... never reimplemented
here", and that holds one layer down too. It takes one actuator per feature,
calls exactly one, and reports what that actuator returned. Per #694 nothing
here derives a result from the plan: if the actuator says it did nothing, the
result says so.

REFUSALS ARE THE DESIGN, because the failure this replaces is silence. An
unknown feature and a feature whose actuator is not wired both RAISE. A relief
rung that quietly no-ops is worse than one that fails, since the ladder would
then ascend believing pressure had been relieved and the next rung would be
chosen against a state that never changed.

ONLY RELIEF. ``geometry_flip`` and ``external`` rungs are refused here. They
are declared stubs with their own handover strategies, and this executor must
not become the place a geometry change quietly starts -- the plan scoped this
cut as "make #287's RELIEF rungs execute, **and only those**".
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Mapping, Optional, Tuple

from sglang.srt.model_executor.kv_pressure_ladder import RELIEF_FEATURES

#: The rung kind this executor will act on. Anything else is refused.
RELIEF_KIND = "relief"


class UnknownReliefFeature(ValueError):
    """A rung names a feature that is not in the ladder's vocabulary."""


class ReliefActuatorMissing(ValueError):
    """A known feature was chosen but no actuator for it was supplied."""


@dataclasses.dataclass(frozen=True)
class ReliefResult:
    """What the ACTUATOR reported. Never what the plan intended (#694)."""

    feature: str
    ok: bool
    detail: str


def supported_features() -> Tuple[str, ...]:
    """The vocabulary, taken from the ladder rather than restated.

    A second list here would be the two-enumerations defect one layer down --
    a sixth feature could be applied that the ladder cannot rank, or a ranked
    one could become unreachable.
    """
    return tuple(RELIEF_FEATURES)


def apply_relief_rung(
    rung: Any,
    actuators: Mapping[str, Callable[[], Tuple[bool, str]]],
) -> ReliefResult:
    """Apply one relief rung by calling the actuator its feature names.

    ``rung`` is a ladder step carrying ``relief_feature`` and ``kind``.
    ``actuators`` maps a feature name to a zero-argument callable returning
    ``(ok, detail)`` -- injected so this stays hermetic and so the caller keeps
    ownership of how its feature is driven.
    """
    kind = getattr(rung, "kind", RELIEF_KIND)
    feature = getattr(rung, "relief_feature", None)

    if kind != RELIEF_KIND:
        raise UnknownReliefFeature(
            f"rung kind {kind!r} is not {RELIEF_KIND!r}; this executor applies "
            "relief rungs and only those. geometry_flip and external rungs "
            "carry their own handover strategies and are deliberately not "
            "actuated here."
        )
    if feature not in RELIEF_FEATURES:
        raise UnknownReliefFeature(
            f"relief feature {feature!r} is not in the ladder's vocabulary. "
            f"Known features: {sorted(RELIEF_FEATURES)!r}. A rung naming an "
            "unknown feature is refused rather than skipped, because a relief "
            "step that silently does nothing lets the ladder ascend believing "
            "pressure was relieved."
        )

    actuator: Optional[Callable[[], Tuple[bool, str]]] = dict(actuators).get(feature)
    if actuator is None:
        raise ReliefActuatorMissing(
            f"relief feature {feature!r} is a valid rung but no actuator for it "
            "was supplied, so applying this rung would change nothing while "
            "reporting success. Wire the feature's actuator, or keep the rung "
            "out of the ladder for this boot."
        )

    ok, detail = actuator()
    return ReliefResult(feature=feature, ok=bool(ok), detail=str(detail))
