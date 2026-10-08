"""Planner-visible GATED CAPABILITIES: quality gate result + measured net.

Binding directive (``PLAN_PERF_PIPELINE_2026-08-16``): a capability like
ReplaySSM must not be an env flag someone remembers to set. The planner has to
see it, decide it, and be able to state why it declined.

The rules are capability-agnostic:

* An UNMEASURED quality gate is a refusal. Not a default-on, and not a silent
  default-off either -- the planner must name the missing measurement.
* An unmeasured net is a refusal for the same reason: a benefit nobody counted
  cannot be traded against a quality cost.
* Byte-identical AND net above the floor enables, non-lossy.
* Anything not byte-identical is LOSSY and is refused unless lossy is
  explicitly permitted, per the standing "byte-identical wins come first"
  policy. A large net does not override that ordering; it only decides whether
  a permitted lossy capability is worth taking.
"""

from __future__ import annotations

import dataclasses
from typing import Optional


@dataclasses.dataclass(frozen=True)
class QualityGate:
    """Result of an identity measurement.

    ``byte_identical is None`` means the gate has not been run. That is a
    distinct state from ``False`` and must never collapse into it: "we did not
    look" and "we looked and it diverges" support different decisions.
    """

    name: str
    byte_identical: Optional[bool]
    changes_emitted_tokens: Optional[bool]
    source: str

    @property
    def is_measured(self) -> bool:
        return self.byte_identical is not None


@dataclasses.dataclass(frozen=True)
class GatedCapability:
    name: str
    quality: QualityGate
    measured_net_fraction: Optional[float]
    min_net_fraction: float = 0.0


@dataclasses.dataclass(frozen=True)
class CapabilityDecision:
    enable: bool
    lossy: bool
    reason: str


def decide_capability(
    capability: GatedCapability, allow_lossy: bool = False
) -> CapabilityDecision:
    """Decide one capability. Every refusal carries a reason the planner can print."""
    q = capability.quality
    lossy = bool(q.byte_identical is False)

    if not q.is_measured:
        return CapabilityDecision(
            enable=False,
            lossy=lossy,
            reason=(
                f"{capability.name}: quality gate {q.name!r} is UNMEASURED "
                f"(source={q.source!r}). Refusing to enable a capability whose "
                "output fidelity nobody has checked; run the identity A/B first."
            ),
        )

    if capability.measured_net_fraction is None:
        return CapabilityDecision(
            enable=False,
            lossy=lossy,
            reason=(
                f"{capability.name}: net benefit is unmeasured, so there is "
                "nothing to weigh the quality cost against."
            ),
        )

    net = float(capability.measured_net_fraction)
    floor = float(capability.min_net_fraction)

    if lossy and not allow_lossy:
        return CapabilityDecision(
            enable=False,
            lossy=True,
            reason=(
                f"{capability.name}: not byte-identical, and lossy capabilities "
                f"are not permitted here. Net {net:.1%} does not override the "
                "standing order that byte-identical wins land first."
            ),
        )

    if net < floor:
        return CapabilityDecision(
            enable=False,
            lossy=lossy,
            reason=(
                f"{capability.name}: measured net {net:.1%} is below the "
                f"{floor:.1%} floor; not worth the change."
            ),
        )

    return CapabilityDecision(
        enable=True,
        lossy=lossy,
        reason=(
            f"{capability.name}: enabled at net {net:.1%}"
            + (" (LOSSY, explicitly permitted)" if lossy else " (byte-identical)")
        ),
    )
