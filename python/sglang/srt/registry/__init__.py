# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""The engine registry (#333 §7): N engines registered, M hot, M derived.

The registry sits *below* ``sgl-model-gateway`` and above the engines. The
gateway picks a worker; the registry decides whether that worker's engine is
resident, and if it is not, what it would cost to make it so. It does not
route between replicas, does not migrate a running request, and does not load
balance one engine across cards -- those are named non-goals (§7.7), not gaps.

Layers, bottom up:

``nvml``
    Physical GPU identity. UUID, never a CUDA index.
``ledger``
    Cross-process VRAM reservations, one file per card, leases and locks.
    Mechanism only. The Class-3 video tenant of M2 was its first client.
``spec``
    ``EngineSpec`` / ``EngineInstance`` / ``Slot``. No model architecture
    appears here.
``adapter`` + ``adapters/``
    The lane contract of §3.5 and the three shipped implementations.
``arbiter``
    Policy: admission, eviction, the derived M, the idle set, the capture lock.
``http_api``
    The control plane of §7.4.

Nothing in ``srt``'s boot path imports this package. A launch without the
registry is byte-for-byte the launch it was before.
"""

from sglang.srt.registry.arbiter import (
    CardView,
    EngineRegistry,
    HotCapacity,
    PlanResult,
    PromotionRejected,
    RegistrationRejected,
    RegistryError,
    UnknownEngineError,
    card_totals_from_nvml,
    free_bytes_from_nvml,
)
from sglang.srt.registry.ledger import (
    CardDemand,
    FeasibilityReport,
    ReservationStore,
    TenantState,
    plan_reservation,
)
from sglang.srt.registry.spec import (
    EngineClass,
    EngineInstance,
    EngineSpec,
    ResidencyState,
    ResourceProfile,
    Slot,
    SpecError,
)

__all__ = [
    "CardDemand",
    "CardView",
    "EngineClass",
    "EngineInstance",
    "EngineRegistry",
    "EngineSpec",
    "FeasibilityReport",
    "HotCapacity",
    "PlanResult",
    "PromotionRejected",
    "RegistrationRejected",
    "RegistryError",
    "ReservationStore",
    "ResidencyState",
    "ResourceProfile",
    "Slot",
    "SpecError",
    "TenantState",
    "UnknownEngineError",
    "card_totals_from_nvml",
    "free_bytes_from_nvml",
    "plan_reservation",
]
