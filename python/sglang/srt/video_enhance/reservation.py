# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Re-export of the shared VRAM reservation ledger.

The implementation moved to :mod:`sglang.srt.registry.ledger` in #333-M1. M2
wrote it here first because the Class-3 video-enhance tenant landed ahead of
the registry and needed the §3.3 invariant before there was an arbiter to own
it. M1 makes that tenant the ledger's first client rather than its owner: one
store, one invariant, one lock discipline, shared by every class.

This module stays as the name M2's code and tests already import. It adds
nothing and overrides nothing -- a second implementation of the invariant is
exactly the failure the ledger exists to prevent.
"""

from __future__ import annotations

from sglang.srt.registry.ledger import (
    DEFAULT_CORRIDOR_BYTES,
    DEFAULT_LEASE_SECONDS,
    GPU_RESIDENT_STATES,
    MIB,
    CardBusyError,
    CardLedger,
    LedgerError,
    ReservationEntry,
    ReservationRejected,
    ReservationStore,
    TenantState,
    UnknownTenantError,
    available_bytes,
    default_store_root,
    tenant_label,
)

__all__ = [
    "DEFAULT_CORRIDOR_BYTES",
    "DEFAULT_LEASE_SECONDS",
    "GPU_RESIDENT_STATES",
    "MIB",
    "CardBusyError",
    "CardLedger",
    "LedgerError",
    "ReservationEntry",
    "ReservationRejected",
    "ReservationStore",
    "TenantState",
    "UnknownTenantError",
    "available_bytes",
    "default_store_root",
    "tenant_label",
]
