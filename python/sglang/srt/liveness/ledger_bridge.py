# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Publishing grace phase into the VRAM ledger, so reclaimers can see it.

The attachment registry knows which streams have gone quiet. The ledger
(#305-M1) knows which tenant holds how many bytes on which card. Neither one
alone answers the question the reclamation ladder actually asks -- "how much
of the pressure on this card is held by somebody nobody is reading from" --
and this is the join.

It is deliberately one-directional and advisory. The bridge writes a flag;
it never releases a reservation, never shortens a lease and never rejects an
acquisition. Reclamation stays with the ladder that owns the policy (#287's
staircase, #341's idle tenant), which is the only place that can weigh a
grace-held claim against what wants the bytes.

Why the ledger and not the in-process registry directly: the ledger is the
cross-process view. A pressure staircase running in the arbiter, a second
serving process on the same card, and ``registry`` CLI output all read the
ledger file and none of them can see this process's Python objects.
"""

from __future__ import annotations

import logging

from sglang.srt.liveness.grace import (
    Attachment,
    AttachmentPhase,
    AttachmentRegistry,
    ClaimKind,
)

logger = logging.getLogger(__name__)

__all__ = ["LedgerGraceBridge", "attach_ledger_grace_bridge"]


class LedgerGraceBridge:
    """Registry observer that mirrors grace phase onto ledger entries.

    Only ``vram_lease`` claims are mirrored, and only those that name both a
    card UUID and a tenant id -- those are the two coordinates a ledger entry
    is addressed by. An attachment that holds a decoder and a queue but no
    declared device bytes has nothing to say to the ledger and is skipped.
    """

    def __init__(self, store) -> None:
        self._store = store

    def __call__(self, attachment: Attachment) -> None:
        in_grace = attachment.phase is AttachmentPhase.GRACE
        for claim in attachment.claims:
            if claim.kind is not ClaimKind.VRAM_LEASE:
                continue
            if not claim.key or not claim.tenant_id:
                continue
            try:
                self._store.set_grace(claim.key, claim.tenant_id, in_grace)
            except Exception as exc:  # noqa: BLE001 - a ledger the bridge
                # cannot write is a degraded view of the rig, not a reason to
                # interfere with the stream that triggered the notification.
                logger.warning(
                    "could not mark %s on card %s as %s in the ledger: %s",
                    claim.tenant_id,
                    claim.key,
                    "in grace" if in_grace else "active",
                    exc,
                )


def attach_ledger_grace_bridge(
    registry: AttachmentRegistry, store
) -> LedgerGraceBridge:
    """Wire the bridge onto a registry and return it (for later removal)."""
    bridge = LedgerGraceBridge(store)
    registry.add_observer(bridge)
    return bridge
