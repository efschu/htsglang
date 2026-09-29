# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""What a suspected-dead client's resources are worth while we still wait.

The directive behind #344 is explicit that a grace window must not be a
pinning window: between "this consumer has gone quiet" and "this consumer is
dead" the bytes and slots it holds belong to the normal reclamation ladder
(idle tenant #341, pressure staircase #287, spill/offload), not to the
attachment. Nothing here reclaims anything itself. It makes the distinction
*visible*, which is the part that was missing -- before this, a stalled
stream and a busy stream looked identical to every reclaimer in the process.

Three phases, and the middle one is the whole point:

``ACTIVE``
    The transport accepted bytes recently. Hands off.
``GRACE``
    Silent for longer than the class tolerates casually, but not yet long
    enough to declare dead. The claims are listed as reclaimable. A reclaimer
    may take them; if it does and the client comes back, the client gets an
    error instead of a stall, which is the better of the two outcomes.
``DEAD``
    Declared. The watchdog's release has run or is running.

The registry is process-global because its consumers are process-global: the
ledger bridge, the pressure staircase and the dashboard all want one answer
to "what is held but idle right now", not one answer per subsystem.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, replace
from enum import Enum
from typing import Callable, Iterable

logger = logging.getLogger(__name__)

__all__ = [
    "Attachment",
    "AttachmentPhase",
    "AttachmentRegistry",
    "ClaimKind",
    "ResourceClaim",
    "global_attachment_registry",
]


class AttachmentPhase(str, Enum):
    ACTIVE = "active"
    GRACE = "grace"
    DEAD = "dead"


class ClaimKind(str, Enum):
    """What kind of thing an attachment is holding.

    Deliberately coarse. A reclaimer does not need to know that a claim is a
    decoder session rather than an encoder session; it needs to know whether
    the claim is device memory it could have, a scheduler slot it could free,
    or a queue that costs nothing and is not worth waking a ladder for.
    """

    #: Device memory declared in the registry ledger, keyed by card UUID.
    VRAM_LEASE = "vram_lease"
    #: KV blocks plus a running-batch slot, keyed by request id.
    KV = "kv"
    #: A tenant job slot (video enhance job, training job), keyed by job id.
    JOB_SLOT = "job_slot"
    #: A decoder/encoder pipeline or an engine session, keyed by job id.
    PIPELINE = "pipeline"
    #: An in-process queue. Cheap; listed for completeness, not for reclaim.
    SUBSCRIBER = "subscriber"


@dataclass(frozen=True)
class ResourceClaim:
    """One thing an attachment holds, in the terms a reclaimer thinks in."""

    kind: ClaimKind
    #: Card UUID, request id or job id, depending on ``kind``.
    key: str
    #: Device bytes, where the claim is device memory. Zero otherwise, and
    #: zero is honest: a subscriber queue has no byte cost worth reporting.
    nbytes: int = 0
    #: Ledger tenant this claim is booked under, where there is one.
    tenant_id: str = ""

    def describe(self) -> str:
        size = f", {self.nbytes / (1024 * 1024):.0f} MiB" if self.nbytes else ""
        return f"{self.kind.value}:{self.key}{size}"


@dataclass(frozen=True)
class Attachment:
    """One client attachment, as the reclamation ladder sees it."""

    attachment_id: str
    endpoint_class: str
    phase: AttachmentPhase
    claims: tuple[ResourceClaim, ...] = ()
    registered_at: float = 0.0
    phase_since: float = 0.0
    silent_for_s: float = 0.0
    timeout_s: float | None = None

    @property
    def reclaimable(self) -> bool:
        """True while the client is a dead suspect and not yet declared.

        ``DEAD`` is deliberately excluded: at that point the watchdog's own
        release is running, and a second reclaimer racing it would be tearing
        down the same objects from two directions.
        """
        return self.phase is AttachmentPhase.GRACE

    def reclaimable_bytes(self, kind: ClaimKind | None = None) -> int:
        if not self.reclaimable:
            return 0
        return sum(c.nbytes for c in self.claims if kind is None or c.kind is kind)

    def snapshot(self) -> dict:
        return {
            "attachment_id": self.attachment_id,
            "endpoint_class": self.endpoint_class,
            "phase": self.phase.value,
            "reclaimable": self.reclaimable,
            "silent_for_s": round(self.silent_for_s, 3),
            "timeout_s": self.timeout_s,
            "claims": [
                {
                    "kind": c.kind.value,
                    "key": c.key,
                    "nbytes": c.nbytes,
                    "tenant_id": c.tenant_id,
                }
                for c in self.claims
            ],
        }


Observer = Callable[[Attachment], None]


class AttachmentRegistry:
    """Every live attachment in this process, and what phase it is in.

    Lock-guarded rather than loop-affine: a watchdog updates it from the
    serving event loop, the ledger bridge reads it from whatever thread the
    arbiter runs on, and the dashboard reads it from a third. The critical
    sections are dictionary writes, so the lock is never held across a
    callback -- observers run outside it, which is also what keeps a slow
    observer off the serving hot path.
    """

    def __init__(self, *, clock: Callable[[], float] | None = None) -> None:
        self._clock = clock or time.monotonic
        self._lock = threading.Lock()
        self._attachments: dict[str, Attachment] = {}
        self._observers: list[Observer] = []

    # -- observers ---------------------------------------------------------

    def add_observer(self, observer: Observer) -> None:
        """Called on every phase change, outside the registry lock."""
        with self._lock:
            self._observers.append(observer)

    def remove_observer(self, observer: Observer) -> None:
        with self._lock:
            if observer in self._observers:
                self._observers.remove(observer)

    def _notify(self, attachment: Attachment) -> None:
        with self._lock:
            observers = tuple(self._observers)
        for observer in observers:
            try:
                observer(attachment)
            except Exception as exc:  # noqa: BLE001 - an observer must not
                # break the stream it is observing. A ledger that cannot be
                # written is a degraded view, not a reason to drop a client.
                logger.warning(
                    "liveness observer failed for %s: %s",
                    attachment.attachment_id,
                    exc,
                )

    # -- lifecycle ---------------------------------------------------------

    def register(
        self,
        attachment_id: str,
        *,
        endpoint_class: str,
        claims: Iterable[ResourceClaim] = (),
        timeout_s: float | None = None,
    ) -> Attachment:
        now = self._clock()
        attachment = Attachment(
            attachment_id=attachment_id,
            endpoint_class=endpoint_class,
            phase=AttachmentPhase.ACTIVE,
            claims=tuple(claims),
            registered_at=now,
            phase_since=now,
            timeout_s=timeout_s,
        )
        with self._lock:
            self._attachments[attachment_id] = attachment
        return attachment

    def set_phase(
        self,
        attachment_id: str,
        phase: AttachmentPhase,
        *,
        silent_for_s: float = 0.0,
    ) -> Attachment | None:
        """Move an attachment; notify observers only on an actual change."""
        with self._lock:
            current = self._attachments.get(attachment_id)
            if current is None:
                return None
            changed = current.phase is not phase
            updated = replace(
                current,
                phase=phase,
                silent_for_s=silent_for_s,
                phase_since=self._clock() if changed else current.phase_since,
            )
            self._attachments[attachment_id] = updated
        if changed:
            logger.info(
                "attachment %s (%s): %s -> %s after %.1fs of silence",
                attachment_id,
                current.endpoint_class,
                current.phase.value,
                phase.value,
                silent_for_s,
            )
            self._notify(updated)
        return updated

    def unregister(self, attachment_id: str) -> Attachment | None:
        """Drop the attachment and tell observers its claims are gone.

        Observers see a final ``DEAD`` notification rather than a bespoke
        "removed" event: from a reclaimer's side "this attachment no longer
        holds anything" and "this attachment is dead" want the same handling,
        and one code path is one code path fewer to get wrong.
        """
        with self._lock:
            attachment = self._attachments.pop(attachment_id, None)
        if attachment is None:
            return None
        final = replace(attachment, phase=AttachmentPhase.DEAD)
        if attachment.phase is not AttachmentPhase.DEAD:
            self._notify(final)
        return final

    # -- queries -----------------------------------------------------------

    def get(self, attachment_id: str) -> Attachment | None:
        with self._lock:
            return self._attachments.get(attachment_id)

    def snapshot(self) -> tuple[Attachment, ...]:
        with self._lock:
            return tuple(self._attachments.values())

    def reclaimable(self) -> tuple[Attachment, ...]:
        return tuple(a for a in self.snapshot() if a.reclaimable)

    def reclaimable_bytes(
        self, kind: ClaimKind | None = None, key: str | None = None
    ) -> int:
        """Device bytes held by dead suspects, optionally for one card.

        This is the number the #287 pressure staircase wants: how much of the
        pressure on a card is being held by attachments nobody is reading.
        """
        total = 0
        for attachment in self.reclaimable():
            for claim in attachment.claims:
                if kind is not None and claim.kind is not kind:
                    continue
                if key is not None and claim.key != key:
                    continue
                total += claim.nbytes
        return total

    def describe(self) -> dict:
        attachments = self.snapshot()
        return {
            "attachments": [a.snapshot() for a in attachments],
            "active": sum(1 for a in attachments if a.phase is AttachmentPhase.ACTIVE),
            "in_grace": sum(1 for a in attachments if a.reclaimable),
            "reclaimable_bytes": self.reclaimable_bytes(),
        }


_GLOBAL_REGISTRY: AttachmentRegistry | None = None
_GLOBAL_LOCK = threading.Lock()


def global_attachment_registry() -> AttachmentRegistry:
    """The process-wide registry. Created on first use, never replaced."""
    global _GLOBAL_REGISTRY
    if _GLOBAL_REGISTRY is None:
        with _GLOBAL_LOCK:
            if _GLOBAL_REGISTRY is None:
                _GLOBAL_REGISTRY = AttachmentRegistry()
    return _GLOBAL_REGISTRY
