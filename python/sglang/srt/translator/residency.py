# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Residency events: when a tenant gives VRAM back, and when it takes it again.

#546 frees ~5.9 GiB on the shared card while the translator is idle. Freeing
it is only half the value: bytes that nobody is told about are bytes nobody
can spend. The serving engine sizes its KV pool once at boot from a reserve
that assumes the co-tenant is entitled to its whole declared budget (the
runbook rule at §4.1: coexistence reserves come from the declared budget,
never from a momentary observation) -- so an idle translator's 5.9 GiB stays
unusable unless something SAYS it is available, and says it in a form that can
be acted on rather than read.

This module is that announcement, and nothing more. #553 (elastic
co-residency) owns the reaction: it will drive the KV-pressure stairs, the
GDN slot ladder and the #330 VRAM budget dial from these events. Nothing here
knows those exist, and deliberately so -- a tenant that reaches into the
serving engine's sizing is a coupling that would have to be undone the moment
a second tenant appears.

THE CONTRACT, which #553 depends on and which tests pin:

* Two events bracket every transition, and they are asymmetric on purpose.
  ``park_complete`` fires AFTER the bytes are back with the driver, because a
  consumer may only claim memory that has actually been released.
  ``wake_start`` fires BEFORE the restore, because a consumer must be told to
  get out of the way while it still can. A symmetric pair (both after) would
  make the wake a race the consumer always loses.
* ``wake_complete`` follows, carrying the measured wake latency. It is a
  report, not a signal to act on.
* Card identity is the NVML UUID, never a torch ordinal. The two enumerations
  diverge on this rig (cuda:0 = 5090 = nvml:1), and an event naming the wrong
  card would have the consumer free memory on a card that never needed it.
  Resolution goes through the ONE identity map (#331,
  ``registry.nvml.IdentityMap``); when it cannot resolve, the event says so
  (``card_resolved: false``) rather than guessing an index.
* MiB is per card, signed by the event type: ``freed`` on a park, ``claimed``
  on a wake. Both are positive magnitudes; the direction is the event name.

Transport is pluggable and default-inert. The log marker always fires, so a
deployment that configures nothing still leaves a machine-readable trail in
its journal. :func:`add_sink` is the in-process route (#553's runtime will
register one), and :func:`http_sink` is the cheap cross-process route for the
period where the two live in different processes -- fire-and-forget on a
daemon thread with a short timeout, because a tenant must never block a turn
on a telemetry consumer, nor die because one is down.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import threading
import time
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "EVENT_PARK_COMPLETE",
    "EVENT_WAKE_COMPLETE",
    "EVENT_WAKE_START",
    "MARKER",
    "CardResidency",
    "ResidencyEvent",
    "add_sink",
    "clear_sinks",
    "emit",
    "http_sink",
    "resolve_card",
]

#: The log marker. Grep-able from a journal, parseable by splitting once on
#: the marker and json-loading the tail. Stable: #553 and the runbook both
#: quote it, so renaming it is an interface change.
MARKER = "RESIDENCY_EVENT"

EVENT_PARK_COMPLETE = "park_complete"
EVENT_WAKE_START = "wake_start"
EVENT_WAKE_COMPLETE = "wake_complete"

_EVENTS = (EVENT_PARK_COMPLETE, EVENT_WAKE_START, EVENT_WAKE_COMPLETE)

_sink_lock = threading.Lock()
_sinks: List[Callable[["ResidencyEvent"], None]] = []


@dataclasses.dataclass(frozen=True)
class CardResidency:
    """How many MiB moved on ONE physical card, named by NVML identity."""

    #: NVML UUID, normalized lowercase hex. Empty when unresolved.
    card_uuid: str
    #: NVML/PCI index. None when unresolved -- never a torch ordinal.
    nvml_index: Optional[int]
    card_name: str
    mib: float

    @property
    def card_resolved(self) -> bool:
        return bool(self.card_uuid)

    def to_json(self) -> Dict[str, object]:
        return {
            "card_uuid": self.card_uuid,
            "nvml_index": self.nvml_index,
            "card_name": self.card_name,
            "card_resolved": self.card_resolved,
            "mib": round(self.mib, 2),
        }


@dataclasses.dataclass(frozen=True)
class ResidencyEvent:
    """One residency transition of one tenant, per card."""

    tenant_id: str
    event: str
    cards: Tuple[CardResidency, ...]
    #: Wall clock, so a consumer in another process can order events against
    #: its own timeline. Monotonic clocks are not comparable across processes.
    at_s: float = dataclasses.field(default_factory=time.time)
    #: Free-form, never load-bearing for a consumer: reason strings, measured
    #: wake milliseconds, per-asset breakdown.
    detail: Dict[str, object] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.event not in _EVENTS:
            raise ValueError(
                f"unknown residency event {self.event!r}; known: {_EVENTS}. "
                "A consumer keys on this string, so a typo would be a signal "
                "that is emitted and never received."
            )

    @property
    def total_mib(self) -> float:
        return sum(c.mib for c in self.cards)

    def to_json(self) -> Dict[str, object]:
        return {
            "tenant_id": self.tenant_id,
            "event": self.event,
            "at_s": round(self.at_s, 3),
            "total_mib": round(self.total_mib, 2),
            "cards": [c.to_json() for c in self.cards],
            "detail": dict(self.detail),
        }

    def marker_line(self) -> str:
        return f"{MARKER} {json.dumps(self.to_json(), sort_keys=True)}"


def add_sink(sink: Callable[[ResidencyEvent], None]) -> None:
    """Register an in-process consumer. Idempotent per callable object."""
    with _sink_lock:
        if sink not in _sinks:
            _sinks.append(sink)


def clear_sinks() -> None:
    """Test hook; the sink list is process-global by design."""
    with _sink_lock:
        _sinks.clear()


def emit(event: ResidencyEvent) -> ResidencyEvent:
    """Log the marker and fan out to every sink. Never raises.

    A sink that throws is logged and skipped, not propagated: the park/wake
    state machine is the thing that must not break, and a telemetry consumer
    is not allowed to take it down.
    """
    logger.info("%s", event.marker_line())
    with _sink_lock:
        sinks = list(_sinks)
    for sink in sinks:
        try:
            sink(event)
        except Exception:  # noqa: BLE001 - see docstring
            logger.exception("residency sink %r failed", getattr(sink, "__name__", sink))
    return event


def http_sink(url: str, timeout_s: float = 1.0) -> Callable[[ResidencyEvent], None]:
    """A fire-and-forget HTTP POST sink for a cross-process consumer.

    Runs on a daemon thread with a short timeout. A tenant blocking a turn on
    a telemetry POST would have traded the latency this whole feature protects
    for a piece of bookkeeping.
    """

    def _post(event: ResidencyEvent) -> None:
        payload = json.dumps(event.to_json()).encode("utf-8")

        def _send() -> None:
            import urllib.request

            request = urllib.request.Request(
                url, data=payload, headers={"Content-Type": "application/json"}
            )
            try:
                with urllib.request.urlopen(request, timeout=timeout_s):
                    pass
            except Exception as exc:  # noqa: BLE001 - never fatal
                logger.debug("residency POST to %s failed: %s", url, exc)

        threading.Thread(target=_send, name="residency-post", daemon=True).start()

    _post.__name__ = f"http_sink({url})"
    return _post


#: Resolved once: the identity map costs an NVML session, and the card a
#: process is pinned to cannot change while it runs.
_card_cache: Dict[str, Optional[Tuple[str, Optional[int], str]]] = {}
_card_lock = threading.Lock()


def resolve_card(device: Optional[str] = None) -> Tuple[str, Optional[int], str]:
    """Resolve a torch device string to (uuid, nvml_index, name) via NVML.

    Returns ``("", None, "")`` when NVML cannot place the card -- on a desk
    with no GPU, or on a rig whose CUDA order the identity map refuses to
    bridge. That is the honest answer and the event carries it as
    ``card_resolved: false``; the alternative, reporting the torch ordinal as
    if it were a physical index, is the exact confusion #331 exists to end.
    """
    key = device or "cuda:0"
    with _card_lock:
        if key in _card_cache:
            cached = _card_cache[key]
            return cached if cached is not None else ("", None, "")

    resolved: Tuple[str, Optional[int], str] = ("", None, "")
    try:
        from sglang.srt.registry import nvml as nvml_registry

        ordinal = 0
        if key.startswith("cuda:"):
            ordinal = int(key.split(":", 1)[1] or 0)
        cards = nvml_registry.identity_map()
        card = cards.by_cuda_ordinal(ordinal)
        if card is None:
            # The process is pinned by CUDA_VISIBLE_DEVICES to exactly one
            # card, so ordinal 0 IS the pin and current_device_uuid answers
            # without needing the CUDA<->NVML bridge at all.
            uuid = nvml_registry.current_device_uuid()
            card = cards.get(uuid)
        if card is not None:
            resolved = (card.uuid, card.nvml_index, card.name)
    except Exception as exc:  # noqa: BLE001 - a desk has no NVML
        logger.debug("could not resolve card identity for %s: %s", key, exc)

    with _card_lock:
        _card_cache[key] = resolved
    return resolved


def reset_card_cache() -> None:
    """Test hook."""
    with _card_lock:
        _card_cache.clear()


def cards_from_bytes(by_device: Dict[str, int]) -> Tuple[CardResidency, ...]:
    """Group a ``{torch device: bytes}`` breakdown into per-CARD residency.

    Two torch devices can never be one card here (the tenant is pinned to one
    card), but the grouping is written for the general case anyway: an
    unresolved card groups under one empty-uuid bucket rather than being
    dropped, because a consumer that sees no card at all can at least log a
    gap, while a silently dropped one is invisible.
    """
    grouped: Dict[Tuple[str, Optional[int], str], float] = {}
    for device, nbytes in sorted(by_device.items()):
        if str(device).startswith("cpu") or str(device) == "meta":
            continue
        identity = resolve_card(str(device))
        grouped[identity] = grouped.get(identity, 0.0) + nbytes / (1 << 20)
    return tuple(
        CardResidency(card_uuid=uuid, nvml_index=index, card_name=name, mib=mib)
        for (uuid, index, name), mib in sorted(
            grouped.items(), key=lambda kv: (kv[0][1] is None, kv[0][1], kv[0][0])
        )
    )
