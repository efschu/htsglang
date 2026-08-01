# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Read-only view of the #305-M1 engine registry, for the OpenAI surface.

The registry control plane is a separate service on its own port (§7.4:
"an engine's port comes and goes with the engine; the question what is
resident has to be answerable while every engine is cold"). This module is the
serving side's read-only client for it, and nothing more: it never registers,
promotes or demotes. Those are control-plane verbs and they stay on the
control plane.

Two properties matter for the endpoints that use this.

*Optional.* A server started without a registry is the ordinary single-model
case, and it must answer ``/v1/models`` exactly as before. An unreachable or
unconfigured registry is therefore not an error -- it is an empty view, and
the caller falls back to what the tokenizer manager knows.

*Honest.* When the registry does answer, the residency state is reported as
the registry has it (``HOT``/``WARM_GPU``/``WARM_HOST``/``COLD``), not as a
boolean "available". A model listed as ``COLD`` is a real, registered model
that would need a promotion first; hiding that behind an omission would make
``/v1/models`` a list of guesses.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

#: Same default port and env var as the planner dashboard uses
#: (``sglang.srt.planner.webui``); one registry, one address, one spelling.
DEFAULT_REGISTRY_URL = "http://127.0.0.1:8500"
REGISTRY_URL_ENV = "SGLANG_REGISTRY_URL"

#: A ``/v1/models`` call must not block on a service that may be down. The
#: cache keeps a listing cheap under polling clients; the timeout keeps a dead
#: registry from turning every listing into a stall.
_CACHE_TTL_S = 2.0
_TIMEOUT_S = 1.5

#: Engine classes, from ``sglang.srt.registry.spec.EngineClass``. Imported as
#: literals so this module stays importable with no registry dependency.
CLASS_AUTOREGRESSIVE = 1
CLASS_DIFFUSION = 2
CLASS_UTILITY = 3

#: The registry's ``GPU_RESIDENT_STATES``, same reason.
GPU_RESIDENT_STATES = frozenset({"HOT", "WARM_GPU"})


def registry_base_url() -> str:
    """Configured registry address, or ``""`` when explicitly disabled.

    Setting ``SGLANG_REGISTRY_URL`` to an empty string disables the lookup
    outright, which is how a deployment says "single model, no control plane"
    without paying a failed connection per listing.
    """
    raw = os.environ.get(REGISTRY_URL_ENV)
    if raw is None:
        return DEFAULT_REGISTRY_URL
    raw = raw.strip().rstrip("/")
    if not raw:
        return ""
    if not raw.startswith("http"):
        raw = "http://" + raw
    return raw


@dataclass(frozen=True)
class RegisteredEngine:
    """One registered engine, as the OpenAI surface needs to see it."""

    engine_id: str
    klass: int
    state: str
    cards: tuple[str, ...] = ()
    pinned: bool = False
    reserved_bytes: int = 0
    health: str = "unknown"
    promotion_cost_ms: Optional[float] = None
    last_error: Optional[str] = None
    adapter: str = ""
    #: Wall-clock of the last time the arbiter handed this engine to a
    #: request. The idle tenant (#341) reads it to decide whether the rig has
    #: been quiet long enough to train on.
    last_used_ts: float = 0.0

    @property
    def is_gpu_resident(self) -> bool:
        return self.state in GPU_RESIDENT_STATES

    @property
    def is_hot(self) -> bool:
        return self.state == "HOT"

    def to_extension(self) -> dict[str, Any]:
        """The ``x-htsglang`` block for this engine's model card."""
        block: dict[str, Any] = {
            "engine_id": self.engine_id,
            "engine_class": self.klass,
            "residency": self.state,
            "gpu_resident": self.is_gpu_resident,
            "cards": list(self.cards),
            "pinned": self.pinned,
            "reserved_mib": self.reserved_bytes // (1 << 20),
            "health": self.health,
        }
        # #305 cut 1: the ladder view. Reported alongside the registry's own
        # fields, never in place of them -- `promote_cost_class` is a-priori
        # (the ladder's measured record) and says so, while
        # `measured_promotion_ms` below stays the only per-engine measurement.
        try:
            from sglang.srt.registry.rungs import rung_extension

            block.update(
                rung_extension(
                    self.state,
                    ever_staged=bool(getattr(self, "ever_staged", True)),
                    reserved_bytes=self.reserved_bytes,
                    pinned=self.pinned,
                    engine_geometry=getattr(self, "world_geometry", None),
                    active_geometry=getattr(self, "active_geometry", None),
                )
            )
        except Exception:
            # A reporting layer must never break the listing. Without the
            # ladder fields a client still gets the registry's own residency.
            pass
        if self.promotion_cost_ms is not None:
            # Measured, not guessed -- the registry only fills this in after it
            # has observed the transition once.
            block["measured_promotion_ms"] = round(self.promotion_cost_ms, 1)
        if self.last_error:
            block["last_error"] = self.last_error
        return block


@dataclass
class RegistryView:
    """Cached snapshot of the registry. Empty when there is none."""

    reachable: bool = False
    url: str = ""
    engines: tuple[RegisteredEngine, ...] = ()
    error: Optional[str] = None
    fetched_at: float = field(default_factory=time.time)

    def by_id(self, engine_id: str) -> Optional[RegisteredEngine]:
        for engine in self.engines:
            if engine.engine_id == engine_id:
                return engine
        return None

    def of_class(self, klass: int) -> tuple[RegisteredEngine, ...]:
        return tuple(e for e in self.engines if e.klass == klass)

    def to_extension(self) -> dict[str, Any]:
        """The ``x-htsglang`` block describing the registry itself."""
        block: dict[str, Any] = {"registry_url": self.url, "reachable": self.reachable}
        if self.error:
            block["registry_error"] = self.error
        return block


_lock = threading.Lock()
_cached: Optional[RegistryView] = None


def _parse(payload: dict[str, Any], url: str) -> RegistryView:
    engines = []
    for item in payload.get("engines") or []:
        try:
            engines.append(
                RegisteredEngine(
                    engine_id=str(item["engine_id"]),
                    klass=int(item.get("klass", CLASS_AUTOREGRESSIVE)),
                    state=str(item.get("state", "COLD")),
                    cards=tuple(item.get("cards") or ()),
                    pinned=bool(item.get("pinned", False)),
                    reserved_bytes=int(item.get("reserved_bytes") or 0),
                    health=str(item.get("health", "unknown")),
                    promotion_cost_ms=(
                        None
                        if item.get("promotion_cost_ms") is None
                        else float(item["promotion_cost_ms"])
                    ),
                    last_error=item.get("last_error"),
                    last_used_ts=float(item.get("last_used_ts") or 0.0),
                    adapter=str((item.get("spec") or {}).get("adapter", "")),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            # One malformed entry must not blank the whole listing.
            logger.debug("registry view: skipping unparsable engine entry: %s", exc)
    return RegistryView(reachable=True, url=url, engines=tuple(engines))


def fetch_registry_view(force: bool = False) -> RegistryView:
    """Current registry snapshot, cached for :data:`_CACHE_TTL_S` seconds."""
    global _cached
    url = registry_base_url()
    if not url:
        return RegistryView(reachable=False, url="", error="registry lookup disabled")

    now = time.time()
    with _lock:
        cached = _cached
        if (
            not force
            and cached is not None
            and cached.url == url
            and now - cached.fetched_at < _CACHE_TTL_S
        ):
            return cached

    try:
        request = urllib.request.Request(url + "/registry", method="GET")
        with urllib.request.urlopen(request, timeout=_TIMEOUT_S) as response:
            view = _parse(json.loads(response.read() or b"{}"), url)
    except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError) as exc:
        # Not an error path for the caller: no registry is a valid deployment.
        view = RegistryView(reachable=False, url=url, error=str(exc))

    with _lock:
        _cached = view
    return view


def reset_cache() -> None:
    """Drop the cached snapshot. For tests and for a forced re-read."""
    global _cached
    with _lock:
        _cached = None
