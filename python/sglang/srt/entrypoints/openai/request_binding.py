# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""#305: the request path's binding to the engine registry.

The #305 determination (``docs/dev/VERDICT_305_multi_model.md``) found the
control plane BUILT and the binding ABSENT: ``EngineRegistry`` has a real
``acquire_for_request`` (``registry/arbiter.py``) and real promote/demote
actuators, but the only caller of any state-changing verb was the admin route
``POST /registry/engines/{id}/state``. The OpenAI serving path never asked the
registry for anything. This module is that ask.

Its shape follows three constraints, in this order.

**Off by default, and free when off.** Every boot on this rig today serves
exactly one model and runs no control plane. Such a boot must not pay a
registry lookup, a socket, or a dictionary probe per request, and it must
return the same bytes it returned before this module existed. So
:func:`binding_enabled` is one module-global boolean read, false until
something calls :func:`enable_binding`, and the serving path branches on it
before touching anything here. That is pinned by test, both as "no lookup
happened" and as "the response is byte-identical".

**A refusal, never a hang.** A model that is registered but cold and cannot be
woken -- no room, an eviction that would cost more than the caller's budget, or
a ladder edge that is not implemented for its class -- produces a named HTTP
error with the arbiter's own numbers in it. The one thing it must never do is
block the connection until something times out somewhere else. Every path here
is bounded: the HTTP client by an explicit socket timeout, the in-process one
by ``max_promotion_wait_ms``, which is what makes the arbiter refuse a
too-expensive promotion up front instead of starting it.

**Two binders, one contract.** The registry is normally a separate service
(§7.4: "an engine's port comes and goes with the engine"), so the ordinary
binder speaks HTTP to it. But the same contract is implemented in-process, and
that is not only for tests: a single process that owns its own registry can
bind without a socket. Both raise the same :class:`BindingRefused`.

The hold is released in a ``finally`` on the serving side, and for a streaming
response only after the last chunk -- a generation that outlives its
``handle_request`` frame is still in flight, and the control tick reads exactly
this count to decide it may not demote.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import urllib.error
import urllib.request
from typing import Any, Mapping, Optional, Protocol

logger = logging.getLogger(__name__)

#: Turns the binding on when no explicit binder is installed by code. The value
#: is the control-plane URL, or "1"/"true" to take the address from
#: ``registry_view.registry_base_url()``.
BINDING_ENV = "SGLANG_REQUEST_BINDING"

#: Per-request promotion budget, milliseconds. A finite default is the point:
#: with no budget the arbiter is allowed to start a promotion of any cost, and
#: a cold 27B boot is minutes -- which the client sees as a hang. With one, an
#: engine that cannot be woken cheaply is a 503 that says how long it WOULD
#: have taken.
BINDING_MAX_WAIT_ENV = "SGLANG_REQUEST_BINDING_MAX_WAIT_MS"
DEFAULT_MAX_PROMOTION_WAIT_MS = 30_000.0

#: Socket timeout for the control-plane call. Separate from the promotion
#: budget: this bounds the conversation, that bounds the work.
DEFAULT_CONTROL_TIMEOUT_S = 5.0


class BindingRefused(Exception):
    """A request that will not be served, with the HTTP shape to say so.

    Carries ``code`` as well as ``err_type`` because the OpenAI error body has
    both, and a client that wants to retry needs to tell "this model does not
    exist here" (never retry) from "this model could not be woken right now"
    (retry, or pick a hot one).
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        err_type: str,
        code: str,
        detail: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = int(status_code)
        self.err_type = err_type
        self.code = code
        self.detail: Mapping[str, Any] = dict(detail or {})


class RequestHold:
    """One acquired engine, released exactly once."""

    __slots__ = ("engine_id", "_binder", "_released", "_lock", "state")

    def __init__(self, engine_id: str, binder: "Binder", state: str = "") -> None:
        self.engine_id = engine_id
        self.state = state
        self._binder = binder
        self._released = False
        self._lock = threading.Lock()

    @property
    def released(self) -> bool:
        return self._released

    def release(self) -> None:
        with self._lock:
            if self._released:
                return
            self._released = True
        try:
            self._binder.release_after_request(self.engine_id)
        except Exception as exc:  # noqa: BLE001 - a release must not fail a response
            logger.warning(
                "request binding: releasing %s failed: %s", self.engine_id, exc
            )

    def __enter__(self) -> "RequestHold":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.release()


class Binder(Protocol):
    def acquire_for_request(self, engine_id: str) -> Mapping[str, Any]: ...

    def release_after_request(self, engine_id: str) -> None: ...


class InProcessBinder:
    """Binds against an :class:`~sglang.srt.registry.arbiter.EngineRegistry`.

    Translates the arbiter's three refusals into the three HTTP shapes. It does
    not re-derive any of them: the message a client sees is the arbiter's own
    text, because that text already carries the numbers (projected wait, which
    tenants would be evicted) that let a caller decide rather than just retry.
    """

    def __init__(self, registry: Any, *, max_promotion_wait_ms: float | None = None):
        self.registry = registry
        self.max_promotion_wait_ms = (
            DEFAULT_MAX_PROMOTION_WAIT_MS
            if max_promotion_wait_ms is None
            else float(max_promotion_wait_ms)
        )

    def acquire_for_request(self, engine_id: str) -> Mapping[str, Any]:
        from sglang.srt.registry.arbiter import (  # noqa: PLC0415
            PromotionRejected,
            RegistryError,
            UnknownEngineError,
        )
        from sglang.srt.registry.ladder import LadderRefusal  # noqa: PLC0415

        try:
            instance = self.registry.acquire_for_request(
                engine_id, max_promotion_wait_ms=self.max_promotion_wait_ms
            )
        except UnknownEngineError as exc:
            raise _not_registered(engine_id, str(exc)) from None
        except LadderRefusal as exc:
            raise _edge_unbuilt(engine_id, str(exc)) from None
        except PromotionRejected as exc:
            raise _not_wakeable(engine_id, str(exc), exc.to_json()) from None
        except RegistryError as exc:
            raise _control_plane_error(engine_id, str(exc)) from None
        return {"engine_id": engine_id, "state": getattr(instance.state, "value", "")}

    def release_after_request(self, engine_id: str) -> None:
        self.registry.release_after_request(engine_id)


class HttpBinder:
    """Binds against the registry control plane over its own HTTP surface.

    The normal case: the serving process and the control plane are separate
    (§7.4). Both calls are bounded by :attr:`timeout_s`; an unreachable control
    plane is a 503 that names it, never a stalled request.
    """

    def __init__(
        self,
        base_url: str,
        *,
        max_promotion_wait_ms: float | None = None,
        timeout_s: float = DEFAULT_CONTROL_TIMEOUT_S,
        api_key: str | None = None,
        opener: Any = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.max_promotion_wait_ms = (
            DEFAULT_MAX_PROMOTION_WAIT_MS
            if max_promotion_wait_ms is None
            else float(max_promotion_wait_ms)
        )
        self.timeout_s = float(timeout_s)
        self.api_key = api_key
        #: Injectable so a test can exercise every refusal without a socket.
        self._opener = opener or urllib.request.urlopen

    def _post(self, path: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        body = json.dumps(dict(payload)).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.base_url + path, data=body, headers=headers, method="POST"
        )
        with self._opener(request, timeout=self.timeout_s) as response:
            return json.loads(response.read() or b"{}")

    def acquire_for_request(self, engine_id: str) -> Mapping[str, Any]:
        try:
            return self._post(
                f"/registry/engines/{engine_id}/acquire",
                {"max_promotion_wait_ms": self.max_promotion_wait_ms},
            )
        except urllib.error.HTTPError as exc:
            raise _from_control_plane_status(engine_id, exc) from None
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise _control_plane_unreachable(engine_id, self.base_url, exc) from None

    def release_after_request(self, engine_id: str) -> None:
        try:
            self._post(f"/registry/engines/{engine_id}/release", {})
        except Exception as exc:  # noqa: BLE001 - see RequestHold.release
            logger.warning(
                "request binding: control plane did not accept the release of "
                "%s: %s. The engine stays counted as in flight until the next "
                "acquire/release pair balances it.",
                engine_id,
                exc,
            )


# -- refusals, in one place so their wording is reviewable together ----------


def _not_registered(engine_id: str, detail: str) -> BindingRefused:
    return BindingRefused(
        f"The model '{engine_id}' is not registered with this deployment's "
        f"engine registry. {detail}",
        status_code=404,
        err_type="invalid_request_error",
        code="model_not_found",
        detail={"engine_id": engine_id},
    )


def _edge_unbuilt(engine_id: str, detail: str) -> BindingRefused:
    # 409, not 503: waiting does not help. The transition this engine would
    # need is not implemented for its class at all, and the message says which
    # rung is missing and why (the ladder quotes the architecture).
    return BindingRefused(
        f"The model '{engine_id}' is registered but cannot be made hot from "
        f"where it currently is: {detail}",
        status_code=409,
        err_type="invalid_request_error",
        code="ladder_edge_unbuilt",
        detail={"engine_id": engine_id},
    )


def _not_wakeable(
    engine_id: str, detail: str, report: Optional[Mapping[str, Any]] = None
) -> BindingRefused:
    return BindingRefused(
        f"The model '{engine_id}' is registered but could not be made hot for "
        f"this request: {detail}",
        status_code=503,
        err_type="service_unavailable",
        code="engine_not_wakeable",
        detail=dict(report or {"engine_id": engine_id}),
    )


def _control_plane_error(engine_id: str, detail: str) -> BindingRefused:
    return BindingRefused(
        f"The engine registry refused to bind '{engine_id}': {detail}",
        status_code=503,
        err_type="service_unavailable",
        code="registry_error",
        detail={"engine_id": engine_id},
    )


def _control_plane_unreachable(
    engine_id: str, url: str, exc: BaseException
) -> BindingRefused:
    return BindingRefused(
        f"The engine registry at {url} could not be reached to bind "
        f"'{engine_id}': {exc}. The request is refused rather than held open.",
        status_code=503,
        err_type="service_unavailable",
        code="registry_unreachable",
        detail={"engine_id": engine_id, "registry_url": url},
    )


def _from_control_plane_status(
    engine_id: str, exc: urllib.error.HTTPError
) -> BindingRefused:
    """Carry the control plane's own status and body through to the client."""
    try:
        body = json.loads(exc.read() or b"{}")
    except Exception:  # noqa: BLE001 - a malformed body is not a second failure
        body = {}
    message = str(body.get("message") or body.get("error") or exc.reason or exc)
    if exc.code == 404:
        return _not_registered(engine_id, message)
    if exc.code == 409:
        return _edge_unbuilt(engine_id, message)
    if exc.code == 503:
        return _not_wakeable(engine_id, message, body)
    return _control_plane_error(engine_id, f"HTTP {exc.code}: {message}")


# -- the module-global switch ------------------------------------------------

_state_lock = threading.Lock()
#: The ONE thing the fast path reads. False until enable_binding() runs.
_enabled: bool = False
_binder: Optional[Binder] = None


def binding_enabled() -> bool:
    """One global boolean. The single-model fast path costs exactly this."""
    return _enabled


def enable_binding(binder: Binder) -> None:
    global _enabled, _binder
    with _state_lock:
        _binder = binder
        _enabled = True
    logger.info(
        "request binding: ON via %s. Every OpenAI request now resolves its "
        "model through the engine registry and holds it for the request's "
        "lifetime.",
        type(binder).__name__,
    )


def disable_binding() -> None:
    global _enabled, _binder
    with _state_lock:
        _enabled = False
        _binder = None


def current_binder() -> Optional[Binder]:
    return _binder


def binder_from_env() -> Optional[Binder]:
    """Build the configured binder, or ``None`` when nothing configures one.

    ``SGLANG_REQUEST_BINDING`` unset is the single-model boot and must stay
    exactly that, so this returns ``None`` rather than guessing a URL.
    """
    raw = (os.environ.get(BINDING_ENV) or "").strip()
    if not raw or raw.lower() in {"0", "false", "off", "no"}:
        return None
    if raw.lower() in {"1", "true", "on", "yes"}:
        from sglang.srt.entrypoints.openai.registry_view import (  # noqa: PLC0415
            registry_base_url,
        )

        url = registry_base_url()
        if not url:
            logger.warning(
                "request binding: %s is on but the registry address is "
                "disabled; binding stays off.",
                BINDING_ENV,
            )
            return None
    else:
        url = raw if raw.startswith("http") else "http://" + raw
    wait = os.environ.get(BINDING_MAX_WAIT_ENV)
    return HttpBinder(
        url,
        max_promotion_wait_ms=None if not wait else float(wait),
        api_key=os.environ.get("SGLANG_REGISTRY_API_KEY") or None,
    )


def init_binding_from_env() -> bool:
    """Install the env-configured binder at server start. Returns whether on."""
    binder = binder_from_env()
    if binder is None:
        return False
    enable_binding(binder)
    return True


# -- the request path's entry point ------------------------------------------


def acquire(model: str) -> RequestHold:
    """Resolve ``model`` to a registered engine and hold it.

    Only ever called with the binding on; the caller checks
    :func:`binding_enabled` first so that the off case reaches no code here.
    """
    binder = _binder
    if binder is None:  # pragma: no cover - enable/disable race, refuse cleanly
        raise _control_plane_error(model, "binding is on but no binder is installed")
    name = (model or "").strip()
    if not name:
        raise BindingRefused(
            "This deployment binds every request to a registered engine, so "
            "the request must name a model.",
            status_code=400,
            err_type="invalid_request_error",
            code="model_required",
        )
    result = binder.acquire_for_request(name)
    return RequestHold(name, binder, state=str(result.get("state") or ""))
