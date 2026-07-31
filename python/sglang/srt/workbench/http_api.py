# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""The workbench's HTTP payloads (DESIGN #347 W8).

Framework-free on purpose, the same split the training surface draws: these
functions parse and shape, the service decides, and the routes in
``entrypoints/http_server.py`` do neither. That is what makes the surface
testable without a server.

Read-only state plus two controls, under the ``x-htsglang`` namespace because
none of it is in anybody's standard protocol:

* ``GET  /x-htsglang/workbench``          the snapshot
* ``GET  /x-htsglang/workbench/events``   the cursor-paginated log
* ``POST /x-htsglang/workbench/pause``    the whole bench, or one tenant
* ``POST /x-htsglang/workbench/enqueue``  one work item into one tenant

A disabled workbench answers with a named 503 rather than a 404 (#305-M1
style): "this route does not exist" and "this route exists and is switched
off" are different problems with different fixes, and only one of them is the
operator's.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from sglang.srt.workbench.service import WorkbenchError, WorkbenchService

#: Every route this module answers, so the server can mount them in one place.
NAMESPACE = "/x-htsglang/workbench"


def error_payload(exc: WorkbenchError) -> tuple[int, dict[str, Any]]:
    return exc.status_code, {
        "error": {
            "code": exc.code,
            "message": str(exc),
            "type": "invalid_request_error",
        }
    }


def snapshot_payload(service: WorkbenchService) -> dict[str, Any]:
    """The whole state of the bench. Answers even when disabled.

    Reporting the queue of a switched-off workbench is the point: an operator
    asking "why is nothing being tuned" gets ``enabled: false`` rather than
    silence.
    """
    return {"ok": True, "workbench": service.snapshot()}


def events_payload(
    service: WorkbenchService, query: Optional[Mapping[str, Any]] = None
) -> dict[str, Any]:
    query = query or {}
    after = _int(query.get("after"), 0, "after")
    limit = max(1, min(1000, _int(query.get("limit"), 200, "limit")))
    return {"ok": True, **service.events(after=after, limit=limit)}


def pause_payload(
    service: WorkbenchService, body: Optional[Mapping[str, Any]] = None
) -> dict[str, Any]:
    body = body or {}
    paused = body.get("paused", True)
    if not isinstance(paused, bool):
        raise WorkbenchError("'paused' must be a boolean")
    tenant = body.get("tenant")
    if tenant is not None and not isinstance(tenant, str):
        raise WorkbenchError("'tenant' must be a tenant name or absent")
    return {"ok": True, **service.pause(paused=paused, tenant=tenant)}


def enqueue_payload(
    service: WorkbenchService, body: Optional[Mapping[str, Any]] = None
) -> dict[str, Any]:
    body = body or {}
    tenant = body.get("tenant")
    if not isinstance(tenant, str) or not tenant:
        raise WorkbenchError(
            "'tenant' is required and names the queue the item goes into"
        )
    item = body.get("item")
    if item is None:
        # Everything except 'tenant' is treated as the item, so a client can
        # post the flat shape as well as the nested one.
        item = {k: v for k, v in body.items() if k != "tenant"}
    if not isinstance(item, Mapping):
        raise WorkbenchError("'item' must be an object")
    return {"ok": True, **service.enqueue(tenant=tenant, item=item)}


def _int(value: Any, default: int, name: str) -> int:
    if value in (None, ""):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        raise WorkbenchError(f"'{name}' must be an integer, got {value!r}") from None


__all__ = [
    "NAMESPACE",
    "enqueue_payload",
    "error_payload",
    "events_payload",
    "pause_payload",
    "snapshot_payload",
]
