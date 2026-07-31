# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""One error shape for the whole OpenAI-compatible surface (#335-M0).

OpenAI's error body is an envelope::

    {"error": {"message": ..., "type": ..., "param": ..., "code": ...}}

Every client in the ecosystem parses that envelope: the official SDKs read
``body["error"]`` before looking at ``message``/``type``/``param``/``code``,
and everything built on top of them (LangChain, LiteLLM, the various shell
wrappers) does the same. A flat body with the four keys at the top level is a
different protocol that happens to carry the same words, so this module owns
the envelope and every handler goes through it.

Rejections that are not the caller's fault carry a namespaced
``x-htsglang`` block inside the error object. It is additive: a vanilla client
sees a perfectly ordinary OpenAI error and ignores the extra key, while a
client that knows the fork gets the numbers -- which engine, which residency
state, what would have to change. A caller told "unavailable" can only retry;
a caller told "no engine of class 2 is registered; register one and promote it
to HOT" can act.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Union

from fastapi.responses import ORJSONResponse

#: Namespaced extension key. Everything the fork adds to an otherwise
#: spec-shaped body lives under this one key, at exactly one nesting level.
EXTENSION_KEY = "x-htsglang"


def openai_error_dict(
    message: str,
    *,
    err_type: str = "invalid_request_error",
    param: Optional[str] = None,
    code: Optional[Union[str, int]] = None,
    extension: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """The envelope, as a plain dict."""
    error: dict[str, Any] = {
        "message": message,
        "type": err_type,
        "param": param,
        "code": code,
    }
    if extension:
        error[EXTENSION_KEY] = dict(extension)
    return {"error": error}


def openai_error_response(
    message: str,
    *,
    status_code: int = 400,
    err_type: str = "invalid_request_error",
    param: Optional[str] = None,
    code: Optional[Union[str, int]] = None,
    extension: Optional[Mapping[str, Any]] = None,
) -> ORJSONResponse:
    """The envelope, as an HTTP response with a faithful status code."""
    return ORJSONResponse(
        content=openai_error_dict(
            message,
            err_type=err_type,
            param=param,
            code=code,
            extension=extension,
        ),
        status_code=status_code,
    )


def parse_error_body(body: Any) -> dict[str, Any]:
    """Read an error body written by either shape.

    Server to server inside this tree, both ends speak the envelope. But an
    sglang process may talk to an older one during a rolling restart, and the
    reader has no way to know which. Unwrapping ``error`` when it is an object
    and falling back to the body itself costs one line and removes the whole
    class of "the error text was empty" reports.
    """
    if not isinstance(body, dict):
        return {}
    error = body.get("error")
    if isinstance(error, dict):
        return error
    if isinstance(error, str):
        return {"message": error}
    return body


def error_message_of(body: Any, default: str = "Unknown error") -> str:
    return str(parse_error_body(body).get("message") or default)


def error_type_for_status(status_code: int) -> str:
    """OpenAI's ``error.type`` vocabulary, keyed by HTTP status.

    The upstream values are a small closed set; anything outside it is
    ``api_error``, which is what OpenAI itself returns for unclassified
    failures.
    """
    return {
        400: "invalid_request_error",
        401: "authentication_error",
        403: "permission_error",
        404: "not_found_error",
        409: "conflict_error",
        413: "invalid_request_error",
        415: "invalid_request_error",
        422: "invalid_request_error",
        429: "rate_limit_error",
        500: "api_error",
        501: "api_error",
        502: "api_error",
        503: "api_error",
        504: "api_error",
    }.get(status_code, "api_error")


class LaneUnavailable(Exception):
    """A capability the surface exposes has no lane that can serve it now.

    Raised by the image and speech adapters. Carries everything the caller
    needs to decide what to do next, which is the registry's rejection
    pattern (#333 §7.5) applied to an OpenAI-shaped body: the requested
    capability, what is registered, and the concrete steps that would make
    the same request succeed.
    """

    def __init__(
        self,
        message: str,
        *,
        capability: str,
        status_code: int = 404,
        code: str = "model_not_found",
        param: Optional[str] = "model",
        remedies: tuple[str, ...] = (),
        detail: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.capability = capability
        self.status_code = status_code
        self.code = code
        self.param = param
        self.remedies = tuple(remedies)
        self.detail = dict(detail or {})

    def to_response(self) -> ORJSONResponse:
        extension: dict[str, Any] = {
            "capability": self.capability,
            "reason": self.code,
            "what_would_make_it_work": list(self.remedies),
        }
        extension.update(self.detail)
        return openai_error_response(
            str(self),
            status_code=self.status_code,
            err_type=error_type_for_status(self.status_code),
            param=self.param,
            code=self.code,
            extension=extension,
        )
