# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""``/v1/audio/speech`` as a thin adapter onto a speech lane (#335-M0).

There is no text-to-speech lane in this tree today. Class 1 covers
autoregressive text, class 2 diffusion, class 3 pooling models and opaque
executors; a streaming speech model is class 1 with an audio detokenizer and
is scoped to #333 M5, which has not started.

The endpoint exists anyway, and answers honestly. A client that probes the
surface -- which is what SDK-driven tools do -- gets a documented OpenAI error
naming the missing capability instead of a 404 from the router that is
indistinguishable from a typo in the base URL. The routing half is written
against the same resolver the image lane uses, so when a speech engine does
land the change is a resolver returning an address, not a new endpoint.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from fastapi.responses import ORJSONResponse, Response

from sglang.srt.entrypoints.openai.errors import LaneUnavailable
from sglang.srt.entrypoints.openai.registry_view import (
    RegistryView,
    fetch_registry_view,
)

logger = logging.getLogger(__name__)

#: Base URL of a server exposing ``POST /v1/audio/speech``. Unset means no
#: lane, which is the current state of the tree.
SPEECH_LANE_URL_ENV = "SGLANG_SPEECH_URL"

_TIMEOUT_S = 300.0

CAPABILITY = "text_to_speech"

#: OpenAI's ``response_format`` set, with the content type each maps to.
_RESPONSE_FORMATS = {
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "audio/pcm",
}


def speech_lane_url() -> str:
    raw = (os.environ.get(SPEECH_LANE_URL_ENV) or "").strip().rstrip("/")
    if raw and not raw.startswith("http"):
        raw = "http://" + raw
    return raw


class OpenAIServingSpeech:
    """Forwards ``/v1/audio/speech`` to a speech lane, or rejects honestly."""

    def __init__(
        self, lane_url_resolver=speech_lane_url, view_resolver=fetch_registry_view
    ):
        self._lane_url_resolver = lane_url_resolver
        self._view_resolver = view_resolver

    def _validate(self, body: dict) -> Optional[LaneUnavailable]:
        response_format = body.get("response_format") or "mp3"
        if response_format not in _RESPONSE_FORMATS:
            return LaneUnavailable(
                f"Unsupported response_format {response_format!r}; expected one of "
                f"{sorted(_RESPONSE_FORMATS)}",
                capability=CAPABILITY,
                status_code=400,
                code="unsupported_value",
                param="response_format",
            )
        return None

    def _reject_no_lane(
        self, view: RegistryView, model: Optional[str]
    ) -> LaneUnavailable:
        detail: dict[str, Any] = {
            "lane_url_env": SPEECH_LANE_URL_ENV,
            "registered_engines": [e.engine_id for e in view.engines],
        }
        detail.update(view.to_extension())
        message = (
            "This server does not serve text-to-speech: no speech engine is "
            "registered and no speech lane is configured. Speech synthesis is "
            "scoped to #333 M5 and is not implemented in this build."
        )
        if model:
            message += f" Requested model: {model!r}."
        return LaneUnavailable(
            message,
            capability=CAPABILITY,
            status_code=404,
            code="model_not_found",
            param="model",
            remedies=(
                f"set {SPEECH_LANE_URL_ENV} to a server exposing POST /v1/audio/speech",
                "use /v1/audio/transcriptions for the speech-to-text direction, "
                "which this server does serve when an ASR model is loaded",
            ),
            detail=detail,
        )

    async def create_speech(self, body: dict, raw_request=None) -> Response:
        invalid = self._validate(body)
        if invalid is not None:
            raise invalid

        lane = self._lane_url_resolver()
        if not lane:
            raise self._reject_no_lane(self._view_resolver(), body.get("model"))

        # #344: same shape as the image lane and the same failure -- one long
        # await with no intermediate writes, and a GPU job on the far side of
        # it that keeps running for a client that already left.
        from sglang.srt.liveness import (  # noqa: PLC0415
            ConsumerGone,
            EndpointClass,
            await_with_liveness,
        )

        coro = self._forward(lane + "/v1/audio/speech", body)
        try:
            return await await_with_liveness(
                coro,
                raw_request=raw_request,
                endpoint_class=EndpointClass.AUDIO_SPEECH,
                job_id=f"speech-{id(coro):x}",
            )
        except ConsumerGone as exc:
            return ORJSONResponse(
                {
                    "error": {
                        "message": str(exc),
                        "type": "client_closed_request",
                        "param": None,
                        "code": 499,
                    }
                },
                status_code=499,
            )

    async def _forward(self, url: str, body: dict) -> Response:
        import aiohttp  # noqa: PLC0415

        media_type = _RESPONSE_FORMATS[body.get("response_format") or "mp3"]
        timeout = aiohttp.ClientTimeout(total=_TIMEOUT_S)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=body) as response:
                    payload = await response.read()
                    if response.status >= 400:
                        # An error from the lane is JSON; relay it as JSON so
                        # the client's error parser still works.
                        import json  # noqa: PLC0415

                        try:
                            return ORJSONResponse(
                                json.loads(payload), status_code=response.status
                            )
                        except ValueError:
                            return ORJSONResponse(
                                {
                                    "error": {
                                        "message": payload.decode(
                                            "utf-8", errors="replace"
                                        ),
                                        "type": "api_error",
                                        "param": None,
                                        "code": response.status,
                                    }
                                },
                                status_code=response.status,
                            )
                    return Response(content=payload, media_type=media_type)
        except LaneUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            logger.warning("speech lane %s unreachable: %s", url, exc)
            raise LaneUnavailable(
                f"The configured speech lane at {url} did not answer: {exc}",
                capability=CAPABILITY,
                status_code=503,
                code="lane_unreachable",
                param=None,
                remedies=(f"check that the speech server at {url} is running",),
                detail={"lane_url": url},
            ) from exc
