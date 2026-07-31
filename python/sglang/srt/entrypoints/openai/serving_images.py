# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""``/v1/images/*`` as a thin adapter onto the diffusion lane (#335-M0).

The fork already generates images: ``sglang.multimodal_gen`` implements
``POST /v1/images/generations`` and ``/v1/images/edits`` in its own process,
with its own scheduler and its own port (#333 class 2). This module does not
reimplement any of that. It gives the serving surface the same route names, so
a client configured against this server does not have to know that image
generation lives one process over, and it forwards the body unchanged.

Two things it therefore must not do. It must not translate the protocol a
second time -- ``multimodal_gen``'s request and response models are already the
OpenAI ones, so re-parsing them here would only add a place for the two copies
to drift. And it must not pretend. When no diffusion lane is reachable the
answer is a structured rejection naming what is missing and what would make
the same request work, not an empty image list and not a generic 500.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from fastapi import Request
from fastapi.responses import ORJSONResponse

from sglang.srt.entrypoints.openai.errors import LaneUnavailable
from sglang.srt.entrypoints.openai.registry_view import (
    CLASS_DIFFUSION,
    RegistryView,
    fetch_registry_view,
)

logger = logging.getLogger(__name__)

#: Base URL of a running ``multimodal_gen`` server. No default port is
#: guessed: the diffusion service is optional and started independently, so an
#: unset value is an honest "not configured" rather than a connection refused
#: on a port nobody promised.
IMAGE_LANE_URL_ENV = "SGLANG_IMAGE_GEN_URL"

#: Image generation is seconds-to-minutes work, not milliseconds. The timeout
#: bounds a hung lane, not a slow one.
_TIMEOUT_S = 900.0

CAPABILITY = "image_generation"


def image_lane_url() -> str:
    raw = (os.environ.get(IMAGE_LANE_URL_ENV) or "").strip().rstrip("/")
    if raw and not raw.startswith("http"):
        raw = "http://" + raw
    return raw


def _diffusion_engines(view: RegistryView) -> tuple[str, ...]:
    return tuple(e.engine_id for e in view.of_class(CLASS_DIFFUSION))


def _reject_no_lane(view: RegistryView, model: Optional[str]) -> LaneUnavailable:
    """The rejection, with the registry's own numbers in it."""
    registered = _diffusion_engines(view)
    hot = tuple(
        e.engine_id for e in view.of_class(CLASS_DIFFUSION) if e.is_gpu_resident
    )
    detail: dict[str, Any] = {
        "registered_diffusion_engines": list(registered),
        "gpu_resident_diffusion_engines": list(hot),
        "lane_url_env": IMAGE_LANE_URL_ENV,
    }
    detail.update(view.to_extension())

    if registered:
        # Registered but unreachable: the operator has declared the engine, so
        # the missing piece is the address of its serving process, not the
        # engine itself.
        message = (
            f"No image-generation lane is reachable. The registry knows "
            f"{len(registered)} diffusion engine(s) ({', '.join(registered)}), but "
            f"this server has no address for a multimodal_gen process serving them."
        )
        remedies = (
            f"start a multimodal_gen server and set {IMAGE_LANE_URL_ENV} to its base URL",
            "promote a diffusion engine to HOT via the registry control plane",
        )
    else:
        message = (
            "This server does not serve image generation: no diffusion engine is "
            "registered and no multimodal_gen lane is configured."
        )
        remedies = (
            f"start a multimodal_gen server and set {IMAGE_LANE_URL_ENV} to its base URL",
            "register a class-2 (diffusion) engine with the registry control plane",
        )

    if model:
        message += f" Requested model: {model!r}."
    return LaneUnavailable(
        message,
        capability=CAPABILITY,
        status_code=404,
        code="model_not_found",
        param="model",
        remedies=remedies,
        detail=detail,
    )


class OpenAIServingImages:
    """Forwards ``/v1/images/*`` to the diffusion lane, or rejects honestly."""

    def __init__(
        self, lane_url_resolver=image_lane_url, view_resolver=fetch_registry_view
    ):
        # Injected rather than read at import time: the lane can be started
        # after this server, and tests need to move it.
        self._lane_url_resolver = lane_url_resolver
        self._view_resolver = view_resolver

    def _resolve_lane(self, model: Optional[str]) -> str:
        url = self._lane_url_resolver()
        if not url:
            raise _reject_no_lane(self._view_resolver(), model)
        return url

    async def generations(self, body: dict, raw_request: Request) -> ORJSONResponse:
        lane = self._resolve_lane(body.get("model"))
        return await self._forward_json(lane + "/v1/images/generations", body)

    async def edits(
        self, form: dict, files: dict, model: Optional[str]
    ) -> ORJSONResponse:
        lane = self._resolve_lane(model)
        return await self._forward_multipart(lane + "/v1/images/edits", form, files)

    async def variations(self, model: Optional[str]) -> ORJSONResponse:
        # Resolve the lane first: with no lane at all, "no image lane" is the
        # more useful answer than "this lane cannot do variations".
        self._resolve_lane(model)
        raise LaneUnavailable(
            "Image variations are not implemented. The diffusion lane serves "
            "/v1/images/generations and /v1/images/edits; it has no "
            "variations pipeline.",
            capability="image_variation",
            status_code=501,
            code="endpoint_not_implemented",
            param=None,
            remedies=(
                "use /v1/images/edits with an explicit prompt and the source image",
            ),
        )

    async def _forward_json(self, url: str, body: dict) -> ORJSONResponse:
        import aiohttp  # noqa: PLC0415

        timeout = aiohttp.ClientTimeout(total=_TIMEOUT_S)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=body) as response:
                    payload = await response.json(content_type=None)
                    # The lane's own status and body are relayed unchanged: it
                    # composed the error, and a second rewording here would be
                    # a drifting copy of the same sentence.
                    return ORJSONResponse(payload, status_code=response.status)
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            raise self._unreachable(url, exc) from exc

    async def _forward_multipart(
        self, url: str, form: dict, files: dict
    ) -> ORJSONResponse:
        import aiohttp  # noqa: PLC0415

        timeout = aiohttp.ClientTimeout(total=_TIMEOUT_S)
        data = aiohttp.FormData()
        for key, value in form.items():
            if value is not None:
                data.add_field(key, str(value))
        for key, (filename, content, content_type) in files.items():
            data.add_field(key, content, filename=filename, content_type=content_type)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, data=data) as response:
                    payload = await response.json(content_type=None)
                    return ORJSONResponse(payload, status_code=response.status)
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            raise self._unreachable(url, exc) from exc

    def _unreachable(self, url: str, exc: Exception) -> LaneUnavailable:
        logger.warning("image lane %s unreachable: %s", url, exc)
        return LaneUnavailable(
            f"The configured image-generation lane at {url} did not answer: {exc}",
            capability=CAPABILITY,
            # 503, not 404: the model exists as far as this server knows, the
            # process serving it is down. A caller can retry a 503.
            status_code=503,
            code="lane_unreachable",
            param=None,
            remedies=(
                f"check that the multimodal_gen server at {url} is running",
                f"correct {IMAGE_LANE_URL_ENV} if the lane moved",
            ),
            detail={"lane_url": url},
        )
