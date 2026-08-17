# SPDX-License-Identifier: Apache-2.0
"""AUTOMATIC1111 ``/sdapi/v1`` surface, composed over the OpenAI images front.

#335. The pattern is the KoboldCpp adapter's (ec44aa37ca) and the rule it
established is the whole design: **translate, never serve.** This module builds
an OpenAI images request and hands it to ``OpenAIServingImages``; it does not
reach the diffusion lane, does not forward HTTP itself, and does not invent a
sampler. The Ollama front's parallel-path mistake -- documented in
``ANALYSE_335_compat_surfaces.md`` §2 as the reason ``format`` could not simply
be wired -- is the thing this shape exists to avoid, and a source pin in the
tests forbids the vocabulary that would signal a relapse.

WHAT COMPOSING BUYS HERE, concretely: the lane-absent refusal is already
written and already carries the registry's own numbers (``_reject_no_lane``,
``serving_images.py:62``), and client-disconnect cancellation (#344) is already
in ``_guard``. A parallel path would have had to re-derive both, and would have
got the second one wrong.

REFUSE, NEVER APPROXIMATE
=========================
A1111's protocol carries diffusion controls the OpenAI images protocol does
not: ``steps``, ``cfg_scale``, ``sampler_name``, ``seed``, ``denoising_
strength``. There is no mapping -- not a lossy one, none: OpenAI's images
request has no field that means "how many denoising steps" or "which sampler".

Passing them through would drop them at the boundary and render something the
caller did not ask for, with nothing saying so. That is exactly the #710
tool-arg-loss family and exactly what this task's own Ollama fix
(9b5a72f826) was for. So a non-default value is REFUSED BY NAME, and the
refusal names the working path. ``rep_pen`` in the Kobold adapter is the same
judgement on the text side.

Fields that are inert rather than wrong -- ``save_images``, ``script_name`` --
are DECLARED inert, following the Ollama ``keep_alive`` precedent: the caller's
intent is either already satisfied or has no meaning here, and saying so is
better than looking honoured by accident.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from fastapi import Request
from fastapi.responses import ORJSONResponse

from sglang.srt.entrypoints.sdapi.protocol import Img2ImgRequest, Txt2ImgRequest

logger = logging.getLogger(__name__)

#: A1111 renders at 512x512 by default and stock clients rely on it.
DEFAULT_SIZE = "512x512"

#: Sizes the OpenAI images protocol expresses. A1111 accepts arbitrary
#: multiples of 8; this surface refuses what it cannot express rather than
#: rounding a caller's canvas to a neighbour without telling them.
SUPPORTED_SIZES: Tuple[str, ...] = (
    "256x256",
    "512x512",
    "1024x1024",
    "1792x1024",
    "1024x1792",
)

#: (field, default) pairs that change the image and have no OpenAI equivalent.
_NO_EQUIVALENT: Tuple[Tuple[str, Any], ...] = (
    ("steps", 50),
    ("cfg_scale", 7.0),
    ("sampler_name", None),
    ("sampler_index", None),
    ("seed", -1),
    ("subseed", -1),
    ("denoising_strength", None),
    ("restore_faces", False),
    ("tiling", False),
)

#: Accepted and knowingly without effect here, with the reason a caller needs.
INERT_FIELDS: Dict[str, str] = {
    "save_images": (
        "this server does not write images to a disk gallery; the images are "
        "returned in the response and nothing is persisted"
    ),
    "send_images": (
        "images are always returned in the response body; there is no other "
        "channel to send them on"
    ),
    "script_name": "there is no script engine behind this surface",
    "script_args": "there is no script engine behind this surface",
    "alwayson_scripts": "there is no script engine behind this surface",
    "styles": (
        "style presets are a webui-side prompt expansion; this surface has no "
        "style library, so compose the prompt yourself"
    ),
    "override_settings": (
        "per-request settings overrides are not honoured; configure the "
        "diffusion lane instead"
    ),
}


class SdapiServing:
    """Translate ``/sdapi/v1`` onto the OpenAI images front. Nothing else."""

    def __init__(self, images_serving, model_name: str = "sglang"):
        # Injected: this adapter must be testable without a lane, a registry
        # or an event loop that reaches the network.
        self._images = images_serving
        self._model_name = model_name

    # -- refusal ---------------------------------------------------------

    def unsupported_reasons(self, request: Txt2ImgRequest) -> List[str]:
        """Every field that would change the image and cannot be expressed."""
        reasons: List[str] = []
        for field, default in _NO_EQUIVALENT:
            value = getattr(request, field, default)
            if value != default:
                reasons.append(
                    f"{field}={value!r} has no equivalent in the OpenAI images "
                    f"protocol this surface composes over, so honouring it is "
                    f"not possible and dropping it would render an image you "
                    f"did not ask for"
                )
        size = f"{int(request.width)}x{int(request.height)}"
        if size not in SUPPORTED_SIZES:
            reasons.append(
                f"width x height {size} is not one of the sizes the OpenAI "
                f"images protocol expresses ({', '.join(SUPPORTED_SIZES)}); "
                f"rounding your canvas to a neighbour without saying so is the "
                f"failure this refusal exists to prevent"
            )
        return reasons

    def refusal_response(self, reasons: List[str]) -> ORJSONResponse:
        """A1111's error shape: ``{"error", "detail", "errors"}``.

        Protocol-conformant on purpose -- a stock client parses this. 400,
        because the request is well-formed but asks for something this surface
        cannot honestly do; that is a client-fixable condition, not a fault.
        """
        return ORJSONResponse(
            status_code=400,
            content={
                "error": "UnsupportedParameters",
                "detail": (
                    "This is an A1111-compatible surface over an OpenAI images "
                    "backend, not a Stable Diffusion webui. The parameters "
                    "below cannot be honoured and will not be silently "
                    "ignored. Remove them, or drive the diffusion lane "
                    "directly via POST /v1/images/generations."
                ),
                "errors": reasons,
            },
        )

    # -- translation -----------------------------------------------------

    def to_images_body(self, request: Txt2ImgRequest) -> Dict[str, Any]:
        """The OpenAI images request this A1111 request means.

        ``negative_prompt`` is NOT folded into the prompt. Concatenating it
        would produce a prompt the caller never wrote, and a diffusion lane
        that does support negatives would then receive them as positives --
        wrong in the one direction that is hard to see in the output.
        """
        n = max(1, int(request.batch_size)) * max(1, int(request.n_iter))
        return {
            "model": self._model_name,
            "prompt": request.prompt,
            "n": n,
            "size": f"{int(request.width)}x{int(request.height)}",
            # A1111 returns base64 always; asking the front for anything else
            # would mean re-fetching a URL to satisfy our own response shape.
            "response_format": "b64_json",
        }

    async def handle_txt2img(
        self, request: Txt2ImgRequest, raw_request: Request
    ) -> ORJSONResponse:
        if request.negative_prompt:
            return self.refusal_response(
                [
                    "negative_prompt is set, and the OpenAI images protocol "
                    "has no negative-prompt field. Folding it into the prompt "
                    "would send it as a POSITIVE instruction, which is wrong "
                    "in the direction hardest to notice in the output"
                ]
            )
        reasons = self.unsupported_reasons(request)
        if reasons:
            return self.refusal_response(reasons)

        body = self.to_images_body(request)
        response = await self._images.generations(body, raw_request)
        return self.to_a1111_response(response, body)

    def to_a1111_response(self, response, body: Dict[str, Any]) -> ORJSONResponse:
        """Wrap the OpenAI images reply in A1111's ``TextToImageResponse``.

        A refusal from the front (no lane configured, lane unreachable) is
        passed through UNCHANGED rather than re-dressed: it already names the
        registry state that caused it, and re-wrapping it in an A1111 envelope
        would hide which layer refused.
        """
        payload = getattr(response, "body", None)
        status = int(getattr(response, "status_code", 200) or 200)
        if status != 200:
            return response
        data = self._decode(payload)
        if data is None:
            return response
        images = [
            item.get("b64_json")
            for item in (data.get("data") or [])
            if isinstance(item, dict) and item.get("b64_json")
        ]
        return ORJSONResponse(
            status_code=200,
            content={
                "images": images,
                "parameters": body,
                # A JSON-encoded STRING, per the protocol: stock clients call
                # json.loads on it.
                "info": json.dumps(
                    {
                        "prompt": body.get("prompt", ""),
                        "size": body.get("size"),
                        "n": body.get("n"),
                        "backend": "sglang-openai-images",
                    }
                ),
            },
        )

    @staticmethod
    def _decode(payload) -> Optional[Dict[str, Any]]:
        if payload is None:
            return None
        try:
            if isinstance(payload, (bytes, bytearray)):
                return json.loads(payload.decode("utf-8"))
            if isinstance(payload, str):
                return json.loads(payload)
            if isinstance(payload, dict):
                return payload
        except (ValueError, UnicodeDecodeError):
            return None
        return None

    # -- endpoints refused outright --------------------------------------

    def refuse_img2img(self) -> ORJSONResponse:
        """img2img is not edits, and pretending otherwise renders nonsense.

        A1111's img2img re-noises an input image to ``denoising_strength`` and
        denoises it back. OpenAI's ``/v1/images/edits`` is MASK INPAINTING: it
        replaces a masked region and leaves the rest. Same inputs, different
        operation -- a client asking for a 0.3-strength restyle would get its
        image back with a hole filled in.
        """
        return ORJSONResponse(
            status_code=501,
            content={
                "error": "NotImplemented",
                "detail": (
                    "img2img is not implemented. It re-noises the whole image "
                    "to denoising_strength; the OpenAI images backend behind "
                    "this surface offers MASK INPAINTING (/v1/images/edits), "
                    "which replaces a masked region and leaves the rest. They "
                    "are different operations and mapping one onto the other "
                    "would return an image nobody asked for. Use POST "
                    "/v1/images/edits directly if inpainting is what you want."
                ),
                "errors": [],
            },
        )

    def progress(self) -> ORJSONResponse:
        """Honest zeros, because there is nothing to report.

        The images front writes ONCE, at the end -- ``serving_images.py``'s own
        ``_guard`` says so: "an image generation writes once, at the end, so
        there are no frames for a progress-based watchdog to count". So there
        is no intermediate progress to expose. This returns the protocol's
        shape with zeros and states why in ``textinfo``, rather than 404ing an
        endpoint stock clients poll on a timer, or inventing a percentage.
        """
        return ORJSONResponse(
            status_code=200,
            content={
                "progress": 0.0,
                "eta_relative": 0.0,
                "state": {
                    "skipped": False,
                    "interrupted": False,
                    "job": "",
                    "job_count": 0,
                    "job_timestamp": "",
                    "job_no": 0,
                    "sampling_step": 0,
                    "sampling_steps": 0,
                },
                "current_image": None,
                "textinfo": (
                    "progress is not observable on this backend: the image "
                    "generation writes once at the end, so there are no "
                    "intermediate steps to report. These zeros are honest, not "
                    "a stalled job."
                ),
            },
        )

    # -- metadata --------------------------------------------------------

    def options(self) -> ORJSONResponse:
        return ORJSONResponse(
            status_code=200,
            content={
                "sd_model_checkpoint": self._model_name,
                "samples_format": "png",
                "sd_checkpoint_hash": "",
            },
        )

    def sd_models(self) -> ORJSONResponse:
        return ORJSONResponse(
            status_code=200,
            content=[
                {
                    "title": self._model_name,
                    "model_name": self._model_name,
                    "hash": None,
                    "sha256": None,
                    "filename": "",
                    "config": None,
                }
            ],
        )

    def inert_fields(self) -> Dict[str, str]:
        """Declared, not silently accepted. Mirrors Ollama's ``keep_alive``."""
        return dict(INERT_FIELDS)
