# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""KoboldCpp-compatible surface (#335), COMPOSED over the OpenAI front.

THIS IS THE SHAPE THE OLLAMA FRONT SHOULD HAVE USED. That adapter drives the
tokenizer manager directly and applies its own chat template, so it is a
second serving path: structured output, reasoning control and every future
sampling feature have to be wired into it separately, and `format` was never
wired at all. This module instead TRANSLATES a Kobold request into a
``CompletionRequest`` and hands it to ``openai_serving_completion``. There is
no sampling here, no template here, and no second place for a feature to be
forgotten.

REFUSAL DISCIPLINE, identical to the Ollama cut (9b5a72f826):

  * a field this adapter cannot honour is REFUSED BY NAME before generation,
    and the refusal ROUTES -- it names the endpoint that does support it;
  * a field whose intent is already satisfied is declared INERT, so it does
    not look honoured by accident;
  * nothing is silently dropped, because a value that vanishes between the
    request and the sampler is the #710 class.

WHY ``rep_pen`` IS REFUSED RATHER THAN MAPPED, as the worked example: Kobold's
``rep_pen`` is a MULTIPLICATIVE repetition penalty over previously seen
tokens; OpenAI's ``frequency_penalty`` is an ADDITIVE logit adjustment scaled
by count. They are not the same function and no constant converts one into the
other. Mapping them would produce output that is plausibly wrong -- the worst
outcome available, because nothing in the response says the knob did something
other than what was asked.
"""

from __future__ import annotations

import logging

from typing import List, Optional

from fastapi import Request
from fastapi.responses import ORJSONResponse

logger = logging.getLogger(__name__)

#: Advertised to Kobold clients that probe for a version before talking.
#: Deliberately NOT a KoboldCpp version number: claiming one would assert
#: compatibility with a build whose behaviour we do not implement.
KOBOLD_VERSION_PAYLOAD = {"result": "sglang", "version": "1.0"}


class KoboldServing:
    """Thin translation onto ``openai_serving_completion``."""

    #: Refused by name, each with the route that DOES support it. A refusal
    #: that only blocks teaches nothing.
    UNSUPPORTED_FIELDS = {
        "rep_pen": (
            "Kobold's rep_pen is a MULTIPLICATIVE repetition penalty; the "
            "OpenAI path this adapter composes offers frequency_penalty, an "
            "ADDITIVE logit adjustment. They are different functions and no "
            "constant converts one to the other, so mapping it would sample "
            "differently than you asked without saying so. Use "
            "/v1/completions with frequency_penalty / presence_penalty."
        ),
        "rep_pen_range": ("see rep_pen: no equivalent on /v1/completions."),
        "rep_pen_slope": ("see rep_pen: no equivalent on /v1/completions."),
        "top_a": "top-a sampling is not implemented on this server.",
        "typical": "typical-p sampling is not implemented on this server.",
        "tfs": "tail-free sampling is not implemented on this server.",
        "sampler_order": (
            "Kobold lets the client reorder the sampler chain; this server's "
            "sampler order is fixed, so it is not implemented here and "
            "ignoring the field would change the distribution silently. "
            "Use /v1/completions and set the samplers you need directly."
        ),
        "mirostat": "mirostat is not implemented on this server.",
        "mirostat_tau": "mirostat is not implemented on this server.",
        "mirostat_eta": "mirostat is not implemented on this server.",
        "dynatemp_range": "dynamic temperature is not implemented on this server.",
        "smoothing_factor": "smoothing sampling is not implemented on this server.",
        "grammar": (
            "GBNF grammars are not supported here. Use /v1/chat/completions "
            "with response_format={'type': 'json_schema', ...} for structured "
            "output."
        ),
        "banned_tokens": (
            "token banning is not exposed on this surface; use "
            "/v1/completions with logit_bias."
        ),
        "logit_bias": (
            "Kobold's logit_bias is keyed differently from the OpenAI one; "
            "passing it through unchecked could bias the wrong tokens. Use "
            "/v1/completions with logit_bias instead."
        ),
        "memory": (
            "prompt synthesis is not implemented here: Kobold's memory field "
            "prepends persistent context, and this adapter deliberately never "
            "builds a prompt the caller did not write. Prepend it to `prompt` "
            "yourself, or use /v1/chat/completions with a system message."
        ),
        "images": (
            "multimodal input is not accepted on the Kobold surface. Use "
            "/v1/chat/completions with image content parts."
        ),
        "sampler_seed": (
            "use `seed`-equivalent determinism via /v1/completions; this "
            "adapter does not silently reinterpret sampler_seed."
        ),
    }

    #: Accepted and deliberately inert, WITH the reason. Not silent drops:
    #: the caller's intent is already satisfied.
    INERT_FIELDS = {
        "max_context_length": (
            "the context window is fixed at server start (--context-length); "
            "a per-request value has nothing to change. Accepted and ignored."
        ),
        "quiet": (
            "this server never echoes the prompt back, so there is nothing to "
            "suppress. Accepted and ignored."
        ),
    }

    #: Kobold field -> OpenAI CompletionRequest field. The only mapping.
    FIELD_MAP = {
        "max_length": "max_tokens",
        "temperature": "temperature",
        "top_p": "top_p",
        "top_k": "top_k",
        "stop_sequence": "stop",
        "n": "n",
    }

    #: Used only when the client omits ``max_length``. See
    #: ``to_completion_request`` for why this adapter declares one at all.
    DEFAULT_MAX_TOKENS = 200

    def __init__(self, model_name: str = "sglang"):
        self.model_name = model_name

    # -- refusal surface ---------------------------------------------------

    def unsupported_reasons(self, request) -> List[str]:
        """Named refusals for anything this adapter would otherwise drop.

        Pure function of the request, so it pins without a server.
        """
        reasons = []
        for field_name, why in self.UNSUPPORTED_FIELDS.items():
            if getattr(request, field_name, None) is not None:
                reasons.append(f"{field_name!r}: {why}")
        return reasons

    def refusal_response(self, reasons: List[str]) -> ORJSONResponse:
        return ORJSONResponse(
            status_code=400,
            content={
                "error": (
                    "this KoboldCpp-compatible surface cannot honour part of "
                    "the request, and will not answer as though it had: "
                    + " | ".join(reasons)
                )
            },
        )

    # -- translation -------------------------------------------------------

    def to_completion_request(self, request):
        """Build the OpenAI ``CompletionRequest`` this maps onto.

        Everything the model will see comes from here, and there is nothing
        else: no template application, no prompt synthesis, no sampling
        defaults of this adapter's own.
        """
        from sglang.srt.entrypoints.openai.protocol import CompletionRequest

        # DECLARED DEFAULT, not a silent one. CompletionRequest defaults
        # max_tokens to 16 -- an OpenAI completions convention that would
        # truncate a Kobold client's first answer to a fragment, since Kobold
        # clients expect the server's own generous default when they omit
        # max_length. Choosing a number here is an adapter responsibility;
        # hiding that it was chosen is not, so it is named and pinned.
        body = {
            "model": self.model_name,
            "prompt": request.prompt,
            "max_tokens": self.DEFAULT_MAX_TOKENS,
        }
        for kobold_field, openai_field in self.FIELD_MAP.items():
            value = getattr(request, kobold_field, None)
            if value is not None:
                body[openai_field] = value
        return CompletionRequest(**body)

    async def handle_generate(self, request, raw_request: Request):
        """``POST /api/v1/generate``, composed onto the OpenAI completion path."""
        reasons = self.unsupported_reasons(request)
        if reasons:
            return self.refusal_response(reasons)

        completion = self.to_completion_request(request)
        completion.stream = False
        serving = raw_request.app.state.openai_serving_completion
        result = await serving.handle_request(completion, raw_request)

        text = self._first_text(result)
        if text is None:
            # Pass an upstream error through untouched rather than dressing
            # it as an empty completion -- a Kobold client reading
            # results[0].text would otherwise see success.
            return result
        return ORJSONResponse(content={"results": [{"text": text}]})

    @staticmethod
    def _first_text(result) -> Optional[str]:
        choices = getattr(result, "choices", None)
        if not choices:
            return None
        return getattr(choices[0], "text", None)

    # -- endpoints that are answered without generating ---------------------

    def get_model(self) -> dict:
        return {"result": self.model_name}

    def get_version(self) -> dict:
        return dict(KOBOLD_VERSION_PAYLOAD)

    def refuse_abort(self) -> ORJSONResponse:
        """``/api/extra/abort`` cannot be honoured, and guessing is worse.

        Kobold aborts THE current generation, which presumes one. This server
        is multi-tenant: there is no "the" generation, and picking one would
        cancel a stranger's request. The OpenAI path cancels by disconnecting
        the request whose completion you want to stop.
        """
        return ORJSONResponse(
            status_code=501,
            content={
                "error": (
                    "abort is not supported on this surface: KoboldCpp aborts "
                    "THE current generation, which presumes a single-user "
                    "server. This one is multi-tenant, so there is no 'the' "
                    "generation and cancelling a guessed one would stop "
                    "another client's request. Cancel by closing your own "
                    "connection."
                )
            },
        )

    def refuse_stream(self) -> ORJSONResponse:
        """``/api/extra/generate/stream``: refused, with the working route.

        KoboldCpp's streaming is a POLLED protocol -- the client posts, then
        repeatedly GETs ``/api/extra/generate/check`` for the partial result --
        and its SSE variant emits Kobold-shaped token events. Emitting an
        OpenAI-shaped stream under a Kobold URL would be a HALF-FAITHFUL
        stream: it would look like it worked until a real client's parser
        disagreed, mid-generation, with no error to read.

        Refusing by name and naming the route that streams honestly is the
        smaller harm, and it is reversible: a faithful implementation can
        replace this without any client having been taught a wrong shape.
        """
        return ORJSONResponse(
            status_code=501,
            content={
                "error": (
                    "streaming is not implemented on the Kobold surface. "
                    "KoboldCpp streams via a polled /api/extra/generate/check "
                    "protocol whose shape this adapter does not reproduce, and "
                    "emitting an OpenAI-shaped stream here would be "
                    "half-faithful: it would parse until it did not. Use "
                    "/api/v1/generate for a complete answer, or "
                    "/v1/completions with stream=true for token streaming."
                )
            },
        )
