# SPDX-License-Identifier: Apache-2.0
"""Ollama-compatible surface, COMPOSED over the OpenAI front (#335).

This module used to be a parallel serving path: it applied the chat template
itself and drove ``tokenizer_manager`` directly, which is why
``ANALYSE_335_compat_surfaces.md`` §2 had to REFUSE Ollama's ``format`` field
rather than wire it -- the request never reached the machinery that implements
structured output. It now translates into an OpenAI request and hands it to the
OpenAI front, the shape the KoboldCpp (``ec44aa37ca``) and sdapi surfaces use.

WHAT THE CHANGE BUYS, concretely and not as tidiness:

* **``format`` works.** It is ``response_format``, and the OpenAI front owns
  that. A refusal became a feature; the field a caller sets is now honoured
  instead of explained.
* **The chat template stops being applied twice-over by two owners.** The
  OpenAI chat front applies it, so there is one place where a template bug can
  live rather than two that must agree.
* **Inherited correctness.** ``serving_base.handle_request`` already wraps
  every streaming reply in the #344 client-disconnect guard
  (``serving_base.py:119``). This adapter no longer builds its own.

WHY WRAPPING THE GUARDED STREAM IS STILL SAFE, since it is the one thing about
this refactor that could quietly regress: ``guard_generate_stream`` installs
its watchdog ON THE RESPONSE'S ``body_iterator``
(``liveness/stream.py:182`` -> ``guard_streaming_response``). This adapter
consumes THAT iterator and re-emits NDJSON, so the watchdog is upstream of the
translation, not bypassed by it: when the Ollama client stops reading, Starlette
stops pulling this generator, which stops pulling the guarded iterator, and the
watchdog releases the KV blocks exactly as before.

WHAT DID NOT CHANGE, and is pinned in ``test_ollama_golden_shapes_335.py``
(committed BEFORE this rewrite, deliberately): every response shape a client
sees, the NDJSON delta semantics, the empty-prompt short circuit, the named
refusals, and the 2048-token default.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import orjson
from fastapi import Request
from fastapi.responses import ORJSONResponse, StreamingResponse

from sglang.srt.entrypoints.ollama.protocol import (
    OllamaChatRequest,
    OllamaChatResponse,
    OllamaChatStreamResponse,
    OllamaGenerateRequest,
    OllamaGenerateResponse,
    OllamaGenerateStreamResponse,
    OllamaMessage,
    OllamaShowResponse,
    OllamaTagsResponse,
)

logger = logging.getLogger(__name__)

#: Ollama clients expect longer replies than the OpenAI default (16) would
#: give them. Kept from the parallel path deliberately: dropping it in this
#: rewrite would truncate every reply, which is the KoboldCpp
#: ``max_tokens``-defaults-to-16 lesson with a different number.
DEFAULT_NUM_PREDICT = 2048


class OllamaServing:
    """Translate Ollama's API onto the OpenAI front. Nothing else."""

    #: Fields declared by the protocol that this surface cannot honour. Each
    #: is REFUSED by name rather than dropped -- a value that vanishes between
    #: the request and the sampler is the #710 tool-arg-loss class.
    #:
    #: ``format`` is NO LONGER HERE: composing made it reachable, which is the
    #: change this module exists for.
    #: Nothing is unconditionally unsupported any more. ``think`` is refused
    #: CONDITIONALLY -- see :meth:`_think_refusal` -- because whether it can be
    #: honoured depends on the shape of the value and on which endpoint asked.
    UNSUPPORTED_FIELDS: Dict[str, str] = {}

    #: Declared, accepted, and knowingly without effect -- the honest third
    #: category between mapped and refused.
    INERT_FIELDS: Dict[str, str] = {
        "keep_alive": (
            "this server keeps the model resident for the process lifetime, "
            "so the intent is already satisfied; stated rather than left "
            "looking honoured by accident"
        ),
    }

    #: Ollama option -> OpenAI request field. Every option outside this map is
    #: refused, because mapping it approximately would sample differently than
    #: the caller asked with nothing saying so.
    OPTION_MAP: Dict[str, str] = {
        "temperature": "temperature",
        "top_p": "top_p",
        "top_k": "top_k",
        "num_predict": "max_tokens",
        "stop": "stop",
        "presence_penalty": "presence_penalty",
        "frequency_penalty": "frequency_penalty",
        "seed": "seed",
    }

    SUPPORTED_OPTIONS = frozenset(OPTION_MAP)

    def __init__(
        self,
        openai_serving_chat,
        openai_serving_completion,
        model_name: str,
        context_len: Optional[int] = None,
    ):
        # The OpenAI FRONTS, not the tokenizer manager. This adapter cannot
        # reach the engine even if a later edit tried to: there is nothing
        # here to reach it with.
        #
        # ``context_len`` is passed as a VALUE rather than read off a manager.
        # /api/show has always reported it and a stub would be a regression,
        # but reading it from the engine would put the coupling back that this
        # rewrite exists to remove -- metadata is not a reason to hold a
        # serving handle.
        self._chat = openai_serving_chat
        self._completion = openai_serving_completion
        self._model_name = model_name
        self._context_len = context_len

    # -- helpers ---------------------------------------------------------

    def _get_timestamp(self) -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    @property
    def model_name(self) -> str:
        return self._model_name

    # -- refusal ---------------------------------------------------------

    def unsupported_reasons(self, request) -> List[str]:
        """Named refusals for anything this adapter would otherwise drop.

        A pure function of the request, so it is pinnable without a server.
        """
        reasons: List[str] = []
        for field_name, why in self.UNSUPPORTED_FIELDS.items():
            if getattr(request, field_name, None) is not None:
                reasons.append(f"{field_name!r}: {why}")
        think_refusal = self._think_refusal(request)
        if think_refusal:
            reasons.append(think_refusal)
        options = getattr(request, "options", None) or {}
        unknown = sorted(k for k in options if k not in self.SUPPORTED_OPTIONS)
        if unknown:
            reasons.append(
                f"options {unknown}: not mapped by this adapter. Supported: "
                f"{sorted(self.SUPPORTED_OPTIONS)}. An unmapped option would "
                f"be dropped, so the model would sample differently than you "
                f"asked and nothing would say so."
            )
        return reasons

    def _think_refusal(self, request) -> Optional[str]:
        """``think`` is honourable only as a BOOLEAN, and only on /api/chat.

        #557 built per-request ``chat_template_kwargs`` and it reaches the
        chat front this surface composes (``ChatCompletionRequest`` carries the
        field; ``merge_chat_template_kwargs`` consumes it). So ``think: true``
        and ``think: false`` are wired -- through the front's OWN
        ``apply_reasoning_enabled``, which knows the model's reasoning family
        and refuses a model that has no reasoning parser at all. Re-deriving
        that capability check here would be a second authority for it.

        TWO CASES STAY REFUSED, each for a reason that is not squeamishness:

        * **an effort level** (``"low"``/``"medium"``/``"high"``) would have to
          become ``reasoning_effort``. The operative checkpoint's semantics are
          effort-BY-OMISSION, with explicit high/max observed to fail at the
          server; sending a value the backend rejects is precisely what an
          adapter must not do, and that hazard is live-model behaviour this
          module cannot verify. Refused with the route named, so a caller who
          wants it owns the choice on a path where the error is theirs to see.
        * **any ``think`` on /api/generate**, because that path composes
          ``CompletionRequest``, which carries no ``chat_template_kwargs`` at
          all -- there is no template being applied to toggle.
        """
        value = getattr(request, "think", None)
        if value is None:
            return None
        if not isinstance(value, bool):
            return (
                f"'think': {value!r} is an effort level, which would have to "
                f"become reasoning_effort. This surface will not send it: the "
                f"served checkpoint takes its effort by OMISSION and explicit "
                f"high/max has been observed to fail at the server, so an "
                f"adapter guessing here turns your request into an error you "
                f"did not cause. Use /v1/chat/completions with "
                f"reasoning_effort if you want to choose a level yourself, or "
                f"send think: true / think: false, which are honoured."
            )
        if not self._is_chat_request(request):
            return (
                "'think' is only honourable on /api/chat: /api/generate is a "
                "raw completion, which applies no chat template and therefore "
                "has no reasoning toggle to set. Use /api/chat, or "
                "/v1/chat/completions."
            )
        return None

    @staticmethod
    def _is_chat_request(request) -> bool:
        return hasattr(request, "messages")

    def _refusal(self, reasons: List[str]) -> ORJSONResponse:
        return ORJSONResponse(
            status_code=400,
            content={
                "error": (
                    "this Ollama-compatible surface cannot honour part of "
                    "the request, and will not answer as though it had: "
                    + " | ".join(reasons)
                )
            },
        )

    # -- translation into OpenAI ----------------------------------------

    def _response_format(self, fmt) -> Optional[Dict[str, Any]]:
        """Ollama ``format`` -> OpenAI ``response_format``. THE WIN.

        ``"json"`` is free-form JSON; a dict is a JSON SCHEMA, and Ollama
        passes it as the schema itself rather than wrapped, so it is wrapped
        here. A schema is given ``strict``: a caller who supplied a schema
        wants it obeyed, not approximated.
        """
        if fmt is None:
            return None
        if fmt == "json":
            return {"type": "json_object"}
        if isinstance(fmt, dict):
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": "ollama_format",
                    "schema": fmt,
                    "strict": True,
                },
            }
        return None

    def _openai_fields(self, request) -> Dict[str, Any]:
        """Mapped options + response_format, ready to splat into a request."""
        fields: Dict[str, Any] = {}
        options = getattr(request, "options", None) or {}
        for ollama_key, openai_key in self.OPTION_MAP.items():
            if ollama_key in options:
                fields[openai_key] = options[ollama_key]
        fields.setdefault("max_tokens", DEFAULT_NUM_PREDICT)
        fmt = self._response_format(getattr(request, "format", None))
        if fmt is not None:
            fields["response_format"] = fmt
        return fields

    # -- /api/chat -------------------------------------------------------

    async def handle_chat(
        self, request: OllamaChatRequest, raw_request: Request
    ) -> Union[OllamaChatResponse, StreamingResponse, ORJSONResponse]:
        reasons = self.unsupported_reasons(request)
        if reasons:
            return self._refusal(reasons)

        from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

        # The messages are already OpenAI-shaped; no template is applied here,
        # which is the whole point.
        chat_request = ChatCompletionRequest(
            model=self._model_name,
            messages=[{"role": m.role, "content": m.content} for m in request.messages],
            stream=bool(request.stream),
            **self._openai_fields(request),
        )

        if isinstance(getattr(request, "think", None), bool):
            # The front owns the capability question (which reasoning family,
            # always-on models, no parser at all). It RAISES rather than
            # silently leaving the model in the wrong mode, and that raise is
            # the per-model refusal this surface promises -- turned into
            # Ollama's own 400 envelope rather than allowed to surface as a
            # server error the client cannot read.
            try:
                self._chat.apply_reasoning_enabled(chat_request, request.think)
            except ValueError as exc:
                return self._refusal(
                    [
                        f"'think': {exc}. This model cannot serve the "
                        f"reasoning mode you asked for, and answering in the "
                        f"other one without saying so would be the silent "
                        f"overrule this surface exists to prevent."
                    ]
                )

        start = time.time_ns()
        response = await self._chat.handle_request(chat_request, raw_request)

        if request.stream:
            return self._ndjson(self._chat_stream(response), response)
        return self._chat_reply(response, start)

    def _chat_reply(self, response, start_ns: int):
        if self._is_error(response):
            return response
        text, prompt_tokens, completion_tokens = self._read_chat(response)
        return OllamaChatResponse(
            model=self._model_name,
            created_at=self._get_timestamp(),
            message=OllamaMessage(role="assistant", content=text),
            done=True,
            done_reason="stop",
            total_duration=time.time_ns() - start_ns,
            prompt_eval_count=prompt_tokens,
            eval_count=completion_tokens,
        )

    async def _chat_stream(self, response) -> AsyncIterator[bytes]:
        async for delta, done in self._sse_deltas(response, chat=True):
            payload = OllamaChatStreamResponse(
                model=self._model_name,
                created_at=self._get_timestamp(),
                message=OllamaMessage(role="assistant", content="" if done else delta),
                done=done,
                done_reason="stop" if done else None,
            )
            yield orjson.dumps(payload.model_dump()) + b"\n"

    # -- /api/generate ---------------------------------------------------

    async def handle_generate(
        self, request: OllamaGenerateRequest, raw_request: Request
    ) -> Union[OllamaGenerateResponse, StreamingResponse, ORJSONResponse]:
        reasons = self.unsupported_reasons(request)
        if reasons:
            return self._refusal(reasons)

        prompt = request.prompt
        if request.system:
            prompt = f"{request.system}\n\n{prompt}"

        # The Ollama CLI sends an empty request on startup. Answering it with a
        # real generation would burn a slot on every client launch, so it is
        # short-circuited here exactly as the parallel path did.
        if not prompt or not prompt.strip():
            empty = OllamaGenerateResponse(
                model=self._model_name,
                created_at=self._get_timestamp(),
                response="",
                done=True,
                done_reason="stop",
            )
            if request.stream:

                async def _empty() -> AsyncIterator[bytes]:
                    yield orjson.dumps(empty.model_dump()) + b"\n"

                return StreamingResponse(_empty(), media_type="application/x-ndjson")
            return empty

        from sglang.srt.entrypoints.openai.protocol import CompletionRequest

        completion_request = CompletionRequest(
            model=self._model_name,
            prompt=prompt,
            stream=bool(request.stream),
            **self._openai_fields(request),
        )

        start = time.time_ns()
        response = await self._completion.handle_request(
            completion_request, raw_request
        )

        if request.stream:
            return self._ndjson(self._generate_stream(response), response)
        return self._generate_reply(response, start)

    def _generate_reply(self, response, start_ns: int):
        if self._is_error(response):
            return response
        text, prompt_tokens, completion_tokens = self._read_completion(response)
        return OllamaGenerateResponse(
            model=self._model_name,
            created_at=self._get_timestamp(),
            response=text,
            done=True,
            done_reason="stop",
            total_duration=time.time_ns() - start_ns,
            prompt_eval_count=prompt_tokens,
            eval_count=completion_tokens,
        )

    async def _generate_stream(self, response) -> AsyncIterator[bytes]:
        async for delta, done in self._sse_deltas(response, chat=False):
            payload = OllamaGenerateStreamResponse(
                model=self._model_name,
                created_at=self._get_timestamp(),
                response="" if done else delta,
                done=done,
                done_reason="stop" if done else None,
            )
            yield orjson.dumps(payload.model_dump()) + b"\n"

    # -- OpenAI response reading ----------------------------------------

    @staticmethod
    def _is_error(response) -> bool:
        """An error from the front is passed through UNCHANGED.

        Re-dressing it in an Ollama envelope would hide which layer refused,
        and the OpenAI error already names its own cause.
        """
        return int(getattr(response, "status_code", 200) or 200) >= 400

    @staticmethod
    def _usage(response):
        usage = getattr(response, "usage", None)
        if usage is None:
            return None, None
        return getattr(usage, "prompt_tokens", None), getattr(
            usage, "completion_tokens", None
        )

    def _read_chat(self, response):
        choices = getattr(response, "choices", None) or []
        text = ""
        if choices:
            message = getattr(choices[0], "message", None)
            text = getattr(message, "content", "") or ""
        prompt_tokens, completion_tokens = self._usage(response)
        return text, prompt_tokens, completion_tokens

    def _read_completion(self, response):
        choices = getattr(response, "choices", None) or []
        text = getattr(choices[0], "text", "") if choices else ""
        prompt_tokens, completion_tokens = self._usage(response)
        return text or "", prompt_tokens, completion_tokens

    async def _sse_deltas(self, response, *, chat: bool):
        """Yield ``(delta, done)`` from an OpenAI SSE stream.

        The OpenAI stream already carries DELTAS, so no running-text
        subtraction is needed here -- unlike the parallel path, which received
        cumulative text and had to difference it. One final ``("", True)`` is
        emitted so the NDJSON always ends with Ollama's terminal object, even
        if the upstream stream ends without ``[DONE]``.
        """
        iterator = getattr(response, "body_iterator", None)
        if iterator is None:
            yield "", True
            return
        buffer = ""
        async for raw in iterator:
            chunk = raw.decode() if isinstance(raw, (bytes, bytearray)) else raw
            buffer += chunk
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    continue
                try:
                    payload = json.loads(data)
                except ValueError:
                    # A malformed chunk must not take the stream down; the
                    # terminal object below still closes it honestly.
                    continue
                for choice in payload.get("choices") or []:
                    piece = (
                        (choice.get("delta") or {}).get("content")
                        if chat
                        else choice.get("text")
                    )
                    if piece:
                        yield piece, False
        yield "", True

    @staticmethod
    def _ndjson(generator: AsyncIterator[bytes], source) -> StreamingResponse:
        """Wrap the translated stream, keeping the guarded iterator upstream.

        ``source`` is the response returned by the OpenAI front, whose
        ``body_iterator`` already carries the #344 watchdog; ``generator``
        consumes it. Returning a new StreamingResponse here does not bypass
        that watchdog, it sits downstream of it.
        """
        return StreamingResponse(generator, media_type="application/x-ndjson")

    # -- metadata --------------------------------------------------------

    def get_tags(self) -> OllamaTagsResponse:
        """/api/tags. Same payload the parallel path produced."""
        from sglang.srt.entrypoints.ollama.protocol import OllamaModelInfo

        name = self._model_name
        return OllamaTagsResponse(
            models=[
                OllamaModelInfo(
                    name=name,
                    model=name,
                    modified_at=self._get_timestamp(),
                    size=0,  # model size is not tracked
                    digest=(
                        "sha256:sglang00000000000000000000000000000000000000"
                        "0000000000000000000000"
                    ),
                    details={
                        "format": "sglang",
                        "family": name.split("/")[-1] if "/" in name else name,
                        "parameter_size": "unknown",
                    },
                )
            ]
        )

    def get_show(self, model: str) -> OllamaShowResponse:
        """/api/show. Same payload the parallel path produced."""
        family = model.split("/")[-1] if "/" in model else model
        for suffix in ("-Instruct", "-Chat", "-Base"):
            if family.endswith(suffix):
                family = family[: -len(suffix)]
                break
        context_len = self._context_len if self._context_len else 4096
        return OllamaShowResponse(
            license="",
            modelfile=f"FROM {model}\nPARAMETER num_ctx {context_len}\n",
            parameters=f"num_ctx {context_len}",
            template="",
            modified_at=self._get_timestamp(),
            details={
                "parent_model": "",
                "format": "sglang",
                "family": family,
                "families": [family],
                "parameter_size": "unknown",
                "quantization_level": "",
            },
            model_info={
                "general.architecture": family,
                "general.name": model,
                "general.parameter_count": 0,
                f"{family}.context_length": context_len,
                f"{family}.block_count": 0,
                f"{family}.embedding_length": 0,
                f"{family}.attention.head_count": 0,
            },
            capabilities=["completion"],
        )

    def inert_fields(self) -> Dict[str, str]:
        return dict(self.INERT_FIELDS)
