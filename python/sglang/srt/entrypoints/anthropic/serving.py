"""Handler for Anthropic Messages API requests.

Converts Anthropic requests to OpenAI ChatCompletion format, delegates to
OpenAIServingChat for processing, and converts responses back to Anthropic format.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import TYPE_CHECKING, Any, AsyncGenerator, Optional, Union

from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ValidationError

from sglang.srt.environ import envs
from sglang.srt.entrypoints.anthropic.protocol import (
    KNOWN_CONTENT_BLOCK_TAGS,
    AnthropicContentBlock,
    AnthropicCountTokensRequest,
    AnthropicCountTokensResponse,
    AnthropicError,
    AnthropicErrorResponse,
    AnthropicMessageEndDelta,
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
    AnthropicStreamEvent,
    AnthropicUsage,
    ContentBlockDeltaEvent,
    ContentBlockStartEvent,
    ContentBlockStopEvent,
    ErrorEvent,
    InputJsonDelta,
    MessageDeltaEvent,
    MessageStartEvent,
    MessageStopEvent,
    PingEvent,
    SignatureDelta,
    TextBlock,
    TextDelta,
    ThinkingBlock,
    ThinkingDelta,
    ToolUseBlock,
    is_server_tool,
)
from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    StreamOptions,
    Tool,
    ToolChoice,
    ToolChoiceFuncName,
)
from sglang.srt.observability.req_time_stats import monotonic_time
from sglang.srt.parser.template_detection import detect_inline_system_support

if TYPE_CHECKING:
    from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat

logger = logging.getLogger(__name__)

# Map OpenAI finish reasons to Anthropic stop reasons. Only the four
# values in ``AnthropicMessagesResponse.stop_reason``'s Literal are valid
# on the wire; ``content_filter`` and ``abort`` have no perfect mapping
# so they fall through to the ``end_turn`` default with a WARNING at the
# call site so operators don't lose the safety/abort signal in logs.
# ``stop_sequence`` is the fourth Anthropic stop reason and is NOT reachable
# from an OpenAI finish_reason: the backend reports a stop-string hit as plain
# ``"stop"`` and carries the matched string out of band in
# ``matched_stop`` (``openai/protocol.py:1107`` non-streaming,
# ``:1169`` streaming). ``_resolve_stop_sequence`` below turns that into the
# Anthropic pair (stop_reason="stop_sequence", stop_sequence="<the string>").
STOP_REASON_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
}

# Cadence for keep-alive ``ping`` events on an otherwise silent stream.
# Module-level so tests can shorten it; read at runtime, never captured.
PING_INTERVAL_SECONDS = 5.0


class ToolCallAssemblyError(Exception):
    """A tool call could not be reassembled from the backend's deltas.

    Raised (non-streaming) or converted into an SSE ``error`` event
    (streaming) whenever the arguments of a tool call are missing,
    truncated, or unclaimed by any function name. The alternative — the
    behaviour this replaces — was to hand the client a well-formed
    ``tool_use`` block with ``input={}``, which is indistinguishable from a
    genuine zero-argument call. Agents acted on those: a ``Bash`` call with
    no ``command`` is not a no-op, it is a lost instruction that the client
    then rejects as an invalid tool input, burning the whole turn. An error
    the caller can see is always preferable to a plausible fabrication.
    """


ERROR_TYPE_MAP = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    408: "request_timeout_error",
    429: "rate_limit_error",
    500: "api_error",
    502: "api_error",
    503: "overloaded_error",
    504: "api_error",
}


def _cached_prompt_tokens(usage) -> int:
    prompt_tokens_details = getattr(usage, "prompt_tokens_details", None)
    return getattr(prompt_tokens_details, "cached_tokens", 0) or 0


def _reported_cached_tokens(usage) -> Optional[int]:
    """Cached prompt tokens as REPORTED, or None when nothing was reported.

    #557. Three states, and collapsing any two of them loses a fact:

      * details present, ``cached_tokens`` > 0 -> a cache hit of that size;
      * details present, ``cached_tokens`` == 0 -> a measured MISS. This must
        surface as 0, not as absence, or a client cannot tell "no cache hit"
        from "this server does not report cache usage";
      * no ``prompt_tokens_details`` at all -> the backend reported nothing.
        This must stay ABSENT. Answering 0 here would publish a measurement
        that was never taken, which is the same defaulted-measurement defect
        the #606 lesson names, and it would silently convert "unknown" into
        "definitely no cache".

    ``_cached_prompt_tokens`` keeps returning an int because the input-token
    arithmetic needs one; this function is for the REPORTING side, where the
    difference between unknown and zero is the whole point.
    """
    prompt_tokens_details = getattr(usage, "prompt_tokens_details", None)
    if prompt_tokens_details is None:
        return None
    cached = getattr(prompt_tokens_details, "cached_tokens", None)
    if cached is None:
        return None
    return int(cached)


def _anthropic_input_tokens(usage) -> int:
    prompt = getattr(usage, "prompt_tokens", 0) or 0
    cached = _cached_prompt_tokens(usage)
    if cached > prompt:
        # Upstream telemetry bug: cached cannot exceed the prompt it caches.
        # Clamping silently here would hide the discrepancy from billing
        # dashboards, so make it visible at WARNING level.
        logger.warning(
            "Cached tokens (%d) exceed prompt tokens (%d); clamping "
            "input_tokens to 0. This usually indicates an upstream "
            "telemetry bug.",
            cached,
            prompt,
        )
    return max(prompt - cached, 0)


def _anthropic_usage_from_openai(
    usage,
    *,
    include_input: bool,
    include_output: bool,
    force_zero_output: bool = False,
) -> AnthropicUsage:
    if usage is None:
        return AnthropicUsage(
            input_tokens=0 if include_input else None,
            output_tokens=0 if include_output else None,
        )

    usage_fields: dict[str, int] = {}
    if include_input:
        usage_fields["input_tokens"] = _anthropic_input_tokens(usage)
        # #557: report a MEASURED zero as 0, and report nothing when nothing
        # was measured.
        #
        # This was `if cached_tokens:`, which is falsy at 0, so a request that
        # simply got no cache hit omitted the field entirely -- and a client
        # then cannot tell "no cache hit" from "this server does not report
        # cache usage". Those are different facts.
        #
        # The cure is not "always emit a number": a backend that sent no
        # prompt_tokens_details reported nothing, and answering 0 for it would
        # publish a measurement never taken. See _reported_cached_tokens.
        reported_cached = _reported_cached_tokens(usage)
        if reported_cached is not None:
            usage_fields["cache_read_input_tokens"] = reported_cached
    if include_output:
        usage_fields["output_tokens"] = (
            0 if force_zero_output else (getattr(usage, "completion_tokens", 0) or 0)
        )
    return AnthropicUsage(**usage_fields)


def _resolve_stop_sequence(
    matched_stop: Any,
    stop_sequences: Optional[list[str]],
    accumulated_text: Optional[str] = None,
) -> Optional[str]:
    """Return the caller's stop sequence that ended this completion, if any.

    Primary source is the backend's ``matched_stop``. It is
    ``Union[None, int, str]``: an ``int`` is a stop TOKEN id (no Anthropic
    equivalent — ``stop_sequences`` is a string API) and only a ``str`` can
    be a stop sequence. The string is additionally required to be one the
    CALLER asked for: the chat template contributes its own stop strings to
    the same OpenAI ``stop`` list, and an EOS-ish template stop is an
    ``end_turn``, not a ``stop_sequence``.

    ``accumulated_text`` drives a fallback for backends that report no
    ``matched_stop`` at all. LIMITATION, stated because it is easy to
    mistake for a guarantee: SGLang strips the matched stop string from the
    emitted text unless ``no_stop_trim`` is set, so the suffix probe
    normally finds nothing and correctly returns None. It only rescues the
    no-trim configuration; it is not a substitute for ``matched_stop``.
    """
    if not stop_sequences:
        return None
    if isinstance(matched_stop, str) and matched_stop in stop_sequences:
        return matched_stop
    if accumulated_text:
        for sequence in stop_sequences:
            if sequence and accumulated_text.endswith(sequence):
                return sequence
    return None


def _anthropic_tool_use_id(backend_id: Optional[str]) -> str:
    """Normalise a backend tool-call id to Anthropic's ``toolu_`` form.

    Anthropic tool_use ids are ``toolu_...``; the OpenAI-compatible backend
    mints ``call_...``. Clients that validate the prefix (and transcripts
    replayed into a real Anthropic endpoint) reject the raw backend id, so
    the OUTBOUND id is normalised in both the streaming and non-streaming
    paths.

    The round trip holds because the rewrite is outbound-only and
    deterministic: whatever id we emit is echoed back verbatim on
    ``tool_result.tool_use_id`` and forwarded to the backend as the
    ``tool_call_id`` unchanged (``_convert_to_chat_completion_request``
    never rewrites an INBOUND id), so older clients and replayed
    transcripts carrying arbitrary id strings keep working.
    """
    if not backend_id:
        return f"toolu_{uuid.uuid4().hex}"
    if backend_id.startswith("toolu_"):
        return backend_id
    if backend_id.startswith("call_"):
        return f"toolu_{backend_id[len('call_') :]}"
    return f"toolu_{backend_id}"


def _extract_system_text(
    content: Union[str, list[AnthropicContentBlock]],
) -> Optional[str]:
    """Flatten a system message's content to a trimmed string, or ``None``."""
    if isinstance(content, str):
        return content.strip() or None
    texts = []
    for block in content:
        if isinstance(block, BaseModel) and getattr(block, "type", None) == "text":
            text = getattr(block, "text", "")
        elif isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text", "")
        else:
            continue
        text = (text or "").strip()
        if text:
            texts.append(text)
    return "\n".join(texts) if texts else None


def _wrap_sse_event(data: str, event_type: str) -> str:
    """Format an Anthropic SSE event with event type and data lines."""
    return f"event: {event_type}\ndata: {data}\n\n"


def _scrub_error_message(message: str, status_code: int) -> str:
    """Return a safe outward-facing error message.

    5xx is always generic — never echo upstream ``str(e)`` payloads, which
    may contain stack frames, file paths, or PII. 4xx keeps the original
    message (truncated and with obvious traceback lines stripped) so
    callers see the real validation failure.
    """
    if status_code >= 500:
        return "Internal server error"
    if not message:
        return "Request failed"
    safe_lines = [
        ln
        for ln in message.splitlines()
        if not ln.startswith("Traceback") and 'File "/' not in ln
    ]
    cleaned = "\n".join(safe_lines).strip()
    if len(cleaned) > 500:
        cleaned = cleaned[:500] + "…"
    return cleaned or "Request failed"


class AnthropicServing:
    """Handler for Anthropic Messages API requests.

    Acts as a translation layer between Anthropic's Messages API and SGLang's
    OpenAI-compatible chat completion infrastructure.
    """

    def __init__(self, openai_serving_chat: OpenAIServingChat):
        self.openai_serving_chat = openai_serving_chat
        self._merge_inline_system = not detect_inline_system_support(
            self._chat_template()
        )

    def _chat_template(self) -> Optional[str]:
        tokenizer_manager = getattr(self.openai_serving_chat, "tokenizer_manager", None)
        if tokenizer_manager is None:
            return None
        tokenizer = getattr(tokenizer_manager, "tokenizer", None)
        if tokenizer is None:
            return None
        return getattr(tokenizer, "chat_template", None)

    async def handle_messages(
        self,
        request: AnthropicMessagesRequest,
        raw_request: Request,
    ) -> Union[JSONResponse, StreamingResponse]:
        """Main entry point for /v1/messages endpoint."""
        try:
            chat_request = self._convert_to_chat_completion_request(request)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception("Error converting Anthropic request: %s", e)
            return self._error_response(
                status_code=400,
                error_type="invalid_request_error",
                message=str(e),
            )

        if request.stream:
            return await self._handle_streaming(chat_request, request, raw_request)
        else:
            return await self._handle_non_streaming(chat_request, request, raw_request)

    def _convert_to_chat_completion_request(
        self, anthropic_request: AnthropicMessagesRequest
    ) -> ChatCompletionRequest:
        """Convert an Anthropic Messages request to an OpenAI ChatCompletion request."""
        openai_messages = []

        def _convert_anthropic_image_source_to_openai_part(
            source: Any,
        ) -> Optional[dict]:
            # Source may arrive as a Pydantic model (typed ImageBlock.source)
            # or as a raw dict when parsed from a nested tool_result payload.
            if isinstance(source, BaseModel):
                source = source.model_dump(exclude_none=True)
            if not isinstance(source, dict):
                return None

            source_type = source.get("type")
            if source_type == "base64":
                media_type = source.get("media_type", "image/png")
                data = source.get("data", "")
                if not data:
                    return None
                return {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{data}",
                    },
                }

            url = source.get("url")
            if url:
                return {
                    "type": "image_url",
                    "image_url": {
                        "url": url,
                    },
                }

            return None

        def _text_from_search_result(item: dict[str, Any]) -> str:
            search_parts = []
            title = item.get("title")
            if title:
                search_parts.append(f"Title: {title}")

            source = item.get("source")
            if isinstance(source, dict):
                source_text = source.get("url") or source.get("text")
                if source_text:
                    search_parts.append(f"Source: {source_text}")
            elif source:
                search_parts.append(f"Source: {source}")

            content = item.get("content")
            content_parts = []
            if isinstance(content, str):
                content_parts.append(content)
            elif isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") == "text" and part.get("text"):
                        content_parts.append(part["text"])
            if content_parts:
                search_parts.append("Content: " + "\n".join(content_parts))

            return "\n".join(search_parts)

        def _convert_tool_result_content(
            content: Any,
        ) -> tuple[Union[str, list[dict]], str]:
            if isinstance(content, list):
                tool_content_parts = []
                tool_text_parts = []

                for raw_item in content:
                    # Items may be typed Pydantic blocks (after request
                    # validation) or raw dicts (from legacy callers). Coerce
                    # to dict so the existing key-based logic still works.
                    if isinstance(raw_item, BaseModel):
                        item = raw_item.model_dump(exclude_none=True)
                    elif isinstance(raw_item, dict):
                        item = raw_item
                    else:
                        continue

                    item_type = item.get("type")
                    if item_type == "text":
                        text = item.get("text", "")
                        if text:
                            tool_text_parts.append(text)
                            tool_content_parts.append({"type": "text", "text": text})
                    elif item_type == "image":
                        image_part = _convert_anthropic_image_source_to_openai_part(
                            item.get("source")
                        )
                        if image_part is not None:
                            tool_content_parts.append(image_part)
                    elif item_type == "tool_reference":
                        # Anthropic uses `tool_name`; the SGLang chat template
                        # matches on `name`. Translate at the boundary.
                        ref_name = item.get("tool_name") or item.get("name")
                        if ref_name:
                            tool_content_parts.append(
                                {"type": "tool_reference", "name": ref_name}
                            )
                    elif item_type == "search_result":
                        search_text = _text_from_search_result(item)
                        if search_text:
                            tool_text_parts.append(search_text)
                            tool_content_parts.append(
                                {"type": "text", "text": search_text}
                            )

                tool_text = "\n".join(tool_text_parts)
                if (
                    len(tool_content_parts) == 1
                    and tool_content_parts[0]["type"] == "text"
                ):
                    return tool_content_parts[0]["text"], tool_text
                if tool_content_parts:
                    return tool_content_parts, tool_text
                return "", tool_text

            tool_text = str(content) if content else ""
            return tool_text, tool_text

        def _convert_assistant_thinking_blocks(
            blocks: list[AnthropicContentBlock],
        ) -> Optional[str]:
            """Re-wrap prior-turn thinking blocks in the parser's own tokens.

            ``redacted_thinking`` carries encrypted bytes that no local
            parser can interpret. It is SKIPPED with a warning rather than
            raised on: a client that replays its own transcript ships the
            redacted block back on every subsequent turn, so raising turns
            one opaque block into a permanently 400-ing conversation. Same
            degrade-per-block rule as unknown content types — the dropped
            content is prior-turn reasoning the model does not need.
            On non-reasoning models (no detector configured) the rewrap is
            best-effort: we log a warning and drop the thinking text so a
            history echo doesn't 400 the whole request — the prior thinking
            is opaque context the model didn't need anyway.
            """
            redacted_count = sum(
                1 for block in blocks if block.type == "redacted_thinking"
            )
            if redacted_count:
                logger.warning(
                    "Skipping %d redacted_thinking block(s) in assistant "
                    "history: the payload is encrypted and no local "
                    "reasoning parser can interpret it",
                    redacted_count,
                )

            thinking_parts = [
                block.thinking
                for block in blocks
                if block.type == "thinking" and block.thinking
            ]
            if not thinking_parts:
                return None

            try:
                return self.openai_serving_chat.wrap_reasoning_history(
                    "\n".join(thinking_parts)
                )
            except ValueError as e:
                logger.warning(
                    "Dropping prior-turn thinking history (%d blocks): %s",
                    len(thinking_parts),
                    e,
                )
                return None

        system_parts: list[str] = []
        if anthropic_request.system:
            if isinstance(anthropic_request.system, str):
                if anthropic_request.system.strip():
                    system_parts.append(anthropic_request.system)
            else:
                for block in anthropic_request.system:
                    if block.type == "text" and block.text:
                        system_parts.append(block.text)

        if self._merge_inline_system:
            for msg in anthropic_request.messages:
                if msg.role != "system":
                    continue
                text = _extract_system_text(msg.content)
                if text:
                    system_parts.append(text)

        if system_parts:
            openai_messages.append(
                {"role": "system", "content": "\n".join(system_parts)}
            )

        def _emit_user_message(parts: list[dict]) -> None:
            """Append accumulated parts as a user message, then clear them.

            Used to flush content collected BEFORE a tool_result so the
            wire order stays user(pre) → tool → user(post). Without this
            flush, text/image parts that appeared before a tool_result
            block would be moved AFTER the tool message at end of loop.
            """
            if not parts:
                return
            if len(parts) == 1 and parts[0]["type"] == "text":
                openai_messages.append({"role": "user", "content": parts[0]["text"]})
            else:
                openai_messages.append({"role": "user", "content": list(parts)})
            parts.clear()

        # Convert messages
        for msg in anthropic_request.messages:
            if msg.role == "system" and self._merge_inline_system:
                continue
            if isinstance(msg.content, str):
                openai_messages.append({"role": msg.role, "content": msg.content})
                continue

            # Complex content with blocks
            openai_msg = {"role": msg.role}
            content_parts: list[dict] = []
            tool_calls: list[dict] = []

            if msg.role == "assistant":
                reasoning_history = _convert_assistant_thinking_blocks(msg.content)
                if reasoning_history is not None:
                    content_parts.append({"type": "text", "text": reasoning_history})

            for block in msg.content:
                # ``thinking``/``redacted_thinking`` blocks are surfaced via
                # the reasoning-history reconstruction above; skip them here
                # to avoid double-injecting their text into the prompt.
                if block.type in ("thinking", "redacted_thinking"):
                    continue

                # ``is not None`` (not truthy) so an empty-string text block
                # still produces a placeholder text part — without it, an
                # assistant turn whose only content is "" vanishes and
                # subsequent user→user pairs trip strict chat templates.
                if block.type == "text" and block.text is not None:
                    content_parts.append({"type": "text", "text": block.text})

                elif block.type == "image" and block.source:
                    image_part = _convert_anthropic_image_source_to_openai_part(
                        block.source
                    )
                    if image_part is not None:
                        content_parts.append(image_part)

                elif block.type == "search_result":
                    search_text = _text_from_search_result(block.model_dump())
                    if search_text:
                        content_parts.append({"type": "text", "text": search_text})

                elif block.type == "tool_use":
                    tool_call = {
                        "id": block.id or f"call_{uuid.uuid4().hex}",
                        "type": "function",
                        "function": {
                            "name": block.name or "",
                            "arguments": json.dumps(block.input or {}),
                        },
                    }
                    tool_calls.append(tool_call)

                elif block.type == "tool_result":
                    tool_content, tool_text = _convert_tool_result_content(
                        block.content
                    )

                    # Use tool_use_id (per spec) with fallback to id
                    tool_call_id = block.tool_use_id or block.id or ""

                    # Tool results from user become separate tool messages.
                    # Flush any pending text/image first so the wire order
                    # is preserved (a tool_result that arrived AFTER a text
                    # block must come AFTER that text in OpenAI form too).
                    if msg.role == "user":
                        _emit_user_message(content_parts)
                        openai_messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "content": tool_content,
                            }
                        )
                    else:
                        content_parts.append(
                            {
                                "type": "text",
                                "text": f"Tool result: {tool_text}",
                            }
                        )

                elif block.type not in KNOWN_CONTENT_BLOCK_TAGS:
                    # Forward compatibility (see UnknownContentBlock in
                    # protocol.py): Anthropic adds block types over time, so
                    # an unmodelled type degrades THIS BLOCK, never the whole
                    # conversation. One warning naming the type so operators
                    # can see what a client is sending that we drop.
                    logger.warning(
                        "Skipping unsupported Anthropic content block type "
                        "%r in a %r message; the rest of the conversation is "
                        "converted normally",
                        block.type,
                        msg.role,
                    )

            # Attach tool calls to assistant messages
            if tool_calls:
                openai_msg["tool_calls"] = tool_calls

            # Attach content
            if content_parts:
                if len(content_parts) == 1 and content_parts[0]["type"] == "text":
                    openai_msg["content"] = content_parts[0]["text"]
                else:
                    openai_msg["content"] = content_parts
                openai_messages.append(openai_msg)
            elif tool_calls:
                openai_messages.append(openai_msg)
            elif msg.role == "user":
                # User turn that was entirely tool_results — the tool
                # messages were already emitted above, nothing left.
                continue
            else:
                # Assistant turn with no content and no tool_calls: emit
                # an empty-string placeholder so strict templates still
                # see a valid role-alternation sequence.
                openai_msg["content"] = ""
                openai_messages.append(openai_msg)

        # Build ChatCompletionRequest
        request_data = {
            "messages": openai_messages,
            "model": anthropic_request.model,
            "max_tokens": anthropic_request.max_tokens,
            "stream": anthropic_request.stream or False,
            # #764: Anthropic's ``max_tokens`` is a CEILING -- "the maximum
            # number of tokens to generate", a stop condition, not an amount the
            # caller is asking to have produced. Clients bake in a fixed budget
            # for the whole model family (claude-cli sends 32000 on every turn)
            # and cannot know this server's context length, so a request whose
            # ceiling is above it is normal traffic, not a client error. Ask the
            # pipeline to lower the ceiling to what fits instead of returning
            # 400; when generation actually reaches it, the completion ends with
            # finish_reason "length", which this front already reports to the
            # caller as stop_reason "max_tokens" -- so a shortened budget stays
            # visible in the response rather than being silently swallowed.
            "clamp_max_tokens": True,
        }

        if anthropic_request.temperature is not None:
            request_data["temperature"] = anthropic_request.temperature
        if anthropic_request.top_p is not None:
            request_data["top_p"] = anthropic_request.top_p
        if anthropic_request.top_k is not None:
            request_data["top_k"] = anthropic_request.top_k
        if anthropic_request.stop_sequences is not None:
            request_data["stop"] = anthropic_request.stop_sequences

        # #557: pass the caller's chat-template arguments through.
        #
        # Set only when present, so a request that did not ask does not
        # acquire a dict it never sent -- an always-present ``{}`` reads
        # downstream as "the caller specified no kwargs" rather than "the
        # caller said nothing", and those differ for any template that
        # branches on the dict's presence.
        #
        # Placed BEFORE construction on purpose: the ``thinking`` handling
        # below runs after and calls ``apply_reasoning_enabled``, which merges
        # into whatever this request carries and overrides only the reasoning
        # toggle. So a caller's unrelated keys survive, while Anthropic's
        # typed ``thinking`` semantics stay authoritative over the toggle --
        # which is the existing OpenAI write-side behaviour, mirrored rather
        # than re-decided here.
        if anthropic_request.chat_template_kwargs is not None:
            request_data["chat_template_kwargs"] = dict(
                anthropic_request.chat_template_kwargs
            )

        # Enable usage in stream so we can report it
        if anthropic_request.stream:
            request_data["stream_options"] = StreamOptions(
                include_usage=True,
                continuous_usage_stats=True,
            )

        chat_request = ChatCompletionRequest(**request_data)

        if anthropic_request.thinking is not None:
            # The protocol layer already enforces SDK shape:
            #   enabled  -> budget_tokens required (>=1024), display optional
            #   disabled -> neither budget_tokens nor display allowed
            #   adaptive -> budget_tokens forbidden, display optional
            # So by the time we get here ``budget_tokens`` can only be
            # set on ``enabled``. It maps onto the OpenAI front's
            # ``thinking_budget`` (#540), which caps the thinking section at
            # that many tokens via the built-in thinking budget logit
            # processor. Enforcement needs the marker tokens, i.e. a
            # configured ``--reasoning-parser``; without one there is nothing
            # to cap, so the budget stays accept-and-log as before rather
            # than 400-ing a request the Anthropic SDK would have accepted.
            if anthropic_request.thinking.budget_tokens is not None:
                if getattr(self.openai_serving_chat, "reasoning_parser", None):
                    chat_request.thinking_budget = (
                        anthropic_request.thinking.budget_tokens
                    )
                else:
                    logger.warning(
                        "Anthropic thinking.budget_tokens=%d is accepted for "
                        "SDK compatibility but this model has no reasoning "
                        "parser configured — the budget is not enforced",
                        anthropic_request.thinking.budget_tokens,
                    )
            # Claude 4.7's ``adaptive`` is treated identically to ``enabled``
            # because the local backend has no auto-throttle equivalent.
            # Anything other than ``disabled`` enables reasoning.
            enabled = anthropic_request.thinking.type != "disabled"
            if anthropic_request.thinking.display == "omitted":
                # Anthropic 4.7 spec: keep reasoning ON but hide reasoning
                # text from the client. The OpenAI streaming pipeline has
                # no equivalent suppress knob — log so operators can see
                # the request, then proceed with normal reasoning emission.
                logger.warning(
                    "Anthropic thinking.display='omitted' is accepted for "
                    "SDK compatibility but reasoning text will still be "
                    "emitted to the client"
                )
            self.openai_serving_chat.apply_reasoning_enabled(chat_request, enabled)
        else:
            # Anthropic semantics: extended thinking is OFF BY DEFAULT. An
            # absent ``thinking`` field means the same as
            # ``{"type":"disabled"}`` — it is NOT "leave the server default
            # alone". Without this branch a boot carrying
            # ``--reasoning-parser qwen3`` answers every plain request with a
            # leading ``thinking`` block that eats the whole ``max_tokens``
            # budget, so a Claude Code tool round trip never gets emitted.
            #
            # This DELIBERATELY overrides the server-level reasoning-parser
            # default, and only on the Anthropic front: the OpenAI front
            # reads its own ``chat_template_kwargs``/``reasoning_effort``
            # and never passes through here, so its behaviour is unchanged.
            try:
                self.openai_serving_chat.apply_reasoning_enabled(chat_request, False)
            except ValueError as e:
                # An always-on reasoning parser cannot be disabled. Explicit
                # ``{"type":"disabled"}`` still surfaces that as a 400 (the
                # caller asked for something the model cannot do), but an
                # ABSENT field is not a request — refusing here would 400
                # every plain message on such a model. Warn and serve.
                logger.warning(
                    "Anthropic default thinking-off could not be applied: %s "
                    "— serving with the model's reasoning default instead",
                    e,
                )

        # Claude 4.7 ``output_config``: map ``effort`` onto the OpenAI
        # ``reasoning_effort`` knob.
        #
        # ``xhigh`` USED TO COLLAPSE TO ``max`` here, and that was backwards for
        # the family this fork serves. ``reasoning_effort`` is injected into the
        # chat-template kwargs (``openai/serving_chat.py:175``), so the value
        # reaches the template verbatim -- and every Qwen3.8 checkpoint on this
        # box accepts exactly ``('xhigh', 'medium', 'low')`` and calls
        # ``raise_exception`` on anything else, which surfaces as HTTP 500. So a
        # client asking for the tier the model DOES support had its request
        # rewritten into one the model rejects, while the same tier was
        # reachable by omitting the field (the template defaults to ``xhigh``).
        # The value now passes through unchanged and ``xhigh`` is accepted by
        # our own schema, which already carries ``max`` as a comparable
        # extension.
        #
        # The collapse survives as an OPT-IN for a deployment whose template
        # names its top tier ``max`` instead, and it says so in the log rather
        # than rewriting a request silently.
        #
        # ``task_budget`` is a soft hint forwarded as a custom param so the
        # model can see it without it becoming a hard cap (``max_tokens``
        # is still the hard cap).
        if anthropic_request.output_config is not None:
            oc = anthropic_request.output_config
            if oc.effort is not None:
                effort = oc.effort
                if effort == "xhigh":
                    target = (
                        envs.SGLANG_ANTHROPIC_XHIGH_EFFORT.get() or "xhigh"
                    ).strip()
                    if target != "xhigh":
                        logger.info(
                            "Anthropic output_config.effort 'xhigh' collapsed to "
                            "%r by SGLANG_ANTHROPIC_XHIGH_EFFORT. The default is "
                            "to pass 'xhigh' through; a template that does not "
                            "accept %r will reject this request.",
                            target,
                            target,
                        )
                        effort = target
                chat_request.reasoning_effort = effort
            if oc.task_budget is not None:
                # Custom params are silently ignored by backends that
                # don't recognise them; logging it makes the propagation
                # visible.
                logger.info(
                    "Anthropic output_config.task_budget hint: %d %s",
                    oc.task_budget.total,
                    oc.task_budget.type,
                )

        # ``betas`` is the Anthropic SDK's opt-in feature list (e.g.
        # ``["thinking-2025-08-04"]``). The local backend has no
        # equivalent beta system; accept-and-log so requests don't 400.
        if anthropic_request.betas:
            logger.info(
                "Anthropic request opted into betas %s — no-op locally",
                anthropic_request.betas,
            )

        # Convert tools. Deferred tools stay in the list with defer_loading=True;
        # the chat template hides them from the initial <tools> block and renders
        # them on demand when a tool_reference block names them.
        if anthropic_request.tools:
            converted_tools = []
            for tool in anthropic_request.tools:
                if is_server_tool(tool):
                    # Anthropic server-side tools (web_search_*, computer_*,
                    # bash_*, text_editor_*) have no client-side input_schema
                    # because Anthropic provides the implementation. We can't
                    # forward them to the OpenAI tools array (which requires a
                    # schema), so skip with a visible log.
                    logger.info(
                        "Skipping built-in Anthropic server tool %r (type=%r): "
                        "no native support in the OpenAI-compatible backend",
                        tool.name,
                        tool.type,
                    )
                    continue

                # Custom tools always have a validated input_schema
                # (enforced at Pydantic parse time).
                converted_tools.append(
                    Tool(
                        type="function",
                        defer_loading=tool.defer_loading,
                        function={
                            "name": tool.name,
                            "description": tool.description or "",
                            "parameters": tool.input_schema,
                        },
                    )
                )

            if converted_tools:
                chat_request.tools = converted_tools

        # Convert tool choice. ``any``/``tool`` express a hard requirement
        # ("the model MUST call a tool"); if every requested tool was a
        # server-side Anthropic built-in that we just skipped, there is
        # no tool the model could call. Silently downgrading to "no tool"
        # would deceive the caller, so raise an explicit 400.
        if anthropic_request.tool_choice is not None:
            tc_type = anthropic_request.tool_choice.type
            if tc_type == "none":
                chat_request.tool_choice = "none"
            elif chat_request.tools:
                if tc_type == "auto":
                    chat_request.tool_choice = "auto"
                elif tc_type == "any":
                    chat_request.tool_choice = "required"
                elif tc_type == "tool":
                    tool_name = anthropic_request.tool_choice.name
                    # ``Tool.function`` is a ``Function`` Pydantic model, not
                    # a dict — access by attribute. A dict ``.get`` would
                    # AttributeError and surface as a 500 instead of the
                    # intended 400 / happy path.
                    if not any(
                        t.function.name == tool_name for t in chat_request.tools
                    ):
                        raise ValueError(
                            f"tool_choice references tool {tool_name!r} but it "
                            f"is not in the forwarded tools list "
                            f"(server-side Anthropic tools cannot be selected)"
                        )
                    chat_request.tool_choice = ToolChoice(
                        type="function",
                        function=ToolChoiceFuncName(name=tool_name),
                    )
            elif tc_type in ("any", "tool"):
                raise ValueError(
                    f"tool_choice={tc_type!r} requires at least one custom "
                    f"tool; all supplied tools were server-side Anthropic "
                    f"built-ins which the OpenAI-compatible backend cannot "
                    f"invoke"
                )
        elif chat_request.tools:
            chat_request.tool_choice = "auto"

        return chat_request

    async def _handle_non_streaming(
        self,
        chat_request: ChatCompletionRequest,
        anthropic_request: AnthropicMessagesRequest,
        raw_request: Request,
    ) -> JSONResponse:
        """Handle non-streaming Anthropic request by delegating to OpenAI handler."""
        # ``monotonic_time`` is ``time.perf_counter`` under the hood; the
        # downstream stats layer subtracts other ``perf_counter`` samples
        # from this, so they must come from the same clock.
        received_time = monotonic_time()

        # Validate
        error_msg = self.openai_serving_chat._validate_request(chat_request)
        if error_msg:
            return self._error_response(
                status_code=400,
                error_type="invalid_request_error",
                message=error_msg,
            )

        try:
            # Convert to internal request
            adapted_request, processed_request = (
                self.openai_serving_chat._convert_to_internal_request(
                    chat_request, raw_request
                )
            )
            adapted_request.received_time = received_time

            # Get response from OpenAI handler
            response = await self.openai_serving_chat._handle_non_streaming_request(
                adapted_request, processed_request, raw_request
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception("Error processing Anthropic request: %s", e)
            return self._error_response(
                status_code=500,
                error_type="api_error",
                message="Internal server error",
                exception_name=type(e).__name__,
            )

        # Check for error responses from OpenAI handler
        if not isinstance(response, ChatCompletionResponse):
            # It's an error response (ORJSONResponse)
            return self._convert_openai_error_response(response)

        # Convert to Anthropic response. A tool call that cannot be
        # assembled is reported as an error the caller can see rather than
        # being smoothed into a tool_use block with an empty input.
        try:
            anthropic_response = self._convert_response(
                response, stop_sequences=anthropic_request.stop_sequences
            )
        except ToolCallAssemblyError as e:
            return self._error_response(
                status_code=502,
                error_type="api_error",
                message=f"Tool call arguments were lost in transport: {e}",
                exception_name=type(e).__name__,
            )
        payload = anthropic_response.model_dump(exclude_none=True)
        # ``exclude_none`` is kept: it drops the optional cache/usage fields
        # the local backend never populates, which would otherwise be pure
        # null noise on every response. The ONE exception is
        # ``stop_sequence``: the Anthropic Message object documents it as
        # always present (null when no stop sequence fired) and strict SDK
        # parsers type it as ``Optional[str]``, for which a MISSING key is a
        # schema violation. ``stop_reason`` is always set by
        # ``_convert_response`` and needs no such repair.
        payload.setdefault("stop_sequence", None)
        return JSONResponse(content=payload)

    async def _handle_streaming(
        self,
        chat_request: ChatCompletionRequest,
        anthropic_request: AnthropicMessagesRequest,
        raw_request: Request,
    ) -> Union[StreamingResponse, JSONResponse]:
        """Handle streaming Anthropic request."""
        received_time = monotonic_time()

        # Validate
        error_msg = self.openai_serving_chat._validate_request(chat_request)
        if error_msg:
            return self._error_response(
                status_code=400,
                error_type="invalid_request_error",
                message=error_msg,
            )

        try:
            adapted_request, processed_request = (
                self.openai_serving_chat._convert_to_internal_request(
                    chat_request, raw_request
                )
            )
            adapted_request.received_time = received_time
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception("Error converting streaming request: %s", e)
            return self._error_response(
                status_code=500,
                error_type="api_error",
                message="Internal server error",
                exception_name=type(e).__name__,
            )

        return StreamingResponse(
            self._generate_anthropic_stream(
                adapted_request,
                processed_request,
                anthropic_request,
                raw_request,
            ),
            media_type="text/event-stream",
            background=self.openai_serving_chat.tokenizer_manager.create_abort_task(
                adapted_request
            ),
        )

    async def _generate_anthropic_stream(
        self,
        adapted_request,
        processed_request: ChatCompletionRequest,
        anthropic_request: AnthropicMessagesRequest,
        raw_request: Request,
    ) -> AsyncGenerator[str, None]:
        """Convert OpenAI chat stream to Anthropic event stream."""
        openai_stream = self.openai_serving_chat._generate_chat_stream(
            adapted_request, processed_request, raw_request
        )

        content_block_index = 0
        content_block_open = False
        content_block_type: Optional[str] = None
        captured_thinking_signature: str = ""
        finish_reason: Optional[str] = None
        matched_stop: Any = None
        accumulated_text: list[str] = []
        final_usage: Optional[AnthropicUsage] = None
        message_started = False
        had_content_delta = False
        message_id = f"msg_{uuid.uuid4().hex}"
        model = anthropic_request.model
        stop_sequences = anthropic_request.stop_sequences

        # --- tool-call assembly state -------------------------------------
        # The backend's incremental function-call detector does not
        # guarantee that a call's function name arrives on the first delta,
        # nor that its argument fragments arrive contiguously. Observed on
        # a thinking-on qwen deployment: a leading delta with ``name=None``,
        # argument fragments separated by a stray content token, and
        # fragments carrying a tool index other than the open block's.
        # Every one of those used to end as a dropped fragment and a
        # ``tool_use`` block with ``input={}`` on the client. The state
        # below makes the adapter robust regardless of what the detector
        # does: fragments are keyed by the backend's tool index and are
        # buffered until a name claims them, never discarded.
        open_tool_index: Optional[int] = None
        tool_names: dict[int, str] = {}
        tool_ids: dict[int, Optional[str]] = {}
        # Every fragment seen per index, delivered or not — the basis for
        # the completeness check at end of stream.
        tool_args: dict[int, list[str]] = {}
        # Fragments not yet attached to a content block (no name yet).
        pending_tool_args: dict[int, list[str]] = {}
        # Text/thinking held back while an open tool_use block still has
        # incomplete arguments, replayed once the block can be closed.
        deferred_content: list[tuple[str, str]] = []

        def _message_start_event(usage) -> MessageStartEvent:
            return MessageStartEvent(
                message=AnthropicMessagesResponse(
                    id=message_id,
                    content=[],
                    model=model,
                    usage=_anthropic_usage_from_openai(
                        usage,
                        include_input=True,
                        include_output=True,
                        force_zero_output=True,
                    ),
                ),
            )

        def _emit(event: AnthropicStreamEvent) -> str:
            return _wrap_sse_event(
                event.model_dump_json(exclude_none=True),
                event.type,
            )

        def _emit_message_start(event: MessageStartEvent) -> str:
            """Serialise message_start with the spec's explicit nulls.

            The shared ``_emit`` uses ``exclude_none=True``, which is right
            for every other event but drops ``stop_reason`` and
            ``stop_sequence`` from the embedded Message. Anthropic's
            ``message_start`` carries both as explicit ``null`` — the
            message has not stopped yet — and strict SDK clients type them
            as required-but-nullable, so a missing key is a schema
            violation. Repaired here only, not by widening ``_emit``.
            """
            payload = event.model_dump(exclude_none=True)
            payload["message"]["stop_reason"] = None
            payload["message"]["stop_sequence"] = None
            return _wrap_sse_event(json.dumps(payload), event.type)

        def _close_content_block_events() -> list[AnthropicStreamEvent]:
            nonlocal content_block_index, content_block_open
            nonlocal content_block_type, captured_thinking_signature

            events: list[AnthropicStreamEvent] = []
            if not content_block_open:
                return events

            # Only emit signature_delta when a real signature is available.
            # Anthropic's spec treats absence as "unsigned thinking"; an
            # empty-string signature would fail downstream verifiers.
            if content_block_type == "thinking" and captured_thinking_signature:
                events.append(
                    ContentBlockDeltaEvent(
                        index=content_block_index,
                        delta=SignatureDelta(
                            signature=captured_thinking_signature,
                        ),
                    )
                )

            events.append(ContentBlockStopEvent(index=content_block_index))
            content_block_open = False
            content_block_type = None
            content_block_index += 1
            captured_thinking_signature = ""
            return events

        def _ensure_content_block_events(
            block_type: str,
            content_block: AnthropicContentBlock,
            force_new: bool = False,
        ) -> list[AnthropicStreamEvent]:
            """Open a content_block, closing the prior one if needed.

            ``force_new=True`` closes an existing block even when its type
            matches — required when a stream emits two consecutive
            ``tool_use`` blocks: each tool needs its own
            ``content_block_start``/``stop`` pair and its own
            ``content_block_index``, otherwise the second tool's
            ``input_json_delta`` chunks would append to the first tool's
            JSON arguments and corrupt both tool calls.
            """
            nonlocal content_block_open, content_block_type

            events: list[AnthropicStreamEvent] = []
            if content_block_open and (force_new or content_block_type != block_type):
                events.extend(_close_content_block_events())
            if not content_block_open:
                events.append(
                    ContentBlockStartEvent(
                        index=content_block_index,
                        content_block=content_block,
                    )
                )
                content_block_open = True
                content_block_type = block_type
            return events

        def _tool_args_parse_state(index: int) -> str:
            """Classify the arguments accumulated for one tool index.

            Returns ``"empty"`` (nothing received yet — either a genuine
            zero-argument call or a call whose fragments are still in
            flight), ``"partial"`` (received but not valid JSON yet), or
            ``"complete"``.
            """
            joined = "".join(tool_args.get(index, []))
            if not joined.strip():
                return "empty"
            try:
                json.loads(joined)
            except (json.JSONDecodeError, ValueError):
                return "partial"
            return "complete"

        def _tool_block_awaits_arguments() -> bool:
            """True while the open tool_use block cannot safely be closed.

            Closing it would strand every argument fragment still to come:
            once the block is gone the fragments have nowhere to go. So
            interleaved text/thinking is held back instead of closing the
            block — the exact case that produced "Dropping tool_call
            argument delta with no open tool_use block" in production.
            """
            if not (content_block_open and content_block_type == "tool_use"):
                return False
            if open_tool_index is None:
                return False
            return _tool_args_parse_state(open_tool_index) != "complete"

        def _release_deferred_content_events() -> list[AnthropicStreamEvent]:
            """Replay text/thinking held back during an incomplete tool call.

            Ordering note: this emits the held content AFTER the tool_use
            block it was interleaved with, rather than splitting the tool
            call in two. The content is preserved in its original relative
            order; only its position against the one tool block moves.
            """
            nonlocal deferred_content

            events: list[AnthropicStreamEvent] = []
            if not deferred_content:
                return events
            held, deferred_content = deferred_content, []
            for kind, text in held:
                if kind == "thinking":
                    events.extend(
                        _ensure_content_block_events(
                            "thinking", ThinkingBlock(thinking="")
                        )
                    )
                    events.append(
                        ContentBlockDeltaEvent(
                            index=content_block_index,
                            delta=ThinkingDelta(thinking=text),
                        )
                    )
                else:
                    events.extend(
                        _ensure_content_block_events("text", TextBlock(text=""))
                    )
                    events.append(
                        ContentBlockDeltaEvent(
                            index=content_block_index,
                            delta=TextDelta(text=text),
                        )
                    )
            return events

        def _tool_assembly_failure() -> Optional[str]:
            """Describe the first tool call that could not be delivered intact.

            Called once at end of stream. Anything it reports is a fragment
            the client would otherwise never see, so the caller turns it
            into a visible error rather than shipping a ``tool_use`` block
            whose ``input`` silently lost its contents.
            """
            indices = set(tool_args) | set(tool_names) | set(pending_tool_args)
            for index in sorted(indices):
                name = tool_names.get(index)
                joined = "".join(tool_args.get(index, []))
                if name is None:
                    return (
                        f"tool call #{index} streamed {len(joined)} characters "
                        "of arguments but the backend never sent a function "
                        "name, so the call has no addressable tool"
                    )
                if pending_tool_args.get(index):
                    undelivered = "".join(pending_tool_args[index])
                    return (
                        f"tool call #{index} ({name}) has "
                        f"{len(undelivered)} characters of arguments that "
                        "were never attached to a content block"
                    )
                state = _tool_args_parse_state(index)
                if state == "partial":
                    return (
                        f"tool call #{index} ({name}) arguments are "
                        f"not valid JSON after {len(joined)} characters"
                    )
            return None

        def _ensure_message_started(usage) -> list[str]:
            """Emit message_start exactly once. Returns SSE frames to yield."""
            nonlocal message_started
            if message_started:
                return []
            message_started = True
            return [_emit_message_start(_message_start_event(usage))]

        def _build_error_event(error_type: str, message: str) -> ErrorEvent:
            return ErrorEvent(
                error=AnthropicError(type=error_type, message=message),
            )

        def _flush_on_error(error_type: str, message: str) -> list[str]:
            """Build a self-contained terminal SSE sequence on error.

            Guarantees that whatever events we emit on the failure path
            leave the wire in a valid state: message_start (if not yet
            sent), close any open content block, then ErrorEvent and
            MessageStopEvent. Strict SDK clients reject streams whose
            content_block_start has no matching content_block_stop, so
            the close step is mandatory even on the error path.
            """
            frames: list[str] = []
            frames.extend(_ensure_message_started(None))
            for event in _close_content_block_events():
                frames.append(_emit(event))
            frames.append(_emit(_build_error_event(error_type, message)))
            frames.append(_emit(MessageStopEvent()))
            return frames

        def _parse_upstream_error(data_str: str) -> Optional[tuple[str, str]]:
            """Detect an OpenAI handler streaming-error envelope.

            ``OpenAIServingChat.create_streaming_error_response`` emits
            ``data: {"error": {"object":"error","message":"...",
            "type":"BadRequestError","code":400}}``; the regular
            ChatCompletionStreamResponse validator rejects it. Pull the
            type/message out so the Anthropic client sees the real
            failure instead of a generic 'Stream processing error'.
            """
            try:
                payload = json.loads(data_str)
            except (json.JSONDecodeError, ValueError):
                return None
            if not isinstance(payload, dict):
                return None
            err = payload.get("error")
            if not isinstance(err, dict):
                return None
            upstream_message = err.get("message") or "Upstream error"
            code = err.get("code")
            error_type = (
                ERROR_TYPE_MAP.get(code, "api_error")
                if isinstance(code, int)
                else "api_error"
            )
            return error_type, str(upstream_message)

        # Pre-first-chunk errors from the OpenAI generator (e.g. tokenization
        # failure that raises ValueError before any chunk is yielded) would
        # otherwise abort the StreamingResponse with no envelope at all and
        # the client would see a half-open SSE / TCP close. Catch them here
        # and emit a clean Anthropic error sequence instead.
        try:
            stream_iter = openai_stream.__aiter__()
        except Exception as e:
            logger.exception("Failed to open OpenAI stream: %s", e)
            for frame in _flush_on_error("api_error", "Internal server error"):
                yield frame
            return

        # message_start goes out IMMEDIATELY, before the first backend chunk.
        # It used to be withheld until usage was known so the client would
        # never see ``input_tokens=0``; the cost was that a client saw
        # nothing at all for the whole prefill — no message id, no model,
        # no signal the request was even accepted. The real API resolves
        # this the other way round: ship message_start at once with the
        # usage known at that stage and CORRECT the totals in the closing
        # ``message_delta``, which now carries input_tokens as well as
        # output_tokens (see the ``chunk.usage`` handler below).
        for frame in _ensure_message_started(None):
            yield frame

        async def _iter_lines_with_pings() -> AsyncGenerator[tuple[str, Any], None]:
            """Yield ``("line", str)`` per upstream chunk and ``("ping", None)``
            while the backend is silent.

            ``__anext__`` is driven through a TASK so a slow backend can be
            waited on with a timeout without losing the pending read: a bare
            ``asyncio.wait_for`` would cancel the coroutine and drop whatever
            chunk it was about to deliver. The task survives the timeout, a
            ping goes out, and the SAME task is awaited again — no polling,
            no sleep loop, no duplicated or dropped chunks.

            The ``finally`` matters for client disconnects: closing the outer
            generator throws ``GeneratorExit`` at a yield, which unwinds
            through here and cancels the orphaned read instead of leaving it
            pending on the event loop.
            """
            pending: Optional[asyncio.Task] = None
            try:
                while True:
                    if pending is None:
                        pending = asyncio.ensure_future(stream_iter.__anext__())
                    done, _ = await asyncio.wait(
                        {pending}, timeout=PING_INTERVAL_SECONDS
                    )
                    if not done:
                        yield ("ping", None)
                        continue
                    task, pending = pending, None
                    try:
                        line = task.result()
                    except StopAsyncIteration:
                        return
                    yield ("line", line)
            finally:
                if pending is not None and not pending.done():
                    pending.cancel()

        line_iter = _iter_lines_with_pings().__aiter__()

        while True:
            try:
                kind, payload = await line_iter.__anext__()
            except StopAsyncIteration:
                break
            except asyncio.CancelledError:
                raise
            except ValueError as e:
                # _generate_chat_stream re-raises ValueError when its own
                # ``stream_started`` flag is still False — surface as a
                # proper Anthropic error event rather than aborting the
                # StreamingResponse generator.
                logger.warning("OpenAI stream raised before first chunk: %s", e)
                for frame in _flush_on_error(
                    "invalid_request_error", str(e) or "Request failed"
                ):
                    yield frame
                return
            except Exception as e:
                logger.exception("OpenAI stream raised mid-flight: %s", e)
                for frame in _flush_on_error("api_error", "Internal server error"):
                    yield frame
                return

            if kind == "ping":
                # Keep-alive only. Anthropic's own stream interleaves these
                # and SDK clients ignore them; proxies and browsers use them
                # to keep an idle connection from being reaped mid-prefill.
                yield _emit(PingEvent())
                continue

            sse_line = payload
            if not sse_line.startswith("data: "):
                continue

            data_str = sse_line[6:].strip()

            if data_str == "[DONE]":
                for frame in _ensure_message_started(None):
                    yield frame

                # No content AND no finish_reason: the backend dropped the
                # stream silently. Surface as api_error so clients see the
                # failure instead of a fake empty success. If finish_reason
                # IS set we trust the backend's signal — a legitimate empty
                # completion (max_tokens=1 stop, content filter, etc.)
                # deserves a normal message_delta/message_stop pair, not
                # an error that triggers SDK retry loops.
                if not had_content_delta and finish_reason is None:
                    logger.warning(
                        "Stream produced no content and no finish_reason "
                        "before [DONE]; emitting api_error event"
                    )
                    yield _emit(
                        _build_error_event("api_error", "Backend produced no content")
                    )
                    yield _emit(MessageStopEvent())
                    continue

                # Replay anything held back while a tool call was still
                # assembling, THEN close the last open block.
                for event in _release_deferred_content_events():
                    yield _emit(event)

                # Close any open content block
                for event in _close_content_block_events():
                    yield _emit(event)

                # A tool call whose arguments never arrived intact must not
                # be handed to the client as a normal completion: an empty
                # or truncated ``input`` is indistinguishable from a real
                # call, and the client acts on it. Fail visibly instead.
                assembly_failure = _tool_assembly_failure()
                if assembly_failure is not None:
                    logger.warning(
                        "Tool call could not be assembled from the backend stream: %s",
                        assembly_failure,
                    )
                    yield _emit(
                        _build_error_event(
                            "api_error",
                            "Tool call arguments were lost in transport: "
                            f"{assembly_failure}",
                        )
                    )
                    yield _emit(MessageStopEvent())
                    continue

                # Emit message_delta with stop_reason and usage
                effective_finish = finish_reason or "stop"
                if effective_finish not in STOP_REASON_MAP:
                    logger.warning(
                        "Unmapped streaming finish_reason %r; defaulting to end_turn",
                        effective_finish,
                    )
                stop_reason = STOP_REASON_MAP.get(effective_finish, "end_turn")

                # Same stop-sequence upgrade as the non-streaming path.
                matched_sequence: Optional[str] = None
                if effective_finish == "stop":
                    matched_sequence = _resolve_stop_sequence(
                        matched_stop,
                        stop_sequences,
                        "".join(accumulated_text) if accumulated_text else None,
                    )
                    if matched_sequence is not None:
                        stop_reason = "stop_sequence"

                yield _emit(
                    MessageDeltaEvent(
                        delta=AnthropicMessageEndDelta(
                            stop_reason=stop_reason,
                            stop_sequence=matched_sequence,
                        ),
                        usage=final_usage or AnthropicUsage(output_tokens=0),
                    )
                )

                yield _emit(MessageStopEvent())
                continue

            # Parse the OpenAI chunk
            try:
                chunk = ChatCompletionStreamResponse.model_validate_json(data_str)
            except (ValidationError, json.JSONDecodeError, UnicodeDecodeError) as e:
                # First check whether this is the OpenAI handler's
                # streaming error envelope (validator rejects it because
                # it lacks id/choices/created/model). Forwarding the real
                # type/message keeps the failure debuggable instead of
                # collapsing every backend error into "Stream processing
                # error".
                upstream = _parse_upstream_error(data_str)
                if upstream is not None:
                    error_type, error_message = upstream
                    logger.warning(
                        "Forwarding upstream stream error (%s): %s",
                        error_type,
                        error_message,
                    )
                    for frame in _flush_on_error(error_type, error_message):
                        yield frame
                    return

                logger.warning(
                    "Failed to parse Anthropic stream chunk (%s): %s",
                    type(e).__name__,
                    data_str[:200],
                )
                for frame in _flush_on_error("api_error", "Stream processing error"):
                    yield frame
                return

            if chunk.usage is not None:
                # ``include_input=True`` because message_start now ships
                # before usage is known: message_delta is the event that
                # CORRECTS the totals, so it has to carry input_tokens too.
                # The field is optional on AnthropicUsage and clients read
                # the last value they saw, so this is a correction, not a
                # duplicate.
                final_usage = _anthropic_usage_from_openai(
                    chunk.usage,
                    include_input=True,
                    include_output=True,
                )

            # Usage-only chunk (empty choices with usage info)
            if not chunk.choices and chunk.usage:
                continue

            if not chunk.choices:
                continue

            choice = chunk.choices[0]

            # Capture finish_reason on this chunk but DO NOT short-circuit:
            # some OpenAI-compatible backends pack the final content token
            # (or last tool-args fragment) into the same chunk as
            # finish_reason. Skipping delta processing would silently drop
            # that payload — sometimes the whole completion if it was a
            # one-token reply. Fall through to the delta handlers below.
            if choice.finish_reason is not None:
                finish_reason = choice.finish_reason
            # The backend reports WHICH stop string ended the completion out
            # of band, on the same chunk as the finish reason.
            if getattr(choice, "matched_stop", None) is not None:
                matched_stop = choice.matched_stop

            delta = choice.delta

            has_delta_payload = bool(
                delta.reasoning_content
                or delta.tool_calls
                or (delta.content is not None and delta.content != "")
                or chunk.usage
            )

            if (
                not has_delta_payload
                and delta.role == "assistant"
                and (delta.content is None or delta.content == "")
            ):
                continue

            # Handle reasoning content deltas
            if delta.reasoning_content:
                if _tool_block_awaits_arguments():
                    deferred_content.append(("thinking", delta.reasoning_content))
                else:
                    for event in _release_deferred_content_events():
                        yield _emit(event)
                    for event in _ensure_content_block_events(
                        "thinking",
                        ThinkingBlock(thinking=""),
                    ):
                        yield _emit(event)

                    yield _emit(
                        ContentBlockDeltaEvent(
                            index=content_block_index,
                            delta=ThinkingDelta(thinking=delta.reasoning_content),
                        )
                    )
                had_content_delta = True

            # Handle tool call deltas
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    tc_func = tc.function
                    tc_name = (tc_func.name or None) if tc_func else None
                    tc_args = (tc_func.arguments or "") if tc_func else ""

                    # The backend index is the only reliable identity for a
                    # call across deltas: ``id`` and ``name`` may be absent
                    # on continuations. When it is absent too, continue the
                    # call currently open rather than guessing a new one.
                    tc_index = tc.index
                    if tc_index is None:
                        tc_index = open_tool_index if open_tool_index is not None else 0

                    if tc.id is not None:
                        tool_ids.setdefault(tc_index, tc.id)

                    # A name is what makes a call addressable, so the block
                    # opens on the FIRST name for this index — not on the
                    # first delta, which may well be name-less. Deltas that
                    # merely repeat a known name are continuations.
                    if tc_name is not None and tool_names.get(tc_index) is None:
                        tool_names[tc_index] = tc_name
                        for event in _ensure_content_block_events(
                            "tool_use",
                            ToolUseBlock(
                                id=_anthropic_tool_use_id(tool_ids.get(tc_index)),
                                name=tc_name,
                                input={},
                            ),
                            force_new=True,
                        ):
                            yield _emit(event)
                        open_tool_index = tc_index
                        # A zero-argument tool call may never emit an
                        # input_json_delta; the tool_use start block itself is
                        # still meaningful content because it carries id/name.
                        had_content_delta = True

                        # Fragments that arrived before the name are flushed
                        # now, in arrival order, ahead of this delta's own.
                        for fragment in pending_tool_args.pop(tc_index, []):
                            yield _emit(
                                ContentBlockDeltaEvent(
                                    index=content_block_index,
                                    delta=InputJsonDelta(partial_json=fragment),
                                )
                            )
                            had_content_delta = True

                    if tc_args:
                        tool_args.setdefault(tc_index, []).append(tc_args)
                        owns_open_block = (
                            content_block_open
                            and content_block_type == "tool_use"
                            and open_tool_index == tc_index
                            and tool_names.get(tc_index) is not None
                        )
                        if owns_open_block:
                            yield _emit(
                                ContentBlockDeltaEvent(
                                    index=content_block_index,
                                    delta=InputJsonDelta(partial_json=tc_args),
                                )
                            )
                            had_content_delta = True
                        else:
                            # No live block owns this index — the name has
                            # not arrived, or the fragment belongs to a
                            # different call than the open one. Buffering is
                            # the only non-lossy option: appending it to
                            # whatever block happens to be open would
                            # corrupt that call's JSON with another call's
                            # arguments.
                            pending_tool_args.setdefault(tc_index, []).append(tc_args)

            # Handle text content deltas
            if delta.content is not None and delta.content != "":
                if _tool_block_awaits_arguments():
                    deferred_content.append(("text", delta.content))
                else:
                    for event in _release_deferred_content_events():
                        yield _emit(event)
                    for event in _ensure_content_block_events(
                        "text",
                        TextBlock(text=""),
                    ):
                        yield _emit(event)

                    yield _emit(
                        ContentBlockDeltaEvent(
                            index=content_block_index,
                            delta=TextDelta(text=delta.content),
                        )
                    )
                had_content_delta = True
                # Kept only to feed the stop-sequence suffix fallback in
                # ``_resolve_stop_sequence``; never re-emitted.
                if stop_sequences:
                    accumulated_text.append(delta.content)

    def _convert_response(
        self,
        response: ChatCompletionResponse,
        stop_sequences: Optional[list[str]] = None,
    ) -> AnthropicMessagesResponse:
        """Convert an OpenAI ChatCompletionResponse to an Anthropic Messages response."""
        if not response.choices:
            return AnthropicMessagesResponse(
                content=[TextBlock(text="")],
                model=response.model,
                stop_reason="end_turn",
                usage=AnthropicUsage(input_tokens=0, output_tokens=0),
            )

        choice = response.choices[0]
        content: list[AnthropicContentBlock] = []

        # Add reasoning content as a thinking block. signature is omitted
        # entirely when the backend doesn't provide one — empty strings
        # would fail downstream Anthropic signature verifiers.
        if choice.message.reasoning_content:
            content.append(ThinkingBlock(thinking=choice.message.reasoning_content))

        # Add text content
        if choice.message.content:
            content.append(TextBlock(text=choice.message.content))

        # Add tool calls
        if choice.message.tool_calls:
            for tool_call in choice.message.tool_calls:
                raw_args = tool_call.function.arguments
                if not tool_call.function.name:
                    raise ToolCallAssemblyError(
                        "backend returned a tool call with no function name "
                        f"({len(raw_args or '')} characters of arguments)"
                    )
                if raw_args is None or not str(raw_args).strip():
                    # A tool with no parameters legitimately serialises to
                    # nothing at all; that is not a failure.
                    tool_input: dict = {}
                else:
                    try:
                        tool_input = json.loads(raw_args)
                    except (json.JSONDecodeError, TypeError) as e:
                        # Substituting ``{}`` here used to make a mangled
                        # tool call look exactly like a valid zero-argument
                        # one. The caller cannot tell the difference, so the
                        # arguments have to fail loudly instead.
                        raise ToolCallAssemblyError(
                            f"tool {tool_call.function.name!r} returned "
                            f"arguments that are not valid JSON "
                            f"({type(e).__name__}): "
                            f"{str(raw_args)[:200]!r}"
                        ) from e
                if not isinstance(tool_input, dict):
                    raise ToolCallAssemblyError(
                        f"tool {tool_call.function.name!r} returned "
                        f"non-object arguments of type "
                        f"{type(tool_input).__name__}"
                    )

                content.append(
                    ToolUseBlock(
                        id=_anthropic_tool_use_id(tool_call.id),
                        name=tool_call.function.name,
                        input=tool_input,
                    )
                )

        # Map stop reason
        finish_reason = choice.finish_reason or "stop"
        if finish_reason not in STOP_REASON_MAP:
            logger.warning(
                "Unmapped OpenAI finish_reason %r; defaulting to end_turn",
                finish_reason,
            )
        stop_reason = STOP_REASON_MAP.get(finish_reason, "end_turn")

        # A caller-supplied stop sequence upgrades ``end_turn`` to the
        # ``stop_sequence`` reason and fills the ``stop_sequence`` field.
        # Only the natural-stop case can be a stop sequence: ``length`` and
        # ``tool_calls`` ended the completion for a different reason.
        matched_sequence: Optional[str] = None
        if finish_reason == "stop":
            matched_sequence = _resolve_stop_sequence(
                getattr(choice, "matched_stop", None),
                stop_sequences,
                choice.message.content,
            )
            if matched_sequence is not None:
                stop_reason = "stop_sequence"

        # Anthropic requires ``content`` to contain at least one block.
        # Empty string completions (max_tokens=1 stop, content filter, etc.)
        # would otherwise ship ``content=[]`` and break strict SDK parsers.
        if not content:
            content.append(TextBlock(text=""))

        return AnthropicMessagesResponse(
            id=f"msg_{uuid.uuid4().hex}",
            content=content,
            model=response.model,
            stop_reason=stop_reason,
            stop_sequence=matched_sequence,
            usage=_anthropic_usage_from_openai(
                response.usage,
                include_input=True,
                include_output=True,
            ),
        )

    def _convert_openai_error_response(self, response) -> JSONResponse:
        """Forward an upstream OpenAI-handler error as an Anthropic error.

        The original error message is preserved for 4xx (after light
        sanitization) so callers see the real validation failure. For 5xx
        we always return a generic ``"Internal server error"`` to avoid
        leaking ``str(e)`` payloads that the OpenAI handler builds from
        raw exceptions (paths, tracebacks, prompt fragments, etc.).
        """
        status_code = getattr(response, "status_code", 500)
        body = getattr(response, "body", b"") or b""
        error_type = ERROR_TYPE_MAP.get(status_code, "api_error")

        upstream_message: Optional[str] = None
        try:
            payload = json.loads(body.decode("utf-8")) if body else None
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Non-JSON body (HTML gateway error, plain text, ...). Use a
            # bounded slice of the raw body so the client still has a
            # useful hint instead of a generic placeholder.
            try:
                upstream_message = body.decode("utf-8", errors="replace")[:500]
            except Exception:
                upstream_message = None
        else:
            if isinstance(payload, dict):
                error_payload = payload.get("error", payload)
                if isinstance(error_payload, dict):
                    upstream_message = error_payload.get("message") or payload.get(
                        "message"
                    )
                    # Honor the upstream error.type only for 4xx; 5xx is
                    # normalized below.
                    if status_code < 500:
                        upstream_type = error_payload.get("type")
                        if isinstance(upstream_type, str) and upstream_type:
                            error_type = upstream_type
                elif isinstance(error_payload, str):
                    upstream_message = error_payload
                elif isinstance(payload.get("message"), str):
                    upstream_message = payload["message"]

        message = _scrub_error_message(upstream_message or "", status_code)
        return self._error_response(
            status_code=status_code,
            error_type=error_type,
            message=message,
        )

    def _error_response(
        self,
        status_code: int,
        error_type: str,
        message: str,
        exception_name: Optional[str] = None,
    ) -> JSONResponse:
        """Create an Anthropic-format error response.

        ``error.type`` is restricted to Anthropic's documented enum so strict
        SDK clients (anthropic-sdk-python / -typescript) keep parsing the
        response into their typed error classes. ``exception_name`` — when
        provided — is logged at WARNING level so operators can still grep
        server-side, but it never reaches the wire.
        """
        if exception_name:
            logger.warning(
                "Anthropic error response %s (exception=%s): %s",
                error_type,
                exception_name,
                message,
            )
        error_resp = AnthropicErrorResponse(
            error=AnthropicError(type=error_type, message=message)
        )
        return JSONResponse(
            status_code=status_code,
            content=error_resp.model_dump(),
        )

    async def handle_count_tokens(
        self,
        request: AnthropicCountTokensRequest,
        raw_request: Request,
    ) -> JSONResponse:
        """Handle /v1/messages/count_tokens endpoint.

        Converts the request to a ChatCompletionRequest, applies the chat
        template via the OpenAI handler to tokenize, and returns the count.
        """
        try:
            # Build a minimal AnthropicMessagesRequest so we can reuse conversion
            messages_request = AnthropicMessagesRequest(
                model=request.model,
                messages=request.messages,
                max_tokens=1,  # dummy, not used for counting
                system=request.system,
                thinking=request.thinking,
                tools=request.tools,
                tool_choice=request.tool_choice,
            )
            chat_request = self._convert_to_chat_completion_request(messages_request)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception("Error converting count_tokens request: %s", e)
            return self._error_response(
                status_code=400,
                error_type="invalid_request_error",
                message=str(e),
            )

        try:
            is_multimodal = (
                self.openai_serving_chat.tokenizer_manager.model_config.is_multimodal
            )
            processed = self.openai_serving_chat._process_messages(
                chat_request, is_multimodal
            )

            if isinstance(processed.prompt_ids, list):
                input_tokens = len(processed.prompt_ids)
            else:
                # prompt_ids is a string (multimodal case) — tokenize it
                tokenizer = self.openai_serving_chat.tokenizer_manager.tokenizer
                input_tokens = len(tokenizer.encode(processed.prompt_ids))

            return JSONResponse(
                content=AnthropicCountTokensResponse(
                    input_tokens=input_tokens
                ).model_dump()
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception("Error counting tokens: %s", e)
            return self._error_response(
                status_code=500,
                error_type="api_error",
                message="Internal server error",
                exception_name=type(e).__name__,
            )
