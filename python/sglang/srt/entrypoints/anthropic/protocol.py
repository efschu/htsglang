"""Pydantic models for Anthropic Messages API protocol.

Mirrors the shape of the official Anthropic Python SDK
(``anthropic-sdk-python``): ``ContentBlock``, ``Tool``, ``MessageStreamEvent``
and ``ContentBlockDelta`` are discriminated unions over the ``type`` field,
so each variant carries only the fields it actually uses.
"""

import uuid
from typing import Annotated, Any, Literal, Optional, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    NonNegativeInt,
    Tag,
    field_validator,
    model_validator,
)


class AnthropicError(BaseModel):
    """Error structure for Anthropic API."""

    type: str
    message: str


class AnthropicErrorResponse(BaseModel):
    """Error response structure for Anthropic API."""

    type: Literal["error"] = "error"
    error: AnthropicError


class AnthropicUsage(BaseModel):
    """Token usage information.

    ``input_tokens``/``output_tokens`` are ``Optional`` because Anthropic's
    streaming ``message_delta`` event omits ``input_tokens`` (the spec
    requires it only on ``message_start``). Non-streaming responses set both.
    """

    input_tokens: Optional[NonNegativeInt] = None
    output_tokens: Optional[NonNegativeInt] = None
    cache_creation_input_tokens: Optional[NonNegativeInt] = None
    cache_read_input_tokens: Optional[NonNegativeInt] = None


# ---------- Content blocks (discriminated by ``type``) ----------


class TextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ImageBlock(BaseModel):
    type: Literal["image"] = "image"
    # Kept loosely typed for compat with both base64 and URL sources; the
    # serving layer normalises to OpenAI ``image_url`` parts.
    source: Optional[Union[dict[str, Any], str]] = None


class ToolUseBlock(BaseModel):
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any] = Field(default_factory=dict)


class ToolResultBlock(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: Optional[str] = None
    # Some legacy payloads use ``id`` instead of ``tool_use_id``.
    id: Optional[str] = None
    content: Optional[Union[str, list["AnthropicContentBlock"]]] = None
    is_error: Optional[bool] = None


class ToolReferenceBlock(BaseModel):
    """sglang extension: references a deferred-loaded tool by name."""

    type: Literal["tool_reference"] = "tool_reference"
    name: Optional[str] = None
    # Anthropic-style payloads sometimes use ``tool_name``; accept both.
    tool_name: Optional[str] = None
    id: Optional[str] = None


class SearchResultBlock(BaseModel):
    type: Literal["search_result"] = "search_result"
    # ``source`` here is a URL/identifier string (unlike ImageBlock.source).
    source: Optional[Union[str, dict[str, Any]]] = None
    title: Optional[str] = None
    content: Optional[list[dict[str, Any]]] = None


class ThinkingBlock(BaseModel):
    type: Literal["thinking"] = "thinking"
    thinking: str
    signature: Optional[str] = None


class RedactedThinkingBlock(BaseModel):
    type: Literal["redacted_thinking"] = "redacted_thinking"
    data: Optional[str] = None


class UnknownContentBlock(BaseModel):
    """Forward-compatible fallback for content-block types we do not model.

    Anthropic keeps adding block types (``document``, ``mcp_tool_use``,
    ``server_tool_use``, ``web_search_tool_result``, ...). A CLOSED
    discriminated union answers 400 for the whole request on the first
    unknown tag, which kills an entire conversation — including all the
    turns the server could have served — over one block it merely does not
    understand. A forward-compatible front must degrade PER BLOCK, not per
    conversation, so unknown tags land here verbatim (``extra="allow"``
    keeps the original fields for logging) and the serving layer skips them
    with one warning naming the type.

    This is deliberately NOT a catch-all for malformed KNOWN blocks: the
    discriminator below routes any known tag to its precise model, so a
    ``{"type":"text"}`` without ``text`` still produces the exact Pydantic
    validation error instead of silently degrading to "unknown".
    """

    model_config = ConfigDict(extra="allow")

    type: str


# The eight tags this front models precisely. Anything else is routed to
# UnknownContentBlock by ``_content_block_discriminator``.
KNOWN_CONTENT_BLOCK_TAGS = frozenset(
    {
        "text",
        "image",
        "tool_use",
        "tool_result",
        "tool_reference",
        "search_result",
        "thinking",
        "redacted_thinking",
    }
)


def _content_block_discriminator(v) -> str:
    """Route a content block to its variant tag, unknown types included.

    Pydantic's ``Field(discriminator=...)`` cannot express a catch-all
    member — an unlisted tag is a validation error by construction. A
    callable ``Discriminator`` can, because it maps the raw value to a tag
    we control, so every unmodelled ``type`` collapses onto the single
    ``"unknown"`` member. Same mechanism as ``_tool_discriminator`` above.
    """
    if isinstance(v, dict):
        t = v.get("type")
    else:
        t = getattr(v, "type", None)
    if isinstance(t, str) and t in KNOWN_CONTENT_BLOCK_TAGS:
        return t
    return "unknown"


AnthropicContentBlock = Annotated[
    Union[
        Annotated[TextBlock, Tag("text")],
        Annotated[ImageBlock, Tag("image")],
        Annotated[ToolUseBlock, Tag("tool_use")],
        Annotated[ToolResultBlock, Tag("tool_result")],
        Annotated[ToolReferenceBlock, Tag("tool_reference")],
        Annotated[SearchResultBlock, Tag("search_result")],
        Annotated[ThinkingBlock, Tag("thinking")],
        Annotated[RedactedThinkingBlock, Tag("redacted_thinking")],
        Annotated[UnknownContentBlock, Tag("unknown")],
    ],
    Discriminator(_content_block_discriminator),
]


class AnthropicMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: Union[str, list[AnthropicContentBlock]]


# ---------- Tools (discriminated by ``type`` family) ----------


class AnthropicCustomTool(BaseModel):
    """Custom tool defined by the API user — requires ``input_schema``."""

    type: Optional[Literal["custom"]] = None  # absent or explicit "custom"
    name: str
    description: Optional[str] = None
    input_schema: dict[str, Any]
    defer_loading: Optional[bool] = None

    @field_validator("input_schema")
    @classmethod
    def _ensure_object_schema(cls, v):
        if not isinstance(v, dict):
            raise ValueError("input_schema must be a dictionary")
        if "type" not in v:
            v["type"] = "object"
        return v


class AnthropicWebSearchTool(BaseModel):
    """Anthropic ``web_search_*`` server tool family.

    No client-side ``input_schema`` — Anthropic provides the backing
    search implementation. Tag format is ``web_search_YYYYMMDD``.
    """

    type: str = Field(pattern=r"^web_search_\d{8}$")
    name: Literal["web_search"] = "web_search"
    description: Optional[str] = None
    defer_loading: Optional[bool] = None
    max_uses: Optional[int] = None
    allowed_domains: Optional[list[str]] = None
    blocked_domains: Optional[list[str]] = None


class AnthropicComputerTool(BaseModel):
    """Anthropic ``computer_*`` server tool family."""

    type: str = Field(pattern=r"^computer_\d{8}$")
    name: Literal["computer"] = "computer"
    description: Optional[str] = None
    defer_loading: Optional[bool] = None
    display_width_px: Optional[int] = None
    display_height_px: Optional[int] = None
    display_number: Optional[int] = None


class AnthropicBashTool(BaseModel):
    """Anthropic ``bash_*`` server tool family."""

    type: str = Field(pattern=r"^bash_\d{8}$")
    name: Literal["bash"] = "bash"
    description: Optional[str] = None
    defer_loading: Optional[bool] = None


class AnthropicTextEditorTool(BaseModel):
    """Anthropic ``text_editor_*`` server tool family."""

    type: str = Field(pattern=r"^text_editor_\d{8}$")
    name: Literal["str_replace_editor", "str_replace_based_edit_tool"]
    description: Optional[str] = None
    defer_loading: Optional[bool] = None


def _tool_discriminator(v) -> str:
    """Pick the right tool variant from a dict or model instance.

    Pydantic discriminators don't accept ``None`` as a tag, and custom
    tools allow ``type`` to be absent. Map missing/``custom`` to
    ``"custom"`` and prefix-match server-tool families.
    """
    if isinstance(v, dict):
        t = v.get("type")
    else:
        t = getattr(v, "type", None)
    if not t or t == "custom":
        return "custom"
    if t.startswith("web_search_"):
        return "web_search"
    if t.startswith("computer_"):
        return "computer"
    if t.startswith("bash_"):
        return "bash"
    if t.startswith("text_editor_"):
        return "text_editor"
    return "custom"


AnthropicTool = Annotated[
    Union[
        Annotated[AnthropicCustomTool, Tag("custom")],
        Annotated[AnthropicWebSearchTool, Tag("web_search")],
        Annotated[AnthropicComputerTool, Tag("computer")],
        Annotated[AnthropicBashTool, Tag("bash")],
        Annotated[AnthropicTextEditorTool, Tag("text_editor")],
    ],
    Discriminator(_tool_discriminator),
]


def is_server_tool(tool) -> bool:
    """Return True for Anthropic built-in server-side tools."""
    return isinstance(
        tool,
        (
            AnthropicWebSearchTool,
            AnthropicComputerTool,
            AnthropicBashTool,
            AnthropicTextEditorTool,
        ),
    )


class AnthropicToolChoice(BaseModel):
    """Tool choice strategy."""

    type: Literal["auto", "any", "tool", "none"]
    name: Optional[str] = None


class AnthropicThinkingParam(BaseModel):
    """Anthropic extended-thinking control on the request.

    Mirrors the Anthropic SDK's ``ThinkingConfigParam`` discriminated
    union of three variants — see ``anthropic-sdk-python``'s
    ``thinking_config_{enabled,disabled,adaptive}_param.py``:

    * ``enabled`` requires ``budget_tokens`` (≥1024) and accepts
      ``display``.
    * ``disabled`` accepts no other fields.
    * ``adaptive`` (Claude 4.7) accepts ``display`` but not
      ``budget_tokens``.

    The serving layer treats ``adaptive`` identically to ``enabled``
    because the local OpenAI-compatible backend has no auto-throttle
    equivalent. That is also why ``adaptive`` carries no budget: the SDK
    forbids ``budget_tokens`` there, and enabled-without-budget is the
    correct local semantics — reasoning runs to its natural end.

    ``budget_tokens`` on ``enabled`` IS enforced when the model has a
    reasoning parser configured: it maps onto the OpenAI front's
    ``thinking_budget``, which caps the thinking section at that many
    tokens (#540). On a model without a reasoning parser there are no
    markers to cap, so the value is accept-and-log as before.
    ``display="omitted"`` is accepted but cannot suppress reasoning
    mid-stream and is logged.
    """

    type: Literal["enabled", "disabled", "adaptive"]
    budget_tokens: Optional[int] = None
    display: Optional[Literal["summarized", "omitted"]] = None

    @model_validator(mode="after")
    def _validate_thinking_shape(self):
        # Cross-field rules mirror the SDK's three discriminated variants.
        if self.type == "enabled":
            if self.budget_tokens is None:
                raise ValueError(
                    "thinking.budget_tokens is required when thinking.type is 'enabled'"
                )
            if self.budget_tokens < 1024:
                raise ValueError(
                    "thinking.budget_tokens must be >= 1024 (got {})".format(
                        self.budget_tokens
                    )
                )
        elif self.type == "disabled":
            if self.budget_tokens is not None:
                raise ValueError(
                    "thinking.budget_tokens is not allowed when "
                    "thinking.type is 'disabled'"
                )
            if self.display is not None:
                raise ValueError(
                    "thinking.display is not allowed when thinking.type is 'disabled'"
                )
        elif self.type == "adaptive":
            if self.budget_tokens is not None:
                raise ValueError(
                    "thinking.budget_tokens is not allowed when "
                    "thinking.type is 'adaptive'"
                )
        return self


class AnthropicTaskBudget(BaseModel):
    """Claude 4.7 ``output_config.task_budget`` — soft hint, not a hard cap.

    Mirrors ``BetaTokenTaskBudgetParam`` in the Anthropic SDK: ``total``
    and ``type`` are required; ``remaining`` is the client-tracked
    countdown used for compaction. The hard cap on generation is still
    ``max_tokens``; we never enforce ``task_budget`` ourselves.
    """

    type: Literal["tokens"]
    total: int = Field(gt=0)
    remaining: Optional[int] = Field(default=None, ge=0)


class AnthropicOutputConfig(BaseModel):
    """Claude 4.7 ``output_config`` block.

    ``effort`` maps to the OpenAI ``reasoning_effort`` knob and is passed
    through VERBATIM, ``xhigh`` included -- the sglang schema carries
    ``xhigh`` for exactly this reason. It used to collapse ``xhigh`` onto
    ``max``, which was backwards for the Qwen3.8 family: its chat template
    accepts only ``('xhigh', 'medium', 'low')`` and raises otherwise, so the
    rewrite turned a supported request into an HTTP 500. Set
    ``SGLANG_ANTHROPIC_XHIGH_EFFORT=max`` to restore the collapse for a
    deployment whose template names its top tier ``max``.
    ``task_budget`` is propagated as a custom-param hint.
    """

    effort: Optional[Literal["low", "medium", "high", "xhigh", "max"]] = None
    task_budget: Optional[AnthropicTaskBudget] = None


class AnthropicCountTokensRequest(BaseModel):
    """Anthropic count_tokens API request."""

    model: str
    messages: list[AnthropicMessage]
    system: Optional[Union[str, list[AnthropicContentBlock]]] = None
    thinking: Optional[AnthropicThinkingParam] = None
    tool_choice: Optional[AnthropicToolChoice] = None
    tools: Optional[list[AnthropicTool]] = None
    # Claude 4.7 / SDK-compatibility fields. Accepted but no-op on count.
    output_config: Optional[AnthropicOutputConfig] = None
    betas: Optional[list[str]] = None


class AnthropicCountTokensResponse(BaseModel):
    """Anthropic count_tokens API response."""

    input_tokens: int


class AnthropicMessagesRequest(BaseModel):
    """Anthropic Messages API request."""

    model: str
    messages: list[AnthropicMessage]
    max_tokens: int
    metadata: Optional[dict[str, Any]] = None
    stop_sequences: Optional[list[str]] = None
    stream: Optional[bool] = False
    system: Optional[Union[str, list[AnthropicContentBlock]]] = None
    temperature: Optional[float] = None
    thinking: Optional[AnthropicThinkingParam] = None
    tool_choice: Optional[AnthropicToolChoice] = None
    tools: Optional[list[AnthropicTool]] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    # Claude 4.7 fields. The Anthropic SDK / Claude Code attach these even
    # when targeting non-Anthropic backends, so the schema must accept them.
    output_config: Optional[AnthropicOutputConfig] = None
    betas: Optional[list[str]] = None
    #: #557: per-request chat-template arguments, passed through to the
    #: template layer.
    #:
    #: DECLARED, because undeclared is not neutral here. This model sets no
    #: ``model_config``, so pydantic's default ``extra="ignore"`` applies and
    #: an undeclared field is SILENTLY DROPPED -- the caller's kwargs never
    #: reach the template and nothing says so. Rejecting would at least be
    #: visible; dropping is not.
    #:
    #: Semantics are the OpenAI front's, not new ones: the adapter converts
    #: into a ``ChatCompletionRequest`` (which has carried
    #: ``chat_template_kwargs`` all along) and delegates. Note that
    #: ``OpenAIServingChat.apply_reasoning_enabled`` MERGES into this dict and
    #: overrides only the reasoning toggle, so unrelated keys survive while
    #: Anthropic's typed ``thinking`` field stays authoritative over the
    #: toggle itself.
    chat_template_kwargs: Optional[dict[str, Any]] = None

    @field_validator("model")
    @classmethod
    def _validate_model(cls, v):
        if not v:
            raise ValueError("Model is required")
        return v

    @field_validator("max_tokens")
    @classmethod
    def _validate_max_tokens(cls, v):
        if v <= 0:
            raise ValueError("max_tokens must be positive")
        return v


# ---------- Stream deltas ----------
# Content-block deltas (discriminated by ``type``) vs message-end delta
# (separate model; the wire format does not put ``type`` inside its payload).


class TextDelta(BaseModel):
    type: Literal["text_delta"] = "text_delta"
    text: str


class InputJsonDelta(BaseModel):
    type: Literal["input_json_delta"] = "input_json_delta"
    partial_json: str


class ThinkingDelta(BaseModel):
    type: Literal["thinking_delta"] = "thinking_delta"
    thinking: str


class SignatureDelta(BaseModel):
    type: Literal["signature_delta"] = "signature_delta"
    signature: str


AnthropicContentDelta = Annotated[
    Union[TextDelta, InputJsonDelta, ThinkingDelta, SignatureDelta],
    Field(discriminator="type"),
]


class AnthropicMessageEndDelta(BaseModel):
    """Delta carried on ``message_delta`` events.

    Anthropic's protocol does NOT put a ``type`` field inside this delta
    payload — the SSE ``event:`` header already says ``message_delta``.
    Stop reason and stop sequence are the only fields.
    """

    stop_reason: Optional[
        Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]
    ] = None
    stop_sequence: Optional[str] = None


# ---------- Stream events (discriminated by ``type``) ----------


class MessageStartEvent(BaseModel):
    type: Literal["message_start"] = "message_start"
    message: "AnthropicMessagesResponse"


class MessageDeltaEvent(BaseModel):
    type: Literal["message_delta"] = "message_delta"
    delta: AnthropicMessageEndDelta
    usage: AnthropicUsage


class MessageStopEvent(BaseModel):
    type: Literal["message_stop"] = "message_stop"


class ContentBlockStartEvent(BaseModel):
    type: Literal["content_block_start"] = "content_block_start"
    index: int
    content_block: AnthropicContentBlock


class ContentBlockDeltaEvent(BaseModel):
    type: Literal["content_block_delta"] = "content_block_delta"
    index: int
    delta: AnthropicContentDelta


class ContentBlockStopEvent(BaseModel):
    type: Literal["content_block_stop"] = "content_block_stop"
    index: int


class PingEvent(BaseModel):
    type: Literal["ping"] = "ping"


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    error: AnthropicError


AnthropicStreamEvent = Annotated[
    Union[
        MessageStartEvent,
        MessageDeltaEvent,
        MessageStopEvent,
        ContentBlockStartEvent,
        ContentBlockDeltaEvent,
        ContentBlockStopEvent,
        PingEvent,
        ErrorEvent,
    ],
    Field(discriminator="type"),
]


class AnthropicMessagesResponse(BaseModel):
    """Anthropic Messages API response."""

    id: str = Field(default_factory=lambda: f"msg_{uuid.uuid4().hex}")
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: list[AnthropicContentBlock]
    model: str
    stop_reason: Optional[
        Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]
    ] = None
    stop_sequence: Optional[str] = None
    usage: Optional[AnthropicUsage] = None


# Resolve forward references for nested types.
ToolResultBlock.model_rebuild()
MessageStartEvent.model_rebuild()
