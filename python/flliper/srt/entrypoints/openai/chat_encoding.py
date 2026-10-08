"""Single home for the chat-encoding dispatch.

Which encoder turns chat messages into prompt tokens is a property of the
model, so the serving path and offline tools (benchmarks, evals) must resolve
it here instead of re-deriving it from model architectures themselves.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def resolve_chat_encoding_spec(
    *,
    hf_config: Any,
    tokenizer: Any,
    tool_call_parser: Optional[str] = None,
) -> Optional[str]:
    """Return the chat encoding spec for a model: "dsv4", "dsv32", or None.

    None means the default path (HF chat template); any non-None spec also owns
    reasoning-history rendering (:func:`spec_owns_reasoning_history`).
    """
    if tool_call_parser == "deepseekv4":
        return "dsv4"
    if tool_call_parser == "deepseekv32":
        return "dsv32"

    architectures = hf_config.architectures
    arch = architectures[0] if architectures else ""

    if "DeepseekV4" in arch:
        return "dsv4"

    has_chat_template = tokenizer is not None and tokenizer.chat_template is not None
    if "DeepseekV3" in arch and not has_chat_template:
        return "dsv32"
    return None


def spec_owns_reasoning_history(spec: Optional[str]) -> bool:
    """Whether the encoder for ``spec`` renders assistant reasoning history itself.

    Custom encoders frame the reasoning and content channels, so history must be
    passed as assistant ``reasoning_content``. Splicing a detector's markers into
    content instead nests a reasoning block inside the content channel and leaves
    the real one empty, teaching the model to emit raw markers as visible text.

    Answered for the whole family rather than a list of specs, so a new spec gets
    the safe default: worst case is dropped history, not a leak.
    """
    return spec is not None


def template_owns_reasoning_history(chat_template: Any) -> bool:
    """Whether an HF chat template renders assistant ``reasoning_content`` itself.

    Such a template (e.g. Qwen3.x: ``<think>`` + ``reasoning_content`` +
    ``</think>`` + content, kept across turns unless ``preserve_thinking`` is
    false) must see prior-turn thinking exactly as an OpenAI chat request with
    ``reasoning_content`` would hand it over. Splicing re-wrapped markers into
    content instead makes the template render an empty think block followed by
    the wrapped text, so the follow-up prompt no longer matches the tokens the
    model generated and the prefix breaks at the previous turn.

    Decided from the template source: a template that never references
    ``reasoning_content`` cannot render it, and keeps the wrap-into-content
    path. A dict of named templates only qualifies when every entry does.
    """
    if isinstance(chat_template, str):
        return "reasoning_content" in chat_template
    if isinstance(chat_template, dict) and chat_template:
        return all(
            isinstance(t, str) and "reasoning_content" in t
            for t in chat_template.values()
        )
    return False


def encode_simple_chat(
    *,
    tokenizer: Any,
    spec: Optional[str],
    messages: List[Dict[str, Any]],
    thinking_mode: str = "chat",
) -> List[int]:
    """Encode a plain-text chat conversation into prompt token ids.

    Minimal encode for offline tools: no tools, no multimodal content, no
    continue_final_message; the serving path keeps its full request-level
    pipeline in ``serving_chat``. Like
    ``serving_chat``, an empty system message is prepended when the
    conversation does not start with one (for the dsv4/dsv32 encoders this
    currently renders to zero tokens, but keeping the insertion explicit ties
    this helper to the serving semantics rather than to that coincidence).
    """
    if spec in ("dsv4", "dsv32"):
        if messages and messages[0]["role"] != "system":
            messages = [{"role": "system", "content": ""}] + list(messages)
        if spec == "dsv4":
            from flliper.srt.entrypoints.openai import encoding_dsv4

            real_input = encoding_dsv4.encode_messages(
                messages, thinking_mode=thinking_mode
            )
        else:
            from flliper.srt.entrypoints.openai import encoding_dsv32

            real_input = encoding_dsv32.encode_messages(
                messages, thinking_mode=thinking_mode
            )
        return tokenizer.encode(real_input)

    if getattr(tokenizer, "chat_template", None) is None:
        raise ValueError(
            "This model has no HF chat template and no custom chat encoder; "
            f"cannot encode chat messages with {getattr(tokenizer, 'name_or_path', tokenizer)!r}."
        )
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True
    )
