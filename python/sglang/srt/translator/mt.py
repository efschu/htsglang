# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Translation through our own OpenAI-compatible endpoint.

This is the dogfood stage: the translation runs on the htsglang server the
rest of the project exists to build, over the same ``/v1/chat/completions``
surface any third-party client would use (client-compat principle -- we do
not reach into the engine, we call it like a stranger would).

Three things make an LLM a usable MT engine here rather than a chatbot that
sometimes translates:

* **A prompt that forbids conversation.** The model is told, per direction,
  that its entire output is the translation. Anything else (a preamble, a
  clarifying question, a note about ambiguity) arrives as speech in the
  listener's ear and is worse than a mediocre translation.
* **Direction as a parameter.** The source and target codes are substituted
  into the prompt from the language matrix. There is no DE/ES string anywhere
  in this module; the falsifier test asserts that by running ``ja->fr``.
* **Sentence-level streaming.** The endpoint streams tokens; TTS needs whole
  clauses. :class:`SentenceAccumulator` regroups the token stream into
  synthesizable units so the first audio can start while the tail of a long
  sentence is still being generated. For short conversational turns this
  saves little; for a 20-word turn it removes most of the MT stage from the
  critical path.

Conversation context is carried as a short rolling history of previous turns.
It measurably helps pronoun and formality choice ("you" -> tu/usted, and
German gender agreement on a referent named two turns ago), which is exactly
where a context-free sentence translator embarrasses itself in a live
exchange. The history is bounded because it is also the only part of this
stage whose cost grows.
"""

from __future__ import annotations

import dataclasses
import json
import re
from collections import deque
from typing import AsyncIterator, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import httpx

from sglang.srt.translator.backends import BackendError
from sglang.srt.translator.languages import display_name

__all__ = [
    "MtConfig",
    "OpenAiMt",
    "SentenceAccumulator",
    "build_prompt",
]


# Sentence-ish boundary: terminal punctuation followed by whitespace, plus the
# inverted marks Spanish opens with (which must NOT be treated as terminators)
# handled by requiring the mark to be preceded by a word character.
_BOUNDARY = re.compile(r"(?<=[\w\"'”»)\]])\s*([.!?…]+)(?=\s|$)")


@dataclasses.dataclass(frozen=True)
class MtConfig:
    """Where the LLM is and how it is asked."""

    base_url: str = "http://127.0.0.1:30000/v1"
    model: str = "default"
    api_key: Optional[str] = None
    #: Sampling. Translation wants determinism, not creativity.
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 512
    #: Per-request timeout. A translation that has not arrived in this long has
    #: already missed the conversation; failing fast lets the session say so.
    timeout_s: float = 20.0
    #: How many previous turns to include as context. 0 disables history.
    history_turns: int = 6
    #: Declared language set. ``None`` = unconstrained (a large multilingual
    #: LLM). Setting it narrows the advertised system language set.
    languages: Optional[Sequence[str]] = None
    #: Extra body fields (e.g. ``{"chat_template_kwargs": {"enable_thinking":
    #: False}}``) -- thinking output would be a disaster in this stage, so a
    #: reasoning-capable model must be told to skip it here.
    extra_body: Dict[str, object] = dataclasses.field(default_factory=dict)


def build_prompt(source: str, target: str) -> str:
    """The system prompt for one direction, built from codes only.

    The English names are looked up for readability -- LLMs translate better
    when told "German to Spanish" than "de to es" -- but they come from the
    same table the matrix uses, so an unknown code degrades to the code itself
    rather than to a wrong language.
    """
    src = display_name(source)
    tgt = display_name(target)
    return (
        f"You are a simultaneous interpreter translating {src['english']} "
        f"({src['code']}) into {tgt['english']} ({tgt['code']}).\n"
        "Rules:\n"
        f"- Output ONLY the {tgt['english']} translation. No preamble, no "
        "explanation, no quotation marks, no notes.\n"
        "- The input is speech recognition output: it may lack punctuation, "
        "contain filler words, or end mid-thought. Translate what was meant, "
        "punctuate naturally, and do not comment on defects.\n"
        "- Preserve the speaker's register and tone, including informality "
        "and politeness level.\n"
        "- Keep proper nouns, numbers, prices, dates and place names exactly.\n"
        f"- If the input is already {tgt['english']}, repeat it unchanged.\n"
        "- If the input is empty or unintelligible, output nothing."
    )


class SentenceAccumulator:
    """Regroups a token stream into units worth handing to a synthesizer.

    A synthesizer called on three words at a time produces choppy prosody and
    pays its fixed per-call cost repeatedly; called once on the whole turn it
    cannot start until generation ends. The compromise: flush on a sentence
    boundary, or when the buffer exceeds ``max_chars`` (so one runaway
    sentence still starts playing), whichever comes first, but never below
    ``min_chars`` (so "Ja." does not become its own synthesis call when more
    text is obviously coming).
    """

    def __init__(self, min_chars: int = 24, max_chars: int = 240) -> None:
        if min_chars >= max_chars:
            raise ValueError("min_chars must be below max_chars")
        self._min = min_chars
        self._max = max_chars
        self._buffer = ""

    def push(self, delta: str) -> List[str]:
        """Add a token; return zero or more complete units."""
        self._buffer += delta
        out: List[str] = []
        while True:
            unit = self._take()
            if unit is None:
                break
            out.append(unit)
        return out

    def _take(self) -> Optional[str]:
        if len(self._buffer) < self._min:
            return None
        match = None
        for match in _BOUNDARY.finditer(self._buffer):
            pass  # last boundary in the buffer
        if match is not None:
            cut = match.end()
            if cut >= self._min:
                unit, self._buffer = self._buffer[:cut], self._buffer[cut:]
                return unit.strip()
        if len(self._buffer) >= self._max:
            # No boundary in sight: cut at the last space so a word is not
            # split across two synthesis calls, which would be audible.
            cut = self._buffer.rfind(" ", self._min, self._max)
            if cut == -1:
                cut = self._max
            unit, self._buffer = self._buffer[:cut], self._buffer[cut:]
            return unit.strip()
        return None

    def flush(self) -> Optional[str]:
        """Emit whatever remains at end of stream."""
        rest, self._buffer = self._buffer.strip(), ""
        return rest or None


class OpenAiMt:
    """MT backend speaking OpenAI chat-completions to our own server."""

    def __init__(
        self,
        config: Optional[MtConfig] = None,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self.name = "openai-mt"
        self.config = config or MtConfig()
        self._client = client
        self._owns_client = client is None
        self._history: Deque[Tuple[str, str]] = deque(
            maxlen=max(self.config.history_turns, 0) or 1
        )
        self._history_enabled = self.config.history_turns > 0

    def supported_languages(self) -> Optional[Iterable[str]]:
        cfg = self.config.languages
        return None if cfg is None else tuple(cfg)

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    def remember(self, source_text: str, target_text: str) -> None:
        """Record a completed turn for the next turn's context."""
        if self._history_enabled:
            self._history.append((source_text, target_text))

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            headers = {}
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            self._client = httpx.AsyncClient(
                base_url=self.config.base_url,
                timeout=self.config.timeout_s,
                headers=headers,
            )
        return self._client

    def _messages(self, text: str, source: str, target: str) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": build_prompt(source, target)}
        ]
        if self._history_enabled:
            for src_text, tgt_text in self._history:
                messages.append({"role": "user", "content": src_text})
                messages.append({"role": "assistant", "content": tgt_text})
        messages.append({"role": "user", "content": text})
        return messages

    def _body(self, text: str, source: str, target: str, stream: bool) -> Dict:
        body: Dict[str, object] = {
            "model": self.config.model,
            "messages": self._messages(text, source, target),
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": self.config.max_tokens,
            "stream": stream,
        }
        body.update(self.config.extra_body)
        return body

    async def translate(self, text: str, source: str, target: str) -> str:
        text = text.strip()
        if not text:
            return ""
        try:
            response = await self._http().post(
                "/chat/completions", json=self._body(text, source, target, False)
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPError as exc:
            raise BackendError(
                "mt",
                f"request to {self.config.base_url} failed: "
                f"{type(exc).__name__}: {exc or '(no detail)'} "
                f"[timeout {self.config.timeout_s:g}s]"
            )
        try:
            return payload["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, AttributeError, TypeError) as exc:
            raise BackendError("mt", f"unexpected response shape: {exc}")

    async def ask(self, system: str, user: str, max_tokens: int = 200) -> str:
        """One non-translation question to the same LLM.

        Used by the name extractor (§17.3). It shares the endpoint and the
        model but NOT the translation prompt or the conversation history: a
        classification carrying twenty turns of dialogue context would answer
        about the conversation instead of about the utterance.
        """
        body: Dict[str, object] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "stream": False,
        }
        body.update(self.config.extra_body)
        try:
            response = await self._http().post("/chat/completions", json=body)
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPError as exc:
            raise BackendError(
                "mt",
                f"request to {self.config.base_url} failed: "
                f"{type(exc).__name__}: {exc or '(no detail)'} "
                f"[timeout {self.config.timeout_s:g}s]"
            )
        try:
            return payload["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, AttributeError, TypeError) as exc:
            raise BackendError("mt", f"unexpected response shape: {exc}")

    async def translate_stream(
        self, text: str, source: str, target: str
    ) -> AsyncIterator[str]:
        text = text.strip()
        if not text:
            return
        body = self._body(text, source, target, True)
        try:
            async with self._http().stream(
                "POST", "/chat/completions", json=body
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data or data == "[DONE]":
                        continue
                    try:
                        event = json.loads(data)
                        delta = event["choices"][0]["delta"].get("content")
                    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                        # A malformed or non-content chunk (role announcement,
                        # usage record, keepalive) is skipped rather than
                        # failing the turn -- servers legitimately interleave
                        # these and none of them carry translation text.
                        continue
                    if delta:
                        yield delta
        except httpx.HTTPError as exc:
            # The TYPE, always, and the timeout with it. `str(exc)` is EMPTY
            # for httpx.ReadTimeout, ConnectTimeout, ReadError and
            # RemoteProtocolError alike, so the message this used to raise was
            # literally "stream from <url> failed:" with nothing after the
            # colon -- true, useless, and it cost a whole investigation to get
            # back to. A read timeout here means MT produced no token within
            # the budget, which on a shared card is what contention with the
            # talker looks like; a protocol error means the server went away.
            # Those need opposite responses, so they must not log the same.
            raise BackendError(
                "mt",
                f"stream from {self.config.base_url} failed: "
                f"{type(exc).__name__}: {exc or '(no detail)'} "
                f"[timeout {self.config.timeout_s:g}s]"
            )
