"""Hint generation and background briefing expansion.

Both talk to the runtime's own ``/v1/chat/completions``, the same seam the #466
translator uses for MT (``translator/mt.py:5-9``: "we do not reach into the
engine, we call it like a stranger would").

Two request classes with different scheduling intent:

* live hints carry ``lane="fast"`` and a high ``priority``,
* background expansion carries no lane and the heavy-tier default priority.

Both fields exist on the chat request schema
(``entrypoints/openai/protocol.py:868-873``, validated at ``:907-911``) and are
forwarded to ``GenerateReqInput.lane`` / ``.priority``. They take effect only
when the runtime runs with ``--enable-fast-lane`` (which force-enables priority
scheduling, ``server_args.py:15069-15070``); on a runtime without it the fields
are inert rather than an error, so the copilot works either way and the
scheduling advantage is a runtime configuration choice.

PREFIX DISCIPLINE (load-bearing): a hint prompt is built as
``topic.prefix_text`` VERBATIM followed by the live tail. Any reformatting of
the prefix -- a stripped newline, a reordered header -- produces a different
token path and therefore a radix miss, which turns the whole warm-topic
mechanism into a no-op. The prefix is concatenated, never rebuilt.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from sglang.srt.copilot.config import CopilotConfig


class RequestKind(str, Enum):
    PRIME = "prime"
    HINT = "hint"
    EXPANSION = "expansion"


HINT_INSTRUCTION = (
    "You support a person who is IN this conversation right now. "
    "Write at most three very short bullet points they can READ at a glance: "
    "a keyword, a number, or one clause of explanation each. "
    "Never write anything for them to say out loud. No greetings, no preamble."
)

EXPANSION_INSTRUCTION = (
    "Extend the briefing for the topic above with what this conversation has "
    "revealed that the briefing does not yet contain. Write one short Markdown "
    "section. State only what was actually said or what follows directly from "
    "it. If nothing new came up, answer exactly: NOTHING NEW."
)

NOTHING_NEW = "NOTHING NEW"


@dataclass
class HintRequest:
    kind: RequestKind
    topic_id: str
    prompt: str
    max_tokens: int
    priority: int
    lane: Optional[str] = None


@dataclass
class HintResult:
    text: str
    usage: Optional[Dict[str, Any]] = None
    latency_ms: float = 0.0
    desk_fake: bool = False
    prompt_tokens: int = 0

    def bullets(self) -> List[str]:
        out: List[str] = []
        for raw in self.text.splitlines():
            line = raw.strip().lstrip("-*• ").strip()
            if line:
                out.append(line)
        return out


def build_hint_prompt(prefix_text: str, transcript_tail: str) -> str:
    """Topic prefix verbatim, then the live tail. See PREFIX DISCIPLINE above."""
    return (
        f"{prefix_text}\n\n## Live transcript\n{transcript_tail}\n\n{HINT_INSTRUCTION}"
    )


def build_expansion_prompt(prefix_text: str, transcript_tail: str) -> str:
    return (
        f"{prefix_text}\n\n## Live transcript\n{transcript_tail}\n\n"
        f"{EXPANSION_INSTRUCTION}"
    )


def build_chat_body(config: CopilotConfig, req: HintRequest) -> Dict[str, Any]:
    """The exact JSON body sent to ``/v1/chat/completions``."""
    body: Dict[str, Any] = {
        "model": config.model,
        "messages": [{"role": "user", "content": req.prompt}],
        "max_tokens": req.max_tokens,
        "temperature": 0.3,
        "stream": False,
        "priority": req.priority,
    }
    if req.lane is not None:
        body["lane"] = req.lane
    return body


def hint_request(
    config: CopilotConfig, topic_id: str, prefix_text: str, transcript_tail: str
) -> HintRequest:
    return HintRequest(
        kind=RequestKind.HINT,
        topic_id=topic_id,
        prompt=build_hint_prompt(prefix_text, transcript_tail),
        max_tokens=config.hint_max_tokens,
        priority=config.hint_priority,
        lane=config.hint_lane,
    )


def expansion_request(
    config: CopilotConfig, topic_id: str, prefix_text: str, transcript_tail: str
) -> HintRequest:
    return HintRequest(
        kind=RequestKind.EXPANSION,
        topic_id=topic_id,
        prompt=build_expansion_prompt(prefix_text, transcript_tail),
        max_tokens=config.expander_max_tokens,
        priority=config.expander_priority,
        lane=None,
    )


def prime_chat_request(
    config: CopilotConfig, topic_id: str, prefix_text: str
) -> HintRequest:
    """A prefill-only request that populates the radix prefix without output.

    ``max_tokens=0`` is permitted (``sampling_params.py:207-211``). Note the
    reach limit: ``Req.is_prefill_only`` additionally requires
    ``speculative_algorithm is None`` (``schedule_batch.py:1097-1103``), so
    under the standing MTP recipe the prefill-only OPTIMISATION is off while
    the request still runs prefill and still populates the tree. Priming works
    under speculation; only its cheapness does not transfer.
    """
    return HintRequest(
        kind=RequestKind.PRIME,
        topic_id=topic_id,
        prompt=prefix_text,
        max_tokens=0,
        priority=config.hint_priority,
        lane=None,
    )


class ChatHintBackend:
    """Real backend: an ordinary OpenAI client against the runtime's port."""

    def __init__(self, config: CopilotConfig, client: Any = None) -> None:
        self.config = config
        self._client = client
        self._owned = client is None

    async def _get_client(self):
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(timeout=60.0)
        return self._client

    async def complete(self, req: HintRequest) -> HintResult:
        client = await self._get_client()
        headers = {}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        body = build_chat_body(self.config, req)
        started = time.perf_counter()
        response = await client.post(self.config.chat_url(), json=body, headers=headers)
        response.raise_for_status()
        data = response.json()
        latency_ms = (time.perf_counter() - started) * 1000.0
        choices = data.get("choices") or []
        text = ""
        if choices:
            text = (choices[0].get("message") or {}).get("content") or ""
        usage = data.get("usage")
        return HintResult(
            text=text,
            usage=usage,
            latency_ms=latency_ms,
            desk_fake=False,
            prompt_tokens=int((usage or {}).get("prompt_tokens", 0)),
        )

    async def aclose(self) -> None:
        if self._client is not None and self._owned:
            await self._client.aclose()
            self._client = None
