"""Topic registry: pre-prefilled contexts, and the probe that verifies them.

Design basis (DESIGN_502 §2.2): this tree has no session-checkpoint API and no
prefix-pin API. ``lock_ref`` protects a radix node only while a request is
in flight (``mem_cache/radix_cache.py:598-625``), and
``--enable-session-radix-cache`` states its own reach in its docstring --
"Tagged KV is ordinary LRU radix -- no pinning, no open"
(``mem_cache/session_radix_cache.py:23-27``). What CAN be done is:

* prime a topic prefix with a prefill-only request (``max_new_tokens = 0``,
  permitted at ``sampling_params.py:207-211``),
* rank it late in the eviction victim order via request priority
  (``mem_cache/evict_policy.py:41-47``, ``mem_cache/radix_cache.py:241``),
* re-issue the prime on a cadence to refresh ``last_access_time``,
* and MEASURE whether it actually stayed resident.

The measurement is the point. "We primed it" is a success message about state,
and this file treats it as one: a topic is never reported WARM because a prime
was sent -- only because a subsequent request came back reporting cached
tokens. Until then the state is UNKNOWN.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional

from sglang.srt.copilot.briefing import Briefing
from sglang.srt.copilot.config import CopilotConfig


class Warmth(str, Enum):
    """Measured residency verdict for a topic prefix."""

    UNKNOWN = "unknown"
    """Primed but never yet observed. NOT a synonym for warm."""

    WARM = "warm"
    PARTIAL = "partial"
    COLD = "cold"


def read_cached_tokens(usage: Optional[Mapping[str, Any]]) -> int:
    """Extract ``cached_tokens`` from an OpenAI usage object.

    The runtime reports a zero cache hit by OMITTING the details object:
    ``entrypoints/openai/usage_processor.py:15`` returns
    ``PromptTokensDetails(cached_tokens=count) if count > 0 else None``.
    A reader that treats a missing ``prompt_tokens_details`` as "unknown" would
    never see a cold prefix -- misses would be invisible, which is the exact
    silent-wrongness shape this probe exists to prevent. Absent means ZERO.
    """
    if not usage:
        return 0
    details = usage.get("prompt_tokens_details")
    if not details:
        return 0
    value = details.get("cached_tokens")
    if value is None:
        return 0
    return int(value)


@dataclass
class TopicState:
    """One pre-prefilled topic context."""

    topic_id: str
    title: str
    prefix_text: str

    primed_tokens: int = 0
    """Prompt token count the runtime reported for the priming request."""

    last_primed_at: float = 0.0
    prime_count: int = 0

    warmth: Warmth = Warmth.UNKNOWN
    last_cached_tokens: Optional[int] = None
    observations: int = 0
    misses: int = 0
    """Observations whose verdict was COLD. Reported, never smoothed away."""

    def hit_ratio(self) -> Optional[float]:
        if self.last_cached_tokens is None or self.primed_tokens <= 0:
            return None
        return self.last_cached_tokens / self.primed_tokens

    def to_json(self) -> Dict[str, Any]:
        return {
            "topic_id": self.topic_id,
            "title": self.title,
            "warmth": self.warmth.value,
            "primed_tokens": self.primed_tokens,
            "cached_tokens": self.last_cached_tokens,
            "hit_ratio": self.hit_ratio(),
            "observations": self.observations,
            "misses": self.misses,
            "prime_count": self.prime_count,
        }


@dataclass
class PrimeRequest:
    """A prefill-only request the caller is expected to send.

    The registry does not perform I/O; it produces the request shape and
    consumes the result. That keeps it hermetically testable and keeps the
    transport decision (which backend, which URL) in one place.
    """

    topic_id: str
    prompt: str
    max_tokens: int = 0
    priority: int = 0
    lane: Optional[str] = None


@dataclass
class TopicRegistry:
    """All topics of one conversation, with their measured warmth."""

    config: CopilotConfig
    topics: Dict[str, TopicState] = field(default_factory=dict)
    order: List[str] = field(default_factory=list)
    focus: Optional[str] = None

    def _now(self) -> float:
        return time.monotonic()

    # --- population -------------------------------------------------------

    def sync_from_briefing(self, briefing: Briefing) -> List[str]:
        """Create or refresh a topic per briefing section.

        Returns the ids of topics whose prefix text changed, since those must
        be re-primed: their old prefix is a different radix path and any warmth
        measured for it does not carry over.
        """
        header = f"# {briefing.title}\n{briefing.preamble}".strip()
        changed: List[str] = []
        for sec in briefing.sections:
            prefix = f"{header}\n\n{sec.text()}".strip() if header else sec.text()
            existing = self.topics.get(sec.anchor)
            if existing is None:
                self.topics[sec.anchor] = TopicState(
                    topic_id=sec.anchor, title=sec.title, prefix_text=prefix
                )
                self.order.append(sec.anchor)
                changed.append(sec.anchor)
            elif existing.prefix_text != prefix:
                existing.prefix_text = prefix
                existing.title = sec.title
                # The prefix moved, so every number measured for the old one is
                # about a path that no longer exists.
                existing.warmth = Warmth.UNKNOWN
                existing.last_cached_tokens = None
                existing.primed_tokens = 0
                changed.append(sec.anchor)
        if self.focus is None and self.order:
            self.focus = self.order[0]
        return changed

    def get(self, topic_id: str) -> Optional[TopicState]:
        return self.topics.get(topic_id)

    def set_focus(self, topic_id: str) -> TopicState:
        topic = self.topics.get(topic_id)
        if topic is None:
            raise KeyError(f"unknown topic {topic_id!r}")
        self.focus = topic_id
        return topic

    def focused(self) -> Optional[TopicState]:
        if self.focus is None:
            return None
        return self.topics.get(self.focus)

    # --- priming ----------------------------------------------------------

    def prime_request(self, topic_id: str) -> PrimeRequest:
        topic = self.topics[topic_id]
        return PrimeRequest(
            topic_id=topic_id,
            prompt=topic.prefix_text,
            max_tokens=0,
            priority=self.config.hint_priority,
            lane=None,
        )

    def prime_candidates(self, limit: Optional[int] = None) -> List[str]:
        """Which topics are worth preparing, in the order to prepare them.

        Two rules, both learned from watching this run against a bounded
        backend:

        1. NEVER ASK FOR MORE THAN THE BACKEND HOLDS. With five sections and a
           capacity of three, preparing all five means every round evicts what
           the previous round prepared -- steady churn, and every topic
           measured cold. When the backend states a capacity, the app respects
           it and leaves the surplus topics honestly unprepared; a switch to one
           of those costs exactly one slow suggestion. ``limit=None`` means the
           backend cannot state a capacity (the rig case), and then nothing is
           held back.
        2. THE FOCUSED TOPIC IS PREPARED LAST, so it is the most recently used
           and therefore the last to be evicted. Preparing in briefing order
           would systematically evict the one topic the conversation is about.

        Beyond the focused topic, briefing order decides -- it is the user's own
        ordering and this module has no better one.
        """
        others = [t for t in self.order if t != self.focus]
        focused = [self.focus] if self.focus in self.topics else []
        if limit is not None and limit > 0:
            others = others[: max(0, limit - len(focused))]
        return others + focused

    def due_for_prime(
        self, now: Optional[float] = None, limit: Optional[int] = None
    ) -> List[PrimeRequest]:
        """Candidates whose prepare has never run or is older than the cadence.

        The cadence exists to refresh ``last_access_time``, which is the second
        key of the priority eviction strategy (``evict_policy.py:41-47``); it
        does not make the prefix un-evictable.
        """
        now = self._now() if now is None else now
        due: List[PrimeRequest] = []
        for topic_id in self.prime_candidates(limit):
            topic = self.topics[topic_id]
            if (
                topic.prime_count == 0
                or now - topic.last_primed_at >= self.config.topic_touch_interval_s
            ):
                due.append(self.prime_request(topic_id))
        return due

    def record_prime(
        self, topic_id: str, prompt_tokens: int, now: Optional[float] = None
    ) -> TopicState:
        """Record that a preparation completed.

        Deliberately does NOT claim warmth. A completed prepare says the tokens
        were prefilled once; it says nothing about whether they are still
        resident when the next request arrives, and that difference is exactly
        what the probe measures.

        It DOES clear the previous verdict back to UNKNOWN, because that verdict
        was about a prefix that has since been re-prepared. Leaving a stale COLD
        in place made the UI report a topic as cold immediately after preparing
        it -- a claim about a state that no longer existed. The cumulative
        ``misses`` count is kept: that is the history, and it is what
        ``miss_report`` reports.
        """
        topic = self.topics[topic_id]
        topic.primed_tokens = int(prompt_tokens)
        topic.last_primed_at = self._now() if now is None else now
        topic.prime_count += 1
        topic.warmth = Warmth.UNKNOWN
        topic.last_cached_tokens = None
        return topic

    # --- probing ----------------------------------------------------------

    def observe_usage(
        self, topic_id: str, usage: Optional[Mapping[str, Any]]
    ) -> Warmth:
        """Update a topic's warmth from a real request's usage report."""
        return self.observe(topic_id, read_cached_tokens(usage))

    def observe(self, topic_id: str, cached_tokens: int) -> Warmth:
        topic = self.topics[topic_id]
        topic.last_cached_tokens = int(cached_tokens)
        topic.observations += 1
        if topic.primed_tokens <= 0:
            # Nothing was primed, so there is nothing to be warm about. This is
            # a definite COLD, not an UNKNOWN: the prefix is measurably absent.
            topic.warmth = Warmth.COLD
            topic.misses += 1
            return topic.warmth
        ratio = cached_tokens / topic.primed_tokens
        if ratio >= self.config.warm_ratio:
            topic.warmth = Warmth.WARM
        elif ratio >= self.config.partial_ratio:
            topic.warmth = Warmth.PARTIAL
        else:
            topic.warmth = Warmth.COLD
            topic.misses += 1
        return topic.warmth

    def record_eviction(self, topic_id: str) -> Optional[TopicState]:
        """A backend reported that it no longer holds this topic's context.

        Counted as an OBSERVED miss, not as a return to UNKNOWN: the backend
        said the prefix is gone, and that is a measurement. ``primed_tokens``
        is deliberately kept -- it records the size of the prefix this app
        primed, which is still the right denominator when the next request
        repopulates it.
        """
        topic = self.topics.get(topic_id)
        if topic is None:
            return None
        topic.last_cached_tokens = 0
        topic.observations += 1
        topic.misses += 1
        topic.warmth = Warmth.COLD
        return topic

    # --- reporting --------------------------------------------------------

    def states(self) -> List[Dict[str, Any]]:
        """Every topic in briefing order, with which one is focused.

        Focus belongs to the registry, not to a topic: exactly one is live at a
        time and a per-topic flag could contradict itself.
        """
        return [
            {**self.topics[t].to_json(), "focus": t == self.focus} for t in self.order
        ]

    def warm_topics(self) -> Iterable[TopicState]:
        return (t for t in self.topics.values() if t.warmth is Warmth.WARM)

    def miss_report(self) -> Dict[str, Any]:
        """Summary an operator can act on: how often the warmth claim failed."""
        total_obs = sum(t.observations for t in self.topics.values())
        total_miss = sum(t.misses for t in self.topics.values())
        return {
            "topics": len(self.topics),
            "observations": total_obs,
            "misses": total_miss,
            "miss_rate": (total_miss / total_obs) if total_obs else None,
        }
