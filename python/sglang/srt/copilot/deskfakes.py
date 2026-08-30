"""Adversarial unit doubles for the three backend protocols.

These are NOT the stub set (``stubs.py``). The distinction is the whole point
of having both, and it is worth stating once, here:

* ``stubs.py`` is the APP's stand-in for the rig. It is realistic -- scripted
  words with human timing, partials, latency -- so a person can use the product
  and an acceptance run means something. All development and all acceptance run
  against it.
* this module is the UNIT SUITE's stand-in. It is deliberately DEGENERATE: it
  behaves in the worst legal way at a named place, so that a component quietly
  depending on the good case fails a test instead of failing on a rig. It is
  also synchronous and instant, which is what makes a fast unit suite possible.

Per the desk-fake law, a fake indistinguishable from the real thing is a trap.
Every double below states its named difference in its docstring, each one is
asserted by a test, and everything they emit is marked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from sglang.srt.copilot.backends import (
    AsrError,
    AsrEvents,
    BackendSet,
    PrepResult,
    SwitchResult,
    TranscriptDelta,
)
from sglang.srt.copilot.config import CopilotConfig
from sglang.srt.copilot.hints import HintRequest, HintResult, RequestKind
from sglang.srt.copilot.protocol import Track


class DeskFakeAsrStream:
    """One track of :class:`DeskFakeAsrBackend`.

    NAMED DIFFERENCE FROM THE REAL ENDPOINT -- read before relying on this:
    the real ``/v1/realtime`` emits a stream of PARTIAL deltas
    (``conversation.item.input_audio_transcription.delta``) while audio is
    still arriving, and only then a completed event. This double emits NOTHING
    until ``commit`` and then exactly one final line. Any component that works
    against it because partials arrive early is untested; any component that
    depends on partials will look correct here and starve in production. The
    dependency is therefore visible at the seam instead of buried.
    """

    MARKER = "[desk-fake-asr]"

    def __init__(
        self,
        track: Track,
        events: AsrEvents,
        phrases: List[str],
    ) -> None:
        self.track = track
        self.events = events
        self.phrases = phrases
        self.appended_bytes = 0
        self.commits = 0
        self.closed = False

    async def append(self, pcm: bytes) -> None:
        if self.closed:
            raise AsrError("invalid_state", "append on a closed ASR stream")
        if len(pcm) % 2 != 0:
            raise AsrError(
                "invalid_audio", f"pcm16 chunk has an odd byte count ({len(pcm)})"
            )
        self.appended_bytes += len(pcm)
        # No partials. See NAMED DIFFERENCE above.

    async def commit(self) -> None:
        if self.appended_bytes == 0:
            raise AsrError("invalid_state", "cannot commit an empty audio buffer")
        phrase = self.phrases[self.commits % len(self.phrases)]
        self.commits += 1
        self.appended_bytes = 0
        await self.events.on_delta(
            TranscriptDelta(
                track=self.track,
                text=f"{self.MARKER} {phrase}",
                item_id=f"fake-{self.track.value}-{self.commits}",
                final=True,
            )
        )

    async def close(self) -> None:
        self.closed = True


DEFAULT_PHRASES = [
    "the quarterly figures came in above plan",
    "we should talk about the migration timeline",
    "what does that mean for the support contract",
]


@dataclass
class DeskFakeAsrBackend:
    """Instant, deterministic transcription with no timing at all.

    Second named difference, beyond the missing partials: a line appears the
    instant ``commit`` is called, with no relation to how much audio arrived.
    Nothing here can be used to reason about latency.
    """

    phrases: List[str] = field(default_factory=lambda: list(DEFAULT_PHRASES))
    streams: Dict[Track, DeskFakeAsrStream] = field(default_factory=dict)

    async def open(self, track: Track, events: AsrEvents) -> DeskFakeAsrStream:
        stream = DeskFakeAsrStream(track, events, self.phrases)
        self.streams[track] = stream
        return stream

    async def aclose(self) -> None:
        for stream in self.streams.values():
            stream.closed = True


@dataclass
class DeskFakeHints:
    """Hints assembled from the prompt, with a fabricated usage report.

    NAMED DIFFERENCE FROM THE REAL BACKEND -- read before relying on this:
    with ``always_warm=True`` (the default) it reports
    ``prompt_tokens_details.cached_tokens`` equal to the FULL prompt on every
    call, i.e. it claims a perfect cache hit unconditionally. The real runtime
    reports whatever the radix tree actually held, and reports a zero hit by
    OMITTING the details object entirely
    (``entrypoints/openai/usage_processor.py:15``).

    Consequence, and the reason the difference is named here: a residency probe
    validated only against this double is UNTESTED, because it never sees a
    miss. Construct ``DeskFakeHints(always_warm=False)`` to get the miss
    behaviour -- including the omitted-details shape -- and test the probe
    against both.

    Second difference: every result carries ``desk_fake=True`` and the text is
    assembled from the prompt, so no output can be mistaken for a model's.
    """

    config: CopilotConfig
    always_warm: bool = True
    cached_fraction: float = 0.0
    """Used only when ``always_warm`` is False."""

    latency_ms: float = 3.0
    fail: bool = False
    """When set, every call raises. For degradation tests."""

    calls: List[HintRequest] = field(default_factory=list)

    MARKER = "[desk-fake-hint]"

    async def complete(self, req: HintRequest) -> HintResult:
        self.calls.append(req)
        if self.fail:
            raise RuntimeError("desk-fake hint backend is in injected-failure mode")
        prompt_tokens = max(1, len(req.prompt.split()))
        if self.always_warm:
            cached = prompt_tokens
        else:
            cached = int(prompt_tokens * self.cached_fraction)
        usage: Dict[str, Any] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": 8,
            "total_tokens": prompt_tokens + 8,
        }
        if cached > 0:
            # Mirrors usage_processor.py:15 -- details exist only for a hit.
            usage["prompt_tokens_details"] = {"cached_tokens": cached}
        if req.kind is RequestKind.PRIME:
            text = ""
        elif req.kind is RequestKind.EXPANSION:
            text = f"{self.MARKER} expansion for {req.topic_id}"
        else:
            head = req.topic_id.replace("-", " ")
            text = f"- {self.MARKER} {head}\n- keyword from the briefing"
        return HintResult(
            text=text,
            usage=usage,
            latency_ms=self.latency_ms,
            desk_fake=True,
            prompt_tokens=prompt_tokens,
        )

    async def aclose(self) -> None:
        return None


@dataclass
class DeskFakePrep:
    """Prepared contexts that are unbounded and UNENUMERABLE.

    NAMED DIFFERENCE, and the reason this double is not simply the stub prep:
    ``report()`` deliberately omits the ``held`` key. That models the RIG, not
    a convenience -- a radix prefix is not addressable by topic from outside
    the runtime, so on a real backend an eviction can only ever be discovered
    by the warmth probe on the next request. The session's reconciliation path
    for "the backend cannot tell me what it holds" is therefore exercised by
    the unit suite, while ``stubs.StubSessionPrep`` exercises the other one.

    Second difference: it never evicts, so a component that only works while
    everything stays prepared passes here.
    """

    config: CopilotConfig
    prepared: Dict[str, int] = field(default_factory=dict)
    prepares: int = 0
    switches: int = 0

    @property
    def capacity(self) -> int:
        return 1 << 30

    async def prepare(self, topic_id: str, prefix_text: str) -> PrepResult:
        tokens = max(1, len(prefix_text.split()))
        self.prepared[topic_id] = tokens
        self.prepares += 1
        return PrepResult(
            topic_id=topic_id, prepared_tokens=tokens, evicted=[], prep_ms=0.0
        )

    async def switch(self, topic_id: str) -> SwitchResult:
        self.switches += 1
        return SwitchResult(
            topic_id=topic_id,
            prepared=topic_id in self.prepared,
            switch_ms=0.0,
        )

    def report(self) -> Dict[str, Any]:
        return {
            "backend": "desk-fake",
            "capacity": None,
            "prepares": self.prepares,
            "switches": self.switches,
            # No "held" key: see the NAMED DIFFERENCE above.
        }


def desk_fake_backend_set(
    config: CopilotConfig,
    always_warm: bool = True,
    hints: Optional[DeskFakeHints] = None,
    phrases: Optional[List[str]] = None,
) -> BackendSet:
    """A :class:`BackendSet` of doubles, for the unit suite.

    Not reachable from the launcher: ``build_backend_set`` knows only ``stub``
    and ``rig``. A degenerate double is a test instrument, and shipping it as a
    third named runtime mode would eventually get one booted by mistake.
    """
    hint_backend = hints or DeskFakeHints(config=config, always_warm=always_warm)
    prep = DeskFakePrep(config=config)
    return BackendSet(
        name="desk-fake",
        hints=hint_backend,
        prep=prep,
        stub=True,
        _asr_factory=lambda session_id: DeskFakeAsrBackend(
            phrases=list(phrases) if phrases else list(DEFAULT_PHRASES)
        ),
    )
