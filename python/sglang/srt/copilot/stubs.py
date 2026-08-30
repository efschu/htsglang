"""The stub set: a whole copilot backend with no rig, no model, no GPU.

All development and all acceptance for this app run against these three
objects. They are reached through the same protocols as the rig set
(``backends.py``), so the rig is a config switch and never a dev dependency.

These are NOT the desk fakes in ``asr_client.py`` / ``hints.py``. The two kinds
answer different questions and both are kept:

* the desk fakes are adversarial unit doubles -- deliberately degenerate
  (``DeskFakeAsr`` emits no partials at all, ``DeskFakeHints`` claims a perfect
  cache hit on every call) so that a component silently depending on the good
  case fails a unit test;
* the stubs here are the APP's stand-in for the rig -- realistic enough that a
  human can use the product and a latency number means something, which is
  exactly what makes them dangerous, hence the named differences below.

NAMED DIFFERENCES FROM THE REAL THING -- read before trusting anything measured
against this module:

1. ``StubAsrBackend`` speaks a FIXED SCRIPT. Its transcript is not a
   transcription of the audio, only gated by it: no audio, no words. It also
   never REVISES a partial -- it only appends words -- while a real streaming
   adapter can replace the whole partial text of an item. A component that
   assumes partial text grows monotonically passes here and breaks there.
2. ``StubHints`` latency is a configured constant plus jitter. Any latency
   figure measured against it measures the harness, not a model. It is useful
   for proving the app's own overhead and nothing else.
3. ``StubSessionPrep`` evicts deterministically by LRU within its own
   capacity. The real radix tree also loses a prefix to OTHER tenants' memory
   pressure at unpredictable times, so "prepared" is stickier here than it will
   ever be on a rig.

Every frame these produce carries ``stub=True`` all the way into the browser,
which paints a permanent STUB banner. A stub whose output is indistinguishable
from the real thing is a trap.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from sglang.srt.copilot.backends import (
    AsrError,
    AsrEvents,
    PrepResult,
    SwitchResult,
    TranscriptDelta,
)
from sglang.srt.copilot.config import CopilotConfig
from sglang.srt.copilot.hints import HintRequest, HintResult, RequestKind
from sglang.srt.copilot.protocol import Track

# ---------------------------------------------------------------- the script


@dataclass(frozen=True)
class ScriptedUtterance:
    track: Track
    text: str


@dataclass(frozen=True)
class StubScript:
    name: str
    utterances: List[ScriptedUtterance]


def _utt(pairs) -> List[ScriptedUtterance]:
    return [ScriptedUtterance(track=t, text=s) for t, s in pairs]


RENEWAL_CALL = StubScript(
    name="renewal-call",
    utterances=_utt(
        [
            (
                Track.OTHER,
                "thanks for making the time, I wanted to walk through the "
                "contract renewal before the quarter closes",
            ),
            (
                Track.SELF,
                "of course, my understanding is that the current term ends in "
                "March and there is a sixty day notice window",
            ),
            (
                Track.OTHER,
                "that matches my note, although procurement flagged the "
                "support tier as the one open point",
            ),
            (
                Track.SELF,
                "we can go through the support SLA, today it is a four hour "
                "response on business days",
            ),
            (
                Track.OTHER,
                "and how does that sit with the migration timeline, we still "
                "have two clusters to move in the third quarter",
            ),
            (
                Track.SELF,
                "the migration lands after the renewal date, so the new "
                "service levels would already apply to it",
            ),
            (
                Track.OTHER,
                "last thing from my side, the price adjustment clause, is that "
                "indexed or is it fixed",
            ),
            (
                Track.SELF,
                "it is indexed to the published list price with a cap of three "
                "percent per year",
            ),
        ]
    ),
)

SHORT_CHECK = StubScript(
    name="short-check",
    utterances=_utt(
        [
            (Track.OTHER, "can you hear me on this side"),
            (Track.SELF, "yes I can hear you clearly"),
        ]
    ),
)

SCRIPTS: Dict[str, StubScript] = {s.name: s for s in (RENEWAL_CALL, SHORT_CHECK)}


def load_script(name: str) -> StubScript:
    script = SCRIPTS.get(name)
    if script is None:
        raise KeyError(
            f"unknown stub script {name!r}; known scripts are {sorted(SCRIPTS)}"
        )
    return script


STUB_BRIEFING = """# Renewal call — Northwind

Counterpart: procurement lead. Goal: close the renewal this quarter without
giving up the response-time commitment.

## Contract renewal
Current term ends 31 March. Notice window is 60 days, so the decision deadline
is 30 January. Silence auto-renews for twelve months.

## Support SLA
Today: 4 hour response, business days only, no service credits. The new tier we
want to sell adds a 1 hour target for P1 and introduces credits at 99.5 percent.

## Migration timeline
Two clusters still to move, planned for Q3. There is a change freeze in
December. The migration starts after the renewal date.

## Price adjustment
Indexed to the published list price, capped at 3 percent per year. Last actual
increase was 2.1 percent.
"""


def stub_briefing_text() -> str:
    """The briefing the stub script is a conversation about.

    Four sections against a default prepared capacity of three, on purpose: the
    fourth topic is measurably NOT prepared, so an eviction and a cold switch
    are both reachable in a demo instead of being theoretical.
    """
    return STUB_BRIEFING


# ------------------------------------------------------------------ stub ASR


class StubAsrStream:
    """One track's scripted transcription stream.

    Holds the events sink and the audio arrival evidence; the words themselves
    produced by the backend's single driver task, because the script is one
    conversation and a per-stream driver would have to coordinate turn-taking
    with its sibling.
    """

    def __init__(self, track: Track, events: AsrEvents) -> None:
        self.track = track
        self.events = events
        self.audio_bytes = 0
        self.total_audio_bytes = 0
        self.last_audio_at = -1e9
        self.commit_requested = False
        self.closed = False

    async def append(self, pcm: bytes) -> None:
        if self.closed:
            raise AsrError("invalid_state", "append on a closed ASR stream")
        if len(pcm) % 2 != 0:
            raise AsrError(
                "invalid_audio", f"pcm16 chunk has an odd byte count ({len(pcm)})"
            )
        self.audio_bytes += len(pcm)
        self.total_audio_bytes += len(pcm)
        self.last_audio_at = time.monotonic()

    async def commit(self) -> None:
        if self.audio_bytes == 0:
            # Same refusal as the real endpoint (realtime/session.py:491-494).
            raise AsrError("invalid_state", "cannot commit an empty audio buffer")
        self.audio_bytes = 0
        self.commit_requested = True

    async def close(self) -> None:
        self.closed = True

    def streaming(self, grace_s: float) -> bool:
        return not self.closed and (time.monotonic() - self.last_audio_at) <= grace_s


class StubAsrBackend:
    """Speaks the script, gated on real audio arriving.

    The gate is the honest part: the words are canned, but they only appear
    while the browser is actually streaming frames for that track. A silent or
    closed track produces no transcript, so a broken capture chain cannot hide
    behind the script.
    """

    def __init__(
        self, config: CopilotConfig, script: StubScript, session_id: str
    ) -> None:
        self.config = config
        self.script = script
        self.session_id = session_id
        self.streams: Dict[Track, StubAsrStream] = {}
        self.cursor = 0
        self.spoken = 0
        self._task: Optional[asyncio.Task] = None

    async def open(self, track: Track, events: AsrEvents) -> StubAsrStream:
        stream = StubAsrStream(track, events)
        self.streams[track] = stream
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="stub-asr-driver")
        return stream

    async def aclose(self) -> None:
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:  # pragma: no cover - already reported by _run
                pass
        for stream in self.streams.values():
            stream.closed = True

    # --- the driver -------------------------------------------------------

    async def _run(self) -> None:
        """Supervisor around the script loop.

        A driver that dies quietly would leave a transcript pane that simply
        stops filling -- the silent freeze this app is not allowed to have. Any
        failure is reported on the ASR error channel of whichever track was
        being spoken.
        """
        try:
            await self._loop()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            track = self.script.utterances[
                self.cursor % len(self.script.utterances)
            ].track
            stream = self.streams.get(track)
            if stream is not None:
                await stream.events.on_error(
                    track, AsrError("stub_driver_failed", repr(exc))
                )

    async def _loop(self) -> None:
        tick = 0.04
        grace = self.config.stub_audio_grace_s
        while True:
            utterance = self.script.utterances[
                self.cursor % len(self.script.utterances)
            ]
            stream = self.streams.get(utterance.track)
            if stream is None or stream.closed:
                # That side is not being captured at all: its turn is skipped
                # rather than queued, so a one-track session still runs.
                self.cursor += 1
                await asyncio.sleep(tick)
                continue
            if not stream.streaming(grace):
                await asyncio.sleep(tick)
                continue
            await self._speak(utterance, stream)
            self.cursor += 1
            self.spoken += 1
            await asyncio.sleep(self.config.stub_gap_ms / 1000.0)

    async def _speak(self, utterance: ScriptedUtterance, stream: StubAsrStream) -> None:
        item_id = f"stub-{utterance.track.value}-{self.cursor}"
        words = utterance.text.split()
        said: List[str] = []
        grace = self.config.stub_audio_grace_s
        for word in words:
            await asyncio.sleep(self.config.stub_word_ms / 1000.0)
            if stream.closed:
                break
            if stream.commit_requested:
                break
            if not stream.streaming(grace):
                # Audio stopped mid-utterance. A real adapter emits nothing
                # further until either audio resumes or the client commits, so
                # neither does this.
                if not await self._await_audio(stream):
                    break
            said.append(word)
            await stream.events.on_delta(
                TranscriptDelta(
                    track=utterance.track,
                    text=" ".join(said),
                    item_id=item_id,
                    final=False,
                    stub=True,
                )
            )
        if stream.closed:
            return
        if not said:
            said = words[:1]
        if not stream.commit_requested:
            await asyncio.sleep(self.config.stub_final_ms / 1000.0)
        stream.commit_requested = False
        await stream.events.on_delta(
            TranscriptDelta(
                track=utterance.track,
                text=" ".join(said),
                item_id=item_id,
                final=True,
                stub=True,
            )
        )

    async def _await_audio(self, stream: StubAsrStream) -> bool:
        """Wait for audio to resume. False if the utterance must end here."""
        deadline = time.monotonic() + self.config.stub_silence_hold_s
        while time.monotonic() < deadline:
            if stream.closed or stream.commit_requested:
                return False
            if stream.streaming(self.config.stub_audio_grace_s):
                return True
            await asyncio.sleep(0.04)
        return False


# ---------------------------------------------------------------- stub hints


HINT_CARDS: Dict[str, List[List[str]]] = {
    "contract-renewal": [
        ["term ends 31 March", "notice window 60 days → decide by 30 Jan"],
        ["silence auto-renews 12 months", "renewal is the lever, not the SLA"],
    ],
    "support-sla": [
        [
            "today: 4 h response, business days",
            "no service credits in the current text",
        ],
        ["new tier: 1 h target for P1", "credits start at 99.5 %"],
    ],
    "migration-timeline": [
        ["2 clusters left, planned Q3", "change freeze in December"],
        ["migration starts AFTER the renewal date", "so the new SLA covers it"],
    ],
    "price-adjustment": [
        ["indexed to published list price", "cap 3 % per year"],
        ["last actual increase 2.1 %", "cap is the concession they already have"],
    ],
}

GENERIC_CARD = ["no briefing section for this topic", "listening for keywords"]

STUB_EXPANSIONS = [
    "Procurement, not the technical owner, is driving the renewal date; the "
    "support tier is the only point they named as open.",
    "The counterpart accepts that the migration falls after the renewal date, "
    "so the new service levels apply to it without a separate amendment.",
    "The price adjustment question was asked as indexed-or-fixed, which means "
    "the 3 percent cap has not been read yet.",
]


class StubFault(RuntimeError):
    """Injected stub failure, so degradation can be demonstrated on purpose."""


@dataclass
class StubHints:
    """Canned suggestion cards with a configurable latency.

    ``fail`` is the acceptance affordance for honest degradation: while it is
    set, every call raises :class:`StubFault`, which the session reports as an
    error event. Nothing about it is silent.
    """

    config: CopilotConfig
    prep: Optional["StubSessionPrep"] = None
    fail: bool = False
    calls: List[HintRequest] = field(default_factory=list)
    _rotation: Dict[str, int] = field(default_factory=dict)
    _expansions: int = 0

    async def complete(self, req: HintRequest) -> HintResult:
        self.calls.append(req)
        prepared_tokens = 0
        if self.prep is not None:
            prepared_tokens = self.prep.prepared_tokens(req.topic_id)
        latency_ms = self.config.stub_hint_latency_ms + random.uniform(
            0.0, self.config.stub_hint_jitter_ms
        )
        if prepared_tokens <= 0 and req.kind is not RequestKind.PRIME:
            # A cold prefix costs prefill. This is the one number the warm-topic
            # mechanism claims to move, so the stub models it rather than
            # pretending every switch is instant.
            latency_ms += self.config.stub_cold_penalty_ms
        started = time.perf_counter()
        await asyncio.sleep(latency_ms / 1000.0)
        if self.fail:
            raise StubFault("stub hint backend is in injected-failure mode")
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        prompt_tokens = max(1, len(req.prompt.split()))
        usage: Dict[str, Any] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": 12,
            "total_tokens": prompt_tokens + 12,
        }
        cached = min(prepared_tokens, prompt_tokens)
        if cached > 0:
            # Details exist only for a hit, exactly as the runtime reports it
            # (entrypoints/openai/usage_processor.py:15).
            usage["prompt_tokens_details"] = {"cached_tokens": cached}

        if prepared_tokens <= 0 and req.kind is not RequestKind.PRIME:
            # The request that paid the prefill leaves the prefix behind, so
            # the NEXT one is warm. That is how the rig behaves -- a radix tree
            # is populated by ordinary traffic, not only by priming -- and it
            # is why a cold switch costs one slow hint rather than all of them.
            if self.prep is not None:
                self.prep.note_populated(req.topic_id, _prefix_tokens(req.prompt))

        if req.kind is RequestKind.PRIME:
            text = ""
        elif req.kind is RequestKind.EXPANSION:
            text = STUB_EXPANSIONS[self._expansions % len(STUB_EXPANSIONS)]
            self._expansions += 1
        else:
            text = "\n".join(f"- {b}" for b in self._bullets(req))
        return HintResult(
            text=text,
            usage=usage,
            latency_ms=elapsed_ms,
            desk_fake=True,
            prompt_tokens=prompt_tokens,
        )

    def _bullets(self, req: HintRequest) -> List[str]:
        cards = HINT_CARDS.get(req.topic_id)
        if cards:
            index = self._rotation.get(req.topic_id, 0)
            self._rotation[req.topic_id] = index + 1
            bullets = list(cards[index % len(cards)])
        else:
            bullets = list(GENERIC_CARD)
        heard = _last_transcript_line(req.prompt)
        if heard:
            bullets.append(f"heard: “{heard}”")
        return bullets

    async def aclose(self) -> None:
        return None


def _last_transcript_line(prompt: str) -> str:
    """The last line of the live tail the prompt carries.

    Echoing it back is what makes a canned card visibly track the conversation
    in a demo; it is also the only part of a stub hint that is not canned.

    The tail is the block between the marker and the blank line that starts the
    instruction (``build_hint_prompt``: prefix, marker, tail, blank line,
    instruction). Reading to the end of the prompt instead quotes the
    INSTRUCTION back at the user, which is what the first browser run showed:
    every card ended with ``heard: "to say out loud. No greetings"``.
    """
    marker = "## Live transcript"
    if marker not in prompt:
        return ""
    tail = prompt.split(marker, 1)[1].lstrip("\n")
    tail = tail.split("\n\n", 1)[0]
    lines = [ln.strip() for ln in tail.splitlines() if ln.strip()]
    if not lines:
        return ""
    line = lines[-1]
    for prefix in ("self:", "other:"):
        if line.startswith(prefix):
            line = line[len(prefix) :].strip()
    words = line.split()
    return " ".join(words[-8:])


def _prefix_tokens(prompt: str) -> int:
    """Token count of the reusable TOPIC PREFIX inside a hint prompt.

    Only the part before the live tail is shared between two requests about the
    same topic, so only that part can come back as a cache hit. Counting the
    whole prompt would report a hit the rig could never deliver.
    """
    head = prompt.split("## Live transcript", 1)[0]
    return max(1, len(head.split()))


# ----------------------------------------------------------- stub prepared set


@dataclass
class _Prepared:
    topic_id: str
    tokens: int
    at: float


@dataclass
class StubSessionPrep:
    """N prepared topic contexts, LRU, with eviction reported.

    Capacity is deliberately smaller than the number of sections in the stub
    briefing, so the honest answer to "is this topic prepared" is sometimes no.
    """

    config: CopilotConfig
    held: Dict[str, _Prepared] = field(default_factory=dict)
    prepares: int = 0
    evictions: int = 0
    switches: int = 0
    cold_switches: int = 0

    @property
    def capacity(self) -> int:
        return self.config.stub_prepared_capacity

    def prepared_tokens(self, topic_id: str) -> int:
        entry = self.held.get(topic_id)
        return entry.tokens if entry is not None else 0

    async def prepare(self, topic_id: str, prefix_text: str) -> PrepResult:
        started = time.perf_counter()
        await asyncio.sleep(self.config.stub_prepare_ms / 1000.0)
        tokens = max(1, len(prefix_text.split()))
        evicted = self._insert(topic_id, tokens)
        self.prepares += 1
        return PrepResult(
            topic_id=topic_id,
            prepared_tokens=tokens,
            evicted=evicted,
            prep_ms=(time.perf_counter() - started) * 1000.0,
        )

    def note_populated(self, topic_id: str, tokens: int) -> List[str]:
        """Record that ordinary traffic left this prefix in the cache.

        Not a prepare: no cadence, no latency, no ``prepares`` increment. It is
        the side effect of a request that already paid for the prefill, which
        is the only reason a cold switch costs ONE slow hint instead of every
        hint that follows it.
        """
        return self._insert(topic_id, tokens)

    def _insert(self, topic_id: str, tokens: int) -> List[str]:
        evicted: List[str] = []
        self.held[topic_id] = _Prepared(topic_id, tokens, time.monotonic())
        while len(self.held) > self.capacity:
            victim = min(self.held.values(), key=lambda e: e.at)
            if victim.topic_id == topic_id:
                break
            self.held.pop(victim.topic_id, None)
            evicted.append(victim.topic_id)
            self.evictions += 1
        return evicted

    async def switch(self, topic_id: str) -> SwitchResult:
        """Answer immediately, and tell the truth about preparedness.

        A switch never fails and never blocks: choosing which prefix the next
        request uses is a local decision. Whether it is CHEAP depends on
        whether the context is still held, which is what ``prepared`` reports
        and what the hint latency then shows.
        """
        started = time.perf_counter()
        entry = self.held.get(topic_id)
        self.switches += 1
        if entry is None:
            self.cold_switches += 1
        else:
            entry.at = time.monotonic()
        return SwitchResult(
            topic_id=topic_id,
            prepared=entry is not None,
            switch_ms=(time.perf_counter() - started) * 1000.0,
        )

    def report(self) -> Dict[str, Any]:
        return {
            "backend": "stub",
            "capacity": self.capacity,
            "held": sorted(self.held),
            "prepares": self.prepares,
            "evictions": self.evictions,
            "switches": self.switches,
            "cold_switches": self.cold_switches,
        }
