"""The one seam between the copilot app and whatever produces text.

Everything below the app -- transcription, hint decoding, prepared topic
contexts -- is reached through exactly three protocols declared here. The app
never imports a concrete backend; it asks :func:`build_backend_set` for a set
and talks to the protocols. Switching between the stub set and the rig set is
therefore a CONFIG change (``CopilotConfig.backend``) and touches no app code.

Why this module exists at all, rather than the app calling the ASR client and
the chat client directly: the acceptance for this app runs entirely against
stubs, and a stub that is reached through a different code path than the real
thing proves nothing about the real thing. One seam, two implementations, and
the app cannot tell them apart except by the ``stub`` flag it forwards to the
UI so a human is never fooled either.

PUSH, NOT POLL (load-bearing). :class:`AsrStream` delivers transcript deltas
through a sink callback instead of returning them from ``append``. That is the
shape of the real endpoint: ``/v1/realtime`` is a WebSocket that emits
``conversation.item.input_audio_transcription.delta`` events whenever the
adapter has something, with no relation to the arrival of a particular audio
frame (``entrypoints/openai/realtime/session.py``). A request/response seam
would fit the desk fake and misfit the endpoint, and the difference would only
surface on a booted rig -- which is the class of error this seam exists to
prevent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol

from sglang.srt.copilot.config import CopilotConfig
from sglang.srt.copilot.hints import HintRequest, HintResult
from sglang.srt.copilot.protocol import Track


class AsrError(RuntimeError):
    """An error surfaced by a transcription backend, reported as-is.

    Lives here rather than next to the real client because it is part of the
    seam: the session handles it identically whichever side raised it. Notably
    includes ``too_many_sessions`` (``--asr-max-concurrent-sessions``,
    ``server_args.py:2718``) and ``buffer_overflow``
    (``--asr-max-buffer-seconds``, ``server_args.py:2714``): both are
    conditions this app can reach by design, since it opens two transcription
    streams per conversation and streams continuously.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


class BackendUnavailable(RuntimeError):
    """A requested backend cannot be built, with the missing piece named.

    Raised at launch, never at first use: an app that boots and then refuses
    every utterance is indistinguishable from a broken model.
    """


@dataclass
class TranscriptDelta:
    """One transcription result: a growing partial, or a final line.

    ``stub`` travels with the text all the way to the browser so the UI can
    mark synthetic transcripts. A stub whose output is indistinguishable from
    the real thing is a trap, and this flag is where that is prevented.
    """

    track: Track
    text: str
    item_id: Optional[str] = None
    final: bool = False
    stub: bool = False


class AsrEvents(Protocol):
    """Where a transcription stream delivers what it produces.

    Two channels, because the real endpoint has two: transcription deltas and
    ``error`` frames arrive on the same socket, and an error is NOT the result
    of the particular ``append`` that happened to be in flight. A seam that
    only returned errors from ``append`` would force every backend to invent a
    synchronous moment for an asynchronous failure.
    """

    async def on_delta(self, delta: TranscriptDelta) -> None: ...

    async def on_error(self, track: Track, error: AsrError) -> None: ...


class AsrStream(Protocol):
    """One track's transcription stream. Text leaves through the events sink."""

    async def append(self, pcm: bytes) -> None:
        """Feed one PCM16LE chunk. Never returns transcript text."""
        ...

    async def commit(self) -> None:
        """Close the current utterance.

        The client owns this boundary because the runtime implements no
        server-side VAD (``realtime/session.py:315-321``).
        """
        ...

    async def close(self) -> None: ...


class AsrBackend(Protocol):
    """Opens one transcription stream per track."""

    async def open(self, track: Track, events: AsrEvents) -> AsrStream: ...

    async def aclose(self) -> None: ...


class HintBackend(Protocol):
    """Decodes one hint or one briefing expansion."""

    async def complete(self, req: HintRequest) -> HintResult: ...

    async def aclose(self) -> None: ...


@dataclass
class PrepResult:
    """Outcome of preparing one topic context."""

    topic_id: str
    prepared_tokens: int
    evicted: List[str]
    """Topics this preparation displaced. Empty is a claim, not a default:
    a backend with a bounded cache MUST report what it threw out."""

    prep_ms: float = 0.0


@dataclass
class SwitchResult:
    """Outcome of switching the live suggestion context to another topic."""

    topic_id: str
    prepared: bool
    """Whether the backend still holds a prepared context for this topic. A
    switch to an unprepared topic SUCCEEDS -- it is just not instant."""

    switch_ms: float = 0.0


class SessionPrep(Protocol):
    """Prepared topic contexts, and what a switch between them costs."""

    capacity: int

    async def prepare(self, topic_id: str, prefix_text: str) -> PrepResult: ...

    async def switch(self, topic_id: str) -> SwitchResult: ...

    def report(self) -> Dict[str, Any]: ...


@dataclass
class BackendSet:
    """The three backends the app runs on, plus the name a human sees."""

    name: str
    hints: HintBackend
    prep: SessionPrep
    stub: bool
    _asr_factory: Callable[[str], AsrBackend]

    def new_asr(self, session_id: str) -> AsrBackend:
        """One ASR backend per conversation.

        Per-session rather than shared because a transcription stream carries
        conversation state: in the rig set each track holds its own
        ``/v1/realtime`` socket (one connection is one stream,
        ``realtime/session.py:143-164``), and in the stub set the scripted
        conversation cursor belongs to one conversation.
        """
        return self._asr_factory(session_id)

    async def aclose(self) -> None:
        await self.hints.aclose()


def build_backend_set(config: CopilotConfig) -> BackendSet:
    """Build the backend set named by ``config.backend``.

    Concrete backends are imported inside this function on purpose: the seam
    module must stay importable without the transport dependencies of either
    side, and ``asr_client`` imports the seam types from here.
    """
    if config.backend == "stub":
        from sglang.srt.copilot.stubs import (
            StubAsrBackend,
            StubHints,
            StubSessionPrep,
            load_script,
        )

        script = load_script(config.stub_script)
        prep = StubSessionPrep(config=config)
        return BackendSet(
            name="stub",
            hints=StubHints(config=config, prep=prep),
            prep=prep,
            stub=True,
            _asr_factory=lambda session_id: StubAsrBackend(
                config=config, script=script, session_id=session_id
            ),
        )

    if config.backend == "rig":
        # The chat seam is complete (hints, expansion, prefill-only priming all
        # ride /v1/chat/completions). The transcription transport is NOT: P1
        # built the /v1/realtime protocol state machine
        # (asr_client.RealtimeAsrProtocol) and deliberately stopped short of
        # moving its frames over a socket. Refusing here, at launch, with the
        # missing piece named beats booting an app that shows a transcript pane
        # which can never fill.
        raise BackendUnavailable(
            "backend 'rig' needs the /v1/realtime WebSocket transport, which is "
            "P2 and not built: asr_client.RealtimeAsrProtocol is the protocol "
            "state machine, but no AsrBackend moves its frames over a socket. "
            "Run with --backend stub, or implement RealtimeAsrBackend against "
            "the AsrBackend protocol in backends.py."
        )

    raise BackendUnavailable(
        f"unknown backend {config.backend!r}; known backends are 'stub' and 'rig'"
    )
