"""One conversation: two audio tracks, one transcript, a hint stream.

Track identity is carried by the connection, not inferred: the microphone
stream and the tab-capture stream arrive tagged
(``protocol.decode_audio_frame``) and are fed to two independent ASR streams.
That is the whole reason for capturing two sources instead of one mixed room --
attribution cannot drift because nothing ever mixes.
"""

from __future__ import annotations

import inspect
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from sglang.srt.copilot.asr_client import AsrError, DeskFakeAsr, TranscriptDelta
from sglang.srt.copilot.briefing import Briefing, parse_briefing
from sglang.srt.copilot.config import CopilotConfig
from sglang.srt.copilot.hints import (
    NOTHING_NEW,
    HintResult,
    expansion_request,
    hint_request,
    prime_chat_request,
)
from sglang.srt.copilot.protocol import (
    AudioFrame,
    Event,
    Journal,
    ServerFrame,
    Track,
)
from sglang.srt.copilot.topics import TopicRegistry

MAX_TAIL_LINES = 12


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


@dataclass
class TranscriptLine:
    line_id: int
    track: Track
    text: str
    at: float
    final: bool = True

    def to_json(self) -> Dict[str, Any]:
        return {
            "line_id": self.line_id,
            "track": self.track.value,
            "text": self.text,
            "at": self.at,
            "final": self.final,
        }


class CopilotSession:
    """State machine for one conversation.

    Deliberately transport-free and clock-injectable so the whole hint/expander
    policy can be driven at wall-clock zero in tests.
    """

    def __init__(
        self,
        session_id: str,
        config: CopilotConfig,
        hint_backend: Any,
        briefing: Optional[Briefing] = None,
        asr_factory: Optional[Callable[[Track], Any]] = None,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        self.session_id = session_id
        self.config = config
        self.hint_backend = hint_backend
        self.clock = clock or time.monotonic
        self.briefing = briefing or parse_briefing("")
        self.topics = TopicRegistry(config=config)
        self.topics.sync_from_briefing(self.briefing)
        self.journal = Journal(max_events=config.journal_max_events)
        self.transcript: List[TranscriptLine] = []
        self._next_line_id = 1
        self._asr_factory = asr_factory or (lambda track: DeskFakeAsr(track))
        self._asr: Dict[Track, Any] = {}
        self._last_hint_at = -1e9
        self._last_expansion_at = -1e9
        self.last_activity = self.clock()
        self.dropped_frames = 0
        self.hint_count = 0
        self.expansion_count = 0

    # --- lifecycle --------------------------------------------------------

    def idle_seconds(self) -> float:
        return self.clock() - self.last_activity

    def touch(self) -> None:
        self.last_activity = self.clock()

    def emit(self, kind: ServerFrame, payload: Dict[str, Any]) -> Event:
        return self.journal.append(Event(kind, payload))

    def open_track(self, track: Track) -> Event:
        self._asr[track] = self._asr_factory(track)
        self.touch()
        return self.emit(
            ServerFrame.TRACK_STATE, {"track": track.value, "state": "open"}
        )

    def close_track(self, track: Track) -> Event:
        self._asr.pop(track, None)
        self.touch()
        return self.emit(
            ServerFrame.TRACK_STATE, {"track": track.value, "state": "closed"}
        )

    def track_open(self, track: Track) -> bool:
        return track in self._asr

    # --- briefing ---------------------------------------------------------

    def set_briefing(self, text: str, source: str = "") -> Event:
        self.briefing = parse_briefing(text, source=source)
        self.topics.sync_from_briefing(self.briefing)
        return self.emit(
            ServerFrame.BRIEFING_UPDATE,
            {
                "reason": "briefing.set",
                "briefing": self.briefing.to_json(),
                "topics": self.topics.states(),
            },
        )

    # --- audio ------------------------------------------------------------

    async def on_audio(self, frame: AudioFrame) -> List[Event]:
        """Feed one tagged audio frame into its track's ASR stream.

        A frame for a track that was never opened is DROPPED and counted, not
        silently routed to the other track: a misattributed frame would put the
        far side's words into the user's own column.
        """
        self.touch()
        asr = self._asr.get(frame.track)
        if asr is None:
            self.dropped_frames += 1
            return [
                self.emit(
                    ServerFrame.ERROR,
                    {
                        "stage": "audio",
                        "message": f"track {frame.track.value} is not open",
                    },
                )
            ]
        try:
            deltas = await _maybe_await(asr.append(frame.payload))
        except AsrError as exc:
            # An ASR-side refusal is a named event for this ONE track, not a
            # reason to drop the conversation: the other track keeps running
            # and the user keeps reading. (Found by the can-fail run: an
            # uncaught AsrError killed the whole WebSocket.)
            return [
                self.emit(
                    ServerFrame.ERROR,
                    {
                        "stage": "asr",
                        "track": frame.track.value,
                        "code": exc.code,
                        "message": exc.message,
                    },
                )
            ]
        return await self._consume_deltas(deltas or [])

    async def on_commit(self, track: Track) -> List[Event]:
        """Close the current utterance on one track.

        The client owns this boundary because the runtime implements no
        server-side VAD (``realtime/session.py:315-321``).
        """
        self.touch()
        asr = self._asr.get(track)
        if asr is None:
            return [
                self.emit(
                    ServerFrame.ERROR,
                    {"stage": "commit", "message": f"track {track.value} is not open"},
                )
            ]
        try:
            deltas = await _maybe_await(asr.commit())
        except AsrError as exc:
            return [
                self.emit(
                    ServerFrame.ERROR,
                    {
                        "stage": "asr",
                        "track": track.value,
                        "code": exc.code,
                        "message": exc.message,
                    },
                )
            ]
        return await self._consume_deltas(deltas or [])

    async def _consume_deltas(self, deltas: List[TranscriptDelta]) -> List[Event]:
        events: List[Event] = []
        got_final = False
        for delta in deltas:
            if delta.final:
                line = TranscriptLine(
                    line_id=self._next_line_id,
                    track=delta.track,
                    text=delta.text,
                    at=self.clock(),
                )
                self._next_line_id += 1
                self.transcript.append(line)
                if len(self.transcript) > self.config.max_transcript_lines:
                    del self.transcript[0]
                events.append(self.emit(ServerFrame.TRANSCRIPT_LINE, line.to_json()))
                got_final = True
            else:
                events.append(
                    self.emit(
                        ServerFrame.TRANSCRIPT_DELTA,
                        {
                            "track": delta.track.value,
                            "text": delta.text,
                            "item_id": delta.item_id,
                            "partial": True,
                        },
                    )
                )
        if got_final:
            hint_event = await self.maybe_hint()
            if hint_event is not None:
                events.append(hint_event)
        return events

    # --- transcript rendering --------------------------------------------

    def transcript_tail(self, lines: int = MAX_TAIL_LINES) -> str:
        tail = self.transcript[-lines:]
        return "\n".join(f"{ln.track.value}: {ln.text}" for ln in tail)

    # --- topics -----------------------------------------------------------

    async def prime_due_topics(self) -> List[Event]:
        """Send the prefill-only priming requests that are due."""
        events: List[Event] = []
        for prime in self.topics.due_for_prime():
            req = prime_chat_request(self.config, prime.topic_id, prime.prompt)
            result: HintResult = await self.hint_backend.complete(req)
            self.topics.record_prime(prime.topic_id, result.prompt_tokens)
            state = self.topics.get(prime.topic_id)
            events.append(
                self.emit(
                    ServerFrame.TOPIC_STATE,
                    {
                        "reason": "primed",
                        **(state.to_json() if state else {}),
                    },
                )
            )
        return events

    def focus_topic(self, topic_id: str) -> Event:
        state = self.topics.set_focus(topic_id)
        return self.emit(
            ServerFrame.TOPIC_STATE, {"reason": "focus", **state.to_json()}
        )

    # --- hints ------------------------------------------------------------

    def hint_due(self, now: Optional[float] = None) -> bool:
        now = self.clock() if now is None else now
        return (now - self._last_hint_at) >= self.config.min_hint_interval_s

    async def maybe_hint(self) -> Optional[Event]:
        if not self.hint_due():
            return None
        topic = self.topics.focused()
        if topic is None or not self.transcript:
            return None
        self._last_hint_at = self.clock()
        req = hint_request(
            self.config, topic.topic_id, topic.prefix_text, self.transcript_tail()
        )
        result: HintResult = await self.hint_backend.complete(req)
        warmth = self.topics.observe_usage(topic.topic_id, result.usage)
        self.hint_count += 1
        return self.emit(
            ServerFrame.HINT,
            {
                "hint_id": self.hint_count,
                "topic_id": topic.topic_id,
                "bullets": result.bullets(),
                "latency_ms": round(result.latency_ms, 2),
                "desk_fake": result.desk_fake,
                "warmth": warmth.value,
                "cached_tokens": topic.last_cached_tokens,
                "primed_tokens": topic.primed_tokens,
            },
        )

    # --- background expansion --------------------------------------------

    def expansion_due(self, now: Optional[float] = None) -> bool:
        now = self.clock() if now is None else now
        return (now - self._last_expansion_at) >= self.config.expander_interval_s

    async def run_expansion(self) -> Optional[Event]:
        """One background briefing-expansion round.

        Runs as an ordinary request one priority tier below the live hints (no
        ``lane``, heavy-tier default priority) rather than in a second runtime
        lane: #274 is unreachable in the homogeneous case (its admission needs
        an explicit non-uniform ``--rank-tp-ratio``, ``server_args.py:9497``
        and ``:9619``) and #347's workbench runs only when the rig is idle
        (``server_args.py:5582-5587``), which a live conversation is not.
        """
        topic = self.topics.focused()
        if topic is None or not self.transcript:
            return None
        self._last_expansion_at = self.clock()
        req = expansion_request(
            self.config, topic.topic_id, topic.prefix_text, self.transcript_tail()
        )
        result: HintResult = await self.hint_backend.complete(req)
        body = result.text.strip()
        if not body or body.upper().startswith(NOTHING_NEW):
            return self.emit(
                ServerFrame.BRIEFING_UPDATE,
                {"reason": "nothing-new", "topic_id": topic.topic_id},
            )
        section = self.briefing.append_generated(
            title=f"{topic.title} — live addendum",
            body=body,
            provenance=f"copilot expansion, session {self.session_id}",
        )
        # The topic prefix of the SOURCE topic is intentionally left untouched:
        # rewriting it would move its token path and throw away the residency
        # it just measured. The new material becomes its own topic.
        self.topics.sync_from_briefing(self.briefing)
        self.expansion_count += 1
        return self.emit(
            ServerFrame.BRIEFING_UPDATE,
            {
                "reason": "expanded",
                "topic_id": topic.topic_id,
                "added_anchor": section.anchor,
                "added_title": section.title,
                "generated": True,
                "briefing_id": self.briefing.briefing_id,
                "desk_fake": result.desk_fake,
            },
        )

    # --- reporting --------------------------------------------------------

    def state_json(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "seq": self.journal.next_seq,
            "tracks": {t.value: (t in self._asr) for t in Track},
            "lines": len(self.transcript),
            "hints": self.hint_count,
            "expansions": self.expansion_count,
            "dropped_frames": self.dropped_frames,
            "briefing": self.briefing.to_json(),
            "topics": self.topics.states(),
            "residency": self.topics.miss_report(),
        }


@dataclass
class SessionManager:
    """Session registry with idle collection, on the #466 pattern."""

    config: CopilotConfig
    hint_backend: Any
    asr_factory: Optional[Callable[[Track], Any]] = None
    clock: Optional[Callable[[], float]] = None
    sessions: Dict[str, CopilotSession] = field(default_factory=dict)
    default_briefing: Optional[Briefing] = None

    def _now(self) -> float:
        return (self.clock or time.monotonic)()

    def collect(self) -> List[str]:
        dead = [
            sid
            for sid, s in self.sessions.items()
            if s.idle_seconds() > self.config.session_idle_timeout_s
        ]
        for sid in dead:
            self.sessions.pop(sid, None)
        return dead

    def open(self, session_id: Optional[str] = None) -> CopilotSession:
        self.collect()
        if session_id and session_id in self.sessions:
            session = self.sessions[session_id]
            session.touch()
            return session
        if len(self.sessions) >= self.config.max_sessions:
            raise RuntimeError(
                f"session limit reached ({self.config.max_sessions}); "
                "close a conversation before opening another"
            )
        import uuid

        sid = session_id or uuid.uuid4().hex[:12]
        # Re-parse rather than share: a session's background expander APPENDS
        # to its own briefing, and a shared object would leak one
        # conversation's generated sections into another's.
        briefing = (
            parse_briefing(
                self.default_briefing.render(), source=self.default_briefing.source
            )
            if self.default_briefing is not None
            else None
        )
        session = CopilotSession(
            session_id=sid,
            config=self.config,
            hint_backend=self.hint_backend,
            briefing=briefing,
            asr_factory=self.asr_factory,
            clock=self.clock,
        )
        self.sessions[sid] = session
        return session

    def close(self, session_id: str) -> bool:
        return self.sessions.pop(session_id, None) is not None
