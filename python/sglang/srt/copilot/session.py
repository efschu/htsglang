"""One conversation: two audio tracks, one transcript, a hint stream.

Track identity is carried by the connection, not inferred: the microphone
stream and the tab-capture stream arrive tagged
(``protocol.decode_audio_frame``) and are fed to two independent transcription
streams. That is the whole reason for capturing two sources instead of one
mixed room -- attribution cannot drift because nothing ever mixes.

SERVER-INITIATED, NOT REQUEST/RESPONSE. Everything a reader sees arrives
without the browser asking for it: transcript partials appear while someone is
still talking, a hint appears when the decode finishes, and the background
expander appends to the briefing on its own cadence. So the session OWNS an
event stream: :meth:`CopilotSession.emit` journals an event and fans it out to
every subscribed connection. A connection is a pipe, not a caller.

Two consequences that are load-bearing:

* A hint request never blocks the audio path. It runs as its own task, and
  while it is in flight the session has already told the client so
  (``hint.pending``) -- a read pane that stops moving must always be able to
  say whether it is thinking or broken.
* An outbound queue that overflows drops the CONNECTION, loudly, instead of
  falling behind silently. The journal retains more events than the queue
  holds, so the reconnect replays what the dropped connection missed.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Set

from sglang.srt.copilot.backends import (
    AsrError,
    AsrStream,
    BackendSet,
    TranscriptDelta,
)
from sglang.srt.copilot.briefing import Briefing, parse_briefing
from sglang.srt.copilot.config import CopilotConfig
from sglang.srt.copilot.hints import (
    NOTHING_NEW,
    HintResult,
    expansion_request,
    hint_request,
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


@dataclass
class TranscriptLine:
    line_id: int
    track: Track
    text: str
    at: float
    final: bool = True
    stub: bool = False

    def to_json(self) -> Dict[str, Any]:
        return {
            "line_id": self.line_id,
            "track": self.track.value,
            "text": self.text,
            "at": self.at,
            "final": self.final,
            "stub": self.stub,
        }


@dataclass
class HintTrigger:
    """What the pending or last hint is an answer to.

    Carries the moment the app first had the text, which is the ``t0`` of the
    only latency figure that means anything to a reader: transcript on screen
    to suggestion on screen.
    """

    kind: str
    at: float
    text: str
    line_id: Optional[int] = None
    item_id: Optional[str] = None

    def to_json(self) -> Dict[str, Any]:
        return {
            "source_kind": self.kind,
            "source_line_id": self.line_id,
            "source_item_id": self.item_id,
        }


class Subscription:
    """One connected client's outbound queue: the ONLY path to its socket.

    Carries journalled :class:`Event` objects and, through
    :meth:`offer_raw`, transient payloads the transport owns (a pong, a
    handshake reply, a close request). Everything goes through one queue
    because one socket may only ever have one writer.

    Bounded on purpose. A slow socket must not become unbounded memory in the
    session, and it must not be served a silently thinned event stream either:
    on overflow the subscription is marked and the connection is closed with a
    named reason, which the client answers with a resume from its cursor.
    """

    def __init__(self, maxsize: int) -> None:
        self.queue: "asyncio.Queue[Any]" = asyncio.Queue(maxsize=maxsize)
        self.overflowed = False

    def offer(self, event: Event) -> None:
        self.offer_raw(event)

    def offer_raw(self, item: Any) -> None:
        if self.overflowed:
            return
        try:
            self.queue.put_nowait(item)
        except asyncio.QueueFull:
            self.overflowed = True


class CopilotSession:
    """State machine for one conversation.

    Transport-free and clock-injectable: the whole hint/expander policy can be
    driven at wall-clock zero in tests.
    """

    def __init__(
        self,
        session_id: str,
        config: CopilotConfig,
        backends: BackendSet,
        briefing: Optional[Briefing] = None,
        clock=None,
    ) -> None:
        self.session_id = session_id
        self.config = config
        self.backends = backends
        self.clock = clock or time.monotonic
        self.briefing = briefing or parse_briefing("")
        self.topics = TopicRegistry(config=config)
        self.topics.sync_from_briefing(self.briefing)
        self.journal = Journal(max_events=config.journal_max_events)
        self.transcript: List[TranscriptLine] = []
        self.asr = backends.new_asr(session_id)
        self._streams: Dict[Track, AsrStream] = {}
        self._partials: Dict[Track, str] = {}
        self._next_line_id = 1
        self._last_hint_at = -1e9
        self._last_hint_text = ""
        self._last_expansion_at = -1e9
        self._trigger: Optional[HintTrigger] = None
        self._hint_task: Optional[asyncio.Task] = None
        self._prime_task: Optional[asyncio.Task] = None
        self._expander_task: Optional[asyncio.Task] = None
        self._subs: Set[Subscription] = set()
        self._held: Optional[Set[str]] = None
        self._pipeline_ms: Deque[float] = deque(maxlen=64)
        self.last_activity = self.clock()
        self.dropped_frames = 0
        self.hint_count = 0
        self.hint_failures = 0
        self.expansion_count = 0

    # --- lifecycle --------------------------------------------------------

    def idle_seconds(self) -> float:
        return self.clock() - self.last_activity

    def touch(self) -> None:
        self.last_activity = self.clock()

    def start_background(self) -> None:
        """Start the prepare loop and the briefing expander.

        Idempotent; needs a running loop. Called when the first client attaches
        rather than in ``__init__``, because a session built in a test has no
        loop and must not acquire tasks it never cancels.
        """
        if self._prime_task is None or self._prime_task.done():
            self._prime_task = asyncio.create_task(
                self._prime_loop(), name=f"copilot-prime-{self.session_id}"
            )
        if self._expander_task is None or self._expander_task.done():
            self._expander_task = asyncio.create_task(
                self._expander_loop(), name=f"copilot-expander-{self.session_id}"
            )

    async def aclose(self) -> None:
        for task in (self._prime_task, self._expander_task, self._hint_task):
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
        self._prime_task = None
        self._expander_task = None
        self._hint_task = None
        for stream in self._streams.values():
            await stream.close()
        self._streams.clear()
        await self.asr.aclose()

    async def settle(self) -> None:
        """Await whatever hint work is in flight. For tests and shutdown."""
        task = self._hint_task
        if task is not None and not task.done():
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    # --- the event stream -------------------------------------------------

    def emit(self, kind: ServerFrame, payload: Dict[str, Any]) -> Event:
        event = self.journal.append(Event(kind, payload))
        for sub in list(self._subs):
            sub.offer(event)
        return event

    def subscribe(self) -> Subscription:
        sub = Subscription(self.config.subscriber_queue_max)
        self._subs.add(sub)
        return sub

    def unsubscribe(self, sub: Subscription) -> None:
        self._subs.discard(sub)

    @property
    def clients(self) -> int:
        return len(self._subs)

    # --- tracks -----------------------------------------------------------

    async def open_track(self, track: Track) -> Event:
        if track in self._streams:
            await self._streams[track].close()
        self._streams[track] = await self.asr.open(track, self)
        self.touch()
        return self.emit(
            ServerFrame.TRACK_STATE, {"track": track.value, "state": "open"}
        )

    async def close_track(self, track: Track) -> Event:
        stream = self._streams.pop(track, None)
        if stream is not None:
            await stream.close()
        self._partials.pop(track, None)
        self.touch()
        return self.emit(
            ServerFrame.TRACK_STATE, {"track": track.value, "state": "closed"}
        )

    def track_open(self, track: Track) -> bool:
        return track in self._streams

    # --- briefing ---------------------------------------------------------

    def set_briefing(self, text: str, source: str = "") -> Event:
        self.briefing = parse_briefing(text, source=source)
        self.topics.sync_from_briefing(self.briefing)
        return self.emit(
            ServerFrame.BRIEFING_UPDATE,
            {
                "reason": "briefing.set",
                "briefing": self.briefing.to_json(),
                "markdown": self.briefing.render(),
                "topics": self.topics.states(),
            },
        )

    # --- audio ------------------------------------------------------------

    async def on_audio(self, frame: AudioFrame) -> None:
        """Feed one tagged audio frame into its track's transcription stream.

        A frame for a track that was never opened is DROPPED and counted, not
        silently routed to the other track: a misattributed frame would put the
        far side's words into the user's own column.
        """
        self.touch()
        stream = self._streams.get(frame.track)
        if stream is None:
            self.dropped_frames += 1
            self.emit(
                ServerFrame.ERROR,
                {
                    "stage": "audio",
                    "track": frame.track.value,
                    "message": f"track {frame.track.value} is not open",
                },
            )
            return
        try:
            await stream.append(frame.payload)
        except AsrError as exc:
            await self.on_error(frame.track, exc)

    async def on_commit(self, track: Track) -> None:
        """Close the current utterance on one track.

        The client owns this boundary because the runtime implements no
        server-side VAD (``realtime/session.py:315-321``).
        """
        self.touch()
        stream = self._streams.get(track)
        if stream is None:
            self.emit(
                ServerFrame.ERROR,
                {"stage": "commit", "message": f"track {track.value} is not open"},
            )
            return
        try:
            await stream.commit()
        except AsrError as exc:
            await self.on_error(track, exc)

    # --- AsrEvents --------------------------------------------------------

    async def on_delta(self, delta: TranscriptDelta) -> None:
        """One transcription result from either track."""
        self.touch()
        if delta.final:
            line = TranscriptLine(
                line_id=self._next_line_id,
                track=delta.track,
                text=delta.text,
                at=self.clock(),
                stub=delta.stub,
            )
            self._next_line_id += 1
            self.transcript.append(line)
            if len(self.transcript) > self.config.max_transcript_lines:
                del self.transcript[0]
            self._partials.pop(delta.track, None)
            self.emit(ServerFrame.TRANSCRIPT_LINE, line.to_json())
            self._trigger = HintTrigger(
                kind="line", at=line.at, text=delta.text, line_id=line.line_id
            )
        else:
            self._partials[delta.track] = delta.text
            self.emit(
                ServerFrame.TRANSCRIPT_DELTA,
                {
                    "track": delta.track.value,
                    "text": delta.text,
                    "item_id": delta.item_id,
                    "partial": True,
                    "stub": delta.stub,
                },
            )
            self._trigger = HintTrigger(
                kind="partial",
                at=self.clock(),
                text=delta.text,
                item_id=delta.item_id,
            )
        self._schedule_hint()

    async def on_error(self, track: Track, error: AsrError) -> None:
        """A transcription-side refusal for ONE track.

        Never a reason to drop the conversation: the other track keeps running
        and the user keeps reading.
        """
        self.emit(
            ServerFrame.ERROR,
            {
                "stage": "asr",
                "track": track.value,
                "code": error.code,
                "message": error.message,
            },
        )

    # --- transcript rendering --------------------------------------------

    def transcript_tail(self, lines: int = MAX_TAIL_LINES) -> str:
        """Committed lines plus whatever is being said right now.

        The live partials belong in the prompt: the utterance boundary is a
        client-side silence timer of several hundred milliseconds
        (DESIGN_502 §2.5 link 4), so a hint that waited for clean sentences
        would always be one turn behind the conversation.
        """
        parts = [f"{ln.track.value}: {ln.text}" for ln in self.transcript[-lines:]]
        for track in Track:
            partial = self._partials.get(track)
            if partial:
                parts.append(f"{track.value}: {partial}")
        return "\n".join(parts)

    # --- topics -----------------------------------------------------------

    async def _prime_loop(self) -> None:
        """Prepare topic contexts, then keep refreshing them.

        Two jobs in one loop. The first round is what makes a briefing loaded
        BEFORE the conversation actually prepared -- without it a session with a
        default briefing runs entirely cold and the prepared-context mechanism
        is dead code that nobody notices. The cadence is the second job: a
        re-issued prepare refreshes the prefix's ``last_access_time``, which is
        the second key of the priority eviction strategy
        (``evict_policy.py:41-47``). It does NOT make the prefix un-evictable,
        and this loop never claims it did -- warmth is only ever reported from a
        measurement.
        """
        try:
            while True:
                try:
                    await self.prime_due_topics()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self.emit(
                        ServerFrame.ERROR,
                        {
                            "stage": "prepare",
                            "message": f"{type(exc).__name__}: {exc}",
                            "degraded": True,
                        },
                    )
                await asyncio.sleep(self.config.topic_touch_interval_s)
        except asyncio.CancelledError:
            return

    def prep_capacity(self) -> Optional[int]:
        """How many contexts the backend says it holds, or None if it cannot.

        A backend that does not state a capacity is not assumed to be
        unlimited; it is assumed to be UNKNOWABLE, which is the rig case, and
        then nothing is held back and eviction is discovered by the probe.
        """
        capacity = self.backends.prep.report().get("capacity")
        return capacity if isinstance(capacity, int) and capacity > 0 else None

    async def prime_due_topics(self) -> List[Event]:
        """Prepare the topic contexts that are due."""
        events: List[Event] = []
        for prime in self.topics.due_for_prime(limit=self.prep_capacity()):
            result = await self.backends.prep.prepare(prime.topic_id, prime.prompt)
            self.topics.record_prime(prime.topic_id, result.prepared_tokens)
            for victim in result.evicted:
                self.topics.record_eviction(victim)
            state = self.topics.get(prime.topic_id)
            events.append(
                self.emit(
                    ServerFrame.TOPIC_STATE,
                    {
                        "reason": "primed",
                        "evicted": result.evicted,
                        "prep_ms": round(result.prep_ms, 2),
                        "topics": self.topics.states(),
                        **(state.to_json() if state else {}),
                    },
                )
            )
        synced = self._sync_prep()
        if synced is not None:
            events.append(synced)
        return events

    async def focus_topic(self, topic_id: str) -> Event:
        """Switch the live suggestion context. Raises ``KeyError`` if unknown.

        The hint floor is deliberately cleared: a switch is an explicit request
        for a different answer, and a read pane that keeps showing the previous
        topic's card for another two seconds looks broken.
        """
        state = self.topics.set_focus(topic_id)
        switch = await self.backends.prep.switch(topic_id)
        self._last_hint_at = -1e9
        self._last_hint_text = ""
        event = self.emit(
            ServerFrame.TOPIC_STATE,
            {
                "reason": "focus",
                "prepared": switch.prepared,
                "switch_ms": round(switch.switch_ms, 3),
                "topics": self.topics.states(),
                **state.to_json(),
            },
        )
        self._schedule_hint()
        return event

    def _sync_prep(self) -> Optional[Event]:
        """Reconcile the app's view of prepared topics with the backend's.

        Only backends that can ENUMERATE what they hold report a ``held`` key.
        The rig cannot: a radix prefix is not addressable by topic from
        outside, so there an eviction is discovered exclusively by the warmth
        probe on the next request. This is that difference, made explicit
        rather than assumed away.
        """
        report = self.backends.prep.report()
        held = report.get("held")
        if held is None:
            return None
        held = set(held)
        if self._held is None:
            self._held = held
            return None
        if held == self._held:
            return None
        gone = sorted(self._held - held)
        self._held = held
        for topic_id in gone:
            self.topics.record_eviction(topic_id)
        return self.emit(
            ServerFrame.TOPIC_STATE,
            {
                "reason": "prep",
                "evicted": gone,
                "prep": report,
                "topics": self.topics.states(),
            },
        )

    # --- hints ------------------------------------------------------------

    def hint_due(self, now: Optional[float] = None) -> bool:
        now = self.clock() if now is None else now
        return (now - self._last_hint_at) >= self.config.min_hint_interval_s

    def _schedule_hint(self) -> None:
        """Start a hint decode if one is warranted, without blocking anything.

        Refuses to start a second decode while one is in flight: stacking
        requests would spend the fast lane on answers nobody will read,
        because only the newest one is still about the current topic.
        """
        if self._hint_task is not None and not self._hint_task.done():
            return
        if not self.hint_due():
            return
        trigger = self._trigger
        if trigger is None or not trigger.text:
            return
        if trigger.text == self._last_hint_text:
            return
        topic = self.topics.focused()
        if topic is None or not (self.transcript or self._partials):
            return
        self._last_hint_at = self.clock()
        self._last_hint_text = trigger.text
        self.emit(
            ServerFrame.HINT_PENDING,
            {
                "topic_id": topic.topic_id,
                "at": trigger.at,
                **trigger.to_json(),
            },
        )
        self._hint_task = asyncio.create_task(
            self._run_hint(topic.topic_id, trigger),
            name=f"copilot-hint-{self.session_id}",
        )

    async def _run_hint(self, topic_id: str, trigger: HintTrigger) -> None:
        topic = self.topics.get(topic_id)
        if topic is None:
            return
        req = hint_request(
            self.config, topic.topic_id, topic.prefix_text, self.transcript_tail()
        )
        try:
            result: HintResult = await self.backends.hints.complete(req)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.hint_failures += 1
            self.emit(
                ServerFrame.ERROR,
                {
                    "stage": "hint",
                    "topic_id": topic_id,
                    "message": f"{type(exc).__name__}: {exc}",
                    "degraded": True,
                    **trigger.to_json(),
                },
            )
            return
        warmth = self.topics.observe_usage(topic.topic_id, result.usage)
        self.hint_count += 1
        pipeline_ms = (self.clock() - trigger.at) * 1000.0
        self._pipeline_ms.append(pipeline_ms)
        self.emit(
            ServerFrame.HINT,
            {
                "hint_id": self.hint_count,
                "topic_id": topic.topic_id,
                "bullets": result.bullets(),
                "latency_ms": round(result.latency_ms, 2),
                "pipeline_ms": round(pipeline_ms, 2),
                "desk_fake": result.desk_fake,
                "stub": self.backends.stub,
                "warmth": warmth.value,
                "cached_tokens": topic.last_cached_tokens,
                "primed_tokens": topic.primed_tokens,
                **trigger.to_json(),
            },
        )
        self._sync_prep()

    def latency_report(self) -> Dict[str, Any]:
        """Line-to-hint latency as measured, with its own denominator.

        ``samples`` is stated so a single lucky hint cannot be read as a
        distribution, and the figures are the app's own pipeline only -- with
        the stub backend the dominant term is a configured constant, which the
        report says out loud via ``stub``.
        """
        samples = sorted(self._pipeline_ms)
        if not samples:
            return {
                "samples": 0,
                "p50_ms": None,
                "max_ms": None,
                "stub": self.backends.stub,
            }
        mid = samples[len(samples) // 2]
        return {
            "samples": len(samples),
            "p50_ms": round(mid, 2),
            "max_ms": round(samples[-1], 2),
            "stub": self.backends.stub,
        }

    # --- background expansion --------------------------------------------

    def expansion_due(self, now: Optional[float] = None) -> bool:
        now = self.clock() if now is None else now
        return (now - self._last_expansion_at) >= self.config.expander_interval_s

    async def _expander_loop(self) -> None:
        """Extend the briefing while the conversation runs.

        First round early, then on cadence: the addendum that matters most is
        the one that lands while the topic is still open, and a first round
        45 seconds in is a feature nobody sees in a short call.
        """
        try:
            await asyncio.sleep(self.config.expander_first_delay_s)
            while True:
                try:
                    await self.run_expansion()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self.emit(
                        ServerFrame.ERROR,
                        {
                            "stage": "expander",
                            "message": f"{type(exc).__name__}: {exc}",
                            "degraded": True,
                        },
                    )
                await asyncio.sleep(self.config.expander_interval_s)
        except asyncio.CancelledError:
            return

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
        result: HintResult = await self.backends.hints.complete(req)
        body = result.text.strip()
        if not body or body.upper().startswith(NOTHING_NEW):
            return self.emit(
                ServerFrame.BRIEFING_UPDATE,
                {"reason": "nothing-new", "topic_id": topic.topic_id},
            )
        section = self.briefing.extend_generated(
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
                "added_body": section.body,
                "generated": True,
                "briefing_id": self.briefing.briefing_id,
                "briefing": self.briefing.to_json(),
                "topics": self.topics.states(),
                "desk_fake": result.desk_fake,
                "stub": self.backends.stub,
            },
        )

    # --- reporting --------------------------------------------------------

    def state_json(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            # NOT "seq": a journalled event's ``seq`` identifies that event,
            # while this is the first sequence the client has NOT been given.
            # One name for both is how a resume loses an event per reconnect.
            "next_seq": self.journal.next_seq,
            "backend": self.backends.name,
            "stub": self.backends.stub,
            "tracks": {t.value: (t in self._streams) for t in Track},
            "lines": len(self.transcript),
            "hints": self.hint_count,
            "hint_failures": self.hint_failures,
            "expansions": self.expansion_count,
            "dropped_frames": self.dropped_frames,
            "clients": self.clients,
            "briefing": self.briefing.to_json(),
            "topics": self.topics.states(),
            "residency": self.topics.miss_report(),
            "prep": self.backends.prep.report(),
            "latency": self.latency_report(),
        }


@dataclass
class SessionManager:
    """Session registry with idle collection, on the #466 pattern."""

    config: CopilotConfig
    backends: BackendSet
    clock: Optional[Any] = None
    sessions: Dict[str, CopilotSession] = field(default_factory=dict)
    default_briefing: Optional[Briefing] = None

    def _now(self) -> float:
        return (self.clock or time.monotonic)()

    async def collect(self) -> List[str]:
        dead = [
            sid
            for sid, s in self.sessions.items()
            if s.idle_seconds() > self.config.session_idle_timeout_s
        ]
        for sid in dead:
            session = self.sessions.pop(sid, None)
            if session is not None:
                await session.aclose()
        return dead

    async def open(self, session_id: Optional[str] = None) -> CopilotSession:
        await self.collect()
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
            backends=self.backends,
            briefing=briefing,
            clock=self.clock,
        )
        self.sessions[sid] = session
        return session

    async def close(self, session_id: str) -> bool:
        session = self.sessions.pop(session_id, None)
        if session is None:
            return False
        await session.aclose()
        return True

    async def aclose(self) -> None:
        for session in list(self.sessions.values()):
            await session.aclose()
        self.sessions.clear()
        await self.backends.aclose()
