# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""The turn pipeline and the session it lives in.

One session is one conversation: a phone connected over the tunnel, a set of
participant languages, a speaker registry, and a journal of everything that
happened. The pipeline per closed segment is

    segment -> ASR (text + detected language)
            -> embed -> speaker assignment + reference buffer update
            -> route: detected language -> target language(s)
            -> MT (streamed, regrouped into clauses)
            -> TTS per clause, conditioned on THAT speaker's reference audio
            -> audio frames to the client

and every arrow emits a journal event, because the two things that go wrong
in the field -- "it did not hear me" and "it used the wrong voice" -- are
indistinguishable from silence unless the client can see the stages.

**Reconnect is a first-class path, not error handling.** The stated top
operational risk is the mobile link dropping mid-turn while roaming. So the
journal is append-only with monotonic sequence numbers, the client tracks the
last sequence it processed, and a reconnect replays from there. Audio frames
are retained under a byte budget; when the cursor is older than what survives,
the client is told explicitly that a gap exists (``resume.gap``) instead of
being handed a silently shortened conversation.

**Turns are serialized per session.** Two overlapping turns would contend for
the same GPU backends and interleave their audio in one earpiece, which is
worse than waiting. A turn that arrives while another is in flight queues,
and the queue is bounded: under sustained overrun the OLDEST queued turn is
dropped, not the newest, because in a live conversation the stale utterance
is the worthless one.
"""

from __future__ import annotations

import asyncio
import dataclasses
import enum
import logging
import time
import uuid
from collections import deque
from typing import (
    Awaitable,
    Callable,
    Deque,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np

from sglang.srt.translator.backends import (
    AudioChunk,
    BackendError,
    Transcript,
)
from sglang.srt.translator.languages import (
    ConversationLanguages,
    LanguageError,
    LanguageMatrix,
)
from sglang.srt.translator.mt import SentenceAccumulator
from sglang.srt.translator.segmenter import (
    Segment,
    SegmentReason,
    SegmenterConfig,
    TurnSegmenter,
    Vad,
)
from sglang.srt.translator.speakers import (
    ReferenceTooShort,
    SpeakerRegistry,
    SpeakerRegistryConfig,
)
from sglang.srt.translator.voices import (
    VoiceAssignment,
    VoiceClass,
    VoiceMode,
    VoicePool,
)

logger = logging.getLogger(__name__)

__all__ = [
    "EventKind",
    "Event",
    "Journal",
    "TranslatorSession",
    "SessionManager",
    "TurnResult",
    "Stopwatch",
]


class EventKind(str, enum.Enum):
    """Everything a client can observe. String-valued so the wire is readable."""

    SESSION_READY = "session.ready"
    SESSION_STATE = "session.state"
    RESUME_GAP = "resume.gap"
    TURN_OPENED = "turn.opened"
    TURN_TRANSCRIPT = "turn.transcript"
    TURN_SPEAKER = "turn.speaker"
    TURN_VOICE = "turn.voice"
    TURN_TRANSLATION = "turn.translation"
    TURN_AUDIO = "turn.audio"
    TURN_DONE = "turn.done"
    TURN_DROPPED = "turn.dropped"
    ERROR = "error"


@dataclasses.dataclass
class Event:
    """One journal entry.

    ``audio`` is kept out of ``payload`` so the journal can account for its
    bytes and evict audio independently of the control events that describe
    it. A client replaying a gap still learns that a turn happened and what it
    said, even when the samples are gone.
    """

    seq: int
    kind: EventKind
    payload: Dict[str, object]
    at: float
    audio: Optional[AudioChunk] = None

    def nbytes(self) -> int:
        return 0 if self.audio is None else int(self.audio.samples.nbytes)

    def to_json(self) -> Dict[str, object]:
        out = {"seq": self.seq, "kind": self.kind.value, "at": round(self.at, 4)}
        out.update(self.payload)
        if self.audio is not None:
            out["sample_rate"] = self.audio.sample_rate
            out["samples"] = len(self.audio.samples)
        return out


class Journal:
    """Bounded append-only event log with a replay cursor.

    Two independent bounds, because the two populations behave differently: a
    control event is tiny and worth keeping for the whole session, an audio
    frame is large and only worth keeping until the client has played it. The
    audio budget therefore evicts audio *payloads* while leaving their events
    in place, so a replay after a long outage yields a complete transcript
    with the samples marked absent.
    """

    def __init__(self, max_events: int = 512, max_audio_bytes: int = 24 << 20) -> None:
        self._events: Deque[Event] = deque(maxlen=max_events)
        self._max_audio_bytes = max_audio_bytes
        self._audio_bytes = 0
        self._seq = 0
        #: Lowest sequence still retrievable. Rises as events fall off the end.
        self.floor = 0

    def __len__(self) -> int:
        return len(self._events)

    @property
    def next_seq(self) -> int:
        return self._seq

    @property
    def audio_bytes(self) -> int:
        return self._audio_bytes

    def append(
        self,
        kind: EventKind,
        payload: Optional[Dict[str, object]] = None,
        audio: Optional[AudioChunk] = None,
        at: Optional[float] = None,
    ) -> Event:
        if len(self._events) == self._events.maxlen and self._events:
            dropped = self._events[0]
            self._audio_bytes -= dropped.nbytes()
            self.floor = dropped.seq + 1
        event = Event(
            seq=self._seq,
            kind=kind,
            payload=dict(payload or {}),
            at=at if at is not None else time.time(),
            audio=audio,
        )
        self._seq += 1
        self._events.append(event)
        self._audio_bytes += event.nbytes()
        self._trim_audio()
        return event

    def _trim_audio(self) -> None:
        """Drop the oldest audio payloads until the byte budget is met."""
        if self._audio_bytes <= self._max_audio_bytes:
            return
        for event in self._events:
            if self._audio_bytes <= self._max_audio_bytes:
                break
            if event.audio is not None:
                self._audio_bytes -= event.nbytes()
                event.audio = None
                event.payload["audio_evicted"] = True

    def since(self, cursor: int) -> Tuple[List[Event], bool]:
        """Events with ``seq >= cursor``, plus whether a gap was skipped."""
        gap = cursor < self.floor
        return [e for e in self._events if e.seq >= cursor], gap


@dataclasses.dataclass
class Stopwatch:
    """Per-turn stage timings, in milliseconds, for the latency budget.

    Recorded on every turn rather than only under a benchmark flag: the
    end-to-end number is the feature's acceptance criterion, and a number that
    is only available when someone remembered to enable it is a number nobody
    has when it matters.
    """

    segment_closed_at: float
    asr_ms: float = 0.0
    embed_ms: float = 0.0
    mt_first_token_ms: float = 0.0
    mt_total_ms: float = 0.0
    tts_first_audio_ms: float = 0.0
    tts_total_ms: float = 0.0
    #: Wall time from segment close to the first synthesized audio frame. This
    #: is THE number: what the listener waits after the speaker stops.
    first_audio_ms: float = 0.0
    total_ms: float = 0.0

    def to_json(self) -> Dict[str, float]:
        return {
            "asr_ms": round(self.asr_ms, 1),
            "embed_ms": round(self.embed_ms, 1),
            "mt_first_token_ms": round(self.mt_first_token_ms, 1),
            "mt_total_ms": round(self.mt_total_ms, 1),
            "tts_first_audio_ms": round(self.tts_first_audio_ms, 1),
            "tts_total_ms": round(self.tts_total_ms, 1),
            "first_audio_ms": round(self.first_audio_ms, 1),
            "total_ms": round(self.total_ms, 1),
        }


@dataclasses.dataclass
class TurnResult:
    """What one completed turn produced. Returned for tests and the harness."""

    turn_id: str
    speaker_id: str
    source_language: str
    source_text: str
    #: target language -> translated text
    translations: Dict[str, str]
    #: target language -> concatenated synthesized audio
    audio: Dict[str, AudioChunk]
    timings: Stopwatch
    used_fallback_voice: bool = False
    speaker_similarity: float = 0.0
    reason: SegmentReason = SegmentReason.PAUSE


class TranslatorSession:
    """One conversation. Owns the segmenter, the speakers and the journal."""

    def __init__(
        self,
        session_id: str,
        asr,
        embedder,
        mt,
        tts,
        matrix: LanguageMatrix,
        conversation: ConversationLanguages,
        segmenter_config: Optional[SegmenterConfig] = None,
        speaker_config: Optional[SpeakerRegistryConfig] = None,
        vad: Optional[Vad] = None,
        journal: Optional[Journal] = None,
        min_reference_seconds: float = 3.0,
        max_queued_turns: int = 2,
        voice_mode: VoiceMode = VoiceMode.CLONE,
        voice_pool: Optional[VoicePool] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        conversation.validate_against(matrix)
        self.session_id = session_id
        self.asr = asr
        self.embedder = embedder
        self.mt = mt
        self.tts = tts
        self.matrix = matrix
        self.conversation = conversation
        self.segmenter = TurnSegmenter(segmenter_config, vad)
        self.speakers = SpeakerRegistry(speaker_config, clock=clock)
        # `journal or Journal()` would be wrong: a freshly constructed journal
        # has zero events and therefore a falsy __len__, so the caller's
        # configured bounds would be silently replaced by the defaults.
        self.journal = journal if journal is not None else Journal()
        self.min_reference_seconds = min_reference_seconds
        self.voice_pool = voice_pool
        # The pool is what makes preset mode possible; asking for it without
        # one is a configuration error, not something to silently ignore, but
        # it must not prevent the session from existing -- so it falls back to
        # clone mode loudly and says so in the journal below.
        self.voice_mode = voice_mode
        if voice_mode is VoiceMode.PRESET and voice_pool is None:
            logger.error(
                "session %s asked for preset voices but no pool is loaded; "
                "falling back to clone mode",
                session_id,
            )
            self.voice_mode = VoiceMode.CLONE
        self._clock = clock
        self._queue: Deque[Segment] = deque()
        self._max_queued = max(1, max_queued_turns)
        self._turn_lock = asyncio.Lock()
        self._closed = False
        self.last_activity = clock()
        self.turns_completed = 0
        self.turns_dropped = 0
        self.journal.append(
            EventKind.SESSION_READY,
            {
                "session_id": session_id,
                "participants": sorted(conversation.participants),
                "languages": matrix.to_json(),
            },
        )

    # -- lifecycle ----------------------------------------------------------

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        self._closed = True
        self._queue.clear()

    def touch(self) -> None:
        self.last_activity = self._clock()

    def idle_seconds(self) -> float:
        return self._clock() - self.last_activity

    def on_reconnect(self) -> None:
        """The transport came back. Reset stream state, keep everything else.

        The segmenter's partial buffer is unrecoverable (the audio in flight
        when the link dropped is gone), but the speaker registry, the
        reference buffers and the journal are the session's real value and
        survive untouched. That asymmetry is the whole point of separating
        them.
        """
        self.segmenter.reset()
        self.touch()

    def state(self) -> Dict[str, object]:
        return {
            "session_id": self.session_id,
            "participants": sorted(self.conversation.participants),
            "speakers": self.speakers.to_json(),
            "turns_completed": self.turns_completed,
            "turns_dropped": self.turns_dropped,
            "queued": len(self._queue),
            "voice_mode": self.voice_mode.value,
            "voice_pool": self.voice_pool.to_json() if self.voice_pool else None,
            "speaking": self.segmenter.speaking,
            "journal_seq": self.journal.next_seq,
            "journal_floor": self.journal.floor,
        }

    # -- enrollment ---------------------------------------------------------

    async def enroll_speaker(
        self,
        label: str,
        audio: AudioChunk,
        text: str = "",
        language: Optional[str] = None,
    ) -> str:
        """Register a known voice from curated samples.

        Returns the speaker id. Enrolled references are exempt from the
        reference-buffer eviction policy (see ``speakers.py``): curated audio
        beats anything a street segment will produce.
        """
        embedding = await self.embedder.embed(audio)
        profile = self.speakers.enroll(
            label=label, embedding=embedding, audio=audio, text=text, language=language
        )
        self.journal.append(
            EventKind.SESSION_STATE,
            {"enrolled": profile.speaker_id, "label": label,
             "reference_seconds": round(profile.reference_seconds(), 2)},
        )
        return profile.speaker_id

    # -- audio in -----------------------------------------------------------

    def push_audio(self, chunk: AudioChunk) -> List[Segment]:
        """Feed transport audio; return segments that closed."""
        if self._closed:
            return []
        self.touch()
        return self.segmenter.feed(chunk)

    def release(self) -> Optional[Segment]:
        """Push-to-talk release."""
        self.touch()
        return self.segmenter.flush(SegmentReason.RELEASED)

    def enqueue(self, segment: Segment) -> bool:
        """Queue a segment for translation. False if it was dropped."""
        while len(self._queue) >= self._max_queued:
            stale = self._queue.popleft()
            self.turns_dropped += 1
            self.journal.append(
                EventKind.TURN_DROPPED,
                {
                    "segment_index": stale.index,
                    "reason": "queue_overrun",
                    "queued": len(self._queue),
            "voice_mode": self.voice_mode.value,
            "voice_pool": self.voice_pool.to_json() if self.voice_pool else None,
                },
            )
        self._queue.append(segment)
        return True

    def pending(self) -> int:
        return len(self._queue)

    async def drain(self) -> List[TurnResult]:
        """Run every queued segment through the pipeline, in order."""
        results: List[TurnResult] = []
        while self._queue and not self._closed:
            segment = self._queue.popleft()
            result = await self.run_turn(segment)
            if result is not None:
                results.append(result)
        return results

    # -- the pipeline -------------------------------------------------------

    async def run_turn(self, segment: Segment) -> Optional[TurnResult]:
        """One segment, all the way to audio. Serialized per session."""
        async with self._turn_lock:
            return await self._run_turn_locked(segment)

    async def _run_turn_locked(self, segment: Segment) -> Optional[TurnResult]:
        turn_id = uuid.uuid4().hex[:12]
        t0 = self._clock()
        watch = Stopwatch(segment_closed_at=t0)
        self.journal.append(
            EventKind.TURN_OPENED,
            {
                "turn_id": turn_id,
                "segment_index": segment.index,
                "duration_s": round(segment.duration_s, 3),
                "reason": segment.reason.value,
            },
        )

        # 1. Recognize. The hint is the last language of the *previous* turn's
        #    speaker, which is only a hint -- the backend contract forbids it
        #    from pinning the result, because pinning would make the direction
        #    routing a configuration rather than an observation.
        try:
            transcript = await self._recognize(segment)
        except BackendError as exc:
            return self._fail(turn_id, "asr", str(exc))
        watch.asr_ms = (self._clock() - t0) * 1000.0

        if not transcript.text.strip():
            self.journal.append(
                EventKind.TURN_DONE,
                {"turn_id": turn_id, "empty": True, "reason": "no_speech_recognized"},
            )
            return None

        self.journal.append(
            EventKind.TURN_TRANSCRIPT,
            {
                "turn_id": turn_id,
                "text": transcript.text,
                "language": transcript.language,
                "language_confidence": round(transcript.language_confidence, 3),
            },
        )

        # 2. Who said it.
        t_embed = self._clock()
        speaker_id, similarity, admitted = await self._identify(segment, transcript)
        watch.embed_ms = (self._clock() - t_embed) * 1000.0
        self.journal.append(
            EventKind.TURN_SPEAKER,
            {
                "turn_id": turn_id,
                "speaker_id": speaker_id,
                "similarity": round(similarity, 3),
                "reference_admitted": admitted,
                "reference_seconds": round(
                    self.speakers.get(speaker_id).reference_seconds(), 2
                )
                if speaker_id
                else 0.0,
            },
        )

        # 3. Which way does it go. Elimination over the conversation's
        #    participant set -- no pair is named anywhere in this file.
        source = self._resolve_source(transcript, speaker_id)
        targets = self.conversation.targets_for(source)
        if not targets:
            self.journal.append(
                EventKind.TURN_DONE,
                {"turn_id": turn_id, "empty": True, "reason": "no_target_language",
                 "source": source},
            )
            return None

        # 4+5. Translate and synthesize, per target.
        translations: Dict[str, str] = {}
        audio_out: Dict[str, AudioChunk] = {}
        used_fallback = False
        first_audio_at: Optional[float] = None

        for target in targets:
            try:
                self.matrix.require_pair(source, target)
            except LanguageError as exc:
                self._fail(turn_id, "routing", str(exc), close=False)
                continue
            try:
                text, audio, fallback, marks = await self._translate_and_speak(
                    turn_id, transcript, source, target, speaker_id, watch
                )
            except BackendError as exc:
                self._fail(turn_id, exc.stage, str(exc), close=False)
                continue
            translations[target] = text
            if audio is not None:
                audio_out[target] = audio
            used_fallback = used_fallback or fallback
            if marks is not None and (first_audio_at is None or marks < first_audio_at):
                first_audio_at = marks

        if first_audio_at is not None:
            watch.first_audio_ms = (first_audio_at - t0) * 1000.0
        watch.total_ms = (self._clock() - t0) * 1000.0

        if translations:
            # Only the first target feeds the MT history: a multi-target fan-out
            # would otherwise stack N assistant turns per user turn and skew the
            # context window towards whichever language happened to sort first.
            first_target = sorted(translations)[0]
            remember = getattr(self.mt, "remember", None)
            if callable(remember):
                remember(transcript.text, translations[first_target])

        self.turns_completed += 1
        self.journal.append(
            EventKind.TURN_DONE,
            {
                "turn_id": turn_id,
                "speaker_id": speaker_id,
                "source": source,
                "targets": sorted(translations),
                "fallback_voice": used_fallback,
                "timings": watch.to_json(),
            },
        )
        return TurnResult(
            turn_id=turn_id,
            speaker_id=speaker_id,
            source_language=source,
            source_text=transcript.text,
            translations=translations,
            audio=audio_out,
            timings=watch,
            used_fallback_voice=used_fallback,
            speaker_similarity=similarity,
            reason=segment.reason,
        )

    # -- stages -------------------------------------------------------------

    async def _recognize(self, segment: Segment) -> Transcript:
        hint = None
        profiles = self.speakers.profiles()
        if profiles:
            recent = max(profiles, key=lambda p: p.last_seen)
            hint = recent.last_language
        return await self.asr.transcribe(segment.audio, hint_language=hint)

    async def _identify(
        self, segment: Segment, transcript: Transcript
    ) -> Tuple[str, float, bool]:
        min_s = getattr(self.embedder, "min_seconds", 0.0)
        if segment.duration_s < min_s:
            # Too short to embed. Attributing it to the most recent speaker is
            # the right guess in a turn-taking conversation (short utterances
            # are backchannels from whoever just spoke) and it must NOT be
            # admitted to any reference buffer, which the registry enforces by
            # never seeing it.
            profiles = self.speakers.profiles()
            if profiles:
                recent = max(profiles, key=lambda p: p.last_seen)
                return recent.speaker_id, 0.0, False
            # Nobody known yet and nothing embeddable: mint a provisional id
            # so the turn still has a speaker to attach to.
            return "speaker-unknown", 0.0, False
        try:
            embedding = await self.embedder.embed(segment.audio)
        except BackendError:
            profiles = self.speakers.profiles()
            if profiles:
                recent = max(profiles, key=lambda p: p.last_seen)
                return recent.speaker_id, 0.0, False
            return "speaker-unknown", 0.0, False
        profile, similarity, admitted = self.speakers.assign(
            embedding=embedding,
            audio=segment.audio,
            text=transcript.text,
            language=transcript.language,
            language_confidence=transcript.language_confidence,
        )
        return profile.speaker_id, similarity, admitted

    def _resolve_source(self, transcript: Transcript, speaker_id: str) -> str:
        """Detected language, with a low-confidence fallback to speaker history.

        People do not usually switch language mid-conversation, so when the
        identifier is unsure the speaker's last confident language is a better
        estimate than a coin flip -- and a coin flip here sends the
        translation in the wrong direction, which is the single most confusing
        failure the user can experience.
        """
        if transcript.language_confidence >= 0.5:
            return transcript.language
        try:
            profile = self.speakers.get(speaker_id)
        except KeyError:
            return transcript.language
        return profile.last_language or transcript.language

    async def _translate_and_speak(
        self,
        turn_id: str,
        transcript: Transcript,
        source: str,
        target: str,
        speaker_id: str,
        watch: Stopwatch,
    ) -> Tuple[str, Optional[AudioChunk], bool, Optional[float]]:
        voice = self._choose_voice(speaker_id, target)
        reference, reference_text = voice.reference, voice.reference_text
        fallback = voice.downgraded
        self.journal.append(
            EventKind.TURN_VOICE,
            {"turn_id": turn_id, "target": target, "speaker_id": speaker_id,
             **voice.to_json()},
        )

        accumulator = SentenceAccumulator()
        pieces: List[str] = []
        chunks: List[AudioChunk] = []
        first_audio_at: Optional[float] = None
        mt_start = self._clock()
        first_token_at: Optional[float] = None

        async def speak(unit: str) -> None:
            nonlocal first_audio_at
            if not unit.strip() or reference is None:
                return
            tts_start = self._clock()
            async for piece in self.tts.synthesize(
                unit, target, reference, reference_text
            ):
                if first_audio_at is None:
                    first_audio_at = self._clock()
                    watch.tts_first_audio_ms = (first_audio_at - tts_start) * 1000.0
                chunks.append(piece)
                self.journal.append(
                    EventKind.TURN_AUDIO,
                    {"turn_id": turn_id, "target": target, "speaker_id": speaker_id},
                    audio=piece,
                )

        async for delta in self.mt.translate_stream(transcript.text, source, target):
            if first_token_at is None:
                first_token_at = self._clock()
                watch.mt_first_token_ms = (first_token_at - mt_start) * 1000.0
            for unit in accumulator.push(delta):
                pieces.append(unit)
                self.journal.append(
                    EventKind.TURN_TRANSLATION,
                    {"turn_id": turn_id, "target": target, "text": unit,
                     "partial": True},
                )
                await speak(unit)
        tail = accumulator.flush()
        watch.mt_total_ms = (self._clock() - mt_start) * 1000.0
        if tail:
            pieces.append(tail)
            self.journal.append(
                EventKind.TURN_TRANSLATION,
                {"turn_id": turn_id, "target": target, "text": tail, "partial": True},
            )
            await speak(tail)

        text = " ".join(p for p in pieces if p).strip()
        self.journal.append(
            EventKind.TURN_TRANSLATION,
            {"turn_id": turn_id, "target": target, "text": text, "partial": False},
        )
        if first_audio_at is not None:
            watch.tts_total_ms = (self._clock() - first_audio_at) * 1000.0

        merged: Optional[AudioChunk] = None
        if chunks:
            rate = chunks[0].sample_rate
            merged = AudioChunk(
                np.concatenate([c.samples for c in chunks]), rate
            )
        return text, merged, fallback, first_audio_at

    def set_voice_mode(self, mode: VoiceMode) -> VoiceMode:
        """Switch the session's voice mode at runtime.

        Preset mode without a pool stays refused rather than silently
        accepted, because a client that thinks it switched and did not would
        have no way to tell from the audio.
        """
        if mode is VoiceMode.PRESET and self.voice_pool is None:
            raise RuntimeError(
                "preset voice mode requires a preset pool; none is loaded in "
                "this deployment"
            )
        self.voice_mode = mode
        self.journal.append(
            EventKind.SESSION_STATE, {"voice_mode": mode.value}
        )
        return self.voice_mode

    def override_voice_class(self, speaker_id: str, voice_class: VoiceClass) -> None:
        """Correct the heuristic class for one speaker and re-assign them."""
        if self.voice_pool is None:
            raise RuntimeError("no preset pool is loaded in this deployment")
        self.voice_pool.override_class(speaker_id, voice_class)
        self.journal.append(
            EventKind.SESSION_STATE,
            {"speaker_id": speaker_id, "voice_class_override": voice_class.value},
        )

    def _speaker_audio(self, speaker_id: str) -> Optional[AudioChunk]:
        """Whatever audio we hold for this speaker, for voice classification."""
        try:
            profile = self.speakers.get(speaker_id)
        except KeyError:
            return None
        if not profile.references:
            return None
        return profile.reference_audio(self.tts.sample_rate)

    def _choose_voice(self, speaker_id: str, target: str) -> VoiceAssignment:
        """Which voice this turn is spoken in, and why.

        Three outcomes, all of them recorded on the turn event:

        * preset mode was asked for -- assign a sticky, class-matched preset;
        * clone mode with a usable reference -- the speaker's own voice;
        * clone mode without one -- DOWNGRADE. A preset is preferred to
          borrowing another participant's voice, because a preset is honestly
          artificial while a borrowed voice attributes words to the wrong
          person. Borrowing survives only as the last resort when no pool is
          loaded at all, and both are marked.
        """
        if self.voice_mode is VoiceMode.PRESET and self.voice_pool is not None:
            return self.voice_pool.choose(
                speaker_id, target, self._speaker_audio(speaker_id)
            )

        try:
            audio, text = self.speakers.reference_for(
                speaker_id, self.min_reference_seconds, self.tts.sample_rate
            )
            return VoiceAssignment(
                mode=VoiceMode.CLONE, reference=audio, reference_text=text
            )
        except (KeyError, ReferenceTooShort) as exc:
            shortfall = (
                f"{exc.have_s:.1f}s of {exc.need_s:.1f}s reference"
                if isinstance(exc, ReferenceTooShort)
                else "no reference"
            )

        if self.voice_pool is not None:
            return self.voice_pool.choose(
                speaker_id,
                target,
                self._speaker_audio(speaker_id),
                downgraded=True,
                reason=f"reference too short to clone ({shortfall})",
            )

        best = None
        for profile in self.speakers.profiles():
            if profile.speaker_id == speaker_id:
                continue
            if profile.reference_seconds() >= self.min_reference_seconds:
                if best is None or profile.reference_seconds() > best.reference_seconds():
                    best = profile
        if best is None:
            return VoiceAssignment(
                mode=VoiceMode.CLONE,
                reference=None,
                reference_text="",
                downgraded=True,
                reason=f"no voice available ({shortfall}, no preset pool)",
            )
        return VoiceAssignment(
            mode=VoiceMode.CLONE,
            reference=best.reference_audio(self.tts.sample_rate),
            reference_text=best.reference_text(),
            downgraded=True,
            reason=(
                f"reference too short to clone ({shortfall}); borrowed "
                f"{best.speaker_id}'s voice, no preset pool loaded"
            ),
        )

    def _fail(
        self, turn_id: str, stage: str, message: str, close: bool = True
    ) -> None:
        logger.warning("session %s turn %s failed in %s: %s",
                       self.session_id, turn_id, stage, message)
        self.journal.append(
            EventKind.ERROR,
            {"turn_id": turn_id, "stage": stage, "message": message},
        )
        if close:
            self.journal.append(
                EventKind.TURN_DONE, {"turn_id": turn_id, "failed": True}
            )
        return None


class SessionManager:
    """Sessions by id, with idle collection and a hard cap.

    The cap exists because each session pins reference buffers and a journal;
    an unbounded map plus a flaky mobile link that mints a new session on every
    reconnect is a memory leak with a user-facing trigger. Reconnect therefore
    RESUMES by id rather than creating, and only a genuinely new conversation
    allocates.
    """

    def __init__(
        self,
        factory: Callable[[str, ConversationLanguages], TranslatorSession],
        max_sessions: int = 8,
        idle_timeout_s: float = 300.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._factory = factory
        self._max = max_sessions
        self._idle_timeout = idle_timeout_s
        self._clock = clock
        self._sessions: Dict[str, TranslatorSession] = {}

    def __len__(self) -> int:
        return len(self._sessions)

    def ids(self) -> Tuple[str, ...]:
        return tuple(self._sessions)

    def collect(self) -> List[str]:
        """Drop idle sessions. Returns the ids collected."""
        dead = [
            sid
            for sid, session in self._sessions.items()
            if session.idle_seconds() > self._idle_timeout
        ]
        for sid in dead:
            self._sessions.pop(sid).close()
        return dead

    def get(self, session_id: str) -> Optional[TranslatorSession]:
        session = self._sessions.get(session_id)
        if session is not None:
            session.touch()
        return session

    def open(
        self,
        conversation: ConversationLanguages,
        session_id: Optional[str] = None,
    ) -> TranslatorSession:
        """Create, or return the existing session with this id (reconnect)."""
        if session_id and session_id in self._sessions:
            session = self._sessions[session_id]
            session.on_reconnect()
            return session
        self.collect()
        if len(self._sessions) >= self._max:
            raise RuntimeError(
                f"session limit reached ({self._max}); "
                f"{len(self._sessions)} active, none idle enough to collect"
            )
        sid = session_id or uuid.uuid4().hex[:12]
        session = self._factory(sid, conversation)
        self._sessions[sid] = session
        return session

    def close(self, session_id: str) -> bool:
        session = self._sessions.pop(session_id, None)
        if session is None:
            return False
        session.close()
        return True


async def run_conversation(
    session: TranslatorSession,
    audio: Sequence[AudioChunk],
) -> List[TurnResult]:
    """Feed a whole scripted conversation through a session.

    The harness the hermetic tests and the offline latency measurement both
    use. Kept here rather than in the tests so the two cannot drift.
    """
    results: List[TurnResult] = []
    for chunk in audio:
        for segment in session.push_audio(chunk):
            result = await session.run_turn(segment)
            if result is not None:
                results.append(result)
    tail = session.release()
    if tail is not None:
        result = await session.run_turn(tail)
        if result is not None:
            results.append(result)
    return results


async def maybe_await(value):
    """Await ``value`` if it is awaitable, else return it.

    Small helper for adapters that may be sync or async; keeps the pipeline
    from having to know which kind of backend it was handed.
    """
    if isinstance(value, Awaitable):
        return await value
    return value
