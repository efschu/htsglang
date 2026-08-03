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
    TTS_QUEUE_WAIT_S,
    AudioChunk,
    BackendError,
    Transcript,
)
from sglang.srt.translator.languages import (
    ConversationLanguages,
    LanguageError,
    LanguageMatrix,
    RoutingTable,
)
from sglang.srt.translator.mt import SentenceAccumulator
from sglang.srt.translator.name_hints import (
    EXTRACTION_PROMPT,
    KIND_SELF,
    NameSuggestion,
    SuggestionTracker,
    looks_like_naming,
    parse_candidates,
)
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
    split_points_by_dispersion,
)
from sglang.srt.translator.transcript_log import (
    CONFIDENCE_EXACT,
    CONFIDENCE_UNCERTAIN,
    ORIGIN_AUTO,
    ORIGIN_MANUAL,
    TranscriptLine,
    TranscriptLog,
)
from sglang.srt.translator.voices import (
    OutputMode,
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
    TURN_SPLIT = "turn.split"
    TURN_UNROUTED = "turn.unrouted"
    TURN_VOICE = "turn.voice"
    TURN_TRANSLATION = "turn.translation"
    TURN_AUDIO = "turn.audio"
    TURN_DONE = "turn.done"
    TURN_DROPPED = "turn.dropped"
    #: This turn is behind another conversation's synthesis (§17.8.2). One
    #: talker serves every session, so the wait is real and bounded by the
    #: other turn -- the event exists so the client can say that instead of
    #: appearing to have frozen.
    TURN_QUEUED = "turn.queued"
    #: The user pressed stop. Everything not yet spoken is abandoned: the
    #: in-flight turn is cancelled and the queued segments are dropped. It is
    #: a journal event and not only a live ack because the record must show
    #: WHY a turn stops mid-sentence -- a line that simply ends looks like a
    #: defect, and this is the difference between a cut-off turn and a lost
    #: one.
    PLAYBACK_STOPPED = "playback.stopped"
    # The written record (§17.2). Separate from TURN_TRANSCRIPT, which
    # reports one stage of one turn: these carry the durable line the reader
    # scrolls, and they are the only events that may mutate history.
    TRANSCRIPT_LINE = "transcript.line"
    TRANSCRIPT_UPDATE = "transcript.update"
    TRANSCRIPT_CLEARED = "transcript.cleared"
    #: A name the LLM heard, waiting for the user to confirm it (§17.3).
    #: Never applied automatically -- the event IS the chip.
    SPEAKER_SUGGESTION = "speaker.suggestion"
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
    #: Of ``tts_first_audio_ms``, how much was spent WAITING for a synthesizer
    #: another conversation was holding. Separated because the two halves have
    #: different fixes: compute wants a faster talker, waiting wants capacity.
    tts_wait_ms: float = 0.0
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
            "tts_wait_ms": round(self.tts_wait_ms, 1),
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
        output_mode: OutputMode = OutputMode.VOICE,
        voice_pool: Optional[VoicePool] = None,
        speaker_change_detection: bool = True,
        speaker_change_threshold: float = 0.62,
        # 2.5 s, not 1.5 s, and the number is measured rather than chosen.
        # `probe_speaker_change.py --pool` over the 17-voice pool, adjacent
        # windows of ONE voice against windows of DIFFERENT voices:
        #
        #   window | within-speaker min | between-speaker max | 0.62 cuts
        #   1.5 s  | 0.392              | 0.679               | 27 of 60 (45%)
        #   2.0 s  | 0.515              | 0.694               |  7 of 40
        #   2.5 s  | 0.624              | 0.734               |  0 of 26
        #   3.0 s  | 0.695              | 0.738               |  0 of 18
        #
        # At 1.5 s this embedder's own within-speaker similarity falls to
        # 0.392 -- BELOW the threshold meant to separate different people --
        # so nearly half of all same-speaker window pairs were being cut. That
        # is what split one German speaker into `speaker-1` and `speaker-2` in
        # the first front-door run, and it is why the reference buffer could
        # never fill: each half fell under `min_slice_s` and nothing was ever
        # admitted. The threshold was never the root cause; the window was.
        speaker_change_window_s: float = 2.5,
        # Must stay >= 2 x window, or `count < 2` and the re-cut never runs.
        # The cost of the wider window is stated plainly: two people who
        # alternate inside a segment shorter than this are no longer split.
        # That is the deliberate trade -- the measurement says a 1.5 s window
        # cannot make that call reliably, and splitting on evidence this noisy
        # is worse than not splitting, because it corrupts identity for
        # everyone rather than protecting the rare back-to-back case.
        speaker_change_min_segment_s: float = 5.0,
        speaker_change_min_piece_s: float = 0.8,
        #: Assembled turn audio below this peak counts as silence.
        #: 1e-3 is two orders below the quietest real synthesis
        #: measured (peak 0.31 worst of five arms) and above any
        #: dither floor, so it separates the two populations
        #: without sitting near either.
        silence_peak_floor: float = 1e-3,
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
        self.speaker_change_detection = speaker_change_detection
        self.speaker_change_threshold = speaker_change_threshold
        self.speaker_change_window_s = speaker_change_window_s
        self.speaker_change_min_segment_s = speaker_change_min_segment_s
        self.speaker_change_min_piece_s = speaker_change_min_piece_s
        self.silence_peak_floor = silence_peak_floor
        # The "must be >= 2 x window" note above is a claim, so it is checked
        # rather than trusted. Violated, the re-cut silently never runs: the
        # `count < 2` guard returns early on every segment and the protection
        # it exists to give is gone with no error anywhere.
        if speaker_change_detection and (
            speaker_change_min_segment_s < 2 * speaker_change_window_s
        ):
            raise ValueError(
                f"speaker_change_min_segment_s={speaker_change_min_segment_s} "
                f"is below 2 x speaker_change_window_s="
                f"{speaker_change_window_s}; the re-cut would never run on any "
                "segment, silently"
            )
        self.voice_pool = voice_pool
        # Manual routing. Empty table => auto mode (language ID plus
        # elimination over the participant set), which stays the default.
        # The table lives on the SESSION, so it survives a reconnect for free
        # -- the same property the preset voice assignment relies on.
        self.routing_table = RoutingTable()
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
        # The written record. Deliberately NOT derived from the journal: the
        # journal evicts under a byte budget, the transcript ends only at an
        # explicit clear (§17.2).
        self.transcript = TranscriptLog(clock=clock)
        # Reading mode (§17.1). Orthogonal to voice_mode: `silent` decides
        # WHETHER anything is spoken, `voice_mode` decides in WHOSE voice.
        self.output_mode = output_mode
        # The speaker button the user armed for the NEXT utterance (§17.5).
        # One utterance only; consumed at identification.
        self._armed_speaker: Optional[str] = None
        # Uncertainty bookkeeping (§17.4). The embedding of every unresolved
        # ambiguous line is kept so a tap can fold it into the chosen speaker
        # -- and ONLY a tap can, which is why it is parked here instead of
        # being folded on the way past.
        self._pending: Dict[int, object] = {}
        self._max_pending = 200
        #: line_id -> (speaker_id, confidence, candidates) before the last tap.
        self._undo: Dict[int, Tuple[str, str, List[Dict[str, object]]]] = {}
        self._last_embedding = None
        self._last_uncertain = False
        self._last_candidates: List[Dict[str, object]] = []
        # Name suggestions (§17.3). The adjacency lives in the tracker; the
        # LLM only ever classifies one utterance's text.
        self.name_hints = SuggestionTracker(clock=clock)
        self.suggestions: Dict[str, NameSuggestion] = {}
        #: Names applied automatically, kept so they can be taken back.
        self.applied_names: Dict[str, NameSuggestion] = {}
        #: Speakers the user named by hand; the automatic path never touches
        #: them again.
        self._manual_names: set = set()
        self.name_suggestions_enabled = True
        self._clock = clock
        self._queue: Deque[Segment] = deque()
        self._max_queued = max(1, max_queued_turns)
        self._turn_lock = asyncio.Lock()
        #: The turn currently being translated/synthesized, or None. Kept so a
        #: stop can NAME what it abandoned; the client discards audio per
        #: turn_id and an ack that cannot say which turn died is not
        #: actionable.
        self._active_turn_id: Optional[str] = None
        #: Audio of the turn being processed, so voice classification has an
        #: input on a speaker's FIRST turn (§19.10). Cleared with the turn.
        self._turn_audio: Optional[AudioChunk] = None
        #: Increments on every stop. Not used to gate anything server-side --
        #: it is the client's cheap "is this frame from before my stop"
        #: check, and a counter is the smallest thing that answers it.
        self._stop_epoch = 0
        self._closed = False
        self.last_activity = clock()
        self._attached = 0
        self._detached_at: Optional[float] = None
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

    def attach(self) -> int:
        """A client connected. Returns the number now attached."""
        self._attached += 1
        self._detached_at = None
        self.touch()
        return self._attached

    def detach(self) -> int:
        """A client's socket went away.

        The session is NOT closed here: a dropped mobile link is the normal
        case and must be resumable by token. But the moment of detachment is
        recorded, because "idle" and "abandoned" need different deadlines --
        a conversation that pauses for two minutes while attached is alive,
        one whose socket died two minutes ago probably is not.
        """
        self._attached = max(0, self._attached - 1)
        if self._attached == 0:
            self._detached_at = self._clock()
            # Anything still queued belongs to a client that is no longer
            # listening. Draining it would keep the synthesizer busy for
            # nobody -- which is precisely how six abandoned sessions turned
            # 4 s of first-audio latency into 52 s.
            self._queue.clear()
        return self._attached

    @property
    def attached(self) -> int:
        return self._attached

    def detached_seconds(self) -> Optional[float]:
        """Seconds since the last client left, or None while one is here."""
        if self._attached > 0 or self._detached_at is None:
            return None
        return self._clock() - self._detached_at

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
            "output_mode": self.output_mode.value,
            "armed_speaker": self._armed_speaker,
            "routing_mode": "manual" if self.routing_table else "auto",
            "routing_pairs": self.routing_table.to_json(),
            # Per-pair usability with the refusing stage named, so the UI
            # can grey a row out and say WHY instead of dropping it.
            "routing_capability": list(
                self.routing_table.capability_report(self.matrix)
            ),
            "voice_pool": self.voice_pool.to_json() if self.voice_pool else None,
            "speaking": self.segmenter.speaking,
            "journal_seq": self.journal.next_seq,
            "journal_floor": self.journal.floor,
            "transcript_lines": len(self.transcript),
            "transcript_next_line_id": self.transcript.next_line_id,
            "transcript_floor": self.transcript.floor,
        }

    # -- the written record (§17.2) -----------------------------------------

    def _speaker_label(self, speaker_id: str) -> str:
        """Confirmed name if there is one, else the id itself."""
        if not speaker_id:
            return ""
        try:
            profile = self.speakers.get(speaker_id)
        except KeyError:
            return speaker_id
        return profile.label or speaker_id

    def _emit_line(self, line: TranscriptLine, update: bool = False) -> None:
        self.journal.append(
            EventKind.TRANSCRIPT_UPDATE if update else EventKind.TRANSCRIPT_LINE,
            {"line": line.to_json()},
        )

    def name_speaker(
        self, speaker_id: str, label: str, manual: bool = True
    ) -> List[TranscriptLine]:
        """Give a speaker a name; applies retroactively AND forward (§17.3).

        Returns the transcript lines that changed. The rename is stored on the
        profile, so lines created later carry it without a second pass.
        """
        profile = self.speakers.get(speaker_id)
        if not manual and speaker_id in self._manual_names:
            # A name the user typed is never overwritten by the automatic
            # path. The machine may be right more often; the human is the one
            # who has to live with the answer.
            return []
        profile.label = label or None
        if manual:
            if label:
                self._manual_names.add(speaker_id)
            else:
                self._manual_names.discard(speaker_id)
        changed = self.transcript.relabel_speaker(speaker_id, label or speaker_id)
        for line in changed:
            self._emit_line(line, update=True)
        self.journal.append(
            EventKind.SESSION_STATE,
            {"speaker_named": speaker_id, "label": label,
             "lines_updated": len(changed)},
        )
        return changed

    # -- name suggestions (§17.3) -------------------------------------------

    async def _suggest_names(
        self, turn_id: str, speaker_id: str, text: str, language: str = ""
    ) -> List[NameSuggestion]:
        """Offer names heard in this turn. Never applies any of them."""
        if not self.name_suggestions_enabled:
            return []
        candidates = []
        # The pre-filter decides whether the LLM is asked at all. An earlier
        # turn's addressed candidate still has to be OFFERED this turn, so the
        # tracker runs either way -- only the extraction is skipped.
        if looks_like_naming(text, language):
            ask = getattr(self.mt, "ask", None)
            if callable(ask):
                try:
                    answer = await maybe_await(ask(EXTRACTION_PROMPT, text))
                    candidates = parse_candidates(answer)
                except BackendError as exc:
                    # A failed suggestion is a missing convenience, not a
                    # failed turn: the translation already happened.
                    logger.info("name extraction failed: %s", exc)
                    candidates = []
        suggestions = self.name_hints.observe(
            turn_id, speaker_id, candidates, uncertain=self._last_uncertain
        )
        for suggestion in suggestions:
            self.suggestions[suggestion.suggestion_id] = suggestion
            payload = suggestion.to_json()
            # A person saying "ich bin X" about THEMSELVES is not a guess that
            # needs confirming -- there is no other party the name could
            # belong to, and the user reported the name plainly not working
            # when it sat in a chip. So `self` applies immediately, is marked
            # as automatic, and stays undoable. The two AMBIGUOUS kinds do
            # not: a third-party introduction belongs to nobody in the room
            # yet, and an addressed name is inferred from turn adjacency,
            # which is exactly where a wrong guess would be expensive.
            if suggestion.kind == KIND_SELF and suggestion.speaker_id:
                try:
                    changed = self.name_speaker(
                        suggestion.speaker_id, suggestion.name, manual=False
                    )
                except KeyError:
                    changed = []
                payload["applied"] = True
                payload["automatic"] = True
                payload["lines_updated"] = len(changed)
                self.suggestions.pop(suggestion.suggestion_id, None)
                self.applied_names[suggestion.suggestion_id] = suggestion
            self.journal.append(EventKind.SPEAKER_SUGGESTION, payload)
        return suggestions

    def undo_name(self, suggestion_id: str) -> bool:
        """Take back an automatically applied name (§17.3)."""
        suggestion = self.applied_names.pop(suggestion_id, None)
        if suggestion is None:
            return False
        try:
            self.name_speaker(suggestion.speaker_id, "", manual=False)
        except KeyError:
            return False
        self.name_hints.discard(suggestion.speaker_id, suggestion.name)
        return True

    def confirm_suggestion(
        self, suggestion_id: str, speaker_id: Optional[str] = None
    ) -> List[TranscriptLine]:
        """Apply a suggested name. The tap is what applies it, never the model.

        ``speaker_id`` is required for a floating third-party suggestion,
        which by construction belongs to nobody yet, and is otherwise taken
        from the suggestion.
        """
        suggestion = self.suggestions.pop(suggestion_id, None)
        if suggestion is None:
            raise KeyError(f"unknown suggestion {suggestion_id!r}")
        target = speaker_id or suggestion.speaker_id
        if not target:
            raise ValueError(
                f"suggestion {suggestion_id!r} is a third-party name and "
                "belongs to nobody yet; name the speaker it applies to"
            )
        return self.name_speaker(target, suggestion.name)

    def discard_suggestion(self, suggestion_id: str) -> bool:
        suggestion = self.suggestions.pop(suggestion_id, None)
        if suggestion is None:
            return False
        self.name_hints.discard(suggestion.speaker_id, suggestion.name)
        self.journal.append(
            EventKind.SESSION_STATE,
            {"suggestion_discarded": suggestion_id, "name": suggestion.name},
        )
        return True

    # -- uncertain attributions (§17.4) -------------------------------------

    def resolve_line(
        self, line_id: int, speaker_id: Optional[str] = None
    ) -> Optional[TranscriptLine]:
        """Settle an ambiguous line by hand. Returns the changed line.

        ``speaker_id`` None mints a new speaker for it -- the "speaker-N
        (new)" candidate, which for two similar young voices is frequently
        the right answer and must therefore be a real option rather than a
        fallback.

        The centroid moves HERE and only here. Folding an ambiguous slice at
        assignment time would corrupt an identity permanently, with nothing
        downstream able to tell it had happened.
        """
        line = self.transcript.get(line_id)
        if line is None:
            return None
        previous = (line.speaker_id, line.confidence, list(line.candidates))
        embedding = self._pending.pop(line_id, None)
        if speaker_id is None:
            speaker_id = self.speakers.create_speaker().speaker_id
        if embedding is not None:
            self.speakers.fold_confirmed(speaker_id, embedding)
        changed = self.transcript.reassign_speaker(
            line_id,
            speaker_id,
            self._speaker_label(speaker_id),
            resolved_by="user",
        )
        if changed is None:
            return None
        self._undo[line_id] = previous
        self._emit_line(changed, update=True)
        return changed

    def undo_resolution(self, line_id: int) -> Optional[TranscriptLine]:
        """Put a line back the way it was before the last tap."""
        if line_id not in self._undo:
            return None
        speaker_id, confidence, candidates = self._undo.pop(line_id)
        line = self.transcript.get(line_id)
        if line is None:
            return None
        line.speaker_id = speaker_id
        line.speaker_label = self._speaker_label(speaker_id)
        line.confidence = confidence
        line.candidates = candidates
        line.origin = ORIGIN_AUTO
        line.resolved_by = None
        self._emit_line(line, update=True)
        return line

    def _auto_resolve(self) -> List[TranscriptLine]:
        """Settle earlier ambiguity that later audio made unambiguous.

        Visible, never silent: the badge CHANGES through an update event. A
        line the user already settled is never re-decided -- their tap is
        ground truth and outranks any later measurement.
        """
        if not self._pending:
            return []
        changed: List[TranscriptLine] = []
        for line_id in list(self._pending):
            line = self.transcript.get(line_id)
            if line is None or line.origin == ORIGIN_MANUAL:
                self._pending.pop(line_id, None)
                continue
            if line.resolved_by is not None:
                self._pending.pop(line_id, None)
                continue
            # Ranked against EVERYBODY, the current speaker included.
            #
            # This used to exclude the line's own speaker, and that was right
            # while an uncertain turn MINTED its own profile: a profile seeded
            # by this very embedding matches itself at ~1.0 forever, so
            # counting it would clear every badge on the next turn while
            # learning nothing. The continuity guard removed that case --
            # a line can only be pending while its similarity sat in the
            # uncertainty band, and in that band the turn is now attributed to
            # an EXISTING speaker rather than to a fresh self-seeded one.
            #
            # So the exclusion no longer protects anything and now costs
            # something: the guess "this was probably the same person" could
            # never be CONFIRMED, only overturned, and an uncertain badge
            # would sit there for the rest of the conversation even after the
            # centroid moved and settled the question.
            ranked = self.speakers.rank(self._pending[line_id])
            if not ranked:
                continue
            best_id, best_sim = ranked[0]
            if best_sim < self.speakers.config.match_threshold:
                continue
            self._pending.pop(line_id, None)
            updated = self.transcript.reassign_speaker(
                line_id,
                best_id,
                self._speaker_label(best_id),
                origin=ORIGIN_AUTO,
                resolved_by="auto",
            )
            if updated is not None:
                self._emit_line(updated, update=True)
                changed.append(updated)
        return changed

    # -- speaker buttons (§17.5) --------------------------------------------

    @property
    def armed_speaker(self) -> Optional[str]:
        return self._armed_speaker

    def arm_speaker(self, speaker_id: Optional[str]) -> Optional[str]:
        """Attribute the NEXT utterance to this speaker, skipping the embedder.

        Passing ``None`` disarms, which is what tapping the lit button again
        does. Arming an unknown speaker is refused rather than remembered: a
        button that silently does nothing is worse than no button.
        """
        if speaker_id is None:
            self._armed_speaker = None
        else:
            self.speakers.get(speaker_id)  # raises KeyError if unknown
            self._armed_speaker = speaker_id
        self.journal.append(
            EventKind.SESSION_STATE, {"armed_speaker": self._armed_speaker}
        )
        return self._armed_speaker

    def add_speaker(
        self,
        label: Optional[str] = None,
        voice_class: Optional[VoiceClass] = None,
    ) -> str:
        """The "+" button: somebody new is here, and the user says so.

        No audio yet, so no centroid yet. Their first attributed utterance
        seeds it, which is why this is not the same as waiting for the
        clustering to notice — the clustering would have compared the new
        person against the existing ones and could have merged them.
        """
        profile = self.speakers.create_speaker(label=label)
        # The user stating the class is better evidence than the F0 heuristic
        # could ever be, and it is available BEFORE the first utterance --
        # which is exactly when a preset voice has to be chosen. Without it a
        # manually added speaker would have to talk once before sounding
        # right.
        if voice_class is not None and self.voice_pool is not None:
            self.override_voice_class(profile.speaker_id, voice_class)
        self.journal.append(
            EventKind.SESSION_STATE,
            {
                "speaker_added": profile.speaker_id,
                "label": label,
                "voice_class": voice_class.value if voice_class else None,
                "manual": True,
            },
        )
        return profile.speaker_id

    def delete_speaker(self, speaker_id: str) -> Dict[str, object]:
        """Remove a speaker and everything attached to them (§17.5).

        Returns what was released, so the caller can say so rather than the
        row simply losing a button.
        """
        profile = self.speakers.remove(speaker_id)
        released = round(profile.reference_seconds(), 2)
        if self._armed_speaker == speaker_id:
            # A button that no longer exists must not stay armed, or the next
            # utterance would be attributed to a speaker who is gone.
            self._armed_speaker = None
        # Pending suggestions for this speaker are meaningless now.
        for sid, suggestion in list(self.suggestions.items()):
            if suggestion.speaker_id == speaker_id:
                self.suggestions.pop(sid, None)
        self.applied_names = {
            sid: sug for sid, sug in self.applied_names.items()
            if sug.speaker_id != speaker_id
        }
        self._manual_names.discard(speaker_id)
        if self.voice_pool is not None:
            # Give the preset voice back to the pool, or a session that
            # corrects a few mistaken speakers runs the pool dry.
            release = getattr(self.voice_pool, "release", None)
            if callable(release):
                release(speaker_id)
        self.journal.append(
            EventKind.SESSION_STATE,
            {
                "speaker_deleted": speaker_id,
                "label": profile.label,
                "reference_seconds_released": released,
            },
        )
        return {
            "speaker_id": speaker_id,
            "label": profile.label,
            "reference_seconds_released": released,
        }

    def clear_transcript(self) -> int:
        """The only thing that empties the record. Returns lines removed."""
        removed = self.transcript.clear()
        self.journal.append(
            EventKind.TRANSCRIPT_CLEARED,
            {"removed": removed, "next_line_id": self.transcript.next_line_id},
        )
        return removed

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
                },
            )
        self._queue.append(segment)
        return True

    def pending(self) -> int:
        return len(self._queue)

    def abort_playback(self, reason: str = "user") -> Dict[str, object]:
        """Abandon everything not yet spoken (§19.12).

        Two halves, and only the first is visible here. This drops the QUEUED
        segments and records the stop; cancelling the in-flight turn is the
        caller's half, because the task handle lives with the connection that
        created it. Both are needed: dropping the queue while a synthesis runs
        would leave the talker busy for the rest of that clause, which is
        exactly the "I pressed stop and it kept talking" the button exists to
        prevent.

        Returns what was abandoned rather than nothing, because the ack has to
        carry it: the client discards audio per ``turn_id``, so it needs to be
        told which ids are dead. Any binary frame for an aborted turn that is
        already in flight arrives AFTER this event, which is what makes the
        rule "drop frames whose turn is aborted" sufficient on the client.
        """
        dropped = len(self._queue)
        self._queue.clear()
        aborted = self._active_turn_id
        self._stop_epoch += 1
        self.journal.append(
            EventKind.PLAYBACK_STOPPED,
            {
                "reason": reason,
                "aborted_turn_id": aborted,
                "dropped_queued": dropped,
                "stop_epoch": self._stop_epoch,
            },
        )
        return {
            "aborted_turn_id": aborted,
            "dropped_queued": dropped,
            "stop_epoch": self._stop_epoch,
        }

    async def drain(self) -> List[TurnResult]:
        """Run every queued segment through the pipeline, in order."""
        results: List[TurnResult] = []
        while self._queue and not self._closed:
            segment = self._queue.popleft()
            results.extend(await self.run_turn_multi(segment))
        return results

    # -- the pipeline -------------------------------------------------------

    async def run_turn(self, segment: Segment) -> Optional[TurnResult]:
        """One segment, all the way to audio. Returns the LAST turn produced.

        Kept for callers that want a single result; a segment containing two
        speakers produces two turns and only the last is returned here. Prefer
        :meth:`run_turn_multi` when every result matters.
        """
        results = await self.run_turn_multi(segment)
        return results[-1] if results else None

    async def run_turn_multi(self, segment: Segment) -> List[TurnResult]:
        """One segment, re-cut at speaker changes, all the way to audio.

        Serialized per session. The split check runs BEFORE recognition, not
        after: two people in one segment must be transcribed separately or the
        ASR produces one run-on utterance whose halves are in different
        languages, and no amount of later attribution repairs that.
        """
        async with self._turn_lock:
            try:
                pieces = await self._split_at_speaker_changes(segment)
                results: List[TurnResult] = []
                for piece in pieces:
                    result = await self._run_turn_locked(piece)
                    if result is not None:
                        results.append(result)
                return results
            finally:
                # Cleared on the way out however we leave -- return, error, or
                # the cancellation a stop delivers. A stale active turn would
                # make the NEXT stop name a turn that finished long ago.
                self._active_turn_id = None
                self._turn_audio = None

    async def _split_at_speaker_changes(self, segment: Segment) -> List[Segment]:
        """Re-cut a segment wherever the voice changes inside it.

        The one real failure mode of per-segment (rather than frame-level)
        diarization: two people speaking back-to-back with no pause between
        them land in a single VAD segment, get one embedding, and one of them
        contributes their voice to the other's reference buffer. A poisoned
        reference buffer is audible in every later turn by that speaker, which
        makes this the expensive kind of mistake.

        Gated on segment length: a short segment cannot hold two turns, and
        paying N extra embedder passes on every utterance to discover that
        would put the cost on the common case to protect the rare one.
        """
        if not self.speaker_change_detection:
            return [segment]
        if segment.duration_s < self.speaker_change_min_segment_s:
            return [segment]

        window_s = self.speaker_change_window_s
        rate = segment.audio.sample_rate
        window = int(window_s * rate)
        count = int(segment.duration_s // window_s)
        if count < 2:
            return [segment]

        embeddings = []
        for index in range(count):
            start = index * window
            chunk = AudioChunk(
                segment.audio.samples[start : start + window], rate
            )
            try:
                embeddings.append(await self.embedder.embed(chunk))
            except BackendError:
                # An unembeddable window (too quiet, too short) makes the whole
                # dispersion test unreliable rather than partially valid, so
                # the segment is left whole. Splitting on incomplete evidence
                # would be worse than not splitting.
                return [segment]

        points = split_points_by_dispersion(
            embeddings, self.speaker_change_threshold
        )
        if not points:
            return [segment]

        boundaries = [0, *[p * window for p in points], len(segment.audio.samples)]
        pieces: List[Segment] = []
        for index in range(len(boundaries) - 1):
            start, end = boundaries[index], boundaries[index + 1]
            piece_audio = AudioChunk(segment.audio.samples[start:end], rate)
            if piece_audio.duration_s < self.speaker_change_min_piece_s:
                continue
            pieces.append(
                dataclasses.replace(
                    segment,
                    audio=piece_audio,
                    start_s=segment.start_s + start / rate,
                )
            )
        if len(pieces) < 2:
            return [segment]

        self.journal.append(
            EventKind.TURN_SPLIT,
            {
                "segment_index": segment.index,
                "pieces": len(pieces),
                "at_seconds": [round(p * window_s, 2) for p in points],
                "durations_s": [round(p.duration_s, 2) for p in pieces],
            },
        )
        logger.info(
            "session %s re-cut segment %d into %d pieces at a speaker change",
            self.session_id,
            segment.index,
            len(pieces),
        )
        return pieces

    async def _run_turn_locked(self, segment: Segment) -> Optional[TurnResult]:
        turn_id = uuid.uuid4().hex[:12]
        self._active_turn_id = turn_id
        self._turn_audio = segment.audio
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
        speaker_id, similarity, admitted, manual = await self._identify(
            segment, transcript
        )
        watch.embed_ms = (self._clock() - t_embed) * 1000.0
        self.journal.append(
            EventKind.TURN_SPEAKER,
            {
                "turn_id": turn_id,
                "speaker_id": speaker_id,
                "similarity": round(similarity, 3),
                "reference_admitted": admitted,
                "manual": manual,
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

        # The written record gets its line HERE, before routing and before
        # translation. A turn that is never routed (no rule for its language)
        # or whose synthesis fails still happened and still belongs in the
        # transcript -- writing the line only on success would make the
        # record disagree with what the user heard themselves say.
        line = self.transcript.append(
            turn_id=turn_id,
            speaker_id=speaker_id,
            speaker_label=self._speaker_label(speaker_id),
            source_language=source,
            source_text=transcript.text,
            origin=ORIGIN_MANUAL if manual else ORIGIN_AUTO,
            confidence=(
                CONFIDENCE_UNCERTAIN if self._last_uncertain else CONFIDENCE_EXACT
            ),
            candidates=self._last_candidates if self._last_uncertain else (),
        )
        if self._last_uncertain and self._last_embedding is not None:
            # Kept so a later tap can fold it into whichever speaker the user
            # picks. Bounded: an hour of ambiguous turns must not become a
            # memory leak, and the oldest unresolved line is the least likely
            # to be corrected.
            self._pending[line.line_id] = self._last_embedding
            while len(self._pending) > self._max_pending:
                self._pending.pop(next(iter(self._pending)))
        self._emit_line(line)
        targets = self._targets_for(source)
        if targets is None:
            # Manual mode with no rule for this language. Passed through
            # untranslated and TAGGED rather than dropped: silence would read
            # as a broken translator, while a visible tag tells the user
            # exactly which rule is missing.
            self.journal.append(
                EventKind.TURN_UNROUTED,
                {
                    "turn_id": turn_id,
                    "speaker_id": speaker_id,
                    "source": source,
                    "text": transcript.text,
                    "reason": f"no routing pair for {source!r}",
                    "known_languages": list(self.routing_table.languages()),
                },
            )
            self.journal.append(
                EventKind.TURN_DONE,
                {"turn_id": turn_id, "empty": True, "reason": "no_routing_pair",
                 "source": source},
            )
            return None
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
            updated = self.transcript.set_translations(line.line_id, translations)
            if updated is not None:
                self._emit_line(updated, update=True)

            # Only the first target feeds the MT history: a multi-target fan-out
            # would otherwise stack N assistant turns per user turn and skew the
            # context window towards whichever language happened to sort first.
            first_target = sorted(translations)[0]
            remember = getattr(self.mt, "remember", None)
            if callable(remember):
                remember(transcript.text, translations[first_target])

        self.turns_completed += 1
        await self._suggest_names(
            turn_id, speaker_id, transcript.text, source
        )
        # Later audio may have settled an earlier ambiguity. Run after this
        # turn's centroid update, which is the thing that could have changed
        # the answer.
        self._auto_resolve()
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
        # Pushed per call, not once at construction: ONE recognizer serves
        # every session in this process, so a whitelist installed when a
        # session opened would be whichever session opened LAST. Setting it
        # immediately before the call makes it this session's, which is the
        # best a shared backend allows without a per-call parameter. (Capacity
        # is one conversation anyway -- §17.8.2 -- so the remaining interleave
        # window is not reachable in practice today.)
        self._push_asr_whitelist()
        hint = None
        profiles = self.speakers.profiles()
        if profiles:
            recent = max(profiles, key=lambda p: p.last_seen)
            hint = recent.last_language
        return await self.asr.transcribe(segment.audio, hint_language=hint)

    async def _identify(
        self, segment: Segment, transcript: Transcript
    ) -> Tuple[str, float, bool, bool]:
        """Returns ``(speaker_id, similarity, admitted, manual)``.

        ``manual`` marks an attribution the user made with a speaker button
        (§17.5). It is ground truth: the line is never badged uncertain and no
        later similarity computation re-decides it.
        """
        self._last_embedding = None
        self._last_uncertain = False
        self._last_candidates = []
        armed = self._armed_speaker
        if armed is not None:
            # Spent on this utterance whatever happens next, which is what
            # "exactly one utterance" means: the button overrides
            # IDENTIFICATION, and identification is happening right now.
            self._armed_speaker = None
            self.journal.append(
                EventKind.SESSION_STATE, {"armed_speaker": None, "consumed_by": armed}
            )
            embedding = None
            try:
                if segment.duration_s >= getattr(self.embedder, "min_seconds", 0.0):
                    embedding = await self.embedder.embed(segment.audio)
            except BackendError:
                # The attribution stands regardless: the user said who spoke,
                # and an embedder failure has no standing against that. Only
                # the centroid update is lost.
                embedding = None
            try:
                profile, admitted = self.speakers.assign_manual(
                    speaker_id=armed,
                    audio=segment.audio,
                    embedding=embedding,
                    text=transcript.text,
                    language=transcript.language,
                )
            except KeyError:
                logger.error(
                    "session %s was armed for unknown speaker %s; falling back "
                    "to automatic identification",
                    self.session_id, armed,
                )
            else:
                return profile.speaker_id, 1.0, admitted, True

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
                return recent.speaker_id, 0.0, False, False
            # Nobody known yet and nothing embeddable: mint a provisional id
            # so the turn still has a speaker to attach to.
            return "speaker-unknown", 0.0, False, False
        try:
            embedding = await self.embedder.embed(segment.audio)
        except BackendError:
            profiles = self.speakers.profiles()
            if profiles:
                recent = max(profiles, key=lambda p: p.last_seen)
                return recent.speaker_id, 0.0, False, False
            return "speaker-unknown", 0.0, False, False
        # Ranked BEFORE the assignment, against the centroids the assignment
        # is about to decide on. Ranking afterwards would score the segment
        # against a centroid it had already moved.
        ranked = self.speakers.rank(embedding)
        profile, similarity, admitted = self.speakers.assign(
            embedding=embedding,
            audio=segment.audio,
            text=transcript.text,
            language=transcript.language,
            language_confidence=transcript.language_confidence,
        )
        uncertain, candidates = self.speakers.uncertainty(ranked, profile.speaker_id)
        self._last_embedding = embedding
        self._last_uncertain = uncertain
        self._last_candidates = candidates
        return profile.speaker_id, similarity, admitted, False

    def _targets_for(self, source: str) -> Optional[Tuple[str, ...]]:
        """Where this utterance goes -- possibly to more than one place.

        Manual mode is a deterministic table lookup and deliberately does NOT
        fall back to elimination: once the user has written pairs, a guessed
        direction is exactly what they asked to be rid of. ``None`` means the
        table is in force and has nothing for this source -- distinct from an
        empty tuple, which means auto mode found no target.

        The table returns every PARTNER of the source language, so
        ``{de<->es, de<->fr}`` renders a German turn twice. That fan-out is the
        intent; the loop that consumes this already handles several targets.
        """
        if self.routing_table:
            partners = self.routing_table.partners_for(source)
            return partners or None
        return self.conversation.targets_for(source)

    def set_routing_pairs(self, pairs: Sequence[Tuple[str, str]]) -> None:
        """Replace the manual routing table. Empty restores auto mode.

        Each entry is an UNORDERED pair: ``(de, es)`` routes both ways, and
        passing ``(es, de)`` as well is a no-op rather than an error.
        """
        table = RoutingTable()
        table.replace_all(pairs)
        table.validate_against(self.matrix)
        self.routing_table = table
        self._push_asr_whitelist()
        self.journal.append(
            EventKind.SESSION_STATE,
            {"routing_mode": "manual" if table else "auto",
             "routing_pairs": table.to_json()},
        )

    def _push_asr_whitelist(self) -> None:
        """Constrain language identification to the languages actually routed.

        The recognizer keeps IDENTIFYING the source language -- the direction
        stays observed, never configured, which is requirement 5. What changes
        is the candidate set: a table naming ``{de, es}`` makes any other
        answer a misdetection by construction, and an unconstrained identifier
        that picks Dutch for a German sentence costs the whole turn. Restricted
        detection returns the best IN-SET language with its own probability as
        the confidence, so a weak decision is TAGGED rather than discarded.

        Duck-typed: a backend that cannot restrict simply does not implement
        the method, and the session neither requires nor pretends otherwise.
        """
        setter = getattr(self.asr, "set_restrict_languages", None)
        if not callable(setter):
            return
        setter(self.detection_whitelist())

    def detection_whitelist(self) -> Tuple[str, ...]:
        """The languages the identifier is allowed to answer with.

        The routing table when the user named pairs; otherwise the
        conversation's PARTICIPANT set, which is the other thing that bounds
        a conversation and is what auto mode routes by elimination over.

        The participant fallback used to be missing, and that WAS the bug: a
        session opened with ``de,es`` and no explicit pairs pushed an EMPTY
        restriction, which `constrained_language_choice` correctly treats as
        "no restriction". Whisper was therefore free to answer ``en`` for a
        German sentence -- and did, on a real device, with English selected
        nowhere. Auto mode is the DEFAULT, so requirement 5's constrained
        detection reached almost nobody: it was wired only for the minority
        of sessions that had set a manual table. A whitelist that is only
        pushed on a path most sessions never take is a whitelist that does
        not exist.
        """
        if self.routing_table:
            return tuple(self.routing_table.languages())
        return tuple(sorted(self.conversation.participants))

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
        # Reading mode (§17.1): everything up to here has already run -- the
        # recognizer, the speaker assignment and its reference admission, the
        # routing decision -- and everything after MT still runs. Only the
        # synthesizer is skipped, and no voice is chosen at all: reporting a
        # clone downgrade for a turn nobody was going to hear would be a
        # reason attached to a non-event.
        silent = self.output_mode is OutputMode.SILENT
        if silent:
            voice = None
            reference, reference_text = None, ""
            fallback = False
            self.journal.append(
                EventKind.TURN_VOICE,
                {"turn_id": turn_id, "target": target, "speaker_id": speaker_id,
                 "output_mode": OutputMode.SILENT.value, "spoken": False},
            )
        else:
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
            # The silent check is explicit rather than left to `reference is
            # None`. Both are true in reading mode today, but a future path
            # that supplies a reference without wanting speech would then
            # start talking, and nothing here would say why.
            if silent or not unit.strip() or reference is None:
                return
            tts_start = self._clock()
            # One talker serves every conversation. A turn that arrives while
            # another is synthesizing does not run slowly, it does not run at
            # all -- so say so NOW rather than letting the user watch nothing
            # happen and conclude the translator is broken. Sampled before the
            # call because the wait is only knowable in advance; the exact
            # duration lands in the stopwatch afterwards.
            if getattr(self.tts, "busy", False):
                self.journal.append(
                    EventKind.TURN_QUEUED,
                    {"turn_id": turn_id, "target": target, "reason": "tts_busy"},
                )
            TTS_QUEUE_WAIT_S.set(0.0)
            async for piece in self.tts.synthesize(
                unit, target, reference, reference_text, voice.backend_voice_id
            ):
                if first_audio_at is None:
                    first_audio_at = self._clock()
                    watch.tts_first_audio_ms = (first_audio_at - tts_start) * 1000.0
                    watch.tts_wait_ms = TTS_QUEUE_WAIT_S.get() * 1000.0
                chunks.append(piece)
                self.journal.append(
                    EventKind.TURN_AUDIO,
                    {"turn_id": turn_id, "target": target, "speaker_id": speaker_id},
                    audio=piece,
                )

        # TEXT MUST NOT WAIT FOR AUDIO (§19.13, user field report: "the
        # translated text should be there much earlier than the audio; it
        # somehow waits for it").
        #
        # It did, and this is where. `speak(unit)` used to be AWAITED inside
        # the MT loop, so the loop stopped consuming translation deltas for
        # the whole duration of a clause's synthesis -- a measured 3.4 s
        # (§18.4). The first clause's TEXT was already emitted before its
        # synthesis, so the defect was invisible on a one-clause turn; from
        # the second clause on, every line of text arrived at the speed of
        # the SPEAKER, not of the translator.
        #
        # The units are handed to a worker instead. Text is journalled the
        # moment MT produces it; synthesis follows at its own pace. One
        # worker, a FIFO queue: the audio order is exactly the text order,
        # which is what keeps `turn_id` attribution and playback sequencing
        # correct. This also makes `mt_total_ms` mean what its name says --
        # it used to include synthesis.
        speech: "asyncio.Queue[Optional[str]]" = asyncio.Queue()

        async def speech_worker() -> None:
            while True:
                unit = await speech.get()
                if unit is None:
                    return
                await speak(unit)

        worker = asyncio.create_task(speech_worker())
        try:
            async for delta in self.mt.translate_stream(
                transcript.text, source, target
            ):
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
                    speech.put_nowait(unit)
            tail = accumulator.flush()
            watch.mt_total_ms = (self._clock() - mt_start) * 1000.0
            if tail:
                pieces.append(tail)
                self.journal.append(
                    EventKind.TURN_TRANSLATION,
                    {"turn_id": turn_id, "target": target, "text": tail,
                     "partial": True},
                )
                speech.put_nowait(tail)
            # Close the queue and let the backlog drain. The turn is not done
            # until its audio is, so the result still carries every chunk --
            # what changed is when the TEXT became visible, not what the turn
            # eventually contains.
            speech.put_nowait(None)
            await worker
        except BaseException:
            # Includes the CancelledError a playback.stop delivers: the
            # worker holds the talker's lock, so leaving it running would
            # keep the card busy for a turn nobody is listening to.
            worker.cancel()
            raise

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
            # NON-SILENCE GATE. A turn that returns the right DURATION of
            # silence is the worst failure shape this pipeline has: every
            # event says success, the client plays nothing, and the listener
            # cannot tell it from a network drop. It happened -- the clone
            # path returned 4.24 s of exact zeros while the reference, the
            # arguments and the synthesizer were each provably fine in
            # isolation -- so the assembled audio is now checked once, here,
            # where every voice mode passes through.
            peak = float(np.abs(merged.samples).max()) if len(merged.samples) else 0.0
            if peak < self.silence_peak_floor:
                reference_peak = (
                    float(np.abs(reference.samples).max())
                    if reference is not None and len(reference.samples)
                    else 0.0
                )
                logger.error(
                    "session %s turn %s synthesized %.2fs of SILENCE "
                    "(peak %.5f < %.5f) in %s mode from a reference of "
                    "%.2fs at peak %.4f -- the audio is dropped rather than "
                    "delivered, because silence that arrives on time is "
                    "indistinguishable from a working system",
                    self.session_id, turn_id, merged.duration_s, peak,
                    self.silence_peak_floor, voice.mode.value,
                    0.0 if reference is None else reference.duration_s,
                    reference_peak,
                )
                self.journal.append(
                    EventKind.ERROR,
                    {
                        "turn_id": turn_id,
                        "target": target,
                        "stage": "tts",
                        "error": (
                            f"synthesis produced {merged.duration_s:.2f}s of "
                            f"silence (peak {peak:.5f}); reference was "
                            f"{0.0 if reference is None else reference.duration_s:.2f}s "
                            f"at peak {reference_peak:.4f}"
                        ),
                    },
                )
                merged = None
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

    def set_output_mode(self, mode: OutputMode) -> OutputMode:
        """Switch between speaking and reading mode at runtime (§17.1).

        Unconditional: unlike preset voice mode there is nothing that can make
        silence impossible, and unlike a TTS failure this is a user decision,
        so it never needs a refusal path.
        """
        self.output_mode = mode
        self.journal.append(
            EventKind.SESSION_STATE, {"output_mode": mode.value}
        )
        return self.output_mode

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
        """Whatever audio we hold for this speaker, for voice classification.

        THE TURN'S OWN AUDIO IS THE FALLBACK, and it is the case that matters
        (§19.10, user field report: "at the start I am still rendered with a
        female voice"). The reference buffer is empty exactly when a
        placeholder voice is needed -- the downgrade path fires BECAUSE there
        is not enough reference to clone from -- so classifying off the
        buffer meant classifying off nothing on every first turn, and
        `classify()` answers UNKNOWN for `None`. The pool then allocated by
        availability instead of by class, which is how a man gets a woman's
        preset.

        The mechanism was there the whole time: a classifier, a class-matched
        pool, sticky assignment. Its reach was zero at the only moment it was
        asked to act. The audio of the turn being spoken is already in hand
        and is a better classification input than a reference buffer anyway --
        it is this speaker, now, by construction.
        """
        try:
            profile = self.speakers.get(speaker_id)
        except KeyError:
            profile = None
        if profile is not None and profile.references:
            return profile.reference_audio(self.tts.sample_rate)
        return self._turn_audio

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
        resume_grace_s: float = 120.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._factory = factory
        self._max = max_sessions
        self._idle_timeout = idle_timeout_s
        #: How long a session survives with nobody attached. Shorter than the
        #: idle timeout on purpose: it is the window in which a reconnect can
        #: still resume by token, not a conversation pause. Long enough for a
        #: tunnel to come back, short enough that an abandoned session does
        #: not hold its share of the machine for five minutes.
        self._resume_grace = resume_grace_s
        self._clock = clock
        self._sessions: Dict[str, TranslatorSession] = {}

    def __len__(self) -> int:
        return len(self._sessions)

    def ids(self) -> Tuple[str, ...]:
        return tuple(self._sessions)

    def collect(self) -> List[str]:
        """Drop sessions nobody is coming back to. Returns the ids collected.

        Two independent deadlines, because two different things go wrong:

        * a session with a client attached is alive no matter how long the
          conversation pauses, and is only collected after ``idle_timeout_s``;
        * a session whose socket died is collected after ``resume_grace_s``,
          which is the window a reconnect has to resume it by token.

        Sticky semantics are untouched inside the grace: the id, the speaker
        registry, the reference buffers and the transcript are all still there
        for a client that comes back.
        """
        dead = []
        for sid, session in self._sessions.items():
            gone = session.detached_seconds()
            if gone is not None and gone > self._resume_grace:
                dead.append(sid)
            elif session.idle_seconds() > self._idle_timeout:
                dead.append(sid)
        for sid in dead:
            self._sessions.pop(sid).close()
        return dead

    def to_json(self) -> List[Dict[str, object]]:
        """Per-session age and idleness, so this class stays diagnosable."""
        return [
            {
                "session_id": sid,
                "attached": session.attached,
                "idle_s": round(session.idle_seconds(), 1),
                "detached_s": (
                    None if session.detached_seconds() is None
                    else round(session.detached_seconds(), 1)
                ),
                "turns": session.turns_completed,
                "queued": session.pending(),
            }
            for sid, session in self._sessions.items()
        ]

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
            results.extend(await session.run_turn_multi(segment))
    tail = session.release()
    if tail is not None:
        results.extend(await session.run_turn_multi(tail))
    return results


async def maybe_await(value):
    """Await ``value`` if it is awaitable, else return it.

    Small helper for adapters that may be sync or async; keeps the pipeline
    from having to know which kind of backend it was handed.
    """
    if isinstance(value, Awaitable):
        return await value
    return value
