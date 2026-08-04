# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Turn segmentation: where an utterance starts and, more importantly, ends.

The end is what costs latency. Everything downstream -- ASR, MT, TTS -- can
only start once the segmenter has declared the turn closed, so the hangover
window (how long we wait after the last speech frame before believing the
speaker is done) is the single largest *fixed* term in the end-to-end budget
and the one knob a user actually feels. It is configuration, not a constant.

The state machine is deliberately boring:

    IDLE --(speech for >= onset_ms)--> SPEAKING
    SPEAKING --(silence for >= hangover_ms)--> emit segment, IDLE
    SPEAKING --(duration >= max_utterance_s)--> emit segment, stay SPEAKING

The last transition is the safety valve: a speaker who does not pause must
still produce translated audio, so a long turn is cut on a *forced* boundary
and flagged as such. A forced cut is a worse translation unit than a natural
pause (it can split a clause), and the flag lets the session tell the client
the difference instead of pretending the sentence ended.

Pre-roll matters and is easy to get wrong: VAD onset is detected a frame or
two *after* speech actually began, so a segment cut at the onset frame clips
the first consonant. The ring buffer keeps ``pre_roll_ms`` of audio ahead of
the onset and prepends it, which is free and audibly better.

The VAD itself is injected. :class:`EnergyVad` is a dependency-free default
that is honest about being crude; a Silero-class VAD implements the same
one-method protocol and is what runs in production.
"""

from __future__ import annotations

import collections
import dataclasses
import enum
import math
from typing import Deque, Iterable, List, Optional, Protocol, runtime_checkable

import numpy as np

from sglang.srt.translator.backends import AudioChunk

__all__ = [
    "Vad",
    "EnergyVad",
    "SegmenterConfig",
    "Segment",
    "SegmentReason",
    "TurnSegmenter",
]


@runtime_checkable
class Vad(Protocol):
    """Frame-level voice activity detection."""

    #: Frame size the detector expects, in milliseconds.
    frame_ms: int

    def is_speech(self, frame: np.ndarray, sample_rate: int) -> bool: ...

    def reset(self) -> None:
        """Drop any per-stream state. Called when a session's audio restarts."""


class EnergyVad:
    """Adaptive-threshold energy VAD. No dependencies, no model, no illusions.

    The noise floor is tracked as a slow exponential minimum of frame RMS, and
    a frame counts as speech when it sits ``margin_db`` above that floor. This
    handles a steadily noisy cafe and fails on impulsive noise (cutlery,
    laughter, a passing scooter) exactly as one would expect from an energy
    detector -- which is why it is the fallback and not the recommendation.
    Its real job is making the hermetic suite runnable without a model file.
    """

    def __init__(
        self,
        frame_ms: int = 20,
        margin_db: float = 9.0,
        floor_attack: float = 0.05,
        floor_release: float = 0.002,
        absolute_floor_db: float = -60.0,
        initial_floor_ceiling_db: float = -35.0,
    ) -> None:
        self.frame_ms = frame_ms
        self._margin_db = margin_db
        self._attack = floor_attack
        self._release = floor_release
        self._absolute_floor_db = absolute_floor_db
        # Seeding the floor from the first frame alone is a real bug, found by
        # the WebSocket test: when the stream OPENS with speech -- which is
        # what push-to-talk always does -- the floor is initialised at speech
        # level and no later frame can clear it by the margin, so the detector
        # goes permanently deaf. Capping the initial estimate keeps the
        # adaptive behaviour for a stream that starts quiet while making a
        # stream that starts loud detectable immediately.
        self._initial_ceiling_db = initial_floor_ceiling_db
        self._floor_db: Optional[float] = None

    def reset(self) -> None:
        self._floor_db = None

    def is_speech(self, frame: np.ndarray, sample_rate: int) -> bool:
        del sample_rate  # threshold is rate-independent
        rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64)))) if len(frame) else 0.0
        db = 20.0 * math.log10(max(rms, 1e-9))
        if self._floor_db is None:
            self._floor_db = min(
                max(db, self._absolute_floor_db), self._initial_ceiling_db
            )
        speech = db > self._floor_db + self._margin_db and db > self._absolute_floor_db
        # Track downwards fast, upwards slowly: the floor should follow a room
        # getting quieter immediately but must not be dragged up by speech.
        rate = self._attack if db < self._floor_db else self._release
        self._floor_db += rate * (db - self._floor_db)
        return speech


class SegmentReason(enum.Enum):
    """Why a segment was closed."""

    #: Natural pause -- the good case, a clause boundary the speaker chose.
    PAUSE = "pause"
    #: ``max_utterance_s`` hit while still speaking. May split mid-clause.
    FORCED = "forced"
    #: Push-to-talk released, or the session closed with speech buffered.
    RELEASED = "released"


@dataclasses.dataclass(frozen=True)
class SegmenterConfig:
    """Timing of the turn machine. Every value is a user-visible latency term.

    Defaults target the stated goal (first translated audio ~2-3 s after the
    speaker stops) on a conversational, half-duplex exchange:

    * ``hangover_ms=550`` is the dominant fixed cost. Below ~400 ms a German
      speaker's clause-internal pauses start cutting turns; above ~800 ms the
      exchange feels laggy. 550 ms is the compromise and is the first thing to
      tune on real recordings.
    * ``onset_ms=120`` rejects single-frame impulses without clipping speech,
      because ``pre_roll_ms`` restores what the onset delay swallowed.
    * ``max_utterance_s=15`` bounds the worst case: a monologue produces a
      translation every 15 s instead of nothing at all.
    """

    sample_rate: int = 16000
    frame_ms: int = 20
    onset_ms: int = 120
    hangover_ms: int = 550
    pre_roll_ms: int = 300
    min_utterance_ms: int = 400
    max_utterance_s: float = 15.0

    def frame_samples(self) -> int:
        return int(self.sample_rate * self.frame_ms / 1000)

    def __post_init__(self) -> None:
        if self.frame_samples() <= 0:
            raise ValueError("frame_ms too small for the configured sample_rate")
        if self.min_utterance_ms >= self.max_utterance_s * 1000:
            raise ValueError("min_utterance_ms must be below max_utterance_s")


@dataclasses.dataclass(frozen=True)
class Segment:
    """One closed utterance, ready for recognition."""

    audio: AudioChunk
    reason: SegmentReason
    #: Session-monotonic index, used by the client to order and to resume.
    index: int
    #: Seconds of audio consumed by the segmenter before this segment started.
    start_s: float
    #: The turn that was already being processed when this segment was queued,
    #: or None. Stamped at `Session.enqueue` because that is the only place the
    #: temporal relation is known, and read after recognition because that is
    #: the earliest the LANGUAGE is known. Together those two facts are the
    #: barge-in predicate: somebody talked over somebody else.
    overlapped_turn_id: Optional[str] = None
    #: The source language of that turn, so the comparison does not have to
    #: reach back into state that may already have moved on.
    overlapped_language: Optional[str] = None

    @property
    def duration_s(self) -> float:
        return self.audio.duration_s


class _State(enum.Enum):
    IDLE = "idle"
    SPEAKING = "speaking"


class TurnSegmenter:
    """Feed it audio, get closed segments back.

    Stateful and single-stream: one instance per session. Not thread-safe;
    the session owns it and feeds it from one task.
    """

    def __init__(
        self,
        config: Optional[SegmenterConfig] = None,
        vad: Optional[Vad] = None,
    ) -> None:
        self.config = config or SegmenterConfig()
        self._vad = vad or EnergyVad(frame_ms=self.config.frame_ms)
        self._state = _State.IDLE
        self._frame_samples = self.config.frame_samples()
        self._carry = np.zeros(0, dtype=np.float32)
        pre_roll_frames = max(
            1, int(self.config.pre_roll_ms / max(self.config.frame_ms, 1))
        )
        self._pre_roll: Deque[np.ndarray] = collections.deque(maxlen=pre_roll_frames)
        self._active: List[np.ndarray] = []
        self._onset_run = 0
        self._silence_run = 0
        self._index = 0
        self._consumed_frames = 0
        self._active_start_frame = 0
        self._forced_continuation = False

    # -- introspection used by the session's status payload -----------------

    @property
    def speaking(self) -> bool:
        return self._state is _State.SPEAKING

    @property
    def segments_emitted(self) -> int:
        return self._index

    def _frames_for(self, ms: int) -> int:
        return max(1, int(round(ms / max(self.config.frame_ms, 1))))

    def reset(self) -> None:
        """Drop buffered speech and VAD state, keeping the segment counter.

        Called on reconnect: the audio stream restarts mid-utterance and the
        partial buffer is not recoverable, but segment indices must keep
        increasing so the client's resume cursor stays meaningful.
        """
        self._vad.reset()
        self._state = _State.IDLE
        self._carry = np.zeros(0, dtype=np.float32)
        self._pre_roll.clear()
        self._active.clear()
        self._onset_run = 0
        self._silence_run = 0
        self._forced_continuation = False

    def feed(self, chunk: AudioChunk) -> List[Segment]:
        """Consume audio, returning every segment that closed inside it."""
        if chunk.sample_rate != self.config.sample_rate:
            raise ValueError(
                f"segmenter runs at {self.config.sample_rate} Hz but got "
                f"{chunk.sample_rate} Hz; resample before feeding"
            )
        buffer = (
            np.concatenate([self._carry, chunk.samples])
            if len(self._carry)
            else chunk.samples
        )
        emitted: List[Segment] = []
        offset = 0
        n = self._frame_samples
        while offset + n <= len(buffer):
            frame = buffer[offset : offset + n]
            offset += n
            segment = self._push_frame(frame)
            if segment is not None:
                emitted.append(segment)
        self._carry = buffer[offset:].copy()
        return emitted

    def flush(self, reason: SegmentReason = SegmentReason.RELEASED) -> Optional[Segment]:
        """Close whatever is buffered. Push-to-talk release and stream end."""
        if self._state is not _State.SPEAKING:
            self._active.clear()
            return None
        return self._close(reason)

    # -- internals ----------------------------------------------------------

    def _push_frame(self, frame: np.ndarray) -> Optional[Segment]:
        self._consumed_frames += 1
        speech = self._vad.is_speech(frame, self.config.sample_rate)
        if self._state is _State.IDLE:
            self._pre_roll.append(frame)
            if speech:
                self._onset_run += 1
                if self._onset_run >= self._frames_for(self.config.onset_ms):
                    self._state = _State.SPEAKING
                    # The pre-roll deque already contains the onset frames, so
                    # taking it wholesale both restores the clipped attack and
                    # avoids double-counting them.
                    self._active = list(self._pre_roll)
                    self._active_start_frame = self._consumed_frames - len(self._active)
                    self._pre_roll.clear()
                    self._onset_run = 0
                    self._silence_run = 0
            else:
                self._onset_run = 0
            return None

        self._active.append(frame)
        if speech:
            self._silence_run = 0
        else:
            self._silence_run += 1
            if self._silence_run >= self._frames_for(self.config.hangover_ms):
                return self._close(SegmentReason.PAUSE)

        active_s = len(self._active) * self.config.frame_ms / 1000.0
        if active_s >= self.config.max_utterance_s:
            return self._close(SegmentReason.FORCED, stay_speaking=True)
        return None

    def _close(
        self, reason: SegmentReason, stay_speaking: bool = False
    ) -> Optional[Segment]:
        frames = self._active
        self._active = []
        duration_ms = len(frames) * self.config.frame_ms
        start_s = self._active_start_frame * self.config.frame_ms / 1000.0

        if stay_speaking:
            # Keep the machine hot and start the next window from here, so a
            # monologue produces back-to-back segments rather than dropping
            # the hangover's worth of audio between them.
            self._active_start_frame = self._consumed_frames
            self._silence_run = 0
            self._forced_continuation = True
        else:
            self._state = _State.IDLE
            self._pre_roll.clear()
            self._onset_run = 0
            self._silence_run = 0
            self._forced_continuation = False

        if duration_ms < self.config.min_utterance_ms:
            # Too short to recognize, too short to embed. Dropped silently:
            # a 200 ms cough is not a turn, and emitting it would cost a full
            # ASR call and an empty translation the user has to sit through.
            return None

        audio = AudioChunk(np.concatenate(frames), self.config.sample_rate)
        segment = Segment(
            audio=audio, reason=reason, index=self._index, start_s=start_s
        )
        self._index += 1
        return segment


def segments_from_audio(
    audio: AudioChunk,
    config: Optional[SegmenterConfig] = None,
    vad: Optional[Vad] = None,
    block_ms: int = 100,
) -> Iterable[Segment]:
    """Run a whole recording through the segmenter, as the stream would.

    Used by tests and by the offline latency harness: feeding one giant chunk
    would not exercise the carry buffer, so the audio is diced into realistic
    transport-sized blocks first.
    """
    seg = TurnSegmenter(config, vad)
    block = int(audio.sample_rate * block_ms / 1000)
    out: List[Segment] = []
    for start in range(0, len(audio.samples), block):
        out.extend(
            seg.feed(AudioChunk(audio.samples[start : start + block], audio.sample_rate))
        )
    tail = seg.flush()
    if tail is not None:
        out.append(tail)
    return out
