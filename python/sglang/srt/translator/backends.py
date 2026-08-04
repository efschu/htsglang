# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Backend interfaces for the four stages, plus hermetic fakes.

The pipeline is a cascade -- recognize, identify the speaker, translate,
synthesize in that speaker's voice -- and every stage is behind a narrow
Protocol so the orchestration can be tested without a GPU, a model download
or a network. The fakes at the bottom of this file are not toys: the whole
hermetic suite runs against them under ``CUDA_VISIBLE_DEVICES=99``, so any
behaviour the fakes cannot express is behaviour the tests cannot cover, and
that is deliberate pressure to keep the interfaces small.

Three rules the interfaces enforce by shape:

1. **Every backend declares its own language set.** No caller may assume a
   language is available; :mod:`sglang.srt.translator.languages` intersects
   what the backends declare. A backend that cannot enumerate its languages
   must say so (MT returns ``None``), which is an explicit claim rather than
   a silent empty set.
2. **Audio is always ``AudioChunk``** -- float32 mono at a declared sample
   rate. Resampling is the adapter's problem, never the pipeline's.
3. **TTS takes reference audio, not a voice id.** Zero-shot cloning is the
   requirement; a backend that only has preset voices cannot implement this
   interface honestly and should not be plugged in as if it could.
"""

from __future__ import annotations

import contextvars
import dataclasses
import math
from typing import (
    AsyncIterator,
    Iterable,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

import numpy as np

__all__ = [
    "AudioChunk",
    "Transcript",
    "SpeakerEmbedding",
    "AsrBackend",
    "SpeakerEmbedder",
    "MtBackend",
    "TtsBackend",
    "BackendError",
    "FakeAsr",
    "FakeEmbedder",
    "FakeMt",
    "FakeTts",
    "TTS_QUEUE_WAIT_S",
]


#: Seconds THIS synthesis call spent waiting for a busy synthesizer, published
#: to the CALLER's context by whichever TTS backend serialises itself.
#:
#: A ContextVar rather than an attribute on the backend, because one backend
#: serves every session: an attribute would carry whichever turn finished
#: last, which is the wrong turn exactly when two conversations are running
#: and the number finally matters. Async generators do not isolate context, so
#: a value set inside ``synthesize`` is read by its driving task and by no
#: other -- verified with concurrent callers, not assumed.
#:
#: It exists because waiting and computing are different defects with
#: different fixes (DESIGN §18.4): a slow synthesizer wants a faster talker, a
#: queued one wants capacity or an honest "someone else is speaking".
TTS_QUEUE_WAIT_S: contextvars.ContextVar = contextvars.ContextVar(
    "tts_queue_wait_s", default=0.0
)


class BackendError(RuntimeError):
    """A backend failed a request. Carries the stage name for the turn event.

    ``retryable`` separates "the backend is momentarily unreachable" from
    "the backend refused this request". Only the first is worth trying again:
    a 400 is a 400 on every attempt, while a read timeout on a card shared
    with the talker is a window that closes by itself. The caller decides the
    policy; this flag only carries the fact, because the backend is the only
    layer that still knows which exception it caught.
    """

    def __init__(self, stage: str, message: str, retryable: bool = False) -> None:
        super().__init__(f"{stage}: {message}")
        self.stage = stage
        self.retryable = retryable


@dataclasses.dataclass(frozen=True, eq=False)
class AudioChunk:
    """Mono float32 PCM in [-1, 1] with its sample rate attached.

    ``eq=False``: a generated ``__eq__`` over an ndarray field returns an
    array, and every ``x in list_of_chunks`` in the codebase would then raise
    "truth value of an array is ambiguous". Identity comparison is what the
    callers actually want; sample equality is a numeric assertion and belongs
    in a test, spelled out.

    The rate travels with the samples because the stages genuinely disagree:
    VAD and ASR want 16 kHz, most neural vocoders emit 22.05 or 24 kHz, and
    the phone plays back at whatever the browser's AudioContext runs at. A
    bare ndarray would make one of those conversions implicit, and an implicit
    resample is a pitch bug waiting for a demo.
    """

    samples: np.ndarray
    sample_rate: int

    def __post_init__(self) -> None:
        if self.samples.ndim != 1:
            raise ValueError(f"audio must be mono 1-D, got shape {self.samples.shape}")
        if self.samples.dtype != np.float32:
            object.__setattr__(self, "samples", self.samples.astype(np.float32))
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")

    @property
    def duration_s(self) -> float:
        return len(self.samples) / float(self.sample_rate)

    def rms(self) -> float:
        if len(self.samples) == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(self.samples, dtype=np.float64))))

    def concat(self, other: AudioChunk) -> AudioChunk:
        if other.sample_rate != self.sample_rate:
            raise ValueError(
                f"cannot concatenate {self.sample_rate} Hz with "
                f"{other.sample_rate} Hz; resample first"
            )
        return AudioChunk(
            np.concatenate([self.samples, other.samples]), self.sample_rate
        )

    def tail(self, seconds: float) -> AudioChunk:
        n = int(seconds * self.sample_rate)
        return AudioChunk(self.samples[-n:] if n else self.samples[:0], self.sample_rate)

    @classmethod
    def silence(cls, seconds: float, sample_rate: int) -> AudioChunk:
        return cls(np.zeros(int(seconds * sample_rate), dtype=np.float32), sample_rate)


@dataclasses.dataclass(frozen=True)
class Transcript:
    """One recognized utterance.

    ``language`` is what the recognizer *heard*, not what anyone configured --
    it is the input to direction routing, so a backend that cannot identify
    the language must not fabricate one; it raises instead.
    """

    text: str
    language: str
    #: 0..1 language-identification confidence. Low confidence does not fail
    #: the turn; it is surfaced so the session can fall back to the last
    #: confident language of the same speaker (see ``session.py``).
    language_confidence: float = 1.0
    #: Recognizer-reported average log-probability or equivalent, if any.
    quality: Optional[float] = None


@dataclasses.dataclass(frozen=True, eq=False)
class SpeakerEmbedding:
    """A fixed-dimension speaker vector, L2-normalised by contract.

    ``eq=False`` for the same reason as :class:`AudioChunk`; similarity, not
    equality, is the meaningful comparison and it has its own method.
    """

    vector: np.ndarray

    def __post_init__(self) -> None:
        vec = np.asarray(self.vector, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vec))
        if norm == 0.0 or not math.isfinite(norm):
            raise ValueError("speaker embedding must have finite non-zero norm")
        object.__setattr__(self, "vector", vec / norm)

    def similarity(self, other: SpeakerEmbedding) -> float:
        """Cosine similarity. Both vectors are unit-norm, so this is a dot."""
        if self.vector.shape != other.vector.shape:
            raise ValueError(
                f"embedding dimension mismatch: {self.vector.shape} vs "
                f"{other.vector.shape}"
            )
        return float(np.dot(self.vector, other.vector))


@runtime_checkable
class AsrBackend(Protocol):
    """Speech recognition with language identification."""

    name: str

    def supported_languages(self) -> Iterable[str]:
        """Every language this recognizer can transcribe."""

    async def transcribe(
        self, audio: AudioChunk, hint_language: Optional[str] = None
    ) -> Transcript:
        """Transcribe one complete utterance.

        ``hint_language`` is a hint only. Passing it must not disable language
        identification: the returned ``Transcript.language`` is authoritative
        and drives routing, so a backend that pins the language to the hint
        breaks requirement 5 and must ignore the hint instead.
        """


@runtime_checkable
class SpeakerEmbedder(Protocol):
    """Speaker embedding for diarization by clustering finished utterances."""

    name: str
    #: Shortest audio that yields a usable vector. Segments below this are not
    #: embedded at all -- a vector from 300 ms of speech is noise with a norm.
    min_seconds: float

    async def embed(self, audio: AudioChunk) -> SpeakerEmbedding: ...


@runtime_checkable
class MtBackend(Protocol):
    """Text translation. Normally our own LLM over the OpenAI-compatible API."""

    name: str

    def supported_languages(self) -> Optional[Iterable[str]]:
        """Languages, or ``None`` for an explicit unconstrained claim."""

    async def translate(
        self,
        text: str,
        source: str,
        target: str,
        *,
        context: Optional[Sequence[Tuple[str, str]]] = None,
    ) -> str: ...

    async def translate_stream(
        self,
        text: str,
        source: str,
        target: str,
        *,
        context: Optional[Sequence[Tuple[str, str]]] = None,
    ) -> AsyncIterator[str]:
        """Yield translated text incrementally, so TTS can start early.

        Implementations yield whatever the transport gives them; the caller
        (``session.py``) is responsible for regrouping into synthesizable
        units. Yielding once with the whole string is a valid implementation.

        ``context`` is the conversation so far in THIS direction, oldest
        first, as ``(source_text, target_text)`` pairs. It is passed per call
        and never accumulated by the backend: one backend instance serves
        every session in the process, so state kept here would be one
        conversation's words leaking into another's prompt.
        """
        ...


@runtime_checkable
class TtsBackend(Protocol):
    """Zero-shot cross-lingual voice cloning synthesis."""

    name: str
    #: Native output sample rate.
    sample_rate: int
    #: Reference audio shorter than this is refused rather than cloned badly.
    min_reference_seconds: float

    def supported_languages(self) -> Iterable[str]:
        """Languages this checkpoint can SPEAK (not merely clone from)."""

    async def synthesize(
        self,
        text: str,
        language: str,
        reference: AudioChunk,
        reference_text: Optional[str] = None,
        voice_id: Optional[str] = None,
    ) -> AsyncIterator[AudioChunk]:
        """Stream synthesized audio for ``text`` in ``reference``'s voice.

        ``voice_id`` names a voice the backend already holds -- a preset
        registered once with a serving-side voice registry. When set, the
        backend may skip re-uploading ``reference``; when None, the reference
        clip is the whole specification. Backends without a registry ignore it
        and clone from the clip every time, which is correct, just slower.

        ``reference_text`` is the transcript of the reference audio; some
        backends (F5-TTS class) require it, others (CosyVoice cross-lingual
        mode) do not. The caller always has it -- it is the ASR output of the
        segment the reference was cut from -- so it is passed unconditionally
        and ignored by backends that do not need it.

        Cross-lingual is the normal case here: ``reference`` is the speaker
        talking in their own language, ``language`` is the *other* one.
        """
        ...


# ---------------------------------------------------------------------------
# Hermetic fakes
# ---------------------------------------------------------------------------


def _tone(frequency: float, seconds: float, sample_rate: int) -> np.ndarray:
    t = np.arange(int(seconds * sample_rate), dtype=np.float32) / sample_rate
    return (0.3 * np.sin(2.0 * math.pi * frequency * t)).astype(np.float32)


class FakeAsr:
    """Deterministic recognizer driven by a scripted queue or by tone pitch.

    Two modes, both used by the suite:

    * ``script``: pop the next ``(text, language)`` per call. Lets a test
      state exactly what the conversation says.
    * pitch mode (empty script): the language is derived from the dominant
      frequency of the audio, which makes it possible to build a synthetic
      two-speaker two-language conversation out of tones and assert that
      routing followed the *audio* rather than a configured default.
    """

    def __init__(
        self,
        languages: Sequence[str],
        script: Optional[Sequence[Tuple[str, str]]] = None,
        pitch_map: Optional[Sequence[Tuple[float, str]]] = None,
        confidence: float = 0.95,
    ) -> None:
        self.name = "fake-asr"
        self._languages = list(languages)
        self._script: List[Tuple[str, str]] = list(script or [])
        self._pitch_map = list(pitch_map or [])
        self._confidence = confidence
        self.calls: List[Tuple[float, Optional[str]]] = []
        #: Whatever the session last pushed as the constrained-detection
        #: whitelist. Recorded rather than acted on: the point of the fake is
        #: to prove the WIRING, and a fake that also reimplemented the
        #: restriction would be testing itself.
        self.restrict_languages: Tuple[str, ...] = ()

    def supported_languages(self) -> Iterable[str]:
        return tuple(self._languages)

    def set_restrict_languages(self, codes: Sequence[str]) -> None:
        self.restrict_languages = tuple(codes)

    async def transcribe(
        self, audio: AudioChunk, hint_language: Optional[str] = None
    ) -> Transcript:
        self.calls.append((audio.duration_s, hint_language))
        if self._script:
            text, language = self._script.pop(0)
            return Transcript(text, language, self._confidence)
        if not self._pitch_map:
            raise BackendError("asr", "fake recognizer has no script left")
        peak = _dominant_frequency(audio)
        best = min(self._pitch_map, key=lambda pair: abs(pair[0] - peak))
        return Transcript(
            text=f"utterance@{int(peak)}Hz", language=best[1], language_confidence=self._confidence
        )


def _dominant_frequency(audio: AudioChunk) -> float:
    if len(audio.samples) < 2:
        return 0.0
    spectrum = np.abs(np.fft.rfft(audio.samples))
    freqs = np.fft.rfftfreq(len(audio.samples), 1.0 / audio.sample_rate)
    return float(freqs[int(np.argmax(spectrum))])


class FakeEmbedder:
    """Embeds a tone by its pitch, so identical pitches cluster together.

    The vector is a narrow Gaussian bump over a log-frequency axis: two
    segments of the same tone land on top of each other, two different tones
    land far apart, and the cosine threshold in ``speakers.py`` is exercised
    for real rather than mocked out.
    """

    def __init__(self, dim: int = 32, min_seconds: float = 0.5) -> None:
        self.name = "fake-embedder"
        self.min_seconds = min_seconds
        self._dim = dim

    async def embed(self, audio: AudioChunk) -> SpeakerEmbedding:
        if audio.duration_s < self.min_seconds:
            raise BackendError(
                "embedder",
                f"segment is {audio.duration_s:.2f}s, need >= {self.min_seconds}s",
            )
        peak = max(_dominant_frequency(audio), 1.0)
        axis = np.linspace(math.log(50.0), math.log(2000.0), self._dim)
        bump = np.exp(-0.5 * ((axis - math.log(peak)) / 0.05) ** 2)
        bump = bump + 1e-3
        return SpeakerEmbedding(bump.astype(np.float32))


class FakeMt:
    """Translates by tagging, so a test can assert the direction verbatim.

    ``"hallo"`` from ``de`` to ``es`` becomes ``"[de>es] hallo"``. No language
    pair is special-cased anywhere in the fake, which is what makes it usable
    as the no-hardcoded-pair falsifier: the same fake serves ``ja>fr``.
    """

    def __init__(
        self,
        languages: Optional[Sequence[str]] = None,
        chunk_size: int = 0,
        fail_on: Optional[str] = None,
    ) -> None:
        self.name = "fake-mt"
        self._languages = None if languages is None else list(languages)
        self._chunk_size = chunk_size
        self._fail_on = fail_on
        self.calls: List[Tuple[str, str, str]] = []
        self.contexts: List[List[Tuple[str, str]]] = []

    def supported_languages(self) -> Optional[Iterable[str]]:
        return None if self._languages is None else tuple(self._languages)

    async def translate(
        self,
        text: str,
        source: str,
        target: str,
        *,
        context: Optional[Sequence[Tuple[str, str]]] = None,
    ) -> str:
        self.calls.append((text, source, target))
        # Recorded so a test can assert what the SESSION built, which is the
        # only place the per-direction context is decided.
        self.contexts.append(list(context or ()))
        if self._fail_on is not None and self._fail_on in text:
            raise BackendError("mt", f"scripted failure on {self._fail_on!r}")
        return f"[{source}>{target}] {text}"

    async def translate_stream(
        self,
        text: str,
        source: str,
        target: str,
        *,
        context: Optional[Sequence[Tuple[str, str]]] = None,
    ) -> AsyncIterator[str]:
        whole = await self.translate(text, source, target, context=context)
        if self._chunk_size <= 0:
            yield whole
            return
        for start in range(0, len(whole), self._chunk_size):
            yield whole[start : start + self._chunk_size]


class FakeTts:
    """Synthesizes a tone whose pitch is copied from the reference audio.

    This is the property the whole voice-preservation requirement rests on,
    reduced to something assertable without a vocoder: the output carries the
    *reference speaker's* pitch, and a test can check that speaker B's
    translated turn came out at speaker B's frequency and not speaker A's.
    """

    def __init__(
        self,
        languages: Sequence[str],
        sample_rate: int = 24000,
        min_reference_seconds: float = 3.0,
        chunk_seconds: float = 0.5,
        seconds_per_char: float = 0.06,
    ) -> None:
        self.name = "fake-tts"
        self.sample_rate = sample_rate
        self.min_reference_seconds = min_reference_seconds
        self._languages = list(languages)
        self._chunk_seconds = chunk_seconds
        self._seconds_per_char = seconds_per_char
        self.calls: List[Tuple[str, str, float, Optional[str]]] = []
        self.voice_ids: List[Optional[str]] = []

    def supported_languages(self) -> Iterable[str]:
        return tuple(self._languages)

    async def synthesize(
        self,
        text: str,
        language: str,
        reference: AudioChunk,
        reference_text: Optional[str] = None,
        voice_id: Optional[str] = None,
    ) -> AsyncIterator[AudioChunk]:
        self.calls.append((text, language, reference.duration_s, reference_text))
        self.voice_ids.append(voice_id)
        if language not in self._languages:
            raise BackendError("tts", f"checkpoint cannot speak {language!r}")
        if reference.duration_s < self.min_reference_seconds:
            raise BackendError(
                "tts",
                f"reference is {reference.duration_s:.2f}s, need "
                f">= {self.min_reference_seconds}s",
            )
        pitch = max(_dominant_frequency(reference), 1.0)
        total = max(self._chunk_seconds, len(text) * self._seconds_per_char)
        emitted = 0.0
        while emitted < total:
            span = min(self._chunk_seconds, total - emitted)
            yield AudioChunk(_tone(pitch, span, self.sample_rate), self.sample_rate)
            emitted += span
