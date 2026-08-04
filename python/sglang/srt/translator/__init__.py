# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Live speech-to-speech translation, voice-preserving, language-pair-agnostic.

Task #466. A phone streams microphone audio over a tunnel; each closed
utterance is recognized, attributed to a speaker, translated by the htsglang
server over its own OpenAI-compatible API, and re-synthesized in *that
speaker's* voice. Everyone keeps their voice and speaks a different language.

The package deliberately imports nothing from ``srt`` internals. Like the
Class-3 video tenant it runs as its own process with its own CUDA context and
its own budget, and it talks to the LLM as an ordinary HTTP client would --
which is both the client-compatibility principle and the reason this can be
developed while the cards are busy.

Nothing here is imported eagerly except the pure-Python core: the backend
adapters pull heavy dependencies inside their constructors, so
``import sglang.srt.translator`` is cheap and GPU-free.
"""

from sglang.srt.translator.backends import (
    AudioChunk,
    BackendError,
    SpeakerEmbedding,
    Transcript,
)
from sglang.srt.translator.config import (
    AsrConfig,
    DiarizationConfig,
    TranslatorConfig,
    TranslatorConfigError,
    TtsConfig,
)
from sglang.srt.translator.languages import (
    ConversationLanguages,
    LanguageError,
    LanguageMatrix,
    canonical_code,
)
from sglang.srt.translator.segmenter import (
    Segment,
    SegmenterConfig,
    SegmentReason,
    TurnSegmenter,
)
from sglang.srt.translator.session import (
    Event,
    EventKind,
    Journal,
    SessionManager,
    TranslatorSession,
    TurnResult,
)
from sglang.srt.translator.speakers import (
    SpeakerProfile,
    SpeakerRegistry,
    SpeakerRegistryConfig,
)

__all__ = [
    "AsrConfig",
    "AudioChunk",
    "BackendError",
    "ConversationLanguages",
    "DiarizationConfig",
    "Event",
    "EventKind",
    "Journal",
    "LanguageError",
    "LanguageMatrix",
    "Segment",
    "SegmentReason",
    "SegmenterConfig",
    "SessionManager",
    "SpeakerEmbedding",
    "SpeakerProfile",
    "SpeakerRegistry",
    "SpeakerRegistryConfig",
    "Transcript",
    "TranslatorConfig",
    "TranslatorConfigError",
    "TranslatorSession",
    "TtsConfig",
    "TurnResult",
    "TurnSegmenter",
    "canonical_code",
]
