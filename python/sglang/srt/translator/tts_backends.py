# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""The real synthesizer: an HTTP client, not an in-process model.

The chosen TTS (Qwen3-TTS-12Hz-0.6B-Base, Apache-2.0 on code and weights) runs
behind **vLLM-Omni** in its own virtual environment and is reached over the
stock OpenAI ``POST /v1/audio/speech`` surface. That is not a stylistic
preference; it is forced. Every candidate TTS package pins a `transformers`
version that conflicts with the one this venv carries for sglang, so a single
environment cannot hold both. Making the boundary an HTTP hop turns a
dependency conflict into a process boundary, which is the same move the rest of
the project makes for the LLM itself (client-compatibility principle: we call
our own services the way a stranger would).

Two vLLM-Omni specifics this adapter is built around:

* **The voice registry.** OpenAI's schema has ``voice`` as a preset *string*,
  with nowhere to put a reference clip. vLLM-Omni resolves this with
  ``/v1/audio/voices``: upload a clip once, get a name back, then pass that
  name as ``voice``. So zero-shot cloning becomes "register, then synthesize",
  and the adapter caches registrations by content hash — a speaker's rolling
  reference buffer changes every few turns, and re-uploading an unchanged clip
  on every utterance would add a round trip to the critical path.
* **Cross-lingual mode.** The mechanism all four leading cloners share is
  keeping the reference transcript out of the LM context. Qwen3-TTS spells it
  ``x_vector_only_mode``. It is exposed here as a constructor flag rather than
  hardcoded, because it is precisely the comparison the GPU ticket runs.

The language set is read from the checkpoint's own ``config.json``
(``talker_config.codec_language_id``), which is the machine-readable source
requirement 5 asks for -- not a list transcribed into this file.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import io
import json
import logging
from pathlib import Path
from typing import AsyncIterator, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from sglang.srt.translator.backends import AudioChunk, BackendError

logger = logging.getLogger(__name__)

__all__ = [
    "QWEN3_TTS_LANGUAGE_NAMES",
    "OpenAiSpeechTts",
    "TtsHttpConfig",
    "languages_from_qwen3_tts_config",
]


# The checkpoint names its languages in English; the rest of the system speaks
# ISO 639-1. This is the only place the two vocabularies meet, and it is a
# translation table rather than a language list: the LIST comes from the
# checkpoint, this only decodes what the checkpoint said.
QWEN3_TTS_LANGUAGE_NAMES: Dict[str, str] = {
    "chinese": "zh",
    "english": "en",
    "german": "de",
    "italian": "it",
    "portuguese": "pt",
    "spanish": "es",
    "japanese": "ja",
    "korean": "ko",
    "french": "fr",
    "russian": "ru",
    "arabic": "ar",
    "dutch": "nl",
    "polish": "pl",
    "turkish": "tr",
    "vietnamese": "vi",
    "indonesian": "id",
    "thai": "th",
    "hindi": "hi",
}


def languages_from_qwen3_tts_config(model_dir: Path) -> Tuple[str, ...]:
    """Read the checkpoint's own language table.

    ``talker_config.codec_language_id`` maps an English language name to the
    codec token id the talker conditions on. A language absent from that table
    cannot be synthesized at all, so this IS the capability set -- reading it
    means swapping the checkpoint changes what the system advertises without a
    code edit.

    An unrecognised name is passed through lowercased rather than dropped: a
    future checkpoint adding a language we have no mapping for should show up
    as an odd code in the matrix, not silently vanish from it.
    """
    config_path = Path(model_dir) / "config.json"
    if not config_path.exists():
        raise BackendError(
            "tts", f"no config.json under {model_dir}; cannot read the language set"
        )
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        table = config["talker_config"]["codec_language_id"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise BackendError(
            "tts",
            f"{config_path} has no talker_config.codec_language_id: {exc}",
        ) from exc
    codes = []
    for name in table:
        key = str(name).strip().lower()
        codes.append(QWEN3_TTS_LANGUAGE_NAMES.get(key, key))
    return tuple(sorted(set(codes)))


@dataclasses.dataclass(frozen=True)
class TtsHttpConfig:
    """Where the synthesizer is and how it is asked."""

    base_url: str = "http://127.0.0.1:30810/v1"
    model: str = "qwen3-tts"
    api_key: Optional[str] = None
    #: Native output rate of the checkpoint. The session resamples once at the
    #: edge if the transport wants something else.
    sample_rate: int = 24000
    #: Reference audio shorter than this is refused rather than cloned badly.
    min_reference_seconds: float = 3.0
    #: Per-request timeout. A synthesis that has not started in this long has
    #: already missed the conversation.
    timeout_s: float = 30.0
    #: Local checkpoint directory, read ONLY for the language table. The
    #: weights are loaded by the serving process, not by us.
    model_dir: Optional[Path] = None
    #: Explicit override, for a backend whose config we cannot read (a remote
    #: server, say). Skips the config.json lookup entirely.
    languages: Optional[Sequence[str]] = None
    #: Cross-lingual mode: keep the reference transcript out of the LM context.
    #: The mechanism every leading zero-shot cloner relies on for
    #: reference-language != output-language. Off means in-context learning,
    #: which is the arm the GPU ticket compares against.
    x_vector_only_mode: bool = True
    #: Streamed response format. vLLM-Omni emits raw little-endian PCM16 when
    #: asked for "pcm", which is what lets audio start before synthesis ends;
    #: "wav" would require the whole file before the header is meaningful.
    response_format: str = "pcm"
    #: How many registered voices to remember before evicting the oldest.
    voice_cache_size: int = 32


def _reference_key(audio: AudioChunk) -> str:
    """Content hash of a reference clip, so an unchanged clip is not re-sent."""
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(audio.samples, dtype=np.float32).tobytes())
    digest.update(str(audio.sample_rate).encode("ascii"))
    return digest.hexdigest()[:24]


def _wav_bytes(audio: AudioChunk) -> bytes:
    """Encode a clip as a WAV upload payload.

    ``soundfile`` rather than a hand-rolled header: the registry accepts a file
    upload and getting a RIFF header subtly wrong produces a server-side error
    that reads like a model failure.
    """
    import soundfile

    buffer = io.BytesIO()
    soundfile.write(
        buffer,
        np.clip(audio.samples, -1.0, 1.0),
        audio.sample_rate,
        format="WAV",
        subtype="PCM_16",
    )
    return buffer.getvalue()


class OpenAiSpeechTts:
    """Zero-shot cloning TTS over ``/v1/audio/speech`` + ``/v1/audio/voices``.

    Implements the :class:`~sglang.srt.translator.backends.TtsBackend` protocol,
    so the session cannot tell it apart from the hermetic fake.
    """

    def __init__(
        self,
        config: Optional[TtsHttpConfig] = None,
        client=None,
    ) -> None:
        self.config = config or TtsHttpConfig()
        self.name = f"openai-speech:{self.config.model}"
        self.sample_rate = self.config.sample_rate
        self.min_reference_seconds = self.config.min_reference_seconds
        self._client = client
        self._owns_client = client is None
        self._voice_cache: Dict[str, str] = {}
        self._registration_lock = asyncio.Lock()

        if self.config.languages is not None:
            self._languages = tuple(str(c) for c in self.config.languages)
        elif self.config.model_dir is not None:
            self._languages = languages_from_qwen3_tts_config(self.config.model_dir)
        else:
            raise BackendError(
                "tts",
                "cannot determine the language set: pass either model_dir (to "
                "read the checkpoint's talker_config.codec_language_id) or an "
                "explicit languages list. Guessing would put unspeakable "
                "languages into the advertised set.",
            )

    # -- capability ---------------------------------------------------------

    def supported_languages(self) -> Iterable[str]:
        return self._languages

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    def _http(self):
        if self._client is None:
            import httpx

            headers = {}
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            self._client = httpx.AsyncClient(
                base_url=self.config.base_url,
                timeout=self.config.timeout_s,
                headers=headers,
            )
        return self._client

    # -- voice registry -----------------------------------------------------

    async def ensure_voice(self, reference: AudioChunk) -> str:
        """Register a reference clip and return its voice name, cached.

        The lock matters: two turns from different speakers can race here, and
        two concurrent uploads of the SAME clip would register two voices and
        waste the cache. Registration is off the hot path for every turn after
        the first with a given clip.
        """
        key = _reference_key(reference)
        cached = self._voice_cache.get(key)
        if cached is not None:
            return cached

        async with self._registration_lock:
            cached = self._voice_cache.get(key)
            if cached is not None:
                return cached
            voice_name = f"ref-{key}"
            try:
                response = await self._http().post(
                    "/audio/voices",
                    files={"file": (f"{voice_name}.wav", _wav_bytes(reference),
                                    "audio/wav")},
                    data={"name": voice_name, "model": self.config.model},
                )
                response.raise_for_status()
            except Exception as exc:
                raise BackendError(
                    "tts", f"registering reference voice failed: {exc}"
                ) from exc

            # Trust the server's name if it renamed the voice; some registries
            # normalise. Falling back to ours keeps a terse server honest.
            try:
                payload = response.json()
                voice_name = str(payload.get("name") or payload.get("voice")
                                 or voice_name)
            except (ValueError, AttributeError):
                pass

            self._voice_cache[key] = voice_name
            while len(self._voice_cache) > self.config.voice_cache_size:
                self._voice_cache.pop(next(iter(self._voice_cache)))
            return voice_name

    # -- synthesis ----------------------------------------------------------

    async def synthesize(
        self,
        text: str,
        language: str,
        reference: AudioChunk,
        reference_text: Optional[str] = None,
        voice_id: Optional[str] = None,
    ) -> AsyncIterator[AudioChunk]:
        """Stream synthesized audio in the reference speaker's voice.

        ``voice_id`` short-circuits registration for a preset the server
        already holds. Everything else goes through the registry.
        """
        text = text.strip()
        if not text:
            return
        if language not in self._languages:
            raise BackendError(
                "tts",
                f"checkpoint cannot speak {language!r}; it speaks "
                f"{list(self._languages)}",
            )
        if voice_id is None:
            if reference is None:
                raise BackendError("tts", "no reference audio and no voice_id")
            if reference.duration_s < self.min_reference_seconds:
                raise BackendError(
                    "tts",
                    f"reference is {reference.duration_s:.2f}s, need "
                    f">= {self.min_reference_seconds}s",
                )
            voice_id = await self.ensure_voice(reference)

        body = {
            "model": self.config.model,
            "input": text,
            "voice": voice_id,
            "response_format": self.config.response_format,
            "language": language,
            "stream": True,
            # Cross-lingual: suppress the reference transcript so the LM does
            # not try to continue the reference's LANGUAGE instead of speaking
            # the requested one. This is the whole mechanism.
            "x_vector_only_mode": self.config.x_vector_only_mode,
        }
        if reference_text and not self.config.x_vector_only_mode:
            body["reference_text"] = reference_text

        carry = b""
        try:
            async with self._http().stream(
                "POST", "/audio/speech", json=body
            ) as response:
                response.raise_for_status()
                async for piece in response.aiter_bytes():
                    if not piece:
                        continue
                    carry += piece
                    # PCM16 is two bytes per sample; a chunk boundary can split
                    # one. Carrying the odd byte is what keeps the stream from
                    # slowly desynchronising into noise.
                    usable = len(carry) - (len(carry) % 2)
                    if usable <= 0:
                        continue
                    frame, carry = carry[:usable], carry[usable:]
                    samples = (
                        np.frombuffer(frame, dtype="<i2").astype(np.float32) / 32768.0
                    )
                    yield AudioChunk(samples, self.sample_rate)
        except BackendError:
            raise
        except Exception as exc:
            raise BackendError(
                "tts", f"synthesis request to {self.config.base_url} failed: {exc}"
            ) from exc
        if carry:
            logger.warning(
                "synthesis stream ended on an odd byte boundary (%d byte left "
                "over); the server's PCM framing may be wrong",
                len(carry),
            )
