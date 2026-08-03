# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Real recognizer, VAD and speaker-embedder adapters.

Every import of a heavy dependency happens inside a constructor, never at
module import time. That is not politeness: the hermetic suite imports this
module under ``CUDA_VISIBLE_DEVICES=99`` to check the registry wiring, and a
top-level ``import nemo`` would make the desk tests require a GPU stack.

Selection guidance, from the #466 component survey (see
``docs/dev/DESIGN_466_live_translator.md`` for the full table and evidence):

* **Whisper is architecturally wrong for this workload and still the safe
  default.** It pads every input to a fixed 30 s mel window, so a 4 s
  conversational turn costs the same encoder pass as a 30 s one. Our entire
  workload is short turns. It is nevertheless the fallback because
  ``faster-whisper`` is MIT, mature, one dependency (CTranslate2) and certain
  to work.
* **Cache-aware streaming CTC/TDT models (Nemotron-3.5-ASR-streaming-0.6B,
  Parakeet-TDT-0.6B-v3) are the right shape**: no 30 s pad, built-in language
  identification, ~2-2.5 GB. Their cost is the NeMo toolkit, which is a heavy
  dependency that must be proven to coexist with the existing venv's torch
  before it can be the default.

Both are implemented here behind the same interface so the choice is a config
string, and the GPU window decides it on measurements rather than on this
comment.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from sglang.srt.translator.backends import (
    AudioChunk,
    BackendError,
    SpeakerEmbedding,
    Transcript,
)

logger = logging.getLogger(__name__)

__all__ = [
    "constrained_language_choice",
    "FasterWhisperAsr",
    "NemoStreamingAsr",
    "SileroVad",
    "OnnxSpeakerEmbedder",
    "WHISPER_LANGUAGES",
    "PARAKEET_V3_LANGUAGES",
]


def _pin_card(card_uuid: Optional[str]) -> None:
    """Pin this process to one physical card before any CUDA context exists.

    Same doctrine as the rest of the project: device identity comes from NVML
    UUIDs, never from a torch ordinal, and isolation happens at the process
    level via ``CUDA_VISIBLE_DEVICES`` rather than through an in-process
    mapping table. Inside the process ``cuda:0`` is then unambiguous.

    A no-op when the process is already pinned or no UUID was configured.
    """
    if not card_uuid:
        return
    current = os.environ.get("CUDA_VISIBLE_DEVICES")
    if current and current != card_uuid:
        logger.warning(
            "CUDA_VISIBLE_DEVICES is already %r; not overriding it with %r. "
            "Pin the process from its launcher instead.",
            current,
            card_uuid,
        )
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = card_uuid


# Whisper's published language table (openai/whisper tokenizer.LANGUAGES).
# Duplicated here rather than imported so that ``supported_languages()`` works
# before the model is loaded -- the language endpoint must answer on a cold
# process. The adapter re-derives it from the installed package when it can,
# and this constant is the floor.
WHISPER_LANGUAGES: Sequence[str] = (
    "en es fr de it pt nl ru zh ja ko ar tr pl ca uk sv he cs da fi el ro id "
    "no hu vi th hi ms bg hr sk ta lt sl lv et sr fa az bn ur kk mk ml gl is "
    "cy ne ka ky af eu mn mr be sq bs sw gu si ta te kn km bo pa am mt lo sn "
    "yo so oc ba br fo ht ln mg mi my ps sa sd su tg tk tl tt uz yi jw am ne "
    "ha lb la nn oc as"
).split()

# Parakeet-TDT-0.6B-v3 / Canary-v2: 25 European locales, published on the card.
PARAKEET_V3_LANGUAGES: Sequence[str] = (
    "bg hr cs da nl en et fi fr de el hu it lv lt mt pl pt ro sk sl es sv ru uk"
).split()

# Nemotron-3.5-ASR-streaming: 40 locales; the transcription-ready tier is what
# we advertise, because the lower tiers are explicitly not production-ready and
# advertising them would put unusable languages in the system's language set.
NEMOTRON_35_LANGUAGES: Sequence[str] = (
    "en es fr de it pt nl pl ru uk cs sk sl hr bg ro hu el da sv no fi et lv "
    "lt mt ga ja ko zh ar he hi tr vi th id ms ca eu"
).split()


def constrained_language_choice(
    detected: str,
    probability: float,
    all_probabilities: Optional[Sequence[Tuple[str, float]]],
    restrict: Sequence[str],
) -> Tuple[str, float]:
    """Pick the best IN-SET language, never discard the utterance.

    The routing table names the languages the conversation can actually route.
    Anything else the identifier proposes is a misdetection by construction --
    a German sentence classified as Dutch costs the entire turn -- so the
    candidate set is narrowed to the table's languages.

    Three cases, and the third is the one worth spelling out:

    1. no restriction: the identifier's answer stands unchanged;
    2. the top answer is in the set: it stands, with its own probability;
    3. the top answer is OUT of the set: fall back to the highest-probability
       in-set language and report ITS probability as the confidence.

    Case 3 is deliberately not "return confidence 0", which the first version
    did. Zero says "we know nothing", and the session's low-confidence
    fallback then reaches for the speaker's history -- fine for a returning
    speaker, useless for the first turn of a new one. Reporting the real in-set
    probability keeps the decision honest AND keeps the number that lets the
    UI tag it as uncertain. Nothing is ever dropped: an utterance always
    resolves to some language in the table.

    When probabilities are unavailable the choice degenerates to the first
    restricted language with a confidence of ``0.0`` -- an explicit guess,
    tagged as one.
    """
    allowed = [str(code).lower() for code in restrict]
    if not allowed:
        return detected, probability
    if detected in allowed:
        return detected, probability
    if all_probabilities:
        in_set = [
            (code, float(score))
            for code, score in all_probabilities
            if str(code).lower() in allowed
        ]
        if in_set:
            best_code, best_score = max(in_set, key=lambda item: item[1])
            return str(best_code).lower(), best_score
    return allowed[0], 0.0


class FasterWhisperAsr:
    """CTranslate2 Whisper. MIT code, MIT weights, one dependency.

    The safe default. ``compute_type='int8_float16'`` is the measured sweet
    spot on consumer cards (1545 MB for large-v3-turbo versus 2537 MB fp16, at
    the same speed).

    ``beam_size=1`` on purpose: beam search buys a fraction of a WER point on
    long-form audio and costs latency on every short turn, and latency is the
    property this feature is judged on.
    """

    def __init__(
        self,
        model: str = "large-v3-turbo",
        device: str = "cuda",
        compute_type: str = "int8_float16",
        download_root: Optional[Path] = None,
        card_uuid: Optional[str] = None,
        restrict_languages: Sequence[str] = (),
        beam_size: int = 1,
        vad_filter: bool = False,
    ) -> None:
        _pin_card(card_uuid)
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise BackendError(
                "asr",
                "faster-whisper is not installed (pip install faster-whisper)",
            ) from exc
        self.name = f"faster-whisper:{model}"
        # TWO DIFFERENT THINGS, and they were one attribute (§19.14).
        #
        # `_deployment_languages` is what this deployment is CONFIGURED to
        # hear -- an operator's bound, set once here, and a genuine capability
        # statement. `_restrict` is the live decode whitelist that
        # `set_restrict_languages` pushes before EVERY recognition from the
        # session's participant set (§17.8.6). Sharing one attribute made the
        # advertised capability collapse onto the current selection: one
        # recognizer serves every session, so `GET /api/translator/languages`
        # answered with whichever conversation spoke last. Measured on the
        # live service: `stages.asr = ["de", "es"]` from a model that hears
        # 102, which is why nothing but German and Spanish was ever
        # selectable.
        self._deployment_languages = [str(c).lower() for c in restrict_languages]
        self._restrict = list(self._deployment_languages)
        self._beam_size = beam_size
        self._vad_filter = vad_filter
        self._model = WhisperModel(
            model,
            device=device,
            compute_type=compute_type,
            download_root=str(download_root) if download_root else None,
        )

    def supported_languages(self) -> Iterable[str]:
        # The DEPLOYMENT bound, never the live decode whitelist. A capability
        # answer that narrows itself to the current selection can only ever
        # confirm the selection -- which is exactly how a picker offering
        # "every supported language" ended up offering the two already in use.
        if self._deployment_languages:
            return tuple(self._deployment_languages)
        try:
            from whisper.tokenizer import LANGUAGES  # type: ignore

            return tuple(LANGUAGES)
        except ImportError:
            return tuple(WHISPER_LANGUAGES)

    async def transcribe(
        self, audio: AudioChunk, hint_language: Optional[str] = None
    ) -> Transcript:
        import asyncio

        # The hint is NOT passed as ``language=``: doing so would pin the
        # direction from OUTSIDE the audio, which is exactly the hardcoded-pair
        # failure requirement 5 forbids. The direction is always discovered
        # from the utterance; what a restriction changes is the candidate set
        # it is discovered from, never that it is discovered.
        del hint_language

        restrict = list(self._restrict)

        def _run():
            chosen: Optional[str] = None
            confidence = 0.0
            if restrict and hasattr(self._model, "detect_language"):
                # TWO PASSES, and this is the whole point of the method.
                #
                # Narrowing the identifier's ANSWER after the fact -- what the
                # first version did -- binds the LABEL and nothing else:
                # `transcribe()` without `language=` runs Whisper's own
                # identification internally and decodes in whatever it picks,
                # across all 98 languages. A German utterance identified as
                # Icelandic therefore came back as Icelandic TEXT wearing a
                # `de` label, and the user reads the text, not the label.
                #
                # So: identify first, narrow that posterior to the routable
                # set, then DECODE with the language constrained
                # identification actually chose. Still discovered, never
                # declared -- discovered from a restricted candidate set.
                # The cost is one extra encoder pass, paid only when a
                # restriction exists; ASR is ~2 % of first-audio latency
                # (DESIGN §18.4), so this buys correctness at a measured
                # ~0.1 s.
                detected, probability, posterior = self._model.detect_language(
                    audio.samples
                )
                if not detected:
                    raise BackendError(
                        "asr", "recognizer returned no language identification"
                    )
                chosen, confidence = constrained_language_choice(
                    str(detected).lower(),
                    float(probability or 0.0),
                    posterior,
                    restrict,
                )
            segments, info = self._model.transcribe(
                audio.samples,
                language=chosen,
                beam_size=self._beam_size,
                vad_filter=self._vad_filter,
                condition_on_previous_text=False,
                task="transcribe",
            )
            text = " ".join(s.text.strip() for s in segments).strip()
            return text, info, chosen, confidence

        text, info, chosen, confidence = await asyncio.get_running_loop().run_in_executor(
            None, _run
        )
        if chosen is not None:
            return Transcript(text, chosen, language_confidence=confidence)
        # Unrestricted (or a model too old to expose `detect_language`): the
        # decode identified the language itself and the narrowing below is a
        # no-op for the empty restriction. Kept as the single-pass path so an
        # unbounded session costs exactly what it always did.
        language = getattr(info, "language", None)
        if not language:
            raise BackendError("asr", "recognizer returned no language identification")
        probability = float(getattr(info, "language_probability", 1.0) or 0.0)
        # Whisper reports the full posterior when asked, which is what makes
        # constrained detection a real narrowing rather than a veto.
        language, probability = constrained_language_choice(
            str(language).lower(),
            probability,
            getattr(info, "all_language_probs", None),
            restrict,
        )
        return Transcript(text, language, language_confidence=probability)

    def set_restrict_languages(self, codes: Sequence[str]) -> None:
        """Narrow language identification at runtime, from the routing table."""
        self._restrict = [str(c).lower() for c in codes]


class NemoStreamingAsr:
    """Cache-aware streaming NeMo recognizer (Nemotron-3.5 / Parakeet class).

    No 30 s pad, so a 4 s turn costs 4 s of encoder rather than 30. Built-in
    language identification. The reason it is not the unconditional default is
    the NeMo toolkit: heavy, torch-version-sensitive, and unproven against
    this venv. That is a GPU-window question, not a desk one.
    """

    def __init__(
        self,
        model: str = "nvidia/nemotron-3.5-asr-streaming-0.6b",
        device: str = "cuda",
        card_uuid: Optional[str] = None,
        restrict_languages: Sequence[str] = (),
        declared_languages: Optional[Sequence[str]] = None,
    ) -> None:
        _pin_card(card_uuid)
        try:
            import nemo.collections.asr as nemo_asr  # type: ignore
        except ImportError as exc:
            raise BackendError(
                "asr",
                "NeMo is not installed (pip install nemo_toolkit[asr]); "
                "use the faster-whisper backend instead",
            ) from exc
        self.name = f"nemo:{model}"
        # Same split as FasterWhisperAsr: the constructor's bound is a
        # capability, the runtime whitelist is a decode constraint.
        self._deployment_languages = [str(c).lower() for c in restrict_languages]
        self._restrict = list(self._deployment_languages)
        if declared_languages is not None:
            self._languages = list(declared_languages)
        elif "parakeet" in model:
            self._languages = list(PARAKEET_V3_LANGUAGES)
        else:
            self._languages = list(NEMOTRON_35_LANGUAGES)
        self._model = nemo_asr.models.ASRModel.from_pretrained(model_name=model)
        self._model = self._model.to(device).eval()

    def supported_languages(self) -> Iterable[str]:
        return tuple(self._deployment_languages or self._languages)

    async def transcribe(
        self, audio: AudioChunk, hint_language: Optional[str] = None
    ) -> Transcript:
        import asyncio

        del hint_language  # see FasterWhisperAsr.transcribe

        def _run():
            # ``target_lang='auto'`` is the built-in LID path; the model emits
            # a locale tag which we collapse to the base code.
            return self._model.transcribe(
                [audio.samples], batch_size=1, target_lang="auto", verbose=False
            )

        results = await asyncio.get_running_loop().run_in_executor(None, _run)
        if not results:
            raise BackendError("asr", "recognizer returned no hypothesis")
        best = results[0]
        text = getattr(best, "text", best if isinstance(best, str) else "")
        language = getattr(best, "langs", None) or getattr(best, "lang", None)
        if isinstance(language, (list, tuple)):
            language = language[0] if language else None
        if not language:
            raise BackendError(
                "asr", "recognizer returned no language identification"
            )
        confidence = float(getattr(best, "lang_prob", 1.0) or 1.0)
        # NeMo's LID emits one locale tag, not a posterior, so constrained
        # detection here can only fall back to the first allowed language with
        # an honest confidence of zero. That is a real limitation of the
        # backend rather than of the contract, and it is why the shared helper
        # takes the posterior as optional rather than assuming one exists.
        #
        # KNOWN GAP, stated rather than papered over: this backend still binds
        # only the LABEL, exactly the defect fixed above for faster-whisper.
        # A second pass would have to re-transcribe with `target_lang=code`,
        # and NeMo is not installed in this venv, so that call cannot be
        # executed or tested here -- writing it blind would be a desk fix. The
        # faster-whisper backend is the shipped one; when NeMo is brought up,
        # this is the first thing to close.
        code, confidence = constrained_language_choice(
            str(language).lower().split("-")[0],
            confidence,
            None,
            self._restrict,
        )
        return Transcript(str(text).strip(), code, confidence)

    def set_restrict_languages(self, codes: Sequence[str]) -> None:
        """Narrow language identification at runtime, from the routing table."""
        self._restrict = [str(c).lower() for c in codes]


class SileroVad:
    """Silero VAD (MIT), the recommended detector. CPU, sub-millisecond.

    Runs on ONNX Runtime, which the venv already has, so it adds no GPU
    footprint and no torch coupling. Silero wants 512-sample frames at 16 kHz
    (32 ms); the segmenter's frame size is therefore adapted here rather than
    forced on the rest of the pipeline.
    """

    #: Silero's native window at 16 kHz is 512 samples = 32 ms.
    NATIVE_SAMPLES = 512

    def __init__(
        self,
        model_path: Optional[Path] = None,
        threshold: float = 0.5,
        frame_ms: int = 32,
    ) -> None:
        try:
            import onnxruntime  # type: ignore
        except ImportError as exc:
            raise BackendError(
                "vad", "onnxruntime is not installed; use EnergyVad instead"
            ) from exc
        if model_path is None or not Path(model_path).exists():
            raise BackendError(
                "vad",
                f"silero_vad.onnx not found at {model_path!r}; download it from "
                "https://github.com/snakers4/silero-vad (MIT)",
            )
        self.frame_ms = frame_ms
        self._threshold = threshold
        options = onnxruntime.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        self._session = onnxruntime.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._carry = np.zeros(0, dtype=np.float32)

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._carry = np.zeros(0, dtype=np.float32)

    def is_speech(self, frame: np.ndarray, sample_rate: int) -> bool:
        buffer = np.concatenate([self._carry, frame.astype(np.float32)])
        decision = False
        n = self.NATIVE_SAMPLES
        offset = 0
        while offset + n <= len(buffer):
            window = buffer[offset : offset + n].reshape(1, -1)
            offset += n
            outputs = self._session.run(
                None,
                {
                    "input": window,
                    "state": self._state,
                    "sr": np.array(sample_rate, dtype=np.int64),
                },
            )
            probability = float(np.asarray(outputs[0]).reshape(-1)[0])
            self._state = np.asarray(outputs[1], dtype=np.float32)
            # A frame counts as speech when ANY native window inside it does:
            # under-triggering costs a clipped word, over-triggering costs a
            # hangover timer reset, and the first is the worse error.
            decision = decision or probability >= self._threshold
        self._carry = buffer[offset:].copy()
        return decision


class OnnxSpeakerEmbedder:
    """WeSpeaker/ECAPA-class speaker embedding through ONNX Runtime.

    Chosen over SpeechBrain because ``onnxruntime`` is already in the venv and
    SpeechBrain is a heavy dependency for one 26 M-parameter forward pass.
    The exported ResNet34-LM (CC-BY-4.0, ungated) and the sherpa-onnx
    3D-Speaker/WeSpeaker exports all fit this adapter; which one is loaded is
    a path, not a code change.

    The 80-dim log-Mel fbank is computed here rather than pulled from
    torchaudio, again to avoid a dependency for something this small.
    """

    def __init__(
        self,
        model_path: Path,
        min_seconds: float = 1.5,
        sample_rate: int = 16000,
        num_mels: int = 80,
        providers: Optional[Sequence[str]] = None,
    ) -> None:
        try:
            import onnxruntime  # type: ignore
        except ImportError as exc:
            raise BackendError("embedder", "onnxruntime is not installed") from exc
        if not Path(model_path).exists():
            raise BackendError("embedder", f"model not found at {model_path!r}")
        self.name = f"onnx-speaker:{Path(model_path).stem}"
        self.min_seconds = min_seconds
        self._sample_rate = sample_rate
        self._num_mels = num_mels
        self._session = onnxruntime.InferenceSession(
            str(model_path),
            providers=list(providers or ["CPUExecutionProvider"]),
        )
        self._input_name = self._session.get_inputs()[0].name
        self._mel_filters = _mel_filterbank(sample_rate, 400, num_mels)

    async def embed(self, audio: AudioChunk) -> SpeakerEmbedding:
        import asyncio

        if audio.duration_s < self.min_seconds:
            raise BackendError(
                "embedder",
                f"segment is {audio.duration_s:.2f}s, need >= {self.min_seconds}s "
                "for a usable speaker vector",
            )
        if audio.sample_rate != self._sample_rate:
            from sglang.srt.translator.audio import resample

            audio = resample(audio, self._sample_rate)

        def _run():
            features = _log_mel(audio.samples, self._mel_filters)
            # Mean normalisation over time: the standard front-end for
            # speaker embedding, and the thing that makes the vector robust to
            # a channel change (different phone, different room).
            features = features - features.mean(axis=0, keepdims=True)
            batch = features[np.newaxis, :, :].astype(np.float32)
            outputs = self._session.run(None, {self._input_name: batch})
            return np.asarray(outputs[0]).reshape(-1)

        vector = await asyncio.get_running_loop().run_in_executor(None, _run)
        return SpeakerEmbedding(vector.astype(np.float32))


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(sample_rate: int, n_fft: int, num_mels: int) -> np.ndarray:
    low, high = 20.0, sample_rate / 2.0
    points = _mel_to_hz(np.linspace(_hz_to_mel(np.array(low)),
                                    _hz_to_mel(np.array(high)), num_mels + 2))
    bins = np.floor((n_fft + 1) * points / sample_rate).astype(int)
    filters = np.zeros((num_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, num_mels + 1):
        left, centre, right = bins[m - 1], bins[m], bins[m + 1]
        if centre == left:
            centre = left + 1
        if right == centre:
            right = centre + 1
        if right >= filters.shape[1]:
            break
        filters[m - 1, left:centre] = (
            np.arange(left, centre) - left
        ) / float(centre - left)
        filters[m - 1, centre:right] = (
            right - np.arange(centre, right)
        ) / float(right - centre)
    return filters


def _log_mel(
    samples: np.ndarray, mel_filters: np.ndarray, n_fft: int = 400, hop: int = 160
) -> np.ndarray:
    if len(samples) < n_fft:
        samples = np.pad(samples, (0, n_fft - len(samples)))
    window = np.hanning(n_fft).astype(np.float32)
    frames: List[np.ndarray] = []
    for start in range(0, len(samples) - n_fft + 1, hop):
        spectrum = np.fft.rfft(samples[start : start + n_fft] * window)
        frames.append(np.abs(spectrum) ** 2)
    if not frames:
        frames = [np.zeros(n_fft // 2 + 1, dtype=np.float32)]
    power = np.stack(frames).astype(np.float32)
    return np.log(np.maximum(power @ mel_filters.T, 1e-10)).astype(np.float32)
