# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Rung B: the talker and codec as in-process modules, under the ledger.

One process, one ledger, no second engine -- the 2026-08-03 architecture order.
The synthesizer is now an ``nn.Module`` living in the translator's own process
tree, its weights registered as the ``audio_modules`` asset class in the #286
register, parkable and evictable on the same importance ladder as every other
asset in the runtime. That is the whole difference from the revoked sidecar:
not where the code runs, but whether the memory is visible to the thing that
has to arbitrate it.

**Honest scope.** This rung drives the reference modeling code with our own
call sequence rather than reimplementing the talker against our layers. It buys
audible output inside the deadline. What it does not buy, and what #488's
native-lane rung exists for:

* no CUDA-graph capture over the nested code predictor;
* no cross-request batching -- one conversation at a time, which is the MVP;
* **no true incremental streaming.** The reference generates a whole utterance
  and then decodes it. We chunk the finished waveform so the transport and the
  client see the same interface they always did, but the first-audio latency is
  whole-utterance, not first-frame. Since ``mt.py`` already splits translations
  into clauses and synthesizes them one at a time, the practical unit is a
  clause rather than a turn -- which is most of the benefit and none of the
  decode-loop surgery. The gap is stated rather than hidden, and the interface
  does not change when the native lane closes it.

The four modules are registered separately on purpose: the codec decoder alone
is 229 MB and is the module a turn needs LAST, so it is the natural first
victim under pressure.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from pathlib import Path
from typing import AsyncIterator, Dict, Iterable, List, Optional, Sequence

import numpy as np

from sglang.srt.translator.backends import AudioChunk, BackendError
from sglang.srt.translator.ledger import AudioAssetLedger
from sglang.srt.translator.talker_config import TalkerGeometry, read_talker_geometry
from sglang.srt.translator.tts_backends import (
    QWEN3_TTS_LANGUAGE_NAMES,
    languages_from_qwen3_tts_config,
)

logger = logging.getLogger(__name__)

__all__ = ["InProcessQwen3Tts", "InProcessTtsConfig"]


@dataclasses.dataclass(frozen=True)
class InProcessTtsConfig:
    """Where the checkpoint is and how the talker is driven."""

    model_dir: Path = Path("/spinning/llm_stuff/translator-models/qwen3-tts-0.6b-base")
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    sample_rate: int = 24000
    min_reference_seconds: float = 3.0
    #: Cross-lingual mode: keep the reference transcript out of the LM context.
    #: The mechanism every leading zero-shot cloner relies on when the
    #: reference language differs from the output language. The GPU A/B
    #: compares this against in-context learning, so it stays a flag.
    x_vector_only_mode: bool = True
    #: Emit the finished waveform in chunks of this length, so the transport
    #: and the client see a stream. Not true incremental synthesis -- see the
    #: module docstring.
    emit_chunk_seconds: float = 0.4
    max_new_tokens: int = 2048
    temperature: float = 0.9
    top_p: float = 0.9
    #: Park every audio module after a turn. Costs a restore on the next turn
    #: and frees ~2 GB between conversations; the right default for a tenant
    #: sharing a card with a 27B model.
    park_when_idle: bool = False


class InProcessQwen3Tts:
    """Zero-shot cloning TTS as an in-process, ledger-registered module."""

    def __init__(
        self,
        config: Optional[InProcessTtsConfig] = None,
        ledger: Optional[AudioAssetLedger] = None,
    ) -> None:
        self.config = config or InProcessTtsConfig()
        self.name = f"inprocess-qwen3-tts:{self.config.model_dir.name}"
        self.sample_rate = self.config.sample_rate
        self.min_reference_seconds = self.config.min_reference_seconds
        self.ledger = ledger if ledger is not None else AudioAssetLedger()

        # The geometry read validates the M-RoPE mapping and will refuse a
        # checkpoint that would build the wrong rotary. Doing it here means no
        # weight is touched before the trap is ruled out.
        self.geometry: TalkerGeometry = read_talker_geometry(self.config.model_dir)
        self._languages = tuple(
            languages_from_qwen3_tts_config(self.config.model_dir)
        )
        # The checkpoint's own table is name -> ISO code; the reference API
        # takes the ENGLISH NAME ("spanish"), not the code ("es"). Inverting
        # it here keeps the whole rest of the system on ISO codes -- which is
        # what requirement 5's language matrix intersects on -- and confines
        # the model's vocabulary to the one call that needs it.
        self._code_to_name = {
            code: name for name, code in QWEN3_TTS_LANGUAGE_NAMES.items()
        }
        self._model = None
        self._lock = asyncio.Lock()

    # -- capability ---------------------------------------------------------

    def supported_languages(self) -> Iterable[str]:
        return self._languages

    def to_json(self) -> Dict[str, object]:
        return {
            "backend": self.name,
            "loaded": self._model is not None,
            "languages": list(self._languages),
            "x_vector_only_mode": self.config.x_vector_only_mode,
            "geometry": {
                "layers": self.geometry.num_hidden_layers,
                "hidden": self.geometry.hidden_size,
                "code_groups": self.geometry.num_code_groups,
                "frame_hz": self.geometry.frame_rate_hz,
            },
            "ledger": self.ledger.to_json(),
        }

    # -- lifecycle ----------------------------------------------------------

    def load(self) -> None:
        """Load the checkpoint and register every module with the ledger."""
        if self._model is not None:
            return
        import torch

        from sglang.srt.translator.qwen3_tts_compat import (
            ensure_qwen3_tts_importable,
            refresh_rotary_buffers,
        )

        shims = ensure_qwen3_tts_importable()
        logger.info("qwen3-tts compat shims: %s", ", ".join(shims))

        from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

        dtype = getattr(torch, self.config.dtype)
        kwargs = {"dtype": dtype}
        # device_map routes through accelerate, which this venv does not carry
        # and which is not worth adding for a single-device tenant. On CUDA it
        # is the convenient path; on CPU we load plainly and move afterwards.
        if self.config.device.startswith("cuda"):
            kwargs["device_map"] = self.config.device
        self._model = Qwen3TTSModel.from_pretrained(
            str(self.config.model_dir), **kwargs
        )
        if not self.config.device.startswith("cuda"):
            inner = getattr(self._model, "model", self._model)
            if hasattr(inner, "to"):
                inner.to(self.config.device)
        # Non-persistent rotary buffers do not survive 5.x's meta-device
        # construction; unrefreshed they are NaN and the failure only surfaces
        # as a NaN probability tensor at sampling time. See
        # qwen3_tts_compat.refresh_rotary_buffers.
        inner = getattr(self._model, "model", self._model)
        refreshed = refresh_rotary_buffers(inner)
        logger.info("refreshed %d rotary buffers", refreshed)
        self._register_modules()

    def _register_modules(self) -> None:
        """Register the four weight blocks as separate ledgered assets.

        Separate, not one blob: they have different sizes and different
        last-needed points in a turn, and the register can only make a good
        victim choice at a grain that reflects that.
        """
        inner = getattr(self._model, "model", self._model)
        candidates = [
            ("talker_trunk", self._resolve(inner, "talker.model")),
            ("code_predictor", self._resolve(inner, "talker.code_predictor")),
            ("speaker_encoder", self._resolve(inner, "speaker_encoder")),
            ("codec_decoder", self._resolve(inner, "code2wav")),
        ]
        for name, module in candidates:
            if module is None:
                logger.warning(
                    "audio module %s not found on the loaded model; it will "
                    "not be ledger-visible and cannot be parked",
                    name,
                )
                continue
            try:
                self.ledger.register(name, module)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("could not register %s with the ledger: %s", name, exc)

    @staticmethod
    def _resolve(root: object, dotted: str) -> Optional[object]:
        node = root
        for part in dotted.split("."):
            node = getattr(node, part, None)
            if node is None:
                return None
        return node

    def park(self) -> int:
        """Park every audio module. Returns bytes freed."""
        return self.ledger.park_all()

    def ensure_resident(self) -> Dict[str, float]:
        return self.ledger.ensure_resident()

    # -- synthesis ----------------------------------------------------------

    async def synthesize(
        self,
        text: str,
        language: str,
        reference: AudioChunk,
        reference_text: Optional[str] = None,
        voice_id: Optional[str] = None,
    ) -> AsyncIterator[AudioChunk]:
        text = text.strip()
        if not text:
            return
        if language not in self._languages:
            raise BackendError(
                "tts",
                f"checkpoint cannot speak {language!r}; it speaks "
                f"{list(self._languages)}",
            )
        if reference is None:
            raise BackendError("tts", "in-process cloning needs reference audio")
        if reference.duration_s < self.min_reference_seconds:
            raise BackendError(
                "tts",
                f"reference is {reference.duration_s:.2f}s, need "
                f">= {self.min_reference_seconds}s",
            )

        # One turn at a time: the modules are shared mutable state and a
        # second concurrent generate would interleave KV caches.
        async with self._lock:
            waveform = await asyncio.get_running_loop().run_in_executor(
                None, self._generate, text, language, reference, reference_text
            )

        chunk = max(1, int(self.config.emit_chunk_seconds * self.sample_rate))
        for start in range(0, len(waveform), chunk):
            yield AudioChunk(waveform[start : start + chunk], self.sample_rate)

        if self.config.park_when_idle:
            self.park()

    def _generate(
        self,
        text: str,
        language: str,
        reference: AudioChunk,
        reference_text: Optional[str],
    ) -> np.ndarray:
        import torch

        if self._model is None:
            self.load()
        self.ensure_resident()

        reference_np = np.asarray(reference.samples, dtype=np.float32)
        # The wrapper builds the voice-clone prompt itself from (audio, sr).
        # Going through create_voice_clone_prompt() and passing the result as
        # voice_clone_prompt= is the documented alternative, but it hands the
        # talker a differently-shaped prompt and fails deep inside the text
        # projection -- so the simple form is also the correct one here.
        model_language = self._code_to_name.get(language, language)
        with torch.inference_mode():
            output = self._model.generate_voice_clone(
                text=[text],
                language=[model_language],
                ref_audio=[(reference_np, reference.sample_rate)],
                ref_text=[reference_text or ""],
                x_vector_only_mode=self.config.x_vector_only_mode,
                non_streaming_mode=True,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=True,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            )
        return self._to_waveform(output)

    @staticmethod
    def _to_waveform(output: object) -> np.ndarray:
        """Coerce whatever the reference returned into mono float32.

        Written defensively on purpose: the return shape is the least stable
        part of the reference API, and a wrong-shape coercion is a silent
        audio bug rather than a crash.
        """
        import torch

        candidate = output
        # Documented return is (List[np.ndarray], sample_rate); take the list
        # and ignore the rate, which the caller already knows.
        if (
            isinstance(candidate, tuple)
            and len(candidate) == 2
            and isinstance(candidate[1], (int, float))
        ):
            candidate = candidate[0]
        for attribute in ("audio_values", "waveform", "audios", "audio"):
            if hasattr(candidate, attribute):
                candidate = getattr(candidate, attribute)
                break
        if isinstance(candidate, dict):
            for key in ("audio_values", "waveform", "audio"):
                if key in candidate:
                    candidate = candidate[key]
                    break
        parts: List[np.ndarray] = []

        def flatten(item) -> None:
            if item is None:
                return
            if isinstance(item, (list, tuple)):
                for sub in item:
                    flatten(sub)
                return
            if torch.is_tensor(item):
                parts.append(item.detach().float().cpu().numpy().reshape(-1))
                return
            if isinstance(item, np.ndarray):
                parts.append(item.astype(np.float32).reshape(-1))

        flatten(candidate)
        if not parts:
            raise BackendError(
                "tts",
                f"could not find audio in the generation result "
                f"({type(output).__name__}); the reference API's return shape "
                "changed and coercing it blindly would be a silent audio bug",
            )
        return np.concatenate(parts).astype(np.float32)


def build_inprocess_tts(
    model_dir: Path,
    device: str = "cuda:0",
    languages: Optional[Sequence[str]] = None,
    **kwargs,
) -> InProcessQwen3Tts:
    """Convenience constructor used by the launcher."""
    del languages  # the checkpoint is the authority; see talker_config
    return InProcessQwen3Tts(
        InProcessTtsConfig(model_dir=Path(model_dir), device=device, **kwargs)
    )
