# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Configuration of a translator deployment.

Same budget doctrine as the Class-3 video tenant (``video_enhance/tenant.py``):
the translator runs as its own process with its own CUDA context, pinned to
one physical card by NVML UUID, with an absolute MiB budget that means the
whole budget -- no implicit ceiling, no safety factor, no rounding down.
Leaving headroom is the operator's job, not something this code second-guesses.

The reason the translator is a separate process rather than an ``srt`` tenant
is recorded here so it is not re-litigated: the ASR and TTS stages are not
autoregressive text engines, they do not share the scheduler's admission
contract, and per DESIGN #333 §2.3 the class that would host them (Class 3)
does not exist yet. Building it is an L-effort project; this feature has a
deadline measured in days. So the translator takes the same escape hatch the
video tenant took -- its own process, its own context, coordination through
the VRAM ledger -- and becomes a Class-3 citizen later without its interfaces
changing, because nothing here imports ``srt`` internals.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Sequence

if TYPE_CHECKING:  # pragma: no cover - import cycle avoidance only
    from sglang.srt.translator.idle_park import IdleParkConfig

__all__ = [
    "TranslatorConfig",
    "AsrConfig",
    "TtsConfig",
    "DiarizationConfig",
    "TranslatorConfigError",
]


def _default_idle_park() -> "IdleParkConfig":
    """Imported lazily: ``idle_park`` imports the ledger, the ledger imports
    torch on its movement path, and this module must stay importable on a desk
    that has neither."""
    from sglang.srt.translator.idle_park import IdleParkConfig

    return IdleParkConfig()


class TranslatorConfigError(ValueError):
    """A deployment configuration that cannot work, refused before allocation."""


@dataclasses.dataclass(frozen=True)
class AsrConfig:
    """Recognizer selection. ``backend`` names an entry in the backend registry."""

    backend: str = "fake"
    model: str = ""
    #: Physical card as an NVML UUID, or None to inherit the process pin.
    card_uuid: Optional[str] = None
    budget_mib: int = 3000
    compute_type: str = "int8_float16"
    #: Restrict language identification to this set. Empty means "all the
    #: model supports". Narrowing it is the single most effective accuracy
    #: lever for LID on short utterances: a two-language conversation should
    #: not be able to be mis-identified as Welsh.
    restrict_languages: Sequence[str] = ()
    extra: Dict[str, object] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class TtsConfig:
    """Synthesizer selection."""

    backend: str = "fake"
    model: str = ""
    card_uuid: Optional[str] = None
    budget_mib: int = 4000
    #: Output rate the client will be fed at. Backends that synthesize at a
    #: different native rate are resampled once, at the edge.
    output_sample_rate: int = 24000
    #: Reference audio the backend needs. Enforced by the session before the
    #: call, so a too-short reference produces a documented fallback rather
    #: than a backend exception mid-turn.
    min_reference_seconds: float = 3.0
    #: Default voice mode: "clone" (each speaker in their own voice) or
    #: "preset" (each speaker gets a distinct artificial voice). Per-session
    #: overridable at runtime. User decision 2026-08-03.
    voice_mode: str = "clone"
    #: Directory of preset voices, laid out as
    #: ``<class>/<voice_id>.<language>.wav`` with ``<class>`` one of
    #: man/woman/boy/girl. None disables preset mode, and the automatic
    #: downgrade for an unclonable speaker then falls back to borrowing
    #: another participant's voice, which is worse -- so a real deployment
    #: should always have a pool.
    preset_voice_dir: Optional[Path] = None
    extra: Dict[str, object] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class DiarizationConfig:
    """Speaker embedding selection."""

    backend: str = "fake"
    model: str = ""
    card_uuid: Optional[str] = None
    budget_mib: int = 500
    extra: Dict[str, object] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class TranslatorConfig:
    """One translator deployment."""

    asr: AsrConfig = dataclasses.field(default_factory=AsrConfig)
    tts: TtsConfig = dataclasses.field(default_factory=TtsConfig)
    diarization: DiarizationConfig = dataclasses.field(
        default_factory=DiarizationConfig
    )
    #: Participant languages of the default conversation. This is a DEFAULT,
    #: overridable per session by the client, and it is the only place a
    #: language pair appears anywhere in the system.
    default_participants: Sequence[str] = ("de", "es")
    #: Model cache root. Never inside the repository.
    model_root: Path = Path("/spinning/llm_stuff/translator-models")
    #: Bounded journal for reconnect replay, per session.
    journal_events: int = 512
    journal_audio_mib: int = 24
    #: A session with no traffic for this long is collected. Long enough to
    #: survive a tunnel re-establishment on mobile data, short enough that an
    #: abandoned session does not pin a reference buffer forever.
    session_idle_timeout_s: float = 300.0
    max_sessions: int = 8
    host: str = "127.0.0.1"
    port: int = 30800
    #: #546: give the VRAM back while nobody is talking. ON by default for
    #: this deployment -- the tenant shares a card with the serving engine and
    #: its duty cycle is a conversation an hour at best. Independent of
    #: ``session_idle_timeout_s``, which collects SESSIONS: a parked tenant
    #: keeps every session, its speakers and its transcript. See
    #: :mod:`sglang.srt.translator.idle_park` for why the park threshold is
    #: derived from the traffic rather than being this timeout's twin.
    idle_park: "IdleParkConfig" = dataclasses.field(
        default_factory=lambda: _default_idle_park()
    )

    def total_budget_mib(self) -> int:
        return self.asr.budget_mib + self.tts.budget_mib + self.diarization.budget_mib

    def validate(self) -> None:
        from sglang.srt.translator.voices import VoiceMode, VoicePoolError

        try:
            mode = VoiceMode.parse(self.tts.voice_mode)
        except VoicePoolError as exc:
            raise TranslatorConfigError(str(exc)) from exc
        if mode is VoiceMode.PRESET and self.tts.preset_voice_dir is None:
            raise TranslatorConfigError(
                "tts.voice_mode is 'preset' but tts.preset_voice_dir is not "
                "set; preset mode needs a pool of voices to assign from"
            )
        if len(set(self.default_participants)) < 2:
            raise TranslatorConfigError(
                "default_participants needs at least two distinct languages, got "
                f"{list(self.default_participants)}"
            )
        if self.journal_events < 8:
            raise TranslatorConfigError("journal_events must be at least 8")
        if self.max_sessions < 1:
            raise TranslatorConfigError("max_sessions must be at least 1")
        for name, budget in (
            ("asr", self.asr.budget_mib),
            ("tts", self.tts.budget_mib),
            ("diarization", self.diarization.budget_mib),
        ):
            if budget <= 0:
                raise TranslatorConfigError(f"{name}.budget_mib must be positive")
        try:
            self.idle_park.validate()
        except ValueError as exc:
            raise TranslatorConfigError(str(exc)) from exc
