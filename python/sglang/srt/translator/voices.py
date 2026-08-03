# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Voice selection: clone the speaker, or give them a preset that suits them.

User decision, 2026-08-03: voice cloning is a per-session OPTION, not the only
mode. Two modes, switchable at runtime:

``clone``
    Each speaker's translated turn is synthesized from THEIR reference audio.
    Accent carry-over from the speaker's own language into the target language
    is explicitly wanted (same dated decision) -- it keeps the speaker
    recognisable and reads as authentic, so nothing here tries to suppress it.

``preset``
    Each detected speaker is assigned a distinct ARTIFICIAL voice from a pool.
    Two requirements drive the design, and they pull in different directions:

    * the listener must always know WHICH speaker is being translated, so the
      assigned voices must be clearly distinguishable from each other;
    * the preset should imitate the speaker's broad voice class (man / woman /
      boy / girl), so the conversation still feels like the room it came from.

    Preset voices speak the target language natively, so unlike clone mode
    there is no accent to carry. That is the honest trade: preset mode buys
    intelligibility and speaker-distinctness at the cost of identity.

``preset`` is also the automatic degradation path. A speaker whose reference
buffer is too short or too noisy to clone gets a preset instead of silence or
a stranger's voice, and the downgrade is recorded per speaker so the client
can show it.

**On the voice classifier.** The class is inferred from median fundamental
frequency, which is a genuinely good adult male/female discriminator and a
good adult/child one. It is labelled heuristic everywhere because it is:
F0 distributions overlap, and boy versus girl is not recoverable from F0 at
all before puberty -- the literature is clear that prepubescent voices are
largely indistinguishable on acoustic grounds. So the classifier returns
``CHILD``, and a pool entry tagged ``BOY`` or ``GIRL`` both match it. Anything
finer is a guess dressed as a measurement, and the API lets a session override
the class per speaker instead.
"""

from __future__ import annotations

import dataclasses
import enum
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

import numpy as np

from sglang.srt.translator.backends import AudioChunk

logger = logging.getLogger(__name__)

__all__ = [
    "VoiceMode",
    "OutputMode",
    "VoiceClass",
    "PresetVoice",
    "VoicePool",
    "VoiceAssignment",
    "VoiceClassifier",
    "F0VoiceClassifier",
    "estimate_median_f0",
    "VoicePoolError",
]


class VoicePoolError(RuntimeError):
    """The preset pool cannot satisfy a request, stated rather than papered over."""


class VoiceMode(str, enum.Enum):
    CLONE = "clone"
    PRESET = "preset"

    @classmethod
    def parse(cls, value: object) -> "VoiceMode":
        if isinstance(value, cls):
            return value
        text = str(value or "").strip().lower()
        for mode in cls:
            if mode.value == text:
                return mode
        raise VoicePoolError(
            f"unknown voice mode {value!r}; expected one of "
            f"{[m.value for m in cls]}"
        )


class OutputMode(str, enum.Enum):
    """Whether a session speaks its translations aloud (§17.1).

    ``silent`` is the reading mode: the pipeline is identical up to and
    excluding synthesis, so the transcript, the speaker attribution and the
    reference buffers all keep filling. It exists for the case where a phone
    is passed around a quiet table, and it is never selected automatically --
    a TTS failure stays a per-turn failure with a reason rather than silently
    becoming a mode.
    """

    VOICE = "voice"
    SILENT = "silent"

    @classmethod
    def parse(cls, value: object) -> "OutputMode":
        if isinstance(value, cls):
            return value
        text = str(value or "").strip().lower()
        for mode in cls:
            if mode.value == text:
                return mode
        raise VoicePoolError(
            f"unknown output mode {value!r}; expected one of "
            f"{[m.value for m in cls]}"
        )


class VoiceClass(str, enum.Enum):
    """Broad voice class. Coarse on purpose -- see the module docstring."""

    MAN = "man"
    WOMAN = "woman"
    #: What the classifier returns for a child. Pool entries tagged BOY or
    #: GIRL both match it, because F0 cannot separate them.
    CHILD = "child"
    BOY = "boy"
    GIRL = "girl"
    UNKNOWN = "unknown"

    def matches(self, other: "VoiceClass") -> bool:
        """Whether a pool entry of class ``self`` may serve a speaker ``other``."""
        if self is other:
            return True
        child_like = {VoiceClass.CHILD, VoiceClass.BOY, VoiceClass.GIRL}
        if self in child_like and other in child_like:
            return True
        return VoiceClass.UNKNOWN in (self, other)


@dataclasses.dataclass(frozen=True)
class PresetVoice:
    """One artificial voice available for assignment.

    A preset is expressed as reference audio per language rather than as an
    opaque backend voice name, so it works with ANY zero-shot cloning backend
    without an interface change: preset mode is clone mode pointed at a
    curated clip instead of at the speaker.

    ``references`` keyed by language is what removes the accent: a preset
    speaking the target language natively has no source language to carry.
    A pool with only one language still works and simply carries that
    language's accent, which is a quality note, not a failure.

    ``backend_voice_id`` is the escape hatch for backends that keep their own
    voice registry (vLLM-Omni's ``/v1/audio/voices``, for one); when set, it
    is passed through and the clips are only a fallback.
    """

    voice_id: str
    label: str
    voice_class: VoiceClass
    references: Dict[str, AudioChunk] = dataclasses.field(default_factory=dict)
    reference_texts: Dict[str, str] = dataclasses.field(default_factory=dict)
    backend_voice_id: Optional[str] = None

    def languages(self) -> Tuple[str, ...]:
        return tuple(sorted(self.references))

    def reference_for(self, language: str) -> Tuple[Optional[AudioChunk], str]:
        """Clip for ``language``, else any clip, else nothing.

        Returning the wrong-language clip rather than refusing is deliberate:
        an accented preset is a far better outcome than a silent turn, and the
        caller records which case it got.
        """
        if language in self.references:
            return self.references[language], self.reference_texts.get(language, "")
        for code, audio in sorted(self.references.items()):
            return audio, self.reference_texts.get(code, "")
        return None, ""

    def speaks_natively(self, language: str) -> bool:
        return language in self.references


@dataclasses.dataclass(frozen=True)
class VoiceAssignment:
    """The outcome of choosing a voice for one turn. Reported on the event."""

    mode: VoiceMode
    reference: Optional[AudioChunk]
    reference_text: str
    #: Set in preset mode.
    preset: Optional[PresetVoice] = None
    #: True when preset mode was NOT what the session asked for.
    downgraded: bool = False
    #: Why the downgrade happened, for the client to show.
    reason: str = ""
    #: Preset mode only: whether the preset speaks the target language natively.
    native_language: bool = True
    #: 0 = the preset as recorded. Above 0 means the pool ran out of distinct
    #: voices for this class and this speaker shares a base voice, shifted.
    variant_index: int = 0
    #: The deterministic shift applied to the reference for a shared voice.
    pitch_shift_semitones: float = 0.0
    #: Operator-facing notice, empty when nothing needs saying. Surfaced in the
    #: UI: a shared base voice is a real degradation of speaker distinctness
    #: and the listener should be told rather than left to wonder.
    notice: str = ""

    @property
    def backend_voice_id(self) -> Optional[str]:
        """A serving-side voice name, when the chosen preset has one.

        Only meaningful in preset mode, and only for a preset registered with
        the backend's voice registry. A pitch-shifted VARIANT deliberately does
        not use it: the registered voice is the unshifted base, so a variant
        must go through the shifted reference clip instead or every sharer
        would get the identical registered voice back.
        """
        if self.preset is None or self.variant_index > 0:
            return None
        return self.preset.backend_voice_id

    def to_json(self) -> Dict[str, object]:
        return {
            "mode": self.mode.value,
            "preset": self.preset.voice_id if self.preset else None,
            "preset_label": self.preset.label if self.preset else None,
            "voice_class": self.preset.voice_class.value if self.preset else None,
            "downgraded": self.downgraded,
            "reason": self.reason,
            "native_language": self.native_language,
            "variant_index": self.variant_index,
            "pitch_shift_semitones": round(self.pitch_shift_semitones, 2),
            "notice": self.notice,
        }


def shift_pitch(audio: AudioChunk, semitones: float) -> AudioChunk:
    """Shift a REFERENCE clip's pitch by resampling and relabelling the rate.

    Deliberately the crude method: resample the samples and then declare them
    to be at the original rate, which moves pitch and duration together. On
    ordinary audio that is a chipmunk artefact; on a cloning *reference* it is
    exactly right, because the backend reads timbre and pitch from the clip and
    does not care how long it is. The result is a base voice that sounds like a
    different person, deterministically, with no vocoder in the path.
    """
    if not semitones:
        return audio
    from fractions import Fraction

    from sglang.srt.translator.audio import resample

    # The intermediate rate has to make an EXACT small rational ratio with the
    # source rate, or the resampler refuses (it will not approximate, because
    # approximating is a silent pitch error). Picking the fraction first is not
    # enough: `rate * p/q` is only an integer when q divides the rate, and
    # truncating it reintroduces exactly the inexact ratio the resampler
    # rejects. So search denominators that DIVIDE the sample rate.
    target = 2.0 ** (-semitones / 12.0)
    rate = audio.sample_rate
    best: Optional[Fraction] = None
    for q in range(2, 65):
        if rate % q:
            continue
        p = max(1, int(round(target * q)))
        candidate = Fraction(p, q)
        if best is None or abs(float(candidate) - target) < abs(float(best) - target):
            best = candidate
    if best is None:
        return audio
    intermediate = rate * best.numerator // best.denominator
    if intermediate < 4000 or intermediate == rate:
        return audio
    shifted = resample(audio, intermediate)
    return AudioChunk(shifted.samples, audio.sample_rate)


class VoiceClassifier(Protocol):
    """Infers a broad voice class from a speaker's audio."""

    name: str

    def classify(self, audio: AudioChunk) -> VoiceClass: ...


def estimate_median_f0(
    audio: AudioChunk,
    minimum_hz: float = 60.0,
    maximum_hz: float = 450.0,
    frame_ms: int = 40,
    voiced_threshold: float = 0.30,
) -> Optional[float]:
    """Median fundamental frequency over voiced frames, or None.

    Autocorrelation with a normalised peak test. Not a research-grade pitch
    tracker -- it is a class discriminator, and the classes are wide. Frames
    whose normalised autocorrelation peak is below ``voiced_threshold`` are
    unvoiced (fricatives, silence, noise) and are excluded, which is what
    keeps a noisy street segment from producing a confident wrong answer.
    """
    rate = audio.sample_rate
    n = int(rate * frame_ms / 1000)
    if n <= 0 or len(audio.samples) < n:
        return None
    min_lag = max(int(rate / maximum_hz), 2)
    max_lag = min(int(rate / minimum_hz), n - 1)
    if max_lag <= min_lag:
        return None

    estimates: List[float] = []
    for start in range(0, len(audio.samples) - n + 1, n):
        frame = audio.samples[start : start + n].astype(np.float64)
        frame = frame - frame.mean()
        energy = float(np.dot(frame, frame))
        if energy <= 1e-9:
            continue
        correlation = np.correlate(frame, frame, mode="full")[n - 1 :]
        window = correlation[min_lag : max_lag + 1]
        if not len(window):
            continue
        peak_index = int(np.argmax(window))
        peak = float(window[peak_index]) / energy
        if peak < voiced_threshold:
            continue
        estimates.append(rate / float(min_lag + peak_index))
    if not estimates:
        return None
    return float(np.median(estimates))


class F0VoiceClassifier:
    """Heuristic man/woman/child classifier over median F0.

    Thresholds follow the conventional speech-science ranges (adult male
    ~85-155 Hz, adult female ~165-255 Hz, child above that). The band between
    the adult ranges is genuinely ambiguous and is resolved at its midpoint
    rather than pretended away; a session that cares can override the class
    for that speaker.
    """

    name = "f0-heuristic"

    def __init__(
        self,
        male_ceiling_hz: float = 155.0,
        female_floor_hz: float = 165.0,
        child_floor_hz: float = 260.0,
    ) -> None:
        self._male_ceiling = male_ceiling_hz
        self._female_floor = female_floor_hz
        self._child_floor = child_floor_hz

    def classify(self, audio: AudioChunk) -> VoiceClass:
        f0 = estimate_median_f0(audio)
        if f0 is None:
            return VoiceClass.UNKNOWN
        if f0 >= self._child_floor:
            return VoiceClass.CHILD
        if f0 <= self._male_ceiling:
            return VoiceClass.MAN
        if f0 >= self._female_floor:
            return VoiceClass.WOMAN
        midpoint = (self._male_ceiling + self._female_floor) / 2.0
        return VoiceClass.MAN if f0 < midpoint else VoiceClass.WOMAN


class VoicePool:
    """The preset voices, and the sticky speaker -> preset mapping.

    Stickiness is a hard requirement, not a nicety: re-shuffling a speaker's
    voice mid-conversation destroys the only cue the listener has for who is
    talking. The mapping therefore lives on the pool for the whole session and
    survives a reconnect, because the session (and with it this pool) outlives
    the WebSocket.
    """

    #: Recommended pool shape (user decision 2026-08-03). The realistic worst
    #: case is 6-8 people in one conversation, usually 2-4. Eight participants
    #: can plausibly skew hard to one adult class, so six voices per adult
    #: class covers the skew without a gallery nobody will ever hear; children
    #: rarely exceed three in such a group, so three each. Total 18.
    RECOMMENDED_PER_CLASS: Dict[str, int] = {
        VoiceClass.MAN.value: 6,
        VoiceClass.WOMAN.value: 6,
        VoiceClass.BOY.value: 3,
        VoiceClass.GIRL.value: 3,
    }
    #: A class with fewer than this many distinct presets cannot keep speakers
    #: apart in a multi-party conversation, and the pool says so at load time
    #: rather than at the third speaker.
    MIN_PER_CLASS = 3
    #: Semitone step between successive shared-voice variants. Large enough to
    #: be audibly a different person, small enough to stay natural.
    VARIANT_STEP_SEMITONES = 1.5

    def __init__(
        self,
        presets: Sequence[PresetVoice],
        classifier: Optional[VoiceClassifier] = None,
    ) -> None:
        if not presets:
            raise VoicePoolError("a preset pool needs at least one voice")
        seen: Dict[str, PresetVoice] = {}
        for preset in presets:
            if preset.voice_id in seen:
                raise VoicePoolError(f"duplicate preset id {preset.voice_id!r}")
            seen[preset.voice_id] = preset
        self._presets: Tuple[PresetVoice, ...] = tuple(presets)
        self._classifier = classifier or F0VoiceClassifier()
        self._assigned: Dict[str, Tuple[str, int]] = {}
        self._overrides: Dict[str, VoiceClass] = {}

    # -- introspection ------------------------------------------------------

    def __len__(self) -> int:
        return len(self._presets)

    def presets(self) -> Tuple[PresetVoice, ...]:
        return self._presets

    def languages(self) -> Tuple[str, ...]:
        codes = set()
        for preset in self._presets:
            codes.update(preset.languages())
        return tuple(sorted(codes))

    def counts_by_class(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for preset in self._presets:
            counts[preset.voice_class.value] = (
                counts.get(preset.voice_class.value, 0) + 1
            )
        return counts

    def thin_classes(self) -> Tuple[str, ...]:
        """Classes with too few voices to keep speakers apart."""
        counts = self.counts_by_class()
        child = sum(
            counts.get(c.value, 0)
            for c in (VoiceClass.CHILD, VoiceClass.BOY, VoiceClass.GIRL)
        )
        thin = [
            name
            for name in (VoiceClass.MAN.value, VoiceClass.WOMAN.value)
            if counts.get(name, 0) < self.MIN_PER_CLASS
        ]
        if 0 < child < self.MIN_PER_CLASS:
            thin.append("child")
        return tuple(thin)

    def to_json(self) -> Dict[str, object]:
        return {
            "voices": [
                {
                    "voice_id": p.voice_id,
                    "label": p.label,
                    "voice_class": p.voice_class.value,
                    "languages": list(p.languages()),
                }
                for p in self._presets
            ],
            "counts_by_class": self.counts_by_class(),
            "recommended_per_class": dict(self.RECOMMENDED_PER_CLASS),
            "thin_classes": list(self.thin_classes()),
            "assigned": {
                speaker: {"voice_id": voice_id, "variant": variant}
                for speaker, (voice_id, variant) in self._assigned.items()
            },
            "classifier": self._classifier.name,
        }

    # -- assignment ---------------------------------------------------------

    def override_class(self, speaker_id: str, voice_class: VoiceClass) -> None:
        """Pin a speaker's class, discarding any assignment made under the old one."""
        self._overrides[speaker_id] = voice_class
        self._assigned.pop(speaker_id, None)

    def assigned_preset(self, speaker_id: str) -> Optional[PresetVoice]:
        slot = self._assigned.get(speaker_id)
        if slot is None:
            return None
        voice_id, _variant = slot
        return next((p for p in self._presets if p.voice_id == voice_id), None)

    def assigned_variant(self, speaker_id: str) -> int:
        slot = self._assigned.get(speaker_id)
        return 0 if slot is None else slot[1]

    def classify(self, speaker_id: str, audio: Optional[AudioChunk]) -> VoiceClass:
        if speaker_id in self._overrides:
            return self._overrides[speaker_id]
        if audio is None or audio.duration_s <= 0.0:
            return VoiceClass.UNKNOWN
        return self._classifier.classify(audio)

    def assign(
        self, speaker_id: str, audio: Optional[AudioChunk] = None
    ) -> Tuple[PresetVoice, int]:
        """Sticky, class-matched ``(preset, variant)`` for this speaker.

        Once assigned, the same slot comes back for the rest of the session
        regardless of what later audio suggests -- a voice that changes because
        one noisy segment shifted the F0 estimate would be worse than a
        slightly mismatched one.

        Allocation order, and the reasoning for it:

        1. an unused voice of the speaker's own class -- the wanted case;
        2. an unused voice of any class -- distinctness beats class match,
           because a listener who cannot tell two speakers apart has lost more
           than one whose preset is the wrong gender;
        3. a SHARED base voice of the speaker's class at the next variant
           index, which applies a deterministic pitch offset so the two
           speakers still do not sound identical, and raises a notice.

        Step 3 is the "ninth man walks in" path. It never crashes and it never
        hands out an identical voice silently.
        """
        existing = self.assigned_preset(speaker_id)
        if existing is not None:
            return existing, self.assigned_variant(speaker_id)

        voice_class = self.classify(speaker_id, audio)
        taken = {voice_id for voice_id, _variant in self._assigned.values()}

        own_class = [
            p
            for p in self._presets
            if p.voice_class.matches(voice_class) and p.voice_id not in taken
        ]
        if own_class:
            chosen, variant = sorted(own_class, key=lambda p: p.voice_id)[0], 0
        else:
            any_class = [p for p in self._presets if p.voice_id not in taken]
            if any_class:
                chosen, variant = sorted(any_class, key=lambda p: p.voice_id)[0], 0
            else:
                pool = [
                    p for p in self._presets if p.voice_class.matches(voice_class)
                ] or list(self._presets)
                usage: Dict[str, int] = {}
                for voice_id, _variant in self._assigned.values():
                    usage[voice_id] = usage.get(voice_id, 0) + 1
                chosen = sorted(
                    pool, key=lambda p: (usage.get(p.voice_id, 0), p.voice_id)
                )[0]
                variant = max(
                    (
                        v
                        for vid, v in self._assigned.values()
                        if vid == chosen.voice_id
                    ),
                    default=0,
                ) + 1
                logger.warning(
                    "voice pool exhausted for class %s: speaker %s shares base "
                    "voice %s at variant %d (%+.1f semitones)",
                    voice_class.value,
                    speaker_id,
                    chosen.voice_id,
                    variant,
                    self.variant_shift(variant),
                )
        self._assigned[speaker_id] = (chosen.voice_id, variant)
        return chosen, variant

    @classmethod
    def variant_shift(cls, variant_index: int) -> float:
        """Deterministic semitone offset for a shared base voice.

        Alternates up and down so successive sharers move apart from each
        other as well as from the original: +1.5, -1.5, +3.0, -3.0, ...
        """
        if variant_index <= 0:
            return 0.0
        step = (variant_index + 1) // 2
        sign = 1.0 if variant_index % 2 == 1 else -1.0
        return sign * step * cls.VARIANT_STEP_SEMITONES

    def choose(
        self,
        speaker_id: str,
        language: str,
        audio: Optional[AudioChunk] = None,
        downgraded: bool = False,
        reason: str = "",
    ) -> VoiceAssignment:
        """Full preset-mode outcome for one turn."""
        preset, variant = self.assign(speaker_id, audio)
        reference, text = preset.reference_for(language)
        shift = self.variant_shift(variant)
        notice = ""
        if variant > 0:
            if reference is not None:
                reference = shift_pitch(reference, shift)
            notice = (
                f"voice pool exhausted for class {preset.voice_class.value}; "
                f"{speaker_id} shares base voice {preset.label} "
                f"({shift:+.1f} semitones)"
            )
        return VoiceAssignment(
            mode=VoiceMode.PRESET,
            reference=reference,
            reference_text=text,
            preset=preset,
            downgraded=downgraded,
            reason=reason,
            native_language=preset.speaks_natively(language),
            variant_index=variant,
            pitch_shift_semitones=shift,
            notice=notice,
        )

    # -- construction -------------------------------------------------------

    @classmethod
    def from_directory(
        cls,
        root: Path,
        classifier: Optional[VoiceClassifier] = None,
        sample_rate: Optional[int] = None,
    ) -> "VoicePool":
        """Load presets from ``<root>/<class>/<voice_id>.<language>.wav``.

        A flat, greppable convention rather than a manifest file: the class and
        the language are in the path, so adding a voice is dropping in a wav
        and nothing can silently disagree with a sidecar. An optional
        ``<voice_id>.<language>.txt`` next to a clip supplies its transcript
        for backends that want one.
        """
        import soundfile

        root = Path(root)
        if not root.is_dir():
            raise VoicePoolError(f"preset pool directory {root} does not exist")

        collected: Dict[str, Dict[str, object]] = {}
        for wav in sorted(root.glob("*/*.wav")):
            class_name = wav.parent.name.lower()
            try:
                voice_class = VoiceClass(class_name)
            except ValueError:
                logger.warning(
                    "ignoring preset directory %r: not a voice class", class_name
                )
                continue
            parts = wav.stem.split(".")
            if len(parts) != 2:
                logger.warning(
                    "ignoring %s: expected <voice_id>.<language>.wav", wav.name
                )
                continue
            voice_id, language = parts[0], parts[1].lower()
            samples, rate = soundfile.read(str(wav), dtype="float32", always_2d=True)
            mono = samples.mean(axis=1).astype(np.float32)
            audio = AudioChunk(mono, int(rate))
            if sample_rate is not None and audio.sample_rate != sample_rate:
                from sglang.srt.translator.audio import resample

                audio = resample(audio, sample_rate)
            entry = collected.setdefault(
                voice_id,
                {"class": voice_class, "refs": {}, "texts": {}},
            )
            entry["refs"][language] = audio
            transcript = wav.with_suffix(".txt")
            if transcript.exists():
                entry["texts"][language] = transcript.read_text(
                    encoding="utf-8"
                ).strip()

        presets = [
            PresetVoice(
                voice_id=voice_id,
                label=voice_id.replace("_", " "),
                voice_class=entry["class"],
                references=entry["refs"],
                reference_texts=entry["texts"],
            )
            for voice_id, entry in sorted(collected.items())
        ]
        if not presets:
            raise VoicePoolError(
                f"no presets found under {root}; expected "
                "<class>/<voice_id>.<language>.wav"
            )
        pool = cls(presets, classifier=classifier)
        thin = pool.thin_classes()
        if thin:
            logger.warning(
                "preset pool is thin in %s (fewer than %d voices); speakers of "
                "those classes may be hard to tell apart",
                ", ".join(thin),
                cls.MIN_PER_CLASS,
            )
        return pool


def synthetic_pool(
    languages: Iterable[str],
    per_class: Optional[Dict[VoiceClass, int]] = None,
    sample_rate: int = 24000,
    seconds: float = 4.0,
) -> VoicePool:
    """A pool of synthetic tone voices, shaped like the recommended real one.

    For the hermetic suite. Real deployments load recorded clips; this exists
    so the assignment logic, the stickiness, the class match and the
    exhaustion path can be tested without shipping audio into the repository.
    Defaults to the 6/6/3/3 shape so a test that fills the pool is filling a
    realistically sized one.
    """
    import math

    codes = [str(c) for c in languages]
    shape = per_class or {
        VoiceClass.MAN: VoicePool.RECOMMENDED_PER_CLASS[VoiceClass.MAN.value],
        VoiceClass.WOMAN: VoicePool.RECOMMENDED_PER_CLASS[VoiceClass.WOMAN.value],
        VoiceClass.BOY: VoicePool.RECOMMENDED_PER_CLASS[VoiceClass.BOY.value],
        VoiceClass.GIRL: VoicePool.RECOMMENDED_PER_CLASS[VoiceClass.GIRL.value],
    }
    bases = {
        VoiceClass.MAN: 110.0,
        VoiceClass.WOMAN: 200.0,
        VoiceClass.CHILD: 300.0,
        VoiceClass.BOY: 300.0,
        VoiceClass.GIRL: 320.0,
    }
    presets: List[PresetVoice] = []
    for voice_class, count in shape.items():
        base = bases.get(voice_class, 180.0)
        for index in range(count):
            frequency = base * (1.0 + 0.06 * index)
            t = np.arange(int(seconds * sample_rate), dtype=np.float32) / sample_rate
            wave = (0.3 * np.sin(2.0 * math.pi * frequency * t)).astype(np.float32)
            presets.append(
                PresetVoice(
                    voice_id=f"{voice_class.value}-{index + 1}",
                    label=f"{voice_class.value} {index + 1}",
                    voice_class=voice_class,
                    references={c: AudioChunk(wave, sample_rate) for c in codes},
                    reference_texts={c: f"preset {voice_class.value}" for c in codes},
                )
            )
    return VoicePool(presets)
