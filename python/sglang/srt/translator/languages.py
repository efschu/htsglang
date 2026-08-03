# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""The language set, derived rather than declared (#466 requirement 5).

The system's supported-language list is never written down as a constant. It
is the **intersection** of what the three stages can actually do:

    speakable(system) = ASR.transcribes  x  MT.translates  x  TTS.synthesizes

A language that the recognizer hears but the synthesizer cannot speak is not
a supported language -- a turn in it would end in silence. A language the
synthesizer speaks but the recognizer cannot hear is not supported either --
nothing would ever be routed to it. Intersecting is the only honest answer,
and it is recomputed from the live backends rather than cached from a table,
so swapping a TTS checkpoint changes the advertised set without a code edit.

Two asymmetries are real and are kept:

* ASR is a **source** capability, TTS is a **target** capability. Some
  backends are asymmetric on purpose (Whisper hears 99 languages; a given TTS
  checkpoint speaks 10). :class:`LanguageMatrix` therefore reports
  ``sources``, ``targets`` and the ``pairs`` that are actually routable,
  not one flat list.
* MT is usually the widest stage and rarely the binding constraint, but it
  still participates: an MT backend that declares a set constrains both ends.

Codes are ISO 639-1 two-letter lowercase wherever one exists, normalised on
the way in (``de-DE`` -> ``de``, ``DE`` -> ``de``, ``deu``/``ger`` -> ``de``).
Backends that speak a regional variant declare the base code; region is a
voice/style choice, not a different language, and collapsing it here is what
lets an ASR that reports ``es-419`` meet a TTS that declares ``es``.
"""

from __future__ import annotations

import dataclasses
from typing import (
    Dict,
    FrozenSet,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

__all__ = [
    "LanguageError",
    "LanguageMatrix",
    "canonical_code",
    "canonical_set",
    "DirectionRule",
    "RoutingTable",
    "display_name",
]


class LanguageError(ValueError):
    """A language code or routing request that cannot be honoured."""


# ISO 639-2/T and 639-2/B forms we accept on input, mapped to 639-1. Only the
# codes worth accepting from a real backend are listed; an unknown three-letter
# code is passed through unchanged rather than guessed at, so a backend using an
# exotic tag still intersects correctly with another backend using the same tag.
_THREE_TO_TWO: Mapping[str, str] = {
    "deu": "de",
    "ger": "de",
    "spa": "es",
    "eng": "en",
    "fra": "fr",
    "fre": "fr",
    "ita": "it",
    "por": "pt",
    "nld": "nl",
    "dut": "nl",
    "pol": "pl",
    "rus": "ru",
    "ces": "cs",
    "cze": "cs",
    "ell": "el",
    "gre": "el",
    "jpn": "ja",
    "kor": "ko",
    "zho": "zh",
    "chi": "zh",
    "cmn": "zh",
    "ara": "ar",
    "tur": "tr",
    "hin": "hi",
    "swe": "sv",
    "dan": "da",
    "nor": "no",
    "nob": "no",
    "fin": "fi",
    "ukr": "uk",
    "ron": "ro",
    "rum": "ro",
    "hun": "hu",
    "vie": "vi",
    "tha": "th",
    "ind": "id",
    "heb": "he",
    "cat": "ca",
    "eus": "eu",
    "glg": "gl",
}

# Endonym + English name, for a client that wants to render a picker. This is
# presentation only -- nothing routes on it, and a code missing from the table
# still works, it just displays as its own code.
_NAMES: Mapping[str, Tuple[str, str]] = {
    "ar": ("العربية", "Arabic"),
    "ca": ("Català", "Catalan"),
    "cs": ("Čeština", "Czech"),
    "da": ("Dansk", "Danish"),
    "de": ("Deutsch", "German"),
    "el": ("Ελληνικά", "Greek"),
    "en": ("English", "English"),
    "es": ("Español", "Spanish"),
    "eu": ("Euskara", "Basque"),
    "fi": ("Suomi", "Finnish"),
    "fr": ("Français", "French"),
    "gl": ("Galego", "Galician"),
    "he": ("עברית", "Hebrew"),
    "hi": ("हिन्दी", "Hindi"),
    "hu": ("Magyar", "Hungarian"),
    "id": ("Bahasa Indonesia", "Indonesian"),
    "it": ("Italiano", "Italian"),
    "ja": ("日本語", "Japanese"),
    "ko": ("한국어", "Korean"),
    "nl": ("Nederlands", "Dutch"),
    "no": ("Norsk", "Norwegian"),
    "pl": ("Polski", "Polish"),
    "pt": ("Português", "Portuguese"),
    "ro": ("Română", "Romanian"),
    "ru": ("Русский", "Russian"),
    "sv": ("Svenska", "Swedish"),
    "th": ("ไทย", "Thai"),
    "tr": ("Türkçe", "Turkish"),
    "uk": ("Українська", "Ukrainian"),
    "vi": ("Tiếng Việt", "Vietnamese"),
    "zh": ("中文", "Chinese"),
}


def canonical_code(code: str) -> str:
    """Normalise one language tag to the code the matrix intersects on.

    ``de-DE``, ``DE``, ``deu``, ``de_DE`` and ``de`` all collapse to ``de``.
    An unknown tag is lowercased and stripped of its region, then returned as
    is -- two backends that agree on an exotic tag still meet.
    """
    if not isinstance(code, str):
        raise LanguageError(f"language code must be a string, got {type(code).__name__}")
    text = code.strip().lower().replace("_", "-")
    if not text:
        raise LanguageError("language code must not be empty")
    base = text.split("-", 1)[0]
    if not base.isalpha():
        raise LanguageError(f"language code {code!r} is not alphabetic")
    return _THREE_TO_TWO.get(base, base)


def canonical_set(codes: Iterable[str]) -> FrozenSet[str]:
    """Normalise a whole capability set, dropping duplicates after collapsing."""
    return frozenset(canonical_code(c) for c in codes)


def display_name(code: str) -> Dict[str, str]:
    """Endonym/English label for a code, falling back to the code itself."""
    canon = canonical_code(code)
    endonym, english = _NAMES.get(canon, (canon, canon))
    return {"code": canon, "endonym": endonym, "english": english}


@dataclasses.dataclass(frozen=True)
class LanguageMatrix:
    """What this deployment can actually hear, translate and speak.

    Built by :meth:`from_backends` at runtime from the live backend objects.
    Nothing here is configured; a mismatch between stages shows up as a
    smaller intersection and is *reported*, not silently papered over.
    """

    #: What the recognizer can transcribe (already canonicalised).
    asr: FrozenSet[str]
    #: What the translator can produce. Empty-as-unconstrained is NOT allowed:
    #: an MT backend that does not know its own set must say so by declaring
    #: ``unconstrained_mt=True`` instead, which is an explicit claim.
    mt: FrozenSet[str]
    #: What the synthesizer can speak.
    tts: FrozenSet[str]
    #: True when the MT stage claims universal coverage (a large multilingual
    #: LLM that will attempt any pair). Then MT drops out of the intersection
    #: and the honest bound is ASR x TTS.
    unconstrained_mt: bool = False

    @classmethod
    def from_backends(
        cls,
        asr_languages: Iterable[str],
        tts_languages: Iterable[str],
        mt_languages: Optional[Iterable[str]] = None,
    ) -> "LanguageMatrix":
        """Derive the matrix. ``mt_languages=None`` means unconstrained MT."""
        return cls(
            asr=canonical_set(asr_languages),
            mt=frozenset() if mt_languages is None else canonical_set(mt_languages),
            tts=canonical_set(tts_languages),
            unconstrained_mt=mt_languages is None,
        )

    @property
    def sources(self) -> FrozenSet[str]:
        """Languages a participant may SPEAK and be understood in."""
        if self.unconstrained_mt:
            return self.asr
        return self.asr & self.mt

    @property
    def targets(self) -> FrozenSet[str]:
        """Languages a translated turn may be RENDERED into."""
        if self.unconstrained_mt:
            return self.tts
        return self.tts & self.mt

    @property
    def bidirectional(self) -> FrozenSet[str]:
        """Languages usable at both ends -- the set a conversation picks from.

        A conversation needs every participant language to be both heard and
        spoken, so this, not :attr:`sources`, is what a client should offer.
        """
        return self.sources & self.targets

    def pairs(self) -> Tuple[Tuple[str, str], ...]:
        """Every routable ``(source, target)`` with ``source != target``."""
        return tuple(
            sorted(
                (s, t)
                for s in self.sources
                for t in self.targets
                if s != t
            )
        )

    def supports_pair(self, source: str, target: str) -> bool:
        src, tgt = canonical_code(source), canonical_code(target)
        return src != tgt and src in self.sources and tgt in self.targets

    def require_pair(self, source: str, target: str) -> Tuple[str, str]:
        """Canonicalise and validate a pair, naming the stage that refuses.

        The error names which stage is missing the language, because "es is
        not supported" is unactionable while "the TTS backend does not speak
        es" tells the operator which checkpoint to swap.
        """
        src, tgt = canonical_code(source), canonical_code(target)
        if src == tgt:
            raise LanguageError(
                f"source and target are both {src!r}; a translation direction "
                "needs two different languages"
            )
        missing = []
        if src not in self.asr:
            missing.append(f"ASR cannot transcribe {src!r}")
        if tgt not in self.tts:
            missing.append(f"TTS cannot speak {tgt!r}")
        if not self.unconstrained_mt:
            if src not in self.mt:
                missing.append(f"MT does not accept source {src!r}")
            if tgt not in self.mt:
                missing.append(f"MT does not produce target {tgt!r}")
        if missing:
            raise LanguageError(
                f"direction {src}->{tgt} is not supported by this deployment: "
                + "; ".join(missing)
            )
        return src, tgt

    def to_json(self) -> Dict[str, object]:
        """The payload behind ``GET /api/translator/languages``.

        Per-stage sets are exposed alongside the intersection on purpose: when
        a language is missing, the client (and the user) can see which stage
        dropped it instead of only learning that it is gone.
        """
        return {
            "bidirectional": [display_name(c) for c in sorted(self.bidirectional)],
            "sources": sorted(self.sources),
            "targets": sorted(self.targets),
            "stages": {
                "asr": sorted(self.asr),
                "mt": None if self.unconstrained_mt else sorted(self.mt),
                "tts": sorted(self.tts),
            },
            "unconstrained_mt": self.unconstrained_mt,
            "pair_count": len(self.pairs()),
        }


@dataclasses.dataclass(frozen=True)
class ConversationLanguages:
    """The languages present in one conversation, and how a turn is routed.

    This is the whole of the "no hardcoded language pair" contract. The
    conversation carries a *set* of participant languages, not a source and a
    target; the detected language of a finished utterance selects the target
    by elimination. Two participants is the common case and reduces to "the
    other one", but three or more is expressible: a turn detected as ``de``
    in a ``{de, es, fr}`` conversation fans out to both ``es`` and ``fr``.

    ``explicit_routes`` overrides elimination for a source that should go
    somewhere specific (or nowhere).
    """

    #: Participant languages, canonicalised, at least two.
    participants: FrozenSet[str]
    #: Optional per-source override: source -> tuple of targets.
    explicit_routes: Mapping[str, Tuple[str, ...]] = dataclasses.field(
        default_factory=dict
    )

    @classmethod
    def of(
        cls,
        participants: Sequence[str],
        explicit_routes: Optional[Mapping[str, Sequence[str]]] = None,
    ) -> "ConversationLanguages":
        canon = canonical_set(participants)
        if len(canon) < 2:
            raise LanguageError(
                "a conversation needs at least two distinct participant "
                f"languages, got {sorted(canon)}"
            )
        routes = {
            canonical_code(src): tuple(canonical_set(tgts))
            for src, tgts in (explicit_routes or {}).items()
        }
        for src, tgts in routes.items():
            if src in tgts:
                raise LanguageError(
                    f"explicit route {src!r} lists itself as a target"
                )
        return cls(participants=canon, explicit_routes=routes)

    def targets_for(self, detected: str) -> Tuple[str, ...]:
        """Targets for an utterance detected as ``detected``.

        An unknown detected language is not an error here: a conversation may
        legitimately pick up a bystander. It routes to every participant
        language, which is the useful behaviour (the user hears it in theirs)
        and keeps the caller from having to special-case it.
        """
        src = canonical_code(detected)
        if src in self.explicit_routes:
            return self.explicit_routes[src]
        return tuple(sorted(self.participants - {src}))

    def validate_against(self, matrix: LanguageMatrix) -> None:
        """Refuse a conversation this deployment cannot actually run."""
        for src in sorted(self.participants):
            for tgt in self.targets_for(src):
                matrix.require_pair(src, tgt)


@dataclasses.dataclass(frozen=True)
class DirectionRule:
    """One explicit source -> target routing rule."""

    source: str
    target: str

    def to_json(self) -> Dict[str, str]:
        return {"source": self.source, "target": self.target}


class RoutingTable:
    """The manual routing mode: several source -> target rules at once.

    The user's point, and the reason this is not just "pick a pair": with a
    table in place the system no longer has to GUESS a direction. The ASR's
    language identification still classifies the source of every utterance --
    that is needed and cheap -- but the target then comes deterministically
    from the table instead of by elimination over the participant set.

    One source maps to exactly one target. A duplicate source is refused
    rather than resolved, because two rules for one source make the routing
    ambiguous and any tie-break would be a silent guess -- which is precisely
    what this mode exists to remove.

    A language with no rule is NOT an error and NOT dropped: the utterance is
    passed through untranslated and tagged, so the user sees "no rule for X"
    in the transcript rather than silence and a shrug.
    """

    def __init__(self, rules: Iterable[DirectionRule] = ()) -> None:
        self._rules: Dict[str, str] = {}
        for rule in rules:
            self.add(rule.source, rule.target)

    def __len__(self) -> int:
        return len(self._rules)

    def __bool__(self) -> bool:
        # Explicit: an empty table means "fall back to auto", and relying on
        # __len__ alone would make that read as a bug at the call site.
        return bool(self._rules)

    def add(self, source: str, target: str) -> DirectionRule:
        src, tgt = canonical_code(source), canonical_code(target)
        if src == tgt:
            raise LanguageError(
                f"rule {src}->{tgt} is a no-op; a routing rule needs two "
                "different languages"
            )
        if src in self._rules:
            raise LanguageError(
                f"a rule for source {src!r} already exists "
                f"({src}->{self._rules[src]}); one source routes to exactly "
                "one target, so change or remove it instead of adding a second"
            )
        self._rules[src] = tgt
        return DirectionRule(src, tgt)

    def remove(self, source: str) -> bool:
        return self._rules.pop(canonical_code(source), None) is not None

    def replace_all(self, rules: Iterable[Tuple[str, str]]) -> None:
        """Set the whole table at once, atomically.

        Validation happens on a scratch table first, so a rejected update
        leaves the live one untouched rather than half-applied.
        """
        scratch = RoutingTable()
        for source, target in rules:
            scratch.add(source, target)
        self._rules = dict(scratch._rules)

    def target_for(self, source: str) -> Optional[str]:
        return self._rules.get(canonical_code(source))

    def sources(self) -> Tuple[str, ...]:
        return tuple(sorted(self._rules))

    def validate_against(self, matrix: "LanguageMatrix") -> None:
        for source, target in sorted(self._rules.items()):
            matrix.require_pair(source, target)

    def to_json(self) -> List[Dict[str, str]]:
        return [
            {"source": s, "target": t} for s, t in sorted(self._rules.items())
        ]
