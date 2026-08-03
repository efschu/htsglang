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
    "LanguagePair",
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
        """Refuse a conversation this deployment cannot actually run.

        ALL-OR-NOTHING, and that is right for the two places that ask it: the
        server advertising a DEFAULT pair (a default the deployment cannot run
        is exactly the failure the runtime derivation exists to prevent), and
        anywhere a caller wants the strict answer. A live session uses
        :meth:`direction_report` instead -- see its docstring for why a
        session must not be refused for one bad direction.
        """
        for src in sorted(self.participants):
            for tgt in self.targets_for(src):
                matrix.require_pair(src, tgt)

    def direction_report(
        self, matrix: LanguageMatrix
    ) -> Tuple[Tuple[Tuple[str, str], ...], Dict[Tuple[str, str], str]]:
        """Split every direction into the servable ones and the rest.

        Returns ``(servable, unroutable)``, the second mapping
        ``(source, target)`` to the reason naming the stage that refuses.

        WHY THIS EXISTS. ``validate_against`` raises on the FIRST direction
        the deployment cannot serve, and a live session called it in its
        constructor -- so adding one participant language whose TTS this
        deployment does not have did not degrade that one direction, it
        refused to open the conversation at all. Everything else the user had
        picked went with it. The picker consequently could not offer the ten
        languages the deployment can hear and translate but not speak, and the
        client carried a constant (`PARTIAL_PARTICIPANTS_SUPPORTED`) waiting
        for exactly this.

        The turn path already knew how to degrade: it calls
        ``matrix.require_pair`` per target inside the loop and skips the ones
        that refuse, and ``turn.unrouted`` is the event that tells the user
        which direction is missing rather than leaving silence. Only the
        constructor was all-or-nothing.

        A conversation with NO servable direction is still refused -- that is
        not a degraded conversation, it is not a conversation.
        """
        servable = []
        unroutable: Dict[Tuple[str, str], str] = {}
        for src in sorted(self.participants):
            for tgt in self.targets_for(src):
                try:
                    matrix.require_pair(src, tgt)
                except LanguageError as exc:
                    unroutable[(canonical_code(src), canonical_code(tgt))] = str(exc)
                else:
                    servable.append((canonical_code(src), canonical_code(tgt)))
        return tuple(servable), unroutable


@dataclasses.dataclass(frozen=True)
class LanguagePair:
    """One UNORDERED pair of languages: ``a <-> b`` means both directions.

    Stored canonically with ``a <= b`` so that ``(de, es)`` and ``(es, de)``
    are the same object and cannot both be added. The ordering is a storage
    detail and carries no direction; :meth:`partner` is how the pipeline asks
    the question it actually has.
    """

    a: str
    b: str

    @classmethod
    def of(cls, first: str, second: str) -> "LanguagePair":
        one, two = canonical_code(first), canonical_code(second)
        if one == two:
            raise LanguageError(
                f"pair {one}<->{two} is a no-op; a language pair needs two "
                "different languages"
            )
        return cls(*sorted((one, two)))

    def partner(self, language: str) -> Optional[str]:
        """The other side, or ``None`` when this pair does not involve it."""
        code = canonical_code(language)
        if code == self.a:
            return self.b
        if code == self.b:
            return self.a
        return None

    def to_json(self) -> Dict[str, str]:
        return {"a": self.a, "b": self.b}


class RoutingTable:
    """The manual routing mode: a set of unordered language pairs.

    **The semantics are the user's, and they invert two earlier decisions.**

    *A pair is bidirectional.* Adding ``de <-> es`` means a German utterance is
    rendered in Spanish AND a Spanish one in German. Nobody configuring a
    conversation wants to state the same relationship twice, and a table that
    accepted only one direction would silently drop every reply.

    *Fan-out is the intent, not an ambiguity.* With ``{de<->es, de<->fr}`` a
    German utterance goes to Spanish AND French -- two outputs, played
    sequentially with a language tag. The earlier "one source routes to exactly
    one target, a duplicate source is refused" rule was the wrong reading: it
    made the second pair unaddable and the three-language case unexpressible.
    A repeated pair is therefore DEDUPLICATED rather than refused -- adding
    ``es <-> de`` to a table that already has ``de <-> es`` is a no-op, not an
    error, because the user asked for a relationship that already holds.

    What survives from the first version, because it was right: the ASR still
    classifies every utterance's source language (the direction is observed,
    never configured), and a language with no pair is passed through
    untranslated and TAGGED rather than dropped.

    Capability refusal moved to the pair level for the same reason: with
    unordered pairs, "es is unusable" is not a useful thing to say when the
    real fact is "de <-> es cannot run because the TTS cannot speak es". See
    :meth:`capability_report`.
    """

    def __init__(self, pairs: Iterable[LanguagePair] = ()) -> None:
        # Insertion-ordered, so the UI can render the rows in the order the
        # user typed them; membership is by canonical pair.
        self._pairs: Dict[Tuple[str, str], LanguagePair] = {}
        for pair in pairs:
            self.add_pair(pair.a, pair.b)

    def __len__(self) -> int:
        return len(self._pairs)

    def __bool__(self) -> bool:
        # Explicit: an empty table means "fall back to auto", and relying on
        # __len__ alone would make that read as a bug at the call site.
        return bool(self._pairs)

    def __iter__(self):
        return iter(self._pairs.values())

    def add_pair(self, first: str, second: str) -> LanguagePair:
        """Add ``first <-> second``. Adding it twice is a no-op, not an error."""
        pair = LanguagePair.of(first, second)
        key = (pair.a, pair.b)
        if key in self._pairs:
            return self._pairs[key]
        self._pairs[key] = pair
        return pair

    def remove_pair(self, first: str, second: str) -> bool:
        pair = LanguagePair.of(first, second)
        return self._pairs.pop((pair.a, pair.b), None) is not None

    def replace_all(self, pairs: Iterable[Tuple[str, str]]) -> None:
        """Set the whole table at once, atomically.

        Validation happens on a scratch table first, so a rejected update
        leaves the live one untouched rather than half-applied.
        """
        scratch = RoutingTable()
        for first, second in pairs:
            scratch.add_pair(first, second)
        self._pairs = dict(scratch._pairs)

    def partners_for(self, language: str) -> Tuple[str, ...]:
        """Every language ``language`` is paired with, sorted.

        This is the fan-out: two entries means the utterance is rendered twice.
        An empty tuple means the table has no pair for this language, which the
        session reports as an unrouted turn rather than as silence.
        """
        code = canonical_code(language)
        partners = {
            partner
            for pair in self._pairs.values()
            if (partner := pair.partner(code)) is not None
        }
        return tuple(sorted(partners))

    def languages(self) -> Tuple[str, ...]:
        """Every language named by any pair, sorted.

        This is the ASR whitelist. Constraining language identification to the
        set the table can actually route is strictly better than letting it
        choose from ninety-nine: a misdetection into an unrouted language costs
        the whole turn, and the languages in the table are by construction the
        only ones the user expects to be spoken.
        """
        codes = set()
        for pair in self._pairs.values():
            codes.add(pair.a)
            codes.add(pair.b)
        return tuple(sorted(codes))

    def capability_report(
        self, matrix: "LanguageMatrix"
    ) -> Tuple[Dict[str, object], ...]:
        """Per-pair usability, with the refusing stage named.

        Named rather than filtered: a pair the deployment cannot run is shown
        greyed out with its reason, so the user learns that the TTS checkpoint
        does not speak the language instead of watching a row vanish. Both
        directions are reported separately -- an asymmetric backend set can
        make ``a->b`` routable while ``b->a`` is not, and collapsing that into
        one flag would hide exactly the case worth seeing.
        """
        report = []
        for pair in self._pairs.values():
            directions = []
            for source, target in ((pair.a, pair.b), (pair.b, pair.a)):
                try:
                    matrix.require_pair(source, target)
                except LanguageError as exc:
                    directions.append(
                        {
                            "source": source,
                            "target": target,
                            "usable": False,
                            "reason": str(exc),
                        }
                    )
                else:
                    directions.append(
                        {
                            "source": source,
                            "target": target,
                            "usable": True,
                            "reason": None,
                        }
                    )
            report.append(
                {
                    **pair.to_json(),
                    "usable": all(d["usable"] for d in directions),
                    "directions": tuple(directions),
                }
            )
        return tuple(report)

    def validate_against(self, matrix: "LanguageMatrix") -> None:
        """Hard check: refuse a table this deployment cannot run at all.

        Used where a silent partial table would be worse than a refusal (the
        HTTP endpoint that accepts a new table). The UI path uses
        :meth:`capability_report` instead, which names rather than raises.
        """
        for pair in self._pairs.values():
            matrix.require_pair(pair.a, pair.b)
            matrix.require_pair(pair.b, pair.a)

    def to_json(self) -> List[Dict[str, str]]:
        return [pair.to_json() for pair in self._pairs.values()]
