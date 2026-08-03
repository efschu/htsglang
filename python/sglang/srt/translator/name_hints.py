# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Who is who: name suggestions from what people actually say (§17.3).

A name in speech means three different things depending on whom it refers to,
and the same surface form carries all three:

``self``
    "ich bin Matthias Ehrenfeuchter" — the speaker names themselves. The name
    belongs to the CURRENT speaker.

``third_party``
    "darf ich vorstellen: Larisa Ehrenfeuchter" — a name is introduced and it
    is explicitly NOT the speaker. It belongs to nobody yet, so it floats.

``addressed``
    "sag hallo, Moritz" — the speaker addresses somebody else. The name
    belongs to whoever ANSWERS.

**The adjacency logic lives here, not in the model.** The LLM is asked one
question about one utterance's text: which of the three kinds, and which name.
It is never asked who was in the room, because it has no reliable access to
turn structure and would confabulate it. Resolving an ``addressed`` candidate
is a question about diarization ids and timestamps, which the session knows
exactly: the next utterance, inside a bounded window, from a DIFFERENT
speaker. No answer, or the same speaker again, and the candidate expires
without a chip ever being shown.

**Nothing is ever auto-applied.** Every resolved candidate becomes a
suggestion the user confirms or discards. A wrong name applied silently is
worse than no name, because it then travels through the whole transcript.

**The pre-filter runs first**, so most utterances never reach the LLM. It is
allowed false positives — they cost one small request — and its misses are
covered by the manual naming path, which is the primary one anyway.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "NameCandidate",
    "NameSuggestion",
    "SuggestionTracker",
    "looks_like_naming",
    "parse_candidates",
    "EXTRACTION_PROMPT",
    "KIND_SELF",
    "KIND_THIRD_PARTY",
    "KIND_ADDRESSED",
]

KIND_SELF = "self"
KIND_THIRD_PARTY = "third_party"
KIND_ADDRESSED = "addressed"
KINDS = (KIND_SELF, KIND_THIRD_PARTY, KIND_ADDRESSED)

#: Introduction and address cues. Deliberately multilingual and deliberately
#: incomplete: this is a cheap gate in front of an expensive model, not a
#: parser. A cue list that tried to be exhaustive would be a second, worse
#: implementation of the classifier it is supposed to protect.
CUES: Tuple[str, ...] = (
    # self
    "ich bin", "ich heisse", "ich heiße", "mein name",
    "soy", "me llamo", "mi nombre",
    "i am", "i'm", "my name",
    "je suis", "je m'appelle",
    "sono", "mi chiamo",
    # third party
    "das ist", "das hier ist", "darf ich vorstellen", "hier ist",
    "este es", "esta es", "les presento", "te presento",
    "this is", "may i introduce", "meet ",
    "voici", "je vous presente", "je vous présente",
    # addressed
    "sag hallo", "sag mal", "grüß", "gruess",
    "di hola", "saluda",
    "say hello", "say hi",
    "dis bonjour",
)

#: A capitalised word that is not at the start of a sentence is the other
#: cheap signal. Sentence-initial capitals are excluded because every
#: sentence has one and they would let everything through.
_MID_SENTENCE_CAPITAL = re.compile(r"(?<![.!?]\s)(?<!^)\b[A-ZÀ-ÖØ-Þ][\wÀ-ÿ'-]{2,}")

EXTRACTION_PROMPT = """You extract personal NAMES from one spoken utterance.

Answer with JSON only: {"candidates": [{"name": ..., "kind": ...}]}
Use an empty list when the utterance contains no personal name.

"kind" is exactly one of:
- "self": the speaker states their OWN name ("ich bin Matthias", "me llamo Ana")
- "third_party": a name is introduced that is NOT the speaker
  ("darf ich vorstellen: Larisa", "this is my brother Tom")
- "addressed": the speaker addresses somebody else BY name
  ("sag hallo, Moritz", "saluda, Ben")

Rules:
- place names, company names, product names and days of the week are NOT
  personal names; return an empty list for them
- do not guess a kind you cannot see evidence for; when a name appears with
  no indication of whom it refers to, omit it
- return the name as spoken, without titles
- JSON only, no explanation"""


@dataclasses.dataclass(frozen=True)
class NameCandidate:
    """One name the extractor found in one utterance."""

    name: str
    kind: str
    turn_id: str = ""
    speaker_id: str = ""
    at: float = 0.0


@dataclasses.dataclass
class NameSuggestion:
    """A candidate that now has somebody to attach to. Still needs a tap."""

    suggestion_id: str
    name: str
    kind: str
    #: Empty for a floating ``third_party`` suggestion.
    speaker_id: str
    at: float
    source_turn_id: str = ""
    resolved_turn_id: str = ""

    def to_json(self) -> Dict[str, object]:
        return {
            "suggestion_id": self.suggestion_id,
            "name": self.name,
            "kind": self.kind,
            "speaker_id": self.speaker_id,
            "at": round(self.at, 4),
            "source_turn_id": self.source_turn_id,
            "resolved_turn_id": self.resolved_turn_id,
        }


def looks_like_naming(text: str) -> bool:
    """Cheap gate: could this utterance contain somebody's name?

    False positives are fine and expected — they cost one small request. A
    false negative costs a suggestion that the manual path would have to
    supply instead, which is a mild degradation rather than a wrong answer.
    """
    if not text or not text.strip():
        return False
    lowered = text.lower()
    if any(cue in lowered for cue in CUES):
        return True
    return bool(_MID_SENTENCE_CAPITAL.search(text.strip()))


def parse_candidates(payload: str) -> List[NameCandidate]:
    """Read the extractor's JSON, refusing anything malformed rather than guessing."""
    text = (payload or "").strip()
    if not text:
        return []
    # Models like to wrap JSON in a fence even when told not to.
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return []
    try:
        data = json.loads(text[start : end + 1])
    except (ValueError, TypeError):
        logger.debug("name extraction returned unparsable JSON: %r", payload)
        return []
    out: List[NameCandidate] = []
    for entry in data.get("candidates") or ():
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "").strip()
        kind = str(entry.get("kind") or "").strip().lower()
        if not name or kind not in KINDS:
            # A kind we do not know is dropped, not mapped to a default: a
            # wrong KIND attaches a real name to the wrong person, which is
            # the failure this whole feature exists to avoid.
            continue
        out.append(NameCandidate(name=name, kind=kind))
    return out


class SuggestionTracker:
    """Turns candidates into suggestions using turn adjacency, not the model.

    ``window_s`` and ``max_turns`` bound how long an ``addressed`` candidate
    waits for its answer. Both are deliberately small: "sag hallo, Moritz"
    followed three minutes later by a stranger is not evidence of anything.
    """

    def __init__(
        self,
        window_s: float = 15.0,
        max_turns: int = 2,
        clock: Callable[[], float] = None,
    ) -> None:
        import time as _time

        self.window_s = window_s
        self.max_turns = max_turns
        self._clock = clock or _time.monotonic
        self._pending_addressed: List[Tuple[NameCandidate, int]] = []
        self._turn_count = 0
        self._next_id = 1
        #: Suggestions the user discarded, per speaker, so the same name is
        #: not offered again in the same conversation.
        self._discarded: Dict[str, set] = {}

    def _make(
        self,
        candidate: NameCandidate,
        speaker_id: str,
        resolved_turn_id: str = "",
    ) -> Optional[NameSuggestion]:
        if candidate.name in self._discarded.get(speaker_id, set()):
            return None
        suggestion = NameSuggestion(
            suggestion_id=f"sug-{self._next_id}",
            name=candidate.name,
            kind=candidate.kind,
            speaker_id=speaker_id,
            at=self._clock(),
            source_turn_id=candidate.turn_id,
            resolved_turn_id=resolved_turn_id,
        )
        self._next_id += 1
        return suggestion

    def observe(
        self,
        turn_id: str,
        speaker_id: str,
        candidates: Sequence[NameCandidate],
        uncertain: bool = False,
    ) -> List[NameSuggestion]:
        """Feed one completed turn; return the suggestions it produced.

        ``uncertain`` suppresses suggestions FOR this speaker entirely (a name
        attached to an identity we are not sure of is the worst outcome
        available) but does not stop this turn from ANSWERING an earlier
        addressed candidate — an answer only has to come from a different
        voice, and how sure we are which voice it was does not change that it
        was not the addresser's.
        """
        now = self._clock()
        self._turn_count += 1
        out: List[NameSuggestion] = []

        # 1. Does this turn answer an earlier "say hello, X"?
        still_pending: List[Tuple[NameCandidate, int]] = []
        for candidate, turn_index in self._pending_addressed:
            expired = (
                now - candidate.at > self.window_s
                or self._turn_count - turn_index > self.max_turns
            )
            if candidate.speaker_id == speaker_id:
                # The addresser talking again is not an answer. It does not
                # expire the candidate either -- somebody may still reply.
                if not expired:
                    still_pending.append((candidate, turn_index))
                continue
            if expired:
                continue
            if not uncertain:
                suggestion = self._make(candidate, speaker_id, resolved_turn_id=turn_id)
                if suggestion is not None:
                    out.append(suggestion)
            # Answered either way: a second reply must not re-suggest.
        self._pending_addressed = still_pending

        # 2. What did this turn itself introduce?
        for candidate in candidates:
            candidate = dataclasses.replace(
                candidate, turn_id=turn_id, speaker_id=speaker_id, at=now
            )
            if candidate.kind == KIND_SELF:
                if uncertain:
                    continue
                suggestion = self._make(candidate, speaker_id)
                if suggestion is not None:
                    out.append(suggestion)
            elif candidate.kind == KIND_THIRD_PARTY:
                # Floats: introduced, but belonging to nobody in the room yet.
                suggestion = self._make(candidate, speaker_id="")
                if suggestion is not None:
                    out.append(suggestion)
            elif candidate.kind == KIND_ADDRESSED:
                self._pending_addressed.append((candidate, self._turn_count))
        return out

    def discard(self, speaker_id: str, name: str) -> None:
        self._discarded.setdefault(speaker_id, set()).add(name)

    def pending(self) -> List[NameCandidate]:
        return [candidate for candidate, _ in self._pending_addressed]
