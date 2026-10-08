# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""The conversation's written record — server-held, and NOT the journal.

Design §17.2. The distinction from ``session.Journal`` is the whole reason
this module exists, and it is easy to get wrong in a way that is invisible
until a long conversation:

* the **journal** is a bounded replay buffer. It is sized in events and audio
  bytes, it is allowed to evict, and it must stay allowed to — its job is to
  let a client that missed a few seconds catch up, not to remember an hour.
* the **transcript** is the record the user reads and scrolls. It ends at an
  explicit clear and at nothing else: not a reconnect, not a mode switch, not
  a journal eviction.

So a transcript derived from journal events would silently lose its beginning
exactly when a conversation got interesting. It is text-only for the same
reason it can afford to be unbounded in practice: an hour of dense speech is a
few hundred kilobytes of text and zero bytes of audio.

The class is ``TranscriptLog`` rather than ``Transcript`` because
``backends.Transcript`` is already the ASR's result type for ONE utterance.
Two things called Transcript in one pipeline — a recognizer output and the
conversation's record — is a name collision waiting to be imported wrongly.

Lines are mutable after the fact, and every mutation returns the changed line
so the caller can put it on the wire. A retroactive rename or a resolved
speaker attribution must be VISIBLE (§17.0 rule 2) — the client patches the
line it already has; it never silently disagrees with the server.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Callable, Dict, List, Optional, Sequence

#: A line the user could disagree with carries how sure the machine was.
CONFIDENCE_EXACT = "exact"
CONFIDENCE_UNCERTAIN = "uncertain"

#: Where the speaker attribution came from. ``manual`` is ground truth and is
#: never re-decided by a later similarity computation (§17.0 rule 4).
ORIGIN_AUTO = "auto"
ORIGIN_MANUAL = "manual"

#: Ordinary utterance vs a notice the transcript writes about itself (the
#: overflow marker). A notice is rendered differently and never carries a
#: speaker, which is why it is a kind rather than a flag.
KIND_UTTERANCE = "utterance"
KIND_NOTICE = "notice"


@dataclasses.dataclass
class TranscriptLine:
    """One utterance as the reader sees it, in both languages."""

    line_id: int
    at: float
    kind: str = KIND_UTTERANCE
    turn_id: str = ""
    speaker_id: str = ""
    #: Confirmed name if there is one, else the speaker id. Denormalised on
    #: purpose: a client rendering a scrollback must not have to join against
    #: a speaker table that may have changed since.
    speaker_label: str = ""
    source_language: str = ""
    source_text: str = ""
    #: target language -> translated text. Filled in after the line is
    #: created, because the line must exist the moment the words are known —
    #: waiting for the translation would make the transcript lag the audio.
    translations: Dict[str, str] = dataclasses.field(default_factory=dict)
    confidence: str = CONFIDENCE_EXACT
    #: Ranked alternatives, populated only when ``confidence`` is uncertain.
    candidates: List[Dict[str, object]] = dataclasses.field(default_factory=list)
    origin: str = ORIGIN_AUTO
    #: Set when an uncertain attribution was later settled, so the client can
    #: show the badge CHANGING rather than quietly rewriting the line.
    resolved_by: Optional[str] = None
    text: str = ""

    def to_json(self) -> Dict[str, object]:
        out: Dict[str, object] = {
            "line_id": self.line_id,
            "at": round(self.at, 4),
            "kind": self.kind,
        }
        if self.kind == KIND_NOTICE:
            out["text"] = self.text
            return out
        out.update(
            {
                "turn_id": self.turn_id,
                "speaker_id": self.speaker_id,
                "speaker_label": self.speaker_label,
                "source_language": self.source_language,
                "source_text": self.source_text,
                "translations": dict(self.translations),
                "confidence": self.confidence,
                "origin": self.origin,
            }
        )
        if self.candidates:
            out["candidates"] = [dict(c) for c in self.candidates]
        if self.resolved_by is not None:
            out["resolved_by"] = self.resolved_by
        return out


class TranscriptLog:
    """The whole conversation, in order, until someone clears it.

    ``max_lines`` is a memory backstop rather than a policy. It is set far
    above any real conversation, and when it does bind the drop is announced
    in the transcript itself: a record that silently loses its beginning is
    worse than one that says it did.
    """

    def __init__(
        self,
        max_lines: int = 10_000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_lines < 2:
            # One slot cannot hold a line and its overflow notice, so the
            # notice would evict the very line it describes.
            raise ValueError(f"max_lines={max_lines} must be at least 2")
        self.max_lines = max_lines
        self._clock = clock
        self._lines: List[TranscriptLine] = []
        self._next_id = 1
        #: Lines dropped to the cap since the last clear, cumulative.
        self.dropped = 0

    def __len__(self) -> int:
        return len(self._lines)

    @property
    def next_line_id(self) -> int:
        return self._next_id

    @property
    def floor(self) -> int:
        """Oldest line id still held. A client below this has lost history."""
        return self._lines[0].line_id if self._lines else self._next_id

    # -- write side ---------------------------------------------------------

    def append(
        self,
        *,
        turn_id: str,
        speaker_id: str,
        speaker_label: str,
        source_language: str,
        source_text: str,
        confidence: str = CONFIDENCE_EXACT,
        candidates: Optional[Sequence[Dict[str, object]]] = None,
        origin: str = ORIGIN_AUTO,
    ) -> TranscriptLine:
        line = TranscriptLine(
            line_id=self._next_id,
            at=self._clock(),
            turn_id=turn_id,
            speaker_id=speaker_id,
            speaker_label=speaker_label or speaker_id,
            source_language=source_language,
            source_text=source_text,
            confidence=confidence,
            candidates=[dict(c) for c in (candidates or ())],
            origin=origin,
        )
        self._next_id += 1
        self._lines.append(line)
        self._trim()
        return line

    def _notice(self, text: str) -> TranscriptLine:
        line = TranscriptLine(
            line_id=self._next_id, at=self._clock(), kind=KIND_NOTICE, text=text
        )
        self._next_id += 1
        self._lines.append(line)
        return line

    def _trim(self) -> None:
        if len(self._lines) <= self.max_lines:
            return
        overflow = len(self._lines) - self.max_lines
        # Reserve one slot for the notice itself, so appending it cannot
        # trigger another trim and recurse.
        del self._lines[: overflow + 1]
        self.dropped += overflow + 1
        marker = TranscriptLine(
            line_id=self._next_id,
            at=self._clock(),
            kind=KIND_NOTICE,
            text=(
                f"{self.dropped} earlier lines were dropped: this conversation "
                f"passed the {self.max_lines}-line limit"
            ),
        )
        self._next_id += 1
        self._lines.insert(0, marker)

    def get(self, line_id: int) -> Optional[TranscriptLine]:
        for line in self._lines:
            if line.line_id == line_id:
                return line
        return None

    def set_translations(
        self, line_id: int, translations: Dict[str, str]
    ) -> Optional[TranscriptLine]:
        line = self.get(line_id)
        if line is None:
            return None
        line.translations.update(translations)
        return line

    def relabel_speaker(self, speaker_id: str, label: str) -> List[TranscriptLine]:
        """Rename every line of one speaker, retroactively (§17.3).

        Returns the lines that changed, so the caller emits exactly those.
        Returning all of them instead would look identical in a test with one
        speaker and flood the wire in a real conversation.
        """
        changed = []
        for line in self._lines:
            if line.kind == KIND_UTTERANCE and line.speaker_id == speaker_id:
                if line.speaker_label != label:
                    line.speaker_label = label
                    changed.append(line)
        return changed

    def reassign_all_of_speaker(
        self, source_id: str, speaker_id: str, speaker_label: str
    ) -> List[TranscriptLine]:
        """Move every line of one speaker to another, for a roster merge.

        Returns only the lines that changed, for the same reason
        :meth:`relabel_speaker` does: the caller emits exactly those.

        Unlike :meth:`reassign_speaker` this does NOT clear uncertainty or
        rewrite ``origin``/``resolved_by``. A merge says two clusters are one
        person; it says nothing about whether the recognizer was sure of any
        individual line at the time, and overwriting that would erase the only
        record of how the split happened. A line that carried candidates keeps
        them -- they name speakers, and the merged-away id among them is
        handled where candidates are rendered.
        """
        changed = []
        for line in self._lines:
            if line.kind != KIND_UTTERANCE or line.speaker_id != source_id:
                continue
            line.speaker_id = speaker_id
            line.speaker_label = speaker_label or speaker_id
            changed.append(line)
        return changed

    def reassign_speaker(
        self,
        line_id: int,
        speaker_id: str,
        speaker_label: str,
        *,
        origin: str = ORIGIN_MANUAL,
        resolved_by: Optional[str] = None,
    ) -> Optional[TranscriptLine]:
        """Move one line to another speaker and clear its uncertainty."""
        line = self.get(line_id)
        if line is None or line.kind != KIND_UTTERANCE:
            return None
        line.speaker_id = speaker_id
        line.speaker_label = speaker_label or speaker_id
        line.confidence = CONFIDENCE_EXACT
        line.candidates = []
        line.origin = origin
        line.resolved_by = resolved_by
        return line

    def clear(self) -> int:
        """The only thing that empties the transcript. Returns lines removed."""
        removed = len(self._lines)
        self._lines.clear()
        self.dropped = 0
        # Line ids keep counting. A client holding a cursor from before the
        # clear must not be handed lines it already saw under the same ids.
        return removed

    # -- read side ----------------------------------------------------------

    def since(self, cursor: int) -> List[TranscriptLine]:
        return [line for line in self._lines if line.line_id > cursor]

    def lines(self) -> List[TranscriptLine]:
        return list(self._lines)

    def to_json(self, since: int = 0) -> Dict[str, object]:
        return {
            "lines": [line.to_json() for line in self.since(since)],
            "next_line_id": self._next_id,
            "floor": self.floor,
            "dropped": self.dropped,
            "max_lines": self.max_lines,
        }
