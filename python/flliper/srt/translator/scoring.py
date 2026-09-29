# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""The intelligibility instrument: ASR round-trip word error rate.

DESIGN_466 §7(b)(3) makes ASR-round-trip WER the **hard gate** for every TTS
candidate: synthesize, transcribe the result in the target language, and
compare against the text that was asked for. Speaker similarity ranks the
survivors; intelligibility decides who survives. Accent is never scored.

The gate is a real one because it fails on the thing that fools every cheaper
signal. Synthesized babble is finite, speech-shaped, correctly pitched and
scores beautifully on speaker similarity -- and transcribes to nothing like
the requested sentence. That is not hypothetical here: a randomly initialised
talker produced exactly that and passed every other check (see
:mod:`test_weight_loading`).

Normalisation is deliberately aggressive and deliberately NOT
language-specific. Case, punctuation and digit grouping are noise for this
purpose, and any per-language rule would put a language name into a module the
whole point of which is that no language is named anywhere in the deciding
code. Unicode is normalised to NFKC and combining marks are kept: dropping
accents would make ``si``/``sí`` and ``uber``/``über`` compare equal, which
hides exactly the vowel errors a cross-lingual synthesizer makes.
"""

from __future__ import annotations

import dataclasses
import re
import unicodedata
from typing import List, Sequence

__all__ = ["normalize_for_wer", "word_error_rate", "WerResult"]

#: Anything that is not a letter, a digit or an intra-word apostrophe.
_NOISE = re.compile(r"[^\w'’]+", re.UNICODE)


def normalize_for_wer(text: str) -> List[str]:
    """Lowercase, strip punctuation, collapse whitespace, keep accents."""
    folded = unicodedata.normalize("NFKC", text).casefold()
    words = [w.strip("'’") for w in _NOISE.split(folded)]
    return [w for w in words if w]


@dataclasses.dataclass(frozen=True)
class WerResult:
    """A WER with the edit counts that produced it.

    The counts are reported because the rate alone cannot distinguish "the
    synthesizer dropped the second half of the sentence" (deletions) from
    "it said something else entirely" (substitutions), and those have
    different causes and different fixes.
    """

    rate: float
    substitutions: int
    deletions: int
    insertions: int
    reference_words: int
    hypothesis_words: int

    def to_json(self) -> dict:
        return dataclasses.asdict(self)


def word_error_rate(reference: str, hypothesis: str) -> WerResult:
    """Levenshtein distance over words, divided by the reference length.

    Standard WER, including its standard sharp edge: the rate can exceed 1.0
    when the hypothesis is longer than the reference. That is left uncapped on
    purpose -- a runaway synthesizer that produces four times the requested
    words should score four times the requested words, not saturate at "1.0,
    same as silence".
    """
    ref = normalize_for_wer(reference)
    hyp = normalize_for_wer(hypothesis)
    if not ref:
        raise ValueError("the reference text is empty; there is nothing to score")

    # Full DP table: the sentences are one utterance long, so the memory saved
    # by a rolling row is not worth losing the backtrace of edit types.
    rows, cols = len(ref) + 1, len(hyp) + 1
    cost = [[0] * cols for _ in range(rows)]
    op = [[""] * cols for _ in range(rows)]
    for i in range(1, rows):
        cost[i][0] = i
        op[i][0] = "d"
    for j in range(1, cols):
        cost[0][j] = j
        op[0][j] = "i"
    for i in range(1, rows):
        for j in range(1, cols):
            if ref[i - 1] == hyp[j - 1]:
                cost[i][j] = cost[i - 1][j - 1]
                op[i][j] = "="
                continue
            substitute = cost[i - 1][j - 1] + 1
            delete = cost[i - 1][j] + 1
            insert = cost[i][j - 1] + 1
            best = min(substitute, delete, insert)
            cost[i][j] = best
            op[i][j] = "s" if best == substitute else ("d" if best == delete else "i")

    substitutions = deletions = insertions = 0
    i, j = len(ref), len(hyp)
    while i > 0 or j > 0:
        move = op[i][j]
        if move == "=":
            i, j = i - 1, j - 1
        elif move == "s":
            substitutions += 1
            i, j = i - 1, j - 1
        elif move == "d":
            deletions += 1
            i -= 1
        else:
            insertions += 1
            j -= 1

    return WerResult(
        rate=(substitutions + deletions + insertions) / len(ref),
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
        reference_words=len(ref),
        hypothesis_words=len(hyp),
    )


def worst_of(results: Sequence[WerResult]) -> float:
    """The gate reads the worst arm, never the mean.

    Averaging hides a single unintelligible utterance behind several good
    ones, and one unintelligible turn in a conversation is the failure the
    gate exists to catch.
    """
    return max((r.rate for r in results), default=0.0)
