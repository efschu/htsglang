#!/usr/bin/env python3
"""#274 / #284: a graded, judge-free score for the lane-spec gate prompts.

WHY A SCORE AND NOT AN IDENTITY CHECK. The lane-spec gate has asked, since
round 8, whether the speculative trajectory is TOKEN-IDENTICAL to the greedy
one. #365 recorded the house finding that text identity is not an instrument;
#360 built the standard that replaced it, and its rule is two numbers, in this
order: the A-vs-A band a cross-arm delta has to clear, and the byte-identity
rate that frames whether a text difference is evidence at all.

The reason the identity check has to go here specifically is stronger than
style. Greedy speculation reproduces the greedy trajectory only if the verify
forward is bit-identical to the decode forward, and it is not: the verify runs
the target as a 2-row batch and the decode as a single row, which reassociates
the reduction. Any position whose top-2 logit margin is under that difference
may flip. The runbook already says so in its own words for stock speculation
on this vehicle (section 6.7, #139: "Do not use topk 1 as a losslessness
oracle on this configuration"). A gate that fails on such a flip is not
measuring the lane; it is measuring the margin at one position.

WHAT IS SCORED. Both gate prompts are forced continuations of a mechanical
sequence, so they can be graded exactly, with no judge and no partial credit
a rerun could re-litigate:

* ``alphabet`` -- the continuation after "...u\\nv\\n" must be w, x, y, z and
  then whatever the model does when the sequence ends. Scored as the number of
  correct next letters emitted in order before the first wrong one, capped at
  the four that are actually determined.
* ``squares`` -- the continuation after "11 121" must be "12 144", "13 169",
  ... Scored as the number of consecutive correct "n n*n" lines.

Both scores are integers a flip can move. That is the point: a trajectory that
diverges at a near tie and still emits the same correct sequence scores the
same, and the gate stays green because nothing about the answer got worse. A
trajectory that goes off the rails scores lower, and the gate goes red for a
reason a reader can check by eye.

This module is import-clean and has no server dependency, so the unit tests
can exercise the scorers directly.
"""

from __future__ import annotations

import re
from typing import Dict, List

ALPHABET_TAIL = ["w", "x", "y", "z"]
"""The letters the `alphabet` prompt determines after its last given one (v).

Past 'z' the continuation is not determined by the task, so it is not scored:
a prompt is only an instrument on the span where the right answer exists.
"""

SQUARES_START = 12
"""The first n the `squares` prompt leaves to the model (it gives 1..11)."""


def score_alphabet(text: str) -> Dict[str, int]:
    """Correct next letters, in order, before the first wrong one."""
    toks = [t.strip().lower() for t in re.split(r"[\s]+", text or "") if t.strip()]
    hit = 0
    for want, got in zip(ALPHABET_TAIL, toks):
        if got != want:
            break
        hit += 1
    return {"score": hit, "max_score": len(ALPHABET_TAIL)}


def score_squares(text: str, n_lines: int = 8) -> Dict[str, int]:
    """Consecutive correct ``n n*n`` lines from n=12, capped at `n_lines`."""
    pairs: List[tuple] = re.findall(r"(\d+)\s+(\d+)", text or "")
    hit = 0
    for i in range(n_lines):
        n = SQUARES_START + i
        if i >= len(pairs):
            break
        try:
            got_n, got_sq = int(pairs[i][0]), int(pairs[i][1])
        except ValueError:
            break
        if got_n != n or got_sq != n * n:
            break
        hit += 1
    return {"score": hit, "max_score": n_lines}


SCORERS = {"alphabet": score_alphabet, "squares": score_squares}


def score(prompt_name: str, text: str) -> Dict[str, int]:
    """Grade one continuation. An unscored prompt returns score -1.

    -1 rather than 0: "no scorer for this prompt" and "the model got nothing
    right" are different statements and must not read the same in a table.
    """
    fn = SCORERS.get(prompt_name)
    if fn is None:
        return {"score": -1, "max_score": -1}
    return fn(text)
