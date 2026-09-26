# SPDX-License-Identifier: Apache-2.0
"""Old and new names side by side, for readers that must accept both.

The rename (package and env prefix to fLLiper, subsystem token to ``PDFLIP``)
changes what the tree WRITES. Evidence written before it keeps the old
spelling forever: boot logs named ``boot_<old>_<tag>_...``, line markers
``<OLD>-GRAPH-POOL``, the front logger ``<old>.front``. The planners read such
logs at every boot (wake-credit, D-card, P-card and power-limit references),
the form/ring identity scans the evidence directory, and the report tools read
whatever log they are pointed at. A reader that only knows one spelling finds
nothing in the other and falls back silently.

WHY THE OLD TOKENS ARE SPLIT IN THIS FILE. The mechanical rename
(``tools/rename_to_flliper.py``) rewrites every contiguous old token in the
tree, readers included. A literal ``(?:OLD|NEW)`` alternation would come out of
it as ``(?:NEW|NEW)`` and accept the new spelling only. So this module never
spells an old token in one piece (``"WE" "G2"``), and it derives, at call
time, both spellings from whichever one the caller passes. A reader keeps its
literal as it is (``has_marker(line, "<OLD>-CORRIDOR")``), the rename turns
that literal into the new spelling, and the call still accepts both. The test
``test_name_compat_survives_rename`` runs this file through the rename tool and
requires the output to be byte-identical.

Readers only. Writers are not touched here; the rename renames them.
"""

from __future__ import annotations

import functools
import re
from typing import Optional, Tuple

# --------------------------------------------------------------------------
# 1a: log markers and boot-log stems
# --------------------------------------------------------------------------

#: The subsystem token, old and new, as it appears in upper-case line
#: markers (``<TOKEN>-GRAPH-POOL``, ``<TOKEN> P-DRAIN``). Old first.
MARKER_TOKENS: Tuple[str, str] = ("WE" "G2", "PDFLIP")
#: The same token in lower case: boot-log stems (``boot_<token>_``) and the
#: front logger name (``<token>.front``).
STEM_TOKENS: Tuple[str, str] = ("we" "g2", "pdflip")

_UP_ALT = "(?:%s|%s)" % MARKER_TOKENS
_LO_ALT = "(?:%s|%s)" % STEM_TOKENS

# A marker token: upper case, not glued to a preceding word character (so an
# env or constant name ``X_<TOKEN>_Y`` is never touched), followed by the
# marker separator -- a dash or a space, literal or regex-escaped, or ``\s``.
_UP_TOKEN_IN_RX = re.compile(
    r"(?<![A-Za-z0-9_])(%s|%s)(?=-|\s|\\-|\\ |\\s)" % MARKER_TOKENS
)
# A lower-case token: the stem ``boot_<token>_`` (the boot TAG that follows,
# e.g. ``<token>xsn412``, is glued to letters and never matches), or a logger
# name ``<token>.front`` (``.`` literal or escaped), not inside a dotted path.
_LO_TOKEN_IN_RX = re.compile(
    r"(?:(?<=boot_)(%s|%s)(?=_))|(?:(?<![A-Za-z0-9_.])(%s|%s)(?=\\?\.[a-z]))"
    % (STEM_TOKENS + STEM_TOKENS)
)


def tolerant_rx(pattern: str) -> str:
    """Regex SOURCE ``pattern`` with every marker/stem token widened to both
    spellings: ``<OLD>-CORRIDOR\\s+phase=`` and ``<NEW>-CORRIDOR\\s+phase=``
    both become ``(?:<OLD>|<NEW>)-CORRIDOR\\s+phase=``. Idempotent, and the
    same for the old and the new spelling of one pattern."""
    out = _UP_TOKEN_IN_RX.sub(_UP_ALT, pattern)
    return _LO_TOKEN_IN_RX.sub(_LO_ALT, out)


def tolerant_compile(pattern: str, flags: int = 0) -> "re.Pattern[str]":
    """``re.compile(tolerant_rx(pattern), flags)``."""
    return re.compile(tolerant_rx(pattern), flags)


@functools.lru_cache(maxsize=512)
def marker_variants(marker: str) -> Tuple[str, ...]:
    """Every spelling of a LITERAL marker or stem, old first:
    ``"<OLD>-FLIP begin"`` -> ``("<OLD>-FLIP begin", "<NEW>-FLIP begin")``.
    A text without a token comes back alone."""
    rx = tolerant_rx(re.escape(marker))
    if rx == re.escape(marker):
        return (marker,)
    out = []
    for up, lo in zip(MARKER_TOKENS, STEM_TOKENS):
        v = _UP_TOKEN_IN_RX.sub(up, marker)
        v = _LO_TOKEN_IN_RX.sub(lo, v)
        if v not in out:
            out.append(v)
    return tuple(out)


def has_marker(text: str, marker: str) -> bool:
    """``marker in text`` for either spelling of ``marker``."""
    return any(v in text for v in marker_variants(marker))


def marker_tail(text: str, marker: str) -> Optional[str]:
    """What follows the FIRST occurrence of ``marker`` (either spelling) in
    ``text``; ``None`` when neither occurs. ``text.split(marker, 1)[1]`` for
    both spellings."""
    best = None
    for v in marker_variants(marker):
        i = text.find(v)
        if i >= 0 and (best is None or i < best[0]):
            best = (i, v)
    if best is None:
        return None
    return text[best[0] + len(best[1]):]

