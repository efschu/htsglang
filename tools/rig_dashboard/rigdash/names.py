# SPDX-License-Identifier: Apache-2.0
"""Old and new names side by side, for the dashboard's READERS (rename F0-B, R5).

The rename (legacy package name -> ``flliper``, legacy subsystem name -> ``pdflip``,
legacy env prefixes -> ``FLLIPER_`` / ``FLLIPER_PDFLIP_``, legacy product name ->
``flliper``) changes what a tree WRITES. Evidence written before it keeps the
old spelling forever (boot logs, ``state.json``, container names), and a
dashboard started after the rename reads old boots and new boots side by side.
A parser that knows one spelling reports the other generation's boot as "no
boot": no flip, no group log, no container.

WHY THE OLD TOKENS ARE SPLIT IN THIS FILE. The mechanical rename
(the rename tool, ``rename_rigdash.py``) rewrites every
contiguous old token in this package, readers included. A literal
``(?:OLD|NEW)`` alternation would come out of it as ``(?:NEW|NEW)``. So no old
token is spelled in one piece here (``"WE" "G2"``), and every helper derives
BOTH spellings at call time from whichever one the caller passes. A reader
keeps its literal as it is (``has_marker(line, "<OLD>-FLIP begin")``), the
rename turns that literal into the new spelling, and the call still accepts
both. ``test_names_dual_f0b_1007`` runs this file through the rename tool and
requires the output to be byte-identical.

Same technique and same token tables as ``srt/name_compat.py`` of the
tree (RENAME_PLAN 8.14). Own copy because the dashboard never imports the
tree (``kartenplan_gate``: modules are loaded by file path). Readers only.
"""

from __future__ import annotations

import functools
import re
from typing import Any, Mapping, Optional, Tuple

#: subsystem token, old first: upper-case line markers ``<TOKEN>-FLIP``, ``<TOKEN> SESSION``
MARKER_TOKENS: Tuple[str, str] = ("WE" "G2", "PDFLIP")
#: lower case: boot-log stems ``boot_<token>_``, logger ``<token>.front``, schemas ``<token>.state/1``,
#: route prefix ``/<token>/state``, image tag part ``-<token>-``
STEM_TOKENS: Tuple[str, str] = ("we" "g2", "pdflip")
#: CapWords class names ``<Token>WakeRefused``; the rename tool maps ``<Token>Flip<X>`` and ``<Token><X>`` both to ``PdFlip<X>``
CAMEL_TOKENS: Tuple[str, str] = ("We" "g2", "PdFlip")
_CAMEL_FLIP = "We" "g2" "Flip"
#: product name: container names, image repository, state volume ``/var/lib/<product>``
PRODUCT_TOKENS: Tuple[str, str] = ("hts" "glang", "flliper")

_UP_ALT = "(?:%s|%s)" % MARKER_TOKENS
_LO_ALT = "(?:%s|%s)" % STEM_TOKENS

# A marker token: upper case, not glued to a preceding word character (so ``X_<TOKEN>_Y`` env names are never
# touched), followed by the marker separator: a dash or a space, literal or regex-escaped, or ``\s``.
_UP_TOKEN_IN_RX = re.compile(r"(?<![A-Za-z0-9_])(%s|%s)(?=-|\s|\\-|\\ |\\s)" % MARKER_TOKENS)
# Lower case: the stem ``boot_<token>_``; a logger/schema/route name ``<token>.x`` / ``<token>/x``
# (``.`` or ``/`` literal or escaped, not inside a dotted path); the image-tag part ``-<token>-``.
_LO_TOKEN_IN_RX = re.compile(
    r"(?:(?<=boot_)(%s|%s)(?=_))"
    r"|(?:(?<![A-Za-z0-9_.])(%s|%s)(?=\\?\.[a-z]))"
    r"|(?:(?<=/)(%s|%s)(?=\\?/[a-z]))"
    r"|(?:(?<=-)(%s|%s)(?=\\?-))" % (STEM_TOKENS * 4)
)
# CapWords: the token at the start of a class name (``<Token>WakeRefused``; ``<Token>FlipRankDisagree`` -> ``PdFlipRankDisagree``).
# ``<Token>Flip`` is tried before ``<Token>`` (the rename tool's order), so ``<Token>FlipX`` is the token ``<Token>Flip`` plus ``X``.
_CAMEL_IN_RX = re.compile(r"(?<![A-Za-z0-9_])(%s|%s|%s)(?=[A-Z])" % (_CAMEL_FLIP, CAMEL_TOKENS[0], CAMEL_TOKENS[1]))


def _camel_spellings(token: str) -> Tuple[str, ...]:
    """Every spelling of a CapWords token: old ``<Token>Flip`` / ``<Token>`` map to ``PdFlip``; ``PdFlip`` may stand for either."""
    if token == CAMEL_TOKENS[1]:
        return (CAMEL_TOKENS[1], _CAMEL_FLIP, CAMEL_TOKENS[0])
    return (token, CAMEL_TOKENS[1])


def _camel_alt(m: "re.Match[str]") -> str:
    return "(?:%s)" % "|".join(_camel_spellings(m.group(1)))


def tolerant_rx(pattern: str) -> str:
    """Regex SOURCE ``pattern`` with every marker/stem/camel token widened to both spellings
    (``<OLD>-FLIP`` and ``<NEW>-FLIP`` both become ``(?:<OLD>|<NEW>)-FLIP``). Idempotent, and the same for the
    old and the new spelling of one pattern."""
    out = _UP_TOKEN_IN_RX.sub(_UP_ALT, pattern)
    out = _LO_TOKEN_IN_RX.sub(_LO_ALT, out)
    return _CAMEL_IN_RX.sub(_camel_alt, out)


def tolerant_compile(pattern: str, flags: int = 0) -> "re.Pattern[str]":
    """``re.compile(tolerant_rx(pattern), flags)``."""
    return re.compile(tolerant_rx(pattern), flags)


@functools.lru_cache(maxsize=512)
def marker_variants(marker: str) -> Tuple[str, ...]:
    """Every spelling of a LITERAL marker/stem, old first:
    ``"<OLD>-FLIP begin"`` -> ``("<OLD>-FLIP begin", "<NEW>-FLIP begin")``; a text without a token comes back alone."""
    rx = tolerant_rx(re.escape(marker))
    if rx == re.escape(marker):
        return (marker,)
    out = []
    camel = _CAMEL_IN_RX.search(marker)
    spell = _camel_spellings(camel.group(1)) if camel else (None,)
    # one spelling per token generation; a CapWords token adds its alternatives (see _camel_spellings)
    for i in (0, 1):
        v = _UP_TOKEN_IN_RX.sub(MARKER_TOKENS[i], marker)
        v = _LO_TOKEN_IN_RX.sub(STEM_TOKENS[i], v)
        for c in (spell if camel else (None,)):
            w = v if c is None else _CAMEL_IN_RX.sub(c, v)
            if w not in out:
                out.append(w)
    return tuple(out)


def has_marker(text: str, marker: str) -> bool:
    """``marker in text`` for either spelling of ``marker``."""
    return any(v in text for v in marker_variants(marker))


def starts_with(text: str, marker: str) -> bool:
    """``text.startswith(marker)`` for either spelling of ``marker``."""
    return any(text.startswith(v) for v in marker_variants(marker))


def find_marker(text: str, marker: str) -> Tuple[int, str]:
    """``(index, spelling)`` of the EARLIEST occurrence of either spelling of ``marker`` in ``text``;
    ``(-1, marker)`` when neither occurs. ``text.find(marker)`` for both spellings."""
    best = (-1, marker)
    for v in marker_variants(marker):
        i = text.find(v)
        if i >= 0 and (best[0] < 0 or i < best[0]):
            best = (i, v)
    return best


def marker_tail(text: str, marker: str) -> Optional[str]:
    """What follows the FIRST occurrence of ``marker`` (either spelling); ``None`` when neither occurs."""
    i, v = find_marker(text, marker)
    return None if i < 0 else text[i + len(v):]


def bytes_variants(marker: str) -> Tuple[bytes, ...]:
    """:func:`marker_variants` as bytes (for readers that scan raw log chunks)."""
    return tuple(v.encode() for v in marker_variants(marker))


def has_marker_bytes(data: bytes, marker: str) -> bool:
    return any(v in data for v in bytes_variants(marker))


def glob_variants(pattern: str) -> Tuple[str, ...]:
    """Both spellings of a glob with a boot-log stem: ``boot_<OLD>_%s_*.D.log`` -> that and ``boot_<NEW>_%s_*.D.log``.
    A glob without a token comes back alone."""
    return marker_variants(pattern)


# --------------------------------------------------------------------------
# environment names, paths, product name
# --------------------------------------------------------------------------

_LEG = "SG" "LANG_"
_PRODUCT_ENV = "HTS" "GLANG_"
#: (legacy, renamed) env prefix pairs, most specific first (same order as name_compat.ENV_PREFIX_PAIRS, plus the
#: product prefix that the rename maps to ``FLLIPER_`` as well, RENAME_PLAN 4.1/3.4)
ENV_PREFIX_PAIRS: Tuple[Tuple[str, str], ...] = (
    (_LEG + "WE" "G2_", "FLLIPER_PDFLIP_"),
    (_LEG + "OPT_" + "WE" "G2_", "FLLIPER_OPT_PDFLIP_"),
    (_LEG, "FLLIPER_"),
    (_PRODUCT_ENV, "FLLIPER_"),
)


def env_variants(name: str) -> Tuple[str, ...]:
    """Every spelling of an env name, the given one first: ``<LEGACY>_<OLD>_X`` <-> ``FLLIPER_PDFLIP_X``,
    ``<LEGACY>_X`` <-> ``FLLIPER_X`` and ``<PRODUCT>_X`` <-> ``FLLIPER_X``; ``FLLIPER_X`` answers to both legacy prefixes (a
    renamed name does not remember which one it came from). A name outside the families comes back alone."""
    for leg, new in ENV_PREFIX_PAIRS:
        if name.startswith(leg) and len(name) > len(leg):
            return (name, new + name[len(leg):])
        if name.startswith(new) and len(name) > len(new):
            rest = name[len(new):]
            if new == "FLLIPER_":
                return (name, _LEG + rest, _PRODUCT_ENV + rest)
            return (name, leg + rest)
    return (name,)


def env_get(env: Optional[Mapping[str, Any]], name: str, default: Any = None) -> Any:
    """``env.get(name, default)`` over every spelling of ``name`` (the given one wins when several are set)."""
    if not env:
        return default
    for v in env_variants(name):
        if v in env:
            return env[v]
    return default


#: the state volume of a container, new first is not meaningful here: both are looked at
STATE_ROOTS: Tuple[str, str] = ("/var/lib/" + PRODUCT_TOKENS[0], "/var/lib/" + PRODUCT_TOKENS[1])


def state_path_variants(path: str) -> Tuple[str, ...]:
    """``/var/lib/<OLD product>/evidence`` -> that and ``/var/lib/<NEW product>/evidence``."""
    for root in STATE_ROOTS:
        if path == root or path.startswith(root + "/"):
            return tuple(r + path[len(root):] for r in STATE_ROOTS)
    return (path,)


def docker_name_filters() -> str:
    """``--filter name=<OLD> --filter name=<NEW>``: docker ORs repeated filters of one key, so a ``docker ps`` that
    lists the old generation's containers lists the renamed ones too."""
    return " ".join("--filter name=%s" % p for p in PRODUCT_TOKENS)


def name_match_rx() -> str:
    """Regex source matching a name of either product spelling (container, image repository)."""
    return "(?:%s|%s)" % PRODUCT_TOKENS


#: (package, subsystem sub-package) of a tree, old first: ``python/<pkg>/srt/<sub>``
PKG_PAIRS: Tuple[Tuple[str, str], ...] = (("sg" "lang", STEM_TOKENS[0]), ("flliper", STEM_TOKENS[1]))


def schema_ok(value: Any, schema: str) -> bool:
    """``value == schema`` for either spelling of ``schema`` (``<OLD>.state/1`` / ``<NEW>.state/1``)."""
    return isinstance(value, str) and value in marker_variants(schema)
