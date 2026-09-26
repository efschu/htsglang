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
import os
import re
import shlex
import sys
from typing import Dict, List, MutableMapping, Optional, Tuple

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


# --------------------------------------------------------------------------
# 1b: environment names
# --------------------------------------------------------------------------

_LEG = "SG" "LANG_"  # the legacy env prefix, split for the rename (module doc)
_LEG_SUB = _LEG + "WE" "G2_"

#: (legacy, renamed) env prefix pairs, the more specific first: the subsystem
#: family ``<LEGACY>_<OLD>_X <-> FLLIPER_PDFLIP_X`` (RENAME_PLAN 8.1) before the
#: generic ``<LEGACY>_X <-> FLLIPER_X`` (RENAME_PLAN 4.1). ``SGL_*`` (upstream
#: legacy aliases) and the product prefix ``HT...`` belong to neither family.
ENV_PREFIX_PAIRS: Tuple[Tuple[str, str], ...] = (
    (_LEG_SUB, "FLLIPER_PDFLIP_"),
    (_LEG, "FLLIPER_"),
)

def _package_name() -> str:
    if __name__ != "__main__":
        return __name__.split(".")[0]
    # run as a script (the container entrypoint): <tree>/python/<pkg>/srt/name_compat.py
    return os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


#: Which side of each pair THIS tree reads. The package name decides, at run
#: time, so the same file bridges in both directions: before the rename the
#: legacy names are canonical (a stray new name is folded onto them), after it
#: the new names are.
_PKG = _package_name()
CANONICAL_SIDE: int = 0 if _PKG == "sg" "lang" else 1

#: Names read by code the rename does not touch, in their LEGACY spelling:
#: the ``sgl_kernel`` wheel (``getenv`` in its csrc, ``os.environ`` in its
#: python), the JIT CUDA sources under ``jit_kernel/csrc`` and the other C++
#: the tool skips (``--cxx`` off): the host-ring TMS and the C++ radix tree;
#: plus the rust gRPC server and the model gateway. Found by a search for
#: ``getenv`` / literal legacy names in sgl-kernel, 3rdparty, rust,
#: sgl-model-gateway and every C/C++/CUDA file of the tree (rename step 1b).
#: Each keeps its legacy spelling set whenever any spelling is set.
FOREIGN_READERS: frozenset = frozenset(
    [_LEG + s for s in (
        # sgl-kernel (csrc getenv, python/sgl_kernel/debug_utils.py)
        "CUSTOM_ALLREDUCE_ALGO", "GGUF_KQ_KERNEL", "RPF_N", "KERNEL_API_LOGLEVEL",
        # jit_kernel/csrc (JIT-compiled CUDA)
        "DEBUG_C128_ONLINE_GUARD", "DEBUG_C128_ONLINE_NO_H2D", "DEBUG_C128_ONLINE_SYNC_H2D",
        "MARLIN_EPILOGUE_SYNC", "MARLIN_NO_K_SPLIT", "MARLIN_SMS_OVERRIDE",
        "OPT_FUSED_MOE_ACTIVATION_QUANT_FUSE", "OPT_FUSED_MOE_ACTIVATION_VEC",
        # mem_cache/cpp_radix_tree (C++)
        "RADIX_CPP_DEBUG_LIMIT",
        # rust/ gRPC server, sgl-model-gateway (foreign packages)
        "TONIC_PAYLOAD", "LOG_MS", "MCP_CONFIG",
    )]
    + [_LEG_SUB + s for s in (
        # the host-ring torch_memory_saver sources (tms_csrc/utils.h getenv)
        "VMM_EXPORTABLE",
    )]
)


def _env_family(name: str) -> Optional[Tuple[str, str, str]]:
    """``(legacy, renamed, canonical)`` spelling of an env name that belongs to
    one of :data:`ENV_PREFIX_PAIRS`, else ``None``."""
    for pair in ENV_PREFIX_PAIRS:
        for side in (0, 1):
            if name.startswith(pair[side]):
                rest = name[len(pair[side]):]
                if not rest:
                    return None
                leg, new = pair[0] + rest, pair[1] + rest
                return leg, new, (leg, new)[CANONICAL_SIDE]
    return None


def canonical_env_name(name: str) -> str:
    """The spelling of ``name`` this tree reads (itself when it has no
    legacy/renamed counterpart)."""
    fam = _env_family(name)
    return fam[2] if fam else name


def canonical_env(env: MutableMapping, *, foreign_keep=FOREIGN_READERS) -> MutableMapping:
    """Fold every legacy/renamed env spelling onto the one this tree reads.

    In place; returns ``env``. When several spellings of one name are
    present, the canonical spelling's value wins (it is what the tree has
    always read, and after the rename it is the explicit new name); when the
    canonical spelling is absent, the other one's value moves onto it. Then
    every non-canonical spelling is REMOVED, so a later ``env.pop(canonical)``
    removes the variable -- no other spelling survives to be mirrored back by a
    child's own import. The exception is ``foreign_keep`` (legacy spellings
    read by code outside the rename): for those the legacy spelling is always
    set, and both spellings, where present, carry the resolved value.

    An env that holds only canonical spellings of non-foreign names comes back
    unchanged, key order included.
    """
    groups: Dict[Tuple[str, str, str], List[str]] = {}
    for k in list(env.keys()):
        fam = _env_family(k)
        if fam is not None:
            groups.setdefault(fam, []).append(k)
    for (leg, new, canon), present in groups.items():
        foreign = leg in foreign_keep
        if present == [canon] and (not foreign or leg == canon):
            continue
        if canon in env:
            value = env[canon]
        else:
            other = new if canon == leg else leg
            value = env[other] if other in env else env[present[0]]
            env[canon] = value
        for k in present:
            if k != canon and not (foreign and k in (leg, new)):
                del env[k]
        if foreign:
            env[leg] = value
            if new in env:
                env[new] = value
    return env


def shell_statements(env: Optional[MutableMapping] = None) -> List[str]:
    """``unset``/``export`` lines that turn ``env`` (default: this process's
    environment) into :func:`canonical_env` of it -- for shell launchers (the
    container entrypoint) that cannot import the package."""
    before = dict(os.environ if env is None else env)
    after = canonical_env(dict(before))
    out = ["unset %s" % k for k in before if k not in after]
    out += ["export %s=%s" % (k, shlex.quote(v)) for k, v in after.items() if before.get(k) != v]
    return out


if __name__ == "__main__":
    # eval "$(python <tree>/python/<pkg>/srt/name_compat.py --shell)"
    if sys.argv[1:] != ["--shell"]:
        sys.exit("usage: name_compat.py --shell")
    print("\n".join(shell_statements()))
