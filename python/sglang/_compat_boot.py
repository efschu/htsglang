"""First statement of the package: environment bridge and rig-state directory (RENAME_PLAN 8.7 step 2).

Runs before any module reads the environment, so every reader sees the canonical spelling and
child processes inherit it. The bridge itself is ``srt/name_compat.py`` (owned by the NF seat,
also called by the launcher's ``build_env`` and the container entrypoint): ONE helper, not two.
A missing helper is an ImportError on purpose -- a silently skipped bridge would let profiles
that set the old spelling fall back to defaults, the most dangerous class of the rename.

What the bridge SAYS (F0-C): when both spellings of a variable are set with different values the
renamed one wins and one warning line names both variables (names only, never values: an env value
may be a key); when legacy-spelled variables were read through the mirror (after the rename) ONE
deprecation line lists them. Both are printed at most once per process; a child of a process that
already folded its environment sees only canonical spellings and says nothing.

This file spells no legacy name in one piece: the mechanical rename rewrites it too and the
announcement must read the same on both sides of it (test_name_compat_survives_rename).
"""

import logging
import os
from typing import Dict, List, Optional

_logger = logging.getLogger(__name__)

#: what this process has already announced (once per process)
_announced: Dict[str, bool] = {}
_announced_conflicts: set = set()

_LIST_MAX = 8


def _names(names: List[str]) -> str:
    names = sorted(set(names))
    more = len(names) - _LIST_MAX
    return ", ".join(names[:_LIST_MAX]) + (" (+%d more)" % more if more > 0 else "")


def reset_announcements() -> None:
    """Forget what was announced (tests only; a process announces once)."""
    _announced.clear()
    _announced_conflicts.clear()


def announce(report: Optional[Dict[str, List]]) -> List[str]:
    """Log what ``name_compat.canonical_env(..., report=...)`` found; returns the lines logged now.

    One warning per conflicting variable pair (the renamed value won), one deprecation line per
    process for the legacy-spelled variables that were read through the mirror."""
    out: List[str] = []
    if not report:
        return out
    for legacy, renamed in report.get("conflicts", ()):
        if (legacy, renamed) in _announced_conflicts:
            continue
        _announced_conflicts.add((legacy, renamed))
        out.append("compat: %s and %s are both set with different values; %s wins (the renamed variable)"
                   % (legacy, renamed, renamed))
    legacy_used = report.get("legacy", ())
    if legacy_used and not _announced.get("deprecation"):
        _announced["deprecation"] = True
        out.append("compat: DEPRECATED legacy-prefixed environment variable(s) were mirrored onto the new names: %s; "
                   "the legacy spelling is read for one release cycle, set the new names (FLLIPER_*, subsystem "
                   "FLLIPER_PDFLIP_*) instead" % _names(list(legacy_used)))
    for line in out:
        _logger.warning(line)
    return out


def bridge_environ(environ=None) -> None:
    from .srt.name_compat import canonical_env

    report: Dict[str, List] = {}
    canonical_env(os.environ if environ is None else environ, report=report)
    announce(report)


def link_state_dir() -> None:
    """Called by the server entry points (launch_server, the flip launcher), never at import."""
    from .srt.compat_shims import link_legacy_cache_dir

    link_legacy_cache_dir()
