"""Confine a caller-supplied filesystem path to a configured root.

Deliberately dependency-free (no torch, no fastapi) so both the HTTP handler
that answers 400 and the scheduler-side sink that raises can import it, and so
it is unit-testable on its own.

Why this exists (audit #506, finding A2-F1): ``POST /hibernate`` accepted a
``hibernate_dir`` in the request body that OVERRODE ``--hibernate-dir`` and
went straight into ``os.makedirs()`` / ``os.path.join()``. A caller who could
reach the port could name the directory every rank's full post-transform
weight shards were written to.
"""

from __future__ import annotations

import os
from typing import Optional

__all__ = ["PathConfinementError", "confine_to_root"]


class PathConfinementError(ValueError):
    """A requested path resolves outside the configured root."""


def confine_to_root(
    requested: Optional[str],
    root: Optional[str],
    *,
    what: str = "directory",
    flag: str = "--hibernate-dir",
) -> str:
    """Resolve ``requested`` and require it to be ``root`` or below it.

    ``requested`` of ``None``/empty means "use the configured root", which is
    the unchanged default path.

    The comparison is on ``os.path.realpath`` of both sides, so ``..``
    segments and symlinks that leave the root are refused rather than
    normalised away, and it is a **path** comparison, not a string-prefix one:
    ``/park-evil`` is not inside ``/park``.

    Neither side has to exist. ``realpath`` resolves the portion that does and
    leaves the rest lexically normalised, which is what makes this usable
    before the directory is created.

    Raises:
        PathConfinementError: if ``root`` is unset (nothing to confine
            against, so a request-supplied path can never be honoured), or if
            ``requested`` resolves outside it.
    """
    if not root:
        if not requested:
            raise PathConfinementError(
                f"no {what} configured: pass {flag} on the command line."
            )
        raise PathConfinementError(
            f"a {what} in the request is only honoured when {flag} is "
            f"configured, because there is otherwise nothing to confine it to; "
            f"got {requested!r}."
        )

    root_real = os.path.realpath(root)
    if not requested:
        return root_real

    requested_real = os.path.realpath(requested)
    if requested_real == root_real:
        return requested_real
    # os.path.commonpath compares path components, so a shared string prefix
    # ("/park" vs "/park-evil") is not a containment.
    try:
        inside = os.path.commonpath([root_real, requested_real]) == root_real
    except ValueError:
        # Different drives / one relative: never containment.
        inside = False
    if not inside:
        raise PathConfinementError(
            f"the requested {what} {requested!r} resolves to {requested_real!r}, "
            f"which is outside the configured {flag} root {root_real!r}. "
            f"Only that directory and paths below it are accepted."
        )
    return requested_real
