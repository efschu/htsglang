"""Keep secrets out of every page and JSON answer (the dashboard is reachable from outside).

Two layers: text taken from a log is cleaned when it is ingested (``clean``;
a line that names a key is dropped whole), and every JSON answer passes
``guard`` before it leaves, so a secret that slipped past the first layer is
cut out at the door instead of published.  Measured 2026-09-27: the weg2
front log carries ``WEG2 ADMIN-KEY minted for this boot -> <path>.adminkey``
and ``RPC auth=bearer ...`` lines, and ``WEG2-GROUP-ENV`` lines with the
whole group environment.
"""

from __future__ import annotations

import re

# a line matching this is dropped whole
DROP_LINE = re.compile(r"ADMIN-KEY|admin-api-key|\.adminkey\b|auth=bearer|Authorization:|WEG2-GROUP-ENV|GITHUB_PAT|openrouter\.key",
                       re.IGNORECASE)
# inline values that are cut out of a kept line
_VALUE = re.compile(
    r"(?i)\b(api[_-]?key|admin[_-]?key|access[_-]?key|secret|password|passwd|bearer|token)(\s*[=:]\s*|\s+)"
    r"(?!s\b)([A-Za-z0-9_\-./+=]{8,})")
_KEYFILE = re.compile(r"(?i)[\w./-]*(\.adminkey|\.pem|\.key|_PAT)\b")


def clean(text: str):
    """None if the line must not be shown at all, else the line with values cut."""
    if text is None:
        return None
    if DROP_LINE.search(text):
        return None
    text = _VALUE.sub(lambda m: m.group(1) + m.group(2) + "<entfernt>", text)
    return _KEYFILE.sub("<Schluesseldatei entfernt>", text)


# the door: a serialized answer must not contain any of these
_DOOR = re.compile(r"ADMIN-KEY|admin-api-key|\.adminkey|auth=bearer|GITHUB_PAT|openrouter\.key|WEG2-GROUP-ENV", re.IGNORECASE)


def guard(body: str) -> str:
    return _DOOR.sub("<entfernt>", body)
