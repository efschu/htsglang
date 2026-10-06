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


# an absolute path into the host (or the container) filesystem: nothing about the box's layout leaves in a pasted issue
_HOSTPATH = re.compile(r"(?<![\w.:/-])/(?:home|root|spinning|opt|var|usr|tmp|mnt|srv|etc|data|run|media|nix|proc|sys|dev)(?:/[^\s|`,;)\]\"'<>]*)*")


def paths(text: str) -> str:
    """Replace absolute host paths (``/root/...``, ``/spinning/...``, ``/var/lib/...``) by ``<Pfad entfernt>``."""
    return _HOSTPATH.sub("<Pfad entfernt>", text or "")


def text_for_issue(text: str) -> str:
    """Text that is pasted into a public issue: secret lines dropped whole, secret values and host paths cut, line by line."""
    out = []
    for line in (text or "").split("\n"):
        c = clean(line)
        if c is not None:
            out.append(paths(c))
    return "\n".join(out)


# the NAME of a flag, variable or env entry says it carries a secret: its value never goes into a pasted issue.  ``clean`` cannot see these
# (``HF_TOKEN=...`` has no word boundary before ``TOKEN``, and a table cell ``| HF_TOKEN | abc |`` has no ``=`` at all).  The name is split into
# its letter runs (``HF_TOKEN`` -> hf, token; ``--max-total-tokens`` -> max, total, tokens), so ``tokens`` / ``tokenizer`` are NOT secrets.
_SECRET_WORDS = frozenset(("secret", "secrets", "password", "passwd", "pwd", "token", "credential", "credentials", "bearer", "authorization", "pat",
                           "apikey", "adminkey", "accesskey", "privatekey"))
_KEY_PREFIX = frozenset(("api", "admin", "access", "private", "auth", "ssh"))


def _words(name: str):
    out, cur = [], []
    for ch in str(name or "").lower():
        if "a" <= ch <= "z":
            cur.append(ch)
        elif cur:
            out.append("".join(cur))
            cur = []
    if cur:
        out.append("".join(cur))
    return out


def secret_name(name: str) -> bool:
    """True when ``name`` (a flag, profile variable or env name) names a secret, e.g. ``HF_TOKEN``, ``--admin-api-key``, ``GITHUB_PAT``."""
    w = _words(name)
    return any(x in _SECRET_WORDS for x in w) or any(a in _KEY_PREFIX and b == "key" for a, b in zip(w, w[1:]))


def value_for_issue(name: str, value) -> str:
    """The value of a named entry for a pasted issue: ``<entfernt>`` when the name says secret, else the value with secrets and host paths cut."""
    if secret_name(name):
        return "<entfernt>"
    return text_for_issue(str(value if value is not None else ""))
