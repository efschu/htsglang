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


# Every OTHER absolute path (``/nvme/hf``, ``/workspace/models/X``, ``/scratch``) and every ``~/`` / ``$HOME/`` path is a host path too: the Laufbericht
# shows no path outside the container.  Exempt: the mount points the image itself owns (``CONTAINER_MOUNTS``, ``/app`` = the baked tree).  Everything
# else shrinks to ``<hostpfad>/<last segment>``; the first pass above keeps its older, stricter ``<Pfad entfernt>`` for the system roots.
CONTAINER_MOUNTS = ("/app", "/api")                                # ``/api/...`` = a dashboard URL route (the report names them), not a directory
_PATHTAIL = r"[^\s|`,;)\]\"'<>]*"
_ANYPATH = re.compile(r"(?<![\w.:/>~<$-])/(?![/\s])" + _PATHTAIL)
_HOMEPATH = re.compile(r"(?<![\w.:/>~<$-])(?:~|\$\{HOME\}|\$HOME)(?:/" + _PATHTAIL + r")+")


def _shrink(m) -> str:
    p = m.group(0)
    if any(p == c or p.startswith(c + "/") for c in CONTAINER_MOUNTS):
        return p
    last = p.rstrip("/").rsplit("/", 1)[-1].strip(".")
    return "<hostpfad>/" + last if last and last not in ("~", "$HOME", "${HOME}") else "<hostpfad>"


def paths(text: str) -> str:
    """Replace absolute host paths: the system roots (``/root/...``, ``/spinning/...``, ``/var/lib/...``) by ``<Pfad entfernt>``, any other absolute
    path and any ``~/...`` path by ``<hostpfad>/<last segment>`` (``CONTAINER_MOUNTS``: the image's ``/app`` and the URL routes ``/api/...`` stay)."""
    text = _HOSTPATH.sub("<Pfad entfernt>", text or "")
    return _ANYPATH.sub(_shrink, _HOMEPATH.sub(_shrink, text))


# a secret recognised by the FORM of its value, whatever the entry is called (a user can name an env anything: ``OPENAI_API_KEYS``, ``HF_AUTH``,
# ``MY_THING``).  Vendor prefixes: ``sk-`` / ``sk-ant-`` / ``sk-proj-`` / ``sk-or-v1-`` (with a digit, so ``task-...`` and ``sk-learn`` stay), ``hf_``,
# ``ghp_/gho_/ghu_/ghs_/ghr_``, ``github_pat_``, ``xox[abpr]-``, ``AKIA`` + 16.  Case sensitive on purpose (``hf_hub_cache`` is a name, ``hf_`` + 8 letters is a token).
_VENDOR = re.compile(r"(?<![A-Za-z0-9_])(?:sk-(?=[A-Za-z0-9_\-]*\d)[A-Za-z0-9_\-]{8,}|hf_[A-Za-z0-9]{8,}|gh[pousr]_[A-Za-z0-9]{16,}"
                     r"|github_pat_[A-Za-z0-9_]{16,}|xox[abpr]-[A-Za-z0-9\-]{8,}|AKIA[0-9A-Z]{16}(?![A-Za-z0-9]))")
_BEARER = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=\-]{8,}")
# ``scheme://user:pass@host`` -> ``scheme://user:<entfernt>@host``
_URLCRED = re.compile(r"(?i)\b([a-z][a-z0-9+.\-]*://[^\s/:@]*:)[^\s/@]+@")
# a long token-looking run right after ``=`` or ``:`` (env / flag lines): >= 32 of [A-Za-z0-9_-], no dot or slash (so model names and paths are not
# runs), see ``_looks_like_secret`` for what is spared (git SHAs, sha256 digests, UPPER_CASE names).
_LONGRUN = re.compile(r"(?<=[=:])([ \t]*[`\"']?)([A-Za-z0-9_\-]{32,})(?![A-Za-z0-9_\-./])")
# a JWT (three base64url parts, the header starts ``eyJ``): the dots keep it from being a run, so it has its own shape
_JWT = re.compile(r"(?<![A-Za-z0-9_\-])eyJ[A-Za-z0-9_\-]{4,}\.[A-Za-z0-9_\-]{4,}\.[A-Za-z0-9_\-]*")
# a long token with NO ``=`` / ``:`` before it (free text, a bare table cell): only the forms no name or hash has -- pure letters+digits (not a 40/64-hex
# SHA), or separators with upper AND lower case AND a digit.  A lowercase hyphenated name (``task-runner-big-name-0123456789``) stays.
_BARERUN = re.compile(r"(?<![A-Za-z0-9_\-./])[A-Za-z0-9_\-]{32,}(?![A-Za-z0-9_\-./])")


def _bare_secret(run: str) -> bool:
    if not _looks_like_secret(run):
        return False
    if run.isalnum():
        return True
    return any(c.isupper() for c in run) and any(c.islower() for c in run) and any(c.isdigit() for c in run)


_FULLRUN = re.compile(r"[A-Za-z0-9_\-]{32,}\Z")


def _looks_like_secret(run: str) -> bool:
    """A long run is a secret unless it is a git SHA / sha256 digest (40 or 64 hex), has no digit or no letter, or is an UPPER_CASE name."""
    if len(run) in (40, 64) and all(c in "0123456789abcdefABCDEF" for c in run):
        return False
    if not any(c.isdigit() for c in run) or not any(c.isalpha() for c in run):
        return False
    return not (run.upper() == run and "_" in run)


def shapes(text: str) -> str:
    """Cut secrets by the form of their value: vendor prefixes, ``Bearer <token>``, ``user:pass@`` in a URL, long token runs after ``=`` / ``:``."""
    text = _URLCRED.sub(lambda m: m.group(1) + "<entfernt>@", text or "")
    text = _BEARER.sub("Bearer <entfernt>", text)
    text = _VENDOR.sub("<entfernt>", text)
    text = _JWT.sub("<entfernt>", text)
    text = _LONGRUN.sub(lambda m: m.group(1) + ("<entfernt>" if _looks_like_secret(m.group(2)) else m.group(2)), text)
    return _BARERUN.sub(lambda m: "<entfernt>" if _bare_secret(m.group(0)) else m.group(0), text)


def text_for_issue(text: str) -> str:
    """Text that is pasted into a public issue: secret lines dropped whole, secret values (by name context AND by the form of the value) and host
    paths cut, line by line."""
    out = []
    for line in (text or "").split("\n"):
        c = clean(line)
        if c is not None:
            out.append(paths(shapes(c)))
    return "\n".join(out)


# the NAME of a flag, variable or env entry says it carries a secret: its value never goes into a pasted issue.  ``clean`` cannot see these
# (``HF_TOKEN=...`` has no word boundary before ``TOKEN``, and a table cell ``| HF_TOKEN | abc |`` has no ``=`` at all).  The name is split into
# its letter runs (``HF_TOKEN`` -> hf, token; ``--max-total-tokens`` -> max, total, tokens), so ``tokens`` / ``tokenizer`` are NOT secrets.
# ``token`` alone is a secret only where it is the NAME's last word (``HF_TOKEN``, ``--auth-token``, ``--token``) or it pairs with an
# access word (``hf-token-file``, ``--token-key``): the catalog is full of flags that merely MENTION a token (``--d-token-placement``,
# ``--uneven-token-vector``, ``SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION``) and their values are exactly what the Laufbericht shows.
# A name that ENDS in ``token`` but carries a token id or a boot marker is on the allow list ``_TOKEN_IS_NOT_SECRET`` (catalog:
# ``--fork-anchor-token``, ``--turn-anchor-token``, ``SGLANG_WEG2_LANE_COVERAGE_TOKEN``) or counts tokens (``..._PER_KI_TOKEN``, a ``per`` word anywhere,
# ``--bucket-time-to-first-token``).
_SECRET_WORDS = frozenset(("secret", "secrets", "password", "passwords", "passwd", "pwd", "passphrase", "credential", "credentials", "bearer",
                           "authorization", "pat", "apikey", "apikeys", "adminkey", "accesskey", "privatekey"))
# words that make a secret only as the LAST word of a name (``HF_AUTH``, ``DB_PASS``); mid-name they are ordinary (``--auth-backend``, ``--pass-through``)
_LAST_WORD_SECRET = frozenset(("auth", "pass", "pat"))
# ... except where the word before ``pass`` says it is a forward pass (catalog: ``SGLANG_WEG2_D_TWIN_PASS``)
_PASS_IS_NOT_SECRET = frozenset(("twin", "forward", "prefill", "decode", "warmup", "draft"))
_KEY_PREFIX = frozenset(("api", "admin", "access", "private", "auth", "ssh"))
# ``key`` as the LAST word of a name is a credential whatever precedes it (``OPENROUTER_KEY``, ``WANDB_KEY``, ``ANTHROPIC_KEY``): a user can set any
# env name, so a closed prefix list lets vendor keys through.  Allow list = catalog names whose ``key`` is a lookup key, not a credential
# (``SGLANG_LOG_DECODE_GRAPH_KEY``, ``SGLANG_WEG2_TOLD_PROBE_TREE_KEY``, ``SGLANG_HICACHE_BIGRAM_KEYS``, ``SGLANG_WEG2_MAMBA_STATE_KEYS``): the word
# before ``key`` / ``keys`` is ``graph`` / ``tree`` / ``bigram`` / ``state``.
_KEY_IS_NOT_SECRET = frozenset(("graph", "tree", "bigram", "state"))
_TOKEN_PREFIX = frozenset(("hf", "hub", "huggingface", "github", "gh", "gitlab", "auth", "access", "bearer", "api", "admin", "boot", "refresh",
                           "session", "bot", "slack", "pypi", "npm"))
_TOKEN_SUFFIX = frozenset(("key", "value", "secret", "string", "file"))
# (word before ``token``) -> the entry is a token ID / a run marker, not a credential
_TOKEN_IS_NOT_SECRET = frozenset(("anchor", "coverage", "first"))


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
    if any(x in _SECRET_WORDS for x in w) or any(a in _KEY_PREFIX and b == "key" for a, b in zip(w, w[1:])):
        return True
    if w and w[-1] in ("key", "keys") and (len(w) == 1 or w[-2] not in _KEY_IS_NOT_SECRET):      # the name ENDS in ``key`` / ``keys``
        return True
    if w and w[-1] in _LAST_WORD_SECRET and not (w[-1] == "pass" and len(w) > 1 and w[-2] in _PASS_IS_NOT_SECRET):   # ``HF_AUTH``, ``DB_PASS``
        return True
    for i, x in enumerate(w):
        if x not in ("token", "tokens"):
            continue
        prev, nxt = (w[i - 1] if i else None), (w[i + 1] if i + 1 < len(w) else None)
        # ``tokens`` after a bearer-ish word is a credential list (``HF_TOKENS``); ``session`` / ``boot`` / ``bot`` + ``tokens`` is a COUNT
        # (catalog: ``--kv-session-offload-budget-session-tokens``)
        if (x == "token" and prev in _TOKEN_PREFIX) or (x == "tokens" and prev in _TOKEN_PREFIX and prev not in ("session", "boot", "bot")):
            return True
        if x == "token" and nxt in _TOKEN_SUFFIX:
            return True
        # the name ENDS in ``token`` (or is just ``token``); a bare ``..._tokens`` counts tokens (``--max-total-tokens``) and is no secret
        if x == "token" and nxt is None and prev not in _TOKEN_IS_NOT_SECRET and "per" not in w:
            return True
    return False


def value_for_issue(name: str, value) -> str:
    """The value of a named entry for a pasted issue: ``<entfernt>`` when the name says secret, else the value with secrets and host paths cut."""
    if secret_name(name):
        return "<entfernt>"
    text = str(value if value is not None else "")
    bare = text.strip().strip("`\"'")
    # a whole value that is one long token run is a secret whatever the entry is called (``MY_THING`` = ``Zq8v...``): the ``=`` / ``:`` the shape layer
    # looks for is not there when the value is cut out of its line and set into a table cell
    if _FULLRUN.match(bare) and _looks_like_secret(bare):
        return "<entfernt>"
    return text_for_issue(text)
