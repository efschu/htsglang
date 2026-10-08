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

import glob
import os
import posixpath
import re
from typing import List, Optional

from . import names as N

# a line matching this is dropped whole
# F0-B: ``WEG2-GROUP-ENV`` (the whole group environment) is stamped as ``PDFLIP-GROUP-ENV`` by a renamed launcher; a reader that
# only knew the old spelling would PUBLISH that line.  Both spellings, so the drop is the same in both generations.
DROP_LINE = N.tolerant_compile(r"ADMIN-KEY|admin-api-key|\.adminkey\b|auth=bearer|Authorization:|WEG2-GROUP-ENV|GITHUB_PAT|openrouter\.key",
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
    text = _VALUE.sub(lambda m: m.group(1) + m.group(2) + "<redacted>", text)
    return _KEYFILE.sub("<key file redacted>", text)


# the door: a serialized answer must not contain any of these
_DOOR = N.tolerant_compile(r"ADMIN-KEY|admin-api-key|\.adminkey|auth=bearer|GITHUB_PAT|openrouter\.key|WEG2-GROUP-ENV", re.IGNORECASE)


def guard(body: str) -> str:
    return _DOOR.sub("<redacted>", body)


# an absolute path into the host (or the container) filesystem: nothing about the box's layout leaves in a pasted issue
_HOSTPATH = re.compile(r"(?<![\w.:/-])/(?:home|root|spinning|opt|var|usr|tmp|mnt|srv|etc|data|run|media|nix|proc|sys|dev)(?:/[^\s|`,;)\]\"'<>]*)*")
_SYSROOTS = ("home", "root", "spinning", "opt", "var", "usr", "tmp", "mnt", "srv", "etc", "data", "run", "media", "nix", "proc", "sys", "dev")


# Every OTHER absolute path (``/nvme/hf``, ``/workspace/models/X``, ``/scratch``) and every ``~/`` / ``$HOME/`` path is a host path too: the Laufbericht
# shows no path outside the container.  Exempt: the mount points the image itself owns (``CONTAINER_MOUNTS``, ``/app`` = the baked tree,
# ``/models-cache`` = the model mount of the run example).  Everything else shrinks to ``<hostpfad>/<last segment>``; system roots keep the stricter
# ``<path redacted>``.  A path is NORMALISED before it is judged (``/app/../../root/x`` is ``/root/x``), a ``file://`` prefix is stripped, and a quoted
# path may hold spaces.
CONTAINER_MOUNTS = ("/app", "/api", "/models-cache")               # ``/api/...`` = a dashboard URL route (the report names them), not a directory
_PATHTAIL = r"[^\s|`,;)\]\"'<>]*"
_ANYPATH = re.compile(r"(?<![\w.:/>~<$-])/(?![/\s])" + _PATHTAIL)
_HOMEPATH = re.compile(r"(?<![\w.:/>~<$-])(?:~|\$\{HOME\}|\$HOME)(?:/" + _PATHTAIL + r")+")
# ``file:///nvme/x`` / ``file://host/x`` / ``file:/x``: the scheme (and an authority) is dropped, the path behind it is judged like any other
_FILEURL = re.compile(r"(?i)\bfile:(?://[^/\s]*)?(?=/)")
# a quoted path may hold spaces: ``"/nvme/my models/Qwen"`` (the quote has to open at the path's first character)
_QUOTEDPATH = re.compile(r"([\"'`])((?:/|~/|\$HOME/|\$\{HOME\}/)[^\"'`\n]*)\1")


def _normal(p: str) -> str:
    """``p`` with ``.`` / ``..`` / ``//`` folded away (``posixpath.normpath``); ``~`` / ``$HOME`` roots stay as they are."""
    if "/." not in p and "//" not in p:
        return p
    head = ""
    for h in ("${HOME}", "$HOME", "~"):
        if p.startswith(h + "/"):
            head, p = h, p[len(h):]
            break
    q = posixpath.normpath(p) if p else p
    if q.startswith("//"):
        q = "/" + q.lstrip("/")
    return head + ("" if (head and q == "/") else q)


def _is_mount(p: str) -> bool:
    return any(p == c or p.startswith(c + "/") for c in CONTAINER_MOUNTS)


def _shrink(m) -> str:
    return _shrink_path(m.group(0))


def _shrink_path(p: str) -> str:
    p = _normal(p)
    if _is_mount(p):
        return p
    first = p.lstrip("/").split("/", 1)[0]
    if p.startswith("/") and first in _SYSROOTS:
        return "<path redacted>"
    last = p.rstrip("/").rsplit("/", 1)[-1].strip(".")
    return "<hostpfad>/" + last if last and last not in ("~", "$HOME", "${HOME}") else "<hostpfad>"


def _quoted(m) -> str:
    return m.group(1) + _shrink_path(m.group(2)) + m.group(1)


def paths(text: str) -> str:
    """Replace absolute host paths: the system roots (``/root/...``, ``/spinning/...``, ``/var/lib/...``) by ``<path redacted>``, any other absolute
    path and any ``~/...`` path by ``<hostpfad>/<last segment>`` (``CONTAINER_MOUNTS``: the image's ``/app``, ``/models-cache`` and the URL routes
    ``/api/...`` stay).  ``..`` segments are folded first, ``file://`` is stripped, a quoted path may hold spaces."""
    text = _FILEURL.sub("", text or "")
    text = _QUOTEDPATH.sub(_quoted, text)
    text = _HOSTPATH.sub(_shrink, text)
    return _ANYPATH.sub(_shrink, _HOMEPATH.sub(_shrink, text))


# a secret recognised by the FORM of its value, whatever the entry is called (a user can name an env anything: ``OPENAI_API_KEYS``, ``HF_AUTH``,
# ``MY_THING``).  Vendor prefixes: ``sk-`` / ``sk-ant-`` / ``sk-proj-`` / ``sk-or-v1-`` (with a digit, so ``task-...`` and ``sk-learn`` stay), ``hf_``,
# ``ghp_/gho_/ghu_/ghs_/ghr_``, ``github_pat_``, ``xox[abpr]-``, ``AKIA`` + 16.  Case sensitive on purpose (``hf_hub_cache`` is a name, ``hf_`` + 8 letters is a token).
_VENDOR = re.compile(r"(?<![A-Za-z0-9_])(?:sk-(?=[A-Za-z0-9_\-]*\d)[A-Za-z0-9_\-]{8,}|hf_[A-Za-z0-9]{8,}|gh[pousr]_[A-Za-z0-9]{16,}"
                     r"|github_pat_[A-Za-z0-9_]{16,}|xox[abpr]-[A-Za-z0-9\-]{8,}|AKIA[0-9A-Z]{16}(?![A-Za-z0-9]))")
_BEARER = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=\-]{8,}")
# ``scheme://user:pass@host`` -> ``scheme://user:<redacted>@host``
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


# standard-alphabet base64 (``/`` and ``+`` inside, ``=`` padding at the end: an AWS secret access key, an Azure storage key): >= 30 characters, upper AND
# lower case AND a digit (a path is no run: a run never STARTS behind ``/``, ``.`` or ``-``, and a lowercase path or a name stays), see ``_b64_secret``.
_B64RUN = re.compile(r"(?<![A-Za-z0-9+/_.\-])[A-Za-z0-9+/]{30,}={0,2}(?![A-Za-z0-9+/=_\-])")
# dot-separated tokens WITHOUT an ``eyJ`` header (a Discord bot token ``<id>.<time>.<hmac>``): see ``_dotted_secret`` for what is spared
_DOTTED = re.compile(r"(?<![A-Za-z0-9_\-.])[A-Za-z0-9_\-]+(?:\.[A-Za-z0-9_\-]+)+(?![A-Za-z0-9_\-])")
_FULLRUN = re.compile(r"[A-Za-z0-9_\-]{32,}\Z")


# a CamelCase identifier the LAUNCHER / weg2 source really contains (the exception class of a refusal, ``Weg2TpOperatingPointInfeasible``, or any ``class X``
# of ``srt/weg2``): such a name is a word of the refusal text, not a secret, and stays readable in the run report.  The exemption is a LIST, never a shape:
# a token-like run of capital+lower-case groups (``AbcdEfghIjklMnopQrstUvwxYz12Ab``) is no source name and is cut like any other base64-looking run.
# Read once from the source tree (``HWPROFIL_TREE`` / ``KARTENPLAN_TREE`` / the tree candidates of ``hwprofil``): every ``Weg2<Word>...`` identifier of
# ``srt/weg2/*.py`` and ``srt/flip_*.py`` (the launcher prints some as text, ``Weg2TpOperatingPointInfeasible`` is no class) plus every ``class X``
# of those files.  No tree found -> only the built-in names below (the ones the run report is known to quote), everything else is cut: the safe side.
# F0-B: a renamed tree spells these ``PdFlip<Word>`` (``Weg2Flip<X>`` -> ``PdFlip<X>``, ``Weg2<X>`` -> ``PdFlip<X>``, rename tool rule), so the
# built-in list carries both spellings (derived, see below) and the tree scan reads either layout.
# The set is DERIVED from one spelling through ``names`` (``marker_variants``), never written out twice: a literal old spelling would be rewritten by the
# rename tool (F0-F) and the old-evidence names would drop out of a renamed dashboard.
_BUILTIN_BASES = tuple(N.CAMEL_TOKENS[0] + _w for _w in ("TpOperatingPointInfeasible", "XchgResidencyUnarmable", "XchgSemaphoreNotRearmed",
                                                       "FlipPeerLegAborted", "DualCompactBreach"))
_KNOWN_IDENT_BUILTIN = frozenset(v for _b in _BUILTIN_BASES for v in N.marker_variants(_b))
_IDENT_IN_SOURCE = re.compile(r"\b(?:%s|%s)[A-Z][A-Za-z0-9]+\b|^class ([A-Z][A-Za-z0-9]+)" % N.CAMEL_TOKENS, re.M)
_known_cache: Optional[frozenset] = None


def _tree_candidates() -> List[str]:
    out = [os.environ.get("HWPROFIL_TREE"), os.environ.get("KARTENPLAN_TREE")]
    try:
        from . import hwprofil
        out += list(hwprofil.TREE_CANDIDATES)
    except Exception:       # noqa: BLE001 -- no candidate list: the environment variables alone
        pass
    return [t for t in out if t]


def _scan_known_idents(tree: str) -> frozenset:
    names = set()
    subs = []
    for pkg, sub in N.PKG_PAIRS:
        subs += [(pkg + "/srt/" + sub, "*.py"), (pkg + "/srt", "flip_*.py")]
    for sub, pat in subs:
        for f in glob.glob(os.path.join(tree, sub, pat)):
            try:
                with open(f, encoding="utf-8", errors="replace") as fh:
                    for m in _IDENT_IN_SOURCE.finditer(fh.read()):
                        names.add(m.group(1) or m.group(0))
            except OSError:
                continue
    return frozenset(names)


def known_idents() -> frozenset:
    """The identifiers the exemption lets through (built-ins + the source tree).  A scan that found no tree is not cached (the next call looks again)."""
    global _known_cache
    if _known_cache is not None:
        return _known_cache
    for t in _tree_candidates():
        if any(os.path.isdir(os.path.join(t, pkg, "srt", sub)) for pkg, sub in N.PKG_PAIRS):
            _known_cache = _KNOWN_IDENT_BUILTIN | _scan_known_idents(t)
            return _known_cache
    return _KNOWN_IDENT_BUILTIN


def _b64_secret(run: str) -> bool:
    body = run.rstrip("=")
    if run == body and body in known_idents():      # ``Weg2TpOperatingPointInfeasible``: a name of the launcher source, not a key
        return False
    return (not body.startswith("/") and len(body) >= 30 and _looks_like_secret(body)
            and any(c.isupper() for c in body) and any(c.islower() for c in body) and any(c.isdigit() for c in body))


def _dotted_secret(tok: str) -> bool:
    """A dot-separated token is a secret when >= 2 of its parts are >= 16 characters long and the whole mixes upper case, lower case and digits (a
    host name or a file name is lower case or has one long part: ``registry.example-company-internal.com`` / ``Qwen3.6-27B-AWQ.gguf`` stay)."""
    return (sum(1 for x in tok.split(".") if len(x) >= 16) >= 2 and any(c.isupper() for c in tok) and any(c.islower() for c in tok)
            and any(c.isdigit() for c in tok))


def _looks_like_secret(run: str) -> bool:
    """A long run is a secret unless it is a git SHA / sha256 digest (40 or 64 hex), has no digit or no letter, or is an UPPER_CASE name."""
    if len(run) in (40, 64) and all(c in "0123456789abcdefABCDEF" for c in run):
        return False
    if not any(c.isdigit() for c in run) or not any(c.isalpha() for c in run):
        return False
    return not (run.upper() == run and "_" in run)


def shapes(text: str) -> str:
    """Cut secrets by the form of their value: vendor prefixes, ``Bearer <token>``, ``user:pass@`` in a URL, JWT, dot-separated tokens, standard
    base64 (``/`` ``+`` ``=``), long token runs."""
    text = _URLCRED.sub(lambda m: m.group(1) + "<redacted>@", text or "")
    text = _BEARER.sub("Bearer <redacted>", text)
    text = _VENDOR.sub("<redacted>", text)
    text = _JWT.sub("<redacted>", text)
    text = _DOTTED.sub(lambda m: "<redacted>" if _dotted_secret(m.group(0)) else m.group(0), text)
    text = _B64RUN.sub(lambda m: "<redacted>" if _b64_secret(m.group(0)) else m.group(0), text)
    text = _LONGRUN.sub(lambda m: m.group(1) + ("<redacted>" if _looks_like_secret(m.group(2)) else m.group(2)), text)
    return _BARERUN.sub(lambda m: "<redacted>" if _bare_secret(m.group(0)) else m.group(0), text)


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


#: what a user-set value of a key the catalog does not know becomes (the key name stays visible)
HIDDEN_UNKNOWN = "<value hidden: unknown key>"


def bare_key(name) -> str:
    """The catalog name of a profile key: ``flag:--p-bs`` -> ``--p-bs``, ``env:P:HF_HOME`` -> ``HF_HOME``, ``var:PROFILE_NAME`` -> ``PROFILE_NAME``
    (a row's ``name`` is already bare); ``extra:P:--x`` -> ``--x``."""
    n = str(name or "").strip()
    for pre in ("flag:", "var:"):
        if n.startswith(pre):
            return n[len(pre):]
    for pre in ("env:", "extra:"):
        if n.startswith(pre):
            parts = n.split(":", 2)
            return parts[2] if len(parts) == 3 else n
    return n


def value_for_issue(name: str, value, known=None) -> str:
    """The value of a named entry for a pasted issue.  STRUCTURAL rule (allow, not block): a value is shown only for a key the catalog knows
    (``known`` = the catalog's flag / env / variable names) and that is no secret by name; ``<redacted>`` when the name says secret, ``HIDDEN_UNKNOWN``
    for every other key a user set (the key itself stays in the table).  ``known=None`` knows nothing, so nothing is shown (closed by default).
    What is shown still passes the value-shape layer (``text_for_issue``: vendor prefixes, base64, JWT, runs, host paths)."""
    key = bare_key(name)
    if secret_name(key):
        return "<redacted>"
    text = str(value if value is not None else "")
    if not text.strip():
        return text
    if known is None or key not in known:
        return HIDDEN_UNKNOWN
    bare = text.strip().strip("`\"'")
    # a whole value that is one long token run is a secret whatever the entry is called: the ``=`` / ``:`` the shape layer looks for is not there
    # when the value is cut out of its line and set into a table cell
    if _FULLRUN.match(bare) and _looks_like_secret(bare):
        return "<redacted>"
    return text_for_issue(text)
