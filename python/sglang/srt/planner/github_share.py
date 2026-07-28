# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""GitHub results-sharing core for the dashboard (design #152).

Shares MEASURED results / quality shots as one GitHub issue per user:

  * :func:`build_report` renders the exact markdown to post -- the EXACT
    start command (argv + env), the full measured metrics (throughput /
    energy / per-card), and, when present, a quality shot (the chess SVG,
    its verdict, and token counts). The caller shows this markdown as a
    PREVIEW; nothing is sent by this function.
  * :func:`submit` posts it via the GitHub REST API -- ONE issue per user,
    UPDATED IN PLACE on re-submit. The user's existing issue is found by a
    stable HTML-comment marker in the body among issues CREATED BY the
    authenticated user, then PATCHed; otherwise a new issue is created.

TOKEN (PAT) HANDLING -- the token is SENSITIVE:

  * accepted per-call only; NEVER stored on disk, in module state, or in any
    returned value,
  * NEVER logged; the default transport puts it only in the Authorization
    header of the outgoing request,
  * REDACTED from every error message: all failures are re-raised as
    :class:`GitHubShareError` whose text has passed :func:`redact`.

EXPLICIT-CONSENT DISCIPLINE -- posting sends data to an EXTERNAL service:
:func:`submit` refuses to do anything unless the caller passes
``confirmed=True``, which the UI may only set after the user approved the
previewed report. There is no auto-submit path in this module.

The report itself deliberately keeps the start command EXACT (argv + env,
that is the point of sharing a reproducible result), with ONE exception:
env values whose NAME looks credential-like (TOKEN / SECRET / KEY /
PASSWORD / PAT) are replaced by ``<redacted>`` -- a launch env sometimes
carries HF_TOKEN and that must never reach a public issue. Suffix matching
keeps non-credential names like SGLANG_UNEVEN_TOKEN_VECTOR exact.

HONEST NOTE: the create/update flow is exercised against a MOCKED GitHub API
in the unit tests; it has not been run against api.github.com from this
module yet (needs a real PAT + network, deferred to live validation).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Tuple

__all__ = [
    "DEFAULT_REPO",
    "MARKER",
    "GitHubShareError",
    "build_report",
    "find_existing_comment",
    "find_existing_issue",
    "redact",
    "submit",
    "upsert_comment",
]

#: Default target repository for shared results.
DEFAULT_REPO = "noonghunna/club-3090"

#: Stable marker embedded in the issue body. Re-submits look for THIS string
#: among issues created by the authenticated user -- that pair (marker +
#: creator) implements "one issue per user, update in place". Do not change
#: the string: existing issues would stop being found.
MARKER = "<!-- htsglang-share v1 -->"

#: GitHub REST API root (overridable for tests / GHE).
API_ROOT = "https://api.github.com"

#: Env-var name SUFFIXES whose VALUES are redacted from the report. Suffix
#: (not substring) matching on purpose: HF_TOKEN / GITHUB_TOKEN / API_KEY are
#: credentials, but e.g. SGLANG_UNEVEN_TOKEN_VECTOR is a tuning knob that
#: must stay EXACT in the shared start command.
_SECRET_ENV_SUFFIXES = ("TOKEN", "SECRET", "PASSWORD", "KEY", "PAT")


class GitHubShareError(Exception):
    """Raised for every failure in this module. The message is guaranteed to
    have passed :func:`redact` for the token in play."""


def redact(text: str, token: Optional[str]) -> str:
    """Remove every occurrence of ``token`` from ``text``. Used on every
    error path so a PAT can never leak through an exception message, HTTP
    error body, or urllib repr."""
    if not text:
        return text or ""
    if token:
        text = text.replace(token, "<redacted-token>")
    return text


def _redact_env_value(name: str, value: str) -> str:
    upper = (name or "").upper()
    if any(upper.endswith(sfx) for sfx in _SECRET_ENV_SUFFIXES):
        return "<redacted>"
    return value


# ===========================================================================
# Report rendering (preview-first; pure, no network).
# ===========================================================================
def _fmt_num(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def _metrics_lines(metrics: Dict[str, Any]) -> List[str]:
    """Render the measured metrics block. Known keys get friendly lines;
    everything else is emitted verbatim as ``- key: value`` so no measured
    number is silently dropped."""
    lines: List[str] = []
    known = {
        "decode_tok_s": "Decode",
        "prefill_tok_s": "Prefill",
        "ttft_ms": "TTFT",
        "j_per_decode_token": "Energy decode",
        "j_per_prefill_token": "Energy prefill",
    }
    units = {
        "decode_tok_s": "tok/s",
        "prefill_tok_s": "tok/s",
        "ttft_ms": "ms",
        "j_per_decode_token": "J/token",
        "j_per_prefill_token": "J/token",
    }
    for key, label in known.items():
        if metrics.get(key) is not None:
            lines.append(f"- {label}: {_fmt_num(metrics[key])} {units[key]}")
    per_card = metrics.get("per_card")
    if isinstance(per_card, dict) and per_card:
        lines.append("- Per-card:")
        lines.append("")
        lines.append("| Card | " + " | ".join(
            k for k in _per_card_columns(per_card)) + " |")
        lines.append("|---" * (1 + len(_per_card_columns(per_card))) + "|")
        for card, vals in per_card.items():
            row = [str(card)]
            for col in _per_card_columns(per_card):
                v = (vals or {}).get(col)
                row.append(_fmt_num(v) if v is not None else "-")
            lines.append("| " + " | ".join(row) + " |")
    handled = set(known) | {"per_card"}
    for key in sorted(metrics):
        if key in handled or metrics[key] is None:
            continue
        lines.append(f"- {key}: {_fmt_num(metrics[key])}")
    return lines


def _per_card_columns(per_card: Dict[str, Any]) -> List[str]:
    cols: List[str] = []
    for vals in per_card.values():
        for k in (vals or {}):
            if k not in cols:
                cols.append(k)
    return cols


def build_report(payload: Dict[str, Any]) -> str:
    """Render the EXACT markdown that would be posted. Pure function, sends
    nothing -- the UI shows this as the preview the user must approve before
    :func:`submit` may be called with ``confirmed=True``.

    ``payload`` keys (all optional except command/metrics being the point):

      * ``model``: served model name (heading)
      * ``hardware``: short hardware summary string (heading)
      * ``command``: ``{"argv": [...], "env": {NAME: VALUE, ...}}`` -- the
        EXACT start command. argv is emitted verbatim; env values with
        credential-looking names are redacted (see module docstring).
      * ``metrics``: measured numbers -- ``decode_tok_s``, ``prefill_tok_s``,
        ``ttft_ms``, ``j_per_decode_token``, ``j_per_prefill_token``,
        ``per_card`` ({card: {metric: value}}), plus any extra keys
        (emitted verbatim).
      * ``quality``: quality shot -- ``{"svg": ..., "verdict": ...,
        "tokens": {"prompt":, "completion":, "total":}, "report": ...}``
        (the chess-SVG benchmark shape from the Quality tab).
      * ``bench_results``: optional list of #151 per-test result dicts;
        rendered as a compact status table.
      * ``notes``: free-text appendix.
    """
    model = payload.get("model") or "unknown model"
    hardware = payload.get("hardware")
    md: List[str] = []
    title_suffix = f" on {hardware}" if hardware else ""
    md.append(f"## htsglang measured result -- {model}{title_suffix}")
    md.append("")

    command = payload.get("command") or {}
    argv = command.get("argv") or []
    env = command.get("env") or {}
    md.append("### Start command (exact)")
    md.append("```")
    for name in env:
        md.append(f"{name}={_redact_env_value(name, str(env[name]))}")
    if argv:
        md.append(" ".join(str(a) for a in argv))
    md.append("```")
    md.append("")

    metrics = payload.get("metrics") or {}
    md.append("### Measured metrics")
    metric_lines = _metrics_lines(metrics)
    md.extend(metric_lines or ["- (none supplied)"])
    md.append("")

    bench = payload.get("bench_results")
    if bench:
        md.append("### Behavioral bench (#151)")
        md.append("| # | Test | Status | Metric |")
        md.append("|---|---|---|---|")
        for r in bench:
            metric = r.get("metric") or {}
            mtxt = ""
            if metric.get("name") and metric.get("value") is not None:
                mtxt = f"{metric['name']}={metric['value']}"
                if metric.get("unit"):
                    mtxt += f" {metric['unit']}"
            md.append(
                f"| {r.get('test_id')} | {r.get('label', '')} | "
                f"{r.get('status', '')} | {mtxt} |"
            )
        md.append("")

    quality = payload.get("quality")
    if quality:
        md.append("### Quality shot (chess-SVG benchmark)")
        if quality.get("verdict"):
            md.append(f"- Verdict: **{quality['verdict']}**")
        tokens = quality.get("tokens") or {}
        if tokens:
            parts = [
                f"{k}: {tokens[k]}"
                for k in ("prompt", "completion", "total")
                if tokens.get(k) is not None
            ]
            if parts:
                md.append("- Tokens: " + ", ".join(parts))
        if quality.get("report"):
            md.append(f"- Grader report: {quality['report']}")
        svg = quality.get("svg")
        if svg:
            md.append("")
            md.append("<details><summary>Generated SVG</summary>")
            md.append("")
            md.append("```svg")
            md.append(svg)
            md.append("```")
            md.append("")
            md.append("</details>")
        md.append("")

    notes = payload.get("notes")
    if notes:
        md.append("### Notes")
        md.append(str(notes))
        md.append("")

    md.append(MARKER)
    return "\n".join(md)


# ===========================================================================
# GitHub REST transport (injectable; tests use a fake).
# ===========================================================================
def _default_api(
    method: str,
    url: str,
    token: str,
    body: Optional[dict] = None,
    timeout: float = 30.0,
) -> Tuple[int, Any]:
    """One authenticated GitHub REST call -> ``(status, parsed_json)``.
    The token appears ONLY in the Authorization header. Errors are raised as
    :class:`GitHubShareError` with the token redacted from the message."""
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "htsglang-planner",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            text = r.read().decode("utf-8", errors="replace")
            return r.getcode() or 200, (json.loads(text) if text else {})
    except urllib.error.HTTPError as e:
        try:
            err = e.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            err = ""
        raise GitHubShareError(
            redact(f"GitHub API {method} {url} failed: HTTP {e.code} {err}",
                   token)
        ) from None
    except Exception as e:
        raise GitHubShareError(
            redact(f"GitHub API {method} {url} failed: {e}", token)
        ) from None


def _authenticated_login(token: str, api: Callable[..., Tuple[int, Any]]) -> str:
    status, data = api("GET", f"{API_ROOT}/user", token)
    if status != 200 or not isinstance(data, dict) or not data.get("login"):
        raise GitHubShareError(
            redact(f"could not resolve the token's GitHub user "
                   f"(HTTP {status})", token)
        )
    return data["login"]


def find_existing_issue(
    token: str,
    repo: str = DEFAULT_REPO,
    *,
    marker: str = MARKER,
    api: Optional[Callable[..., Tuple[int, Any]]] = None,
) -> Optional[dict]:
    """Find THIS user's share issue in ``repo``: the first issue CREATED BY
    the authenticated user whose body contains ``marker``. Returns the issue
    dict (``number`` / ``html_url`` / ...) or None.

    ``marker`` is a parameter rather than the constant so a second KIND of
    report (the #271 rig artifact) can have its own issue with the same
    one-per-user, update-in-place semantics, instead of overwriting the
    measured-results issue this module was written for. Default unchanged.
    """
    api = api or _default_api
    login = _authenticated_login(token, api)
    status, issues = api(
        "GET",
        f"{API_ROOT}/repos/{repo}/issues"
        f"?state=all&creator={login}&per_page=100",
        token,
    )
    if status != 200 or not isinstance(issues, list):
        raise GitHubShareError(
            redact(f"could not list issues of {repo} (HTTP {status})", token)
        )
    for issue in issues:
        if isinstance(issue, dict) and marker in (issue.get("body") or ""):
            return issue
    return None


def find_existing_comment(
    token: str,
    repo: str,
    issue_number: int,
    marker: str,
    *,
    api: Optional[Callable[..., Tuple[int, Any]]] = None,
) -> Optional[dict]:
    """The first comment on ``issue_number`` whose body contains ``marker``.

    Comments are how one issue holds SEVERAL rigs (#271): the body is an
    index, and each rig fingerprint owns one comment that is updated in
    place. Returns the comment dict (``id`` / ``html_url`` / ``body``) or
    None.
    """
    api = api or _default_api
    status, comments = api(
        "GET",
        f"{API_ROOT}/repos/{repo}/issues/{issue_number}/comments?per_page=100",
        token,
    )
    if status != 200 or not isinstance(comments, list):
        raise GitHubShareError(
            redact(
                f"could not list comments of {repo}#{issue_number} "
                f"(HTTP {status})", token)
        )
    for c in comments:
        if isinstance(c, dict) and marker in (c.get("body") or ""):
            return c
    return None


def upsert_comment(
    token: str,
    repo: str,
    issue_number: int,
    marker: str,
    body: str,
    *,
    api: Optional[Callable[..., Tuple[int, Any]]] = None,
) -> dict:
    """Create ``body`` as a comment, or PATCH the one already carrying
    ``marker``. Returns ``{"action", "comment_id", "url"}``.

    Consent is the CALLER's to obtain: this is transport. Nothing here is
    reachable except through :func:`submit`, which refuses without
    ``confirmed=True``.
    """
    api = api or _default_api
    if marker not in body:
        body = body + "\n\n" + marker
    found = find_existing_comment(token, repo, issue_number, marker, api=api)
    if found is not None:
        status, data = api(
            "PATCH",
            f"{API_ROOT}/repos/{repo}/issues/comments/{found.get('id')}",
            token, {"body": body})
        action = "updated"
    else:
        status, data = api(
            "POST",
            f"{API_ROOT}/repos/{repo}/issues/{issue_number}/comments",
            token, {"body": body})
        action = "created"
    if status not in (200, 201) or not isinstance(data, dict):
        raise GitHubShareError(
            redact(f"comment {action[:-1]}e failed: HTTP {status} from GitHub",
                   token))
    return {"action": action, "comment_id": data.get("id"),
            "url": data.get("html_url")}


def submit(
    report: str,
    token: str,
    repo: str = DEFAULT_REPO,
    existing_issue: Optional[int] = None,
    *,
    confirmed: bool = False,
    title: str = "htsglang measured results",
    marker: str = MARKER,
    api: Optional[Callable[..., Tuple[int, Any]]] = None,
) -> dict:
    """Create-or-update the user's share issue with ``report``.

    POSTING SENDS DATA TO AN EXTERNAL SERVICE (github.com). The caller MUST
    have shown the report as a preview and obtained the user's explicit
    approval; only then may it pass ``confirmed=True``. Without it this
    function raises and performs NO network call whatsoever.

    Flow: ``existing_issue`` (a number) wins when given; otherwise the
    user's issue is located via :func:`find_existing_issue` (marker +
    creator). Found -> PATCH in place (update); not found -> POST a new
    issue. Returns ``{"action": "updated"|"created", "number": n,
    "url": html_url}`` -- never the token.
    """
    if not confirmed:
        raise GitHubShareError(
            "refusing to submit: the caller did not confirm. Posting shares "
            "data with an external service (GitHub); show the report "
            "preview, get the user's explicit approval, then call "
            "submit(..., confirmed=True)."
        )
    if not token:
        raise GitHubShareError("no GitHub token given")
    api = api or _default_api

    body = report if marker in report else report + "\n\n" + marker

    number = existing_issue
    if number is None:
        found = find_existing_issue(token, repo, marker=marker, api=api)
        if found is not None:
            number = found.get("number")

    try:
        if number is not None:
            status, data = api(
                "PATCH",
                f"{API_ROOT}/repos/{repo}/issues/{number}",
                token,
                {"title": title, "body": body},
            )
            action = "updated"
        else:
            status, data = api(
                "POST",
                f"{API_ROOT}/repos/{repo}/issues",
                token,
                {"title": title, "body": body},
            )
            action = "created"
    except GitHubShareError:
        raise
    except Exception as e:
        raise GitHubShareError(redact(f"submit failed: {e}", token)) from None

    if status not in (200, 201) or not isinstance(data, dict):
        raise GitHubShareError(
            redact(f"submit failed: HTTP {status} from GitHub", token)
        )
    return {
        "action": action,
        "number": data.get("number", number),
        "url": data.get("html_url"),
    }
