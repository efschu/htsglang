"""Compose a shareable report and, when a target is configured, post it.

Benchmark results, system details and energy efficiency, bundled into the
selections that are actually worth sharing, previewed as Markdown, and only
then sent.

THIS MODULE CREATES NOTHING BY DEFAULT
--------------------------------------
GitHub Discussions are the destination, but no discussion is created here and
none is posted to unless a target has been configured explicitly. With no
target, :func:`submit` returns the preview and says "no target configured".
That is a deliberate gate, not an unfinished path: the send is implemented
and tested, it is simply not armed. Nothing about running this module
produces a public artefact.

WHY GRAPHQL, AND WHY BY HAND
----------------------------
The REST v3 API that ``github_share.py`` uses does not cover Discussions;
they exist only in GraphQL. A GraphQL request is one POST of
``{"query": ..., "variables": ...}`` to a single endpoint, which is what
``urllib`` already does two modules over -- a client library would be larger
than the code it replaced and would be the first runtime dependency this
package has. So: same ``urllib``, same header handling, same redaction as
``github_share``.

TOKEN HANDLING
--------------
The token is read from a FILE whose path comes from the environment
(:data:`PAT_FILE_ENV`), never from a literal in this repository, never from a
command line where it would land in a shell history, and never from a URL
where it would land in a proxy log. It appears in exactly one place, the
``Authorization`` header. Every error path runs through
:func:`github_share.redact` so a token cannot escape in an exception message.

REDACTION
---------
System details always pass through :mod:`planner.scrub` before they reach the
Markdown. Card models and driver versions are the point of sharing them; host
names, IP addresses and filesystem paths are not, and must not leave the
machine. :func:`build_markdown` has no path that skips the scrubber.
"""

from __future__ import annotations

import dataclasses
import json
import os
import urllib.request
from typing import Callable, Dict, List, Optional

from sglang.srt.planner import github_share, scrub

__all__ = [
    "PAT_FILE_ENV",
    "TARGET_ENV",
    "Bundle",
    "BUNDLES",
    "ENERGY_GROUPS",
    "MARKER",
    "build_markdown",
    "read_token",
    "configured_target",
    "preview",
    "submit",
    "DiscussionError",
]

#: Path to a file holding the PAT. The path is configuration; the token never
#: appears in this repository, in a command line or in a URL.
PAT_FILE_ENV = "HTSGLANG_GITHUB_PAT_FILE"

#: The discussion to post into, as ``owner/repo#number`` or a node id. Absent
#: means the send is not armed -- see the module docstring.
TARGET_ENV = "HTSGLANG_DISCUSSION_TARGET"

#: Stable body marker, so a re-submit updates the previous comment instead of
#: adding another one. Same mechanism ``github_share`` uses for issues.
MARKER = "<!-- htsglang-discussion v1 -->"

API_URL = "https://api.github.com/graphql"


class DiscussionError(RuntimeError):
    """Raised with a message that has already been token-redacted."""


# ---------------------------------------------------------------------------
# Bundles
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Bundle:
    """One selectable package of sections.

    ``sections`` names what goes in. The point of offering several is that
    the useful shares are genuinely different documents: a bench table alone
    is a comparison, the same table with the system behind it is a
    reproduction, and adding energy makes it an efficiency claim that needs
    the hardware to be meaningful.
    """

    key: str
    label: str
    sections: tuple
    note: str = ""


BUNDLES: Dict[str, Bundle] = {
    "bench": Bundle(
        key="bench",
        label="Benchmark results only",
        sections=("bench",),
        note="The measurements, with no hardware context. Useful when the "
        "rig is already known to the reader.",
    ),
    "bench_system": Bundle(
        key="bench_system",
        label="Benchmark + system",
        sections=("bench", "system"),
        note="Enough to reproduce: what was measured and what it ran on.",
    ),
    "bench_system_energy": Bundle(
        key="bench_system_energy",
        label="Benchmark + system + energy",
        sections=("bench", "system", "energy"),
        note="Adds efficiency. Energy figures only mean something next to "
        "the cards that produced them, so this bundle always carries the "
        "system section.",
    ),
    "energy_system": Bundle(
        key="energy_system",
        label="Energy + system",
        sections=("system", "energy"),
        note="An efficiency report without the behavioural tests.",
    ),
    "full": Bundle(
        key="full",
        label="Everything",
        sections=("bench", "system", "energy", "quality", "notes"),
        note="Every section that has data.",
    ),
}


#: Energy metrics, grouped the ways that make sense to read together. A
#: per-token figure and a per-million figure are the same measurement at two
#: scales; mixing them into one flat list makes both harder to read.
ENERGY_GROUPS: Dict[str, dict] = {
    "per_token": {
        "label": "Energy per token",
        "fields": [
            ("j_per_prefill_token", "J / prefill token", "J"),
            ("j_per_decode_token", "J / decode token", "J"),
        ],
    },
    "per_million": {
        "label": "Energy per million tokens",
        "fields": [
            ("kwh_per_1m_prefill", "kWh / 1M prefill tokens", "kWh"),
            ("kwh_per_1m_decode", "kWh / 1M decode tokens", "kWh"),
        ],
    },
    "power": {
        "label": "Average power",
        "fields": [
            ("avg_prefill_watts", "average prefill power", "W"),
            ("avg_decode_watts", "average decode power", "W"),
            ("idle_watts", "idle power", "W"),
        ],
    },
    "per_card": {
        "label": "Per-card efficiency",
        "fields": [],  # rendered from the per_card list, one row per card
    },
}


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def _table(headers: List[str], rows: List[List[str]]) -> str:
    if not rows:
        return ""
    out = ["| " + " | ".join(headers) + " |"]
    out.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        out.append("| " + " | ".join("" if c is None else str(c) for c in r) + " |")
    return "\n".join(out)


def _bench_section(payload: dict) -> str:
    results = payload.get("bench_results") or []
    if not results:
        return ""
    rows = []
    for r in results:
        m = r.get("metric") or {}
        detail = r.get("detail") or {}
        measures = []
        if m.get("name") and m.get("name") != "none" and m.get("value") is not None:
            measures.append(f"{m['name']}={m['value']}{(' ' + m['unit']) if m.get('unit') else ''}")
        for key, label, unit in (
            ("ttft_ms", "ttft", "ms"),
            ("prefill_tps", "prefill", "tok/s"),
        ):
            if detail.get(key) is not None:
                measures.append(f"{label}={detail[key]} {unit}")
        rows.append([
            r.get("test_id"),
            scrub.scrub_text(str(r.get("label") or "")),
            r.get("status"),
            "; ".join(measures) or "—",
        ])
    body = _table(["#", "test", "status", "measure / value"], rows)
    return "## Benchmark\n\n" + body + "\n"


def _lead_metrics_section(payload: dict) -> str:
    lead = payload.get("lead_metrics") or {}
    if not lead:
        return ""
    labels = {
        "ms_per_verify_round": "ms / verify round",
        "ms_per_decode_round": "ms / decode round",
        "ms_per_1k_prefill_tokens": "ms / 1k prefill tokens",
        "ms_per_draft_pass": "ms / draft pass",
        "accept_length": "accepted tokens per round",
    }
    rows = [
        [labels.get(k, k), f"{v:.2f}"]
        for k, v in lead.items()
        if k in labels and v is not None
    ]
    if not rows:
        return ""
    return "### Round times\n\n" + _table(["measure", "value"], rows) + "\n"


def _system_section(payload: dict) -> str:
    """Hardware and versions, scrubbed.

    Card models and driver versions are the reason to share a system
    section. Host names, addresses and paths are not, and every string here
    goes through the scrubber on its way in -- there is no branch that skips
    it.
    """
    sysd = payload.get("system") or {}
    rows = []
    for key, label in (
        ("cards", "cards"),
        ("driver", "driver"),
        ("cuda", "CUDA"),
        ("torch", "torch"),
        ("commit", "commit"),
        ("model", "model"),
        ("quant", "quantisation"),
        ("interconnect", "interconnect"),
    ):
        v = sysd.get(key)
        if v in (None, "", [], {}):
            continue
        if isinstance(v, (list, tuple)):
            v = ", ".join(str(x) for x in v)
        rows.append([label, scrub.scrub_text(str(v))])
    out = ""
    if rows:
        out += "## System\n\n" + _table(["item", "value"], rows) + "\n"
    flags = payload.get("launch_flags") or []
    if flags:
        cleaned = scrub.scrub_launch_flags([str(f) for f in flags])
        out += "\n<details><summary>launch flags</summary>\n\n```\n"
        out += " ".join(cleaned)
        out += "\n```\n</details>\n"
    return out


def _energy_section(payload: dict, groups: Optional[List[str]] = None) -> str:
    energy = payload.get("energy") or {}
    if not energy:
        return ""
    wanted = groups or list(ENERGY_GROUPS)
    out = ["## Energy"]
    for gkey in wanted:
        spec = ENERGY_GROUPS.get(gkey)
        if not spec:
            continue
        if gkey == "per_card":
            cards = energy.get("per_card") or []
            rows = [
                [
                    scrub.scrub_text(str(c.get("name") or "")),
                    c.get("j_per_decode_token"),
                    c.get("watts"),
                    c.get("efficiency"),
                ]
                for c in cards
            ]
            if rows:
                out.append("\n### " + spec["label"] + "\n")
                out.append(
                    _table(
                        ["card", "J / decode token", "W", "relative efficiency"],
                        rows,
                    )
                )
            continue
        rows = [
            [label, f"{energy[field]:.4g}", unit]
            for field, label, unit in spec["fields"]
            if energy.get(field) is not None
        ]
        if rows:
            out.append("\n### " + spec["label"] + "\n")
            out.append(_table(["measure", "value", "unit"], rows))
    if len(out) == 1:
        return ""
    return "\n".join(out) + "\n"


def _quality_section(payload: dict) -> str:
    q = payload.get("quality") or {}
    if not q:
        return ""
    rows = []
    for key, label in (
        ("verdict", "verdict"),
        ("representation", "representation"),
    ):
        if q.get(key):
            rows.append([label, scrub.scrub_text(str(q[key]))])
    tokens = q.get("tokens") or {}
    for key in ("prompt", "completion", "total"):
        if tokens.get(key) is not None:
            rows.append([f"{key} tokens", tokens[key]])
    if not rows:
        return ""
    return "## Quality (chess)\n\n" + _table(["measure", "value"], rows) + "\n"


def _notes_section(payload: dict) -> str:
    notes = (payload.get("notes") or "").strip()
    if not notes:
        return ""
    return "## Notes\n\n" + scrub.scrub_text(notes) + "\n"


def build_markdown(
    payload: dict,
    bundle: str = "bench_system",
    *,
    energy_groups: Optional[List[str]] = None,
    title: str = "htsglang measured results",
) -> str:
    """Render the selected bundle. Pure: no network, no filesystem, no token.

    This is what the preview shows and, byte for byte, what would be posted.
    A preview that is assembled differently from the payload is not a
    preview.
    """
    b = BUNDLES.get(bundle)
    if b is None:
        raise DiscussionError(
            f"unknown bundle {bundle!r}; choose one of {', '.join(sorted(BUNDLES))}"
        )
    parts = [MARKER, f"# {title}", ""]
    for section in b.sections:
        if section == "bench":
            parts.append(_bench_section(payload))
            parts.append(_lead_metrics_section(payload))
        elif section == "system":
            parts.append(_system_section(payload))
        elif section == "energy":
            parts.append(_energy_section(payload, energy_groups))
        elif section == "quality":
            parts.append(_quality_section(payload))
        elif section == "notes":
            parts.append(_notes_section(payload))
    body = "\n".join(p for p in parts if p)
    if body.strip() == MARKER + f"\n# {title}".strip():
        body += "\nNothing selected had any data.\n"
    return body.rstrip() + "\n"


# ---------------------------------------------------------------------------
# Token and target
# ---------------------------------------------------------------------------


def read_token(path: Optional[str] = None) -> Optional[str]:
    """Read the PAT from its file. Returns None when unconfigured.

    Never logged, never echoed back to a caller, never placed in a URL. The
    only thing that leaves this function is the token itself, to the
    Authorization header.
    """
    p = path or os.environ.get(PAT_FILE_ENV) or ""
    if not p:
        return None
    p = os.path.expanduser(p)
    try:
        with open(p, encoding="utf-8") as f:
            token = f.read().strip()
    except OSError:
        return None
    return token or None


def configured_target(target: Optional[str] = None) -> Optional[str]:
    """The discussion to post into, or None when the send is not armed."""
    t = (target or os.environ.get(TARGET_ENV) or "").strip()
    return t or None


# ---------------------------------------------------------------------------
# Preview and send
# ---------------------------------------------------------------------------


def preview(
    payload: dict,
    bundle: str = "bench_system",
    *,
    energy_groups: Optional[List[str]] = None,
    target: Optional[str] = None,
    token_path: Optional[str] = None,
) -> dict:
    """The Markdown, plus an honest statement of whether it could be sent."""
    markdown = build_markdown(payload, bundle, energy_groups=energy_groups)
    tgt = configured_target(target)
    has_token = read_token(token_path) is not None
    if tgt is None:
        ready, why = False, "no target configured"
    elif not has_token:
        ready, why = False, (
            f"no token: set {PAT_FILE_ENV} to a file holding a GitHub PAT"
        )
    else:
        ready, why = True, ""
    return {
        "ok": True,
        "markdown": markdown,
        "bundle": bundle,
        "target": tgt,
        "can_send": ready,
        "reason": why,
        "bundles": [
            {"key": b.key, "label": b.label, "sections": list(b.sections),
             "note": b.note}
            for b in BUNDLES.values()
        ],
        "energy_groups": [
            {"key": k, "label": v["label"]} for k, v in ENERGY_GROUPS.items()
        ],
    }


def _graphql(
    query: str,
    variables: dict,
    token: str,
    *,
    opener: Optional[Callable[[urllib.request.Request], bytes]] = None,
) -> dict:
    """One POST. The token appears here and nowhere else."""
    body = json.dumps({"query": query, "variables": variables}).encode()
    req = urllib.request.Request(
        API_URL,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "User-Agent": "htsglang-planner",
        },
    )
    try:
        if opener is not None:
            raw = opener(req)
        else:  # pragma: no cover - exercised against a mock, never live
            with urllib.request.urlopen(req, timeout=30) as r:
                raw = r.read()
    except Exception as e:
        raise DiscussionError(github_share.redact(f"{type(e).__name__}: {e}", token))
    try:
        out = json.loads(raw.decode("utf-8"))
    except Exception as e:
        raise DiscussionError(github_share.redact(f"malformed response: {e}", token))
    if out.get("errors"):
        msgs = "; ".join(
            str(e.get("message") or e) for e in out["errors"]
        )
        raise DiscussionError(github_share.redact(msgs, token))
    return out.get("data") or {}


_Q_COMMENTS = """
query($id: ID!) {
  node(id: $id) {
    ... on Discussion {
      id
      url
      comments(first: 100) {
        nodes { id body viewerDidAuthor }
      }
    }
  }
}
"""

_M_ADD = """
mutation($id: ID!, $body: String!) {
  addDiscussionComment(input: {discussionId: $id, body: $body}) {
    comment { id url }
  }
}
"""

_M_UPDATE = """
mutation($id: ID!, $body: String!) {
  updateDiscussionComment(input: {commentId: $id, body: $body}) {
    comment { id url }
  }
}
"""


def submit(
    payload: dict,
    bundle: str = "bench_system",
    *,
    energy_groups: Optional[List[str]] = None,
    target: Optional[str] = None,
    token_path: Optional[str] = None,
    confirmed: bool = False,
    opener: Optional[Callable[[urllib.request.Request], bytes]] = None,
) -> dict:
    """Post the previewed Markdown -- if, and only if, everything is armed.

    Four gates, in order, and any one of them stops the call before a single
    byte leaves the machine: a configured target, a readable token, an
    explicit ``confirmed``, and a bundle that renders. With no target this
    returns the preview and says so; it does not create a discussion to post
    into. Creating one is a decision for a human, not a side effect of
    pressing a button labelled "share".

    A re-submit UPDATES the previous comment rather than adding another,
    found by :data:`MARKER` among the comments this token's user authored --
    the same mechanism ``github_share`` uses for issues.
    """
    pv = preview(
        payload, bundle, energy_groups=energy_groups, target=target,
        token_path=token_path,
    )
    if not pv["can_send"]:
        return dict(pv, sent=False)
    if not confirmed:
        return dict(
            pv,
            sent=False,
            can_send=True,
            reason="not confirmed: nothing is sent without an explicit confirm",
        )

    token = read_token(token_path)
    if not token:  # pragma: no cover - preview already gated on this
        return dict(pv, sent=False, reason="no token")
    discussion_id = pv["target"]
    markdown = pv["markdown"]

    data = _graphql(_Q_COMMENTS, {"id": discussion_id}, token, opener=opener)
    node = (data or {}).get("node") or {}
    existing = None
    for c in ((node.get("comments") or {}).get("nodes") or []):
        if c.get("viewerDidAuthor") and MARKER in (c.get("body") or ""):
            existing = c["id"]
            break

    if existing:
        out = _graphql(
            _M_UPDATE, {"id": existing, "body": markdown}, token, opener=opener
        )
        comment = (out.get("updateDiscussionComment") or {}).get("comment") or {}
        action = "updated"
    else:
        out = _graphql(
            _M_ADD, {"id": discussion_id, "body": markdown}, token, opener=opener
        )
        comment = (out.get("addDiscussionComment") or {}).get("comment") or {}
        action = "created"

    return dict(
        pv,
        sent=True,
        action=action,
        url=comment.get("url"),
        comment_id=comment.get("id"),
        reason="",
    )
