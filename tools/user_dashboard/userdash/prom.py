"""Prometheus text exposition: parser, aggregation and histogram quantiles (stdlib only).

The parser follows the same contract as the planner's ``parse_prometheus_metrics`` and rigdash's
tolerant readers: a malformed line is skipped, never raised, so one odd family cannot blank the page.

Aggregation rule for the server's own families (``sglang:*``): every PP stage's stats rank exports
the same request/token figures (measured on the live 27B front 01.10.: ``realtime_tokens_total`` of P
identical on pp_rank 0/1/2; static families of D also on tp_rank 0 and 1). So within one source (front
group ``weg2_group`` or a scraped group URL) the MAXIMUM is taken over the rank labels (``RANK_LABELS``)
and the SUM over everything else that differs (``dp_rank``, ``via``, ...) -- ``collapse``.
"""

from __future__ import annotations

import math
import re
from typing import Dict, Iterable, List, Optional, Tuple

Labels = Tuple[Tuple[str, str], ...]
Sample = Tuple[str, Labels, float]

_LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{(.*)\})?\s+(\S+)(\s+\S+)?\s*$")
_LABEL = re.compile(r'\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*"((?:[^"\\]|\\.)*)"\s*,?')


def _unescape(v: str) -> str:
    return v.replace('\\"', '"').replace("\\n", "\n").replace("\\\\", "\\")


def _value(s: str) -> Optional[float]:
    s = s.strip()
    if s in ("+Inf", "Inf"):
        return math.inf
    if s == "-Inf":
        return -math.inf
    try:
        v = float(s)
    except ValueError:
        return None
    return None if math.isnan(v) else v


def parse_labels(body: str) -> Optional[Labels]:
    out = []
    pos = 0
    body = body.strip()
    while pos < len(body):
        m = _LABEL.match(body, pos)
        if not m:
            return None
        out.append((m.group(1), _unescape(m.group(2))))
        pos = m.end()
    return tuple(sorted(out))


def parse(text: str) -> List[Sample]:
    """All samples of an exposition as ``(name, labels, value)``; comments and bad lines dropped."""
    out: List[Sample] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _LINE.match(line)
        if not m:
            continue
        labels: Optional[Labels] = ()
        if m.group(3) is not None:
            labels = parse_labels(m.group(3))
            if labels is None:
                continue
        v = _value(m.group(4))
        if v is None:
            continue
        out.append((m.group(1), labels, v))
    return out


def label(labels: Labels, key: str, default: Optional[str] = None) -> Optional[str]:
    for k, v in labels:
        if k == key:
            return v
    return default


def without(labels: Labels, *keys: str) -> Labels:
    return tuple((k, v) for k, v in labels if k not in keys)


def select(samples: Iterable[Sample], name: str, **match: str) -> List[Tuple[Labels, float]]:
    out = []
    for n, lb, v in samples:
        if n != name:
            continue
        if all(label(lb, k) == want for k, want in match.items()):
            out.append((lb, v))
    return out


def has_family(samples: Iterable[Sample], prefix: str) -> bool:
    return any(n.startswith(prefix) for n, _, _ in samples)


def has_label(samples: Iterable[Sample], key: str) -> bool:
    return any(label(lb, key) is not None for _, lb, _ in samples)


def source_of(labels: Labels, group_label: str = "weg2_group") -> str:
    """The server process a sample came from: the front's group relabel, else the scrape source tag."""
    return label(labels, group_label) or label(labels, "_src") or ""


#: labels whose values repeat the same figure (one model replica split over ranks): max, never sum
RANK_LABELS = ("pp_rank", "tp_rank", "moe_ep_rank", "attn_tp_rank", "attn_cp_rank", "pid")


def collapse(rows: Iterable[Tuple[Labels, float]], group_label: str = "weg2_group") -> Dict[str, float]:
    """Per source: max over the rank labels, sum over the remaining label sets (see module doc)."""
    per: Dict[str, Dict[Labels, float]] = {}
    for lb, v in rows:
        if v is None or not math.isfinite(v):
            continue
        src = source_of(lb, group_label)
        rest = without(lb, group_label, "_src", *RANK_LABELS)
        d = per.setdefault(src, {})
        d[rest] = max(d.get(rest, -math.inf), v)
    return {src: sum(d.values()) for src, d in per.items() if d}


def total(rows: Iterable[Tuple[Labels, float]]) -> Optional[float]:
    by = collapse(rows)
    return sum(by.values()) if by else None


# --- histograms ---------------------------------------------------------------------------------

def histogram(samples: Iterable[Sample], name: str) -> Dict[float, float]:
    """Cumulative bucket counts ``{le: count}`` of family ``name`` summed over all series (sources and
    label sets such as ``via``), PP stages collapsed like the counters."""
    per_le: Dict[float, List[Tuple[Labels, float]]] = {}
    for n, lb, v in samples:
        if n != name + "_bucket":
            continue
        le = _value(label(lb, "le", "") or "")
        if le is None:
            continue
        per_le.setdefault(le, []).append((without(lb, "le"), v))
    out = {}
    for le, rows in per_le.items():
        t = total(rows)
        if t is not None:
            out[le] = t
    return out


def bucket_delta(new: Dict[float, float], old: Optional[Dict[float, float]]) -> Dict[float, float]:
    """``new - old`` per bucket; a bucket that shrank means a restart of the server -> take ``new``."""
    if not old:
        return dict(new)
    out = {}
    reset = any(new.get(le, 0.0) < old.get(le, 0.0) for le in new)
    for le, c in new.items():
        out[le] = c if reset else c - old.get(le, 0.0)
    return out


def quantile(q: float, buckets: Dict[float, float]) -> Optional[float]:
    """Prometheus ``histogram_quantile``: linear inside the bucket; a quantile in the +Inf bucket is
    reported as the largest finite bound. None without observations."""
    if not buckets:
        return None
    les = sorted(buckets)
    count = buckets[les[-1]]
    if count <= 0:
        return None
    rank = q * count
    prev_le, prev_c = 0.0, 0.0
    for le in les:
        c = buckets[le]
        if c >= rank:
            if math.isinf(le):
                finite = [x for x in les if not math.isinf(x)]
                return finite[-1] if finite else None
            if c == prev_c:
                return le
            return prev_le + (le - prev_le) * (rank - prev_c) / (c - prev_c)
        prev_le, prev_c = le, c
    return None


def count_of(buckets: Dict[float, float]) -> float:
    if not buckets:
        return 0.0
    return buckets[max(buckets)]
