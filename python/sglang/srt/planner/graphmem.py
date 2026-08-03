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
"""CUDA-graph memory: boot-log capture parsing, a measured-anchor store, and a
calibrated heuristic estimator for the config-planner dashboard.

THREE LAYERS (measured always beats estimated, the power-calibration
discipline):

1. PARSER -- the scheduler boot log prints one line per finished capture::

     [ts TPn] Capture target decode CUDA graph end. elapsed=..., \
mem usage=0.27 GB, avail mem=...
     [ts TPn] Capture draft extend CUDA graph end. ...

   Kinds seen in real boots: ``target prefill`` / ``target decode`` /
   ``target verify`` / ``draft decode`` / ``draft extend``. With the adaptive
   ladder (#93/#102 high-accept) the DRAFT kinds repeat once per rung -- each
   repetition is a SEPARATE graph with its own memory (a 5-rung ladder OOM'd
   a 3080 on exactly this), so the parser keys every draft capture by its
   occurrence index (rung) and, when the paired ``begin`` line is present, by
   that rung's token count (``num_tokens_per_bs``).

2. ANCHOR STORE -- every parsed boot is persisted keyed by (model, tp, spec
   shape, kv dtype, attention backend, page size, capture-bs list):
   ``~/.cache/sglang/graph_mem_anchors.json`` (override:
   ``SGLANG_PLANNER_GRAPH_ANCHORS``). A prospective config whose key matches
   an anchor gets the MEASURED numbers (provenance "measured"), never the
   heuristic -- which is why the key has to be complete: an incomplete one
   does not degrade to an estimate, it hands over another config's
   measurement still labelled "measured" (#513).

3. HEURISTIC -- when no anchor matches::

     per-rank graph MiB (one graph kind) =
         BASE + SLOPE * n_bs * (1 + TOK_ALPHA * (tokens_per_bs - 1)) * K
     with K = hidden * total_layers / tp_size

   plus, for the draft ladder: per rung one decode graph (~DRAFT_DECODE_MIB)
   and one extend graph (~DRAFT_EXTEND_MIB), and a one-time first-capture
   workspace (~DRAFT_FIRST_MIB_PER_BS * n_bs). The constants below are FIT
   against the real capture lines in the /tmp boot logs of this rig
   (Qwen3.6-27B FP8 tp=3 with/without the 5-rung adaptive ladder, and
   Qwen3-0.6B tp=1): observed error band ABOUT +-30% -- surfaced to the UI so
   the segment is honestly labeled an estimate, never a measurement.
"""

from __future__ import annotations

import glob
import json
import math
import os
import re
from typing import Dict, List, Optional, Sequence

__all__ = [
    "parse_capture_lines",
    "parse_boot_meta",
    "summarize_captures",
    "anchor_key",
    "ANCHOR_KEY_VERSION",
    "AnchorStore",
    "scan_boot_logs",
    "heuristic_estimate",
    "estimate",
    "DEFAULT_ANCHOR_PATH",
    "DEFAULT_BOOT_LOG_GLOBS",
    "ERROR_BAND_PCT",
]

_GIB_TO_MIB = 1024.0

#: Stated heuristic error band (percent) -- verified against the calibration
#: logs in the unit tests; shown verbatim in the UI tooltip.
ERROR_BAND_PCT = 30

DEFAULT_ANCHOR_PATH = os.path.expanduser(
    os.environ.get(
        "SGLANG_PLANNER_GRAPH_ANCHORS",
        os.path.join("~", ".cache", "sglang", "graph_mem_anchors.json"),
    )
)

#: Where managed/orchestrated boots leave their logs (server_manager writes
#: ``sglang_boot_<port>.log``; the energy bench writes ``energy_boot_*.log``).
DEFAULT_BOOT_LOG_GLOBS = (
    "/tmp/sglang_boot_*.log",
    "/tmp/energy_boot_*.log",
)

# ---------------------------------------------------------------------------
# Heuristic constants (calibrated 2026-07 against the real /tmp boot logs).
# ---------------------------------------------------------------------------
#: Fixed workspace per captured graph kind (flashinfer workspace, pool
#: bookkeeping): the intercept of the linear fit, stable across model sizes
#: (27B tp3 decode -> 282-85=197 slope part; 0.6B tp1 decode 110-85=26).
BASE_MIB = 85.0
#: MiB per batch-size entry per K activation unit (K = hidden*layers/tp).
#: Fit: 27B tp3 decode 16.4 MiB/bs at K=109227 and 0.6B tp1 decode
#: 4.3 MiB/bs at K=28672 -> 1.50e-4 / 1.46e-4.
SLOPE_MIB_PER_K_PER_BS = 1.5e-4
#: Sub-linear growth with tokens-per-bs (verify graphs replay draft+1 tokens
#: per row): verify(6 tok) slope / decode(1 tok) slope ~ 3.2 -> alpha 0.44.
TOK_ALPHA = 0.44
#: Draft ladder: rung >= 1 captures cost roughly one BASE each (measured
#: 0.06-0.11 GB); the FIRST draft capture additionally allocates the draft
#: workspace, which scales with the bs count (~0.35 GB at 6 bs, 0.15 at 2).
DRAFT_DECODE_MIB = 85.0
DRAFT_EXTEND_MIB = 70.0
DRAFT_FIRST_MIB_PER_BS = 45.0

#: Default decode capture-bs ladder (server default, truncated at
#: max_running_requests) -- used when the caller does not know the list.
_DEFAULT_DECODE_BS = (1, 2, 4, 8, 12, 16, 24)
#: Spec (verify + draft) graphs use a short dense bs list in the observed
#: boots ([1..6] at draft 6); capped by max_running_requests.
_SPEC_BS_CAP = 6


# ---------------------------------------------------------------------------
# Parser.
# ---------------------------------------------------------------------------
_CAPTURE_END_RE = re.compile(
    r"\[(?P<ts>[^\]]*?)(?:\s+TP(?P<rank>\d+))?\]\s+"
    r"Capture\s+(?P<kind>(?:target|draft)\s+\w+)\s+CUDA graph end\."
    r".*?mem usage=(?P<gb>[0-9.]+)\s*GB"
)
_CAPTURE_BEGIN_RE = re.compile(
    r"\[(?P<ts>[^\]]*?)(?:\s+TP(?P<rank>\d+))?\]\s+"
    r"Capture\s+(?P<kind>(?:target|draft)\s+\w+)\s+CUDA graph begin\."
    r"(?:.*?num_tokens_per_bs=(?P<tok>\d+))?"
    r"(?:.*?bs=\[(?P<bs>[0-9,\s]*)\])?"
)


def parse_capture_lines(text: str) -> List[dict]:
    """Parse every ``Capture ... CUDA graph end`` line of a boot log.

    Returns one entry per finished capture::

        {"kind": "draft extend", "rank": 0, "mib": 143.4, "rung": 1,
         "tokens_per_bs": 5, "n_bs": 6}

    ``rung`` counts repeated captures of the SAME kind on the SAME rank (the
    adaptive ladder captures each rung separately; target kinds are captured
    once, so their rung is 0). ``tokens_per_bs`` / ``n_bs`` come from the
    matching ``begin`` line when present (None otherwise).
    """
    # Pending begin metadata per (rank, kind): the runtime always logs
    # begin -> end in order per rank, so the last unconsumed begin matches.
    pending: Dict[tuple, dict] = {}
    counts: Dict[tuple, int] = {}
    out: List[dict] = []
    for line in text.splitlines():
        if "CUDA graph" not in line:
            continue
        mb = _CAPTURE_BEGIN_RE.search(line)
        if mb and "begin" in line:
            rank = int(mb.group("rank")) if mb.group("rank") else None
            kind = " ".join(mb.group("kind").split())
            bs = mb.group("bs")
            pending[(rank, kind)] = {
                "tokens_per_bs": (
                    int(mb.group("tok")) if mb.group("tok") else None
                ),
                "n_bs": (len([x for x in bs.split(",") if x.strip()])
                         if bs else None),
            }
            continue
        me = _CAPTURE_END_RE.search(line)
        if not me:
            continue
        rank = int(me.group("rank")) if me.group("rank") else None
        kind = " ".join(me.group("kind").split())
        key = (rank, kind)
        rung = counts.get(key, 0)
        counts[key] = rung + 1
        meta = pending.pop(key, {})
        out.append(
            {
                "kind": kind,
                "rank": rank,
                "mib": float(me.group("gb")) * _GIB_TO_MIB,
                "rung": rung,
                "tokens_per_bs": meta.get("tokens_per_bs"),
                "n_bs": meta.get("n_bs"),
            }
        )
    return out


_META_PATTERNS = {
    "model_path": r"model_path='([^']*)'",
    "tp_size": r"tp_size=(\d+)",
    "kv_cache_dtype": r"kv_cache_dtype='([^']*)'",
    # #513: both change captured graph memory and were missing from the key.
    "attention_backend": r"attention_backend='([^']*)'",
    "page_size": r"page_size=(\d+)",
    "speculative_algorithm": r"speculative_algorithm='([^']*)'",
    "speculative_num_steps": r"speculative_num_steps=(\d+)",
    "speculative_num_draft_tokens": r"speculative_num_draft_tokens=(\d+)",
    "speculative_adaptive": r"speculative_adaptive=(True|False)",
}


def parse_boot_meta(text: str) -> Optional[dict]:
    """Extract the anchor-key config from the ``server_args=ServerArgs(...)``
    line of a boot log. None when no such line exists (not an sglang boot)."""
    if "server_args=ServerArgs(" not in text:
        return None
    meta: Dict[str, object] = {}
    for k, pat in _META_PATTERNS.items():
        m = re.search(pat, text)
        if not m:
            continue
        v = m.group(1)
        if v in ("True", "False"):
            meta[k] = v == "True"
        elif v.isdigit():
            meta[k] = int(v)
        else:
            meta[k] = v
    if "model_path" not in meta:
        return None
    m = re.search(
        r"decode=PhaseConfig\(backend='[^']*', max_bs=\d+, bs=\[([0-9,\s]*)\]",
        text,
    )
    if m:
        meta["decode_bs"] = [
            int(x) for x in m.group(1).split(",") if x.strip()
        ]
    return meta


def summarize_captures(entries: Sequence[dict]) -> dict:
    """Aggregate parsed capture entries for display / anchoring.

    Returns::

        {"items": [{"kind", "rung", "label", "per_rank": {rank: mib},
                    "total_mib", "tokens_per_bs"}...],
         "per_rank_mib": {rank: summed mib},
         "total_mib": float, "n_captures": int}

    ``label`` itemizes the ladder: draft kinds get ``draft decode rung 0`` /
    ``draft extend k=5`` (the rung's token count when the begin line carried
    it); target kinds keep their plain name.
    """
    items: Dict[tuple, dict] = {}
    per_rank: Dict[int, float] = {}
    for e in entries:
        key = (e["kind"], e["rung"])
        it = items.setdefault(
            key,
            {
                "kind": e["kind"],
                "rung": e["rung"],
                "label": None,
                "per_rank": {},
                "total_mib": 0.0,
                "tokens_per_bs": e.get("tokens_per_bs"),
            },
        )
        r = e["rank"] if e["rank"] is not None else 0
        it["per_rank"][r] = it["per_rank"].get(r, 0.0) + e["mib"]
        it["total_mib"] += e["mib"]
        per_rank[r] = per_rank.get(r, 0.0) + e["mib"]
    out = []
    for (kind, rung), it in sorted(items.items()):
        if kind.startswith("draft"):
            tok = it.get("tokens_per_bs")
            # extend graphs carry the rung's k as num_tokens_per_bs; decode
            # graphs always run 1 token/bs, so their rung index is the key.
            it["label"] = (
                f"{kind} k={tok}"
                if (tok is not None and tok > 1)
                else f"{kind} rung {rung}"
            )
        else:
            it["label"] = kind
        out.append(it)
    return {
        "items": out,
        "per_rank_mib": per_rank,
        "total_mib": sum(per_rank.values()),
        "n_captures": len(entries),
    }


# ---------------------------------------------------------------------------
# Anchor store (measured-overrides-estimate, like the power calibration).
# ---------------------------------------------------------------------------
#: Bumped whenever the key composition changes, so a reader can tell "this
#: anchor was written under a different key recipe" from "this rig never
#: booted that shape". v2 (#513) added the attention backend, the page size
#: and the capture-bs LIST; v1 anchors no longer match and are rebuilt by
#: re-running ``scan_boot_logs`` over the boot logs they came from.
ANCHOR_KEY_VERSION = "v2"


def anchor_key(meta: dict) -> str:
    """Stable anchor key from a boot/prospective config.

    Key fields: model basename, tp, spec shape (algo/steps/draft/adaptive),
    kv dtype, attention backend, page size, and the decode capture-bs LIST --
    exactly the knobs that change graph memory, so changing k or adaptive in
    the runner changes the key (and falls back to the ladder-aware heuristic
    unless that shape was booted before).

    #513 (audit #506, finding A3-4) fixed two omissions. The key carried
    ``nbs:{len(bs)}``, the NUMBER of capture batch sizes, so two different
    ``--cuda-graph-bs`` lists of equal length shared an anchor and the second
    one was handed the first one's MEASURED numbers -- provenance "measured",
    not "estimate", which is what makes that failure silent. And the
    attention backend was absent although ``BASE_MIB`` below is documented as
    the "flashinfer workspace", i.e. the module's own account of the quantity
    is backend-dependent. ``page_size`` joins them for the same reason.
    """
    model = os.path.basename(str(meta.get("model_path") or "").rstrip("/"))
    algo = meta.get("speculative_algorithm") or "off"
    bs = meta.get("decode_bs") or []
    return "|".join(
        [
            ANCHOR_KEY_VERSION,
            model,
            f"tp{meta.get('tp_size') or 1}",
            f"spec:{algo}",
            f"steps:{meta.get('speculative_num_steps') or 0}",
            f"draft:{meta.get('speculative_num_draft_tokens') or 0}",
            f"adaptive:{int(bool(meta.get('speculative_adaptive')))}",
            f"kv:{meta.get('kv_cache_dtype') or 'auto'}",
            f"attn:{meta.get('attention_backend') or 'auto'}",
            f"page:{meta.get('page_size') or 1}",
            "bs:" + ",".join(str(int(x)) for x in bs),
        ]
    )


class AnchorStore:
    """Tiny JSON store: anchor key -> measured capture summary."""

    def __init__(self, path: str = DEFAULT_ANCHOR_PATH):
        self.path = os.path.expanduser(path)

    def _read(self) -> dict:
        try:
            with open(self.path) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}

    def record(self, meta: dict, summary: dict, source: str = "") -> str:
        """Persist one parsed boot (latest boot of a key wins). Returns the
        key. per_rank dicts are stored with string keys (JSON)."""
        data = self._read()
        key = anchor_key(meta)
        data[key] = {
            "provenance": "measured",
            "meta": {k: meta[k] for k in meta},
            "summary": summary,
            "source": source,
        }
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=1)
        os.replace(tmp, self.path)
        return key

    def lookup(self, meta: dict) -> Optional[dict]:
        """Exact anchor key first; else the same config shape ignoring the
        capture-bs count (the planner cannot know the server's bs list --
        a measured boot of the same model/tp/spec/kv shape still beats the
        heuristic)."""
        data = self._read()
        key = anchor_key(meta)
        hit = data.get(key)
        if hit is not None:
            return hit
        prefix = key.rsplit("|nbs:", 1)[0]
        for k in sorted(data):
            if k.rsplit("|nbs:", 1)[0] == prefix:
                return data[k]
        return None


#: Per-process scan cache: (path, mtime, size) already parsed. The planner
#: re-checks anchors on every placement refresh; unchanged logs are skipped.
_SCANNED: dict = {}


def scan_boot_logs(
    paths: Optional[Sequence[str]] = None,
    store: Optional[AnchorStore] = None,
) -> int:
    """Parse every boot log on disk into the anchor store (idempotent --
    latest boot of a config shape wins). Returns the number of logs anchored.
    This is how the store self-populates: every managed/orchestrated boot
    leaves its capture lines in /tmp, so measured anchors accumulate without
    any explicit calibration step. Unchanged files (same mtime+size) are
    skipped within a process."""
    if paths is None:
        paths = []
        for pat in DEFAULT_BOOT_LOG_GLOBS:
            paths.extend(sorted(glob.glob(pat)))
    store = store or AnchorStore()
    n = 0
    for p in paths:
        try:
            st = os.stat(p)
            sig = (st.st_mtime, st.st_size)
            if _SCANNED.get(p) == sig:
                continue
            with open(p, errors="replace") as f:
                text = f.read()
            _SCANNED[p] = sig
        except OSError:
            continue
        meta = parse_boot_meta(text)
        if not meta:
            continue
        entries = parse_capture_lines(text)
        if not entries:
            continue
        store.record(meta, summarize_captures(entries), source=p)
        n += 1
    return n


# ---------------------------------------------------------------------------
# Heuristic estimator (fallback; ladder-aware).
# ---------------------------------------------------------------------------
def _per_graph_mib(n_bs: int, tokens_per_bs: int, k_units: float) -> float:
    return BASE_MIB + (
        SLOPE_MIB_PER_K_PER_BS
        * n_bs
        * (1.0 + TOK_ALPHA * (tokens_per_bs - 1))
        * k_units
    )


def ladder_rungs(spec: dict) -> List[int]:
    """The draft-ladder token counts the estimator models. Adaptive
    (#93/#102 high-accept): one rung per k step from ``draft_tokens`` down
    (observed boots: draft 6 / steps 5 -> extends at k=6..2; draft 4 /
    steps 3 -> k=4..1); non-adaptive: a single rung at ``draft_tokens``."""
    dt = int(spec.get("speculative_num_draft_tokens") or 0)
    steps = int(spec.get("speculative_num_steps") or 0)
    if dt <= 0:
        return []
    if not spec.get("speculative_adaptive"):
        return [dt]
    n = max(steps, dt - 1, 1)
    return [max(dt - i, 1) for i in range(n)]


def heuristic_estimate(geom: dict, spec: Optional[dict] = None) -> dict:
    """Ladder-aware heuristic CUDA-graph memory estimate, per rank.

    ``geom``: {hidden_size, num_hidden_layers, tp_size,
    max_running_requests?}. ``spec``: {speculative_algorithm,
    speculative_num_steps, speculative_num_draft_tokens,
    speculative_adaptive} or None/empty for spec-off.

    Returns {"provenance": "heuristic", "per_rank_mib": float (same on every
    rank), "items": [{"label", "mib", "formula"}...], "error_band_pct",
    "formula"} -- items itemize target graphs and EVERY draft rung
    separately, so changing k or adaptive visibly changes the estimate.
    """
    spec = spec or {}
    hidden = int(geom.get("hidden_size") or 0)
    layers = int(geom.get("num_hidden_layers") or 0)
    tp = max(int(geom.get("tp_size") or 1), 1)
    mrr = int(geom.get("max_running_requests") or 16)
    k_units = hidden * layers / tp

    spec_on = bool(spec.get("speculative_algorithm"))
    items: List[dict] = []

    if spec_on:
        dt = int(spec.get("speculative_num_draft_tokens") or 4)
        n_bs = max(1, min(_SPEC_BS_CAP, mrr))
        mib = _per_graph_mib(n_bs, dt, k_units)
        items.append(
            {
                "label": "target verify",
                "mib": mib,
                "formula": (
                    f"{BASE_MIB:.0f} + {SLOPE_MIB_PER_K_PER_BS:g}*{n_bs}bs"
                    f"*(1+{TOK_ALPHA}*({dt}-1))*K, K=hidden*layers/tp"
                    f"={k_units:.0f}"
                ),
            }
        )
        rungs = ladder_rungs(spec)
        for i, k in enumerate(rungs):
            mib_r = DRAFT_DECODE_MIB + DRAFT_EXTEND_MIB
            if i == 0:
                mib_r += DRAFT_FIRST_MIB_PER_BS * n_bs
            items.append(
                {
                    "label": (
                        f"draft graphs k={k}"
                        + (" (base rung, first-capture workspace)"
                           if i == 0 else " (+rung)")
                    ),
                    "mib": mib_r,
                    "formula": (
                        f"decode {DRAFT_DECODE_MIB:.0f} + extend "
                        f"{DRAFT_EXTEND_MIB:.0f}"
                        + (f" + first-capture {DRAFT_FIRST_MIB_PER_BS:.0f}"
                           f"*{n_bs}bs" if i == 0 else "")
                    ),
                }
            )
    else:
        bs_list = geom.get("decode_bs") or [
            b for b in _DEFAULT_DECODE_BS if b <= max(mrr, 1)
        ]
        n_bs = max(len(bs_list), 1)
        mib = _per_graph_mib(n_bs, 1, k_units)
        items.append(
            {
                "label": "target decode",
                "mib": mib,
                "formula": (
                    f"{BASE_MIB:.0f} + {SLOPE_MIB_PER_K_PER_BS:g}*{n_bs}bs"
                    f"*K, K=hidden*layers/tp={k_units:.0f}"
                ),
            }
        )

    total = sum(it["mib"] for it in items)
    return {
        "provenance": "heuristic",
        "per_rank_mib": total,
        "items": items,
        "error_band_pct": ERROR_BAND_PCT,
        "formula": (
            "per graph: BASE + SLOPE*n_bs*(1+TOK_ALPHA*(tokens-1))"
            "*hidden*layers/tp; draft ladder: one decode+extend pair per "
            "rung + one-time first-capture workspace. Calibrated on the "
            "boot logs of this rig; estimate, +-"
            f"{ERROR_BAND_PCT}%."
        ),
    }


def estimate(
    geom: dict,
    spec: Optional[dict] = None,
    model_path: Optional[str] = None,
    kv_cache_dtype: Optional[str] = None,
    store: Optional[AnchorStore] = None,
    scan: bool = True,
) -> dict:
    """Measured-anchor lookup first, heuristic fallback.

    When ``model_path`` is known, the anchor store (self-populated from the
    boot logs on disk when ``scan``) is consulted with the config's anchor
    key; a hit returns the MEASURED per-kind/per-rung numbers (provenance
    "measured"). Otherwise :func:`heuristic_estimate`.
    """
    spec = spec or {}
    if model_path:
        store = store or AnchorStore()
        if scan:
            try:
                scan_boot_logs(store=store)
            except Exception:
                pass
        meta = {
            "model_path": model_path,
            "tp_size": geom.get("tp_size") or 1,
            "kv_cache_dtype": kv_cache_dtype or "auto",
            "speculative_algorithm": spec.get("speculative_algorithm"),
            "speculative_num_steps": spec.get("speculative_num_steps"),
            "speculative_num_draft_tokens": spec.get(
                "speculative_num_draft_tokens"
            ),
            "speculative_adaptive": spec.get("speculative_adaptive"),
            "decode_bs": geom.get("decode_bs"),
        }
        hit = store.lookup(meta)
        if hit:
            s = hit.get("summary") or {}
            per_rank = s.get("per_rank_mib") or {}
            vals = list(per_rank.values())
            return {
                "provenance": "measured",
                "per_rank_mib": (max(vals) if vals else 0.0),
                "per_rank_map": per_rank,
                "items": [
                    {
                        "label": it["label"],
                        "mib": it["total_mib"],
                        "formula": "measured capture line (boot log)",
                    }
                    for it in s.get("items", [])
                ],
                "error_band_pct": 0,
                "formula": "measured capture lines from a real boot of this "
                "exact config shape (anchor: " + anchor_key(meta) + ")",
                "source": hit.get("source"),
            }
    return heuristic_estimate(geom, spec)
