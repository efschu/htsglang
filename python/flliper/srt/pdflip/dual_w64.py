"""DUAL-TP3PP3: W64 for group D judged on D's own MEASURED budget posts.

Why (metal 30.09., boot a3t5js ...09300606, D TP0 on the 5090): the family
model behind W64 prices D r0 at weights 13362 + mamba 1522 + reserves 2304 MiB
(17188). The runtime's own ``KV budget posts`` line says 12.020 + 1.388 +
0.098 + 1.000 GiB (~14853 MiB), and D sized 79138 KV tokens where the model
predicted 4224. In the dual layout D's budget is what the 5090 leaves after P,
so a ~2.3 GiB pessimism there caps P's KV pool.

The runtime's arithmetic under an absolute per-rank budget is
``KV = budget - posts``, and its posts line states every term. So feasibility
at a new budget is read off the same identity: tokens_r = (budget_r -
posts_r) / cell. The minimum tokens per rank stay the model's own
(``_PREDICT_MIN_RANK_TOKENS``).

Only a D log of the same model whose dual-share D published the SAME installed
weight vector counts (the posts depend on the shard). Pure stdlib, streamed:
a boot log can be hundreds of MB, and the posts sit in its first minutes.
"""

from __future__ import annotations

import dataclasses
import glob
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

MIB = 1 << 20
GIB = 1 << 30
#: newest logs looked at, lines read per log (the posts appear before READY)
MAX_LOGS = 24
MAX_LINES = 400000

_RE_RANK = re.compile(r"\sTP(\d+)\]")
_RE_MODEL = re.compile(r"model_path='([^']*)'")
_RE_RATIOS = re.compile(r"PDFLIP-UNION D ratios published for the dual P stage: \{'tp': \[([\d, ]+)\]")
_RE_POSTS = re.compile(r"KV budget posts \(GiB\):(.*?)\|\s*rest=")
_RE_VAL = re.compile(r"=\s*([0-9]+(?:\.[0-9]+)?)")
_RE_CELL = re.compile(r"KV pool sizing: .*cell_size=(\d+)")


@dataclasses.dataclass(frozen=True)
class DualDMeasurement:
    source: str
    weights: Tuple[int, ...]
    posts_mib: Tuple[float, ...]
    cell_bytes: int


@dataclasses.dataclass(frozen=True)
class DualW64Verdict:
    feasible: bool
    tokens: Tuple[int, ...]
    line: str


def _scan(path: str, model: str, tp_size: int) -> Optional[DualDMeasurement]:
    weights = None
    model_ok = False
    posts: Dict[int, float] = {}
    cell = None
    try:
        with open(path, "r", errors="replace") as f:
            for i, line in enumerate(f):
                if i >= MAX_LINES:
                    break
                if not model_ok and "model_path='" in line:
                    model_ok = any(model == m or m.rstrip("/") == model.rstrip("/") for m in _RE_MODEL.findall(line))
                    if not model_ok:
                        return None
                if weights is None and "PDFLIP-UNION D ratios published" in line:
                    m = _RE_RATIOS.search(line)
                    if m:
                        weights = tuple(int(x) for x in m.group(1).split(",") if x.strip())
                if "KV budget posts (GiB):" in line:
                    r, p = _RE_RANK.search(line), _RE_POSTS.search(line)
                    if r and p:
                        posts.setdefault(int(r.group(1)),
                                         sum(float(v) for v in _RE_VAL.findall(p.group(1))) * 1024.0)
                if cell is None and "KV pool sizing:" in line:
                    c = _RE_CELL.search(line)
                    if c:
                        cell = int(c.group(1))
                if model_ok and weights and cell and len(posts) >= tp_size:
                    break
    except OSError:
        return None
    if not (model_ok and weights and cell and all(r in posts for r in range(tp_size))):
        return None
    return DualDMeasurement(os.path.basename(path), weights,
                            tuple(round(posts[r], 1) for r in range(tp_size)), cell)


def find_dual_d_measurement(evidence_dirs: Sequence[str], model: str,
                            weights: Sequence[int]) -> Optional[DualDMeasurement]:
    """Newest dual-share D log of ``model`` whose D installed ``weights``."""
    want = tuple(int(w) for w in weights)
    logs: List[str] = []
    for d in evidence_dirs:
        logs.extend(glob.glob(os.path.join(d, "boot_*.D.log")))
    logs = sorted(set(logs), key=lambda p: os.path.getmtime(p), reverse=True)[:MAX_LOGS]
    for path in logs:
        m = _scan(path, model, len(want))
        if m is not None and m.weights == want:
            return m
    return None


def judge(m: DualDMeasurement, budgets: Sequence[int], min_tokens: int) -> DualW64Verdict:
    toks = tuple(int((float(b) - p) * MIB // m.cell_bytes) for b, p in zip(budgets, m.posts_mib))
    ok = all(t >= int(min_tokens) for t in toks)
    rows = "; ".join(
        "r%d budget=%d - measured posts=%.0f => %d tokens%s" % (
            r, int(b), p, t, "" if t >= min_tokens else "  <-- BELOW %d" % min_tokens)
        for r, (b, p, t) in enumerate(zip(budgets, m.posts_mib, toks)))
    line = ("W64-DUAL MEASURED (%s, D weights %s, cell %d B): %s -> %s" % (
        m.source, list(m.weights), m.cell_bytes, rows, "feasible" if ok else "INFEASIBLE"))
    return DualW64Verdict(ok, toks, line)
