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
#: newest DUAL-SHARE D logs of the model looked at, lines read per log (the posts appear before READY).
#: The cap counts only logs that ARE dual-share D measurements of the asked model (rotation 23.10.: 26 newer
#: int4ar/duo D logs in the shared 27b evidence dir pushed the newest dual log of 22.10. to place 27 by mtime,
#: past the old "newest 24 of ALL logs" cut, and W64 refused a boot the measurement would have lifted).
MAX_LOGS = 24
MAX_LINES = 400000
#: cost bound on files opened at all (a foreign-model log costs ~20 lines, a same-model non-dual log ~4000)
MAX_FILES = 2000

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
                if weights is None and "KV pool sizing:" in line:
                    # a dual-share D publishes its ratios BEFORE it sizes KV (metal 22.10. ..._021600:
                    # ratios at line 3784/5502, first "KV pool sizing" at 5591); KV sized without them =
                    # not a dual-share D log (flip D, duo, ...). Stop here instead of reading MAX_LINES.
                    return None
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
    logs = sorted(set(logs), key=_mtime, reverse=True)[:MAX_FILES]
    seen = 0   # dual-share D measurements of THIS model -- the only logs the MAX_LOGS cap counts
    for path in logs:
        m = _scan(path, model, len(want))
        if m is None:
            continue   # other model, not a dual-share D, or incomplete: does not use up the cap
        if m.weights == want:
            return m
        seen += 1
        if seen >= MAX_LOGS:
            break
    return None


def _mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except OSError:   # rotated away between glob and sort
        return 0.0


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
