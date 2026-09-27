"""Energy per token per class (P-Prefill, D-Prefill, Decode), accumulated while the dashboard runs.

Assignment of power to a class, as stated on the page: every closed 5-s
interval of a live boot costs  E = (mean over the interval of the SUM of
nvidia-smi power.draw of all cards) x 5 s.  E goes to the classes that
computed in that interval (a log line of that class fell in it), split in
equal parts when several did; an interval in which no class computed is
counted as idle energy of the boot and given to no class.  Tokens are counted
in the same covered intervals only, so J/token never mixes a covered
numerator with an uncovered denominator.  Intervals without a power sample
(dashboard down longer than its 15-min power history) are not covered;
the coverage is reported next to the numbers.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Dict, List, Optional

CLASSES = ("P", "D", "dec")
LAG_S = 4.0            # a bucket is closed once the log lines of its end had time to land


def _blank() -> dict:
    return {"last_bucket": None, "covered_s": 0.0, "idle_j": 0.0,
            "cls": {c: {"j": 0.0, "tok": 0, "s": 0.0} for c in CLASSES}}


class EnergyBook:
    def __init__(self, state_dir: Optional[str] = None, bucket_s: float = 5.0):
        self.path = os.path.join(state_dir, "energy.json") if state_dir else None
        self.bucket_s = bucket_s
        self.lock = threading.Lock()
        self.books: Dict[str, dict] = {}
        self._saved = 0.0
        if self.path:
            try:
                with open(self.path) as fh:
                    self.books = json.load(fh)
            except (OSError, ValueError):
                self.books = {}

    def update(self, stem: str, first_t: Optional[float], activity_fn, power_fn, now: float) -> None:
        """Account every closed, not yet accounted bucket of one boot."""
        if first_t is None:
            return
        b_s = self.bucket_s
        with self.lock:
            bk = self.books.setdefault(stem, _blank())
            start = bk["last_bucket"] + b_s if bk["last_bucket"] is not None else (first_t // b_s) * b_s
            start = max(start, ((now - 15 * 60) // b_s) * b_s)      # the power history reaches back 15 min
            end = ((now - LAG_S) // b_s) * b_s                       # first bucket that is NOT closed yet
            n = int((end - start) // b_s)
            if n <= 0:
                return
            act = activity_fn(start, n, b_s)
            pw = power_fn(start, n, b_s)
            for i in range(n):
                t = start + i * b_s
                if pw[i] is None:
                    continue
                e = pw[i] * b_s
                bk["covered_s"] += b_s
                on = [c for c in CLASSES if act[i][c]]
                if on:
                    for c in on:
                        bk["cls"][c]["j"] += e / len(on)
                        bk["cls"][c]["s"] += b_s / len(on)
                else:
                    bk["idle_j"] += e
                bk["cls"]["P"]["tok"] += act[i]["P_tok"]
                bk["cls"]["D"]["tok"] += act[i]["D_tok"]
                bk["cls"]["dec"]["tok"] += act[i]["dec_tok"]
            bk["last_bucket"] = start + (n - 1) * b_s

    def view(self, stem: str, boot_wall_s: Optional[float]) -> Optional[dict]:
        with self.lock:
            bk = self.books.get(stem)
            if not bk:
                return None
            out = {"covered_s": bk["covered_s"], "idle_j": bk["idle_j"],
                   "coverage": (min(1.0, bk["covered_s"] / boot_wall_s) if boot_wall_s else None), "cls": {}}
            for c, v in bk["cls"].items():
                j, tok = v["j"], v["tok"]
                out["cls"][c] = {"j": j, "tok": tok, "s": v["s"],
                                 "j_per_tok": (j / tok) if tok else None,
                                 "wh_per_1k": (j / tok * 1000 / 3600) if tok else None,
                                 "tok_per_j": (tok / j) if j else None}
            return out

    def save(self, now: float, every: float = 30.0) -> None:
        if not self.path or now - self._saved < every:
            return
        self._saved = now
        with self.lock:
            for k in [k for k, v in self.books.items() if (v.get("last_bucket") or 0) < now - 2 * 86400]:
                del self.books[k]        # boots older than two days drop out of the book
            data = json.dumps(self.books)
        try:
            tmp = self.path + ".tmp"
            with open(tmp, "w") as fh:
                fh.write(data)
            os.replace(tmp, self.path)
        except OSError:
            pass


def power_buckets(gpu_series: dict, start: float, n: int, bucket_s: float) -> List[Optional[float]]:
    """Mean of the per-sample SUM of power.draw over all cards, per bucket (None: no sample)."""
    acc = [[0.0, 0] for _ in range(n)]
    for t, row in zip(gpu_series.get("t") or [], gpu_series.get("power") or []):
        i = int((t - start) // bucket_s)
        if 0 <= i < n and row and all(x is not None for x in row):
            acc[i][0] += sum(row)
            acc[i][1] += 1
    return [(a / k) if k else None for a, k in acc]
