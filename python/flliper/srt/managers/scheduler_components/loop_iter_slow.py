"""#1158c instrument 2: LOOP-ITER-SLOW -- one line per overlap-loop iteration that took longer than ``THRESHOLD_S``.

Log only, no behaviour. Boot dkr27bggufrabar1fs10100058 (10.10.): D's scheduler took its first intake at 01:07:15, the
front's /flush_cache was issued at 01:07:29.294 and D's `#1458 CTRL-RECV` came only at 01:07:52.430 -- 23 s inside ONE
iteration, and none of its steps (recv, lane tick, WAR barrier, schedule, run, result) writes a line. The fnFL2 H58 spans
already time recv/sched/result; this collects them per iteration plus lane/war/run and prints the split when the
iteration was slow. Bounded: at most ``MAX_LINES`` lines per process, so a box whose iterations are legitimately long
(large prefill chunks) costs a few lines and nothing more.
"""

from __future__ import annotations

import logging
import time
from typing import Dict

THRESHOLD_S = 2.0
MAX_LINES = 20


class LoopIterSlow:
    def __init__(self, threshold_s: float = THRESHOLD_S, max_lines: int = MAX_LINES):
        self.threshold_s = threshold_s
        self.lines_left = max_lines
        self.iter = 0
        self._t0 = 0.0
        self._spans: Dict[str, float] = {}

    def begin(self) -> None:
        self.iter += 1
        self._t0 = time.perf_counter()
        self._spans = {}

    def span(self, name: str, t0: float) -> None:
        self._spans[name] = self._spans.get(name, 0.0) + (time.perf_counter() - t0) * 1e3

    def end(self, logger: logging.Logger) -> None:
        if self.lines_left <= 0:
            return
        total_s = time.perf_counter() - self._t0
        if total_s < self.threshold_s:
            return
        self.lines_left -= 1
        split = " ".join(f"{k}={v:.1f}" for k, v in self._spans.items())
        logger.warning("LOOP-ITER-SLOW iter=%d total_ms=%.1f %s t=%.3f", self.iter, total_s * 1e3, split, time.time())
