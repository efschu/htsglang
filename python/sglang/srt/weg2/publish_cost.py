"""fnFL2 H49: the per-chunk publish on group P, priced per chunk AND per request.

``_weg2_publish_at_chunk`` (xsn346, ``SGLANG_WEG2_PUBLISH_AT_CHUNK``) runs
synchronously on the scheduler thread between two forwards of the prefill,
bounded by ``SGLANG_WEG2_PUBLISH_AT_CHUNK_BUDGET_MS`` (default 150). Its
existing lines (``WEG2 CHUNK-PUBLISH`` + ``H2-WRITE-PATH chunk``) are SAMPLED
(n <= 16, then every 256th per rank), name neither the budget nor the chunk's
position in its request, and no line sums a request -- so "what did the
publish cost this prompt's prefill" had to be joined by hand.

This ledger prints one ``WEG2-PUBLISH-CHUNK`` line per chunk (not sampled:
one per chunk per rank) and one ``WEG2-PUBLISH-REQ`` line per request at its
retain (the P end of the request; the front's ``WEG2-SERVED group=P leg=1``
of the same rid follows it). Pure bookkeeping on numbers the publish site
already measures; no clock of its own.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Optional

logger = logging.getLogger(__name__)

__all__ = ["LEDGER", "PublishCostLedger"]


class _Acc:
    __slots__ = ("chunks", "sum_ms", "max_ms", "over", "stopped", "budget_ms")

    def __init__(self) -> None:
        self.chunks = 0
        self.sum_ms = 0.0
        self.max_ms = 0.0
        self.over = 0
        self.stopped = 0
        self.budget_ms = 0.0


class PublishCostLedger:
    """Per-rid sums of the chunk publishes of one rank."""

    #: Requests kept open at once (a request that never retains -- aborted --
    #: must not grow the ledger without bound).
    MAX_OPEN: int = 256

    def __init__(self) -> None:
        self._open: "OrderedDict[str, _Acc]" = OrderedDict()

    def note_chunk(
        self,
        *,
        rid: str,
        ms: float,
        budget_ms: float,
        stopped: Optional[str],
        issued: int,
        parts: str,
    ) -> str:
        acc = self._open.get(rid)
        if acc is None:
            acc = _Acc()
            self._open[rid] = acc
            while len(self._open) > self.MAX_OPEN:
                self._open.popitem(last=False)
        acc.chunks += 1
        acc.sum_ms += float(ms)
        acc.max_ms = max(acc.max_ms, float(ms))
        over = budget_ms > 0 and ms > budget_ms
        if over:
            acc.over += 1
        if stopped:
            acc.stopped += 1
        acc.budget_ms = float(budget_ms)
        return (
            "WEG2-PUBLISH-CHUNK rid=%s chunk=%d ms=%.1f budget_ms=%.0f over=%s "
            "stopped=%s issued=%d sum_ms=%.1f parts=%s"
            % (
                rid[:12],
                acc.chunks,
                ms,
                budget_ms,
                "yes" if over else "no",
                stopped or "-",
                int(issued),
                acc.sum_ms,
                parts,
            )
        )

    def close(self, rid: str, *, retain_ms: Optional[float]) -> Optional[str]:
        """The request retained: its sum line (None when no chunk of it
        published on this rank and no retain publish ran)."""
        acc = self._open.pop(rid, None)
        if acc is None and retain_ms is None:
            return None
        acc = acc or _Acc()
        total = acc.sum_ms + (retain_ms or 0.0)
        return (
            "WEG2-PUBLISH-REQ rid=%s chunks=%d chunk_sum_ms=%.1f chunk_max_ms=%.1f "
            "over_budget=%d stopped_budget=%d budget_ms=%.0f retain_ms=%s total_ms=%.1f "
            "(scheduler-thread wall of this request's publishes on this rank, "
            "between forwards)"
            % (
                rid[:12],
                acc.chunks,
                acc.sum_ms,
                acc.max_ms,
                acc.over,
                acc.stopped,
                acc.budget_ms,
                "-" if retain_ms is None else "%.1f" % retain_ms,
                total,
            )
        )


LEDGER = PublishCostLedger()


def format_parts(clock) -> str:
    """``parts=`` from a finished ``hicache_write_path.PublishClock``."""
    try:
        d = clock.delta
        return "issue:%.1f,move:%.1f,cpu:%.1f,blocked:%.1f,ops:%d" % (
            float(d.issue_wall_ms),
            float(d.move_wall_ms),
            float(clock.cpu_ms),
            max(0.0, float(clock.wall_ms) - float(clock.cpu_ms)),
            int(d.ops),
        )
    except Exception:  # noqa: BLE001 -- an instrument never raises
        return "-"
