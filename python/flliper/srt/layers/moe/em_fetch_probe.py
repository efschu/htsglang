# SPDX-License-Identifier: Apache-2.0
"""S78 EM-FETCH: the expert host->device share of a D mixed / extend forward, per rank, without a sync.

WHY. S78 (rc17 D ``..._1019_171636``): 388 of 703 ``D-DECODE-GAP cause=other`` gaps (473.9 of 648.9 s) were
opened by a MIXED-D round; its forward (start -> ``PDFLIP-VRAM-PEAK phase=chunk``) took 444.8 s, fit
639 ms + 1.07 ms/token over the EAGER-DEVICE-EM range. Which part of that is the expert rows the pool steps
copy over each rank's link (5090 x8 14.42 GB/s, 3080 x8 13.4, 3080 x4 6.47 -- S51 card probe) and which the
rest no log line said: S51's H107 bytes lines exist only on the host plan, and all three ranks return from
the forward within 2 ms of each other (the collectives couple them), so the host stamps cannot name the
pacer either. This names it on the device, per forward and rank:

    EM-FETCH fwd=<forward_ct> T=<tokens> layers=<EM layers> waves=<pool steps> rows=<rows copied>
      gib=<bytes copied> copy_ms=<device time of those copy_rows> em_ms=<device time first EM op -> last>
      link_gbps=<bytes / copy_ms>

HOW. ``run_pool_em`` brackets each wave's ``copy_rows`` with two timing events and adds the step's device
``gather_count`` (rows) and ``gather_count x row bytes`` into a device counter -- no host read. The
scheduler funnel closes the forward at the next forward's start: one non-blocking D2H of the counter into
pinned memory plus an event; the line is written when that event has completed (``query``), a forward or
two later. Never a ``synchronize``, never an ``item`` on a busy stream. Rank-local, read-only, no collective;
off under graph capture. Switch ``FLLIPER_PDFLIP_EM_FETCH`` (default ON).
"""
from __future__ import annotations

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

MARK = "EM-FETCH"
MAX_PENDING = 16
#: the device types the probe times (a CPU harness adds "cpu" with a fake event class)
DEVICE_TYPES = ("cuda",)
_STATE = {"on": None, "cur": None, "pending": [], "fwd": 0, "dropped": 0}


def enabled() -> bool:
    if _STATE["on"] is None:
        try:
            from flliper.srt.environ import envs

            _STATE["on"] = bool(envs.FLLIPER_PDFLIP_EM_FETCH.get())
        except Exception:  # noqa: BLE001
            _STATE["on"] = False
    return bool(_STATE["on"])


def reset_for_tests() -> None:
    _STATE.update({"on": None, "cur": None, "pending": [], "fwd": 0, "dropped": 0})


class Record:
    """One forward's EM layers on this rank (device counters + timing events)."""

    __slots__ = ("fwd", "start", "end", "pairs", "counter", "layers", "waves", "tokens", "host", "done")

    def __init__(self, fwd: int, device, cuda):
        import torch

        self.fwd = int(fwd)
        self.start = cuda.Event(enable_timing=True)
        self.start.record()
        self.end = None
        self.pairs: List[tuple] = []
        self.counter = torch.zeros(2, dtype=torch.int64, device=device)  # [rows, bytes]
        self.layers = 0
        self.waves = 0
        self.tokens = 0
        self.host = None
        self.done = None

    def mark(self):
        ev = _cuda().Event(enable_timing=True)
        ev.record()
        return ev

    def copied(self, e0, gather_count, row_bytes: int) -> None:
        """After one wave's ``copy_rows``: close its timing pair, add its rows and bytes (device ops only)."""
        import torch

        e1 = _cuda().Event(enable_timing=True)
        e1.record()
        self.pairs.append((e0, e1))
        self.waves += 1
        n = gather_count.reshape(-1)[:1].to(torch.int64)
        self.counter[0:1].add_(n)
        self.counter[1:2].add_(n * int(row_bytes))

    def end_layer(self, tokens: int) -> None:
        ev = _cuda().Event(enable_timing=True)
        ev.record()
        self.end = ev
        self.layers += 1
        self.tokens = max(self.tokens, int(tokens))


def _cuda():
    import torch

    return torch.cuda


def begin_layer(device) -> Optional[Record]:
    """The record of the forward this EM layer belongs to (opened on its first EM layer); None = no probe."""
    try:
        if not enabled() or getattr(device, "type", None) not in DEVICE_TYPES:
            return None
        cuda = _cuda()
        if bool(cuda.is_current_stream_capturing()):
            return None
        rec = _STATE["cur"]
        if rec is None:
            rec = Record(_STATE["fwd"], device, cuda)
            _STATE["cur"] = rec
        return rec
    except Exception:  # noqa: BLE001 -- an instrument never stops a forward
        return None


def _close_current(cuda) -> None:
    import torch

    rec = _STATE["cur"]
    _STATE["cur"] = None
    if rec is None or rec.end is None:
        return
    rec.host = torch.empty(2, dtype=torch.int64, pin_memory=bool(torch.cuda.is_available()))
    rec.host.copy_(rec.counter, non_blocking=True)
    rec.done = cuda.Event()
    rec.done.record()
    pend = _STATE["pending"]
    pend.append(rec)
    while len(pend) > MAX_PENDING:
        pend.pop(0)
        _STATE["dropped"] += 1


def line_of(rec: Record) -> str:
    copy_ms = sum(float(a.elapsed_time(b)) for a, b in rec.pairs)
    em_ms = float(rec.start.elapsed_time(rec.end))
    rows, nbytes = int(rec.host[0]), int(rec.host[1])
    gbps = (nbytes / (copy_ms / 1000.0) / 1e9) if copy_ms > 0 else 0.0
    return ("%s fwd=%d T=%d layers=%d waves=%d rows=%d gib=%.3f copy_ms=%.1f em_ms=%.1f link_gbps=%.2f "
            "dropped=%d (device time of this forward's EAGER-DEVICE-EM row copies vs the whole EM span; rank-local)"
            % (MARK, rec.fwd, rec.tokens, rec.layers, rec.waves, rows, nbytes / float(1 << 30), copy_ms, em_ms,
               gbps, _STATE["dropped"]))


def drain() -> int:
    """Log every closed record whose device work has completed (event query, never a wait). Returns lines."""
    n = 0
    pend = _STATE["pending"]
    while pend:
        rec = pend[0]
        if not bool(rec.done.query()):
            break
        pend.pop(0)
        logger.info(line_of(rec))
        n += 1
    return n


def on_forward(fwd: int) -> None:
    """The scheduler funnel at a forward's start: close the previous forward's record, tag the next one, log
    the completed ones. Never raises, never waits."""
    try:
        if not enabled():
            return
        cuda = _cuda()
        if _STATE["cur"] is not None:
            _close_current(cuda)
        _STATE["fwd"] = int(fwd)
        if _STATE["pending"]:
            drain()
    except Exception:  # noqa: BLE001 -- an instrument never stops a forward
        _STATE["cur"] = None
