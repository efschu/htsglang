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

S78 review A1: D runs ``event_loop_overlap`` -- the forward (and with it the counter ``add_``s and the timing
events) is on ``forward_stream``, the scheduler funnel that closes the record on ``schedule_stream``; nothing
orders the two. The record therefore remembers the stream of its first EM layer, and the counter D2H plus the
``done`` event are enqueued on THAT stream (``torch.cuda.stream(rec.stream)``) -- stream order, no
``wait_event`` on the schedule stream (that would serialize the overlap), no sync.

HOW. ``run_pool_em`` brackets each wave's ``copy_rows`` with two timing events and adds the step's device
``gather_count`` (rows) and ``gather_count x row bytes`` into a device counter -- no host read. The
scheduler funnel closes the forward at the next forward's start: one non-blocking D2H of the counter into
pinned memory plus an event; the line is written when that event has completed (``query``), a forward or
two later. Never a ``synchronize``, never an ``item`` on a busy stream. Rank-local, read-only, no collective;
off under graph capture. Switch ``FLLIPER_PDFLIP_EM_FETCH`` (default ON).

F3 (20.10.) TIME PER WAVE. ``run_pool_em`` runs every wave's apply over ALL ``T x K`` lanes (a lane the wave does
not serve points at the zero row with weight 0). On today's D prefills every wave-1 copy fits the 105 rows of LRU +
staging (rows <= ~63 per layer), so wave 2 would serve nothing -- yet its apply still runs: a possible second full
expert GEMM per layer that ``em_ms - copy_ms`` cannot separate. The line therefore also carries, per wave index
(summed over the forward's EM layers): ``wave_copy_ms`` (copy events), ``wave_apply_ms`` (copy end -> the wave's
last apply slice, device time) and ``wave_served`` (lanes the wave served, device counter, read with the same
non-blocking D2H). ``wave_served`` 0 with ``wave_apply_ms`` > 0 is the empty wave's GEMM, measured. No sync.

EM-PREFETCH (21.10.). A forward whose layers prefetched the NEXT layer's predicted rows on the pool prefetch side
stream (``MoEExpertOffloadCache.prefetch_pool_em``) carries, summed over its layers::

    pf_layers=<prefetches issued> pf_predicted=<distinct experts predicted> pf_fetched=<rows the side stream
      copied> pf_hits=<of those, rows the target layer's first wave really used> pf_copy_ms=<side-stream copy time>
      pf_wait_ms=<review A4: time the forward stream stood at a join for the side stream (join -> copy end, >= 0)>

``copy_ms`` stays the SYNCHRONOUS row copies on the forward stream (what the prefetch is meant to shrink);
``pf_fetched - pf_hits`` is the wasted prefetch. Counters are device adds on the forward stream (the
predicted count before the side stream forks, fetched and hits after the target layer joined it), so they
ride the same one non-blocking D2H; the side-stream copy events are complete once the forward stream has
joined every prefetch, which it does before its record closes. No sync.
"""
from __future__ import annotations

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

MARK = "EM-FETCH"
MAX_PENDING = 16
#: F3: per-wave counters kept (EM waves are capped far below this; a higher wave index folds into the last)
MAX_WAVES = 8
#: EM-PREFETCH counter slots after the per-wave ones: predicted, fetched, hits
PF_BASE = 2 + 2 * MAX_WAVES
PF_PRED, PF_FETCHED, PF_HITS = PF_BASE, PF_BASE + 1, PF_BASE + 2
#: the one device counter a record carries (rows, bytes, per-wave served + computed, prefetch)
COUNTER_LEN = PF_BASE + 3
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

    __slots__ = ("fwd", "stream", "start", "end", "pairs", "counter", "layers", "waves", "tokens", "host", "done",
                 "wave_copy", "wave_apply", "last_copy_end", "pf_layers", "pf_pairs", "pf_waits")

    def __init__(self, fwd: int, device, cuda):
        import torch

        self.fwd = int(fwd)
        self.stream = cuda.current_stream()  # review A1: the stream the EM work (adds, events) runs on
        self.start = cuda.Event(enable_timing=True)
        self.start.record()
        self.end = None
        self.pairs: List[tuple] = []
        # [rows, bytes, served lanes of wave 1 .. MAX_WAVES (F3), computed lanes of wave 1 .. MAX_WAVES (W2)]
        # EM-PREFETCH: + [predicted, fetched, hits] (PF_BASE ..)
        self.counter = torch.zeros(COUNTER_LEN, dtype=torch.int64, device=device)
        self.pf_layers = 0
        self.pf_pairs: List[tuple] = []  # EM-PREFETCH: side-stream copy events
        self.pf_waits: List[tuple] = []  # review A4: (join event on the forward stream, copy end on the side)
        self.wave_copy: dict = {}   # F3: wave -> [(copy start, copy end)]
        self.wave_apply: dict = {}  # F3: wave -> [(copy end, apply end)]
        self.last_copy_end = None
        self.layers = 0
        self.waves = 0
        self.tokens = 0
        self.host = None
        self.done = None

    def mark(self):
        ev = _cuda().Event(enable_timing=True)
        ev.record()
        return ev

    def copied(self, e0, gather_count, row_bytes: int, wave: int = 0) -> None:
        """After one wave's ``copy_rows``: close its timing pair, add its rows and bytes (device ops only)."""
        import torch

        e1 = _cuda().Event(enable_timing=True)
        e1.record()
        self.pairs.append((e0, e1))
        if wave > 0:
            self.wave_copy.setdefault(min(int(wave), MAX_WAVES), []).append((e0, e1))
            self.last_copy_end = e1
        self.waves += 1
        n = gather_count.reshape(-1)[:1].to(torch.int64)
        self.counter[0:1].add_(n)
        self.counter[1:2].add_(n * int(row_bytes))

    def applied(self, wave: int, served, computed=None) -> None:
        """F3: after one wave's apply slices -- its device time (copy end -> now) and its served lanes (a device
        sum into the counter; no host read). W2-APPLY: ``computed`` = the lanes the wave's GEMMs computed (a
        device mask, or the host lane count of the zero-row form)."""
        import torch

        e2 = _cuda().Event(enable_timing=True)
        e2.record()
        w = min(max(1, int(wave)), MAX_WAVES)
        if self.last_copy_end is not None:
            self.wave_apply.setdefault(w, []).append((self.last_copy_end, e2))
            self.last_copy_end = None
        self.counter[1 + w:2 + w].add_(served.reshape(-1).to(torch.int64).sum().reshape(1))
        if computed is not None:
            c = 1 + MAX_WAVES + w
            if isinstance(computed, int):
                self.counter[c:c + 1].add_(int(computed))
            else:
                self.counter[c:c + 1].add_(computed.reshape(-1).to(torch.int64).sum().reshape(1))

    def prefetch_issued(self, predicted) -> None:
        """EM-PREFETCH: one layer issued the next layer's prefetch; ``predicted`` = its distinct predicted
        experts (device scalar/tensor, summed on the forward stream before the side stream forks)."""
        import torch

        self.pf_layers += 1
        self.counter[PF_PRED:PF_PRED + 1].add_(predicted.reshape(-1).to(torch.int64).sum().reshape(1))

    def prefetch_copy(self, e0, e1) -> None:
        """EM-PREFETCH: the side stream's copy events of one prefetch (read once the forward has joined it)."""
        self.pf_pairs.append((e0, e1))

    def prefetch_wait(self, join_ev, end_ev) -> None:
        """Review A4: the forward stream reached a prefetch's join at ``join_ev``; the side stream's copy ended at
        ``end_ev``. ``max(0, join -> end)`` is the time the forward stood still for it (summed at line time over
        completed pairs only, never waited for)."""
        self.pf_waits.append((join_ev, end_ev))

    def prefetch_landed(self, fetched, hits) -> None:
        """EM-PREFETCH: on the forward stream after the target layer joined the prefetch -- the rows it copied
        (the prefetch step's gather count) and the rows of those its first wave used (pf hit delta)."""
        import torch

        self.counter[PF_FETCHED:PF_FETCHED + 1].add_(fetched.reshape(-1)[:1].to(torch.int64))
        self.counter[PF_HITS:PF_HITS + 1].add_(hits.reshape(-1)[:1].to(torch.int64))

    def end_layer(self, tokens: int) -> None:
        ev = _cuda().Event(enable_timing=True)
        ev.record()
        self.end = ev
        self.layers += 1
        self.tokens = max(self.tokens, int(tokens))


def _cuda():
    import torch

    return torch.cuda


def current() -> Optional[Record]:
    """The record of the forward that is open (None = off, under capture, or no EM layer yet). Never opens one."""
    try:
        if not enabled():
            return None
        return _STATE["cur"]
    except Exception:  # noqa: BLE001 -- an instrument never stops a forward
        return None


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


def _d2h(host, counter) -> None:
    """The counter's non-blocking copy into pinned memory (one seam for the stream-order test)."""
    host.copy_(counter, non_blocking=True)


def _close_current(cuda) -> None:
    import torch

    rec = _STATE["cur"]
    _STATE["cur"] = None
    if rec is None or rec.end is None:
        return
    rec.host = torch.empty(int(rec.counter.numel()), dtype=torch.int64, pin_memory=bool(torch.cuda.is_available()))
    rec.done = cuda.Event()
    # review A1: behind the counter's adds and the end event in stream order (the funnel runs on the
    # schedule stream, the EM work on the forward stream)
    with cuda.stream(rec.stream):
        _d2h(rec.host, rec.counter)
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
            "dropped=%d (device time of this forward's EAGER-DEVICE-EM row copies vs the whole EM span; rank-local)%s"
            % (MARK, rec.fwd, rec.tokens, rec.layers, rec.waves, rows, nbytes / float(1 << 30), copy_ms, em_ms,
               gbps, _STATE["dropped"], wave_terms(rec) + pf_terms(rec)))


def pf_terms(rec: Record) -> str:
    """EM-PREFETCH: `` pf_layers=.. pf_predicted=.. pf_fetched=.. pf_hits=.. pf_copy_ms=..`` ('' when the forward
    issued none). A copy pair whose end has not completed is left out, never waited for."""
    n = int(getattr(rec, "pf_layers", 0) or 0)
    if n <= 0:
        return ""
    host = rec.host
    vals = [int(host[i]) if host is not None and i < len(host) else 0 for i in (PF_PRED, PF_FETCHED, PF_HITS)]
    ms = 0.0
    for a, b in getattr(rec, "pf_pairs", ()) or ():
        try:
            if bool(b.query()):
                ms += float(a.elapsed_time(b))
        except Exception:  # noqa: BLE001 -- an instrument never stops a forward
            pass
    wait = 0.0
    for a, b in getattr(rec, "pf_waits", ()) or ():
        try:
            if bool(a.query()) and bool(b.query()):
                wait += max(0.0, float(a.elapsed_time(b)))
        except Exception:  # noqa: BLE001 -- an instrument never stops a forward
            pass
    return (" pf_layers=%d pf_predicted=%d pf_fetched=%d pf_hits=%d pf_copy_ms=%.1f pf_wait_ms=%.1f"
            % (n, vals[0], vals[1], vals[2], ms, wait))


def wave_terms(rec: Record) -> str:
    """F3: `` wave_copy_ms=a/b wave_apply_ms=a/b wave_served=a/b`` over the wave indices seen ('' when none)."""
    waves = sorted(set(getattr(rec, "wave_copy", {}) or {}) | set(getattr(rec, "wave_apply", {}) or {}))
    if not waves:
        return ""

    def _ms(pairs):
        return sum(float(a.elapsed_time(b)) for a, b in pairs)

    host = rec.host
    served = [int(host[1 + w]) if host is not None and 1 + w < len(host) else 0 for w in waves]
    lanes = [int(host[1 + MAX_WAVES + w]) if host is not None and 1 + MAX_WAVES + w < len(host) else 0
             for w in waves]
    return (" wave_copy_ms=%s wave_apply_ms=%s wave_served=%s wave_lanes=%s"
            % ("/".join("%.1f" % _ms(rec.wave_copy.get(w, ())) for w in waves),
               "/".join("%.1f" % _ms(rec.wave_apply.get(w, ())) for w in waves),
               "/".join(str(x) for x in served), "/".join(str(x) for x in lanes)))


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
