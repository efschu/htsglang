"""fnFL2 H58: where a D decode round's host time goes -- the split of H49's
``host_other_ms`` -- and how long the GPU waited for the host.

WHY. H49 (``host_round_cost.py``) put the x162 D round at wall 26.0 ms,
result_wait 0.0, hicache 0.5 and host_other 25.6 ms on all three ranks, and
the zero result wait read as "the host never waits for the device, the round
is host-clocked". The instrument has a blind spot that makes that reading
wrong: H49 times exactly ONE device wait -- the overlap loop's
``copy_done.synchronize`` in ``process_batch_result_decode`` -- while the
spec-v2 round can block the host on the device at three more places, all
inside ``run_batch``, where H49 counts them as host work:

* ``_wait_ctl_event`` (``barlink_bar1.py``, reached from
  ``_read_status_for_check``): the BAR1 abort check's forced wait after
  ``SGLANG_BARLINK_BAR1_ABORT_MAX_LAG`` (default 4) unresolved staged reads.
  Every graph replay is a check once the TP transport has captured launches,
  so NF's TP0 checks FIVE times per round (draft replay, draft-token
  broadcast, verify replay, accept broadcast, draft-extend replay) against
  ONE resync (the PLE sync below): the draft-token broadcast is the fourth
  unresolved check after the verify replay staged its read, and waits for the
  PREVIOUS verify graph's end -- every round. The workers check three times
  per round with no resync, so they are forced every fourth check;
* ``finish_ple_verify_stage`` (``qwen4_exp_ple_decode_pread.py``, TP0 only):
  waits for the draft's tokens, then hashes and preads the verify row's PLE
  rows -- the GPU idles for that part, and so do the workers, which meet TP0
  in the verify's first all-reduce;
* ``FutureMap.resolve_seq_lens_cpu`` (``overlap_utils.py``): waits for the
  previous round's publish ONLY when ``needs_cpu_seq_lens`` is True. NF's
  backends (hybrid QSA/GDN target, QSA multi-step draft) all opt out, so NF
  takes the GPU-only branch and ``seq_wait`` stays 0; a flashinfer target
  (the 27B line) waits here -- which is where 27B's wait moved once the
  abort check stopped forcing it.

The workers have a fourth, host-to-host wait: ``recv_requests`` broadcasts
TP0's request list over the TP CPU group every loop iteration
(``request_receiver._broadcast_reqs_across_ranks``), so a worker's loop is
paced by TP0's host -- that wait is in ``recv`` below.

x162's own WEG2-POST-WAKE-PASS census agrees: ``run_ms`` (the launch) is the
pass gap minus 3-9 ms on every rank, i.e. the host sits inside ``run_batch``
for nearly the whole round, and the result sync after it finds the device
done. A zero ``result_wait_ms`` therefore means "the wait happened earlier",
not "there was no wait". x165 (``..._ABORT_MAX_LAG=64``, 24.09.) is the
control: the round stayed at 26.0 ms and TP0's PLE sync_ms went from ~1.3 to
7-29 ms -- the host's wait moved into the next device wait on the same chain,
it did not go away.

HOST SPANS (``perf_counter`` around calls that already run; no sync, no
collective, no device allocation). Leaf spans, disjoint in time, all on the
scheduler thread, closed per decode round at the SAME boundaries as H49
(``DecodeRoundLog.begin_round`` / ``end_round``):

* ``recv`` -- ``recv_requests`` + ``process_input_requests`` (on a worker:
  the wait for TP0's per-iteration request broadcast);
* ``sched`` -- ``get_next_batch_to_run`` (contains H49's ``hicache``);
* ``seq_wait`` -- the device wait in ``resolve_seq_lens_cpu`` (0 where the
  backends opt out of the CPU mirror, as NF's do);
* ``draft`` -- ``EagleDraftWorker.draft`` (a shadow rank: the draft-token
  broadcast it receives); on NF TP0 it contains the forced ``ctl_wait``;
* ``verify`` -- ``EAGLEWorkerV2.verify``, itself ``vprep`` (plan stream,
  ``load_batch``, the PLE D2H) + ``ple_sync`` + ``ple_stage`` + ``launch``
  (the target forward: graph replay and the replay-boundary abort check) +
  ``accept`` (``eagle_sample`` with the accept broadcast, mamba commit,
  bonus tokens);
* ``dext`` -- publish + ``_draft_extend_for_decode`` (a shadow: the stub);
* ``result`` -- the overlap loop's ``pop_and_process`` (contains H49's
  ``result_wait``);
* ``other`` = wall - the sum of the above (``resolve_forward_inputs``, the
  forward isolation, the result D2H launch, the sampling launch, the loop).

Nested terms, contained in the spans above and reported apart, never added:
``bcast`` (every host-path barlink broadcast: draft tokens, accept),
``ctl_wait`` (``_wait_ctl_event``), ``fetch_plan`` (host expert planning,
``ExpertResidencyPlanner.resolve``; the pool mode plans on the device, so it
is non-zero only in a round with an eager MoE pass), and H49's ``hicache`` and
``result_wait``. ``sync = seq_wait + ple_sync + result_wait + ctl_wait`` is
the time the host was blocked on the device; ``host_work = wall - sync``.

GPU GAPS. Six timing events per decode round on the forward stream -- draft
begin, draft end, verify launch, verify end, draft-extend begin, draft-extend
end -- read one or two rounds late with ``query()``, never
``synchronize()``: ``gpu_draft``, ``gpu_gap_ple`` (draft end -> the verify
launch marker: the stream idled until the host launched the verify),
``gpu_verify``, ``gpu_accept`` (sample, accept broadcast, mamba commit, bonus
tokens), ``gpu_dext``, and ``gpu_gap_round`` (the previous round's
draft-extend end -> this round's draft begin, consecutive rounds only).
``gpu_idle = gap_round + gap_ple`` is the device time the HOST cost the
round; everything else in ``wall`` is device work (verify's collectives and
expert fetches included).

COST. ~25 ``perf_counter`` calls and float adds per round, always (like
H49). With the switch on: six event records per round and one ``query`` per
pending round. ``SGLANG_DEBUG_DECODE_HOST_SPLIT=0``: no event is ever
created or recorded and no line is written.
"""

from __future__ import annotations

import functools
import logging
import time
from collections import deque
from typing import Any, Deque, List, Optional, Sequence, Tuple

import msgspec

from sglang.srt.managers.scheduler_components.host_round_cost import (
    COUNTERS as H49_COUNTERS,
)
from sglang.srt.managers.scheduler_components.host_round_cost import (
    HostCostCounters,
    med_max,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DECODE_MARKS",
    "DecodeHostSplit",
    "DeviceGapProbe",
    "MARK_DEXT_BEGIN",
    "MARK_DEXT_END",
    "MARK_DRAFT_BEGIN",
    "MARK_DRAFT_END",
    "MARK_VERIFY_END",
    "MARK_VERIFY_LAUNCH",
    "RoundGpu",
    "RoundSplit",
    "SPLIT",
    "SplitCounters",
    "active_split",
    "mark",
    "note_span",
    "register_active",
    "timed",
]


class SplitCounters:
    """Process-wide, monotone host-span counters (ms, and call counts for the
    nested terms). Written by the sites that already run; read as deltas at
    every decode round boundary. One process is one rank."""

    __slots__ = (
        "recv_ms",
        "sched_ms",
        "seq_wait_ms",
        "draft_ms",
        "verify_ms",
        "vprep_ms",
        "ple_sync_ms",
        "ple_stage_ms",
        "launch_ms",
        "accept_ms",
        "dext_ms",
        "result_ms",
        "bcast_ms",
        "bcast_n",
        "ctl_wait_ms",
        "ctl_wait_n",
        "fetch_plan_ms",
        "fetch_plan_n",
    )

    def __init__(self) -> None:
        for name in self.__slots__:
            setattr(self, name, 0.0)

    def snapshot(self) -> Tuple[float, ...]:
        return tuple(getattr(self, name) for name in self.__slots__)


SPLIT = SplitCounters()
_IDX = {name: i for i, name in enumerate(SplitCounters.__slots__)}
_H49_IDX = {name: i for i, name in enumerate(HostCostCounters.__slots__)}


def note_span(name: str, t0: float, count: Optional[str] = None) -> float:
    """Add ``perf_counter() - t0`` (as ms) to ``SPLIT.<name>``; bump
    ``SPLIT.<count>`` by one when given. Returns the ``perf_counter`` reading
    it took, so a caller can chain the next span from it."""
    t1 = time.perf_counter()
    setattr(SPLIT, name, getattr(SPLIT, name) + (t1 - t0) * 1000.0)
    if count is not None:
        setattr(SPLIT, count, getattr(SPLIT, count) + 1.0)
    return t1


def timed(name: str, count: Optional[str] = None):
    """Decorator form of :func:`note_span` for a call with several returns:
    every exit (a raise included) adds its wall to ``SPLIT.<name>``."""

    def wrap(fn):
        @functools.wraps(fn)
        def inner(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                note_span(name, t0, count)

        return inner

    return wrap


class RoundSplit(msgspec.Struct, frozen=True, kw_only=True):
    """The host side of one decode round's interval (see module docstring)."""

    round_id: int
    wall_ms: float
    recv_ms: float
    sched_ms: float
    seq_wait_ms: float
    draft_ms: float
    verify_ms: float
    vprep_ms: float
    ple_sync_ms: float
    ple_stage_ms: float
    launch_ms: float
    accept_ms: float
    dext_ms: float
    result_ms: float
    bcast_ms: float
    bcast_n: int
    ctl_wait_ms: float
    ctl_wait_n: int
    fetch_plan_ms: float
    fetch_plan_n: int
    hicache_ms: float
    result_wait_ms: float

    @property
    def sync_ms(self) -> float:
        return self.seq_wait_ms + self.ple_sync_ms + self.result_wait_ms + self.ctl_wait_ms

    @property
    def host_work_ms(self) -> float:
        return max(self.wall_ms - self.sync_ms, 0.0)

    @property
    def other_ms(self) -> float:
        spans = (
            self.recv_ms
            + self.sched_ms
            + self.seq_wait_ms
            + self.draft_ms
            + self.verify_ms
            + self.dext_ms
            + self.result_ms
        )
        return max(self.wall_ms - spans, 0.0)


# -- the device half ------------------------------------------------------

MARK_DRAFT_BEGIN = 0
MARK_DRAFT_END = 1
MARK_VERIFY_LAUNCH = 2
MARK_VERIFY_END = 3
MARK_DEXT_BEGIN = 4
MARK_DEXT_END = 5
DECODE_MARKS = (
    "draft_begin",
    "draft_end",
    "verify_launch",
    "verify_end",
    "dext_begin",
    "dext_end",
)
_FULL = (1 << len(DECODE_MARKS)) - 1


class RoundGpu(msgspec.Struct, frozen=True, kw_only=True):
    """Device durations of one decode round, between the six marks."""

    round_id: int
    draft_ms: float
    gap_ple_ms: float
    verify_ms: float
    accept_ms: float
    dext_ms: float
    #: previous round's dext end -> this round's draft begin; None when the
    #: previous decode round is not the immediately preceding one (a prefill
    #: batch, an idle tick or an unread round in between).
    gap_round_ms: Optional[float] = None

    @property
    def idle_ms(self) -> Optional[float]:
        if self.gap_round_ms is None:
            return None
        return self.gap_round_ms + self.gap_ple_ms


class _TorchEvents:
    """Production event backend: CUDA timing events on the current stream."""

    def event(self):
        import torch

        return torch.cuda.Event(enable_timing=True)

    def is_capturing(self) -> bool:
        import torch

        return bool(torch.cuda.is_current_stream_capturing())


class DeviceGapProbe:
    """Six timing events per decode round in a fixed ring; read in round
    order with ``query()`` once the round's last mark has completed."""

    #: Rounds the ring holds. Harvesting runs one to two rounds behind the
    #: host (the round opened at k is complete once k+2 opens), so 8 leaves
    #: room for a burst of slow rounds; a round overwritten unread is counted.
    DEPTH: int = 8

    def __init__(self, backend: Any = None) -> None:
        self._backend = backend if backend is not None else _TorchEvents()
        self._events: Optional[List[List[Any]]] = None
        self._round: List[Optional[int]] = [None] * self.DEPTH
        self._mask: List[int] = [0] * self.DEPTH
        self._next = 0
        self._cur: Optional[int] = None
        self._queue: Deque[int] = deque()
        #: (round_id, slot) of the last harvested round, for gap_round.
        self._last_end: Optional[Tuple[int, int]] = None
        self.unread = 0

    def begin(self, round_id: int) -> None:
        """A decode round opened: marks from now on belong to it."""
        if self._backend.is_capturing():
            self._cur = None
            return
        if self._events is None:
            self._events = [
                [self._backend.event() for _ in DECODE_MARKS]
                for _ in range(self.DEPTH)
            ]
        slot = self._next
        self._next = (slot + 1) % self.DEPTH
        if slot in self._queue:
            # overwritten before it became readable
            self._queue.remove(slot)
            self.unread += 1
        if self._last_end is not None and self._last_end[1] == slot:
            self._last_end = None
        self._round[slot] = int(round_id)
        self._mask[slot] = 0
        self._cur = slot
        self._queue.append(slot)

    def end(self) -> None:
        """The open round ended (non-decode batch, idle tick): no more marks."""
        self._cur = None

    def mark(self, point: int) -> None:
        slot = self._cur
        if slot is None or self._events is None:
            return
        if self._backend.is_capturing():
            return
        self._events[slot][point].record()
        self._mask[slot] |= 1 << point

    def harvest(self) -> List[RoundGpu]:
        """Every queued round whose last mark completed, in round order.

        A round missing a mark (an exception path, a round that never
        reached the forward) is dropped and counted; nothing waits."""
        out: List[RoundGpu] = []
        # never read the round that is still receiving marks
        while self._queue and self._queue[0] != self._cur:
            slot = self._queue[0]
            if self._mask[slot] != _FULL:
                self._queue.popleft()
                self.unread += 1
                self._last_end = None
                continue
            ev = self._events[slot]
            # All six, not just the last: the marks share one stream today,
            # and a mark moved to another stream must not turn into an
            # elapsed_time on an unfinished event.
            if not all(e.query() for e in ev):
                break  # in round order: no later round is read first
            rid = int(self._round[slot])
            last = self._last_end
            try:
                gap_round = None
                if last is not None and last[0] == rid - 1:
                    gap_round = float(
                        self._events[last[1]][MARK_DEXT_END].elapsed_time(
                            ev[MARK_DRAFT_BEGIN]
                        )
                    )
                rg = RoundGpu(
                    round_id=rid,
                    draft_ms=float(ev[0].elapsed_time(ev[1])),
                    gap_ple_ms=float(ev[1].elapsed_time(ev[2])),
                    verify_ms=float(ev[2].elapsed_time(ev[3])),
                    accept_ms=float(ev[3].elapsed_time(ev[4])),
                    dext_ms=float(ev[4].elapsed_time(ev[5])),
                    gap_round_ms=gap_round,
                )
            except Exception:  # noqa: BLE001 - an instrument never stops a round
                self._queue.popleft()
                self.unread += 1
                self._last_end = None
                continue
            out.append(rg)
            self._last_end = (rid, slot)
            self._queue.popleft()
        return out


# -- the round boundary ---------------------------------------------------


class DecodeHostSplit:
    """Closes one ``RoundSplit`` per decode round, harvests the device marks
    and states a DECODE-HOST-SPLIT line every ``period`` rounds."""

    def __init__(
        self,
        *,
        rank: int,
        period: int,
        counters: SplitCounters = SPLIT,
        h49: HostCostCounters = H49_COUNTERS,
        probe: Optional[DeviceGapProbe] = None,
    ) -> None:
        self._rank = int(rank)
        self._period = max(0, int(period))
        self._counters = counters
        self._h49 = h49
        if probe is None and self._period > 0:
            probe = DeviceGapProbe()
        #: None when the switch is off: no event is created or recorded.
        self.probe = probe
        self._open_id: Optional[int] = None
        self._open_t = 0.0
        self._snap: Tuple[float, ...] = ()
        self._snap49: Tuple[float, ...] = ()
        self._window: List[RoundSplit] = []
        self._gpu: List[RoundGpu] = []
        self._unread0 = 0
        self.last: Optional[RoundSplit] = None

    @classmethod
    def from_env(cls, *, rank: int) -> "DecodeHostSplit":
        from sglang.srt.environ import envs

        return cls(rank=rank, period=envs.SGLANG_DEBUG_DECODE_HOST_SPLIT.get())

    @property
    def on(self) -> bool:
        return self._period > 0

    def on_round_open(self, *, round_id: int, mono: float) -> None:
        self._close(mono)
        self._open_id = int(round_id)
        self._open_t = float(mono)
        self._snap = self._counters.snapshot()
        self._snap49 = self._h49.snapshot()
        if self.probe is not None:
            try:
                self._gpu.extend(self.probe.harvest())
                self.probe.begin(round_id)
            except Exception as exc:  # noqa: BLE001 - never stop a round
                self._drop_probe(exc)

    def on_round_end(self, *, mono: float) -> None:
        if self._open_id is not None:
            self._close(mono)
        if self.probe is not None:
            self.probe.end()

    def mark(self, point: int) -> None:
        if self.probe is not None:
            try:
                self.probe.mark(point)
            except Exception as exc:  # noqa: BLE001 - never stop a round
                self._drop_probe(exc)

    def _drop_probe(self, exc: BaseException) -> None:
        """The device half failed (no CUDA, an event error): the host half
        goes on, the gpu_* fields print ``-`` from here on."""
        logger.warning(
            "DECODE-HOST-SPLIT rank=%d: device marks off after %s: %s",
            self._rank,
            type(exc).__name__,
            exc,
        )
        self.probe = None

    def _close(self, mono: float) -> None:
        if self._open_id is None:
            return
        d = [a - b for a, b in zip(self._counters.snapshot(), self._snap)]
        d49 = [a - b for a, b in zip(self._h49.snapshot(), self._snap49)]
        vals = {name: d[i] for name, i in _IDX.items()}
        for name in ("bcast_n", "ctl_wait_n", "fetch_plan_n"):
            vals[name] = int(round(vals[name]))
        rs = RoundSplit(
            round_id=self._open_id,
            wall_ms=1000.0 * max(float(mono) - self._open_t, 0.0),
            hicache_ms=d49[_H49_IDX["hc_ms"]],
            result_wait_ms=d49[_H49_IDX["result_wait_ms"]],
            **vals,
        )
        self._open_id = None
        self.last = rs
        if self._period > 0:
            self._window.append(rs)
            if len(self._window) >= self._period:
                if self.probe is not None:
                    try:
                        self._gpu.extend(self.probe.harvest())
                    except Exception as exc:  # noqa: BLE001
                        self._drop_probe(exc)
                unread = 0
                if self.probe is not None:
                    unread = self.probe.unread - self._unread0
                    self._unread0 = self.probe.unread
                logger.info(
                    self.period_line(
                        self._window, self._gpu, rank=self._rank, gpu_unread=unread
                    )
                )
                self._window = []
                self._gpu = []

    @staticmethod
    def period_line(
        window: Sequence[RoundSplit],
        gpu: Sequence[RoundGpu],
        *,
        rank: int,
        gpu_unread: int = 0,
    ) -> str:
        def mm(values) -> str:
            values = [v for v in values if v is not None]
            if not values:
                return "-"
            return "%.1f/%.1f" % med_max(values)

        def f(name):
            return mm([getattr(r, name) for r in window])

        def g(name):
            return mm([getattr(r, name) for r in gpu])

        n = len(window)
        return (
            "DECODE-HOST-SPLIT rank=%d n=%d last_round=%d wall_ms=%s sync_ms=%s "
            "host_work_ms=%s seq_wait_ms=%s ple_sync_ms=%s result_wait_ms=%s "
            "ctl_wait_ms=%s ctl_wait_n=%d recv_ms=%s sched_ms=%s hicache_ms=%s "
            "draft_ms=%s verify_ms=%s vprep_ms=%s ple_stage_ms=%s launch_ms=%s "
            "accept_ms=%s dext_ms=%s result_ms=%s bcast_ms=%s bcast_n=%d "
            "fetch_plan_ms=%s fetch_plan_n=%d other_ms=%s gpu_n=%d gpu_unread=%d "
            "gpu_draft_ms=%s gpu_gap_ple_ms=%s gpu_verify_ms=%s gpu_accept_ms=%s "
            "gpu_dext_ms=%s gpu_gap_round_ms=%s gpu_idle_ms=%s (med/max over n "
            "decode rounds; sync = seq_wait + ple_sync + result_wait + ctl_wait, "
            "the host blocked on the device; host_work = wall - sync; verify = "
            "vprep + ple_sync + ple_stage + launch + accept; other = wall - recv - "
            "sched - seq_wait - draft - verify - dext - result; bcast, ctl_wait, "
            "fetch_plan, hicache, result_wait are inside the spans; gpu_* from six "
            "stream events per round read late by query, gpu_idle = gap_round + "
            "gap_ple = device time the host cost)"
            % (
                rank,
                n,
                window[-1].round_id if n else -1,
                f("wall_ms"),
                f("sync_ms"),
                f("host_work_ms"),
                f("seq_wait_ms"),
                f("ple_sync_ms"),
                f("result_wait_ms"),
                f("ctl_wait_ms"),
                sum(r.ctl_wait_n for r in window),
                f("recv_ms"),
                f("sched_ms"),
                f("hicache_ms"),
                f("draft_ms"),
                f("verify_ms"),
                f("vprep_ms"),
                f("ple_stage_ms"),
                f("launch_ms"),
                f("accept_ms"),
                f("dext_ms"),
                f("result_ms"),
                f("bcast_ms"),
                sum(r.bcast_n for r in window),
                f("fetch_plan_ms"),
                sum(r.fetch_plan_n for r in window),
                f("other_ms"),
                len(gpu),
                int(gpu_unread),
                g("draft_ms"),
                g("gap_ple_ms"),
                g("verify_ms"),
                g("accept_ms"),
                g("dext_ms"),
                g("gap_round_ms"),
                g("idle_ms"),
            )
        )


_ACTIVE: Optional[DecodeHostSplit] = None


def register_active(split: Optional[DecodeHostSplit]) -> None:
    """The DecodeRoundLog's split is the one the forward-path marks reach."""
    global _ACTIVE
    _ACTIVE = split


def active_split() -> Optional[DecodeHostSplit]:
    return _ACTIVE


def mark(point: int) -> None:
    """Record decode mark ``point`` on the current stream (no-op when the
    split is off, outside a decode round, or under stream capture)."""
    split = _ACTIVE
    if split is not None:
        split.mark(point)
