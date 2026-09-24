"""fnFL2 H49: the scheduler HOST share of a decode round and of a PP pass.

WHY. The 27B line found three host paths that sit between two forwards on
the scheduler thread (24.09.): the HiCache poll of every round
(``check_hicache_events`` -> ``writing_check`` / ``loading_check`` /
``drain_storage_control_queues`` with its CPU ``all_reduce`` of the queue
sizes over the attention-TP group), the synchronous per-chunk publish on P,
and the PP stage's ``d2h_event.synchronize`` before the next chunk's host
work. ``DECODE-ROUND-COST`` prices the DEVICE side of a round (gpu_ms and
its families) and its host WALL, but not what the host did in that wall.
This module is the missing half, read off the sites that already run: a
``perf_counter`` around existing calls, no collective, no device sync, no
allocation on the device.

THE DECODE HALF (``DecodeHostCost``). A round's host interval is the same
interval ``DECODE-ROUND-COST wall_ms`` prices: from this round's open (the
run_batch funnel, ``DecodeRoundLog.begin_round``) to the next round's open,
or to the non-decode batch / idle tick that closed it
(``DecodeRoundLog.end_round``). Inside it:

* ``result_wait_ms`` -- the overlap loop's ``copy_done.synchronize`` in
  ``process_batch_result_decode`` (the host blocked on the PREVIOUS round's
  device result; GPU time, not host work);
* ``host_ms`` = ``wall_ms - result_wait_ms`` -- everything the scheduler
  thread did (launch, result processing, recv, scheduling, HiCache poll);
* ``hicache_ms`` -- ``check_hicache_events`` wall, every call in the
  interval, of which ``drain_ms`` / ``writing_ms`` / ``loading_ms`` are the
  named parts;
* ``hc_allreduce_ms`` / ``hc_allreduce_n`` -- EVERY HiCache CPU collective
  in the interval (``_all_reduce_attn_groups``: the drain's queue-size
  reduce and any prefetch-progress / group-max reduce that ran), wall
  including the wait on the slowest rank; ``drain_ar_ms`` is the drain's own;
* ``host_other_ms`` = ``host_ms - hicache_ms`` -- the event loop's remaining
  host work between two forwards.

THE PP HALF (``PPHostPeriod``). Per stage: ``sync_wait_ms`` is the
``d2h_event.synchronize`` of the slot's forward, ``host_work_ms`` the wall
from that sync's return to this stage's NEXT launch (the pass's remaining
host work, the next pass's recv/scheduling -- upstream recv waits included,
named so), ``launch_to_sync_ms`` the host work that runs while the launched
forward is queued. A gap that crosses a pass which launched nothing is
counted as ``starved`` and not timed (queue starvation is not host work).

COST. Nine float adds per HiCache poll, one per result sync, one tuple per
round open. A period line every ``SGLANG_DEBUG_DECODE_HOST_PERIOD`` rounds
(0 = none) and every ``SGLANG_DEBUG_PP_HOST_PERIOD`` synced chunks (0 = none).
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import List, Optional, Sequence, Tuple

import msgspec

logger = logging.getLogger(__name__)

__all__ = [
    "COUNTERS",
    "DecodeHostCost",
    "HostCostCounters",
    "PPHostPeriod",
    "RoundHost",
    "med_max",
    "note_result_wait",
    "pp_host_period",
]


class HostCostCounters:
    """Process-wide, monotone host-cost counters (ms). Written by the sites
    that already run; read as deltas at every decode round open. One
    process is one rank, so a module singleton is per rank."""

    __slots__ = (
        "hc_calls",
        "hc_ms",
        "drain_ms",
        "drain_ar_ms",
        "writing_ms",
        "loading_ms",
        "ar_ms",
        "ar_n",
        "result_wait_ms",
    )

    def __init__(self) -> None:
        for name in self.__slots__:
            setattr(self, name, 0.0)

    def snapshot(self) -> Tuple[float, ...]:
        return tuple(getattr(self, name) for name in self.__slots__)


COUNTERS = HostCostCounters()
_IDX = {name: i for i, name in enumerate(HostCostCounters.__slots__)}


def note_result_wait(ms: float) -> None:
    """The overlap loop's result sync (``copy_done.synchronize``) took ``ms``."""
    COUNTERS.result_wait_ms += ms


def med_max(values: Sequence[float]) -> Tuple[float, float]:
    """Median (upper median for an even count, a value that occurred) and max."""
    if not values:
        return 0.0, 0.0
    s = sorted(values)
    return float(s[len(s) // 2]), float(s[-1])


class RoundHost(msgspec.Struct, frozen=True, kw_only=True):
    """The host side of one decode round's interval (see module docstring)."""

    round_id: int
    wall_ms: float
    result_wait_ms: float
    hicache_ms: float
    hicache_calls: int
    drain_ms: float
    drain_ar_ms: float
    writing_ms: float
    loading_ms: float
    hc_allreduce_ms: float
    hc_allreduce_n: int

    @property
    def host_ms(self) -> float:
        return max(self.wall_ms - self.result_wait_ms, 0.0)

    @property
    def host_other_ms(self) -> float:
        return max(self.host_ms - self.hicache_ms, 0.0)


class DecodeHostCost:
    """Closes one ``RoundHost`` per decode round and states a period line."""

    #: Rounds kept for the DECODE-ROUND-COST join (a round is emitted a few
    #: rounds after it closed, when its device events are readable).
    KEEP: int = 64

    def __init__(
        self,
        *,
        rank: int,
        period: int,
        counters: HostCostCounters = COUNTERS,
    ) -> None:
        self._rank = int(rank)
        self._period = max(0, int(period))
        self._counters = counters
        self._open_id: Optional[int] = None
        self._open_t = 0.0
        self._snap: Tuple[float, ...] = ()
        self._recent: "OrderedDict[int, RoundHost]" = OrderedDict()
        self._window: List[RoundHost] = []

    @classmethod
    def from_env(cls, *, rank: int) -> "DecodeHostCost":
        from sglang.srt.environ import envs

        return cls(rank=rank, period=envs.SGLANG_DEBUG_DECODE_HOST_PERIOD.get())

    # -- round boundary (called by DecodeRoundLog) ------------------------

    def on_round_open(self, *, round_id: int, mono: float) -> None:
        """Round ``round_id`` opened at ``mono`` (perf_counter seconds): close
        the previous interval there and start this one."""
        self._close(mono)
        self._open_id = int(round_id)
        self._open_t = float(mono)
        self._snap = self._counters.snapshot()

    def on_round_end(self, *, mono: float) -> None:
        """A non-decode batch or an idle tick: the open interval ends here."""
        if self._open_id is not None:
            self._close(mono)

    def round_host(self, round_id: int) -> Optional[RoundHost]:
        return self._recent.get(int(round_id))

    # -- internals --------------------------------------------------------

    def _close(self, mono: float) -> None:
        if self._open_id is None:
            return
        now = self._counters.snapshot()
        d = [a - b for a, b in zip(now, self._snap)]
        rh = RoundHost(
            round_id=self._open_id,
            wall_ms=1000.0 * max(float(mono) - self._open_t, 0.0),
            result_wait_ms=d[_IDX["result_wait_ms"]],
            hicache_ms=d[_IDX["hc_ms"]],
            hicache_calls=int(d[_IDX["hc_calls"]]),
            drain_ms=d[_IDX["drain_ms"]],
            drain_ar_ms=d[_IDX["drain_ar_ms"]],
            writing_ms=d[_IDX["writing_ms"]],
            loading_ms=d[_IDX["loading_ms"]],
            hc_allreduce_ms=d[_IDX["ar_ms"]],
            hc_allreduce_n=int(d[_IDX["ar_n"]]),
        )
        self._open_id = None
        self._recent[rh.round_id] = rh
        while len(self._recent) > self.KEEP:
            self._recent.popitem(last=False)
        if self._period > 0:
            self._window.append(rh)
            if len(self._window) >= self._period:
                logger.info(self.period_line(self._window, rank=self._rank))
                self._window = []

    @staticmethod
    def period_line(window: Sequence[RoundHost], *, rank: int) -> str:
        def mm(values):
            return "%.1f/%.1f" % med_max(values)

        n = len(window)
        return (
            "DECODE-HOST-PERIOD rank=%d n=%d last_round=%d wall_ms=%s host_ms=%s "
            "result_wait_ms=%s hicache_ms=%s drain_ms=%s allreduce_ms=%s "
            "allreduce_n=%d writing_ms=%s loading_ms=%s host_other_ms=%s "
            "hicache_calls=%d (med/max over n decode rounds; host = wall - "
            "result_wait; host_other = host - hicache; allreduce = every "
            "HiCache CPU collective in the round)"
            % (
                rank,
                n,
                window[-1].round_id if n else -1,
                mm([r.wall_ms for r in window]),
                mm([r.host_ms for r in window]),
                mm([r.result_wait_ms for r in window]),
                mm([r.hicache_ms for r in window]),
                mm([r.drain_ms for r in window]),
                mm([r.hc_allreduce_ms for r in window]),
                sum(r.hc_allreduce_n for r in window),
                mm([r.writing_ms for r in window]),
                mm([r.loading_ms for r in window]),
                mm([r.host_other_ms for r in window]),
                sum(r.hicache_calls for r in window),
            )
        )


class PPHostPeriod:
    """Per PP stage: sync wait vs host work around ``d2h_event.synchronize``."""

    def __init__(self, *, every: int) -> None:
        self._every = max(0, int(every))
        self._last_launch: Optional[float] = None
        self._last_sync_exit: Optional[float] = None
        self._starved_since_sync = False
        self._pending_host: Optional[float] = None
        self._waits: List[float] = []
        self._hosts: List[float] = []
        self._l2s: List[float] = []
        self._starved = 0
        self._n_total = 0

    @classmethod
    def from_env(cls) -> "PPHostPeriod":
        from sglang.srt.environ import envs

        return cls(every=envs.SGLANG_DEBUG_PP_HOST_PERIOD.get())

    @property
    def on(self) -> bool:
        return self._every > 0

    def note_no_batch(self) -> None:
        """This pass launched no forward on this stage."""
        if self._last_sync_exit is not None:
            self._starved_since_sync = True

    def note_launch(self, mono: float) -> None:
        if self._last_sync_exit is not None:
            if self._starved_since_sync:
                self._starved += 1
            else:
                self._hosts.append(1000.0 * (mono - self._last_sync_exit))
            self._last_sync_exit = None
            self._starved_since_sync = False
        self._last_launch = mono

    def note_sync(self, *, stage: int, entry: float, exit: float) -> Optional[str]:
        """One ``d2h_event.synchronize`` of a slot's result; returns (and
        logs) the period line when ``every`` syncs are collected."""
        if self._every <= 0:
            return None
        self._waits.append(1000.0 * max(exit - entry, 0.0))
        if self._last_launch is not None:
            self._l2s.append(1000.0 * max(entry - self._last_launch, 0.0))
            self._last_launch = None
        self._last_sync_exit = exit
        self._starved_since_sync = False
        self._n_total += 1
        if len(self._waits) < self._every:
            return None
        line = self.period_line(stage=stage)
        logger.info(line)
        self._waits, self._hosts, self._l2s, self._starved = [], [], [], 0
        return line

    def period_line(self, *, stage: int) -> str:
        def mm(values):
            return "%.1f/%.1f" % med_max(values) if values else "-"

        return (
            "PP-HOST-PERIOD stage=%d n=%d total=%d sync_wait_ms=%s host_work_ms=%s "
            "launch_to_sync_ms=%s host_work_n=%d starved=%d (med/max; sync_wait = "
            "d2h_event.synchronize of the slot's forward; host_work = sync return "
            "-> this stage's next launch, upstream recv waits included; "
            "launch_to_sync = host work while the launched forward is queued; "
            "starved = gaps across a pass without a launch, not timed)"
            % (
                stage,
                len(self._waits),
                self._n_total,
                mm(self._waits),
                mm(self._hosts),
                mm(self._l2s),
                len(self._hosts),
                self._starved,
            )
        )


_PP_HOST_PERIOD: Optional[PPHostPeriod] = None


def pp_host_period() -> PPHostPeriod:
    """The per-process (= per-stage) PP period, built from the env once."""
    global _PP_HOST_PERIOD
    if _PP_HOST_PERIOD is None:
        _PP_HOST_PERIOD = PPHostPeriod.from_env()
    return _PP_HOST_PERIOD
