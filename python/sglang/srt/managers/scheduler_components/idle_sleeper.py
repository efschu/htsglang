from typing import Tuple

import zmq

from sglang.srt.environ import envs
from sglang.srt.observability.req_time_stats import real_time
from sglang.srt.platforms import current_platform

# #547 step-up ladder. Read as: "after N consecutive idle ticks, poll for
# T milliseconds". The first rung is deliberately a ZERO-poll rung: while the
# server has only just gone idle the loop keeps its current spin behaviour
# byte-for-byte (no poll syscall at all), so a gap between two batches shorter
# than the first rung is indistinguishable from today. Only a loop that has
# been idle for the whole first rung starts to block.
#
#   ticks <=  32 : no poll        (~2 ms of real time -- legacy behaviour)
#   ticks <= 160 : poll   1 ms    (reached after ~2 ms idle)
#   ticks <= 288 : poll  10 ms    (reached after ~130 ms idle)
#   ticks >  288 : poll  50 ms    (reached after ~1.4 s idle, the cap)
#
# The cap is what bounds the cadence of everything the loop does per tick that
# is NOT woken by a socket (round counters, ladder samples, metrics): at the
# cap those still tick at ~20 Hz. It is deliberately far below the 1000 ms flat
# poll this class used before, because this fork has several per-tick consumers
# that a one-second park would visibly stall.
IDLE_POLL_LADDER: Tuple[Tuple[int, int], ...] = (
    (32, 0),
    (160, 1),
    (288, 10),
)
IDLE_POLL_CAP_MS = 50


def idle_poll_timeout_ms(idle_ticks: int) -> int:
    """Poll timeout in ms for the `idle_ticks`-th consecutive idle tick.

    `idle_ticks` is 1-based: the first idle tick after any work is tick 1.
    Returns 0 for "do not poll at all" (not "poll with a zero timeout") --
    the caller must skip the syscall so the legacy path stays byte-identical.
    """
    for threshold, timeout_ms in IDLE_POLL_LADDER:
        if idle_ticks <= threshold:
            return timeout_ms
    return IDLE_POLL_CAP_MS


class IdleSleeper:
    """
    In setups which have long inactivity periods it is desirable to reduce
    system power consumption when sglang does nothing. This would lead not only
    to power savings, but also to more CPU thermal headroom when a request
    eventually comes. This is important in cases when multiple GPUs are connected
    as each GPU would otherwise pin one thread at 100% CPU usage.

    The simplest solution is to use zmq.Poller on all sockets that may receive
    data that needs handling immediately.

    #547: the poll timeout is no longer a flat 1000 ms. It steps up along
    IDLE_POLL_LADDER, so the transition out of a loaded phase costs nothing and
    only a genuinely quiet loop parks. `reset()` puts the ladder back on its
    first rung and MUST be called whenever the scheduler had work in an
    iteration -- that is what makes "any running or queued batch => exact
    current behaviour" a property of the code rather than a hope.
    """

    def __init__(self, sockets):
        self.poller = zmq.Poller()
        self.last_empty_time = real_time()
        for s in sockets:
            self.poller.register(s, zmq.POLLIN)

        self.empty_cache_interval = envs.SGLANG_EMPTY_CACHE_INTERVAL.get()
        self.idle_ticks = 0

    def reset(self) -> None:
        """Drop back to the first (zero-poll) rung. Called from the loaded path."""
        self.idle_ticks = 0

    def maybe_sleep(self):
        self.idle_ticks += 1
        timeout_ms = idle_poll_timeout_ms(self.idle_ticks)
        if timeout_ms > 0:
            self.poller.poll(timeout_ms)
        if (
            self.empty_cache_interval > 0
            and real_time() - self.last_empty_time > self.empty_cache_interval
        ):
            self.last_empty_time = real_time()
            current_platform.empty_cache()
