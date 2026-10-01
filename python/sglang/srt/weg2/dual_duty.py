"""DUAL-TP3PP3: P's duty cycle while D decodes -- the latency guard without MPS.

MEASURED 29.09. (risk-1 bench, scripts/dual_layout): a bandwidth-bound decode
step and a compute-bound prefill share a card almost zero-sum (E 0.87-1.05).
Unbounded, the prefill takes ~90 % of the card and D's step runs ~7x slower;
the driver's time slice (no MPS) splits ~50/50; MPS with a P SM share is a
knob but needs MPS. This is the knob WITHOUT MPS: while D holds decodes, the
first P stage idles after each forward for ``t_fwd * (1 - duty) / duty``, so
P computes at most ``duty`` of the wall time and D has the card alone for the
rest. When D is idle P runs flat out.

The signal is one byte in a file the front rewrites when D's outstanding set
turns empty / non-empty (``DBusyWriter``); PP0 reads it at most every
``READ_EVERY_S`` (``DutyThrottle``). Pure stdlib; off unless both env
variables are set.
"""

from __future__ import annotations

import os
import time
from typing import Callable, Optional

DUTY_ENV = "SGLANG_WEG2_DUAL_P_DUTY"
DBUSY_FILE_ENV = "SGLANG_WEG2_DUAL_DBUSY_FILE"
READ_EVERY_S = 0.02
#: A single sleep never exceeds this; a stuck D signal must not stall P.
MAX_SLEEP_S = 0.5


def dbusy_path(tag: str) -> str:
    import hashlib

    return f"/dev/shm/wdb-{hashlib.sha1(str(tag).encode()).hexdigest()[:10]}"


class DBusyWriter:
    """Front side: write '1' / '0' only when the state changes (atomic rename)."""

    def __init__(self, path: str):
        self.path = path
        self._last: Optional[bool] = None

    def update(self, busy: bool) -> bool:
        busy = bool(busy)
        if busy == self._last:
            return False
        tmp = f"{self.path}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            f.write("1" if busy else "0")
        os.replace(tmp, self.path)
        self._last = busy
        return True


class DutyThrottle:
    """P stage 0 side. ``before_forward()`` sleeps when D is busy, sized by the
    previous forward (``after_forward(seconds)``)."""

    def __init__(self, duty: float, path: str, clock: Callable[[], float] = time.perf_counter,
                 sleep: Callable[[float], None] = time.sleep):
        if not (0.0 < duty <= 1.0):
            raise ValueError(f"{DUTY_ENV} must be in (0, 1], got {duty}")
        self.duty = float(duty)
        self.path = path
        self._clock = clock
        self._sleep = sleep
        self._busy = False
        self._read_t = -1e9
        self._last_fwd_s = 0.0
        self.slept_s = 0.0
        self.throttled = 0

    @classmethod
    def from_env(cls, env=None) -> Optional["DutyThrottle"]:
        env = os.environ if env is None else env
        raw, path = env.get(DUTY_ENV, "").strip(), env.get(DBUSY_FILE_ENV, "").strip()
        if not raw or not path:
            return None
        duty = float(raw)
        return None if duty >= 1.0 else cls(duty, path)

    def d_busy(self) -> bool:
        now = self._clock()
        if now - self._read_t >= READ_EVERY_S:
            self._read_t = now
            try:
                with open(self.path) as f:
                    self._busy = f.read(1) == "1"
            except OSError:
                self._busy = False  # no signal = no throttle (never stall P on a missing file)
        return self._busy

    def pause_s(self) -> float:
        if self._last_fwd_s <= 0.0 or not self.d_busy():
            return 0.0
        return min(MAX_SLEEP_S, self._last_fwd_s * (1.0 - self.duty) / self.duty)

    def before_forward(self) -> float:
        s = self.pause_s()
        if s > 0.0:
            self._sleep(s)
            self.slept_s += s
            self.throttled += 1
        return s

    def after_forward(self, seconds: float) -> None:
        self._last_fwd_s = max(0.0, float(seconds))


# --------------------------------------------------------------------------
# DUAL-TP3PP3 gang window (01.10., variant C1): the duty throttle above idles
# only PP0, so the other P stages keep computing and D -- a TP3 step that
# needs all three cards for every collective -- is never alone on the rig
# (MEASURED dual1g ...09301639: D round 32.8 -> 146 ms p50 while P was held
# at 65 % of its contended bound; P 38 % + D 23 % of solo, E ~0.6).
# The gate instead runs P in BURSTS while D is busy: PP0 launches K chunks
# back to back, then launches nothing until the LAST stage reports the burst
# done (pipeline empty on every card), then holds for
# burst_wall * (1 - duty) / duty (capped) so D runs alone on all cards.
# D is never paused (no ITL stall); it runs time-sliced during a burst and
# free during the hold. Off unless GANG_ENV is set next to the duty envs.
GANG_ENV = "SGLANG_WEG2_DUAL_P_GANG_CHUNKS"
GANG_POLL_S = 0.002
GANG_MAX_DRAIN_S = 2.0
GANG_MAX_HOLD_S = 2.5


def gang_done_path(busy_path: str) -> str:
    return f"{busy_path}.gang"


def gang_chunks_from_env(env=None) -> int:
    env = os.environ if env is None else env
    raw = env.get(GANG_ENV, "").strip()
    try:
        k = int(raw) if raw else 0
    except ValueError:
        k = 0
    return k if k >= 1 else 0


class GangDonePublisher:
    """LAST P stage: one ``submit(wait_fn)`` per forward; a daemon thread runs
    ``wait_fn`` (the forward's completion wait, e.g. ``event.synchronize``) in
    order and then publishes the count of completed forwards to ``path``.
    The forward loop itself never waits."""

    def __init__(self, path: str, start_thread: bool = True):
        import queue
        import threading

        self.path = path
        self.done = 0
        self._q = queue.Queue()
        self._publish(0)
        if start_thread:
            threading.Thread(target=self._run, name="dual-gang-done", daemon=True).start()

    def _publish(self, n: int) -> None:
        tmp = f"{self.path}.{os.getpid()}.tmp"
        try:
            with open(tmp, "w") as f:
                f.write(str(n))
            os.replace(tmp, self.path)
        except OSError:
            pass  # PP0 times out its drain wait (bounded) -- never stall the last stage

    def step(self, wait_fn) -> None:
        try:
            wait_fn()
        finally:
            self.done += 1
            self._publish(self.done)

    def _run(self) -> None:
        while True:
            self.step(self._q.get())

    def submit(self, wait_fn) -> None:
        self._q.put(wait_fn)


class GangGate(DutyThrottle):
    """FIRST P stage: same busy signal as DutyThrottle, burst/drain/hold
    instead of a sleep after every forward. Counts its own launches; the last
    stage's published count reaching it means the pipeline is empty. A count
    that does not arrive within GANG_MAX_DRAIN_S re-bases (never a long stall)."""

    def __init__(self, duty: float, path: str, chunks: int, done_path: Optional[str] = None,
                 clock: Callable[[], float] = time.perf_counter,
                 sleep: Callable[[float], None] = time.sleep):
        super().__init__(duty, path, clock=clock, sleep=sleep)
        if chunks < 1:
            raise ValueError(f"{GANG_ENV} must be >= 1, got {chunks}")
        self.chunks = int(chunks)
        self.done_path = done_path or gang_done_path(path)
        self.launched = 0
        self._offset = 0
        self._burst_n = 0
        self._burst_t0 = 0.0
        self.bursts = 0
        self.held_s = 0.0
        self.drain_s = 0.0
        self.rebased = 0
        self.blind = False
        self._last_timeout_done: Optional[int] = -1

    @classmethod
    def from_env(cls, env=None) -> Optional["GangGate"]:
        env = os.environ if env is None else env
        k = gang_chunks_from_env(env)
        base = DutyThrottle.from_env(env)
        if base is None or k < 1:
            return None
        return cls(base.duty, base.path, k)

    def read_done(self) -> Optional[int]:
        try:
            with open(self.done_path) as f:
                return int(f.read().strip() or "0")
        except (OSError, ValueError):
            return None

    def _drained(self) -> bool:
        d = self.read_done()
        return d is not None and d + self._offset >= self.launched

    def _launch(self) -> None:
        self.launched += 1
        if self._burst_n == 0:
            self._burst_t0 = self._clock()
        self._burst_n += 1

    def before_forward(self) -> float:
        if not self.d_busy():
            self._burst_n = 0
            self._launch()
            return 0.0
        if self._burst_n < self.chunks:
            self._launch()
            return 0.0
        t0 = self._clock()
        while not self.blind and not self._drained():
            if self._clock() - t0 >= GANG_MAX_DRAIN_S:
                d = self.read_done()
                if d is None or d == self._last_timeout_done:
                    self.blind = True  # no completions arrive at all: hold by time only
                else:
                    self._offset = self.launched - d  # re-base: counts drifted
                self._last_timeout_done = d
                self.rebased += 1
                break
            if not self.d_busy():
                break
            self._sleep(GANG_POLL_S)
        t1 = self._clock()
        self.drain_s += t1 - t0
        burst_wall = max(0.0, t1 - self._burst_t0)
        hold = min(GANG_MAX_HOLD_S, burst_wall * (1.0 - self.duty) / self.duty)
        while self.d_busy():
            left = hold - (self._clock() - t1)
            if left <= 1e-4:
                break
            self._sleep(min(READ_EVERY_S, left))
        t2 = self._clock()
        self.held_s += t2 - t1
        self.bursts += 1
        self.throttled += 1
        self.slept_s += t2 - t0
        self._burst_n = 0
        self._launch()
        return t2 - t0
