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

DUTY_ENV = "FLLIPER_PDFLIP_DUAL_P_DUTY"
DBUSY_FILE_ENV = "FLLIPER_PDFLIP_DUAL_DBUSY_FILE"
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
