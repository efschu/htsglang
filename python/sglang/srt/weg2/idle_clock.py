"""#55 F2: lock the cards' graphics clock after 1 s of idle in the default layout.

THE COST THIS REMOVES. With a CUDA context open, the driver keeps these
GeForce cards at P2/P1 with their full SM clock while nothing runs: 121-126 W
per 3080 and ~80 W on the 5090 at util 0, against 35-50 W / 25-28 W with no
process -- ~200 W for the rig while the model sits idle (PA 28.09., all 64
NF vram_*.csv since 26.09.). Sleeping ranks keep their contexts, so the cards
never reach P8 on their own.

THE LEVER, user order 28.09.: "warum erst nach 60s ? warum nicht nach 1s
leerlauf im default layout, das 'hochtakten' wird doch nur ms brauchen??".
So: 1 s after the front comes to rest in its configured idle layout, the
graphics clock is locked to its floor; the first sign of work unlocks it
BEFORE that work proceeds.

WHERE "IDLE" COMES FROM -- the front, never GPU load. Three facts together:
  * no POST in flight through the front (counted by an aiohttp middleware,
    so every generate/abort/flip request counts from its first byte),
  * the controller's own idle decision says REST (``Front._idle_disposition``:
    awake group IS ``--idle-layout``, nothing queued, handed off or outstanding),
  * no flip open (``state == "serving"``) and nothing queued.
GPU utilisation is deliberately not an input: during a flip the SLEEPING
group's legs run on the cards too, and a load reading per card would see a
quiet card and lock it under a flip.

UNLOCK BEFORE THE FIRST GPU WORK. The memory-clock unlock costs ~30 ms
(3080 micro test 28.09.: -lmc 405 -> P8 at 38.7 W, -67 W; unlock ~30 ms,
ramp to full clock ~33 ms, first round x0.996). On the REQUEST path that
unlock is STARTED at the request's first byte (middleware, in a worker
thread) and AWAITED only where the front first reaches a group -- an aiohttp
``on_request_start`` trace hook on the front's group session, which aiohttp
awaits before it opens the connection. Tokenizing, pricing and routing run
beside the unlock. Before a FLIP it stays synchronous (``before_flip``), and
a controller pass that is not at rest unlocks synchronously too (the safety
net). An unlock that fails closes the connection, which is itself the
release (the daemon resets on close).

FAIL-OPEN. The switch is off by default. With it on and no daemon, the lock
simply never happens (one ``UNAVAILABLE`` line, a retry after ``retry_s``);
the clocks stay as they are today. See ``idle_clock_daemon.py`` for the host
side and its lease semantics.

Env: SGLANG_WEG2_IDLE_CLOCK=1 (off by default), SGLANG_WEG2_IDLE_CLOCK_S=1.0,
SGLANG_WEG2_IDLE_CLOCK_ADDR=tcp:172.17.0.1:8779 (docker0 on the host) or
unix:/path, SGLANG_WEG2_IDLE_CLOCK_TIMEOUT_S=0.25.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import threading
import time
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("weg2.front")

ENV_ON = "SGLANG_WEG2_IDLE_CLOCK"
ENV_IDLE_S = "SGLANG_WEG2_IDLE_CLOCK_S"
ENV_ADDR = "SGLANG_WEG2_IDLE_CLOCK_ADDR"
ENV_TIMEOUT_S = "SGLANG_WEG2_IDLE_CLOCK_TIMEOUT_S"
DEFAULT_ADDR = "tcp:172.17.0.1:8779"
DEFAULT_IDLE_S = 1.0
DEFAULT_TIMEOUT_S = 0.25
RETRY_S = 30.0


class DaemonClient:
    """One persistent connection to the host daemon; the connection is the lease."""

    def __init__(self, addr: str, timeout_s: float) -> None:
        self.addr, self.timeout_s = addr, timeout_s
        self._sock: Optional[socket.socket] = None
        self._buf = b""

    @property
    def connected(self) -> bool:
        return self._sock is not None

    def _connect(self) -> None:
        kind, _, rest = self.addr.partition(":")
        if kind == "unix":
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.settimeout(self.timeout_s)
            s.connect(rest)
        elif kind == "tcp":
            host, _, port = rest.rpartition(":")
            s = socket.create_connection((host, int(port)), timeout=self.timeout_s)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        else:
            raise ValueError(f"{ENV_ADDR}={self.addr!r}: want tcp:HOST:PORT or unix:/path")
        self._sock, self._buf = s, b""

    def call(self, op: str, **extra) -> Dict:
        if self._sock is None:
            self._connect()
        assert self._sock is not None
        self._sock.sendall((json.dumps({"op": op, **extra}) + "\n").encode())
        while b"\n" not in self._buf:
            chunk = self._sock.recv(4096)
            if not chunk:
                raise ConnectionError("idle-clock daemon closed the connection")
            self._buf += chunk
        line, _, self._buf = self._buf.partition(b"\n")
        return json.loads(line)

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
        self._sock, self._buf = None, b""


class IdleClock:
    def __init__(self, client, idle_s: float = DEFAULT_IDLE_S, retry_s: float = RETRY_S,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.client, self.idle_s, self.retry_s, self._clock = client, float(idle_s), float(retry_s), clock
        self.inflight = 0
        self.rest_since: Optional[float] = None
        self.locked = False
        self._down_until = 0.0
        self._down_logged = False
        self.counters: Dict[str, int] = {"lock": 0, "unlock": 0, "unavailable": 0, "unlock_via_close": 0,
                                         "gate_waits": 0}
        # the async unlock started at a request's first byte; awaited by ready() / before_flip()
        self._pending: Optional["asyncio.Future"] = None
        self._io = threading.Lock()  # one daemon call at a time (loop thread vs. unlock worker thread)

    def _unlock_pending(self) -> bool:
        return self._pending is not None and not self._pending.done()

    # -- the front hooks ------------------------------------------------------
    def enter(self) -> None:
        """A POST reaches the front: count it and START the unlock, without waiting for it."""
        self.inflight += 1
        self.rest_since = None
        if self.locked and not self._unlock_pending():
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                self._unlock("request")  # no loop (unit use): synchronous
                return
            self._pending = loop.run_in_executor(None, self._unlock, "request")

    async def ready(self) -> None:
        """Awaited before the front's first request to a group: the unlock must have landed."""
        p = self._pending
        if p is None or p.done():
            return
        t0 = time.perf_counter()
        await asyncio.shield(p)
        self.counters["gate_waits"] += 1
        logger.info("WEG2 IDLE-CLOCK GATE waited_ms=%.2f (the first group request waited for the unlock "
                    "started at the request's first byte)", (time.perf_counter() - t0) * 1e3)

    async def before_flip(self) -> None:
        """A flip: unlock synchronously -- both groups' legs use the cards."""
        self.rest_since = None
        if self._unlock_pending():
            await asyncio.shield(self._pending)
        elif self.locked:
            self._unlock("flip")

    def leave(self) -> None:
        self.inflight = max(0, self.inflight - 1)

    def note_rest(self, queued: int = 0, serving: bool = True) -> None:
        """The controller's idle decision said REST in the configured layout."""
        if self._unlock_pending():
            return
        if self.inflight or queued or not serving:
            self.note_busy("request" if self.inflight else "queued" if queued else "not-serving")
            return
        now = self._clock()
        if self.rest_since is None:
            self.rest_since = now
            return
        if self.locked or now - self.rest_since < self.idle_s or now < self._down_until:
            return
        self._lock(now - self.rest_since)

    def note_busy(self, why: str) -> None:
        """Controller safety net: not at rest -> unlocked, synchronously unless a request already started it."""
        self.rest_since = None
        if self.locked and not self._unlock_pending():
            self._unlock(why)

    # -- the daemon calls -----------------------------------------------------
    def _lock(self, rest_s: float) -> None:
        t0 = time.perf_counter()
        try:
            with self._io:
                r = self.client.call("lock")
        except (OSError, ValueError) as e:
            self._unavailable(f"{type(e).__name__}: {e}")
            return
        rtt_ms = (time.perf_counter() - t0) * 1e3
        if not r.get("ok"):
            self.client.close()
            self._unavailable(f"daemon refused: {r.get('err', r)}")
            return
        self.locked, self._down_logged = True, False
        self.counters["lock"] += 1
        logger.info("WEG2 IDLE-CLOCK LOCK rest_s=%.2f idle_s=%.2f rtt_ms=%.2f daemon_ms=%.3f cards=%s mhz=%s "
                    "mem=%s mem_locked=%s mem_mhz=%s%s (the front rested in its idle layout with nothing in "
                    "flight; the clocks are held at their floor until the next request or flip)",
                    rest_s, self.idle_s, rtt_ms, float(r.get("ms", 0.0)), r.get("cards"), r.get("mhz"),
                    r.get("mem_mode", "off"), r.get("mem_locked", False), r.get("mem_mhz"),
                    f" mem_err={r['mem_err']}" if r.get("mem_err") else "")

    def _unlock(self, why: str) -> None:
        t0 = time.perf_counter()
        via = "unlock"
        daemon_ms = None
        try:
            with self._io:
                r = self.client.call("unlock")
            daemon_ms = r.get("ms")
            if not r.get("ok"):
                raise ConnectionError(f"daemon unlock not ok: {r.get('err', r)}")
        except (OSError, ValueError) as e:
            # The connection is the lease: closing it IS the release (daemon resets on close).
            with self._io:
                self.client.close()
            via = f"close ({type(e).__name__})"
            self.counters["unlock_via_close"] += 1
        self.locked = False
        self.counters["unlock"] += 1
        logger.info("WEG2 IDLE-CLOCK UNLOCK why=%s via=%s rtt_ms=%.2f daemon_ms=%s",
                    why, via, (time.perf_counter() - t0) * 1e3,
                    "-" if daemon_ms is None else f"{float(daemon_ms):.3f}")

    def _unavailable(self, detail: str) -> None:
        self.counters["unavailable"] += 1
        self._down_until = self._clock() + self.retry_s
        self.locked = False
        if not self._down_logged:
            self._down_logged = True
            logger.warning("WEG2 IDLE-CLOCK UNAVAILABLE addr=%s detail=%s -- fail-open: clocks untouched, "
                           "next try in %.0f s", getattr(self.client, "addr", "?"), detail, self.retry_s)


def _on(v: Optional[str]) -> bool:
    return (v or "").strip().lower() in ("1", "true", "on", "yes")


def from_env(env=None) -> Optional[IdleClock]:
    """``None`` unless SGLANG_WEG2_IDLE_CLOCK is on (the default is OFF)."""
    env = os.environ if env is None else env
    if not _on(env.get(ENV_ON)):
        return None
    idle_s = float(env.get(ENV_IDLE_S) or DEFAULT_IDLE_S)
    timeout_s = float(env.get(ENV_TIMEOUT_S) or DEFAULT_TIMEOUT_S)
    addr = env.get(ENV_ADDR) or DEFAULT_ADDR
    logger.info("WEG2 IDLE-CLOCK ARMED idle_s=%.2f addr=%s timeout_s=%.2f (#55 F2: lock after idle in the "
                "configured layout, unlock before every request and flip; fail-open without the daemon)",
                idle_s, addr, timeout_s)
    return IdleClock(DaemonClient(addr, timeout_s), idle_s=idle_s)


def middlewares(ic: Optional[IdleClock]) -> List:
    """The in-flight count: every POST through the front, from its first byte to its handler's end."""
    if ic is None:
        return []
    from aiohttp import web

    @web.middleware
    async def idle_clock_inflight(request, handler):
        if request.method != "POST":
            return await handler(request)
        ic.enter()
        try:
            return await handler(request)
        finally:
            ic.leave()

    return [idle_clock_inflight]


def trace_configs(ic: Optional[IdleClock]) -> List:
    """The request-path gate: aiohttp awaits ``on_request_start`` before it connects, so every request
    the front sends to a group waits for an unlock started at the client request's first byte."""
    if ic is None:
        return []
    from aiohttp import TraceConfig

    async def _gate(session, ctx, params):
        await ic.ready()

    tc = TraceConfig()
    tc.on_request_start.append(_gate)
    return [tc]
