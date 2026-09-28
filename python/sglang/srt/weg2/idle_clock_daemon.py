#!/usr/bin/env python3
"""#55 F2: host-side clock daemon for the front's idle clock lock.

WHY A DAEMON AND NOT THE FRONT ITSELF. On these GeForce cards an open CUDA
context keeps the card at P2/P1 with its full SM clock while nothing runs:
~200 W more for the rig while the model sits idle (PA, 28.09., 64 vram_*.csv
of NF). Locking the graphics clock to its floor during idle is the lever, but
the driver refuses ``nvidia-smi -lgc`` / ``nvmlDeviceSetGpuLockedClocks`` from
inside a container even as root (docker/README.htsglang.md, "Clock control
does not work from a container"). So the NVML write runs HERE, on the host,
and the front only says lock / unlock.

STANDALONE ON PURPOSE: stdlib + ctypes on ``libnvidia-ml.so.1``, no sglang
import, so the host's system python runs this file straight out of a tree:

    python3 idle_clock_daemon.py --listen tcp:172.17.0.1:8779 --cards all

EVENT-DRIVEN, NO POLLER: nothing here wakes on a timer. A request changes
state, and so does a connection closing.

FAIL-OPEN, the only safe direction. A card is locked only while at least one
connected client holds the lock. A client that sends ``unlock``, closes its
connection, or dies (the kernel closes the socket) releases it, and when the
last holder is gone every card is reset. The daemon also resets every card at
start and on SIGTERM/SIGINT. So a dead front, a dead network path or a killed
daemon all leave the cards unlocked -- the rig then runs exactly as it did
before this existed.

PROTOCOL, one JSON object per line each way:
    -> {"op": "lock"}   <- {"ok": true, "locked": true, "ms": 0.4, "cards": [...], "mhz": 210}
    -> {"op": "unlock"} <- {"ok": true, "locked": false, "ms": 0.3, ...}
    -> {"op": "status"} <- {"ok": true, "locked": ..., "holders": n, ...}
``ms`` is the NVML time of this op on this host, not the round trip.
"""

from __future__ import annotations

import argparse
import asyncio
import ctypes
import json
import logging
import os
import signal
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger("idle_clock_daemon")

DEFAULT_LISTEN = "tcp:172.17.0.1:8779"
FALLBACK_MHZ = 210


class NvmlError(RuntimeError):
    pass


class Nvml:
    """The five NVML calls this daemon needs, via ctypes (no pynvml on the host)."""

    def __init__(self, lib: str = "libnvidia-ml.so.1") -> None:
        self._l = ctypes.CDLL(lib)
        self._l.nvmlErrorString.restype = ctypes.c_char_p
        self._ck(self._l.nvmlInit_v2())

    def _ck(self, rc: int) -> None:
        if rc != 0:
            raise NvmlError(f"NVML rc={rc} {self._l.nvmlErrorString(rc).decode(errors='replace')}")

    def count(self) -> int:
        n = ctypes.c_uint()
        self._ck(self._l.nvmlDeviceGetCount_v2(ctypes.byref(n)))
        return int(n.value)

    def handle(self, index: int):
        h = ctypes.c_void_p()
        self._ck(self._l.nvmlDeviceGetHandleByIndex_v2(ctypes.c_uint(index), ctypes.byref(h)))
        return h

    def uuid(self, h) -> str:
        buf = ctypes.create_string_buffer(96)
        self._ck(self._l.nvmlDeviceGetUUID(h, buf, ctypes.c_uint(96)))
        return buf.value.decode()

    def min_graphics_mhz(self, h) -> int:
        """Lowest graphics clock supported at the highest memory clock."""
        n = ctypes.c_uint(32)
        mem = (ctypes.c_uint * 32)()
        self._ck(self._l.nvmlDeviceGetSupportedMemoryClocks(h, ctypes.byref(n), mem))
        top = max(mem[i] for i in range(n.value))
        n2 = ctypes.c_uint(512)
        gfx = (ctypes.c_uint * 512)()
        self._ck(self._l.nvmlDeviceGetSupportedGraphicsClocks(h, ctypes.c_uint(top), ctypes.byref(n2), gfx))
        return int(min(gfx[i] for i in range(n2.value)))

    def lock(self, h, lo: int, hi: int) -> None:
        self._ck(self._l.nvmlDeviceSetGpuLockedClocks(h, ctypes.c_uint(lo), ctypes.c_uint(hi)))

    def reset(self, h) -> None:
        self._ck(self._l.nvmlDeviceResetGpuLockedClocks(h))


class Card:
    def __init__(self, index: int, uuid: str, handle, mhz: int) -> None:
        self.index, self.uuid, self.handle, self.mhz = index, uuid, handle, mhz


def select_cards(nvml, spec: str, mhz_arg: str) -> List[Card]:
    """``spec``: ``all``, NVML indices ``0,2``, or UUIDs ``GPU-...,GPU-...``."""
    n = nvml.count()
    want = [s.strip() for s in spec.split(",") if s.strip()] if spec != "all" else None
    cards = []
    for i in range(n):
        h = nvml.handle(i)
        u = nvml.uuid(h)
        if want is not None and str(i) not in want and u not in want:
            continue
        if mhz_arg == "auto":
            try:
                mhz = nvml.min_graphics_mhz(h)
            except NvmlError as e:
                logger.warning("IDLE-CLOCK-D card=%d supported-clock query failed (%s), floor %d MHz",
                               i, e, FALLBACK_MHZ)
                mhz = FALLBACK_MHZ
        else:
            mhz = int(mhz_arg)
        cards.append(Card(i, u, h, mhz))
    if want is not None and len(cards) != len(want):
        raise SystemExit(f"--cards {spec}: matched {len(cards)} of {len(want)} cards")
    return cards


class ClockState:
    """Who holds the lock, and the NVML writes. No I/O besides NVML."""

    def __init__(self, nvml, cards: Sequence[Card], clock=time.perf_counter) -> None:
        self.nvml, self.cards, self._clock = nvml, list(cards), clock
        self.holders: set = set()
        self.locked = False

    def _reset_all(self) -> Tuple[float, List[str]]:
        t0 = self._clock()
        errs = []
        for c in self.cards:
            try:
                self.nvml.reset(c.handle)
            except NvmlError as e:
                errs.append(f"card{c.index}: {e}")
        self.locked = False
        return (self._clock() - t0) * 1e3, errs

    def _reply(self, ok: bool, ms: float, errs: List[str], op: str) -> Dict:
        r = {"ok": ok, "op": op, "locked": self.locked, "ms": round(ms, 3), "holders": len(self.holders),
             "cards": [c.index for c in self.cards], "mhz": [c.mhz for c in self.cards]}
        if errs:
            r["err"] = "; ".join(errs)
        return r

    def lock(self, who) -> Dict:
        self.holders.add(who)
        if self.locked:
            return self._reply(True, 0.0, [], "lock")
        t0 = self._clock()
        done: List[Card] = []
        try:
            for c in self.cards:
                self.nvml.lock(c.handle, c.mhz, c.mhz)
                done.append(c)
        except NvmlError as e:
            # FAIL-OPEN: a partial lock is undone, the holder is dropped, the answer is "not locked".
            for c in done:
                try:
                    self.nvml.reset(c.handle)
                except NvmlError:
                    pass
            self.holders.discard(who)
            self.locked = False
            return self._reply(False, (self._clock() - t0) * 1e3, [f"card{c.index}: {e}"], "lock")
        self.locked = True
        return self._reply(True, (self._clock() - t0) * 1e3, [], "lock")

    def release(self, who, op: str = "unlock") -> Dict:
        self.holders.discard(who)
        if self.holders or not self.locked:
            return self._reply(True, 0.0, [], op)
        ms, errs = self._reset_all()
        return self._reply(not errs, ms, errs, op)

    def status(self) -> Dict:
        return self._reply(True, 0.0, [], "status")


async def serve_conn(state: ClockState, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    who = object()
    peer = writer.get_extra_info("peername") or writer.get_extra_info("sockname")
    try:
        while True:
            line = await reader.readline()
            if not line:
                break
            try:
                op = json.loads(line).get("op")
            except (ValueError, AttributeError):
                op = None
            if op == "lock":
                r = state.lock(who)
            elif op == "unlock":
                r = state.release(who)
            elif op == "status":
                r = state.status()
            else:
                r = {"ok": False, "err": f"unknown op {op!r}"}
            if op in ("lock", "unlock"):
                logger.info("IDLE-CLOCK-D %s ok=%s locked=%s ms=%.3f holders=%d peer=%s%s", op, r.get("ok"),
                            r.get("locked"), r.get("ms", 0.0), r.get("holders", 0), peer,
                            f" err={r['err']}" if "err" in r else "")
            writer.write((json.dumps(r) + "\n").encode())
            await writer.drain()
    except (ConnectionError, asyncio.IncompleteReadError):
        pass
    finally:
        # The connection is the lease: gone -> released (fail-open).
        if who in state.holders:
            r = state.release(who, op="release-on-close")
            logger.info("IDLE-CLOCK-D release-on-close locked=%s ms=%.3f holders=%d peer=%s",
                        r["locked"], r["ms"], r["holders"], peer)
        try:
            writer.close()
        except Exception:  # noqa: BLE001
            pass


def parse_listen(spec: str) -> Tuple[str, str, int]:
    kind, _, rest = spec.partition(":")
    if kind == "unix":
        return "unix", rest, 0
    if kind == "tcp":
        host, _, port = rest.rpartition(":")
        return "tcp", host, int(port)
    raise SystemExit(f"--listen {spec!r}: want tcp:HOST:PORT or unix:/path")


async def amain(args) -> None:
    nvml = Nvml()
    state = ClockState(nvml, select_cards(nvml, args.cards, args.mhz))
    ms, errs = state._reset_all()
    logger.info("IDLE-CLOCK-D start cards=%s mhz=%s reset_ms=%.3f%s",
                [(c.index, c.uuid) for c in state.cards], [c.mhz for c in state.cards], ms,
                f" reset_err={errs}" if errs else "")
    servers = []
    for spec in args.listen:
        kind, host, port = parse_listen(spec)
        if kind == "unix":
            try:
                os.unlink(host)
            except FileNotFoundError:
                pass
            os.makedirs(os.path.dirname(host) or ".", exist_ok=True)
            servers.append(await asyncio.start_unix_server(lambda r, w: serve_conn(state, r, w), path=host))
        else:
            servers.append(await asyncio.start_server(lambda r, w: serve_conn(state, r, w), host, port))
        logger.info("IDLE-CLOCK-D listening %s", spec)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    await stop.wait()
    for s in servers:
        s.close()
    ms, errs = state._reset_all()
    logger.info("IDLE-CLOCK-D stop reset_ms=%.3f%s", ms, f" reset_err={errs}" if errs else "")


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--listen", action="append", default=None,
                    help=f"tcp:HOST:PORT or unix:/path, repeatable (default {DEFAULT_LISTEN})")
    ap.add_argument("--cards", default="all", help="all | NVML indices 0,2 | UUIDs")
    ap.add_argument("--mhz", default="auto",
                    help="locked graphics clock; auto = lowest supported at the top memory clock")
    args = ap.parse_args(argv)
    args.listen = args.listen or [DEFAULT_LISTEN]
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", stream=sys.stdout)
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
