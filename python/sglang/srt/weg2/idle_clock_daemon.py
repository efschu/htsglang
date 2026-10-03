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
    -> {"op": "lock"}   <- {"ok": true, "locked": true, "ms": 0.4, "cards": [...], "mhz": [210],
                            "mem_mode": "lock", "mem_locked": true, "mem_mhz": [405]}
       optional {"op": "lock", "mem": "off"|"lock"|"app"} overrides --mem for this lock
    -> {"op": "unlock"} <- {"ok": true, "locked": false, "ms": 0.3, ...}
    -> {"op": "status"} <- {"ok": true, "locked": ..., "holders": n, ...}
``ms`` is the NVML time of this op on this host, not the round trip.
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
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


NVML_NOT_SUPPORTED = 3
NVML_NO_PERMISSION = 4

#: Memory-clock modes. The 3080 test (28.09., idle_clock_micro_0_0928_130702) showed the SM lock
#: alone saves only 12 W (109.9 -> 98.3 W, still P2) because the memory clock keeps running at
#: 9501 MHz; without a context the card sits at P8 / ~40 W. So the memory clock is the lever:
#:   off  -- graphics clock only (the first version);
#:   lock -- nvmlDeviceSetMemoryLockedClocks(min, min) (= nvidia-smi -lmc);
#:   app  -- nvmlDeviceSetApplicationsClocks(mem_min, gfx at mem_min), the older interface, as a
#:           fallback where -lmc is refused.
MEM_MODES = ("off", "lock", "app")


class NvmlError(RuntimeError):
    def __init__(self, msg: str, rc: int = -1) -> None:
        super().__init__(msg)
        self.rc = rc


class Nvml:
    """The NVML calls this daemon needs, via ctypes (no pynvml on the host)."""

    def __init__(self, lib: str = "libnvidia-ml.so.1") -> None:
        self._l = ctypes.CDLL(lib)
        self._l.nvmlErrorString.restype = ctypes.c_char_p
        self._ck(self._l.nvmlInit_v2())

    def _ck(self, rc: int) -> None:
        if rc != 0:
            raise NvmlError(f"NVML rc={rc} {self._l.nvmlErrorString(rc).decode(errors='replace')}", rc)

    def _fn(self, name: str):
        try:
            return getattr(self._l, name)
        except AttributeError:
            raise NvmlError(f"NVML symbol {name} missing in this driver", NVML_NOT_SUPPORTED) from None

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

    def mem_clocks(self, h) -> List[int]:
        n = ctypes.c_uint(32)
        mem = (ctypes.c_uint * 32)()
        self._ck(self._l.nvmlDeviceGetSupportedMemoryClocks(h, ctypes.byref(n), mem))
        return sorted(int(mem[i]) for i in range(n.value))

    def gfx_clocks(self, h, mem_mhz: int) -> List[int]:
        n = ctypes.c_uint(512)
        gfx = (ctypes.c_uint * 512)()
        self._ck(self._l.nvmlDeviceGetSupportedGraphicsClocks(h, ctypes.c_uint(mem_mhz), ctypes.byref(n), gfx))
        return sorted(int(gfx[i]) for i in range(n.value))

    def min_graphics_mhz(self, h) -> int:
        """Lowest graphics clock supported at the highest memory clock."""
        return self.gfx_clocks(h, self.mem_clocks(h)[-1])[0]

    def lock(self, h, lo: int, hi: int) -> None:
        self._ck(self._fn("nvmlDeviceSetGpuLockedClocks")(h, ctypes.c_uint(lo), ctypes.c_uint(hi)))

    def reset(self, h) -> None:
        self._ck(self._fn("nvmlDeviceResetGpuLockedClocks")(h))

    def lock_mem(self, h, lo: int, hi: int) -> None:
        self._ck(self._fn("nvmlDeviceSetMemoryLockedClocks")(h, ctypes.c_uint(lo), ctypes.c_uint(hi)))

    def reset_mem(self, h) -> None:
        self._ck(self._fn("nvmlDeviceResetMemoryLockedClocks")(h))

    def set_app(self, h, mem_mhz: int, gfx_mhz: int) -> None:
        self._ck(self._fn("nvmlDeviceSetApplicationsClocks")(h, ctypes.c_uint(mem_mhz), ctypes.c_uint(gfx_mhz)))

    def reset_app(self, h) -> None:
        self._ck(self._fn("nvmlDeviceResetApplicationsClocks")(h))


class Card:
    def __init__(self, index: int, uuid: str, handle, mhz: int, mem_min: int = 0, gfx_at_mem_min: int = 0) -> None:
        self.index, self.uuid, self.handle, self.mhz = index, uuid, handle, mhz
        self.mem_min, self.gfx_at_mem_min = mem_min, gfx_at_mem_min


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
        mem_min = gfx_at_mem_min = 0
        try:
            mems = nvml.mem_clocks(h)
            mem_min = mems[0]
            gfx_at_mem_min = nvml.gfx_clocks(h, mem_min)[0]
            mhz = nvml.gfx_clocks(h, mems[-1])[0]
        except NvmlError as e:
            logger.warning("IDLE-CLOCK-D card=%d supported-clock query failed (%s), floor %d MHz",
                           i, e, FALLBACK_MHZ)
            mhz = FALLBACK_MHZ
        if mhz_arg != "auto":
            mhz = int(mhz_arg)
        cards.append(Card(i, u, h, mhz, mem_min, gfx_at_mem_min))
    if want is not None and len(cards) != len(want):
        raise SystemExit(f"--cards {spec}: matched {len(cards)} of {len(want)} cards")
    return cards


class ClockState:
    """Who holds the lock, and the NVML writes. No I/O besides NVML.

    The GRAPHICS lock is the lock: if it fails the whole lock is undone (fail-open). The MEMORY step
    (``mem`` mode) is best effort on top of it: refused -> the graphics lock stays, the reply says
    ``mem_locked: false`` and why, so the front's lock line and the micro test name it.
    """

    def __init__(self, nvml, cards: Sequence[Card], clock=time.perf_counter, mem: str = "off") -> None:
        if mem not in MEM_MODES:
            raise ValueError(f"mem mode {mem!r} not in {MEM_MODES}")
        self.nvml, self.cards, self._clock, self.default_mem = nvml, list(cards), clock, mem
        self.holders: set = set()
        self.locked = False
        self.mem_mode = "off"      # mode of the lock currently held
        self.mem_locked = False
        self.mem_err: List[str] = []
        # a memory mode may still sit on a card although no lock says so (a failed undo of a partial
        # memory lock, a refused reset of the held one): the next request-path unlock resets mem AND
        # app as the full reset did -- the fast path holds only while nothing is left over
        self.mem_dirty = False
        self.card_ms: List[float] = []  # NVML ms per card of the last lock / reset (they run at once)
        self._pool: Optional[concurrent.futures.ThreadPoolExecutor] = None

    def _per_card(self, cards: Sequence[Card], work) -> List[Tuple[Card, Optional[NvmlError], float]]:
        """``work(card)`` on every card AT ONCE, one thread per card; (card, error, ms) in card order.

        NF 28.09. (rc12z26): the unlock took 98-107 ms on the request path, the serial sum of
        3 cards x 3 resets; the 3080 micro test put one card's unlock at ~30 ms. NVML is
        thread-safe and ctypes drops the GIL for the call, so the cards' writes overlap.
        """
        def one(c):
            t0 = self._clock()
            try:
                work(c)
                return c, None, (self._clock() - t0) * 1e3
            except NvmlError as e:
                return c, e, (self._clock() - t0) * 1e3
        if len(cards) <= 1:
            return [one(c) for c in cards]
        if self._pool is None:
            self._pool = concurrent.futures.ThreadPoolExecutor(len(self.cards), thread_name_prefix="idle-clock")
        return list(self._pool.map(one, cards))

    def _reset_all(self, only_set: bool = False) -> Tuple[float, List[str]]:
        """Reset what this daemon can set, all cards at once.

        ``only_set=False`` (start, stop): EVERYTHING, whatever it believes it set.
        ``only_set=True`` (the last holder leaves, i.e. the request path): the graphics lock plus the
        memory mode of the lock it holds -- a mode never set costs a driver call per card for nothing
        (the app reset in ``mem=lock`` mode was a third of the unlock's calls).
        """
        t0 = self._clock()
        modes = [("gfx", self.nvml.reset), ("mem", getattr(self.nvml, "reset_mem", None)),
                 ("app", getattr(self.nvml, "reset_app", None))]
        # the reset that undoes the memory mode of the lock held now (mem_mode "lock" -> reset "mem")
        held = {"lock": "mem", "app": "app"}.get(self.mem_mode) if self.locked and self.mem_locked else None
        if only_set and not self.mem_dirty:
            modes = [m for m in modes if m[0] in ("gfx", held)]
        errs: List[str] = []

        def work(c):
            for name, fn in modes:
                if fn is None:
                    continue
                try:
                    fn(c.handle)
                except NvmlError as e:
                    # A mode never used on this card may be unsupported; only the gfx reset must work.
                    if name in ("gfx", held) or e.rc not in (NVML_NOT_SUPPORTED, NVML_NO_PERMISSION):
                        errs.append(f"card{c.index} {name}: {e}")

        self.card_ms = [round(ms, 3) for _, _, ms in self._per_card(self.cards, work)]
        self.locked, self.mem_locked, self.mem_mode = False, False, "off"
        # a refused mem/app reset may have left the mode on the card: the next unlock tries again
        self.mem_dirty = any(" gfx:" not in e for e in errs)
        return (self._clock() - t0) * 1e3, sorted(errs)

    def _reply(self, ok: bool, ms: float, errs: List[str], op: str) -> Dict:
        r = {"ok": ok, "op": op, "locked": self.locked, "ms": round(ms, 3), "holders": len(self.holders),
             "cards": [c.index for c in self.cards], "mhz": [c.mhz for c in self.cards],
             "mem_mode": self.mem_mode, "mem_locked": self.mem_locked,
             "mem_mhz": [c.mem_min for c in self.cards] if self.mem_locked else None,
             "card_ms": list(self.card_ms) if ms else []}
        if self.mem_err and self.locked:
            r["mem_err"] = "; ".join(self.mem_err)
        if errs:
            r["err"] = "; ".join(errs)
        return r

    def _lock_mem(self, mode: str) -> List[str]:
        def work(c):
            if not c.mem_min:
                raise NvmlError("no supported memory clock known", NVML_NOT_SUPPORTED)
            if mode == "lock":
                self.nvml.lock_mem(c.handle, c.mem_min, c.mem_min)
            else:
                self.nvml.set_app(c.handle, c.mem_min, c.gfx_at_mem_min or c.mhz)
        res = self._per_card(self.cards, work)
        errs = [f"card{c.index} {mode}: {e}" for c, e, _ in res if e is not None]
        if errs:  # all or nothing for the memory step too
            undo = self.nvml.reset_mem if mode == "lock" else self.nvml.reset_app
            undone = self._per_card([c for c, e, _ in res if e is None], lambda c: undo(c.handle))
            if any(e is not None for _, e, _ in undone):
                # the undo itself failed: that card may still hold the mode -- never forget it
                self.mem_dirty = True
                errs += [f"card{c.index} {mode}-undo: {e}" for c, e, _ in undone if e is not None]
        self.card_ms = [round(a + ms, 3) for a, (_, _, ms) in zip(self.card_ms, res)]
        return errs

    def lock(self, who, mem: Optional[str] = None) -> Dict:
        mode = self.default_mem if mem is None else mem
        if mode not in MEM_MODES:
            return {"ok": False, "op": "lock", "err": f"mem mode {mode!r} not in {MEM_MODES}"}
        self.holders.add(who)
        if self.locked:
            return self._reply(True, 0.0, [], "lock")
        t0 = self._clock()
        res = self._per_card(self.cards, lambda c: self.nvml.lock(c.handle, c.mhz, c.mhz))
        self.card_ms = [round(ms, 3) for _, _, ms in res]
        bad = [f"card{c.index}: {e}" for c, e, _ in res if e is not None]
        if bad:
            # FAIL-OPEN: a partial lock is undone, the holder is dropped, the answer is "not locked".
            self._per_card([c for c, e, _ in res if e is None], lambda c: self.nvml.reset(c.handle))
            self.holders.discard(who)
            self.locked = False
            return self._reply(False, (self._clock() - t0) * 1e3, bad, "lock")
        self.locked, self.mem_err = True, []
        if mode != "off":
            self.mem_err = self._lock_mem(mode)
            self.mem_locked = not self.mem_err
            self.mem_mode = mode if self.mem_locked else "off"
        return self._reply(True, (self._clock() - t0) * 1e3, [], "lock")

    def release(self, who, op: str = "unlock") -> Dict:
        self.holders.discard(who)
        if self.holders or not self.locked:
            return self._reply(True, 0.0, [], op)
        ms, errs = self._reset_all(only_set=True)
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
                msg = json.loads(line)
                op, mem = msg.get("op"), msg.get("mem")
            except (ValueError, AttributeError):
                op, mem = None, None
            if op == "lock":
                r = state.lock(who, mem)
            elif op == "unlock":
                r = state.release(who)
            elif op == "status":
                r = state.status()
            else:
                r = {"ok": False, "err": f"unknown op {op!r}"}
            if op in ("lock", "unlock"):
                logger.info("IDLE-CLOCK-D %s ok=%s locked=%s mem=%s/%s ms=%.3f card_ms=%s holders=%d peer=%s%s%s",
                            op, r.get("ok"), r.get("locked"), r.get("mem_mode"), r.get("mem_locked"),
                            r.get("ms", 0.0), r.get("card_ms"), r.get("holders", 0), peer,
                            f" err={r['err']}" if "err" in r else "",
                            f" mem_err={r['mem_err']}" if "mem_err" in r else "")
            writer.write((json.dumps(r) + "\n").encode())
            await writer.drain()
    except (ConnectionError, asyncio.IncompleteReadError):
        pass
    finally:
        # The connection is the lease: gone -> released (fail-open).
        if who in state.holders:
            r = state.release(who, op="release-on-close")
            logger.info("IDLE-CLOCK-D release-on-close locked=%s ms=%.3f card_ms=%s holders=%d peer=%s",
                        r["locked"], r["ms"], r.get("card_ms"), r["holders"], peer)
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
    state = ClockState(nvml, select_cards(nvml, args.cards, args.mhz), mem=args.mem)
    ms, errs = state._reset_all()
    logger.info("IDLE-CLOCK-D start cards=%s mhz=%s mem=%s mem_min=%s gfx_at_mem_min=%s reset_ms=%.3f card_ms=%s%s",
                [(c.index, c.uuid) for c in state.cards], [c.mhz for c in state.cards], args.mem,
                [c.mem_min for c in state.cards], [c.gfx_at_mem_min for c in state.cards], ms, state.card_ms,
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
    logger.info("IDLE-CLOCK-D stop reset_ms=%.3f card_ms=%s%s", ms, state.card_ms,
                f" reset_err={errs}" if errs else "")


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--listen", action="append", default=None,
                    help=f"tcp:HOST:PORT or unix:/path, repeatable (default {DEFAULT_LISTEN})")
    ap.add_argument("--cards", default="all", help="all | NVML indices 0,2 | UUIDs")
    ap.add_argument("--mhz", default="auto",
                    help="locked graphics clock; auto = lowest supported at the top memory clock")
    ap.add_argument("--mem", default="off", choices=MEM_MODES,
                    help="memory clock while locked: off | lock (-lmc to the lowest supported) | app "
                         "(application clocks, fallback); a lock request may override it with {\"mem\": ...}")
    args = ap.parse_args(argv)
    args.listen = args.listen or [DEFAULT_LISTEN]
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", stream=sys.stdout)
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
