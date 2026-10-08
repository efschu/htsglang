"""D's own decode rounds for the Flipzeit (Nutzer 02.10. ~17:50Z, fuenfte Ruege):

  "letztes decode token wurde erzeugt ->(alles hier ist flipzeit)->erster chunk prefill -> letzer cunk prefill
   -> (alles hier ist flipzeit)->erstes decode token wurde erzeugt"

The D>P flip starts at the LAST decode token D really produced, the P>D flip ends at the FIRST one -- read from
D's own decode-round line, not from a front marker (park, flip_begin, arrival).  The rank files carry no time
per round (rankstats decode.rounds is a counter, sampled once a second: 1 s resolution on a ~2 s flip), so the
one exact source today is TP0's ``Decode rank batch, rank: 0, #round: N, t: <open>, ... gpu-ms: G`` line
(managers/scheduler_components/decode_round_log.py: ``t`` = the round's OPEN, epoch ms).  The token of a round
exists at its end, read as ``t + gpu-ms`` (device time; the real end is a few host-ms later, so a D>P start
read this way is never later than the real last token).

The line is parsed by the FROZEN ``parse.parse_line`` (DASHBOARD-AUS-IPC: no new log parser; test
test_no_new_log_parsers).  IPC follow-up: the rank writer stamps ``decode.last_t`` / ``decode.first_t_after_wake``
into rankstats, then this reader falls.

The log is tailed by byte offset, only lines carrying the TP0 marker are parsed, one reader per D log.
"""

from __future__ import annotations

import glob
import os
import threading
from typing import Dict, List, Optional, Tuple

from . import names as N
from . import parse

MARK = b"Decode rank batch, rank: 0,"
MAX_ROUNDS = 200000          # (open, end) pairs kept per log -- ~10 h of bs1 decode
MAX_READERS = 16
READ_CHUNK = 8 * 1024 * 1024
MANIFEST_ENV = "SGLANG_WEIGHT_LOADER_SHARED_CACHE_MANIFEST"   # .../evidence/<stem>.shared_cache, both models


def _log_path(ipc: dict, suffix: str, pattern: str) -> Optional[str]:
    """The boot's group log on this host: ``<line>/evidence/<stem><suffix>`` next to ``<line>/state/<boot_id>``.
    The stem comes from the launcher's shared-cache manifest path (D or P env), else from the launcher tag."""
    d = ipc.get("dir")
    if not d:
        return None
    ev = os.path.join(os.path.dirname(os.path.dirname(os.path.normpath(d))), "evidence")
    for g in ("D", "P"):
        env = (((ipc.get("launch") or {}).get(g) or {}).get("env") or {})
        m = N.env_get(env, MANIFEST_ENV)
        if m and m.endswith(".shared_cache"):
            p = os.path.join(ev, os.path.basename(m)[:-len(".shared_cache")] + suffix)
            if os.path.exists(p):
                return p
    tag = ipc.get("tag")
    if tag:
        # F0-B: logs of the old tree are ``boot_<old>_<tag>_*``, of a renamed tree ``boot_<new>_<tag>_*``
        cands = [c for pat in N.glob_variants(pattern) for c in glob.glob(os.path.join(ev, pat % tag))]
        if cands:
            return max(cands, key=lambda p: os.path.getmtime(p))
    return None


def d_log_path(ipc: dict) -> Optional[str]:
    """The boot's D log on this host (``<stem>.D.log``)."""
    return _log_path(ipc, ".D.log", "boot_weg2_%s_*.D.log")


def front_log_path(ipc: dict) -> Optional[str]:
    """The boot's front log on this host (``<stem>.front.log``)."""
    return _log_path(ipc, ".front.log", "boot_weg2_%s_*.front.log")


class DecodeRounds:
    """(open, end) of every TP0 decode round of one D log, in log order."""

    def __init__(self, path: str):
        self.path = path
        self.off = 0
        self.tail = b""
        self.rounds: List[Tuple[float, float]] = []
        self.lock = threading.Lock()

    def poll(self) -> List[Tuple[float, float]]:
        with self.lock:
            try:
                size = os.path.getsize(self.path)
            except OSError:
                return self.rounds
            if size < self.off:                       # rewritten: start over
                self.off, self.tail, self.rounds = 0, b"", []
            if size > self.off:
                with open(self.path, "rb") as fh:
                    fh.seek(self.off)
                    while self.off < size:
                        buf = fh.read(min(READ_CHUNK, size - self.off))
                        if not buf:
                            break
                        self.off += len(buf)
                        lines = (self.tail + buf).split(b"\n")
                        self.tail = lines.pop()
                        for ln in lines:
                            if MARK in ln:
                                self._add(ln)
                if len(self.rounds) > MAX_ROUNDS:
                    del self.rounds[:len(self.rounds) - MAX_ROUNDS]
            return self.rounds

    def _add(self, ln: bytes) -> None:
        ev = parse.parse_line(ln.decode("utf-8", "replace"))
        if not ev or ev.get("kind") != "decode_rank" or ev.get("rk") != "TP" or ev.get("rank") != 0:
            return
        t = ev.get("t_exact")
        if t is None:
            return
        end = t + (ev.get("gpu_ms") or 0.0) / 1000.0
        if self.rounds and t < self.rounds[-1][0]:
            return                                    # a line out of order is never a later round
        self.rounds.append((t, end))


_READERS: Dict[str, DecodeRounds] = {}
_LOCK = threading.Lock()


def decode_rounds(ipc: dict) -> Optional[List[Tuple[float, float]]]:
    """The boot's TP0 decode rounds as (open, end), oldest first; None when its D log is not found here."""
    p = d_log_path(ipc)
    if p is None:
        return None
    with _LOCK:
        r = _READERS.get(p)
        if r is None:
            if len(_READERS) >= MAX_READERS:
                _READERS.pop(next(iter(_READERS)))        # the oldest reader goes
            r = _READERS[p] = DecodeRounds(p)
    return list(r.poll())


# Arrival of a request at the front (Nutzer 02.10. ~18:25Z via NF: the D>P Vorlauf is split into "leer ohne Request"
# (D's last token -> arrival of the request that needs P), "halt" (arrival -> park RPC / flip_begin) and "park").
# The front stamps ``WEG2 SESSION rid=<rid>`` on every request it accepts, before pricing; the IPC event
# flip_user_time names the rid that triggered the flip but carries no arrival stamp (only the park-RPC send, or the
# pricing verdict as ``oldest_waiter_arrival``).  Only the line prefix (frozen parse.RE_PREFIX / parse_ts) and the
# rid token are read.  IPC follow-up: flip_user_time.arrival_ts at the front, then this reader falls.
SESSION_MARK = b"WEG2 SESSION rid="
SESSION_MARKS = N.bytes_variants("WEG2 SESSION rid=")      # F0-B: the old and the renamed spelling
MAX_ARRIVALS = 200000         # rids kept per log (first stamp wins)


class FrontArrivals:
    """rid -> first ``WEG2 SESSION`` time of one front log."""

    def __init__(self, path: str):
        self.path = path
        self.off = 0
        self.tail = b""
        self.first: Dict[str, float] = {}
        self.lock = threading.Lock()

    def poll(self) -> Dict[str, float]:
        with self.lock:
            try:
                size = os.path.getsize(self.path)
            except OSError:
                return self.first
            if size < self.off:                       # rewritten: start over
                self.off, self.tail, self.first = 0, b"", {}
            if size > self.off:
                with open(self.path, "rb") as fh:
                    fh.seek(self.off)
                    while self.off < size:
                        buf = fh.read(min(READ_CHUNK, size - self.off))
                        if not buf:
                            break
                        self.off += len(buf)
                        lines = (self.tail + buf).split(b"\n")
                        self.tail = lines.pop()
                        for ln in lines:
                            if any(m in ln for m in SESSION_MARKS):
                                self._add(ln)
            return self.first

    def _add(self, ln: bytes) -> None:
        txt = ln.decode("utf-8", "replace")
        m = parse.RE_PREFIX.match(txt)
        if not m:
            return
        tail = N.marker_tail(txt, "WEG2 SESSION rid=")
        if tail is None:
            return
        rid = tail.split(" ", 1)[0].strip()
        if rid and rid not in self.first and len(self.first) < MAX_ARRIVALS:
            self.first[rid] = parse.parse_ts(m)


_FRONT: Dict[str, FrontArrivals] = {}


def front_arrivals(ipc: dict) -> Optional[Dict[str, float]]:
    """The boot's request arrivals at the front, rid -> epoch s; None when its front log is not found here."""
    p = front_log_path(ipc)
    if p is None:
        return None
    with _LOCK:
        r = _FRONT.get(p)
        if r is None:
            if len(_FRONT) >= MAX_READERS:
                _FRONT.pop(next(iter(_FRONT)))
            r = _FRONT[p] = FrontArrivals(p)
    return dict(r.poll())
