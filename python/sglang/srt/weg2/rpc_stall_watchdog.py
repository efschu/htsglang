# SPDX-License-Identifier: Apache-2.0
"""RPC-STALL-WATCHDOG (30.09., hauenh P->D epoch 6): all thread stacks of a sleep / wake RPC that
stalls, written into a file of its own per rank.

MEASURED NEED: boot hauenh (dkr27browauthorityls12bar1fs09301342), the P->D flip at 13:46:09: D TP0
was silent from 13:46:11 to 13:46:17 -- no line from ANY thread -- inside the wake RPC's #1443 hold
release (WEG2-WAKE-TAIL store_rescan=6048 ms). Modelled on NF's TAG-STALL-SENTINEL (14df988ba7).

THE SAMPLER IS PYTHON, UNDER THE GIL (z30y8, 30.09. 18:41:55): the first form called
``faulthandler.dump_traceback_later``, whose C thread walks every other thread's frames WITHOUT the GIL.
On z30y8 P's PP1 it fired at 1790793715.525 while the main thread was inside ``logging.emit``; the dump
file stops in the middle of that thread's stack, PP1's last log line is 715.527, and the rank died by
SIGSEGV (W17 for the group). A lock-free walk over a frame that is changing is best effort by design.
Now a daemon thread waits on an event with the timeout and, when it expires, reads
``sys._current_frames()`` and formats each stack -- plain Python, holding the GIL, so no frame changes
under it. The price, named: a stall that HOLDS the GIL in C (the hauenh shape was one candidate) delays
the dump until the GIL is released; the file then shows where it went next, and the log line still
names the stall. ``faulthandler.enable()`` stays for real fatal signals only.

A second dump follows at +``REPEAT_S`` (8 s) when the RPC still runs (the implementer's proposal):
two snapshots show whether a stall moves. Both land in the same file.

:func:`watched` wraps a handler: before it, :func:`arm` opens
``<evidence>/weg2_rpcstall_<group>_r<rank>_<kind>_<unix_ms>_p<pid>.txt`` with one header line and starts the
sampler; after it (also on a raise), :func:`disarm` stops the sampler (bounded join, never a hang) and
reads the file size: grown past the header = fired -> one ``RPC-STALL-WATCHDOG fired`` line naming
rank, kind and file; not grown -> the file is removed.

``SGLANG_WEG2_RPC_STALL_WATCHDOG_S`` (default 3.0: a normal wake/sleep leg is 1.5-2.3 s on this rig)
is the timeout; 0 = off (no file, no thread).
"""

from __future__ import annotations

import functools
import logging
import os
import re
import time
from typing import Callable, Optional

from sglang.srt.weg2 import stall_sampler as _ss

logger = logging.getLogger(__name__)

MARKER = "RPC-STALL-WATCHDOG"
DEFAULT_TIMEOUT_S = 3.0
#: the second dump, this long after the first one, when the RPC still runs
REPEAT_S = _ss.REPEAT_S
#: disarm waits at most this long for a sampler that is writing (never a hang)
JOIN_S = _ss.JOIN_S


class Armed:
    __slots__ = ("fh", "path", "header_len", "kind", "rank", "group", "t0", "sampler")

    def __init__(self, fh, path: str, header_len: int, kind: str, rank, group: str, t0: float):
        self.fh = fh
        self.path = path
        self.header_len = header_len
        self.kind = kind
        self.rank = rank
        self.group = group
        self.t0 = t0
        self.sampler = None


def timeout_s() -> float:
    try:
        from sglang.srt.environ import envs

        v = float(envs.SGLANG_WEG2_RPC_STALL_WATCHDOG_S.get() or 0.0)
    except Exception:  # noqa: BLE001 -- an instrument never breaks a leg
        return 0.0
    return v if v > 0.0 else 0.0


def dump_dir() -> str:
    return os.environ.get("SGLANG_WEG2_EVIDENCE_DIR") or "/spinning/evidence-665-f1"


def _slug(value) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", str(value)) or "x"


def arm(kind: str, *, rank, group: str, directory: Optional[str] = None,
        timeout: Optional[float] = None) -> Optional[Armed]:
    """Before one RPC: open its dump file and arm the C watchdog. ``None`` when off or when the
    file cannot be opened (named once per call)."""
    t = timeout_s() if timeout is None else float(timeout)
    if t <= 0.0:
        return None
    try:
        d = directory or dump_dir()
        os.makedirs(d, exist_ok=True)
        now = time.time()
        # RPCSTALL-NAME (z30y10, D 20:03:51): every rank of a group arms the SAME kind in the SAME
        # millisecond; without a rank (``r_``) and without the pid the three processes opened ONE
        # path, and the first disarm unlinked the file under the other two (FileNotFoundError on
        # TP0/TP1). The pid makes the path this process's own whatever the rank reads.
        path = os.path.join(d, "weg2_rpcstall_%s_r%s_%s_%d_p%d.txt" % (
            _slug(group), _slug(rank), _slug(kind), int(now * 1000), os.getpid()))
        fh = open(path, "w")
        header = "%s group=%s rank=%s kind=%s armed_unix=%.3f timeout_s=%.3f pid=%d\n" % (
            MARKER, group, rank, kind, now, t, os.getpid())
        fh.write(header)
        fh.flush()
        armed = Armed(fh, path, len(header.encode()), str(kind), rank, str(group), time.perf_counter())
        armed.sampler = _ss.arm(fh, t, repeat_s=REPEAT_S, name="rpc-stall-sampler")
        return armed
    except Exception as exc:  # noqa: BLE001 -- an instrument never breaks a leg
        logger.warning("%s not armed for %s: %s: %s", MARKER, kind, type(exc).__name__, exc)
        return None


def disarm(armed: Optional[Armed]) -> Optional[str]:
    """After the RPC: cancel the watchdog; the dump path when it fired (logged once), else remove
    the header-only file and return ``None``."""
    if armed is None:
        return None
    try:
        _ss.disarm(armed.sampler, JOIN_S)      # a sampler mid-write finishes; never an unbounded wait
        with armed.sampler.lock:
            armed.fh.flush()
            size = os.fstat(armed.fh.fileno()).st_size
            armed.fh.close()
        if size > armed.header_len:
            logger.warning(
                "%s fired group=%s rank=%s kind=%s file=%s bytes=%d rpc_ms=%.0f -- the RPC outlived the "
                "watchdog; the file holds every thread's stack at the timeout (Python sampler under the "
                "GIL; a second dump at +%.0f s if it still ran)", MARKER, armed.group, armed.rank, armed.kind, armed.path,
                size, (time.perf_counter() - armed.t0) * 1000, REPEAT_S)
            return armed.path
        try:
            os.unlink(armed.path)
        except FileNotFoundError:
            pass                                # already gone: the end state disarm wants
    except Exception as exc:  # noqa: BLE001 -- an instrument never breaks a leg
        logger.warning("%s disarm for %s: %s: %s", MARKER, armed.kind, type(exc).__name__, exc)
    return None


def rank_of(scheduler):
    """THIS process's rank for the dump name: the world group's ``rank_in_group``, else the flat
    ``ps`` rank (``pp_rank * tp_size + tp_rank``), else ``tp_rank``, else ``"?"``.

    RPCSTALL-NAME (z30y10): the hold-release call site read ``getattr(self, "tp_rank", "?")`` -- the
    Scheduler has no ``tp_rank`` (its identity lives on ``world_group`` / ``ps``, see
    ``SchedulerWeightUpdaterManager._weg2_rank``), so every rank armed ``r?`` -> ``r_``."""
    try:
        r = getattr(getattr(scheduler, "world_group", None), "rank_in_group", None)
        if isinstance(r, int):
            return r
    except Exception:  # noqa: BLE001
        pass
    try:
        ps = scheduler.ps
        return int(ps.pp_rank) * int(ps.tp_size) + int(ps.tp_rank)
    except Exception:  # noqa: BLE001
        pass
    r = getattr(scheduler, "tp_rank", None)
    return r if isinstance(r, int) else "?"


def _who(obj) -> tuple:
    rank, group = "?", "?"
    try:
        rank = obj._weg2_rank()
    except Exception:  # noqa: BLE001
        rank = rank_of(getattr(obj, "scheduler", obj))
    try:
        group = obj._weg2_group_name()
    except Exception:  # noqa: BLE001
        group = os.environ.get("SGLANG_WEG2_GROUP", "") or "?"
    return rank, group


def watched(kind: str) -> Callable:
    """Decorator for a sleep / wake RPC handler method: armed before, disarmed after (also on a
    raise). The handler's return value and exceptions pass through unchanged."""

    def deco(fn):
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            rank, group = _who(self)
            armed = arm(kind, rank=rank, group=group)
            try:
                return fn(self, *args, **kwargs)
            finally:
                disarm(armed)

        return wrapper

    return deco
