# SPDX-License-Identifier: Apache-2.0
"""RPC-STALL-WATCHDOG (30.09., hauenh P->D epoch 6): all thread stacks of a sleep / wake RPC that
stalls, written by faulthandler's own C watchdog thread into a file of its own per rank.

MEASURED NEED: boot hauenh (dkr27browauthorityls12bar1fs09301342), the P->D flip at 13:46:09: D TP0
was silent from 13:46:11 to 13:46:17 -- no line from ANY thread, while TP1/TP2 wrote their periodic
census at :13 -- inside the wake RPC's #1443 hold release (WEG2-WAKE-TAIL store_rescan=6048 ms,
dispatch_ms=6158 on every rank). No instrument named the frame: a Python-side sampler needs the GIL
such a stall holds. ``faulthandler.dump_traceback_later`` runs its timer in a C thread and writes every
thread's stack without the GIL. Modelled on NF's TAG-STALL-SENTINEL (14df988ba7, per sleep tag); here
per RPC -- the release (sleep) and resume (wake) handlers of BOTH groups.

:func:`watched` wraps a handler: before it, :func:`arm` opens
``<evidence>/weg2_rpcstall_<group>_r<rank>_<kind>_<unix_ms>.txt`` with one header line and arms the
timer; after it (also on a raise), :func:`disarm` cancels the timer and reads the file SIZE (fstat, no
device, no host sync): grown past the header = the watchdog fired -> one ``RPC-STALL-WATCHDOG fired``
line naming rank, kind and file; not grown -> the file is removed. Nothing goes into the boot log but
that one line.

``SGLANG_WEG2_RPC_STALL_WATCHDOG_S`` (default 3.0: a normal wake/sleep leg is 1.5-2.3 s on this rig)
is the timeout; 0 = off (no file, no timer). Cost without a stall: one small file create/unlink and two
C calls per RPC. faulthandler keeps ONE later-dump per process: the handlers are never nested.
"""

from __future__ import annotations

import faulthandler
import functools
import logging
import os
import re
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)

MARKER = "RPC-STALL-WATCHDOG"
DEFAULT_TIMEOUT_S = 3.0


class Armed:
    __slots__ = ("fh", "path", "header_len", "kind", "rank", "group", "t0")

    def __init__(self, fh, path: str, header_len: int, kind: str, rank, group: str, t0: float):
        self.fh = fh
        self.path = path
        self.header_len = header_len
        self.kind = kind
        self.rank = rank
        self.group = group
        self.t0 = t0


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
        path = os.path.join(d, "weg2_rpcstall_%s_r%s_%s_%d.txt" % (
            _slug(group), _slug(rank), _slug(kind), int(now * 1000)))
        fh = open(path, "w")
        header = "%s group=%s rank=%s kind=%s armed_unix=%.3f timeout_s=%.3f pid=%d\n" % (
            MARKER, group, rank, kind, now, t, os.getpid())
        fh.write(header)
        fh.flush()
        faulthandler.dump_traceback_later(t, repeat=False, file=fh, exit=False)
        return Armed(fh, path, len(header.encode()), str(kind), rank, str(group), time.perf_counter())
    except Exception as exc:  # noqa: BLE001 -- an instrument never breaks a leg
        logger.warning("%s not armed for %s: %s: %s", MARKER, kind, type(exc).__name__, exc)
        return None


def disarm(armed: Optional[Armed]) -> Optional[str]:
    """After the RPC: cancel the watchdog; the dump path when it fired (logged once), else remove
    the header-only file and return ``None``."""
    if armed is None:
        return None
    try:
        faulthandler.cancel_dump_traceback_later()
        armed.fh.flush()
        size = os.fstat(armed.fh.fileno()).st_size
        armed.fh.close()
        if size > armed.header_len:
            logger.warning(
                "%s fired group=%s rank=%s kind=%s file=%s bytes=%d rpc_ms=%.0f -- the RPC outlived the "
                "watchdog; the file holds every thread's stack at the timeout (faulthandler C thread, "
                "written without the GIL)", MARKER, armed.group, armed.rank, armed.kind, armed.path,
                size, (time.perf_counter() - armed.t0) * 1000)
            return armed.path
        os.unlink(armed.path)
    except Exception as exc:  # noqa: BLE001 -- an instrument never breaks a leg
        logger.warning("%s disarm for %s: %s: %s", MARKER, armed.kind, type(exc).__name__, exc)
    return None


def _who(obj) -> tuple:
    rank, group = "?", "?"
    try:
        rank = obj._weg2_rank()
    except Exception:  # noqa: BLE001
        pass
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
