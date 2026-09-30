# SPDX-License-Identifier: Apache-2.0
"""TAG-STALL-SENTINEL (30.09., NF y3z ep52): all thread stacks of a sleep tag
that stalls, written by faulthandler's own C watchdog thread.

NF y3z ep52: PP0 stood still for 5.4 s process-wide -- every thread at once --
at the first tag of P's sleep. Candidates: a C call holding the GIL (CUDA /
TMS behind a driver lock) or a generation-2 GC. A Python-side sampler cannot
tell them apart: it needs the GIL the stall holds. ``faulthandler``'s
``dump_traceback_later`` runs its timer in a C thread and writes the stacks of
every thread without taking the GIL, so a stall that outlives the timeout
names the frame it sits in.

Per tag of the sleep loop (``_weg2_xchg_deposit_before_sleep`` -> sync ->
pause -> credit): :func:`arm` opens ``<evidence>/weg2_tagstall_<group>_r<rank>_
<tag>_<unix>.txt`` with one header line and arms the timer; :func:`disarm`
cancels it and reads the file's SIZE (``fstat``, no device, no host sync):
grown past the header = the watchdog fired -> one ``TAG-STALL-SENTINEL fired``
line naming rank, tag and file; not grown -> the file is removed.

``SGLANG_WEG2_TAG_STALL_SENTINEL_S`` (default 1.5) is the timeout; 0 = off
(no file, no timer). Cost without a stall: one small file create/unlink and
two C calls per tag.
"""

from __future__ import annotations

import faulthandler
import logging
import os
import re
import time
from typing import Optional

logger = logging.getLogger(__name__)

MARKER = "TAG-STALL-SENTINEL"
DEFAULT_TIMEOUT_S = 1.5


class Armed:
    """One armed tag: the open dump file and its header length."""

    __slots__ = ("fh", "path", "header_len", "tag", "rank", "group", "t0")

    def __init__(self, fh, path: str, header_len: int, tag: str, rank, group: str, t0: float):
        self.fh = fh
        self.path = path
        self.header_len = header_len
        self.tag = tag
        self.rank = rank
        self.group = group
        self.t0 = t0


def timeout_s() -> float:
    """The armed timeout, 0.0 when off."""
    try:
        from sglang.srt.environ import envs

        v = float(envs.SGLANG_WEG2_TAG_STALL_SENTINEL_S.get() or 0.0)
    except Exception:  # noqa: BLE001 -- an instrument never breaks a sleep
        return 0.0
    return v if v > 0.0 else 0.0


def dump_dir() -> str:
    return os.environ.get("SGLANG_WEG2_EVIDENCE_DIR") or "/spinning/evidence-665-f1"


def _slug(value) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", str(value)) or "x"


def arm(tag: str, *, rank, group: str, directory: Optional[str] = None,
        timeout: Optional[float] = None) -> Optional[Armed]:
    """Before one sleep tag: open its dump file and arm the C watchdog.
    ``None`` when off or when the file cannot be opened (named once)."""
    t = timeout_s() if timeout is None else float(timeout)
    if t <= 0.0:
        return None
    try:
        d = directory or dump_dir()
        os.makedirs(d, exist_ok=True)
        now = time.time()
        path = os.path.join(d, "weg2_tagstall_%s_r%s_%s_%d.txt" % (
            _slug(group), _slug(rank), _slug(tag), int(now * 1000)))
        fh = open(path, "w")
        header = "%s group=%s rank=%s tag=%s armed_unix=%.3f timeout_s=%.3f pid=%d\n" % (
            MARKER, group, rank, tag, now, t, os.getpid())
        fh.write(header)
        fh.flush()
        faulthandler.dump_traceback_later(t, repeat=False, file=fh, exit=False)
        return Armed(fh, path, len(header.encode()), str(tag), rank, str(group), time.perf_counter())
    except Exception as exc:  # noqa: BLE001 -- an instrument never breaks a sleep
        logger.warning("%s not armed for tag %s: %s: %s", MARKER, tag, type(exc).__name__, exc)
        return None


def disarm(armed: Optional[Armed]) -> Optional[str]:
    """After the tag: cancel the watchdog; return the dump path when it fired
    (and log it), else remove the empty file and return ``None``."""
    if armed is None:
        return None
    try:
        faulthandler.cancel_dump_traceback_later()
        armed.fh.flush()
        size = os.fstat(armed.fh.fileno()).st_size
        armed.fh.close()
        if size > armed.header_len:
            logger.warning(
                "%s fired rank=%s tag=%s file=%s bytes=%d tag_ms=%.0f -- the tag outlived "
                "the watchdog; the file holds every thread's stack at the timeout "
                "(faulthandler C thread, written without the GIL)",
                MARKER, armed.rank, armed.tag, armed.path, size,
                (time.perf_counter() - armed.t0) * 1000)
            return armed.path
        os.unlink(armed.path)
    except Exception as exc:  # noqa: BLE001 -- an instrument never breaks a sleep
        logger.warning("%s disarm for tag %s: %s: %s", MARKER, armed.tag, type(exc).__name__, exc)
    return None
