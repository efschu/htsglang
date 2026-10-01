# SPDX-License-Identifier: Apache-2.0
"""TAG-STALL-SENTINEL (30.09., NF y3z ep52): all thread stacks of a sleep tag
that stalls, written by the shared GIL sampler (weg2/stall_sampler.py).

NF y3z ep52: PP0 stood still for 5.4 s process-wide -- every thread at once --
at the first tag of P's sleep. Candidates: a C call holding the GIL (CUDA /
TMS behind a driver lock) or a generation-2 GC. A Python-side sampler cannot
tell them apart: it needs the GIL the stall holds. The first form used
``faulthandler.dump_traceback_later`` (C thread, no GIL); 27B z30y8 showed that
walk racing a changing frame into a PP1 SIGSEGV, so it now arms
``stall_sampler`` (``sys._current_frames()`` under the GIL). The price is
named, not hidden: a stall holding the GIL in C delays the dump until release,
and ``late_ms`` (the sampler's own monotonic stamp of its FIRST dump minus the
deadline) says by how much -- a large ``late_ms`` IS the GIL-held-in-C
verdict, a small one a Python stall. The dump then shows the stacks AFTER the
GIL came back (the C frame has returned): it names the duration, not the frame.

y6b (review cda5a88fee): ``late_ms`` was the file's mtime -- the LAST write, so
any tag that outlived timeout+8 s read ~8000 ms "late" (the repeat dump), and
on the wall clock (an NTP step moved it). A GIL held in C until the tag ENDED
let the sampler in only after ``disarm`` had set ``stop``: no dump, the file
was removed, nothing logged -- now ``fired ... dump=missed`` names it with the
overdue time. A GIL NEVER released means no ``disarm`` at all: the file stays
at its header line; :func:`unreleased` lists such files (header only, older
than their timeout) and the first :func:`arm` of a process logs what an earlier
process left (``TAG-STALL-SENTINEL unreleased``).

Per tag of the sleep loop (``_weg2_xchg_deposit_before_sleep`` -> sync ->
pause -> credit): :func:`arm` opens ``<evidence>/weg2_tagstall_<group>_r<rank>_
<tag>_<unix>.txt`` with one header line and arms the timer; :func:`disarm`
stops the sampler and reads the file's SIZE (``fstat``, no device, no host sync):
grown past the header = the watchdog fired -> one ``TAG-STALL-SENTINEL fired``
line naming rank, tag and file; not grown -> the file is removed.

``SGLANG_WEG2_TAG_STALL_SENTINEL_S`` (default 1.5) is the timeout; 0 = off
(no file, no timer). Cost without a stall: one small file create/unlink and
two C calls per tag.
"""

from __future__ import annotations

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

    __slots__ = ("fh", "path", "header_len", "tag", "rank", "group", "t0", "sampler", "due_unix")

    def __init__(self, fh, path: str, header_len: int, tag: str, rank, group: str, t0: float,
                 sampler=None, due_unix: float = 0.0):
        self.fh = fh
        self.path = path
        self.header_len = header_len
        self.tag = tag
        self.rank = rank
        self.group = group
        self.t0 = t0
        self.sampler = sampler
        self.due_unix = due_unix


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
    """Before one sleep tag: open its dump file and arm the GIL sampler.
    ``None`` when off or when the file cannot be opened (named once)."""
    t = timeout_s() if timeout is None else float(timeout)
    if t <= 0.0:
        return None
    fh = path = None
    try:
        d = directory or dump_dir()
        os.makedirs(d, exist_ok=True)
        _note_unreleased_once(d)
        now = time.time()
        path = os.path.join(d, "weg2_tagstall_%s_r%s_%s_%d.txt" % (
            _slug(group), _slug(rank), _slug(tag), int(now * 1000)))
        fh = open(path, "w")
        header = "%s group=%s rank=%s tag=%s armed_unix=%.3f timeout_s=%.3f pid=%d\n" % (
            MARKER, group, rank, tag, now, t, os.getpid())
        fh.write(header)
        fh.flush()
        from sglang.srt.weg2 import stall_sampler

        sampler = stall_sampler.arm(fh, t, name="tag-stall-sampler")
        return Armed(fh, path, len(header.encode()), str(tag), rank, str(group), time.perf_counter(),
                     sampler, now + t)
    except Exception as exc:  # noqa: BLE001 -- an instrument never breaks a sleep
        logger.warning("%s not armed for tag %s: %s: %s", MARKER, tag, type(exc).__name__, exc)
        # y6b (F7): an arm that failed after the open leaves neither an fd nor
        # a header-only file (that would read as "GIL never released")
        if fh is not None:
            try:
                fh.close()
            except Exception:  # noqa: BLE001
                pass
        if path is not None:
            try:
                os.unlink(path)
            except OSError:
                pass
        return None


def disarm(armed: Optional[Armed]) -> Optional[str]:
    """After the tag: stop the sampler; return the dump path when it fired
    (and log it), else remove the empty file and return ``None``."""
    if armed is None:
        return None
    try:
        from sglang.srt.weg2 import stall_sampler

        ended = time.monotonic()   # the tag's end, before the join
        stall_sampler.disarm(armed.sampler)
        smp = armed.sampler
        tag_ms = (time.perf_counter() - armed.t0) * 1000
        missed = False
        lock = smp.lock if smp is not None else None
        # y6b (F6): bounded like the join -- a sampler mid-dump waiting for a
        # foreign C section's GIL must not park the sleep loop
        locked = lock.acquire(timeout=stall_sampler.JOIN_S) if lock is not None else False
        if lock is not None and not locked:
            logger.warning("%s fired rank=%s tag=%s file=%s tag_ms=%.0f dump=in_progress -- the "
                           "sampler still writes (GIL busy); the file is left open to it",
                           MARKER, armed.rank, armed.tag, armed.path, tag_ms)
            return armed.path
        try:
            # under the sampler's lock: a dump it was writing has landed (or never will)
            missed = smp is not None and stall_sampler.missed(smp, ended)
            armed.fh.flush()
            st = os.fstat(armed.fh.fileno())
            if missed:
                armed.fh.write("No dump: the deadline passed %.0f ms before the tag ended and the "
                               "sampler never got the GIL (a C call held it to the end)\n"
                               % smp.overdue_ms(ended))
                armed.fh.flush()
            armed.fh.close()
        finally:
            if locked:
                lock.release()
        if missed:
            # y6b (F1): the stall held the GIL until the tag ended; the sampler
            # woke to `stop` and left. Before: file removed, nothing logged.
            logger.warning(
                "%s fired rank=%s tag=%s file=%s tag_ms=%.0f late_ms=%.0f dump=missed -- the tag "
                "outlived the timeout and the sampler never got the GIL before it ended: the "
                "stall held the GIL in C (no stacks; the duration is the evidence)",
                MARKER, armed.rank, armed.tag, armed.path, tag_ms, smp.overdue_ms(ended))
            return armed.path
        if st.st_size > armed.header_len:
            late = smp.first_late_ms if smp is not None else None
            logger.warning(
                "%s fired rank=%s tag=%s file=%s bytes=%d tag_ms=%.0f late_ms=%s dumps=%d -- the "
                "tag outlived the sampler timeout; the file holds every thread's stack AFTER the "
                "sampler got the GIL (late_ms = first dump past its deadline, monotonic; >> 0 = "
                "the stall held the GIL in C, the stalled C frame has returned by then)",
                MARKER, armed.rank, armed.tag, armed.path, st.st_size, tag_ms,
                "%.0f" % late if late is not None else "?",
                smp.dumps if smp is not None else 0)
            return armed.path
        os.unlink(armed.path)
    except Exception as exc:  # noqa: BLE001 -- an instrument never breaks a sleep
        logger.warning("%s disarm for tag %s: %s: %s", MARKER, armed.tag, type(exc).__name__, exc)
    return None


_UNRELEASED_NOTED = False


def unreleased(directory: Optional[str] = None, now_unix: Optional[float] = None) -> list:
    """y6b (F4): the dump files that never reached :func:`disarm` -- header
    line only, past their own timeout. Each is a tag whose GIL was never
    released (a C call wedged for good) or whose process died inside it.
    Returns ``[(path, armed_unix, timeout_s, pid)]``, oldest first."""
    d = directory or dump_dir()
    now = time.time() if now_unix is None else float(now_unix)
    out = []
    try:
        names = [n for n in os.listdir(d) if n.startswith("weg2_tagstall_") and n.endswith(".txt")]
    except OSError:
        return out
    for n in names:
        p = os.path.join(d, n)
        try:
            with open(p) as f:
                head = f.readline()
                rest = f.read(1)
        except OSError:
            continue
        if rest or not head.startswith(MARKER):
            continue
        m = re.search(r"armed_unix=([0-9.]+) timeout_s=([0-9.]+) pid=(\d+)", head)
        if not m:
            continue
        armed_unix, t = float(m.group(1)), float(m.group(2))
        if armed_unix + t < now:
            out.append((p, armed_unix, t, int(m.group(3))))
    out.sort(key=lambda r: r[1])
    return out


def _note_unreleased_once(directory: str) -> None:
    """The first arm of a process names what earlier processes left behind (one scan per process)."""
    global _UNRELEASED_NOTED
    if _UNRELEASED_NOTED:
        return
    _UNRELEASED_NOTED = True
    mine = os.getpid()
    rows = [r for r in unreleased(directory) if r[3] != mine]
    for p, armed_unix, t, pid in rows[-8:]:
        logger.warning("%s unreleased file=%s armed_unix=%.3f timeout_s=%.3f pid=%d -- header only: "
                       "that tag never reached disarm (the GIL was never released, or the process "
                       "died inside it)", MARKER, p, armed_unix, t, pid)
    if len(rows) > 8:
        logger.warning("%s unreleased: %d more in %s", MARKER, len(rows) - 8, directory)
