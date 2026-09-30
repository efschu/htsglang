# SPDX-License-Identifier: Apache-2.0
"""A stall sampler that reads every thread's stack UNDER THE GIL -- the shared replacement for
``faulthandler.dump_traceback_later`` in the Weg-2 stall instruments.

WHY (z30y8, 30.09. 18:41:55, P PP1 SIGSEGV -> W17): faulthandler's later-dump runs in a C thread and
walks the frames of every other thread WITHOUT the GIL. It fired while the main thread was inside
``logging.emit``; the dump file stops in the middle of that thread's stack and the rank died. A
lock-free walk over a frame that is changing is best effort by design. This module does the same job
in plain Python: a daemon thread waits on an event with a timeout and, if it expires, formats
``sys._current_frames()`` -- holding the GIL, so no frame changes under it. The price, named: a stall
that holds the GIL in C delays the dump until the GIL is released. ``faulthandler.enable()`` (real
fatal signals) is untouched.

Interface (used by weg2/rpc_stall_watchdog.py; NF's weg2/tag_stall_sentinel.py can switch to it):

    s = arm(fh, timeout_s, repeat_s=8.0, name="rpc-stall-sampler")   # starts the thread
    ...  # the guarded work
    disarm(s, join_s=2.0)          # stops it; bounded join (never a hang); then fh may be read/closed

``fh`` is an open text file the caller owns (header already written). On expiry the sampler appends
"Timeout (...)" + every thread's stack (its own thread left out); if the work still runs after
``repeat_s`` more seconds, one "Second dump" follows (``repeat_s <= 0``: none). ``dump_stacks`` is
the formatter on its own. Writes and the caller's close are serialised by ``s.lock``.
"""

from __future__ import annotations

import sys
import threading
import traceback
from typing import Optional

REPEAT_S = 8.0
JOIN_S = 2.0


def dump_stacks(fh, title: str, skip_ident: Optional[int] = None) -> None:
    """Every thread's stack via ``sys._current_frames()`` (under the GIL); ``skip_ident`` left out."""
    names = {t.ident: t.name for t in threading.enumerate()}
    out = [title]
    for ident, frame in sys._current_frames().items():
        if ident == skip_ident:
            continue
        out.append("Thread 0x%016x (%s) (most recent call last):" % (ident, names.get(ident, "?")))
        out.extend(line.rstrip("\n") for line in traceback.format_stack(frame))
        out.append("")
    fh.write("\n".join(out) + "\n")
    fh.flush()


class Sampler:
    __slots__ = ("fh", "timeout_s", "repeat_s", "stop", "lock", "thread", "dumps")

    def __init__(self, fh, timeout_s: float, repeat_s: float = REPEAT_S):
        self.fh = fh
        self.timeout_s = float(timeout_s)
        self.repeat_s = float(repeat_s)
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None
        self.dumps = 0

    def _run(self) -> None:
        me = threading.get_ident()
        plan = [(self.timeout_s, "Timeout (%.1f s): every thread's stack (Python sampler, under the GIL)"
                 % self.timeout_s)]
        if self.repeat_s > 0:
            plan.append((self.repeat_s, "Second dump (+%.1f s): the guarded work still runs" % self.repeat_s))
        for wait_s, title in plan:
            if self.stop.wait(wait_s):
                return
            with self.lock:
                if self.stop.is_set() or getattr(self.fh, "closed", False):
                    return
                try:
                    dump_stacks(self.fh, title, skip_ident=me)
                    self.dumps += 1
                except Exception as exc:  # noqa: BLE001 -- an instrument never breaks the work
                    try:
                        self.fh.write("dump failed: %s: %s\n" % (type(exc).__name__, exc))
                        self.fh.flush()
                    except Exception:  # noqa: BLE001
                        pass
                    return


def arm(fh, timeout_s: float, repeat_s: float = REPEAT_S, name: str = "stall-sampler") -> Sampler:
    s = Sampler(fh, timeout_s, repeat_s)
    s.thread = threading.Thread(target=s._run, name=name, daemon=True)
    s.thread.start()
    return s


def disarm(s: Optional[Sampler], join_s: float = JOIN_S) -> None:
    """Stop the sampler; a dump in progress finishes (bounded by ``join_s``, never a hang)."""
    if s is None:
        return
    s.stop.set()
    if s.thread is not None:
        s.thread.join(join_s)
