"""FP FORWARD-PROGRESS BEACON (NF rc12p, 27.09. 14:13:39): "the group computes" is never "the group
is dead".

THE GAP. FH (pdflip/front_health.py) counts an http_ok=False streak >= 2 as unhealthy (503; W17 off a
flip). NF measured one /health failure while D ran a 91k extend (200 again 7 s later); an extend of
91k-262k tokens is one 10-60 s forward, so two slow probes in a row read as a dead group.

THE BEACON (switch ``FLLIPER_PDFLIP_PROGRESS_BEACON``, default on; ``0`` = no file, no reading -- the
FH rule byte for byte). Every scheduler rank of a Weg-2 group (``FLLIPER_PDFLIP_GROUP``) keeps one
32-byte mmap file ``<arena>/progress/<GROUP>-pid<pid>.bin`` = (forward_ct, t_start_ns, t_done_ns,
pid): ``beat_start`` at the top of every forward (``Scheduler._run_batch_forward``), ``beat_done``
after its result is processed (``process_batch_result``). One ``struct.pack_into`` each -- no
syscall, no collective, no new endpoint (the /health path and ``/get_server_info`` both wait on the
busy process; the file does not).

THE READING (front, every FH poll): a group shows PROGRESS when any of its live ranks' files
(pid in the group's session) moved ``forward_ct`` since the previous poll, or is inside a forward
(t_start > t_done) begun less than ``BUSY_BOUND_S`` (120 s) ago. An http_ok=False with progress is
logged ``PDFLIP-HEALTH-BUSY`` and does not count toward the streak. A hold and process_alive=False
stay fatal at once, unchanged.
"""
from __future__ import annotations

import glob
import mmap
import os
import struct
import time
from typing import Callable, Dict, Optional, Tuple

ENV = "FLLIPER_PDFLIP_PROGRESS_BEACON"
SUBDIR = "progress"
_FMT = "<qqqq"
_SIZE = struct.calcsize(_FMT)
#: a forward running longer than this is not progress (the scheduler
#: watchdog's own bound is 300 s; a 262k extend measured <= 60 s).
BUSY_BOUND_S = 120.0


def enabled(env=None) -> bool:
    e = os.environ if env is None else env
    raw = (e.get(ENV, "") or "").strip().lower()
    return raw not in ("0", "false", "no", "off")


def beacon_dir(tag: str = "", env=None) -> str:
    from flliper.srt.pdflip.resume_via_p import arena_dir

    base = arena_dir(tag, env)
    return os.path.join(base, SUBDIR) if base else ""


class _Writer:
    def __init__(self):
        self.mm = None
        self.dead = False

    def _open(self):
        if self.dead or self.mm is not None:
            return self.mm
        try:
            if not enabled():
                self.dead = True
                return None
            group = (os.environ.get("FLLIPER_PDFLIP_GROUP", "") or "").strip().upper()
            d = beacon_dir()
            if not group or not d:
                self.dead = True
                return None
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, f"{group}-pid{os.getpid()}.bin")
            fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
            try:
                os.ftruncate(fd, _SIZE)
                self.mm = mmap.mmap(fd, _SIZE)
            finally:
                os.close(fd)
            struct.pack_into(_FMT, self.mm, 0, 0, 0, 0, os.getpid())
        except Exception:  # noqa: BLE001 -- a beacon never stops a forward
            self.dead = True
            self.mm = None
        return self.mm

    def beat(self, forward_ct: int, start: bool) -> None:
        mm = self._open()
        if mm is None:
            return
        try:
            ct, ts, td, pid = struct.unpack_from(_FMT, mm, 0)
            now = time.time_ns()
            if start:
                struct.pack_into(_FMT, mm, 0, int(forward_ct), now, td, pid)
            else:
                struct.pack_into(_FMT, mm, 0, int(forward_ct), ts, now, pid)
        except Exception:  # noqa: BLE001
            pass


_W = _Writer()


def beat_start(forward_ct: int) -> None:
    _W.beat(forward_ct, True)


def beat_done(forward_ct: int) -> None:
    _W.beat(forward_ct, False)


def read_group(directory: str, group: str, sid: int,
               session_of: Optional[Callable[[int], Optional[int]]] = None) -> Dict[int, Tuple[int, int, int]]:
    """{pid: (forward_ct, t_start_ns, t_done_ns)} of the group's LIVE ranks."""
    if not directory or not sid:
        return {}
    if session_of is None:
        from flliper.srt.pdflip.front_health import pid_session as session_of
    out: Dict[int, Tuple[int, int, int]] = {}
    for path in glob.glob(os.path.join(directory, f"{group}-pid*.bin")):
        try:
            with open(path, "rb") as fh:
                raw = fh.read(_SIZE)
            ct, ts, td, pid = struct.unpack(_FMT, raw)
        except Exception:  # noqa: BLE001
            continue
        if session_of(int(pid)) != int(sid):
            continue
        out[int(pid)] = (int(ct), int(ts), int(td))
    return out


def progress(prev: Dict[int, Tuple[int, int, int]], cur: Dict[int, Tuple[int, int, int]],
             now_ns: Optional[int] = None, bound_s: float = BUSY_BOUND_S) -> Optional[str]:
    """Why the group shows progress (a short reason), or None."""
    now_ns = time.time_ns() if now_ns is None else now_ns
    for pid, (ct, ts, td) in cur.items():
        p = prev.get(pid)
        if p is not None and ct != p[0]:
            return f"forward_ct {p[0]}->{ct} (pid {pid})"
        if ts > td and (now_ns - ts) < bound_s * 1e9:
            return f"in forward {ct} for {(now_ns - ts) / 1e9:.1f}s (pid {pid})"
    return None
