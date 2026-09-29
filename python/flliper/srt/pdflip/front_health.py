"""FH FRONT HEALTH FROM THE POLLER'S FACTS (27B rc12k27 b1, 27.09.2026).

THE GAP. The front's ``/health`` asked both groups' HTTP ``/health`` live and
answered 200 while both said 200 and ``state != STOP``. A group's HTTP
``/health`` is the tokenizer manager's (``FLLIPER_ENABLE_HEALTH_ENDPOINT_GENERATION=0``
on both groups, K2): it never asks a scheduler. So a dead rank was invisible to
it. Boot b1 (dkr27breleasedraftbar1w109270932), measured:

  09:45:58 P  PP1 PPWidthDivergenceRefused (#1233 W27) -> 09:45:59 #1223 DEBUG-HOLD
           rank=1 holding (dump written, the process stays alive, 1800 s bound)
  09:46..09:53 P  GET /health 200 every 15 s; front /health 200 (curl every 30 s)
  09:53:17 P  Scheduler watchdog timeout (300 s, forward counter frozen) on PP1/PP2
  09:53:22 P  SIGQUIT received -> crash diagnostics (5 s sleep, 60 s coredump wait):
           the HTTP server stops answering
  09:53:47 front PDFLIP-HEALTH group=P http_ok=False process_alive=True streak=1
           (the 25 s probe timeout after the SIGQUIT)
  09:54:27 front streak=2 -> W17 PdFlipGroupDead

So the 8 min were not the poll interval: the group's HTTP stayed 200 for the
whole hold, the first HTTP failure is the WATCHDOG's SIGQUIT (300 s after the
last forward, detected ~440 s after the death), plus one 25 s probe timeout.
``process_alive`` is "some process of the group's session exists" -- true for
as long as the held rank (or any process) lives.

THE RULE (switch ``FLLIPER_PDFLIP_FRONT_HEALTH_FACTS``, default on; ``0`` = the old
poller and the old live ``/health`` byte for byte):

  * the poller probes both groups CONCURRENTLY every 5 s with an 8 s timeout
    (old: sequentially, every 15 s, 25 s each);
  * it also looks for a DEBUG-HOLD dump (``#1223``, ``FLLIPER_DEBUG_HOLD_DIR``)
    whose pid is a live process of the group's session -- a held rank is a
    named stop, a fact in every state, like a dead session: W17 at once;
  * the front's ``/health`` answers from these facts: 503 when ``state`` is
    STOP or any group is DEAD -- ``process_alive=False``, a known hold, or
    ``http_ok=False`` for a streak >= 2. During a flip only a DEAD group counts
    (dead session or hold); a group whose HTTP is silent behind a flip leg is
    asleep, not dead (#1378 xsn39).
  * facts older than ``STALE_S`` (or none yet) -> that group is probed live, as
    before.

Bounds after the change: a held rank -> <= 5 s (+ one dir scan) to the first
PDFLIP-HEALTH line, the W17 stop and the 503; a rank death without hold (the
group's HTTP hangs after the SIGQUIT) -> first failure <= 13 s, 503/W17 at
streak 2 <= 26 s.

Pure helpers only; the front calls them.
"""

from __future__ import annotations

import glob
import os
import re
from typing import Callable, Iterable, List, NamedTuple, Optional

ENV = "FLLIPER_PDFLIP_FRONT_HEALTH_FACTS"
POLL_S = 5.0
PROBE_TIMEOUT_S = 8.0
OLD_POLL_S = 15.0
OLD_PROBE_TIMEOUT_S = 25.0
#: a fact older than this is not a fact any more: /health probes live.
STALE_S = 60.0
#: bytes of a dump read for its header (marker, utc, pid, exception).
_HEAD_BYTES = 4096

_PID_RE = re.compile(r"_pid(\d+)_")


def enabled(env=None) -> bool:
    e = os.environ if env is None else env
    raw = (e.get(ENV, "") or "").strip().lower()
    return raw not in ("0", "false", "no", "off")


def poll_interval_s() -> float:
    return POLL_S if enabled() else OLD_POLL_S


def probe_timeout_s() -> float:
    return PROBE_TIMEOUT_S if enabled() else OLD_PROBE_TIMEOUT_S


class HoldFact(NamedTuple):
    pid: int
    path: str
    exception: str


def hold_dirs(env=None) -> List[str]:
    """Where a held rank writes its dump: the group env's dir (the container
    form sets it for the whole container) and the module default."""
    from flliper.srt.managers.debug_hold import DEFAULT_DUMP_DIR, HOLD_DIR_ENV

    e = os.environ if env is None else env
    out = []
    for d in (e.get(HOLD_DIR_ENV) or "", DEFAULT_DUMP_DIR):
        if d and d not in out:
            out.append(d)
    return out


def pid_session(pid: int) -> Optional[int]:
    """The session id of a live pid (``/proc/<pid>/stat`` field 6), else None."""
    try:
        with open(f"/proc/{int(pid)}/stat", "rb") as fh:
            raw = fh.read().decode("ascii", "replace")
        rest = raw[raw.rindex(")") + 2:].split()
        return int(rest[3])  # state ppid pgrp session
    except (OSError, ValueError, IndexError):
        return None


def _exception_of(path: str) -> str:
    try:
        with open(path, "rb") as fh:
            head = fh.read(_HEAD_BYTES).decode("utf-8", "replace")
    except OSError:
        return "?"
    for line in head.splitlines():
        if line.startswith("exception: "):
            return line[len("exception: "):][:300]
    return "?"


def find_hold(sid: int, since: float, dirs: Iterable[str],
              session_of: Optional[Callable[[int], Optional[int]]] = None) -> Optional[HoldFact]:
    """A DEBUG-HOLD dump written since ``since`` by a LIVE process of session
    ``sid``, or None. The pid in the dump's name is mapped to its session, so a
    shared dump dir (other boots, other groups) never names the wrong group."""
    if not sid:
        return None
    session_of = session_of or pid_session
    best = None
    for d in dirs:
        for path in glob.glob(os.path.join(d, "*_pid*_*.txt")):
            m = _PID_RE.search(os.path.basename(path))
            if m is None:
                continue
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                continue
            if mtime < since:
                continue
            pid = int(m.group(1))
            if session_of(pid) != int(sid):
                continue
            if best is None or mtime < best[0]:
                best = (mtime, pid, path)
    if best is None:
        return None
    return HoldFact(best[1], best[2], _exception_of(best[2]))


class GroupFacts(NamedTuple):
    http_ok: bool
    alive: bool
    streak: int
    hold: Optional[HoldFact]
    t: float


def dead_reason(facts: GroupFacts) -> Optional[str]:
    """The group is DEAD in every state: a held rank or no process left."""
    if facts.hold is not None:
        return (f"debug-hold pid={facts.hold.pid} ({facts.hold.exception}) "
                f"dump={facts.hold.path}")
    if not facts.alive:
        return "process_alive=False"
    return None


def unhealthy_reason(facts: GroupFacts, state: str) -> Optional[str]:
    """Why this group makes the front's /health 503, or None. During a flip only
    a dead group counts -- a silent HTTP behind a flip leg is a sleeping group."""
    dead = dead_reason(facts)
    if dead is not None:
        return dead
    if state == "flipping":
        return None
    if not facts.http_ok and facts.streak >= 2:
        return f"http_ok=False streak={facts.streak}"
    return None
