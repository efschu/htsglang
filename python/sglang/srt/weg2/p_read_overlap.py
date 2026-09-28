"""RO (28.09.2026): P computes other queued work while a store read runs.

THE SPECIMEN. 27B parkdraft boot dkr27bparkdraftbar1w109281421 (82fa502795),
front ``--p-concurrency 1`` with ``SGLANG_WEG2_P_QUEUE_AHEAD`` 1, so at most
TWO leg-1 calls are on P at once. 14:23:14 weg2-0-1 registers its store read
(span 31950), 14:23:15 weg2-0-2 is its fork twin (``#TW TWIN-DEFER``
shared=31792): both slots taken by requests P can only SKIP, the rest of the
backlog sat in the front, P ran nothing. 14:24:17-49 the same with weg2-1-5
reading alone. The release-draft boot (13:42) adds the #1416e pacing window:
weg2-0-2's read took 15.4 s, then PP0 paced it another 10 s -- P idle for
both, with work waiting in the front.

WHY P ITSELF WAS NOT THE BRAKE. P's admission already skips a held rid and
admits everything behind it on every rank (``weg2_store_told.admission``
returns ``None`` -> ``continue``; PP0 decides, the followers follow the told
wire). What was missing is work BEHIND it: the front's dispatch cap counted a
leg whose request only waits for its read like a leg that computes.

THE RULE (switch ``SGLANG_WEG2_P_READ_OVERLAP``, default on; ``0`` = the old
cap byte for byte):

  * P side (PP0 of a told-armed group, top of every pass right after
    ``pp0_publish``): the rids P can only skip -- held (store read in flight,
    or a twin deferred behind its sibling) or paced (read done, window
    running) -- go into ``/dev/shm/weg2_p_reading_<port>.json``, written
    atomically and only when the set changes. Bookkeeping only: nothing on
    P's admission, its wire or its order changes, so group agreement (PP3,
    same admission on every rank) is untouched.
  * front side (``_p_drain_pool``): every in-flight leg that P names as
    reading frees one extra dispatch slot, at most
    ``SGLANG_WEG2_P_READ_OVERLAP_MAX`` (2) beyond ``p_concurrency + ahead``.
    The queue head still goes first, the phase cap and the pool-token budget
    still apply to it. While legs are in flight the pool re-reads the file
    every ``POLL_S`` instead of only on a leg completion.

A missing, unreadable or foreign file means "nothing is reading" -- the old
cap. Stdlib only; the front imports it.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Iterable, Optional, Set

logger = logging.getLogger(__name__)

ENV = "SGLANG_WEG2_P_READ_OVERLAP"
ENV_MAX = "SGLANG_WEG2_P_READ_OVERLAP_MAX"
ENV_DIR = "SGLANG_WEG2_P_READ_STATE_DIR"
MAX_DEFAULT = 2
DIR_DEFAULT = "/dev/shm"
#: how often the front's drain pool re-reads P's reading set while legs run
POLL_S = 0.25
_LOG_FIRST = 8
_LOG_EVERY = 256


def enabled(env=None) -> bool:
    e = os.environ if env is None else env
    raw = (e.get(ENV, "") or "").strip().lower()
    return raw not in ("0", "false", "no", "off")


def max_extra(env=None) -> int:
    e = os.environ if env is None else env
    try:
        n = int(e.get(ENV_MAX, "") or MAX_DEFAULT)
    except ValueError:
        return MAX_DEFAULT
    return max(0, min(16, n))


def state_path(port) -> str:
    base = os.environ.get(ENV_DIR, "") or DIR_DEFAULT
    return os.path.join(base, f"weg2_p_reading_{int(port)}.json")


# ---------------------------------------------------------------------------
# P side (PP0)
# ---------------------------------------------------------------------------

_LAST_ATTR = "_weg2_ro_last"


def reading_rids(scheduler) -> Set[str]:
    """PP0: the rids its admission can only skip right now."""
    out: Set[str] = set()
    for attr in ("_weg2_store_held", "_weg2_told_pacing"):
        d = getattr(scheduler, attr, None)
        if d:
            out.update(str(k) for k in d)
    if out:
        # only what is QUEUED: a rid parked in the dormant hold / post-wake
        # settle is not a leg P skips in its admission
        out &= {str(getattr(r, "rid", "")) for r in (getattr(scheduler, "waiting_queue", None) or ())}
    return out


def _scheduler_port(scheduler) -> Optional[int]:
    sa = getattr(scheduler, "server_args", None)
    port = getattr(sa, "port", None)
    try:
        return int(port) if port is not None else None
    except (TypeError, ValueError):
        return None


def export(scheduler) -> bool:
    """PP0, once per pass: write the reading set when it changed. Returns True
    when a file was written. Never raises into the pass."""
    try:
        if not enabled():
            return False
        port = _scheduler_port(scheduler)
        if port is None:
            return False
        rids = frozenset(reading_rids(scheduler))
        if rids == getattr(scheduler, _LAST_ATTR, None):
            return False
        path = state_path(port)
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w") as fh:
            json.dump({"rids": sorted(rids), "pid": os.getpid(), "t": time.time()}, fh)
        os.replace(tmp, path)
        setattr(scheduler, _LAST_ATTR, rids)
        return True
    except Exception as e:  # noqa: BLE001 - an instrument never breaks the pass
        n = getattr(scheduler, "_weg2_ro_err_n", 0) + 1
        try:
            scheduler._weg2_ro_err_n = n
        except Exception:  # noqa: BLE001
            pass
        if n <= _LOG_FIRST:
            logger.warning("RO P-READ-STATE write failed (n=%d): %r", n, e)
        return False


# ---------------------------------------------------------------------------
# front side
# ---------------------------------------------------------------------------


class ReadState:
    """The front's reader of one P group's reading set (mtime-cached)."""

    def __init__(self, port) -> None:
        self.path = state_path(port)
        self._key = None
        self._rids: Set[str] = set()
        # rids restart at weg2-0-1 on every boot: a file a previous boot left
        # behind would name this boot's first rids as reading. Removed here;
        # missing = nothing reads, P's next change writes it anew.
        try:
            os.unlink(self.path)
        except OSError:
            pass

    def rids(self) -> Set[str]:
        try:
            st = os.stat(self.path)
        except OSError:
            self._key, self._rids = None, set()
            return self._rids
        key = (st.st_mtime_ns, st.st_size, st.st_ino)
        if key != self._key:
            try:
                with open(self.path) as fh:
                    js = json.load(fh)
                self._rids = {str(r) for r in (js.get("rids") or ())}
            except Exception:  # noqa: BLE001 - half a file or none: nothing reads
                self._rids = set()
            self._key = key
        return self._rids


def extra_slots(inflight_rids: Iterable[Any], reading: Set[str], cap: int) -> int:
    """Dispatch slots beyond the base cap: one per in-flight leg P reads for."""
    if cap <= 0 or not reading:
        return 0
    return min(cap, sum(1 for r in inflight_rids if str(r) in reading))
