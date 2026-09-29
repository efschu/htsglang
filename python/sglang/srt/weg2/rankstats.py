"""RANKSTATS (DASHBOARD-AUS-IPC, 29.09.): each scheduler rank's cumulative counters
as a small file in the group's RankState directory, every ``PERIOD_S`` -- so the
rigdash reads prefill/decode/queue/errors of a rank from the IPC, not from the
rank's log lines (user order via 27B: "das dashboard soll auch aus der inter
prozess kommunikation gespeist werden, nicht aus logs").

Schema ``weg2.rankstats/1`` (agreed with the M2 IPC seat), file
``<rankstate dir>/<G>.tp<t>pp<p>.rankstats`` (JSON; not ``.json``, so the
RankState reader and the W7/W10 gate never see it):

  head      schema, pid, ts, seq, group, tp_rank, pp_rank, rank_state_seq
  work      forward_ct (the round path's own counter; the dashboard's per-rank
            heartbeat together with ts)
  tokens    prefill_total, decode_total (MetricsReporter's monotone counters)
  spec      accept_tokens_total, forward_ct_total (lifetime spec counters)
  sched     waiting, running
  errors    n, last[8]{t, logger, level, exc, text}  (ERROR/CRITICAL records)
  last_post_wake  the latest WEG2-POST-WAKE-PASS census as a dict, or null

THE RULE (27B review of the plan, 29.09.): the decode/forward path writes NOTHING.
One timer thread ``weg2-rankstats`` wakes every ``PERIOD_S``, READS counters the
round path increments anyway (plain int attributes, no lock in the path), and
writes the file atomically (tmp + os.replace). The switch off starts no thread.
``last_post_wake`` is one attribute assignment in the post-wake census (the first
8 passes after a wake, never a decode round). This heartbeat is display only; it
replaces no deadman riegel.
"""
from __future__ import annotations

import collections
import json
import logging
import os
import threading
import time
from typing import Any, Callable, Dict, Optional

SCHEMA = "weg2.rankstats/1"
SUFFIX = ".rankstats"
THREAD_NAME = "weg2-rankstats"
ERRORS_KEEP = 8


def enabled() -> bool:
    from sglang.srt.environ import envs

    return bool(envs.SGLANG_WEG2_ENABLE_RANKSTATS.get())


def period_s() -> float:
    from sglang.srt.environ import envs

    return max(0.2, float(envs.SGLANG_WEG2_RANKSTATS_PERIOD_S.get()))


def rankstats_path(state_dir: str, group: str, tp_rank: int, pp_rank: int) -> str:
    return os.path.join(state_dir, f"{group or 'G'}.tp{int(tp_rank)}pp{int(pp_rank)}{SUFFIX}")


class ErrorTally(logging.Handler):
    """ERROR/CRITICAL records of this process as a count and the last
    ``ERRORS_KEEP`` -- structured from the record objects, never parsed from a
    log line. ``emit`` is an int increment and a bounded deque append."""

    def __init__(self, keep: int = ERRORS_KEEP) -> None:
        super().__init__(level=logging.ERROR)
        self.n = 0
        self.last: "collections.deque[dict]" = collections.deque(maxlen=int(keep))

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D401 -- logging API
        try:
            exc = record.exc_info[0].__name__ if record.exc_info and record.exc_info[0] else None
            self.n += 1
            self.last.append({"t": round(record.created, 3), "logger": record.name,
                              "level": record.levelname, "exc": exc,
                              "text": str(record.getMessage())[:300]})
        except Exception:  # noqa: BLE001 -- a tally never breaks logging
            pass

    def snapshot(self) -> dict:
        return {"n": self.n, "last": list(self.last)}


def install_error_tally(logger_name: str = "sglang") -> ErrorTally:
    """One tally per process on ``logger_name`` (idempotent)."""
    lg = logging.getLogger(logger_name)
    for h in lg.handlers:
        if isinstance(h, ErrorTally):
            return h
    h = ErrorTally()
    lg.addHandler(h)
    return h


def scheduler_counters(scheduler) -> Dict[str, Any]:
    """READ ONLY: the counters the scheduler's round path increments anyway."""
    mr = getattr(scheduler, "metrics_reporter", None)
    rb = getattr(scheduler, "running_batch", None)
    try:
        running = len(getattr(rb, "reqs", None) or ())
    except Exception:  # noqa: BLE001 -- a racing swap of the batch reads as unknown
        running = None
    return {
        "work": {"forward_ct": int(getattr(scheduler, "forward_ct", 0) or 0)},
        "tokens": {"prefill_total": int(getattr(mr, "prefill_tokens_total", 0) or 0),
                   "decode_total": int(getattr(mr, "gen_tokens_total", 0) or 0)},
        "spec": {"accept_tokens_total": int(getattr(mr, "spec_total_num_accept_tokens", 0) or 0),
                 "forward_ct_total": int(getattr(mr, "spec_total_num_forward_ct", 0) or 0)},
        "sched": {"waiting": len(getattr(scheduler, "waiting_queue", None) or ()),
                  "running": running},
    }


def _rank_state_seq() -> Optional[int]:
    try:
        from sglang.srt.weg2 import vram_actual

        return int(vram_actual._REC.seq)
    except Exception:  # noqa: BLE001 -- no vram block: no seq
        return None


class RankStats:
    """The timer: reads ``read_counters()`` every ``period`` and writes one file."""

    def __init__(self, *, state_dir: str, group: str, tp_rank: int, pp_rank: int,
                 read_counters: Callable[[], Dict[str, Any]], period: float,
                 errors: Optional[ErrorTally] = None) -> None:
        self.path = rankstats_path(state_dir, group, tp_rank, pp_rank)
        self.group, self.tp_rank, self.pp_rank = group, int(tp_rank), int(pp_rank)
        self.read_counters = read_counters
        self.period = float(period)
        self.errors = errors
        self.last_post_wake: Optional[dict] = None
        self.seq = 0
        self.writes = 0
        self.failed = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def record(self) -> dict:
        rec = {"schema": SCHEMA, "pid": os.getpid(), "ts": round(time.time(), 3),
               "seq": self.seq, "group": self.group, "tp_rank": self.tp_rank,
               "pp_rank": self.pp_rank, "rank_state_seq": _rank_state_seq()}
        rec.update(self.read_counters())
        rec["errors"] = self.errors.snapshot() if self.errors is not None else None
        rec["last_post_wake"] = self.last_post_wake
        return rec

    def write_once(self) -> None:
        self.seq += 1
        body = json.dumps(self.record(), separators=(",", ":"))
        tmp = f"{self.path}.tmp.{os.getpid()}"
        with open(tmp, "w") as f:
            f.write(body)
        os.replace(tmp, self.path)
        self.writes += 1

    def _run(self) -> None:
        while not self._stop.wait(self.period):
            try:
                self.write_once()
            except Exception:  # noqa: BLE001 -- a lost sample never stops the rank
                self.failed += 1

    def start(self) -> "RankStats":
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._thread = threading.Thread(target=self._run, name=THREAD_NAME, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()


_CURRENT: Optional[RankStats] = None


def maybe_start(scheduler, *, tp_rank: int, pp_rank: int) -> Optional[RankStats]:
    """run_scheduler_process, once. Switch off, or no RankState directory (a
    boot outside the weg2 launcher): no thread, nothing written."""
    global _CURRENT
    if not enabled():
        return None
    from sglang.srt.environ import envs

    d = envs.SGLANG_WEG2_RANK_STATE_DIR.get()
    if not d:
        return None
    group = (os.environ.get("SGLANG_WEG2_GROUP", "") or "").strip().upper()
    _CURRENT = RankStats(state_dir=d, group=group, tp_rank=tp_rank, pp_rank=pp_rank,
                         read_counters=lambda: scheduler_counters(scheduler),
                         period=period_s(), errors=install_error_tally()).start()
    return _CURRENT


def note_post_wake(census: dict) -> None:
    """The post-wake census (first passes after a wake, never a decode round):
    one attribute assignment; the timer writes it with the next sample."""
    rs = _CURRENT
    if rs is not None:
        rs.last_post_wake = dict(census)
