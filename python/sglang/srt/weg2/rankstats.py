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
  sched     waiting, running, queue_req, running_req, pending_tokens,
            full_token_usage (the last Prefill/Decode batch line's pool stats)
  cap       kv_tokens, seats (max_total_num_tokens, max_running_requests); the
            timer also hands a change to the RankState record (C5,
            rank_state.note_capacity) -- off the round path
  prefill   §3: chunks, new_tokens, cached_tokens, gpu_ms, split_ms, compute_ms,
            wait_ms, bubble_ms, last{t, new, gpu_ms, compute_ms}
            (RankPrefillLog.cum -- the ``Prefill rank batch`` numbers, summed)
  decode    §3: rounds, gpu_ms, gpu_ms_by_bs{bs: [rounds, gpu_ms]}, tokens,
            running, accept_len_ewma, accept_rate_ewma, cuda_graph
            (DecodeRoundLog.cum_* + the logged Decode batch values)
  cache     §3: loadback_n, loadback_tok, mamba_resume_n, mamba_tok,
            store_incomplete_n, prefetch{landed, deferred, refused, expired,
            issued, attempted, defer_refused, timeout} (the #988 / #1324 /
            #915 / #1157 counters as they are; mamba_tok = the summed depths of
            the MAMBA-HOST-RESUME acceptances)
  errors    n, last[8]{t, logger, level, exc, text}  (ERROR/CRITICAL records)
  last_post_wake  the latest WEG2-POST-WAKE-PASS census as a dict, or null
  stops     n, last[8]{t, reason, code, exc, ticket, text}: the scheduler's
            death path (A14), written synchronously by note_stop before the
            #1223 hold and the SIGQUIT; the front publishes each as the event
            ``rank_stop`` (front_state_ipc.publish_rank_stops)

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
import re
import threading
import time
from typing import Any, Callable, Dict, Optional

SCHEMA = "weg2.rankstats/1"
SUFFIX = ".rankstats"
THREAD_NAME = "weg2-rankstats"
ERRORS_KEEP = 8
STOPS_KEEP = 8
STOP_TEXT_MAX = 400


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


def _prefill_block(mr) -> Optional[Dict[str, Any]]:
    """§3 prefill: RankPrefillLog.cum, copied (the path writes, the timer reads)."""
    rpl = getattr(mr, "rank_prefill_log", None)
    cum = getattr(rpl, "cum", None)
    if not isinstance(cum, dict):
        return None
    out = dict(cum)
    for k in ("gpu_ms", "split_ms", "compute_ms", "wait_ms", "bubble_ms"):
        if isinstance(out.get(k), float):
            out[k] = round(out[k], 1)
    return out


def _decode_block(mr) -> Optional[Dict[str, Any]]:
    """§3 decode: DecodeRoundLog.cum_* plus the Decode batch line's values."""
    if mr is None:
        return None
    drl = getattr(mr, "decode_round_log", None)
    by_bs = getattr(drl, "cum_by_bs", None)
    ewma = getattr(mr, "accept_len_ewma", None)
    rate = getattr(mr, "accept_rate_ewma", None)
    return {
        "rounds": getattr(drl, "cum_rounds", None),
        "gpu_ms": None if drl is None else round(float(getattr(drl, "cum_gpu_ms", 0.0)), 1),
        "gpu_ms_by_bs": None if by_bs is None else {
            str(bs): [int(v[0]), round(float(v[1]), 1)] for bs, v in list(by_bs.items())},
        "tokens": int(getattr(mr, "gen_tokens_total", 0) or 0),
        "running": getattr(mr, "last_running_reqs", None),
        "accept_len_ewma": None if ewma is None else round(float(ewma), 3),
        "accept_rate_ewma": None if rate is None else round(float(rate), 3),
        "cuda_graph": getattr(mr, "last_cuda_graph", None),
    }


def _cache_block(scheduler) -> Dict[str, Any]:
    """§3 cache: the #988 / #1324 / #915 counters the paths already keep."""
    out: Dict[str, Any] = {"loadback_n": None, "loadback_tok": None,
                           "mamba_resume_n": None, "mamba_tok": None,
                           "store_incomplete_n": None, "prefetch": None}
    try:
        from sglang.srt.managers import schedule_policy as _sp

        seen = getattr(_sp, "_988_LOADBACK_SEEN", None) or {}
        out["loadback_n"] = int(seen.get("n", 0))
        out["loadback_tok"] = int(seen["tok"]) if "tok" in seen else None
        out["mamba_resume_n"] = int(seen.get("mamba", 0))
    except Exception:  # noqa: BLE001 -- a missing module reads as unknown
        pass
    try:
        from sglang.srt.mem_cache.unified_cache_components.mamba_component import (
            MambaComponent,
        )

        out["mamba_tok"] = int(getattr(MambaComponent, "_host_resume_tok", 0))
    except Exception:  # noqa: BLE001 -- a missing module reads as unknown
        pass
    out["store_incomplete_n"] = int(getattr(scheduler, "_weg2_store_short_seen", 0) or 0)
    try:
        from sglang.srt.mem_cache import match_refusal_census as _mrc

        counts = dict(getattr(_mrc, "PREFETCH_GATE_COUNTS", {}) or {})
        # refused = the gate exits that print '#915 PREFETCH REFUSED' (the decline
        # order); anchor_pool_exhausted is the CAUSE counter of an exit counted
        # there already (match_refusal_census), so it is not summed twice.
        declines = [k for k in getattr(_mrc, "PREFETCH_DECLINE_ORDER", ())
                    if k != "anchor_pool_exhausted"]
        tree = getattr(scheduler, "tree_cache", None)
        out["prefetch"] = {
            "attempted": counts.get("attempted", 0),
            "issued": counts.get("issued", 0),
            "landed": counts.get("landed", 0),
            "deferred": counts.get("deferred", 0),
            "defer_refused": counts.get("defer_refused", 0),
            "expired": counts.get("defer_expired", 0),
            "refused": sum(int(counts.get(k, 0)) for k in declines),
            "timeout": None if tree is None else int(getattr(tree, "_1157_reaped_n", 0) or 0),
        }
    except Exception:  # noqa: BLE001 -- a missing module reads as unknown
        pass
    return out


def scheduler_counters(scheduler) -> Dict[str, Any]:
    """READ ONLY: the counters the scheduler's round path increments anyway."""
    mr = getattr(scheduler, "metrics_reporter", None)
    rb = getattr(scheduler, "running_batch", None)
    try:
        running = len(getattr(rb, "reqs", None) or ())
    except Exception:  # noqa: BLE001 -- a racing swap of the batch reads as unknown
        running = None
    waiting = len(getattr(scheduler, "waiting_queue", None) or ())
    return {
        "prefill": _prefill_block(mr),
        "decode": _decode_block(mr),
        "cache": _cache_block(scheduler),
        "work": {"forward_ct": int(getattr(scheduler, "forward_ct", 0) or 0)},
        "tokens": {"prefill_total": int(getattr(mr, "prefill_tokens_total", 0) or 0),
                   "decode_total": int(getattr(mr, "gen_tokens_total", 0) or 0)},
        "spec": {"accept_tokens_total": int(getattr(mr, "spec_total_num_accept_tokens", 0) or 0),
                 "forward_ct_total": int(getattr(mr, "spec_total_num_forward_ct", 0) or 0)},
        "sched": {"waiting": waiting, "running": running,
                  "queue_req": waiting, "running_req": running,
                  "pending_tokens": getattr(mr, "last_pending_tokens", None),
                  "full_token_usage": _round_or_none(getattr(mr, "last_full_token_usage", None), 4)},
        "cap": {"kv_tokens": getattr(scheduler, "max_total_num_tokens", None),
                "seats": getattr(scheduler, "max_running_requests", None)},
    }


def _round_or_none(v, nd: int):
    try:
        return None if v is None else round(float(v), nd)
    except (TypeError, ValueError):
        return None


_W_RE = re.compile(r"\b(W\d+[a-z]?)\b")
_TICKET_RE = re.compile(r"#(\d+[a-z]?)\b")


def stop_entry(exc: BaseException, reason: str = "scheduler_exception",
               now: Optional[float] = None) -> Dict[str, Any]:
    """A14: one stop of the rank as a record. ``code`` is the W-number plus the
    exception name when the text names a W-number (IPC §2.2 cause.code, never
    the bare W-number), else the exception name. Pure."""
    name = type(exc).__name__
    msg = str(exc)
    w = _W_RE.search(msg)
    t = _TICKET_RE.search(msg)
    return {"t": round(time.time() if now is None else float(now), 3), "reason": str(reason),
            "code": f"{w.group(1)}_{name}" if w else name, "exc": name,
            "ticket": f"#{t.group(1)}" if t else None, "text": msg[:STOP_TEXT_MAX]}


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
        self.stops: collections.deque = collections.deque(maxlen=STOPS_KEEP)
        self.stops_n = 0
        self._last_cap: Optional[dict] = None
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
        rec["stops"] = {"n": self.stops_n, "last": list(self.stops)}
        self._last_cap = rec.get("cap")
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
            self.sync_capacity()

    def sync_capacity(self) -> None:
        """C5: a changed pool/seat counter (the KV-stage dial moves
        max_total_num_tokens) into the RankState record -- from this thread,
        so the round path writes nothing. Unchanged = nothing written."""
        try:
            cap = self._last_cap or {}
            if cap.get("kv_tokens") is None and cap.get("seats") is None:
                return
            from sglang.srt.weg2 import rank_state

            rank_state.note_capacity(kv_tokens=cap.get("kv_tokens"), seats=cap.get("seats"))
        except Exception:  # noqa: BLE001 -- a display record never stops the rank
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


def note_stop(exc: BaseException, reason: str = "scheduler_exception") -> Optional[dict]:
    """A14: the rank's stop, from the scheduler's death path. Appended to
    ``stops`` and written AT ONCE (the process may not live to the next
    sample). Switch off / no timer: nothing, None. Never raises."""
    rs = _CURRENT
    if rs is None:
        return None
    try:
        entry = stop_entry(exc, reason)
        rs.stops.append(entry)
        rs.stops_n += 1
        rs.write_once()
        return entry
    except Exception:  # noqa: BLE001 -- the death path goes on regardless
        rs.failed += 1
        return None


def note_post_wake(census: dict) -> None:
    """The post-wake census (first passes after a wake, never a decode round):
    one attribute assignment; the timer writes it with the next sample."""
    rs = _CURRENT
    if rs is not None:
        rs.last_post_wake = dict(census)
