"""Line parsers for the weg2 boot logs (P, D and front).

One small regex per FIELD, not one per line: the instruments grow fields over
time, and a monolithic pattern turns every added field into a silent total
parse failure.  Same rule as /spinning/gpu-arb/devtools/logindex.py, whose
field patterns these mirror.

Honesty of the numbers (the reason this module exists):

* ``Prefill rank batch ... gpu-ms: X (compute Y, wait Z)`` is COMPUTE-HONEST:
  device time of that chunk on that rank.  tok/s = #new-token / compute_ms.
* ``Prefill batch ... input throughput (token/s)`` is WALL-CONFOUNDED: it
  divides by scheduler wall time and collapses whenever the loop idled.  It is
  parsed, but only ever shown under that name.
* ``Decode batch ... gen throughput (token/s)`` is tokens generated since the
  previous decode log line over the wall time between them -- the rate the
  clients see while decode runs.
* ``Decode rank batch ... bs, gpu-ms`` is device time per decode round.
"""

from __future__ import annotations

import calendar
import re
import time
from typing import Optional

# "[2026-09-27 09:20:28 PP1] ..." / "[2026-09-27 09:20:42,596] INFO weg2.front: ..."
RE_PREFIX = re.compile(
    r"^\[(\d{4})-(\d{2})-(\d{2})[ T](\d{2}):(\d{2}):(\d{2})(?:[,.](\d{1,6}))?Z?"
    r"(?: (PP|TP|DP)(\d+))?\] ?"
)


def _f(p):
    return re.compile(p)


F_NEW_SEQ = _f(r"#new-seq: (\d+)")
F_NEW_TOK = _f(r"#new-token: (\d+)")
F_CACHED = _f(r"#cached-token: (\d+)")
F_GPUMS = _f(r"gpu-ms: ([\d.]+) \(compute ([\d.]+), wait ([\d.]+)\)")
F_GPUMS_BARE = _f(r"gpu-ms: ([\d.]+)")
F_RUNREQ = _f(r"#running-req: (\d+)")
F_QUEUEREQ = _f(r"#queue-req: (\d+)")
# D phase seats (H95, D log): "WEG2 D-PHASE-SEATS (H95) epoch=.. handoff_n=6 parked_n=4 -> n=6 of cap 6 (CLAMPED ...)"
F_DSEATS = _f(r"WEG2 D-PHASE-SEATS \(H95\) epoch=\S+ handoff_n=(\d+) parked_n=(\d+) -> n=(\d+) of cap (\d+)")
# scheduler caps at start: "max_total_num_tokens=262144, ..., max_running_requests=6, ..."
F_SCHEDCAP = _f(r"^max_total_num_tokens=(\d+),.*\bmax_running_requests=(\d+)")
F_PENDTOK = _f(r"#pending-token: (\d+)")
F_INTPS = _f(r"input throughput \(token/s\): ([\d.]+)")
F_GENTPS = _f(r"gen throughput \(token/s\): ([\d.]+)")
F_FULLTOK = _f(r"#full token: (\d+)")
F_FULLUSE = _f(r"full token usage: ([\d.]+)")
F_ACCLEN = _f(r"accept len: ([\d.]+)")
F_ACCRATE = _f(r"accept rate: ([\d.]+)")
F_BS = _f(r"\bbs: (\d+)")
F_ROUND = _f(r"#round: (\d+)")
F_RANK_T = _f(r", t: (\d+\.\d+),")
# exact work times (WACH-OHNE-ARBEIT-0929: the rank lines are written a pass / a pipeline late)
F_FWD_PREFILL = _f(r"FWD-TIMING-PREFILL forward=(\d+) tokens=(\d+) .*?\btotal_ms=([\d.]+)")
F_FLUSH_WAIT = _f(r"TIMING-FLUSH-WAIT .*?\bforward=(\d+) .*?\bt_unix_ms=(\d+)")
F_ANON_EXTEND = _f(r"HOST-ANON-PASS pass=\d+ phase=EXTEND .*?\bwall_ms=(\d+)")
F_POST_WAKE0 = _f(r"WEG2-POST-WAKE-PASS n=0 mode=(\w+) .*?\bschedule_ms=(-?\d+) run_ms=(-?\d+)"
                  r".*?\bprepare_ms=(-?\d+)")
F_BUBBLE = _f(r"bubble_ms=([\d.]+)")
F_CUDAG = _f(r"cuda graph: (\w+)")

F_FLIP_BEGIN = _f(r"WEG2-FLIP begin epoch=(\d+) sleep=(\w+) wake=(\w+)")
F_FLIP_DONE = _f(r"WEG2-FLIP done epoch=(\d+) slept=(\w+) woke=(\w+)")
F_FLIP_TOTAL = _f(r"flip_total=(\d+(?:\.\d+)?) ms")
F_FLIP_DRAIN = _f(r"drain\+quiesce=(\d+(?:\.\d+)?) ms")
F_FLIP_SLEEP = _f(r"\bsleep=(\d+(?:\.\d+)?) ms")
F_FLIP_WAKE = _f(r"\bwake=(\d+(?:\.\d+)?) ms")
F_CORRIDOR_PHASE = _f(r"WEG2-CORRIDOR phase=(\w)\((\w+)\)")
F_ROUTE = _f(r"WEG2-ROUTE .*?\(awake=(\w+)\b.*?queue=(\d+)")
F_HEALTH = _f(r"WEG2-HEALTH group=(\w+) http_ok=(\w+) process_alive=(\w+)(?: streak=(\d+))?")

# Named stops in the P/D scheduler logs (operator list 2026-09-27): a refusal
# the runtime raises on purpose, a scheduler exception, or an OOM.  The
# harmless "FI-GRAPH-SPLIT off" status line also contains "SPLIT" and must not
# count.  DEBUG-HOLD is the #1223 hold a rank enters after such a stop -- the
# process is alive and deliberately parked, which from outside is a hang.
# 27.09. ~14:40Z: the bare "W27 " also matched PROSE in a WEG2-LAUNCH line (the
# launcher explains its guards) and raised a false alarm; a real W27 stop is a
# named refusal ("...Refused: #1233 W27 PP WIDTH DIVERGENCE REFUSED") under a
# scheduler traceback, which the patterns below still catch.
STOP_RE = _f(r"\b\w+Refused: #\d+|#791b|SPLIT refused|ADMISSION SPLIT|Traceback \(most recent|CUDA out of memory|DEBUG-HOLD rank=")
STOP_EXCLUDE = ("FI-GRAPH-SPLIT off", "WEG2-LAUNCH")


def stop_match(line: str) -> bool:
    return bool(STOP_RE.search(line)) and not any(x in line for x in STOP_EXCLUDE)
F_MODEL_PATH = _f(r"model_path='([^']*)'")
F_SERVED_NAME = _f(r"served_model_name='([^']*)'")
F_TP = _f(r"\btp_size=(\d+)")
F_PP = _f(r"\bpp_size=(\d+)")
F_SERVED = _f(r"WEG2-SERVED group=(\w+) leg=(\d+) rid=(\S+)")
F_PROMPT_T = _f(r"\bprompt_tokens=(\d+)")
F_CACHED_T = _f(r"\bcached_tokens=(\d+)")
F_COMPL_T = _f(r"\bcompletion_tokens=(\d+)")
F_LOADBACK = _f(r"^#988 LOADBACK rid=(\S+)")
F_ANCHOR = _f(r"\banchor_depth=(\d+)")
F_KV_APPLIED = _f(r"\bkv_applied=(\d+)")
F_MAMBA_RESUME = _f(r"^MAMBA-HOST-RESUME n=\d+: anchor accepted at depth=(\d+)")
F_STORE_INC = _f(r"^#\d+ STORE READ INCOMPLETE rid=(\S+) delivered=(\d+) deliverable=(\d+)")
F_PREFETCH = _f(r"^#\d+ PREFETCH (LANDED|REFUSED|DEFERRED|TIMEOUT)\b")
F_BOOT = _f(r"WEG2 BOOT tag=(\S+) tree=(\S+) @ (\w+)")
F_FORM_MODEL = _f(r"WEG2-FORM .*?\bmodel=(\S+)")
F_FORM = _f(r"WEG2-FORM (.*?) \(sources:")
F_EXC = _f(r"(?:^|\s)([A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception)): ")
F_LEVEL = _f(r"^\s*(ERROR|CRITICAL)\b")


def _num(rx, s, cast=float):
    m = rx.search(s)
    if not m:
        return None
    try:
        return cast(m.group(1))
    except ValueError:
        return None


def parse_ts(m) -> float:
    """UTC epoch seconds from a RE_PREFIX match (the containers log in UTC)."""
    y, mo, d, h, mi, s = (int(m.group(i)) for i in range(1, 7))
    frac = m.group(7)
    t = float(calendar.timegm((y, mo, d, h, mi, s, 0, 0, 0)))
    if frac:
        t += int(frac) / (10 ** len(frac))
    return t


def parse_line(line: str) -> Optional[dict]:
    """Return one event dict for a line we understand, else None.

    Every event carries ``t`` (UTC epoch, from the log's own timestamp),
    ``kind``, and for scheduler lines ``rk`` ('PP'/'TP') and ``rank``.
    """
    m = RE_PREFIX.match(line)
    if not m:
        return None
    rest = line[m.end():]
    ev = {"t": parse_ts(m)}
    if m.group(8):
        ev["rk"] = m.group(8)
        ev["rank"] = int(m.group(9))

    if rest.startswith("Prefill rank batch"):
        ev["kind"] = "prefill_rank"
        ev["new_tok"] = _num(F_NEW_TOK, rest, int)
        ev["cached"] = _num(F_CACHED, rest, int)
        g = F_GPUMS.search(rest)
        if g:
            ev["gpu_ms"] = float(g.group(1))
            ev["compute_ms"] = float(g.group(2))
            ev["wait_ms"] = float(g.group(3))
        else:
            ev["gpu_ms"] = _num(F_GPUMS_BARE, rest)
            ev["compute_ms"] = None
            ev["wait_ms"] = None
        ev["bubble_ms"] = _num(F_BUBBLE, rest)
        return ev
    if rest.startswith("Prefill batch"):
        ev["kind"] = "prefill_batch"
        ev["new_seq"] = _num(F_NEW_SEQ, rest, int)
        ev["new_tok"] = _num(F_NEW_TOK, rest, int)
        ev["cached"] = _num(F_CACHED, rest, int)
        ev["running"] = _num(F_RUNREQ, rest, int)
        ev["queue"] = _num(F_QUEUEREQ, rest, int)
        ev["pending_tok"] = _num(F_PENDTOK, rest, int)
        ev["wall_tps"] = _num(F_INTPS, rest)
        ev["full_use"] = _num(F_FULLUSE, rest)
        return ev
    if rest.startswith("Decode rank batch"):
        ev["kind"] = "decode_rank"
        ev["bs"] = _num(F_BS, rest, int)
        ev["round"] = _num(F_ROUND, rest, int)
        ev["t_exact"] = _num(F_RANK_T, rest)     # the round's own epoch stamp (ms), not the whole-second prefix
        ev["gpu_ms"] = _num(F_GPUMS_BARE, rest)
        return ev
    if rest.startswith("FWD-TIMING-PREFILL"):
        m2 = F_FWD_PREFILL.search(rest)
        if m2:
            ev.update(kind="fwd_prefill", forward=int(m2.group(1)), tokens=int(m2.group(2)),
                      total_ms=float(m2.group(3)))
            return ev
    if rest.startswith("TIMING-FLUSH-WAIT"):
        m2 = F_FLUSH_WAIT.search(rest)
        if m2:
            ev.update(kind="flush_wait", forward=int(m2.group(1)), t_unix=int(m2.group(2)) / 1000.0)
            return ev
    if rest.startswith("HOST-ANON-PASS"):
        m2 = F_ANON_EXTEND.search(rest)
        if m2:
            ev.update(kind="anon_extend", wall_ms=float(m2.group(1)))
            return ev
    if rest.startswith("WEG2-POST-WAKE-PASS n=0"):
        m2 = F_POST_WAKE0.search(rest)
        if m2:
            ev.update(kind="post_wake0", mode=m2.group(1), schedule_ms=int(m2.group(2)),
                      run_ms=int(m2.group(3)), prepare_ms=int(m2.group(4)))
            return ev
    if rest.startswith("Decode batch"):
        ev["kind"] = "decode_batch"
        ev["running"] = _num(F_RUNREQ, rest, int)
        ev["queue"] = _num(F_QUEUEREQ, rest, int)
        ev["full_tok"] = _num(F_FULLTOK, rest, int)
        ev["full_use"] = _num(F_FULLUSE, rest)
        ev["accept_len"] = _num(F_ACCLEN, rest)
        ev["accept_rate"] = _num(F_ACCRATE, rest)
        ev["gen_tps"] = _num(F_GENTPS, rest)
        cg = F_CUDAG.search(rest)
        ev["cuda_graph"] = (cg.group(1) == "True") if cg else None
        return ev

    # front-log families
    if "WEG2 D-PHASE-SEATS" in rest:
        d = F_DSEATS.search(rest)
        if d:
            ev.update(kind="d_seats", handoff_n=int(d.group(1)), parked_n=int(d.group(2)), n=int(d.group(3)),
                      cap=int(d.group(4)), clamped="CLAMPED" in rest)
            return ev
    if rest.startswith("max_total_num_tokens="):
        c = F_SCHEDCAP.search(rest)
        if c:
            ev.update(kind="sched_cap", max_total_tokens=int(c.group(1)), max_running=int(c.group(2)))
            return ev
    if "WEG2-FLIP " in rest:
        b = F_FLIP_BEGIN.search(rest)
        if b:
            ev.update(kind="flip_begin", epoch=int(b.group(1)), sleep=b.group(2), wake=b.group(3))
            return ev
        d = F_FLIP_DONE.search(rest)
        if d:
            ev.update(kind="flip_done", epoch=int(d.group(1)), slept=d.group(2), woke=d.group(3))
            ev["total_ms"] = _num(F_FLIP_TOTAL, rest)
            ev["drain_ms"] = _num(F_FLIP_DRAIN, rest)
            ev["sleep_ms"] = _num(F_FLIP_SLEEP, rest)
            ev["wake_ms"] = _num(F_FLIP_WAKE, rest)
            return ev
    c = F_CORRIDOR_PHASE.search(rest)
    if c and "WEG2-CORRIDOR" in rest:
        ev.update(kind="phase", awake=c.group(1), state=c.group(2))
        return ev
    r = F_ROUTE.search(rest)
    if r:
        ev.update(kind="route", awake=r.group(1), queue=int(r.group(2)))
        return ev
    h = F_HEALTH.search(rest)
    if h:
        ev.update(kind="health", group=h.group(1), http_ok=h.group(2) == "True",
                  alive=h.group(3) == "True", streak=int(h.group(4)) if h.group(4) else None)
        return ev
    # --- cache families (scheduler lines, per rank; the reader keeps rank 0) ---
    if rest.startswith("#988 LOADBACK"):
        m2 = F_LOADBACK.search(rest)
        ev.update(kind="loadback", rid=m2.group(1) if m2 else None,
                  depth=_num(F_ANCHOR, rest, int), kv_applied=_num(F_KV_APPLIED, rest, int))
        return ev
    if rest.startswith("MAMBA-HOST-RESUME"):
        m2 = F_MAMBA_RESUME.search(rest)
        ev.update(kind="mamba_host_resume", depth=int(m2.group(1)) if m2 else None)
        return ev
    m2 = F_STORE_INC.search(rest)
    if m2:
        ev.update(kind="store_read_incomplete", rid=m2.group(1), delivered=int(m2.group(2)),
                  deliverable=int(m2.group(3)))
        return ev
    m2 = F_PREFETCH.search(rest)
    if m2:
        ev.update(kind="prefetch", outcome=m2.group(1))
        return ev
    # --- front: one line per served leg of one request (rid): exact, no rank duplication ---
    m2 = F_SERVED.search(rest)
    if m2:
        ev.update(kind="served", group=m2.group(1), leg=int(m2.group(2)), rid=m2.group(3),
                  prompt_tokens=_num(F_PROMPT_T, rest, int), cached_tokens=_num(F_CACHED_T, rest, int),
                  completion_tokens=_num(F_COMPL_T, rest, int))
        return ev
    bt = F_BOOT.search(rest)
    if bt:
        ev.update(kind="boot", tag=bt.group(1), tree=bt.group(2), sha=bt.group(3))
        return ev
    fm = F_FORM_MODEL.search(rest)
    if fm:
        fo = F_FORM.search(rest)
        ev.update(kind="form", model=fm.group(1), form=fo.group(1) if fo else None)
        return ev
    if "server_args=" in rest or "ServerArgs(" in rest:
        mp = F_MODEL_PATH.search(rest)
        if mp:
            ev.update(kind="server_args", model_path=mp.group(1),
                      served_name=(F_SERVED_NAME.search(rest) or [None, None])[1],
                      tp=_num(F_TP, rest, int), pp=_num(F_PP, rest, int))
            return ev
    # Errors: only what the process itself classed as an error (level
    # ERROR/CRITICAL) or a scheduler exception report.  A WARNING that merely
    # names a TimeoutError is not an error and must not inflate the count.
    if F_LEVEL.match(rest) or " ERROR " in rest[:48] or " CRITICAL " in rest[:48] \
            or "hit an exception" in rest or rest.startswith("Traceback (most recent call last)"):
        x = F_EXC.search(rest)
        ev.update(kind="error", exc=x.group(1) if x else None, text=rest[:240])
        return ev
    return None


def now() -> float:
    return time.time()
