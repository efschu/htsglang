"""Live state of every weg2 boot whose logs are still being written.

Read-only by construction: the only thing this module does to a running
model is read the files its launcher already writes.  No request reaches the
server from here.

A boot is the triple ``<stem>.P.log``, ``<stem>.D.log``, ``<stem>.front.log``
(the weg2 launcher's naming); a lone ``<stem>.log`` is accepted as a
single-group boot.  Each file is tailed by byte offset (never slurped): on
first sight the tail starts ``BACKFILL_BYTES`` before EOF so the charts have
recent history, and the first ``HEAD_BYTES`` are read once for the identity
lines (launcher tag, model, form, topology).
"""

from __future__ import annotations

import collections
import glob
import os
import re
import threading
import time
from typing import Dict, List, Optional

from . import parse, redact, stops

DEFAULT_LOG_GLOBS = [
    "/spinning/docker-acceptance/*/evidence/boot_*.log",
    "/spinning/evidence-665-f1/boot_*.log",
]
BACKFILL_BYTES = None     # None = read every log from its start (totals "seit Boot" need all of it)
HEAD_BYTES = 512 * 1024
MAX_READ_PER_POLL = 8 * 1024 * 1024
LIVE_S = 90.0          # a log written within this many seconds is "live"
SHOW_S = 6 * 3600.0    # boots are listed while their newest log is younger
WINDOW_S = 60.0        # headline window for the rates
BUCKET_S = 5.0
HISTORY_S = 15 * 60.0
PHASE_GAP_S = 5.0      # no line of a class for longer than this -> that phase paused

# Launcher summary lines worth showing as the boot's "start form" (read-only
# view of what the weg2 launcher actually emitted; the full list is ~250 lines).
LAUNCH_KEYS = (
    "WEG2 BOOT tag=", "WEG2-FORM ", "NVML -> CUDA ordinal map", "POWER-LIMIT nvml",
    "SCHEDULING FLAGS AS EMITTED", "SCHEDULING KNOBS", "X PROVENANCE", "X CEILING",
    "IDLE POLICY", "budget P group=", "budget D group=", "host preflight", "WEG2-L2 D hicache_size",
)
MAX_LAUNCH_LINES = 40

RE_GROUP = re.compile(r"^(?P<stem>.+?)\.(?P<group>P|D|front)\.log$")


def split_log_name(path: str):
    """``(stem, group)`` for a boot log path; group is P, D, front or single."""
    base = os.path.basename(path)
    m = RE_GROUP.match(base)
    if m:
        return m.group("stem"), m.group("group")
    return base[:-4] if base.endswith(".log") else base, "single"


class Tail:
    """Offset-based line reader that survives truncation and partial lines."""

    def __init__(self, path: str, backfill: int = BACKFILL_BYTES):
        self.path = path
        self.offset = None
        self.backfill = backfill
        self.partial = b""
        self.size = 0
        self.mtime = 0.0
        self.ino = None

    def read_head(self, n: int = HEAD_BYTES) -> List[str]:
        try:
            with open(self.path, "rb") as fh:
                data = fh.read(n)
        except OSError:
            return []
        return data.decode("utf-8", "replace").splitlines()

    def poll(self) -> List[str]:
        try:
            st = os.stat(self.path)
        except OSError:
            return []
        self.size, self.mtime = st.st_size, st.st_mtime
        if self.offset is None or st.st_ino != self.ino or st.st_size < self.offset:
            self.ino = st.st_ino
            start = (max(0, st.st_size - self.backfill) if self.backfill else 0) if self.offset is None else 0
            self.offset = start
            self.partial = b""
            skip_first = start > 0
        else:
            skip_first = False
        if st.st_size == self.offset:
            return []
        try:
            with open(self.path, "rb") as fh:
                fh.seek(self.offset)
                data = fh.read(min(MAX_READ_PER_POLL, st.st_size - self.offset))
        except OSError:
            return []
        self.offset += len(data)
        data = self.partial + data
        lines = data.split(b"\n")
        self.partial = lines.pop()
        if skip_first and lines:
            lines = lines[1:]
        return [ln.decode("utf-8", "replace") for ln in lines]


def launch_lines(lines: List[str]) -> List[str]:
    """The launcher's key summary lines, de-duplicated, in order, prefix stripped."""
    out, seen = [], set()
    for ln in lines:
        i = ln.find("WEG2-LAUNCH ")
        if i < 0:
            continue
        body = redact.clean(ln[i + len("WEG2-LAUNCH "):].strip())
        if body is None or not any(k in body for k in LAUNCH_KEYS) or body in seen:
            continue
        seen.add(body)
        out.append(body[:600])
        if len(out) >= MAX_LAUNCH_LINES:
            break
    return out


def fill_series(ts, vals, flip_ts, first_t, bucket_s):
    """Hold / zero / none, see Boot.series.  Pure, unit-tested."""
    raw = [i for i, v in enumerate(vals) if v is not None]
    gaps = [b - a for a, b in zip(raw, raw[1:]) if b - a > 0]
    gaps.sort()
    typical = gaps[len(gaps) // 2] if gaps else 1
    hold = max(1, 2 * typical)
    out, last_i = [], None
    for i, v in enumerate(vals):
        t = ts[i]
        if v is not None:
            out.append(v)
            last_i = i
            continue
        if first_t is None or t + bucket_s <= first_t:
            out.append(None)
            continue
        if last_i is not None and i - last_i <= hold and not any(ts[last_i] < f <= t + bucket_s for f in flip_ts):
            out.append(vals[last_i])
        else:
            out.append(0.0)
    return out


def _rate(tok, ms):
    return (tok / (ms / 1000.0)) if (tok and ms and ms > 0) else None


ONE_S = 1.0            # the "jetzt 1 s" window
GEN_GAP_FACTOR = 2.5   # a Decode batch line later than this x the usual line interval covers a pause


def mark_gen_artefacts(batches, pauses, gap_s=PHASE_GAP_S, factor=GEN_GAP_FACTOR):
    """Drop the 'gen throughput' of Decode batch lines whose interval was not
    continuous decode.  Pure, unit-tested.

    sglang's figure is tokens since the PREVIOUS Decode batch line / the wall
    time since it.  When decode paused in between (a flip, an extend of the
    same group, idle), the pause sits in the denominator: measured rc12z20
    12:34:57 bs6 13.65 tok/s, 4 s later bs5 133.  Such a line keeps its raw
    value in ``gen_tps_raw`` (the tokens are real, the rate is not) and gets
    ``gen_tps = None`` plus the reason in ``gen_art``:

      * ``first``  -- no previous line: the interval is unknown;
      * ``pause``  -- a pause stamp (flip begin/done, own-group prefill line)
        fell in (previous line, this line + 1 s);
      * ``gap``    -- the line came later than max(gap_s, factor x median of
        the preceding clean intervals).

    ``batches``: the group's Decode batch events in log order; ``pauses``:
    sorted stamps.  Idempotent (always re-derived from ``gen_tps_raw``)."""
    import bisect
    prev_t = None
    dts = collections.deque(maxlen=20)
    for e in batches:
        raw = e["gen_tps_raw"] if "gen_tps_raw" in e else e.get("gen_tps")
        e["gen_tps_raw"] = raw
        t = e["t"]
        art = None
        if raw is not None:
            if prev_t is None:
                art = "first"
            else:
                dt = t - prev_t
                i = bisect.bisect_right(pauses, prev_t)
                if i < len(pauses) and pauses[i] < t + 1.0:
                    art = "pause"
                else:
                    med = sorted(dts)[len(dts) // 2] if dts else None
                    if dt > max(gap_s, factor * med if med else gap_s):
                        art = "gap"
                    else:
                        dts.append(max(dt, 1.0))
        e["gen_art"] = art
        e["gen_tps"] = None if art else raw
        prev_t = t


def _mid(t):
    """P/D log stamps are whole seconds, truncated: the line fell in [t, t+1)."""
    return t + 0.5 if t == int(t) else t


def spread_rate(intervals, t_ref, w=ONE_S):
    """tok/s over [t_ref - w, t_ref]: every interval (start, end, tokens)
    spreads its tokens evenly over its own compute interval; the rate is the
    sum of the shares inside the window, / w.  Pure, unit-tested."""
    lo = t_ref - w
    tok = 0.0
    for s, e, n in intervals:
        if e <= lo or s >= t_ref or not n:
            continue
        d = e - s
        if d <= 0:
            tok += n if lo < e <= t_ref else 0
            continue
        tok += n * (min(e, t_ref) - max(s, lo)) / d
    return tok / w


def compute_rate(intervals, w=ONE_S):
    """tok/s over the newest ``w`` seconds of COMPUTE time of one rank: walk
    back from its newest line, each line's tokens spread evenly over its own
    compute interval.  Compute-honest like the tile's 60-s figure (bubbles
    and idle between lines do not dilute it), and it does not depend on the
    whole-second log stamps.  Pure, unit-tested."""
    acc, tok = 0.0, 0.0
    for s, e, n in reversed(intervals):
        d = e - s
        if d <= 0:
            continue
        if acc + d >= w:
            tok += (n or 0) * (w - acc) / d
            acc = w
            break
        acc += d
        tok += n or 0
    return tok / acc if acc > 0 else 0.0


def one_s_rate(per_rank, now, other_after=None, gone_after=None, gap_s=PHASE_GAP_S, rate=None):
    """The live 1-s rate of one class (user order 27.09.: 'es zeigt meistens
    null an und springt dann auf 5000 / 10000').

    ``per_rank``: {rank: [(start, end, tokens), ...]} sorted by end.  Each
    rank's window is anchored at ITS newest logged end (``rate``: prefill
    compute_rate = the newest 1 s of compute; default spread_rate over wall
    time, used for the decode rounds) (the log lags, and a
    PP stage logs a chunk after the stage before it already works on the
    next), so between two chunks of a running class there is no 0.  The
    class rate is the slowest rank's, as in the tile.  0 when the class is
    no longer active: its newest line is older than max(gap_s, 1.5 x the last
    interval + 1 s), or another class of the group / a flip putting the group
    to sleep came after it (``other_after`` / ``gone_after``: epoch or None).
    """
    ends = [iv[-1][1] for iv in per_rank.values() if iv]
    if not ends:
        return 0.0
    t_last = max(ends)
    last_dur = max((iv[-1][1] - iv[-1][0]) for iv in per_rank.values() if iv)
    if now - t_last > max(gap_s, 1.5 * last_dur + 1.0):
        return 0.0
    if other_after is not None and other_after > t_last + 0.5:
        return 0.0
    # a flip putting the group to sleep begins after draining its last chunk; the
    # chunk's stamp is truncated to the second, its end is taken mid-second
    if gone_after is not None and gone_after > t_last - 0.5:
        return 0.0
    rates = []
    for iv in per_rank.values():
        if not iv or iv[-1][1] < t_last - max(gap_s, 1.5 * last_dur + 1.0):
            continue            # a rank that stopped logging does not bound the class
        rates.append(rate(iv) if rate else spread_rate(iv, iv[-1][1]))
    return min(rates) if rates else 0.0


def max_rate(per_rank, t_lo, t_hi, w=3.0, rate=None):
    """Maximum of the class rate (MIN over the ranks) over all anchor time
    points in [t_lo, t_hi]: an anchor is the end (``end``) of any interval of
    any rank in that span.  At anchor ``t`` a rank uses only its intervals
    with ``end <= t`` and counts only if its newest such interval is not
    older than max(PHASE_GAP_S, 1.5 x its duration + 1 s) before ``t``
    (as in one_s_rate).  Per-rank rate: ``rate(iv, w)`` when given (prefill:
    compute_rate with w=3.0), else spread_rate(iv, t, w) (decode).  The class
    is the min over the counting ranks; the result is the max over the
    anchors, or None when there is no anchor or no rate > 0 (a silent or
    finished boot shows the dash, not 0)."""
    anchors = sorted({e for iv in per_rank.values() for _, e, _ in iv if t_lo <= e <= t_hi})
    best = None
    for t in anchors:
        rates = []
        for iv in per_rank.values():
            iv_t = [x for x in iv if x[1] <= t]
            if not iv_t:
                continue
            s, e, _ = iv_t[-1]
            if t - e > max(PHASE_GAP_S, 1.5 * (e - s) + 1.0):
                continue        # too old to count at this anchor
            rates.append(rate(iv_t, w) if rate else spread_rate(iv_t, t, w))
        if not rates:
            continue
        v = min(rates)
        if best is None or v > best:
            best = v
    return best if best and best > 0 else None


def _flip_intervals(begins, dones, open_begin, t1):
    """Pair ``WEG2-FLIP begin`` with its ``done`` (the done carries the NEW
    epoch = begin epoch + 1, and slept/woke = sleep/wake).  Without a begin
    in memory the start is done - flip_total.  An open flip runs to ``t1``."""
    out = []
    bs = sorted(begins, key=lambda e: e["t"])
    for d in dones:
        b = None
        for e in bs:
            if e["t"] <= d["t"] and e.get("sleep") == d.get("slept") and (
                    e.get("epoch") == (d.get("epoch") or 0) - 1 or d["t"] - e["t"] < 600):
                b = e
        start = b["t"] if b else (d["t"] - (d.get("total_ms") or 0) / 1000.0)
        out.append({"b": start, "d": d["t"], "slept": d.get("slept"), "woke": d.get("woke"),
                    "total_ms": d.get("total_ms"), "drain_ms": d.get("drain_ms"), "open": False})
    if open_begin is not None and open_begin["t"] <= t1:
        out.append({"b": open_begin["t"], "d": t1, "slept": open_begin.get("sleep"),
                    "woke": open_begin.get("wake"), "total_ms": None, "drain_ms": None, "open": True})
    out.sort(key=lambda f: f["b"])
    return out


def _run_stats(cls, evs):
    """Mean rate of one phase run.  Prefill: compute-honest, per rank
    sum(#new-token) / sum(compute-ms), the SLOWEST rank (as in the tiles).
    Decode: mean 'gen throughput' of the run's 'Decode batch' lines."""
    if cls == "dec":
        g = [e["gen_tps"] for e in evs if e.get("gen_tps") is not None]
        run = [e["running"] for e in evs if e.get("running") is not None]
        rb = [e["bs"] for e in evs if e.get("kind") == "decode_rank" and e.get("bs") is not None]
        rounds = sum(1 for e in evs if e.get("kind") == "decode_rank")
        return {"tps": (sum(g) / len(g)) if g else None, "n": len(g), "rounds": rounds,
                "bs_min": min(rb) if rb else None, "bs_max": max(rb) if rb else None,
                "bs_mean": (sum(rb) / len(rb)) if rb else None,
                "bs": (sum(run) / len(run)) if run else None}
    per = {}
    for e in evs:
        if e.get("compute_ms") is None:
            continue
        a = per.setdefault("%s%s" % (e.get("rk", ""), e.get("rank", 0)), [0, 0.0, 0])
        a[0] += e.get("new_tok") or 0
        a[1] += e["compute_ms"]
        a[2] += 1
    rated = [_rate(a[0], a[1]) for a in per.values()]
    rated = [r for r in rated if r]
    r0 = per.get("PP0") or per.get("TP0") or (next(iter(per.values())) if per else None)
    return {"tps": min(rated) if rated else None, "tok": r0[0] if r0 else 0,
            "n": r0[2] if r0 else len(evs)}


def phase_timeline(acts, flips, t0, t1, first_t=None, awake_hint=None, gap_s=PHASE_GAP_S, work_from=None):
    """Segments of the phase bar over [t0, t1].  Pure, unit-tested.

    ``acts``: dicts {t, s, cls, ev}: ``t`` = the log line's stamp (the END of
    the work it reports), ``s`` = its start where the line says how long it
    took (prefill: t - gpu-ms), else ``t``; ``cls`` P / D (prefill of that
    group) or dec.  ``flips``: from _flip_intervals.

    Rules, nothing guessed beyond them:
      * a run = consecutive lines of ONE class, no gap > ``gap_s``, no flip
        in between; it starts at its first line's start, or at the end of the
        piece before it when that is at most ``gap_s`` earlier (the work
        between two lines is still that work), else 1 s before its first line;
      * a flip is grey from ``WEG2-FLIP begin`` to ``done``; while the old
        group still logs work inside it (drain), that work is shown as work
        and the grey starts after its last line;
      * whatever is left is ``idle`` (awake group from the flips: no work);
        before the boot's first work line (``work_from``) it is the boot
        loading (``boot``), not an awake group waiting;
      * the run still going (last line within ``gap_s`` of t1) reaches t1.
    """
    acts = sorted(acts, key=lambda a: a["t"])
    greys = []
    for f in flips:
        if f["d"] < t0 - 600 or f["b"] > t1:
            continue
        inside = [a["t"] for a in acts if f["b"] <= a["t"] <= f["d"]]
        gs = max([f["b"]] + inside)
        greys.append(dict(f, s=min(gs, f["d"]), e=f["d"]))

    def flip_between(a, b):
        return any(g["s"] < b and g["e"] > a for g in greys)

    runs = []
    for a in acts:
        r = runs[-1] if runs else None
        if r and r["cls"] == a["cls"] and a["t"] - r["e"] <= gap_s and not flip_between(r["e"], a["t"]):
            r["e"] = a["t"]
            r["s"] = min(r["s"], a["s"])
            r["evs"].append(a["ev"])
        else:
            runs.append({"cls": a["cls"], "s": min(a["s"], a["t"] - 1.0) if a["s"] >= a["t"] else a["s"],
                         "e": a["t"], "evs": [a["ev"]], "t_first": a["t"]})
    # starts: fill a short gap from the piece before, never overlap it
    pieces = sorted([("run", r) for r in runs] + [("flip", g) for g in greys], key=lambda p: p[1]["e"])
    prev_end = None
    for kind, p in pieces:
        if kind == "run" and prev_end is not None:
            if p["t_first"] - prev_end <= gap_s or p["s"] < prev_end:
                p["s"] = prev_end
        if kind == "flip" and prev_end is not None and p["s"] < prev_end:
            p["s"] = prev_end
        p["s"] = min(p["s"], p["e"])
        prev_end = p["e"] if prev_end is None else max(prev_end, p["e"])
    if runs and not any(g["e"] > runs[-1]["e"] for g in greys) and t1 - runs[-1]["e"] <= gap_s:
        runs[-1]["e"] = t1
        runs[-1]["running"] = True

    def awake_at(t):
        before = [g for g in greys if g["e"] <= t and not g["open"]]
        if before:
            return before[-1]["woke"]
        after = [g for g in greys if g["s"] >= t]
        return after[0]["slept"] if after else awake_hint

    segs = []
    for r in runs:
        st = _run_stats(r["cls"], r["evs"])
        segs.append(dict(st, k=r["cls"], s=r["s"], e=r["e"], running=bool(r.get("running"))))
    for g in greys:
        segs.append({"k": "flip", "s": g["s"], "e": g["e"], "slept": g["slept"], "woke": g["woke"],
                     "total_ms": g["total_ms"], "drain_ms": g["drain_ms"], "open": g["open"]})
    segs.sort(key=lambda x: x["s"])
    lo = max(t0, first_t) if first_t else t0
    out, cur = [], lo
    for x in segs:
        if x["e"] <= lo or x["s"] >= t1:
            continue
        if x["s"] - cur > 0.5:
            out.append({"k": "idle", "s": cur, "e": x["s"], "awake": awake_at((cur + x["s"]) / 2)})
        x = dict(x, s=max(x["s"], lo, cur), e=min(x["e"], t1))
        if x["e"] > x["s"]:
            out.append(x)
        cur = max(cur, x["e"])
    if t1 - cur > 0.5 and (first_t is None or first_t < t1):
        out.append({"k": "idle", "s": cur, "e": t1, "awake": awake_at((cur + t1) / 2)})
    for x in out:
        if x["k"] == "idle" and work_from is not None and x["e"] <= work_from + 0.5:
            x["awake"], x["boot"] = None, True
        x["s"], x["e"] = round(x["s"], 2), round(x["e"], 2)
    return out


class Boot:
    """Everything one boot's logs have said, bounded in memory."""

    def __init__(self, stem: str, dirpath: str):
        self.stem = stem
        self.dir = dirpath
        self.tails: Dict[str, Tail] = {}
        self.meta: dict = {"stem": stem, "dir": dirpath}
        self.ev = {k: collections.deque(maxlen=n) for k, n in (
            ("P_prefill_rank", 6000), ("P_prefill_batch", 3000),
            ("D_prefill_rank", 6000), ("D_prefill_batch", 3000),
            ("P_decode_batch", 2000), ("D_decode_batch", 4000),
            ("P_decode_rank", 2000), ("D_decode_rank", 20000),
            ("flips", 400), ("flip_begins", 400), ("errors", 200), ("stops", 60),
            ("single_prefill_rank", 6000), ("single_prefill_batch", 3000),
            ("single_decode_batch", 4000), ("single_decode_rank", 20000),
        )}
        self.last = {}          # kind -> last event
        self.health = {}        # group -> last health event
        self.flip_open = None
        self.counts = collections.Counter()
        self._last_t = {}       # group -> newest log timestamp seen in that file
        self.first_t = None     # first timestamp of this boot's logs
        self.first_work_t = None   # first prefill/decode line: before it the boot is still loading
        self.lock = threading.Lock()
        # totals since boot (whole file read) and a 60-s event window, per group
        self.tot = collections.defaultdict(collections.Counter)
        self.win = collections.deque(maxlen=20000)
        self.rank_tot = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0.0]))
        self.served_ev = collections.deque(maxlen=5000)    # (t, completion_tokens) of served D legs

    def add_file(self, group: str, path: str):
        if group in self.tails:
            return
        t = Tail(path)
        self.tails[group] = t
        head = t.read_head()
        if group == "front":
            self.meta["launch"] = launch_lines(head)
        for line in head:
            ev = parse.parse_line(line)
            if ev and ev["kind"] in ("boot", "form", "server_args"):
                self._ingest(group, ev, head=True)

    @property
    def newest_mtime(self) -> float:
        return max((t.mtime for t in self.tails.values()), default=0.0)

    def poll(self):
        with self.lock:
            self._poll()

    def read_progress(self) -> float:
        size = sum(t.size for t in self.tails.values())
        done = sum((t.offset or 0) for t in self.tails.values())
        return 1.0 if size == 0 else min(1.0, done / size)

    def _poll(self):
        for group, t in self.tails.items():
            for line in t.poll():
                ev = parse.parse_line(line)
                if ev:
                    self._last_t[group] = ev["t"]
                    if self.first_t is None or ev["t"] < self.first_t:
                        self.first_t = ev["t"]
                    self._ingest(group, ev)
                elif group != "front":
                    m = parse.RE_PREFIX.match(line)
                    if m:
                        self._last_t[group] = parse.parse_ts(m)
                if group != "front" and parse.stop_match(line):
                    self._stop(group, line, t)

    def _stop(self, group: str, line: str, tail: "Tail"):
        """A named stop.  Unprefixed lines (the exception text under a
        traceback) take the newest timestamp seen in the same file."""
        m = parse.RE_PREFIX.match(line)
        ts = parse.parse_ts(m) if m else self._last_t.get(group, tail.mtime)
        text = redact.clean(line[m.end():] if m else line)
        if text is None:
            return
        self.ev["stops"].append({
            "t": ts, "group": group, "text": text.strip()[:400],
            # the bare "Traceback (most recent call last):" says THAT, the
            # named line (W27 ..., OOM ...) says WHY -- the view prefers WHY
            "bare": "Traceback (most recent" in text and not any(
                k in text for k in ("Refused: #", "#791b", "SPLIT refused", "ADMISSION SPLIT", "CUDA out of memory")),
        })
        self.counts["stop"] += 1

    def last_activity(self, group: Optional[str] = None) -> Optional[float]:
        """Newest prefill/decode line (any group, or one group)."""
        best = None
        for g in ((group,) if group else ("P", "D", "single")):
            for k in ("prefill_rank", "prefill_batch", "decode_batch", "decode_rank"):
                e = self.last.get("%s_%s" % (g, k))
                if e and (best is None or e["t"] > best):
                    best = e["t"]
        return best

    def _ingest(self, group: str, ev: dict, head: bool = False):
        k = ev["kind"]
        if k == "boot":
            self.meta.update(tag=ev["tag"], tree=ev["tree"], sha=ev["sha"])
            return
        if k == "form":
            self.meta["model"] = ev["model"]
            self.meta["form"] = ev["form"]
            return
        if k == "server_args":
            self.meta.setdefault("model", os.path.basename(ev["model_path"].rstrip("/")))
            self.meta.setdefault("model_path", ev["model_path"])
            topo = self.meta.setdefault("topology", {})
            topo[group] = {"tp": ev.get("tp"), "pp": ev.get("pp")}
            return
        if head:
            return
        self.counts[k] += 1
        rank0 = ev.get("rank") in (None, 0)
        if k in ("prefill_rank", "prefill_batch", "decode_batch", "decode_rank"):
            if self.first_work_t is None or ev["t"] < self.first_work_t:
                self.first_work_t = ev["t"]
            self.ev["%s_%s" % (group, k)].append(ev)
            self.last["%s_%s" % (group, k)] = ev
            if k == "prefill_rank" and ev.get("compute_ms") is not None:
                a = self.rank_tot[group]["%s%s" % (ev.get("rk", ""), ev.get("rank", 0))]
                a[0] += ev.get("new_tok") or 0
                a[1] += ev["compute_ms"]
            if k == "decode_rank" and rank0 and ev.get("gpu_ms"):
                self.tot[group]["dec_gpu_ms"] += ev["gpu_ms"]
                self.tot[group]["dec_rounds"] += 1
            if k == "prefill_batch" and rank0:
                # one "Prefill batch" line per chunk on the FIRST rank only (PP0/TP0):
                # the other ranks log the same chunk again
                new, cached = ev.get("new_tok") or 0, ev.get("cached") or 0
                tt = self.tot[group]
                tt["new"] += new
                tt["cached"] += cached
                tt["chunks"] += 1
                self.win.append((ev["t"], group, "pb", new, cached))
            return
        if k in ("loadback", "mamba_host_resume", "store_read_incomplete", "prefetch"):
            if not rank0:
                return
            tt = self.tot[group]
            if k == "loadback":
                tt["loadback_n"] += 1
                tt["loadback_tok"] += ev.get("depth") or 0
                self.win.append((ev["t"], group, "lb", ev.get("depth") or 0, 0))
            elif k == "mamba_host_resume":
                tt["mamba_n"] += 1
                tt["mamba_tok"] += ev.get("depth") or 0
                self.win.append((ev["t"], group, "mb", ev.get("depth") or 0, 0))
            elif k == "store_read_incomplete":
                tt["l3inc_n"] += 1
                tt["l3inc_delivered"] += ev["delivered"]
                tt["l3inc_deliverable"] += ev["deliverable"]
                self.win.append((ev["t"], group, "l3", ev["delivered"], ev["deliverable"]))
            else:
                tt["prefetch_" + ev["outcome"].lower()] += 1
            return
        if k == "served":
            key = "served_" + ev["group"]
            tt = self.tot[key]
            tt["n"] += 1
            tt["prompt"] += ev.get("prompt_tokens") or 0
            tt["cached"] += ev.get("cached_tokens") or 0
            tt["completion"] += ev.get("completion_tokens") or 0
            if ev.get("completion_tokens"):
                self.served_ev.append((ev["t"], ev["completion_tokens"]))
            self.win.append((ev["t"], key, "sv", ev.get("prompt_tokens") or 0, ev.get("cached_tokens") or 0))
            return
        if k == "flip_begin":
            self.flip_open = ev
            self.ev["flip_begins"].append(ev)
            return
        if k == "flip_done":
            self.flip_open = None
            self.ev["flips"].append(ev)
            self.last["awake"] = {"t": ev["t"], "awake": ev["woke"], "src": "flip"}
            return
        if k in ("phase", "route"):
            prev = self.last.get("awake")
            if not prev or prev["t"] <= ev["t"]:
                self.last["awake"] = {"t": ev["t"], "awake": ev["awake"], "src": k}
            if k == "route":
                self.last["queue"] = {"t": ev["t"], "queue": ev["queue"]}
            return
        if k == "health":
            self.health[ev["group"]] = ev
            return
        if k in ("d_seats", "sched_cap"):
            if ev.get("rank") in (None, 0):
                self.last["%s_%s" % (group, k)] = ev
            return
        if k == "error":
            ev["group"] = group
            ev["text"] = redact.clean(ev.get("text"))
            if ev["text"] is not None:
                self.ev["errors"].append(ev)

    # ---------------------------------------------------------------- views

    @staticmethod
    def _prefill_window(ranks, batches, t0):
        """Compute-honest prefill rate over rank lines with t >= t0.

        Per rank: sum(#new-token) / sum(compute_ms).  The group's rate is the
        SLOWEST rank's (a PP stage or a TP peer bounds the chunk), which is
        the honest pipeline figure.  Rows without the gpu-ms split (some TP
        peers log only the counts) are counted but not rated.
        """
        per = {}
        unrated = 0
        for e in ranks:
            if e["t"] < t0:
                continue
            ms = e.get("compute_ms")
            if ms is None:
                unrated += 1
                continue
            a = per.setdefault("%s%s" % (e.get("rk", ""), e.get("rank", 0)),
                               {"tok": 0, "ms": 0.0, "wait_ms": 0.0, "chunks": 0, "cached": 0})
            a["tok"] += e.get("new_tok") or 0
            a["ms"] += ms
            a["wait_ms"] += e.get("wait_ms") or 0.0
            a["chunks"] += 1
            a["cached"] += e.get("cached") or 0
        for a in per.values():
            a["tps"] = _rate(a["tok"], a["ms"])
            a["mean_chunk"] = a["tok"] / a["chunks"] if a["chunks"] else None
        rated = [a["tps"] for a in per.values() if a["tps"]]
        wall = [b["wall_tps"] for b in batches if b["t"] >= t0 and b.get("wall_tps") is not None]
        rank0 = per.get("PP0") or per.get("TP0") or (next(iter(per.values())) if per else None)
        return {
            "tps": min(rated) if rated else None,
            "ranks": dict(sorted(per.items())),
            "unrated_rows": unrated,
            "tokens": rank0["tok"] if rank0 else 0,
            "chunks": rank0["chunks"] if rank0 else 0,
            "mean_chunk": rank0["mean_chunk"] if rank0 else None,
            "cached": rank0["cached"] if rank0 else 0,
            "wall_confounded_tps": (sum(wall) / len(wall)) if wall else None,
        }

    def _prefill_view(self, g: str, now: float) -> dict:
        ranks = self.ev["%s_prefill_rank" % g]
        batches = self.ev["%s_prefill_batch" % g]
        win = self._prefill_window(ranks, batches, now - WINDOW_S)
        # last burst: the rows of the newest 20 s of activity, however old
        last_t = ranks[-1]["t"] if ranks else None
        burst = self._prefill_window(ranks, batches, last_t - 20.0) if last_t else None
        lb = self.last.get("%s_prefill_batch" % g)
        return {
            "window_s": WINDOW_S,
            "one_s": self._one_s(g, "prefill", now),
            "max3s_120": self._max3s(g, "prefill", now),
            "now": win,
            "last_burst": burst,
            "last_t": last_t,
            "queue": lb.get("queue") if lb else None,
            "pending_tok": lb.get("pending_tok") if lb else None,
        }

    def _intervals(self, g: str, kind: str, since: float) -> Dict[str, list]:
        """{rank: [(start, end, tokens)]} of one class, newest ``since`` on.
        Prefill: each rank's 'Prefill rank batch' line, interval = its
        compute-ms (gpu-ms where the line has no split) up to the
        (mid-second) stamp, tokens #new-token.  Decode:
        each round of 'Decode rank batch' rank 0, interval = its gpu-ms up to
        its exact t:, tokens = bs x accept len of the newest 'Decode batch'
        line before it."""
        out: Dict[str, list] = {}
        if kind == "prefill":
            for e in self.ev["%s_prefill_rank" % g]:
                ms = e.get("compute_ms") or e.get("gpu_ms")
                if e["t"] < since - 1 or not ms:
                    continue
                t = _mid(e["t"])
                out.setdefault("%s%s" % (e.get("rk", ""), e.get("rank", 0)), []).append(
                    (t - ms / 1000.0, t, e.get("new_tok") or 0))
        else:
            # 'gen throughput' of a Decode batch line = tokens since the previous
            # Decode batch line / the time between; those tokens are spread evenly
            # over the rounds that fell in between (bs x accept len over-counts:
            # measured 27.09. NF bs 4 -> ~180 against gen throughput ~100)
            # raw figure on purpose: gen throughput x (interval) are the tokens of
            # that interval even when a pause made the RATE an artefact
            batches = [e for e in self.ev["%s_decode_batch" % g]
                       if e.get("gen_tps_raw", e.get("gen_tps")) is not None and e["t"] >= since - 30]
            rows = [e for e in self.ev["%s_decode_rank" % g]
                    if e["t"] >= since - 30 and e.get("rank", 0) == 0 and e.get("t_exact")]
            if rows and batches:
                rows.sort(key=lambda e: e["t_exact"])
                bt = [_mid(e["t"]) for e in batches]
                per_round = [None] * len(rows)
                j = 0
                for k in range(1, len(batches)):
                    lo, hi = bt[k - 1], bt[k]
                    idx = []
                    while j < len(rows) and rows[j]["t_exact"] <= hi:
                        if rows[j]["t_exact"] > lo:
                            idx.append(j)
                        j += 1
                    if idx:
                        tok = batches[k].get("gen_tps_raw", batches[k].get("gen_tps")) * (hi - lo) / len(idx)
                        for i in idx:
                            per_round[i] = tok
                last_tok = None
                for i, e in enumerate(rows):
                    if per_round[i] is None:
                        per_round[i] = last_tok       # after the newest batch line: the newest per-round figure
                    last_tok = per_round[i] if per_round[i] is not None else last_tok
                    if per_round[i] is None or e["t_exact"] < since:
                        continue
                    ms = e.get("gpu_ms") or 0.0
                    out.setdefault("TP0", []).append((e["t_exact"] - ms / 1000.0, e["t_exact"], per_round[i]))
        for v in out.values():
            v.sort(key=lambda x: x[1])
        return out

    def _one_s(self, g: str, kind: str, now: float) -> float:
        if self.newest_mtime and now - self.newest_mtime > 120.0:
            return 0.0          # a finished boot: nothing runs now
        per = self._intervals(g, kind, now - 120.0)
        if kind == "decode" and not per:
            # no per-round lines with t: in this boot: the newest gen throughput while active
            last = self.last.get("%s_decode_batch" % g)
            if last and last.get("gen_tps") is not None and now - _mid(last["t"]) <= PHASE_GAP_S:
                return last["gen_tps"]
            return 0.0
        other_kind = "decode" if kind == "prefill" else "prefill"
        other = None
        if g in ("D", "single"):
            ends = [iv[-1][1] for iv in self._intervals(g, other_kind, now - 120.0).values() if iv]
            other = max(ends) if ends else None
        gone = None
        for e in list(self.ev["flip_begins"])[-4:] + ([self.flip_open] if self.flip_open else []):
            if e and e.get("sleep") == g:
                gone = e["t"] if gone is None else max(gone, e["t"])
        return one_s_rate(per, now, other_after=other, gone_after=gone,
                          rate=compute_rate if kind == "prefill" else None)

    def _max3s(self, g: str, kind: str, now: float):
        if self.newest_mtime and now - self.newest_mtime > 120.0:
            return None         # a finished boot: nothing ran in the last 120 s
        per = self._intervals(g, kind, now - 123.0)
        if not per:
            return None
        return max_rate(per, now - 120.0, now, w=3.0,
                        rate=compute_rate if kind == "prefill" else None)

    def _round_bs(self, g: str) -> Optional[dict]:
        """bs of the newest 'Decode rank batch' round of rank 0: the requests
        that really compute now (below the seats when KV is short)."""
        for e in reversed(self.ev["%s_decode_rank" % g]):
            if e.get("rank", 0) == 0 and e.get("bs") is not None:
                return {"bs": e["bs"], "t": e.get("t_exact") or e["t"]}
        return None

    def _decode_view(self, g: str, now: float) -> dict:
        rows = [e for e in self.ev["%s_decode_batch" % g] if e["t"] >= now - WINDOW_S]
        rr = [e for e in self.ev["%s_decode_rank" % g]
              if e["t"] >= now - WINDOW_S and e.get("rank", 0) == 0 and e.get("gpu_ms")]
        last = self.last.get("%s_decode_batch" % g)
        gen = [e["gen_tps"] for e in rows if e.get("gen_tps") is not None]
        # the newest figure that is a real decode rate (mark_gen_artefacts), not
        # the first line after a flip or extend
        clean_last = next((e["gen_tps"] for e in reversed(self.ev["%s_decode_batch" % g])
                           if e.get("gen_tps") is not None), None)
        compute = None
        if rr and last and last.get("accept_len"):
            ms = sum(e["gpu_ms"] for e in rr)
            bs = sum((e.get("bs") or 0) for e in rr)
            compute = (bs * last["accept_len"]) / (ms / 1000.0) if ms > 0 else None
        return {
            "window_s": WINDOW_S,
            "one_s": self._one_s(g, "decode", now),
            "max3s_120": self._max3s(g, "decode", now),
            "gen_tps": (sum(gen) / len(gen)) if gen else None,
            "gen_tps_last": clean_last,
            "artefacts": sum(1 for e in rows if e.get("gen_art")),
            "running": last.get("running") if last else None,
            "queue_req": last.get("queue") if last else None,
            "round_bs": self._round_bs(g),
            "seats": self.last.get("%s_d_seats" % g),
            "max_running": (self.last.get("%s_sched_cap" % g) or {}).get("max_running"),
            "accept_len": last.get("accept_len") if last else None,
            "accept_rate": last.get("accept_rate") if last else None,
            "full_use": last.get("full_use") if last else None,
            "cuda_graph": last.get("cuda_graph") if last else None,
            "rows": len(rows),
            "compute_tps": compute,
            "compute_rounds": len(rr),
            "last_t": last["t"] if last else None,
        }

    def series(self, now: float) -> dict:
        """5-s buckets over the last 15 min: the chart feed."""
        n = int(HISTORY_S / BUCKET_S)
        t_start = (now // BUCKET_S) * BUCKET_S - (n - 1) * BUCKET_S
        ts = [t_start + i * BUCKET_S for i in range(n)]

        def idx(t):
            i = int((t - t_start) // BUCKET_S)
            return i if 0 <= i < n else None

        out = {"t": ts}
        for g in self.groups_with("prefill_rank"):
            per_rank = [dict() for _ in range(n)]
            thr = [0] * n
            for e in self.ev["%s_prefill_rank" % g]:
                i = idx(e["t"])
                if i is None or e.get("compute_ms") is None:
                    continue
                key = "%s%s" % (e.get("rk", ""), e.get("rank", 0))
                a = per_rank[i].setdefault(key, [0, 0.0])
                a[0] += e.get("new_tok") or 0
                a[1] += e["compute_ms"]
            for i, d in enumerate(per_rank):
                if d:
                    thr[i] = max(a[0] for a in d.values())
            out["%s_prefill_tps" % g] = [
                (min(_rate(a[0], a[1]) or 0 for a in d.values()) if d else None) for d in per_rank]
            out["%s_prefill_tok_per_s_wall" % g] = [
                (x / BUCKET_S if x else None) for x in thr]
        for g in self.groups_with("decode_batch"):
            s = [[] for _ in range(n)]
            for e in self.ev["%s_decode_batch" % g]:
                i = idx(e["t"])
                if i is not None and e.get("gen_tps") is not None:
                    s[i].append(e["gen_tps"])
            out["%s_decode_tps" % g] = [(sum(v) / len(v)) if v else None for v in s]
        # Fill the gaps logically (user order 2026-09-27): a bucket without a line
        # is NOT a failure.  Within 2x the series' usual line interval after a
        # value and with no flip in between the group is still computing between
        # two log lines -> hold the last value (step); otherwise it slept, was
        # flipped away or had no work -> 0.  Before the boot's first line there is
        # nothing (None); after its end / death the server cuts the series.
        flips = [e["t"] for e in self.ev["flips"]]
        first = self.first_t
        for key in [k for k in out if k.endswith("_tps")]:
            out[key] = fill_series(ts, out[key], flips, first, BUCKET_S)
        out["filled"] = True
        return out

    def timeline(self, now: float, span: float = HISTORY_S) -> dict:
        """The phase bar feed: P-Prefill / D-Prefill / Decode runs and the
        flips between them over the last ``span`` seconds (see phase_timeline)."""
        t0 = now - span
        lo = t0 - 120.0       # whole runs that began before the window
        acts = []

        def mid(t):   # P/D log stamps are whole seconds (truncated): the line fell in [t, t+1)
            return t + 0.5 if t == int(t) else t
        for g, cls in (("P", "P"), ("single", "P"), ("D", "D")):
            for e in self.ev.get("%s_prefill_rank" % g, ()):
                if e["t"] >= lo:
                    t = mid(e["t"])
                    ms = e.get("gpu_ms") or e.get("compute_ms")
                    acts.append({"t": t, "s": t - ms / 1000.0 if ms else t, "cls": cls, "ev": e})
        for g in ("D", "single"):
            # every round's own line (exact stamp + gpu-ms) times the decode; the
            # 'Decode batch' lines carry the rate (gen throughput)
            for e in self.ev.get("%s_decode_rank" % g, ()):
                if e["t"] >= lo and e.get("rank", 0) == 0:
                    t = e.get("t_exact") or mid(e["t"])
                    ms = e.get("gpu_ms")
                    acts.append({"t": t, "s": t - ms / 1000.0 if ms else t, "cls": "dec", "ev": e})
            for e in self.ev.get("%s_decode_batch" % g, ()):
                if e["t"] >= lo:
                    acts.append({"t": mid(e["t"]), "s": mid(e["t"]), "cls": "dec", "ev": e})
        flips = _flip_intervals([e for e in self.ev["flip_begins"] if e["t"] >= lo - 600],
                                [e for e in self.ev["flips"] if e["t"] >= lo],
                                self.flip_open, now)
        hint = (self.last.get("awake") or {}).get("awake")
        return {"t0": t0, "t1": now, "span_s": span, "gap_s": PHASE_GAP_S,
                "segs": phase_timeline(acts, flips, t0, now, self.first_t, hint,
                                       work_from=self.first_work_t if self.first_work_t is not None else now)}

    def groups_with(self, kind: str) -> List[str]:
        return [g for g in ("P", "D", "single") if self.ev.get("%s_%s" % (g, kind))]

    def _mark_gen(self):
        """Re-derive every Decode batch line's shown rate (mark_gen_artefacts):
        pauses are the flips (begin and done) and the group's own prefill lines."""
        flips = [e["t"] for e in self.ev["flip_begins"]] + [e["t"] for e in self.ev["flips"]]
        if self.flip_open:
            flips.append(self.flip_open["t"])
        for g in self.groups_with("decode_batch"):
            pauses = sorted(flips + [e["t"] for e in self.ev["%s_prefill_rank" % g]])
            mark_gen_artefacts(self.ev["%s_decode_batch" % g], pauses)

    def view(self, now: float, with_series: bool = True) -> dict:
        self._mark_gen()
        age = now - self.newest_mtime if self.newest_mtime else None
        v = {
            "stem": self.stem,
            "meta": self.meta,
            "files": {g: {"path": t.path, "size": t.size, "age_s": round(now - t.mtime, 1) if t.mtime else None}
                      for g, t in self.tails.items()},
            "age_s": round(age, 1) if age is not None else None,
            "last_log_t": self.newest_mtime or None,
            "live": age is not None and age < LIVE_S,
            "awake": self.last.get("awake"),
            "queue": self.last.get("queue"),
            "flip_open": self.flip_open,
            "flips": list(self.ev["flips"])[-12:],
            "flip_count": self.counts.get("flip_done", 0),
            "health": self.health,
            "errors": list(self.ev["errors"])[-8:],
            "stops": list(self.ev["stops"])[-12:],
            "stop_count": self.counts.get("stop", 0),
            "last_activity": {g: self.last_activity(g) for g in ("P", "D", "single")},
            "last_activity_any": self.last_activity(),
            "queue_log": (self.last.get("P_prefill_batch") or {}).get("queue"),
            "error_count": self.counts.get("error", 0),
            "prefill": {g: self._prefill_view(g, now) for g in self.groups_with("prefill_rank")},
            "decode": {g: self._decode_view(g, now) for g in self.groups_with("decode_batch")},
        }
        v["totals"] = self.totals_view()
        v["cache"] = self.cache_view(now)
        if with_series:
            v["series"] = self.series(now)
            # a finished boot's bar ends with its last log line, not with the wall clock
            v["timeline"] = self.timeline(now if v["live"] or not self.newest_mtime else min(now, self.newest_mtime))
        return v

    def bucket_activity(self, t0: float, n: int, bucket_s: float = BUCKET_S) -> List[dict]:
        """Per bucket [t0 + i*b, t0 + (i+1)*b): which class computed and how many tokens.

        P/D: a 'Prefill rank batch' line of that group = the class computed; its
        tokens are the first rank's 'Prefill batch' #new-token.  Decode: a
        'Decode batch' or 'Decode rank batch' line of D = decode computed; its
        tokens are the completion_tokens of the requests the front served in
        the bucket (exact per request, attributed to the bucket it ended in).
        """
        out = [{"P": False, "D": False, "dec": False, "P_tok": 0, "D_tok": 0, "dec_tok": 0} for _ in range(n)]

        def idx(t):
            i = int((t - t0) // bucket_s)
            return i if 0 <= i < n else None

        for g, key in (("P", "P"), ("single", "P"), ("D", "D")):
            for e in self.ev.get("%s_prefill_rank" % g, ()):
                i = idx(e["t"])
                if i is not None:
                    out[i][key] = True
            for e in self.ev.get("%s_prefill_batch" % g, ()):
                i = idx(e["t"])
                if i is not None and e.get("rank") in (None, 0):
                    out[i][key + "_tok"] += e.get("new_tok") or 0
        for g in ("D", "single"):
            for k in ("decode_batch", "decode_rank"):
                for e in self.ev.get("%s_%s" % (g, k), ()):
                    i = idx(e["t"])
                    if i is not None:
                        out[i]["dec"] = True
        for t, tok in self.served_ev:
            i = idx(t)
            if i is not None:
                out[i]["dec_tok"] += tok
        return out

    def totals_view(self) -> dict:
        p, d = self.tot.get("P", {}), self.tot.get("D", {})
        s1 = self.tot.get("single", {})
        dec = sum(self.tot.get(k, {}).get("completion", 0) for k in list(self.tot) if k.startswith("served_"))
        def gpu_rate(g):
            rated = [a[0] / (a[1] / 1000.0) for a in self.rank_tot.get(g, {}).values() if a[1] > 0]
            return min(rated) if rated else None     # the slowest rank bounds the group, as in the tiles
        last_t = max([t for t in self._last_t.values() if t] or [0]) or None
        wall = (last_t - self.first_t) if (last_t and self.first_t and last_t > self.first_t) else None
        p_new = p.get("new", 0) + s1.get("new", 0)
        dec_ms = d.get("dec_gpu_ms", 0) + s1.get("dec_gpu_ms", 0)
        return {
            "boot_wall_s": wall,
            "p_rate_gpu": gpu_rate("P") or gpu_rate("single"), "d_rate_gpu": gpu_rate("D"),
            "p_rate_wall": (p_new / wall) if wall else None, "d_rate_wall": (d.get("new", 0) / wall) if wall else None,
            "dec_rate_gpu": (dec / (dec_ms / 1000.0)) if dec_ms else None,
            "dec_rate_wall": (dec / wall) if wall else None,
            "p_new": p_new, "d_new": d.get("new", 0),
            "p_chunks": p.get("chunks", 0), "d_chunks": d.get("chunks", 0),
            "decoded": dec, "served_requests": self.tot.get("served_D", {}).get("n", 0),
            "read_progress": round(self.read_progress(), 4),
        }

    def cache_view(self, now: float) -> dict:
        t0 = now - WINDOW_S
        w = collections.defaultdict(collections.Counter)
        pb_rows = collections.defaultdict(list)   # group -> every "pb" (t, new, cached) the ring holds
        for t, g, kind, a, b in self.win:
            if kind == "pb":
                pb_rows[g].append((t, a, b))      # any age: the last burst may predate the 60-s window
            if t < t0:
                continue
            c = w[g]
            if kind == "pb":
                c["new"] += a; c["cached"] += b; c["chunks"] += 1
            elif kind == "lb":
                c["loadback_n"] += 1; c["loadback_tok"] += a
            elif kind == "mb":
                c["mamba_n"] += 1; c["mamba_tok"] += a
            elif kind == "l3":
                c["l3inc_n"] += 1; c["l3inc_delivered"] += a; c["l3inc_deliverable"] += b
            elif kind == "sv":
                c["n"] += 1; c["prompt"] += a; c["cached"] += b

        def one(c):
            c = dict(c)
            seen = c.get("new", 0) + c.get("cached", 0)
            if "new" not in c and "chunks" not in c:
                seen = 0          # front rows (served_*) carry no #new-token; their share is req_hit_share
            c["hit_share"] = (c.get("cached", 0) / seen) if seen else None
            c["l2_share"] = (c.get("loadback_tok", 0) / seen) if seen else None
            dl = c.get("l3inc_deliverable", 0)
            c["l3inc_share"] = (c.get("l3inc_delivered", 0) / dl) if dl else None
            pr = c.get("prompt", 0)
            c["req_hit_share"] = (c.get("cached", 0) / pr) if pr else None
            return c

        burst = {}
        for g, rows in pb_rows.items():
            t_last = max(t for t, _, _ in rows)
            c = collections.Counter()
            for t, a, b in rows:
                if t_last - 20.0 <= t <= t_last:  # the burst itself, not the idle rows before it
                    c["new"] += a; c["cached"] += b; c["chunks"] += 1
            burst[g] = (one(c), t_last)

        out = {}
        for g in ("P", "D", "single", "served_P", "served_D"):
            if g in self.tot or g in w:
                bs, bt = burst.get(g, (None, None))
                out[g] = {"window": one(w.get(g, {})), "boot": one(self.tot.get(g, {})),
                          "last_burst": bs, "last_burst_t": bt}
        return out


class LiveLogs:
    """Discovers boots and keeps them tailed; thread-safe snapshot."""

    def __init__(self, globs=None, interval: float = 1.0):
        self.globs = globs or DEFAULT_LOG_GLOBS
        self.interval = interval
        self.boots: Dict[str, Boot] = {}
        self.lock = threading.Lock()
        self._last_scan = 0.0
        self.scan_s = 10.0
        self.harness = stops.HarnessLogs()   # planned stop vs death (stops.py)

    def scan(self, now: Optional[float] = None):
        now = now or time.time()
        seen = {}
        for pat in self.globs:
            for p in glob.glob(pat):
                try:
                    mt = os.stat(p).st_mtime
                except OSError:
                    continue
                if now - mt > SHOW_S:
                    continue
                stem, group = split_log_name(p)
                key = os.path.join(os.path.dirname(p), stem)
                seen.setdefault(key, []).append((group, p))
        with self.lock:
            for key, files in seen.items():
                b = self.boots.get(key)
                if b is None:
                    b = self.boots[key] = Boot(os.path.basename(key), os.path.dirname(key))
                for group, p in files:
                    b.add_file(group, p)
            for key in list(self.boots):
                if key not in seen:
                    del self.boots[key]
        self._last_scan = now

    def poll(self):
        now = time.time()
        if now - self._last_scan >= self.scan_s:
            self.scan(now)
        with self.lock:
            boots = list(self.boots.values())
        # newest first, and a LIVE boot gets up to 8 slices per cycle, so after a
        # restart its verdict and numbers are current long before the history rows
        for b in sorted(boots, key=lambda x: -x.newest_mtime):
            b.poll()
            if now - b.newest_mtime < LIVE_S:
                for _ in range(7):
                    if b.read_progress() >= 0.999:
                        break
                    b.poll()
        self.harness.poll([b.dir for b in boots])

    def snapshot(self, with_series: bool = True, max_boots: int = 10) -> List[dict]:
        now = time.time()
        with self.lock:
            boots = sorted(self.boots.values(), key=lambda b: -b.newest_mtime)[:max_boots]
            newest_in_dir = {}
            for b in boots:
                newest_in_dir.setdefault(b.dir, b)
            views = []
        for b in boots:
            primary = (now - b.newest_mtime < LIVE_S) or newest_in_dir.get(b.dir) is b
            with b.lock:
                v = b.view(now, with_series and primary)
            v["primary"] = primary
            last_line = max([t for t in b._last_t.values() if t] or [0]) or None
            v["end"] = stops.classify(b.stem, b.first_t, last_line or b.newest_mtime or None,
                                      self.harness.for_dir(b.dir))
            views.append(v)
        views.sort(key=lambda v: (not v["live"], not v["primary"],
                                  v["age_s"] if v["age_s"] is not None else 1e12))
        return views

    def run_forever(self, stop: threading.Event):
        while not stop.is_set():
            try:
                self.poll()
            except Exception as e:  # keep the collector alive; report in /api
                self.last_error = "%s: %s" % (type(e).__name__, e)
            stop.wait(self.interval)
