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

from . import parse, redact

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
            ("flips", 400), ("errors", 200), ("stops", 60),
            ("single_prefill_rank", 6000), ("single_prefill_batch", 3000),
            ("single_decode_batch", 4000), ("single_decode_rank", 20000),
        )}
        self.last = {}          # kind -> last event
        self.health = {}        # group -> last health event
        self.flip_open = None
        self.counts = collections.Counter()
        self._last_t = {}       # group -> newest log timestamp seen in that file
        self.first_t = None     # first timestamp of this boot's logs
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
        # the last 1 s: log stamps are whole seconds, so "last second" = lines
        # stamped in the previous or the current second; 0 if none fell
        one = self._prefill_window(ranks, batches, float(int(now)) - 1.0)
        return {
            "window_s": WINDOW_S,
            "one_s": one["tps"] or 0.0,
            "now": win,
            "last_burst": burst,
            "last_t": last_t,
            "queue": lb.get("queue") if lb else None,
            "pending_tok": lb.get("pending_tok") if lb else None,
        }

    def _decode_view(self, g: str, now: float) -> dict:
        rows = [e for e in self.ev["%s_decode_batch" % g] if e["t"] >= now - WINDOW_S]
        rr = [e for e in self.ev["%s_decode_rank" % g]
              if e["t"] >= now - WINDOW_S and e.get("rank", 0) == 0 and e.get("gpu_ms")]
        last = self.last.get("%s_decode_batch" % g)
        gen = [e["gen_tps"] for e in rows if e.get("gen_tps") is not None]
        compute = None
        if rr and last and last.get("accept_len"):
            ms = sum(e["gpu_ms"] for e in rr)
            bs = sum((e.get("bs") or 0) for e in rr)
            compute = (bs * last["accept_len"]) / (ms / 1000.0) if ms > 0 else None
        one = [e["gen_tps"] for e in self.ev["%s_decode_batch" % g]
               if e["t"] >= float(int(now)) - 1.0 and e.get("gen_tps") is not None]
        return {
            "window_s": WINDOW_S,
            "one_s": (sum(one) / len(one)) if one else 0.0,
            "gen_tps": (sum(gen) / len(gen)) if gen else None,
            "gen_tps_last": last.get("gen_tps") if last else None,
            "running": last.get("running") if last else None,
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

    def groups_with(self, kind: str) -> List[str]:
        return [g for g in ("P", "D", "single") if self.ev.get("%s_%s" % (g, kind))]

    def view(self, now: float, with_series: bool = True) -> dict:
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
        for t, g, kind, a, b in self.win:
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

        out = {}
        for g in ("P", "D", "single", "served_P", "served_D"):
            if g in self.tot or g in w:
                out[g] = {"window": one(w.get(g, {})), "boot": one(self.tot.get(g, {}))}
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
