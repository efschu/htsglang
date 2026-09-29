"""Persistent history for the Grafana-style panels (DASHBOARD-GRAFIKEN, user order 29.09. ~13:40Z).

rigdash kept 15 min in memory (sources.gpu_hist, energy.py); the panels need days.  One SQLite file
in ``--state-dir`` (``history.sqlite``), no extra service:

  p0   raw samples: NVML every 1 s, host every 5 s, model series in closed 5-s buckets
  p1   10-s means of p0          (compacted once p0's 10-s bucket is closed)
  p2   60-s means of p1
  marks  boot start / boot end / flip, per model, with the flip time as value

Retention p0 3 h, p1 3 d, p2 30 d; the file is capped (``MAX_MB``): past it the oldest p2 day goes.
A query picks the coarsest tier whose step fits the requested resolution and fills the part a
tier does not hold (not yet compacted / already expired) from its neighbour.

Every series is a RATE or a LEVEL, never a counter, so a mean over a bucket stays correct at every
tier (tokens/s averaged over 60 s = tokens in those 60 s / 60).

Sources (IPC and hardware first; what still comes from a log line is labelled "aus Log (Übergang)"
in the answer's ``src``):

  cards        NVML: temperature (GPU core -- the 3080/5090 expose NO hotspot/junction sensor via
               NVML, measured 29.09.: nvmlDeviceGetThermalSettings has one sensor, target GPU),
               power, SM clock, utilisation, memory used; name/uuid; roles from state.json
               groups.<G>.launch (CUDA_VISIBLE_DEVICES + --rank-gpu-id + tp/pp)
  host         Proxmox host /proc/stat, /proc/meminfo, memory.current of the htsglang containers'
               cgroups (the host currency) over the existing ssh alias; without ssh: this LXC's /proc
  model        P-/D-Prefill tok/s (#new-token of the first rank's 'Prefill batch' per 5 s), Decode
               tok/s and per stream (gen throughput / #running-req, spread over its interval),
               KV usage (full token usage) -- log lines (Übergang); input-token classes
               (cacheacct.py): state.json front.served_tokens with D_after_P when the image has it,
               else the WEG2-SERVED lines paired by rid (Übergang); flips: events.jsonl
               flip_first_work (IPC), else the log flip rows
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import subprocess
import threading
import time
from typing import Dict, List, Optional, Tuple

from . import cacheacct, live

LOG_LABEL = "aus Log (Übergang)"
TIERS = (("p0", 1, 3 * 3600), ("p1", 10, 3 * 86400), ("p2", 60, 30 * 86400))
MAX_MB = 256
MODEL_BUCKET_S = 5.0
MODEL_LAG_S = 30.0          # a decode line reports the interval BEFORE it: wait for it
DEC_GAP_MAX_S = 30.0        # two decode lines further apart did not decode continuously
RANGES = {"15m": 900, "1h": 3600, "6h": 6 * 3600, "24h": 86400, "7d": 7 * 86400}
MAX_POINTS = 1440
STEPS = (5, 10, 20, 30, 60, 120, 300, 600, 900)    # multiples of the tier steps: no jitter
MODELS = ("27B", "NF")


def model_of(meta: dict, stem: str) -> str:
    return "27B" if live.is_27b_boot(meta or {}, stem) else "NF"


# ----------------------------------------------------------------------------- storage

class HistoryDB:
    def __init__(self, path: Optional[str], max_mb: int = MAX_MB):
        self.path = path or ":memory:"
        self.max_mb = max_mb
        self.lock = threading.RLock()
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        self.db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self.db.execute("PRAGMA auto_vacuum=INCREMENTAL")   # only takes on a new file
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS series(id INTEGER PRIMARY KEY, name TEXT UNIQUE)")
        for t, _, _ in TIERS:
            self.db.execute("CREATE TABLE IF NOT EXISTS %s(sid INTEGER, ts INTEGER, v REAL, "
                            "PRIMARY KEY(sid, ts)) WITHOUT ROWID" % t)
            self.db.execute("CREATE INDEX IF NOT EXISTS %s_ts ON %s(ts)" % (t, t))
        self.db.execute("CREATE TABLE IF NOT EXISTS marks(ts REAL, model TEXT, kind TEXT, label TEXT, v REAL, "
                        "PRIMARY KEY(ts, model, kind))")
        self.db.execute("CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT)")
        self._sid: Dict[str, int] = {}
        for sid, name in self.db.execute("SELECT id, name FROM series"):
            self._sid[name] = sid

    # --- small helpers ---------------------------------------------------
    def sid(self, name: str) -> int:
        s = self._sid.get(name)
        if s is None:
            self.db.execute("INSERT OR IGNORE INTO series(name) VALUES (?)", (name,))
            s = self.db.execute("SELECT id FROM series WHERE name=?", (name,)).fetchone()[0]
            self._sid[name] = s
        return s

    def get(self, k: str, default=None):
        with self.lock:
            r = self.db.execute("SELECT v FROM meta WHERE k=?", (k,)).fetchone()
        if r is None:
            return default
        try:
            return json.loads(r[0])
        except ValueError:
            return default

    def set(self, k: str, v) -> None:
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO meta(k, v) VALUES (?, ?)", (k, json.dumps(v)))

    def put(self, rows: List[Tuple[str, float, Optional[float]]]) -> None:
        """(name, ts, value) into p0; None values are not stored (a gap)."""
        with self.lock:
            data = [(self.sid(n), int(ts), float(v)) for n, ts, v in rows if v is not None
                    and not (isinstance(v, float) and math.isnan(v))]
            if data:
                self.db.executemany("INSERT OR REPLACE INTO p0(sid, ts, v) VALUES (?, ?, ?)", data)

    def mark(self, ts: float, model: str, kind: str, label: str, v: Optional[float] = None) -> None:
        with self.lock:
            self.db.execute("INSERT OR IGNORE INTO marks(ts, model, kind, label, v) VALUES (?, ?, ?, ?, ?)",
                            (round(float(ts), 3), model, kind, label, v))

    # --- compaction, retention, size cap ----------------------------------
    def compact(self, now: float) -> dict:
        """p0 -> p1 (10 s) and p1 -> p2 (60 s) for closed buckets; then retention and the size cap.
        The cursors live in meta, so a restart continues where it stopped (INSERT OR REPLACE makes
        a repeated bucket idempotent)."""
        done = {}
        lag = MODEL_LAG_S + 2 * MODEL_BUCKET_S + 10
        for (src, _, _), (dst, step, _) in zip(TIERS[:-1], TIERS[1:]):
            cur = self.get("cursor." + dst)
            hi = int(((now - lag - (60 if dst == "p2" else 0)) // step) * step)
            with self.lock:
                if cur is None:
                    r = self.db.execute("SELECT MIN(ts) FROM %s" % src).fetchone()[0]
                    cur = int((r // step) * step) if r is not None else hi
                if hi > cur:
                    self.db.execute(
                        "INSERT OR REPLACE INTO %s(sid, ts, v) SELECT sid, (ts / %d) * %d AS b, AVG(v) FROM %s "
                        "WHERE ts >= ? AND ts < ? GROUP BY sid, b" % (dst, step, step, src), (cur, hi))
                    done[dst] = hi - cur
                    cur = hi
            self.set("cursor." + dst, cur)
        with self.lock:
            for t, _, keep in TIERS:
                self.db.execute("DELETE FROM %s WHERE ts < ?" % t, (int(now - keep),))
            self.db.execute("DELETE FROM marks WHERE ts < ?", (now - TIERS[-1][2],))
            size = self.size_mb()
            while size > self.max_mb:
                oldest = self.db.execute("SELECT MIN(ts) FROM p2").fetchone()[0]
                if oldest is None:
                    break
                self.db.execute("DELETE FROM p2 WHERE ts < ?", (oldest + 86400,))
                self.db.execute("PRAGMA incremental_vacuum")
                new = self.size_mb()
                if new >= size:
                    break
                size = new
        return done

    def rebucket(self, names: List[str], lo: float, hi: float) -> None:
        """Late rows (a boot's backfill, written after the tiers were compacted past them): redo the
        coarse buckets of THESE series over [lo, hi) only -- other series' fine rows there may be
        expired already, and their coarse buckets must stay as they are."""
        with self.lock:
            sids = [self._sid[n] for n in set(names) if n in self._sid]
            if not sids:
                return
            q = ",".join("?" * len(sids))
            for (src, _, _), (dst, step, _) in zip(TIERS[:-1], TIERS[1:]):
                cur = self.get("cursor." + dst)
                if cur is None:
                    continue
                a = int(lo // step) * step
                b = min(int(math.ceil(hi / step)) * step, int(cur))
                if a < b:
                    self.db.execute(
                        "INSERT OR REPLACE INTO %s(sid, ts, v) SELECT sid, (ts / %d) * %d AS bk, AVG(v) FROM %s "
                        "WHERE sid IN (%s) AND ts >= ? AND ts < ? GROUP BY sid, bk" % (dst, step, step, src, q),
                        (*sids, a, b))

    def size_mb(self) -> float:
        pc = self.db.execute("PRAGMA page_count").fetchone()[0]
        fl = self.db.execute("PRAGMA freelist_count").fetchone()[0]
        ps = self.db.execute("PRAGMA page_size").fetchone()[0]
        return (pc - fl) * ps / 1048576.0

    # --- query -------------------------------------------------------------
    def _plan(self, lo: int, hi: int, k: int, now: float) -> List[Tuple[str, int, int]]:
        """Segments [a, b) per table: tier k where it holds data, finer tiers for what tier k has
        not compacted yet, coarser tiers for what tier k has already dropped."""
        name, _, keep = TIERS[k]
        out = []
        cursor = self.get("cursor." + name) if k > 0 else None
        a = max(lo, int(now - keep))
        b = hi if cursor is None else min(hi, int(cursor))
        if a < b:
            out.append((name, a, b))
        if cursor is not None and b < hi and k > 0:
            out += self._plan(max(lo, b), hi, k - 1, now)
        if lo < a and k + 1 < len(TIERS):
            out += self._plan(lo, min(a, hi), k + 1, now)
        return out

    def query(self, names: List[str], lo: float, hi: float, step: int, now: Optional[float] = None) -> Dict[str, Dict[int, float]]:
        now = now or time.time()
        k = 0
        for i, (_, tstep, _) in enumerate(TIERS):
            if tstep <= step:
                k = i
        lo_i, hi_i = int(lo // step) * step, int(hi)
        out: Dict[str, Dict[int, float]] = {n: {} for n in names}
        with self.lock:
            sids = {self._sid[n]: n for n in names if n in self._sid}
            if not sids:
                return out
            q = ",".join("?" * len(sids))
            acc: Dict[Tuple[int, int], List[float]] = {}
            for table, a, b in self._plan(lo_i, hi_i, k, now):
                for sid, bkt, s, c in self.db.execute(
                        "SELECT sid, (ts / %d) * %d AS b, SUM(v), COUNT(*) FROM %s WHERE sid IN (%s) AND ts >= ? "
                        "AND ts < ? GROUP BY sid, b" % (step, step, table, q), (*sids.keys(), a, b)):
                    x = acc.setdefault((sid, bkt), [0.0, 0])
                    x[0] += s
                    x[1] += c
        for (sid, bkt), (s, c) in acc.items():
            out[sids[sid]][bkt] = s / c
        return out

    def marks(self, model: str, lo: float, hi: float) -> List[dict]:
        with self.lock:
            rows = self.db.execute("SELECT ts, kind, label, v FROM marks WHERE model=? AND ts>=? AND ts<=? "
                                   "ORDER BY ts", (model, lo, hi)).fetchall()
        return [{"t": t, "kind": k, "label": lab, "v": v} for t, k, lab, v in rows]

    def stats(self) -> dict:
        with self.lock:
            return {"size_mb": round(self.size_mb(), 2), "max_mb": self.max_mb,
                    "rows": {t: self.db.execute("SELECT COUNT(*) FROM %s" % t).fetchone()[0] for t, _, _ in TIERS},
                    "series": len(self._sid)}


# ----------------------------------------------------------------------------- model buckets (pure)

def prefill_tokens(events, lo: float, n: int, step: float) -> List[float]:
    """#new-token of the first rank's 'Prefill batch' lines per bucket (each chunk once)."""
    out = [0.0] * n
    for e in events:
        if e.get("rank") not in (None, 0):
            continue
        i = int((e["t"] - lo) // step)
        if 0 <= i < n:
            out[i] += e.get("new_tok") or 0
    return out


def decode_buckets(lines, lo: float, n: int, step: float) -> Tuple[List[float], List[float], List[Optional[float]]]:
    """'Decode batch' lines -> (tokens, covered seconds, per-stream tok/s) per bucket.

    A line's gen throughput is the rate over the interval since the previous line of the same group;
    it is spread over that interval (time-weighted), never over a gap longer than DEC_GAP_MAX_S or
    across a line marked as an artefact (flip/extend, live.mark_gen_artefacts).  Per stream = the
    rate / #running-req of the line."""
    tok = [0.0] * n
    cov = [0.0] * n
    sw = [0.0] * n
    hi = lo + n * step
    prev = None
    for e in lines:
        t = e["t"]
        if prev is not None and e.get("gen_tps") is not None and not e.get("gen_art") \
                and 0 < t - prev <= DEC_GAP_MAX_S and t > lo and prev < hi:
            r = float(e["gen_tps"])
            run = e.get("running") or 0
            per = r / run if run else None
            a = max(prev, lo)
            while a < min(t, hi):
                i = int((a - lo) // step)
                b = min(t, lo + (i + 1) * step, hi)
                d = b - a
                tok[i] += r * d
                cov[i] += d
                if per is not None:
                    sw[i] += per * d
                a = b
        prev = t
    stream = [(sw[i] / cov[i]) if cov[i] > 0 and sw[i] > 0 else None for i in range(n)]
    return tok, cov, stream


def level_buckets(events, key: str, lo: float, n: int, step: float, scale: float = 1.0) -> List[Optional[float]]:
    acc = [[0.0, 0] for _ in range(n)]
    for e in events:
        v = e.get(key)
        if v is None:
            continue
        i = int((e["t"] - lo) // step)
        if 0 <= i < n:
            acc[i][0] += float(v) * scale
            acc[i][1] += 1
    return [(s / c) if c else None for s, c in acc]


def cache_buckets(legs, lo: float, n: int, step: float, p_rids: set) -> List[Dict[str, int]]:
    """Token classes per bucket from the served legs (cacheacct.split_legs), in time order; the
    P rids before ``lo`` must already be in ``p_rids``."""
    out = [cacheacct.blank() for _ in range(n)]
    for leg in sorted(legs, key=lambda x: x[0]):
        t, g, _leg, rid = leg[0], leg[1], leg[2], leg[3]
        if t < lo:
            continue
        i = int((t - lo) // step)
        if i >= n:
            break
        if g == "P" and rid is not None:
            p_rids.add(rid)
        for k, v in cacheacct.classify_leg(g, rid, leg[4], leg[5], p_rids).items():
            out[i][k] += v
    return out


def card_roles(launch: dict, cards: List[dict]) -> Dict[str, List[str]]:
    """uuid -> ["P-PP0", "D-TP0", ...] from groups.<G>.launch (IPC): CUDA_VISIBLE_DEVICES lists the
    visible cards, --rank-gpu-id maps rank -> visible index, rank = pp_rank * tp + tp_rank."""
    out: Dict[str, List[str]] = {}
    by_idx = {str(c.get("index")): c.get("uuid") for c in cards}
    for g, l in sorted((launch or {}).items()):
        argv = (l or {}).get("argv") or []
        env = (l or {}).get("env") or {}

        def opt(name):
            for i, a in enumerate(argv):
                if a == name and i + 1 < len(argv):
                    return argv[i + 1]
                if isinstance(a, str) and a.startswith(name + "="):
                    return a.split("=", 1)[1]
            return None
        try:
            tp = int(opt("--tp-size") or 1)
            pp = int(opt("--pp-size") or 1)
        except ValueError:
            continue
        vis = [x.strip() for x in str(env.get("CUDA_VISIBLE_DEVICES") or "").split(",") if x.strip()]
        rg = opt("--rank-gpu-id")
        ids = [int(x) for x in rg.split(",")] if rg else list(range(tp * pp))
        for r, gi in enumerate(ids[:tp * pp]):
            dev = vis[gi] if gi < len(vis) else str(gi)
            uuid = dev if dev.startswith("GPU-") else by_idx.get(dev)
            if not uuid:
                continue
            p, t = divmod(r, tp)
            lab = "%s-%s" % (g, "PP%d" % p if tp == 1 and pp > 1 else ("TP%d" % t if pp == 1 else "PP%dTP%d" % (p, t)))
            out.setdefault(uuid, []).append(lab)
    return out


def short_name(name: str) -> str:
    return (name or "").replace("NVIDIA GeForce ", "").replace("NVIDIA ", "")


# ----------------------------------------------------------------------------- recorder

class Recorder:
    """The sampling threads: NVML 1 s, host 5 s, model buckets 5 s, compaction 60 s."""

    def __init__(self, db: HistoryDB, logs, docker_ssh: Optional[List[str]] = None):
        self.db = db
        self.logs = logs
        self.ssh = docker_ssh or []
        self.cards: List[dict] = db.get("cards", []) or []
        self.errors: Dict[str, str] = {}
        self.now_vals: Dict[str, dict] = {}
        self._nvml = None
        self._cpu_prev = None
        self._cache_prev: Dict[str, dict] = {}
        self._tier_prev: Dict[str, dict] = {}
        self._p_rids: Dict[str, set] = {}

    # --- NVML -------------------------------------------------------------
    def sample_gpus(self, now: float) -> None:
        import pynvml as n
        if self._nvml is None:
            n.nvmlInit()
            self._nvml = n
        rows, cards = [], []
        ts = int(now)
        for i in range(n.nvmlDeviceGetCount()):
            h = n.nvmlDeviceGetHandleByIndex(i)
            name = n.nvmlDeviceGetName(h)
            name = name.decode() if isinstance(name, bytes) else name
            uuid = n.nvmlDeviceGetUUID(h)
            uuid = uuid.decode() if isinstance(uuid, bytes) else uuid
            cards.append({"index": i, "name": name, "short": short_name(name), "uuid": uuid})
            p = "g%d." % i

            def rd(fn, *a):
                try:
                    return fn(*a)
                except Exception:
                    return None
            rows.append((p + "temp", ts, rd(n.nvmlDeviceGetTemperature, h, n.NVML_TEMPERATURE_GPU)))
            pw = rd(n.nvmlDeviceGetPowerUsage, h)
            rows.append((p + "power", ts, pw / 1000.0 if pw is not None else None))
            rows.append((p + "clock", ts, rd(n.nvmlDeviceGetClockInfo, h, n.NVML_CLOCK_SM)))
            u = rd(n.nvmlDeviceGetUtilizationRates, h)
            rows.append((p + "util", ts, u.gpu if u is not None else None))
            m = rd(n.nvmlDeviceGetMemoryInfo, h)
            rows.append((p + "mem", ts, m.used / 1048576.0 if m is not None else None))
        self.db.put(rows)
        if cards != self.cards:
            self.cards = cards
            self.db.set("cards", cards)

    # --- host -------------------------------------------------------------
    HOST_CMD = ("head -1 /proc/stat; grep -E '^(MemTotal|MemAvailable):' /proc/meminfo; "
                "for id in $(docker ps -q --no-trunc --filter name=htsglang); do "
                "echo cg $(cat /sys/fs/cgroup/system.slice/docker-$id.scope/memory.current 2>/dev/null); done")

    def sample_host(self, now: float) -> None:
        if self.ssh:
            p = subprocess.run(self.ssh + [self.HOST_CMD], capture_output=True, text=True, timeout=12)
            text, where = p.stdout, "proxmox"
        else:
            with open("/proc/stat") as fh:
                text = fh.readline()
            with open("/proc/meminfo") as fh:
                text += fh.read()
            where = "lxc"
        vals = parse_host(text)
        rows = []
        if vals.get("cpu") is not None:
            if self._cpu_prev is not None:
                tot = vals["cpu"][0] - self._cpu_prev[0]
                idle = vals["cpu"][1] - self._cpu_prev[1]
                if tot > 0:
                    rows.append(("host.cpu", int(now), 100.0 * (tot - idle) / tot))
            self._cpu_prev = vals["cpu"]
        mt, ma = vals.get("MemTotal"), vals.get("MemAvailable")
        if mt:
            if ma is not None:
                rows.append(("host.mem_pct", int(now), 100.0 * (mt - ma) / mt))
            if vals.get("cg") is not None:
                rows.append(("host.bootmem_pct", int(now), 100.0 * vals["cg"] / (mt * 1024.0)))
                rows.append(("host.bootmem_gib", int(now), vals["cg"] / 1073741824.0))
        self.db.put(rows)
        self.host_where = where

    # --- model buckets ----------------------------------------------------
    def ingest_models(self, now: float) -> None:
        with self.logs.lock:
            boots = list(self.logs.boots.values())
        for b in sorted(boots, key=lambda x: x.first_t or 0):
            try:
                self._ingest_boot(b, now)
            except Exception as e:  # one boot's defect never stops the others
                self.errors["boot:" + b.stem] = "%s: %s" % (type(e).__name__, e)

    def _ingest_boot(self, b, now: float) -> None:
        if b.first_t is None or b.read_progress() < 0.999 or getattr(self.logs.ipc, "last_poll", None) is None:
            return          # wait for the whole log AND the first IPC poll: a boot is backfilled once
        model = model_of(b.meta, b.stem)
        step = MODEL_BUCKET_S
        key = "done." + b.stem
        done = self.db.get(key)
        is_live = now - b.newest_mtime < live.LIVE_S
        ipc = self.logs.ipc.for_tag((b.meta or {}).get("tag"), now)
        if done is None:
            done = (b.first_t // step) * step
            self.db.mark(b.first_t, model, "boot", (b.meta or {}).get("tag") or b.stem)
        end_t = now - MODEL_LAG_S if is_live else b.newest_mtime + step
        hi = (end_t // step) * step
        if ipc and ipc.get("launch"):
            roles = card_roles(ipc["launch"], self.cards)
            if roles and roles != self.db.get("roles." + model):
                self.db.set("roles." + model, roles)
        if hi > done:
            lo = max(done, hi - 2 * 86400)
            n = int((hi - lo) // step)
            with b.lock:
                b._mark_gen()       # flip/extend artefacts marked BEFORE the spread (view() does it too, but later)
                pb = {g: list(b.ev.get("%s_prefill_batch" % g, ())) for g in ("P", "single", "D")}
                dec = [list(b.ev.get("%s_decode_batch" % g, ())) for g in ("D", "single")]
                legs = list(b.served_legs)
                flips = b.flip_rows()
            p_tok = [a + c for a, c in zip(prefill_tokens(pb["P"], lo, n, step), prefill_tokens(pb["single"], lo, n, step))]
            d_tok = prefill_tokens(pb["D"], lo, n, step)
            dtok, dcov, stream = [0.0] * n, [0.0] * n, [None] * n
            sw = [0.0] * n
            for lines in dec:
                t1, c1, s1 = decode_buckets(lines, lo, n, step)
                for i in range(n):
                    dtok[i] += t1[i]
                    dcov[i] += c1[i]
                    if s1[i] is not None:
                        sw[i] += s1[i] * c1[i]
            stream = [(sw[i] / dcov[i]) if dcov[i] > 0 and sw[i] > 0 else None for i in range(n)]
            kv = level_buckets(pb["D"] + dec[0], "full_use", lo, n, step, 100.0)
            rows = []
            pre = "m.%s." % model
            for i in range(n):
                ts = int(lo + i * step)
                rows += [(pre + "p_tps", ts, p_tok[i] / step), (pre + "d_tps", ts, d_tok[i] / step),
                         (pre + "dec_tps", ts, dtok[i] / step), (pre + "stream_tps", ts, stream[i]),
                         (pre + "kv_pct", ts, kv[i])]
            st = ((ipc or {}).get("front") or {}).get("served_tokens")
            if is_live and cacheacct.has_handoff_row(st):
                # IPC counters are cumulative: their deltas land at "now"; a finished boot's history
                # is backfilled from its legs instead (same rule, cacheacct.split_legs)
                cur = cacheacct.from_served_tokens(st)
                d = cacheacct.delta(self._cache_prev.get(b.stem), cur)
                self._cache_prev[b.stem] = cur
                ts = int((now // step) * step)
                rows += [(pre + "tok_" + k, ts, v / step) for k, v in d.items() if v]
                tiers = cacheacct.tiers_from_served_tokens(st)
                if tiers is not None:
                    dt = {k: tiers[k] - (self._tier_prev.get(b.stem) or tiers).get(k, 0) for k in tiers}
                    self._tier_prev[b.stem] = tiers
                    rows += [(pre + "tier_" + k, ts, v / step) for k, v in dt.items() if v > 0]
                    self.db.set("src.%s.tiers" % model, "ipc")
                self.db.set("src.%s.cache" % model, "ipc")
            else:
                pr = self._p_rids.get(b.stem)
                if pr is None:
                    pr = self._p_rids[b.stem] = {x[3] for x in legs if x[1] == "P" and x[0] < lo and x[3]}
                cb = cache_buckets(legs, lo, n, step, pr)
                for i in range(n):
                    ts = int(lo + i * step)
                    rows += [(pre + "tok_" + k, ts, cb[i][k] / step) for k in cacheacct.CLASSES]
                self.db.set("src.%s.cache" % model, "log")
            self.db.put(rows)
            c1 = self.db.get("cursor.p1")
            if c1 is not None and lo < c1:
                self.db.rebucket([r[0] for r in rows], lo, hi)
            # flips: IPC flip_first_work (the front's one clock), else the log rows
            ffw = [e for e in ((ipc or {}).get("ipc_events") or []) if e.get("type") == "flip_first_work"]
            if ffw:
                for e in ffw:
                    dd = e.get("data") or {}
                    t0 = dd.get("flip_begin_ts") or e.get("ts")
                    if t0 is not None and dd.get("dir") in ("P>D", "D>P"):
                        self.db.mark(t0, model, "flip", dd["dir"], dd.get("flip_time_ms"))
                self.db.set("src.%s.flip" % model, "ipc")
            else:
                for r in flips:
                    if lo - 60 <= r["t"] < hi and r.get("dir") in ("P>D", "D>P"):
                        self.db.mark(r["t"], model, "flip", r["dir"], r.get("ms"))
                if flips:
                    self.db.set("src.%s.flip" % model, "log")
            self.db.set(key, hi)
        if not is_live and not self.db.get("ended." + b.stem):
            self.db.mark(b.newest_mtime, model, "end", "Boot-Ende (letzte Logzeile)")
            self.db.set("ended." + b.stem, True)

    # --- now values for the tiles (live boots only) -----------------------
    def sample_now(self, now: float) -> None:
        with self.logs.lock:
            boots = list(self.logs.boots.values())
        vals: Dict[str, dict] = {}
        for b in sorted(boots, key=lambda x: -x.newest_mtime):
            if now - b.newest_mtime >= live.LIVE_S:
                continue
            model = model_of(b.meta, b.stem)
            if model in vals:
                continue
            with b.lock:
                pg = [g for g in b.groups_with("prefill_rank") if g in ("P", "single")]
                v = {"stem": b.stem, "t": now,
                     "p_tps": b._one_s(pg[0], "prefill", now) if pg else None,
                     "d_tps": b._one_s("D", "prefill", now) if "D" in b.groups_with("prefill_rank") else None,
                     "dec_tps": None, "stream_tps": None, "kv_pct": None}
                dg = [g for g in b.groups_with("decode_batch") if g in ("D", "single")]
                if dg:
                    v["dec_tps"] = b._one_s(dg[0], "decode", now)
                    last = b.last.get("%s_decode_batch" % dg[0]) or {}
                    if last.get("gen_tps") is not None and last.get("running") and not last.get("gen_art") \
                            and now - last["t"] <= 10:
                        v["stream_tps"] = last["gen_tps"] / last["running"]
                    if last.get("full_use") is not None:
                        v["kv_pct"] = 100.0 * last["full_use"]
            vals[model] = v
        self.now_vals = vals

    # --- loop ----------------------------------------------------------------
    def run_forever(self, stop: threading.Event) -> None:
        def loop(name, period, fn):
            while not stop.is_set():
                t0 = time.time()
                try:
                    fn(t0)
                    self.errors.pop(name, None)
                except Exception as e:
                    self.errors[name] = "%s: %s" % (type(e).__name__, str(e)[:300])
                stop.wait(max(0.05, period - (time.time() - t0)))

        for name, period, fn in (("nvml", 1.0, self.sample_gpus), ("host", 5.0, self.sample_host),
                                 ("model", MODEL_BUCKET_S, self.ingest_models), ("now", 1.0, self.sample_now),
                                 ("compact", 60.0, self.db.compact)):
            threading.Thread(target=loop, args=(name, period, fn), name="rigdash-hist-" + name, daemon=True).start()
        stop.wait()


def parse_host(text: str) -> dict:
    out: dict = {}
    cg = None
    for line in (text or "").splitlines():
        p = line.split()
        if not p:
            continue
        if p[0] == "cpu" and len(p) >= 6:
            f = [int(x) for x in p[1:9] if x.isdigit()]
            out["cpu"] = (sum(f), f[3] + (f[4] if len(f) > 4 else 0))
        elif p[0] in ("MemTotal:", "MemAvailable:") and len(p) >= 2:
            out[p[0][:-1]] = int(p[1])
        elif p[0] == "cg" and len(p) >= 2 and p[1].isdigit():
            cg = (cg or 0) + int(p[1])
    if cg is not None:
        out["cg"] = cg
    return out


# ----------------------------------------------------------------------------- the answer

def _pct(xs: List[float], p: float) -> Optional[float]:
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    return xs[max(0, math.ceil(p * len(xs)) - 1)]


def view(db: HistoryDB, rec: Optional[Recorder], model: str, range_key: str, now: Optional[float] = None) -> dict:
    now = now or time.time()
    rng = RANGES.get(range_key, 3600)
    step = next((s for s in STEPS if s >= rng / MAX_POINTS), STEPS[-1])
    lo = now - rng
    cards = (rec.cards if rec else None) or db.get("cards", []) or []
    names = []
    for c in cards:
        names += ["g%d.%s" % (c["index"], k) for k in ("temp", "power", "clock", "util", "mem")]
    names += ["host.cpu", "host.mem_pct", "host.bootmem_pct", "host.bootmem_gib"]
    mp = "m.%s." % model
    msr = ["p_tps", "d_tps", "dec_tps", "stream_tps", "kv_pct"] + ["tok_" + k for k in cacheacct.CLASSES]
    names += [mp + k for k in msr] + [mp + "tier_" + k for k in cacheacct.TIERS]
    data = db.query(names, lo, now, step, now)
    t0 = int(lo // step) * step
    ts = list(range(t0, int(now) + 1, step))
    series = {}
    for n in names:
        d = data.get(n) or {}
        series[n.replace(mp, "m.")] = [d.get(t) for t in ts]
    # cards: newest first for "now"
    def latest(n):
        arr = series.get(n) or []
        for v in reversed(arr[-6:]):
            if v is not None:
                return v
        return None
    temps = [latest("g%d.temp" % c["index"]) for c in cards]
    powers = [latest("g%d.power" % c["index"]) for c in cards]
    tok = {k: sum((v or 0) for v in series["m.tok_" + k]) * step for k in cacheacct.CLASSES}
    src_tiers = db.get("src.%s.tiers" % model)
    tiers = ({k: sum((v or 0) for v in series["m.tier_" + k]) * step for k in cacheacct.TIERS}
             if src_tiers == "ipc" else None)
    marks = db.marks(model, lo, now)
    flips_pd = [m["v"] for m in marks if m["kind"] == "flip" and m["label"] == "P>D" and m["v"] is not None]
    last_pd = next((m for m in reversed(marks) if m["kind"] == "flip" and m["label"] == "P>D" and m["v"] is not None), None)
    nv = (rec.now_vals.get(model) if rec else None) or {}
    roles = {m: (db.get("roles." + m) or {}) for m in MODELS}
    card_info = [dict(c, roles={m: roles[m].get(c["uuid"], []) for m in MODELS}) for c in cards]
    src_cache = db.get("src.%s.cache" % model)
    src_flip = db.get("src.%s.flip" % model)
    tiles = {
        "out_p50": _pct(series["m.stream_tps"], 0.5), "out_p90": _pct(series["m.stream_tps"], 0.9),
        "cache_hit": cacheacct.hit_share(tok), "tok": tok, "tiers": tiers,
        "handoff_share": (tok["handoff"] / (tok["handoff"] + tok["cache"] + tok["comp_p"] + tok["comp_d"])
                          if sum(tok.values()) else None),
        "hottest_c": max([t for t in temps if t is not None], default=None),
        "hottest_card": (cards[temps.index(max([t for t in temps if t is not None]))]["short"]
                         if any(t is not None for t in temps) else None),
        "power_mean_w": (sum(p for p in powers if p is not None) / len([p for p in powers if p is not None])
                         if any(p is not None for p in powers) else None),
        "power_sum_w": sum(p for p in powers if p is not None) if any(p is not None for p in powers) else None,
        "cpu_pct": latest("host.cpu"),
        "p_tps": nv.get("p_tps"), "d_tps": nv.get("d_tps"), "dec_tps": nv.get("dec_tps"),
        "live_stem": nv.get("stem"),
        "flip_last_ms": last_pd["v"] if last_pd else None, "flip_last_t": last_pd["t"] if last_pd else None,
        "flip_median_ms": _pct(flips_pd, 0.5), "flip_n": len(flips_pd),
    }
    thin = [m for m in marks if m["kind"] != "flip"]
    fl = [m for m in marks if m["kind"] == "flip"]
    if len(fl) > 800:
        k = len(fl) / 800.0
        fl = [fl[int(i * k)] for i in range(800)]
    return {
        "model": model, "range": range_key, "range_s": rng, "step": step, "t": ts, "now": now,
        "series": series, "cards": card_info, "marks": sorted(thin + fl, key=lambda m: m["t"]),
        "marks_total": len(marks), "tiles": tiles,
        "src": {
            "cards": "NVML", "temp": "NVML GPU-Kern (Hotspot/Junction per NVML nicht lesbar)",
            "host": ("Proxmox-Host /proc + memory.current der htsglang-Container"
                     if getattr(rec, "host_where", None) == "proxmox" else "LXC /proc"),
            "roles": "state.json groups.launch",
            "prefill": LOG_LABEL, "decode": LOG_LABEL, "kv": LOG_LABEL,
            "cache": "state.json front.served_tokens (D_after_P)" if src_cache == "ipc" else LOG_LABEL,
            "cache_tiers": ("state.json front.served_tokens.*.cached_tier" if src_tiers == "ipc"
                            else "– (Feld served_tokens.*.cached_tier ab Image z30y2, 9266bdfb8d)"),
            "flip": "events.jsonl flip_first_work" if src_flip == "ipc" else (LOG_LABEL if src_flip else None),
            "now_tiles": LOG_LABEL,
        },
        "errors": dict(rec.errors) if rec else {},
        "db": db.stats(),
        "ranges": list(RANGES),
    }
