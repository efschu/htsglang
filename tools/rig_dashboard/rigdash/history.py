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

Sources -- IPC and hardware only; no series and no mark is read from a boot log (Nutzer 30.09.:
"keine IPC über Logs", ohne Ausnahme):

  cards        NVML: temperature (GPU core -- the 3080/5090 expose NO hotspot/junction sensor via
               NVML, measured 29.09.: nvmlDeviceGetThermalSettings has one sensor, target GPU),
               power, SM clock, utilisation, memory used; name/uuid; roles from state.json
               groups.<G>.launch (CUDA_VISIBLE_DEVICES + --rank-gpu-id + tp/pp)
  host         Proxmox host /proc/stat, /proc/meminfo, memory.current of the htsglang containers'
               cgroups (the host currency) over the existing ssh alias; without ssh: this LXC's /proc
  model        IPC (Nutzer 30.09.: "keine IPC über Logs"): every 5 s the first rank of each group's
               rankstats (<state>/rankstate/<G>/*.rankstats, pdflip.rankstats/1, timer-written) --
               P-/D-Prefill tok/s = Δprefill.new_tokens / Δts, Decode tok/s = Δdecode.tokens / Δts,
               Decode je Stream = that / decode.running (only while D decoded continuously),
               KV = sched.full_token_usage (level), input-token classes per CHUNK from
               prefill.new_tokens / prefill.cached_tokens (P cached = cache, D cached = Übergabe P->D;
               the rank cannot tell a D-direct prefix hit from the hand-over, so it is never counted as
               cache), cache tiers from state.json front.served_tokens.*.cached_tier;
               flips: events.jsonl flip_first_work.
               A stretch the recorder did not watch live (rigdash was down, or before this image had
               rankstats) stays empty: "no data (before IPC recording)", never a log backfill.
               Boot start/end marks: state.json (boot id stamp, lifecycle since_ts).
"""

from __future__ import annotations

import datetime
import json
import math
import os
import sqlite3
import subprocess
import threading
import time
from typing import Dict, List, Optional, Tuple

from . import activity, cacheacct, flipzeit, ipcstate
from . import names as N

NO_DATA_LABEL = "no data (before IPC recording)"
TIERS = (("p0", 1, 3 * 3600), ("p1", 10, 3 * 86400), ("p2", 60, 30 * 86400))
MAX_MB = 256
MODEL_BUCKET_S = 5.0        # the recorder's loop period
MODEL_STEP_S = 1.0          # model rows per second: a phase change (flip ~2 s) never shares a row
MODEL_LAG_S = 30.0          # a decode line reports the interval BEFORE it: wait for it
DEC_GAP_MAX_S = 30.0        # two decode lines further apart did not decode continuously
RANGES = {"15m": 900, "1h": 3600, "6h": 6 * 3600, "24h": 86400, "7d": 7 * 86400}
MAX_POINTS = 1440
STEPS = (1, 5, 10, 20, 30, 60, 120, 300, 600, 900)    # multiples of the tier steps: no jitter
MODELS = ("27B", "NF")
IPC_LABEL = "rankstats (IPC)"
IPC_STALE_S = 20.0          # a rank file older than this: the rank is gone, its group gives no value
#: model series prefix.  "mi." since 30.09. ~17Z: the rows written before under "m." came from the
#: per-sample counter deltas (a whole 16k chunk in the bucket its counter moved in -- "16k/s"); they
#: are not shown any more, that stretch reads "no data (before IPC recording)"
SERIES = "mi.%s."
HIST_LAG_S = 20.0           # buckets are written once the last pipeline stage has reported their chunks


# ----------------------------------------------------------------------------- storage

def agg_of(name: str) -> str:
    """How a series folds into a coarser bucket: rates and levels by their mean (tokens/s over 60 s =
    tokens in those 60 s / 60); ``*_min`` / ``*_max`` (the batch-size span) by MIN / MAX."""
    return "MIN" if name.endswith("_min") else "MAX" if name.endswith("_max") else "AVG"


class HistoryDB:
    def __init__(self, path: Optional[str], max_mb: int = MAX_MB):
        self.path = path or ":memory:"
        self.max_mb = max_mb
        self.lock = threading.RLock()
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        # two processes share the file since 30.09. ~21:40Z: the sampler writes, the web server reads
        self.db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None, timeout=30)
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

    def reload_sids(self) -> None:
        with self.lock:
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
    def _agg_sql(self) -> str:
        """``CASE`` over the series ids: MIN/MAX for the span series, AVG for the rest (one statement)."""
        mins = [str(i) for n, i in self._sid.items() if agg_of(n) == "MIN"]
        maxs = [str(i) for n, i in self._sid.items() if agg_of(n) == "MAX"]
        parts = []
        if mins:
            parts.append("WHEN sid IN (%s) THEN MIN(v)" % ",".join(mins))
        if maxs:
            parts.append("WHEN sid IN (%s) THEN MAX(v)" % ",".join(maxs))
        return ("CASE %s ELSE AVG(v) END" % " ".join(parts)) if parts else "AVG(v)"

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
                        "INSERT OR REPLACE INTO %s(sid, ts, v) SELECT sid, (ts / %d) * %d AS b, %s FROM %s "
                        "WHERE ts >= ? AND ts < ? GROUP BY sid, b" % (dst, step, step, self._agg_sql(), src), (cur, hi))
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
                        "INSERT OR REPLACE INTO %s(sid, ts, v) SELECT sid, (ts / %d) * %d AS bk, %s FROM %s "
                        "WHERE sid IN (%s) AND ts >= ? AND ts < ? GROUP BY sid, bk" % (dst, step, step, self._agg_sql(), src, q),
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
            if any(n not in self._sid for n in names):
                self.reload_sids()          # a series the sampler process created after we opened the file
            sids = {self._sid[n]: n for n in names if n in self._sid}
            if not sids:
                return out
            q = ",".join("?" * len(sids))
            acc: Dict[Tuple[int, int], List[float]] = {}
            for table, a, b in self._plan(lo_i, hi_i, k, now):
                for sid, bkt, s, c, mn, mx in self.db.execute(
                        "SELECT sid, (ts / %d) * %d AS b, SUM(v), COUNT(*), MIN(v), MAX(v) FROM %s WHERE sid IN (%s) "
                        "AND ts >= ? AND ts < ? GROUP BY sid, b" % (step, step, table, q), (*sids.keys(), a, b)):
                    x = acc.setdefault((sid, bkt), [0.0, 0, mn, mx])
                    x[0] += s
                    x[1] += c
                    x[2], x[3] = min(x[2], mn), max(x[3], mx)
        for (sid, bkt), (s, c, mn, mx) in acc.items():
            ag = agg_of(sids[sid])
            out[sids[sid]][bkt] = mn if ag == "MIN" else mx if ag == "MAX" else s / c
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

def boot_ts(boot_id: str) -> Optional[float]:
    """The UTC stamp in a state dir name ``…-boot-20260930T153426Z-051f`` (an id, not a log line)."""
    for part in (boot_id or "").split("-"):
        if len(part) == 16 and part[8] == "T" and part.endswith("Z"):
            try:
                return datetime.datetime.strptime(part, "%Y%m%dT%H%M%SZ").replace(tzinfo=datetime.timezone.utc).timestamp()
            except ValueError:
                return None
    return None


def model_of_ipc(ipc: dict) -> str:
    """27B or NF from the state dir's line (``/spinning/docker-acceptance/<line>/state/...``)."""
    d = (ipc or {}).get("dir") or ""
    tag = ((ipc or {}).get("tag") or "") + " " + ((ipc or {}).get("line") or "")
    return "27B" if "/27b/" in d or "27b" in tag.lower() else "NF"


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


# ----------------------------------------------------------------------------- counters (pure)

def spread_counter(acc: Dict[int, float], t0: float, t1: float, delta: float) -> None:
    """Book a counter's Δ between two readings at t0 < t1 onto the whole seconds they cover, in
    proportion to the overlap (Nutzer 30.09. ~21:40Z: a late reading must lose nothing -- it only
    makes the interval longer).  ``acc`` = {second: amount}."""
    if t1 <= t0 or delta is None:
        return
    rate = delta / (t1 - t0)
    x = t0
    while x < t1:
        sec = int(math.floor(x))
        z = min(t1, sec + 1.0)
        acc[sec] = acc.get(sec, 0.0) + rate * (z - x)
        x = z


def pop_complete(acc: Dict[int, float], upto: float, since: Optional[float] = None) -> Dict[int, float]:
    """The seconds of ``acc`` that are complete (sec + 1 <= upto) -- removed from ``acc``.  A second
    that began before ``since`` (the first reading) is only partly covered and is dropped."""
    out = {}
    for sec in sorted(k for k in acc if k + 1 <= upto):
        v = acc.pop(sec)
        if since is not None and sec < since:
            continue
        out[sec] = v
    return out


# ----------------------------------------------------------------------------- recorder

class Recorder:
    """The sampling threads: NVML 1 s, host 5 s, model buckets 5 s, compaction 60 s."""

    def __init__(self, db: HistoryDB, boots=None, docker_ssh: Optional[List[str]] = None):
        self.db = db
        # the 1-s rank samples of ipcboot.IpcBoots (the same ring the boot cards read)
        self.boots = boots
        self.ssh = docker_ssh or []
        self.cards: List[dict] = db.get("cards", []) or []
        self.errors: Dict[str, str] = {}
        self._nvml = None
        self._cpu_prev = None
        self._tier_prev: Dict[str, dict] = {}
        self._flips_done: Dict[str, int] = {}
        # counter state: NVML energy (mJ) per card, host CPU jiffies; levels: the last written second
        self._e_prev: Dict[int, Tuple[float, float]] = {}
        self._e_acc: Dict[int, Dict[int, float]] = {}
        self._e_first: Dict[int, float] = {}
        self._lvl_last: Optional[Tuple[int, List[tuple]]] = None
        self._cpu_t: Optional[float] = None
        #: the emergency fill: seconds a LEVEL (temp, clock, util, mem, power without energy counter, KV)
        #: was held from the previous reading instead of read -- target 0 (the sampler has its own process)
        self.held = {"total": 0, "nvml": 0, "kv": 0, "recent": []}
        self.energy_counter: Dict[int, bool] = {}

    # --- NVML -------------------------------------------------------------
    def sample_gpus(self, now: float) -> None:
        import pynvml as n
        if self._nvml is None:
            n.nvmlInit()
            self._nvml = n
        rows, cards, levels = [], [], []
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
            t_read = time.time()
            e = rd(n.nvmlDeviceGetTotalEnergyConsumption, h)          # mJ since driver load: a counter
            lv = [(p + "temp", rd(n.nvmlDeviceGetTemperature, h, n.NVML_TEMPERATURE_GPU)),
                  (p + "clock", rd(n.nvmlDeviceGetClockInfo, h, n.NVML_CLOCK_SM)),
                  (p + "memclock", rd(n.nvmlDeviceGetClockInfo, h, getattr(n, "NVML_CLOCK_MEM", 2)))]
            u = rd(n.nvmlDeviceGetUtilizationRates, h)
            lv.append((p + "util", u.gpu if u is not None else None))
            m = rd(n.nvmlDeviceGetMemoryInfo, h)
            lv.append((p + "mem", m.used / 1048576.0 if m is not None else None))
            if e is not None:
                self.energy_counter[i] = True
                rows += self.power_from_energy(i, t_read, float(e))
            else:
                self.energy_counter[i] = False
                pw = rd(n.nvmlDeviceGetPowerUsage, h)
                lv.append((p + "power", pw / 1000.0 if pw is not None else None))
            levels += lv
        rows += self.level_rows(ts, levels)
        self.db.put(rows)
        if cards != self.cards:
            self.cards = cards
            self.db.set("cards", cards)

    def power_from_energy(self, i: int, t: float, e_mj: float) -> List[tuple]:
        """Leistung lückenlos als Energie-Differenz (nvmlDeviceGetTotalEnergyConsumption): the mJ between
        two readings are spread over the seconds they cover, each complete second gets its mean watts --
        a reading that comes late (a slipped loop) makes one interval longer and loses nothing."""
        prev = self._e_prev.get(i)
        self._e_prev[i] = (t, e_mj)
        if prev is None:
            self._e_first[i] = t
            return []
        if e_mj < prev[1]:                      # the counter restarted (driver reload)
            self._e_acc[i] = {}
            self._e_first[i] = t
            return []
        acc = self._e_acc.setdefault(i, {})
        spread_counter(acc, prev[0], t, e_mj - prev[1])
        done = pop_complete(acc, t, since=math.ceil(self._e_first[i]))
        return [("g%d.power" % i, sec, mj / 1000.0) for sec, mj in done.items()]

    def level_rows(self, ts: int, levels: List[tuple]) -> List[tuple]:
        """Levels (no counter): the reading at its second.  A second the loop skipped is filled from the
        previous reading and COUNTED as held (the emergency fill; the sampler's own process keeps it 0)."""
        rows = [(nme, ts, v) for nme, v in levels]
        last = self._lvl_last
        if last is not None and ts - last[0] > 1:
            gap = [sec for sec in range(last[0] + 1, ts) if ts - sec <= 5]
            for sec in gap:
                rows += [(nme, sec, v) for nme, v in last[1]]
            self._held("nvml", len(gap))
        if last is None or ts > last[0]:
            self._lvl_last = (ts, levels)
        return rows

    def _held(self, kind: str, n: int) -> None:
        if n <= 0:
            return
        self.held[kind] = self.held.get(kind, 0) + n
        self.held["total"] += n
        now = time.time()
        self.held["recent"] = [t for t in self.held["recent"] if t > now - 900][-500:] + [now] * min(n, 50)

    def held_view(self, now: Optional[float] = None) -> dict:
        now = now or time.time()
        return {"total": self.held["total"], "nvml": self.held.get("nvml", 0), "kv": self.held.get("kv", 0),
                "last_5min": sum(1 for t in self.held["recent"] if t > now - 300),
                "energy_counter": {str(k): v for k, v in sorted(self.energy_counter.items())}}

    # --- host -------------------------------------------------------------
    HOST_CMD = ("head -1 /proc/stat; grep -E '^(MemTotal|MemAvailable):' /proc/meminfo; "
                "for id in $(docker ps -q --no-trunc %s); do " % N.docker_name_filters() +      # F0-B: containers of either product name
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
                    # /proc/stat is a counter: its share holds for every second since the last reading
                    pct = 100.0 * (tot - idle) / tot
                    a = int(self._cpu_t) + 1 if self._cpu_t is not None else int(now)
                    rows += [("host.cpu", sec, pct) for sec in range(max(a, int(now) - 30), int(now) + 1)]
            self._cpu_prev = vals["cpu"]
            self._cpu_t = now
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
    # --- model series from IPC (rankstats), every 5 s --------------------------
    def ingest_ipc(self, now: float) -> None:
        """Model series from the 1-s rank samples of ipcboot.IpcBoots (``self.boots``): the work of
        every boot at the time it was DONE (activity.Model -- a prefill chunk over the seconds it
        computed, not at the sample that saw its counter jump), written for the closed 5-s buckets up
        to now - HIST_LAG_S (a chunk's end on the last pipeline stage arrives a few seconds late).
        ``m.<model>.ipc`` = 1 marks every bucket with a sample of a live boot."""
        boots = self.boots
        if boots is None:
            return
        step = MODEL_STEP_S
        with boots.ipc.lock:
            items = [(d, st) for d, st in boots.ipc._st.items() if st.get("kind") == "boot"]
        for d, st in items:
            ipc = ipcstate.boot_view(d, st, boots.ipc._ev.get(d), now)
            key = ipc.get("boot_id") or d
            try:
                model = model_of_ipc(ipc)
                self._boot_marks(key, model, ipc)
                self._mark_flips(key, model, ipc)
                if ipc.get("launch") and not ipc.get("terminal"):
                    roles = card_roles(ipc["launch"], self.cards)
                    if roles and roles != self.db.get("roles." + model):
                        self.db.set("roles." + model, roles)
                m = boots.model(key)
                if m is None or not m.ring:
                    continue
                self._mark_flip_views(key, model, ipc, m, now)
                first_t, last_t = m.ring[0]["t"], m.ring[-1]["t"]
                cur = self.db.get("hcur." + key)
                lo = int(((cur if cur is not None else first_t) // step) * step)
                lo = max(lo, int((first_t // step) * step))
                hi = int(((min(now - HIST_LAG_S, last_t + step)) // step) * step)
                if hi <= lo:
                    continue
                n = int((hi - lo) // step)
                b = m.buckets(lo, n, step)
                dct = (((m.ring[-1].get("front") or {}).get("d_cached_tokens")) or {})
                split = (float(dct.get("handoff") or 0) / float(dct["total"])) if dct.get("total") else None
                pre = SERIES % model
                rows = []
                self._held("kv", sum(1 for i in range(n) if b["held"][i] is not None and b["kv_pct"][i] is not None))
                for i in range(n):
                    ts = lo + i * step
                    if b["ipc"][i] is None:
                        continue
                    dc = b["tok_dcached"][i] or 0.0
                    ho = dc if split is None else dc * split
                    vals = {"p_tps": b["p_tps"][i], "d_tps": b["d_tps"][i], "dec_tps": b["dec_tps"][i],
                            "stream_tps": b["stream_tps"][i], "kv_pct": b["kv_pct"][i], "kv_p_pct": b["kv_p_pct"][i],
                            "tok_cache": (b["tok_cache"][i] or 0.0) + (dc - ho), "tok_comp_p": b["tok_comp_p"][i],
                            "tok_comp_d": b["tok_comp_d"][i], "tok_handoff": ho, "ipc": 1.0,
                            # the honest denominators (shares of the row): rate while working = tps / busy,
                            # seats = dec_seat / dec_busy, per stream = dec_tps / dec_seat -- at every tier
                            "p_busy": b["p_busy"][i], "d_busy": b["d_busy"][i], "dec_busy": b["dec_busy"][i],
                            "dec_seat": b["dec_seat"][i], "dec_bs_min": b["dec_bs_min"][i],
                            "dec_bs_max": b["dec_bs_max"][i]}
                    vals.update({"dec_bs%d_%s" % (k, f): b["dec_bs%d_%s" % (k, f)][i] for k in activity.BS_CLASSES for f in ("tps", "busy")})
                    vals.update({"ph_" + k: b["ph_" + k][i] for k in activity.STATES})
                    # dual boots only (activity.CO_STATES): the part of ph_P with D working at the same time
                    vals.update({"ph_" + k: b["ph_" + k][i] for k in activity.CO_STATES if "ph_" + k in b})
                    rows += [(pre + k, ts, v) for k, v in vals.items() if v is not None]
                self.db.put(rows)
                c1 = self.db.get("cursor.p1")
                if c1 is not None and lo < c1:
                    self.db.rebucket([r[0] for r in rows], lo, hi)
                self.db.set("hcur." + key, hi)
                self.db.set("src.%s.dsplit" % model, "front.d_cached_tokens" if split is not None else "none")
                if self.db.get("ipc0." + key) is None:
                    self.db.set("ipc0." + key, first_t)
                self._tiers(key, model, ipc, hi)
            except Exception as e:  # one boot's defect never stops the others
                self.errors["ipc:" + str(key)] = "%s: %s" % (type(e).__name__, e)

    def _tiers(self, key: str, model: str, ipc: dict, ts: int) -> None:
        st = ((ipc or {}).get("front") or {}).get("served_tokens")
        tiers = cacheacct.tiers_from_served_tokens(st) if isinstance(st, dict) else None
        if tiers is None:
            return
        prev = self._tier_prev.get(key)
        self._tier_prev[key] = tiers
        if prev is not None:
            # the loop's delta as a rate over each of its seconds: one row per 5 s summed to 1/5 of the
            # tokens at the 1-s raster (the tile multiplies by the step)
            nsec = int(MODEL_BUCKET_S)
            self.db.put([(SERIES % model + "tier_" + k, ts - j, (tiers[k] - prev.get(k, 0)) / MODEL_BUCKET_S)
                         for k in tiers if tiers[k] > prev.get(k, 0) for j in range(nsec)])
        self.db.set("src.%s.tiers" % model, "ipc")

    def _boot_marks(self, key: str, model: str, ipc: dict) -> None:
        """Boot start / end from state.json (no log line): start = the boot id's UTC stamp (the launcher
        names the state dir ``<tag>-boot-<YYYYMMDDTHHMMSSZ>-<id>``), else serving_since_ts; end = the
        lifecycle's since_ts once it is terminal.  Card roles from groups.<G>.launch."""
        if not self.db.get("ipcboot." + key):
            t0 = boot_ts(key) or ipc.get("serving_since_ts")
            if t0:
                self.db.mark(t0, model, "boot", (ipc.get("tag") or key) + " ipc")
                self.db.set("ipcboot." + key, True)
        if ipc.get("terminal") and not self.db.get("ipcend." + key):
            t1 = ipc.get("lifecycle_since")
            if t1:
                self.db.mark(t1, model, "end", "Boot end (%s) ipc" % (ipc.get("lifecycle") or "?"))
                self.db.set("ipcend." + key, True)
        if ipc.get("launch") and not ipc.get("terminal"):
            roles = card_roles(ipc["launch"], self.cards)
            if roles and roles != self.db.get("roles." + model):
                self.db.set("roles." + model, roles)

    def _mark_flips(self, key: str, model: str, ipc: dict) -> None:
        """One flip LINE per flip (events.jsonl, the front's one clock), new ones only, without a value: Nutzer
        02.10. -- the front's own numbers (flip_first_work.flip_time_ms, flip_user_time.flip_user_ms up to the
        leg-1 dispatch) are no Flipzeit.  The value marks come from _mark_flip_views (kind flip_t2t)."""
        rows = (ipc or {}).get("flip_first_work") or []
        done = self._flips_done.get(key, 0)
        for dd in rows[done:] if done <= len(rows) else rows:
            t0 = dd.get("flip_begin_ts") or dd.get("t")
            if t0 is not None and dd.get("dir") in ("P>D", "D>P"):
                self.db.mark(t0, model, "flip", dd["dir"] + " ipc", None)
        self._flips_done[key] = len(rows)

    def _mark_flip_views(self, key: str, model: str, ipc: dict, m, now: float) -> None:
        """Nutzer 06.10.: the Flipzeit of every COUNTED flip (flipzeit.counted: kind ok, measured, not provisional),
        both directions (ipcboot.flip_views), one mark each (kind flip_t2t, value = total, label = the partition,
        flipzeit.mark_label).  A Leerlauf flip (nothing was waiting: no Prefill or Decode pending) is not counted;
        only its tally is written (kind flip_skip, no value)."""
        from . import ipcboot
        segs = ipcboot.timeline_view(m, not ipc.get("terminal"), None, now, ipcboot.boot_start(ipc),
                                     detail=False)["segs"]
        for x in ipcboot.flip_views(segs, ipc, now, m.ring):
            if x.get("kind") == "leerlauf" and x.get("begin") is not None and x.get("dir") in flipzeit.DIRS:
                self.db.mark(x["begin"], model, flipzeit.SKIP_KIND, "%s leerlauf ipc" % x["dir"])
            if not flipzeit.counted(x):
                continue
            self.db.mark(x["begin"], model, flipzeit.MARK_KIND, flipzeit.mark_label(x) + " ipc", x["total_ms"])

    # --- loop ----------------------------------------------------------------
    def publish(self, now: Optional[float] = None) -> None:
        """What the web server shows about the recorder, which runs in the sampler process."""
        self.db.set("rec.state", {"t": now or time.time(), "errors": dict(self.errors),
                                  "host_where": getattr(self, "host_where", None), "held": self.held_view(now)})

    def run_forever(self, stop: threading.Event) -> None:
        def loop(name, period, fn, phase=0.0):
            # on the clock's tick, not "period after the last run": a slow run delays the next reading,
            # it never shifts the grid (the counters book the longer interval completely)
            nxt = (math.floor(time.time() / period) + 1) * period + phase
            while not stop.is_set():
                stop.wait(max(0.0, nxt - time.time()))
                if stop.is_set():
                    break
                t0 = time.time()
                try:
                    fn(t0)
                    self.errors.pop(name, None)
                except Exception as e:
                    self.errors[name] = "%s: %s" % (type(e).__name__, str(e)[:300])
                nxt += period
                if nxt < time.time():
                    nxt = (math.floor(time.time() / period) + 1) * period + phase

        for name, period, fn, ph in (("nvml", 1.0, self.sample_gpus, 0.25), ("host", 5.0, self.sample_host, 0.6),
                                     ("model", MODEL_BUCKET_S, self.ingest_ipc, 2.4),
                                     ("compact", 60.0, self.db.compact, 31.0), ("publish", 2.0, self.publish, 0.9)):
            threading.Thread(target=loop, args=(name, period, fn, ph), name="rigdash-hist-" + name, daemon=True).start()
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

def model_src(ipc_marks: List[Optional[float]]) -> Optional[str]:
    """Source label of the model series over the shown range: only IPC (Nutzer 30.09.: keine Reihe aus
    Boot-Logs, ohne Ausnahme).  A range without any IPC sample says so instead of naming a log."""
    return IPC_LABEL if any(v is not None for v in ipc_marks) else NO_DATA_LABEL


BS_SERIES = tuple("dec_bs%d_%s" % (k, f) for k in activity.BS_CLASSES for f in ("tps", "busy"))
BUSY_SERIES = ("p_busy", "d_busy", "dec_busy", "dec_seat", "dec_bs_min", "dec_bs_max") + BS_SERIES


def derive_rates(series: Dict[str, List[Optional[float]]]) -> None:
    """The honest rates from the stored shares, per shown bucket (any tier: both are means over the same
    rows).  Rows written before the busy series existed (30.09. < ~21:30Z) take the phase share
    ``ph_dec`` / ``ph_P`` / ``ph_D`` as busy -- the phase bar's decode time, the same data -- and keep
    their stored per-stream value; they have no seat count."""
    s = series
    n = len(s.get("m.dec_tps") or [])

    def busy(k, ph):
        b, f = s.get("m." + k) or [None] * n, s.get("m.ph_" + ph) or [None] * n
        return [x if x is not None else y for x, y in zip(b, f)]
    s["m.dec_rate"] = _ratio(s["m.dec_tps"], busy("dec_busy", "dec"))
    s["m.p_rate"] = _ratio(s["m.p_tps"], busy("p_busy", "P"))
    s["m.d_rate"] = _ratio(s["m.d_tps"], busy("d_busy", "D"))
    s["m.seats"] = _ratio(s.get("m.dec_seat") or [None] * n, s.get("m.dec_busy") or [None] * n)
    new_stream = _ratio(s["m.dec_tps"], s.get("m.dec_seat") or [None] * n)
    old = s.get("m.stream_tps") or [None] * n
    s["m.stream_tps"] = [a if (s.get("m.dec_seat") or [None] * n)[i] is not None else b
                         for i, (a, b) in enumerate(zip(new_stream, old))]
    for k in activity.BS_CLASSES:
        s["m.dec_bs%d_rate" % k] = _ratio(s.get("m.dec_bs%d_tps" % k) or [None] * n, s.get("m.dec_bs%d_busy" % k) or [None] * n)
    s["m.seats_min"] = list(s.get("m.dec_bs_min") or [None] * n)
    s["m.seats_max"] = list(s.get("m.dec_bs_max") or [None] * n)


def seat_tiles(series, step: float = 1.0) -> dict:
    """Nutzer 30.09. ~21:10Z: "bei verlauf wo decode je stream steht soll auch die anzahl oder das
    mittel der sitze" -- over the shown stretch: seats time-weighted over the decode time only,
    their span, the rate while decoding; per stream = that rate / those seats."""
    tps, busy, seat = series.get("m.dec_tps") or [], series.get("m.dec_busy") or [], series.get("m.dec_seat") or []
    rows = [(a, b, c) for a, b, c in zip(tps, busy, seat) if a is not None and b is not None and c is not None]
    stok = sum(a for a, _, _ in rows)
    sb = sum(b for _, b, _ in rows)
    sc = sum(c for _, _, c in rows)
    mins = [v for v in series.get("m.seats_min") or [] if v is not None]
    maxs = [v for v in series.get("m.seats_max") or [] if v is not None]
    by_bs = {}
    for k in activity.BS_CLASSES:
        t_k = [(a, b) for a, b in zip(series.get("m.dec_bs%d_tps" % k) or [], series.get("m.dec_bs%d_busy" % k) or [])
               if a is not None and b is not None]
        busy_k = sum(b for _, b in t_k)
        # rate while working at exactly this batch size over the shown stretch; None = no interval of this size
        by_bs[str(k)] = {"rate": (sum(a for a, _ in t_k) / busy_k) if busy_k > 1e-6 else None, "busy_s": busy_k * step}
    return {"seats_mean": (sc / sb) if sb > 1e-6 else None, "dec_by_bs": by_bs,
            "seats_min": min(mins) if mins else None, "seats_max": max(maxs) if maxs else None,
            "dec_rate_mean": (stok / sb) if sb > 1e-6 else None,
            "stream_mean": (stok / sc) if sc > 1e-6 else None}


def _ratio(num: List[Optional[float]], den: List[Optional[float]], eps: float = 1e-6) -> List[Optional[float]]:
    return [(a / b) if a is not None and b is not None and b > eps else None for a, b in zip(num, den)]


def _hold(arr: List[Optional[float]], max_buckets: int) -> int:
    """A level sampled at a fixed period holds until the next sample, at most ``max_buckets`` rows:
    the NVML loop slips like the rank sampler (measured 30.09. ~21:10Z: the 1-s raster missed a second
    of the 5090 every ~10 s), and a missing second is not a missing value."""
    last, age, n, pend = None, 0, 0, 0
    for i, v in enumerate(arr):
        if v is not None:
            last, age = v, 0
            n += pend              # counted only where a real reading follows: a gap, not the live edge
            pend = 0
        elif last is not None and age < max_buckets:
            age += 1
            arr[i] = last
            pend += 1
        else:
            last, pend = None, 0
    return n


def flip_tile(db: HistoryDB, model: str, lo: float, hi: float, label: str) -> dict:
    """The Flipzeit tile of one window out of the history marks: THE function behind the Ueberblick tile (last
    flipzeit.OVERVIEW_S), the Verlauf tile (its range) and the Boot-Liste (one boot) -- same marks, same
    ``flipzeit.tile``, so the same set gives the same numbers."""
    marks = [dict(m, label=m["label"][:-4]) for m in db.marks(model, lo, hi) if (m["label"] or "").endswith(" ipc")]
    return flipzeit.tile(marks, lo, hi, label)


def zoom_step(span_s: float) -> int:
    return next((s for s in STEPS if s >= span_s / MAX_POINTS), STEPS[-1])


def view(db: HistoryDB, rec: Optional[Recorder], model: str, range_key: str, now: Optional[float] = None,
         lo_hi: Optional[Tuple[float, float]] = None) -> dict:
    """``lo_hi`` = a zoomed stretch (Nutzer 30.09. ~21:05Z: Klicken und Ziehen zoomt): the rows of
    exactly that stretch, re-read at the step that fits it (1 s from a stretch of 24 min down), not
    the range's rows stretched."""
    now = now or time.time()
    rng = RANGES.get(range_key, 3600)
    lo = now - rng
    zoom = None
    if lo_hi is not None:
        a, b = float(lo_hi[0]), min(float(lo_hi[1]), now)
        if b - a >= 10:
            lo, zoom = a, (a, b)
            rng = b - a
    step = zoom_step(rng)
    hi = zoom[1] if zoom else now
    cards = (rec.cards if rec else None) or db.get("cards", []) or []
    names = []
    for c in cards:
        names += ["g%d.%s" % (c["index"], k) for k in ("temp", "power", "clock", "util", "mem", "memclock")]
    names += ["host.cpu", "host.mem_pct", "host.bootmem_pct", "host.bootmem_gib"]
    mp = SERIES % model
    msr = ["p_tps", "d_tps", "dec_tps", "stream_tps", "kv_pct", "kv_p_pct", "ipc"] + ["tok_" + k for k in cacheacct.CLASSES] \
        + ["ph_" + k for k in activity.STATES] + ["ph_" + k for k in activity.CO_STATES] + list(BUSY_SERIES)
    names += [mp + k for k in msr] + [mp + "tier_" + k for k in cacheacct.TIERS]
    data = db.query(names, lo, hi, step, now)
    t0 = int(lo // step) * step
    ts = list(range(t0, int(hi), step))      # the buckets the query can fill (rows with ts < int(hi))
    series = {}
    for n in names:
        d = data.get(n) or {}
        series[n.replace(mp, "m.")] = [d.get(t) for t in ts]
    # host rows come every 5 s: at the 1-s raster each value holds until the next sample (no dotted line)
    if step < 5:
        for n in [k for k in series if k.startswith("host.")]:
            arr, last, age = series[n], None, 0
            for i, v in enumerate(arr):
                if v is not None:
                    last, age = v, 0
                elif last is not None and age < 5 - step:
                    age += step
                    arr[i] = last
    # NVML rows every 1 s (power from the energy counter, the levels filled by the recorder): the view's
    # hold is only the emergency fill for rows written before that (<= 2 s), and it is counted
    held_view = 0
    if step < 5:
        for n in [k for k in series if k.startswith("g")]:
            held_view += _hold(series[n], max(1, int(2 // step)))
    # Leistungsaufnahme (Nutzer 30.09.: "power draw von allen karten gemeinsam"): the sum of ALL cards;
    # a bucket where one card has no reading is a gap, never the sum of the others (that was a
    # 200-400 W drop whenever the 5090's second slipped -- the same jump class as the decode curve)
    pw = [series.get("g%d.power" % c["index"]) or [] for c in cards]
    series["gsum.power"] = [(sum(a[i] for a in pw) if pw and all(i < len(a) and a[i] is not None for a in pw)
                             else None) for i in range(len(ts))]
    # only what the IPC sampler wrote: rows an older rigdash derived from boot logs (buckets without
    # m.ipc) are not shown -- that part is "no data (before IPC recording)", a gap, not a log
    alive = series["m.ipc"]
    for k in msr:
        if k != "ipc":
            series["m." + k] = [v if alive[i] is not None else None for i, v in enumerate(series["m." + k])]
    src_model = model_src(alive)
    derive_rates(series)
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
    # marks: only those written from IPC (label "<…> ipc"); the ones an older rigdash took from log lines
    # (flip "<dir> log", boot/end without the suffix) are not shown
    marks = []
    for m in db.marks(model, lo, hi):
        lab = m["label"] or ""
        if not lab.endswith(" ipc"):
            continue
        m["label"] = lab[:-4]
        marks.append(m)
    # Nutzer 06.10.: ONE Flipzeit computation (flipzeit.tile over the flip_t2t marks of exactly this window); the
    # Ueberblick tile and the Boot-Liste call the same function (flip_tile below) over their own window
    flip_win = flipzeit.tile(marks, lo, hi, flipzeit.window_label(hi - lo, zoomed=zoom is not None))
    roles = {m: (db.get("roles." + m) or {}) for m in MODELS}
    card_info = [dict(c, roles={m: roles[m].get(c["uuid"], []) for m in MODELS}) for c in cards]
    tiles = {
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
        "live": latest("m.ipc") is not None,
        "flip": flip_win,
    }
    tiles.update(seat_tiles(series, step))
    thin = [m for m in marks if m["kind"] not in ("flip", "flip_user", "flip_pd_user", flipzeit.MARK_KIND, flipzeit.SKIP_KIND)] + \
        [m for m in marks if m["kind"] == flipzeit.MARK_KIND]
    fl = [dict(m, v=None) for m in marks if m["kind"] == "flip"]
    if len(fl) > 800:
        k = len(fl) / 800.0
        fl = [fl[int(i * k)] for i in range(800)]
    return {
        "model": model, "range": range_key, "range_s": rng, "step": step, "t": ts, "now": now,
        "zoom": list(zoom) if zoom else None, "lo": lo, "hi": hi,
        "series": series, "cards": card_info, "marks": sorted(thin + fl, key=lambda m: m["t"]),
        "marks_total": len(marks), "tiles": tiles,
        "src": {
            "cards": "NVML", "temp": "NVML GPU core (hotspot/junction not readable via NVML)",
            "host": ("Proxmox host /proc + memory.current of the htsglang containers"
                     if (getattr(rec, "host_where", None) if rec else (db.get("rec.state") or {}).get("host_where")) == "proxmox"
                     else "LXC /proc"),
            "roles": "state.json groups.launch",
            "prefill": src_model, "decode": src_model, "kv": src_model, "cache": src_model,
            "rates": "Tokens / time in which the phase worked (dec_busy/p_busy/d_busy); pauses = gap",
            "seats": "rankstats D decode.gpu_ms_by_bs (Δ per sample, weighted by round time), otherwise decode.running",
            "power": "NVML, sum of all cards",
            "cache_tiers": ("state.json front.served_tokens.*.cached_tier" if src_tiers == "ipc"
                            else "– (field served_tokens.*.cached_tier from image z30y2, 9266bdfb8d)"),
            "flip": "Flip time (flipzeit.py, user 06.10.): P→D %s; D→P %s. %s Count: only completed, measured flips (marks flip_t2t)" % (flipzeit.DEFINITION["P>D"], flipzeit.DEFINITION["D>P"], flipzeit.EXCEPTION),
            "marks": "state.json (boot id, lifecycle) + events.jsonl flip_first_work",
        },
        "errors": dict(rec.errors) if rec else dict((db.get("rec.state") or {}).get("errors") or {}),
        "held": {"view_filled": held_view, "recorder": (rec.held_view() if rec else (db.get("rec.state") or {}).get("held"))},
        "db": db.stats(),
        "ranges": list(RANGES),
    }
