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
               rankstats (<state>/rankstate/<G>/*.rankstats, weg2.rankstats/1, timer-written) --
               P-/D-Prefill tok/s = Δprefill.new_tokens / Δts, Decode tok/s = Δdecode.tokens / Δts,
               Decode je Stream = that / decode.running (only while D decoded continuously),
               KV = sched.full_token_usage (level), input-token classes per CHUNK from
               prefill.new_tokens / prefill.cached_tokens (P cached = cache, D cached = Übergabe P->D;
               the rank cannot tell a D-direct prefix hit from the hand-over, so it is never counted as
               cache), cache tiers from state.json front.served_tokens.*.cached_tier;
               flips: events.jsonl flip_first_work.
               A stretch the recorder did not watch live (rigdash was down, or before this image had
               rankstats) stays empty: "keine Daten (vor IPC-Aufzeichnung)", never a log backfill.
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

from . import cacheacct, ipcstate

NO_DATA_LABEL = "keine Daten (vor IPC-Aufzeichnung)"
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
#: are not shown any more, that stretch reads "keine Daten (vor IPC-Aufzeichnung)"
SERIES = "mi.%s."
HIST_LAG_S = 20.0           # buckets are written once the last pipeline stage has reported their chunks


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
                for i in range(n):
                    ts = lo + i * step
                    if b["ipc"][i] is None:
                        continue
                    dc = b["tok_dcached"][i] or 0.0
                    ho = dc if split is None else dc * split
                    vals = {"p_tps": b["p_tps"][i], "d_tps": b["d_tps"][i], "dec_tps": b["dec_tps"][i],
                            "stream_tps": b["stream_tps"][i], "kv_pct": b["kv_pct"][i], "kv_p_pct": b["kv_p_pct"][i],
                            "tok_cache": (b["tok_cache"][i] or 0.0) + (dc - ho), "tok_comp_p": b["tok_comp_p"][i],
                            "tok_comp_d": b["tok_comp_d"][i], "tok_handoff": ho, "ipc": 1.0}
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
            self.db.put([(SERIES % model + "tier_" + k, ts, (tiers[k] - prev.get(k, 0)) / MODEL_BUCKET_S)  # per loop
                         for k in tiers if tiers[k] > prev.get(k, 0)])
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
                self.db.mark(t1, model, "end", "Boot-Ende (%s) ipc" % (ipc.get("lifecycle") or "?"))
                self.db.set("ipcend." + key, True)
        if ipc.get("launch") and not ipc.get("terminal"):
            roles = card_roles(ipc["launch"], self.cards)
            if roles and roles != self.db.get("roles." + model):
                self.db.set("roles." + model, roles)

    def _mark_flips(self, key: str, model: str, ipc: dict) -> None:
        """One mark per flip (events.jsonl, the front's one clock), new ones only.  Value = Flipzeit
        after the user's definition, never flip_total: P>D = flip_first_work.flip_time_ms (what="none"
        gives no value); D>P = flip_user_time.flip_user_ms (build y4z on) -- a D>P flip_first_work
        only draws the flip line, without a value (no fallback)."""
        rows = (ipc or {}).get("flip_first_work") or []
        done = self._flips_done.get(key, 0)
        for dd in rows[done:] if done <= len(rows) else rows:
            t0 = dd.get("flip_begin_ts") or dd.get("t")
            if t0 is not None and dd.get("dir") in ("P>D", "D>P"):
                v = dd.get("flip_time_ms") if dd.get("dir") == "P>D" and dd.get("what") != "none" else None
                self.db.mark(t0, model, "flip", dd["dir"] + " ipc", v)
        self._flips_done[key] = len(rows)
        ut = (ipc or {}).get("flip_user_time") or []
        done_u = self._flips_done.get(key + "#ut", 0)
        for u in ut[done_u:] if done_u <= len(ut) else ut:
            if u.get("flip_user_ms") is not None and u.get("start_ts") is not None:
                self.db.mark(u["start_ts"], model, "flip_user", "D>P ipc", u["flip_user_ms"])
        self._flips_done[key + "#ut"] = len(ut)

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
                                 ("model", MODEL_BUCKET_S, self.ingest_ipc),
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


def model_src(ipc_marks: List[Optional[float]]) -> Optional[str]:
    """Source label of the model series over the shown range: only IPC (Nutzer 30.09.: keine Reihe aus
    Boot-Logs, ohne Ausnahme).  A range without any IPC sample says so instead of naming a log."""
    return IPC_LABEL if any(v is not None for v in ipc_marks) else NO_DATA_LABEL


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
    mp = SERIES % model
    msr = ["p_tps", "d_tps", "dec_tps", "stream_tps", "kv_pct", "kv_p_pct", "ipc"] + ["tok_" + k for k in cacheacct.CLASSES]
    names += [mp + k for k in msr] + [mp + "tier_" + k for k in cacheacct.TIERS]
    data = db.query(names, lo, now, step, now)
    t0 = int(lo // step) * step
    ts = list(range(t0, int(now) + 1, step))
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
    # Leistungsaufnahme (Nutzer 30.09.: "power draw von allen karten gemeinsam"): the sum of the cards
    # that reported in the bucket; None only when no card did
    pw = [series.get("g%d.power" % c["index"]) or [] for c in cards]
    series["gsum.power"] = [(sum(a[i] for a in pw if i < len(a) and a[i] is not None)
                             if any(i < len(a) and a[i] is not None for a in pw) else None) for i in range(len(ts))]
    # only what the IPC sampler wrote: rows an older rigdash derived from boot logs (buckets without
    # m.ipc) are not shown -- that part is "keine Daten (vor IPC-Aufzeichnung)", a gap, not a log
    alive = series["m.ipc"]
    for k in msr:
        if k != "ipc":
            series["m." + k] = [v if alive[i] is not None else None for i, v in enumerate(series["m." + k])]
    src_model = model_src(alive)
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
    for m in db.marks(model, lo, now):
        lab = m["label"] or ""
        if not lab.endswith(" ipc"):
            continue
        m["label"] = lab[:-4]
        marks.append(m)
    flips_pd = [m["v"] for m in marks if m["kind"] == "flip" and m["label"] == "P>D" and m["v"] is not None]
    last_pd = next((m for m in reversed(marks) if m["kind"] == "flip" and m["label"] == "P>D" and m["v"] is not None), None)
    roles = {m: (db.get("roles." + m) or {}) for m in MODELS}
    card_info = [dict(c, roles={m: roles[m].get(c["uuid"], []) for m in MODELS}) for c in cards]
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
        "live": latest("m.ipc") is not None,
        "flip_last_ms": last_pd["v"] if last_pd else None, "flip_last_t": last_pd["t"] if last_pd else None,
        "flip_median_ms": _pct(flips_pd, 0.5), "flip_n": len(flips_pd),
    }
    thin = [m for m in marks if m["kind"] not in ("flip", "flip_user")] + [m for m in marks if m["kind"] == "flip_user"]
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
            "prefill": src_model, "decode": src_model, "kv": src_model, "cache": src_model,
            "power": "NVML, Summe aller Karten",
            "cache_tiers": ("state.json front.served_tokens.*.cached_tier" if src_tiers == "ipc"
                            else "– (Feld served_tokens.*.cached_tier ab Image z30y2, 9266bdfb8d)"),
            "flip": "events.jsonl flip_first_work (P→D) + flip_user_time (D→P)",
            "marks": "state.json (Boot-ID, lifecycle) + events.jsonl flip_first_work",
        },
        "errors": dict(rec.errors) if rec else {},
        "db": db.stats(),
        "ranges": list(RANGES),
    }
