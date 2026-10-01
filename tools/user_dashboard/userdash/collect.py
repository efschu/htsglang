"""The user dashboard's one sampler: server endpoints + NVML in this process, an in-process ring.

What it asks the server, and nothing else (all GET, never a POST, never a generation):

* ``/health``      -- liveness/readiness (front: 503 on STOP or a sick group; plain server: 200 when up)
* ``/metrics``     -- Prometheus text (front: own ``weg2_*`` families + both groups relabelled
                      ``weg2_group``; plain server: ``sglang:*``)
* ``/v1/models``   -- the served model name and context length, at most once a minute
* once per contact, only when the front's ``/metrics`` carries no group label (an older flip front
  that passes ``/metrics`` through to whichever group is awake): ``/weg2/state`` resp.
  ``/pdflip/state`` to recognise that case; the groups' own ``/metrics`` are then read directly
  (``--group-metrics``) so the counters never jump between the two servers.

Cadence (design law: nothing slows decode or a phase switch): ``active_s`` while a browser asked
within ``hold_s``, else ``idle_s`` (0 = no sampling without a browser). The VictoriaMetrics scrape of
the rig reads the same front ``/metrics`` every 15 s; the default here is 5 s only while somebody
looks and 30 s otherwise.

Every number handed out is a user figure; the internals of the server (phase switches, group names,
seats, corridors, boot tags ...) are read only where needed to compute those figures and never
leave this module.
"""

from __future__ import annotations

import collections
import json
import math
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from . import prom

GROUP_LABEL = "weg2_group"
TTFT_FRONT = "weg2_ttft_seconds"
TTFT_SERVER = "sglang:time_to_first_token_seconds"

STATUS_TEXT = {
    "bereit": "Bereit",
    "bootet": "Startet (Modell wird geladen)",
    "gestoert": "Gestört",
    "tot": "Nicht erreichbar",
}


@dataclass
class Config:
    front: str = "http://127.0.0.1:30030"
    #: "auto" = only for an old pass-through flip front; "" = never; else comma list of base URLs
    group_metrics: str = "auto"
    auto_group_urls: Tuple[str, ...] = ("http://127.0.0.1:30031", "http://127.0.0.1:30032")
    active_s: float = 5.0
    idle_s: float = 30.0
    hold_s: float = 90.0
    history_s: float = 3600.0
    models_every_s: float = 60.0
    timeout_s: float = 3.0
    #: consecutive failed contacts after which a server that WAS ready counts as gone
    dead_after: int = 2
    gpu: bool = True


def http_get(url: str, timeout: float) -> Tuple[int, bytes]:
    req = urllib.request.Request(url, method="GET", headers={"User-Agent": "userdash"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read() if e.fp else b""


# --- cards ---------------------------------------------------------------------------------------

NVSMI_QUERY = "index,name,memory.used,memory.total,utilization.gpu,power.draw,power.limit,temperature.gpu"


def _num(s: str) -> Optional[float]:
    s = (s or "").strip()
    try:
        v = float(s)
    except ValueError:
        return None
    return v if math.isfinite(v) else None


def parse_nvsmi(text: str) -> List[dict]:
    out = []
    for line in (text or "").strip().splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) != 8:
            continue
        idx = _num(p[0])
        out.append({"index": int(idx) if idx is not None else len(out), "name": p[1],
                    "mem_used_mib": _num(p[2]), "mem_total_mib": _num(p[3]), "util_pct": _num(p[4]),
                    "power_w": _num(p[5]), "power_limit_w": _num(p[6]), "temp_c": _num(p[7])})
    return out


class GpuReader:
    """NVML in this process (no CUDA context is created); ``nvidia-smi`` when pynvml is missing."""

    def __init__(self):
        self._nv = None
        self._tried = False
        self.error: Optional[str] = None

    def _init(self):
        self._tried = True
        try:
            import pynvml  # nvidia-ml-py, in the image's venv

            pynvml.nvmlInit()
            self._nv = pynvml
        except Exception as e:  # noqa: BLE001
            self._nv = None
            self.error = "NVML: %s" % type(e).__name__

    def read(self) -> List[dict]:
        if not self._tried:
            self._init()
        if self._nv is not None:
            try:
                return self._read_nvml()
            except Exception as e:  # noqa: BLE001
                self.error = "NVML: %s" % type(e).__name__
        if shutil.which("nvidia-smi"):
            try:
                txt = subprocess.run(["nvidia-smi", "--query-gpu=" + NVSMI_QUERY, "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=5).stdout
                cards = parse_nvsmi(txt)
                if cards:
                    self.error = None
                return cards
            except Exception as e:  # noqa: BLE001
                self.error = "nvidia-smi: %s" % type(e).__name__
        elif self.error is None:
            self.error = "keine Karten sichtbar (weder NVML noch nvidia-smi)"
        return []

    def _read_nvml(self) -> List[dict]:
        nv = self._nv
        out = []
        for i in range(nv.nvmlDeviceGetCount()):
            h = nv.nvmlDeviceGetHandleByIndex(i)
            d = {"index": i, "name": None, "mem_used_mib": None, "mem_total_mib": None, "util_pct": None,
                 "power_w": None, "power_limit_w": None, "temp_c": None}
            try:
                n = nv.nvmlDeviceGetName(h)
                d["name"] = n.decode() if isinstance(n, bytes) else str(n)
            except Exception:  # noqa: BLE001
                pass
            try:
                m = nv.nvmlDeviceGetMemoryInfo(h)
                d["mem_used_mib"], d["mem_total_mib"] = m.used / 1048576.0, m.total / 1048576.0
            except Exception:  # noqa: BLE001
                pass
            try:
                d["util_pct"] = float(nv.nvmlDeviceGetUtilizationRates(h).gpu)
            except Exception:  # noqa: BLE001
                pass
            try:
                d["power_w"] = nv.nvmlDeviceGetPowerUsage(h) / 1000.0
            except Exception:  # noqa: BLE001
                pass
            try:
                d["power_limit_w"] = nv.nvmlDeviceGetEnforcedPowerLimit(h) / 1000.0
            except Exception:  # noqa: BLE001
                pass
            try:
                d["temp_c"] = float(nv.nvmlDeviceGetTemperature(h, 0))  # NVML_TEMPERATURE_GPU
            except Exception:  # noqa: BLE001
                pass
            out.append(d)
        self.error = None
        return out


# --- counters ------------------------------------------------------------------------------------

class Monotone:
    """Per-series accumulation that survives a server restart: a value that went DOWN restarted at 0."""

    def __init__(self):
        self.last: Dict[tuple, float] = {}
        self.acc: Dict[tuple, float] = {}

    def feed(self, key: tuple, raw: float) -> float:
        prev = self.last.get(key)
        if prev is None:
            self.acc[key] = 0.0
        elif raw < prev:
            self.acc[key] = self.acc.get(key, 0.0) + raw
        else:
            self.acc[key] = self.acc.get(key, 0.0) + raw - prev
        self.last[key] = raw
        return self.acc[key]

    def sum(self, name: str, per_source: Dict[str, float]) -> Optional[float]:
        if not per_source:
            return None
        return sum(self.feed((name, src), v) for src, v in per_source.items())


@dataclass
class Figures:
    """The user figures of one metrics read (cumulative values; rates come from two of these)."""
    t: float
    running: Optional[float] = None
    waiting: Optional[float] = None
    decode_acc: Optional[float] = None
    prefill_acc: Optional[float] = None
    cache_acc: Optional[float] = None
    kv_ratio: Optional[float] = None
    kv_total: Optional[float] = None
    ttft_family: Optional[str] = None
    ttft_cum: Dict[float, float] = field(default_factory=dict)


def _max_per_source(rows) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for lb, v in rows:
        if v is None or not math.isfinite(v):
            continue
        s = prom.source_of(lb, GROUP_LABEL)
        out[s] = max(out.get(s, -math.inf), v)
    return out


class Collector:
    RATE_WINDOW_S = 15.0

    def __init__(self, cfg: Config, fetch: Optional[Callable[[str, float], Tuple[int, bytes]]] = None,
                 gpu_reader=None, clock: Callable[[], float] = time.time):
        self.cfg = cfg
        self.fetch = fetch or http_get
        self.gpus = gpu_reader if gpu_reader is not None else (GpuReader() if cfg.gpu else None)
        self.clock = clock
        self.lock = threading.Lock()
        self.mono = Monotone()
        self.ring: collections.deque = collections.deque(maxlen=20000)
        self.ttft_ring: collections.deque = collections.deque(maxlen=20000)   # (t, family, cum buckets)
        self._ttft_raw_prev: Optional[Dict[float, float]] = None
        self._ttft_cum: Dict[float, float] = {}
        self._ttft_family: Optional[str] = None
        self._prev: Optional[Figures] = None
        self._recent: collections.deque = collections.deque(maxlen=64)
        self.started = clock()
        self.ready_since: Optional[float] = None
        self.last_contact: Optional[float] = None
        self.fail_streak = 0
        self.status = "bootet"
        self.error: Optional[str] = None
        self.model: Optional[str] = None
        self.context_len: Optional[int] = None
        self._models_at = -1e18
        self.legacy_groups: Optional[List[str]] = None   # set when the front only passes /metrics through
        self._legacy_checked = False
        self.latest: dict = {}
        self.last_sample_t: Optional[float] = None
        self.last_browser = -1e18
        self.wake = threading.Event()
        self.stop = threading.Event()
        self.thread: Optional[threading.Thread] = None

    # --- cadence -----------------------------------------------------------------------------
    def touch(self) -> None:
        """A browser asked: sample at the live cadence for the next ``hold_s``."""
        was_idle = not self.active()
        self.last_browser = self.clock()
        if was_idle:
            self.wake.set()

    def active(self) -> bool:
        return self.clock() - self.last_browser < self.cfg.hold_s

    def period(self) -> Optional[float]:
        if self.active():
            return self.cfg.active_s
        return self.cfg.idle_s if self.cfg.idle_s > 0 else None

    def start(self) -> None:
        self.thread = threading.Thread(target=self._run, name="userdash-sampler", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while not self.stop.is_set():
            try:
                self.sample_once()
            except Exception as e:  # noqa: BLE001 - one bad read never ends the sampler
                self.error = "Abtastung: %s" % type(e).__name__
            p = self.period()
            self.wake.wait(p if p is not None else 3600.0)
            self.wake.clear()

    def alive(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    # --- one sample ----------------------------------------------------------------------------
    def _get(self, path: str, base: Optional[str] = None) -> Tuple[int, bytes]:
        return self.fetch((base or self.cfg.front).rstrip("/") + path, self.cfg.timeout_s)

    def _health(self) -> Optional[int]:
        try:
            code, _ = self._get("/health")
            return code
        except Exception as e:  # noqa: BLE001
            self.error = "keine Verbindung (%s)" % type(e).__name__
            return None

    def _update_status(self, code: Optional[int], now: float) -> None:
        if code is None:
            self.fail_streak += 1
            if self.ready_since is not None and self.fail_streak >= self.cfg.dead_after:
                self.status = "tot"
            elif self.ready_since is None:
                self.status = "bootet"
            return
        self.fail_streak = 0
        self.last_contact = now
        if code == 200:
            if self.ready_since is None or self.status == "tot":
                self.ready_since = now        # first ready, or back after the server was gone
            if self.status != "bereit":
                self._models_at = -1e18       # read the model name now
            self.status = "bereit"
            self.error = None
        elif self.ready_since is None:
            self.status = "bootet"        # HTTP is up, the model is still loading
            self.error = None
        else:
            self.status = "gestoert"
            self.error = "Server meldet HTTP %d auf /health" % code

    def _models(self, now: float) -> None:
        if now - self._models_at < self.cfg.models_every_s:
            return
        self._models_at = now
        try:
            code, body = self._get("/v1/models")
            if code != 200:
                return
            data = (json.loads(body.decode("utf-8", "replace")) or {}).get("data") or []
        except Exception:  # noqa: BLE001
            return
        pick = None
        for m in data:
            if isinstance(m, dict) and m.get("live", True) is not False:
                pick = m
                break
        pick = pick or (data[0] if data and isinstance(data[0], dict) else None)
        if pick:
            self.model = str(pick.get("id") or "") or None
            ml = pick.get("max_model_len") or pick.get("context_length")
            self.context_len = int(ml) if isinstance(ml, (int, float)) else None

    def _legacy_front(self) -> bool:
        """An older flip front: /metrics is one group's at a time. Recognised by its state route."""
        for path in ("/weg2/state", "/pdflip/state"):
            try:
                code, body = self._get(path)
            except Exception:  # noqa: BLE001
                continue
            if code == 200:
                try:
                    d = json.loads(body.decode("utf-8", "replace"))
                except ValueError:
                    continue
                if isinstance(d, dict) and "awake" in d:
                    return True
        return False

    def _metrics(self) -> Optional[List[prom.Sample]]:
        if self.legacy_groups:
            out: List[prom.Sample] = []
            ok = False
            for i, base in enumerate(self.legacy_groups):
                try:
                    code, body = self._get("/metrics", base)
                except Exception:  # noqa: BLE001
                    continue
                if code != 200:
                    continue
                ok = True
                tag = "g%d" % i
                for n, lb, v in prom.parse(body.decode("utf-8", "replace")):
                    out.append((n, tuple(sorted(lb + (("_src", tag),))), v))
            return out if ok else None
        try:
            code, body = self._get("/metrics")
        except Exception:  # noqa: BLE001
            return None
        if code != 200:
            return None
        samples = prom.parse(body.decode("utf-8", "replace"))
        if not self._legacy_checked and samples:
            self._legacy_checked = True
            if not prom.has_label(samples, GROUP_LABEL) and self.cfg.group_metrics != "" and self._legacy_front():
                urls = (list(self.cfg.auto_group_urls) if self.cfg.group_metrics == "auto"
                        else [u.strip() for u in self.cfg.group_metrics.split(",") if u.strip()])
                self.legacy_groups = urls or None
                if self.legacy_groups:
                    return self._metrics()
        return samples

    def figures(self, samples: List[prom.Sample], now: float) -> Figures:
        f = Figures(t=now)
        # requests: the front's own gauges, else the servers' scheduler gauges
        out_rows = prom.select(samples, "weg2_outstanding")
        q_rows = prom.select(samples, "weg2_queue_len")
        if out_rows or q_rows:
            f.running = sum(v for _, v in out_rows) if out_rows else None
            f.waiting = sum(v for _, v in q_rows) if q_rows else None
        else:
            f.running = prom.total(prom.select(samples, "sglang:num_running_reqs"))
            f.waiting = prom.total(prom.select(samples, "sglang:num_queue_reqs"))
        # tokens: the scheduler's live counter (updated every log interval), else per finished request
        rt = prom.select(samples, "sglang:realtime_tokens_total")
        if rt:
            def mode(m):
                return [(lb, v) for lb, v in rt if prom.label(lb, "mode") == m]
            f.decode_acc = self.mono.sum("decode", prom.collapse(mode("decode")))
            f.prefill_acc = self.mono.sum("prefill", prom.collapse(mode("prefill_compute")))
            f.cache_acc = self.mono.sum("cache", prom.collapse(mode("prefill_cache")))
        else:
            gen = prom.collapse(prom.select(samples, "sglang:generation_tokens_total"))
            prompt = prom.collapse(prom.select(samples, "sglang:prompt_tokens_total"))
            cached = prom.collapse(prom.select(samples, "sglang:cached_tokens_total"))
            f.decode_acc = self.mono.sum("generation", gen)
            p = self.mono.sum("prompt", prompt)
            c = self.mono.sum("cached", cached) if cached else (0.0 if p is not None else None)
            f.prefill_acc = (p - c) if p is not None and c is not None else p
            f.cache_acc = c
        # KV: the awake server when the front names it, else the fullest one
        kv = _max_per_source(prom.select(samples, "sglang:token_usage")) or \
            _max_per_source(prom.select(samples, "sglang:full_token_usage"))
        awake = [prom.label(lb, "group") for lb, v in prom.select(samples, "weg2_awake") if v >= 1]
        src = awake[0] if awake and awake[0] in kv else (max(kv, key=kv.get) if kv else None)
        if src is not None:
            f.kv_ratio = kv[src]
            f.kv_total = _max_per_source(prom.select(samples, "sglang:max_total_num_tokens")).get(src)
        # TTFT: arrival -> first token at the front, else the server's own histogram
        fam = TTFT_FRONT if prom.has_family(samples, TTFT_FRONT + "_bucket") else (
            TTFT_SERVER if prom.has_family(samples, TTFT_SERVER + "_bucket") else None)
        if fam is not None:
            raw = prom.histogram(samples, fam)
            if fam != self._ttft_family:
                self._ttft_family, self._ttft_raw_prev, self._ttft_cum = fam, None, {}
                self.ttft_ring.clear()
            d = prom.bucket_delta(raw, self._ttft_raw_prev) if self._ttft_raw_prev is not None else \
                {le: 0.0 for le in raw}
            for le, c in d.items():
                self._ttft_cum[le] = self._ttft_cum.get(le, 0.0) + c
            self._ttft_raw_prev = raw
            f.ttft_family, f.ttft_cum = fam, dict(self._ttft_cum)
        return f

    def ttft_window(self, now: float, window_s: float) -> Tuple[Optional[float], Optional[float], int]:
        if not self.ttft_ring:
            return None, None, 0
        _, fam, cur = self.ttft_ring[-1]
        base = None
        for t, f2, cum in self.ttft_ring:
            if f2 == fam and t >= now - window_s:
                base = cum
                break
        d = prom.bucket_delta(cur, base) if base is not None else {}
        n = int(round(prom.count_of(d)))
        if n <= 0:
            return None, None, 0
        return prom.quantile(0.5, d), prom.quantile(0.9, d), n

    def sample_once(self) -> dict:
        now = self.clock()
        code = self._health()
        self._update_status(code, now)
        fig = None
        if code is not None:
            if self.status == "bereit":
                self._models(now)
            samples = self._metrics()
            if samples:
                fig = self.figures(samples, now)
        cards = []
        if self.gpus is not None:
            try:
                cards = self.gpus.read()
            except Exception:  # noqa: BLE001
                cards = []
        point = {"t": now, "status": self.status, "running": None, "waiting": None, "decode_tps": None,
                 "prefill_tps": None, "cache_hit_pct": None, "kv_pct": None, "ttft_p50_s": None,
                 "ttft_p90_s": None,
                 "gpus": [[c.get("util_pct"), c.get("mem_used_mib"), c.get("power_w"), c.get("temp_c")]
                          for c in cards]}
        # rates over at least RATE_WINDOW_S: the scheduler's token counter moves once per log interval, a 5-s
        # difference alone jumps between 0 and the double (live 27B 01.10.: 0, 93, 209, 125, 251 tok/s)
        prev = self._prev
        for old in self._recent:
            if fig is not None and old.t >= now - self.RATE_WINDOW_S and old.t < fig.t:
                prev = old
                break
        if fig is not None:
            point["running"], point["waiting"] = fig.running, fig.waiting
            point["kv_pct"] = fig.kv_ratio * 100.0 if fig.kv_ratio is not None else None
            if prev is not None and fig.t > prev.t:
                dt = fig.t - prev.t
                if fig.decode_acc is not None and prev.decode_acc is not None:
                    point["decode_tps"] = max(0.0, (fig.decode_acc - prev.decode_acc) / dt)
                if fig.prefill_acc is not None and prev.prefill_acc is not None:
                    point["prefill_tps"] = max(0.0, (fig.prefill_acc - prev.prefill_acc) / dt)
                if None not in (fig.cache_acc, prev.cache_acc, fig.prefill_acc, prev.prefill_acc):
                    dc = fig.cache_acc - prev.cache_acc
                    dp = fig.prefill_acc - prev.prefill_acc
                    if dc + dp > 0:
                        point["cache_hit_pct"] = 100.0 * max(0.0, dc) / (dc + max(0.0, dp))
            if fig.ttft_family is not None:
                self.ttft_ring.append((now, fig.ttft_family, fig.ttft_cum))
                p50, p90, _ = self.ttft_window(now, 300.0)
                point["ttft_p50_s"], point["ttft_p90_s"] = p50, p90
            self._prev = fig
            self._recent.append(fig)
        with self.lock:
            self.ring.append(point)
            cut = now - self.cfg.history_s
            while self.ring and self.ring[0]["t"] < cut:
                self.ring.popleft()
            while self.ttft_ring and self.ttft_ring[0][0] < cut - 60:
                self.ttft_ring.popleft()
            self.latest = self._snapshot(point, cards, fig, now)
            self.last_sample_t = now
        return point

    def _snapshot(self, point: dict, cards: List[dict], fig: Optional[Figures], now: float) -> dict:
        p50_5, p90_5, n5 = self.ttft_window(now, 300.0)
        p50_h, p90_h, nh = self.ttft_window(now, 3600.0)
        basis = None
        if self._ttft_family == TTFT_FRONT:
            basis = "Ankunft der Anfrage bis zum ersten Token"
        elif self._ttft_family == TTFT_SERVER:
            basis = "im Server gemessen (ohne Wartezeit davor)"
        return {
            "server": {
                "status": self.status,
                "status_text": STATUS_TEXT.get(self.status, self.status),
                "model": self.model,
                "context_len": self.context_len,
                "ready_since": self.ready_since if self.status == "bereit" else None,
                "uptime_s": (now - self.ready_since) if (self.status == "bereit" and self.ready_since) else None,
                "watching_since": self.started,
                "last_contact": self.last_contact,
                "error": self.error,
            },
            "requests": {"running": point["running"], "waiting": point["waiting"]},
            "throughput": {"decode_tps": point["decode_tps"], "prefill_tps": point["prefill_tps"],
                           "cache_hit_pct": point["cache_hit_pct"]},
            "ttft": {"p50_5m_s": p50_5, "p90_5m_s": p90_5, "n_5m": n5,
                     "p50_1h_s": p50_h, "p90_1h_s": p90_h, "n_1h": nh, "basis": basis},
            # the percentage is the server's own token_usage; kv_used_tokens counts a different pool on
            # hybrid models (live 27B 01.10.: 62576 used vs 21 % of 768256), so only the capacity is shown
            "kv": {"used_pct": point["kv_pct"], "total_tokens": fig.kv_total if fig else None},
            "gpus": [{k: c.get(k) for k in ("index", "name", "mem_used_mib", "mem_total_mib", "util_pct",
                                             "power_w", "power_limit_w", "temp_c")} for c in cards],
            "gpu_error": getattr(self.gpus, "error", None) if self.gpus is not None else "abgeschaltet",
            "sample": {"t": now, "period_s": self.period(), "live": self.active()},
        }

    # --- reads for the HTTP side ---------------------------------------------------------------------
    def now_view(self) -> dict:
        with self.lock:
            return dict(self.latest)

    def history(self, span_s: float = 3600.0) -> dict:
        span_s = max(60.0, min(float(span_s), self.cfg.history_s))
        with self.lock:
            rows = [p for p in self.ring if p["t"] >= self.clock() - span_s]
            names = [c.get("name") for c in (self.latest.get("gpus") or [])]
        keys = ("decode_tps", "prefill_tps", "ttft_p50_s", "ttft_p90_s", "kv_pct", "running", "waiting")
        out = {"t": [p["t"] for p in rows], "span_s": span_s}
        for k in keys:
            out[k] = [p.get(k) for p in rows]
        ng = max([len(p["gpus"]) for p in rows] + [len(names)])
        g = {"names": names, "util_pct": [], "mem_used_mib": [], "power_w": [], "temp_c": []}
        for i in range(ng):
            for j, k in enumerate(("util_pct", "mem_used_mib", "power_w", "temp_c")):
                g[k].append([(p["gpus"][i][j] if i < len(p["gpus"]) else None) for p in rows])
        out["gpus"] = g
        return out

    def health(self) -> Tuple[int, dict]:
        """/healthz: the dashboard itself (not the model server) -- 200 while its sampler lives."""
        age = (self.clock() - self.last_sample_t) if self.last_sample_t else None
        ok = self.alive() or self.thread is None
        return (200 if ok else 503), {
            "ok": ok, "dienst": "userdash", "sampler_alive": self.alive(),
            "server_status": self.status, "last_sample_age_s": round(age, 1) if age is not None else None,
            "period_s": self.period(),
        }
