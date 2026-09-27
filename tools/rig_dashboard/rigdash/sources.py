"""Non-log sources: nvidia-smi, the Docker host, the weg2 front, gpuq.

Every source is a small sampler with its own period and its own last error,
so one dead source (ssh to the host down, the front not listening) shows as
that source's error on the page instead of blanking the page.

Nothing here writes to or requests work from a model server:

* nvidia-smi is a query of the driver (no CUDA context is created, so the
  dashboard never pins the driver the way the old planner did).
* the Docker host is asked ``docker ps`` / ``docker inspect`` over ssh.
* the weg2 front is asked ``GET /weg2/state`` -- a dict it already keeps.
* gpuq is asked ``GET /api/v1/cards`` and ``GET /api/v1/bookings``.
"""

from __future__ import annotations

import collections
import json
import os
import subprocess
import threading
import time
import urllib.request
from typing import Callable, Dict, List, Optional

NVSMI_FIELDS = [
    "index", "name", "uuid", "power.draw", "power.limit", "memory.used",
    "memory.total", "utilization.gpu", "temperature.gpu", "clocks.sm",
]


def _to_num(s: str):
    s = s.strip()
    if s in ("", "[N/A]", "N/A", "[Not Supported]"):
        return None
    try:
        return float(s) if "." in s else int(s)
    except ValueError:
        return s


def parse_nvsmi_csv(text: str) -> List[dict]:
    cards = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != len(NVSMI_FIELDS):
            continue
        d = {}
        for k, v in zip(NVSMI_FIELDS, parts):
            d[k] = v if k in ("name", "uuid") else _to_num(v)
        cards.append(d)
    return cards


DOCKER_PS_FIELDS = ["Names", "Image", "Status", "State", "Ports", "CreatedAt", "RunningFor"]
# Explicit fields, NOT '{{json .}}': the json form includes .Size, which makes
# the daemon compute every container's disk usage -- measured 12-13 s per call
# on the Proxmox host 2026-09-27 against 0.6 s for the field list.
DOCKER_PS_FORMAT = "\\t".join("{{.%s}}" % f for f in DOCKER_PS_FIELDS)


def parse_docker_ps(text: str) -> List[dict]:
    out = []
    for line in text.strip().splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != len(DOCKER_PS_FIELDS):
            continue
        out.append(dict(zip(DOCKER_PS_FIELDS, parts)))
    return out


def container_log_dirs(inspect: List[dict], host_prefix: str) -> Dict[str, str]:
    """name -> local evidence dir, from the container's bind mounts.

    The Docker host sees this container's filesystem under ``host_prefix``
    (``/spinning/subvol-999-disk-0``); stripping it gives the path here.
    """
    res = {}
    for c in inspect:
        name = (c.get("Name") or "").lstrip("/")
        for m in c.get("Mounts") or []:
            if m.get("Destination") == "/var/lib/htsglang/evidence":
                src = m.get("Source") or ""
                if host_prefix and src.startswith(host_prefix):
                    src = src[len(host_prefix):] or "/"
                res[name] = src
    return res


def container_health_output(inspect: List[dict]) -> Dict[str, str]:
    """name -> the last healthcheck's output (redacted, one line, 300 chars)."""
    from . import redact
    out = {}
    for c in inspect:
        name = (c.get("Name") or "").lstrip("/")
        log = (((c.get("State") or {}).get("Health") or {}).get("Log") or [])
        if log:
            txt = " ".join(str(log[-1].get("Output") or "").split())
            txt = redact.clean(txt)
            if txt:
                out[name] = txt[:300]
    return out


class Sampler:
    def __init__(self, name: str, period: float, fn: Callable[[], object]):
        self.name, self.period, self.fn = name, period, fn
        self.value = None
        self.t = None
        self.error = None
        self.error_t = None
        self.duration_ms = None

    def tick(self, now: float):
        if self.t is not None and now - self.t < self.period and self.error is None:
            return
        if self.error_t is not None and now - self.error_t < self.period:
            return
        t0 = time.time()
        try:
            self.value = self.fn()
            self.t = time.time()
            self.error = None
            self.error_t = None
        except Exception as e:
            self.error = "%s: %s" % (type(e).__name__, str(e)[:300])
            self.error_t = time.time()
        self.duration_ms = round((time.time() - t0) * 1000, 1)

    def view(self, now: float) -> dict:
        return {
            "value": self.value,
            "age_s": round(now - self.t, 1) if self.t else None,
            "error": self.error,
            "period_s": self.period,
            "duration_ms": self.duration_ms,
        }


def run(cmd: List[str], timeout: float) -> str:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError("rc=%d %s" % (p.returncode, (p.stderr or p.stdout).strip()[:200]))
    return p.stdout


def http_json(url: str, timeout: float = 3.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


class Sources:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        ssh = cfg.get("docker_ssh") or []
        self.samplers: Dict[str, Sampler] = {
            "gpus": Sampler("gpus", cfg.get("gpu_period", 2.0), self.sample_gpus),
            "gpuq": Sampler("gpuq", cfg.get("gpuq_period", 15.0), self.sample_gpuq),
        }
        if ssh:
            self.samplers["docker"] = Sampler("docker", cfg.get("docker_period", 20.0), self.sample_docker)
        for ep in cfg.get("weg2_fronts") or []:
            self.samplers["front:" + ep] = Sampler(
                "front:" + ep, cfg.get("front_period", 10.0),
                (lambda ep=ep: http_json(ep.rstrip("/") + "/weg2/state", 3.0)))
        self.gpu_hist = collections.deque(maxlen=int(15 * 60 / cfg.get("gpu_period", 2.0)) + 5)
        self.lock = threading.Lock()
        self.state_dir = cfg.get("state_dir")
        self._last_save = 0.0
        self.load_history()

    # --- the 15-min card history survives a restart (the boot series need no
    # file: they are rebuilt from the logs' own backfill) ---------------
    def _hist_path(self):
        return os.path.join(self.state_dir, "gpu_hist.json") if self.state_dir else None

    def load_history(self):
        p = self._hist_path()
        if not p:
            return
        try:
            with open(p) as fh:
                rows = json.load(fh)
        except (OSError, ValueError):
            return
        cut = time.time() - 15 * 60
        with self.lock:
            for t, cs in rows:
                if t >= cut:
                    self.gpu_hist.append((t, [tuple(c) for c in cs]))

    def save_history(self, now: float, every: float = 30.0):
        p = self._hist_path()
        if not p or now - self._last_save < every:
            return
        self._last_save = now
        with self.lock:
            rows = [[t, [list(c) for c in cs]] for t, cs in self.gpu_hist]
        try:
            os.makedirs(self.state_dir, exist_ok=True)
            tmp = p + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(rows, fh)
            os.replace(tmp, p)
        except OSError:
            pass

    # --- samplers ------------------------------------------------------
    def sample_gpus(self):
        out = run(["nvidia-smi", "--query-gpu=" + ",".join(NVSMI_FIELDS),
                   "--format=csv,noheader,nounits"], 8.0)
        cards = parse_nvsmi_csv(out)
        now = time.time()
        with self.lock:
            self.gpu_hist.append((now, [(c.get("power.draw"), c.get("memory.used"),
                                          c.get("utilization.gpu")) for c in cards]))
        self.save_history(now)
        return cards

    def sample_docker(self):
        ssh = self.cfg["docker_ssh"]
        ps = run(ssh + ["docker ps -a --filter name=htsglang --format '%s'" % DOCKER_PS_FORMAT], 12.0)
        rows = parse_docker_ps(ps)
        running = [r["Names"] for r in rows if (r.get("State") == "running")]
        dirs, health_out = {}, {}
        if running:
            ins = run(ssh + ["docker inspect " + " ".join(running)], 12.0)
            try:
                parsed = json.loads(ins)
                dirs = container_log_dirs(parsed, self.cfg.get("docker_host_prefix", ""))
                health_out = container_health_output(parsed)
            except ValueError:
                dirs = {}
        rows.sort(key=lambda r: (r.get("State") != "running", r.get("CreatedAt", "")), reverse=False)
        for r in rows:
            r["evidence_dir"] = dirs.get(r.get("Names"))
            r["health_output"] = health_out.get(r.get("Names"))
        return rows[:12]

    def sample_gpuq(self):
        base = self.cfg.get("gpuq", "http://127.0.0.1:8770").rstrip("/")
        cards = http_json(base + "/api/v1/cards", 3.0)
        bookings = http_json(base + "/api/v1/bookings", 3.0)
        live = [b for b in bookings if b.get("state") in ("running", "pending")]
        live.sort(key=lambda b: b.get("position", 0))
        return {"cards": cards, "bookings": live[:20]}

    # --- loop ----------------------------------------------------------
    def tick(self):
        now = time.time()
        for s in self.samplers.values():
            s.tick(now)

    def gpu_series(self) -> dict:
        with self.lock:
            h = list(self.gpu_hist)
        return {
            "t": [t for t, _ in h],
            "power": [[c[0] for c in cs] for _, cs in h],
            "mem": [[c[1] for c in cs] for _, cs in h],
            "util": [[c[2] for c in cs] for _, cs in h],
        }

    def view(self) -> dict:
        now = time.time()
        return {k: s.view(now) for k, s in self.samplers.items()}

    def run_forever(self, stop: threading.Event):
        """One thread per sampler: a 12-s ssh stall must not freeze the GPU tiles."""

        def loop(s: Sampler):
            while not stop.is_set():
                s.tick(time.time())
                stop.wait(0.5)

        threads = [threading.Thread(target=loop, args=(s,), name="rigdash-" + s.name, daemon=True)
                   for s in self.samplers.values()]
        for t in threads:
            t.start()
        stop.wait()


def front_for_boot(fronts: Dict[str, dict], tag: Optional[str]) -> Optional[dict]:
    """The /weg2/state sample whose ``tag`` is this boot's launcher tag."""
    if not tag:
        return None
    for ep, v in fronts.items():
        val = v.get("value") if v else None
        if isinstance(val, dict) and val.get("tag") == tag:
            return dict(val, endpoint=ep.split(":", 1)[1], age_s=v.get("age_s"))
    return None
