"""HTTP front of the rig dashboard (stdlib only, no CUDA, no sglang import).

Routes
  GET /               the live page
  GET /api/live       one JSON snapshot: boots (from their logs), GPUs,
                      containers, gpuq plan, every source's age and error
  GET /api/health     liveness of the dashboard itself
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

from . import health, live, redact, sources, weg2line

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, "static")
VERSION_FILE = os.path.join(HERE, "VERSION")


def _version():
    try:
        with open(VERSION_FILE) as fh:
            return fh.read().strip()
    except OSError:
        return "dev"


def _container_token(name: str) -> str:
    n = re.sub(r"^htsglang-(acc-)?", "", name or "")
    return n.replace("-", "").replace("_", "")


def finish_series(b: dict, gpu_series: Optional[dict], now: float, bucket_s: float) -> None:
    """Cut a series where the model really failed, and add tok/s per watt.

    Gap = GRUPPE TOT (from the first dead reason on) or the boot has ended
    (no log written and no running container) -- marked by ``gap_from`` and
    ``gap_reason``.  Power = sum of nvidia-smi power.draw over ALL cards in the
    bucket (mean of the samples); the idle draw of the sleeping group counts,
    because it is really spent.
    """
    ser = b.get("series")
    if not ser or not ser.get("t"):
        return
    ts = ser["t"]
    gap_from, reason = None, None
    a = b.get("alarm") or {}
    if a.get("state") == "TOT":
        dead = [r["t"] for r in a.get("reasons") or [] if r.get("level") == "dead" and r.get("t")]
        if dead:
            gap_from, reason = min(dead), "GRUPPE TOT"
    c = b.get("container") or {}
    if not b.get("live") and c.get("State") != "running" and b.get("last_log_t"):
        if gap_from is None or b["last_log_t"] < gap_from:
            gap_from, reason = b["last_log_t"], "Boot beendet / Container weg"
    keys = [k for k in ser if k.endswith("_tps")]
    if gap_from is not None:
        for k in keys:
            ser[k] = [None if t + bucket_s > gap_from else v for t, v in zip(ts, ser[k])]
        ser["gap_from"], ser["gap_reason"] = gap_from, reason
    power = [None] * len(ts)
    if gpu_series and gpu_series.get("t"):
        acc = [[0.0, 0] for _ in ts]
        t0 = ts[0]
        for t, row in zip(gpu_series["t"], gpu_series["power"]):
            i = int((t - t0) // bucket_s)
            if 0 <= i < len(ts) and row and all(x is not None for x in row):
                acc[i][0] += sum(row)
                acc[i][1] += 1
        power = [(a0 / n) if n else None for a0, n in acc]
    ser["power_sum_w"] = power
    for k in keys:
        ser[k + "_per_w"] = [(v / p) if (v is not None and p) else None for v, p in zip(ser[k], power)]


def attach_containers(boots, containers):
    """Give every boot the container that writes its logs, if any."""
    containers = containers or []
    by_dir = {}
    for c in containers:
        d = c.get("evidence_dir")
        if d:
            by_dir.setdefault(os.path.normpath(d), []).append(c)
    claimed = set()
    for b in boots:
        cands = by_dir.get(os.path.normpath(b["meta"].get("dir", "")), [])
        tag = b["meta"].get("tag") or b["stem"]
        hit = None
        for c in cands:
            tok = _container_token(c.get("Names", ""))
            if tok and tok in tag:
                hit = c
                break
        if hit is None and cands and b["live"]:
            hit = cands[0]
        if hit is not None and hit.get("Names") not in claimed:
            claimed.add(hit.get("Names"))
            b["container"] = {k: hit.get(k) for k in ("Names", "Image", "Status", "State", "Ports", "RunningFor")}
    return boots


class App:
    def __init__(self, args):
        self.args = args
        self.logs = live.LiveLogs(args.log_glob or None, interval=1.0)
        cfg = {
            "gpu_period": 2.0,
            "docker_ssh": shlex.split(args.docker_ssh) if args.docker_ssh else [],
            "docker_host_prefix": args.docker_host_prefix,
            "weg2_fronts": args.front or [],
            "gpuq": args.gpuq,
            "state_dir": args.state_dir or None,
        }
        self.src = sources.Sources(cfg)
        self.weg2 = weg2line.Weg2Lines(cfg["docker_ssh"], args.release_profile or [])
        self.stop = threading.Event()
        self.t0 = time.time()
        self.version = _version()

    def start(self):
        self.logs.scan()
        for target, name in ((self.logs.run_forever, "rigdash-logs"),
                             (self.src.run_forever, "rigdash-sources")):
            threading.Thread(target=target, args=(self.stop,), name=name, daemon=True).start()

    def snapshot(self, with_series=True) -> dict:
        now = time.time()
        sv = self.src.view()
        boots = self.logs.snapshot(with_series)
        docker = sv.get("docker", {}).get("value")
        attach_containers(boots, docker)
        fronts = {k: v for k, v in sv.items() if k.startswith("front:")}
        for b in boots:
            b["front"] = sources.front_for_boot(fronts, b["meta"].get("tag"))
            b["alarm"] = health.assess(b, now)
        gser = self.src.gpu_series() if with_series else None
        for b in boots:
            finish_series(b, gser, now, live.BUCKET_S)
        return {
            "t": now,
            "version": self.version,
            "uptime_s": round(now - self.t0, 1),
            "boots": boots,
            "gpus": sv.get("gpus"),
            "gpu_series": gser,
            "docker": sv.get("docker"),
            "gpuq": sv.get("gpuq"),
            "fronts": {k: {kk: vv for kk, vv in v.items() if kk != "value"} for k, v in fronts.items()},
            "collector_error": getattr(self.logs, "last_error", None),
            "windows": {"rate_s": live.WINDOW_S, "bucket_s": live.BUCKET_S, "history_s": live.HISTORY_S,
                        "live_s": live.LIVE_S},
        }


def make_handler(app: App):
    class H(BaseHTTPRequestHandler):
        server_version = "rigdash/" + app.version

        def log_message(self, fmt, *a):  # quiet; journald gets errors only
            pass

        def _send(self, code, body, ctype):
            if isinstance(body, str):
                body = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, redact.guard(json.dumps(obj, default=str)), "application/json")

        def _weg2(self, path):
            from urllib.parse import parse_qs, urlsplit

            q = {k: v[0] for k, v in parse_qs(urlsplit(self.path).query).items()}
            if not app.weg2.release_profiles:
                return self._json({"ok": False, "error": "keine Release-Profile konfiguriert (--release-profile)"}, 400)
            if path == "/api/weg2/options":
                return self._json(dict(app.weg2.options(), ok=True))
            built = app.weg2.build(q.get("profile", ""), q.get("image", ""), q.get("transport", "bar1"),
                                   q.get("house_guard", "memlimit"))
            if path == "/api/weg2/line":
                return self._json(dict(built, ok=True))
            if path == "/api/weg2/dry":
                return self._json(dict(app.weg2.dry_run(built), ok=True, line=built))
            return self._send(404, "not found", "text/plain")

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            try:
                if path in ("/", "/index.html"):
                    with open(os.path.join(STATIC, "index.html"), "rb") as fh:
                        return self._send(200, fh.read(), "text/html; charset=utf-8")
                if path == "/api/live":
                    series = "noseries" not in self.path
                    return self._json(app.snapshot(series))
                if path in ("/weg2", "/weg2.html"):
                    with open(os.path.join(STATIC, "weg2.html"), "rb") as fh:
                        return self._send(200, fh.read(), "text/html; charset=utf-8")
                if path.startswith("/api/weg2/"):
                    return self._weg2(path)
                if path == "/api/health":
                    return self._json({"ok": True, "version": app.version,
                                       "uptime_s": round(time.time() - app.t0, 1)})
                return self._send(404, "not found", "text/plain")
            except BrokenPipeError:
                return None
            except ValueError as e:
                return self._json({"ok": False, "error": str(e)}, 400)
            except Exception as e:
                return self._json({"ok": False, "error": "%s: %s" % (type(e).__name__, e)}, 500)

    return H


def main(argv=None):
    ap = argparse.ArgumentParser(prog="rigdash", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8890)
    ap.add_argument("--log-glob", action="append", default=[],
                    help="boot-log glob (repeatable); default: %s" % live.DEFAULT_LOG_GLOBS)
    ap.add_argument("--docker-ssh", default="ssh -o BatchMode=yes -o ConnectTimeout=5 proxmox",
                    help="command prefix that reaches the Docker host ('' disables)")
    ap.add_argument("--docker-host-prefix", default="/spinning/subvol-999-disk-0",
                    help="where the Docker host sees this machine's filesystem")
    ap.add_argument("--front", action="append", default=[],
                    help="weg2 front base URL to read /weg2/state from (repeatable)")
    ap.add_argument("--gpuq", default="http://127.0.0.1:8770")
    ap.add_argument("--state-dir", default="", help="keeps the 15-min card history across restarts")
    ap.add_argument("--release-profile", action="append", default=[],
                    help="profile name offered by the start-line wizard (repeatable; the unit names the release ones)")
    args = ap.parse_args(argv)
    app = App(args)
    app.start()
    srv = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    srv.daemon_threads = True
    print("rigdash %s listening on %s:%d" % (app.version, args.host, args.port), flush=True)
    try:
        srv.serve_forever()
    finally:
        app.stop.set()
    return 0
