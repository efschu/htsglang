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

from . import live, sources

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
            "gpuq_public": args.gpuq_public,
        }
        self.src = sources.Sources(cfg)
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
        return {
            "t": now,
            "version": self.version,
            "uptime_s": round(now - self.t0, 1),
            "boots": boots,
            "gpus": sv.get("gpus"),
            "gpu_series": self.src.gpu_series() if with_series else None,
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
            self._send(code, json.dumps(obj, default=str), "application/json")

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            try:
                if path in ("/", "/index.html"):
                    with open(os.path.join(STATIC, "index.html"), "rb") as fh:
                        return self._send(200, fh.read(), "text/html; charset=utf-8")
                if path == "/api/live":
                    series = "noseries" not in self.path
                    return self._json(app.snapshot(series))
                if path == "/api/health":
                    return self._json({"ok": True, "version": app.version,
                                       "uptime_s": round(time.time() - app.t0, 1)})
                return self._send(404, "not found", "text/plain")
            except BrokenPipeError:
                return None
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
    ap.add_argument("--gpuq-public", default="http://192.168.0.101:8770/")
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
