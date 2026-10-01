"""HTTP side of the user dashboard: one page, two JSON reads, one health route. GET only.

Routes
  /                 the page (static/index.html)
  /mark.svg         the logo
  /api/now          current figures (counts as a browser: the sampler goes to the live cadence)
  /api/history?s=N  ring buffer of the last N seconds (default and maximum: the ring, 1 h)
  /healthz          the dashboard itself: 200 + JSON while its sampler lives, 503 otherwise;
                    does NOT count as a browser, so a watcher polling it never raises the cadence
"""

from __future__ import annotations

import json
import os
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .collect import Collector

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
FILES = {"/": ("index.html", "text/html; charset=utf-8"),
         "/index.html": ("index.html", "text/html; charset=utf-8"),
         "/mark.svg": ("mark.svg", "image/svg+xml")}


def make_handler(col: Collector):
    class Handler(BaseHTTPRequestHandler):
        server_version = "userdash"
        sys_version = ""

        def log_message(self, fmt, *args):  # quiet: no line per request in the container log
            return

        def _send(self, code: int, body: bytes, ctype: str, cache: str = "no-store"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", cache)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, code: int, obj) -> None:
            self._send(code, json.dumps(obj, separators=(",", ":"), allow_nan=False,
                                        default=str).encode("utf-8"), "application/json; charset=utf-8")

        def do_HEAD(self):
            self.do_GET()

        def do_GET(self):
            u = urllib.parse.urlsplit(self.path)
            path = u.path
            try:
                if path == "/healthz":
                    code, body = col.health()
                    return self._json(code, body)
                if path == "/api/now":
                    col.touch()
                    return self._json(200, _clean(col.now_view()))
                if path == "/api/history":
                    col.touch()
                    q = urllib.parse.parse_qs(u.query)
                    try:
                        span = float((q.get("s") or ["3600"])[0])
                    except ValueError:
                        span = 3600.0
                    return self._json(200, _clean(col.history(span)))
                if path in FILES:
                    name, ctype = FILES[path]
                    with open(os.path.join(STATIC, name), "rb") as fh:
                        return self._send(200, fh.read(), ctype, "max-age=60")
                return self._json(404, {"error": "unbekannter Pfad"})
            except (BrokenPipeError, ConnectionResetError):
                return None
            except Exception as e:  # noqa: BLE001
                try:
                    return self._json(500, {"error": type(e).__name__})
                except Exception:  # noqa: BLE001
                    return None

    return Handler


def _clean(obj):
    """NaN/Inf are not JSON: they become null."""
    if isinstance(obj, float):
        return obj if obj == obj and obj not in (float("inf"), float("-inf")) else None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return obj


def serve(col: Collector, bind: str, port: int) -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer((bind, port), make_handler(col))
    httpd.daemon_threads = True
    return httpd
