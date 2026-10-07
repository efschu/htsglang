"""Kartenplaner (Item 510): Vorschau des Reiters OHNE den Dashboard-Dienst.

    python3 -m rigdash.kartenplan_preview --port 18890 [--tree <baum>/python]

Startet einen eigenen kleinen Server auf einem freien Port (nur 127.0.0.1) mit der echten index.html des Dashboards
(Edition rig), kartenplan.js und den /api/kartenplan/*-Antworten.  Alles andere (/api/live ...) antwortet leer; der
Reiter "Kartenplaner" ist über http://127.0.0.1:<port>/#t=kartenplan zu sehen.  Berührt weder den laufenden Dienst
(:8890) noch Container, GPU, Launcher oder Docker.
"""

from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

from . import kartenplan, redact, server

STATIC = server.STATIC


def make_handler(kp: kartenplan.Kartenplaner):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
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

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            try:
                if path in ("/", "/index.html"):
                    with open(os.path.join(STATIC, "index.html"), encoding="utf-8") as fh:
                        return self._send(200, server.edition_page(fh.read(), "rig"), "text/html; charset=utf-8")
                if path in server.STATIC_FILES or path in server.DEV_STATIC_FILES:
                    name, ctype = {**server.STATIC_FILES, **server.DEV_STATIC_FILES}[path]
                    with open(os.path.join(STATIC, name), "rb") as fh:
                        return self._send(200, fh.read(), ctype)
                if path in ("/logo.svg", "/logo-dark.svg", "/mark.svg", "/favicon.svg"):
                    with open(os.path.join(STATIC, "mark.svg" if path == "/favicon.svg" else path[1:]), "rb") as fh:
                        return self._send(200, fh.read(), "image/svg+xml")
                if path == "/api/kartenplan/catalog":
                    return self._json(dict(kp.catalog(), ok=True))
                if path == "/api/kartenplan/plan":
                    raw = (parse_qs(urlsplit(self.path).query).get("q") or [""])[0]
                    return self._json(kp.plan(json.loads(raw)))
                return self._json({"ok": False, "error": "Preview: only the card planner answers"}, 404)
            except ValueError as exc:
                return self._json({"ok": False, "error": str(exc)}, 400)
            except Exception as exc:  # noqa: BLE001
                return self._json({"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}, 500)

    return H


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=18890)
    ap.add_argument("--tree", default=None, help="Planner tree (<tree>/python) with card_identity.py/topology.py")
    ns = ap.parse_args(argv)
    kp = kartenplan.Kartenplaner(tree=ns.tree)
    srv = ThreadingHTTPServer(("127.0.0.1", ns.port), make_handler(kp))
    print("Card planner preview http://127.0.0.1:%d/#t=kartenplan  (planner tree: %s)" % (ns.port, kp.tree), flush=True)
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
