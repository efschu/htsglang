#!/usr/bin/env python3
"""Minimal stand-in local backend for router buffer live validation (task #675).

Not part of the router itself -- this is throwaway validation tooling, a
"simple python http server you start/stop" as required by the task. It
answers every path with 200 and a small JSON body. Router buffering only
cares about the CONNECT-level outcome (refused vs accepted), not response
content, so a generic 200 is sufficient to prove the behavior.

Usage: fake_local_backend.py <port>
"""
import http.server
import json
import sys


class Handler(http.server.BaseHTTPRequestHandler):
    def _reply(self):
        body = json.dumps({"backend": "fake-local", "path": self.path}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._reply()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        if length:
            self.rfile.read(length)
        self._reply()

    def log_message(self, fmt, *args):
        sys.stderr.write("fake-local: " + (fmt % args) + "\n")


if __name__ == "__main__":
    port = int(sys.argv[1])
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.serve_forever()
